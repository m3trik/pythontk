/*
  Inspect — what this deliverable costs, measured on the device rendering it
  rather than inferred from the file.

  A GLB's size says little about whether it previews well in a headset. Draw
  calls follow mesh PRIMITIVES (a room measured 1219 of them against 57
  materials), GPU memory follows DECODED texture dimensions rather than the
  compressed bytes on the wire (5.97 MB of images decoding to ~555 MB of RGBA,
  ~740 MB with mipmaps), and whether a frame is fast enough is the display's to
  judge. So all of it is read here, off the renderer that pays it:

    Frame    frame time (average, 95th percentile, worst) against the display's
             budget -- the headset's own rate in a session -- and the frames
             that missed it by half again; the page's CPU time, the part of it
             spent submitting the draw, and the GPU's time where the browser
             offers a timer query; draw calls and triangles per frame.
    Model    the page's own specs: objects, draws, triangles, materials,
             lightmap coverage, animation.
    Memory   estimated GPU memory: textures at their real size (block-
             compressed KTX2 as transcoded, mip chains, an image shared by
             several materials once) and geometry.
    File     what the GLB spends its bytes on: images, geometry, animation.
    Load     download, parse, setup, and the first frame's compile and upload.
    Issues   what the load found wrong with the deliverable.

  The panel toggles with the Inspect button or `i`. In a headset -- where the
  page's own chrome is not drawn at all -- it rides beside the view as a card,
  toggled with B or Y. Copy Report puts all of it on the clipboard as JSON for a
  bug report, and every load logs a summary table to the console.

  Activate:  bridge.push(scripts=["inspect"])
*/

const WINDOW_MS = 2000;   // the frame statistics span this much recent time
const PAINT_MS = 250;     // the panel and card repaint at most this often
// A frame that took this much longer than the budget was a missed frame: the
// display showed the previous one again (a headset reprojects it).
const MISSED = 1.5;
// xr-standard button 5: B on the right controller, Y on the left.
const TOGGLE_BUTTON = 5;
// Display rates a budget is snapped to when the session does not state its own.
const RATES = [60, 72, 75, 80, 90, 100, 120, 144, 165, 240];

export default function inspect(viewer) {
  const { THREE, renderer } = viewer;
  const panel = viewer.addPanel('Inspect');
  const frames = frameWindow(WINDOW_MS);
  const gpu = gpuTimer(renderer);
  const card = headsetCard(viewer);
  let file = null;     // this load's byte breakdown, from its JSON chunk
  let memory = null;   // this load's GPU memory estimate
  let open = false;
  let painted = 0;
  const held = {};     // handedness -> the toggle button's last state

  function stats() {
    const session = renderer.xr.isPresenting ? renderer.xr.getSession() : null;
    return frames.stats(typeof session?.frameRate === 'number' ? session.frameRate : null);
  }

  function paint() {
    painted = performance.now();
    const frame = stats();
    panel.setRows(panelRows(viewer, frame, memory, file, gpu));
    if (renderer.xr.isPresenting) card.draw(cardLines(viewer, frame, memory));
  }

  function toggle() {
    open = !open;
    panel.show(open);
    if (open) paint();
  }

  const button = viewer.addButton('Inspect', toggle);
  button.title = 'Frame time, draw calls, GPU memory and load cost, measured here (i)';
  const copy = panel.addButton('Copy report', () => {
    const text = JSON.stringify(report(), null, 2);
    const said = (label) => {
      copy.textContent = label;
      setTimeout(() => { copy.textContent = 'Copy report'; }, 1500);
    };
    const logged = () => {
      console.log(text);
      said('Logged to console');
    };
    // The clipboard exists only in a secure context -- and the page is as
    // validly opened over a plain-HTTP LAN address, where it is undefined and
    // calling it threw -- and can still refuse the write. The console always
    // takes the report.
    if (!navigator.clipboard) {
      logged();
      return;
    }
    navigator.clipboard.writeText(text).then(() => said('Copied'), logged);
  });

  function report() {
    const canvas = renderer.domElement;
    return {
      at: new Date().toISOString(),
      page: {
        three: THREE.REVISION,
        userAgent: navigator.userAgent,
        presenting: renderer.xr.isPresenting,
        pixelRatio: renderer.getPixelRatio(),
        canvas: [canvas.width, canvas.height],
      },
      specs: viewer.specs,
      frame: stats(),
      gpuObjects: gpuObjects(renderer),
      memory,
      file,
    };
  }

  viewer.on('key', (event) => {
    if (event.key === 'i') toggle();
  });

  // Once per load, and logged whether or not the panel is open, so a push that
  // quietly doubles the cost says so without anyone having to think to ask.
  viewer.on('load', (detail) => {
    file = fileBreakdown(detail.gltf, detail.specs);
    memory = gpuMemory(viewer, detail.model);
    frames.clear();
    console.table(summary(detail.specs, memory, file));
    if (open) paint();
  });

  viewer.on('frame', () => {
    // Only while someone is looking: a query a frame is cheap, not free, and
    // the script may stay active a whole session with the panel shut.
    if (open) gpu?.begin();
    // The headset's own toggle: its controllers are the only input there.
    const session = renderer.xr.isPresenting ? renderer.xr.getSession() : null;
    for (const source of session?.inputSources || []) {
      const pressed = Boolean(source.gamepad?.buttons?.[TOGGLE_BUTTON]?.pressed);
      if (pressed && !held[source.handedness]) toggle();
      held[source.handedness] = pressed;
    }
  });

  viewer.on('rendered', (detail) => {
    gpu?.end();
    // The GPU reading lags a frame or two (it is read back, never waited on),
    // and is only taken while open -- a closed panel's last one is stale.
    frames.push(detail, renderer.info.render, open ? gpu?.ms ?? null : null);
    card.follow(open, detail.delta);
    if (open && performance.now() - painted > PAINT_MS) paint();
  });
}

/* ------------------------------------------------------------- measuring --- */

// The frames of the last `span` ms: their times, and what each drew.
function frameWindow(span) {
  let samples = [];
  const mean = (list) => (list.length ? list.reduce((a, b) => a + b, 0) / list.length : 0);
  return {
    push({ delta, cpuMs, renderMs }, info, gpuMs) {
      const at = performance.now();
      samples.push({ at, ms: delta * 1000, cpuMs, renderMs, gpuMs, calls: info.calls, triangles: info.triangles });
      while (samples.length && at - samples[0].at > span) samples.shift();
    },
    clear() {
      samples = [];
    },
    /**
     * The window's statistics against a budget of *rate* Hz -- the session's
     * own when it states one; else the display's, estimated from the window's
     * FASTEST frames (a page that misses its budget still lands some frames on
     * the display's interval, where a median would call a slow display fast).
     */
    stats(rate) {
      if (samples.length < 3) return null;
      const sorted = samples.map((sample) => sample.ms).sort((a, b) => a - b);
      const at = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
      const fastest = at(0.1);
      const hz = rate || RATES.reduce((best, r) => (
        Math.abs(1000 / r - fastest) < Math.abs(1000 / best - fastest) ? r : best
      ));
      const budget = 1000 / hz;
      const timed = samples.filter((sample) => typeof sample.gpuMs === 'number');
      const last = samples[samples.length - 1];
      return {
        frames: samples.length,
        fps: 1000 / mean(sorted),
        frameMs: { mean: mean(sorted), p95: at(0.95), worst: sorted[sorted.length - 1] },
        budget: {
          hz,
          ms: budget,
          stated: Boolean(rate),
          missed: samples.filter((sample) => sample.ms > budget * MISSED).length,
        },
        cpuMs: mean(samples.map((sample) => sample.cpuMs)),
        renderMs: mean(samples.map((sample) => sample.renderMs)),
        gpuMs: timed.length ? mean(timed.map((sample) => sample.gpuMs)) : null,
        calls: last.calls,
        triangles: last.triangles,
      };
    },
  };
}

// GPU time of each frame's draw, from a timer query bracketing it ('frame' to
// 'rendered'). Read back frames later, when the GPU has caught up, so it never
// stalls the frame it measures. Null where the browser offers no such query
// (many mobile and headset browsers); a context that loses its queries turns
// it off rather than throwing every frame.
function gpuTimer(renderer) {
  const gl = renderer.getContext();
  const ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');
  if (!ext) return null;
  const pending = [];
  let active = null;
  let latest = null;
  let broken = false;
  return {
    get ms() { return latest; },
    begin() {
      // Bounded: a GPU that never answers must not collect queries forever.
      if (broken || active || pending.length > 8) return;
      try {
        active = gl.createQuery();
        gl.beginQuery(ext.TIME_ELAPSED_EXT, active);
      } catch {
        broken = true;
        active = null;
      }
    },
    end() {
      if (broken) return;
      try {
        if (active) {
          gl.endQuery(ext.TIME_ELAPSED_EXT);
          pending.push(active);
          active = null;
        }
        while (pending.length && gl.getQueryParameter(pending[0], gl.QUERY_RESULT_AVAILABLE)) {
          const query = pending.shift();
          // A disjoint interval (a clock change, a power state) is not a time.
          if (!gl.getParameter(ext.GPU_DISJOINT_EXT)) {
            latest = gl.getQueryParameter(query, gl.QUERY_RESULT) / 1e6;
          }
          gl.deleteQuery(query);
        }
      } catch {
        broken = true;
      }
    },
  };
}

function gpuObjects(renderer) {
  const { memory } = renderer.info;
  return {
    textures: memory.textures,
    geometries: memory.geometries,
    programs: renderer.info.programs?.length ?? 0,
  };
}

// A texture's size in GPU memory. A block-compressed one (KTX2, transcoded to
// ASTC / BC7 / ETC for this device) is exactly its levels' bytes; anything else
// is uploaded uncompressed, texel by texel, plus a third for the mip chain when
// one is generated.
function textureBytes(texture, THREE) {
  if (texture.isCompressedTexture) {
    return (texture.mipmaps || []).reduce((sum, level) => sum + (level.data?.byteLength || 0), 0);
  }
  const { width = 0, height = 0, depth = 1 } = texture.image || {};
  const channels = new Map([[THREE.RedFormat, 1], [THREE.RGFormat, 2], [THREE.AlphaFormat, 1]])
    .get(texture.format) ?? 4;
  const component = new Map([
    [THREE.HalfFloatType, 2], [THREE.FloatType, 4], [THREE.UnsignedShortType, 2],
    [THREE.ShortType, 2], [THREE.IntType, 4], [THREE.UnsignedIntType, 4],
  ]).get(texture.type) ?? 1;
  const mipmapped = texture.generateMipmaps
    && texture.minFilter !== THREE.NearestFilter
    && texture.minFilter !== THREE.LinearFilter;
  return width * height * depth * channels * component * (mipmapped ? 4 / 3 : 1);
}

// What the model holds in GPU memory, estimated from what it uploads. Textures
// are counted by IMAGE (`texture.source`): the lightmap pass gives each baked
// object its own texture carrying its atlas rect, and every one of them shares
// one uploaded image -- counted per texture, one atlas read as forty-six.
function gpuMemory(viewer, model) {
  const { THREE, materialsOf } = viewer;
  const images = new Map();
  const arrays = new Set();
  let geometry = 0;
  const upload = (attribute) => {
    const array = attribute?.isInterleavedBufferAttribute ? attribute.data.array : attribute?.array;
    if (!array || arrays.has(array)) return;
    arrays.add(array);
    geometry += array.byteLength;
  };
  model?.traverse((node) => {
    const shape = node.geometry;
    if (shape) {
      Object.values(shape.attributes).forEach(upload);
      upload(shape.index);
      Object.values(shape.morphAttributes || {}).forEach((list) => list.forEach(upload));
    }
    if (node.isInstancedMesh) {
      upload(node.instanceMatrix);
      upload(node.instanceColor);
    }
    for (const material of materialsOf(node)) {
      for (const value of Object.values(material)) {
        if (!value?.isTexture || !value.image) continue;
        const image = value.source ?? value;
        if (!images.has(image)) {
          images.set(image, { bytes: textureBytes(value, THREE), compressed: Boolean(value.isCompressedTexture) });
        }
      }
    }
  });
  const all = [...images.values()];
  return {
    textures: {
      images: all.length,
      compressed: all.filter((image) => image.compressed).length,
      bytes: all.reduce((sum, image) => sum + image.bytes, 0),
    },
    geometry: { bytes: geometry },
  };
}

// What the GLB's bytes are: images, geometry and animation, by the buffer views
// each references, and its JSON chunk. The rest -- padding, the header, views
// nothing names -- is `other`. Measured on the production assembly this is the
// number that said 73 of its 101 MB were animation accessors.
function fileBreakdown(gltf, specs) {
  const json = gltf?.parser?.json;
  if (!json || !specs?.file?.bytes) return null;
  const views = json.bufferViews || [];
  const accessors = json.accessors || [];
  const viewOf = (index) => accessors[index]?.bufferView;
  const bytesOf = (indices) => [...new Set(indices.filter((index) => index !== undefined && index !== null))]
    .reduce((sum, index) => sum + (views[index]?.byteLength || 0), 0);
  const images = bytesOf((json.images || []).map((image) => image.bufferView));
  const animation = bytesOf((json.animations || []).flatMap((clip) => (
    (clip.samplers || []).flatMap((sampler) => [viewOf(sampler.input), viewOf(sampler.output)])
  )));
  const geometry = bytesOf([
    ...(json.meshes || []).flatMap((mesh) => (mesh.primitives || []).flatMap((primitive) => [
      ...Object.values(primitive.attributes || {}),
      primitive.indices,
      ...(primitive.targets || []).flatMap((target) => Object.values(target)),
    ])).map(viewOf),
    ...(json.skins || []).map((skin) => viewOf(skin.inverseBindMatrices)),
  ]);
  const total = specs.file.bytes;
  const described = specs.file.jsonBytes || 0;
  return {
    bytes: total,
    images,
    geometry,
    animation,
    json: described,
    other: Math.max(0, total - images - geometry - animation - described),
  };
}

/* --------------------------------------------------------------- showing --- */

const count = (n) => Math.round(n).toLocaleString();
const plural = (n, word, many = `${word}s`) => `${count(n)} ${Math.round(n) === 1 ? word : many}`;
const milliseconds = (ms) => (typeof ms === 'number' ? `${ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms` : '—');
const share = (part, whole) => (whole ? ` (${Math.round((100 * part) / whole)}%)` : '');

function lightmapLine(lightmaps) {
  if (!lightmaps) return 'none';
  const baked = lightmaps.baked === null ? '' : ` · ${count(lightmaps.baked)} marked baked`;
  return `${count(lightmaps.lit)} of ${plural(lightmaps.objects, 'object')} lit${baked} · ${plural(lightmaps.maps, 'map')}`;
}

function issueLine(issue) {
  const names = issue.names || [];
  return names.length ? `${issue.text}: ${names.join(', ')}` : issue.text;
}

function panelRows(viewer, frame, memory, file, gpu) {
  // The page's own spelling of a size, so the panel and the status line agree.
  const { formatBytes } = viewer;
  const rows = [{ heading: 'Frame' }];
  if (frame) {
    const { budget } = frame;
    rows.push(
      ['rate', `${Math.round(frame.fps)} fps · ${milliseconds(frame.frameMs.mean)} avg`],
      ['frame time', `p95 ${milliseconds(frame.frameMs.p95)} · worst ${milliseconds(frame.frameMs.worst)}`],
      [
        'budget',
        `${milliseconds(budget.ms)} at ${budget.hz} Hz${budget.stated ? '' : ' (est.)'} · ${budget.missed} of ${frame.frames} missed`,
        budget.missed ? 'warn' : '',
      ],
      ['CPU', `${milliseconds(frame.cpuMs)} · draw submit ${milliseconds(frame.renderMs)}`],
      ['GPU', frame.gpuMs !== null ? milliseconds(frame.gpuMs) : gpu ? 'measuring…' : 'no timer query in this browser'],
      ['drawn', `${count(frame.calls)} calls · ${count(frame.triangles)} tris`],
    );
  } else {
    rows.push({ text: 'waiting for frames…' });
  }
  const objects = gpuObjects(viewer.renderer);
  rows.push(['in GPU', `${objects.textures} textures · ${objects.geometries} geometries · ${objects.programs} programs`]);

  const specs = viewer.specs;
  if (specs) {
    rows.push(
      { heading: 'Model' },
      ['objects', `${count(specs.objects)} · ${plural(specs.meshes, 'mesh', 'meshes')} (one draw each)`],
      ['triangles', `${count(specs.triangles)} · ${count(specs.vertices)} vertices`],
      ['materials', `${count(specs.materials.file)} in the file · ${count(specs.materials.instances)} in three.js`],
      ['lightmaps', lightmapLine(specs.lightmaps)],
    );
    if (specs.animation.clips) {
      rows.push(['animation', `${plural(specs.animation.clips, 'clip')} · ${plural(specs.animation.materials, 'animated material')}`]);
    }
  }
  if (memory) {
    const { textures } = memory;
    rows.push(
      { heading: 'GPU memory (estimated)' },
      [
        'textures',
        `${formatBytes(textures.bytes)} · ${plural(textures.images, 'image')}${textures.compressed ? `, ${count(textures.compressed)} block-compressed` : ''}`,
      ],
      ['geometry', formatBytes(memory.geometry.bytes)],
    );
  }
  if (file) {
    rows.push(
      { heading: 'File' },
      ['size', formatBytes(file.bytes)],
      ['images', `${formatBytes(file.images)}${share(file.images, file.bytes)}`],
      ['geometry', `${formatBytes(file.geometry)}${share(file.geometry, file.bytes)}`],
      ['animation', `${formatBytes(file.animation)}${share(file.animation, file.bytes)}`],
      ['JSON', `${formatBytes(file.json)}${share(file.json, file.bytes)}`],
    );
  }
  if (specs) {
    rows.push(
      { heading: 'Load' },
      ['download', milliseconds(specs.load.fetchMs)],
      ['parse', milliseconds(specs.load.parseMs)],
      ['setup', milliseconds(specs.load.setupMs)],
      ['first frame', milliseconds(specs.load.firstFrameMs)],
    );
    if (specs.issues.length) {
      rows.push({ heading: 'Issues' });
      for (const issue of specs.issues) rows.push({ text: issueLine(issue), level: 'warn' });
    }
  }
  return rows;
}

// The headset card's lines: the panel's, cut to what reads at arm's length.
function cardLines(viewer, frame, memory) {
  const { specs, formatBytes } = viewer;
  const lines = [];
  if (frame) {
    const { budget } = frame;
    lines.push({ text: `${Math.round(frame.fps)} fps · ${milliseconds(frame.frameMs.mean)} (p95 ${milliseconds(frame.frameMs.p95)})` });
    lines.push({
      text: `budget ${milliseconds(budget.ms)} at ${budget.hz} Hz · ${budget.missed} missed`,
      level: budget.missed ? 'warn' : '',
    });
    lines.push({ text: `CPU ${milliseconds(frame.cpuMs)} · GPU ${milliseconds(frame.gpuMs ?? undefined)}` });
    lines.push({ text: `${count(frame.calls)} calls · ${count(frame.triangles)} tris` });
  }
  if (memory) {
    lines.push({ text: `textures ${formatBytes(memory.textures.bytes)} · geometry ${formatBytes(memory.geometry.bytes)}` });
  }
  if (specs) {
    const lit = specs.lightmaps ? ` · ${specs.lightmaps.lit} of ${specs.lightmaps.objects} lightmapped` : '';
    lines.push({ text: `${plural(specs.objects, 'object')}${lit}` });
    if (specs.issues.length) lines.push({ text: `⚠ ${specs.issues[0].text}`, level: 'warn' });
  }
  return lines.slice(0, 8);
}

// What every load logs, in the numbers a bug report quotes.
function summary(specs, memory, file) {
  return {
    objects: specs?.objects,
    meshes: specs?.meshes,
    triangles: specs?.triangles,
    materials: specs?.materials.file,
    lightmapped: specs?.lightmaps ? `${specs.lightmaps.lit} of ${specs.lightmaps.objects} objects` : 'none',
    textureMB: memory ? Math.round(memory.textures.bytes / (1024 * 1024)) : null,
    geometryMB: memory ? Math.round(memory.geometry.bytes / (1024 * 1024)) : null,
    fileMB: file ? Number((file.bytes / (1024 * 1024)).toFixed(1)) : null,
    issues: specs?.issues.length ?? 0,
  };
}

// The panel as a card in the headset: the page's DOM is not drawn in an
// immersive session, so a profile taken where it matters -- in the headset,
// at its own frame rate -- would otherwise have nowhere to show. The page's own
// headset card (`viewer.addHeadsetCard`), over the model: it rides a little
// left of and below where the head faces, eased after it rather than fixed to
// it, and is shown only while the panel is open in a session.
function headsetCard(viewer) {
  const { THREE, renderer } = viewer;
  const card = viewer.addHeadsetCard({ name: 'inspect', width: 720, height: 400, metres: 0.36, over: true });
  const head = new THREE.Vector3();
  const look = new THREE.Vector3();
  const turn = new THREE.Quaternion();

  return {
    draw(lines) {
      card.draw((g, width) => {
        g.font = '600 30px system-ui, sans-serif';
        g.fillStyle = '#e8eaed';
        g.fillText('Inspect', 28, 48);
        g.font = '25px system-ui, sans-serif';
        lines.forEach((line, row) => {
          g.fillStyle = line.level === 'warn' ? '#f2b84b' : '#c8ccd1';
          g.fillText(line.text, 28, 92 + row * 38, width - 56);
        });
      });
    },
    // Called after each frame is drawn, when the headset's pose for it is known.
    follow(open, delta) {
      if (!open || !renderer.xr.isPresenting) {
        card.mesh.visible = false;
        return;
      }
      const eye = renderer.xr.getCamera();
      eye.getWorldPosition(head);
      look.set(0, 0, -1).applyQuaternion(eye.getWorldQuaternion(turn));
      // ~23 degrees left of the view, a little below it, at a reading distance.
      card.place(head, look, { ahead: 0.75, aside: 0.4, below: 0.18, ease: 0.4 }, delta);
      card.mesh.visible = true;
    },
  };
}
