import { el } from './dom.js';
import { plural } from './format.js';
import { materialsOf, modelBounds } from './model.js';
import { probeEnvironment, probeState } from './probe.js';

// What the deliverable on screen IS, measured once per load -- the HUD's lines,
// the 'load' event's `specs`, and `viewer.specs` for a script that reports more
// (the packaged `inspect`). One pass, one set of units, so the HUD, the console
// and a script cannot disagree about what "objects" or "lightmapped" means.
//
// The units are the author's, not three.js's, because every three.js count here
// is inflated by the pipeline rather than by the scene:
//
//   * OBJECTS are the file's mesh nodes -- what the DCC exported. three.js makes
//     one Mesh per glTF PRIMITIVE, so a machine body with glass and metal is two
//     "meshes", and the count tracks materials per object, not objects.
//   * MATERIALS are the file's, by glTF index. GLTFLoader clones a material per
//     vertex-colour / tangent variant, and the lightmap pass clones one per baked
//     object (`<name>~lm<N>`), so the count grows with the bake, not the scene.
//   * LIGHTMAPPED is said of objects. The HUD once read "51/57 lightmapped
//     materials": a fraction of those inflated instances, against a denominator
//     that included every material nobody meant to bake, and blind to an object
//     wearing a baked material it was never baked with. How much of the MODEL is
//     lit by its bake is the fact the lighting policy turns on (applyLighting),
//     and it is counted in objects -- the unit the bake is made in, and the one
//     the DCC panel's push report (`PreviewBridge.lightmap_summary`) uses.
//
// Where the file says which objects were BAKED -- each carries its bake's
// `lightmapInfo` marker in its node extras -- intent and outcome are compared,
// and every disagreement is an issue: a baked object with no lightmap bound
// renders unlit, an object with one it was never baked with renders someone
// else's lighting, and both look like a plausible, merely-wrong room.
export let specs = null;
// The first frame after a load compiles its programs and uploads its textures;
// the loop times it into `specs.load.firstFrameMs` (see the render loop).
let firstFramePending = false;

// The JSON chunk's length from a GLB's header (0 for anything else): what the
// file spends describing itself rather than carrying data.
export function glbJsonBytes(buffer) {
  if (buffer.byteLength < 20) return 0;
  const view = new DataView(buffer);
  return view.getUint32(0, true) === 0x46546c67 ? view.getUint32(12, true) : 0;
}

// The object a Mesh belongs to: the glTF node it came from. GLTFLoader makes a
// single-primitive node's Mesh the node itself and groups a multi-primitive
// node's Meshes under a Group standing for it, and records which is which in
// the parser's associations (`nodes`); a file the parser did not annotate falls
// back to the Mesh.
function objectOf(mesh, associations) {
  if (associations?.get(mesh)?.nodes !== undefined) return mesh;
  const parent = mesh.parent;
  if (parent && associations?.get(parent)?.nodes !== undefined) return parent;
  return mesh;
}

// The name the file gives *object* (three.js sanitises its own copy: a Maya
// namespace's ':' is dropped), for the lists an issue names.
function nameOf(object) {
  return object.userData?.name || object.name || '(unnamed)';
}

// Whether the file marks *object* as baked: the bake's per-object marker, in
// either shape it rides (FBX2glTF nests user properties under `fromFBX`; a
// native glTF export writes them at the top of the node's extras), read the way
// MeshConvert._reconcile_node_markers reads them. A cleared marker is empty.
function carriesBakeMarker(object) {
  const data = object.userData || {};
  const raw = data.fromFBX?.userProperties?.lightmapInfo ?? data.lightmapInfo;
  const value = raw && typeof raw === 'object' && 'value' in raw ? raw.value : raw;
  return typeof value === 'string' ? value.trim().length > 0 : Boolean(value);
}

// Measure *gltf* as loaded. `load` carries what only the load path knows: the
// file's bytes, its phases' timings, the clip count and the animated materials.
// The probe's facts are probe.js' (probeState), read after it went up.
export function measureModel(gltf, load, issues) {
  const associations = gltf.parser?.associations;
  const objects = new Map();  // object -> {name, baked, lit}
  const materials = new Set();
  const instances = new Set();
  const lightmaps = new Set();
  const noBakeUv = new Set();
  let bakeChannel = 1;
  let meshes = 0;
  let triangles = 0;
  let vertices = 0;
  gltf.scene.traverse((node) => {
    if (!node.isMesh || !node.geometry) return;
    meshes += 1;
    const { geometry } = node;
    const copies = node.isInstancedMesh ? node.count : 1;
    const position = geometry.attributes.position;
    const drawn = geometry.index ? geometry.index.count : position ? position.count : 0;
    triangles += Math.floor(drawn / 3) * copies;
    vertices += (position ? position.count : 0) * copies;
    const owner = objectOf(node, associations);
    let entry = objects.get(owner);
    if (!entry) {
      entry = { name: nameOf(owner), baked: carriesBakeMarker(owner), lit: false };
      objects.set(owner, entry);
    }
    // The primitive too: a marker on mesh DATA (a native glTF export's mesh
    // extras) lands on each primitive's Mesh rather than on the node's Group.
    if (node !== owner && carriesBakeMarker(node)) entry.baked = true;
    for (const material of materialsOf(node)) {
      instances.add(material);
      materials.add(associations?.get(material)?.materials ?? material);
      if (!material.lightMap) continue;
      entry.lit = true;
      lightmaps.add(material.lightMap.source ?? material.lightMap);
      // three.js samples a lightMap on the UV set its channel names -- `uv`
      // for 0, `uvN` past it -- and applyLightmaps sets that channel from the
      // manifest (`uv`, the second set unless it says otherwise). Without that
      // attribute every fragment reads one texel -- flat, plausible, wrong.
      bakeChannel = material.lightMap.channel;
      if (!geometry.attributes[bakeChannel ? `uv${bakeChannel}` : 'uv']) noBakeUv.add(entry.name);
    }
  });

  const all = [...objects.values()];
  const lit = all.filter((object) => object.lit);
  const baked = all.filter((object) => object.baked);
  const coverage = lit.length || baked.length
    ? {
      lit: lit.length,
      objects: all.length,
      // null where the file marks nothing: intent unknown, so unclaimed.
      baked: baked.length || null,
      unlit: baked.filter((object) => !object.lit).map((object) => object.name),
      foreign: baked.length ? lit.filter((object) => !object.baked).map((object) => object.name) : [],
      maps: lightmaps.size,
      // Whether the room the bake lit is on screen as the environment
      // (probe.js): the light on every unbaked object and every reflection.
      probe: Boolean(probeEnvironment()),
    }
    : null;
  if (coverage?.unlit.length) {
    issues.push({
      level: 'warning',
      text: `${plural(coverage.unlit.length, 'baked object')} ${coverage.unlit.length === 1 ? 'renders' : 'render'} unlit — no lightmap bound`,
      names: coverage.unlit,
    });
  }
  if (coverage?.foreign.length) {
    issues.push({
      level: 'warning',
      text: `${plural(coverage.foreign.length, 'object')} ${coverage.foreign.length === 1 ? 'wears a lightmap it was' : 'wear a lightmap they were'} never baked with`,
      names: coverage.foreign,
    });
  }
  if (noBakeUv.size) {
    const set = bakeChannel === 1 ? 'second UV set' : `UV set ${bakeChannel} (TEXCOORD_${bakeChannel})`;
    issues.push({
      level: 'warning',
      text: `${plural(noBakeUv.size, 'lightmapped object')} ${noBakeUv.size === 1 ? 'has' : 'have'} no ${set}, so the bake reads one texel`,
      names: [...noBakeUv],
    });
  }

  const { size } = modelBounds;
  const probe = probeState();
  return {
    file: { bytes: load.bytes, jsonBytes: load.jsonBytes },
    load: {
      fetchMs: load.fetchMs,
      parseMs: load.parseMs,
      setupMs: load.setupMs,
      // The part of setup the reflection probe took -- its decode and its
      // prefilter -- or null without one.
      probeMs: probe ? probe.decodeMs + probe.prefilterMs : null,
      firstFrameMs: null,
    },
    objects: objects.size,
    meshes,
    triangles,
    vertices,
    size: { x: size.x, y: size.y, z: size.z },
    materials: { file: materials.size, instances: instances.size },
    lightmaps: coverage,
    animation: { clips: load.clips, materials: load.animated.materials, tracks: load.animated.tracks },
    // The bake's reflection probe, or null: `{on, width, height, bytes,
    // bufferView, position, box, decodeMs, prefilterMs}` (see probeState).
    // `on` follows the Environment window's switch, as `lightmaps.probe`
    // does.
    probe,
    issues,
  };
}

// Size in metres: glTF's unit, and the one a headset presents 1:1 -- which is
// the whole reason the page shows the model at its true size.
function formatSize({ x, y, z }) {
  const largest = Math.max(x, y, z);
  const places = largest < 10 ? 2 : largest < 100 ? 1 : 0;
  return `${[x, y, z].map((n) => n.toFixed(places)).join(' × ')} m`;
}

// The HUD's lines, short enough for its 46ch: what the model is, then how it is
// lit and animated. Objects, not meshes; lightmapped as a count of objects, not
// a fraction of material instances (see the section comment above).
function specLines(measured) {
  const lines = [
    [
      plural(measured.objects, 'object'),
      plural(measured.triangles, 'tri'),
      formatSize(measured.size),
    ].join(' · '),
  ];
  const detail = [];
  const coverage = measured.lightmaps;
  if (coverage?.lit) {
    // "n of m", never "n/m": a bare fraction reads as a score against a total
    // it does not name, which is how "51/57 lightmapped materials" misled.
    if (coverage.lit < coverage.objects) {
      detail.push(`${coverage.lit} of ${plural(coverage.objects, 'object')} lightmapped`);
    } else {
      detail.push(coverage.objects === 1 ? 'lightmapped' : `all ${coverage.objects} objects lightmapped`);
    }
  }
  // Whether the bake's reflection probe lights the model -- said when it is
  // switched off as well, since the render then looks unlike the deliverable.
  if (measured.probe) detail.push(measured.probe.on ? 'probe' : 'probe off');
  const { clips, materials } = measured.animation;
  if (clips) detail.push(plural(clips, 'clip'));
  // Invisible from the render when it fails: a fade that is not being driven
  // looks like an object that simply pops, which is how the fades went
  // unnoticed for as long as they did.
  if (materials) detail.push(plural(materials, 'animated material'));
  if (detail.length) lines.push(detail.join(' · '));
  return lines;
}

function issueText(issue, cap = 3) {
  const names = issue.names || [];
  if (!names.length) return issue.text;
  const listed = names.slice(0, cap).join(', ');
  return `${issue.text}: ${listed}${names.length > cap ? `, +${names.length - cap} more` : ''}`;
}

// The HUD's spec lines for *measured* -- again after a switch changes what
// they say (the Environment window's probe), without logging the issues anew.
export function paintStats(measured) {
  el.stats.replaceChildren(...specLines(measured).map((text) => {
    const line = document.createElement('div');
    line.textContent = text;
    return line;
  }));
}

export function showSpecs(measured) {
  paintStats(measured);
  const { issues } = measured;
  el.issues.hidden = !issues.length;
  el.issues.textContent = issues.length
    ? `⚠ ${issueText(issues[0])}${issues.length > 1 ? ` (+${issues.length - 1} more in the console)` : ''}`
    : '';
  // Every one, whole, where a bug report is copied from; the HUD holds one line.
  for (const issue of issues) {
    (issue.level === 'error' ? console.error : console.warn)(`preview: ${issueText(issue, Infinity)}`);
  }
}

// A load's measurement becomes the page's (`viewer.specs`), its first frame
// still to be timed.
export function setSpecs(measured) {
  specs = measured;
  firstFramePending = true;
}

// The render loop's word on the first frame after a load (see above): its
// time inside `renderer.render`, kept once.
export function noteFirstFrame(ms) {
  if (!firstFramePending || !specs) return;
  firstFramePending = false;
  specs.load.firstFrameMs = ms;
}
