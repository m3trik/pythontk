/*
  Export Playblast — record the clip the transport is on to a movie file.

  Whatever the picker is showing: one declared shot, the whole-timeline clip, or
  FULL SEQUENCE (every shot laid back onto the timeline it was cut from), which
  records as one continuous movie exactly as it plays.

  This is a PLAYBLAST, not a screen recording. The clip is stepped: the page
  poses frame N, renders it, and hands the pixels over before asking for N+1.
  A throttled tab, a slow headset and a desktop therefore all produce the same
  file, at the deliverable's own authoring frame rate — which is the property
  that makes the result comparable with a viewport playblast of the same shot
  rather than merely similar to it. Sampling the display instead would drop and
  duplicate frames wherever the device was busy, and the busiest moment is
  always the one worth reviewing.

  The frames are POSTed to the preview server as they are produced, one per
  request, and encoded there by the same `pythontk.SequenceEncoder` that turns
  Maya's viewport captures into an mp4. Streaming rather than batching is not an
  optimisation: a canvas readback is a Blob, and holding a clip's worth of them
  in page memory is how a headset tab dies.

  Compressing a frame to PNG is the whole cost of a recording — measured, every
  other stage put together is under 5% of it — so the compression happens on
  WORKER THREADS while the main thread gets on with rendering the next frame.
  See `createEncoder`.

  Pressing the button opens an options prompt rather than recording at once:
  the burn-in is drawn into the file and cannot be taken out of it again, so it
  is asked at the moment the file is written rather than armed by a toggle
  sitting beside the button, where it is left on and found in the movie later.

  Activated by the deliverable (`PreviewServer.AUTO_SCRIPTS`) whenever it ships
  an animation manifest — the button appears exactly when the clip picker does.
*/

//: What the export prompt offers, lowest to highest. `maxEdge` is the recorded
//: frame's long edge; `quality` is the 0-100 the server maps onto the H.264 CRF
//: (`SequenceEncoder._quality_to_crf`: 100 -> 16, 85 -> 20, 70 -> 23). Before
//: presets every recording was 1280 px at quality 100 -- Draft's size.
//:
//: The frame is RENDERED at the preset's size, never upscaled to it: a view
//: whose drawing buffer is smaller has its pixel ratio raised for the recording
//: and put back afterwards (see `captureSize`), so High from a 1280-wide window
//: is a real 2560-wide render. A larger view is downsampled by the capture,
//: which antialiases for free.
//:
//: Size is no longer the wall-clock dial it was when frames compressed on the
//: main thread (1920 -> 1280 then measured 2.1x faster); with the worker pool
//: below, 1920 and 1280 measured 46.7ms and 45.1ms per frame. What a bigger
//: preset still costs is file size and wire time -- what a reviewer on a
//: headset over Wi-Fi actually waits for, and what Draft is for.
//:
//: `key` is what the tab remembers between prompts; the order is the menu's.
const QUALITY_PRESETS = [
  {
    key: 'draft',
    label: 'Draft — 720p',
    maxEdge: 1280,
    quality: 70,
    title: '1280 px long edge, CRF 23. Smallest file: for sending over a slow link.',
  },
  {
    key: 'standard',
    label: 'Standard — 1080p',
    maxEdge: 1920,
    quality: 85,
    title: '1920 px long edge, CRF 20.',
  },
  {
    key: 'high',
    label: 'High — 1440p',
    maxEdge: 2560,
    quality: 100,
    title: '2560 px long edge, CRF 16.',
  },
  {
    key: 'maximum',
    label: 'Maximum — 4K',
    maxEdge: 3840,
    quality: 100,
    title: '3840 px long edge, CRF 16. Largest file; the GPU\'s own limit can cap it lower.',
  },
];

//: High, not Draft: a playblast is most often looked at on the screen it was
//: made on, where a 720p-class file reads as soft next to the page itself.
const DEFAULT_PRESET = 'high';

function presetFor(key) {
  return QUALITY_PRESETS.find((preset) => preset.key === key)
    || QUALITY_PRESETS.find((preset) => preset.key === DEFAULT_PRESET);
}

//: `{width, height, pixelRatio}` for a recording whose long edge is *maxEdge*:
//: the size every frame is resized to, and the pixel ratio to render at while
//: recording -- null when the view is already at least that large.
//:
//: Clamped to what the GPU will allocate. A canvas asked for a buffer past
//: MAX_RENDERBUFFER_SIZE does not fail; it silently allocates a smaller one,
//: and the capture would then read that stretched across the frame.
function captureSize(renderer, maxEdge) {
  const canvas = renderer.domElement;
  const gl = renderer.getContext();
  const [viewportWidth, viewportHeight] = gl.getParameter(gl.MAX_VIEWPORT_DIMS);
  const limit = Math.min(gl.getParameter(gl.MAX_RENDERBUFFER_SIZE), viewportWidth, viewportHeight);
  const scale = Math.min(maxEdge, limit) / Math.max(canvas.width, canvas.height);
  return {
    width: Math.max(1, Math.round(canvas.width * scale)),
    height: Math.max(1, Math.round(canvas.height * scale)),
    pixelRatio: scale > 1 ? renderer.getPixelRatio() * scale : null,
  };
}

//: Frames are sent as PNG -- lossless, and NOT the slow choice, which is the
//: opposite of how it looks. JPEG frames do compress faster, but a JPEG decodes
//: to FULL-range YUV, so ffmpeg carries that through and tags the movie
//: `yuvj420p (pc)`. Levels round-trip correctly for a decoder that reads the
//: tag; one that ignores it renders the preview with crushed blacks, which is a
//: poor property for a lookdev artifact that gets passed around. Converting
//: back to limited range (`out_range=tv` on the encoder's scale filter) makes
//: the scaler touch every frame and costs about what JPEG saved. A PNG is RGB,
//: which ffmpeg converts to limited-range YUV natively and for free. So the
//: correct choice and the fast one are the same here, and the shared encoder
//: needs no filter for this route (see
//: `SequenceEncoder._EVEN_DIMENSIONS_FILTER`, which records the same).
//:
//: Once the compression moved onto workers this stopped being a close call at
//: all: there is no longer a main-thread cost for a lossy format to undercut.
const FRAME_TYPE = 'image/png';
//: Ignored for PNG (lossless); kept so the pair stays coherent if this ever
//: moves back to a lossy format.
const FRAME_QUALITY = undefined;

//: Frames compressing at once. Half the cores, because the main thread needs
//: one to render on and the machine is usually running a DCC as well, and
//: never more than 8 -- measured, the curve is flat past 4 (2 workers 4.4x,
//: 4 workers 5.2x, 8 workers 5.8x, 12 workers slightly worse than 8). The
//: floor of 2 is what a phone or a standalone headset gets, and 2 is already
//: most of the win.
const POOL_SIZE = Math.max(2, Math.min(8, Math.floor((navigator.hardwareConcurrency || 4) / 2)));

//: One worker: take a frame, compress it, POST it, say which frame it was.
//:
//: Inlined rather than shipped as a file beside this one because it has to be
//: loadable as a worker from the page's own origin, and a Blob URL gets that
//: without adding a served route, a fetch before the first frame, or a second
//: file to keep in step with this one.
//:
//: It POSTs the frame itself rather than handing the blob back: the point of
//: the exercise is to keep work off the main thread, and a blob that returns
//: only to be uploaded would put the upload back there.
const WORKER_SOURCE = `
let target = null;
self.onmessage = async (event) => {
  const job = event.data;
  if (job.target !== undefined) { target = job.target; return; }
  try {
    const canvas = new OffscreenCanvas(job.bitmap.width, job.bitmap.height);
    canvas.getContext('2d').drawImage(job.bitmap, 0, 0);
    job.bitmap.close();
    const blob = await canvas.convertToBlob({ type: job.type, quality: job.quality });
    const response = await fetch(target + '&index=' + job.index, {
      method: 'POST',
      body: blob,
      headers: { 'Content-Type': job.type },
    });
    if (!response.ok) {
      throw new Error(response.statusText || ('HTTP ' + response.status));
    }
    self.postMessage({ index: job.index });
  } catch (error) {
    self.postMessage({ index: job.index, error: String((error && error.message) || error) });
  }
};
`;

//: Burn-in text: white on a dark outline rather than on a plate, because a
//: plate hides the pixels under it and the corner of frame is exactly where a
//: reviewer looks for a clipped silhouette. Monospaced so the frame counter
//: does not jitter the line as digits change width.
const OVERLAY_FONT = 'ui-monospace, "DejaVu Sans Mono", Consolas, monospace';

//: Burn-in height as a fraction of the frame, with a floor: at 720p this is a
//: 23px line, and at a phone-sized capture it stays readable rather than
//: scaling into nothing.
const OVERLAY_SCALE = 0.032;
const OVERLAY_MIN_SIZE = 11;

export default function playblast(viewer) {
  const button = viewer.addButton('Export Playblast', () => {
    // Nothing to cancel once the frames are all in and ffmpeg has them: the
    // encode is the server's, and "cancelling" it here would only detach the
    // page from a file that is written anyway.
    if (job) {
      if (!job.done) abort();
    } else {
      openOptions();
    }
  });
  button.title = 'Record the selected clip to a movie file — asks for export '
    + 'options first';

  let job = null;

  //: What the last prompt was answered with, so a second recording re-offers
  //: the choices rather than the defaults -- a reviewer annotating one shot is
  //: about to annotate the next one the same way. Session-lived on purpose:
  //: nothing here is worth persisting past the tab, and a burn-in silently
  //: remembered from last week is a surprise baked into a deliverable.
  //:
  //: Both burn-ins OFF by default, because a burn-in is drawn into the movie and
  //: cannot be taken out again: unless someone asks for the annotation, the
  //: recording is what the reviewer saw.
  let options = { stamp: false, describe: false, preset: DEFAULT_PRESET };

  //: Asked BEFORE the recording rather than armed by a toggle beside the
  //: button. A burn-in is a property of the file about to be written, so the
  //: moment to decide it is the moment it is written -- and a mode carried on
  //: a button is exactly the kind of state that gets left on and discovered in
  //: the movie afterwards.
  async function openOptions() {
    // Refused before the prompt, never after: these are the reasons a
    // recording cannot happen at all, and asking how to annotate a movie that
    // will not be written is a worse answer than saying why.
    const clip = recordable();
    if (!clip) return;
    const answer = await viewer.showDialog({
      title: 'Export Playblast',
      fields: [
        // What is about to be written, stated rather than offered: the clip is
        // the picker's to choose and the rate is the deliverable's, so the one
        // thing this prompt can usefully do about them is show them.
        {
          key: 'summary',
          type: 'note',
          label: `${clip.name} · ${frameCount(clip)} frames · ${clip.fps} fps`,
        },
        {
          key: 'preset',
          type: 'choice',
          label: 'Quality',
          value: options.preset,
          choices: QUALITY_PRESETS.map(({ key, label, title }) => ({ value: key, label, title })),
          title: 'Resolution and compression of the movie. The shot is rendered '
            + 'at the chosen size, so a small window still records a full-size '
            + 'frame.',
        },
        {
          key: 'stamp',
          label: 'Burn in shot name and frame',
          value: options.stamp,
          title: 'Draw the shot name, the DCC frame number and the time into '
            + 'the movie itself. On FULL SEQUENCE the name follows the '
            + 'playhead across shots, marking a gap as a hold.',
        },
        {
          key: 'describe',
          label: 'Burn in shot description',
          value: options.describe,
          title: "Draw the shot's description -- the note the DCC's Shots "
            + 'panel carries -- above the name. Shots with no description '
            + 'record without one.',
        },
      ],
      confirm: 'Record',
    });
    if (!answer) return;   // cancelled
    options = { stamp: answer.stamp, describe: answer.describe, preset: answer.preset };
    // Re-resolved from scratch by `start`, never carried from the prompt: a
    // push can land while the dialog is open, and recording the clip object
    // that check produced would record a model the page no longer has.
    start();
  }

  // A push landing mid-recording swaps the model under the capture: every frame
  // after it would be of a different scene, and the file would silently be a
  // cut between two versions. The recording is dropped rather than finished.
  viewer.on('load', () => {
    if (job) abort('the model was replaced by a new push');
  });

  // The capture rides the page's own loop because that is the only place a
  // WebGL canvas can be read: without `preserveDrawingBuffer` the drawing
  // buffer is valid until the browser composites, so the render and the
  // snapshot have to happen in the same task. 'frame' is emitted before the
  // page renders, so this poses, renders and snapshots its own frames — the
  // page's render right after it draws the last pose again to no ill effect.
  //
  // As many frames per tick as the pool has room for, not one: with four
  // workers idle, capturing one frame per animation frame would leave three of
  // them waiting on the display's refresh rate for no reason.
  viewer.on('frame', () => {
    if (!job || job.done || !job.encoder) return;
    while (job.issued < job.frames && job.encoder.free > 0) step();
  });

  /* ------------------------------------------------------------- capture --- */

  //: The selected clip if it can be recorded, else null with the status line
  //: already saying why. Two callers: the options prompt, which will not ask
  //: about a movie that cannot be written, and `start`, which re-asks because
  //: the page can change between the two.
  function recordable() {
    // The XR framebuffer is not the canvas: in an immersive session
    // `renderer.render` targets the device, and reading the canvas back would
    // capture whatever was last drawn to the mirror -- a file that looks broken
    // with nothing to say why. Refused rather than recorded wrong.
    if (viewer.renderer.xr?.isPresenting) {
      viewer.setStatus('exit VR to record a playblast', 'error');
      return null;
    }
    const clip = viewer.clip;
    if (!clip) {
      viewer.setStatus('nothing to record — this model has no clips', 'error');
      return null;
    }
    if (!clip.fps) {
      // The manifest is where the authoring rate lives; without it the frames
      // could still be captured but not placed in time, and a movie at a guessed
      // rate is worse than none — it plays at the wrong speed and looks correct.
      viewer.setStatus('cannot record — the deliverable states no frame rate', 'error');
      return null;
    }
    return clip;
  }

  //: Frames INCLUSIVE of both bounds, which is the count `clipInfo` builds its
  //: two frame numbers to satisfy -- the same arithmetic for a single shot and
  //: for the reconstructed sequence.
  function frameCount(clip) {
    return Math.max(1, Math.round(clip.endFrame - clip.startFrame) + 1);
  }

  function start() {
    const clip = recordable();
    if (!clip) return;
    const fps = clip.fps;
    const frames = frameCount(clip);

    // Measured ONCE, here. Every frame of a movie must be the same size, and
    // the canvas is not: resizing the window mid-recording (or rotating a
    // phone) would otherwise change the frame size part way through, which
    // ffmpeg answers with a garbled encode rather than an error.
    const preset = presetFor(options.preset);
    const size = captureSize(viewer.renderer, preset.maxEdge);

    job = {
      clip,
      fps,
      frames,
      issued: 0,
      completed: 0,
      done: false,
      token: null,
      encoder: null,
      width: size.width,
      height: size.height,
      quality: preset.quality,
      // The ratio to put back, set only when this recording raised it.
      restorePixelRatio: null,
      pad: null,
      padContext: null,
      // Snapshotted onto the job rather than read from `options` per frame, so
      // the whole file is annotated the way the prompt that started it was
      // answered -- a recording is one thing, and half of it stamped is not a
      // state this can produce.
      stamp: options.stamp,
      describe: options.describe,
      wasPlaying: viewer.playing,
    };
    if (job.stamp || job.describe) {
      // ONE pad for the whole recording, reused every frame -- and it is worth
      // saying why that is safe, because the capture loop issues several frames
      // per tick and this canvas is redrawn under each of them. `submit`
      // snapshots what it is handed AT THE POINT IT IS CALLED (both
      // `createImageBitmap` and `toBlob` are specified that way), so a frame's
      // pixels are already taken before the next frame draws over them. A ring
      // of pads was built for this and measured unnecessary; what remains of it
      // is the test, which compares ADJACENT frames' burn-ins and fails if they
      // ever come out identical.
      job.pad = document.createElement('canvas');
      job.pad.width = job.width;
      job.pad.height = job.height;
      // Fetched and configured ONCE: `getContext` returns the same object every
      // time, so doing this per frame was re-setting a state that never changed.
      job.padContext = job.pad.getContext('2d');
      job.padContext.imageSmoothingQuality = 'high';
    }
    if (size.pixelRatio) {
      // Raised before the first frame is issued (that waits on `begin`), and
      // not touched again until `releaseResolution`. Only the drawing buffer
      // grows: the canvas keeps its size on the page.
      job.restorePixelRatio = viewer.renderer.getPixelRatio();
      viewer.renderer.setPixelRatio(size.pixelRatio);
    }
    // Paused for the whole recording: the clock is this script's, and a playing
    // transport would advance the clip between the pose and the render.
    viewer.setPlaying(false);
    label(`Starting ${clip.name}…`);

    post('begin', {
      name: clip.name,
      fps,
      start_frame: Math.round(clip.startFrame),
      frames,
      content_type: FRAME_TYPE,
      quality: job.quality,
    })
      .then((reply) => {
        if (!job) {
          // Aborted while the request was in flight. The abort had no token to
          // cancel with, so this is the only place that knows the server is
          // holding a recording nobody will ever send a frame to.
          post('cancel', { token: reply.token }).catch(() => {});
          return;
        }
        job.token = reply.token;
        // Absolute: a worker made from a Blob URL has no useful base to resolve
        // a relative path against.
        const target = new URL(
          `playblast/frame?token=${encodeURIComponent(reply.token)}`,
          document.baseURI,
        ).href;
        job.encoder = createEncoder(target, onFrameDone, fail);
        label(`Recording 0/${frames}`);
      })
      .catch((error) => fail(error));
  }

  function step() {
    const index = job.issued;
    job.issued += 1;
    // The clip's own clock, never the wall clock: frame i is at i/fps, and the
    // last frame is clamped to the duration so a rounding remainder cannot ask
    // for a pose past the end of the clip.
    const seconds = Math.min(index / job.fps, job.clip.duration);
    viewer.poseAt(seconds);
    viewer.renderer.render(viewer.scene, viewer.camera);

    let source = viewer.renderer.domElement;
    if (job.pad) {
      // Annotated HERE rather than inside the encoder: the worker pool and the
      // main-thread fallback would otherwise need a copy of this each, and the
      // pool's copy would have to travel to the workers as source text. The
      // pre-pass costs a readback that measured a fraction of a millisecond --
      // the expensive part of a frame is compressing it, not moving it.
      const context = job.padContext;
      context.drawImage(source, 0, 0, job.width, job.height);
      drawOverlay(
        context,
        job.width,
        job.height,
        leftLines(seconds),
        job.stamp ? stamp(index, seconds) : null,
      );
      source = job.pad;
    }
    // Same task as the render above, which is what makes this legal -- see the
    // note on the 'frame' hook. `submit` must therefore SNAPSHOT here and may
    // not await anything first; letting the page composite in between yields a
    // blank frame, silently, and a movie of nothing.
    job.encoder.submit(source, index, job.width, job.height);
  }

  /* ------------------------------------------------------------- burn-in --- */

  //: What the frame is OF. On the whole-timeline clip that is the shot the
  //: playhead is inside -- the same name, and the same `(hold)` for a gap, that
  //: the transport readout shows, so the movie and the page agree. On a single
  //: shot the sequence has nothing to say and the clip's own name is the answer.
  function shotName(seconds) {
    return viewer.shotAt(seconds) || job.clip.name;
  }

  //: WHERE in the shot. The frame number is the DCC's own -- the clip's start
  //: frame plus the offset -- so a note about "frame 112" means the frame the
  //: animator will open. Zero-padded to keep the line still, except on a
  //: negative frame, where padding the digits would bury the sign.
  //:
  //: The time is the one the frame was POSED at, not `index / fps` recomputed:
  //: the pose is clamped to the clip's duration, so on a clip whose frame count
  //: and duration disagree by a rounding remainder the two part company at the
  //: end -- and the stamp would then label a held last frame with a time past
  //: the clip it is holding on.
  function stamp(index, seconds) {
    const frame = Math.round(job.clip.startFrame) + index;
    const padded = frame < 0 ? String(frame) : String(frame).padStart(4, '0');
    return `${padded}  ${seconds.toFixed(2)}s`;
  }

  //: The left-hand column, BOTTOM-UP: the name sits on the frame's foot beside
  //: the counter, and the description stacks above it. Built per frame because
  //: both answers move with the playhead -- across the sequence each shot
  //: brings its own name and its own note.
  //:
  //: The two options are independent rather than nested, which is what lets the
  //: prompt ask them as two plain checkboxes: with neither there is no pad and
  //: nothing is drawn, and with only the description the note lands alone on
  //: the foot of the frame. A shot with no description simply records without
  //: one -- the deliverable states it or it does not, and an empty line held
  //: open for it would be a burn-in that says nothing.
  function leftLines(seconds) {
    const lines = [];
    if (job.stamp) lines.push(shotName(seconds));
    if (job.describe) {
      // No argument, unlike `shotAt`: a description is a property of the shot
      // the page is POSED at, and `poseAt` above has already put it there.
      const text = viewer.descriptionAt();
      if (text) lines.push(text);
    }
    return lines;
  }

  function drawOverlay(context, width, height, lines, right) {
    if (!lines.length && !right) return;
    const size = Math.max(OVERLAY_MIN_SIZE, Math.round(height * OVERLAY_SCALE));
    const inset = Math.round(size * 0.75);
    const baseline = height - inset;
    // Stacked upward from the foot, so a description pushes the NAME up rather
    // than moving the counter off the edge it is measured against: the two
    // corners have to stay on one line for the right-hand column to read as
    // the left one's counterpart.
    const leading = Math.round(size * 1.35);
    context.save();
    context.font = `600 ${size}px ${OVERLAY_FONT}`;
    context.textBaseline = 'alphabetic';
    context.lineJoin = 'round';
    context.lineWidth = Math.max(2, size / 6);
    context.strokeStyle = 'rgba(0, 0, 0, 0.75)';
    context.fillStyle = 'rgba(255, 255, 255, 0.96)';
    // The counter is fixed-width and never elided; the left column yields to
    // it. A description is prose written by whoever authored the shot, so it
    // is the one string here that can be longer than the frame.
    const reserved = right ? context.measureText(right).width + size : 0;
    const draws = lines.map((text, row) => [
      elide(context, text, width - inset * 2 - (row ? 0 : reserved)),
      'left',
      inset,
      baseline - row * leading,
    ]);
    if (right) draws.push([right, 'right', width - inset, baseline]);
    for (const [text, align, x, y] of draws) {
      context.textAlign = align;
      // Stroke then fill: the outline is what keeps the text legible over a
      // blown-out sky as well as over a dark background.
      context.strokeText(text, x, y);
      context.fillText(text, x, y);
    }
    context.restore();
  }

  //: *text* trimmed to *maxWidth*, with an ellipsis when it had to give. Text
  //: that overran used to run under the counter and off the frame, which reads
  //: as a corrupted burn-in rather than as a long description.
  //:
  //: Proportional estimate first, then a character at a time: the font is
  //: monospaced, so the estimate is usually exact and the loop runs once or
  //: not at all -- rather than the couple of hundred `measureText` calls a
  //: trim from full length would cost on every frame of a recording.
  function elide(context, text, maxWidth) {
    if (maxWidth <= 0) return '';
    const full = context.measureText(text).width;
    if (full <= maxWidth) return text;
    let cut = Math.max(0, Math.min(text.length - 1, Math.floor(text.length * (maxWidth / full)) - 1));
    while (cut > 0 && context.measureText(`${text.slice(0, cut)}…`).width > maxWidth) cut -= 1;
    while (cut < text.length - 1 && context.measureText(`${text.slice(0, cut + 1)}…`).width <= maxWidth) cut += 1;
    return cut > 0 ? `${text.slice(0, cut)}…` : '';
  }

  function onFrameDone() {
    if (!job || job.done) return;
    job.completed += 1;
    label(`Recording ${job.completed}/${job.frames}`);
    if (job.completed >= job.frames) finish();
  }

  async function finish() {
    job.done = true;
    if (job.encoder) job.encoder.close();
    // Now rather than after the encode: every frame is in, and the page should
    // not go on rendering at export size for however long ffmpeg takes.
    releaseResolution();
    // Read before the await: a push landing during the encode resets the job.
    const { width, height } = job;
    // No ✕: see the button handler -- there is nothing left to cancel.
    button.textContent = 'Encoding…';
    try {
      const report = await post('finish', { token: job.token });
      const seconds = Number(report.duration || 0).toFixed(1);
      viewer.setStatus(
        `playblast saved · ${report.frames} frames · ${seconds}s · ${width}×${height}`,
        'live',
      );
      // Offered rather than forced when the file has a durable home: a movie
      // written beside the GLB it was recorded from is already where its owner
      // wants it, and a second copy in Downloads is noise. When the deliverable
      // was a scene push there IS no such file — its GLB is the bridge's own
      // scratch — so the download is the only copy that survives the session,
      // and the one viewer who cannot open a folder is the one in the headset.
      if (report.in_serve_root) download(report.url);
      else console.info(`playblast written to ${report.output}`);
      reset();
    } catch (error) {
      fail(error);
    }
  }

  /* ------------------------------------------------------------- encoder --- */

  //: Compresses frames and POSTs them, off the main thread where possible.
  //:
  //: One interface, two implementations, so the capture loop above never asks
  //: which one it got:
  //:   free    — how many frames it can take right now (its backpressure)
  //:   submit  — snapshot this canvas SYNCHRONOUSLY, then compress and send it
  //:   close   — release everything; safe to call twice
  //:
  //: Frames finish out of order once several are in flight, which is why the
  //: recording counts completions rather than tracking an index, and why the
  //: server stores frames in a dict keyed by frame number.
  function createEncoder(target, onDone, onError) {
    const capable =
      typeof Worker === 'function' &&
      typeof OffscreenCanvas === 'function' &&
      typeof OffscreenCanvas.prototype.convertToBlob === 'function' &&
      typeof createImageBitmap === 'function';
    return capable
      ? createPoolEncoder(target, onDone, onError)
      : createInlineEncoder(target, onDone, onError);
  }

  function createPoolEncoder(target, onDone, onError) {
    const blobUrl = URL.createObjectURL(
      new Blob([WORKER_SOURCE], { type: 'text/javascript' }),
    );
    const idle = [];
    let closed = false;

    const release = (worker) => {
      if (closed) worker.terminate();
      else idle.push(worker);
    };

    for (let i = 0; i < POOL_SIZE; i++) {
      const worker = new Worker(blobUrl);
      worker.postMessage({ target });
      worker.onmessage = (event) => {
        release(worker);
        if (closed) return;
        if (event.data.error) onError(new Error(event.data.error));
        else onDone(event.data.index);
      };
      worker.onerror = (event) => {
        release(worker);
        if (!closed) onError(new Error(event.message || 'frame encoder failed'));
      };
      idle.push(worker);
    }

    return {
      get free() {
        return closed ? 0 : idle.length;
      },
      submit(source, index, width, height) {
        const worker = idle.pop();
        // Issued in the caller's task, before anything is awaited: this is the
        // snapshot, and it is only valid here. It also does the downscale,
        // which keeps the main thread off a 2D canvas entirely -- and measured
        // bit-identical (max channel delta 0) to the drawImage it replaces.
        createImageBitmap(source, 0, 0, source.width, source.height, {
          resizeWidth: width,
          resizeHeight: height,
          resizeQuality: 'high',
        })
          .then((bitmap) => {
            if (closed) {
              bitmap.close();
              release(worker);
              return;
            }
            worker.postMessage(
              { bitmap, index, type: FRAME_TYPE, quality: FRAME_QUALITY },
              [bitmap],
            );
          })
          .catch((error) => {
            release(worker);
            if (!closed) onError(error);
          });
      },
      close() {
        closed = true;
        while (idle.length) idle.pop().terminate();
        // Held until now: a worker fetches its script asynchronously after
        // construction, and revoking the URL up front can race that fetch.
        URL.revokeObjectURL(blobUrl);
      },
    };
  }

  //: The fallback, for a browser without OffscreenCanvas or workers: the same
  //: contract, one frame at a time, on the main thread. Slower by roughly the
  //: pool size, and correct.
  function createInlineEncoder(target, onDone, onError) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    let busy = false;
    let closed = false;

    return {
      get free() {
        return closed || busy ? 0 : 1;
      },
      submit(source, index, width, height) {
        busy = true;
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
        }
        context.imageSmoothingQuality = 'high';
        // Same-task readback, for the same reason `submit` is called where it is.
        context.drawImage(source, 0, 0, width, height);
        canvas.toBlob(
          async (blob) => {
            if (closed) return;
            try {
              if (!blob) throw new Error('the canvas could not be read');
              const response = await fetch(`${target}&index=${index}`, {
                method: 'POST',
                body: blob,
                headers: { 'Content-Type': FRAME_TYPE },
              });
              if (!response.ok) throw new Error(await reason(response));
              busy = false;
              if (!closed) onDone(index);
            } catch (error) {
              busy = false;
              if (!closed) onError(error);
            }
          },
          FRAME_TYPE,
          FRAME_QUALITY,
        );
      },
      close() {
        closed = true;
      },
    };
  }

  /* ---------------------------------------------------------------- misc --- */

  function abort(why) {
    const token = job && job.token;
    reset();
    viewer.setStatus(why ? `recording dropped — ${why}` : 'recording cancelled', 'error');
    // Best effort, and deliberately unawaited: the page has already forgotten
    // the recording, and the server sweeps an abandoned one on its own.
    if (token) post('cancel', { token }).catch(() => {});
  }

  function fail(error) {
    const token = job && job.token;
    reset();
    viewer.setStatus(`playblast failed — ${error.message || error}`, 'error');
    console.warn('playblast failed:', error);
    if (token) post('cancel', { token }).catch(() => {});
  }

  //: Put back the pixel ratio a recording raised. Idempotent: `finish` calls it
  //: as soon as the last frame is in, and `reset` -- which every ending goes
  //: through, cancels and failures included -- calls it again.
  function releaseResolution() {
    if (job && job.restorePixelRatio) {
      viewer.renderer.setPixelRatio(job.restorePixelRatio);
      job.restorePixelRatio = null;
    }
  }

  function reset() {
    releaseResolution();
    if (job && job.encoder) job.encoder.close();
    if (job && job.wasPlaying) viewer.setPlaying(true);
    job = null;
    button.textContent = 'Export Playblast';
  }

  //: Progress goes on the BUTTON, not the status line: the status line says
  //: what the page is showing, and a per-frame counter there would bury the
  //: version and the load state for the length of the recording. Clicking it
  //: cancels, which is exactly what a progress label wants to be attached to --
  //: hence the ✕, which is only shown while cancelling is possible.
  function label(text) {
    button.textContent = `${text}  ✕`;
  }

  async function post(action, payload) {
    const response = await fetch(`playblast/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await reason(response));
    return response.json();
  }

  //: The server states WHY it refused (a frame ceiling, a missing frame, no
  //: ffmpeg); showing "500" instead would send the user to a console they
  //: cannot open in a headset.
  //:
  //: `send_error` puts the message in the status line AND in its HTML body, so
  //: statusText is the cheap read and the body is the fallback for a proxy or a
  //: browser that drops the reason phrase. The body match stops at the tag, not
  //: at the first full stop -- these messages are sentences.
  async function reason(response) {
    if (response.statusText) return response.statusText;
    const text = await response.text().catch(() => '');
    const match = text.match(/<p>Message:\s*([^<]+)/i);
    return (match ? match[1] : `HTTP ${response.status}`).trim().replace(/\.$/, '');
  }

  function download(url) {
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = '';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  }
}
