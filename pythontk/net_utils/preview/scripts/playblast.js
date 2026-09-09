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

  Activated by the deliverable (`PreviewServer.AUTO_SCRIPTS`) whenever it ships
  an animation manifest — the button appears exactly when the clip picker does.
*/

//: Long edge of the recorded frame. This is a PREVIEW: the point is to send
//: someone what the shot looks like while they still care, so the default is
//: 720p-class rather than whatever delivery resolution the canvas happens to
//: be at. A canvas SMALLER than the cap is captured as it is; it never upscales.
//:
//: This USED to be the wall-clock dial -- at one frame at a time on the main
//: thread the cost was all PNG compression, so it tracked pixel count almost
//: exactly (1920 -> 1280 cut 2.25x the pixels and measured 2.1x faster). With
//: the worker pool below it no longer is: 1920 and 1280 measured 46.7ms and
//: 45.1ms per frame, because the compression now happens behind the render
//: rather than in front of it. What this number still buys is file size and
//: wire time -- which is what a reviewer on a headset over Wi-Fi actually
//: waits for -- so it stays at 720p-class by default. Raising it is now a
//: question about bytes, not about minutes.
const MAX_EDGE = 1280;

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
      start();
    }
  });
  button.title = 'Record the selected clip to a movie file (r)';

  let job = null;
  let burnIn = false;

  //: Opt-in, and OFF by default: a burn-in is drawn into the movie and cannot
  //: be taken out again, so the recording is what the reviewer saw unless
  //: someone asks for the annotation. It is also the only thing this tool draws
  //: that the scene did not -- hence a control of its own rather than a mode of
  //: the record button.
  const stampButton = viewer.addButton('Burn-in: off', () => {
    // Mid-recording it would annotate half the frames and not the others.
    if (job) {
      viewer.setStatus('finish or cancel the recording before changing burn-in', 'error');
      return;
    }
    burnIn = !burnIn;
    stampButton.textContent = `Burn-in: ${burnIn ? 'on' : 'off'}`;
  });
  stampButton.title = 'Draw the shot name, frame and time into the recorded movie';

  viewer.on('key', (event) => {
    if (event.key === 'r') button.click();
  });

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

  function start() {
    // The XR framebuffer is not the canvas: in an immersive session
    // `renderer.render` targets the device, and reading the canvas back would
    // capture whatever was last drawn to the mirror -- a file that looks broken
    // with nothing to say why. Refused rather than recorded wrong.
    if (viewer.renderer.xr?.isPresenting) {
      viewer.setStatus('exit VR to record a playblast', 'error');
      return;
    }
    const clip = viewer.clip;
    if (!clip) {
      viewer.setStatus('nothing to record — this model has no clips', 'error');
      return;
    }
    const fps = clip.fps;
    if (!fps) {
      // The manifest is where the authoring rate lives; without it the frames
      // could still be captured but not placed in time, and a movie at a guessed
      // rate is worse than none — it plays at the wrong speed and looks correct.
      viewer.setStatus('cannot record — the deliverable states no frame rate', 'error');
      return;
    }
    const frames = Math.max(1, Math.round(clip.endFrame - clip.startFrame) + 1);

    // Measured ONCE, here. Every frame of a movie must be the same size, and
    // the canvas is not: resizing the window mid-recording (or rotating a
    // phone) would otherwise change the frame size part way through, which
    // ffmpeg answers with a garbled encode rather than an error.
    const source = viewer.renderer.domElement;
    const scale = Math.min(1, MAX_EDGE / Math.max(source.width, source.height));

    job = {
      clip,
      fps,
      frames,
      issued: 0,
      completed: 0,
      done: false,
      token: null,
      encoder: null,
      width: Math.max(1, Math.round(source.width * scale)),
      height: Math.max(1, Math.round(source.height * scale)),
      pad: null,
      padContext: null,
      wasPlaying: viewer.playing,
    };
    if (burnIn) {
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
        shotName(seconds),
        stamp(index, seconds),
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

  function drawOverlay(context, width, height, left, right) {
    const size = Math.max(OVERLAY_MIN_SIZE, Math.round(height * OVERLAY_SCALE));
    const inset = Math.round(size * 0.75);
    const baseline = height - inset;
    context.save();
    context.font = `600 ${size}px ${OVERLAY_FONT}`;
    context.textBaseline = 'alphabetic';
    context.lineJoin = 'round';
    context.lineWidth = Math.max(2, size / 6);
    context.strokeStyle = 'rgba(0, 0, 0, 0.75)';
    context.fillStyle = 'rgba(255, 255, 255, 0.96)';
    for (const [text, align, x] of [
      [left, 'left', inset],
      [right, 'right', width - inset],
    ]) {
      context.textAlign = align;
      // Stroke then fill: the outline is what keeps the text legible over a
      // blown-out sky as well as over a dark background.
      context.strokeText(text, x, baseline);
      context.fillText(text, x, baseline);
    }
    context.restore();
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
    // No ✕: see the button handler -- there is nothing left to cancel.
    button.textContent = 'Encoding…';
    try {
      const report = await post('finish', { token: job.token });
      const seconds = Number(report.duration || 0).toFixed(1);
      viewer.setStatus(`playblast saved · ${report.frames} frames · ${seconds}s`, 'live');
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

  function reset() {
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
