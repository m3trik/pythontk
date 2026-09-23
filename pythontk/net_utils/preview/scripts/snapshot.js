/*
  Export Image — save the current view as a PNG.

  What is on screen, through the camera the page is looking through: the orbit
  the reviewer left it at, the pose the transport is holding, the turntable
  wherever it has got to. The HUD and the control bar are not in it -- they are
  page markup laid over the canvas, and the capture reads the canvas.

  RENDERED at the size the prompt asks for, never grabbed off the screen and
  scaled up: a view smaller than that size has its pixel ratio raised for the
  one frame (`viewer.captureSize`, the rule a playblast's frames follow), so 4K
  from a laptop window is a real 4K render. A larger view is downsampled.

  The PNG is POSTed to the preview server, which decides where it lands by the
  rule a playblast follows (`PreviewServer.save_snapshot`): beside the published
  file when there is one on disk, else in the serve root -- and then the page
  downloads it, because that is the only copy that outlives the session.

  Opt-in, like `turntable`: a Viewer Scripts row on the WebXR Preview panel.
*/

//: What the prompt offers. `maxEdge` is the still's long edge; null keeps the
//: drawing buffer exactly as the page draws it. The sized entries carry the
//: playblast presets' own labels and edges, so "High — 1440p" means one thing
//: on this page. `key` is what the tab remembers between prompts.
const SIZE_PRESETS = [
  {
    key: 'view',
    maxEdge: null,
    title: 'The view exactly as the page draws it now: this window\'s size at '
      + 'this display\'s pixel density.',
  },
  { key: 'standard', label: 'Standard — 1080p', maxEdge: 1920, title: '1920 px long edge.' },
  { key: 'high', label: 'High — 1440p', maxEdge: 2560, title: '2560 px long edge.' },
  {
    key: 'maximum',
    label: 'Maximum — 4K',
    maxEdge: 3840,
    title: '3840 px long edge. The GPU\'s own limit can cap it lower.',
  },
];

//: High, as the playblast defaults to: a still is most often looked at on a
//: screen at least the size of the one it was taken on, and one frame at 1440p
//: costs nothing a reviewer waits for.
const DEFAULT_PRESET = 'high';

//: Lossless: a lookdev still is compared pixel for pixel against the next one,
//: and the one route this posts to takes nothing else.
const IMAGE_TYPE = 'image/png';

function presetFor(key) {
  return SIZE_PRESETS.find((preset) => preset.key === key)
    || SIZE_PRESETS.find((preset) => preset.key === DEFAULT_PRESET);
}

export default function snapshot(viewer) {
  //: The size the last prompt was answered with, re-offered by the next: a
  //: reviewer taking one angle is about to take the next the same way.
  //: Session-lived, like the playblast's options.
  let chosen = DEFAULT_PRESET;
  let busy = false;

  const button = viewer.addButton('Export Image', () => {
    if (!busy) openOptions();
  });
  button.title = 'Save the current view as a PNG — asks for the size first';

  async function openOptions() {
    // Refused before the prompt: asking what size to save a view that cannot
    // be read is a worse answer than saying why.
    if (!capturable()) return;
    const canvas = viewer.renderer.domElement;
    const answer = await viewer.showDialog({
      title: 'Export Image',
      fields: [
        {
          key: 'preset',
          type: 'choice',
          label: 'Size',
          value: chosen,
          choices: SIZE_PRESETS.map(({ key, label, maxEdge, title }) => ({
            value: key,
            // Stated in pixels, read off the canvas NOW: "as shown" means a
            // different size on every window, and the prompt should say which.
            label: maxEdge ? label : `As shown — ${canvas.width}×${canvas.height}`,
            title,
          })),
          title: 'The size of the saved image. The view is rendered at that '
            + 'size, so a small window still saves a full-size image.',
        },
      ],
      confirm: 'Save',
    });
    if (!answer) return;   // cancelled -- or a push landed and dismissed it
    chosen = answer.preset;
    // Re-checked, not carried from the prompt: the page can enter VR or lose
    // its context while the prompt is up.
    if (!capturable()) return;
    save(presetFor(chosen));
  }

  //: Whether the view can be read right now, with the status line saying why
  //: when it cannot.
  function capturable() {
    // The XR framebuffer is not the canvas: in an immersive session `render`
    // targets the device, and reading the canvas would save the mirror -- an
    // image that looks broken with nothing to say why.
    if (viewer.renderer.xr?.isPresenting) {
      viewer.setStatus('exit VR to export an image', 'error');
      return false;
    }
    // A lost context draws nothing, so the save would be a blank PNG. The
    // page's own status already says so, and outranks this one.
    if (viewer.renderer.getContext().isContextLost()) return false;
    if (!viewer.model) {
      viewer.setStatus('nothing to export — no model is loaded', 'error');
      return false;
    }
    return true;
  }

  //: The view rendered at *preset*'s size, drawn onto a 2D canvas.
  //:
  //: Synchronous, render and readback in ONE task. Without
  //: `preserveDrawingBuffer` a WebGL drawing buffer is valid only until the
  //: browser composites, so a readback after anything is awaited reads an
  //: empty canvas -- and saves it, silently. The playblast rides the frame
  //: hook for the same reason; a still is one frame, so it renders its own.
  function capture(preset) {
    const { renderer, scene, camera } = viewer;
    const source = renderer.domElement;
    const size = preset.maxEdge
      ? viewer.captureSize(preset.maxEdge)
      : { width: source.width, height: source.height, pixelRatio: null };
    const pad = document.createElement('canvas');
    pad.width = size.width;
    pad.height = size.height;
    const context = pad.getContext('2d');
    context.imageSmoothingQuality = 'high';
    const restore = size.pixelRatio ? renderer.getPixelRatio() : null;
    try {
      if (restore) renderer.setPixelRatio(size.pixelRatio);
      renderer.render(scene, camera);
      context.drawImage(source, 0, 0, size.width, size.height);
    } finally {
      if (restore) {
        renderer.setPixelRatio(restore);
        // Resizing the buffer back clears it, and the browser composites at
        // the end of this task -- so without a render here the page shows one
        // blank frame between the capture and its own next one.
        renderer.render(scene, camera);
      }
    }
    return pad;
  }

  async function save(preset) {
    busy = true;
    button.textContent = 'Saving…';
    try {
      const pad = capture(preset);
      // `toBlob` copies the pad when it is CALLED, so what follows may await.
      const blob = await new Promise((resolve) => pad.toBlob(resolve, IMAGE_TYPE));
      if (!blob) throw new Error('the view could not be read');
      const response = await fetch('snapshot', {
        method: 'POST',
        body: blob,
        headers: { 'Content-Type': IMAGE_TYPE },
      });
      if (!response.ok) throw new Error(await viewer.refusal(response));
      const report = await response.json();
      viewer.setStatus(`image saved · ${report.name} · ${pad.width}×${pad.height}`, 'live');
      // Downloaded only when the server had nowhere durable to put it: a still
      // written beside the GLB it was taken of is already where its owner
      // keeps it, and a second copy in Downloads is noise.
      if (report.in_serve_root && report.url) viewer.download(report.url);
      else console.info(`image written to ${report.output}`);
    } catch (error) {
      viewer.setStatus(`image export failed — ${error.message || error}`, 'error');
      console.warn('image export failed:', error);
    } finally {
      busy = false;
      button.textContent = 'Export Image';
    }
  }
}
