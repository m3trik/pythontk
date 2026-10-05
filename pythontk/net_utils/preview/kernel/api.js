import * as THREE from 'three';
import {
  clipInfo, clips, mixer, playing, seek, selectClip, shotDescription, shotLabel, togglePlay,
} from './animation.js';
import { showDialog } from './dialog.js';
import { environment } from './environment.js';
import { formatBytes } from './format.js';
import { grab } from './grab.js';
import { addHeadsetCard } from './headset.js';
import { locomotion } from './locomotion.js';
import { current, materialsOf, modelBounds } from './model.js';
import { rig } from './rig.js';
import { camera, controls, pivot, policy, renderer, scene } from './scene.js';
import { headset } from './session.js';
import { specs } from './specs.js';
import { setStatus } from './status.js';
import { addPanel, windowFor } from './windows.js';

//: The version of the `viewer` object below -- the kernel API every script is
//: written against, and the one surface of this page that is PUBLISHED: a
//: script outside this package, and an app that vendors one, depend on it. A
//: member renamed or removed bumps it, and the old name stays as a warning
//: alias for one window first (CODE_STANDARD section 5, as in Python).
export const API_VERSION = 1;

// The extension seam. `PreviewServer` names a set of ES modules in the
// manifest; each is imported once and its default export called with the API
// object below. That is deliberately the ONLY way the page grows behaviour:
// this file stays the stable path, and a new capability (a turntable, a cost
// report, a debug overlay) is a module plus a registry entry rather than
// another branch in here.
//
// A script's failure is contained and, above all, quiet in the status line: an
// optional module that throws must not make a perfectly good preview LOOK
// broken, because the one place this is read is a headset where the console is
// not visible. It is logged, and the model keeps rendering.
//: url -> the module's import and registration, in flight or done.
const loadedScripts = new Map();
const scriptHooks = { load: [], frame: [], rendered: [], key: [] };

// Set from the manifest: a view-only share (the server's guest listener) says
// so, and the page then offers nothing that writes. That listener refuses every
// write, so each such control would be one that fails.
let isGuest = false;

export function setGuest(guest) {
  isGuest = guest;
}

// glTF node index -> Object3D, through the loader's association table rather
// than by name (the production assembly ships duplicate names). A name-verified
// entry wins over an unverified one, and an index the table lacks (a mesh
// shared by several nodes shares one association record) falls back to the
// unique object carrying that glTF name in its userData. Null when neither
// says. Every manifest a script binds by node index resolves through this.
function nodeResolver(gltf, model) {
  const parser = gltf.parser;
  const defs = parser.json.nodes || [];
  const table = new Map();
  for (const [object, assoc] of parser.associations) {
    if (!object?.isObject3D || assoc?.nodes === undefined) continue;
    const index = assoc.nodes;
    const def = defs[index];
    const named = !def?.name || object.name === THREE.PropertyBinding.sanitizeNodeName(def.name);
    if (named || !table.has(index)) table.set(index, object);
  }
  return (index) => {
    if (!Number.isInteger(index)) return null;
    if (table.has(index)) return table.get(index);
    const name = defs[index]?.name;
    if (!name) return null;
    const matches = [];
    model.traverse((object) => { if (object.userData?.name === name) matches.push(object); });
    return matches.length === 1 ? matches[0] : null;
  };
}

export const viewer = {
  /** The version of this API (`API_VERSION`): what a script checks it is written against. */
  apiVersion: API_VERSION,
  THREE,
  scene,
  renderer,
  camera,
  controls,
  pivot,
  setStatus,
  /**
   * Where the headset stands in the scene (see the rig section): `place` is
   * `{x, y, z, yaw}`, `toScene(position)` and `directionToScene(orientation)`
   * map a tracked pose into the scene, and `fade` is the blink's black.
   */
  rig,
  /**
   * The headset's frame on plain values -- what the page runs from each
   * XRFrame, so it runs as well on synthetic input: `begin()` as a session
   * starts, `step(delta, input)` each frame, `end()` as it ends, `recenter()`
   * as the headset's own recenter does, and `read(frame)` the XRFrame read
   * into the input `step` takes. `start` is where a session starts,
   * `{point, heading}`, or null.
   */
  headset,
  /**
   * Getting around (see the locomotion section): `enabled`, which the server's
   * switch sets on every poll, and the aim -- `aiming`, `arc`, `target` -- and
   * the walking `vignette`.
   */
  locomotion,
  /**
   * Grabbing (see the grab section): `register(object, {begin, move, end})`
   * makes an object and everything under it something a hand takes hold of,
   * and returns the unregister; `press(source, origin, direction, pose)`,
   * `drag(source, point, rotation)` and `release(source)` drive a hold on
   * plain values, as the pointer and the headset do; `pick(origin,
   * direction, far)` asks what a ray would take; `holding` lists the sources
   * holding something.
   */
  grab,
  /**
   * `nodeResolver(gltf, model)` -> `(index) => Object3D | null`: a glTF node
   * index (what a manifest in the file's extras binds by) to the object the
   * loader made of it -- by the loader's own association, never by a name a
   * production assembly repeats.
   */
  nodeResolver,
  /** The model currently on screen, or null between loads. */
  get model() { return current; },
  /** `{size, center}` of the current model, in metres (glTF's unit). */
  get bounds() { return modelBounds; },
  /**
   * What the model on screen is, measured once per load (see measureModel),
   * or null before the first: `file` `{bytes, jsonBytes}`; `load`
   * `{fetchMs, parseMs, setupMs, firstFrameMs}` (the last filled once the
   * first frame after the load has drawn); `objects` (the file's mesh nodes),
   * `meshes` (three.js Meshes -- one per primitive, each its own draw),
   * `triangles`, `vertices`; `size` `{x, y, z}` in metres; `materials`
   * `{file, instances}` (the file's, and the three.js copies made of them);
   * `lightmaps` -- null when nothing is baked -- `{lit, objects, baked,
   * unlit, foreign, maps, probe}`: objects lit by a lightmap, of all objects;
   * objects the file marks as baked (null when it marks none), those of them
   * with no lightmap bound, and objects lit by one they were never baked
   * with, by name; the distinct lightmap images; and whether the bake's
   * reflection probe is the environment; `animation` `{clips,
   * materials, tracks}`; and `issues` `[{level, text, names?}]` -- what the
   * load found wrong, as the HUD's warning line reports the first;
   * `probe` -- null without one -- the bake's reflection probe `{on, width,
   * height, bytes, bufferView, position, box, decodeMs, prefilterMs}` (as
   * `environment.probe`), its decode and prefilter also `load.probeMs`.
   */
  get specs() { return specs; },
  /** The live rendering policy, as read from the deliverable. */
  get policy() { return policy; },
  /**
   * What lights the model (the Environment window): `probe` -- the bake's
   * reflection probe on screen, or null -- `{on, width, height, bytes,
   * bufferView, position, box, decodeMs, prefilterMs}`, `position` and `box`
   * (`{min, max}`, null when read as distant) in metres; `setProbe(on)` turns
   * it on or off for this view, the studio lighting the model while it is off
   * as a file without one is lit, and returns the state it ended in; `maps`,
   * the environment maps the page holds in GPU memory now, `[{name, texture,
   * active}]` (`probe`, `studio`).
   */
  environment,
  /**
   * True on a view-only share (the server's guest listener), which refuses
   * every write -- so a script that posts to the server should offer nothing.
   */
  get guest() { return isGuest; },
  /** The AnimationMixer driving the current model's clips, or null. */
  get mixer() { return mixer; },
  /**
   * The clip the transport is on, or null when the model has none:
   * `{name, duration, fps, startFrame, endFrame, sequence}`. `sequence` marks
   * the synthetic whole-timeline entry — every declared shot laid back onto
   * the timeline it was cut from — which is a set of clips rather than one,
   * and is exactly as recordable as a single shot.
   */
  get clip() { return clipInfo(); },
  /** Whether the transport is playing. */
  get playing() { return playing; },
  /** Play or pause the transport; returns the state it ended in. */
  setPlaying(state) {
    if (playing !== !!state) togglePlay();
    return playing;
  },
  /**
   * Pose the model at `seconds` into the current clip, without advancing it.
   * Clamped to the clip. This is what makes a recording a playblast rather
   * than a screen capture: the caller owns the clock.
   */
  poseAt(seconds) {
    const info = clipInfo();
    if (!info) return false;
    seek(info.duration ? seconds / info.duration : 0);
    return true;
  },
  /**
   * The shot covering `seconds` of the whole-timeline clip — the same name the
   * transport readout shows, `'<name> (hold)'` inside a gap — or null when the
   * current clip is a single shot rather than the sequence.
   *
   * Reads the segment the last pose resolved to, so it answers for wherever the
   * model is NOW: call it after `poseAt(seconds)` with the same `seconds`, not
   * before, or it describes the previous frame.
   */
  shotAt(seconds) { return shotLabel(seconds); },
  /**
   * The description the DCC's Shots panel recorded for whatever the model is
   * posed at, or null when the deliverable states none. Reads the last pose,
   * so — like `shotAt` — call it after `poseAt`, not before.
   *
   * Answers on a single clip as well as on the sequence: it is written for a
   * burn-in, where the picker naming the clip on screen is no help.
   */
  descriptionAt() { return shotDescription(); },
  /**
   * Play a clip by name (the names `extras.animation_web` lists, which are the
   * DCC's shot names on a shot-split deliverable). Returns whether it matched.
   */
  playClip(name) {
    // `entry.clip` is null on the synthetic whole-timeline entry, which is a
    // set of shots rather than one clip -- reading `.name` off it threw on
    // every shots-only deliverable. Addressable by its own name too, so a
    // consumer can ask for the sequence the way it asks for a shot.
    const index = clips.findIndex((entry) => (
      entry.clip ? entry.clip.name === name : entry.sequence?.name === name
    ));
    if (index >= 0) selectClip(index);
    return index >= 0;
  },
  /**
   * Subscribe to a page event.
   *  'load'     ({model, gltf, lightmapped, materialCount, clips, specs}) after
   *             each model swap -- `specs` as `viewer.specs`; `lightmapped` and
   *             `materialCount` count three.js material INSTANCES (the lighting
   *             policy's gate, and its old denominator), not the file's
   *  'frame'    ({delta, time}) once per frame, before it is drawn
   *  'rendered' ({delta, time, cpuMs, renderMs}) once per frame, after it is
   *             drawn: the page's CPU time for the frame, and the part of it
   *             inside `renderer.render` (CPU-side; the GPU's is not measured)
   *  'key'      (KeyboardEvent) for keydown, after the page's own shortcuts
   */
  on(event, fn) { (scriptHooks[event] || (scriptHooks[event] = [])).push(fn); },
  /**
   * Add a button to the control bar, styled like the page's own. Scripts get
   * this rather than reaching for document.createElement so a module cannot
   * drift from the page's look. A script's controls belong in a category's
   * window (`viewer.window`), which keeps the bar to one button per category;
   * a bar button of its own is for the rare action that cannot wait a click.
   */
  addButton(label, onClick) {
    const button = document.createElement('button');
    button.className = 'ui';
    button.textContent = label;
    button.addEventListener('click', onClick);
    document.getElementById('controls').appendChild(button);
    return button;
  },
  /**
   * A panel of rows in the page's chrome, stacked top-right after the
   * windows, for a script with a report of its own -- the reason `addButton`
   * exists, for output: one look, one place. Hidden until `show()`.
   *
   * Returns `{element, shown, show(on), setRows(rows), addButton(label,
   * onClick), addSlider(...), addToggle(...)}`. `rows` are `[label, value,
   * level]` pairs, `{heading}` for a section title, or `{text, level}` for a
   * line of its own; `level: 'warn'` tints it. Everything is written as text,
   * never markup.
   */
  addPanel,
  /**
   * The window of a CATEGORY -- `'View'`, `'Environment'`, `'Inspect'`,
   * `'Export'`, `'Rig'`, or one of the script's own. The page sorts what it
   * offers into these rather than a bar button per feature: the bar carries
   * one button per category that has something in it, and each opens its
   * window, where every part of the page filing under that category has a
   * section. The first ask makes the window; every later one with the same
   * name returns it. `{title}` is its bar button's tooltip.
   *
   * Returns `{category, element, button, shown, show(on), toggle(),
   * onShow(fn), setBadge(text, owner), section(title)}`. `section` adds a
   * block the caller owns -- `{element, shown, show(on), remove(),
   * setRows(rows), addButton(label, onClick), addSlider(label, opts,
   * onInput), addToggle(label, {value, title}, onChange)}` -- `addToggle`
   * returns `{element, value, hidden, disabled}`, its `value` settable
   * without calling `onChange`. `onShow(fn)` calls `fn(shown)` as the window
   * opens and closes; `setBadge(text, owner)` marks the bar button while a
   * job runs with the window shut, `''` clears it -- per *owner* (the
   * script's name), since scripts share a window: the button shows the
   * newest one standing.
   */
  window: windowFor,
  /**
   * A card of text in the headset, for a script with something to say where
   * the page's chrome is not drawn -- the reason `addPanel` exists, for a
   * session: one look, one place. `{name, width, height, metres, radius,
   * over}`: a `width` x `height` canvas on a plane `metres` wide (the canvas's
   * aspect), its corners rounded by `radius` px, drawn `over` the model (never
   * behind a wall) when asked. It rides the headset's layer, so a still or a
   * playblast never carries it, and starts hidden.
   *
   * Returns `{mesh, draw(paint), place(head, look, where, delta)}`: `mesh` to
   * show, hide or fade; `draw` repaints the card and hands `paint(context,
   * width, height)` the canvas to write on; `place` stands it `where.ahead` m
   * in front of `head` (a scene point) looking along `look`, `where.aside`
   * radians to its left and `where.below` m under the eyes, facing them --
   * easing a shown card there over `where.ease` s when given, since text
   * locked to the head is hard to read.
   */
  addHeadsetCard,
  /**
   * *bytes* as the page quotes a size (`B`, `KB`, `MB`; tenths below 10), so
   * a script's figures read the way the status line's do.
   */
  formatBytes,
  /**
   * A node's materials as a list: `node.material` is one material, an ARRAY
   * of them (a multi-material mesh) or absent, and a walk that forgets a case
   * skips half a mesh's materials -- one normalization, the page's own.
   */
  materialsOf,
  /**
   * Ask for options before an action, in a modal wearing the page's chrome.
   *
   * `fields` are `{key, label, type, value, title}` — `'check'` (the default)
   * is a checkbox, `'choice'` a picker over `choices: [{value, label, title}]`
   * (an unknown `value` opens on the first), and `'note'` a read-only line,
   * which is how a prompt states what it is about to do without offering it
   * as a choice. Resolves to `{key: answer}` on confirm — `checked` for a
   * check, the chosen `value` for a choice — and to **null** on cancel (the
   * Cancel button, Escape, or the backdrop), so a caller branches on the
   * answer rather than on a flag inside it.
   *
   * While it is up the page behind it is inert — no shortcut fires, and focus
   * cannot leave the prompt.
   *
   * Scripts get this rather than `window.confirm` for the reason they get
   * `addButton` rather than `createElement`: one look, one place.
   */
  showDialog,
  /**
   * `{width, height, pixelRatio}` for a capture of the view whose long edge is
   * `maxEdge`: the size to hand over, and the pixel ratio to RENDER at for it
   * -- null when the drawing buffer is already at least that large, and the
   * capture is downsampled instead. A capture is therefore a real render at
   * its size, never an upscale of a smaller one.
   *
   * Clamped to what the GPU will allocate. A canvas asked for a buffer past
   * MAX_RENDERBUFFER_SIZE does not fail; it silently allocates a smaller one,
   * and a capture would read that stretched across the frame.
   *
   * The one sizing rule for everything the page writes to a file -- a
   * playblast's frames and an exported still -- so the two cannot disagree
   * about what a size means.
   */
  captureSize(maxEdge) {
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
  },
  /**
   * Why the server refused `response`, as a sentence to put in the status
   * line. The server states it (a frame ceiling, no ffmpeg, a body that is not
   * an image); showing "500" instead sends the user to a console they cannot
   * open in a headset.
   *
   * `send_error` puts the message in the status line AND in its HTML body, so
   * statusText is the cheap read and the body the fallback for a proxy or a
   * browser that drops the reason phrase. The body match stops at the tag, not
   * at the first full stop -- these messages are sentences.
   */
  async refusal(response) {
    if (response.statusText) return response.statusText;
    const text = await response.text().catch(() => '');
    const match = text.match(/<p>Message:\s*([^<]+)/i);
    return (match ? match[1] : `HTTP ${response.status}`).trim().replace(/\.$/, '');
  },
  /**
   * Save a file this server serves as a download, rather than navigating the
   * page to it -- which in a headset would replace the preview with an image
   * viewer or a video player.
   *
   * A browser honours `download` on the page's OWN origin only, and this page
   * is as validly open at localhost as at 127.0.0.1: pass a path relative to
   * the page, or a route that answers with `Content-Disposition: attachment`
   * (a recording's does), or a tab opened by the other spelling navigates.
   */
  download(url) {
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = '';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  },
};

export function emit(event, detail) {
  for (const fn of scriptHooks[event] || []) {
    try {
      fn(detail, viewer);
    } catch (error) {
      console.warn(`viewer script threw on '${event}':`, error);
    }
  }
}

// Import and register every script *urls* names, in order. Resolves once each
// has registered its hooks -- including one another poll is still importing,
// so no caller can run ahead of a script's registration: the poll that names
// a script and an asset together holds the asset's load until the script
// would see its 'load' event, however the two polls interleave.
export async function loadScripts(urls) {
  for (const url of urls || []) {
    let pending = loadedScripts.get(url);
    if (!pending) {
      // Claimed BEFORE the await: poll() fires once a second and the import is
      // async, so a second poll landing mid-import would otherwise load and
      // initialise the same module twice — two turntables fighting over one
      // rotation, and two buttons for it. The claim is the import itself, so a
      // later poll waits on it rather than skipping past it.
      pending = registerScript(url);
      loadedScripts.set(url, pending);
    }
    await pending;
  }
}

async function registerScript(url) {
  try {
    // Against the PAGE, not this module: the manifest's urls are the serve
    // root's, and a module's own `import()` resolves against its own URL.
    const module = await import(new URL(url, document.baseURI).href);
    await module.default?.(viewer);
  } catch (error) {
    console.warn(`viewer script failed to load: ${url}`, error);
  }
}
