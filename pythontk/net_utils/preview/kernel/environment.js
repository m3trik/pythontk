// The Environment window: what lights the model, and the switch for the
// deliverable's own reflection probe.
//
// A baked deliverable carries the room its bake lit as a reflection probe
// (probe.js); the page lights the model with it, or with its studio when the
// file has none. The window says which, what the probe is -- its size in the
// file, where it was captured, the box its reflections project onto -- and
// turns it off and on: off, the model draws exactly as a deliverable without
// a probe does (the studio, the materials on their own programs), which is a
// preview the export can produce, so the switch shows the deliverable both
// ways rather than a look it cannot ship. Inspect, open beside it, times the
// difference. Each load starts with the probe on: the deliverable as it ships.
//
// "Show capture point and box" draws where the probe was captured and the box
// its reflections project onto, over the model -- the check that a bake put
// its probe where it can see the room. An open face (a courtyard's sky, an
// exterior's sides) is recorded far out and drawn at the model's reach.

import * as THREE from 'three';
import { formatBytes } from './format.js';
import { applyLighting } from './lightmaps.js';
import { current } from './model.js';
import { probeEnvironment, probeState, probeTarget, setProbeOn } from './probe.js';
import { DEFAULT_PROBE_INTENSITY, onBeforeDraw, policy, scene, studioTarget } from './scene.js';
import { paintStats, specs } from './specs.js';
import { windowFor } from './windows.js';

//: A box face this far from the capture point is an OPEN one: the baker
//: records an open face a kilometre or more out (mayatk ProbePlacement), and
//: no room is half a kilometre across.
const OPEN_FACE_M = 500;
//: The helpers' colour: the page's accent.
const HELPER_COLOR = 0x7cc3ff;
//: Box faces in the order a probe's `{min, max}` holds them, by axis, as the
//: window names an open one (glTF is Y-up).
const FACE_NAMES = [['−X side', '+X side'], ['floor', 'ceiling'], ['−Z side', '+Z side']];

const win = windowFor('Environment', {
  title: "What lights the model: the bake's reflection probe, or the page's studio",
});
const section = win.section();
const probeSwitch = section.addToggle(
  'Reflection probe',
  {
    value: true,
    title: "Light the model with the room its bake captured (on), or with the page's "
      + 'studio as a file without a probe is lit (off). Inspect times either.',
  },
  (on) => environment.setProbe(on),
);
const helperSwitch = section.addToggle(
  'Show capture point and box',
  {
    value: false,
    title: 'Draw where the probe was captured and the box its reflections project onto',
  },
  (on) => showHelpers(on),
);

// The helpers drawn while asked for: the capture point and the box's edges, in
// the MODEL's frame -- they follow it each frame, as the probe's lookups do --
// but never children of it, so they are no part of what Frame and the HUD
// measure.
let helpers = [];
let helpersShown = false;

onBeforeDraw(() => {
  if (!current) return;
  for (const helper of helpers) helper.matrixWorld.copy(current.matrixWorld);
});

// Which faces of *probe*'s box are open (see OPEN_FACE_M), as `[[min], [max]]`
// booleans, or null when it has no box.
function openFaces(probe) {
  if (!probe?.box) return null;
  return ['min', 'max'].map((side) => probe.box[side].map(
    (value, axis) => Math.abs(value - probe.position[axis]) > OPEN_FACE_M,
  ));
}

function disposeHelpers() {
  for (const helper of helpers) {
    helper.removeFromParent();
    helper.geometry.dispose();
    helper.material.dispose();
  }
  helpers = [];
}

function buildHelpers(probe) {
  disposeHelpers();
  if (!probe) return;
  const position = new THREE.Vector3().fromArray(probe.position);
  // The model's reach: where an open face is drawn, and the marker's scale.
  const reach = new THREE.Box3().setFromObject(current || new THREE.Object3D());
  const span = reach.isEmpty() ? 4 : reach.getSize(new THREE.Vector3()).length();
  const radius = THREE.MathUtils.clamp(span * 0.01, 0.03, 0.15);
  const material = () => new THREE.MeshBasicMaterial({
    color: HELPER_COLOR, depthTest: false, transparent: true, opacity: 0.9,
  });
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 16, 12).translate(position.x, position.y, position.z), material());
  const made = [marker];
  const open = openFaces(probe);
  if (open) {
    const lo = new THREE.Vector3().fromArray(probe.box.min);
    const hi = new THREE.Vector3().fromArray(probe.box.max);
    const out = Math.max(span, 1);
    for (let axis = 0; axis < 3; axis += 1) {
      if (open[0][axis]) lo.setComponent(axis, position.getComponent(axis) - out);
      if (open[1][axis]) hi.setComponent(axis, position.getComponent(axis) + out);
    }
    const size = hi.clone().sub(lo);
    const middle = hi.clone().add(lo).multiplyScalar(0.5);
    const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(size.x, size.y, size.z));
    edges.translate(middle.x, middle.y, middle.z);
    made.push(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({
      color: HELPER_COLOR, depthTest: false, transparent: true, opacity: 0.9,
    })));
  }
  for (const helper of made) {
    helper.name = 'probeHelper';
    // Placed by hand each frame (above), after the scene graph updates.
    helper.matrixAutoUpdate = false;
    helper.matrixWorldAutoUpdate = false;
    helper.frustumCulled = false;
    helper.renderOrder = 999;
    helper.visible = helpersShown;
    if (current) helper.matrixWorld.copy(current.matrixWorld);
    scene.add(helper);
  }
  helpers = made;
}

function showHelpers(on) {
  helpersShown = Boolean(on);
  if (helpersShown && !helpers.length) buildHelpers(probeState());
  for (const helper of helpers) helper.visible = helpersShown;
}

const metres = (values) => values.map((value) => value.toFixed(2)).join(', ');

// The window's rows and switches for the model on screen.
function rowsFor(probe) {
  if (!probe) {
    return [
      ['lighting', 'the studio (three.js RoomEnvironment)'],
      ['level', policy.environmentIntensity.toFixed(2)],
      {
        text: 'This file carries no reflection probe. The Lightmap Baker captures one with '
          + 'each bake (the camera button on Packing).',
      },
    ];
  }
  const rows = [
    ['lighting', probe.on ? "the bake's reflection probe" : 'the studio (probe off)'],
    ['probe', `${probe.width} × ${probe.height} HDR · ${formatBytes(probe.bytes)} in the file`],
    ['captured at', `${metres(probe.position)} m`],
  ];
  const open = openFaces(probe);
  if (!open) {
    rows.push(['reflections', 'read as distant (an open scene)']);
  } else {
    const closed = (axis) => !open[0][axis] && !open[1][axis];
    const extent = [0, 1, 2].map((axis) => (
      closed(axis) ? (probe.box.max[axis] - probe.box.min[axis]).toFixed(1) : '∞'
    ));
    const named = open.flatMap((sides, side) => sides.flatMap(
      (isOpen, axis) => (isOpen ? [FACE_NAMES[axis][side]] : []),
    ));
    rows.push([
      'reflections',
      `projected onto the room, ${extent.join(' × ')} m`
        + (named.length ? ` · open (read as distant): ${named.join(', ')}` : ''),
    ]);
  }
  rows.push(['level', probe.on ? probeLevel() : policy.environmentIntensity.toFixed(2)]);
  return rows;
}

// The level the probe plays at: the deliverable's, which a producer may
// publish other than the default -- named as the bake's unit only when it is.
function probeLevel() {
  const level = policy.probeIntensity;
  const unit = Math.abs(level - DEFAULT_PROBE_INTENSITY) < 1e-6;
  return `${level.toFixed(2)}${unit ? " (1/π: the bake's units)" : ''}`;
}

// Bring the window, the HUD's line and the helpers up to date with the model
// on screen -- after each load, and after the switch.
export function syncEnvironment() {
  const probe = probeState();
  section.setRows(rowsFor(probe));
  probeSwitch.hidden = !probe;
  helperSwitch.hidden = !probe;
  probeSwitch.value = Boolean(probe?.on);
  if (specs) {
    specs.probe = probe;
    if (specs.lightmaps) specs.lightmaps.probe = Boolean(probeEnvironment());
    paintStats(specs);
  }
}

// A new model is up: its probe's helpers replace the last one's.
export function environmentLoaded() {
  disposeHelpers();
  if (helpersShown) buildHelpers(probeState());
  syncEnvironment();
}

export const environment = {
  /**
   * The bake's reflection probe on screen, or null: `{on, width, height,
   * bytes, bufferView, position, box, decodeMs, prefilterMs}` -- `position`
   * and `box` (`{min, max}`, null when it is read as distant) in metres, the
   * file's frame.
   */
  get probe() {
    return probeState();
  },
  /**
   * Turn the probe on or off for this view (the Environment window's switch);
   * returns the state it ended in -- false with no probe loaded.
   */
  setProbe(on) {
    const state = setProbeOn(on);
    applyLighting();
    syncEnvironment();
    return state;
  },
  /**
   * The environment maps the page holds in GPU memory now, `[{name, texture,
   * active}]` -- `probe` while one is loaded (on or off), `studio` while it
   * is built: what Inspect counts.
   */
  get maps() {
    const maps = [];
    const probe = probeTarget();
    if (probe) maps.push({ name: 'probe', texture: probe.texture, active: scene.environment === probe.texture });
    const studio = studioTarget();
    if (studio) maps.push({ name: 'studio', texture: studio.texture, active: scene.environment === studio.texture });
    return maps;
  },
};

syncEnvironment();
