/*
  Articulated rig — pose the DCC's articulated rigs, and let a hand move them.

  An articulated rig (mayatk `ArticulatedRig`) is a prop of rigid parts on
  hinge, swivel, ball and slide joints: one joint per moving part, the part
  parented under it, every channel 0 at rest. The joints' own keys ride the
  FBX -> glTF hop as ordinary node animation, so a clip plays with no help;
  what does NOT survive is the rig -- which channels each joint turns on,
  their limits, which parts a hand grabs. `MeshConvert.apply_glb_articulation`
  (pythontk) writes that as root `extras.articulation_web`: the
  `articulation` scene record with every joint and grabbed part bound to its
  glTF node index. This script is the runtime half:

    load   read the manifest; map node indices to Object3Ds through the
           loader's association table (names are not unique); build each
           rig's model; measure the file's unit against the record's off the
           rest translations of the joints that cannot slide; register every
           grabbed part with the page's `viewer.grab`, and show each rig's
           section of the Rig window (a slider per channel, the joints'
           values, Reset, Return to animation, joint axes).
    grab   a hold reads the pose on screen back into the model's state (a
           playing clip included), records the held point in its joint's
           frame, and each move solves the state that brings it to the hand
           -- a held link hanging off a ball takes the hand's turn, and with a
           mouse it keeps the rotation it was taken with.
    frame  a rig a hand or a slider has posed is re-posed every frame, AFTER
           the animation mixer; it holds that pose until Return to animation,
           or until the transport starts playing again.

  THE MODEL is `ArticulationModel` (model.js beside this): pythontk's
  `geo_utils/articulation/model.py` ported operation for operation, pinned
  against it by the reference's golden cases, with no three.js -- so a
  production WebXR app on any framework can vendor it without the preview.
  This module is its three.js adapter, and re-exports it for a script that
  imports the model from here.

  Activate:  automatic -- PreviewServer.AUTO_SCRIPTS turns it on for any
             deliverable whose root extras carry `articulation_web`;
             or bridge.push(scripts=["articulated_rig"]).
*/

import { norm, qmul, scale } from '../../kernel/math.js';
import { ARTICULATION } from '../../kernel/records.js';
import { ArticulationModel, ROTATE_CHANNELS } from './model.js';

export { ArticulationModel, ROTATE_CHANNELS, ROTATE_ORDERS, TRANSLATE_CHANNELS } from './model.js';

//: The manifest this reads: the articulation record's declared web projection.
const MANIFEST_KEY = ARTICULATION.webKey;
//: How often the panel's values refresh while it is shown, in seconds.
const PANEL_INTERVAL = 0.1;

/* -------------------------------------------------------------- the file --- */

function readManifest(parser) {
  let raw = parser?.json?.extras?.[MANIFEST_KEY];
  if (typeof raw === 'string') {
    try { raw = JSON.parse(raw); } catch { return null; }
  }
  return raw && typeof raw === 'object' ? raw : null;
}

/* ------------------------------------------------------------- the rigs --- */

// A rig on screen: its model, its joints' nodes, the space its root joints
// hang in, the file's unit against the record's, and the state last posed.
function bindRig(THREE, record, resolve) {
  const model = new ArticulationModel(record);
  const nodes = record.joints.map((joint) => resolve(joint.node));
  const missing = record.joints.filter((_joint, i) => !nodes[i]).map((joint) => joint.name);
  if (missing.length) {
    console.warn(`articulated_rig: ${record.name}: joints not in the scene: ${missing.join(', ')}`);
    return null;
  }
  // The model's parents, not the record's: it reads a missing one as a root.
  const roots = nodes.filter((_node, i) => model.joints[i].parent === null);
  const space = roots[0].parent;
  if (roots.some((node) => node.parent !== space)) {
    console.warn(`articulated_rig: ${record.name}: its root joints hang under different nodes`);
    return null;
  }
  const locals = nodes.map((node) => [node.position.toArray(), node.quaternion.toArray()]);
  const unit = model.scaleOf(locals);
  const rig = {
    record,
    model,
    nodes,
    space,
    unit,
    state: model.read(locals.map(([t, q]) => [scale(t, 1 / unit), q])),
    posed: false,
    holds: new Map(),
    grab: [],
    helpers: [],
  };
  for (const entry of record.grab || []) {
    const node = resolve(entry.node);
    if (node) rig.grab.push({ node, joint: entry.joint, name: entry.name ?? node.name });
  }
  return rig;
}

// The pose on screen as a state, read with the last state as the hint so a
// wound hinge or a ball past 90 degrees reads on rather than flipping.
function readRig(rig) {
  const locals = rig.nodes.map((node) => [scale(node.position.toArray(), 1 / rig.unit), node.quaternion.toArray()]);
  return rig.model.read(locals, rig.state);
}

function applyRig(rig, state) {
  rig.state = state;
  rig.model.pose(state).forEach(([t, q], index) => {
    const node = rig.nodes[index];
    node.position.fromArray(scale(t, rig.unit));
    node.quaternion.fromArray(q);
  });
}

// World <-> the rig's space (the root joints' parent), in record units.
function toRig(THREE, rig, point) {
  rig.space.updateWorldMatrix(true, false);
  const local = rig.space.worldToLocal(point.clone());
  return scale(local.toArray(), 1 / rig.unit);
}

function spaceRotation(THREE, rig) {
  rig.space.updateWorldMatrix(true, false);
  const q = new THREE.Quaternion();
  rig.space.matrixWorld.decompose(new THREE.Vector3(), q, new THREE.Vector3());
  return q;
}

function handlerFor(THREE, session, rig, entry) {
  return {
    begin({ source, point }) {
      rig.state = readRig(rig);
      const local = rig.model.toLocal(rig.state, entry.joint, toRig(THREE, rig, point));
      rig.holds.set(source, {
        joint: entry.joint,
        local,
        rotation: rig.model.world(rig.state)[entry.joint][1],
      });
      rig.posed = true;
      return true;
    },
    move({ source, point, rotation }) {
      const hold = rig.holds.get(source);
      if (!hold) return;
      let wanted = hold.rotation;
      if (rotation) {
        // A world turn seen from the rig's space: out of it, turn, back in.
        const s = spaceRotation(THREE, rig);
        const turn = s.clone().invert().multiply(rotation).multiply(s);
        wanted = qmul(turn.toArray(), hold.rotation);
      }
      const state = rig.model.solve(rig.state, hold.joint, hold.local, toRig(THREE, rig, point), wanted);
      applyRig(rig, state);
      session.dirty = true;
    },
    end({ source }) {
      rig.holds.delete(source);
    },
  };
}

/* ------------------------------------------------------------- the panel --- */

// "<rig>_<part>_jnt" read as "<part> <channel>".
function channelLabel(rig, slot) {
  const [joint, channel] = rig.model.channels[slot];
  let name = rig.model.joints[joint].name;
  const prefix = `${rig.record.name}_`;
  if (name.startsWith(prefix)) name = name.slice(prefix.length);
  if (name.endsWith('_jnt')) name = name.slice(0, -4);
  return `${name} ${channel}`;
}

function formatValue(rig, slot, value) {
  const channel = rig.model.channels[slot][1];
  if (ROTATE_CHANNELS.includes(channel)) return `${value.toFixed(1)}°`;
  // Record units times the file's unit is metres in a glTF.
  return `${(value * rig.unit * 100).toFixed(1)} cm`;
}

function buildPanel(viewer, session, rig) {
  // A section of the Rig window (`viewer.window`), one per rig, opened with
  // the model on the desktop: the sliders are how a rig is posed by hand.
  const rigWindow = viewer.window('Rig', { title: 'Pose the articulated rigs by hand' });
  const panel = rigWindow.section(`Articulated rig: ${rig.record.name}`);
  const reach = rig.model.joints.reduce((sum, joint) => sum + norm(joint.t), 0) || 1;
  rig.sliders = rig.model.channels.map((_c, slot) => {
    let [lo, hi] = rig.model.limits(slot);
    const rotation = ROTATE_CHANNELS.includes(rig.model.channels[slot][1]);
    if (lo === null) lo = rotation ? -180 : -reach;
    if (hi === null) hi = rotation ? 180 : reach;
    return panel.addSlider(
      channelLabel(rig, slot),
      {
        min: lo,
        max: hi,
        step: rotation ? 0.1 : (hi - lo) / 1000,
        value: rig.state[slot],
        format: (v) => formatValue(rig, slot, v),
      },
      (value) => {
        const state = rig.state.slice();
        state[slot] = value;
        applyRig(rig, rig.model.clamp(state));
        rig.posed = true;
      },
    );
  });
  panel.addButton('Reset', () => {
    applyRig(rig, rig.model.restState());
    rig.posed = true;
  });
  panel.addButton('Return to animation', () => {
    rig.posed = false;
  });
  panel.addButton('Joint axes', () => {
    const on = !rig.helpers.some((helper) => helper.visible);
    for (const helper of rig.helpers) helper.visible = on;
  });
  if (!viewer.renderer.xr.isPresenting) rigWindow.show();
  rig.panel = panel;
  const size = 0.12 * reach * rig.unit;
  for (const node of rig.nodes) {
    const helper = new viewer.THREE.AxesHelper(size);
    helper.visible = false;
    node.add(helper);
    rig.helpers.push(helper);
  }
}

function refreshPanel(rig) {
  if (!rig.panel?.shown) return;
  const rows = [];
  rows.push(['grab', rig.grab.map((g) => g.name).join(', ') || '—']);
  rows.push(['mode', rig.posed ? (rig.holds.size ? 'held' : 'posed (holding)') : 'animation']);
  rig.panel.setRows(rows);
  rig.sliders.forEach((slider, slot) => { slider.value = rig.state[slot]; });
}

/* --------------------------------------------------------------- session --- */

function build(viewer, model, gltf) {
  const { THREE } = viewer;
  const manifest = readManifest(gltf.parser);
  if (!manifest || !Array.isArray(manifest.rigs) || !manifest.rigs.length) return null;
  const resolve = viewer.nodeResolver(gltf, model);
  const session = { rigs: [], unregister: [], dirty: false, sinceRefresh: 0, wasPlaying: viewer.playing };
  for (const record of manifest.rigs) {
    let rig = null;
    try {
      rig = bindRig(THREE, record, resolve);
    } catch (error) {
      console.warn(`articulated_rig: ${record?.name}: not readable`, error);
    }
    if (!rig) continue;
    for (const entry of rig.grab) {
      session.unregister.push(viewer.grab.register(entry.node, handlerFor(THREE, session, rig, entry)));
    }
    buildPanel(viewer, session, rig);
    session.rigs.push(rig);
  }
  return session.rigs.length ? session : null;
}

function teardown(session) {
  if (!session) return;
  for (const unregister of session.unregister) unregister();
  for (const rig of session.rigs) {
    for (const helper of rig.helpers) helper.removeFromParent();
    rig.panel?.remove();
  }
}

export default function articulatedRig(viewer) {
  let session = null;
  if (viewer.model) {
    console.warn('articulated_rig: a model loaded before this script; its rigs are not grabbable until the next push');
  }

  viewer.on('load', ({ model, gltf }) => {
    teardown(session);
    session = null;
    try {
      session = build(viewer, model, gltf);
    } catch (error) {
      console.warn('articulated_rig: could not read the articulation manifest', error);
    }
    // Hung on the model so it dies with it, and so a test can read what this
    // script made of the file without the script exposing anything global.
    if (session) model.userData.articulatedRig = session;
  });

  viewer.on('frame', ({ delta }) => {
    if (!session) return;
    // Pressing play hands every rig back to its clip.
    const playing = viewer.playing;
    if (playing && !session.wasPlaying) for (const rig of session.rigs) rig.posed = rig.holds.size > 0;
    session.wasPlaying = playing;
    session.sinceRefresh += delta;
    const refresh = session.sinceRefresh >= PANEL_INTERVAL;
    if (refresh) session.sinceRefresh = 0;
    for (const rig of session.rigs) {
      // After the mixer: a posed rig wins over the clip underneath it.
      if (rig.posed) applyRig(rig, rig.state);
      else if (refresh) rig.state = readRig(rig);
      if (refresh) refreshPanel(rig);
    }
  });
}
