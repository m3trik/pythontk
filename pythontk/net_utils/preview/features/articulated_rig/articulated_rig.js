/*
  Articulated rig — pose the DCC's articulated rigs, and let a hand move them.

  An articulated rig (mayatk `ArticulatedRig`) is a prop of rigid parts on
  hinge, swivel, ball and slide joints: one joint per moving part, every
  channel 0 at rest. The joints' own keys ride the FBX -> glTF hop as ordinary
  node animation, so a clip plays with no help; what does NOT survive is the
  rig -- which channels each joint turns on, about which anchor, their limits,
  which parts a hand grabs. The GLB carries that as the `articulation` record
  itself, in its root extras (`MeshConvert.apply_glb_articulation`, pythontk:
  one layout, two containers), in glTF's frame (metres). This script is the
  runtime half:

    load   read the record; resolve each rig's joints by name over every node
           of the file, as one assignment, and each grabbed part by name under
           its joint's node (kernel/resolve.js, the rule every record's reader
           shares: exactly, else by the name after the last ':'; a name two
           nodes answer binds neither, and its rig is not posed); build each
           rig's model; register every grabbed part with the page's
           `viewer.grab`, and show each rig's section of the Rig window (a
           slider per channel, the joints' values, Reset, Return to
           animation, joint axes).
    pose   every part is posed from its parent JOINT's pose, never its scene
           parent's: a part nested under its parent joint's node takes its
           joint's local as it stands, and one hanging anywhere else (a flat
           model, every part under the root) is placed where that joint's
           pose puts it. A root joint's chain is posed and solved in the
           space its root hangs in.
    grab   a hold reads the pose on screen back into the model's state (a
           playing clip included), records the held point in its joint's
           frame, and each move solves the state that brings it to the hand
           -- a held link hanging off a ball takes the hand's turn, and with a
           mouse it keeps the rotation it was taken with.
    frame  a rig a hand or a slider has posed is re-posed every frame, AFTER
           the animation mixer; it holds that pose until Return to animation,
           or until the transport starts playing again.

  THE MODEL is `ArticulationModel` (model.js beside this): pythontk's
  `geo_utils/articulation/model.py` ported operation for operation, anchors
  included, pinned against it by the reference's golden cases, with no
  three.js -- so a production WebXR app on any framework can vendor it
  without the preview. This module is its three.js adapter, and re-exports it
  for a script that imports the model from here.

  Activate:  automatic -- PreviewServer.AUTO_SCRIPTS turns it on for any
             deliverable whose root extras carry `articulation`;
             or bridge.push(scripts=["articulated_rig"]).
*/

import { readBlock } from '../../kernel/extras.js';
import { norm, qmul } from '../../kernel/math.js';
import { ARTICULATION } from '../../kernel/records.js';
import { indexNodes, resolveIndexed } from '../../kernel/resolve.js';
import { ArticulationModel, ROTATE_CHANNELS } from './model.js';

export { ArticulationModel, ROTATE_CHANNELS, ROTATE_ORDERS, TRANSLATE_CHANNELS } from './model.js';

//: How often the panel's values refresh while it is shown, in seconds.
const PANEL_INTERVAL = 0.1;

/* -------------------------------------------------------------- the file --- */

// The articulation record (the ROOT's extras, where the conversion carries it
// and PreviewServer.AUTO_SCRIPTS looks).
/** @returns {import('../../kernel/records.js').ArticulationRecord | null} */
function readRecord(parser) {
  return readBlock(parser?.json?.extras, ARTICULATION);
}

// The one node a resolution names, or null (absent, or a tie).
const single = (found) => (found.nodes && found.nodes.length === 1 ? found.nodes[0] : null);

/* ------------------------------------------------------------- the rigs --- */

// A rig on screen: its model, its joints' nodes, each joint's frame (its
// parent joint's node, or for a root the node it hangs under), the space each
// joint's chain is solved in, and the state last posed. *found* is the
// resolution of each of the rig's joints (resolve.js); *nameOf* names a node
// as the file spells it. 'absent' for a rig none of whose joints is here.
/** @param {import('../../kernel/records.js').ArticulationRig} record */
function bindRig(THREE, json, record, found, resolve, nameOf) {
  const joints = Array.isArray(record.joints) ? record.joints : [];
  const tied = found.findIndex((entry) => entry.status === 'ambiguous');
  if (tied >= 0) {
    console.warn(`articulated_rig: ${record.name}: joint ${joints[tied]?.name} names several nodes (${found[tied].leaves.join(', ')}) -- ambiguous, the rig is not posed`);
    return null;
  }
  const indices = found.map(single);
  if (!joints.length || indices.every((index) => index === null)) return /** @type {'absent'} */ ('absent');
  const nodes = indices.map((index) => (index === null ? null : resolve(index)));
  const missing = joints.filter((_joint, i) => !nodes[i]).map((joint) => joint?.name);
  if (missing.length) {
    console.warn(`articulated_rig: ${record.name}: joints not in the scene: ${missing.join(', ')}`);
    return null;
  }
  // Each joint named by its node, as the file spells it.
  const model = new ArticulationModel({
    ...record,
    joints: joints.map((joint, i) => ({ ...joint, name: nameOf(indices[i]) })),
  });
  // The model's parents, not the record's: it reads a missing one as a root.
  const frames = model.joints.map((joint, i) => (joint.parent === null ? nodes[i].parent : nodes[joint.parent]));
  const spaces = model.joints.map((_joint, i) => nodes[model.chain(i)[0]].parent);
  if (spaces.some((space) => !space)) {
    console.warn(`articulated_rig: ${record.name}: a root joint hangs under nothing`);
    return null;
  }
  const rig = {
    record,
    model,
    nodes,
    frames,
    spaces,
    // The space of the first root's chain: every joint's, where the roots
    // share the node they hang under (a DCC rig's group).
    space: spaces[0],
    // Whether a joint's node hangs under its frame: posed by its local alone.
    nested: nodes.map((node, i) => node.parent === frames[i]),
    state: [],
    posed: false,
    holds: new Map(),
    grab: [],
    helpers: [],
    scratch: { a: new THREE.Matrix4(), b: new THREE.Matrix4(), v: new THREE.Vector3(), s: new THREE.Vector3(), q: new THREE.Quaternion(), one: new THREE.Vector3(1, 1, 1) },
  };
  rig.state = model.read(localsOf(rig));
  for (const entry of record.grab || []) {
    const joint = entry?.joint;
    if (!(Number.isInteger(joint) && joint >= 0 && joint < nodes.length)) {
      console.warn(`articulated_rig: ${record.name}: grab part ${entry?.node} rides no joint of the rig (${joint}); it is not grabbable`);
      continue;
    }
    // A part by its name under its joint's node: a production assembly
    // repeats part names, and the joint says which one is meant.
    const [hit] = resolveIndexed(indexNodes(json, { meshes: false, under: indices[joint] }), [{ name: entry.node }]);
    const index = single(hit);
    const node = index === null ? null : resolve(index);
    if (!node) {
      console.warn(`articulated_rig: ${record.name}: grab part ${entry.node} is ${hit.status === 'ambiguous' ? 'ambiguous' : 'not found'} under joint ${model.joints[joint].name}; it is not grabbable`);
      continue;
    }
    rig.grab.push({ node, joint, name: nameOf(index) });
  }
  return rig;
}

// *node*'s world matrix without its own scale -- its place and turn: the
// frame a joint's node gives the parts posed from it.
function frameMatrix(rig, node, out) {
  node.parent.updateWorldMatrix(true, false);
  out.compose(node.position, node.quaternion, rig.scratch.one);
  return out.premultiply(node.parent.matrixWorld);
}

// Each joint's local as the scene holds it: a nested node's own, else its
// place and turn in its parent joint's frame.
function localsOf(rig) {
  const { a, b, v, s, q } = rig.scratch;
  return rig.nodes.map((node, i) => {
    if (rig.nested[i]) return [node.position.toArray(), node.quaternion.toArray()];
    frameMatrix(rig, rig.frames[i], a).invert();
    a.multiply(frameMatrix(rig, node, b)).decompose(v, q, s);
    return [v.toArray(), q.toArray()];
  });
}

// The pose on screen as a state, read with the last state as the hint so a
// wound hinge or a ball past 90 degrees reads on rather than flipping.
function readRig(rig) {
  return rig.model.read(localsOf(rig), rig.state);
}

// Every part posed from its parent joint's pose, parents first: a nested
// node takes its local as it stands, any other is placed where its parent
// joint's frame and its local put it, its own scale kept.
function applyRig(rig, state) {
  const { a, b, v, s, q } = rig.scratch;
  rig.state = state;
  rig.model.pose(state).forEach(([t, r], index) => {
    const node = rig.nodes[index];
    if (rig.nested[index]) {
      node.position.fromArray(t);
      node.quaternion.fromArray(r);
      return;
    }
    frameMatrix(rig, rig.frames[index], a);
    a.multiply(b.compose(v.fromArray(t), q.fromArray(r), rig.scratch.one));
    node.parent.updateWorldMatrix(true, false);
    b.copy(node.parent.matrixWorld).invert().multiply(a).decompose(v, q, s);
    node.position.copy(v);
    node.quaternion.copy(q);
  });
}

// World <-> the space joint *joint*'s chain is solved in (its root joint's
// parent), in the record's metres as the file holds that space.
function toRig(rig, joint, point) {
  const space = rig.spaces[joint];
  space.updateWorldMatrix(true, false);
  return space.worldToLocal(point.clone()).toArray();
}

function spaceRotation(THREE, rig, joint) {
  const space = rig.spaces[joint];
  space.updateWorldMatrix(true, false);
  const q = new THREE.Quaternion();
  space.matrixWorld.decompose(new THREE.Vector3(), q, new THREE.Vector3());
  return q;
}

function handlerFor(THREE, session, rig, entry) {
  return {
    begin({ source, point }) {
      rig.state = readRig(rig);
      const local = rig.model.toLocal(rig.state, entry.joint, toRig(rig, entry.joint, point));
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
        const s = spaceRotation(THREE, rig, hold.joint);
        const turn = s.clone().invert().multiply(rotation).multiply(s);
        wanted = qmul(turn.toArray(), hold.rotation);
      }
      const state = rig.model.solve(rig.state, hold.joint, hold.local, toRig(rig, hold.joint, point), wanted);
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
  // A slide is in metres: the record is in glTF's frame.
  return `${(value * 100).toFixed(1)} cm`;
}

function buildPanel(viewer, session, rig) {
  // A section of the Rig window (`viewer.window`, Rig ▸ Pose rigs), one per
  // rig, opened with the model on the desktop: the sliders are how a rig is
  // posed by hand.
  const rigWindow = viewer.window('Rig', {
    title: "Each rig's joints as sliders, to pose it by hand",
    label: 'Pose rigs',
  });
  const panel = rigWindow.section(`Articulated rig: ${rig.record.name}`);
  const reach = rigReach(rig);
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
  // On each joint's anchor -- the point it turns about and the axes its
  // channels turn and slide along (its pivot and rest axes, for the
  // identity) -- in the joint's frame, which its node carries.
  const size = 0.12 * reach;
  rig.nodes.forEach((node, index) => {
    const helper = new viewer.THREE.AxesHelper(size);
    const anchor = rig.model.joints[index].anchor;
    if (anchor) {
      helper.position.fromArray(anchor[0]);
      helper.quaternion.fromArray(anchor[1]);
    }
    helper.visible = false;
    node.add(helper);
    rig.helpers.push(helper);
  });
}

// The rig's size: every joint's pivot from its parent's (a root's from its
// space), pivot to pivot (the model's bones, a flat model's from its
// anchors) -- the slide range a slider without limits spans and the joint
// axes' size. 1 m for a rig with no length at all.
function rigReach(rig) {
  return rig.model.joints.reduce((sum, _joint, index) => sum + norm(rig.model._bone(index)), 0) || 1;
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
  const record = readRecord(gltf.parser);
  if (!record || !Array.isArray(record.rigs) || !record.rigs.length) return null;
  const json = gltf.parser.json;
  const resolve = viewer.nodeResolver(gltf, model);
  const nameOf = (index) => viewer.nodeName(gltf, index);
  const rigs = record.rigs.filter((rig) => rig && typeof rig === 'object');
  // Every rig's joints as ONE assignment over every node: two rigs' joints
  // never bind one node.
  const joints = rigs.map((rig) => (Array.isArray(rig.joints) ? rig.joints : []));
  const resolved = resolveIndexed(indexNodes(json, { meshes: false }), joints.flat());
  const session = { rigs: [], unregister: [], dirty: false, sinceRefresh: 0, wasPlaying: viewer.playing };
  let at = 0;
  let outOfScope = 0;
  rigs.forEach((rigRecord, i) => {
    const found = resolved.slice(at, at + joints[i].length);
    at += joints[i].length;
    let rig = null;
    try {
      rig = bindRig(THREE, json, rigRecord, found, resolve, nameOf);
    } catch (error) {
      console.warn(`articulated_rig: ${rigRecord?.name}: not readable`, error);
    }
    if (rig === 'absent') outOfScope += 1;
    if (!rig || rig === 'absent') return;
    for (const entry of rig.grab) {
      session.unregister.push(viewer.grab.register(entry.node, handlerFor(THREE, session, rig, entry)));
    }
    buildPanel(viewer, session, rig);
    session.rigs.push(rig);
  });
  if (outOfScope) {
    console.info(`articulated_rig: ${outOfScope} rig(s) the record covers are not in this file (an exported subset)`);
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
      console.warn('articulated_rig: could not read the articulation record', error);
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
