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
           grabbed part with the page's `viewer.grab`, and show the rig panel
           (a slider per channel, the joints' values, Reset, Return to
           animation, joint axes).
    grab   a hold reads the pose on screen back into the model's state (a
           playing clip included), records the held point in its joint's
           frame, and each move solves the state that brings it to the hand
           -- a held link hanging off a ball takes the hand's turn, and with a
           mouse it keeps the rotation it was taken with.
    frame  a rig a hand or a slider has posed is re-posed every frame, AFTER
           the animation mixer; it holds that pose until Return to animation,
           or until the transport starts playing again.

  THE MODEL is `ArticulationModel`, a port of pythontk's
  `geo_utils/articulation/model.py` operation for operation, pinned against it
  by `test_articulated_rig_web.py` over `ArticulationConformance`'s golden
  cases. It is exported by name and uses no three.js -- plain arrays in and
  out -- so a production WebXR app on any framework can vendor this file and
  import `{ ArticulationModel }` without the preview. Change a constant or a
  step there, and here, together.

  Activate:  automatic -- PreviewServer.AUTO_SCRIPTS turns it on for any
             deliverable whose root extras carry `articulation_web`;
             or bridge.push(scripts=["articulated_rig"]).
*/

const MANIFEST_KEY = 'articulation_web';
//: How often the panel's values refresh while it is shown, in seconds.
const PANEL_INTERVAL = 0.1;

/* ------------------------------------------------------- the model, ported --- */

export const ROTATE_CHANNELS = ['rx', 'ry', 'rz'];
export const TRANSLATE_CHANNELS = ['tx', 'ty', 'tz'];
export const ROTATE_ORDERS = ['xyz', 'yzx', 'zxy', 'xzy', 'yxz', 'zyx'];
const CHANNELS = [...ROTATE_CHANNELS, ...TRANSLATE_CHANNELS];
const AXIS = { x: 0, y: 1, z: 2 };
const IDENTITY = [0, 0, 0, 1];
const ORIGIN = [0, 0, 0];
const TO_RAD = Math.PI / 180;
const TO_DEG = 180 / Math.PI;

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm = (a) => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);

// a (x) b: the rotation b followed by a.
function qmul(a, b) {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}
const qconj = (q) => [-q[0], -q[1], -q[2], q[3]];
function qnormalize(q) {
  const n = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n < 1e-12) return IDENTITY.slice();
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}
function qrot(q, v) {
  const u = [q[0], q[1], q[2]];
  const t = scale(cross(u, v), 2.0);
  return add(add(v, scale(t, q[3])), cross(u, t));
}
function qaxis(axis, degrees) {
  const half = degrees * TO_RAD * 0.5;
  const s = Math.sin(half);
  return [axis === 0 ? s : 0, axis === 1 ? s : 0, axis === 2 ? s : 0, Math.cos(half)];
}
function eulerQuat(order, angles) {
  let q = IDENTITY;
  for (const c of order) q = qmul(qaxis(AXIS[c], angles['r' + c] ?? 0), q);
  return q;
}
function qmatrix(q) {
  const [x, y, z, w] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}
function matrixEuler(order, m) {
  const a = AXIS[order[2]];
  const b = AXIS[order[1]];
  const c = AXIS[order[0]];
  const s = (((b - a) % 3) + 3) % 3 === 1 ? 1.0 : -1.0;
  const sinB = Math.max(-1.0, Math.min(1.0, s * m[a][c]));
  const angleB = Math.asin(sinB);
  let angleA;
  let angleC;
  if (Math.cos(angleB) > 1e-6) {
    angleA = Math.atan2(-s * m[b][c], m[c][c]);
    angleC = Math.atan2(-s * m[a][b], m[a][a]);
  } else {
    angleC = 0.0;
    angleA = Math.atan2(s * m[c][b], m[b][b]);
  }
  const out = { rx: 0.0, ry: 0.0, rz: 0.0 };
  out['r' + order[2]] = angleA * TO_DEG;
  out['r' + order[1]] = angleB * TO_DEG;
  out['r' + order[0]] = angleC * TO_DEG;
  return out;
}
const unwrap = (value, reference) => value + 360.0 * Math.floor((reference - value) / 360.0 + 0.5);
function eulerFor(order, m, channels, reference) {
  const first = matrixEuler(order, m);
  const second = {};
  second['r' + order[2]] = first['r' + order[2]] + 180.0;
  second['r' + order[1]] = 180.0 - first['r' + order[1]];
  second['r' + order[0]] = first['r' + order[0]] + 180.0;
  const scored = [first, second].map((triple) => {
    const wound = {};
    for (const c of ROTATE_CHANNELS) wound[c] = unwrap(triple[c], reference[c] ?? 0.0);
    let missing = 0.0;
    let drift = 0.0;
    for (const c of ROTATE_CHANNELS) {
      if (channels.includes(c)) drift += Math.abs(wound[c] - (reference[c] ?? 0.0));
      else missing += Math.abs(unwrap(triple[c], 0.0));
    }
    return [missing, drift, wound];
  });
  const [[m1, d1, w1], [m2, d2, w2]] = scored;
  if (m2 < m1 - 1e-6 || (Math.abs(m2 - m1) <= 1e-6 && d2 < d1)) return w2;
  return w1;
}
function solve3(a, e) {
  const [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]] = a;
  const c00 = a11 * a22 - a12 * a21;
  const c01 = a12 * a20 - a10 * a22;
  const c02 = a10 * a21 - a11 * a20;
  const det = a00 * c00 + a01 * c01 + a02 * c02;
  if (Math.abs(det) < 1e-300) return ORIGIN.slice();
  const inv = 1.0 / det;
  const c10 = a02 * a21 - a01 * a22;
  const c11 = a00 * a22 - a02 * a20;
  const c12 = a01 * a20 - a00 * a21;
  const c20 = a01 * a12 - a02 * a11;
  const c21 = a02 * a10 - a00 * a12;
  const c22 = a00 * a11 - a01 * a10;
  return [
    (c00 * e[0] + c10 * e[1] + c20 * e[2]) * inv,
    (c01 * e[0] + c11 * e[1] + c21 * e[2]) * inv,
    (c02 * e[0] + c12 * e[1] + c22 * e[2]) * inv,
  ];
}

/**
 * One rig's joints, channels and limits, and the solver that poses them --
 * pythontk's `ArticulationModel`, ported. Built from one rig of the
 * `articulation` record (`{name, joints: [{name, parent, t, q, rotate_order,
 * channels: [{channel, min, max, weight}]}], grab}`); a state is one number
 * per channel, joint by joint, degrees for rotations and record units for
 * slides. Vectors are `[x, y, z]`, rotations `[x, y, z, w]`.
 */
export class ArticulationModel {
  static ITERATIONS = 32;
  static PASSES = 2;
  static DAMPING = 0.05;
  static TOLERANCE = 1.0e-4;
  static MAX_STEP = 0.25;
  static EPS = 1.0e-9;

  constructor(rig) {
    this.name = String(rig.name || '');
    this.joints = [];
    this.channels = [];
    this._limits = [];
    this._weights = [];
    this._slots = [];
    (rig.joints || []).forEach((spec, index) => this._addJoint(index, spec));
  }

  _addJoint(index, spec) {
    const name = String(spec.name || `joint${index}`);
    let parent = spec.parent;
    if (parent !== null && parent !== undefined) {
      parent = Number(parent);
      if (!(parent >= 0 && parent < index)) {
        throw new Error(`Joint ${name}: parent ${parent} is not an earlier joint.`);
      }
    } else {
      parent = null;
    }
    const order = String(spec.rotate_order || 'xyz');
    if (!ROTATE_ORDERS.includes(order)) throw new Error(`Joint ${name}: unknown rotate order ${order}.`);
    const t = (spec.t || ORIGIN).map(Number);
    const q = qnormalize((spec.q || IDENTITY).map(Number));
    const slots = [];
    const seen = new Set();
    for (const entry of spec.channels || []) {
      const channel = String(entry.channel);
      if (!CHANNELS.includes(channel)) throw new Error(`Joint ${name}: unknown channel ${channel}.`);
      if (seen.has(channel)) throw new Error(`Joint ${name}: channel ${channel} named twice.`);
      seen.add(channel);
      const lo = entry.min === null || entry.min === undefined ? null : Number(entry.min);
      const hi = entry.max === null || entry.max === undefined ? null : Number(entry.max);
      if (lo !== null && hi !== null && lo > hi) {
        throw new Error(`Joint ${name}: ${channel} limits [${lo}, ${hi}] are reversed.`);
      }
      const weight = entry.weight === null || entry.weight === undefined ? 1.0 : Number(entry.weight);
      const slot = this.channels.length;
      this.channels.push([index, channel]);
      this._limits.push([lo, hi]);
      this._weights.push(Math.max(0.0, weight));
      slots.push([slot, channel]);
    }
    this.joints.push({ name, parent, t, q, rotate_order: order });
    this._slots.push(slots);
  }

  static fromRecord(payload, name = null) {
    for (const rig of payload.rigs || []) {
      if (name === null || rig.name === name) return new ArticulationModel(rig);
    }
    throw new Error(`No articulated rig ${name} in the record.`);
  }

  jointIndex(name) {
    const index = this.joints.findIndex((joint) => joint.name === name);
    if (index < 0) throw new Error(`No joint ${name} in rig ${this.name}.`);
    return index;
  }

  chain(joint) {
    const out = [];
    for (let index = joint; index !== null; index = this.joints[index].parent) out.push(index);
    return out.reverse();
  }

  restState() { return new Array(this.channels.length).fill(0.0); }

  limits(slot) { return this._limits[slot]; }

  clamp(state) { return state.map((v, s) => this._clampSlot(s, v)); }

  _clampSlot(slot, value) {
    const [lo, hi] = this._limits[slot];
    if (lo !== null && value < lo) return lo;
    if (hi !== null && value > hi) return hi;
    return value;
  }

  _values(state, joint) {
    const out = {};
    for (const [slot, channel] of this._slots[joint]) out[channel] = state[slot];
    return out;
  }

  local(state, joint) {
    const spec = this.joints[joint];
    const values = this._values(state, joint);
    const rotation = qmul(spec.q, eulerQuat(spec.rotate_order, values));
    const slide = [values.tx ?? 0.0, values.ty ?? 0.0, values.tz ?? 0.0];
    const translation = add(spec.t, qrot(spec.q, slide));
    return [translation, rotation];
  }

  pose(state) { return this.joints.map((_j, index) => this.local(state, index)); }

  world(state) {
    const out = [];
    this.joints.forEach((spec, index) => {
      const [t, q] = this.local(state, index);
      if (spec.parent === null) {
        out.push([t, q]);
      } else {
        const [pp, pq] = out[spec.parent];
        out.push([add(pp, qrot(pq, t)), qmul(pq, q)]);
      }
    });
    return out;
  }

  point(state, joint, localPoint) {
    const [p, q] = this.world(state)[joint];
    return add(p, qrot(q, localPoint));
  }

  toLocal(state, joint, point) {
    const [p, q] = this.world(state)[joint];
    return qrot(qconj(q), sub(point, p));
  }

  read(locals, hint = null) {
    const state = hint === null ? this.restState() : hint.map(Number);
    locals.forEach(([t, q], index) => {
      const slots = this._slots[index];
      if (!slots.length) return;
      const spec = this.joints[index];
      const relative = qmul(qconj(spec.q), qnormalize(q));
      const reference = {};
      for (const [slot, channel] of slots) reference[channel] = state[slot];
      const angles = eulerFor(
        spec.rotate_order,
        qmatrix(relative),
        slots.map(([, c]) => c).filter((c) => ROTATE_CHANNELS.includes(c)),
        reference,
      );
      const slide = qrot(qconj(spec.q), sub(t, spec.t));
      for (const [slot, channel] of slots) {
        state[slot] = ROTATE_CHANNELS.includes(channel)
          ? angles[channel]
          : slide[TRANSLATE_CHANNELS.indexOf(channel)];
      }
    });
    return state;
  }

  scaleOf(locals) {
    const ratios = [];
    locals.forEach(([t], index) => {
      if (this._slots[index].some(([, c]) => TRANSLATE_CHANNELS.includes(c))) return;
      const rest = norm(this.joints[index].t);
      const got = norm(t);
      if (rest > ArticulationModel.EPS && got > ArticulationModel.EPS) ratios.push(got / rest);
    });
    if (!ratios.length) return 1.0;
    ratios.sort((a, b) => a - b);
    const mid = Math.floor(ratios.length / 2);
    return ratios.length % 2 ? ratios[mid] : 0.5 * (ratios[mid - 1] + ratios[mid]);
  }

  solve(state, joint, localPoint, target, rotation = null) {
    let current = this.clamp(state);
    const chain = this.chain(joint);
    const slots = [];
    for (const j of chain) for (const [s] of this._slots[j]) slots.push(s);
    const turns = new Set(this._slots[joint].map(([, c]) => c).filter((c) => ROTATE_CHANNELS.includes(c)));
    const wrist = rotation !== null && turns.size === 3;
    const reach = this._reach(chain, localPoint);
    if (!wrist) return this._dls(current, slots, joint, localPoint, target, reach);
    const wanted = qnormalize(rotation);
    const own = new Set(this._slots[joint].filter(([, c]) => ROTATE_CHANNELS.includes(c)).map(([s]) => s));
    const placed = slots.filter((s) => !own.has(s));
    let held = wanted;
    for (let pass = 0; pass < ArticulationModel.PASSES; pass += 1) {
      const centre = sub(target, qrot(held, localPoint));
      current = this._dls(current, placed, joint, ORIGIN, centre, reach);
      current = this._turn(current, joint, wanted);
      held = this.world(current)[joint][1];
    }
    return this._dls(current, slots, joint, localPoint, target, reach);
  }

  _reach(chain, localPoint) {
    let length = norm(localPoint);
    for (const index of chain.slice(1)) length += norm(this.joints[index].t);
    return Math.max(length, ArticulationModel.EPS);
  }

  _turn(state, joint, rotation) {
    const spec = this.joints[joint];
    const parentQ = spec.parent === null ? IDENTITY : this.world(state)[spec.parent][1];
    const frame = qmul(parentQ, spec.q);
    const local = qmul(qconj(frame), rotation);
    const out = state.slice();
    const reference = {};
    for (const [slot, channel] of this._slots[joint]) reference[channel] = out[slot];
    const angles = eulerFor(spec.rotate_order, qmatrix(local), ROTATE_CHANNELS, reference);
    for (const [slot, channel] of this._slots[joint]) {
      if (ROTATE_CHANNELS.includes(channel)) out[slot] = this._clampSlot(slot, angles[channel]);
    }
    return out;
  }

  _column(world, state, slot, point, reach) {
    const [joint, channel] = this.channels[slot];
    const spec = this.joints[joint];
    const parentQ = spec.parent === null ? IDENTITY : world[spec.parent][1];
    let frame = qmul(parentQ, spec.q);
    if (TRANSLATE_CHANNELS.includes(channel)) {
      const axis = [0, 0, 0];
      axis[TRANSLATE_CHANNELS.indexOf(channel)] = 1.0;
      return scale(qrot(frame, axis), reach);
    }
    const values = this._values(state, joint);
    const wanted = AXIS[channel[1]];
    for (const c of [...spec.rotate_order].reverse()) {
      if (AXIS[c] === wanted) {
        const axis = [0, 0, 0];
        axis[wanted] = 1.0;
        return cross(qrot(frame, axis), sub(point, world[joint][0]));
      }
      frame = qmul(frame, qaxis(AXIS[c], values['r' + c] ?? 0.0));
    }
    return ORIGIN.slice();
  }

  _unit(slot, reach) {
    return ROTATE_CHANNELS.includes(this.channels[slot][1]) ? TO_DEG : reach;
  }

  _step(columns, weights, error, damping) {
    const a = [[damping, 0, 0], [0, damping, 0], [0, 0, damping]];
    columns.forEach((col, i) => {
      const w = weights[i];
      if (w <= 0.0) return;
      for (let r = 0; r < 3; r += 1) for (let c = 0; c < 3; c += 1) a[r][c] += w * col[r] * col[c];
    });
    const y = solve3(a, error);
    return columns.map((col, i) => weights[i] * dot(col, y));
  }

  _dls(state, slots, joint, localPoint, target, reach) {
    const current = state.slice();
    if (!slots.length) return current;
    const damping = (ArticulationModel.DAMPING * reach) ** 2;
    const tolerance = ArticulationModel.TOLERANCE * reach;
    const step = ArticulationModel.MAX_STEP * reach;
    for (let iteration = 0; iteration < ArticulationModel.ITERATIONS; iteration += 1) {
      const world = this.world(current);
      const [p, q] = world[joint];
      const point = add(p, qrot(q, localPoint));
      let error = sub(target, point);
      const distance = norm(error);
      if (distance <= tolerance) break;
      if (distance > step) error = scale(error, step / distance);
      const columns = slots.map((s) => this._column(world, current, s, point, reach));
      const weights = slots.map((s) => this._weights[s]);
      let du = this._step(columns, weights, error, damping);
      let blocked = false;
      slots.forEach((slot, i) => {
        const [lo, hi] = this._limits[slot];
        const value = current[slot];
        if ((lo !== null && value <= lo && du[i] < 0.0) || (hi !== null && value >= hi && du[i] > 0.0)) {
          weights[i] = 0.0;
          blocked = true;
        }
      });
      if (blocked) du = this._step(columns, weights, error, damping);
      slots.forEach((slot, i) => {
        current[slot] = this._clampSlot(slot, current[slot] + du[i] * this._unit(slot, reach));
      });
    }
    return current;
  }
}

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
  const panel = viewer.addPanel(`Articulated rig: ${rig.record.name}`);
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
  panel.show(!viewer.renderer.xr.isPresenting);
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
    rig.panel?.show(false);
    rig.panel?.element.remove();
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
