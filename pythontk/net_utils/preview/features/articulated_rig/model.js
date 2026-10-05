/*
  pythontk's `ArticulationModel` (geo_utils/articulation/model.py), ported
  operation for operation: one rig's joints, channels and limits, forward
  kinematics, reading a state back from local transforms, and the grab solver.
  Plain arrays in and out, no three.js -- the articulated_rig feature poses
  the page's nodes with it, and an app on any framework can vendor this file
  with kernel/math.js. Pinned against the reference by
  `ptk.Conformance.cases("articulation")` in test_articulated_rig_web.py:
  change a constant or a step there, and here, together.
*/

import {
  IDENTITY, ORIGIN, TO_DEG, add, cross, dot, norm, qaxis, qconj, qmul, qnormalize, qrot, scale, sub,
} from '../../kernel/math.js';

export const ROTATE_CHANNELS = ['rx', 'ry', 'rz'];
export const TRANSLATE_CHANNELS = ['tx', 'ty', 'tz'];
export const ROTATE_ORDERS = ['xyz', 'yzx', 'zxy', 'xzy', 'yxz', 'zyx'];
const CHANNELS = [...ROTATE_CHANNELS, ...TRANSLATE_CHANNELS];
const AXIS = { x: 0, y: 1, z: 2 };

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
    return /** @type {[number, number, Object<string, number>]} */ ([missing, drift, wound]);
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
  static HALVINGS = 6;
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
    let current = state.slice();
    if (!slots.length) return current;
    const damping = (ArticulationModel.DAMPING * reach) ** 2;
    const tolerance = ArticulationModel.TOLERANCE * reach;
    const step = ArticulationModel.MAX_STEP * reach;
    let [world, point, distance] = this._held(current, joint, localPoint, target);
    for (let iteration = 0; iteration < ArticulationModel.ITERATIONS; iteration += 1) {
      if (distance <= tolerance) break;
      let error = sub(target, point);
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
      // Halved until it brings the point nearer; none does -- as near as the
      // limits and the reach allow.
      let fraction = 1.0;
      let trial = null;
      let held = null;
      for (let halving = 0; halving <= ArticulationModel.HALVINGS; halving += 1) {
        trial = current.slice();
        slots.forEach((slot, i) => {
          trial[slot] = this._clampSlot(slot, current[slot] + fraction * du[i] * this._unit(slot, reach));
        });
        held = this._held(trial, joint, localPoint, target);
        if (held[2] < distance) break;
        held = null;
        fraction *= 0.5;
      }
      if (held === null) break;
      current = trial;
      [world, point, distance] = held;
    }
    return current;
  }

  _held(state, joint, localPoint, target) {
    const world = this.world(state);
    const [p, q] = world[joint];
    const point = add(p, qrot(q, localPoint));
    return /** @type {[any[], number[], number]} */ ([world, point, norm(sub(target, point))]);
  }
}
