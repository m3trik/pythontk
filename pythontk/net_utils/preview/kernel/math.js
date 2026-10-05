// Vectors and quaternions on plain arrays -- `[x, y, z]` and `[x, y, z, w]` --
// with no three.js: the part every port of a pythontk model copies
// (`_ArticulationModelInternal` in Python), shared here by the features'
// models. `qmul(a, b)` is the rotation b followed by a, as in the reference.
// A step changed here is a change in every model built on it; the models'
// golden cases (`ptk.Conformance`) say so.

export const IDENTITY = Object.freeze([0, 0, 0, 1]);
export const ORIGIN = Object.freeze([0, 0, 0]);
export const TO_RAD = Math.PI / 180;
export const TO_DEG = 180 / Math.PI;

export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const norm = (a) => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);

// a (x) b: the rotation b followed by a.
export function qmul(a, b) {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}
export const qconj = (q) => [-q[0], -q[1], -q[2], q[3]];
export function qnormalize(q) {
  const n = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n < 1e-12) return IDENTITY.slice();
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}
// *v* rotated by the unit quaternion *q*.
export function qrot(q, v) {
  const u = [q[0], q[1], q[2]];
  const t = scale(cross(u, v), 2.0);
  return add(add(v, scale(t, q[3])), cross(u, t));
}
// A rotation of *degrees* about the unit axis *axis* (0, 1, 2).
export function qaxis(axis, degrees) {
  const half = degrees * TO_RAD * 0.5;
  const s = Math.sin(half);
  return [axis === 0 ? s : 0, axis === 1 ? s : 0, axis === 2 ? s : 0, Math.cos(half)];
}
