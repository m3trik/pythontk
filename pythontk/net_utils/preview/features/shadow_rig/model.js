/*
  pythontk's `ShadowProjection.model`, `ShadowModel.placement`,
  `ShadowProjection.fade` and `ShadowProjection.far_point`
  (geo_utils/shadow_projection.py), ported term for term for up = Y: where a
  shadow plane goes, from the source and the target's bounding cylinder, and
  how strongly it draws. Plain arrays in and out, no three.js -- the
  shadow_rig feature places and fades its planes with it every frame. Pinned
  against the reference by `ptk.Conformance.cases("shadow_projection")` in
  test_shadow_web.py: change a constant or a term there, and here, together.
*/

//: ShadowProjection.OVERHEAD_BEARING / FAR_FACTOR / EPS, verbatim.
export const OVERHEAD_BEARING = [0, 1];
export const FAR_FACTOR = 1.0e6;
export const EPS = 1.0e-6;
//: ShadowProjection.DEFAULT_FALLOFF_POWER, verbatim: the fade's exponent on
//: the stretch where a plane states none.
export const DEFAULT_FALLOFF_POWER = 1.2;

// ShadowProjection.model for up = Y, term for term: the bounding cylinder's
// base and top disks project through the light at k = (L - G) / (L - h); the
// reach is capped at maxStretch heights and no factor exceeds 1 + maxStretch.
export function shadowModel(contact, light, ground, radius, height, maxStretch) {
  const [cx, cy, cz] = contact;
  const [lx, ly, lz] = light;
  const dx = cx - lx;
  const dz = cz - lz;
  const dist = Math.hypot(dx, dz);
  let bearing = OVERHEAD_BEARING;
  let overhead = true;
  if (dist > EPS) {
    bearing = [dx / dist, dz / dist];
    overhead = false;
  }
  const kMax = 1 + maxStretch;
  const kBase = Math.min(Math.max((ly - ground) / Math.max(1e-4, ly - cy), 0), kMax);
  const kCap = Math.min(kMax, kBase + (maxStretch * height) / Math.max(dist, EPS));
  const kTop = Math.min(Math.max((ly - ground) / Math.max(1e-4, ly - cy - height), 0), kCap);
  const reach = Math.max(0, dist * (kTop - kBase));
  const width = 2 * radius * Math.max(kTop, kBase);
  return {
    anchor: [lx + dx * kBase, lz + dz * kBase],
    bearing,
    kBase,
    kTop,
    reach,
    base: radius * kBase,
    top: radius * kTop,
    width,
    overhead,
  };
}

// ShadowModel.placement: the canvas fractions (far edge in top radii from
// the head, near edge as a fraction of the far edge -- so the anchor, a
// grounded target's feet, keeps its place in the texture -- sides as
// fractions of the width) to the quad's centre and its extents along and
// across the bearing.
export function placement(model, fractions) {
  const [u0, u1, w0, w1] = fractions;
  const uHi = Math.max(0, model.reach + u1 * model.top);
  const uLo = u0 * uHi;
  const wLo = w0 * model.width;
  const wHi = w1 * model.width;
  const cu = 0.5 * (uLo + uHi);
  const cw = 0.5 * (wLo + wHi);
  const [ux, uz] = model.bearing;
  const wx = uz;
  const wz = -ux;
  return {
    centre: [model.anchor[0] + ux * cu + wx * cw, model.anchor[1] + uz * cu + wz * cw],
    along: uHi - uLo,
    across: wHi - wLo,
  };
}

// ShadowProjection.fade for up = Y, term for term, off the model `shadowModel`
// made of the same contact and light (a sun's far point): the shadow lightens
// as it stretches past its footprint, ramps in over *sourceRamp* metres (the
// plane's `source_ramp` times the record's unit_scale: the ramp its producer
// previews) as the source rises over the contact, and fades out as the
// target rises *fadeHeight* off the ground; every length in metres. The
// record holds the ramp over 0; the 1e-4 floor guards a file that does not.
// 0 to 1: a plane draws at its shadow times this.
export function shadowFade(model, contact, light, ground, radius, intensity, falloffPower, fadeHeight, sourceRamp) {
  const length = model.reach + model.base + model.top;
  const stretch = Math.max(1, length / Math.max(1e-4, 2 * radius));
  const dist = intensity / Math.max(0.001, stretch ** falloffPower);
  const source = Math.min(Math.max((light[1] - contact[1]) / Math.max(1e-4, sourceRamp), 0), 1);
  const rise = Math.min(Math.max(1 - Math.max(0, contact[1] - ground) / Math.max(0.001, fadeHeight), 0), 1);
  return Math.min(Math.max(dist * source * rise, 0), 1);
}

// ShadowProjection.far_point: a directional source written as a point a long
// way back along the way it shines, so the one model body serves both.
export function farPoint(contact, direction, scale) {
  const far = FAR_FACTOR * Math.max(scale, 1e-3);
  return [
    contact[0] - direction[0] * far,
    contact[1] - direction[1] * far,
    contact[2] - direction[2] * far,
  ];
}
