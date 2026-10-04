// The deliverable's reflection probe: the room its bake lit, captured by the
// bake as an HDR from one point (MeshConvert._embed_lightmap_probe ->
// `extras.lightmap_web.probe`) -- what a lightmap cannot hold.
//
// A lightmap is diffuse irradiance with no direction in it, so a baked METAL,
// which has no diffuse, showed only what the page's studio environment gave
// it, and every unbaked object took that studio whole. Measured on a
// production soldering table against Arnold's render of the same room
// (2026-10-03): its metal button housing at 0.12 of Arnold's luminance, its
// pedestal 0.60, its trays 0.4, while the unbaked magnifier ran up to 2.8x
// too bright. With the probe as the environment, the reflections on every
// baked surface and all the light on every unbaked one are the room's own:
// through the bake, its GLB and this page, the housing 0.62, the pedestal
// 0.78, the trays 0.89-1.07, the magnifier's parts 0.65-1.45x, the room's own
// walls and floor within 3%. What remains on metal is three.js' metal model
// against Arnold's, not the lighting (the same probe lighting both gave
// 0.8-0.9). The light-grey metal buttons read near-white in both: Arnold
// renders them so in this room too.
//
// Three details decide whether it looks right:
//
//   * Units. A lightmap holds a white card's radiance (irradiance / pi), which
//     three.js reads as irradiance; the probe holds the room's radiance, so it
//     plays at `policy.probeIntensity` (1/pi) to stay in the bake's units (see
//     DEFAULT_PROBE_INTENSITY). Played at 1, every reflection and every unbaked
//     object came out pi too bright beside the bake.
//   * Frame. The probe was captured in the deliverable's own axes, and the page
//     re-centres the model (layout) and may spin it (the turntable), so every
//     lookup is made in MODEL space through a world-to-model matrix synced each
//     frame, never along a world direction.
//   * Box projection. One probe stands for a whole room: a reflection ray is
//     followed to where it leaves the probe's box -- the baked room's bounds --
//     and the probe is read toward THAT point from where it was captured, so a
//     wall reflects where it stands rather than as if infinitely far. A probe
//     with no box (an open scene) is read as distant. Diffuse irradiance is
//     read along the normal unprojected, as runtimes do.

import * as THREE from 'three';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { materialsOf } from './model.js';
import { LIGHTMAPS } from './records.js';
import { onBeforeDraw, prefilter, readExtras } from './scene.js';

// Both edits land in three.js 0.169's `envmap_physical_pars_fragment` chunk, on
// the two lines that turn a view-space direction into the world direction the
// environment is read along. Checked at boot against the chunk this three.js
// ships: a line it no longer carries makes the edit a silent no-op -- the room
// reflected along world axes from the probe's point, wrong wherever the model
// stands anywhere but where it was captured -- so a miss is an ERROR.
const IRRADIANCE_GLSL = 'vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );';
const RADIANCE_GLSL = 'reflectVec = inverseTransformDirection( reflectVec, viewMatrix );';
for (const line of [IRRADIANCE_GLSL, RADIANCE_GLSL]) {
  if (!THREE.ShaderChunk.envmap_physical_pars_fragment.includes(line)) {
    console.error('three.js changed envmap_physical_pars_fragment: the probe lookup edit no longer applies');
  }
}

// ONE set, shared by every patched program: a load sets the probe's point and
// box, and the frame sync below keeps the matrix current, with no recompile.
const uniforms = {
  probeWorldToModel: { value: new THREE.Matrix4() },
  probePosition: { value: new THREE.Vector3() },
  probeBoxMin: { value: new THREE.Vector3() },
  probeBoxMax: { value: new THREE.Vector3() },
  probeBoxed: { value: 0 },
};

const LOOKUP_GLSL = [
  'uniform mat4 probeWorldToModel;',
  'uniform vec3 probePosition;',
  'uniform vec3 probeBoxMin;',
  'uniform vec3 probeBoxMax;',
  'uniform float probeBoxed;',
  'varying vec3 vProbeWorld;',
  // A world direction as the probe reads it: into the model's axes and, for a
  // reflection in a boxed probe, from where the ray leaves the box as seen from
  // the capture point. A point outside the box (nothing to project onto) keeps
  // the plain direction.
  'vec3 probeLookup( vec3 worldDir, bool reflection ) {',
  '  vec3 d = normalize( mat3( probeWorldToModel ) * worldDir );',
  '  if ( reflection && probeBoxed > 0.5 ) {',
  '    vec3 p = ( probeWorldToModel * vec4( vProbeWorld, 1.0 ) ).xyz;',
  '    vec3 t = max( ( probeBoxMax - p ) / d, ( probeBoxMin - p ) / d );',
  '    float hit = min( min( t.x, t.y ), t.z );',
  '    if ( hit > 0.0 ) d = normalize( p + d * hit - probePosition );',
  '  }',
  '  return d;',
  '}',
].join('\n');

const WORLD_POSITION_GLSL = [
  '#include <worldpos_vertex>',
  '{',
  '  vec4 probeWorld = vec4( transformed, 1.0 );',
  '  #ifdef USE_BATCHING',
  '    probeWorld = batchingMatrix * probeWorld;',
  '  #endif',
  '  #ifdef USE_INSTANCING',
  '    probeWorld = instanceMatrix * probeWorld;',
  '  #endif',
  '  vProbeWorld = ( modelMatrix * probeWorld ).xyz;',
  '}',
].join('\n');

// The probe's model root, its prefiltered environment and what the file says
// of it, while one is loaded; null between loads and for a deliverable
// without one. `on` is this view's switch (the Environment window): off, the
// studio lights the model and its materials are back on their own programs,
// exactly as a deliverable without a probe draws -- so a frame timed either
// way is the probe's cost, and nothing else's.
let root = null;
let target = null;
let facts = null;
let boxed = false;
let on = true;
const patched = new Set();
// material -> its own onBeforeCompile and cache key from before the probe's
// edit (undefined: none, the prototype's), what unpatchProbeShader restores.
// Not on the material's userData, which three.js clones and serializes.
const restores = new WeakMap();

// The probe's environment texture while it is up, or null -- what
// applyLighting reads to choose between it and the studio.
export function probeEnvironment() {
  return target && on ? target.texture : null;
}

// The probe's prefiltered target while one is loaded, on or off -- what the
// page holds in GPU memory for it (`viewer.environment.maps`).
export function probeTarget() {
  return target;
}

// What the loaded probe is, or null: `{on, width, height, bytes, bufferView,
// position, box, decodeMs, prefilterMs}` -- `position` and `box` (`{min, max}`,
// null when it is read as distant) in the file's metres.
export function probeState() {
  return facts ? { on, ...facts } : null;
}

// Every frame, before the draw: the model's world matrix inverted, after the
// scene graph updated it -- layout moves the model, the turntable spins it.
onBeforeDraw(() => {
  if (root && on) uniforms.probeWorldToModel.value.copy(root.matrixWorld).invert();
});

function patchProbeShader(material) {
  const own = (key) => (Object.prototype.hasOwnProperty.call(material, key) ? material[key] : undefined);
  restores.set(material, {
    onBeforeCompile: own('onBeforeCompile'),
    customProgramCacheKey: own('customProgramCacheKey'),
  });
  const previous = material.onBeforeCompile;
  const previousKey = material.customProgramCacheKey.bind(material);
  material.onBeforeCompile = (shader, renderer) => {
    if (previous) previous.call(material, shader, renderer);
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vProbeWorld;')
      .replace('#include <worldpos_vertex>', WORLD_POSITION_GLSL);
    if (!shader.fragmentShader.includes('#include <envmap_physical_pars_fragment>')) {
      console.error('three.js changed its fragment shader: the probe lookup edit no longer applies');
    }
    const chunk = THREE.ShaderChunk.envmap_physical_pars_fragment
      .replace(IRRADIANCE_GLSL, `${IRRADIANCE_GLSL}\nworldNormal = probeLookup( worldNormal, false );`)
      .replace(RADIANCE_GLSL, `${RADIANCE_GLSL}\nreflectVec = probeLookup( reflectVec, true );`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <envmap_physical_pars_fragment>', `${LOOKUP_GLSL}\n${chunk}`);
  };
  // A distinct program from the same material unpatched, or the renderer
  // would hand one of them the other's shader.
  material.customProgramCacheKey = () => `${previousKey()}|probe`;
  material.needsUpdate = true;
}

// *material* back on the program it had before the probe's edit.
function unpatchProbeShader(material) {
  const restore = restores.get(material);
  if (!restore) return;
  for (const [key, value] of Object.entries(restore)) {
    if (value === undefined) delete material[key];
    else material[key] = value;
  }
  restores.delete(material);
  material.needsUpdate = true;
}

// The probe as the switch leaves it: its point and box in the shared
// uniforms and every standard material of the model patched to read it, or
// none of that.
function applySwitch() {
  for (const material of patched) {
    if (on) {
      if (!restores.has(material)) patchProbeShader(material);
    } else {
      unpatchProbeShader(material);
    }
  }
  uniforms.probeBoxed.value = on && boxed ? 1 : 0;
  if (on && root) {
    root.updateMatrixWorld(true);
    uniforms.probeWorldToModel.value.copy(root.matrixWorld).invert();
  }
}

// Turn the loaded probe on or off for this view (see `on` above); returns the
// state it ended in -- off when nothing is loaded. The caller re-applies the
// lighting (applyLighting), which puts the probe or the studio up.
export function setProbeOn(state) {
  const next = Boolean(state) && Boolean(target);
  if (next !== on) {
    on = next;
    applySwitch();
  }
  return on;
}

// The model on screen is going: its probe goes with it, and the studio comes
// back on the next applyLighting. The next probe starts on: each load shows
// the deliverable as it ships.
export function releaseProbe() {
  for (const material of patched) unpatchProbeShader(material);
  patched.clear();
  if (target) target.dispose();
  target = null;
  root = null;
  facts = null;
  boxed = false;
  on = true;
  uniforms.probeBoxed.value = 0;
}

// Load the deliverable's probe, if it carries one, as the environment for the
// model (`gltf.scene`). Resolves to the loaded probe or null; `issues` collects
// why one the file names did not load (see measureModel). Async -- the
// Radiance bytes come out of the file's own buffer -- so the caller must
// recheck that its load is still the current one before the probe goes up
// (see load.js). The decode and the prefilter are timed apart: the first is
// the CPU's, the second the GPU's, and a headset pays both inside the load.
export async function loadProbe(gltf, issues = []) {
  const entry = readExtras(gltf, LIGHTMAPS.webKey)?.probe;
  if (!entry || !Number.isInteger(entry.bufferView)) return null;
  const decoding = performance.now();
  let parsed;
  try {
    const bytes = await gltf.parser.getDependency('bufferView', entry.bufferView);
    parsed = new RGBELoader().setDataType(THREE.HalfFloatType).parse(bytes);
  } catch (error) {
    issues.push({ level: 'warning', text: `reflection probe not loaded: ${error.message || error}` });
    return null;
  }
  const prefiltering = performance.now();
  const equirect = new THREE.DataTexture(
    parsed.data, parsed.width, parsed.height, THREE.RGBAFormat, parsed.type);
  equirect.mapping = THREE.EquirectangularReflectionMapping;
  equirect.colorSpace = THREE.LinearSRGBColorSpace;
  equirect.minFilter = THREE.LinearFilter;
  equirect.magFilter = THREE.LinearFilter;
  equirect.generateMipmaps = false;
  // Radiance rows run top-down; three.js' equirect reads its first row as the
  // nadir, as RGBELoader's own textures are flipped for.
  equirect.flipY = true;
  equirect.needsUpdate = true;
  const prefiltered = prefilter((pmrem) => pmrem.fromEquirectangular(equirect));
  equirect.dispose();
  const done = performance.now();
  return {
    prefiltered,
    position: entry.position,
    box: entry.box || null,
    width: parsed.width,
    height: parsed.height,
    bytes: gltf.parser.json.bufferViews?.[entry.bufferView]?.byteLength ?? 0,
    bufferView: entry.bufferView,
    decodeMs: prefiltering - decoding,
    prefilterMs: done - prefiltering,
  };
}

// Put a loaded probe (loadProbe) up for the model `root`: its point and box
// into the shared uniforms, and every standard material of the model patched
// to read it.
export function useProbe(probe, model) {
  releaseProbe();
  const { prefiltered, ...rest } = probe;
  target = prefiltered;
  root = model;
  facts = rest;
  const [x, y, z] = probe.position;
  uniforms.probePosition.value.set(x, y, z);
  boxed = Boolean(probe.box);
  if (boxed) {
    uniforms.probeBoxMin.value.fromArray(probe.box.min);
    uniforms.probeBoxMax.value.fromArray(probe.box.max);
  }
  model.traverse((node) => {
    for (const material of materialsOf(node)) {
      if (material.isMeshStandardMaterial) patched.add(material);
    }
  });
  applySwitch();
  return patched.size;
}
