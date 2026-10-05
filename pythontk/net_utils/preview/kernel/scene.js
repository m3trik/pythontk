// The stage every model is shown on: the renderer and its canvas, the scene
// with its environment and key light, the floor grid, the desktop camera and
// its orbit, and the pivot each model hangs off. Also the rendering policy --
// the lighting recipe a deliverable publishes, and the fallbacks it is read
// against -- since the lights it drives live here.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

export const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.xr.enabled = true;
document.body.appendChild(renderer.domElement);

export const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15171a);

// Work that must land between the scene graph's update and the draw -- a
// matrix copied off the model after layout moved it or the turntable spun it
// -- run once per draw, in the order registered. three.js calls the scene's
// `onBeforeRender` there; one list rather than a chain of wrappers, each
// module saving and calling the last.
const beforeDraw = [];
scene.onBeforeRender = (...args) => {
  for (const fn of beforeDraw) fn(...args);
};

export function onBeforeDraw(fn) {
  beforeDraw.push(fn);
}

// An environment map prefiltered for image-based lighting (PMREM), by a
// generator made for the one job and freed after it. A kept generator keeps
// its scratch target -- a second map the size of the first (6.3 MB of GPU
// memory for the studio's and a desktop probe's alike) that nothing reads
// again -- for as long as the page is open.
export function prefilter(make) {
  const generator = new THREE.PMREMGenerator(renderer);
  try {
    return make(generator);
  } finally {
    generator.dispose();
  }
}

// The studio: the main light for everything that is not baked, and the
// reflections for everything that is -- unless the deliverable carries its
// own reflection probe, the room its bake lit (probe.js), which then stands
// in for it. On the SCENE, never a material: what a baked material takes from
// it is decided in its shader, not by a per-material intensity (see
// patchBakedShader). Built when first asked for and freed while a probe
// lights the model (see applyLighting): its 6.3 MB held in GPU memory beside
// a probe lit nothing.
let studio = null;

export function studioEnvironment() {
  if (!studio) {
    const room = new RoomEnvironment();
    studio = prefilter((pmrem) => pmrem.fromScene(room, 0.04));
    room.dispose();
  }
  return studio.texture;
}

// The studio's prefiltered target while it is held, else null -- what the
// page has in GPU memory for it (`viewer.environment.maps`).
export function studioTarget() {
  return studio;
}

export function releaseStudio() {
  if (!studio) return;
  if (scene.environment === studio.texture) scene.environment = null;
  studio.dispose();
  studio = null;
}

scene.environment = studioEnvironment();

// The RoomEnvironment IBL above is the main light. Piling a bright hemisphere
// and key light on top of it (as this once did, at 1.6 and 2.2) drives every
// surface to clipping, and a blown-out surface cannot show emissive at all:
// the emissive term is *added* to the lit result, so once the diffuse is
// already past white it contributes nothing visible. Measured before the fix:
// a material with emissiveFactor [1,0,0] rendered (247,242,243) — a red bias
// of five 8-bit levels, indistinguishable from white. One modest key light for
// form definition is enough alongside the environment.
export const DEFAULT_KEY_INTENSITY = 0.9;
// The environment level, for the whole model, and the exposure the look was
// approved through. Declared as named constants for the same reason as the
// one above: they are the FALLBACKS the published policy is read against (see
// readRenderingPolicy), and a fallback that only exists inline is one a test
// cannot compare with what the deliverable actually ships.
export const DEFAULT_ENV_INTENSITY = 1;
export const DEFAULT_TONE_EXPOSURE = 1;
// How strongly a BAKED material reflects that environment (its specular term;
// the diffuse is its bake -- see patchBakedShader). The export chooses it, as
// `lightmappedMaterials.envMapIntensity`; this is the recipe's own default.
// Full, because each texel's share is already scaled by its bake
// (`reflectionNormalization`, BAKED_ENV_GLSL).
export const DEFAULT_BAKED_ENV_INTENSITY = 1.0;
// The level a deliverable's own reflection probe plays at (probe.js). Not a
// look but a unit: the probe holds the room's radiance, while a lightmap
// holds a white card's (irradiance / pi) and three.js reads it as
// irradiance, so the page draws every baked surface pi below physical; the
// probe at 1/pi stays in the bake's units, which is what makes its
// reflections, and the light it puts on unbaked objects, agree with the bake.
export const DEFAULT_PROBE_INTENSITY = 0.3183099;

// The rig this page renders with. The values above are what it FALLS BACK to;
// the live ones come from the deliverable itself, which publishes the whole
// recipe as `handoff.rendering` inside its own scene sidecar. That direction
// matters: the asset carries no lights, so the policy is part of the hand-off,
// and while the viewer defined its own numbers the published recipe could only
// be kept honest by a test pinning the two spellings together. Now the file is
// the source and this page is one of its readers.
export const POLICY_FALLBACK = Object.freeze({
  keyIntensity: DEFAULT_KEY_INTENSITY,
  environmentIntensity: DEFAULT_ENV_INTENSITY,
  toneMappingExposure: DEFAULT_TONE_EXPOSURE,
  bakedEnvIntensity: DEFAULT_BAKED_ENV_INTENSITY,
  probeIntensity: DEFAULT_PROBE_INTENSITY,
});
export let policy = { ...POLICY_FALLBACK };

// One `extras` block, wherever it rides. Three holders are real: the first
// scene's extras (a native DCC export writes there), the glTF ROOT's (where
// every MeshConvert applier writes), and `scene.userData` (GLTFLoader's parsed
// copy of the scene's). A value can also be a JSON string rather than an
// object, depending on the producer. This page reads three such blocks
// (`scene_sidecar`, `lightmap_web`, `animation_web`) and hand-rolling the probe
// per block is how one of them ends up missing a holder and going silently
// inert on a deliverable the others handle.
export function readExtras(gltf, key) {
  for (const holder of [
    gltf.parser?.json?.scenes?.[0]?.extras,
    gltf.parser?.json?.extras,
    gltf.scene?.userData,
  ]) {
    const raw = holder && holder[key];
    if (!raw) continue;
    if (typeof raw !== 'string') return raw;
    try { return JSON.parse(raw); } catch { /* keep looking */ }
  }
  return null;
}

// Reads `extras.scene_sidecar.handoff.rendering`, written by
// MeshConvert.apply_scene_sidecar. Every field is taken only if it is a finite
// NUMBER: the published policy is documentation as much as data, so some of its
// entries are deliberately prose (`lightMapIntensity` says "per material, from
// extras.lightmap_web"), and a viewer that assigned a string to an intensity
// would render black with nothing in the file looking wrong. An older GLB, or
// one from another producer, simply keeps the fallbacks.
export function readRenderingPolicy(gltf) {
  const published = readExtras(gltf, 'scene_sidecar')?.handoff?.rendering;
  const num = (value, fallback) =>
    (typeof value === 'number' && isFinite(value) ? value : fallback);
  // Falling back through POLICY_FALLBACK rather than through the constants
  // again: one table names what this page does without a published policy, and
  // it is the same table the initial value is taken from.
  policy = {
    keyIntensity: num(published?.keyLight?.intensity, POLICY_FALLBACK.keyIntensity),
    environmentIntensity: num(
      published?.environment?.intensity, POLICY_FALLBACK.environmentIntensity),
    toneMappingExposure: num(
      published?.renderer?.toneMappingExposure, POLICY_FALLBACK.toneMappingExposure),
    bakedEnvIntensity: num(
      published?.lightmappedMaterials?.envMapIntensity, POLICY_FALLBACK.bakedEnvIntensity),
    probeIntensity: num(
      published?.environment?.probeIntensity, POLICY_FALLBACK.probeIntensity),
  };
  renderer.toneMappingExposure = policy.toneMappingExposure;
}

export const keyLight = new THREE.DirectionalLight(0xffffff, DEFAULT_KEY_INTENSITY);
keyLight.position.set(3, 6, 4);
scene.add(keyLight);

export const GRID_SIZE = 20;  // m across, centred on the origin
export const grid = new THREE.GridHelper(GRID_SIZE, 40, 0x3a3f45, 0x25292e);
// A reference for where there is NO model, so the model always covers it: drawn
// first of everything (the opaque pass, renderOrder -1) and never into depth.
// As a transparent object it drew AFTER the model and depth-tested against it,
// so a floor in its own plane -- where a room's floor sits -- z-fought it into a
// dark 0.5 m grid, dashed at a distance (a production room, 2026-10-02). Custom
// blending keeps its translucency outside the transparent pass.
grid.material.opacity = 0.55;
grid.material.blending = THREE.CustomBlending;
grid.material.blendSrc = THREE.SrcAlphaFactor;
grid.material.blendDst = THREE.OneMinusSrcAlphaFactor;
grid.material.depthWrite = false;
grid.renderOrder = -1;
scene.add(grid);
// What a view frames when the model has nothing drawable in it (see homeView).
export const GRID_BOX = new THREE.Box3(
  new THREE.Vector3(-GRID_SIZE / 2, 0, -GRID_SIZE / 2),
  new THREE.Vector3(GRID_SIZE / 2, 0, GRID_SIZE / 2),
);

// Degrees, vertical: the desktop view's own, whenever no start camera sets one.
export const DEFAULT_FOV = 55;
export const camera = new THREE.PerspectiveCamera(DEFAULT_FOV, innerWidth / innerHeight, 0.01, 500);
camera.position.set(2.2, 1.8, 2.8);

export const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0.8, 0);

// The nearest a headset session draws, at most. The desktop view sizes its near
// plane to the model -- 0.23 m for a 150 m ground framed whole, for the depth
// precision a view from 230 m away wants -- and three.js hands `camera.near` to
// the session as its depthNear. In a session the viewer stands IN the model at
// human scale, and that plane cut away the blink, the arrival fade and the
// comfort vignette (one quad, OVERLAY_DISTANCE in front of each eye) and a
// controller brought within a hand of the face.
export const HEADSET_NEAR = 0.02;

// The desktop view -- where the orbit camera stands and looks, its lens, its
// clip planes and what it orbits -- kept aside while a headset presents, and
// put back as the session ends. three.js drives the page's camera from the
// headset meanwhile (WebXRManager.updateUserCamera writes each XR frame's head
// pose, field of view and projection into it) and restores none of it, so a
// session otherwise left the desktop view where the headset last was, at the
// headset's field of view. Null outside a session.
export let desktopView = null;

export function viewOf() {
  return {
    position: camera.position.clone(),
    quaternion: camera.quaternion.clone(),
    fov: camera.fov,
    zoom: camera.zoom,
    near: camera.near,
    far: camera.far,
    target: controls.target.clone(),
  };
}

export function applyView(view) {
  camera.position.copy(view.position);
  camera.quaternion.copy(view.quaternion);
  camera.fov = view.fov;
  camera.zoom = view.zoom;
  camera.near = view.near;
  camera.far = view.far;
  controls.target.copy(view.target);
  camera.updateProjectionMatrix();
  controls.update();
}

// The near plane a session draws with: the desktop's, down to HEADSET_NEAR.
export function holdHeadsetNear() {
  if (camera.near <= HEADSET_NEAR) return;
  camera.near = HEADSET_NEAR;
  camera.updateProjectionMatrix();
}

// The clip planes a framing chose (frameCamera, homeView), and the view as it
// then stands. Mid-session -- a push landing, the Frame button -- the framing is
// the view the session hands back at its end, and the camera keeps the
// headset's near plane meanwhile.
export function settleView(near, far) {
  camera.near = near;
  camera.far = far;
  camera.updateProjectionMatrix();
  controls.update();
  if (!desktopView) return;
  desktopView = viewOf();
  holdHeadsetNear();
}

// A headset session begins: the desktop view is kept aside for its end, and
// the session draws with a headset's near plane.
export function beginHeadsetView() {
  desktopView = viewOf();
  holdHeadsetNear();
}

// The session has ended: the desktop view is put back as it was kept.
export function endHeadsetView() {
  if (desktopView) applyView(desktopView);
  desktopView = null;
}

// Everything loaded hangs off this pivot, so swapping models and re-scaling
// never touches the lights, grid or camera rig.
export const pivot = new THREE.Group();
scene.add(pivot);
