// The model on screen: the loader, the current model and its bounds, how it
// is stood on the floor, freed and framed.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { camera, controls, desktopView, pivot, renderer, settleView } from './scene.js';

export const loader = new GLTFLoader();
// KTX2/Basis support (KHR_texture_basisu) — what a KTX2 Texture File Type
// row emits (glb_options), and REQUIRED by those GLBs (they carry no
// fallback image). One transcoder serves both Basis codecs (ETC1S and UASTC);
// detectSupport picks the GPU target per device — ASTC on a standalone
// headset, BC7/DXT on desktop — which is the whole point: the texture stays
// block-compressed in GPU memory instead of decoding to RGBA. The wasm rides
// the same unpkg CDN as every import above, a dependency the boot watchdog
// already owns. Inert for WebP/PNG deliverables.
const ktx2Loader = new KTX2Loader()
  .setTranscoderPath('https://unpkg.com/three@0.169.0/examples/jsm/libs/basis/')
  .detectSupport(renderer);
loader.setKTX2Loader(ktx2Loader);
export let current = null;
export let modelBounds = null;

// `node.material` is a single material, an ARRAY of them (a multi-material mesh), or
// absent (any non-mesh in the graph). Every walk below has to handle all three, so the
// normalization lives here once — three hand-rolled copies is exactly how one of them ends
// up quietly missing the array case and skipping half a mesh's materials.
export function materialsOf(node) {
  if (Array.isArray(node.material)) return node.material;
  return node.material ? [node.material] : [];
}

export function disposeModel(root) {
  // GLTFLoader allocates GPU resources that garbage collection cannot reclaim;
  // without this an edit-and-republish loop leaks a full copy of the model's
  // buffers and textures every single push.
  root.traverse((node) => {
    if (node.geometry) node.geometry.dispose();
    for (const material of materialsOf(node)) {
      for (const value of Object.values(material)) {
        // Everything here belongs to the MODEL and dies with it. The session's
        // environment lives on the scene, never on a material -- while baked
        // materials briefly carried it as their own `envMap`, this loop needed
        // a guard or the second push rendered unlit.
        if (value && value.isTexture) value.dispose();
      }
      material.dispose();
    }
  });
}

export function layout() {
  if (!current || !modelBounds) return;
  const { size, center } = modelBounds;
  // At its true size, centred on the pivot in X and Z and stood on the floor in
  // Y. The pivot stays at the origin and carries nothing of its own, which is
  // what makes rotating it a spin rather than an orbit: `viewer.pivot` is handed
  // to every viewer script (the packaged `turntable` is exactly that rotation),
  // so the group they are given has to turn about the model.
  current.position.set(-center.x, -(center.y - size.y / 2), -center.z);
}

// *model*'s box in the PIVOT's frame -- the frame layout() places it in --
// rather than the world's. A script may have turned the pivot (the turntable
// keeps its angle across pushes, on purpose), and the world box of a turned
// model is neither its size nor, applied as the pivot-local offset above, its
// centre: measured, a push under a quarter turn stood an off-origin slab at
// [3.0, 0.5, -17.0], and the HUD quoted the size of the turned box. Measured
// detached, so the world matrices it is read through are the model's own.
export function boxInPivot(model) {
  pivot.remove(model);
  const box = new THREE.Box3().setFromObject(model);
  pivot.add(model);
  return box;
}

// The whole model, framed from the front, right and above, on the lens `fov`
// (degrees, vertical): by default the desktop view's -- mid-session the camera
// holds the headset's (see desktopView), and the framing is the one the
// session hands back. A model with nothing drawable leaves the view alone.
export function frameCamera(fov = desktopView ? desktopView.fov : camera.fov) {
  if (!current) return;
  const box = new THREE.Box3().setFromObject(current);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.length() / 2, 0.05);
  const distance = radius / Math.sin(THREE.MathUtils.degToRad(fov / 2));

  camera.fov = fov;
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(0.7, 0.5, 1).normalize().multiplyScalar(distance * 1.3));
  settleView(Math.max(distance / 1000, 0.01), distance * 100);
}

// The model on screen becomes *root*: the previous one taken off the pivot and
// freed (see disposeModel), the new one hung on it -- a load's first step once
// it owns the screen.
export function setModel(root) {
  if (current) {
    pivot.remove(current);
    disposeModel(current);
  }
  current = root;
  pivot.add(current);
}

// The current model's size and centre in the pivot's frame (see boxInPivot),
// measured once a load's passes have run: what layout() stands it by.
export function measureBounds() {
  const box = boxInPivot(current);
  modelBounds = {
    size: box.getSize(new THREE.Vector3()),
    center: box.getCenter(new THREE.Vector3()),
  };
}
