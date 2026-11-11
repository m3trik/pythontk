// LOD groups: a GLB states each in MSFT_lod (pythontk's GlbLods), which
// three's GLTFLoader does not read -- it draws LOD0 alone, the spec's fallback.
// This builds a THREE.LOD per group so the page switches levels as the
// camera moves, the way an engine that reads the extension does.
//
// MSFT_lod sits on the LOD0 node: `extensions.MSFT_lod.ids` names the lower
// levels (kept out of the node tree), `extras.MSFT_screencoverage` holds one
// screen-relative height per level, the cull last. A level is drawn while the
// object stands at least that tall on the screen; THREE.LOD switches by
// camera DISTANCE, so each height becomes the distance at which the object is
// that tall: d = size / (2 h tan(fov / 2)), at the field of view of the camera
// that renders the frame -- read off its projection each time the LOD updates
// (a file's start camera, the desktop view and a headset's eyes each set their
// own), the distances retuned only when it changes.

import * as THREE from 'three';
import { camera } from './scene.js';

// The user data mark on each level's root: its index in its group. A reader
// counting what is drawn (specs.js) counts level 0 alone.
export const LOD_LEVEL = 'lodLevel';

// Whether *node* hangs under a group's lower level (one not drawn up close).
export function underLowerLevel(node) {
  for (let walk = node; walk; walk = walk.parent) {
    if ((walk.userData?.[LOD_LEVEL] ?? 0) > 0) return true;
  }
  return false;
}

// The vertical field of view (radians) a perspective camera renders with: off
// its projection, which every camera the renderer hands a LOD holds (a
// headset's eyes carry no meaningful `fov`); an array camera's first eye.
function fovOf(view) {
  const eye = view?.cameras?.length ? view.cameras[0] : view;
  const e = eye?.projectionMatrix?.elements;
  return e && e[5] > 0 ? 2 * Math.atan(1 / e[5]) : THREE.MathUtils.degToRad(eye?.fov || 50);
}

// The distance at which *size* (metres) stands *height* (0..1) of the screen.
function distanceFor(size, height, fov) {
  return size / (2 * height * Math.tan(fov / 2));
}

// Each level's switch distance at *fov*: level k (k > 0) from where the object
// stands as tall as level k-1 is drawn down to, an empty last level at the cull.
function retune(lod, fov) {
  const { lodSize: size, lodSwitches: switches } = lod.userData;
  lod.levels.forEach((entry, level) => {
    if (level > 0) entry.distance = distanceFor(size, switches[level], fov);
  });
  lod.userData.lodFov = fov;
}

// Build a THREE.LOD for every MSFT_lod group of *gltf* in place; returns how
// many groups were built. A level the parser cannot produce leaves its group
// as the loader drew it (LOD0 alone).
export async function bindLods(gltf) {
  const json = gltf.parser?.json;
  const nodes = json?.nodes || [];
  let built = 0;
  for (const [index, node] of nodes.entries()) {
    const ids = node.extensions?.MSFT_lod?.ids;
    if (!Array.isArray(ids) || !ids.length) continue;
    const lod0 = await gltf.parser.getDependency('node', index);
    const parent = lod0?.parent;
    if (!parent) continue;
    let levels;
    try {
      levels = [lod0, ...(await Promise.all(ids.map((id) => gltf.parser.getDependency('node', id))))];
    } catch (error) {
      console.warn(`LOD group ${node.name || index}: a level did not load (${error.message || error}); LOD0 alone`);
      continue;
    }
    const coverage = node.extras?.MSFT_screencoverage;
    const heights = Array.isArray(coverage) && coverage.length === levels.length ? coverage : null;
    const box = new THREE.Box3().setFromObject(lod0);
    const size = box.isEmpty() ? 1 : Math.max(...box.getSize(new THREE.Vector3()).toArray());
    // Where the object stands, in LOD0's own frame: a DCC's group sits at
    // identity and its levels carry the transform, so the group's origin can
    // be anywhere -- THREE.LOD measures from its own, so each update measures
    // from here instead (moving with LOD0 when it is animated).
    const centre = lod0.worldToLocal(box.isEmpty() ? lod0.getWorldPosition(new THREE.Vector3()) : box.getCenter(new THREE.Vector3()));
    const lod = new THREE.LOD();
    lod.name = `${node.name || 'node'}_lod`;
    // At LOD0's place in the tree; the levels keep their own transforms (each
    // replaces LOD0's node in MSFT_lod, so they share its parent).
    parent.add(lod);
    // The screen height each level takes over below: level k once the object
    // is shorter than level k-1 is drawn down to (the default ladder halves
    // without stated heights); LOD0 from up close.
    const switches = levels.map((_, level) => (level === 0 ? null : heights ? heights[level - 1] : 0.5 ** level));
    const fov = fovOf(camera);
    for (const [level, object] of levels.entries()) {
      object.userData[LOD_LEVEL] = level;
      lod.addLevel(object, level === 0 ? 0 : distanceFor(size, switches[level], fov));
    }
    if (heights && heights[heights.length - 1] > 0) {
      // Below the cull height nothing is drawn: an empty last level.
      switches.push(heights[heights.length - 1]);
      lod.addLevel(new THREE.Object3D(), distanceFor(size, heights[heights.length - 1], fov));
    }
    Object.assign(lod.userData, { lodSize: size, lodSwitches: switches, lodFov: fov });
    const update = lod.update.bind(lod);
    const at = new THREE.Vector3();
    lod.update = (view) => {
      const now = fovOf(view);
      if (Math.abs(now - lod.userData.lodFov) > 1e-6) retune(lod, now);
      // THREE.LOD reads its distance off its world matrix's translation: lend
      // it the object's centre for the call, then give its own back.
      const e = lod.matrixWorld.elements;
      const own = [e[12], e[13], e[14]];
      lod0.localToWorld(at.copy(centre));
      [e[12], e[13], e[14]] = [at.x, at.y, at.z];
      try {
        update(view);
      } finally {
        [e[12], e[13], e[14]] = own;
      }
    };
    built += 1;
  }
  return built;
}
