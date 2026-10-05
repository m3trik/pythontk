import * as THREE from 'three';
import { locomotion } from './locomotion.js';
import { current, frameCamera } from './model.js';
import { rig } from './rig.js';
import { camera, controls, DEFAULT_FOV, GRID_BOX, settleView } from './scene.js';
import { surfaces } from './surfaces.js';

// Where views start. The server names a node (`manifest.userPos`, "user_pos"
// unless the DCC side chose another) -- a camera, so the start can be looked
// through in Maya or Blender before it is ever pushed. The desktop view opens
// through it (homeView), and a headset session stands under it, facing where
// it looks (stepStart). Where the viewer goes from there is locomotion's
// business; with locomotion off, they stay.

export let startNode = null;  // the current model's start node, or null
// The name the server says views start at (`manifest.userPos`); null: none.
let startName = null;

// Matched on the name the file carries rather than three.js's sanitised copy,
// which drops the ':' a Maya namespace hangs on; a namespaced one
// ("set:user_pos") answers when no node has the name exactly. Null when the
// model has none.
function findStartNode(root, name) {
  if (!root || !name) return null;
  let exact = null;
  let spaced = null;
  root.traverse((node) => {
    const own = node.userData.name ?? node.name;
    if (exact || typeof own !== 'string') return;
    if (own === name) exact = node;
    else if (!spaced && own.slice(own.lastIndexOf(':') + 1) === name) spaced = node;
  });
  return exact || spaced;
}

const viewRay = new THREE.Raycaster();

// The desktop view a load opens on: through the start camera when the model
// has one, else the whole model, framed. Through it means its position, the
// way it looks, and its HORIZONTAL field of view -- the one Maya (film gate
// fill) and Blender (sensor fit auto) both hold for a landscape frame, and the
// one the file keeps faithfully (the vertical it stores assumes the sensor's
// 3:2, not the render's aspect; the clip planes arrive in the wrong units and
// are not used). The orbit then turns about what the camera looks at. A start
// node that is not a camera places only the headset: a marker on the floor is
// no place to put an eye.
export function homeView() {
  const start = startNode?.isPerspectiveCamera ? startNode : null;
  if (!start) {
    frameCamera(DEFAULT_FOV);
    return;
  }
  const across = Math.tan(THREE.MathUtils.degToRad(start.fov) / 2) * start.aspect;
  camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(across / camera.aspect));
  current.updateWorldMatrix(true, true);
  const eye = start.getWorldPosition(new THREE.Vector3());
  const look = new THREE.Vector3(0, 0, -1).applyQuaternion(start.getWorldQuaternion(new THREE.Quaternion()));
  const box = new THREE.Box3().setFromObject(current);
  // A model with nothing drawable -- a push of the start camera alone -- is an
  // EMPTY box, and an empty box lies at Infinity from every point: the far
  // plane came out Infinity and the projection NaN, and in a session three.js
  // hands that plane to `updateRenderState`, which refuses a non-finite
  // depthFar and stops the XR loop. The page's floor grid is all there is to
  // see then, so the planes reach that.
  if (box.isEmpty()) box.copy(GRID_BOX);
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 0.05);
  viewRay.set(eye, look);
  const hit = viewRay.intersectObject(current, true).find((h) => h.object.isMesh && h.object.visible);
  const toCenter = box.getCenter(new THREE.Vector3()).sub(eye).dot(look);
  camera.position.copy(eye);
  controls.target.copy(eye).addScaledVector(look, hit ? hit.distance : Math.max(toCenter, radius / 10, 0.1));
  settleView(Math.max(radius / 2000, 0.005), (box.distanceToPoint(eye) + 2 * radius) * 2 + 1);
}

const START_LIFT = 0.05;  // m above the start node its floor is sought from

// Where a session starts: `{point, heading}`, read from the start node as it
// stands NOW (a script may have turned the model since it loaded), the point
// dropped to the floor under it -- so a camera at eye height stands the viewer
// where its feet would be -- or kept at the node's own height with nothing
// walkable below. Only where and which way: the height of the eyes, the tilt of
// the head and the field of view are the headset's, because the floor has to
// stay where the viewer's feet are. Null without a start node.
export function startPlace() {
  if (!startNode) return null;
  surfaces.refresh();
  startNode.updateWorldMatrix(true, false);
  const point = startNode.getWorldPosition(new THREE.Vector3());
  const heading = rig.headingOf(startNode.getWorldQuaternion(new THREE.Quaternion()));
  const floor = surfaces.cast(
    point.clone().setY(point.y + START_LIFT),
    point.clone().setY(Math.min(point.y, 0) - 1),
  );
  if (floor && floor.valid) point.y = floor.point.y;
  return { point, heading };
}

// Whether this session has been placed at its start (true between sessions:
// nothing is due), and the rig as that left it -- the origin, on a model with
// no start -- which is how a recenter, and locomotion switched off, tell a
// viewer still standing there from one who has moved on.
let arrived = true;
let arrivedAt = null;

// Whether the rig still stands where this session was placed.
export function atStart() {
  if (!arrivedAt) return false;
  const place = rig.place;
  return Object.keys(arrivedAt).every((key) => place[key] === arrivedAt[key]);
}

// Due back at the start on the next frame with a head -- once there has been
// one to go back to.
export function returnToStart() {
  if (arrivedAt) arrived = false;
}

// A session's first frame with a head and a model: to the start, if the model
// has one, under black. It waits for both, so a session begun while the model
// is still downloading lands at the start when it arrives, rather than never.
// Due again later (`returnToStart`), it goes back the same way: under the start
// again, facing its way, or -- with no start -- to where the rig began.
export function stepStart(input) {
  if (arrived || !input.head || !current) return [];
  arrived = true;
  const start = startPlace();
  if (start) {
    rig.arrive(start.point, start.heading, input.head);
  } else if (arrivedAt && !atStart()) {
    rig.restore(arrivedAt);
  } else {
    arrivedAt = arrivedAt || rig.place;
    return [];
  }
  arrivedAt = rig.place;
  return ['arrive'];
}

// A recenter (holding the headset's system button) resets the session's space
// under the viewer. One still standing where the start put them -- always, with
// locomotion off -- is put back there, facing its way: the viewpoint holds. One
// who has moved on keeps their place.
export function recenter() {
  if (!locomotion.enabled || atStart()) returnToStart();
}

// The current model's start node, found again -- a load's, once its model is
// laid out.
export function findStart() {
  startNode = findStartNode(current, startName);
}

// The server's start name (`manifest.userPos`). Live: a new name finds its node
// for the NEXT session without a publish, while the desktop view and a session
// under way stay where they are.
export function setStartName(name) {
  if (name === startName) return;
  startName = name;
  if (current) startNode = findStartNode(current, startName);
}

// A session begins: its start is due on its first frame with a head.
export function beginArrival() {
  arrived = false;
  arrivedAt = null;
}

// A session has ended: nothing is due.
export function endArrival() {
  arrived = true;
  arrivedAt = null;
}
