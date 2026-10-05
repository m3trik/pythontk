import * as THREE from 'three';

// Where the headset stands in the scene -- the one thing every headset
// behaviour moves. The viewer moves, never the scene: the tracked space is
// re-anchored through an offset reference space, so the headset, the
// controllers and every camera three.js derives from them move as one while
// the model stays put (and the desktop view, which three.js drives from the
// headset meanwhile, is put back as the session ends -- see desktopView).
// `place` is that anchor -- a scene point is R(yaw) * tracked + (x, y, z) --
// so `y` is the height of the floor under the viewer: 0 on the page's floor, a
// platform's top after a jump onto it.
//
// The rig decides nothing. Where a session begins is the start's to say (the
// start section), and getting around is locomotion's, which is optional:
// switched off, the viewer stays where the session started. Both move it
// through the calls below, and a cut -- a session's first place, a jump, a
// return to the start -- goes through the blink, to black and back, so it is
// never seen.
//
// Poses are plain values in the TRACKED space, as the headset reports them:
// `{position: [x, y, z], orientation: [x, y, z, w]}`.
const BLINK_OUT = 0.08;       // s to black before a jump ...
const BLINK_IN = 0.14;        // ... and back after it

export const rig = (() => {
  const UP = new THREE.Vector3(0, 1, 0);
  const FORWARD = new THREE.Vector3(0, 0, -1);
  const place = { x: 0, y: 0, z: 0, yaw: 0 };
  let blink = null;                      // {phase: 'out' | 'in', t, to}
  let fade = 0;
  let changed = false;

  function toScene(position) {
    const c = Math.cos(place.yaw), s = Math.sin(place.yaw);
    const [x, y, z] = position;
    return new THREE.Vector3(c * x + s * z + place.x, y + place.y, -s * x + c * z + place.z);
  }

  function directionToScene(orientation) {
    return FORWARD.clone()
      .applyQuaternion(new THREE.Quaternion().fromArray(orientation))
      .applyAxisAngle(UP, place.yaw);
  }

  // A tracked orientation as a scene one: the rig's yaw after it, as
  // `directionToScene` applies it to one axis.
  function orientationToScene(orientation) {
    return new THREE.Quaternion()
      .setFromAxisAngle(UP, place.yaw)
      .multiply(new THREE.Quaternion().fromArray(orientation));
  }

  // The way a rotation faces, as a turn about +Y from -Z: its view axis
  // flattened -- or, looking straight down (or up), the way the top of its view
  // points. That second case is not only a camera aimed at the floor: a Blender
  // Empty arrives turned -90 degrees about X, its -Z pointing down and its +Y,
  // Blender's forward, pointing the way it faces.
  function headingOf(quaternion) {
    for (const axis of [FORWARD, UP]) {
      const d = axis.clone().applyQuaternion(quaternion);
      if (d.x * d.x + d.z * d.z > 0.01) return Math.atan2(-d.x, -d.z);
    }
    return 0;
  }

  // Turn by `angle` about the vertical through `about` -- the head, so the
  // view pivots where the viewer stands instead of swinging them round.
  function turn(angle, about) {
    const c = Math.cos(angle), s = Math.sin(angle);
    const dx = place.x - about.x, dz = place.z - about.z;
    place.x = about.x + c * dx + s * dz;
    place.z = about.z - s * dx + c * dz;
    place.yaw += angle;
    changed = true;
  }

  // Move across the floor by (dx, dz): a walk.
  function move(dx, dz) {
    place.x += dx;
    place.z += dz;
    changed = true;
  }

  // Stand on a floor `y` high: a walk over a step.
  function setFloor(y) {
    place.y = y;
    changed = true;
  }

  // Put the head -- a scene point, or null for the rig's own origin -- over
  // `point`, and the floor at its height; the heading holds.
  function land(point, head) {
    place.x += point.x - (head ? head.x : place.x);
    place.z += point.z - (head ? head.z : place.z);
    place.y = point.y;
    changed = true;
  }

  // The back half of a blink, from black: the frame being drawn was posed from
  // where the viewer was (three.js poses it before the loop runs), so a cut
  // made now goes out black and the view fades in where it lands.
  function blinkIn() {
    blink = { phase: 'in', t: 0 };
    fade = 1;
  }

  // Stand the head (`pose`) over `point`, facing `heading`, from black.
  function arrive(point, heading, pose) {
    const facing = place.yaw + headingOf(new THREE.Quaternion().fromArray(pose.orientation));
    turn(heading - facing, toScene(pose.position));
    land(point, toScene(pose.position));
    blinkIn();
  }

  // Stand the rig back at `at`, a `place` it had, from black.
  function restore(at) {
    Object.assign(place, { x: at.x, y: at.y, z: at.z, yaw: at.yaw });
    changed = true;
    blinkIn();
  }

  // Jump the head over `point`, the heading kept, through a whole blink: the
  // move waits for black, since a visible cut is what makes a teleport jarring.
  // Refused (false) while a blink is under way.
  function jump(point) {
    if (blink) return false;
    blink = { phase: 'out', t: 0, to: point.clone() };
    return true;
  }

  /**
   * Advance the blink a frame, landing a jump -- over wherever the head
   * (`pose`) is by then -- once the view is black. Returns ['land'] on the
   * frame a jump lands.
   */
  function step(delta, pose) {
    if (!blink) return [];
    const events = [];
    // A hitch (a tab regaining focus, a long collection) must not become a leap.
    blink.t += Math.min(Math.max(delta, 0), 0.1);
    if (blink.phase === 'out') {
      fade = Math.min(blink.t / BLINK_OUT, 1);
      if (blink.t >= BLINK_OUT) {
        land(blink.to, pose ? toScene(pose.position) : null);
        events.push('land');
        blink = { phase: 'in', t: 0 };
      }
    } else {
      fade = Math.max(1 - blink.t / BLINK_IN, 0);
      if (blink.t >= BLINK_IN) {
        blink = null;
        fade = 0;
      }
    }
    return events;
  }

  // The rig as the XRRigidTransform an offset reference space is built from.
  // WebXR places the new space's origin at the transform, so poses in it are
  // the transform's INVERSE applied to tracked ones -- and a scene point is
  // R(yaw) * tracked + (x, y, z), so the transform is that map's inverse:
  // R(-yaw) applied to -(x, y, z), then R(-yaw).
  function offset() {
    const c = Math.cos(place.yaw), s = Math.sin(place.yaw);
    return {
      position: { x: -(c * place.x - s * place.z), y: -place.y, z: -(s * place.x + c * place.z), w: 1 },
      orientation: { x: 0, y: -Math.sin(place.yaw / 2), z: 0, w: Math.cos(place.yaw / 2) },
    };
  }

  function reset() {
    Object.assign(place, { x: 0, y: 0, z: 0, yaw: 0 });
    blink = null;
    fade = 0;
    changed = true;
  }

  return {
    /** Where the tracked space stands in the scene: `{x, y, z, yaw}` (a copy). */
    get place() { return { ...place }; },
    /** 0-1: the blink's black. */
    get fade() { return fade; },
    /** Whether a blink is under way; a jump is refused until it ends. */
    get blinking() { return blink !== null; },
    /** Whether the rig has moved since the loop last re-anchored to it. */
    get changed() { return changed; },
    /** `changed`, cleared: the loop, re-anchoring. */
    takeChange() {
      const was = changed;
      changed = false;
      return was;
    },
    toScene,
    directionToScene,
    orientationToScene,
    headingOf,
    turn,
    move,
    setFloor,
    land,
    arrive,
    restore,
    jump,
    step,
    offset,
    reset,
  };
})();
