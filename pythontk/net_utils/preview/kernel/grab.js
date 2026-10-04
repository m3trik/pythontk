import * as THREE from 'three';
import { rig } from './rig.js';
import { camera, renderer } from './scene.js';

// Grabbing: the page's one hand on the model. A script registers what a hand
// may take hold of (`viewer.grab.register(object, handler)`), and the page
// delivers every way of holding one -- the desktop pointer, either headset
// controller -- as the same three calls on plain values, so a script is
// written once and a test drives it with synthetic input:
//
//   begin({source, point, pose})     the world point taken hold of, and the
//                                    world pose ({position, quaternion}) of
//                                    what took it -- the camera, a grip.
//                                    Return false to refuse the hold.
//   move({source, point, rotation})  where that point should now be (world),
//                                    and the holder's turn since `begin` (a
//                                    world THREE.Quaternion), or null: a
//                                    mouse holds a position only.
//   end({source})
//
// `source` is 'mouse', 'left' or 'right'. The page moves nothing itself --
// what a hold does is the handler's: an articulated rig solves its joints, a
// door would swing. Handlers run outside the render, so a script that poses
// nodes re-applies its pose in its own 'frame' hook -- which runs after the
// animation mixer; posed from here alone, a playing clip would overwrite it.
//
// On the desktop a press on a registered object takes it AHEAD of the orbit
// controls (the page listens on the window in the capture phase and stops
// the event), the drag moves the point in the camera-facing plane through
// it, and the wheel pushes the plane away or pulls it closer. In a headset
// the grip (or the trigger) takes whatever the hand is inside of or within
// GRAB_NEAR of -- held where the hand is -- else whatever the controller's
// ray points at within GRAB_FAR; the held point then rides the controller.
const GRAB_NEAR = 0.1;        // m
const GRAB_FAR = 20;          // m
const GRAB_WHEEL = 0.1;       // of the point's distance, per wheel notch

export const grab = (() => {
  const entries = new Map();          // Object3D -> handler
  const holds = new Map();            // source -> hold
  const pressed = { left: false, right: false };
  const raycaster = new THREE.Raycaster();
  const box = new THREE.Box3();
  const scratch = new THREE.Vector3();
  const pointer = new THREE.Vector2();
  let lastPointer = null;
  let hoverQueued = false;

  // A handler is script code: contained like a script's hook, so a throw
  // never breaks the page's input.
  function call(handler, name, detail) {
    try {
      return handler[name]?.(detail);
    } catch (error) {
      console.warn(`grab handler threw on '${name}':`, error);
      return false;
    }
  }

  function register(object, handler) {
    entries.set(object, handler);
    return () => {
      entries.delete(object);
      for (const [source, hold] of holds) if (hold.object === object) holds.delete(source);
    };
  }

  // The registered object a hit belongs to: itself or its nearest registered
  // ancestor, so a script registers a part and a hit on any mesh under it
  // counts.
  function ownerOf(object) {
    for (let o = object; o; o = o.parent) if (entries.has(o)) return o;
    return null;
  }

  const live = () => [...entries.keys()].filter((object) => object.parent);

  /** The nearest registered object on a ray: `{object, point, distance}`, or null. */
  function pick(origin, direction, far = Infinity) {
    const roots = live();
    if (!roots.length) return null;
    raycaster.set(origin, direction.clone().normalize());
    raycaster.far = far;
    for (const hit of raycaster.intersectObjects(roots, true)) {
      const object = ownerOf(hit.object);
      if (object) return { object, point: hit.point.clone(), distance: hit.distance };
    }
    return null;
  }

  // The registered object a hand at `position` is inside or within GRAB_NEAR
  // of -- boxes, not triangles: a hand closing on a thin rod must not miss.
  function near(position) {
    let best = null;
    for (const object of live()) {
      box.setFromObject(object);
      const distance = box.clampPoint(position, scratch).distanceTo(position);
      if (distance <= GRAB_NEAR && (!best || distance < best.distance)) {
        best = { object, point: position.clone(), distance };
      }
    }
    return best;
  }

  // Begin a hold on `found` for `source`, remembering the point in the
  // holder's frame so it can ride the holder.
  function take(source, found, pose) {
    if (!found || holds.has(source)) return false;
    const handler = entries.get(found.object);
    if (call(handler, 'begin', { source, point: found.point.clone(), pose }) === false) return false;
    holds.set(source, {
      object: found.object,
      handler,
      offset: found.point.clone().sub(pose.position).applyQuaternion(pose.quaternion.clone().invert()),
      start: pose.quaternion.clone(),
      plane: null,
    });
    return true;
  }

  /** Move `source`'s hold to the world `point`, with the holder's turn (or null). */
  function drag(source, point, rotation = null) {
    const hold = holds.get(source);
    if (hold) call(hold.handler, 'move', { source, point: point.clone(), rotation });
  }

  // The held point riding a holder now at `pose`.
  function follow(source, pose) {
    const hold = holds.get(source);
    if (!hold) return;
    const point = hold.offset.clone().applyQuaternion(pose.quaternion).add(pose.position);
    drag(source, point, pose.quaternion.clone().multiply(hold.start.clone().invert()));
  }

  function release(source) {
    const hold = holds.get(source);
    if (!hold) return;
    holds.delete(source);
    call(hold.handler, 'end', { source });
    if (source === 'mouse') renderer.domElement.style.cursor = '';
  }

  /**
   * Take hold along a ray: what `origin` + `direction` (world) hits first,
   * held by a holder at `pose` (default: at `origin`, unturned). Returns
   * whether a hold began.
   */
  function press(source, origin, direction, pose = null) {
    const found = pick(origin, direction);
    return take(source, found, pose || { position: origin.clone(), quaternion: new THREE.Quaternion() });
  }

  // ---- the desktop pointer
  function cameraRay(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.setFromCamera(pointer, camera);
    return raycaster.ray.clone();
  }

  function pointerDrag() {
    const hold = holds.get('mouse');
    if (!hold || !lastPointer) return;
    const point = cameraRay(lastPointer).intersectPlane(hold.plane, new THREE.Vector3());
    if (point) drag('mouse', point, null);
  }

  function hover() {
    hoverQueued = false;
    if (!lastPointer || holds.has('mouse')) return;
    const ray = cameraRay(lastPointer);
    renderer.domElement.style.cursor = pick(ray.origin, ray.direction) ? 'grab' : '';
  }

  window.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 || event.target !== renderer.domElement || renderer.xr.isPresenting) return;
    if (!entries.size) return;
    const ray = cameraRay(event);
    const found = pick(ray.origin, ray.direction);
    const pose = {
      position: camera.getWorldPosition(new THREE.Vector3()),
      quaternion: camera.getWorldQuaternion(new THREE.Quaternion()),
    };
    if (!take('mouse', found, pose)) return;
    holds.get('mouse').plane = new THREE.Plane().setFromNormalAndCoplanarPoint(
      camera.getWorldDirection(new THREE.Vector3()), found.point,
    );
    lastPointer = { clientX: event.clientX, clientY: event.clientY };
    renderer.domElement.style.cursor = 'grabbing';
    event.stopPropagation();
    event.preventDefault();
  }, true);

  window.addEventListener('pointermove', (event) => {
    lastPointer = { clientX: event.clientX, clientY: event.clientY };
    if (holds.has('mouse')) {
      event.stopPropagation();
      pointerDrag();
    } else if (entries.size && !hoverQueued) {
      hoverQueued = true;
      requestAnimationFrame(hover);
    }
  }, true);

  window.addEventListener('pointerup', (event) => {
    if (!holds.has('mouse')) return;
    event.stopPropagation();
    release('mouse');
  }, true);

  window.addEventListener('wheel', (event) => {
    const hold = holds.get('mouse');
    if (!hold) return;
    event.stopPropagation();
    event.preventDefault();
    // Push (wheel down) or pull the plane along the view, by a share of the
    // point's own distance so a notch is as big near the model as far off.
    const eye = camera.getWorldPosition(new THREE.Vector3());
    const distance = Math.max(Math.abs(hold.plane.distanceToPoint(eye)), 0.01);
    hold.plane.constant -= Math.sign(event.deltaY) * GRAB_WHEEL * distance;
    pointerDrag();
  }, { capture: true, passive: false });

  // ---- the headset
  /** One headset frame of it: `input` as `readHeadsetInput` returns it. */
  function step(input = {}) {
    for (const hand of ['left', 'right']) {
      const squeezing = !!input.squeeze?.[hand];
      const was = pressed[hand];
      pressed[hand] = squeezing;
      const grip = input.grips?.[hand];
      const pose = grip
        ? { position: rig.toScene(grip.position), quaternion: rig.orientationToScene(grip.orientation) }
        : null;
      if (squeezing && !was && pose && entries.size) {
        let found = near(pose.position);
        const aim = input.aims?.[hand];
        if (!found && aim) {
          found = pick(rig.toScene(aim.position), rig.directionToScene(aim.orientation), GRAB_FAR);
        }
        take(hand, found, pose);
      } else if (!squeezing && was) {
        release(hand);
      } else if (squeezing && pose) {
        follow(hand, pose);
      }
    }
  }

  // A session's end lets go of whatever the hands held.
  function end() {
    release('left');
    release('right');
    pressed.left = pressed.right = false;
  }

  return {
    register,
    pick,
    press,
    drag,
    release,
    step,
    end,
    /** The sources holding something now: 'mouse', 'left', 'right'. */
    get holding() { return [...holds.keys()]; },
    /** How many objects are registered. */
    get count() { return entries.size; },
  };
})();
