import * as THREE from 'three';
import { addHeadsetCard, headsetLayer, overlay } from './headset.js';
import { rig } from './rig.js';
import { surfaces } from './surfaces.js';

// Getting around in a headset, on the thumbsticks of the two controllers every
// standalone headset ships with -- the scheme most VR apps default to:
//
//   left stick    walk where you look: eased in and out, a comfort vignette
//                 narrowing the view while moving, and following the floor up a
//                 step or down a ledge
//   right stick   flick sideways to snap-turn 45 degrees about your head, one
//                 turn per flick; push forward to aim a teleport arc, release
//                 to jump there through a short blink, pull back to cancel
//
// Optional, and live: the server's switch (`locomotion` in the manifest, read
// on every poll) turns it off, and then the sticks do nothing and nothing of it
// is drawn -- the viewer stays where the session started, free to look round
// and step about their room but not to go anywhere; one it had carried off
// before the switch is taken back there (see `headset.step`). All it moves is
// the rig.
const MOVE_SPEED = 2.0;       // m/s at full deflection: a brisk walk
const MOVE_EASE = 0.1;        // s: time constant of speeding up and slowing down
const STICK_DEADZONE = 0.15;  // a resting stick drifts; this much reads as zero
const SNAP_TURN = THREE.MathUtils.degToRad(45);
const FLICK_ON = 0.7;         // a turn fires past this sideways ...
const FLICK_OFF = 0.35;       // ... and re-arms once the stick is back inside this
const AIM_ON = 0.6;           // forward past this starts aiming
const AIM_RELEASE = 0.3;      // back inside this jumps
const AIM_CANCEL = 0.5;       // pulled back past this cancels
const ARC_SPEED = 7.5;        // m/s launch: about 6 m of reach on the level
const ARC_GRAVITY = 9.81;
const ARC_STEP = 0.04;        // s of flight per traced segment
const ARC_SEGMENTS = 45;      // 1.8 s of flight, then the throw is lost below
const STEP_UP = 0.45;         // m a walk climbs onto without a teleport
const STEP_DOWN = 1.0;        // m a walk steps down; past it, the floor holds
const FLOOR_EASE = 0.08;      // s: a step up or down, smoothed
const VIGNETTE_MAX = 0.6;
const VIGNETTE_EASE = 0.12;   // s
const ARC_VALID = 0x4fc3f7;
const ARC_INVALID = 0xf2504b;  // the page's error red
const ARC_DOT_SPACING = 0.16;  // m between the arc's dots ...
const ARC_DOT_SPEED = 0.5;     // ... flowing at this m/s toward the landing
const ARC_DOTS = 64;
const HINT_SECONDS = 8;
const HINT_FADE_MS = 400;

export const locomotion = (() => {
  let enabled = true;
  const velocity = new THREE.Vector2();  // scene x and z, m/s
  let floorTarget = null;                // the height a walk's floor eases to
  let turnArmed = true;
  let aiming = false;
  let target = null;                     // {point, normal, valid} while aiming
  let arc = [];                          // the aimed arc, scene points, to its end
  let vignette = 0;

  // The arc: dots flowing along it toward the landing -- cyan where a jump would
  // land, red where it would not.
  const arcDots = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.012, 8, 6),
    new THREE.MeshBasicMaterial({ color: ARC_VALID, transparent: true, opacity: 0.9, depthWrite: false }),
    ARC_DOTS,
  );
  arcDots.count = 0;
  arcDots.frustumCulled = false;
  headsetLayer.add(arcDots);
  const arcDot = new THREE.Object3D();

  function drawArc(points, valid, time) {
    arcDots.material.color.setHex(valid ? ARC_VALID : ARC_INVALID);
    let count = 0;
    let next = ((time / 1000) * ARC_DOT_SPEED) % ARC_DOT_SPACING;  // the next dot, along the arc
    let walked = 0;
    for (let i = 1; i < points.length && count < ARC_DOTS; i++) {
      const length = points[i - 1].distanceTo(points[i]);
      while (length && next <= walked + length && count < ARC_DOTS) {
        arcDot.position.lerpVectors(points[i - 1], points[i], (next - walked) / length);
        arcDot.updateMatrix();
        arcDots.setMatrixAt(count++, arcDot.matrix);
        next += ARC_DOT_SPACING;
      }
      walked += length;
    }
    arcDots.count = count;
    arcDots.instanceMatrix.needsUpdate = true;
  }

  // The landing: a ring laid on the surface, and an arrow on the way you will
  // face -- a jump keeps your heading, and the arrow says so before you commit.
  const reticle = new THREE.Group();
  {
    const line = new THREE.MeshBasicMaterial({
      color: ARC_VALID, transparent: true, opacity: 0.95, depthWrite: false, side: THREE.DoubleSide,
    });
    const fill = new THREE.MeshBasicMaterial({
      color: ARC_VALID, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide,
    });
    const arrow = new THREE.Shape();
    arrow.moveTo(0, 0.34);
    arrow.lineTo(-0.07, 0.26);
    arrow.lineTo(0.07, 0.26);
    arrow.closePath();
    reticle.add(new THREE.Mesh(new THREE.RingGeometry(0.2, 0.235, 48), line));
    reticle.add(new THREE.Mesh(new THREE.CircleGeometry(0.2, 48), fill));
    reticle.add(new THREE.Mesh(new THREE.ShapeGeometry(arrow), line));
  }
  reticle.visible = false;
  headsetLayer.add(reticle);
  const reticleBasis = new THREE.Matrix4();

  function placeReticle(landing, facing, time) {
    const normal = landing.normal;
    const ahead = facing.clone().addScaledVector(normal, -facing.dot(normal));
    if (ahead.lengthSq() < 1e-6) ahead.set(normal.y, -normal.x, 0);  // any tangent
    ahead.normalize();
    const side = new THREE.Vector3().crossVectors(ahead, normal);
    reticle.quaternion.setFromRotationMatrix(reticleBasis.makeBasis(side, ahead, normal));
    reticle.position.copy(landing.point).addScaledVector(normal, 0.01);
    reticle.scale.setScalar(1 + 0.05 * Math.sin(time * 0.006));
  }

  // The controls, said once per session where the viewer first looks. World-
  // locked, a little below eye level: text that follows the head is hard to
  // read. It fades after HINT_SECONDS, or as soon as the controls are in use.
  const hint = addHeadsetCard({ name: 'controls', width: 1024, height: 300, metres: 0.64, radius: 36 });
  hint.draw((g) => {
    g.font = '600 42px system-ui, sans-serif';
    const rows = [
      ['Left stick', 'walk'],
      ['Right stick sideways', 'turn'],
      ['Right stick forward', 'aim, release to teleport'],
    ];
    rows.forEach(([control, action], row) => {
      const y = 90 + row * 74;
      g.fillStyle = '#9aa0a6';
      g.fillText(control, 56, y);
      g.fillStyle = '#e8eaed';
      g.fillText(action, 520, y);
    });
  });
  let hintUntil = 0;  // ms the card holds until; 0 until it is placed this session

  function updateHint(input, events, time) {
    if (!hint.mesh.visible) {
      if (hintUntil || !input.head) return;  // gone for this session, or no head yet
      hint.place(
        rig.toScene(input.head.position),
        rig.directionToScene(input.head.orientation),
        { ahead: 1.2, below: 0.25 },
      );
      hint.mesh.visible = true;
      hintUntil = time + HINT_SECONDS * 1000;
    }
    // The first move says the controls are known: fade then, rather than hold.
    if (events.some((event) => event !== 'arrive') || vignette > 0.02) {
      hintUntil = Math.min(hintUntil, time + HINT_FADE_MS);
    }
    const remaining = hintUntil - time;
    hint.mesh.material.opacity = Math.min(1, remaining / HINT_FADE_MS);
    if (remaining <= 0) hint.mesh.visible = false;
  }

  // A tick in the hand for a turn and a landing: a move the body registers is
  // much of what makes a snap turn feel intended rather than done to you.
  function pulse(source, intensity, milliseconds) {
    try {
      source?.gamepad?.hapticActuators?.[0]?.pulse?.(intensity, milliseconds);
    } catch { /* a controller without rumble */ }
  }

  // The teleport arc: a throw along the aim ray, traced segment by segment
  // until it strikes something or has fallen for ARC_SEGMENTS * ARC_STEP s.
  function trace(origin, direction) {
    const points = [origin.clone()];
    let from = origin;
    for (let step = 1; step <= ARC_SEGMENTS; step++) {
      const t = step * ARC_STEP;
      const to = new THREE.Vector3(
        origin.x + direction.x * ARC_SPEED * t,
        origin.y + direction.y * ARC_SPEED * t - 0.5 * ARC_GRAVITY * t * t,
        origin.z + direction.z * ARC_SPEED * t,
      );
      const hit = surfaces.cast(from, to);
      if (hit) {
        points.push(hit.point);
        return { points, hit };
      }
      points.push(to);
      from = to;
    }
    return { points, hit: null };
  }

  // The floor under (x, z) a walk can follow: a step up to STEP_UP, a drop down
  // to STEP_DOWN. A ledge taller than a step is not a floor -- the downward cast
  // starts below its top -- so the height holds, and a wall is walked through.
  function floorUnder(x, z) {
    const y = rig.place.y;
    const hit = surfaces.cast(
      new THREE.Vector3(x, y + STEP_UP, z),
      new THREE.Vector3(x, y - STEP_DOWN, z),
    );
    return hit && hit.valid ? hit.point.y : null;
  }

  // One frame of the sticks. `events` are the frame's so far: a jump that just
  // landed, or a session just placed at its start -- a frame that moves nothing
  // else. Returns its own: 'aim' as the arc comes up, 'turn' on a snap turn.
  function update(delta, input, events) {
    const own = [];
    // A hitch (a tab regaining focus, a long collection) must not become a leap.
    const dt = Math.min(Math.max(delta, 0), 0.1);
    if (events.includes('land') || events.includes('arrive')) {
      velocity.set(0, 0);
      floorTarget = null;
    }
    if (events.includes('arrive')) return own;
    const head = input.head ? rig.toScene(input.head.position) : null;
    const [rx, ry] = input.right || [0, 0];
    const [lx, ly] = input.left || [0, 0];

    // Right stick: aim, jump or cancel -- or, while not aiming, turn.
    if (aiming) {
      if (ry > AIM_CANCEL || !input.aim) {
        aiming = false;
      } else if (Math.hypot(rx, ry) < AIM_RELEASE) {
        aiming = false;
        if (target && target.valid) rig.jump(target.point);
      }
    } else if (ry < -AIM_ON && Math.abs(rx) < AIM_ON && input.aim && !rig.blinking) {
      aiming = true;
      own.push('aim');
    }
    if (aiming) {
      const thrown = trace(rig.toScene(input.aim.position), rig.directionToScene(input.aim.orientation));
      arc = thrown.points;
      target = thrown.hit;
    } else {
      arc = [];
      target = null;
      if (turnArmed && Math.abs(rx) > FLICK_ON && Math.abs(ry) < FLICK_ON) {
        const { x, y, z } = rig.place;
        rig.turn(-Math.sign(rx) * SNAP_TURN, head || new THREE.Vector3(x, y, z));
        turnArmed = false;
        own.push('turn');
      } else if (Math.abs(rx) < FLICK_OFF) {
        turnArmed = true;
      }
    }

    // Left stick: walk where the head looks, eased in and out. The response
    // curve keeps the first half of the throw for creeping up to something.
    const deflection = Math.min(Math.hypot(lx, ly), 1);
    const amount = deflection > STICK_DEADZONE
      ? ((deflection - STICK_DEADZONE) / (1 - STICK_DEADZONE)) ** 1.5
      : 0;
    let wantX = 0, wantZ = 0;
    if (amount && input.head) {
      const forward = rig.directionToScene(input.head.orientation).setY(0);
      if (forward.lengthSq() > 1e-6) {
        forward.normalize();
        // Right of forward, seen from above, is (-forward.z, forward.x).
        const scale = (amount * MOVE_SPEED) / Math.hypot(lx, ly);
        wantX = (forward.x * -ly - forward.z * lx) * scale;
        wantZ = (forward.z * -ly + forward.x * lx) * scale;
      }
    }
    const ease = 1 - Math.exp(-dt / MOVE_EASE);
    velocity.x += (wantX - velocity.x) * ease;
    velocity.y += (wantZ - velocity.y) * ease;
    if (!amount && velocity.lengthSq() < 1e-4) velocity.set(0, 0);
    if (velocity.x || velocity.y) {
      rig.move(velocity.x * dt, velocity.y * dt);
      if (head) {
        const floor = floorUnder(head.x + velocity.x * dt, head.z + velocity.y * dt);
        if (floor !== null) floorTarget = floor;
      }
    }
    // The floor follows a walk over a step, smoothed so it reads as a step.
    if (floorTarget !== null) {
      const y = rig.place.y;
      const gap = floorTarget - y;
      const next = Math.abs(gap) < 1e-3 ? floorTarget : y + gap * (1 - Math.exp(-dt / FLOOR_EASE));
      rig.setFloor(next);
      if (next === floorTarget) floorTarget = null;
    }

    // The comfort vignette narrows the view with the walking speed: movement
    // the eyes see but the body does not feel is what the periphery objects to.
    const speed = Math.min(velocity.length() / MOVE_SPEED, 1);
    vignette += (speed * VIGNETTE_MAX - vignette) * (1 - Math.exp(-dt / VIGNETTE_EASE));
    if (!speed && vignette < 1e-3) vignette = 0;
    return own;
  }

  // What the frame shows of it, and the ticks in the hand.
  function draw(input, events, time) {
    if (events.includes('turn')) pulse(input.sources?.right, 0.2, 14);
    if (events.includes('land')) pulse(input.sources?.right, 0.35, 26);
    if (aiming) drawArc(arc, !!(target && target.valid), time);
    else arcDots.count = 0;
    reticle.visible = !!(target && target.valid);
    if (reticle.visible) {
      const facing = input.head
        ? rig.directionToScene(input.head.orientation).setY(0)
        : new THREE.Vector3(0, 0, -1);
      if (facing.lengthSq() < 1e-6) facing.set(0, 0, -1);
      placeReticle(target, facing.normalize(), time);
    }
    overlay.material.uniforms.vignette.value = vignette;
    // Said again where the viewer now stands: a card placed while the model was
    // still loading would be left behind at the origin.
    if (events.includes('arrive')) {
      hint.mesh.visible = false;
      hintUntil = 0;
    }
    updateHint(input, events, time);
  }

  // One headset frame of it, while on. `events` are the frame's so far.
  function step(delta, input, events, time) {
    if (!enabled) return [];
    const own = update(delta, input, events);
    draw(input, [...events, ...own], time);
    return own;
  }

  // Idle, with nothing of it drawn: a session's start and end, and switching off.
  function reset() {
    velocity.set(0, 0);
    floorTarget = null;
    turnArmed = true;
    aiming = false;
    target = null;
    arc = [];
    vignette = 0;
    arcDots.count = 0;
    reticle.visible = false;
    hint.mesh.visible = false;
    hintUntil = 0;
    overlay.material.uniforms.vignette.value = 0;
  }

  return {
    /** Whether the sticks move the viewer. The server's switch sets it on every poll. */
    get enabled() { return enabled; },
    set enabled(on) {
      if (enabled && !on) reset();
      enabled = !!on;
    },
    /** The aimed landing, `{point, normal, valid}`, while aiming; else null. */
    get target() { return target; },
    /** The aimed arc's scene points, up to where it ends; empty unless aiming. */
    get arc() { return arc; },
    get aiming() { return aiming; },
    /** 0-1: the walking vignette's strength. */
    get vignette() { return vignette; },
    step,
    reset,
  };
})();
