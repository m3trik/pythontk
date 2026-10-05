import { grab } from './grab.js';
import {
  headsetLayer, headsetSpace, overlay, pinching, readHeadsetInput, setHeadsetSpace,
} from './headset.js';
import { locomotion } from './locomotion.js';
import { rig } from './rig.js';
import { beginHeadsetView, endHeadsetView, renderer } from './scene.js';
import {
  atStart, beginArrival, endArrival, recenter, returnToStart, startPlace, stepStart,
} from './start.js';
import { surfaces } from './surfaces.js';

// A headset frame on plain values -- what the page runs from each XRFrame, and
// the tests on synthetic input. The rig's blink first, then the start (a
// session's first place), then locomotion while it is on: the one place the
// three meet, and none of them knows the others are there.
export const headset = {
  /** As a session starts: the rig at the origin, locomotion idle, the start due. */
  begin() {
    rig.reset();
    locomotion.reset();
    beginArrival();
  },
  /**
   * One frame. `input` is `{head, left, right, aim, grips, aims, squeeze}` in
   * the tracked space (see `readHeadsetInput`), each stick `[x, y]` with y
   * negative pushed forward, as XR gamepads report it. Returns the frame's
   * events: 'arrive' as a session is placed at its start, 'aim', 'turn', and
   * 'land' as a jump arrives.
   */
  step(delta, input = {}, time = 0) {
    const events = rig.step(delta, input.head);
    surfaces.refresh();
    // Locomotion off keeps a session where it started -- so a viewer it had
    // carried off before it was switched off, mid-session, is taken back.
    if (!locomotion.enabled && !atStart()) returnToStart();
    events.push(...stepStart(input));
    events.push(...locomotion.step(delta, input, events, time));
    // After locomotion: a held point rides the hand where the rig now stands.
    grab.step(input);
    overlay.material.uniforms.fade.value = rig.fade;
    overlay.visible = rig.fade > 0 || overlay.material.uniforms.vignette.value > 0;
    return events;
  },
  /** As a session ends: everything back, and nothing due. */
  end() {
    rig.reset();
    locomotion.reset();
    grab.end();
    endArrival();
  },
  /** What a recenter does: see `recenter`. */
  recenter,
  /** An XRFrame read into what `step` takes: see `readHeadsetInput`. */
  read: readHeadsetInput,
  /** Where a session starts, `{point, heading}`, as the model stands now; null without a start node. */
  get start() { return startPlace(); },
};

// The XRFrame side: the poses read, the frame stepped, and the session's space
// re-anchored whenever the rig moved.
export function stepHeadset(frame, delta, time) {
  headset.step(delta, headset.read(frame), time);
  if (rig.takeChange()) {
    const { position, orientation } = rig.offset();
    renderer.xr.setReferenceSpace(
      headsetSpace.getOffsetReferenceSpace(new XRRigidTransform(position, orientation)),
    );
  }
}

// Each session starts afresh -- at the model's start node, on its first frame
// with a model in -- and everything drawn for it goes when the session does.
// The desktop view is kept aside for its end (see desktopView), and the
// session draws with a headset's near plane.
renderer.xr.addEventListener('sessionstart', () => {
  beginHeadsetView();
  setHeadsetSpace(renderer.xr.getReferenceSpace());
  headsetSpace.addEventListener('reset', headset.recenter);
  // A tracked hand's pinch is a 'select'; a controller's trigger is too, and
  // is read off its gamepad as well -- either way the hand holds.
  const session = renderer.xr.getSession();
  const pinch = (on) => (event) => {
    const hand = event.inputSource?.handedness;
    if (hand === 'left' || hand === 'right') pinching[hand] = on;
  };
  session.addEventListener('selectstart', pinch(true));
  session.addEventListener('selectend', pinch(false));
  headset.begin();
  headsetLayer.visible = true;
  surfaces.warm();
});
renderer.xr.addEventListener('sessionend', () => {
  setHeadsetSpace(null);
  pinching.left = pinching.right = false;
  headset.end();
  headsetLayer.visible = false;
  endHeadsetView();
});
