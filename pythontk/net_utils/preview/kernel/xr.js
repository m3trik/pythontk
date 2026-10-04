// Whether a headset can present, and the one button that starts a session --
// kept current as devices come and go, and honest about why it cannot.

import { VRButton } from 'three/addons/webxr/VRButton.js';
import { el } from './dom.js';
import { renderer } from './scene.js';
import { contextLost } from './status.js';

// Support is re-checked on `devicechange`, not only at load. A headset powered
// on, plugged in or woken AFTER the page opened is the normal case -- the page
// is left open for the next push, and the headset is not -- and a load-time-only
// check answers "no device" forever, curable only by a reload. Finding the tab
// to reload it is exactly what nobody can do while wearing one.
let vrButton = null;
let enterTimer = null;
const HINT_READY = 'Headset detected — press Enter VR.';
const HINT_STARTING = 'Starting session…';

// Set from the manifest, because this is the one thing the SERVER can see and
// the page cannot: whether an OpenXR runtime is installed at all. Without the
// distinction "no immersive device" reads as "connect a headset" to someone
// whose headset IS connected — the cable is in, the desktop app is running,
// and the only missing piece is a Link session, which nothing on screen names.
// Null until the first poll answers, so the wording stays generic until known.
let xrRuntimeInstalled = null;

function noDeviceHint() {
  if (!xrRuntimeInstalled) {
    return 'No headset — connect one, this page will notice.';
  }
  return 'No headset presenting — start Link or SteamVR.';
}

export function refreshXrSupport() {
  return navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
    if (supported && !vrButton) {
      vrButton = VRButton.createButton(renderer);
      // CAPTURE phase, so this runs BEFORE the button's own handler: a failure
      // message from the previous attempt has to be cleared on the way in, or a
      // press that now succeeds leaves the page still reporting the old
      // failure. Bubble phase would run after the throw and wipe the fresh
      // message instead of the stale one.
      vrButton.addEventListener('click', () => {
        // Clears a failure message from the previous attempt, and starts the
        // watchdog below.
        el.hint.textContent = HINT_STARTING;
        clearTimeout(enterTimer);
        // A refused session rejects and is reported; a headset that is asleep
        // or not presenting leaves `requestSession` PENDING instead, so nothing
        // rejects, nothing starts, and the button sits there saying ENTER VR --
        // the same dead-button symptom, with no error anywhere to explain it.
        // Same treatment as the CDN watchdog: say so rather than hang silently.
        enterTimer = setTimeout(() => {
          if (renderer.xr.isPresenting) return;
          // Only overwrite our own "starting" text: a rejection that already
          // reported a REASON is more useful than this generic timeout.
          if (el.hint.textContent === HINT_STARTING) {
            el.hint.textContent = 'Headset did not answer — is Link running?';
          }
        }, 12000);
      }, true);
      // Born inert under an open prompt, like everything else behind it.
      vrButton.inert = !el.dialog.hidden;
      document.body.appendChild(vrButton);
    } else if (!supported && vrButton && !renderer.xr.isPresenting) {
      // A button whose device has gone is a button that throws when pressed.
      // NEVER while presenting, though: that same button is the only way out
      // of the session, so removing it would strand whoever is wearing it.
      vrButton.remove();
      vrButton = null;
    }
    // Not while the context is lost: this fires UNPROMPTED on `devicechange`,
    // and "Headset detected — press Enter VR" under a red "context lost"
    // status invites the one action that cannot work. The button itself is
    // still kept current, so a restore needs no second device event.
    if (contextLost) return;
    el.hint.textContent = supported ? HINT_READY : noDeviceHint();
  }).catch((error) => {
    // Say which case it is in rather than rejecting into the console, where a
    // headset user cannot see it. Same reasoning as the CDN watchdog above.
    el.hint.textContent = 'WebXR device check failed: ' + error;
  });
}

// A refused `requestSession` rejects INSIDE VRButton's own click handler, where
// nothing catches it, so the button just sits there saying ENTER VR -- the
// failure reads as a dead button rather than as a refusal with a reason. The
// commonest cause is an immersive session still held by another tab, another
// browser or a crashed one, which is exclusive per device, invisible from this
// page, and unguessable from inside a headset.
// Both events, because which one fires is not ours to choose: Chromium raises
// InvalidStateError from `requestSession` SYNCHRONOUSLY, so it surfaces as an
// uncaught error rather than a rejected promise, while other refusals reject.
// Listening to only one of them is how the first attempt at this failed.
function reportXrFailure(text) {
  if (!/XRSession|requestSession|immersive/i.test(text)) return false;
  el.hint.textContent = 'Enter VR failed — ' + text;
  return true;
}
addEventListener('unhandledrejection', (event) => {
  const error = event.reason;
  if (reportXrFailure(String((error && error.message) || error || ''))) {
    event.preventDefault();
  }
});
addEventListener('error', (event) => {
  const error = event.error;
  reportXrFailure(String((error && error.message) || event.message || ''));
});

// A restored context hands the hint line back to whoever owns it, rather than
// leaving a stale recovery message sitting where the XR state belongs. After
// status.js's own listener, which clears the lost flag this reads.
renderer.domElement.addEventListener('webglcontextrestored', () => {
  if (navigator.xr) refreshXrSupport();
  else el.hint.textContent = '';
});

// The boot check, run once by the composition root (main.js).
export function initXr() {
  if (navigator.xr) {
    refreshXrSupport();
    // Fired by the UA whenever the set of available XR devices changes. Guarded
    // because the boot continues past this -- the model load, the poll loop: a
    // throw here would cost the whole page, turning "no VR button" into
    // "nothing works", and no browser is worth that trade.
    if (typeof navigator.xr.addEventListener === 'function') {
      navigator.xr.addEventListener('devicechange', refreshXrSupport);
    }
  } else {
    // Reachable when the page is opened over a plain-HTTP LAN address rather
    // than localhost: WebXR is gated on a secure context, and the failure is
    // otherwise silent (the button simply never appears).
    el.hint.textContent = location.hostname === 'localhost' || location.hostname === '127.0.0.1'
      ? 'WebXR unavailable here — try Edge or Chrome.'
      : 'WebXR needs localhost or HTTPS.';
  }
}

// The manifest's word on whether an XR runtime is installed (`xrRuntime`): a
// change re-words the no-device hint at once, on the FIRST poll rather than
// the first publish -- what turns "connect a headset" into "start Link" for
// someone who already did connect one.
export function setXrRuntime(installed) {
  if (installed === undefined || installed === xrRuntimeInstalled) return;
  xrRuntimeInstalled = installed;
  if (navigator.xr) refreshXrSupport();
}
