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
const HINT_IN_VR = 'In VR.';
const HINT_GPU_MOVED = "The browser moved to the headset's GPU — press Enter VR again"
  + ' (to skip this, set the browser to High performance in Windows Settings > Display > Graphics).';
// A failed entry's message, which outranks "headset detected" until the next
// press: the device check rewrites the hint on its own (a restore, a device
// event) and would otherwise wipe the one line that says what to do.
let notice = null;
// Per press: whether the one automatic retry (see `recoverEntry`) is spent, and
// whether the graphics context was lost along the way -- the GPU move a retry
// cures, as opposed to a failure that would only repeat.
let retried = false;
let lostDuringEntry = false;

// Set from the manifest, because this is the one thing the SERVER can see and
// the page cannot: which OpenXR runtime is active (its name; false for none).
// Without the distinction "no immersive device" reads as "connect a headset"
// to someone whose headset IS connected — the cable is in, the desktop app is
// running, and the only missing piece is a Link session, which nothing on
// screen names. Null until the first poll answers, so the wording stays
// generic until known.
let xrRuntime = null;
// The first runtime this page heard of: the one the browser's XR service most
// likely loaded. Chromium loads a runtime ONCE per XR service and never
// re-reads the registry, so one made active later is invisible to this browser
// until it restarts -- no `devicechange` comes, and every check answers "no
// device". Measured 2026-10-18: Virtual Desktop was the active runtime while
// Edge's XR service, started the day before, still held Oculus's; a fresh Edge
// answered supported at once. A reload forgets this; the service, which
// outlives the page, does not -- so the plain hint names the cure as well.
let xrRuntimeLoaded = null;
const RESTART_URL = /\bEdg\//.test(navigator.userAgent) ? 'edge://restart' : 'chrome://restart';

function noDeviceHint() {
  if (!xrRuntime) {
    return 'No headset — connect one, this page will notice.';
  }
  if (xrRuntimeLoaded && xrRuntime !== xrRuntimeLoaded) {
    return `${xrRuntime} became the VR runtime after this browser loaded ${xrRuntimeLoaded}`
      + ` — restart the browser to use it (${RESTART_URL}).`;
  }
  // A page opened after the switch cannot see it, so the cure is named here too
  // -- after "connect": Virtual Desktop RUNNING but not connected to this PC
  // offers no headset either (measured 2026-10-18), and a restart cures nothing.
  return 'No headset presenting — start Link or SteamVR, or connect Virtual Desktop'
    + ' (connected and still this? restart the browser).';
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
      vrButton.addEventListener('click', (event) => {
        // The same button exits: a press while presenting starts nothing, and
        // must not leave "Starting session…" behind it.
        if (renderer.xr.isPresenting) return;
        // Clears a failure message from the previous attempt, and starts the
        // watchdog below.
        el.hint.textContent = HINT_STARTING;
        notice = null;
        lostDuringEntry = false;
        // A person's press earns a retry; the page's own retry (untrusted) does not.
        if (event.isTrusted) retried = false;
        clearTimeout(enterTimer);
        // A refused session rejects and is reported; a headset that is asleep
        // or not presenting leaves `requestSession` PENDING instead, so nothing
        // rejects, nothing starts, and the button sits there saying ENTER VR --
        // the same dead-button symptom, with no error anywhere to explain it.
        // Same treatment as the page's boot watchdog (viewer.html): say so
        // rather than hang silently.
        enterTimer = setTimeout(() => {
          if (renderer.xr.isPresenting) return;
          // Only overwrite our own "starting" text: a rejection that already
          // reported a REASON is more useful than this generic timeout.
          // A pending request is as often the browser's own VR permission
          // prompt, waiting in this window, as a runtime that is not running.
          if (el.hint.textContent === HINT_STARTING) {
            el.hint.textContent = 'Headset did not answer — allow the VR prompt in this window,'
              + ' or check that Link or Virtual Desktop is running.';
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
    // A press under way owns the line until it starts, fails or times out.
    if (supported && el.hint.textContent === HINT_STARTING) return;
    el.hint.textContent = supported ? notice || HINT_READY : noDeviceHint();
  }).catch((error) => {
    // Say which case it is in rather than rejecting into the console, where a
    // headset user cannot see it. Same reasoning as the page's boot watchdog.
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

// three hands a granted session to `renderer.xr.setSession`, which awaits
// `makeXRCompatible` AFTER taking it -- and nothing up the chain catches a
// rejection. On a hybrid-GPU laptop whose browser started on the integrated
// GPU that rejection is the normal first press: Chromium relaunches its GPU
// process on the headset's GPU, the page's context is lost and back a second
// later on the right one, and three has already given up -- leaving the session
// OPEN with nothing drawing it, a black headset for as long as anyone waits
// (measured 2026-10-18, Edge on Iris Xe + RTX 3070 Ti through Virtual Desktop:
// InvalidStateError, context lost, restored 1.0 s later; started on the RTX,
// the same press presented at once). So a setup that fails ends its session,
// and once the context is back the press is made again -- once, and only while
// the browser still counts the click as a gesture, which a session needs.
// Bound: three's setSession reaches its manager through `this` (setFoveation).
const setSession = renderer.xr.setSession.bind(renderer.xr);
renderer.xr.setSession = async (session) => {
  try {
    await setSession(session);
  } catch (error) {
    console.warn('Enter VR: the session could not be set up, so it is ended.', error);
    // Read NOW: the gesture lapses while the context comes back.
    const retry = !retried && !!navigator.userActivation?.isActive;
    retried = true;
    const ended = new Promise((resolve) => {
      session.addEventListener('end', resolve, { once: true });
      setTimeout(resolve, 3000);  // a session already ended sends none
    });
    // three's own end handler then throws on the frame loop it never started
    // ("reading 'cancelAnimationFrame'" of null): noise, nothing left open.
    session.end().catch(() => {});
    // Resolved rather than rethrown: VRButton catches nothing, and resolving
    // lets its own 'end' handler put the button back.
    recoverEntry(ended, retry, error);
  }
};

async function recoverEntry(ended, retry, error) {
  await ended;
  if (!(await whenDrawable())) return;  // still lost: status.js's "reload" stands
  if (retry && lostDuringEntry && navigator.userActivation?.isActive && vrButton) {
    vrButton.click();
    return;
  }
  // Without a GPU move, another press would only repeat it: say why instead.
  notice = lostDuringEntry ? HINT_GPU_MOVED
    : 'Enter VR failed — ' + String((error && error.message) || error);
  el.hint.textContent = notice;
}

// Resolves true once the graphics context can draw, false if it never comes back.
function whenDrawable() {
  if (!renderer.getContext().isContextLost()) return Promise.resolve(true);
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(false), 10000);
    renderer.domElement.addEventListener('webglcontextrestored', () => {
      clearTimeout(timer);
      resolve(true);
    }, { once: true });
  });
}

renderer.domElement.addEventListener('webglcontextlost', () => {
  lostDuringEntry = true;
});

// A press that worked hands the line on: in VR it says so, and on the way out
// the device check takes it back.
renderer.xr.addEventListener('sessionstart', () => {
  clearTimeout(enterTimer);
  el.hint.textContent = HINT_IN_VR;
});
renderer.xr.addEventListener('sessionend', () => {
  if (navigator.xr) refreshXrSupport();
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

// The manifest's word on which XR runtime is active (`xrRuntime`): a change
// re-words the no-device hint at once, on the FIRST poll rather than the first
// publish -- what turns "connect a headset" into "start Link" for someone who
// already did connect one, and names a runtime switched under the browser.
export function setXrRuntime(runtime) {
  if (runtime === undefined || runtime === xrRuntime) return;
  xrRuntime = runtime;
  // The first one there is: with none at all, the service retries, and loads
  // whichever comes first.
  if (!xrRuntimeLoaded && runtime) xrRuntimeLoaded = runtime;
  if (navigator.xr) refreshXrSupport();
}
