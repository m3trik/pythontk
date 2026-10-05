// The manifest poll: what the server says the page should be showing -- the
// asset and its version, the active scripts, and the switches (title, start
// node, locomotion, guest, XR runtime) -- read once a second.

import { setGuest, loadScripts } from './api.js';
import { el } from './dom.js';
import { load } from './load.js';
import { locomotion } from './locomotion.js';
import { setStartName } from './start.js';
import { setStatus } from './status.js';
import { setXrRuntime } from './xr.js';

const POLL_MS = 1000;
const LOAD_RETRIES = 3;

// This tab's id, on every manifest poll and on the close beacon. A share counts
// its guests by it -- one per open tab, however many sit behind one address --
// and the owner's own listener ignores it.
const VIEWER_ID = Math.random().toString(36).slice(2, 12);

let seenVersion = -1;
let failedVersion = -1;
let failCount = 0;
// Fingerprint of THIS page as the server published it -- the page and every
// kernel module. The poll swaps the model but never the script running it, so
// when the server starts naming a different page the one in this tab is stale
// and reloads itself -- a viewer fix otherwise reaches a session left open only
// via F5.
let viewerStamp = null;

async function poll() {
  try {
    const manifest = await (await fetch(`manifest.json?id=${VIEWER_ID}`, { cache: 'no-store' })).json();
    if (manifest.viewer) {
      if (viewerStamp !== null && manifest.viewer !== viewerStamp) {
        setStatus('viewer updated - reloading…');
        location.reload();
        return;
      }
      viewerStamp = manifest.viewer;
    }
    if (manifest.title) el.title.textContent = manifest.title;
    // Ahead of the load, which looks the node up by it.
    setStartName(manifest.userPos ?? null);
    // Live too, and at once: switched off, a headset mid-session stops going
    // anywhere on this poll -- a guest's included.
    locomotion.enabled = manifest.locomotion !== false;
    // Ahead of the scripts below, which may ask `viewer.guest` as they load.
    const guest = manifest.guest === true;
    setGuest(guest);
    el.normalSave.hidden = guest;
    // Ahead of the version check, so it lands on the FIRST poll rather than
    // waiting for a publish.
    setXrRuntime(manifest.xrRuntime);
    // Ahead of the version check: the active set is decided per push, and the
    // FIRST poll happens before anything is published at all — gated behind
    // the early return, a script named on a server that has yet to publish
    // would never load.
    //
    // AWAITED, so a script activated by the same push that named the asset is
    // registered before the model loads and cannot miss its 'load' event: a
    // small GLB parses in less time than an ES module takes to arrive
    // (measured on KB-scale fixtures), which left the shadow shim with no
    // scene. loadScripts waits for an import another poll has under way as
    // well, so a poll landing mid-import cannot run the load ahead of it. A
    // push whose scripts fail still loads the model — loadScripts catches per
    // module.
    await loadScripts(manifest.scripts);
    if (manifest.version === seenVersion) return;
    // A version that has failed its budget is abandoned until a new publish
    // supersedes it — otherwise a genuinely corrupt GLB re-parses every second.
    if (manifest.version === failedVersion && failCount >= LOAD_RETRIES) return;

    if (!manifest.asset) {
      seenVersion = manifest.version;
      setStatus('waiting for the first publish…');
      return;
    }

    seenVersion = manifest.version;
    // The query string is what actually defeats a warm HTTP cache when the
    // asset keeps a stable name across pushes.
    const ok = await load(`${manifest.asset}?v=${manifest.version}`, manifest.version);
    if (ok === null) return; // superseded mid-load — the newer call owns the bookkeeping
    if (ok) {
      failedVersion = -1;
      failCount = 0;
      return;
    }
    // Leaving the version marked as seen is what would strand the page: a
    // transient failure must not stop the next poll from trying it again.
    failCount = manifest.version === failedVersion ? failCount + 1 : 1;
    failedVersion = manifest.version;
    seenVersion = -1;
  } catch (error) {
    setStatus('server unreachable', 'error');
  }
}

// The poll's start, run once by the composition root (main.js).
export function startPolling() {
  poll();
  setInterval(poll, POLL_MS);

  // Tell the server the moment this page goes away, so the next push reopens a
  // tab instead of publishing to nothing. Without it the server can only infer a
  // closed viewer from polls stopping, and that inference has to wait out a
  // window long enough to tolerate hidden-tab timer throttling (~1/min) — so a
  // close-then-push would otherwise show nothing at all for a minute and a half.
  // `pagehide` rather than `unload`: it is the event that still fires on mobile
  // and bfcache paths, and sendBeacon is the only send that survives teardown.
  // A page restored from bfcache simply polls again and re-arms the server.
  addEventListener('pagehide', () => { navigator.sendBeacon(`viewer-closed?id=${VIEWER_ID}`, ''); });
}
