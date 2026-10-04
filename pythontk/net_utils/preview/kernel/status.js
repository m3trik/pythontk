// The status line, and the one fact that outranks everything it says: whether
// the GPU context is gone.

import { el } from './dom.js';
import { renderer } from './scene.js';

// A lost GPU context takes every texture and buffer with it, and NOTHING on
// this page noticed: measured against a forced loss, the canvas went blank
// while the overlay still showed a green dot, "v4 · updated 3:04:24 PM",
// "1500 meshes · 96,928 tris" and "Headset detected" — the page reporting
// perfect health with nothing being drawn. That reads as "the textures went
// missing" and is unguessable from the outside. The causes here are all
// silent: a display-driver reset, GPU memory pressure, and a headset link
// dropping (which renegotiates the adapter under an immersive session).
// three.js registers its own handlers and re-uploads everything if a restore
// arrives — measured, the compressed textures come back — but the browser is
// under no obligation to send one, so a page can sit blank indefinitely.
// Hence: say which it is, and say the one thing that fixes the stuck case.
// Sticky, and it outranks every other status: the canvas stays blank until
// the context comes back, so the NEXT push repainting the overlay green (see
// `setStatus`) would reinstate the exact lie this exists to stop -- and
// pushing again is the first thing anyone does when a preview looks wrong.
export let contextLost = false;
const LOST_MESSAGE = 'graphics context lost — reload';

renderer.domElement.addEventListener('webglcontextlost', () => {
  contextLost = true;
  setStatus(LOST_MESSAGE, 'error');
  el.hint.textContent = 'The GPU dropped this page. Nothing is drawing — reload.';
});
// Registered before the XR block's own listener (xr.js imports this module),
// which hands the hint line back to whoever owns it.
renderer.domElement.addEventListener('webglcontextrestored', () => {
  contextLost = false; // Cleared FIRST, or the guard below swallows this.
  setStatus('context restored', 'live');
});

export function setStatus(text, cls = '') {
  // A lost context outranks whatever else the page has to report, including a
  // fresh publish and an unreachable server: nothing is being DRAWN, so no
  // other status is the one to act on. The hint line still carries the detail.
  if (contextLost) {
    text = LOST_MESSAGE;
    cls = 'error';
  }
  el.status.textContent = text;
  el.status.className = cls;
}
