/*
  The live WebXR preview's kernel -- the composition root.

  The page (`viewer.html`) is markup and this one module. The kernel is the
  viewer itself, one part per module below, each importing only parts beneath
  it: the stage (scene.js) and the page's own elements (dom.js) at the bottom;
  the model, its lightmaps and reflection probe, specs, animation and the
  headset's rig, surfaces, grab, start and locomotion above them; the page's
  menubar (menus.js -- one button per category, each a dropdown of its
  items) and the windows its items open (windows.js), and what the kernel
  files there itself: the Environment menu (environment.js), the published
  parts (parts.js) and the clip captions (captions.js); then the session that
  runs a headset frame, the published `viewer` API (api.js), a deliverable's
  load (load.js), the manifest poll and the page's controls. Importing them builds the page;
  this module starts it, in the one order that matters: the XR check, then
  the poll, then the frame loop.

  A FEATURE (`features/<name>`) is not imported here: the server activates it
  through the manifest and api.js imports it and hands it `viewer` -- the one
  seam the page grows through. A feature reads the deliverable's manifests by
  the keys the scene records declare (records.js, generated from pythontk's
  SceneRecords) and may build on the pure math every port shares (math.js).
*/

import { initXr } from './xr.js';
import { startPolling } from './poll.js';
import { startLoop } from './ui.js';
// Its listeners are its whole job: the headset session's start and end.
import './session.js';

window.__viewerBooted = true; // imports resolved — stand the boot watchdog down

initXr();
startPolling();
startLoop();
