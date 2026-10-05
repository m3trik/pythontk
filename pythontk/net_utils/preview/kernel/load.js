// A deliverable's load: fetched, parsed, and put through the per-model passes
// in their one order -- the fades wired, the lighting read and spent, the
// lightmaps bound, the clips set up, the model measured, stood and framed --
// before the scripts are told it is in.

import * as THREE from 'three';
import { setupAnimation, wirePointerTracks } from './animation.js';
import { emit } from './api.js';
import { closeDialog } from './dialog.js';
import { environmentLoaded } from './environment.js';
import { formatBytes } from './format.js';
import { applyLighting, applyLightmaps, resetLightmaps } from './lightmaps.js';
import { syncLookdev } from './lookdev.js';
import { current, disposeModel, layout, loader, measureBounds, setModel } from './model.js';
import { loadProbe, releaseProbe, useProbe } from './probe.js';
import { readRenderingPolicy } from './scene.js';
import { glbJsonBytes, measureModel, setSpecs, showSpecs } from './specs.js';
import { findStart, homeView } from './start.js';
import { setStatus } from './status.js';
import { surfaces } from './surfaces.js';

// Loads can outlive the poll interval — a multi-megabyte GLB takes seconds to
// fetch and parse — and setInterval keeps polling meanwhile, so two loads can
// be in flight at once after a quick pair of pushes. Completion order is not
// publish order: without this token the OLDER load can finish last, replace
// the newer model, and label itself live. Each load claims the token at entry;
// a completion that no longer holds it discards its result and stays silent.
let loadGeneration = 0;

export async function load(url, version) {
  const generation = ++loadGeneration;
  let gltf;
  // Fetched and parsed as two steps rather than one `loadAsync`, so each is
  // timed on its own and the file's size is known: "the preview is slow to
  // appear" is a download on a share link and a parse on a big GLB, and the
  // two want different fixes.
  const loadTiming = { bytes: 0, jsonBytes: 0 };
  try {
    const fetching = performance.now();
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`.trim());
    const buffer = await response.arrayBuffer();
    const parsing = performance.now();
    loadTiming.fetchMs = parsing - fetching;
    loadTiming.bytes = buffer.byteLength;
    loadTiming.jsonBytes = glbJsonBytes(buffer);
    gltf = await loader.parseAsync(buffer, THREE.LoaderUtils.extractUrlBase(url));
    loadTiming.parseMs = performance.now() - parsing;
  } catch (error) {
    if (generation !== loadGeneration) return null; // superseded — newer load owns the status line
    setStatus(`load failed: ${error.message || error}`, 'error');
    return false;
  }
  const settingUp = performance.now();
  // The bake's reflection probe (probe.js), decoded off the file's own buffer
  // BEFORE the check below: it awaits, and a load superseded meanwhile must
  // put neither its model nor its probe up.
  const probeIssues = [];
  const probe = await loadProbe(gltf, probeIssues);
  if (generation !== loadGeneration) {
    if (probe) probe.prefiltered.dispose();
    disposeModel(gltf.scene); // parsed but never shown; free it rather than leak it
    return null;
  }
  // A prompt is ALWAYS about the deliverable on screen — what to record, what
  // to write into it — so a push that replaces the model invalidates whatever
  // it was asking. Dismissed rather than left standing with a stale summary
  // over a different scene; it resolves as a cancel, which is the safe answer
  // to a question whose subject has gone. (A recording already RUNNING is
  // dropped by the same event, in `playblast.js`.)
  closeDialog(false);
  setModel(gltf.scene);

  // A lightmapped material already contains its DIFFUSE lighting, so the viewer's own
  // light is what has to give way: a second rig on top of a bake leaves the baked shadows
  // in place while everything around them lifts, which reads as a washed out model rather
  // than as double lighting and is easy to misdiagnose as a bad bake. Only for the BAKED
  // materials though — see applyLighting and patchBakedShader.
  //
  // Reset before applying, not after: these describe the model now on screen, and a load
  // that failed or was superseded above must not leave the previous model's materials
  // being driven by the toggle.
  resetLightmaps();
  // What this load finds wrong with the deliverable, gathered as the passes run
  // (see measureModel). Local, not module state: a newer load can start while
  // this one awaits, and must not clear what this one found.
  const issues = [];
  // Before setupAnimation: the mixer binds a clip's tracks when its action is
  // created, so tracks added afterwards would never be played.
  const animated = await wirePointerTracks(gltf);
  // Read before applying: the deliverable's own policy is what applyLighting
  // then spends, so a load that read nothing must fall back BEFORE the lights
  // are set rather than after.
  readRenderingPolicy(gltf);
  const lightmapped = applyLightmaps(gltf, issues);
  // After the lightmaps: the probe's lookup chains onto the baked shader edit.
  issues.push(...probeIssues);
  if (probe) useProbe(probe, gltf.scene);
  else releaseProbe();
  applyLighting();
  syncLookdev();
  const clipCount = setupAnimation(gltf);

  measureBounds();

  // What the model is and how much of it is lit by its bake, in objects --
  // invisible from the render, where a mostly-baked room and a fully-baked one
  // look alike until the difference is blamed on the baker. See measureModel
  // for the units, and why the count is no longer a fraction of materials.
  loadTiming.setupMs = performance.now() - settingUp;
  const specs = measureModel(gltf, { ...loadTiming, clips: clipCount, animated }, issues);
  const materialCount = specs.materials.instances;
  setSpecs(specs);
  showSpecs(specs);
  // The Environment window's rows and switches, and the probe helpers, for
  // this model.
  environmentLoaded();
  // Laid out, its start found, its surfaces read and the view opened BEFORE
  // the event: a 'load' subscriber is told the model is in, and asking
  // `viewer.headset.start` then answered from the model it replaced -- nothing
  // on the first push, the disposed model's node on the next -- or from this
  // one before it stood on the floor.
  layout();
  findStart();
  surfaces.setRoot(current);
  homeView();
  emit('load', { model: current, gltf, lightmapped, materialCount, clips: clipCount, specs });
  setStatus(
    `v${version} · ${formatBytes(loadTiming.bytes)} · updated ${new Date().toLocaleTimeString()}`,
    'live',
  );
  return true;
}
