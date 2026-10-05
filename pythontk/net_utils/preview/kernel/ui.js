// The page's own controls and keys, the window's size, and the frame loop.

import * as THREE from 'three';
import { advance, mixer, seek, selectClip, setScrubbing, stepClip, togglePlay } from './animation.js';
import { emit } from './api.js';
import { closeDialog, dialogResolve } from './dialog.js';
import { el } from './dom.js';
import { headsetSpace } from './headset.js';
import { applyNormalScale, saveNormalScale } from './lookdev.js';
import { frameCamera } from './model.js';
import { camera, controls, renderer, scene } from './scene.js';
import { stepHeadset } from './session.js';
import { noteFirstFrame } from './specs.js';

el.frameBtn.addEventListener('click', () => frameCamera());
el.normalScale.addEventListener('input', () => applyNormalScale(Number(el.normalScale.value)));
el.normalSave.addEventListener('click', saveNormalScale);
el.clipSelect.addEventListener('change', () => selectClip(Number(el.clipSelect.value)));
el.playToggle.addEventListener('click', togglePlay);
// `input` fires throughout the drag (and on every arrow key), `change` at the
// end of one — which is what hands the playhead back to the frame loop.
el.scrub.addEventListener('input', () => { setScrubbing(true); seek(Number(el.scrub.value) / 1000); });
el.scrub.addEventListener('change', () => { setScrubbing(false); });
addEventListener('keydown', (event) => {
  // A modal takes the keyboard whole, scripts included: Space is a checkbox in
  // there and would otherwise also pause the transport, and a script's own
  // shortcut would act behind the prompt. Escape dismisses it.
  if (dialogResolve) {
    if (event.key === 'Escape') closeDialog(false);
    return;
  }
  // A letter typed into the clip picker is that control's own type-ahead, so
  // the page's single-letter shortcuts must not fire on top of it -- reaching
  // for "(full range)" would otherwise re-frame the camera on the way. Only
  // the picker: the sliders use arrows, which this page has no shortcut for,
  // and excluding them would cost Space-to-pause after a scrub.
  if (event.target === el.clipSelect) return;
  if (event.key === 'f') frameCamera();
  // Guarded on there being a mixer: with no clips these keys mean nothing, and
  // Space would otherwise flip a play state the next animated push inherits.
  if (mixer) {
    if (event.code === 'Space') { event.preventDefault(); togglePlay(); }
    if (event.key === '[') stepClip(-1);
    if (event.key === ']') stepClip(1);
  }
  // Last, so the page's own shortcuts can never be shadowed by a script.
  emit('key', event);
});

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// `delta` rather than a raw timestamp: a script animating anything has to be
// framerate-independent, and a headset runs this loop at 72-120 Hz against a
// desktop's 60.
const frameClock = new THREE.Clock();

// The frame loop's start, run once by the composition root (main.js).
export function startLoop() {
  // `frame` is the XRFrame while a headset is presenting, and absent otherwise.
  renderer.setAnimationLoop((time, frame) => {
    const began = performance.now();
    controls.update();
    const delta = frameClock.getDelta();
    if (frame && headsetSpace) stepHeadset(frame, delta, time);
    // Before the hook, so a script's 'frame' callback sees the model posed for
    // THIS frame rather than the previous one.
    advance(delta);
    emit('frame', { delta, time });
    const rendering = performance.now();
    renderer.render(scene, camera);
    const rendered = performance.now();
    // The first frame after a load is where its programs compile and its
    // textures upload -- the hitch a big deliverable costs before it moves.
    noteFirstFrame(rendered - rendering);
    // After the draw, for a script that measures one (the packaged `inspect`
    // brackets it with a GPU timer query between 'frame' and here). The times
    // are the page's CPU time for the frame and the part of it spent inside
    // `renderer.render` submitting the draw; the GPU works asynchronously, and
    // its own time is only what a query reads back.
    emit('rendered', { delta, time, cpuMs: rendered - began, renderMs: rendered - rendering });
  });
}
