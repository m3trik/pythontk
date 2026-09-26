# !/usr/bin/python
# coding=utf-8
"""Run the WebXR viewer page and assert what it DOES, not how it is spelled.

Every other assertion about ``preview/viewer.html`` is
``assertIn("<a literal line of JS>", page)``, which pins the spelling rather
than the behaviour: reflowing a line fails with no defect, while an inversion
that keeps the same tokens passes green. That is not hypothetical -- the key
light was gated on the wrong condition for part of 2026-08-12 and shipped past
a suite that pinned the inverted line verbatim.

The blocker was always "there is no JS runtime in this workspace". There is one
now: headless Edge, driven through Playwright, running the real page with real
three.js over a real ``PreviewServer``. So these tests load a GLB and ask the
page what it ended up with.

Skipped, never failed, when the runtime is absent -- Playwright is a test-only
tool and the default story stays pure Python:

    python -m pip install playwright     # drives the INSTALLED Edge; no download

This covers the animation transport, the load path, and the lighting policy
-- the last on the pixels the page draws as well as on its materials, because
the policy's baked-material half is a shader edit, and a shader edit that
matches nothing fails silently to every property check.
"""

import json
import math
import os
import pathlib
import struct
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pythontk as ptk  # noqa: E402

#: An ES module that reports what the page ended up with, through the viewer's
#: own documented script seam -- so what is measured is what a real push shows,
#: rather than a probe reaching into internals the page is free to change.
PROBE_JS = """
export default function probe(viewer) {
  const report = { ready: false, errors: [] };
  window.__probe = report;
  window.__api = viewer;
  // The centre of the fixture's one face, in world space.
  const faceCentre = () => {
    const { THREE } = viewer;
    let mesh = null;
    viewer.model.traverse((n) => { if (!mesh && n.isMesh) mesh = n; });
    const position = mesh.geometry.attributes.position;
    const centroid = new THREE.Vector3();
    for (let i = 0; i < position.count; i += 1) {
      centroid.add(new THREE.Vector3().fromBufferAttribute(position, i));
    }
    centroid.divideScalar(position.count);
    mesh.updateWorldMatrix(true, false);
    return centroid.applyMatrix4(mesh.matrixWorld);
  };
  // Display luminance (0-255) at the centre of the fixture's one face, read
  // back off the page's own canvas after a render -- so what is measured is
  // the picture, tone mapping and all, rather than a material property.
  window.__sample = () => {
    const { renderer, scene, camera } = viewer;
    const ndc = faceCentre().project(camera);
    renderer.render(scene, camera);
    const gl = renderer.getContext();
    const x = Math.round((ndc.x + 1) / 2 * gl.drawingBufferWidth);
    const y = Math.round((ndc.y + 1) / 2 * gl.drawingBufferHeight);
    const pixels = new Uint8Array(4 * 9);
    gl.readPixels(x - 1, y - 1, 3, 3, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let sum = 0;
    for (let i = 0; i < 9; i += 1) {
      sum += 0.2126 * pixels[4 * i] + 0.7152 * pixels[4 * i + 1] + 0.0722 * pixels[4 * i + 2];
    }
    return sum / 9;
  };
  // The same face turned to point along `normal` (world) with its +U along
  // `tangent`, then read head-on. The key light stays where the page put it,
  // so this is how a surface facing that way renders under it. The turn rides
  // the pivot, the group every viewer script is handed to spin.
  window.__sampleFacing = (normal, tangent) => {
    const { THREE, camera, pivot } = viewer;
    const n = new THREE.Vector3(...normal).normalize();
    const t = new THREE.Vector3(...tangent).normalize();
    const b = new THREE.Vector3().crossVectors(n, t);
    pivot.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(t, b, n));
    // Clear of the page's floor grid: a turn can leave the face lying in the
    // grid's plane, where the two z-fight, or crossing it edge-on, where a grid
    // line runs through the sample.
    pivot.position.y += 10 - faceCentre().y;
    const centre = faceCentre();
    camera.position.copy(centre).addScaledVector(n, 1);
    camera.up.set(0, 1, 0);
    if (Math.abs(n.y) > 0.99) camera.up.set(0, 0, 1);
    camera.lookAt(centre);
    camera.updateMatrixWorld();
    return window.__sample();
  };
  viewer.on('load', (detail) => {
    try {
      const select = document.getElementById('clipSelect');
      report.openedOn = select
        ? (select.options[select.selectedIndex] || {}).textContent
        : null;
      report.labels = select ? [...select.options].map((o) => o.textContent) : [];
      report.clipCount = detail.clips;
      report.meshes = 0;
      detail.model.traverse((n) => { if (n.isMesh) report.meshes += 1; });
      // What the HUD SAYS, line by line, and the page's own measurement: the
      // wording is the contract here, since the numbers were once right and
      // the sentence still misled.
      report.specs = detail.specs;
      report.statLines = [...document.getElementById('stats').children].map((n) => n.textContent);
      const issues = document.getElementById('issues');
      report.issueLine = issues.hidden ? null : issues.textContent;
      report.playedNamed = viewer.playClip(report.namedTarget || 'MOVES');
      report.playedMissing = viewer.playClip('__nope__');
      report.hasMixer = !!viewer.mixer;

      // The lighting policy, read off the LIVE objects rather than off the
      // page text. Every one of these is a one-line mistake away from a
      // silently wrong render and invisible to a substring assertion.
      report.lightmapped = detail.lightmapped;
      report.materials = [];
      detail.model.traverse((n) => {
        if (!n.isMesh) return;
        for (const m of [].concat(n.material)) {
          report.materials.push({
            name: m.name,
            hasLightMap: !!m.lightMap,
            lightMapChannel: m.lightMap ? m.lightMap.channel : null,
            lightMapSRGB: m.lightMap
              ? m.lightMap.colorSpace === viewer.THREE.SRGBColorSpace
              : null,
            lightMapIntensity: m.lightMapIntensity,
            hasAoMap: !!m.aoMap,
            hasNormalMap: !!m.normalMap,
            // The shader patch a baked material declares through its program
            // cache key: 'baked' (environment specular only) or 'baked-relief'
            // (that, plus the normal-map relief). An unpatched material
            // reports the base class's key, which is neither.
            programKey: m.customProgramCacheKey(),
          });
        }
      });
      // The lookdev area and the dial inside it, read separately. The area
      // ships OFF (LOOKDEV_ENABLED), and on this fixture the dial itself would
      // otherwise be showing -- so the pair is what distinguishes "the gate is
      // closed" from "this model gave the dial nothing to do".
      const lookdev = document.getElementById('lookdev');
      const normals = document.getElementById('normals');
      report.lookdevHidden = !lookdev || lookdev.hidden;
      report.normalsHidden = !normals || normals.hidden;

      report.lights = [];
      viewer.scene.traverse((n) => {
        if (n.isLight) {
          report.lights.push({ type: n.type, intensity: n.intensity });
        }
      });
      // The session's environment, on the SCENE: it lights every un-baked
      // material and is what every baked one reflects, and a second push
      // must not take it away (see disposeModel).
      report.environment = {
        present: !!viewer.scene.environment,
        intensity: viewer.scene.environmentIntensity,
      };
      // Where a session would start, as a 'load' subscriber is told it -- one
      // entry per load: the model just loaded, laid out on the floor, never
      // the one it replaced.
      const start = viewer.headset.start;
      report.startsAtLoad = (report.startsAtLoad || []).concat([start && start.point.toArray()]);
      // One entry per model swap, so a test can assert what the SECOND push
      // rendered. `disposeModel` frees the outgoing model's textures; a
      // second push rendering unlit is a failure no first-push check can see.
      report.loads = (report.loads || 0) + 1;
      report.ready = true;
    } catch (error) {
      report.errors.push(String(error));
      report.ready = true;
    }
  });
}
"""


#: Drives the SEQUENCE transport: selects the synthetic whole-timeline entry a
#: shots-only deliverable grows, then scrubs it and reads the pose back off the
#: model. The scrub is dispatched on the real slider rather than by calling an
#: internal, so what is measured is the control a reviewer drags.
SEQUENCE_PROBE = """
export default function probe(viewer) {
  const report = { ready: false, errors: [] };
  window.__probe = report;
  viewer.on('load', (detail) => {
    try {
      const select = document.getElementById('clipSelect');
      report.labels = [...select.options].map((o) => o.textContent);
      report.selected = viewer.playClip('FULL SEQUENCE');
      report.openedOn = (select.options[select.selectedIndex] || {}).textContent;

      const cube = detail.model.getObjectByName('cube');
      const scrub = document.getElementById('scrub');
      const at = (fraction) => {
        scrub.value = String(Math.round(fraction * 1000));
        scrub.dispatchEvent(new Event('input'));
        return {
          y: Number(cube.position.y.toFixed(4)),
          readout: document.getElementById('clipTime').textContent,
          // Read AFTER the pose, which is the contract: it answers for where
          // the model is now, not for where it is about to be.
          description: viewer.descriptionAt(),
        };
      };
      report.start = at(0);
      report.gap = at(2.5 / 5.0);    // between the shots
      report.inShotB = at(4.0 / 5.0);  // one second into the second shot
      report.end = at(1);
      report.ready = true;
    } catch (error) {
      report.errors.push(String(error));
      report.ready = true;
    }
  });
}
"""


#: Selects a clip by name and reports what the page then says is selected, for
#: the recording tests: the movie's length is asserted against the clip's own
#: frame range, so the range has to be read from the page rather than assumed.
RECORD_PROBE = """
export default function probe(viewer) {
  const report = { ready: false, errors: [] };
  window.__probe = report;
  window.__select = (name) => {
    const ok = viewer.playClip(name);
    return { ok, clip: viewer.clip };
  };
  // Read before and after a recording: a preset larger than the view raises
  // the pixel ratio for the capture, and it has to be back afterwards.
  window.__pixelRatio = () => viewer.renderer.getPixelRatio();
  viewer.on('load', () => {
    report.buttons = [...document.querySelectorAll('#controls button')]
      .map((b) => b.textContent);
    report.ready = true;
  });
}
"""


#: Same probe, with Worker taken away: the playblast script reads its
#: capabilities when a recording STARTS, so removing it here -- at page init,
#: long before the button is pressed -- is what a browser without workers looks
#: like from the script's point of view.
NO_WORKER_PROBE = RECORD_PROBE.replace(
    "export default function probe(viewer) {",
    "export default function probe(viewer) {\n  window.Worker = undefined;",
).replace(
    "report.ready = true;",
    # Reported so the test can tell the fallback actually ran. Without it, an
    # assignment that silently failed would leave the pool in play and the test
    # would pass while proving nothing.
    "report.workerType = typeof Worker;\n    report.ready = true;",
)


def bake_face_level(irradiance, albedo=1.0):
    """The 8-bit display level a Lambert face lit by *irradiance* alone lands on.

    three.js adds a lightmap through ``BRDF_Lambert`` (``irradiance * albedo /
    pi``), tone-maps with ACES filmic at exposure 1 and writes sRGB -- all
    three pinned by the published rendering policy. On a grey value the ACES
    fit's input and output matrices are the identity, so the transfer is the
    fit alone, ``fit(x / 0.6)``, then the sRGB curve.
    """
    v = irradiance * albedo / math.pi / 0.6
    fit = (v * (v + 0.0245786) - 0.000090537) / (
        v * (0.983729 * v + 0.4329510) + 0.238081
    )
    fit = min(max(fit, 0.0), 1.0)
    srgb = 12.92 * fit if fit <= 0.0031308 else 1.055 * fit ** (1 / 2.4) - 0.055
    return 255.0 * srgb


def normal_texel_png(degrees):
    """A 1x1 tangent-space normal map leaning *degrees* toward +U, as a data URI.

    0 is a FLAT map. The lean rides red alone, so it tips the normal along the
    fixture's U direction whichever convention the map's green follows.
    """
    import base64

    import cv2
    import numpy as np

    lean = math.radians(degrees)
    rgb = [round((c * 0.5 + 0.5) * 255) for c in (math.sin(lean), 0.0, math.cos(lean))]
    ok, png = cv2.imencode(".png", np.array([[rgb[::-1]]], dtype=np.uint8))
    assert ok, "cv2 could not encode the normal texel"
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


def _unit(vector):
    length = math.sqrt(sum(c * c for c in vector))
    return tuple(c / length for c in vector)


def _yaw_pitch(yaw_deg, pitch_deg):
    """The glTF quaternion ``[x, y, z, w]`` of a yaw about +y, then a pitch.

    R_y(yaw) * R_x(pitch): how both DCCs' cameras arrive in the GLB (a Maya
    camera and a Blender one aimed alike come out as the same quaternion).
    """
    sy, cy = math.sin(math.radians(yaw_deg) / 2), math.cos(math.radians(yaw_deg) / 2)
    sx, cx = (
        math.sin(math.radians(pitch_deg) / 2),
        math.cos(math.radians(pitch_deg) / 2),
    )
    return [cy * sx, sy * cx, -sy * sx, cy * cx]


def _floor(x0, x1, z0, z1, y):
    """Two triangles covering ``[x0, x1] x [z0, z1]`` at height *y*, facing +y
    (counter-clockwise from above)."""
    return [(x0, y, z0), (x0, y, z1), (x1, y, z0),
            (x1, y, z0), (x0, y, z1), (x1, y, z1)]  # fmt: skip


#: Drives the headset through the page's own frame, `headset.step(delta,
#: input)` -- the step the page runs from each XRFrame -- on synthetic input
#: over the walkable fixture (`_walkable_glb`), and reports what the rig did.
#: Poses are in the TRACKED space, as the headset reports them; a stick's y is
#: negative pushed forward. One page for every scenario, each a fresh session
#: (`headset.begin()`).
LOCOMOTION_JS = """
() => {
  const api = window.__api;
  const H = api.headset;
  const R = api.rig;
  const L = api.locomotion;
  const THREE = api.THREE;
  const DT = 1 / 72;
  const out = {};
  // Yaw (about +y, positive turning left) then pitch (positive looking up).
  const quat = (yawDeg, pitchDeg = 0) => new THREE.Quaternion()
    .setFromEuler(new THREE.Euler(
      THREE.MathUtils.degToRad(pitchDeg), THREE.MathUtils.degToRad(yawDeg), 0, 'YXZ'))
    .toArray();
  const run = (frames, input) => {
    const events = [];
    for (let i = 0; i < frames; i++) events.push(...H.step(DT, input));
    return events;
  };
  const rig = () => R.place;
  const inScene = (pose) => R.toScene(pose.position).toArray();
  const boxOf = (object) => {
    const box = new THREE.Box3().setFromObject(object);
    return [box.min.toArray(), box.max.toArray()];
  };

  // --- where the model stands: true scale, centred, on the floor ---------
  out.layout = {
    scaleToggle: !!document.getElementById('scaleToggle'),
    scale: api.model.scale.toArray(),
    box: boxOf(api.model),
  };
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'r' }));
  out.layout.boxAfterR = boxOf(api.model);

  // --- snap turn: one per flick, about the head ---------------------------
  H.begin();
  const head = { position: [0.4, 1.6, -0.3], orientation: quat(0) };
  const before = inScene(head);
  out.turn = { events: run(20, { head, right: [1, 0] }) };
  out.turn.yaw = rig().yaw;
  out.turn.headBefore = before;
  out.turn.headAfter = inScene(head);
  run(3, { head, right: [0, 0] });
  run(3, { head, right: [-1, 0] });
  out.turn.yawBack = rig().yaw;

  // --- a jump onto the floor ahead ---------------------------------------
  H.begin();
  const aim = { position: [0.2, 1.2, -0.3], orientation: quat(0, -35) };
  out.teleport = { aimEvents: run(3, { head, right: [0, -1], aim }) };
  out.teleport.arcPoints = L.arc.length;
  out.teleport.target = L.target && { point: L.target.point.toArray(), valid: L.target.valid };
  const released = run(2, { head, right: [0, 0], aim });
  out.teleport.fadeWhileLeaving = R.fade;
  out.teleport.rigWhileLeaving = rig();
  out.teleport.events = released.concat(run(40, { head, right: [0, 0], aim }));
  out.teleport.headAfter = inScene(head);
  out.teleport.rigAfter = rig();
  out.teleport.fadeAfter = R.fade;

  // --- a jump onto the raised platform: the first pitch whose arc lands on it
  H.begin();
  const east = { position: [0, 1.6, 0], orientation: quat(-90) };
  let onPlatform = null;
  for (let pitch = -60; pitch <= 30 && !onPlatform; pitch += 1) {
    const ray = { position: [0, 1.2, 0], orientation: quat(-90, pitch) };
    run(1, { head: east, right: [0, -1], aim: ray });
    const found = L.target;
    if (found && found.valid && Math.abs(found.point.y - 0.3) < 1e-3) {
      onPlatform = { ray, point: found.point.toArray() };
    }
    run(1, { head: east, right: [0, 0.9], aim: ray });  // pulled back: cancelled
  }
  out.platform = { found: !!onPlatform };
  if (onPlatform) {
    run(2, { head: east, right: [0, -1], aim: onPlatform.ray });
    run(40, { head: east, right: [0, 0], aim: onPlatform.ray });
    out.platform.point = onPlatform.point;
    out.platform.rig = rig();
    out.platform.headAfter = inScene(east);
  }

  // --- the wall: struck, and no landing ----------------------------------
  H.begin();
  const nearWall = { position: [0, 1.6, -2], orientation: quat(0) };
  const atWall = { position: [0, 1.2, -2.1], orientation: quat(0, 25) };
  run(3, { head: nearWall, right: [0, -1], aim: atWall });
  out.wall = { target: L.target && { point: L.target.point.toArray(), valid: L.target.valid } };
  run(40, { head: nearWall, right: [0, 0], aim: atWall });
  out.wall.rig = rig();

  // --- pulled back: cancelled, no jump -----------------------------------
  H.begin();
  run(3, { head, right: [0, -1], aim });
  out.cancel = { aimedValid: !!(L.target && L.target.valid) };
  run(1, { head, right: [0, 0.8], aim });
  out.cancel.aimingAfterPullBack = L.aiming;
  run(40, { head, right: [0, 0], aim });
  out.cancel.rig = rig();

  // --- walking where the head looks: +z, clear of the step and the ledge ---
  H.begin();
  const south = { position: [0, 1.6, 0], orientation: quat(180) };
  run(1, { head: south, left: [0, -1] });
  out.walk = { firstStep: rig().z };
  run(71, { head: south, left: [0, -1] });
  out.walk.oneSecond = rig();
  out.walk.vignetteMoving = L.vignette;
  run(36, { head: south, left: [0, 0] });
  out.walk.stopped = rig();
  run(72, { head: south, left: [0, 0] });
  out.walk.settled = rig();
  out.walk.vignetteStopped = L.vignette;

  // --- the floor follows a step up, not a ledge --------------------------
  H.begin();
  run(100, { head: east, left: [0, -1] });
  run(60, { head: east, left: [0, 0] });
  out.stepUp = rig();
  H.begin();
  const towardLedge = { position: [-3, 1.6, 0], orientation: quat(180) };
  run(100, { head: towardLedge, left: [0, -1] });
  run(60, { head: towardLedge, left: [0, 0] });
  out.ledge = { rig: rig(), head: inScene(towardLedge) };

  // --- the offset reference space is the rig's inverse --------------------
  H.begin();
  run(3, { head, right: [1, 0] });
  run(3, { head, right: [0, 0] });
  run(40, { head: south, left: [0.5, -1] });
  const off = R.offset();
  const rotation = new THREE.Quaternion(
    off.orientation.x, off.orientation.y, off.orientation.z, off.orientation.w);
  const tracked = [0.7, 1.3, -0.4];
  // WebXR: a pose in the offset space is the offset's INVERSE applied to the
  // tracked one -- which three.js then treats as the scene.
  out.offset = {
    viaOffset: new THREE.Vector3(...tracked)
      .sub(new THREE.Vector3(off.position.x, off.position.y, off.position.z))
      .applyQuaternion(rotation.clone().invert())
      .toArray(),
    toScene: R.toScene(tracked).toArray(),
  };

  // --- a pivot turned since load: the wall is struck where it NOW stands --
  H.begin();
  api.pivot.rotation.y = Math.PI;  // the wall at z -5 now stands at z +5, facing -z
  const turnedHead = { position: [0, 1.6, 2], orientation: quat(180) };
  const turnedAim = { position: [0, 1.2, 2.1], orientation: quat(180, 25) };
  run(3, { head: turnedHead, right: [0, -1], aim: turnedAim });
  out.turnedPivot = { target: L.target && { point: L.target.point.toArray(), valid: L.target.valid } };
  run(1, { head: turnedHead, right: [0, 0.9], aim: turnedAim });
  api.pivot.rotation.y = 0;
  H.begin();
  return out;
}
"""

#: Where a view starts, over the walkable fixture with its start nodes
#: (`_walkable_glb(start=True)`): the desktop view the load opened on, the
#: start the rig reads, and a session's first frames through `update`.
START_JS = """
() => {
  const api = window.__api;
  const H = api.headset;
  const R = api.rig;
  const THREE = api.THREE;
  const DT = 1 / 72;
  const quat = (yawDeg, pitchDeg = 0) => new THREE.Quaternion()
    .setFromEuler(new THREE.Euler(
      THREE.MathUtils.degToRad(pitchDeg), THREE.MathUtils.degToRad(yawDeg), 0, 'YXZ'))
    .toArray();
  const out = {};
  out.desktop = {
    position: api.camera.position.toArray(),
    look: api.camera.getWorldDirection(new THREE.Vector3()).toArray(),
    target: api.controls.target.toArray(),
    fov: api.camera.fov,
    aspect: api.camera.aspect,
    near: api.camera.near,
    far: api.camera.far,
  };
  const start = H.start;
  out.start = start && { point: start.point.toArray(), heading: start.heading };

  // A session's first frame with a head: placed at the start, under black.
  H.begin();
  R.takeChange();  // the begin's own reset, applied
  const head = { position: [0.4, 1.6, -0.3], orientation: quat(30, 10) };
  out.arrive = { events: H.step(DT, { head }), fade: R.fade, changed: R.changed };
  out.arrive.rig = R.place;
  out.arrive.head = R.toScene(head.position).toArray();
  out.arrive.facing = R.directionToScene(head.orientation).toArray();
  const later = [];
  for (let i = 0; i < 30; i++) later.push(...H.step(DT, { head }));
  out.arrive.later = later;
  out.arrive.fadeAfter = R.fade;
  out.arrive.rigAfter = R.place;
  // Re-armed by the next session's begin, and only by it.
  H.begin();
  out.again = H.step(DT, { head });

  // A recenter -- the tracked space reset under the viewer, who may be
  // anywhere in their room, facing anywhere -- while still at the start: put
  // back there, facing its way. Once they have walked off, it keeps them.
  for (let i = 0; i < 20; i++) H.step(DT, { head });
  const elsewhere = { position: [1.2, 1.6, 0.9], orientation: quat(-70) };
  H.recenter();
  out.recenter = { events: H.step(DT, { head: elsewhere }) };
  out.recenter.head = R.toScene(elsewhere.position).toArray();
  out.recenter.facing = R.directionToScene(elsewhere.orientation).toArray();
  for (let i = 0; i < 20; i++) H.step(DT, { head: elsewhere });
  for (let i = 0; i < 36; i++) H.step(DT, { head: elsewhere, left: [0, -1] });
  H.recenter();
  out.recenter.afterWalking = H.step(DT, { head: elsewhere });
  H.begin();
  return out;
}
"""

#: A session's first frame on a page whose manifest names no start node; then a
#: walk, and locomotion switched off under it (restored before returning).
NO_START_JS = """
() => {
  const H = window.__api.headset;
  const R = window.__api.rig;
  const L = window.__api.locomotion;
  H.begin();
  const head = { position: [0.4, 1.6, -0.3], orientation: [0, 0, 0, 1] };
  const events = H.step(1 / 72, { head });
  const out = { events, rig: R.place, fade: R.fade, start: H.start };
  for (let i = 0; i < 72; i++) H.step(1 / 72, { head, left: [0, -1] });
  out.walked = R.place;
  L.enabled = false;
  out.switchedOff = { events: H.step(1 / 72, { head }), rig: R.place, fade: R.fade };
  L.enabled = true;
  H.begin();
  return out;
}
"""

#: A session with locomotion switched off (the server's switch, read on the
#: poll): placed at the start, then every stick every way -- a walk, a flick
#: each way, an aim and its release -- and a recenter from across the room.
#: Then the switch thrown MID-session, on a viewer the sticks had carried off
#: while it was still on. Synchronous throughout, so no poll can land between
#: the scenario's own writes to `enabled`; it ends switched off, as it began.
LOCKED_JS = """
() => {
  const api = window.__api;
  const H = api.headset;
  const R = api.rig;
  const L = api.locomotion;
  const THREE = api.THREE;
  const DT = 1 / 72;
  const quat = (yawDeg, pitchDeg = 0) => new THREE.Quaternion()
    .setFromEuler(new THREE.Euler(
      THREE.MathUtils.degToRad(pitchDeg), THREE.MathUtils.degToRad(yawDeg), 0, 'YXZ'))
    .toArray();
  const head = { position: [0.4, 1.6, -0.3], orientation: quat(30, 10) };
  const aim = { position: [0.2, 1.2, -0.3], orientation: quat(30, -35) };
  const across = { position: [-0.8, 1.6, 1.1], orientation: quat(-120) };
  const out = { enabled: L.enabled };
  H.begin();
  out.arrive = H.step(DT, { head });
  for (let i = 0; i < 20; i++) H.step(DT, { head });
  out.placed = R.place;
  const events = [];
  for (let i = 0; i < 72; i++) events.push(...H.step(DT, { head, left: [0.3, -1] }));
  for (const right of [[1, 0], [0, 0], [-1, 0], [0, 0], [0, -1], [0, -1], [0, 0]]) {
    events.push(...H.step(DT, { head, right, aim }));
  }
  for (let i = 0; i < 40; i++) events.push(...H.step(DT, { head }));
  out.events = events;
  out.after = R.place;
  out.aiming = L.aiming;
  out.arc = L.arc.length;
  out.vignette = L.vignette;
  out.head = R.toScene(head.position).toArray();
  H.recenter();
  out.recenter = H.step(DT, { head: across });
  out.recenterHead = R.toScene(across.position).toArray();
  out.recenterFacing = R.directionToScene(across.orientation).toArray();

  // On while the viewer walks off, then switched off (as the next poll does).
  L.enabled = true;
  H.begin();
  for (let i = 0; i < 21; i++) H.step(DT, { head });
  const mid = { placed: R.place };
  for (let i = 0; i < 72; i++) H.step(DT, { head, left: [0, -1] });
  mid.walked = R.place;
  L.enabled = false;
  mid.events = H.step(DT, { head });
  mid.fade = R.fade;
  mid.head = R.toScene(head.position).toArray();
  mid.later = [];
  for (let i = 0; i < 30; i++) mid.later.push(...H.step(DT, { head }));
  mid.fadeAfter = R.fade;
  H.recenter();
  mid.recenter = H.step(DT, { head: across });
  mid.recenterHead = R.toScene(across.position).toArray();
  out.midSession = mid;
  H.begin();
  return out;
}
"""

#: Two headset sessions over a large ground (`_ground_glb`), begun and ended
#: the way WebXRManager does it -- `isPresenting` set, THEN the event, and the
#: reverse at the end -- on a stand-in for the session's reference space, which
#: is all of a session the page itself touches. Between the two events, what
#: three.js does to the page's camera on every XR frame is done by hand
#: (`WebXRManager.updateUserCamera`: the head's pose, the headset's lens and its
#: projection written in, with the page's loop running `controls.update()` on
#: top). The second session re-frames from the desktop half-way -- the Frame
#: button, or a push landing mid-session.
SESSION_JS = """
() => {
  const api = window.__api;
  const { THREE, camera, controls, renderer } = api;
  const xr = renderer.xr;
  const H = api.headset;
  const R = api.rig;
  const view = () => ({
    position: camera.position.toArray(),
    quaternion: camera.quaternion.toArray(),
    fov: camera.fov,
    near: camera.near,
    far: camera.far,
    target: controls.target.toArray(),
    // The projection as drawn: a lens put back but never rebuilt into the
    // matrix would go on drawing the headset's.
    focal: camera.projectionMatrix.elements[5],
  });
  const heard = [];
  const space = { addEventListener: (type, fn) => heard.push([type, fn]), removeEventListener() {} };
  const ownSpace = xr.getReferenceSpace;
  const begin = () => {
    xr.getReferenceSpace = () => space;
    xr.isPresenting = true;
    xr.dispatchEvent({ type: 'sessionstart' });
  };
  const finish = () => {
    xr.isPresenting = false;
    xr.dispatchEvent({ type: 'sessionend' });
    xr.getReferenceSpace = ownSpace;
  };
  const headsetFrame = () => {
    camera.position.set(4, 1.6, -3);
    camera.quaternion.setFromEuler(new THREE.Euler(0.1, 2.0, 0));
    camera.fov = 101;
    camera.projectionMatrix.makePerspective(-0.05, 0.05, 0.05, -0.05, 0.05, 100);
    controls.update();
  };
  const layerShown = () => {
    const card = api.scene.getObjectByName('controls');
    return card ? card.parent.visible : null;
  };
  const press = (key) => window.dispatchEvent(new KeyboardEvent('keydown', { key }));
  const out = {};
  press('f');
  out.framed = view();
  // The desktop view orbited somewhere of the user's own before entering.
  camera.position.set(60, 45, 90);
  controls.target.set(10, 0, -10);
  controls.update();
  out.orbited = view();
  R.move(2, 3);  // a rig some earlier use left off the origin

  begin();
  out.started = {
    near: camera.near,
    place: R.place,
    heard: heard.map(([type]) => type),
    recenters: heard.length === 1 && heard[0][1] === H.recenter,
    layer: layerShown(),
  };
  headsetFrame();
  headsetFrame();
  finish();
  out.ended = { view: view(), layer: layerShown(), place: R.place };

  begin();
  headsetFrame();
  press('f');
  out.reframed = { near: camera.near };
  headsetFrame();
  finish();
  out.reframed.view = view();
  return out;
}
"""

#: A dense rolling ground (`_terrain_glb`) walked and aimed across, counting
#: the triangle tests each headset frame costs. Every triangle a raycast tests
#: goes through `Ray.intersectTriangle` -- three.js's own raycast and anything
#: faster alike -- so the count is the work, independent of how it is found.
#: The heights are checked against three.js's own raycast straight down.
DENSE_JS = """
() => {
  const api = window.__api;
  const { THREE } = api;
  const H = api.headset;
  const R = api.rig;
  const L = api.locomotion;
  const DT = 1 / 72;
  const FRAMES = 60;
  const quat = (yawDeg, pitchDeg = 0) => new THREE.Quaternion()
    .setFromEuler(new THREE.Euler(
      THREE.MathUtils.degToRad(pitchDeg), THREE.MathUtils.degToRad(yawDeg), 0, 'YXZ'))
    .toArray();
  const ground = (x, z) => {
    const down = new THREE.Raycaster(new THREE.Vector3(x, 50, z), new THREE.Vector3(0, -1, 0));
    const hit = down.intersectObject(api.model, true)[0];
    return hit ? hit.point.y : null;
  };
  const proto = THREE.Ray.prototype;
  const stock = proto.intersectTriangle;
  let tests = 0;
  proto.intersectTriangle = function (...args) {
    tests += 1;
    return stock.apply(this, args);
  };
  const out = {};
  try {
    const head = { position: [0, 1.6, 0], orientation: quat(0) };
    H.begin();
    H.step(DT, { head });
    tests = 0;
    let began = performance.now();
    for (let i = 0; i < FRAMES; i++) H.step(DT, { head, left: [0, -1] });
    out.walk = { tests: tests / FRAMES, ms: (performance.now() - began) / FRAMES };
    for (let i = 0; i < 120; i++) H.step(DT, { head });  // coasted to a stop, settled
    const at = R.toScene(head.position);
    out.walk.rig = R.place;
    out.walk.ground = ground(at.x, at.z);

    H.begin();
    H.step(DT, { head });
    tests = 0;
    began = performance.now();
    let aim = null;
    for (let i = 0; i < FRAMES; i++) {  // swept slowly: a new arc every frame
      aim = { position: [0.2, 1.2, -0.3], orientation: quat(15, -30 + i * 0.25) };
      H.step(DT, { head, right: [0, -1], aim });
    }
    out.aim = { tests: tests / FRAMES, ms: (performance.now() - began) / FRAMES };
    const target = L.target;
    out.aim.target = target && { point: target.point.toArray(), valid: target.valid };
    if (target) out.aim.ground = ground(target.point.x, target.point.z);
    H.step(DT, { head, right: [0, 0.9], aim });  // pulled back: cancelled
  } finally {
    proto.intersectTriangle = stock;
    H.begin();
  }
  return out;
}
"""

#: The packaged `turntable` on the desktop and while a headset presents --
#: `renderer.xr.isPresenting` is a plain flag, and the script reads nothing
#: else of a session -- over real frames.
TURNTABLE_JS = """
async () => {
  const api = window.__api;
  const xr = api.renderer.xr;
  const turned = async () => {
    const from = api.pivot.rotation.y;
    await new Promise((resolve) => setTimeout(resolve, 500));
    return api.pivot.rotation.y - from;
  };
  const out = { desktop: await turned() };
  xr.isPresenting = true;
  try {
    out.presenting = await turned();
  } finally {
    xr.isPresenting = false;
  }
  out.after = await turned();
  return out;
}
"""

#: The packaged `inspect` in a session: B on the right controller -- button 5
#: of an xr-standard gamepad -- pressed, held and pressed again, over real
#: frames, through a stand-in session holding just that controller.
INSPECT_XR_JS = """
async () => {
  const api = window.__api;
  const xr = api.renderer.xr;
  const frames = (count) => new Promise((resolve) => {
    let left = count;
    const tick = () => ((left -= 1) <= 0 ? resolve() : requestAnimationFrame(tick));
    requestAnimationFrame(tick);
  });
  const b = { pressed: false };
  const session = {
    frameRate: 90,
    inputSources: [{ handedness: 'right', gamepad: { buttons: [{}, {}, {}, {}, {}, b], axes: [] } }],
  };
  const panel = [...document.querySelectorAll('#panels .panel')]
    .find((node) => node.querySelector('header').textContent === 'Inspect');
  const state = () => {
    const card = api.scene.getObjectByName('inspect');
    return { panel: !panel.hidden, card: card ? card.visible : null };
  };
  const ownSession = xr.getSession;
  xr.getSession = () => session;
  xr.isPresenting = true;
  const out = {};
  try {
    await frames(3);
    out.before = state();
    b.pressed = true;
    await frames(3);
    out.pressed = state();
    await frames(5);
    out.held = state();
    b.pressed = false;
    await frames(3);
    b.pressed = true;
    await frames(3);
    out.again = state();
    b.pressed = false;
  } finally {
    xr.isPresenting = false;
    xr.getSession = ownSession;
  }
  await frames(2);
  out.desktop = state();
  return out;
}
"""

#: `headset.read`, the page's XRFrame reader, on stand-in frames: the head and
#: the right controller's aim in the session's space, and each stick from the
#: axes the controller reports it on.
READ_JS = """
() => {
  const H = window.__api.headset;
  const AIM = {};
  const pose = ([x, y, z]) => ({
    transform: { position: { x, y, z }, orientation: { x: 0, y: 0, z: 0, w: 1 } },
  });
  const frame = (sources) => ({
    session: { inputSources: sources },
    getViewerPose: () => pose([0.1, 1.6, 0.2]),
    getPose: (space) => (space === AIM ? pose([0.3, 1.2, -0.2]) : null),
  });
  const source = (handedness, axes) => ({ handedness, targetRaySpace: AIM, gamepad: { axes, buttons: [] } });
  const plain = (input) => ({
    head: input.head,
    left: input.left,
    right: input.right,
    aim: input.aim,
    hands: Object.keys(input.sources).sort(),
  });
  return {
    standard: plain(H.read(frame([
      source('left', [0.1, 0.2, 0.3, -0.4]),
      source('right', [0.5, 0.6, -0.7, 0.8]),
    ]))),
    touchpad: plain(H.read(frame([source('left', [0.25, -0.5])]))),
    unhanded: plain(H.read(frame([source('none', [1, 1, 1, 1]), source('right', [0.9])]))),
  };
}
"""

#: The model on screen, read after a push: its world box, the page's own
#: measurement of it, and the clip planes the view opened with.
PLACED_JS = """
() => {
  const api = window.__api;
  const { THREE, camera } = api;
  const box = new THREE.Box3().setFromObject(api.model);
  return {
    centre: box.getCenter(new THREE.Vector3()).toArray(),
    min: box.min.toArray(),
    size: api.specs.size,
    position: camera.position.toArray(),
    near: camera.near,
    far: camera.far,
    finite: [camera.near, camera.far, ...camera.projectionMatrix.elements].every(Number.isFinite),
  };
}
"""


def _runtime_available():
    """Playwright installed AND an Edge/Chrome channel it can drive."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            browser.close()
        return True
    except Exception:  # noqa: BLE001 — no browser is a skip, not a failure
        return False


@unittest.skipUnless(
    _runtime_available(), "needs playwright + an installed Edge/Chrome channel"
)
class TestPreviewViewerLive(unittest.TestCase):
    """The page, loaded and interrogated."""

    FPS = 30.0

    @classmethod
    def setUpClass(cls):
        cls.temp = ptk.TempArtifacts("preview_viewer_live", policy="scoped")
        cls.probe = cls.temp.path(extension=".js")
        with open(cls.probe, "w", encoding="utf-8") as fh:
            fh.write(PROBE_JS)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    # ------------------------------------------------------------------ fixture
    def _animated_glb(self):
        """A GLB whose FIRST declared shot is empty and whose second plays.

        The measured production shape: Maya's split emits an AnimStack per
        declared range but bakes no curve for a range in which nothing moves.
        """
        accessors = [{"type": "SCALAR", "min": [0.0], "max": [2.0]}]
        gltf = {
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [0]}],
            "scene": 0,
            "nodes": [{"name": "cube", "mesh": 0}, {"name": "data_export"}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 1}}]}],
            "accessors": accessors,
            "animations": [
                {"name": "HOLDS", "samplers": [], "channels": []},
                {
                    "name": "MOVES",
                    "samplers": [{"input": 0, "output": 2}],
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "translation"}}
                    ],
                },
            ],
        }
        takes = [
            {"name": "HOLDS", "start": 1, "end": 10},
            {"name": "MOVES", "start": 20, "end": 40},
        ]
        gltf["nodes"][1]["extras"] = {
            "fromFBX": {
                "userProperties": {
                    "fbx_takes": {"type": "eFbxString", "value": json.dumps(takes)},
                    "shot_metadata": {
                        "type": "eFbxString",
                        "value": json.dumps({"version": 1, "fps": self.FPS}),
                    },
                }
            }
        }
        path = self._write(gltf)
        # What the real conversion always does, and what the page reads: without
        # the block the viewer has only the file's raw animation ORDER to go on,
        # which is the very thing the block exists to make answerable.
        ptk.MeshConvert.apply_glb_animations(path)
        return path

    #: The note the DCC's Shots panel carries for SHOT_B, long enough that a
    #: burn-in of it cannot be mistaken for the shot's name.
    SHOT_B_DESCRIPTION = "hero turns to camera and holds"

    #: A description WIDER than the recorded frame. Prose is what a shot
    #: description is, so this is a shape the burn-in has to survive rather
    #: than an abuse of it: the text scales with the frame, so a line holds
    #: roughly 90 characters of monospace at any preset (2560px at 46px, 1280px
    #: at 23px), and this is well past that.
    LONG_DESCRIPTION = (
        "hero turns to camera, holds for the reaction, then walks out of "
        "frame left while the crowd behind him keeps moving — note the "
        "hand-off on the last eight frames, it is still the old timing"
    )

    def _shots_only_glb(
        self,
        empty_tail=False,
        with_sequence=False,
        half_take=False,
        described=False,
    ):
        """A deliverable that ships ONLY shots, with a GAP between them.

        *empty_tail* adds a third declared shot that bakes no curve -- the
        production shape for a range in which nothing moves.

        *described* gives SHOT_B the description a shot record carries and
        leaves SHOT_A without one -- the mixed case, which is the only one that
        can show a burn-in skipping the shots that state nothing. Pass a string
        to choose the note rather than take the default one.

        What ``Animation Clips: Shots Only`` writes. The two shots sit 30
        frames apart, so a sequence that concatenated them would run 4s where
        the timeline is 5s and put the second shot a second early -- the gap is
        the thing that makes placement testable.
        """
        gltf = {
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [0]}],
            "scene": 0,
            "nodes": [{"name": "cube", "mesh": 0}, {"name": "data_export"}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 1}}]}],
            "accessors": [{"type": "SCALAR", "min": [0.0], "max": [2.0]}],
            "animations": [
                {
                    "name": name,
                    "samplers": [{"input": 0, "output": 2}],
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "translation"}}
                    ],
                }
                for name in ("SHOT_A", "SHOT_B")
            ],
        }
        takes = [
            {"name": "SHOT_A", "start": 0, "end": 60},
            {"name": "SHOT_B", "start": 90, "end": 150},
        ]
        if empty_tail:
            gltf["animations"].append(
                {"name": "SHOT_C", "samplers": [], "channels": []}
            )
            takes.append({"name": "SHOT_C", "start": 180, "end": 300})
        if half_take:
            # A take the carrier declares with no "end". The manifest builds
            # its clip rows with `take.get("start")` / `take.get("end")`, so
            # this reaches the page as start_frame=210, end_frame=null -- the
            # shape a DCC writing a partial take produces.
            gltf["animations"].append(
                {
                    "name": "SHOT_D",
                    "samplers": [{"input": 0, "output": 2}],
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "translation"}}
                    ],
                }
            )
            takes.append({"name": "SHOT_D", "start": 210})
        metadata = {"version": 1, "fps": self.FPS}
        if described:
            # Keyed by CLIP, which is how `apply_glb_animations` joins a shot
            # record to the animation it became. SHOT_A is left out: a shot
            # with no note is the common case, and the burn-in has to record it
            # without holding an empty line open for one.
            note = described if isinstance(described, str) else self.SHOT_B_DESCRIPTION
            metadata["shots"] = [{"clip": "SHOT_B", "description": note}]
        if with_sequence:
            # An UNDECLARED clip: the whole-timeline stack "Shots + Full
            # Sequence" keeps beside the shots.
            gltf["animations"].append(
                {
                    "name": "FULL_SEQUENCE",
                    "samplers": [{"input": 0, "output": 2}],
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "translation"}}
                    ],
                }
            )
        gltf["nodes"][1]["extras"] = {
            "fromFBX": {
                "userProperties": {
                    "fbx_takes": {"type": "eFbxString", "value": json.dumps(takes)},
                    "shot_metadata": {
                        "type": "eFbxString",
                        "value": json.dumps(metadata),
                    },
                }
            }
        }
        path = self._write(gltf)
        ptk.MeshConvert.apply_glb_animations(path)
        return path

    # ------------------------------------------- shots played as a sequence
    # A shots-only deliverable drops the whole-timeline clip its shots were cut
    # from -- that clip holds the same performance twice over (66.5 MB of the
    # production assembly). The page rebuilds the sequence from the shots so a
    # reviewer can still watch and scrub the whole thing.

    def test_a_shots_only_file_offers_a_whole_sequence_first(self):
        found = self._load(self._shots_only_glb(), probe=self._sequence_probe())

        self.assertEqual(found["errors"], [])
        self.assertIs(found["selected"], True, "playClip must reach the sequence")
        self.assertIn("FULL SEQUENCE", found["labels"][0])
        self.assertIn("2 shots", found["labels"][0])
        self.assertIn("FULL SEQUENCE", found["openedOn"])

    def test_the_sequence_spans_the_whole_authored_range(self):
        """0-150 at 30fps is 5s -- the gap included, not 4s of shot content."""
        found = self._load(self._shots_only_glb(), probe=self._sequence_probe())

        self.assertIn("/ 5.00s", found["end"]["readout"])
        self.assertIn("f150", found["end"]["readout"])
        self.assertIn("f0", found["start"]["readout"])

    def test_a_shot_the_carrier_never_finished_is_not_placed(self):
        """The manifest reads both bounds with `.get`, so a take written
        without an "end" arrives as ``end_frame: null``.

        Placed anyway, such a shot becomes a segment at its own start offset
        inside a span that never accounted for it -- here at 7.0s of a 5.00s
        sequence, which the scrubber cannot reach and the picker counts as a
        shot you can watch. `Math.max` coerces the null to 0, so the SPAN
        stays right and nothing else gives the miscount away. A shot the page
        was not told the end of is one it cannot place, so it is left out.
        """
        found = self._load(
            self._shots_only_glb(half_take=True), probe=self._sequence_probe()
        )

        self.assertIn("0–150, 2 shots", found["labels"][0], found["labels"][0])
        # The span is unchanged either way -- which is exactly why the
        # miscount is invisible without this assertion.
        self.assertIn("/ 5.00s", found["end"]["readout"])

    def test_scrubbing_lands_in_the_right_shot(self):
        """The whole point: one slider, and it addresses both shots.

        At 4.0s the playhead is one second into a shot that starts at 3.0s, so
        the model must be posed at that shot's OWN one-second mark (y=1) --
        not at 4 seconds of a clip that is only two long.
        """
        found = self._load(self._shots_only_glb(), probe=self._sequence_probe())

        self.assertEqual(found["start"]["y"], 0.0)
        self.assertEqual(found["inShotB"]["y"], 1.0)

    def test_a_gap_holds_the_previous_shot(self):
        """2.5s is past SHOT_A's end and before SHOT_B's start.

        The whole-timeline clip held the last pose across a gap; a sequence
        that left the model wherever the previous frame drew it, or snapped it
        to the next shot's first frame, would not be showing the same thing.
        """
        found = self._load(self._shots_only_glb(), probe=self._sequence_probe())

        self.assertEqual(found["gap"]["y"], 2.0, "the gap must hold SHOT_A's last pose")

    def test_an_empty_trailing_shot_still_lengthens_the_timeline(self):
        """A shot that holds still bakes no curve, but its frames are real.

        The whole-timeline clip played through them holding the last pose. A
        sequence that ended at the last MOVING shot would run 150 frames short
        of the range the file declares.
        """
        found = self._load(
            self._shots_only_glb(empty_tail=True), probe=self._sequence_probe()
        )

        self.assertEqual(found["errors"], [])
        # 0-300 at 30fps, not 0-150: the empty shot extends the span.
        self.assertIn("/ 10.00s", found["end"]["readout"])
        self.assertIn("f300", found["end"]["readout"])
        # ... and it holds the last MOVING pose out there, as the gap does.
        self.assertEqual(found["end"]["y"], 2.0)

    def test_a_file_that_ships_its_own_sequence_is_left_alone(self):
        """No synthetic entry when a real whole-timeline clip is present.

        Two ways to watch the same thing, one of them a reconstruction, is a
        worse picker than one.
        """
        found = self._load(
            self._shots_only_glb(with_sequence=True), probe=self._sequence_probe()
        )

        self.assertIs(found["selected"], False)
        self.assertFalse(
            [label for label in found["labels"] if "FULL SEQUENCE" in label],
            found["labels"],
        )

    # ------------------------------------------ which shot is on screen
    # Scrubbing a five-second sequence, the readout said only how far in the
    # playhead was. Which of the shots that was is the question a reviewer is
    # actually asking, and the picker cannot answer it: it is sitting on FULL
    # SEQUENCE the whole way through.

    def test_the_readout_names_the_shot_the_playhead_is_standing_in(self):
        found = self._load(self._shots_only_glb(), probe=self._sequence_probe())

        self.assertEqual(found["errors"], [])
        self.assertIn("SHOT_A", found["start"]["readout"])
        self.assertIn("SHOT_B", found["inShotB"]["readout"])

    def test_the_description_follows_the_playhead_across_the_sequence(self):
        """A shot's description is read off the SEGMENT the sequence is
        standing in, not off the picker's row -- the picker is on FULL
        SEQUENCE throughout, so a reading taken from it would give every shot
        the same note (or none at all).

        Only SHOT_B states one, so the two readings differ, which is what makes
        this an assertion about the segment rather than about the manifest.
        """
        found = self._load(
            self._shots_only_glb(described=True), probe=self._sequence_probe()
        )

        self.assertEqual(found["errors"], [])
        self.assertIsNone(found["start"]["description"], "SHOT_A states none")
        self.assertEqual(found["inShotB"]["description"], self.SHOT_B_DESCRIPTION)
        # A gap holds the previous shot's pose, so it holds its note too -- the
        # same rule the readout's '(hold)' follows.
        self.assertIsNone(found["gap"]["description"])

    def test_a_gap_is_labelled_as_a_hold_of_the_shot_before_it(self):
        """The pose on screen in a gap is the previous shot's last frame, held.

        Naming that shot outright would claim it plays through frames it does
        not cover -- which is a bug report waiting to be filed against a shot
        that is behaving exactly as the whole-timeline clip did.
        """
        found = self._load(self._shots_only_glb(), probe=self._sequence_probe())

        self.assertIn("SHOT_A (hold)", found["gap"]["readout"])
        self.assertNotIn("SHOT_B", found["gap"]["readout"])

    def test_a_single_clip_readout_is_not_labelled(self):
        """The picker already names it; repeating it beside the playhead is
        noise on the only clip where the picker is unambiguous."""
        # A file that ships its OWN whole-timeline clip builds no synthetic
        # sequence, so the transport is on a single clip throughout.
        found = self._load(
            self._shots_only_glb(with_sequence=True), probe=self._sequence_probe()
        )

        self.assertEqual(found["errors"], [])
        self.assertIs(found["selected"], False, "this fixture ships no sequence")
        self.assertNotIn("·", found["start"]["readout"])

    # ------------------------------------------------- recording a clip
    # The page's transport can play one shot or the whole sequence; these are
    # the tests that it can also WRITE what it plays, at the deliverable's own
    # frame rate rather than at whatever rate the device managed.

    @unittest.skipUnless(
        ptk.VidUtils.resolve_ffmpeg(required=False), "needs ffmpeg on PATH"
    )
    def test_recording_the_selected_shot_writes_that_shots_frames(self):
        found, movie = self._record("SHOT_B")

        # SHOT_B is frames 90-150 at 30fps: 61 frames, 2.00s. Not the
        # sequence's five seconds, and not the two seconds' worth of frames a
        # count that forgot its far bound would produce.
        self.assertEqual(found["clip"]["startFrame"], 90)
        self.assertEqual(found["clip"]["endFrame"], 150)
        self.assertTrue(movie.is_file() and movie.stat().st_size > 0)
        self._assert_not_blank(movie)
        self.assertIn("SHOT_B", movie.name)
        self.assertIn("61 frames", found["status"])

    @unittest.skipUnless(
        ptk.VidUtils.resolve_ffmpeg(required=False), "needs ffmpeg on PATH"
    )
    def test_recording_the_full_sequence_writes_the_whole_timeline(self):
        """Including the gap: the movie is as long as the timeline the shots
        were cut from, not as long as the shots add up to."""
        found, movie = self._record("FULL SEQUENCE")

        self.assertEqual(found["clip"]["startFrame"], 0)
        self.assertEqual(found["clip"]["endFrame"], 150)
        self.assertTrue(movie.is_file() and movie.stat().st_size > 0)
        self._assert_not_blank(movie)
        self.assertIn("151 frames", found["status"])
        self.assertIn("5.0s", found["status"])

    @unittest.skipUnless(
        ptk.VidUtils.resolve_ffmpeg(required=False), "needs ffmpeg on PATH"
    )
    def test_recording_falls_back_when_the_browser_has_no_workers(self):
        """Frames compress on worker threads, and a browser without them still
        records -- more slowly, and to the same movie.

        The fallback is a second encoder implementation that never runs on any
        browser this suite otherwise drives, so without this it would be code
        nothing executes until it is someone's only path.
        """
        found, movie = self._record("SHOT_B", probe=self._no_worker_probe())

        # Frames arrived with no Worker in the page, so the main-thread encoder
        # is the only thing that can have produced them.
        self.assertEqual(found["workerType"], "undefined")
        self.assertIn("61 frames", found["status"])
        self.assertTrue(movie.is_file() and movie.stat().st_size > 0)
        self._assert_not_blank(movie)

    @unittest.skipUnless(
        ptk.VidUtils.resolve_ffmpeg(required=False), "needs ffmpeg on PATH"
    )
    def test_burn_in_draws_the_shot_name_and_time_into_the_movie(self):
        """Opt-in scene data lands in the PIXELS, not merely in the filename.

        Recorded twice, and the bottom of the frame is compared: everything
        else about the two runs is identical, so a difference there is the
        burn-in and nothing else.
        """
        _, plain = self._record("SHOT_B")
        _, stamped = self._record("SHOT_B", burn_in=True)

        before = self._bottom_strips(plain, 2)
        after = self._bottom_strips(stamped, 2)
        self.assertEqual(len(before[0]), len(after[0]), "frames differ in size")
        changed = sum(1 for a, b in zip(before[0], after[0]) if abs(a - b) > 40)
        self.assertGreater(
            changed,
            200,
            "the burn-in run is indistinguishable from the plain one at the "
            "foot of the frame -- nothing was drawn into the movie",
        )

        # ADJACENT frames, which is the pair that matters. The capture loop
        # issues as many frames per tick as the encoder has room for, so several
        # frames are annotated between one animation frame and the next; if the
        # canvas they are drawn on were reused before its snapshot was taken,
        # they would all carry the LAST one's stamp. Comparing distant frames
        # cannot see that -- they land in different ticks and differ either way.
        moved = sum(1 for a, b in zip(after[0], after[1]) if abs(a - b) > 40)
        self.assertGreater(
            moved,
            80,
            "consecutive frames carry the same burn-in -- the counter is not "
            "advancing per frame, so frames share a stale annotation",
        )

    @unittest.skipUnless(
        ptk.VidUtils.resolve_ffmpeg(required=False), "needs ffmpeg on PATH"
    )
    def test_burn_in_draws_the_shot_description_into_the_movie(self):
        """The note the DCC's Shots panel carries is the half of a shot record
        a reviewer watching the movie cannot otherwise see -- so it has to
        reach the PIXELS, not merely the page."""
        _, plain = self._record("SHOT_B", described=True)
        _, noted = self._record("SHOT_B", describe=True, described=True)

        before = self._bottom_strips(plain, 1)[0]
        after = self._bottom_strips(noted, 1)[0]
        self.assertEqual(len(before), len(after), "frames differ in size")
        changed = sum(1 for a, b in zip(before, after) if abs(a - b) > 40)
        self.assertGreater(
            changed,
            200,
            "the described run is indistinguishable from the plain one at the "
            "foot of the frame -- the shot description was not drawn",
        )

    @unittest.skipUnless(
        ptk.VidUtils.resolve_ffmpeg(required=False), "needs ffmpeg on PATH"
    )
    def test_a_shot_with_no_description_records_without_one(self):
        """Most shots carry no note, and asking for the description on one that
        has none must leave the frame alone rather than draw a blank line or a
        placeholder into it.

        The pair with the test above is what makes either meaningful: the same
        option, the same fixture, and the only difference is whether the shot
        the page is on states a description. SHOT_A states none.
        """
        _, plain = self._record("SHOT_A", described=True)
        _, asked = self._record("SHOT_A", describe=True, described=True)

        before = self._bottom_strips(plain, 1)[0]
        after = self._bottom_strips(asked, 1)[0]
        changed = sum(1 for a, b in zip(before, after) if abs(a - b) > 40)
        self.assertLess(
            changed,
            50,
            "asking for a description on a shot that states none drew "
            "something into the frame anyway",
        )

    @unittest.skipUnless(
        ptk.VidUtils.resolve_ffmpeg(required=False), "needs ffmpeg on PATH"
    )
    def test_a_description_wider_than_the_frame_is_elided(self):
        """A shot description is prose, so it can be longer than the frame --
        and a canvas CLIPS text rather than refusing it, so an untrimmed one
        runs to the last column and reads as a corrupted burn-in.

        Measured at the far right of the strip, past the inset every burn-in
        line is drawn within: ink there is text that overran.
        """
        _, plain = self._record("SHOT_B", described=True)
        _, long = self._record("SHOT_B", describe=True, described=self.LONG_DESCRIPTION)

        before = self._bottom_strips(plain, 1, crop=self.FAR_RIGHT_CROP)[0]
        after = self._bottom_strips(long, 1, crop=self.FAR_RIGHT_CROP)[0]
        self.assertEqual(len(before), len(after), "frames differ in size")
        changed = sum(1 for a, b in zip(before, after) if abs(a - b) > 40)
        self.assertLess(
            changed,
            30,
            "the description reached the far edge of the frame -- it was "
            "clipped by the canvas rather than elided",
        )

    def test_the_export_prompt_can_be_cancelled_without_recording(self):
        """Pressing the button asks before it writes, and the answer 'no' has
        to be a real one: a prompt that records anyway is worse than no prompt,
        because the burn-in it was asked about is already in the file."""
        glb = self._shots_only_glb()
        beside = os.path.dirname(glb)
        before = set(os.listdir(beside))

        def drive(server, page):
            page.click("#controls button:has-text('Export Playblast')")
            page.wait_for_selector("#dialog:not([hidden])", timeout=30_000)
            page.click("#dialogCancel")
            page.wait_for_selector("#dialog", state="hidden", timeout=30_000)
            return {
                "button": page.eval_on_selector(
                    "#controls button:has-text('Export Playblast')",
                    "el => el.textContent",
                ),
            }

        found = self._load(glb, probe=self._record_probe(), then=drive)

        self.assertEqual(found["errors"], [])
        # The button is the recording's own progress readout, so it saying
        # anything else is this having started one.
        self.assertEqual(found["button"], "Export Playblast")
        self.assertEqual(
            [n for n in set(os.listdir(beside)) - before if n.endswith(".mp4")],
            [],
            "a cancelled prompt wrote a movie anyway",
        )

    def test_the_prompt_keeps_the_keyboard_off_the_page_behind_it(self):
        """Blocking the page's shortcuts did not take the keyboard whole: Tab
        still walked focus out of the prompt to the controls behind it, which
        precede it in the document -- and an arrow key on the clip picker
        changed the clip the recording takes, not the one the prompt names.
        The page behind is inert while the prompt is up, and live once it
        closes."""
        glb = self._shots_only_glb()
        picker = "() => document.getElementById('clipSelect').selectedIndex"
        outside = (
            "() => { const a = document.activeElement;"
            " return a && a !== document.body"
            " && !document.getElementById('dialog').contains(a)"
            " ? a.id || a.tagName : null; }"
        )

        def drive(server, page):
            before = page.evaluate(picker)
            page.click("#controls button:has-text('Export Playblast')")
            page.wait_for_selector("#dialog:not([hidden])", timeout=30_000)
            escaped = []
            # Backwards, past the prompt's own controls and twice over the bar
            # and the transport, pressing the key that moves a picker each time.
            for _ in range(24):
                page.keyboard.press("Shift+Tab")
                where = page.evaluate(outside)
                if where:
                    escaped.append(where)
                page.keyboard.press("ArrowDown")
            after = page.evaluate(picker)
            page.keyboard.press("Escape")
            page.wait_for_selector("#dialog", state="hidden", timeout=30_000)
            live = page.eval_on_selector(
                "#clipSelect",
                "s => { s.focus(); return document.activeElement === s; }",
            )
            return {"escaped": escaped, "before": before, "after": after, "live": live}

        found = self._load(glb, probe=self._record_probe(), then=drive)

        self.assertEqual(found["errors"], [])
        self.assertEqual(
            found["escaped"], [], "focus left the prompt for the page behind it"
        )
        self.assertEqual(
            found["after"],
            found["before"],
            "a key pressed in the prompt changed the clip behind it",
        )
        self.assertTrue(found["live"], "the page stayed inert after the prompt closed")

    @unittest.skipUnless(
        ptk.VidUtils.resolve_ffmpeg(required=False), "needs ffmpeg on PATH"
    )
    def test_the_quality_preset_sets_the_size_the_shot_is_rendered_at(self):
        """Draft keeps the old 1280 px frame; the default (High) is a 2560 px
        RENDER -- the headless view is smaller than that, so an upscaled
        capture would pass a size check while adding no detail. The pixel
        ratio the recording raised has to be back once it is done."""
        import cv2

        def long_edge(movie):
            capture = cv2.VideoCapture(str(movie))
            try:
                return max(
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
            finally:
                capture.release()

        draft_found, draft = self._record("SHOT_B", preset="draft")
        high_found, high = self._record("SHOT_B")

        self.assertLess(
            high_found["viewEdge"], 2560, "the view must be smaller than High"
        )
        self.assertEqual(long_edge(draft), 1280)
        self.assertEqual(long_edge(high), 2560)
        for found in (draft_found, high_found):
            self.assertEqual(
                found["pixelRatio"],
                found["pagePixelRatio"],
                "the raised ratio was kept",
            )
        # The status names what was written, so a reviewer can see the size
        # without opening the file.
        self.assertIn("2560×", high_found["status"])
        self._assert_not_blank(high)

    #: The bottom 15% of the frame, full width -- where the burn-in is drawn.
    BOTTOM_CROP = "crop=iw:trunc(ih*0.15):0:ih-trunc(ih*0.15)"

    #: The last 1% of that strip: OUTSIDE the burn-in's own inset, which is
    #: 0.75 of the text size -- and the text scales with the frame, so the
    #: inset holds its share of the width at every preset (35px of 2560 at the
    #: default High preset's 1440p capture, 17px of 1280 at Draft's 720p) and
    #: text reaches ~98.6% at the most. Nothing the burn-in draws may light this
    #: band up -- text that lands here ran off the frame instead of being elided.
    FAR_RIGHT_CROP = (
        "crop=trunc(iw*0.01):trunc(ih*0.15):iw-trunc(iw*0.01):ih-trunc(ih*0.15)"
    )

    def _bottom_strips(self, movie, count, crop=None):
        """The bottom 15% of the first *count* frames, as 8-bit luminance.

        *crop* narrows that to a band (see the CROP constants above).

        Decoded in ONE pass and split, rather than seeking per frame: adjacent
        frames are the interesting pair here, and `-ss` cannot address them
        reliably at that spacing.
        """
        ffmpeg = ptk.VidUtils.resolve_ffmpeg(required=True)
        raw = pathlib.Path(self.temp.path(extension=".gray"))
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(movie),
                "-frames:v",
                str(count),
                "-vf",
                crop or self.BOTTOM_CROP,
                "-pix_fmt",
                "gray",
                "-f",
                "rawvideo",
                str(raw),
            ],
            check=True,
        )
        data = raw.read_bytes()
        self.assertEqual(len(data) % count, 0, "raw frames are not equal-sized")
        size = len(data) // count
        return [data[i * size : (i + 1) * size] for i in range(count)]

    def _record(
        self,
        clip_name,
        probe=None,
        burn_in=False,
        describe=False,
        described=False,
        preset=None,
    ):
        """Select *clip_name* in the real page, press Export Playblast, answer
        the options prompt, and return the findings plus the movie it wrote.

        *burn_in* and *describe* tick the prompt's two boxes -- driven as a user
        drives them (a click on the label), not by reaching into the script's
        state, so the prompt itself is under test on every recording run.
        *described* builds the fixture with a shot description to burn in.
        *preset* picks a Quality preset by key; None leaves the default.

        A FRESH fixture per call, so two recordings of the same clip land in
        different directories: the movie is named after the clip, and a second
        one beside the first would overwrite it and leave this looking for a
        file that was never new.
        """
        glb = self._shots_only_glb(described=described)
        # The movie lands beside the file that was published; the test looks for
        # it the way its user would, rather than being handed the path.
        beside = os.path.dirname(glb)
        before = set(os.listdir(beside))

        def drive(server, page):
            selection = page.evaluate("(name) => window.__select(name)", clip_name)
            # The page's OWN ratio, read before a recording can raise it rather
            # than re-derived from how the page happens to set it, so the
            # restore check holds whatever the runner's display and clamp.
            page_ratio = page.evaluate("() => window.__pixelRatio()")
            page.click("#controls button:has-text('Export Playblast')")
            page.wait_for_selector("#dialog:not([hidden])", timeout=30_000)
            if burn_in:
                page.click("#dialogFields label:has-text('shot name')")
            if describe:
                page.click("#dialogFields label:has-text('shot description')")
            if preset is not None:
                page.select_option(
                    "#dialogFields label:has-text('Quality') select", preset
                )
            page.click("#dialogConfirm")
            # The status line is the page's own completion signal, and waiting
            # on it rather than on a file appearing is what makes a failure read
            # as "the page said why" instead of as a timeout.
            page.wait_for_function(
                "() => /playblast (saved|failed)/.test("
                "document.getElementById('status').textContent)",
                timeout=300_000,
            )
            return {
                "clip": selection["clip"],
                "selected": selection["ok"],
                "status": page.eval_on_selector("#status", "el => el.textContent"),
                "pixelRatio": page.evaluate("() => window.__pixelRatio()"),
                "pagePixelRatio": page_ratio,
                "viewEdge": page.evaluate(
                    "(ratio) => Math.max(innerWidth, innerHeight) * ratio", page_ratio
                ),
            }

        found = self._load(glb, probe=probe or self._record_probe(), then=drive)
        self.assertEqual(found["errors"], [])
        self.assertTrue(found["selected"], f"{clip_name} was not selectable")
        self.assertIn("saved", found["status"], found["status"])
        new = [n for n in set(os.listdir(beside)) - before if n.endswith(".mp4")]
        self.assertEqual(len(new), 1, f"expected one movie, got {new}")
        return found, pathlib.Path(beside, new[0])

    def _assert_not_blank(self, movie):
        """The recorded movie must contain the SCENE, not an empty canvas.

        A capture that snapshots the WebGL canvas one task too late reads a
        drawing buffer the browser has already composited away: every frame
        comes back blank, the recording succeeds, the file is written, and
        nothing about its size or duration says anything is wrong. This is the
        assertion that fails instead -- it decodes a frame and looks at it.
        """
        # required=True: every caller is already gated on ffmpeg being present,
        # so an absent one is a broken test rather than a skip -- and it says so
        # here instead of putting None into an argv.
        ffmpeg = ptk.VidUtils.resolve_ffmpeg(required=True)
        raw = pathlib.Path(self.temp.path(extension=".gray"))
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(movie),
                "-frames:v",
                "1",
                "-pix_fmt",
                "gray",
                "-f",
                "rawvideo",
                str(raw),
            ],
            check=True,
        )
        pixels = raw.read_bytes()
        self.assertTrue(pixels, f"no frame could be decoded from {movie.name}")
        # A blank frame is one value everywhere. The fixture is a lit model on a
        # dark background, so a real one covers a wide range.
        self.assertGreater(
            max(pixels) - min(pixels),
            8,
            f"{movie.name} decodes to a near-uniform frame -- the capture "
            "recorded an empty canvas, not the scene",
        )

    def _no_worker_probe(self):
        path = self.temp.path(extension=".js")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(NO_WORKER_PROBE)
        return path

    def _record_probe(self):
        path = self.temp.path(extension=".js")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(RECORD_PROBE)
        return path

    def _sequence_probe(self):
        path = self.temp.path(extension=".js")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(SEQUENCE_PROBE)
        return path

    #: The HDR divisor `apply_glb_lightmaps` records as `intensity`, chosen so
    #: the page's arithmetic is checkable rather than merely present.
    BAKE_VALUE = 4.0

    #: A 1x1 PNG, for a fixture that needs a texture the loader will decode.
    PIXEL_PNG = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
        "2mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    def _lightmapped_glb(
        self,
        normal_map=False,
        bake_value=None,
        pbr=None,
        environment=None,
        baked_reflections=None,
    ):
        """A GLB whose material wears a BAKED map, built by the real applier.

        Hand-writing `extras.lightmap_web` would test the page against a
        fixture rather than against the pipeline, and the whole point of these
        assertions is that the two agree. So the manifest rides the
        `data_export` carrier exactly as an FBX delivers it and
        `apply_glb_lightmaps` produces the block the viewer reads.

        *normal_map* also gives the material a normal map, which is what
        turns on the page's bake-relief shader patch: True for a stock pixel,
        or a data URI for a texel of the test's own (`normal_texel_png`).
        *bake_value* is the
        bake's irradiance (default `BAKE_VALUE`); *pbr* a
        `pbrMetallicRoughness` block for the material, which otherwise takes
        glTF's defaults -- a ROUGH METAL, which no lightmap can light, so a
        pixel test says what it wants. *environment* publishes that
        environment intensity in the file's own rendering policy, the way a
        deliverable states its rig, so the page is driven through the read
        path a real GLB uses rather than by poking its materials; and
        *baked_reflections* the level a baked material reflects it at
        (``lightmappedMaterials.envMapIntensity``), the export's choice.
        """
        import cv2
        import numpy as np

        exr_dir = self.temp.dir_path()
        exr = os.path.join(exr_dir, "room_Lightmap.exr")
        if bake_value is None:
            bake_value = self.BAKE_VALUE
        cv2.imwrite(exr, np.full((8, 8, 3), bake_value, dtype=np.float32))

        manifest = {
            # Load-bearing: the reader refuses a manifest whose version it does
            # not recognise, and an ABSENT version reads as newer than v1 --
            # so a fixture without it binds nothing and every assertion below
            # would pass vacuously. That is what the `bound` check catches.
            "version": 1,
            "objects": [
                {
                    "name": "room",
                    "map": "room_Lightmap.exr",
                    "uvIndex": 1,
                    "intensity": 1.0,
                    "scaleOffset": [1.0, 1.0, 0.0, 0.0],
                }
            ],
        }
        gltf = {
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [0, 1]}],
            "scene": 0,
            "nodes": [
                {"name": "room", "mesh": 0},
                {
                    "name": "data_export",
                    "extras": {
                        "fromFBX": {
                            "userProperties": {
                                "lightmap_metadata": {
                                    "type": "eFbxString",
                                    "value": json.dumps(manifest),
                                }
                            }
                        }
                    },
                },
            ],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": 1,
                                "TEXCOORD_0": 3,
                                "TEXCOORD_1": 3,
                            },
                            "material": 0,
                        }
                    ]
                }
            ],
            "materials": [{"name": "room_MAT"}],
        }
        if pbr:
            gltf["materials"][0]["pbrMetallicRoughness"] = dict(pbr)
        if normal_map:
            uri = normal_map if isinstance(normal_map, str) else self.PIXEL_PNG
            gltf["images"] = [{"name": "room_N", "uri": uri}]
            gltf["textures"] = [{"source": 0}]
            gltf["materials"][0]["normalTexture"] = {"index": 0}
        rendering = {}
        if environment is not None:
            rendering["environment"] = {"intensity": environment}
        if baked_reflections is not None:
            rendering["lightmappedMaterials"] = {"envMapIntensity": baked_reflections}
        if rendering:
            gltf["extras"] = {"scene_sidecar": {"handoff": {"rendering": rendering}}}
        path = self._write(gltf, uvs=True)
        bound = ptk.MeshConvert.apply_glb_lightmaps(path, search_dirs=[exr_dir])
        # A fixture that silently bound nothing would make every assertion
        # below vacuous, and they would all still pass.
        self.assertTrue(bound, "the fixture itself must carry a bound lightmap")
        return path

    def _baked_room_glb(self, objects):
        """A room of several objects, baked by the real applier.

        *objects* are ``(name, material indices, map[, rect])``: one mesh node
        per object with one primitive per material, *map* the EXR its bake
        names (``None``: never baked; a file that does not exist: baked, with
        its map lost) and *rect* its atlas ``scaleOffset``. Every baked object
        carries the bake's ``lightmapInfo`` marker in its node extras, shaped as
        FBX2glTF transcribes it -- which is what lets the page tell an object
        the bake meant to light from one it merely wears a lit material.
        """
        import cv2
        import numpy as np

        exr_dir = self.temp.dir_path()
        cv2.imwrite(
            os.path.join(exr_dir, "room_Lightmap.exr"),
            np.full((8, 8, 3), self.BAKE_VALUE, dtype=np.float32),
        )
        attrs = {"POSITION": 1, "TEXCOORD_0": 3, "TEXCOORD_1": 3}
        entries, nodes, meshes = [], [], []
        for name, materials, lightmap, *rect in objects:
            node = {"name": name, "mesh": len(meshes)}
            meshes.append(
                {
                    "primitives": [
                        {"attributes": dict(attrs), "material": m} for m in materials
                    ]
                }
            )
            if lightmap:
                scale_offset = rect[0] if rect else [1.0, 1.0, 0.0, 0.0]
                entries.append(
                    {
                        "name": name,
                        "map": lightmap,
                        "uvIndex": 1,
                        "intensity": 1.0,
                        "scaleOffset": scale_offset,
                    }
                )
                marker = {"map": lightmap, "uvIndex": 1, "scaleOffset": scale_offset}
                node["extras"] = {
                    "fromFBX": {
                        "userProperties": {
                            "lightmapInfo": {
                                "type": "eFbxString",
                                "value": json.dumps(marker),
                            }
                        }
                    }
                }
            nodes.append(node)
        manifest = {"version": 1, "objects": entries}
        nodes.append(
            {
                "name": "data_export",
                "extras": {
                    "fromFBX": {
                        "userProperties": {
                            "lightmap_metadata": {
                                "type": "eFbxString",
                                "value": json.dumps(manifest),
                            }
                        }
                    }
                },
            }
        )
        count = 1 + max(m for _name, materials, *_rest in objects for m in materials)
        gltf = {
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": list(range(len(nodes)))}],
            "scene": 0,
            "nodes": nodes,
            "meshes": meshes,
            "materials": [{"name": f"mat{i}"} for i in range(count)],
        }
        path = self._write(gltf, uvs=True)
        ptk.MeshConvert.apply_glb_lightmaps(path, search_dirs=[exr_dir])
        return path

    def _write(self, gltf, uvs=False):
        """A minimal but VALID binary glTF: JSON chunk plus a real BIN chunk.

        The page runs a real GLTFLoader, so a fixture the loader rejects tests
        the fixture rather than the viewer -- hence actual position/time buffers
        rather than dangling accessor indices.
        """
        positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
        times = struct.pack("<3f", 0.0, 1.0, 2.0)
        translations = struct.pack("<9f", 0, 0, 0, 0, 1, 0, 0, 2, 0)
        texcoords = struct.pack("<6f", 0, 0, 1, 0, 0, 1)
        blob = times + positions + translations + (texcoords if uvs else b"")
        gltf["buffers"] = [{"byteLength": len(blob)}]
        gltf["bufferViews"] = [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(times)},
            {"buffer": 0, "byteOffset": len(times), "byteLength": len(positions)},
            {
                "buffer": 0,
                "byteOffset": len(times) + len(positions),
                "byteLength": len(translations),
            },
        ]
        if uvs:
            gltf["bufferViews"].append(
                {
                    "buffer": 0,
                    "byteOffset": len(times) + len(positions) + len(translations),
                    "byteLength": len(texcoords),
                }
            )
        gltf["accessors"] = [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "SCALAR",
                "min": [0.0],
                "max": [2.0],
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [1, 1, 0],
            },
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC3"},
        ]
        if uvs:
            gltf["accessors"].append(
                {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC2"}
            )
        json_bytes = json.dumps(gltf).encode("utf-8")
        json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
        blob += b"\0" * ((4 - len(blob) % 4) % 4)
        total = 12 + 8 + len(json_bytes) + 8 + len(blob)
        # Its own tracked folder: a still or a recording the page writes
        # BESIDE the deliverable lands in there and goes with cleanup(), and
        # a before/after listing of it sees nothing another process wrote.
        out = os.path.join(self.temp.dir_path(), "fixture.glb")
        with open(out, "wb") as fh:
            fh.write(struct.pack("<4sII", b"glTF", 2, total))
            fh.write(struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes)
            fh.write(struct.pack("<I4s", len(blob), b"BIN\0") + blob)
        return out

    # ------------------------------------------------- exporting a still
    # The page's Export Image: the view it is showing, rendered at the size the
    # prompt asks for and written beside the deliverable. What these check is
    # the PICTURE -- a readback one task too late saves an empty canvas, and
    # nothing about the file's existence or size says so.

    def test_export_image_saves_the_view_beside_the_deliverable(self):
        """At the default (High) size, which the headless view is smaller than
        -- so a 2560 px file is a raised-ratio RENDER, not an upscale, and the
        raised ratio has to be back once the still is taken."""
        import cv2

        found, image = self._export_image()

        self.assertLess(found["viewEdge"], 2560, "the view must be smaller than High")
        self.assertEqual(image.name, f"{found['stem']}_view_001.png")
        pixels = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(max(pixels.shape), 2560)
        self._assert_still_not_blank(pixels, image)
        self.assertEqual(found["pixelRatio"], found["pagePixelRatio"], "ratio kept")
        # The status names what was written, so the reviewer need not go look.
        self.assertIn(image.name, found["status"])
        self.assertIn("2560×", found["status"])

    def test_as_shown_saves_the_drawing_buffer_unscaled(self):
        import cv2

        found, image = self._export_image(preset="view")

        pixels = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(
            (pixels.shape[1], pixels.shape[0]), tuple(found["canvas"]), "resized"
        )
        self._assert_still_not_blank(pixels, image)

    def test_the_image_prompt_can_be_cancelled_without_saving(self):
        glb = self._shots_only_glb()
        beside = os.path.dirname(glb)
        before = set(os.listdir(beside))

        def drive(server, page):
            page.click("#controls button:has-text('Export Image')")
            page.wait_for_selector("#dialog:not([hidden])", timeout=30_000)
            page.click("#dialogCancel")
            page.wait_for_selector("#dialog", state="hidden", timeout=30_000)

        found = self._load(
            glb, probe=self._record_probe(), then=drive, scripts=["snapshot"]
        )

        self.assertEqual(found["errors"], [])
        self.assertEqual(
            [n for n in set(os.listdir(beside)) - before if n.endswith(".png")],
            [],
            "a cancelled prompt saved an image anyway",
        )

    def test_a_scene_push_still_is_downloaded_by_the_page(self):
        """With no file on disk to sit beside, the serve root holds the still
        and the page's download is the only copy that outlives the session.

        Under BOTH spellings of the loopback host. A browser honours an
        anchor's `download` only on the page's own origin, and the page is as
        validly open at localhost as at 127.0.0.1 -- so a link spelled the
        server's way navigated a localhost tab to the bare PNG, replacing the
        preview, instead of saving it.
        """
        for host in ("127.0.0.1", "localhost"):
            with self.subTest(host=host):
                glb = self._shots_only_glb()

                def drive(server, page, glb=glb):
                    # What a scene push looks like from here: the published
                    # file is the bridge's scratch, and is gone by the time
                    # anyone presses a button.
                    os.remove(glb)
                    page.click("#controls button:has-text('Export Image')")
                    page.wait_for_selector("#dialog:not([hidden])", timeout=30_000)
                    with page.expect_download(timeout=60_000) as download:
                        page.click("#dialogConfirm")
                    path = pathlib.Path(self.temp.path(extension=".png"))
                    download.value.save_as(str(path))
                    return {
                        "downloaded": download.value.suggested_filename,
                        "saved": str(path),
                        "still_on_page": page.evaluate("() => !!window.__probe"),
                    }

                found = self._load(
                    glb,
                    probe=self._record_probe(),
                    then=drive,
                    scripts=["snapshot"],
                    host=host,
                )

                self.assertEqual(found["errors"], [])
                self.assertTrue(found["still_on_page"], "the tab navigated away")
                # Not named after the scratch payload (a random tag): with no
                # deliverable on disk a still is a plain numbered view.
                self.assertEqual(found["downloaded"], "view_001.png")
                self.assertTrue(
                    pathlib.Path(found["saved"]).read_bytes().startswith(b"\x89PNG"),
                    "the download is not the PNG the page posted",
                )

    def _export_image(self, preset=None):
        """Press Export Image in the real page, answer the prompt, and return
        the findings plus the PNG it wrote beside the deliverable.

        *preset* picks a Size by key; None leaves the default. A fresh fixture
        per call, so a numbered still is the first in its folder.
        """
        glb = self._shots_only_glb()
        beside = os.path.dirname(glb)
        before = set(os.listdir(beside))

        def drive(server, page):
            page_ratio = page.evaluate("() => window.__pixelRatio()")
            page.click("#controls button:has-text('Export Image')")
            page.wait_for_selector("#dialog:not([hidden])", timeout=30_000)
            if preset is not None:
                page.select_option(
                    "#dialogFields label:has-text('Size') select", preset
                )
            page.click("#dialogConfirm")
            page.wait_for_function(
                "() => /image (saved|export failed)/.test("
                "document.getElementById('status').textContent)",
                timeout=120_000,
            )
            return {
                "status": page.eval_on_selector("#status", "el => el.textContent"),
                "pixelRatio": page.evaluate("() => window.__pixelRatio()"),
                "pagePixelRatio": page_ratio,
                "viewEdge": page.evaluate(
                    "(ratio) => Math.max(innerWidth, innerHeight) * ratio", page_ratio
                ),
                "canvas": page.evaluate(
                    "() => { const c = document.querySelector('canvas');"
                    " return [c.width, c.height]; }"
                ),
            }

        found = self._load(
            glb, probe=self._record_probe(), then=drive, scripts=["snapshot"]
        )
        self.assertEqual(found["errors"], [])
        self.assertIn("image saved", found["status"], found["status"])
        new = [n for n in set(os.listdir(beside)) - before if n.endswith(".png")]
        self.assertEqual(len(new), 1, f"expected one image, got {new}")
        found["stem"] = pathlib.Path(glb).stem
        return found, pathlib.Path(beside, new[0])

    def _assert_still_not_blank(self, pixels, image):
        """The saved still must hold the SCENE, not an empty canvas: the
        fixture is a lit model on a dark background, so a real one covers a
        wide range and a blank readback is one value everywhere."""
        self.assertGreater(
            int(pixels.max()) - int(pixels.min()),
            8,
            f"{image.name} is near-uniform -- the capture read an empty canvas",
        )

    # ------------------------------------------------------------------ driver
    def _load(
        self, glb, then_publish=None, probe=None, then=None, scripts=(), host=None
    ):
        """Serve *glb*, open it in the real page, return the probe's findings.

        *scripts* are packaged viewer scripts to activate alongside the probe
        -- the ones a push names, as opposed to those a deliverable turns on
        by itself. *host* opens the page under another loopback spelling
        (``"localhost"``) than the server's own URL uses.

        *then_publish* publishes a SECOND version once the page is up and waits
        for the swap -- the only way to reach `disposeModel`, which frees the
        outgoing model's textures between pushes.

        *then* is the general form: ``then(server, page)`` runs once the page
        is up and may return a dict merged into the findings.

        *probe* overrides the class-wide script for tests that need to measure
        something else (the fade suite drives the playhead); it reports through
        the same ``window.__probe`` handle, so the wait below is unchanged.
        """
        from playwright.sync_api import sync_playwright

        # port=0: the default is the production port, and a real preview tab
        # the user has open would poll this test server.
        server = ptk.PreviewServer(viewer=True, title="live-test", port=0)
        server.start()
        for name in scripts:
            server.add_script(name)
        server.add_script("probe", probe or self.probe)
        server.publish(glb)
        console = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    channel="msedge",
                    headless=True,
                    # Software WebGL: with no GPU in a test runner the renderer
                    # never initializes and every assertion below would read as
                    # a viewer bug rather than a missing device.
                    args=["--enable-unsafe-swiftshader"],
                )
                page = browser.new_page()
                page.on(
                    "console",
                    lambda m: (
                        console.append(f"[{m.type}] {m.text}")
                        if m.type == "error"
                        else None
                    ),
                )
                page.on("pageerror", lambda e: console.append(f"[pageerror] {e}"))
                url = server.url.replace(server.host, host) if host else server.url
                page.goto(url, wait_until="domcontentloaded", timeout=120_000)
                page.wait_for_function(
                    "() => window.__probe && window.__probe.ready === true",
                    timeout=180_000,
                )
                if then_publish is not None:
                    # The page polls for a new version and swaps itself; waiting
                    # on the probe's OWN load counter is what makes this the
                    # second render rather than a re-read of the first.
                    server.publish(then_publish)
                    page.wait_for_function(
                        "() => window.__probe.loads >= 2", timeout=180_000
                    )
                extra = then(server, page) if then is not None else None
                found = page.evaluate("() => window.__probe")
                if extra:
                    found.update(extra)
                browser.close()
        finally:
            server.stop()
        found["console_errors"] = console
        return found

    # ------------------------------------------------------------------- tests
    def test_a_superseded_page_reloads_itself(self):
        """The poll swaps the model but never the script running it, so a
        viewer fix reached an open tab only via F5 (2026-09-05: a highlight
        the GLB carried, a page from before the binding existed). The manifest
        now fingerprints the page; when the fingerprint changes under an open
        tab, the tab reloads and comes back ready."""

        def bump(server, page):
            stamp = page.evaluate(
                "() => performance.getEntriesByType('navigation')[0].type"
            )
            with server._lock:
                server._viewer_stamp = "0123456789ab"
            page.wait_for_function(
                "() => performance.getEntriesByType('navigation')[0].type === 'reload'"
                " && window.__probe && window.__probe.ready === true",
                timeout=120_000,
            )
            return {"navigation_before": stamp}

        found = self._load(self._animated_glb(), then=bump)

        self.assertEqual(found["navigation_before"], "navigate")
        self.assertEqual(found["errors"], [])
        self.assertEqual(found["meshes"], 1, "the reloaded page loads the model again")

    def test_the_page_loads_a_model_and_mounts_its_clips(self):
        """The whole load path, executed: container, loader, scene, mixer."""
        found = self._load(self._animated_glb())

        self.assertEqual(found["errors"], [])
        self.assertEqual(found["meshes"], 1)
        self.assertEqual(found["clipCount"], 2)
        self.assertTrue(found["hasMixer"])

    def test_it_opens_on_a_clip_that_plays_and_says_which_are_empty(self):
        """A deliverable must not open on 0.00s of nothing.

        The behaviour, not the line: an inversion that still spells
        ``default_clip`` correctly would pass a substring check and fail here.
        """
        found = self._load(self._animated_glb())

        self.assertIn("MOVES", found["openedOn"])
        self.assertNotIn("empty", found["openedOn"])
        empty_label = next(lbl for lbl in found["labels"] if "HOLDS" in lbl)
        self.assertIn("empty", empty_label)

    def test_a_named_clip_is_selectable_and_an_unknown_one_is_refused(self):
        """``playClip`` is the consumer-facing contract for shot names."""
        found = self._load(self._animated_glb())

        self.assertIs(found["playedNamed"], True)
        self.assertIs(found["playedMissing"], False)

    def test_a_clean_load_logs_no_console_error(self):
        """Catches the whole class of regression a substring check cannot see.

        A JS exception, a missing viewer script, a 404 for an asset the page
        asks for -- none of which change the page's text.
        """
        found = self._load(self._animated_glb())

        self.assertEqual(found["console_errors"], [])

    # -------------------------------------------------- the lighting policy
    # glTF has no lightmap slot, so a bake travels DISGUISED as occlusion on
    # TEXCOORD_1 and only the page's rebind turns it back into light. Every
    # step of that was previously asserted by string-matching the page source,
    # which pins the spelling and not the behaviour: a genuine inversion that
    # keeps the same tokens passes, and one did -- the key-light gate shipped
    # inverted for part of 2026-08-12 and every partly-baked room rendered
    # blown out, reported as a *baker* regression because the extra light is
    # added downstream of the EXR.

    def _lit(self, found):
        """The one material the fixture bakes."""
        named = [m for m in found["materials"] if m["name"] == "room_MAT"]
        self.assertEqual(len(named), 1, f"fixture material missing: {found}")
        return named[0]

    def test_a_baked_map_is_rebound_as_LIGHT_not_as_occlusion(self):
        """The whole disguise, undone: it arrives in `occlusionTexture` and has
        to end up in `lightMap`, or the room renders as dirt instead of light."""
        found = self._load(self._lightmapped_glb())

        # A COUNT of rebound materials, not a flag -- which is what the key
        # light is gated on, so the distinction is load-bearing rather than
        # cosmetic (the gate that shipped inverted asked whether the model was
        # FULLY baked instead of whether ANYTHING was).
        self.assertEqual(found["lightmapped"], 1)
        material = self._lit(found)
        self.assertTrue(material["hasLightMap"], "the bake never became light")
        self.assertFalse(
            material["hasAoMap"],
            "the carrier slot must be released, or the bake ALSO darkens the "
            "surface it is lighting",
        )

    def test_the_rebound_map_samples_the_bake_UV_SET_in_sRGB(self):
        """Two independent ways to render a correct bake wrongly: sampling it
        on UV0 (the material's own layout, so the lighting smears across the
        atlas) and reading it as linear (the encode is sRGB, so every texel
        comes back the wrong brightness)."""
        material = self._lit(self._load(self._lightmapped_glb()))

        self.assertEqual(material["lightMapChannel"], 1)
        self.assertIs(material["lightMapSRGB"], True)

    def test_the_HDR_divisor_reaches_the_material_as_its_intensity(self):
        """The encode divides the bake down into an 8-bit map and records the
        divisor; the page multiplies it back. Dropped, the room renders at a
        fraction of the brightness it was approved at -- and looks plausible,
        which is why it needs a number rather than a presence check."""
        material = self._lit(self._load(self._lightmapped_glb()))

        self.assertAlmostEqual(material["lightMapIntensity"], self.BAKE_VALUE, places=3)

    def test_a_baked_scene_turns_the_key_light_OFF(self):
        """A lightmap already CONTAINS its lighting, so a scene-wide key light
        added on top double-lights it. This is the gate that shipped inverted."""
        found = self._load(self._lightmapped_glb())
        keys = [
            light
            for light in found["lights"]
            if light["type"] == "DirectionalLight" and light["intensity"] > 0
        ]
        self.assertEqual(keys, [], f"a baked scene kept a key light: {found['lights']}")

    def test_an_UNBAKED_scene_keeps_its_key_light(self):
        """The other half, and the reason the gate cannot simply be removed:
        geometry carrying no bake has no lighting of its own."""
        found = self._load(self._animated_glb())

        self.assertEqual(found["lightmapped"], 0)
        self.assertTrue(
            any(
                light["type"] == "DirectionalLight" and light["intensity"] > 0
                for light in found["lights"]
            ),
            f"an unbaked scene lost its key light: {found['lights']}",
        )

    def test_the_SECOND_push_is_still_lit(self):
        """`disposeModel` frees the outgoing model's textures between pushes and
        must leave the session's environment alone: it is the scene's, not the
        model's, and once (when baked materials carried it as their own
        `envMap`) a push disposed it. The second push then renders unlit --
        invisible to every first-push check, and the failure an artist hits
        on their second click rather than their first."""
        found = self._load(
            self._lightmapped_glb(), then_publish=self._lightmapped_glb()
        )

        self.assertGreaterEqual(found["loads"], 2, "the page never swapped")
        material = self._lit(found)
        self.assertTrue(material["hasLightMap"], "the second push lost its bake")
        self.assertTrue(
            found["environment"]["present"],
            "the environment was disposed with the outgoing model",
        )
        self.assertGreater(found["environment"]["intensity"], 0)
        self.assertEqual(found["console_errors"], [])

    # ------------------------------------------------------ what the HUD says
    # The HUD once read "51/57 lightmapped materials": a fraction of three.js
    # material INSTANCES -- one per lightmapped object after the applier's
    # copies, one per loader variant -- against a denominator of every
    # material, baked or never meant to be. Right numbers, misleading
    # sentence. It now counts OBJECTS, and says what disagrees with the bake.

    @staticmethod
    def _rewire(path, edit):
        """Rewrite the GLB at *path* in place through ``edit(gltf)``."""
        with ptk.MeshConvert.open_glb(path) as session:
            edit(session.gltf)
            session.dirty = True
        return path

    def test_the_HUD_counts_objects_lightmapped_not_material_instances(self):
        """A baked machine wearing two materials beside an unbaked prop: the old
        line read "2/3 lightmapped materials"; it is one of two OBJECTS."""
        found = self._load(
            self._baked_room_glb(
                [("machine", [0, 1], "room_Lightmap.exr"), ("prop", [2], None)]
            )
        )
        self.assertEqual(found["errors"], [])
        self.assertEqual(found["lightmapped"], 2, "two material instances bound")
        specs = found["specs"]
        self.assertEqual((specs["objects"], specs["meshes"]), (2, 3))
        self.assertEqual(specs["materials"]["file"], 3)
        self.assertEqual(
            {
                k: specs["lightmaps"][k]
                for k in ("lit", "objects", "baked", "unlit", "foreign")
            },
            {"lit": 1, "objects": 2, "baked": 1, "unlit": [], "foreign": []},
        )
        self.assertEqual(
            found["statLines"],
            ["2 objects · 3 tris · 1.00 × 1.00 × 0.00 m", "1 of 2 objects lightmapped"],
        )
        self.assertIsNone(found["issueLine"])
        self.assertEqual(found["console_errors"], [])

    def test_a_fully_baked_room_says_so_in_words(self):
        """Every object lit -- "all 2 objects", never "2/2"."""
        found = self._load(
            self._baked_room_glb(
                [
                    ("floor", [0], "room_Lightmap.exr"),
                    ("wall", [1], "room_Lightmap.exr"),
                ]
            )
        )
        self.assertEqual(found["statLines"][1], "all 2 objects lightmapped")
        self.assertIsNone(found["issueLine"])

    def test_an_unbaked_model_has_no_lightmap_clause(self):
        found = self._load(self._animated_glb())
        self.assertIsNone(found["specs"]["lightmaps"])
        self.assertEqual(found["statLines"][1], "2 clips")
        self.assertFalse(any("lightmap" in line for line in found["statLines"]))

    def test_a_baked_object_whose_lightmap_did_not_bind_is_named(self):
        """Its map was lost: it renders exactly like an object never baked,
        which is why the page -- not the render -- has to say which."""
        found = self._load(
            self._baked_room_glb(
                [("floor", [0], "room_Lightmap.exr"), ("wall", [1], "gone.exr")]
            )
        )
        self.assertEqual(found["specs"]["lightmaps"]["unlit"], ["wall"])
        self.assertEqual(
            found["issueLine"],
            "⚠ 1 baked object renders unlit — no lightmap bound: wall",
        )
        self.assertEqual(found["console_errors"], [], "a warning, not an error")

    def test_an_object_lit_by_a_bake_it_was_never_in_is_named(self):
        """What a deliverable built before the applier stopped binding in place
        over a material an unbaked object wears looks like: the prop renders
        the floor's lighting, and the old HUD counted that material as
        lightmapped. The bake's own markers say the prop was never baked."""

        def share_the_floors_material(gltf):
            floor = gltf["meshes"][0]["primitives"][0]["material"]
            gltf["meshes"][1]["primitives"][0]["material"] = floor

        glb = self._rewire(
            self._baked_room_glb(
                [("floor", [0], "room_Lightmap.exr"), ("prop", [1], None)]
            ),
            share_the_floors_material,
        )
        found = self._load(glb)
        self.assertEqual(found["specs"]["lightmaps"]["foreign"], ["prop"])
        self.assertIn("never baked with: prop", found["issueLine"])

    def test_an_unreadable_carrier_is_reported_not_swallowed(self):
        """A carrier the page does not read binds nothing -- deliberately -- and
        used to say nothing either, so the room just rendered unlit."""

        def unknown_carrier(gltf):
            gltf["extras"]["lightmap_web"]["carrier"] = "sheen"

        found = self._load(self._rewire(self._lightmapped_glb(), unknown_carrier))
        self.assertEqual(found["lightmapped"], 0)
        self.assertIn("'sheen'", found["issueLine"])
        self.assertTrue(
            any("does not read" in line for line in found["console_errors"]),
            found["console_errors"],
        )

    @staticmethod
    def _drop_second_uv(gltf):
        for mesh in gltf["meshes"]:
            for primitive in mesh["primitives"]:
                primitive["attributes"].pop("TEXCOORD_1", None)

    def test_the_uv_warning_checks_the_set_the_bake_is_sampled_on(self):
        """A bake is sampled on the UV set its manifest names
        (`lightmap_web.uv`), and the warning for a mesh without it looked for
        the SECOND set whatever the manifest said: a bake laid out on the
        first -- which the page renders correctly -- was reported as reading
        one texel. It is checked against the set actually sampled."""

        def on_the_first_set(gltf):
            gltf["extras"]["lightmap_web"]["uv"] = 0
            self._drop_second_uv(gltf)

        found = self._load(self._rewire(self._lightmapped_glb(), on_the_first_set))
        self.assertEqual(self._lit(found)["lightMapChannel"], 0)
        texts = [issue["text"] for issue in found["specs"]["issues"]]
        self.assertFalse(any("UV set" in text for text in texts), texts)

    def test_a_mesh_without_its_bakes_uv_set_is_named(self):
        """The other half: the manifest's set (the second, here) missing from
        the mesh reads one texel of the bake, and the load says so."""
        found = self._load(self._rewire(self._lightmapped_glb(), self._drop_second_uv))
        self.assertEqual(self._lit(found)["lightMapChannel"], 1)
        self.assertIn("no second UV set", found["issueLine"])
        self.assertIn("room", found["issueLine"])

    def test_the_page_measures_the_file_and_its_load(self):
        """Download and parse are timed apart -- a share link's slow first view
        is the download, a big GLB's is the parse -- and the first frame after
        the load, where the programs compile, once it has drawn."""
        glb = self._lightmapped_glb()

        def after_a_frame(server, page):
            page.wait_for_function(
                "() => window.__api.specs && window.__api.specs.load.firstFrameMs !== null",
                timeout=60_000,
            )
            return {
                "later": page.evaluate("() => window.__api.specs.load"),
                "status": page.evaluate(
                    "() => document.getElementById('status').textContent"
                ),
            }

        found = self._load(glb, then=after_a_frame)
        specs = found["specs"]
        self.assertEqual(specs["file"]["bytes"], os.path.getsize(glb))
        self.assertGreater(specs["file"]["jsonBytes"], 0)
        for phase in ("fetchMs", "parseMs", "setupMs"):
            self.assertGreaterEqual(specs["load"][phase], 0, phase)
        self.assertGreater(found["later"]["firstFrameMs"], 0)
        self.assertRegex(found["status"], r"^v1 · \d+(\.\d)? KB · updated ")

    def test_inspect_measures_frames_memory_and_the_file(self):
        """The packaged profiler, opened as a user opens it (the `i` key). The
        atlas both objects bake into is ONE image in GPU memory: each object
        samples it through its own rect, on its own texture, and counted per
        texture one atlas read as two."""
        glb = self._baked_room_glb(
            [
                ("left", [0], "room_Lightmap.exr", [0.5, 1.0, 0.0, 0.0]),
                ("right", [0], "room_Lightmap.exr", [0.5, 1.0, 0.5, 0.0]),
            ]
        )

        def open_inspect(server, page):
            page.keyboard.press("i")
            page.wait_for_function(
                "() => [...document.querySelectorAll('#panels .panel')]"
                ".some((p) => !p.hidden && p.textContent.includes('fps'))",
                timeout=60_000,
            )
            return {
                "panel": page.evaluate(
                    "() => [...document.querySelectorAll('#panels .panel .rows > div')]"
                    ".map((n) => n.textContent)"
                )
            }

        found = self._load(glb, scripts=["inspect"], then=open_inspect)
        self.assertEqual(found["console_errors"], [])
        rows = found["panel"]
        pairs = dict(zip(rows, rows[1:]))  # each label cell, to the cell after it
        for heading in ("Frame", "Model", "GPU memory (estimated)", "File", "Load"):
            self.assertIn(heading, rows)
        self.assertIn("2 of 2 objects lit", pairs["lightmaps"])
        self.assertIn("1 map", pairs["lightmaps"])
        self.assertRegex(pairs["textures"], r"· 1 image$")
        self.assertRegex(pairs["rate"], r"^\d+ fps")

    def test_copy_report_without_a_clipboard_logs_it_instead(self):
        """The clipboard exists only in a secure context, and the page is as
        validly opened over a plain-HTTP LAN address (where it says WebXR needs
        HTTPS). There `navigator.clipboard` is undefined: Copy Report threw in
        its click handler and said nothing. It logs the report and says so."""
        panel = "[...document.querySelectorAll('#panels .panel')][0]"

        def copy_without_a_clipboard(server, page):
            page.evaluate(
                "() => Object.defineProperty(navigator, 'clipboard',"
                " { value: undefined, configurable: true })"
            )
            page.keyboard.press("i")
            page.wait_for_function(f"() => !{panel}.hidden", timeout=30_000)
            page.click("#panels .panel footer button")
            page.wait_for_function(
                f"() => {panel}.querySelector('footer button').textContent"
                " !== 'Copy report'",
                timeout=10_000,
            )
            return {
                "label": page.evaluate(
                    f"() => {panel}.querySelector('footer button').textContent"
                )
            }

        found = self._load(
            self._lightmapped_glb(), scripts=["inspect"], then=copy_without_a_clipboard
        )
        self.assertEqual(found["label"], "Logged to console")
        self.assertEqual(found["console_errors"], [])

    #: Display levels (0-255) a BAKED rough dielectric's face may sit above or
    #: below the level the bake alone dictates (`bake_face_level`). With the
    #: environment published at zero that is the bake path itself -- divisor,
    #: sRGB decode, Lambert, ACES -- and 8-bit rounding is all that moves it.
    BAKE_PATH_TOLERANCE = 6
    #: With the environment at full, only its specular is left to move the same
    #: face, and at roughness 1 that is a few levels (measured: 5). With the
    #: environment's diffuse added on top -- the washed-out bake reported
    #: 2026-09-21 -- the face sat 44 levels above the bake at a QUARTER of the
    #: environment.
    ENV_SPECULAR_TOLERANCE = 20
    #: And the least a baked glossy METAL's face must move by between the two,
    #: its reflections at full strength: it has no diffuse for a bake to carry,
    #: so the environment is all it shows.
    ENV_REFLECTION_FLOOR = 60
    #: The least that metal's face must drop between each published
    #: baked-reflection level and the next one down (full -> quarter -> off).
    REFLECTION_LEVEL_STEP = 8

    def _face_with_and_without_environment(self, pbr, reflections=1.0):
        """Sample the fixture's face lit by the full environment, then again
        after republishing the same fixture with the environment published
        at zero in its own rendering policy. Returns (lit, unlit). Both
        publish *reflections* as the baked-reflection level (full, unless a
        test says otherwise -- the recipe's own default is a quarter)."""
        dark = self._lightmapped_glb(
            bake_value=1.0, pbr=pbr, environment=0.0, baked_reflections=reflections
        )

        def republish_dark(server, page):
            lit = page.evaluate("() => window.__sample()")
            server.publish(dark)
            page.wait_for_function("() => window.__probe.loads >= 2", timeout=180_000)
            return {"lit": lit, "unlit": page.evaluate("() => window.__sample()")}

        found = self._load(
            self._lightmapped_glb(
                bake_value=1.0, pbr=pbr, baked_reflections=reflections
            ),
            then=republish_dark,
        )
        self.assertEqual(found["console_errors"], [])
        self.assertEqual(found["lightmapped"], 1, "fixture lost its bake")
        return found["lit"], found["unlit"]

    def test_a_baked_dielectric_takes_no_DIFFUSE_from_the_environment(self):
        """A lightmap already holds the surface's diffuse lighting, every light
        and the sky included, so the environment's irradiance added on top
        lights it twice. Measured on a production room: a quarter of the
        environment on top of the bake doubled every shadow and lifted every
        surface -- the washed-out look that gets reported as a bake regression.
        The environment stays (see the metal below) and is confined to the
        specular term, so the face of a rough dielectric barely moves with it.
        Asserted on the PIXELS the page draws: the fix is a shader edit, and
        the two ways a shader edit goes wrong -- a replace that matches nothing,
        a chunk that changed under it -- are silent to every property check."""
        lit, unlit = self._face_with_and_without_environment(
            {"metallicFactor": 0.0, "roughnessFactor": 1.0}
        )
        expected = bake_face_level(1.0)

        # The bake path itself, with nothing else in the render: what the file
        # says the surface receives is what reaches the screen.
        self.assertLess(
            abs(unlit - expected),
            self.BAKE_PATH_TOLERANCE,
            f"the bake alone lands at {unlit:.0f}, not the {expected:.0f} its "
            "irradiance dictates",
        )
        # Absolute rather than relative to `unlit`, because the defect this
        # pins was unreachable from the file: the old per-material dimming held
        # a baked material at a quarter of the environment whatever the policy
        # published, so lit and unlit agreed with each other while both sat
        # well above the bake.
        self.assertLess(
            abs(lit - expected),
            self.ENV_SPECULAR_TOLERANCE,
            f"the environment moved a baked matte face to {lit:.0f} from the "
            f"{expected:.0f} its bake dictates: its diffuse is landing on the bake",
        )

    def test_a_baked_metal_still_REFLECTS_the_environment(self):
        """The other half, and why the environment cannot simply go off for a
        baked model (as it once did): a lightmap carries no specular and no
        direction, so without the environment every metal, every gloss and
        every normal map on a baked surface renders dead flat. A glossy metal
        has no diffuse for the bake to hold; the environment is all it shows."""
        lit, unlit = self._face_with_and_without_environment(
            {"metallicFactor": 1.0, "roughnessFactor": 0.3}
        )

        self.assertGreater(
            lit - unlit,
            self.ENV_REFLECTION_FLOOR,
            f"a baked metal shows no environment ({unlit:.0f} -> {lit:.0f})",
        )

    def test_the_published_level_scales_a_baked_materials_reflections(self):
        """REGRESSION (2026-09-21): the export's Baked Reflections level reaches
        the pixels. A studio environment is brighter than the room a bake lit,
        so its reflections at full strength lifted the darkest baked surfaces
        of a production room from 0.06 to 0.22 of display. The level scales
        everything the environment gives a baked material -- measured on a
        glossy metal, which shows nothing else -- and at zero the face is the
        bake path alone, exactly as with the environment published off."""
        pbr = {"metallicFactor": 1.0, "roughnessFactor": 0.3}
        steps = [
            (
                "quarter",
                self._lightmapped_glb(bake_value=1.0, pbr=pbr, baked_reflections=0.25),
            ),
            (
                "off",
                self._lightmapped_glb(bake_value=1.0, pbr=pbr, baked_reflections=0.0),
            ),
            ("dark", self._lightmapped_glb(bake_value=1.0, pbr=pbr, environment=0.0)),
        ]

        def republish(server, page):
            faces = {"full": page.evaluate("() => window.__sample()")}
            for loads, (name, glb) in enumerate(steps, start=2):
                server.publish(glb)
                page.wait_for_function(
                    f"() => window.__probe.loads >= {loads}", timeout=180_000
                )
                faces[name] = page.evaluate("() => window.__sample()")
            return faces

        found = self._load(
            self._lightmapped_glb(bake_value=1.0, pbr=pbr, baked_reflections=1.0),
            then=republish,
        )
        self.assertEqual(found["console_errors"], [])
        self.assertGreater(
            found["full"] - found["quarter"],
            self.REFLECTION_LEVEL_STEP,
            f"full {found['full']:.0f} vs quarter {found['quarter']:.0f}",
        )
        self.assertGreater(
            found["quarter"] - found["off"],
            self.REFLECTION_LEVEL_STEP,
            f"quarter {found['quarter']:.0f} vs off {found['off']:.0f}",
        )
        self.assertLess(
            abs(found["off"] - found["dark"]),
            self.BAKE_PATH_TOLERANCE,
            f"level 0 left {found['off']:.0f} against the env-less "
            f"{found['dark']:.0f}: something still reflects",
        )

    def test_a_baked_normal_mapped_material_relieves_the_bake_and_compiles(self):
        """A lightmap is direction-free irradiance that never consults the
        normal, so on a baked surface the normal map reached the picture only
        through the environment's specular -- measured on a production room as
        ~1% of pixels moving by ~0.4/255 between normalScale 1 and 4. The page
        patches the bake by the map through `onBeforeCompile`.

        Asserted on the program key the patch declares AND on a clean console,
        because the two ways this patch has already gone wrong are both silent
        to everything else: a replace aimed inside an `#include` the hook never
        sees is a no-op, and a GLSL reserved word (`flat`) fails the compile
        with the material falling back to nothing -- both logged, neither
        thrown.
        """
        found = self._load(self._lightmapped_glb(normal_map=True))

        material = self._lit(found)
        self.assertTrue(material["hasNormalMap"], "fixture lost its normal map")
        self.assertEqual(
            material["programKey"], "baked-relief", "the relief patch is not installed"
        )
        self.assertEqual(found["console_errors"], [])
        # And NOT on a baked material without a normal map: there is nothing
        # to relieve by, and the pure bake must stay the pure bake. It still
        # carries the baked patch that keeps the environment off its diffuse.
        plain = self._lit(self._load(self._lightmapped_glb()))
        self.assertEqual(plain["programKey"], "baked")

    #: The key light the relief takes a bake's light to arrive along, as the
    #: published recipe states it (`test_preview_server` pins the page to it).
    KEY_LIGHT = tuple(ptk.MeshConvert.RENDERING_POLICY["keyLight"]["position"])
    #: A matte dielectric: with the environment published at zero, the bake --
    #: and the relief on it -- is the whole picture.
    MATTE = {"metallicFactor": 0.0, "roughnessFactor": 1.0}
    #: The least a 20-degree bump must move a face that faces the key light,
    #: so a relief made inert cannot pass the symmetry check by moving nothing
    #: anywhere (measured: 6 levels on the floor, 10-12 on the walls).
    RELIEF_FLOOR = 3

    def _turn(self, direction):
        """(normal, tangent) that turn the fixture face to point along
        *direction*, its +U along the key light's component in the face's
        plane -- so a bump toward +U leans toward the light's side on every
        face, and a face and its mirror image across the light's horizon wear
        it alike. Straight at or away from the light, any horizontal
        perpendicular does."""
        normal = _unit(direction)
        key = _unit(self.KEY_LIGHT)
        facing = sum(k * n for k, n in zip(key, normal))
        along = tuple(k - facing * n for k, n in zip(key, normal))
        if math.hypot(*along) < 1e-3:
            along = (normal[2], 0.0, -normal[0])
        return [normal, _unit(along)]

    def _baked_face_levels(self, degrees, facings):
        """Display level of the baked fixture face turned each way in *facings*
        (label -> world direction), wearing a normal map leaning *degrees*,
        sampled head-on under the page's own key light."""
        glb = self._lightmapped_glb(
            normal_map=normal_texel_png(degrees),
            bake_value=1.0,
            pbr=self.MATTE,
            environment=0.0,
        )
        turns = {label: self._turn(direction) for label, direction in facings.items()}

        def sample(server, page):
            return {
                "levels": {
                    label: page.evaluate(
                        "(turn) => window.__sampleFacing(...turn)", turn
                    )
                    for label, turn in turns.items()
                }
            }

        found = self._load(glb, then=sample)
        self.assertEqual(found["console_errors"], [])
        self.assertEqual(self._lit(found)["programKey"], "baked-relief")
        return found["levels"]

    def test_a_flat_normal_map_is_the_pure_bake_whichever_way_the_face_points(self):
        """REGRESSION (2026-09-23), reported as the preview rendering some
        normals wrong. The relief took a bake's light to arrive along the
        page's key light as it stood, so on a surface facing AWAY from the
        light the flat response it divides by fell toward zero: a FLAT normal
        map -- which must leave the bake exactly alone -- rendered a face
        turned straight away from the key black (0 where the bake alone puts
        169), 10 degrees off that at 50 and 20 off at 134. The fixture's own
        face points along +Z, where the relief was always right, which is how
        it shipped; so here the face is turned every way under the key light.
        """
        key = _unit(self.KEY_LIGHT)
        away = tuple(-c for c in key)
        aside = _unit((-key[2], 0.0, key[0]))  # perpendicular to the key

        def off_away(degrees):
            lean = math.radians(degrees)
            return tuple(
                a * math.cos(lean) + s * math.sin(lean) for a, s in zip(away, aside)
            )

        levels = self._baked_face_levels(
            0.0,
            {
                "toward the key": key,
                "straight away from the key": away,
                "10 degrees off away": off_away(10),
                "20 degrees off away": off_away(20),
                "floor": (0, 1, 0),
                "ceiling": (0, -1, 0),
                "+X wall": (1, 0, 0),
                "-X wall": (-1, 0, 0),
                "+Z wall": (0, 0, 1),
                "-Z wall": (0, 0, -1),
            },
        )
        expected = bake_face_level(1.0)
        for label, level in levels.items():
            self.assertLess(
                abs(level - expected),
                self.BAKE_PATH_TOLERANCE,
                f"a flat normal map moved the face {label} to {level:.0f} from "
                f"the {expected:.0f} its bake dictates: {levels}",
            )

    def test_a_bump_reads_alike_on_either_side_of_the_key_light(self):
        """REGRESSION (2026-09-23), the other half: the relief's contrast grew
        as a surface turned away from the key light, so one normal map read
        faint on a floor and harsh on the ceiling above it -- the same
        20-degree bump moved the ceiling 40 levels where it moved the floor 6.
        A bake's light reached a surface from its FRONT whichever side of it
        the key sits on, so a face wears a bump exactly as its mirror image
        across the key light's horizon does."""
        levels = self._baked_face_levels(
            20.0,
            {
                "floor": (0, 1, 0),
                "ceiling": (0, -1, 0),
                "+X wall": (1, 0, 0),
                "-X wall": (-1, 0, 0),
                "+Z wall": (0, 0, 1),
                "-Z wall": (0, 0, -1),
            },
        )
        flat = bake_face_level(1.0)
        for toward, away in (
            ("floor", "ceiling"),
            ("+X wall", "-X wall"),
            ("+Z wall", "-Z wall"),
        ):
            self.assertGreater(
                levels[toward] - flat,
                self.RELIEF_FLOOR,
                f"the bump barely moved the {toward} ({flat:.0f} -> "
                f"{levels[toward]:.0f}): the relief has gone inert",
            )
            self.assertLess(
                abs(levels[away] - levels[toward]),
                self.BAKE_PATH_TOLERANCE,
                f"one bump reads {levels[toward] - flat:+.0f} on the {toward} and "
                f"{levels[away] - flat:+.0f} on the {away}: {levels}",
            )

    def test_the_lookdev_area_ships_hidden(self):
        """The normals dial is finished and wired, and deliberately not offered:
        `LOOKDEV_ENABLED` holds the whole lookdev area back until there is a set
        of dials worth a permanent seat in a control bar that must survive a
        phone-sized viewport.

        Asserted on the fixture that WOULD show it -- baked, and carrying a
        normal map -- and on both elements, because "hidden" has two causes
        here: the gate, and a model with nothing for the dial to do. The dial
        itself reporting visible inside a hidden area is what proves it is the
        gate doing the hiding, so removing the gate fails this test rather than
        quietly re-exposing the control.
        """
        found = self._load(self._lightmapped_glb(normal_map=True))

        self.assertTrue(self._lit(found)["hasNormalMap"], "fixture lost its normal map")
        self.assertFalse(
            found["normalsHidden"],
            "the dial hid itself, so this fixture cannot prove the gate closed it",
        )
        self.assertTrue(found["lookdevHidden"], "the lookdev area is being offered")

    def test_the_normals_dial_keeps_the_loaders_green_orientation(self):
        """REGRESSION (2026-09-23): the dial wrote `normalScale.set(value,
        value)`, but GLTFLoader negates `normalScale.y` on a material whose mesh
        carries no TANGENT -- every GLB this pipeline shipped before the DCC
        hand-offs pinned tangents, and this fixture -- because the derivative
        frame it falls back to runs green the other way. So the first touch of
        the dial turned green over on every such baked normal map
        (measured: the loader's (1, -1) became (1.5, 1.5)), which reads as
        inverted normals. Driven through the dial's own input: the area is
        held back behind LOOKDEV_ENABLED, but its wiring is live and ships the
        moment the gate opens. Through 0 as well, where a sign read off the
        value itself is lost."""
        read = """() => {
          const scales = [];
          window.__api.model.traverse((n) => {
            if (n.isMesh) scales.push(n.material.normalScale.toArray());
          });
          return scales;
        }"""

        def drive(server, page):
            loaded = page.evaluate(read)
            for value in ("0", "1.5"):
                page.evaluate(
                    """(value) => {
                      const dial = document.getElementById('normalScale');
                      dial.value = value;
                      dial.dispatchEvent(new Event('input'));
                    }""",
                    value,
                )
            return {"loaded": loaded, "dialled": page.evaluate(read)}

        found = self._load(self._lightmapped_glb(normal_map=True), then=drive)
        self.assertEqual(found["console_errors"], [])
        self.assertEqual(
            found["loaded"], [[1, -1]], "the fixture no longer loads tangent-less"
        )
        self.assertEqual(found["dialled"], [[1.5, -1.5]])

    # ------------------------------------------------------- authored fades
    def _faded_glb(self, manifest=True):
        """Two nodes on ONE material; only one of them is named by a fade.

        The shared material is the point. glTF gives every primitive using a
        material the same instance, and the production assembly shares one
        material across 46 meshes -- so a reader that drives the instance
        instead of a per-subtree copy dissolves the room along with the prop.

        The ramp is authored so the arithmetic is checkable: alpha rises 0 -> 1
        over frames 0-60 at 30fps, i.e. exactly ``clip_time / 2``.

        *manifest=False* strips ``extras.animation_web`` after the fact, which
        is the proof that the fade is carried by the file's OWN animation
        channels and by nothing else.
        """
        gltf = {
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [0, 1, 2, 4]}],
            "scene": 0,
            "nodes": [
                {"name": "FADER", "children": [3]},
                {"name": "UNTOUCHED", "mesh": 0},
                {"name": "data_export"},
                {"name": "FADER_GEO", "mesh": 0},
                # A SECOND node of the same name. glTF does not require unique
                # names and production scenes do not have them (the assembly
                # this was written against ships two `prop533`), so a reader
                # that takes the first match fades one and leaves the other.
                {"name": "FADER", "children": [5]},
                {"name": "FADER_GEO_TWIN", "mesh": 0},
            ],
            "meshes": [
                {"primitives": [{"attributes": {"POSITION": 1}, "material": 0}]}
            ],
            "materials": [{"name": "SHARED", "pbrMetallicRoughness": {}}],
            "animations": [
                {
                    "name": "SHOT",
                    "samplers": [{"input": 0, "output": 2}],
                    "channels": [
                        {"sampler": 0, "target": {"node": 1, "path": "translation"}}
                    ],
                }
            ],
        }
        channels = {
            "fbx_takes": [{"name": "SHOT", "start": 0, "end": 60}],
            "shot_metadata": {"version": 1, "fps": self.FPS},
            "visibility_tracks": {
                "version": 1,
                "fps": self.FPS,
                # Visibility is constant-ON, so no gate is written and the only
                # thing moving the material is the ramp under test.
                "tracks": [
                    {
                        "node": "FADER",
                        "visibility": [[0, 1], [60, 1]],
                        "opacity": [[0, 0.0], [60, 1.0]],
                    }
                ],
                "clip_span": {"*": [0, 60], "SHOT": [0, 60]},
            },
        }
        gltf["nodes"][2]["extras"] = {
            "fromFBX": {
                "userProperties": {
                    k: {"type": "eFbxString", "value": json.dumps(v)}
                    for k, v in channels.items()
                }
            }
        }
        path = self._write(gltf)
        # The real post-conversion chain, in its real order: the gate, then the
        # fade (which needs the gate to have made the node present), then the
        # manifest. What the page loads is what the exporter ships.
        ptk.MeshConvert.apply_glb_visibility(path)
        ptk.MeshConvert.apply_glb_fades(path)
        ptk.MeshConvert.apply_glb_animations(path)
        if not manifest:
            # No `animation_web` at all: what a reader that never ran the
            # manifest pass would hand over. The fade must not depend on it.
            with ptk.MeshConvert.open_glb(path) as session:
                session.gltf.get("extras", {}).pop("animation_web", None)
                session.dirty = True
        return path

    #: Parks the playhead and reads back what the page did to the materials.
    #: Written as its own module rather than folded into PROBE_JS so the
    #: general probe stays the one that describes a plain load.
    FADE_PROBE_JS = """
export default function probe(viewer) {
  const report = { ready: false, errors: [], samples: [] };
  window.__probe = report;
  const TIMES = [0.0, 1.0, 2.0];
  let phase = 0;
  let action = null;
  let faded = null;
  let untouched = null;

  let twin = null;
  viewer.on('load', (detail) => {
    try {
      viewer.scene.traverse((o) => {
        if (o.name === 'FADER_GEO') faded = o;
        if (o.name === 'FADER_GEO_TWIN') twin = o;
        if (o.name === 'UNTOUCHED') untouched = o;
      });
      report.found = { faded: !!faded, untouched: !!untouched, twin: !!twin };
      // Which meshes carry the depth rule, as an OWN property -- an unassigned
      // `onBeforeRender` resolves to Object3D's no-op through the prototype,
      // so identity alone would report every mesh as hooked. The two faded
      // meshes share one material instance, which is exactly why reading
      // `depthWrite` off them cannot tell the two cases apart.
      const hooked = (o) =>
        !!o && Object.prototype.hasOwnProperty.call(o, 'onBeforeRender');
      report.depthHooked = {
        faded: hooked(faded), twin: hooked(twin), untouched: hooked(untouched),
      };
      viewer.playClip('SHOT');
      action = viewer.mixer.clipAction(detail.gltf.animations[0]);
      report.sharedInstance = !!(faded && untouched
        && faded.material === untouched.material);
    } catch (e) { report.errors.push(String(e)); report.ready = true; }
  });

  // The page's loop is mixer.update -> this hook -> render. A time parked here
  // is applied by the NEXT tick's update and drawn by that tick's render, so a
  // reading taken one tick after that sees both the alpha the mixer set AND
  // the depth state the renderer derived from it (`onBeforeRender` runs inside
  // render). Reading on the very next tick instead reports the alpha of the
  // parked time against the depth state of the one BEFORE it -- which passes
  // mid-ramp by coincidence and fails at full alpha.
  let settle = -1;   // ticks to wait before reading the parked time
  viewer.on('frame', () => {
    if (report.ready || !action || !faded) return;
    try {
      if (settle > 0) { settle -= 1; return; }
      if (settle === 0) {
        report.samples.push({
          t: TIMES[phase - 1],
          faded: faded.material.opacity,
          fadedTransparent: faded.material.transparent,
          // The depth rule's RESULT, off the material the renderer drew with
          // last tick -- not a restatement of the alpha.
          fadedDepthWrite: faded.material.depthWrite,
          twin: twin ? twin.material.opacity : null,
          untouched: untouched ? untouched.material.opacity : null,
        });
        settle = -1;
      }
      if (phase >= TIMES.length) { report.ready = true; return; }
      action.paused = true;
      action.time = TIMES[phase];
      phase += 1;
      settle = 1;
    } catch (e) { report.errors.push(String(e)); report.ready = true; }
  });
}
"""

    def _highlighted_glb(self):
        """One node on a shared material, named by a HIGHLIGHT ramp only.

        Visibility is constant-ON and there is no opacity ramp, so the only
        thing moving the material is the emissive channel under test. The ramp
        is authored so the arithmetic is checkable: intensity rises 0 -> 1 over
        frames 0-60 at 30fps, i.e. exactly ``clip_time / 2``; the colour is
        (0.2, 0.5, 1.0), so at t=1.0 the emissive reads (0.1, 0.25, 0.5).
        """
        gltf = {
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [0, 1, 2]}],
            "scene": 0,
            "nodes": [
                {"name": "GLOWER", "children": [3]},
                {"name": "UNTOUCHED", "mesh": 0},
                {"name": "data_export"},
                {"name": "GLOWER_GEO", "mesh": 0},
            ],
            "meshes": [
                {"primitives": [{"attributes": {"POSITION": 1}, "material": 0}]}
            ],
            "materials": [{"name": "SHARED", "pbrMetallicRoughness": {}}],
            "animations": [
                {
                    "name": "SHOT",
                    "samplers": [{"input": 0, "output": 2}],
                    "channels": [
                        {"sampler": 0, "target": {"node": 1, "path": "translation"}}
                    ],
                }
            ],
        }
        channels = {
            "fbx_takes": [{"name": "SHOT", "start": 0, "end": 60}],
            "shot_metadata": {"version": 1, "fps": self.FPS},
            "visibility_tracks": {
                "version": 1,
                "fps": self.FPS,
                "tracks": [
                    {
                        "node": "GLOWER",
                        "highlight": [[0, 0.0], [60, 1.0]],
                        "highlight_color": [0.2, 0.5, 1.0],
                    }
                ],
                "clip_span": {"*": [0, 60], "SHOT": [0, 60]},
            },
        }
        gltf["nodes"][2]["extras"] = {
            "fromFBX": {
                "userProperties": {
                    k: {"type": "eFbxString", "value": json.dumps(v)}
                    for k, v in channels.items()
                }
            }
        }
        path = self._write(gltf)
        ptk.MeshConvert.apply_glb_visibility(path)
        ptk.MeshConvert.apply_glb_fades(path)
        ptk.MeshConvert.apply_glb_animations(path)
        return path

    HIGHLIGHT_PROBE_JS = """
export default function probe(viewer) {
  const report = { ready: false, errors: [], samples: [] };
  window.__probe = report;
  const TIMES = [0.0, 1.0, 2.0];
  let phase = 0;
  let action = null;
  let glower = null;
  let untouched = null;
  viewer.on('load', (detail) => {
    try {
      viewer.scene.traverse((o) => {
        if (o.name === 'GLOWER_GEO') glower = o;
        if (o.name === 'UNTOUCHED') untouched = o;
      });
      report.found = { glower: !!glower, untouched: !!untouched };
      report.sharedInstance = !!(glower && untouched
        && glower.material === untouched.material);
      viewer.playClip('SHOT');
      action = viewer.mixer.clipAction(detail.gltf.animations[0]);
    } catch (e) { report.errors.push(String(e)); report.ready = true; }
  });
  let settle = -1;
  viewer.on('frame', () => {
    if (report.ready || !action || !glower) return;
    try {
      if (settle > 0) { settle -= 1; return; }
      if (settle === 0) {
        const e = glower.material.emissive;
        const u = untouched ? untouched.material.emissive : null;
        report.samples.push({
          t: TIMES[phase - 1],
          glower: [e.r, e.g, e.b],
          glowerTransparent: glower.material.transparent,
          untouched: u ? [u.r, u.g, u.b] : null,
        });
        settle = -1;
      }
      if (phase >= TIMES.length) { report.ready = true; return; }
      action.paused = true;
      action.time = TIMES[phase];
      phase += 1;
      settle = 1;
    } catch (e) { report.errors.push(String(e)); report.ready = true; }
  });
}
"""

    def test_an_authored_highlight_drives_the_materials_emissive(self):
        """The second channel of the pointer table plays like the first.

        A highlight is an additive emissive ramp written to
        ``/materials/N/emissiveFactor``; the page binds it to
        ``material.emissive`` as a colour track. Same mixer, same playhead --
        and no ``transparent``: an additive glow must not pay for alpha sorting.
        """
        probe = self.temp.path(extension=".js")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write(self.HIGHLIGHT_PROBE_JS)
        found = self._load(self._highlighted_glb(), probe=probe)

        self.assertEqual(found["errors"], [])
        self.assertTrue(found["found"]["glower"], "fixture node missing")
        samples = {round(s["t"], 3): s for s in found["samples"]}
        self.assertEqual(sorted(samples), [0.0, 1.0, 2.0])
        for got, want in zip(samples[0.0]["glower"], [0.0, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=3)
        for got, want in zip(samples[1.0]["glower"], [0.1, 0.25, 0.5]):
            self.assertAlmostEqual(got, want, places=3)
        for got, want in zip(samples[2.0]["glower"], [0.2, 0.5, 1.0]):
            self.assertAlmostEqual(got, want, places=3)
        self.assertFalse(samples[2.0]["glowerTransparent"], "highlight must not blend")
        # The material was isolated: the untouched sharer never glows.
        self.assertFalse(found["sharedInstance"])
        for c in samples[2.0]["untouched"]:
            self.assertAlmostEqual(c, 0.0, places=3)

    def _fade_load(self, glb):
        """`_load`, but driving the fade probe instead of the general one."""
        probe = self.temp.path(extension=".js")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write(self.FADE_PROBE_JS)
        return self._load(glb, probe=probe)

    def test_an_authored_fade_actually_ramps_in_the_preview(self):
        """The deliverable's alpha channels are PLAYED, not merely carried.

        three.js implements no extension that animates alpha, so its loader
        drops the KHR_animation_pointer channels on the floor; until the page
        wired them into the mixer itself, every authored fade rendered as a pop
        -- the preview showing something the DCC does not.
        """
        found = self._fade_load(self._faded_glb())

        self.assertEqual(found["errors"], [])
        self.assertTrue(found["found"]["faded"], "fixture node missing")
        samples = {round(s["t"], 3): s for s in found["samples"]}
        self.assertEqual(sorted(samples), [0.0, 1.0, 2.0])
        # alpha = frame / 60 = (t * 30) / 60 = t / 2
        self.assertAlmostEqual(samples[0.0]["faded"], 0.0, places=3)
        self.assertAlmostEqual(samples[1.0]["faded"], 0.5, places=3)
        self.assertAlmostEqual(samples[2.0]["faded"], 1.0, places=3)
        self.assertTrue(samples[1.0]["fadedTransparent"], "alphaMode BLEND not set")

    def test_the_fade_is_carried_by_the_files_own_channels(self):
        """ONE statement of the fade, and it is the glTF's.

        The page plays the deliverable's KHR_animation_pointer channels -- the
        same bytes any extension-aware viewer plays -- by handing them to the
        mixer as ordinary keyframe tracks. With `extras.animation_web` gone
        there is no side block to fall back on, so a fade that still plays can
        only have come from the animation itself.
        """
        found = self._fade_load(self._faded_glb(manifest=False))

        self.assertEqual(found["errors"], [])
        samples = {round(s["t"], 3): s for s in found["samples"]}
        self.assertAlmostEqual(samples[0.0]["faded"], 0.0, places=3)
        self.assertAlmostEqual(samples[1.0]["faded"], 0.5, places=3)
        self.assertAlmostEqual(samples[2.0]["faded"], 1.0, places=3)

    def test_every_node_of_that_name_fades_not_just_the_first(self):
        """glTF names are not unique, and the gate already drives all matches.

        A fade that drove only the first would leave the twin at full alpha
        while its visibility gate switched -- which reads as the fade being
        broken on some objects and not others.
        """
        found = self._fade_load(self._faded_glb())

        self.assertEqual(found["errors"], [])
        self.assertTrue(found["found"]["twin"], "fixture twin missing")
        samples = {round(s["t"], 3): s for s in found["samples"]}
        self.assertAlmostEqual(samples[1.0]["twin"], 0.5, places=3)
        self.assertAlmostEqual(samples[2.0]["twin"], 1.0, places=3)

    def test_the_fade_does_not_dissolve_everything_sharing_the_material(self):
        """Only the named node's subtree fades; the shared instance is copied."""
        found = self._fade_load(self._faded_glb())

        self.assertEqual(found["errors"], [])
        self.assertFalse(
            found["sharedInstance"],
            "the faded subtree still shares its material with un-faded geometry",
        )
        samples = {round(s["t"], 3): s for s in found["samples"]}
        self.assertAlmostEqual(samples[0.0]["untouched"], 1.0, places=3)
        self.assertAlmostEqual(samples[1.0]["untouched"], 1.0, places=3)

    def test_a_faded_surface_writes_depth_only_once_it_is_opaque(self):
        """The artifact that reads as inverted normals.

        glTF has no depth field, so GLTFLoader derives one: ``alphaMode: BLEND``
        turns depth writes OFF. Right for a window, wrong for a solid object
        that merely fades -- and since alphaMode belongs to the material rather
        than to the clip, an object that fades in during one shot then renders
        with its far faces drawn over its near ones in every other shot too
        (measured on the production assembly: 15 materials, all of them
        depthWrite false at full alpha). Restoring it wholesale would be the
        opposite bug, so the state follows the alpha the mixer is driving.
        """
        found = self._fade_load(self._faded_glb())

        self.assertEqual(found["errors"], [])
        samples = {round(s["t"], 3): s for s in found["samples"]}
        # Mid-ramp at half alpha: no depth writes, or a half-transparent
        # object punches a hole in whatever is drawn behind it.
        self.assertAlmostEqual(samples[1.0]["faded"], 0.5, places=3)
        self.assertIs(samples[1.0]["fadedDepthWrite"], False)
        # Fully faded in: depth writes back on, and the mesh occludes itself.
        self.assertAlmostEqual(samples[2.0]["faded"], 1.0, places=3)
        self.assertIs(samples[2.0]["fadedDepthWrite"], True)

    def test_the_depth_rule_reaches_every_mesh_sharing_a_faded_material(self):
        """One representative mesh per material carries the mixer's track --
        two would drive the same property twice a frame -- but the depth rule
        is an onBeforeRender hook, so it only runs for meshes it is installed
        on. Left on the representative alone it goes stale the moment that one
        is frustum-culled and its twin is not.

        Asserted on the hook rather than on ``depthWrite``: the two faded
        meshes share one material instance, so the VALUE agrees whether or not
        the fix is in.
        """
        found = self._fade_load(self._faded_glb())

        self.assertEqual(found["errors"], [])
        self.assertTrue(found["found"]["twin"], "fixture twin missing")
        self.assertTrue(found["depthHooked"]["faded"])
        self.assertTrue(found["depthHooked"]["twin"])
        # Not on the un-faded mesh: it keeps its own opaque material, and
        # hooking it would be claiming it fades.
        self.assertFalse(found["depthHooked"]["untouched"])

    # ------------------------------------------------------------ a share
    def test_a_guest_sees_the_model_and_nothing_that_writes(self):
        """The page as a share delivers it: through the guest listener, under
        a public name, in a secure context -- which is what a tunnel adds, so
        the flags stand in for one with no network. The guest loads the model
        and can enter VR (``navigator.xr``), is marked a guest, never receives
        the owner-only script (no Export Image button), is counted while open
        and retired by its own close beacon -- and none of it touches the
        owner's viewer state."""
        import time

        from playwright.sync_api import sync_playwright

        server = ptk.PreviewServer(viewer=True, title="live-test", port=0).start()
        server.set_scripts(["snapshot"])
        server.add_script("probe", self.probe)
        server.publish(self._animated_glb())
        port = server.start_guest()
        server.admit_host(f"share.example.test:{port}")
        origin = f"http://share.example.test:{port}"
        console = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    channel="msedge",
                    headless=True,
                    args=[
                        "--enable-unsafe-swiftshader",
                        "--host-resolver-rules=MAP share.example.test 127.0.0.1",
                        f"--unsafely-treat-insecure-origin-as-secure={origin}",
                    ],
                )
                page = browser.new_page()
                page.on("pageerror", lambda e: console.append(f"[pageerror] {e}"))
                page.goto(origin + "/", wait_until="domcontentloaded", timeout=120_000)
                page.wait_for_function(
                    "() => window.__probe && window.__probe.ready === true",
                    timeout=180_000,
                )
                found = page.evaluate(
                    """() => ({
                      guest: window.__api.guest,
                      secure: window.isSecureContext,
                      xr: 'xr' in navigator,
                      meshes: window.__probe.meshes,
                      buttons: [...document.querySelectorAll('#controls button')]
                        .filter((b) => !b.closest('[hidden]'))
                        .map((b) => b.textContent.trim()),
                      saveHidden: document.getElementById('normalSave').hidden,
                    })"""
                )
                found["guests_open"] = server.guest_count()
                page.close(run_before_unload=True)
                deadline = time.monotonic() + 10  # the beacon is asynchronous
                while server.guest_count() and time.monotonic() < deadline:
                    time.sleep(0.1)
                found["guests_closed"] = server.guest_count()
                found["owner_viewer"] = server.has_viewer()
                browser.close()
        finally:
            server.stop()

        self.assertEqual(console, [])
        self.assertIs(found["guest"], True)
        self.assertIs(found["secure"], True)
        self.assertIs(found["xr"], True, "a secure-context guest must get WebXR")
        self.assertEqual(found["meshes"], 1)
        self.assertNotIn("Export Image", found["buttons"])
        self.assertIs(found["saveHidden"], True)
        self.assertEqual((found["guests_open"], found["guests_closed"]), (1, 0))
        self.assertIs(found["owner_viewer"], False)

    # ------------------------------------------------- getting around in VR
    # The page's headset locomotion, driven through the same `update(delta,
    # input)` the page feeds from each XRFrame -- a headless browser has no
    # headset, but the rig's behaviour is all in that step, on plain values.
    # One page serves every scenario (`_locomotion`), the rig reset between.

    def _walkable_glb(self, start=False):
        """A floor to walk on, a 0.3 m step, a 1 m ledge and a wall facing +z.

        Authored away from the origin and off the floor (every point shifted
        by SHIFT), so the page has to centre it and stand it down: laid out,
        the scene coordinates are the ones written here.

        *start* adds the nodes a view starts at, as the DCCs deliver them: a
        camera ``user_pos`` 1.6 m above the platform, looking back along -x
        and 20 degrees down, its clip planes in the wrong units the way a
        Blender one arrives; and, under a group, a namespaced
        ``set:spawn_empty`` on the floor the way a Blender Empty arrives --
        turned -90 degrees about x and scaled 100, so its -z points down and
        its +y, Blender's forward, along its heading of -45 degrees.
        """

        def facing_z(x0, x1, y0, y1, z):
            return [(x0, y0, z), (x1, y0, z), (x0, y1, z),
                    (x1, y0, z), (x1, y1, z), (x0, y1, z)]  # fmt: skip

        shift = (10.0, 2.0, -7.0)
        gltf, blob = self._parts_gltf(
            {
                "floor": _floor(-6, 6, -6, 6, 0),
                "platform": _floor(2, 4, -1, 1, 0.3),
                "ledge": _floor(-4, -2, 1, 3, 1.0),
                "wall": facing_z(-3, 3, 0, 3, -5),
            },
            shift,
        )
        if start:
            first = len(gltf["nodes"])
            gltf["cameras"] = [
                {
                    "type": "perspective",
                    "perspective": {
                        "yfov": 0.5,
                        "aspectRatio": 1.5,
                        "znear": 10.0,
                        "zfar": 100000.0,
                    },
                }
            ]
            gltf["nodes"] += [
                {
                    "name": "user_pos",
                    "camera": 0,
                    "translation": [3 + shift[0], 1.9 + shift[1], 0 + shift[2]],
                    "rotation": _yaw_pitch(90, -20),
                },
                {
                    "name": "set_grp",
                    "translation": list(shift),
                    "children": [first + 2],
                },
                {
                    "name": "set:spawn_empty",
                    "translation": [-5, 0, -4],
                    "rotation": _yaw_pitch(-45, -90),
                    "scale": [100, 100, 100],
                },
            ]
            gltf["scenes"][0]["nodes"] += [first, first + 1]
        return self._pack_glb(
            gltf, blob, "walkable_start.glb" if start else "walkable.glb"
        )

    @staticmethod
    def _parts_gltf(parts, shift=(0.0, 0.0, 0.0)):
        """A glTF of one mesh node per ``name: triangles`` in *parts*, every
        point moved by *shift*; returns ``(gltf, blob)`` for :meth:`_pack_glb`."""
        blob = b""
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": list(range(len(parts)))}],
            "nodes": [],
            "meshes": [],
            "accessors": [],
            "bufferViews": [],
        }
        for index, (name, triangles) in enumerate(parts.items()):
            points = [tuple(c + s for c, s in zip(p, shift)) for p in triangles]
            data = struct.pack(f"<{len(points) * 3}f", *(c for p in points for c in p))
            gltf["bufferViews"].append(
                {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
            )
            gltf["accessors"].append(
                {
                    "bufferView": index,
                    "componentType": 5126,
                    "count": len(points),
                    "type": "VEC3",
                    "min": [min(p[axis] for p in points) for axis in range(3)],
                    "max": [max(p[axis] for p in points) for axis in range(3)],
                }
            )
            gltf["meshes"].append(
                {"name": name, "primitives": [{"attributes": {"POSITION": index}}]}
            )
            gltf["nodes"].append({"name": name, "mesh": index})
            blob += data
        return gltf, blob

    def _pack_glb(self, gltf, blob, name):
        """*gltf* and its binary *blob* as the GLB *name*, in a tracked folder.

        No BIN chunk (and no buffer) when *blob* is empty: a file carrying
        nothing but nodes is a valid GLB, and a zero-length buffer is not.
        """
        if blob:
            gltf["buffers"] = [{"byteLength": len(blob)}]
        json_bytes = json.dumps(gltf).encode("utf-8")
        json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
        blob += b"\0" * ((4 - len(blob) % 4) % 4)
        chunks = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
        if blob:
            chunks += struct.pack("<I4s", len(blob), b"BIN\0") + blob
        out = os.path.join(self.temp.dir_path(), name)
        with open(out, "wb") as fh:
            fh.write(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)))
            fh.write(chunks)
        return out

    def _ground_glb(self, size=150.0):
        """One flat ground *size* m square: framed whole, the desktop camera
        stands ~230 m off, and the near plane sized to that is 0.23 m."""
        half = size / 2
        gltf, blob = self._parts_gltf({"ground": _floor(-half, half, -half, half, 0)})
        return self._pack_glb(gltf, blob, "ground.glb")

    def _terrain_glb(self, cells=200, span=40.0, amplitude=1.0, wavelength=12.0):
        """Dense rolling ground: *cells* x *cells* quads -- 80,000 triangles by
        default, as ONE indexed mesh -- over *span* m, at height
        ``amplitude * (1 - cos(kx) cos(kz))``: 0 at the origin, where a session
        with no start stands, rising to twice *amplitude*, and walkable
        everywhere (28 degrees at its steepest). Symmetric about the origin
        and lowest there, so the page's layout leaves it where it is written.
        """
        import numpy as np

        axis = np.linspace(-span / 2, span / 2, cells + 1)
        x, z = np.meshgrid(axis, axis)  # rows run along z, columns along x
        k = 2 * np.pi / wavelength
        y = amplitude * (1 - np.cos(k * x) * np.cos(k * z))
        positions = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
        column, row = np.meshgrid(np.arange(cells), np.arange(cells))
        a = (row * (cells + 1) + column).ravel()
        b, c = a + 1, a + cells + 1
        # (a, c, b) and (b, c, d) face +y, as `_floor`'s pair does.
        indices = np.stack([a, c, b, b, c, c + 1], axis=1).astype(np.uint32).ravel()
        vertex_bytes = positions.tobytes()
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"name": "terrain", "mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": len(vertex_bytes)},
                {
                    "buffer": 0,
                    "byteOffset": len(vertex_bytes),
                    "byteLength": indices.nbytes,
                },
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": len(positions),
                    "type": "VEC3",
                    "min": positions.min(axis=0).tolist(),
                    "max": positions.max(axis=0).tolist(),
                },
                {
                    "bufferView": 1,
                    "componentType": 5125,
                    "count": len(indices),
                    "type": "SCALAR",
                },
            ],
        }
        return self._pack_glb(gltf, vertex_bytes + indices.tobytes(), "terrain.glb")

    def _camera_only_glb(self):
        """A push holding the start camera and nothing drawable: ``user_pos``
        1.6 m up, 4 m back along +z, looking down -z."""
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "cameras": [
                {
                    "type": "perspective",
                    "perspective": {"yfov": 0.8, "aspectRatio": 1.5, "znear": 0.1},
                }
            ],
            "nodes": [{"name": "user_pos", "camera": 0, "translation": [0, 1.6, 4]}],
        }
        return self._pack_glb(gltf, b"", "camera_only.glb")

    _locomotion_found = None

    def _locomotion(self):
        """Every scenario's findings, from one page load shared by the class."""
        cls = type(self)
        if cls._locomotion_found is None:
            found = self._load(
                self._walkable_glb(),
                then=lambda server, page: {"motion": page.evaluate(LOCOMOTION_JS)},
            )
            cls._locomotion_found = found
        return cls._locomotion_found

    def test_the_model_stands_at_true_scale_centred_on_the_floor(self):
        """No fitted mode: the model keeps its size, and only moves to stand
        centred on the floor. Fitted, this 3 m fixture would be 1.5 m tall."""
        found = self._locomotion()
        layout = found["motion"]["layout"]

        self.assertEqual(found["console_errors"], [])
        self.assertIs(layout["scaleToggle"], False, "the Scale toggle is gone")
        self.assertEqual(layout["scale"], [1, 1, 1])
        for got, want in zip(layout["box"], ([-6, 0, -6], [6, 3, 6])):
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, places=4)
        self.assertEqual(layout["boxAfterR"], layout["box"], "'r' still rescales")

    def test_a_flick_turns_45_degrees_once_about_the_head(self):
        """A held stick is one turn, not one a frame, and the view pivots
        where you stand instead of swinging you round the play space."""
        turn = self._locomotion()["motion"]["turn"]

        self.assertEqual(turn["events"], ["turn"])
        self.assertAlmostEqual(turn["yaw"], -math.pi / 4)  # right: clockwise
        for before, after in zip(turn["headBefore"], turn["headAfter"]):
            self.assertAlmostEqual(before, after, places=9)
        self.assertAlmostEqual(turn["yawBack"], 0.0, places=12)

    def test_forward_aims_and_release_jumps_through_a_blink(self):
        """The jump waits for black, then puts the HEAD over the landing (not
        the play space's centre) at the landing's height, heading unchanged."""
        tp = self._locomotion()["motion"]["teleport"]

        self.assertEqual(tp["aimEvents"], ["aim"])
        self.assertGreater(tp["arcPoints"], 2)
        self.assertIs(tp["target"]["valid"], True)
        self.assertAlmostEqual(tp["target"]["point"][1], 0.0, places=5)
        self.assertGreater(tp["fadeWhileLeaving"], 0.0)
        self.assertEqual(
            (tp["rigWhileLeaving"]["x"], tp["rigWhileLeaving"]["z"]),
            (0, 0),
            "the jump happened before the view went black",
        )
        self.assertEqual(tp["events"].count("land"), 1)
        self.assertAlmostEqual(tp["headAfter"][0], tp["target"]["point"][0], places=6)
        self.assertAlmostEqual(tp["headAfter"][2], tp["target"]["point"][2], places=6)
        self.assertAlmostEqual(tp["rigAfter"]["y"], 0.0, places=5)
        self.assertEqual(tp["rigAfter"]["yaw"], 0)
        self.assertEqual(tp["fadeAfter"], 0)

    def test_a_jump_onto_a_raised_floor_stands_you_on_it(self):
        platform = self._locomotion()["motion"]["platform"]

        self.assertIs(platform["found"], True, "no aim landed on the platform")
        self.assertAlmostEqual(platform["rig"]["y"], 0.3, places=5)
        self.assertAlmostEqual(platform["headAfter"][1], 1.9, places=5)
        self.assertAlmostEqual(platform["headAfter"][0], platform["point"][0], places=6)

    def test_a_wall_is_no_landing(self):
        wall = self._locomotion()["motion"]["wall"]

        self.assertIsNotNone(wall["target"], "the arc never reached the wall")
        self.assertAlmostEqual(wall["target"]["point"][2], -5.0, places=5)
        self.assertIs(wall["target"]["valid"], False)
        self.assertEqual(wall["rig"], {"x": 0, "y": 0, "z": 0, "yaw": 0})

    def test_a_pivot_turned_since_load_is_struck_where_it_now_stands(self):
        """A script turning the pivot (the turntable, before a session) moves
        every mesh; surfaces read only at load would leave each box where its
        mesh used to be, and the arc would pass through the wall."""
        turned = self._locomotion()["motion"]["turnedPivot"]

        self.assertIsNotNone(turned["target"], "the arc struck nothing")
        self.assertAlmostEqual(turned["target"]["point"][2], 5.0, places=4)
        self.assertIs(turned["target"]["valid"], False)

    def test_pulling_back_cancels_the_aim(self):
        cancel = self._locomotion()["motion"]["cancel"]

        self.assertIs(cancel["aimedValid"], True)
        self.assertIs(cancel["aimingAfterPullBack"], False)
        self.assertEqual(cancel["rig"], {"x": 0, "y": 0, "z": 0, "yaw": 0})

    def test_the_left_stick_walks_where_you_look_eased_in_and_out(self):
        walk = self._locomotion()["motion"]["walk"]
        full_step = 2.0 / 72  # a frame at full speed

        self.assertGreater(walk["firstStep"], 0.0)
        self.assertLess(walk["firstStep"], full_step / 2, "it starts at full speed")
        self.assertAlmostEqual(walk["oneSecond"]["x"], 0.0, places=6)
        self.assertTrue(1.7 < walk["oneSecond"]["z"] < 1.9, walk["oneSecond"])
        self.assertGreater(walk["vignetteMoving"], 0.3)
        coasted = walk["stopped"]["z"] - walk["oneSecond"]["z"]
        self.assertTrue(0.0 < coasted < 0.3, f"coasted {coasted} m to a stop")
        self.assertLess(walk["settled"]["z"] - walk["stopped"]["z"], 0.01)
        self.assertLess(walk["vignetteStopped"], 0.02)

    def test_walking_follows_a_step_up_but_not_a_ledge(self):
        found = self._locomotion()["motion"]

        self.assertTrue(2 < found["stepUp"]["x"] < 4, found["stepUp"])
        self.assertAlmostEqual(found["stepUp"]["y"], 0.3, places=3)
        self.assertTrue(1 < found["ledge"]["head"][2] < 3, "never reached the ledge")
        self.assertAlmostEqual(found["ledge"]["rig"]["y"], 0.0, places=6)

    def test_the_reference_offset_is_the_rigs_inverse(self):
        """What the page hands WebXR must put tracked poses exactly where the
        rig says they are -- an error here moves the world, not the viewer."""
        offset = self._locomotion()["motion"]["offset"]

        for via, direct in zip(offset["viaOffset"], offset["toScene"]):
            self.assertAlmostEqual(via, direct, places=9)

    # --------------------------------------------------- where a view starts
    # The start node the server names (`user_pos` by default): a camera the
    # desktop view opens through and a headset session stands under. One page
    # for every scenario, as above; the server's name changed live on it.

    _start_found = None

    def _start(self):
        """The start scenarios' findings, from one page load shared by the class."""
        cls = type(self)
        if cls._start_found is None:

            def scenario(server, page):
                found = {"first": page.evaluate(START_JS)}
                server.locomotion = False
                page.wait_for_function(
                    "() => window.__api.locomotion.enabled === false", timeout=30_000
                )
                found["locked"] = page.evaluate(LOCKED_JS)
                server.locomotion = True
                page.wait_for_function(
                    "() => window.__api.locomotion.enabled === true", timeout=30_000
                )
                server.user_pos = "spawn_empty"
                page.wait_for_function(
                    "() => { const s = window.__api.headset.start;"
                    " return !!s && Math.abs(s.point.x + 5) < 1e-6; }",
                    timeout=30_000,
                )
                found["renamed"] = page.evaluate(
                    "() => { const s = window.__api.headset.start;"
                    " return { point: s.point.toArray(), heading: s.heading,"
                    " camera: window.__api.camera.position.toArray() }; }"
                )
                server.user_pos = None
                page.wait_for_function(
                    "() => window.__api.headset.start === null", timeout=30_000
                )
                found["none"] = page.evaluate(NO_START_JS)
                return found

            cls._start_found = self._load(self._walkable_glb(start=True), then=scenario)
        return cls._start_found

    def test_a_start_camera_opens_the_desktop_view_through_it(self):
        """Its position, the way it looks, and its HORIZONTAL field of view --
        the one both DCCs hold for a landscape frame -- with the orbit turning
        about what it looks at: the floor 1.9 m below, 20 degrees down."""
        found = self._start()
        desktop = found["first"]["desktop"]

        self.assertEqual(found["console_errors"], [])
        for got, want in zip(desktop["position"], (3, 1.9, 0)):
            self.assertAlmostEqual(got, want, places=4)
        down = math.radians(20)
        for got, want in zip(desktop["look"], (-math.cos(down), -math.sin(down), 0)):
            self.assertAlmostEqual(got, want, places=4)
        reach = 1.9 / math.sin(down)
        for got, want in zip(desktop["target"], (3 - reach * math.cos(down), 0, 0)):
            self.assertAlmostEqual(got, want, places=3)
        across = math.tan(0.25) * 1.5  # the file's yfov and aspect: tan(hfov / 2)
        want_fov = math.degrees(2 * math.atan(across / desktop["aspect"]))
        self.assertAlmostEqual(desktop["fov"], want_fov, places=4)
        self.assertLess(desktop["far"], 1000, "the file's clip planes were used")
        self.assertLess(desktop["near"], 0.05)

    def test_a_session_starts_on_the_floor_under_the_start_facing_where_it_looks(self):
        """Only where and which way: the camera's height is the viewer's eyes,
        not their floor, and its pitch is the headset's to decide. The first
        frame goes out black -- it was posed from where the viewer stood -- and
        the view fades in at the start."""
        first = self._start()["first"]
        start, arrive = first["start"], first["arrive"]

        for got, want in zip(start["point"], (3, 0.3, 0)):
            self.assertAlmostEqual(got, want, places=5)
        self.assertAlmostEqual(start["heading"], math.pi / 2, places=6)
        self.assertEqual(arrive["events"], ["arrive"])
        self.assertEqual(arrive["fade"], 1)
        self.assertIs(arrive["changed"], True)
        for got, want in zip(arrive["head"], (3, 1.9, 0)):
            self.assertAlmostEqual(got, want, places=6)
        facing = arrive["facing"]
        self.assertAlmostEqual(
            math.atan2(-facing[0], -facing[2]), math.pi / 2, places=6
        )
        self.assertAlmostEqual(arrive["rig"]["y"], 0.3, places=6)
        self.assertEqual(arrive["later"], [], "a session arrives once")
        self.assertEqual(arrive["fadeAfter"], 0)
        self.assertEqual(arrive["rigAfter"], arrive["rig"])
        self.assertEqual(
            first["again"], ["arrive"], "the next session starts there too"
        )

    def test_the_start_name_is_live_and_finds_a_namespaced_empty(self):
        """A name set on the server reaches the page without a publish; a
        namespace prefix is looked through; an Empty's heading is its +y
        (its -z points at the floor); and the desktop view stays put."""
        found = self._start()
        renamed = found["renamed"]

        for got, want in zip(renamed["point"], (-5, 0, -4)):
            self.assertAlmostEqual(got, want, places=4)
        self.assertAlmostEqual(renamed["heading"], -math.pi / 4, places=6)
        for got, want in zip(renamed["camera"], found["first"]["desktop"]["position"]):
            self.assertAlmostEqual(got, want, places=9)

    def test_a_recenter_puts_a_viewer_still_at_the_start_back_there(self):
        """The headset's own recenter resets the tracked space under the viewer:
        one still standing where the start put them is put back -- head over
        it, facing its way -- while one who has walked off keeps their place."""
        recenter = self._start()["first"]["recenter"]

        self.assertEqual(recenter["events"], ["arrive"])
        self.assertAlmostEqual(recenter["head"][0], 3, places=6)
        self.assertAlmostEqual(recenter["head"][2], 0, places=6)
        facing = recenter["facing"]
        self.assertAlmostEqual(
            math.atan2(-facing[0], -facing[2]), math.pi / 2, places=6
        )
        self.assertEqual(recenter["afterWalking"], [])

    def test_with_locomotion_off_a_start_camera_holds_the_viewer(self):
        """Switched off on the server, live: the session still arrives under
        the camera, then no stick moves it -- no walk, turn, aim or jump, and
        nothing of it drawn -- and a recenter from across the room puts the
        viewer back on the viewpoint."""
        locked = self._start()["locked"]

        self.assertIs(locked["enabled"], False)
        self.assertEqual(locked["arrive"], ["arrive"])
        self.assertEqual(locked["events"], [])
        self.assertEqual(locked["after"], locked["placed"])
        self.assertIs(locked["aiming"], False)
        self.assertEqual(locked["arc"], 0)
        self.assertEqual(locked["vignette"], 0)
        for got, want in ((locked["head"][0], 3), (locked["head"][2], 0)):
            self.assertAlmostEqual(got, want, places=6)
        self.assertEqual(locked["recenter"], ["arrive"])
        self.assertAlmostEqual(locked["recenterHead"][0], 3, places=6)
        self.assertAlmostEqual(locked["recenterHead"][2], 0, places=6)
        facing = locked["recenterFacing"]
        self.assertAlmostEqual(
            math.atan2(-facing[0], -facing[2]), math.pi / 2, places=6
        )

    def test_switched_off_mid_session_the_viewer_is_taken_back_to_the_start(self):
        """Locomotion off keeps a session where it started -- also when the
        switch is thrown on a viewer the sticks already carried off. They are
        taken back under the camera on the next frame, through the blink, and
        a recenter still returns them. Before, they stayed wherever they had
        walked to, and a recenter -- which re-arrived only a viewer still
        standing at the start -- left them there too."""
        mid = self._start()["locked"]["midSession"]

        walked = math.hypot(
            mid["walked"]["x"] - mid["placed"]["x"],
            mid["walked"]["z"] - mid["placed"]["z"],
        )
        self.assertGreater(walked, 1.0, "the fixture walk went nowhere")
        self.assertEqual(mid["events"], ["arrive"])
        self.assertEqual(mid["fade"], 1, "the move was not hidden")
        for got, want in ((mid["head"][0], 3), (mid["head"][2], 0)):
            self.assertAlmostEqual(got, want, places=6)
        self.assertEqual(mid["later"], [], "taken back once, then held")
        self.assertEqual(mid["fadeAfter"], 0)
        self.assertEqual(mid["recenter"], ["arrive"])
        self.assertAlmostEqual(mid["recenterHead"][0], 3, places=6)
        self.assertAlmostEqual(mid["recenterHead"][2], 0, places=6)

    def test_without_a_start_a_session_starts_where_the_headset_stands(self):
        none = self._start()["none"]

        self.assertIsNone(none["start"])
        self.assertEqual(none["events"], [])
        self.assertEqual(none["rig"], {"x": 0, "y": 0, "z": 0, "yaw": 0})
        self.assertEqual(none["fade"], 0)

    def test_without_a_start_locomotion_off_holds_where_the_rig_began(self):
        """With no start node a session starts where the headset stands, and
        that is where switching locomotion off holds it: a viewer walked off
        is taken back to it, from black."""
        none = self._start()["none"]

        self.assertLess(none["walked"]["z"], -1.0, "the fixture walk went nowhere")
        self.assertEqual(none["switchedOff"]["events"], ["arrive"])
        self.assertEqual(none["switchedOff"]["rig"], none["rig"])
        self.assertEqual(none["switchedOff"]["fade"], 1)

    # ------------------------------------------------ a push, as it lands
    # What a push does to the page around it: where the new model stands when
    # a script has turned the pivot, and what a 'load' subscriber is told.

    _repush_found = None

    def _repush(self):
        """A room with a start camera, then -- the pivot turned 45 degrees in
        between, as the turntable leaves it -- the same room without one."""
        cls = type(self)
        if cls._repush_found is None:
            second = self._walkable_glb()

            def turn_then_push(server, page):
                page.evaluate("() => { window.__api.pivot.rotation.y = Math.PI / 4; }")
                server.publish(second)
                page.wait_for_function(
                    "() => window.__probe.loads >= 2", timeout=180_000
                )
                return {"placed": page.evaluate(PLACED_JS)}

            cls._repush_found = self._load(
                self._walkable_glb(start=True), then=turn_then_push
            )
        return cls._repush_found

    def test_a_push_under_a_turned_pivot_lands_centred_at_its_own_size(self):
        """The turntable keeps its angle across pushes, and the model is laid
        out in the PIVOT's frame -- but was measured in the world's, so under
        a turned pivot an off-origin model landed off-centre (measured: a
        quarter turn stood one at [3.0, 0.5, -17.0]) and the HUD quoted the
        size of its turned bounding box. Laid out right, the room turns about
        its own centre: its world box stays centred on the axis, on the floor."""
        found = self._repush()
        placed = found["placed"]

        self.assertEqual(found["errors"], [])
        self.assertEqual(found["console_errors"], [])
        self.assertAlmostEqual(placed["centre"][0], 0.0, places=4)
        self.assertAlmostEqual(placed["centre"][2], 0.0, places=4)
        self.assertAlmostEqual(placed["min"][1], 0.0, places=4)
        for axis, want in zip("xyz", (12, 3, 12)):
            self.assertAlmostEqual(placed["size"][axis], want, places=4, msg=axis)

    def test_a_load_subscriber_is_told_the_start_of_the_model_just_loaded(self):
        """`'load'` fired before the page had found the new model's start node
        or laid it out, so `viewer.headset.start` read during it answered from
        the model before -- nothing on the first push, and the REPLACED model's
        node (disposed with it) on the next. Now it is the model just loaded,
        standing on the floor: the first push's camera, the second's none."""
        starts = self._repush()["startsAtLoad"]

        self.assertEqual(len(starts), 2, starts)
        self.assertIsNotNone(starts[0], "the first push's start was not found")
        for got, want in zip(starts[0], (3, 0.3, 0)):
            self.assertAlmostEqual(got, want, places=5)
        self.assertIsNone(starts[1], "the second push answered with the first's")

    def test_a_push_of_the_start_camera_alone_keeps_finite_clip_planes(self):
        """Nothing drawable is an EMPTY box, whose distance to any point is
        Infinity: the view opened through the camera with a far plane of
        Infinity and a NaN projection -- in a session three.js hands that far
        plane to `updateRenderState`, which refuses it and stops the XR loop."""
        found = self._load(
            self._camera_only_glb(),
            then=lambda server, page: {"placed": page.evaluate(PLACED_JS)},
        )
        placed = found["placed"]

        self.assertEqual(found["errors"], [])
        self.assertIs(placed["finite"], True, placed)
        for got, want in zip(placed["position"], (0, 1.6, 4)):
            self.assertAlmostEqual(got, want, places=5)
        self.assertLess(placed["near"], 0.05)
        self.assertGreater(placed["far"], 10, "the floor grid is out of reach")

    # ------------------------------------------------- a session, begun and ended
    _session_found = None

    def _session(self):
        """The session scenarios' findings (`SESSION_JS`) over a 150 m ground."""
        cls = type(self)
        if cls._session_found is None:
            cls._session_found = self._load(
                self._ground_glb(),
                then=lambda server, page: {"session": page.evaluate(SESSION_JS)},
            )
        return cls._session_found

    def _assertSameView(self, got, want):
        for key in ("position", "quaternion", "target"):
            for g, w in zip(got[key], want[key]):
                self.assertAlmostEqual(g, w, places=6, msg=key)
        for key in ("fov", "near", "far", "focal"):
            self.assertAlmostEqual(got[key], want[key], places=6, msg=key)

    def test_a_large_scene_does_not_clip_the_headsets_overlay(self):
        """three.js hands `camera.near` to a session as its depthNear, and the
        page sizes it to the model: 0.23 m for a 150 m ground framed whole --
        past the blink, the arrival fade and the comfort vignette, one quad at
        0.1 m, and past a controller brought near the face. A session draws
        with a headset's near plane, also after a re-frame mid-session."""
        found = self._session()
        session = found["session"]

        self.assertEqual(found["console_errors"], [])
        self.assertGreater(session["framed"]["near"], 0.2, "the fixture is too small")
        self.assertLess(session["started"]["near"], 0.1)
        self.assertLess(session["reframed"]["near"], 0.1)

    def test_the_desktop_view_comes_back_after_a_session(self):
        """three.js writes each XR frame's head pose, lens and projection into
        the page's camera and puts none of it back, so a session left the
        desktop view where the headset last was, at the headset's field of
        view. It comes back as it was -- or, re-framed mid-session, as framed,
        on the desktop's lens -- with its near plane the desktop's again."""
        session = self._session()["session"]

        self._assertSameView(session["ended"]["view"], session["orbited"])
        self._assertSameView(session["reframed"]["view"], session["framed"])

    def test_a_session_start_and_end_are_wired(self):
        """The session's own events: it begins the headset afresh (the rig at
        the origin), listens for the headset's recenter on the session's space,
        shows the headset's layer -- and at the end hides it and resets."""
        session = self._session()["session"]
        started, ended = session["started"], session["ended"]
        origin = {"x": 0, "y": 0, "z": 0, "yaw": 0}

        self.assertEqual(started["place"], origin)
        self.assertEqual(started["heard"], ["reset"])
        self.assertIs(started["recenters"], True)
        self.assertIs(started["layer"], True)
        self.assertIs(ended["layer"], False)
        self.assertEqual(ended["place"], origin)

    # ------------------------------------------------------ dense surfaces
    def test_walking_and_aiming_over_a_dense_mesh_test_only_nearby_triangles(self):
        """Every headset frame casts at the model -- a walk once for the floor
        underfoot, an aimed arc once per segment -- and each cast tested every
        triangle of every mesh its segment's box touched: 10 ms a frame on a
        320k-triangle mesh, the whole 90 Hz budget. The surfaces are indexed
        now, so a frame tests the triangles near the segment. Counted in
        triangle tests (`Ray.intersectTriangle`), not milliseconds: the work,
        not the machine. And the answers are three.js's own raycast's."""
        found = self._load(
            self._terrain_glb(),
            then=lambda server, page: {"dense": page.evaluate(DENSE_JS)},
        )
        dense = found["dense"]
        walk, aim = dense["walk"], dense["aim"]

        self.assertEqual(found["console_errors"], [])
        self.assertLess(walk["tests"], 500, f"walking: {walk}")
        self.assertLess(aim["tests"], 3000, f"aiming: {aim}")
        self.assertGreater(walk["ground"], 0.1, "the walk stayed on the flat")
        self.assertAlmostEqual(walk["rig"]["y"], walk["ground"], places=3)
        self.assertIsNotNone(aim["target"], "the arc struck nothing")
        self.assertIs(aim["target"]["valid"], True)
        self.assertAlmostEqual(aim["target"]["point"][1], aim["ground"], places=4)

    # ------------------------------------------- the scripts, in a session
    _scripts_found = None

    def _scripts_in_session(self):
        """`turntable` and `inspect` on one page, each driven as a session
        drives it; and the page's XRFrame reader, on stand-in frames."""
        cls = type(self)
        if cls._scripts_found is None:

            def scenario(server, page):
                return {
                    "turntable": page.evaluate(TURNTABLE_JS),
                    "inspect": page.evaluate(INSPECT_XR_JS),
                    "read": page.evaluate(READ_JS),
                }

            cls._scripts_found = self._load(
                self._walkable_glb(), scripts=["turntable", "inspect"], then=scenario
            )
        return cls._scripts_found

    def test_the_turntable_holds_still_while_a_headset_presents(self):
        """At true scale a turning model is the world turning round the viewer,
        with the button that stops it out of reach behind the headset."""
        turntable = self._scripts_in_session()["turntable"]

        self.assertGreater(turntable["desktop"], 0.0, "it never turned")
        self.assertEqual(turntable["presenting"], 0.0)
        self.assertGreater(turntable["after"], 0.0, "it did not resume")

    def test_inspect_toggles_on_B_in_a_session_and_rides_a_card(self):
        """The page's chrome is not drawn in a session, so Inspect's toggle is
        a controller button and its panel a card beside the view: one toggle
        per press, however long it is held, and no card off the headset."""
        found = self._scripts_in_session()
        inspect = found["inspect"]

        self.assertEqual(found["console_errors"], [])
        self.assertEqual(inspect["before"], {"panel": False, "card": False})
        self.assertEqual(inspect["pressed"], {"panel": True, "card": True})
        self.assertEqual(inspect["held"], {"panel": True, "card": True})
        self.assertEqual(inspect["again"], {"panel": False, "card": False})
        self.assertEqual(inspect["desktop"], {"panel": False, "card": False})

    def test_the_headset_reader_takes_each_stick_from_its_own_axes(self):
        """xr-standard reports a thumbstick on axes 2-3 (0-1 is a touchpad, on
        controllers that have one); a controller with only a touchpad reports
        its stick on 0-1; an unhanded source is no hand, and fewer than two
        axes no stick. Poses come through in the session's own space."""
        read = self._scripts_in_session()["read"]
        standard = read["standard"]

        self.assertEqual(standard["left"], [0.3, -0.4])
        self.assertEqual(standard["right"], [-0.7, 0.8])
        self.assertEqual(standard["hands"], ["left", "right"])
        self.assertEqual(
            standard["head"], {"position": [0.1, 1.6, 0.2], "orientation": [0, 0, 0, 1]}
        )
        self.assertEqual(standard["aim"]["position"], [0.3, 1.2, -0.2])
        self.assertEqual(read["touchpad"]["left"], [0.25, -0.5])
        self.assertIsNone(read["touchpad"]["right"])
        self.assertIsNone(read["touchpad"]["aim"])
        self.assertEqual(read["unhanded"]["hands"], ["right"])
        self.assertIsNone(read["unhanded"]["left"])
        self.assertIsNone(read["unhanded"]["right"], "one axis is no stick")
        self.assertEqual(read["unhanded"]["aim"]["position"], [0.3, 1.2, -0.2])


if __name__ == "__main__":
    # A direct run loads no conftest: sandbox the temp root and the browser.
    from pythontk.core_utils.test_sandbox import TestSandbox

    TestSandbox.activate()
    unittest.main()
