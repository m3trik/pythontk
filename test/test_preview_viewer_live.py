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
  // Display luminance (0-255) at the centre of the fixture's one face, read
  // back off the page's own canvas after a render -- so what is measured is
  // the picture, tone mapping and all, rather than a material property.
  window.__sample = () => {
    const { renderer, scene, camera, THREE } = viewer;
    let mesh = null;
    viewer.model.traverse((n) => { if (!mesh && n.isMesh) mesh = n; });
    const position = mesh.geometry.attributes.position;
    const centroid = new THREE.Vector3();
    for (let i = 0; i < position.count; i += 1) {
      centroid.add(new THREE.Vector3().fromBufferAttribute(position, i));
    }
    centroid.divideScalar(position.count);
    mesh.updateWorldMatrix(true, false);
    const ndc = centroid.applyMatrix4(mesh.matrixWorld).project(camera);
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
        turns on the page's bake-relief shader patch. *bake_value* is the
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
            gltf["images"] = [{"name": "room_N", "uri": self.PIXEL_PNG}]
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


if __name__ == "__main__":
    # A direct run loads no conftest: sandbox the temp root and the browser.
    from pythontk.core_utils.test_sandbox import TestSandbox

    TestSandbox.activate()
    unittest.main()
