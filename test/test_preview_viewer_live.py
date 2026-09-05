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

Deliberately narrow. This covers the animation transport and the load path,
which the backlog names as the best first target because they need no material
fakes. The lighting policy wants a baked fixture and is left to follow.
"""

import json
import os
import struct
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
            envMapIntensity: m.envMapIntensity,
            hasEnvMap: !!m.envMap,
            hasNormalMap: !!m.normalMap,
            // The bake-relief shader patch, as an OWN property: the base
            // class carries a no-op through the prototype.
            reliefHook: Object.prototype.hasOwnProperty.call(m, 'onBeforeCompile'),
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
      // One entry per model swap, so a test can assert what the SECOND push
      // rendered. `disposeModel` frees the outgoing model's textures and has
      // to spare the shared environment map; missing that renders the second
      // push unlit, which no first-push check can see.
      report.loads = (report.loads || 0) + 1;
      report.ready = true;
    } catch (error) {
      report.errors.push(String(error));
      report.ready = true;
    }
  });
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

    #: The HDR divisor `apply_glb_lightmaps` records as `intensity`, chosen so
    #: the page's arithmetic is checkable rather than merely present.
    BAKE_VALUE = 4.0

    #: A 1x1 PNG, for a fixture that needs a texture the loader will decode.
    PIXEL_PNG = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
        "2mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    def _lightmapped_glb(self, normal_map=False):
        """A GLB whose material wears a BAKED map, built by the real applier.

        Hand-writing `extras.lightmap_web` would test the page against a
        fixture rather than against the pipeline, and the whole point of these
        assertions is that the two agree. So the manifest rides the
        `data_export` carrier exactly as an FBX delivers it and
        `apply_glb_lightmaps` produces the block the viewer reads.

        *normal_map* also gives the material a normal map, which is what
        turns on the page's bake-relief shader patch.
        """
        import cv2
        import numpy as np

        exr_dir = self.temp.dir_path()
        exr = os.path.join(exr_dir, "room_Lightmap.exr")
        cv2.imwrite(exr, np.full((8, 8, 3), self.BAKE_VALUE, dtype=np.float32))

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
        if normal_map:
            gltf["images"] = [{"name": "room_N", "uri": self.PIXEL_PNG}]
            gltf["textures"] = [{"source": 0}]
            gltf["materials"][0]["normalTexture"] = {"index": 0}
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
        out = self.temp.path(extension=".glb")
        with open(out, "wb") as fh:
            fh.write(struct.pack("<4sII", b"glTF", 2, total))
            fh.write(struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes)
            fh.write(struct.pack("<I4s", len(blob), b"BIN\0") + blob)
        return out

    # ------------------------------------------------------------------ driver
    def _load(self, glb, then_publish=None, probe=None):
        """Serve *glb*, open it in the real page, return the probe's findings.

        *then_publish* publishes a SECOND version once the page is up and waits
        for the swap -- the only way to reach `disposeModel`, which frees the
        outgoing model's textures between pushes.

        *probe* overrides the class-wide script for tests that need to measure
        something else (the fade suite drives the playhead); it reports through
        the same ``window.__probe`` handle, so the wait below is unchanged.
        """
        from playwright.sync_api import sync_playwright

        server = ptk.PreviewServer(viewer=True, title="live-test")
        server.start()
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
                page.goto(server.url, wait_until="domcontentloaded", timeout=120_000)
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
                found = page.evaluate("() => window.__probe")
                browser.close()
        finally:
            server.stop()
        found["console_errors"] = console
        return found

    # ------------------------------------------------------------------- tests
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
        must spare the shared environment map, which every baked material was
        given as its own `envMap`. Miss that and the second push renders unlit
        -- invisible to every first-push check, and the failure an artist hits
        on their second click rather than their first."""
        found = self._load(
            self._lightmapped_glb(), then_publish=self._lightmapped_glb()
        )

        self.assertGreaterEqual(found["loads"], 2, "the page never swapped")
        material = self._lit(found)
        self.assertTrue(material["hasLightMap"], "the second push lost its bake")
        self.assertTrue(
            material["hasEnvMap"],
            "the shared environment was disposed with the outgoing model",
        )
        self.assertEqual(found["console_errors"], [])

    def test_a_baked_material_keeps_SOME_environment_light(self):
        """The per-material opt-out: a bake carries diffuse light and no
        specular, so dropping the environment entirely leaves every metal and
        every gloss dead flat. It is turned DOWN, never off."""
        material = self._lit(self._load(self._lightmapped_glb()))

        self.assertTrue(material["hasEnvMap"], "the environment was removed outright")
        self.assertGreater(material["envMapIntensity"], 0)
        self.assertLess(
            material["envMapIntensity"],
            1.0,
            "a baked material must not take the full environment on top",
        )

    def test_a_baked_normal_mapped_material_relieves_the_bake_and_compiles(self):
        """A lightmap is direction-free irradiance that never consults the
        normal, so on a baked surface the normal map reached the picture only
        through the dimmed environment -- measured on a production room as
        ~1% of pixels moving by ~0.4/255 between normalScale 1 and 4. The page
        patches the bake by the map through `onBeforeCompile`.

        Asserted on the hook AND on a clean console, because the two ways this
        patch has already gone wrong are both silent to everything else: a
        replace aimed inside an `#include` the hook never sees is a no-op, and
        a GLSL reserved word (`flat`) fails the compile with the material
        falling back to nothing -- both logged, neither thrown.
        """
        found = self._load(self._lightmapped_glb(normal_map=True))

        material = self._lit(found)
        self.assertTrue(material["hasNormalMap"], "fixture lost its normal map")
        self.assertTrue(material["reliefHook"], "the relief patch is not installed")
        self.assertEqual(found["console_errors"], [])
        # And NOT on a baked material without a normal map: there is nothing
        # to relieve by, and the pure bake must stay the pure bake.
        plain = self._lit(self._load(self._lightmapped_glb()))
        self.assertFalse(plain["reliefHook"])

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
                # this was written against ships two `vdat533`), so a reader
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
    unittest.main()
