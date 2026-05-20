# !/usr/bin/python
# coding=utf-8
"""Smoke tests for the pure :class:`MapCompositor` engine.

The engine emits status via ``self.logger`` (LoggingMixin). Tests attach
an in-memory handler to capture records without going through Qt.
"""

import logging
import os
import shutil
import tempfile
import unittest
from typing import List

from PIL import Image

from pythontk.img_utils.map_compositor import BatchResult, MapCompositor, NormalOutputMode


class _CapturingHandler(logging.Handler):
    """Capture every record emitted on the engine's logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self) -> List[str]:
        return [r.getMessage() for r in self.records]

    def levels(self) -> List[str]:
        return [r.levelname for r in self.records]


def _load(path: str) -> Image.Image:
    """Materialize a PIL image into memory and release the file handle."""
    with Image.open(path) as im:
        return im.copy()


def _solid_rgba(size, color):
    return Image.new("RGBA", size, color)


class _LoggerCaptureMixin:
    """Attach a capturing handler to ``engine.logger`` for the test lifetime."""

    def attach_capture(self, engine: MapCompositor) -> _CapturingHandler:
        handler = _CapturingHandler()
        engine.logger.addHandler(handler)
        prior_level = engine.logger.level
        engine.logger.setLevel(logging.DEBUG)
        self.addCleanup(engine.logger.removeHandler, handler)
        self.addCleanup(engine.logger.setLevel, prior_level)
        return handler


class TestEnginePurity(unittest.TestCase):
    """The engine module must not import Qt."""

    def test_no_qt_imports(self):
        import pythontk.img_utils.map_compositor as eng

        forbidden = {"qtpy", "PySide2", "PySide6", "PyQt5", "PyQt6"}
        with open(eng.__file__, "r", encoding="utf-8") as f:
            src = f.read()
        for name in forbidden:
            self.assertNotIn(name, src, f"compositor.py should not reference {name}")


class TestComposite(unittest.TestCase, _LoggerCaptureMixin):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_layer(self, name: str, img: Image.Image) -> str:
        path = os.path.join(self.tmp, name)
        img.save(path)
        return path

    def test_single_uniform_bg_writes_output_and_emits_messages(self):
        gray = _solid_rgba((4, 4), (127, 127, 127, 255))
        path = self._write_layer("source_Base_Color.png", gray)

        engine = MapCompositor()
        cap = self.attach_capture(engine)
        engine.total_len = 1

        failed = engine.composite_images(
            {"Base_Color": [(path, _load(path))]}, self.tmp, name="test"
        )

        self.assertEqual(failed, {})
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "test_Base_Color.png")))

        msgs = cap.messages()
        # Section header for Base_Color, plus the file name tick.
        self.assertTrue(any("Base_Color" in m for m in msgs))
        self.assertTrue(any("source_Base_Color.png" in m for m in msgs))
        self.assertNotIn("ERROR", cap.levels())

    def test_non_uniform_bg_defers_to_failed(self):
        a = _solid_rgba((4, 4), (127, 127, 127, 255))
        b = _solid_rgba((4, 4), (0, 0, 0, 255))
        pa = self._write_layer("a_Base_Color.png", a)
        pb = self._write_layer("b_Base_Color.png", b)

        engine = MapCompositor()
        engine.total_len = 2

        failed = engine.composite_images(
            {"Base_Color": [(pa, _load(pa)), (pb, _load(pb))]},
            self.tmp,
            name="test",
        )
        self.assertEqual(set(failed.keys()), {"Base_Color"})

    def test_normal_directx_auto_converts_to_opengl(self):
        flat = _solid_rgba((4, 4), (127, 127, 255, 255))
        path = self._write_layer("source_Normal_DirectX.png", flat)

        engine = MapCompositor()
        cap = self.attach_capture(engine)
        engine.total_len = 1

        engine.composite_images(
            {"Normal_DirectX": [(path, _load(path))]}, self.tmp, name="test"
        )

        self.assertTrue(
            os.path.exists(os.path.join(self.tmp, "test_Normal_DirectX.png"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp, "test_Normal_OpenGL.png"))
        )
        # The "Created using ..." line should appear after the OpenGL section.
        self.assertTrue(any("Created using" in m for m in cap.messages()))

    def test_progress_callback_invoked_per_layer(self):
        gray = _solid_rgba((4, 4), (127, 127, 127, 255))
        p1 = self._write_layer("a_Base_Color.png", gray)
        p2 = self._write_layer("b_Base_Color.png", gray)

        progress_pcts: List[float] = []
        engine = MapCompositor(progress_callback=progress_pcts.append)
        engine.total_len = 2

        engine.composite_images(
            {"Base_Color": [(p1, _load(p1)), (p2, _load(p2))]},
            self.tmp,
            name="test",
        )

        self.assertEqual(len(progress_pcts), 2)
        self.assertEqual(progress_pcts[-1], 100.0)

    def test_default_progress_callback_is_noop(self):
        gray = _solid_rgba((4, 4), (127, 127, 127, 255))
        path = self._write_layer("source_Base_Color.png", gray)

        engine = MapCompositor()  # no progress_callback
        engine.total_len = 1
        failed = engine.composite_images(
            {"Base_Color": [(path, _load(path))]}, self.tmp, name="test"
        )
        self.assertEqual(failed, {})


class TestProcessBatch(unittest.TestCase, _LoggerCaptureMixin):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_batch_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name: str, color):
        path = os.path.join(self.tmp, name)
        Image.new("RGBA", (4, 4), color).save(path)
        return path

    def test_clean_batch_reports_success(self):
        p = self._write("a_Base_Color.png", (127, 127, 127, 255))
        engine = MapCompositor()
        result = engine.process_batch(
            {"Base_Color": [(p, _load(p))]}, self.tmp, name="test"
        )
        self.assertIs(result, BatchResult.SUCCESS)

    def test_process_batch_resets_state_between_runs(self):
        engine = MapCompositor()
        engine.masks = [Image.new("L", (4, 4), 128)]
        engine.total_progress = 999

        p = self._write("a_Base_Color.png", (127, 127, 127, 255))
        engine.process_batch({"Base_Color": [(p, _load(p))]}, self.tmp, name="test")

        self.assertEqual(engine.masks, [])
        self.assertEqual(engine.total_progress, 1)

    def test_process_batch_sets_total_len(self):
        p1 = self._write("a_Base_Color.png", (127, 127, 127, 255))
        p2 = self._write("b_Base_Color.png", (127, 127, 127, 255))
        engine = MapCompositor()
        engine.process_batch(
            {"Base_Color": [(p1, _load(p1)), (p2, _load(p2))]},
            self.tmp,
            name="test",
        )
        self.assertEqual(engine.total_len, 2)


class TestOutputTemplate(unittest.TestCase, _LoggerCaptureMixin):
    """Engine post-processes composited output with a pythontk workflow preset."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_template_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _stage(self):
        """Drop in a complete set of single-layer maps and return them grouped."""
        sorted_images = {}
        layers = [
            ("Base_Color", "RGB", (200, 100, 50)),
            ("Metallic", "L", 200),
            ("Roughness", "L", 100),
            ("Ambient_Occlusion", "L", 150),
            ("Normal_OpenGL", "RGB", (128, 128, 255)),
        ]
        for typ, mode, color in layers:
            path = os.path.join(self.tmp, f"layer_{typ}.png")
            Image.new(mode, (4, 4), color).save(path)
            sorted_images[typ] = [(path, _load(path))]
        return sorted_images

    def test_default_template_is_noop(self):
        engine = MapCompositor()
        sorted_images = self._stage()
        result = engine.process_batch(sorted_images, self.tmp, name="mat")
        self.assertIs(result, BatchResult.SUCCESS)
        files = set(os.listdir(self.tmp))
        # No template selected → no packed MSAO/ORM output added.
        self.assertFalse(any(n.endswith("_MSAO.png") for n in files))
        self.assertFalse(any(n.endswith("_ORM.png") for n in files))

    def test_unity_hdrp_template_emits_msao(self):
        engine = MapCompositor()
        engine.output_template = "Unity HDRP"
        sorted_images = self._stage()
        result = engine.process_batch(sorted_images, self.tmp, name="mat")
        self.assertIs(result, BatchResult.SUCCESS)
        files = set(os.listdir(self.tmp))
        self.assertIn("mat_MSAO.png", files)
        # Composited siblings stay on disk.
        self.assertIn("mat_Base_Color.png", files)

    def test_unknown_template_warns_and_skips(self):
        engine = MapCompositor()
        engine.output_template = "Not A Real Workflow"
        cap = self.attach_capture(engine)
        sorted_images = self._stage()
        result = engine.process_batch(sorted_images, self.tmp, name="mat")
        self.assertIs(result, BatchResult.SUCCESS)
        self.assertTrue(
            any("Unknown output template" in m for m in cap.messages()),
            "Expected a warning for an unknown template",
        )

    def test_apply_output_template_skips_when_unset(self):
        engine = MapCompositor()
        # Drop a single file in the dir; with no template, the method returns [].
        Image.new("L", (4, 4), 200).save(os.path.join(self.tmp, "mat_Metallic.png"))
        result = engine.apply_output_template(self.tmp)
        self.assertEqual(result, [])

    def test_apply_output_template_invalid_dir_warns(self):
        engine = MapCompositor()
        engine.output_template = "Unity HDRP"
        cap = self.attach_capture(engine)
        result = engine.apply_output_template(os.path.join(self.tmp, "does_not_exist"))
        self.assertEqual(result, [])
        self.assertTrue(
            any("is not a directory" in m for m in cap.messages()),
            "Expected a warning for an invalid output dir",
        )


class TestRetryFailed(unittest.TestCase, _LoggerCaptureMixin):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_retry_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_retry_fills_layers_when_mask_available(self):
        bg_a = _solid_rgba((4, 4), (127, 127, 127, 255))
        bg_b = _solid_rgba((4, 4), (0, 0, 0, 255))
        pa = os.path.join(self.tmp, "a_Base_Color.png")
        bg_a.save(pa)
        pb = os.path.join(self.tmp, "b_Base_Color.png")
        bg_b.save(pb)

        engine = MapCompositor()
        engine.total_len = 2
        engine.masks = [
            Image.new("L", (4, 4), 255),
            Image.new("L", (4, 4), 255),
        ]

        failed = {"Base_Color": [(pa, _load(pa)), (pb, _load(pb))]}
        retried = engine.retry_failed(failed, name="test")

        self.assertIn("Base_Color", retried)
        self.assertEqual(len(retried["Base_Color"]), 2)

    def test_retry_emits_error_when_mask_missing(self):
        bg = _solid_rgba((4, 4), (127, 127, 127, 255))
        p = os.path.join(self.tmp, "x_Base_Color.png")
        bg.save(p)

        engine = MapCompositor()
        cap = self.attach_capture(engine)
        engine.masks = []

        engine.retry_failed({"Base_Color": [(p, _load(p))]}, name="test")
        self.assertIn("ERROR", cap.levels())


class TestSeedMasks(unittest.TestCase, _LoggerCaptureMixin):
    """`_seed_masks` should combine alpha across every eligible map type
    so an antialiased / eroded boundary in one source is recovered from
    another."""

    def _rgba_with_alpha_pixels(self, size, alpha_pixels):
        """size=(w,h); alpha_pixels: iterable of (x,y) to set to alpha=255."""
        im = Image.new("RGBA", size, (200, 200, 200, 0))
        for x, y in alpha_pixels:
            im.putpixel((x, y), (200, 200, 200, 255))
        return im

    def test_or_combines_alpha_across_sources(self):
        # Source A covers (3,3) and (4,4); source B covers (3,4) and (4,3).
        # Together they describe a 2x2 content block; alone each is incomplete.
        size = (8, 8)
        a = self._rgba_with_alpha_pixels(size, [(3, 3), (4, 4)])
        b = self._rgba_with_alpha_pixels(size, [(3, 4), (4, 3)])
        sorted_images = {
            "Base_Color": [("a.png", a)],
            "Roughness": [("b.png", b)],
        }

        engine = MapCompositor()
        masks = engine._seed_masks(
            sorted_images,
            fallback_typ="Base_Color",
            fallback_layers=sorted_images["Base_Color"],
            fallback_bg=(200, 200, 200, 0),
        )
        self.assertEqual(len(masks), 1)
        mask = masks[0]
        self.assertEqual(mask.mode, "L")
        for px in [(3, 3), (3, 4), (4, 3), (4, 4)]:
            self.assertEqual(mask.getpixel(px), 255, f"content lost at {px}")
        # A corner pixel that was alpha=0 in BOTH sources must remain bg.
        self.assertEqual(mask.getpixel((0, 0)), 0)

    def test_falls_back_to_create_mask_when_no_alpha_source(self):
        # RGB-only sources (no alpha band) — must hit the legacy path.
        a = Image.new("RGB", (4, 4), (10, 20, 30))
        a.putpixel((1, 1), (200, 200, 200))
        sorted_images = {"Base_Color": [("a.png", a)]}

        engine = MapCompositor()
        cap = self.attach_capture(engine)
        masks = engine._seed_masks(
            sorted_images,
            fallback_typ="Base_Color",
            fallback_layers=sorted_images["Base_Color"],
            fallback_bg=(10, 20, 30, 255),
        )
        self.assertEqual(len(masks), 1)
        self.assertEqual(masks[0].mode, "L")
        self.assertTrue(
            any("Attempting to create masks" in m for m in cap.messages()),
            "expected fallback log line",
        )

    def test_skips_sources_with_mismatched_layer_count(self):
        # 2 layers in the triggering type, 1 in a mismatched alpha source —
        # the mismatched source must be ignored so positional alignment holds.
        # Content kept off the corners so get_background() still detects a
        # uniform alpha=0 bg.
        size = (6, 6)
        triggering = [
            ("a0.png", self._rgba_with_alpha_pixels(size, [(1, 1)])),
            ("a1.png", self._rgba_with_alpha_pixels(size, [(2, 2)])),
        ]
        mismatched = [
            ("b0.png", self._rgba_with_alpha_pixels(size, [(3, 3), (4, 4)])),
        ]
        sorted_images = {"Base_Color": triggering, "Roughness": mismatched}

        engine = MapCompositor()
        masks = engine._seed_masks(
            sorted_images,
            fallback_typ="Base_Color",
            fallback_layers=triggering,
            fallback_bg=(200, 200, 200, 0),
        )
        self.assertEqual(len(masks), 2)
        # Layer-0 mask carries only the (1,1) content — the mismatched
        # source's (3,3)/(4,4) pixels must not bleed in.
        self.assertEqual(masks[0].getpixel((1, 1)), 255)
        self.assertEqual(masks[0].getpixel((3, 3)), 0)
        self.assertEqual(masks[0].getpixel((4, 4)), 0)

    def test_skips_alpha_source_with_opaque_background(self):
        # Source has alpha band but bg alpha is 255 → not the transparent-bg
        # path the seeder is meant for; must be skipped.
        size = (4, 4)
        opaque_bg = Image.new("RGBA", size, (50, 50, 50, 255))
        opaque_bg.putpixel((1, 1), (200, 200, 200, 255))

        transparent_bg = self._rgba_with_alpha_pixels(size, [(2, 2)])

        sorted_images = {
            "Roughness": [("opaque.png", opaque_bg)],
            "Base_Color": [("trans.png", transparent_bg)],
        }

        engine = MapCompositor()
        masks = engine._seed_masks(
            sorted_images,
            fallback_typ="Base_Color",
            fallback_layers=sorted_images["Base_Color"],
            fallback_bg=(200, 200, 200, 0),
        )
        # Only the transparent-bg source counts → content at (2,2) only.
        self.assertEqual(masks[0].getpixel((2, 2)), 255)
        self.assertEqual(masks[0].getpixel((1, 1)), 0)


class TestMapInfoBundle(unittest.TestCase):
    def test_mapinfo_is_frozen(self):
        from pythontk.img_utils.map_compositor import _MapInfo

        info = _MapInfo(
            mode="RGB", bit_depth="24bit (8x3)", ext="png", width=4, height=4
        )
        with self.assertRaises(Exception):
            info.mode = "RGBA"


class TestAlphaCompositeErrorDiagnostic(unittest.TestCase, _LoggerCaptureMixin):
    """When alpha_composite raises on a layer, the error must name the file."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_diag_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_value_error_includes_filename(self):
        gray = (127, 127, 127, 255)
        Image.new("RGBA", (4, 4), gray).save(
            pa := os.path.join(self.tmp, "a_Base_Color.png")
        )
        Image.new("RGBA", (8, 8), gray).save(
            pb := os.path.join(self.tmp, "b_Base_Color.png")
        )

        engine = MapCompositor()
        cap = self.attach_capture(engine)
        engine.total_len = 2

        engine.composite_images(
            {"Base_Color": [(pa, _load(pa)), (pb, _load(pb))]},
            self.tmp,
            name="test",
        )

        error_msgs = [r.getMessage() for r in cap.records if r.levelname == "ERROR"]
        self.assertEqual(len(error_msgs), 1)
        self.assertIn("b_Base_Color.png", error_msgs[0])


class TestSetBitDepthIntegration(unittest.TestCase):
    """The save path coerces mode/bit-depth via ptk.ImgUtils.set_bit_depth."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_sbd_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roughness_saved_as_grayscale(self):
        rgb = Image.new("RGBA", (4, 4), (200, 200, 200, 255))
        p = os.path.join(self.tmp, "src_Roughness.png")
        rgb.save(p)

        engine = MapCompositor()
        engine.total_len = 1
        engine.composite_images({"Roughness": [(p, _load(p))]}, self.tmp, name="test")

        out = os.path.join(self.tmp, "test_Roughness.png")
        with Image.open(out) as saved:
            self.assertEqual(saved.mode, "L")


class TestEdgeHaloPreservation(unittest.TestCase, _LoggerCaptureMixin):
    """Partial-alpha edge pixels with RGB=0 (a common export artifact from
    Substance / Painter where transparent regions aren't propagated into RGB)
    must not produce a dark rim halo when composited against the map type's
    default background.

    Two failure modes are covered:
      * Single layer: ``paste(composited, mask=composited)`` blends src RGB
        with the white roughness bg using alpha as the weight. With src RGB=0
        at a partial-alpha edge, the blend collapses toward 0/255 mid-gray.
      * Multi layer: ``alpha_composite`` blends a subsequent layer's
        partial-alpha edge (RGB=0) into the previously-composited base,
        darkening the underlying content at those positions.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_halo_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_partial_alpha_roughness(self, size=(16, 16), content=180):
        """Roughness source: transparent corners (so the compositor takes the
        transparent-bg path), a center block of opaque content, and an
        intermediate ring of (0,0,0,128) — the dark-halo trigger."""
        im = Image.new("RGBA", size, (0, 0, 0, 0))
        w, h = size
        cx0, cx1 = w // 4, 3 * w // 4
        cy0, cy1 = h // 4, 3 * h // 4
        # Opaque content block in the middle.
        for x in range(cx0, cx1):
            for y in range(cy0, cy1):
                im.putpixel((x, y), (content, content, content, 255))
        # Partial-alpha edge ring with RGB=0 — only at pixels still at the
        # transparent background. Don't disturb the opaque content.
        ring = []
        for x in range(cx0 - 1, cx1 + 1):
            ring.append((x, cy0 - 1))
            ring.append((x, cy1))
        for y in range(cy0 - 1, cy1 + 1):
            ring.append((cx0 - 1, y))
            ring.append((cx1, y))
        for x, y in ring:
            if 0 <= x < w and 0 <= y < h and im.getpixel((x, y)) == (0, 0, 0, 0):
                im.putpixel((x, y), (0, 0, 0, 128))
        return im

    def test_single_layer_partial_alpha_edges_have_no_dark_halo(self):
        # Single-layer Roughness; the registry default bg is (255,255,255).
        # Interior content must survive; edge ring must not darken below content.
        src = self._make_partial_alpha_roughness(size=(16, 16), content=180)
        path = os.path.join(self.tmp, "src_Roughness.png")
        src.save(path)

        engine = MapCompositor()
        engine.total_len = 1
        engine.composite_images(
            {"Roughness": [(path, _load(path))]}, self.tmp, name="test"
        )

        out = os.path.join(self.tmp, "test_Roughness.png")
        with Image.open(out) as saved:
            saved = saved.copy()
        self.assertEqual(saved.mode, "L")

        # Interior content (alpha=255 in source) must round-trip exactly.
        for px in [(5, 5), (6, 6), (8, 8), (10, 10)]:
            self.assertEqual(
                saved.getpixel(px),
                180,
                f"interior content was altered at {px}: got {saved.getpixel(px)}",
            )

        # Edge ring pixels (alpha=128 RGB=0 in source) must NOT be a dark
        # halo value. Acceptable outcomes: bg (255) or content (180);
        # anything strictly less than the content is the halo bug.
        for px in [(3, 7), (12, 7), (7, 3), (7, 12)]:
            val = saved.getpixel(px)
            self.assertGreaterEqual(
                val,
                180,
                f"edge halo at {px}: got {val} — expected >= content (180)",
            )

    def test_multi_layer_subsequent_partial_alpha_does_not_darken_base(self):
        # Two roughness layers: first is solid opaque content; second has
        # partial-alpha edges with RGB=0 overlapping the first's content.
        # The underlying content must not be darkened by the blend.
        size = (16, 16)

        # Layer A: fully opaque content (no alpha at the corners — pick a
        # uniform opaque bg so bg detection picks an opaque color).
        first = Image.new("RGBA", size, (255, 255, 255, 255))  # white bg
        for x in range(4, 12):
            for y in range(4, 12):
                first.putpixel((x, y), (180, 180, 180, 255))
        path_a = os.path.join(self.tmp, "a_Roughness.png")
        first.save(path_a)

        # Layer B: same uniform white bg so bg detection agrees, but with
        # partial-alpha edges at RGB=0 overlapping layer A's content.
        second = Image.new("RGBA", size, (255, 255, 255, 255))
        for x in range(5, 11):
            for y in range(5, 11):
                second.putpixel((x, y), (0, 0, 0, 128))  # partial alpha black
        path_b = os.path.join(self.tmp, "b_Roughness.png")
        second.save(path_b)

        engine = MapCompositor()
        engine.total_len = 2
        engine.composite_images(
            {"Roughness": [(path_a, _load(path_a)), (path_b, _load(path_b))]},
            self.tmp,
            name="test",
        )

        out = os.path.join(self.tmp, "test_Roughness.png")
        with Image.open(out) as saved:
            saved = saved.copy()
        # Pixel under the partial-alpha edge of layer B: the underlying
        # layer A content (180) blended with bg (255) is the acceptable
        # outcome; anything below 180 means RGB=0 from layer B leaked in.
        for px in [(5, 5), (6, 6), (10, 10), (7, 8)]:
            val = saved.getpixel(px)
            self.assertGreaterEqual(
                val,
                180,
                f"layer-A content darkened at {px}: got {val} (< 180)",
            )


class TestFilterRedundantMapsIntegration(unittest.TestCase):
    def test_orm_drops_metallic_roughness_ao(self):
        import pythontk as ptk

        sorted_maps = {
            "ORM": ["fake_ORM.png"],
            "Metallic": ["fake_Metallic.png"],
            "Roughness": ["fake_Roughness.png"],
            "Ambient_Occlusion": ["fake_AO.png"],
            "Base_Color": ["fake_BC.png"],
        }
        ptk.MapFactory.filter_redundant_maps(sorted_maps)

        self.assertIn("ORM", sorted_maps)
        self.assertIn("Base_Color", sorted_maps)
        self.assertNotIn("Metallic", sorted_maps)
        self.assertNotIn("Roughness", sorted_maps)
        self.assertNotIn("Ambient_Occlusion", sorted_maps)


class TestNormalFormatMismatchWarning(unittest.TestCase, _LoggerCaptureMixin):
    """When detect_normal_map_format disagrees with declared format, warn."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_normfmt_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, stub_returns):
        import pythontk as ptk

        original = ptk.MapFactory.detect_normal_map_format
        try:
            ptk.MapFactory.detect_normal_map_format = staticmethod(
                lambda image, threshold=0.1: stub_returns
            )

            flat = Image.new("RGBA", (4, 4), (127, 127, 255, 255))
            p = os.path.join(self.tmp, "src_Normal_DirectX.png")
            flat.save(p)

            engine = MapCompositor()
            cap = self.attach_capture(engine)
            engine.total_len = 1

            engine.composite_images(
                {"Normal_DirectX": [(p, _load(p))]}, self.tmp, name="test"
            )
            return cap
        finally:
            ptk.MapFactory.detect_normal_map_format = original

    def test_mismatch_emits_warning(self):
        cap = self._run("OpenGL")
        warnings = [r for r in cap.records if r.levelname == "WARNING"]
        self.assertTrue(any("declared" in r.getMessage() for r in warnings))

    def test_match_emits_no_warning(self):
        cap = self._run("DirectX")
        warnings = [r for r in cap.records if r.levelname == "WARNING"]
        self.assertEqual([w for w in warnings if "declared" in w.getMessage()], [])


class TestNormalOutputMode(unittest.TestCase, _LoggerCaptureMixin):
    """The normal_output_mode setting controls which DX/GL variant(s) survive."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_norm_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_dx(self) -> str:
        flat = Image.new("RGBA", (4, 4), (127, 127, 255, 255))
        path = os.path.join(self.tmp, "src_Normal_DirectX.png")
        flat.save(path)
        return path

    def test_both_default_writes_dx_and_gl(self):
        path = self._write_dx()
        engine = MapCompositor()
        engine.total_len = 1
        # Default is BOTH.
        self.assertIs(engine.normal_output_mode, NormalOutputMode.BOTH)
        engine.composite_images(
            {"Normal_DirectX": [(path, _load(path))]}, self.tmp, name="t"
        )
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "t_Normal_DirectX.png")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "t_Normal_OpenGL.png")))

    def test_none_skips_auto_conversion(self):
        path = self._write_dx()
        engine = MapCompositor()
        engine.normal_output_mode = NormalOutputMode.NONE
        engine.total_len = 1
        engine.composite_images(
            {"Normal_DirectX": [(path, _load(path))]}, self.tmp, name="t"
        )
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "t_Normal_DirectX.png")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "t_Normal_OpenGL.png")))

    def test_opengl_only_replaces_dx_input_with_gl_output(self):
        path = self._write_dx()
        engine = MapCompositor()
        engine.normal_output_mode = NormalOutputMode.OPENGL_ONLY
        engine.total_len = 1
        engine.composite_images(
            {"Normal_DirectX": [(path, _load(path))]}, self.tmp, name="t"
        )
        # DirectX file is removed; only OpenGL survives.
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "t_Normal_DirectX.png")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "t_Normal_OpenGL.png")))

    def test_directx_only_keeps_dx_input_when_already_matching(self):
        path = self._write_dx()
        engine = MapCompositor()
        engine.normal_output_mode = NormalOutputMode.DIRECTX_ONLY
        engine.total_len = 1
        engine.composite_images(
            {"Normal_DirectX": [(path, _load(path))]}, self.tmp, name="t"
        )
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "t_Normal_DirectX.png")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "t_Normal_OpenGL.png")))


class TestOptimizeOutput(unittest.TestCase, _LoggerCaptureMixin):
    """When optimize_output is on, the save path runs ImgUtils.optimize_texture."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_opt_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_optimize_called_when_enabled(self):
        import pythontk as ptk

        calls = []
        original = ptk.ImgUtils.optimize_texture
        try:
            ptk.ImgUtils.optimize_texture = classmethod(
                lambda cls, path, **kw: calls.append((path, kw)) or path
            )

            gray = Image.new("RGBA", (4, 4), (127, 127, 127, 255))
            p = os.path.join(self.tmp, "src_Base_Color.png")
            gray.save(p)

            engine = MapCompositor()
            engine.optimize_output = True
            engine.total_len = 1
            engine.composite_images({"Base_Color": [(p, _load(p))]}, self.tmp, name="t")

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1].get("map_type"), "Base_Color")
        finally:
            ptk.ImgUtils.optimize_texture = original

    def test_optimize_not_called_when_disabled(self):
        import pythontk as ptk

        calls = []
        original = ptk.ImgUtils.optimize_texture
        try:
            ptk.ImgUtils.optimize_texture = classmethod(
                lambda cls, path, **kw: calls.append(path) or path
            )

            gray = Image.new("RGBA", (4, 4), (127, 127, 127, 255))
            p = os.path.join(self.tmp, "src_Base_Color.png")
            gray.save(p)

            engine = MapCompositor()  # optimize_output defaults False
            engine.total_len = 1
            engine.composite_images({"Base_Color": [(p, _load(p))]}, self.tmp, name="t")
            self.assertEqual(calls, [])
        finally:
            ptk.ImgUtils.optimize_texture = original


class TestNormalModeConflictPrefilter(unittest.TestCase):
    """Slot must pre-filter so OPENGL_ONLY / DIRECTX_ONLY with both sources
    doesn't end up order-dependent. Tests the slot-level filter logic
    against the same dict shape the engine receives.
    """

    @staticmethod
    def _prefilter(sorted_images, mode):
        """Inline copy of the slot's pre-filter logic so we can exercise
        the rules without instantiating a Switchboard UI."""
        if (
            mode is NormalOutputMode.OPENGL_ONLY
            and "Normal_OpenGL" in sorted_images
            and "Normal_DirectX" in sorted_images
        ):
            del sorted_images["Normal_DirectX"]
        elif (
            mode is NormalOutputMode.DIRECTX_ONLY
            and "Normal_OpenGL" in sorted_images
            and "Normal_DirectX" in sorted_images
        ):
            del sorted_images["Normal_OpenGL"]

    def test_opengl_only_drops_directx_when_both_present(self):
        d = {"Normal_OpenGL": ["gl.png"], "Normal_DirectX": ["dx.png"]}
        self._prefilter(d, NormalOutputMode.OPENGL_ONLY)
        self.assertIn("Normal_OpenGL", d)
        self.assertNotIn("Normal_DirectX", d)

    def test_directx_only_drops_opengl_when_both_present(self):
        d = {"Normal_OpenGL": ["gl.png"], "Normal_DirectX": ["dx.png"]}
        self._prefilter(d, NormalOutputMode.DIRECTX_ONLY)
        self.assertIn("Normal_DirectX", d)
        self.assertNotIn("Normal_OpenGL", d)

    def test_both_mode_does_not_drop(self):
        d = {"Normal_OpenGL": ["gl.png"], "Normal_DirectX": ["dx.png"]}
        self._prefilter(d, NormalOutputMode.BOTH)
        self.assertEqual(set(d.keys()), {"Normal_OpenGL", "Normal_DirectX"})

    def test_none_mode_does_not_drop(self):
        d = {"Normal_OpenGL": ["gl.png"], "Normal_DirectX": ["dx.png"]}
        self._prefilter(d, NormalOutputMode.NONE)
        self.assertEqual(set(d.keys()), {"Normal_OpenGL", "Normal_DirectX"})

    def test_no_conflict_when_only_one_format_present(self):
        d = {"Normal_DirectX": ["dx.png"]}
        self._prefilter(d, NormalOutputMode.OPENGL_ONLY)
        self.assertEqual(set(d.keys()), {"Normal_DirectX"})


class TestEdgeCases(unittest.TestCase, _LoggerCaptureMixin):
    """Production-relevant edge cases."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_edge_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_sorted_images_is_clean_success(self):
        engine = MapCompositor()
        result = engine.process_batch({}, self.tmp, name="t")
        self.assertIs(result, BatchResult.SUCCESS)
        self.assertEqual(engine.total_len, 0)
        self.assertEqual(engine.total_progress, 0)

    def test_combined_batch_of_multiple_map_types(self):
        """One process_batch with BaseColor + Normal_DirectX + Roughness.

        Locks in: each type composites independently, normal auto-converts
        to OpenGL (BOTH mode default), Roughness comes out as 8-bit L.
        """
        gray = Image.new("RGBA", (4, 4), (127, 127, 127, 255))
        flat = Image.new("RGBA", (4, 4), (127, 127, 255, 255))

        bc = os.path.join(self.tmp, "src_Base_Color.png")
        gray.save(bc)
        nrm = os.path.join(self.tmp, "src_Normal_DirectX.png")
        flat.save(nrm)
        rough = os.path.join(self.tmp, "src_Roughness.png")
        gray.save(rough)

        engine = MapCompositor()
        result = engine.process_batch(
            {
                "Base_Color": [(bc, _load(bc))],
                "Normal_DirectX": [(nrm, _load(nrm))],
                "Roughness": [(rough, _load(rough))],
            },
            self.tmp,
            name="t",
        )

        self.assertIs(result, BatchResult.SUCCESS)
        for filename, expected_mode in (
            ("t_Base_Color.png", "RGB"),
            ("t_Normal_DirectX.png", "RGB"),
            ("t_Normal_OpenGL.png", "RGB"),  # auto-generated
            ("t_Roughness.png", "L"),  # set_bit_depth coerces
        ):
            path = os.path.join(self.tmp, filename)
            self.assertTrue(os.path.exists(path), f"missing: {filename}")
            with Image.open(path) as im:
                self.assertEqual(im.mode, expected_mode, f"{filename} mode")

    def test_save_to_nonexistent_dir_raises(self):
        """Engine doesn't pre-create output_dir — confirm PIL's error
        propagates up so the slot's try/except can report it."""
        gray = Image.new("RGBA", (4, 4), (127, 127, 127, 255))
        p = os.path.join(self.tmp, "src_Base_Color.png")
        gray.save(p)

        missing_dir = os.path.join(self.tmp, "does_not_exist")
        engine = MapCompositor()
        engine.total_len = 1
        with self.assertRaises(Exception):
            engine.composite_images(
                {"Base_Color": [(p, _load(p))]}, missing_dir, name="t"
            )


class TestHandlerHygiene(unittest.TestCase):
    """The class-level logger must not accumulate stale text-widget handlers
    when the UI is created multiple times in one session.
    """

    def test_only_one_text_widget_handler_per_redirect(self):
        # Simulate the slot's sweep logic against two fake widgets with .append.
        class _FakeWidget:
            def __init__(self):
                self.lines: List[str] = []

            def append(self, msg):
                self.lines.append(msg)

        engine = MapCompositor()
        widget_a = _FakeWidget()
        engine.logger.setup_logging_redirect(widget_a)

        # New "session" — sweep stale handlers, attach a fresh one.
        widget_b = _FakeWidget()
        for h in list(engine.logger.handlers):
            if hasattr(h, "widget"):
                engine.logger.removeHandler(h)
        engine.logger.setup_logging_redirect(widget_b)

        text_handlers = [h for h in engine.logger.handlers if hasattr(h, "widget")]
        self.assertEqual(
            len(text_handlers),
            1,
            "Stale text-widget handlers should be swept before redirecting",
        )
        # And the surviving handler points to the new widget.
        self.assertIs(text_handlers[0].widget, widget_b)

        # Cleanup so subsequent tests don't see the leftover handler.
        for h in list(engine.logger.handlers):
            if hasattr(h, "widget"):
                engine.logger.removeHandler(h)


class TestPublicApi(unittest.TestCase):
    """Engine surface is reachable from pythontk's top level."""

    def test_engine_symbols_importable(self):
        import pythontk

        for name in ("MapCompositor", "BatchResult", "NormalOutputMode"):
            self.assertTrue(
                hasattr(pythontk, name), f"pythontk.{name} must be importable"
            )


class TestRetryPassRespectsExistingComplement(unittest.TestCase, _LoggerCaptureMixin):
    """When the source folder already contains both Normal_DirectX and
    Normal_OpenGL, the engine must not auto-invert — even if Normal_DirectX
    fails the first composite pass and is processed through the retry path.
    Previously the retry pass only saw the failed subset and would clobber
    the user-provided OpenGL output with an inverted copy of the DX file.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_retry_complement_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_retry_does_not_overwrite_existing_opengl(self):
        # Generic Normal pair — content in the center, alpha=0 at all
        # four corners so get_background() agrees on a uniform bg and
        # the first pass succeeds, populating the mask the DX retry needs.
        def _alpha_with_center_content(size, content_color):
            im = Image.new("RGBA", size, (0, 0, 0, 0))
            for x in range(size[0] // 2 - 1, size[0] // 2 + 1):
                for y in range(size[1] // 2 - 1, size[1] // 2 + 1):
                    im.putpixel((x, y), content_color)
            return im

        n_a = _alpha_with_center_content((16, 16), (200, 100, 50, 255))
        n_b = _alpha_with_center_content((16, 16), (50, 100, 200, 255))
        n_a_path = os.path.join(self.tmp, "a_Normal.png")
        n_b_path = os.path.join(self.tmp, "b_Normal.png")
        n_a.save(n_a_path)
        n_b.save(n_b_path)

        # Two DX layers with mismatched solid backgrounds → forces the
        # first pass to defer to the mask-retry path.
        dx_a = _solid_rgba((16, 16), (127, 127, 255, 255))
        dx_b = _solid_rgba((16, 16), (0, 0, 0, 255))
        dx_a_path = os.path.join(self.tmp, "a_Normal_DirectX.png")
        dx_b_path = os.path.join(self.tmp, "b_Normal_DirectX.png")
        dx_a.save(dx_a_path)
        dx_b.save(dx_b_path)

        # Distinct user-provided OpenGL — pure green so any inversion-
        # clobber from the DX retry path would be detectable.
        gl = _solid_rgba((16, 16), (0, 255, 0, 255))
        gl_path = os.path.join(self.tmp, "a_Normal_OpenGL.png")
        gl.save(gl_path)

        engine = MapCompositor()
        engine.process_batch(
            {
                "Normal": [
                    (n_a_path, _load(n_a_path)),
                    (n_b_path, _load(n_b_path)),
                ],
                "Normal_DirectX": [
                    (dx_a_path, _load(dx_a_path)),
                    (dx_b_path, _load(dx_b_path)),
                ],
                "Normal_OpenGL": [(gl_path, _load(gl_path))],
            },
            self.tmp,
            name="batch",
        )

        gl_out_path = os.path.join(self.tmp, "batch_Normal_OpenGL.png")
        self.assertTrue(os.path.exists(gl_out_path))
        result = _load(gl_out_path).convert("RGBA")
        # The user's pure-green OpenGL must survive — not be replaced by
        # an inversion of the (127,127,255) DX neutral.
        self.assertEqual(result.getpixel((0, 0)), (0, 255, 0, 255))


class TestFormatProbeUsesOnDiskSource(unittest.TestCase, _LoggerCaptureMixin):
    """The integrability check must run against the on-disk source, not
    the engine's in-memory copy. The retry pass overwrites the un-baked
    area with the map type's default background (127,127,255), seeding a
    faint gradient at the mask boundary; that synthetic gradient is enough
    to push borderline correlations across the detector threshold and
    fire a false-positive ``declared X but pixel analysis suggests Y``
    warning.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_probe_src_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_detect_receives_unmodified_on_disk_image(self):
        import numpy as np
        import pythontk as ptk

        # Generic Normal layers — content in the center, alpha=0 at all
        # four corners so get_background() agrees on a uniform bg and
        # the first pass succeeds, populating the mask the DX retry needs.
        def _alpha_with_center_content(size, content_color):
            im = Image.new("RGBA", size, (0, 0, 0, 0))
            for x in range(size[0] // 2 - 1, size[0] // 2 + 1):
                for y in range(size[1] // 2 - 1, size[1] // 2 + 1):
                    im.putpixel((x, y), content_color)
            return im

        n_a = _alpha_with_center_content((16, 16), (200, 100, 50, 255))
        n_b = _alpha_with_center_content((16, 16), (50, 100, 200, 255))
        n_a_path = os.path.join(self.tmp, "a_Normal.png")
        n_b_path = os.path.join(self.tmp, "b_Normal.png")
        n_a.save(n_a_path)
        n_b.save(n_b_path)

        # Two DX layers with different solid bgs to force retry.
        # Crucially the bg colors differ from the map type's default
        # (127,127,255), so the retry-path fill actually rewrites the
        # source image and the difference between probe and on-disk
        # becomes detectable.
        dx_a = _solid_rgba((16, 16), (50, 50, 200, 255))
        dx_b = _solid_rgba((16, 16), (0, 0, 0, 255))
        dx_a_path = os.path.join(self.tmp, "a_Normal_DirectX.png")
        dx_b_path = os.path.join(self.tmp, "b_Normal_DirectX.png")
        dx_a.save(dx_a_path)
        dx_b.save(dx_b_path)

        seen: list = []
        original = ptk.MapFactory.detect_normal_map_format

        def spy(image, threshold=0.25, min_gradient_std=1.0):
            seen.append(image.copy() if hasattr(image, "copy") else image)
            return None  # Never trip the warning — we only inspect the input.

        ptk.MapFactory.detect_normal_map_format = staticmethod(spy)
        try:
            engine = MapCompositor()
            engine.process_batch(
                {
                    "Normal": [
                        (n_a_path, _load(n_a_path)),
                        (n_b_path, _load(n_b_path)),
                    ],
                    "Normal_DirectX": [
                        (dx_a_path, _load(dx_a_path)),
                        (dx_b_path, _load(dx_b_path)),
                    ],
                },
                self.tmp,
                name="batch",
            )
        finally:
            ptk.MapFactory.detect_normal_map_format = original

        self.assertTrue(seen, "detect_normal_map_format was not invoked")
        probe = seen[0]
        on_disk = _load(dx_a_path).convert("RGB")
        self.assertTrue(
            np.array_equal(np.array(probe.convert("RGB")), np.array(on_disk)),
            "probe image must match the on-disk source byte-for-byte",
        )


if __name__ == "__main__":
    unittest.main()
