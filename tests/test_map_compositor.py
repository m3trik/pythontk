# !/usr/bin/python
# coding=utf-8
"""Smoke tests for the pure :class:`MapCompositor` engine.

The engine emits status via ``self.logger`` (LoggingMixin). Tests attach
an in-memory handler to capture records without going through Qt.
"""

import logging
import os
import shutil
import sys
import tempfile
import unittest

try:  # the EXR writer routes through OpenCV; CI runners ship without it
    import cv2  # noqa: F401

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
from typing import List

import numpy as np
from PIL import Image

from pythontk.core_utils.engines.textures import map_compositor as mc_module
from pythontk.core_utils.engines.textures.map_compositor import (
    BatchResult,
    MapCompositor,
    NormalOutputMode,
)


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


# The engine's "must not import Qt" guard used to live here as a substring
# search over this one module. It is superseded by
# test_packaging_metadata.TestNoDccImports, which walks the AST of all 142
# modules and distinguishes a module-level import (the real hazard) from a
# lazily-resolved one -- where the substring version read its own comments as
# violations and would have missed ``import  maya`` with two spaces.


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

    @staticmethod
    def _edge_island(cols, size=16, fg=255, bg=0, mode="L"):
        """A flat island running to the UV edge -- a padding-off export."""
        im = Image.new(mode, (size, size), bg)
        for y in range(size):
            for x in cols:
                im.putpixel((x, y), fg)
        return im

    def test_flat_edge_islands_survive_the_retry_pass(self):
        """End-to-end guard for the corner-tie background pick.

        Roughness/Metallic/AO/Opacity are routinely FLAT, and a padding-off
        export runs their islands to the UV edge -- so two layers can put an
        island in two corners each and tie the true background on corner
        count. When the tie resolved to the island colour the mask inverted,
        the retry filled the ISLANDS, and process_batch still returned RETRIED:
        destroyed maps reported as a successful recovery. (Worse than the older
        behaviour, which failed loudly with MASK_FAILURE.)
        """
        a = self._edge_island(range(0, 6))
        b = self._edge_island(range(10, 16))

        MapCompositor().process_batch(
            {"Roughness": [("a.png", a), ("b.png", b)]}, self.tmp, name="AB"
        )

        out = Image.open(os.path.join(self.tmp, "AB_Roughness.png")).convert("L")
        self.assertEqual(out.getpixel((2, 8)), 255, "layer A's island was erased")
        self.assertEqual(out.getpixel((12, 8)), 255, "layer B's island was erased")

    def test_a_corner_tie_does_not_contaminate_other_types(self):
        """`_seed_masks` ORs masks across types, so one type resolving its
        background to the island colour used to drive the union to full
        coverage and degrade every OTHER type to last-layer-wins -- silently.
        """
        rough_a = self._edge_island(range(0, 6))
        rough_b = self._edge_island(range(10, 16))

        def rgb(cols, colour):
            im = Image.new("RGB", (16, 16), (0, 0, 0))
            for y in range(16):
                for x in cols:
                    im.putpixel((x, y), colour)
            return im

        MapCompositor().process_batch(
            {
                "Roughness": [("a.png", rough_a), ("b.png", rough_b)],
                "Base_Color": [
                    ("a.png", rgb(range(0, 6), (255, 6, 40))),
                    ("b.png", rgb(range(10, 16), (12, 200, 80))),
                ],
            },
            self.tmp,
            name="AB",
        )

        out = Image.open(os.path.join(self.tmp, "AB_Base_Color.png")).convert("RGB")
        self.assertEqual(out.getpixel((2, 8)), (255, 6, 40))
        self.assertEqual(out.getpixel((12, 8)), (12, 200, 80))

    def test_clean_batch_reports_success(self):
        p = self._write("a_Base_Color.png", (127, 127, 127, 255))
        engine = MapCompositor()
        result = engine.process_batch(
            {"Base_Color": [(p, _load(p))]}, self.tmp, name="test"
        )
        self.assertIs(result, BatchResult.SUCCESS)

    def test_process_batch_resets_state_between_runs(self):
        engine = MapCompositor()
        stale = Image.new("L", (4, 4), 128)
        engine.masks = [stale]
        engine.total_progress = 999

        p = self._write("a_Base_Color.png", (127, 127, 127, 255))
        engine.process_batch({"Base_Color": [(p, _load(p))]}, self.tmp, name="test")

        self.assertNotIn(stale, engine.masks)
        # A clean batch (no retry) never pays for mask creation.
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

    def test_opaque_uniform_bg_type_seeds_masks_for_retry(self):
        """Regression: an opaque single-colour bg (e.g. an ``L`` Height map)
        must seed masks for the retry pass. Previously masks were only
        seeded from transparent bgs, so a batch whose only clean set was
        opaque failed every masked type with "Unable to create masks"
        despite the message promising a single-colour bg is enough."""
        size = (8, 8)
        # Height: uniform opaque bg with a 2x2 content block per layer.
        h_a = Image.new("L", size, 128)
        h_b = Image.new("L", size, 128)
        for x, y in [(2, 2), (3, 2), (2, 3), (3, 3)]:
            h_a.putpixel((x, y), 200)
        for x, y in [(5, 5), (6, 5), (5, 6), (6, 6)]:
            h_b.putpixel((x, y), 60)
        # Base_Color: non-uniform (dilated / no padding) bg per layer.
        c_a = Image.new("RGB", size, (255, 0, 0))
        c_a.putpixel((0, 0), (1, 2, 3))
        c_b = Image.new("RGB", size, (0, 0, 255))
        c_b.putpixel((7, 7), (4, 5, 6))
        paths = {}
        for stem, im in [
            ("A_Height", h_a),
            ("B_Height", h_b),
            ("A_Base_Color", c_a),
            ("B_Base_Color", c_b),
        ]:
            paths[stem] = os.path.join(self.tmp, f"{stem}.png")
            im.save(paths[stem])
        sorted_images = {
            "Height": [(paths["A_Height"], h_a), (paths["B_Height"], h_b)],
            "Base_Color": [(paths["A_Base_Color"], c_a), (paths["B_Base_Color"], c_b)],
        }
        engine = MapCompositor()
        cap = self.attach_capture(engine)
        result = engine.process_batch(sorted_images, self.tmp, name="AB")

        self.assertIs(result, BatchResult.RETRIED, cap.messages())
        self.assertEqual(len(engine.masks), 2)
        self.assertNotIn("ERROR", cap.levels(), cap.messages())
        out = _load(os.path.join(self.tmp, "AB_Base_Color.png")).convert("RGB")
        self.assertEqual(out.getpixel((2, 2)), (255, 0, 0))
        self.assertEqual(out.getpixel((5, 5)), (0, 0, 255))

    def test_16bit_height_is_rescaled_not_clipped(self):
        """Regression: Painter exports Height as ``I;16``. Pillow's I;16 ->
        L/RGBA convert CLIPS at 255, so a mid-grey (~32767) height map
        loaded as solid white, its bg looked uniform, the composite came
        out white and the mask seeded from it was empty — every masked
        type then filled solid with its default bg."""
        size = (8, 8)
        h_a = Image.new("I;16", size, 32767)
        h_b = Image.new("I;16", size, 32767)
        for x, y in [(2, 2), (3, 2), (2, 3), (3, 3)]:
            h_a.putpixel((x, y), 50000)
        for x, y in [(5, 5), (6, 5), (5, 6), (6, 6)]:
            h_b.putpixel((x, y), 10000)
        c_a = Image.new("RGB", size, (255, 0, 0))
        c_a.putpixel((0, 0), (1, 2, 3))
        c_b = Image.new("RGB", size, (0, 0, 255))
        c_b.putpixel((7, 7), (4, 5, 6))
        paths = {}
        for stem, im in [
            ("A_Height", h_a),
            ("B_Height", h_b),
            ("A_Base_Color", c_a),
            ("B_Base_Color", c_b),
        ]:
            paths[stem] = os.path.join(self.tmp, f"{stem}.png")
            im.save(paths[stem])
        sorted_images = {
            "Height": [
                (paths["A_Height"], _load(paths["A_Height"])),
                (paths["B_Height"], _load(paths["B_Height"])),
            ],
            "Base_Color": [(paths["A_Base_Color"], c_a), (paths["B_Base_Color"], c_b)],
        }
        self.assertEqual(sorted_images["Height"][0][1].mode, "I;16")
        engine = MapCompositor()
        cap = self.attach_capture(engine)
        result = engine.process_batch(sorted_images, self.tmp, name="AB")

        self.assertIs(result, BatchResult.RETRIED, cap.messages())
        height = _load(os.path.join(self.tmp, "AB_Height.png"))
        self.assertEqual(height.mode, "L")
        self.assertEqual(height.getpixel((0, 0)), 127)  # 32767/257, not clipped white
        self.assertEqual(height.getpixel((2, 2)), 195)  # 50000/257
        self.assertEqual(height.getpixel((5, 5)), 39)  # 10000/257
        out = _load(os.path.join(self.tmp, "AB_Base_Color.png")).convert("RGB")
        self.assertEqual(out.getpixel((2, 2)), (255, 0, 0))
        self.assertEqual(out.getpixel((5, 5)), (0, 0, 255))

    def test_masks_union_all_types_and_tolerate_content_in_corners(self):
        """Regression (Painter export, padding off): Height's flat content
        equals its grey bg, so a mask keyed off Height alone covered ~5% of
        the islands and every map came out ~85% default bg. Base_Color has
        a real black bg but islands touch two corners, so its bg was never
        detected. Masks must be the UNION over every type of
        'pixel != that type's bg', with the bg detected by majority-corner
        + dominance rather than four identical corners."""
        size = (16, 16)
        # Two disjoint islands: A = cols 0-6, B = cols 9-15, all rows.
        # Height: everything mid-grey (flat), one bump per island.
        h_a = Image.new("L", size, 127)
        h_a.putpixel((2, 2), 200)
        h_b = Image.new("L", size, 127)
        h_b.putpixel((12, 12), 60)
        # Base_Color: black bg, island painted; islands run into the
        # bottom corners so corners are NOT uniform.
        c_a = Image.new("RGB", size, (0, 0, 0))
        c_b = Image.new("RGB", size, (0, 0, 0))
        for y in range(16):
            for x in range(0, 7):
                c_a.putpixel((x, y), (255, 0, 0))
            for x in range(9, 16):
                c_b.putpixel((x, y), (0, 0, 255))
        sorted_images = {
            "Height": [("A_Height.png", h_a), ("B_Height.png", h_b)],
            "Base_Color": [("A_Base_Color.png", c_a), ("B_Base_Color.png", c_b)],
        }
        engine = MapCompositor()
        cap = self.attach_capture(engine)
        result = engine.process_batch(sorted_images, self.tmp, name="AB")
        self.assertIs(result, BatchResult.RETRIED, cap.messages())
        self.assertEqual(len(engine.masks), 2)
        cov = [(np.array(m) > 0).mean() for m in engine.masks]
        self.assertAlmostEqual(cov[0], 7 / 16, places=2)
        self.assertAlmostEqual(cov[1], 7 / 16, places=2)
        out = _load(os.path.join(self.tmp, "AB_Base_Color.png")).convert("RGB")
        self.assertEqual(out.getpixel((3, 8)), (255, 0, 0))
        self.assertEqual(out.getpixel((12, 8)), (0, 0, 255))
        self.assertEqual(out.getpixel((0, 15)), (255, 0, 0))  # corner content kept
        self.assertEqual(out.getpixel((15, 15)), (0, 0, 255))
        height = _load(os.path.join(self.tmp, "AB_Height.png"))
        self.assertEqual(height.getpixel((2, 2)), 200)
        self.assertEqual(height.getpixel((12, 12)), 60)

    def test_no_uniform_bg_anywhere_is_one_diagnosis(self):
        """Edge-to-edge (fully dilated) sources: no type can seed a mask.
        Expect MASK_FAILURE with ONE explanatory error, not a per-file
        'Composite failed' for every layer."""
        size = (8, 8)
        layers = {}
        for stem, color in [
            ("A_Base_Color", (255, 0, 0)),
            ("B_Base_Color", (0, 0, 255)),
            ("A_Height", 100),
            ("B_Height", 150),
        ]:
            im = Image.new("RGB" if "Base" in stem else "L", size, color)
            im.putpixel((0, 0), (9, 9, 9) if "Base" in stem else 9)  # break uniformity
            layers[stem] = (os.path.join(self.tmp, f"{stem}.png"), im)
        sorted_images = {
            "Base_Color": [layers["A_Base_Color"], layers["B_Base_Color"]],
            "Height": [layers["A_Height"], layers["B_Height"]],
        }
        engine = MapCompositor()
        cap = self.attach_capture(engine)
        result = engine.process_batch(sorted_images, self.tmp, name="AB")
        self.assertIs(result, BatchResult.MASK_FAILURE)
        errors = [m for m, lvl in zip(cap.messages(), cap.levels()) if lvl == "ERROR"]
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("detectable background", errors[0])
        self.assertNotIn("Composite failed", errors[0])


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


class TestOutputTemplateScoping(unittest.TestCase, _LoggerCaptureMixin):
    """The template post-pass must only see maps this batch actually wrote.

    Regression: the post-pass used to re-scan ``output_dir``, so compositing
    into a shared library folder (e.g. a project ``sourceimages``) swept every
    pre-existing texture set into ``prepare_maps`` and generated packed /
    converted siblings for unrelated materials.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_scope_")
        self.src = os.path.join(self.tmp, "src")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.src)
        os.makedirs(self.out)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # DirectX normal source so BOTH mode synthesizes the OpenGL complement —
    # the auto-generated sibling has to be tracked too, not just direct saves.
    _MAPS = (
        ("Base_Color", "RGB", (200, 100, 50)),
        ("Metallic", "L", 200),
        ("Roughness", "L", 100),
        ("Ambient_Occlusion", "L", 150),
        ("Normal_DirectX", "RGB", (128, 128, 255)),
    )

    def _stage_sources(self):
        """Layers to composite — written to ``src``, composited into ``out``."""
        sorted_images = {}
        for typ, mode, color in self._MAPS:
            path = os.path.join(self.src, f"layer_{typ}.png")
            Image.new(mode, (4, 4), color).save(path)
            sorted_images[typ] = [(path, _load(path))]
        return sorted_images

    def _plant_foreign_set(self, name="UNRELATED"):
        """A complete, pre-existing texture set already sitting in ``out``."""
        planted = {}
        for typ, mode, color in self._MAPS:
            path = os.path.join(self.out, f"{name}_{typ}.png")
            Image.new(mode, (4, 4), color).save(path)
            with open(path, "rb") as f:
                planted[path] = f.read()
        return planted

    def test_template_ignores_preexisting_sets_in_output_dir(self):
        planted = self._plant_foreign_set()
        engine = MapCompositor()
        engine.output_template = "Unity HDRP"

        result = engine.process_batch(self._stage_sources(), self.out, name="mat")
        self.assertIs(result, BatchResult.SUCCESS)

        # Our own set still gets its packed map.
        self.assertTrue(os.path.isfile(os.path.join(self.out, "mat_MSAO.png")))

        # The foreign set gained nothing …
        produced = {n for n in os.listdir(self.out) if n.startswith("UNRELATED_")}
        self.assertEqual(
            produced,
            {os.path.basename(p) for p in planted},
            "Template post-pass generated files for an unrelated texture set",
        )
        # … and lost nothing: every planted file is byte-identical.
        for path, before in planted.items():
            with open(path, "rb") as f:
                self.assertEqual(f.read(), before, f"{path} was rewritten in place")

    def test_written_paths_tracks_batch_output(self):
        engine = MapCompositor()
        self._plant_foreign_set()
        engine.process_batch(self._stage_sources(), self.out, name="mat")

        written = set(engine.written_paths)
        self.assertTrue(written, "Engine did not record any written paths")
        self.assertTrue(all(os.path.isfile(p) for p in written))
        self.assertTrue(
            all(os.path.basename(p).startswith("mat_") for p in written),
            f"written_paths leaked non-batch files: {sorted(written)}",
        )
        # Includes the auto-generated OpenGL complement, not just direct saves.
        self.assertIn(os.path.join(self.out, "mat_Normal_OpenGL.png"), written)

    def test_stale_packed_map_is_repacked_not_passed_through(self):
        """A stale packed map from a previous run must not shadow this batch.

        The old directory scan fed the pre-existing ``<name>_MSAO.png`` back
        into ``prepare_maps`` as an *input*, so the packer saw MSAO already
        in the inventory and passed it through untouched — the freshly
        composited Metallic/AO/Roughness never reached the packed output.
        """
        stale = os.path.join(self.out, "mat_MSAO.png")
        Image.new("RGBA", (4, 4), (1, 2, 3, 4)).save(stale)
        with open(stale, "rb") as f:
            before = f.read()

        engine = MapCompositor()
        engine.output_template = "Unity HDRP"
        engine.process_batch(self._stage_sources(), self.out, name="mat")

        with open(stale, "rb") as f:
            self.assertNotEqual(
                f.read(), before, "Stale packed map was passed through, not repacked"
            )

    def test_pruned_normal_source_is_untracked(self):
        """OPENGL_ONLY deletes the DirectX source it just wrote — the
        post-pass must not be handed a path that no longer exists.
        """
        engine = MapCompositor()
        engine.normal_output_mode = NormalOutputMode.OPENGL_ONLY
        engine.process_batch(self._stage_sources(), self.out, name="mat")

        written = engine.written_paths
        self.assertIn(os.path.join(self.out, "mat_Normal_OpenGL.png"), written)
        self.assertNotIn(os.path.join(self.out, "mat_Normal_DirectX.png"), written)
        for path in written:
            self.assertTrue(os.path.isfile(path), f"tracked but missing: {path}")

    def test_written_paths_reset_between_batches(self):
        engine = MapCompositor()
        engine.process_batch(self._stage_sources(), self.out, name="mat")
        first = list(engine.written_paths)
        engine.process_batch(self._stage_sources(), self.out, name="mat2")
        self.assertTrue(first)
        self.assertTrue(
            all(os.path.basename(p).startswith("mat2_") for p in engine.written_paths),
            "written_paths carried over from the prior batch",
        )

    def test_explicit_files_argument_overrides_tracking(self):
        engine = MapCompositor()
        engine.output_template = "Unity HDRP"
        for typ, mode, color in self._MAPS:
            Image.new(mode, (4, 4), color).save(
                os.path.join(self.out, f"solo_{typ}.png")
            )
        files = [
            os.path.join(self.out, f"solo_{typ}.png") for typ, _m, _c in self._MAPS
        ]
        engine.apply_output_template(self.out, files=files)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "solo_MSAO.png")))

    def test_standalone_call_still_scans_dir(self):
        """No batch run + no explicit files → back-compat directory scan."""
        engine = MapCompositor()
        engine.output_template = "Unity HDRP"
        for typ, mode, color in self._MAPS:
            Image.new(mode, (4, 4), color).save(
                os.path.join(self.out, f"legacy_{typ}.png")
            )
        engine.apply_output_template(self.out)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "legacy_MSAO.png")))


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
    """`_seed_masks` builds per-layer content masks as the UNION over every
    map type with a detectable background — transparent alpha or an opaque
    solid colour — so a boundary that is flat/eroded in one source is
    recovered from another."""

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
        masks = engine._seed_masks(sorted_images)
        self.assertEqual(len(masks), 1)
        mask = masks[0]
        self.assertEqual(mask.mode, "L")
        for px in [(3, 3), (3, 4), (4, 3), (4, 4)]:
            self.assertEqual(mask.getpixel(px), 255, f"content lost at {px}")
        # A corner pixel that was alpha=0 in BOTH sources must remain bg.
        self.assertEqual(mask.getpixel((0, 0)), 0)

    def test_opaque_solid_bg_source_contributes(self):
        # RGB-only source (no alpha band) with a solid bg — content is
        # every pixel that differs from that bg.
        a = Image.new("RGB", (4, 4), (10, 20, 30))
        a.putpixel((1, 1), (200, 200, 200))
        sorted_images = {"Base_Color": [("a.png", a)]}

        engine = MapCompositor()
        cap = self.attach_capture(engine)
        masks = engine._seed_masks(sorted_images)
        self.assertEqual(len(masks), 1)
        self.assertEqual(masks[0].mode, "L")
        self.assertEqual(masks[0].getpixel((1, 1)), 255)
        self.assertEqual(masks[0].getpixel((0, 0)), 0)
        self.assertTrue(
            any("Creating masks from" in m for m in cap.messages()),
            "expected source log line",
        )

    def test_alpha_and_opaque_sources_union(self):
        # Opaque solid-bg Roughness marks (1,1); transparent Base_Color
        # marks (2,2). Both count.
        size = (4, 4)
        opaque_bg = Image.new("RGBA", size, (50, 50, 50, 255))
        opaque_bg.putpixel((1, 1), (200, 200, 200, 255))
        transparent_bg = self._rgba_with_alpha_pixels(size, [(2, 2)])
        sorted_images = {
            "Roughness": [("opaque.png", opaque_bg)],
            "Base_Color": [("trans.png", transparent_bg)],
        }
        engine = MapCompositor()
        masks = engine._seed_masks(sorted_images)
        self.assertEqual(masks[0].getpixel((2, 2)), 255)
        self.assertEqual(masks[0].getpixel((1, 1)), 255)
        self.assertEqual(masks[0].getpixel((0, 0)), 0)

    def test_skips_sources_with_mismatched_layer_count(self):
        # 2 layers in the majority, 1 in a mismatched alpha source — the
        # mismatched source must be ignored so positional alignment holds.
        size = (6, 6)
        triggering = [
            ("t0.png", self._rgba_with_alpha_pixels(size, [(1, 1)])),
            ("t1.png", self._rgba_with_alpha_pixels(size, [(2, 2)])),
        ]
        mismatched = [("m0.png", self._rgba_with_alpha_pixels(size, [(3, 3), (4, 4)]))]
        sorted_images = {"Base_Color": triggering, "Roughness": mismatched}

        engine = MapCompositor()
        masks = engine._seed_masks(sorted_images)
        self.assertEqual(len(masks), 2)
        self.assertEqual(masks[0].getpixel((1, 1)), 255)
        self.assertEqual(masks[0].getpixel((3, 3)), 0)
        self.assertEqual(masks[0].getpixel((4, 4)), 0)

    def test_no_detectable_bg_returns_empty(self):
        # Every corner distinct and no dominant colour: no source qualifies.
        a = Image.new("RGB", (4, 4), (0, 0, 0))
        for i, xy in enumerate([(0, 0), (3, 0), (0, 3), (3, 3)]):
            a.putpixel(xy, (i + 1, 0, 0))
        engine = MapCompositor()
        self.assertEqual(engine._seed_masks({"Base_Color": [("a.png", a)]}), [])

    def test_solid_bg_tolerates_content_in_corners(self):
        # Two of four corners are content; the other two plus 60% of the
        # image are the bg → still detected.
        a = Image.new("RGB", (10, 10), (0, 0, 0))
        for y in range(10):
            for x in range(6, 10):
                a.putpixel((x, y), (255, 0, 0))  # right strip incl. 2 corners
        self.assertEqual(
            MapCompositor._solid_background([np.asarray(a.convert("RGBA"))]),
            (0, 0, 0, 255),
        )

    def test_solid_bg_prefers_the_larger_area_when_corners_tie(self):
        """Mirror of test_solid_bg_tolerates_content_in_corners, with the
        content strip on the LEFT so it is the first corner scanned.

        Both colours hold exactly two corners, so corner count ties. The old
        tie-break was ``max()``, which returns the FIRST maximum -- i.e. scan
        order -- so the 40% content strip won and the mask came out inverted:
        the composite fills the islands instead of the background, and because
        the wrong colour is then recorded, the retry pass "succeeds" via the
        known-bg shortcut and the batch reports success over destroyed maps.
        Coverage is the tie-break now: 60% bg beats 40% content.
        """
        a = Image.new("RGB", (10, 10), (0, 0, 0))  # bg = black, 60%
        for y in range(10):
            for x in range(0, 4):
                a.putpixel((x, y), (255, 0, 0))  # left strip incl. 2 corners

        self.assertEqual(
            MapCompositor._solid_background([np.asarray(a.convert("RGBA"))]),
            (0, 0, 0, 255),
        )

    def test_solid_bg_rejects_minor_corner_colour(self):
        # 3 corners share a colour that covers only 3 pixels — content, not bg.
        a = Image.new("RGB", (10, 10), (0, 0, 0))
        for xy in [(0, 0), (9, 0), (0, 9)]:
            a.putpixel(xy, (7, 7, 7))
        self.assertIsNone(
            MapCompositor._solid_background([np.asarray(a.convert("RGBA"))])
        )

    def test_overlap_warning(self):
        # Two layers whose masks coincide → warn (shared UV space).
        a = Image.new("RGB", (8, 8), (0, 0, 0))
        for y in range(8):
            for x in range(4):
                a.putpixel((x, y), (255, 0, 0))
        b = a.copy()
        engine = MapCompositor()
        cap = self.attach_capture(engine)
        engine._seed_masks({"Base_Color": [("a.png", a), ("b.png", b)]})
        self.assertTrue(any("overlap" in m for m in cap.messages()), cap.messages())

    def _two_layer_strips(self, size=(8, 8), pad=0):
        """Layer 0 = left half, layer 1 = right half, each island dilated
        by ``pad`` px into the other's half (Painter edge padding)."""
        w, h = size
        a = Image.new("RGBA", size, (0, 0, 0, 0))
        b = Image.new("RGBA", size, (0, 0, 0, 0))
        for y in range(h):
            for x in range(w):
                if x < w // 2 + pad:
                    a.putpixel((x, y), (255, 0, 0, 255))
                if x >= w // 2 - pad:
                    b.putpixel((x, y), (0, 0, 255, 255))
        return a, b

    def test_noisy_source_dropped_when_a_clean_one_exists(self):
        """A source whose own layers overlap (dilated islands) must not
        inflate the union when another source outlines them cleanly: the
        clean source wins, the noisy one is reported dropped, and no
        overlap WARNING is emitted."""
        ta, tb = self._two_layer_strips(pad=0)  # tight
        da, db = self._two_layer_strips(pad=2)  # dilated -> 4/8 cols overlap
        engine = MapCompositor()
        cap = self.attach_capture(engine)
        masks = engine._seed_masks(
            {
                "Roughness": [("da.png", da), ("db.png", db)],
                "Base_Color": [("ta.png", ta), ("tb.png", tb)],
            }
        )
        self.assertEqual(len(masks), 2)
        m0 = np.array(masks[0]) > 0
        m1 = np.array(masks[1]) > 0
        self.assertFalse((m0 & m1).any(), "union still carries the dilation")
        self.assertNotIn("WARNING", cap.levels(), cap.messages())
        msgs = " ".join(cap.messages())
        self.assertIn("Roughness", msgs)
        self.assertIn("Base_Color", msgs)

    def test_overlap_in_every_source_still_warns(self):
        """When no source disagrees, the overlap is real (shared UV space /
        object in both sets) and the user must hear about it."""
        da, db = self._two_layer_strips(pad=2)
        engine = MapCompositor()
        cap = self.attach_capture(engine)
        engine._seed_masks(
            {
                "Roughness": [("da.png", da), ("db.png", db)],
                "Base_Color": [("da2.png", da.copy()), ("db2.png", db.copy())],
            }
        )
        self.assertIn("WARNING", cap.levels(), cap.messages())


class TestMapInfoBundle(unittest.TestCase):
    def test_mapinfo_is_frozen(self):
        from pythontk.core_utils.engines.textures.map_compositor import _MapInfo

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

    def test_clean_no_alpha_roughness_round_trips_byte_identical(self):
        # Regression guard: when the source has no partial-alpha pixels,
        # the fill must be a strict no-op. Locks in that the fix doesn't
        # touch opaque content under any circumstance.
        # Corners must be uniform so the compositor takes the happy path
        # rather than the mask-retry route.
        size = (16, 16)
        im = Image.new("L", size, 255)  # uniform white bg / corners
        # Paint a deterministic pattern in the interior only, leaving the
        # outermost border at 255 so corners stay uniform.
        for x in range(1, size[0] - 1):
            for y in range(1, size[1] - 1):
                im.putpixel((x, y), (x * 13 + y * 7) % 256)
        path = os.path.join(self.tmp, "src_Roughness.png")
        im.save(path)

        engine = MapCompositor()
        engine.total_len = 1
        engine.composite_images(
            {"Roughness": [(path, _load(path))]}, self.tmp, name="test"
        )

        out = os.path.join(self.tmp, "test_Roughness.png")
        src_arr = np.array(_load(path).convert("L"))
        out_arr = np.array(_load(out))
        self.assertEqual(out_arr.shape, src_arr.shape)
        # Every pixel must round-trip byte-identical; the fill_transparent_rgb
        # path must not touch fully-opaque content.
        diff = out_arr.astype(int) - src_arr.astype(int)
        self.assertEqual(
            int(np.abs(diff).max()),
            0,
            "clean L-mode roughness must round-trip byte-identical",
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
        return self._write_normal("Normal_DirectX")

    def _write_normal(self, typ: str, green: int = 127) -> str:
        flat = Image.new("RGBA", (4, 4), (127, green, 255, 255))
        path = os.path.join(self.tmp, f"src_{typ}.png")
        flat.save(path)
        return path

    def test_both_is_symmetric_generates_dx_from_gl_source(self):
        """BOTH means both, whichever format the batch supplies.

        The guard that stops the engine clobbering a user-provided
        complement used to short-circuit the whole branch whenever the
        batch carried OpenGL — so a GL-only batch (the common case) never
        got its DirectX complement, and the DX-from-GL branch below it was
        unreachable.
        """
        path = self._write_normal("Normal_OpenGL")
        engine = MapCompositor()
        engine.total_len = 1
        engine.composite_images(
            {"Normal_OpenGL": [(path, _load(path))]}, self.tmp, name="t"
        )
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "t_Normal_OpenGL.png")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "t_Normal_DirectX.png")))

    def test_both_does_not_clobber_a_supplied_complement(self):
        """Symmetry must not cost the anti-clobber guard: when the batch
        carries both formats, neither is regenerated from the other.

        Both key orders are exercised because a stray inversion is only
        observable when it lands *after* the victim's own composite — with
        one order the bad write is masked by the map that follows it, so a
        single-order test silently passes even with the guard removed.
        """
        # An off-centre green so an inversion is unmistakable: invert_channels
        # would turn 60 into 195, not a neighbouring value.
        green = 60
        dx = self._write_normal("Normal_DirectX", green=green)
        gl = self._write_normal("Normal_OpenGL", green=green)
        layers = {
            "Normal_DirectX": [(dx, _load(dx))],
            "Normal_OpenGL": [(gl, _load(gl))],
        }

        for order in (
            ("Normal_DirectX", "Normal_OpenGL"),
            ("Normal_OpenGL", "Normal_DirectX"),
        ):
            out_dir = os.path.join(self.tmp, "_".join(order))
            os.makedirs(out_dir)
            engine = MapCompositor()
            engine.total_len = 2
            engine.composite_images(
                {typ: layers[typ] for typ in order}, out_dir, name="t"
            )
            # Each output keeps its own green — not the counterpart's inversion.
            for typ in order:
                out = _load(os.path.join(out_dir, f"t_{typ}.png"))
                self.assertEqual(
                    out.convert("RGB").getpixel((0, 0))[1],
                    green,
                    f"{typ} was overwritten by an inversion (order={order})",
                )

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
    """When optimize_output is on, the save path runs MapOptimizer.optimize_map."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_opt_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_optimize_called_when_enabled(self):
        import pythontk as ptk

        calls = []
        original = ptk.MapOptimizer.optimize_map
        try:
            ptk.MapOptimizer.optimize_map = classmethod(
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
            ptk.MapOptimizer.optimize_map = original

    def test_written_path_follows_optimizer_output(self):
        """optimize_map resolves its own output name — track what it returned.

        Recording the pre-optimization path would leave ``written_paths``
        pointing at a file the optimizer renamed away, silently dropping the
        map from the output-template post-pass.
        """
        import pythontk as ptk

        original = ptk.MapOptimizer.optimize_map
        try:

            def _renaming(cls, path, **kw):
                renamed = f"{os.path.splitext(path)[0]}.tga"
                os.replace(path, renamed)
                return renamed

            ptk.MapOptimizer.optimize_map = classmethod(_renaming)

            p = os.path.join(self.tmp, "src_Base_Color.png")
            Image.new("RGBA", (4, 4), (127, 127, 127, 255)).save(p)

            engine = MapCompositor()
            engine.optimize_output = True
            engine.total_len = 1
            engine.composite_images({"Base_Color": [(p, _load(p))]}, self.tmp, name="t")

            self.assertEqual(
                engine.written_paths, [os.path.join(self.tmp, "t_Base_Color.tga")]
            )
        finally:
            ptk.MapOptimizer.optimize_map = original

    def test_optimize_not_called_when_disabled(self):
        import pythontk as ptk

        calls = []
        original = ptk.MapOptimizer.optimize_map
        try:
            ptk.MapOptimizer.optimize_map = classmethod(
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
            ptk.MapOptimizer.optimize_map = original


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


class TestModelessPackedTypeComposite(unittest.TestCase, _LoggerCaptureMixin):
    """Packed map types with ``mode=None`` (MSAO/MRAO) are filtered out of
    ``MapRegistry.get_map_modes()``. ``_composite_type`` must not subscript
    that dict directly — the mode-less key raises ``KeyError`` and crashes
    the whole batch. It must fall back to the image's natural mode instead.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_modeless_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, typ: str) -> str:
        # Solid, uniform, fully-opaque bg → composites on the first pass.
        img = _solid_rgba((4, 4), (10, 20, 30, 255))
        path = os.path.join(self.tmp, f"src_{typ}.png")
        img.save(path)

        engine = MapCompositor()
        cap = self.attach_capture(engine)
        # Must not raise KeyError on the mode-less packed key.
        failed = engine.composite_images(
            {typ: [(path, _load(path))]}, self.tmp, name="mat"
        )
        self.assertEqual(failed, {})
        self.assertNotIn("ERROR", cap.levels())
        return os.path.join(self.tmp, f"mat_{typ}.png")

    def test_msao_composites_without_keyerror(self):
        out = self._run("MSAO")
        self.assertTrue(os.path.exists(out))

    def test_mrao_composites_without_keyerror(self):
        out = self._run("MRAO")
        self.assertTrue(os.path.exists(out))


class TestWriterRouting(unittest.TestCase, _LoggerCaptureMixin):
    """The engine's two writes were raw ``PIL.Image.save``.

    Every other texture write goes through ``ImgUtils.save_image``, which
    routes by extension: PIL for the ordinary containers, OpenCV for the float
    formats (EXR/HDR), the external encoder for KTX2, plus bit-depth and lossy
    handling. Raw PIL cannot write ``.exr`` or ``.hdr`` at all -- and the
    engine *discovers its inputs* with ``ImgUtils.texture_file_types``, which
    lists both, then takes the output extension from the first input. So an
    EXR set raised ``ValueError: unknown file extension: .exr`` out of
    ``process_batch`` with nothing written.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc_writer_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _island(cols, size=16, mode="RGB", fg=(255, 0, 0), bg=(0, 0, 0)):
        im = Image.new(mode, (size, size), bg)
        for y in range(size):
            for x in cols:
                im.putpixel((x, y), fg)
        return im

    @unittest.skipUnless(HAS_CV2, "cv2 required for EXR")
    def test_an_exr_set_is_written_instead_of_raising(self):
        MapCompositor().process_batch(
            {
                "Base_Color": [
                    ("a.exr", self._island(range(0, 6))),
                    ("b.exr", self._island(range(10, 16))),
                ]
            },
            self.tmp,
            name="AB",
        )
        out = os.path.join(self.tmp, "AB_Base_Color.exr")
        self.assertTrue(os.path.isfile(out), f"not written: {os.listdir(self.tmp)}")
        self.assertGreater(os.path.getsize(out), 0)

    @unittest.skipUnless(HAS_CV2, "cv2 required for EXR")
    def test_the_normal_map_counterpart_is_written_through_the_same_writer(self):
        """The second raw save: the inverted-green counterpart map."""
        engine = MapCompositor()
        engine.process_batch(
            {
                "Normal_OpenGL": [
                    ("a.exr", self._island(range(0, 6), fg=(128, 128, 255))),
                    ("b.exr", self._island(range(10, 16), fg=(128, 128, 255))),
                ]
            },
            self.tmp,
            name="AB",
        )
        written = sorted(os.listdir(self.tmp))
        self.assertTrue(
            any(f.endswith(".exr") for f in written), f"nothing written: {written}"
        )

    def test_png_output_is_unchanged(self):
        """The routing change must not alter the ordinary path."""
        MapCompositor().process_batch(
            {
                "Base_Color": [
                    ("a.png", self._island(range(0, 6))),
                    ("b.png", self._island(range(10, 16))),
                ]
            },
            self.tmp,
            name="AB",
        )
        out = os.path.join(self.tmp, "AB_Base_Color.png")
        self.assertTrue(os.path.isfile(out))
        self.assertEqual(Image.open(out).size, (16, 16))

    def test_the_module_imports_without_pillow(self):
        """Its six siblings in the engine tolerate a missing Pillow; this one
        imported it bare, so an install without Pillow failed at import.

        Grepping for ``except ImportError`` is not enough: this module has a
        module-level ``Layers = List[Tuple[str, Image.Image]]`` alias and seven
        ``Image.Image`` annotations, all evaluated at import, so a try/except
        alone still dies on ``NoneType has no attribute 'Image'``. Import it
        for real with PIL denied.
        """
        import builtins
        import importlib

        name = mc_module.__name__
        real_import = builtins.__import__

        def deny_pil(module, *args, **kwargs):
            if module == "PIL" or module.startswith("PIL."):
                raise ImportError("simulated: no Pillow")
            return real_import(module, *args, **kwargs)

        saved = {k: v for k, v in sys.modules.items() if k.startswith("PIL")}
        saved[name] = sys.modules.get(name)
        for key in list(saved):
            sys.modules.pop(key, None)
        builtins.__import__ = deny_pil
        try:
            reloaded = importlib.import_module(name)
            self.assertIsNone(reloaded.Image, "PIL should be absent here")
        finally:
            builtins.__import__ = real_import
            sys.modules.pop(name, None)
            sys.modules.update({k: v for k, v in saved.items() if v is not None})
            importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()
