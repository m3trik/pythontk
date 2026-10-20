# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.engines.lighting.look_match.sheet: comparison sheets."""

import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from pythontk.core_utils.engines.lighting.look_match.measures import LookScore
from pythontk.core_utils.engines.lighting.look_match.sheet import LookSheet
from pythontk.core_utils.engines.lighting.look_match.target import LookRegion, LookTarget


class TestLookSheet(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="look_sheet_")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        rng = np.random.default_rng(0)
        self.ref = rng.random((60, 100, 3)).astype(np.float32)
        self.ours = np.clip(self.ref * 1.2, 0, 1)
        path = os.path.join(self.dir, "ref.png")
        Image.fromarray((self.ref * 255).astype(np.uint8)).save(path)
        self.target = LookTarget(
            path,
            [
                LookRegion("left", "area", {"rect": [0.05, 0.1, 0.4, 0.6]}),
                LookRegion("band", "line", {"line": [[0.1, 0.8], [0.9, 0.8]]}),
            ],
        )

    def test_a_sheet_has_the_frames_and_one_row_per_region(self):
        sheet = LookSheet(width=800)
        image = sheet.render(
            self.ours,
            self.ref,
            self.target,
            LookScore(self.ours, self.ref, self.target),
            title="mix A",
        )
        self.assertEqual(image.width, 800)
        self.assertEqual(sheet.last_layout["regions"], ["left", "band"])
        self.assertGreater(
            image.height, 400
        )  # the frames row, the difference row and two region rows

    def test_one_regions_row_alone_for_a_panels_detail(self):
        sheet = LookSheet(width=600)
        score = LookScore(self.ours, self.ref, self.target)
        row = sheet.region_row(
            self.ours, self.ref, self.target, "left", score, {"key": 0.7, "fill": 0.3}
        )
        self.assertEqual(row.width, 600)
        full = sheet.render(self.ours, self.ref, self.target, score, regions=["left"])
        self.assertLess(row.height, full.height)  # no frames, no difference: the row
        with self.assertRaises(KeyError):
            sheet.region_row(self.ours, self.ref, self.target, "nowhere")

    def test_the_difference_is_red_where_ours_is_brighter(self):
        ours = self.ref.copy()
        ours[:, :50] = np.clip(ours[:, :50] * 2.0, 0, 1)
        ours[:, 50:] *= 0.5
        diff = np.asarray(LookSheet().difference(ours, self.ref)).astype(int)
        left, right = diff[:, :50].mean(axis=(0, 1)), diff[:, 50:].mean(axis=(0, 1))
        self.assertGreater(left[0], left[2])  # brighter: red
        self.assertGreater(right[2], right[0])  # darker: blue

    def test_regions_can_be_chosen(self):
        sheet = LookSheet(width=600)
        sheet.render(self.ours, self.ref, self.target, regions=["band"])
        self.assertEqual(sheet.last_layout["regions"], ["band"])

    def test_contribution_bars_are_drawn_when_given(self):
        sheet = LookSheet(width=600)
        plain = sheet.render(self.ours, self.ref, self.target)
        with_bars = sheet.render(
            self.ours,
            self.ref,
            self.target,
            contributions={"left": {"key": 0.7, "fill": 0.3}},
        )
        self.assertGreater(with_bars.height, plain.height)

    def test_a_painted_region_is_outlined_by_its_own_edge(self):
        layer = np.zeros((60, 100, 4), np.uint8)
        layer[20:40, 60:90] = (40, 200, 40, 140)
        Image.fromarray(layer, "RGBA").save(os.path.join(self.dir, "markers.png"))
        target = LookTarget(
            self.target.reference,
            [
                LookRegion(
                    "painted", "area", {"mask": "markers.png", "color": [40, 200, 40]}
                )
            ],
            base=self.dir,
        )
        sheet = LookSheet(width=600)
        image = sheet.render(
            self.ours, self.ref, target, LookScore(self.ours, self.ref, target)
        )
        self.assertEqual(sheet.last_layout["regions"], ["painted"])
        self.assertGreater(image.height, 200)

    def test_a_pair_shows_both_frames_whole_and_close_up(self):
        sheet = LookSheet(width=400)
        image = sheet.pair(self.ours, self.ref, crops={"left": (0.0, 0.0, 0.5, 1.0)})
        self.assertEqual(image.width, 400)
        # the frames row (100x60 -> 200x120) and the crop row (50x60 -> 200x240)
        self.assertEqual(image.height, 26 + (120 + 20) + (240 + 20))
        pixels = np.asarray(image).astype(float) / 255.0
        right = pixels[
            26 + 18 : 26 + 18 + 120, 200:400
        ]  # the reference sits on the right
        left = pixels[26 + 18 : 26 + 18 + 120, :200]
        self.assertAlmostEqual(right.mean(), float(self.ref.mean()), delta=0.03)
        self.assertGreater(left.mean(), right.mean())  # ours is the brighter

    def test_tiles_lay_labelled_frames_in_a_grid(self):
        frames = {
            name: np.full((40, 60, 3), value, np.float32)
            for name, value in (("a", 0.2), ("b", 0.5), ("c", 0.8))
        }
        image = LookSheet(width=600).tiles(frames, columns=2)
        self.assertEqual(image.size, (600, 400))  # two rows of 300x200 tiles
        pixels = np.asarray(image)
        self.assertLess(pixels[100, 150][0], pixels[100, 450][0])  # a darker than b
        self.assertGreater(pixels[300, 150][0], pixels[100, 450][0])  # c below a
        self.assertEqual(tuple(pixels[300, 450]), LookSheet.BACKGROUND)  # no fourth

    def test_a_region_at_the_top_is_named_inside_it(self):
        """Clamped to the frame's top, the name sat under the OURS caption."""
        sheet, size = LookSheet(width=600), (100, 60)
        top = LookRegion("wall", "area", {"rect": [0.05, 0.0, 0.9, 0.2]})
        low = LookRegion("floor", "area", {"rect": [0.05, 0.5, 0.9, 0.9]})
        self.assertEqual(sheet._label_origin(top, "ref", size)[1], 3.0)
        self.assertEqual(sheet._label_origin(low, "ref", size)[1], 30.0 - 15)
        # with the caption's band: never in it, room above or not
        self.assertEqual(sheet._label_origin(top, "ref", size, top=8)[1], 8)
        near = LookRegion("band", "area", {"rect": [0.05, 0.3, 0.9, 0.5]})
        self.assertEqual(sheet._label_origin(near, "ref", size, top=8)[1], 21.0)

    def test_a_measure_reads_good_or_bad_as_the_summary_does(self):
        """One rule for a miss: the residuals inside the tolerance. The sheet
        judged every non-log measure by its score over 15, whatever its scale:
        a warmth the summary called out read good, a grey's hue read bad."""
        from pythontk.core_utils.engines.lighting.look_match.measures import (
            MeasureResult,
        )

        results = {  # warmth: scale 5; hue: scale 30, weighed down on a grey
            "warmth": MeasureResult("warmth", 0.0, 0.0, 1.0, np.asarray([0.2]), "OKLab b x100"),
            "hue": MeasureResult("hue", 0.0, 0.0, 20.0, np.asarray([0.067]), "degrees"),
            "mean": MeasureResult("mean", 0.5, 0.5, 0.05, np.asarray([0.05]), "log"),
        }
        colours = {
            text.split()[0]: colour for text, colour in LookSheet()._score_lines(results)
        }
        self.assertEqual(colours["warmth"], LookSheet.BAD)
        self.assertEqual(colours["hue"], LookSheet.GOOD)
        self.assertEqual(colours["mean"], LookSheet.GOOD)

    def test_save_writes_a_png(self):
        path = os.path.join(self.dir, "sheet.png")
        LookSheet(width=500).save(path, self.ours, self.ref, self.target)
        with Image.open(path) as img:
            self.assertEqual(img.width, 500)


if __name__ == "__main__":
    unittest.main(exit=False)
