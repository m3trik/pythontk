# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.engines.lighting.look_match.measures: what a match is scored on."""

import math
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from pythontk.core_utils.engines.lighting.look_match.measures import LookMeasures, LookScore
from pythontk.core_utils.engines.lighting.look_match.target import LookRegion, LookTarget


def _target(regions, focus=None):
    d = tempfile.mkdtemp(prefix="look_measures_")
    path = os.path.join(d, "ref.png")
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(path)
    return d, LookTarget(path, regions, focus=focus)


class TestLookMeasures(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.ref = (0.2 + 0.3 * rng.random((40, 60, 3))).astype(
            np.float32
        )  # display sRGB 0..1
        self.box = LookRegion(
            "box",
            "area",
            {"rect": [0.25, 0.25, 0.75, 0.75]},
            measures=["mean", "peak", "footprint", "chroma", "hue", "warmth"],
        )
        self.dir, self.target = _target([self.box])
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def score(self, ours, target=None):
        return LookScore(ours, self.ref, target or self.target)

    def test_identical_frames_score_zero(self):
        score = self.score(self.ref.copy())
        for result in score.region("box").values():
            np.testing.assert_allclose(result.residuals, 0.0, atol=1e-6)
            self.assertAlmostEqual(result.score, 0.0, places=6)

    def test_a_brighter_region_scores_its_ratio(self):
        ours = self.ref.copy()
        ours[10:30, 15:45] *= 1.5
        mean = self.score(ours).region("box")["mean"]
        self.assertAlmostEqual(mean.score, math.log(1.5), delta=0.02)
        self.assertGreater(mean.ours, mean.ref)

    def test_peak_reads_the_brightest_percent(self):
        ours = self.ref.copy()
        ours[20, 30] = 1.0  # one hot pixel: ~0.17% of the region, inside its top 1%
        result = self.score(ours).region("box")
        self.assertGreater(result["peak"].score, 0.05)
        self.assertAlmostEqual(result["mean"].score, 0.0, delta=0.01)

    def test_footprint_sees_a_wider_highlight(self):
        ref = self.ref.copy()
        ref[18:22, 28:32] = 0.95  # a small highlight
        ours = ref.copy()
        ours[14:26, 22:38] = 0.95  # the same level, spread wider
        result = LookScore(ours, ref, self.target).region("box")
        self.assertGreater(result["footprint"].score, 0.5)

    def test_a_warmer_region_scores_warmth_and_hue(self):
        ours = self.ref.copy()
        ours[..., 0] *= 1.2
        ours[..., 2] *= 0.8
        result = self.score(ours).region("box")
        self.assertGreater(result["warmth"].score, 0.0)
        self.assertNotAlmostEqual(result["hue"].score, 0.0, places=2)

    def test_a_line_profile_follows_the_light(self):
        line = LookRegion(
            "ramp",
            "line",
            {"line": [[0.1, 0.5], [0.9, 0.5]]},
            measures=["profile", "peak_floor"],
        )
        d, target = _target([line])
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        ours = self.ref.copy()
        ours[:, :30] *= 2.0  # brighter on the left half
        result = LookScore(ours, self.ref, target).region("ramp")
        res = result["profile"].residuals
        self.assertGreater(res[: len(res) // 3].mean(), 0.3)
        self.assertLess(abs(res[-len(res) // 3 :].mean()), 0.05)

    def test_a_quad_scores_masked_median_columns(self):
        quad = LookRegion(
            "board",
            "quad",
            {"quad": [[0, 0], [1, 0], [1, 1], [0, 1]], "grid": [6, 4]},
            measures=["columns", "rows"],
            exclude={
                "ours": [{"rect": [0, 0, 1 / 6, 1]}],
                "ref": [{"rect": [0, 0, 1 / 6, 1]}],
            },
        )
        d, target = _target([quad])
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        ours = self.ref.copy()
        ours[:, :10] = 1.0  # a pole in front of the board, excluded on both sides
        result = LookScore(ours, self.ref, target).region("board")
        self.assertEqual(
            result["columns"].residuals.shape, (5,)
        )  # the excluded column is dropped
        np.testing.assert_allclose(result["columns"].residuals, 0.0, atol=0.02)
        self.assertEqual(result["rows"].residuals.shape, (4,))

    def test_focus_share_reads_where_the_brightest_light_lands(self):
        d, target = _target([self.box], focus="box")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        ref = self.ref.copy()
        ref[12:28, 18:42] += 0.4  # the reference's brightest light is on the subject
        ours = self.ref.copy()
        ours[:8] += 0.4  # ours puts it at the top of the frame
        frame = LookScore(ours, ref, target).frame()
        self.assertLess(frame["focus_share"].score, -0.5)

    def test_frame_measures_without_a_focus(self):
        frame = self.score(self.ref * 0.5).frame()
        self.assertIn("median", frame)
        self.assertIn("p95", frame)
        self.assertNotIn("focus_share", frame)
        self.assertLess(frame["median"].score, -0.3)

    def test_a_tolerance_band_zeroes_small_misses(self):
        ours = self.ref.copy()
        ours[10:30, 15:45] *= 1.05
        score = self.score(ours)
        self.assertGreater(np.abs(score.residuals(tolerance=0.0)).max(), 0.01)
        tight = score.residuals(tolerance=0.0)
        loose = score.residuals(tolerance=0.2)
        self.assertLess(np.abs(loose).sum(), np.abs(tight).sum())

    def test_the_region_weight_scales_its_residuals(self):
        ours = self.ref * 1.3
        light = LookScore(ours, self.ref, self.target).residuals(tolerance=0.0)
        self.box.weight = 3.0
        heavy = LookScore(ours, self.ref, self.target).residuals(tolerance=0.0)
        self.assertAlmostEqual(
            np.abs(heavy).sum() / np.abs(light).sum(), 3.0, delta=0.6
        )

    def test_an_unknown_measure_is_named(self):
        bad = LookRegion("box", "area", {"rect": [0, 0, 1, 1]}, measures=["sparkle"])
        d, target = _target([bad])
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        with self.assertRaises(KeyError) as ctx:
            LookScore(self.ref, self.ref, target)
        self.assertIn("sparkle", str(ctx.exception))

    def test_a_measure_can_be_registered(self):
        measures = LookMeasures.default()
        measures.register(
            "max",
            "area",
            lambda o, r, **_: (float(o.max()), float(r.max())),
            unit="luma",
        )
        region = LookRegion("box", "area", {"rect": [0, 0, 1, 1]}, measures=["max"])
        d, target = _target([region])
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        result = LookScore(self.ref * 0.5, self.ref, target, measures=measures).region(
            "box"
        )["max"]
        self.assertLess(result.score, 0.0)

    def test_a_one_value_statistic_reads_as_a_number(self):
        """A profile left with one usable value (a quad with one column both
        sides keep): ``float`` of a 1-element array is a TypeError in numpy 2."""
        measures = LookMeasures.default()
        measures.register(
            "single",
            "area",
            lambda o, r, **_: (np.asarray([float(o.mean())]), np.asarray([float(r.mean())])),
        )
        region = LookRegion("box", "area", {"rect": [0, 0, 1, 1]}, measures=["single"])
        d, target = _target([region])
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        (row,) = [
            r for r in LookScore(self.ref * 0.5, self.ref, target, measures=measures).rows()
            if r["measure"] == "single"
        ]
        self.assertAlmostEqual(row["ours"], float(self.ref.mean()) * 0.5, places=5)
        self.assertIsInstance(row["ref"], float)

    def test_a_score_reads_the_regions_it_scored_after_the_target_changes(self):
        """A panel edits the target while an analysis runs: a region renamed or
        added after scoring raised KeyError from every reading of the score."""
        line = LookRegion("rim", "line", {"line": [[0.1, 0.1], [0.9, 0.9]]})
        d, target = _target([self.box, line])
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        score = LookScore(self.ref * 1.3, self.ref, target)
        before = (score.summary(), score.rows(), score.residuals(), score.cost())
        target.rename("box", "centre")
        target.add(LookRegion("late", "area", {"rect": [0, 0, 0.5, 0.5]}))
        summary, rows, residuals, cost = (
            score.summary(),
            score.rows(),
            score.residuals(),
            score.cost(),
        )
        self.assertEqual([r["region"] for r in summary], ["box", "rim"])
        self.assertEqual(summary, before[0])
        self.assertEqual(rows, before[1])
        np.testing.assert_array_equal(residuals, before[2])
        self.assertEqual(cost, before[3])

    def test_summary_rows_name_region_and_measure(self):
        rows = self.score(self.ref * 1.2).rows()
        self.assertTrue(
            any(r["region"] == "box" and r["measure"] == "mean" for r in rows)
        )
        self.assertTrue(any(r["region"] == "(frame)" for r in rows))

    def test_a_region_reads_as_one_line_a_person_can_act_on(self):
        """Within tolerance or not, how much brighter on screen, and a word for
        a colour that is off: what a panel's region table shows."""
        line = LookRegion("rim", "line", {"line": [[0.1, 0.1], [0.9, 0.9]]})
        d, target = _target([self.box, line])
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        same = LookScore(self.ref.copy(), self.ref, target).summary()
        self.assertEqual([r["region"] for r in same], ["box", "rim"])
        self.assertTrue(all(r["within"] for r in same))
        self.assertEqual([r["colour"] for r in same], ["", ""])

        ours = self.ref.copy()
        ours *= 1.5
        ours[..., 0] *= 1.25
        ours[..., 2] *= 0.7
        box, rim = LookScore(np.clip(ours, 0, 1), self.ref, target).summary()
        self.assertFalse(box["within"])
        self.assertGreater(box["brightness"], 0.3)  # about 40-50% brighter on screen
        self.assertIn("warm", box["colour"])
        self.assertIn(box["worst"], ("mean", "peak", "chroma", "warmth", "hue", "footprint"))
        self.assertGreater(rim["brightness"], 0.3)  # a line reads its profile's level
        dark = LookScore(self.ref * 0.5, self.ref, target).summary(tolerance=0.1)[0]
        self.assertLess(dark["brightness"], -0.3)
        plain = LookRegion("plain", "area", {"rect": [0.25, 0.25, 0.75, 0.75]})
        d, defaults = _target([plain])  # the default measures (no footprint)
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        (near,) = LookScore(self.ref * 1.05, self.ref, defaults).summary(tolerance=0.1)
        self.assertTrue(near["within"])
        self.assertAlmostEqual(near["brightness"], 0.05, delta=0.02)


if __name__ == "__main__":
    unittest.main(exit=False)
