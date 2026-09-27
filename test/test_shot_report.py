# !/usr/bin/python
# coding=utf-8
"""ShotReport -- the Shots panels' one-line summaries (pure)."""

import unittest
from types import SimpleNamespace

from pythontk.core_utils.engines.shots.shot_report import ShotReport


def _shot(start, end, *objects):
    return SimpleNamespace(
        start=start, end=end, duration=end - start, objects=list(objects)
    )


class TestSummary(unittest.TestCase):
    def test_no_shots_is_empty(self):
        self.assertEqual(ShotReport.summary([]), "")

    def test_counts_frames_objects_and_span(self):
        shots = [_shot(1, 101, "a", "b"), _shot(111, 211, "b", "c")]
        self.assertEqual(
            ShotReport.summary(shots), "2 shots · 200f · 3 objects · [1–211]"
        )

    def test_singular_forms(self):
        self.assertEqual(
            ShotReport.summary([_shot(0, 10, "a")]),
            "1 shot · 10f · 1 object · [0–10]",
        )


class TestDeltas(unittest.TestCase):
    def test_nothing_moved(self):
        deltas = [(0.0, 0.0), (1e-9, -1e-9)]
        self.assertFalse(ShotReport.moved(deltas))
        self.assertEqual(
            ShotReport.delta_summary("Trimmed", deltas), "Trimmed: nothing to do"
        )

    def test_head_and_tail_frames_sum_by_magnitude(self):
        deltas = [(-4.0, 2.0), (3.0, -1.0)]
        self.assertTrue(ShotReport.moved(deltas))
        self.assertEqual(
            ShotReport.delta_summary("Trimmed", deltas), "Trimmed: 7f head, 3f tail"
        )

    def test_no_deltas_is_nothing_to_do(self):
        self.assertEqual(
            ShotReport.delta_summary("Added leading space", []),
            "Added leading space: nothing to do",
        )


if __name__ == "__main__":
    unittest.main()
