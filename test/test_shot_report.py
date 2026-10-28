# !/usr/bin/python
# coding=utf-8
"""ShotReport -- the Shots panels' one-line summaries (pure)."""

import unittest
from types import SimpleNamespace

from pythontk.core_utils.engines.shots.shot_model import ShotStore
from pythontk.core_utils.engines.shots.shot_report import ShotReport


def _shot(start, end, *objects):
    return SimpleNamespace(
        start=start, end=end, duration=end - start, objects=list(objects)
    )


def _named(name, start, end):
    return SimpleNamespace(name=name, start=start, end=end)


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


class TestDeleteScopes(unittest.TestCase):
    """The Delete Shots words the Shots panel and the Shot Sequencer share."""

    def test_the_labels_are_the_stores_scopes_in_its_order(self):
        self.assertEqual(
            [scope for _label, scope in ShotReport.SCOPE_LABELS],
            list(ShotStore.DELETE_SCOPES),
        )
        self.assertEqual(set(ShotReport.SCOPE_CLAUSES), set(ShotStore.DELETE_SCOPES))

    def test_a_scope_has_its_menu_word(self):
        self.assertEqual(ShotReport.scope_label("empty"), "Empty")
        with self.assertRaises(ValueError):
            ShotReport.scope_label("unknown")

    def test_the_count_says_what_the_scope_takes(self):
        shots = [_named("A", 0, 10), _named("B", 20, 30)]
        self.assertEqual(ShotReport.scope_count(shots, "all"), "all 2 shot(s)")
        self.assertEqual(
            ShotReport.scope_count(shots, "empty"),
            "2 shot(s) with nothing keyed in their frames",
        )

    def test_the_question_names_the_shots_and_what_stays(self):
        title, text = ShotReport.removal_question([_named("Gone", 40, 60)], "stale")
        self.assertEqual(title, "Delete Stale Shots")
        self.assertTrue(text.startswith("Delete 1 shot(s) whose objects are all gone"))
        self.assertIn("  Gone [40–60]", text)
        self.assertIn("no keyframe is touched and no other shot moves", text)

    def test_a_long_list_is_cut_short(self):
        shots = [_named(f"S{i}", i * 10, i * 10 + 5) for i in range(14)]
        _title, text = ShotReport.removal_question(shots, "all")
        self.assertIn("  S11 [110–115]", text)
        self.assertNotIn("  S12 ", text)
        self.assertIn("… and 2 more", text)


if __name__ == "__main__":
    unittest.main()
