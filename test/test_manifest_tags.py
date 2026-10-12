# !/usr/bin/python
# coding=utf-8
"""Tests for ``ManifestTags`` (``core_utils/engines/shots/manifest/manifest_tags.py``):
the Shot Manifest's colour tags -- the user's, kept per source on the scene's
store, and the automatic ``color_by`` rule.
"""

import os
import sys
import unittest

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pythontk.core_utils.engines.shots.manifest.manifest_model import BuilderStep
from pythontk.core_utils.engines.shots.manifest.manifest_tags import ManifestTags
from pythontk.core_utils.engines.shots.shot_model import ShotStore

SLOTS = ["tag1", "tag2", "tag3"]


def _steps(*pairs):
    """``BuilderStep`` per ``(step_id, section)``."""
    return [BuilderStep(sid, section, "", "", []) for sid, section in pairs]


class TestAuto(unittest.TestCase):
    """The automatic layer: a rule groups steps; groups take slots in order."""

    def test_sections_take_slots_in_order_of_first_appearance(self):
        tags = ManifestTags.auto(
            _steps(("A01", "A"), ("A02", "A"), ("B01", "B"), ("C01", "C")),
            "section",
            SLOTS,
        )
        self.assertEqual(
            tags, {"A01": "tag1", "A02": "tag1", "B01": "tag2", "C01": "tag3"}
        )

    def test_slots_wrap_past_the_last(self):
        tags = ManifestTags.auto(
            _steps(("A01", "A"), ("B01", "B"), ("C01", "C"), ("D01", "D")),
            "section",
            SLOTS,
        )
        self.assertEqual(tags["D01"], "tag1")

    def test_a_step_without_a_section_gets_no_colour(self):
        """A detected step has no section: nothing to group it by."""
        tags = ManifestTags.auto(_steps(("A01", "A"), ("x", "")), "section", SLOTS)
        self.assertEqual(tags, {"A01": "tag1"})

    def test_none_and_unknown_rules_colour_nothing(self):
        steps = _steps(("A01", "A"))
        self.assertEqual(ManifestTags.auto(steps, "none", SLOTS), {})
        self.assertEqual(ManifestTags.auto(steps, "bogus", SLOTS), {})
        self.assertEqual(ManifestTags.auto(steps, "section", []), {})

    def test_hiding_steps_never_reshuffles_the_colours(self):
        """The sheet ahead of the shown steps fixes the order: with section B
        hidden, C keeps the colour it had, and a step only the shown list has
        (one Local Edits added) still gets its section's."""
        sheet = _steps(("A01", "A"), ("B01", "B"), ("C01", "C"))
        shown = _steps(("A01", "A"), ("C01", "C"), ("NEW_STEP", "C"))
        tags = ManifestTags.auto([*sheet, *shown], "section", SLOTS)
        self.assertEqual(tags["C01"], "tag3")
        self.assertEqual(tags["NEW_STEP"], "tag3")

    def test_neighbouring_sections_never_take_neighbouring_hues(self):
        """Bug: sections took the hue-ordered tag palette in turn -- red,
        orange, yellow, green... -- so the strip read as one gradient and
        hid where a section began and ended ("too difficult to see the start
        and end of a section, defeating the purpose").
        Fixed: 2026-10-10"""
        from pythontk.core_utils.color import Palette

        slots = list(Palette.tags())  # hue order: tag1 red ... tag8 pink
        sections = "ABCDEFGH"
        tags = ManifestTags.auto(
            _steps(*((f"{s}01", s) for s in sections)), "section", slots
        )
        order = [slots.index(tags[f"{s}01"]) for s in sections]
        self.assertEqual(len(set(order)), 8, "every section its own colour")
        for a, b in zip(order, order[1:]):
            gap = min((a - b) % 8, (b - a) % 8)
            self.assertGreaterEqual(gap, 3, f"neighbours {slots[a]} / {slots[b]}")


class TestUserTags(unittest.TestCase):
    """The user's layer: per source on the store, dirtying it only on change."""

    def setUp(self):
        self.store = ShotStore()
        self.dirtied = 0
        original = self.store.mark_dirty

        def counting(*args, **kwargs):
            self.dirtied += 1
            return original(*args, **kwargs)

        self.store.mark_dirty = counting

    def test_assign_sets_and_clears_per_source(self):
        self.assertTrue(
            ManifestTags.assign(self.store, "a.csv", ["A01", "A02"], "tag2")
        )
        self.assertTrue(ManifestTags.assign(self.store, "", ["S1"], "tag5"))
        self.assertEqual(
            ManifestTags.read(self.store, "a.csv"), {"A01": "tag2", "A02": "tag2"}
        )
        self.assertEqual(ManifestTags.read(self.store, ""), {"S1": "tag5"})
        ManifestTags.assign(self.store, "a.csv", ["A01"], None)
        self.assertEqual(ManifestTags.read(self.store, "a.csv"), {"A02": "tag2"})

    def test_an_emptied_layer_is_dropped(self):
        ManifestTags.assign(self.store, "a.csv", ["A01"], "tag1")
        ManifestTags.assign(self.store, "a.csv", ["A01"], None)
        self.assertNotIn("a.csv", self.store.manifest_tags)

    def test_only_a_change_dirties_the_store(self):
        """A dirty store re-saves the scene's record: re-picking the colour a
        row already has, or clearing an untagged row, must not."""
        ManifestTags.assign(self.store, "a.csv", ["A01"], "tag1")
        self.assertEqual(self.dirtied, 1)
        self.assertFalse(ManifestTags.assign(self.store, "a.csv", ["A01"], "tag1"))
        self.assertFalse(ManifestTags.assign(self.store, "a.csv", ["B01"], None))
        self.assertEqual(self.dirtied, 1)

    def test_read_is_a_copy_and_tolerates_no_store(self):
        ManifestTags.assign(self.store, "a.csv", ["A01"], "tag1")
        ManifestTags.read(self.store, "a.csv")["A01"] = "tag8"
        self.assertEqual(ManifestTags.read(self.store, "a.csv"), {"A01": "tag1"})
        self.assertEqual(ManifestTags.read(None, "a.csv"), {})

    def test_a_moved_sheet_keeps_its_tags_and_a_pick_re_keys_them(self):
        """Bug: tags were found by the sheet's exact path, so a moved sheet
        showed none.  Found as the Local Edits are
        (``ManifestEdits.layer_key``); a pick re-keys tags and edits together.
        Fixed: 2026-10-11"""
        ManifestTags.assign(self.store, "O:/a/x.csv", ["A01"], "tag1")
        self.store.manifest_edits["O:/a/x.csv"] = {"steps": {"A01": {"name": "N"}}}
        self.assertEqual(ManifestTags.read(self.store, "C:/b/x.csv"), {"A01": "tag1"})
        self.assertTrue(ManifestTags.assign(self.store, "C:/b/x.csv", ["A02"], "tag2"))
        self.assertEqual(list(self.store.manifest_tags), ["C:/b/x.csv"])
        self.assertEqual(list(self.store.manifest_edits), ["C:/b/x.csv"])
        self.assertEqual(
            ManifestTags.read(self.store, "C:/b/x.csv"), {"A01": "tag1", "A02": "tag2"}
        )
        # Two sheets of that name keep layers: neither is found from a third
        # path, and each keeps its own.
        self.store.manifest_tags["D:/c/x.csv"] = {"B01": "tag4"}
        self.assertEqual(ManifestTags.read(self.store, "E:/d/x.csv"), {})
        self.assertEqual(ManifestTags.read(self.store, "D:/c/x.csv"), {"B01": "tag4"})


if __name__ == "__main__":
    unittest.main()
