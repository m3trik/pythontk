#!/usr/bin/python
# coding=utf-8
"""Tests for ``pythontk.SceneListing`` (``file_utils/scene_listing.py``).

The Reference Manager's listing rules and its Naming conventions -- how a scene
file reads as a row, which rows a filter keeps, and where Save To Workspace
writes -- hoisted out of tentacle's panel, so a script reproduces the panel's
save path. Pure rules over plain values and a scratch project; no host.
"""

import os
import unittest

import pythontk as ptk

from conftest import BaseTestCase


class TestListingRules(BaseTestCase):
    def test_a_suffix_hides_only_at_the_end_of_the_stem(self):
        self.assertEqual(
            ptk.SceneListing.display_name("C:/p/ITA_LOCKHANDLE.ma", False, "_LOC"),
            "ITA_LOCKHANDLE.ma",
        )
        self.assertEqual(
            ptk.SceneListing.display_name("C:/p/hero_LOC.ma", False, "_LOC"),
            "hero.ma",
        )
        self.assertEqual(
            ptk.SceneListing.display_name("C:/p/hero_LOC.ma", True, "_LOC"), "hero"
        )
        self.assertEqual(
            ptk.SceneListing.display_name("C:/p/hero.ma", False, ""), "hero.ma"
        )

    def test_patterns_split_on_every_delimiter(self):
        self.assertEqual(
            ptk.SceneListing.split_patterns(" a*, b ; c,,"), ["a*", "b", "c"]
        )
        self.assertEqual(ptk.SceneListing.split_patterns(""), [])

    def test_a_name_matches_any_pattern(self):
        m = ptk.SceneListing.text_matches_filter
        self.assertTrue(m("hero_v01.ma", ["villain*", "HERO*"]))
        self.assertFalse(m("hero_v01.ma", ["HERO*"], ignore_case=False))
        self.assertFalse(m("", ["*"]))
        self.assertFalse(m("hero.ma", []))

    def test_notes_match_whole_or_by_segment(self):
        m = ptk.SceneListing.matches_notes_filter
        self.assertTrue(m("Layout, Speedrun", "*Speed*"))
        self.assertTrue(m("Layout, Speedrun", "Layout*"))
        self.assertTrue(m("Layout; Speedrun", "Speedrun"))
        self.assertTrue(m("layout", "LAYOUT"))
        self.assertFalse(m("layout", "LAYOUT", ignore_case=False))
        self.assertTrue(m("Layout", "nope, Layout"))
        self.assertFalse(m("", "*"))
        self.assertFalse(m("Layout", ""))

    def test_the_folder_structure_resolves_scenes(self):
        kept = ptk.SceneListing.filter_by_folder_structure(
            ["C:/proj/scenes/hero/hero_v01.ma", "C:/proj/loose.ma"],
            "C:/proj",
            "{scenes}/{name}",
            "_v01",
            "scenes",
        )
        self.assertEqual(kept, ["C:/proj/scenes/hero/hero_v01.ma"])

    def test_format_name_keeps_the_suffix_out_of_the_case(self):
        fmt = ptk.SceneListing.format_name
        self.assertEqual(fmt("hero", "upper", "_v01"), "HERO_v01")
        self.assertEqual(fmt("hero", "Case: upper"), "HERO")
        self.assertEqual(fmt("hero", "None", ""), "hero")

    def test_every_trailing_suffix_comes_off(self):
        self.assertEqual(
            ptk.SceneListing.strip_all_suffixes("hero_v01_v01", "_v01"), "hero"
        )
        self.assertEqual(
            ptk.SceneListing.strip_all_suffixes("hero_v01x", "_v01"), "hero_v01x"
        )


class TestNamingConventions(BaseTestCase):
    """The Save To Workspace path, over a scratch project."""

    def setUp(self):
        self.tmp_store = ptk.TempArtifacts("ptk_scene_listing")
        self.addCleanup(self.tmp_store.cleanup)
        self.root = self.tmp_store.dir_path()
        self.ws = os.path.join(self.root, "proj")
        os.makedirs(os.path.join(self.ws, "scenes"))

    def test_the_save_target_is_workspace_structure_and_suffixed_name(self):
        target = ptk.SceneListing.save_target(
            self.ws, "hero", "None", "_v01", "{scenes}/{name}", ".ma"
        )
        self.assertEqual(
            target,
            os.path.normpath(os.path.join(self.ws, "scenes", "hero", "hero_v01.ma")),
        )

    def test_the_case_reaches_the_folder_and_the_file_not_the_suffix(self):
        target = ptk.SceneListing.save_target(
            self.ws, "hero", "upper", "_v01", "{scenes}/{name}", ".blend"
        )
        self.assertEqual(
            target,
            os.path.normpath(os.path.join(self.ws, "scenes", "HERO", "HERO_v01.blend")),
        )

    def test_no_pattern_saves_in_the_workspace_root(self):
        target = ptk.SceneListing.save_target(self.ws, "hero", extension=".ma")
        self.assertEqual(target, os.path.normpath(os.path.join(self.ws, "hero.ma")))

    def test_the_pattern_is_normalized_as_the_save_does(self):
        # The preview and the save read the same path for a padded field and
        # the {scene} typo.
        target = ptk.SceneListing.save_target(
            self.ws, "hero", "None", "", " {scene}/{name} ", ".ma"
        )
        self.assertEqual(
            target,
            os.path.normpath(os.path.join(self.ws, "scenes", "hero", "hero.ma")),
        )

    def test_the_scenes_placeholder_reads_the_workspace_scene_rule(self):
        with open(os.path.join(self.ws, "workspace.mel"), "w") as fh:
            fh.write('workspace -fr "scene" "shots";\n')
        self.assertEqual(ptk.SceneListing.scenes_folder(self.ws), "shots")
        target = ptk.SceneListing.save_target(
            self.ws, "hero", pattern="{scenes}", extension=".ma"
        )
        self.assertEqual(
            target, os.path.normpath(os.path.join(self.ws, "shots", "hero.ma"))
        )

    def test_scenes_falls_back_without_a_relative_rule(self):
        self.assertEqual(ptk.SceneListing.scenes_folder(""), "scenes")
        self.assertEqual(ptk.SceneListing.scenes_folder(self.ws), "scenes")

    def test_an_invalid_workspace_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            ptk.SceneListing.save_target(
                os.path.join(self.root, "gone"), "hero", extension=".ma"
            )
        self.assertEqual(
            str(caught.exception), "Current workspace directory is invalid."
        )
        with self.assertRaises(ValueError):
            ptk.SceneListing.save_target("", "hero", extension=".ma")

    def test_an_unbalanced_pattern_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            ptk.SceneListing.save_target(
                self.ws, "hero", pattern="{scenes", extension=".ma"
            )
        self.assertTrue(
            str(caught.exception).startswith("Invalid folder structure pattern: ")
        )

    def test_the_default_save_name_drops_every_trailing_suffix_and_keeps_dots(self):
        self.assertEqual(
            ptk.SceneListing.default_save_name(
                "C:/x/hero.rig_v01_v01.ma", "None", "_v01"
            ),
            "hero.rig",
        )
        self.assertEqual(
            ptk.SceneListing.default_save_name("C:/x/hero.ma", "upper", ""), "HERO"
        )
        self.assertEqual(ptk.SceneListing.default_save_name("", "None", "_v01"), "")

    def test_the_scene_typo_reads_as_scenes(self):
        norm = ptk.SceneListing.normalize_structure_pattern
        self.assertEqual(norm(" {scene}/{name} "), "{scenes}/{name}")
        self.assertEqual(norm("{scenes}/{name}"), "{scenes}/{name}")
        self.assertEqual(norm(None), "")


if __name__ == "__main__":
    unittest.main(exit=False)
