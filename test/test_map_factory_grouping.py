#!/usr/bin/python
# coding=utf-8
"""
Tests for MapFactory grouping/filtering helpers used by mayatk.MatUpdater:
- group_textures_by_set
- filter_redundant_maps

These drive multi-set detection and PBR map dedup. They have no on-disk
side effects — pure dict transforms.
"""
import unittest

from pythontk import MapFactory

from conftest import BaseTestCase


class GroupTexturesBySetTest(BaseTestCase):
    def test_single_set(self):
        files = [
            "/x/asset_BaseColor.png",
            "/x/asset_Normal.png",
            "/x/asset_Roughness.png",
        ]
        sets = MapFactory.group_textures_by_set(files)
        self.assertEqual(len(sets), 1, f"Expected one set, got {sets}")
        only = next(iter(sets.values()))
        self.assertEqual(len(only), 3)

    def test_multiple_sets(self):
        files = [
            "/x/wood_BaseColor.png",
            "/x/wood_Normal.png",
            "/x/metal_BaseColor.png",
            "/x/metal_Roughness.png",
        ]
        sets = MapFactory.group_textures_by_set(files)
        self.assertEqual(len(sets), 2, f"Expected two sets, got {list(sets)}")
        # Each set has 2 files
        for files_in_set in sets.values():
            self.assertEqual(len(files_in_set), 2)

    def test_empty_input(self):
        self.assertEqual(MapFactory.group_textures_by_set([]), {})

    def test_returns_full_paths_not_basenames(self):
        files = ["/some/dir/asset_Normal.png"]
        sets = MapFactory.group_textures_by_set(files)
        only_files = next(iter(sets.values()))
        self.assertEqual(only_files, files)


class FilterRedundantMapsTest(BaseTestCase):
    def test_empty_dict_no_op(self):
        d = {}
        MapFactory.filter_redundant_maps(d)
        self.assertEqual(d, {})

    def test_no_redundancy_preserved(self):
        d = {
            "Base_Color": ["/x/a_BaseColor.png"],
            "Normal": ["/x/a_Normal.png"],
        }
        original = {k: list(v) for k, v in d.items()}
        MapFactory.filter_redundant_maps(d)
        self.assertEqual(d, original)

    def test_in_place_mutation(self):
        """Should modify the dict in place (returns None)."""
        d = {"Base_Color": ["/x/a.png"]}
        ret = MapFactory.filter_redundant_maps(d)
        self.assertIsNone(ret)

    def test_dominant_removes_redundant(self):
        """Verify a precedence rule actually removes a redundant entry.

        Uses live precedence rules so this stays correct as the registry
        evolves.
        """
        rules = MapFactory.get_precedence_rules()
        if not rules:
            self.skipTest("No precedence rules registered")
        # Find a rule where dominant has a non-empty redundant list
        dominant, redundants = next(
            ((d, r) for d, r in rules.items() if r), (None, None)
        )
        if not dominant:
            self.skipTest("No precedence rule with redundants")
        redundant = redundants[0]
        d = {
            dominant: ["/x/dominant.png"],
            redundant: ["/x/redundant.png"],
        }
        MapFactory.filter_redundant_maps(d)
        self.assertIn(dominant, d)
        self.assertNotIn(redundant, d, f"{redundant} should have been removed")

    def test_redundant_kept_when_dominant_empty(self):
        rules = MapFactory.get_precedence_rules()
        dominant, redundants = next(
            ((d, r) for d, r in rules.items() if r), (None, None)
        )
        if not dominant:
            self.skipTest("No precedence rule with redundants")
        redundant = redundants[0]
        d = {
            dominant: [],  # empty -> not actually present
            redundant: ["/x/redundant.png"],
        }
        MapFactory.filter_redundant_maps(d)
        self.assertIn(redundant, d, "Redundant kept when dominant has no files")


if __name__ == "__main__":
    unittest.main()
