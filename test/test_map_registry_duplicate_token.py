#!/usr/bin/python
# coding=utf-8
"""Tests for MapRegistry.split_duplicate_token and the resolver retry it feeds.

A duplicate marker appended AFTER the map-type token ("rock_Base_Color_1")
leaves the name ending in no registry alias, so the map classifies as nothing
and every consumer -- packed-map unpacking, shader wiring, grouping -- silently
skips it.  Added: 2026-08-18
"""
import unittest

from pythontk import MapRegistry, MapFactory

from conftest import BaseTestCase


class SplitDuplicateTokenTest(BaseTestCase):
    def setUp(self):
        self.reg = MapRegistry()

    def test_splits_a_trailing_copy_index(self):
        for name, expected in (
            ("rock_Base_Color_1", ("rock_Base_Color", "_1")),
            ("rock_Base_Color_2", ("rock_Base_Color", "_2")),
            ("rock_Base_Color-3", ("rock_Base_Color", "-3")),
            ("rock_Base_Color_99", ("rock_Base_Color", "_99")),
        ):
            self.assertEqual(self.reg.split_duplicate_token(name), expected, name)

    def test_leaves_names_without_a_marker_alone(self):
        for name in ("rock_Base_Color", "rock", "rock_Normal.1001", ""):
            self.assertEqual(self.reg.split_duplicate_token(name), (name, ""), name)

    def test_technical_tails_are_not_copy_indices(self):
        """Bit depth is not a copy index.

        ``im_Height_16`` is a 16-BIT height map, not the sixteenth copy of one;
        collapsing it onto its 8-bit sibling would hand a network builder two
        Height maps for one material.
        """
        for name in (
            "im_Height_8",
            "im_Height_16",
            "im_Height_24",
            "im_Height_32",
            "im_Height_48",
            "im_Height_64",
        ):
            self.assertEqual(self.reg.split_duplicate_token(name), (name, ""), name)

    def test_resolution_tags_are_out_of_range(self):
        """``_512`` / ``_1024`` / ``_2048`` need no exclusion -- 3+ digits."""
        for name in ("wall_Normal_512", "wall_Normal_1024", "wall_Normal_2048"):
            self.assertEqual(self.reg.split_duplicate_token(name), (name, ""), name)


class ResolveWithDuplicateTokenTest(BaseTestCase):
    def test_a_duplicate_marker_no_longer_hides_the_map_type(self):
        for name, expected in (
            ("TURRETS_Base_Color_1.png", "Base_Color"),
            ("TURRETS_Metallic_2.png", "Metallic"),
            ("TURRETS_Normal_OpenGL_1.png", "Normal_OpenGL"),
            ("TURRETS_Roughness_1.png", "Roughness"),
            ("X_ORM_1.png", "ORM"),
        ):
            self.assertEqual(MapFactory.resolve_map_type(name), expected, name)

    def test_the_marker_comes_off_before_the_tile_token(self):
        """The marker is appended last, so it sits OUTSIDE the tile token."""
        self.assertEqual(
            MapFactory.resolve_map_type("rock_Normal.1001_1.png"), "Normal"
        )

    def test_a_dotted_stem_survives_the_retry(self):
        """The retry must not re-run extension stripping on a stem.

        ``os.path.splitext`` on the already-stripped ``rock.v2_Base_Color``
        returns ``('rock', '.v2_Base_Color')``, so a retry that re-entered the
        path handling silently lost every dotted version / LOD name -- while
        the same name WITHOUT a marker resolved fine.
        """
        for base, expected in (
            ("rock.v2_Base_Color", "Base_Color"),
            ("asset.LOD1_Roughness", "Roughness"),
        ):
            self.assertEqual(MapFactory.resolve_map_type(f"{base}.png"), expected)
            self.assertEqual(
                MapFactory.resolve_map_type(f"{base}_1.png"),
                expected,
                "a duplicate marker must not cost a dotted stem its map type",
            )

    def test_key_false_stays_strict_because_its_answer_names_a_file(self):
        """The two forms must NOT agree here.

        ``key=False``'s answer is spliced back into a filename, and
        ``resolve_texture_filename`` appends when it cannot find the suffix at
        the tail: a tolerant ``key=False`` would turn ``X_Base_Color_1.png``
        into ``X_Base_Color_1_Base_Color.png``. The strict ``None`` yields the
        empty suffix that leaves the name untouched.
        """
        import os

        name = "X_Base_Color_1.png"
        self.assertIsNone(MapFactory.resolve_map_type(name, key=False))
        self.assertEqual(MapFactory.resolve_map_type(name, key=True), "Base_Color")
        # The empty suffix that None produces is what keeps the name intact.
        self.assertEqual(
            os.path.basename(MapFactory.resolve_texture_filename(name, "")), name
        )

    def test_the_retry_never_changes_an_answer_the_registry_already_had(self):
        """Gated on a miss: it can add matches, never alter one."""
        for name, expected in (
            ("rock_Base_Color.png", "Base_Color"),
            ("im_Height_16.png", None),
            ("im_Height_8.png", None),
            ("wall_Normal_1024.png", None),
            ("just_a_photo_2.png", None),
            ("wall_12.png", None),
        ):
            self.assertEqual(MapFactory.resolve_map_type(name), expected, name)


if __name__ == "__main__":
    unittest.main()
