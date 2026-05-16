#!/usr/bin/python
# coding=utf-8
"""Tests for prefix/suffix hardening across StrUtils and the texture-name helpers."""
import unittest

from pythontk import StrUtils, ImgUtils
from pythontk.img_utils.map_factory import MapFactory


class TestStripKnownAffix(unittest.TestCase):
    """`strip_known_affix` is a conservative primitive: it strips only the
    configured affix and adjacent `_` separators, never other underscores."""

    def test_prefix_case_insensitive(self):
        self.assertEqual(StrUtils.strip_known_affix("Mat_brick", prefix="Mat_"), "brick")
        self.assertEqual(StrUtils.strip_known_affix("MAT_brick", prefix="Mat_"), "brick")
        self.assertEqual(StrUtils.strip_known_affix("mat_brick", prefix="Mat_"), "brick")

    def test_prefix_separator_required(self):
        # 'Matte' must NOT be misread as 'Mat' + 'te'
        self.assertEqual(
            StrUtils.strip_known_affix("Matte_door", prefix="Mat_"), "Matte_door"
        )

    def test_prefix_doubled_underscores(self):
        self.assertEqual(StrUtils.strip_known_affix("Mat__brick", prefix="Mat_"), "brick")

    def test_prefix_tolerates_leading_underscores(self):
        # Stray leading `_` between the start and the prefix is consumed.
        self.assertEqual(
            StrUtils.strip_known_affix("_Mat_brick", prefix="Mat_"), "brick"
        )
        self.assertEqual(
            StrUtils.strip_known_affix("__Mat_brick", prefix="Mat_"), "brick"
        )

    def test_suffix_case_insensitive(self):
        self.assertEqual(StrUtils.strip_known_affix("brick_MAT", suffix="_MAT"), "brick")
        self.assertEqual(StrUtils.strip_known_affix("brick_mat", suffix="_MAT"), "brick")

    def test_suffix_no_false_positive(self):
        # 'Diagram' must NOT be misread as 'Dia' + 'gram' when suffix is '_MAT'
        self.assertEqual(StrUtils.strip_known_affix("Diagram", suffix="_MAT"), "Diagram")

    def test_suffix_tolerates_trailing_underscores(self):
        # Stray trailing `_` after the suffix is consumed.
        self.assertEqual(
            StrUtils.strip_known_affix("brick_MAT_", suffix="_MAT"), "brick"
        )
        self.assertEqual(
            StrUtils.strip_known_affix("brick_MAT__", suffix="_MAT"), "brick"
        )

    def test_conservative_no_match_no_change(self):
        # When no affix matches, the input is returned unchanged — including
        # leading/trailing underscores. Callers wanting full normalization should
        # use `apply_affix` or chain `.strip("_")` themselves.
        self.assertEqual(
            StrUtils.strip_known_affix("_brick_", prefix="Mat_"), "_brick_"
        )

    def test_partial_match_preserves_other_side(self):
        # Only the prefix side is touched; trailing `_` is left intact.
        self.assertEqual(
            StrUtils.strip_known_affix("Mat_brick_", prefix="Mat_"), "brick_"
        )

    def test_empty_affix_passthrough(self):
        # No affixes -> no-op (importantly: does NOT strip underscores).
        self.assertEqual(StrUtils.strip_known_affix("_my_thing_"), "_my_thing_")
        self.assertEqual(StrUtils.strip_known_affix("Mat_brick"), "Mat_brick")

    def test_both_affixes(self):
        self.assertEqual(
            StrUtils.strip_known_affix(
                "Mat_brick_MAT", prefix="Mat_", suffix="_MAT"
            ),
            "brick",
        )


class TestApplyAffix(unittest.TestCase):
    def test_empty_affixes_passthrough(self):
        # No affixes -> exact passthrough, no underscore mutation.
        self.assertEqual(StrUtils.apply_affix("_my_thing_"), "_my_thing_")
        self.assertEqual(StrUtils.apply_affix("brick"), "brick")

    def test_idempotent_prefix(self):
        # Re-applying the same prefix doesn't duplicate it.
        self.assertEqual(StrUtils.apply_affix("Mat_brick", prefix="Mat_"), "Mat_brick")
        self.assertEqual(
            StrUtils.apply_affix(
                StrUtils.apply_affix("brick", prefix="Mat_"), prefix="Mat_"
            ),
            "Mat_brick",
        )

    def test_apply_to_unprefixed(self):
        self.assertEqual(StrUtils.apply_affix("brick", prefix="Mat_"), "Mat_brick")

    def test_apply_both(self):
        self.assertEqual(
            StrUtils.apply_affix("brick", prefix="Mat_", suffix="_MAT"),
            "Mat_brick_MAT",
        )

    def test_replace_different_case(self):
        # Different case of existing prefix is normalized to the configured form.
        self.assertEqual(
            StrUtils.apply_affix("MAT_brick", prefix="Mat_"), "Mat_brick"
        )

    def test_strips_dangling_underscores_on_affix_side(self):
        # When applying a prefix, leading `_` is collapsed.
        self.assertEqual(StrUtils.apply_affix("_brick", prefix="Mat_"), "Mat_brick")
        # When applying only a suffix, leading `_` is preserved (not the affix side).
        self.assertEqual(StrUtils.apply_affix("_brick", suffix="_MAT"), "_brick_MAT")
        # Symmetric: trailing `_` collapsed only when a suffix is applied.
        self.assertEqual(StrUtils.apply_affix("brick_", suffix="_MAT"), "brick_MAT")
        self.assertEqual(StrUtils.apply_affix("brick_", prefix="Mat_"), "Mat_brick_")

    def test_preserves_internal_underscores(self):
        # Internal `_` (between non-affix tokens) is preserved.
        self.assertEqual(
            StrUtils.apply_affix("my_brick_thing", prefix="Mat_"),
            "Mat_my_brick_thing",
        )


class TestGetBaseTextureNameAffix(unittest.TestCase):
    """Verify both texture-name resolvers strip configured prefixes/suffixes."""

    cases = [
        # (path, prefix, suffix, expected)
        ("Mat_brick_Albedo.png", "Mat_", "", "brick"),
        ("MAT_brick_Albedo.png", "Mat_", "", "brick"),
        ("mat_brick_Albedo.png", "Mat_", "", "brick"),
        ("Mat_brick.png", "Mat_", "", "brick"),
        ("Mat_brick_.png", "Mat_", "", "brick"),
        # Leading _ before the prefix
        ("_Mat_brick_Albedo.png", "Mat_", "", "brick"),
        # No false positive
        ("Matte_door_Normal.png", "Mat_", "", "Matte_door"),
        # Suffix mode: file already carries the user suffix in its base name.
        ("brick_MAT_Albedo.png", "", "_MAT", "brick"),
        # Without affixes
        ("brick_Albedo.png", "", "", "brick"),
    ]

    def test_map_factory(self):
        for path, pfx, sfx, expected in self.cases:
            with self.subTest(path=path, prefix=pfx, suffix=sfx):
                self.assertEqual(
                    MapFactory.get_base_texture_name(path, prefix=pfx, suffix=sfx),
                    expected,
                )

    def test_img_utils(self):
        for path, pfx, sfx, expected in self.cases:
            with self.subTest(path=path, prefix=pfx, suffix=sfx):
                self.assertEqual(
                    ImgUtils.get_base_texture_name(path, prefix=pfx, suffix=sfx),
                    expected,
                )

    def test_backward_compat_no_affix_args(self):
        # Without affix args, behavior is unchanged (preserves original .rstrip).
        self.assertEqual(
            MapFactory.get_base_texture_name("Mat_brick_Albedo.png"), "Mat_brick"
        )
        self.assertEqual(
            ImgUtils.get_base_texture_name("Mat_brick_Albedo.png"), "Mat_brick"
        )
        # Trailing underscore still gets cleaned up even without affix args.
        self.assertEqual(
            MapFactory.get_base_texture_name("Mat_brick_.png"), "Mat_brick"
        )
        self.assertEqual(
            ImgUtils.get_base_texture_name("Mat_brick_.png"), "Mat_brick"
        )


if __name__ == "__main__":
    unittest.main()
