#!/usr/bin/python
# coding=utf-8
"""Tests for prefix/suffix hardening across StrUtils and the texture-name helpers."""
import unittest

from pythontk import StrUtils, ImgUtils
from pythontk.core_utils.engines.textures.map_factory import MapFactory


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
        # Underscore-delimited short map suffixes strip case-insensitively —
        # both resolvers must agree (they once drifted: ImgUtils required a
        # capital first letter even behind an explicit "_" delimiter).
        ("brick_ao.png", "", "", "brick"),
        ("Mat_brick_ao.png", "Mat_", "", "brick"),
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


class TestInferAffixMode(unittest.TestCase):
    """`infer_affix_mode` is the pure primitive: given a text and a
    delimiter, decide whether it should be applied as a prefix or suffix."""

    def test_leading_delimiter_means_suffix(self):
        self.assertEqual(StrUtils.infer_affix_mode("_MAT"), "suffix")

    def test_trailing_delimiter_means_prefix(self):
        self.assertEqual(StrUtils.infer_affix_mode("MAT_"), "prefix")

    def test_no_delimiter_in_text_falls_back_to_default(self):
        # Library default is "prefix".
        self.assertEqual(StrUtils.infer_affix_mode("MAT"), "prefix")
        # Explicit override.
        self.assertEqual(
            StrUtils.infer_affix_mode("MAT", default="suffix"), "suffix"
        )

    def test_both_edges_have_delimiter_is_ambiguous(self):
        self.assertEqual(StrUtils.infer_affix_mode("_MAT_"), "prefix")
        self.assertEqual(
            StrUtils.infer_affix_mode("_MAT_", default="suffix"), "suffix"
        )

    def test_empty_text_returns_default(self):
        self.assertEqual(StrUtils.infer_affix_mode(""), "prefix")
        self.assertEqual(StrUtils.infer_affix_mode("", default="suffix"), "suffix")

    def test_empty_delimiter_disables_detection(self):
        # Even with a clear leading underscore, empty delimiter → default.
        self.assertEqual(
            StrUtils.infer_affix_mode("_MAT", delimiter="", default="prefix"),
            "prefix",
        )
        self.assertEqual(
            StrUtils.infer_affix_mode("_MAT", delimiter="", default="suffix"),
            "suffix",
        )

    def test_custom_delimiter(self):
        # The primitive is delimiter-agnostic; common alternates include "-" and ".".
        self.assertEqual(StrUtils.infer_affix_mode("-MAT", delimiter="-"), "suffix")
        self.assertEqual(StrUtils.infer_affix_mode("MAT-", delimiter="-"), "prefix")
        self.assertEqual(StrUtils.infer_affix_mode(".obj", delimiter="."), "suffix")

    def test_multi_char_delimiter(self):
        self.assertEqual(
            StrUtils.infer_affix_mode("__MAT", delimiter="__"), "suffix"
        )
        self.assertEqual(
            StrUtils.infer_affix_mode("MAT__", delimiter="__"), "prefix"
        )

    def test_default_default_is_prefix(self):
        # Sanity: the library-level fallback is "prefix" — DCC asset naming
        # convention favors type-leading prefixes.
        self.assertEqual(StrUtils.infer_affix_mode("ambiguous"), "prefix")

    def test_unknown_default_is_coerced_to_prefix(self):
        # An invalid default string (typo, garbage) must not leak into the
        # return value. The documented return contract is "prefix" | "suffix".
        self.assertEqual(
            StrUtils.infer_affix_mode("ambiguous", default="middle"), "prefix"
        )
        self.assertEqual(
            StrUtils.infer_affix_mode("", default="garbage"), "prefix"
        )

    def test_default_is_keyword_only(self):
        # Prevent the positional footgun where a caller writes
        # infer_affix_mode("text", "_", "suffix") expecting the 3rd arg
        # to be `default` — it would silently bind to the wrong slot.
        with self.assertRaises(TypeError):
            StrUtils.infer_affix_mode("MAT", "_", "suffix")


class TestSplitAffix(unittest.TestCase):
    """`split_affix` converts a user-entered affix string + mode declaration
    into a `(prefix, suffix)` pair for `apply_affix` to consume."""

    def test_explicit_prefix(self):
        self.assertEqual(StrUtils.split_affix("MAT_", mode="prefix"), ("MAT_", ""))

    def test_explicit_suffix(self):
        self.assertEqual(StrUtils.split_affix("_MAT", mode="suffix"), ("", "_MAT"))

    def test_explicit_mode_ignores_punctuation(self):
        # Mode wins over heuristic when explicit.
        self.assertEqual(StrUtils.split_affix("_MAT", mode="prefix"), ("_MAT", ""))
        self.assertEqual(StrUtils.split_affix("MAT_", mode="suffix"), ("", "MAT_"))

    def test_auto_leading_underscore_means_suffix(self):
        # "_MAT" reads as a trailing affix on the base name.
        self.assertEqual(StrUtils.split_affix("_MAT", mode="auto"), ("", "_MAT"))

    def test_auto_trailing_underscore_means_prefix(self):
        # "MAT_" reads as a leading affix on the base name.
        self.assertEqual(StrUtils.split_affix("MAT_", mode="auto"), ("MAT_", ""))

    def test_auto_ambiguous_falls_back_to_default(self):
        # No boundary underscore — falls back to default mode.
        self.assertEqual(
            StrUtils.split_affix("MAT", mode="auto", default="suffix"), ("", "MAT")
        )
        self.assertEqual(
            StrUtils.split_affix("MAT", mode="auto", default="prefix"), ("MAT", "")
        )

    def test_auto_double_boundary_is_ambiguous(self):
        # Both edges have underscores → can't decide → default wins.
        self.assertEqual(
            StrUtils.split_affix("_MAT_", mode="auto", default="suffix"),
            ("", "_MAT_"),
        )
        self.assertEqual(
            StrUtils.split_affix("_MAT_", mode="auto", default="prefix"),
            ("_MAT_", ""),
        )

    def test_empty_text_returns_empty_pair(self):
        self.assertEqual(StrUtils.split_affix("", mode="suffix"), ("", ""))
        self.assertEqual(StrUtils.split_affix("", mode="auto"), ("", ""))

    def test_default_mode_is_auto(self):
        # No mode argument → "auto".
        self.assertEqual(StrUtils.split_affix("_MAT"), ("", "_MAT"))
        self.assertEqual(StrUtils.split_affix("MAT_"), ("MAT_", ""))

    def test_composes_with_apply_affix(self):
        # Round-trip: split_affix's output is what apply_affix expects.
        prefix, suffix = StrUtils.split_affix("_MAT", mode="auto")
        self.assertEqual(StrUtils.apply_affix("brick", prefix=prefix, suffix=suffix), "brick_MAT")

        prefix, suffix = StrUtils.split_affix("MAT_", mode="auto")
        self.assertEqual(StrUtils.apply_affix("brick", prefix=prefix, suffix=suffix), "MAT_brick")

    def test_library_default_is_prefix(self):
        # No mode/default given, no boundary underscore → falls back to "prefix".
        self.assertEqual(StrUtils.split_affix("MAT"), ("MAT", ""))

    def test_custom_delimiter(self):
        self.assertEqual(
            StrUtils.split_affix("MAT-", mode="auto", delimiter="-"), ("MAT-", "")
        )
        self.assertEqual(
            StrUtils.split_affix("-MAT", mode="auto", delimiter="-"), ("", "-MAT")
        )

    def test_empty_delimiter_disables_auto_detection(self):
        # Even with a clear boundary, empty delimiter ignores it → use default.
        self.assertEqual(
            StrUtils.split_affix("_MAT", mode="auto", delimiter="", default="prefix"),
            ("_MAT", ""),
        )

    def test_unknown_mode_is_coerced_to_auto(self):
        # A typo'd mode string must not silently route to the suffix branch.
        # "garbage" should be normalized to "auto" and run the heuristic.
        self.assertEqual(StrUtils.split_affix("_MAT", mode="garbage"), ("", "_MAT"))
        self.assertEqual(StrUtils.split_affix("MAT_", mode="typo"), ("MAT_", ""))

    def test_default_and_delimiter_are_keyword_only(self):
        # Guard against positional-arg confusion between split_affix and
        # infer_affix_mode (which order delimiter/default differently).
        with self.assertRaises(TypeError):
            StrUtils.split_affix("MAT", "auto", "suffix")
        with self.assertRaises(TypeError):
            StrUtils.split_affix("MAT", "auto", "prefix", "_")


if __name__ == "__main__":
    unittest.main()
