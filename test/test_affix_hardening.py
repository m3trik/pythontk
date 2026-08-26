#!/usr/bin/python
# coding=utf-8
"""Tests for prefix/suffix hardening across StrUtils and the texture-name helpers."""

import unittest

from pythontk import StrUtils, ImgUtils
from pythontk.core_utils.engines.textures.map_factory import MapFactory
from pythontk.core_utils.engines.textures.map_registry import MapRegistry


class TestStripKnownAffix(unittest.TestCase):
    """`strip_known_affix` is a conservative primitive: it strips only the
    configured affix and adjacent `_` separators, never other underscores."""

    def test_prefix_case_insensitive(self):
        self.assertEqual(
            StrUtils.strip_known_affix("Mat_brick", prefix="Mat_"), "brick"
        )
        self.assertEqual(
            StrUtils.strip_known_affix("MAT_brick", prefix="Mat_"), "brick"
        )
        self.assertEqual(
            StrUtils.strip_known_affix("mat_brick", prefix="Mat_"), "brick"
        )

    def test_prefix_separator_required(self):
        # 'Matte' must NOT be misread as 'Mat' + 'te'
        self.assertEqual(
            StrUtils.strip_known_affix("Matte_door", prefix="Mat_"), "Matte_door"
        )

    def test_prefix_doubled_underscores(self):
        self.assertEqual(
            StrUtils.strip_known_affix("Mat__brick", prefix="Mat_"), "brick"
        )

    def test_prefix_tolerates_leading_underscores(self):
        # Stray leading `_` between the start and the prefix is consumed.
        self.assertEqual(
            StrUtils.strip_known_affix("_Mat_brick", prefix="Mat_"), "brick"
        )
        self.assertEqual(
            StrUtils.strip_known_affix("__Mat_brick", prefix="Mat_"), "brick"
        )

    def test_suffix_case_insensitive(self):
        self.assertEqual(
            StrUtils.strip_known_affix("brick_MAT", suffix="_MAT"), "brick"
        )
        self.assertEqual(
            StrUtils.strip_known_affix("brick_mat", suffix="_MAT"), "brick"
        )

    def test_suffix_no_false_positive(self):
        # 'Diagram' must NOT be misread as 'Dia' + 'gram' when suffix is '_MAT'
        self.assertEqual(
            StrUtils.strip_known_affix("Diagram", suffix="_MAT"), "Diagram"
        )

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
            StrUtils.strip_known_affix("Mat_brick_MAT", prefix="Mat_", suffix="_MAT"),
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
        self.assertEqual(StrUtils.apply_affix("MAT_brick", prefix="Mat_"), "Mat_brick")

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
        self.assertEqual(ImgUtils.get_base_texture_name("Mat_brick_.png"), "Mat_brick")


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
        self.assertEqual(StrUtils.infer_affix_mode("MAT", default="suffix"), "suffix")

    def test_both_edges_have_delimiter_is_ambiguous(self):
        self.assertEqual(StrUtils.infer_affix_mode("_MAT_"), "prefix")
        self.assertEqual(StrUtils.infer_affix_mode("_MAT_", default="suffix"), "suffix")

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
        self.assertEqual(StrUtils.infer_affix_mode("__MAT", delimiter="__"), "suffix")
        self.assertEqual(StrUtils.infer_affix_mode("MAT__", delimiter="__"), "prefix")

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
        self.assertEqual(StrUtils.infer_affix_mode("", default="garbage"), "prefix")

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
        self.assertEqual(
            StrUtils.apply_affix("brick", prefix=prefix, suffix=suffix), "brick_MAT"
        )

        prefix, suffix = StrUtils.split_affix("MAT_", mode="auto")
        self.assertEqual(
            StrUtils.apply_affix("brick", prefix=prefix, suffix=suffix), "MAT_brick"
        )

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


class TestSplitMapSuffix(unittest.TestCase):
    """`MapRegistry.split_map_suffix` — the reversible split `get_base_texture_name`
    and `FileNaming(base_names=True)` both resolve base names through."""

    def setUp(self):
        self.split = MapRegistry().split_map_suffix

    def test_tail_is_the_suffix_plus_any_tile_token(self):
        cases = {
            "rock_Normal": ("rock", "_Normal"),
            "rock_AO": ("rock", "_AO"),
            "rock-basecolor": ("rock", "-basecolor"),
            "rockAO": ("rock", "AO"),  # attached: capital after lowercase
            "rock_ORM.1001": ("rock", "_ORM.1001"),  # tile comes off first
            "rock.1001": ("rock", ".1001"),  # tile alone
        }
        for stem, expected in cases.items():
            with self.subTest(stem=stem):
                self.assertEqual(self.split(stem), expected)

    def test_no_recognized_suffix_leaves_the_name_whole(self):
        for stem in ("rock", "sphere", "char_thigh", "Agilent_E4419B", "im_Height_16"):
            with self.subTest(stem=stem):
                self.assertEqual(self.split(stem), (stem, ""))

    def test_split_is_reversible(self):
        """base + tail is the input verbatim — what lets a caller re-attach it."""
        for stem in ("rock_Normal", "rockAO", "rock_ORM.1001", "char_thigh", ""):
            with self.subTest(stem=stem):
                self.assertEqual("".join(self.split(stem)), stem)

    def test_base_agrees_with_get_base_texture_name(self):
        """The two must not drift: one is implemented on the other."""
        for stem in ("rock_Normal", "rockAO", "rock_ORM.1001", "sphere"):
            with self.subTest(stem=stem):
                self.assertEqual(
                    self.split(stem)[0], ImgUtils.get_base_texture_name(stem + ".png")
                )


class StripAnyAffixTest(unittest.TestCase):
    """``strip_any_affix`` — the table-driven companion to ``strip_known_affix``.

    A type-suffix pass knows the whole vocabulary but not which entry a given
    name is wearing, nor (once a convention can be spelled either way) which END
    it wears it on.
    """

    KNOWN = ["_GEO", "_MAT", "_SG", "_LSG"]

    def test_strips_a_trailing_affix(self):
        self.assertEqual(StrUtils.strip_any_affix("body_GEO", self.KNOWN), "body")

    def test_strips_a_leading_affix(self):
        """The same vocabulary entry, worn on the front."""
        self.assertEqual(StrUtils.strip_any_affix("GEO_body", self.KNOWN), "body")

    def test_longest_first(self):
        """'_SG' is a tail of '_LSG'; testing the short one first eats half."""
        self.assertEqual(StrUtils.strip_any_affix("body_LSG", self.KNOWN), "body")

    def test_exclude_leaves_the_target_in_place(self):
        """An already-correct name must not be churned."""
        self.assertEqual(
            StrUtils.strip_any_affix("body_GEO", self.KNOWN, exclude=["_GEO"]),
            "body_GEO",
        )

    def test_unknown_affix_is_left_alone(self):
        self.assertEqual(StrUtils.strip_any_affix("body_XYZ", self.KNOWN), "body_XYZ")

    def test_boundary_is_respected(self):
        """'GEOMETRY' merely starts with the token — it is not wearing it."""
        self.assertEqual(
            StrUtils.strip_any_affix("GEOMETRY_thing", self.KNOWN), "GEOMETRY_thing"
        )

    def test_anchored_affix_is_not_stripped_mid_name(self):
        self.assertEqual(
            StrUtils.strip_any_affix("body_GEO_02", self.KNOWN), "body_GEO_02"
        )

    def test_one_stops_after_the_first_match(self):
        """One type marker per name — stripping further eats real content."""
        self.assertEqual(
            StrUtils.strip_any_affix("MAT_body_GEO", self.KNOWN), "MAT_body"
        )
        self.assertEqual(
            StrUtils.strip_any_affix("MAT_body_GEO", self.KNOWN, one=False), "body"
        )

    def test_order_is_deterministic_across_processes(self):
        """Regression: ordering a SET by length alone left equal-length tokens
        to set-iteration order, and Python randomizes string hashing per
        process — so which affix got stripped varied between sessions."""
        import subprocess
        import sys

        code = (
            "from pythontk.str_utils._str_utils import StrUtils;"
            "print(StrUtils.strip_any_affix('MAT_body_GEO',"
            "['_GEO','_MAT','_SG','_LSG']))"
        )
        seen = {
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True
            ).stdout.strip()
            for _ in range(3)
        }
        self.assertEqual(len(seen), 1, f"order varied across processes: {seen}")

    def test_empty_entries_are_ignored(self):
        self.assertEqual(StrUtils.strip_any_affix("body", ["", None]), "body")

    def test_round_trips_with_apply_affix(self):
        """Strip-then-apply is what a convention change actually runs."""
        stripped = StrUtils.strip_any_affix("body_GEO", self.KNOWN)
        self.assertEqual(StrUtils.apply_affix(stripped, prefix="GEO_"), "GEO_body")


class StripAnyAffixCaseTest(unittest.TestCase):
    """A type vocabulary is fixed-case; the words a name is made of are not.

    ``strip_any_affix`` feeds a whole convention table (~19 uppercase tokens)
    at one name. Folding case there makes every token match any English word
    that happens to spell it, at either end -- so ordinary node names quietly
    lose real content before the type affix is applied. These pin the words
    that were being eaten.
    """

    # The shipped mayatk/blendertk table, near enough for these.
    KNOWN = [
        "_GEO",
        "_CAM",
        "_SET",
        "_CON",
        "_LAT",
        "_BS",
        "_GRP",
        "_LOC",
        "_JNT",
    ]

    def test_an_english_word_matching_a_token_is_not_stripped(self):
        for name in (
            "security_cam",
            "tile_set",
            "con_rod",
            "bolt_bs",
            "lat_beam",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    StrUtils.strip_any_affix(name, self.KNOWN),
                    name,
                    f"{name!r} lost real name content to a case-folded token",
                )

    def test_the_real_affix_still_strips_from_either_end(self):
        self.assertEqual(StrUtils.strip_any_affix("body_GEO", self.KNOWN), "body")
        self.assertEqual(StrUtils.strip_any_affix("GEO_body", self.KNOWN), "body")

    def test_case_insensitive_is_still_available_opt_in(self):
        self.assertEqual(
            StrUtils.strip_any_affix("body_geo", self.KNOWN, case_sensitive=False),
            "body",
        )
        self.assertEqual(
            StrUtils.strip_any_affix("body_geo", self.KNOWN),
            "body_geo",
        )

    def test_strip_known_affix_default_stays_case_insensitive(self):
        """It is released; its default must not move."""
        self.assertEqual(
            StrUtils.strip_known_affix("Mat_brick", prefix="MAT_"), "brick"
        )
        self.assertEqual(
            StrUtils.strip_known_affix("Mat_brick", prefix="MAT_", case_sensitive=True),
            "Mat_brick",
        )


class ApplyAffixWholeNameTest(unittest.TestCase):
    """The de-duplication strip must never consume the whole name.

    ``apply_affix`` strips a pre-existing copy of the affix so it is idempotent.
    A node whose name IS its affix -- a camera called ``Cam`` under a ``_CAM``
    rule, or a literal ``GEO`` under ``_GEO`` -- stripped to nothing and came
    back as the bare affix with the name gone.

    The guard is on EMPTINESS, not on case: case-folding is what usefully
    normalises a sloppy lowercase affix, so ``body_geo`` still resolves to
    ``body_GEO`` instead of doubling.
    """

    def test_a_name_that_is_its_own_affix_survives(self):
        self.assertEqual(StrUtils.apply_affix("Cam", suffix="_CAM"), "Cam_CAM")
        self.assertEqual(StrUtils.apply_affix("GEO", suffix="_GEO"), "GEO_GEO")
        self.assertEqual(StrUtils.apply_affix("MAT", prefix="MAT_"), "MAT_MAT")

    def test_a_lowercase_affix_is_still_normalised(self):
        """The case-folded strip is load-bearing; the guard must not cost it."""
        self.assertEqual(StrUtils.apply_affix("body_geo", suffix="_GEO"), "body_GEO")
        self.assertEqual(StrUtils.apply_affix("mat_brick", prefix="MAT_"), "MAT_brick")

    def test_it_is_still_idempotent(self):
        for name, kwargs in (
            ("Cam", {"suffix": "_CAM"}),
            ("body_GEO", {"suffix": "_GEO"}),
            ("GEO", {"suffix": "_GEO"}),
        ):
            with self.subTest(name=name):
                once = StrUtils.apply_affix(name, **kwargs)
                self.assertEqual(StrUtils.apply_affix(once, **kwargs), once)

    def test_the_convention_path_keeps_the_name(self):
        from pythontk import NamingConvention

        self.assertEqual(NamingConvention.apply("Cam", "camera"), "Cam_CAM")
        self.assertEqual(NamingConvention.apply("Cube", "mesh"), "Cube_GEO")


class DelimitAffixTest(unittest.TestCase):
    """``delimit_affix`` — the forgiving counterpart to ``apply_affix``.

    ``apply_affix`` concatenates VERBATIM (the separator is part of the affix),
    which is right for an engine and wrong for a field a user types into.
    """

    def test_bare_token_gets_its_delimiter_on_the_selected_side(self):
        self.assertEqual(StrUtils.delimit_affix("MAT", "prefix"), "MAT_")
        self.assertEqual(StrUtils.delimit_affix("MAT", "suffix"), "_MAT")

    def test_already_delimited_is_left_alone_under_auto(self):
        """Under "auto" the delimiter IS the declaration, so it stands."""
        self.assertEqual(StrUtils.delimit_affix("_MAT", "auto"), "_MAT")
        self.assertEqual(StrUtils.delimit_affix("MAT_", "auto"), "MAT_")

    def test_explicit_mode_re_sides_a_carried_delimiter(self):
        """The picker outranks the spelling the field was pre-filled with.

        A convention row shows the suffix spelling (``_GEO``). Flipping its
        picker to Prefix has to mean ``GEO_``: storing ``('_GEO', 'prefix')``
        would apply as ``_GEObody`` -- no separator, affix on the wrong side
        of it -- and that malformed pair persists into the shared convention
        JSON that mayatk, blendertk, uitk and extapps all read.
        """
        self.assertEqual(StrUtils.delimit_affix("_GEO", "prefix"), "GEO_")
        self.assertEqual(StrUtils.delimit_affix("GEO_", "suffix"), "_GEO")
        # ...and the applied result is well-formed on both sides.
        self.assertEqual(
            StrUtils.apply_affix(
                "body", prefix=StrUtils.delimit_affix("_GEO", "prefix")
            ),
            "GEO_body",
        )
        self.assertEqual(
            StrUtils.apply_affix(
                "body", suffix=StrUtils.delimit_affix("GEO_", "suffix")
            ),
            "body_GEO",
        )

    def test_a_bare_delimiter_re_sides_to_nothing(self):
        """Stripping the delimiter can empty the token; do not emit "__"."""
        self.assertEqual(StrUtils.delimit_affix("_", "prefix"), "")
        self.assertEqual(StrUtils.delimit_affix("_", "suffix"), "")

    def test_auto_falls_back_to_suffix(self):
        """Auto by definition could not decide; suffix matches split_affix."""
        self.assertEqual(StrUtils.delimit_affix("MAT", "auto"), "_MAT")

    def test_empty_stays_empty(self):
        self.assertEqual(StrUtils.delimit_affix("", "prefix"), "")
        self.assertEqual(StrUtils.delimit_affix(None, "prefix"), "")

    def test_whitespace_is_trimmed(self):
        self.assertEqual(StrUtils.delimit_affix("  MAT  ", "suffix"), "_MAT")

    def test_empty_delimiter_disables_the_whole_operation(self):
        self.assertEqual(StrUtils.delimit_affix("MAT", "prefix", delimiter=""), "MAT")

    def test_round_trips_into_apply_affix(self):
        """The point of the helper: a typed token lands with a separator."""
        prefix = StrUtils.delimit_affix("MAT", "prefix")
        self.assertEqual(StrUtils.apply_affix("brick", prefix=prefix), "MAT_brick")


if __name__ == "__main__":
    unittest.main()
