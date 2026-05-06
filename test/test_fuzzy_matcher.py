#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk FuzzyMatcher.

Run with:
    python -m pytest test_fuzzy_matcher.py -v
    python test_fuzzy_matcher.py
"""
import unittest

from pythontk.str_utils.fuzzy_matcher import FuzzyMatcher

from conftest import BaseTestCase


class FuzzyMatcherTest(BaseTestCase):
    """Tests for FuzzyMatcher class."""

    def test_get_base_name_with_digits(self):
        """Test removing trailing digits."""
        self.assertEqual(FuzzyMatcher.get_base_name("object123"), "object")
        self.assertEqual(FuzzyMatcher.get_base_name("mesh_01"), "mesh_")
        self.assertEqual(FuzzyMatcher.get_base_name("cube1"), "cube")

    def test_get_base_name_without_digits(self):
        """Test name without trailing digits."""
        self.assertEqual(FuzzyMatcher.get_base_name("object"), "object")
        self.assertEqual(FuzzyMatcher.get_base_name("mesh_name"), "mesh_name")

    def test_get_base_name_only_digits(self):
        """Test name that is only digits."""
        self.assertEqual(FuzzyMatcher.get_base_name("123"), "")

    def test_find_best_match_exact(self):
        """Test exact match returns 1.0 score."""
        candidates = ["mesh_01", "mesh_02", "cube_01"]
        result = FuzzyMatcher.find_best_match("mesh_01", candidates)

        self.assertEqual(result, ("mesh_01", 1.0))

    def test_find_best_match_fuzzy(self):
        """Test fuzzy match with same base name."""
        candidates = ["mesh_01", "mesh_02", "cube_01"]
        result = FuzzyMatcher.find_best_match("mesh_03", candidates)

        self.assertIsNotNone(result)
        self.assertIn(result[0], ["mesh_01", "mesh_02"])
        self.assertGreaterEqual(result[1], 0.5)

    def test_find_best_match_no_match(self):
        """Test no match when below threshold."""
        candidates = ["mesh_01", "cube_01"]
        result = FuzzyMatcher.find_best_match("xyz", candidates, score_threshold=0.9)

        self.assertIsNone(result)

    def test_find_all_matches(self):
        """Test finding matches for multiple targets."""
        targets = ["mesh_03", "cube_02"]
        candidates = ["mesh_01", "mesh_02", "cube_01", "sphere_01"]

        matches = FuzzyMatcher.find_all_matches(targets, candidates)

        self.assertIn("mesh_03", matches)
        self.assertIn("cube_02", matches)

    def test_find_trailing_digit_matches(self):
        """Test finding trailing digit matches in paths."""
        missing = ["group1|mesh_01", "group2|cube_02"]
        extra = ["group1|mesh_03", "group2|cube_05"]

        matches, missing_remove, extra_remove = (
            FuzzyMatcher.find_trailing_digit_matches(missing, extra)
        )

        self.assertEqual(len(matches), 2)
        self.assertEqual(len(missing_remove), 2)
        self.assertEqual(len(extra_remove), 2)

    def test_find_trailing_digit_matches_different_parents(self):
        """Test that paths with different parents don't match."""
        missing = ["group1|mesh_01"]
        extra = ["group2|mesh_02"]

        matches, _, _ = FuzzyMatcher.find_trailing_digit_matches(missing, extra)

        self.assertEqual(len(matches), 0)

    def test_find_trailing_digit_matches_no_digits(self):
        """Test that names without trailing digits don't match."""
        missing = ["group1|mesh"]  # No trailing digits
        extra = ["group1|mesh_02"]

        matches, _, _ = FuzzyMatcher.find_trailing_digit_matches(missing, extra)

        self.assertEqual(len(matches), 0)

    def test_calculate_similarity_identical(self):
        """Test similarity of identical names."""
        score = FuzzyMatcher._calculate_similarity("mesh", "mesh")
        self.assertEqual(score, 1.0)

    def test_calculate_similarity_same_base(self):
        """Test similarity with same base name."""
        score = FuzzyMatcher._calculate_similarity("mesh_01", "mesh_02")
        self.assertEqual(score, 0.9)

    def test_calculate_similarity_substring(self):
        """Test similarity with substring match."""
        score = FuzzyMatcher._calculate_similarity("mesh", "mesh_complete")
        self.assertGreater(score, 0.0)

    def test_calculate_levenshtein_distance_same(self):
        """Test Levenshtein distance for identical strings."""
        distance = FuzzyMatcher.calculate_levenshtein_distance("hello", "hello")
        self.assertEqual(distance, 0)

    def test_calculate_levenshtein_distance_one_change(self):
        """Test Levenshtein distance for one character change."""
        distance = FuzzyMatcher.calculate_levenshtein_distance("hello", "hallo")
        self.assertEqual(distance, 1)

    def test_calculate_levenshtein_distance_different(self):
        """Test Levenshtein distance for different strings."""
        distance = FuzzyMatcher.calculate_levenshtein_distance("abc", "xyz")
        self.assertEqual(distance, 3)

    def test_calculate_levenshtein_distance_empty(self):
        """Test Levenshtein distance with empty string."""
        distance = FuzzyMatcher.calculate_levenshtein_distance("hello", "")
        self.assertEqual(distance, 5)

    def test_similarity_from_distance_identical(self):
        """Test similarity from distance for identical strings."""
        score = FuzzyMatcher.similarity_from_distance("hello", "hello")
        self.assertEqual(score, 1.0)

    def test_similarity_from_distance_one_change(self):
        """Test similarity from distance for one character difference."""
        score = FuzzyMatcher.similarity_from_distance("hello", "hallo")
        self.assertEqual(score, 0.8)  # 1 - (1/5)

    def test_similarity_from_distance_both_empty(self):
        """Test similarity when both strings are empty."""
        score = FuzzyMatcher.similarity_from_distance("", "")
        self.assertEqual(score, 1.0)

    # --- find_unique_match -------------------------------------------------

    def test_find_unique_match_exact(self):
        """Exact target presence returns ('unique', 1.0)."""
        name, score, status = FuzzyMatcher.find_unique_match(
            "wood", ["wood", "stone"]
        )
        self.assertEqual((name, score, status), ("wood", 1.0, "unique"))

    def test_find_unique_match_no_match(self):
        """No candidate above threshold returns no_match."""
        name, score, status = FuzzyMatcher.find_unique_match(
            "xyz", ["aaa", "bbb"], score_threshold=0.5
        )
        self.assertEqual((name, score, status), (None, 0.0, "no_match"))

    def test_find_unique_match_substring_unique(self):
        """Substring containment with one clear winner returns unique."""
        # The texture-editor scenario: missing 'foo_ao', candidate set has the
        # 'demo_' prefixed version plus sibling textures with different suffixes.
        # use_base_name=False so trailing-digit fusion never fires.
        candidates = [
            "demo_foo_ao",
            "demo_foo_diff",
            "demo_foo_norm",
            "demo_foo_spec",
        ]
        name, score, status = FuzzyMatcher.find_unique_match(
            "foo_ao",
            candidates,
            score_threshold=0.5,
            use_base_name=False,
        )
        self.assertEqual(status, "unique")
        self.assertEqual(name, "demo_foo_ao")
        self.assertGreater(score, 0.5)

    def test_find_unique_match_ambiguous(self):
        """When two candidates score within ambiguity_delta, status is ambiguous."""
        # Both contain 'wood' as a substring with the same length difference.
        name, score, status = FuzzyMatcher.find_unique_match(
            "wood",
            ["wood_a", "wood_b"],
            score_threshold=0.5,
            use_base_name=False,
        )
        self.assertEqual(status, "ambiguous")
        self.assertIn(name, ["wood_a", "wood_b"])

    def test_find_unique_match_disable_base_name(self):
        """use_base_name=False stops trailing-digit fusion."""
        # With base-name on, 'tx_001' and 'tx_002' share base 'tx_' → score 0.9.
        # With it off (and substring/prefix off to isolate), the score is 0.0.
        score_on = FuzzyMatcher._calculate_similarity("tx_001", "tx_002")
        score_off = FuzzyMatcher._calculate_similarity(
            "tx_001",
            "tx_002",
            use_base_name=False,
            use_substring=False,
            use_prefix=False,
        )
        self.assertGreaterEqual(score_on, 0.9)
        self.assertEqual(score_off, 0.0)

    def test_calculate_similarity_strategy_flags(self):
        """Disabling strategies shrinks the set of pairs that score above 0."""
        # Substring path:
        s_on = FuzzyMatcher._calculate_similarity("foo", "foobar")
        s_off = FuzzyMatcher._calculate_similarity(
            "foo", "foobar", use_substring=False, use_prefix=False
        )
        self.assertGreater(s_on, 0)
        self.assertEqual(s_off, 0.0)

    def test_calculate_similarity_ratio_off_by_default(self):
        """use_ratio defaults to False so it doesn't change legacy callers."""
        # 'cat'/'dog' have no shared structure but difflib gives a small ratio.
        s_default = FuzzyMatcher._calculate_similarity("cat", "dog")
        s_ratio = FuzzyMatcher._calculate_similarity(
            "cat",
            "dog",
            use_base_name=False,
            use_substring=False,
            use_prefix=False,
            use_ratio=True,
        )
        self.assertEqual(s_default, 0.0)
        # difflib returns 0.0 here too (no matching chars), but the toggle shouldn't error.
        self.assertGreaterEqual(s_ratio, 0.0)

    # --- find_with_fallbacks -----------------------------------------------

    def test_find_with_fallbacks_returns_first_unique(self):
        """Pipeline returns the first strategy that produces a unique match."""
        name, score, status, strat = FuzzyMatcher.find_with_fallbacks(
            "wood", ["wood", "stone"], strategies=["exact", "substring"]
        )
        self.assertEqual((name, status, strat), ("wood", "unique", "exact"))

    def test_find_with_fallbacks_falls_through_no_match(self):
        """No-match advances; later strategy resolves."""
        # 'foo_ao' isn't in candidates → exact misses.
        # substring matches 'demo_foo_ao' uniquely.
        name, score, status, strat = FuzzyMatcher.find_with_fallbacks(
            "foo_ao",
            ["demo_foo_ao", "demo_foo_diff"],
            strategies=["exact", "substring"],
            score_threshold=0.4,
        )
        self.assertEqual((name, status, strat), ("demo_foo_ao", "unique", "substring"))

    def test_find_with_fallbacks_stops_on_ambiguous(self):
        """Default behavior: ambiguous tier halts the pipeline."""
        name, score, status, strat = FuzzyMatcher.find_with_fallbacks(
            "wood",
            ["wood_a", "wood_b"],
            strategies=["substring", "ratio"],
            score_threshold=0.5,
        )
        self.assertEqual(status, "ambiguous")
        self.assertEqual(strat, "substring")

    def test_find_with_fallbacks_continues_past_ambiguous(self):
        """stop_on_ambiguous=False lets later tiers run after an ambiguous tier."""

        sentinel = []

        def custom_after(target, candidates):
            sentinel.append("ran")
            return "z", 1.0, "unique"

        # 'wood' has two equally-good substring matches → ambiguous.
        # With stop_on_ambiguous=False, the next tier should still execute.
        name, score, status, strat = FuzzyMatcher.find_with_fallbacks(
            "wood",
            ["wood_a", "wood_b"],
            strategies=["substring", custom_after],
            score_threshold=0.5,
            stop_on_ambiguous=False,
        )
        self.assertEqual(sentinel, ["ran"])
        self.assertEqual((name, status, strat), ("z", "unique", "custom_after"))

    def test_find_with_fallbacks_all_no_match(self):
        """All strategies return no_match → final status is no_match."""
        name, score, status, strat = FuzzyMatcher.find_with_fallbacks(
            "xyz",
            ["aaa", "bbb"],
            strategies=["exact", "substring"],
            score_threshold=0.9,
        )
        self.assertEqual((name, status, strat), (None, "no_match", ""))

    def test_find_with_fallbacks_custom_callable(self):
        """Custom callable strategies plug in alongside built-ins."""

        def texture_aware(target, candidates):
            # Trivial example: prefer candidate ending with same suffix.
            suffix = target.rsplit("_", 1)[-1]
            same = [c for c in candidates if c.endswith("_" + suffix)]
            if len(same) == 1:
                return same[0], 1.0, "unique"
            return None, 0.0, "no_match"

        name, score, status, strat = FuzzyMatcher.find_with_fallbacks(
            "x_ao",
            ["demo_x_ao", "demo_x_diff"],
            strategies=[texture_aware, "substring"],
        )
        self.assertEqual((name, status, strat), ("demo_x_ao", "unique", "texture_aware"))

    def test_find_with_fallbacks_unknown_strategy_name(self):
        """Unknown built-in strategy name raises ValueError when reached."""
        # 'exact' misses (target not in candidates), so the pipeline advances and
        # tries 'made_up' which should raise.
        with self.assertRaises(ValueError):
            FuzzyMatcher.find_with_fallbacks(
                "x", ["a", "b"], strategies=["exact", "made_up"]
            )


if __name__ == "__main__":
    unittest.main(exit=False)
