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


if __name__ == "__main__":
    unittest.main(exit=False)
