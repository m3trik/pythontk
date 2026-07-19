#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk HierarchyMatching.

Run with:
    python -m pytest test_hierarchy_matching.py -v
    python test_hierarchy_matching.py
"""
import unittest

from pythontk.core_utils.hierarchy_utils.hierarchy_matching import HierarchyMatching

from conftest import BaseTestCase


class HierarchyMatchingTest(BaseTestCase):
    """Tests for HierarchyMatching strategies.

    Items are plain path strings — matching uses source items as dict
    keys, so items must be hashable.
    """

    @staticmethod
    def _path(item):
        return item

    @staticmethod
    def _name(item):
        return item.split("|")[-1]

    def test_exact_path_match_ignores_namespaces(self):
        matches = HierarchyMatching.exact_path_match(
            ["ns:grp|ns:child"], ["grp|child", "grp|other"], self._path
        )
        self.assertEqual(matches["ns:grp|ns:child"], ["grp|child"])

    def test_tail_path_match(self):
        matches = HierarchyMatching.tail_path_match(
            ["oldroot|grp|child"],
            ["newroot|grp|child", "newroot|grp|other"],
            self._path,
            num_components=2,
        )
        self.assertEqual(matches["oldroot|grp|child"], ["newroot|grp|child"])

    def test_fuzzy_name_match_threshold(self):
        matches = HierarchyMatching.fuzzy_name_match(
            ["grp|item1"],
            ["x|item12", "x|unrelated"],
            self._name,
            similarity_threshold=0.8,
        )
        self.assertEqual(matches["grp|item1"], ["x|item12"])

        none = HierarchyMatching.fuzzy_name_match(
            ["grp|item1"],
            ["x|item12", "x|unrelated"],
            self._name,
            similarity_threshold=0.99,
        )
        self.assertEqual(none, {})

    def test_multi_strategy_prefers_earlier_strategy(self):
        matches = HierarchyMatching.multi_strategy_match(
            ["grp|child", "lost|node99"],
            ["grp|child", "found|node91"],
            self._path,
            get_name_func=self._name,
            fuzzy_threshold=0.7,
        )
        # First source matched exactly; second only via fuzzy fallback.
        self.assertEqual(matches["grp|child"], ["grp|child"])
        self.assertEqual(matches["lost|node99"], ["found|node91"])

    def test_multi_strategy_skips_fuzzy_without_name_func(self):
        matches = HierarchyMatching.multi_strategy_match(
            ["lost|node99"], ["found|node91"], self._path
        )
        self.assertEqual(matches, {})

    def test_multi_strategy_accepts_callable(self):
        def match_everything(items, targets):
            return {item: list(targets) for item in items}

        matches = HierarchyMatching.multi_strategy_match(
            ["a|b"], ["z|z"], self._path, strategies=[match_everything]
        )
        self.assertEqual(matches["a|b"], ["z|z"])

    def test_multi_strategy_unknown_name_skipped(self):
        matches = HierarchyMatching.multi_strategy_match(
            ["a|b"], ["a|b"], self._path, strategies=["bogus", "exact_path"]
        )
        self.assertEqual(matches["a|b"], ["a|b"])

    def test_deprecated_private_alias_still_works(self):
        """Released mayatk builds call this — keep it delegating."""
        self.assertEqual(HierarchyMatching._clean_namespace("ns:x"), "x")


if __name__ == "__main__":
    unittest.main(exit=False)
