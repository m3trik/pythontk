#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk HierarchyIndexer.

Run with:
    python -m pytest test_hierarchy_indexer.py -v
    python test_hierarchy_indexer.py
"""
import unittest

from pythontk.core_utils.hierarchy_utils.hierarchy_indexer import HierarchyIndexer

from conftest import BaseTestCase


class HierarchyIndexerTest(BaseTestCase):
    """Tests for HierarchyIndexer."""

    ITEMS = [
        {"path": "ns:grp|ns:child"},
        {"path": "grp|child"},
        {"path": "grp|other"},
        {"path": ""},
    ]

    @staticmethod
    def _path(item):
        return item["path"]

    def test_build_path_index_groups_namespace_variants(self):
        index = HierarchyIndexer.build_path_index(self.ITEMS, self._path)
        self.assertEqual(len(index["grp|child"]), 2)
        self.assertEqual(len(index["grp|other"]), 1)
        self.assertNotIn("", index)  # empty paths are skipped

    def test_build_path_index_raw(self):
        index = HierarchyIndexer.build_path_index(
            self.ITEMS, self._path, clean_namespaces=False
        )
        self.assertIn("ns:grp|ns:child", index)
        self.assertEqual(len(index["grp|child"]), 1)

    def test_find_by_path_normalizes_target(self):
        index = HierarchyIndexer.build_path_index(self.ITEMS, self._path)
        matches = HierarchyIndexer.find_by_path(index, "other_ns:grp|other_ns:child")
        self.assertEqual(len(matches), 2)
        self.assertEqual(HierarchyIndexer.find_by_path(index, "nope"), [])

    def test_find_by_tail_path(self):
        index = HierarchyIndexer.build_path_index(
            [{"path": "root|grp|child"}, {"path": "other|grp|child"}], self._path
        )
        matches = HierarchyIndexer.find_by_tail_path(index, "grp|child", 2)
        self.assertEqual(len(matches), 2)
        self.assertEqual(HierarchyIndexer.find_by_tail_path(index, "x|child", 2), [])

    def test_get_path_components_index(self):
        index = HierarchyIndexer.get_path_components_index(
            [{"path": "ns:grp|child"}, {"path": "grp|other"}], self._path
        )
        self.assertEqual(len(index["grp"]), 2)  # namespace-cleaned + deduped
        self.assertEqual(len(index["child"]), 1)

    def test_get_depth_index(self):
        index = HierarchyIndexer.get_depth_index(
            [{"path": "a"}, {"path": "a|b"}, {"path": "c|d"}], self._path
        )
        self.assertEqual(len(index[1]), 1)
        self.assertEqual(len(index[2]), 2)

    def test_deprecated_private_aliases_still_work(self):
        """Released mayatk builds call these — keep them delegating."""
        self.assertEqual(HierarchyIndexer._clean_namespace("ns:x"), "x")
        self.assertEqual(HierarchyIndexer._join_hierarchy_path(["a", "b"]), "a|b")
        self.assertEqual(HierarchyIndexer._split_hierarchy_path("a|b"), ["a", "b"])
        self.assertEqual(HierarchyIndexer._clean_hierarchy_path("ns:a|b"), "a|b")
        self.assertEqual(HierarchyIndexer._normalize_path("ns:a|b"), "a|b")


if __name__ == "__main__":
    unittest.main(exit=False)
