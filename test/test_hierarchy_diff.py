#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk HierarchyDiff.

Run with:
    python -m pytest test_hierarchy_diff.py -v
    python test_hierarchy_diff.py
"""
import json
import os
import tempfile
import unittest

from pythontk.core_utils.hierarchy_diff import HierarchyDiff

from conftest import BaseTestCase


class HierarchyDiffTest(BaseTestCase):
    """HierarchyDiff test class."""

    def test_init_empty(self):
        """Test that HierarchyDiff initializes with empty lists."""
        diff = HierarchyDiff()
        self.assertEqual(diff.missing, [])
        self.assertEqual(diff.extra, [])
        self.assertEqual(diff.renamed, [])
        self.assertEqual(diff.reparented, [])
        self.assertEqual(diff.fuzzy_matches, [])
        self.assertEqual(diff.metadata, {})

    def test_is_valid_when_empty(self):
        """Test is_valid returns True for empty diff."""
        diff = HierarchyDiff()
        self.assertTrue(diff.is_valid())

    def test_is_valid_with_missing(self):
        """Test is_valid returns False when items are missing."""
        diff = HierarchyDiff()
        diff.missing = ["item1"]
        self.assertFalse(diff.is_valid())

    def test_is_valid_with_extra_only(self):
        """Test is_valid returns True when only extra items exist."""
        diff = HierarchyDiff()
        diff.extra = ["item1"]
        self.assertTrue(diff.is_valid())  # Extra items don't affect validity

    def test_has_differences_empty(self):
        """Test has_differences returns False for empty diff."""
        diff = HierarchyDiff()
        self.assertFalse(diff.has_differences())

    def test_has_differences_with_extra(self):
        """Test has_differences returns True when extra items exist."""
        diff = HierarchyDiff()
        diff.extra = ["item1"]
        self.assertTrue(diff.has_differences())

    def test_total_issues(self):
        """Test total_issues counts all difference types."""
        diff = HierarchyDiff()
        diff.missing = ["a", "b"]
        diff.extra = ["c"]
        diff.renamed = ["d", "e", "f"]
        diff.reparented = ["g"]
        self.assertEqual(diff.total_issues(), 7)

    def test_as_dict(self):
        """Test as_dict returns correct dictionary structure."""
        diff = HierarchyDiff()
        diff.missing = ["item1"]
        diff.extra = ["item2"]
        diff.metadata = {"key": "value"}

        result = diff.as_dict()
        self.assertIn("missing", result)
        self.assertIn("extra", result)
        self.assertIn("metadata", result)
        self.assertEqual(result["missing"], ["item1"])
        self.assertEqual(result["extra"], ["item2"])
        self.assertEqual(result["metadata"], {"key": "value"})

    def test_as_json(self):
        """Test as_json returns valid JSON string."""
        diff = HierarchyDiff()
        diff.missing = ["item1"]

        json_str = diff.as_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["missing"], ["item1"])

    def test_save_and_load_file(self):
        """Test saving and loading from file."""
        diff = HierarchyDiff()
        diff.missing = ["item1", "item2"]
        diff.extra = ["item3"]
        diff.metadata = {"test": "value"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            diff.save_to_file(filepath)
            loaded = HierarchyDiff.load_from_file(filepath)

            self.assertEqual(loaded.missing, diff.missing)
            self.assertEqual(loaded.extra, diff.extra)
            self.assertEqual(loaded.metadata, diff.metadata)
        finally:
            os.unlink(filepath)

    def test_clear(self):
        """Test clear removes all data."""
        diff = HierarchyDiff()
        diff.missing = ["item1"]
        diff.extra = ["item2"]
        diff.metadata = {"key": "value"}

        diff.clear()

        self.assertEqual(diff.missing, [])
        self.assertEqual(diff.extra, [])
        self.assertEqual(diff.metadata, {})

    def test_merge(self):
        """Test merge combines two diffs."""
        diff1 = HierarchyDiff()
        diff1.missing = ["a", "b"]
        diff1.metadata = {"key1": "value1"}

        diff2 = HierarchyDiff()
        diff2.missing = ["c"]
        diff2.extra = ["d"]
        diff2.metadata = {"key2": "value2"}

        diff1.merge(diff2)

        self.assertEqual(diff1.missing, ["a", "b", "c"])
        self.assertEqual(diff1.extra, ["d"])
        self.assertEqual(diff1.metadata, {"key1": "value1", "key2": "value2"})

    def test_get_summary(self):
        """Test get_summary returns correct counts."""
        diff = HierarchyDiff()
        diff.missing = ["a", "b"]
        diff.extra = ["c"]
        diff.renamed = ["d"]

        summary = diff.get_summary()
        self.assertEqual(summary["missing"], 2)
        self.assertEqual(summary["extra"], 1)
        self.assertEqual(summary["renamed"], 1)
        self.assertEqual(summary["total_issues"], 4)

    def test_filter_by_pattern(self):
        """Test filter_by_pattern filters with regex."""
        diff = HierarchyDiff()
        diff.missing = ["apple", "banana", "apricot", "cherry"]

        result = diff.filter_by_pattern(r"^ap", "missing")
        self.assertEqual(result, ["apple", "apricot"])

    def test_add_metadata(self):
        """Test add_metadata adds key-value pairs."""
        diff = HierarchyDiff()
        diff.add_metadata("timestamp", "2024-01-01")
        diff.add_metadata("source", "test")

        self.assertEqual(diff.metadata["timestamp"], "2024-01-01")
        self.assertEqual(diff.metadata["source"], "test")

    def test_str_representation(self):
        """Test string representation."""
        diff = HierarchyDiff()
        diff.missing = ["a"]
        diff.extra = ["b", "c"]

        str_repr = str(diff)
        self.assertIn("missing=1", str_repr)
        self.assertIn("extra=2", str_repr)

    def test_repr(self):
        """Test repr is same as str."""
        diff = HierarchyDiff()
        self.assertEqual(str(diff), repr(diff))


if __name__ == "__main__":
    unittest.main(exit=False)
