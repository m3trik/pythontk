#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk hierarchy_utils.

Run with:
    python -m pytest test_hierarchy_utils.py -v
    python test_hierarchy_utils.py
"""
import unittest

from pythontk.core_utils.hierarchy_utils.hierarchy_analyzer import (
    HierarchyAnalyzer,
    HierarchyDifference,
    DifferenceType,
)

from conftest import BaseTestCase


class DifferenceTypeTest(BaseTestCase):
    """Tests for DifferenceType enum."""

    def test_enum_values(self):
        """Test that all expected values exist."""
        self.assertEqual(DifferenceType.MISSING.value, "missing")
        self.assertEqual(DifferenceType.EXTRA.value, "extra")
        self.assertEqual(DifferenceType.MODIFIED.value, "modified")
        self.assertEqual(DifferenceType.MOVED.value, "moved")


class HierarchyDifferenceTest(BaseTestCase):
    """Tests for HierarchyDifference dataclass."""

    def test_init(self):
        """Test creating a HierarchyDifference."""
        diff = HierarchyDifference(
            type=DifferenceType.MISSING,
            path="parent|child",
            details={"reason": "Not found"},
        )

        self.assertEqual(diff.type, DifferenceType.MISSING)
        self.assertEqual(diff.path, "parent|child")
        self.assertEqual(diff.details, {"reason": "Not found"})

    def test_str(self):
        """Test string representation."""
        diff = HierarchyDifference(
            type=DifferenceType.EXTRA,
            path="root|item",
            details={"reason": "Unexpected"},
        )

        str_repr = str(diff)
        self.assertIn("extra", str_repr)
        self.assertIn("root|item", str_repr)


class HierarchyAnalyzerTest(BaseTestCase):
    """Tests for HierarchyAnalyzer class."""

    def test_compare_path_sets(self):
        """Test comparing path sets."""
        current = {"a|b", "a|c", "a|d"}
        reference = {"a|b", "a|c", "a|e"}

        result = HierarchyAnalyzer.compare_path_sets(current, reference)

        self.assertEqual(result["missing"], {"a|e"})
        self.assertEqual(result["extra"], {"a|d"})
        self.assertEqual(result["common"], {"a|b", "a|c"})

    def test_compare_path_sets_identical(self):
        """Test comparing identical path sets."""
        paths = {"a|b", "a|c"}
        result = HierarchyAnalyzer.compare_path_sets(paths, paths)

        self.assertEqual(result["missing"], set())
        self.assertEqual(result["extra"], set())
        self.assertEqual(result["common"], paths)

    def test_analyze_hierarchy_differences_missing(self):
        """Test finding missing items."""
        current_items = [{"path": "a|b"}, {"path": "a|c"}]
        reference_items = [{"path": "a|b"}, {"path": "a|c"}, {"path": "a|d"}]

        diffs = HierarchyAnalyzer.analyze_hierarchy_differences(
            current_items,
            reference_items,
            path_extractor=lambda x: x["path"],
        )

        missing_diffs = [d for d in diffs if d.type == DifferenceType.MISSING]
        self.assertEqual(len(missing_diffs), 1)
        self.assertEqual(missing_diffs[0].path, "a|d")

    def test_analyze_hierarchy_differences_extra(self):
        """Test finding extra items."""
        current_items = [{"path": "a|b"}, {"path": "a|c"}, {"path": "a|d"}]
        reference_items = [{"path": "a|b"}, {"path": "a|c"}]

        diffs = HierarchyAnalyzer.analyze_hierarchy_differences(
            current_items,
            reference_items,
            path_extractor=lambda x: x["path"],
        )

        extra_diffs = [d for d in diffs if d.type == DifferenceType.EXTRA]
        self.assertEqual(len(extra_diffs), 1)
        self.assertEqual(extra_diffs[0].path, "a|d")

    def test_analyze_hierarchy_differences_with_attributes(self):
        """Test finding modified items via attribute comparison."""
        current_items = [{"path": "a|b", "value": 10}]
        reference_items = [{"path": "a|b", "value": 20}]

        diffs = HierarchyAnalyzer.analyze_hierarchy_differences(
            current_items,
            reference_items,
            path_extractor=lambda x: x["path"],
            attribute_extractors={"value": lambda x: x["value"]},
        )

        modified_diffs = [d for d in diffs if d.type == DifferenceType.MODIFIED]
        self.assertEqual(len(modified_diffs), 1)
        self.assertIn("value", modified_diffs[0].details["attribute_changes"])

    def test_detect_moved_items(self):
        """Test detecting moved items."""
        missing = HierarchyDifference(
            type=DifferenceType.MISSING,
            path="parent1|item",
            details={},
        )
        extra = HierarchyDifference(
            type=DifferenceType.EXTRA,
            path="parent2|item",
            details={},
        )

        moves = HierarchyAnalyzer.detect_moved_items(
            [missing, extra], similarity_threshold=0.8
        )

        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].type, DifferenceType.MOVED)

    def test_categorize_differences_by_type(self):
        """Test categorizing differences by type."""
        diffs = [
            HierarchyDifference(DifferenceType.MISSING, "a|b", {}),
            HierarchyDifference(DifferenceType.MISSING, "a|c", {}),
            HierarchyDifference(DifferenceType.EXTRA, "a|d", {}),
        ]

        categorized = HierarchyAnalyzer.categorize_differences(diffs)

        self.assertEqual(len(categorized["by_type"]["missing"]), 2)
        self.assertEqual(len(categorized["by_type"]["extra"]), 1)

    def test_categorize_differences_by_level(self):
        """Test categorizing differences by level."""
        diffs = [
            HierarchyDifference(DifferenceType.MISSING, "a", {}),
            HierarchyDifference(DifferenceType.MISSING, "a|b", {}),
            HierarchyDifference(DifferenceType.MISSING, "a|b|c", {}),
        ]

        categorized = HierarchyAnalyzer.categorize_differences(diffs)

        self.assertEqual(len(categorized["by_level"][1]), 1)
        self.assertEqual(len(categorized["by_level"][2]), 1)
        self.assertEqual(len(categorized["by_level"][3]), 1)

    def test_generate_diff_report_no_differences(self):
        """Test report generation with no differences."""
        report = HierarchyAnalyzer.generate_diff_report([])
        self.assertIn("No differences found", report)

    def test_generate_diff_report_with_differences(self):
        """Test report generation with differences."""
        diffs = [
            HierarchyDifference(DifferenceType.MISSING, "a|b", {"reason": "test"}),
        ]

        report = HierarchyAnalyzer.generate_diff_report(diffs)

        self.assertIn("HIERARCHY DIFFERENCE ANALYSIS REPORT", report)
        self.assertIn("1", report)  # Total count
        self.assertIn("MISSING", report)

    def test_export_differences_to_dict(self):
        """Test exporting differences to dictionary."""
        diffs = [
            HierarchyDifference(DifferenceType.MISSING, "a|b", {"reason": "test"}),
            HierarchyDifference(DifferenceType.EXTRA, "a|c", {}),
        ]

        result = HierarchyAnalyzer.export_differences_to_dict(diffs)

        self.assertEqual(result["total_count"], 2)
        self.assertEqual(len(result["differences"]), 2)
        self.assertEqual(result["summary"]["missing"], 1)
        self.assertEqual(result["summary"]["extra"], 1)

    def test_filter_differences_by_type(self):
        """Test filtering differences by type."""
        diffs = [
            HierarchyDifference(DifferenceType.MISSING, "a|b", {}),
            HierarchyDifference(DifferenceType.EXTRA, "a|c", {}),
        ]

        filtered = HierarchyAnalyzer.filter_differences(
            diffs, types=[DifferenceType.MISSING]
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].type, DifferenceType.MISSING)

    def test_filter_differences_by_pattern(self):
        """Test filtering differences by path pattern."""
        diffs = [
            HierarchyDifference(DifferenceType.MISSING, "group1|mesh", {}),
            HierarchyDifference(DifferenceType.MISSING, "group2|cube", {}),
        ]

        filtered = HierarchyAnalyzer.filter_differences(
            diffs, path_patterns=["group1*"]
        )

        self.assertEqual(len(filtered), 1)
        self.assertIn("group1", filtered[0].path)

    def test_filter_differences_exclude_pattern(self):
        """Test filtering differences with exclusion pattern."""
        diffs = [
            HierarchyDifference(DifferenceType.MISSING, "group1|mesh", {}),
            HierarchyDifference(DifferenceType.MISSING, "group2|cube", {}),
        ]

        filtered = HierarchyAnalyzer.filter_differences(
            diffs, exclude_patterns=["group1*"]
        )

        self.assertEqual(len(filtered), 1)
        self.assertIn("group2", filtered[0].path)


if __name__ == "__main__":
    unittest.main(exit=False)
