# !/usr/bin/python
# coding=utf-8
import fnmatch
from difflib import SequenceMatcher
from typing import Any, Dict, List, Set, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum

from .hierarchy_path import HierarchyPath


class DifferenceType(Enum):
    """Types of differences that can be found between hierarchies."""

    MISSING = "missing"
    EXTRA = "extra"
    MODIFIED = "modified"
    MOVED = "moved"


@dataclass
class HierarchyDifference:
    """Represents a single difference between hierarchies."""

    type: DifferenceType
    path: str
    details: Dict[str, Any] = field(default_factory=dict)
    item: Any = None

    def __str__(self) -> str:
        return f"{self.type.value}: {self.path} - {self.details}"


class HierarchyAnalyzer:
    """
    Analyzer for comparing hierarchical structures and identifying differences.

    This class provides comprehensive analysis capabilities for comparing
    two hierarchical structures and identifying various types of differences.
    """

    @staticmethod
    def compare_path_sets(
        current_paths: Set[str], reference_paths: Set[str]
    ) -> Dict[str, Set[str]]:
        """
        Compare two sets of hierarchical paths and categorize differences.

        Args:
            current_paths: Set of paths in current hierarchy
            reference_paths: Set of paths in reference hierarchy

        Returns:
            Dictionary with 'missing', 'extra', and 'common' path sets
        """
        return {
            "missing": reference_paths - current_paths,
            "extra": current_paths - reference_paths,
            "common": current_paths & reference_paths,
        }

    @staticmethod
    def analyze_hierarchy_differences(
        current_items: List[Any],
        reference_items: List[Any],
        path_extractor: Callable[[Any], str],
        attribute_extractors: Optional[Dict[str, Callable[[Any], Any]]] = None,
    ) -> List[HierarchyDifference]:
        """
        Perform comprehensive analysis of differences between hierarchies.

        Args:
            current_items: Items in current hierarchy
            reference_items: Items in reference hierarchy
            path_extractor: Function to extract path from item
            attribute_extractors: Functions to extract attributes for comparison

        Returns:
            List of detected differences
        """
        if attribute_extractors is None:
            attribute_extractors = {}

        differences = []

        # Build path mappings
        current_map = {path_extractor(item): item for item in current_items}
        reference_map = {path_extractor(item): item for item in reference_items}

        path_comparison = HierarchyAnalyzer.compare_path_sets(
            set(current_map), set(reference_map)
        )

        # Missing items (sorted — set iteration order is not deterministic)
        for missing_path in sorted(path_comparison["missing"]):
            differences.append(
                HierarchyDifference(
                    type=DifferenceType.MISSING,
                    path=missing_path,
                    details={"reason": "Path exists in reference but not in current"},
                    item=reference_map[missing_path],
                )
            )

        # Extra items
        for extra_path in sorted(path_comparison["extra"]):
            differences.append(
                HierarchyDifference(
                    type=DifferenceType.EXTRA,
                    path=extra_path,
                    details={"reason": "Path exists in current but not in reference"},
                    item=current_map[extra_path],
                )
            )

        # Compare attributes of common items
        for common_path in sorted(path_comparison["common"]):
            current_item = current_map[common_path]
            reference_item = reference_map[common_path]

            attribute_diffs = HierarchyAnalyzer._compare_item_attributes(
                current_item, reference_item, attribute_extractors
            )

            if attribute_diffs:
                differences.append(
                    HierarchyDifference(
                        type=DifferenceType.MODIFIED,
                        path=common_path,
                        details={"attribute_changes": attribute_diffs},
                        item=current_item,
                    )
                )

        return differences

    @staticmethod
    def _compare_item_attributes(
        current_item: Any,
        reference_item: Any,
        attribute_extractors: Dict[str, Callable[[Any], Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare attributes of two items.

        Args:
            current_item: Item from current hierarchy
            reference_item: Item from reference hierarchy
            attribute_extractors: Functions to extract attributes

        Returns:
            Dictionary of attribute differences
        """
        differences = {}

        for attr_name, extractor in attribute_extractors.items():
            try:
                current_value = extractor(current_item)
                reference_value = extractor(reference_item)

                if current_value != reference_value:
                    differences[attr_name] = {
                        "current": current_value,
                        "reference": reference_value,
                    }
            except Exception as e:
                differences[attr_name] = {
                    "error": f"Failed to compare attribute: {str(e)}"
                }

        return differences

    @staticmethod
    def detect_moved_items(
        differences: List[HierarchyDifference],
        similarity_threshold: float = 0.8,
        path_separator: str = "|",
    ) -> List[HierarchyDifference]:
        """
        Detect items that may have been moved rather than deleted/added.

        Pairs MISSING with EXTRA differences by leaf-name similarity.
        Assignment is one-to-one and globally greedy: the most similar
        pair claims first, with path-text tiebreaks, so the result is
        deterministic regardless of input order. The input list is not
        modified — callers that want a refined view can drop the MISSING/
        EXTRA entries whose paths appear as a move's ``from_path`` /
        ``to_path`` (or use :meth:`HierarchyDiff.from_differences`).

        Args:
            differences: List of differences from analyze_hierarchy_differences
            similarity_threshold: Minimum similarity to consider items as moved
            path_separator: Character separating path components

        Returns:
            List of move differences, best match first
        """
        missing_diffs = [d for d in differences if d.type == DifferenceType.MISSING]
        extra_diffs = [d for d in differences if d.type == DifferenceType.EXTRA]
        if not missing_diffs or not extra_diffs:
            return []

        # Score every missing/extra leaf-name pairing above the threshold.
        candidates = []
        extra_names = [
            HierarchyPath.leaf(extra.path, path_separator) for extra in extra_diffs
        ]
        for m_idx, missing_diff in enumerate(missing_diffs):
            missing_name = HierarchyPath.leaf(missing_diff.path, path_separator)
            for e_idx, extra_name in enumerate(extra_names):
                similarity = SequenceMatcher(None, missing_name, extra_name).ratio()
                if similarity >= similarity_threshold:
                    candidates.append((similarity, m_idx, e_idx))

        candidates.sort(
            key=lambda c: (-c[0], missing_diffs[c[1]].path, extra_diffs[c[2]].path)
        )

        move_differences = []
        used_missing: Set[int] = set()
        used_extra: Set[int] = set()
        for similarity, m_idx, e_idx in candidates:
            if m_idx in used_missing or e_idx in used_extra:
                continue
            used_missing.add(m_idx)
            used_extra.add(e_idx)

            missing_diff = missing_diffs[m_idx]
            move_differences.append(
                HierarchyDifference(
                    type=DifferenceType.MOVED,
                    path=missing_diff.path,
                    details={
                        "from_path": missing_diff.path,
                        "to_path": extra_diffs[e_idx].path,
                        "similarity": similarity,
                    },
                    item=missing_diff.item,
                )
            )

        return move_differences

    @staticmethod
    def categorize_differences(
        differences: List[HierarchyDifference], path_separator: str = "|"
    ) -> Dict[str, Dict[str, List[HierarchyDifference]]]:
        """
        Categorize differences by type and hierarchy level.

        Args:
            differences: List of differences to categorize
            path_separator: Character separating path components

        Returns:
            Nested dictionary organizing differences by type and level
        """
        categorized = {"by_type": {}, "by_level": {}, "by_root": {}}

        for diff in differences:
            categorized["by_type"].setdefault(diff.type.value, []).append(diff)

            level = HierarchyPath.depth(diff.path, path_separator)
            categorized["by_level"].setdefault(level, []).append(diff)

            root = (
                HierarchyPath.root(diff.path, path_separator)
                if diff.path
                else "unknown"
            )
            categorized["by_root"].setdefault(root, []).append(diff)

        return categorized

    @staticmethod
    def generate_diff_report(
        differences: List[HierarchyDifference],
        include_details: bool = True,
        max_items_per_category: int = 10,
    ) -> str:
        """
        Generate a human-readable report of differences.

        Args:
            differences: List of differences to report
            include_details: Whether to include detailed information
            max_items_per_category: Maximum items to show per category

        Returns:
            Formatted difference report string
        """
        if not differences:
            return "No differences found between hierarchies."

        categorized = HierarchyAnalyzer.categorize_differences(differences)

        report_lines = [
            "=" * 60,
            "HIERARCHY DIFFERENCE ANALYSIS REPORT",
            "=" * 60,
            f"Total differences found: {len(differences)}",
            "",
        ]

        # Summary by type
        report_lines.append("SUMMARY BY TYPE:")
        for diff_type, type_diffs in categorized["by_type"].items():
            report_lines.append(f"  {diff_type.upper()}: {len(type_diffs)} items")
        report_lines.append("")

        # Detailed breakdown
        if include_details:
            for diff_type, type_diffs in categorized["by_type"].items():
                report_lines.append(f"{diff_type.upper()} ITEMS:")

                for diff in type_diffs[:max_items_per_category]:
                    report_lines.append(f"  • {diff.path}")
                    for key, value in diff.details.items():
                        report_lines.append(f"    {key}: {value}")

                if len(type_diffs) > max_items_per_category:
                    remaining = len(type_diffs) - max_items_per_category
                    report_lines.append(f"  ... and {remaining} more items")

                report_lines.append("")

        # Level distribution
        report_lines.append("DISTRIBUTION BY HIERARCHY LEVEL:")
        for level in sorted(categorized["by_level"]):
            report_lines.append(
                f"  Level {level}: {len(categorized['by_level'][level])} items"
            )

        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    @staticmethod
    def export_differences_to_dict(
        differences: List[HierarchyDifference],
    ) -> Dict[str, Any]:
        """
        Export differences to a dictionary format for serialization.

        Args:
            differences: List of differences to export

        Returns:
            Dictionary representation of differences
        """
        return {
            "total_count": len(differences),
            "differences": [
                {"type": diff.type.value, "path": diff.path, "details": diff.details}
                for diff in differences
            ],
            "summary": {
                diff_type.value: len([d for d in differences if d.type == diff_type])
                for diff_type in DifferenceType
            },
        }

    @staticmethod
    def filter_differences(
        differences: List[HierarchyDifference],
        types: Optional[List[DifferenceType]] = None,
        path_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[HierarchyDifference]:
        """
        Filter differences based on type and path patterns.

        Args:
            differences: List of differences to filter
            types: List of difference types to include
            path_patterns: List of path patterns to include (supports wildcards)
            exclude_patterns: List of path patterns to exclude

        Returns:
            Filtered list of differences
        """
        filtered = differences

        if types:
            filtered = [d for d in filtered if d.type in types]

        if path_patterns:
            filtered = [
                d
                for d in filtered
                if any(fnmatch.fnmatch(d.path, pattern) for pattern in path_patterns)
            ]

        if exclude_patterns:
            filtered = [
                d
                for d in filtered
                if not any(
                    fnmatch.fnmatch(d.path, pattern) for pattern in exclude_patterns
                )
            ]

        return filtered


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    ...

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
