# !/usr/bin/python
# coding=utf-8
from typing import Any, Dict, List, Set, Tuple, Callable, Optional
from dataclasses import dataclass
from enum import Enum


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
    details: Dict[str, Any]
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
        current_paths: Set[str], reference_paths: Set[str], path_separator: str = "|"
    ) -> Dict[str, Set[str]]:
        """
        Compare two sets of hierarchical paths and categorize differences.

        Args:
            current_paths: Set of paths in current hierarchy
            reference_paths: Set of paths in reference hierarchy
            path_separator: Character separating path components

        Returns:
            Dictionary with 'missing', 'extra', and 'common' path sets
        """
        missing_paths = reference_paths - current_paths
        extra_paths = current_paths - reference_paths
        common_paths = current_paths & reference_paths

        return {"missing": missing_paths, "extra": extra_paths, "common": common_paths}

    @staticmethod
    def analyze_hierarchy_differences(
        current_items: List[Any],
        reference_items: List[Any],
        path_extractor: Callable[[Any], str],
        attribute_extractors: Dict[str, Callable[[Any], Any]] = None,
        path_separator: str = "|",
    ) -> List[HierarchyDifference]:
        """
        Perform comprehensive analysis of differences between hierarchies.

        Args:
            current_items: Items in current hierarchy
            reference_items: Items in reference hierarchy
            path_extractor: Function to extract path from item
            attribute_extractors: Functions to extract attributes for comparison
            path_separator: Character separating path components

        Returns:
            List of detected differences
        """
        if attribute_extractors is None:
            attribute_extractors = {}

        differences = []

        # Build path mappings
        current_map = {path_extractor(item): item for item in current_items}
        reference_map = {path_extractor(item): item for item in reference_items}

        current_paths = set(current_map.keys())
        reference_paths = set(reference_map.keys())

        # Basic path comparison
        path_comparison = HierarchyAnalyzer.compare_path_sets(
            current_paths, reference_paths, path_separator
        )

        # Missing items
        for missing_path in path_comparison["missing"]:
            differences.append(
                HierarchyDifference(
                    type=DifferenceType.MISSING,
                    path=missing_path,
                    details={"reason": "Path exists in reference but not in current"},
                    item=reference_map[missing_path],
                )
            )

        # Extra items
        for extra_path in path_comparison["extra"]:
            differences.append(
                HierarchyDifference(
                    type=DifferenceType.EXTRA,
                    path=extra_path,
                    details={"reason": "Path exists in current but not in reference"},
                    item=current_map[extra_path],
                )
            )

        # Compare attributes of common items
        for common_path in path_comparison["common"]:
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

        Args:
            differences: List of differences from analyze_hierarchy_differences
            similarity_threshold: Minimum similarity to consider items as moved
            path_separator: Character separating path components

        Returns:
            List of move differences detected
        """
        try:
            from difflib import SequenceMatcher
        except ImportError:
            # If difflib not available, return empty list
            return []

        move_differences = []

        missing_diffs = [d for d in differences if d.type == DifferenceType.MISSING]
        extra_diffs = [d for d in differences if d.type == DifferenceType.EXTRA]

        for missing_diff in missing_diffs[
            :
        ]:  # Use slice to allow removal during iteration
            missing_name = missing_diff.path.split(path_separator)[-1]
            best_match = None
            best_similarity = 0

            for extra_diff in extra_diffs[:]:
                extra_name = extra_diff.path.split(path_separator)[-1]
                similarity = SequenceMatcher(None, missing_name, extra_name).ratio()

                if similarity >= similarity_threshold and similarity > best_similarity:
                    best_match = extra_diff
                    best_similarity = similarity

            if best_match:
                # Create move difference
                move_diff = HierarchyDifference(
                    type=DifferenceType.MOVED,
                    path=missing_diff.path,
                    details={
                        "from_path": missing_diff.path,
                        "to_path": best_match.path,
                        "similarity": best_similarity,
                    },
                    item=missing_diff.item,
                )
                move_differences.append(move_diff)

                # Remove the matched differences from the lists
                if missing_diff in missing_diffs:
                    missing_diffs.remove(missing_diff)
                if best_match in extra_diffs:
                    extra_diffs.remove(best_match)

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

        # Categorize by type
        for diff in differences:
            diff_type = diff.type.value
            if diff_type not in categorized["by_type"]:
                categorized["by_type"][diff_type] = []
            categorized["by_type"][diff_type].append(diff)

        # Categorize by hierarchy level
        for diff in differences:
            level = len(diff.path.split(path_separator)) if diff.path else 0
            if level not in categorized["by_level"]:
                categorized["by_level"][level] = []
            categorized["by_level"][level].append(diff)

        # Categorize by root component
        for diff in differences:
            root = diff.path.split(path_separator)[0] if diff.path else "unknown"
            if root not in categorized["by_root"]:
                categorized["by_root"][root] = []
            categorized["by_root"][root].append(diff)

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

                items_to_show = type_diffs[:max_items_per_category]
                for diff in items_to_show:
                    report_lines.append(f"  • {diff.path}")
                    if diff.details:
                        for key, value in diff.details.items():
                            report_lines.append(f"    {key}: {value}")

                if len(type_diffs) > max_items_per_category:
                    remaining = len(type_diffs) - max_items_per_category
                    report_lines.append(f"  ... and {remaining} more items")

                report_lines.append("")

        # Level distribution
        report_lines.append("DISTRIBUTION BY HIERARCHY LEVEL:")
        for level in sorted(categorized["by_level"].keys()):
            level_diffs = categorized["by_level"][level]
            report_lines.append(f"  Level {level}: {len(level_diffs)} items")

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
        types: List[DifferenceType] = None,
        path_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
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

        # Filter by type
        if types:
            filtered = [d for d in filtered if d.type in types]

        # Filter by path patterns
        if path_patterns:
            import fnmatch

            pattern_filtered = []
            for diff in filtered:
                for pattern in path_patterns:
                    if fnmatch.fnmatch(diff.path, pattern):
                        pattern_filtered.append(diff)
                        break
            filtered = pattern_filtered

        # Exclude patterns
        if exclude_patterns:
            import fnmatch

            exclude_filtered = []
            for diff in filtered:
                should_exclude = False
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(diff.path, pattern):
                        should_exclude = True
                        break
                if not should_exclude:
                    exclude_filtered.append(diff)
            filtered = exclude_filtered

        return filtered


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    ...

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
