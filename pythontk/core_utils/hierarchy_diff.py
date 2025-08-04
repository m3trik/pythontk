# !/usr/bin/python
# coding=utf-8
from typing import List, Dict, Any, Optional
import json


class HierarchyDiff:
    """Generic data class to hold hierarchical difference results.

    This class can be used for any hierarchical structure comparison,
    not just Maya scenes. Examples include file systems, XML/JSON structures,
    organizational charts, etc.
    """

    def __init__(self):
        """Initialize empty diff result."""
        self.missing: List[str] = []
        self.extra: List[str] = []
        self.renamed: List[str] = []
        self.reparented: List[str] = []
        self.fuzzy_matches: List[Dict[str, str]] = []
        self.metadata: Dict[str, Any] = {}

    def is_valid(self) -> bool:
        """Check if hierarchy has no significant differences.

        Returns:
            True if no missing, renamed, or reparented items exist
        """
        return not (self.missing or self.renamed or self.reparented)

    def has_differences(self) -> bool:
        """Check if any differences exist (including extra items).

        Returns:
            True if any differences exist
        """
        return bool(self.missing or self.extra or self.renamed or self.reparented)

    def total_issues(self) -> int:
        """Get total count of all issues.

        Returns:
            Sum of all difference counts
        """
        return (
            len(self.missing)
            + len(self.extra)
            + len(self.renamed)
            + len(self.reparented)
        )

    def as_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with difference lists
        """
        return {
            "missing": self.missing,
            "extra": self.extra,
            "renamed": self.renamed,
            "reparented": self.reparented,
            "fuzzy_matches": self.fuzzy_matches,
            "metadata": self.metadata,
        }

    def as_json(self, indent: Optional[int] = 2) -> str:
        """Convert to JSON string.

        Parameters:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.as_dict(), indent=indent)

    def save_to_file(self, filepath: str, indent: Optional[int] = 2) -> None:
        """Save diff result to JSON file.

        Parameters:
            filepath: Path to save the JSON file
            indent: JSON indentation level
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=indent)

    @classmethod
    def load_from_file(cls, filepath: str) -> "HierarchyDiff":
        """Load diff result from JSON file.

        Parameters:
            filepath: Path to the JSON file

        Returns:
            HierarchyDiff instance
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = cls()
        result.missing = data.get("missing", [])
        result.extra = data.get("extra", [])
        result.renamed = data.get("renamed", [])
        result.reparented = data.get("reparented", [])
        result.fuzzy_matches = data.get("fuzzy_matches", [])
        result.metadata = data.get("metadata", {})

        return result

    def clear(self) -> None:
        """Clear all diff results."""
        self.missing.clear()
        self.extra.clear()
        self.renamed.clear()
        self.reparented.clear()
        self.fuzzy_matches.clear()
        self.metadata.clear()

    def merge(self, other: "HierarchyDiff") -> None:
        """Merge another diff result into this one.

        Parameters:
            other: Another HierarchyDiff to merge
        """
        self.missing.extend(other.missing)
        self.extra.extend(other.extra)
        self.renamed.extend(other.renamed)
        self.reparented.extend(other.reparented)
        self.fuzzy_matches.extend(other.fuzzy_matches)
        self.metadata.update(other.metadata)

    def get_summary(self) -> Dict[str, int]:
        """Get summary counts of all difference types.

        Returns:
            Dictionary with counts for each difference type
        """
        return {
            "missing": len(self.missing),
            "extra": len(self.extra),
            "renamed": len(self.renamed),
            "reparented": len(self.reparented),
            "fuzzy_matches": len(self.fuzzy_matches),
            "total_issues": self.total_issues(),
        }

    def filter_by_pattern(self, pattern: str, field: str = "missing") -> List[str]:
        """Filter items by regex pattern.

        Parameters:
            pattern: Regex pattern to match
            field: Field to filter ("missing", "extra", etc.)

        Returns:
            List of items matching the pattern
        """
        import re

        field_data = getattr(self, field, [])
        if isinstance(field_data, list):
            return [item for item in field_data if re.search(pattern, item)]
        return []

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the diff result.

        Parameters:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def __str__(self) -> str:
        """String representation of diff result."""
        summary = self.get_summary()
        return (
            f"HierarchyDiff(missing={summary['missing']}, "
            f"extra={summary['extra']}, renamed={summary['renamed']}, "
            f"reparented={summary['reparented']})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    print("=== HierarchyDiff Example ===")

    # Create a diff result
    diff = HierarchyDiff()
    diff.missing = ["item1", "item2"]
    diff.extra = ["item3"]
    diff.renamed = ["item4"]
    diff.add_metadata("analysis_time", "2024-08-02T10:30:00")
    diff.add_metadata("source", "test_hierarchy")

    print(f"Diff summary: {diff}")
    print(f"Is valid: {diff.is_valid()}")
    print(f"Has differences: {diff.has_differences()}")
    print(f"Total issues: {diff.total_issues()}")

    # JSON export/import
    json_str = diff.as_json()
    print(f"\nJSON representation:\n{json_str}")

    # Save and load example
    # diff.save_to_file("hierarchy_diff.json")
    # loaded_diff = HierarchyDiff.load_from_file("hierarchy_diff.json")

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
# This module provides a general-purpose data structure for hierarchy difference analysis.
# It can be used for:
#
# - File system comparison
# - Database schema comparison
# - XML/JSON structure comparison
# - Maya scene hierarchy comparison
# - Any hierarchical data structure analysis
#
# The class includes utilities for JSON serialization, filtering, and metadata management.
# --------------------------------------------------------------------------------------------
