# !/usr/bin/python
# coding=utf-8
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .hierarchy_path import HierarchyPath
from .hierarchy_analyzer import DifferenceType, HierarchyDifference


@dataclass
class HierarchyDiff:
    """Generic, JSON-serializable container for hierarchy difference results.

    This class can be used for any hierarchical structure comparison,
    not just scene hierarchies. Examples include file systems, XML/JSON
    structures, organizational charts, etc.

    It is the serialization-friendly counterpart of the analyzer's
    ``List[HierarchyDifference]`` records — build one from analyzer output
    with :meth:`from_differences`.
    """

    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)
    renamed: List[str] = field(default_factory=list)
    reparented: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    fuzzy_matches: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_differences(
        cls,
        differences: Iterable[HierarchyDifference],
        path_separator: str = "|",
    ) -> "HierarchyDiff":
        """Build a diff container from analyzer difference records.

        Routing: MISSING -> ``missing``, EXTRA -> ``extra``, MODIFIED ->
        ``modified``. A MOVED record (see
        :meth:`HierarchyAnalyzer.detect_moved_items`) contributes its
        ``from_path`` to ``renamed`` when both paths share a parent,
        otherwise to ``reparented`` (a move that also renames counts as
        reparented), and its pairing is preserved in ``fuzzy_matches`` as
        ``{"from", "to", "similarity"}``. MISSING/EXTRA records whose path
        is consumed by a MOVED pairing are dropped — the move supersedes
        them. Paths are compared verbatim; normalize upstream if needed.

        Parameters:
            differences: Analyzer difference records.
            path_separator: Character separating path components.

        Returns:
            A populated HierarchyDiff.
        """
        differences = list(differences)
        result = cls()

        moved_from = set()
        moved_to = set()
        for record in differences:
            if record.type is not DifferenceType.MOVED:
                continue
            from_path = record.details.get("from_path", record.path)
            to_path = record.details.get("to_path", "")
            moved_from.add(from_path)
            moved_to.add(to_path)

            same_parent = HierarchyPath.parent(
                from_path, path_separator
            ) == HierarchyPath.parent(to_path, path_separator)
            (result.renamed if same_parent else result.reparented).append(from_path)
            result.fuzzy_matches.append(
                {
                    "from": from_path,
                    "to": to_path,
                    "similarity": record.details.get("similarity"),
                }
            )

        for record in differences:
            if record.type is DifferenceType.MISSING and record.path not in moved_from:
                result.missing.append(record.path)
            elif record.type is DifferenceType.EXTRA and record.path not in moved_to:
                result.extra.append(record.path)
            elif record.type is DifferenceType.MODIFIED:
                result.modified.append(record.path)

        return result

    def is_valid(self) -> bool:
        """Check if hierarchy has no significant differences.

        Returns:
            True if no missing, renamed, reparented, or modified items
            exist (extra items alone do not invalidate)
        """
        return not (self.missing or self.renamed or self.reparented or self.modified)

    def has_differences(self) -> bool:
        """Check if any differences exist (including extra items).

        Returns:
            True if any differences exist
        """
        return bool(
            self.missing
            or self.extra
            or self.renamed
            or self.reparented
            or self.modified
        )

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
            + len(self.modified)
        )

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with difference lists
        """
        return {
            "missing": self.missing,
            "extra": self.extra,
            "renamed": self.renamed,
            "reparented": self.reparented,
            "modified": self.modified,
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

        return cls(
            missing=data.get("missing", []),
            extra=data.get("extra", []),
            renamed=data.get("renamed", []),
            reparented=data.get("reparented", []),
            modified=data.get("modified", []),
            fuzzy_matches=data.get("fuzzy_matches", []),
            metadata=data.get("metadata", {}),
        )

    def clear(self) -> None:
        """Clear all diff results."""
        self.missing.clear()
        self.extra.clear()
        self.renamed.clear()
        self.reparented.clear()
        self.modified.clear()
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
        self.modified.extend(other.modified)
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
            "modified": len(self.modified),
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
            f"reparented={summary['reparented']}, modified={summary['modified']})"
        )

    __repr__ = __str__


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    print("=== HierarchyDiff Example ===")

    diff = HierarchyDiff(missing=["item1", "item2"], extra=["item3"])
    diff.add_metadata("source", "test_hierarchy")

    print(f"Diff summary: {diff}")
    print(f"Is valid: {diff.is_valid()}")
    print(f"Total issues: {diff.total_issues()}")
    print(f"\nJSON representation:\n{diff.as_json()}")

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
# General-purpose serializable container for hierarchy difference analysis:
# file systems, database schemas, XML/JSON structures, scene hierarchies.
# Compose with HierarchyAnalyzer via HierarchyDiff.from_differences().
# --------------------------------------------------------------------------------------------
