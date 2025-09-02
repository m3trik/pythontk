# !/usr/bin/python
# coding=utf-8
from typing import List, Dict, Any, Optional, Callable, Tuple
from .hierarchy_indexer import HierarchyIndexer


class HierarchyMatching:
    """Generic matching strategies for hierarchical data."""

    @staticmethod
    def _clean_namespace(name: str, separator: str = ":") -> str:
        """Remove namespace prefix from a name."""
        return name.split(separator)[-1] if separator in name else name

    @staticmethod
    def _get_path_tail(path: str, num_components: int = 1, separator: str = "|") -> str:
        """Get the last N components of a hierarchy path."""
        components = path.split(separator) if path else []
        tail_components = components[-num_components:] if components else []
        return separator.join(tail_components)

    @staticmethod
    def _path_ends_with(path: str, suffix: str, separator: str = "|") -> bool:
        """Check if a hierarchical path ends with the given suffix."""
        if not suffix:
            return True
        path_components = path.split(separator)
        suffix_components = suffix.split(separator)

        if len(suffix_components) > len(path_components):
            return False

        return path_components[-len(suffix_components) :] == suffix_components

    @staticmethod
    def exact_path_match(
        source_items: List[Any],
        target_items: List[Any],
        get_path_func: Callable[[Any], str],
        path_separator: str = "|",
        clean_namespaces: bool = True,
        namespace_separator: str = ":",
    ) -> Dict[Any, List[Any]]:
        """Find exact path matches between source and target items.

        Args:
            source_items: Items to find matches for
            target_items: Items to search within
            get_path_func: Function to extract path from an item
            path_separator: Character separating path components
            clean_namespaces: Whether to remove namespace prefixes
            namespace_separator: Character separating namespace from name

        Returns:
            Dictionary mapping source items to lists of matching target items
        """
        # Build index of target items
        target_index = HierarchyIndexer.build_path_index(
            target_items,
            get_path_func,
            path_separator,
            clean_namespaces,
            namespace_separator,
        )

        matches = {}
        for source_item in source_items:
            source_path = get_path_func(source_item)
            if source_path:
                target_matches = HierarchyIndexer.find_by_path(
                    target_index,
                    source_path,
                    clean_namespaces,
                    path_separator,
                    namespace_separator,
                )
                if target_matches:
                    matches[source_item] = target_matches

        return matches

    @staticmethod
    def tail_path_match(
        source_items: List[Any],
        target_items: List[Any],
        get_path_func: Callable[[Any], str],
        num_components: int = 2,
        path_separator: str = "|",
        clean_namespaces: bool = True,
        namespace_separator: str = ":",
    ) -> Dict[Any, List[Any]]:
        """Find matches by comparing tail portions of paths.

        Args:
            source_items: Items to find matches for
            target_items: Items to search within
            get_path_func: Function to extract path from an item
            num_components: Number of tail components to compare
            path_separator: Character separating path components
            clean_namespaces: Whether to remove namespace prefixes
            namespace_separator: Character separating namespace from name

        Returns:
            Dictionary mapping source items to lists of matching target items
        """
        # Build index of target items
        target_index = HierarchyIndexer.build_path_index(
            target_items,
            get_path_func,
            path_separator,
            clean_namespaces,
            namespace_separator,
        )

        matches = {}
        for source_item in source_items:
            source_path = get_path_func(source_item)
            if source_path:
                normalized_path = HierarchyIndexer._normalize_path(
                    source_path, clean_namespaces, path_separator, namespace_separator
                )
                source_tail = HierarchyMatching._get_path_tail(
                    normalized_path, num_components, path_separator
                )

                target_matches = HierarchyIndexer.find_by_tail_path(
                    target_index, source_tail, num_components, path_separator
                )
                if target_matches:
                    matches[source_item] = target_matches

        return matches

    @staticmethod
    def fuzzy_name_match(
        source_items: List[Any],
        target_items: List[Any],
        get_name_func: Callable[[Any], str],
        similarity_threshold: float = 0.8,
    ) -> Dict[Any, List[Any]]:
        """Find matches using fuzzy string matching on names.

        Args:
            source_items: Items to find matches for
            target_items: Items to search within
            get_name_func: Function to extract name from an item
            similarity_threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            Dictionary mapping source items to lists of matching target items
        """
        try:
            from difflib import SequenceMatcher
        except ImportError:
            # Fallback to simple exact matching if difflib not available
            return HierarchyMatching._exact_name_match(
                source_items, target_items, get_name_func
            )

        matches = {}

        for source_item in source_items:
            source_name = get_name_func(source_item)
            if not source_name:
                continue

            source_clean = HierarchyMatching._clean_namespace(source_name)
            fuzzy_matches = []

            for target_item in target_items:
                target_name = get_name_func(target_item)
                if not target_name:
                    continue

                target_clean = HierarchyMatching._clean_namespace(target_name)

                # Calculate similarity
                similarity = SequenceMatcher(None, source_clean, target_clean).ratio()
                if similarity >= similarity_threshold:
                    fuzzy_matches.append(target_item)

            if fuzzy_matches:
                matches[source_item] = fuzzy_matches

        return matches

    @staticmethod
    def _exact_name_match(
        source_items: List[Any],
        target_items: List[Any],
        get_name_func: Callable[[Any], str],
    ) -> Dict[Any, List[Any]]:
        """Fallback exact name matching when fuzzy matching is unavailable."""
        # Build index of target items by clean name
        target_index = {}
        for target_item in target_items:
            target_name = get_name_func(target_item)
            if target_name:
                clean_name = HierarchyMatching._clean_namespace(target_name)
                if clean_name not in target_index:
                    target_index[clean_name] = []
                target_index[clean_name].append(target_item)

        matches = {}
        for source_item in source_items:
            source_name = get_name_func(source_item)
            if source_name:
                clean_name = HierarchyMatching._clean_namespace(source_name)
                if clean_name in target_index:
                    matches[source_item] = target_index[clean_name]

        return matches

    @staticmethod
    def multi_strategy_match(
        source_items: List[Any],
        target_items: List[Any],
        get_path_func: Callable[[Any], str],
        get_name_func: Optional[Callable[[Any], str]] = None,
        strategies: List[str] = None,
        path_separator: str = "|",
        clean_namespaces: bool = True,
        namespace_separator: str = ":",
        fuzzy_threshold: float = 0.8,
    ) -> Dict[Any, List[Any]]:
        """Apply multiple matching strategies in order of preference.

        Args:
            source_items: Items to find matches for
            target_items: Items to search within
            get_path_func: Function to extract path from an item
            get_name_func: Function to extract name from an item (for fuzzy matching)
            strategies: List of strategy names to try in order
            path_separator: Character separating path components
            clean_namespaces: Whether to remove namespace prefixes
            namespace_separator: Character separating namespace from name
            fuzzy_threshold: Minimum similarity for fuzzy matching

        Returns:
            Dictionary mapping source items to lists of matching target items
        """
        if strategies is None:
            strategies = ["exact_path", "tail_path", "fuzzy_name"]

        all_matches = {}
        unmatched_items = list(source_items)

        for strategy in strategies:
            if not unmatched_items:
                break

            if strategy == "exact_path":
                matches = HierarchyMatching.exact_path_match(
                    unmatched_items,
                    target_items,
                    get_path_func,
                    path_separator,
                    clean_namespaces,
                    namespace_separator,
                )
            elif strategy == "tail_path":
                matches = HierarchyMatching.tail_path_match(
                    unmatched_items,
                    target_items,
                    get_path_func,
                    2,
                    path_separator,
                    clean_namespaces,
                    namespace_separator,
                )
            elif strategy == "fuzzy_name" and get_name_func:
                matches = HierarchyMatching.fuzzy_name_match(
                    unmatched_items, target_items, get_name_func, fuzzy_threshold
                )
            else:
                continue

            # Add matches and remove matched items from unmatched list
            for source_item, target_matches in matches.items():
                all_matches[source_item] = target_matches
                if source_item in unmatched_items:
                    unmatched_items.remove(source_item)

        return all_matches


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    ...

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
