# !/usr/bin/python
# coding=utf-8
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Callable, Union

from .hierarchy_path import HierarchyPath
from .hierarchy_indexer import HierarchyIndexer

# A matching strategy takes (source_items, target_items) and returns a
# mapping of source item -> list of matching target items.
MatchStrategy = Callable[[List[Any], List[Any]], Dict[Any, List[Any]]]


class HierarchyMatching:
    """Generic matching strategies for hierarchical data."""

    @staticmethod
    def _clean_namespace(name: str, separator: str = ":") -> str:
        """Deprecated — use :meth:`HierarchyPath.clean_namespace`.

        Kept because released downstream code (mayatk hierarchy-sync)
        called this before the public API existed.
        """
        return HierarchyPath.clean_namespace(name, separator)

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
                normalized_path = HierarchyPath.normalize(
                    source_path, clean_namespaces, path_separator, namespace_separator
                )
                source_tail = HierarchyPath.tail(
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
        # Clean target names once — not per source item.
        cleaned_targets = []
        for target_item in target_items:
            target_name = get_name_func(target_item)
            if target_name:
                cleaned_targets.append(
                    (target_item, HierarchyPath.clean_namespace(target_name))
                )

        matches = {}
        for source_item in source_items:
            source_name = get_name_func(source_item)
            if not source_name:
                continue

            source_clean = HierarchyPath.clean_namespace(source_name)
            fuzzy_matches = [
                target_item
                for target_item, target_clean in cleaned_targets
                if SequenceMatcher(None, source_clean, target_clean).ratio()
                >= similarity_threshold
            ]

            if fuzzy_matches:
                matches[source_item] = fuzzy_matches

        return matches

    @staticmethod
    def multi_strategy_match(
        source_items: List[Any],
        target_items: List[Any],
        get_path_func: Callable[[Any], str],
        get_name_func: Optional[Callable[[Any], str]] = None,
        strategies: Optional[List[Union[str, MatchStrategy]]] = None,
        path_separator: str = "|",
        clean_namespaces: bool = True,
        namespace_separator: str = ":",
        fuzzy_threshold: float = 0.8,
        tail_components: int = 2,
    ) -> Dict[Any, List[Any]]:
        """Apply multiple matching strategies in order of preference.

        Each strategy only sees the items the previous strategies left
        unmatched. Built-in strategy names: ``"exact_path"``,
        ``"tail_path"``, ``"fuzzy_name"`` (the latter requires
        *get_name_func* and is skipped without it). A strategy may also be
        a callable ``(source_items, target_items) -> {source: [targets]}``
        for custom matching without touching this module.

        Args:
            source_items: Items to find matches for
            target_items: Items to search within
            get_path_func: Function to extract path from an item
            get_name_func: Function to extract name from an item (for fuzzy matching)
            strategies: Strategy names and/or callables to try in order
            path_separator: Character separating path components
            clean_namespaces: Whether to remove namespace prefixes
            namespace_separator: Character separating namespace from name
            fuzzy_threshold: Minimum similarity for fuzzy matching
            tail_components: Tail length used by the "tail_path" strategy

        Returns:
            Dictionary mapping source items to lists of matching target items
        """
        if strategies is None:
            strategies = ["exact_path", "tail_path", "fuzzy_name"]

        builtin: Dict[str, Optional[MatchStrategy]] = {
            "exact_path": lambda items, targets: HierarchyMatching.exact_path_match(
                items,
                targets,
                get_path_func,
                path_separator,
                clean_namespaces,
                namespace_separator,
            ),
            "tail_path": lambda items, targets: HierarchyMatching.tail_path_match(
                items,
                targets,
                get_path_func,
                tail_components,
                path_separator,
                clean_namespaces,
                namespace_separator,
            ),
            "fuzzy_name": (
                (
                    lambda items, targets: HierarchyMatching.fuzzy_name_match(
                        items, targets, get_name_func, fuzzy_threshold
                    )
                )
                if get_name_func
                else None
            ),
        }

        all_matches: Dict[Any, List[Any]] = {}
        unmatched_items = list(source_items)

        for strategy in strategies:
            if not unmatched_items:
                break

            strategy_func = strategy if callable(strategy) else builtin.get(strategy)
            if strategy_func is None:
                continue

            matches = strategy_func(unmatched_items, target_items)
            if matches:
                all_matches.update(matches)
                unmatched_items = [
                    item for item in unmatched_items if item not in matches
                ]

        return all_matches


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    ...

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
