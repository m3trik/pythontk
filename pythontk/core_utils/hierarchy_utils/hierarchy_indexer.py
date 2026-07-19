# !/usr/bin/python
# coding=utf-8
from typing import Dict, List, Any, Callable

from .hierarchy_path import HierarchyPath


class HierarchyIndexer:
    """Generic utilities for building and querying tree indices."""

    # --- Deprecated private aliases -------------------------------------------------------
    # Path-string primitives now live publicly on HierarchyPath; these shims
    # remain because released downstream code (mayatk hierarchy-sync) called
    # them before the public API existed. Use HierarchyPath directly.

    @staticmethod
    def _clean_namespace(name: str, separator: str = ":") -> str:
        """Deprecated — use :meth:`HierarchyPath.clean_namespace`."""
        return HierarchyPath.clean_namespace(name, separator)

    @staticmethod
    def _split_hierarchy_path(path: str, separator: str = "|") -> List[str]:
        """Deprecated — use :meth:`HierarchyPath.split`."""
        return HierarchyPath.split(path, separator)

    @staticmethod
    def _join_hierarchy_path(components: List[str], separator: str = "|") -> str:
        """Deprecated — use :meth:`HierarchyPath.join`."""
        return HierarchyPath.join(components, separator)

    @staticmethod
    def _clean_hierarchy_path(
        path: str, path_separator: str = "|", namespace_separator: str = ":"
    ) -> str:
        """Deprecated — use :meth:`HierarchyPath.strip_namespaces`."""
        return HierarchyPath.strip_namespaces(path, path_separator, namespace_separator)

    @staticmethod
    def _normalize_path(
        path: str,
        clean_namespaces: bool = True,
        path_separator: str = "|",
        namespace_separator: str = ":",
    ) -> str:
        """Deprecated — use :meth:`HierarchyPath.normalize`."""
        return HierarchyPath.normalize(
            path, clean_namespaces, path_separator, namespace_separator
        )

    # --- Index building / querying --------------------------------------------------------

    @staticmethod
    def build_path_index(
        items: List[Any],
        get_path_func: Callable[[Any], str],
        path_separator: str = "|",
        clean_namespaces: bool = True,
        namespace_separator: str = ":",
    ) -> Dict[str, List[Any]]:
        """Build an index mapping normalized paths to items.

        Args:
            items: List of tree items to index
            get_path_func: Function to extract path from an item
            path_separator: Character separating path components
            clean_namespaces: Whether to remove namespace prefixes
            namespace_separator: Character separating namespace from name

        Returns:
            Dictionary mapping normalized paths to lists of items
        """
        index: Dict[str, List[Any]] = {}

        for item in items:
            raw_path = get_path_func(item)
            if raw_path:
                normalized_path = HierarchyPath.normalize(
                    raw_path, clean_namespaces, path_separator, namespace_separator
                )
                index.setdefault(normalized_path, []).append(item)

        return index

    @staticmethod
    def find_by_path(
        index: Dict[str, List[Any]],
        target_path: str,
        clean_namespaces: bool = True,
        path_separator: str = "|",
        namespace_separator: str = ":",
    ) -> List[Any]:
        """Find items in index by path.

        Args:
            index: Path index created by build_path_index
            target_path: Path to search for
            clean_namespaces: Whether to normalize the target path
            path_separator: Character separating path components
            namespace_separator: Character separating namespace from name

        Returns:
            List of items matching the path
        """
        normalized_path = HierarchyPath.normalize(
            target_path, clean_namespaces, path_separator, namespace_separator
        )
        return index.get(normalized_path, [])

    @staticmethod
    def find_by_tail_path(
        index: Dict[str, List[Any]],
        target_tail: str,
        num_components: int = 2,
        path_separator: str = "|",
    ) -> List[Any]:
        """Find items by matching the tail portion of their paths.

        Args:
            index: Path index created by build_path_index
            target_tail: Tail path to match
            num_components: Number of components in the tail
            path_separator: Character separating path components

        Returns:
            List of items whose paths end with the target tail
        """
        matches = []

        for path, items in index.items():
            if HierarchyPath.tail(path, num_components, path_separator) == target_tail:
                matches.extend(items)

        return matches

    @staticmethod
    def get_path_components_index(
        items: List[Any],
        get_path_func: Callable[[Any], str],
        path_separator: str = "|",
        namespace_separator: str = ":",
    ) -> Dict[str, List[Any]]:
        """Build an index mapping individual path components to items.

        Args:
            items: List of tree items to index
            get_path_func: Function to extract path from an item
            path_separator: Character separating path components
            namespace_separator: Character separating namespace from name

        Returns:
            Dictionary mapping namespace-cleaned component names to lists
            of items whose paths contain that component
        """
        index: Dict[str, List[Any]] = {}

        for item in items:
            path = get_path_func(item)
            if path:
                for component in HierarchyPath.split(path, path_separator):
                    clean_component = HierarchyPath.clean_namespace(
                        component, namespace_separator
                    )
                    bucket = index.setdefault(clean_component, [])
                    if item not in bucket:
                        bucket.append(item)

        return index

    @staticmethod
    def get_depth_index(
        items: List[Any], get_path_func: Callable[[Any], str], path_separator: str = "|"
    ) -> Dict[int, List[Any]]:
        """Build an index mapping path depths to items.

        Args:
            items: List of tree items to index
            get_path_func: Function to extract path from an item
            path_separator: Character separating path components

        Returns:
            Dictionary mapping depths (int) to lists of items
        """
        index: Dict[int, List[Any]] = {}

        for item in items:
            path = get_path_func(item)
            if path:
                depth = HierarchyPath.depth(path, path_separator)
                index.setdefault(depth, []).append(item)

        return index


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    ...

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
