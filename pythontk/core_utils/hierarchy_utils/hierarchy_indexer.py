# !/usr/bin/python
# coding=utf-8
from typing import Dict, List, Any, Optional, Callable


class HierarchyIndexer:
    """Generic utilities for building and querying tree indices."""

    @staticmethod
    def _clean_namespace(name: str, separator: str = ":") -> str:
        """Remove namespace prefix from a name."""
        return name.split(separator)[-1] if separator in name else name

    @staticmethod
    def _split_hierarchy_path(path: str, separator: str = "|") -> List[str]:
        """Split a hierarchy path into components."""
        return path.split(separator) if path else []

    @staticmethod
    def _join_hierarchy_path(components: List[str], separator: str = "|") -> str:
        """Join path components into a hierarchy path."""
        return separator.join(components)

    @staticmethod
    def _clean_hierarchy_path(
        path: str, path_separator: str = "|", namespace_separator: str = ":"
    ) -> str:
        """Clean namespaces from all components of a hierarchy path."""
        components = HierarchyIndexer._split_hierarchy_path(path, path_separator)
        cleaned_components = [
            HierarchyIndexer._clean_namespace(comp, namespace_separator)
            for comp in components
        ]
        return HierarchyIndexer._join_hierarchy_path(cleaned_components, path_separator)

    @staticmethod
    def _normalize_path(
        path: str,
        clean_namespaces: bool = True,
        path_separator: str = "|",
        namespace_separator: str = ":",
    ) -> str:
        """Normalize a hierarchy path by cleaning and standardizing it."""
        if not path:
            return ""
        if clean_namespaces:
            return HierarchyIndexer._clean_hierarchy_path(
                path, path_separator, namespace_separator
            )
        else:
            return path

    @staticmethod
    def build_path_index(
        items: List[Any],
        get_path_func: Callable[[Any], str],
        path_separator: str = "|",
        clean_namespaces: bool = True,
        namespace_separator: str = ":",
    ) -> Dict[str, List[Any]]:
        """Build an index mapping cleaned paths to items.

        Args:
            items: List of tree items to index
            get_path_func: Function to extract path from an item
            path_separator: Character separating path components
            clean_namespaces: Whether to remove namespace prefixes
            namespace_separator: Character separating namespace from name

        Returns:
            Dictionary mapping normalized paths to lists of items
        """
        index = {}

        for item in items:
            raw_path = get_path_func(item)
            if raw_path:
                normalized_path = HierarchyIndexer._normalize_path(
                    raw_path, clean_namespaces, path_separator, namespace_separator
                )
                if normalized_path not in index:
                    index[normalized_path] = []
                index[normalized_path].append(item)

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
        normalized_path = HierarchyIndexer._normalize_path(
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
            components = HierarchyIndexer._split_hierarchy_path(path, path_separator)
            tail_components = components[-num_components:] if components else []
            path_tail = HierarchyIndexer._join_hierarchy_path(
                tail_components, path_separator
            )
            if path_tail == target_tail:
                matches.extend(items)

        return matches

    @staticmethod
    def get_path_components_index(
        items: List[Any], get_path_func: Callable[[Any], str], path_separator: str = "|"
    ) -> Dict[str, List[Any]]:
        """Build an index mapping individual path components to items.

        Args:
            items: List of tree items to index
            get_path_func: Function to extract path from an item
            path_separator: Character separating path components

        Returns:
            Dictionary mapping component names to lists of items
        """
        index = {}

        for item in items:
            path = get_path_func(item)
            if path:
                components = HierarchyIndexer._split_hierarchy_path(
                    path, path_separator
                )
                for component in components:
                    clean_component = HierarchyIndexer._clean_namespace(component)
                    if clean_component not in index:
                        index[clean_component] = []
                    if item not in index[clean_component]:
                        index[clean_component].append(item)

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
        index = {}

        for item in items:
            path = get_path_func(item)
            if path:
                depth = len(
                    HierarchyIndexer._split_hierarchy_path(path, path_separator)
                )
                if depth not in index:
                    index[depth] = []
                index[depth].append(item)

        return index


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    ...

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
