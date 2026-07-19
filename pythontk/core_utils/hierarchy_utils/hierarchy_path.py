# !/usr/bin/python
# coding=utf-8
"""Pure string primitives for delimited hierarchy paths.

A *hierarchy path* is a plain string of components joined by a separator
(``"grp|child|leaf"``), where each component may carry a namespace prefix
(``"ns:name"``, nested namespaces allowed: ``"a:b:name"``). Default
separators follow the Maya convention (``"|"`` / ``":"``) but every
function takes them as parameters — the primitives are DCC-agnostic and
work for any delimited tree path (file systems, XML paths, org charts).

Semantics are faithful to ``str.split``: an absolute path with a leading
separator (``"|grp|child"``) splits to a leading empty component, exactly
as ``str.split`` produces it. Callers that want rooted and relative paths
to compare equal should ``lstrip`` the separator first.

This module is the single home for these operations —
:class:`HierarchyIndexer`, :class:`HierarchyMatching` and
:class:`HierarchyAnalyzer` all delegate here, as should downstream
packages (mayatk's hierarchy-sync previously duplicated these).
"""
from typing import List


class HierarchyPath:
    """Namespace for pure hierarchy-path string operations.

    All methods are static, side-effect free, and take/return plain
    values (str, List[str]).
    """

    @staticmethod
    def clean_namespace(name: str, namespace_separator: str = ":") -> str:
        """Remove any namespace prefix from a single component name.

        Nested namespaces are collapsed to the final segment
        (``"a:b:name"`` -> ``"name"``).

        Parameters:
            name: Component name, with or without namespace prefix.
            namespace_separator: Character separating namespace from name.

        Returns:
            The name with all namespace prefixes removed.
        """
        if namespace_separator in name:
            return name.split(namespace_separator)[-1]
        return name

    @staticmethod
    def split(path: str, path_separator: str = "|") -> List[str]:
        """Split a hierarchy path into its components.

        Parameters:
            path: The hierarchy path.
            path_separator: Character separating path components.

        Returns:
            List of components; empty list for an empty path.
        """
        return path.split(path_separator) if path else []

    @staticmethod
    def join(components: List[str], path_separator: str = "|") -> str:
        """Join components into a hierarchy path.

        Parameters:
            components: Path components.
            path_separator: Character separating path components.

        Returns:
            The joined path string.
        """
        return path_separator.join(components)

    @staticmethod
    def strip_namespaces(
        path: str, path_separator: str = "|", namespace_separator: str = ":"
    ) -> str:
        """Remove namespace prefixes from every component of a path.

        Parameters:
            path: The hierarchy path.
            path_separator: Character separating path components.
            namespace_separator: Character separating namespace from name.

        Returns:
            The path with all component namespaces removed.
        """
        return HierarchyPath.join(
            [
                HierarchyPath.clean_namespace(comp, namespace_separator)
                for comp in HierarchyPath.split(path, path_separator)
            ],
            path_separator,
        )

    @staticmethod
    def normalize(
        path: str,
        clean_namespaces: bool = True,
        path_separator: str = "|",
        namespace_separator: str = ":",
    ) -> str:
        """Normalize a path for comparison, optionally stripping namespaces.

        Parameters:
            path: The hierarchy path.
            clean_namespaces: Whether to remove namespace prefixes.
            path_separator: Character separating path components.
            namespace_separator: Character separating namespace from name.

        Returns:
            The normalized path ("" for an empty input).
        """
        if not path:
            return ""
        if clean_namespaces:
            return HierarchyPath.strip_namespaces(
                path, path_separator, namespace_separator
            )
        return path

    @staticmethod
    def leaf(path: str, path_separator: str = "|") -> str:
        """Return the last component of a path ("" for an empty path).

        Parameters:
            path: The hierarchy path.
            path_separator: Character separating path components.

        Returns:
            The final component, namespace intact.
        """
        components = HierarchyPath.split(path, path_separator)
        return components[-1] if components else ""

    @staticmethod
    def root(path: str, path_separator: str = "|") -> str:
        """Return the first component of a path ("" for an empty path).

        Parameters:
            path: The hierarchy path.
            path_separator: Character separating path components.

        Returns:
            The first component.
        """
        components = HierarchyPath.split(path, path_separator)
        return components[0] if components else ""

    @staticmethod
    def parent(path: str, path_separator: str = "|") -> str:
        """Return the path with its last component removed.

        Parameters:
            path: The hierarchy path.
            path_separator: Character separating path components.

        Returns:
            The parent path, or "" when the path has fewer than two
            components.
        """
        components = HierarchyPath.split(path, path_separator)
        return HierarchyPath.join(components[:-1], path_separator)

    @staticmethod
    def depth(path: str, path_separator: str = "|") -> int:
        """Return the number of components in a path (0 for an empty path).

        Parameters:
            path: The hierarchy path.
            path_separator: Character separating path components.

        Returns:
            Component count.
        """
        return len(HierarchyPath.split(path, path_separator))

    @staticmethod
    def tail(path: str, num_components: int = 1, path_separator: str = "|") -> str:
        """Return the last N components of a path as a path string.

        Parameters:
            path: The hierarchy path.
            num_components: Number of trailing components to keep.
            path_separator: Character separating path components.

        Returns:
            The tail path (the whole path when it has <= N components,
            "" when N <= 0).
        """
        if num_components <= 0:
            return ""
        components = HierarchyPath.split(path, path_separator)
        return HierarchyPath.join(components[-num_components:], path_separator)

    @staticmethod
    def ends_with(path: str, suffix: str, path_separator: str = "|") -> bool:
        """Check whether a path ends with the given component suffix.

        Component-wise comparison — ``"a|bc"`` does not end with ``"c"``.

        Parameters:
            path: The hierarchy path.
            suffix: Trailing sub-path to test for (empty matches anything).
            path_separator: Character separating path components.

        Returns:
            True when the path's trailing components equal the suffix.
        """
        if not suffix:
            return True
        path_components = HierarchyPath.split(path, path_separator)
        suffix_components = HierarchyPath.split(suffix, path_separator)
        if len(suffix_components) > len(path_components):
            return False
        return path_components[-len(suffix_components) :] == suffix_components


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    ...

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
