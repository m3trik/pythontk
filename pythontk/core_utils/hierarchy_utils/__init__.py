# coding=utf-8
"""
Hierarchy utilities subpackage providing generic, reusable hierarchy matching and manipulation logic.

This subpackage contains framework-agnostic utilities for:
- Path manipulation and namespace handling
- Tree-like structure indexing and matching
- Hierarchy comparison and diff operations
- Widget-agnostic tree matching strategies

These utilities can be used across different GUI frameworks (Qt, Tkinter, etc.)
and different applications (Maya, Blender, etc.) without dependencies on specific frameworks.
"""

from .hierarchy_indexer import HierarchyIndexer
from .hierarchy_matching import HierarchyMatching
from .hierarchy_analyzer import HierarchyDifference, HierarchyAnalyzer

__all__ = [
    "HierarchyIndexer",
    "HierarchyMatching",
    "HierarchyDifference",
    "HierarchyAnalyzer",
]
