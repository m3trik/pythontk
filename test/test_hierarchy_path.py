#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk HierarchyPath.

Run with:
    python -m pytest test_hierarchy_path.py -v
    python test_hierarchy_path.py
"""
import unittest

from pythontk.core_utils.hierarchy_utils.hierarchy_path import HierarchyPath

from conftest import BaseTestCase


class HierarchyPathTest(BaseTestCase):
    """Tests for the HierarchyPath string primitives."""

    def test_clean_namespace(self):
        self.assertEqual(HierarchyPath.clean_namespace("ns:name"), "name")
        self.assertEqual(HierarchyPath.clean_namespace("a:b:name"), "name")
        self.assertEqual(HierarchyPath.clean_namespace("name"), "name")
        self.assertEqual(HierarchyPath.clean_namespace(""), "")

    def test_split_and_join_roundtrip(self):
        self.assertEqual(HierarchyPath.split("a|b|c"), ["a", "b", "c"])
        self.assertEqual(HierarchyPath.split(""), [])
        self.assertEqual(HierarchyPath.join(["a", "b", "c"]), "a|b|c")
        self.assertEqual(HierarchyPath.join([]), "")
        # Absolute (leading-separator) paths keep str.split semantics.
        self.assertEqual(HierarchyPath.split("|a|b"), ["", "a", "b"])
        self.assertEqual(HierarchyPath.join(["", "a", "b"]), "|a|b")

    def test_strip_namespaces(self):
        self.assertEqual(
            HierarchyPath.strip_namespaces("ns:grp|ns:child|leaf"), "grp|child|leaf"
        )
        self.assertEqual(HierarchyPath.strip_namespaces("plain"), "plain")
        self.assertEqual(HierarchyPath.strip_namespaces(""), "")

    def test_normalize(self):
        self.assertEqual(HierarchyPath.normalize("ns:a|ns:b"), "a|b")
        self.assertEqual(
            HierarchyPath.normalize("ns:a|ns:b", clean_namespaces=False), "ns:a|ns:b"
        )
        self.assertEqual(HierarchyPath.normalize(""), "")

    def test_leaf_root_parent_depth(self):
        self.assertEqual(HierarchyPath.leaf("a|b|c"), "c")
        self.assertEqual(HierarchyPath.leaf("solo"), "solo")
        self.assertEqual(HierarchyPath.leaf(""), "")
        self.assertEqual(HierarchyPath.root("a|b|c"), "a")
        self.assertEqual(HierarchyPath.root(""), "")
        self.assertEqual(HierarchyPath.parent("a|b|c"), "a|b")
        self.assertEqual(HierarchyPath.parent("solo"), "")
        self.assertEqual(HierarchyPath.parent(""), "")
        self.assertEqual(HierarchyPath.depth("a|b|c"), 3)
        self.assertEqual(HierarchyPath.depth(""), 0)

    def test_tail(self):
        self.assertEqual(HierarchyPath.tail("a|b|c"), "c")
        self.assertEqual(HierarchyPath.tail("a|b|c", 2), "b|c")
        self.assertEqual(HierarchyPath.tail("a|b|c", 99), "a|b|c")
        self.assertEqual(HierarchyPath.tail("a|b|c", 0), "")

    def test_ends_with(self):
        self.assertTrue(HierarchyPath.ends_with("a|b|c", "b|c"))
        self.assertTrue(HierarchyPath.ends_with("a|b|c", ""))
        self.assertFalse(HierarchyPath.ends_with("a|bc", "c"))
        self.assertFalse(HierarchyPath.ends_with("c", "b|c"))

    def test_custom_separators(self):
        self.assertEqual(HierarchyPath.leaf("a/b/c", path_separator="/"), "c")
        self.assertEqual(
            HierarchyPath.strip_namespaces(
                "ns.a/b", path_separator="/", namespace_separator="."
            ),
            "a/b",
        )

    def test_root_export(self):
        import pythontk as ptk

        self.assertIs(ptk.HierarchyPath, HierarchyPath)
        self.assertTrue(hasattr(ptk, "DifferenceType"))


if __name__ == "__main__":
    unittest.main(exit=False)
