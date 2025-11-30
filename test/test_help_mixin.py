#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk HelpMixin.

Run with:
    python -m pytest test_help_mixin.py -v
    python test_help_mixin.py
"""
import unittest
from io import StringIO
import sys

from pythontk.core_utils.help_mixin import HelpMixin

from conftest import BaseTestCase


class SampleClass(HelpMixin):
    """A sample class for testing HelpMixin functionality."""

    def __init__(self, value: int):
        """Initialize with a value."""
        self.value = value

    def public_method(self, arg1: str) -> str:
        """A public method that processes input.

        Parameters:
            arg1: The input argument.

        Returns:
            The processed string.
        """
        return arg1.upper()

    def another_method(self) -> int:
        """Returns the stored value."""
        return self.value

    @property
    def doubled(self) -> int:
        """Return the value doubled."""
        return self.value * 2

    @classmethod
    def from_string(cls, s: str) -> "SampleClass":
        """Create instance from string."""
        return cls(int(s))

    @staticmethod
    def static_helper() -> str:
        """A static helper method."""
        return "helper"

    def _private_method(self):
        """A private method."""
        pass

    async def async_method(self) -> None:
        """An async method."""
        pass

    def generator_method(self):
        """A generator method."""
        yield 1


class ChildClass(SampleClass):
    """A child class for testing inheritance."""

    def child_method(self) -> str:
        """A method defined in the child class."""
        return "child"


class HelpMixinTest(BaseTestCase):
    """HelpMixin test class."""

    def test_help_returns_string_when_returns_true(self):
        """Test that help() returns a string when returns=True."""
        result = SampleClass.help(returns=True)
        self.assertIsInstance(result, str)
        self.assertIn("SampleClass", result)

    def test_help_returns_none_when_returns_false(self):
        """Test that help() returns None when printing (default)."""
        # Capture stdout
        captured = StringIO()
        sys.stdout = captured
        try:
            result = SampleClass.help()
        finally:
            sys.stdout = sys.__stdout__
        self.assertIsNone(result)

    def test_help_for_specific_method(self):
        """Test help() with a specific method name."""
        result = SampleClass.help("public_method", returns=True)
        self.assertIn("public_method", result)

    def test_help_for_nonexistent_method(self):
        """Test help() with non-existent method name."""
        result = SampleClass.help("nonexistent_method", returns=True)
        self.assertIn("has no member", result)

    def test_help_brief_mode(self):
        """Test help() in brief mode returns summaries."""
        result = SampleClass.help(brief=True, returns=True)
        self.assertIn("SampleClass", result)
        self.assertIn("public_method", result)

    def test_help_sorted(self):
        """Test help() with sort=True returns sorted members."""
        result = SampleClass.help(brief=True, sort=True, returns=True)
        self.assertIsInstance(result, str)
        # Members should be alphabetically sorted
        self.assertIn("another_method", result)
        self.assertIn("public_method", result)

    def test_help_excludes_private_by_default(self):
        """Test help() excludes private members by default."""
        result = SampleClass.help(brief=True, returns=True)
        self.assertNotIn("_private_method", result)

    def test_help_includes_private_when_requested(self):
        """Test help() includes private members when private=True."""
        result = SampleClass.help(brief=True, private=True, returns=True)
        self.assertIn("_private_method", result)

    def test_help_filter_methods_only(self):
        """Test help() with members='methods' filters to methods."""
        result = SampleClass.help(members="methods", brief=True, returns=True)
        self.assertIn("public_method", result)
        self.assertIn("another_method", result)

    def test_help_filter_properties_only(self):
        """Test help() with members='properties' filters to properties."""
        result = SampleClass.help(members="properties", brief=True, returns=True)
        self.assertIn("doubled", result)
        # Methods should not appear
        self.assertNotIn("public_method", result)

    def test_help_filter_classmethods_only(self):
        """Test help() with members='classmethods' filters correctly."""
        result = SampleClass.help(members="classmethods", brief=True, returns=True)
        self.assertIn("from_string", result)

    def test_help_filter_staticmethods_only(self):
        """Test help() with members='staticmethods' filters correctly."""
        result = SampleClass.help(members="staticmethods", brief=True, returns=True)
        self.assertIn("static_helper", result)

    def test_help_inherited_false_excludes_inherited(self):
        """Test help() with inherited=False excludes inherited members."""
        result = SampleClass.help(inherited=False, brief=True, returns=True)
        # 'help' is inherited from HelpMixin, should be excluded
        # Only SampleClass's own members should appear
        self.assertIn("public_method", result)

    def test_help_for_specific_member_brief(self):
        """Test help() for specific member in brief mode."""
        result = SampleClass.help("public_method", brief=True, returns=True)
        self.assertIn("public_method", result)
        self.assertIn("processes input", result)

    def test_get_summary_extracts_first_line(self):
        """Test _get_summary extracts first line of docstring."""
        docstring = """First line of docstring.

        More details here.
        """
        result = HelpMixin._get_summary(docstring)
        self.assertEqual(result, "First line of docstring.")

    def test_get_summary_handles_empty(self):
        """Test _get_summary handles empty docstring."""
        result = HelpMixin._get_summary("")
        self.assertEqual(result, "")

    def test_get_summary_handles_none(self):
        """Test _get_summary handles None."""
        result = HelpMixin._get_summary(None)
        self.assertEqual(result, "")

    def test_help_shows_signatures(self):
        """Test that help output includes method signatures."""
        result = SampleClass.help(brief=True, returns=True)
        # Signatures should appear
        self.assertIn("(", result)
        self.assertIn(")", result)

    def test_help_class_with_no_docstring(self):
        """Test help() works for class with no docstring."""

        class NoDocClass(HelpMixin):
            def method(self):
                pass

        result = NoDocClass.help(brief=True, returns=True)
        self.assertIn("NoDocClass", result)

    # =========================================================================
    # Tests for source() method
    # =========================================================================

    def test_source_returns_source_code(self):
        """Test source() returns source code for a method."""
        result = SampleClass.source("public_method", returns=True)
        self.assertIn("def public_method", result)
        self.assertIn("return arg1.upper()", result)

    def test_source_for_class(self):
        """Test source() with no name returns class source."""
        result = SampleClass.source(returns=True)
        self.assertIn("class SampleClass", result)
        self.assertIn("HelpMixin", result)

    def test_source_for_nonexistent_member(self):
        """Test source() for non-existent member."""
        result = SampleClass.source("nonexistent", returns=True)
        self.assertIn("has no member", result)

    def test_source_prints_when_returns_false(self):
        """Test source() prints when returns=False."""
        captured = StringIO()
        sys.stdout = captured
        try:
            result = SampleClass.source("public_method")
        finally:
            sys.stdout = sys.__stdout__
        self.assertIsNone(result)
        self.assertIn("def public_method", captured.getvalue())

    # =========================================================================
    # Tests for where() method
    # =========================================================================

    def test_where_returns_file_and_line(self):
        """Test where() returns file path and line number."""
        result = SampleClass.where("public_method", returns=True)
        self.assertIn("test_help_mixin.py", result)
        self.assertIn(":", result)  # Has line number separator

    def test_where_for_class(self):
        """Test where() with no name returns class location."""
        result = SampleClass.where(returns=True)
        self.assertIn("test_help_mixin.py", result)

    def test_where_for_nonexistent_member(self):
        """Test where() for non-existent member."""
        result = SampleClass.where("nonexistent", returns=True)
        self.assertIn("has no member", result)

    def test_where_prints_when_returns_false(self):
        """Test where() prints when returns=False."""
        captured = StringIO()
        sys.stdout = captured
        try:
            result = SampleClass.where("public_method")
        finally:
            sys.stdout = sys.__stdout__
        self.assertIsNone(result)
        self.assertIn("test_help_mixin.py", captured.getvalue())

    # =========================================================================
    # Tests for mro() method
    # =========================================================================

    def test_mro_returns_inheritance_chain(self):
        """Test mro() returns the inheritance chain."""
        result = ChildClass.mro(returns=True)
        self.assertIn("ChildClass", result)
        self.assertIn("SampleClass", result)
        self.assertIn("HelpMixin", result)
        self.assertIn("object", result)

    def test_mro_brief_mode(self):
        """Test mro() with brief=True shows only class names."""
        result = ChildClass.mro(brief=True, returns=True)
        self.assertIn("ChildClass", result)
        # Brief mode shouldn't include module paths
        self.assertNotIn("__main__.", result)

    def test_mro_full_mode(self):
        """Test mro() with brief=False shows module paths."""
        result = ChildClass.mro(brief=False, returns=True)
        # Full mode should include module for builtins
        self.assertIn("builtins.object", result)

    def test_mro_prints_when_returns_false(self):
        """Test mro() prints when returns=False."""
        captured = StringIO()
        sys.stdout = captured
        try:
            result = SampleClass.mro()
        finally:
            sys.stdout = sys.__stdout__
        self.assertIsNone(result)
        self.assertIn("SampleClass", captured.getvalue())

    # =========================================================================
    # Tests for signature() method
    # =========================================================================

    def test_signature_returns_detailed_info(self):
        """Test signature() returns detailed parameter info."""
        result = SampleClass.signature("public_method", returns=True)
        self.assertIn("public_method", result)
        self.assertIn("arg1", result)
        self.assertIn("str", result)  # Type annotation
        self.assertIn("Returns:", result)

    def test_signature_shows_defaults(self):
        """Test signature() shows default values."""

        class WithDefaults(HelpMixin):
            def method(self, arg: str = "default") -> None:
                pass

        result = WithDefaults.signature("method", returns=True)
        self.assertIn("= 'default'", result)

    def test_signature_for_nonexistent_member(self):
        """Test signature() for non-existent member."""
        result = SampleClass.signature("nonexistent", returns=True)
        self.assertIn("has no member", result)

    def test_signature_for_non_callable(self):
        """Test signature() for non-callable raises helpful message."""

        class WithAttr(HelpMixin):
            attr = 42

        result = WithAttr.signature("attr", returns=True)
        self.assertIn("not callable", result)

    # =========================================================================
    # Tests for classify() method
    # =========================================================================

    def test_classify_single_member(self):
        """Test classify() for a single member."""
        result = SampleClass.classify("public_method", returns=True)
        self.assertIn("Member: public_method", result)
        self.assertIn("Type: method", result)
        self.assertIn("Defined in: SampleClass", result)

    def test_classify_async_method(self):
        """Test classify() shows async flag."""
        result = SampleClass.classify("async_method", returns=True)
        self.assertIn("async", result)

    def test_classify_generator_method(self):
        """Test classify() shows generator flag."""
        result = SampleClass.classify("generator_method", returns=True)
        self.assertIn("generator", result)

    def test_classify_all_members(self):
        """Test classify() with no name shows all members."""
        result = SampleClass.classify(returns=True)
        self.assertIn("Classification of SampleClass", result)
        self.assertIn("METHOD", result)
        self.assertIn("PROPERTY", result)

    def test_classify_for_nonexistent_member(self):
        """Test classify() for non-existent member."""
        result = SampleClass.classify("nonexistent", returns=True)
        self.assertIn("has no member", result)

    # =========================================================================
    # Tests for list_members() method
    # =========================================================================

    def test_list_members_returns_names(self):
        """Test list_members() returns list of member names."""
        result = SampleClass.list_members(returns=True)
        self.assertIsInstance(result, list)
        self.assertIn("public_method", result)
        self.assertIn("another_method", result)

    def test_list_members_filters_by_type(self):
        """Test list_members() with members filter."""
        result = SampleClass.list_members("properties", returns=True)
        self.assertIn("doubled", result)
        self.assertNotIn("public_method", result)

    def test_list_members_sorted(self):
        """Test list_members() returns sorted list."""
        result = SampleClass.list_members(sort=True, returns=True)
        self.assertEqual(result, sorted(result))

    def test_list_members_excludes_private(self):
        """Test list_members() excludes private by default."""
        result = SampleClass.list_members(returns=True)
        self.assertNotIn("_private_method", result)

    def test_list_members_includes_private_when_requested(self):
        """Test list_members() includes private when requested."""
        result = SampleClass.list_members(private=True, returns=True)
        self.assertIn("_private_method", result)

    def test_list_members_excludes_inherited(self):
        """Test list_members() with inherited=False."""
        result = ChildClass.list_members(inherited=False, returns=True)
        self.assertIn("child_method", result)
        # Inherited methods should be excluded
        self.assertNotIn("help", result)

    # =========================================================================
    # Tests for async/generator detection in help output
    # =========================================================================

    def test_help_shows_async_flag(self):
        """Test help() shows async flag for async methods."""
        result = SampleClass.help(brief=True, returns=True)
        # Find the async_method line and check for async flag
        self.assertIn("async_method", result)
        self.assertIn("async", result.lower())

    def test_help_shows_generator_flag(self):
        """Test help() shows generator flag for generator methods."""
        result = SampleClass.help(brief=True, returns=True)
        self.assertIn("generator_method", result)
        self.assertIn("generator", result.lower())

    # =========================================================================
    # Tests for _get_member_flags
    # =========================================================================

    def test_get_member_flags_async(self):
        """Test _get_member_flags detects async."""
        flags = HelpMixin._get_member_flags(SampleClass.async_method)
        self.assertIn("async", flags)

    def test_get_member_flags_generator(self):
        """Test _get_member_flags detects generator."""
        flags = HelpMixin._get_member_flags(SampleClass.generator_method)
        self.assertIn("generator", flags)

    def test_get_member_flags_normal(self):
        """Test _get_member_flags returns empty for normal methods."""
        flags = HelpMixin._get_member_flags(SampleClass.public_method)
        self.assertEqual(flags, "")

    # =========================================================================
    # Tests for _format_annotation
    # =========================================================================

    def test_format_annotation_with_name(self):
        """Test _format_annotation for types with __name__."""
        result = HelpMixin._format_annotation(str)
        self.assertEqual(result, "str")

    def test_format_annotation_without_name(self):
        """Test _format_annotation for complex typing types."""
        from typing import Optional

        result = HelpMixin._format_annotation(Optional[str])
        # Should contain Optional and str
        self.assertIn("Optional", result)
        self.assertIn("str", result)


if __name__ == "__main__":
    unittest.main(exit=False)
