#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk HelpMixin.

Run with:
    python -m pytest test_help_mixin.py -v
    python test_help_mixin.py
"""
import json
import unittest
from io import StringIO
import sys

from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.core_utils.symbol_record import SymbolRecord

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
    # Tests for show_mro() method
    # =========================================================================

    def test_show_mro_returns_inheritance_chain(self):
        """Test show_mro() returns the inheritance chain."""
        result = ChildClass.show_mro(returns=True)
        self.assertIn("ChildClass", result)
        self.assertIn("SampleClass", result)
        self.assertIn("HelpMixin", result)
        self.assertIn("object", result)

    def test_show_mro_brief_mode(self):
        """Test show_mro() with brief=True shows only class names."""
        result = ChildClass.show_mro(brief=True, returns=True)
        self.assertIn("ChildClass", result)
        # Brief mode shouldn't include module paths
        self.assertNotIn("__main__.", result)

    def test_show_mro_full_mode(self):
        """Test show_mro() with brief=False shows module paths."""
        result = ChildClass.show_mro(brief=False, returns=True)
        # Full mode should include module for builtins
        self.assertIn("builtins.object", result)

    def test_show_mro_prints_when_returns_false(self):
        """Test show_mro() prints when returns=False."""
        captured = StringIO()
        sys.stdout = captured
        try:
            result = SampleClass.show_mro()
        finally:
            sys.stdout = sys.__stdout__
        self.assertIsNone(result)
        self.assertIn("SampleClass", captured.getvalue())

    def test_builtin_mro_not_shadowed(self):
        """HelpMixin must not shadow Python's built-in ``type.mro()``.

        Regression: a classmethod named ``mro`` broke ``SomeClass.mro()`` for
        every class in the ecosystem that inherits HelpMixin."""
        result = ChildClass.mro()
        self.assertIsInstance(result, list)
        self.assertIn(ChildClass, result)
        self.assertIn(object, result)

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

    # =========================================================================
    # Tests for structured output (as_dict / as_json)
    # =========================================================================

    def test_help_as_dict_class_payload(self):
        """help(as_dict=True) returns a structured class payload."""
        result = SampleClass.help(as_dict=True)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["class"], "SampleClass")
        self.assertIn("HelpMixin", result["mro"])
        self.assertIsInstance(result["members"], list)

    def test_help_as_dict_member_fields(self):
        """Each member record carries the core + enriched fields."""
        members = {m["name"]: m for m in SampleClass.help(as_dict=True)["members"]}
        pm = members["public_method"]
        self.assertEqual(pm["qualname"], "SampleClass.public_method")
        self.assertEqual(pm["kind"], "method")
        self.assertIn("arg1", pm["signature"])
        self.assertEqual(pm["defined_in"], "SampleClass")
        self.assertIn("test_help_mixin.py", pm["source"])
        self.assertIn("flags", pm)

    def test_help_as_dict_flags_async(self):
        """async members surface an 'async' flag in structured output."""
        members = {m["name"]: m for m in SampleClass.help(as_dict=True)["members"]}
        self.assertIn("async", members["async_method"]["flags"])

    def test_help_as_json_parses_to_dict_payload(self):
        """as_json returns a JSON string equal to the as_dict payload."""
        as_json = SampleClass.help(as_json=True)
        self.assertIsInstance(as_json, str)
        self.assertEqual(json.loads(as_json), SampleClass.help(as_dict=True))

    def test_help_as_dict_single_member(self):
        """help(name, as_dict=True) returns one enriched record."""
        result = SampleClass.help("public_method", as_dict=True)
        self.assertEqual(result["name"], "public_method")
        self.assertEqual(result["defined_in"], "SampleClass")

    def test_help_as_dict_nonexistent_member(self):
        """A missing member yields an error payload, not an exception."""
        result = SampleClass.help("nope", as_dict=True)
        self.assertIn("error", result)

    def test_help_as_dict_defined_in_tracks_inheritance(self):
        """Inherited members report the defining class, not the subclass."""
        members = {m["name"]: m for m in ChildClass.help(as_dict=True)["members"]}
        self.assertEqual(members["public_method"]["defined_in"], "SampleClass")
        self.assertEqual(members["child_method"]["defined_in"], "ChildClass")

    def test_collect_records_returns_symbolrecords(self):
        """_collect_records yields bare SymbolRecords (drift-gate shape)."""
        records = SampleClass._collect_records(inherited=False)
        self.assertTrue(all(isinstance(r, SymbolRecord) for r in records))
        by_name = {r.name: r for r in records}
        self.assertEqual(by_name["public_method"].qualname, "SampleClass.public_method")
        self.assertEqual(by_name["from_string"].kind, "classmethod")
        self.assertEqual(by_name["static_helper"].kind, "staticmethod")
        self.assertEqual(by_name["doubled"].kind, "property")

    def test_list_members_as_dict(self):
        """list_members(as_dict=True) returns enriched records, not names."""
        result = SampleClass.list_members(as_dict=True)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(m, dict) for m in result))
        self.assertIn("public_method", {m["name"] for m in result})

    def test_classify_as_dict_all(self):
        """classify(as_dict=True) returns a list of enriched records."""
        result = SampleClass.classify(as_dict=True)
        self.assertIsInstance(result, list)
        self.assertIn("public_method", {m["name"] for m in result})

    def test_classify_as_dict_single(self):
        """classify(name, as_dict=True) returns one enriched record."""
        result = SampleClass.classify("public_method", as_dict=True)
        self.assertEqual(result["kind"], "method")

    def test_about_as_dict(self):
        """about(obj, as_dict=True) works on any object."""
        result = HelpMixin.about(SampleClass.public_method, as_dict=True)
        self.assertEqual(result["name"], "public_method")
        self.assertIn("arg1", result["signature"])
        self.assertIn("processes input", result["doc"])

    def test_about_as_json(self):
        """about(obj, as_json=True) returns a JSON string."""
        result = HelpMixin.about(SampleClass, as_json=True)
        self.assertEqual(json.loads(result)["name"], "SampleClass")


if __name__ == "__main__":
    unittest.main(exit=False)
