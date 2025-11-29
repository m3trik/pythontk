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


class HelpMixinTest(BaseTestCase):
    """HelpMixin test class."""

    def test_detect_format_google(self):
        """Test detection of Google-style docstrings."""
        docstring = """
        This is a function.
        
        Args:
            param1: First parameter.
            param2: Second parameter.
            
        Returns:
            Some value.
        """
        self.assertEqual(HelpMixin._detect_format(docstring), "google")

    def test_detect_format_numpy(self):
        """Test detection of NumPy-style docstrings."""
        docstring = """
        This is a function.
        
        Parameters
        ----------
        param1 : int
            First parameter.
            
        Returns
        -------
        int
            Some value.
        """
        self.assertEqual(HelpMixin._detect_format(docstring), "numpy")

    def test_detect_format_restructuredtext(self):
        """Test detection of reStructuredText-style docstrings."""
        docstring = """
        This is a function.
        
        :param param1: First parameter.
        :param param2: Second parameter.
        :returns: Some value.
        """
        self.assertEqual(HelpMixin._detect_format(docstring), "restructuredtext")

    def test_detect_format_custom(self):
        """Test detection of custom-style docstrings."""
        # Use headers unique to custom format: "Properties:", "Methods:", "Attributes:", "Usage:"
        docstring = """
        This is a function.
        
        Properties:
            prop1: A property.
            
        Usage:
            Call this function.
        """
        self.assertEqual(HelpMixin._detect_format(docstring), "custom")

    def test_detect_format_unknown_defaults_to_custom(self):
        """Test that unknown format defaults to custom."""
        docstring = """
        This is a simple docstring with no recognizable format.
        Just a description.
        """
        self.assertEqual(HelpMixin._detect_format(docstring), "custom")

    def test_format_docstring_preserves_content(self):
        """Test that format_docstring includes the content."""
        docstring = """This is a description.
        
        Parameters:
            value: The input value.
        """
        formatted = HelpMixin._format_docstring(docstring)
        self.assertIn("Description:", formatted)
        self.assertIn("Parameters:", formatted)
        self.assertIn("value", formatted)

    def test_help_with_class(self):
        """Test help() displays class information."""

        class TestClass(HelpMixin):
            def test_method(self, arg1: str) -> str:
                """A test method.

                Parameters:
                    arg1: An argument.

                Returns:
                    A string.
                """
                return arg1

        # Capture stdout
        captured = StringIO()
        sys.stdout = captured
        try:
            TestClass.help()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("Class: TestClass", output)
        self.assertIn("test_method", output)

    def test_help_with_specific_method(self):
        """Test help() with a specific method name."""

        class TestClass(HelpMixin):
            def my_method(self, value: int) -> int:
                """Doubles the input value.

                Parameters:
                    value: The input value to double.

                Returns:
                    The doubled value.
                """
                return value * 2

        # Capture stdout
        captured = StringIO()
        sys.stdout = captured
        try:
            TestClass.help("my_method")
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("my_method", output)
        self.assertIn("Doubles the input value", output)

    def test_help_with_invalid_method(self):
        """Test help() with non-existent method name."""

        class TestClass(HelpMixin):
            pass

        # Capture stdout
        captured = StringIO()
        sys.stdout = captured
        try:
            TestClass.help("nonexistent_method")
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("No help available", output)

    def test_formats_dict_structure(self):
        """Test that FORMATS dictionary has expected structure."""
        self.assertIn("google", HelpMixin.FORMATS)
        self.assertIn("numpy", HelpMixin.FORMATS)
        self.assertIn("restructuredtext", HelpMixin.FORMATS)
        self.assertIn("custom", HelpMixin.FORMATS)

        # Each format should have a list of headers
        for format_name, headers in HelpMixin.FORMATS.items():
            self.assertIsInstance(headers, list)
            self.assertTrue(len(headers) > 0)


if __name__ == "__main__":
    unittest.main(exit=False)
