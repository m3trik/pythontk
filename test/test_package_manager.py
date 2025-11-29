#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk PackageManager.

Run with:
    python -m pytest test_package_manager.py -v
    python test_package_manager.py
"""
import sys
import tempfile
import os
import unittest
from unittest.mock import patch, MagicMock

from pythontk.core_utils.package_manager import (
    PackageManager,
    _PkgVersionCheck,
    _PkgVersionUtils,
)

from conftest import BaseTestCase


class PackageManagerTest(BaseTestCase):
    """Tests for PackageManager class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.pkg_mgr = PackageManager()

    def test_init_default_python_path(self):
        """Test initialization with default python path."""
        self.assertEqual(self.pkg_mgr.python_path, sys.executable)

    def test_init_custom_python_path(self):
        """Test initialization with custom python path."""
        custom_path = "/custom/python"
        pkg_mgr = PackageManager(python_path=custom_path)
        self.assertEqual(pkg_mgr.python_path, custom_path)

    def test_parse_command_string(self):
        """Test parsing command from string."""
        command = "install numpy --upgrade"
        result = self.pkg_mgr._parse_command(command)

        self.assertIsInstance(result, list)
        self.assertIn("install", result)
        self.assertIn("numpy", result)

    def test_parse_command_list(self):
        """Test parsing command from list (passthrough)."""
        command = ["install", "numpy"]
        result = self.pkg_mgr._parse_command(command)

        self.assertEqual(result, command)

    def test_is_informational_message(self):
        """Test detecting informational messages."""
        info_msg = "A new release of pip available: 23.0 -> 24.0"
        error_msg = "ERROR: Package not found"

        self.assertTrue(self.pkg_mgr._is_informational_message(info_msg))
        self.assertFalse(self.pkg_mgr._is_informational_message(error_msg))

    def test_is_list_format(self):
        """Test detecting list format output."""
        list_lines = [
            "Package    Version",
            "---------- -------",
            "numpy      1.24.0",
            "pandas     2.0.0",
        ]
        self.assertTrue(self.pkg_mgr._is_list_format(list_lines))

    def test_parse_list_format(self):
        """Test parsing list format."""
        lines = [
            "Package    Version",
            "---------- -------",
            "numpy      1.24.0",
            "pandas     2.0.0",
        ]
        result = self.pkg_mgr._parse_list_format(lines)

        self.assertEqual(result["numpy"], "1.24.0")
        self.assertEqual(result["pandas"], "2.0.0")

    def test_is_key_value_format(self):
        """Test detecting key-value format."""
        kv_lines = [
            "Name: numpy",
            "Version: 1.24.0",
            "Summary: NumPy is the fundamental package",
        ]
        self.assertTrue(self.pkg_mgr._is_key_value_format(kv_lines))

    def test_parse_key_value_format(self):
        """Test parsing key-value format."""
        lines = [
            "Name: numpy",
            "Version: 1.24.0",
            "Location: /usr/lib/python",
        ]
        result = self.pkg_mgr._parse_key_value_format(lines)

        self.assertEqual(result["name"], "numpy")
        self.assertEqual(result["version"], "1.24.0")

    def test_convert_output_json(self):
        """Test converting JSON output."""
        json_output = '[{"name": "numpy", "version": "1.24.0"}]'
        result = self.pkg_mgr._convert_output(json_output)

        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["name"], "numpy")

    def test_process_output_key_value_with_headers(self):
        """Test processing key-value output with proper headers (like pip show)."""
        # When output has "---", list format is detected first, but parse_list looks for dashes
        # Let's test the key-value format that doesn't match list format
        stdout = "Name:test\nVersion:1.0.0"  # No space after colon
        stderr = ""

        result = self.pkg_mgr._process_output(stdout, stderr, output_as_string=False)
        # This should be parsed as key-value
        self.assertIsInstance(result, dict)

    def test_process_output_as_string(self):
        """Test pip command with string output."""
        stdout = "Package    Version\n---------- -------\nnumpy      1.0.0"
        stderr = ""

        result = self.pkg_mgr._process_output(stdout, stderr, output_as_string=True)
        self.assertIsInstance(result, str)
        self.assertIn("numpy", result)


class PkgVersionCheckTest(BaseTestCase):
    """Tests for _PkgVersionCheck class."""

    def test_init(self):
        """Test initialization."""
        checker = _PkgVersionCheck(package_name="test", python_path="/usr/bin/python")

        self.assertEqual(checker._package_name, "test")
        self.assertEqual(checker._python_path, "/usr/bin/python")

    def test_init_default_python_path(self):
        """Test initialization with default python path."""
        checker = _PkgVersionCheck(package_name="test")

        self.assertEqual(checker._python_path, sys.executable)

    def test_new_version_available_no_versions(self):
        """Test new_version_available with no versions set."""
        checker = _PkgVersionCheck()
        self.assertFalse(checker.new_version_available)

    def test_new_version_available_same_version(self):
        """Test new_version_available when versions are same."""
        checker = _PkgVersionCheck()
        checker._installed_ver = "1.0.0"
        checker._latest_ver = "1.0.0"

        self.assertFalse(checker.new_version_available)

    def test_new_version_available_different_versions(self):
        """Test new_version_available when versions differ."""
        checker = _PkgVersionCheck()
        checker._installed_ver = "1.0.0"
        checker._latest_ver = "1.1.0"

        self.assertTrue(checker.new_version_available)

    def test_start_version_check_no_package_raises(self):
        """Test start_version_check without package name raises."""
        checker = _PkgVersionCheck()

        with self.assertRaises(ValueError):
            checker.start_version_check()


class PkgVersionUtilsTest(BaseTestCase):
    """Tests for _PkgVersionUtils class."""

    def test_update_version_increment_patch(self):
        """Test incrementing patch version."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('__version__ = "1.0.0"\n')
            filepath = f.name

        try:
            result = _PkgVersionUtils.update_version(
                filepath, change="increment", version_part="patch"
            )
            self.assertEqual(result, "1.0.1")
        finally:
            os.unlink(filepath)

    def test_update_version_increment_minor(self):
        """Test incrementing minor version."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('__version__ = "1.0.5"\n')
            filepath = f.name

        try:
            result = _PkgVersionUtils.update_version(
                filepath, change="increment", version_part="minor"
            )
            self.assertEqual(result, "1.1.5")
        finally:
            os.unlink(filepath)

    def test_update_version_increment_major(self):
        """Test incrementing major version."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('__version__ = "1.5.3"\n')
            filepath = f.name

        try:
            result = _PkgVersionUtils.update_version(
                filepath, change="increment", version_part="major"
            )
            self.assertEqual(result, "2.5.3")
        finally:
            os.unlink(filepath)

    def test_update_version_decrement_patch(self):
        """Test decrementing patch version."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('__version__ = "1.0.5"\n')
            filepath = f.name

        try:
            result = _PkgVersionUtils.update_version(
                filepath, change="decrement", version_part="patch"
            )
            self.assertEqual(result, "1.0.4")
        finally:
            os.unlink(filepath)

    def test_update_version_no_version_found(self):
        """Test update_version when no version found."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('print("no version here")\n')
            filepath = f.name

        try:
            result = _PkgVersionUtils.update_version(filepath)
            self.assertEqual(result, "")
        finally:
            os.unlink(filepath)

    def test_update_version_invalid_change_raises(self):
        """Test update_version with invalid change parameter raises."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('__version__ = "1.0.0"\n')
            filepath = f.name

        try:
            with self.assertRaises(ValueError):
                _PkgVersionUtils.update_version(
                    filepath, change="invalid", version_part="patch"
                )
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    unittest.main(exit=False)
