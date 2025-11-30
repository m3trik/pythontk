#!/usr/bin/python
# coding=utf-8
"""
Pytest configuration and shared fixtures for pythontk tests.

This module provides:
- Shared test utilities and base classes
- Common fixtures for all test modules
- Path management for test resources
"""
import os
import re
import unittest
from pathlib import Path


# =============================================================================
# Test Utilities & Base Classes
# =============================================================================


class TestPaths:
    """Centralized test path management."""

    BASE_DIR = Path(__file__).parent
    TEST_FILES_DIR = BASE_DIR / "test_files"
    IMGTK_TEST_DIR = TEST_FILES_DIR / "imgtk_test"

    @classmethod
    def get(cls, *parts: str) -> str:
        """Get absolute path to a test file."""
        return str(cls.TEST_FILES_DIR.joinpath(*parts))

    @classmethod
    def get_imgtk(cls, filename: str) -> str:
        """Get path to an image test file."""
        return str(cls.IMGTK_TEST_DIR / filename)


class BaseTestCase(unittest.TestCase):
    """Base test case with common utilities and assertions."""

    @staticmethod
    def replace_mem_address(obj: object) -> str:
        """Normalize memory addresses in string representations for comparison.

        Parameters:
            obj: Object to convert and normalize.

        Returns:
            String with memory addresses replaced by '0x00000000000'.

        Example:
            >>> replace_mem_address("<Widget at 0x1ebe2677e80>")
            "<Widget at 0x00000000000>"
        """
        return re.sub(r"0x[a-fA-F\d]+", "0x00000000000", str(obj))

    def assertImageMode(self, image, expected_mode: str, msg: str = None):
        """Assert that a PIL Image has the expected mode."""
        self.assertEqual(image.mode, expected_mode, msg)

    def assertImageSize(self, image, expected_size: tuple, msg: str = None):
        """Assert that a PIL Image has the expected size."""
        self.assertEqual(image.size, expected_size, msg)

    def assertPathExists(self, path: str, msg: str = None):
        """Assert that a file or directory exists."""
        self.assertTrue(os.path.exists(path), msg or f"Path does not exist: {path}")
