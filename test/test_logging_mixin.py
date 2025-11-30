#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk LoggingMixin and LoggerExt.

Run with:
    python -m pytest test_logging_mixin.py -v
    python test_logging_mixin.py
"""
import io
import logging
import tempfile
import os
import unittest

from pythontk.core_utils.logging_mixin import (
    LoggingMixin,
    LoggerExt,
    DefaultTextLogHandler,
)

from conftest import BaseTestCase


class LoggerExtTest(BaseTestCase):
    """Tests for LoggerExt class."""

    def setUp(self):
        """Create a fresh logger for each test."""
        super().setUp()
        self.logger = logging.Logger("test_logger", logging.DEBUG)
        self.logger.handlers = []  # Clear any handlers

    def tearDown(self):
        """Clean up handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        super().tearDown()

    def test_patch_adds_custom_methods(self):
        """Test that patch adds custom methods to logger."""
        LoggerExt.patch(self.logger)

        self.assertTrue(hasattr(self.logger, "success"))
        self.assertTrue(hasattr(self.logger, "result"))
        self.assertTrue(hasattr(self.logger, "notice"))
        self.assertTrue(hasattr(self.logger, "log_box"))
        self.assertTrue(hasattr(self.logger, "log_divider"))

    def test_patch_idempotent(self):
        """Test that patching twice doesn't cause issues."""
        LoggerExt.patch(self.logger)
        LoggerExt.patch(self.logger)

        self.assertTrue(self.logger._logger_ext_patched)

    def test_custom_log_levels_registered(self):
        """Test that custom log levels are registered."""
        LoggerExt.patch(self.logger)

        self.assertEqual(logging.getLevelName(LoggerExt.SUCCESS), "SUCCESS")
        self.assertEqual(logging.getLevelName(LoggerExt.RESULT), "RESULT")
        self.assertEqual(logging.getLevelName(LoggerExt.NOTICE), "NOTICE")

    def test_set_level_with_string(self):
        """Test setting log level with string."""
        LoggerExt.patch(self.logger)
        self.logger.setLevel("DEBUG")
        self.assertEqual(self.logger.level, logging.DEBUG)

    def test_set_level_with_int(self):
        """Test setting log level with integer."""
        LoggerExt.patch(self.logger)
        self.logger.setLevel(logging.WARNING)
        self.assertEqual(self.logger.level, logging.WARNING)

    def test_success_log(self):
        """Test success log method."""
        LoggerExt.patch(self.logger)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(LoggerExt.SUCCESS)
        self.logger.addHandler(handler)
        self.logger.setLevel(LoggerExt.SUCCESS)

        self.logger.success("Test message")

        output = stream.getvalue()
        self.assertIn("Test message", output)

    def test_log_prefix_suffix(self):
        """Test log prefix and suffix."""
        LoggerExt.patch(self.logger)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.logger.set_log_prefix("[PREFIX] ")
        self.logger.set_log_suffix(" [SUFFIX]")
        self.logger.info("Test message")

        output = stream.getvalue()
        self.assertIn("[PREFIX]", output)
        self.assertIn("[SUFFIX]", output)

    def test_add_stream_handler(self):
        """Test adding stream handler."""
        LoggerExt.patch(self.logger)
        initial_count = len(self.logger.handlers)
        self.logger.add_stream_handler()

        self.assertGreater(len(self.logger.handlers), initial_count)

    def test_add_file_handler(self):
        """Test adding file handler."""
        LoggerExt.patch(self.logger)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            filepath = f.name

        try:
            initial_count = len(self.logger.handlers)
            self.logger.add_file_handler(filename=filepath)

            self.assertGreater(len(self.logger.handlers), initial_count)
        finally:
            # Close handlers before deleting file
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
            os.unlink(filepath)

    def test_hide_logger_name(self):
        """Test hiding logger name in output."""
        LoggerExt.patch(self.logger)
        self.logger.hide_logger_name(True)
        self.assertTrue(self.logger._hide_logger_name)

    def test_spam_prevention_enabled(self):
        """Test spam prevention is enabled by default."""
        LoggerExt.patch(self.logger)
        self.assertTrue(self.logger._spam_prevention_enabled)

    def test_clear_error_cache(self):
        """Test clearing error cache."""
        LoggerExt.patch(self.logger)
        self.logger._error_cache = {"key": (0, 1)}
        self.logger.clear_error_cache()
        self.assertEqual(self.logger._error_cache, {})

    def test_set_spam_prevention(self):
        """Test setting spam prevention."""
        LoggerExt.patch(self.logger)
        self.logger.set_spam_prevention(enabled=False, cache_duration=60)

        self.assertFalse(self.logger._spam_prevention_enabled)
        self.assertEqual(self.logger._cache_duration, 60)


class LoggingMixinTest(BaseTestCase):
    """Tests for LoggingMixin class."""

    def test_logger_property(self):
        """Test that logger property returns a Logger."""

        class TestClass(LoggingMixin):
            pass

        self.assertIsInstance(TestClass.logger, logging.Logger)

    def test_logger_has_correct_name(self):
        """Test that logger has correct name."""

        class TestClass(LoggingMixin):
            pass

        logger = TestClass.logger
        self.assertIn("TestClass", logger.name)

    def test_class_logger_property(self):
        """Test that class_logger property returns a Logger."""

        class TestClass(LoggingMixin):
            pass

        self.assertIsInstance(TestClass.class_logger, logging.Logger)

    def test_logging_property_returns_module(self):
        """Test that logging property returns the logging module."""

        class TestClass(LoggingMixin):
            pass

        self.assertEqual(TestClass.logging, logging)

    def test_set_log_level(self):
        """Test setting log level on class."""

        class TestClass(LoggingMixin):
            pass

        TestClass.set_log_level("DEBUG")
        self.assertEqual(TestClass.logger.level, logging.DEBUG)

    def test_set_log_level_with_string(self):
        """Test setting log level with string."""

        class TestClass(LoggingMixin):
            pass

        TestClass.set_log_level("WARNING")
        self.assertEqual(TestClass.logger.level, logging.WARNING)

    def test_logger_is_patched(self):
        """Test that logger is patched with custom methods."""

        class TestClass(LoggingMixin):
            pass

        logger = TestClass.logger
        self.assertTrue(hasattr(logger, "success"))
        self.assertTrue(hasattr(logger, "result"))
        self.assertTrue(hasattr(logger, "notice"))

    def test_logger_propagate_is_false(self):
        """Test that logger propagate is False."""

        class TestClass(LoggingMixin):
            pass

        self.assertFalse(TestClass.logger.propagate)

    def test_separate_loggers_per_class(self):
        """Test that different classes get different loggers."""

        class TestClassA(LoggingMixin):
            pass

        class TestClassB(LoggingMixin):
            pass

        self.assertIsNot(TestClassA.logger, TestClassB.logger)
        self.assertIn("TestClassA", TestClassA.logger.name)
        self.assertIn("TestClassB", TestClassB.logger.name)


class DefaultTextLogHandlerTest(BaseTestCase):
    """Tests for DefaultTextLogHandler."""

    def test_init(self):
        """Test handler initialization."""
        widget = MockTextWidget()
        handler = DefaultTextLogHandler(widget)

        self.assertEqual(handler.widget, widget)
        self.assertTrue(handler.use_html)
        self.assertFalse(handler.monospace)

    def test_get_color(self):
        """Test color mapping for log levels."""
        widget = MockTextWidget()
        handler = DefaultTextLogHandler(widget)

        self.assertEqual(handler.get_color("DEBUG"), "#AAAAAA")
        self.assertEqual(handler.get_color("ERROR"), "#FFCCCC")
        self.assertEqual(handler.get_color("SUCCESS"), "#CCFFCC")

    def test_get_color_unknown_level(self):
        """Test color for unknown level."""
        widget = MockTextWidget()
        handler = DefaultTextLogHandler(widget)

        self.assertEqual(handler.get_color("UNKNOWN"), "#FFFFFF")


class MockTextWidget:
    """Mock text widget for testing."""

    def __init__(self):
        self.messages = []

    def append(self, text):
        self.messages.append(text)


if __name__ == "__main__":
    unittest.main(exit=False)
