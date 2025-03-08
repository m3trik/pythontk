# !/usr/bin/python
# coding=utf-8
import logging as internal_logging
from typing import Any
import unittest
from pythontk.core_utils import ClassProperty


class LoggingNamespace:
    """Namespace for logging utilities.

    This class provides a namespace for logging utilities, such as adding
    handlers and setting log levels. It also provides a custom handler for
    appending log messages to a text widget.
    """

    def __init__(self, logger: internal_logging.Logger):
        self._logger = logger

    def __getattr__(self, name):
        return getattr(internal_logging, name)

    def add_file_handler(
        self, filename: str, level: int = internal_logging.DEBUG
    ) -> None:
        """Adds a file handler for logging."""
        file_handler = internal_logging.FileHandler(filename)
        file_handler.setLevel(level)
        formatter = internal_logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        self._logger.debug(f"Added file handler: {filename}")

    def add_stream_handler(
        self, stream=None, level: int = internal_logging.DEBUG
    ) -> None:
        """Adds a stream handler for logging."""
        stream_handler = internal_logging.StreamHandler(stream)
        stream_handler.setLevel(level)
        formatter = internal_logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)
        self._logger.debug(f"Added stream handler: {stream}")

    def add_text_widget_handler(self, text_widget: object) -> None:
        """Adds a text widget handler for logging."""
        text_edit_handler = TextEditHandler(text_widget)
        text_edit_handler.setFormatter(
            internal_logging.Formatter("%(levelname)s - %(message)s")
        )
        self._logger.addHandler(text_edit_handler)
        self._logger.debug("Added text widget handler")


class TextEditHandler(internal_logging.Handler):
    """Custom handler to append log messages to a text widget.

    This class is a custom logging handler that appends log messages to a
    text widget. The log messages are formatted with HTML tags to allow for
    custom styling.
    """

    def __init__(self, widget: object):
        super().__init__()
        self.widget = widget

    def emit(self, record: internal_logging.LogRecord) -> None:
        msg = self.format(record)
        color = self.get_color(record.levelname)
        formatted_msg = f'<span style="color:{color}">{msg}</span>'
        self.widget.append(formatted_msg)

    def get_color(self, level: str) -> str:
        colors = {
            "DEBUG": "gray",
            "INFO": "white",
            "WARNING": "#FFFF99",
            "ERROR": "#FF9999",
            "CRITICAL": "#CC6666",
        }
        return colors.get(level, "white")


class LoggingMixin:
    """Mixin class for logging utilities.

    This class provides logging utilities for classes. It provides a logger
    for the class and a class logger that is shared across instances. It also
    provides methods for setting log levels, adding handlers, and getting the
    log level of the class logger.
    """

    _logger = None
    _class_logger = None

    @ClassProperty
    def logger(cls):
        """Return the logger for the class."""
        if not cls._logger:
            cls._logger = internal_logging.getLogger(cls.__name__)
            cls._logger.setLevel(internal_logging.DEBUG)  # Default to DEBUG
        return cls._logger

    @ClassProperty
    def class_logger(cls):
        """Return the class logger (shared across instances)."""
        if not cls._class_logger:
            cls._class_logger = internal_logging.getLogger(f"{cls.__name__}.class")
            cls._class_logger.setLevel(internal_logging.DEBUG)  # Default to DEBUG
        return cls._class_logger

    @ClassProperty
    def log_level(cls):
        """Get the log level of the class logger."""
        return cls.logger.level

    @property
    def logging(self):
        return self

    def set_log_level(self, level):
        """Set the log level for the instance."""
        self.logger.setLevel(level)

    @classmethod
    def set_class_log_level(cls, level):
        """Set the log level for the class logger."""
        cls.class_logger.setLevel(level)

    def add_stream_handler(self):
        """Add a stream handler to the logger."""
        handler = internal_logging.StreamHandler()
        self.logger.addHandler(handler)

    def add_text_widget_handler(self, widget):
        """Add a text widget handler."""
        handler = TextEditHandler(widget)
        self.logger.addHandler(handler)

    def add_file_handler(self, filename):
        """Add a file handler to the logger."""
        handler = internal_logging.FileHandler(filename)
        self.logger.addHandler(handler)


class TestLoggingMixin(unittest.TestCase):
    """Unit tests for the LoggingMixin class."""

    def setUp(self):
        class TestClass(LoggingMixin):
            pass

        self.test_instance = TestClass()

    def test_logging_initialization(self):
        logger = self.test_instance.logger

        # Ensure logger is initialized
        self.assertIsInstance(logger, internal_logging.Logger)

        # Ensure default log level is DEBUG (10)
        self.assertEqual(logger.level, internal_logging.DEBUG)

        # Ensure the logger name is correct (should be the class name)
        self.assertEqual(logger.name, "TestClass")

    def test_logging_namespace(self):
        logging_ns = self.test_instance.logging

        # Ensure logging namespace is initialized
        self.assertTrue(hasattr(logging_ns, "logger"))
        self.assertTrue(hasattr(logging_ns, "class_logger"))

        # Ensure default attributes are set
        self.assertEqual(logging_ns.logger.level, internal_logging.DEBUG)
        self.assertEqual(logging_ns.class_logger.level, internal_logging.DEBUG)

    def test_add_stream_handler(self):
        try:
            self.test_instance.add_stream_handler()
            logger = self.test_instance.logger

            # Ensure stream handler is added
            self.assertTrue(
                any(
                    isinstance(h, internal_logging.StreamHandler)
                    for h in logger.handlers
                )
            )
        except PermissionError:
            self.skipTest("File permission error in adding stream handler")

    def test_add_text_widget_handler(self):
        try:

            class MockWidget:
                def append(self, text):
                    pass

            self.test_instance.add_text_widget_handler(MockWidget())
            logger = self.test_instance.logger

            # Ensure text widget handler is added
            self.assertTrue(
                any(isinstance(h, TextEditHandler) for h in logger.handlers)
            )
        except PermissionError:
            self.skipTest("Permission error while adding text widget handler")

    def test_add_file_handler(self):
        try:
            self.test_instance.add_file_handler("test.log")
            logger = self.test_instance.logger

            # Ensure file handler is added
            self.assertTrue(
                any(
                    isinstance(h, internal_logging.FileHandler) for h in logger.handlers
                )
            )
        except PermissionError:
            self.skipTest("File permission error in adding file handler")

    def test_class_logger(self):
        # Test that the class logger is initialized correctly
        class TestClass(LoggingMixin):
            pass

        class_logger = TestClass.class_logger

        # Ensure class logger is an instance of internal_logging.Logger
        self.assertIsInstance(class_logger, internal_logging.Logger)

        # Ensure class logger has the correct name (should be the class name + ".class")
        self.assertEqual(class_logger.name, "TestClass.class")

    def test_class_log_level(self):
        # Test setting class-level log level
        class TestClass(LoggingMixin):
            pass

        # Set class-level log level to INFO
        TestClass.set_class_log_level(internal_logging.INFO)

        # Ensure the class logger level is updated
        class_logger = TestClass.class_logger
        self.assertEqual(class_logger.level, internal_logging.INFO)

    def test_instance_log_level(self):
        # Test setting instance-level log level
        self.test_instance.set_log_level(internal_logging.WARNING)

        # Ensure the instance logger level is updated
        instance_logger = self.test_instance.logger
        self.assertEqual(instance_logger.level, internal_logging.WARNING)

    def test_instance_vs_class_logger(self):
        # Test instance logger and class logger are different
        class TestClass(LoggingMixin):
            pass

        # Get instance and class loggers
        instance_logger = self.test_instance.logger
        class_logger = TestClass.class_logger

        # Ensure they are separate loggers
        self.assertIsNot(instance_logger, class_logger)

        # Ensure instance logger has the correct name
        self.assertEqual(instance_logger.name, "TestClass")

        # Ensure class logger has the correct name
        self.assertEqual(class_logger.name, "TestClass.class")


if __name__ == "__main__":
    from logging.handlers import MemoryHandler

    # Use TextTestRunner to run tests to prevent SystemExit issue.
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLoggingMixin)
    runner = unittest.TextTestRunner()
    runner.run(suite)
