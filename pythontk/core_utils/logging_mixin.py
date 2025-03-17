# !/usr/bin/python
# coding=utf-8
import logging as internal_logging
from typing import Any
from pythontk.core_utils import ClassProperty


class LoggingMixin:
    """Mixin class for logging utilities.

    Provides a logger for each class and a shared class logger across instances.
    Includes methods for setting log levels, adding handlers, and redirecting logs.
    """

    _logger = None
    _class_logger = None

    def __getattr__(self, name):
        """Allows access to the instance/class attribute first, then to logging."""
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(internal_logging, name)

    @ClassProperty
    def logger(cls) -> internal_logging.Logger:
        """Completely isolate each logger to avoid cross-level changes."""
        if not cls._logger:
            # Create a brand-new Logger instance (bypasses getLogger).
            unique_name = f"{cls.__module__}.{cls.__qualname__}"
            cls._logger = internal_logging.Logger(unique_name, internal_logging.DEBUG)
            cls._logger.propagate = False
            cls._logger.parent = None  # Ensures no parent-level interactions
        return cls._logger

    @ClassProperty
    def class_logger(cls) -> internal_logging.Logger:
        """Use a distinct name for the class-level logger."""
        if not cls._class_logger:
            logger_name = f"{cls.__module__}.{cls.__name__}.class"
            cls._class_logger = internal_logging.getLogger(logger_name)
            cls._class_logger.setLevel(internal_logging.DEBUG)
            cls._class_logger.propagate = False
        return cls._class_logger

    @ClassProperty
    def log_level(cls) -> int:
        """Get the log level of the class logger."""
        return cls.logger.level

    @property
    def logging(self):
        """Allows instance-level access to logging."""
        return self

    def add_file_handler(
        self, filename: str, level: int = internal_logging.DEBUG
    ) -> None:
        """Adds a file handler to the logger."""
        handler = internal_logging.FileHandler(filename)
        handler.setLevel(level)
        self.logger.addHandler(handler)  # ✅ FIX: Use self.logger
        self.logger.debug(f"Added file handler: {filename}")

    def add_stream_handler(
        self, stream=None, level: int = internal_logging.DEBUG
    ) -> None:
        """Adds a stream handler to the logger."""
        handler = internal_logging.StreamHandler(stream)
        handler.setLevel(level)
        self.logger.addHandler(handler)  # ✅ FIX: Use self.logger
        self.logger.debug(f"Added stream handler: {stream}")

    def add_text_widget_handler(self, text_widget: object) -> None:
        """Adds a text widget handler for logging output."""
        text_edit_handler = TextEditHandler(text_widget)
        text_edit_handler.setFormatter(
            internal_logging.Formatter("%(levelname)s - %(message)s")
        )
        self.logger.addHandler(text_edit_handler)  # ✅ FIX: Use self.logger
        self.logger.debug("Added text widget handler")

    def setup_logging_redirect(
        self, widget: object, level: int = internal_logging.INFO
    ) -> None:
        handler = TextEditHandler(widget)
        handler.setFormatter(internal_logging.Formatter("%(levelname)s - %(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(level)


class TextEditHandler(internal_logging.Handler):
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


if __name__ == "__main__":
    import unittest
    from logging.handlers import MemoryHandler

    class TestLoggingMixin(unittest.TestCase):
        def setUp(self):
            class TestClass(LoggingMixin):
                pass

            self.test_instance = TestClass()

        def test_logging_initialization(self):
            logger = self.test_instance.logger

            # Ensure logger is initialized
            self.assertIsInstance(logger, internal_logging.Logger)

            # Ensure default log level is DEBUG
            self.assertEqual(logger.level, internal_logging.DEBUG)

            # Ensure the logger name is correct
            self.assertEqual(logger.name, f"{__name__}.Logging")

        def test_logging_namespace(self):
            logging_ns = self.test_instance.logging

            # Ensure logging namespace is initialized
            self.assertTrue(hasattr(logging_ns, "_log_file"))
            self.assertTrue(hasattr(logging_ns, "_hide_log_file"))
            self.assertTrue(hasattr(logging_ns, "_text_widget"))

            # Ensure default attributes
            self.assertIsNone(logging_ns._log_file)
            self.assertFalse(logging_ns._hide_log_file)
            self.assertIsNone(logging_ns._text_widget)

        def test_add_stream_handler(self):
            self.test_instance.logging.add_stream_handler()
            logger = self.test_instance.logger

            # Ensure stream handler is added
            self.assertTrue(
                any(
                    isinstance(h, internal_logging.StreamHandler)
                    for h in logger.handlers
                )
            )

        def test_add_file_handler(self):
            self.test_instance.logging.add_file_handler("test.log")
            logger = self.test_instance.logger

            # Ensure file handler is added
            self.assertTrue(
                any(
                    isinstance(h, internal_logging.FileHandler) for h in logger.handlers
                )
            )

        def test_add_text_widget_handler(self):
            class MockWidget:
                def append(self, text):
                    pass

            self.test_instance.logging.add_text_widget_handler(MockWidget())
            logger = self.test_instance.logger

            # Ensure text widget handler is added
            self.assertTrue(
                any(isinstance(h, TextEditHandler) for h in logger.handlers)
            )

        def test_logging_namespace_handler_logging(self):
            # Ensure the logger is initialized first
            logger = self.test_instance.logger

            # Add a memory handler to capture the logs
            memory_handler = MemoryHandler(
                capacity=10000, flushLevel=internal_logging.DEBUG
            )
            logger.addHandler(memory_handler)

            with self.assertLogs(f"{__name__}.Logging", level="DEBUG") as cm:
                self.test_instance.logging
                logger.debug("Test log message to verify capture")

            # Flush the memory handler to capture logs
            memory_handler.flush()
            logger.removeHandler(memory_handler)

            self.assertIn(
                "DEBUG:__main__.Logging:Test log message to verify capture",
                cm.output,
            )

    # Use TextTestRunner to run tests to prevent SystemExit issue.
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLoggingMixin)
    runner = unittest.TextTestRunner()
    runner.run(suite)
