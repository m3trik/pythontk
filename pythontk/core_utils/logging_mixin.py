# !/usr/bin/python
# coding=utf-8
import logging as internal_logging
from typing import Any, Union
from pythontk.core_utils import ClassProperty


class LoggerExt:
    @staticmethod
    def patch(logger: internal_logging.Logger) -> None:
        logger.setLevel_orig = logger.setLevel
        logger.setLevel = LoggerExt._set_level.__get__(logger)

        logger.add_file_handler = LoggerExt._add_file_handler.__get__(logger)
        logger.add_stream_handler = LoggerExt._add_stream_handler.__get__(logger)
        logger.add_text_widget_handler = LoggerExt._add_text_widget_handler.__get__(
            logger
        )
        logger.setup_logging_redirect = LoggerExt._setup_logging_redirect.__get__(
            logger
        )

    def _set_level(self, level: Union[int, str]) -> None:
        if isinstance(level, str):
            level = internal_logging._nameToLevel.get(
                level.upper(), internal_logging.INFO
            )
        self.setLevel_orig(level)

    def _add_file_handler(
        self, filename: str, level: int = internal_logging.DEBUG
    ) -> None:
        handler = internal_logging.FileHandler(filename)
        handler.setLevel(level)
        self.addHandler(handler)
        self.debug(f"Added file handler: {filename}")

    def _add_stream_handler(self, level: int = internal_logging.DEBUG) -> None:
        if not any(
            isinstance(h, internal_logging.StreamHandler) for h in self.handlers
        ):
            handler = internal_logging.StreamHandler()
            handler.setLevel(level)
            handler.setFormatter(
                internal_logging.Formatter("%(levelname)s - %(message)s")
            )
            self.addHandler(handler)
            self.debug("Stream handler attached.")

    def _add_text_widget_handler(self, text_widget: object) -> None:
        handler = TextEditHandler(text_widget)
        handler.setFormatter(internal_logging.Formatter("%(levelname)s - %(message)s"))
        self.addHandler(handler)
        self.debug("Text widget handler added.")

    def _setup_logging_redirect(
        self, widget: object, level: int = internal_logging.INFO
    ) -> None:
        handler = TextEditHandler(widget)
        handler.setFormatter(internal_logging.Formatter("%(levelname)s - %(message)s"))
        self.addHandler(handler)
        self.setLevel(level)


class LoggingMixin:
    """Mixin class for logging utilities.

    Provides a logger for each class and a shared class logger across instances.
    Includes methods for setting log levels, adding handlers, and redirecting logs.
    """

    _logger = None
    _class_logger = None

    @ClassProperty
    def logger(cls) -> internal_logging.Logger:
        if cls.__dict__.get("_logger") is None:
            name = f"{cls.__module__}.{cls.__qualname__}"
            logger = internal_logging.Logger(name, internal_logging.DEBUG)
            logger.propagate = False
            logger.parent = None
            LoggerExt.patch(logger)
            cls._logger = logger
        return cls._logger

    @ClassProperty
    def class_logger(cls) -> internal_logging.Logger:
        if cls.__dict__.get("_class_logger") is None:
            name = f"{cls.__module__}.{cls.__name__}.class"
            logger = internal_logging.getLogger(name)
            logger.setLevel(internal_logging.DEBUG)
            logger.propagate = False
            LoggerExt.patch(logger)
            cls._class_logger = logger
        return cls._class_logger

    @ClassProperty
    def logging(cls):
        """Access to Python's internal logging module (aliased)."""
        return internal_logging


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

        def test_logger_initialization(self):
            logger = self.test_instance.logger
            self.assertIsInstance(logger, internal_logging.Logger)
            self.assertEqual(logger.level, internal_logging.DEBUG)

            expected_name = (
                f"{self.test_instance.__class__.__module__}."
                f"{self.test_instance.__class__.__qualname__}"
            )
            self.assertEqual(logger.name, expected_name)

        def test_logging_property(self):
            self.assertIs(self.test_instance.logging, internal_logging)

        def test_add_stream_handler(self):
            self.test_instance.add_stream_handler()
            stream_handlers = [
                h
                for h in self.test_instance.logger.handlers
                if isinstance(h, internal_logging.StreamHandler)
            ]
            self.assertTrue(stream_handlers, "Stream handler not added")

        def test_add_file_handler(self):
            self.test_instance.logger.add_file_handler("test.log")
            file_handlers = [
                h
                for h in self.test_instance.logger.handlers
                if isinstance(h, internal_logging.FileHandler)
            ]
            self.assertTrue(file_handlers, "File handler not added")

        def test_add_text_widget_handler(self):
            class MockWidget:
                def append(self, text):
                    pass

            self.test_instance.logger.add_text_widget_handler(MockWidget())
            widget_handlers = [
                h
                for h in self.test_instance.logger.handlers
                if isinstance(h, TextEditHandler)
            ]
            self.assertTrue(widget_handlers, "Text widget handler not added")

        def test_log_message_capture_with_memory_handler(self):
            logger = self.test_instance.logger
            memory_handler = MemoryHandler(
                capacity=10000, flushLevel=internal_logging.DEBUG
            )
            logger.addHandler(memory_handler)

            logger.debug("Test log message to verify capture")
            memory_handler.flush()

            captured = any(
                "Test log message to verify capture" in record.getMessage()
                for record in memory_handler.buffer
            )
            logger.removeHandler(memory_handler)
            self.assertTrue(captured, "Expected log message not captured")

    # Use TextTestRunner to run tests to prevent SystemExit issue.
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLoggingMixin)
    runner = unittest.TextTestRunner()
    runner.run(suite)
