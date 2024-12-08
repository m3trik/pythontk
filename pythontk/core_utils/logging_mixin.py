# !/usr/bin/python
# coding=utf-8
import logging as internal_logging
from typing import Any


class LoggingNamespace:
    def __init__(self, logger: internal_logging.Logger):
        self._logger = logger

    def __getattr__(self, name):
        return getattr(internal_logging, name)

    def add_file_handler(
        self, filename: str, level: int = internal_logging.DEBUG
    ) -> None:
        handler = internal_logging.FileHandler(filename)
        handler.setLevel(level)
        self._logger.addHandler(handler)
        self._logger.debug(f"Added file handler: {filename}")

    def add_stream_handler(
        self, stream=None, level: int = internal_logging.DEBUG
    ) -> None:
        handler = internal_logging.StreamHandler(stream)
        handler.setLevel(level)
        self._logger.addHandler(handler)
        self._logger.debug(f"Added stream handler: {stream}")

    def add_text_widget_handler(self, text_widget: object) -> None:
        text_edit_handler = TextEditHandler(text_widget)
        text_edit_handler.setFormatter(
            internal_logging.Formatter("%(levelname)s - %(message)s")
        )
        self._logger.addHandler(text_edit_handler)
        self._logger.debug("Added text widget handler")

    def setup_logging_redirect(
        self, widget: object, level: int = internal_logging.INFO
    ) -> None:
        handler = TextEditHandler(widget)
        handler.setFormatter(internal_logging.Formatter("%(levelname)s - %(message)s"))
        root_logger = internal_logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(level)


class LoggingMixin:
    _loggers = {}

    @property
    def logging(self) -> Any:
        class_name = self.__class__.__name__
        if class_name not in self._loggers:
            self._loggers[class_name] = self._init_logging_namespace()
        return self._loggers[class_name]

    @property
    def logger(self) -> internal_logging.Logger:
        return self.logging._logger

    def _init_logging_namespace(self) -> LoggingNamespace:
        class_name = self.__class__.__name__
        logger = internal_logging.getLogger(class_name)
        logger.setLevel(internal_logging.DEBUG)

        if not logger.hasHandlers():
            stream_handler = internal_logging.StreamHandler()
            stream_handler.setLevel(internal_logging.DEBUG)
            formatter = internal_logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            logger.debug(f"Logger initialized for class: {class_name}")

        return LoggingNamespace(logger)


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
