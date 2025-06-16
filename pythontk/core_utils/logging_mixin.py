# !/usr/bin/python
# coding=utf-8
import threading
import logging as internal_logging
from typing import Union, List, Optional
from pythontk.core_utils import ClassProperty


import logging


class LoggerExt:
    _text_handler = None  # Can be instance or class

    # Define custom log levels
    SUCCESS = 25
    RESULT = 35
    NOTICE = 45

    @classmethod
    def patch(cls, logger: logging.Logger) -> None:
        """Patch the logger with additional methods and setup."""

        # Register custom log levels
        logging.addLevelName(cls.SUCCESS, "SUCCESS")
        logging.addLevelName(cls.RESULT, "RESULT")
        logging.addLevelName(cls.NOTICE, "NOTICE")

        if not hasattr(logger, "setLevel_orig"):
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

        logger.set_text_handler = LoggerExt._set_text_handler
        logger.get_text_handler = LoggerExt._get_text_handler.__get__(logger)

        # Core methods for logging
        logger.log_raw = LoggerExt._log_raw.__get__(logger)
        LoggerExt._patch_logger(logger)

        # Public methods
        logger.success = LoggerExt._success.__get__(logger)
        logger.result = LoggerExt._result.__get__(logger)
        logger.notice = LoggerExt._notice.__get__(logger)

        # Set global log formatters
        logger.log_format = "[%(levelname)s] %(message)s"
        logger.debug_log_format = "[%(levelname)s] %(name)s: %(message)s"
        logger._formatter_selector = LoggerExt._select_formatter.__get__(logger)

        if not logger.handlers:
            logger.add_stream_handler()

    @staticmethod
    def _patch_logger(logger: internal_logging.Logger) -> None:
        """Bind all LoggerExt log methods except _log_raw."""
        for name in dir(LoggerExt):
            if name.startswith("_log_") and name != "_log_raw":
                func = getattr(LoggerExt, name)
                if callable(func):
                    setattr(logger, name[1:], func.__get__(logger))

    @staticmethod
    def _select_formatter(self, level: int) -> logging.Formatter:
        """Select formatter based on the log level."""
        if level == LoggerExt.SUCCESS:
            return logging.Formatter(
                "[%(levelname)s] %(message)s"
            )  # Example for SUCCESS
        elif level == LoggerExt.RESULT:
            return logging.Formatter(
                "[%(levelname)s] %(message)s"
            )  # Example for RESULT
        elif level == LoggerExt.NOTICE:
            return logging.Formatter(
                "[%(levelname)s] %(message)s"
            )  # Example for NOTICE
        elif level <= logging.DEBUG:
            return logging.Formatter(self.debug_log_format)
        return logging.Formatter(self.log_format)

    @staticmethod
    def _set_level(self, level: Union[int, str]) -> None:
        if isinstance(level, str):
            level = logging._nameToLevel.get(level.upper(), logging.INFO)
        self.setLevel_orig(level)

    @staticmethod
    def _add_stream_handler(self, level: int = logging.DEBUG) -> None:
        if not any(isinstance(h, logging.StreamHandler) for h in self.handlers):
            handler = logging.StreamHandler()
            handler.setLevel(level)
            handler.setFormatter(self._formatter_selector(level))
            self.addHandler(handler)
            self.debug("Stream handler attached")

    @staticmethod
    def _add_file_handler(self, filename: str, level: int = logging.DEBUG) -> None:
        handler = logging.FileHandler(filename)
        handler.setLevel(level)
        handler.setFormatter(self._formatter_selector(level))
        self.addHandler(handler)
        self.debug(f"File handler added: {filename}")

    @staticmethod
    def _add_text_widget_handler(self, text_widget: object) -> None:
        handler_cls = LoggerExt._get_text_handler()
        handler = handler_cls(text_widget)
        handler.setFormatter(self._formatter_selector(logging.DEBUG))
        self.addHandler(handler)
        self.debug("Text widget handler added")

    @staticmethod
    def _setup_logging_redirect(
        self, widget: object, level: int = logging.INFO
    ) -> None:
        handler_cls = LoggerExt._get_text_handler()

        # Prevent duplicate handlers for same widget
        for h in self.handlers:
            if isinstance(h, handler_cls) and getattr(h, "widget", None) is widget:
                self.debug("Text widget handler already registered")
                return

        handler = handler_cls(widget)
        handler.setFormatter(self._formatter_selector(level))
        self.addHandler(handler)
        self.setLevel(level)
        self.debug("Redirected logging output to widget")

    @staticmethod
    def _set_text_handler(handler: Union[type, object]) -> None:
        """Set a custom text handler class or instance."""
        LoggerExt._text_handler = handler

    @staticmethod
    def _get_text_handler() -> type:
        if LoggerExt._text_handler is None:
            return DefaultTextLogHandler
        if isinstance(LoggerExt._text_handler, type):
            return LoggerExt._text_handler  # it's a class
        return LoggerExt._text_handler.__class__  # it's an instance

    @staticmethod
    def _log_raw(self, message: str) -> None:
        """Write a raw message without level, prefix, or formatting."""
        # Write directly to all handler streams (console, files)
        for handler in self.handlers:
            stream = getattr(handler, "stream", None)
            if stream:
                try:
                    stream.write(message + "\n")
                    stream.flush()
                except Exception as e:
                    print(f"Logging error (raw write): {e}")
            else:  # For handlers without a stream (e.g., DefaultTextLogHandler), use emit
                try:
                    record = self.makeRecord(
                        name=self.name,
                        level=logging.INFO,
                        fn="",
                        lno=0,
                        msg=message,
                        args=None,
                        exc_info=None,
                    )
                    record.raw = True  # <-- ADD THIS
                    handler.emit(record)

                except Exception as e:
                    print(f"Logging error (raw emit): {e}")

    @staticmethod
    def _log_box(self, title: str, items: List[str] = None, align: str = "left") -> int:
        """Print an ASCII box with title and optional list of lines. Returns box width."""
        padding = 2
        content = [title] + (items or [])
        longest = max(len(line) for line in content)
        inner_width = longest + padding * 2
        width = inner_width + 2  # full box width including sides

        top = "╔" + "═" * inner_width + "╗"

        if align == "left":
            title_text = " " * padding + title.ljust(longest) + " " * padding
        elif align == "right":
            title_text = " " * padding + title.rjust(longest) + " " * padding
        else:  # center
            title_text = " " * padding + title.center(longest) + " " * padding

        mid = "║" + title_text + "║"
        sep = "╟" + "─" * inner_width + "╢"
        bottom = "╚" + "═" * inner_width + "╝"

        LoggerExt._log_raw(self, top)
        LoggerExt._log_raw(self, mid)

        if items:
            LoggerExt._log_raw(self, sep)
            for item in items:
                item_line = " " + item.ljust(inner_width - 1)
                LoggerExt._log_raw(self, f"║{item_line}║")

        LoggerExt._log_raw(self, bottom)

        return width

    @staticmethod
    def _log_divider(self, width: Optional[int] = None, char: str = "─") -> None:
        """Print a clean divider line. If width is given, use that."""
        if width is None:
            width = 60  # fallback default
        LoggerExt._log_raw(self, char * width)

    # Public API for the custom log levels (SUCCESS, RESULT, NOTICE)
    @staticmethod
    def _success(self, msg: str, *args, **kwargs) -> None:
        # Call the original log method to avoid recursion
        logging.Logger.__dict__["log"](
            self, LoggerExt.SUCCESS, f"{msg}", *args, **kwargs
        )

    @staticmethod
    def _result(self, msg: str, *args, **kwargs) -> None:
        logging.Logger.__dict__["log"](
            self, LoggerExt.RESULT, f"{msg}", *args, **kwargs
        )

    @staticmethod
    def _notice(self, msg: str, *args, **kwargs) -> None:
        logging.Logger.__dict__["log"](
            self, LoggerExt.NOTICE, f"{msg}", *args, **kwargs
        )


class DefaultTextLogHandler(internal_logging.Handler):
    """
    A generic thread-safe logging handler that writes logs to any widget
    supporting .append(str). Supports raw output, optional HTML color formatting,
    and optional monospace font styling.
    """

    def __init__(self, widget: object, use_html: bool = True, monospace: bool = False):
        super().__init__()
        self.widget = widget
        self.setLevel(internal_logging.NOTSET)
        self.use_html = use_html
        self.monospace = monospace

    def emit(self, record: internal_logging.LogRecord) -> None:
        try:
            if getattr(record, "raw", False):
                msg = record.getMessage()
                threading.Timer(0, self._safe_append, args=(msg,)).start()
            else:
                msg = self.format(record)
                if self.use_html:
                    color = self.get_color(record.levelname)
                    formatted = f'<span style="color:{color}">{msg}</span>'
                    if self.monospace:
                        formatted = f'<pre style="margin:0">{formatted}</pre>'
                    threading.Timer(0, self._safe_append, args=(formatted,)).start()
                else:
                    threading.Timer(0, self._safe_append, args=(msg,)).start()
        except Exception as e:
            print(f"DefaultTextLogHandler emit error: {e}")

    def _safe_append(self, formatted_msg: str) -> None:
        try:
            if hasattr(self.widget, "append"):
                self.widget.append(formatted_msg)
            elif hasattr(self.widget, "insert"):
                self.widget.insert("end", formatted_msg + "\n")
            else:
                print(formatted_msg)
        except Exception as e:
            print(f"DefaultTextLogHandler append error: {e}")

    def get_color(self, level: str) -> str:
        return {
            "DEBUG": "#AAAAAA",  # Neutral gray
            "INFO": "#FFFFFF",  # Pure white
            "WARNING": "#FFF5B7",  # Pastel yellow
            "ERROR": "#FFCCCC",  # Pastel pink
            "CRITICAL": "#CC3333",  # Strong red
            "SUCCESS": "#CCFFCC",  # Pastel green
            "RESULT": "#CCFFFF",  # Pastel teal
            "NOTICE": "#E5CCFF",  # Pastel lavender
        }.get(
            level, "#FFFFFF"
        )  # fallback: pure white


class LoggingMixin:
    """Mixin class for logging utilities.

    Provides a logger for each class and a shared class logger across instances.
    Includes methods for setting log levels, adding handlers, and redirecting logs.
    """

    _logger: internal_logging.Logger = None
    _class_logger = None

    @ClassProperty
    def logger(cls) -> internal_logging.Logger:
        if cls.__dict__.get("_logger") is None:
            name = f"{cls.__module__}.{cls.__qualname__}"
            logger = internal_logging.Logger(name, internal_logging.WARNING)
            logger.propagate = False
            logger.parent = None
            LoggerExt.patch(logger)

            if not logger.handlers:
                logger.add_stream_handler()

            cls._logger = logger

        # sync all handler levels to match logger level
        for handler in cls._logger.handlers:
            handler.setLevel(cls._logger.level)

        return cls._logger

    @classmethod
    def set_log_level(cls, level: int | str):
        """Set log level for the class logger and its handlers."""
        if isinstance(level, str):
            level = getattr(internal_logging, level.upper(), internal_logging.WARNING)

        cls.logger.setLevel(level)
        for handler in cls.logger.handlers:
            handler.setLevel(level)

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


# -------------------------------------------------------------------------------

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
                if isinstance(h, DefaultTextLogHandler)
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

# -------------------------------------------------------------------------------
