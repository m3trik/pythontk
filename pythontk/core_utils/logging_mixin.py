# !/usr/bin/python
# coding=utf-8
import threading
import logging as internal_logging
from typing import Union, List, Optional, Any
from pythontk.core_utils.class_property import ClassProperty


class LoggerExt:
    _text_handler = None  # Can be instance or class

    # Define custom log levels
    SUCCESS = 25
    RESULT = 35
    NOTICE = 45

    # Base log formats
    BASE_FORMATS = {
        "default": "[%(levelname)s] %(name)s: %(message)s",
        "debug": "[%(levelname)s] %(name)s: %(message)s",
        "success": "[%(levelname)s] %(message)s",
        "result": "[%(levelname)s] %(message)s",
        "notice": "[%(levelname)s] %(message)s",
    }

    @classmethod
    def patch(cls, logger: internal_logging.Logger) -> None:
        """Patch the logger with additional methods and setup."""
        if getattr(logger, "_logger_ext_patched", False):
            return
        logger._logger_ext_patched = True

        # Preserve the original setLevel method
        if not hasattr(logger, "internal_setLevel"):
            logger.internal_setLevel = logger.setLevel

        # Register custom levels
        cls._register_custom_levels()

        # Initialize error spam prevention
        logger._error_cache = {}
        logger._spam_prevention_enabled = True
        logger._cache_duration = 300  # 5 minutes default

        # Patch logger methods
        cls._patch_logger_methods(logger)

        # Initialize log formats and formatter selector
        logger._formatter_selector = cls._select_formatter.__get__(logger)
        logger.set_log_prefix = cls._set_log_prefix.__get__(logger)
        logger.set_log_suffix = cls._set_log_suffix.__get__(logger)
        logger._log_prefix = ""
        logger._log_suffix = ""
        logger._log_timestamp = None  # "%H:%M:%S" example of time only

        # Add property for log_timestamp that updates formatters
        def get_log_timestamp(self):
            return getattr(self, "_log_timestamp", None)

        def set_log_timestamp(self, value):
            self._log_timestamp = value
            LoggerExt._update_handler_formatters(self)

        logger.__class__.log_timestamp = property(get_log_timestamp, set_log_timestamp)

        # Add default handlers if none exist
        if not logger.handlers:
            cls._add_handler(logger, handler_type="stream")

    @staticmethod
    def _register_custom_levels() -> None:
        """Register custom log levels."""
        levels = {
            LoggerExt.SUCCESS: "SUCCESS",
            LoggerExt.RESULT: "RESULT",
            LoggerExt.NOTICE: "NOTICE",
        }
        for level, name in levels.items():
            internal_logging.addLevelName(level, name)

    @staticmethod
    def _patch_logger_methods(logger: internal_logging.Logger) -> None:
        """Patch logger with additional methods."""
        # Handle methods that don't need self (take no args or non-self first arg)
        direct_methods = {
            "set_text_handler": LoggerExt._set_text_handler,
            "get_text_handler": LoggerExt._get_text_handler,
        }
        for name, method in direct_methods.items():
            setattr(logger, name, method)

        # Handle methods that need self as first argument
        def make_method_wrapper(method):
            def wrapper(*args, **kwargs):
                return method(logger, *args, **kwargs)

            return wrapper

        wrapped_methods = {
            "setLevel": LoggerExt._set_level,
            "add_file_handler": LoggerExt._add_file_handler,
            "add_stream_handler": LoggerExt._add_stream_handler,
            "add_text_widget_handler": LoggerExt._add_text_widget_handler,
            "setup_logging_redirect": LoggerExt._setup_logging_redirect,
            "success": LoggerExt._success,
            "result": LoggerExt._result,
            "notice": LoggerExt._notice,
            "log_box": LoggerExt._log_box,
            "log_divider": LoggerExt._log_divider,
            "log_raw": LoggerExt._log_raw,
            "hide_logger_name": LoggerExt._hide_logger_name,
            "error_once": LoggerExt._error_once,
            "warning_once": LoggerExt._warning_once,
            "set_spam_prevention": LoggerExt._set_spam_prevention,
            "clear_error_cache": LoggerExt._clear_error_cache,
        }

        for name, method in wrapped_methods.items():
            setattr(logger, name, make_method_wrapper(method))

    @staticmethod
    def _select_formatter(self, level: int) -> internal_logging.Formatter:
        """Select formatter based on the log level."""
        prefix = getattr(self, "_log_prefix", "")
        suffix = getattr(self, "_log_suffix", "")
        base_format = LoggerExt._get_base_format(
            level, logger=self
        )  # Pass self as logger here
        format_string = base_format.replace(
            "%(message)s", f"{prefix}%(message)s{suffix}"
        )
        log_timestamp = getattr(self, "_log_timestamp", None)
        if log_timestamp:
            format_string = "[%(asctime)s] " + format_string
            formatter = internal_logging.Formatter(format_string, datefmt=log_timestamp)
        else:
            formatter = internal_logging.Formatter(format_string)
        return formatter

    @staticmethod
    def _get_base_format(level: int, logger=None) -> str:
        """Return the base format string based on the log level."""
        # Get the original format based on level
        if level == LoggerExt.SUCCESS:
            fmt = LoggerExt.BASE_FORMATS["success"]
        elif level == LoggerExt.RESULT:
            fmt = LoggerExt.BASE_FORMATS["result"]
        elif level == LoggerExt.NOTICE:
            fmt = LoggerExt.BASE_FORMATS["notice"]
        elif level <= internal_logging.DEBUG:
            fmt = LoggerExt.BASE_FORMATS["debug"]
        else:
            fmt = LoggerExt.BASE_FORMATS["default"]

        # If we have a logger instance and it has _hide_logger_name set to False,
        # strip the name from the format
        if logger and hasattr(logger, "_hide_logger_name") and logger._hide_logger_name:
            fmt = fmt.replace("%(name)s: ", "")

        return fmt

    @staticmethod
    def _add_handler(
        logger: internal_logging.Logger, handler_type: str, **kwargs
    ) -> None:
        """Add a handler to the logger."""
        handler = None
        if handler_type == "stream":
            handler = internal_logging.StreamHandler()
        elif handler_type == "file":
            handler = internal_logging.FileHandler(
                kwargs.get("filename", "logfile.log")
            )
        elif handler_type == "text_widget":
            handler_cls = LoggerExt._get_text_handler()
            handler = handler_cls(kwargs.get("widget"))

        if handler:
            level = kwargs.get("level", internal_logging.WARNING)
            handler.setLevel(level)
            handler.setFormatter(logger._formatter_selector(level))
            logger.addHandler(handler)
            logger.debug(f"{handler_type.capitalize()} handler added")

    @staticmethod
    def _add_file_handler(
        self, filename: str = "logfile.log", level: int = internal_logging.WARNING
    ) -> None:
        """Add a file handler to the logger."""
        LoggerExt._add_handler(
            self, handler_type="file", filename=filename, level=level
        )

    @staticmethod
    def _add_stream_handler(self, level: int = internal_logging.WARNING) -> None:
        """Add a stream handler to the logger."""
        LoggerExt._add_handler(self, handler_type="stream", level=level)

    @staticmethod
    def _add_text_widget_handler(
        self, text_widget: object, level: int = internal_logging.WARNING
    ) -> None:
        """Add a text widget handler to the logger."""
        LoggerExt._add_handler(
            self, handler_type="text_widget", widget=text_widget, level=level
        )

    @staticmethod
    def _set_log_prefix(self, prefix: str) -> None:
        """Set a prefix that will appear before all log messages."""
        self._log_prefix = prefix
        LoggerExt._update_handler_formatters(self)

    @staticmethod
    def _set_log_suffix(self, suffix: str) -> None:
        """Set a suffix that will appear after all log messages."""
        self._log_suffix = suffix
        LoggerExt._update_handler_formatters(self)

    @staticmethod
    def _update_handler_formatters(logger: internal_logging.Logger) -> None:
        """Update all handler formatters with the current prefix/suffix."""
        for handler in logger.handlers:
            handler.setFormatter(logger._formatter_selector(handler.level))

    @staticmethod
    def _set_level(self, level: Union[int, str]) -> None:
        """Set the log level."""
        if isinstance(level, str):
            level = internal_logging._nameToLevel.get(
                level.upper(), internal_logging.INFO
            )
        self.internal_setLevel(level)  # Call the preserved original method

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
                        level=internal_logging.INFO,
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
        internal_logging.Logger.__dict__["log"](
            self, LoggerExt.SUCCESS, f"{msg}", *args, **kwargs
        )

    @staticmethod
    def _result(self, msg: str, *args, **kwargs) -> None:
        internal_logging.Logger.__dict__["log"](
            self, LoggerExt.RESULT, f"{msg}", *args, **kwargs
        )

    @staticmethod
    def _notice(self, msg: str, *args, **kwargs) -> None:
        internal_logging.Logger.__dict__["log"](
            self, LoggerExt.NOTICE, f"{msg}", *args, **kwargs
        )

    @staticmethod
    def _set_log_prefix(self, prefix: str) -> None:
        """Set a prefix that will appear before all log messages."""
        self._log_prefix = prefix
        # Update all existing handlers with the new prefix
        for handler in self.handlers:
            handler.setFormatter(self._formatter_selector(handler.level))

    @staticmethod
    def _set_log_suffix(self, suffix: str) -> None:
        """Set a suffix that will appear after all log messages."""
        self._log_suffix = suffix
        # Update all existing handlers with the new suffix
        for handler in self.handlers:
            handler.setFormatter(self._formatter_selector(handler.level))

    @staticmethod
    def _setup_logging_redirect(
        self, target: Union[str, object], level: int = internal_logging.INFO
    ) -> None:
        """Redirect logging output to a specified target.
        :param target: Can be a filename (str), a stream (e.g., sys.stdout), or a widget (object).
        :param level: The log level for the redirection.
        """
        self.setLevel(level)  # <-- Always set logger level!
        if isinstance(target, str):
            self.add_file_handler(filename=target, level=level)
        elif hasattr(target, "write"):
            stream_handler = internal_logging.StreamHandler(stream=target)
            stream_handler.setLevel(level)
            stream_handler.setFormatter(self._formatter_selector(level))
            self.addHandler(stream_handler)
        elif hasattr(target, "append"):
            self.add_text_widget_handler(text_widget=target, level=level)
        else:
            raise ValueError("Unsupported target type for logging redirection.")

    @staticmethod
    def _hide_logger_name(self, show: bool = False) -> None:
        """Control whether the logger name is displayed in log messages.

        Args:
            show: If True (default), include the logger name in messages.
                  If False, omit the name portion for cleaner output.
        """
        self._hide_logger_name = show
        LoggerExt._update_handler_formatters(self)

    @staticmethod
    def _should_log_error(
        logger: internal_logging.Logger, message: str, cache_key: Optional[str] = None
    ) -> tuple[bool, str]:
        """Check if an error should be logged to prevent spam."""
        if not getattr(logger, "_spam_prevention_enabled", True):
            return True, ""

        import time

        # Generate cache key if not provided
        if cache_key is None:
            import hashlib

            cache_key = hashlib.md5(message.encode()).hexdigest()[:12]

        current_time = time.time()
        cache_duration = getattr(logger, "_cache_duration", 300)
        error_cache = getattr(logger, "_error_cache", {})

        # Check if we've seen this error recently
        if cache_key in error_cache:
            last_time, count = error_cache[cache_key]

            # If within cache duration, increment count and skip logging
            if current_time - last_time < cache_duration:
                error_cache[cache_key] = (last_time, count + 1)
                return (
                    False,
                    f" (suppressed {count} similar error{'s' if count != 1 else ''})",
                )

        # Log the error and cache it
        error_cache[cache_key] = (current_time, 1)
        return True, ""

    @staticmethod
    def _error_once(
        logger: internal_logging.Logger,
        message: str,
        *args,
        cache_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log an error with automatic spam prevention."""
        should_log, suffix = LoggerExt._should_log_error(logger, message, cache_key)

        if should_log:
            # Log the full error with suffix
            full_message = f"{message}{suffix}"
            logger.error(full_message, *args, **kwargs)
        else:
            # Log a brief debug message for suppressed errors
            logger.debug(f"Suppressed duplicate error: {message[:50]}...")

    @staticmethod
    def _warning_once(
        logger: internal_logging.Logger,
        message: str,
        *args,
        cache_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a warning with automatic spam prevention."""
        should_log, suffix = LoggerExt._should_log_error(logger, message, cache_key)

        if should_log:
            full_message = f"{message}{suffix}"
            logger.warning(full_message, *args, **kwargs)
        else:
            logger.debug(f"Suppressed duplicate warning: {message[:50]}...")

    @staticmethod
    def _set_spam_prevention(
        logger: internal_logging.Logger, enabled: bool = True, cache_duration: int = 300
    ) -> None:
        """Configure spam prevention settings."""
        logger._spam_prevention_enabled = enabled
        logger._cache_duration = cache_duration
        if not enabled:
            logger._error_cache.clear()

    @staticmethod
    def _clear_error_cache(logger: internal_logging.Logger) -> None:
        """Clear the error cache."""
        logger._error_cache.clear()


class DefaultTextLogHandler(internal_logging.Handler):
    """A generic thread-safe logging handler that writes logs to any widget
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


class TableMixin:
    """Mixin for formatting data as ASCII tables."""

    def format_table(
        self,
        data: List[List[Any]],
        headers: List[str],
        title: Optional[str] = None,
        col_max_width: int = 60,
    ) -> str:
        """Formats a list of lists as an ASCII table.

        Args:
            data: List of rows, where each row is a list of values.
            headers: List of column headers.
            title: Optional title for the table.
            col_max_width: Maximum width for any column.

        Returns:
            Formatted table string.
        """
        if not data:
            return ""

        # Ensure data matches headers
        num_cols = len(headers)
        processed_data = []
        for row in data:
            # Pad row if too short
            if len(row) < num_cols:
                row = list(row) + [""] * (num_cols - len(row))
            # Truncate row if too long
            elif len(row) > num_cols:
                row = row[:num_cols]
            processed_data.append([str(item) for item in row])

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in processed_data:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        # Clamp widths
        col_widths = [min(w, col_max_width) for w in col_widths]

        # Create format string
        # e.g. "{:<20} | {:<10} | {:<10}"
        fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])

        lines = []

        # Title
        if title:
            lines.append(title)
            lines.append("-" * len(title))

        # Header
        lines.append(fmt.format(*headers))
        lines.append("-+-".join(["-" * w for w in col_widths]))

        # Rows
        for row in processed_data:
            # Truncate values if needed
            trunc_row = []
            for i, val in enumerate(row):
                w = col_widths[i]
                if len(val) > w:
                    val = val[: w - 3] + "..."
                trunc_row.append(val)

            lines.append(fmt.format(*trunc_row))

        return "\n".join(lines)

    def log_table(
        self,
        data: List[List[Any]],
        headers: List[str],
        title: Optional[str] = None,
        level: str = "info",
    ) -> None:
        """Logs a formatted table.

        Args:
            data: List of rows.
            headers: List of column headers.
            title: Optional title.
            level: Logging level (info, warning, error, etc.)
        """
        table_str = self.format_table(data, headers, title)
        if not table_str:
            return

        # Log each line
        # Assumes self has a logger property or attribute
        logger = getattr(self, "logger", None)

        if logger:
            # If logger is a standard python logger
            log_method = getattr(logger, level.lower(), logger.info)

            # If using LoggerExt custom levels, handle them if passed as string
            if hasattr(logger, "log_raw"):
                # Use log_raw if available to avoid prefix duplication if any
                # But log_raw might not respect level.
                # Let's stick to standard logging for now, or check if log_raw exists.
                pass

            # For tables, we often want to print them raw without prefixes if possible,
            # or just log them line by line.

            # If the logger has a 'log_raw' method (from LoggerExt maybe?), use it?
            # LoggerExt doesn't seem to have log_raw in the snippet I read,
            # but SceneDiagnostics uses self.logger.log_raw.
            # Let's check if log_raw is available.
            if hasattr(logger, "log_raw"):
                # We rely on format_table to include the title if provided.
                lines = table_str.split("\n")
                for line in lines:
                    logger.log_raw(line)
            else:
                for line in table_str.split("\n"):
                    log_method(line)
        else:
            print(table_str)


class LoggingMixin(TableMixin):
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
            logger = internal_logging.Logger(name, internal_logging.NOTSET)  # CHANGED
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

    @ClassProperty
    def class_logger(cls) -> internal_logging.Logger:
        if cls.__dict__.get("_class_logger") is None:
            name = f"{cls.__module__}.{cls.__name__}.class"
            logger = internal_logging.getLogger(name)
            logger.setLevel(internal_logging.NOTSET)  # CHANGED
            logger.propagate = False
            LoggerExt.patch(logger)
            cls._class_logger = logger
        return cls._class_logger

    @ClassProperty
    def logging(cls):
        """Access to Python's internal logging module (aliased)."""
        return internal_logging

    @classmethod
    def set_log_level(cls, level: int | str):
        """Set log level for the class logger and its handlers."""
        if isinstance(level, str):
            level = getattr(internal_logging, level.upper(), internal_logging.WARNING)

        cls.logger.setLevel(level)
        for handler in cls.logger.handlers:
            handler.setLevel(level)


# -------------------------------------------------------------------------------

if __name__ == "__main__":
    ...

# -------------------------------------------------------------------------------
