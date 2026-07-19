# !/usr/bin/python
# coding=utf-8
"""Class-scoped logging toolkit.

``LoggerExt`` patches stdlib loggers with custom levels (PROGRESS/SUCCESS/
RESULT/NOTICE), HTML color presets, raw block output (boxes, groups,
dividers, tables), a managed file tee, and a capped in-memory ring buffer.
``LoggingMixin`` exposes one such patched logger per class.
"""
from __future__ import annotations

import os
import sys
import logging as internal_logging
import re
import unicodedata
from collections import deque
from typing import Union, List, Optional, Any
from pythontk.core_utils.class_property import ClassProperty


class StripHtmlFormatter(internal_logging.Formatter):
    """Formatter that strips HTML tags from the message."""

    def format(self, record):
        # Save original message
        original_msg = record.msg
        if (
            isinstance(original_msg, str)
            and "<" in original_msg
            and ">" in original_msg
        ):
            # Strip tags for this formatting operation
            record.msg = LoggerExt.strip_html(original_msg)

        # Format with stripped message
        formatted = super().format(record)

        # Restore original message (in case other handlers need the HTML)
        record.msg = original_msg
        return formatted


class LevelAwareFormatter(internal_logging.Formatter):
    """Formatter that dynamically selects format per-record based on log level.

    Unlike a static Formatter, this ensures RESULT/SUCCESS/NOTICE messages
    use their designated formats (without logger name) even when the handler
    accepts multiple levels.
    """

    def __init__(self, logger=None, strip_html=False):
        super().__init__()
        self._logger_ref = logger
        self._strip_html = strip_html

    def format(self, record):
        logger = self._logger_ref
        base_fmt = LoggerExt._get_base_format(record.levelno, logger=logger)
        prefix = getattr(logger, "_log_prefix", "") if logger else ""
        suffix = getattr(logger, "_log_suffix", "") if logger else ""
        fmt = base_fmt.replace("%(message)s", f"{prefix}%(message)s{suffix}")

        ts = getattr(logger, "_log_timestamp", None) if logger else None
        if ts:
            fmt = f"[%(asctime)s] {fmt}"
            self.datefmt = ts
        else:
            self.datefmt = None

        self._style._fmt = fmt

        if self._strip_html:
            original = record.msg
            if isinstance(original, str) and "<" in original and ">" in original:
                record.msg = LoggerExt.strip_html(original)
            result = super().format(record)
            record.msg = original
            return result

        return super().format(record)


class LoggerExt:
    _text_handler = None  # Can be instance or class

    # Define custom log levels
    PROGRESS = 15
    SUCCESS = 25
    RESULT = 35
    NOTICE = 45

    DEFAULT_BOX_WIDTH = 100

    # Default log colors (Hex)
    LOG_COLORS = {
        "DEBUG": "#AAAAAA",  # Neutral gray
        "INFO": "#FFFFFF",  # Pure white
        "PROGRESS": "#00CCFF",  # Cyan
        "WARNING": "#FFF5B7",  # Pastel yellow
        "ERROR": "#FFCCCC",  # Pastel pink
        "CRITICAL": "#CC3333",  # Strong red
        "SUCCESS": "#CCFFCC",  # Pastel green
        "RESULT": "#CCFFFF",  # Pastel teal
        "NOTICE": "#E5CCFF",  # Pastel lavender
    }

    # HTML Presets for formatting log messages.
    # Block-level tags (<h3>, <blockquote>, <div>) are avoided inside presets
    # because handlers wrap each record in an inline <span> — nesting a block
    # element inside an inline element is invalid and Qt's QTextDocument
    # renders it unpredictably. `header` keeps <h3> deliberately (it has
    # always been block-level and callers expect the larger heading style).
    HTML_PRESETS = {
        "default": '<span style="color:{color}">{message}</span>',
        "bold": '<span style="color:{color}; font-weight:bold">{message}</span>',
        "italic": '<span style="color:{color}; font-style:italic">{message}</span>',
        # ~1em top margin restores a single line of breathing room above
        # section headers without re-introducing the doubled gap that the
        # original `<br><h3>` produced.
        "header": '<h3 style="color:{color}; margin:1em 0 0.1em 0">{message}</h3>',
        "highlight": '<br><hl style="color:{color}">{message}</hl>',
        # Slack-style left rule: U+258E "▎" (left one-quarter block) in a
        # muted gray, paired with a softened message color. The bar reads
        # as a quiet column when stacked on consecutive lines in a
        # monospace widget; the message stays legible but recedes from
        # the prominence of section headers above it. {color} is
        # intentionally not used — the muted palette is fixed so the
        # blockquote always reads as secondary content regardless of the
        # log level it's emitted at.
        "blockquote": (
            '<span style="color:#666666">▎</span> '
            '<span style="color:#AAAAAA">{message}</span>'
        ),
    }

    # Default presets for log levels
    LEVEL_PRESETS = {
        "DEBUG": "italic",
        "CRITICAL": "bold",
    }

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

        # Initialize prefix/suffix/timestamp state
        logger.set_log_prefix = cls._set_log_prefix.__get__(logger)
        logger.set_log_suffix = cls._set_log_suffix.__get__(logger)
        logger._log_prefix = ""
        logger._log_suffix = ""
        logger._log_timestamp = None  # "%H:%M:%S" example of time only

        # ``log_timestamp`` property, scoped to this logger's patched class.
        cls._install_log_timestamp_property(logger)

        # Add default handlers if none exist
        if not logger.handlers:
            cls._add_handler(logger, handler_type="stream")

    # Per-base-class Logger subclasses carrying the ``log_timestamp``
    # property; all patched loggers of the same base share one subclass.
    _patched_logger_classes: dict = {}

    @classmethod
    def _install_log_timestamp_property(cls, logger: internal_logging.Logger) -> None:
        """Give *logger* a working ``log_timestamp`` property without touching
        the shared ``logging.Logger`` class.

        The instance is reassigned to a cached per-base subclass carrying the
        property. Installing the property on ``logger.__class__`` directly
        (the previous approach) planted a data descriptor on the GLOBAL
        Logger class, so assigning ``.log_timestamp`` on any foreign,
        unpatched logger ran our setter and replaced that logger's handler
        formatters.
        """
        base = type(logger)
        if getattr(base, "_logger_ext_class", False):
            return  # already a patched class
        patched = cls._patched_logger_classes.get(base)
        if patched is None:

            def _get_log_timestamp(self):
                return getattr(self, "_log_timestamp", None)

            def _set_log_timestamp(self, value):
                self._log_timestamp = value
                LoggerExt._update_handler_formatters(self)

            patched = type(
                f"LoggerExt{base.__name__}",
                (base,),
                {
                    "log_timestamp": property(
                        _get_log_timestamp, _set_log_timestamp
                    ),
                    "_logger_ext_class": True,
                },
            )
            cls._patched_logger_classes[base] = patched
        try:
            logger.__class__ = patched
        except TypeError:
            # Exotic Logger subclass whose layout forbids __class__
            # reassignment — fall back to the legacy in-place property.
            base.log_timestamp = patched.log_timestamp

    @staticmethod
    def _register_custom_levels() -> None:
        """Register custom log levels."""
        levels = {
            LoggerExt.PROGRESS: "PROGRESS",
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
            "log_link": LoggerExt._log_link,
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
            "set_log_file": LoggerExt._set_log_file,
            "enable_log_buffer": LoggerExt._enable_log_buffer,
            "disable_log_buffer": LoggerExt._disable_log_buffer,
            "clear_log_buffer": LoggerExt._clear_log_buffer,
            "dump_log": LoggerExt._dump_log,
            "setup_logging_redirect": LoggerExt._setup_logging_redirect,
            "get_redirect_width": LoggerExt._get_redirect_width,
            "success": LoggerExt._success,
            "result": LoggerExt._result,
            "notice": LoggerExt._notice,
            "progress": LoggerExt._progress,
            "log_box": LoggerExt._log_box,
            "log_divider": LoggerExt._log_divider,
            "log_group": LoggerExt._log_group,
            "log_raw": LoggerExt._log_raw,
            "hide_logger_name": LoggerExt._hide_logger_name,
            "error_once": LoggerExt._error_once,
            "warning_once": LoggerExt._warning_once,
            "set_spam_prevention": LoggerExt._set_spam_prevention,
            "clear_error_cache": LoggerExt._clear_error_cache,
            "info": LoggerExt._info,
            "debug": LoggerExt._debug,
            "warning": LoggerExt._warning,
            "error": LoggerExt._error,
            "critical": LoggerExt._critical,
        }

        for name, method in wrapped_methods.items():
            setattr(logger, name, make_method_wrapper(method))

    @staticmethod
    def _log_custom(logger, level_int, msg, *args, **kwargs):
        """Log with optional custom formatting via explicit ``preset=`` /
        ``color=`` kwargs.

        Positional args are always treated as stdlib ``%``-format arguments —
        no styling heuristic. (The old heuristic silently swallowed ``%s``
        substitution whenever an argument happened to match a preset or
        color name like ``"default"`` or ``"error"``.)
        """
        if not logger.isEnabledFor(level_int):
            return  # skip styling work for records that will be dropped

        preset = kwargs.pop("preset", None)
        color_level = kwargs.pop("color", None)

        if preset or color_level:
            if not color_level:
                color_level = internal_logging.getLevelName(level_int)
            msg = LoggerExt.format_message_as_html(msg, color_level, preset)

        internal_logging.Logger.log(logger, level_int, msg, *args, **kwargs)

    @staticmethod
    def _info(logger, msg, *args, **kwargs):
        LoggerExt._log_custom(logger, internal_logging.INFO, msg, *args, **kwargs)

    @staticmethod
    def _debug(logger, msg, *args, **kwargs):
        LoggerExt._log_custom(logger, internal_logging.DEBUG, msg, *args, **kwargs)

    @staticmethod
    def _warning(logger, msg, *args, **kwargs):
        LoggerExt._log_custom(logger, internal_logging.WARNING, msg, *args, **kwargs)

    @staticmethod
    def _error(logger, msg, *args, **kwargs):
        LoggerExt._log_custom(logger, internal_logging.ERROR, msg, *args, **kwargs)

    @staticmethod
    def _critical(logger, msg, *args, **kwargs):
        LoggerExt._log_custom(logger, internal_logging.CRITICAL, msg, *args, **kwargs)

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

        # Strip the logger name from the format when hidden.
        if logger is not None and getattr(logger, "_hide_logger_name", False):
            fmt = fmt.replace("%(name)s: ", "")

        return fmt

    @staticmethod
    def _add_handler(
        logger: internal_logging.Logger, handler_type: str, **kwargs
    ) -> None:
        """Add a handler to the logger, skipping if an equivalent one exists."""
        # Deduplicate: check for an existing handler of the same type
        # targeting the same widget / file / stream.
        target_widget = kwargs.get("widget")
        target_stream = kwargs.get("stream")
        for existing in logger.handlers:
            if handler_type == "text_widget" and target_widget is not None:
                if getattr(existing, "widget", None) is target_widget:
                    return  # already attached
            elif handler_type == "file":
                target_file = os.path.abspath(kwargs.get("filename", "logfile.log"))
                if (
                    isinstance(existing, internal_logging.FileHandler)
                    and getattr(existing, "baseFilename", None) == target_file
                ):
                    return
            elif handler_type == "stream":
                # Exact type check: FileHandler subclasses StreamHandler.
                resolved = target_stream if target_stream is not None else sys.stderr
                if (
                    type(existing) is internal_logging.StreamHandler
                    and existing.stream is resolved
                ):
                    return

        handler = None
        if handler_type == "stream":
            handler = internal_logging.StreamHandler(stream=target_stream)
        elif handler_type == "file":
            handler = internal_logging.FileHandler(
                kwargs.get("filename", "logfile.log")
            )
        elif handler_type == "text_widget":
            handler_cls = LoggerExt._get_text_handler()
            # Check if the handler accepts monospace argument
            import inspect

            sig = inspect.signature(handler_cls)
            handler_kwargs = {"widget": kwargs.get("widget")}
            if "monospace" in sig.parameters:
                handler_kwargs["monospace"] = kwargs.get("monospace", True)
            if "use_html" in sig.parameters:
                handler_kwargs["use_html"] = kwargs.get("use_html", True)

            handler = handler_cls(**handler_kwargs)

        if handler:
            level = kwargs.get("level", internal_logging.WARNING)
            handler.setLevel(level)

            # Use level-aware formatter with HTML stripping for file/stream
            strip_html = handler_type in ["file", "stream"]
            handler.setFormatter(
                LevelAwareFormatter(logger=logger, strip_html=strip_html)
            )

            logger.addHandler(handler)

    @staticmethod
    def _add_file_handler(
        self, filename: str = "logfile.log", level: int = internal_logging.WARNING
    ) -> None:
        """Add a file handler to the logger."""
        LoggerExt._add_handler(
            self, handler_type="file", filename=filename, level=level
        )

    @staticmethod
    def _add_stream_handler(
        self,
        level: int = internal_logging.WARNING,
        stream: Optional[object] = None,
    ) -> None:
        """Add a stream handler (``sys.stderr`` when *stream* is ``None``).

        Deduplicated per target stream — repeat calls do not stack
        duplicate console output.
        """
        LoggerExt._add_handler(self, handler_type="stream", level=level, stream=stream)

    @staticmethod
    def _add_text_widget_handler(
        self,
        text_widget: object,
        level: int = internal_logging.WARNING,
        monospace: bool = True,
    ) -> None:
        """Add a text widget handler to the logger."""
        LoggerExt._add_handler(
            self,
            handler_type="text_widget",
            widget=text_widget,
            level=level,
            monospace=monospace,
        )

    @staticmethod
    def _update_handler_formatters(logger: internal_logging.Logger) -> None:
        """Update all handler formatters with level-aware formatting."""
        for handler in logger.handlers:
            is_file_or_stream = isinstance(
                handler,
                (internal_logging.FileHandler, internal_logging.StreamHandler),
            ) and not isinstance(handler, DefaultTextLogHandler)

            handler.setFormatter(
                LevelAwareFormatter(logger=logger, strip_html=is_file_or_stream)
            )

    @staticmethod
    def _set_level(self, level: Union[int, str]) -> None:
        """Set the log level and sync all unpinned handler levels."""
        level = LoggerExt._coerce_level(level, default=internal_logging.INFO)
        self.internal_setLevel(level)  # Call the preserved original method
        # These loggers are built via the Logger() constructor (see the `logger`
        # property), so they are NOT in manager.loggerDict and the stdlib
        # setLevel's manager._clear_cache() never reaches this logger's own
        # isEnabledFor cache. Clear it directly so a level change re-evaluates
        # every level — otherwise a custom level (SUCCESS/RESULT/NOTICE) whose
        # isEnabledFor was cached False while the level was high stays silently
        # disabled after the level is lowered again.
        self._cache.clear()
        # Sync handler levels whenever the level changes — except handlers
        # whose level was explicitly pinned by the caller (a file tee or
        # ring buffer given its own level must not start flooding when the
        # logger is later opened up to DEBUG).
        for handler in self.handlers:
            if getattr(handler, "_pinned_level", False):
                continue
            handler.setLevel(level)

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
                    # Mirror the formatter pipeline's HTML-stripping decision.
                    # The default stream/file handlers attach
                    # ``LevelAwareFormatter(strip_html=True)``; without this
                    # second branch, raw output (boxes, dividers) leaks
                    # ``<span>`` markup into terminals and log files.
                    formatter = handler.formatter
                    should_strip = isinstance(formatter, StripHtmlFormatter) or (
                        isinstance(formatter, LevelAwareFormatter)
                        and getattr(formatter, "_strip_html", False)
                    )
                    msg_to_write = (
                        LoggerExt.strip_html(message) if should_strip else message
                    )

                    # Serialize with concurrent emits on the same handler.
                    handler.acquire()
                    try:
                        stream.write(msg_to_write + "\n")
                        stream.flush()
                    finally:
                        handler.release()
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
                    record.raw = True
                    # emit() is called directly (raw output bypasses level
                    # filtering by design) — take the handler lock ourselves,
                    # as Handler.handle() would for a normal record.
                    handler.acquire()
                    try:
                        handler.emit(record)
                    finally:
                        handler.release()
                except Exception as e:
                    print(f"Logging error (raw emit): {e}")

    @staticmethod
    def _char_width(ch: str) -> int:
        """Return the display/column width of a single character.

        Uses ``wcwidth`` if available, otherwise falls back to a heuristic
        based on ``unicodedata.east_asian_width`` plus known wide symbol
        ranges (Dingbats, Miscellaneous Symbols, emoji, etc.) whose glyphs
        typically occupy two columns in modern monospace fonts even though
        Unicode classifies them as narrow or ambiguous.
        """
        try:
            import wcwidth as _wcwidth

            w = _wcwidth.wcwidth(ch)
            return max(w, 0)
        except Exception:
            pass

        cp = ord(ch)
        # East Asian Fullwidth / Wide → always 2
        eaw = unicodedata.east_asian_width(ch)
        if eaw in ("W", "F"):
            return 2
        # Common symbol blocks that render as 2 columns in many terminals
        # (Dingbats, Misc Symbols, Misc Symbols & Arrows, Supplemental Arrows-B)
        if (
            0x2600 <= cp <= 0x27BF  # Misc Symbols + Dingbats
            or 0x2B50 <= cp <= 0x2B55  # stars, circles
            or 0x1F300
            <= cp
            <= 0x1FAFF  # Misc Symbols & Pictographs … Symbols Extended-A
            or 0xFE00 <= cp <= 0xFE0F  # Variation Selectors
            or 0x200D == cp  # ZWJ
        ):
            return 2
        return 1

    # Regex for stripping HTML log markup (spans, links, presets).
    _HTML_TAG_RE = re.compile(r"<[^>]+>")

    @classmethod
    def strip_html(cls, text: str) -> str:
        """Remove HTML tags from *text*, leaving the visible plain text.

        The single strip used everywhere log markup meets a plain-text
        sink (stream/file formatters, raw writes, buffer dumps, width
        measurement).
        """
        return cls._HTML_TAG_RE.sub("", text)

    @staticmethod
    def _display_width(text: str) -> int:
        """Return the display/column width of *text*.

        HTML tags (``<span ...>``, ``<a ...>``, etc.) are stripped before
        measuring so that embedded markup does not inflate the count.
        """
        visible = LoggerExt.strip_html(text)
        return sum(LoggerExt._char_width(ch) for ch in visible)

    @staticmethod
    def _pad(text: str, target_width: int, fill: str = " ", align: str = "left") -> str:
        """Pad *text* to *target_width* display columns using *fill* char."""
        current = LoggerExt._display_width(text)
        deficit = max(target_width - current, 0)
        if align == "right":
            return fill * deficit + text
        elif align == "center":
            left = deficit // 2
            right = deficit - left
            return fill * left + text + fill * right
        else:  # left
            return text + fill * deficit

    @staticmethod
    def _truncate(text: str, max_display_width: int, ellipsis: str = "…") -> str:
        """Truncate *text* to *max_display_width* display columns with ellipsis."""
        dw = LoggerExt._display_width
        if dw(text) <= max_display_width:
            return text
        ellipsis_w = dw(ellipsis)
        result = []
        current_width = 0
        for ch in text:
            ch_w = LoggerExt._char_width(ch)
            if current_width + ch_w + ellipsis_w > max_display_width:
                break
            result.append(ch)
            current_width += ch_w
        return "".join(result) + ellipsis

    @staticmethod
    def _hard_wrap_word(word: str, max_display_width: int) -> tuple:
        """Hard-wrap a single word that exceeds *max_display_width*.

        Returns ``(complete_lines, remaining_fragment, remaining_width)``.
        """
        lines = []
        chars = []
        w = 0
        for ch in word:
            ch_w = LoggerExt._char_width(ch)
            if w + ch_w > max_display_width:
                lines.append("".join(chars))
                chars = [ch]
                w = ch_w
            else:
                chars.append(ch)
                w += ch_w
        return lines, "".join(chars), w

    # Sentinel that never appears in real text, used to protect spaces
    # inside HTML tags from being split by _wrap_text.
    _TAG_SPACE = "\x00"

    @staticmethod
    def _wrap_text(text: str, max_display_width: int) -> List[str]:
        """Wrap *text* to fit within *max_display_width* display columns.

        Breaks at word boundaries when possible, otherwise hard-wraps.
        HTML tags are protected so that spaces inside attributes are never
        used as break points.
        Returns a list of wrapped lines.
        """
        dw = LoggerExt._display_width
        if dw(text) <= max_display_width:
            return [text]

        # Protect spaces inside HTML tags by replacing them with a sentinel
        sentinel = LoggerExt._TAG_SPACE
        has_tags = "<" in text and ">" in text
        if has_tags:

            def _protect_tag_spaces(m: "re.Match") -> str:
                return m.group(0).replace(" ", sentinel)

            text = LoggerExt._HTML_TAG_RE.sub(_protect_tag_spaces, text)

        words = text.split(" ")
        lines = []
        current_line = ""
        current_width = 0

        for word in words:
            word_width = dw(word)
            # Words containing HTML tags must never be hard-wrapped
            # (splitting characters inside tags produces broken markup).
            word_has_tag = has_tags and "<" in word
            if current_width == 0:
                if word_width <= max_display_width or word_has_tag:
                    current_line = word
                    current_width = word_width
                else:
                    extra, current_line, current_width = LoggerExt._hard_wrap_word(
                        word, max_display_width
                    )
                    lines.extend(extra)
            elif current_width + 1 + word_width <= max_display_width:
                current_line += " " + word
                current_width += 1 + word_width
            else:
                lines.append(current_line)
                if word_width <= max_display_width or word_has_tag:
                    current_line = word
                    current_width = word_width
                else:
                    extra, current_line, current_width = LoggerExt._hard_wrap_word(
                        word, max_display_width
                    )
                    lines.extend(extra)

        if current_line:
            lines.append(current_line)

        # Restore protected spaces inside HTML tags
        if has_tags:
            lines = [line.replace(sentinel, " ") for line in lines]

        return lines

    @staticmethod
    def _log_box(
        self,
        title: str,
        items: List[str] = None,
        align: str = "left",
        level: str = None,
        max_width: Optional[int] = None,
        bg: Optional[str] = None,
    ) -> int:
        """Print an ASCII box with title and optional list of lines. Returns box width.

        Parameters:
            max_width: Maximum box width in display columns.  Falls back to
                ``self.box_width`` if set, then to the narrowest column
                count reported by attached handlers (see
                ``get_redirect_width``), otherwise ``DEFAULT_BOX_WIDTH``.
            bg: Solid background color for the box. Accepts a log-level name
                (``"ERROR"``, ``"SUCCESS"``…) which resolves via
                ``LOG_COLORS``, or any CSS color string (``"#222"``,
                ``"steelblue"``, ``"rgb(40,40,40)"``). Each box row is
                wrapped in its own span so the background renders as a
                contiguous block in HTML handlers; ignored by plain-text
                handlers (HTML is stripped).
        """
        padding = 1
        # Use non-breaking space to prevent HTML space collapsing in handlers
        space = "\u00a0"

        if max_width is None:
            max_width = getattr(self, "box_width", None)
        if max_width is None:
            max_width = LoggerExt._get_redirect_width(self)
        if max_width is None:
            max_width = LoggerExt.DEFAULT_BOX_WIDTH

        dw = LoggerExt._display_width
        wrap = LoggerExt._wrap_text
        content = [title] + (items or [])
        longest = max(dw(line) for line in content)

        # Clamp to max_width (subtract 2 for borders, 2 for padding)
        max_content = max_width - 2 - padding * 2
        if max_content < 4:
            max_content = 4  # minimum usable width

        needs_wrap = longest > max_content
        all_wrapped_title = None
        all_wrapped_items = None

        # Pre-wrap all content to the clamped width so we can measure the
        # actual longest wrapped line and shrink the box to fit.
        if needs_wrap:
            wrap_width = max_content
            all_wrapped_title = wrap(title, wrap_width)
            all_wrapped_items = []
            for item in (items or []):
                all_wrapped_items.append(wrap(item, wrap_width))

            # Recalculate longest from the wrapped output
            longest = 0
            for wl in all_wrapped_title:
                w = dw(wl)
                if w > longest:
                    longest = w
            for wrapped_group in all_wrapped_items:
                for wl in wrapped_group:
                    w = dw(wl)
                    if w > longest:
                        longest = w

        inner_width = longest + padding * 2
        width = inner_width + 2  # full box width including sides

        top = "╔" + "═" * inner_width + "╗"

        # Wrap title lines to fit
        title_lines = all_wrapped_title if needs_wrap else wrap(title, longest)
        title_rows = []
        for tl in title_lines:
            tl_padded = LoggerExt._pad(tl, longest, fill=space, align=align)
            title_rows.append("║" + space * padding + tl_padded + space * padding + "║")

        sep = "╟" + "─" * inner_width + "╢"
        bottom = "╚" + "═" * inner_width + "╝"

        lines = [top] + title_rows
        if items:
            lines.append(sep)
            for idx, item in enumerate(items):
                wrapped = all_wrapped_items[idx] if needs_wrap else wrap(item, inner_width - 1)
                for wl in wrapped:
                    item_padded = LoggerExt._pad(wl, inner_width - 1, fill=space)
                    item_line = space + item_padded
                    lines.append(f"║{item_line}║")
        lines.append(bottom)

        text_color = LoggerExt._resolve_color(level) if level else None
        bg_color = LoggerExt._resolve_color(bg) if bg else None

        # Box drawing chars (╔ ═ ║ ╚) only align when rendered in a
        # monospace cell. HTML viewers (QTextEdit / browser) often inherit
        # a proportional font from outer wrappers, which collapses the
        # column math. Carry the font-family inline on each span so the
        # boxes render correctly regardless of enclosing context.
        mono_css = (
            "font-family:'Consolas','Courier New',Monaco,monospace;white-space:pre"
        )

        if bg_color:
            # Per-line wrapping: a single span across "\n" does not extend its
            # background through line breaks in HTML rendering, so each row
            # needs its own span to render as a contiguous solid block.
            style_parts = [mono_css]
            if text_color:
                style_parts.append(f"color:{text_color}")
            style_parts.append(f"background-color:{bg_color}")
            style = ";".join(style_parts)
            lines = [f'<span style="{style}">{ln}</span>' for ln in lines]
            box_text = "\n".join(lines)
        else:
            box_text = "\n".join(lines)
            if text_color:
                box_text = (
                    f'<span style="color:{text_color};{mono_css}">{box_text}</span>'
                )

        LoggerExt._log_raw(self, box_text)

        return width

    @staticmethod
    def _log_link(text: str, action: str, **params: str) -> str:
        """Return an HTML ``<a>`` tag for embedding clickable links in log messages.

        The link uses a custom ``action://`` URI scheme that is never opened
        by a browser — handlers (e.g. ``QTextBrowser.anchorClicked``) parse
        the URL and dispatch the action.

        Parameters:
            text:   Visible link label (HTML-escaped automatically).
            action: Action verb (e.g. ``"select"``, ``"reveal"``).
            **params: Arbitrary key-value pairs appended as query string.

        Returns:
            An ``<a href="action://ACTION?k=v&…">text</a>`` string that can
            be embedded in any log message via f-string::

                link = logger.log_link("pCube1", "select", node="|group1|pCube1")
                logger.info(f"Missing object: {link}")
        """
        import html
        from urllib.parse import urlencode

        safe_text = html.escape(text, quote=False)
        query = urlencode(params) if params else ""
        href = f"action://{action}?{query}" if query else f"action://{action}"
        # No spaces in the tag — _wrap_text splits on spaces and would
        # break the tag if any are present inside attributes.
        return f'<a href="{href}" style="text-decoration:underline">' f"{safe_text}</a>"

    @staticmethod
    def _log_group(
        self,
        title: str,
        items: List[str],
        level: str = "INFO",
        item_color: str = "#888888",
        indent: int = 2,
    ) -> None:
        """Emit a bold title + indented item list as one log entry.

        Renders as a single visual block in HTML widgets — no per-line
        ``[LEVEL] name:`` prefix, no paragraph margin between items — and
        as a clean header + indented lines in stripped text output
        (console, files).

        Use when several related lines belong together (e.g. a category
        header with its members) rather than as separate log records that
        each pick up the standard prefix and paragraph spacing.

        Parameters:
            title:      The group header text.
            items:      The lines listed under the title.
            level:      Log level name (``"INFO"``, ``"SUCCESS"``…) — only
                        used to colour the title; items use ``item_color``.
            item_color: CSS colour for item lines and the left-rule bar;
                        muted gray by default so items recede visually
                        from the title.
            indent:     Total leading-column position of each item line.
                        The first column is occupied by a U+258E "▎" left
                        one-quarter block which acts as a Slack-style
                        blockquote rule; remaining columns are spaces.
                        Because the whole group is one ``log_raw`` record
                        (single QTextBlock with ``white-space:pre``), the
                        bar characters stack into a continuous vertical
                        line in monospace — no inter-paragraph gaps.
        """
        if not items:
            LoggerExt._log_raw(self, title)
            return

        title_color = LoggerExt.get_color(level)
        # Bar at column 0; pad fills the remaining (indent - 1) columns so
        # the item text starts at the requested indent column.
        pad = " " * max(indent - 1, 0)
        item_lines = "\n".join(
            f'<span style="color:{item_color}">▎{pad}{item}</span>' for item in items
        )
        # Leading "\n" gives one blank line above each group so consecutive
        # groups (and a group following a regular log line) read as
        # separate visual chunks. The outer handler wrapper uses
        # white-space:pre, so the newline renders as a hard line break in
        # widgets and survives HTML stripping for console/file output.
        html = (
            f'\n<span style="color:{title_color}; font-weight:bold">{title}</span>\n'
            f"{item_lines}"
        )
        LoggerExt._log_raw(self, html)

    @staticmethod
    def _log_divider(self, width: Optional[int] = None, char: str = "─") -> None:
        """Print a clean divider line. If width is given, use that."""
        if width is None:
            width = 60  # fallback default
        LoggerExt._log_raw(self, char * width)

    # Public API for the custom log levels — routed through _log_custom so
    # they accept the same ``preset=`` / ``color=`` styling kwargs as
    # info/debug/warning/error/critical.
    @staticmethod
    def _success(self, msg: str, *args, **kwargs) -> None:
        LoggerExt._log_custom(self, LoggerExt.SUCCESS, msg, *args, **kwargs)

    @staticmethod
    def _result(self, msg: str, *args, **kwargs) -> None:
        LoggerExt._log_custom(self, LoggerExt.RESULT, msg, *args, **kwargs)

    @staticmethod
    def _notice(self, msg: str, *args, **kwargs) -> None:
        LoggerExt._log_custom(self, LoggerExt.NOTICE, msg, *args, **kwargs)

    @staticmethod
    def _progress(self, msg: str, *args, **kwargs) -> None:
        LoggerExt._log_custom(self, LoggerExt.PROGRESS, msg, *args, **kwargs)

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
    def _setup_logging_redirect(
        self,
        target: Union[str, object],
        level: int = internal_logging.INFO,
        monospace: bool = True,
    ) -> None:
        """Redirect logging output to a specified target.
        :param target: Can be a filename (str), a stream (e.g., sys.stdout), or a widget (object).
        :param level: The log level for the redirection.
        :param monospace: Whether to use monospace font for text widgets.
        """
        self.setLevel(level)  # <-- Always set logger level!
        if isinstance(target, str):
            self.add_file_handler(filename=target, level=level)
        elif hasattr(target, "write"):
            self.add_stream_handler(level=level, stream=target)
        elif hasattr(target, "append"):
            self.add_text_widget_handler(
                text_widget=target, level=level, monospace=monospace
            )
        else:
            raise ValueError("Unsupported target type for logging redirection.")

    @staticmethod
    def _get_redirect_width(self) -> Optional[int]:
        """Return the narrowest column count reported by attached handlers.

        Sources, in priority order per handler:
        1. A custom ``available_columns()`` method on the handler (host
           integrations can supply this to report e.g. viewport width).
        2. For ``StreamHandler`` whose stream is a TTY, the live terminal
           width via ``os.get_terminal_size``. Files are skipped since
           ``isatty()`` is False for them.

        The minimum across all reporting handlers is returned so a single
        box fits every redirect target without being wrapped by terminal
        autowrap. Returns ``None`` when no handler reports a width.
        """
        widths = []
        for handler in self.handlers:
            fn = getattr(handler, "available_columns", None)
            if callable(fn):
                try:
                    w = fn()
                except Exception:
                    w = None
                if w and w > 0:
                    widths.append(int(w))
                    continue

            stream = getattr(handler, "stream", None)
            if stream is None:
                continue
            try:
                if not stream.isatty():
                    continue
                size = os.get_terminal_size(stream.fileno())
            except (OSError, AttributeError, ValueError):
                continue
            if size.columns and size.columns > 0:
                widths.append(int(size.columns))
        return min(widths) if widths else None

    @staticmethod
    def _hide_logger_name(self, hide: bool = True) -> None:
        """Control whether the logger name is displayed in log messages.

        Args:
            hide: If True (default), omit the logger name for cleaner output.
                  If False, include the logger name in messages.
        """
        self._hide_logger_name = hide
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

            # Within the window: count it and suppress silently. The caller
            # emits its own brief debug line; the accumulated count is
            # surfaced later, on the first genuine emission after the window
            # expires (below), not on this discarded drop path.
            if current_time - last_time < cache_duration:
                error_cache[cache_key] = (last_time, count + 1)
                return False, ""

            # Window expired: reset the entry and surface how many were
            # suppressed since the last real emission. The first occurrence
            # was emitted, so the suppressed total is count - 1.
            error_cache[cache_key] = (current_time, 1)
            suppressed = count - 1
            if suppressed > 0:
                return True, (
                    f" (suppressed {suppressed} similar "
                    f"error{'s' if suppressed != 1 else ''})"
                )
            return True, ""

        # First time we've seen this error — log it and cache it.
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

    # ------------------------------------------------------------------
    # File tee + in-memory ring buffer (optional, off by default)
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_level(
        level: Union[int, str], default: int = internal_logging.NOTSET
    ) -> int:
        """Map a level name to its int (*default* for unknown names);
        pass ints through unchanged."""
        if isinstance(level, str):
            return internal_logging._nameToLevel.get(level.upper(), default)
        return level

    @staticmethod
    def _set_log_file(
        logger: internal_logging.Logger,
        filename: Optional[str],
        level: Union[int, str] = internal_logging.NOTSET,
    ) -> Optional[internal_logging.FileHandler]:
        """Tee every record this logger emits to *filename* (continuous).

        Manages exactly one "log file" handler per logger so the call is a
        clean on/off toggle: passing a path attaches (replacing any prior
        managed file), passing ``None`` detaches and closes it. Off by
        default — until called there is zero file I/O.

        The handler defaults to ``NOTSET`` so it captures whatever passes
        the logger's level (control verbosity with ``set_log_level``). An
        explicit *level* is pinned: later ``set_log_level`` calls do not
        override it. Output is plain text (HTML stripped), matching the
        console/file formatter pipeline. Returns the handler (or ``None``
        when detached).
        """
        existing = getattr(logger, "_managed_file_handler", None)
        if existing is not None:
            # Detach before closing: a concurrent emit must never reach a
            # handler whose stream is already closed (I/O on closed file).
            logger.removeHandler(existing)
            existing.close()
            logger._managed_file_handler = None

        if filename is None:
            return None

        coerced = LoggerExt._coerce_level(level)
        handler = internal_logging.FileHandler(filename)
        handler.setLevel(coerced)
        handler._pinned_level = coerced != internal_logging.NOTSET
        handler.setFormatter(LevelAwareFormatter(logger=logger, strip_html=True))
        logger.addHandler(handler)
        logger._managed_file_handler = handler
        return handler

    @staticmethod
    def _enable_log_buffer(
        logger: internal_logging.Logger,
        capacity: int = 2000,
        level: Union[int, str] = internal_logging.NOTSET,
    ) -> "RingBufferHandler":
        """Start capturing records into a capped in-memory ring buffer.

        Emit is O(1) and does no string formatting (records are stored by
        reference and rendered only on ``dump_log``), so an enabled buffer
        adds negligible hot-path cost. Oldest records drop once *capacity*
        is exceeded. Re-calling re-sizes in place, preserving the most
        recent records. An explicit *level* is pinned against later
        ``set_log_level`` syncs. Returns the handler.
        """
        coerced = LoggerExt._coerce_level(level)
        existing = getattr(logger, "_ring_buffer_handler", None)
        if existing is not None:
            if existing.capacity != capacity:
                existing.buffer = deque(existing.buffer, maxlen=capacity)
                existing.capacity = capacity
            existing.setLevel(coerced)
            existing._pinned_level = coerced != internal_logging.NOTSET
            return existing

        handler = RingBufferHandler(capacity=capacity, level=coerced)
        handler._pinned_level = coerced != internal_logging.NOTSET
        handler.setFormatter(LevelAwareFormatter(logger=logger, strip_html=True))
        logger.addHandler(handler)
        logger._ring_buffer_handler = handler
        return handler

    @staticmethod
    def _disable_log_buffer(logger: internal_logging.Logger) -> None:
        """Stop ring-buffer capture and discard buffered records."""
        handler = getattr(logger, "_ring_buffer_handler", None)
        if handler is not None:
            logger.removeHandler(handler)
            handler.clear()
            handler.close()  # drop from logging's module-level handler registry
            logger._ring_buffer_handler = None

    @staticmethod
    def _clear_log_buffer(logger: internal_logging.Logger) -> None:
        """Drop buffered records but keep capturing."""
        handler = getattr(logger, "_ring_buffer_handler", None)
        if handler is not None:
            handler.clear()

    @staticmethod
    def _dump_log(
        logger: internal_logging.Logger,
        target: Union[str, object, None] = None,
        mode: str = "w",
        encoding: str = "utf-8",
    ) -> str:
        """Render the ring buffer to plain text and return it.

        *target* may be a file path (str), any object with ``write`` (a
        stream), or ``None`` to only return the text. Requires
        ``enable_log_buffer`` first; otherwise warns once and returns ``""``.
        """
        handler = getattr(logger, "_ring_buffer_handler", None)
        if handler is None:
            logger.warning_once(
                "dump_log() called but no log buffer is enabled; "
                "call enable_log_buffer() first."
            )
            return ""

        formatter = LevelAwareFormatter(logger=logger, strip_html=True)
        text = handler.format_records(formatter)
        payload = text + "\n" if text else ""

        if isinstance(target, str):
            with open(target, mode, encoding=encoding) as fh:
                fh.write(payload)
        elif target is not None and hasattr(target, "write"):
            target.write(payload)
            flush = getattr(target, "flush", None)
            if callable(flush):
                flush()

        return text

    @classmethod
    def get_color(cls, level: str) -> str:
        """Get the color code for a given log level."""
        return cls.LOG_COLORS.get(level.upper(), "#FFFFFF")

    @classmethod
    def _resolve_color(cls, value: str) -> str:
        """Resolve a color string. Maps a known level name to its ``LOG_COLORS``
        hex; otherwise returns the value unchanged so raw CSS colors
        (``"#222"``, ``"steelblue"``, ``"rgb(0,0,0)"``) pass through.
        """
        if value is None:
            return None
        if isinstance(value, str) and value.upper() in cls.LOG_COLORS:
            return cls.LOG_COLORS[value.upper()]
        return value

    @classmethod
    def register_html_preset(cls, name: str, format_str: str) -> None:
        """Register a new HTML preset."""
        cls.HTML_PRESETS[name] = format_str

    @classmethod
    def get_html_preset(cls, name: str) -> str:
        """Get an HTML preset by name."""
        return cls.HTML_PRESETS.get(name, cls.HTML_PRESETS["default"])

    @classmethod
    def format_message_as_html(
        cls, message: str, level: str, preset: str = None
    ) -> str:
        """Format a message using HTML presets."""
        color = cls.get_color(level)

        # Determine preset
        if not preset:
            # Use level-based default if available, otherwise default
            preset = cls.LEVEL_PRESETS.get(level.upper(), "default")

        fmt = cls.get_html_preset(preset)
        return fmt.format(color=color, message=message)


class DefaultTextLogHandler(internal_logging.Handler):
    """A generic logging handler that writes logs to any widget supporting
    ``.append(str)``. Supports raw output, optional HTML color formatting,
    and optional monospace font styling.

    Appends are synchronous on the emitting thread (serialized by the
    handler lock, like every stdlib handler) so records arrive in order —
    never deferred to worker threads, which would surrender delivery order
    to the OS scheduler. GUI toolkits that require appends on their UI
    thread should register a marshaling handler class via
    ``set_text_handler`` instead — e.g. uitk's ``TextEditLogHandler``,
    which posts cross-thread records through a queued Qt signal.
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
                if self.monospace:
                    msg = f'<span style="font-family:monospace; white-space:pre;">{msg}</span>'
            else:
                msg = self.format(record)
                if self.use_html:
                    # Check for preset in extra args
                    preset = getattr(record, "preset", None)
                    msg = LoggerExt.format_message_as_html(
                        msg, record.levelname, preset
                    )
                    if self.monospace:
                        msg = f'<span style="font-family:monospace; white-space:pre-wrap;">{msg}</span>'
            self._safe_append(msg)
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
        return LoggerExt.get_color(level)


class RingBufferHandler(internal_logging.Handler):
    """In-memory capped ring buffer of log records.

    ``emit`` is O(1) and does no string formatting — records are stored by
    reference and rendered only when dumped (see ``format_records``), so an
    enabled buffer adds negligible cost to the logging hot path. Once
    ``capacity`` is exceeded the oldest record is dropped (``deque(maxlen)``).
    """

    def __init__(self, capacity: int = 2000, level: int = internal_logging.NOTSET):
        super().__init__(level=level)
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def emit(self, record: internal_logging.LogRecord) -> None:
        # Deliberately minimal: store the record, defer all formatting.
        self.buffer.append(record)

    def clear(self) -> None:
        self.buffer.clear()

    def format_records(self, formatter: internal_logging.Formatter = None) -> str:
        """Render buffered records to a single plain-text string.

        Raw records (boxes/dividers emitted via ``log_raw``) bypass the
        level formatter and have their HTML stripped, mirroring the
        stream/file output path so the dump reads like the console did.
        """
        fmt = formatter or self.formatter or internal_logging.Formatter()
        lines = []
        for record in list(self.buffer):
            if getattr(record, "raw", False):
                lines.append(LoggerExt.strip_html(record.getMessage()))
            else:
                lines.append(fmt.format(record))
        return "\n".join(lines)


class TableMixin:
    """Mixin for formatting data as ASCII tables."""

    def format_table(
        self,
        data: List[List[Any]],
        headers: List[str],
        title: Optional[str] = None,
        col_max_width: int = 60,
        max_width: int = 160,
    ) -> str:
        """Formats a list of lists as an ASCII table.

        Widths are measured in display columns (``LoggerExt._display_width``)
        rather than ``len()``, so emoji/CJK cells keep the table aligned.

        Args:
            data: List of rows, where each row is a list of values.
            headers: List of column headers.
            title: Optional title for the table.
            col_max_width: Maximum width for any single column.
            max_width: Maximum total table width in display columns.

        Returns:
            Formatted table string.
        """
        if not data:
            return ""

        dw = LoggerExt._display_width

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
        col_widths = [dw(h) for h in headers]
        for row in processed_data:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], dw(val))

        # Clamp per-column widths
        col_widths = [min(w, col_max_width) for w in col_widths]

        # Clamp total table width: separators add 3 chars (" | ") between columns
        separator_width = 3 * (num_cols - 1) if num_cols > 1 else 0
        total = sum(col_widths) + separator_width
        if total > max_width:
            available = max_width - separator_width
            if available < num_cols:
                available = num_cols  # at least 1 char per column
            # Shrink columns proportionally
            ratio = available / sum(col_widths)
            col_widths = [max(1, int(w * ratio)) for w in col_widths]
            # Distribute any remaining space due to rounding
            diff = available - sum(col_widths)
            for i in range(abs(diff)):
                if diff > 0:
                    col_widths[i % num_cols] += 1
                elif col_widths[i % num_cols] > 1:
                    col_widths[i % num_cols] -= 1

        def clip(text: str, w: int) -> str:
            # "..." when the column can afford it; hard-cut when shrunk to
            # w <= 3 (``_pad`` only pads — an over-long cell would overflow
            # the column and misalign the table).
            ellipsis = "..." if w > 3 else ""
            return LoggerExt._truncate(text, w, ellipsis=ellipsis)

        def render_row(cells: List[str]) -> str:
            return " | ".join(
                LoggerExt._pad(clip(c, w), w) for c, w in zip(cells, col_widths)
            )

        lines = []

        # Title
        if title:
            table_total = sum(col_widths) + separator_width
            lines.append(clip(title, max_width) if dw(title) > max_width else title)
            lines.append("-" * min(dw(title), table_total, max_width))

        # Header + separator + rows
        lines.append(render_row(headers))
        lines.append("-+-".join("-" * w for w in col_widths))
        for row in processed_data:
            lines.append(render_row(row))

        return "\n".join(lines)

    def log_table(
        self,
        data: List[List[Any]],
        headers: List[str],
        title: Optional[str] = None,
        level: str = "info",
    ) -> None:
        """Logs a formatted table.

        On a LoggerExt-patched logger the whole table goes through a single
        ``log_raw`` record — one monospace block in widget handlers, exactly
        like ``log_box``/``log_group`` — instead of one prefixed record per
        line. *level* applies only on the plain-logger fallback path.

        Args:
            data: List of rows.
            headers: List of column headers.
            title: Optional title.
            level: Logging level (info, warning, error, etc.)
        """
        table_str = self.format_table(data, headers, title)
        if not table_str:
            return

        logger = getattr(self, "logger", None)
        if logger is None:
            print(table_str)
        elif hasattr(logger, "log_raw"):
            logger.log_raw(table_str)
        else:
            log_method = getattr(logger, level.lower(), logger.info)
            for line in table_str.split("\n"):
                log_method(line)


class LoggingMixin(TableMixin):
    """Mixin class for logging utilities.

    Provides a logger for each class and a shared class logger across instances.
    Includes methods for setting log levels, adding handlers, and redirecting logs.
    """

    _logger: internal_logging.Logger = None
    _class_logger = None

    # Expose formatting constants
    LOG_COLORS = LoggerExt.LOG_COLORS
    HTML_PRESETS = LoggerExt.HTML_PRESETS
    log_link = staticmethod(LoggerExt._log_link)

    def __init__(
        self,
        *args,
        log_level: Optional[Union[int, str]] = None,
        log_file: Optional[str] = None,
        log_buffer: Union[bool, int, None] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if log_level is not None:
            self.set_log_level(log_level)
        if log_file is not None:
            self.set_log_file(log_file)
        if log_buffer:
            # log_buffer may be True (default capacity) or an int capacity.
            # bool is an int subclass, so test it first.
            if isinstance(log_buffer, int) and not isinstance(log_buffer, bool):
                self.enable_log_buffer(capacity=log_buffer)
            else:
                self.enable_log_buffer()

    @ClassProperty
    def logger(cls) -> internal_logging.Logger:
        if cls.__dict__.get("_logger") is None:
            name = f"{cls.__module__}.{cls.__qualname__}"
            logger = internal_logging.Logger(name, internal_logging.NOTSET)
            logger.propagate = False
            logger.parent = None
            LoggerExt.patch(logger)

            if not logger.handlers:
                logger.add_stream_handler()

            cls._logger = logger

        return cls._logger

    @ClassProperty
    def class_logger(cls) -> internal_logging.Logger:
        if cls.__dict__.get("_class_logger") is None:
            name = f"{cls.__module__}.{cls.__name__}.class"
            logger = internal_logging.getLogger(name)
            logger.setLevel(internal_logging.NOTSET)
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
        """Set log level for the class logger and its handlers.

        Delegates to the patched ``logger.setLevel`` (``LoggerExt._set_level``),
        which resolves both standard and custom level NAMES
        (PROGRESS/SUCCESS/RESULT/NOTICE) via ``logging._nameToLevel`` and syncs
        every handler's level — except handlers whose level was explicitly
        pinned (``set_log_file`` / ``enable_log_buffer`` with an explicit
        level). The previous ``getattr(logging, name)`` lookup
        mapped those custom names to WARNING (the ``logging`` module has no such
        attributes — they live in ``_nameToLevel``), silently suppressing the
        very levels the caller asked to enable. Accessing ``cls.logger`` also
        guarantees the custom levels are registered before name resolution.
        """
        cls.logger.setLevel(level)

    @classmethod
    def set_log_file(
        cls, filename: Optional[str], level: Union[int, str] = internal_logging.NOTSET
    ) -> None:
        """Tee this class's log output to *filename* (or ``None`` to stop).

        Off by default; continuous once enabled. See ``LoggerExt._set_log_file``.
        Class-scoped, since ``logger`` is shared across instances.
        """
        cls.logger.set_log_file(filename, level)

    @classmethod
    def enable_log_buffer(
        cls, capacity: int = 2000, level: Union[int, str] = internal_logging.NOTSET
    ) -> None:
        """Capture this class's log records into a capped ring buffer.

        Near-zero cost until ``dump_log`` is called. See
        ``LoggerExt._enable_log_buffer``.
        """
        cls.logger.enable_log_buffer(capacity, level)

    @classmethod
    def disable_log_buffer(cls) -> None:
        """Stop ring-buffer capture and discard buffered records."""
        cls.logger.disable_log_buffer()

    @classmethod
    def clear_log_buffer(cls) -> None:
        """Drop buffered records but keep capturing."""
        cls.logger.clear_log_buffer()

    @classmethod
    def dump_log(
        cls,
        target: Union[str, object, None] = None,
        mode: str = "w",
        encoding: str = "utf-8",
    ) -> str:
        """Render the ring buffer to text, optionally writing it to *target*.

        *target* is a file path, a writable stream, or ``None`` (return only).
        Returns the rendered text. Requires ``enable_log_buffer`` first.
        """
        return cls.logger.dump_log(target, mode, encoding)


# -------------------------------------------------------------------------------

if __name__ == "__main__":
    ...

# -------------------------------------------------------------------------------
