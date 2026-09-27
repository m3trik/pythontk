# !/usr/bin/python
# coding=utf-8
"""``LoggerExt``: the patch that turns a stdlib logger into a toolkit logger.

Custom levels (PROGRESS/SUCCESS/RESULT/NOTICE), HTML color presets, raw block
output (boxes, groups, dividers), a managed file tee, a capped in-memory ring
buffer and duplicate-error suppression -- plus the formatters
(:class:`StripHtmlFormatter`, :class:`LevelAwareFormatter`) and handlers
(:class:`DefaultTextLogHandler`, :class:`RingBufferHandler`) it attaches. The
column math of every block is :class:`~._text_layout.TextLayout`'s, through the
one instance ``LoggerExt._layout`` holds.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import sys
import time
import logging as internal_logging
from collections import deque
from typing import Union, List, Optional, Any
from pythontk.core_utils.logging_mixin._text_layout import TextLayout


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
        # The prefix/suffix are spliced into a %-style format string: a
        # literal "%" in either must be escaped or the stdlib raises
        # "not enough arguments for format string" and drops the record.
        prefix = prefix.replace("%", "%%")
        suffix = suffix.replace("%", "%%")
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

    # The text layout every box, divider and table is measured with
    # (``TableMixin`` reads it too): the ONE owner of the layout state. That
    # state (the resolved ``wcwidth`` function) lives on the instance, so a
    # caller that must measure differently builds its own
    # ``TextLayout(wcwidth=...)`` instead of mutating a shared attribute.
    _layout = TextLayout()

    # CSS font stack for markup whose columns must line up (boxes, and the
    # text handler's records). Real families first: the generic ``monospace``
    # names no installed family on Windows, so Qt substitutes by the widget
    # font's style hint -- the proportional UI font when there is none. uitk's
    # TextEditLogHandler measures its columns in this same stack.
    MONOSPACE_FAMILIES = "'Consolas','Courier New',Monaco,monospace"

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
                    "log_timestamp": property(_get_log_timestamp, _set_log_timestamp),
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
        # ``log_link`` is a pure string builder and genuinely needs no logger.
        # ``set_text_handler`` / ``get_text_handler`` used to live here too,
        # and that was the bug: bound without the logger they could only write
        # a class attribute, so ``self.logger.set_text_handler(X)`` -- which
        # reads as configuring THIS logger -- changed it for every unrelated
        # class in the process.
        direct_methods = {
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
            "set_text_handler": LoggerExt._set_text_handler,
            "get_text_handler": LoggerExt._get_text_handler,
            "setLevel": LoggerExt._set_level,
            "set_log_prefix": LoggerExt._set_log_prefix,
            "set_log_suffix": LoggerExt._set_log_suffix,
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

        The record is attributed to the CALLER of the patched method
        (``record.funcName``/``lineno``/``pathname``), not to this module:
        every patched level method reaches ``Logger.log`` through three
        frames of ours (``wrapper`` → ``_info``/``_success``/… → here), so
        the stdlib ``stacklevel`` is advanced past them. A caller's own
        ``stacklevel=`` composes on top, as with a plain logger.
        """
        if not logger.isEnabledFor(level_int):
            return  # skip styling work for records that will be dropped

        preset = kwargs.pop("preset", None)
        color_level = kwargs.pop("color", None)

        if preset or color_level:
            if not color_level:
                color_level = internal_logging.getLevelName(level_int)
            msg = LoggerExt.format_message_as_html(msg, color_level, preset)

        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 3
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
            handler_cls = LoggerExt._get_text_handler(logger)
            # Check if the handler accepts monospace argument
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
        """Update all handler formatters with level-aware formatting.

        Plain-text sinks (stream/file — ``FileHandler`` is a
        ``StreamHandler`` — and the ring buffer, whose dump is plain text)
        get the HTML-stripping variant; widget handlers keep markup.
        """
        for handler in logger.handlers:
            plain_text = isinstance(
                handler, (internal_logging.StreamHandler, RingBufferHandler)
            )
            handler.setFormatter(
                LevelAwareFormatter(logger=logger, strip_html=plain_text)
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
    def set_default_text_handler(handler: Union[type, object, None]) -> None:
        """Set the process-wide DEFAULT text handler class or instance.

        The deliberate form of what used to happen by accident. Before this,
        every ``logger.set_text_handler(...)`` wrote a class attribute, so one
        panel configuring its own logger silently gave every other logger in
        the process the same handler -- and panels that call
        ``setup_logging_redirect`` WITHOUT setting a handler (blendertk's
        ``telescope_rig`` / ``shader_templates`` / ``mat_updater``, extapps'
        ``substance_workflow``) were relying on that: whether their log pane
        got a Qt handler depended on which unrelated panel had opened first.

        A host that genuinely wants one Qt handler for the whole process --
        uitk registering ``TextEditLogHandler`` at startup, say -- calls this
        once. ``None`` restores :class:`DefaultTextLogHandler`. A per-logger
        :meth:`_set_text_handler` still wins over it.
        """
        LoggerExt._text_handler = handler

    #: Attribute the per-logger handler is stashed under. Namespaced because
    #: it lives on a stdlib ``Logger`` shared with every other library.
    _TEXT_HANDLER_ATTR = "_ptk_text_handler"

    @staticmethod
    def _set_text_handler(logger, handler: Union[type, object]) -> None:
        """Set a custom text handler class or instance for THIS logger.

        Stored on the logger, not on :class:`LoggerExt`. Assigning the class
        attribute made one class's call reconfigure every logger in the
        process -- which is why extapps' compositor attaches its handler by
        hand instead of using this seam.

        :attr:`LoggerExt._text_handler` remains the process-wide DEFAULT for a
        caller that deliberately wants one; a per-logger handler wins over it.
        """
        setattr(logger, LoggerExt._TEXT_HANDLER_ATTR, handler)

    @staticmethod
    def _get_text_handler(logger=None) -> type:
        """The text handler CLASS for *logger*, else the process default.

        Always a class, whether a class or an instance was set, because
        :meth:`_add_handler` constructs it.
        """
        handler = (
            getattr(logger, LoggerExt._TEXT_HANDLER_ATTR, None)
            if logger is not None
            else None
        )
        if handler is None:
            handler = LoggerExt._text_handler
        if handler is None:
            return DefaultTextLogHandler
        if isinstance(handler, type):
            return handler  # it's a class
        return handler.__class__  # it's an instance

    @staticmethod
    def _log_raw(self, message: str) -> None:
        """Write a raw message without level, prefix, or formatting."""
        # Write directly to all handler streams (console, files)
        for handler in self.handlers:
            stream = getattr(handler, "stream", None)
            if stream is not None:
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

    @classmethod
    def strip_html(cls, text: str) -> str:
        """Remove HTML tags from *text*, leaving the visible plain text.

        The single strip used everywhere log markup meets a plain-text
        sink (stream/file formatters, raw writes, buffer dumps, width
        measurement). The layout's: what it strips is what takes no width.
        """
        return cls._layout.strip_html(text)

    @staticmethod
    def _reported_width(self) -> Optional[int]:
        """``self.box_width`` if set; else the narrowest column count
        reported by attached handlers (see ``get_redirect_width``); ``None``
        when neither says."""
        width = getattr(self, "box_width", None)
        if width is None:
            width = LoggerExt._get_redirect_width(self)
        return width

    @staticmethod
    def _resolve_width(self, width: Optional[int]) -> int:
        """Column budget for raw block output (boxes, dividers).

        *width* when given; else :meth:`_reported_width`; else
        ``DEFAULT_BOX_WIDTH``.
        """
        if width is None:
            width = LoggerExt._reported_width(self)
        if width is None:
            width = LoggerExt.DEFAULT_BOX_WIDTH
        return width

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

        Non-string items are coerced with ``str``; a newline inside the
        title or an item starts a new row (a row is one physical line).

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
        max_width = LoggerExt._resolve_width(self, max_width)
        lines, width = LoggerExt._layout.box(title, items, max_width, align=align)

        text_color = LoggerExt._resolve_color(level) if level else None
        bg_color = LoggerExt._resolve_color(bg) if bg else None

        # Box drawing chars (╔ ═ ║ ╚) only align when rendered in a
        # monospace cell. HTML viewers (QTextEdit / browser) often inherit
        # a proportional font from outer wrappers, which collapses the
        # column math. Carry the font-family inline on each span so the
        # boxes render correctly regardless of enclosing context.
        mono_css = f"font-family:{LoggerExt.MONOSPACE_FAMILIES};white-space:pre"

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
    def _log_link(text: str, action: str, /, **params: str) -> str:
        """Return an HTML ``<a>`` tag for embedding clickable links in log messages.

        The link uses a custom ``action://`` URI scheme that is never opened
        by a browser — handlers (e.g. ``QTextBrowser.anchorClicked``) parse
        the URL and dispatch the action.

        Parameters:
            text:   Visible link label (HTML-escaped automatically).
            action: Action verb (e.g. ``"select"``, ``"reveal"``).
            **params: Arbitrary key-value pairs appended as query string.
                Any key is valid — ``text``/``action`` included (the label
                and verb parameters are positional-only), so e.g. a copy
                action can pass its payload as ``text=...``.

        Returns:
            An ``<a href="action://ACTION?k=v&…">text</a>`` string that can
            be embedded in any log message via f-string::

                link = logger.log_link("pCube1", "select", node="|group1|pCube1")
                logger.info(f"Missing object: {link}")
        """
        import html

        safe_text = html.escape(text, quote=False)
        href = html.escape(LoggerExt._action_url(action, **params))
        # No spaces in the tag — TextLayout.wrap_text splits on spaces and would
        # break the tag if any are present inside attributes.
        return f'<a href="{href}" style="text-decoration:underline">{safe_text}</a>'

    @staticmethod
    def _action_url(action: str, /, **params: Any) -> str:
        """``action://ACTION?k=v&...`` -- the one builder of the link grammar the
        DCC dispatchers parse (``UiUtils.dispatch_log_link``): values are
        URL-encoded, so a node path's ``|`` or a name's ``&`` round-trips."""
        from urllib.parse import urlencode

        query = urlencode({k: str(v) for k, v in params.items()})
        return f"action://{action}?{query}" if query else f"action://{action}"

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
        """Print a divider rule *width* columns wide.

        *width* resolves exactly as a box's ``max_width`` does
        (``self.box_width`` → narrowest attached handler →
        ``DEFAULT_BOX_WIDTH``), so a rule never wraps in the panel it is
        drawn in and lines up with the boxes beside it.
        """
        LoggerExt._log_raw(self, char * LoggerExt._resolve_width(self, width))

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

        # Generate cache key if not provided
        if cache_key is None:
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
        # Two more frames of ours (this + the ``error`` wrapper) sit between
        # the caller and ``_log_custom``; keep the record attributed to them.
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 2

        if should_log:
            # Log the full error with suffix
            full_message = f"{message}{suffix}"
            logger.error(full_message, *args, **kwargs)
        else:
            # Log a brief debug message for suppressed errors
            logger.debug(
                f"Suppressed duplicate error: {message[:50]}...",
                stacklevel=kwargs["stacklevel"],
            )

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
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 2

        if should_log:
            full_message = f"{message}{suffix}"
            logger.warning(full_message, *args, **kwargs)
        else:
            logger.debug(
                f"Suppressed duplicate warning: {message[:50]}...",
                stacklevel=kwargs["stacklevel"],
            )

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
            families = LoggerExt.MONOSPACE_FAMILIES
            if getattr(record, "raw", False):
                msg = record.getMessage()
                # A raw record is a preformatted block (box, group, table): an
                # HTML sink needs it monospace with its whitespace kept whatever
                # ``monospace`` says -- bare, a group's line breaks collapse. A
                # plain-text sink gets the block bare: markup means nothing there.
                if self.use_html:
                    msg = f'<span style="font-family:{families}; white-space:pre;">{msg}</span>'
            else:
                msg = self.format(record)
                if self.use_html:
                    # Check for preset in extra args
                    preset = getattr(record, "preset", None)
                    msg = LoggerExt.format_message_as_html(
                        msg, record.levelname, preset
                    )
                    if self.monospace:
                        msg = f'<span style="font-family:{families}; white-space:pre-wrap;">{msg}</span>'
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
