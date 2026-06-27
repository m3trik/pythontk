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
    LevelAwareFormatter,
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

    def test_setlevel_clears_custom_level_cache(self):
        """Lowering the level re-enables custom levels (SUCCESS/RESULT/NOTICE).

        Regression: these loggers are built via Logger() (not getLogger), so
        they are absent from manager.loggerDict and the stdlib setLevel's
        manager._clear_cache() never clears their per-logger isEnabledFor
        cache. Once isEnabledFor(SUCCESS) was cached False (while the level
        was above SUCCESS), success() stayed silently disabled even after the
        level was lowered. _set_level now clears the logger's own cache.
        Fixed: 2026-06-27
        """
        LoggerExt.patch(self.logger)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)

        # Raise above SUCCESS and poison the isEnabledFor cache for level 25.
        self.logger.setLevel(logging.ERROR)  # 40 > SUCCESS (25)
        self.assertFalse(self.logger.isEnabledFor(LoggerExt.SUCCESS))
        self.logger.success("suppressed at ERROR")
        self.assertEqual(stream.getvalue(), "")

        # Lower below SUCCESS — the custom level must re-enable, not stay stuck.
        self.logger.setLevel(logging.INFO)  # 20 < SUCCESS (25)
        self.assertTrue(self.logger.isEnabledFor(LoggerExt.SUCCESS))
        self.logger.success("now visible")
        self.assertIn("now visible", stream.getvalue())

    def test_percent_format_args_always_substituted(self):
        """Positional args are %-format args — never styling.

        Regression: a heuristic treated args matching a preset/color name
        ("default", "bold", "error", …) as styling and silently skipped the
        ``%s`` substitution."""
        LoggerExt.patch(self.logger)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        self.logger.addHandler(handler)
        self.logger.setLevel("INFO")

        self.logger.info("mode was %s", "default")
        self.logger.info("level was %s", "error")
        handler.flush()
        output = stream.getvalue()

        self.assertIn("mode was default", output)
        self.assertIn("level was error", output)
        self.assertNotIn("%s", output)

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


class LogBoxTest(BaseTestCase):
    """Tests for _log_box and _truncate."""

    def setUp(self):
        super().setUp()
        self.logger = logging.Logger("test_box", logging.DEBUG)
        self.logger.handlers = []
        LoggerExt.patch(self.logger)
        # Capture raw output via a stream handler
        self.stream = io.StringIO()
        handler = logging.StreamHandler(self.stream)
        handler.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)

    def tearDown(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        super().tearDown()

    def test_truncate_short_text_unchanged(self):
        """Text shorter than limit is returned unchanged."""
        self.assertEqual(LoggerExt._truncate("hello", 10), "hello")

    def test_truncate_long_text(self):
        """Text exceeding limit is truncated with ellipsis."""
        result = LoggerExt._truncate("abcdefghij", 6)
        self.assertTrue(result.endswith("…"))
        self.assertLessEqual(LoggerExt._display_width(result), 6)

    def test_truncate_exact_fit(self):
        """Text exactly at the limit is returned unchanged."""
        self.assertEqual(LoggerExt._truncate("abcde", 5), "abcde")

    def test_log_box_single_emit(self):
        """log_box emits exactly one write to the stream (single string).

        Bug: Previously each box line was a separate _log_raw call, causing
        paragraph-level spacing in QTextEdit widgets.
        Fixed: 2026-03-08
        """
        self.logger.log_box("TITLE", ["item1", "item2"])
        output = self.stream.getvalue()
        # Should contain box-drawing chars with newlines between them
        # but only ONE trailing newline from _log_raw (not N separate writes)
        lines = output.strip().split("\n")
        self.assertEqual(lines[0][0], "╔")
        self.assertEqual(lines[-1][0], "╚")
        # 5 lines: top, title, separator, item1, item2... wait, that's 6:
        # top, title, sep, item1, item2, bottom
        self.assertEqual(len(lines), 6)

    def test_log_box_no_items(self):
        """Box with title only has 3 lines: top, title, bottom."""
        self.logger.log_box("TITLE")
        output = self.stream.getvalue().strip()
        lines = output.split("\n")
        self.assertEqual(len(lines), 3)

    def test_log_box_max_width_truncates(self):
        """Items wider than max_width are truncated with ellipsis."""
        long_item = "a" * 80
        self.logger.log_box("T", [long_item], max_width=30)
        output = self.stream.getvalue().strip()
        for line in output.split("\n"):
            self.assertLessEqual(
                LoggerExt._display_width(line), 30, f"Line exceeds max_width: {line!r}"
            )

    def test_log_box_max_width_from_attribute(self):
        """log_box reads self.box_width when max_width is not passed."""
        self.logger.box_width = 30
        long_item = "a" * 80
        self.logger.log_box("T", [long_item])
        output = self.stream.getvalue().strip()
        for line in output.split("\n"):
            self.assertLessEqual(LoggerExt._display_width(line), 30)

    def test_log_box_returns_width(self):
        """log_box returns the computed box width."""
        width = self.logger.log_box("HELLO")
        # "HELLO" = 5 chars + 2 padding + 2 borders = 9
        self.assertEqual(width, 9)

    def test_log_box_shrinks_after_wrap(self):
        """Box shrinks to the longest wrapped line, not max_width.

        Bug: A single long line forced the box to max_width even though
        wrapped fragments were much shorter.
        Fixed: 2026-04-10
        """
        # Three 15-char words (total 47). max_content at max_width=50 is 46.
        # Wrapping produces: "aaa..a bbb..b" (31) and "ccc..c" (15).
        # Longest = 31, so box = 31 + 4 = 35, well under 50.
        long_item = "a" * 15 + " " + "b" * 15 + " " + "c" * 15
        width = self.logger.log_box("OK", [long_item], max_width=50)
        self.assertLess(width, 50)

    def test_log_box_solid_background(self):
        """bg= wraps each line in a span carrying background-color so HTML
        handlers render a contiguous solid block. Accepts both level names
        and raw CSS colors."""
        widget = MockTextWidget()
        handler = DefaultTextLogHandler(widget, use_html=True, monospace=True)
        handler.setLevel(logging.DEBUG)
        self.logger.handlers = [handler]

        # Level-name form
        self.logger.log_box("TITLE", ["item"], bg="ERROR")
        # Raw CSS color form
        self.logger.log_box("TITLE2", ["item"], bg="#222", level="SUCCESS")

        import time

        time.sleep(0.1)

        self.assertEqual(len(widget.messages), 2)

        first = widget.messages[0]
        # Level "ERROR" resolved to its hex
        self.assertIn("background-color:#FFCCCC", first)
        # Per-line wrapping: one span per box row.
        # With items, rows are: top, title, sep, item, bottom = 5
        self.assertEqual(first.count("<span style="), 5 + 1)  # +1 monospace wrapper

        second = widget.messages[1]
        # Raw color passed through unchanged
        self.assertIn("background-color:#222", second)
        # Combined with text color from level
        self.assertIn("color:#CCFFCC", second)

    def test_log_box_clamps_to_terminal_width(self):
        """When a stream handler is attached to a TTY, log_box clamps to the
        live terminal width so the box is not broken by autowrap.

        This was previously broken: a 60-col terminal would still get a
        100-col box and every line would visually wrap."""
        from unittest.mock import MagicMock, patch
        import os as _os

        # Replace stream with a TTY-pretending mock
        self.logger.handlers = []
        fake_stream = MagicMock()
        fake_stream.isatty.return_value = True
        fake_stream.fileno.return_value = 1
        handler = logging.StreamHandler(fake_stream)
        handler.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)

        fake_size = _os.terminal_size((40, 24))
        with patch("os.get_terminal_size", return_value=fake_size):
            long_item = "a" * 80
            width = self.logger.log_box("T", [long_item])

        # Box must fit the (narrower) reported terminal width
        self.assertLessEqual(width, 40)

    def test_log_box_ignores_non_tty_stream(self):
        """File/StringIO streams are not TTYs, so terminal-size probing is
        skipped and the existing fallback chain (max_width / box_width /
        DEFAULT_BOX_WIDTH) still applies."""
        # The test's setUp already uses StringIO, which reports isatty()=False.
        # Without max_width or box_width, this should fall back to
        # DEFAULT_BOX_WIDTH (100). A long item must therefore wrap to <=100.
        self.logger.log_box("T", ["x" * 200])
        output = self.stream.getvalue().strip()
        for line in output.split("\n"):
            self.assertLessEqual(LoggerExt._display_width(line), 100)

    def test_log_box_strips_html_for_levelaware_stream(self):
        """Default stream/file handlers attach ``LevelAwareFormatter(strip_html=True)``.
        Raw output (log_box, log_divider) must respect that and not leak
        ``<span>`` / ``<a>`` markup into the terminal or log file.

        Regression: previously ``_log_raw`` only honored ``StripHtmlFormatter``,
        so level-colored or bg= boxes printed visible HTML to consoles."""
        self.logger.handlers = []
        handler = logging.StreamHandler(self.stream)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            LevelAwareFormatter(logger=self.logger, strip_html=True)
        )
        self.logger.addHandler(handler)

        self.logger.log_box("TITLE", ["item"], bg="ERROR", level="ERROR")
        output = self.stream.getvalue()
        self.assertNotIn("<span", output)
        self.assertNotIn("background-color", output)
        self.assertIn("TITLE", output)
        self.assertIn("item", output)

    def test_log_box_widget_handler_single_append(self):
        """Widget handler receives exactly one append call for the whole box.

        Bug: Multiple append() calls caused paragraph spacing in QTextEdit.
        Fixed: 2026-03-08
        """
        widget = MockTextWidget()
        handler = DefaultTextLogHandler(widget, use_html=False)
        handler.setLevel(logging.DEBUG)
        # Replace stream handler with widget handler
        self.logger.handlers = [handler]

        self.logger.log_box("TITLE", ["item1"])
        # Give the threading.Timer a moment to fire
        import time

        time.sleep(0.1)

        self.assertEqual(
            len(widget.messages),
            1,
            f"Expected 1 append call, got {len(widget.messages)}",
        )


class MockTextWidget:
    """Mock text widget for testing."""

    def __init__(self):
        self.messages = []

    def append(self, text):
        self.messages.append(text)


class LogGroupTest(BaseTestCase):
    """Tests for _log_group: bold title + bar-prefixed indented items.

    Routes through a stream handler so we can assert on the stripped
    plain-text output that downstream consoles/files see, plus a mock
    text widget to confirm the HTML emitted to Qt widgets keeps the
    structure (single QTextBlock per group → continuous left rule).
    """

    BAR = "▎"  # U+258E LEFT ONE QUARTER BLOCK

    def setUp(self):
        super().setUp()
        self.logger = logging.Logger("test_group", logging.DEBUG)
        self.logger.handlers = []
        LoggerExt.patch(self.logger)
        self.stream = io.StringIO()
        handler = logging.StreamHandler(self.stream)
        handler.setLevel(logging.DEBUG)
        # log_group always wraps items in <span> for color; only handlers
        # configured to strip HTML produce the plain-text form that
        # console/file consumers see. Match that here so the assertions
        # check real-world output rather than raw HTML.
        handler.setFormatter(LevelAwareFormatter(logger=self.logger, strip_html=True))
        self.logger.addHandler(handler)

    def tearDown(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        super().tearDown()

    def test_log_group_prefixes_items_with_bar(self):
        """Each item line is prefixed with the U+258E bar character."""
        self.logger.log_group("Header:", ["a.png", "b.png", "c.png"])
        lines = self.stream.getvalue().split("\n")
        item_lines = [ln for ln in lines if "png" in ln]
        self.assertEqual(len(item_lines), 3)
        for ln in item_lines:
            self.assertTrue(
                ln.lstrip().startswith(self.BAR),
                f"Expected line to start with bar after strip: {ln!r}",
            )

    def test_log_group_indent_positions_item(self):
        """Default indent=2 → bar at col 0, item text at col 2."""
        self.logger.log_group("Header:", ["item"])
        lines = self.stream.getvalue().split("\n")
        item_line = next(ln for ln in lines if "item" in ln)
        # Format: "<bar><space>item" → 3 chars before "item"
        self.assertEqual(item_line[0], self.BAR)
        self.assertEqual(item_line[1], " ")
        self.assertTrue(item_line[2:].startswith("item"))

    def test_log_group_emits_one_log_raw_call(self):
        """All items + title go through a single _log_raw — critical for
        the bar to render as a continuous rule in Qt (one QTextBlock).
        """
        widget = MockTextWidget()
        # Swap stream handler for a text-widget handler.
        for h in self.logger.handlers[:]:
            self.logger.removeHandler(h)
        self.logger.addHandler(DefaultTextLogHandler(widget))

        self.logger.log_group("Header:", ["a", "b", "c"])
        # DefaultTextLogHandler uses a Timer; pump it.
        import time

        time.sleep(0.05)
        # Exactly one append() call — the whole group is one message.
        self.assertEqual(
            len(widget.messages),
            1,
            f"Expected 1 append (single QTextBlock), got {len(widget.messages)}: "
            f"{widget.messages!r}",
        )
        msg = widget.messages[0]
        # All three items must be in that single message.
        for token in ("a", "b", "c", "Header:"):
            self.assertIn(token, msg)

    def test_log_group_empty_items_falls_back_to_title(self):
        """An empty items list emits the title alone (no bar, no formatting)."""
        self.logger.log_group("Just a header", [])
        output = self.stream.getvalue().strip()
        self.assertEqual(output, "Just a header")
        self.assertNotIn(self.BAR, output)

    def test_log_group_leading_blank_line(self):
        """A blank line precedes each group so consecutive groups separate
        visually rather than butting together.
        """
        self.logger.log_group("First:", ["x"])
        # Output starts with "\n" before the title (the leading "\n" in html).
        out = self.stream.getvalue()
        self.assertTrue(out.startswith("\n"), f"Expected leading newline, got {out!r}")


class FileTeeAndBufferTest(BaseTestCase):
    """Tests for the optional file tee + in-memory ring buffer dump."""

    def _tmp(self):
        fd, path = tempfile.mkstemp(suffix=".log")
        os.close(fd)
        return path

    def test_set_log_file_tees_records(self):
        """set_log_file attaches a continuous file handler; records land in it."""

        class Foo(LoggingMixin):
            pass

        path = self._tmp()
        try:
            Foo.set_log_file(path, level="DEBUG")
            Foo.logger.info("hello-tee")
            Foo.set_log_file(None)  # closes/detaches → safe to read
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("hello-tee", content)
        finally:
            Foo.set_log_file(None)
            os.unlink(path)

    def test_set_log_file_toggle_replaces_and_detaches(self):
        """Re-calling replaces the managed handler; None detaches it."""

        class Foo(LoggingMixin):
            pass

        self.assertIsNone(getattr(Foo.logger, "_managed_file_handler", None))
        p1, p2 = self._tmp(), self._tmp()
        try:
            Foo.set_log_file(p1)
            h1 = Foo.logger._managed_file_handler
            Foo.set_log_file(p2)
            h2 = Foo.logger._managed_file_handler
            self.assertIsNot(h1, h2)
            self.assertNotIn(h1, Foo.logger.handlers)
            Foo.set_log_file(None)
            self.assertIsNone(Foo.logger._managed_file_handler)
        finally:
            Foo.set_log_file(None)
            os.unlink(p1)
            os.unlink(p2)

    def test_buffer_dump_returns_text_and_writes_file(self):
        """enable_log_buffer captures records; dump_log returns + writes them."""

        class Foo(LoggingMixin):
            pass

        Foo.enable_log_buffer(capacity=10)
        Foo.logger.info("alpha")
        Foo.logger.warning("beta")

        text = Foo.dump_log()
        self.assertIn("alpha", text)
        self.assertIn("beta", text)

        path = self._tmp()
        try:
            returned = Foo.dump_log(path)
            self.assertEqual(returned, text)
            with open(path, encoding="utf-8") as f:
                self.assertIn("alpha", f.read())
        finally:
            os.unlink(path)

    def test_buffer_capacity_drops_oldest(self):
        """The ring buffer is capped: oldest records are evicted."""

        class Foo(LoggingMixin):
            pass

        Foo.enable_log_buffer(capacity=3)
        for i in range(5):
            Foo.logger.info("msg%d", i)

        text = Foo.dump_log()
        self.assertNotIn("msg0", text)
        self.assertNotIn("msg1", text)
        self.assertIn("msg4", text)

    def test_dump_log_to_stream(self):
        """dump_log writes to any object with .write()."""

        class Foo(LoggingMixin):
            pass

        Foo.enable_log_buffer()
        Foo.logger.info("streamed")
        sink = io.StringIO()
        Foo.dump_log(sink)
        self.assertIn("streamed", sink.getvalue())

    def test_dump_log_without_buffer_returns_empty(self):
        """dump_log is a safe no-op (returns '') when no buffer is enabled."""

        class Foo(LoggingMixin):
            pass

        self.assertEqual(Foo.dump_log(), "")

    def test_clear_and_disable_buffer(self):
        """clear empties but keeps capturing; disable removes the buffer."""

        class Foo(LoggingMixin):
            pass

        Foo.enable_log_buffer()
        Foo.logger.info("x")
        Foo.clear_log_buffer()
        self.assertEqual(Foo.dump_log(), "")
        Foo.disable_log_buffer()
        self.assertIsNone(getattr(Foo.logger, "_ring_buffer_handler", None))

    def test_constructor_log_file_and_buffer(self):
        """log_file= and log_buffer= constructor kwargs wire everything up."""

        class Foo(LoggingMixin):
            pass

        path = self._tmp()
        try:
            Foo(log_level="DEBUG", log_file=path, log_buffer=True)
            Foo.logger.info("ctor-msg")
            self.assertIn("ctor-msg", Foo.dump_log())
            Foo.set_log_file(None)
            with open(path, encoding="utf-8") as f:
                self.assertIn("ctor-msg", f.read())
        finally:
            Foo.set_log_file(None)
            os.unlink(path)

    def test_enable_log_buffer_resize_preserves_recent(self):
        """Re-enabling with a new capacity resizes, keeping recent records."""

        class Foo(LoggingMixin):
            pass

        Foo.enable_log_buffer(capacity=5)
        for i in range(5):
            Foo.logger.info("r%d", i)
        Foo.enable_log_buffer(capacity=2)  # shrink in place
        text = Foo.dump_log()
        self.assertIn("r4", text)
        self.assertIn("r3", text)
        self.assertNotIn("r0", text)


if __name__ == "__main__":
    unittest.main(exit=False)
