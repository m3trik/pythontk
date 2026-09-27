#!/usr/bin/python
# coding=utf-8
"""Unit tests for ``TextLayout``, the logging toolkit's column math.

``TextLayout`` (``pythontk/core_utils/logging_mixin/_text_layout.py``) is the one
owner of the display-width state every box, divider and table is measured with.
Output of the blocks it lays out is pinned character for character in
``test_logging_mixin.LayoutCharacterizationTest``; these tests cover the layout
itself: width resolution, injection, and that the logger and the tables share
one instance.

Run with:
    python -m pytest test/test_text_layout.py -v
"""

import io
import logging
import unittest
from unittest import mock

from pythontk.core_utils.logging_mixin import LoggerExt, TableMixin
from pythontk.core_utils.logging_mixin._text_layout import TextLayout

from conftest import BaseTestCase


class CharWidthTest(BaseTestCase):
    """``char_width`` resolves the optional ``wcwidth`` import ONCE.

    It is called per character of every box/divider line; a failed import
    inside it re-walked ``sys.path`` on every call (measured 2 ms a
    character -- 1.2 s for one scene-export summary box). Added: 2026-09-02
    """

    def test_import_is_resolved_once_and_widths_still_measure(self):
        layout = TextLayout()
        self.assertIsNone(layout._wcwidth_fn)  # unresolved until first asked
        self.assertEqual(layout.char_width("a"), 1)
        self.assertIsNotNone(layout._wcwidth_fn)  # a callable, or False
        resolved = layout._wcwidth_fn
        layout.char_width("b")
        self.assertIs(layout._wcwidth_fn, resolved)
        # The fallback heuristic answers the same wide glyphs either way.
        heuristic = TextLayout(wcwidth=False)
        self.assertEqual(heuristic.char_width("✓"), 2)
        self.assertEqual(heuristic.display_width("<b>ab</b>✓"), 4)

    def test_fallback_char_width_treats_zero_width_marks_as_zero(self):
        """Bug: without ``wcwidth`` the heuristic counted variation
        selectors (U+FE0F) and the ZWJ (U+200D) as TWO columns and combining
        marks as one — ``⚠️`` measured 4, ``é`` (decomposed) measured 2, so
        every box/table containing one was over-padded in DCC pythons
        (which ship no ``wcwidth``)."""
        layout = TextLayout(wcwidth=False)
        self.assertEqual(layout.char_width("\ufe0f"), 0)  # VS16
        self.assertEqual(layout.char_width("\u200d"), 0)  # ZWJ
        self.assertEqual(layout.char_width("\u0301"), 0)  # combining acute
        self.assertEqual(layout.display_width("\u26a0\ufe0f"), 2)  # ⚠️
        self.assertEqual(layout.display_width("e\u0301"), 1)  # decomposed é


class TextLayoutTest(BaseTestCase):
    """The layout's own contract: widths, injection, and one shared owner."""

    #: Characters whose width ``wcwidth`` and the heuristic agree on, markup
    #: included: the goldens below hold with or without the optional package.
    SAMPLES = [
        "abc",
        "漢字",
        "✅",
        "\U0001f642",
        "cafe\u0301",
        "<b>ab</b>✅",
        '<a href="action://x?y=1">link</a>',
    ]
    WIDTHS = [3, 4, 2, 2, 4, 4, 4]

    def test_display_widths_with_and_without_wcwidth(self):
        for layout in (TextLayout(), TextLayout(wcwidth=False)):
            with self.subTest(wcwidth=layout._wcwidth_fn):
                self.assertEqual(
                    [layout.display_width(s) for s in self.SAMPLES], self.WIDTHS
                )

    def test_an_injected_width_function_is_used_and_clamped(self):
        self.assertEqual(TextLayout(wcwidth=lambda ch: 2).display_width("ab"), 4)
        # wcwidth's contract: negative for a non-printable -- counted as 0.
        self.assertEqual(TextLayout(wcwidth=lambda ch: -1).char_width("a"), 0)

        def broken(ch):
            raise ValueError(ch)

        # A raising function falls through to the heuristic.
        self.assertEqual(TextLayout(wcwidth=broken).char_width("漢"), 2)

    def test_one_layouts_resolution_never_reaches_another(self):
        """The state is the instance's: the old class attribute was shared by
        every measurement in the process, so forcing the heuristic in one
        place forced it everywhere."""
        forced = TextLayout(wcwidth=False)
        fresh = TextLayout()
        fresh.char_width("a")
        self.assertIs(forced._wcwidth_fn, False)
        self.assertEqual(forced.char_width("✓"), 2)

    def test_the_logger_and_the_tables_measure_with_one_layout(self):
        """``LoggerExt`` owns the one instance and ``TableMixin`` reads it:
        swap it and both a box and a table are measured by the new one."""
        self.assertFalse(hasattr(LoggerExt, "_wcwidth_fn"))  # no second owner
        seen = []

        def recording(ch):
            seen.append(ch)
            return 1

        logger = logging.Logger("one_layout", logging.DEBUG)
        LoggerExt.patch(logger)
        logger.handlers = [logging.StreamHandler(io.StringIO())]
        with mock.patch.object(LoggerExt, "_layout", TextLayout(wcwidth=recording)):
            logger.log_box("Z")
            TableMixin().format_table([["x"]], ["y"])
        self.assertTrue({"Z", "x", "y"} <= set(seen))

    def test_strip_html_is_the_layouts(self):
        text = '<span style="color:red">a <a href="x">b</a></span>'
        self.assertEqual(TextLayout.strip_html(text), "a b")
        self.assertEqual(LoggerExt.strip_html(text), "a b")

    def test_box_is_geometry_only(self):
        """Colour, CSS and emission are the logger's: ``box`` returns rows
        of glyphs and NBSP fill, and the width they measure."""
        layout = TextLayout()
        lines, width = layout.box("Title", ["a", "bb"], max_width=40)
        self.assertEqual(lines[0], "╔" + "═" * (width - 2) + "╗")
        self.assertEqual({layout.display_width(ln) for ln in lines}, {width})
        self.assertFalse(any("<" in ln for ln in lines))

    def test_table_markup_false_keeps_tag_shaped_text(self):
        out = TextLayout().table(
            [["wood_<UDIM>.png"]], ["file"], None, 60, 160, wrap=False, markup=False
        )
        self.assertEqual(out.split("\n")[-1], "wood_<UDIM>.png")


if __name__ == "__main__":
    unittest.main(exit=False)
