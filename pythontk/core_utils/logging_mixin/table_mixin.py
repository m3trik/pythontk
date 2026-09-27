# !/usr/bin/python
# coding=utf-8
"""``TableMixin``: tables and titled groups on any class that has a ``logger``.

The layout is :class:`~._text_layout.TextLayout`'s (the instance ``LoggerExt``
holds); this mixin sizes it to the logger's sinks and emits it as ONE record.
Degrades on a plain stdlib logger, or no logger at all, instead of raising.
"""

from __future__ import annotations

import logging as internal_logging
from typing import Any, List, Optional

from pythontk.core_utils.logging_mixin.logger_ext import LoggerExt


class TableMixin:
    """Mixin for formatting data as ASCII tables."""

    def format_table(
        self,
        data: List[List[Any]],
        headers: List[str],
        title: Optional[str] = None,
        col_max_width: int = 60,
        max_width: int = 160,
        wrap: bool = False,
        markup: bool = True,
    ) -> str:
        """Formats a list of lists as an ASCII table.

        Widths are measured in display columns (``TextLayout.display_width``)
        rather than ``len()``, so emoji/CJK cells keep the table aligned.

        Args:
            data: List of rows, where each row is a list of values.
            headers: List of column headers.
            title: Optional title for the table.
            col_max_width: Maximum width for any single column.
            max_width: Maximum total table width in display columns.
            wrap: Continue a cell wider than its column on the lines below
                (word-wrapped, the other cells blank) instead of clipping it
                with an ellipsis -- for columns of prose, where the clipped
                tail is the part that says what to do. A newline in a cell
                starts a new line there.
            markup: Cells may carry log markup (``<span ...>``), which takes
                no width. ``False`` for plain text, where ``wood_<UDIM>.png``
                is fifteen visible characters, not a tag.

        Returns:
            Formatted table string.
        """
        return LoggerExt._layout.table(
            data, headers, title, col_max_width, max_width, wrap, markup
        )

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
        line, capped at the width a box would take (``box_width``, else the
        narrowest attached handler's ``get_redirect_width``). *level* applies
        only on the plain-logger fallback path.

        Args:
            data: List of rows.
            headers: List of column headers.
            title: Optional title.
            level: Logging level (info, warning, error, etc.)
        """
        logger = getattr(self, "logger", None)
        # A width nothing reports -- or a stand-in that is no real logger --
        # leaves format_table's own cap standing.
        width = (
            LoggerExt._reported_width(logger)
            if isinstance(logger, internal_logging.Logger)
            else None
        )
        table_str = self.format_table(
            data, headers, title, **({"max_width": width} if width else {})
        )
        if not table_str:
            return

        if logger is None:
            print(table_str)
        elif hasattr(logger, "log_raw"):
            logger.log_raw(table_str)
        else:
            log_method = getattr(logger, level.lower(), logger.info)
            for line in table_str.split("\n"):
                log_method(line)

    def log_group(
        self,
        title: str,
        items: List[str],
        level: str = "info",
    ) -> None:
        """Log a titled list of related lines as a single record.

        Sibling of :meth:`log_table` with the same contract: on a
        LoggerExt-patched logger the whole group is ONE ``log_raw`` record —
        widget handlers append one QTextBlock per record, so a line-per-item
        loop renders as N blank-line-separated paragraphs instead of a list.

        Use this (rather than ``logger.log_group``) from any class that may be
        handed a caller-supplied ``logger=``: a plain stdlib logger carries no
        ``log_group``, and this degrades to one prefixed line per item instead
        of raising.

        Args:
            title: The group header text.
            items: The lines listed under the title. Empty means nothing worth
                a section, so nothing is emitted — never a bare orphan title.
            level: Severity of the group, case-insensitive. Colours the title
                on a patched logger and selects the log method on the plain
                fallback — one meaning on both paths, so a caller never has to
                know which one it hit.
        """
        if not items:
            return

        logger = getattr(self, "logger", None)
        if logger is None:
            print("\n".join([title] + [f"  {i}" for i in items]))
        elif hasattr(logger, "log_group"):
            # No .upper() needed — get_color normalizes the level name itself.
            logger.log_group(title, items, level=level)
        else:
            log_method = getattr(logger, level.lower(), logger.info)
            log_method(title)
            for item in items:
                log_method(f"  {item}")
