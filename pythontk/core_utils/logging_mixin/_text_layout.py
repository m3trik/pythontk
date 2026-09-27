# !/usr/bin/python
# coding=utf-8
"""Monospace text layout: display widths, wrapping, box and table geometry.

:class:`TextLayout` is the one owner of the column math the logging toolkit
draws with -- ``LoggerExt`` lays its boxes out through it and ``TableMixin``
its tables, both through the one instance ``LoggerExt`` holds. Pure string
work: no logger, no handler, no colour or CSS, so a block measures the same
whichever sink it ends up in.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, List, Optional, Tuple, Union


class TextLayout:
    """Lay text out in the display columns of a monospace grid.

    A column is what one terminal cell or monospace glyph occupies: CJK and
    the emoji blocks take two, combining marks and format controls none. Log
    markup (``<span ...>``, ``<a ...>``) takes none and is never split.

    Parameters:
        wcwidth: Per-character width function with ``wcwidth.wcwidth``'s
            contract (an int; negative for a non-printable, counted as 0).
            ``None`` (the default) resolves the optional ``wcwidth`` package on
            first use; ``False`` forces the built-in heuristic.
    """

    #: Log markup: what :meth:`strip_html` removes and what takes no width.
    _HTML_TAG_RE = re.compile(r"<[^>]+>")

    #: Characters a word too long for its line breaks AFTER before it is cut
    #: mid-character (:meth:`_hard_wrap_word`): path separators.
    _WORD_BREAKS = frozenset("/\\")

    # Sentinel that never appears in real text, used to protect spaces
    # inside HTML tags from being split by wrap_text.
    _TAG_SPACE = "\x00"

    def __init__(self, wcwidth: Union[Callable[[str], int], bool, None] = None):
        #: ``wcwidth.wcwidth`` when the optional package is installed, else
        #: ``False``; ``None`` until the first width query resolves it. Resolved
        #: ONCE: this is called per character, and a failed ``import`` inside it
        #: re-walked every ``sys.path`` entry each time -- measured at 2 ms per
        #: character, 1.2 s for one scene-export summary box.
        self._wcwidth_fn: Any = wcwidth

    def char_width(self, ch: str) -> int:
        """Return the display/column width of a single character.

        Uses ``wcwidth`` if available, otherwise falls back to a heuristic
        based on ``unicodedata.east_asian_width`` plus known wide symbol
        ranges (Dingbats, Miscellaneous Symbols, emoji, etc.) whose glyphs
        typically occupy two columns in modern monospace fonts even though
        Unicode classifies them as narrow or ambiguous.
        """
        fn = self._wcwidth_fn
        if fn is None:
            try:
                from wcwidth import wcwidth as fn
            except Exception:
                fn = False
            self._wcwidth_fn = fn
        if fn:
            try:
                return max(fn(ch), 0)
            except Exception:
                pass

        # Zero-width first: combining marks (Mn/Me — accents, variation
        # selectors), format controls (Cf — ZWJ, soft hyphen) and control
        # chars (Cc) occupy no column, as ``wcwidth`` reports. Counting
        # them over-pads every box/table holding a ``⚠️`` or a decomposed
        # ``é`` wherever ``wcwidth`` is absent (DCC pythons).
        if unicodedata.category(ch) in ("Mn", "Me", "Cf", "Cc"):
            return 0
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
        ):
            return 2
        return 1

    @classmethod
    def strip_html(cls, text: str) -> str:
        """Remove HTML tags from *text*, leaving the visible plain text.

        The single strip used everywhere log markup meets a plain-text
        sink (stream/file formatters, raw writes, buffer dumps, width
        measurement).
        """
        return cls._HTML_TAG_RE.sub("", text)

    def display_width(self, text: str) -> int:
        """Return the display/column width of *text*.

        HTML tags (``<span ...>``, ``<a ...>``, etc.) are stripped before
        measuring so that embedded markup does not inflate the count.
        """
        visible = self.strip_html(text)
        return sum(self.char_width(ch) for ch in visible)

    def pad(
        self, text: str, target_width: int, fill: str = " ", align: str = "left"
    ) -> str:
        """Pad *text* to *target_width* display columns using *fill* char."""
        current = self.display_width(text)
        deficit = max(target_width - current, 0)
        if align == "right":
            return fill * deficit + text
        elif align == "center":
            left = deficit // 2
            right = deficit - left
            return fill * left + text + fill * right
        else:  # left
            return text + fill * deficit

    def truncate(self, text: str, max_display_width: int, ellipsis: str = "…") -> str:
        """Truncate *text* to *max_display_width* display columns with ellipsis."""
        dw = self.display_width
        if dw(text) <= max_display_width:
            return text
        ellipsis_w = dw(ellipsis)
        result = []
        current_width = 0
        for ch in text:
            ch_w = self.char_width(ch)
            if current_width + ch_w + ellipsis_w > max_display_width:
                break
            result.append(ch)
            current_width += ch_w
        return "".join(result) + ellipsis

    def _hard_wrap_word(self, word: str, max_display_width: int) -> tuple:
        """Hard-wrap a single word that exceeds *max_display_width*.

        A line ends after its last path separator (:attr:`_WORD_BREAKS`) when
        it holds one, and at the width only when it does not: cut at the width,
        a path split mid-name (``.../gone_Rough`` over ``ness.png``) -- no longer
        readable, copyable or findable, at a point that moved with the length of
        its folders.

        Returns ``(complete_lines, remaining_fragment, remaining_width)``.
        """
        width_of = self.char_width
        lines = []
        chars = []
        w = 0
        for ch in word:
            ch_w = width_of(ch)
            while chars and w + ch_w > max_display_width:
                cut = next(
                    (
                        i + 1
                        # Never at index 0: that cut emits a line holding
                        # only the separator.
                        for i in range(len(chars) - 1, 0, -1)
                        if chars[i] in self._WORD_BREAKS
                    ),
                    len(chars),
                )
                lines.append("".join(chars[:cut]))
                chars = chars[cut:]
                w = sum(map(width_of, chars))
            chars.append(ch)
            w += ch_w
        return lines, "".join(chars), w

    def wrap_text(self, text: str, max_display_width: int) -> List[str]:
        """Wrap *text* to fit within *max_display_width* display columns.

        Breaks at word boundaries when possible, otherwise hard-wraps.
        HTML tags are protected so that spaces inside attributes are never
        used as break points.
        Returns a list of wrapped lines.
        """
        dw = self.display_width
        if dw(text) <= max_display_width:
            return [text]

        # Protect spaces inside HTML tags by replacing them with a sentinel
        sentinel = self._TAG_SPACE
        has_tags = "<" in text and ">" in text
        if has_tags:

            def _protect_tag_spaces(m: "re.Match") -> str:
                return m.group(0).replace(" ", sentinel)

            text = self._HTML_TAG_RE.sub(_protect_tag_spaces, text)

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
                    extra, current_line, current_width = self._hard_wrap_word(
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
                    extra, current_line, current_width = self._hard_wrap_word(
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
    def split_lines(values: List[Any]) -> List[str]:
        """Coerce *values* to strings and split embedded newlines, so each
        box row measures and pads as one physical line."""
        return [line for value in values for line in str(value).split("\n")]

    def box(
        self,
        title: str,
        items: Optional[List[str]],
        max_width: int,
        align: str = "left",
    ) -> Tuple[List[str], int]:
        """Frame *title* over *items* in box-drawing glyphs (``log_box``).

        Non-string items are coerced with ``str``; a newline inside the
        title or an item starts a new row (a row is one physical line). The
        box shrinks to its longest row, and wraps what does not fit
        *max_width* display columns. Rows are filled with NBSP, which an
        HTML sink does not collapse.

        Parameters:
            title: The title row(s), aligned per *align*.
            items: The rows under the separator (``None`` for none); a bare
                string is one row.
            max_width: Widest the whole box may be, borders included -- the
                caller resolves it (``LoggerExt._resolve_width``).
            align: ``"left"``, ``"center"`` or ``"right"`` for the title.

        Returns:
            ``(lines, width)``: the box's rows, top border to bottom, and its
            full width in display columns.
        """
        padding = 1
        # Use non-breaking space to prevent HTML space collapsing in handlers
        space = "\u00a0"

        dw = self.display_width
        wrap = self.wrap_text
        title_lines = self.split_lines([title])
        # A bare string is iterable, so it used to render one box row per
        # CHARACTER (it raised TypeError before the split-lines rewrite, which
        # at least said so). Take it as one row, matching the scalar-or-list
        # tolerance CoreUtils.listify gives the rest of the package.
        if isinstance(items, str):
            items = [items]
        items = self.split_lines(items or [])
        content = title_lines + items
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
            all_wrapped_title = [
                wl for tl in title_lines for wl in wrap(tl, wrap_width)
            ]
            all_wrapped_items = []
            for item in items:
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

        # Unwrapped title rows already fit ``longest``
        if needs_wrap:
            title_lines = all_wrapped_title
        title_rows = []
        for tl in title_lines:
            tl_padded = self.pad(tl, longest, fill=space, align=align)
            title_rows.append("║" + space * padding + tl_padded + space * padding + "║")

        sep = "╟" + "─" * inner_width + "╢"
        bottom = "╚" + "═" * inner_width + "╝"

        lines = [top] + title_rows
        if items:
            lines.append(sep)
            for idx, item in enumerate(items):
                wrapped = (
                    all_wrapped_items[idx]
                    if needs_wrap
                    else wrap(item, inner_width - 1)
                )
                for wl in wrapped:
                    item_padded = self.pad(wl, inner_width - 1, fill=space)
                    item_line = space + item_padded
                    lines.append(f"║{item_line}║")
        lines.append(bottom)
        return lines, width

    def table(
        self,
        data: List[List[Any]],
        headers: List[str],
        title: Optional[str],
        col_max_width: int,
        max_width: int,
        wrap: bool,
        markup: bool,
    ) -> str:
        """Lay *data* out as an ASCII table (``TableMixin.format_table``).

        Widths are measured in display columns (:meth:`display_width`)
        rather than ``len()``, so emoji/CJK cells keep the table aligned.
        Every argument is required: the defaults are the public facade's
        (``TableMixin.format_table``), kept in one place.

        Parameters:
            data: List of rows, where each row is a list of values.
            headers: List of column headers.
            title: Optional title for the table.
            col_max_width: Maximum width for any single column.
            max_width: Maximum total table width in display columns.
            wrap: Continue a cell wider than its column on the lines below
                (word-wrapped, the other cells blank) instead of clipping it
                with an ellipsis. A newline in a cell starts a new line there.
            markup: Cells may carry log markup (``<span ...>``), which takes
                no width. ``False`` for plain text, where ``wood_<UDIM>.png``
                is fifteen visible characters, not a tag.

        Returns:
            Formatted table string (``""`` for no rows).
        """
        if not data:
            return ""
        if not markup:
            # The width math strips anything tag-shaped: stand "<" / ">" in with
            # private-use code points (one column each, never a tag), put back after.
            enc, dec = {60: "\ue000", 62: "\ue001"}, {0xE000: "<", 0xE001: ">"}
            return self.table(
                [[str(c).translate(enc) for c in row] for row in data],
                [str(h).translate(enc) for h in headers],
                title.translate(enc) if title else title,
                col_max_width,
                max_width,
                wrap,
                True,  # markup: the stand-ins are tag-free by construction
            ).translate(dec)

        dw = self.display_width

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

        # Calculate column widths (wrapping, a cell's widest line)
        col_widths = [dw(h) for h in headers]
        for row in processed_data:
            for i, val in enumerate(row):
                width = max(map(dw, val.split("\n"))) if wrap else dw(val)
                col_widths[i] = max(col_widths[i], width)

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
            # w <= 3 (``pad`` only pads — an over-long cell would overflow
            # the column and misalign the table).
            ellipsis = "..." if w > 3 else ""
            return self.truncate(text, w, ellipsis=ellipsis)

        def render_row(cells: List[str]) -> str:
            return " | ".join(
                self.pad(clip(c, w), w) for c, w in zip(cells, col_widths)
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
            if not wrap:
                lines.append(render_row(row))
                continue
            wrapped = [
                [
                    line
                    for part in c.split("\n")
                    for line in self.wrap_text(part, w) or [""]
                ]
                for c, w in zip(row, col_widths)
            ]
            for i in range(max(len(cell) for cell in wrapped)):
                lines.append(
                    render_row([cell[i] if i < len(cell) else "" for cell in wrapped])
                )

        return "\n".join(lines)
