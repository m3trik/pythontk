# !/usr/bin/python
# coding=utf-8
"""``ReportDoc`` -- a report built once as blocks, rendered as HTML or plain text.

Tool reports land in two places at once: a rich-text viewer (uitk's
``TextViewBox``, which renders Qt's HTML subset) and the host's console or a log
file. Formatting each by hand is how they drift, and how they break: the scene
audit used to print ASCII tables through a logger and wrap the capture in
``<pre>``, so the viewer showed a monospace dump with no real tables, and a
texture bucket labelled ``<512`` vanished as an unknown tag. A ``ReportDoc``
records WHAT the report says -- headings, key/value fields, tables, bullet
items -- and renders it either way.
"""

import html as _html
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote

# From this package:
from pythontk.core_utils.color import Palette
from pythontk.core_utils.logging_mixin import LoggerExt, TableMixin


class ReportDoc:
    """A report as an ordered list of blocks, rendered to HTML or plain text.

    Build it with the block methods (each returns the document, so they chain),
    put :class:`Inline` values wherever a cell or item takes text -- a link, a
    toned word -- then render with :meth:`to_html` or :meth:`to_text`::

        doc = ReportDoc()
        doc.heading("Summary").fields([("Meshes", "1,505"), ("Triangles", "57,012")])
        doc.table(
            ["Object", "Tris"],
            [[ReportDoc.action("DOOR_A", "select", node="|DOOR_A"), "1,188"]],
            align="lr",
        )
        viewer_html, console_text = doc.to_html(), doc.to_text()

    The HTML targets Qt's rich-text subset (tables, inline colour and weight --
    no class selectors, no flexbox) and is emitted without a single newline: a
    viewer that turns free newlines into ``<br>`` (uitk's ``RichTextFormatter``
    does) would otherwise tear the tables apart. Every plain value is
    HTML-escaped on the way out, so scene data can never become markup; markup
    comes only from :class:`Inline` values, which escape their own text.
    """

    class Inline:
        """A run of text that carries its own markup -- a link, a toned word.

        ``text`` is what :meth:`ReportDoc.to_text` shows and ``html`` what
        :meth:`ReportDoc.to_html` embeds verbatim. Build them with
        :meth:`ReportDoc.span`, :meth:`ReportDoc.link`, :meth:`ReportDoc.action`,
        :meth:`ReportDoc.file` and :meth:`ReportDoc.join` rather than by hand,
        so the HTML side is always escaped.
        """

        __slots__ = ("text", "html")

        def __init__(self, text: str, html: str) -> None:
            self.text = text
            self.html = html

        def __str__(self) -> str:
            return self.text

        def __repr__(self) -> str:
            return f"ReportDoc.Inline({self.text!r})"

    #: Tone name -> CSS colour. Palette-sourced so a report reads like the rest
    #: of the dark theme; a tone that is not listed here is used as a CSS colour
    #: verbatim, so a caller can pass ``"#ffcc00"`` without registering it.
    TONES: Dict[str, str] = {
        "heading": Palette.status()["info"][0],
        "info": Palette.status()["info"][0],
        "warn": Palette.status()["warn"][0],
        "error": Palette.status()["error"][0],
        # The status palette has no success tier; a muted green of the same
        # weight as its warn/error pastels.
        "ok": "#9CC69B",
        "dim": Palette.ui()["text_dim"].hex,
        "strong": Palette.ui()["text_bright"].hex,
        # Light blue, as MatReport's path links -- readable on a dark viewer.
        "link": "#99CCFF",
    }

    #: Row tints as translucent white, so a table reads on whatever background
    #: the host viewer paints (Maya's, Blender's, a light theme's).
    HEADER_BG = "rgba(255,255,255,26)"
    STRIPE_BG = "rgba(255,255,255,10)"

    _HEADING_TAGS = {1: "h2", 2: "h3", 3: "h4"}
    _HEADING_MARGINS = {1: "0 0 2px 0", 2: "14px 0 4px 0", 3: "8px 0 2px 0"}
    _ALIGN = {"l": "left", "r": "right", "c": "center"}

    def __init__(self) -> None:
        self._blocks: List[Tuple[str, Dict[str, Any]]] = []

    def __bool__(self) -> bool:
        return bool(self._blocks)

    # ---- blocks ------------------------------------------------------------
    def heading(
        self, text: Any, level: int = 2, tone: Optional[str] = "heading"
    ) -> "ReportDoc":
        """A heading. Level 1 titles a whole report, 2 a section, 3 a subsection."""
        level = max(1, min(3, int(level)))
        self._blocks.append(("heading", {"text": text, "level": level, "tone": tone}))
        return self

    def text(self, text: Any, tone: Optional[str] = None) -> "ReportDoc":
        """A paragraph. *text* may be a string, an :class:`Inline` or a list of both."""
        self._blocks.append(("text", {"text": text, "tone": tone}))
        return self

    def fields(self, rows: Iterable[Tuple[Any, Any]]) -> "ReportDoc":
        """Label/value pairs, rendered as an aligned two-column grid."""
        rows = [(label, value) for label, value in rows]
        if rows:
            self._blocks.append(("fields", {"rows": rows}))
        return self

    def table(
        self,
        headers: Sequence[Any],
        rows: Iterable[Sequence[Any]],
        align: Optional[Union[str, Sequence[str]]] = None,
        title: Optional[Any] = None,
        footer: Optional[Any] = None,
        wrap: Optional[Iterable[int]] = None,
    ) -> "ReportDoc":
        """A table with a header row.

        Parameters:
            headers: Column titles.
            rows: One sequence of cell values per row. A short row is padded,
                a long one truncated to the header width.
            align: One of ``l`` / ``r`` / ``c`` per column, as a string
                (``"lrr"``) or a sequence. Missing columns align left.
            title: Optional caption shown above the table.
            footer: Optional note shown below it (e.g. ``"... 12 more"``).
            wrap: Indexes of the columns whose text may wrap in HTML. Default:
                the last column only -- the usual place for prose -- so a name
                or a number is never broken mid-word to make room for it.
                Plain-text rendering wraps any cell wider than its column.
        """
        headers = list(headers)
        width = len(headers)
        body = []
        for row in rows:
            row = list(row)[:width]
            body.append(row + [""] * (width - len(row)))
        aligns = [self._ALIGN.get(a, "left") for a in (align or "")][:width]
        aligns += ["left"] * (width - len(aligns))
        wrapping = {width - 1} if wrap is None else set(wrap)
        self._blocks.append(
            (
                "table",
                {
                    "headers": headers,
                    "rows": body,
                    "align": aligns,
                    "nowrap": [i not in wrapping for i in range(width)],
                    "title": title,
                    "footer": footer,
                },
            )
        )
        return self

    def items(self, items: Iterable[Any], tone: Optional[str] = None) -> "ReportDoc":
        """A bullet list; *tone* colours the bullets."""
        items = list(items)
        if items:
            self._blocks.append(("items", {"items": items, "tone": tone}))
        return self

    def extend(self, other: "ReportDoc") -> "ReportDoc":
        """Append *other*'s blocks to this document."""
        self._blocks.extend(other._blocks)
        return self

    # ---- inline values -----------------------------------------------------
    @classmethod
    def color(cls, tone: Optional[str]) -> Optional[str]:
        """The CSS colour for *tone* (a :attr:`TONES` key, else used verbatim)."""
        if not tone:
            return None
        return cls.TONES.get(tone, tone)

    @staticmethod
    def _escape(text: str) -> str:
        """HTML-escape *text*; a newline in the data becomes ``<br>``, never a
        free newline (see the class docstring)."""
        return _html.escape(text).replace("\n", "<br>")

    @classmethod
    def span(cls, text: Any, tone: Optional[str] = None, bold: bool = False) -> Inline:
        """*text* in a tone and/or bold."""
        text = str(text)
        style = []
        if tone:
            style.append(f"color:{cls.color(tone)}")
        if bold:
            style.append("font-weight:bold")
        escaped = cls._escape(text)
        if not style:
            return cls.Inline(text, escaped)
        return cls.Inline(text, f"<span style='{'; '.join(style)}'>{escaped}</span>")

    @classmethod
    def link(cls, text: Any, href: str, tone: Optional[str] = "link") -> Inline:
        """*text* linked to *href* (any scheme; what opens it is the viewer's)."""
        text = str(text)
        color = f"color:{cls.color(tone)}; " if tone else ""
        # href double-quoted: Qt decodes entities (the "&amp;" between query
        # params) only there -- single-quoted, the handler gets "&amp;" verbatim.
        return cls.Inline(
            text,
            f'<a href="{_html.escape(href, quote=True)}" '
            f"style='{color}text-decoration:none;'>{cls._escape(text)}</a>",
        )

    @classmethod
    def action(cls, text: Any, verb: str, /, **params: Any) -> Inline:
        """An ``action://VERB?k=v`` link -- the host dispatches the verb.

        The URL is ``LoggingMixin.log_link``'s (one builder), so the DCC link
        dispatchers (``mayatk.UiUtils.dispatch_log_link``, blendertk's twin)
        route it: ``ReportDoc.action("DOOR_A", "select", node="|ROOT|DOOR_A")``.
        """
        return cls.link(text, LoggerExt._action_url(verb, **params))

    @classmethod
    def file(cls, path: str, text: Optional[Any] = None) -> Inline:
        """A ``file:`` link to *path*, shown as *text* (default: the path).

        The href is URL-encoded (spaces, ``&``, parentheses survive a Qt link
        handler) with forward slashes; the displayed path is left as given. A UNC
        path keeps its host: ``\\\\srv\\share\\a.png`` -> ``file://srv/share/a.png``.
        """
        target = str(path).replace("\\", "/")
        if target.startswith("//"):  # UNC: the host is the URL's authority
            href = "file:" + quote(target, safe="/:")
        else:
            href = "file:///" + quote(target.lstrip("/"), safe="/:")
        return cls.link(path if text is None else text, href)

    @classmethod
    def join(cls, parts: Iterable[Any], sep: str = ", ") -> Inline:
        """Concatenate values (strings and/or :class:`Inline`) with *sep*."""
        parts = list(parts)
        return cls.Inline(
            sep.join(cls._cell_text(p) for p in parts),
            cls._escape(sep).join(cls._cell_html(p) for p in parts),
        )

    # ---- rendering ---------------------------------------------------------
    def to_html(self) -> str:
        """The document as newline-free HTML for a Qt rich-text viewer."""
        return "".join(self._html_block(kind, spec) for kind, spec in self._blocks)

    def to_text(self) -> str:
        """The document as plain text for a console or a log file."""
        out: List[str] = []
        for kind, spec in self._blocks:
            if kind == "heading" and out:
                out.append("")  # breathing room above every heading but the first
            out.append(self._text_block(kind, spec))
        return "\n".join(out)

    @classmethod
    def _cell_html(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, cls.Inline):
            return value.html
        if isinstance(value, (list, tuple)):
            return "".join(cls._cell_html(v) for v in value)
        return cls._escape(str(value))

    @classmethod
    def _cell_text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, cls.Inline):
            return value.text
        if isinstance(value, (list, tuple)):
            return "".join(cls._cell_text(v) for v in value)
        return str(value)

    def _html_block(self, kind: str, spec: Dict[str, Any]) -> str:
        dim = self.color("dim")
        if kind == "heading":
            tag = self._HEADING_TAGS[spec["level"]]
            color = self.color(spec["tone"])
            style = f"margin:{self._HEADING_MARGINS[spec['level']]};"
            if color:
                style = f"color:{color}; " + style
            return f"<{tag} style='{style}'>{self._cell_html(spec['text'])}</{tag}>"

        if kind == "text":
            color = self.color(spec["tone"])
            style = "margin:2px 0;" + (f" color:{color};" if color else "")
            return f"<p style='{style}'>{self._cell_html(spec['text'])}</p>"

        if kind == "fields":
            rows = "".join(
                f"<tr><td valign='top' style='color:{dim}; padding-right:14px;'>"
                f"{self._cell_html(label)}</td>"
                f"<td valign='top'>{self._cell_html(value)}</td></tr>"
                for label, value in spec["rows"]
            )
            return (
                "<table cellspacing='0' cellpadding='1' style='margin:2px 0 4px 0;'>"
                f"{rows}</table>"
            )

        if kind == "items":
            bullet = self.color(spec["tone"]) or dim
            rows = "".join(
                f"<tr><td valign='top' style='color:{bullet}; padding-right:6px;'>"
                f"&#8226;</td><td valign='top'>{self._cell_html(item)}</td></tr>"
                for item in spec["items"]
            )
            return (
                "<table cellspacing='0' cellpadding='1' style='margin:2px 0 4px 0;'>"
                f"{rows}</table>"
            )

        if kind == "table":
            parts = []
            if spec["title"] is not None:
                parts.append(
                    f"<p style='margin:6px 0 2px 0; color:{dim};'>"
                    f"{self._cell_html(spec['title'])}</p>"
                )
            head = "".join(
                f"<th align='{a}' style='color:{dim}; font-weight:normal;'>"
                f"{self._cell_html(h)}</th>"
                for h, a in zip(spec["headers"], spec["align"])
            )
            body = []
            for i, row in enumerate(spec["rows"]):
                stripe = f" style='background-color:{self.STRIPE_BG};'" if i % 2 else ""
                cells = "".join(
                    f"<td align='{a}' valign='top'"
                    + (" style='white-space:nowrap;'" if nowrap else "")
                    + f">{self._cell_html(c)}</td>"
                    for c, a, nowrap in zip(row, spec["align"], spec["nowrap"])
                )
                body.append(f"<tr{stripe}>{cells}</tr>")
            if not body:
                span = len(spec["headers"]) or 1
                body.append(
                    f"<tr><td colspan='{span}' style='color:{dim};'>(none)</td></tr>"
                )
            parts.append(
                "<table cellspacing='0' cellpadding='3' style='margin:2px 0 4px 0;'>"
                f"<tr style='background-color:{self.HEADER_BG};'>{head}</tr>"
                f"{''.join(body)}</table>"
            )
            if spec["footer"] is not None:
                parts.append(
                    f"<p style='margin:0 0 4px 0; color:{dim};'>"
                    f"{self._cell_html(spec['footer'])}</p>"
                )
            return "".join(parts)

        raise ValueError(f"Unknown ReportDoc block kind: {kind!r}")

    def _text_block(self, kind: str, spec: Dict[str, Any]) -> str:
        if kind == "heading":
            title = self._cell_text(spec["text"])
            if spec["level"] == 3:
                return f"{title}:"
            rule = "=" if spec["level"] == 1 else "-"
            return f"{title}\n{rule * max(len(title), 3)}"

        if kind == "text":
            return self._cell_text(spec["text"])

        if kind == "fields":
            rows = [(self._cell_text(k), self._cell_text(v)) for k, v in spec["rows"]]
            pad = max(len(k) for k, _ in rows)
            return "\n".join(f"{k:<{pad}} : {v}" for k, v in rows)

        if kind == "items":
            return "\n".join(f"  - {self._cell_text(i)}" for i in spec["items"])

        if kind == "table":
            lines = []
            if spec["title"] is not None:
                lines.append(self._cell_text(spec["title"]))
            headers = [self._cell_text(h) for h in spec["headers"]]
            rows = [[self._cell_text(c) for c in row] for row in spec["rows"]]
            # markup=False: a "<UDIM>" token in a file name is text, not a tag.
            lines.append(
                TableMixin().format_table(rows, headers, wrap=True, markup=False)
                if rows
                else "(none)"
            )
            if spec["footer"] is not None:
                lines.append(self._cell_text(spec["footer"]))
            return "\n".join(lines)

        raise ValueError(f"Unknown ReportDoc block kind: {kind!r}")


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
