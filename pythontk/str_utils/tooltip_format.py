# !/usr/bin/python
# coding=utf-8
"""Rich-text tooltip DSL -- :class:`TooltipFormat`.

A pure string builder (``kbd`` / ``hl`` / ``fmt`` / ``placeholder_preview`` /
``stored_items``) plus the layout rules a shown tooltip goes through (``wrap`` /
``display_ms``). No Qt: the markup is the HTML subset Qt's rich-text engine
renders, and the plain/rich decision is a port of ``Qt.mightBeRichText``, so
headless code builds the same tooltips a panel shows. uitk's
``TooltipPresenter`` is the Qt side that displays them.
"""
import html as _html
import re


# --- Line layout (TooltipFormat.wrap) --------------------------------------

#: Tags that end a rendered line -- a column count never carries across one.
_BLOCK_TAGS = frozenset(
    "address blockquote body br caption center dd div dl dt h1 h2 h3 h4 h5 h6 "
    "head hr html li ol p pre qt table tbody td tfoot th thead title tr ul".split()
)
#: Tags whose content is laid out as written (or not shown) -- never re-broken.
_VERBATIM_TAGS = frozenset("head nobr pre script style title".split())
#: The elements Qt's rich-text parser knows: ``Qt.mightBeRichText`` calls text
#: rich only when its first tag is one of these (``<Enter>`` stays plain).
_QT_ELEMENTS = frozenset(
    "a address b big blockquote body br caption center cite code dd dfn div dl "
    "dt em font h1 h2 h3 h4 h5 h6 head hr html i img kbd li link meta nobr ol p "
    "pre qt s samp script small span strong style sub sup table tbody td tfoot "
    "th thead title tr tt u ul var".split()
)
_TAG_SPLIT_RE = re.compile(r"(<[^>]*>)")
_TAG_NAME_RE = re.compile(r"<\s*(/?)\s*([A-Za-z][A-Za-z0-9]*)")
_TAG_RE = re.compile(r"<[^>]*>")
_NOWRAP_STYLE_RE = re.compile(r"white-space\s*:\s*(?:pre|nowrap)", re.I)
_RICH_SPACE_RE = re.compile(r"([ \t\r\n\f]+)")
_PLAIN_SPACE_RE = re.compile(r"([ \t]+)")
#: Zero-width split AFTER each path separator: where a too-wide path may break.
_PATH_BREAK_RE = re.compile(r"(?<=[/\\])")
#: A plain line's indent plus list marker; its continuation lines hang under it.
_PLAIN_LEAD_RE = re.compile(r"[ \t]*(?:[-*•▪▸][ \t]+|\d{1,3}[.)][ \t]+)?")
#: A word that closes a sentence (or clause), allowing trailing quotes/brackets.
_SENTENCE_END_RE = re.compile(r"[.!?:;][\"'’”)\]]*$")


# --- Color palette ---------------------------------------------------------
# Tuned for Qt's default dark tooltip background. Kept on the cool side so
# colored fragments don't fight the bold-default text Qt renders for tooltips.

_C_MUTED = "#9a9a9a"  # de-emphasized labels (row keys, notes prefix)
_C_NOTE = "#bda36a"  # warm muted for note/tip callout body
_C_ACCENT = "#6fb5d6"  # soft cyan — keywords, headings, term highlights
_C_TITLE = "#cfe6f5"  # off-white with cool tint for the top title


class _TooltipFormatInternal:
    """Line layout behind :meth:`TooltipFormat.wrap` -- pure string work.

    Text is cut into tokens -- ``(kind, raw, width, visible)`` with kind one of
    ``word`` / ``space`` / ``tag`` (inline, zero width) / ``block`` (ends the
    line) / ``lead`` (a plain line's indent + list marker) -- and one greedy pass
    then re-emits every ``raw`` verbatim, swapping a ``space`` for a line break
    where the soft-width rule says so. Nothing else in the text is touched.
    """

    #: Opens the container a wrapped rich tooltip is emitted in. Qt squeezes rich
    #: text under four lines to half, then a quarter, of its wrap width and
    #: re-wraps it there; ``nowrap`` (inherited by every block inside) makes it
    #: keep the breaks placed here. Doubles as the already-wrapped marker.
    _NOWRAP_OPEN = "<div style='white-space:nowrap'>"

    @classmethod
    def _is_rich(cls, text: str) -> bool:
        """Qt's plain/rich call for *text* -- through the Qt-free port, which
        classifies exactly as ``Qt.mightBeRichText`` does (uitk's
        ``test_tooltip_mixin`` checks the two against each other on a corpus)."""
        return cls._is_rich_port(text)

    @staticmethod
    def _is_rich_port(text: str) -> bool:
        """A port of ``Qt::mightBeRichText`` (qtextdocument.cpp) for headless use.

        Qt decides from the FIRST line only: it is rich when that line's first
        tag names a known element, or ``&lt;`` comes before any tag.
        """
        n = len(text)
        start = 0
        while start < n and text[start].isspace():
            start += 1
        if text[start : start + 5] == "<?xml":
            end = text.find("?>", start)
            start = n if end < 0 else end + 2
            while start < n and text[start].isspace():
                start += 1
        if text[start : start + 5].lower() == "<!doc":
            return True
        open_ = start
        while open_ < n and text[open_] not in "<\n":
            if text.startswith("&lt;", open_):
                return True
            open_ += 1
        if open_ >= n or text[open_] != "<":
            return False
        close = text.find(">", open_)
        if close < 0:
            return False
        tag = ""
        for i in range(open_ + 1, close):
            ch = text[i]
            if ch.isalnum():
                tag += ch.lower()
            elif tag and ch.isspace():
                break
            elif tag and ch == "/" and i + 1 == close:
                break
            elif not ch.isspace() and (tag or ch != "!"):
                return False
        return tag in _QT_ELEMENTS

    @staticmethod
    def _visible_text(html: str) -> str:
        """*html*'s rendered words: block tags part them, inline tags vanish."""

        def _tag(match):
            name = _TAG_NAME_RE.match(match.group(0))
            return " " if name and name.group(2).lower() in _BLOCK_TAGS else ""

        return _html.unescape(_TAG_RE.sub(_tag, html))

    @staticmethod
    def _plain_tokens(text: str) -> list:
        tokens = []
        for n, line in enumerate(text.replace("\r\n", "\n").split("\n")):
            if n:
                tokens.append(("block", "\n", 0, ""))
            lead = _PLAIN_LEAD_RE.match(line).group(0)
            if lead:
                tokens.append(("lead", lead, len(lead), ""))
                line = line[len(lead) :]
            for piece in _PLAIN_SPACE_RE.split(line):
                if piece:
                    kind = "space" if piece[0] in " \t" else "word"
                    tokens.append((kind, piece, len(piece), piece))
        return tokens

    @classmethod
    def _rich_tokens(cls, text: str) -> list:
        tokens = []
        verbatim = None  # [tag name, nesting depth, raw parts] while inside one
        for i, part in enumerate(_TAG_SPLIT_RE.split(text)):
            if not part:
                continue
            is_tag = i % 2 == 1
            match = _TAG_NAME_RE.match(part) if is_tag else None
            name = match.group(2).lower() if match else ""
            closing = bool(match and match.group(1))
            self_closing = part.rstrip().endswith("/>")
            if verbatim is not None:
                verbatim[2].append(part)
                if name == verbatim[0] and not self_closing:
                    verbatim[1] += -1 if closing else 1
                    if not verbatim[1]:
                        tokens.append(cls._verbatim_token(*verbatim[::2]))
                        verbatim = None
                continue
            if is_tag:
                if (
                    match
                    and not closing
                    and not self_closing
                    and (name in _VERBATIM_TAGS or _NOWRAP_STYLE_RE.search(part))
                ):
                    verbatim = [name, 1, [part]]
                else:
                    kind = "block" if name in _BLOCK_TAGS else "tag"
                    tokens.append((kind, part, 0, ""))
                continue
            for piece in _RICH_SPACE_RE.split(part):
                if not piece:
                    continue
                if _RICH_SPACE_RE.fullmatch(piece):
                    tokens.append(("space", piece, 1, ""))
                else:
                    visible = _html.unescape(piece)
                    tokens.append(("word", piece, len(visible), visible))
        if verbatim is not None:  # never closed: keep it as written
            tokens.append(cls._verbatim_token(*verbatim[::2]))
        return tokens

    @staticmethod
    def _verbatim_token(name: str, parts: list) -> tuple:
        """A ``<pre>``-like element as ONE token: a line of its own when it is a
        block, else an unbreakable word as wide as its visible text."""
        raw = "".join(parts)
        if name in _BLOCK_TAGS or name in ("script", "style"):
            return ("block", raw, 0, "")
        visible = _html.unescape(_TAG_RE.sub("", raw))
        return ("word", raw, len(visible), visible)

    @classmethod
    def _layout(
        cls, tokens: list, width: int, slack: int, brk: str, split_paths: bool = False
    ) -> str:
        """Re-emit *tokens*, breaking with *brk* where the soft-width rule says.

        With *split_paths*, a word too wide for any line breaks after its path
        separators (the rich layout: its nowrap container stops Qt's own
        wider-than-the-screen fallback, so the word would run off-screen).
        """
        out = []
        col = 0
        hang, hang_w = "", 0
        count = len(tokens)
        for i, (kind, raw, w, _) in enumerate(tokens):
            if split_paths and kind == "word" and w > width + slack and "<" not in raw:
                pieces = [p for p in _PATH_BREAK_RE.split(raw) if p]
                if len(pieces) > 1:
                    for piece in pieces:
                        piece_w = len(_html.unescape(piece))
                        if col > hang_w and col + piece_w > width:
                            out.append(brk + hang)
                            col = hang_w
                        out.append(piece)
                        col += piece_w
                    continue
            if kind == "block":
                out.append(raw)
                col, hang, hang_w = 0, "", 0
                continue
            if kind == "lead":
                out.append(raw)
                col += w
                hang = "".join(c if c == "\t" else " " for c in raw)
                hang_w = w
                continue
            if kind != "space" or col == 0:
                out.append(raw)
                col += w if kind != "space" else 0
                continue
            # A space: the one place a break can go. Width of the word after it:
            nxt, j = 0, i + 1
            while j < count and tokens[j][0] in ("word", "tag"):
                nxt += tokens[j][2]
                j += 1
            if not nxt or col + w + nxt <= width:
                out.append(raw)
                col += w
            elif cls._fits_to_sentence_end(tokens, i, width + slack - col):
                out.append(raw)  # nearly done: finish the sentence on this line
                col += w
            else:
                out.append(brk + hang)
                col = hang_w
        return "".join(out)

    @staticmethod
    def _fits_to_sentence_end(tokens: list, i: int, budget: int) -> bool:
        """Whether the sentence running on from the space at *i* ends within
        *budget* characters. A paragraph end closes a sentence too."""
        total, last_word = 0, ""
        for j in range(i, len(tokens)):
            kind, _, w, visible = tokens[j]
            if kind in ("block", "lead"):
                return True
            if kind == "space" and j > i and _SENTENCE_END_RE.search(last_word):
                return True
            if kind == "word":
                last_word = visible
            total += w
            if total > budget:
                return False
        return True


class TooltipFormat(_TooltipFormatInternal):
    """Rich-text tooltip formatting DSL — ``kbd`` / ``hl`` / ``fmt`` — plus the
    layout rules every shown tooltip goes through (``wrap`` / ``display_ms``).

    Staticmethods so the module carries no top-level function *definitions*.
    Imported and called class-qualified across the ecosystem
    (``from pythontk import TooltipFormat`` then ``TooltipFormat.fmt(...)`` /
    ``TooltipFormat.kbd(...)`` / ``TooltipFormat.hl(...)``). Pure string work
    with no Qt in it, so a Qt-free engine surface (a tool's task definitions
    imported in a headless session) builds its tooltips with it too; uitk's
    ``TooltipPresenter`` shows the result.
    """

    @classmethod
    def is_rich(cls, text: str) -> bool:
        """Whether Qt will render *text* as rich text (``Qt.mightBeRichText``).

        Qt decides from the first line only: rich when that line's first tag
        names an element Qt knows, or ``&lt;`` comes before any tag.

        Parameters:
            text: Tooltip text.

        Returns:
            (bool) True when Qt treats *text* as rich text.
        """
        return cls._is_rich(text)

    @staticmethod
    def kbd(*keys: str) -> str:
        """Render keyboard key(s) as styled ``<kbd>``-like chips.

        Multiple keys are joined with " + " between chips, matching the
        convention used in keyboard shortcut docs (e.g. ``Ctrl`` + ``Z``).

        Example::

            f"{kbd('Ctrl', 'Z')} — Undo"
            f"Press {kbd('Enter')} to confirm"
        """
        chip = (
            "<span style='background:#2f2f2f; border:1px solid #555; "
            "border-radius:3px; padding:0 4px; font-family:monospace; "
            "font-size:90%; color:#e0e0e0'>{key}</span>"
        )
        return " + ".join(chip.format(key=k) for k in keys)

    @staticmethod
    def hl(text: str, color: str = _C_ACCENT) -> str:
        """Highlight ``text`` in ``color`` (defaults to the accent color).

        Use sparingly — color highlights work best for short terms (a feature
        name, a state, a value), not whole sentences.

        Example::

            f"Status: {hl('On', color='#7c7')}"
            f"{hl('Edges')} only — vertices and faces are ignored."
        """
        return f"<span style='color:{color}'>{text}</span>"

    @staticmethod
    def fmt(
        title: str = None,
        body: str = None,
        bullets: list = None,
        steps: list = None,
        rows: list = None,
        sections: list = None,
        notes: list = None,
    ) -> str:
        """Build a rich-text HTML tooltip string.

        Any combination of parameters may be supplied; sections are stacked in
        order: title → body → bullets → steps → rows → sections → notes.

        Parameters:
            title:    Header line shown above everything else, rendered in the
                      accent title color and bold.
            body:     Paragraph of plain prose beneath the title.
            bullets:  Strings rendered as an unordered ``<ul>`` list.
                      Inline HTML (e.g. ``<b>On:</b> …``) is supported.
            steps:    Strings rendered as a numbered ``<ol>`` list.
                      Use for sequential workflow instructions.
            rows:     ``(key, value)`` pairs rendered as a compact two-column table.
                      Keys are rendered in a muted colour; values in default colour.
            sections: ``(title, [items])`` pairs for multi-section tooltips.
                      Each section renders a colored sub-heading followed by a ``<ul>``.
            notes:    Strings rendered as italic muted "note:"-style callouts after
                      the main content. Use for caveats, tips, or "see also" hints.

        Returns:
            An HTML string that Qt's tooltip engine renders as rich text.

        See also:
            :meth:`TooltipFormat.kbd` for keyboard-shortcut chips.
            :meth:`TooltipFormat.hl` for inline color highlights.

        Example::

            fmt(
                title="Export Mode",
                bullets=[
                    "<b>Composite</b> — Single mixed WAV of all keyed clips.",
                    "<b>Keyed Tracks</b> — Individual source clips keyed on the timeline.",
                ],
                notes=[f"{kbd('Shift')} while clicking to keep the previous mode."],
            )

            fmt(
                title="Image to Plane",
                body="Creates textured polygon planes from images.",
                steps=["Press Browse…", "Choose material type.", "Press Create Planes."],
            )

            fmt(
                title="Shot Manifest",
                body="Build and validate shots from a CSV file or scene animation.",
                sections=[
                    ("Quick Start", ["Check CSV and browse to a file.", "Click Build."]),
                    ("Table Columns", ["<b>Step</b> — Step ID.", "<b>Start / End</b> — Frame range."]),
                ],
            )
        """
        parts = []
        if title:
            parts.append(
                f"<p style='margin:0 0 3px 0; color:{_C_TITLE}'><b>{title}</b></p>"
            )
        if body:
            parts.append(f"<p style='margin:2px 0'>{body}</p>")
        if bullets:
            items = "".join(f"<li>{b}</li>" for b in bullets)
            parts.append(f"<ul style='margin:3px 0; padding-left:14px'>{items}</ul>")
        if steps:
            items = "".join(f"<li>{s}</li>" for s in steps)
            parts.append(f"<ol style='margin:3px 0; padding-left:16px'>{items}</ol>")
        if rows:
            cells = "".join(
                f"<tr><td style='padding-right:8px; color:{_C_MUTED}'>{k}</td>"
                f"<td>{v}</td></tr>"
                for k, v in rows
            )
            parts.append(f"<table style='margin:3px 0'>{cells}</table>")
        if sections:
            for section_title, section_items in sections:
                parts.append(
                    f"<p style='margin:6px 0 1px 0; color:{_C_ACCENT}'>"
                    f"<b>{section_title}</b></p>"
                )
                items = "".join(f"<li>{item}</li>" for item in section_items)
                parts.append(
                    f"<ul style='margin:1px 0; padding-left:14px'>{items}</ul>"
                )
        if notes:
            for note in notes:
                parts.append(
                    f"<p style='margin:3px 0 0 0; color:{_C_NOTE}; font-style:italic'>"
                    f"<span style='color:{_C_MUTED}'>note:</span> {note}</p>"
                )
        return "".join(parts)

    @staticmethod
    def placeholder_preview(
        template: str,
        context: dict,
        *,
        title: str = None,
        body: str = None,
        descriptions: dict = None,
        wildcards: dict = None,
        final: str = None,
        final_label: str = "→",
        empty_text: str = None,
        notes: list = None,
    ) -> str:
        """Build a live, self-documenting tooltip for a pattern/template field.

        Combines *instruction* (what the field does + what each placeholder means)
        with a *live preview* (each placeholder's current value + the fully-resolved
        result). Designed for ``widget.tooltip.bind`` on a pattern ``QLineEdit`` so
        the hover both teaches the syntax and reflects the current text + live
        context — no manual refresh needed. Because ``bind`` replaces the widget's
        static tooltip, fold the field's help text in here (via *title* / *body* /
        *descriptions*) rather than leaving it on ``setToolTip``.

        Table contents:
            - With *descriptions*: every supported placeholder (the *descriptions*
              keys, in order) is listed as ``{token} | meaning | value`` — so the
              user sees all available keys and what they mean even when the current
              pattern uses only some. Any token typed in *template* that isn't a
              supported key is appended and flagged ``unknown``.
            - Without *descriptions*: only the tokens actually present in *template*
              are listed as ``{token} | value`` (a terse "resolves to" view).
            - With *wildcards*: each bare token is listed first, above the
              placeholder rows, borrowing the meaning and live value of the key it
              stands for — the field's whole wildcard vocabulary, and only that
              vocabulary, so a user reads what is supported rather than guessing
              from the general grammar.

        Resolution goes through :meth:`pythontk.StrUtils.resolve_placeholders`.
        Placeholder values and the resolved *final* are treated as **data** and
        HTML-escaped (a path may hold ``&``; a ``<none>`` sentinel must render
        literally). *title* / *body* / *descriptions* / *notes* / *final_label* are
        caller **markup**, passed through verbatim (so ``<b>…</b>`` still works).

        Parameters:
            template:     The current pattern text (typically ``lineedit.text()``).
            context:      ``{token: value}`` for every placeholder the field
                          supports, at its current live value.
            title:        Optional heading (the field's name).
            body:         Optional purpose paragraph — what the field controls.
            descriptions: Optional ``{token: meaning}`` for the supported keys.
                          Its presence + order drive the full-key table (see above).
            wildcards:    Optional ``{wildcard: key}`` — each bare token the field
                          accepts as sugar for one placeholder (e.g. ``{"*": "name"}``
                          where ``*`` means ``{name}``). Pass *template* already
                          expanded (``pythontk.StrUtils.expand_wildcard``) so the
                          preview resolves the same string the field will.
            final:        Optional fully-resolved string (e.g. an absolute output
                          path) shown under the table. ``None`` defaults to the
                          resolved ``result``; pass ``final=""`` to suppress it.
            final_label:  Muted label shown before *final* (default ``"→"``).
            empty_text:   Returned when *template* is blank **and** no instructional
                          content (*title* / *body* / *descriptions*) is supplied;
                          defaults to a muted "type a pattern…" hint.
            notes:        Extra caller notes (examples, a domain-specific typo
                          warning), appended after the auto unknown-token note.

        Returns:
            An HTML tooltip string (see :meth:`fmt`).

        Example::

            self.ui.txt_pattern.tooltip.bind(
                lambda: TooltipFormat.placeholder_preview(
                    self.ui.txt_pattern.text(),
                    {"scenes": self.scenes_dir, "name": self.scene_name},
                    title="Folder Structure",
                    body="Subfolder pattern for <b>Save</b>.",
                    descriptions={
                        "scenes": "workspace scenes folder",
                        "name": "scene name (excludes the suffix)",
                    },
                    final=self.resolved_dir,
                    final_label="save dir →",
                )
            )
        """
        from html import escape as _esc

        from pythontk import StrUtils

        tmpl = (template or "").strip()
        # Nothing to say and no pattern yet — just the muted hint.
        if not tmpl and not (title or body or descriptions):
            return empty_text or (
                f"<p style='margin:0; color:{_C_MUTED}; font-style:italic'>"
                f"Type a pattern to preview its resolved value.</p>"
            )

        try:
            info = StrUtils.resolve_placeholders(tmpl, **(context or {}))
        except ValueError as e:
            return TooltipFormat.fmt(
                title=title, body=body, notes=[f"invalid pattern: {e}"]
            )

        ctx = context or {}

        def _value_cell(name):
            """The colour-coded live-value cell for one placeholder. A supported key
            has a value straight from *context* (present even when the current
            pattern doesn't use it); a token typed but not supplied is flagged."""
            if name in ctx:
                val = format(ctx[name])
                return (
                    TooltipFormat.hl(_esc(val))
                    if val
                    else f"<i style='color:{_C_MUTED}'>(empty)</i>"
                )
            return TooltipFormat.hl(
                "unknown" if descriptions is not None else "unresolved",
                color=_C_NOTE,
            )

        # Which placeholders to list, and in what order.
        if descriptions is not None:
            names = list(descriptions.keys())
            for n in info["fields"]:  # surface typed-but-unsupported tokens too
                if n not in descriptions:
                    names.append(n)
        else:
            names = list(info["fields"])

        def _row(token, name, meaning):
            """One table row: the literal token, its meaning, its live value."""
            cells = [f"<td style='padding-right:8px'>{_esc(token)}</td>"]
            if descriptions is not None:
                # author markup — not escaped
                cells.append(
                    f"<td style='padding-right:8px; color:{_C_MUTED}'>{meaning}</td>"
                )
            cells.append(f"<td>{_value_cell(name)}</td>")
            return "<tr>" + "".join(cells) + "</tr>"

        rows = []
        # The wildcard vocabulary first — it is the shorthand a user reaches for,
        # and each entry reads off the placeholder it stands for.
        for token, name in (wildcards or {}).items():
            rows.append(_row(token, name, (descriptions or {}).get(name, "")))
        for name in names:
            rows.append(
                _row("{" + name + "}", name, (descriptions or {}).get(name, ""))
            )
        table_html = (
            f"<table style='margin:3px 0'>{''.join(rows)}</table>" if rows else ""
        )

        note_list = []
        if info["unresolved"]:
            toks = ", ".join(_esc("{" + n + "}") for n in info["unresolved"])
            label = "unknown" if descriptions is not None else "unresolved"
            note_list.append(f"{label}: {toks}")
        if notes:
            note_list.extend(notes)

        resolved_final = info["result"] if final is None else final
        final_html = (
            f"<p style='margin:4px 0 0 0'>"
            f"<span style='color:{_C_MUTED}'>{final_label}</span> "
            f"<span style='color:{_C_TITLE}'>{_esc(resolved_final)}</span></p>"
            if resolved_final
            else ""
        )

        # instruction (title + body) → table → resolved-final line → warnings.
        return (
            TooltipFormat.fmt(title=title, body=body)
            + table_html
            + final_html
            + (TooltipFormat.fmt(notes=note_list) if note_list else "")
        )

    #: Default cap on how many entries :meth:`stored_items` lists before it
    #: elides the tail. Sized so a hover stays a glance rather than a panel --
    #: past roughly this many rows Qt's popup is taller than the widget that
    #: spawned it, and the count line is the only part still worth reading.
    STORED_ITEMS_MAX = 12

    @staticmethod
    def stored_items(
        items,
        *,
        title: str = None,
        body: str = None,
        formatter=None,
        max_items: int = None,
        noun: str = "item(s)",
        empty_text: str = None,
        notes: list = None,
    ) -> str:
        """Build a live tooltip listing what a control currently has STORED.

        The counterpart to :meth:`placeholder_preview` for the other common
        "this widget owns hidden state" shape: a *Set From Selection* button
        whose captured set is invisible until something consumes it. Bound with
        ``widget.tooltip.bind``, the hover answers "what did I capture, and is
        it still what I want?" without a round trip through the tool.

        Renders instruction (*title* / *body*), then the live count, then the
        entries -- capped at *max_items* with an elided-tail line, because a
        captured selection can run to hundreds of objects and a tooltip that
        long is both unreadable and slow to paint.

        Entries are treated as **data** and HTML-escaped (a node name may hold
        ``&`` or ``<``). *title* / *body* / *noun* / *empty_text* / *notes* are
        caller **markup**, passed through verbatim.

        Parameters:
            items:      The stored entries (``None`` reads as empty).
            title:      Optional heading -- typically the control's own name.
                        Because ``bind`` replaces the widget's static tooltip,
                        fold its help text in here rather than leaving it on
                        ``setToolTip``.
            body:       Optional purpose paragraph -- what the control captures.
            formatter:  ``callable(item) -> str`` producing each entry's text.
                        Defaults to ``str``. Use it to shorten a DCC's long path
                        (``lambda n: n.rsplit("|", 1)[-1]``) or to read a name
                        off a live object (``lambda o: o.name``).
            max_items:  Cap on listed entries. ``None`` uses
                        :attr:`STORED_ITEMS_MAX`; a non-positive value lists
                        every entry.
            noun:       Noun phrase for the count line, e.g.
                        ``"stored source mesh(es)"``.
            empty_text: Shown instead of the list when nothing is stored.
            notes:      Extra callouts appended after the list (see :meth:`fmt`)
                        -- a stale-entry warning, a "what reads this" pointer.

        Returns:
            An HTML tooltip string (see :meth:`fmt`).

        Example::

            self.sb.tooltip.bind(btn, self._source_tooltip)

            def _source_tooltip(self):
                return TooltipFormat.stored_items(
                    self._sources,
                    title="Set Source From Selection",
                    body="Capture the current selection as the stored sources.",
                    formatter=lambda n: n.rsplit("|", 1)[-1],
                    noun="stored source mesh(es)",
                    empty_text="Nothing stored yet.",
                )
        """
        from html import escape as _esc

        # Not ``items or []``: that asks the input for its truth value, which
        # a numpy array refuses outright and a generator answers without
        # consuming. ``None`` is the only "nothing" this has to absorb.
        entries = [] if items is None else list(items)
        head = TooltipFormat.fmt(title=title, body=body)
        tail = TooltipFormat.fmt(notes=notes) if notes else ""

        if not entries:
            return (
                head
                + f"<p style='margin:3px 0 0 0; color:{_C_MUTED}; font-style:italic'>"
                f"{empty_text or 'Nothing stored yet.'}</p>" + tail
            )

        cap = TooltipFormat.STORED_ITEMS_MAX if max_items is None else max_items
        shown = entries if cap <= 0 else entries[:cap]
        render = formatter or str
        rows = "".join(f"<li>{_esc(str(render(item)))}</li>" for item in shown)
        hidden = len(entries) - len(shown)
        return (
            head + f"<p style='margin:4px 0 1px 0'>"
            f"{TooltipFormat.hl(str(len(entries)))} {noun}</p>"
            + f"<ul style='margin:1px 0; padding-left:14px'>{rows}</ul>"
            + (
                f"<p style='margin:0; color:{_C_MUTED}; font-style:italic'>"
                f"\u2026and {hidden} more</p>"
                if hidden
                else ""
            )
            + tail
        )

    # --- Layout: applied to every tooltip TooltipPresenter shows ------------

    #: Soft line width, in characters (~45-75 is the readable band; Qt's own
    #: rich-text wrap is ~80 average-width characters, and plain text it never
    #: wraps below the screen width). A non-positive value disables wrapping.
    WRAP_WIDTH = 60
    #: How far past :attr:`WRAP_WIDTH` a line may run -- only to finish the
    #: sentence it is in, so a nearly complete one isn't broken to strand its
    #: last word or two on a line of their own.
    WRAP_SLACK = 15
    #: Display-time floor: Qt's own base timer.
    DISPLAY_BASE_MS = 10000
    #: Display time added per visible word -- 400 ms is ~150 words a minute, an
    #: unhurried read with room to glance back at the control.
    DISPLAY_MS_PER_WORD = 400

    @classmethod
    def wrap(
        cls, text: str, width: int = None, slack: int = None, rich: bool = None
    ) -> str:
        """Break *text* into lines of a readable width, as a tooltip shows it.

        Greedy, word by word: a line breaks before the word that would take it
        past *width* -- unless the sentence under way ends within *slack* more
        characters, in which case the line runs on to finish it. Authored line
        breaks, paragraphs and list items are kept; a word is never split, so an
        unbreakable token may still run wide -- except a path too wide for any
        line in rich text, which breaks after its separators (the nowrap
        container would otherwise keep it on one line past the screen edge;
        Qt still wraps a plain one at the screen width).

        Plain text gets ``\\n`` breaks (a bullet's continuation lines hang under
        its text) and stays plain. Rich text gets ``<br>`` breaks, counting only
        visible characters (markup is free, an entity is one), and comes back
        inside a ``white-space:nowrap`` container, without which Qt squeezes a
        short rich tooltip narrow and re-wraps it. ``<pre>`` / ``<nobr>`` /
        ``white-space:pre|nowrap`` content is left as written. Idempotent.

        Parameters:
            text: Tooltip text, plain or rich.
            width: Soft line width in characters. Default :attr:`WRAP_WIDTH`;
                non-positive returns *text* unchanged.
            slack: The sentence-finishing allowance. Default :attr:`WRAP_SLACK`.
            rich: Whether *text* is rich text. ``None`` decides exactly as Qt
                will (``Qt.mightBeRichText``, which reads only the first line):
                breaking a plain tooltip as HTML would make Qt render it as HTML.

        Returns:
            (str) The wrapped text.
        """
        width = cls.WRAP_WIDTH if width is None else width
        slack = cls.WRAP_SLACK if slack is None else slack
        if not text or width <= 0:
            return text
        if rich is None:
            rich = cls._is_rich(text)
        if not rich:
            return cls._layout(cls._plain_tokens(text), width, slack, "\n")
        if text.startswith(cls._NOWRAP_OPEN):
            return text
        body = cls._layout(
            cls._rich_tokens(text), width, slack, "<br>", split_paths=True
        )
        return f"{cls._NOWRAP_OPEN}{body}</div>"

    @classmethod
    def display_ms(cls, text: str, rich: bool = None) -> int:
        """How long a tooltip showing *text* should stay up, in milliseconds.

        :attr:`DISPLAY_BASE_MS` plus :attr:`DISPLAY_MS_PER_WORD` per visible
        word, and never less than Qt's own timer (``10 s + 40 ms`` per visible
        character past 100), which a word count undersells for one long token.

        Parameters:
            text: Tooltip text, plain or rich (markup is not counted).
            rich: Whether *text* is rich text; ``None`` decides as Qt will.

        Returns:
            (int) The display time.
        """
        if rich is None:
            rich = cls._is_rich(text)
        visible = cls._visible_text(text) if rich else text
        qt_default = 10000 + 40 * max(0, len(visible) - 100)
        by_words = cls.DISPLAY_BASE_MS + cls.DISPLAY_MS_PER_WORD * len(visible.split())
        return max(qt_default, by_words)
