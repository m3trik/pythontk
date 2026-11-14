#!/usr/bin/python
# coding=utf-8
"""Unit tests for pythontk.TooltipFormat -- the rich-text tooltip DSL.

``kbd`` / ``hl`` / ``fmt`` / ``placeholder_preview`` / ``stored_items`` and the
layout rules (``wrap`` / ``display_ms``). Pure string work, so no Qt: the check
that the plain/rich detector agrees with ``Qt.mightBeRichText`` lives in uitk's
``test_tooltip_mixin`` (it needs a binding), beside ``TooltipPresenter``.

Run with:
    python -m pytest test_tooltip_format.py -v
"""

import unittest

from pythontk import TooltipFormat


class TestKbd(unittest.TestCase):
    """Tests for kbd() keyboard chip helper."""

    def test_single_key_renders_as_chip(self):
        html = TooltipFormat.kbd("Enter")
        self.assertIn("Enter", html)
        self.assertIn("<span", html)
        # The chip styling distinguishes it from normal text
        self.assertIn("border-radius", html)

    def test_multiple_keys_joined_with_plus(self):
        html = TooltipFormat.kbd("Ctrl", "Z")
        self.assertIn("Ctrl", html)
        self.assertIn("Z", html)
        self.assertIn(" + ", html)
        # Each key gets its own <span>
        self.assertEqual(html.count("<span"), 2)

    def test_no_keys_returns_empty(self):
        self.assertEqual(TooltipFormat.kbd(), "")


class TestHl(unittest.TestCase):
    """Tests for hl() inline-highlight helper."""

    def test_wraps_text_with_color_span(self):
        html = TooltipFormat.hl("foo")
        self.assertIn("foo", html)
        self.assertIn("<span", html)
        self.assertIn("color:", html)

    def test_custom_color(self):
        html = TooltipFormat.hl("warn", color="#f00")
        self.assertIn("#f00", html)
        self.assertIn("warn", html)


class TestFmt(unittest.TestCase):
    """Tests for fmt() rich-text tooltip builder."""

    def test_empty_call_returns_empty(self):
        self.assertEqual(TooltipFormat.fmt(), "")

    def test_title_only(self):
        html = TooltipFormat.fmt(title="My Tool")
        self.assertIn("My Tool", html)
        self.assertIn("<b>", html)

    def test_body_only(self):
        html = TooltipFormat.fmt(body="Tool description.")
        self.assertIn("Tool description.", html)
        self.assertIn("<p", html)

    def test_bullets_render_as_unordered_list(self):
        html = TooltipFormat.fmt(bullets=["First", "Second"])
        self.assertIn("<ul", html)
        self.assertIn("<li>First</li>", html)
        self.assertIn("<li>Second</li>", html)

    def test_steps_render_as_ordered_list(self):
        html = TooltipFormat.fmt(steps=["Open file", "Click button"])
        self.assertIn("<ol", html)
        self.assertIn("<li>Open file</li>", html)
        self.assertIn("<li>Click button</li>", html)

    def test_rows_render_as_table(self):
        html = TooltipFormat.fmt(rows=[("Type", "int"), ("Default", "0")])
        self.assertIn("<table", html)
        self.assertIn("<td", html)
        self.assertIn("Type", html)
        self.assertIn("int", html)

    def test_sections_render_with_headings(self):
        html = TooltipFormat.fmt(sections=[("Quick Start", ["Step 1", "Step 2"])])
        self.assertIn("Quick Start", html)
        self.assertIn("<li>Step 1</li>", html)
        self.assertIn("<li>Step 2</li>", html)

    def test_notes_render_after_main_content(self):
        html = TooltipFormat.fmt(title="X", notes=["Tip: use Ctrl-click."])
        self.assertIn("Tip: use Ctrl-click.", html)
        self.assertIn("note:", html)
        # Notes come after the title
        self.assertLess(html.index("X"), html.index("Tip: use Ctrl-click."))

    def test_ordering_title_body_bullets_steps_rows_sections_notes(self):
        html = TooltipFormat.fmt(
            title="T",
            body="B",
            bullets=["bul"],
            steps=["stp"],
            rows=[("k", "v")],
            sections=[("Sec", ["si"])],
            notes=["nt"],
        )
        # Verify each segment appears in declared order.
        order = ["T", "B", "bul", "stp", "k", "Sec", "si", "nt"]
        positions = [html.index(s) for s in order]
        self.assertEqual(positions, sorted(positions))

    def test_inline_html_in_bullets_is_preserved(self):
        html = TooltipFormat.fmt(bullets=["<b>Bold</b> — desc"])
        self.assertIn("<b>Bold</b>", html)

    def test_kbd_embeds_into_bullets(self):
        html = TooltipFormat.fmt(bullets=[f"{TooltipFormat.kbd('Ctrl', 'Z')} — Undo"])
        self.assertIn("Ctrl", html)
        self.assertIn("Z", html)
        self.assertIn("Undo", html)


class TestPlaceholderPreview(unittest.TestCase):
    """Tests for placeholder_preview() live resolved-token tooltip builder."""

    def test_resolved_tokens_render_as_rows_with_values(self):
        html = TooltipFormat.placeholder_preview(
            "{scenes}/{name}",
            {"scenes": "scenes", "name": "shot"},
        )
        self.assertIn("{scenes}", html)
        self.assertIn("{name}", html)
        self.assertIn("scenes", html)
        self.assertIn("shot", html)
        self.assertIn("<table", html)

    def test_final_line_shows_resolved_path(self):
        html = TooltipFormat.placeholder_preview(
            "{scenes}/{name}",
            {"scenes": "scenes", "name": "shot"},
            final="C:/proj/scenes/shot",
            final_label="save dir →",
        )
        self.assertIn("C:/proj/scenes/shot", html)
        self.assertIn("save dir →", html)

    def test_final_defaults_to_resolved_result(self):
        html = TooltipFormat.placeholder_preview(
            "{scenes}/{name}", {"scenes": "scenes", "name": "shot"}
        )
        self.assertIn("scenes/shot", html)

    def test_final_empty_string_suppresses_line(self):
        html = TooltipFormat.placeholder_preview("{name}", {"name": "shot"}, final="")
        # The value still appears in the table, but there is no "→" final line.
        self.assertIn("shot", html)
        self.assertNotIn("→", html)

    def test_unresolved_token_flagged_in_note(self):
        html = TooltipFormat.placeholder_preview("{name}/{missing}", {"name": "shot"})
        self.assertIn("unresolved", html)
        self.assertIn("{missing}", html)
        self.assertIn("note:", html)

    def test_wildcard_rows_borrow_the_meaning_and_value_of_their_key(self):
        """A field that accepts a bare token as sugar for one placeholder
        documents both, and the wildcard reads first."""
        html = TooltipFormat.placeholder_preview(
            "WIP_{name}",
            {"name": "myScene", "date": "2026-09-12"},
            descriptions={"name": "the default name", "date": "YYYY-MM-DD"},
            wildcards={"*": "name"},
            final="",  # table only — the resolved line repeats the value
        )
        self.assertIn("<td style='padding-right:8px'>*</td>", html)
        self.assertLess(html.index(">*<"), html.index("{name}"))
        # The wildcard row carries its key's meaning and its live value.
        self.assertEqual(html.count("the default name"), 2)
        self.assertEqual(html.count("myScene"), 2)

    def test_wildcards_are_the_only_bare_tokens_documented(self):
        """Only the listed vocabulary is exposed — a token the field does not
        accept never appears as a row."""
        html = TooltipFormat.placeholder_preview(
            "{name}", {"name": "x"}, descriptions={"name": "n"}, wildcards={"*": "name"}
        )
        self.assertNotIn(">?<", html)

    def test_extra_notes_appended(self):
        html = TooltipFormat.placeholder_preview(
            "{scenes}", {"scenes": "scenes"}, notes=["<b>{scene}</b> is a typo"]
        )
        self.assertIn("is a typo", html)

    def test_descriptions_show_all_keys_even_if_unused_in_pattern(self):
        """The instructional view lists every supported key + meaning + value,
        even keys the current pattern doesn't use (so the help isn't lost)."""
        html = TooltipFormat.placeholder_preview(
            "{scenes}",  # pattern uses only one key
            {"scenes": "scenes", "name": "shot", "ws": "MyGame"},
            title="Folder Structure",
            body="Subfolder pattern for Save.",
            descriptions={
                "scenes": "workspace scenes folder",
                "name": "scene name (excludes the suffix)",
                "ws": "workspace folder name",
            },
        )
        # purpose (body) + every key's meaning + every key's current value.
        self.assertIn("Subfolder pattern for Save.", html)
        for token, meaning, value in (
            ("{scenes}", "workspace scenes folder", "scenes"),
            ("{name}", "scene name (excludes the suffix)", "shot"),
            ("{ws}", "workspace folder name", "MyGame"),
        ):
            self.assertIn(token, html)
            self.assertIn(meaning, html)
            self.assertIn(value, html)

    def test_descriptions_flag_unknown_typed_token(self):
        """A typed token that isn't a supported key is appended + flagged unknown."""
        html = TooltipFormat.placeholder_preview(
            "{scenes}/{nmae}",  # typo: nmae
            {"scenes": "scenes", "name": "shot"},
            descriptions={"scenes": "the scenes folder", "name": "the scene name"},
        )
        self.assertIn("{nmae}", html)
        self.assertIn("unknown", html)

    def test_body_renders_without_title(self):
        html = TooltipFormat.placeholder_preview(
            "{name}", {"name": "x"}, body="What this field does."
        )
        self.assertIn("What this field does.", html)

    def test_meanings_are_not_escaped(self):
        """Descriptions are author markup — inline HTML must survive."""
        html = TooltipFormat.placeholder_preview(
            "{name}",
            {"name": "x"},
            descriptions={"name": "the <b>scene</b> name"},
        )
        self.assertIn("the <b>scene</b> name", html)

    def test_blank_template_with_instruction_still_shows_help(self):
        """Clearing the field must not wipe the purpose/keys — instruction persists."""
        html = TooltipFormat.placeholder_preview(
            "",
            {"scenes": "scenes"},
            title="Folder Structure",
            body="Subfolder pattern.",
            descriptions={"scenes": "the scenes folder"},
        )
        self.assertIn("Folder Structure", html)
        self.assertIn("the scenes folder", html)
        self.assertNotIn("Type a pattern", html)

    def test_blank_template_returns_hint(self):
        html = TooltipFormat.placeholder_preview("   ", {"name": "x"})
        self.assertIn("Type a pattern", html)

    def test_blank_template_custom_empty_text(self):
        html = TooltipFormat.placeholder_preview("", {}, empty_text="nothing yet")
        self.assertEqual(html, "nothing yet")

    def test_empty_value_marked(self):
        html = TooltipFormat.placeholder_preview("{suffix}", {"suffix": ""})
        self.assertIn("(empty)", html)

    def test_data_values_are_html_escaped(self):
        """Token values are data: '<none>' / '&' must render literally, not as markup."""
        html = TooltipFormat.placeholder_preview(
            "{name}/{ws}",
            {"name": "<none>", "ws": "Rock & Roll"},
            final="C:/x/<none>/Rock & Roll",
        )
        self.assertIn("&lt;none&gt;", html)
        self.assertNotIn("<none>", html)  # would be eaten by Qt's tag parser
        self.assertIn("Rock &amp; Roll", html)

    def test_notes_are_not_escaped(self):
        """Notes are caller markup — bold etc. must survive."""
        html = TooltipFormat.placeholder_preview(
            "{name}", {"name": "x"}, notes=["<b>{scene}</b> is a typo"]
        )
        self.assertIn("<b>{scene}</b>", html)

    def test_invalid_pattern_reports_note(self):
        # A lone '{' is a malformed format string.
        html = TooltipFormat.placeholder_preview("{name}/{", {"name": "x"})
        self.assertIn("invalid pattern", html)

    def test_ordering_table_before_final_before_notes(self):
        html = TooltipFormat.placeholder_preview(
            "{name}/{missing}",
            {"name": "shot"},
            title="Resolves to:",
            final="X_FINAL_X",
        )
        # table (name) -> final line -> unresolved note ("note:" anchors the note,
        # since "unresolved" also appears in the missing token's table cell).
        self.assertLess(html.index("shot"), html.index("X_FINAL_X"))
        self.assertLess(html.index("X_FINAL_X"), html.index("note:"))


class TestStoredItems(unittest.TestCase):
    """`stored_items` — the live "what did I capture?" list, with a hard cap."""

    def test_lists_every_entry_under_the_cap(self):
        html = TooltipFormat.stored_items(["a", "b", "c"])
        for name in ("a", "b", "c"):
            self.assertIn(f"<li>{name}</li>", html)
        self.assertNotIn("more", html)

    def test_count_reflects_the_whole_set_not_the_shown_slice(self):
        html = TooltipFormat.stored_items([f"m{i}" for i in range(40)], max_items=3)
        self.assertIn(">40<", html)  # the count is of everything stored
        self.assertEqual(html.count("<li>"), 3)

    def test_truncates_with_an_elided_tail_line(self):
        html = TooltipFormat.stored_items(["a", "b", "c", "d", "e"], max_items=2)
        self.assertIn("<li>a</li>", html)
        self.assertIn("<li>b</li>", html)
        self.assertNotIn("<li>c</li>", html)
        self.assertIn("3 more", html)

    def test_default_cap_is_the_class_constant(self):
        n = TooltipFormat.STORED_ITEMS_MAX
        html = TooltipFormat.stored_items([str(i) for i in range(n + 5)])
        self.assertEqual(html.count("<li>"), n)
        self.assertIn("5 more", html)

    def test_non_positive_cap_lists_everything(self):
        html = TooltipFormat.stored_items(list("abcdefgh"), max_items=0)
        self.assertEqual(html.count("<li>"), 8)
        self.assertNotIn("more", html)

    def test_formatter_shortens_entries(self):
        html = TooltipFormat.stored_items(
            ["|grp|pCube1", "|grp|pCube2"], formatter=lambda n: n.rsplit("|", 1)[-1]
        )
        self.assertIn("<li>pCube1</li>", html)
        self.assertNotIn("grp", html)

    def test_entries_are_escaped_but_caller_markup_is_not(self):
        """A node name is DATA (it can hold `&`/`<`); title/body are markup."""
        html = TooltipFormat.stored_items(
            ["a<b>&c"], title="Sources", body="Pick <b>meshes</b>."
        )
        self.assertIn("a&lt;b&gt;&amp;c", html)
        self.assertIn("<b>meshes</b>", html)

    def test_empty_renders_the_empty_text_and_no_list(self):
        html = TooltipFormat.stored_items([], empty_text="Nothing captured.")
        self.assertIn("Nothing captured.", html)
        self.assertNotIn("<li>", html)

    def test_none_reads_as_empty(self):
        self.assertNotIn("<li>", TooltipFormat.stored_items(None))

    def test_instruction_and_notes_survive_the_empty_case(self):
        """Bind replaces the static tooltip, so the help text must still show
        when nothing is stored — that is exactly when it is needed."""
        html = TooltipFormat.stored_items(
            [], title="Set Source", body="Capture the selection.", notes=["a caveat"]
        )
        self.assertIn("Set Source", html)
        self.assertIn("Capture the selection.", html)
        self.assertIn("a caveat", html)

    def test_noun_labels_the_count(self):
        html = TooltipFormat.stored_items(["a"], noun="stored source mesh(es)")
        self.assertIn("stored source mesh(es)", html)


#: 85 characters, one sentence -- over any readable soft width on its own.
_SENT = (
    "Exports the selected objects to the configured output folder using the "
    "active preset."
)


class _WrapCase(unittest.TestCase):
    """Pins the width knobs so a retuned default can't move these assertions."""

    W, S = 60, 15

    def wrap(self, text, rich=False, **kw):
        kw.setdefault("width", self.W)
        kw.setdefault("slack", self.S)
        return TooltipFormat.wrap(text, rich=rich, **kw)

    @staticmethod
    def visible_lines(html):
        """The lines a rich tooltip renders: split at every break, markup dropped."""
        import html as _html
        import re

        body = re.sub(
            r"(?i)<br\s*/?>|</?(p|li|ul|ol|div|tr|td|table)\b[^>]*>", "\n", html
        )
        text = _html.unescape(re.sub(r"<[^>]*>", "", body))
        return [" ".join(ln.split()) for ln in text.split("\n") if ln.strip()]


class TestWrapPlain(_WrapCase):
    """``wrap`` breaks PLAIN text at a readable width.

    Qt never wraps a plain-text tooltip until it is wider than the whole screen
    (``QTipLabel::updateSize`` turns word-wrap on for rich text only), so a
    three-sentence ``.ui`` tooltip rendered as ONE line 1333px wide (measured,
    Segoe UI 9pt on a 1920px screen). The presenter hands Qt pre-broken lines.
    """

    def test_a_long_paragraph_breaks_near_the_soft_width(self):
        text = " ".join([_SENT] * 3)
        lines = self.wrap(text).split("\n")
        self.assertGreater(len(lines), 3)
        for line in lines:
            self.assertLessEqual(len(line), self.W + self.S, line)
        # Words are only moved between lines -- never dropped, split, or reordered.
        self.assertEqual(" ".join(lines).split(), text.split())

    def test_only_a_finishing_sentence_may_pass_the_soft_width(self):
        """The one licence to run long: the sentence is nearly done."""
        text = " ".join([_SENT, "Then it opens the folder.", _SENT, "Done."])
        for line in self.wrap(text).split("\n"):
            if len(line) > self.W:
                self.assertRegex(line, r"[.!?:;]$", f"ran long mid-sentence: {line!r}")

    def test_a_nearly_complete_sentence_finishes_on_its_line(self):
        # Over the soft width, but it ends inside the slack: breaking would strand
        # a single word on a line of its own.
        text = "Rebuilds the preview mesh from the current source selection now."
        self.assertTrue(self.W < len(text) <= self.W + self.S)
        self.assertEqual(self.wrap(text), text)

    def test_a_sentence_far_from_done_breaks_at_the_soft_width(self):
        text = (
            "Rebuilds the preview mesh from the current source selection and "
            "then re-projects every stored UV set onto the rebuilt surface"
        )
        first = self.wrap(text).split("\n")[0]
        self.assertLessEqual(len(first), self.W)

    def test_short_text_is_unchanged(self):
        self.assertEqual(self.wrap("Write to disk"), "Write to disk")

    def test_authored_line_breaks_are_kept(self):
        text = "Line one.\nLine two."
        self.assertEqual(self.wrap(text), text)
        wrapped = self.wrap(_SENT * 2 + "\nShort tail.")
        self.assertTrue(wrapped.endswith("\nShort tail."))

    def test_a_bullet_hangs_its_continuation_lines(self):
        wrapped = self.wrap("- " + " ".join([_SENT] * 2))
        lines = wrapped.split("\n")
        self.assertTrue(lines[0].startswith("- "))
        for line in lines[1:]:
            self.assertTrue(line.startswith("  ") and not line.startswith("   "), line)

    def test_an_unbreakable_token_is_never_split(self):
        path = "C:/projects/" + "very_long_folder_name/" * 5 + "scene.ma"
        wrapped = self.wrap(f"Saves to {path} when done.")
        self.assertIn(path, wrapped)

    def test_wrapping_is_idempotent(self):
        once = self.wrap(" ".join([_SENT] * 4))
        self.assertEqual(self.wrap(once), once)


class TestWrapRich(_WrapCase):
    """``wrap`` breaks RICH text with ``<br>`` and keeps Qt from re-wrapping it.

    Qt does wrap rich text, but its sizing squeezes anything under four lines to
    half, then a quarter, of its ~80-character width: a two-line tooltip came out
    155px wide and six lines tall (measured). The explicit breaks only hold inside
    a ``white-space:nowrap`` container.
    """

    def test_breaks_are_br_inside_a_nowrap_container(self):
        wrapped = self.wrap(f"<p>{' '.join([_SENT] * 3)}</p>", rich=True)
        self.assertIn("<br>", wrapped)
        self.assertIn("white-space:nowrap", wrapped.split(">", 1)[0])
        for line in self.visible_lines(wrapped):
            self.assertLessEqual(len(line), self.W + self.S, line)

    def test_markup_does_not_count_toward_the_width(self):
        # 12 words, 59 visible characters -- but ~450 characters of markup.
        text = " ".join(["<span style='color:#6fb5d6'>word</span>"] * 12)
        self.assertNotIn("<br>", self.wrap(text, rich=True))

    def test_an_entity_counts_as_one_character(self):
        text = "<b>x</b> " + "&amp; " * 25  # 51 visible characters
        self.assertNotIn("<br>", self.wrap(text, rich=True))

    def test_block_tags_restart_the_line(self):
        half = "Exports the selected objects to the output folder."  # 50 chars
        text = f"<p>{half}</p><p>{half}</p><ul><li>{half}</li></ul>"
        self.assertNotIn("<br>", self.wrap(text, rich=True))

    def test_list_items_wrap_individually(self):
        wrapped = self.wrap(
            f"<ul><li>{_SENT} {_SENT}</li><li>Short.</li></ul>", rich=True
        )
        self.assertIn("<br>", wrapped.split("</li>")[0])
        self.assertIn("<li>Short.</li>", wrapped)

    def test_preformatted_content_is_left_alone(self):
        pre = f"<pre>{_SENT} {_SENT}</pre>"
        self.assertIn(pre, self.wrap(f"<p>Log:</p>{pre}", rich=True))

    def test_a_path_too_wide_for_any_line_breaks_at_its_separators(self):
        # Inside the nowrap container Qt never re-wraps: an unbroken 130-char
        # path ran off the screen edge where Qt used to wrap it.
        path = "C:/projects/" + "very_long_folder_name/" * 5 + "scene.ma"
        lines = self.visible_lines(
            self.wrap(f"<p>Saves to {path} when done.</p>", rich=True)
        )
        for line in lines:
            self.assertLessEqual(len(line), self.W + self.S, line)
        self.assertIn(path, "".join(lines).replace(" ", ""))
        self.assertTrue(any(line.endswith("/") for line in lines), lines)

    def test_the_fmt_dsl_output_wraps(self):
        html = TooltipFormat.fmt(title="Export", body=" ".join([_SENT] * 2))
        lines = self.visible_lines(self.wrap(html, rich=True))
        self.assertEqual(lines[0], "Export")
        self.assertGreater(len(lines), 2)

    def test_wrapping_is_idempotent(self):
        once = self.wrap(f"<p>{' '.join([_SENT] * 3)}</p>", rich=True)
        self.assertEqual(self.wrap(once, rich=True), once)


class TestDisplayMs(unittest.TestCase):
    """The dynamic display time: longer content stays up longer.

    Qt's own timer is ``10s + 40ms`` per character past 100 -- about 18s for a
    300-character help text, which reads slower than that on a squeezed or
    screen-wide line. ``display_ms`` scales with the words actually shown.
    """

    @staticmethod
    def qt_default(chars):
        return 10000 + 40 * max(0, chars - 100)

    def test_longer_content_stays_up_longer(self):
        short = TooltipFormat.display_ms("Write to disk")
        long_ = TooltipFormat.display_ms(" ".join([_SENT] * 4))
        self.assertGreater(long_, short)

    def test_never_shorter_than_qts_own_timer(self):
        for text in ("x", _SENT, " ".join([_SENT] * 5), "C:/" + "a" * 400):
            self.assertGreaterEqual(
                TooltipFormat.display_ms(text, rich=False), self.qt_default(len(text))
            )

    def test_a_long_help_text_gets_meaningfully_longer_than_qt(self):
        text = " ".join([_SENT] * 4)  # ~340 chars, 52 words
        self.assertGreater(
            TooltipFormat.display_ms(text, rich=False), self.qt_default(len(text)) * 1.4
        )

    def test_markup_does_not_count(self):
        plain = "alpha beta gamma"
        rich = "<span style='color:#6fb5d6'>alpha</span> <b>beta</b> <i>gamma</i>"
        self.assertEqual(
            TooltipFormat.display_ms(plain, rich=False),
            TooltipFormat.display_ms(rich, rich=True),
        )


if __name__ == "__main__":
    unittest.main()
