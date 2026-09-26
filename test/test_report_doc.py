#!/usr/bin/python
# coding=utf-8
"""Tests for pythontk.ReportDoc -- the block report rendered as HTML or plain text."""

import unittest

from pythontk import ReportDoc


class TestEscaping(unittest.TestCase):
    """Scene data is text, never markup -- in cells, fields, items and headings."""

    def test_plain_values_are_escaped_in_html(self):
        # The regression that motivated the class: a histogram bucket named
        # "<512" was captured into HTML unescaped and vanished as a tag.
        doc = ReportDoc().fields([("<512", "a & b")]).table(["<th>"], [["x<y"]])
        html = doc.to_html()
        self.assertIn("&lt;512", html)
        self.assertIn("a &amp; b", html)
        self.assertIn("x&lt;y", html)
        self.assertIn("&lt;th&gt;", html)
        self.assertNotIn("<512", html)

    def test_text_rendering_keeps_values_verbatim(self):
        text = ReportDoc().fields([("<512", "a & b")]).to_text()
        self.assertIn("<512", text)
        self.assertIn("a & b", text)

    def test_html_has_no_newlines(self):
        # A viewer that turns free newlines into <br> would tear tables apart.
        doc = ReportDoc().heading("T", 1).text("a\tb").table(["A", "B"], [[1, 2]])
        doc.fields([("k", "v")]).items(["i"])
        self.assertNotIn("\n", doc.to_html())

    def test_a_newline_in_the_data_is_a_line_break_not_a_free_newline(self):
        doc = ReportDoc().table(["A"], [["one\ntwo"]]).text(ReportDoc.span("x\ny"))
        html = doc.to_html()
        self.assertNotIn("\n", html)
        self.assertIn("one<br>two", html)
        self.assertIn("x<br>y", html)


class TestInline(unittest.TestCase):
    def test_action_link_carries_encoded_params(self):
        link = ReportDoc.action("DOOR_A", "select", node="|ROOT|DOOR A&B")
        self.assertEqual(link.text, "DOOR_A")
        self.assertIn("action://select?node=%7CROOT%7CDOOR+A%26B", link.html)

    def test_action_params_may_be_named_text_or_verb(self):
        # The href is DOUBLE-quoted: Qt decodes the "&amp;" between params only
        # there -- single-quoted, the handler received "amp;verb" as a key.
        link = ReportDoc.action("copy", "copy", text="payload", verb="x")
        self.assertIn('href="action://copy?text=payload&amp;verb=x"', link.html)

    def test_action_url_is_the_log_links(self):
        from pythontk.core_utils.logging_mixin import LoggerExt

        url = LoggerExt._action_url("select", node="|a|b")
        self.assertIn(
            f'href="{url}"', ReportDoc.action("b", "select", node="|a|b").html
        )

    def test_file_link_encodes_href_but_shows_the_path(self):
        link = ReportDoc.file(r"C:\Dropbox (A+B)\a b.png")
        self.assertEqual(link.text, r"C:\Dropbox (A+B)\a b.png")
        href = link.html.split('href="', 1)[1].split('"', 1)[0]
        self.assertTrue(href.startswith("file:///C:/"))
        self.assertIn("%20", href)
        self.assertNotIn("\\", href)

    def test_file_link_keeps_a_unc_host(self):
        # file:///srv/share/... resolves to a root folder, not the share.
        link = ReportDoc.file(r"\\srv\share\a b.png")
        self.assertIn('href="file://srv/share/a%20b.png"', link.html)

    def test_a_link_without_a_tone_sets_no_colour(self):
        self.assertNotIn("color:", ReportDoc.link("x", "https://h/", tone=None).html)

    def test_span_tone_and_unknown_tone_as_css_colour(self):
        self.assertIn(ReportDoc.TONES["warn"], ReportDoc.span("x", "warn").html)
        self.assertIn("#123456", ReportDoc.span("x", "#123456").html)
        self.assertEqual(ReportDoc.span("a<b").html, "a&lt;b")

    def test_join_mixes_strings_and_inlines(self):
        joined = ReportDoc.join(["a<", ReportDoc.span("b", bold=True)], sep=" / ")
        self.assertEqual(joined.text, "a< / b")
        self.assertIn("a&lt; / <span", joined.html)


class TestBlocks(unittest.TestCase):
    def test_empty_document(self):
        doc = ReportDoc()
        self.assertFalse(doc)
        self.assertEqual(doc.to_html(), "")
        self.assertEqual(doc.to_text(), "")

    def test_empty_fields_and_items_add_nothing(self):
        self.assertFalse(ReportDoc().fields([]).items([]))

    def test_blocks_chain(self):
        doc = ReportDoc()
        self.assertIs(doc.heading("x").text("y").fields([("a", 1)]), doc)

    def test_table_alignment_and_row_padding(self):
        doc = ReportDoc().table(["A", "B", "C"], [[1], [1, 2, 3, 4]], align="lr")
        html = doc.to_html()
        self.assertIn("<th align='right'", html)
        self.assertEqual(html.count("<td"), 6)  # padded short row, truncated long row

    def test_only_the_last_column_wraps_by_default(self):
        # Names broke mid-word ("MAT_PANELS_deca|ls") when a prose column
        # squeezed the table; only the prose column may wrap.
        html = ReportDoc().table(["Name", "N", "Notes"], [["a", 1, "b"]]).to_html()
        self.assertEqual(html.count("white-space:nowrap"), 2)
        cells = html.split("<td")[1:]
        self.assertNotIn("nowrap", cells[-1])

    def test_wrap_names_the_wrapping_columns(self):
        doc = ReportDoc().table(["Path", "Materials"], [["p", "m"]], wrap=[0, 1])
        self.assertNotIn("nowrap", doc.to_html())

    def test_empty_table_says_none(self):
        doc = ReportDoc().table(["A"], [], title="Missing files")
        self.assertIn("(none)", doc.to_html())
        self.assertIn("(none)", doc.to_text())

    def test_text_table_is_aligned(self):
        text = ReportDoc().table(["Name", "Tris"], [["a", "1"], ["long_name", "10"]])
        header, rule, *rows = text.to_text().splitlines()
        self.assertTrue(header.startswith("Name"))
        # Every row puts its column separator where the header does.
        column = header.index("|")
        self.assertEqual(rule.index("+"), column)
        self.assertEqual([row.index("|") for row in rows], [column, column])

    def test_text_table_counts_angle_brackets_as_text(self):
        # A texture's "<UDIM>" token was measured as a tag: its row's separators
        # landed six columns right of everyone else's.
        rows = [["wood_<UDIM>.png", "4096"], ["plain_name_here.png", "2048"]]
        header, rule, *lines = (
            ReportDoc().table(["File", "px"], rows).to_text().splitlines()
        )
        self.assertIn("wood_<UDIM>.png", lines[0])
        self.assertEqual({line.index("|") for line in lines}, {header.index("|")})

    def test_text_table_breaks_a_cell_at_its_newlines(self):
        doc = ReportDoc().table(["A", "B"], [["one\ntwo", "x"]])
        header, rule, first, second = doc.to_text().splitlines()
        self.assertTrue(first.startswith("one") and second.startswith("two"))
        self.assertEqual(first.index("|"), header.index("|"))
        self.assertEqual(second.index("|"), header.index("|"))

    def test_text_headings_are_ruled_and_spaced(self):
        text = ReportDoc().heading("Report", 1).text("x").heading("Sec").to_text()
        self.assertEqual(text.splitlines(), ["Report", "======", "x", "", "Sec", "---"])

    def test_fields_text_pads_labels(self):
        text = ReportDoc().fields([("a", 1), ("long", 2)]).to_text()
        self.assertEqual(text.splitlines(), ["a    : 1", "long : 2"])

    def test_extend_appends_blocks(self):
        a = ReportDoc().text("a")
        a.extend(ReportDoc().text("b"))
        self.assertEqual(a.to_text(), "a\nb")


if __name__ == "__main__":
    unittest.main()
