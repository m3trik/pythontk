# !/usr/bin/python
# coding=utf-8
"""Tests for ``ManifestExport`` (``core_utils/engines/shots/manifest/manifest_export.py``):
the Shot Manifest's rows as a table -- records, the table's layouts, and CSV,
JSON and Markdown for a file or the clipboard.
"""

import json
import os
import sys
import unittest

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import pythontk as ptk
from pythontk.core_utils.engines.shots.manifest.manifest_export import ManifestExport
from pythontk.core_utils.engines.shots.manifest.manifest_model import (
    BuilderObject,
    BuilderStep,
)
from pythontk.core_utils.engines.shots.shot_model import ShotStore


def _step(step_id, *objects, section="A", title="Access", description="", name=""):
    """A ``BuilderStep`` listing ``(name, [behaviors])`` *objects*."""
    return BuilderStep(
        step_id,
        section,
        title,
        description,
        [BuilderObject(n, list(b)) for n, b in objects],
        audio=f"{step_id} voice",
        name=name,
    )


class _Rows(unittest.TestCase):
    """A01 built (renamed since, not yet rebuilt), A02 built and locked, A03
    not built, and a shot no step pairs with -- the four kinds of manifest
    row."""

    def setUp(self):
        store = ShotStore()
        self.a01 = _step(
            "A01",
            ("door", ["fade_in"]),
            ("handle", ["highlight", "fade_out"]),
            description="Door opens",
            name="Door_Open",
        )
        self.a02 = _step("A02", ("lid", []), description="Lid | lifts")
        self.a03 = _step("A03", description="Bolt\nturns")
        self.shot1 = store.define_shot("A01", 1, 40)
        self.shot2 = store.define_shot("A02", 41, 80, locked=True)
        self.orphan = store.define_shot(
            "A02_2", 81, 100, objects=["|grp|lamp"], description="Split off"
        )
        self.rows = [
            (self.a01, self.shot1),
            (self.a02, self.shot2),
            (None, self.orphan),
            (self.a03, None),
        ]

    def _records(self, **kwargs):
        kwargs.setdefault("ranges", {"A03": (101.0, 140.5)})
        return ManifestExport.records(self.rows, **kwargs)


class TestRecords(_Rows):
    def test_a_built_step_has_its_shots_name_frames_and_status(self):
        rec = self._records()[0]
        self.assertEqual(rec["step"], "A01")
        self.assertEqual(rec["name"], "Door_Open", "the name Local Edits gave it")
        self.assertEqual(rec["shot"], "A01", "the scene's, until Build renames it")
        self.assertEqual((rec["start"], rec["end"]), (1, 40))
        self.assertEqual(rec["status"], "built")
        self.assertEqual(
            rec["objects"],
            [
                {"name": "door", "behaviors": ["fade_in"], "kind": "scene"},
                {
                    "name": "handle",
                    "behaviors": ["highlight", "fade_out"],
                    "kind": "scene",
                },
            ],
        )
        self.assertEqual((rec["section"], rec["section_title"]), ("A", "Access"))
        self.assertEqual(rec["voice"], "A01 voice")

    def test_a_locked_shot_and_a_step_with_none_say_so(self):
        recs = self._records()
        self.assertEqual(recs[1]["status"], "locked")
        unbuilt = recs[3]
        self.assertEqual(unbuilt["status"], "missing_shot")
        self.assertEqual((unbuilt["name"], unbuilt["shot"]), ("A03", ""))
        self.assertEqual(
            (unbuilt["start"], unbuilt["end"]), (101.0, 140.5), "where it would go"
        )

    def test_a_shot_no_step_pairs_with_is_a_row_of_its_own(self):
        rec = self._records(object_name=lambda n: n.split("|")[-1])[2]
        self.assertIsNone(rec["step"])
        self.assertEqual((rec["name"], rec["shot"]), ("", "A02_2"))
        self.assertEqual(rec["status"], "not_in_doc")
        self.assertEqual(rec["description"], "Split off")
        self.assertEqual(
            rec["objects"], [{"name": "lamp", "behaviors": [], "kind": "scene"}]
        )

    def test_an_assessment_names_the_status(self):
        recs = self._records(statuses={"A01": "missing_object", "A03": "x"})
        self.assertEqual(recs[0]["status"], "missing_object")
        self.assertEqual(recs[2]["status"], "not_in_doc", "never assessed")


class TestTable(_Rows):
    def test_objects_a_row_each_the_steps_cells_on_the_first(self):
        table = ManifestExport.table(
            self._records()[:1], ["step", "objects", "behaviors", "start"]
        )
        self.assertEqual(
            table,
            [
                ["Step", "Objects", "Behaviors", "Start"],
                ["A01", "door", "fade_in", "1"],
                ["", "handle", "highlight, fade_out", ""],
            ],
        )

    def test_a_step_listing_no_objects_still_has_its_row(self):
        table = ManifestExport.table(self._records()[3:], ["step", "objects"])
        self.assertEqual(table, [["Step", "Objects"], ["A03", ""]])

    def test_one_cell_a_line_per_object(self):
        table = ManifestExport.table(
            self._records()[:1], ["objects", "behaviors"], objects="cell"
        )
        self.assertEqual(table[1], ["door\nhandle", "fade_in\nhighlight, fade_out"])

    def test_columns_come_in_table_order_and_unknown_ones_are_left_out(self):
        table = ManifestExport.table(
            self._records(), ["status", "nonsense", "step", "end", "section"]
        )
        self.assertEqual(table[0], ["Step", "Section", "End", "Status"])
        self.assertEqual(table[2], ["A02", "A: Access", "80", "locked"])
        self.assertEqual(table[3], ["", "", "100", "not in doc"])
        self.assertEqual(table[4], ["A03", "A: Access", "140.5", "missing shot"])

    def test_an_unknown_layout_is_refused(self):
        with self.assertRaises(ValueError):
            ManifestExport.table(self._records(), objects="columns")


class TestRender(_Rows):
    def test_csv_quotes_the_cells_that_need_it(self):
        text = ManifestExport.render(
            self._records()[2:], "csv", ["step", "shot", "description"]
        )
        self.assertEqual(
            text, 'Step,Shot,Description\n,A02_2,Split off\nA03,,"Bolt\nturns"\n'
        )

    def test_json_keeps_each_record_less_the_columns_left_out(self):
        out = json.loads(
            ManifestExport.render(
                self._records(), "json", ["step", "objects", "section"]
            )
        )
        self.assertEqual(
            out[0],
            {
                "step": "A01",
                "section": "A",
                "section_title": "Access",
                "objects": [
                    {"name": "door", "kind": "scene"},
                    {"name": "handle", "kind": "scene"},
                ],
            },
        )
        self.assertIsNone(out[2]["step"])

    def test_markdown_escapes_pipes_and_breaks_lines(self):
        text = ManifestExport.render(
            self._records()[1:], "markdown", ["step", "description"]
        )
        self.assertEqual(
            text.splitlines(),
            [
                "| Step | Description |",
                "| --- | --- |",
                "| A02 | Lid \\| lifts |",
                "|  | Split off |",
                "| A03 | Bolt<br>turns |",
            ],
        )

    def test_an_unknown_format_is_refused(self):
        with self.assertRaises(ValueError):
            ManifestExport.render(self._records(), "xlsx")

    def test_the_clipboard_forms_paste_into_a_spreadsheet(self):
        tsv, html = ManifestExport.clipboard([["a", "b\nc"], ['d"e', "f"]])
        self.assertEqual(tsv, 'a\t"b\nc"\n"d""e"\tf')
        self.assertIn("<tr><td>a</td><td>b<br>c</td></tr>", html)
        self.assertIn("d&quot;e", html)


class TestWrite(_Rows):
    def setUp(self):
        super().setUp()
        self.tmp = ptk.TempArtifacts("ptk_manifest_export", policy="scoped")
        self.addCleanup(self.tmp.cleanup)

    def _path(self, name):
        return os.path.join(self.tmp.dir_path(), name)

    def test_the_extension_names_the_format(self):
        recs = self._records()
        for name, fmt in (("m.csv", "csv"), ("m.JSON", "json"), ("m.md", "markdown")):
            path = ManifestExport.write(self._path(name), recs, ["step"])
            with open(path, encoding="utf-8-sig") as fh:
                self.assertEqual(
                    fh.read().replace("\r\n", "\n"),
                    ManifestExport.render(recs, fmt, ["step"]),
                )

    def test_a_csv_carries_the_byte_order_mark_excel_needs(self):
        path = ManifestExport.write(self._path("m.csv"), self._records(), ["step"])
        with open(path, "rb") as fh:
            self.assertTrue(fh.read().startswith(b"\xef\xbb\xbfStep"))

    def test_an_extension_naming_no_format_writes_nothing(self):
        path = self._path("m.txt")
        with self.assertRaises(ValueError):
            ManifestExport.write(path, self._records())
        self.assertFalse(os.path.exists(path))

    def test_the_format_of_a_path(self):
        self.assertEqual(ManifestExport.format_of("C:/a/b.Md"), "markdown")
        self.assertIsNone(ManifestExport.format_of("C:/a/b.csv.bak"))


if __name__ == "__main__":
    unittest.main()
