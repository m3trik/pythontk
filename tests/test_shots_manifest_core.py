# !/usr/bin/python
# coding=utf-8
"""Tests for the pure Shot Manifest core (``pythontk.core_utils.engines.shots.manifest``).

DCC-free.  Covers the parser + column map + behavior detection (``manifest_model``),
the column-mapping templates (``mapping``), the behavior template loading + keying
math (``behaviors``), and the range resolver (``range_resolver``).  The DCC-hooked
engine (``manifest_engine``) is exercised separately.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pythontk.core_utils.engines.shots.shot_model import SHOT_STATUS_LEVELS
from pythontk.core_utils.engines.shots.manifest.manifest_model import (
    ManifestModel,
    ColumnMap,
    BuilderStep,
    BuilderObject,
    StepStatus,
    ObjectStatus,
    ShotPairing,
    _ManifestModelInternal,
)
from pythontk.net_utils.remote_file import RemoteFile
from pythontk.core_utils.engines.shots.manifest import mapping as mapping_mod
from pythontk.core_utils.engines.shots.manifest.mapping._mapping import (
    _AUDIO_BUILDERS,
)
from pythontk.core_utils.engines.shots.manifest import behaviors as beh
from pythontk.core_utils.engines.shots.manifest import range_resolver as rr
from pythontk.core_utils.engines.shots.manifest.manifest_engine import ShotManifest
from pythontk.core_utils.engines.shots.effect_recipe import EffectRecipe
from pythontk.core_utils.engines.shots.shot_model import PureShotsHost, ShotStore
from pythontk.core_utils.engines.shots.shot_sequencer import ShotSequencer
from pythontk.core_utils.schema_spec import SchemaError


def _write_csv(text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.write(fd, text.encode("utf-8"))
    os.close(fd)
    return path


_SAMPLE_CSV = """SECTION A: AILERON RIGGING
Step,Step Contents,Asset Names,Voice Support,Priority
A01.),Aileron fades in then fades out,aileron_geo,Narrator intro,high
,wing detail continuation,wing_geo,,
A02.),Static fuselage step,fuselage,,low
SETUP,should be excluded,setup_obj,,
"""


class TestManifestRemoteSource(unittest.TestCase):
    """A URL is a CSV source: bytes come from RemoteFile, decoding is shared."""

    _URL = "https://docs.google.com/spreadsheets/d/1AbC/edit#gid=0"

    def test_url_parses_through_remote_file(self):
        raw = _SAMPLE_CSV.encode("utf-8")
        with patch.object(RemoteFile, "read_bytes", return_value=raw) as fetch:
            steps = ManifestModel.parse_csv(self._URL)
        fetch.assert_called_once_with(self._URL)
        self.assertEqual([s.step_id for s in steps], ["A01", "A02"])

    def test_local_path_never_touches_the_network(self):
        path = _write_csv(_SAMPLE_CSV)
        try:
            with patch.object(
                RemoteFile, "read_bytes", side_effect=AssertionError("fetched")
            ):
                steps = ManifestModel.parse_csv(path)
        finally:
            os.remove(path)
        self.assertEqual(len(steps), 2)

    def test_fetch_failure_propagates_as_remote_error(self):
        err = RemoteFile.Error("Can't fetch x")
        with patch.object(RemoteFile, "read_bytes", side_effect=err):
            with self.assertRaises(RemoteFile.Error):
                ManifestModel.parse_csv(self._URL)

    def test_decode_strips_bom_and_falls_back_to_cp1252(self):
        """The bytes-level decode is shared by both sources: BOM stripped
        first, so the cp1252 fallback still resolves the header."""
        body = "Step,Step Contents\nA01.),caf\u00e9\n".encode("cp1252")
        rows = _ManifestModelInternal._decode_csv_bytes(b"\xef\xbb\xbf" + body)
        self.assertEqual(rows[0], ["Step", "Step Contents"])
        self.assertEqual(rows[1][1], "caf\u00e9")


class TestManifestParse(unittest.TestCase):
    def setUp(self):
        self.path = _write_csv(_SAMPLE_CSV)

    def tearDown(self):
        os.remove(self.path)

    def test_step_count_excludes_setup(self):
        steps = ManifestModel.parse_csv(self.path)
        self.assertEqual([s.step_id for s in steps], ["A01", "A02"])

    def test_section_and_title(self):
        steps = ManifestModel.parse_csv(self.path)
        self.assertEqual(steps[0].section, "A")
        self.assertEqual(steps[0].section_title, "AILERON RIGGING")

    def test_continuation_row_merges_description_and_object(self):
        steps = ManifestModel.parse_csv(self.path)
        a01 = steps[0]
        self.assertIn("wing detail continuation", a01.description)
        self.assertEqual([o.name for o in a01.objects], ["aileron_geo", "wing_geo"])

    def test_lines_keep_each_row_apart_in_sheet_order(self):
        """``description`` joins the rows; ``lines`` keeps them, for a reader
        acting on one row at a time (a row the joined text cannot split back)."""
        path = _write_csv(
            """SECTION A: T
Step,Step Contents,Asset Names,Voice Support
A01.),Highlight the door,door_geo,vo
,,lid_geo,
,Selecting the door triggers Door open Animation,,
A02.),,,
"""
        )
        try:
            a01, a02 = ManifestModel.parse_csv(path)
        finally:
            os.remove(path)
        self.assertEqual(
            a01.lines,
            ["Highlight the door", "Selecting the door triggers Door open Animation"],
            "a blank cell is no line",
        )
        self.assertEqual(a01.description, " ".join(a01.lines))
        self.assertEqual(a02.lines, [])

    def test_rows_keep_each_rows_text_names_and_own_cells(self):
        """``rows``: each row of a step with its own text, its own Asset Names and the
        template's per-row cells (``row_pass``), for a reader binding a row's words
        to that row's names alone."""
        path = _write_csv(
            """Step,Step Contents,Asset Names,Action,Animation Clip Names
A01,Highlight the red door locks,RED_DOOR_LOCKS,,
,Click the top lock to open it,"LOCK_UPPER
KEYS",use,A01_1
,,,,
A02,Auto Advance,,,
"""
        )
        col_map = ColumnMap.from_dict({"step_pattern": [r"^([A-Z]+_?\d+(?:\.\d+)?)$"],
                                       "row_pass": {"action": ["Action"], "clip": ["Animation Clip Names"]}})
        try:
            a01, a02 = ManifestModel.parse_csv(path, col_map)
        finally:
            os.remove(path)
        self.assertEqual(a01.rows, [
            {"text": "Highlight the red door locks", "assets": ["RED_DOOR_LOCKS"], "action": "", "clip": ""},
            {"text": "Click the top lock to open it", "assets": ["LOCK_UPPER", "KEYS"], "action": "use", "clip": "A01_1"},
        ], "a row with nothing in it is no row")
        self.assertEqual(a02.rows, [{"text": "Auto Advance", "assets": [], "action": "", "clip": ""}])

    def test_the_headers_no_field_takes_are_named(self):
        col_map = ColumnMap.from_dict({"row_pass": {"clip": ["Animation Clip Names"]}})
        header = ["Step", "Step Contents", "Asset Names", "Narration", "Animation Clip Names", ""]
        self.assertEqual(col_map.unmapped(header), ["Narration"])

    def test_a_sheets_unmapped_columns_are_read_off_every_header_row(self):
        path = _write_csv(
            "SECTION A: Open\nStep,Step Contents,Asset Names,Remarks\nA01,Highlight the lid,LID,Hello\n"
            "SECTION B: Close\nStep,Step Contents,Asset Names,Comments,Remarks,Narration\nB01,Highlight the lid,LID,,,\n"
        )
        try:
            self.assertEqual(mapping_mod.Mapping.unmapped_columns(path, mapping_mod.Mapping.load_mapping("linear")),
                             ["Remarks", "Comments"], "Narration is the Linear template's voice column")
        finally:
            os.remove(path)

    def test_a_data_row_holding_a_step_id_alias_is_no_header(self):
        """A data cell that happens to read as a step-ID alias ("Number": a
        placard's number pad) is no header row: only a row the parser would
        read as one is."""
        path = _write_csv(
            "Step,Step Contents,Asset Names\nA01,Highlight the lid,LID\n,Number,PAD\n"
        )
        try:
            self.assertEqual(
                mapping_mod.Mapping.unmapped_columns(path, mapping_mod.Mapping.load_mapping("interactive")), []
            )
        finally:
            os.remove(path)

    def test_the_format_templates_ship_a_starter_sheet_their_template_reads(self):
        for name, first in (("linear", "A01"), ("interactive", "IN_01")):
            with self.subTest(name=name):
                sheet = mapping_mod.Mapping.starter_sheet(name)
                steps = mapping_mod.Mapping.resolve(str(sheet), mapping_mod.Mapping.load_mapping(name))
                self.assertEqual(steps[0].step_id, first)
                self.assertEqual(mapping_mod.Mapping.unmapped_columns(str(sheet), mapping_mod.Mapping.load_mapping(name)), [])
        self.assertIsNone(mapping_mod.Mapping.starter_sheet("default"), "the tolerant template reads any sheet")

    def test_a_sheet_in_cells_says_which_template_reads_it(self):
        """A panel read every sheet with its user's last template: the format's own starter, under
        the tolerant default, stopped at "Column 'description' not found". A sheet written in cells
        names its template by its header and its words. Added: 2026-11-25"""
        for name in ("interactive", "linear"):
            self.assertEqual(mapping_mod.Mapping.template_for(str(mapping_mod.Mapping.starter_sheet(name))), name)
        # A word no template has (a typo) still reads as the template the rest of the sheet is written for.
        typo = _write_csv("Step,Step Name,Voice,Action,Target,With,Clip,Label,To,Settings\n"
                          "A01,Open,Open it.,,,,,,,\n,,,hilight,DOOR,,,,,\n,,,use,DOOR,,A01,,,\n")
        sentences = _write_csv("Step,Step Contents,Asset Names\nA01,Highlight the door,DOOR\n")
        own = _write_csv("Step,Step Name,Voice,Action,Target,Settings,Reviewer\nA01,Open,Open it.,,,,\n,,,highlight,DOOR,,\n")
        try:
            self.assertEqual(mapping_mod.Mapping.template_for(typo), "interactive")
            self.assertIsNone(mapping_mod.Mapping.template_for(sentences), "a sheet of sentences names none")
            self.assertEqual(mapping_mod.Mapping.template_for(own), "linear", "the smallest template that reads its cells")
        finally:
            for path in (typo, sentences, own):
                os.remove(path)

    def test_a_new_sheet_is_its_templates_starter_and_never_replaces_a_file(self):
        """The starter was reached by a deep import alone: no operation started a sheet. Added: 2026-11-25"""
        import tempfile

        import pythontk as ptk

        folder = tempfile.mkdtemp()
        out = os.path.join(folder, "new", "door_service.csv")
        planned = ptk.OpContext(dry_run=True)
        mapping_mod.Mapping.new_sheet(out, ctx=planned)
        self.assertFalse(os.path.exists(out), "a dry run plans it")
        self.assertEqual([c.kind for c in planned.changes], ["write"])
        made = mapping_mod.Mapping.new_sheet(out, "interactive")
        with open(out, encoding="utf-8") as written, open(mapping_mod.Mapping.starter_sheet("interactive"), encoding="utf-8") as starter:
            self.assertEqual(written.read(), starter.read())
        self.assertEqual((made["sheet"], made["template"], made["columns"][:3]), (out, "interactive", ["Step", "Step Name", "Voice"]))
        with self.assertRaises(FileExistsError):
            mapping_mod.Mapping.new_sheet(out, "linear")
        with self.assertRaises(ValueError) as refused:
            mapping_mod.Mapping.new_sheet(os.path.join(folder, "other.csv"), "default")
        self.assertIn("interactive, linear", str(refused.exception))

    def test_pass_through_is_a_copy_of_the_steps_pass_through_columns(self):
        step = ManifestModel.parse_csv(self.path)[0]
        step._pass_through = {"hint": "the lid"}
        seen = step.pass_through
        seen["hint"] = "changed"
        self.assertEqual(step.pass_through, {"hint": "the lid"})

    def test_behaviors_detected_from_description(self):
        steps = ManifestModel.parse_csv(self.path)
        a01 = steps[0]
        self.assertEqual(a01.objects[0].behaviors, ["show", "hide"])

    def test_audio_column_captured(self):
        steps = ManifestModel.parse_csv(self.path)
        self.assertEqual(steps[0].audio, "Narrator intro")

    def test_continuation_row_own_description_overrides_to_no_behaviors(self):
        """A row describing itself as static must not inherit the step's fades.

        Gating the override on *detected behaviors* rather than on the
        description's presence made "no behaviors" unsayable: a static prop
        listed beside a fading one silently picked up the fades and got keyed.
        """
        path = _write_csv(
            """SECTION A: T
Step,Step Contents,Asset Names,Voice Support
A01.),Aileron fades in then fades out,aileron_geo,vo
,static support bracket does not move,bracket_geo,
"""
        )
        try:
            objs = ManifestModel.parse_csv(path)[0].objects
            self.assertEqual([o.name for o in objs], ["aileron_geo", "bracket_geo"])
            self.assertEqual(objs[0].behaviors, ["show", "hide"])
            self.assertEqual(objs[1].behaviors, [])
        finally:
            os.remove(path)

    def test_continuation_row_without_description_inherits_step_behaviors(self):
        path = _write_csv(
            """SECTION A: T
Step,Step Contents,Asset Names,Voice Support
A01.),The table fades in,table_geo,vo
,,unit_geo,
"""
        )
        try:
            objs = ManifestModel.parse_csv(path)[0].objects
            self.assertEqual(objs[1].name, "unit_geo")
            self.assertEqual(objs[1].behaviors, ["show"])
        finally:
            os.remove(path)

    def test_continuation_inherit_ignores_sibling_row_text(self):
        """Inheritance reads the parent STEP, not the accumulated description.

        Continuation descriptions are merged into ``current_step.description``
        before the next row is read, so inheriting from it let one asset's text
        leak onto every later blank row -- making the result order-dependent.
        """
        path = _write_csv(
            """SECTION A: T
Step,Step Contents,Asset Names,Voice Support
A01.),Static setup with no motion,obj_a,vo
,the panel fades in,obj_b,
,,obj_c,
"""
        )
        try:
            objs = ManifestModel.parse_csv(path)[0].objects
            self.assertEqual([o.name for o in objs], ["obj_a", "obj_b", "obj_c"])
            self.assertEqual(objs[0].behaviors, [])
            self.assertEqual(objs[1].behaviors, ["show"])
            self.assertEqual(objs[2].behaviors, [])
        finally:
            os.remove(path)

    def test_an_object_on_two_rows_is_one_object(self):
        """Bug: every asset-cell row became an object of its own, so one
        object listed on two rows of a step was two -- Build placed their
        behaviors together, Assess and Apply each row's alone, and an Apply
        deleted the other row's keys.  Behaviors keep row order, repeats too.
        Fixed: 2026-10-04
        """
        path = _write_csv(
            "Step,Step Contents,Asset Names\n"
            "A01.),Door fades in then fades out,door\n"
            ",Door fades in again,door\n"
            ",,lid\n"
        )
        try:
            (step,) = ManifestModel.parse_csv(path)
        finally:
            os.remove(path)
        self.assertEqual(
            [(o.name, o.behaviors) for o in step.objects],
            [
                ("door", ["show", "hide", "show"]),
                ("lid", ["show", "hide"]),
            ],
        )

    def test_metadata_pass_first_row_wins(self):
        cm = ColumnMap(metadata_pass={"priority": ("Priority",)})
        steps = ManifestModel.parse_csv(self.path, columns=cm)
        self.assertEqual(steps[0]._pass_through.get("priority"), "high")
        self.assertEqual(steps[1]._pass_through.get("priority"), "low")

    def test_missing_header_yields_zero_steps(self):
        path = _write_csv("just,some,data\nwith,no,header\n")
        try:
            self.assertEqual(ManifestModel.parse_csv(path), [])
        finally:
            os.remove(path)


class TestColumnMap(unittest.TestCase):
    def test_defaults(self):
        cm = ColumnMap()
        self.assertEqual(cm.step_id, ("Step",))
        self.assertIn("Asset Names", cm.assets)

    def test_dict_round_trip_preserves_tuples(self):
        cm = ColumnMap(metadata_pass={"priority": ("Priority", "Prio")})
        back = ColumnMap.from_dict(cm.to_dict())
        self.assertEqual(back.step_id, cm.step_id)
        self.assertEqual(back.metadata_pass["priority"], ("Priority", "Prio"))

    def test_from_dict_ignores_unknown_keys(self):
        back = ColumnMap.from_dict({"step_id": ["S"], "bogus": 1})
        self.assertEqual(back.step_id, ("S",))


class TestDetectBehaviors(unittest.TestCase):
    def test_show(self):
        self.assertEqual(
            ManifestModel.detect_behaviors("the panel fades in slowly"), ["show"]
        )

    def test_hide(self):
        self.assertEqual(
            ManifestModel.detect_behaviors("then it fades out"), ["hide"]
        )

    def test_both_independent(self):
        self.assertEqual(
            ManifestModel.detect_behaviors("fades in then fades out"),
            ["show", "hide"],
        )

    def test_none(self):
        self.assertEqual(ManifestModel.detect_behaviors("a static object"), [])


class TestBehaviorsFromText(unittest.TestCase):
    """Behavior templates declare their own phrases (``detect``): the parser
    reads prose and a behaviors column through them, so a new template
    brings its words with it instead of an edit to a regex list."""

    def test_phrases_live_in_the_templates(self):
        detect = beh.Behaviors.detect
        self.assertEqual(detect("Keys fade away from hand"), ["hide"])
        self.assertEqual(detect("Highlight Red Door Locks"), ["highlight"])
        self.assertEqual(detect("Destination circle fades in"), ["show"])
        self.assertEqual(detect("User is teleported"), [])

    def test_a_column_cell_takes_names_or_phrases_and_keeps_unknowns(self):
        cell = beh.Behaviors.from_cell
        self.assertEqual(cell("Fade In, Fade Out"), ["show", "hide"])
        self.assertEqual(cell("fades away\nhighlight"), ["hide", "highlight"])
        # Unknown items stay (spelled as a name) so Assess can flag them.
        self.assertEqual(cell("Wobble; show"), ["wobble", "show"])
        self.assertEqual(cell("  "), [])

    def test_a_family_reads_as_one_behavior_with_a_suffix(self):
        """User 2026-11-23: consolidate the manifest's behaviors, a suffix
        saying how one is set. Added: 2026-11-24 -- a template's ``family``
        and ``variant``: show and hide read as one Fade, each keeping
        its own name in saved data and deliveries."""
        B = beh.Behaviors
        self.assertEqual(B.label("show"), "Visibility · show")
        self.assertEqual(B.label("hide"), "Visibility · hide")
        self.assertEqual(B.label("set_clip"), "Set Clip", "no family: its own name")
        self.assertEqual(B.label("wobble"), "Wobble", "no template: its own name")
        self.assertEqual(B.family_of("hide"), "visibility")
        self.assertEqual(B.family_of("highlight"), "highlight")
        self.assertEqual(
            B.families(["highlight", "hide", "wobble", "show"]),
            {"highlight": ["highlight"], "visibility": ["hide", "show"], "wobble": ["wobble"]},
        )
        self.assertEqual(B.from_cell("Visibility · hide, Show"), ["hide", "show"], "its label reads back")

    def test_shipped_templates_validate_with_their_phrases(self):
        for name in beh.Behaviors.list_behaviors():
            res = beh.BehaviorSpec.validate(beh.Behaviors.load_behavior(name))
            self.assertTrue(res.ok, f"{name}: {res.errors}")


class TestBehaviorSource(unittest.TestCase):
    """Where a row's behaviors come from: ``ColumnMap.behavior_source`` --
    ``description`` (the default, as before), ``column``, or
    ``column_else_description``."""

    CSV = (
        "Step,Step Contents,Asset Names,Behaviors\n"
        "A01.),Door fades in,door,Highlight\n"
        ",Lid fades away,lid,\n"
    )

    def _objects(self, source):
        path = _write_csv(self.CSV)
        try:
            (step,) = ManifestModel.parse_csv(
                path,
                columns=ColumnMap(behaviors=("Behaviors",), behavior_source=source),
            )
        finally:
            os.remove(path)
        return [(o.name, o.behaviors) for o in step.objects]

    def test_description(self):
        self.assertEqual(
            self._objects("description"),
            [("door", ["show"]), ("lid", ["hide"])],
        )

    def test_column(self):
        # A row with an empty cell inherits its step's behaviors -- the
        # parser's rule for a row that says nothing of its own.
        self.assertEqual(
            self._objects("column"), [("door", ["highlight"]), ("lid", ["highlight"])]
        )

    def test_column_else_description(self):
        self.assertEqual(
            self._objects("column_else_description"),
            [("door", ["highlight"]), ("lid", ["hide"])],
        )

    def test_a_bad_source_is_a_schema_error(self):
        res = mapping_mod.MappingSpec.validate(
            {"columns": {"behavior_source": "telepathy"}}
        )
        self.assertFalse(res.ok)


class TestProseObjects(unittest.TestCase):
    """A sheet whose asset cells are empty names its objects in prose
    ("Highlight Red Door", "Destination circle fades in"): a behavior
    template's ``detect`` phrase with an ``object`` group reads the name, and
    ``ColumnMap.object_source = column_else_description`` lists it with that
    behavior -- named by the template's case and legal-name rule."""

    CSV = (
        "INTRO,,,\n"
        "Step,Voice Support,Step Contents,Asset Names\n"
        "IN_01,Hi,Destination circle fades in at the travel spot,\n"
        ",,Highlight the 2 phillips screws,\n"
        ",,Destination circle fades away,\n"
        ",,Selecting the highlighted area triggers the latch animation,\n"
        "IN_02,,Highlight Red Door on the machine,door_geo\n"
        "IN_03,,Auto Advance,\n"
        "IN_04,,When the user has finished every single step the keys fade away,\n"
        "IN_05,,Highlight the lever,N/A\n"
    )
    COLUMNS = dict(
        step_pattern=(r"^([A-Z]+_?\d+)$",),
        section_pattern=("^(INTRO)$",),
        exclude_steps=(),
    )

    def _steps(self, **columns):
        path = _write_csv(self.CSV)
        try:
            steps = ManifestModel.parse_csv(
                path, columns=ColumnMap(**dict(self.COLUMNS, **columns))
            )
        finally:
            os.remove(path)
        return {
            s.step_id: [(o.name, o.behaviors, o.origin) for o in s.objects]
            for s in steps
        }

    def test_subjects_reads_the_object_each_phrase_names(self):
        self.assertEqual(
            beh.Behaviors.subjects("Highlight the 2 phillips screws on the table."),
            [("phillips screws", "highlight")],
        )
        self.assertEqual(
            beh.Behaviors.subjects("Both screws appear highlighted next to the holes"),
            [("screws", "highlight")],
        )
        self.assertEqual(
            beh.Behaviors.subjects("Pip screen fades away"),
            [("Pip screen", "hide")],
        )
        # Detected, but no object named: nothing to list.
        self.assertEqual(beh.Behaviors.subjects("Selecting the highlighted area"), [])
        self.assertEqual(
            beh.Behaviors.detect("Selecting the highlighted area"), ["highlight"]
        )

    def test_an_asset_less_step_lists_what_its_prose_names(self):
        steps = self._steps(object_source="column_else_description")
        self.assertEqual(
            steps["IN_01"],
            [
                ("Destination_circle", ["show", "hide"], "description"),
                ("phillips_screws", ["highlight"], "description"),
            ],
        )
        # The asset column wins: its step never takes prose names.
        self.assertEqual(steps["IN_02"], [("door_geo", ["highlight"], "column")])
        self.assertEqual(steps["IN_03"], [])
        # A capture that runs to a sentence is not a name.
        self.assertEqual(steps["IN_04"], [])
        # "N/A" is the sheet saying the step has none -- not a blank to fill.
        self.assertEqual(steps["IN_05"], [])

    def test_prose_is_not_read_unless_the_template_asks(self):
        self.assertEqual(self._steps()["IN_01"], [])

    def test_case_and_name_rule_shape_the_names(self):
        steps = self._steps(
            object_source="column_else_description",
            object_case="upper",
            object_name_rule="keep",
        )
        self.assertEqual(
            [name for name, _, _ in steps["IN_01"]],
            ["DESTINATION CIRCLE", "PHILLIPS SCREWS"],
        )

    def test_one_object_per_name_whatever_the_casing(self):
        path = _write_csv(
            "Step,Step Contents,Asset Names\n"
            "A01.),Highlight Socket Wrench,\n"
            ",the socket wrench fades away,\n"
        )
        try:
            (step,) = ManifestModel.parse_csv(
                path, columns=ColumnMap(object_source="column_else_description")
            )
        finally:
            os.remove(path)
        self.assertEqual(
            [(o.name, o.behaviors) for o in step.objects],
            [("Socket_Wrench", ["highlight", "hide"])],
        )

    def test_the_default_template_reads_prose_and_offers_strutils_cases(self):
        from pythontk import StrUtils

        tpl = mapping_mod.Mapping.load_mapping("default")
        specs = {o["key"]: o for o in mapping_mod.Mapping.option_specs(tpl)}
        self.assertEqual(
            [v for _, v, _ in specs["object_case"]["choices"]],
            ["keep", *StrUtils.CASES],
        )
        self.assertEqual(
            [v for _, v, _ in specs["object_name_rule"]["choices"]],
            ["keep", *StrUtils.NAME_RULES],
        )
        cols = mapping_mod.Mapping.apply_options(tpl, {"object_case": "lower"})[
            "columns"
        ]
        self.assertEqual(cols["object_case"], "lower")
        self.assertEqual(cols["object_source"], "column_else_description")
        only = mapping_mod.Mapping.apply_options(tpl, {"objects": "column"})["columns"]
        self.assertEqual(only["object_source"], "column")
        # An unknown saved value falls back to the default.
        cols = mapping_mod.Mapping.apply_options(tpl, {"object_case": "shout"})[
            "columns"
        ]
        self.assertEqual(cols["object_case"], "keep")

    def test_a_field_choice_must_name_a_field_with_choices(self):
        errors = " | ".join(
            mapping_mod.MappingSpec.validate(
                {
                    "options": {
                        "a": {"kind": "choice", "field": "columns.description"},
                        "b": {
                            "kind": "choice",
                            "field": "columns.object_case",
                            "default": "shout",
                        },
                        "c": {
                            "kind": "choice",
                            "field": "columns.object_case",
                            "choices": {"loud": {"label": "Loud"}},
                        },
                    }
                }
            ).errors
        )
        self.assertIn("'a': field 'columns.description' is not a schema field", errors)
        self.assertIn("'b': default 'shout' is not a choice", errors)
        self.assertIn("'loud' is not one of", errors)

    def test_a_headerless_sheet_names_the_file_it_read(self):
        path = _write_csv("no,header,here\n1,2,3\n")
        try:
            with self.assertLogs(level="WARNING") as logs:
                ManifestModel.parse_csv(path)
        finally:
            os.remove(path)
        self.assertIn(os.path.basename(path), " ".join(logs.output))


class TestTheMoveKeepsTheSavedNames(unittest.TestCase):
    """Names a user's files and scenes already hold, pinned before the Shot
    Manifest engine moves to tentacle (2026-10-27): its template folders stay
    under ``shots/`` (never the new package's name) and the markers a build
    writes keep their owner."""

    def test_the_mapping_templates_stay_under_shots(self):
        self.assertEqual(
            mapping_mod.Mapping.templates().user_dir.parts[-2:],
            ("shots", "manifest_mappings"),
        )

    def test_the_behavior_templates_stay_under_shots(self):
        self.assertEqual(
            beh.Behaviors.templates().user_dir.parts[-2:],
            ("shots", "manifest_behaviors"),
        )

    def test_a_builds_markers_keep_their_owner(self):
        self.assertEqual(ShotManifest.MARKER_OWNER, "shot_manifest")


class TestTemplateSettings(unittest.TestCase):
    """Settings a mapping template carries."""

    def test_match_is_a_template_setting(self):
        Mapping = mapping_mod.Mapping
        data = Mapping.load_mapping("default")
        self.assertEqual(Mapping.apply_options(data).get("match"), "name")
        eff = Mapping.apply_options(data, {"match": "name_then_order"})
        self.assertEqual(eff["match"], "name_then_order")
        self.assertFalse(mapping_mod.MappingSpec.validate({"match": "vibes"}).ok)


class TestMapping(unittest.TestCase):
    def test_audio_builder_registry_matches_descriptor_registry(self):
        # OCP guard: the resolver builders and the validate/docs descriptors
        # must enumerate the same methods.
        self.assertEqual(set(_AUDIO_BUILDERS), set(mapping_mod.AUDIO_METHODS))

    def test_discover_builtins(self):
        names = mapping_mod.Mapping.discover()
        self.assertIn("default", names)
        # One template per production scenario (a sheet's layouts stay options of each, not
        # sibling files): any sheet, linear animation, an interactive task.
        self.assertEqual(names, ["default", "interactive", "linear"])

    def test_load_default(self):
        spec = mapping_mod.Mapping.load_mapping("default")
        self.assertIsNotNone(spec)

    def test_resolve_csv_to_steps(self):
        path = _write_csv(_SAMPLE_CSV)
        try:
            steps = mapping_mod.Mapping.resolve(path, name="default")
            self.assertTrue(all(isinstance(s, BuilderStep) for s in steps))
            self.assertEqual([s.step_id for s in steps], ["A01", "A02"])
        finally:
            os.remove(path)


class TestMappingSpec(unittest.TestCase):
    """The mapping-file schema: self-validating skeleton, shipped files clean,
    bad audio methods rejected, unknown keys tolerated."""

    def test_skeleton_is_valid_against_its_own_schema(self):
        spec = mapping_mod.MappingSpec
        self.assertTrue(spec.validate(spec.skeleton()).ok)

    def test_shipped_mappings_validate_clean(self):
        Mapping = mapping_mod.Mapping
        for name in Mapping.discover():
            path = mapping_mod.DEFAULT_DIR / f"{name}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            res = mapping_mod.MappingSpec.validate(data)
            self.assertTrue(res.ok, f"{name}: errors={res.errors}")
            # Every value of every option must yield a valid template too.
            for spec in Mapping.option_specs(data):
                values = (
                    [v for _label, v, _tip in spec["choices"]]
                    if spec["kind"] == "choice"
                    else [False, True]
                )
                for value in values:
                    eff = Mapping.apply_options(data, {spec["key"]: value})
                    res = mapping_mod.MappingSpec.validate(eff)
                    self.assertTrue(
                        res.ok, f"{name} {spec['key']}={value!r}: {res.errors}"
                    )

    def test_color_by_offers_exactly_the_tag_rules(self):
        """The field's choices ARE ``ManifestTags.RULES``, so the template
        option (a ``field`` choice) can never offer a rule the panel lacks;
        a value outside them is a schema error."""
        from pythontk.core_utils.engines.shots.manifest.manifest_tags import (
            ManifestTags,
        )

        Mapping = mapping_mod.Mapping
        self.assertEqual(
            mapping_mod.MappingSpec.field_choices("color_by"),
            tuple(ManifestTags.RULES),
        )
        self.assertFalse(mapping_mod.MappingSpec.validate({"color_by": "hue"}).ok)
        default = Mapping.load_mapping("default")
        spec = {o["key"]: o for o in Mapping.option_specs(default)}["color_by"]
        self.assertEqual(spec["default"], "none")
        self.assertEqual(
            [v for _label, v, _tip in spec["choices"]], list(ManifestTags.RULES)
        )
        effective = Mapping.apply_options(default, {"color_by": "section"})
        self.assertEqual(effective["color_by"], "section")

    def test_unknown_audio_method_is_error(self):
        res = mapping_mod.MappingSpec.validate({"audio_resolve": {"method": "bogus"}})
        self.assertFalse(res.ok)
        self.assertTrue(any("audio_resolve" in e for e in res.errors))

    def test_regex_method_requires_pattern(self):
        res = mapping_mod.MappingSpec.validate({"audio_resolve": {"method": "regex"}})
        self.assertFalse(res.ok)
        self.assertTrue(any("pattern" in e for e in res.errors))

    def test_unknown_top_level_key_is_warning_not_error(self):
        res = mapping_mod.MappingSpec.validate({"bogus": 1})
        self.assertTrue(res.ok)  # tolerated -- does not reject the file
        self.assertTrue(res.warnings)

    def test_load_mapping_raises_schema_error_on_bad_method(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            p.write_text(
                json.dumps({"audio_resolve": {"method": "nope"}}), encoding="utf-8"
            )
            with self.assertRaises(SchemaError):
                mapping_mod.Mapping.load_mapping(str(p))


class TestMappingResolveRoundTrip(unittest.TestCase):
    """A CSV parsed through the ``default`` mapping yields exactly what the
    underlying ColumnMap path does."""

    CSV = (
        "SECTION A: INTRO\n"
        "Step,Step Contents,Asset Names,Voice Support\n"
        "A01.),Aileron fades in,wing_L,Welcome to the task\n"
        "A02.),Rudder appears,rudder,N/A\n"
    )

    def _csv(self, d):
        p = Path(d) / "m.csv"
        p.write_text(self.CSV, encoding="utf-8")
        return str(p)

    def test_resolve_matches_direct_columnmap_path(self):
        with tempfile.TemporaryDirectory() as d:
            csv = self._csv(d)
            data = mapping_mod.Mapping.load_mapping("default")
            via_resolve = mapping_mod.Mapping.resolve(csv, mapping=data)
            via_direct = ManifestModel.parse_csv(
                csv, columns=ColumnMap.from_dict(data["columns"])
            )
            self.assertTrue(via_resolve)  # non-empty
            self.assertEqual(
                [s.step_id for s in via_resolve], [s.step_id for s in via_direct]
            )
            self.assertEqual(
                [s.description for s in via_resolve],
                [s.description for s in via_direct],
            )

    def test_derived_audio_is_an_option_of_the_default(self):
        """What the retired ``speedrun`` template did: a clip per voiced step."""
        with tempfile.TemporaryDirectory() as d:
            csv = self._csv(d)
            steps = mapping_mod.Mapping.resolve(
                csv, name="default", options={"audio": "derive"}
            )
        audio = [(o.name, o.behaviors) for o in steps[0].objects if o.kind == "audio"]
        self.assertEqual(audio, [("A01_WelcomeToThe", ["set_clip"])])
        self.assertFalse([o for o in steps[1].objects if o.kind == "audio"])  # N/A


class TestAudioChain(unittest.TestCase):
    """A step's audio found by a list of methods, the first that finds a file winning: the sheet's
    Audio Files cell, a recording by prefix, then narration looked up by its words -- never spoken
    while a sheet is read. Relative folders resolve against the base the caller names."""

    CSV = (
        "SECTION A: INTRO\n"
        "Step,Step Contents,Asset Names,Voice Support,Audio Files\n"
        "A01.),Aileron fades in,wing_L,Welcome to the task,take_a01.wav\n"
        "A02.),Rudder appears,rudder,Open the panel,\n"
        "A03.),Flap moves,flap,Check the flap,\n"
        "A04.),Gear drops,gear,Lower the gear,\n"
    )

    def _sheet(self, d):
        sheets = Path(d) / "exports"
        sheets.mkdir()
        p = sheets / "Sheet (3).csv"
        p.write_text(self.CSV, encoding="utf-8")
        return str(p)

    def _audio(self, steps):
        return {s.step_id: [(o.name, Path(o.source_path).name if o.source_path else "") for o in s.objects if o.kind == "audio"]
                for s in steps}

    def test_a_relative_folder_resolves_against_the_base_not_the_sheet(self):
        with tempfile.TemporaryDirectory() as d:
            sheet = self._sheet(d)
            project = Path(d) / "project"
            (project / "rec").mkdir(parents=True)
            (project / "rec" / "A02_open.wav").write_bytes(b"RIFF")
            mapping = mapping_mod.Mapping.load_mapping("default")
            mapping["audio_resolve"] = {"method": "prefix", "directory": "rec"}
            steps = mapping_mod.Mapping.resolve(sheet, mapping=mapping, base=str(project))
        self.assertEqual(self._audio(steps)["A02"], [("A02_open", "A02_open.wav")])

    def test_a_list_takes_the_first_method_that_finds_a_file(self):
        from pythontk.audio_utils.narrator import Narrator

        with tempfile.TemporaryDirectory() as d:
            sheet = self._sheet(d)
            project = Path(d) / "project"
            for sub in ("rec", "narration"):
                (project / sub).mkdir(parents=True)
            (project / "rec" / "take_a01.wav").write_bytes(b"RIFF")
            (project / "rec" / "A02_open.wav").write_bytes(b"RIFF")
            narrator = Narrator(pronounce={"flap": "flap"})
            (project / "narration" / (narrator.key("Check the flap") + ".wav")).write_bytes(b"RIFF")
            mapping = mapping_mod.Mapping.load_mapping("default")
            mapping["audio_resolve"] = [
                {"method": "cell", "directory": "rec"},
                {"method": "prefix", "directory": "rec"},
                {"method": "narrate", "directory": "narration", "pronounce": {"flap": "flap"}},
            ]
            from pythontk.core_utils.engines.shots.manifest.mapping._spec import _MappingSpecInternal

            self.assertEqual(_MappingSpecInternal.validate_audio_resolve(mapping["audio_resolve"]), [], "a list validates")
            steps = mapping_mod.Mapping.resolve(sheet, mapping=mapping, base=str(project))
            narrated = list((project / "narration").iterdir())
        audio = self._audio(steps)
        self.assertEqual(audio["A01"], [("take_a01", "take_a01.wav")], "the sheet's own cell first")
        self.assertEqual(audio["A02"], [("A02_open", "A02_open.wav")], "then a recording by prefix")
        self.assertEqual(audio["A03"], [("A03_CheckTheFlap", narrator.key("Check the flap") + ".wav")], "then narration")
        self.assertEqual(audio["A04"], [("A04_LowerTheGear", "")],
                         "nothing found: the first placeholder clip, so the step still gets one")
        self.assertEqual(len(narrated), 1, "reading a sheet never speaks a line")

    def test_a_cell_naming_a_missing_file_is_a_clip_with_no_source(self):
        with tempfile.TemporaryDirectory() as d:
            sheet = self._sheet(d)
            mapping = mapping_mod.Mapping.load_mapping("default")
            mapping["audio_resolve"] = {"method": "cell", "directory": "rec"}
            steps = mapping_mod.Mapping.resolve(sheet, mapping=mapping, base=d)
        self.assertEqual(self._audio(steps)["A01"], [("take_a01", "")])
        self.assertEqual(self._audio(steps)["A02"], [], "an empty cell names nothing")


class TestAudioMethodSettings(unittest.TestCase):
    """A reader outside the mapping (the narrator) asks it for an audio
    method's settings, instead of reaching into its internals.
    Added: 2026-11-22"""

    def test_a_methods_settings_come_with_their_paths_under_the_base(self):
        mapping = {"audio_resolve": [{"method": "files"}, {"method": "narrate", "directory": "narration"}]}
        Mapping = mapping_mod.Mapping
        found = Mapping.audio_method(mapping, "narrate", base=os.path.join("C:" + os.sep, "task"))
        self.assertEqual(found["directory"], os.path.join("C:" + os.sep, "task", "narration"))
        self.assertEqual(Mapping.audio_method({"audio_resolve": {"method": "narrate"}}, "narrate")["method"], "narrate")
        self.assertIsNone(Mapping.audio_method(mapping, "speak"))
        self.assertIsNone(Mapping.audio_method({}, "narrate"))


class TestSequenceDocMapping(unittest.TestCase):
    """The default template's ``numbered`` step IDs read an interactive
    task's sequence doc: ``IN_01`` / ``A01`` / ``B03.5`` step IDs, an
    ``INTRO`` banner, a header repeated per section, multi-line voice cells,
    and a quiz block after the last step.  Its ``classic`` IDs (``A01.)``)
    parse none of them."""

    CSV = (
        "INTRO,,,,,,,\n"
        "Step,Step Name,Hint,Text On Placard,Voice Support,Step Contents,"
        "Asset Names,Frame Ranges\n"
        "IN_01,Welcome,,,Welcome to the task.,User starts in the room.,,\n"
        ",,,,,Auto Advance,,\n"
        "IN_02,Teleport 01,,,\"Open the machine.\n\nSelect 'Next'.\","
        "Destination circle fades in,,\n"
        "SECTION B: Solenoid Lock Removal,,,,,,,\n"
        "Step,Step Name,Hint,Text On Placard,Voice Support,Step Contents,"
        "Asset Names,Frame Ranges\n"
        "B03,Door Open?,,,Did the door open?,YES and NO buttons appear,,\n"
        "B03.5,Incorrect Answer 01,,,That's Incorrect.,Try Again button appears,,\n"
        ",,,,,Solenoid fades away,solenoid_geo,(XX-XX)\n"
        ",,,,,,,\n"
        "QUIZ QUESTIONS,,,,,,,\n"
        "Why are the baffles closed before removing the solenoid?,,,,,,,\n"
        "To gain access to the solenoid,,,,,,,\n"
    )

    def _resolve(self, step_ids="numbered"):
        path = _write_csv(self.CSV)
        try:
            return mapping_mod.Mapping.resolve(
                path, name="default", options={"step_ids": step_ids}
            )
        finally:
            os.remove(path)

    def test_every_step_is_read_with_a_legal_shot_name(self):
        steps = self._resolve()
        # B03.5 becomes B03_5: a shot name is its exported clip name.
        self.assertEqual([s.step_id for s in steps], ["IN_01", "IN_02", "B03", "B03_5"])

    def test_sections_include_the_intro_banner(self):
        steps = self._resolve()
        self.assertEqual(
            [(s.section, s.section_title) for s in steps],
            [
                ("INTRO", ""),
                ("INTRO", ""),
                ("B", "Solenoid Lock Removal"),
                ("B", "Solenoid Lock Removal"),
            ],
        )

    def test_contents_voice_name_and_objects(self):
        intro, travel, _b03, branch = self._resolve()
        self.assertEqual(intro.description, "User starts in the room. Auto Advance")
        self.assertEqual(intro.audio, "Welcome to the task.")
        self.assertEqual(travel.audio, "Open the machine.\n\nSelect 'Next'.")
        self.assertEqual(intro._pass_through.get("step_name"), "Welcome")
        self.assertEqual(
            [(o.name, o.behaviors) for o in branch.objects],
            [("solenoid_geo", ["hide"])],
        )

    def test_the_quiz_block_adds_no_steps_or_text(self):
        *_, branch = self._resolve()
        self.assertNotIn("baffles", branch.description)

    def test_classic_ids_are_unchanged_by_the_new_patterns(self):
        # Still ``A01.)`` / bare caps: this layout's IDs are not steps to it.
        self.assertEqual(self._resolve("classic"), [])

    def test_auto_reads_it_as_numbered_and_is_the_default(self):
        """Which layout a sheet is written in is the sheet's, not the user's:
        a per-user choice left on ``classic`` by the last sheet read none of
        this one (2026-10-11, a production doc read as 0 steps)."""
        numbered = [s.step_id for s in self._resolve("numbered")]
        self.assertEqual([s.step_id for s in self._resolve("auto")], numbered)
        path = _write_csv(self.CSV)
        try:
            unset = mapping_mod.Mapping.resolve(path, name="default")
        finally:
            os.remove(path)
        self.assertEqual([s.step_id for s in unset], numbered)
        self.assertEqual(unset[0].section, "INTRO")  # the layout's sections too

    def test_infer_options_reads_back_what_a_build_recorded(self):
        """A scene built before it kept its sheet's options: its shots'
        ``csv_objects`` say how the sheet was read (a production scene:
        Object Case = upper), and the user's empty options read it otherwise.
        Added: 2026-10-12"""
        template = mapping_mod.Mapping.load_mapping("default")
        path = _write_csv(self.CSV)
        try:
            built = {"IN_02": ["DESTINATION_CIRCLE"], "B03_5": ["solenoid_geo"]}
            got = mapping_mod.Mapping.infer_options(path, template, built, {})
            self.assertEqual(got["object_case"], "upper")
            mine = {"step_ids": "numbered", "color_by": "section"}
            nothing = mapping_mod.Mapping.infer_options(path, template, {}, mine)
            self.assertEqual(
                {k: nothing[k] for k in mine}, mine, "no build: the user's own"
            )
            as_written = {"IN_02": ["Destination_circle"]}
            kept = mapping_mod.Mapping.infer_options(path, template, as_written, {})
            self.assertEqual(kept["object_case"], "keep", "a tie keeps the start")
        finally:
            os.remove(path)

    def test_auto_keeps_a_classic_sheet_classic(self):
        csv = (
            "SECTION X: OPENING SETUP\n"
            "Step,Step Contents,Asset Names,Voice Support\n"
            "SETUP,Room appears,,\n"
            "SECTION A: AILERON\n"
            "Step,Step Contents,Asset Names,Voice Support\n"
            "A01.),Aileron fades in,wing_L,Welcome\n"
            "A02.),Rudder appears,rudder,\n"
        )
        path = _write_csv(csv)
        try:
            steps = mapping_mod.Mapping.resolve(path, name="default")
        finally:
            os.remove(path)
        self.assertEqual([s.step_id for s in steps], ["A01", "A02"])  # SETUP excluded
        self.assertEqual(steps[0].section_title, "AILERON")


class TestColumnMapLayouts(unittest.TestCase):
    """``ColumnMap.layouts``: alternative row grammars tried on one read of the
    sheet; the one reading the most steps wins, the first on a tie."""

    CSV = (
        "Step,Step Contents,Asset Names\n"
        "A01.),Door fades in,door\n"
        "B2,Lid fades in,lid\n"
        "B3,Lid fades out,lid\n"
    )
    CLASSIC = {}
    NUMBERED = {"step_pattern": [r"^([A-Z]\d+)$"]}

    def _parse(self, *layouts):
        path = _write_csv(self.CSV)
        try:
            return ManifestModel.parse_csv(
                path, columns=ColumnMap.from_dict({"layouts": list(layouts)})
            )
        finally:
            os.remove(path)

    def test_the_layout_reading_the_most_steps_wins(self):
        steps = self._parse(self.CLASSIC, self.NUMBERED)
        self.assertEqual([s.step_id for s in steps], ["B2", "B3"])

    def test_a_tie_keeps_the_first(self):
        only_a = {"step_pattern": [r"^(A\d+)\.\)"]}
        steps = self._parse(only_a, {"step_pattern": [r"^(B2)$"]})
        self.assertEqual([s.step_id for s in steps], ["A01"])

    def test_round_trips_through_a_dict(self):
        cm = ColumnMap.from_dict({"layouts": [self.CLASSIC, self.NUMBERED]})
        self.assertEqual(ColumnMap.from_dict(cm.to_dict()), cm)

    def test_a_header_with_no_step_rows_says_why(self):
        path = _write_csv(self.CSV)
        try:
            with self.assertLogs(
                "pythontk.core_utils.engines.shots.manifest.manifest_model",
                level="WARNING",
            ) as logs:
                steps = ManifestModel.parse_csv(
                    path, columns=ColumnMap(step_pattern=(r"^(Z\d+)$",))
                )
        finally:
            os.remove(path)
        self.assertEqual(steps, [])
        self.assertIn("'A01.)'", "\n".join(logs.output))


class TestTemplateOptions(unittest.TestCase):
    """A template's ``options``: UI-ready specs, applied as patches merged over
    the template, validated as data."""

    TEMPLATE = {
        "columns": {"step_id": ["Step"], "exclude_steps": ["SETUP"]},
        "options": {
            "ids": {
                "kind": "choice",
                "label": "Step IDs",
                "default": "a",
                "choices": {
                    "a": {"label": "A", "set": {}},
                    "b": {
                        "label": "B",
                        "tooltip": "bee",
                        "set": {"columns": {"step_pattern": ["^(B\\d+)$"]}},
                    },
                },
            },
            "fill": {"kind": "bool", "set": {"fill_missing_assets": True}},
        },
    }

    def test_an_option_names_the_section_a_ui_groups_it_under(self):
        """A UI lists a template's options under titled sections, each named
        by its options; a section a file does not name is ``""``."""
        template = {
            "options": {
                "ids": {"kind": "bool", "section": "Sheet"},
                "fill": {"kind": "bool"},
            }
        }
        specs = mapping_mod.Mapping.option_specs(template)
        self.assertEqual([s["section"] for s in specs], ["Sheet", ""])
        # A template read without validation can carry a null: still "".
        unvalidated = {"options": {"ids": {"kind": "bool", "section": None}}}
        self.assertEqual(
            mapping_mod.Mapping.option_specs(unvalidated)[0]["section"], ""
        )
        res = mapping_mod.MappingSpec.validate(
            {"options": {"ids": {"kind": "bool", "section": 3}}}
        )
        self.assertFalse(res.ok)
        self.assertTrue(any("section" in e for e in res.errors), res.errors)

    def test_the_shipped_template_groups_every_option(self):
        """Grouped wherever the file puts them: a section's options are
        listed together, the sections in first-appearance order."""
        default = mapping_mod.Mapping.load_mapping("default")
        grouped = mapping_mod.Mapping.option_sections(default)
        self.assertEqual(
            [(title, [s["key"] for s in specs]) for title, specs in grouped],
            [
                ("Sheet", ["step_ids", "behaviors", "audio", "color_by"]),
                (
                    "Objects",
                    [
                        "objects",
                        "object_case",
                        "object_name_rule",
                        "fill_missing_assets",
                    ],
                ),
                ("Shots", ["match", "shots"]),
            ],
        )
        untitled = {
            "options": {"a": {"kind": "bool"}, "b": {"kind": "bool", "section": "X"}}
        }
        self.assertEqual(
            [t for t, _s in mapping_mod.Mapping.option_sections(untitled)], ["", "X"]
        )

    def test_specs_are_shaped_for_a_widget_factory(self):
        specs = mapping_mod.Mapping.option_specs(self.TEMPLATE)
        self.assertEqual(
            specs,
            [
                {
                    "key": "ids",
                    "kind": "choice",
                    "label": "Step IDs",
                    "tooltip": "",
                    "section": "",
                    "choices": [("A", "a", ""), ("B", "b", "bee")],
                    "default": "a",
                },
                {
                    "key": "fill",
                    "kind": "bool",
                    "label": "fill",
                    "tooltip": "",
                    "section": "",
                    "default": False,
                },
            ],
        )

    def test_defaults_leave_the_template_as_written(self):
        eff = mapping_mod.Mapping.apply_options(self.TEMPLATE)
        self.assertEqual(eff, {"columns": self.TEMPLATE["columns"]})

    def test_values_merge_their_patches_and_reapplying_is_a_no_op(self):
        eff = mapping_mod.Mapping.apply_options(
            self.TEMPLATE, {"ids": "b", "fill": True}
        )
        self.assertEqual(
            eff["columns"],
            {
                "step_id": ["Step"],
                "exclude_steps": ["SETUP"],
                "step_pattern": ["^(B\\d+)$"],
            },
        )
        self.assertTrue(eff["fill_missing_assets"])
        self.assertEqual(mapping_mod.Mapping.apply_options(eff, {"ids": "a"}), eff)

    def test_an_unknown_value_falls_back_to_the_default(self):
        eff = mapping_mod.Mapping.apply_options(self.TEMPLATE, {"ids": "gone"})
        self.assertNotIn("step_pattern", eff["columns"])

    def test_bad_options_are_schema_errors(self):
        bad = {
            "options": {
                "k": {"kind": "slider"},
                "c": {"kind": "choice", "default": "z", "choices": {"a": {}}},
                "p": {"kind": "bool", "set": {"options": {}}},
            }
        }
        errors = " | ".join(mapping_mod.MappingSpec.validate(bad).errors)
        self.assertIn("'slider'", errors)
        self.assertIn("not a choice", errors)
        self.assertIn("sets 'options'", errors)

    def test_a_try_choice_tries_the_named_choices_column_layouts(self):
        template = json.loads(json.dumps(self.TEMPLATE))
        template["options"]["ids"]["choices"]["auto"] = {"try": ["a", "b"]}
        self.assertFalse(mapping_mod.MappingSpec.validate(template).errors)
        eff = mapping_mod.Mapping.apply_options(template, {"ids": "auto"})
        self.assertEqual(
            eff["columns"]["layouts"], [{}, {"step_pattern": ["^(B\\d+)$"]}]
        )
        self.assertNotIn("step_pattern", eff["columns"])  # the base is the template's

    def test_a_bad_try_is_a_schema_error(self):
        template = json.loads(json.dumps(self.TEMPLATE))
        choices = template["options"]["ids"]["choices"]
        choices["auto"] = {"try": ["a", "gone"]}
        choices["loop"] = {"try": ["auto"]}
        choices["both"] = {"try": ["a"], "set": {}}
        choices["c"] = {"set": {"match": "name_then_order"}}
        choices["wide"] = {"try": ["c"]}
        errors = " | ".join(mapping_mod.MappingSpec.validate(template).errors)
        self.assertIn("'gone'", errors)
        self.assertIn("'loop'", errors)
        self.assertIn("'both'", errors)
        self.assertIn("'wide'", errors)


class TestAutoFillAssets(unittest.TestCase):
    """Steps a sheet lists no assets for take their objects from the scene,
    and the generated names go back out as a paste-ready Asset Names column."""

    CSV = (
        "INTRO,,,\n"
        "Step,Voice Support,Step Contents,Asset Names\n"
        "IN_01,Hi,Room appears,\n"
        ",,Auto Advance,\n"
        "SECTION A: Doors,,,\n"
        "Step,Voice Support,Step Contents,Asset Names\n"
        "A01,Open,Door fades in,door_geo\n"
        ",,Handle turns,\n"
        "A02,Close,Door closes,\n"
    )
    COLUMNS = dict(
        step_pattern=(r"^([A-Z]+_?\d+)$",),
        section_pattern=(r"(?i)^SECTION\s+([A-Z0-9]+)\s*:\s*(.*)", "^(INTRO)$"),
        exclude_steps=(),
    )

    def setUp(self):
        self.store = ShotStore()
        self.path = _write_csv(self.CSV)

    def tearDown(self):
        os.remove(self.path)

    def _steps(self):
        return ManifestModel.parse_csv(self.path, columns=ColumnMap(**self.COLUMNS))

    def test_a_multi_line_asset_cell_is_one_object_per_line(self):
        path = _write_csv(
            'Step,Step Contents,Asset Names\nA01.),Arm fades in,"arm_geo\n\nhand_geo"\n'
        )
        try:
            (step,) = ManifestModel.parse_csv(path)
        finally:
            os.remove(path)
        self.assertEqual(
            [(o.name, o.behaviors) for o in step.objects],
            [("arm_geo", ["show"]), ("hand_geo", ["show"])],
        )

    def test_a_paired_shot_fills_from_its_members_and_its_animation(self):
        """A paired shot's range is real, so what animates inside it counts
        (what Assess reports as additional); an unpaired step's range would be
        a guess, so A02 is never read."""
        self.store.define_shot("IN_01", 1, 40, objects=["|room|walls", "|room|floor"])
        mani = ShotManifest(self.store)
        read = []

        def discover(start, end, exclude):
            read.append((start, end))
            return ["|room|lamp"]

        mani._discover_scene_objects = discover
        steps = self._steps()
        filled = mani.fill_missing_assets(steps)

        # IN_01 from its shot; A01 lists an asset, untouched; A02 has no shot.
        self.assertEqual(filled, {"IN_01": ["floor", "walls", "lamp"]})
        self.assertEqual(read, [(1, 40)])
        by_id = {s.step_id: s for s in steps}
        self.assertEqual([o.name for o in by_id["A01"].objects], ["door_geo"])
        self.assertEqual(by_id["A02"].objects, [])
        self.assertEqual({o.origin for o in by_id["IN_01"].objects}, {"shot"})
        self.assertEqual(by_id["A01"].objects[0].origin, "column")

    def test_a_shots_stale_members_are_not_offered_as_assets(self):
        """A stored member the scene no longer holds would be written back to
        the sheet as an asset that does not exist."""
        self.store.define_shot("IN_01", 1, 40, objects=["|room|walls", "|room|gone"])
        mani = ShotManifest(self.store)
        mani._object_exists = lambda name: not name.endswith("gone")
        filled = mani.fill_missing_assets(self._steps())
        self.assertEqual(filled, {"IN_01": ["walls"]})

    def test_a_step_without_a_shot_stays_empty(self):
        filled = ShotManifest(self.store).fill_missing_assets(self._steps())
        self.assertEqual(filled, {})

class TestColumnMapRoundTrip(unittest.TestCase):
    """``ColumnMap.to_dict`` / ``from_dict`` round-trip every field, the string
    ones (``behavior_source``) included.

    Bug: ``to_dict`` listed every non-dict field, so a string came back as a
    tuple of its characters.
    Fixed: 2026-10-01
    """

    def test_every_field_survives(self):
        cm = ColumnMap(
            behaviors=("Behaviors",),
            behavior_source="column_else_description",
            step_pattern=(r"^(A\d+)$",),
            metadata_pass={"step_name": ("Step Name",)},
        )
        self.assertEqual(ColumnMap.from_dict(cm.to_dict()), cm)
        self.assertEqual(cm.to_dict()["behavior_source"], "column_else_description")


class TestRowPatterns(unittest.TestCase):
    """``ColumnMap.step_pattern`` / ``section_pattern``: the row grammar is
    data, validated as regexes."""

    def test_defaults_reproduce_the_historical_grammar(self):
        cm = ColumnMap()
        self.assertEqual(cm.step_pattern, (r"^([A-Z]\d+)\.\)", r"^([A-Z]{2,})$"))
        self.assertEqual(
            cm.section_pattern, (r"(?i)^SECTION\s+([A-Z0-9]+)\s*:\s*(.*)",)
        )

    def test_a_bad_regex_is_a_schema_error(self):
        res = mapping_mod.MappingSpec.validate({"columns": {"step_pattern": ["("]}})
        self.assertFalse(res.ok)
        self.assertTrue(any("step_pattern" in e for e in res.errors), res.errors)

    def test_fade_away_reads_as_hide(self):
        self.assertEqual(ManifestModel.detect_behaviors("Keys fade away"), ["hide"])


class TestBehaviors(unittest.TestCase):
    def test_list_builtins(self):
        self.assertEqual(
            sorted(beh.Behaviors.list_behaviors()),
            ["hide", "highlight", "set_clip", "show"],
        )

    def test_load_behavior(self):
        fi = beh.Behaviors.load_behavior("show")
        self.assertEqual((fi["effect"], fi["place"]), ("fade_in", "start"))
        self.assertNotIn("attributes", fi)  # the recipe keys it
        self.assertIn("visibility", beh.Behaviors.keyed(fi)["attributes"])

    def test_an_edited_user_template_is_read_again(self):
        """Bug: the validated load was cached per NAME for the life of the
        process, so a user template added over a built-in already loaded -- or
        edited afterwards -- changed nothing until a restart.
        Fixed: 2026-10-03
        """
        import copy

        base = copy.deepcopy(beh.Behaviors.load_behavior("show"))  # cached
        store = beh.Behaviors.templates().store
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(store, "_user_dir", Path(tmp)),
        ):
            path = Path(tmp) / "show.json"

            def place_after_writing(place, stamp):
                base["place"] = place
                path.write_text(json.dumps(base), encoding="utf-8")
                os.utime(path, (stamp, stamp))
                return beh.Behaviors.load_behavior("show")["place"]

            self.assertEqual(place_after_writing("end", 1_000_000_000), "end")
            self.assertEqual(place_after_writing(0.5, 1_000_000_100), 0.5)
        # Out of the override again: the built-in, not the last user copy.
        self.assertEqual(beh.Behaviors.load_behavior("show")["place"], "start")

    def test_resolve_keys_start_anchored_block(self):
        block = beh.Behaviors.keyed("show")["attributes"]["visibility"]["in"]
        keys = beh.Behaviors.resolve_keys(block, 10, 60)
        # anchor=start, offset=0, duration=15, values=[0,1] -> keys at 10 and 25
        self.assertEqual([k["time"] for k in keys], [10.0, 25.0])
        self.assertEqual([k["value"] for k in keys], [0.0, 1.0])

    def test_resolve_keys_end_anchored_block(self):
        out = beh.Behaviors.keyed("hide")["attributes"]["visibility"]["out"]
        keys = beh.Behaviors.resolve_keys(out, 10, 60)
        # end-anchored: keys land near the range end (60), ascending times
        self.assertTrue(keys)
        self.assertLessEqual(keys[-1]["time"], 60.0 + 1e-6)
        self.assertEqual([k["time"] for k in keys], sorted(k["time"] for k in keys))

    def test_phase_durations_sums_in_and_out_blocks(self):
        # The shared phase-walk primitive behind compute_duration and the
        # engine's resolve_duration: sums per-phase across all attributes.
        tmpl = {
            "attributes": {
                "visibility": {"in": {"duration": 15}, "out": {"duration": 10}},
                "scaleX": {"in": {"duration": 5}},
            }
        }
        self.assertEqual(beh.Behaviors.phase_durations(tmpl), (20.0, 10.0))
        self.assertEqual(beh.Behaviors.phase_durations({}), (0.0, 0.0))
        # None/absent durations count as zero, not a crash.
        self.assertEqual(
            beh.Behaviors.phase_durations(
                {"attributes": {"v": {"in": {"duration": None}}}}
            ),
            (0.0, 0.0),
        )

    def test_anchor_overrides_spread_points_and_leave_spans_whole(self):
        """Bug: one anchor per behavior, spread over ALL of an object's
        behaviors, was forced onto every block of a highlight beside any other
        behavior -- its in and out ramps collapsed onto the same frames, and the
        fades beside it were pushed off their ends (show keyed mid-shot).
        Points spread in doc order; a span keeps its own anchors.
        Fixed: 2026-10-03
        """
        overrides = beh.Behaviors.anchor_overrides
        self.assertEqual(overrides(["show"]), [None])
        self.assertEqual(overrides(["show", "hide"]), [0.0, 1.0])
        self.assertEqual(overrides(["show", "hide", "show"]), [0.0, 0.5, 1.0])
        self.assertEqual(overrides(["highlight", "hide"]), [None, None])
        self.assertEqual(
            overrides(["highlight", "show", "hide"]), [None, 0.0, 1.0]
        )
        # A name no template answers keeps its own (its apply records the failure).
        self.assertEqual(overrides(["no_such_behavior", "hide"]), [None, None])

    def test_entry_anchors_place_a_shots_entries_as_the_build_does(self):
        """Audio clips start the shot; each object's other behaviors spread
        over its own non-audio list, whatever lies between them."""
        entries = [
            {"name": "door", "behavior": "show"},
            {"name": "vo", "behavior": "set_clip", "kind": "audio"},
            {"name": "door", "behavior": "hide"},
            {"name": "lamp", "behavior": "show"},
            {"name": "vo2", "behavior": "set_clip", "source_path": "a.wav"},
        ]
        self.assertEqual(
            [beh.Behaviors.is_audio_entry(e) for e in entries],
            [False, True, False, False, True],
        )
        self.assertEqual(
            beh.Behaviors.entry_anchors(entries), [0.0, None, 1.0, None, None]
        )

    def test_anchored_is_the_one_anchor_precedence(self):
        """Override, then the block's own anchor, then its phase's end."""
        anchored = beh.Behaviors.anchored
        block = {"duration": 5, "values": [0, 1]}
        self.assertEqual(anchored(block, "in")["anchor"], "start")
        self.assertEqual(anchored(block, "out")["anchor"], "end")
        own = dict(block, anchor=0.25)
        self.assertEqual(anchored(own, "out")["anchor"], 0.25)
        self.assertEqual(anchored(own, "out", anchor_override=1.0)["anchor"], 1.0)
        self.assertNotIn("anchor", block, "the block itself is never written")

    def test_key_window_is_where_the_keys_would_fall(self):
        """Where an applier puts a behavior's keys, without keying: an effect
        at the recipe's window (the effect writer's own placement), an audio
        clip at the start, a template's own blocks by their anchors."""
        recipe = EffectRecipe(fade_frames=10.0)
        window = beh.Behaviors.key_window
        self.assertEqual(window("show", 100, 300, recipe=recipe), (100.0, 110.0))
        self.assertEqual(window("hide", 100, 300, recipe=recipe), (290.0, 300.0))
        self.assertEqual(
            window("show", 100, 300, anchor_override=0.5, recipe=recipe),
            recipe.window("fade_in", 100, 300, "start", 0.5),
        )
        self.assertEqual(window("highlight", 100, 300, recipe=recipe), (100.0, 300.0))
        self.assertEqual(window("set_clip", 100, 300, recipe=recipe), (100.0, 100.0))
        own = {
            "attributes": {
                "visibility": {
                    "in": {"duration": 4, "values": [0, 1]},
                    "out": {"duration": 6, "values": [1, 0]},
                }
            }
        }
        self.assertEqual(window(own, 100, 300), (100.0, 300.0))
        self.assertEqual(window(own, 100, 300, anchor_override="start"), (100.0, 106.0))
        self.assertIsNone(window("no_such_behavior", 100, 300))

    def test_a_highlight_beside_a_fade_keeps_its_ramps_apart(self):
        """What a host keys for a highlight beside a fade: the pulse over the
        whole shot, dim at both ends -- before the fix the highlight's ramps
        collapsed to (100, 1), (110, 0), a glow held for the whole timeline
        before the shot."""
        recipe = EffectRecipe()
        tmpl = beh.Behaviors.load_behavior("highlight")
        anchor = beh.Behaviors.anchor_overrides(["highlight", "hide"])[0]
        self.assertIsNone(anchor)
        window = recipe.window("pulse", 100, 300, tmpl["place"], anchor)
        self.assertEqual(window, (100.0, 300.0))
        keys = recipe.plan("pulse", *window, fps=30.0)
        self.assertEqual((keys[0], keys[-1]), ((100.0, 0.0), (300.0, 0.0)))
        self.assertEqual(max(v for _t, v in keys), 1.0)

    def test_compute_duration_dict_and_object_forms_agree(self):
        dict_form = beh.Behaviors.compute_duration(
            [{"name": "g", "behavior": "show"}]
        )
        obj_form = beh.Behaviors.compute_duration(
            [BuilderObject(name="g", behaviors=["show"])]
        )
        self.assertEqual(dict_form, obj_form)

    def test_compute_duration_fallback_when_empty(self):
        self.assertEqual(beh.Behaviors.compute_duration([], fallback=200), 200)

    def test_compute_duration_audio_callback(self):
        obj = BuilderObject(
            name="vo", behaviors=["set_clip"], kind="audio", source_path="x.wav"
        )
        got = beh.Behaviors.compute_duration(
            [obj], fallback=10, audio_duration_fn=lambda src: 123.0
        )
        self.assertGreaterEqual(got, 123.0)

    def test_compute_duration_resolve_source_fallback(self):
        # An entry with NO source_path resolves via resolve_source_fn and is
        # then measured by audio_duration_fn — the seam a DCC layer binds to
        # its own track registry (populated independently of the CSV).
        obj = BuilderObject(
            name="vo", behaviors=["set_clip"], kind="audio", source_path=""
        )
        seen = {}

        def resolver(name, kind):
            seen["args"] = (name, kind)
            return "resolved.wav"

        got = beh.Behaviors.compute_duration(
            [obj],
            fallback=10,
            audio_duration_fn=lambda src: 77.0 if src == "resolved.wav" else None,
            resolve_source_fn=resolver,
        )
        self.assertEqual(seen["args"], ("vo", "audio"))
        self.assertGreaterEqual(got, 77.0)

    def test_compute_duration_raising_resolver_falls_back(self):
        # A raising resolver is swallowed; the unresolvable entry contributes
        # nothing so the manifest falls back instead of crashing the build.
        obj = BuilderObject(
            name="vo", behaviors=["set_clip"], kind="audio", source_path=""
        )

        def boom(name, kind):
            raise RuntimeError("no registry")

        got = beh.Behaviors.compute_duration(
            [obj],
            fallback=10,
            audio_duration_fn=lambda src: 99.0,
            resolve_source_fn=boom,
        )
        self.assertEqual(got, 10)


class TestRangeResolver(unittest.TestCase):
    def _steps(self, n):
        return [
            BuilderStep(f"A0{i}", "A", "t", "", [BuilderObject(f"g{i}")])
            for i in range(1, n + 1)
        ]

    def test_incremental_ranges_keep_the_built_and_place_the_new(self):
        """A Build once some steps have shots: the built keep their shots'
        bounds -- a range typed before the build never puts the old ones
        back -- a new step lands at its predecessor's end, zero-length, and
        the first at the first placed step's start."""
        steps = self._steps(4)  # A01 .. A04
        ranges = rr.RangeResolver.incremental_ranges(
            steps,
            shot_ranges={"A02": (100.0, 150.0)},
            last_resolved=[("A03", 160.0, 200.0, False), ("A02", 1.0, 2.0, False)],
            user_ranges={"A02": (5.0, 9.0)},
        )
        self.assertEqual(
            ranges,
            {
                "A01": (100.0, 100.0),
                "A02": (100.0, 150.0),
                "A03": (160.0, 200.0),
                "A04": (200.0, 200.0),
            },
        )

    def test_incremental_ranges_take_a_typed_range_and_a_typed_start(self):
        steps = self._steps(3)
        ranges = rr.RangeResolver.incremental_ranges(
            steps,
            shot_ranges={"A01": (0.0, 50.0)},
            user_ranges={"A02": (60.0, 80.0), "A03": (300.0, None)},
        )
        self.assertEqual(ranges["A02"], (60.0, 80.0))
        self.assertEqual(ranges["A03"], (300.0, 300.0), "a start alone: there")
        self.assertEqual(
            rr.RangeResolver.incremental_ranges(steps, {}),
            {s.step_id: (1.0, 1.0) for s in steps},
            "nothing placed: frame 1",
        )

    def test_a_step_with_no_shot_takes_no_detected_region(self):
        """Surplus regions are pruned to the steps that take one: a step
        with no shot (``animated`` False) counts for none, so the animated
        steps keep the major boundaries, not the small ones."""
        steps = self._steps(3)
        steps[1].animated = False
        resolved = rr.RangeResolver.resolve_ranges(
            steps, {}, [0.0, 1.0, 2.0, 100.0, 101.0], {}, 0.0, True, []
        )
        self.assertEqual([(r[0], r[1]) for r in resolved], [("A01", 0.0), ("A03", 100.0)])

    def test_prune_to_top_boundaries(self):
        # starts 0,1,2,100,101 -> largest gap before 100; keep 2 -> [0,100]
        self.assertEqual(
            rr.RangeResolver.prune_to_top_boundaries([0, 1, 2, 100, 101], 2), [0, 100]
        )

    def test_prune_noop_when_within_budget(self):
        self.assertEqual(rr.RangeResolver.prune_to_top_boundaries([0, 10], 3), [0, 10])

    def test_sequential_default_duration(self):
        steps = self._steps(3)
        out = rr.RangeResolver.resolve_ranges(
            steps,
            {},
            [],
            {},
            gap=10,
            use_selected_keys=False,
            last_resolved=[],
            default_duration=200,
        )
        # uniform 200f each, gap 10, anchored at 0
        self.assertEqual(
            [(sid, s, e) for sid, s, e, _ in out],
            [("A01", 0.0, 200.0), ("A02", 210.0, 410.0), ("A03", 420.0, 620.0)],
        )

    def test_user_range_pins_and_advances_cursor(self):
        steps = self._steps(2)
        out = rr.RangeResolver.resolve_ranges(
            steps,
            {"A01": (50.0, 100.0)},
            [],
            {},
            gap=10,
            use_selected_keys=False,
            last_resolved=[],
            default_duration=200,
        )
        first = out[0]
        self.assertEqual(
            (first[0], first[1], first[2], first[3]), ("A01", 50.0, 100.0, True)
        )
        # A02 starts after A01's end + gap
        self.assertEqual(out[1][1], 110.0)

    def test_gap_boundaries_place_steps(self):
        steps = self._steps(2)
        out = rr.RangeResolver.resolve_ranges(
            steps,
            {},
            [30.0, 80.0],
            {},
            gap=5,
            use_selected_keys=False,
            last_resolved=[],
            default_duration=0,
        )
        self.assertEqual(out[0][1], 30.0)
        self.assertEqual(out[1][1], 80.0)

    def test_selected_keys_no_regions_returns_empty(self):
        out = rr.RangeResolver.resolve_ranges(
            self._steps(2),
            {},
            [],
            {},
            gap=0,
            use_selected_keys=True,
            last_resolved=[],
        )
        self.assertEqual(out, [])


class TestStatusRollup(unittest.TestCase):
    def test_worst_of_children(self):
        st = StepStatus(
            step_id="A01",
            built=True,
            objects=[
                ObjectStatus("a", True, "valid"),
                ObjectStatus("b", False, "missing_object"),
            ],
        )
        self.assertEqual(st.status, "missing_object")
        self.assertEqual(st.missing_count, 1)

    def test_locked_wins(self):
        st = StepStatus(
            step_id="A01",
            built=True,
            locked=True,
            objects=[ObjectStatus("a", False, "missing_object")],
        )
        self.assertEqual(st.status, "locked")

    def test_unbuilt(self):
        st = StepStatus(step_id="A01", built=False)
        self.assertEqual(st.status, "missing_shot")

    def test_edited_is_named_after_locked(self):
        st = StepStatus(
            step_id="A01",
            built=True,
            edited=True,
            objects=[ObjectStatus("a", False, "missing_object")],
        )
        self.assertEqual(st.status, "edited")
        self.assertFalse(st.needs_build)
        self.assertIn("edited", StepStatus.HELP)

    def test_the_rollup_names_the_worst_severity_first(self):
        """A step is named after its worst object at the severity a panel
        shows it at (``SHOT_STATUS_LEVELS``, the DCC panels' ``worst()``).

        Bug: ``behavior_conflict`` (an error) ranked below
        ``ambiguous_object`` and ``not_in_shot`` (warnings): a step holding
        both was named by the warning while its icon showed the error.
        Fixed: 2026-10-12
        """
        from pythontk.core_utils.engines.shots.shot_model import SHOT_STATUS_LEVELS

        severity = ("valid", "locked", "info", "warn", "error").index
        ranks = [severity(SHOT_STATUS_LEVELS[s]) for s in StepStatus.ROLLUP]
        self.assertEqual(ranks, sorted(ranks, reverse=True), StepStatus.ROLLUP)
        st = StepStatus(
            step_id="A01",
            built=True,
            objects=[
                ObjectStatus("a", True, "ambiguous_object"),
                ObjectStatus("b", True, "behavior_conflict"),
            ],
        )
        self.assertEqual(st.status, "behavior_conflict")

    def test_a_hand_edited_step_is_marked_apart_from_a_locked_one(self):
        """Edited-by-hand and locked shared one severity, so a panel drew both
        with the lock's glyph and colour: two states, one mark.
        Fixed: 2026-11-22
        """
        from pythontk.core_utils.engines.shots.shot_model import SHOT_STATUS_LEVELS

        self.assertNotEqual(SHOT_STATUS_LEVELS["edited"], "locked")
        self.assertNotIn(SHOT_STATUS_LEVELS["edited"], ("warn", "error"), "kept on purpose: no alarm")

    def test_every_status_says_itself_in_a_few_words(self):
        """A row says its cause without a hover: each status a panel shows has
        a label of three words at most, beside its sentence (HELP).
        Added: 2026-11-22
        """
        from pythontk.core_utils.engines.shots.shot_model import SHOT_STATUS_LEVELS

        shown = set(StepStatus.HELP) | set(StepStatus.ROLLUP) | {
            s for s, level in SHOT_STATUS_LEVELS.items() if level != "valid"
        }
        for status in sorted(shown):
            label = StepStatus.LABELS.get(status, "")
            self.assertTrue(label, status)
            self.assertLessEqual(len(label.split()), 3, f"{status}: {label}")

    def test_a_steps_label_names_its_cause_and_how_many(self):
        """Added: 2026-11-22"""
        st = StepStatus(
            step_id="A01",
            built=True,
            objects=[
                ObjectStatus("a", False, "missing_object"),
                ObjectStatus("b", False, "missing_object"),
                ObjectStatus("c", True, "missing_behavior"),
            ],
        )
        self.assertEqual(st.label, f"2 {StepStatus.LABELS['missing_object']}")
        self.assertEqual(StepStatus(step_id="A02", built=False).label, StepStatus.LABELS["missing_shot"])
        self.assertEqual(
            StepStatus(step_id="A03", built=True, objects=[ObjectStatus("a", True, "valid")]).label, ""
        )
        renamed = StepStatus(
            step_id="A04", built=True, objects=[ObjectStatus("a", True, "valid")], updates=["name"]
        )
        self.assertEqual(renamed.label, StepStatus.LABELS["shot_update"], "valid, yet Build changes it")

    def test_a_plan_counts_what_a_build_does_keeps_and_cannot_do(self):
        """What Build would do with assessed steps, before it is pressed: the
        shots it creates, those it updates, those it keeps (edited by hand),
        those it skips (locked), and the steps waiting on a person (an object
        no Build can supply). Added: 2026-11-22
        """
        ok = ObjectStatus("a", True, "valid")
        results = [
            StepStatus(step_id="A01", built=False, objects=[ok]),
            StepStatus(step_id="A02", built=True, objects=[ObjectStatus("b", True, "missing_behavior")]),
            StepStatus(step_id="A03", built=True, edited=True, objects=[ok]),
            StepStatus(step_id="A04", built=True, locked=True, objects=[ok]),
            StepStatus(step_id="A05", built=True, objects=[ObjectStatus("c", False, "missing_object")]),
            StepStatus(
                step_id="A06",
                built=True,
                objects=[ObjectStatus("d", False, "missing_object"), ObjectStatus("e", True, "not_in_shot")],
            ),
            StepStatus(step_id="A07", built=True, objects=[ok]),
        ]
        plan = StepStatus.plan(results)
        self.assertEqual(
            plan,
            {"create": 1, "update": 2, "keep": 1, "skip": 1, "needs_you": 2, "in_sync": 1, "no_shot": 0},
        )
        self.assertEqual(plan["create"] + plan["update"], sum(1 for r in results if r.needs_build))
        text = StepStatus.plan_text(plan)
        for part in ("1 to create", "2 to update", "1 kept", "1 locked", "2 need you"):
            self.assertIn(part, text)
        self.assertEqual(StepStatus.plan_text(StepStatus.plan([results[-1]])), "Nothing to build: 1 in sync.")


class TestManifestEngine(unittest.TestCase):
    """The pure ShotManifest planner + commit against a real ShotStore."""

    def setUp(self):
        self.store = ShotStore()
        self.mani = ShotManifest(self.store)

    def _steps(self, ids):
        return [
            BuilderStep(sid, "A", "t", "", [BuilderObject(f"{sid}_geo")]) for sid in ids
        ]

    def test_sync_creates_shots_sequentially(self):
        actions, _, assessment = self.mani.sync(
            self._steps(["A01", "A02"]), initial_shot_length=100
        )
        self.assertEqual(actions, {"A01": "created", "A02": "created"})
        shots = self.store.sorted_shots()
        self.assertEqual([s.name for s in shots], ["A01", "A02"])
        self.assertEqual((shots[0].start, shots[0].end), (1.0, 101.0))
        self.assertEqual({a.status for a in assessment}, {"valid"})

    def test_a_step_another_shot_holds_ignoring_case_is_refused_not_a_crash(self):
        """``a02`` and step ``A02`` are one clip: the store refuses the step's
        name, and the build reports it instead of aborting part-way."""
        self.store.define_shot("a02", 500, 600)
        with self.assertLogs("pythontk.core_utils.engines.shots.manifest", "WARNING"):
            actions, _, _ = self.mani.sync(
                self._steps(["A01", "A02"]),
                remove_missing=False,
                initial_shot_length=100,
            )
        self.assertEqual(actions, {"A01": "created", "A02": "refused"})
        self.assertEqual(sorted(s.name for s in self.store.shots), ["A01", "a02"])

    def test_a_step_whose_clash_is_being_removed_is_built(self):
        """Removals commit first, so the name is free by the time it is built."""
        self.store.define_shot("a02", 500, 600)
        actions, _, _ = self.mani.sync(
            self._steps(["A01", "A02"]), initial_shot_length=100
        )
        self.assertEqual(actions["A02"], "created")
        self.assertEqual(sorted(s.name for s in self.store.shots), ["A01", "A02"])

    def test_resync_unchanged_is_skipped(self):
        steps = self._steps(["A01"])
        self.mani.sync(steps, initial_shot_length=100)
        actions, _, _ = self.mani.sync(steps, initial_shot_length=100)
        self.assertEqual(actions, {"A01": "skipped"})

    def test_remove_missing(self):
        self.mani.sync(self._steps(["A01", "A02"]), initial_shot_length=100)
        actions, _, _ = self.mani.sync(self._steps(["A01"]), initial_shot_length=100)
        self.assertEqual(actions["A02"], "removed")
        self.assertEqual([s.name for s in self.store.sorted_shots()], ["A01"])

    def test_explicit_ranges_place_shots(self):
        steps = self._steps(["A01", "A02"])
        ranges = {"A01": (10.0, 40.0), "A02": (50.0, 90.0)}
        self.mani.sync(steps, ranges=ranges, initial_shot_length=100)
        shots = {s.name: (s.start, s.end) for s in self.store.sorted_shots()}
        self.assertEqual(shots["A01"], (10.0, 40.0))
        self.assertEqual(shots["A02"], (50.0, 90.0))

    def test_resync_changed_objects_is_patched(self):
        self.mani.sync(self._steps(["A01"]), initial_shot_length=100)
        steps2 = [BuilderStep("A01", "A", "t", "", [BuilderObject("new_geo")])]
        actions, _, _ = self.mani.sync(steps2, initial_shot_length=100)
        self.assertEqual(actions["A01"], "patched")
        self.assertIn("new_geo", self.store.shot_by_name("A01").objects)

    def test_existing_audio_shot_regrows_when_clip_lengthens(self):
        measured = {"v": 100.0}

        class AudioMani(ShotManifest):
            def _measure_audio(self, obj):
                return measured["v"]

        store = ShotStore()
        mani = AudioMani(store)
        step = [
            BuilderStep(
                "V01",
                "V",
                "t",
                "",
                [BuilderObject("vo", kind="audio", source_path="x.wav")],
            )
        ]
        mani.sync(step, initial_shot_length=50)
        self.assertAlmostEqual(store.sorted_shots()[0].duration, 100.0)
        # clip lengthens -> re-sync must grow the EXISTING shot (the fixed path)
        measured["v"] = 300.0
        mani.sync(step, initial_shot_length=50)
        self.assertAlmostEqual(store.sorted_shots()[0].duration, 300.0)

    def test_a_shot_grown_after_a_part_frame_grow_upstream_still_holds_its_clip(self):
        # A production sheet: an earlier shot grew by part of a frame, the store
        # snapped each bound on its own, and a later shot grown for its
        # 138.51-frame narration came out 138 long, its clip's OFF key (start
        # + 139: the key writer rounds as shot bounds do) past its end.
        lengths = {"vo_a": 100.0, "vo_b": 138.0}

        class AudioMani(ShotManifest):
            def _measure_audio(self, obj):
                return lengths[obj.name]

        store = ShotStore()
        mani = AudioMani(store)
        steps = [
            BuilderStep("A01", "A", "t", "", [BuilderObject("vo_a", kind="audio", source_path="a.wav")]),
            BuilderStep("B01", "B", "t", "", [BuilderObject("vo_b", kind="audio", source_path="b.wav")]),
        ]
        mani.sync(steps, initial_shot_length=50)
        b_start = store.shot_by_name("B01").start
        lengths.update(vo_a=100.6, vo_b=138.51)
        mani.sync(steps, initial_shot_length=50)
        a, b = store.shot_by_name("A01"), store.shot_by_name("B01")
        self.assertEqual(a.end - a.start, 101.0)
        self.assertEqual(b.start, b_start + 1.0, "moved by the whole frame A grew")
        self.assertEqual(b.end - b.start, 139.0, "ends on its clip's OFF key, round(138.51)")

    def test_a_grown_shot_moves_the_shots_after_it_through_the_sequencer(self):
        # Written as bounds alone, a grow moved the later shots off their
        # animation (a production sheet: every shot after the first narration).
        # It is the sequencer's edge drag, whose ripple moves a shot's keys
        # and audio with it on a host.
        from pythontk.core_utils.engines.shots.shot_sequencer import ShotSequencer

        lengths = {"vo_a": 100.0, "vo_b": 50.0}
        ripples = []

        class Seq(ShotSequencer):
            def ripple_downstream(self, shot_id, after_frame, delta, carry_gap=True):
                ripples.append((self.store.shot_by_id(shot_id).name, after_frame, delta))
                super().ripple_downstream(shot_id, after_frame, delta, carry_gap=carry_gap)

        class AudioMani(ShotManifest):
            def _measure_audio(self, obj):
                return lengths[obj.name]

            def _sequencer(self):
                return Seq(store=self.store)

        store = ShotStore()
        mani = AudioMani(store)
        steps = [
            BuilderStep("A01", "A", "t", "", [BuilderObject("vo_a", kind="audio", source_path="a.wav")]),
            BuilderStep("B01", "B", "t", "", [BuilderObject("vo_b", kind="audio", source_path="b.wav")]),
        ]
        mani.sync(steps, initial_shot_length=50)
        a_end, b_start = store.shot_by_name("A01").end, store.shot_by_name("B01").start
        lengths["vo_a"] = 120.0
        mani.sync(steps, initial_shot_length=50)
        self.assertEqual(ripples, [("A01", a_end, 20.0)])
        self.assertEqual(store.shot_by_name("A01").end, a_end + 20.0)
        self.assertEqual(store.shot_by_name("B01").start, b_start + 20.0)

    def test_a_shot_grown_for_its_narration_leaves_its_take_on_the_animation(self):
        # A step's narration is its shot's, never its take's: grown with the shot, the take shipped a clip
        # with seconds of nothing after the motion (a production sheet: a 50-frame latch in a 267-frame clip).
        lengths = {"vo": 40.0}

        class AudioMani(ShotManifest):
            def _measure_audio(self, obj):
                return lengths[obj.name]

        store = ShotStore()
        mani = AudioMani(store)
        step = BuilderStep("A01", "A", "t", "", [BuilderObject("vo", kind="audio", source_path="a.wav")])
        step.takes = ["A01"]
        mani.sync([step], initial_shot_length=50)
        shot = store.shot_by_name("A01")
        self.assertEqual(store.takes_of(shot), [("A01", shot.start, shot.end)], "its one take: the whole shot")
        was = shot.end
        lengths["vo"] = 300.0
        mani.sync([step], initial_shot_length=50)
        self.assertEqual(shot.end - shot.start, 300.0)
        self.assertEqual(store.takes_of(shot), [("A01", shot.start, was)], "where the shot ended before it grew")
        mani.sync([step], initial_shot_length=50)
        self.assertEqual(store.takes_of(shot), [("A01", shot.start, was)], "and a build after keeps it")

    def test_a_step_that_only_speaks_is_as_long_as_its_narration(self):
        # The room a new shot starts with is for animating in; a step that plays no take needs none
        # (a production sheet: 41 pages at 200 frames each for lines of 30 to 90).
        class AudioMani(ShotManifest):
            def _measure_audio(self, obj):
                return 40.0

        def step(sid, takes):
            made = BuilderStep(sid, "A", "t", "", [BuilderObject("vo_" + sid, kind="audio", source_path="a.wav")])
            made.takes = takes
            return made

        store = ShotStore()
        AudioMani(store).sync([step("A01", []), step("A02", ["A02"])], initial_shot_length=200)
        self.assertEqual([(s.name, s.end - s.start) for s in store.sorted_shots()], [("A01", 40.0), ("A02", 200.0)])

    def test_audio_grow_via_measure_hook(self):
        class AudioMani(ShotManifest):
            def _measure_audio(self, obj):
                return 500.0

        store = ShotStore()
        mani = AudioMani(store)
        step = [
            BuilderStep(
                "V01",
                "V",
                "t",
                "",
                [BuilderObject("vo", kind="audio", source_path="x.wav")],
            )
        ]
        mani.sync(step, initial_shot_length=50)
        sh = store.sorted_shots()[0]
        self.assertAlmostEqual(sh.end - sh.start, 500.0)

    def test_audio_grow_duration_override_hook_is_respected(self):
        # A DCC layer may override _audio_grow_duration (e.g. to route through
        # its own compute_duration binding); the re-sync grow pass must honor
        # the override, not just the _measure_audio default underneath it.
        class GrowMani(ShotManifest):
            def _measure_audio(self, obj):
                return 100.0

            def _audio_grow_duration(self, audio_objs):
                return 250.0

        store = ShotStore()
        mani = GrowMani(store)
        step = [
            BuilderStep(
                "V01",
                "V",
                "t",
                "",
                [BuilderObject("vo", kind="audio", source_path="x.wav")],
            )
        ]
        mani.sync(step, initial_shot_length=50)
        mani.sync(step, initial_shot_length=50)  # grow pass on the existing shot
        sh = store.sorted_shots()[0]
        self.assertAlmostEqual(sh.end - sh.start, 250.0)

    def test_audio_grow_duration_default_is_max_of_template_and_measured(self):
        # Pure default contract: the larger of the behavior-template duration
        # and the measured clip length.
        objs = [
            BuilderObject(
                "vo", kind="audio", source_path="x.wav", behaviors=["show"]
            )
        ]

        class ShortClip(ShotManifest):
            def _measure_audio(self, obj):
                return 5.0

        class LongClip(ShotManifest):
            def _measure_audio(self, obj):
                return 99.0

        self.assertEqual(ShortClip(ShotStore())._audio_grow_duration(objs), 15.0)
        self.assertEqual(LongClip(ShotStore())._audio_grow_duration(objs), 99.0)

    def test_from_csv_classmethod_returns_subclass(self):
        class Sub(ShotManifest):
            pass

        st = ShotStore()
        m = Sub(st)
        self.assertIsInstance(m, Sub)

    def test_resolve_duration_pure_behavior_summation(self):
        step = BuilderStep(
            "A01", "A", "t", "", [BuilderObject("g", behaviors=["show"])]
        )
        dur, beh_span, aud = ShotManifest.resolve_duration(
            step, initial_shot_length=200, fit_mode="extend_only", fps=24.0
        )
        self.assertGreaterEqual(dur, 0.0)


class _ScenicHost(PureShotsHost):
    """A host whose scene holds long, namespaced paths: ``door`` lives at
    ``|set|AC:door``; names in ``missing`` answer to no node, names in
    ``ambiguous`` to two."""

    def __init__(self):
        self.missing, self.ambiguous = set(), set()

    def member_nodes(self, name):
        if name in self.missing:
            return []
        if name in self.ambiguous:
            return [f"|a|{name}", f"|b|{name}"]
        return [f"|set|AC:{ShotStore.member_key(name)}"]


class _EngineCase(unittest.TestCase):
    def setUp(self):
        self.store = ShotStore(host=_ScenicHost())

    @staticmethod
    def _step(sid, *objs, behaviors=None):
        return BuilderStep(
            sid,
            "A",
            "t",
            "",
            [
                BuilderObject(o, behaviors=list((behaviors or {}).get(o, [])))
                for o in objs
            ],
        )


class TestShotPairing(_EngineCase):
    """``ShotManifest.pair``: the one place a step finds its shot -- the
    stored binding, then the (legal) name, then, when the template asks,
    timeline order.  A shot bound to a step the doc no longer has is an
    orphan, never re-paired by name."""

    def test_binding_beats_name_and_survives_a_rename(self):
        self.store.define_shot("Step_2_1", 1, 10, metadata={"step": "A01"})
        self.store.define_shot("A01", 20, 30)
        pairing = ShotManifest(self.store).pair([self._step("A01")])
        self.assertEqual(pairing.shots["A01"].name, "Step_2_1")
        self.assertEqual(pairing.how["A01"], "binding")
        self.assertEqual([s.name for s in pairing.orphans], ["A01"])

    def test_a_shot_bound_elsewhere_is_not_taken_by_name(self):
        self.store.define_shot("A02", 1, 10, metadata={"step": "A01"})
        pairing = ShotManifest(self.store).pair([self._step("A02")])
        self.assertNotIn("A02", pairing.shots)
        self.assertEqual([s.name for s in pairing.orphans], ["A02"])

    def test_order_pairs_the_rest_only_when_asked(self):
        self.store.define_shot("Step_1", 1, 10)
        self.store.define_shot("Step_2", 20, 30)
        steps = [self._step("IN_01"), self._step("IN_02")]
        self.assertEqual(ShotManifest(self.store).pair(steps).shots, {})
        pairing = ShotManifest(self.store, match="name_then_order").pair(steps)
        self.assertEqual(
            {k: v.name for k, v in pairing.shots.items()},
            {"IN_01": "Step_1", "IN_02": "Step_2"},
        )
        self.assertEqual(set(pairing.how.values()), {"order"})

    def test_the_scenes_own_shots_pair_with_themselves_whatever_their_binding(self):
        """Scene-shots mode: a shot bound to a doc step (order-paired once)
        is that step, so its step is the binding, and it pairs.

        Bug: ``from_shots`` named the step after the shot, so a shot bound to
        another step id was 'bound elsewhere' and never paired with itself.
        Fixed: 2026-10-01"""
        self.store.define_shot("Step_1", 1, 10, metadata={"step": "IN_01"})
        self.store.define_shot("Step_2", 20, 30)
        steps, ranges = BuilderStep.from_shots(self.store.sorted_shots())
        self.assertEqual([s.step_id for s in steps], ["IN_01", "Step_2"])
        self.assertEqual(ranges["IN_01"], (1, 10))
        pairing = ShotManifest(self.store).pair(steps)
        self.assertEqual(
            {k: v.name for k, v in pairing.shots.items()},
            {"IN_01": "Step_1", "Step_2": "Step_2"},
        )
        self.assertEqual(pairing.orphans, [])

    def test_a_renamed_step_pairs_with_the_shot_of_its_new_name(self):
        self.store.define_shot("Intro", 1, 10)
        step = self._step("IN_01")
        step.name = "Intro"
        pairing = ShotManifest(self.store).pair([step])
        self.assertEqual(pairing.shots["IN_01"].name, "Intro")
        self.assertEqual(pairing.how["IN_01"], "name")

    def _split_scene(self):
        """A sheet built, then edited in the Shot Sequencer: A01 split (its
        second half unbound), a lead-in before everything and a tail shot."""
        self.store.define_shot("Lead", -20, -10)
        self.store.define_shot("A01_1", 0, 10, metadata={"step": "A01"})
        self.store.define_shot("A01_2", 12, 20)
        self.store.define_shot("A02", 30, 40, metadata={"step": "A02"})
        self.store.define_shot("Tail", 50, 60)
        steps = [self._step(s) for s in ("IN_01", "A01", "A01b", "A02", "A03")]
        return steps, ShotManifest(self.store).pair(steps)

    @staticmethod
    def _names(rows):
        return [step.step_id if step else f"[{shot.name}]" for step, shot in rows]

    def test_unpaired_shots_sit_where_the_timeline_has_them(self):
        """Bug: a shot no step pairs with (split off or added in the Shot
        Sequencer) was listed after the last step, away from the shots it
        sits between ("added shots are appended to the end of the manifest
        instead of shown in shot order").  Each now follows the row of the
        shot before it on the timeline -- one with none before it leads the
        row of the shot after it.
        Fixed: 2026-10-10"""
        steps, pairing = self._split_scene()
        self.assertEqual(
            self._names(pairing.rows(steps)),
            ["IN_01", "[Lead]", "A01", "[A01_2]", "A01b", "A02", "[Tail]", "A03"],
        )
        step, shot = pairing.rows(steps)[2]
        self.assertEqual((step.step_id, shot.name), ("A01", "A01_1"))

    def test_hiding_missing_shots_keeps_every_shot(self):
        steps, pairing = self._split_scene()
        self.assertEqual(
            self._names(pairing.rows(steps, hide_missing=True)),
            ["[Lead]", "A01", "[A01_2]", "A02", "[Tail]"],
        )
        shown = pairing.rows(steps, hide_missing=True, keep={"A03"})
        self.assertIn("A03", self._names(shown), "a kept step shows unbuilt")

    def test_a_pairing_without_a_timeline_still_lists_its_orphans(self):
        """Bug: a ``ShotPairing`` made by hand (orphans, no ``timeline``)
        dropped every orphan from its rows -- there was no timeline to place
        them by.  They close the list instead.
        Fixed: 2026-10-10"""
        steps, pairing = self._split_scene()
        bare = ShotPairing(shots=pairing.shots, orphans=pairing.orphans)
        self.assertEqual(
            self._names(bare.rows(steps)),
            ["IN_01", "A01", "A01b", "A02", "A03", "[Lead]", "[A01_2]", "[Tail]"],
        )

    def test_rows_without_orphans_are_the_steps(self):
        steps, pairing = self._split_scene()
        self.assertEqual(
            self._names(pairing.rows(steps, orphans=False)),
            ["IN_01", "A01", "A01b", "A02", "A03"],
        )

    def test_bind_makes_a_shot_its_steps_and_takes_the_binding_from_another(self):
        steps, _ = self._split_scene()
        mani = ShotManifest(self.store)
        mani.bind(self.store.shot_by_name("A01_2"), "A01b")
        pairing = mani.pair(steps)
        self.assertEqual(pairing.shots["A01b"].name, "A01_2")
        self.assertEqual(pairing.how["A01b"], "binding")
        mani.bind(self.store.shot_by_name("Tail"), "A01b")
        self.assertNotIn("step", self.store.shot_by_name("A01_2").metadata)
        self.assertEqual(mani.pair(steps).shots["A01b"].name, "Tail")
        mani.bind(self.store.shot_by_name("Tail"), None)
        self.assertNotIn("A01b", mani.pair(steps).shots)

    def test_a_split_off_shot_is_linked_to_its_step_never_paired(self):
        """Bug: a shot split off a step's shot had no relation to that step,
        so nothing could say which section it was in (Unity read a null
        section for 8 of a production scene's 28 clips).  The split's link
        -- or, for a tail split before it wrote one, its ``<step>_<n>`` name
        -- now links it; it stays an orphan, the step's shot unchanged.
        Fixed: 2026-10-11"""
        steps, pairing = self._split_scene()
        a01_2 = self.store.shot_by_name("A01_2").shot_id
        self.assertEqual(pairing.links, {a01_2: "A01"})
        self.assertEqual(pairing.shots["A01"].name, "A01_1")
        self.assertIn(a01_2, [s.shot_id for s in pairing.orphans])
        self.store.update_shot(a01_2, metadata={"step_link": "A02"})
        self.assertEqual(ShotManifest(self.store).pair(steps).links, {a01_2: "A02"})

    def test_order_pairing_passes_a_split_off_shot_by(self):
        """Bug: with ``match="name_then_order"`` a tail split off a step's
        shot was taken, in timeline order, as the shot of the next unbuilt
        step (A01b), where it is its own step's tail.
        Fixed: 2026-10-12 (maintainer) -- linked shots are left out of the
        order pass; the others pair in order as before."""
        steps, _ = self._split_scene()
        pairing = ShotManifest(self.store, match="name_then_order").pair(steps)
        self.assertEqual(
            {sid: shot.name for sid, shot in pairing.shots.items()},
            {"IN_01": "Lead", "A01": "A01_1", "A01b": "Tail", "A02": "A02"},
        )
        self.assertEqual(pairing.how["A01b"], "order")
        a01_2 = self.store.shot_by_name("A01_2").shot_id
        self.assertEqual(pairing.links, {a01_2: "A01"})
        self.assertEqual([s.name for s in pairing.orphans], ["A01_2"])

    def test_build_writes_the_binding_so_order_no_longer_matters(self):
        self.store.define_shot("Step_1", 1, 10)
        mani = ShotManifest(self.store, match="name_then_order")
        mani.update([self._step("IN_01")], remove_missing=False)
        self.assertEqual(self.store.shot_by_name("Step_1").metadata["step"], "IN_01")
        # Name-only matching now finds it through the binding.
        pairing = ShotManifest(self.store).pair([self._step("IN_01")])
        self.assertEqual(pairing.how["IN_01"], "binding")


class TestStepRename(_EngineCase):
    """A step's ``name`` renames its shot; its ID stays the shot's binding,
    so a rebuild patches the same shot instead of creating a second one."""

    def test_a_new_step_builds_under_its_name_bound_to_its_id(self):
        step = self._step("A01", "door")
        step.name = "Door_Open"
        actions = ShotManifest(self.store).update([step], initial_shot_length=50)
        self.assertEqual(actions, {"A01": "created"})
        shot = self.store.shot_by_name("Door_Open")
        self.assertEqual(shot.metadata["step"], "A01")
        self.assertIsNone(self.store.shot_by_name("A01"))

    def test_renaming_a_built_step_renames_its_shot_in_place(self):
        """Bug: the step ID was both the binding and the shot name, and a
        patch never wrote the name -- a renamed built step built a second
        shot and left the first as "not in doc".
        Fixed: 2026-10-06"""
        mani = ShotManifest(self.store)
        mani.update([self._step("A01", "door")], initial_shot_length=50)
        shot_id = self.store.shot_by_name("A01").shot_id
        step = self._step("A01", "door")
        step.name = "Door_Open"
        actions = mani.update([step], remove_missing=False)
        self.assertEqual(actions, {"A01": "patched"})
        self.assertEqual([s.name for s in self.store.shots], ["Door_Open"])
        self.assertEqual(self.store.shot_by_name("Door_Open").shot_id, shot_id)
        # The sheet's step again (the edit dropped): the build named the
        # shot, so it names it back, still paired through the binding.
        actions = mani.update([self._step("A01", "door")], remove_missing=False)
        self.assertEqual(actions["A01"], "patched")
        self.assertEqual([s.name for s in self.store.shots], ["A01"])
        self.assertEqual(self.store.shot_by_name("A01").shot_id, shot_id)

    def test_a_step_without_a_name_keeps_a_sequencer_rename(self):
        mani = ShotManifest(self.store)
        mani.update([self._step("A01", "door")], initial_shot_length=50)
        shot = self.store.shot_by_name("A01")
        self.store.update_shot(shot.shot_id, name="Hand_Named")
        actions = mani.update([self._step("A01", "door")], remove_missing=False)
        # A hand rename is an edit: the whole shot is kept (TestEditedShotsAreKept).
        self.assertEqual(actions["A01"], "kept")
        self.assertEqual([s.name for s in self.store.shots], ["Hand_Named"])

    def test_a_hand_rename_after_an_edit_rename_is_kept_once_the_edit_goes(self):
        mani = ShotManifest(self.store)
        step = self._step("A01", "door")
        step.name = "Door_Open"
        mani.update([step], initial_shot_length=50)
        shot = self.store.shot_by_name("Door_Open")
        self.assertEqual(shot.metadata[ShotManifest.BUILD_NAME], "Door_Open")
        self.store.update_shot(shot.shot_id, name="Hand_Named")
        mani.update([self._step("A01", "door")], remove_missing=False)
        self.assertEqual([s.name for s in self.store.shots], ["Hand_Named"])

    def test_a_name_another_shot_holds_is_refused_and_logged(self):
        mani = ShotManifest(self.store)
        mani.update([self._step("A01", "door")], initial_shot_length=50)
        self.store.define_shot("Taken", 500, 600)
        step = self._step("A01", "door")
        step.name = "taken"  # the store compares names ignoring case
        with self.assertLogs(level="WARNING") as logs:
            mani.update([step], remove_missing=False)
        self.assertIn("not renamed", "\n".join(logs.output))
        self.assertIsNotNone(self.store.shot_by_name("A01"))

    def test_from_shots_carries_a_renamed_shots_name(self):
        self.store.define_shot("Door_Open", 1, 10, metadata={"step": "A01"})
        (step,), _ = BuilderStep.from_shots(self.store.sorted_shots())
        self.assertEqual((step.step_id, step.name), ("A01", "Door_Open"))
        self.assertEqual(step.shot_name, "Door_Open")


class TestReconcileMembership(_EngineCase):
    """Membership is compared through ``ShotStore.names_member``: a doc name
    (a path's tail read whole) against a stored long, namespaced node."""

    def test_an_object_dropped_from_the_doc_leaves_the_shot(self):
        """Bug: the planner diffed stored long paths against the doc's short
        names, so every doc member read as scene-discovered and an animated
        object removed from the doc stayed in the shot.
        Fixed: 2026-10-01"""
        mani = ShotManifest(self.store)
        mani._filter_to_animated = lambda names, s, e: list(names)  # all animated
        mani.update([self._step("A01", "door", "lid")], initial_shot_length=50)
        self.assertEqual(
            sorted(self.store.shot_by_name("A01").objects),
            ["|set|AC:door", "|set|AC:lid"],
        )
        mani.update([self._step("A01", "door")], remove_missing=False)
        self.assertEqual(self.store.shot_by_name("A01").objects, ["|set|AC:door"])

    def test_a_member_removed_from_its_shot_is_restored_by_build(self):
        mani = ShotManifest(self.store)
        mani.update([self._step("A01", "door")], initial_shot_length=50)
        shot = self.store.shot_by_name("A01")
        self.store.update_shot(shot.shot_id, objects=[])
        (status,) = mani.assess([self._step("A01", "door")])
        self.assertEqual(status.objects[0].status, "not_in_shot")
        # A hand's edit, so only a build asked to rebuild edited shots puts
        # it back (by default the shot is kept: TestEditedShotsAreKept).
        mani = ShotManifest(self.store, rebuild_edited=True)
        actions = mani.update([self._step("A01", "door")], remove_missing=False)
        self.assertEqual(actions["A01"], "patched")
        self.assertEqual(self.store.shot_by_name("A01").objects, ["|set|AC:door"])

    def test_build_keeps_metadata_it_does_not_own(self):
        mani = ShotManifest(self.store)
        mani.update([self._step("A01", "door")], initial_shot_length=50)
        shot = self.store.shot_by_name("A01")
        meta = dict(shot.metadata, object_status={"door": "valid"}, review="ok")
        self.store.update_shot(shot.shot_id, metadata=meta)
        mani.update([self._step("A01", "door", "lid")], remove_missing=False)
        kept = self.store.shot_by_name("A01").metadata
        self.assertEqual(kept["review"], "ok")
        self.assertEqual(kept["object_status"], {"door": "valid"})
        self.assertEqual([e["name"] for e in kept["csv_objects"]], ["door", "lid"])

    def test_a_cleared_pass_through_cell_leaves_the_shot(self):
        """Bug: a pass-through column (a hint, a priority) cleared in the doc
        kept its old value on the shot -- only a cell that said something was
        recorded, so nothing told the build the key was the doc's to drop.  A
        doc that says nothing of the column (no sheet) keeps it.
        Fixed: 2026-10-04
        """
        columns = ColumnMap(metadata_pass={"hint": ("Hint",)})

        def parse(hint):
            path = _write_csv(
                "Step,Step Contents,Asset Names,Hint\n"
                f"A01.),Door fades in,door,{hint}\n"
            )
            try:
                return ManifestModel.parse_csv(path, columns=columns)
            finally:
                os.remove(path)

        mani = ShotManifest(self.store)
        mani.update(parse("Press the red button"), initial_shot_length=50)
        shot = self.store.shot_by_name("A01")
        self.assertEqual(shot.metadata["hint"], "Press the red button")

        steps, ranges = BuilderStep.from_shots(self.store.sorted_shots())
        mani.update(steps, ranges=ranges, remove_missing=False)
        self.assertEqual(shot.metadata["hint"], "Press the red button")

        mani.update(parse(""), remove_missing=False)
        self.assertNotIn("hint", self.store.shot_by_name("A01").metadata)

    def test_a_shot_the_manifest_never_built_keeps_its_members(self):
        """Bug: a shot with no ``csv_objects`` (made in the Shot Sequencer, or
        split off there) read its members as the doc objects of a previous
        build, so a build pairing a step with it -- by name, by a binding the
        panel wrote, or as a step adopted from it -- dropped every member the
        step did not list, animated or not: they now count as what they are,
        members the scene found (kept while they animate, as any).
        Fixed: 2026-10-10"""
        members = ["|set|AC:keys", "|set|AC:lock"]
        self.store.define_shot("A01", 1, 50, objects=list(members))
        self.store.define_shot("A01_2", 60, 90, objects=list(members))
        mani = ShotManifest(self.store)
        mani._filter_to_animated = lambda names, s, e: list(names)  # all move
        mani.update(
            [self._step("A01", "door"), self._step("A01_2")], remove_missing=False
        )
        self.assertEqual(
            sorted(self.store.shot_by_name("A01").objects),
            ["|set|AC:door", *members],
        )
        self.assertEqual(sorted(self.store.shot_by_name("A01_2").objects), members)
        # Built now: a doc object dropped later still leaves, the rest stay.
        mani.update([self._step("A01")], remove_missing=False)
        self.assertEqual(sorted(self.store.shot_by_name("A01").objects), members)

    def test_two_objects_sharing_a_leaf_are_two_members_across_a_rebuild(self):
        """Bug: the planner compared a doc name with the members by its leaf,
        so a shot holding ``PANEL|LATCH`` read a sheet adding ``DOOR|LATCH``
        as nothing new: the sync skipped it, its node never joined the shot,
        and Assess called it valid (only a fresh create kept both).
        Fixed: 2026-11-26"""

        class _Paths(PureShotsHost):
            NODES = ("|rig|PANEL|LATCH", "|rig|DOOR|LATCH")

            def member_nodes(self, name):  # a path's tail: the nodes whose paths end with it
                tail = [n for n in name.split("|") if n]
                return [p for p in self.NODES if p.split("|")[-len(tail):] == tail]

        store = ShotStore(host=_Paths())
        mani = ShotManifest(store)
        mani._filter_to_animated = lambda names, s, e: list(names)  # all animated
        both = self._step("A01", "PANEL|LATCH", "DOOR|LATCH")
        mani.update([self._step("A01", "PANEL|LATCH")], initial_shot_length=50)
        self.assertEqual(store.shot_by_name("A01").objects, ["|rig|PANEL|LATCH"])
        (status,) = mani.assess([both])
        self.assertEqual([o.status for o in status.objects], ["valid", "not_in_shot"])
        self.assertEqual(mani.update([both], remove_missing=False), {"A01": "patched"})
        self.assertEqual(store.shot_by_name("A01").objects, ["|rig|DOOR|LATCH", "|rig|PANEL|LATCH"])
        self.assertEqual(mani.update([both], remove_missing=False), {"A01": "skipped"}, "nothing new the second time")
        mani.update([self._step("A01", "DOOR|LATCH")], remove_missing=False)
        self.assertEqual(store.shot_by_name("A01").objects, ["|rig|DOOR|LATCH"], "the one the sheet dropped leaves")

    def test_an_orphan_shot_is_kept_unless_removal_is_asked_for(self):
        mani = ShotManifest(self.store)
        mani.update([self._step("A01"), self._step("A02")], initial_shot_length=10)
        mani.update([self._step("A01")], remove_missing=False)
        self.assertIsNotNone(self.store.shot_by_name("A02"))
        self.assertEqual(
            [s.name for s in mani.pair([self._step("A01")]).orphans], ["A02"]
        )


class TestTheExportShipsWhatTheManifestAuthors(_EngineCase):
    """A shot's ``animation`` record entry ships the metadata the manifest
    writes from what its user wrote in the sheet (its section, objects,
    behaviors, voice, step and the template's own columns), and nothing the
    tools keep on the shot for themselves (Assess's ``object_status``, the
    split's ``step_link``, the build's records): those stay in the scene's
    ``shot_store``.  The user's call (2026-11-05): "Manifest only: we should
    be exporting only what is useful.  Everything else stays in the scene
    data only." """

    COLUMNS = ColumnMap(metadata_pass={"hint": ("Hint",)})
    BOOKKEEPING = {
        "object_status": {"door": "valid"},
        ShotStore.STEP_LINK: "A00",
        ShotStore.SYSTEM_MEMBERS: ["door"],
    }

    def _steps(self, voice="Open the door."):
        path = _write_csv(
            "SECTION A: DOORS\n"
            "Step,Step Contents,Asset Names,Voice Support,Hint\n"
            f"A01.),Door fades in,door,{voice},Press the red button\n"
        )
        try:
            return ManifestModel.parse_csv(path, columns=self.COLUMNS)
        finally:
            os.remove(path)

    def _entries(self):
        view = self.store.to_export_view()
        return {e["clip"]: e for e in view["animation"]["shots"]}

    def _built(self):
        ShotManifest(self.store).update(self._steps(), initial_shot_length=50)
        shot = self.store.shot_by_name("A01")
        self.store.update_shot(
            shot.shot_id, metadata={**shot.metadata, **self.BOOKKEEPING}
        )
        return shot

    def test_a_built_shot_ships_the_manifests_keys_alone(self):
        shot = self._built()
        shipped = self._entries()["A01"]["metadata"]
        self.assertEqual(
            set(shipped),
            {
                "section",
                "section_title",
                "csv_objects",
                "behaviors",
                "voice_text",
                "step",
                "hint",
            },
        )
        self.assertEqual(shipped["hint"], "Press the red button")
        self.assertEqual(shipped["section_title"], "DOORS")
        # The rest stays in the scene's own record, whole.
        held = self.store.to_dict()["shots"][0]["metadata"]
        for key in (*self.BOOKKEEPING, ShotManifest.BUILT, ShotManifest.BUILD_NAME):
            self.assertIn(key, held)
            self.assertNotIn(key, shipped)
        self.assertEqual(held, shot.metadata)

    def test_a_shot_with_only_bookkeeping_ships_no_metadata(self):
        self._built()
        self.store.define_shot("Extra", 60, 90, metadata=dict(self.BOOKKEEPING))
        self.assertEqual(self._entries()["Extra"], {"clip": "Extra"})

    def test_a_shot_the_build_keeps_still_ships_its_keys(self):
        """A shot edited by hand is left as it is by a rebuild, and still
        ships what the manifest wrote on it: the keys are the manifest's,
        not each shot's last build's."""
        shot = self._built()
        self.store.update_shot(shot.shot_id, description="the artist's note")
        actions = ShotManifest(self.store).update(self._steps(), remove_missing=False)
        self.assertEqual(actions, {"A01": "kept"})
        self.assertEqual(self._entries()["A01"]["metadata"]["step"], "A01")

    def test_the_keys_travel_with_the_saved_scene(self):
        self._built()
        reopened = ShotStore.from_dict(self.store.to_dict())
        self.assertEqual(reopened.shipped_metadata, self.store.shipped_metadata)
        self.assertIn("hint", reopened.shipped_metadata)

    def test_the_manifests_keys_are_what_a_step_writes(self):
        """:attr:`ShotManifest.MANIFEST_METADATA` is derived from the one
        table a step's metadata is written from, never kept beside it."""
        (step,) = self._steps()
        self.assertEqual(
            set(ShotManifest._step_metadata(step)), ShotManifest.MANIFEST_METADATA
        )
        (silent,) = self._steps(voice="N/A")
        self.assertEqual(
            set(ShotManifest._step_metadata(silent)),
            ShotManifest.MANIFEST_METADATA - {"voice_text"},
        )

    def test_the_manifests_fields_are_the_stores_one_declaration(self):
        """The names a step writes (:attr:`ShotManifest._STEP_FIELDS`) are the
        ones the store declares (``ShotStore.MANIFEST_FIELDS``: what a scene
        saved before ``shipped_metadata`` ships), never a second list."""
        self.assertEqual(frozenset(ShotManifest._STEP_FIELDS), ShotStore.MANIFEST_FIELDS)
        self.assertIs(ShotManifest.MANIFEST_METADATA, ShotStore.MANIFEST_FIELDS)


class TestReconcileStatuses(_EngineCase):
    """Assess reports; it never writes the store."""

    def _built(self, step):
        mani = ShotManifest(self.store, disabled_behaviors=())  # every behavior keyed
        mani.update([step], initial_shot_length=50)
        return mani

    def test_missing_and_ambiguous_objects(self):
        step = self._step("A01", "door", "lid")
        mani = self._built(step)
        self.store.host.missing, self.store.host.ambiguous = {"door"}, {"lid"}
        (st,) = mani.assess([step])
        self.assertEqual(
            [o.status for o in st.objects], ["missing_object", "ambiguous_object"]
        )
        self.assertEqual(st.status, "missing_object")

    def test_what_the_sheets_check_found_is_the_steps_status_and_waits_on_a_person(self):
        """The reader checked each row and kept its issues on it, and the assessment read none: a
        row with an unknown word, an empty needed cell or a setting it lacks showed as built and
        in order. Added: 2026-11-25"""
        step = self._step("A01", "door")
        step.issues = [("To", "the sheet has no step Z99: write a step's ID")]
        step.rows = [
            {"issues": [("Clip", "use needs its Clip: write the take it plays")], "warnings": []},
            {"issues": [], "warnings": [("Clip", "the take OPEN is not named for its step: A01 says which step it is")]},
        ]
        self.assertEqual(step.sheet_issues(), [
            ("sheet", "(To): the sheet has no step Z99: write a step's ID"),
            ("sheet", "row 1 (Clip): use needs its Clip: write the take it plays"),
            ("sheet_warning", "row 2 (Clip): the take OPEN is not named for its step: A01 says which step it is"),
        ])
        mani = self._built(step)
        (st,) = mani.assess([step])
        self.assertEqual(st.sheet, step.sheet_issues())
        self.assertEqual((st.status, st.label, st.quiet), ("sheet", "2 sheet issues", False))
        self.assertEqual(StepStatus.plan([st])["needs_you"], 1, "no Build fixes a sheet")
        self.assertFalse(st.needs_build, "its shot is built and in order")
        # A warning alone leaves the step's status its own, and is still something to report.
        step.issues, step.rows = [], step.rows[1:]
        (st,) = mani.assess([step])
        self.assertEqual((st.status, st.quiet, [s for s, _ in st.sheet]), ("valid", False, ["sheet_warning"]))
        # A step not built yet with a sheet error says the error: Build makes its shot, the sheet's fix is a person's.
        fresh = self._step("A02", "lid")
        fresh.rows = [{"issues": [("Action", "'hilight' is no action of this template: write one of highlight")]}]
        (st,) = ShotManifest(self.store).assess([fresh])
        self.assertEqual((st.status, st.label, st.needs_build), ("sheet", "sheet issue", True))
        for status in ("sheet", "sheet_warning"):
            self.assertIn(status, StepStatus.LABELS)
            self.assertIn(status, StepStatus.HELP)
            self.assertIn(status, SHOT_STATUS_LEVELS)

    def test_a_step_in_cells_that_names_no_object_is_in_order(self):
        """A question or a line of narration in a sheet written in cells read "no objects" on its
        row: most of a real sheet's rows, hiding the ones with something to say. Added: 2026-11-25"""
        asked = BuilderStep("B04", "B", "Did it light?", "", [], celled=True)
        prose = BuilderStep("B05", "B", "Did it light?", "", [])
        mani = ShotManifest(self.store)
        mani.update([asked, prose], initial_shot_length=50)
        celled, sentence = mani.assess([asked, prose])
        self.assertEqual((celled.status, celled.quiet), ("valid", True))
        self.assertEqual(sentence.status, "no_objects", "a sheet of sentences may have left them out")

    def test_unknown_behavior_is_reported_not_verified(self):
        step = self._step("A01", "door", behaviors={"door": ["wobble"]})
        mani = self._built(step)
        mani._verify_behavior = lambda *a, **k: self.fail("verified an unknown")
        (st,) = mani.assess([step])
        self.assertEqual(st.objects[0].status, "unknown_behavior")
        self.assertEqual(st.objects[0].unknown_behaviors, ["wobble"], "named, per behavior")

    def test_a_broken_behavior_over_animator_keys_is_a_conflict(self):
        step = self._step("A01", "door", behaviors={"door": ["show"]})
        mani = self._built(step)
        mani._verify_behavior = lambda *a, **k: False
        mani._key_samples = lambda obj, behavior, s, e: [("door_opacity", 5.0)]
        (st,) = mani.assess([step])
        self.assertEqual(st.objects[0].status, "behavior_conflict")
        self.assertEqual(st.objects[0].conflict_behaviors, ["show"], "named, per behavior")
        self.assertEqual(
            [s for _b, s, _w in StepStatus.behavior_states(st, "door", ["show"])], ["conflict"]
        )
        # The same keys, claimed by the ledger, are the manifest's own: fixable.
        self.store.edit_ledger.record_authored(
            "door_opacity",
            5.0,
            self.store.shot_by_name("A01").shot_id,
            "show",
            "door",
        )
        (st,) = mani.assess([step])
        self.assertEqual(st.objects[0].status, "missing_behavior")

    def test_a_step_listing_nothing_says_so(self):
        step = self._step("A01")
        mani = self._built(step)
        (st,) = mani.assess([step])
        self.assertEqual(st.status, "no_objects")

    def test_a_clip_a_build_will_key_is_the_builds_and_one_turned_off_says_nothing(self):
        """Before its first Build a narrated sheet read "13 need you": each step's clip is not in
        the scene yet, and that read as an object a person must find, though its file is there
        and Build keys it. Added: 2026-11-25"""
        import tempfile

        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self.addCleanup(os.remove, wav)
        step = BuilderStep("A01", "A", "t", "", [
            BuilderObject("A01_Welcome", behaviors=["set_clip"], kind="audio", source_path=wav),
            BuilderObject("A01_Lost", behaviors=["set_clip"], kind="audio", source_path=wav + ".gone"),
        ])
        mani = self._built(self._step("A00", "door"))
        absent = lambda name: False  # noqa: E731 -- no clip of the name is in the scene
        (st,) = mani.assess([step], audio_exists_fn=absent)
        self.assertEqual([(o.name, o.status, o.broken_behaviors) for o in st.objects], [
            ("A01_Welcome", "missing_behavior", ["set_clip"]),   # its file is there: Build keys it
            ("A01_Lost", "missing_object", []),                  # no file: a person's (a line not narrated yet)
        ])
        off = ShotManifest(self.store, disabled_behaviors=["set_clip"])
        (st,) = off.assess([step], audio_exists_fn=absent)
        self.assertEqual([o.status for o in st.objects], ["valid", "valid"], "Off: not reported")

    def test_what_moves_inside_a_listed_object_is_that_object_never_unlisted(self):
        """A step lists a door's group and its take keys the door's own node inside it: 16 of a
        real scene's 61 steps read "not listed" for nodes of the objects they list. A member
        inside a listed object is part of it; one beside it is still said. Added: 2026-11-25"""
        step = self._step("A01", "door")
        mani = self._built(step)
        mani._discover_scene_objects = lambda s, e, known: [
            "|set|AC:door|AC:door_loc", "|set|AC:door|hinge|pin", "|set|AC:doorstop", "|set|AC:camera"]
        (st,) = mani.assess([step])
        self.assertEqual(st.additional_objects, ["|set|AC:doorstop", "|set|AC:camera"])

    def test_assess_leaves_the_store_untouched(self):
        step = self._step("A01", "door")
        mani = self._built(step)
        mani._discover_scene_objects = lambda s, e, known: ["|set|AC:camera"]
        before = self.store.to_dict()
        (st,) = mani.assess([step])
        self.assertEqual(st.additional_objects, ["|set|AC:camera"])
        self.assertEqual(self.store.to_dict(), before)


class TestBehaviorOwnership(_EngineCase):
    """Behavior keys go in the ledger's ``authored`` register: Build records
    what the applier reports writing, and re-applying first takes out only
    the manifest's own keys."""

    def test_sync_records_what_the_applier_wrote(self):
        mani = ShotManifest(self.store)
        step = self._step("A01", "door", behaviors={"door": ["show"]})
        shot_holder = {}

        def apply_behaviors():
            shot = self.store.shot_by_name("A01")
            shot_holder["id"] = shot.shot_id
            return {
                "applied": [
                    {
                        "object": "door",
                        "behavior": "show",
                        "shot": "A01",
                        "shot_id": shot.shot_id,
                        "keys": [("door_opacity", 1.0), ("door_opacity", 16.0)],
                    }
                ],
                "skipped": [],
            }

        mani.apply_behaviors = apply_behaviors
        mani.sync([step], initial_shot_length=50)
        self.assertEqual(
            self.store.edit_ledger.authored(owner=shot_holder["id"]),
            [("door_opacity", 1.0), ("door_opacity", 16.0)],
        )

    def test_release_takes_out_only_the_manifests_own_keys(self):
        mani = ShotManifest(self.store)
        led = self.store.edit_ledger
        led.record_authored("door_opacity", 1.0, 7, "show", "door")
        led.record_authored("door_opacity", 90.0, 7, "hide", "door")
        deleted = []
        mani._delete_keys = lambda curve, times: deleted.append((curve, list(times)))
        self.assertEqual(mani.release_authored(7, "door", "show"), 1)
        self.assertEqual(deleted, [("door_opacity", [1.0])])
        self.assertEqual(led.authored(owner=7), [("door_opacity", 90.0)])


class TestEditedShotsAreKept(_EngineCase):
    """A rebuild leaves a shot edited by hand since its last build as it is,
    unless Rebuild Edited Shots is on (2026-10-07: "make sure that the
    manifest does not override edited shots on re-build by default.  this
    can be a separate global option").  A build records each shot as it left
    it; a shot that no longer matches that record was edited."""

    def _built(self, *objs, desc="the doc's text"):
        step = self._step("A01", *objs)
        step.description = desc
        ShotManifest(self.store).update([step], initial_shot_length=50)
        return step, self.store.shot_by_name("A01")

    def _rebuild(self, step, **kwargs):
        return ShotManifest(self.store, **kwargs).update([step], remove_missing=False)

    def test_a_build_records_the_shot_as_it_left_it(self):
        _step, shot = self._built("door")
        self.assertFalse(ShotManifest(self.store).edited_by_hand(shot))
        self.assertIn(ShotManifest.BUILT, shot.metadata)

    def test_an_edited_description_survives_a_rebuild(self):
        step, shot = self._built("door")
        self.store.update_shot(shot.shot_id, description="the artist's note")
        self.assertEqual(self._rebuild(step), {"A01": "kept"})
        self.assertEqual(
            self.store.shot_by_name("A01").description, "the artist's note"
        )

    def test_a_member_removed_by_hand_stays_removed(self):
        step, shot = self._built("door", "window")
        door = [n for n in shot.objects if n.endswith("door")]
        self.store.update_shot(shot.shot_id, objects=door)
        self.assertEqual(self._rebuild(step), {"A01": "kept"})
        self.assertEqual(self.store.shot_by_name("A01").objects, door)

    def test_a_retimed_shot_keeps_its_length_and_misses_doc_changes(self):
        step, shot = self._built("door")
        self.store.update_shot(shot.shot_id, end=shot.end + 20)
        length = self.store.shot_by_name("A01").duration
        step.objects.append(BuilderObject("lamp"))
        self.assertEqual(self._rebuild(step), {"A01": "kept"})
        shot = self.store.shot_by_name("A01")
        self.assertEqual(shot.duration, length)
        self.assertFalse([n for n in shot.objects if n.endswith("lamp")])

    def test_a_shot_a_ripple_moved_is_not_edited(self):
        step, shot = self._built("door")
        self.store.update_shot(shot.shot_id, start=shot.start + 30, end=shot.end + 30)
        self.assertEqual(self._rebuild(step), {"A01": "skipped"})

    def test_a_hand_rename_is_an_edit(self):
        step, shot = self._built("door")
        self.store.update_shot(shot.shot_id, name="Hand_Named")
        step.name = "From_The_Doc"
        self.assertEqual(self._rebuild(step), {"A01": "kept"})
        self.assertEqual([s.name for s in self.store.shots], ["Hand_Named"])

    def test_an_unedited_shot_still_takes_the_docs_changes(self):
        step, _shot = self._built("door")
        step.objects.append(BuilderObject("lamp"))
        self.assertEqual(self._rebuild(step), {"A01": "patched"})

    def test_rebuild_edited_shots_overrides_and_records_again(self):
        step, shot = self._built("door", "window")
        self.store.update_shot(shot.shot_id, description="the artist's note")
        self.assertEqual(self._rebuild(step, rebuild_edited=True)["A01"], "skipped")
        shot = self.store.shot_by_name("A01")
        self.assertEqual(shot.description, "the doc's text")
        self.assertFalse(ShotManifest(self.store).edited_by_hand(shot))

    def test_rebuild_edited_shots_names_a_hand_renamed_shot_back(self):
        """Bug: Rebuild Edited Shots rebuilt a hand-renamed shot's content but
        kept the hand's name, so it read as edited again the moment the
        option went off -- and was kept by every later build.
        Fixed: 2026-10-10"""
        step, shot = self._built("door")
        self.store.update_shot(shot.shot_id, name="Hand_Named")
        self.assertEqual(self._rebuild(step, rebuild_edited=True)["A01"], "patched")
        self.assertEqual([s.name for s in self.store.shots], ["A01"])
        shot = self.store.shot_by_name("A01")
        self.assertFalse(ShotManifest(self.store).edited_by_hand(shot))

    def test_a_bound_shot_keeps_its_own_name_under_rebuild_edited(self):
        # "Pair with Step": the shot is the step's under the name it has --
        # no build named it, so there is no hand rename to undo.
        self.store.define_shot("Tail", 1, 50)
        mani = ShotManifest(self.store, rebuild_edited=True)
        mani.bind(self.store.shot_by_name("Tail"), "A01")
        mani.update([self._step("A01", "door")], remove_missing=False)
        self.assertEqual([s.name for s in self.store.shots], ["Tail"])

    def test_a_shot_built_before_the_record_existed_is_not_edited(self):
        self.store.define_shot("A01", 1, 50, metadata={"step": "A01"})
        shot = self.store.shot_by_name("A01")
        self.assertFalse(ShotManifest(self.store).edited_by_hand(shot))

    def test_assess_marks_an_edited_shot_and_build_leaves_it(self):
        step, shot = self._built("door")
        self.store.update_shot(shot.shot_id, description="the artist's note")
        step.objects.append(BuilderObject("lamp"))
        (status,) = ShotManifest(self.store).assess([step])
        self.assertTrue(status.edited)
        self.assertEqual(status.status, "edited")
        self.assertFalse(status.needs_build, "a build would not touch it")
        (status,) = ShotManifest(self.store, rebuild_edited=True).assess([step])
        self.assertFalse(status.edited)
        self.assertTrue(status.needs_build)

    # -- the shot system's own member additions (BACKLOG 2026-10-10) ---------
    # The sequencer fills an empty shot's members when it shows it, and adds
    # an object keyed in the active shot: read as a hand edit, every later
    # Build kept that shot, doc changes and all.  Maintainer 2026-10-11: tag
    # them so the check skips them; a hand's add or remove still counts.

    def test_a_member_the_system_added_is_no_edit(self):
        step, shot = self._built("door")
        self.store.add_system_members(shot.shot_id, ["keyed_cube"])
        self.assertFalse(ShotManifest(self.store).edited_by_hand(shot))
        step.objects.append(BuilderObject("lamp"))
        self.assertEqual(self._rebuild(step), {"A01": "patched"})

    def test_a_step_with_missing_objects_filled_by_the_sequencer_is_no_edit(self):
        """The doc's objects are not in the scene; the sequencer fills the
        shot with what is keyed in it, keeping the doc's names inert."""
        step, shot = self._built("ghost_a", "ghost_b")
        self.store.add_system_members(shot.shot_id, ["found_1", "found_2"])
        shot = self.store.shot_by_name("A01")
        self.assertEqual(len(shot.objects), 4, shot.objects)
        self.assertFalse(ShotManifest(self.store).edited_by_hand(shot))

    def test_a_member_a_hand_adds_beside_system_ones_is_an_edit(self):
        step, shot = self._built("door")
        self.store.add_system_members(shot.shot_id, ["keyed_cube"])
        shot = self.store.shot_by_name("A01")
        self.store.update_shot(shot.shot_id, objects=shot.objects + ["hand_cube"])
        self.assertTrue(ShotManifest(self.store).edited_by_hand(shot))
        self.assertEqual(self._rebuild(step), {"A01": "kept"})

    def test_a_built_member_a_hand_removes_is_an_edit_with_system_ones_there(self):
        step, shot = self._built("door", "window")
        self.store.add_system_members(shot.shot_id, ["keyed_cube"])
        shot = self.store.shot_by_name("A01")
        kept = [n for n in shot.objects if not n.endswith("window")]
        self.store.update_shot(shot.shot_id, objects=kept)
        self.assertTrue(ShotManifest(self.store).edited_by_hand(shot))

    def test_a_system_member_a_hand_removes_and_adds_back_is_the_hands(self):
        step, shot = self._built("door")
        self.store.add_system_members(shot.shot_id, ["keyed_cube"])
        shot = self.store.shot_by_name("A01")
        door = [n for n in shot.objects if n.endswith("door")]
        self.store.update_shot(shot.shot_id, objects=door)
        self.assertFalse(ShotManifest(self.store).edited_by_hand(shot), "back as built")
        self.store.update_shot(shot.shot_id, objects=door + ["keyed_cube"])
        self.assertTrue(ShotManifest(self.store).edited_by_hand(shot))

    def test_a_kept_shot_gets_no_behavior_pass(self):
        step, shot = self._built("door")
        self.store.define_shot("B01", 200, 250, metadata={"step": "B01"})
        self.store.update_shot(shot.shot_id, description="the artist's note")
        names = [s.name for s in ShotManifest(self.store)._behavior_shots()]
        self.assertEqual(names, ["B01"])
        self.store.edit_ledger.record_authored(
            "door_op", 2.0, shot.shot_id, "show", "door"
        )
        self.assertEqual(ShotManifest(self.store)._dropped_pairs([step]), [])


class TestAStepWithNoShot(_EngineCase):
    """A step that plays no animation gets no shot: an interactive task's
    "Auto Advance" or "Selecting 'Next'" step is a page of the tool that runs
    the steps, not a take
    (the user, 2026-11-22: "we need a way to mark shots that have no
    animation ... a shot isn't built for that row"; on a production sheet 37 of
    57 steps, each projected and counted by Build).  The sheet says it: under
    the template's Shots option "Steps that play a take", a step none of whose
    rows names a clip (Animation Clip Names) has none; a Local Edit marks one
    either way.  Added: 2026-11-22.  The Interactive template's own choice is
    every step since 2026-11-25 (a shot is its step's and holds its narration;
    a step that only speaks ships no take), so these ask for the rule."""

    SHEET = (
        "Step,Step Contents,Asset Names,Animation Clip Names\n"
        "A01,Selecting the door triggers Door Open Animation,DOOR,A01\n"
        "A02,Auto Advance,,\n"
        "A03,Highlight the lever,LEVER,\n"
        ",Selecting the lever triggers Lever Animation,LEVER,A03_2\n"
    )

    def _read(self, name, text=None, **options):
        path = _write_csv(text or self.SHEET)
        self.addCleanup(os.remove, path)
        return mapping_mod.Mapping.resolve(path, name=name, options=options or None)

    def test_the_sheet_says_which_steps_play_animation(self):
        steps = self._read("interactive", shots="clip")
        self.assertEqual({s.step_id: s.animated for s in steps}, {"A01": True, "A02": False, "A03": True})

    def test_every_step_is_a_shot_where_the_template_says_so(self):
        """A video's step plays its narration whether or not anything moves:
        the Linear and Default templates give every step a shot, unless their
        Shots option says otherwise."""
        for name in ("linear", "default"):
            with self.subTest(template=name):
                self.assertTrue(all(s.animated for s in self._read(name)))
        steps = self._read("default", shots="clip")
        self.assertEqual([s.step_id for s in steps if not s.animated], ["A02"])

    def test_a_sheet_with_no_clip_column_names_no_clip(self):
        """A named switch does its whole job: no clip column, no step names a
        clip, so none gets a shot -- and the reading says why."""
        text = "Step,Step Contents,Asset Names\nA01,Door Open Animation,DOOR\n"
        steps = self._read("interactive", text, shots="clip")
        self.assertEqual([s.animated for s in steps], [False])

    def test_each_step_reads_its_own_sections_header(self):
        """A sheet repeats its header per section, with different columns: a
        step's clip is read under the header above it, never the last one."""
        text = (
            "Step,Step Contents,Asset Names,Animation Clip Names\n"
            "A01,Open the door,DOOR,A01\n"
            "Step,Step Contents,Asset Names\n"
            "B01,Read the placard,PLACARD\n"
        )
        steps = self._read("interactive", text, shots="clip")
        self.assertEqual({s.step_id: s.animated for s in steps}, {"A01": True, "B01": False})

    def test_a_clip_cell_the_template_excludes_names_no_clip(self):
        """A cell the template's ``exclude_values`` treats as empty ("N/A",
        "-") names no clip, so its step plays nothing."""
        text = (
            "Step,Step Contents,Asset Names,Animation Clip Names\n"
            "A01,Open the door,DOOR,N/A\n"
            "A02,Close the door,DOOR,-\n"
            "A03,Lock the door,DOOR,A03\n"
        )
        steps = self._read("interactive", text, shots="clip")
        self.assertEqual({s.step_id: s.animated for s in steps}, {"A01": False, "A02": False, "A03": True})

    def test_build_makes_no_shot_for_it_and_assess_says_so(self):
        steps = self._read("interactive", shots="clip")
        mani = ShotManifest(self.store)
        mani.update(steps, initial_shot_length=50)
        self.assertEqual(sorted(s.name for s in self.store.shots), ["A01", "A03"])
        by_id = {r.step_id: r for r in mani.assess(steps)}
        self.assertEqual(by_id["A02"].status, "no_shot")
        self.assertFalse(by_id["A02"].needs_build)
        self.assertEqual(by_id["A02"].label, "", "nothing to say: the frames say no shot")
        plan = StepStatus.plan(list(by_id.values()))
        self.assertEqual((plan["create"], plan["no_shot"]), (0, 1))
        self.assertIn("1 with no shot", StepStatus.plan_text(plan))

    def test_it_takes_no_place_on_the_timeline(self):
        steps = self._read("interactive", shots="clip")
        ShotManifest(self.store).update(steps, initial_shot_length=50)
        without = ShotStore(host=_ScenicHost())
        ShotManifest(without).update([s for s in steps if s.step_id != "A02"], initial_shot_length=50)
        self.assertEqual(
            [(s.name, s.start, s.end) for s in self.store.sorted_shots()],
            [(s.name, s.start, s.end) for s in without.sorted_shots()],
            "A02 left no hole: laid out as a sheet without it",
        )
        resolved = rr.RangeResolver.resolve_ranges(
            steps=steps, user_ranges={}, gap_starts=[], gap_end_map={}, gap=10.0,
            use_selected_keys=False, last_resolved=[], default_duration=50,
        )
        self.assertEqual([r[0] for r in resolved], ["A01", "A03"])

    def test_an_insert_skips_it(self):
        steps = self._read("interactive", shots="clip")
        ShotManifest(self.store).update([steps[0]], initial_shot_length=50)
        plans = ShotManifest(self.store).plan_insertions(steps, initial_shot_length=50)
        self.assertEqual([p.step.step_id for p in plans], ["A03"])

    def test_a_shot_it_already_has_stays_its_shot(self):
        """The mark stops a Build making a shot; one bound to the step (made
        before, or by hand) is still its shot, assessed like any."""
        steps = self._read("interactive", shots="clip")
        self.store.define_shot("A02", 500, 550, metadata={"step": "A02"})
        (a02,) = [r for r in ShotManifest(self.store).assess(steps) if r.step_id == "A02"]
        self.assertTrue(a02.built)
        self.assertNotEqual(a02.status, "no_shot")

    def test_a_local_edit_marks_a_step_either_way(self):
        from pythontk.core_utils.engines.shots.manifest.manifest_edits import ManifestEdits

        steps = self._read("interactive", shots="clip")
        edits = ManifestEdits()
        edits.set_step("A02", "animated", True, sheet=False)
        edits.set_step("A01", "animated", False, sheet=True)
        edits = ManifestEdits.from_dict(json.loads(json.dumps(edits.to_dict())))
        by_id = {s.step_id: s for s in edits.apply(steps)}
        self.assertEqual((by_id["A01"].animated, by_id["A02"].animated), (False, True))
        self.assertEqual((by_id["A01"].sheet, by_id["A02"].sheet), ({"animated": True}, {"animated": False}))
        edits.set_step("A02", "animated", False, sheet=False)  # back to the sheet's: no edit
        self.assertNotIn("animated", edits.steps.get("A02", {}))


class TestARowRebuiltFromTheSheet(_EngineCase):
    """``sync(only=...)``: the rows a person asked to rebuild from the sheet.
    Their shots are rebuilt though a hand edited them, and no other shot is
    touched: not another edited one, not one a full Build would update.  (A
    row's own "Rebuild from Sheet" beside the header's Edits mode, which
    decides for every shot at once.)  Added: 2026-11-22"""

    def _two(self):
        """Two built steps, each shot's description then changed by hand."""
        steps = [self._step("A01", "door"), self._step("A02", "window")]
        for step in steps:
            step.description = "the doc's text"
        ShotManifest(self.store).update(steps, initial_shot_length=50)
        for shot in list(self.store.shots):
            self.store.update_shot(shot.shot_id, description="the artist's note")
        return steps

    def test_only_the_chosen_steps_shot_is_rebuilt_from_the_sheet(self):
        steps = self._two()
        mani = ShotManifest(self.store)
        actions, _behaviors, assessment = mani.sync(steps, remove_missing=False, only=["A01"])
        self.assertEqual(list(actions), ["A01"], "the other step is not planned at all")
        self.assertNotEqual(actions["A01"], "kept")
        self.assertEqual(self.store.shot_by_name("A01").description, "the doc's text")
        self.assertEqual(self.store.shot_by_name("A02").description, "the artist's note")
        self.assertEqual(
            {s.step_id: s.status for s in assessment},
            {"A01": "valid", "A02": "edited"},
            "the assessment is of every step given, and the scope ended with the build",
        )
        self.assertFalse(mani.edited_by_hand(self.store.shot_by_name("A01")))

    def test_a_shot_a_full_build_would_update_is_left_alone(self):
        steps = [self._step("A01", "door"), self._step("A02", "window")]
        ShotManifest(self.store).update(steps, initial_shot_length=50)
        steps[1].objects.append(BuilderObject("lamp"))  # A02: the sheet changed
        ShotManifest(self.store).sync(steps, remove_missing=False, only=["A01"])
        self.assertFalse(
            [n for n in self.store.shot_by_name("A02").objects if n.endswith("lamp")]
        )

    def test_the_behavior_pass_reaches_the_chosen_shots_alone(self):
        steps = self._two()
        reached = []

        class Recording(ShotManifest):
            def apply_behaviors(self):
                reached.extend(shot.name for shot in self._behavior_shots())
                return {"applied": [], "skipped": []}

        Recording(self.store).sync(steps, remove_missing=False, only=["A02"])
        self.assertEqual(reached, ["A02"])
        reached.clear()
        Recording(self.store, rebuild_edited=True).sync(steps, remove_missing=False)
        self.assertEqual(reached, ["A01", "A02"], "no scope: every shot a build reaches")

    def test_a_locked_shot_stays_as_it_is(self):
        steps = self._two()
        shot = self.store.shot_by_name("A01")
        self.store.update_shot(shot.shot_id, locked=True)
        actions, _behaviors, _assessment = ShotManifest(self.store).sync(
            steps, remove_missing=False, only=["A01"]
        )
        self.assertEqual(actions, {"A01": "locked"})
        self.assertEqual(self.store.shot_by_name("A01").description, "the artist's note")

    def test_a_hand_renamed_shot_is_named_back(self):
        steps = self._two()
        self.store.update_shot(self.store.shot_by_name("A01").shot_id, name="Hand_Named")
        ShotManifest(self.store).sync(steps, remove_missing=False, only=["A01"])
        self.assertEqual(sorted(s.name for s in self.store.shots), ["A01", "A02"])


class TestAssessSeesWhatBuildRewrites(_EngineCase):
    """``StepStatus.updates``: what a Build rewrites on a built shot that no
    object status shows -- a step's Local Edits rename or re-description, an
    object the last build listed that the step no longer does.

    Bug: Assess judged only the objects, so after a Local Edit renamed a
    built step an Assess left Build disabled, and the rename was never built.
    Fixed: 2026-10-11"""

    def _built(self, *objs):
        step = self._step("A01", *objs)
        step.description = "the doc's text"
        ShotManifest(self.store).update([step], initial_shot_length=50)
        return step, self.store.shot_by_name("A01")

    def _status(self, step):
        (status,) = ShotManifest(self.store).assess([step])
        return status

    def _rebuild(self, step):
        ShotManifest(self.store).update([step], remove_missing=False)

    def test_a_fresh_build_leaves_nothing_to_rewrite(self):
        step, _shot = self._built("door", "window")
        status = self._status(step)
        self.assertEqual(status.updates, [])
        self.assertFalse(status.needs_build)

    def test_a_renamed_step_needs_its_shot_renamed(self):
        step, _shot = self._built("door")
        step.name = "Intro"
        status = self._status(step)
        self.assertEqual(status.updates, ["name"])
        self.assertTrue(status.needs_build)
        self._rebuild(step)
        self.assertEqual([s.name for s in self.store.shots], ["Intro"])
        self.assertEqual(self._status(step).updates, [])

    def test_a_new_description_is_rewritten(self):
        step, _shot = self._built("door")
        step.description = "the edited text"
        self.assertEqual(self._status(step).updates, ["description"])
        self._rebuild(step)
        self.assertEqual(self._status(step).updates, [])

    def test_an_object_the_step_no_longer_lists_is_taken_out(self):
        step, _shot = self._built("door", "window")
        del step.objects[1]  # hidden, or moved to another step
        self.assertEqual(self._status(step).updates, ["objects"])
        self._rebuild(step)
        self.assertEqual(self._status(step).updates, [])

    def test_a_rename_the_store_refuses_is_not_pending(self):
        step, _shot = self._built("door")
        self.store.define_shot("Intro", 200, 250)
        step.name = "Intro"
        self.assertEqual(self._status(step).updates, [])

    def test_a_kept_shot_has_nothing_pending(self):
        step, shot = self._built("door")
        self.store.update_shot(shot.shot_id, description="the artist's note")
        step.name = "Intro"
        self.assertEqual(self._status(step).updates, [])


class TestDisabledBehaviors(_EngineCase):
    """A behavior type turned off is neither keyed by Build nor reported
    missing by Assess; the keys it already has stay (2026-10-07: "ability to
    disable/enable certain behaviors")."""

    def _built_step(self):
        step = self._step("A01", "door", behaviors={"door": ["show", "hide"]})
        ShotManifest(self.store).update([step], initial_shot_length=50)
        return step, self.store.shot_by_name("A01")

    def test_assess_does_not_report_a_disabled_behavior(self):
        step, _shot = self._built_step()

        def verify(node, behavior, start, end, anchor_override=None):
            return behavior != "show"

        (status,) = ShotManifest(self.store, disabled_behaviors=()).assess([step], verify_fn=verify)
        self.assertEqual(status.objects[0].status, "missing_behavior")
        mani = ShotManifest(self.store, disabled_behaviors={"show"})
        (status,) = mani.assess([step], verify_fn=verify)
        self.assertEqual(status.objects[0].status, "valid")
        self.assertEqual(status.objects[0].behaviors, ["show", "hide"])

    def test_each_behavior_entry_ships_how_its_build_made_it(self):
        """Each object's behavior in a shot's metadata says how the build made
        it (key, marker; none: off), so another reader's manifest built from the
        same sheet knows per object what the delivery carries (2026-11-23:
        "behaviors work hand-in-hand when both manifests are used in the same
        pipeline, and work independently when not")."""
        step = self._step("A01", "door", behaviors={"door": ["show", "hide"]})
        ShotManifest(self.store, disabled_behaviors=()).update([step], initial_shot_length=50)
        def modes():
            return {e["behavior"]: e.get(ShotManifest.MODE) for e in self.store.shot_by_name("A01").metadata["behaviors"]}

        self.assertEqual(modes(), {"show": "key", "hide": "key"})
        ShotManifest(self.store, disabled_behaviors={"show"}, marked_behaviors={"hide"}).update(
            [step], remove_missing=False
        )
        self.assertEqual(modes(), {"show": None, "hide": "marker"}, "off says nothing; the old mode goes")

    def test_the_build_loop_skips_it_and_keeps_the_others_anchors(self):
        _step, shot = self._built_step()

        def run(skip):
            calls = []

            def apply_fn(
                node, behavior, start, end, source_path="", anchor_override=None
            ):
                calls.append((behavior, anchor_override))
                return []

            beh.Behaviors.apply_to_shots([shot], apply_fn, skip_behaviors=skip)
            return calls

        both = run(())
        self.assertEqual([b for b, _a in both], ["show", "hide"])
        self.assertEqual(run({"show"}), [both[1]], "hide lands where it did")

    def test_reapply_leaves_a_disabled_behavior_and_its_keys(self):
        step, shot = self._built_step()
        led = self.store.edit_ledger
        led.record_authored("door_op", 2.0, shot.shot_id, "show", "door")
        mani = ShotManifest(self.store, disabled_behaviors={"show"})
        released, applied = [], []
        mani._delete_keys = lambda curve, times: released.append(curve)
        mani._apply_one = lambda node, b, s, e, **kw: applied.append(b) or []
        self.assertTrue(mani.reapply_object(shot, step.objects[0]))
        self.assertEqual(applied, ["hide"])
        self.assertEqual(released, [], "show's keys stay")
        self.assertEqual(led.authored(owner=shot.shot_id), [("door_op", 2.0)])


class TestMarkedBehaviors(_EngineCase):
    """A behavior type set to Marker is not keyed: Build drops a timeline
    marker where its keys would go, and Assess checks for that marker
    (2026-10-11: "a third option ... of adding a marker in the sequencer
    instead of keying or simply ignoring all together")."""

    def _built_step(self, behaviors=None):
        behaviors = behaviors or {"door": ["show", "hide"], "lamp": ["show"]}
        step = self._step("A01", *behaviors, behaviors=behaviors)
        ShotManifest(self.store).update([step], initial_shot_length=50)
        return step, self.store.shot_by_name("A01")

    def _marks(self):
        return [
            m
            for m in self.store.markers
            if ShotStore.marker_owner(m) == "shot_manifest"
        ]

    def test_a_marked_behavior_is_not_keyed(self):
        mani = ShotManifest(
            self.store, disabled_behaviors={"highlight"}, marked_behaviors={"show"}
        )
        self.assertEqual(mani.unkeyed_behaviors, {"show", "highlight"})
        both = ShotManifest(
            self.store, disabled_behaviors={"show"}, marked_behaviors={"show"}
        )
        self.assertEqual(both.marked_behaviors, frozenset(), "off wins")

    def test_build_marks_one_frame_per_behavior_naming_its_objects(self):
        _step, shot = self._built_step()
        records = ShotManifest(
            self.store, marked_behaviors={"show", "hide"}
        ).place_markers()
        fade = self.store.effect_recipe.fade_frames
        got = [(m["time"], m["note"], m["data"]["objects"]) for m in self._marks()]
        self.assertEqual(
            got,
            [
                (shot.start, "Visibility · show: door, lamp", ["door", "lamp"]),
                (shot.end - fade, "Visibility · hide: door", ["door"]),
            ],
        )
        (first, _second) = self._marks()
        self.assertEqual(first["data"]["shot_id"], shot.shot_id)
        self.assertFalse(first["draggable"], "the next build would put it back")
        self.assertEqual(
            sorted((r["object"], r["behavior"]) for r in records),
            [("door", "hide"), ("door", "show"), ("lamp", "show")],
        )

    def test_a_rebuild_moves_its_markers_and_never_a_hands(self):
        _step, shot = self._built_step()
        self.store.markers.append({"time": 3.0, "note": "the animator's"})
        ShotManifest(self.store, marked_behaviors={"show"}).place_markers()
        self.store.update_shot(shot.shot_id, start=101, end=151)
        ShotManifest(self.store, marked_behaviors={"show"}).place_markers()
        self.assertEqual([m["time"] for m in self._marks()], [101.0])
        # Keyed again: its marker goes.
        ShotManifest(self.store, disabled_behaviors=()).place_markers()
        self.assertEqual(self._marks(), [])
        self.assertEqual(self.store.markers, [{"time": 3.0, "note": "the animator's"}])

    def test_a_turned_off_behavior_keeps_its_marker(self):
        self._built_step()
        ShotManifest(self.store, marked_behaviors={"show"}).place_markers()
        ShotManifest(self.store, disabled_behaviors={"show"}).place_markers()
        self.assertEqual(len(self._marks()), 1)

    def test_a_kept_shot_keeps_its_markers(self):
        _step, shot = self._built_step()
        ShotManifest(self.store, marked_behaviors={"show"}).place_markers()
        self.store.update_shot(shot.shot_id, description="the artist's note")
        ShotManifest(self.store).place_markers()
        self.assertEqual(len(self._marks()), 1, "a build leaves a kept shot alone")

    def test_assess_checks_the_marker_not_the_keys(self):
        step, shot = self._built_step()
        mani = ShotManifest(self.store, marked_behaviors={"show", "hide"})

        def no_keys(*_args, **_kwargs):
            return False

        (status,) = mani.assess([step], verify_fn=no_keys)
        self.assertEqual(status.objects[0].status, "missing_behavior")
        self.assertTrue(status.needs_build, "Build places the missing markers")
        mani.place_markers()
        (status,) = mani.assess([step], verify_fn=no_keys)
        self.assertEqual([o.status for o in status.objects], ["valid", "valid"])
        self.assertFalse(status.needs_build)
        # A marker the shot moved away from is missing where the build puts it.
        self.store.update_shot(shot.shot_id, start=101, end=151)
        (status,) = mani.assess([step], verify_fn=no_keys)
        self.assertEqual(status.objects[1].broken_behaviors, ["show"])

    def test_sync_reports_what_it_marked(self):
        step = self._step("A01", "door", behaviors={"door": ["show"]})
        mani = ShotManifest(self.store, marked_behaviors={"show"})
        _actions, result, _assessment = mani.sync([step], initial_shot_length=50)
        self.assertEqual([r["object"] for r in result["marked"]], ["door"])
        self.assertEqual(len(self._marks()), 1)


class TestModesInForce(_EngineCase):
    """Bug: a build from code (``ShotManifest.from_csv`` + ``sync``) keyed
    every behavior while the panel started a scene that keeps no modes at Key
    only for what the sheet's cells name: one scene built two ways.
    Fixed: 2026-11-26 -- with no mode given, ``sync`` and ``assess`` read
    ``ManifestEdits.modes_in_force`` for the sheet, as the panel does."""

    def _sheet(self, text, template):
        path = _write_csv(text)
        self.addCleanup(os.remove, path)
        return path, mapping_mod.Mapping.resolve(path, name=template)

    def _modes(self, shot):
        return {e["behavior"]: e.get(ShotManifest.MODE) for e in shot.metadata.get("behaviors") or ()}

    def test_a_code_build_keys_a_cell_rows_highlight_and_leaves_a_sentences_off(self):
        path, steps = self._sheet(
            "Step,Step Name,Voice,Action,Target,Settings\nA01,Open,Open it.,,,\n,,,highlight,DOOR,\n", "linear"
        )
        from pythontk.core_utils.engines.shots.manifest.manifest_edits import ManifestEdits

        mani = ShotManifest(self.store, source=path)
        mani.sync(steps, initial_shot_length=50)
        self.assertEqual(self._modes(self.store.shot_by_name("A01")), {"highlight": "key"})
        self.assertEqual(
            ManifestEdits.behavior_modes(self.store, path), {"key": ["highlight"], "marker": []},
            "the build records what it built with, as the panel's Build does",
        )
        self.assertNotIn("highlight", mani.unkeyed_behaviors)
        self.assertIn("show", mani.disabled_behaviors, "every other behavior Off until chosen")
        sentence_store = ShotStore(host=_ScenicHost())
        path, _steps = self._sheet("Step,Step Contents,Asset Names\nA01.),Highlight the door,DOOR\n", "default")
        mani, steps = ShotManifest.from_csv(path, store=sentence_store)
        self.assertEqual(mani.source, path, "from_csv names its sheet")
        mani.sync(steps, initial_shot_length=50)
        self.assertEqual(self._modes(sentence_store.shot_by_name("A01")), {"highlight": None}, "a sentence asks nothing")

    def test_the_scenes_own_modes_and_given_modes_win(self):
        from pythontk.core_utils.engines.shots.manifest.manifest_edits import ManifestEdits

        path, steps = self._sheet(
            "Step,Step Name,Voice,Action,Target,Settings\nA01,Open,Open it.,,,\n,,,highlight,DOOR,\n", "linear"
        )
        ManifestEdits.set_behavior_modes(self.store, path, [], ["highlight"])
        ShotManifest(self.store, source=path).sync(steps, initial_shot_length=50)
        self.assertEqual(self._modes(self.store.shot_by_name("A01")), {"highlight": "marker"})
        ShotManifest(self.store, disabled_behaviors={"highlight"}, source=path).update(steps, remove_missing=False)
        self.assertEqual(self._modes(self.store.shot_by_name("A01")), {"highlight": None}, "given modes are as given")


class TestNamesNeverLeaves(_EngineCase):
    """What a build, Assess, discovery and auto-fill compare and keep are
    names (a node's own, or a path's tail), never a bare leaf: two nodes
    sharing a leaf stay two."""

    class _BareNames(PureShotsHost):
        """Members kept as bare object names, the path in the parents (a host that keeps no paths)."""

        PARENTS = {"LIGHTS": "SOLENOID_LOC"}

        def member_nodes(self, name):
            return ["LIGHTS"] if name == "SOLENOID_LOC|LIGHTS" else [name]

        def member_lineage(self, member):
            return [self.PARENTS[member], member] if member in self.PARENTS else str(member).split("|")

    def test_a_steps_path_tail_owns_its_bare_named_member(self):
        """Bug: a step listing ``SOLENOID_LOC|LIGHTS`` showed its own ``LIGHTS``
        unlisted on a host that keeps bare names, and Assess said it
        was not in its shot.
        Fixed: 2026-11-26 -- compared through the store's host
        (``ShotsHost.member_lineage``)."""
        store = ShotStore(host=self._BareNames())
        store.define_shot("A01", 1, 50, objects=["LIGHTS"], metadata={"step": "A01"})
        (st,) = ShotManifest(store, disabled_behaviors=()).assess([self._step("A01", "SOLENOID_LOC|LIGHTS")])
        self.assertEqual(st.additional_objects, [])
        self.assertEqual(st.objects[0].status, "valid")

    def test_discovery_is_given_names_so_a_member_hides_only_its_own_node(self):
        """Bug: discovery was given leaves (``{member_key(...)}``), so a member
        ``|rig|PANEL|LATCH`` hid an animated ``|rig|DOOR|LATCH``.
        Fixed: 2026-11-26 -- the doc names and the members as stored."""
        given = []

        class Host(ShotManifest):
            def _discover_scene_objects(self, start, end, exclude_names):
                given.append(set(exclude_names))
                animated = ["|rig|PANEL|LATCH", "|rig|DOOR|LATCH"]
                known = {m for _n, m in self._pairs(exclude_names, animated)}
                return [a for a in animated if a not in known]

        self.store.define_shot("A01", 1, 50, objects=["|rig|PANEL|LATCH"], metadata={"step": "A01"})
        mani = Host(self.store, disabled_behaviors=())
        (st,) = mani.assess([self._step("A01", "PANEL|LATCH")])
        self.assertEqual(given, [{"PANEL|LATCH", "|rig|PANEL|LATCH"}])
        self.assertEqual(st.additional_objects, ["|rig|DOOR|LATCH"])
        given.clear()
        empty = self._step("A01")
        self.assertEqual(mani.fill_missing_assets([empty]), {"A01": ["PANEL|LATCH", "DOOR|LATCH"]},
                         "auto-fill names two same-leaf members apart, by the tail that tells them apart")
        self.assertEqual(given, [{"|rig|PANEL|LATCH"}])

    def test_a_hand_removing_one_of_two_same_leaf_members_is_an_edit(self):
        """Bug: the build's record kept members by leaf, so a hand removing
        ``|rig|DOOR|LATCH`` beside ``|rig|PANEL|LATCH`` read as no edit and the
        next build undid it.
        Fixed: 2026-11-26 -- the record keeps names (``ShotStore.member_names``)."""
        mani = ShotManifest(self.store, disabled_behaviors=())
        shot = self.store.define_shot("A01", 1, 50, objects=["|rig|PANEL|LATCH", "|rig|DOOR|LATCH", "|set|AC:door"])
        mani._record_build([shot.shot_id])
        self.assertEqual(shot.metadata[ShotManifest.BUILT]["member_names"], ["DOOR|LATCH", "PANEL|LATCH", "door"])
        self.assertFalse(mani.edited_by_hand(shot))
        self.store.update_shot(shot.shot_id, objects=["|rig|PANEL|LATCH", "|set|AC:door"])
        self.assertTrue(mani.edited_by_hand(shot))

    def test_an_older_leaf_record_still_matches_and_the_systems_additions_are_no_edit(self):
        shot = self.store.define_shot("A01", 1, 50, objects=["|rig|PANEL|LATCH", "|rig|DOOR|LATCH"])
        meta = dict(shot.metadata or {})
        meta[ShotManifest.BUILT] = {"length": 49.0, "members": ["LATCH"], "description": ""}  # an older build's
        self.store.update_shot(shot.shot_id, metadata=meta)
        mani = ShotManifest(self.store, disabled_behaviors=())
        self.assertFalse(mani.edited_by_hand(shot), "a leaf record reads by leaf")
        self.store.update_shot(shot.shot_id, objects=["|rig|PANEL|LATCH"])
        mani._record_build([shot.shot_id])
        self.assertEqual(shot.metadata[ShotManifest.BUILT]["member_names"], ["LATCH"])
        self.store.add_system_members(shot.shot_id, ["|rig|DOOR|LATCH", "|set|lid"])
        self.assertFalse(mani.edited_by_hand(shot), "what the system filled in is no hand's")
        self.store.update_shot(shot.shot_id, objects=list(shot.objects) + ["|set|bolt"])
        self.assertTrue(mani.edited_by_hand(shot), "a member a hand adds still counts")


class TestFillFromPairs(_EngineCase):
    """Auto-fill reads only the members of a step's paired shot: no shot, no
    names (an unbuilt scene has no step-to-time link to read)."""

    def test_names_come_from_the_paired_shot_through_the_candidate_filter(self):
        self.store.define_shot("Step_1", 1, 10, objects=["|set|AC:door", "|cam"])
        mani = ShotManifest(self.store, match="name_then_order")
        mani._is_asset_candidate = lambda name: name != "|cam"
        steps = [self._step("IN_01"), self._step("IN_02")]
        filled = mani.fill_missing_assets(steps)
        self.assertEqual(filled, {"IN_01": ["door"]})
        self.assertEqual(steps[1].objects, [])


class TestRangeBookkeeping(unittest.TestCase):
    """The user-range rules both DCC manifest panels apply (RangeResolver)."""

    RR = rr.RangeResolver

    def _steps(self, *ids):
        return [BuilderStep(i, "A", "t", "", [BuilderObject(f"{i}_geo")]) for i in ids]

    def test_step_index(self):
        steps = self._steps("A01", "A02", "A03")
        self.assertEqual(self.RR.step_index(steps, "A02"), 1)
        self.assertEqual(self.RR.step_index(steps, "nope"), -1)

    def test_all_ranges_complete_needs_both_ends_on_every_step(self):
        steps = self._steps("A01", "A02")
        self.assertFalse(self.RR.all_ranges_complete([], {}))
        self.assertFalse(
            self.RR.all_ranges_complete(steps, {"A01": (1, 10), "A02": (20, None)})
        )
        self.assertFalse(self.RR.all_ranges_complete(steps, {"A01": (1, 10)}))
        self.assertTrue(
            self.RR.all_ranges_complete(steps, {"A01": (1, 10), "A02": (20, 30)})
        )

    def test_cascade_from_drops_only_downstream_ranges(self):
        steps = self._steps("A01", "A02", "A03", "A04")
        ranges = {"A01": (1, 5), "A02": (6, 9), "A04": (30, None)}
        dropped = self.RR.cascade_from(steps, ranges, 1)
        self.assertEqual(dropped, ["A04"])
        self.assertEqual(ranges, {"A01": (1, 5), "A02": (6, 9)})

    def test_parse_range_edit(self):
        parse = self.RR.parse_range_edit
        self.assertIsNone(parse("", " "), "both empty clears")
        self.assertEqual(parse("10", ""), (10.0, None))
        self.assertEqual(parse(" 10 ", "25"), (10.0, 25.0))
        for start, end in (("x", ""), ("1", "y"), ("", "5"), ("-1", ""), ("5", "5")):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    parse(start, end)

    def test_previous_end_skips_unresolved_predecessors(self):
        steps = self._steps("A01", "A02", "A03")
        # Sparse: A02 did not resolve (selected-keys mode skips it).
        last = [("A01", 1.0, 40.0, False), ("A03", 50.0, 60.0, False)]
        self.assertEqual(self.RR.previous_end(steps, last, 2), 40.0)
        self.assertIsNone(self.RR.previous_end(steps, last, 0))
        self.assertIsNone(self.RR.previous_end(steps, [], 2))

    def test_find_collisions(self):
        resolved = [
            ("A01", 1.0, 20.0, True),
            ("A02", 15.0, 30.0, False),  # starts inside A01
            ("A03", 31.0, None, False),  # no end: occupies its start only
            ("A04", 31.0, 40.0, False),
        ]
        self.assertEqual(self.RR.find_collisions(resolved), [("A01", "A02")])
        resolved[2] = ("A03", 32.0, None, False)
        self.assertEqual(
            self.RR.find_collisions(resolved), [("A01", "A02"), ("A03", "A04")]
        )
        self.assertEqual(self.RR.find_collisions(resolved[:1]), [])

    def test_gaps_from_regions(self):
        regions = [
            {"start": 1.0, "end": 10.0},
            {"start": 20.0},
            {"start": 30.0, "end": None},
        ]
        self.assertEqual(
            self.RR.gaps_from_regions(regions), ([1.0, 20.0, 30.0], {1.0: 10.0})
        )
        self.assertEqual(self.RR.gaps_from_regions([]), ([], {}))
        self.assertEqual(self.RR.gaps_from_regions(None), ([], {}))


class TestFindObjectStatus(unittest.TestCase):
    def test_scoped_to_the_step_when_given(self):
        results = [
            StepStatus("A01", True, [ObjectStatus("door", True, "valid")]),
            StepStatus(
                "A05",
                True,
                [
                    ObjectStatus(
                        "door", True, "missing_behavior", ["hide"], ["hide"]
                    )
                ],
            ),
        ]
        self.assertEqual(StepStatus.find_object(results, "door").status, "valid")
        found = StepStatus.find_object(results, "door", "A05")
        self.assertEqual(found.broken_behaviors, ["hide"])
        self.assertIsNone(StepStatus.find_object(results, "door", "A09"))
        self.assertIsNone(StepStatus.find_object(results, "wall"))
        self.assertIsNone(StepStatus.find_object([], "door"))


class TestBehaviorStates(unittest.TestCase):
    """``StepStatus.behavior_states``: each behavior of an object row in the
    state a panel shows it in, judged once here so every panel says the same.

    Bug: the Shot Manifest panel judged each behavior chip itself and called
    one "built" on a step with no shot: an existing object of an unbuilt step
    assesses ``valid``, and the panel read that as built.
    Fixed: 2026-11-23 (the judgment moved here)
    """

    @staticmethod
    def _states(step, behaviors=("show",), **kwargs):
        return {
            b: state
            for b, state, _words in StepStatus.behavior_states(
                step, "door", list(behaviors), **kwargs
            )
        }

    @staticmethod
    def _step(status="valid", built=True, **fields):
        door = ObjectStatus("door", status != "missing_object", status, ["show"], **fields)
        return StepStatus("A01", built, [door])

    def test_an_unbuilt_steps_behavior_is_to_build_never_built(self):
        self.assertEqual(self._states(self._step(built=False)), {"show": "to_build"})
        status, words = StepStatus.BEHAVIOR_STATES["to_build"]
        self.assertEqual(status, "missing_shot", "a note, as the step's own row says")
        self.assertNotIn("built", words.split(":")[0])

    def test_each_state(self):
        cases = {
            # state: (step, behavior_states' keywords)
            "built": (self._step(), {}),
            "marked": (self._step(), {"marked": {"show"}}),
            "off": (self._step(), {"off": {"show"}}),
            "taken": (self._step(), {"sheet": ["show"], "behaviors": []}),
            "added": (self._step(), {"sheet": []}),
            "object_missing": (self._step("missing_object"), {}),
            "object_ambiguous": (self._step("ambiguous_object"), {}),
            "not_in_shot": (self._step("not_in_shot"), {}),
            "missing": (self._step("missing_behavior", broken_behaviors=["show"]), {}),
            "marker_missing": (
                self._step("missing_behavior", broken_behaviors=["show"]),
                {"marked": {"show"}},
            ),
            "stale": (self._step("stale_behavior", stale_behaviors=["show"]), {}),
            "unknown": (
                self._step(
                    "unknown_behavior",
                    broken_behaviors=["show"],
                    unknown_behaviors=["show"],
                ),
                {},
            ),
            "conflict": (
                self._step(
                    "behavior_conflict",
                    broken_behaviors=["show"],
                    conflict_behaviors=["show"],
                ),
                {},
            ),
            "to_build": (self._step(built=False), {}),
            "no_shot": (StepStatus("A01", False, self._step().objects, animated=False), {}),
            "locked": (StepStatus("A01", True, self._step().objects, locked=True), {}),
            "": (None, {}),  # not assessed
        }
        self.assertEqual(set(cases), set(StepStatus.BEHAVIOR_STATES), "every state is covered")
        for state, (step, kwargs) in cases.items():
            with self.subTest(state=state):
                behaviors = kwargs.pop("behaviors", ["show"])
                self.assertEqual(self._states(step, behaviors, **kwargs), {"show": state})

    def test_every_state_says_itself_at_a_status_a_panel_colours(self):
        from pythontk.core_utils.engines.shots.shot_model import SHOT_PALETTE

        for state, (status, words) in StepStatus.BEHAVIOR_STATES.items():
            with self.subTest(state=state):
                self.assertTrue(status is None or status in SHOT_PALETTE, status)
                self.assertEqual(bool(words), bool(state), "a state has words; none has none")

    def test_a_taken_behavior_follows_the_objects_own(self):
        """The sheet's behaviors a Local Edit took away close the list, after
        the object's own, each once."""
        got = StepStatus.behavior_states(
            self._step(), "door", ["highlight", "hide"], sheet=["show", "highlight"]
        )
        self.assertEqual(
            [(b, s) for b, s, _w in got],
            [("highlight", "built"), ("hide", "added"), ("show", "taken")],
        )

    def test_an_object_this_step_did_not_assess_is_not_judged(self):
        """A step judged, its object not (renamed since): no state but an edit's."""
        self.assertEqual(self._states(StepStatus("A01", True, [])), {"show": ""})
        self.assertEqual(self._states(StepStatus("A01", True, []), sheet=[]), {"show": "added"})

    def test_a_quiet_step_has_nothing_to_report(self):
        """``StepStatus.quiet``: what a view of the steps that need something
        leaves out, read off the assessment (a panel read its cells' text)."""
        ok = ObjectStatus("door", True, "valid")
        self.assertTrue(StepStatus("A01", True, [ok]).quiet)
        self.assertTrue(StepStatus("A01", True, [ok], shrinkable_frames=12.0).quiet, "a trim is no issue")
        self.assertTrue(StepStatus("A01", False, [ok], animated=False).quiet, "plays no animation: needs none")
        for loud in (
            StepStatus("A01", False, [ok]),
            StepStatus("A01", True, [ObjectStatus("door", False, "missing_object")]),
            StepStatus("A01", True, [ObjectStatus("door", True, "user_animated")]),
            StepStatus("A01", True, [ok], additional_objects=["stray"]),
            StepStatus("A01", True, [ok], updates=["name"]),
            StepStatus("A01", True, [ok], dropped_behaviors=[["door", "highlight"]]),
            StepStatus("A01", True, [ok], locked=True),
            StepStatus("A01", True, [ok], edited=True),
            StepStatus("A01", True, []),
        ):
            with self.subTest(step=loud):
                self.assertFalse(loud.quiet)


class TestStepsFromShots(unittest.TestCase):
    """``BuilderStep.from_shots``: the scene's own shots as manifest steps."""

    def test_a_manifest_built_shot_round_trips_its_step(self):
        from pythontk.core_utils.engines.shots.shot_model import ShotBlock

        step = BuilderStep(
            step_id="A01",
            section="A",
            section_title="AILERON RIGGING",
            description="Aileron fades in",
            objects=[
                BuilderObject(name="aileron_geo", behaviors=["show"]),
                BuilderObject(name="wing_geo"),
                BuilderObject(name="vo_A01", kind="audio", behaviors=["set_clip"]),
            ],
            audio="Narrator intro",
        )
        shot = ShotBlock(
            shot_id=0,
            name="A01",
            start=1.0,
            end=40.0,
            # A discovered member the CSV never named stays out of the step,
            # so assess reports it as additional, exactly as in CSV mode.
            objects=["|grp|aileron_geo", "|grp|wing_geo", "|grp|extra_geo"],
            metadata=ShotManifest._step_metadata(step),
            description="Aileron fades in",
        )
        steps, ranges = BuilderStep.from_shots([shot])
        self.assertEqual(ranges, {"A01": (1.0, 40.0)})
        (got,) = steps
        self.assertEqual(
            (got.step_id, got.section, got.section_title, got.description, got.audio),
            ("A01", "A", "AILERON RIGGING", "Aileron fades in", "Narrator intro"),
        )
        self.assertEqual(
            [(o.name, o.kind, o.behaviors) for o in got.objects],
            [
                ("aileron_geo", "scene", ["show"]),
                ("wing_geo", "scene", []),
                ("vo_A01", "audio", ["set_clip"]),
            ],
        )

    def test_an_object_an_older_build_listed_per_row_is_one_object(self):
        """A build before the parse merged one object's rows listed it once
        per row; read back, each copy took every behavior, doubling them.
        Fixed: 2026-10-04
        """
        from pythontk.core_utils.engines.shots.shot_model import ShotBlock

        step = BuilderStep(
            step_id="A01",
            section="A",
            section_title="",
            description="",
            objects=[
                BuilderObject(name="door", behaviors=["show", "hide"]),
                BuilderObject(name="door", behaviors=["show"]),
            ],
        )
        shot = ShotBlock(
            0, "A01", 1.0, 40.0, metadata=ShotManifest._step_metadata(step)
        )
        (got,), _ranges = BuilderStep.from_shots([shot])
        self.assertEqual(
            [(o.name, o.behaviors) for o in got.objects],
            [("door", ["show", "hide", "show"])],
        )

    def test_a_manifest_step_with_no_assets_keeps_its_members_additional(self):
        """An empty CSV object list is still the manifest's plan: members the
        build discovered must not become planned objects."""
        from pythontk.core_utils.engines.shots.shot_model import ShotBlock

        step = BuilderStep(step_id="A02", section="A", section_title="", description="")
        shot = ShotBlock(
            shot_id=1,
            name="A02",
            start=1.0,
            end=9.0,
            objects=["|grp|discovered_geo"],
            metadata=ShotManifest._step_metadata(step),
        )
        (got,), _ = BuilderStep.from_shots([shot])
        self.assertEqual(got.objects, [])

    def test_a_shot_built_without_a_manifest_keeps_its_description(self):
        from pythontk.core_utils.engines.shots.shot_model import ShotBlock

        shot = ShotBlock(
            shot_id=3,
            name="intro",
            start=10.0,
            end=50.0,
            objects=["|rig|arm", "|rig|hand"],
            metadata={"section": "B"},
            description="Arm reaches for the lever",
        )
        (got,), ranges = BuilderStep.from_shots([shot])
        self.assertEqual(got.description, "Arm reaches for the lever")
        self.assertEqual(got.section, "B")
        self.assertEqual([o.name for o in got.objects], ["|rig|arm", "|rig|hand"])
        self.assertEqual(ranges, {"intro": (10.0, 50.0)})


class TestDescribeReadFailure(unittest.TestCase):
    """An unreadable CSV is explained by likely causes, never one asserted."""

    @staticmethod
    def _describe(path, *, free, placeholder):
        exc = OSError(22, "Invalid argument")
        with (
            patch("pythontk.FileUtils.free_space", return_value=free),
            patch("pythontk.FileUtils.is_cloud_placeholder", return_value=placeholder),
        ):
            return ManifestModel.describe_read_failure(path, exc).lower()

    def test_low_disk_surfaces_the_free_space_figure(self):
        msg = self._describe("X:/seq/m.csv", free=786 * 1024 * 1024, placeholder=True)
        self.assertIn("786 mb free", msg)
        self.assertIn("disk may be full", msg)
        self.assertNotIn("available offline", msg)

    def test_cloud_file_names_the_sync_client_without_a_figure(self):
        msg = self._describe("X:/seq/m.csv", free=10 * 1024**3, placeholder=True)
        self.assertIn("sync", msg)
        self.assertNotIn("mb free", msg)

    def test_local_file_omits_the_cloud_cause_and_keeps_the_error(self):
        msg = self._describe("C:/local/m.csv", free=None, placeholder=False)
        self.assertNotIn("cloud", msg)
        self.assertIn("invalid argument", msg)


class TestSeedUserFolder(unittest.TestCase):
    """Mapping.seed_user_folder: an empty user folder gets an example + reference."""

    def _ts(self, tmp):
        from pythontk import TemplateSet

        return TemplateSet(
            "seed_test",
            mapping_mod.Mapping.templates().spec,
            "pythontk",
            user_dir=Path(tmp),
        )

    def test_seeds_empty_then_noops(self):
        with tempfile.TemporaryDirectory() as tmp:
            ts = self._ts(tmp)
            self.assertTrue(mapping_mod.Mapping.seed_user_folder(ts))
            names = sorted(p.name for p in Path(tmp).iterdir())
            self.assertIn("example.json", names)
            self.assertIn("MAPPING_FORMAT.md", names)
            (Path(tmp) / "example.json").write_text("EDIT", encoding="utf-8")
            self.assertFalse(mapping_mod.Mapping.seed_user_folder(ts))
            self.assertEqual(
                (Path(tmp) / "example.json").read_text(encoding="utf-8"), "EDIT"
            )

    def test_the_active_pointer_does_not_count_as_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            ts = self._ts(tmp)
            (Path(tmp) / ".active").write_text("speedrun", encoding="utf-8")
            self.assertTrue(mapping_mod.Mapping.seed_user_folder(ts))
            self.assertTrue((Path(tmp) / "MAPPING_FORMAT.md").is_file())

    def test_a_missing_folder_is_a_quiet_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            ts = self._ts(os.path.join(tmp, "absent"))
            self.assertFalse(mapping_mod.Mapping.seed_user_folder(ts))


class TestReapplyObject(unittest.TestCase):
    """The per-object "Apply [..]": every behavior re-keyed where a build keys it."""

    class _Host(ShotManifest):
        """One curve; each behavior keys it at the frame ``plan`` names."""

        def __init__(self, store):
            super().__init__(store, disabled_behaviors=())
            self.plan, self.curve, self.anchors = {}, {}, []

        def _apply_one(self, node, behavior, start, end, **kwargs):
            self.anchors.append(kwargs.get("anchor_override"))
            t = self.plan[behavior]
            self.curve[t] = behavior
            return [("crv", t)]

        def _delete_keys(self, curve, times):
            for t in times:
                self.curve.pop(t, None)

    def setUp(self):
        self.store = ShotStore()
        self.shot = self.store.define_shot("A01", 100, 300, objects=["Obj"])
        self.host = self._Host(self.store)

    def test_places_each_behavior_where_a_build_does(self):
        obj = BuilderObject(name="Obj", behaviors=["highlight", "hide"])
        self.host.plan = {"highlight": 100, "hide": 285}
        self.host.reapply_object(self.shot, obj)
        self.assertEqual(self.host.anchors, [None, None])

    def test_a_shared_frame_survives_the_reapply(self):
        """Bug: each behavior's old keys were released just before it was
        re-keyed, so a later behavior's release deleted the key an earlier one
        had just written on a frame they share -- the loss the build avoids by
        releasing everything first.
        Fixed: 2026-10-03
        """
        obj = BuilderObject(name="Obj", behaviors=["show", "hide"])
        self.host.plan = {"show": 100, "hide": 200}
        self.host.reapply_object(self.shot, obj)
        self.host.plan = {"show": 200, "hide": 300}  # the shot moved
        self.host.reapply_object(self.shot, obj)
        self.assertEqual(
            sorted(self.host.curve.items()), [(200, "show"), (300, "hide")]
        )

    def test_an_audio_row_is_keyed_by_its_track_name(self):
        """Bug: Apply on an audio row resolved the track's name as a scene
        object; a DCC resolves only nodes, so the clip read as missing and
        nothing was keyed.  A build keys a clip by its name.
        Fixed: 2026-10-04
        """
        calls = []

        class Host(ShotManifest):
            def _resolve_object(self, name):
                return name, "missing"  # no scene node answers to the name

            def _apply_one(self, node, behavior, start, end, **kwargs):
                calls.append((node, behavior, kwargs.get("source_path")))
                return [("vo_track", start)]

        host = Host(self.store, disabled_behaviors=())
        clip = BuilderObject(
            "A01_Hello", ["set_clip"], kind="audio", source_path="a.wav"
        )
        self.assertTrue(host.reapply_object(self.shot, clip))
        self.assertEqual(calls, [("A01_Hello", "set_clip", "a.wav")])
        self.assertEqual(
            self.store.edit_ledger.authored(owner=self.shot.shot_id),
            [("vo_track", 100.0)],
        )
        # A scene object nothing resolves is still left alone.
        self.assertFalse(
            host.reapply_object(self.shot, BuilderObject("door", ["show"]))
        )
        self.assertEqual(len(calls), 1)

    def test_an_object_on_two_rows_reapplies_where_its_build_keyed(self):
        """One object on two rows of a step is one object (the parse merges
        it), so its Apply re-keys every behavior at the build's anchors and
        takes out none of the keys the build left."""
        path = _write_csv(
            "Step,Step Contents,Asset Names\n"
            "A01.),Door fades in then fades out,door\n"
            ",Door fades in again,door\n"
        )
        try:
            (step,) = ManifestModel.parse_csv(path)
        finally:
            os.remove(path)

        class Host(ShotManifest):
            def __init__(self, store):
                super().__init__(store, disabled_behaviors=())
                self.curve, self.anchors = {}, []

            def apply_behaviors(self):
                return beh.Behaviors.apply_to_shots(
                    self.store.sorted_shots(),
                    apply_fn=self._apply_one,
                    release_fn=lambda shot, name, b: self.release_authored(
                        shot.shot_id, name, b
                    ),
                )

            def _apply_one(self, node, behavior, start, end, **kwargs):
                anchor = kwargs.get("anchor_override")
                self.anchors.append(anchor)
                t = start + (anchor or 0.0) * (end - start)
                self.curve[t] = behavior
                return [("door.v", t)]

            def _delete_keys(self, curve, times):
                for t in times:
                    self.curve.pop(t, None)

        host = Host(self.store)
        host.sync([step], ranges={"A01": (100.0, 300.0)})
        built = dict(host.curve)
        self.assertEqual(built, {100.0: "show", 200.0: "hide", 300.0: "show"})
        host.anchors.clear()
        (door,) = step.objects
        host.reapply_object(host.pair([step]).shots["A01"], door)
        self.assertEqual(host.anchors, [0.0, 0.5, 1.0])
        self.assertEqual(host.curve, built)


class TestRecipeOwnership(unittest.TestCase):
    """The manifest keys from the scene's effect recipe: its keys carry the
    recipe they were made under, Assess flags what an older recipe made, and a
    build takes out the keys of behaviors the doc dropped."""

    class _Host(ShotManifest):
        """Each behavior keys ``<node>.<behavior>`` at the shot start (a fade
        out at the end) and reports it, as a host applier does."""

        def __init__(self, store):
            super().__init__(store, disabled_behaviors=())  # every behavior keyed
            self.curves = {}

        def apply_behaviors(self):
            return beh.Behaviors.apply_to_shots(
                self.store.sorted_shots(),
                apply_fn=self._key,
                release_fn=lambda shot, name, b: self.release_authored(
                    shot.shot_id, name, b
                ),
            )

        def _key(self, node, behavior, start, end, **_kwargs):
            curve = f"{node}.{behavior}"
            t = end if behavior == "hide" else start
            self.curves.setdefault(curve, {})[t] = behavior
            return [(curve, t)]

        def _apply_one(self, node, behavior, start, end, **kwargs):
            return self._key(node, behavior, start, end)

        def _delete_keys(self, curve, times):
            for t in times:
                self.curves.get(curve, {}).pop(t, None)

    def setUp(self):
        self.store = ShotStore()
        self.host = self._Host(self.store)

    def _steps(self, *behaviors, name="Door"):
        objects = [BuilderObject(name=name, behaviors=list(behaviors))]
        return [BuilderStep("A01", "A", "", "", objects=objects)]

    def _build(self, steps):
        self.host.sync(steps, ranges={"A01": (100.0, 300.0)})
        return self.host.pair(steps).shots["A01"]

    def _door(self, steps):
        (status,) = self.host.assess(steps)
        return status, status.objects[0]

    def test_a_build_stamps_its_keys_with_the_recipe(self):
        shot = self._build(self._steps("show", "highlight"))
        led, recipe = self.store.edit_ledger, self.store.effect_recipe
        self.assertEqual(
            led.authored_stamps(shot.shot_id, "Door", "highlight"),
            {recipe.fingerprint("pulse")},
        )
        self.assertEqual(
            led.authored_stamps(shot.shot_id, "Door", "show"),
            {recipe.fingerprint("fade_in")},
        )

    def test_a_changed_recipe_flags_its_keys_until_a_build(self):
        steps = self._steps("show", "highlight")
        self._build(steps)
        step, door = self._door(steps)
        self.assertEqual(door.status, "valid")
        self.assertFalse(step.needs_build)

        self.store.update_effect_recipe(pulse_period=2.0)
        step, door = self._door(steps)
        self.assertEqual(door.status, "stale_behavior")
        self.assertEqual(door.stale_behaviors, ["highlight"])  # the fade is current
        self.assertEqual(step.status, "stale_behavior")
        self.assertTrue(step.needs_build)

        self._build(steps)
        step, door = self._door(steps)
        self.assertEqual(door.status, "valid")
        self.assertFalse(step.needs_build)

    def test_new_colours_leave_the_pulse_current(self):
        steps = self._steps("highlight")
        self._build(steps)
        self.store.update_effect_recipe(pulse_bright=(1.0, 0.0, 0.0))
        self.assertEqual(self._door(steps)[1].status, "valid")

    def test_a_claim_from_before_stamps_reads_stale(self):
        shot = self._build(self._steps("highlight"))
        led = self.store.edit_ledger
        ((curve, t),) = led.authored(owner=shot.shot_id, obj="Door")
        led.release_authored(curve, t)
        led.record_authored(curve, t, shot.shot_id, "highlight", "Door")  # no stamp
        self.assertTrue(self.host.is_stale(shot.shot_id, "Door", "highlight"))

    def test_build_releases_the_keys_of_a_dropped_behavior(self):
        """Bug: untick a behavior (or drop it from the sheet) and Build left
        its keys in the scene for good, still claimed as the manifest's.
        Fixed: 2026-10-03
        """
        shot = self._build(self._steps("show", "highlight"))
        self.assertEqual(self.host.curves["Door.highlight"], {100.0: "highlight"})

        steps = self._steps("show")
        step, _door = self._door(steps)
        self.assertEqual(step.dropped_behaviors, [["Door", "highlight"]])
        self.assertTrue(step.needs_build)

        self._build(steps)
        self.assertEqual(self.host.curves["Door.highlight"], {})
        led = self.store.edit_ledger
        self.assertEqual(led.authored(owner=shot.shot_id, behavior="highlight"), [])
        self.assertEqual(len(led.authored(owner=shot.shot_id, behavior="show")), 1)
        self.assertFalse(self._door(steps)[0].needs_build)

    def test_an_object_dropped_from_its_step_releases_too(self):
        self._build(self._steps("highlight"))
        self._build(self._steps("highlight", name="Lid"))
        self.assertEqual(self.host.curves["Door.highlight"], {})
        self.assertEqual(self.host.curves["Lid.highlight"], {100.0: "highlight"})

    def test_a_locked_shot_keeps_its_dropped_keys(self):
        shot = self._build(self._steps("highlight"))
        self.store.update_shot(shot.shot_id, locked=True)
        self._build(self._steps())
        self.assertEqual(self.host.curves["Door.highlight"], {100.0: "highlight"})

    def test_a_removed_shots_keys_never_pass_to_the_shot_that_takes_its_id(self):
        """Bug: removing a shot (the panel's orphan removal, Delete All Shots)
        left its behavior claims on its id; the next shot built was given that
        id, and its Build deleted the removed shot's keys as behaviors its doc
        had dropped.
        Fixed: 2026-10-04
        """
        from pythontk.core_utils.engines.shots.shot_ledger import NO_OWNER

        def step(sid, name, behavior):
            return BuilderStep(
                sid, "A", "", "", objects=[BuilderObject(name, [behavior])]
            )

        door, lid = step("A01", "Door", "show"), step("A02", "Lid", "hide")
        self.host.sync(
            [door, lid], ranges={"A01": (100.0, 300.0), "A02": (300.0, 500.0)}
        )
        removed = self.store.shot_by_name("A02")
        self.assertEqual(self.host.curves["Lid.hide"], {500.0: "hide"})
        self.store.remove_shot(removed.shot_id)  # the doc dropped A02; keys stay

        gate = step("A03", "Gate", "show")
        self.host.sync([door, gate], remove_missing=False)
        self.assertEqual(self.store.shot_by_name("A03").shot_id, removed.shot_id)
        self.assertEqual(self.host.curves["Lid.hide"], {500.0: "hide"})
        self.assertEqual(
            self.store.edit_ledger.authored(owner=NO_OWNER), [("Lid.hide", 500.0)]
        )
        self.assertEqual(self.host.assess([door, gate])[1].dropped_behaviors, [])

    def test_a_stale_behavior_over_animator_keys_is_a_conflict(self):
        """Bug: a fade keyed under an older recipe, with an animator key on
        its channel inside the shot, read "stale -- Build re-keys it"; Build
        leaves a behavior over keys it does not own, so the step asked for a
        Build that never changed it.
        Fixed: 2026-10-04
        """

        class Host(self._Host):
            """Guards a build as both DCC hosts do."""

            def apply_behaviors(self):
                return beh.Behaviors.apply_to_shots(
                    self.store.sorted_shots(),
                    apply_fn=self._key,
                    conflict_fn=lambda node, b, s, e: bool(
                        self.unowned_keys(node, b, s, e)
                    ),
                    release_fn=lambda shot, name, b: self.release_authored(
                        shot.shot_id, name, b
                    ),
                )

            def _key_samples(self, obj, behavior, start, end):
                curve = f"{obj}.{behavior}"
                times = sorted(self.curves.get(curve, {}))
                return [(curve, t) for t in times if start <= t <= end]

        self.host = Host(self.store)
        steps = self._steps("show")
        self._build(steps)
        self.store.update_effect_recipe(fade_frames=30)
        self.assertEqual(self._door(steps)[1].status, "stale_behavior")

        self.host.curves["Door.show"][200.0] = "animator"
        step, door = self._door(steps)
        self.assertEqual(door.status, "behavior_conflict")
        self.assertEqual(door.stale_behaviors, [])
        self.assertEqual(door.conflict_behaviors, ["show"], "a stale one in conflict is named too")
        self.assertFalse(step.needs_build)
        # Build agrees: it leaves the fade and the animator's key alone.
        _actions, result, _status = self.host.sync(
            steps, ranges={"A01": (100.0, 300.0)}
        )
        self.assertEqual(
            [(r["object"], r["behavior"]) for r in result["skipped"]],
            [("Door", "show")],
        )
        self.assertEqual(
            self.host.curves["Door.show"], {100.0: "show", 200.0: "animator"}
        )

    def test_apply_releases_what_the_doc_dropped_for_the_object(self):
        shot = self._build(self._steps("show", "highlight"))
        self.host.reapply_object(shot, BuilderObject("Door", ["show"]))
        self.assertEqual(self.host.curves["Door.highlight"], {})
        self.assertEqual(self.host.curves["Door.show"], {100.0: "show"})

    def test_a_placed_clip_nobody_claimed_is_adopted(self):
        """A build no longer clears a whole track before placing a clip, so a
        clip a pre-claim build placed is adopted -- or, once its shot moved,
        it would play at its old place too."""

        class Host(self._Host):
            def _placed_clip_keys(self, name, start, end):
                return [("vo_track", start), ("vo_track", start + 40.0)]

        host = Host(self.store)
        shot = self.store.define_shot("A01", 100.0, 300.0)
        self.store.update_shot(
            shot.shot_id,
            metadata={
                "behaviors": [{"name": "vo", "behavior": "set_clip", "kind": "audio"}]
            },
        )
        self.assertEqual(host.adopt_placed_clips(), 2)
        self.assertEqual(
            self.store.edit_ledger.authored(owner=shot.shot_id, obj="vo"),
            [("vo_track", 100.0), ("vo_track", 140.0)],
        )
        self.assertEqual(host.adopt_placed_clips(), 0)  # claimed now

    def test_the_audio_pass_releases_its_clip_before_placing_it(self):
        calls = []
        shot = self.store.define_shot("A01", 100.0, 300.0)
        self.store.update_shot(
            shot.shot_id,
            metadata={
                "behaviors": [{"name": "vo", "behavior": "set_clip", "kind": "audio"}]
            },
        )

        def place(*_args, **_kwargs):
            calls.append("apply")
            return [("vo_track", 100.0)]

        result = beh.Behaviors.apply_to_shots(
            self.store.sorted_shots(),
            apply_fn=place,
            release_fn=lambda *a: calls.append("release"),
        )
        self.assertEqual(calls, ["release", "apply"])
        self.assertEqual(result["applied"][0]["keys"], [("vo_track", 100.0)])
        # Already placed: left alone, nothing released.
        calls.clear()
        beh.Behaviors.apply_to_shots(
            self.store.sorted_shots(),
            apply_fn=place,
            has_keys_fn=lambda *a: True,
            release_fn=lambda *a: calls.append("release"),
        )
        self.assertEqual(calls, [])

    def test_a_behavior_whose_release_failed_is_not_keyed_again(self):
        """Bug: the audio pass skipped a clip its old keys could not be
        released for, but the pass for every other behavior keyed over the old
        keys anyway -- reported failed AND applied.
        Fixed: 2026-10-04
        """
        shot = self.store.define_shot("A01", 100.0, 300.0)
        entries = [
            {"name": name, "behavior": "show", "kind": "scene"}
            for name in ("door", "lid")
        ]
        self.store.update_shot(shot.shot_id, metadata={"behaviors": entries})

        def release(_shot, name, _behavior):
            if name == "door":
                raise RuntimeError("curve is locked")

        with self.assertLogs(beh.__name__, "WARNING"):
            result = beh.Behaviors.apply_to_shots(
                self.store.sorted_shots(),
                apply_fn=lambda node, b, s, e, **_kw: [(f"{node}.v", s)],
                release_fn=release,
            )
        self.assertEqual(
            [(r["object"], r["behavior"]) for r in result["failed"]],
            [("door", "show")],
        )
        self.assertEqual([r["object"] for r in result["applied"]], ["lid"])

    def test_a_highlight_sizes_its_shot_to_hold_a_beat(self):
        """The pulse keys from the recipe, so a shot sized to fit one holds
        its leads and one whole cycle -- at the SCENE's rate."""
        step = self._steps("highlight")[0]
        recipe = self.store.effect_recipe
        dur, _beh, _aud = ShotManifest.resolve_duration(
            step, 0.0, "fit_contents", 24.0, recipe=recipe
        )
        self.assertAlmostEqual(dur, (0.72 + 2.86 + 0.72) * 24.0)
        longer = recipe.replace(pulse_period=4.0)
        dur, _beh, _aud = ShotManifest.resolve_duration(
            step, 0.0, "fit_contents", 24.0, recipe=longer
        )
        self.assertAlmostEqual(dur, (0.72 + 4.0 + 0.72) * 24.0)

    def test_effect_templates_validate(self):
        from pythontk.core_utils.engines.shots.manifest.behaviors import BehaviorSpec

        self.assertTrue(BehaviorSpec.validate({"effect": "pulse", "place": "span"}).ok)
        self.assertFalse(BehaviorSpec.validate({"effect": "glow"}).ok)
        self.assertFalse(BehaviorSpec.validate({"place": 2.0}).ok)
        for name in beh.Behaviors.list_behaviors():
            tmpl = beh.Behaviors.load_behavior(name)
            self.assertTrue(BehaviorSpec.validate(tmpl).ok, name)
            self.assertIsNotNone(beh.Behaviors.effect_of(tmpl), name)


class TestContentEndPlacesLikeTheBuild(unittest.TestCase):
    """Assess's unused tail (``StepStatus.shrinkable_frames``) is measured where
    the build keys: an object's point behaviors spread over the range
    (``Behaviors.anchor_overrides``, the one placement rule). Two fade-ins key
    the second at the shot's END; sized as if both sat at the start, the tail
    read as unused -- shrinkable over the very frames the keys are on
    (2026-10-12)."""

    def test_a_spread_point_behavior_ends_the_content_where_it_is_keyed(self):
        from types import SimpleNamespace

        names = ["show", "show"]
        step = BuilderStep("A01", "A", "t", "", [BuilderObject("g", behaviors=names)])
        status = [ObjectStatus("g", True, "valid", behaviors=list(names))]
        scene = SimpleNamespace(start=0.0, end=100.0)
        recipe = EffectRecipe()

        end = ShotManifest._compute_content_end(step, scene, status, recipe)

        last = max(
            beh.Behaviors.key_window(name, 0.0, 100.0, anchor, recipe)[1]
            for name, anchor in zip(names, beh.Behaviors.anchor_overrides(names))
        )
        self.assertEqual(last, 100.0)  # the second fade-in keys at the end
        self.assertEqual(end, last)

    def test_an_animated_object_with_a_behavior_ends_the_content_at_its_keys(self):
        """A drain pan that fades in AND slides until frame 130 of a 200-frame
        shot leaves a 70-frame tail. Assess queried the animator's keys only
        for an object with no behavior, so the tail read as 185 frames (all
        but the fade). Fixed: 2026-11-08"""
        store = ShotStore()
        store.define_shot("A02", 0, 200, objects=["pan"])
        step = BuilderStep(
            "A02", "A", "t", "", [BuilderObject("pan", behaviors=["show"])]
        )
        [status] = ShotManifest(store).assess(
            [step],
            exists_fn=lambda name: True,
            verify_fn=lambda *args, **kwargs: True,
            keyframe_range_fn=lambda name: (30.0, 130.0),
            skip_scene_discovery=True,
        )
        self.assertEqual(status.shrinkable_frames, 70.0)

    def test_the_hosts_sequencer_ends_the_content_where_a_trim_would(self):
        """With a scene, the unused tail is what Trim Trailing Space would cut
        (``ShotSequencer.content_end``, through the host's ``_sequencer``):
        an object's whole-timeline key extent runs into a later shot (its
        keys to 900), which read as no tail at all. Added: 2026-11-08"""

        class _Scanned(ShotSequencer):
            def content_end(self, shot_id):
                return 150.0

        class _Hosted(ShotManifest):
            def _sequencer(self):
                return _Scanned(store=self.store)

        store = ShotStore()
        store.define_shot("A02", 0, 200, objects=["pan"])
        step = BuilderStep(
            "A02", "A", "t", "", [BuilderObject("pan", behaviors=["show"])]
        )
        [status] = _Hosted(store).assess(
            [step],
            exists_fn=lambda name: True,
            verify_fn=lambda *args, **kwargs: True,
            keyframe_range_fn=lambda name: (30.0, 900.0),
            skip_scene_discovery=True,
        )
        self.assertEqual(status.shrinkable_frames, 50.0)

    def test_content_end_is_where_a_trailing_trim_puts_the_end(self):
        """``ShotSequencer.content_end``: the shot's sequences and its keys
        inside it, snapped up as the trim snaps, changing nothing; ``None``
        for a shot holding nothing the sequencer reads. Added: 2026-11-08"""

        class _Keys(ShotSequencer):
            inner_end = 120.4
            segments = [{"kind": "anim", "obj": "pan", "start": 10.0, "end": 90.0}]

            def collect_shot_sequences(self, shot_id, include_audio=True):
                return list(self.segments)

            def _key_extent(self, shot, probe_outside, reach=None):
                return 10.0, self.inner_end, None, None, []

        store = ShotStore()
        store.define_shot("A02", 0, 200, objects=["pan"])
        seq = _Keys(store=store)
        shot_id = store.shots[0].shot_id
        self.assertEqual(seq.content_end(shot_id), store.snap(120.4, "up"))
        self.assertEqual((store.shots[0].start, store.shots[0].end), (0, 200))
        seq.inner_end, seq.segments = None, []
        self.assertIsNone(seq.content_end(shot_id))
        self.assertIsNone(ShotSequencer(store=store).content_end(shot_id))


class TestInsertInSheetOrder(unittest.TestCase):
    """``plan_insertions`` / ``insert_planned``: on a built timeline a step
    with no shot gets one in sheet order -- after its previous step's shot and
    that shot's split tails -- and every later shot moves out of its way.

    Bug: Build with Hide Missing Shots off laid the unbuilt steps' shots over
    the built ones -- zero-length, overlapping, out of sheet order (SOL:
    37 new shots, dozens of overlaps, 9 zero-length).
    Fixed: 2026-10-12
    """

    def setUp(self):
        self.store = ShotStore()
        self.store.gap = 10.0
        self.store.define_shot("A01", 0, 100)
        self.store.define_shot("A01_2", 110, 150)  # split off A01: no step's
        self.store.define_shot("A03", 160, 260)
        self.mani = ShotManifest(self.store)
        self.steps = [self._step(s) for s in ("A00", "A01", "A02", "A03", "A04")]

    @staticmethod
    def _step(sid):
        return BuilderStep(sid, "A", "t", "", [BuilderObject(f"{sid}_geo")])

    def _layout(self):
        return [(s.name, s.start, s.end) for s in self.store.sorted_shots()]

    def test_new_shots_go_in_sheet_order_and_later_shots_ripple(self):
        plans = self.mani.plan_insertions(self.steps, initial_shot_length=50)
        self.assertEqual([p.step.step_id for p in plans], ["A00", "A02", "A04"])
        projected = [(p.step.step_id, p.start, p.end) for p in plans]
        actions = self.mani.insert_planned(plans)
        self.assertEqual(actions, dict.fromkeys(("A00", "A02", "A04"), "created"))
        self.assertEqual(
            self._layout(),
            [
                ("A00", 0, 50),
                ("A01", 60, 160),
                ("A01_2", 170, 210),  # a split tail stays with its step
                ("A02", 220, 270),
                ("A03", 280, 380),
                ("A04", 390, 440),
            ],
        )
        got = {name: (start, end) for name, start, end in self._layout()}
        for sid, start, end in projected:  # the plan is what the inserts do
            self.assertEqual(got[sid], (start, end), sid)
        self.assertEqual(
            set(self.mani.pair(self.steps).shots), {s.step_id for s in self.steps}
        )

    def test_a_name_the_store_refuses_is_not_planned(self):
        """``a00`` is A00's clip ignoring case: A00 is left for the build to
        report as refused, and the next step still takes the head."""
        self.store.define_shot("a00", 900, 950)
        steps = [self._step("A00"), self._step("A00b"), self._step("A01")]
        plans = self.mani.plan_insertions(steps, initial_shot_length=50)
        self.assertEqual([p.step.step_id for p in plans], ["A00b"])
        self.assertEqual(self.mani.insert_planned(plans), {"A00b": "created"})
        self.assertEqual(self._layout()[0], ("A00b", 0, 50))

    def test_a_typed_step_is_not_inserted(self):
        """A step with a typed range goes there: it is no insert."""
        plans = self.mani.plan_insertions(
            self.steps, typed={"A02": (900.0, 950.0)}, initial_shot_length=50
        )
        self.assertEqual([p.step.step_id for p in plans], ["A00", "A04"])

    def test_a_step_after_a_typed_one_follows_where_that_one_lands(self):
        """A typed step will have its shot at its range -- moved with the
        shots the inserts move -- so the next new step follows it there.

        Bug: X, after P typed past the last shot, appended after the last
        SHOT: projected at (570, 620), over P's (570, 610); Build made both.
        Fixed: 2026-10-13
        """
        store = ShotStore()
        store.gap = 10.0
        for name, start, end in (("Q", 1, 100), ("R", 200, 300), ("S", 400, 500)):
            store.define_shot(name, start, end)
        mani = ShotManifest(store)
        steps = [self._step(s) for s in ("Q", "N", "R", "S", "P", "X")]
        typed = {"P": (510.0, 550.0)}
        plans = mani.plan_insertions(steps, typed=typed, initial_shot_length=50)
        self.assertEqual(
            [(p.step.step_id, p.start, p.end) for p in plans],
            [("N", 110, 160), ("X", 620, 670)],
        )
        self.assertEqual(plans[1].after_step_id, "P", "it follows P's shot")
        ridden = ShotManifest.ride_inserts(typed, plans)
        self.assertEqual(ridden, {"P": (570, 610)})
        mani.insert_planned(plans, typed=typed)
        shots = {sid: (s.start, s.end) for sid, s in mani.pair(steps).shots.items()}
        ranges = rr.RangeResolver.incremental_ranges(steps, shots, user_ranges=ridden)
        mani.sync(
            steps, ranges=ranges, remove_missing=False, zero_duration_fallback=True
        )
        self.assertEqual(
            [(s.name, s.start, s.end) for s in store.sorted_shots()],
            [
                ("Q", 1, 100),
                ("N", 110, 160),
                ("R", 260, 360),
                ("S", 460, 560),
                ("P", 570, 610),
                ("X", 620, 670),
            ],
        )

    def test_a_typed_step_between_shots_is_followed_and_ridden(self):
        """A typed range between two shots: the step after it is inserted
        there, moving the next shot -- and the typed range rides an earlier
        insert like the shots after it."""
        store = ShotStore()
        store.gap = 10.0
        store.define_shot("Q", 1, 100)
        store.define_shot("S", 400, 500)
        mani = ShotManifest(store)
        steps = [self._step(s) for s in ("Q", "N", "P", "X", "S")]
        typed = {"P": (200.0, 250.0)}
        plans = mani.plan_insertions(steps, typed=typed, initial_shot_length=50)
        # N after Q pushes P and S 60; X after P (310, 360) pushes S 60 more.
        self.assertEqual(
            [(p.step.step_id, p.start, p.end, p.push) for p in plans],
            [("N", 110, 160, 60), ("X", 320, 370, 60)],
        )
        self.assertEqual(ShotManifest.ride_inserts(typed, plans), {"P": (260, 310)})
        mani.insert_planned(plans, typed=typed)
        got = {s.name: (s.start, s.end) for s in store.sorted_shots()}
        self.assertEqual(
            got, {"Q": (1, 100), "N": (110, 160), "X": (320, 370), "S": (520, 620)}
        )

    def test_an_unbuilt_timeline_plans_nothing(self):
        """A first build places every step itself."""
        self.assertEqual(ShotManifest(ShotStore()).plan_insertions(self.steps), [])

    def test_an_inserted_shot_carries_what_a_first_build_writes(self):
        """The name a build gave it, and the step's metadata: a hand's rename
        of an inserted shot is an edit, as it is of any built one.

        Bug: ``insert_planned`` bound the step and wrote nothing else, so
        ``edited_by_hand`` (Keep All Edits, the Edits status) never saw an
        inserted shot renamed by hand.
        Fixed: 2026-10-12
        """
        store = ShotStore()
        store.gap = 10.0
        mani = ShotManifest(store)
        a01, a02, a03 = (self._step(s) for s in ("A01", "A02", "A03"))
        mani.sync([a01, a03], initial_shot_length=100)  # the first build
        plans = mani.plan_insertions([a01, a02, a03], initial_shot_length=50)
        self.assertEqual([p.step.step_id for p in plans], ["A02"])
        mani.insert_planned(plans)
        mani.sync([a01, a02, a03], remove_missing=False, zero_duration_fallback=True)
        first, inserted = store.shot_by_name("A01"), store.shot_by_name("A02")
        self.assertEqual(set(inserted.metadata), set(first.metadata))
        self.assertEqual(inserted.metadata[ShotManifest.BUILD_NAME], "A02")
        self.assertEqual(inserted.metadata["step"], "A02")
        self.assertFalse(mani.edited_by_hand(inserted))
        store.update_shot(first.shot_id, name="A01_hand")
        store.update_shot(inserted.shot_id, name="A02_hand")
        self.assertTrue(mani.edited_by_hand(store.shot_by_id(first.shot_id)))
        self.assertTrue(mani.edited_by_hand(store.shot_by_id(inserted.shot_id)))

    def test_content_trailing_a_moved_last_shot_moves_with_it(self):
        """A step appended after the last shot clears that shot's trailing
        content where the earlier inserts left it.

        Bug: the plan took the larger of the moved end and the content's
        unmoved extent: with 30 frames trailing each shot, A04 was planned at
        (280, 330) and landed at (310, 360).
        Fixed: 2026-10-12
        """
        store = ShotStore()
        store.gap = 10.0
        store.define_shot("A01", 0, 100)
        store.define_shot("A03", 110, 210)
        mani = _TrailingManifest(store)
        steps = [self._step(s) for s in ("A01", "A02", "A03", "A04")]
        plans = mani.plan_insertions(steps, initial_shot_length=50)
        planned = {p.step.step_id: (p.start, p.end) for p in plans}
        self.assertEqual(planned, {"A02": (110, 160), "A04": (310, 360)})
        mani.insert_planned(plans)
        got = {s.name: (s.start, s.end) for s in store.sorted_shots()}
        for sid, bounds in planned.items():
            self.assertEqual(got[sid], bounds, sid)

    def test_a_shot_sharing_the_insert_point_keeps_its_place(self):
        """The shots an insert moves are those after it in the timeline's
        order, not every shot starting at its frame.

        Bug: a zero-length shot Z on A03's start frame moved with A03, so A02,
        planned after Z, landed before it: A01, A02, Z, A03.
        Fixed: 2026-10-12
        """
        store = ShotStore()
        store.gap = 10.0
        store.define_shot("A01", 0, 100)
        store.define_shot("Z", 110, 110)  # no step's
        store.define_shot("A03", 110, 210)
        mani = ShotManifest(store)
        steps = [self._step(s) for s in ("A01", "A02", "A03")]
        plans = mani.plan_insertions(steps, initial_shot_length=50)
        self.assertEqual([(p.start, p.end) for p in plans], [(120, 170)])
        mani.insert_planned(plans)
        layout = [(s.name, s.start, s.end) for s in store.sorted_shots()]
        self.assertEqual(
            layout,
            [("A01", 0, 100), ("Z", 110, 110), ("A02", 120, 170), ("A03", 170, 270)],
        )

    def test_part_of_a_plan_keeps_its_place_in_the_sheet(self):
        """``insert_planned`` given some of a plan's inserts puts each where
        the plan says, after the shot its chain follows -- never at the head.

        Bug: an insert following one left out took the timeline's head.
        Fixed: 2026-10-12
        """
        steps = [self._step(s) for s in ("A01", "B01", "B02", "B03", "A03")]
        plans = self.mani.plan_insertions(steps, initial_shot_length=50)
        self.assertEqual([p.step.step_id for p in plans], ["B01", "B02", "B03"])
        self.assertEqual(
            [p.after_step_id for p in plans], [None, "B01", "B02"], "the chain"
        )
        self.assertEqual(
            self.mani.insert_planned(plans[1:]),
            dict.fromkeys(("B02", "B03"), "created"),
        )
        names = [name for name, _start, _end in self._layout()]
        self.assertEqual(names, ["A01", "A01_2", "B02", "B03", "A03"])

    def test_every_plan_is_where_its_shot_lands(self):
        """A step inserted before an earlier insert (the timeline holds its
        shots out of sheet order) moves that one: each plan says where its
        shot ends up once every insert is in."""
        store = ShotStore()
        store.gap = 10.0
        store.define_shot("S3", 0, 100)
        store.define_shot("S1", 110, 210)
        mani = ShotManifest(store)
        steps = [self._step(s) for s in ("S1", "S2", "S3", "S4")]
        plans = mani.plan_insertions(steps, initial_shot_length=50)
        planned = {p.step.step_id: (p.start, p.end) for p in plans}
        mani.insert_planned(plans)
        got = {s.name: (s.start, s.end) for s in store.sorted_shots()}
        self.assertEqual(planned, {"S2": got["S2"], "S4": got["S4"]})
        self.assertEqual(got["S2"], (280, 330))

    def test_a_typed_range_rides_the_inserts_before_it(self):
        """A range typed for an unbuilt step is in the timeline's current
        frames, so it moves with every shot an earlier insert pushes
        (maintainer 2026-10-12).

        Bug: the typed A04 (220, 270) was placed as typed, over A03, which
        A02's insert had pushed to [170, 270].
        Fixed: 2026-10-12
        """
        store = ShotStore()
        store.gap = 10.0
        store.define_shot("A01", 0, 100)
        store.define_shot("A03", 110, 210)
        mani = ShotManifest(store)
        steps = [self._step(s) for s in ("A01", "A02", "A03", "A04")]
        typed = {"A04": (220.0, 270.0)}
        plans = mani.plan_insertions(steps, typed=typed, initial_shot_length=50)
        self.assertEqual([(p.pivot, p.push) for p in plans], [(110, 60)])
        ridden = ShotManifest.ride_inserts(typed, plans)
        self.assertEqual(ridden, {"A04": (280, 330)})
        self.assertEqual(typed, {"A04": (220.0, 270.0)}, "the input is left")
        mani.insert_planned(plans)
        shots = {sid: (s.start, s.end) for sid, s in mani.pair(steps).shots.items()}
        ranges = rr.RangeResolver.incremental_ranges(steps, shots, user_ranges=ridden)
        mani.sync(
            steps, ranges=ranges, remove_missing=False, zero_duration_fallback=True
        )
        layout = [(s.name, s.start, s.end) for s in store.sorted_shots()]
        self.assertEqual(
            layout,
            [("A01", 0, 100), ("A02", 110, 160), ("A03", 170, 270), ("A04", 280, 330)],
        )


class _TrailingSequencer(ShotSequencer):
    """A bounds-only sequencer whose shots that had content (*keyed*: their
    ids) trail it 30 frames past their end; a new shot has none."""

    def __init__(self, keyed, **kwargs):
        super().__init__(**kwargs)
        self.keyed = keyed

    def _trailing_content_extent(self, shot) -> float:
        return shot.end + (30 if shot.shot_id in self.keyed else 0)


class _TrailingManifest(ShotManifest):
    """A manifest over :class:`_TrailingSequencer`: the shots its store holds
    now carry the content."""

    def __init__(self, store, **kwargs):
        super().__init__(store, **kwargs)
        self.keyed = {shot.shot_id for shot in store.shots}

    def _sequencer(self):
        return _TrailingSequencer(self.keyed, store=self.store)


class TestInsertPlanMatchesTheInserts(unittest.TestCase):
    """``plan_insertions`` and ``ShotSequencer.insert_shot`` place a shot by
    one rule (``ShotSequencer.insert_placement``): on any timeline -- shots
    out of sheet order, zero-length shots, shots sharing a start (overlapping
    ones moved by the frame rule), content trailing past a shot's end, typed
    ranges the inserts follow and move -- every plan is where its insert
    lands, and a typed range where ``ride_inserts`` says."""

    def _step(self, sid):
        return BuilderStep(sid, "A", "t", "", [BuilderObject(f"{sid}_geo")])

    def test_on_random_timelines(self):
        import random

        rng = random.Random(20261012)
        for case in range(150):
            store = ShotStore()
            store.gap = rng.choice((0.0, 5.0, 10.0))
            frame, names = 0.0, []
            for i in range(rng.randint(1, 6)):
                length = rng.choice((0.0, 20.0, 50.0))
                store.define_shot(f"S{i}", frame, frame + length)
                names.append(f"S{i}")
                if rng.random() < 0.2:
                    continue  # the next shares this start: an overlap tie
                frame += length + rng.choice((0.0, 0.0, 10.0))
            rng.shuffle(names)
            sheet, typed = [], {}
            for i, name in enumerate(names):
                if rng.random() < 0.8:  # some shots no step pairs with
                    sheet.append(name)
                for j in range(rng.randint(0, 2)):
                    sid = f"N{i}_{j}"
                    sheet.append(sid)
                    if rng.random() < 0.25:  # off the shots' frames: no tie
                        at = rng.randrange(0, int(frame) + 60, 10) + 3.0
                        typed[sid] = (at, at + rng.choice((0.0, 15.0)))
            steps = [self._step(sid) for sid in sheet]
            cls = _TrailingManifest if rng.random() < 0.5 else ShotManifest
            mani = cls(store)
            plans = mani.plan_insertions(steps, typed=typed, initial_shot_length=40)
            planned = {p.step.step_id: (p.start, p.end) for p in plans}
            order_before = [s.name for s in store.sorted_shots()]
            mani.insert_planned(plans, typed=typed)
            got = {s.name: (s.start, s.end) for s in store.sorted_shots()}
            with self.subTest(case=case, sheet=sheet, typed=typed, before=order_before):
                for sid, bounds in planned.items():
                    self.assertEqual(got[sid], bounds, sid)


class TestManifestPlacement(unittest.TestCase):
    """``ManifestPlacement.first_pose``: the plain first pose of a helper a
    placing row is given where the scene has none -- read from bounds alone.
    The cases each host's making of a helper is held to."""

    BOX = ((-10.0, 100.0, 40.0), (10.0, 120.0, 60.0))  # a part 20 cm across, 1.1 m up, half a metre out
    ROOM = ((-100.0, 0.0, -100.0), (100.0, 250.0, 100.0))  # the delivery around it, its floor at 0

    def setUp(self):
        from pythontk.core_utils.engines.shots.manifest.manifest_placement import ManifestPlacement

        self.P = ManifestPlacement

    def test_a_camera_stands_on_the_outer_side_far_enough_to_frame_its_part_looking_at_it(self):
        import math

        position, aim = self.P.first_pose("view", self.BOX, self.ROOM, unit=100.0)
        self.assertEqual(aim, (0.0, 110.0, 50.0), "it looks at the part's middle")
        radius = math.sqrt(3 * 10.0 ** 2)
        distance = radius * self.P.MARGIN / math.sin(math.radians(self.P.VIEW_ANGLE) / 2)
        # Toward the eyes of the part's viewpoint (out of the delivery), from a little above: looking down onto it.
        rise = math.radians(10.0)
        self.assertEqual(position[0], 0.0, "straight out")
        self.assertAlmostEqual(position[1], 110.0 + distance * math.sin(rise), places=6)
        self.assertAlmostEqual(position[2], 50.0 + distance * math.cos(rise), places=6, msg="at the distance that frames it")

    def test_a_tiny_part_is_framed_with_what_it_sits_on(self):
        dot = ((-0.5, 110.0, 49.5), (0.5, 111.0, 50.5))  # a centimetre across
        position, _ = self.P.first_pose("view", dot, self.ROOM, unit=100.0)
        self.assertGreater(position[2] - 50.0, 30.0, "never a macro of a dot: the least it frames is a 6 cm radius")

    def test_a_viewpoint_stands_on_the_floor_the_stand_off_outside_its_part_facing_it_level(self):
        position, aim = self.P.first_pose("stand", self.BOX, self.ROOM, unit=100.0)
        self.assertEqual(position, (0.0, 0.0, 50.0 + 10.0 + self.P.STAND_OFF * 100.0), "on the floor, outside the part's side")
        self.assertEqual(aim, (0.0, 0.0, 50.0), "facing it, level")
        # A part's own group may start above the floor (a cabinet on feet, the room another group): a host that knows
        # the scene's floor says it, and the viewpoint stands there.
        raised = ((-100.0, 24.0, -100.0), (100.0, 250.0, 100.0))
        self.assertEqual(self.P.first_pose("stand", self.BOX, raised, unit=100.0)[0][1], 24.0, "the delivery's lowest point, with nothing said")
        position, aim = self.P.first_pose("stand", self.BOX, raised, unit=100.0, floor=0.0)
        self.assertEqual((position[1], aim[1]), (0.0, 0.0), "the scene's floor, where the host says it")

    def test_an_arrows_tip_rests_just_above_its_part_pointing_down_at_it(self):
        position, aim = self.P.first_pose("point", self.BOX, self.ROOM, unit=100.0)
        self.assertEqual(position, (0.0, 120.0 + self.P.GAP * 100.0, 50.0))
        self.assertEqual(aim, (0.0, 110.0, 50.0))

    def test_the_up_axis_and_the_unit_are_the_scenes(self):
        # A Z-up scene in metres: the same part, the same room.
        box = ((-0.1, -0.6, 1.0), (0.1, -0.4, 1.2))
        room = ((-1.0, -1.0, 0.0), (1.0, 1.0, 2.5))
        position, aim = self.P.first_pose("stand", box, room, up=2, unit=1.0)
        self.assertEqual((position[0], position[2]), (0.0, 0.0), "on the floor, Z up")
        self.assertAlmostEqual(position[1], -0.5 - 0.1 - self.P.STAND_OFF, places=6, msg="out of the room on the part's side")
        self.assertEqual(aim, (0.0, -0.5, 0.0))

    def test_a_part_in_the_middle_of_its_delivery_is_seen_from_the_level_axis_after_up(self):
        middle = ((-10.0, 100.0, -10.0), (10.0, 120.0, 10.0))
        position, _ = self.P.first_pose("view", middle, self.ROOM, unit=100.0)
        self.assertGreater(position[2], 10.0)
        self.assertEqual(position[0], 0.0)
        self.assertEqual(self.P.outer(middle, None), (0.0, 0.0, 1.0), "with no delivery: the same")
        # A cabinet's body, a latch on its front pulling the delivery's middle 2 cm forward: the body stands over that
        # middle, so no side of it is outer and the hair's breadth decides nothing (a viewpoint stood behind it).
        body = ((-50.0, 0.0, -30.0), (50.0, 200.0, 30.0))
        whole = ((-50.0, 0.0, -30.0), (50.0, 200.0, 34.0))
        self.assertEqual(self.P.outer(body, whole), (0.0, 0.0, 1.0))

    def test_a_rows_own_helper_is_named_by_its_purpose_step_and_target_alike_in_every_tool(self):
        self.assertEqual(self.P.helper_name("view", "B02", "ROOT|CABINET|DOOR_GRP"), "CAM_B02_DOOR_GRP")
        self.assertEqual(self.P.helper_name("stand", "C05A", "SCREWS_GRP"), "VIEW_C05A_SCREWS_GRP")
        self.assertEqual(self.P.helper_name("point", "B03.5", "panel light #2"), "POINT_B03_5_panel_light_2", "a legal name")
        self.assertNotEqual(self.P.helper_name("view", "C09", "DIALS"), self.P.helper_name("view", "C09", "LIGHT"),
                            "two pictures in a step: two names")
        # Bug: only the leaf was kept, so two targets sharing it in a step named one helper. The name keeps the
        # end of the target's path that tells it apart from the step's other targets (Fixed: 2026-11-26).
        among = ["PANEL|LATCH", "DOOR|LATCH", "LATCH", "ROOT|CABINET|DOOR_GRP"]
        self.assertEqual([self.P.helper_name("view", "A01", t, among) for t in among],
                         ["CAM_A01_PANEL_LATCH", "CAM_A01_DOOR_LATCH", "CAM_A01_LATCH", "CAM_A01_DOOR_GRP"])
        for bad in (("hold", "A01", "X"), ("view", "", "X"), ("view", "A01", "")):
            with self.assertRaises(ValueError):
                self.P.helper_name(*bad)

    def test_a_held_parts_place_is_its_takes_and_no_helper(self):
        with self.assertRaises(ValueError):
            self.P.first_pose("hold", self.BOX, self.ROOM)
        self.assertEqual(self.P.KINDS, {"stand": "locator", "point": "locator", "view": "camera"})
        self.assertAlmostEqual(self.P.focal_length(24.0), 44.785, places=2, msg="30 degrees over a 24 mm high film")


class TestThePlacingRowsHelpers(_EngineCase):
    """``ShotManifest.place_helpers``: a build makes the helper a placing row
    names in ``where`` where the scene has none, and never touches one it has.
    The real task's viewpoints, its arrow and its pictures' cameras were made
    by hand, or left to the tool that runs the steps."""

    class _Manifest(ShotManifest):
        def __init__(self, store, have=(), **kw):
            super().__init__(store, **kw)
            self.have, self.made = set(have), []

        def _object_exists(self, name):
            return name in self.have

        def _make_helper(self, name, purpose, target, frame):
            if target == "GONE":
                return None
            self.made.append((name, purpose, target, frame))
            return name

    @staticmethod
    def _rows(sid, *rows):
        return BuilderStep(sid, "A", "t", "", rows=[dict(r) for r in rows])

    def test_a_row_read_by_cells_carries_what_its_word_places_for(self):
        import pythontk

        starter = Path(pythontk.__file__).parent / "core_utils/engines/shots/manifest/mapping/starters/interactive.csv"
        steps = mapping_mod.Mapping.resolve(str(starter), name="interactive")
        placing = {r["word"]: r["places"] for s in steps for r in s.rows if r.get("places")}
        self.assertEqual(placing, {"travel": "stand", "indicate": "point", "picture": "view"},
                         "as the template declares each word (its 'places'), and nothing else places")

    def test_two_rows_of_one_placing_word_and_target_in_a_step_are_told_apart_only_by_a_where(self):
        from pythontk.core_utils.engines.shots.manifest.manifest_actions import ManifestActions

        actions = {"picture": {"places": "view", "settings": {"where": {"type": "object"}}}, "highlight": {}}
        rows = [{"word": "picture", "assets": ["DIALS"], "settings": {}},
                {"word": "picture", "assets": ["DIALS"], "settings": {}},
                {"word": "picture", "assets": ["DIALS"], "settings": {"where": "CAM_WIDE"}},
                {"word": "highlight", "assets": ["DIALS"], "settings": {}},
                {"word": "highlight", "assets": ["DIALS"], "settings": {}}]
        found = [(n, issue[1]) for n, issue, warn in ManifestActions.check_step(rows, actions, {}, "A01", ["A01"]) if not warn]
        self.assertEqual([n for n, _ in found], [1], "the second unnamed picture of one target; a named one, or a word that places nothing, is fine")
        self.assertIn("where=<name>", found[0][1])

    def test_the_check_keys_on_the_helpers_name_so_two_rows_sharing_one_are_said(self):
        """Bug: the check keyed on the Target as written while the helper kept
        only its leaf, so ``picture PANEL|LATCH`` and ``picture DOOR|LATCH`` passed and
        a build made one camera for both, silent of the other.
        Fixed: 2026-11-26"""
        from pythontk.core_utils.engines.shots.manifest.manifest_actions import ManifestActions

        actions = {"picture": {"places": "view"}, "snap": {"places": "view"}}
        told = [{"word": "picture", "assets": ["PANEL|LATCH"], "settings": {}},
                {"word": "picture", "assets": ["DOOR|LATCH"], "settings": {}}]
        self.assertEqual(ManifestActions.check_step(told, actions, {}, "A01", ["A01"]), [], "two targets, two helpers")
        shared = [{"word": "picture", "assets": ["LATCH"], "settings": {}},
                  {"word": "snap", "assets": ["LATCH"], "settings": {}}]
        found = [(n, issue[1]) for n, issue, _ in ManifestActions.check_step(shared, actions, {}, "A01", ["A01"])]
        self.assertEqual([n for n, _ in found], [1], "two words of one purpose would be one helper")
        self.assertIn("CAM_A01_LATCH", found[0][1])

    def test_a_row_that_names_no_helper_gets_its_own_by_the_derived_name(self):
        steps = [self._rows("A01", {"word": "picture", "places": "view", "assets": ["PANEL|LATCH"], "settings": {}},
                            {"word": "picture", "places": "view", "assets": ["DIALS"], "settings": {}},
                            {"word": "travel", "places": "stand", "assets": ["DOOR"], "settings": {"where": "VIEW_FRONT"}})]
        mani = self._Manifest(self.store)
        self.assertEqual(mani.place_helpers(steps)["made"], ["CAM_A01_LATCH", "CAM_A01_DIALS", "VIEW_FRONT"],
                         "two pictures in one step: two cameras; a named one keeps its name")

    def test_two_targets_sharing_a_leaf_get_a_helper_each_and_a_name_two_rows_share_is_said(self):
        """Bug: ``place_helpers`` skipped a helper name it had seen, unsaid:
        ``picture PANEL|LATCH`` and ``picture DOOR|LATCH`` were one camera.
        Fixed: 2026-11-26"""
        steps = [self._rows("A01", {"word": "picture", "places": "view", "assets": ["PANEL|LATCH"], "settings": {}},
                            {"word": "picture", "places": "view", "assets": ["DOOR|LATCH"], "settings": {}},
                            {"word": "picture", "places": "view", "assets": ["DIALS"], "settings": {}},
                            {"word": "picture", "places": "view", "assets": ["DIALS"], "settings": {}},
                            {"word": "travel", "places": "stand", "assets": ["DOOR"], "settings": {"where": "VIEW_FRONT"}},
                            {"word": "travel", "places": "stand", "assets": ["LID"], "settings": {"where": "VIEW_FRONT"}})]
        placed = self._Manifest(self.store).place_helpers(steps)
        self.assertEqual(placed["made"], ["CAM_A01_PANEL_LATCH", "CAM_A01_DOOR_LATCH", "CAM_A01_DIALS", "VIEW_FRONT"])
        self.assertEqual([n for n, _ in placed["skipped"]], ["CAM_A01_DIALS"], "a name two rows derive is said; one written twice is one object")
        self.assertIn("where=<name>", placed["skipped"][0][1])

    def test_a_build_makes_the_helper_a_placing_row_names_and_keeps_one_the_scene_has(self):
        steps = [
            self._rows("A01",
                       {"word": "travel", "places": "stand", "assets": ["DOOR"], "settings": {"where": "VIEW_A01"}},
                       {"word": "picture", "places": "view", "assets": ["LATCH"], "settings": {"where": "CAM_A01", "until": "A02"}},
                       {"word": "indicate", "places": "point", "assets": ["LATCH"], "settings": {}},  # names none: its own, POINT_A01_LATCH
                       {"word": "highlight", "places": "", "assets": ["LATCH"], "settings": {}}),
            self._rows("A02",
                       {"word": "travel", "places": "stand", "assets": ["DOOR"], "settings": {"where": "VIEW_A01"}},  # the same viewpoint again
                       {"word": "indicate", "places": "point", "assets": ["BOLT"], "settings": {"where": "POINT_BOLT"}},
                       {"word": "picture", "places": "view", "assets": ["GONE"], "settings": {"where": "CAM_A02"}},
                       {"word": "picture", "places": "view", "assets": [], "settings": {"where": "CAM_NONE"}}),
        ]
        mani = self._Manifest(self.store, have={"POINT_BOLT"})
        placed = mani.place_helpers(steps)
        self.assertEqual(placed["made"], ["VIEW_A01", "CAM_A01", "POINT_A01_LATCH"], "each made once, under the name its row gives")
        self.assertEqual([(n, p, t) for n, p, t, _ in mani.made][:2], [("VIEW_A01", "stand", "DOOR"), ("CAM_A01", "view", "LATCH")])
        self.assertEqual(placed["kept"], ["POINT_BOLT"], "one the scene has is the truth: never made again, never moved")
        self.assertEqual([name for name, _ in placed["skipped"]], ["CAM_A02", "CAM_NONE"], "what could not be made is said, with why")

        mani.have.update(placed["made"])
        again = mani.place_helpers(steps)
        self.assertEqual((again["made"], sorted(again["kept"])), ([], ["CAM_A01", "POINT_A01_LATCH", "POINT_BOLT", "VIEW_A01"]),
                         "a second build makes nothing")

    def test_a_word_turned_off_places_nothing_and_a_sync_reports_what_it_placed(self):
        steps = [self._rows("A01", {"word": "picture", "places": "view", "assets": ["LATCH"], "settings": {"where": "CAM_A01"}},
                            {"word": "travel", "places": "stand", "assets": ["DOOR"], "settings": {"where": "VIEW_A01"}})]
        off = self._Manifest(self.store, disabled_behaviors=["picture"])
        self.assertEqual(off.place_helpers(steps)["made"], ["VIEW_A01"], "the picture's word is off: its camera is left to the tool that runs the steps")
        _, result, _ = self._Manifest(self.store).sync(steps)
        self.assertEqual(result["placed"]["made"], ["CAM_A01", "VIEW_A01"], "in the order its rows name them")
        self.assertEqual(ShotManifest(self.store).place_helpers(steps), {"made": [], "kept": ["CAM_A01", "VIEW_A01"], "skipped": []},
                         "with no scene (the pure engine) nothing is missing, so nothing is made")


class TheStatesARowNames(_EngineCase):
    """``ShotManifest.set_states``: a build makes sure what a row that sets a
    state is about has that state, so the delivery carries it.  The real
    task's indicators were given theirs by a script written for that task."""

    class _Manifest(ShotManifest):
        def __init__(self, store, have=None, **kw):
            super().__init__(store, **kw)
            self.have = dict(have or {})  # target -> its states; a target with none has no group

        def _ensure_state(self, target, state):
            if target not in self.have:
                return "skipped", "no emissive group on it"
            if not self.have[target]:
                self.have[target] = ["off", "green", "amber"]  # the tool's defaults
                return ("given", "") if state in self.have[target] else ("skipped", "its states do not include it")
            return ("kept", "") if state in self.have[target] else ("skipped", "its states do not include it")

    @staticmethod
    def _rows(sid, *rows):
        return BuilderStep(sid, "A", "t", "", rows=[dict(r) for r in rows])

    def test_a_build_gives_each_set_rows_target_its_states_and_says_what_it_could_not(self):
        steps = [
            self._rows("A01",
                       {"word": "set", "assets": ["LED_A"], "settings": {"state": "off"}},
                       {"word": "set", "assets": ["LED_B"], "settings": {"state": "amber"}},
                       {"word": "highlight", "assets": ["LED_A"], "settings": {}}),
            self._rows("A02",
                       {"word": "set", "assets": ["LED_A"], "settings": {"state": "green"}},
                       {"word": "set", "assets": ["LED_A"], "settings": {"state": "purple"}},
                       {"word": "set", "assets": ["PLAIN_PART"], "settings": {"state": "off"}}),
        ]
        mani = self._Manifest(self.store, have={"LED_A": [], "LED_B": ["off", "amber"]})
        states = mani.set_states(steps)
        self.assertEqual(states["given"], [("LED_A", "off")], "a group with none takes the tool's defaults")
        self.assertEqual(states["kept"], [("LED_B", "amber"), ("LED_A", "green")], "states that are there are left as they are")
        self.assertEqual([(t, s) for t, s, _ in states["skipped"]], [("LED_A", "purple"), ("PLAIN_PART", "off")],
                         "a state its states lack, and a part with no states to give: said, never guessed")
        self.assertTrue(all(why for _t, _s, why in states["skipped"]))

    def test_a_word_turned_off_sets_nothing_and_the_pure_engine_leaves_it_to_the_tool_that_runs_the_steps(self):
        steps = [self._rows("A01", {"word": "set", "assets": ["LED_A"], "settings": {"state": "off"}})]
        off = self._Manifest(self.store, have={"LED_A": []}, disabled_behaviors=["set"])
        self.assertEqual(off.set_states(steps), {"given": [], "kept": [], "skipped": []})
        _, result, _ = self._Manifest(self.store, have={"LED_A": []}).sync(steps)
        self.assertEqual(result["states"]["given"], [("LED_A", "off")], "a build reports the states it gave")
        pure = ShotManifest(self.store).set_states(steps)
        self.assertEqual([(t, s) for t, s, _ in pure["skipped"]], [("LED_A", "off")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
