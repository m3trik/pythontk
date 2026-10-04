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

from pythontk.core_utils.engines.shots.manifest.manifest_model import (
    ManifestModel,
    ColumnMap,
    BuilderStep,
    BuilderObject,
    StepStatus,
    ObjectStatus,
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
from pythontk.core_utils.engines.shots.shot_model import ShotStore
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

    def test_behaviors_detected_from_description(self):
        steps = ManifestModel.parse_csv(self.path)
        a01 = steps[0]
        self.assertEqual(a01.objects[0].behaviors, ["fade_in", "fade_out"])

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
            self.assertEqual(objs[0].behaviors, ["fade_in", "fade_out"])
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
            self.assertEqual(objs[1].behaviors, ["fade_in"])
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
            self.assertEqual(objs[1].behaviors, ["fade_in"])
            self.assertEqual(objs[2].behaviors, [])
        finally:
            os.remove(path)

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
    def test_fade_in(self):
        self.assertEqual(
            ManifestModel.detect_behaviors("the panel fades in slowly"), ["fade_in"]
        )

    def test_fade_out(self):
        self.assertEqual(
            ManifestModel.detect_behaviors("then it fades out"), ["fade_out"]
        )

    def test_both_independent(self):
        self.assertEqual(
            ManifestModel.detect_behaviors("fades in then fades out"),
            ["fade_in", "fade_out"],
        )

    def test_none(self):
        self.assertEqual(ManifestModel.detect_behaviors("a static object"), [])


class TestBehaviorsFromText(unittest.TestCase):
    """Behavior templates declare their own phrases (``detect``): the parser
    reads prose and a behaviors column through them, so a new template
    brings its words with it instead of an edit to a regex list."""

    def test_phrases_live_in_the_templates(self):
        detect = beh.Behaviors.detect
        self.assertEqual(detect("Keys fade away from hand"), ["fade_out"])
        self.assertEqual(detect("Highlight Red Door Locks"), ["highlight"])
        self.assertEqual(detect("Destination circle fades in"), ["fade_in"])
        self.assertEqual(detect("User is teleported"), [])

    def test_a_column_cell_takes_names_or_phrases_and_keeps_unknowns(self):
        cell = beh.Behaviors.from_cell
        self.assertEqual(cell("Fade In, Fade Out"), ["fade_in", "fade_out"])
        self.assertEqual(cell("fades away\nhighlight"), ["fade_out", "highlight"])
        # Unknown items stay (spelled as a name) so Assess can flag them.
        self.assertEqual(cell("Wobble; fade_in"), ["wobble", "fade_in"])
        self.assertEqual(cell("  "), [])

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
            [("door", ["fade_in"]), ("lid", ["fade_out"])],
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
            [("door", ["highlight"]), ("lid", ["fade_out"])],
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
        "IN_01,Hi,Destination circle fades in at the teleport spot,\n"
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
            [("Pip screen", "fade_out")],
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
                ("Destination_circle", ["fade_in", "fade_out"], "description"),
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
            [("Socket_Wrench", ["highlight", "fade_out"])],
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


class TestRetiredTemplates(unittest.TestCase):
    """A saved ``.active`` pointer still names ``speedrun``: it resolves to the
    default template with its derived-audio option, and warns, for one
    deprecation window (CODE_STANDARD §5 -- saved data is published)."""

    def test_speedrun_loads_as_default_with_derived_audio(self):
        Mapping = mapping_mod.Mapping
        with self.assertWarns(DeprecationWarning):
            data = Mapping.load_mapping("speedrun")
        eff = Mapping.apply_options(data)
        self.assertEqual(eff["audio_resolve"]["method"], "derive")
        self.assertEqual(Mapping.retired("speedrun"), ("default", {"audio": "derive"}))
        self.assertIsNone(Mapping.retired("default"))

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
        # One flexible built-in: layouts are its options, not sibling files.
        self.assertEqual(names, ["default"])

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
        "A01.),Aileron fades in,wing_L,Welcome to the course\n"
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


class TestSequenceDocMapping(unittest.TestCase):
    """The default template's ``numbered`` step IDs read the interactive-
    training sequence doc: ``IN_01`` / ``A01`` / ``B03.5`` step IDs, an
    ``INTRO`` banner, a header repeated per section, multi-line voice cells,
    and a quiz block after the last step.  Its ``classic`` IDs (``A01.)``)
    parse none of them."""

    CSV = (
        "INTRO,,,,,,,\n"
        "Step,Step Name,Hint,Text On Placard,Voice Support,Step Contents,"
        "Asset Names,Frame Ranges\n"
        "IN_01,Welcome,,,Welcome to the training.,User starts in the room.,,\n"
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
        intro, teleport, _b03, branch = self._resolve()
        self.assertEqual(intro.description, "User starts in the room. Auto Advance")
        self.assertEqual(intro.audio, "Welcome to the training.")
        self.assertEqual(teleport.audio, "Open the machine.\n\nSelect 'Next'.")
        self.assertEqual(intro._pass_through.get("step_name"), "Welcome")
        self.assertEqual(
            [(o.name, o.behaviors) for o in branch.objects],
            [("solenoid_geo", ["fade_out"])],
        )

    def test_the_quiz_block_adds_no_steps_or_text(self):
        *_, branch = self._resolve()
        self.assertNotIn("baffles", branch.description)

    def test_classic_ids_are_unchanged_by_the_new_patterns(self):
        # Still ``A01.)`` / bare caps: this layout's IDs are not steps to it.
        self.assertEqual(self._resolve("classic"), [])


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
                    "choices": [("A", "a", ""), ("B", "b", "bee")],
                    "default": "a",
                },
                {
                    "key": "fill",
                    "kind": "bool",
                    "label": "fill",
                    "tooltip": "",
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
        self._prefs = tempfile.mkdtemp(prefix="mani_fill_")
        ShotStore._prefs_dir_override = self._prefs
        self.store = ShotStore()
        self.path = _write_csv(self.CSV)

    def tearDown(self):
        ShotStore._prefs_dir_override = None
        os.remove(self.path)

    def _steps(self):
        return ManifestModel.parse_csv(self.path, columns=ColumnMap(**self.COLUMNS))

    def test_steps_record_their_asset_cell(self):
        cells = {s.step_id: s.asset_cell for s in self._steps()}
        self.assertEqual(cells, {"IN_01": (2, 3), "A01": (6, 3), "A02": (8, 3)})

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
            [("arm_geo", ["fade_in"]), ("hand_geo", ["fade_in"])],
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

    def test_the_asset_column_lines_up_with_the_sheet_and_keeps_its_values(self):
        column = ManifestModel.asset_column(
            self.path,
            {"IN_01": ["walls", "floor"], "A01": ["ignored"], "A02": ["door_geo"]},
            columns=ColumnMap(**self.COLUMNS),
        )
        self.assertEqual(
            column,
            [
                "",
                "Asset Names",
                "walls\nfloor",
                "",
                "",
                "Asset Names",
                "door_geo",  # A01's own value stays; a fill never overwrites
                "",
                "door_geo",
            ],
        )

    def test_clipboard_formats_quote_multi_line_cells(self):
        tsv, html = ManifestModel.column_clipboard(["a", "b\nc", 'd"e'])
        self.assertEqual(tsv, 'a\n"b\nc"\n"d""e"')
        self.assertIn("<td>b<br>c</td>", html)
        self.assertIn("d&quot;e", html)


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

    def test_fade_away_reads_as_fade_out(self):
        self.assertEqual(ManifestModel.detect_behaviors("Keys fade away"), ["fade_out"])


class TestBehaviors(unittest.TestCase):
    def test_list_builtins(self):
        self.assertEqual(
            sorted(beh.Behaviors.list_behaviors()),
            ["fade_in", "fade_out", "highlight", "set_clip"],
        )

    def test_load_behavior(self):
        fi = beh.Behaviors.load_behavior("fade_in")
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

        base = copy.deepcopy(beh.Behaviors.load_behavior("fade_in"))  # cached
        store = beh.Behaviors.templates().store
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(store, "_user_dir", Path(tmp)),
        ):
            path = Path(tmp) / "fade_in.json"

            def place_after_writing(place, stamp):
                base["place"] = place
                path.write_text(json.dumps(base), encoding="utf-8")
                os.utime(path, (stamp, stamp))
                return beh.Behaviors.load_behavior("fade_in")["place"]

            self.assertEqual(place_after_writing("end", 1_000_000_000), "end")
            self.assertEqual(place_after_writing(0.5, 1_000_000_100), 0.5)
        # Out of the override again: the built-in, not the last user copy.
        self.assertEqual(beh.Behaviors.load_behavior("fade_in")["place"], "start")

    def test_resolve_keys_start_anchored_block(self):
        block = beh.Behaviors.keyed("fade_in")["attributes"]["visibility"]["in"]
        keys = beh.Behaviors.resolve_keys(block, 10, 60)
        # anchor=start, offset=0, duration=15, values=[0,1] -> keys at 10 and 25
        self.assertEqual([k["time"] for k in keys], [10.0, 25.0])
        self.assertEqual([k["value"] for k in keys], [0.0, 1.0])

    def test_resolve_keys_end_anchored_block(self):
        out = beh.Behaviors.keyed("fade_out")["attributes"]["visibility"]["out"]
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
        fades beside it were pushed off their ends (fade_in keyed mid-shot).
        Points spread in doc order; a span keeps its own anchors.
        Fixed: 2026-10-03
        """
        overrides = beh.Behaviors.anchor_overrides
        self.assertEqual(overrides(["fade_in"]), [None])
        self.assertEqual(overrides(["fade_in", "fade_out"]), [0.0, 1.0])
        self.assertEqual(overrides(["fade_in", "fade_out", "fade_in"]), [0.0, 0.5, 1.0])
        self.assertEqual(overrides(["highlight", "fade_out"]), [None, None])
        self.assertEqual(
            overrides(["highlight", "fade_in", "fade_out"]), [None, 0.0, 1.0]
        )
        # A name no template answers keeps its own (its apply records the failure).
        self.assertEqual(overrides(["no_such_behavior", "fade_out"]), [None, None])

    def test_a_highlight_beside_a_fade_keeps_its_ramps_apart(self):
        """What a host keys for a highlight beside a fade: the pulse over the
        whole shot, dim at both ends -- before the fix the highlight's ramps
        collapsed to (100, 1), (110, 0), a glow held for the whole timeline
        before the shot."""
        recipe = EffectRecipe()
        tmpl = beh.Behaviors.load_behavior("highlight")
        anchor = beh.Behaviors.anchor_overrides(["highlight", "fade_out"])[0]
        self.assertIsNone(anchor)
        window = recipe.window("pulse", 100, 300, tmpl["place"], anchor)
        self.assertEqual(window, (100.0, 300.0))
        keys = recipe.plan("pulse", *window, fps=30.0)
        self.assertEqual((keys[0], keys[-1]), ((100.0, 0.0), (300.0, 0.0)))
        self.assertEqual(max(v for _t, v in keys), 1.0)

    def test_compute_duration_dict_and_object_forms_agree(self):
        dict_form = beh.Behaviors.compute_duration(
            [{"name": "g", "behavior": "fade_in"}]
        )
        obj_form = beh.Behaviors.compute_duration(
            [BuilderObject(name="g", behaviors=["fade_in"])]
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


class TestManifestEngine(unittest.TestCase):
    """The pure ShotManifest planner + commit against a real ShotStore."""

    def setUp(self):
        self._prefs = tempfile.mkdtemp(prefix="mani_engine_")
        ShotStore._prefs_dir_override = self._prefs
        self.store = ShotStore()
        self.mani = ShotManifest(self.store)

    def tearDown(self):
        ShotStore._prefs_dir_override = None

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
                "vo", kind="audio", source_path="x.wav", behaviors=["fade_in"]
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
            "A01", "A", "t", "", [BuilderObject("g", behaviors=["fade_in"])]
        )
        dur, beh_span, aud = ShotManifest.resolve_duration(
            step, initial_shot_length=200, fit_mode="extend_only", fps=24.0
        )
        self.assertGreaterEqual(dur, 0.0)


class _ScenicStore(ShotStore):
    """A pure store whose scene holds long, namespaced paths: ``door`` lives
    at ``|set|AC:door``; names in ``missing`` / ``ambiguous`` resolve so."""

    missing: set = set()
    ambiguous: set = set()

    def resolve_member(self, name):
        if name in self.missing:
            return name, "missing"
        if name in self.ambiguous:
            return name, "ambiguous"
        return f"|set|AC:{ShotStore.member_key(name)}", "found"


class _EngineCase(unittest.TestCase):
    def setUp(self):
        self._prefs = tempfile.mkdtemp(prefix="mani_reconcile_")
        ShotStore._prefs_dir_override = self._prefs
        _ScenicStore.missing, _ScenicStore.ambiguous = set(), set()
        self.store = _ScenicStore()

    def tearDown(self):
        ShotStore._prefs_dir_override = None

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

    def test_build_writes_the_binding_so_order_no_longer_matters(self):
        self.store.define_shot("Step_1", 1, 10)
        mani = ShotManifest(self.store, match="name_then_order")
        mani.update([self._step("IN_01")], remove_missing=False)
        self.assertEqual(self.store.shot_by_name("Step_1").metadata["step"], "IN_01")
        # Name-only matching now finds it through the binding.
        pairing = ShotManifest(self.store).pair([self._step("IN_01")])
        self.assertEqual(pairing.how["IN_01"], "binding")


class TestReconcileMembership(_EngineCase):
    """Membership is compared through ``ShotStore.member_key``: a doc name
    against a stored long, namespaced node."""

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

    def test_an_orphan_shot_is_kept_unless_removal_is_asked_for(self):
        mani = ShotManifest(self.store)
        mani.update([self._step("A01"), self._step("A02")], initial_shot_length=10)
        mani.update([self._step("A01")], remove_missing=False)
        self.assertIsNotNone(self.store.shot_by_name("A02"))
        self.assertEqual(
            [s.name for s in mani.pair([self._step("A01")]).orphans], ["A02"]
        )


class TestReconcileStatuses(_EngineCase):
    """Assess reports; it never writes the store."""

    def _built(self, step):
        mani = ShotManifest(self.store)
        mani.update([step], initial_shot_length=50)
        return mani

    def test_missing_and_ambiguous_objects(self):
        step = self._step("A01", "door", "lid")
        mani = self._built(step)
        _ScenicStore.missing, _ScenicStore.ambiguous = {"door"}, {"lid"}
        (st,) = mani.assess([step])
        self.assertEqual(
            [o.status for o in st.objects], ["missing_object", "ambiguous_object"]
        )
        self.assertEqual(st.status, "missing_object")

    def test_unknown_behavior_is_reported_not_verified(self):
        step = self._step("A01", "door", behaviors={"door": ["wobble"]})
        mani = self._built(step)
        mani._verify_behavior = lambda *a, **k: self.fail("verified an unknown")
        (st,) = mani.assess([step])
        self.assertEqual(st.objects[0].status, "unknown_behavior")

    def test_a_broken_behavior_over_animator_keys_is_a_conflict(self):
        step = self._step("A01", "door", behaviors={"door": ["fade_in"]})
        mani = self._built(step)
        mani._verify_behavior = lambda *a, **k: False
        mani._key_samples = lambda obj, behavior, s, e: [("door_opacity", 5.0)]
        (st,) = mani.assess([step])
        self.assertEqual(st.objects[0].status, "behavior_conflict")
        # The same keys, claimed by the ledger, are the manifest's own: fixable.
        self.store.edit_ledger.record_authored(
            "door_opacity",
            5.0,
            self.store.shot_by_name("A01").shot_id,
            "fade_in",
            "door",
        )
        (st,) = mani.assess([step])
        self.assertEqual(st.objects[0].status, "missing_behavior")

    def test_a_step_listing_nothing_says_so(self):
        step = self._step("A01")
        mani = self._built(step)
        (st,) = mani.assess([step])
        self.assertEqual(st.status, "no_objects")

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
        step = self._step("A01", "door", behaviors={"door": ["fade_in"]})
        shot_holder = {}

        def apply_behaviors():
            shot = self.store.shot_by_name("A01")
            shot_holder["id"] = shot.shot_id
            return {
                "applied": [
                    {
                        "object": "door",
                        "behavior": "fade_in",
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
        led.record_authored("door_opacity", 1.0, 7, "fade_in", "door")
        led.record_authored("door_opacity", 90.0, 7, "fade_out", "door")
        deleted = []
        mani._delete_keys = lambda curve, times: deleted.append((curve, list(times)))
        self.assertEqual(mani.release_authored(7, "door", "fade_in"), 1)
        self.assertEqual(deleted, [("door_opacity", [1.0])])
        self.assertEqual(led.authored(owner=7), [("door_opacity", 90.0)])


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
                        "door", True, "missing_behavior", ["fade_out"], ["fade_out"]
                    )
                ],
            ),
        ]
        self.assertEqual(StepStatus.find_object(results, "door").status, "valid")
        found = StepStatus.find_object(results, "door", "A05")
        self.assertEqual(found.broken_behaviors, ["fade_out"])
        self.assertIsNone(StepStatus.find_object(results, "door", "A09"))
        self.assertIsNone(StepStatus.find_object(results, "wall"))
        self.assertIsNone(StepStatus.find_object([], "door"))


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
                BuilderObject(name="aileron_geo", behaviors=["fade_in"]),
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
                ("aileron_geo", "scene", ["fade_in"]),
                ("wing_geo", "scene", []),
                ("vo_A01", "audio", ["set_clip"]),
            ],
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
            super().__init__(store)
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
        obj = BuilderObject(name="Obj", behaviors=["highlight", "fade_out"])
        self.host.plan = {"highlight": 100, "fade_out": 285}
        self.host.reapply_object(self.shot, obj)
        self.assertEqual(self.host.anchors, [None, None])

    def test_a_shared_frame_survives_the_reapply(self):
        """Bug: each behavior's old keys were released just before it was
        re-keyed, so a later behavior's release deleted the key an earlier one
        had just written on a frame they share -- the loss the build avoids by
        releasing everything first.
        Fixed: 2026-10-03
        """
        obj = BuilderObject(name="Obj", behaviors=["fade_in", "fade_out"])
        self.host.plan = {"fade_in": 100, "fade_out": 200}
        self.host.reapply_object(self.shot, obj)
        self.host.plan = {"fade_in": 200, "fade_out": 300}  # the shot moved
        self.host.reapply_object(self.shot, obj)
        self.assertEqual(
            sorted(self.host.curve.items()), [(200, "fade_in"), (300, "fade_out")]
        )


class TestRecipeOwnership(unittest.TestCase):
    """The manifest keys from the scene's effect recipe: its keys carry the
    recipe they were made under, Assess flags what an older recipe made, and a
    build takes out the keys of behaviors the doc dropped."""

    class _Host(ShotManifest):
        """Each behavior keys ``<node>.<behavior>`` at the shot start (a fade
        out at the end) and reports it, as a host applier does."""

        def __init__(self, store):
            super().__init__(store)
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
            t = end if behavior == "fade_out" else start
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
        shot = self._build(self._steps("fade_in", "highlight"))
        led, recipe = self.store.edit_ledger, self.store.effect_recipe
        self.assertEqual(
            led.authored_stamps(shot.shot_id, "Door", "highlight"),
            {recipe.fingerprint("pulse")},
        )
        self.assertEqual(
            led.authored_stamps(shot.shot_id, "Door", "fade_in"),
            {recipe.fingerprint("fade_in")},
        )

    def test_a_changed_recipe_flags_its_keys_until_a_build(self):
        steps = self._steps("fade_in", "highlight")
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
        shot = self._build(self._steps("fade_in", "highlight"))
        self.assertEqual(self.host.curves["Door.highlight"], {100.0: "highlight"})

        steps = self._steps("fade_in")
        step, _door = self._door(steps)
        self.assertEqual(step.dropped_behaviors, [["Door", "highlight"]])
        self.assertTrue(step.needs_build)

        self._build(steps)
        self.assertEqual(self.host.curves["Door.highlight"], {})
        led = self.store.edit_ledger
        self.assertEqual(led.authored(owner=shot.shot_id, behavior="highlight"), [])
        self.assertEqual(len(led.authored(owner=shot.shot_id, behavior="fade_in")), 1)
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

    def test_apply_releases_what_the_doc_dropped_for_the_object(self):
        shot = self._build(self._steps("fade_in", "highlight"))
        self.host.reapply_object(shot, BuilderObject("Door", ["fade_in"]))
        self.assertEqual(self.host.curves["Door.highlight"], {})
        self.assertEqual(self.host.curves["Door.fade_in"], {100.0: "fade_in"})

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
