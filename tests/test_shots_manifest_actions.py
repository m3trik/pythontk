# !/usr/bin/python
# coding=utf-8
"""Tests for a sheet in the cell format (``ManifestActions`` and the reader's cell rows).

DCC-free. A template that declares its actions reads a row by its cells: the word in its
Action cell, the names in Target, With and Members, its take in Clip, a choice's Label and
To, and its Settings. These hold the declared words to their cells, each issue to its cell
and fix, a row said back as a sentence, and a shot per take.
"""

import csv
import io
import os
import sys
import tempfile
import unittest

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pythontk.core_utils.engines.shots.manifest.manifest_actions import ManifestActions
from pythontk.core_utils.engines.shots.manifest.manifest_model import ColumnMap
from pythontk.core_utils.engines.shots.manifest.mapping import Mapping
from pythontk.core_utils.schema_spec import SchemaError

HEADER = "Step,Step Name,Voice,Placard,Audio,Action,Target,With,Clip,Members,Label,To,Settings,Notes"


def _as_written(row: str) -> list:
    """*row* as the sheet writes it. A test writes a step and its first action on one line for
    brevity; a sheet writes the step's own row, then the action on the row below it."""
    cells = next(csv.reader(io.StringIO(row)))
    if len(cells) != len(HEADER.split(",")) or not cells[0] or not any(cells[5:11]):
        return [cells]
    return [cells[:5] + [""] * 9, [""] * 5 + cells[5:]]


def _sheet(*rows: str, as_written: bool = False) -> str:
    """A sheet of *rows* under the cell format's header, as a temp CSV's path; *as_written* writes
    each row as the test gives it (:func:`_as_written` otherwise)."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER.split(","))
        for row in rows:
            writer.writerows(
                [next(csv.reader(io.StringIO(row)))] if as_written else _as_written(row)
            )
    return path


def _read(
    *rows: str, template: str = "interactive", as_written: bool = False, **options
):
    path = _sheet(*rows, as_written=as_written)
    try:
        return Mapping.resolve(path, name=template, options=options or None)
    finally:
        os.remove(path)


def _issues(steps):
    return [
        issue for step in steps for row in step.rows for issue in row.get("issues", ())
    ]


class TestWordsAndSettings(unittest.TestCase):
    def test_a_word_is_read_without_regard_to_case_or_spacing(self):
        self.assertEqual(ManifestActions.word(" Fade In "), "fade_in")
        self.assertEqual(ManifestActions.word("USE"), "use")

    def test_a_settings_cell_is_pairs_and_says_what_is_no_pair(self):
        settings, errors = ManifestActions.read_settings("Trigger=hold; seconds = 2")
        self.assertEqual(settings, {"trigger": "hold", "seconds": "2"})
        self.assertEqual(errors, [])
        _, errors = ManifestActions.read_settings("hold; trigger=a; trigger=b")
        self.assertEqual(len(errors), 2)
        self.assertIn("name=value", errors[0])
        self.assertIn("twice", errors[1])

    def test_a_template_refuses_a_malformed_declaration(self):
        bad = {
            "Use": {
                "cells": ["target", "tool"],
                "needs": ["clip"],
                "when": "later",
                "settings": {"x": {"type": "date"}},
            },
            "go": {
                "cells": ["target"],
                "places": "fly",
                "needs_any": [["target", "nowhere"]],
            },
        }
        errors = ManifestActions.validate_actions(bad)
        text = " ".join(errors)
        for word in (
            "lower case",
            "tool",
            "needs",
            "when",
            "setting 'x'",
            "places one of",
            "a 'where' setting",
            "needs_any",
        ):
            self.assertIn(word, text)
        with self.assertRaises(SchemaError):
            ColumnMap.validate({"actions": bad}).raise_or_warn()

    def test_the_words_a_build_keys_are_declared_and_linear_declares_no_other(self):
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

        keyed = set(Behaviors.list_behaviors())
        linear = set(Mapping.load_mapping("linear")["columns"]["actions"]) - {""}
        self.assertLessEqual(
            linear, keyed, "a Linear sheet's every word is one the Shot Manifest keys"
        )
        self.assertLessEqual(
            {"fade_in", "fade_out", "highlight"},
            set(Mapping.load_mapping("interactive")["columns"]["actions"]),
        )

    def test_the_built_in_templates_declare_their_words(self):
        interactive = Mapping.load_mapping("interactive")["columns"]["actions"]
        self.assertEqual(len(interactive), 17)
        self.assertEqual(ManifestActions.validate_actions(interactive), [])
        linear = Mapping.load_mapping("linear")["columns"]["actions"]
        self.assertEqual(sorted(linear), ["", "fade_in", "fade_out", "highlight"])


class TestTheDeclarationIsRestatedAlike(unittest.TestCase):
    """The templates declare the format once; each starter and the authors'
    guide (``docs/build_sheets.md``) restate it, and each is held to it here."""

    GUIDE = os.path.join(_PKG_PARENT, "docs", "build_sheets.md")
    CELLS = {
        "Target": "target",
        "With": "with",
        "Clip": "clip",
        "Members": "members",
        "Label": "label",
        "To": "to",
    }
    MOMENTS = {
        "as the step starts": "start",
        "start": "start",
        "the user": "user",
        "as the step ends": "end",
        "end": "end",
        "never": "none",
    }

    @classmethod
    def _guide(cls) -> str:
        with open(cls.GUIDE, encoding="utf-8") as handle:
            return handle.read()

    @classmethod
    def _table(cls, *header: str) -> list:
        """The rows of the guide's table whose header begins with *header*, each its cells (a
        ``\\|`` in a cell is its own ``|``, as the guide's tables write a path)."""
        import re

        rows, inside = [], False
        for line in cls._guide().splitlines():
            line = line.strip()
            if not line.startswith("|"):
                if inside:
                    break
                continue
            cells = [
                cell.strip().replace("\\|", "|")
                for cell in re.split(r"(?<!\\)\|", line[1:-1])
            ]
            if not inside:
                inside = tuple(cells[: len(header)]) == header
            elif not set(line) <= set("|:- "):
                rows.append(cells)
        return rows

    @staticmethod
    def _header(template: str) -> list:
        with open(Mapping.starter_sheet(template), encoding="utf-8") as handle:
            return next(csv.reader(handle))

    def test_each_starter_uses_every_word_its_template_declares_and_reads_clean(self):
        for name in ("interactive", "linear"):
            with self.subTest(name):
                sheet = str(Mapping.starter_sheet(name))
                steps = Mapping.resolve(sheet, name=name)
                declared = set(Mapping.load_mapping(name)["columns"]["actions"])
                self.assertEqual(
                    {row.get("word", "") for step in steps for row in step.rows},
                    declared,
                )
                self.assertEqual(
                    _issues(steps) + [i for step in steps for i in step.sheet_issues()],
                    [],
                )
                self.assertEqual(
                    [
                        w
                        for step in steps
                        for row in step.rows
                        for w in row.get("warnings", ())
                    ],
                    [],
                )
                self.assertEqual(
                    Mapping.unmapped_columns(sheet, Mapping.load_mapping(name)), []
                )

    def test_the_guide_names_each_strict_templates_columns_as_its_starter_writes_them(
        self,
    ):
        rows = {
            row[0].strip("`"): row[2]
            for row in self._table("Template", "For", "Columns")
        }
        self.assertEqual(
            [c.strip() for c in rows["linear"].split(",")], self._header("linear")
        )
        added = rows["interactive"].split("plus", 1)[1].replace(" and ", ", ")
        self.assertEqual(
            set(self._header("linear")) | {c.strip() for c in added.split(",")},
            set(self._header("interactive")),
        )
        self.assertEqual(
            [row[0] for row in self._table("Column", "Row", "Holds")],
            self._header("interactive"),
            "the Columns table: every column a sheet in the format has, in the starter's order",
        )

    def test_the_guide_states_each_word_as_its_template_declares_it(self):
        import re

        actions = Mapping.load_mapping("interactive")["columns"]["actions"]
        named = re.compile(r"(\*\*)?\b(%s)\b" % "|".join(self.CELLS))
        family, stated = "", {}
        for fam, word, happens, cells, settings in self._table(
            "Family", "Word", "Happens"
        ):
            family = fam.lower() or family
            word = word.strip("`")
            found = named.findall(cells)
            stated[word] = {
                "family": family,
                "when": self.MOMENTS[happens],
                "cells": [self.CELLS[c] for _, c in found],
                "needs": [self.CELLS[c] for bold, c in found if bold],
                "settings": sorted(
                    re.findall(r"`(\w+)`", re.sub(r"\([^)]*\)", "", settings))
                ),
            }
            for name, said in re.findall(r"`(\w+)`\s*\(([^)]*)\)", settings):
                setting = actions[word]["settings"][name]
                if "choices" in setting:
                    self.assertEqual(
                        {c.strip() for c in said.split(",")},
                        set(setting["choices"]),
                        (word, name),
                    )
                self.assertEqual(
                    said == "required", bool(setting.get("required")), (word, name)
                )
        declared = {
            word: {
                "family": d.get("family"),
                "when": d.get("when"),
                "cells": list(d.get("cells") or ()),
                "needs": list(d.get("needs") or ()),
                "settings": sorted(d.get("settings") or {}),
            }
            for word, d in actions.items()
        }
        self.assertEqual(stated, declared)
        linear = re.search(
            r"The Linear template declares the three[^)]*\)", self._guide()
        ).group(0)
        self.assertEqual(
            set(re.findall(r"`(\w+)`", linear)),
            set(Mapping.load_mapping("linear")["columns"]["actions"]) - {""},
        )

    def test_the_guide_files_each_setting_under_its_one_part(self):
        import re

        filed = []
        for _part, said in self._table("Part", "Its settings"):
            filed += re.findall(r"`(\w+)`", re.sub(r"\([^)]*\)", "", said))
        declared = {
            name
            for d in Mapping.load_mapping("interactive")["columns"]["actions"].values()
            for name in d.get("settings") or ()
        }
        self.assertEqual(
            sorted(filed),
            sorted(declared),
            "each declared setting under one part, and no other",
        )


class TestRowsReadByCells(unittest.TestCase):
    def test_a_row_is_read_by_its_cells_and_never_by_its_notes(self):
        steps = _read(
            "A01,Open,Open it.,,,use,DOOR,KEY,A01,,,,,The door fades in and is highlighted",
        )
        row = steps[0].rows[0]
        self.assertEqual(
            (row["word"], row["assets"], row["with"], row["clip"]),
            ("use", ["DOOR"], ["KEY"], "A01"),
        )
        self.assertTrue(row["take"])
        self.assertEqual(row["issues"], [])
        # The notes name a fade and a highlight: neither is read.
        self.assertEqual(
            [(o.name, o.behaviors) for o in steps[0].objects],
            [("DOOR", []), ("KEY", [])],
        )
        self.assertEqual(
            steps[0].description, "Selecting DOOR with KEY in hand plays A01"
        )

    def test_each_issue_names_its_cell_and_its_fix(self):
        steps = _read(
            "A01,Open,Open it.,,,hilight,DOOR,,,,,,,",
            ",,,,,use,DOOR,,,,,B09,trigger=slow; speed=2,",
            ",,,,,button,DOOR,,,,Next,B09,,",
            ",,,,,,DOOR,,,,Next,,,",
            ",,,,,set,LAMP,,,,,,,",
        )
        found = {(header, text.split(":")[0]) for header, text in _issues(steps)}
        self.assertIn(("Action", "'hilight' is no action of this template"), found)
        self.assertIn(("Clip", "use needs its Clip"), found)
        self.assertIn(("To", "use takes no To"), found)
        self.assertIn(("Settings", "'slow' is no trigger"), found)
        self.assertIn(("Settings", "use has no setting 'speed'"), found)
        self.assertIn(("Target", "button takes no Target"), found)
        self.assertIn(("To", "the sheet has no step B09"), found)
        self.assertIn(
            ("Action", "the row fills Target and Label and names no action"), found
        )
        self.assertIn(("Settings", "set needs its state"), found)

    def test_a_settings_value_is_held_to_its_type(self):
        steps = _read(
            "A01,Open,Open it.,,,use,DOOR,,A01,,,,hold=long,",
            ",,,,,highlight,DOOR,,,,,,color=bluish,",
            ",,,,,fade_out,DOOR,,,,,,fade=-1,",
            ",,,,,pip,DOOR,,,,,,until=Z99,",
            ",,,,,pip,LID,,,,,,until=step,",
            ",,,,,highlight,LID,,,,,,color=#3399FF; style=steady,",
        )
        texts = [text for _, text in _issues(steps)]
        self.assertEqual(len(texts), 4)
        self.assertTrue(any("takes a number" in t for t in texts))
        self.assertTrue(any("#RRGGBB" in t for t in texts))
        self.assertTrue(any("is at least 0" in t for t in texts))
        self.assertTrue(any("the sheet has no step Z99" in t for t in texts))

    def test_a_choice_goes_to_a_step_the_sheet_has_or_a_destination(self):
        steps = _read(
            "B03,Ask,Did it light?,,,button,,,,,YES,B04,,",
            ",,,,,button,,,,,NO,B03.5,,",
            ",,,,,button,,,,,Skip,Next,,",
            "B03.5,Wrong,Try again.,,,button,,,,,Try Again,retry,,",
            "B04,Done,Well done.,,,button,,,,,Exit,exit,,",
        )
        self.assertEqual(
            _issues(steps), [("To", "'Next' is written in lower case: write next")]
        )
        self.assertEqual(
            steps[0].description.split("; ")[1], "Selecting 'NO' goes to step B03.5"
        )
        self.assertEqual(steps[2].description, "Selecting 'Exit' ends the task")

    def test_a_take_is_one_rows_and_an_effect_names_a_take_its_step_plays(self):
        steps = _read(
            "A01,Unlock,Unlock both.,,,use,LATCH_UPPER,KEY,A01_1,,,,,",
            ",,,,,use,LATCH_LOWER,KEY,A01_1,,,,,",
            ",,,,,fade_out,BOLT,,A01_9,,,,,",
            ",,,,,play,,,B07,LID,,,after=A01_7,",
        )
        texts = [text.split(":")[0] for _, text in _issues(steps)]
        self.assertIn("the take A01_1 is played by an earlier row of the step", texts)
        self.assertIn("no row of the step plays a take A01_9", texts)
        self.assertIn(
            "after names the take A01_7, which no other row of the step plays", texts
        )
        warnings = [w for row in steps[0].rows for w in row["warnings"]]
        self.assertEqual(
            len(warnings), 1
        )  # B07 is not named for A01: said, not refused (another tool's take)
        self.assertIn("not named for its step", warnings[0][1])

    def test_a_take_is_named_for_its_step_by_its_id_alone_or_its_id_and_a_number(self):
        """Bug: the warning held a take only to starting with its step's ID,
        so ``A01B`` -- the step the format adds after A01 -- passed as A01's
        own, though the message says A01 or A01_<n>.
        Fixed: 2026-11-26"""
        steps = _read(
            "A01,Open,Open it.,,,use,DOOR,,A01,,,,,",
            ",,,,,use,LID,,A01_2,,,,,",
            ",,,,,use,LATCH,,A01B,,,,,",
            ",,,,,use,BOLT,,A01_X,,,,,",
        )
        warned = [w[1].split(":")[0] for row in steps[0].rows for w in row["warnings"]]
        self.assertEqual(
            warned,
            [
                "the take A01B is not named for its step",
                "the take A01_X is not named for its step",
            ],
        )

    def test_the_audio_column_is_the_clip_file_never_the_narration(self):
        """Bug: both strict templates listed Audio among the narration's
        headers as well as the clip file's, so a sheet with no Voice column
        read ``Audio=a01_open.wav`` as the step's voice line too (narration
        would speak the file's name).
        Fixed: 2026-11-26"""
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(
                "Step,Step Name,Audio,Action,Target\nA01,Open,a01_open.wav,,\n,,,highlight,LID\n"
            )
        try:
            for template in ("interactive", "linear"):
                with self.subTest(template):
                    (step,) = Mapping.resolve(path, name=template)
                    self.assertEqual(
                        (step.audio, step.pass_through.get("audio_file")),
                        ("", "a01_open.wav"),
                    )
        finally:
            os.remove(path)

    def test_a_steps_row_is_the_step_alone(self):
        steps = _read(
            "B03,Ask,Did it light?,,,,,,,,,,,",
            ",,,,,button,,,,,YES,B04,,",
            ",,,,,button,,,,,NO,B03A,,",
            "B03A,Wrong,Try again.,,,,,,,,,B03,,",
            "B04,Done,Well done.,,,,,,,,,exit,dwell=3,",
            "B05,Late,One more.,,,highlight,DOOR,,,,,Z99,,",
            as_written=True,
        )
        by_id = {s.step_id: s for s in steps}
        # A step added later is spelled as every tool spells it, and goes back by itself, with no button.
        self.assertEqual(
            (by_id["B03A"].to, by_id["B03A"].rows, by_id["B03A"].issues),
            ("B03", [], []),
        )
        self.assertEqual(by_id["B03"].to, "")
        self.assertEqual(by_id["B04"].to, "exit")
        self.assertEqual(
            [text for _, text in by_id["B04"].issues],
            ["a step has no setting 'dwell': clear it (it has none)"],
        )
        # An action on a step's own row is read, and said to belong on the row below.
        row = by_id["B05"].rows[0]
        self.assertEqual(row["word"], "highlight")
        self.assertIn(
            (
                "Action",
                "a step's row holds no action: write this one on the row below it",
            ),
            row["issues"],
        )

    def test_a_step_going_to_no_step_and_a_teleport_naming_nothing_are_said(self):
        steps = _read(
            "A01,Go,Step up.,,,,,,,,,Z99,,",
            ",,,,,teleport,,,,,,,,",
            ",,,,,teleport,,,,,,,where=VIEW_FRONT,",
            as_written=True,
        )
        self.assertEqual([header for header, _ in steps[0].issues], ["To"])
        self.assertEqual(
            [i for row in steps[0].rows for i in row["issues"]],
            [("Target", "teleport needs its Target or its where: fill one of them")],
        )

    def test_where_names_the_object_a_placing_row_happens_at_and_with_is_the_tool_alone(
        self,
    ):
        # 2026-11-25, the maintainer: "the same field acting differently by object type is probably bad design".
        # Target is what a row is about for every word; `where` is the object where it happens; With, the tool.
        steps = _read(
            "A01,Go,Step up.,,,,,,,,,,,",
            ",,,,,teleport,CABINET,,,,,,where=VIEW_FRONT,",
            ",,,,,indicate,KEY,,,,,,where=KEY_ARROW,",
            ",,,,,teleport,CABINET,VIEW_BACK,,,,,,",
            ",,,,,pip,LAMP,,,,,,where=two words,",
            as_written=True,
        )
        issues = [row["issues"] for row in steps[0].rows]
        self.assertEqual(
            issues[0] + issues[1], [], "Target and where, each its own meaning"
        )
        self.assertEqual(
            issues[2],
            [("With", "teleport takes no With: clear it")],
            "With is no place",
        )
        self.assertEqual(
            issues[3],
            [
                (
                    "Settings",
                    "where names one object, as the scene spells it, not 'two words': "
                    "write its name alone, or the end of its path that tells it apart "
                    "where two share it",
                )
            ],
        )
        actions = Mapping.load_mapping("interactive")["columns"]["actions"]
        self.assertEqual(
            {w: d.get("places") for w, d in actions.items() if d.get("places")},
            {"teleport": "stand", "indicate": "point", "pip": "view"},
        )

    def test_every_issue_ends_in_its_fix_after_its_one_last_colon(self):
        """A reader splits an issue into its reason and its fix at the last colon. Messages with none
        gave an empty fix, and "(it has: a, b)" was cut in two there. Added: 2026-11-25"""
        steps = _read(
            "A01,Open,Open it.,,,hilight,DOOR,,,,,,,",
            ",,,,,use,,,,,,,trigger=slow; speed=2,",
            ",,,,,button,,,,,,,,",
            ",,,,,play,,,,,,,after=A01_7,",
            ",,,,,highlight,DOOR,,,,Look,,color=red; fade,",
            ",,,,,fade_out,DOOR,,,,,,fade=fast,",
            ",,,,,fade_in,DOOR,,,,,,fade=-2,",
            ",,,,,pip,DOOR,,,,,,until=Z99; where=two words,",
            ",,,,,teleport,,,,,,,,",
            ",,,,,set,LAMP,,,,,,,",
            ",,,,,button,,,,,Go,Next,,",
            "A02,Close,Close it.,,,,,,,,,Z98,dwell=3,",
            as_written=True,
        )
        found = _issues(steps) + [i for step in steps for i in step.issues]
        self.assertGreater(len(found), 15)
        for header, text in found:
            reason, colon, fix = text.rpartition(": ")
            self.assertTrue(colon and fix.strip(), "no fix: %s" % text)
            self.assertEqual(
                reason.count("("), reason.count(")"), "cut inside a bracket: %s" % text
            )
        texts = [text for _, text in found]
        self.assertIn(
            "use needs its Target: write the part it is about, by its scene name", texts
        )
        self.assertIn(
            "use has no setting 'speed': clear it, or write one of hold, trigger", texts
        )

    def test_two_settings_split_by_a_comma_are_said_as_that(self):
        """``trigger=hold, hold=2`` read as one value and said "'hold, hold=2' is no trigger". Added: 2026-11-25"""
        steps = _read('A01,Open,Open it.,,,use,DOOR,,A01,,,,"trigger=hold, hold=2",')
        self.assertEqual(
            _issues(steps),
            [
                (
                    "Settings",
                    "'trigger=hold, hold=2' holds two settings: write a ; between "
                    "them, not a comma",
                )
            ],
        )
        # A reply is free text: a comma and an equals sign in it are its own.
        steps = _read(
            'Q01,Ask,Which?,,,button,,,,,Two,retry,"reply=No, 2+2=4",',
            ",,,,,button,,,,,Four,next,,",
        )
        self.assertEqual(_issues(steps), [])

    def test_two_rows_of_a_word_share_no_cell_the_word_declares_distinct(self):
        """Two YES buttons going to different steps, and two answers taking any other value, read
        with no issue: which one a press chose was undefined. Added: 2026-11-25"""
        steps = _read(
            "B03,Ask,Did it light?,,,button,,,,,YES,B04,,",
            ",,,,,button,,,,,yes,B03,,",
            ",,,,,answer,,,,,*,B04,,",
            ",,,,,answer,,,,,*,B03,,",
            ",,,,,answer,,,,,11,B04,,",
            "B04,Done,Well done.,,,button,,,,,Exit,exit,,",
        )
        self.assertEqual(
            _issues(steps),
            [
                (
                    "Label",
                    "another button row of the step has the Label 'yes': give each its own",
                ),
                (
                    "Label",
                    "another answer row of the step has the Label '*': give each its own",
                ),
            ],
        )
        actions = Mapping.load_mapping("interactive")["columns"]["actions"]
        self.assertEqual(
            {w: d["distinct"] for w, d in actions.items() if d.get("distinct")},
            {"button": ["label"], "answer": ["label"]},
        )

    def test_a_span_that_names_a_step_the_sheet_is_past_is_a_warning(self):
        """``until=A01`` on a row of B03 read with nothing said: the picture then never left. Added: 2026-11-25"""
        steps = _read(
            "A01,Open,Open it.,,,pip,DOOR,,,,,,until=A02,",
            "A02,Look,Look at it.,,,pip,DOOR,,,,,,until=A01,",
            ",,,,,pip,LID,,,,,,until=A02,",
            ",,,,,pip,LAMP,,,,,,until=step,",
        )
        self.assertEqual(_issues(steps), [])
        warned = [
            (row["assets"], text)
            for step in steps
            for row in step.rows
            for _, text in row["warnings"]
        ]
        self.assertEqual([names for names, _ in warned], [["DOOR"], ["LID"]])
        self.assertIn(
            "so what the row shows stays: name a later step, or step", warned[0][1]
        )

    def test_a_look_that_names_no_take_is_its_steps_and_only_a_sheet_played_through_keys_it(
        self,
    ):
        """A task built from the Interactive starter with its looks keyed: the new filter the user
        is asked to take stayed hidden, its fade-in keyed into a take that plays only once they
        bring it to its slot. A look with no Clip happens at its step's moment, which in an
        interactive sheet is the task's to run; one that names its take rides it. Added: 2026-11-25"""
        rows = (
            "B02,Fit,Fit the new filter.,,,fade_in,NEW_FILTER,,,,,,,",
            ",,,,,use,SLOT,NEW_FILTER,B02,,,,,",
            ",,,,,fade_out,OLD_FILTER,,B02,,,,,",
            ",,,,,highlight,SLOT,,,,,,,",
        )
        held = {o.name: o.behaviors for o in _read(*rows)[0].objects}
        self.assertEqual(
            held,
            {"NEW_FILTER": [], "SLOT": [], "OLD_FILTER": ["fade_out"]},
            "interactive: the look that names its take is keyed; the step's own are members, nothing keyed",
        )
        linear = _read(
            "B02,Fit,Fit the new filter.,,,fade_in,NEW_FILTER,,,,,,,",
            ",,,,,highlight,SLOT,,,,,,,",
            template="linear",
        )
        self.assertEqual(
            {o.name: o.behaviors for o in linear[0].objects},
            {"NEW_FILTER": ["fade_in"], "SLOT": ["highlight"]},
            "linear: the step is its shot, played through",
        )
        self.assertEqual(
            (
                Mapping.load_mapping("interactive")["columns"]["step_looks"],
                Mapping.load_mapping("linear")["columns"].get("step_looks", "key"),
            ),
            ("leave", "key"),
        )

    def test_a_row_with_no_action_is_read_as_before(self):
        # The cell format is per row: a sheet's older rows keep their sentence's behaviors.
        steps = _read("A01,Open,Open it.,,,,DOOR,,,,,,,The door fades in")
        self.assertNotIn("cells", steps[0].rows[0])
        self.assertEqual(
            [(o.name, o.behaviors) for o in steps[0].objects], [("DOOR", ["fade_in"])]
        )


class TestRowsNoStepReads(unittest.TestCase):
    """A row the reader cannot give to a step is said, with the sheet's own row number, never dropped."""

    def test_a_second_step_of_one_id_is_said_at_the_first(self):
        """The second step and its rows were dropped with a log line alone, and the check passed. Added: 2026-11-25"""
        steps = _read(
            "A01,Open,Open it.,,,,,,,,,,,",
            ",,,,,highlight,DOOR,,,,,,,",
            "A01,Close,Close it.,,,,,,,,,,,",
            ",,,,,highlight,LID,,,,,,,",
            "A02,Done,Done.,,,,,,,,,,,",
            as_written=True,
        )
        self.assertEqual([s.step_id for s in steps], ["A01", "A02"])
        self.assertEqual(
            steps[0].issues,
            [
                (
                    "Step",
                    "sheet row 4 is step A01 too, so it and the rows under it are not "
                    "read: give that step its own ID",
                )
            ],
        )
        self.assertEqual(
            [row["assets"] for row in steps[0].rows],
            [["DOOR"]],
            "the second step's rows are not the first's",
        )

    def test_a_step_cell_that_is_no_step_and_no_section_is_said_and_its_rows_join_no_step(
        self,
    ):
        """``a02`` (lower case), ``A 02`` and ``Section C - Close`` read as rows of the step above: their
        actions were built into it and nothing was said. Added: 2026-11-25"""
        for written in ("a02", "A 02", "A-02", "Section C - Close"):
            steps = _read(
                "A01,Open,Open it.,,,,,,,,,,,",
                ",,,,,highlight,DOOR,,,,,,,",
                "%s,Close,Close it.,,,,,,,,,,," % written,
                ",,,,,highlight,LID,,,,,,,",
                "A03,Done,Done.,,,,,,,,,,,",
                as_written=True,
            )
            self.assertEqual([s.step_id for s in steps], ["A01", "A03"], written)
            self.assertEqual(
                [row["assets"] for row in steps[0].rows], [["DOOR"]], written
            )
            self.assertEqual(
                steps[0].issues,
                [
                    (
                        "Step",
                        "sheet row 4: '%s' is no step ID and no section, so it and the "
                        "rows under it are not read: write a step's ID (A02, IN_01, "
                        "B03A), or a section as SECTION C, a colon, its title"
                        % written,
                    )
                ],
                written,
            )

    def test_an_action_row_above_its_sections_first_step_is_said_at_that_step(self):
        steps = _read(
            ",,,,,highlight,DOOR,,,,,,,",
            "SECTION A: Open",
            ",,,,,highlight,LID,,,,,,,",
            "A01,Open,Open it.,,,,,,,,,,,",
            as_written=True,
        )
        self.assertEqual(
            steps[0].issues,
            [
                (
                    "Action",
                    "sheet row %d: the row stands above its section's first step, so no step reads it: move it "
                    "under its step" % n,
                )
                for n in (2, 4)
            ],
        )

    def test_a_column_the_rows_need_and_the_sheet_lacks_is_said_once(self):
        """A sheet with no Clip column said "use needs its Clip" on every row that plays a take. Added: 2026-11-25"""
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(
                "Step,Step Name,Voice,Action,Target,With,Label,To,Settings\n"
                "A01,Open,Open it.,,,,,,\n,,,use,DOOR,,,,\n,,,use,LID,,,,\n"
                "A02,Close,Close it.,,,,,,\n,,,play,,,,,\n"
            )
        try:
            steps = Mapping.resolve(path, name="interactive")
        finally:
            os.remove(path)
        self.assertEqual(_issues(steps), [])
        self.assertEqual(
            [i for step in steps for i in step.issues],
            [
                (
                    "Action",
                    "the sheet has no Clip column, which play and use need: add the column",
                ),
                (
                    "Action",
                    "the sheet has no Members column, which play needs: add the column",
                ),
            ],
        )


class TestSentences(unittest.TestCase):
    ACTIONS = Mapping.load_mapping("interactive")["columns"]["actions"]

    def _say(self, **row):
        row.setdefault("settings", {})
        return ManifestActions.sentence(row, self.ACTIONS)

    def test_a_row_reads_back_as_the_sentence_its_cells_mean(self):
        self.assertEqual(
            self._say(word="use", assets=["LATCH"], **{"with": ["KEY"]}, clip="A01"),
            "Selecting LATCH with KEY in hand plays A01",
        )
        self.assertEqual(
            self._say(word="use", assets=["DOOR"], clip="A02"),
            "Selecting DOOR plays A02",
        )
        self.assertEqual(
            self._say(
                word="use", assets=["BUTTON"], clip="A07", settings={"trigger": "hold"}
            ),
            "Holding BUTTON plays A07",
        )
        self.assertEqual(
            self._say(word="highlight", assets=["A", "B", "C"]),
            "A, B and C is highlighted",
        )
        self.assertEqual(
            self._say(word="answer", label="*", to="C10.5"),
            "Entering anything else goes to step C10.5",
        )
        self.assertEqual(
            self._say(word="pip", assets=["LAMP"], settings={"until": "B07"}),
            "A picture shows LAMP, until B07",
        )
        self.assertEqual(
            self._say(
                word="teleport", assets=["CABINET"], settings={"where": "VIEW_FRONT"}
            ),
            "The user is brought to VIEW_FRONT",
        )
        self.assertEqual(
            self._say(word="teleport", assets=["CABINET"]),
            "The user is brought to face CABINET",
        )
        self.assertEqual(self._say(word="note"), "")
        self.assertEqual(self._say(word="nothing"), "")


class TestShots(unittest.TestCase):
    ROWS = (
        "A01,Unlock,Unlock both latches.,,,highlight,LATCHES,,,,,,,",
        ",,,,,take,KEY,,,,,,,",
        ",,,,,use,LATCH_UPPER,KEY,A01_1,,,,,",
        ",,,,,use,LATCH_LOWER,KEY,A01_2,,,,,",
        "A02,Open,Open the door.,,,highlight,DOOR,,,,,,,",
        ",,,,,use,HANDLE,,A02,HANDLE\nDOOR,,,,",
        "B01,Remove,Remove the bolt.,,,use,BOLT,WRENCH,B01_1,,,,,",
        ",,,,,fade_out,BOLT,,B01_1,,,,,",
        ",,,,,play,,,B01_2,LID,,,after=B01_1,",
        "B02,Ask,Ready?,,,button,,,,,Next,next,,",
    )

    def _steps(self, **options):
        rows = tuple(r.replace("HANDLE\nDOOR", '"HANDLE\nDOOR"') for r in self.ROWS)
        return _read(*rows, **options)

    def test_a_step_is_one_shot_and_names_the_takes_its_rows_play(self):
        # The maintainer (2026-11-25): a shot is its step's, named A01, never split into parts; its
        # either-order actions each still play a take of their own, inside it.
        steps = self._steps()
        self.assertEqual(_issues(steps), [])
        self.assertEqual(
            [(s.step_id, s.animated, s.takes) for s in steps],
            [
                ("A01", True, ["A01_1", "A01_2"]),
                ("A02", True, ["A02"]),
                ("B01", True, ["B01_1", "B01_2"]),
                ("B02", True, []),
            ],
        )
        # A step with no row under it but its own plays none either (a production sheet: five such steps shipped
        # their narration's shot as a take of nothing).
        bare = _read(
            "A01,Welcome,Welcome.,,,,,,,,,,,", "A02,Open,Open it.,,,use,DOOR,,A02,,,,,"
        )
        self.assertEqual(
            [(s.step_id, s.takes) for s in bare], [("A01", []), ("A02", ["A02"])]
        )
        by_id = {s.step_id: s for s in steps}
        # What its takes animate, and what a build keys on it, are the step's shot's: a look that names its take
        # (B01's fade) is keyed into it; one that names none (A01's highlight) is its step's, the task's to build.
        self.assertEqual(
            [(o.name, o.behaviors) for o in by_id["A01"].objects],
            [("LATCHES", []), ("LATCH_UPPER", []), ("KEY", []), ("LATCH_LOWER", [])],
        )
        self.assertEqual(
            [(o.name, o.behaviors) for o in by_id["B01"].objects],
            [("BOLT", ["fade_out"]), ("WRENCH", []), ("LID", [])],
        )

    def test_a_steps_narration_is_its_own_whether_or_not_it_plays_a_take(self):
        by_id = {s.step_id: s for s in self._steps(audio="derive")}

        def clips(step_id):
            return [o.name for o in by_id[step_id].objects if o.kind == "audio"]

        self.assertEqual(clips("A01"), ["A01_UnlockBothLatches"])
        self.assertEqual(clips("A02"), ["A02_OpenTheDoor"])
        self.assertEqual(
            clips("B02"),
            ["B02_Ready"],
            "a step that only speaks has its shot to hold it",
        )

    def test_the_shots_option_leaves_a_step_that_plays_no_take_without_a_shot(self):
        by_id = {s.step_id: s for s in self._steps(shots="clip")}
        self.assertEqual(
            [(i, s.animated) for i, s in by_id.items()],
            [("A01", True), ("A02", True), ("B01", True), ("B02", False)],
        )

    def test_a_linear_sheet_makes_every_step_a_shot_of_its_rows_targets(self):
        steps = _read(
            "A01,Open,The cover comes off.,,,fade_out,COVER_BOLT,,,,,,,",
            ",,,,,,ACCESS_COVER,,,,,,,",
            ",,,,,highlight,FILTER,,,,,,style=steady,",
            template="linear",
        )
        self.assertEqual(_issues(steps), [])
        self.assertTrue(steps[0].animated)
        self.assertEqual(
            [(o.name, o.behaviors) for o in steps[0].objects],
            [
                ("COVER_BOLT", ["fade_out"]),
                ("ACCESS_COVER", []),
                ("FILTER", ["highlight"]),
            ],
        )
        self.assertIsNone(
            steps[0].takes,
            "a template with no word that plays a take: the shot is its own",
        )

    def test_a_linear_sheet_of_sentences_still_reads_its_sentences(self):
        # Linear says what a row with no Action is, but only a sheet with an Action column is in cells.
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(
                "Step,Step Contents,Asset Names\nA01,The bolts fade out.,COVER_BOLT\n"
            )
        try:
            steps = Mapping.resolve(path, name="linear")
        finally:
            os.remove(path)
        self.assertEqual(
            [(o.name, o.behaviors) for o in steps[0].objects],
            [("COVER_BOLT", ["fade_out"])],
        )
        self.assertNotIn("cells", steps[0].rows[0])
        self.assertIsNone(steps[0].takes)


class TestBuiltFromCells(unittest.TestCase):
    """The sheet's cells built into shots by the host-free engine, as a shot builder reads it."""

    def _steps(self, *rows, **options):
        path = _sheet(*rows)
        try:
            return Mapping.resolve(path, name="interactive", options=options or None)
        finally:
            os.remove(path)

    def _build(self, store, *rows, **options):
        from pythontk.core_utils.engines.shots.manifest.manifest_engine import (
            ShotManifest,
        )

        steps = self._steps(*rows, **options)
        actions, _, _ = ShotManifest(store).sync(steps, initial_shot_length=100)
        return steps, actions

    def test_a_build_makes_a_shot_per_step_holding_its_takes(self):
        from pythontk.core_utils.engines.shots.manifest.manifest_engine import (
            ShotManifest,
        )
        from pythontk.core_utils.engines.shots.shot_model import ShotStore

        store = ShotStore()
        steps, actions = self._build(store, *TestShots.ROWS)
        self.assertEqual(
            [s.name for s in store.sorted_shots()], ["A01", "A02", "B01", "B02"]
        )
        self.assertEqual(set(actions.values()), {"created"})
        by_name = {s.name: s for s in store.sorted_shots()}
        a01, a02 = by_name["A01"], by_name["A02"]
        # Two takes share their step's shot, in row order, until someone says where each runs; one is the whole shot.
        mid = store.snap((a01.start + a01.end) / 2)
        self.assertEqual(
            store.takes_of(a01), [("A01_1", a01.start, mid), ("A01_2", mid, a01.end)]
        )
        self.assertEqual(store.takes_of(a02), [("A02", a02.start, a02.end)])
        self.assertEqual(
            store.takes_of(by_name["B02"]), [], "a step that only speaks plays none"
        )
        ShotStore.set_active(store)
        self.addCleanup(ShotStore.clear_active)
        self.assertEqual(
            [t["name"] for t in ShotStore.declared_takes()],
            ["A01_1", "A01_2", "A02", "B01_1", "B01_2"],
        )
        # Assessed, nothing is left to record; built again from the same sheet, nothing changes.
        self.assertFalse(
            [
                s.step_id
                for s in ShotManifest(store).assess(steps)
                if "takes" in s.updates
            ]
        )
        _, again = self._build(store, *TestShots.ROWS)
        self.assertFalse(
            [k for k, v in again.items() if v in ("created", "patched")], again
        )

    def test_a_take_someone_placed_keeps_its_span_and_a_new_one_takes_the_room_after_it(
        self,
    ):
        from pythontk.core_utils.engines.shots.manifest.manifest_engine import (
            ShotManifest,
        )
        from pythontk.core_utils.engines.shots.shot_model import ShotStore

        store = ShotStore()
        one = ("A05,Open,Open it.,,,use,DOOR,,A05,,,,,",)
        self._build(store, *one)
        shot = store.shot_by_name("A05")
        store.set_takes(
            shot.shot_id, [("A05", shot.start, shot.start + 40)]
        )  # an artist's
        two = one + (",,,,,use,LID,,A05_2,,,,,",)
        self.assertIn(
            "takes",
            ShotManifest(store).assess(self._steps(*two))[0].updates,
            "said before a build",
        )
        self._build(store, *two)
        self.assertEqual(
            [s.name for s in store.sorted_shots()],
            ["A05"],
            "no shot is made for a take",
        )
        self.assertEqual(
            store.takes_of(shot),
            [
                ("A05", shot.start, shot.start + 40),
                ("A05_2", shot.start + 40, shot.end),
            ],
        )
        self._build(store, *one)
        self.assertEqual(
            store.takes_of(shot),
            [("A05", shot.start, shot.start + 40)],
            "a take the sheet dropped goes",
        )

    def test_takes_cut_as_shots_of_their_own_are_joined_into_their_steps_shot(self):
        from pythontk.core_utils.engines.shots.manifest.manifest_engine import (
            ShotManifest,
        )
        from pythontk.core_utils.engines.shots.shot_model import ShotStore

        # As a scene made before stands: the step's shot cut in two by hand and renamed, the first
        # half still bound to the step.
        store = ShotStore()
        first = store.define_shot(
            "A01_1", 1, 40, objects=["LATCH_UPPER", "KEY"], metadata={"step": "A01"}
        )
        store.define_shot("A01_2", 50, 90, objects=["LATCH_LOWER", "KEY"])
        store.define_shot("A02", 100, 160, objects=["HANDLE"])
        steps = self._steps(*TestShots.ROWS[:6])
        manifest = ShotManifest(store)
        pairing = manifest.pair(steps)
        self.assertEqual(
            {sid: shot.name for sid, shot in pairing.shots.items()},
            {"A01": "A01_1", "A02": "A02"},
        )
        self.assertEqual(
            pairing.links,
            {store.shot_by_name("A01_2").shot_id: "A01"},
            "its other take is the step's too",
        )
        self.assertIn(
            "takes", {s.step_id: s for s in manifest.assess(steps)}["A01"].updates
        )

        # Built as a panel builds: each paired step's range read first, as its shot stands (the first part's).
        from pythontk.core_utils.engines.shots.manifest.range_resolver import (
            RangeResolver,
        )

        ranges = RangeResolver.incremental_ranges(
            steps, {sid: (s.start, s.end) for sid, s in pairing.shots.items()}
        )
        manifest.sync(
            steps,
            ranges=ranges,
            initial_shot_length=100,
            remove_missing=False,
            zero_duration_fallback=True,
        )
        self.assertEqual([s.name for s in store.sorted_shots()], ["A01", "A02"])
        joined = store.shot_by_name("A01")
        self.assertEqual(joined.shot_id, first.shot_id, "the step's shot, kept")
        self.assertEqual((joined.start, joined.end), (1, 90))
        self.assertEqual(
            store.takes_of(joined),
            [("A01_1", 1, 40), ("A01_2", 50, 90)],
            "each where it was cut",
        )
        self.assertTrue({"KEY", "LATCH_LOWER", "LATCH_UPPER"} <= set(joined.objects))
        a02 = store.shot_by_name("A02")
        self.assertEqual(store.takes_of(a02), [("A02", a02.start, a02.end)])

    def test_takes_with_another_shot_between_them_are_left_and_said(self):
        from pythontk.core_utils.engines.shots.manifest.manifest_engine import (
            ShotManifest,
        )
        from pythontk.core_utils.engines.shots.shot_model import ShotStore

        store = ShotStore()
        store.define_shot("A01_1", 1, 40)
        store.define_shot("Other", 50, 90)
        store.define_shot("A01_2", 100, 140)
        steps = self._steps(*TestShots.ROWS[:4])
        with self.assertLogs("pythontk", level="WARNING") as logged:
            ShotManifest(store).sync(
                steps, initial_shot_length=100, remove_missing=False
            )
        self.assertTrue(
            any("A01" in line and "Other" in line for line in logged.output),
            logged.output,
        )
        self.assertEqual(
            [s.name for s in store.sorted_shots()],
            ["A01_1", "Other", "A01_2"],
            "nothing joined across it",
        )

    def test_the_shots_option_decides_which_steps_get_a_shot(self):
        from pythontk.core_utils.engines.shots.shot_model import ShotStore

        for rule, names in (("clip", ["A01"]), ("every", ["A01", "B02"])):
            store = ShotStore()
            self._build(store, *TestShots.ROWS[:4], TestShots.ROWS[-1], shots=rule)
            self.assertEqual([s.name for s in store.sorted_shots()], names, rule)


if __name__ == "__main__":
    unittest.main()
