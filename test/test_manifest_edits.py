# !/usr/bin/python
# coding=utf-8
"""Tests for ``ManifestEdits`` (``core_utils/engines/shots/manifest/manifest_edits.py``):
a user's edits, recorded against the sheet's own identities and composed over
freshly parsed steps -- never written into them.
"""

import os
import sys
import unittest

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pythontk.core_utils.engines.shots.manifest.manifest_edits import ManifestEdits
from pythontk.core_utils.engines.shots.manifest.manifest_engine import ShotManifest
from pythontk.core_utils.engines.shots.manifest.manifest_model import (
    BuilderObject,
    BuilderStep,
)
from pythontk.core_utils.engines.shots.shot_model import ShotStore


def _sheet():
    """Three steps, as a parse gives them."""
    return [
        BuilderStep(
            "A01",
            "A",
            "Access",
            "Open the door",
            [
                BuilderObject("door", behaviors=["fade_in"]),
                BuilderObject("bolt01"),
            ],
        ),
        BuilderStep("A02", "A", "Access", "Unhook", [BuilderObject("latch")]),
        BuilderStep("B01", "B", "Repair", "Swap", [BuilderObject("solenoid")]),
    ]


def _names(steps):
    return {s.step_id: [o.name for o in s.objects] for s in steps}


class TestApply(unittest.TestCase):
    def test_no_edits_is_the_sheet_as_is_and_never_the_same_objects(self):
        sheet = _sheet()
        out = ManifestEdits().apply(sheet)
        self.assertEqual(out, sheet)
        self.assertIsNot(out[0], sheet[0])

    def test_the_sheet_is_never_changed(self):
        sheet = _sheet()
        edits = ManifestEdits()
        edits.set_step("A01", "name", "Door_Open")
        edits.set_object("A01", "bolt01", "name", "bolt_01")
        edits.set_object("A01", "door", "step", "A02")
        edits.apply(sheet)
        self.assertEqual(sheet, _sheet())

    def test_step_rename_and_description_keep_the_id(self):
        edits = ManifestEdits()
        edits.set_step("A01", "name", "Door_Open")
        edits.set_step("A01", "description", "Open it", sheet="Open the door")
        step = edits.apply(_sheet())[0]
        self.assertEqual((step.step_id, step.shot_name), ("A01", "Door_Open"))
        self.assertEqual(step.description, "Open it")
        self.assertEqual(step.sheet, {"name": "A01", "description": "Open the door"})

    def test_object_rename_points_the_step_at_another_name(self):
        edits = ManifestEdits()
        edits.set_object("A01", "bolt01", "name", "bolt_01")
        step = edits.apply(_sheet())[0]
        self.assertEqual([o.name for o in step.objects], ["door", "bolt_01"])
        self.assertEqual(step.objects[1].sheet, {"name": "bolt01"})

    def test_a_rename_onto_another_objects_sheet_name_keeps_both_edits(self):
        edits = ManifestEdits()
        edits.set_object("A01", "door", "name", "bolt01")
        edits.set_object("A01", "bolt01", "name", "door")
        step = edits.apply(_sheet())[0]
        self.assertEqual([o.name for o in step.objects], ["bolt01", "door"])
        self.assertEqual(step.objects[0].behaviors, ["fade_in"])

    def test_hide_and_move_objects(self):
        edits = ManifestEdits()
        edits.set_object("A01", "bolt01", "hidden", True)
        edits.set_object("A01", "door", "step", "B01")
        out = edits.apply(_sheet())
        self.assertEqual(
            _names(out), {"A01": [], "A02": ["latch"], "B01": ["solenoid", "door"]}
        )
        moved = out[2].objects[1]
        self.assertEqual(moved.sheet, {"step": "A01"})
        self.assertEqual(moved.behaviors, ["fade_in"])  # moves whole
        self.assertEqual(ManifestEdits.anchor(out[2], moved), ("A01", "door"))
        self.assertEqual(edits.hidden_objects("A01"), [("A01", "bolt01")])

    def test_a_hidden_object_lists_where_it_was_moved(self):
        edits = ManifestEdits()
        edits.set_object("A01", "door", "step", "B01")
        edits.set_object("A01", "door", "hidden", True)
        self.assertEqual(edits.hidden_objects("A01"), [])
        self.assertEqual(edits.hidden_objects("B01"), [("A01", "door")])
        self.assertEqual(_names(edits.apply(_sheet()))["B01"], ["solenoid"])

    def test_behaviors_edit_records_the_sheets(self):
        edits = ManifestEdits()
        edits.set_object("A01", "door", "behaviors", ["fade_out"], sheet=["fade_in"])
        obj = edits.apply(_sheet())[0].objects[0]
        self.assertEqual(obj.behaviors, ["fade_out"])
        self.assertEqual(obj.sheet, {"behaviors": ["fade_in"]})

    def test_hidden_steps_are_left_out(self):
        edits = ManifestEdits()
        edits.set_step("A02", "hidden", True)
        self.assertEqual([s.step_id for s in edits.apply(_sheet())], ["A01", "B01"])
        self.assertEqual(edits.hidden_steps(), ["A02"])

    def test_added_steps_go_after_their_anchor_and_take_its_section(self):
        edits = ManifestEdits()
        new = edits.add_step("A02", taken=["A01", "A02", "B01"])
        top = edits.add_step("", taken=["A01", "A02", "B01"])
        self.assertEqual((new, top), ("NEW_STEP", "NEW_STEP_2"))
        edits.set_object("A01", "door", "step", new)
        out = edits.apply(_sheet())
        self.assertEqual(
            [s.step_id for s in out], ["NEW_STEP_2", "A01", "A02", "NEW_STEP", "B01"]
        )
        self.assertEqual((out[3].section, out[3].section_title), ("A", "Access"))
        self.assertEqual([o.name for o in out[3].objects], ["door"])
        self.assertTrue(edits.is_added(new))
        self.assertFalse(edits.is_added("A01"))

    def test_add_step_numbers_past_names_already_used(self):
        edits = ManifestEdits()
        self.assertEqual(edits.add_step("A01", taken=["new_step"]), "NEW_STEP_2")

    def test_an_added_step_can_take_the_id_it_is_given(self):
        """A shot added to the manifest keeps its name as its step's ID;
        numbered past the steps and names already used, as NEW_STEP is."""
        edits = ManifestEdits()
        self.assertEqual(edits.add_step("A01", step_id="A01_2"), "A01_2")
        self.assertEqual(
            edits.add_step("A01", taken=["a01_3"], step_id="A01_3"), "A01_3_2"
        )
        self.assertEqual(edits.add_step("A01", step_id="A01_2"), "A01_2_2")
        self.assertEqual(edits.add_step("A01", step_id="  "), "NEW_STEP")

    def test_removing_an_added_step_puts_its_objects_home(self):
        edits = ManifestEdits()
        new = edits.add_step("A01")
        later = edits.add_step(new)
        edits.set_step(new, "name", "Prep")
        edits.set_object("A01", "door", "step", new)
        self.assertTrue(edits.remove_step(new))
        self.assertFalse(edits.remove_step("A01"))  # a sheet step: not removable
        self.assertEqual(edits.steps, {})
        self.assertEqual(edits.objects, {})
        # The step added after it takes its place, not the top.
        self.assertEqual(edits.added, [{"step_id": later, "after": "A01"}])
        self.assertEqual(_names(edits.apply(_sheet()))["A01"], ["door", "bolt01"])

    def test_a_move_into_a_step_that_lists_the_name_stays_home(self):
        """The sheet later lists the moved object in its target step: no
        duplicate there -- the object stays where the sheet has it."""
        sheet = _sheet()
        sheet[1].objects.append(BuilderObject("door"))
        edits = ManifestEdits()
        edits.set_object("A01", "door", "step", "A02")
        out = edits.apply(sheet)
        self.assertEqual(_names(out)["A01"], ["bolt01", "door"])
        self.assertEqual(_names(out)["A02"], ["latch", "door"])
        self.assertEqual(out[0].objects[1].sheet, {})
        self.assertEqual(edits.stale(sheet), ["A01 / door: A02 lists it already"])

    def test_stale_edits_are_skipped_and_described(self):
        edits = ManifestEdits()
        edits.set_step("Z99", "name", "Gone")
        edits.set_object("A01", "ghost", "name", "x")
        edits.set_object("A01", "door", "step", "Z99")
        self.assertEqual(edits.apply(_sheet()), _sheet())
        self.assertEqual(
            edits.stale(_sheet()),
            [
                "Z99: no such step",
                "A01 / ghost: no such object",
                "A01 / door: no step Z99 to move to",
            ],
        )


class TestEditing(unittest.TestCase):
    def test_setting_the_sheets_value_drops_the_edit(self):
        edits = ManifestEdits()
        edits.set_step("A01", "name", "X")
        edits.set_step("A01", "name", "A01")
        edits.set_object("A01", "door", "step", "B01")
        edits.set_object("A01", "door", "step", "A01")
        edits.set_object("A01", "door", "behaviors", ["a"], sheet=("a",))
        self.assertFalse(edits.has_edits)
        self.assertEqual(edits.count(), 0)

    def test_values_are_validated(self):
        edits = ManifestEdits()
        with self.assertRaises(ValueError):
            edits.set_step("A01", "name", "  ")
        with self.assertRaises(ValueError):
            edits.set_step("A01", "section", "x")
        with self.assertRaises(ValueError):
            edits.set_step("A01", "description", "x")  # sheet value required
        with self.assertRaises(ValueError):
            edits.set_object("A01", "door", "behaviors", ["x"])

    def test_revert_one_field_or_all(self):
        edits = ManifestEdits()
        edits.set_step("A01", "name", "X")
        edits.set_step("A01", "hidden", True)
        self.assertTrue(edits.revert_step("A01", "hidden"))
        self.assertEqual(edits.step_record("A01"), {"name": "X"})
        edits.set_object("A01", "door", "name", "d")
        edits.set_object("A01", "door", "hidden", True)
        self.assertTrue(edits.revert_object("A01", "door"))
        self.assertFalse(edits.revert_object("A01", "door"))
        self.assertEqual(edits.count(), 1)

    def test_rename_all_renames_each_occurrence_in_its_own_step(self):
        sheet = _sheet()
        sheet[2].objects.append(BuilderObject("door"))
        edits = ManifestEdits()
        edits.set_object("A01", "door", "step", "A02")
        n = edits.rename_all(edits.apply(sheet), "door", "red_door")
        self.assertEqual(n, 2)
        self.assertEqual(edits.object_record("A01", "door")["name"], "red_door")
        self.assertEqual(edits.object_record("B01", "door"), {"name": "red_door"})
        self.assertEqual(
            _names(edits.apply(sheet)),
            {
                "A01": ["bolt01"],
                "A02": ["latch", "red_door"],
                "B01": ["solenoid", "red_door"],
            },
        )

    def test_name_taken_ignores_case_and_the_step_itself(self):
        steps = _sheet()
        steps[1].name = "Unhook"
        self.assertEqual(ManifestEdits.name_taken(steps, "unhook"), "A02")
        self.assertIsNone(ManifestEdits.name_taken(steps, "unhook", step_id="A02"))
        self.assertIsNone(ManifestEdits.name_taken(steps, "A02"))  # renamed away

    def test_round_trip_and_tolerant_read(self):
        edits = ManifestEdits()
        edits.set_step("A01", "name", "X")
        edits.set_object("A01", "door", "hidden", True)
        edits.add_step("A01")
        again = ManifestEdits.from_dict(edits.to_dict())
        self.assertEqual(again.to_dict(), edits.to_dict())
        junk = {
            "steps": {"A01": {"name": "X", "colour": "red"}, "A02": "bad"},
            "objects": {"A01": {"door": {"bogus": 1}}, "A02": []},
            "added": [{"after": "A01"}, "bad"],
        }
        junk["steps"]["A03"] = {"name": " ", "hidden": "yes", "description": 7}
        junk["objects"]["B01"] = {"x": {"behaviors": "fade_in", "step": ""}}
        read = ManifestEdits.from_dict(junk)  # wrong types are left out too
        self.assertEqual(read.steps, {"A01": {"name": "X"}})
        self.assertEqual((read.objects, read.added), ({}, []))
        self.assertFalse(ManifestEdits.from_dict(None).has_edits)

    def test_a_section_of_the_wrong_shape_is_left_out_not_raised_on(self):
        """Bug: the tolerant reader raised on a saved section of the wrong
        shape (``steps`` a list, ``added`` a number) instead of dropping it.
        Fixed: 2026-10-10"""
        read = ManifestEdits.from_dict(
            {"steps": ["A01"], "objects": ["door"], "added": 3}
        )
        self.assertFalse(read.has_edits)
        read = ManifestEdits.from_dict(
            {"steps": {"A01": {"name": "X"}}, "objects": "door", "added": "A02"}
        )
        self.assertEqual(
            (read.steps, read.objects, read.added), ({"A01": {"name": "X"}}, {}, [])
        )


class TestRebuildThroughEdits(unittest.TestCase):
    """What a build does with edited steps: the shot follows the edits, and
    turning them off (the sheet as-is) puts it back."""

    def test_rename_hide_and_move_reach_the_shots(self):
        store = ShotStore()
        mani = ShotManifest(store)
        mani.update(_sheet(), initial_shot_length=50)
        edits = ManifestEdits()
        edits.set_step("A01", "name", "Door_Open")
        edits.set_object("A01", "bolt01", "hidden", True)
        edits.set_object("A01", "door", "step", "A02")
        mani.update(edits.apply(_sheet()), remove_missing=False)
        self.assertEqual(
            [s.name for s in store.sorted_shots()], ["Door_Open", "A02", "B01"]
        )
        self.assertEqual(store.shot_by_name("Door_Open").objects, [])
        self.assertEqual(sorted(store.shot_by_name("A02").objects), ["door", "latch"])
        # The sheet as-is: the same shots, back the way the sheet has them.
        mani.update(_sheet(), remove_missing=False)
        self.assertEqual([s.name for s in store.sorted_shots()], ["A01", "A02", "B01"])
        self.assertEqual(sorted(store.shot_by_name("A01").objects), ["bolt01", "door"])
        self.assertEqual(store.shot_by_name("A02").objects, ["latch"])
        self.assertEqual(len(store.shots), 3)

    def test_an_added_step_builds_its_own_shot(self):
        store = ShotStore()
        mani = ShotManifest(store)
        mani.update(_sheet(), initial_shot_length=50)
        edits = ManifestEdits()
        new = edits.add_step("A02")
        edits.set_step(new, "name", "Inspect")
        edits.set_object("B01", "solenoid", "step", new)
        actions = mani.update(edits.apply(_sheet()), remove_missing=False)
        self.assertEqual(actions[new], "created")
        shot = store.shot_by_name("Inspect")
        self.assertEqual(shot.metadata["step"], new)
        self.assertEqual(shot.objects, ["solenoid"])
        self.assertEqual(store.shot_by_name("B01").objects, [])


if __name__ == "__main__":
    unittest.main()
