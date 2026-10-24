#!/usr/bin/python
# coding=utf-8
"""``LightmapRecord``: the ``lightmaps`` record's model, written in one shape.

Both hosts' bakers publish the record and every reader takes it (the GLB
build, unitytk's importer), so its shape is declared once. Version 2
(2026-10-20) states each thing once -- one record-level ``unit_scale``, the
probe's box as ``{min, max}``, no direction on a point light -- and a reader
reads that version alone.
"""

import copy
import unittest

from pythontk import LightmapRecord, SceneRecords

OBJECTS = [
    {
        "name": "room",
        "hierarchy": ["ROOM", "room"],
        "map": "room_Lightmap.exr",
        "uvIndex": 1,
        "scaleOffset": [1, 1, 0, 0],
    }
]


class TestPayload(unittest.TestCase):
    """What a producer publishes (:meth:`LightmapRecord.payload`)."""

    def test_nothing_baked_is_no_record(self):
        self.assertIsNone(LightmapRecord.payload([]))
        self.assertIsNone(LightmapRecord.payload(None, probe={"map": "p.exr"}))

    def test_an_entry_states_what_its_shape_declares_and_no_more(self):
        """An object's level is its material's in the GLB block, not the
        entry's: a producer's ``intensity`` (stated until 2026-10-20) is left
        out, and the cut warned -- its drift said, never shipped."""
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ) as logs:
            record = LightmapRecord.payload([dict(OBJECTS[0], intensity=1.0)])
        self.assertEqual(record["objects"], OBJECTS)
        self.assertIn("objects[].intensity", "\n".join(logs.output))

    def test_the_record_is_stamped_with_its_declared_version(self):
        record = LightmapRecord.payload(OBJECTS)
        self.assertEqual(record["version"], SceneRecords.LIGHTMAPS.version)
        self.assertEqual(SceneRecords.LIGHTMAPS.version, 2)
        self.assertEqual(record["objects"], OBJECTS)
        self.assertNotIn("unit_scale", record, "no lengths, no unit to state")
        self.assertNotIn("probe", record)
        self.assertNotIn("lights", record)

    def test_one_unit_for_every_length_the_record_holds(self):
        """The probe's lengths and each light's are in the file's unit, so the
        record states that unit once -- never again on the probe or a light."""
        record = LightmapRecord.payload(
            OBJECTS,
            probe={
                "map": "room_Probe.exr",
                "position": [0, 150, 0],
                "unit_scale": 0.01,
            },
            lights=[
                {
                    "type": "point",
                    "position": [0, 300, 0],
                    "unit_scale": 0.01,
                }
            ],
            unit_scale=0.01,
        )
        self.assertEqual(record["unit_scale"], 0.01)
        self.assertNotIn("unit_scale", record["probe"])
        self.assertNotIn("unit_scale", record["lights"][0])

    def test_a_probe_or_lights_need_their_unit(self):
        for unit in (None, 0, -1.0, "cm"):
            with self.subTest(unit=unit):
                with self.assertRaises(ValueError):
                    LightmapRecord.payload(
                        OBJECTS, probe={"map": "p.exr"}, unit_scale=unit
                    )
                with self.assertRaises(ValueError):
                    LightmapRecord.payload(
                        OBJECTS, lights=[{"type": "point"}], unit_scale=unit
                    )

    def test_the_probe_box_is_min_and_max(self):
        """``[min, max]`` points arrive as the GLB carries them: ``{min, max}``."""
        for box in ([[-1, 0, -1], [1, 2, 1]], {"min": [-1, 0, -1], "max": [1, 2, 1]}):
            with self.subTest(box=box):
                record = LightmapRecord.payload(
                    OBJECTS,
                    probe={"map": "p.exr", "position": [0, 1, 0], "box": box},
                    unit_scale=1.0,
                )
                self.assertEqual(
                    record["probe"]["box"],
                    {"min": [-1.0, 0.0, -1.0], "max": [1.0, 2.0, 1.0]},
                )

    def test_only_a_point_light_loses_its_direction(self):
        """A point light radiates every way, so a direction on one states
        nothing; every other type keeps its own -- an aimed light, and a type
        no version knows yet (a directional sun): the one field dropped is the
        one known to mean nothing. ``-0.0`` reads as ``0.0``."""
        record = LightmapRecord.payload(
            OBJECTS,
            lights=[
                {
                    "type": "point",
                    "position": [-0.0, 2.0, 0.0],
                    "direction": [0.0, -1.0, 0.0],
                    "right": [1.0, 0.0, 0.0],
                },
                {
                    "type": "spot",
                    "position": [1.0, 2.0, 3.0],
                    "direction": [0.0, -1.0, -0.0],
                },
                {
                    "type": "directional",
                    "direction": [0.0, -1.0, 0.0],
                    "right": [-0.0, 0.0, 1.0],
                },
            ],
            unit_scale=1.0,
        )
        bulb, spot, sun = record["lights"]
        self.assertNotIn("direction", bulb)
        self.assertNotIn("right", bulb)
        self.assertEqual(bulb["position"], [0.0, 2.0, 0.0])
        self.assertEqual(str(bulb["position"][0]), "0.0")
        self.assertEqual(spot["direction"], [0.0, -1.0, 0.0])
        self.assertEqual(str(spot["direction"][2]), "0.0")
        self.assertEqual(sun["direction"], [0.0, -1.0, 0.0])
        self.assertEqual(sun["right"], [0.0, 0.0, 1.0])
        self.assertEqual(str(sun["right"][0]), "0.0")

    def test_the_callers_entries_are_not_changed(self):
        probe = {
            "map": "p.exr",
            "position": [0, 1, 0],
            "box": [[0, 0, 0], [1, 1, 1]],
            "unit_scale": 1.0,
        }
        lights = [{"type": "point", "position": [0, 1, 0], "direction": [0, -1, 0]}]
        before = copy.deepcopy((OBJECTS, probe, lights))
        LightmapRecord.payload(OBJECTS, probe=probe, lights=lights, unit_scale=1.0)
        self.assertEqual((OBJECTS, probe, lights), before)


class TestCurrent(unittest.TestCase):
    """What a reader reads: the current schema alone
    (``SceneRecords.LIGHTMAPS.is_current``). A record of any other version --
    an older one too -- is refused rather than read through a branch for its
    shape; the cure is a re-export."""

    def test_what_a_producer_publishes_is_current(self):
        record = LightmapRecord.payload(
            OBJECTS, probe={"map": "p.exr", "position": [0, 1, 0]}, unit_scale=1.0
        )
        self.assertTrue(SceneRecords.LIGHTMAPS.is_current(record))

    def test_every_other_version_is_refused(self):
        current = SceneRecords.LIGHTMAPS.version
        for version in (current - 1, current + 1, str(current), None):
            with self.subTest(version=version):
                record = {"version": version, "objects": OBJECTS}
                self.assertFalse(SceneRecords.LIGHTMAPS.is_current(record))
        for record in ({"objects": OBJECTS}, None, ["not", "a", "record"]):
            with self.subTest(record=record):
                self.assertFalse(SceneRecords.LIGHTMAPS.is_current(record))


class TestBox(unittest.TestCase):
    """:meth:`LightmapRecord.box`: either spelling of a box, as ``{min, max}``."""

    def test_either_spelling_reads_as_min_and_max(self):
        want = {"min": [-1.0, 0.0, -1.0], "max": [1.0, 2.0, 1.0]}
        self.assertEqual(LightmapRecord.box([[-1, 0, -1], [1, 2, 1]]), want)
        self.assertEqual(LightmapRecord.box(((-1, 0, -1), (1, 2, 1))), want)
        self.assertEqual(
            LightmapRecord.box({"min": [-1, 0, -1], "max": [1, 2, 1]}), want
        )

    def test_an_open_scene_or_a_malformed_box_has_none(self):
        for box in (
            None,
            [],
            [[0, 0, 0]],
            [[0, 0], [1, 1]],
            [[0, 0, "x"], [1, 1, 1]],
            {"min": [0, 0, 0]},
            "box",
        ):
            with self.subTest(box=box):
                self.assertIsNone(LightmapRecord.box(box))

    def test_a_negative_zero_corner_reads_as_zero(self):
        box = LightmapRecord.box([[-0.0, 0.0, -0.0], [1, 1, 1]])
        self.assertEqual([str(c) for c in box["min"]], ["0.0", "0.0", "0.0"])


if __name__ == "__main__":
    unittest.main()
