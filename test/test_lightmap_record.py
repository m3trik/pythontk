#!/usr/bin/python
# coding=utf-8
"""``LightmapRecord``: the ``lightmaps`` record's model, written in one shape.

Both hosts' bakers publish the record and every reader takes it (the GLB
build, unitytk's importer), so its shape is declared once. Version 3
(2026-11-07) states each thing once -- one record-level ``unit_scale``, the
probe's box as ``{min, max}``, no direction on a point light, and no UV set:
the contract reads every map through the second -- and a reader reads that
version alone.
"""

import copy
import unittest

from pythontk import LightmapRecord, SceneRecords

OBJECTS = [
    {
        "name": "room",
        "hierarchy": ["ROOM", "room"],
        "map": "room_Lightmap.exr",
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

    def test_an_entry_names_no_uv_set_the_contract_reads_the_second(self):
        """Version 3 (2026-11-07): every map is read through the second UV set
        (index 1), where an engine reads a lightmap, so an entry names
        none -- a producer's ``uvIndex`` is left out and the cut warned -- and
        the record's contract states the set (a producer whose set sits
        elsewhere refuses to publish: ``LightmapRecord.check_uv_sets``)."""
        from pythontk.core_utils.engines.scene_export.lightmap_record import (
            LightmapEntry,
        )

        self.assertNotIn("uvIndex", LightmapEntry.json_schema()["properties"])
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ) as logs:
            record = LightmapRecord.payload([dict(OBJECTS[0], uvIndex=1)])
        self.assertEqual(record["objects"], OBJECTS)
        self.assertIn("objects[].uvIndex", "\n".join(logs.output))
        self.assertEqual(LightmapRecord.UV_SET, 1)
        self.assertIn("second UV set (index 1)", SceneRecords.LIGHTMAPS.assembly)

    def test_a_uv_set_off_the_second_slot_is_refused_with_its_fix(self):
        """``check_uv_sets``: the one refusal both hosts' producers raise --
        each object whose lightmap set sits elsewhere named with its set and
        slot, and the host's fix -- and nothing for sets in the second slot."""
        LightmapRecord.check_uv_sets([("crate", "lightmap", 1)], fix="Move it.")
        with self.assertRaises(LightmapRecord.UvSetRefused) as caught:
            LightmapRecord.check_uv_sets(
                [("crate", "lightmap", 1), ("wall", "UV2", 2), ("door", "lm", 0)],
                fix="Re-run create_lightmap_uvs on them.",
            )
        text = str(caught.exception)
        self.assertIsInstance(caught.exception, ValueError)
        for part in (
            "second UV set (index 1)",
            "wall ('UV2' at index 2)",
            "door ('lm' at index 0)",
            "Re-run create_lightmap_uvs on them.",
        ):
            with self.subTest(part=part):
                self.assertIn(part, text)
        self.assertNotIn("crate", text)

    def test_a_committed_set_gone_since_the_bake_is_refused(self):
        """A set renamed or removed since the bake (index ``None``) is where
        the second slot may hold another set: refused like one elsewhere,
        never published on faith (it was a warning). Added: 2026-11-07"""
        with self.assertRaises(LightmapRecord.UvSetRefused) as caught:
            LightmapRecord.check_uv_sets(
                [("wall", "lightmap", None)], fix="Bake again."
            )
        text = str(caught.exception)
        self.assertIn("wall ('lightmap', which it no longer has)", text)
        self.assertIn("Bake again.", text)

    def test_an_exports_producer_notes_its_refusal(self):
        """An export's producer says a refusal on its context's notes
        (``note=ctx.note``) and publishes none, so the delivered record is
        cleared -- a raise was caught as a failed producer and the stale
        stored record shipped. Logged by the snapshot, once: not here too.
        Added: 2026-11-07"""
        import logging

        host = logging.getLogger("test_lightmap_record.export_note")
        notes = []

        def refused():
            LightmapRecord.check_uv_sets([("wall", "UV2", 2)], fix="Move it.")

        with self.assertNoLogs(host, level="WARNING"):
            self.assertIsNone(
                LightmapRecord.unless_refused(refused, host, note=notes.append)
            )
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("wall ('UV2' at index 2)", notes[0])
        self.assertIn("Move it.", notes[0])

    def test_an_authoring_publish_ships_no_refused_record(self):
        """``unless_refused``: what a bake's or a revert's publish ships -- the
        record built, or none when it is refused, the refusal said on the
        host's logger -- so the work that published goes on and no write
        carries a record that reads a map through the wrong UVs. Anything
        else a build raises is no refusal and propagates."""
        import logging

        host = logging.getLogger("test_lightmap_record.unless_refused")

        def refused():
            LightmapRecord.check_uv_sets([("wall", "UV2", 2)], fix="Move it.")

        self.assertEqual(LightmapRecord.unless_refused(lambda: "record", host), "record")
        with self.assertLogs(host, level="WARNING") as logs:
            self.assertIsNone(LightmapRecord.unless_refused(refused, host))
        self.assertIn("wall ('UV2' at index 2)", logs.output[0])
        with self.assertRaises(KeyError):
            LightmapRecord.unless_refused(lambda: {}["x"], host)

    def test_the_record_is_stamped_with_its_declared_version(self):
        record = LightmapRecord.payload(OBJECTS)
        self.assertEqual(record["version"], SceneRecords.LIGHTMAPS.version)
        self.assertEqual(SceneRecords.LIGHTMAPS.version, 3)
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


def _descriptions(node):
    """Every ``description`` in a JSON Schema *node*, at any depth."""
    if isinstance(node, dict):
        found = (
            [node["description"]] if isinstance(node.get("description"), str) else []
        )
        for value in node.values():
            found += _descriptions(value)
        return found
    if isinstance(node, list):
        return [d for value in node for d in _descriptions(value)]
    return []


class TestTheTextSaysEachThingOnce(unittest.TestCase):
    """The reference generated from this declaration (``DELIVERABLE_FORMAT.md``,
    the schema, each reader's types) states each rule once, where it belongs:
    a reading rule in the record's assembly, which rides in the file, and in
    each field's help what the field holds. Measured 2026-11-07: "a reader
    lights none of 'lights' while a probe is present" four times, the
    highlight copies' meaning three, the second UV set in four wordings.
    Added: 2026-11-07"""

    @staticmethod
    def _text():
        spec = SceneRecords.LIGHTMAPS
        return " ".join(
            [spec.description, spec.assembly_for()]
            + _descriptions(LightmapRecord.json_schema())
        )

    def test_each_rule_is_stated_once(self):
        text = self._text()
        for phrase in (
            "lights none of",
            "UV set",
            "in the probe's place",
            "one copy at most",
            "box projection shows them",
            "specular only",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(text.count(phrase), 1, phrase)
        self.assertIn("second UV set (index 1)", SceneRecords.LIGHTMAPS.assembly)

    def test_no_generated_reader_is_told_of_a_field_it_does_not_hold(self):
        """Each reader's types leave out the extensions another reader is
        served ('lights' and 'probe.highlights' are one reader's), so no help
        a reader's types keep names a field they drop: a generated
        ``Probe.map`` doc named both. Added: 2026-11-07"""
        whole = LightmapRecord.json_schema()

        def keys(node):
            if isinstance(node, dict):
                out = set(node.get("properties", {}))
                for value in node.values():
                    out |= keys(value)
                return out
            if isinstance(node, list):
                return set().union(*(keys(v) for v in node)) if node else set()
            return set()

        narrowed = 0
        for reader in SceneRecords.LIGHTMAPS.consumers:
            view = LightmapRecord.json_schema(reader=reader)
            dropped = keys(whole) - keys(view)
            narrowed += bool(dropped)
            for help_text in _descriptions(view):
                for name in dropped:
                    with self.subTest(reader=reader, name=name, help=help_text[:50]):
                        self.assertNotIn(f"'{name}'", help_text)
        self.assertTrue(narrowed, "some reader's types leave an extension out")

    def test_the_occlusion_rule_is_the_producers_alone(self):
        """A material only lightmapped objects wear ships without its occlusion
        map; a shared one keeps it, warned: the producer's rule, stated where
        the materials record states occlusion -- no reader applies one, and
        the lightmaps' assembly no longer says the material stays as
        authored. Added: 2026-11-07"""
        from pythontk.core_utils.engines.scene_export.material_record import (
            MaterialOcclusionTextureInfo,
        )

        doc = " ".join(MaterialOcclusionTextureInfo.__doc__.split())
        self.assertNotIn("takes none", doc)
        for part in ("only lightmapped objects wear", "keeps it", "no reader"):
            with self.subTest(part=part):
                self.assertIn(part, doc)
        assembly = SceneRecords.LIGHTMAPS.assembly_for()
        self.assertNotIn("as authored", assembly)
        self.assertNotIn("lightmappedMaterials", assembly)


if __name__ == "__main__":
    unittest.main()
