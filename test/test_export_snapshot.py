# !/usr/bin/python
# coding=utf-8
"""ExportContext / ExportSnapshot -- what a write ships, decided once.

Every export writes data for the objects it ships, unless the whole scene is
relevant (the shots, the audio: timeline-level). The host names the nodes the
write ships once (``ExportContext.nodes``) and the snapshot narrows every
export-scoped record by them: the copy its producer returns, and -- on a
hand-off, which refreshes only the derived records -- the copy shipped as
stored, putting the scene's whole copy back after the write.

Runs against an in-memory store; the hosts fill ``nodes`` and are pinned by
their own suites (mayatk / blendertk ``test_fbx_utils``).
"""

import unittest
from typing import Any, Dict

import pythontk as ptk
from pythontk.core_utils.engines.scene_export.export_snapshot import (
    ExportContext,
    ExportSnapshot,
)
from pythontk.core_utils.engines.scene_export.scene_records import (
    SceneRecords as SR,
    Scope,
)
from pythontk.core_utils.engines.scene_export.scene_store import SceneStoreBase


class _Store(SceneStoreBase):
    """One dict per scope, created on first write."""

    data: Dict[Scope, Dict[str, Any]] = {}

    @classmethod
    def read(cls, scope, key):
        value = cls.data.get(Scope(scope), {}).get(key)
        return value if isinstance(value, str) and value else None

    @classmethod
    def write(cls, scope, key, text):
        scope = Scope(scope)
        if not text:
            group = cls.data.get(scope)
            if group is None or key not in group:
                return None
            group[key] = ""
            return cls.name(scope)
        cls.data.setdefault(scope, {})[key] = text
        return cls.name(scope)

    @classmethod
    def values(cls, scope):
        return dict(cls.data.get(Scope(scope), {}))


def _lightmaps(*names):
    return {"objects": [{"name": n, "map": f"{n}_Lightmap.exr"} for n in names]}


def _shadows(*names):
    return {"unit_scale": 0.01, "planes": [{"name": n, "texture": f"{n}.png"} for n in names]}


def _rig(name, *joints):
    return {"name": name, "joints": [{"name": j} for j in joints], "grab": []}


class _Case(unittest.TestCase):
    def setUp(self):
        _Store.data = {}


class TestDeclarations(_Case):
    def test_the_node_keyed_records_say_where_their_nodes_are(self):
        """``RecordSpec.nodes`` is the one declaration a narrowing reads
        (FbxMedia.NODE_ENTRIES moved here, 2026-10-31). Added: 2026-10-31"""
        self.assertEqual(SR.LIGHTMAPS.nodes, ("objects", "name"))
        self.assertEqual(SR.SHADOWS.nodes, ("planes", "name"))
        self.assertEqual(SR.ARTICULATION.nodes, ("rigs", ("joints", "*", "name")))
        self.assertFalse(hasattr(ptk.FbxMedia, "NODE_ENTRIES"))

    def test_every_record_naming_scene_nodes_is_export_scoped(self):
        """Every export writes data for the objects it ships; the timeline's
        records ship whole. Added: 2026-10-31"""
        scoped = {s.key for s in SR.deliverable() if s.export_scoped}
        self.assertEqual(
            scoped,
            {
                SR.MATERIALS.key,
                SR.LIGHTMAPS.key,
                SR.SHADOWS.key,
                SR.ARTICULATION.key,
                SR.VISIBILITY.key,
            },
        )
        for spec in (SR.SHOTS, SR.AUDIO, SR.HANDOFF):
            self.assertFalse(spec.export_scoped, spec.key)

class TestShips(_Case):
    def test_a_name_matches_as_a_reader_binds_it(self):
        ctx = ExportContext(nodes=frozenset({"NS:cube", "cubeShape", "ball"}))
        for name in ("NS:cube", "cube", "|grp|NS:cube", "OTHER:cube", "|cubeShape", "ball"):
            self.assertTrue(ctx.ships(name), name)
        for name in ("floor", "", None, "cube1"):
            self.assertFalse(ctx.ships(name), name)

    def test_no_nodes_ships_everything(self):
        self.assertTrue(ExportContext().ships("anything"))
        self.assertTrue(ExportContext(scope=("|a",)).ships("b"), "scope alone names no node")

    def test_narrowing_keeps_entries_naming_a_shipped_node(self):
        names = ["KEEP_cube"]
        payload = _lightmaps("KEEP_cube", "DROP_cube")
        self.assertEqual(SR.LIGHTMAPS.narrowed(payload, names), _lightmaps("KEEP_cube"))
        self.assertIsNone(SR.LIGHTMAPS.narrowed(payload, ["floor"]), "none ships")
        same = _lightmaps("KEEP_cube")
        self.assertIs(SR.LIGHTMAPS.narrowed(same, names), same, "all ship: as it is")
        rigs = {"rigs": [_rig("door", "hinge", "leaf"), _rig("lid", "lid_jnt")]}
        self.assertEqual(
            [r["name"] for r in SR.ARTICULATION.narrowed(rigs, ["leaf"])["rigs"]],
            ["door"],
            "a rig ships when any joint does",
        )
        shots = {"shots": [{"clip": "a", "start": 1, "end": 9}]}
        self.assertIs(SR.SHOTS.narrowed(shots, names), shots, "no node entries")


class TestProducedCopy(_Case):
    def test_a_producer_that_ignores_the_scope_ships_narrowed(self):
        """The articulated rig's producer names every rig; a scoped write ships
        the rigs whose joints it ships, and the scene keeps them all.
        Added: 2026-10-31"""
        rigs = {"rigs": [_rig("door", "hinge"), _rig("lid", "lid_jnt")]}
        producers = {SR.ARTICULATION: lambda ctx: SR.ARTICULATION.make(rigs)}
        ctx = ExportContext(scope=("|door_grp",), nodes=frozenset({"door_grp", "hinge"}))
        snap = ExportSnapshot.assemble(producers, ctx)
        self.assertEqual([r["name"] for r in snap.record(SR.ARTICULATION)["rigs"]], ["door"])
        snap.commit(_Store)
        self.assertEqual(len(SR.ARTICULATION.load(_Store)["rigs"]), 1)
        snap.restore_scene_wide(_Store)
        self.assertEqual(len(SR.ARTICULATION.load(_Store)["rigs"]), 2, "the scene keeps it whole")

    def test_a_record_none_of_whose_nodes_ships_is_cleared_for_the_write(self):
        producers = {SR.SHADOWS: lambda ctx: SR.SHADOWS.make(_shadows("DROP_shadow"))}
        snap = ExportSnapshot.assemble(
            producers, ExportContext(scope=("|KEEP",), nodes=frozenset({"KEEP"}))
        )
        self.assertIsNone(snap.record(SR.SHADOWS))
        self.assertIn(SR.SHADOWS.key, snap.produced, "produced as None: cleared on commit")

    def test_a_timeline_record_ships_whole(self):
        shots = {"fps": 24.0, "shots": [{"clip": "a", "start": 1, "end": 9}]}
        snap = ExportSnapshot.assemble(
            {SR.SHOTS: lambda ctx: SR.SHOTS.make(shots)},
            ExportContext(scope=("|x",), nodes=frozenset({"x"})),
        )
        self.assertEqual(len(snap.record(SR.SHOTS)["shots"]), 1)
        self.assertEqual(snap.scene_wide, {})


class TestStoredCopy(_Case):
    """A hand-off refreshes only the derived records: the authored ones it
    ships as stored. Measured 2026-10-31 (fresh mayapy, two cubes, the Unity
    bridge sending one): the FBX named both cubes' lightmaps and the other
    cube's shadow plane, and embedded their maps."""

    def _stored(self):
        SR.LIGHTMAPS.save(_Store, _lightmaps("KEEP_cube", "DROP_cube"))
        SR.SHADOWS.save(_Store, _shadows("DROP_cube_shadow"))
        SR.SHOTS.save(_Store, {"fps": 24.0, "shots": [{"clip": "a", "start": 1, "end": 5}]})

    def _handoff(self, nodes=frozenset({"KEEP_cube", "KEEP_cubeShape"})):
        return ExportContext(mode=ExportContext.HANDOFF, scope=("|KEEP_cube",), nodes=nodes)

    def test_a_hand_off_ships_the_stored_copy_narrowed_and_puts_it_back(self):
        """Added: 2026-10-31"""
        self._stored()
        snap = ExportSnapshot.assemble({SR.LIGHTMAPS: lambda ctx: None}, self._handoff())
        self.assertNotIn(SR.LIGHTMAPS.key, snap.produced, "an authored record is not refreshed")
        snap.commit(_Store)
        self.assertEqual(
            [o["name"] for o in SR.LIGHTMAPS.load(_Store)["objects"]], ["KEEP_cube"]
        )
        self.assertIsNone(SR.SHADOWS.load(_Store), "no plane it ships: none")
        self.assertEqual(len(SR.SHOTS.load(_Store)["shots"]), 1, "the timeline ships whole")
        self.assertEqual(
            [o["name"] for o in snap.record(SR.LIGHTMAPS)["objects"]],
            ["KEEP_cube"],
            "the snapshot is what shipped",
        )
        snap.restore_scene_wide(_Store)
        self.assertEqual(len(SR.LIGHTMAPS.load(_Store)["objects"]), 2)
        self.assertEqual(SR.SHADOWS.load(_Store)["planes"][0]["name"], "DROP_cube_shadow")

    def test_what_ships_names_only_the_shipped_objects_maps(self):
        """The post-write embed reads the carrier as written. Added: 2026-10-31"""
        self._stored()
        snap = ExportSnapshot.assemble({}, self._handoff())
        snap.commit(_Store)
        channels = {
            spec.key: [spec.load(_Store)] for spec in (SR.LIGHTMAPS, SR.SHADOWS)
        }
        self.assertEqual(ptk.FbxMedia.named_files(channels), ["KEEP_cube_Lightmap.exr"])

    def test_a_write_that_names_no_node_ships_the_store_as_it_is(self):
        self._stored()
        for ctx in (
            ExportContext(mode=ExportContext.HANDOFF),
            ExportContext(mode=ExportContext.HANDOFF, scope=("|KEEP_cube",)),
        ):
            snap = ExportSnapshot.assemble({}, ctx)
            snap.commit(_Store)
            self.assertEqual(len(SR.LIGHTMAPS.load(_Store)["objects"]), 2)
            self.assertEqual(snap.scene_wide, {})

    def test_a_stored_copy_that_lists_no_entry_ships_as_stored(self):
        """Nothing in it names a node to narrow by (a payload of another
        shape): the write ships it as the scene stored it, as before.
        Added: 2026-10-31"""
        SR.LIGHTMAPS.save(_Store, {"version": 1})
        snap = ExportSnapshot.assemble({}, self._handoff())
        snap.commit(_Store)
        self.assertEqual(SR.LIGHTMAPS.load(_Store), {"version": 1})
        self.assertEqual(snap.scene_wide, {})

    def test_an_authoring_publish_never_narrows(self):
        self._stored()
        ExportSnapshot.publish(
            _Store,
            {SR.SHOTS: {"fps": 24.0, "shots": []}},
            ExportContext(mode=ExportContext.AUTHORING, nodes=frozenset({"x"})),
        )
        self.assertEqual(len(SR.LIGHTMAPS.load(_Store)["objects"]), 2)

    def test_a_failed_producers_record_stays_as_stored(self):
        self._stored()

        def boom(ctx):
            raise RuntimeError("no markers")

        ctx = ExportContext(scope=("|KEEP_cube",), nodes=frozenset({"KEEP_cube"}))
        snap = ExportSnapshot.assemble({SR.LIGHTMAPS: boom}, ctx)
        snap.commit(_Store)
        self.assertEqual(len(SR.LIGHTMAPS.load(_Store)["objects"]), 2)
        self.assertIn(SR.LIGHTMAPS.key, snap.failed)


class TestAStoreThatRefusesTheWrite(_Case):
    """A store that cannot write the scene's carrier without writing over
    another's (blendertk: a module's own-named ``data_export`` parented under
    its group holds the name) refuses it (``SceneStoreBase.WriteRefused``).
    The commit says it ONCE, with its fix and no traceback, notes it for the
    exporter, and leaves each record failed, as stored. Added: 2026-11-08"""

    FIX = "The name is a module's carrier: rename or unparent it."

    class _Refusing(_Store):
        @classmethod
        def write(cls, scope, key, text):
            if Scope(scope) is Scope.DELIVERABLE and text:
                raise SceneStoreBase.WriteRefused(TestAStoreThatRefusesTheWrite.FIX)
            return super().write(scope, key, text)

    def test_a_refusal_is_said_once_and_noted(self):
        producers = {
            SR.LIGHTMAPS: lambda ctx: SR.LIGHTMAPS.make(_lightmaps("wall")),
            SR.SHOTS: lambda ctx: SR.SHOTS.make({"fps": 24.0, "shots": []}),
        }
        snap = ExportSnapshot.assemble(producers, ExportContext())
        with self.assertLogs("pythontk", level="WARNING") as logs:
            snap.commit(self._Refusing)
        said = [line for line in logs.output if self.FIX in line]
        self.assertEqual(len(said), 1, logs.output)
        self.assertNotIn("Traceback", " ".join(logs.output))
        self.assertEqual(snap.failed, {SR.LIGHTMAPS.key, SR.SHOTS.key})
        noted = [n for notes in snap.noted.values() for n in notes if self.FIX in n]
        self.assertEqual(len(noted), 1, snap.noted)
        self.assertEqual(self._Refusing.data, {})


class TestARefusedRecordShipsNone(_Case):
    """No write carries a ``lightmaps`` record that reads a map through the
    wrong UVs (lightmaps v3, 2026-11-07). An export's producer notes its
    refusal and publishes none, so the delivered record is cleared -- a raise
    was a failed producer, whose stale stored record shipped. A hand-off,
    which ships the authored record as stored, ships none its host withheld
    (``ExportContext.withheld``) and puts it back after the write. Either
    way the materials read what ships: no lightmaps, so each keeps its
    occlusion. Added: 2026-11-07"""

    WHY = "Lightmaps not published: wall ('UV2' at index 2). Move it."

    def _readers(self, seen):
        """A materials producer that records the lightmaps it would read."""

        def materials(ctx):
            seen.append(ctx.record(SR.LIGHTMAPS, store=_Store))
            return None

        return {SR.MATERIALS: materials}

    def test_the_materials_read_the_lightmaps_this_write_ships(self):
        self.assertEqual(SR.MATERIALS.after, (SR.LIGHTMAPS.key,))
        seen = []
        producers = dict(self._readers(seen))
        producers[SR.LIGHTMAPS] = lambda ctx: SR.LIGHTMAPS.make(_lightmaps("wall"))
        ExportSnapshot.assemble(producers, ExportContext())
        self.assertEqual([o["name"] for o in seen[0]["objects"]], ["wall"])

    def test_an_exports_refusal_is_noted_and_clears_the_record(self):
        SR.LIGHTMAPS.save(_Store, _lightmaps("wall"))  # published before the edit

        def refused():
            ptk.LightmapRecord.check_uv_sets([("wall", "UV2", 2)], fix="Move it.")

        seen = []
        producers = dict(self._readers(seen))
        producers[SR.LIGHTMAPS] = lambda ctx: ptk.LightmapRecord.unless_refused(
            refused, None, note=ctx.note
        )
        snap = ExportSnapshot.assemble(producers, ExportContext())
        self.assertNotIn(SR.LIGHTMAPS.key, snap.failed, "a refusal is no failure")
        self.assertIsNone(snap.produced[SR.LIGHTMAPS.key])
        self.assertEqual(seen, [None], "the materials read no lightmaps")
        self.assertTrue(any("wall ('UV2' at index 2)" in n for n in snap.ctx.notes))
        snap.commit(_Store)
        self.assertIsNone(SR.LIGHTMAPS.load(_Store), "the stale record is cleared")

    def _handoff(self, **kwargs):
        return ExportContext(
            mode=ExportContext.HANDOFF, withheld={SR.LIGHTMAPS.key: self.WHY}, **kwargs
        )

    def test_a_hand_off_ships_none_of_a_withheld_record_and_puts_it_back(self):
        for kwargs in (
            {},
            {"scope": ("|wall",), "nodes": frozenset({"wall", "wallShape"})},
        ):
            with self.subTest(**kwargs):
                _Store.data = {}
                SR.LIGHTMAPS.save(_Store, _lightmaps("wall"))
                SR.SHOTS.save(_Store, {"fps": 24.0, "shots": []})
                stored = SR.LIGHTMAPS.read_text(_Store)
                seen = []
                snap = ExportSnapshot.assemble(
                    self._readers(seen), self._handoff(**kwargs)
                )
                self.assertEqual(seen, [None], "the materials read no lightmaps")
                self.assertEqual(snap.ctx.notes.count(self.WHY), 1, snap.ctx.notes)
                self.assertEqual(snap.noted[SR.LIGHTMAPS.key], [self.WHY])
                snap.commit(_Store)
                self.assertIsNone(SR.LIGHTMAPS.load(_Store), "the write ships none")
                self.assertIsNone(snap.record(SR.LIGHTMAPS))
                handoff = SR.HANDOFF.load(_Store)
                self.assertNotIn(
                    SR.LIGHTMAPS.key, [row.get("part") for row in handoff["assembly"]]
                )
                snap.restore_scene_wide(_Store)
                self.assertEqual(
                    SR.LIGHTMAPS.read_text(_Store), stored, "the scene keeps it"
                )

    def test_a_record_the_assembly_produces_is_its_producers_call(self):
        produced = SR.LIGHTMAPS.make(_lightmaps("wall"))
        ctx = ExportContext(withheld={SR.LIGHTMAPS.key: self.WHY})
        snap = ExportSnapshot.assemble({SR.LIGHTMAPS: lambda ctx: produced}, ctx)
        self.assertEqual(snap.record(SR.LIGHTMAPS), produced.payload)
        self.assertNotIn(self.WHY, snap.ctx.notes)


class TestWriterUnits(_Case):
    """The writer states its unit, once; every producer labels its record with
    it and converts nothing. Measured 2026-10-17: Blender's native glTF writer
    ignores scale_length and an FBX written with apply_unit_scale off ships
    one BU as one metre, while every record was labelled with the scene's."""

    def test_each_writer_states_the_metres_one_scene_unit_is_in_its_file(self):
        """Added: 2026-10-31"""
        unit = ExportContext.writer_unit
        self.assertEqual(unit("maya", "fbx", 0.01), 0.01)
        self.assertEqual(unit("maya", "usd", 0.01), 0.01)
        self.assertEqual(unit("blender", "fbx", 0.01), 0.01, "apply_unit_scale on")
        self.assertEqual(unit("blender", "fbx", 0.01, {"apply_unit_scale": False}), 1.0)
        self.assertAlmostEqual(unit("blender", "fbx", 0.01, {"global_scale": 2.0}), 0.02)
        self.assertEqual(unit("blender", "gltf", 0.01), 1.0)
        self.assertEqual(unit("blender", "usd", 0.01, {"convert_scene_units": "CENTIMETERS"}), 1.0)
        with self.assertRaises(KeyError):
            unit("maya", "obj", 0.01)

    def test_a_record_takes_the_writers_label_else_the_scenes(self):
        """Added: 2026-10-31"""
        read = []

        def scene():
            read.append(True)
            return 0.01

        self.assertEqual(ExportContext(unit_scale=1.0).unit(scene), 1.0)
        self.assertEqual(read, [], "the scene's unit is read only without a writer's")
        self.assertEqual(ExportContext().unit(scene), 0.01)


if __name__ == "__main__":
    unittest.main()
