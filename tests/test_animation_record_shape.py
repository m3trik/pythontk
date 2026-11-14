# !/usr/bin/python
# coding=utf-8
"""The ``animation`` record's declared shape against what its producers write.

:class:`AnimationRecord` is held to ``ShotStore.export_records`` -- the one
producer: mayatk's and blendertk's stores publish through it and override
scene lookups only -- and to what ``MeshConvert.apply_glb_animations`` carries
into a GLB, which is the record itself, run on a real store's export and on
the converter shapes ``test_mesh_convert`` builds (one take per shot, each one
clip). Exact both ways: every payload validates with no unknown-key warning,
and every declared key is one a producer writes. Added: 2026-10-20; record v3
(one take per shot): 2026-11-05; the GLB carries the record itself (its block
retired): 2026-11-05.
"""

import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pythontk as ptk
from pythontk import MeshConvert
from pythontk.core_utils.engines.scene_export.export_snapshot import ExportContext
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.core_utils.engines.shots.animation_record import (
    AnimationRecord,
    AnimationShot,
)
from pythontk.core_utils.engines.shots.shot_model import PureShotsHost, ShotStore

ptk.TestSandbox.activate()

# The measured converter shapes, and the GLB writer they are built with.
import test_mesh_convert as measured

#: The rate the measured shapes are built at, and the stores here too, so a
#: store's takes and the fixture's clips count alike.
FPS = measured.TestApplyGlbAnimations.FPS


# ============================================================================
# Fixtures
# ============================================================================


def _declared(spec):
    """The keys *spec* declares.

    Parameters:
        spec: A :class:`~pythontk.SchemaSpec` subclass.

    Returns:
        The set of its field names.
    """
    return {doc.name for doc in spec.describe()}


def _required(spec):
    """The keys *spec* requires, in declaration order.

    Parameters:
        spec: A :class:`~pythontk.SchemaSpec` subclass.

    Returns:
        The list of its required field names.
    """
    return [doc.name for doc in spec.describe() if doc.required]


def _assert_exact(case, spec, payload):
    """Assert *payload* is *spec* as declared: no error, and no key it does
    not name, at any depth.

    Parameters:
        case: The running test.
        spec: The declared shape.
        payload: What a producer wrote.
    """
    result = spec.validate(payload)
    case.assertEqual(result.errors, [], json.dumps(payload))
    case.assertEqual(result.warnings, [], "a key the declaration does not name")


def _edited(payload, path, value=None, drop=False):
    """A deep copy of *payload* with the value at *path* replaced, or dropped.

    Parameters:
        payload: The valid payload to start from.
        path: The keys and indices down to the value.
        value: Its replacement.
        drop: Remove the key instead.

    Returns:
        The edited copy.
    """
    out = copy.deepcopy(payload)
    parent = out
    for step in path[:-1]:
        parent = parent[step]
    if drop:
        del parent[path[-1]]
    else:
        parent[path[-1]] = value
    return out


def _store(host=None, fps=FPS):
    """A store holding each kind of shot the record carries, made active: a
    built step's shot (members, a description, metadata -- a tuple among it),
    a tail split off it (no members, nothing a clip cannot hold) and a hold.

    Parameters:
        host: The store's host -- the pure one (``None``), or a DCC's
            stand-in.
        fps: The scene's frame rate.

    Returns:
        The store.
    """
    store = ShotStore([], host=host)
    store.scene_fps = fps
    # The keys its author (a manifest's Build) ships; the rest is bookkeeping.
    store.set_shipped_metadata(["step", "section", "behaviors", "odd"])
    store.define_shot(
        "A01_1",
        1,
        40,
        objects=["|set|door_geo", "|set|NS:handle_geo"],
        description="The door swings open.",
        metadata={
            "step": "A01",
            "section": "A",
            "behaviors": ("open", "hold"),
            "object_status": {"door_geo": "valid"},
        },
    )
    store.define_shot("A01_2", 41, 80)
    store.define_shot("Hold", 81, 100, objects=["|set|lamp_geo"])
    ShotStore.set_active(store)
    return store


class _HostScene(PureShotsHost):
    """A DCC host's scene lookups -- what mayatk's and blendertk's hosts
    answer -- from a fixed scene: one member renamed behind the store's back
    (Maya finds it by its UUID, Blender by the rename it follows, so it is
    still held), one deleted, and nothing keyed in the frames of the shot
    whose every member is gone, so that shot is stale."""

    GONE = {"|set|lamp_geo"}

    def existing_objects(self, store, names):
        return {n for n in names if n not in self.GONE}

    def keyed_windows(self, store, windows):
        return [False] * len(windows)


class _ShotsCase(unittest.TestCase):
    """No store active, and no persistence, before or after each test."""

    def setUp(self):
        ShotStore.clear_active()
        self.addCleanup(ShotStore.clear_active)


# ============================================================================
# The record
# ============================================================================


class TestTheRecordIsWhatTheStorePublishes(_ShotsCase):
    """``ShotStore.export_records`` -- the record's one producer."""

    def _produced(self, store, **decisions):
        (record,) = store.export_records(ExportContext(**decisions))
        self.assertIs(record.spec, SceneRecords.SHOTS)
        return record.payload

    def test_the_store_writes_the_declared_shape_and_nothing_else(self):
        """Per shot its clip, and its description and metadata only where it
        has them: the frames are the take's, the members the scene's -- and
        the scene's rate, which a GLB cannot hold."""
        payload = self._produced(_store())
        _assert_exact(self, AnimationRecord, payload)
        self.assertEqual(set(payload), {"version", "shots", "fps"})
        self.assertEqual(payload["fps"], FPS)
        self.assertEqual(payload["version"], SceneRecords.SHOTS.version)
        self.assertEqual(
            [set(shot) for shot in payload["shots"]],
            [{"clip", "description", "metadata"}, {"clip"}, {"clip"}],
        )

    def test_the_metadata_ships_as_json(self):
        """The scene's dict, JSON whole: a tuple is a list, as JSON spells it,
        and a value JSON cannot hold ships as its string form, warned."""
        store = _store()
        payload = self._produced(store)
        self.assertEqual(
            payload["shots"][0]["metadata"],
            {"step": "A01", "section": "A", "behaviors": ["open", "hold"]},
        )
        store.update_shot(store.sorted_shots()[1].shot_id, metadata={"odd": {1, 2}})
        with self.assertLogs("pythontk", level="WARNING") as caught:
            odd = self._produced(store)
        self.assertIsInstance(odd["shots"][1]["metadata"]["odd"], str)
        self.assertTrue(any("A01_2" in m for m in caught.output), caught.output)
        _assert_exact(self, AnimationRecord, odd)

    def test_the_producer_both_dccs_name_writes_it(self):
        """``produce_export_records`` -- what both DCCs' ``FbxUtils.PRODUCERS``
        name -- with no context and with an authoring one."""
        _store()
        for ctx in (None, ExportContext(mode=ExportContext.AUTHORING)):
            with self.subTest(ctx=ctx and ctx.mode):
                (record,) = ShotStore.produce_export_records(ctx)
                _assert_exact(self, AnimationRecord, record.payload)

    def test_a_dcc_stores_lookups_change_its_values_never_its_shape(self):
        """A renamed member, a deleted one, a stale shot, an NTSC rate and an
        export set narrower than the scene: the payload a DCC store
        publishes, through the same producer."""
        store = _store(_HostScene(), fps=30000 / 1001)
        payload = self._produced(store, scope=("|set|door_geo",))
        _assert_exact(self, AnimationRecord, payload)
        self.assertAlmostEqual(payload["fps"], 30000 / 1001, places=5)
        # The stale "Hold" ships unless the store leaves stale shots out.
        self.assertEqual(
            [s["clip"] for s in payload["shots"]], ["A01_1", "A01_2", "Hold"]
        )
        store.drop_stale_shots = True
        dropped = self._produced(store, scope=("|set|door_geo",))
        _assert_exact(self, AnimationRecord, dropped)
        self.assertEqual([s["clip"] for s in dropped["shots"]], ["A01_1", "A01_2"])
        # A shot ships no members (2026-10-20): the export set narrows nothing.
        self.assertNotIn("objects", payload["shots"][0])

    def test_the_takes_are_the_stores_and_name_the_records_clips(self):
        """The ranges the record no longer states: ``ShotStore.declared_takes``,
        one per shot the record names, in its order."""
        store = _store()
        payload = self._produced(store)
        self.assertEqual(
            ShotStore.declared_takes(),
            [
                {"name": "A01_1", "start": 1, "end": 40},
                {"name": "A01_2", "start": 41, "end": 80},
                {"name": "Hold", "start": 81, "end": 100},
            ],
        )
        self.assertEqual(
            [t["name"] for t in ShotStore.declared_takes()],
            [s["clip"] for s in payload["shots"]],
        )

    def test_every_declared_key_is_one_the_store_writes(self):
        payload = self._produced(_store())
        self.assertEqual(set(payload), _declared(AnimationRecord))
        self.assertEqual(set().union(*payload["shots"]), _declared(AnimationShot))


# ============================================================================
# The record in a GLB
# ============================================================================


class TestTheGlbCarriesTheRecordItself(_ShotsCase):
    """``MeshConvert.apply_glb_animations`` carries the FBX's ``animation``
    record into the GLB as it stands: one layout, two containers."""

    def setUp(self):
        super().setUp()
        self.temp = ptk.TempArtifacts.scoped("animation_record_shape")
        self.addCleanup(self.temp.cleanup)
        # test_mesh_convert's own builders, writing into this test's scratch.
        self.shapes = measured.TestApplyGlbAnimations()
        self.shapes.tmp = self.temp.dir_path()

    def _block(self, path):
        """The record the pass writes into *path*'s root ``extras``, checked
        to be the one it returns and the carrier's own, verbatim."""
        with MeshConvert.open_glb(path) as edit:
            carried = MeshConvert.data_export_channel(edit.gltf, SceneRecords.SHOTS.key)
        returned = MeshConvert.apply_glb_animations(path)
        with MeshConvert.open_glb(path) as edit:
            written = SceneRecords.glb_block(edit.gltf, SceneRecords.SHOTS)
        self.assertEqual(written, returned)
        self.assertEqual(written, carried, "the record, as the FBX carries it")
        return returned

    def _from_store(self):
        """The GLB of a real store's export: its record on the carrier and,
        as FBX2glTF writes them, one clip per take -- one per shot, each
        rebased to t=0."""
        _store()
        (record,) = ShotStore.produce_export_records(ExportContext())
        takes = [(t["name"], t["start"], t["end"]) for t in ShotStore.declared_takes()]
        return self.shapes._glb(
            takes,
            channels={SceneRecords.SHOTS.key: record.payload},
            name="store.glb",
        )

    def _blocks(self):
        """``{fixture: block}`` for every shape the pass meets."""
        return {
            "a real store's export": self._block(self._from_store()),
            "one take per shot, with notes": self._block(self.shapes._shot_shape()),
            "a declared take with no keys": self._block(self.shapes._with_empty_shot()),
        }

    def test_every_shape_the_pass_meets_carries_the_declared_record(self):
        for fixture, block in self._blocks().items():
            with self.subTest(fixture):
                _assert_exact(self, AnimationRecord, block)

    def test_a_file_with_no_record_carries_none(self):
        """A scene with no shots ships one whole-timeline take and no record:
        the GLB carries none either -- nothing is made up for it."""
        path = self.shapes._glb([("Take 001", 1, 30)], name="unsplit.glb")
        self.assertIsNone(MeshConvert.apply_glb_animations(path))
        with MeshConvert.open_glb(path) as edit:
            self.assertIsNone(SceneRecords.glb_block(edit.gltf, SceneRecords.SHOTS))


# ============================================================================
# What no producer writes
# ============================================================================


class TestWhatNoProducerWritesIsRefused(unittest.TestCase):
    """Each refusal named where it is, so a reader's validator can say so."""

    RECORD = {
        "version": 3,
        "fps": 30.0,
        "shots": [
            {
                "clip": "Intro",
                "description": "The door swings open.",
                "metadata": {"section": "A"},
            }
        ],
    }
    def _assert_refused(self, spec, valid, cases):
        """Assert *valid* passes and each case fails with its message.

        Parameters:
            spec: The declared shape.
            valid: A payload it accepts.
            cases: ``{message fragment: broken payload}``.
        """
        self.assertEqual(spec.validate(valid).errors, [])
        for expected, payload in cases.items():
            with self.subTest(expected):
                errors = spec.validate(payload).errors
                self.assertTrue(any(expected in e for e in errors), errors)

    def test_the_record_refuses_what_no_producer_writes(self):
        r = self.RECORD
        self._assert_refused(
            AnimationRecord,
            r,
            {
                "missing required key 'shots'": _edited(r, ["shots"], drop=True),
                "shots[0].missing required key 'clip'": _edited(
                    r, ["shots", 0, "clip"], drop=True
                ),
                "shots[0].description: expected string, got null": _edited(
                    r, ["shots", 0, "description"], None
                ),
                "shots[0].metadata: expected object, got array": _edited(
                    r, ["shots", 0, "metadata"], ["section", "A"]
                ),
                "shots: expected a list": _edited(r, ["shots"], {}),
                "version: expected integer, got string": _edited(r, ["version"], "2"),
                "fps: expected number, got string": _edited(r, ["fps"], "30"),
            },
        )

    def test_a_version_1_record_is_no_current_record(self):
        """A file written before one take per shot -- each shot a window of a
        whole-timeline take -- is not read as v2: its readers refuse it and
        name the cure, re-export (``RecordSpec.is_current``,
        ``SceneRecords.problems``)."""
        old = {
            "version": 1,
            "fps": 30.0,
            "clip_mode": "both",
            "shots": [{"clip": "Intro", "start": 1, "end": 40}],
        }
        self.assertFalse(SceneRecords.SHOTS.is_current(old))
        self.assertTrue(SceneRecords.SHOTS.is_current(self.RECORD))
        (problem,) = SceneRecords.problems(SceneRecords.SHOTS, old)
        self.assertIn("v1 is not the v3", problem)

    def test_the_retired_keys_are_none_the_shapes_declare(self):
        """Record v2 retires every frame field: the takes are the timing, and
        no reader needs a frame number. A payload still carrying one is an
        unknown key, warned. (``fps`` is no frame number: the rate glTF
        cannot hold.)"""
        for spec, payload, path in (
            (AnimationRecord, self.RECORD, ["clip_mode"]),
            (AnimationRecord, self.RECORD, ["default_clip"]),
            (AnimationRecord, self.RECORD, ["shots", 0, "start"]),
            (AnimationRecord, self.RECORD, ["shots", 0, "end"]),
            (AnimationRecord, self.RECORD, ["shots", 0, "start_frame"]),
        ):
            with self.subTest(spec=spec.__name__, key=".".join(map(str, path))):
                result = spec.validate(_edited(payload, path, 1))
                self.assertEqual(result.errors, [])
                self.assertTrue(result.warnings, "a retired key read as declared")


# ============================================================================
# The published forms
# ============================================================================


class TestThePublishedForms(unittest.TestCase):
    """The JSON Schema the other languages' types are generated from, and the
    reference and skeleton an external reader is handed."""

    #: Each root and the types it nests.
    SHAPES = ((AnimationRecord, (AnimationShot,)),)

    def test_each_json_schema_builds_with_its_nested_types(self):
        """``AnimationShot``: the C# generator strips the root's family, so it
        reads ``AnimationRecord.Shot`` -- no UnityEngine or System type's
        name."""
        for spec, nested in self.SHAPES:
            with self.subTest(spec.__name__):
                doc = spec.json_schema()
                json.dumps(doc)
                self.assertEqual(doc["$schema"], spec.JSON_SCHEMA_DIALECT)
                self.assertEqual(doc["title"], spec.__name__)
                self.assertEqual(sorted(doc["$defs"]), [n.__name__ for n in nested])

    def test_the_required_keys_are_the_ones_every_producer_always_writes(self):
        self.assertEqual(_required(AnimationRecord), ["version", "shots", "fps"])
        self.assertEqual(_required(AnimationShot), ["clip"])

    def test_the_metadata_is_a_json_object(self):
        metadata = AnimationShot.json_schema()["properties"]["metadata"]
        self.assertEqual(metadata["type"], "object")

    def test_the_skeleton_is_a_valid_payload_and_the_reference_names_every_key(self):
        for spec, nested in self.SHAPES:
            with self.subTest(spec.__name__):
                _assert_exact(self, spec, spec.skeleton())
                reference = spec.to_markdown()
                for key in _declared(spec).union(*map(_declared, nested)):
                    self.assertIn(f"`{key}`", reference)


if __name__ == "__main__":
    unittest.main()
