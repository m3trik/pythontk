# !/usr/bin/python
# coding=utf-8
"""The ``animation`` record's and its GLB block's declared shapes against what
their producers write.

:class:`AnimationRecord` is held to ``ShotStore.export_records`` -- the one
producer: mayatk's and blendertk's stores publish through it and override
scene lookups only -- and :class:`AnimationBlock` to
``MeshConvert.apply_glb_animations``, run on a real store's export and on the
converter shapes ``test_mesh_convert`` measured (Maya 2025 -> FBX2glTF
0.13.1). Exact both ways: every payload validates with no unknown-key
warning, and every declared key is one a producer writes. Added: 2026-10-20
"""

import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pythontk as ptk
from pythontk import MeshConvert
from pythontk.core_utils.engines.scene_export.export_profile import ExportProfile
from pythontk.core_utils.engines.scene_export.export_snapshot import ExportContext
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.core_utils.engines.shots.animation_record import (
    AnimationBlock,
    AnimationBlockClip,
    AnimationRecord,
    AnimationShot,
)
from pythontk.core_utils.engines.shots.shot_model import PureShotsHost, ShotStore

ptk.TestSandbox.activate()

# The measured converter shapes, and the GLB writer they are built with.
import test_mesh_convert as measured

#: The rate the measured shapes are built at, and the stores here too, so a
#: store's takes and the fixture's clip spans count alike.
FPS = measured.TestApplyGlbAnimations.FPS
#: The exporter's Animation Clips modes, from the table that declares them.
CLIP_MODES = tuple(ExportProfile.ANIMATION_CLIPS_OPTIONS.values())


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
    built step's shot (members, a description, a section), a tail split off
    it (no members; its step's section, by its name) and a hold.

    Parameters:
        host: The store's host -- the pure one (``None``), or a DCC's
            stand-in.
        fps: The scene's frame rate.

    Returns:
        The store.
    """
    store = ShotStore([], host=host)
    store.scene_fps = fps
    store.define_shot(
        "A01_1",
        1,
        40,
        objects=["|set|door_geo", "|set|NS:handle_geo"],
        description="The door swings open.",
        metadata={"step": "A01", "section": "A"},
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

    def test_every_clip_mode_and_an_undeclared_one_write_the_declared_shape(self):
        store = _store()
        for mode in (None,) + CLIP_MODES:
            with self.subTest(clip_mode=mode):
                payload = self._produced(store, clip_mode=mode)
                _assert_exact(self, AnimationRecord, payload)
                self.assertEqual(payload.get("clip_mode"), mode)

    def test_the_producer_both_dccs_name_writes_it_with_no_mode_unless_given(self):
        """``produce_export_records`` -- what both DCCs' ``FbxUtils.PRODUCERS``
        name -- with no context and with an authoring one: no exporter
        decided a mode, so none is declared."""
        _store()
        for ctx in (None, ExportContext(mode=ExportContext.AUTHORING)):
            with self.subTest(ctx=ctx and ctx.mode):
                (record,) = ShotStore.produce_export_records(ctx)
                _assert_exact(self, AnimationRecord, record.payload)
                self.assertNotIn("clip_mode", record.payload)

    def test_a_dcc_stores_lookups_change_its_values_never_its_shape(self):
        """A renamed member, a deleted one, a stale shot, an NTSC rate and an
        export set narrower than the scene: the payload a DCC store
        publishes, through the same producer."""
        store = _store(_HostScene(), fps=30000 / 1001)
        payload = self._produced(store, clip_mode="shots", scope=("|set|door_geo",))
        _assert_exact(self, AnimationRecord, payload)
        # The stale "Hold" ships unless the store leaves stale shots out.
        self.assertEqual(
            [s["clip"] for s in payload["shots"]], ["A01_1", "A01_2", "Hold"]
        )
        store.drop_stale_shots = True
        dropped = self._produced(store, clip_mode="shots", scope=("|set|door_geo",))
        _assert_exact(self, AnimationRecord, dropped)
        self.assertEqual([s["clip"] for s in dropped["shots"]], ["A01_1", "A01_2"])
        # A shot ships no members (2026-10-20): the export set narrows nothing.
        self.assertNotIn("objects", payload["shots"][0])

    def test_every_declared_key_is_one_the_store_writes(self):
        store = _store()
        payloads = [self._produced(store, clip_mode=m) for m in (None,) + CLIP_MODES]
        self.assertEqual(set().union(*payloads), _declared(AnimationRecord))
        shots = [shot for payload in payloads for shot in payload["shots"]]
        self.assertEqual(set().union(*shots), _declared(AnimationShot))


# ============================================================================
# The GLB block
# ============================================================================


class TestTheBlockIsWhatTheConversionWrites(_ShotsCase):
    """``MeshConvert.apply_glb_animations`` -- the block's one producer."""

    def setUp(self):
        super().setUp()
        self.temp = ptk.TempArtifacts.scoped("animation_record_shape")
        self.addCleanup(self.temp.cleanup)
        # test_mesh_convert's own builders, writing into this test's scratch.
        self.shapes = measured.TestApplyGlbAnimations()
        self.shapes.tmp = self.temp.dir_path()

    def _block(self, path):
        """The block the pass writes into *path*, checked to be the one it
        returns."""
        returned = MeshConvert.apply_glb_animations(path)
        with MeshConvert.open_glb(path) as edit:
            written = SceneRecords.glb_block(edit.gltf, SceneRecords.SHOTS)
        self.assertEqual(written, returned)
        return returned

    def _from_store(self):
        """The GLB of a real store's export: its record on the carrier and,
        as FBX2glTF writes them, the whole-timeline take first, then one clip
        per shot, each rebased to t=0."""
        _store()
        (record,) = ShotStore.produce_export_records(ExportContext(clip_mode="both"))
        takes = [(s["clip"], s["start"], s["end"]) for s in record.payload["shots"]]
        return self.shapes._glb(
            [("Take 001", 1, 100)] + takes,
            channels={SceneRecords.SHOTS.key: record.payload},
            name="store.glb",
        )

    def _blocks(self):
        """``{fixture: block}`` for every shape the pass meets."""
        return {
            "a real store's export": self._block(self._from_store()),
            "Maya's split, whole-timeline take first": self._block(
                self.shapes._maya_shape()
            ),
            "a take with a lead-in, spans published": self._block(
                self.shapes._lead_in_shape()
            ),
            "a declared take with no keys": self._block(self.shapes._with_empty_shot()),
            "no shots, no rate": self._block(
                self.shapes._glb([("Take 001", 1, 30)], name="unsplit.glb")
            ),
        }

    def test_every_shape_the_pass_meets_writes_the_declared_block(self):
        for fixture, block in self._blocks().items():
            with self.subTest(fixture):
                _assert_exact(self, AnimationBlock, block)

    def test_every_declared_key_is_one_the_pass_writes(self):
        blocks = list(self._blocks().values())
        self.assertEqual(set().union(*blocks), _declared(AnimationBlock))
        clips = [clip for block in blocks for clip in block["clips"]]
        self.assertEqual(set().union(*clips), _declared(AnimationBlockClip))


# ============================================================================
# What no producer writes
# ============================================================================


class TestWhatNoProducerWritesIsRefused(unittest.TestCase):
    """Each refusal named where it is, so a reader's validator can say so."""

    RECORD = {
        "version": 1,
        "fps": 30.0,
        "shots": [
            {
                "clip": "Intro",
                "start": 1,
                "end": 40,
                "description": "The door swings open.",
            }
        ],
        "clip_mode": "shots",
    }
    BLOCK = {
        "version": 2,
        "shots": [
            {
                "clip": "Intro",
                "start": 1,
                "end": 40,
                "description": "The door swings open.",
            }
        ],
        "clips": [{"name": "Intro", "animation": 0, "zero_frame": 1.0}],
        "default_clip": "Intro",
        "fps": 30.0,
        "clip_mode": "shots",
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
                "missing required key 'fps'": _edited(r, ["fps"], drop=True),
                "shots[0].missing required key 'clip'": _edited(
                    r, ["shots", 0, "clip"], drop=True
                ),
                "shots[0].start: expected integer, got number": _edited(
                    r, ["shots", 0, "start"], 1.5
                ),
                "shots[0].end: expected integer, got string": _edited(
                    r, ["shots", 0, "end"], "40"
                ),
                "shots[0].description: expected string, got null": _edited(
                    r, ["shots", 0, "description"], None
                ),
                "clip_mode: 'all' is not one of": _edited(r, ["clip_mode"], "all"),
                "shots: expected a list": _edited(r, ["shots"], {}),
                "fps: expected number, got string": _edited(r, ["fps"], "30"),
                "version: expected integer, got string": _edited(r, ["version"], "1"),
            },
        )

    def test_the_block_refuses_what_no_producer_writes(self):
        b = self.BLOCK
        self._assert_refused(
            AnimationBlock,
            b,
            {
                "missing required key 'shots'": _edited(b, ["shots"], drop=True),
                "shots[0].missing required key 'clip'": _edited(
                    b, ["shots", 0, "clip"], drop=True
                ),
                "shots[0].start: expected integer, got number": _edited(
                    b, ["shots", 0, "start"], 1.5
                ),
                "clips[0].missing required key 'animation'": _edited(
                    b, ["clips", 0, "animation"], drop=True
                ),
                "clips[0].animation: expected integer, got string": _edited(
                    b, ["clips", 0, "animation"], "0"
                ),
                "clips[0].zero_frame: expected number, got string": _edited(
                    b, ["clips", 0, "zero_frame"], "1"
                ),
                "clips[0].expected a mapping": _edited(b, ["clips"], ["Intro"]),
                "clip_mode: 'all' is not one of": _edited(b, ["clip_mode"], "all"),
            },
        )


# ============================================================================
# The published forms
# ============================================================================


class TestThePublishedForms(unittest.TestCase):
    """The JSON Schema the other languages' types are generated from, and the
    reference and skeleton an external reader is handed."""

    #: Each root and the types it nests: the block inherits the record's
    #: shot as it is, beside its own clip.
    SHAPES = (
        (AnimationRecord, (AnimationShot,)),
        (AnimationBlock, (AnimationBlockClip, AnimationShot)),
    )

    def test_each_json_schema_builds_with_its_nested_types(self):
        """``AnimationShot`` and ``AnimationBlockClip``: the C# generator
        strips the root's family, so they read ``AnimationRecord.Shot`` and
        ``AnimationBlock.Clip`` -- no UnityEngine or System type's name."""
        for spec, nested in self.SHAPES:
            with self.subTest(spec.__name__):
                doc = spec.json_schema()
                json.dumps(doc)
                self.assertEqual(doc["$schema"], spec.JSON_SCHEMA_DIALECT)
                self.assertEqual(doc["title"], spec.__name__)
                self.assertEqual(sorted(doc["$defs"]), [n.__name__ for n in nested])

    def test_the_required_keys_are_the_ones_every_producer_always_writes(self):
        self.assertEqual(_required(AnimationRecord), ["version", "fps", "shots"])
        self.assertEqual(_required(AnimationShot), ["clip", "start", "end"])
        self.assertEqual(_required(AnimationBlock), ["version", "shots", "clips"])
        self.assertEqual(_required(AnimationBlockClip), ["name", "animation"])

    def test_the_clip_mode_is_the_exporters_own_enumeration(self):
        clip_mode = AnimationRecord.json_schema()["properties"]["clip_mode"]
        self.assertEqual(clip_mode["type"], "string")
        self.assertEqual(clip_mode["enum"], list(CLIP_MODES))

    def test_the_skeleton_is_a_valid_payload_and_the_reference_names_every_key(self):
        for spec, nested in self.SHAPES:
            with self.subTest(spec.__name__):
                _assert_exact(self, spec, spec.skeleton())
                reference = spec.to_markdown()
                for key in _declared(spec).union(*map(_declared, nested)):
                    self.assertIn(f"`{key}`", reference)


if __name__ == "__main__":
    unittest.main()
