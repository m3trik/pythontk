# !/usr/bin/python
# coding=utf-8
"""The ``audio`` record's declared shape (:class:`AudioRecord`) against what
its producers write.

mayatk's ``AudioClips.export_record`` and blendertk's
``AudioUtils.export_record`` build the record alike: their baked
``(frame, track)`` events scoped by ``ptk.AudioUtils.scope_events`` to the
takes the ``animation`` record declares (``SceneRecords.declared_takes``;
none in a ``full`` export), made with ``SceneRecords.AUDIO.make({"events",
"fps"})``. That composition, less the scene reads, runs here in a real
``ExportSnapshot`` assembly after the real shots producer, once with each
host's origin for an unscoped event. Exact both ways: every payload validates
with no unknown-key warning, and every declared key is one the producers
write. Added: 2026-10-20
"""

import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pythontk as ptk
from pythontk import AudioUtils
from pythontk.audio_utils.audio_record import AudioCue, AudioRecord
from pythontk.core_utils.engines.scene_export.export_snapshot import (
    ExportContext,
    ExportSnapshot,
)
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.core_utils.engines.shots.shot_model import ShotStore

ptk.TestSandbox.activate()

FPS = 30.0
#: The scene's playback range and the exported stack's first frame differ:
#: the two hosts count an unscoped event from different ones.
PLAYBACK_MIN = 1
#: The scene's baked events, by frame: one in each shot, one on the frame the
#: two shots share, one past every shot, and a track name with a space.
EVENTS = [
    (5, "footstep"),
    (40, "door_slam"),
    (52, "Lamp Hum"),
    (130, "late"),
]
#: Each host's origin for an unscoped event, given the export's context:
#: mayatk counts from the playback range's first frame, blendertk from the
#: exported stack's (``ctx.clip_span``, else the scene's start).
ORIGINS = {
    "mayatk": lambda ctx: PLAYBACK_MIN,
    "blendertk": lambda ctx: ctx.clip_span[0] if ctx.clip_span else PLAYBACK_MIN,
}


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


def _assert_exact(case, payload):
    """Assert *payload* is :class:`AudioRecord` as declared: no error, and no
    key it does not name, at any depth.

    Parameters:
        case: The running test.
        payload: What a producer wrote.
    """
    result = AudioRecord.validate(payload)
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


def _audio_producer(origin):
    """A DCC's ``audio`` producer over :data:`EVENTS`: the composition both
    hosts' ``export_record`` share, less their scene reads.

    Parameters:
        origin: The host's origin for an unscoped event, given the context
            (:data:`ORIGINS`).

    Returns:
        The producer, taking the export's context.
    """

    def produce(ctx):
        declared = SceneRecords.declared_takes(lambda key: ctx.record(key))
        takes = [] if ctx.clip_mode == "full" else declared
        scoped, _dropped = AudioUtils.scope_events(EVENTS, takes, origin(ctx))
        if not scoped:
            return None
        return SceneRecords.AUDIO.make({"events": scoped, "fps": FPS})

    return produce


def _shots(shots=True):
    """A store with two shots sharing frame 40, made active.

    Parameters:
        shots: Define the shots; ``False`` leaves the store empty.

    Returns:
        The store.
    """
    store = ShotStore([])
    store.scene_fps = FPS
    if shots:
        store.define_shot("Intro", 1, 40, objects=["door_geo"])
        store.define_shot("Fight", 40, 80)
    ShotStore.set_active(store)
    return store


# ============================================================================
# The record
# ============================================================================


class TestTheRecordIsWhatTheProducersWrite(unittest.TestCase):
    """Both hosts' composition, in a real assembly after the real shots
    producer."""

    #: The export decisions a record is produced under.
    DECISIONS = (
        {},
        {"clip_mode": "shots", "clip_span": (1, 80)},
        {"clip_mode": "both", "clip_span": (1, 80)},
        {"clip_mode": "full", "clip_span": (3, 130)},
    )

    def setUp(self):
        ShotStore.clear_active()
        self.addCleanup(ShotStore.clear_active)

    def _produced(self, host, **decisions):
        """The ``audio`` payload *host*'s producer writes under *decisions*."""
        snapshot = ExportSnapshot.assemble(
            {
                SceneRecords.SHOTS: ShotStore.produce_export_records,
                SceneRecords.AUDIO: _audio_producer(ORIGINS[host]),
            },
            ExportContext(**decisions),
        )
        record = snapshot.produced[SceneRecords.AUDIO.key]
        self.assertIsNotNone(record)
        return record.payload

    def _payloads(self):
        """``{(host, decisions): payload}`` over every host and decision."""
        _shots()
        return {
            (host, json.dumps(decisions)): self._produced(host, **decisions)
            for host in ORIGINS
            for decisions in self.DECISIONS
        }

    def test_every_host_and_clip_mode_writes_the_declared_shape(self):
        for key, payload in self._payloads().items():
            with self.subTest(key):
                _assert_exact(self, payload)

    def test_a_file_with_no_shot_takes_writes_its_events_unscoped(self):
        _shots(shots=False)
        for host in ORIGINS:
            with self.subTest(host):
                payload = self._produced(host, clip_span=(1, 130))
                _assert_exact(self, payload)
                self.assertEqual({e["clip"] for e in payload["events"]}, {""})

    def test_every_declared_key_is_one_the_producers_write(self):
        payloads = list(self._payloads().values())
        self.assertEqual(set().union(*payloads), _declared(AudioRecord))
        events = [event for payload in payloads for event in payload["events"]]
        self.assertEqual(set().union(*events), _declared(AudioCue))


# ============================================================================
# What no producer writes
# ============================================================================


class TestWhatNoProducerWritesIsRefused(unittest.TestCase):
    """Each refusal named where it is, so a reader's validator can say so."""

    RECORD = {
        "version": 2,
        "events": [
            {"clip": "Intro", "frame": 4, "name": "footstep"},
            {"clip": "", "frame": 12, "name": "Lamp Hum"},
        ],
        "fps": 30.0,
    }

    def test_the_record_refuses_what_no_producer_writes(self):
        r = self.RECORD
        self.assertEqual(AudioRecord.validate(r).errors, [])
        cases = {
            "missing required key 'fps'": _edited(r, ["fps"], drop=True),
            "missing required key 'events'": _edited(r, ["events"], drop=True),
            "events[0].missing required key 'clip'": _edited(
                r, ["events", 0, "clip"], drop=True
            ),
            "events[0].clip: expected string, got null": _edited(
                r, ["events", 0, "clip"], None
            ),
            "events[0].frame: expected integer, got number": _edited(
                r, ["events", 0, "frame"], 4.5
            ),
            "events[0].frame: expected integer, got string": _edited(
                r, ["events", 0, "frame"], "4"
            ),
            "events[1].name: expected string, got integer": _edited(
                r, ["events", 1, "name"], 7
            ),
            "events: expected a list": _edited(r, ["events"], {}),
            "events[0].expected a mapping": _edited(r, ["events"], ["footstep"]),
            "version: expected integer, got string": _edited(r, ["version"], "2"),
        }
        for expected, payload in cases.items():
            with self.subTest(expected):
                errors = AudioRecord.validate(payload).errors
                self.assertTrue(any(expected in e for e in errors), errors)


# ============================================================================
# The published forms
# ============================================================================


class TestThePublishedForms(unittest.TestCase):
    """The JSON Schema unitytk's types are generated from, and the reference
    and skeleton an external reader is handed."""

    def test_the_json_schema_builds_with_its_one_nested_type(self):
        """``AudioCue``: the C# generator strips the root's family, so it
        reads ``AudioRecord.Cue`` -- not ``Event``, a C# contextual keyword
        and UnityEngine's IMGUI type."""
        doc = AudioRecord.json_schema()
        json.dumps(doc)
        self.assertEqual(doc["$schema"], AudioRecord.JSON_SCHEMA_DIALECT)
        self.assertEqual(doc["title"], "AudioRecord")
        self.assertEqual(sorted(doc["$defs"]), ["AudioCue"])
        self.assertEqual(doc["required"], ["version", "events", "fps"])
        self.assertEqual(doc["$defs"]["AudioCue"]["required"], ["clip", "frame", "name"])
        self.assertEqual(_required(AudioCue), ["clip", "frame", "name"])

    def test_the_skeleton_is_a_valid_payload_and_the_reference_names_every_key(self):
        _assert_exact(self, AudioRecord.skeleton())
        reference = AudioRecord.to_markdown()
        for key in _declared(AudioRecord) | _declared(AudioCue):
            self.assertIn(f"`{key}`", reference)


if __name__ == "__main__":
    unittest.main()
