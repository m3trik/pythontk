# !/usr/bin/python
# coding=utf-8
"""The ``audio`` record's declared shape (:class:`AudioRecord`) against what
its producers write, and the one rule they cut cues by.

mayatk's ``AudioClips.export_record`` and blendertk's
``AudioUtils.export_record`` build the record alike: each sound's span on the
timeline cut by ``ptk.AudioUtils.clip_cues`` into one cue per clip -- the
takes the export splits (``ShotStore.declared_takes``: one per shot), or a
shotless file's one take, the span it holds -- in seconds from the clip's
start, with the offset into the sound and the stop; made with
``SceneRecords.AUDIO.make({"cues"})``. ``media`` is the FBX dependency
pass's to write (``FbxMedia.embed_dependencies``, from the bytes it
embeds). That composition, less the scene reads, runs here in a real
``ExportSnapshot`` assembly after the real shots producer. Exact both ways:
every payload validates with no unknown-key warning, and every declared key
is one the producers write. Added: 2026-10-20; version 3 (cues per clip, in
seconds): 2026-11-05
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
#: The scene's sounds on the timeline, by frame: one in each shot, one across
#: the frame the two shots share, one past every shot.
SOUNDS = [
    {"sound": "footstep.wav", "start": 5, "stop": 15},
    {"sound": "door_slam.wav", "start": 30, "stop": 50},
    {"sound": "Lamp Hum.ogg", "start": 52, "stop": 60},
    {"sound": "late.wav", "start": 130, "stop": 140},
]
#: The window a shotless file's one take declares: what it holds, measured
#: off the written file (``FbxMedia.fit_take_windows``).
WINDOW = (0, 100)


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


def _audio_producer(sounds=SOUNDS):
    """A DCC's ``audio`` producer over *sounds*: the composition both hosts'
    ``export_record`` share, less their scene reads.

    Returns:
        The producer, taking the export's context.
    """

    def produce(ctx):
        takes = ShotStore.declared_takes()
        cues, _dropped = AudioUtils.clip_cues(sounds, takes, FPS, window=WINDOW)
        if not cues:
            return None
        return SceneRecords.AUDIO.make({"cues": cues})

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

    def setUp(self):
        ShotStore.clear_active()
        self.addCleanup(ShotStore.clear_active)

    def _produced(self, sounds=SOUNDS):
        """The ``audio`` payload a producer writes, or ``None``."""
        snapshot = ExportSnapshot.assemble(
            {
                SceneRecords.SHOTS: ShotStore.produce_export_records,
                SceneRecords.AUDIO: _audio_producer(sounds),
            },
            ExportContext(),
        )
        record = snapshot.produced.get(SceneRecords.AUDIO.key)
        return record.payload if record is not None else None

    def _payloads(self):
        """``{shots: payload}``, with and without shots."""
        payloads = {}
        for shots in (True, False):
            _shots(shots)
            payloads[shots] = self._produced()
        return payloads

    def test_every_payload_is_the_declared_shape_at_version_3(self):
        for key, payload in self._payloads().items():
            with self.subTest(shots=key):
                _assert_exact(self, payload)
                self.assertEqual(payload["version"], 3)
                self.assertEqual(SceneRecords.AUDIO.version, 3)

    def test_the_cues_are_cut_per_shot(self):
        _shots()
        payload = self._produced()
        self.assertEqual(
            [
                (c["clip"], c["sound"], round(c["time"] * FPS), round(c["stop"] * FPS))
                for c in payload["cues"]
            ],
            [
                ("Intro", "footstep.wav", 4, 14),
                ("Intro", "door_slam.wav", 29, 39),
                ("Fight", "door_slam.wav", 0, 10),
                ("Fight", "Lamp Hum.ogg", 12, 20),
            ],
        )

    def test_a_shotless_file_cues_its_one_take(self):
        _shots(shots=False)
        payload = self._produced()
        self.assertEqual({c["clip"] for c in payload["cues"]}, {""})
        self.assertEqual(
            [round(c["time"] * FPS) for c in payload["cues"]], [5, 30, 52]
        )

    def test_a_scene_with_no_audio_writes_no_record(self):
        _shots()
        self.assertIsNone(self._produced(sounds=[]))

    def test_every_declared_key_is_one_the_producers_or_the_dependency_pass_write(self):
        """``media`` is the FBX dependency pass's (``FbxMedia.embed_dependencies``):
        it digests the bytes it embedded, so a write without it -- a raw File >
        Export -- names no audio it does not carry (2026-10-31)."""
        payloads = list(self._payloads().values())
        self.assertEqual(set().union(*payloads), _declared(AudioRecord) - {"media"})
        cues = [cue for payload in payloads for cue in payload["cues"]]
        self.assertEqual(set().union(*cues), _declared(AudioCue))

    def test_no_frame_and_no_rate_leaves_the_dcc(self):
        """Version 3: seconds from the clip's start; ``fps`` and frame fields
        retire (the ``animation`` record keeps the scene's rate)."""
        self.assertNotIn("fps", _declared(AudioRecord))
        self.assertFalse({"frame", "name"} & _declared(AudioCue))


# ============================================================================
# What no producer writes
# ============================================================================


class TestWhatNoProducerWritesIsRefused(unittest.TestCase):
    """Each refusal named where it is, so a reader's validator can say so."""

    RECORD = {
        "version": 3,
        "cues": [
            {"clip": "Intro", "sound": "footstep.wav", "time": 0.4, "offset": 0.0, "stop": 1.2},
            {"clip": "", "sound": "Lamp Hum.ogg", "time": 0.0, "offset": 0.25, "stop": 2.0},
        ],
    }

    def test_a_record_without_digests_still_reads(self):
        """A raw export carries no ``media``: still the current record."""
        _assert_exact(self, self.RECORD)
        self.assertTrue(SceneRecords.AUDIO.is_current(self.RECORD))

    def test_versions_1_and_2_are_refused_with_their_fix(self):
        """Frames at a rate (v2) and the flat wire string (v1) read as
        another schema: refused, re-export."""
        v2 = {
            "version": 2,
            "events": [{"clip": "Intro", "frame": 4, "name": "footstep"}],
            "fps": 30.0,
        }
        for payload in (v2, {"version": 1}):
            with self.subTest(payload["version"]):
                self.assertFalse(SceneRecords.AUDIO.is_current(payload))
                problems = SceneRecords.problems(SceneRecords.AUDIO, payload)
                self.assertTrue(problems, payload)

    def test_the_record_refuses_what_no_producer_writes(self):
        r = dict(self.RECORD, media={"footstep.wav": "ab" * 32})
        self.assertEqual(AudioRecord.validate(r).errors, [])
        cases = {
            "missing required key 'cues'": _edited(r, ["cues"], drop=True),
            "cues[0].missing required key 'clip'": _edited(
                r, ["cues", 0, "clip"], drop=True
            ),
            "cues[0].missing required key 'stop'": _edited(
                r, ["cues", 0, "stop"], drop=True
            ),
            "cues[0].missing required key 'offset'": _edited(
                r, ["cues", 0, "offset"], drop=True
            ),
            "cues[0].clip: expected string, got null": _edited(
                r, ["cues", 0, "clip"], None
            ),
            "cues[0].time: expected number, got string": _edited(
                r, ["cues", 0, "time"], "0.4"
            ),
            "cues[1].sound: expected string, got integer": _edited(
                r, ["cues", 1, "sound"], 7
            ),
            "cues: expected a list": _edited(r, ["cues"], {}),
            "cues[0].expected a mapping": _edited(r, ["cues"], ["footstep.wav"]),
            "version: expected integer, got string": _edited(r, ["version"], "3"),
            "media.footstep.wav: expected string, got integer": _edited(
                r, ["media", "footstep.wav"], 7
            ),
            "media: expected object, got array": _edited(r, ["media"], ["ab"]),
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
        self.assertEqual(doc["required"], ["version", "cues"])
        media = doc["properties"]["media"]
        self.assertEqual(
            (media["type"], media["additionalProperties"]),
            ("object", {"type": "string"}),
        )
        expected = ["clip", "sound", "time", "offset", "stop"]
        self.assertEqual(doc["$defs"]["AudioCue"]["required"], expected)
        self.assertEqual(_required(AudioCue), expected)

    def test_the_skeleton_is_a_valid_payload_and_the_reference_names_every_key(self):
        _assert_exact(self, AudioRecord.skeleton())
        reference = AudioRecord.to_markdown()
        for key in _declared(AudioRecord) | _declared(AudioCue):
            self.assertIn(f"`{key}`", reference)

    def test_a_glb_carries_the_record_itself(self):
        """One layout, two containers: the record rides a GLB as it stands,
        under its key in the root extras, like ``animation``."""
        spec = SceneRecords.AUDIO
        self.assertTrue(spec.glb_verbatim)
        self.assertIs(SceneRecords.shape(spec), AudioRecord)
        self.assertEqual(spec.version, 3)
        self.assertIn(spec, SceneRecords.glb_records())

    def test_the_record_says_who_owns_the_cues(self):
        """The source owns the delivered cues; what the engine's user adds in
        its importer's Events tab is the engine's."""
        text = " ".join([SceneRecords.AUDIO.description, SceneRecords.AUDIO.assembly])
        self.assertIn("source", text)
        self.assertIn("Events tab", text)


if __name__ == "__main__":
    unittest.main()
