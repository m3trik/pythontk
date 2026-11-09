# !/usr/bin/python
# coding=utf-8
"""The ``animation`` record's payload, declared once.

:attr:`pythontk.SceneRecords.SHOTS` names :class:`AnimationRecord` as its
shape: what the shot store publishes on the ``data_export`` carrier
(``ShotStore.export_records``: mayatk's and blendertk's stores publish through
it, overriding scene lookups only) -- per shot what a clip cannot hold, its
description and the metadata its author wrote for it
(``ShotStore.shipped_metadata``), joined to the file's take of the same name,
and the scene's frame rate. The file carries one take per shot, named after
it and spanning its range; the takes are the timing.

One layout, two containers: a GLB carries this record as it stands, under its
key in the root ``extras`` (``MeshConvert.apply_glb_animations``), each take
the glTF animation of its name -- so the rate rides it for the reader of a
format that holds none (glTF is in seconds), and a reader of one that states
its own (an FBX's time mode) uses that.

A typed :class:`~pythontk.SchemaSpec` declaration, so this one definition
validates a payload (``AnimationRecord.validate(payload)``), publishes its
JSON Schema, and is what ``m3trik/scripts/sync_scene_records.py`` generates the
WebXR runtime's JSDoc typedefs and unitytk's C# record types from -- a field
renamed here fails their checks until they are regenerated. Change a field
together with its producer, and bump the declared version
(``SceneRecords.SHOTS.version``).
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from pythontk.core_utils.schema_spec import SchemaSpec

_field = SchemaSpec.spec_field


@dataclass
class AnimationShot(SchemaSpec):
    """One shot: the take it is, by name, and what a clip cannot hold."""

    TYPED = True

    clip: str = _field(
        help=(
            "The shot's clip name, unique ignoring case: the name of the "
            "file's take for this shot (one take per shot, spanning its range) "
            "and the join key to the imported animation clip."
        ),
        required=True,
        example="Intro",
    )
    description: str = _field(
        help="What the shot shows, as free text; absent when it has none.",
        example="The door swings open.",
    )
    metadata: Dict[str, Any] = _field(
        help=(
            "What the shot's author wrote for it, every value JSON: the keys "
            "the scene's Shot Manifest writes from what its sheet says "
            "(ShotManifest.MANIFEST_METADATA and the sheet's pass-through "
            "columns); absent when it has none. The tools' own bookkeeping on "
            "a shot stays in the scene."
        ),
        example={"section": "A"},
    )


@dataclass
class AnimationRecord(SchemaSpec):
    """The ``animation`` record's payload: per shot its description and its
    metadata, each joined by name to the take the file carries for it, and
    the scene's frame rate."""

    TYPED = True

    version: int = _field(
        help=(
            "The record's schema version (SceneRecords.SHOTS.version): 3. A "
            "GLB carries the record under the key its retired GLB block rode "
            "at version 2, so 2 is skipped -- two shapes under one key and "
            "version would be read half-way; a reader refuses 1 (the FBX "
            "record before one take per shot) and 2 (that GLB block) alike, "
            "and the cure is a re-export."
        ),
        required=True,
        example=3,
    )
    shots: List[AnimationShot] = _field(
        help=(
            "The scene's shots in timeline order, the order of the file's "
            "takes -- a stale one (every object it names gone and nothing "
            "keyed in its frames) too, unless the scene leaves stale shots "
            "out. Empty in a scene that animates and declares no shot: the "
            "record then states the rate alone."
        ),
        required=True,
        nested=[AnimationShot],
        default_factory=list,
    )
    fps: float = _field(
        help=(
            "The scene's frame rate, the rate its takes were cut at: for a "
            "reader of a format that states none (a glTF's animations are in "
            "seconds), so a GLB, which carries this record as it stands, can "
            "be stepped frame by frame. A reader of a format that states its "
            "own (an FBX's time mode) uses that; the two agree "
            "(ExportVerifier.check_animation_rate)."
        ),
        required=True,
        example=30.0,
    )
