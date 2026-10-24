# !/usr/bin/python
# coding=utf-8
"""The ``audio`` record's payload, declared once.

:attr:`pythontk.SceneRecords.AUDIO` names :class:`AudioRecord` as its shape:
what both DCCs' audio producers publish on the ``data_export`` carrier --
mayatk's ``AudioClips.export_record`` from its keyed tracks, blendertk's
``AudioUtils.export_record`` from its sound strips -- and what unitytk's
``AudioEventImporter`` injects into the imported clips as animation events.
Both producers scope their events to the takes the ``animation`` record
declares by the one rule they share, :meth:`AudioUtils.scope_events
<pythontk.AudioUtils.scope_events>`, which is why the shape lives here,
beside it.

A typed :class:`~pythontk.SchemaSpec`: this one definition validates a payload
(``AudioRecord.validate(payload)``), publishes its JSON Schema, and is what
``m3trik/scripts/sync_scene_records.py`` generates unitytk's C# record types
from -- a field renamed here fails their checks until they are regenerated.
Change a field together with ``AudioUtils.scope_events`` and the producers,
and bump ``SceneRecords.AUDIO.version``.
"""

from dataclasses import dataclass
from typing import List

from pythontk.core_utils.schema_spec import SchemaSpec

_field = SchemaSpec.spec_field


@dataclass
class AudioCue(SchemaSpec):
    """One audio event: a track that starts on a frame of the clip that plays
    it."""

    TYPED = True

    clip: str = _field(
        help=(
            "The take that plays the event, as the 'animation' record's 'clip' "
            "names it (matched ignoring case), or '' when the file has no shot "
            "takes and every clip plays it."
        ),
        required=True,
        example="Intro",
    )
    frame: int = _field(
        help=(
            "The frame the track starts on, a whole number of frames at the "
            "record's 'fps' from the start of its clip: its take's 'start', or "
            "for an unscoped event the first frame of the whole-timeline take."
        ),
        required=True,
        example=12,
    )
    name: str = _field(
        help=(
            "The track that plays: the audio file whose stem it is, ignoring "
            "case, which the deliverable carries as media of its own."
        ),
        required=True,
        example="footstep",
    )


@dataclass
class AudioRecord(SchemaSpec):
    """The ``audio`` record's payload: the scene's audio events, each scoped
    to the clip that plays it."""

    TYPED = True

    version: int = _field(
        help="The record's schema version (SceneRecords.AUDIO.version).",
        required=True,
        example=2,
    )
    events: List[AudioCue] = _field(
        help=(
            "Every event in authoring-timeline order: a scoped one listed once "
            "for each take that holds its frame (one outside every take is left "
            "out), an unscoped one once."
        ),
        required=True,
        nested=[AudioCue],
        default_factory=list,
    )
    fps: float = _field(
        help=(
            "The scene's frame rate: how many of the frames every event's "
            "'frame' counts make one second."
        ),
        required=True,
        example=30.0,
    )
