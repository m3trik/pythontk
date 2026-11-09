# !/usr/bin/python
# coding=utf-8
"""The ``audio`` record's payload, declared once.

:attr:`pythontk.SceneRecords.AUDIO` names :class:`AudioRecord` as its shape:
what both DCCs' audio producers publish on the ``data_export`` carrier --
mayatk's ``AudioClips.export_record`` from its keyed tracks, blendertk's
``AudioUtils.export_record`` from its sound strips -- and what unitytk's
Animation Audio importer writes into the imported clips as animation events.
Both producers cut their sounds into cues per clip by the one rule they share,
:meth:`AudioUtils.clip_cues <pythontk.AudioUtils.clip_cues>`, fed the takes
the write makes and, for a file with no shots, the span its one take holds;
the FBX dependency pass that embeds the sound files
(``FbxMedia.embed_dependencies``) digests what it embedded by another,
:meth:`AudioUtils.media_digests <pythontk.AudioUtils.media_digests>` -- which
is why the shape lives here, beside them.

**Whoever owns the animation owns its cues.** The cues are timing on the
animation, and the animation is the source's: every delivery and re-import
carries the current timing, so a retimed motion never drifts from its sound,
and a delivered cue is changed at the source, like a delivered key. What
the importing engine's own user adds to a clip at import is the engine's,
kept beside the delivered cues and never overwritten by them (the record's
declaration names the engine: ``SceneRecords.AUDIO``).

A typed :class:`~pythontk.SchemaSpec`: this one definition validates a payload
(``AudioRecord.validate(payload)``), publishes its JSON Schema, and is what
``m3trik/scripts/sync_scene_records.py`` generates unitytk's C# record types
and the WebXR runtime's typedef from -- a field renamed here fails their
checks until they are regenerated. Change a field together with
``AudioUtils.clip_cues`` / ``AudioUtils.media_digests``, the producers and
that pass, and bump ``SceneRecords.AUDIO.version`` -- unless the field is
optional and a reader that lacks it reads the record as before (a reader
reads one version alone).
"""

from dataclasses import dataclass
from typing import Dict, List

from pythontk.core_utils.schema_spec import SchemaSpec

_field = SchemaSpec.spec_field


@dataclass
class AudioCue(SchemaSpec):
    """One cue: a sound that plays in one clip, from a time of that clip to
    its stop."""

    TYPED = True

    clip: str = _field(
        help=(
            "The clip that plays the cue: the name of the file's take for it "
            "(one take per shot, named after the shot), matched ignoring case; "
            "'' in a file with no shots, whose one take is the clip."
        ),
        required=True,
        example="Intro",
    )
    sound: str = _field(
        help=(
            "The sound that plays: the name of the audio file the delivery "
            "carries for it, extension included, matched ignoring case."
        ),
        required=True,
        example="footstep.wav",
    )
    time: float = _field(
        help="When the sound starts, in seconds from the start of its clip.",
        required=True,
        example=0.4,
    )
    offset: float = _field(
        help=(
            "Where in the sound it starts, in seconds from the sound's start: "
            "0 for a sound heard from its beginning, more for one that began "
            "before its clip (a sound a cut between shots divides) or that its "
            "author trimmed at the head."
        ),
        required=True,
        example=0.0,
    )
    stop: float = _field(
        help=(
            "When the sound stops, in seconds from the start of its clip: its "
            "end on the timeline, or the end of its clip when it runs past it. "
            "No later than the clip's end, and after 'time'."
        ),
        required=True,
        example=1.2,
    )


@dataclass
class AudioRecord(SchemaSpec):
    """The ``audio`` record's payload: the scene's sounds, cut into cues per
    clip, in seconds."""

    TYPED = True

    version: int = _field(
        help="The record's schema version (SceneRecords.AUDIO.version).",
        required=True,
        example=3,
    )
    cues: List[AudioCue] = _field(
        help=(
            "Every cue, clip by clip in the order of the takes and by time "
            "within a clip: a sound that spans several clips is cut into one "
            "cue per clip, each starting at its offset into the sound and "
            "stopping at the cut. A sound in no clip is left out."
        ),
        required=True,
        nested=[AudioCue],
        default_factory=list,
    )
    media: Dict[str, str] = _field(
        help=(
            "The content digest of each audio file the delivery embeds, by the "
            "'sound' that names it: the SHA-256 of the bytes inside, in "
            "lowercase hex, written by the pass that embedded them. A "
            "same-named sound found elsewhere stands in for it only when its "
            "bytes hash alike; else the delivery's own is taken. A sound it "
            "carries no file for is left out, and a file written without that "
            "pass (a raw export) carries no 'media' at all."
        ),
        example={
            "footstep.wav": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        },
        default_factory=dict,
    )
