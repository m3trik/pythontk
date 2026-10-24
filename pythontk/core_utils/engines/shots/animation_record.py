# !/usr/bin/python
# coding=utf-8
"""The ``animation`` record's payload and its GLB block, declared once.

:attr:`pythontk.SceneRecords.SHOTS` names these as its shapes:

- :class:`AnimationRecord` -- what the shot store publishes on the
  ``data_export`` carrier (``ShotStore.export_records``: mayatk's and
  blendertk's stores publish through it, overriding scene lookups only): the
  scene's shots, each the take a clip is cut from, its frames counted at the
  record's ``fps``.
- :class:`AnimationBlock` -- the ``extras.animation`` block
  ``MeshConvert.apply_glb_animations`` writes for a GLB: the record's shots
  inherited whole, whatever clips the file holds, and every glTF animation by
  index, placed on the authoring timeline by the frame its ``t=0`` sits on.
  A shot plays the clip it names, else its window of the whole-timeline clip.

They are typed :class:`~pythontk.SchemaSpec` declarations, so this one
definition validates a payload (``AnimationRecord.validate(payload)``),
publishes its JSON Schema, and is what ``m3trik/scripts/sync_scene_records.py``
generates the WebXR runtime's JSDoc typedefs and unitytk's C# record types
from -- a field renamed here fails their checks until they are regenerated.
Change a field together with its producer, and bump the declared version
(``SceneRecords.SHOTS.version``; the block's own,
``SceneRecords.SHOTS.glb.version``).
"""

from dataclasses import dataclass
from typing import List

from pythontk.core_utils.engines.scene_export.export_profile import ExportProfile
from pythontk.core_utils.schema_spec import SchemaSpec

_field = SchemaSpec.spec_field

#: ``clip_mode``'s values: the exporter's Animation Clips modes, from the one
#: table that declares them.
_CLIP_MODES = tuple(ExportProfile.ANIMATION_CLIPS_OPTIONS.values())


@dataclass
class AnimationShot(SchemaSpec):
    """One shot: the span of the authoring timeline its take is cut from, and
    what it shows."""

    TYPED = True

    clip: str = _field(
        help=(
            "The shot's clip name, unique ignoring case: the name of the take "
            "or animation cut for it (a 'full' file cuts none) and the join "
            "key to the imported animation clip."
        ),
        required=True,
        example="Intro",
    )
    start: int = _field(
        help=(
            "The shot's first frame on the authoring timeline, a whole frame "
            "at the 'fps' beside it."
        ),
        required=True,
        example=1,
    )
    end: int = _field(
        help=(
            "The shot's last frame on the authoring timeline (inclusive), a "
            "whole frame at the 'fps' beside it."
        ),
        required=True,
        example=120,
    )
    description: str = _field(
        help="What the shot shows, as free text; absent when it has none.",
        example="The door swings open.",
    )


@dataclass
class AnimationRecord(SchemaSpec):
    """The ``animation`` record's payload: the scene's shots, each one take of
    the file."""

    TYPED = True

    version: int = _field(
        help="The record's schema version (SceneRecords.SHOTS.version).",
        required=True,
        example=1,
    )
    fps: float = _field(
        help=(
            "The scene's frame rate: how many of the authoring timeline's "
            "frames, which every 'start' and 'end' counts, make one second."
        ),
        required=True,
        example=30.0,
    )
    shots: List[AnimationShot] = _field(
        help=(
            "The scene's shots in start-frame order -- a stale one (every "
            "object it names gone and nothing keyed in its frames) too, "
            "unless the scene leaves stale shots out."
        ),
        required=True,
        nested=[AnimationShot],
        default_factory=list,
    )
    clip_mode: str = _field(
        help=(
            "How the exporter cut the file's takes: 'shots', one per shot; "
            "'full', one whole-timeline take in which each shot is the window "
            "'start'..'end'; 'both', both; absent when no exporter declared one "
            "(an authoring-time publish, an export outside the Scene Exporter), "
            "which reads as 'both'."
        ),
        choices=_CLIP_MODES,
        example="shots",
    )


# ------------------------------------------------------------- the GLB block
@dataclass
class AnimationBlockClip(SchemaSpec):
    """One glTF animation: its name, its index, and the authoring frame its
    time 0 sits on."""

    TYPED = True

    name: str = _field(
        help=(
            "The glTF animation's name ('animation_<index>' when it has none): "
            "the clip a shot whose 'clip' it is plays."
        ),
        required=True,
        example="Intro",
    )
    animation: int = _field(
        help="The clip's index in the glTF's 'animations' array.",
        required=True,
        example=1,
    )
    zero_frame: float = _field(
        help=(
            "The authoring frame the clip's time 0 sits on, so a frame f is "
            "(f - zero_frame) / fps seconds into it: a shot's own clip opens "
            "on its 'start', the whole-timeline clip on its first key; present "
            "whenever the block states 'fps', unless the file declares no take "
            "and publishes no span for the clip."
        ),
        example=1.0,
    )


@dataclass
class AnimationBlock(SchemaSpec):
    """The ``extras.animation`` block: the scene's shots as the ``animation``
    record states them, and the file's clips by glTF animation index, each
    placed on the authoring timeline."""

    TYPED = True

    version: int = _field(
        help="The block's own schema version (SceneRecords.SHOTS.glb.version).",
        required=True,
        example=2,
    )
    shots: List[AnimationShot] = _field(
        help=(
            "The scene's shots, in start-frame order, as the 'animation' "
            "record states them -- every one, whatever clips the file holds."
        ),
        required=True,
        nested=[AnimationShot],
        default_factory=list,
    )
    clips: List[AnimationBlockClip] = _field(
        help="One entry per glTF animation, in the file's order.",
        required=True,
        nested=[AnimationBlockClip],
        default_factory=list,
    )
    default_clip: str = _field(
        help=(
            "The name of the clip to open on: the first shot's clip that has "
            "channels, else the first clip that has channels, else the first "
            "shot's clip, else the first clip; absent when the file holds no "
            "clip."
        ),
        example="Intro",
    )
    fps: float = _field(
        help=(
            "The frame rate every frame number here counts at: the "
            "'animation' record's 'fps', else the 'render_effects' record's, "
            "else one derived from a shot's range and its clip's duration; "
            "absent when none gives one."
        ),
        example=30.0,
    )
    clip_mode: str = _field(
        help=(
            "The 'animation' record's 'clip_mode': how the exporter cut the "
            "takes the clips came from; absent when it declared none."
        ),
        choices=_CLIP_MODES,
        example="shots",
    )
