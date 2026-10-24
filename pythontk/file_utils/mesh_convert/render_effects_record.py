# !/usr/bin/python
# coding=utf-8
"""The ``render_effects`` record's payload, declared once.

:attr:`pythontk.SceneRecords.VISIBILITY` names :class:`RenderEffectsRecord` as
its shape: what both DCCs' render effects publish through
``MeshConvert.build_visibility_tracks`` -- per node its stepped visibility and
the ramps of the effects glTF has no channel for, which the GLB build plays as
``KHR_animation_pointer`` material channels (``GlbFades.CHANNELS``) -- and
what ``MeshConvert.apply_glb_visibility`` reads back. A track's effect keys
are ``GlbFades.CHANNELS``' names and colour stops, pinned to that table by
the tests.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from pythontk.core_utils.schema_spec import SchemaSpec

_field = SchemaSpec.spec_field

#: A ramp: ``[[frame, value], ...]``, in the record's ``fps``.
_Ramp = List[Tuple[float, float]]
#: A colour: linear ``[r, g, b]``.
_Colour = Tuple[float, float, float]


@dataclass
class RenderEffectsTrack(SchemaSpec):
    """One node's tracks: its presence and the effect ramps it carries."""

    TYPED = True

    node: str = _field(
        help="The node's name, without namespace: the join key to the file's node.",
        required=True,
    )
    visibility: _Ramp = _field(
        help="Stepped presence keys ([[frame, 0 or 1], ...]); absent when not keyed.",
    )
    opacity: _Ramp = _field(
        help=(
            "The fade ramp ([[frame, 0..1], ...], linear): the node's presence "
            "where it has no visibility keys; absent when not keyed."
        ),
    )
    highlight: _Ramp = _field(
        help="The highlight ramp ([[frame, 0..1], ...], linear); absent when not keyed.",
    )
    highlight_color: _Colour = _field(
        help="The highlight's colour at 1.0, linear RGB; absent for the default.",
    )
    highlight_color_dim: _Colour = _field(
        help="The highlight's colour at 0.0, linear RGB; absent for the default.",
    )


@dataclass
class RenderEffectsRecord(SchemaSpec):
    """The ``render_effects`` record's payload: the keyed visibility and the
    effect ramps per node, and the whole timeline's span."""

    TYPED = True

    version: int = _field(
        help="The record's schema version (SceneRecords.VISIBILITY.version).",
        required=True,
    )
    tracks: List[RenderEffectsTrack] = _field(
        help="One per node that carries a keyed effect; empty when only spans ride.",
        required=True,
        nested=[RenderEffectsTrack],
        default_factory=list,
    )
    fps: float = _field(
        help="The rate every frame number here is quoted in; absent when unknown.",
    )
    clip_span: Dict[str, Tuple[float, float]] = _field(
        help=(
            "The first and last frame each named stack carries -- the whole "
            "timeline's under '*' (MeshConvert.DEFAULT_CLIP_SPAN) -- which "
            "places every clip cut from it."
        ),
    )
