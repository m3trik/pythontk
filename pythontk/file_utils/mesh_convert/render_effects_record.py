# !/usr/bin/python
# coding=utf-8
"""The ``render_effects`` record's payload, declared once.

:attr:`pythontk.SceneRecords.VISIBILITY` names :class:`RenderEffectsRecord` as
its shape: what both DCCs' render effects publish through
``MeshConvert.build_visibility_tracks`` -- per node the effect channels it
carries, per material the channels its emission is animated by -- and what
the GLB build and every importer read back.

The record names channels; it holds no curve. Each curve rides the file's
animated-value transport (:attr:`pythontk.SceneRecords.TRANSPORT`): in an FBX
a curve-proxy transform per channel, in each take the shot plays, and in a
GLB a ``KHR_animation_pointer`` (or, for presence, a stepped ``scale``)
channel per clip. So a curve scopes to its clip by construction, and no
frame number or rate leaves the DCC. A track's channel names are
``SceneRecords.TRANSPORT``'s node channels and a material entry's its
material channels, held to that table by the tests.

Version 2 (2026-11-05): v1's ramps (``[[frame, value], ...]`` per channel),
its ``fps`` and its whole-timeline placement retired; v1 is refused, its fix
a re-export.
"""

from dataclasses import dataclass
from typing import List, Tuple

from pythontk.core_utils.schema_spec import SchemaSpec

_field = SchemaSpec.spec_field

#: A colour: linear ``[r, g, b]``.
_Colour = Tuple[float, float, float]


def _names_in(allowed: Tuple[str, ...]):
    """A ``validate`` callable: every name in the list is one of *allowed*,
    and none is repeated."""

    def check(value) -> List[str]:
        if not isinstance(value, list):
            return ["expected a list of channel names"]
        errors = [
            f"unknown channel {name!r}; expected one of {', '.join(allowed)}"
            for name in value
            if name not in allowed
        ]
        if len(set(map(str, value))) != len(value):
            errors.append("a channel is named twice")
        return errors

    return check


def _transport():
    """The transport declaration, whose channel tables the names are held to
    (read at validation: ``scene_records`` resolves this module lazily)."""
    from pythontk.core_utils.engines.scene_export.scene_records import Transport

    return Transport()


@dataclass
class RenderEffectsTrack(SchemaSpec):
    """One node's effects: which channels its subtree carries, and the
    highlight's colour stops."""

    TYPED = True

    node: str = _field(
        help=(
            "The node's name as its host names it: the join key to its curves "
            "(by the leaf after the last ':'). The node a curve plays on is its "
            "proxy's parent wherever the file carries the proxy; only a track "
            "with no proxy joins a node by this name, the exact name first, "
            "then the one node of its leaf, a tie never guessed."
        ),
        required=True,
        example="SIGN",
    )
    channels: List[str] = _field(
        help=(
            "The channels the node carries, each a curve on the transport: "
            "'visibility' (stepped: shown at 1, hidden at 0), 'opacity' (the "
            "subtree's materials' alpha, 0..1) and 'highlight' (an added glow, "
            "0..1)."
        ),
        required=True,
        example=["opacity", "highlight"],
        default_factory=list,
        validate=lambda value: _names_in(_transport().node_channels)(value),
    )
    highlight_color: _Colour = _field(
        help="The highlight's colour at 1.0, linear RGB; absent for the default (white).",
        example=[0.1, 0.4, 1.0],
    )
    highlight_color_dim: _Colour = _field(
        help="The highlight's colour at 0.0, linear RGB; absent for the default (black).",
        example=[0.0, 0.0, 0.0],
    )


@dataclass
class RenderEffectsMaterial(SchemaSpec):
    """One material whose emission is animated: the channel names it, the
    curve (a weight 0..1) rides the transport."""

    TYPED = True

    material: str = _field(
        help=(
            "The material's name, as the delivered material is named (the "
            "'materials' record's entry); where an import drops the namespace, "
            "the name after the last ':' joins."
        ),
        required=True,
        example="screen_mat",
    )
    channels: List[str] = _field(
        help="The material's animated channels: 'emission', its emission's weight 0..1.",
        required=True,
        example=["emission"],
        default_factory=list,
        validate=lambda value: _names_in(_transport().material_channels)(value),
    )
    emission_color: _Colour = _field(
        help=(
            "The emission's colour at weight 1, linear RGB; absent: the "
            "material's own emissive colour."
        ),
        example=[1.0, 0.5, 0.0],
    )
    emission_strength: float = _field(
        help=(
            "The emission's strength at weight 1 (a multiplier of the colour, "
            "above 1 for a glow brighter than white); absent: the material's own."
        ),
        example=4.0,
    )


@dataclass
class RenderEffectsRecord(SchemaSpec):
    """The ``render_effects`` record's payload: which nodes and materials
    carry an animated effect, and the colours their curves scale."""

    TYPED = True

    version: int = _field(
        help="The record's schema version (SceneRecords.VISIBILITY.version).",
        required=True,
        example=2,
    )
    tracks: List[RenderEffectsTrack] = _field(
        help="One per node that carries an animated effect.",
        required=True,
        nested=[RenderEffectsTrack],
        default_factory=list,
    )
    materials: List[RenderEffectsMaterial] = _field(
        help="One per material whose emission is animated; absent when none is.",
        nested=[RenderEffectsMaterial],
    )
