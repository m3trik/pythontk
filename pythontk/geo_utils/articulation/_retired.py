# !/usr/bin/python
# coding=utf-8
"""Held until 0.22.0: the articulation block names v0.20.0 noticed for removal.

``ArticulationWeb`` and its rig, joint and grab were noticed for 0.18.0 on
2026-10-24, so their window closes 2026-11-23, after 0.21.0 ships. The
index-bound block they name retired whole on 2026-11-08 (a GLB carries the
``articulation`` record as it stands), so the block's shapes are restored
here from v0.20.0 -- a joint's pose without the anchor version 2 added,
joints and grabbed parts by glTF node index -- under their notices, for a
caller of the old names. ``record`` re-exports them; delete this module and
that re-export at 0.22.0 (``pythontk/__init__.py``, "Held until 0.22.0").
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from pythontk.core_utils.deprecation import Deprecation
from pythontk.core_utils.schema_spec import SchemaSpec
from pythontk.geo_utils.articulation.model import ROTATE_ORDERS
from pythontk.geo_utils.articulation.record import ArticulationChannel, ArticulationRig

_field = SchemaSpec.spec_field


@dataclass
class _BlockJointPose(SchemaSpec):
    """A joint's rest frame in its parent's space and its channels, as
    v0.20.0's block stated it."""

    TYPED = True

    parent: Optional[int] = _field(
        help="An earlier joint's index, or null for a root.", required=True
    )
    t: Tuple[float, float, float] = _field(
        help="The rest translate in the parent's space, in the file's units.",
        required=True,
    )
    q: Tuple[float, float, float, float] = _field(
        help="The rest rotation, a quaternion x, y, z, w.", required=True
    )
    rotate_order: str = _field(
        help="The order the rotation channels compose in.",
        required=True,
        choices=ROTATE_ORDERS,
    )
    channels: List[ArticulationChannel] = _field(
        help="The channels it turns or slides on, in state order.",
        required=True,
        nested=[ArticulationChannel],
        default_factory=list,
    )


@dataclass
class _BlockJoint(_BlockJointPose):
    """A joint bound to the file: its pose and its glTF node."""

    node: int = _field(help="The joint's glTF node index.", required=True)


@dataclass
class _BlockGrab(SchemaSpec):
    """A grabbed part bound to its glTF node."""

    TYPED = True

    node: int = _field(help="The part's glTF node index.", required=True)
    joint: int = _field(help="The index of the joint the part rides.", required=True)


@dataclass
class _BlockRig(ArticulationRig):
    """One rig bound to the file: its joints and grabbed parts by node index."""

    joints: List[_BlockJoint] = _field(
        help="Its joints, each after its parent.",
        required=True,
        nested=[_BlockJoint],
        default_factory=list,
    )
    grab: List[_BlockGrab] = _field(
        help="The parts a hand takes hold of that resolved to a node.",
        required=True,
        nested=[_BlockGrab],
        default_factory=list,
    )


@dataclass
class _Block(SchemaSpec):
    """The retired ``extras.articulation`` block: the rigs bound to the
    file's glTF nodes."""

    TYPED = True

    version: int = _field(help="The block's schema version.", required=True)
    rigs: List[_BlockRig] = _field(
        help="The rigs whose every joint is in the file.",
        required=True,
        nested=[_BlockRig],
        default_factory=list,
    )


_RECORD = "ArticulationRecord -- a GLB carries the record as it stands"


@Deprecation.symbol(_RECORD, remove_in="0.18.0", since="2026-10-24")
class ArticulationWeb(_Block):
    """Retired 2026-10-20 (removed after 0.21.0): :class:`ArticulationRecord`."""


@Deprecation.symbol(_RECORD, remove_in="0.18.0", since="2026-10-24")
class ArticulationWebRig(_BlockRig):
    """Retired 2026-10-20 (removed after 0.21.0): ``ArticulationRig``."""


@Deprecation.symbol(_RECORD, remove_in="0.18.0", since="2026-10-24")
class ArticulationWebJoint(_BlockJoint):
    """Retired 2026-10-20 (removed after 0.21.0): ``ArticulationJoint``."""


@Deprecation.symbol(_RECORD, remove_in="0.18.0", since="2026-10-24")
class ArticulationWebGrab(_BlockGrab):
    """Retired 2026-10-20 (removed after 0.21.0): ``ArticulationGrab``."""
