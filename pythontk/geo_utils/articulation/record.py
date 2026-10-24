# !/usr/bin/python
# coding=utf-8
"""The ``articulation`` record's payload and its GLB block, declared once.

:attr:`pythontk.SceneRecords.ARTICULATION` names these as its shapes:

- :class:`ArticulationRecord` -- what the DCC's ``ArticulatedRig`` publishes on
  the ``data_export`` carrier, and what every runtime builds its
  :class:`~pythontk.ArticulationModel` from (one rig at a time).
- :class:`ArticulationBlock` -- the ``extras.articulation`` block
  ``MeshConvert.apply_glb_articulation`` derives from it for a GLB: the same
  rigs, each joint and grabbed part bound to its glTF node.

They are typed :class:`~pythontk.SchemaSpec` declarations, so this one
definition validates a payload (``ArticulationRecord.validate(payload)``),
publishes its JSON Schema, and is what ``m3trik/scripts/sync_scene_records.py``
generates the WebXR runtime's JSDoc typedefs and unitytk's C# record types
from -- a field renamed here fails their checks until they are regenerated.
Change a field together with :class:`~pythontk.ArticulationModel` and the
producers, and bump the record's declared version.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from pythontk.core_utils.deprecation import Deprecation
from pythontk.core_utils.schema_spec import SchemaSpec
from pythontk.geo_utils.articulation.model import CHANNELS, ROTATE_ORDERS

_field = SchemaSpec.spec_field


@dataclass
class ArticulationChannel(SchemaSpec):
    """One channel a joint turns or slides on, with its limits."""

    TYPED = True

    channel: str = _field(
        help=(
            "rx/ry/rz turn about the rest frame's own axes (degrees); tx/ty/tz "
            "slide along them (the file's units, in the parent's space)."
        ),
        required=True,
        choices=CHANNELS,
    )
    min: Optional[float] = _field(
        help="The lower limit, or null for none.", required=True
    )
    max: Optional[float] = _field(
        help="The upper limit, or null for none.", required=True
    )
    weight: float = _field(
        help="How readily a grab moves this channel against the others (1 = evenly).",
        required=True,
        default=1.0,
    )


@dataclass
class ArticulationJointPose(SchemaSpec):
    """A joint's rest frame in its parent's space and its channels, the same
    on every carrier."""

    TYPED = True

    parent: Optional[int] = _field(
        help="An earlier joint's index, or null for a root (its parent is the rig's space).",
        required=True,
    )
    t: Tuple[float, float, float] = _field(
        help=(
            "The rest translate in the parent's space, in the file's units (a "
            "runtime measures its scale against the rest pose: "
            "ArticulationModel.scale_of)."
        ),
        required=True,
    )
    q: Tuple[float, float, float, float] = _field(
        help="The rest rotation (the DCC's joint orient), a quaternion x, y, z, w.",
        required=True,
    )
    rotate_order: str = _field(
        help="The order the rotation channels compose in; the first letter applies first.",
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
class ArticulationJoint(ArticulationJointPose):
    """One joint of the record: its pose, joined to its node by name."""

    name: str = _field(
        help="The joint node's name: the join key to the imported transform.",
        required=True,
    )


@dataclass
class ArticulationGrab(SchemaSpec):
    """A part a hand grabs, and the joint it rides."""

    TYPED = True

    node: str = _field(
        help="The part's node name; a runtime resolves it under its joint's node.",
        required=True,
    )
    joint: int = _field(help="The index of the joint the part rides.", required=True)


@dataclass
class ArticulationRig(SchemaSpec):
    """One rig: its joints, parents first, and the parts a hand grabs."""

    TYPED = True

    name: str = _field(help="The rig's name.", required=True)
    joints: List[ArticulationJoint] = _field(
        help="Its joints, each after its parent.",
        required=True,
        nested=[ArticulationJoint],
        default_factory=list,
    )
    grab: List[ArticulationGrab] = _field(
        help="The parts a hand takes hold of.",
        required=True,
        nested=[ArticulationGrab],
        default_factory=list,
    )


@dataclass
class ArticulationRecord(SchemaSpec):
    """The ``articulation`` record's payload: every articulated rig in the scene."""

    TYPED = True

    version: int = _field(
        help="The record's schema version (SceneRecords.ARTICULATION.version).",
        required=True,
    )
    rigs: List[ArticulationRig] = _field(
        help="The scene's articulated rigs.",
        required=True,
        nested=[ArticulationRig],
        default_factory=list,
    )


# ------------------------------------------------------------- the GLB block
@dataclass
class ArticulationBlockJoint(ArticulationJointPose):
    """A joint bound to the file: its pose, and the glTF node that carried
    its name (whose own name it is)."""

    node: int = _field(help="The joint's glTF node index.", required=True)


@dataclass
class ArticulationBlockGrab(SchemaSpec):
    """A grabbed part bound to its glTF node."""

    TYPED = True

    node: int = _field(
        help="The part's glTF node index: the one of its name under its joint's node.",
        required=True,
    )
    joint: int = _field(help="The index of the joint the part rides.", required=True)


@dataclass
class ArticulationBlockRig(ArticulationRig):
    """One rig bound to the file: its joints and grabbed parts carry node indices."""

    joints: List[ArticulationBlockJoint] = _field(
        help="Its joints, each after its parent.",
        required=True,
        nested=[ArticulationBlockJoint],
        default_factory=list,
    )
    grab: List[ArticulationBlockGrab] = _field(
        help="The parts a hand takes hold of that resolved to a node.",
        required=True,
        nested=[ArticulationBlockGrab],
        default_factory=list,
    )


@dataclass
class ArticulationBlock(SchemaSpec):
    """The ``extras.articulation`` block: the record's rigs, bound to the
    file's glTF nodes."""

    TYPED = True

    version: int = _field(
        help="The block's own schema version (SceneRecords.ARTICULATION.glb.version).",
        required=True,
    )
    rigs: List[ArticulationBlockRig] = _field(
        help="The rigs whose every joint is in the file.",
        required=True,
        nested=[ArticulationBlockRig],
        default_factory=list,
    )


@Deprecation.symbol(
    "ArticulationBlock -- the block is named for the file it rides",
    remove_in="0.18.0",
    since="2026-10-24",
)
class ArticulationWeb(ArticulationBlock):
    """Retired 2026-10-20 (removed in 0.18.0): :class:`ArticulationBlock`."""


@Deprecation.symbol("ArticulationBlockRig", remove_in="0.18.0", since="2026-10-24")
class ArticulationWebRig(ArticulationBlockRig):
    """Retired 2026-10-20 (removed in 0.18.0): :class:`ArticulationBlockRig`."""


@Deprecation.symbol("ArticulationBlockJoint", remove_in="0.18.0", since="2026-10-24")
class ArticulationWebJoint(ArticulationBlockJoint):
    """Retired 2026-10-20 (removed in 0.18.0): :class:`ArticulationBlockJoint`."""


@Deprecation.symbol("ArticulationBlockGrab", remove_in="0.18.0", since="2026-10-24")
class ArticulationWebGrab(ArticulationBlockGrab):
    """Retired 2026-10-20 (removed in 0.18.0): :class:`ArticulationBlockGrab`."""
