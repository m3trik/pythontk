# !/usr/bin/python
# coding=utf-8
"""The ``articulation`` record's payload and its web projection, declared once.

:attr:`pythontk.SceneRecords.ARTICULATION` names these as its shapes:

- :class:`ArticulationRecord` -- what the DCC's ``ArticulatedRig`` publishes on
  the ``data_export`` carrier, and what every runtime builds its
  :class:`~pythontk.ArticulationModel` from (one rig at a time).
- :class:`ArticulationWeb` -- the ``extras.articulation_web`` manifest
  ``MeshConvert.apply_glb_articulation`` derives from it for the glTF
  runtimes: the same rigs, each joint and grabbed part bound to its glTF node.

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
            "slide along them (the parent's units)."
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
class ArticulationJoint(SchemaSpec):
    """One joint: its node, its rest frame in its parent's space, its channels."""

    TYPED = True

    name: str = _field(
        help="The joint node's name: the join key to the imported transform.",
        required=True,
    )
    parent: Optional[int] = _field(
        help="An earlier joint's index, or null for a root (its parent is the rig's space).",
        required=True,
    )
    t: Tuple[float, float, float] = _field(
        help="The rest translate in the parent's space, in the DCC's units.",
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


# --------------------------------------------------------- the web projection
@dataclass
class ArticulationWebJoint(ArticulationJoint):
    """A joint bound to the one glTF node carrying its name."""

    node: int = _field(help="The joint's glTF node index.", required=True)


@dataclass
class ArticulationWebGrab(SchemaSpec):
    """A grabbed part bound to its glTF node."""

    TYPED = True

    node: int = _field(
        help="The part's glTF node index: the one of its name under its joint's node.",
        required=True,
    )
    joint: int = _field(help="The index of the joint the part rides.", required=True)
    name: str = _field(
        help="The part's node name, as the record spells it.", required=True
    )


@dataclass
class ArticulationWebRig(ArticulationRig):
    """One rig bound to the file: its joints and grabbed parts carry node indices."""

    joints: List[ArticulationWebJoint] = _field(
        help="Its joints, each after its parent.",
        required=True,
        nested=[ArticulationWebJoint],
        default_factory=list,
    )
    grab: List[ArticulationWebGrab] = _field(
        help="The parts a hand takes hold of that resolved to a node.",
        required=True,
        nested=[ArticulationWebGrab],
        default_factory=list,
    )


@dataclass
class ArticulationWeb(SchemaSpec):
    """The ``extras.articulation_web`` manifest: the record's rigs, bound to the file's glTF nodes."""

    TYPED = True

    version: int = _field(
        help="The manifest's own schema version (SceneRecords.ARTICULATION.web.version).",
        required=True,
    )
    metadata_version: int = _field(
        help="The version of the articulation record it was bound from.",
        required=True,
    )
    rigs: List[ArticulationWebRig] = _field(
        help="The rigs whose every joint is in the file.",
        required=True,
        nested=[ArticulationWebRig],
        default_factory=list,
    )
