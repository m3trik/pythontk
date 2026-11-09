# !/usr/bin/python
# coding=utf-8
"""The ``articulation`` record's payload, declared once.

:attr:`pythontk.SceneRecords.ARTICULATION` names :class:`ArticulationRecord`
as its shape, in both containers (one layout, two containers): what a DCC's
``ArticulatedRig`` publishes on the ``data_export`` carrier, the record a GLB
carries as it stands in its root ``extras``, and what every runtime builds its
:class:`~pythontk.ArticulationModel` from (one rig at a time). Version 2 states
it in glTF's frame (:attr:`pythontk.SceneRecords.FRAME`: metres, quaternions
``(x, y, z, w)``) -- a length is the FBX's own value x 0.01 whatever its
UnitScaleFactor, as FBX2glTF reads it, so a GLB node's local is the record's
exactly, and an FBX importer that converts the file's unit into the locals
scales it by that conversion (measured 2026-11-08) -- and gives every joint an anchor frame
(:class:`ArticulationAnchor`), the frame its channels turn and slide in.

It is a typed :class:`~pythontk.SchemaSpec` declaration, so this one
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
            "rx/ry/rz turn about the anchor's axes through its point "
            "(degrees); tx/ty/tz slide along them (metres, as 't' is)."
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
class ArticulationAnchor(SchemaSpec):
    """The frame a joint's channels turn and slide in: a point and an
    orientation in the joint's REST frame. The identity -- the rest frame
    itself -- is a joint whose part nests on its pivot.

    Prior art: USD Physics joints' ``localPos1`` / ``localRot1`` and a
    game engine's ``ArticulationBody`` ``anchorPosition`` / ``anchorRotation`` -- the
    child body's joint frame, as here; glTF's ``KHR_physics_rigid_bodies``
    states it as a node, and FBX has a pivot point but no pivot frame (and
    FBX2glTF bakes it away). The weighing is :class:`~pythontk.ArticulationModel`'s."""

    TYPED = True
    #: The identity anchor, as a record states it (read it through
    #: :meth:`identity`, a copy of its own). A class constant, never a field:
    #: it carries no annotation.
    IDENTITY = {"t": (0.0, 0.0, 0.0), "q": (0.0, 0.0, 0.0, 1.0)}

    t: Tuple[float, float, float] = _field(
        help="The point the joint turns about, in its rest frame (metres, as 't' is).",
        required=True,
        default_factory=lambda: list(ArticulationAnchor.IDENTITY["t"]),
    )
    q: Tuple[float, float, float, float] = _field(
        help=(
            "The orientation of the axes its channels turn about and slide "
            "along, in its rest frame: a quaternion x, y, z, w."
        ),
        required=True,
        default_factory=lambda: list(ArticulationAnchor.IDENTITY["q"]),
    )

    @classmethod
    def identity(cls) -> dict:
        """The identity anchor as a record writes it: ``{"t": [0, 0, 0],
        "q": [0, 0, 0, 1]}``, a fresh copy."""
        return {key: list(value) for key, value in cls.IDENTITY.items()}


@dataclass
class ArticulationJointPose(SchemaSpec):
    """A joint's rest frame in its parent joint's frame, its anchor and its
    channels."""

    TYPED = True

    parent: Optional[int] = _field(
        help=(
            "An earlier joint's index, or null for a root (its parent is the "
            "rig's space): the joint chain a reader poses the part from, "
            "whatever node its part hangs under."
        ),
        required=True,
    )
    t: Tuple[float, float, float] = _field(
        help=(
            "The rest translate in the parent joint's frame, in metres as "
            "glTF frames the file: an FBX's own value x 0.01 whatever its "
            "UnitScaleFactor, a GLB node's local exactly (a scale a node "
            "above carries scales it). A reader whose import converts the "
            "file's unit into the locals (an FBX importer applying its "
            "UnitScaleFactor) scales it by that conversion x 100."
        ),
        required=True,
    )
    q: Tuple[float, float, float, float] = _field(
        help=(
            "The rest rotation in the parent joint's frame (the DCC's joint "
            "orient), a quaternion x, y, z, w."
        ),
        required=True,
    )
    anchor: ArticulationAnchor = _field(
        help=(
            "The frame the channels turn and slide in, in the rest frame: the "
            "part poses as rest . anchor . pose(channels) . anchor^-1 (the "
            "identity: rest . pose(channels))."
        ),
        required=True,
        nested=ArticulationAnchor,
        default_factory=ArticulationAnchor.identity,
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
        help=(
            "The record's schema version (SceneRecords.ARTICULATION.version); "
            "a reader reads the version it knows and refuses any other -- a "
            "version 1 record (no anchor, lengths in the file's own units) is "
            "re-exported."
        ),
        required=True,
    )
    rigs: List[ArticulationRig] = _field(
        help="The scene's articulated rigs.",
        required=True,
        nested=[ArticulationRig],
        default_factory=list,
    )



# Held until 0.22.0 (``pythontk/__init__.py``, "Held until 0.22.0"): the block
# names v0.20.0 noticed for removal, whose window closes 2026-11-23.
from pythontk.geo_utils.articulation._retired import (  # noqa: E402
    ArticulationWeb,
    ArticulationWebGrab,
    ArticulationWebJoint,
    ArticulationWebRig,
)
