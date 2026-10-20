# !/usr/bin/python
# coding=utf-8
"""An articulated rig's PLAN -- what a DCC builds, edited as plain data.

The plan is the JSON a DCC stamps on its rig group (mayatk's and blendertk's
``ArticulatedRig``), in the group's own space::

    {"version": 1, "name": "magnifier",
     "links": [[{"uuid": ..., "name": "BASE"}], ...],
     "joints": [{"id": "LEG_1", "link": 1, "parent": 0, "type": "universal",
                 "position": [x, y, z], "aim": [...], "normal": [...],
                 "limits": {"rz": [min, max]}, "weights": {"rz": 1.0},
                 "size": float, "reason": str}, ...],
     "end": {"part": uuid, "frame": [16 floats]}}

plus what only the host knows (``group_uuid``, the parts' captured
``attrs``). Link 0 is the fixed root; each other link has exactly one joint,
its rest frame X along ``aim``, Z the ``normal`` made perpendicular to it.

Everything here is the host-free half of the rig: checking and ordering a
plan, the joints' rest frames and the ``articulation`` record they ship
(:class:`~pythontk.geo_utils.articulation.model.ArticulationModel` poses
them), the end control's placement and offset, the post-rig edits (a joint
inserted, removed, edited, a limit or a weight set), the pivots put out for
adjusting and read back, and the panel's text for limits and give. A host
measures (node paths, geometry, its matrices) and builds; it passes plain
values in and gets plain values out.

Matrices are 16 floats, row by row, for ROW vectors (``p' = p M``): rows 0-2
are a frame's X, Y and Z axes and row 3 its position -- Maya's ``MMatrix``
layout and the layout the plan stamps (``end.frame``). ``a`` then ``b`` is
``a M_a M_b``. A matrix is a ``Sequence[float]`` in, a ``List[float]`` out.
"""

from __future__ import annotations

import copy
import math
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from pythontk.geo_utils.articulation.analysis import ArticulationAnalysis
from pythontk.geo_utils.articulation.model import _ArticulationModelInternal

Vec = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]
Matrix = List[float]

_IDENTITY: Tuple[float, ...] = tuple(float(r == c) for r in range(4) for c in range(4))


class _ArticulationPlanInternal(_ArticulationModelInternal):
    """Row-vector 4 x 4 matrix helpers over the model's vector, quaternion
    and Euler ones."""

    #: A length below this is no length: an aim, or a normal left along it.
    _DEGENERATE = 1.0e-9
    #: Below this cosine of its middle angle an xyz Euler reading is gimbal
    #: locked (Maya's threshold, measured: 1.00e-5 locks, 1.01e-5 does not).
    _GIMBAL = 1.0e-5

    @staticmethod
    def _mul(a: Sequence[float], b: Sequence[float]) -> Matrix:
        """``a`` then ``b``: the matrix product ``a b`` (row vectors)."""
        return [
            a[r * 4] * b[c]
            + a[r * 4 + 1] * b[4 + c]
            + a[r * 4 + 2] * b[8 + c]
            + a[r * 4 + 3] * b[12 + c]
            for r in range(4)
            for c in range(4)
        ]

    @staticmethod
    def _inverse(m: Sequence[float]) -> Matrix:
        """The inverse of the affine matrix *m* (any scale or shear)."""
        a, b, c = m[0], m[1], m[2]
        d, e, f = m[4], m[5], m[6]
        g, h, k = m[8], m[9], m[10]
        c00, c01, c02 = e * k - f * h, c * h - b * k, b * f - c * e
        c10, c11, c12 = f * g - d * k, a * k - c * g, c * d - a * f
        c20, c21, c22 = d * h - e * g, b * g - a * h, a * e - b * d
        inv = 1.0 / (a * c00 + d * c01 + g * c02)
        r = [
            [c00 * inv, c01 * inv, c02 * inv],
            [c10 * inv, c11 * inv, c12 * inv],
            [c20 * inv, c21 * inv, c22 * inv],
        ]
        tx, ty, tz = m[12], m[13], m[14]
        t = [-(tx * r[0][i] + ty * r[1][i] + tz * r[2][i]) for i in range(3)]
        return [*r[0], 0.0, *r[1], 0.0, *r[2], 0.0, *t, 1.0]

    @staticmethod
    def _point(p: Sequence[float], m: Sequence[float]) -> List[float]:
        """The point *p* through the matrix *m*."""
        return [
            p[0] * m[c] + p[1] * m[4 + c] + p[2] * m[8 + c] + m[12 + c]
            for c in range(3)
        ]

    @staticmethod
    def _vector(v: Sequence[float], m: Sequence[float]) -> List[float]:
        """The direction *v* through the matrix *m* (no translation)."""
        return [v[0] * m[c] + v[1] * m[4 + c] + v[2] * m[8 + c] for c in range(3)]

    @classmethod
    def _unit(cls, v: Sequence[float]) -> List[float]:
        """*v* at length 1; a zero vector stays zero (the frame refuses it)."""
        n = cls._norm(v)
        return [float(c) for c in v] if n == 0.0 else [c / n for c in v]

    @classmethod
    def _decompose(
        cls, m: Sequence[float]
    ) -> Tuple[List[float], List[float], List[float]]:
        """``(translation, joint orient xyz degrees, rotation xyzw)`` of a
        rigid matrix, read as Maya's ``MTransformationMatrix`` reads it.

        Maya's Euler (``MEulerRotation.decompose``, measured on 20,000
        random rotations, 2026-10-16): the xyz triple whose middle angle is
        in [-90, 90] -- at gimbal lock (``cos y`` under :attr:`_GIMBAL`) z is
        0 and x takes the whole turn -- unless the triple half a turn round
        each axis from it, ``(x + 180, 180 - y, z + 180)`` unwrapped, sums
        smaller in magnitude. Its quaternion is the first triple's,
        composed (:meth:`_euler_quat`), whichever triple it reports.
        """
        cos_y = math.hypot(m[0], m[1])
        y = math.atan2(-m[2], cos_y)
        if cos_y < cls._GIMBAL:
            x, z = math.atan2(-m[9], m[5]), 0.0
        else:
            x, z = math.atan2(m[6], m[10]), math.atan2(m[1], m[0])
        first = [math.degrees(x), math.degrees(y), math.degrees(z)]
        q = cls._euler_quat("xyz", {"rx": first[0], "ry": first[1], "rz": first[2]})
        flipped = [first[0] + 180.0, 180.0 - first[1], first[2] + 180.0]
        if sum(abs(v) for v in flipped) < sum(abs(v) for v in first):
            first = flipped
        return [m[12], m[13], m[14]], first, list(q)

    @staticmethod
    def _world_limits(joint: Mapping[str, Any], scale: float) -> Dict[str, Any]:
        """A planned joint's limits in world units: a slide's travel grows by
        *scale* (what the group's space does to a length)."""
        out = {}
        for channel, (lo, hi) in (joint.get("limits") or {}).items():
            if channel.startswith("t"):
                lo = None if lo is None else lo * scale
                hi = None if hi is None else hi * scale
            out[channel] = [lo, hi]
        return out


class ArticulationPlan(_ArticulationPlanInternal):
    """The plan of an articulated rig: checked, framed, recorded and edited.

    Every method takes the plan (a ``spec`` dict, the shape in this module's
    docstring) or a part of it, and plain values; none touches a scene. The
    ones that edit a plan edit it IN PLACE (pass a copy to keep the original)
    and return what they changed.
    """

    #: The plan's schema.
    VERSION = 1
    #: What each joint type moves and in which rotate order
    #: (``ArticulationAnalysis.JOINT_TYPES``).
    JOINT_TYPES = ArticulationAnalysis.JOINT_TYPES
    #: The key of the end control in what is keyed by joint id -- the
    #: pivots put out for adjusting, a host's parked animation (a joint id is
    #: a node name, which never holds ``<``).
    END_KEY = "<end>"
    #: A pivot moved by less than this moved nothing (a length in the rig's
    #: units, or an axis component).
    ADJUST_EPS = 1.0e-6

    # ================================================================ frames
    @classmethod
    def frame(
        cls, position: Sequence[float], aim: Sequence[float], normal: Sequence[float]
    ) -> Matrix:
        """A rest frame: X along *aim*, Z the *normal* made perpendicular to
        it, Y completing a right-handed set, at *position*.

        Raises:
            ValueError: *aim* has no length, or *normal* runs along it (no
                hinge axis left).
        """
        n = cls._norm(aim)
        if n < cls._DEGENERATE:
            raise ValueError(f"Joint at {list(position)} has no aim.")
        x = [float(c) / n for c in aim]
        z = [float(c) for c in normal]
        along = cls._dot(z, x)
        z = [zc - xc * along for zc, xc in zip(z, x)]
        n = cls._norm(z)
        if n < cls._DEGENERATE:
            raise ValueError(
                f"Joint normal {list(normal)} runs along its aim {list(aim)}."
            )
        z = [c / n for c in z]
        y = cls._cross(z, x)
        return [
            *x,
            0.0,
            *y,
            0.0,
            *z,
            0.0,
            float(position[0]),
            float(position[1]),
            float(position[2]),
            1.0,
        ]

    @classmethod
    def frames(cls, spec: Mapping[str, Any]) -> Dict[int, Matrix]:
        """Each link's rest frame in the group's space, by link index: its
        joint's (the fixed root link's is the group's own, the identity)."""
        frames = {0: list(_IDENTITY)}
        for joint in spec["joints"]:
            frames[joint["link"]] = cls.frame(
                joint["position"], joint["aim"], joint["normal"]
            )
        return frames

    @classmethod
    def rest(cls, spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Each planned joint's rest in its parent's space, in plan order:
        ``{"t", "jo" (degrees, xyz), "q" (xyzw), "parent" (index or None),
        "axes" (its X, Y and Z in the parent's space -- the lines a slide
        travels)}`` -- what a build sets and the record ships, from one
        computation.

        Raises:
            ValueError: A joint's frame is degenerate (:meth:`frame`).
        """
        frames = cls.frames(spec)
        index: Dict[int, int] = {}
        out = []
        for i, joint in enumerate(spec["joints"]):
            local = cls._mul(
                frames[joint["link"]], cls._inverse(frames[joint["parent"]])
            )
            t, jo, q = cls._decompose(local)
            out.append(
                {
                    "t": t,
                    "jo": jo,
                    "q": q,
                    "parent": index.get(joint["parent"]),
                    "axes": [local[0:3], local[4:7], local[8:11]],
                }
            )
            index[joint["link"]] = i
        return out

    # ================================================================ record
    @classmethod
    def joint_records(cls, spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """The record's joints: each one's rest (translate and rotation in its
        parent's space), rotate order and channels with their limits and
        weights, in plan order; ``parent`` an index into them. Named
        ``<rig>_<id>_jnt``, the node a host builds for it."""
        joints = []
        for joint, pose in zip(spec.get("joints", []), cls.rest(spec)):
            channels, order = cls.JOINT_TYPES[joint["type"]]
            limits = joint.get("limits") or {}
            weights = joint.get("weights") or {}
            joints.append(
                {
                    "name": f"{spec['name']}_{joint['id']}_jnt",
                    "parent": pose["parent"],
                    "t": pose["t"],
                    "q": pose["q"],
                    "rotate_order": order,
                    "channels": [
                        {
                            "channel": channel,
                            "min": (limits.get(channel) or (None, None))[0],
                            "max": (limits.get(channel) or (None, None))[1],
                            "weight": float(weights.get(channel, 1.0)),
                        }
                        for channel in channels
                    ],
                }
            )
        return joints

    @classmethod
    def record(
        cls,
        spec: Mapping[str, Any],
        name_of: Optional[Callable[[Mapping[str, Any]], str]] = None,
    ) -> Dict[str, Any]:
        """The rig's entry in the ``articulation`` record: its joints
        (:meth:`joint_records`) and the parts a hand grabs.

        Parameters:
            spec: The plan.
            name_of: A part's name as the scene has it NOW, from its link
                member (``{"uuid", "name"}``) -- a runtime binds a grab by
                name, so a part renamed after the build must ship its new
                one. Default: the name the plan stamped.
        """
        name_of = name_of or (lambda member: member["name"])
        grab = []
        for index, joint in enumerate(spec["joints"]):
            grab += [
                {"node": name_of(m), "joint": index}
                for m in spec["links"][joint["link"]]
            ]
        return {"name": spec["name"], "joints": cls.joint_records(spec), "grab": grab}

    @staticmethod
    def riders(spec: Mapping[str, Any]) -> Dict[str, int]:
        """The joint index each moving part rides, by the part's uuid (the
        fixed root link's parts ride none)."""
        return {
            m["uuid"]: i
            for i, joint in enumerate(spec["joints"])
            for m in spec["links"][joint["link"]]
        }

    @classmethod
    def slots(cls, spec: Mapping[str, Any]) -> List[Tuple[str, str]]:
        """``(joint id, channel)`` per state slot, in the model's order."""
        return [
            (joint["id"], channel)
            for joint in spec.get("joints", [])
            for channel in cls.JOINT_TYPES[joint["type"]][0]
        ]

    # ================================================================= build
    @staticmethod
    def from_proposal(
        proposal: Mapping[str, Any], links: Sequence[Any], control_scale: float
    ) -> Dict[str, Any]:
        """A plan to build from an ``ArticulationAnalysis.propose`` result.

        Parameters:
            proposal: The analysis of *links* (one entry per link, in order).
            links: What the host calls each link (its nodes), in the order
                the analysis was given them.
            control_scale: A control's size over its joint's body radius.

        Returns:
            ``{"links": [...], "joints": [{"link", "parent", "type",
            "position", "aim", "normal", "limits", "size", "reason"}, ...],
            "unreached": [...]}``: *links* in the proposal's order (link 0
            the root), each joint's ``link`` / ``parent`` an index into them,
            the links nothing reached by their first member.
        """
        at = {part: i for i, part in enumerate(proposal["order"])}
        joints = []
        for joint in proposal["joints"]:
            joints.append(
                {
                    "link": at[joint["part"]],
                    "parent": at[joint["parent"]],
                    "type": joint["type"],
                    "position": joint["position"],
                    "aim": joint["aim"],
                    "normal": joint["normal"],
                    "limits": joint["limits"],
                    "size": control_scale * joint["radius"],
                    "reason": joint["reason"],
                }
            )
        return {
            "links": [list(links[i]) for i in proposal["order"]],
            "joints": joints,
            "unreached": [links[i][0] for i in proposal["unreached"]],
        }

    @classmethod
    def plan_joint(
        cls, joint: Mapping[str, Any], joint_id: str, inverse: Sequence[float]
    ) -> Dict[str, Any]:
        """One joint, as a caller states it in WORLD space, moved into the rig
        group's space (*inverse*: world to group). A slide's travel and a
        control's size scale by what that does to a length (the prop's own
        scale).

        Parameters:
            joint: ``{"link", "parent", "type", "position", "aim", "normal",
                "limits", "weights", "size", "reason"}`` (``type`` default
                hinge; ``limits`` / ``weights`` keyed by channel, a channel
                the type lacks dropped).
            joint_id: The joint's id (a host names it after its link's first
                part).
            inverse: The world-to-group matrix.

        Raises:
            ValueError: An unknown or fixed joint type.
        """
        kind = joint.get("type", "hinge")
        if kind not in cls.JOINT_TYPES or kind == "fixed":
            raise ValueError(f"Unknown or fixed joint type {kind!r}.")
        channels, _order = cls.JOINT_TYPES[kind]
        position = cls._point(joint["position"], inverse)
        aim = cls._vector(cls._unit(joint["aim"]), inverse)
        normal = cls._vector(joint["normal"], inverse)
        scale = cls._norm(aim)
        limits = {}
        for channel in channels:
            bounds = (joint.get("limits") or {}).get(channel)
            if bounds is None:
                continue
            lo, hi = bounds
            if channel.startswith("t"):
                lo = None if lo is None else lo * scale
                hi = None if hi is None else hi * scale
            limits[channel] = [lo, hi]
        return {
            "id": joint_id,
            "link": int(joint["link"]),
            "parent": int(joint["parent"]),
            "type": kind,
            "position": position,
            "aim": cls._unit(aim),
            "normal": cls._unit(normal),
            "limits": limits,
            "weights": {
                k: float(v)
                for k, v in (joint.get("weights") or {}).items()
                if k in channels
            },
            "size": float(joint.get("size") or 0.0) * scale,
            "reason": str(joint.get("reason") or "as planned"),
        }

    @classmethod
    def check(
        cls,
        spec: Dict[str, Any],
        parts: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Refuse a plan the build cannot keep: a part named twice, a link
        without exactly one joint, a joint the root does not reach, a
        degenerate frame. Orders the joints parent first and numbers a
        repeated id, in place.

        Parameters:
            spec: The plan.
            parts: The host's check of the parts themselves (gone from the
                scene, driven by something), run once the plan names each
                part once; it raises ValueError to refuse.

        Raises:
            ValueError: The plan cannot be built.
        """
        uuids = [m["uuid"] for link in spec["links"] for m in link]
        if len(set(uuids)) != len(uuids):
            raise ValueError("A part is named in two links.")
        if parts is not None:
            parts(spec)
        # Ids unique in the caller's order: a joint already stamped (listed
        # first by every edit) keeps its id -- the key its parked animation is
        # filed under -- and a newcomer repeating a part name is numbered.
        seen: set = set()
        for joint in spec["joints"]:
            stem, n = joint["id"], joint["link"]
            while joint["id"] in seen:
                joint["id"] = f"{stem}_{n}"
                n += 1
            seen.add(joint["id"])
        count = len(spec["links"])
        by_link: Dict[int, Dict[str, Any]] = {}
        for joint in spec["joints"]:
            if not 0 < joint["link"] < count or joint["link"] in by_link:
                raise ValueError(f"Link {joint['link']} has no single joint slot.")
            by_link[joint["link"]] = joint
        missing = [i for i in range(1, count) if i not in by_link]
        if missing:
            raise ValueError(f"Links {missing} have no joint.")
        ordered: List[Dict[str, Any]] = []
        placed = {0}
        while len(ordered) < len(by_link):
            ready = sorted(
                (
                    j
                    for j in by_link.values()
                    if j not in ordered and j["parent"] in placed
                ),
                key=lambda j: j["link"],
            )
            if not ready:
                raise ValueError("The joints do not join every link to the root.")
            for joint in ready:
                ordered.append(joint)
                placed.add(joint["link"])
        spec["joints"] = ordered
        cls.rest(spec)  # a frame with no aim, or a normal along it, fails HERE

    # =========================================================== end control
    @staticmethod
    def end_joint(spec: Mapping[str, Any]) -> Optional[int]:
        """The index of the joint the end control holds -- the joint of the
        link its part rides -- or None (no end control, or its part is in
        the fixed root link)."""
        end = spec.get("end")
        if not end:
            return None
        for index, joint in enumerate(spec.get("joints", [])):
            if any(m["uuid"] == end["part"] for m in spec["links"][joint["link"]]):
                return index
        return None

    @staticmethod
    def link_of(spec: Mapping[str, Any], uuid: Optional[str]) -> Optional[int]:
        """The index of the link holding the part *uuid*, or None."""
        return next(
            (
                i
                for i, members in enumerate(spec["links"])
                if any(m["uuid"] == uuid for m in members)
            ),
            None,
        )

    @staticmethod
    def end_link(spec: Mapping[str, Any]) -> Optional[int]:
        """The link at the end of the plan's longest chain (the deepest leaf;
        the highest index among equals) -- where an end control goes by
        default -- or None (no joints)."""
        joints = {j["link"]: j for j in spec["joints"]}

        def depth(i: int) -> int:
            return 0 if i not in joints else 1 + depth(joints[i]["parent"])

        parents = {j["parent"] for j in spec["joints"]}
        leaves = [i for i in joints if i not in parents]
        return max(leaves, key=lambda i: (depth(i), i)) if leaves else None

    @classmethod
    def default_end(
        cls,
        spec: Mapping[str, Any],
        inverse: Sequence[float],
        points: Callable[[int], Iterable[Sequence[float]]],
        link: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Where the end control goes on a plan AT REST: on *link* (default
        :meth:`end_link`), aligned with that link's joint and centred on its
        geometry (the magnifier's lens).

        Parameters:
            spec: The plan.
            inverse: The world-to-group matrix.
            points: A link's geometry, world space, from its index (asked
                only for the link taken).
            link: The link to take; None takes :meth:`end_link`.

        Returns:
            ``{"part": uuid, "frame": [16 floats]}``, the frame in group
            space; None for the fixed root link (nothing to solve).
        """
        joints = {j["link"]: j for j in spec["joints"]}
        if link is None:
            link = cls.end_link(spec)
        joint = joints.get(link)
        if joint is None:
            return None
        frame = cls.frame(joint["position"], joint["aim"], joint["normal"])
        to_frame = cls._mul(inverse, cls._inverse(frame))
        local = [cls._point(p, to_frame) for p in points(link)]
        centre = [
            0.5 * (min(p[i] for p in local) + max(p[i] for p in local))
            for i in range(3)
        ]
        placed = list(frame)
        placed[12:15] = cls._point(centre, frame)
        return {"part": spec["links"][link][0]["uuid"], "frame": placed}

    @classmethod
    def end_offset(cls, spec: Mapping[str, Any], end_index: int) -> Matrix:
        """The end link's rest frame in the end control's rest frame: the
        link rides the control as it did at rest (``link = offset *
        control``)."""
        joint = spec["joints"][end_index]
        link = cls.frame(joint["position"], joint["aim"], joint["normal"])
        return cls._mul(link, cls._inverse(spec["end"]["frame"]))

    @classmethod
    def end_pivot_turn(
        cls, spec: Mapping[str, Any], end_index: int
    ) -> Tuple[Vec, Quat]:
        """What a solve of the end control needs of :meth:`end_offset`: the
        control's pivot in the end link's frame, and the link's turn in the
        control's frame (``(x, y, z, w)``)."""
        offset = cls.end_offset(spec, end_index)
        _t, _jo, turn = cls._decompose(offset)
        pivot = cls._inverse(offset)[12:15]
        return (pivot[0], pivot[1], pivot[2]), (turn[0], turn[1], turn[2], turn[3])

    # ================================================================= edits
    @staticmethod
    def joint_entry(spec: Mapping[str, Any], joint_id: str) -> Dict[str, Any]:
        """The plan's joint *joint_id* (the dict itself: edits land in the
        plan).

        Raises:
            KeyError: No such joint.
        """
        joint = next((j for j in spec.get("joints", []) if j["id"] == joint_id), None)
        if joint is None:
            raise KeyError(f"No joint {joint_id!r} in rig {spec.get('name')!r}.")
        return joint

    @staticmethod
    def split_link(
        spec: Mapping[str, Any], uuids: Iterable[str]
    ) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """``(link, kept members, split members)`` for parts *uuids* leaving
        the one link they share.

        Raises:
            ValueError: *uuids* are not all in one link, or are the whole of
                it.
        """
        moved = set(uuids)
        host = next(
            (
                i
                for i, link in enumerate(spec["links"])
                if moved <= {m["uuid"] for m in link}
            ),
            None,
        )
        if host is None or not moved:
            raise ValueError("The members to split off are not all in one link.")
        keep = [m for m in spec["links"][host] if m["uuid"] not in moved]
        split = [m for m in spec["links"][host] if m["uuid"] in moved]
        if not keep:
            raise ValueError("Splitting off a whole link adds no joint.")
        return host, keep, split

    @classmethod
    def insert_joint(
        cls,
        spec: Dict[str, Any],
        uuids: Iterable[str],
        joint: Dict[str, Any],
        world: Sequence[float],
        keep_points: Iterable[Sequence[float]],
        split_points: Iterable[Sequence[float]],
    ) -> Dict[str, Any]:
        """Split a link: parts *uuids* leave the link they ride
        (:meth:`split_link`) for a new one, last, on *joint*. The joints that
        hung off the split link and sit nearer the split parts than the kept
        ones move onto the new joint, so the chain beyond rides it.

        Parameters:
            spec: The plan, edited in place.
            uuids: The parts to split off.
            joint: The new joint, planned (:meth:`plan_joint`) on the new
                link (``len(spec["links"])``) hanging off the split one.
            world: The group-to-world matrix (the joints' positions are
                measured against the parts' world geometry).
            keep_points: The kept parts' geometry, world space.
            split_points: The split parts' geometry, world space.

        Returns:
            *spec*.

        Raises:
            ValueError: As :meth:`split_link`.
        """
        host, keep, split = cls.split_link(spec, uuids)
        new_link = len(spec["links"])
        near_keep = [tuple(p) for p in keep_points]
        near_split = [tuple(p) for p in split_points]
        for child in spec["joints"]:
            if child["parent"] != host:
                continue
            p = cls._point(child["position"], world)
            d_keep = min(math.dist(p, q) for q in near_keep)
            d_split = min(math.dist(p, q) for q in near_split)
            if d_split < d_keep:
                child["parent"] = new_link
        spec["links"][host] = keep
        spec["links"].append(split)
        spec["joints"].append(joint)
        return spec

    @classmethod
    def remove_joint(cls, spec: Dict[str, Any], joint_id: str) -> Dict[str, Any]:
        """Fold *joint_id*'s link back into the link it hangs off: its parts
        ride that joint again and the joints beyond it move up with them.

        Returns:
            *spec*, edited in place.

        Raises:
            KeyError: No such joint.
        """
        joint = cls.joint_entry(spec, joint_id)
        gone, into = joint["link"], joint["parent"]
        spec["links"][into] += spec["links"][gone]
        del spec["links"][gone]

        def renumber(link: int) -> int:
            link = into if link == gone else link
            return link - 1 if link > gone else link

        joints = []
        for other in spec["joints"]:
            if other is joint:
                continue
            other["parent"] = renumber(other["parent"])
            other["link"] = renumber(other["link"])
            joints.append(other)
        spec["joints"] = joints
        return spec

    @classmethod
    def edit_joint(
        cls,
        spec: Dict[str, Any],
        joint_id: str,
        world: Sequence[float],
        **fields: Any,
    ) -> Dict[str, Any]:
        """Change a joint's ``type``, ``position``, ``aim`` or ``normal``
        (world space), or its ``limits`` / ``weights`` (any of
        :meth:`plan_joint`'s fields; None leaves one as it is). The joint keeps
        its id.

        Parameters:
            spec: The plan, edited in place.
            joint_id: The joint.
            world: The group-to-world matrix.

        Returns:
            *spec*.

        Raises:
            KeyError: No such joint.
            ValueError: An unknown or fixed joint type.
        """
        joint = cls.joint_entry(spec, joint_id)
        aim = cls._vector(joint["aim"], world)
        current = {
            "type": joint["type"],
            "link": joint["link"],
            "parent": joint["parent"],
            "position": cls._point(joint["position"], world),
            "aim": aim,
            "normal": cls._vector(joint["normal"], world),
            "limits": cls._world_limits(joint, cls._norm(aim)),
            "weights": joint.get("weights"),
            "size": joint.get("size", 0.0) * cls._norm(aim),
            "reason": joint.get("reason"),
        }
        current.update({k: v for k, v in fields.items() if v is not None})
        planned = cls.plan_joint(current, joint["id"], cls._inverse(world))
        spec["joints"][spec["joints"].index(joint)] = planned
        return spec

    @classmethod
    def set_limits(
        cls,
        spec: Mapping[str, Any],
        joint_id: str,
        channel: str,
        minimum: Optional[float],
        maximum: Optional[float],
    ) -> Dict[str, Any]:
        """Bound one channel (degrees, or a slide's units in the rig's
        space); None frees that side.

        Returns:
            The joint, edited in place.

        Raises:
            KeyError: No such joint, or no such channel on it.
            ValueError: *minimum* above *maximum*.
        """
        joint = cls.joint_entry(spec, joint_id)
        if channel not in cls.JOINT_TYPES[joint["type"]][0]:
            raise KeyError(f"No channel {channel!r} on joint {joint_id!r}.")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"Limits [{minimum}, {maximum}] are reversed.")
        joint.setdefault("limits", {})[channel] = [minimum, maximum]
        return joint

    @classmethod
    def side_limits(
        cls,
        spec: Mapping[str, Any],
        joint_id: str,
        channel: str,
        side: str,
        value: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """The channel's ``(min, max)`` with *value* taken as its *side*
        (``"min"`` or ``"max"``) -- a limit read off the pose -- the other
        side as it stands.

        Raises:
            KeyError: No such joint.
            ValueError: *side* is neither.
        """
        joint = cls.joint_entry(spec, joint_id)
        lo, hi = (joint.get("limits") or {}).get(channel) or (None, None)
        if side == "min":
            lo = value
        elif side == "max":
            hi = value
        else:
            raise ValueError(f"side must be 'min' or 'max', not {side!r}.")
        return lo, hi

    @classmethod
    def set_weight(
        cls,
        spec: Mapping[str, Any],
        joint_id: str,
        weight: float,
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """How readily *joint_id* moves when the end control or a grab pulls
        the rig, against the other joints: 2 twice as readily as 1, 0 not at
        all -- every channel of the joint, or just *channel*.

        Returns:
            The joint, edited in place.

        Raises:
            KeyError: No such joint, or no such channel on it.
            ValueError: A negative weight.
        """
        joint = cls.joint_entry(spec, joint_id)
        channels = cls.JOINT_TYPES[joint["type"]][0]
        if channel is not None and channel not in channels:
            raise KeyError(f"No channel {channel!r} on joint {joint_id!r}.")
        if weight < 0.0:
            raise ValueError(f"A weight is never negative (got {weight}).")
        weights = joint.setdefault("weights", {})
        for c in [channel] if channel else channels:
            weights[c] = float(weight)
        return joint

    @classmethod
    def retype(cls, joint: Dict[str, Any], kind: str) -> Dict[str, Any]:
        """Make *joint* a *kind*, keeping only the limits that type has -- a
        slide's travel means nothing to a hinge.

        Returns:
            *joint*, edited in place.
        """
        joint["type"] = kind
        channels = cls.JOINT_TYPES[kind][0]
        joint["limits"] = {
            c: v for c, v in (joint.get("limits") or {}).items() if c in channels
        }
        return joint

    # ================================================================ adjust
    @classmethod
    def pivots(cls, spec: Mapping[str, Any]) -> Dict[str, Matrix]:
        """Where each joint turns, in the frame of the link it hangs off (its
        rest frame there), by joint id -- and where the end control sits in
        the end link's frame, under :attr:`END_KEY` -- the handles a host
        puts out for adjusting (:meth:`adjusted` reads them back)."""
        frames = cls.frames(spec)
        out = {
            j["id"]: cls._mul(frames[j["link"]], cls._inverse(frames[j["parent"]]))
            for j in spec["joints"]
        }
        index = cls.end_joint(spec)
        if index is not None:
            link = spec["joints"][index]["link"]
            out[cls.END_KEY] = cls._mul(
                spec["end"]["frame"], cls._inverse(frames[link])
            )
        return out

    @classmethod
    def adjusted(
        cls,
        spec: Mapping[str, Any],
        handles: Mapping[str, Sequence[float]],
        eps: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """A copy of *spec* with each joint's rest frame and the end
        control's where the pivots were moved to, or None when none moved.

        Parameters:
            spec: The plan.
            handles: Each pivot's matrix in the frame of what it hangs off,
                as :meth:`pivots` keys them -- a joint's read against the
                frames as they WERE, so a child's pivot stays put when its
                parent's moves. A missing key moved nothing.
            eps: A change below this moved nothing (default
                :attr:`ADJUST_EPS`).
        """
        eps = cls.ADJUST_EPS if eps is None else eps
        frames = cls.frames(spec)
        out = copy.deepcopy(spec)
        moved = False

        def placed(key: str, under: Sequence[float]) -> Optional[Matrix]:
            local = handles.get(key)
            if not local:
                return None
            return cls._mul(list(local), under)

        for joint in out["joints"]:
            frame = placed(joint["id"], frames[joint["parent"]])
            if frame is None:
                continue
            rows = [frame[r * 4 : r * 4 + 3] for r in range(4)]
            before = frames[joint["link"]]
            if any(
                abs(frame[r * 4 + c] - before[r * 4 + c]) > eps
                for r in range(4)
                for c in range(3)
            ):
                moved = True
            joint.update(position=rows[3], aim=rows[0], normal=rows[2])
        index = cls.end_joint(spec)
        if index is not None and spec.get("end"):
            frame = placed(cls.END_KEY, frames[spec["joints"][index]["link"]])
            if frame is not None:
                if any(abs(a - b) > eps for a, b in zip(frame, spec["end"]["frame"])):
                    moved = True
                out["end"]["frame"] = frame
        return out if moved else None

    # ================================================================== text
    @classmethod
    def give_of(cls, joint: Mapping[str, Any]) -> float:
        """A joint's give: its channels' weight (the stiffest, when they
        differ -- a panel sets them together)."""
        channels = cls.JOINT_TYPES[joint["type"]][0]
        weights = joint.get("weights") or {}
        return min(float(weights.get(c, 1.0)) for c in channels)

    @staticmethod
    def limits_text(
        limits: Optional[Mapping[str, Sequence[Optional[float]]]],
    ) -> str:
        """*limits* as a panel shows them: ``rz -30..95`` per channel, ``-``
        for a free side, ``free`` for none (what :meth:`parse_limits`
        reads). Six significant digits: an edit of the cell sets every
        channel it shows, so what it shows is what each keeps."""

        def fmt(v: Optional[float]) -> str:
            return "-" if v is None else f"{v:.6g}"

        return (
            ", ".join(
                f"{c} {fmt(lo)}..{fmt(hi)}" for c, (lo, hi) in (limits or {}).items()
            )
            or "free"
        )

    @staticmethod
    def parse_limits(
        text: str, channels: Sequence[str]
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """The limits typed into a Limits cell: ``rz -30..95`` per channel,
        commas between (a one-channel joint may drop the name: ``-30..95``),
        ``-`` or nothing for a free side, ``free`` (or nothing) for none.
        Every channel of *channels* gets an entry; one not named is free.

        Raises:
            ValueError: A channel the joint lacks, or a part that is no
                ``min..max``.
        """
        out: Dict[str, Tuple[Optional[float], Optional[float]]] = {
            c: (None, None) for c in channels
        }
        text = (text or "").strip()
        if not text or text.lower() == "free":
            return out
        bound = r"(-?\d*\.?\d+(?:[eE][-+]?\d+)?|-)?"
        pattern = re.compile(rf"^(?:([rt][xyz])\s*:?\s*)?{bound}\s*\.\.\s*{bound}$")

        def value(s: Optional[str]) -> Optional[float]:
            return None if s in (None, "", "-") else float(s)

        for piece in re.split(r"[,;]", text):
            piece = piece.strip()
            if not piece:
                continue
            match = pattern.match(piece)
            if match is None:
                raise ValueError(f"{piece!r} is not 'channel min..max'.")
            channel, lo, hi = match.groups()
            if channel is None:
                if len(channels) != 1:
                    raise ValueError(
                        f"{piece!r}: name the channel ({', '.join(channels)})."
                    )
                channel = channels[0]
            if channel not in out:
                raise ValueError(
                    f"This joint has no {channel} ({', '.join(channels)})."
                )
            low, high = value(lo), value(hi)
            if low is not None and high is not None and low > high:
                raise ValueError(f"{piece!r}: the minimum is above the maximum.")
            out[channel] = (low, high)
        return out
