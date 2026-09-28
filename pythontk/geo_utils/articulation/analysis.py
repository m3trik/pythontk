# !/usr/bin/python
# coding=utf-8
"""Propose an articulated rig from the parts' geometry alone -- no DCC.

A prop like a desk magnifier is modelled as a handful of rigid PARTS (one
mesh, or one group, per piece that moves on its own), each holding several
SHELLS (connected pieces of geometry: a tube, its end housings, a wing knob).
Where two parts touch, the shells there say how they move against each other:

========== ================================================================
Joint      Evidence
========== ================================================================
slide      the two parts' bodies are tubes on one axis, one inside the
           other, overlapping -- a telescope; its travel is the overlap
ball       a small, near-spherical shell where they meet
hinge      a small shell off the arm's plane where they meet -- the knob or
           bolt a hinge turns on -- else the point where the two bodies'
           axes cross
swivel     a stub of one part sitting on the other's axis inside a housing
universal  a swivel with a hinge knob on it (a clamp-base's tilt and turn)
========== ================================================================

An arm whose parts all lie in one plane (a lamp, a boom, a magnifier) hinges
about that plane's normal -- measured on the production magnifier, every knob
stands 3.4 cm off the plane, while a knob's own long axis is its wing span and
lies IN the plane, so the plane, not the knob, gives the axis.

:meth:`ArticulationAnalysis.propose` returns plain values the DCC builds from
(and a panel shows for the artist to correct): the parts' order, and one joint
per non-root part with its type, rig-space position, aim (the joint's X: down
the link), normal (its Z: the hinge axis) and any limits geometry can state.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np


class ShellFacts(NamedTuple):
    """One shell's measured shape (see :meth:`ArticulationAnalysis.shell`)."""

    centroid: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    #: Rows: the principal axes, longest extent first.
    axes: np.ndarray
    #: Extent along each principal axis, longest first.
    extents: np.ndarray
    #: The two extreme points along the longest axis, on its line: a tube's
    #: end caps. Not centroid +- half the length -- a rod detailed at one end
    #: has its centroid pulled toward it (measured: 5 cm on a 32 cm rod).
    ends: np.ndarray
    #: Furthest point from the longest axis's line through the centroid.
    radius: float
    #: RMS of a least-squares sphere's residual over its radius (0 = a sphere).
    sphericity: float

    @property
    def length(self) -> float:
        return float(self.extents[0])

    @property
    def isotropy(self) -> float:
        """Shortest extent over the longest: 1 for a ball, near 0 for a rod."""
        return float(self.extents[2] / self.extents[0]) if self.extents[0] > 0 else 0.0


class ArticulationAnalysis:
    """The rules that read an articulated rig off its parts' shells."""

    #: What each joint type moves, and in which rotate order (Maya's naming:
    #: ``"zyx"`` applies z first). X runs down the link, Z is the hinge axis:
    #: a hinge turns about Z, a swivel spins about X, a universal turns about
    #: X OUTERMOST (the housing spins on the post, then tilts in the spun
    #: frame), a slide travels along X.
    JOINT_TYPES: Dict[str, Tuple[Tuple[str, ...], str]] = {
        "fixed": ((), "xyz"),
        "hinge": (("rz",), "xyz"),
        "swivel": (("rx",), "xyz"),
        "universal": (("rx", "rz"), "zyx"),
        "ball": (("rx", "ry", "rz"), "xyz"),
        "slide": (("tx",), "xyz"),
    }
    #: Two shells touch within this fraction of the whole prop's size.
    CONTACT_GAP = 0.01
    #: A connector (knob, ball, stub) is under this fraction of the length of
    #: the shorter of the two links' bodies it joins.
    CONNECTOR = 0.5
    #: A ball is under this fraction of it, and round: its shortest extent
    #: at least this fraction of its longest, a sphere fit within this
    #: residual. Loose on purpose -- the production ball is squashed (1.4 x
    #: 1.4 x 1.0, residual 0.13); the smallest round shell at the joint wins.
    BALL = 0.25
    BALL_ISOTROPY = 0.6
    BALL_SPHERICITY = 0.2
    #: Two axes are one when within this many degrees and their lines within
    #: this fraction of the larger radius.
    COAXIAL_DEGREES = 10.0
    COAXIAL_OFFSET = 0.5
    #: One tube is inside another when its radius is smaller by this fraction.
    NESTED = 0.1
    #: The parts lie in a plane when their bodies' centroids spread less than
    #: this fraction across it than along it.
    PLANAR = 0.15
    #: A knob stands off the plane by more than this fraction of its link's
    #: radius (a housing on the link's own axis does not).
    OFF_PLANE = 0.5
    #: A slide keeps this fraction of its overlap inserted at full extension,
    #: and retracts at most this fraction of its exposed length.
    MIN_INSERTION = 0.2
    RETRACT = 0.8

    # ---------------------------------------------------------------- shells
    @staticmethod
    def shell(points: Sequence[Sequence[float]]) -> Optional[ShellFacts]:
        """Measure one shell: its principal axes and extents, its radius about
        the longest axis, and how nearly it is a sphere. ``None`` for fewer
        than four points (nothing to measure).
        """
        from pythontk.geo_utils.pointcloud import PointCloud

        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(pts) < 4:
            return None
        centroid = pts.mean(axis=0)
        basis = PointCloud.pca_basis(pts)
        axes = np.asarray(basis, dtype=float).reshape(4, 4)[:3, :3]
        local = (pts - centroid) @ axes.T
        extents = local.max(axis=0) - local.min(axis=0)
        # Longest EXTENT first: variance orders a squashed ball's axes by its
        # vertex spacing rather than by its size.
        rank = np.argsort(-extents, kind="stable")
        axes, extents, local = axes[rank], extents[rank], local[:, rank]
        ends = np.array(
            [
                centroid + local[:, 0].min() * axes[0],
                centroid + local[:, 0].max() * axes[0],
            ]
        )
        radial = np.linalg.norm(local[:, 1:], axis=1)
        # Algebraic sphere fit: |p|^2 = 2 c.p + (r^2 - |c|^2), linear in c.
        a = np.c_[2.0 * pts, np.ones(len(pts))]
        b = (pts * pts).sum(axis=1)
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        centre = sol[:3]
        r2 = sol[3] + centre @ centre
        if r2 > 0.0:
            r = math.sqrt(r2)
            dist = np.linalg.norm(pts - centre, axis=1)
            sphericity = float(np.sqrt(np.mean((dist - r) ** 2)) / r)
        else:
            sphericity = float("inf")
        return ShellFacts(
            centroid,
            pts.min(axis=0),
            pts.max(axis=0),
            axes,
            extents,
            ends,
            float(radial.max()),
            sphericity,
        )

    # --------------------------------------------------------------- propose
    @classmethod
    def propose(
        cls,
        parts: Sequence[Dict[str, Any]],
        root: Optional[int] = None,
        order: Optional[Sequence[int]] = None,
        up: Sequence[float] = (0.0, 1.0, 0.0),
    ) -> Dict[str, Any]:
        """A rig proposal for *parts*.

        Parameters:
            parts: ``{"name": str, "shells": [points, ...]}`` per part, every
                point in one space (the rig's).
            root: The part that stays put; default the lowest along *up*.
            order: The parts as a chain, root first -- a selection order. The
                default walks the parts' contacts from *root* instead, nearest
                first, so a branching prop proposes a tree; parts nothing
                touches are left out (``"unreached"``).
            up: The scene's up axis, for the default root.

        Returns:
            ``{"order": [...], "parents": {child: parent}, "normal": [x, y,
            z] or None, "unreached": [...], "joints": [...]}`` with one joint
            per non-root part in *order*: ``{"part", "parent", "type",
            "channels", "rotate_order", "position", "aim", "normal",
            "limits": {channel: [min, max]}, "radius", "reason"}`` --
            positions in the parts' space, slide limits in its units,
            rotations unbounded (geometry cannot say how far a knob lets an
            arm fold); ``radius`` the thinner joined link's, for sizing.
        """
        facts: List[List[ShellFacts]] = []
        for part in parts:
            measured = [cls.shell(s) for s in part.get("shells") or []]
            facts.append([f for f in measured if f is not None])
        if not facts or any(not f for f in facts):
            empty = [p.get("name") for p, f in zip(parts, facts) if not f]
            raise ValueError(f"Parts with no measurable geometry: {empty}.")
        body = [max(fs, key=lambda f: f.length) for fs in facts]
        lo = np.min([f.lo for fs in facts for f in fs], axis=0)
        hi = np.max([f.hi for fs in facts for f in fs], axis=0)
        gap = cls.CONTACT_GAP * float(np.linalg.norm(hi - lo))
        touching = cls._contacts(facts, gap)

        if order is not None:
            chain = [int(i) for i in order]
            parents = {chain[k]: chain[k - 1] for k in range(1, len(chain))}
            unreached = [i for i in range(len(parts)) if i not in chain]
        else:
            if root is None:
                up_v = np.asarray(up, dtype=float)
                root = int(np.argmin([min(f.lo @ up_v for f in fs) for fs in facts]))
            chain, parents = cls._walk(root, touching, body)
            unreached = [i for i in range(len(parts)) if i not in parents and i != root]
        normal = cls._plane([body[i] for i in chain])

        joints: Dict[int, Dict[str, Any]] = {}
        for child in chain[1:]:
            parent = parents[child]
            joints[child] = cls._joint(parent, child, facts, body, normal, gap)
            # The joint's girth is the thinner link's: a head's ring is wide,
            # the rod it swivels on is not.
            radius = min(body[child].radius, body[parent].radius)
            joints[child].update(part=child, parent=parent, radius=radius)
        cls._aim(chain, parents, joints, body)
        return {
            "order": chain,
            "parents": parents,
            "normal": None if normal is None else [float(v) for v in normal],
            "unreached": unreached,
            "joints": [cls._plain(joints[c]) for c in chain[1:]],
        }

    # ------------------------------------------------------------- internals
    @staticmethod
    def _overlap(a: ShellFacts, b: ShellFacts, gap: float) -> bool:
        return bool(np.all(a.lo - gap <= b.hi) and np.all(b.lo - gap <= a.hi))

    @classmethod
    def _contacts(
        cls, facts: Sequence[Sequence[ShellFacts]], gap: float
    ) -> Dict[int, set]:
        """Which parts touch: any shell of one within *gap* of any of the
        other's (boxes -- the parts of a prop are fitted, not near-missed)."""
        touching: Dict[int, set] = {i: set() for i in range(len(facts))}
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                if any(cls._overlap(a, b, gap) for a in facts[i] for b in facts[j]):
                    touching[i].add(j)
                    touching[j].add(i)
        return touching

    @staticmethod
    def _walk(
        root: int, touching: Dict[int, set], body: Sequence[ShellFacts]
    ) -> Tuple[List[int], Dict[int, int]]:
        """Breadth-first from *root* over the contacts, nearest body first:
        the order a chain reads in, and a tree's parents."""
        order, parents, queue = [root], {}, deque([root])
        seen = {root}
        while queue:
            here = queue.popleft()
            ahead = sorted(
                (n for n in touching[here] if n not in seen),
                key=lambda n: float(
                    np.linalg.norm(body[n].centroid - body[here].centroid)
                ),
            )
            for n in ahead:
                seen.add(n)
                parents[n] = here
                order.append(n)
                queue.append(n)
        return order, parents

    @classmethod
    def _plane(cls, bodies: Sequence[ShellFacts]) -> Optional[np.ndarray]:
        """The normal of the plane the bodies' centroids lie in, or None --
        fewer than three, on one line (which spans no plane: its normal would
        be the SVD's rounding noise), or spread out of any one plane. Signed
        so its largest component is positive: the same arm always hinges the
        same way round."""
        if len(bodies) < 3:
            return None
        pts = np.array([b.centroid for b in bodies])
        centred = pts - pts.mean(axis=0)
        _u, s, vt = np.linalg.svd(centred, full_matrices=False)
        if s[0] <= 0.0 or s[1] <= 1e-6 * s[0] or s[-1] > cls.PLANAR * s[0]:
            return None
        normal = vt[-1] / np.linalg.norm(vt[-1])
        if normal[int(np.argmax(np.abs(normal)))] < 0.0:
            normal = -normal
        return normal

    @classmethod
    def _coaxial(cls, a: ShellFacts, b: ShellFacts) -> bool:
        cos = abs(float(a.axes[0] @ b.axes[0]))
        if cos < math.cos(math.radians(cls.COAXIAL_DEGREES)):
            return False
        offset = b.centroid - a.centroid
        off_axis = offset - (offset @ a.axes[0]) * a.axes[0]
        return float(np.linalg.norm(off_axis)) <= cls.COAXIAL_OFFSET * max(
            a.radius, b.radius
        )

    @staticmethod
    def _span(
        shell: ShellFacts, origin: np.ndarray, axis: np.ndarray
    ) -> Tuple[float, float]:
        """The shell's extent along *axis* from *origin*: its end caps."""
        a, b = ((end - origin) @ axis for end in shell.ends)
        return (float(a), float(b)) if a <= b else (float(b), float(a))

    @classmethod
    def _nested(
        cls,
        a: ShellFacts,
        b: ShellFacts,
        gap: float,
        axis: np.ndarray,
        origin: np.ndarray,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Both spans along *axis* when *a* and *b* are one inside the other
        and overlap there; else None.

        One inside the other is a radius a clear step smaller (a tenth), not
        smaller by the contact gap: a telescope's two tubes differ by less than
        a finger's width at any prop scale (measured: 0.9 cm on the magnifier,
        and a 0.5 cm post in its housing)."""
        if not cls._coaxial(a, b):
            return None
        if abs(a.radius - b.radius) <= cls.NESTED * max(a.radius, b.radius):
            return None
        sa, sb = cls._span(a, origin, axis), cls._span(b, origin, axis)
        if min(sa[1], sb[1]) - max(sa[0], sb[0]) <= gap:
            return None
        return sa, sb

    @classmethod
    def _joint(
        cls,
        parent: int,
        child: int,
        facts: Sequence[Sequence[ShellFacts]],
        body: Sequence[ShellFacts],
        normal: Optional[np.ndarray],
        gap: float,
    ) -> Dict[str, Any]:
        a, b = body[parent], body[child]
        axis = (
            a.axes[0]
            if float((b.centroid - a.centroid) @ a.axes[0]) >= 0
            else -a.axes[0]
        )
        origin = a.centroid

        # A telescope: the two bodies nested on one axis.
        spans = cls._nested(a, b, gap, axis, origin)
        if spans is not None:
            return cls._slide(a, b, spans, axis, origin, normal)

        shorter = min(a.length, b.length)
        other = {parent: child, child: parent}
        connectors = [
            (owner, f)
            for owner in (parent, child)
            for f in facts[owner]
            if f is not body[owner]
            and f.length < cls.CONNECTOR * shorter
            and any(cls._overlap(f, g, gap) for g in facts[other[owner]])
        ]

        balls = [
            f
            for _o, f in connectors
            if f.length < cls.BALL * shorter
            and f.isotropy >= cls.BALL_ISOTROPY
            and f.sphericity <= cls.BALL_SPHERICITY
        ]
        if balls:
            ball = min(balls, key=lambda f: f.length)
            return cls._make(
                "ball", ball.centroid, normal, "a sphere where the parts meet"
            )

        knobs = []
        if normal is not None:
            for owner, f in connectors:
                lever = float((f.centroid - body[owner].centroid) @ normal)
                if abs(lever) > cls.OFF_PLANE * body[owner].radius:
                    knobs.append(f)

        # A stub of one part sitting inside a housing of the other, on one axis.
        swivel = swivel_point = None
        for fa in facts[parent]:
            for fb in facts[child]:
                if fa is body[parent] and fb is body[child]:
                    continue  # two bodies nested: the slide above
                if cls._nested(fa, fb, gap, fb.axes[0], fb.centroid) is not None:
                    swivel = (
                        fb.axes[0]
                        if float((b.centroid - fb.centroid) @ fb.axes[0]) >= 0
                        else -fb.axes[0]
                    )
                    swivel_point = fb.centroid
                    break
            if swivel is not None:
                break

        if knobs:
            contact = cls._contact_point(a, b)
            knob = min(knobs, key=lambda f: float(np.linalg.norm(f.centroid - contact)))
            point = (
                knob.centroid - float((knob.centroid - a.centroid) @ normal) * normal
            )
            if swivel is not None:
                # The tilt's centre on the swivel's axis.
                rel = point - swivel_point
                point = swivel_point + float(rel @ swivel) * swivel
                joint = cls._make("universal", point, normal, "a knob on a swivel")
                joint["aim"] = swivel
                return joint
            return cls._make("hinge", point, normal, "a knob off the arm's plane")

        if swivel is not None:
            contact = cls._contact_point(a, b)
            point = swivel_point + float((contact - swivel_point) @ swivel) * swivel
            joint = cls._make("swivel", point, normal, "a stub turning in a housing")
            joint["aim"] = swivel
            return joint

        point = cls._crossing(a, b)
        if point is None:
            point = cls._contact_point(a, b)
            reason = "where the parts meet"
        else:
            reason = "where the parts' axes cross"
        if normal is not None:
            point = point - float((point - a.centroid) @ normal) * normal
        hinge_axis = normal
        if hinge_axis is None:
            crossed = np.cross(a.axes[0], b.axes[0])
            if np.linalg.norm(crossed) > 1e-6:
                hinge_axis = crossed / np.linalg.norm(crossed)
        return cls._make("hinge", point, hinge_axis, reason)

    @classmethod
    def _slide(
        cls,
        a: ShellFacts,
        b: ShellFacts,
        spans: Tuple[Tuple[float, float], Tuple[float, float]],
        axis: np.ndarray,
        origin: np.ndarray,
        normal: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        (a0, a1), (b0, b1) = spans
        if a.radius > b.radius:
            # The child slides out of the parent's far end.
            at = a1
            inserted = a1 - b0
            travel_out = inserted * (1.0 - cls.MIN_INSERTION)
            travel_in = min((b1 - a1) * cls.RETRACT, max(b0 - a0, 0.0))
            reason = "the child's tube runs inside the parent's"
        else:
            # The child is a sleeve riding over the parent's rod.
            at = b0
            inserted = a1 - b0
            travel_out = inserted * (1.0 - cls.MIN_INSERTION)
            travel_in = min(max(b0 - a0, 0.0) * cls.RETRACT, max(b1 - a1, 0.0))
            reason = "the child's sleeve rides over the parent's tube"
        point = origin + at * axis
        joint = cls._make("slide", point, normal, reason)
        joint["aim"] = axis
        joint["limits"]["tx"] = [-float(travel_in), float(travel_out)]
        return joint

    @staticmethod
    def _contact_point(a: ShellFacts, b: ShellFacts) -> np.ndarray:
        """The middle of the two bodies' overlapping box, or of the gap
        between them."""
        lo = np.maximum(a.lo, b.lo)
        hi = np.minimum(a.hi, b.hi)
        return 0.5 * (lo + hi)

    @staticmethod
    def _crossing(a: ShellFacts, b: ShellFacts) -> Optional[np.ndarray]:
        """The midpoint of the closest approach of the two bodies' axes, or
        None when they run parallel."""
        d1, d2 = a.axes[0], b.axes[0]
        w = a.centroid - b.centroid
        aa, bb, cc = d1 @ d1, d1 @ d2, d2 @ d2
        dd, ee = d1 @ w, d2 @ w
        denom = aa * cc - bb * bb
        if abs(denom) < 1e-9:
            return None
        s = (bb * ee - cc * dd) / denom
        t = (aa * ee - bb * dd) / denom
        return 0.5 * ((a.centroid + s * d1) + (b.centroid + t * d2))

    @classmethod
    def _make(
        cls, kind: str, point: np.ndarray, normal: Optional[np.ndarray], reason: str
    ) -> Dict[str, Any]:
        channels, rotate_order = cls.JOINT_TYPES[kind]
        return {
            "type": kind,
            "channels": list(channels),
            "rotate_order": rotate_order,
            "position": np.asarray(point, dtype=float),
            "aim": None,
            "normal": None if normal is None else np.asarray(normal, dtype=float),
            "limits": {},
            "reason": reason,
        }

    @classmethod
    def _aim(
        cls,
        chain: Sequence[int],
        parents: Dict[int, int],
        joints: Dict[int, Dict[str, Any]],
        body: Sequence[ShellFacts],
    ) -> None:
        """Point each joint's X down its link: at its one child joint, else
        on along its parent's aim (a head at the end of the arm), else out
        along its body. A slide, swivel or universal keeps the axis it
        travels or spins on."""
        children: Dict[int, List[int]] = {}
        for child, parent in parents.items():
            children.setdefault(parent, []).append(child)
        for part in chain[1:]:
            joint = joints[part]
            if joint["aim"] is not None:
                continue
            below = [c for c in children.get(part, []) if c in joints]
            aim = None
            if len(below) == 1:
                aim = joints[below[0]]["position"] - joint["position"]
            if aim is None or np.linalg.norm(aim) < 1e-9:
                above = joints.get(parents[part])
                aim = None if above is None else above["aim"]
            if aim is None or np.linalg.norm(aim) < 1e-9:
                axis = body[part].axes[0]
                away = body[part].centroid - joint["position"]
                aim = axis if float(axis @ away) >= 0 else -axis
            joint["aim"] = np.asarray(aim, dtype=float) / np.linalg.norm(aim)
        for part in chain[1:]:
            joint = joints[part]
            if joint["normal"] is None:
                joint["normal"] = cls._perpendicular(joint["aim"])

    @staticmethod
    def _perpendicular(aim: np.ndarray) -> np.ndarray:
        """The world axis least along *aim*, made perpendicular to it."""
        axis = np.eye(3)[int(np.argmin(np.abs(aim)))]
        axis = axis - (axis @ aim) * aim
        return axis / np.linalg.norm(axis)

    @staticmethod
    def _plain(joint: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(joint)
        for key in ("position", "aim", "normal"):
            if out.get(key) is not None:
                out[key] = [float(v) for v in out[key]]
        return out
