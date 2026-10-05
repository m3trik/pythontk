# !/usr/bin/python
# coding=utf-8
"""The articulated-rig joint model and its grab solver -- pure math, no DCC.

A rig is a tree of joints in RIG SPACE (the space the root joints' parent
defines). Each joint carries a rest frame -- ``t``, its translate in the
parent's space, and ``q``, its rotation (the DCC's joint orient) -- and a set
of CHANNELS, the only values an animator or a grab ever changes:

``rx`` / ``ry`` / ``rz``
    degrees about the rest frame's own axes, composed in the joint's
    ``rotate_order`` exactly as Maya composes them (``"xyz"`` applies x first);
``tx`` / ``ty`` / ``tz``
    a slide along the rest frame's axes, in the parent's units.

so a joint's local transform is::

    rotation    = q (x) R(rotate_order; rx, ry, rz)
    translation = t + q . (tx, ty, tz)

which is Maya's ``[R][JO][T]`` for a joint whose ``jointOrient`` is ``q`` --
and, identically, the local TRS the FBX hop delivers (Unity folds the orient
into ``localRotation``, FBX2glTF into the node's ``rotation``). A joint whose
channels are all 0 stands at rest.

The STATE is one float per channel, joint by joint in channel order
(:attr:`ArticulationModel.channels`). Everything here maps a state to poses
and back, and solves the state that brings a grabbed point to a target:

- :meth:`ArticulationModel.pose` / :meth:`~ArticulationModel.world` --
  forward kinematics;
- :meth:`ArticulationModel.read` -- a state back from local transforms (a
  playing clip, a DCC scene);
- :meth:`ArticulationModel.solve` -- the grab: damped least squares over the
  channels between the root and the grabbed link, with limits never left. A
  grabbed link hanging off a ball joint takes the hand's rotation, and the
  rest of the chain places the ball's centre (the wrist split).

Every runtime runs a PORT of this module and is pinned against it by golden
cases (pythontk ``test_articulated_rig_web.py``, unitytk
``test_articulated_rig_runtime.py``), so it is written to port: plain
3-vectors and ``(x, y, z, w)`` quaternions, one 3 x 3 solve, no numpy. A
constant or a step changed here is a change in every port.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

Vec = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]

#: The channels a joint may carry: rotations first, in axis order, then slides.
ROTATE_CHANNELS: Tuple[str, ...] = ("rx", "ry", "rz")
TRANSLATE_CHANNELS: Tuple[str, ...] = ("tx", "ty", "tz")
CHANNELS: Tuple[str, ...] = ROTATE_CHANNELS + TRANSLATE_CHANNELS
#: Maya's rotate orders, in its enum order (``rotateOrder`` 0-5).
ROTATE_ORDERS: Tuple[str, ...] = ("xyz", "yzx", "zxy", "xzy", "yxz", "zyx")

_AXIS = {"x": 0, "y": 1, "z": 2}
_IDENTITY: Quat = (0.0, 0.0, 0.0, 1.0)
_ORIGIN: Vec = (0.0, 0.0, 0.0)


class _ArticulationModelInternal:
    """Vector, quaternion and Euler helpers -- the part every port copies.

    Quaternions are ``(x, y, z, w)``; ``a (x) b`` applies ``b`` first, so a
    child's world rotation is ``parent (x) local``. Matrices are 3 x 3 row
    lists for COLUMN vectors (``v' = M v``).
    """

    @staticmethod
    def _add(a: Sequence[float], b: Sequence[float]) -> Vec:
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    @staticmethod
    def _sub(a: Sequence[float], b: Sequence[float]) -> Vec:
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    @staticmethod
    def _scale(a: Sequence[float], s: float) -> Vec:
        return (a[0] * s, a[1] * s, a[2] * s)

    @staticmethod
    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _cross(a: Sequence[float], b: Sequence[float]) -> Vec:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    @staticmethod
    def _norm(a: Sequence[float]) -> float:
        return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

    @staticmethod
    def _qmul(a: Sequence[float], b: Sequence[float]) -> Quat:
        """``a (x) b``: the rotation ``b`` followed by ``a``."""
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    @staticmethod
    def _qconj(q: Sequence[float]) -> Quat:
        return (-q[0], -q[1], -q[2], q[3])

    @staticmethod
    def _qnormalize(q: Sequence[float]) -> Quat:
        n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if n < 1.0e-12:
            return _IDENTITY
        return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)

    @classmethod
    def _qrot(cls, q: Sequence[float], v: Sequence[float]) -> Vec:
        """*v* rotated by the unit quaternion *q*."""
        u = (q[0], q[1], q[2])
        t = cls._scale(cls._cross(u, v), 2.0)
        return cls._add(cls._add(v, cls._scale(t, q[3])), cls._cross(u, t))

    @staticmethod
    def _qaxis(axis: int, degrees: float) -> Quat:
        """A rotation of *degrees* about the unit axis *axis* (0, 1, 2)."""
        half = math.radians(degrees) * 0.5
        s = math.sin(half)
        return (
            s if axis == 0 else 0.0,
            s if axis == 1 else 0.0,
            s if axis == 2 else 0.0,
            math.cos(half),
        )

    @classmethod
    def _euler_quat(cls, order: str, angles: Mapping[str, float]) -> Quat:
        """The rotation of Euler *angles* (``{"rx": deg, ...}``, a missing
        channel 0) in Maya's *order*: ``"xyz"`` applies x first, so the
        rotation is ``Rz (x) Ry (x) Rx``."""
        q = _IDENTITY
        for c in order:  # innermost first; each later one is applied after
            q = cls._qmul(cls._qaxis(_AXIS[c], angles.get("r" + c, 0.0)), q)
        return q

    @staticmethod
    def _qmatrix(q: Sequence[float]) -> List[List[float]]:
        x, y, z, w = q
        return [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]

    @staticmethod
    def _matrix_euler(order: str, m: Sequence[Sequence[float]]) -> Dict[str, float]:
        """Euler degrees ``{"rx", "ry", "rz"}`` of the rotation matrix *m* in
        Maya's *order*.

        *order* ``o0 o1 o2`` composes ``M = R_o2 R_o1 R_o0``. With ``a, b, c =
        o2, o1, o0`` and ``s`` the parity of ``(a, b, c)`` (+1 cyclic), the
        middle angle is ``asin(s M[a][c])`` and the outer two come from the
        row and column it leaves; at gimbal lock the innermost is taken as 0
        and the outermost absorbs the whole turn. The middle angle is always
        in [-90, 90].
        """
        a, b, c = _AXIS[order[2]], _AXIS[order[1]], _AXIS[order[0]]
        s = 1.0 if (b - a) % 3 == 1 else -1.0
        sin_b = max(-1.0, min(1.0, s * m[a][c]))
        angle_b = math.asin(sin_b)
        if math.cos(angle_b) > 1.0e-6:
            angle_a = math.atan2(-s * m[b][c], m[c][c])
            angle_c = math.atan2(-s * m[a][b], m[a][a])
        else:
            angle_c = 0.0
            angle_a = math.atan2(s * m[c][b], m[b][b])
        out = {"rx": 0.0, "ry": 0.0, "rz": 0.0}
        out["r" + order[2]] = math.degrees(angle_a)
        out["r" + order[1]] = math.degrees(angle_b)
        out["r" + order[0]] = math.degrees(angle_c)
        return out

    @classmethod
    def _euler_for(
        cls,
        order: str,
        m: Sequence[Sequence[float]],
        channels: Sequence[str],
        reference: Mapping[str, float],
    ) -> Dict[str, float]:
        """Euler degrees of *m* in *order* for a joint that turns on
        *channels*, each unwrapped toward *reference* (the joint's values so
        far).

        Every rotation has two Euler triples, ``(A, B, C)`` and ``(A + 180,
        180 - B, C + 180)``; :meth:`_matrix_euler` returns the one whose
        middle angle is within +-90. Which one is right depends on the joint:

        - lacking a channel, the triple that puts THAT channel at 0 -- a hinge
          turning on its order's middle axis past 90 degrees reads as the
          other triple, and dropping its two "missing" 180s would read it
          wrong;
        - a ball (all three), the triple nearer *reference* -- a clip posing
          the middle axis past 90 must not flip to the other triple halfway
          through, or the next grab starts from a jump.
        """
        first = cls._matrix_euler(order, m)
        second = {
            "r" + order[2]: first["r" + order[2]] + 180.0,
            "r" + order[1]: 180.0 - first["r" + order[1]],
            "r" + order[0]: first["r" + order[0]] + 180.0,
        }
        scored = []
        for triple in (first, second):
            wound = {
                c: cls._unwrap(v, reference.get(c, 0.0)) for c, v in triple.items()
            }
            missing = 0.0
            drift = 0.0
            for c in ("rx", "ry", "rz"):
                if c in channels:
                    drift += abs(wound[c] - reference.get(c, 0.0))
                else:
                    missing += abs(cls._unwrap(triple[c], 0.0))
            scored.append((missing, drift, wound))
        (m1, d1, w1), (m2, d2, w2) = scored
        # The channels a joint lacks decide first (to 1e-6 degrees), then
        # continuity; a tie keeps the first. Spelled out rather than a sort
        # on rounded keys: every port must break the tie the same way.
        if m2 < m1 - 1.0e-6 or (abs(m2 - m1) <= 1.0e-6 and d2 < d1):
            return w2
        return w1

    @staticmethod
    def _unwrap(value: float, reference: float) -> float:
        """*value* shifted by whole turns to lie nearest *reference* (a half
        turn rounds up -- ``floor(x + 0.5)``, which every port spells the
        same, unlike Python's round-half-to-even)."""
        return value + 360.0 * math.floor((reference - value) / 360.0 + 0.5)

    @staticmethod
    def _solve3(a: Sequence[Sequence[float]], e: Sequence[float]) -> Vec:
        """``A^-1 e`` for a symmetric positive-definite 3 x 3 *a* (the damped
        normal matrix); the zero vector when it is singular."""
        (a00, a01, a02), (a10, a11, a12), (a20, a21, a22) = a
        c00 = a11 * a22 - a12 * a21
        c01 = a12 * a20 - a10 * a22
        c02 = a10 * a21 - a11 * a20
        det = a00 * c00 + a01 * c01 + a02 * c02
        if abs(det) < 1.0e-300:
            return _ORIGIN
        inv = 1.0 / det
        c10 = a02 * a21 - a01 * a22
        c11 = a00 * a22 - a02 * a20
        c12 = a01 * a20 - a00 * a21
        c20 = a01 * a12 - a02 * a11
        c21 = a02 * a10 - a00 * a12
        c22 = a00 * a11 - a01 * a10
        return (
            (c00 * e[0] + c10 * e[1] + c20 * e[2]) * inv,
            (c01 * e[0] + c11 * e[1] + c21 * e[2]) * inv,
            (c02 * e[0] + c12 * e[1] + c22 * e[2]) * inv,
        )


class ArticulationModel(_ArticulationModelInternal):
    """One rig's joints, channels and limits, and the solver that poses them.

    Built from one rig of the ``articulation`` scene record
    (``ptk.SceneRecords.ARTICULATION``)::

        {"name": "magnifier",
         "joints": [{"name": "magnifier_base_jnt", "parent": null,
                     "t": [x, y, z], "q": [x, y, z, w],
                     "rotate_order": "zyx",
                     "channels": [{"channel": "rx", "min": -180, "max": 180,
                                   "weight": 1.0}, ...]}, ...],
         "grab": [...]}

    ``parent`` is an earlier joint's index or null (a root: its parent is the
    rig space); ``min`` / ``max`` are null for an unbounded channel;
    ``weight`` (default 1) is how readily the grab moves that channel against
    the others. ``grab`` is the runtimes' business and read here only by
    :meth:`from_record`.

    Parameters:
        rig: One rig mapping, as above.

    Raises:
        ValueError: A joint whose parent is not an earlier joint, an unknown
            channel or rotate order, a channel named twice on one joint, or a
            limit whose ``min`` exceeds its ``max``.
    """

    #: Solver iterations per position solve; it stops early once within
    #: :attr:`TOLERANCE`.
    ITERATIONS = 32
    #: Position solves of a wrist split: the ball's rotation is clamped after
    #: the first, which moves the grab point the second solve then corrects.
    PASSES = 2
    #: The damping, as a fraction of the chain's reach: small enough to reach,
    #: large enough that a stretched-out arm does not snap through a
    #: singularity.
    DAMPING = 0.05
    #: Converged within this fraction of the reach.
    TOLERANCE = 1.0e-4
    #: The furthest one iteration chases the target, as a fraction of the
    #: reach -- a far target is approached, not leapt at.
    MAX_STEP = 0.25
    #: Halvings a step may take to bring the held point nearer its target. A
    #: step that would leave it further is never taken -- past the reach the
    #: full step overshoots, and a solve that took it thrashed from flail to
    #: flail -- so the arm settles where it comes nearest.
    HALVINGS = 6
    #: A length below this is no length (a chain with no reach, a zero axis).
    EPS = 1.0e-9

    def __init__(self, rig: Mapping[str, Any]):
        self.name = str(rig.get("name") or "")
        self.joints: List[Dict[str, Any]] = []
        #: ``(joint index, channel)`` per state slot, in state order.
        self.channels: List[Tuple[int, str]] = []
        self._limits: List[Tuple[Optional[float], Optional[float]]] = []
        self._weights: List[float] = []
        self._slots: List[List[Tuple[int, str]]] = []
        for index, spec in enumerate(rig.get("joints") or []):
            self._add_joint(index, spec)

    def _add_joint(self, index: int, spec: Mapping[str, Any]) -> None:
        name = str(spec.get("name") or f"joint{index}")
        parent = spec.get("parent")
        if parent is not None:
            parent = int(parent)
            if not 0 <= parent < index:
                raise ValueError(
                    f"Joint {name!r}: parent {parent} is not an earlier joint."
                )
        order = str(spec.get("rotate_order") or "xyz")
        if order not in ROTATE_ORDERS:
            raise ValueError(f"Joint {name!r}: unknown rotate order {order!r}.")
        t = tuple(float(v) for v in (spec.get("t") or _ORIGIN))
        q = self._qnormalize(tuple(float(v) for v in (spec.get("q") or _IDENTITY)))
        slots: List[Tuple[int, str]] = []
        seen = set()
        for entry in spec.get("channels") or []:
            channel = str(entry.get("channel"))
            if channel not in CHANNELS:
                raise ValueError(f"Joint {name!r}: unknown channel {channel!r}.")
            if channel in seen:
                raise ValueError(f"Joint {name!r}: channel {channel!r} named twice.")
            seen.add(channel)
            lo, hi = entry.get("min"), entry.get("max")
            lo = None if lo is None else float(lo)
            hi = None if hi is None else float(hi)
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(
                    f"Joint {name!r}: {channel} limits [{lo}, {hi}] are reversed."
                )
            weight = entry.get("weight")
            slot = len(self.channels)
            self.channels.append((index, channel))
            self._limits.append((lo, hi))
            self._weights.append(max(0.0, 1.0 if weight is None else float(weight)))
            slots.append((slot, channel))
        self.joints.append(
            {"name": name, "parent": parent, "t": t, "q": q, "rotate_order": order}
        )
        self._slots.append(slots)

    @classmethod
    def from_record(
        cls, payload: Mapping[str, Any], name: Optional[str] = None
    ) -> "ArticulationModel":
        """The model of the rig *name* (default: the first) in an
        ``articulation`` record payload.

        Raises:
            KeyError: No rig by that name (or no rig at all).
        """
        for rig in payload.get("rigs") or []:
            if name is None or rig.get("name") == name:
                return cls(rig)
        raise KeyError(f"No articulated rig {name!r} in the record.")

    # ---------------------------------------------------------------- queries
    def joint_index(self, name: str) -> int:
        """The index of the joint called *name*.

        Raises:
            KeyError: No such joint.
        """
        for index, joint in enumerate(self.joints):
            if joint["name"] == name:
                return index
        raise KeyError(f"No joint {name!r} in rig {self.name!r}.")

    def chain(self, joint: int) -> List[int]:
        """The joints from *joint*'s root down to *joint*, root first."""
        out = []
        index: Optional[int] = joint
        while index is not None:
            out.append(index)
            index = self.joints[index]["parent"]
        return out[::-1]

    def rest_state(self) -> List[float]:
        """Every channel at 0: the rig at rest."""
        return [0.0] * len(self.channels)

    def limits(self, slot: int) -> Tuple[Optional[float], Optional[float]]:
        """``(min, max)`` of state slot *slot*; ``None`` is unbounded."""
        return self._limits[slot]

    def clamp(self, state: Sequence[float]) -> List[float]:
        """*state* with every channel inside its limits."""
        return [self._clamp_slot(s, v) for s, v in enumerate(state)]

    def _clamp_slot(self, slot: int, value: float) -> float:
        lo, hi = self._limits[slot]
        if lo is not None and value < lo:
            return lo
        if hi is not None and value > hi:
            return hi
        return value

    def _values(self, state: Sequence[float], joint: int) -> Dict[str, float]:
        return {channel: state[slot] for slot, channel in self._slots[joint]}

    # ---------------------------------------------------------------- forward
    def local(self, state: Sequence[float], joint: int) -> Tuple[Vec, Quat]:
        """Joint *joint*'s local ``(translation, rotation)`` at *state*."""
        spec = self.joints[joint]
        values = self._values(state, joint)
        rotation = self._qmul(spec["q"], self._euler_quat(spec["rotate_order"], values))
        slide = (values.get("tx", 0.0), values.get("ty", 0.0), values.get("tz", 0.0))
        translation = self._add(spec["t"], self._qrot(spec["q"], slide))
        return translation, rotation

    def pose(self, state: Sequence[float]) -> List[Tuple[Vec, Quat]]:
        """Every joint's local ``(translation, rotation)`` at *state*."""
        return [self.local(state, j) for j in range(len(self.joints))]

    def world(self, state: Sequence[float]) -> List[Tuple[Vec, Quat]]:
        """Every joint's ``(position, rotation)`` in rig space at *state*."""
        out: List[Tuple[Vec, Quat]] = []
        for index, spec in enumerate(self.joints):
            t, q = self.local(state, index)
            parent = spec["parent"]
            if parent is None:
                out.append((t, q))
            else:
                pp, pq = out[parent]
                out.append((self._add(pp, self._qrot(pq, t)), self._qmul(pq, q)))
        return out

    def point(
        self, state: Sequence[float], joint: int, local_point: Sequence[float]
    ) -> Vec:
        """*local_point* (in joint *joint*'s frame) in rig space at *state*."""
        p, q = self.world(state)[joint]
        return self._add(p, self._qrot(q, local_point))

    def to_local(
        self, state: Sequence[float], joint: int, point: Sequence[float]
    ) -> Vec:
        """The rig-space *point* in joint *joint*'s frame at *state* -- where a
        grab records what it holds."""
        p, q = self.world(state)[joint]
        return self._qrot(self._qconj(q), self._sub(point, p))

    # ---------------------------------------------------------------- inverse
    def read(
        self,
        locals_: Sequence[Tuple[Sequence[float], Sequence[float]]],
        hint: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """The state that poses the joints at *locals_* -- one local
        ``(translation, rotation)`` per joint, as a clip or a scene has them.

        A rotation is read in the joint's rotate order and, for a joint with
        fewer than three rotation channels, as the Euler triple that keeps the
        channels it lacks at 0 (:meth:`_euler_for`); any turn outside its
        channels is dropped. Angles come back in (-180, 180], or as near
        *hint* (a previous state) as whole turns allow, so a hinge wound past
        180 reads on. Nothing is clamped: a clip may pose outside the limits.
        """
        state = self.rest_state() if hint is None else [float(v) for v in hint]
        for index, (t, q) in enumerate(locals_):
            slots = self._slots[index]
            if not slots:
                continue
            spec = self.joints[index]
            relative = self._qmul(self._qconj(spec["q"]), self._qnormalize(q))
            angles = self._euler_for(
                spec["rotate_order"],
                self._qmatrix(relative),
                [c for _s, c in slots if c in ROTATE_CHANNELS],
                {c: state[s] for s, c in slots},
            )
            slide = self._qrot(self._qconj(spec["q"]), self._sub(t, spec["t"]))
            for slot, channel in slots:
                if channel in ROTATE_CHANNELS:
                    state[slot] = angles[channel]
                else:
                    state[slot] = slide[TRANSLATE_CHANNELS.index(channel)]
        return state

    def scale_of(
        self, locals_: Sequence[Tuple[Sequence[float], Sequence[float]]]
    ) -> float:
        """How many of the runtime's units one of the record's makes, measured
        on joints that cannot slide: the median ratio of their imported rest
        translation to the recorded one (1.0 when none can say).

        A runtime measures this rather than trusting its import settings: FBX
        unit conversion, glTF's metres and a scaled rig parent all land in the
        same number.
        """
        ratios = []
        for index, (t, _q) in enumerate(locals_):
            if any(c in TRANSLATE_CHANNELS for _s, c in self._slots[index]):
                continue
            rest = self._norm(self.joints[index]["t"])
            got = self._norm(t)
            if rest > self.EPS and got > self.EPS:
                ratios.append(got / rest)
        if not ratios:
            return 1.0
        ratios.sort()
        mid = len(ratios) // 2
        return ratios[mid] if len(ratios) % 2 else 0.5 * (ratios[mid - 1] + ratios[mid])

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        state: Sequence[float],
        joint: int,
        local_point: Sequence[float],
        target: Sequence[float],
        rotation: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """The state that brings *local_point* on joint *joint*'s link to the
        rig-space *target* -- a grab's step.

        Only the channels between the root and *joint* move; the links beyond
        it ride along untouched. Every channel stays inside its limits, and a
        target out of reach leaves the point as near it as the limits allow.
        The solve starts from *state*, so a grab followed frame by frame moves
        each channel as little as it can.

        With *rotation* (the grabbed link's wanted rig-space rotation) and a
        *joint* carrying all three rotation channels -- a ball -- the grab is
        split the way a wrist is: the ball takes *rotation* (clamped), and the
        chain above it places the ball's centre where that rotation puts the
        grabbed point on the target; a last position solve over the whole
        chain then closes whatever the ball's limits left open, so the point
        always wins over the rotation. Without a ball, *rotation* is ignored:
        a hinge's orientation is its position's consequence.

        Parameters:
            state: Where the rig is now, one float per channel.
            joint: The index of the joint whose link is held.
            local_point: The held point, in that joint's frame (see
                :meth:`to_local`).
            target: Where the held point should be, in rig space.
            rotation: The held link's wanted rotation ``(x, y, z, w)`` in rig
                space, or None.

        Returns:
            The new state.
        """
        current = self.clamp(state)
        chain = self.chain(joint)
        slots = [s for j in chain for s, _c in self._slots[j]]
        spec_rotations = {c for _s, c in self._slots[joint] if c in ROTATE_CHANNELS}
        wrist = rotation is not None and spec_rotations == set(ROTATE_CHANNELS)
        reach = self._reach(chain, local_point)
        if not wrist:
            return self._dls(current, slots, joint, local_point, target, reach)
        rotation = self._qnormalize(rotation)
        placed = [s for s in slots if self.channels[s] not in self._rotations(joint)]
        held = rotation
        for _ in range(self.PASSES):
            centre = self._sub(target, self._qrot(held, local_point))
            current = self._dls(current, placed, joint, _ORIGIN, centre, reach)
            current = self._turn(current, joint, rotation)
            held = self.world(current)[joint][1]
        # Position wins: a hand turned further than the ball's limits allow
        # leaves the ball clamped and the point off target, so the whole
        # chain -- ball included -- closes the gap, moving as little as it
        # can from the wrist's answer (measured on the conformance arm, 160
        # grabs from rest onto poses inside the limits: 11 missed by over 1%
        # of the reach without this, 4 with it, each pinned against a limit;
        # a position-only grab misses 2).
        return self._dls(current, slots, joint, local_point, target, reach)

    def _rotations(self, joint: int) -> List[Tuple[int, str]]:
        return [(joint, c) for _s, c in self._slots[joint] if c in ROTATE_CHANNELS]

    def _reach(self, chain: Sequence[int], local_point: Sequence[float]) -> float:
        """The chain's length: its bones (every joint's rest offset below the
        first) plus the held point's lever -- the scale the solver's damping,
        tolerance and step are fractions of, so a rig in millimetres and one
        in metres solve alike."""
        length = self._norm(local_point)
        for index in chain[1:]:
            length += self._norm(self.joints[index]["t"])
        return max(length, self.EPS)

    def _turn(self, state: List[float], joint: int, rotation: Quat) -> List[float]:
        """*state* with *joint*'s rotation channels set so its link's rig-space
        rotation is *rotation*, each channel clamped (a ball's wrist half)."""
        spec = self.joints[joint]
        parent = spec["parent"]
        parent_q = _IDENTITY if parent is None else self.world(state)[parent][1]
        frame = self._qmul(parent_q, spec["q"])
        local = self._qmul(self._qconj(frame), rotation)
        out = list(state)
        angles = self._euler_for(
            spec["rotate_order"],
            self._qmatrix(local),
            ROTATE_CHANNELS,
            {c: out[s] for s, c in self._slots[joint]},
        )
        for slot, channel in self._slots[joint]:
            if channel in ROTATE_CHANNELS:
                out[slot] = self._clamp_slot(slot, angles[channel])
        return out

    def _column(
        self,
        world: Sequence[Tuple[Vec, Quat]],
        state: Sequence[float],
        slot: int,
        point: Sequence[float],
        reach: float,
    ) -> Vec:
        """How *point* moves per unit of slot *slot*: per RADIAN of a rotation
        channel, per *reach* of a slide (the solver's units, which put both on
        one length scale)."""
        joint, channel = self.channels[slot]
        spec = self.joints[joint]
        parent = spec["parent"]
        parent_q = _IDENTITY if parent is None else world[parent][1]
        frame = self._qmul(parent_q, spec["q"])
        if channel in TRANSLATE_CHANNELS:
            axis = [0.0, 0.0, 0.0]
            axis[TRANSLATE_CHANNELS.index(channel)] = 1.0
            return self._scale(self._qrot(frame, axis), reach)
        values = self._values(state, joint)
        wanted = _AXIS[channel[1]]
        # Outermost first: each rotation's axis is carried by those applied
        # after it (the ones nearer the parent).
        for c in reversed(spec["rotate_order"]):
            if _AXIS[c] == wanted:
                axis = [0.0, 0.0, 0.0]
                axis[wanted] = 1.0
                world_axis = self._qrot(frame, axis)
                return self._cross(world_axis, self._sub(point, world[joint][0]))
            frame = self._qmul(frame, self._qaxis(_AXIS[c], values.get("r" + c, 0.0)))
        return _ORIGIN  # unreachable: every channel's axis is in the order

    def _unit(self, slot: int, reach: float) -> float:
        """One solver unit of slot *slot* in channel units: degrees per radian
        of a rotation, parent units per reach of a slide."""
        return math.degrees(1.0) if self.channels[slot][1] in ROTATE_CHANNELS else reach

    def _step(
        self,
        columns: Sequence[Vec],
        weights: Sequence[float],
        error: Vec,
        damping: float,
    ) -> List[float]:
        """One weighted damped-least-squares step:
        ``du = W J^T (J W J^T + lambda^2 I)^-1 e``."""
        a = [[damping if r == c else 0.0 for c in range(3)] for r in range(3)]
        for col, w in zip(columns, weights):
            if w <= 0.0:
                continue
            for r in range(3):
                for c in range(3):
                    a[r][c] += w * col[r] * col[c]
        y = self._solve3(a, error)
        return [w * self._dot(col, y) for col, w in zip(columns, weights)]

    def _dls(
        self,
        state: List[float],
        slots: Sequence[int],
        joint: int,
        local_point: Sequence[float],
        target: Sequence[float],
        reach: float,
    ) -> List[float]:
        """Move *slots* to bring *local_point* on *joint* to *target*: damped
        least-squares steps, each halved (up to :attr:`HALVINGS` times) until
        it brings the point nearer -- none does, and the point is as near as
        the limits and the reach allow."""
        current = list(state)
        if not slots:
            return current
        damping = (self.DAMPING * reach) ** 2
        tolerance = self.TOLERANCE * reach
        step = self.MAX_STEP * reach
        world, point, distance = self._held(current, joint, local_point, target)
        for _ in range(self.ITERATIONS):
            if distance <= tolerance:
                break
            error = self._sub(target, point)
            if distance > step:
                error = self._scale(error, step / distance)
            columns = [self._column(world, current, s, point, reach) for s in slots]
            weights = [self._weights[s] for s in slots]
            du = self._step(columns, weights, error, damping)
            # A channel already on a limit and pushed further out is taken out
            # of THIS step, so the others make up for it rather than the step
            # being spent against a wall.
            blocked = False
            for i, slot in enumerate(slots):
                lo, hi = self._limits[slot]
                value = current[slot]
                if (lo is not None and value <= lo and du[i] < 0.0) or (
                    hi is not None and value >= hi and du[i] > 0.0
                ):
                    weights[i] = 0.0
                    blocked = True
            if blocked:
                du = self._step(columns, weights, error, damping)
            fraction = 1.0
            for _ in range(self.HALVINGS + 1):
                trial = list(current)
                for i, slot in enumerate(slots):
                    value = current[slot] + fraction * du[i] * self._unit(slot, reach)
                    trial[slot] = self._clamp_slot(slot, value)
                held = self._held(trial, joint, local_point, target)
                if held[2] < distance:
                    break
                fraction *= 0.5
            else:
                break  # no step brings it nearer
            current = trial
            world, point, distance = held
        return current

    def _held(
        self,
        state: Sequence[float],
        joint: int,
        local_point: Sequence[float],
        target: Sequence[float],
    ) -> Tuple[List[Tuple[Vec, Quat]], Vec, float]:
        """``(world, point, distance)``: the rig-space poses at *state*, where
        *local_point* on *joint* is, and how far it is from *target*."""
        world = self.world(state)
        p, q = world[joint]
        point = self._add(p, self._qrot(q, local_point))
        return world, point, self._norm(self._sub(target, point))
