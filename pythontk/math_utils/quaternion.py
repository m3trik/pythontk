# !/usr/bin/python
# coding=utf-8
"""Rotation math on plain tuples -- quaternions, rotation matrices and Euler
angles in ONE convention, the one every consumer here shares.

- A quaternion is ``(x, y, z, w)`` -- the glTF, Unity and ``MQuaternion``
  layout.
- ``multiply(a, b)`` is ``a (x) b``: ``b`` applied first, so a child's world
  rotation is ``multiply(parent, local)``.
- A matrix is a 3 x 3 list of ROWS for COLUMN vectors: ``v' = M v``.
- Euler angles are degrees per axis, ``(x, y, z)``, in Maya's rotate order:
  ``"xyz"`` applies x first, so the rotation is ``Rz Ry Rx`` -- FBX's
  ``eEulerXYZ`` too.

Zero-dependency and written to port: the articulated rig's web and Unity
runtimes carry ports of the same arithmetic, pinned by golden cases, so a step
changed here is a step changed in every port.
"""

import math
from typing import Callable, List, Sequence, Tuple, Union

Quat = Tuple[float, float, float, float]
Vec = Tuple[float, float, float]


class Quaternion:
    """Unit-quaternion rotation math on ``(x, y, z, w)`` tuples (module
    docstring: the convention).

    Example:
        >>> q = Quaternion.from_euler("zxy", (10.0, 20.0, 30.0))
        >>> m = Quaternion.to_matrix(q)
        >>> [round(v, 9) for v in Quaternion.matrix_euler("zxy", m)]
        [10.0, 20.0, 30.0]
    """

    IDENTITY: Quat = (0.0, 0.0, 0.0, 1.0)
    #: An Euler order's letters -> axis index.
    AXES = {"x": 0, "y": 1, "z": 2}
    #: Under this norm a quaternion is no rotation (:meth:`normalize`).
    DEGENERATE = 1.0e-12
    #: Above this ``|dot|`` two rotations are interpolated linearly
    #: (:meth:`slerp`): ``sin(theta)`` is too small to divide by.
    PARALLEL = 0.9995
    #: Under this ``cos`` of the middle angle an Euler triple is at gimbal
    #: lock (:meth:`matrix_euler`).
    GIMBAL = 1.0e-6

    @staticmethod
    def multiply(a: Sequence[float], b: Sequence[float]) -> Quat:
        """``a (x) b``: the rotation *b* followed by *a*."""
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    @staticmethod
    def conjugate(q: Sequence[float]) -> Quat:
        """The inverse of the unit quaternion *q*."""
        return (-q[0], -q[1], -q[2], q[3])

    @classmethod
    def normalize(cls, q: Sequence[float]) -> Quat:
        """*q* at unit length; :attr:`IDENTITY` for one too short to be a
        rotation (under :attr:`DEGENERATE`)."""
        n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if n < cls.DEGENERATE:
            return cls.IDENTITY
        return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)

    @staticmethod
    def rotate(q: Sequence[float], v: Sequence[float]) -> Vec:
        """*v* rotated by the unit quaternion *q*."""
        ux, uy, uz, w = q[0], q[1], q[2], q[3]
        tx = (uy * v[2] - uz * v[1]) * 2.0
        ty = (uz * v[0] - ux * v[2]) * 2.0
        tz = (ux * v[1] - uy * v[0]) * 2.0
        return (
            (v[0] + tx * w) + (uy * tz - uz * ty),
            (v[1] + ty * w) + (uz * tx - ux * tz),
            (v[2] + tz * w) + (ux * ty - uy * tx),
        )

    @classmethod
    def from_axis_angle(cls, axis: Union[int, Sequence[float]], degrees: float) -> Quat:
        """A rotation of *degrees* about *axis*: an axis index (0, 1, 2 for
        x, y, z -- the other components exactly 0) or a 3-vector (normalized
        here; a zero one is no rotation)."""
        half = math.radians(degrees) * 0.5
        s = math.sin(half)
        if isinstance(axis, int):
            return (
                s if axis == 0 else 0.0,
                s if axis == 1 else 0.0,
                s if axis == 2 else 0.0,
                math.cos(half),
            )
        n = math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
        if n < cls.DEGENERATE:
            return cls.IDENTITY
        return (axis[0] / n * s, axis[1] / n * s, axis[2] / n * s, math.cos(half))

    @classmethod
    def from_euler(cls, order: str, degrees: Sequence[float]) -> Quat:
        """The rotation of Euler *degrees* ``(x, y, z)`` in Maya's *order*:
        ``"xyz"`` applies x first, so it is ``Rz (x) Ry (x) Rx``. *order* may
        name fewer axes (``"z"``): those it leaves out do not turn."""
        q = cls.IDENTITY
        for c in order:  # innermost first; each later one is applied after
            axis = cls.AXES[c]
            q = cls.multiply(cls.from_axis_angle(axis, degrees[axis]), q)
        return q

    @staticmethod
    def to_matrix(q: Sequence[float]) -> List[List[float]]:
        """The rotation matrix of the unit quaternion *q* (rows, column
        vectors). Normalize a quaternion read from a file first."""
        x, y, z, w = q
        return [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]

    @staticmethod
    def from_matrix(m: Sequence[Sequence[float]]) -> Quat:
        """The unit quaternion of rotation matrix *m* (rows, column vectors);
        Shepperd's method, so no branch divides by a small number."""
        trace = m[0][0] + m[1][1] + m[2][2]
        if trace > 0.0:
            s = 2.0 * math.sqrt(trace + 1.0)
            return (
                (m[2][1] - m[1][2]) / s,
                (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s,
                0.25 * s,
            )
        i = max(range(3), key=lambda k: m[k][k])
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + m[i][i] - m[j][j] - m[k][k])
        q = [0.0, 0.0, 0.0, (m[k][j] - m[j][k]) / s]
        q[i], q[j], q[k] = 0.25 * s, (m[j][i] + m[i][j]) / s, (m[k][i] + m[i][k]) / s
        return (q[0], q[1], q[2], q[3])

    @classmethod
    def matrix_euler(cls, order: str, m: Sequence[Sequence[float]]) -> Vec:
        """Euler degrees ``(x, y, z)`` of rotation matrix *m* in Maya's
        *order* -- :meth:`from_euler` composes them back to *m*.

        *order* ``o0 o1 o2`` composes ``M = R_o2 R_o1 R_o0``. With ``a, b, c =
        o2, o1, o0`` and ``s`` the parity of ``(a, b, c)`` (+1 cyclic), the
        middle angle is ``asin(s M[a][c])`` -- always in [-90, 90] -- and the
        outer two come from the row and column it leaves. At gimbal lock
        (``cos`` of the middle under :attr:`GIMBAL`) the innermost is 0 and
        the outermost takes the whole turn. Of a rotation's two triples this
        is the one with the middle in range; the other is ``(A + 180, 180 -
        B, C + 180)`` in order.
        """
        a, b, c = cls.AXES[order[2]], cls.AXES[order[1]], cls.AXES[order[0]]
        s = 1.0 if (b - a) % 3 == 1 else -1.0
        sin_b = max(-1.0, min(1.0, s * m[a][c]))
        angle_b = math.asin(sin_b)
        if math.cos(angle_b) > cls.GIMBAL:
            angle_a = math.atan2(-s * m[b][c], m[c][c])
            angle_c = math.atan2(-s * m[a][b], m[a][a])
        else:
            angle_c = 0.0
            angle_a = math.atan2(s * m[c][b], m[b][b])
        out = [0.0, 0.0, 0.0]
        out[a] = math.degrees(angle_a)
        out[b] = math.degrees(angle_b)
        out[c] = math.degrees(angle_c)
        return (out[0], out[1], out[2])

    @classmethod
    def arc(cls, a: Sequence[float], b: Sequence[float]) -> Callable[[float], Quat]:
        """The shorter arc from *a* to *b*, as ``u -> rotation``: :meth:`slerp`
        with the work of the pair done once, for a caller reading one pair at
        many *u* (measured: a third off each read)."""
        dot = sum(x * y for x, y in zip(a, b))
        if dot < 0.0:
            b = tuple(-x for x in b)
            dot = -dot
        if dot > cls.PARALLEL:  # nearly parallel: lerp, then renormalize

            def at(u: float) -> Quat:
                out = tuple(x + (y - x) * u for x, y in zip(a, b))
                norm = math.sqrt(sum(x * x for x in out)) or 1.0
                return tuple(x / norm for x in out)  # type: ignore[return-value]

            return at
        theta = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta = math.sin(theta)

        def at(u: float) -> Quat:
            wa = math.sin((1.0 - u) * theta) / sin_theta
            wb = math.sin(u * theta) / sin_theta
            out = tuple(x * wa + y * wb for x, y in zip(a, b))
            norm = math.sqrt(sum(x * x for x in out)) or 1.0
            return tuple(x / norm for x in out)  # type: ignore[return-value]

        return at

    @classmethod
    def slerp(cls, a: Sequence[float], b: Sequence[float], u: float) -> Quat:
        """The rotation *u* of the way from *a* to *b* along the shorter arc,
        unit length -- glTF's LINEAR rule for a rotation channel. Nearly
        parallel rotations (``|dot|`` over :attr:`PARALLEL`) interpolate
        linearly, then normalize."""
        return cls.arc(a, b)(u)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
