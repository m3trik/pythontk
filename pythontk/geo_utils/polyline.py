# !/usr/bin/python
# coding=utf-8
"""Pure polyline / curve geometry — generate, measure, sample, reshape.

A *polyline* is just an ordered sequence of world-space points (open or closed).
Everything here is a stateless ``staticmethod`` / ``classmethod`` returning plain
values (lists of points, floats, frames), so it composes freely and is
unit-testable without any DCC. It is the reusable primitive under anything that
follows a line — a curtain rail, a sweep/loft profile path, a tube centerline, a
rope, a ribbon, a fence — so resolving a polyline from a DCC *selection* (edges,
NURBS CVs, locators) stays in the DCC adapter, which then feeds the points back
here.

Scalar/vector primitives (``lerp``, ``cross_product``, ``distance_between_points``,
``remap``, ``point_segment_distance``, …) live in :class:`pythontk.MathUtils`;
this class composes them into operations on point *sequences*.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

from pythontk.math_utils._math_utils import MathUtils

Vec = Tuple[float, float, float]

_lerp = MathUtils.lerp                # point/vector lerp
_unit = MathUtils.safe_normalize      # normalize with a degenerate fallback


class Polyline:
    """Stateless polyline/curve geometry (the line other tools follow).

    Methods take plain points and return plain values, so this composes with any
    consumer and is testable without building anything. Resolving a polyline from
    a DCC *selection* is the adapter's job (e.g. ``mayatk``'s ``Rail``).
    """

    # ----------------------------------------------------------- generators

    @staticmethod
    def make(
        width: float = 6.0,
        curvature: float = 0.0,
        segments: int = 24,
        closed: bool = False,
        center: Vec = (0.0, 0.0, 0.0),
    ) -> Tuple[List[Vec], bool]:
        """Build a default polyline: a straight line of ``width`` (``curvature == 0``).

        ``curvature`` (-1..1) bows the line by a parabola — positive forward in
        +Z, negative back in -Z — so the default is flat and bowing is opt-in.
        ``closed`` makes a ring instead. ``center`` (x, y, z) is where the line
        is centered — the straight line's midpoint / the ring's center (default
        origin). Returns ``(points, closed)``.
        """
        segments = max(2, int(segments))
        cx, cy, cz = (float(c) for c in center)
        pts: List[Vec] = []
        if closed:
            r = width / 2.0
            for i in range(segments):
                a = (i / segments) * 2.0 * math.pi
                pts.append((cx + r * math.sin(a), cy, cz + r * math.cos(a)))
        else:
            bow = curvature * width * 0.5
            for i in range(segments + 1):
                f = i / segments
                pts.append(
                    (
                        cx + (f - 0.5) * width,
                        cy,
                        cz + bow * (1.0 - (2.0 * f - 1.0) ** 2),
                    )
                )
        return pts, closed

    @classmethod
    def from_point_cloud(
        cls,
        points: Sequence,
        count: int,
        axis: Optional[int] = None,
        precision: Optional[int] = None,
    ) -> List[List[float]]:
        """Extract an ordered centerline polyline from a tube-shaped **point cloud**.

        Slices the cloud along its dominant axis into slabs and averages each (the
        ring-centers approach) — DCC-neutral geometry (pure Python); the adapters
        supply world-space vertices and build from the result. Axis-aligned
        slicing approximates a mildly-curved tube but degrades on one that doubles
        back along its own dominant axis (use an explicit edge selection there).
        Each slab center is already the mean of its ring's vertices (smooth by
        construction), so no extra smoothing pass is applied; raise *precision* to
        better resolve a curve.

        Parameters:
            points: Iterable of 3D points (``[x, y, z]`` sequences or ``.x/.y/.z``
                objects).
            count: Number of evenly-spaced centerline points to return (``>= 2``).
            axis: Force the slice axis (0/1/2); ``None`` auto-picks the longest
                bounding-box extent.
            precision: Number of sampling slabs (a higher value better captures a
                curve); ``None`` = ``max(count, 12)``. The slab centers are
                resampled to exactly *count* points.

        Returns:
            ``count`` ordered ``[x, y, z]`` points start->end, or ``[]`` if the
            cloud is degenerate (fewer than two non-empty slabs / zero extent
            along the axis).
        """
        def _xyz(p):
            return (p.x, p.y, p.z) if hasattr(p, "x") else (float(p[0]), float(p[1]), float(p[2]))

        pts = [_xyz(p) for p in points]
        if len(pts) < 2 or count < 2:
            return []

        mins = [min(p[i] for p in pts) for i in range(3)]
        maxs = [max(p[i] for p in pts) for i in range(3)]
        if axis is None:
            extents = [maxs[i] - mins[i] for i in range(3)]
            axis = extents.index(max(extents))
        lo, span = mins[axis], (maxs[axis] - mins[axis])
        if span <= 1e-9:
            return []

        slabs = max(count, 12) if precision is None else max(2, int(precision))
        bins: List[List[tuple]] = [[] for _ in range(slabs)]
        for p in pts:
            idx = min(slabs - 1, int((p[axis] - lo) / span * slabs))
            bins[idx].append(p)
        centers = [
            [sum(p[i] for p in b) / len(b) for i in range(3)] for b in bins if b
        ]
        if len(centers) < 2:
            return []
        return cls.resample(centers, count)

    @staticmethod
    def order_points(
        points: List[List[float]],
        closed_path: bool = False,
        distance_metric: Optional[Callable[[List[float], List[float]], float]] = None,
    ) -> List[List[float]]:
        """Order scattered points into a continuous path (greedy nearest-neighbour).

        Starts at the first point and repeatedly appends the nearest remaining
        one — turning an unordered set into a polyline. Resolving a DCC selection
        into points hands the cloud here to sequence it.

        Parameters:
            points (List): The points to order. Each entry may be a plain
                ``[x, y, z]`` sequence or an object exposing ``.x``, ``.y`` and
                ``.z`` (e.g. ``om.MPoint``, ``om.MVector``).
            closed_path (bool): Append the start point at the end (closed loop).
            distance_metric: Optional callable ``(p1, p2) -> float``. Defaults to
                Euclidean distance that handles both forms above.

        Returns:
            List: Ordered points (same element type as the input) forming a
            continuous path. The input sequence is not modified.
        """
        if not points:
            return []
        points = list(points)  # work on a copy; also accepts tuples

        if distance_metric is None:
            def distance_metric(p1, p2):
                # Branchless dispatch: prefer .x/.y/.z (MPoint/MVector/dt.Point),
                # fall back to subscripting (lists, tuples, numpy arrays).
                if hasattr(p1, "x"):
                    dx, dy, dz = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
                else:
                    dx, dy, dz = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
                return (dx * dx + dy * dy + dz * dz) ** 0.5

        sorted_points = [points.pop(0)]
        while points:
            last_point = sorted_points[-1]
            next_point = min(points, key=lambda p: distance_metric(p, last_point))
            sorted_points.append(next_point)
            points.remove(next_point)

        if closed_path:
            sorted_points.append(sorted_points[0])

        return sorted_points

    # ------------------------------------------------------------- measure

    @staticmethod
    def length(points: Sequence[Vec], closed: bool = False) -> float:
        """Total arc length of the polyline (wrapping last->first if closed)."""
        dist = MathUtils.distance_between_points
        total = sum(dist(points[i - 1], points[i]) for i in range(1, len(points)))
        if closed:
            total += dist(points[-1], points[0])
        return total

    @staticmethod
    def point_at(points: Sequence[Sequence[float]], t: float) -> List[float]:
        """Interpolated point at parameter ``t`` (0..1) along the polyline.

        ``t`` is in *index* space (uniform per segment, not arc-length), clamped
        to ``[0, 1]``; ``0`` is the first point, ``1`` the last.
        """
        total_length = len(points) - 1  # number of segments
        t = max(0, min(t, 1))
        index = int(t * total_length)

        if index == total_length:
            return list(points[-1])

        p1 = points[index]
        p2 = points[index + 1]
        frac = t * total_length - index
        return [p1[i] + (p2[i] - p1[i]) * frac for i in range(3)]

    # -------------------------------------------------------------- resample

    @classmethod
    def resample(
        cls,
        points: Sequence[Sequence[float]],
        count: int,
        reverse: bool = False,
        interpolation: Callable[[Sequence[Sequence[float]], float], List[float]] = None,
        start_offset: float = 0.0,
        end_offset: float = 0.0,
    ) -> List[List[float]]:
        """Distribute ``count`` evenly-spaced points along the polyline.

        Parameters:
            points: The polyline vertices.
            count: Number of points to distribute (``>= 2``).
            reverse: Reverse the order of the returned points.
            interpolation: Sampler ``(points, t) -> point``; defaults to
                :meth:`point_at`.
            start_offset: Skip this fraction (0..1) in from the start.
            end_offset: Skip this fraction (0..1) in from the end.

        Returns:
            The evenly-distributed ``[x, y, z]`` points.
        """
        if start_offset < 0 or end_offset < 0 or start_offset + end_offset >= 1:
            raise ValueError("Invalid start or end offset values.")

        if interpolation is None:
            interpolation = cls.point_at

        count = max(2, int(count))
        positions = [
            interpolation(
                points,
                MathUtils.remap(i, (0, count - 1), (start_offset, 1 - end_offset)),
            )
            for i in range(count)
        ]
        return positions[::-1] if reverse else positions

    # --------------------------------------------------------------- reshape

    @staticmethod
    def smooth(
        points: Sequence[Union[tuple, object]], window_size: int = 1
    ) -> list:
        """Moving-average smooth of a point sequence.

        Parameters:
            points: A sequence of (x, y, z) tuples or dt.Point objects.
            window_size: Number of points in each averaging window.

        Returns:
            A list of smoothed points in the same format as the input.
        """
        if not points or window_size <= 1:
            return list(points)

        n = len(points)
        window_size = min(window_size, n)
        smoothed_points = []

        is_dt_point = hasattr(points[0], "__add__") and hasattr(
            points[0], "__truediv__"
        )

        # Initialize running sum
        running_sum = points[0] * 0 if is_dt_point else (0.0, 0.0, 0.0)
        count = 0

        for i in range(n):
            # Add new point to the running sum
            new_point = points[i]
            running_sum = (
                running_sum + new_point
                if is_dt_point
                else tuple(running_sum[j] + new_point[j] for j in range(3))
            )
            count += 1

            # Remove the old point when the window slides forward
            if i >= window_size:
                old_point = points[i - window_size]
                running_sum = (
                    running_sum - old_point
                    if is_dt_point
                    else tuple(running_sum[j] - old_point[j] for j in range(3))
                )
                count -= 1

            # Compute and store the smoothed point
            avg_point = (
                running_sum / count
                if is_dt_point
                else tuple(v / count for v in running_sum)
            )
            smoothed_points.append(avg_point)

        return smoothed_points

    @staticmethod
    def simplify(points: Sequence[Sequence[float]], tolerance: float) -> List[int]:
        """Ramer-Douglas-Peucker indices: which points to keep to stay within
        ``tolerance`` of the original polyline.

        The classic curve-simplification / flattening algorithm. It keeps the
        endpoints, then recursively keeps the point of greatest deviation from
        the current chord while that deviation exceeds ``tolerance`` — so the
        retained points **concentrate where the polyline bends** (corners, tight
        arcs) and near-collinear runs collapse to their endpoints. Ideal for
        deciding *where to spend* samples/divisions along a curve.

        Iterative (explicit stack) so it is safe on very dense polylines. Points
        may be any fixed dimension (2D, 3D, …).

        Parameters:
            points (Sequence[Sequence[float]]): The ordered polyline vertices.
            tolerance (float): Max allowed deviation (same units as ``points``).
                Smaller keeps more points; ``<= 0`` keeps every point.

        Returns:
            (list) Sorted indices into ``points`` to keep (always includes the
            first and last). For ``< 3`` points, all indices are returned.
        """
        n = len(points)
        if n < 3:
            return list(range(n))
        if tolerance <= 0:
            return list(range(n))

        seg_dist = MathUtils.point_segment_distance
        keep = [False] * n
        keep[0] = keep[n - 1] = True
        stack = [(0, n - 1)]
        while stack:
            i, j = stack.pop()
            if j <= i + 1:
                continue
            a, b = points[i], points[j]
            d_max, idx = -1.0, -1
            for k in range(i + 1, j):
                d = seg_dist(points[k], a, b)
                if d > d_max:
                    d_max, idx = d, k
            if d_max > tolerance:
                keep[idx] = True
                stack.append((i, idx))
                stack.append((idx, j))
        return [i for i in range(n) if keep[i]]

    # ------------------------------------------------------------- framing

    @staticmethod
    def frames(
        points: Sequence[Vec],
        segments: int,
        closed: bool,
        up: Vec = (0.0, 1.0, 0.0),
    ) -> List[Tuple[Vec, Vec, Vec]]:
        """Resample to ``segments + 1`` even points with a local frame at each.

        Each frame is ``(position, tangent, normal)``; the ``normal`` is the
        in-plane perpendicular ``cross(up, tangent)`` — the direction a sweep
        bows/offsets along. ``up`` is the reference up-vector (default world +Y);
        the ``_unit`` fallback handles the degenerate case where the tangent is
        parallel to ``up`` (a vertical run / coincident points).
        """
        dist = MathUtils.distance_between_points
        pts = list(points)
        if closed and dist(pts[-1], pts[0]) > 1e-6:
            pts = pts + [pts[0]]

        cum = [0.0]
        for i in range(1, len(pts)):
            cum.append(cum[-1] + dist(pts[i - 1], pts[i]))
        total = cum[-1]

        def sample(s: float) -> Vec:
            s = max(0.0, min(total, s))
            for i in range(1, len(cum)):
                if s <= cum[i] or i == len(cum) - 1:
                    span = cum[i] - cum[i - 1]
                    t = (s - cum[i - 1]) / span if span > 1e-9 else 0.0
                    return _lerp(pts[i - 1], pts[i], t)
            return pts[-1]

        frames: List[Tuple[Vec, Vec, Vec]] = []
        eps = max(total * 1e-3, 1e-5)
        for c in range(segments + 1):
            s = (c / segments) * total if total > 0 else 0.0
            pos = sample(s)
            # get_vector_from_two_points(a, b) -> b - a, so this is the forward
            # tangent; _unit guards the degenerate (vertical / coincident) case.
            tan = _unit(
                MathUtils.get_vector_from_two_points(sample(s - eps), sample(s + eps)),
                (0.0, 0.0, 1.0),
            )
            normal = _unit(MathUtils.cross_product(up, tan), (1.0, 0.0, 0.0))
            frames.append((pos, tan, normal))
        return frames
