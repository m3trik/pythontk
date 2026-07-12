# !/usr/bin/python
# coding=utf-8
"""Rail-driven parametric surface — a general geometry primitive.

:class:`RailSurface` turns a guide *rail* (any :class:`~pythontk.geo_utils.polyline.Polyline`)
into a ``(u_segs+1) × (v_segs+1)`` grid of points, applying a caller-supplied
**displacement field** at each grid vertex. It owns only the invariant machinery
— measure the rail, build oriented frames along it, and walk the grid — and knows
nothing about *what* the surface represents. The domain (a hanging curtain, a
draped banner, a ribbon, a terrain strip along a path…) lives entirely in the
``displace`` callable the caller plugs in.

This is the reusable substrate under curve-driven surface generators such as
the ``CurtainDrape`` engine vendored in the DCC packages
(``mayatk``/``blendertk`` ``edit_utils._curtain_drape``): they resolve their
own resolution + precompute their own feature state, then hand this primitive
a displacement closure. A new generator is a new displacement — no new
machinery.

Pure geometry, no DCC: it emits plain vertex positions; building the mesh from
them is the adapter's job. Sits beside :class:`Polyline` (which it consumes to
frame the rail) and :class:`PointCloud` in ``geo_utils``.

    surface = RailSurface(rail, u_segs=64, v_segs=16)
    u, v, pts = surface.grid_points(lambda u, v, pos, tan, nrm: (
        pos[0], pos[1] - (1.0 - v) * drop, pos[2]     # a plain vertical drop
    ))
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from pythontk.geo_utils.polyline import Polyline

Vec = Tuple[float, float, float]
Frame = Tuple[Vec, Vec, Vec]                      # (position, tangent, normal)
Displace = Callable[[float, float, Vec, Vec, Vec], Vec]


class RailSurface:
    """A parametric grid spanning from a rail, displaced by a caller field.

    Parameters:
        rail: Ordered world-space points the surface is framed along.
        u_segs: Segment count along the rail (``u``); the grid has ``u_segs + 1``
            columns. ``u`` runs ``0 → 1`` head-to-tail.
        v_segs: Segment count across the span (``v``); ``v_segs + 1`` rows.
            ``v`` runs ``0 → 1`` — the caller's ``displace`` decides what the two
            edges mean (e.g. hem ↔ rail for a curtain).
        closed: Treat the rail as a closed loop (wraps the frames).

    The oriented :attr:`frames` (``(pos, tan, normal)`` per column, from
    :meth:`Polyline.frames`) and the rail :attr:`length` are precomputed once at
    construction; :meth:`grid_points` reuses them across every row.
    """

    def __init__(
        self,
        rail: Sequence[Vec],
        u_segs: int,
        v_segs: int,
        closed: bool = False,
    ):
        rail = [tuple(float(c) for c in p) for p in rail]
        if len(rail) < 2:
            raise ValueError("rail must contain at least two points.")
        self.rail = rail
        self.u_segs = max(1, int(u_segs))
        self.v_segs = max(1, int(v_segs))
        self.closed = bool(closed)
        self.length: float = Polyline.length(self.rail, self.closed)
        self.frames: List[Frame] = Polyline.frames(self.rail, self.u_segs, self.closed)

    def grid_points(self, displace: Displace) -> Tuple[int, int, List[Vec]]:
        """Return ``(u_segs, v_segs, points)`` — the displaced grid, row-major.

        *displace* is called once per vertex as
        ``displace(u, v, pos, tan, normal) -> (x, y, z)`` and returns the final
        world position of that vertex. ``points`` is row-major over
        ``(v_segs + 1)`` rows of ``(u_segs + 1)`` columns —
        ``points[row * (u_segs + 1) + col]``.
        """
        u_segs, v_segs, frames = self.u_segs, self.v_segs, self.frames
        pts: List[Vec] = []
        for r in range(v_segs + 1):
            v = r / v_segs
            for c in range(u_segs + 1):
                pos, tan, normal = frames[c]
                pts.append(displace(c / u_segs, v, pos, tan, normal))
        return u_segs, v_segs, pts


__all__ = ["RailSurface"]
