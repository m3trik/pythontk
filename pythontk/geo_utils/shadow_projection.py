# !/usr/bin/python
# coding=utf-8
"""Planar shadow projection — the geometry of a ground shadow, pure numpy, no DCC.

A shadow on the ground is the occluder projected onto the ground plane along
the light's rays: through the light's POSITION for a point, spot or area
source (a perspective projection — the shadow grows and stretches as the
source nears the occluder's height) and along one fixed DIRECTION for a sun
(a parallel projection — the shadow's length is height x cot(elevation),
wherever the sun sits). Both are the same map once the sun is written as a
source a very long way off, which is how :meth:`ShadowProjection.model`
treats it, so a live expression needs a single body for either.

Two levels, for the two things a shadow rig needs:

- :meth:`ShadowProjection.project` maps world points (mesh vertices) onto the
  ground exactly — the shape a texture is rasterized from. Each point also
  reports its penumbra *spread*: how much wider the source's finite size
  blurs that point's shadow, which grows with the occluder's distance from
  the ground (sharp at the contact, soft at the tip — the real thing).
- :meth:`ShadowProjection.model` reduces the occluder to its bounding
  cylinder (footprint radius, height) and returns the rectangle that
  cylinder's shadow spans, in a frame the plane is placed in: the anchor
  (where the base centre lands), the bearing (away from the light), and the
  near edge / length / width. Every term is a few clamps and ratios, so a
  Maya expression or a Blender driver evaluates it per frame; the DCCs mirror
  it symbolically and their tests pin the two against this reference.

A texture rasterized at one light position is exact only there. Between
rasterizations the plane follows the model via the canvas *fractions* the
rasterizer records: the canvas's near edge is stamped in projected-footprint
(base disk) radii from the anchor and its far edge in projected-head (top
disk) radii from where the head lands. Ground points project onto themselves
at any light height, so the near edge of a grounded target's shadow stays at
its feet while only the far edge follows the top's projection — a stamp
measured as a fraction of the whole length instead slid the texture away
from the feet as the shadow grew (the reported gap). So the shadow's
direction, reach and perspective growth track the light live, while the
silhouette inside is re-rendered on demand.

Axis convention: ``up`` is the index of the vertical axis (Maya ``1``,
Blender ``2``); the two horizontal axes are the remaining indices in order,
so a 2D horizontal coordinate is ``(x, z)`` in Maya and ``(x, y)`` in
Blender. The across-bearing axis is ``w = (u[1], -u[0])`` in both — the
plane's local +X in either DCC — which is what keeps one rasterized texture
readable by both.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np

Vec2 = Tuple[float, float]
Rect = Tuple[float, float, float, float]  # (u_lo, u_hi, w_lo, w_hi)


class ShadowModel(NamedTuple):
    """The analytic shadow of a bounding cylinder (see :meth:`ShadowProjection.model`).

    Lengths are along the bearing ``u`` (away from the light) and across it
    (``w``); both are measured from :attr:`anchor`, where the base disk's
    centre lands on the ground. The shadow runs from the projected base disk
    (radius :attr:`base`, centred on the anchor) to the projected top disk
    (radius :attr:`top`, centred :attr:`reach` further along ``u``).
    """

    #: Where the base-disk centre lands on the ground — 2D horizontal, world.
    anchor: Vec2
    #: Unit bearing ``u``: horizontal, from the light through the target.
    bearing: Vec2
    #: Base-disk projection factor: 1 while grounded, >1 as the target rises.
    k_base: float
    #: Top-disk factor — the perspective growth a near source gives the head
    #: of the shadow. Clamped so a source dropping toward the top stays finite.
    k_top: float
    #: How far past the anchor the top's projection lands, along ``u``.
    reach: float
    #: The base disk's projected radius (``radius x k_base``) — the footprint
    #: on the ground, centred on the anchor.
    base: float
    #: The top disk's projected radius (``radius x k_top``) — the head of the
    #: shadow, centred ``reach`` along ``u``.
    top: float
    #: Extent across ``u`` (``2 x radius x max(k_top, k_base)``).
    width: float
    #: True when the light had no horizontal bearing and the fallback was used.
    overhead: bool

    @property
    def near(self) -> float:
        """The near edge along ``u``, relative to the anchor (``-base``)."""
        return -self.base

    @property
    def length(self) -> float:
        """Extent along ``u`` (``reach + base + top``)."""
        return self.reach + self.base + self.top

    @property
    def across(self) -> Vec2:
        """Unit ``w`` — across the bearing, the plane's local +X."""
        return (self.bearing[1], -self.bearing[0])

    def rect(self, fractions: Sequence[float]) -> Rect:
        """The canvas rectangle *fractions* denote at this model, absolute in
        the ``(u, w)`` frame — the inverse of :meth:`ShadowProjection.fractions`.

        ``u0`` is the near edge in base radii from the anchor (``-1`` = the
        footprint's near side), ``u1`` the far edge in top radii from the
        head's centre (``+1`` = its far side); ``w0`` / ``w1`` are fractions of
        the width from the centre line. Each disk's factor scales its own
        edge, so a grounded target's near edge (``k_base = 1``) is pinned to
        its feet at every light height.
        """
        u0, u1, w0, w1 = (float(f) for f in fractions)
        return (
            u0 * self.base,
            self.reach + u1 * self.top,
            w0 * self.width,
            w1 * self.width,
        )

    def placement(self, fractions: Sequence[float]) -> Tuple[Vec2, float, float]:
        """Where a plane carrying a canvas of *fractions* sits at this model:
        ``(centre 2D world, extent along u, extent along w)``."""
        u_lo, u_hi, w_lo, w_hi = self.rect(fractions)
        cu, cw = 0.5 * (u_lo + u_hi), 0.5 * (w_lo + w_hi)
        ux, uz = self.bearing
        wx, wz = self.across
        centre = (
            self.anchor[0] + ux * cu + wx * cw,
            self.anchor[1] + uz * cu + wz * cw,
        )
        return centre, u_hi - u_lo, w_hi - w_lo


class ShadowProjection:
    """Planar shadow projection: exact per-point mapping plus the live model."""

    #: ``u`` when the light is straight overhead: the second horizontal axis
    #: (Maya +Z, Blender +Y). The DCC expressions use the same fallback.
    OVERHEAD_BEARING: Vec2 = (0.0, 1.0)
    #: How many object sizes back a directional source is placed when it is
    #: written as a point — far enough that its rays are parallel to well
    #: under a milliradian, near enough to stay in double precision.
    FAR_FACTOR = 1.0e6
    #: Default cap on the shadow's reach, in object heights (a sun at
    #: ``atan(1/6)`` = 9.5 deg elevation); the DCC ``maxStretch`` attr.
    DEFAULT_MAX_STRETCH = 6.0
    EPS = 1.0e-6

    @staticmethod
    def horizontal_axes(up: int = 1) -> Tuple[int, int]:
        """The two horizontal axis indices, in order, for the vertical *up*."""
        return tuple(i for i in range(3) if i != int(up))  # type: ignore[return-value]

    @staticmethod
    def _unit(vec) -> np.ndarray:
        v = np.asarray(vec, dtype=float).reshape(3)
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            raise ValueError("ShadowProjection: a zero-length direction.")
        return v / n

    @classmethod
    def far_point(cls, contact, direction, scale: float) -> Tuple[float, float, float]:
        """A directional source written as a point: *scale* x :attr:`FAR_FACTOR`
        back along *direction* (the way the light shines) from *contact*."""
        d = cls._unit(direction)
        far = cls.FAR_FACTOR * max(float(scale), 1e-3)
        c = np.asarray(contact, dtype=float).reshape(3)
        return tuple(float(v) for v in (c - d * far))

    # ---------------------------------------------------------------- model
    @classmethod
    def model(
        cls,
        contact,
        light=None,
        ground: float = 0.0,
        radius: float = 0.5,
        height: float = 1.0,
        *,
        up: int = 1,
        direction=None,
        max_stretch: Optional[float] = None,
    ) -> ShadowModel:
        """The shadow of the bounding cylinder standing on *contact*.

        The cylinder has its base disk (radius *radius*) centred on *contact*
        and its top disk *height* above it. Each disk projects through the
        light onto the ground at its own factor ``k = (L - G) / (L - disk
        height)``; the shadow runs from the base disk's projection (the
        anchor) to the top disk's, which lands ``reach`` further along the
        bearing and ``k_top`` times larger. Clamps keep the result finite as
        the source drops toward the top: the reach never exceeds
        *max_stretch* object heights and no factor exceeds ``1 + max_stretch``.

        Parameters:
            contact: World point at the footprint centre, on the target's
                underside (the DCC contact locator).
            light: World position of a positional source. Ignored when
                *direction* is given.
            ground: Height of the ground plane along the up axis.
            radius: Footprint radius (half the footprint's diagonal).
            height: Target height above *contact*.
            up: Index of the vertical axis (Maya 1, Blender 2).
            direction: Unit direction a directional source shines along;
                written as a far point (:meth:`far_point`).
            max_stretch: Reach cap in object heights (default
                :attr:`DEFAULT_MAX_STRETCH`).

        Returns:
            :class:`ShadowModel`.
        """
        if max_stretch is None:
            max_stretch = cls.DEFAULT_MAX_STRETCH
        c = np.asarray(contact, dtype=float).reshape(3)
        if direction is not None:
            light = cls.far_point(c, direction, max(height, 2.0 * radius))
        if light is None:
            raise ValueError(
                "ShadowProjection.model: a light position or direction is required."
            )
        light_v = np.asarray(light, dtype=float).reshape(3)
        a, b = cls.horizontal_axes(up)
        ch, lh = c[[a, b]], light_v[[a, b]]
        cu, lu = float(c[up]), float(light_v[up])
        g = float(ground)
        d = ch - lh
        dist = float(math.hypot(d[0], d[1]))
        if dist > cls.EPS:
            bearing = (float(d[0] / dist), float(d[1] / dist))
            overhead = False
        else:
            bearing = cls.OVERHEAD_BEARING
            overhead = True

        k_max = 1.0 + float(max_stretch)
        k_base = min(max((lu - g) / max(1e-4, lu - cu), 0.0), k_max)
        k_cap = min(k_max, k_base + float(max_stretch) * height / max(dist, cls.EPS))
        k_top = min(max((lu - g) / max(1e-4, lu - cu - height), 0.0), k_cap)
        reach = max(0.0, dist * (k_top - k_base))
        width = 2.0 * radius * max(k_top, k_base)
        anchor = (float(lh[0] + d[0] * k_base), float(lh[1] + d[1] * k_base))
        return ShadowModel(
            anchor=anchor,
            bearing=bearing,
            k_base=float(k_base),
            k_top=float(k_top),
            reach=float(reach),
            base=float(radius * k_base),
            top=float(radius * k_top),
            width=float(width),
            overhead=overhead,
        )

    # -------------------------------------------------------------- project
    @classmethod
    def project(
        cls,
        points,
        light=None,
        ground: float = 0.0,
        *,
        up: int = 1,
        direction=None,
        max_length: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Project world *points* onto the ground plane along the light's rays.

        Parameters:
            points: ``(N, 3)`` world points.
            light: World position of a positional source (ignored with
                *direction*).
            ground: Height of the ground plane along the up axis.
            up: Index of the vertical axis.
            direction: Unit direction a directional source shines along.
            max_length: Cap on any point's horizontal displacement — a
                source level with a point would otherwise send its shadow to
                infinity. Always pass one when the result is rasterized.

        Returns:
            ``(ground, spread)`` — ``ground`` the ``(N, 2)`` horizontal
            coordinates where each point's shadow lands, ``spread`` the
            ``(N,)`` penumbra growth: multiply by a positional source's
            diameter, or a directional source's angular diameter in radians,
            for the penumbra width (world units) at that point's shadow.
            ``None`` when the source casts nothing onto the ground (a
            position at or below it; a direction not pointing down).
            Points below the ground stay where they are.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        a, b = cls.horizontal_axes(up)
        ph, pu = pts[:, [a, b]], pts[:, up]
        g = float(ground)
        if direction is not None:
            d = cls._unit(direction)
            du = float(d[up])
            if du >= -cls.EPS:
                return None
            ray = np.maximum(0.0, (pu - g) / -du)  # ray length to the ground
            disp = ray[:, None] * d[[a, b]][None, :]
            spread = ray
        else:
            if light is None:
                raise ValueError(
                    "ShadowProjection.project: a light position or direction is required."
                )
            light_v = np.asarray(light, dtype=float).reshape(3)
            lh, lu = light_v[[a, b]], float(light_v[up])
            if lu <= g + cls.EPS:
                return None
            denom = lu - pu
            # A point level with (or above) the source has no finite shadow:
            # cap its factor, then let max_length bound the displacement.
            safe = np.where(denom > cls.EPS, denom, cls.EPS)
            t = np.minimum((lu - g) / safe, 1.0e6)
            t = np.maximum(t, 1.0)  # below the ground: cast nowhere, stay put
            disp = (ph - lh[None, :]) * (t - 1.0)[:, None]
            spread = t - 1.0
        if max_length is not None:
            cap = max(float(max_length), 0.0)
            mag = np.hypot(disp[:, 0], disp[:, 1])
            over = mag > cap
            if over.any():
                scale = np.where(over, cap / np.where(over, mag, 1.0), 1.0)
                disp = disp * scale[:, None]
                spread = spread * scale
        return ph + disp, spread

    # ----------------------------------------------------------------- frame
    @staticmethod
    def to_frame(ground_points, model: ShadowModel) -> np.ndarray:
        """``(N, 2)`` ground coordinates -> ``(u, w)`` relative to the model's
        anchor along its bearing and across it."""
        p = np.asarray(ground_points, dtype=float).reshape(-1, 2)
        rel = p - np.asarray(model.anchor, dtype=float)[None, :]
        ux, uz = model.bearing
        wx, wz = model.across
        return np.column_stack(
            [rel[:, 0] * ux + rel[:, 1] * uz, rel[:, 0] * wx + rel[:, 1] * wz]
        )

    @staticmethod
    def fractions(rect: Rect, model: ShadowModel) -> Tuple[float, float, float, float]:
        """Express a ``(u, w)`` canvas *rect* as the stamp a plane carries so a
        live expression re-places it at any light position — the near edge in
        base-disk radii from the anchor, the far edge in top-disk radii from
        the head's centre, the across-extents as fractions of the width
        (:meth:`ShadowModel.rect` inverts it, and explains why)."""
        u_lo, u_hi, w_lo, w_hi = (float(v) for v in rect)
        base = max(model.base, 1e-9)
        top = max(model.top, 1e-9)
        width = max(model.width, 1e-9)
        return (
            u_lo / base,
            (u_hi - model.reach) / top,
            w_lo / width,
            w_hi / width,
        )


class ShadowRaster(NamedTuple):
    """What a rasterized shadow texture was drawn into (``ImgUtils.rasterize_shadow``)."""

    #: The live model at the light position the texture was rendered from.
    model: ShadowModel
    #: The canvas the texture covers, absolute in the model's ``(u, w)`` frame.
    rect: Rect
    #: The canvas as fractions of the model (:meth:`ShadowProjection.fractions`).
    fractions: Tuple[float, float, float, float]
    #: The widest penumbra drawn, world units (0 for a sizeless source).
    penumbra: float
