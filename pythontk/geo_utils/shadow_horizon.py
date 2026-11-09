# !/usr/bin/python
# coding=utf-8
"""Height-field shadow maps: a ground shadow that follows the light at runtime.

The projected shadow (:class:`~pythontk.ShadowProjection`) draws one silhouette
for one source position. A **horizon rig** instead ships a map an engine can
ask "is this ground texel in shadow?" for *any* source position: the prop's
footprint rasterised into per-pixel solid vertical **spans** (a depth-peeled
height field), which the shader **marches** -- the ray from the texel toward
the source, through the columns it crosses, blocked where its height falls
inside a span. A ray march, not a lookup.

It replaced the coverage-aware horizon map (per texel, per azimuth bin, one
elevation interval and an occupancy mask per layer) on 2026-09-05, measured on
a 192-pixel canvas against the exact projection at three lights: that map
disagreed 3 % on a box, 7 % on a table, 8 % on a chair (20 % under a grazing
light), 14 % on a lamp, 19 % on a stool with stretchers and 22 % on an arch --
thin members' shadows drawn as bin-quantised combs, a seat over a stretcher
filled solid because one interval per column per layer cannot hold two. The
march on the same canvas: 1, 2, 2, 1.6, 2.8 and 5 %, at a bake of 0.05 s
instead of 4 s and 128 x 512 texels instead of 1024 x 512.

Per footprint pixel (``S x S`` over the prop's footprint, in the target's own
frame: origin at the contact, ``up`` vertical, the two horizontal axes of
:meth:`ShadowProjection.horizontal_axes`) the map keeps:

* up to ``K`` solid spans ``[lo, hi]``, heights above the ground plane at
  16 bits;
* for the penumbra, the distance to the nearest solid column and that
  column's hull;
* and, for the shader, a min/max **pyramid** over the columns: the ray skips
  a whole cell it can prove it clears, and descends only near the occluder
  (measured: 7-16 steps per ray on average, against 300-550 for a plain
  half-pixel march).

Everything lands in ONE RGBA8 image, ``S`` tall and ``S x (K + 2)`` wide --
the ``K`` span tiles, the aux tile, the pyramid tile with its levels stacked
top to bottom (:meth:`HeightFieldMap.to_rgba`).

:meth:`HeightFieldMap.alpha` is the reference every engine shader is pinned
to: an exact walk over the level-0 pixels the ray crosses. The shader's
quadtree traversal skips only cells it can prove empty for the ray, so it
visits the same pixels and returns the same alpha. :meth:`ShadowHorizon.measure`
scores a map against :meth:`ImgUtils.rasterize_shadow`, the exact projection,
at random source positions, and :meth:`ShadowHorizon.bake_adaptive` doubles
the footprint resolution until that disagreement is under a threshold.
"""

from __future__ import annotations

import base64
import json
import math
import warnings
import zlib
from importlib import resources
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from pythontk.core_utils.deprecation import Deprecation
from pythontk.geo_utils.shadow_projection import ShadowProjection

__all__ = ["HeightFieldMap", "HorizonMap", "ShadowHorizon", "ShadowHorizonConformance"]

_TWO_PI = 2.0 * math.pi
#: The 16-bit range a height or a distance is quantised over.
MAX16 = 65535.0
#: Fixed-point scale of the distance field: 1/64 of a footprint pixel.
DIST_SCALE = 64.0
#: The distance an empty map reports everywhere (the field's largest value).
DIST_FAR = MAX16 / DIST_SCALE
#: Fixed-point scale of the interior distance field: 1/4 of a pixel, 8-bit.
INTERIOR_SCALE = 4.0
#: The deepest interior distance the field holds (pixels); deeper is all one
#: to the penumbra -- a ray buried that far sees none of any source.
INTERIOR_FAR = 255.0 / INTERIOR_SCALE
#: The marker texel (pyramid tile, ``(S/2, S/2)``) that says a map carries the
#: interior field: "SH", version 1, 0xA5. A map baked before it has zeros
#: there, and the shader then takes the inner penumbra from heights alone.
INTERIOR_MARKER = (83, 72, 1, 165)


class HeightFieldMap(NamedTuple):
    """A baked height-field shadow map and the frame it was baked in.

    Heights are above the ground plane, in frame units, already quantised
    the way the PNG stores them (16-bit over ``height_scale``; the nearest
    column's hull 8-bit; the distance field in 1/64 pixel), so a map and its
    PNG round trip evaluate alike: decoded fields within 1.2e-7 of the
    height scale and the alpha within 1.2e-6 (float32 decode; measured over
    the conformance cases of four seeds), never byte-identical on a re-pack,
    whose pyramid re-derives from the 8-bit hull and can sit one code below
    the bake's conservative ceiling. ``NaN`` marks an absent span.
    """

    size: int  #: footprint pixels per side, ``S`` (a power of two)
    spans: int  #: solid spans per column, ``K``
    bounds: Tuple[float, float, float, float]  #: ``(a0, a1, b0, b1)`` in the frame
    ground: float  #: height of the ground plane along ``up`` in the frame
    up: int  #: vertical axis index (1 for Y-up, 2 for Z-up)
    height_scale: float  #: the height a 16-bit channel value of 65535 stands for
    lo: np.ndarray  #: ``(K, S, S)`` span bottoms, indexed ``[k, ib, ia]``
    hi: np.ndarray  #: ``(K, S, S)`` span tops
    dist: np.ndarray  #: ``(S, S)`` distance to the nearest column, in the smaller pitch
    near_lo: np.ndarray  #: ``(S, S)`` that column's hull bottom (its own on a solid)
    near_hi: np.ndarray  #: ``(S, S)`` that column's hull top
    #: ``(S, S)`` the distance from a solid pixel to the nearest empty one, in
    #: the smaller pitch (0 on an empty pixel), quantised to 1/4 pixel; how far
    #: a blocked ray must turn sideways to clear the column. ``None`` on a map
    #: baked before it (the inner penumbra then comes from heights alone).
    interior: Optional[np.ndarray] = None

    # -- layout ------------------------------------------------------------
    @property
    def levels(self) -> int:
        """Pyramid levels above level 0: ``log2(S)`` (the top is one cell)."""
        return int(round(math.log2(self.size)))

    @property
    def tiles(self) -> int:
        """Tiles across the image: ``K`` spans, the aux tile, the pyramid."""
        return self.spans + 2

    @property
    def pixel(self) -> Tuple[float, float]:
        """A footprint pixel's extent along the two horizontal axes."""
        a0, a1, b0, b1 = self.bounds
        return (a1 - a0) / self.size, (b1 - b0) / self.size

    @property
    def aspect(self) -> float:
        """The larger pixel pitch over the smaller, ``>= 1``. The distance
        field is measured in the smaller pitch, so one pixel step along the
        other axis is this many units: the half-pixel slack the penumbra
        takes off a distance, and the one-pixel reach of its bilinear read,
        scale by it."""
        pw, ph = self.pixel
        lo, hi = min(pw, ph), max(pw, ph)
        return hi / lo if lo > 1e-12 else 1.0

    def hull(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(lo, hi)`` ``(S, S)``: each column's outermost span bounds
        (``NaN`` where empty)."""
        return np.fmin.reduce(self.lo, axis=0), np.fmax.reduce(self.hi, axis=0)

    def pyramid(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per level ``1 .. levels``, ``(lo, hi, dist)`` over ``(n, n)``
        cells of ``2^level`` pixels: the cell's least distance -- zero
        exactly when it holds a solid column -- and, for such a cell, the
        hull of every column a ray through it can be scored against: the
        nearest column of each of its pixels **and of their one-pixel ring**,
        which is what :meth:`_march` reads (a solid pixel its own spans, an
        empty one the nearest hull bilinearly from the four pixels around the
        ray's midpoint). The shader skips a column cell only when the ray
        clears this hull by the source's margin, so the bound has to hold
        for every one of those reads; a hull dilated by whole cells did not
        (a neighbour pixel's nearest column can lie a cell and a half away).
        ``NaN`` where the cell holds no column: the shader never consults
        the hull there, only the distance."""
        lo = self._dilate(self.near_lo, np.fmin)
        hi = self._dilate(self.near_hi, np.fmax)
        dist = self.dist
        out = []
        while lo.shape[0] > 1:
            n = lo.shape[0] // 2
            lo = np.fmin.reduce(lo.reshape(n, 2, n, 2), axis=(1, 3))
            hi = np.fmax.reduce(hi.reshape(n, 2, n, 2), axis=(1, 3))
            dist = dist.reshape(n, 2, n, 2).min(axis=(1, 3))
            column = dist <= 0.0
            out.append(
                (
                    np.where(column, lo, np.nan),
                    np.where(column, hi, np.nan),
                    dist,
                )
            )
        return out

    @staticmethod
    def _dilate(field: np.ndarray, extreme: np.ufunc) -> np.ndarray:
        """*field* reduced by *extreme* (``np.fmin`` / ``np.fmax``: a ``NaN``
        ignored, kept where all are) over each pixel's 3 x 3 neighbourhood."""
        h, w = field.shape
        p = np.pad(field, 1, constant_values=np.nan)
        return extreme.reduce(
            np.stack([p[i : i + h, j : j + w] for i in range(3) for j in range(3)]),
            axis=0,
        )

    # -- encoding ----------------------------------------------------------
    def to_rgba(self) -> np.ndarray:
        """The map as one ``uint8`` RGBA image ``(S, S x (K + 2), 4)``.

        Span tile ``k``: ``R, G`` the 16-bit ``lo`` (high byte, low byte),
        ``B, A`` the 16-bit ``hi``; an absent span is all zero (``hi == 0``
        is the flag: a span that never rises above the ground blocks nothing
        and is dropped at bake). Aux tile: ``R, G`` the 16-bit distance in
        1/64 pixel, ``B`` the nearest column's hull top and ``A`` its bottom,
        8-bit. Pyramid tile: level ``l`` occupies ``S >> l`` rows from row
        ``S - (S >> (l - 1))`` and as many columns, ``R`` the bottom and
        ``G`` the top of the cell's hull (:meth:`pyramid`; 8-bit, floor and
        ceil, ``G == 0`` = the cell holds no column), ``B, A`` the cell's
        least distance. The pyramid's levels fill the tile's left half; its
        right half carries the :attr:`interior` field, 8-bit in 1/4 pixel:
        texel ``(S/2 + x/2, y/2)`` holds the four pixels it covers as ``R``
        (even ``x``, even ``y``), ``G`` (odd, even), ``B`` (even, odd), ``A``
        (odd, odd), and texel ``(S/2, S/2)`` holds :data:`INTERIOR_MARKER`.
        Row ``0`` of every tile is ``ib = 0``, the frame's ``b0`` edge --
        data, never flipped.
        """
        S, K, hs = self.size, self.spans, self.height_scale
        img = np.zeros((S, S * self.tiles, 4), np.uint8)

        def split16(value):
            v = np.clip(np.round(value), 0, MAX16).astype(np.int64)
            return (v // 256).astype(np.uint8), (v % 256).astype(np.uint8)

        for k in range(K):
            lo16 = np.where(np.isnan(self.lo[k]), 0.0, self.lo[k] / hs * MAX16)
            hi16 = np.where(np.isnan(self.hi[k]), 0.0, self.hi[k] / hs * MAX16)
            tile = img[:, k * S : (k + 1) * S]
            tile[..., 0], tile[..., 1] = split16(lo16)
            tile[..., 2], tile[..., 3] = split16(hi16)
        aux = img[:, K * S : (K + 1) * S]
        aux[..., 0], aux[..., 1] = split16(self.dist * DIST_SCALE)
        aux[..., 2] = np.clip(np.round(self.near_hi / hs * 255.0), 0, 255).astype(
            np.uint8
        )
        aux[..., 3] = np.clip(np.round(self.near_lo / hs * 255.0), 0, 255).astype(
            np.uint8
        )
        pyr = img[:, (K + 1) * S : (K + 2) * S]
        for level, (lo, hi, dist) in enumerate(self.pyramid(), start=1):
            n = S >> level
            y0 = S - (S >> (level - 1))
            cell = pyr[y0 : y0 + n, :n]
            empty = np.isnan(hi)
            cell[..., 0] = np.where(
                empty, 0, np.clip(np.floor(np.nan_to_num(lo) / hs * 255.0), 0, 255)
            ).astype(np.uint8)
            cell[..., 1] = np.where(
                empty, 0, np.clip(np.ceil(np.nan_to_num(hi) / hs * 255.0), 1, 255)
            ).astype(np.uint8)
            cell[..., 2], cell[..., 3] = split16(dist * DIST_SCALE)
        if self.interior is not None:
            q = np.clip(np.round(self.interior * INTERIOR_SCALE), 0, 255).astype(
                np.uint8
            )
            h = S // 2
            block = pyr[:h, h:]
            block[..., 0] = q[0::2, 0::2]
            block[..., 1] = q[0::2, 1::2]
            block[..., 2] = q[1::2, 0::2]
            block[..., 3] = q[1::2, 1::2]
            pyr[h, h] = INTERIOR_MARKER
        return img

    @classmethod
    def from_rgba(
        cls,
        rgba: np.ndarray,
        *,
        size: int,
        spans: int,
        bounds: Sequence[float],
        ground: float,
        up: int,
        height_scale: float,
    ) -> "HeightFieldMap":
        """A map decoded from the image :meth:`to_rgba` wrote (or a copy of
        it -- the pyramid is re-derived, never trusted)."""
        img = np.asarray(rgba)
        if img.ndim != 3 or img.shape[2] != 4:
            raise ValueError("HeightFieldMap.from_rgba: expected an (H, W, 4) image.")
        S, K, hs = int(size), int(spans), float(height_scale)
        if S < 2 or S & (S - 1) or K < 1:
            raise ValueError(
                f"HeightFieldMap.from_rgba: size must be a power of two and spans "
                f"at least 1 (the pyramid halves the footprint down to one cell); "
                f"got size {S}, spans {K}."
            )
        if img.shape[0] < S or img.shape[1] < S * (K + 2):
            raise ValueError(
                f"HeightFieldMap.from_rgba: a {S} x {S * (K + 2)} image is needed "
                f"for size {S} and {K} spans; got {img.shape[1]} x {img.shape[0]}."
            )
        f = img.astype(np.float64)

        def join16(hi_byte, lo_byte):
            return hi_byte * 256.0 + lo_byte

        lo = np.full((K, S, S), np.nan, np.float32)
        hi = np.full((K, S, S), np.nan, np.float32)
        for k in range(K):
            tile = f[:S, k * S : (k + 1) * S]
            v_lo = join16(tile[..., 0], tile[..., 1])
            v_hi = join16(tile[..., 2], tile[..., 3])
            present = v_hi > 0
            lo[k] = np.where(present, v_lo / MAX16 * hs, np.nan)
            hi[k] = np.where(present, v_hi / MAX16 * hs, np.nan)
        aux = f[:S, K * S : (K + 1) * S]
        dist = (join16(aux[..., 0], aux[..., 1]) / DIST_SCALE).astype(np.float32)
        near_hi = (aux[..., 2] / 255.0 * hs).astype(np.float32)
        near_lo = (aux[..., 3] / 255.0 * hs).astype(np.float32)
        pyr = img[:S, (K + 1) * S : (K + 2) * S]
        h = S // 2
        interior = None
        if tuple(int(v) for v in pyr[h, h]) == INTERIOR_MARKER:
            block = pyr[:h, h:].astype(np.float32) / INTERIOR_SCALE
            interior = np.zeros((S, S), np.float32)
            interior[0::2, 0::2] = block[..., 0]
            interior[0::2, 1::2] = block[..., 1]
            interior[1::2, 0::2] = block[..., 2]
            interior[1::2, 1::2] = block[..., 3]
        return cls(
            S,
            K,
            tuple(float(v) for v in bounds),
            float(ground),
            int(up),
            hs,
            lo,
            hi,
            dist,
            near_lo,
            near_hi,
            interior,
        )

    # -- the reference -------------------------------------------------------
    def alpha(
        self,
        points,
        light=None,
        *,
        direction=None,
        source_size: float = 0.0,
        source_angle: float = 0.0,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """Shadow alpha ``(N,)`` at frame *points* for one source -- the
        reference the engine shaders are pinned to.

        A point's height is not merely ignored, it is **replaced by**
        :attr:`ground` before the ray is formed, and a shader must do the
        same: the spans are heights above the ground plane and the ray
        starts on it. A fragment sits above that datum (the DCC rigs lift
        the plane by their own ``GROUND_OFFSET``), so a ray formed from the
        fragment would climb from the wrong height.

        Parameters:
            points: ``(N, 3)`` points in the map's frame.
            light: Positional source in the frame (ignored with *direction*).
            direction: Unit direction a directional source shines along.
            source_size: Diameter of a positional source, frame units.
            source_angle: Angular diameter of a directional source, radians.
            intensity: Multiplier on the result.

        Returns:
            ``(N,)`` float32 in ``[0, 1]``: the share of the source's disc the
            prop hides -- 1 buried, 1/2 on the hard shadow's edge, 0 clear;
            for a sizeless source 1 where the ray toward it crosses a span
            and 0 elsewhere; 0 where the source is below the ground.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        a, b = ShadowProjection.horizontal_axes(self.up)
        n = pts.shape[0]
        out = np.zeros(n, dtype=np.float32)
        if n == 0:
            return out
        if direction is not None:
            d = -np.asarray(direction, dtype=float).reshape(3)
            norm = float(np.linalg.norm(d))
            if norm < 1e-12:
                return out
            d = d / norm
            if d[self.up] <= 0.0:
                return out
            L = np.broadcast_to(d, (n, 3)).astype(float)
            rho = np.full(n, 0.5 * float(source_angle))
        else:
            if light is None:
                raise ValueError(
                    "HeightFieldMap.alpha: a light position or direction is required."
                )
            lv = np.asarray(light, dtype=float).reshape(3)
            ground_pts = pts.copy()
            ground_pts[:, self.up] = self.ground
            L = lv[None, :] - ground_pts
            dist = np.linalg.norm(L, axis=1)
            dist = np.where(dist > 1e-12, dist, 1e-12)
            L = L / dist[:, None]
            rho = np.arcsin(np.clip(0.5 * float(source_size) / dist, 0.0, 1.0))
        horiz = np.hypot(L[:, a], L[:, b])
        live = (horiz > 1e-9) & (L[:, self.up] > 0.0)
        if not live.any():
            return out
        origin = pts[:, [a, b]]
        horiz_safe = np.where(live, horiz, 1.0)
        direction2 = np.column_stack([L[:, a], L[:, b]]) / horiz_safe[:, None]
        slope = np.where(live, L[:, self.up] / horiz_safe, 0.0)
        alpha = self._march(origin, direction2, slope, rho, horiz_safe)
        out[:] = np.where(live, np.clip(alpha * float(intensity), 0.0, 1.0), 0.0)
        return out

    def _march(self, origin, direction, slope, rho, cos_el=None) -> np.ndarray:
        """The exact walk: ``(N,)`` alpha for rays from the frame-horizontal
        *origin* ``(N, 2)`` along the horizontal unit *direction* ``(N, 2)``,
        rising *slope* per unit of horizontal travel, against a source of
        angular radius *rho* ``(N,)``; *cos_el* ``(N,)`` is the cosine of
        each ray's elevation (1 when omitted).

        Every level-0 pixel the ray crosses is visited (Amanatides-Woo, all
        rays at once). Angles are true angles at the ray's origin: the ray
        runs ``t / cos_el`` to a point ``t`` out, so a gap ``h`` above or
        below it subtends ``h cos_el^2 / t`` and a gap ``l`` beside it
        ``l cos_el / t``. A pixel the ray passes contributes its clearance --
        on a solid pixel the vertical gap to its spans, on an empty one the
        lateral distance to the nearest column combined with the vertical
        distance to that column's hull. A blocked ray's depth is the least
        turn that frees it: up, over every span above it anywhere along the
        walk (tilting up only moves where a ray meets a thin seat, so its
        thickness is no escape), or sideways, out of the deepest column it
        crosses by the :attr:`interior` distance -- which also finds a
        slab's nearest edge. The alpha is :meth:`disc` of the least
        clearance, or of minus the depth, over ``rho``: the share of the
        source's disc hidden, so the penumbra straddles the hard edge and a
        member thinner than it greys out instead of staying black. A blocked
        ray stops once it is buried ``rho`` deep every way.
        """
        S, K = self.size, self.spans
        a0, a1, b0, b1 = self.bounds
        pw, ph = self.pixel
        px = min(pw, ph)
        aspect = self.aspect
        n = origin.shape[0]
        c = np.ones(n) if cos_el is None else np.asarray(cos_el, dtype=float)
        c2 = c * c
        rho = np.broadcast_to(np.asarray(rho, dtype=float), (n,))
        ox = (origin[:, 0] - a0) / pw
        oy = (origin[:, 1] - b0) / ph
        dx = direction[:, 0] / pw  # pixels per unit t
        dy = direction[:, 1] / ph
        zx = np.abs(dx) < 1e-12
        zy = np.abs(dy) < 1e-12
        dx_s = np.where(zx, 1.0, dx)
        dy_s = np.where(zy, 1.0, dy)
        inx = (ox >= 0.0) & (ox < S)
        iny = (oy >= 0.0) & (oy < S)
        tx0, tx1 = (0.0 - ox) / dx_s, (S - ox) / dx_s
        ty0, ty1 = (0.0 - oy) / dy_s, (S - oy) / dy_s
        tx_lo = np.where(zx, np.where(inx, -np.inf, np.inf), np.minimum(tx0, tx1))
        tx_hi = np.where(zx, np.where(inx, np.inf, -np.inf), np.maximum(tx0, tx1))
        ty_lo = np.where(zy, np.where(iny, -np.inf, np.inf), np.minimum(ty0, ty1))
        ty_hi = np.where(zy, np.where(iny, np.inf, -np.inf), np.maximum(ty0, ty1))
        t_enter = np.maximum(np.maximum(tx_lo, ty_lo), 0.0)
        t_exit = np.minimum(tx_hi, ty_hi)
        active = t_exit > t_enter
        t = t_enter.copy()
        eps = 1e-6
        cx = np.floor(ox + dx * t + np.sign(dx) * eps).astype(np.int64)
        cy = np.floor(oy + dy * t + np.sign(dy) * eps).astype(np.int64)
        stepx = np.where(dx > 0, 1, np.where(dx < 0, -1, 0))
        stepy = np.where(dy > 0, 1, np.where(dy < 0, -1, 0))
        tdx = np.where(zx, np.inf, 1.0 / np.abs(dx_s))
        tdy = np.where(zy, np.inf, 1.0 / np.abs(dy_s))
        tmx = np.where(zx, np.inf, (np.where(dx > 0, cx + 1, cx) - ox) / dx_s)
        tmy = np.where(zy, np.inf, (np.where(dy > 0, cy + 1, cy) - oy) / dy_s)
        hit = np.zeros(n, bool)
        clear = np.full(n, np.inf)
        rise = np.zeros(n)
        side = np.zeros(n)
        for _ in range(2 * S + 4):
            act = active.copy()
            if not act.any():
                break
            te = np.minimum(np.minimum(tmx, tmy), t_exit)
            h0, h1 = slope * t, slope * te
            inb = (cx >= 0) & (cx < S) & (cy >= 0) & (cy < S)
            cxc, cyc = np.clip(cx, 0, S - 1), np.clip(cy, 0, S - 1)
            gap = np.full(n, np.inf)
            solid = np.zeros(n, bool)
            blocked = np.zeros(n, bool)
            up = np.full(n, -1.0)
            for k in range(K):
                lo_k, hi_k = self.lo[k, cyc, cxc], self.hi[k, cyc, cxc]
                has = inb & ~np.isnan(lo_k)
                overlap = has & (h1 >= lo_k) & (h0 <= hi_k)
                blocked |= overlap
                solid |= has
                up = np.where(has & (hi_k >= h0), np.maximum(up, hi_k - h0), up)
                g = np.where(h0 > hi_k, h0 - hi_k, lo_k - h1)
                gap = np.where(has & ~overlap, np.minimum(gap, g), gap)
            empty = inb & ~solid
            tmid = np.maximum(0.5 * (t + te), 1e-9)
            rise = np.where(act & (up >= 0.0), np.maximum(rise, up * c2 / tmid), rise)
            # The distance fields and the nearest column's hull, read
            # BILINEARLY at the ray's midpoint through the pixel: per-pixel
            # values would make the penumbra jump a quarter of its width
            # when a boundary-hugging ray lands one pixel over on a GPU's
            # float32 (measured: 0.21 against 0.07 at one viewer sample).
            # Inside and out alike, so the two meet at zero on the edge. Each
            # corner clamps to the tile on its own (clamp-to-edge), so the read
            # stays continuous across the outermost texel centres: clamped as
            # a pair, the half pixel inside the low edge read the next texel.
            mx_ = ox + dx * tmid - 0.5
            my_ = oy + dy * tmid - 0.5
            fx_, fy_ = np.floor(mx_), np.floor(my_)
            wx = np.clip(mx_ - fx_, 0.0, 1.0)
            wy = np.clip(my_ - fy_, 0.0, 1.0)
            ix0 = np.clip(fx_.astype(np.int64), 0, S - 1)
            iy0 = np.clip(fy_.astype(np.int64), 0, S - 1)
            ix1 = np.clip(fx_.astype(np.int64) + 1, 0, S - 1)
            iy1 = np.clip(fy_.astype(np.int64) + 1, 0, S - 1)

            def bilerp(field):
                top = field[iy0, ix0] * (1.0 - wx) + field[iy0, ix1] * wx
                bottom = field[iy1, ix0] * (1.0 - wx) + field[iy1, ix1] * wx
                return top * (1.0 - wy) + bottom * wy

            # the half pixel from a pixel's centre to its edge, in the
            # smaller pitch: the larger pitch is ``aspect`` of them
            lateral = np.maximum(bilerp(self.dist) - 0.5 * aspect, 0.0) * px * c
            nlo, nhi = bilerp(self.near_lo), bilerp(self.near_hi)
            vertical = np.where(h0 > nhi, h0 - nhi, np.where(h1 < nlo, nlo - h1, 0.0))
            gap = np.where(empty, np.hypot(lateral, vertical * c2), gap * c2)
            clear = np.where(
                act & ~hit & ~blocked, np.minimum(clear, gap / tmid), clear
            )
            if self.interior is not None:
                inner = bilerp(self.interior)
                turn = np.maximum(inner - 0.5 * aspect, 0.0) * px * c / tmid
            else:
                turn = np.full(n, np.inf)
            side = np.where(act & blocked, np.maximum(side, turn), side)
            hit |= act & blocked
            mx = tmx <= tmy
            my = tmy <= tmx
            t = np.where(act, te, t)
            cx = np.where(act & mx, cx + stepx, cx)
            tmx = np.where(act & mx, tmx + tdx, tmx)
            cy = np.where(act & my, cy + stepy, cy)
            tmy = np.where(act & my, tmy + tdy, tmy)
            depth = np.minimum(rise, side)
            active &= (t < t_exit - 1e-12) & ~(hit & (depth >= rho))
        rho_s = np.maximum(rho, 1e-12)
        depth = np.minimum(rise, side)
        soft = self.disc(np.where(hit, -depth / rho_s, clear / rho_s))
        return np.where(rho > 1e-12, soft, np.where(hit, 1.0, 0.0)).astype(np.float32)

    @staticmethod
    def disc(s) -> np.ndarray:
        """The share of a disc source a straight edge hides when the disc's
        centre lies *s* radii past it: 0 at ``s >= 1`` (clear), 1/2 at
        ``s = 0`` (the hard shadow's edge), 1 at ``s <= -1`` (buried) -- the
        circular segment's area, the S-shaped falloff of a round source."""
        s = np.clip(np.asarray(s, dtype=float), -1.0, 1.0)
        return (np.arccos(s) - s * np.sqrt(1.0 - s * s)) / math.pi


#: The former map type's name, for one release: the height-field map took
#: over its role on 2026-09-05 (see the module doc); nothing decodes the old
#: encoding any more.
HorizonMap = HeightFieldMap


class ShadowHorizon:
    """Bake, measure and lay out height-field shadow maps (module doc)."""

    #: Footprint pixels per side; 128 keeps a 5 cm leg one pixel wide on a
    #: 1.2 m table (measured: 2 % disagreement with the exact projection).
    DEFAULT_SIZE = 128
    #: Solid spans per column: two hold a seat over a stretcher, a shelf over
    #: a drawer; a column with more merges its smallest gaps first.
    DEFAULT_SPANS = 2
    #: Margin around the geometry on every side, as a fraction of the larger
    #: of its horizontal extent and its height. The march runs only inside
    #: the footprint, so this is also how far a penumbra can reach sideways
    #: past the prop's outline -- a ray that never enters the footprint is lit
    #: outright -- and a penumbra's reach grows with the HEIGHT the shadow
    #: falls from: padded by its width alone, a 5 cm pole's penumbra was cut
    #: at 5 mm and drew a quarter of its shadow too light.
    DEFAULT_PADDING = 0.1
    #: Retired 2026-10-20 (removed in 0.18.0): the ``shadows`` record's
    #: version states the map's one representation and encoding.
    ENCODING = Deprecation.class_attribute(
        2, "SceneRecords.SHADOWS.version", remove_in="0.18.0", since="2026-10-24"
    )
    MAPPING = Deprecation.class_attribute(
        "heightfield",
        "SceneRecords.SHADOWS.version",
        remove_in="0.18.0",
        since="2026-10-24",
    )
    #: Footprint sizes :meth:`bake_adaptive` walks through.
    ADAPTIVE_SIZES = (128, 256)

    #: The shared shader body, beside this module. One text, every consumer.
    SHADER_FILE = "shadow_horizon.glsl"
    #: Languages :meth:`shader_source` will spell the body in, and the macro
    #: that switches it. GLSL is the authored form, so it adds nothing.
    SHADER_LANGUAGES = {"glsl": "", "hlsl": "#define SH_HLSL 1\n"}

    @classmethod
    def shader_source(cls, language: str = "glsl") -> str:
        """The shared shadow evaluation, spelled for *language*.

        The returned text defines ``ShAlpha`` and its helpers and nothing
        else: it declares no uniform, samples no texture and names no engine.
        A host prepends its own uniforms and defines ``SH_Fetch`` (see the
        body's header), then pastes this in.

        Parameters:
            language: ``"glsl"`` or ``"hlsl"`` — a *language*, never an engine.
                Two engines that speak one language get one text; the
                per-engine part is ``SH_Fetch`` and the uniform block, which
                live with the engine. A ``shader_source("unity")`` would put
                the fork back where this exists to remove it.

        Returns:
            The body with the language's prologue on front, newline-terminated
            and with LF endings whatever the checkout's are — a host splices
            it into its own text, and CRLF creeping in reads as drift.

        Raises:
            ValueError: *language* is not one of :attr:`SHADER_LANGUAGES`.
        """
        key = str(language).lower()
        try:
            prologue = cls.SHADER_LANGUAGES[key]
        except KeyError:
            raise ValueError(
                f"ShadowHorizon.shader_source: unknown language {language!r}; "
                f"expected one of {sorted(cls.SHADER_LANGUAGES)}."
            ) from None
        # ``importlib.resources`` rather than ``__file__``: the body ships as
        # package data, and a wheel install has no source tree to walk.
        body = (
            resources.files("pythontk.geo_utils")
            .joinpath(cls.SHADER_FILE)
            .read_text(encoding="utf-8")
        )
        return prologue + body.replace("\r\n", "\n")

    # -- the record ------------------------------------------------------------
    @classmethod
    @Deprecation.parameter(
        "levels",
        remove_in="0.18.0",
        drop=True,
        reason="A map's pyramid levels are log2(size).",
        since="2026-10-24",
    )
    def record(
        cls,
        *,
        texture: str,
        size: int,
        spans: int,
        bounds: Sequence[float],
        height_scale: float,
        frame_a: Sequence[float],
        frame_b: Sequence[float],
        rect: Sequence[float],
    ) -> Dict[str, object]:
        """The ``horizon`` block of a ``shadows`` plane record
        (``ShadowPlaneHorizon``).

        One builder, in the package whose GLB writer and viewer read the
        block, so the DCC rigs (which stamp these values on the plane and
        publish them at export) cannot spell the schema two ways. It states
        the layout and nothing a reader derives: the pyramid's levels are
        log2(size), and the representation and encoding are the record
        version's.

        Parameters:
            texture: The map's file name -- its atlas when packed.
            size, spans: The map's layout (:class:`HeightFieldMap`).
            bounds: ``(a0, a1, b0, b1)`` footprint in the contact frame.
            height_scale: The height a 16-bit value of 65535 stands for.
            frame_a, frame_b: The frame's horizontal axes in exporter axes.
            rect: ``(su, sv, ou, ov)`` UV rect of the map in its texture.

        Returns:
            The block.
        """
        return {
            "texture": str(texture),
            "size": int(size),
            "spans": int(spans),
            "bounds": [round(float(v), 6) for v in bounds],
            "height_scale": round(float(height_scale), 6),
            "frame_a": list(frame_a),
            "frame_b": list(frame_b),
            "rect": list(rect),
        }

    # -- bake ----------------------------------------------------------------
    @classmethod
    def bake(
        cls,
        meshes,
        *,
        ground: float = 0.0,
        up: int = 1,
        size: int = DEFAULT_SIZE,
        spans: int = DEFAULT_SPANS,
        bounds=None,
        padding: float = DEFAULT_PADDING,
    ) -> HeightFieldMap:
        """Bake the map of *meshes* in the map's frame.

        Parameters:
            meshes: Iterable of ``(points, tris)`` — ``(N, 3)`` points in the
                frame (the contact at the horizontal origin), ``(M, 3)``
                triangles.
            ground: Height of the ground plane along *up* in the frame.
            up: Vertical axis index (1 for Y-up, 2 for Z-up).
            size: Footprint pixels per side; rounded up to a power of two
                (the pyramid halves it down to one cell).
            spans: Solid spans kept per column.
            bounds: ``(a0, a1, b0, b1)`` footprint to rasterise; fitted to the
                geometry with *padding* when None. It need not be square:
                an elongated prop keeps its full resolution on the short
                axis, and the distance field is measured in the smaller
                pixel pitch with the other axis weighted by
                :attr:`HeightFieldMap.aspect` (a square footprint had cost
                a shelf and an arch half their short-axis texels, measured
                as 2 to 4 % disagreement instead of 2 to 3).
            padding: Margin around the geometry, every side, as a fraction
                of the larger of its horizontal extent and its height
                (:attr:`DEFAULT_PADDING`).

        Returns:
            The :class:`HeightFieldMap`.
        """
        from pythontk.img_utils._img_utils import ImgUtils

        meshes = [
            (
                np.asarray(p, dtype=float).reshape(-1, 3),
                np.asarray(t, dtype=np.int64).reshape(-1, 3),
            )
            for p, t in meshes
        ]
        meshes = [(p, t) for p, t in meshes if len(p) and len(t)]
        if not meshes:
            raise ValueError("ShadowHorizon.bake: no geometry provided.")
        size = 1 << max(3, int(math.ceil(math.log2(max(int(size), 1)))))
        spans = max(int(spans), 1)
        corners = np.concatenate([p[t].reshape(-1, 3) for p, t in meshes])
        tall = float(corners[:, up].max()) - float(ground)
        if bounds is None:
            a, b = ShadowProjection.horizontal_axes(up)
            lo_a, hi_a = float(corners[:, a].min()), float(corners[:, a].max())
            lo_b, hi_b = float(corners[:, b].min()), float(corners[:, b].max())
            pad = float(padding) * max(hi_a - lo_a, hi_b - lo_b, tall, 1e-3) + 1e-6
            bounds = (lo_a - pad, hi_a + pad, lo_b - pad, hi_b + pad)
        # A gap no wider than one step of the 16-bit height code at the
        # geometry's height is no daylight: it encodes as touching, and two
        # parts meeting face to face (an awning's post under its slab) come
        # through a DCC's export and an engine's import a float apart
        # (measured 2026-11-08: 1.2e-7 m between an awning's post and slab).
        merge_gap = max(tall, 1e-3) / MAX16
        lo, hi, bounds = ImgUtils.rasterize_height_spans(
            meshes,
            up=up,
            size=size,
            ground=ground,
            bounds=bounds,
            spans=spans,
            merge_gap=merge_gap,
        )
        top = float(np.fmax.reduce(hi, axis=None))  # NaN: no column at all
        height_scale = max(top, 1e-3) if np.isfinite(top) else 1e-3
        # Quantised as the PNG will hold them, conservatively: bottoms down,
        # tops up, so no span thins out of a ray that met it in float.
        lo_q = np.floor(lo / height_scale * MAX16) / MAX16 * height_scale
        hi_q = np.ceil(hi / height_scale * MAX16) / MAX16 * height_scale
        gone = ~np.isnan(hi_q) & (hi_q <= 0.0)
        lo_q[gone] = np.nan
        hi_q[gone] = np.nan
        lo_q = lo_q.astype(np.float32)
        hi_q = hi_q.astype(np.float32)
        hull_lo = np.fmin.reduce(lo_q, axis=0)
        hull_hi = np.fmax.reduce(hi_q, axis=0)
        solid = ~np.isnan(hull_hi)
        # distances in the smaller pixel pitch, the other axis weighted up
        a0, a1, b0, b1 = (float(v) for v in bounds)
        pw, ph = (a1 - a0) / size, (b1 - b0) / size
        pitch = max(min(pw, ph), 1e-12)
        dist, near_lo, near_hi = cls._distance_field(
            solid, hull_lo, hull_hi, weights=(pw / pitch, ph / pitch)
        )
        dist = np.minimum(np.round(dist * DIST_SCALE) / DIST_SCALE, DIST_FAR)
        # and from every solid pixel to the nearest EMPTY one: the sideways
        # turn that frees a blocked ray (the inner penumbra)
        interior = cls._distance_field(
            ~solid, hull_lo, hull_hi, weights=(pw / pitch, ph / pitch)
        )[0]
        interior = np.where(
            solid,
            np.minimum(
                np.round(interior * INTERIOR_SCALE) / INTERIOR_SCALE, INTERIOR_FAR
            ),
            0.0,
        )
        near_lo = np.floor(near_lo / height_scale * 255.0) / 255.0 * height_scale
        near_hi = np.ceil(near_hi / height_scale * 255.0) / 255.0 * height_scale
        return HeightFieldMap(
            size,
            spans,
            tuple(float(v) for v in bounds),
            float(ground),
            int(up),
            float(height_scale),
            lo_q,
            hi_q,
            dist.astype(np.float32),
            near_lo.astype(np.float32),
            near_hi.astype(np.float32),
            interior.astype(np.float32),
        )

    @staticmethod
    def _distance_field(
        solid: np.ndarray,
        hull_lo: np.ndarray,
        hull_hi: np.ndarray,
        weights: Tuple[float, float] = (1.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(dist, near_lo, near_hi)`` ``(S, S)``: the Euclidean distance
        (pixel centres) from every pixel to the nearest solid one, and that
        pixel's hull. *weights* ``(wx, wy)`` scale a step along columns and
        along rows -- the pixel pitches over the smaller of the two, so the
        distance comes out in that pitch whatever the footprint's aspect.
        Exact, separable: a row's nearest solid column first, then the best
        row per pixel, in row blocks that keep the ``(block, S, S)`` working
        array small."""
        S = int(solid.shape[0])
        if not solid.any():
            return (
                np.full((S, S), DIST_FAR, np.float32),
                np.zeros((S, S), np.float32),
                np.zeros((S, S), np.float32),
            )
        wx, wy = (float(w) for w in weights)
        big = 4 * S * S
        # nearest solid column within each row, sweeping right then left
        arg = np.full((S, S), -1, np.int64)
        last = np.full(S, -big, np.int64)
        for x in range(S):
            last = np.where(solid[:, x], x, last)
            arg[:, x] = last
        fwd = np.where(arg >= 0, np.arange(S)[None, :] - arg, big)
        arg_b = np.full((S, S), -1, np.int64)
        nxt = np.full(S, 2 * big, np.int64)
        for x in range(S - 1, -1, -1):
            nxt = np.where(solid[:, x], x, nxt)
            arg_b[:, x] = nxt
        bwd = np.where(arg_b < S, arg_b - np.arange(S)[None, :], big)
        row_arg = np.where(fwd <= bwd, arg, arg_b)
        row_d2 = (np.minimum(fwd, bwd).astype(np.float64) * wx) ** 2
        # the best row for every pixel: ((y - k) wy)^2 + row_d2[k, x]
        ks = np.arange(S)
        dist = np.zeros((S, S), np.float32)
        best_row = np.zeros((S, S), np.int64)
        block = max(1, min(S, (1 << 22) // (S * S)))
        for y0 in range(0, S, block):
            ys = np.arange(y0, min(S, y0 + block))
            d2 = ((ys[:, None] - ks[None, :]).astype(np.float64) * wy) ** 2
            total = d2[:, :, None] + row_d2[None, :, :]  # (block, S, S)
            k_best = np.argmin(total, axis=1)
            dist[ys] = np.sqrt(
                np.take_along_axis(total, k_best[:, None, :], axis=1)[:, 0, :]
            )
            best_row[ys] = k_best
        xs = np.arange(S)[None, :]
        best_col = row_arg[best_row, np.broadcast_to(xs, (S, S))]
        near_lo = np.nan_to_num(hull_lo)[best_row, best_col].astype(np.float32)
        near_hi = np.nan_to_num(hull_hi)[best_row, best_col].astype(np.float32)
        dist[solid] = 0.0
        return dist, near_lo, near_hi

    # -- measure ---------------------------------------------------------------
    @classmethod
    def measure(
        cls,
        hmap: HeightFieldMap,
        meshes,
        *,
        samples: int = 8,
        size: int = 256,
        seed: int = 0,
        max_stretch: Optional[float] = None,
        radius: Optional[float] = None,
        height: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compare :meth:`HeightFieldMap.alpha` with the exact projection at
        random source positions.

        For each sample :meth:`ImgUtils.rasterize_shadow` draws the hard
        shadow, every canvas texel is mapped back to the frame, and the two
        binary masks are compared over the canvas.

        Returns:
            ``{"mean", "max", "tolerant_mean", "tolerant_max", "samples"}`` —
            the disagreeing fraction of the masks' union, averaged and at its
            worst, raw and with a one-texel edge tolerance (a thin shadow's
            raw score is dominated by sub-texel registration; the tolerant
            score is what :meth:`bake_adaptive` gates on).
        """
        from pythontk.img_utils._img_utils import ImgUtils

        meshes = [
            (
                np.asarray(p, dtype=float).reshape(-1, 3),
                np.asarray(t, dtype=np.int64).reshape(-1, 3),
            )
            for p, t in meshes
        ]
        a, b = ShadowProjection.horizontal_axes(hmap.up)
        allp = np.concatenate([p for p, _ in meshes], axis=0)
        mn, mx = allp.min(axis=0), allp.max(axis=0)
        if radius is None:
            radius = 0.5 * math.hypot(mx[a] - mn[a], mx[b] - mn[b])
        if height is None:
            height = mx[hmap.up] - hmap.ground
        stretch = (
            ShadowProjection.DEFAULT_MAX_STRETCH
            if max_stretch is None
            else float(max_stretch)
        )
        reach = float(radius) + stretch * float(height)
        contact = np.zeros(3)
        contact[hmap.up] = hmap.ground
        rng = np.random.default_rng(seed)
        scores: List[float] = []
        tolerant: List[float] = []
        for _ in range(int(samples)):
            bearing = rng.uniform(0.0, _TWO_PI)
            elev = math.radians(rng.uniform(12.0, 70.0))
            dist = rng.uniform(1.5, 3.0) * reach
            light = np.zeros(3)
            light[a] = dist * math.cos(elev) * math.cos(bearing)
            light[b] = dist * math.cos(elev) * math.sin(bearing)
            light[hmap.up] = hmap.ground + dist * math.sin(elev)
            rgba, raster = ImgUtils.rasterize_shadow(
                meshes,
                light,
                hmap.ground,
                size=size,
                up=hmap.up,
                max_stretch=max_stretch,
                contact=contact,
                radius=radius,
                height=height,
                blur_amount=0.0,
            )
            exact = rgba[:, :, 3] > 127
            u_lo, u_hi, w_lo, w_hi = raster.rect
            # saved rows run from the light-side edge (u_lo) at the top
            uu = u_lo + (np.arange(size) + 0.5) / size * (u_hi - u_lo)
            ww = w_lo + (np.arange(size) + 0.5) / size * (w_hi - w_lo)
            U, W = np.meshgrid(uu, ww, indexing="ij")
            model = raster.model
            ux, uz = model.bearing
            wx, wz = model.across
            pa = model.anchor[0] + U * ux + W * wx
            pb = model.anchor[1] + U * uz + W * wz
            pts = np.zeros((size * size, 3))
            pts[:, a] = pa.ravel()
            pts[:, b] = pb.ravel()
            pts[:, hmap.up] = hmap.ground
            got = (hmap.alpha(pts, light) > 0.5).reshape(size, size)
            union = float((exact | got).sum())
            if not union:
                scores.append(0.0)
                tolerant.append(0.0)
                continue
            scores.append(float((exact ^ got).sum() / union))
            slack = ((exact & ~cls._dilate(got)) | (got & ~cls._dilate(exact))).sum()
            tolerant.append(float(slack / union))
        return {
            "mean": float(np.mean(scores)),
            "max": float(np.max(scores)),
            "tolerant_mean": float(np.mean(tolerant)),
            "tolerant_max": float(np.max(tolerant)),
            "samples": len(scores),
        }

    @staticmethod
    def _dilate(mask: np.ndarray) -> np.ndarray:
        """The mask grown by one texel (4-connected)."""
        p = np.pad(mask, 1)
        return mask | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]

    @classmethod
    def bake_adaptive(
        cls,
        meshes,
        *,
        threshold: float = 0.05,
        max_size: int = 256,
        measure_samples: int = 6,
        **kwargs,
    ) -> Tuple[HeightFieldMap, Dict[str, float]]:
        """Bake at :attr:`ADAPTIVE_SIZES` in turn until :meth:`measure`'s
        one-texel-tolerant mean disagreement is under *threshold* or
        *max_size* is reached.

        *max_size* caps the ladder; it does not choose a size. A value below
        the smallest rung clamps up to it with a warning rather than returning
        no map — the ladder is the menu, and the declared return type has no
        ``None`` in it.

        Returns:
            ``(map, score)`` — the last map baked and its measurement.

        Raises:
            TypeError: ``size`` was passed. It is a ``bake`` keyword and an
                adaptive bake chooses the size itself.
        """
        if "size" in kwargs:
            raise TypeError(
                f"{cls.__name__}.bake_adaptive() chooses its own footprint size "
                f"from ADAPTIVE_SIZES; pass max_size= to cap the ladder, or call "
                f"{cls.__name__}.bake(size=...) for a fixed size."
            )
        measure_kw = {
            k: kwargs.pop(k) for k in ("max_stretch", "radius", "height") if k in kwargs
        }
        rungs = [s for s in cls.ADAPTIVE_SIZES if s <= int(max_size)]
        if not rungs:
            rungs = [cls.ADAPTIVE_SIZES[0]]
            warnings.warn(
                f"max_size={max_size} is below the smallest adaptive rung "
                f"({cls.ADAPTIVE_SIZES[0]}); baking at {cls.ADAPTIVE_SIZES[0]}. "
                f"Call {cls.__name__}.bake(size=...) for a size off the ladder.",
                RuntimeWarning,
                stacklevel=2,
            )
        hmap, score = None, {}
        for size in rungs:
            hmap = cls.bake(meshes, size=size, **kwargs)
            score = cls.measure(hmap, meshes, samples=measure_samples, **measure_kw)
            score["size"] = size
            if score["tolerant_mean"] <= threshold:
                break
        return hmap, score


class ShadowHorizonConformance:
    """The golden cases every port of :meth:`ShadowHorizon.bake` and
    :meth:`HeightFieldMap.alpha` is held to.

    A reader of the ``shadows`` record that bakes a plane's map itself ports
    the bake, and a reader that draws it ports the evaluation as the oracle
    its shader is tested against. A port runs every case through its own
    code and compares within :attr:`TOLERANCE`; the cases are generated from
    this reference at test time (``Conformance.cases("shadow_horizon")``), so
    they cannot drift from it.

    A case is geometry in and the map out: triangles in the map's frame (the
    contact at the horizontal origin), the ground, the up axis, the size, the
    spans and the bounds (``None`` = fitted with the padding) go in; the
    layout the bake chose and the whole packed image
    (:meth:`HeightFieldMap.to_rgba`) come out, its bytes deflated (zlib) and
    base64-encoded: a 256-pixel map is 1 MB raw and 12 to 50 KB so. Each
    case also carries :meth:`HeightFieldMap.alpha` at sampled points for a
    soft and a hard point source and a soft and a hard sun, evaluated on the
    map **decoded from that image** (:meth:`HeightFieldMap.from_rgba`), what
    a shader reads, so a port's evaluation is held to the delivered bytes
    independently of its bake. The named cases come first (each edge the
    bake has, every spans count, both up axes and both adaptive sizes), then
    random ones; a case whose input the reference refuses carries
    ``"error"`` and no map.
    """

    #: Per quantity, the largest difference a port may show. ``layout``: the
    #: bounds and the height scale, frame units: closed forms in double and
    #: one float32 maximum, equal to rounding. ``texel``: codes, a 16-bit
    #: pair (the span tiles' ``RG`` and ``BA``, the aux tile's ``RG``, the
    #: pyramid tile's left-half ``BA``) read as one code and every other
    #: channel as its 8-bit one. The bake quantises float32 values by
    #: rounding, flooring and ceiling, so a port whose value is one unit in
    #: the last place off, where the reference's sits on a code boundary,
    #: lands one code over: one code is what an exact port can be held to.
    #: ``alpha``: the evaluation, float32 out of double arithmetic; a port
    #: differs only where its ``asin``, ``acos``, ``sqrt`` sums and ``hypot``
    #: differ from numpy's in the last place, which moves a soft alpha by
    #: ~1e-7, never a hard one. Measured on a C# port (.NET 10, x64, seed 3,
    #: ``per_kind=4``): 0 of 4.8 M codes and 0 of 5,664 alphas off; the
    #: bounds stay for a runtime whose libm rounds otherwise.
    TOLERANCE: Dict[str, float] = {
        "layout": 1.0e-12,
        "texel": 1.0,
        "alpha": 1.0e-6,
    }

    #: Points evaluated per source in each case.
    POINTS = 48

    # -- fixtures --------------------------------------------------------------
    @staticmethod
    def _box(x0, x1, y0, y1, z0, z1) -> Tuple[np.ndarray, np.ndarray]:
        """A closed axis-aligned box, Y-up, as ``(points, tris)``."""
        pts = np.array(
            [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)],
            dtype=float,
        )
        quads = (
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        )
        tris = []
        for a, b, c, d in quads:
            tris += [(a, b, c), (a, c, d)]
        return pts, np.array(tris, dtype=np.int64)

    @staticmethod
    def _box_flipped(x0, x1, y0, y1, z0, z1) -> Tuple[np.ndarray, np.ndarray]:
        """The box of :meth:`_box` as a DCC's cube primitive lays it out
        (the ShadowPlane sample's producer), its corners and triangles in
        that order: every quad split on the other diagonal, so a ray down a
        face's diagonal runs along the other pair's shared edge."""
        pts = np.array(
            [
                [x0, y0, z1],
                [x1, y0, z1],
                [x0, y1, z1],
                [x1, y1, z1],
                [x0, y1, z0],
                [x1, y1, z0],
                [x0, y0, z0],
                [x1, y0, z0],
            ],
            dtype=float,
        )
        tris = (
            (0, 1, 2), (2, 1, 3), (2, 3, 4), (4, 3, 5), (4, 5, 6), (6, 5, 7),
            (6, 7, 0), (0, 7, 1), (1, 7, 3), (3, 7, 5), (6, 0, 4), (4, 0, 2),
        )  # fmt: skip
        return pts, np.array(tris, dtype=np.int64)

    @staticmethod
    def _fan() -> Tuple[np.ndarray, np.ndarray]:
        """A closed prism, Y-up, from the ground to 1, whose top and bottom
        are fans of eight triangles round a vertex at (0.0625, 0.0625): over
        the bounds ``(-1, 1, -1, 1)`` at 16 pixels the fan's centre is a
        pixel centre and six of its spokes run along rows, columns and
        diagonals of pixel centres, so rays meet a shared vertex and run
        along shared edges, exactly."""
        lo, mid, hi = -0.5625, 0.0625, 0.5625
        ring = [
            (hi, lo),
            (hi, mid),
            (hi, hi),
            (mid, hi),
            (lo, hi),
            (lo, mid),
            (lo, lo),
            (mid, lo),
        ]
        pts = np.array(
            [[a, y, b] for y in (1.0, 0.0) for a, b in ring + [(mid, mid)]],
            dtype=float,
        )
        top, bottom, n = 8, 17, 9  # the centres' indices; a ring's offset
        tris = []
        for i in range(8):
            j = (i + 1) % 8
            tris += [(top, i, j), (bottom, n + j, n + i)]
            tris += [(n + i, n + j, j), (n + i, j, i)]
        return pts, np.array(tris, dtype=np.int64)

    @classmethod
    def _boxes(cls, *extents) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [cls._box(*e) for e in extents]

    @staticmethod
    def _z_up(meshes) -> List[Tuple[np.ndarray, np.ndarray]]:
        """*meshes* turned from Y-up to Z-up: ``(x, y, z) -> (x, -z, y)``."""
        return [(np.column_stack([p[:, 0], -p[:, 2], p[:, 1]]), t) for p, t in meshes]

    @classmethod
    def _random_boxes(cls, rng, up: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Two to four boxes, each turned about a random axis, round the contact."""
        out = []
        for _ in range(int(rng.integers(2, 5))):
            half = rng.uniform(0.03, 0.5, 3)
            pts, tris = cls._box(
                -half[0], half[0], -half[1], half[1], -half[2], half[2]
            )
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(0.0, math.pi)
            k = np.array(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ]
            )
            rot = np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)
            centre = np.array(
                [rng.uniform(-0.6, 0.6), rng.uniform(0.0, 1.2), rng.uniform(-0.6, 0.6)]
            )
            out.append((pts @ rot.T + centre, tris))
        return out if up == 1 else cls._z_up(out)

    @staticmethod
    def _soup(rng, up: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Six to twenty open triangles in general position, some under the ground."""
        n = int(rng.integers(6, 21))
        a, b = ShadowProjection.horizontal_axes(up)
        pts = np.zeros((3 * n, 3))
        centres = rng.uniform(-0.7, 0.7, (n, 2))
        pts[:, a] = np.repeat(centres[:, 0], 3) + rng.uniform(-0.3, 0.3, 3 * n)
        pts[:, b] = np.repeat(centres[:, 1], 3) + rng.uniform(-0.3, 0.3, 3 * n)
        pts[:, up] = rng.uniform(-0.2, 1.4, 3 * n)
        return [(pts, np.arange(3 * n, dtype=np.int64).reshape(n, 3))]

    # -- one case --------------------------------------------------------------
    @classmethod
    def _sources(cls, hmap: HeightFieldMap, rng) -> List[Dict[str, Any]]:
        """A soft and a hard point source, a soft and a hard sun, over the map."""
        a, b = ShadowProjection.horizontal_axes(hmap.up)
        a0, a1, b0, b1 = hmap.bounds
        extent = max(a1 - a0, b1 - b0)
        out = []
        for kind, spread, intensity in (
            ("point", 0.15 * extent, 1.0),
            ("point", 0.0, 1.0),
            ("sun", 0.06, 1.5),
            ("sun", 0.0, 1.0),
        ):
            bearing = rng.uniform(0.0, _TWO_PI)
            elev = math.radians(rng.uniform(15.0, 65.0))
            toward = np.zeros(3)
            toward[a] = math.cos(elev) * math.cos(bearing)
            toward[b] = math.cos(elev) * math.sin(bearing)
            toward[hmap.up] = math.sin(elev)
            src = {
                "light": None,
                "direction": None,
                "source_size": 0.0,
                "source_angle": 0.0,
                "intensity": float(intensity),
            }
            if kind == "point":
                light = toward * rng.uniform(1.5, 3.0) * extent
                light[hmap.up] += hmap.ground
                src["light"] = [float(v) for v in light]
                src["source_size"] = float(spread)
            else:
                src["direction"] = [float(v) for v in -toward]
                src["source_angle"] = float(spread)
            out.append(src)
        return out

    @classmethod
    def _points(cls, hmap: HeightFieldMap, rng) -> np.ndarray:
        """Points over the footprint and a quarter of it round, at random
        heights (the evaluation replaces each with the ground)."""
        a, b = ShadowProjection.horizontal_axes(hmap.up)
        a0, a1, b0, b1 = hmap.bounds
        ma, mb = 0.25 * (a1 - a0), 0.25 * (b1 - b0)
        pts = np.zeros((cls.POINTS, 3))
        pts[:, a] = rng.uniform(a0 - ma, a1 + ma, cls.POINTS)
        pts[:, b] = rng.uniform(b0 - mb, b1 + mb, cls.POINTS)
        pts[:, hmap.up] = hmap.ground + rng.uniform(-1.0, 1.0, cls.POINTS)
        return pts

    @classmethod
    def case(
        cls,
        meshes,
        *,
        ground: float = 0.0,
        up: int = 1,
        size: int = ShadowHorizon.DEFAULT_SIZE,
        spans: int = ShadowHorizon.DEFAULT_SPANS,
        bounds: Optional[Sequence[float]] = None,
        padding: float = ShadowHorizon.DEFAULT_PADDING,
        seed: int = 0,
        extra_sources: Sequence[Dict[str, Any]] = (),
        name: str = "",
    ) -> Dict[str, Any]:
        """One case: its inputs and what the reference makes of them.

        Parameters:
            meshes: ``(points, tris)`` pairs in the map's frame.
            ground, up, size, spans, bounds, padding: As :meth:`ShadowHorizon.bake`.
            seed: The stream the case's sources and points are drawn from.
            extra_sources: Sources evaluated after the drawn four, each
                ``{"light", "direction", "source_size", "source_angle",
                "intensity"}`` and, optionally, the ``"points"`` it is
                evaluated at (``x, y, z`` rows; drawn when absent).
            name: A label for a failure message.

        Returns:
            ``{"name", "input": {"meshes": [{"points", "triangles"}],
            "ground", "up", "size", "spans", "bounds", "padding"}, "map":
            {"size", "spans", "levels", "bounds", "ground", "up",
            "height_scale", "width", "height", "rgba"}, "alpha": [{"light",
            "direction", "source_size", "source_angle", "intensity",
            "points", "alpha"}]}``: points and triangles flat (``x, y, z``
            and ``i, j, k`` runs), ``rgba`` the image's rows top first (row 0
            is the frame's ``b0`` edge), ``R, G, B, A`` per texel, deflated
            (zlib) and base64-encoded. A refused input carries ``"error"``
            (the exception's type name) and no ``map`` or ``alpha``.
        """
        flat = []
        for p, t in meshes:
            p = np.asarray(p, dtype=float).reshape(-1, 3)
            t = np.asarray(t, dtype=np.int64).reshape(-1, 3)
            flat.append(
                {
                    "points": [float(v) for v in p.ravel()],
                    "triangles": [int(v) for v in t.ravel()],
                }
            )
        doc: Dict[str, Any] = {
            "name": name,
            "input": {
                "meshes": flat,
                "ground": float(ground),
                "up": int(up),
                "size": int(size),
                "spans": int(spans),
                "bounds": None if bounds is None else [float(v) for v in bounds],
                "padding": float(padding),
            },
        }
        try:
            hmap = ShadowHorizon.bake(
                meshes,
                ground=ground,
                up=up,
                size=size,
                spans=spans,
                bounds=bounds,
                padding=padding,
            )
        except ValueError as e:
            doc["error"] = type(e).__name__
            return doc
        rgba = hmap.to_rgba()
        doc["map"] = {
            "size": hmap.size,
            "spans": hmap.spans,
            "levels": hmap.levels,
            "bounds": [float(v) for v in hmap.bounds],
            "ground": hmap.ground,
            "up": hmap.up,
            "height_scale": hmap.height_scale,
            "width": int(rgba.shape[1]),
            "height": int(rgba.shape[0]),
            "rgba": base64.b64encode(
                zlib.compress(np.ascontiguousarray(rgba).tobytes(), 9)
            ).decode("ascii"),
        }
        delivered = HeightFieldMap.from_rgba(
            rgba,
            size=hmap.size,
            spans=hmap.spans,
            bounds=hmap.bounds,
            ground=hmap.ground,
            up=hmap.up,
            height_scale=hmap.height_scale,
        )
        rng = np.random.default_rng(seed)
        evaluations = []
        for src in cls._sources(hmap, rng) + list(extra_sources):
            if src.get("points") is None:
                pts = cls._points(hmap, rng)
            else:
                pts = np.asarray(src["points"], dtype=float).reshape(-1, 3)
            alpha = delivered.alpha(
                pts,
                src["light"],
                direction=src["direction"],
                source_size=src["source_size"],
                source_angle=src["source_angle"],
                intensity=src["intensity"],
            )
            evaluations.append(
                dict(
                    src,
                    points=[float(v) for v in pts.ravel()],
                    alpha=[float(v) for v in alpha],
                )
            )
        doc["alpha"] = evaluations
        return doc

    # -- the document ----------------------------------------------------------
    @classmethod
    def cases(cls, seed: int = 0, per_kind: int = 3) -> Dict[str, Any]:
        """The conformance document: ``{"tolerance", "cases"}``.

        Parameters:
            seed: The random stream's seed -- the same seed, the same cases.
            per_kind: Random cases of each kind (turned boxes, a triangle
                soup) after the named ones, whose own draws are fixed.
        """
        rng = np.random.default_rng(seed)
        randoms = []
        for k in range(int(per_kind)):
            for kind in ("boxes", "soup"):
                up = int(rng.choice([1, 2]))
                meshes = (
                    cls._random_boxes(rng, up)
                    if kind == "boxes"
                    else cls._soup(rng, up)
                )
                randoms.append(
                    cls.case(
                        meshes,
                        ground=float(rng.uniform(-0.1, 0.3)),
                        up=up,
                        size=int(rng.choice([16, 32, 64, 128])),
                        spans=int(rng.integers(1, 4)),
                        seed=int(rng.integers(0, 2**31)),
                        name=f"{kind}_{k}",
                    )
                )
        return {"tolerance": dict(cls.TOLERANCE), "cases": cls._named() + randoms}

    #: The named cases as JSON text, built on first use: they take no seed,
    #: and their bakes and marches are most of a document's cost.
    _NAMED_JSON: Optional[str] = None

    @classmethod
    def _named(cls) -> List[Dict[str, Any]]:
        """The named cases, a fresh copy each call."""
        if cls._NAMED_JSON is None:
            cls._NAMED_JSON = json.dumps(cls._build_named())
        return json.loads(cls._NAMED_JSON)

    @classmethod
    def _build_named(cls) -> List[Dict[str, Any]]:
        """Each edge the bake has, every spans count, both up axes and both
        adaptive sizes, in a fixed order."""
        legs = ((-0.55, -0.35), (0.5, 0.3), (-0.55, 0.3), (0.5, -0.35))
        box = cls._boxes((-1, 1, 0, 2, -1, 1))
        table = cls._boxes(
            (-0.6, 0.6, 0.7, 0.75, -0.4, 0.4),
            *((x, x + 0.05, 0, 0.7, z, z + 0.05) for x, z in legs),
        )
        chair = cls._boxes(
            (-0.25, 0.25, 0.40, 0.45, -0.25, 0.25),
            (-0.25, 0.25, 0.45, 0.90, 0.20, 0.25),
            *(
                (x, x + 0.05, 0.0, 0.40, z, z + 0.05)
                for x in (-0.23, 0.18)
                for z in (-0.23, 0.18)
            ),
        )
        stool = cls._boxes(
            (-0.2, 0.2, 0.45, 0.48, -0.2, 0.2),
            *(
                (x, x + 0.03, 0.0, 0.45, z, z + 0.03)
                for x in (-0.19, 0.16)
                for z in (-0.19, 0.16)
            ),
            *((-0.16, 0.16, 0.15, 0.18, z, z + 0.03) for z in (-0.19, 0.16)),
        )
        arch = cls._boxes(
            (-0.65, -0.55, 0.0, 1.8, -0.05, 0.05),
            (0.55, 0.65, 0.0, 1.8, -0.05, 0.05),
            (-0.7, 0.7, 1.8, 1.95, -0.05, 0.05),
        )
        # four boards over one footprint and a post: more spans than K
        shelf = cls._boxes(
            *((-0.4, 0.4, y, y + 0.03, -0.15, 0.15) for y in (0.1, 0.45, 0.62, 1.1)),
            (-0.4, -0.37, 0.0, 1.2, -0.15, 0.15),
        )
        # a square frame round an open middle: an interior hole
        frame = cls._boxes(
            (-0.6, 0.6, 0.0, 0.5, -0.6, -0.4),
            (-0.6, 0.6, 0.0, 0.5, 0.4, 0.6),
            (-0.6, -0.4, 0.0, 0.5, -0.4, 0.4),
            (0.4, 0.6, 0.0, 0.5, -0.4, 0.4),
        )
        triangle = [
            (
                np.array([[-0.4, 0.2, -0.3], [0.5, 0.9, -0.1], [0.1, 0.5, 0.45]]),
                np.array([[0, 1, 2]]),
            )
        ]
        # a single-sided board: an open surface still blocks at its height
        plank = [
            (
                np.array(
                    [
                        [-0.5, 0.6, -0.1],
                        [0.5, 0.6, -0.1],
                        [0.5, 0.62, 0.1],
                        [-0.5, 0.62, 0.1],
                    ]
                ),
                np.array([[0, 1, 2], [0, 2, 3]]),
            )
        ]
        # a 1 cm strip leaning across the footprint: thinner than a pixel
        strip = [
            (
                np.array(
                    [
                        [-0.5, 0.0, -0.3],
                        [-0.49, 0.0, -0.3],
                        [0.6, 1.5, 0.4],
                        [0.61, 1.5, 0.4],
                    ]
                ),
                np.array([[0, 1, 2], [1, 3, 2]]),
            )
        ]
        below = {
            "light": [0.5, -1.0, 0.2],
            "direction": None,
            "source_size": 0.2,
            "source_angle": 0.0,
            "intensity": 1.0,
        }
        # a sun near noon: the plank's shadow falls inside its own footprint
        steep = {
            "light": None,
            "direction": [float(v) for v in ShadowProjection._unit((0.15, -1.0, 0.08))],
            "source_size": 0.0,
            "source_angle": 0.02,
            "intensity": 1.0,
        }
        small, large = ShadowHorizon.ADAPTIVE_SIZES
        named = [
            cls.case(box, extra_sources=[below], seed=1, name="box"),
            cls.case(triangle, seed=2, name="single_triangle"),
            cls.case(plank, extra_sources=[steep], seed=3, name="open_plank"),
            cls.case(table, seed=4, name="overhang"),
            cls.case(frame, seed=5, name="interior_hole"),
            cls.case(stool, seed=6, name="stool"),
            cls.case(shelf, spans=1, seed=7, name="spans_1"),
            cls.case(shelf, spans=2, seed=8, name="spans_2"),
            cls.case(shelf, spans=3, seed=9, name="spans_3"),
            # the bounds cut the box on two sides; the post straddles the a0 edge
            cls.case(
                cls._boxes(
                    (-0.8, 0.3, 0.0, 1.0, 0.2, 0.9),
                    (-0.52, -0.48, 0.0, 0.8, -0.3, -0.2),
                ),
                bounds=(-0.5, 0.5, -0.5, 0.5),
                size=64,
                seed=10,
                name="canvas_edge",
            ),
            # an empty footprint, twice: the bounds miss the geometry, or it
            # lies wholly under the ground
            cls.case(
                box, bounds=(3.0, 4.0, 3.0, 4.0), size=32, seed=11, name="empty_outside"
            ),
            cls.case(
                cls._boxes((-0.5, 0.5, 0.0, 0.4, -0.5, 0.5)),
                ground=0.5,
                size=32,
                seed=12,
                name="empty_buried",
            ),
            cls.case(
                cls._boxes((-0.5, 0.5, -0.5, 1.0, -0.3, 0.3)),
                seed=13,
                name="ground_cut",
            ),
            cls.case(strip, seed=14, name="thin_member"),
            cls.case(
                cls._boxes((-1.0, 1.0, 0.0, 0.45, -0.12, 0.12)),
                seed=15,
                name="elongated",
            ),
            cls.case(cls._z_up(chair), up=2, seed=16, name="z_up"),
            cls.case(arch, size=small, seed=17, name="adaptive_small"),
            cls.case(arch, size=large, seed=18, name="adaptive_large"),
            cls.case(
                chair, size=large, spans=3, seed=19, name="adaptive_large_spans_3"
            ),
            cls.case(table, size=100, seed=20, name="size_rounds_up"),
            cls.case(box, size=3, seed=21, name="size_floor"),
            cls.case([], name="no_geometry"),
            # Watertight: rays down a face's diagonal run along the edge its
            # two triangles share (the footprint centred, its bounds
            # symmetric), and either split bakes the same solid; a fan whose
            # centre and spokes sit on pixel centres, exactly.
            cls.case(
                [cls._box_flipped(-30, 30, 0, 60, -30, 30)],
                size=64,
                seed=22,
                name="edge_rays_flipped",
            ),
            cls.case(
                cls._boxes((-30, 30, 0, 60, -30, 30)),
                size=64,
                seed=22,
                name="edge_rays_split",
            ),
            cls.case(
                [cls._fan()],
                size=16,
                bounds=(-1.0, 1.0, -1.0, 1.0),
                seed=23,
                name="edge_rays_fan",
            ),
            # Two parts meeting face to face a float32 step apart -- an
            # awning's post under its slab, as an engine imported one DCC's
            # delivery (measured 2026-11-08) -- bake one span, not daylight.
            cls.case(
                cls._boxes(
                    (-0.06, 0.06, 0.0, float(np.nextafter(np.float32(1.1), np.float32(0.0))), -0.06, 0.06),
                    (-0.3, 0.5, 1.1, 1.18, -0.2, 0.2),
                ),
                size=64,
                seed=24,
                name="faces_a_float_apart",
            ),
        ]
        return named + cls._exact_named()

    @classmethod
    def _exact_named(cls) -> List[Dict[str, Any]]:
        """The named cases whose values sit exactly on a tie or a boundary,
        each pinning one rule a port could spell otherwise and still pass
        the rest: over 16 pixels of ``(-1, 1)``, where every pixel edge and
        centre is a binary fraction."""
        unit = (-1.0, 1.0, -1.0, 1.0)
        # three boards, evenly spaced, over K = 2: the two gaps tie, and the
        # first (lowest) is merged
        boards = cls._boxes(
            *((-0.5, 0.5, y, y + 0.125, -0.25, 0.25) for y in (0.25, 0.5, 0.75))
        )
        # pixels 129/128 as wide as tall: a pixel beside a column along a is
        # 1.0078125 pitches away, 64.5 distance codes -- a half, to even
        wide = (-1.0078125, 1.0078125, -1.0, 1.0)

        # one-pixel posts of two heights, a pixel (column 6, row 8) sqrt(5)
        # from both: one row up, two along; two rows down, one along. The
        # lower row's hull is the nearest's.
        def post(i, j, height):
            a, b = -1.0 + i / 8.0, -1.0 + j / 8.0
            return (a + 0.02, a + 0.105, 0.0, height, b + 0.02, b + 0.105)

        posts = cls._boxes(post(8, 9, 0.5), post(7, 6, 1.0))
        # a post whose -a face lies on the pixel edge a = 0, and points on
        # that edge: a ray leaving it toward -a starts in the pixel it heads
        # into, not the solid one behind it
        edge_post = cls._boxes((0.0, 0.1, 0.0, 1.0, 0.02, 0.105))
        on_edge = {
            "light": None,
            "direction": [float(v) for v in ShadowProjection._unit((1.0, -1.0, 0.0))],
            "source_size": 0.0,
            "source_angle": 0.0,
            "intensity": 1.0,
            "points": [[0.0, 0.0, 0.0625], [0.0, 0.0, 0.03]],
        }
        # a sun within 1e-9 of the zenith: no horizontal bearing, no shadow,
        # even under the post
        zenith = {
            "light": None,
            "direction": [1e-10, -1.0, 0.0],
            "source_size": 0.0,
            "source_angle": 0.0,
            "intensity": 1.0,
            "points": [[0.05, 0.0, 0.06], [0.04, 0.0, 0.05]],
        }
        return [
            cls.case(boards, size=16, spans=2, bounds=unit, seed=24, name="equal_gaps"),
            cls.case(
                cls._boxes((-0.2, 0.2, 0.0, 0.5, -0.2, 0.2)),
                size=16,
                bounds=wide,
                seed=25,
                name="half_code",
            ),
            cls.case(posts, size=16, bounds=unit, seed=26, name="row_tie"),
            cls.case(
                edge_post,
                size=16,
                bounds=unit,
                extra_sources=[on_edge, zenith],
                seed=27,
                name="march_edges",
            ),
        ]
