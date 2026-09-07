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

import math
import warnings
from importlib import resources
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from pythontk.geo_utils.shadow_projection import ShadowProjection

__all__ = ["HeightFieldMap", "HorizonMap", "ShadowHorizon"]

_TWO_PI = 2.0 * math.pi
#: The 16-bit range a height or a distance is quantised over.
MAX16 = 65535.0
#: Fixed-point scale of the distance field: 1/64 of a footprint pixel.
DIST_SCALE = 64.0
#: The distance an empty map reports everywhere (the field's largest value).
DIST_FAR = MAX16 / DIST_SCALE


class HeightFieldMap(NamedTuple):
    """A baked height-field shadow map and the frame it was baked in.

    Heights are above the ground plane, in frame units, already quantised
    the way the PNG stores them (16-bit over ``height_scale``; the nearest
    column's hull 8-bit; the distance field in 1/64 pixel), so a map and its
    PNG round trip evaluate identically. ``NaN`` marks an absent span.
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmin(self.lo, axis=0), np.nanmax(self.hi, axis=0)

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
        lo = self._dilate_min(self.near_lo)
        hi = self._dilate_max(self.near_hi)
        dist = self.dist
        out = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            while lo.shape[0] > 1:
                n = lo.shape[0] // 2
                lo = np.nanmin(lo.reshape(n, 2, n, 2), axis=(1, 3))
                hi = np.nanmax(hi.reshape(n, 2, n, 2), axis=(1, 3))
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
    def _dilate_min(field: np.ndarray) -> np.ndarray:
        p = np.pad(field, 1, constant_values=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stack = np.stack(
                [
                    p[i : i + field.shape[0], j : j + field.shape[1]]
                    for i in range(3)
                    for j in range(3)
                ]
            )
            return np.nanmin(stack, axis=0)

    @staticmethod
    def _dilate_max(field: np.ndarray) -> np.ndarray:
        p = np.pad(field, 1, constant_values=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stack = np.stack(
                [
                    p[i : i + field.shape[0], j : j + field.shape[1]]
                    for i in range(3)
                    for j in range(3)
                ]
            )
            return np.nanmax(stack, axis=0)

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
        least distance. Row ``0`` of every tile is ``ib = 0``, the frame's
        ``b0`` edge -- data, never flipped.
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
            ``(N,)`` float32 in ``[0, 1]``: 1 where the ray toward the source
            crosses a span, a penumbra fading with the ray's angular
            clearance from the nearest column against the source's angular
            radius, 0 where the source is below the ground.
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
        alpha = self._march(origin, direction2, slope, rho)
        out[:] = np.where(live, np.clip(alpha * float(intensity), 0.0, 1.0), 0.0)
        return out

    def _march(self, origin, direction, slope, rho) -> np.ndarray:
        """The exact walk: ``(N,)`` alpha for rays from the frame-horizontal
        *origin* ``(N, 2)`` along the horizontal unit *direction* ``(N, 2)``,
        rising *slope* per unit of horizontal travel, against a source of
        angular radius *rho* ``(N,)``.

        Every level-0 pixel the ray crosses is visited (Amanatides-Woo, all
        rays at once). A pixel with spans blocks where the ray's height range
        across it overlaps a span; otherwise it contributes its gap -- the
        vertical distance to the nearest span on a solid pixel, on an empty
        pixel the lateral distance to the nearest column combined with the
        vertical distance to that column's hull -- divided by the distance
        along the ray, an angular clearance. The penumbra is
        ``1 - min clearance / rho``.
        """
        S, K = self.size, self.spans
        a0, a1, b0, b1 = self.bounds
        pw, ph = self.pixel
        px = min(pw, ph)
        n = origin.shape[0]
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
        for _ in range(2 * S + 4):
            act = active & ~hit
            if not act.any():
                break
            te = np.minimum(np.minimum(tmx, tmy), t_exit)
            h0, h1 = slope * t, slope * te
            inb = (cx >= 0) & (cx < S) & (cy >= 0) & (cy < S)
            cxc, cyc = np.clip(cx, 0, S - 1), np.clip(cy, 0, S - 1)
            gap = np.full(n, np.inf)
            solid = np.zeros(n, bool)
            blocked = np.zeros(n, bool)
            for k in range(K):
                lo_k, hi_k = self.lo[k, cyc, cxc], self.hi[k, cyc, cxc]
                has = inb & ~np.isnan(lo_k)
                overlap = has & (h1 >= lo_k) & (h0 <= hi_k)
                blocked |= overlap
                solid |= has
                g = np.where(h0 > hi_k, h0 - hi_k, lo_k - h1)
                gap = np.where(has & ~overlap, np.minimum(gap, g), gap)
            empty = inb & ~solid
            tmid = np.maximum(0.5 * (t + te), 1e-9)
            # The distance field and the nearest column's hull, read
            # BILINEARLY at the ray's midpoint through the pixel: per-pixel
            # values would make the penumbra jump a quarter of its width
            # when a boundary-hugging ray lands one pixel over on a GPU's
            # float32 (measured: 0.21 against 0.07 at one viewer sample).
            mx_ = ox + dx * tmid - 0.5
            my_ = oy + dy * tmid - 0.5
            fx_, fy_ = np.floor(mx_), np.floor(my_)
            wx = np.clip(mx_ - fx_, 0.0, 1.0)
            wy = np.clip(my_ - fy_, 0.0, 1.0)
            ix0 = np.clip(fx_.astype(np.int64), 0, S - 1)
            iy0 = np.clip(fy_.astype(np.int64), 0, S - 1)
            ix1 = np.minimum(ix0 + 1, S - 1)
            iy1 = np.minimum(iy0 + 1, S - 1)

            def bilerp(field):
                top = field[iy0, ix0] * (1.0 - wx) + field[iy0, ix1] * wx
                bottom = field[iy1, ix0] * (1.0 - wx) + field[iy1, ix1] * wx
                return top * (1.0 - wy) + bottom * wy

            # the half pixel from a pixel's centre to its edge, in the
            # smaller pitch: the larger pitch is ``aspect`` of them
            lateral = np.maximum(bilerp(self.dist) - 0.5 * self.aspect, 0.0) * px
            nlo, nhi = bilerp(self.near_lo), bilerp(self.near_hi)
            vertical = np.where(h0 > nhi, h0 - nhi, np.where(h1 < nlo, nlo - h1, 0.0))
            gap = np.where(empty, np.hypot(lateral, vertical), gap)
            clear = np.where(act & ~blocked, np.minimum(clear, gap / tmid), clear)
            hit |= act & blocked
            mx = tmx <= tmy
            my = tmy <= tmx
            t = np.where(act, te, t)
            cx = np.where(act & mx, cx + stepx, cx)
            tmx = np.where(act & mx, tmx + tdx, tmx)
            cy = np.where(act & my, cy + stepy, cy)
            tmy = np.where(act & my, tmy + tdy, tmy)
            active &= t < t_exit - 1e-12
        rho = np.asarray(rho, dtype=float)
        soft = np.clip(1.0 - clear / np.maximum(rho, 1e-12), 0.0, 1.0)
        return np.where(hit, 1.0, np.where(rho > 1e-12, soft, 0.0)).astype(np.float32)


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
    #: Margin around the geometry as a fraction of its larger extent, on
    #: every side. The march runs only inside the footprint, so this is also
    #: how far a penumbra can reach sideways past the prop's outline: a ray
    #: that never enters the footprint is lit outright.
    DEFAULT_PADDING = 0.1
    ENCODING = 2
    MAPPING = "heightfield"
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
    def record(
        cls,
        *,
        texture: str,
        size: int,
        spans: int,
        levels: int,
        bounds: Sequence[float],
        height_scale: float,
        frame_a: Sequence[float],
        frame_b: Sequence[float],
        rect: Sequence[float],
    ) -> Dict[str, object]:
        """The ``horizon`` block of a ``shadow_metadata`` v2 plane record.

        One builder, in the package whose GLB writer and viewer read the
        block, so the DCC rigs (which stamp these values on the plane and
        publish them at export) cannot spell the schema two ways.

        Parameters:
            texture: The map's file name -- its atlas when packed.
            size, spans, levels: The map's layout (:class:`HeightFieldMap`).
            bounds: ``(a0, a1, b0, b1)`` footprint in the contact frame.
            height_scale: The height a 16-bit value of 65535 stands for.
            frame_a, frame_b: The frame's horizontal axes in exporter axes.
            rect: ``(su, sv, ou, ov)`` UV rect of the map in its texture.

        Returns:
            The block, with :attr:`MAPPING` and :attr:`ENCODING` stamped.
        """
        return {
            "texture": str(texture),
            "mapping": cls.MAPPING,
            "encoding": cls.ENCODING,
            "size": int(size),
            "spans": int(spans),
            "levels": int(levels),
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
            padding: Margin around the geometry as a fraction of its extent.

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
        lo, hi, bounds = ImgUtils.rasterize_height_spans(
            meshes,
            up=up,
            size=size,
            ground=ground,
            bounds=bounds,
            padding=padding,
            spans=spans,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            top = float(np.nanmax(hi)) if np.isfinite(np.nanmax(hi)) else 0.0
        height_scale = max(top, 1e-3)
        # Quantised as the PNG will hold them, conservatively: bottoms down,
        # tops up, so no span thins out of a ray that met it in float.
        lo_q = np.floor(lo / height_scale * MAX16) / MAX16 * height_scale
        hi_q = np.ceil(hi / height_scale * MAX16) / MAX16 * height_scale
        gone = ~np.isnan(hi_q) & (hi_q <= 0.0)
        lo_q[gone] = np.nan
        hi_q[gone] = np.nan
        lo_q = lo_q.astype(np.float32)
        hi_q = hi_q.astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hull_lo = np.nanmin(lo_q, axis=0)
            hull_hi = np.nanmax(hi_q, axis=0)
        solid = ~np.isnan(hull_hi)
        # distances in the smaller pixel pitch, the other axis weighted up
        a0, a1, b0, b1 = (float(v) for v in bounds)
        pw, ph = (a1 - a0) / size, (b1 - b0) / size
        pitch = max(min(pw, ph), 1e-12)
        dist, near_lo, near_hi = cls._distance_field(
            solid, hull_lo, hull_hi, weights=(pw / pitch, ph / pitch)
        )
        dist = np.minimum(np.round(dist * DIST_SCALE) / DIST_SCALE, DIST_FAR)
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
