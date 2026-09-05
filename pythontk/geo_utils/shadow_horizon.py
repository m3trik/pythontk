# !/usr/bin/python
# coding=utf-8
"""Coverage-aware horizon maps: a ground shadow that follows the light at runtime.

The projected shadow (:class:`~pythontk.ShadowProjection`) draws one silhouette
for one source position. A horizon map instead stores, per ground texel and per
azimuth bin, what the occluder blocks as seen *from that texel*, so an engine can
answer "is this texel in shadow?" for any source position with a few texture
fetches. Per texel, bin and **layer** the map keeps:

* ``lo, hi`` — the elevation interval the occluder blocks along the ray at the
  bin's first covered sub-bin (``[0, θ]`` outside a grounded footprint,
  ``[θ, 90°]`` under an overhang, ``[θ1, θ2]`` for a slab seen edge-on);
* a 16-bit **occupancy mask** over 16 sub-bins of the bin — where within the
  bin the occluder actually is. This is what keeps thin members (chair legs)
  casting one shadow each on their true bearing: a plain horizon map blends
  neighbouring bins and turns a leg into two ghosts, and an azimuth *hull* per
  bin was measured to merge two legs whenever they share a bin — at four
  metres that happens even with 64 bins — and to shadow the ground between.

Two layers, because furniture is legs under a top: the **grounded** layer holds
the footprint columns that touch the ground (legs, walls, boxes; ``lo`` is
always 0) and the **floating** layer the overhangs (tops, seats). A table top
spans a whole bin while its legs are thin, so one interval per bin cannot serve
both; the shader evaluates both layers and takes their union.

The map lives in the target's own frame — origin at the contact point, the
vertical axis ``up``, bearing zero along the first horizontal axis and
increasing toward the second (:meth:`ShadowProjection.horizontal_axes`) — so a
prop moved or rotated at runtime carries its shadow with it, and one bake serves
any number of sources. Texels are laid out **log-polar** around the contact
(bearing × log-distance): the resolution sits where the shadow is sharp, and the
far field, where the penumbra is widest anyway, is coarse.

Two passes make the bake. Height fields (:meth:`ImgUtils.rasterize_height_fields`)
give the occluder's top and bottom surface over its footprint; per layer the
mask of each bin then comes from the angular extents of the layer's outline
pixels, and the interval from marching the first covered sub-bin's ray through
the layer's columns. Measured around a second at 256 × 128 texels for a
chair-sized prop.

:meth:`HorizonMap.alpha` is the reference every engine shader is pinned to;
:meth:`ShadowHorizon.measure` compares it against :meth:`ImgUtils.rasterize_shadow`,
the exact projection, at random source positions, and :meth:`ShadowHorizon.bake_adaptive`
doubles the bin count until that disagreement is under a threshold.
"""

from __future__ import annotations

import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from pythontk.geo_utils.shadow_projection import ShadowProjection

__all__ = ["HorizonMap", "ShadowHorizon"]

_TWO_PI = 2.0 * math.pi
_HALF_PI = 0.5 * math.pi
#: Layer order in ``HorizonMap.data`` and in the PNG's tile grid.
GROUNDED, FLOATING = 0, 1
LAYERS = 2
#: Sub-bins of the occupancy mask per bin (8 bits in B, 8 in A).
SUBBINS = 16
_BIT_WEIGHTS = (1 << np.arange(8)).astype(np.int32)


class HorizonMap(NamedTuple):
    """A baked horizon map and the frame it was baked in.

    ``data`` is ``(2, bins, H, W, 4)`` float32 — layer 0 the grounded columns,
    layer 1 the floating ones — holding what the PNG holds, scaled to
    ``[0, 1]``. Floating: ``R = cot(lo)``, ``G = cot(hi)``, both divided by
    ``max_stretch`` (:meth:`encode_angle`), sampled along the ray at the
    bin's coverage centre. Grounded (``lo`` is always the ground): ``R`` and
    ``G`` are ``cot(hi)`` of the first and of the later runs of set
    sub-bins, so two legs in one bin keep their own shadow lengths. ``B`` and
    ``A`` are the two mask bytes as ``byte / 255`` (exact to quantise). Row
    ``0`` of a tile is the ``r_min`` ring (the PNG's top row), column ``x``
    spans bearings ``[x, x + 1) · 2π / W``. All-zero texels are empty bins.
    """

    bins: int
    size: Tuple[int, int]  #: (W bearing samples, H radial samples) per tile
    r_min: float
    r_max: float
    ground: float  #: height of the ground plane along ``up`` in the map's frame
    up: int
    max_stretch: float  #: the reach cap: ``R, G`` hold ``cot(angle) / max_stretch``
    data: np.ndarray

    # -- encoding ----------------------------------------------------------
    def encode_angle(self, angle: np.ndarray) -> np.ndarray:
        """``cot(angle) / max_stretch`` clamped to ``[0, 1]`` — the interval
        channels' encoding. A shadow boundary moves as ``cot(elevation)``, so
        storing the cotangent keeps the boundary's quantisation error linear
        (``height × max_stretch / 255``, a couple of centimetres) where 8-bit
        degrees put it at tens of centimetres under a grazing light. ``0``
        is the zenith, ``1`` the reach-cap elevation ``atan(1 / max_stretch)``
        and everything below it."""
        a = np.asarray(angle, dtype=float)
        with np.errstate(divide="ignore"):
            cot = np.where(a > 1e-9, 1.0 / np.tan(np.maximum(a, 1e-9)), np.inf)
        return np.clip(cot / self.max_stretch, 0.0, 1.0)

    def decode_cot(self, value: np.ndarray) -> np.ndarray:
        """The cotangent an encoded channel value stands for."""
        return np.asarray(value, dtype=float) * self.max_stretch

    # -- layout ------------------------------------------------------------
    @property
    def layers(self) -> int:
        return LAYERS

    @property
    def tiles(self) -> int:
        """Tiles in the PNG: one per layer per bin."""
        return LAYERS * self.bins

    @property
    def layout(self) -> Tuple[int, int]:
        """``(cols, rows)`` of the tile grid a PNG lays the tiles out in."""
        return ShadowHorizon.layout(self.tiles)

    def tile_index(self, layer: int, k: int) -> int:
        """Tile ``t`` of bin *k*'s *layer*: grounded tiles first."""
        return int(layer) * self.bins + int(k) % self.bins

    def tile_rects(self) -> List[Tuple[float, float, float, float]]:
        """Per tile, its ``(scaleX, scaleY, offsetX, offsetY)`` inside the
        block, bottom-left origin (the lightmap convention)."""
        cols, rows = self.layout
        rects = []
        for t in range(self.tiles):
            c, r = t % cols, t // cols
            rects.append((1.0 / cols, 1.0 / rows, c / cols, 1.0 - (r + 1) / rows))
        return rects

    def to_rgba(self) -> np.ndarray:
        """The map as one ``uint8`` RGBA image: ``2 × bins`` tiles in a
        ``cols × rows`` grid, tile ``t`` at column ``t mod cols`` and row
        ``t div cols`` (row 0 at the top), each tile's top row the ``r_min``
        ring."""
        cols, rows = self.layout
        w, h = self.size
        image = np.zeros((rows * h, cols * w, 4), dtype=np.uint8)
        quant = np.clip(np.rint(self.data * 255.0), 0, 255).astype(np.uint8)
        for layer in range(LAYERS):
            for k in range(self.bins):
                t = self.tile_index(layer, k)
                c, r = t % cols, t // cols
                image[r * h : (r + 1) * h, c * w : (c + 1) * w] = quant[layer, k]
        return image

    @classmethod
    def from_rgba(
        cls,
        pixels: np.ndarray,
        *,
        bins: int,
        size: Sequence[int],
        r_min: float,
        r_max: float,
        ground: float = 0.0,
        up: int = 1,
        max_stretch: Optional[float] = None,
    ) -> "HorizonMap":
        """Rebuild a map from an image written by :meth:`to_rgba`."""
        if max_stretch is None:
            max_stretch = ShadowProjection.DEFAULT_MAX_STRETCH
        pix = np.asarray(pixels)
        if pix.dtype != np.uint8:
            raise ValueError("HorizonMap.from_rgba: expects a uint8 RGBA image.")
        w, h = int(size[0]), int(size[1])
        bins = int(bins)
        cols, rows = ShadowHorizon.layout(LAYERS * bins)
        if (
            pix.ndim != 3
            or pix.shape[0] < rows * h
            or pix.shape[1] < cols * w
            or pix.shape[2] != 4
        ):
            raise ValueError(
                f"HorizonMap.from_rgba: image {pix.shape} cannot hold {LAYERS * bins} "
                f"tiles of {w}×{h}."
            )
        data = np.zeros((LAYERS, bins, h, w, 4), dtype=np.float32)
        for layer in range(LAYERS):
            for k in range(bins):
                t = layer * bins + k
                c, r = t % cols, t // cols
                data[layer, k] = (
                    pix[r * h : (r + 1) * h, c * w : (c + 1) * w].astype(np.float32)
                    / 255.0
                )
        return cls(
            bins,
            (w, h),
            float(r_min),
            float(r_max),
            float(ground),
            int(up),
            float(max_stretch),
            data,
        )

    # -- mapping -----------------------------------------------------------
    def texel_positions(self) -> np.ndarray:
        """Horizontal frame coordinates ``(H, W, 2)`` of every texel centre."""
        w, h = self.size
        theta = (np.arange(w) + 0.5) / w * _TWO_PI
        v = (np.arange(h) + 0.5) / h
        r = self.r_min * (self.r_max / self.r_min) ** v
        a = r[:, None] * np.cos(theta)[None, :]
        b = r[:, None] * np.sin(theta)[None, :]
        return np.stack([a, b], axis=-1)

    def uv(self, horizontal) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Log-polar ``(u, v, inside)`` for horizontal frame points ``(N, 2)``:
        ``u = θ / 2π``, ``v = ln(r / r_min) / ln(r_max / r_min)`` clamped to
        ``[0, 1]``; ``inside`` is false beyond ``r_max``."""
        p = np.asarray(horizontal, dtype=float).reshape(-1, 2)
        r = np.hypot(p[:, 0], p[:, 1])
        theta = np.mod(np.arctan2(p[:, 1], p[:, 0]), _TWO_PI)
        span = math.log(self.r_max / self.r_min)
        v = np.log(np.maximum(r, 1e-12) / self.r_min) / span
        return theta / _TWO_PI, np.clip(v, 0.0, 1.0), r <= self.r_max

    def taps(self, layer: int, k, u, v) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The four texels around ``(u, v)`` in *layer*'s bin ``k`` — the way
        the engine shaders fetch, so the reference and the pixels agree.

        Returns:
            ``(values, weights, nearest)`` — ``(N, 4, 4)`` texel values,
            ``(N, 4)`` bilinear weights, and ``(N,)`` the index of the
            heaviest tap. ``u`` wraps; ``v`` clamps to the tile's rows.
        """
        w, h = self.size
        k = np.mod(np.asarray(k, dtype=int), self.bins)
        x = np.asarray(u, dtype=float) * w - 0.5
        y = np.clip(np.asarray(v, dtype=float) * h - 0.5, 0.0, h - 1.0)
        x0 = np.floor(x)
        fx = x - x0
        x0 = np.mod(x0.astype(int), w)
        x1 = np.mod(x0 + 1, w)
        y0 = np.floor(y).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)
        fy = y - y0
        d = self.data[int(layer)]
        values = np.stack(
            [d[k, y0, x0], d[k, y0, x1], d[k, y1, x0], d[k, y1, x1]], axis=1
        )
        weights = np.stack(
            [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1
        )
        # Tie-break toward the HIGHER texel, as the shaders do. Not cosmetic:
        # the nearest tap alone decides the grounded run index, so at a dead
        # tie ``argmax`` (which takes tap 0) selects a different branch.
        nearest = (fx >= 0.5).astype(int) + 2 * (fy >= 0.5).astype(int)
        return values, weights, nearest

    @staticmethod
    def mask_bits(values: np.ndarray) -> np.ndarray:
        """``(..., 16)`` booleans from the ``B, A`` channels of texel values
        ``(..., 4)``: bit ``j`` is sub-bin ``j``."""
        b = np.rint(values[..., 2] * 255.0).astype(np.int32)
        a = np.rint(values[..., 3] * 255.0).astype(np.int32)
        lo = (b[..., None] & _BIT_WEIGHTS) > 0
        hi = (a[..., None] & _BIT_WEIGHTS) > 0
        return np.concatenate([lo, hi], axis=-1)

    # -- the reference -----------------------------------------------------
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
        """Shadow alpha ``(N,)`` at frame *points* for one source — the
        reference the engine shaders are pinned to.

        A point's height is not merely ignored, it is **replaced by**
        :attr:`ground` before the light vector is formed, and a shader must do
        the same. The map's intervals are elevations *as seen from the ground
        plane*: the bake's texel positions are horizontal and ``_march``
        measures from height 0. A fragment sits above that datum — the DCC
        rigs lift the plane by their own ``GROUND_OFFSET``, and an artist can
        raise it further — so forming ``L`` from the fragment measures against
        a plane the map was never baked on.

        Parameters:
            points: ``(N, 3)`` points in the map's frame; the height is
                replaced by :attr:`ground` (the receiver is the ground plane).
            light: Positional source in the frame (ignored with *direction*).
            direction: Unit direction a directional source shines along.
            source_size: Diameter of a positional source, frame units.
            source_angle: Angular diameter of a directional source, radians.
            intensity: Multiplier on the result.

        Returns:
            ``(N,)`` float32 in ``[0, 1]``; 0 beyond ``r_max`` and where the
            source is below the ground.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        a, b = ShadowProjection.horizontal_axes(self.up)
        n = pts.shape[0]
        out = np.zeros(n, dtype=np.float32)
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
                    "HorizonMap.alpha: a light position or direction is required."
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
        elev = np.arctan2(L[:, self.up], horiz)
        phi = np.mod(np.arctan2(L[:, b], L[:, a]), _TWO_PI)
        u, v, inside = self.uv(pts[:, [a, b]])
        live = inside & (elev > 0.0)
        if not live.any():
            return out
        step = _TWO_PI / self.bins
        fb = phi / step
        k = np.floor(fb).astype(int)
        s = fb - k
        a_g = self._layer_alpha(GROUNDED, k, s, u, v, elev, rho, step)
        a_f = self._layer_alpha(FLOATING, k, s, u, v, elev, rho, step)
        union = 1.0 - (1.0 - a_g) * (1.0 - a_f)
        out[:] = np.where(live, np.clip(union * float(intensity), 0.0, 1.0), 0.0)
        return out

    @staticmethod
    def _span(bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(has, centre)`` of a ``(N, 16)`` mask: whether any bit is set and
        the midpoint of the first and last set sub-bins as a bin fraction."""
        has = bits.any(axis=1)
        first = np.argmax(bits, axis=1)
        last = SUBBINS - 1 - np.argmax(bits[:, ::-1], axis=1)
        return has, 0.5 * (first + last + 1) / SUBBINS

    @staticmethod
    def _run_starts(bits: np.ndarray) -> np.ndarray:
        """Where each run begins in a ``(..., 16)`` mask (``0 -> 1``
        transitions). ``.sum(-1)`` counts the runs, ``cumsum(-1)`` indexes
        them — the grounded layer needs both."""
        prev = np.concatenate([np.zeros_like(bits[..., :1]), bits[..., :-1]], axis=-1)
        return bits & ~prev

    def _interval(
        self,
        vals: np.ndarray,
        wts: np.ndarray,
        nearest: np.ndarray,
        covered: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Bilinear ``(N, 2)`` of the interval channels over the taps that
        carry coverage — an empty texel encodes the zenith and must not pull
        a neighbour's interval toward it.

        *covered* overrides which taps contribute. The grounded later-run top
        passes ``run_count >= 2``: ``G`` is the SECOND run's top and is written
        ``0`` on a one-run texel, so the default predicate lets a one-run
        neighbour drag it toward the zenith and lengthen the shadow.
        """
        if covered is None:
            covered = self.mask_bits(vals).any(axis=-1)  # (N, 4)
        w = wts * covered
        total = w.sum(axis=1)
        idx = np.arange(vals.shape[0])
        fallback = vals[idx, nearest, :2]
        blended = (vals[:, :, :2] * w[:, :, None]).sum(axis=1) / np.where(
            total > 1e-9, total, 1.0
        )[:, None]
        return np.where((total > 1e-9)[:, None], blended, fallback)

    def _layer_alpha(self, layer, k, s, u, v, elev, rho, step) -> np.ndarray:
        """The contract's per-bin math for one layer (``(N,)`` in ``[0, 1]``)."""
        idx = np.arange(k.shape[0])
        vals_a, wts, nearest = self.taps(layer, k, u, v)
        # Three tiles per layer: k and its two neighbours. The interval's
        # neighbour (k + side) is always one of them, so it is selected, not
        # fetched again — an engine reads two tiles for a point source (the
        # coverage collapses to the centre sub-bin of k) and three for a disc
        # that straddles a bin edge.
        vals_p, _, _ = self.taps(layer, k - 1, u, v)
        vals_n, _, _ = self.taps(layer, k + 1, u, v)
        # Decode each tile's masks ONCE: every span, coverage and run test
        # below is a view of these three, not another decode.
        mask_p, mask_a, mask_n = (
            self.mask_bits(vals_p),
            self.mask_bits(vals_a),
            self.mask_bits(vals_n),
        )
        has_a, mid_a = self._span(mask_a[idx, nearest])
        side = np.where(s > mid_a, 1, -1)
        pick = (side == 1)[:, None, None]
        vals_b = np.where(pick, vals_n, vals_p)
        # Only b's NEAREST tap is ever read (its span), so select that row
        # rather than materialising the whole neighbour mask.
        has_b, mid_b = self._span(
            np.where((side == 1)[:, None], mask_n[idx, nearest], mask_p[idx, nearest])
        )
        # -- coverage: the disc's sub-bin span against each tap's bits,
        #    across bins k-1, k, k+1 (the disc may straddle a bin edge)
        bits = np.concatenate(
            [mask_p, mask_a, mask_n], axis=-1
        )  # (N, 4, 48): sub-bin j of bin k is column 16 + j
        rho_sub = (rho / step) * SUBBINS
        centre = (s * SUBBINS) + SUBBINS
        lo_d, hi_d = centre - rho_sub, centre + rho_sub
        edges = np.arange(3 * SUBBINS + 1, dtype=float)
        seg_lo = np.maximum(lo_d[:, None], edges[None, :-1])
        seg_hi = np.minimum(hi_d[:, None], edges[None, 1:])
        overlap = np.maximum(0.0, seg_hi - seg_lo)  # (N, 48)
        width = 2.0 * rho_sub
        point = width <= 1e-9
        centre_bin = np.clip(np.floor(centre).astype(int), 0, 3 * SUBBINS - 1)
        cov_tap = np.where(
            point[:, None],
            bits[idx[:, None], np.arange(4)[None, :], centre_bin[:, None]].astype(
                float
            ),
            (bits * overlap[:, None, :]).sum(axis=-1)
            / np.where(point, 1.0, width)[:, None],
        )
        cov_phi = np.clip((cov_tap * wts).sum(axis=1), 0.0, 1.0)
        # -- interval: bin k's sample (at its coverage centre) lerped toward
        #    the neighbour on the light's side when that bin is covered too
        gap = (mid_b + side) - mid_a
        t = np.clip((s - mid_a) / np.where(np.abs(gap) > 1e-9, gap, 1.0), 0.0, 1.0)
        t = np.where(has_a & has_b, t, np.where(has_a, 0.0, 1.0))
        int_a, int_b = (
            self._interval(vals_a, wts, nearest),
            self._interval(vals_b, wts, nearest),
        )
        if layer == FLOATING:
            lohi = int_a * (1.0 - t)[:, None] + int_b * t[:, None]
            cot_lo = self.decode_cot(lohi[:, 0])
            cot_hi = self.decode_cot(lohi[:, 1])
        else:
            # grounded: R is the first run's top, G the later runs'; the
            # light's sub-bin picks the run, and only the first run lerps
            # across bins (a leg's top barely varies over a bin)
            # Run boundaries for the whole tile at once: the light's sub-bin
            # picks the run, and G belongs to the SECOND one — a one-run tap
            # writes 0 there, so the later-run top blends only over the taps
            # that HAVE a second run.
            starts = self._run_starts(mask_a)  # (N, 4, 16)
            j = np.clip(np.floor(s * SUBBINS).astype(int), 0, SUBBINS - 1)
            run_id = np.cumsum(starts[idx, nearest], axis=1)[idx, j]
            multi = starts.sum(axis=-1) >= 2  # (N, 4)
            later_g = self._interval(vals_a, wts, nearest, covered=multi)[:, 1]
            later = (run_id >= 2) & (later_g > 0.0)
            hi_first = int_a[:, 0] * (1.0 - t) + int_b[:, 0] * t
            cot_lo = np.full(k.shape[0], self.max_stretch)
            cot_hi = self.decode_cot(np.where(later, later_g, hi_first))
        # -- the elevation test in cotangent space: blocked while
        #    cot(hi) <= cot(e) <= cot(lo); the disc is [cot(e+ρ), cot(e−ρ)]
        e_lo = np.maximum(elev - rho, 1e-6)
        e_hi = np.maximum(elev + rho, 1e-6)
        c_near, c_far = 1.0 / np.tan(e_hi), 1.0 / np.tan(e_lo)
        cov_e = self._overlap(c_near, c_far, cot_hi, cot_lo, 0.5 * (c_far - c_near))
        # ``cov_phi`` is already 0 when no tap of bins k-1..k+1 covers the
        # light's sub-bin, and it sums over all four taps — so gating on the
        # NEAREST tap alone would zero a texel its neighbours cover. The
        # contract's rule is "an all-zero mask at EVERY tap -> 0".
        return cov_phi * cov_e

    @staticmethod
    def _overlap(lo1, hi1, lo2, hi2, radius) -> np.ndarray:
        """Fraction of ``[lo1, hi1]`` (a disc of half-width *radius*) inside
        ``[lo2, hi2]``; a point-in-interval test when the disc collapses."""
        inter = np.maximum(0.0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2))
        width = 2.0 * np.asarray(radius, dtype=float)
        centre = (lo1 + hi1) * 0.5
        point = (centre >= lo2) & (centre <= hi2) & (hi2 > lo2)
        return np.where(
            width > 1e-9,
            inter / np.where(width > 1e-9, width, 1.0),
            point.astype(float),
        )


class ShadowHorizon:
    """Bake, measure and lay out coverage-aware horizon maps (module doc)."""

    #: Measured on box / table / chair fixtures against the exact projection
    #: (one-texel-tolerant disagreement, mean over random sources): 32 bins
    #: halve the error of 16 at every tile size; a 128-pixel footprint keeps
    #: a 5 cm leg 5 cm wide (32 pixels made it 13 cm); 64 radial rows score
    #: within half a point of 128; and 256 bearing columns are what keep a
    #: thin member's shadow crisp at distance (a 5 cm pole's shadow at 3 m:
    #: alpha 0.81 with 256 columns, 0.30 with 128, which smears it over a
    #: 15 cm texel). Box 1.2 %, table 3.1 %, chair 5.3 % at 4 MB (32 bins ×
    #: 2 layers × 256 × 64 × RGBA8) and about three seconds.
    DEFAULT_BINS = 32
    DEFAULT_SIZE = (256, 64)
    DEFAULT_FOOTPRINT = 128
    ENCODING = 1
    MAPPING = "logpolar"
    LAYERS = LAYERS
    SUBBINS = SUBBINS
    #: Bin counts :meth:`bake_adaptive` walks through.
    ADAPTIVE_BINS = (32, 64)
    #: A column whose lowest surface sits within this fraction of the object
    #: height above the ground counts as grounded.
    GROUND_EPS = 0.02

    #: The shared shader body, beside this module. One text, every consumer.
    SHADER_FILE = "shadow_horizon.glsl"
    #: Languages :meth:`shader_source` will spell the body in, and the macro
    #: that switches it. GLSL is the authored form, so it adds nothing.
    SHADER_LANGUAGES = {"glsl": "", "hlsl": "#define SH_HLSL 1\n"}

    @classmethod
    def shader_source(cls, language: str = "glsl") -> str:
        """The shared horizon evaluation, spelled for *language*.

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

    @staticmethod
    def layout(tiles: int) -> Tuple[int, int]:
        """``(cols, rows)`` of the grid holding *tiles*: ``cols = ceil(sqrt(tiles))``."""
        cols = int(math.ceil(math.sqrt(max(int(tiles), 1))))
        rows = int(math.ceil(int(tiles) / cols))
        return cols, rows

    @classmethod
    def range_for(
        cls, radius: float, height: float, max_stretch: Optional[float] = None
    ) -> Tuple[float, float]:
        """``(r_min, r_max)``: an eighth of the footprint radius to the reach
        cap ``radius + max_stretch × height``. The inner ring must sit well
        inside the footprint: the shadow of a leg or an overhang crosses the
        ground under the object, and a texel inside ``r_min`` clamps to the
        ring (measured: half a pole's shadow lost at ``radius / 2``)."""
        stretch = (
            ShadowProjection.DEFAULT_MAX_STRETCH
            if max_stretch is None
            else float(max_stretch)
        )
        r_min = max(0.125 * float(radius), 1e-3)
        r_max = max(float(radius) + stretch * float(height), r_min * 1.5)
        return r_min, r_max

    # -- bake ----------------------------------------------------------------
    #: Elements in one ``_march`` working array. ``_march`` vectorises every
    #: (texel, bin) ray against every sample, so it allocates about a dozen
    #: ``(chunk * bins, samples)`` arrays at once; budgeting their SIZE keeps
    #: the bake's footprint fixed, where a fixed texel count let ``bins`` and
    #: ``samples`` multiply it silently. Measured at the defaults on a 20-core
    #: box: a fixed 1024-texel chunk peaked at 4306 MiB for one chair bake and
    #: exhausted memory inside a full test run (``Unable to allocate 64.0 MiB``
    #: from ``_march``). This budget measured 616-762 MiB over three runs of
    #: the same bake at 5.52-5.81 s, against 6.01 s before -- ~6x less memory
    #: and no slower, because the smaller arrays stay closer to cache.
    RAY_BUDGET = 1 << 20

    #: Cap on worker threads when the caller names none. Measured 5.09 s at 8
    #: against 5.36 s at the ``ThreadPoolExecutor`` default of 24 on a 20-core
    #: box -- the extra threads bought no speed, only a bigger simultaneous
    #: working set, since each holds a full march in flight.
    MAX_BAKE_THREADS = 8

    @staticmethod
    def _solve_chunk_size(bins: int, samples: int) -> int:
        """Texels per solve chunk so one march stays inside :attr:`RAY_BUDGET`.

        Never zero: a chunk of 0 makes the job list empty and the bake returns
        an unwritten map rather than failing.
        """
        per_texel = max(1, int(bins) * int(samples))
        return max(1, int(ShadowHorizon.RAY_BUDGET) // per_texel)

    @classmethod
    def _bake_workers(cls, threads):
        """Worker count for the solve pool; *threads* wins when given.

        The cap applies only to the default. A caller that knows its machine
        passes a number, and ``threads=1`` still selects the serial path.
        """
        if threads:
            return int(threads)
        return min(cls.MAX_BAKE_THREADS, (os.cpu_count() or 1))

    @classmethod
    def bake(
        cls,
        meshes,
        *,
        ground: float = 0.0,
        up: int = 1,
        radius: Optional[float] = None,
        height: Optional[float] = None,
        bins: int = DEFAULT_BINS,
        size: Sequence[int] = DEFAULT_SIZE,
        r_min: Optional[float] = None,
        r_max: Optional[float] = None,
        max_stretch: Optional[float] = None,
        footprint: int = DEFAULT_FOOTPRINT,
        threads: Optional[int] = None,
    ) -> HorizonMap:
        """Bake the map of *meshes* in the map's frame.

        Parameters:
            meshes: Iterable of ``(points, tris)`` — ``(N, 3)`` points in the
                frame (the contact at the horizontal origin), ``(M, 3)``
                triangles.
            ground: Height of the ground plane along *up* in the frame.
            up: Vertical axis index (1 for Y-up, 2 for Z-up).
            radius, height: The projection model's footprint radius and
                object height; derived from the mesh bounds when None.
            bins: Azimuth bins K.
            size: ``(W, H)`` texels per tile.
            r_min, r_max: The log-polar range; :meth:`range_for` when None.
            max_stretch: The reach cap in object heights (the model default).
            footprint: Height-field resolution over the footprint.
            threads: Worker threads for the per-texel passes (numpy releases
                the GIL); ``None`` = the machine's count, ``1`` = serial.

        Returns:
            The :class:`HorizonMap`.
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
        a, b = ShadowProjection.horizontal_axes(up)
        allp = np.concatenate([p for p, _ in meshes], axis=0)
        mn, mx = allp.min(axis=0), allp.max(axis=0)
        if radius is None:
            radius = 0.5 * math.hypot(mx[a] - mn[a], mx[b] - mn[b])
        if height is None:
            height = mx[up] - float(ground)
        radius, height = max(float(radius), 1e-3), max(float(height), 1e-3)
        lo_r, hi_r = cls.range_for(radius, height, max_stretch)
        r_min = float(r_min) if r_min is not None else lo_r
        r_max = float(r_max) if r_max is not None else hi_r
        if r_max <= r_min:
            raise ValueError("ShadowHorizon.bake: r_max must exceed r_min.")
        w, h = int(size[0]), int(size[1])
        bins = int(bins)

        z_top, z_bot, mask, bounds = ImgUtils.rasterize_height_fields(
            meshes, up=up, size=int(footprint), ground=ground
        )
        fields = _HeightFields(
            z_top, z_bot, mask, bounds, ground_eps=cls.GROUND_EPS * height
        )
        stretch = (
            ShadowProjection.DEFAULT_MAX_STRETCH
            if max_stretch is None
            else float(max_stretch)
        )
        hmap = HorizonMap(
            bins,
            (w, h),
            r_min,
            r_max,
            float(ground),
            int(up),
            stretch,
            np.zeros((LAYERS, bins, h, w, 4), np.float32),
        )
        if not mask.any():
            return hmap
        texels = hmap.texel_positions().reshape(-1, 2)
        workers = cls._bake_workers(threads)
        chunk = cls._solve_chunk_size(bins, fields.samples)
        jobs = [(s, texels[s : s + chunk]) for s in range(0, texels.shape[0], chunk)]

        def solve(job):
            start, pts = job
            return start, cls._solve_chunk(pts, fields, bins, hmap.encode_angle)

        if workers == 1:
            results = [solve(j) for j in jobs]
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(solve, jobs))
        scratch = np.zeros((h * w, LAYERS, bins, 4), dtype=np.float32)  # texel-major
        for start, block in results:
            scratch[start : start + block.shape[0]] = block
        data = np.transpose(scratch.reshape(h, w, LAYERS, bins, 4), (2, 3, 0, 1, 4))
        return hmap._replace(data=np.ascontiguousarray(data, dtype=np.float32))

    @classmethod
    def _solve_chunk(
        cls, texels: np.ndarray, fields: "_HeightFields", bins: int, encode
    ) -> np.ndarray:
        """``(n, 2, bins, 4)`` for a chunk of texel positions ``(n, 2)``;
        *encode* is :meth:`HorizonMap.encode_angle`."""
        n = texels.shape[0]
        out = np.zeros((n, LAYERS, bins, 4), dtype=np.float32)
        for layer in range(LAYERS):
            mask = fields.layer_mask(layer)
            if not mask.any():
                continue
            occ = cls._occupancy(
                texels, fields.outline_points(mask), fields, bins
            )  # (n, bins, 16)
            if layer == GROUNDED:
                # two runs of set sub-bins get their own top elevation: two
                # legs in one bin must not share the nearer leg's shadow length
                run1, run2 = cls._runs(occ)
                _, hi1, keep1 = cls._intervals(texels, run1, mask, fields, bins)
                _, hi2, keep2 = cls._intervals(texels, run2, mask, fields, bins)
                occ = (run1 & keep1[:, :, None]) | (run2 & keep2[:, :, None])
                keep = keep1 | keep2
                out[:, layer, :, 0] = np.where(
                    keep1, encode(hi1), np.where(keep2, encode(hi2), 0.0)
                )
                out[:, layer, :, 1] = np.where(keep1 & keep2, encode(hi2), 0.0)
            else:
                lo, hi, keep = cls._intervals(texels, occ, mask, fields, bins)
                occ &= keep[:, :, None]
                out[:, layer, :, 0] = np.where(keep, encode(lo), 0.0)
                out[:, layer, :, 1] = np.where(keep, encode(hi), 0.0)
            out[:, layer, :, 2] = (occ[:, :, :8] * _BIT_WEIGHTS).sum(axis=2) / 255.0
            out[:, layer, :, 3] = (occ[:, :, 8:] * _BIT_WEIGHTS).sum(axis=2) / 255.0
        return out

    @staticmethod
    def _runs(occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split ``(n, bins, 16)`` occupancy into its first run of set
        sub-bins and the union of every later run."""
        prev = np.concatenate([np.zeros_like(occ[:, :, :1]), occ[:, :, :-1]], axis=2)
        starts = occ & ~prev
        run_id = np.cumsum(starts, axis=2)  # 1 inside the first run, 2 in the second, …
        first = occ & (run_id == 1)
        rest = occ & (run_id >= 2)
        return first, rest

    @staticmethod
    def _occupancy(
        texels: np.ndarray, outline: np.ndarray, fields: "_HeightFields", bins: int
    ) -> np.ndarray:
        """``(n, bins, 16)`` sub-bin occupancy from the outline's angular
        extents.

        Every outline pixel subtends an angular interval from the texel (its
        centre's bearing widened by the pixel's half-diagonal); a sub-bin is
        set when any interval crosses it. Built as a difference array over
        the ``bins × 16`` sub-bins of the full circle, so wrap-around costs
        nothing.
        """
        n = texels.shape[0]
        total = bins * SUBBINS
        step_sub = _TWO_PI / total
        dx = outline[None, :, 0] - texels[:, 0, None]
        dy = outline[None, :, 1] - texels[:, 1, None]
        d = np.hypot(dx, dy)
        centre = np.mod(np.arctan2(dy, dx), _TWO_PI) / step_sub
        widen = np.arctan2(fields.half_diag, np.maximum(d, 1e-9)) / step_sub
        start = np.floor(centre - widen).astype(int)
        end = np.ceil(centre + widen).astype(int)  # exclusive
        length = np.clip(end - start, 1, total)
        s0 = np.mod(start, total)
        s1 = s0 + length
        rows = np.broadcast_to(np.arange(n)[:, None], s0.shape)
        diff = np.zeros((n, total + 1), dtype=np.int32)
        wrap = s1 > total
        keep = ~wrap
        np.add.at(diff, (rows[keep], s0[keep]), 1)
        np.add.at(diff, (rows[keep], s1[keep]), -1)
        if wrap.any():
            np.add.at(diff, (rows[wrap], s0[wrap]), 1)
            np.add.at(diff, (rows[wrap], np.full(int(wrap.sum()), total)), -1)
            np.add.at(diff, (rows[wrap], np.zeros(int(wrap.sum()), dtype=int)), 1)
            np.add.at(diff, (rows[wrap], s1[wrap] - total), -1)
        occ = np.cumsum(diff[:, :total], axis=1) > 0
        return occ.reshape(n, bins, SUBBINS)

    @classmethod
    def _intervals(
        cls,
        texels: np.ndarray,
        occ: np.ndarray,
        mask: np.ndarray,
        fields: "_HeightFields",
        bins: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(lo, hi, keep)`` ``(n, bins)`` — the elevation interval in
        radians along each covered bin's coverage-centre ray (the midpoint
        of its first and last set sub-bins); a bin whose rays all miss the
        layer is dropped (``keep`` false)."""
        n = texels.shape[0]
        step = _TWO_PI / bins
        lo = np.zeros((n, bins), dtype=np.float32)
        hi = np.zeros((n, bins), dtype=np.float32)
        keep = np.zeros((n, bins), dtype=bool)
        covered = occ.any(axis=2)
        ti, ki = np.nonzero(covered)
        if ti.size == 0:
            return lo, hi, keep
        sub = occ[ti, ki]  # (m, 16)
        first = np.argmax(sub, axis=1)
        last = SUBBINS - 1 - np.argmax(sub[:, ::-1], axis=1)
        centre = 0.5 * (first + last + 1) / SUBBINS
        origin = texels[ti]
        lo_c, hi_c, hit = cls._march(origin, (ki + centre) * step, mask, fields)
        # a centre ray that misses (two members with a gap between, or a
        # thin diagonal at pixel resolution): the union of the first and
        # last sub-bin rays
        miss = ~hit
        if miss.any():
            lf, hf, hitf = cls._march(
                origin[miss],
                (ki[miss] + (first[miss] + 0.5) / SUBBINS) * step,
                mask,
                fields,
            )
            ll, hl, hitl = cls._march(
                origin[miss],
                (ki[miss] + (last[miss] + 0.5) / SUBBINS) * step,
                mask,
                fields,
            )
            lo_u = np.minimum(np.where(hitf, lf, np.inf), np.where(hitl, ll, np.inf))
            hi_u = np.maximum(np.where(hitf, hf, -np.inf), np.where(hitl, hl, -np.inf))
            any_hit = hitf | hitl
            lo_c[miss] = np.where(any_hit, lo_u, 0.0)
            hi_c[miss] = np.where(any_hit, hi_u, 0.0)
            hit[miss] = any_hit
        lo[ti, ki] = np.where(hit, lo_c, 0.0)
        hi[ti, ki] = np.where(hit, hi_c, 0.0)
        keep[ti, ki] = hit
        return lo, hi, keep

    @staticmethod
    def _march(
        origin: np.ndarray,
        bearing: np.ndarray,
        mask: np.ndarray,
        fields: "_HeightFields",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """March rays from ``origin`` ``(n, 2)`` at ``bearing`` ``(n,)`` through
        the layer's columns (*mask*): ``(lo, hi, hit)`` elevation extremes in
        radians."""
        dirx, diry = np.cos(bearing), np.sin(bearing)
        a0, a1, b0, b1 = fields.bounds
        with np.errstate(divide="ignore", invalid="ignore"):
            tx0 = (a0 - origin[:, 0]) / dirx
            tx1 = (a1 - origin[:, 0]) / dirx
            ty0 = (b0 - origin[:, 1]) / diry
            ty1 = (b1 - origin[:, 1]) / diry
        tmin = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
        tmax = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
        tmin = np.nan_to_num(tmin, nan=0.0, posinf=0.0, neginf=0.0)
        tmax = np.nan_to_num(tmax, nan=-1.0, posinf=1e9, neginf=-1.0)
        enter = np.maximum(tmin, 0.0)
        valid = tmax > enter
        steps = fields.samples
        t = (
            enter[:, None]
            + (tmax - enter)[:, None] * (np.arange(steps) + 0.5)[None, :] / steps
        )
        px = origin[:, 0, None] + dirx[:, None] * t
        py = origin[:, 1, None] + diry[:, None] * t
        ia, ib = fields.pixel_index(px, py)
        inb = (
            (ia >= 0)
            & (ia < fields.size)
            & (ib >= 0)
            & (ib < fields.size)
            & valid[:, None]
        )
        ia_c, ib_c = np.clip(ia, 0, fields.size - 1), np.clip(ib, 0, fields.size - 1)
        solid = inb & mask[ib_c, ia_c]
        dist = np.maximum(t, 1e-9)
        e_top = np.arctan2(fields.z_top[ib_c, ia_c], dist)
        e_bot = np.arctan2(fields.z_bot[ib_c, ia_c], dist)
        hi = np.where(solid, e_top, -np.inf).max(axis=1)
        lo = np.where(solid, e_bot, np.inf).min(axis=1)
        hit = solid.any(axis=1)
        return np.where(hit, lo, 0.0), np.where(hit, hi, 0.0), hit

    # -- measure ---------------------------------------------------------------
    @classmethod
    def measure(
        cls,
        hmap: HorizonMap,
        meshes,
        *,
        samples: int = 8,
        size: int = 256,
        seed: int = 0,
        max_stretch: Optional[float] = None,
        radius: Optional[float] = None,
        height: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compare :meth:`HorizonMap.alpha` with the exact projection at random
        source positions.

        For each sample :meth:`ImgUtils.rasterize_shadow` draws the hard shadow,
        every canvas texel is mapped back to the frame, and the two binary
        masks are compared inside ``r_max``.

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
        contact = np.zeros(3)
        contact[hmap.up] = hmap.ground
        rng = np.random.default_rng(seed)
        scores: List[float] = []
        tolerant: List[float] = []
        for _ in range(int(samples)):
            bearing = rng.uniform(0.0, _TWO_PI)
            elev = math.radians(rng.uniform(12.0, 70.0))
            dist = rng.uniform(1.5, 3.0) * hmap.r_max
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
            inside = (np.hypot(pts[:, a], pts[:, b]) <= hmap.r_max).reshape(size, size)
            e, g = exact & inside, got & inside
            union = float((e | g).sum())
            if not union:
                scores.append(0.0)
                tolerant.append(0.0)
                continue
            scores.append(float((e ^ g).sum() / union))
            slack = ((e & ~cls._dilate(g)) | (g & ~cls._dilate(e))).sum()
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
        max_bins: int = 64,
        measure_samples: int = 6,
        **kwargs,
    ) -> Tuple[HorizonMap, Dict[str, float]]:
        """Bake at :attr:`ADAPTIVE_BINS` in turn until :meth:`measure`'s
        one-texel-tolerant mean disagreement is under *threshold* or
        *max_bins* is reached.

        *max_bins* caps the ladder; it does not choose a count. A value below
        the smallest rung clamps up to it with a warning rather than returning
        no map — the ladder is the menu, and the declared return type has no
        ``None`` in it.

        Returns:
            ``(map, score)`` — the last map baked and its measurement.

        Raises:
            TypeError: ``bins`` was passed. It is a ``bake`` keyword and an
                adaptive bake chooses the count itself.
        """
        if "bins" in kwargs:
            raise TypeError(
                f"{cls.__name__}.bake_adaptive() chooses its own bin count from "
                f"ADAPTIVE_BINS; pass max_bins= to cap the ladder, or call "
                f"{cls.__name__}.bake(bins=...) for a fixed count."
            )
        measure_kw = {
            k: kwargs[k] for k in ("max_stretch", "radius", "height") if k in kwargs
        }
        rungs = [b for b in cls.ADAPTIVE_BINS if b <= int(max_bins)]
        if not rungs:
            rungs = [cls.ADAPTIVE_BINS[0]]
            warnings.warn(
                f"max_bins={max_bins} is below the smallest adaptive rung "
                f"({cls.ADAPTIVE_BINS[0]}); baking at {cls.ADAPTIVE_BINS[0]}. "
                f"Call {cls.__name__}.bake(bins=...) for a count off the ladder.",
                RuntimeWarning,
                stacklevel=2,
            )
        hmap, score = None, {}
        for bins in rungs:
            hmap = cls.bake(meshes, bins=bins, **kwargs)
            score = cls.measure(hmap, meshes, samples=measure_samples, **measure_kw)
            score["bins"] = bins
            if score["tolerant_mean"] <= threshold:
                break
        return hmap, score


class _HeightFields:
    """The footprint's top and bottom surfaces plus the helpers the bake needs."""

    def __init__(
        self,
        z_top: np.ndarray,
        z_bot: np.ndarray,
        mask: np.ndarray,
        bounds: Tuple[float, float, float, float],
        *,
        ground_eps: float = 0.0,
    ):
        self.z_top = np.asarray(z_top, dtype=float)
        self.z_bot = np.asarray(z_bot, dtype=float)
        self.mask = np.asarray(mask, dtype=bool)
        self.bounds = tuple(float(v) for v in bounds)
        self.size = int(self.mask.shape[0])
        a0, a1, b0, b1 = self.bounds
        self.px = (a1 - a0) / self.size
        self.py = (b1 - b0) / self.size
        self.half_diag = 0.5 * math.hypot(self.px, self.py)
        self.samples = max(2 * self.size, 8)
        self.grounded = self.mask & (self.z_bot <= float(ground_eps))
        self.floating = self.mask & ~self.grounded

    def layer_mask(self, layer: int) -> np.ndarray:
        return self.grounded if layer == GROUNDED else self.floating

    def pixel_index(
        self, pa: np.ndarray, pb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integer pixel indices ``(ia, ib)`` of frame points (unclamped)."""
        a0, _, b0, _ = self.bounds
        return np.floor((pa - a0) / self.px).astype(int), np.floor(
            (pb - b0) / self.py
        ).astype(int)

    def outline_points(self, mask: np.ndarray) -> np.ndarray:
        """Centres ``(M, 2)`` of *mask*'s outline pixels (a solid pixel with an
        empty 4-neighbour or on the border) — where the angular extremes live."""
        pad = np.pad(mask, 1, constant_values=False)
        inner = pad[1:-1, 2:] & pad[1:-1, :-2] & pad[2:, 1:-1] & pad[:-2, 1:-1]
        edge = mask & ~inner
        if not edge.any():
            edge = mask
        ib, ia = np.nonzero(edge)
        a0, _, b0, _ = self.bounds
        return np.column_stack([a0 + (ia + 0.5) * self.px, b0 + (ib + 0.5) * self.py])
