# !/usr/bin/python
# coding=utf-8
"""Geometry rasterized into images: UV triangles, silhouettes, height fields
and shadows (the bodies behind the :class:`ImgUtils` facade).
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


class _ImgRasterizeInternal:
    """Bodies of :class:`ImgUtils`' rasterizers (an ``ImgUtils`` base).

    The public signatures and docstrings stay on the facade; each delegates
    here.
    """

    @staticmethod
    def radial_gradient(
        size: Tuple[int, int],
        center: Tuple[float, float] = (0.5, 0.5),
        max_radius: Optional[float] = None,
        falloff_power: float = 1.0,
        invert: bool = False,
        dtype: type = None,
    ) -> "np.ndarray":
        """Body of :meth:`ImgUtils.radial_gradient`."""
        w, h = int(size[0]), int(size[1])
        cx = float(center[0]) * (w - 1)
        cy = float(center[1]) * (h - 1)

        if max_radius is None:
            max_radius = math.hypot(w - 1, h - 1)
        max_radius = max(float(max_radius), 1.0)

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        norm = np.clip(dist / max_radius, 0.0, 1.0)
        if falloff_power != 1.0:
            norm = norm ** float(falloff_power)
        result = 1.0 - norm
        if invert:
            result = 1.0 - result

        if dtype is None or dtype == np.float32:
            return result.astype(np.float32, copy=False)
        if dtype == np.uint8:
            return (result * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        return result.astype(dtype, copy=False)

    @classmethod
    def rasterize_uv_triangles(
        cls,
        triangles,
        size: int = 512,
        supersample: int = 4,
    ) -> "np.ndarray":
        """Body of :meth:`ImgUtils.rasterize_uv_triangles`."""
        tris = np.asarray(triangles, dtype=float).reshape(-1, 3, 2)
        ss = max(1, int(supersample))
        dim = int(size) * ss
        pts = np.stack([tris[..., 0] * dim, (1.0 - tris[..., 1]) * dim], axis=-1)
        mask = cls._scanline_fill(pts, dim)
        if ss > 1:
            # Accumulate each ss x ss block into one output-sized integer
            # buffer rather than casting the whole supersampled grid to
            # float32: that cast is a transient FOUR TIMES the size of the
            # grid it reduces -- measured 268 MB to downsample a 2048 map at
            # supersample 4 -- allocated inside whatever host process is
            # baking. The arithmetic is unchanged (every sample is 0 or 255,
            # so the block mean is its sum over ss*ss either way) and the
            # arithmetic stays integer end to end, so no float copy of either
            # the grid or the result is ever materialized.
            view = mask.reshape(size, ss, size, ss)
            n = ss * ss
            acc = np.zeros((size, size), dtype=np.uint32)
            for i in range(ss):
                for j in range(ss):
                    acc += view[:, i, :, j]
            # Round half up. Every sample is 0 or 255, so the only quotient
            # that can land exactly on .5 is a half-covered texel (128 either
            # way, matching the round-half-to-even this replaces), and the
            # maximum is 255 exactly -- nothing to clip.
            mask = ((acc + n // 2) // n).astype(np.uint8)
        return mask

    #: Cells of the row difference array :meth:`_scanline_fill` holds at once
    #: (a band of rows): 8 bytes each, 32 MB, where the whole grid's is 537 MB
    #: for a 2048 map at supersample 4.
    _SCANLINE_CELLS = 1 << 22
    #: Row spans :meth:`_scanline_fill` builds at once. Each holds ~140 bytes
    #: of scratch while it is cut against its triangle's edges, so this keeps
    #: that near 70 MB however tall or many the triangles are.
    _SCANLINE_SPANS = 1 << 19

    @classmethod
    def _scanline_fill(cls, pts: "np.ndarray", dim: int) -> "np.ndarray":
        """``(dim, dim)`` uint8 (0/255) of the pixel CENTERS inside any triangle of *pts*.

        :meth:`_fill_triangle`'s rule exactly -- a centre on an edge is inside,
        vertices stay in floating point, geometry past the frame is cropped --
        for many triangles at once: each covered row of each triangle becomes
        one ``[first, last]`` span of centres (:meth:`_row_spans`), written as
        +1/-1 into a row difference array and integrated once. The
        per-triangle loop it replaces paid a Python iteration and a float grid
        over each triangle's whole bbox: 13 s of a 96-tile production bake's
        refill, against well under one here.

        Its scratch is bounded, inside whatever host process is baking: the
        difference array is held a band of rows at a time
        (:attr:`_SCANLINE_CELLS`), and a band's spans are built a chunk of
        triangles at a time (:attr:`_SCANLINE_SPANS`). Built all at once, the
        spans cost scratch in proportion to the triangles' SUMMED height --
        2048 tall strips at 2048 x 4 supersample took 2.0 GB.
        """
        mask = np.zeros((dim, dim), dtype=np.uint8)
        if not len(pts):
            return mask
        a, b, c = pts[:, 0], pts[:, 1], pts[:, 2]
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (
            b[:, 1] - a[:, 1]
        )
        pts = pts[np.abs(area) > 1e-12]  # degenerate triangles cover nothing
        if not len(pts):
            return mask
        ys = pts[..., 1]
        r0 = np.clip(np.ceil(ys.min(1) - 0.5), 0, dim).astype(np.int64)
        r1 = np.clip(np.floor(ys.max(1) - 0.5), -1, dim - 1).astype(np.int64)
        width = dim + 1
        band = max(1, int(cls._SCANLINE_CELLS) // width)
        budget = max(1, int(cls._SCANLINE_SPANS))
        for top in range(0, dim, band):
            rows_here = min(band, dim - top)
            first = np.maximum(r0, top)
            count = np.minimum(r1, top + rows_here - 1) - first + 1
            here = np.flatnonzero(count > 0)
            if not len(here):
                continue
            size = rows_here * width
            diff = None
            # Coverage is a union, so chunks of triangles add into the band
            # independently: each chunk's spans within the budget, a triangle
            # taller than it on its own a chunk by itself.
            ends = np.cumsum(count[here])
            start = 0
            while start < len(here):
                done = int(ends[start - 1]) if start else 0
                stop = max(
                    start + 1,
                    int(np.searchsorted(ends, done + budget, side="right")),
                )
                tris = here[start:stop]
                row, c0, c1 = cls._row_spans(pts[tris], first[tris], count[tris], dim)
                local = (row - top) * width
                if diff is None:
                    diff = np.bincount(local + c0, minlength=size)
                else:
                    diff += np.bincount(local + c0, minlength=size)
                diff -= np.bincount(local + c1 + 1, minlength=size)
                start = stop
            covered = np.cumsum(diff.reshape(rows_here, width), axis=1)[:, :dim] > 0
            mask[top : top + rows_here][covered] = 255
        return mask

    @staticmethod
    def _row_spans(pts, first, count, dim: int):
        """``(row, c0, c1)``: the centres ``c0..c1`` each triangle of *pts*
        covers on each of its *count* rows from *first* (rows whose span holds
        no centre left out)."""
        tri = np.repeat(np.arange(len(pts)), count)
        row = np.arange(len(tri)) - np.repeat(np.cumsum(count) - count, count)
        row = row + first[tri]
        y = row + 0.5
        lo = np.full(len(row), np.inf)
        hi = np.full(len(row), -np.inf)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            p, q = pts[tri, i], pts[tri, j]
            dy = q[:, 1] - p[:, 1]
            crosses = (y >= np.minimum(p[:, 1], q[:, 1])) & (
                y <= np.maximum(p[:, 1], q[:, 1])
            )
            flat = crosses & (dy == 0)
            slant = crosses & (dy != 0)
            t = np.where(slant, (y - p[:, 1]) / np.where(dy == 0, 1.0, dy), 0.0)
            x = p[:, 0] + t * (q[:, 0] - p[:, 0])
            lo = np.where(slant, np.minimum(lo, x), lo)
            hi = np.where(slant, np.maximum(hi, x), hi)
            # A horizontal edge lying ON the row: its whole span is inside.
            lo = np.where(flat, np.minimum(lo, np.minimum(p[:, 0], q[:, 0])), lo)
            hi = np.where(flat, np.maximum(hi, np.maximum(p[:, 0], q[:, 0])), hi)
        c0 = np.clip(np.ceil(lo - 0.5), 0, dim).astype(np.int64)
        c1 = np.clip(np.floor(hi - 0.5), -1, dim - 1).astype(np.int64)
        keep = c1 >= c0
        return row[keep], c0[keep], c1[keep]

    @staticmethod
    def _fill_triangle(mask, tri, value=255):
        """Fill a 2D triangle (3x2 float pixel coords) into ``mask`` with *value*
        (255 by default; into a float buffer the value is MAX-composited, so a
        per-triangle scalar field — a penumbra width — survives overlaps).

        Samples at pixel CENTERS and keeps the vertices in floating point.
        Both matter to any caller that reads the downsampled result as a
        coverage FRACTION: testing at pixel corners offsets every edge by
        half a sample, and rounding the vertices onto the sample grid first
        moves them by up to a whole one. Against a supersampled grid those
        biases are sub-texel, so they never showed as a visibly wrong mask --
        but a consumer thresholding on FULL coverage reads the result as
        "this texel lies entirely inside the shape" when part of it does not,
        which is the one question a coverage mask exists to answer exactly.

        Geometry outside the image is CROPPED (the bbox is clamped, the
        vertices are not): clamping a vertex would drag the edge it belongs
        to across the image and smear a triangle that merely overhangs into
        a wedge along the border.
        """
        h, w = mask.shape[:2]
        xs, ys = tri[:, 0], tri[:, 1]
        x0, x1 = int(np.floor(xs.min())), int(np.ceil(xs.max()))
        y0, y1 = int(np.floor(ys.min())), int(np.ceil(ys.max()))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w - 1), min(y1, h - 1)
        if x1 < x0 or y1 < y0:
            return  # wholly outside the image
        ax, ay = float(tri[0][0]), float(tri[0][1])
        bx, by = float(tri[1][0]), float(tri[1][1])
        cx, cy = float(tri[2][0]), float(tri[2][1])
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denom) < 1e-12:
            return  # degenerate
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        xx = xx + 0.5
        yy = yy + 0.5
        l1 = ((by - cy) * (xx - cx) + (cx - bx) * (yy - cy)) / denom
        l2 = ((cy - ay) * (xx - cx) + (ax - cx) * (yy - cy)) / denom
        l3 = 1.0 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        sub = mask[y0 : y1 + 1, x0 : x1 + 1]
        if mask.dtype.kind == "f":
            sub[inside] = np.maximum(sub[inside], value)
        else:
            sub[inside] = value

    @classmethod
    def _contact_falloff(cls, mask, falloff_source, falloff_power, vertical_weight):
        """Radial + vertical contact-falloff weight (0..1) for a silhouette mask (pre-flip coords)."""
        h, w = mask.shape
        rows = np.where(mask.max(axis=1) > 0)[0]
        cols = np.where(mask.max(axis=0) > 0)[0]
        if not (len(rows) and len(cols)):
            return np.ones((h, w), dtype=np.float32)
        top_row, bottom_row = rows[0], rows[-1]
        center_col = (cols[0] + cols[-1]) // 2
        if falloff_source is not None:  # saved-PNG coords -> pre-flip (v mirrored)
            src = (float(falloff_source[0]), 1.0 - float(falloff_source[1]))
        else:
            src = (center_col / max(w - 1, 1), bottom_row / max(h - 1, 1))
        radial_w = max(1.0 - vertical_weight, 0.0)
        vertical_w = max(min(vertical_weight, 1.0), 0.0)
        radial = cls.radial_gradient(
            (w, h),
            center=src,
            max_radius=max(bottom_row - top_row, 1),
            falloff_power=falloff_power,
        )
        vertical = np.zeros((h, w), dtype=np.float32)
        span = max(bottom_row - top_row, 1)
        t = np.clip((np.arange(h) - top_row) / span, 0.0, 1.0) ** 0.6
        vertical[:, :] = t[:, None]
        vertical[np.arange(h) < top_row, :] = 0.0
        vertical[np.arange(h) > bottom_row, :] = 1.0
        return radial * radial_w + vertical * vertical_w

    @classmethod
    def rasterize_silhouette(
        cls,
        meshes,
        size=512,
        axis="auto",
        *,
        uniform_alpha=False,
        falloff_source=None,
        falloff_power=0.8,
        vertical_weight=0.3,
        blur_amount=1.5,
    ):
        """Body of :meth:`ImgUtils.rasterize_silhouette`."""
        meshes = [
            (
                np.asarray(p, dtype=float).reshape(-1, 3),
                np.asarray(t, dtype=np.int64).reshape(-1, 3),
            )
            for p, t in meshes
            if len(p) and len(t)
        ]
        if not meshes:
            raise ValueError("rasterize_silhouette: no geometry provided.")

        all_pts = np.concatenate([p for p, _ in meshes], axis=0)
        mn, mx = all_pts.min(axis=0), all_pts.max(axis=0)
        a = axis.lower()
        if a == "auto":
            a = "x" if (mx[2] - mn[2]) > (mx[0] - mn[0]) else "z"
        u_idx, v_idx = {"y": (0, 2), "x": (2, 1)}.get(a, (0, 1))
        # 1.1 = 10% padding; `or 1.0` guards a zero-extent (single-point/degenerate) mesh against /0.
        extent = max(mx[u_idx] - mn[u_idx], mx[v_idx] - mn[v_idx]) * 1.1 or 1.0
        u_c, v_c = (mn[u_idx] + mx[u_idx]) / 2.0, (mn[v_idx] + mx[v_idx]) / 2.0

        mask = np.zeros((size, size), dtype=np.uint8)
        for pts, tris in meshes:
            pu = np.clip(
                ((pts[:, u_idx] - u_c) / extent + 0.5) * size, 0, size - 1
            ).astype(np.int32)
            pv = np.clip(
                (1.0 - ((pts[:, v_idx] - v_c) / extent + 0.5)) * size, 0, size - 1
            ).astype(np.int32)
            proj = np.stack([pu, pv], axis=1)
            for tri in tris:
                cls._fill_triangle(mask, proj[tri])

        if blur_amount and blur_amount > 0:
            mask = cls.gaussian_blur(mask, radius=blur_amount)
        combined = (
            np.ones(mask.shape, dtype=np.float32)
            if uniform_alpha
            else cls._contact_falloff(
                mask, falloff_source, falloff_power, vertical_weight
            )
        )
        alpha = np.flipud(
            (mask.astype(np.float32) / 255.0 * combined * 255).astype(np.uint8)
        )
        result = np.zeros((size, size, 4), dtype=np.uint8)
        result[:, :, 3] = alpha
        return result

    @classmethod
    def rasterize_height_fields(
        cls,
        meshes,
        *,
        up: int = 1,
        size: int = 64,
        ground: float = 0.0,
        bounds=None,
        padding: float = 0.02,
    ):
        """Body of :meth:`ImgUtils.rasterize_height_fields`."""

        lo, hi, bounds = cls.rasterize_height_spans(
            meshes, up=up, size=size, ground=ground, bounds=bounds, padding=padding
        )
        mask = ~np.isnan(hi[0])
        z_top = np.where(mask, hi[0], 0.0).astype(np.float32)
        z_bot = np.where(mask, lo[0], 0.0).astype(np.float32)
        return z_top, z_bot, mask, bounds

    @classmethod
    def rasterize_height_spans(
        cls,
        meshes,
        *,
        up: int = 1,
        size: int = 64,
        ground: float = 0.0,
        bounds=None,
        padding: float = 0.02,
        spans: int = 1,
    ):
        """Body of :meth:`ImgUtils.rasterize_height_spans`."""
        from pythontk.geo_utils.shadow_projection import ShadowProjection

        a, b = ShadowProjection.horizontal_axes(up)
        spans = max(int(spans), 1)
        tris_all = []
        for pts, tris in meshes:
            pts = np.asarray(pts, dtype=float).reshape(-1, 3)
            tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
            if len(pts) and len(tris):
                tris_all.append(pts[tris])  # (M, 3, 3)
        size = int(size)
        lo = np.full((spans, size, size), np.nan, np.float32)
        hi = np.full((spans, size, size), np.nan, np.float32)
        default_bounds = (
            (0.0, 1.0, 0.0, 1.0) if bounds is None else tuple(float(v) for v in bounds)
        )
        if not tris_all:
            return lo, hi, default_bounds
        T = np.concatenate(tris_all, axis=0)
        # Heights above the ground, clamped AT the ground: a face below it
        # still crosses a column -- a box cut by the ground plane enters the
        # solid at its buried floor -- and clamping puts that crossing on the
        # ground, where the column's span then starts. A wholly buried mesh
        # collapses to spans of no height, which block nothing and are dropped
        # below.
        z = np.maximum(T[:, :, up] - float(ground), 0.0)
        pa, pb = T[:, :, a], T[:, :, b]
        if bounds is None:
            lo_a, hi_a, lo_b, hi_b = pa.min(), pa.max(), pb.min(), pb.max()
            pad = float(padding) * max(hi_a - lo_a, hi_b - lo_b, 1e-3) + 1e-6
            bounds = (lo_a - pad, hi_a + pad, lo_b - pad, hi_b + pad)
        a0, a1, b0, b1 = (float(v) for v in bounds)
        sa, sb = max(a1 - a0, 1e-9), max(b1 - b0, 1e-9)
        x = (pa - a0) / sa * size  # (M, 3) pixel coords
        y = (pb - b0) / sb * size
        # Which way each face looks along up: a vertical ray ENTERS the solid
        # through a face looking down and LEAVES through one looking up, so
        # the crossings of a closed mesh count in and out whatever its
        # winding, and a duplicate from a shared edge is a duplicate.
        normal_up = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])[:, up]
        facing = np.where(normal_up < 0.0, 1, -1).astype(np.int8)
        # -- surface crossings at pixel centres, per triangle
        cross_px: list = []
        cross_z: list = []
        cross_sign: list = []
        for i in range(len(T)):
            xs, ys, zs = x[i], y[i], z[i]
            x0, x1 = int(np.floor(xs.min())), int(np.ceil(xs.max()))
            y0, y1 = int(np.floor(ys.min())), int(np.ceil(ys.max()))
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, size - 1), min(y1, size - 1)
            if x1 < x0 or y1 < y0:
                continue
            ax_, ay_ = xs[0], ys[0]
            bx_, by_ = xs[1], ys[1]
            cx_, cy_ = xs[2], ys[2]
            denom = (by_ - cy_) * (ax_ - cx_) + (cx_ - bx_) * (ay_ - cy_)
            if abs(denom) < 1e-12:
                continue
            yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
            xx = xx + 0.5
            yy = yy + 0.5
            l1 = ((by_ - cy_) * (xx - cx_) + (cx_ - bx_) * (yy - cy_)) / denom
            l2 = ((cy_ - ay_) * (xx - cx_) + (ax_ - cx_) * (yy - cy_)) / denom
            l3 = 1.0 - l1 - l2
            inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
            if not inside.any():
                continue
            zz = l1 * zs[0] + l2 * zs[1] + l3 * zs[2]
            cross_px.append(((yy - 0.5) * size + (xx - 0.5))[inside].astype(np.int64))
            cross_z.append(zz[inside].astype(np.float64))
            cross_sign.append(np.full(int(inside.sum()), facing[i], np.int8))
        # -- edge splat at half-pixel steps, all edges at once: the hull of
        #    the samples a pixel receives, one span, where the fill missed
        e0 = np.concatenate([x[:, [0, 1]], x[:, [1, 2]], x[:, [2, 0]]], axis=0)
        e1 = np.concatenate([y[:, [0, 1]], y[:, [1, 2]], y[:, [2, 0]]], axis=0)
        ez = np.concatenate([z[:, [0, 1]], z[:, [1, 2]], z[:, [2, 0]]], axis=0)
        length = np.hypot(e0[:, 1] - e0[:, 0], e1[:, 1] - e1[:, 0])
        counts = np.minimum(np.ceil(length * 2.0).astype(int) + 1, 4 * size)
        total = int(counts.sum())
        edge_id = np.repeat(np.arange(len(counts)), counts)
        offsets = np.cumsum(counts) - counts
        frac = (np.arange(total) - offsets[edge_id]) / np.maximum(
            counts[edge_id] - 1, 1
        )
        sx = e0[edge_id, 0] + (e0[edge_id, 1] - e0[edge_id, 0]) * frac
        sy = e1[edge_id, 0] + (e1[edge_id, 1] - e1[edge_id, 0]) * frac
        sz = ez[edge_id, 0] + (ez[edge_id, 1] - ez[edge_id, 0]) * frac
        ia = np.floor(sx).astype(int)
        ib = np.floor(sy).astype(int)
        inb = (ia >= 0) & (ia < size) & (ib >= 0) & (ib < size)
        splat_top = np.full(size * size, -np.inf)
        splat_bot = np.full(size * size, np.inf)
        flat = ib[inb] * size + ia[inb]
        np.maximum.at(splat_top, flat, sz[inb])
        np.minimum.at(splat_bot, flat, sz[inb])
        # -- the crossings into spans, pixel by pixel: sorted by height, the
        #    running count of entries minus exits is non-zero INSIDE the
        #    solid, so each gap between consecutive crossings with a non-zero
        #    count is a span. Every crossing also stands as a zero-thickness
        #    span of its own: an open surface (a single-sided plank) still
        #    blocks at its height, and inside a closed span it merges away.
        filled = np.zeros(size * size, bool)
        if cross_px:
            px = np.concatenate(cross_px)
            zc = np.concatenate(cross_z)
            sg = np.concatenate(cross_sign).astype(np.int64)
            order = np.lexsort((sg, zc, px))
            px, zc, sg = px[order], zc[order], sg[order]
            # a shared edge reports the same surface twice at a pixel centre
            # (the two triangles interpolate one edge, equal to rounding --
            # a bit-exact compare let the pair through, and an unmatched
            # entry fills the gap up to the next crossing solid)
            same_z = np.abs(zc[1:] - zc[:-1]) <= 1e-9 * (1.0 + np.abs(zc[1:]))
            dup = np.r_[False, (px[1:] == px[:-1]) & same_z & (sg[1:] == sg[:-1])]
            px, zc, sg = px[~dup], zc[~dup], sg[~dup]
            new_pixel = np.r_[True, px[1:] != px[:-1]]
            seg = np.cumsum(new_pixel) - 1
            total = np.cumsum(sg)
            base = np.r_[0, total[:-1]][np.flatnonzero(new_pixel)][seg]
            count = total - base  # entries minus exits below this crossing, inclusive
            same_next = np.r_[px[1:] == px[:-1], False]
            solid = same_next & (count != 0)
            span_px = np.concatenate([px[solid], px])
            span_lo = np.concatenate([zc[solid], zc])
            span_hi = np.concatenate([np.r_[zc[1:], zc[-1:]][solid], zc])
            filled[np.unique(px)] = True
            cls._assign_spans(lo, hi, span_px, span_lo, span_hi, size, spans)
        # -- the splat hull where the fill left a pixel empty
        only_splat = np.isfinite(splat_top) & ~filled & (splat_top > 0.0)
        idx = np.flatnonzero(only_splat)
        if idx.size:
            lo[0].reshape(-1)[idx] = np.maximum(splat_bot[idx], 0.0)
            hi[0].reshape(-1)[idx] = splat_top[idx]
        # a span that never rises above the ground blocks nothing
        buried = ~np.isnan(hi) & (hi <= 0.0)
        lo[buried] = np.nan
        hi[buried] = np.nan
        return lo, hi, (a0, a1, b0, b1)

    @staticmethod
    def _assign_spans(lo, hi, px, span_lo, span_hi, size, spans):
        """Merge touching spans per pixel, keep the *spans* thickest (the
        rest merged into the neighbour across the smallest gap) and write
        them, sorted by height, into ``lo`` / ``hi`` ``(K, size, size)``."""
        order = np.lexsort((span_lo, px))
        px, span_lo, span_hi = px[order], span_lo[order], span_hi[order]
        # A span merges into the one before it (same pixel) when it starts at
        # or below that one's top: the running maximum of the tops, restarted
        # per pixel by lifting each pixel's values onto their own plateau.
        new_pixel = np.r_[True, px[1:] != px[:-1]]
        seg = np.cumsum(new_pixel) - 1
        lift = float(np.nanmax(span_hi) - np.nanmin(span_lo)) + 1.0
        run_top = np.maximum.accumulate(span_hi + seg * lift) - seg * lift
        prev_top = np.r_[-np.inf, run_top[:-1]]
        start = new_pixel | (span_lo > prev_top + 1e-9)
        group = np.cumsum(start) - 1
        n_groups = int(group[-1]) + 1
        g_lo = np.full(n_groups, np.inf)
        g_hi = np.full(n_groups, -np.inf)
        np.minimum.at(g_lo, group, span_lo)
        np.maximum.at(g_hi, group, span_hi)
        g_px = px[start]
        # Per pixel the merged spans are sorted by height. Up to K of them
        # scatter straight in; a pixel with more merges its smallest gaps
        # first (rare: a shelf's boards, a chair's rungs) until K remain.
        g_new = np.r_[True, g_px[1:] != g_px[:-1]]
        g_seg = np.cumsum(g_new) - 1
        g_rank = np.arange(len(g_px)) - np.flatnonzero(g_new)[g_seg]
        g_count = np.diff(np.r_[np.flatnonzero(g_new), len(g_px)])[g_seg]
        fits = g_count <= spans
        iy, ix = np.divmod(g_px[fits], size)
        lo[g_rank[fits], iy, ix] = g_lo[fits]
        hi[g_rank[fits], iy, ix] = g_hi[fits]
        for s in np.flatnonzero(g_new & ~fits):
            c = int(g_count[s])
            pixel = int(g_px[s])
            py_, px_ = divmod(pixel, size)
            l_ = list(g_lo[s : s + c])
            h_ = list(g_hi[s : s + c])
            while len(l_) > spans:
                gaps = [l_[j + 1] - h_[j] for j in range(len(l_) - 1)]
                j = int(np.argmin(gaps))
                h_[j] = max(h_[j], h_[j + 1])
                del l_[j + 1], h_[j + 1]
            for k in range(len(l_)):
                lo[k, py_, px_] = l_[k]
                hi[k, py_, px_] = h_[k]

    @classmethod
    def rasterize_shadow(
        cls,
        meshes,
        light=None,
        ground=0.0,
        size=512,
        *,
        up=1,
        direction=None,
        source_size=0.0,
        max_stretch=None,
        canvas=None,
        contact=None,
        radius=None,
        height=None,
        padding=0.04,
        uniform_alpha=True,
        falloff_power=0.8,
        vertical_weight=0.3,
        blur_amount=1.0,
    ):
        """Body of :meth:`ImgUtils.rasterize_shadow`."""
        from pythontk.geo_utils.shadow_projection import (
            ShadowProjection,
            ShadowRaster,
        )

        meshes = [
            (
                np.asarray(p, dtype=float).reshape(-1, 3),
                np.asarray(t, dtype=np.int64).reshape(-1, 3),
            )
            for p, t in meshes
            if len(p) and len(t)
        ]
        if not meshes:
            raise ValueError("rasterize_shadow: no geometry provided.")
        size = int(size)
        a, b = ShadowProjection.horizontal_axes(up)
        if contact is None or radius is None or height is None:
            all_pts = np.concatenate([p for p, _ in meshes], axis=0)
            mn, mx = all_pts.min(axis=0), all_pts.max(axis=0)
            if contact is None:
                contact = np.zeros(3)
                contact[a], contact[b] = 0.5 * (mn[a] + mx[a]), 0.5 * (mn[b] + mx[b])
                contact[up] = mn[up]
            if radius is None:
                radius = 0.5 * math.hypot(mx[a] - mn[a], mx[b] - mn[b])
            if height is None:
                height = mx[up] - mn[up]
        contact = np.asarray(contact, dtype=float).reshape(3)
        radius = max(float(radius), 1e-3)
        height = max(float(height), 1e-3)
        model = ShadowProjection.model(
            contact,
            light,
            ground,
            radius,
            height,
            up=up,
            direction=direction,
            max_stretch=max_stretch,
        )
        stretch = (
            ShadowProjection.DEFAULT_MAX_STRETCH if max_stretch is None else max_stretch
        )
        slide = math.hypot(model.anchor[0] - contact[a], model.anchor[1] - contact[b])
        max_len = stretch * height + slide

        projected = []  # (uw, penumbra widths, tris)
        for pts, tris in meshes:
            res = ShadowProjection.project(
                pts, light, ground, up=up, direction=direction, max_length=max_len
            )
            if res is None:
                projected = []
                break
            uw = ShadowProjection.to_frame(res[0], model)
            # A point level with the source spreads without bound; the
            # penumbra can never usefully exceed the reach cap.
            widths = np.minimum(float(source_size) * res[1], max_len)
            projected.append((uw, widths, tris))

        if canvas is not None:
            rect = tuple(float(v) for v in canvas)
        elif projected:
            all_uw = np.concatenate([uw for uw, _, _ in projected], axis=0)
            pen = max(float(w.max()) for _, w, _ in projected)
            lo, hi = all_uw.min(axis=0), all_uw.max(axis=0)
            pad = float(padding) * max(hi[0] - lo[0], hi[1] - lo[1], 1e-3) + 0.5 * pen
            rect = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)
        else:  # nothing cast — the model's own rect, fully transparent
            rect = model.rect((-1.0, 1.0, -0.5, 0.5))
        u_lo, u_hi, w_lo, w_hi = rect
        du, dw = max(u_hi - u_lo, 1e-9), max(w_hi - w_lo, 1e-9)

        # Pre-flip orientation (the light-side edge at the BOTTOM row, as
        # rasterize_silhouette's contact row): rows run far -> near.
        mask = np.zeros((size, size), dtype=np.uint8)
        soft = np.zeros((size, size), dtype=np.float32)  # penumbra sigma, px
        px_per_unit = size / (0.5 * (du + dw))
        pen_max = 0.0
        for uw, widths, tris in projected:
            cols = (uw[:, 1] - w_lo) / dw * size
            rows = (1.0 - (uw[:, 0] - u_lo) / du) * size
            proj = np.stack([cols, rows], axis=1)
            sigma = widths * px_per_unit * 0.25  # a step blurs over ~4 sigma
            pen_max = max(pen_max, float(widths.max()))
            for tri in tris:
                cls._fill_triangle(mask, proj[tri])
                if sigma[tri].max() > 0.0:
                    cls._fill_triangle(soft, proj[tri], float(sigma[tri].mean()))

        base = mask
        if blur_amount and blur_amount > 0:
            base = cls.gaussian_blur(base, radius=blur_amount)
        s_max = float(soft.max())
        if s_max >= 0.5:
            base = cls._variable_blur(base, mask, soft, s_max)

        alpha = base.astype(np.float32) / 255.0
        if not uniform_alpha:
            # The footprint centre (the frame origin) in saved-image coords:
            # rows there run near -> far, so the origin's row is measured
            # from the near edge.
            origin = ((0.0 - w_lo) / dw, (0.0 - u_lo) / du)
            alpha *= cls._contact_falloff(mask, origin, falloff_power, vertical_weight)
        result = np.zeros((size, size, 4), dtype=np.uint8)
        result[:, :, 3] = np.flipud(np.clip(alpha * 255.0, 0, 255).astype(np.uint8))
        raster = ShadowRaster(
            model=model,
            rect=rect,
            fractions=ShadowProjection.fractions(rect, model),
            penumbra=pen_max,
        )
        return result, raster

    @classmethod
    def _variable_blur(cls, base, mask, sigma, s_max, levels=(0.125, 0.25, 0.5, 1.0)):
        """Blur *base* by the per-pixel *sigma* field (pixels), as a blend
        between a few uniformly blurred levels. *sigma* is defined inside
        *mask* only; it is carried past the edge by a normalized convolution
        so the blur reaches as far out as the penumbra does."""
        m = (mask > 0).astype(np.uint8) * 255
        q = np.clip(sigma / s_max * 255.0, 0, 255).astype(np.uint8)
        reach = max(s_max, 1.0)
        num = cls.gaussian_blur(q, radius=reach).astype(np.float32)
        den = cls.gaussian_blur(m, radius=reach).astype(np.float32)
        field = np.where(den > 1.0, num / np.maximum(den, 1.0) * s_max, 0.0)
        field = np.clip(field, 0.0, s_max)
        radii = [0.0] + [s_max * f for f in levels]
        stack = [base.astype(np.float32)] + [
            cls.gaussian_blur(base, radius=r).astype(np.float32) for r in radii[1:]
        ]
        out = stack[0].copy()
        for i in range(len(radii) - 1):
            lo, hi = radii[i], radii[i + 1]
            sel = (field >= lo) & (field <= hi)
            if not sel.any():
                continue
            t = (field[sel] - lo) / max(hi - lo, 1e-9)
            out[sel] = stack[i][sel] * (1.0 - t) + stack[i + 1][sel] * t
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)
