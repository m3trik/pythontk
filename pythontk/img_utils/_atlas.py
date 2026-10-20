# !/usr/bin/python
# coding=utf-8
"""Texture-atlas layout: tile placement, pixel rects, insets/snapping and
assembly (the bodies behind the :class:`ImgUtils` facade).
"""

from __future__ import annotations

import math
from typing import List, Tuple, Union, Optional, Sequence

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from pythontk.img_utils._filters import _ImgFilterInternal


class _ImgAtlasInternal:
    """Bodies of :class:`ImgUtils`' atlas-layout methods (an ``ImgUtils`` base).

    The public signatures and docstrings stay on the facade; each delegates
    here.
    """

    @staticmethod
    def compute_atlas_layout(
        weights: Sequence[float],
        *,
        rows: Optional[int] = None,
    ) -> List[Tuple[float, float, float, float]]:
        """Body of :meth:`ImgUtils.compute_atlas_layout`."""
        n = len(weights)
        if n == 0:
            return []
        if n == 1:
            return [(1.0, 1.0, 0.0, 0.0)]

        w = [max(float(x), 0.0) for x in weights]
        total = sum(w)
        if total <= 0.0:  # no information -> equal shares
            w = [1.0] * n
            total = float(n)

        rects: List[Tuple[float, float, float, float]] = [(1.0, 1.0, 0.0, 0.0)] * n

        if rows is None:
            # Squarified treemap (Bruls/Huizing/van Wijk): rows are grown along
            # the remaining rect's SHORTER side, admitting the next item only
            # while doing so does not make the row's worst aspect ratio worse.
            # When it would, the row is closed and the next starts in what is
            # left. Areas stay exactly proportional; what changes is the SHAPE
            # each area is delivered in.
            eps = 1e-12
            order = sorted(range(n), key=lambda k: (-w[k], k))
            areas = [w[i] / total for i in order]

            def worst(
                a_max: float, a_min: float, row_area: float, side: float
            ) -> float:
                """Worst aspect in a row of *row_area* laid along *side*.

                The row occupies a strip ``t = row_area / side`` thick, in which
                an item of area ``a`` gets an extent of ``a / t``. Areas arrive
                descending, so the row's largest and smallest items are the only
                two that can hold the worst ratio.
                """
                t2 = max((row_area / max(side, eps)) ** 2, eps)
                return max(a_max / t2, t2 / max(a_min, eps))

            x, y, bw, bh = 0.0, 0.0, 1.0, 1.0
            pos = 0
            while pos < n:
                horizontal = bw >= bh  # a wide rect takes a VERTICAL strip
                side = bh if horizontal else bw
                end, row_area = pos + 1, areas[pos]
                best = worst(areas[pos], areas[pos], row_area, side)
                while end < n:
                    trial_area = row_area + areas[end]
                    trial = worst(areas[pos], areas[end], trial_area, side)
                    if trial > best:
                        break
                    row_area, best, end = trial_area, trial, end + 1

                # The last row takes the whole remaining extent, so float drift
                # can never leave an unassigned seam down the atlas edge.
                span = bw if horizontal else bh
                thick = span if end >= n else min(row_area / max(side, eps), span)
                # Cells run cumulatively along `side`, and the last one is
                # derived from the row's far edge rather than from its own area
                # -- the same "one shared coordinate" discipline that makes the
                # tiling exact rather than merely close.
                cur, far = (y, y + bh) if horizontal else (x, x + bw)
                for j in range(pos, end):
                    nxt = far if j == end - 1 else cur + areas[j] / max(thick, eps)
                    nxt = min(max(nxt, cur), far)
                    if horizontal:
                        rects[order[j]] = (thick, nxt - cur, x, cur)
                    else:
                        rects[order[j]] = (nxt - cur, thick, cur, y)
                    cur = nxt
                if horizontal:
                    x, bw = x + thick, bw - thick
                else:
                    y, bh = y + thick, bh - thick
                pos = end
            return rects

        r = max(1, min(int(rows), n))

        # Balance items across `r` shelves by weight (LPT): assign the heaviest
        # remaining item to the currently-lightest shelf. Keeps shelf weight-sums
        # (and thus heights) even, which keeps rect aspect ratios reasonable.
        # With r <= n and zero-weight shelves being "lightest", the first r items
        # seed distinct shelves, so no shelf is ever left empty.
        shelves: List[List[int]] = [[] for _ in range(r)]
        shelf_w = [0.0] * r
        for i in sorted(range(n), key=lambda k: w[k], reverse=True):
            j = min(range(r), key=lambda k: shelf_w[k])
            shelves[j].append(i)
            shelf_w[j] += w[i]

        rects: List[Tuple[float, float, float, float]] = [(1.0, 1.0, 0.0, 0.0)] * n
        oy = 0.0
        for j, shelf in enumerate(shelves):
            sh = shelf_w[j] / total  # shelf height == its weight share
            ox = 0.0
            for i in sorted(shelf):  # input order within the row, for stable output
                sw = w[i] / shelf_w[j] if shelf_w[j] > 0 else 1.0 / len(shelf)
                rects[i] = (sw, sh, ox, oy)
                ox += sw
            oy += sh
        return rects

    @staticmethod
    def pack_atlas_rects(
        extents: Sequence[Tuple[float, float]],
        size: Union[int, Tuple[int, int]],
        gutter: int = 2,
        min_texels: int = 4,
        align: int = 4,
    ) -> Optional[List[Tuple[float, float, float, float]]]:
        """Body of :meth:`ImgUtils.pack_atlas_rects`."""
        from pythontk.img_utils.rect_packer import RectPacker

        w_px, h_px = (size, size) if isinstance(size, int) else size
        w_px, h_px = int(w_px), int(h_px)
        if not extents:
            return []
        result = RectPacker.fit(
            extents,
            (w_px, h_px),
            padding=max(0, int(gutter)),
            align=max(1, int(align)),
            min_size=max(1, int(min_texels)),
            fold=True,
        )
        if result is None:
            return None
        return [
            (w / w_px, h / h_px, x / w_px, 1.0 - (y + h) / h_px)
            for (x, y), (w, h) in zip(result.positions, result.sizes)
        ]

    @staticmethod
    def atlas_extent(
        area: float,
        uv_area: Optional[float] = None,
        extent: Tuple[float, float] = (1.0, 1.0),
        fill: float = 0.75,
        stretch: float = 1.0,
    ) -> Tuple[float, float]:
        """Body of :meth:`ImgUtils.atlas_extent`."""
        eu, ev = max(float(extent[0]), 0.0), max(float(extent[1]), 0.0)
        covered = float(uv_area) if uv_area and uv_area > 0.0 else 0.0
        if covered <= 0.0:
            covered = max(float(fill), 1e-6) * max(eu * ev, 1e-12)
        k = math.sqrt(max(float(area), 0.0) / covered)
        # The stretch moves texels between the axes, never in or out: the
        # box's area -- the density -- stays what the surface area set.
        stretch = float(stretch) if stretch else 1.0
        r = math.sqrt(stretch) if math.isfinite(stretch) and stretch > 0.0 else 1.0
        return (eu * k * r, ev * k / r)

    @staticmethod
    def atlas_layout_extent(
        area: float,
        uvs=None,
        stretch: float = 1.0,
        max_coverage: float = 0.85,
        fill: float = 0.75,
    ) -> Tuple[float, float]:
        """Body of :meth:`ImgUtils.atlas_layout_extent`."""
        tris = None if uvs is None else np.asarray(uvs, dtype=np.float64)
        if tris is None or tris.ndim != 3 or not len(tris):
            return _ImgAtlasInternal.atlas_extent(area, None, (1.0, 1.0), fill)
        flat = tris.reshape(-1, 2)
        (u0, v0), (u1, v1) = flat.min(axis=0), flat.max(axis=0)
        box = _ImgAtlasInternal.uv_crop_extent(
            (float(u0), float(v0), float(u1), float(v1)), max_coverage
        )
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        uv_area = 0.5 * float(
            np.abs(
                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
            ).sum()
        )
        return _ImgAtlasInternal.atlas_extent(
            area, uv_area if uv_area > 0.0 else None, box, fill, stretch
        )

    @staticmethod
    def uv_axis_scale(points, uvs) -> Optional[Tuple[float, float]]:
        """Body of :meth:`ImgUtils.uv_axis_scale`."""
        p = np.asarray(points, dtype=np.float64)
        t = np.asarray(uvs, dtype=np.float64)
        if p.ndim != 3 or t.ndim != 3 or not len(p) or len(p) != len(t):
            return None
        e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        d1, d2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
        # A triangle collapsed in uv owns no texels and has no derivative.
        keep = np.abs(det) > 1e-12
        if not keep.any():
            return None
        e1, e2, d1, d2, det = e1[keep], e2[keep], d1[keep], d2[keep], det[keep]
        # The derivative of the surface point along each uv axis (solving
        # (u, v) - t0 = a*d1 + b*d2 for a, b): its length is the surface one
        # unit of that axis spans there.
        du = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / det[:, None]
        dv = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / det[:, None]
        du, dv = np.linalg.norm(du, axis=1), np.linalg.norm(dv, axis=1)
        live = (du > 0.0) & (dv > 0.0)  # a triangle with no surface spans nothing
        if not live.any():
            return None
        # A GEOMETRIC mean weighted by uv area (the texels each triangle owns):
        # the scale that makes texels square on the surface in the least-squares
        # sense of their log aspect, and one a sliver cannot swing -- its
        # derivative grows as 1/area while its weight shrinks with it, where a
        # root mean square (area x derivative^2 ~ 1/area) ran away.
        w = np.abs(det[live])
        su = math.exp(float((w * np.log(du[live])).sum() / w.sum()))
        sv = math.exp(float((w * np.log(dv[live])).sum() / w.sum()))
        return (su, sv)

    @staticmethod
    def atlas_pixel_rects(
        rects: Sequence[Tuple[float, float, float, float]],
        size: Union[int, Tuple[int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        """Body of :meth:`ImgUtils.atlas_pixel_rects`."""
        w_px, h_px = (size, size) if isinstance(size, int) else size
        out: List[Tuple[int, int, int, int]] = []
        for sx, sy, ox, oy in rects:
            col0 = int(round(ox * w_px))
            col1 = int(round((ox + sx) * w_px))
            row0 = int(round((1.0 - (oy + sy)) * h_px))
            row1 = int(round((1.0 - oy) * h_px))
            out.append((row0, row1, col0, col1))
        return out

    @staticmethod
    def flip_rect_v(rect: Sequence[float]) -> List[float]:
        """Body of :meth:`ImgUtils.flip_rect_v`."""
        sx, sy, ox, oy = (float(v) for v in rect)
        return [sx, sy, ox, 1.0 - sy - oy]

    @staticmethod
    def compose_rect(
        outer: Optional[Sequence[float]], inner: Sequence[float]
    ) -> List[float]:
        """Body of :meth:`ImgUtils.compose_rect`."""
        osx, osy, oox, ooy = (float(v) for v in (outer or (1.0, 1.0, 0.0, 0.0)))
        isx, isy, iox, ioy = (float(v) for v in inner)
        return [isx * osx, isy * osy, iox * osx + oox, ioy * osy + ooy]

    @staticmethod
    def inset_atlas_rects(
        rects: Sequence[Tuple[float, float, float, float]],
        size: Union[int, Tuple[int, int]],
        gutter: int,
    ) -> List[Tuple[float, float, float, float]]:
        """Body of :meth:`ImgUtils.inset_atlas_rects`."""
        w_px, h_px = (size, size) if isinstance(size, int) else size
        out: List[Tuple[float, float, float, float]] = []
        for sx, sy, ox, oy in rects:
            gx = min(float(gutter), max(0.0, (sx * w_px - 2.0) / 4.0))
            gy = min(float(gutter), max(0.0, (sy * h_px - 2.0) / 4.0))
            out.append(
                (
                    sx - 2.0 * gx / w_px,
                    sy - 2.0 * gy / h_px,
                    ox + gx / w_px,
                    oy + gy / h_px,
                )
            )
        return out

    @classmethod
    def snap_atlas_rects(
        cls,
        rects: Sequence[Tuple[float, float, float, float]],
        size: Union[int, Tuple[int, int]],
    ) -> List[Tuple[float, float, float, float]]:
        """Body of :meth:`ImgUtils.snap_atlas_rects`."""
        w_px, h_px = (size, size) if isinstance(size, int) else size
        out: List[Tuple[float, float, float, float]] = []
        pixel_rects = cls.atlas_pixel_rects(rects, (w_px, h_px))
        for rect, (row0, row1, col0, col1) in zip(rects, pixel_rects):
            if row1 - row0 <= 0 or col1 - col0 <= 0:
                out.append(tuple(float(v) for v in rect))
                continue
            sx = (col1 - col0) / w_px
            sy = (row1 - row0) / h_px
            ox = col0 / w_px
            oy = 1.0 - row1 / h_px  # back to bottom-left origin
            out.append((sx, sy, ox, oy))
        return out

    @staticmethod
    def inset_rects_to_texel_centers(
        rects: Sequence[Tuple[float, float, float, float]],
        size: Union[int, Tuple[int, int]],
        bboxes: Optional[Sequence[Optional[Tuple[float, float, float, float]]]] = None,
    ) -> List[Tuple[float, float, float, float]]:
        """Body of :meth:`ImgUtils.inset_rects_to_texel_centers`."""
        w_px, h_px = (size, size) if isinstance(size, int) else size
        eps = 1e-6  # tolerate float noise on exact texel boundaries
        out: List[Tuple[float, float, float, float]] = []
        for i, (sx, sy, ox, oy) in enumerate(rects):
            bbox = bboxes[i] if bboxes is not None else None
            u0, v0, u1, v1 = bbox if bbox is not None else (0.0, 0.0, 1.0, 1.0)
            if u1 - u0 <= 0 or v1 - v0 <= 0:
                out.append((float(sx), float(sy), float(ox), float(oy)))
                continue
            x0 = math.floor((ox + sx * u0) * w_px + eps) + 0.5
            x1 = math.ceil((ox + sx * u1) * w_px - eps) - 0.5
            y0 = math.floor((oy + sy * v0) * h_px + eps) + 0.5
            y1 = math.ceil((oy + sy * v1) * h_px - eps) - 0.5
            if x1 - x0 < 1.0 or y1 - y0 < 1.0:
                out.append((float(sx), float(sy), float(ox), float(oy)))
                continue
            nsx = (x1 - x0) / ((u1 - u0) * w_px)
            nsy = (y1 - y0) / ((v1 - v0) * h_px)
            out.append((nsx, nsy, x0 / w_px - u0 * nsx, y0 / h_px - v0 * nsy))
        return out

    @staticmethod
    def uv_crop_extent(
        bbox: Optional[Tuple[float, float, float, float]],
        max_coverage: float = 0.85,
    ) -> Tuple[float, float]:
        """Body of :meth:`ImgUtils.uv_crop_extent`."""
        if bbox is None:
            return (1.0, 1.0)
        u0, v0, u1, v1 = (min(max(float(v), 0.0), 1.0) for v in bbox)
        eu, ev = u1 - u0, v1 - v0
        if eu <= 0.0 or ev <= 0.0 or (eu >= max_coverage and ev >= max_coverage):
            return (1.0, 1.0)
        return (eu, ev)

    @classmethod
    def crop_to_uv_bbox(
        cls,
        img: "np.ndarray",
        bbox: Optional[Tuple[float, float, float, float]],
        cell: Sequence[float],
        max_coverage: float = 0.85,
    ) -> Tuple["np.ndarray", List[float], Tuple[float, float, float, float]]:
        """Body of :meth:`ImgUtils.crop_to_uv_bbox`."""
        full = (0.0, 0.0, 1.0, 1.0)
        cell = [float(v) for v in cell]
        if cls.uv_crop_extent(bbox, max_coverage) == (1.0, 1.0):
            return img, cell, full
        u0, v0, u1, v1 = (min(max(float(v), 0.0), 1.0) for v in bbox)
        h, w = img.shape[:2]
        eps = 1e-6  # an edge ON a texel boundary must not claim the next one
        c0 = max(0, math.floor(u0 * w + eps))
        c1 = min(w, math.ceil(u1 * w - eps))
        r0 = max(0, math.floor((1.0 - v1) * h + eps))
        r1 = min(h, math.ceil((1.0 - v0) * h - eps))
        if c1 - c0 < 2 or r1 - r0 < 2:
            return img, cell, full
        cu0, cu1 = c0 / w, c1 / w
        cv0, cv1 = 1.0 - r1 / h, 1.0 - r0 / h
        sx = cell[0] / (cu1 - cu0)
        sy = cell[1] / (cv1 - cv0)
        return (
            img[r0:r1, c0:c1],
            [sx, sy, cell[2] - cu0 * sx, cell[3] - cv0 * sy],
            (cu0, cv0, cu1, cv1),
        )

    @staticmethod
    def _footprints(n_in: int, n_out: int, edge_centers: bool):
        """Each output texel's box footprint ``(lo, hi)`` in source-texel units."""
        i = np.arange(n_out, dtype=np.float64)
        if edge_centers:
            # Centres span the source edge to edge; the border boxes are clipped
            # to the half that lies inside it.
            step = n_in / (n_out - 1)
            return (
                np.clip(i * step - step / 2, 0.0, n_in),
                np.clip(i * step + step / 2, 0.0, n_in),
            )
        step = n_in / n_out
        return i * step, (i + 1) * step

    @staticmethod
    def _box_integrate(a: "np.ndarray", lo, hi) -> "np.ndarray":
        """Integral of 2-D *a* (one constant value per texel) over ``[lo, hi)`` down axis 0.

        Prefix sums in float64, a column block at a time: O(texels) at any
        ratio, and an HDR peak (65504) cannot swamp the sums of texels far
        from it, as float32 prefix sums let it. The result is float32 -- the
        sums are bounded, only their running totals need the precision.
        """
        n, m = a.shape
        out = np.empty((len(lo), m), dtype=np.float32)
        k_lo = np.minimum(np.floor(lo).astype(np.int64), n - 1)
        k_hi = np.minimum(np.floor(hi).astype(np.int64), n - 1)
        f_lo = (lo - k_lo)[:, None]
        f_hi = (hi - k_hi)[:, None]
        block = max(1, 4_000_000 // max(n, 1))
        for c0 in range(0, m, block):
            c1 = min(m, c0 + block)
            cs = np.zeros((n + 1, c1 - c0), dtype=np.float64)
            np.cumsum(a[:, c0:c1], axis=0, dtype=np.float64, out=cs[1:])
            at_lo = cs[k_lo] + f_lo * (cs[k_lo + 1] - cs[k_lo])
            at_hi = cs[k_hi] + f_hi * (cs[k_hi + 1] - cs[k_hi])
            out[:, c0:c1] = at_hi - at_lo
        return out

    @classmethod
    def resize_into_cell(
        cls,
        image: "np.ndarray",
        size: Tuple[int, int],
        coverage: Optional["np.ndarray"] = None,
        edge_centers: bool = True,
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """Body of :meth:`ImgUtils.resize_into_cell`."""
        width, height = int(size[0]), int(size[1])
        src = np.asarray(image)
        h, w = src.shape[:2]
        # inset_rects_to_texel_centers passes a rect under two texels on EITHER
        # axis through unchanged; the content must match it.
        centred = bool(edge_centers) and width >= 2 and height >= 2
        ylo, yhi = cls._footprints(h, height, centred)
        xlo, xhi = cls._footprints(w, width, centred)
        area = (yhi - ylo)[:, None] * (xhi - xlo)[None, :]

        def integrate(plane):
            # One channel at a time: no full-tile intermediate is ever held.
            rows = cls._box_integrate(plane, ylo, yhi)
            return cls._box_integrate(rows.T, xlo, xhi).T

        weight = None
        if coverage is not None:
            weight = np.clip(np.asarray(coverage, dtype=np.float32), 0.0, 1.0)
        planes = [src] if src.ndim == 2 else [src[..., c] for c in range(src.shape[2])]
        sums = [
            integrate(p if weight is None else p.astype(np.float32) * weight)
            for p in planes
        ]
        den = area if weight is None else integrate(weight).astype(np.float64)
        covered = den > 1e-9 * np.maximum(area, 1e-12)
        safe = np.where(covered, den, 1.0)
        out = np.stack([np.where(covered, s / safe, 0.0) for s in sums], axis=-1)
        if src.ndim == 2:
            out = out[..., 0]
        fraction = np.where(covered, den / np.maximum(area, 1e-12), 0.0)
        dtype = src.dtype if np.issubdtype(src.dtype, np.floating) else np.float32
        return out.astype(dtype), np.clip(fraction, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _bilinear_taps(xs, ys, h: int, w: int):
        """The four texels and weights a bilinear read at ``(xs, ys)`` takes.

        Pixel-centre convention (texel ``i`` is at ``x == i``); taps past the
        frame are clamped onto it, as a clamped sampler reads them.
        """
        x0 = np.floor(xs).astype(np.int64)
        y0 = np.floor(ys).astype(np.int64)
        fx, fy = xs - x0, ys - y0
        cols = np.stack([x0, x0 + 1, x0, x0 + 1], 1).clip(0, w - 1)
        rows = np.stack([y0, y0, y0 + 1, y0 + 1], 1).clip(0, h - 1)
        weights = np.stack(
            [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1
        )
        return rows * w + cols, weights

    @classmethod
    def stitch_seams(
        cls,
        image: "np.ndarray",
        pairs: "np.ndarray",
        iterations: int = 64,
        hold: Optional["np.ndarray"] = None,
    ) -> "np.ndarray":
        """Body of :meth:`ImgUtils.stitch_seams`."""
        out = np.array(image, copy=True)
        pairs = np.asarray(pairs, dtype=np.float64).reshape(-1, 4)
        h, w = out.shape[:2]
        held = None
        if hold is not None:
            held = np.asarray(hold, dtype=bool)
            if held.shape != (h, w):
                raise ValueError(f"hold shape {held.shape} != image {(h, w)}")
            held = held.reshape(-1)
        if not len(pairs):
            return out
        flat = out.reshape(h * w, -1)
        taps = [
            cls._bilinear_taps(pairs[:, 0], pairs[:, 1], h, w),
            cls._bilinear_taps(pairs[:, 2], pairs[:, 3], h, w),
        ]
        # The taps a correction may move: every one a read weighs, less the
        # held texels -- read, never written.
        free = [
            wts > 0 if held is None else (wts > 0) & ~held[idx] for idx, wts in taps
        ]
        movable = [np.where(f, wts, 0.0) for f, (_i, wts) in zip(free, taps)]
        # Least-norm share of a sample's correction per tap (w / sum w^2) over
        # the taps it may move, and how many samples move each texel (their
        # updates are averaged, so a texel several samples share is not
        # corrected several times over).
        shares = [m / np.maximum((m**2).sum(1, keepdims=True), 1e-12) for m in movable]
        # Each side moves half the difference -- the whole of it when the
        # other side cannot move, none when neither can.
        if held is None:
            signs = (-0.5, 0.5)
            active = slice(None)
        else:
            can = [m.sum(1) > 0 for m in movable]
            both = can[0] & can[1]
            signs = (
                np.where(both, -0.5, np.where(can[0], -1.0, 0.0))[:, None, None],
                np.where(both, 0.5, np.where(can[1], 1.0, 0.0))[:, None, None],
            )
            active = can[0] | can[1]
            if not active.any():
                return out
        n = h * w
        read = np.zeros(n, np.float64)
        touched = np.zeros(n, np.float64)
        for (idx, wts), f in zip(taps, free):
            read += np.bincount(idx[wts > 0], minlength=n)
            touched += np.bincount(idx[f], minlength=n)
        live = np.nonzero(read)[0]
        # The solve runs on the texels the reads tap only, gathered once:
        # bincount over those indices instead of add.at over the whole atlas.
        local = np.zeros(n, np.int64)
        local[live] = np.arange(len(live))
        # A zero-weight tap (a read ON a texel row) points at a live texel it
        # leaves untouched, never off the end of the solve's array.
        taps = [(np.where(wts > 0, local[idx], 0), wts) for idx, wts in taps]
        values = flat[live].astype(np.float64)
        weight = np.maximum(touched[live], 1.0)[:, None]
        settled = 1e-4 * max(float(np.abs(values).mean()), 1e-12)
        for _ in range(max(1, int(iterations))):
            reads = [(values[idx] * wts[..., None]).sum(1) for idx, wts in taps]
            diff = reads[0] - reads[1]  # (N, C): meet halfway
            if float(np.abs(diff[active]).max()) <= settled:
                break
            delta = np.zeros_like(values)
            for (idx, _wts), f, share, sign in zip(taps, free, shares, signs):
                upd = sign * diff[:, None, :] * share[..., None]
                upd[~f] = 0.0
                for ch in range(values.shape[1]):
                    delta[:, ch] += np.bincount(
                        idx.ravel(), weights=upd[..., ch].ravel(), minlength=len(live)
                    )
            values += delta / weight
        if held is not None:
            moved = ~held[live]
            live, values = live[moved], values[moved]
        flat[live] = _ImgFilterInternal._restore_dtype(values, out.dtype)
        return out

    @staticmethod
    def atlas_seam_pairs(
        edges: Sequence[Optional["np.ndarray"]],
        rects: Sequence[Sequence[float]],
        size: Union[int, Tuple[int, int]],
        tolerance: float = 1e-4,
        coplanar: float = 0.99,
    ) -> "np.ndarray":
        """Body of :meth:`ImgUtils.atlas_seam_pairs`."""
        if len(edges) != len(rects):
            raise ValueError(
                f"edges ({len(edges)}) and rects ({len(rects)}) length differ"
            )
        none = np.zeros((0, 4))
        width, height = (size, size) if np.ndim(size) == 0 else size
        width, height = int(width), int(height)
        rows, owner, rect_rows = [], [], []
        for i, (member, rect) in enumerate(zip(edges, rects)):
            if member is None:
                continue
            member = np.asarray(member, dtype=np.float64).reshape(-1, 16)
            if not len(member):
                continue
            rows.append(member)
            owner.append(np.full(len(member), i))
            rect_rows.append(np.tile(np.asarray(rect, np.float64), (len(member), 1)))
        if len(rows) < 2:
            return none
        e = np.concatenate(rows)
        own = np.concatenate(owner)
        rect = np.concatenate(rect_rows)
        p0, p1, uv0, uv1 = e[:, 0:3], e[:, 3:6], e[:, 6:8], e[:, 8:10]

        def unit_rows(v):
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

        normal, inward = unit_rows(e[:, 10:13]), unit_rows(e[:, 13:16])
        d = p1 - p0
        length = np.linalg.norm(d, axis=1)
        unit = unit_rows(d)
        lo_box = np.minimum(p0, p1)
        hi_box = np.maximum(p0, p1)
        tol = max(
            min(
                float(tolerance) * float(np.linalg.norm(hi_box.max(0) - lo_box.min(0))),
                1e-3 * float(np.median(length)),
            ),
            1e-9,
        )
        # Each edge's length in atlas texels, through its own rect.
        texels = np.hypot(
            (uv1[:, 0] - uv0[:, 0]) * rect[:, 0] * width,
            (uv1[:, 1] - uv0[:, 1]) * rect[:, 1] * height,
        )

        def to_px(uv, r):
            x = (uv[:, 0] * r[0] + r[2]) * width - 0.5
            y = (1.0 - (uv[:, 1] * r[1] + r[3])) * height - 0.5
            return x, y

        pairs = []
        order = np.argsort((lo_box[:, 0] + hi_box[:, 0]) * 0.5, kind="stable")
        for a0 in range(0, len(order), 256):
            a = order[a0 : a0 + 256]
            near = np.nonzero(
                (hi_box >= lo_box[a].min(0) - tol).all(1)
                & (lo_box <= hi_box[a].max(0) + tol).all(1)
            )[0]
            rel0 = p0[near][None, :, :] - p0[a][:, None, :]
            rel1 = p1[near][None, :, :] - p0[a][:, None, :]
            ua = unit[a][:, None, :]
            t0 = (rel0 * ua).sum(-1)
            t1 = (rel1 * ua).sum(-1)
            off0 = np.linalg.norm(rel0 - t0[..., None] * ua, axis=-1)
            off1 = np.linalg.norm(rel1 - t1[..., None] * ua, axis=-1)
            lo = np.maximum(np.minimum(t0, t1), 0.0)
            hi = np.minimum(np.maximum(t0, t1), length[a][:, None])
            hit = (
                (own[a][:, None] < own[near][None, :])
                & (off0 < tol)
                & (off1 < tol)
                & (hi - lo > tol)
                & ((normal[a] @ normal[near].T) > float(coplanar))
                & ((inward[a] @ inward[near].T) < 0.0)
            )
            for i, jn in zip(*np.nonzero(hit)):
                ia, j = a[i], near[jn]
                overlap = hi[i, jn] - lo[i, jn]
                # Twice per atlas texel along the overlap, on the finer side.
                n = max(
                    2,
                    int(
                        np.ceil(
                            2.0
                            * max(
                                texels[ia] * overlap / max(length[ia], 1e-12),
                                texels[j] * overlap / max(length[j], 1e-12),
                            )
                        )
                    ),
                )
                along = lo[i, jn] + (np.arange(n) + 0.5) / n * overlap
                ta = (along / max(length[ia], 1e-12))[:, None]
                point = p0[ia] + ta * d[ia]
                tb = (((point - p0[j]) @ d[j]) / max(float(d[j] @ d[j]), 1e-24))[
                    :, None
                ]
                xa, ya = to_px(uv0[ia] + ta * (uv1[ia] - uv0[ia]), rect[ia])
                xb, yb = to_px(uv0[j] + tb * (uv1[j] - uv0[j]), rect[j])
                pairs.append(np.stack([xa, ya, xb, yb], 1))
        return np.concatenate(pairs) if pairs else none

    @classmethod
    def assemble_atlas(
        cls,
        images: Sequence["np.ndarray"],
        rects: Sequence[Tuple[float, float, float, float]],
        size: Union[int, Tuple[int, int]],
        *,
        background: float = 0.0,
        canvas: Optional["np.ndarray"] = None,
    ) -> "np.ndarray":
        """Body of :meth:`ImgUtils.assemble_atlas`."""
        if len(images) != len(rects):
            raise ValueError(
                f"images ({len(images)}) and rects ({len(rects)}) length differ"
            )
        if not images and canvas is None:
            raise ValueError("assemble_atlas requires at least one image")

        w_px, h_px = (size, size) if isinstance(size, int) else size
        if canvas is not None:
            # In place: the atlas starts as a copy of *canvas*, so every texel
            # no rect covers keeps its value exactly (a float32 copy of float32
            # data, restored to its own dtype).
            base = np.asarray(canvas)
            if base.shape[:2] != (h_px, w_px):
                raise ValueError(
                    f"canvas shape {base.shape[:2]} != atlas {(h_px, w_px)}"
                )
            dtype = base.dtype
            squeeze = base.ndim == 2
            channels = 1 if squeeze else base.shape[2]
            canvas = base.astype(np.float32, copy=True)
            if squeeze:
                canvas = canvas[..., None]
            if not images:
                return _ImgFilterInternal._restore_dtype(
                    canvas[..., 0] if squeeze else canvas, dtype
                )
        else:
            first = np.asarray(images[0])
            dtype = first.dtype
            squeeze = first.ndim == 2
            channels = 1 if squeeze else first.shape[2]
            canvas = np.full((h_px, w_px, channels), background, dtype=np.float32)

        import cv2

        # UV v is bottom-up; image rows are top-down -> atlas_pixel_rects owns
        # the flip + rounding so mask-building consumers can't drift from it.
        pixel_rects = cls.atlas_pixel_rects(rects, (w_px, h_px))
        for img, (row0, row1, col0, col1) in zip(images, pixel_rects):
            tw, th = col1 - col0, row1 - row0
            if tw <= 0 or th <= 0:
                continue  # degenerate (e.g. zero-weight) rect -- nothing to place
            # A rect that rounds past the canvas clips the DESTINATION slice
            # while the resized source keeps its full size -- a shape-mismatch
            # ValueError that would lose a whole atlas. Clamp the destination
            # first and crop the source to match it, so the two agree by
            # construction (deriving the crop from the overhang instead lets a
            # rect that lies ENTIRELY off-canvas produce a negative slice stop,
            # which silently wraps and mismatches again).
            dst_r0, dst_r1 = max(row0, 0), min(row1, h_px)
            dst_c0, dst_c1 = max(col0, 0), min(col1, w_px)
            if dst_r1 <= dst_r0 or dst_c1 <= dst_c0:
                continue  # entirely outside the canvas

            a = np.asarray(img, dtype=np.float32)
            if a.ndim == 2:
                a = a[..., None]
            if a.shape[2] != channels:
                raise ValueError(
                    f"image channel count {a.shape[2]} != atlas {channels}"
                )
            # INTER_AREA is the correct downscale kernel (the common atlas case);
            # use bilinear only when a rect happens to be larger than its source.
            interp = (
                cv2.INTER_AREA
                if th <= a.shape[0] and tw <= a.shape[1]
                else cv2.INTER_LINEAR
            )
            resized = cv2.resize(a, (tw, th), interpolation=interp)
            if resized.ndim == 2:
                resized = resized[..., None]
            if (dst_r1 - dst_r0, dst_c1 - dst_c0) != (th, tw):
                resized = resized[
                    dst_r0 - row0 : dst_r1 - row0, dst_c0 - col0 : dst_c1 - col0, :
                ]
            canvas[dst_r0:dst_r1, dst_c0:dst_c1, :] = resized

        if squeeze:
            canvas = canvas[..., 0]
        return _ImgFilterInternal._restore_dtype(canvas, dtype)
