# !/usr/bin/python
# coding=utf-8
"""Texture transfer between two UV layouts of the SAME triangles (arrays in -> arrays out).

A UV re-layout -- repacking an atlas, consolidating several materials' maps
into one, moving a mesh's textures from UV set A to UV set B -- is a texel
remap, not a bake: every target triangle corresponds to a known source
triangle, so each target texel maps (barycentrically) to exactly one source
UV. Ray-cast bakers get this wrong by construction wherever a mesh touches
itself (a wire lying on a nut picks up the wire), and a renderer is pure
overhead for it. This engine does the remap directly in numpy:

1. :meth:`UvTransfer.build` rasterizes the *target* triangles once into a
   :class:`TransferTable` -- per texel (and per sub-sample), which triangle
   covers it and the source UV it maps to. Building is the expensive part and
   is independent of the maps, so one table serves every channel.
2. :meth:`UvTransfer.transfer` applies the table to a source image (or one
   image per source material, for a consolidation) and returns the remapped
   image plus its coverage mask. :meth:`UvTransfer.transfer_normals` is the
   SAME resample for a tangent-space normal map, plus the one step a
   direction stored in an island's own frame needs: where the target layout
   rotates or mirrors an island, the stored XY turn with it. The turn is read
   off the two UV layouts alone -- never positions or normals -- so this is
   no normal-map bake (re-deriving normals from geometry, a ray-cast baker's
   job and a separate operation): a target that moved or deforms transfers
   exactly like one that did not.
3. :meth:`UvTransfer.pad` fills the gutter from the coverage mask so
   filtering and mips never pull background across an island edge.

Conventions (pinned by ``test_uv_transfer``): UVs are V-up in [0, 1]; images
are stored V-flipped ((0, 0) = top-left) with texel centers at ``+0.5``,
matching :meth:`pythontk.ImgUtils.rasterize_uv_triangles`. An identity
transfer (same layout both sides, ``supersample=1``) reproduces the source
texel-for-texel to within the table's UV quantization (``uint16`` over
[0, 1]: 1/16 texel at 4k, i.e. a bilinear weight of at most 6% on a
neighbour -- invisible on textures, measurable on white noise).

Correspondence is **by triangle index**: ``src_tris[i]`` and ``dst_tris[i]``
must be the same 3D triangle's two parameterizations, corner for corner. The
host adapter owns that guarantee (mayatk/blendertk build both arrays from one
triangulation pass, face-vertex UVs on both sides, so seams are respected and
concave polygons are triangulated properly). Tangent-frame re-encoding is
exact for the per-island rigid transforms a packer applies (rotate / uniform
scale / mirror); when the target was re-cut along different seams, the
vertex-averaged tangents a baker would compute differ near the new seams by
a small amount this per-triangle model cannot see.

Memory: a table holds, per sub-sample pass, one ``int32`` triangle id and two
``uint16`` UV coordinates per texel -- 8 bytes x ``supersample**2`` passes x
``size**2``. A 4k table at the default ``supersample=2`` is ~540 MB; use
``supersample=1`` (135 MB) for 8k or memory-constrained hosts. Supersampling
is what makes islands that were packed SMALLER resample correctly (box
filter) and gives anti-aliased island edges; ``1`` is point sampling.
"""

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.str_utils._str_utils import StrUtils

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None


# Bucket edges for the batched rasterizer: triangles whose pixel bounding box
# fits in k x k are rasterized together as a (B, k, k) grid, B chunked so a
# grid never exceeds _BATCH_CELLS elements (~10 float32 temporaries of that
# size are live). Bigger than the largest bucket -> one at a time: a large
# triangle's own bbox grid is already a big enough numpy op to amortize the
# call, while batching it pads a rectangle out to k x k and rounds k up
# (measured 6x SLOWER with buckets to 1024 on a 3k-triangle 4k layout).
_BATCH_BUCKETS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
_BATCH_CELLS = 4_000_000


@dataclass
class TransferTable:
    """The per-texel correspondence from a target layout back to its source.

    ``tri[p]`` is the triangle id covering each texel at sub-sample pass ``p``
    (``-1`` = uncovered); ``uv[p]`` the source UV it maps to, fixed-point
    ``uint16`` over [0, 1]. ``source_ids[i]`` is the caller's source-material
    id for triangle ``i`` (all zero when the transfer has one source).
    """

    size: Tuple[int, int]  # (height, width)
    supersample: int
    tri: List["np.ndarray"]
    uv: List["np.ndarray"]
    src_tris: "np.ndarray"
    dst_tris: "np.ndarray"
    source_ids: "np.ndarray"
    overlaps: int = 0
    skipped: int = 0
    _frames: Optional["np.ndarray"] = field(default=None, repr=False)

    @property
    def passes(self) -> int:
        return len(self.tri)

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.tri) + sum(a.nbytes for a in self.uv)

    @property
    def coverage(self) -> "np.ndarray":
        """Fraction of sub-samples covered per texel, ``float32`` in [0, 1]."""
        acc = np.zeros(self.size, dtype=np.float32)
        for t in self.tri:
            acc += t >= 0
        acc /= float(self.passes)
        return acc

    @property
    def mask(self) -> "np.ndarray":
        """Bool: texels touched by any sub-sample (what the output owns)."""
        out = np.zeros(self.size, dtype=bool)
        for t in self.tri:
            out |= t >= 0
        return out

    @property
    def frames(self) -> "np.ndarray":
        """``(N, 2, 2)`` tangent-frame rotations, source -> target, per triangle."""
        if self._frames is None:
            self._frames = UvTransfer.triangle_frames(self.src_tris, self.dst_tris)
        return self._frames


class UvTransfer(HelpMixin):
    """Remap textures between two UV layouts of the same triangles (see module doc)."""

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        src_tris,
        dst_tris,
        size: Union[int, Tuple[int, int]],
        *,
        supersample: int = 2,
        source_ids=None,
    ) -> TransferTable:
        """Rasterize *dst_tris* and record, per texel, the source UV it maps to.

        Parameters:
            src_tris: ``(N, 3, 2)`` source-layout triangles (V up, in [0, 1]).
            dst_tris: ``(N, 3, 2)`` target-layout triangles, same order and
                corner order as *src_tris*.
            size: Output resolution -- ``int`` (square) or ``(height, width)``.
            supersample: Sub-samples per texel axis (``s**2`` passes). 1 =
                point sampling; 2 (default) anti-aliases edges and box-filters
                islands packed smaller than their source.
            source_ids: Optional ``(N,)`` ints -- which source image triangle
                ``i`` reads from (see :meth:`transfer`'s ``sources`` mapping).

        Returns:
            :class:`TransferTable`. ``overlaps`` counts texels a second
            triangle claimed in the same pass (overlapping target islands --
            last writer wins); ``skipped`` counts degenerate triangles.
        """
        cls._require_numpy()
        src = np.asarray(src_tris, dtype=np.float64).reshape(-1, 3, 2)
        dst = np.asarray(dst_tris, dtype=np.float64).reshape(-1, 3, 2)
        if src.shape != dst.shape:
            raise ValueError(
                f"src_tris {src.shape} and dst_tris {dst.shape} must match "
                "triangle for triangle"
            )
        h, w = cls._size_hw(size)
        ss = max(1, int(supersample))
        n = len(src)
        ids = (
            np.zeros(n, dtype=np.int32)
            if source_ids is None
            else np.asarray(source_ids, dtype=np.int32).reshape(-1)
        )
        if len(ids) != n:
            raise ValueError(f"source_ids has {len(ids)} entries for {n} triangles")

        # Target triangles in pixel space (x right, y down; V flipped).
        px = np.empty_like(dst)
        px[..., 0] = dst[..., 0] * w
        px[..., 1] = (1.0 - dst[..., 1]) * h

        tri_maps: List[np.ndarray] = []
        uv_maps: List[np.ndarray] = []
        overlaps = 0
        skipped = 0
        for i in range(ss):
            for j in range(ss):
                off = ((j + 0.5) / ss, (i + 0.5) / ss)  # (x, y) sub-sample offset
                tri_map, uv_map, ov, sk = cls._rasterize_pass(px, src, h, w, off)
                tri_maps.append(tri_map)
                uv_maps.append(uv_map)
                overlaps += ov
                skipped = max(skipped, sk)
        return TransferTable(
            size=(h, w),
            supersample=ss,
            tri=tri_maps,
            uv=uv_maps,
            src_tris=src,
            dst_tris=dst,
            source_ids=ids,
            overlaps=int(overlaps),
            skipped=int(skipped),
        )

    @classmethod
    def _rasterize_pass(cls, px, src, h, w, off):
        """One sub-sample pass: ``(tri int32 HxW, uv uint16 HxWx2, overlaps, skipped)``.

        Sample positions are ``(col + off_x, row + off_y)``; a texel is
        claimed when that point lies inside the triangle (edge-inclusive).
        Triangles are batched by bounding-box size so a dense mesh does not
        cost one numpy round-trip per triangle; the few large ones run alone.
        """
        tri_map = np.full((h, w), -1, dtype=np.int32)
        uv_map = np.zeros((h, w, 2), dtype=np.uint16)
        n = len(px)
        if n == 0:
            return tri_map, uv_map, 0, 0

        xs, ys = px[..., 0], px[..., 1]
        ox, oy = off
        # Integer sample columns/rows whose sample point can fall inside.
        c0 = np.ceil(xs.min(axis=1) - ox).astype(np.int64)
        c1 = np.floor(xs.max(axis=1) - ox).astype(np.int64)
        r0 = np.ceil(ys.min(axis=1) - oy).astype(np.int64)
        r1 = np.floor(ys.max(axis=1) - oy).astype(np.int64)
        c0 = np.clip(c0, 0, w - 1)
        c1 = np.clip(c1, 0, w - 1)
        r0 = np.clip(r0, 0, h - 1)
        r1 = np.clip(r1, 0, h - 1)

        ax, ay = xs[:, 0], ys[:, 0]
        bx, by = xs[:, 1], ys[:, 1]
        cx, cy = xs[:, 2], ys[:, 2]
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        valid = np.abs(denom) > 1e-12
        skipped = int((~valid).sum())
        # Triangles whose sample range is empty contribute nothing (sliver
        # between two sample rows/cols, or cropped off the image).
        live = valid & (c1 >= c0) & (r1 >= r0)
        if not live.any():
            return tri_map, uv_map, 0, skipped

        bw = c1 - c0 + 1
        bh = r1 - r0 + 1
        extent = np.maximum(bw, bh)
        overlaps = 0

        def _scatter(ids, rows, cols, l1, l2, l3):
            nonlocal overlaps
            su = l1 * src[ids, 0, 0] + l2 * src[ids, 1, 0] + l3 * src[ids, 2, 0]
            sv = l1 * src[ids, 0, 1] + l2 * src[ids, 1, 1] + l3 * src[ids, 2, 1]
            # Overlap = one texel claimed by two triangles whose SOURCE UVs
            # disagree. Two triangles sharing an edge both claim the samples
            # on it (edge-inclusive test) but map them to the same source UV
            # -- continuity across the edge -- so that is not an overlap;
            # two islands stacked in the target are.
            tol = 1e-3
            prior = tri_map[rows, cols]
            had = prior >= 0
            if had.any():
                pu = uv_map[rows[had], cols[had], 0].astype(np.float64) / 65535.0
                pv = uv_map[rows[had], cols[had], 1].astype(np.float64) / 65535.0
                overlaps += int(
                    ((np.abs(pu - su[had]) > tol) | (np.abs(pv - sv[had]) > tol)).sum()
                )
            # Within the batch `prior` cannot see a sibling's write.
            lin = rows * w + cols
            order = np.argsort(lin, kind="stable")
            same = lin[order][1:] == lin[order][:-1]
            if same.any():
                a, b = order[1:][same], order[:-1][same]
                overlaps += int(
                    (
                        (np.abs(su[a] - su[b]) > tol) | (np.abs(sv[a] - sv[b]) > tol)
                    ).sum()
                )
            tri_map[rows, cols] = ids
            uv_map[rows, cols, 0] = np.clip(np.rint(su * 65535.0), 0, 65535)
            uv_map[rows, cols, 1] = np.clip(np.rint(sv * 65535.0), 0, 65535)

        done = np.zeros(n, dtype=bool)
        batches = []
        for k in _BATCH_BUCKETS:
            sel = np.nonzero(live & ~done & (extent <= k))[0]
            done[sel] = True
            if not len(sel):
                continue
            step = max(1, _BATCH_CELLS // (k * k))
            batches.extend((k, sel[i : i + step]) for i in range(0, len(sel), step))
        for k, sel in batches:
            # (B, k, k) sample grid anchored at each triangle's bbox origin.
            gy, gx = np.mgrid[0:k, 0:k]
            cols = c0[sel, None, None] + gx[None]
            rows = r0[sel, None, None] + gy[None]
            in_box = (cols <= c1[sel, None, None]) & (rows <= r1[sel, None, None])
            # Barycentrics relative to corner C, in float32: at 4k a float32
            # sample position is exact to ~2e-4 texel, and the (B, k, k)
            # grids below are the hot loop -- halving their width is the
            # difference between seconds and tens of seconds on a dense mesh.
            f32 = np.float32
            sx = (cols + ox).astype(f32) - cx[sel, None, None].astype(f32)
            sy = (rows + oy).astype(f32) - cy[sel, None, None].astype(f32)
            inv = (1.0 / denom[sel]).astype(f32)[:, None, None]
            l1 = (
                (by - cy)[sel, None, None].astype(f32) * sx
                + (cx - bx)[sel, None, None].astype(f32) * sy
            ) * inv
            l2 = (
                (cy - ay)[sel, None, None].astype(f32) * sx
                + (ax - cx)[sel, None, None].astype(f32) * sy
            ) * inv
            l3 = f32(1.0) - l1 - l2
            inside = in_box & (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
            if not inside.any():
                continue
            ids = np.broadcast_to(sel[:, None, None], inside.shape)[inside]
            _scatter(
                ids, rows[inside], cols[inside], l1[inside], l2[inside], l3[inside]
            )

        # Anything larger than the biggest bucket: one triangle at a time.
        for t in np.nonzero(live & ~done)[0]:
            rows, cols = np.mgrid[r0[t] : r1[t] + 1, c0[t] : c1[t] + 1]
            sx = cols + ox
            sy = rows + oy
            l1 = (
                (by[t] - cy[t]) * (sx - cx[t]) + (cx[t] - bx[t]) * (sy - cy[t])
            ) / denom[t]
            l2 = (
                (cy[t] - ay[t]) * (sx - cx[t]) + (ax[t] - cx[t]) * (sy - cy[t])
            ) / denom[t]
            l3 = 1.0 - l1 - l2
            inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
            if not inside.any():
                continue
            ids = np.full(int(inside.sum()), t, dtype=np.int64)
            _scatter(
                ids, rows[inside], cols[inside], l1[inside], l2[inside], l3[inside]
            )
        return tri_map, uv_map, overlaps, skipped

    # --------------------------------------------------------------- transfer
    @classmethod
    def transfer(
        cls,
        table: TransferTable,
        sources,
        *,
        source_masks=None,
        bilinear: bool = True,
        value_max: Optional[float] = None,
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """Remap *sources* through *table*.

        Parameters:
            table: From :meth:`build`.
            sources: One image (``HxW`` or ``HxWxC`` array), or a mapping
                ``{source_id: image | constant}`` keyed by the ``source_ids``
                the table was built with. A *constant* (scalar or per-channel
                sequence) stands in for a source material that has no map for
                this channel -- the texels its triangles own are filled with
                it.
            source_masks: Optional image or ``{source_id: mask}`` coverage
                masks for the source images. Where given, the source's empty
                texels are filled from their nearest covered neighbour BEFORE
                sampling, so an island edge never bilinearly pulls in the
                source's gutter/background.
            bilinear: Bilinear (default) vs nearest sampling.
            value_max: The sources' full scale (255 for 8-bit data), used as
                the OPAQUE value for an alpha channel a source lacks when
                sources of different channel counts share the layout. None
                guesses it from each source's data, which reads an all-dark
                8-bit map as 0..1 -- pass it whenever channel counts can mix.

        Returns:
            ``(image float32 HxWxC, coverage float32 HxW)`` -- *image* holds the
            mean of every covered sub-sample (partially covered edge texels are
            NOT darkened), and is 0 where *coverage* is 0. Pad with
            :meth:`pad` before writing to disk.
        """
        cls._require_numpy()
        h, w = table.size
        src_by_id, channels = cls._normalize_sources(sources, table, value_max)
        if source_masks is not None:
            src_by_id = cls._prefill_sources(src_by_id, source_masks)

        acc = np.zeros((h, w, channels), dtype=np.float32)
        cnt = np.zeros((h, w), dtype=np.float32)
        for tri_map, uv_map in zip(table.tri, table.uv):
            samples, covered = cls._dense_pass(
                table, tri_map, uv_map, src_by_id, channels, bilinear
            )
            if samples is None:
                continue
            acc += samples
            cnt += covered
        out = np.zeros_like(acc)
        nz = cnt > 0
        out[nz] = acc[nz] / cnt[nz][:, None]
        coverage = cnt / float(table.passes)
        return out, coverage

    @classmethod
    def transfer_normals(
        cls,
        table: TransferTable,
        sources,
        *,
        convention: str = "opengl",
        source_masks=None,
        bilinear: bool = True,
        value_range: Tuple[float, float] = (0.0, 255.0),
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """Remap tangent-space normal maps, re-expressing XY in the target frame.

        A tangent-space normal is stored relative to the island's UV-derived
        tangent frame. When the target layout rotates / mirrors an island,
        that frame turns with it, so the stored XY must turn too -- copying
        the texels verbatim (what :meth:`transfer` would do) leaves every
        rotated island shading wrong. Per triangle the 2x2 source->target UV
        map is polar-decomposed to its rotation (+ reflection for mirrored
        islands), exactly the change an orthonormalized tangent basis sees;
        XY are rotated by it, Z is kept, and the vector is renormalized. Only
        the two UV layouts enter it -- no positions, no normals -- so it stays
        a resample: where the target geometry stands, or how it is shaped,
        changes nothing.

        Parameters:
            table: From :meth:`build`.
            sources: As :meth:`transfer` -- normal-map image(s) in
                *value_range* encoding, or a constant for a flat normal.
            convention: ``"opengl"`` (Y+ = V up, Maya/Blender/Painter default)
                or ``"directx"`` (Y+ = V down). Rotation mixes X and Y, so the
                Y sign has to be known -- a DirectX map rotated as OpenGL is
                not merely flipped, it is wrong.
            source_masks, bilinear: As :meth:`transfer`.
            value_range: ``(lo, hi)`` of the stored encoding mapped onto
                [-1, 1]; ``(0, 255)`` for 8-bit, ``(0, 65535)`` for 16-bit,
                ``(0, 1)`` for float images.

        Returns:
            ``(image float32 HxWx3 in value_range, coverage)``.
        """
        cls._require_numpy()
        conv = str(convention).lower()
        if conv not in ("opengl", "directx"):
            raise ValueError("convention must be 'opengl' or 'directx'")
        lo, hi = float(value_range[0]), float(value_range[1])
        span = hi - lo
        h, w = table.size
        src_by_id, channels = cls._normalize_sources(sources, table, hi)
        if channels < 3:
            raise ValueError("normal maps need 3 channels")
        if source_masks is not None:
            src_by_id = cls._prefill_sources(src_by_id, source_masks)
        frames = table.frames  # (N, 2, 2)
        ysign = -1.0 if conv == "directx" else 1.0

        # Per-texel frame, gathered by component: (H, W) each, no (H, W, 2, 2).
        fa, fb = frames[:, 0, 0].astype(np.float32), frames[:, 0, 1].astype(np.float32)
        fc, fd = frames[:, 1, 0].astype(np.float32), frames[:, 1, 1].astype(np.float32)
        acc = np.zeros((h, w, 3), dtype=np.float32)
        cnt = np.zeros((h, w), dtype=np.float32)
        for tri_map, uv_map in zip(table.tri, table.uv):
            samples, covered = cls._dense_pass(
                table, tri_map, uv_map, src_by_id, channels, bilinear
            )
            if samples is None:
                continue
            vec = (samples[..., :3] - lo) * (2.0 / span) - 1.0
            safe = np.maximum(tri_map, 0)
            x = vec[..., 0].copy()  # a view here would see its own overwrite
            y = vec[..., 1] * ysign
            vec[..., 0] = fa[safe] * x + fb[safe] * y
            vec[..., 1] = (fc[safe] * x + fd[safe] * y) * ysign
            vec *= covered[..., None]
            acc += vec
            cnt += covered
        out = np.zeros_like(acc)
        nz = cnt > 0
        mean = acc[nz] / cnt[nz][:, None]
        norm = np.linalg.norm(mean, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        mean = mean / norm
        out[nz] = (mean + 1.0) * 0.5 * span + lo
        coverage = cnt / float(table.passes)
        return out, coverage

    @classmethod
    def pad(
        cls,
        image,
        coverage,
        width: int = -1,
    ) -> "np.ndarray":
        """Fill the gutter around covered texels (edge padding / dilation).

        Parameters:
            image: Output of :meth:`transfer` / :meth:`transfer_normals`.
            coverage: Its coverage array (any value > 0 counts as owned).
            width: Gutter width in texels; ``-1`` (default) fills every empty
                texel from its nearest owned one, which is what a mip chain
                needs so no level ever averages background into an island.

        Returns:
            Padded copy, same dtype as *image*.
        """
        cls._require_numpy()
        from pythontk.img_utils._img_utils import ImgUtils

        mask = np.asarray(coverage) > 0
        if width is None or int(width) < 0:
            return ImgUtils.fill_empty_texels(image, mask)
        if int(width) == 0:
            return np.array(image, copy=True)
        return ImgUtils.dilate_image(image, mask, iterations=int(width))

    # ------------------------------------------------------------- materials
    #: How much smaller a source's share of the target layout must be before
    #: :meth:`_auto_size` says so. 1.05 is ~5% of linear resolution -- below
    #: that the repack is effectively density-preserving and a line about it
    #: would be noise on every ordinary transfer.
    _SQUEEZE_REPORT_THRESHOLD: float = 1.05

    #: Logical PBR channel -> token used in output filenames.
    CHANNEL_TOKENS: Dict[str, str] = {
        "baseColor": "BaseColor",
        "emission": "Emissive",
        "specular": "Specular",
        "roughness": "Roughness",
        "metallic": "Metallic",
        "opacity": "Opacity",
        "normal": "Normal",
        "ambientOcclusion": "AO",
    }

    #: Neutral fill (0..1) for a source that has neither a map nor a constant.
    NEUTRAL: Dict[str, Tuple[float, ...]] = {
        "baseColor": (0.5, 0.5, 0.5),
        "emission": (0.0, 0.0, 0.0),
        "specular": (0.5, 0.5, 0.5),
        "roughness": (0.5,),
        "metallic": (0.0,),
        "opacity": (1.0,),
        "normal": (0.5, 0.5, 1.0),
        "ambientOcclusion": (1.0,),
    }

    @classmethod
    def merge_layouts(
        cls,
        jobs: Dict[str, Dict[str, Any]],
        name: str,
        *,
        probe_size: int = 256,
    ) -> Dict[str, Dict[str, Any]]:
        """Merge per-material *jobs* that share one UV layout into one job.

        The unit of a transfer is a LAYOUT (an atlas), not a material: a mesh
        with three materials on one UV set is one atlas and wants one output,
        while two meshes whose materials each fill their own 0-1 space under
        the same set name are two atlases. The tell is overlap -- islands of
        one atlas never cover another's -- so the jobs are rasterized at
        *probe_size* and merged only when no pair overlaps; otherwise they are
        returned as given (one output per material), and the caller can say so.

        Parameters:
            jobs: ``{label: job}`` as :meth:`transfer_materials` takes, all for
                the same target UV set. Any extra keys a job carries (e.g. the
                host's ``members``) are concatenated when lists, else kept
                from the first job.
            name: Label for the merged job (conventionally the UV set name).

        Returns:
            ``{name: merged}`` when the layouts are disjoint, else *jobs*.
        """
        cls._require_numpy()
        from pythontk.img_utils._img_utils import ImgUtils

        if len(jobs) <= 1:
            return dict(jobs)
        covers = [
            ImgUtils.rasterize_uv_triangles(j["dst"], size=probe_size, supersample=1)
            > 0
            for j in jobs.values()
        ]
        seen = np.zeros_like(covers[0])
        for c in covers:
            if (seen & c).any():
                return dict(jobs)
            seen |= c
        merged: Dict[str, Any] = {}
        first = next(iter(jobs.values()))
        for key in first:
            vals = [j.get(key) for j in jobs.values()]
            if key in ("src", "dst", "ids"):
                merged[key] = np.concatenate([np.asarray(v) for v in vals])
            elif key == "sources":
                merged[key] = first[key]  # the shared source registry, not per job
            elif all(isinstance(v, list) for v in vals):
                merged[key] = [x for v in vals for x in v]
            else:
                merged[key] = first[key]
        return {name: merged}

    @classmethod
    def layout_jobs(
        cls,
        parts: Sequence[Dict[str, Any]],
        sources: Sequence[Dict[str, Any]],
        *,
        log: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """A run's outputs, ``{label: job}``, from what each target contributed.

        The unit of a transfer is a LAYOUT, and no name a host reads off a
        target says which layout a triangle is in. One UV set name can hold
        two atlases (two meshes each filling 0-1), and two names can be one: a
        production table whose parts came in through Maya (``map1``) and an
        FBX (``UVChannel_1``) was ONE combined layout, which grouping by set
        name transferred into two materials. So parts are grouped by (target
        material, target UV set) -- the finest unit that is still one image --
        and every group competes in ONE :meth:`merge_layouts`: one output when
        no two overlap, one output per group otherwise.

        Parameters:
            parts: One per (target, target material) the host read:
                ``{"material": its name, or None for faces that wear nothing,
                "uv_set": the target UV set, "src", "dst", "ids"`` (as
                :meth:`transfer_materials` takes them) ``, "members": [...]}``.
                ``members`` is the host's own, carried through untouched.
            sources: The run's source-material registry
                (:meth:`transfer_materials`'s ``sources``), shared by every job.
            log: Optional ``callable(str)``, told how the groups came out.

        Returns:
            ``{label: job}`` -- one job named after the UV set(s) when every
            group merged; otherwise one per group, named after its material
            (``<set>_<material>`` for a material two sets share, the set alone
            for faces that wear nothing). Empty when nothing was contributed.
        """
        cls._require_numpy()
        groups: Dict[Tuple[Optional[str], str], Dict[str, list]] = {}
        for part in parts:
            group = groups.setdefault(
                (part.get("material"), part["uv_set"]),
                {"src": [], "dst": [], "ids": [], "members": []},
            )
            for key in ("src", "dst", "ids"):
                group[key].append(np.asarray(part[key]))
            group["members"].extend(part.get("members") or [])
        if not groups:
            return {}
        shared = Counter(material for material, _uv_set in groups)
        jobs: Dict[str, Dict[str, Any]] = {}
        for (material, uv_set), group in groups.items():
            if material is None:
                label = uv_set
            else:
                label = material if shared[material] == 1 else f"{uv_set}_{material}"
            while label in jobs:  # a material named like a set it shares
                label = f"{uv_set}_{label}"
            jobs[label] = {
                "src": np.concatenate(group["src"]),
                "dst": np.concatenate(group["dst"]),
                "ids": np.concatenate(group["ids"]).astype(np.int32),
                "sources": sources,
                "members": group["members"],
            }
        name = "_".join(dict.fromkeys(uv_set for _material, uv_set in groups))
        merged = cls.merge_layouts(jobs, name)
        if log and len(jobs) > 1:
            log(
                f"{len(jobs)} target material / UV set group(s) -> "
                + (
                    f"one layout ({name})"
                    if len(merged) == 1
                    else f"{len(merged)} overlapping layouts, kept apart"
                )
            )
        return merged

    #: Two generic projections of a normalized position into the unit square,
    #: with independent null directions, so no single pair of distinct surface
    #: points can collapse onto one key under both. Non-negative weights summing
    #: below 1 per column keep every key inside ``[0, 1]``.
    _OVERLAP_KEYS = (
        ((0.61, 0.13), (0.27, 0.52), (0.11, 0.34)),
        ((0.19, 0.48), (0.07, 0.29), (0.71, 0.21)),
    )

    @classmethod
    def layout_overlaps(cls, uv_tris, points, size: int = 256) -> Tuple[int, int]:
        """``(overlapping texels, covered texels)`` of a layout -- whether it bakes.

        A lightmap gives every surface point its own texels, so a layout is
        bakeable only when no texel holds two different points: stacked or
        mirrored islands that suit a texture each read the other's light. The
        layout is rasterized with each corner's normalized surface POSITION as
        its key (:meth:`build`, which counts a texel two triangles claim with
        different keys): triangles sharing an edge agree along it and never
        count, distinct points do. Two projections, any overlap either sees.

        Parameters:
            uv_tris: ``(N, 3, 2)`` the layout's triangles (V up).
            points: ``(N, 3, 3)`` the same triangles' surface positions, same
                corner order.
            size: Probe resolution -- an overlap smaller than a texel here is
                below what a lightmap of that size can show.

        Returns:
            ``(overlaps, covered)`` in sample texels; judge one against the other.
        """
        cls._require_numpy()
        uv = np.asarray(uv_tris, dtype=np.float64).reshape(-1, 3, 2)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3, 3)
        if len(pts) != len(uv):
            raise ValueError(f"{len(uv)} UV triangles but {len(pts)} point triangles")
        if not len(uv):
            return 0, 0
        flat = pts.reshape(-1, 3)
        span = float(np.ptp(flat, axis=0).max()) or 1.0
        unit = (pts - flat.min(axis=0)) / span
        worst, covered = 0, 0
        for weights in cls._OVERLAP_KEYS:
            table = cls.build(unit @ np.asarray(weights), uv, size, supersample=1)
            worst = max(worst, table.overlaps)
            covered = int(sum((t >= 0).sum() for t in table.tri))
        return worst, covered

    @staticmethod
    def output_labels(
        labels: Sequence[str],
        base: str,
        *,
        prefix: str = "",
        suffix: str = "",
    ) -> Dict[str, str]:
        """``{label: label to name the output by}`` for a run named *base*.

        A named run with several layouts calls each output ``<base>_<label>``,
        and a layout's label is its TARGET material's name -- which, on a
        re-run, is the material the previous run assigned. Taken as it stood,
        every run stacked another copy of the name (measured on a production
        table: ``SolderingTable_SolderingTable_SolderingTable_TABLE_ASSETS_MAT``
        after four runs, the lightmap baker naming its maps after it). Every
        leading ``<base>_`` comes off, and the *prefix* / *suffix* the result
        material takes, so the first run and any re-run name the same files and
        the same material.

        A label that would strip to nothing, or onto a label another layout
        keeps or strips to, keeps its own spelling: two layouts must never
        share one stem, or their maps overwrite each other.

        Parameters:
            labels: The run's layout labels.
            base: The run's output name (empty = no renaming; labels as given).
            prefix / suffix: The affix applied to the assigned material.

        Returns:
            ``{label: output label}``, every label present.
        """

        def own(label: str) -> str:
            core = label
            while True:
                stripped = StrUtils.strip_known_affix(
                    core, prefix=f"{base}_", case_sensitive=True
                )
                stripped = StrUtils.strip_known_affix(
                    stripped, prefix=prefix, suffix=suffix
                ).strip("_")
                if stripped == core or not stripped:
                    return core
                core = stripped

        if not base:
            return {label: label for label in labels}
        proposed = {label: own(label) for label in labels}
        counts = Counter(proposed.values())
        kept = set(labels)
        return {
            label: (
                out if out == label or (counts[out] == 1 and out not in kept) else label
            )
            for label, out in proposed.items()
        }

    @classmethod
    def transfer_materials(
        cls,
        jobs: Dict[str, Dict[str, Any]],
        *,
        output_dir: str,
        channels: Optional[Sequence[str]] = None,
        size: Optional[int] = None,
        supersample: int = 2,
        padding: int = -1,
        name_format: str = "{material}_{channel}",
        normal_convention: Optional[str] = None,
        source_mask_from_uvs: bool = True,
        avoid: Optional[Sequence[str]] = None,
        log=None,
    ) -> Dict[str, Dict[str, str]]:
        """Transfer every channel of every target material and write the maps.

        The DCC-agnostic half of a texture transfer: the host adapter builds
        *jobs* (triangle correspondence + which source material each triangle
        wears + those materials' maps / constants) and this does the rest --
        sizing, table build, per-channel remap (normals re-encoded), padding,
        naming, saving.

        Parameters:
            jobs: ``{target material name: job}`` with job keys
                ``src`` ``(N,3,2)``, ``dst`` ``(N,3,2)``, ``ids`` ``(N,)``
                source-material index per triangle, and ``sources``: a list
                indexed by those ids of ``{"maps": {channel: path},
                "constants": {channel: tuple 0..1}}``.
            output_dir: Folder the PNGs are written into (created).
            channels: Channels to transfer; default = every channel any
                contributing source has a MAP for (constants alone do not
                create an output).
            size: Output resolution; default = the largest source map feeding
                the material (2048 if none). When a consolidation gives some
                source a smaller share of the layout than it had, that default
                carries less of its detail than the number suggests --
                :meth:`_auto_size` reports the squeeze per source so the choice
                to raise this is an informed one.
            supersample / padding: See :meth:`build` / :meth:`pad`.
            name_format: Stem with ``{material}`` / ``{channel}``.
            normal_convention: ``"opengl"`` / ``"directx"`` for every
                source, or ``None`` (default): each source map's own -- its
                content, then its filename (:meth:`normal_convention`) -- with
                a source that disagrees converted to the convention covering
                most of the layout (:meth:`_unify_normal_conventions`).
            source_mask_from_uvs: Rasterize each source layout into a coverage
                mask and pre-fill that source's gutter before sampling.
            avoid: Paths never to write: files something still reads once
                the run is done (the maps of a source material the host
                keeps). An output whose name lands on one takes the first
                free ``<stem>_<k>.png`` instead (:meth:`FileUtils.unique_path`).
                Two outputs of one run never share a file either way; any
                other name is the run's to replace, so a re-run rewrites its
                own maps rather than writing a new set beside them.
            log: Optional ``callable(str)`` for progress lines.

        Returns:
            ``{target material: {channel: written path}}``.
        """
        from pythontk.file_utils._file_utils import FileUtils

        cls._require_numpy()
        say = log or (lambda m: None)
        os.makedirs(output_dir, exist_ok=True)
        taken: set = set()
        avoid = tuple(avoid or ())
        results: Dict[str, Dict[str, str]] = {}
        for t_mat, job in jobs.items():
            src_tris = np.asarray(job["src"], dtype=float).reshape(-1, 3, 2)
            dst_tris = np.asarray(job["dst"], dtype=float).reshape(-1, 3, 2)
            ids = np.asarray(job["ids"], dtype=np.int32).reshape(-1)
            sources = job["sources"]
            used = sorted(set(int(i) for i in np.unique(ids)))

            def say_layout(m: str, _t: str = t_mat) -> None:
                say(f"{_t}: {m}")

            wanted = (
                list(channels)
                if channels
                else [
                    c
                    for c in cls.CHANNEL_TOKENS
                    if any(c in sources[i].get("maps", {}) for i in used)
                ]
            )
            if not wanted:
                say_layout("no source material carries a texture map; skipped.")
                results[t_mat] = {}
                continue
            res = size or cls._auto_size(
                used,
                sources,
                wanted,
                src_tris,
                dst_tris,
                ids,
                say=say_layout,
            )
            say_layout(
                f"{len(dst_tris)} triangles from {len(used)} source "
                f"material(s) -> {res}px"
            )
            table = cls.build(
                src_tris, dst_tris, res, supersample=supersample, source_ids=ids
            )
            if table.overlaps:
                say_layout(
                    f"WARNING {table.overlaps} texel(s) claimed by "
                    "overlapping target islands (last writer wins)."
                )
            mask_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
            written: Dict[str, str] = {}
            for channel in wanted:
                # Maps are loaded per channel and dropped after it: a float32
                # 4k RGB map is 200 MB, so caching every channel of every
                # source across materials would cost gigabytes for the price
                # of a PNG decode.
                loaded: Dict[int, Tuple[np.ndarray, float]] = {}
                missing: List[str] = []
                for sid in used:
                    path = sources[sid].get("maps", {}).get(channel)
                    if path and os.path.isfile(path):
                        loaded[sid] = cls.load_map(path)
                    elif path:
                        missing.append(os.path.basename(path))
                if missing:
                    # A material that names a map the disk does not hold (a
                    # renamed file) is said, never dropped quietly.
                    say_layout(
                        f"WARNING {channel}: {', '.join(missing)} not on disk -- "
                        + (
                            "those sources take their material's constant."
                            if loaded
                            else "the channel is skipped."
                        )
                    )
                if not loaded:
                    continue
                if channel == "normal":
                    conv = cls._unify_normal_conventions(
                        loaded,
                        {sid: sources[sid]["maps"]["normal"] for sid in loaded},
                        ids,
                        dst_tris,
                        override=normal_convention,
                        say=say_layout,
                    )
                value_max = max(v for _, v in loaded.values())
                sources_for: Dict[int, Any] = {}
                masks_for: Dict[int, Any] = {}
                for sid in used:
                    if sid in loaded:
                        arr, vmax = loaded[sid]
                        sources_for[sid] = (
                            arr if vmax == value_max else arr / vmax * value_max
                        )
                        if source_mask_from_uvs:
                            key = (sid, arr.shape[0], arr.shape[1])
                            if key not in mask_cache:
                                from pythontk.img_utils._img_utils import ImgUtils

                                mask_cache[key] = (
                                    ImgUtils.rasterize_uv_triangles(
                                        src_tris[ids == sid],
                                        size=arr.shape[0],
                                        supersample=1,
                                    )
                                    > 0
                                )
                            masks_for[sid] = mask_cache[key]
                        continue
                    const = sources[sid].get("constants", {}).get(channel)
                    if const is None:
                        const = cls.NEUTRAL[channel]
                    sources_for[sid] = tuple(float(c) * value_max for c in const)
                if channel == "normal":
                    img, cov = cls.transfer_normals(
                        table,
                        sources_for,
                        convention=conv,
                        source_masks=masks_for or None,
                        value_range=(0.0, value_max),
                    )
                    cls._say_invalid_normals(
                        img,
                        cov,
                        value_max,
                        table,
                        sources,
                        say=say_layout,
                    )
                else:
                    img, cov = cls.transfer(
                        table,
                        sources_for,
                        source_masks=masks_for or None,
                        value_max=value_max,
                    )
                img = cls.pad(img, cov, padding)
                stem = name_format.format(
                    material=cls._safe_name(t_mat),
                    channel=cls.CHANNEL_TOKENS.get(channel, channel),
                )
                path = FileUtils.unique_path(
                    output_dir, stem, ".png", taken, avoid=avoid
                ).replace("\\", "/")
                cls.save_map(path, img, value_max)
                written[channel] = path
                say(f"  {channel}: {path}")
            results[t_mat] = written
        return results

    # -------------------------------------------------------- correspondence
    #: World-space drift under which a part counts as standing where its slice
    #: of the target stands -- what tells two pieces of identical topology
    #: apart (two copies of one module combined into one mesh).
    POSITION_TOLERANCE: float = 1e-4

    @classmethod
    def concatenation_order(
        cls, target, parts, tolerance: Optional[float] = None
    ) -> Optional[List[int]]:
        """Which *parts*, in which order, join end to end into *target*.

        A mesh combined from several others (Maya's Combine, Blender's Join)
        is their concatenation: each part's faces follow the previous part's,
        and its vertex indices are offset by the vertices before it. Reading
        that order back is what lets a texture transfer treat the parts as
        ONE source of the target's topology -- the same face-for-face
        correspondence as between two copies of one mesh.

        Parts with identical topology (copies of one module) are told apart
        by where they stand: a part whose points coincide with the target's
        slice is preferred. A topology match that stands elsewhere is still
        accepted when nothing coincides -- the topology is the contract; the
        host warns about the positions, as it does for one mesh.

        Parameters:
            target: ``(counts, verts, points)`` -- per-face corner counts,
                each corner's vertex index (flat, face order), and the
                ``(V, 3)`` vertex positions in world space.
            parts: The candidates, each in the same form.
            tolerance: Largest point drift that counts as coincident;
                default :attr:`POSITION_TOLERANCE`.

        Returns:
            Indices into *parts* in concatenation order (a subset when the
            target is built from only some of them), or ``None`` when no
            ordering of the parts reproduces the target's topology exactly.
        """
        cls._require_numpy()
        tol = cls.POSITION_TOLERANCE if tolerance is None else float(tolerance)

        def unpack(mesh):
            counts, verts, points = mesh
            return (
                np.asarray(counts, dtype=np.int64).ravel(),
                np.asarray(verts, dtype=np.int64).ravel(),
                np.asarray(points, dtype=np.float64).reshape(-1, 3),
            )

        t_counts, t_verts, t_points = unpack(target)
        if not len(t_counts):
            return None
        cand = [unpack(p) for p in parts]
        # Each part's first point: one vectorized distance per step finds the
        # parts standing at the current offset, so a kit of many identical
        # pieces is not a full topology scan per placement.
        firsts = np.array(
            [c[2][0] if len(c[2]) else (np.inf,) * 3 for c in cand], dtype=np.float64
        ).reshape(-1, 3)
        used = set()

        def fits(k, face, corner, vert):
            """None = no topology match here; else whether it coincides."""
            counts, verts, points = cand[k]
            n_f, n_c, n_v = len(counts), len(verts), len(points)
            if not n_f or face + n_f > len(t_counts) or vert + n_v > len(t_points):
                return None
            # Scalar pre-check: most candidates fail on their first face.
            if t_counts[face] != counts[0] or t_verts[corner] != verts[0] + vert:
                return None
            if not np.array_equal(t_counts[face : face + n_f], counts):
                return None
            if not np.array_equal(t_verts[corner : corner + n_c], verts + vert):
                return None
            drift = np.abs(t_points[vert : vert + n_v] - points)
            return not drift.size or float(drift.max()) <= tol

        def options(face, corner, vert):
            """Parts that fit at this offset: coincident ones first, then
            (lazily, only if those dead-end) any topology match.

            Lazy, so ``used`` is read when the frame resumes -- by then it
            holds exactly the placements above this frame.
            """
            given = set()
            if vert < len(t_points):
                drift = np.abs(firsts - t_points[vert]).max(axis=1)
                for k in np.flatnonzero(drift <= tol).tolist():
                    if k not in used and fits(k, face, corner, vert):
                        given.add(k)
                        yield k
            for k in range(len(cand)):
                if k in used or k in given:
                    continue
                if fits(k, face, corner, vert) is not None:
                    yield k

        # Depth-first over placements, iteratively: a mesh combined from
        # hundreds of pieces must not hit the recursion limit. Backtracking
        # only matters when one part's topology is a prefix of another's.
        order: List[int] = []
        offsets = [(0, 0, 0)]
        stack = [options(0, 0, 0)]
        while stack:
            face, corner, vert = offsets[-1]
            if face == len(t_counts):
                if vert == len(t_points):
                    return order
            else:
                k = next(stack[-1], None)
                if k is not None:
                    counts, verts, points = cand[k]
                    order.append(k)
                    used.add(k)
                    nxt = (face + len(counts), corner + len(verts), vert + len(points))
                    offsets.append(nxt)
                    stack.append(options(*nxt))
                    continue
            # Dead end (or a complete face run with vertices left over).
            stack.pop()
            offsets.pop()
            if order:
                used.discard(order.pop())
        return None

    @classmethod
    def find_combined(
        cls,
        face_counts: Sequence[int],
        read: Callable[[int], Tuple[Any, Any, Any]],
        tolerance: Optional[float] = None,
    ) -> Optional[Tuple[int, List[int]]]:
        """The one mesh of a selection combined from ALL the others.

        What lets a tool read a selection without being told which mesh is
        which: a combined mesh holds every face of its parts, so it is the
        only one whose face count is the rest's total. That test needs only
        the counts, so a selection with no such mesh -- the usual case --
        never pays for a full topology read; *read* runs only for a candidate
        and the meshes it is checked against.

        Parameters:
            face_counts: Each mesh's face count. At least three meshes: with
                two, a copy of one mesh is equally "combined" from the other,
                and which is which is the caller's call.
            read: ``read(i) -> (counts, verts, points)`` for mesh *i*, as
                :meth:`concatenation_order` takes them. Each mesh is read
                at most once.
            tolerance: As :meth:`concatenation_order`.

        Returns:
            ``(target index, part indices in concatenation order)``, or
            ``None`` when no mesh is exactly the others combined.
        """
        n = len(face_counts)
        total = sum(face_counts)
        # Its faces must be exactly everyone else's.
        candidates = [i for i, f in enumerate(face_counts) if 2 * f == total]
        if n < 3 or not candidates:
            return None
        topo: Dict[int, Any] = {}

        def get(i):
            if i not in topo:
                topo[i] = read(i)
            return topo[i]

        for i in candidates:
            others = [j for j in range(n) if j != i]
            order = cls.concatenation_order(get(i), [get(j) for j in others], tolerance)
            if order is not None and len(order) == len(others):
                return i, [others[k] for k in order]
        return None

    # ------------------------------------------------------------- lightmaps
    #: Corner drift (UV units) under which two parameterizations are one
    #: layout. A 64k map's texel is 1.5e-5, so this is below a texel at any
    #: lightmap size and far above the float32 noise of re-reading a set.
    LAYOUT_TOLERANCE: float = 1e-5

    @classmethod
    def layouts_match(
        cls, src_tris, dst_tris, tolerance: Optional[float] = None
    ) -> bool:
        """True when two parameterizations of the same triangles are one layout.

        What decides whether a lightmap has to be resampled at all: a target
        whose lightmap UVs ARE the source's reads the source's map (and its
        atlas rect) as it stands, so it is rebound rather than re-encoded --
        exact, and an atlas stays shared.

        Parameters:
            src_tris / dst_tris: ``(N, 3, 2)`` triangles, corner for corner.
            tolerance: Largest corner drift that still counts as the same;
                default :attr:`LAYOUT_TOLERANCE`.
        """
        cls._require_numpy()
        a = np.asarray(src_tris, dtype=np.float64)
        b = np.asarray(dst_tris, dtype=np.float64)
        if a.shape != b.shape:
            return False
        tol = cls.LAYOUT_TOLERANCE if tolerance is None else float(tolerance)
        return not a.size or float(np.abs(a - b).max()) <= tol

    @classmethod
    def remap_lightmap(
        cls,
        image,
        src_tris,
        dst_tris,
        *,
        scale_offset: Optional[Sequence[float]] = None,
        size: Optional[int] = None,
        supersample: int = 2,
        padding: int = -1,
    ) -> "np.ndarray":
        """Resample one object's lightmap into another lightmap layout.

        A lightmap is not a material map: it is one object's lighting, laid out
        in that object's own lightmap UVs and often sharing one map with other
        objects through a per-object rect. So the source is read through that
        rect, and only the object's own island counts as source -- a
        neighbour's texels across the cell edge are never sampled. Values are
        HDR and are never clipped; the host reads and writes the file (EXR in
        both hosts, through different libraries).

        Parameters:
            image: The source map, ``HxW`` or ``HxWxC`` float, stored top-down
                (row 0 = V 1), as the host read it.
            src_tris: ``(N, 3, 2)`` the object's 0-1 lightmap UVs.
            dst_tris: ``(N, 3, 2)`` the target's lightmap UVs, same triangles
                and corner order.
            scale_offset: The object's rect in *image* -- ``[scaleX, scaleY,
                offsetX, offsetY]``, mapping its UVs into the map as
                ``uv * scale + offset`` with V up (Unity's
                ``lightmapScaleOffset``). ``None`` = a map of its own.
            size: Output resolution (square). Default: the texels the object
                owned in the source -- the map's width times the rect's larger
                scale, rounded UP to a power of two so no density is lost, and
                never larger than the map itself.
            supersample / padding: See :meth:`build` / :meth:`pad`.

        Returns:
            ``(size, size, C)`` float32, top-down, gutter padded.
        """
        cls._require_numpy()
        img = np.asarray(image, dtype=np.float32)
        if img.ndim == 2:
            img = img[..., None]
        if img.ndim != 3:
            raise ValueError(f"lightmap image must be HxW or HxWxC, got {img.shape}")
        sx, sy, ox, oy = (float(v) for v in (scale_offset or (1.0, 1.0, 0.0, 0.0)))
        src = np.asarray(src_tris, dtype=np.float64).reshape(-1, 3, 2) * (
            sx,
            sy,
        ) + (ox, oy)
        dst = np.asarray(dst_tris, dtype=np.float64).reshape(-1, 3, 2)
        width = int(img.shape[1])
        if size:
            res = int(size)
        else:
            owned = max(1.0, width * max(sx, sy))
            res = min(width, max(16, 2 ** math.ceil(math.log2(owned))))
        # The object's own island, at the map's resolution: everything else
        # in the map is gutter or another object's lighting, and is filled
        # from this island before sampling so a bilinear tap at the island
        # edge never reaches it.
        mask = None
        if img.shape[0] == img.shape[1]:
            from pythontk.img_utils._img_utils import ImgUtils

            mask = ImgUtils.rasterize_uv_triangles(src, size=width, supersample=1) > 0
            if not mask.any():
                mask = None
        # Only the island's texels (and a bilinear margin) are ever read, so
        # the fill works on them alone: on an atlas shared by dozens of
        # objects, filling the whole map once per object costs the whole map
        # every time.
        img, src, mask = cls._crop_to_uvs(img, src, mask)
        table = cls.build(src, dst, res, supersample=supersample)
        out, coverage = cls.transfer(table, img, source_masks=mask)
        return cls.pad(out, coverage, padding).astype(np.float32, copy=False)

    @staticmethod
    def _crop_to_uvs(image, tris, mask=None, margin: int = 2):
        """``(image, tris, mask)`` cropped to *tris*' texel bounds plus *margin*.

        *tris* (V up, in *image*'s UV space) come back re-expressed in the
        crop's own 0-1 space, so sampling the crop reads the very texels the
        whole image would have. Returned unchanged when the crop is the image.
        """
        h, w = image.shape[:2]
        uv = tris.reshape(-1, 2)
        if not len(uv):
            return image, tris, mask
        x0 = max(0, math.floor(float(uv[:, 0].min()) * w) - margin)
        x1 = min(w, math.ceil(float(uv[:, 0].max()) * w) + margin)
        y0 = max(0, math.floor((1.0 - float(uv[:, 1].max())) * h) - margin)
        y1 = min(h, math.ceil((1.0 - float(uv[:, 1].min())) * h) + margin)
        if x1 <= x0 or y1 <= y0 or (x0, y0, x1, y1) == (0, 0, w, h):
            return image, tris, mask
        local = np.empty_like(tris)
        local[..., 0] = (tris[..., 0] * w - x0) / (x1 - x0)
        local[..., 1] = 1.0 - ((1.0 - tris[..., 1]) * h - y0) / (y1 - y0)
        return (
            image[y0:y1, x0:x1],
            local,
            None if mask is None else mask[y0:y1, x0:x1],
        )

    @classmethod
    def resample_lightmaps(
        cls,
        jobs: Sequence[Dict[str, Any]],
        *,
        output_dir: str,
        read,
        write,
        output_name: Optional[str] = None,
        claims=None,
        size: Optional[int] = None,
        supersample: int = 2,
        padding: int = -1,
        log=None,
    ) -> Dict[str, str]:
        """Write each job's lightmap resampled into its target's lightmap layout.

        The DCC-agnostic half of a host's ``LightmapRecords.transfer_lightmaps``,
        as :meth:`transfer_materials` is of ``transfer``: one naming rule and one
        remap for both hosts, which supply only their EXR reader and writer. A
        job whose two layouts match (:meth:`layouts_match`) needs no new map --
        its target reads the source's map through the source's rect -- so it
        is skipped, and the host binds it to that map.

        Parameters:
            jobs: One per (target, source) pair: ``{"owner": the target, as
                *claims* names readers, "name": its name in the file name,
                "path": the source's map, "scale_offset": the source's rect,
                "src" / "dst": (N, 3, 2) the source's and the target's lightmap
                UVs, triangle for triangle}``. Other keys are ignored.
            output_dir: Folder the maps are written into.
            read: ``callable(path) -> HxWxC float``, top-down -- the host's
                reader. Called once per map, however many jobs share it.
            write: ``callable(path, image)`` -- the host's writer.
            output_name: Names the maps ``<output_name>_Lightmap`` (one map) or
                ``<output_name>_<name>_Lightmap`` (several); without it,
                ``<name>_Lightmap``.
            claims: ``{file name: readers}``. A name another object reads is
                never written over (:meth:`FileUtils.unique_path`; a job's own
                target may replace its own map), and neither is any job's
                source map.
            size / supersample / padding: See :meth:`remap_lightmap`.
            log: Optional ``callable(str)`` for one line per map.

        Returns:
            ``{owner: written path}`` for each job resampled.
        """
        from pythontk.file_utils._file_utils import FileUtils

        say = log or (lambda m: None)
        todo = [j for j in jobs if not cls.layouts_match(j["src"], j["dst"])]
        base = StrUtils.sanitize(output_name, preserve_case=True) if output_name else ""
        taken: set = set()
        avoid = {j["path"] for j in jobs}
        # Every name first, in the jobs' order: what a job's map is called
        # never depends on the order the source maps are read in.
        outputs: List[str] = []
        for job in todo:
            parts = [base] if base else []
            if not base or len(todo) > 1:
                parts.append(StrUtils.sanitize(str(job["name"]), preserve_case=True))
            outputs.append(
                FileUtils.unique_path(
                    output_dir,
                    "_".join(parts + ["Lightmap"]),
                    ".exr",
                    taken,
                    claims,
                    owners=[job["owner"]],
                    avoid=avoid,
                ).replace("\\", "/")
            )
        # Then a source map at a time, read once for every job on it and
        # dropped before the next is read: a 4k float map is ~200 MB, and the
        # host's job order interleaves the maps freely.
        by_map: Dict[str, List[int]] = {}
        for i, job in enumerate(todo):
            by_map.setdefault(job["path"], []).append(i)
        for source, members in by_map.items():
            image = read(source)
            for i in members:
                job = todo[i]
                write(
                    outputs[i],
                    cls.remap_lightmap(
                        image,
                        job["src"],
                        job["dst"],
                        scale_offset=job.get("scale_offset"),
                        size=size,
                        supersample=supersample,
                        padding=padding,
                    ),
                )
                say(
                    f"{job['name']}: lightmap resampled into its own layout "
                    f"-> {outputs[i]}"
                )
            del image
        return {job["owner"]: path for job, path in zip(todo, outputs)}

    @staticmethod
    def _uv_areas(tris) -> "np.ndarray":
        """Unsigned UV area per triangle of an ``(N, 3, 2)`` array."""
        tris = np.asarray(tris, dtype=float).reshape(-1, 3, 2)
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        )

    @classmethod
    def dominant_source(cls, job: Dict[str, Any]) -> Optional[int]:
        """Index of the source covering the most of *job*'s target layout.

        Weighted by TARGET-UV area, not triangle count, so a dense trim strip
        cannot outvote the panel that fills the layout. The host adapters
        model a layout's assigned material on that source's.

        Parameters:
            job: A :meth:`transfer_materials` job -- ``dst``, ``ids`` and
                ``sources``.

        Returns:
            int | None: Index into ``job["sources"]``; None when it names none.
        """
        cls._require_numpy()
        sources = job.get("sources") or []
        if not sources:
            return None
        ids = np.asarray(job["ids"], dtype=np.int64).reshape(-1)
        weight = np.bincount(
            ids, weights=cls._uv_areas(job["dst"]), minlength=len(sources)
        )
        return int(weight.argmax())

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "material"

    @classmethod
    def _auto_size(
        cls,
        used,
        sources,
        channels,
        src_tris=None,
        dst_tris=None,
        ids=None,
        say=None,
    ) -> int:
        """The largest source map feeding this material (2048 if none).

        That is the right size for the operation as specified -- the caller
        asked to move these maps into a new layout, not to re-budget them --
        and *size* is the dial for anything else. What this must never do is
        let a consolidation lose detail SILENTLY.

        Consolidating several texture sets into one layout is where the number
        stops meaning what it looks like: each set now owns a fraction of a map
        it used to own outright, so its texel density falls by the ratio of
        those shares while the file keeps the reassuring old resolution.
        Measured on a delivered asset (TURRETS_WIRES.glb): two 2048 sets into
        one 2048 layout, the turrets taking 57.5% of the target and the wires
        9.4% having owned ~94% of their own map -- so ~2014px of wire content
        was resampled into ~629px, a 3.2x linear loss that shipped as visibly
        flattened roughness with nothing in the log to say so.

        So the size is unchanged and the squeeze is REPORTED: per source,

            squeeze = sqrt(uv_area_at_source / uv_area_at_target)

        and anything above 1 means that source keeps ``1 / squeeze`` of its
        linear resolution here. Whether that matters is the artist's call --
        a 2k map is ample for a small asset, and inflating one on the tool's
        own initiative would quadruple every consolidation's cost uninvited.
        Resolution cannot buy back a layout that gave a set too little room
        anyway; the fix for a bad squeeze is usually the packing, and the log
        names which set so that choice can be made with the numbers in hand.

        Geometry is optional: without it there is nothing to report and the
        size is the plain floor.
        """
        from PIL import Image

        report = say or (lambda m: None)
        floor = 0
        #: source id -> (largest map edge, the map that measured it). The label
        #: comes from the SAME map as the size so a warning cannot name a map
        #: that was never read (an unresolvable path, or a channel this run
        #: never touched).
        per_source: Dict[int, Tuple[int, str]] = {}
        for sid in used:
            biggest, label = 0, f"source {sid}"
            maps = sources[sid].get("maps") or {}
            for ch in channels:
                p = maps.get(ch)
                if p and os.path.isfile(p):
                    try:
                        with Image.open(p) as im:  # header only, no decode
                            edge = max(im.size)
                    except Exception:  # noqa: BLE001
                        continue
                    if edge > biggest:
                        biggest, label = edge, os.path.splitext(os.path.basename(p))[0]
            if biggest:
                per_source[sid] = (biggest, label)
                floor = max(floor, biggest)
        if not floor:
            return 2048
        if src_tris is None or dst_tris is None or ids is None:
            return floor

        src_area = cls._uv_areas(src_tris)
        dst_area = cls._uv_areas(dst_tris)
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)

        squeezed: List[Tuple[str, int, float]] = []
        for sid, (px, label) in per_source.items():
            hit = ids == sid
            at_src = float(src_area[hit].sum())
            at_dst = float(dst_area[hit].sum())
            if at_src <= 0.0 or at_dst <= 0.0:
                continue  # contributes no area on one side; nothing to compare
            squeeze = math.sqrt(at_src / at_dst)
            if squeeze > cls._SQUEEZE_REPORT_THRESHOLD:
                squeezed.append((label, px, squeeze))
        if squeezed:
            worst = max(squeezed, key=lambda s: s[2])
            report(
                f"{len(squeezed)} source(s) take a smaller share of this layout "
                f"than they had at source, so {floor}px carries less of their "
                f"detail than it did: "
                + ", ".join(
                    f"{lbl!r} {sq:.2f}x (~{floor / sq:.0f}px of its {px}px)"
                    for lbl, px, sq in sorted(squeezed, key=lambda s: -s[2])
                )
                + f". Raise `size` above {floor} to hold {worst[0]!r} at its own "
                f"density, or give it more of the target UV layout."
            )
        return floor

    @classmethod
    def normal_convention(cls, path: str, override: Optional[str] = None) -> str:
        """The tangent-space convention of the normal map at *path*.

        Read off the map's CONTENT first, its filename second, OpenGL last.

        The content (:meth:`pythontk.ImgUtils.detect_normal_map_format`:
        whether the X and Y channels integrate as one height field) is first
        because what a transfer needs is exactly what it measures -- the
        RELATIVE handedness of X and Y, which decides which way an island's
        rotation turns them (:meth:`transfer_normals`). A filename only claims
        it, and an untagged ``_Normal`` claims nothing. The statistic's blind
        spot (an inverted X reads as an inverted Y) is no blind spot here:
        either inversion reverses a rotation's sense identically. The map
        factory, which must name a map's ABSOLUTE convention for an engine,
        rightly ranks the two the other way round.

        The filename decides when the content cannot -- a flat map, shallow
        relief, a file that is not there (where a rotation changes the least):
        :meth:`_filename_normal_convention`.

        Parameters:
            path: The normal map's file path.
            override: Returned verbatim (lowercased) instead of classifying --
                an explicit convention from the caller always wins.

        Returns:
            str: ``"directx"`` or ``"opengl"``.
        """
        if override:
            return str(override).lower()
        if os.path.isfile(str(path)):
            from pythontk.img_utils._img_utils import ImgUtils

            found = ImgUtils.detect_normal_map_format(str(path))
            if found:
                return found.lower()
        return cls._filename_normal_convention(path)

    @classmethod
    def _unify_normal_conventions(
        cls,
        loaded: Dict[int, Tuple["np.ndarray", float]],
        paths: Dict[int, str],
        ids: "np.ndarray",
        dst_tris: "np.ndarray",
        *,
        override: Optional[str] = None,
        say: Optional[Callable[[str], None]] = None,
    ) -> str:
        """The convention a consolidated normal map is written in.

        Each source map's own (:meth:`normal_convention`). Where they differ,
        the convention covering the most of the target layout wins, and every
        map in the other is converted to it in *loaded* (green flipped) before
        the remap: the output is ONE map, and a part left in the other
        convention would be lit upside down.

        Parameters:
            loaded: ``{source id: (image, value max)}`` -- updated in place.
            paths: ``{source id: normal map path}`` for every entry of *loaded*.
            ids / dst_tris: The job's source id and target UV triangle per
                triangle, which weigh each source's share of the layout.
            override: An explicit convention for every source.
            say: Optional ``callable(str)``, told about a conversion.

        Returns:
            str: ``"opengl"`` or ``"directx"``.
        """
        conv = {sid: cls.normal_convention(paths[sid], override) for sid in loaded}
        if len(set(conv.values())) <= 1:
            return next(iter(conv.values()), "opengl")
        areas = cls._uv_areas(dst_tris)
        share: Counter = Counter()
        for sid, c in conv.items():
            share[c] += float(areas[ids == sid].sum())
        out = max(share, key=share.get)
        flipped = []
        for sid, c in conv.items():
            if c != out:
                arr, vmax = loaded[sid]
                arr = arr.copy()
                arr[..., 1] = vmax - arr[..., 1]
                loaded[sid] = (arr, vmax)
                flipped.append(os.path.basename(paths[sid]))
        if say:
            say(
                f"normal maps mix conventions: {', '.join(flipped)} converted "
                f"to {out}, the convention of most of the layout"
            )
        return out

    @staticmethod
    def _say_invalid_normals(
        image: "np.ndarray",
        coverage: "np.ndarray",
        value_max: float,
        table: TransferTable,
        sources: Sequence[Dict[str, Any]],
        say: Callable[[str], None],
    ) -> None:
        """Say which source maps put texels that are no tangent-space normal --
        Z below zero, pointing into the surface: a black, unpainted background
        -- into the remapped *image*.

        They transfer as they are: the transfer is a resample, and the source
        is what the artist shipped. Turned with their island they read as a
        transfer artefact (black turned 180 degrees is yellow), so the map is
        named instead.
        """
        bad = (image[..., 2] < 0.5 * value_max) & (coverage > 0)
        if not bad.any():
            return
        tri = table.tri[0][bad]
        sids = sorted({int(s) for s in table.source_ids[tri[tri >= 0]]})
        names = [
            os.path.basename(sources[s].get("maps", {}).get("normal") or "")
            or str(sources[s].get("name") or s)
            for s in sids
        ]
        share = float(bad.sum()) / max(int((coverage > 0).sum()), 1)
        say(
            f"WARNING normal: {share:.1%} of the map reads texels that are no "
            f"tangent-space normal (Z below zero: black or unpainted) in "
            f"{', '.join(names)}; transferred as they are."
        )

    @staticmethod
    def _filename_normal_convention(path: str) -> str:
        """The tangent-space convention *path*'s filename declares.

        Classification goes through the shared map registry
        (:class:`pythontk.MapRegistry`), not a local token regex: the registry
        already enumerates every handedness spelling the ecosystem's exporters
        emit -- ``_DX`` / ``DirectX`` / ``NRMLDX`` / ``N-dx`` and their OpenGL
        twins, in EITHER order (``NormalDX`` and ``DX_Normal`` alike), across
        every delimiter it accepts -- and it strips a trailing UDIM or
        duplicate token first, so ``rock_NormalDX.1001_1.png`` classifies
        exactly like ``rock_NormalDX.png``. A pattern maintained here could
        only ever be a subset of that, and every spelling it missed silently
        flipped a normal map's green channel the wrong way.

        Anything that does not resolve to ``Normal_DirectX`` -- the untagged
        ``Normal`` type, an unrecognised name -- is reported as OpenGL, which
        is the registry's own reading of an untagged map (see
        ``MapRegistry.NORMAL_TYPES``): the convention is unknown, and leaving
        it alone is the cheaper error, since flipping a guess inverts a map
        that may already have been right.

        The line is ADJACENCY, not position: a tag touching the token is part
        of the suffix and counts (``rock_directx_normal``), while one loose
        elsewhere in the name is not a declaration and does not
        (``DirectX_rock_Normal``, ``rock_directx_final_normal``, or a
        ``dx_project/`` directory in the path).

        Parameters:
            path: The map's file path (only its filename is read).

        Returns:
            str: ``"directx"`` or ``"opengl"``.
        """
        from pythontk.core_utils.engines.textures.map_registry import MapRegistry

        map_type = MapRegistry().resolve_type_from_path(str(path))
        return "directx" if map_type == "Normal_DirectX" else "opengl"

    @staticmethod
    def load_map(path: str) -> Tuple["np.ndarray", float]:
        """``(HxWxC float32, value max)`` -- 255 for 8-bit, 65535 for 16-bit."""
        from pythontk.img_utils._img_utils import ImgUtils

        img = ImgUtils.load_image(path)
        mode = img.mode
        if mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(img, dtype=np.float32)
            return (arr[..., None] if arr.ndim == 2 else arr), 65535.0
        if mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGBA" if "A" in mode else "RGB")
        arr = np.asarray(img, dtype=np.float32)
        return (arr[..., None] if arr.ndim == 2 else arr), 255.0

    @staticmethod
    def save_map(path: str, arr: "np.ndarray", value_max: float = 255.0) -> str:
        """Write *arr* (in ``0..value_max``) as PNG -- 16-bit grey, else 8-bit."""
        from PIL import Image

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arr = np.clip(np.asarray(arr, dtype=np.float32), 0.0, value_max)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[..., 0]
        if value_max > 255.0:
            if arr.ndim == 2:
                Image.fromarray(np.rint(arr).astype(np.uint16), mode="I;16").save(path)
                return path
            arr = arr / value_max * 255.0  # PIL has no 16-bit RGB PNG writer
        Image.fromarray(np.rint(arr).astype(np.uint8)).save(path)
        return path

    # ----------------------------------------------------------------- frames
    @classmethod
    def triangle_frames(cls, src_tris, dst_tris) -> "np.ndarray":
        """``(N, 2, 2)`` rotation (+reflection) taking each source triangle's
        tangent frame to its target frame.

        The affine map ``A`` between the two UV triangles is polar-decomposed
        (``A = R P``, ``R`` orthogonal); ``R`` is returned, carrying a
        reflection (``det R = -1``) for mirrored islands. Degenerate
        triangles get the identity.
        """
        cls._require_numpy()
        src = np.asarray(src_tris, dtype=np.float64).reshape(-1, 3, 2)
        dst = np.asarray(dst_tris, dtype=np.float64).reshape(-1, 3, 2)
        n = len(src)
        out = np.tile(np.eye(2), (n, 1, 1))
        if n == 0:
            return out
        S = np.stack([src[:, 1] - src[:, 0], src[:, 2] - src[:, 0]], axis=2)  # (N,2,2)
        D = np.stack([dst[:, 1] - dst[:, 0], dst[:, 2] - dst[:, 0]], axis=2)
        detS = np.linalg.det(S)
        ok = np.abs(detS) > 1e-14
        if not ok.any():
            return out
        A = D[ok] @ np.linalg.inv(S[ok])  # dst = A @ src
        U, _, Vt = np.linalg.svd(A)
        out[ok] = U @ Vt
        return out

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _require_numpy():
        if np is None:  # pragma: no cover
            raise RuntimeError("UvTransfer requires numpy")

    @staticmethod
    def _size_hw(size) -> Tuple[int, int]:
        if isinstance(size, (int, np.integer)):
            s = int(size)
            if s <= 0:
                raise ValueError("size must be positive")
            return s, s
        h, w = int(size[0]), int(size[1])
        if h <= 0 or w <= 0:
            raise ValueError("size must be positive")
        return h, w

    @classmethod
    def _normalize_sources(
        cls, sources, table: TransferTable, value_max: Optional[float] = None
    ):
        """``({id: float32 array | constant array}, channels)``.

        Arrays are promoted to ``HxWxC`` float32; constants to ``(C,)``; every
        entry is widened to the common channel count by meaning, not position:
        grey (or grey+alpha) fills the COLOUR channels, and an alpha the source
        does not have is OPAQUE -- *value_max*, else guessed from the data.
        Repeating grey into all four channels wrote a metal-0 map's region of a
        packed RGBA layout at alpha 0.
        """
        if isinstance(sources, dict):
            raw = {int(k): v for k, v in sources.items()}
        else:
            raw = {int(i): sources for i in np.unique(table.source_ids)}
        norm: Dict[int, Any] = {}
        channels = 1
        for k, v in raw.items():
            arr = np.asarray(v)
            if arr.ndim >= 2:
                arr = arr.astype(np.float32, copy=False)
                if arr.ndim == 2:
                    arr = arr[..., None]
                norm[k] = arr
                channels = max(channels, arr.shape[2])
            else:
                c = np.asarray(arr, dtype=np.float32).reshape(-1)
                norm[k] = c
                channels = max(channels, len(c))
        # Colour / alpha split of each layout: L, LA, RGB, RGBA.
        colour_of = {1: 1, 2: 1, 3: 3, 4: 3}
        for k, v in list(norm.items()):
            have = v.shape[2] if v.ndim == 3 else len(v)
            if have >= channels:
                continue
            flat = v.reshape(-1, have)
            colour = flat[:, : colour_of[have]]
            want = colour_of[channels]
            if colour.shape[1] < want:  # grey -> every colour channel
                colour = np.repeat(colour, want, axis=1)
            parts = [colour]
            if channels in (2, 4):
                if have in (2, 4):
                    alpha = flat[:, -1:]
                else:
                    one = (
                        float(value_max)
                        if value_max is not None
                        else (255.0 if float(np.max(v, initial=0.0)) > 1.0 else 1.0)
                    )
                    alpha = np.full((flat.shape[0], 1), one, np.float32)
                parts.append(alpha)
            widened = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
            norm[k] = widened.reshape(v.shape[:-1] + (channels,))
        return norm, channels

    @staticmethod
    def _prefill_sources(src_by_id, source_masks):
        from pythontk.img_utils._img_utils import ImgUtils

        masks = (
            {int(k): v for k, v in source_masks.items()}
            if isinstance(source_masks, dict)
            else {k: source_masks for k in src_by_id}
        )
        out = dict(src_by_id)
        for k, m in masks.items():
            img = out.get(k)
            if img is None or not (isinstance(img, np.ndarray) and img.ndim == 3):
                continue
            mask = np.asarray(m).astype(bool)
            if mask.shape != img.shape[:2]:
                raise ValueError(
                    f"source mask {mask.shape} != source image {img.shape[:2]} (id {k})"
                )
            if mask.all():
                continue
            out[k] = ImgUtils.fill_empty_texels(img, mask).astype(
                np.float32, copy=False
            )
        return out

    @classmethod
    def _dense_pass(cls, table, tri_map, uv_map, src_by_id, channels, bilinear):
        """One sub-sample pass as dense ``(H, W, C)`` samples + ``(H, W)`` cover.

        Dense on purpose: a pass touches most texels of the output, and
        whole-image numpy ops (one bilinear gather per corner, masked adds)
        run several times faster than gathering the covered texels into a
        list and scattering the results back by index. Uncovered texels are
        sampled at UV (0, 0) and zeroed by the mask.
        """
        covered = tri_map >= 0
        if not covered.any():
            return None, covered
        h, w = tri_map.shape
        u = uv_map[..., 0].astype(np.float32) * np.float32(1.0 / 65535.0)
        v = uv_map[..., 1].astype(np.float32) * np.float32(1.0 / 65535.0)
        ids_present = np.unique(table.source_ids)
        out = np.zeros((h, w, channels), dtype=np.float32)
        sid_map = (
            table.source_ids[np.maximum(tri_map, 0)] if len(ids_present) > 1 else None
        )
        for sid in ids_present:
            sel = covered if sid_map is None else (covered & (sid_map == sid))
            if sid_map is not None and not sel.any():
                continue
            value = src_by_id.get(int(sid))
            if value is None:
                raise KeyError(
                    f"triangles reference source id {int(sid)} but `sources` "
                    f"has no entry for it (have {sorted(src_by_id)})"
                )
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                s = cls._sample_dense(value, u, v, bilinear)
            else:  # constant: (1, 1, C) broadcasts like an image
                s = np.asarray(value, dtype=np.float32).reshape(1, 1, -1)
            out = np.where(sel[..., None], s, out)
        return out, covered

    @staticmethod
    def _sample_dense(image, u, v, bilinear: bool) -> "np.ndarray":
        """Sample ``HxWxC`` *image* at dense ``(h, w)`` UV grids (V up)."""
        ih, iw, c = image.shape
        flat = image.reshape(-1, c)
        x = u * np.float32(iw) - np.float32(0.5)
        y = (np.float32(1.0) - v) * np.float32(ih) - np.float32(0.5)
        if not bilinear:
            xi = np.clip(np.rint(x).astype(np.int32), 0, iw - 1)
            yi = np.clip(np.rint(y).astype(np.int32), 0, ih - 1)
            return flat[yi * iw + xi]
        x0 = np.floor(x)
        y0 = np.floor(y)
        fx = (x - x0)[..., None]
        fy = (y - y0)[..., None]
        x0 = x0.astype(np.int32)
        y0 = y0.astype(np.int32)
        x1 = np.clip(x0 + 1, 0, iw - 1)
        y1 = np.clip(y0 + 1, 0, ih - 1)
        np.clip(x0, 0, iw - 1, out=x0)
        np.clip(y0, 0, ih - 1, out=y0)
        r0 = y0 * iw
        r1 = y1 * iw
        top = flat[r0 + x0] * (1.0 - fx) + flat[r0 + x1] * fx
        bot = flat[r1 + x0] * (1.0 - fx) + flat[r1 + x1] * fx
        return top * (1.0 - fy) + bot * fy
