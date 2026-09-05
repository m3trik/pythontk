# !/usr/bin/python
# coding=utf-8
"""One atlas per shadow-rig type: equal cells, a tile rewritten in place.

The lightmap baker packs maps of *varying* size by area (a squarified treemap,
:meth:`ImgUtils.compute_atlas_layout`); shadow tiles are fixed squares — the
projected rig's canvas is absorbed by the plane's scale and a horizon map's
block is a fixed grid — so a plain grid of equal cells is both optimal and
stable: adding a rig appends a cell, and recalculating one rewrites its texels
without moving anyone else's rect. Rects follow the lightmap convention
(``[scaleX, scaleY, offsetX, offsetY]``, bottom-left origin) and reuse the
same inset / snap helpers, so an engine applies them exactly as it applies a
lightmap's ``scaleOffset``. Pure numpy: both DCC twins (one of them without
PIL) composite with it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pythontk.img_utils._img_utils import ImgUtils

__all__ = ["ShadowAtlas"]

Rect = Tuple[float, float, float, float]
PixelRect = Tuple[int, int, int, int]  # (row0, row1, col0, col1), half-open, top-down


class ShadowAtlas:
    """Grid-pack equal shadow tiles into one RGBA atlas (module doc)."""

    #: Texels of gutter inset on every side of a published rect.
    GUTTER = 2

    @staticmethod
    def grid(count: int, cell: Sequence[int]) -> Tuple[int, int, Tuple[int, int]]:
        """``(cols, rows, (width, height))`` of the atlas holding *count*
        cells of ``cell = (w, h)`` texels, as square a grid as the count
        allows."""
        n = max(int(count), 1)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        w, h = int(cell[0]), int(cell[1])
        return cols, rows, (cols * w, rows * h)

    @classmethod
    def cell_pixel_rect(
        cls, index: int, cols: int, cell: Sequence[int], tile: Sequence[int]
    ) -> PixelRect:
        """Pixel rect of the tile ``tile = (w, h)`` sitting at the top-left of
        cell *index* in a *cols*-wide grid of ``cell``-sized cells."""
        c, r = int(index) % int(cols), int(index) // int(cols)
        cw, ch = int(cell[0]), int(cell[1])
        tw, th = int(tile[0]), int(tile[1])
        return r * ch, r * ch + th, c * cw, c * cw + tw

    @staticmethod
    def uv_rect(pixel_rect: PixelRect, atlas_size: Sequence[int]) -> Rect:
        """The bottom-left ``scaleOffset`` of a top-down pixel rect — the
        same arithmetic :meth:`ImgUtils.snap_atlas_rects` publishes."""
        row0, row1, col0, col1 = pixel_rect
        w, h = float(atlas_size[0]), float(atlas_size[1])
        return ((col1 - col0) / w, (row1 - row0) / h, col0 / w, 1.0 - row1 / h)

    @classmethod
    def pack(
        cls,
        tiles: Dict[str, np.ndarray],
        *,
        gutter: int = GUTTER,
        cell: Optional[Sequence[int]] = None,
        order: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Rect], Dict[str, PixelRect]]:
        """Composite *tiles* into one atlas.

        Parameters:
            tiles: ``{name: (h, w, 4) uint8}``; tiles may differ in size, the
                cell is the largest (or *cell*).
            gutter: Texels inset on every side of the published rect so
                bilinear taps and mips never read a neighbour.
            cell: ``(w, h)`` cell size override.
            order: Cell order; sorted names when None, so the same set packs
                the same way every time.

        Returns:
            ``(atlas, rects, pixel_rects)`` — the ``uint8`` RGBA atlas, the
            published (inset) ``scaleOffset`` per name, and each tile's
            placement rect for :meth:`write_tile`.
        """
        if not tiles:
            raise ValueError("ShadowAtlas.pack: no tiles.")
        names = list(order) if order is not None else sorted(tiles)
        arrays = {n: cls._rgba(tiles[n]) for n in names}
        if cell is None:
            cell = (
                max(a.shape[1] for a in arrays.values()),
                max(a.shape[0] for a in arrays.values()),
            )
        cols, rows, size = cls.grid(len(names), cell)
        atlas = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        rects: Dict[str, Rect] = {}
        pixel_rects: Dict[str, PixelRect] = {}
        for i, name in enumerate(names):
            tile = arrays[name]
            # The rect below is sized from the TILE but placed at the CELL's
            # origin, so a tile wider or taller than the cell silently writes
            # over its neighbours' cells and publishes a rect spanning them.
            # write_tile's own size guard cannot catch that: the rect it
            # checks against was just derived from this tile's shape. Only an
            # explicit `cell=` can get here -- the computed default is the max.
            if tile.shape[1] > cell[0] or tile.shape[0] > cell[1]:
                raise ValueError(
                    f"ShadowAtlas.pack: tile {name!r} is "
                    f"{tile.shape[1]}x{tile.shape[0]}, larger than the cell "
                    f"{cell[0]}x{cell[1]}."
                )
            prect = cls.cell_pixel_rect(i, cols, cell, (tile.shape[1], tile.shape[0]))
            cls.write_tile(atlas, prect, tile)
            pixel_rects[name] = prect
            rects[name] = ImgUtils.inset_atlas_rects(
                [cls.uv_rect(prect, size)], size, gutter
            )[0]
        return atlas, rects, pixel_rects

    @staticmethod
    def write_tile(atlas: np.ndarray, pixel_rect: PixelRect, tile: np.ndarray) -> None:
        """Rewrite one tile in place (Recalculate): the tile must match the
        rect it was packed into."""
        row0, row1, col0, col1 = pixel_rect
        t = ShadowAtlas._rgba(tile)
        if t.shape[0] != row1 - row0 or t.shape[1] != col1 - col0:
            raise ValueError(
                f"ShadowAtlas.write_tile: tile {t.shape[1]}×{t.shape[0]} does not fit "
                f"its rect {col1 - col0}×{row1 - row0}."
            )
        atlas[row0:row1, col0:col1] = t

    @staticmethod
    def _rgba(tile: np.ndarray) -> np.ndarray:
        arr = np.asarray(tile)
        if arr.ndim == 2:
            arr = np.dstack([np.zeros_like(arr)] * 3 + [arr])
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise ValueError(
                "ShadowAtlas: tiles must be (h, w, 4) RGBA or (h, w) alpha."
            )
        if arr.size == 0:
            # Checked before the reduction below: arr.max() on an empty array
            # raises numpy's "zero-size array to reduction" instead of the
            # clear error this helper's contract promises.
            raise ValueError(
                f"ShadowAtlas: tile is empty ({arr.shape[1]}x{arr.shape[0]})."
            )
        if arr.dtype != np.uint8:
            # The observed peak picks the range; the DTYPE only picks the
            # divisor once the values are provably wider than 8-bit. Reducing
            # from the dtype unconditionally would take an int64 tile holding
            # 0-255 -- what `np.zeros(..., dtype=int)` gives a caller building
            # one from plain Python ints -- straight to black.
            peak = float(arr.max())
            if peak <= 1.0:
                scale = 255.0  # normalized floats, and boolean masks
            elif peak > 255.0 and np.issubdtype(arr.dtype, np.integer):
                # Genuinely wide: divide by the dtype's range, not by the
                # peak, or a DIM 16-bit tile is wrongly brightened to full.
                # This is the case that used to scale by 1.0 and clip, so a
                # uint16 tile came out pure white.
                scale = 255.0 / float(np.iinfo(arr.dtype).max)
            else:
                scale = 1.0  # already 8-bit valued, whatever the container
            arr = np.clip(np.rint(np.asarray(arr, dtype=float) * scale), 0, 255).astype(
                np.uint8
            )
        return arr

    @classmethod
    def uv_corners(cls, rect: Rect) -> List[Tuple[float, float]]:
        """The four UV corners ``(u, v)`` of a plane whose unit UVs are
        remapped into *rect*: ``(0,0), (1,0), (1,1), (0,1)`` → ``uv * scale +
        offset``."""
        sx, sy, ox, oy = rect
        return [(ox, oy), (ox + sx, oy), (ox + sx, oy + sy), (ox, oy + sy)]
