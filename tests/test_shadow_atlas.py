# !/usr/bin/python
# coding=utf-8
"""ShadowAtlas -- equal-cell grid packing of shadow tiles, rewritten in place."""

import unittest

import numpy as np

from pythontk import ImgUtils, ShadowAtlas


class TestShadowAtlas(unittest.TestCase):
    @staticmethod
    def _tile(size, alpha):
        t = np.zeros((size, size, 4), dtype=np.uint8)
        t[:, :, 3] = alpha
        return t

    def test_grid_is_as_square_as_the_count_allows(self):
        self.assertEqual(ShadowAtlas.grid(1, (8, 8)), (1, 1, (8, 8)))
        self.assertEqual(ShadowAtlas.grid(3, (8, 8)), (2, 2, (16, 16)))
        self.assertEqual(ShadowAtlas.grid(5, (4, 2)), (3, 2, (12, 4)))

    def test_pack_places_tiles_and_publishes_inset_rects(self):
        """Three 8 × 8 tiles pack into a 16 × 16 atlas in name order; every
        published rect is the tile's pixel rect inset by the gutter, exactly
        as the lightmap helpers inset it."""
        tiles = {"c": self._tile(8, 30), "a": self._tile(8, 10), "b": self._tile(8, 20)}
        atlas, rects, pixel_rects = ShadowAtlas.pack(tiles, gutter=2)
        self.assertEqual(atlas.shape, (16, 16, 4))
        self.assertEqual(list(rects), ["a", "b", "c"])
        self.assertEqual(pixel_rects["a"], (0, 8, 0, 8))
        self.assertEqual(pixel_rects["b"], (0, 8, 8, 16))
        self.assertEqual(pixel_rects["c"], (8, 16, 0, 8))
        for name, alpha in (("a", 10), ("b", 20), ("c", 30)):
            r0, r1, c0, c1 = pixel_rects[name]
            self.assertTrue((atlas[r0:r1, c0:c1, 3] == alpha).all())
            raw = ShadowAtlas.uv_rect(pixel_rects[name], (16, 16))
            self.assertEqual(
                rects[name], ImgUtils.inset_atlas_rects([raw], (16, 16), 2)[0]
            )
        self.assertEqual(
            ShadowAtlas.uv_rect(pixel_rects["a"], (16, 16)), (0.5, 0.5, 0.0, 0.5)
        )
        self.assertEqual(atlas[8:16, 8:16, 3].max(), 0)  # the empty cell

    def test_write_tile_rewrites_one_cell_only(self):
        tiles = {"a": self._tile(8, 10), "b": self._tile(8, 20)}
        atlas, _, pixel_rects = ShadowAtlas.pack(tiles)
        ShadowAtlas.write_tile(atlas, pixel_rects["b"], self._tile(8, 99))
        r0, r1, c0, c1 = pixel_rects["b"]
        self.assertTrue((atlas[r0:r1, c0:c1, 3] == 99).all())
        r0, r1, c0, c1 = pixel_rects["a"]
        self.assertTrue((atlas[r0:r1, c0:c1, 3] == 10).all())
        with self.assertRaises(ValueError):
            ShadowAtlas.write_tile(atlas, pixel_rects["b"], self._tile(4, 1))

    def test_mixed_sizes_take_the_largest_cell(self):
        atlas, rects, pixel_rects = ShadowAtlas.pack(
            {"big": self._tile(8, 1), "small": self._tile(4, 2)}
        )
        self.assertEqual(atlas.shape, (8, 16, 4))
        self.assertEqual(pixel_rects["small"], (0, 4, 8, 12))
        self.assertEqual(
            ShadowAtlas.uv_rect(pixel_rects["small"], (16, 8)), (0.25, 0.5, 0.5, 0.5)
        )

    def test_alpha_only_tiles_and_uv_corners(self):
        alpha = np.full((4, 4), 7, dtype=np.uint8)
        atlas, rects, _ = ShadowAtlas.pack({"x": alpha}, gutter=0)
        self.assertEqual(atlas.shape, (4, 4, 4))
        self.assertTrue((atlas[:, :, 3] == 7).all())
        self.assertEqual(rects["x"], (1.0, 1.0, 0.0, 0.0))
        self.assertEqual(
            ShadowAtlas.uv_corners((0.5, 0.5, 0.0, 0.5)),
            [(0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)],
        )

    def test_a_tile_larger_than_an_explicit_cell_is_refused(self):
        """``pack`` placed each tile at its CELL's origin but sized the rect
        from the TILE, with nothing checking tile <= cell.

        With an explicit undersized ``cell=`` the oversized tile wrote over
        its neighbours' cells and published a rect spanning them -- and
        ``write_tile``'s size guard could never catch it, because the rect it
        checks against had just been derived from ``tile.shape``. At the
        atlas edge it died instead with an opaque numpy broadcast error.
        """
        tiles = {
            "a": self._tile(32, 255),
            "b": self._tile(16, 128),
            "c": self._tile(16, 64),
            "d": self._tile(16, 32),
        }
        with self.assertRaises(ValueError) as caught:
            ShadowAtlas.pack(tiles, cell=(16, 16))
        self.assertIn("a", str(caught.exception))

    def test_zero_size_tile_is_the_documented_error(self):
        """``_rgba``'s contract is a clear ValueError for a bad tile, but the
        non-uint8 branch reduced with ``arr.max()`` first, so an empty tile
        surfaced numpy's "zero-size array to reduction operation" instead."""
        with self.assertRaises(ValueError) as caught:
            ShadowAtlas.pack({"a": np.zeros((0, 0, 4), dtype=np.float32)})
        self.assertIn("ShadowAtlas", str(caught.exception))

    def test_uint16_tile_is_rescaled_not_clipped_to_white(self):
        """A uint16 tile hit ``arr.max() <= 1.0`` as False, so it was scaled
        by 1.0 and clipped: every value above 255 collapsed to pure white.
        Range-reduce from the dtype instead, the way Ktx2Encoder does."""
        tile = np.zeros((4, 4, 4), dtype=np.uint16)
        tile[:, :, 3] = 65535  # full alpha
        tile[0, 0, 3] = 32768  # ~half
        atlas, _, prects = ShadowAtlas.pack({"a": tile})
        row0, _, col0, _ = prects["a"]
        self.assertEqual(int(atlas[row0 + 1, col0 + 1, 3]), 255)
        self.assertAlmostEqual(int(atlas[row0, col0, 3]), 128, delta=1)

    def test_wide_int_tile_already_in_8bit_range_is_not_darkened(self):
        """Range-reducing from the DTYPE fixes uint16, but must not touch a
        wide integer type that merely holds 0-255 values.

        ``np.zeros((h, w, 4), dtype=int)`` is int64, so a caller building a
        tile from plain Python ints lands here; scaling that by
        ``255 / iinfo(int64).max`` would take every texel to black.
        """
        tile = np.zeros((4, 4, 4), dtype=np.int64)
        tile[:, :, 3] = 200
        atlas, _, prects = ShadowAtlas.pack({"a": tile})
        row0, _, col0, _ = prects["a"]
        self.assertEqual(int(atlas[row0, col0, 3]), 200)

    def test_empty_input_is_an_error(self):
        with self.assertRaises(ValueError):
            ShadowAtlas.pack({})


if __name__ == "__main__":
    unittest.main(verbosity=2)
