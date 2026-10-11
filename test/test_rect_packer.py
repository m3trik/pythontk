# !/usr/bin/python
# coding=utf-8
"""RectPacker -- integer rectangle packing with padding, block alignment and a uniform-scale fit."""

import random
import time
import unittest

from pythontk import RectPacker, RectPackResult


def _gap(a, b):
    """Chebyshev separation of two (row0, row1, col0, col1) rects; <= 0 when they touch or overlap."""
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return max(bc0 - ac1, ac0 - bc1, br0 - ar1, ar0 - br1)


class TestPack(unittest.TestCase):
    def assertPacked(self, result, bin_size, padding, align):
        width, height = bin_size
        rects = result.pixel_rects()
        for i, (r0, r1, c0, c1) in enumerate(rects):
            self.assertGreaterEqual(c0, padding, rects[i])
            self.assertGreaterEqual(r0, padding, rects[i])
            self.assertLessEqual(c1, width - padding, rects[i])
            self.assertLessEqual(r1, height - padding, rects[i])
            # The padded slot starts on the block grid.
            self.assertEqual((c0 - padding) % align, 0, rects[i])
            self.assertEqual((r0 - padding) % align, 0, rects[i])
            for j in range(i):
                self.assertGreaterEqual(_gap(rects[i], rects[j]), 2 * padding)

    def test_places_every_rect_inside_the_bin_without_overlap(self):
        sizes = [(30, 10), (10, 30), (20, 20), (5, 5), (40, 8), (8, 40)]
        result = RectPacker.pack(sizes, (64, 64))
        self.assertIsInstance(result, RectPackResult)
        self.assertEqual(result.sizes, tuple(sizes))
        self.assertPacked(result, (64, 64), 0, 1)

    def test_padding_keeps_twice_its_width_between_rects_and_its_width_from_the_edge(
        self,
    ):
        sizes = [(12, 12)] * 9
        result = RectPacker.pack(sizes, (64, 64), padding=2)
        self.assertPacked(result, (64, 64), 2, 1)

    def test_no_compression_block_holds_two_rects(self):
        """A block-compressed encode reads 4 x 4 texel blocks: a block holding
        texels of two rects mixes them in the encode, a seam no gutter hides."""
        sizes = [(5, 7), (3, 3), (9, 2), (6, 6), (1, 1), (13, 4), (2, 9)]
        result = RectPacker.pack(sizes, (64, 64), padding=1, align=4)
        self.assertPacked(result, (64, 64), 1, 4)
        blocks = []
        for (r0, r1, c0, c1), (w, h) in zip(result.pixel_rects(), sizes):
            self.assertEqual((c1 - c0, r1 - r0), (w, h))
            blocks.append(
                {
                    (bx, by)
                    for bx in range(c0 // 4, (c1 - 1) // 4 + 1)
                    for by in range(r0 // 4, (r1 - 1) // 4 + 1)
                }
            )
        for i in range(len(blocks)):
            for j in range(i):
                self.assertFalse(blocks[i] & blocks[j], (i, j))

    def test_reports_none_when_the_rects_do_not_fit(self):
        self.assertIsNone(RectPacker.pack([(40, 40), (40, 40)], (64, 64)))
        self.assertIsNone(RectPacker.pack([(65, 1)], (64, 64)))

    def test_rotation_turns_a_rect_that_only_fits_on_its_side(self):
        self.assertIsNone(RectPacker.pack([(10, 70)], (80, 20)))
        result = RectPacker.pack([(10, 70)], (80, 20), rotate=True)
        self.assertEqual(result.rotated, (True,))
        self.assertEqual(result.sizes, ((70, 10),))

    def test_equal_rects_keep_the_callers_order(self):
        """Copies of one mesh are identical rects: they must land in the order
        the caller gives (its names), never shuffled by the sort."""
        result = RectPacker.pack([(10, 10)] * 6, (64, 16))
        xs = [x for x, _y in result.positions]
        self.assertEqual(xs, sorted(xs))

    def test_empty_input_packs(self):
        result = RectPacker.pack([], (8, 8))
        self.assertEqual(result.positions, ())
        self.assertEqual(result.fill, 0.0)

    def test_rejects_degenerate_input(self):
        with self.assertRaises(ValueError):
            RectPacker.pack([(0, 4)], (8, 8))
        with self.assertRaises(ValueError):
            RectPacker.pack([(4, 4)], (0, 8))
        with self.assertRaises(ValueError):
            RectPacker.pack([(4, 4)], (8, 8), padding=-1)


class TestFit(unittest.TestCase):
    def test_one_scale_for_every_rect_and_each_keeps_its_aspect(self):
        extents = [(4.0, 1.0), (1.0, 1.0), (2.0, 3.0), (0.5, 0.5)]
        result = RectPacker.fit(extents, (256, 256), padding=2, align=1)
        for (w, h), (ew, eh) in zip(result.sizes, extents):
            self.assertAlmostEqual(w / ew, result.scale, delta=1.0 / ew + 1e-9)
            self.assertAlmostEqual(h / eh, result.scale, delta=1.0 / eh + 1e-9)

    def test_the_scale_found_is_near_the_largest_that_fits(self):
        random.seed(11)
        extents = [(random.uniform(0.5, 4), random.uniform(0.5, 4)) for _ in range(40)]
        result = RectPacker.fit(extents, (512, 512), padding=2, align=4, min_size=4)
        # 3% larger must not fit in any order: the bisection converged.
        bigger = [
            (
                max(4, int(-(-(result.scale * 1.03 * w) // 1))),
                max(4, int(-(-(result.scale * 1.03 * h) // 1))),
            )
            for w, h in extents
        ]
        self.assertIsNone(RectPacker.pack(bigger, (512, 512), padding=2, align=4))

    def test_a_tight_fit_fills_most_of_the_bin(self):
        random.seed(5)
        extents = [(random.uniform(0.3, 3), random.uniform(0.3, 3)) for _ in range(60)]
        result = RectPacker.fit(extents, (1024, 1024), padding=0, align=1)
        self.assertGreater(result.fill, 0.8)

    def test_tiny_items_clamp_to_the_minimum_size(self):
        extents = [(100.0, 100.0), (0.001, 0.001), (0.0, 0.0)]
        result = RectPacker.fit(extents, (128, 128), padding=1, align=4, min_size=4)
        self.assertEqual(result.sizes[1], (4, 4))
        self.assertEqual(result.sizes[2], (4, 4))

    def test_none_when_even_minimum_sizes_do_not_fit(self):
        self.assertIsNone(
            RectPacker.fit([(1.0, 1.0)] * 20, (16, 16), padding=1, align=4, min_size=4)
        )

    def test_extents_equal_but_for_float_noise_size_and_order_as_equal(self):
        """Maya reports surface area in single precision: four instances of one
        baseboard, rotated in place, measured these areas. Copies of one mesh
        must not straddle a texel boundary into different sizes, nor trade
        places in the layout between two bakes of one room."""
        measured = [0.8500002, 0.8500004, 0.8500005, 0.8499908]
        noisy = [(5.0 * (a / 0.85) ** 0.5, 0.17 * (a / 0.85) ** 0.5) for a in measured]
        clean = [(5.0, 0.17)] * 4
        a = RectPacker.fit(noisy, (256, 256), padding=2, align=4, min_size=4)
        b = RectPacker.fit(clean, (256, 256), padding=2, align=4, min_size=4)
        self.assertEqual(len(set(a.sizes)), 1)
        self.assertEqual(a.positions, b.positions)
        self.assertEqual(a.sizes, b.sizes)

    def test_a_tie_run_is_measured_from_its_first_member(self):
        """Near-equal is not transitive: steps of 0.07% chained across ten
        items must not merge sizes 0.6% apart into one run."""
        steps = [(1.0 * 1.0007**k, 1.0) for k in range(10)]
        unified = RectPacker._unify_ties(steps)
        self.assertGreater(len(set(unified)), 1)
        self.assertGreaterEqual(unified[-1][0], steps[-1][0])

    def test_fold_keeps_one_long_rect_from_capping_the_scale_for_all(self):
        """A rect longer than the bin at the scale the set could otherwise
        reach caps that scale for every rect; folded toward square at its own
        area, it costs only its aspect. Unfolded, the 100:1 strip held the
        squares to the 4-texel floor."""
        extents = [(100.0, 1.0)] + [(1.0, 1.0)] * 20
        stiff = RectPacker.fit(extents, (256, 256), padding=2, align=4, min_size=4)
        folded = RectPacker.fit(
            extents, (256, 256), padding=2, align=4, min_size=4, fold=True
        )
        self.assertLessEqual(max(stiff.sizes[1]), 4)
        self.assertGreater(min(folded.sizes[1]), 3 * max(stiff.sizes[1]))
        strip_w, strip_h = folded.sizes[0]
        self.assertLessEqual(strip_w, 256 - 2 * 2)
        # Its texels are its own: about its area at the scale the set got.
        self.assertAlmostEqual(
            strip_w * strip_h / (100.0 * folded.scale**2), 1.0, delta=0.15
        )

    def test_fold_leaves_a_rect_that_fits_in_its_shape(self):
        extents = [(4.0, 1.0), (1.0, 1.0), (2.0, 3.0)]
        stiff = RectPacker.fit(extents, (256, 256), padding=2, align=4, min_size=4)
        folded = RectPacker.fit(
            extents, (256, 256), padding=2, align=4, min_size=4, fold=True
        )
        self.assertEqual(stiff.sizes, folded.sizes)

    def test_fold_never_caps_a_long_bin_below_the_unfolded_fit(self):
        """The fold's scale bound took the bin's SHORTER side as the side of
        the square a folded rect must fit: in a 1000 x 10 bin a 100 x 1 strip
        that fits unfolded at scale 10 was held to scale 1."""
        stiff = RectPacker.fit([(100.0, 1.0)], (1000, 10))
        folded = RectPacker.fit([(100.0, 1.0)], (1000, 10), fold=True)
        self.assertAlmostEqual(stiff.scale, 10.0)
        self.assertGreaterEqual(folded.scale, stiff.scale * (1.0 - 0.002))

    def test_a_zero_tolerance_converges(self):
        """Bisected to adjacent floats, the midpoint rounds onto an end and
        the interval stops shrinking: a tolerance of 0 spun forever."""
        result = RectPacker.fit(
            [(1.0, 1.0)] * 7 + [(2.0, 1.0)] * 3,
            (64, 64),
            padding=1,
            align=4,
            tolerance=0.0,
        )
        self.assertIsNotNone(result)
        self.assertGreater(result.scale, 0.0)

    def test_packs_hundreds_quickly(self):
        random.seed(2)
        extents = [(random.uniform(0.1, 5), random.uniform(0.1, 5)) for _ in range(400)]
        start = time.perf_counter()
        result = RectPacker.fit(extents, (2048, 2048), padding=8, align=4, min_size=4)
        self.assertIsNotNone(result)
        self.assertLess(time.perf_counter() - start, 10.0)


if __name__ == "__main__":
    unittest.main()
