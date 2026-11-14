# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.engines.lighting.look_match.target: regions and the reference."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from pythontk.core_utils.engines.lighting.look_match.target import (
    LookRegion,
    LookTarget,
    RegionShape,
)


class TestLookRegion(unittest.TestCase):
    def test_a_rect_mask_covers_its_pixels(self):
        region = LookRegion("box", "area", {"rect": [0.25, 0.5, 0.75, 1.0]})
        mask = region.mask("ours", (8, 4))
        self.assertEqual(mask.shape, (4, 8))
        self.assertEqual(int(mask.sum()), 8)  # columns 2..5, rows 2..3
        self.assertTrue(mask[2:4, 2:6].all())

    def test_each_side_has_its_own_shape(self):
        region = LookRegion(
            "box", "area", {"rect": [0, 0, 0.5, 0.5]}, ref={"rect": [0.5, 0.5, 1, 1]}
        )
        self.assertTrue(region.mask("ours", (4, 4))[0, 0])
        self.assertTrue(region.mask("ref", (4, 4))[3, 3])
        self.assertFalse(region.mask("ref", (4, 4))[0, 0])

    def test_the_ref_side_defaults_to_ours(self):
        region = LookRegion("box", "area", {"ellipse": [0.5, 0.5, 0.25, 0.25]})
        np.testing.assert_array_equal(
            region.mask("ref", (16, 16)), region.mask("ours", (16, 16))
        )

    def test_an_exclusion_cuts_the_mask(self):
        region = LookRegion(
            "box",
            "area",
            {"rect": [0, 0, 1, 1]},
            exclude={"ours": [{"rect": [0, 0, 0.5, 1]}]},
        )
        mask = region.mask("ours", (4, 2))
        self.assertFalse(mask[:, :2].any())
        self.assertTrue(mask[:, 2:].all())
        self.assertTrue(region.mask("ref", (4, 2)).all())  # the exclusion is per side

    def test_a_line_samples_its_profile(self):
        img = np.tile(
            np.linspace(0, 1, 11, dtype=np.float32), (5, 1)
        )  # brightens left to right
        region = LookRegion("ramp", "line", {"line": [[0.0, 0.5], [1.0, 0.5]]})
        profile = region.sample_line(img, "ours", n=5)
        self.assertEqual(profile.shape, (5,))
        self.assertTrue(np.all(np.diff(profile) > 0))

    def test_a_quad_rectifies_and_reports_its_kept_cells(self):
        img = np.zeros((10, 10), np.float32)
        img[:, 5:] = 1.0
        region = LookRegion(
            "board",
            "quad",
            {"quad": [[0, 0], [1, 0], [1, 1], [0, 1]], "grid": [4, 2]},
            exclude={"ours": [{"rect": [0, 0, 0.25, 1]}]},
        )
        grid, keep = region.rectify(img, "ours")
        self.assertEqual(grid.shape, (2, 4))
        np.testing.assert_allclose(grid[:, 3], 1.0, atol=1e-6)
        self.assertTrue(
            (grid[:, 1] < 0.1).all()
        )  # the dark half (bilinear blends at the edge)
        self.assertFalse(keep[:, 0].any())
        self.assertTrue(keep[:, 1:].all())

    def test_a_registered_shape_is_a_region_like_the_built_ins(self):
        """The seam: a new shape is a subclass plus one registration -- its
        pixels, outline and label anchor reach the region (and the sheet)."""

        class Band(RegionShape):
            key = "band"

            def mask(self, region, spec, size):
                y0, y1 = spec["band"]
                out = np.zeros((size[1], size[0]), bool)
                out[int(y0 * size[1]) : int(y1 * size[1])] = True
                return out

            def outline(self, spec):
                y0, y1 = spec["band"]
                return [(0, y0), (1, y0), (1, y1), (0, y1)], True

            def anchor(self, spec):
                return 0.0, spec["band"][0]

        LookRegion.register_shape(Band())
        self.addCleanup(LookRegion.SHAPES.pop, "band")
        region = LookRegion("floor", "area", {"band": [0.5, 1.0]})
        self.assertEqual(int(region.mask("ours", (8, 4)).sum()), 16)
        self.assertEqual(region.anchor("ours"), (0.0, 0.5))
        self.assertEqual(region.outline("ours")[0][2], (1, 1.0))
        with self.assertRaises(ValueError):  # an area shape is no line
            LookRegion("x", "line", {"band": [0, 1]})
        with self.assertRaises(ValueError):  # one shape per side
            LookRegion("x", "area", {"band": [0, 1], "rect": [0, 0, 1, 1]})

    def test_a_quad_with_a_grid_per_side_is_refused(self):
        """Its columns and rows are compared cell for cell: two grids met only
        at scoring, as a broadcast error."""
        corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
        with self.assertRaises(ValueError) as ctx:
            LookRegion(
                "board",
                "quad",
                {"quad": corners, "grid": [8, 4]},
                ref={"quad": corners, "grid": [4, 4]},
            )
        self.assertIn("grid", str(ctx.exception))
        with self.assertRaises(ValueError):  # a grid left out is the default one
            LookRegion("board", "quad", {"quad": corners, "grid": [8, 4]}, ref={"quad": corners})
        LookRegion(  # the default, named on one side only: one grid
            "board", "quad", {"quad": corners, "grid": list(LookRegion.GRID)}, ref={"quad": corners}
        )

    def test_the_window_bounds_the_shape(self):
        region = LookRegion("box", "area", {"rect": [0.25, 0.5, 0.75, 1.0]})
        self.assertEqual(region.window("ours", (8, 4)), (2, 2, 6, 4))

    def test_an_unknown_kind_or_shape_is_refused(self):
        with self.assertRaises(ValueError):
            LookRegion("x", "blob", {"rect": [0, 0, 1, 1]})
        with self.assertRaises(ValueError):
            LookRegion("x", "area", {"star": [0, 0, 1]})

    def test_round_trips_through_a_dict(self):
        region = LookRegion(
            "rim",
            "area",
            {"rect": [0.1, 0.2, 0.3, 0.4]},
            ref={"rect": [0.1, 0.25, 0.3, 0.45]},
            measures=["mean", "peak"],
            weight=2.0,
            notes="the lit edge",
        )
        again = LookRegion.from_dict(json.loads(json.dumps(region.to_dict())))
        self.assertEqual(again.to_dict(), region.to_dict())


class TestLookTarget(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="look_target_")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        img = np.zeros((40, 60, 3), np.uint8)
        img[:, 30:] = 255
        self.ref = os.path.join(self.dir, "ref.png")
        Image.fromarray(img).save(self.ref)

    def test_the_reference_resamples_in_display_light(self):
        target = LookTarget(self.ref)
        small = target.reference_display((2, 1))
        self.assertEqual(small.shape, (1, 2, 3))
        np.testing.assert_allclose(small[0, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(small[0, 1], 1.0, atol=1e-6)
        mid = target.reference_display((1, 1))[0, 0, 0]
        self.assertAlmostEqual(
            float(mid), 0.7354, places=3
        )  # half the light, encoded: not 0.5

    def test_saves_the_reference_relative_to_the_target(self):
        target = LookTarget(
            self.ref,
            [LookRegion("box", "area", {"rect": [0, 0, 1, 1]})],
            focus="box",
            notes="n",
        )
        path = os.path.join(self.dir, "target.json")
        target.save(path)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["reference"], "ref.png")
        again = LookTarget.load(path)
        self.assertEqual(os.path.normcase(again.reference), os.path.normcase(self.ref))
        self.assertEqual([r.name for r in again.regions], ["box"])
        self.assertEqual(again.focus, "box")

    def _markers(self):
        """A painted marker layer: a red stroke on the left, a blue one on the
        right, half-transparent, soft at the edges -- what a person paints."""
        layer = np.zeros((40, 60, 4), np.uint8)
        layer[5:35, 5:25] = (230, 20, 20, 128)
        layer[5:35, 35:55] = (20, 40, 230, 128)
        layer[5:35, 25] = (230, 20, 20, 40)  # a faint edge still counts
        path = os.path.join(self.dir, "markers.png")
        Image.fromarray(layer, "RGBA").save(path)
        return path

    def test_a_quad_cuts_out_a_painted_marker(self):
        """An exclusion is any area shape: a quad's kept cells were drawn from
        rects, ellipses and polygons only, so a painted cut-out raised KeyError."""
        self._markers()
        target = LookTarget(
            self.ref,
            [
                LookRegion(
                    "board",
                    "quad",
                    {"quad": [[0, 0], [1, 0], [1, 1], [0, 1]], "grid": [6, 4]},
                    exclude={"ours": [{"mask": "markers.png", "color": [230, 20, 20]}]},
                )
            ],
            base=self.dir,
        )
        _grid, keep = target.region("board").rectify(
            np.zeros((40, 60), np.float32), "ours"
        )
        self.assertFalse(keep[1, 1])  # under the red stroke
        self.assertTrue(keep[:, 4:].all())

    def test_a_painted_marker_is_a_region(self):
        self._markers()
        target = LookTarget(
            self.ref,
            [
                LookRegion(
                    "left", "area", {"mask": "markers.png", "color": [230, 20, 20]}
                ),
                LookRegion(
                    "right", "area", {"mask": "markers.png", "color": [20, 40, 230]}
                ),
            ],
            base=self.dir,
        )
        left = target.region("left").mask("ref", (60, 40))
        right = target.region("right").mask("ref", (60, 40))
        self.assertEqual(int(left.sum()), 30 * 21)
        self.assertEqual(int(right.sum()), 30 * 20)
        self.assertFalse((left & right).any())
        self.assertFalse(left[0].any(), "unpainted is no region")
        small = target.region("right").mask("ref", (30, 20))  # coverage, resampled
        self.assertTrue(small[10, 22] and not small[10, 5] and not small[0, 22])

    def test_an_exclusion_on_one_side_leaves_a_shared_painted_mask_whole(self):
        """A painted mask is the target's, shared by every side that paints it
        (the reference defaults to ours): a cut made in place on one side cut
        the other side, and every later read of the marker."""
        self._markers()
        target = LookTarget(
            self.ref,
            [
                LookRegion(
                    "left",
                    "area",
                    {"mask": "markers.png", "color": [230, 20, 20]},
                    exclude={"ref": [{"rect": [0, 0, 1, 0.5]}]},
                )
            ],
            base=self.dir,
        )
        region = target.region("left")
        ours = region.mask("ours", (60, 40)).copy()
        ref = region.mask("ref", (60, 40))
        self.assertFalse(ref[:20].any())  # the cut, on its side
        self.assertTrue(ref[20:35, 5:25].all())
        np.testing.assert_array_equal(region.mask("ours", (60, 40)), ours)
        self.assertEqual(int(ours.sum()), 30 * 21)
        np.testing.assert_array_equal(
            target.painted("markers.png", [230, 20, 20], (60, 40)), ours
        )

    def test_regions_sharing_a_marker_read_it_once(self):
        """One nearest-colour assignment per marker file, whatever the regions
        painting it: each colour rebuilt the whole palette's distances."""
        from unittest import mock

        from pythontk.core_utils.engines.lighting.look_match import target as module

        self._markers()
        target = LookTarget(
            self.ref,
            [
                LookRegion("left", "area", {"mask": "markers.png", "color": [230, 20, 20]}),
                LookRegion("right", "area", {"mask": "markers.png", "color": [20, 40, 230]}),
            ],
            base=self.dir,
        )
        real_open = module.Image.open
        with mock.patch.object(module.Image, "open", side_effect=real_open) as opened:
            left = target.region("left").mask("ours", (60, 40))
            right = target.region("right").mask("ours", (60, 40))
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(int(left.sum()), 30 * 21)
        self.assertEqual(int(right.sum()), 30 * 20)

    def test_objects_are_a_region_on_our_side(self):
        ids = np.zeros((40, 60, 3), np.uint8)
        ids[:, :20] = (10, 0, 0)
        ids[:, 20:40] = (20, 0, 0)
        ids[:, 40:] = (30, 0, 0)
        Image.fromarray(ids).save(os.path.join(self.dir, "ids.png"))
        legend = [
            {
                "color": [10, 0, 0],
                "names": ["TABLE", "SOLDER_PASS_TRAY", "RIM"],
                "material": "MAT_TRAY",
            },
            {
                "color": [20, 0, 0],
                "names": ["TABLE", "SOLDER_FAIL_TRAY"],
                "material": "MAT_TRAY",
            },
            {"color": [30, 0, 0], "names": ["TABLE", "MAT"], "material": "MAT_MAT"},
        ]
        trays = LookRegion(
            "trays",
            "area",
            {"objects": ["^SOLDER_(PASS|FAIL)_TRAY$", "!^RIM$"]},
            ref={"rect": [0, 0, 1, 1]},
        )
        target = LookTarget(
            self.ref, [trays], ids={"image": "ids.png", "legend": legend}, base=self.dir
        )
        mask = target.region("trays").mask("ours", (60, 40))
        self.assertTrue(mask[:, 20:40].all())
        self.assertFalse(mask[:, :20].any(), "the rim is kept out")
        self.assertFalse(mask[:, 40:].any())
        path = os.path.join(self.dir, "target.json")
        target.save(path)
        again = LookTarget.load(path)
        np.testing.assert_array_equal(
            again.region("trays").mask("ours", (60, 40)), mask
        )

    def test_a_painted_shape_without_a_target_is_named(self):
        region = LookRegion("x", "area", {"mask": "m.png", "color": [1, 2, 3]})
        with self.assertRaises(ValueError):
            region.mask("ours", (10, 10))

    def test_a_paint_template_starts_from_the_regions_and_reads_the_same(self):
        target = LookTarget(
            self.ref,
            [
                LookRegion("box", "area", {"rect": [0.1, 0.1, 0.4, 0.5]}),
                LookRegion("disc", "area", {"ellipse": [0.7, 0.5, 0.15, 0.2]}),
                LookRegion("band", "line", {"line": [[0.1, 0.9], [0.9, 0.9]]}),
            ],
        )
        before = {
            n: target.region(n).mask("ref", (60, 40)).copy() for n in ("box", "disc")
        }
        target.reference_display()  # its cache holds the reference by size: kept
        target.reference_display((30, 20))
        palette = target.paint_template(self.dir, side="ref")
        self.assertEqual(sorted(palette), ["box", "disc"])  # area regions only
        self.assertNotEqual(palette["box"], palette["disc"])
        for name in ("box", "disc"):
            shape = target.region(name).shapes["ref"]
            self.assertEqual(shape, {"mask": "ref_markers.png", "color": palette[name]})
            np.testing.assert_array_equal(
                target.region(name).mask("ref", (60, 40)), before[name]
            )
        self.assertIn("line", target.region("band").shapes["ref"])
        self.assertTrue(os.path.isfile(os.path.join(self.dir, "ref_markers_guide.png")))
        target.save(os.path.join(self.dir, "target.json"))
        again = LookTarget.load(os.path.join(self.dir, "target.json"))
        np.testing.assert_array_equal(
            again.region("box").mask("ref", (60, 40)), before["box"]
        )

    def test_overlapping_regions_are_refused_on_one_layer(self):
        target = LookTarget(
            self.ref,
            [
                LookRegion("tray", "area", {"rect": [0.1, 0.1, 0.6, 0.6]}),
                LookRegion("rim", "area", {"rect": [0.1, 0.5, 0.6, 0.6]}),
            ],
        )
        with self.assertRaises(ValueError) as ctx:
            target.paint_template(self.dir)
        self.assertIn("rim", str(ctx.exception))
        target.paint_template(self.dir, regions=["tray"])
        target.paint_template(self.dir, regions=["rim"], layer="ref_rims")
        self.assertEqual(target.region("rim").shapes["ref"]["mask"], "ref_rims.png")
        self.assertEqual(target.region("tray").shapes["ref"]["mask"], "ref_markers.png")

    def test_region_names_are_unique(self):
        box = LookRegion("box", "area", {"rect": [0, 0, 1, 1]})
        with self.assertRaises(ValueError):
            LookTarget(self.ref, [box, box])

    def test_a_region_is_found_by_name(self):
        target = LookTarget(
            self.ref, [LookRegion("box", "area", {"rect": [0, 0, 1, 1]})]
        )
        self.assertEqual(target.region("box").name, "box")
        with self.assertRaises(KeyError):
            target.region("nope")

    def test_regions_are_added_renamed_and_removed_by_name(self):
        """What a region editor does, the names kept unique and the focus followed."""
        target = LookTarget(self.ref, [LookRegion("box", "area", {"rect": [0, 0, 0.5, 1]})], focus="box")
        added = target.add(LookRegion("disc", "area", {"ellipse": [0.7, 0.5, 0.2, 0.3]}))
        self.assertIs(added.target, target)  # a painted or object shape reads its target
        with self.assertRaises(ValueError):
            target.add(LookRegion("disc", "area", {"rect": [0, 0, 1, 1]}))
        target.rename("box", "left")
        self.assertEqual([r.name for r in target.regions], ["left", "disc"])
        self.assertEqual(target.focus, "left")
        with self.assertRaises(ValueError):
            target.rename("disc", "left")
        target.remove("left")
        self.assertIsNone(target.focus)
        with self.assertRaises(KeyError):
            target.remove("left")
        path = target.save(os.path.join(self.dir, "target.json"))
        self.assertEqual([r.name for r in LookTarget.load(path).regions], ["disc"])


class TestDrawnShapes(unittest.TestCase):
    """A shape drawn in image pixels becomes its spec: the inverse of its outline."""

    SIZE = (200, 100)

    def test_each_drawable_shape_reads_its_points(self):
        spec = LookRegion.spec_from_points
        self.assertEqual(spec("rect", [(150, 80), (50, 20)], self.SIZE), {"rect": [0.25, 0.2, 0.75, 0.8]})
        self.assertEqual(spec("ellipse", [(50, 20), (150, 80)], self.SIZE), {"ellipse": [0.5, 0.5, 0.25, 0.3]})
        self.assertEqual(
            spec("poly", [(0, 0), (100, 0), (100, 50)], self.SIZE),
            {"poly": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]},
        )
        self.assertEqual(spec("line", [(0, 50), (200, 50)], self.SIZE), {"line": [[0.0, 0.5], [1.0, 0.5]]})
        drawn = LookRegion("drawn", "area", spec("rect", [(-20, 10), (100, 200)], self.SIZE))
        self.assertEqual(drawn.shapes["ours"]["rect"], [0.0, 0.1, 0.5, 1.0])  # clamped to the frame

    def test_a_shape_that_cannot_be_drawn_or_too_few_points_is_refused(self):
        for key, points in (("mask", [(0, 0), (1, 1)]), ("poly", [(0, 0), (1, 1)]), ("rect", [(5, 5), (5, 40)]), ("nope", [])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                LookRegion.spec_from_points(key, points, self.SIZE)
        self.assertEqual(sorted(LookRegion.drawable()), ["ellipse", "line", "poly", "rect"])


if __name__ == "__main__":
    unittest.main(exit=False)
