# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.geo_utils.uv.combine (UvCombine -- texture-set atlas plan).

Clustering and planning are pure numpy; the pack tests need the optional
xatlas engine and skip cleanly without it.
"""

import unittest

import pythontk as ptk

try:
    import numpy as np
except ImportError:
    np = None

_XATLAS = ptk.UvPack.available()


def _quad(size=1.0, offset=(0.0, 0.0)):
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64) * size
    return uvs + np.asarray(offset, dtype=np.float64), np.array([[0, 1, 2], [0, 2, 3]])


def _shell(key, group, size=0.2, offset=(0.0, 0.0), area_3d=None, scale=1.0):
    uvs, tris = _quad(size, offset)
    return ptk.CombineShell(
        key=key,
        group=group,
        uvs=uvs,
        triangles=tris,
        area_3d=size * size if area_3d is None else area_3d,
        scale=scale,
    )


@unittest.skipUnless(np is not None, "Requires numpy")
class TestClusters(unittest.TestCase):
    def test_stacked_copies_group(self):
        shells = [
            _shell("a", "m", offset=(0.1, 0.1)),
            _shell("b", "m", offset=(0.1, 0.1)),  # exact stack
            _shell("c", "m", offset=(0.6, 0.6)),
        ]
        self.assertEqual(ptk.UvCombine.clusters(shells), [[0, 1], [2]])

    def test_partial_overlap_groups_transitively(self):
        shells = [
            _shell("a", "m", offset=(0.0, 0.0)),
            _shell("b", "m", offset=(0.15, 0.0)),  # overlaps a
            _shell("c", "m", offset=(0.3, 0.0)),  # overlaps b, not a
        ]
        self.assertEqual(ptk.UvCombine.clusters(shells), [[0, 1, 2]])

    def test_touching_edges_do_not_group(self):
        shells = [_shell("a", "m"), _shell("b", "m", offset=(0.2, 0.0))]
        self.assertEqual(ptk.UvCombine.clusters(shells), [[0], [1]])

    def test_sets_never_group(self):
        """Overlap across two materials is two maps, not shared texels."""
        shells = [_shell("a", "m1"), _shell("b", "m2")]
        self.assertEqual(ptk.UvCombine.clusters(shells), [[0], [1]])


@unittest.skipUnless(np is not None, "Requires numpy")
class TestIslandShells(unittest.TestCase):
    """One mesh's arrays -> one shell per UV island (the hosts' gather)."""

    @staticmethod
    def _mesh():
        # Island 0: faces 0-1 (a quad), island 1: faces 2-3, island 2: face 4.
        quad = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        tri = np.array([[5, 5], [6, 5], [5, 6]], dtype=np.float64)
        uvs = np.vstack([quad, quad + [2, 0], tri])
        shell_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        tri_uv = np.array(
            [[0, 1, 2], [4, 5, 6], [0, 2, 3], [4, 6, 7], [8, 9, 10]], dtype=np.int64
        )
        tri_face = np.array([0, 2, 1, 3, 4])
        tri_area = np.array([1.0, 1.0, 1.0, 3.0, 0.5])
        # Island 1: face 2 wears "a" (area 1), face 3 wears "b" (area 3).
        face_material = np.array([0, 0, 0, 1, -1])
        return uvs, shell_ids, tri_uv, tri_face, tri_area, ["a", "b"], face_material

    def test_one_shell_per_island_voting_the_material_by_area(self):
        uvs, shell_ids, tri_uv, tri_face, tri_area, mats, fm = self._mesh()
        shells, faces, points = ptk.UvCombine.island_shells(
            "m", uvs, shell_ids, tri_uv, tri_face, tri_area, mats, fm
        )
        self.assertEqual([s.key for s in shells], [("m", 0), ("m", 1), ("m", 2)])
        self.assertEqual([s.group for s in shells], ["a", "b", None])
        self.assertEqual([f.tolist() for f in faces], [[0, 1], [2, 3], [4]])
        self.assertEqual(
            [p.tolist() for p in points], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
        )
        self.assertEqual([s.area_3d for s in shells], [2.0, 4.0, 0.5])
        for shell, pts in zip(shells, points):
            np.testing.assert_array_equal(shell.uvs, uvs[pts])

    def test_matches_a_per_island_mask_on_a_shuffled_mesh(self):
        """The sort-based grouping reads exactly what one mask per island read."""
        rng = np.random.default_rng(7)
        n_islands, per = 40, 6
        uvs = rng.random((n_islands * per, 2))
        shell_ids = np.repeat(np.arange(n_islands), per)
        tri_uv = np.concatenate(
            [
                i * per + np.array([[0, 1, 2], [2, 3, 4], [3, 4, 5]])
                for i in range(n_islands)
            ]
        )
        order = rng.permutation(len(tri_uv))
        tri_uv = tri_uv[order]
        tri_face = np.arange(len(tri_uv))[order] // 2
        tri_area = rng.random(len(tri_uv))
        mats = ["a", "b", "c"]
        fm = rng.integers(-1, 3, size=int(tri_face.max()) + 1)
        shells, faces, _ = ptk.UvCombine.island_shells(
            7, uvs, shell_ids, tri_uv, tri_face, tri_area, mats, fm
        )
        tri_shell = shell_ids[tri_uv[:, 0]]
        for shell, covered in zip(shells, faces):
            pick = tri_shell == shell.key[1]
            used = np.unique(tri_uv[pick])
            np.testing.assert_array_equal(shell.uvs[shell.triangles], uvs[tri_uv[pick]])
            np.testing.assert_array_equal(shell.uvs, uvs[used])
            np.testing.assert_array_equal(covered, np.unique(tri_face[pick]))
            worn = fm[tri_face[pick]]
            shaded = worn >= 0
            votes = np.bincount(worn[shaded], weights=tri_area[pick][shaded])
            expected = mats[int(np.argmax(votes))] if shaded.any() else None
            self.assertEqual(shell.group, expected)
        self.assertEqual(len(shells), n_islands)

    def test_a_mesh_with_no_triangles_has_no_shells(self):
        empty = np.zeros((0, 3), dtype=np.int64)
        self.assertEqual(
            ptk.UvCombine.island_shells("m", [], [], empty, [], [], [], []),
            ([], [], []),
        )


@unittest.skipUnless(np is not None, "Requires numpy")
class TestPlan(unittest.TestCase):
    def test_density_equalizes_world_texel_density(self):
        # Same UV size, the second has 4x the surface: twice the linear scale.
        shells = [
            _shell("a", "m1", area_3d=1.0),
            _shell("b", "m2", offset=(0.5, 0.5), area_3d=4.0),
        ]
        plan = ptk.UvCombine.plan(shells, sizing="density")
        self.assertAlmostEqual(plan.scales[1] / plan.scales[0], 2.0)

    def test_proportional_keeps_a_sets_internal_ratios(self):
        # m1's two islands: same surface, different UV sizes (an artist's
        # texel-density choice); proportional keeps their 2:1 UV ratio.
        shells = [
            _shell("a", "m1", size=0.4, area_3d=1.0),
            _shell("b", "m1", size=0.2, offset=(0.5, 0.5), area_3d=1.0),
            _shell("c", "m2", size=0.3, area_3d=1.0),
        ]
        plan = ptk.UvCombine.plan(shells, sizing="proportional")
        self.assertAlmostEqual(plan.scales[0], plan.scales[1])
        density = ptk.UvCombine.plan(shells, sizing="density")
        self.assertNotAlmostEqual(density.scales[0], density.scales[1])

    def test_resolution_normalize_follows_map_size(self):
        shells = [_shell("a", "4k"), _shell("b", "2k", offset=(0.5, 0.5))]
        plan = ptk.UvCombine.plan(
            shells,
            sizing="proportional",
            normalize="resolution",
            resolutions={"4k": 4096, "2k": 2048},
        )
        self.assertAlmostEqual(plan.scales[0] / plan.scales[1], 2.0)

    def test_override_multiplies_and_shares_sum_to_one(self):
        shells = [
            _shell("a", "m1"),
            _shell("b", "m2", offset=(0.5, 0.5), scale=1.0 / 3.0),
        ]
        plan = ptk.UvCombine.plan(shells)
        self.assertAlmostEqual(plan.scales[1] / plan.scales[0], 1.0 / 3.0)
        self.assertAlmostEqual(sum(g.share for g in plan.groups.values()), 1.0)
        self.assertLess(plan.groups["m2"].share, plan.groups["m1"].share)

    def test_stack_footprint_counts_once(self):
        stacked = [_shell("a", "m"), _shell("b", "m")]
        single = [_shell("a", "m")]
        a = ptk.UvCombine.plan(stacked).groups["m"]
        b = ptk.UvCombine.plan(single).groups["m"]
        self.assertAlmostEqual(a.coverage, b.coverage)
        self.assertEqual((a.shells, a.stacks), (2, 1))

    def test_bad_mode_raises(self):
        with self.assertRaises(ValueError):
            ptk.UvCombine.plan([_shell("a", "m")], sizing="nope")


@unittest.skipUnless(_XATLAS and np is not None, "Requires xatlas + numpy")
class TestPack(unittest.TestCase):
    def _boxes(self, layout, indices):
        return [(layout.uvs[i].min(0), layout.uvs[i].max(0)) for i in indices]

    def test_stacks_stay_stacked_and_groups_do_not_overlap(self):
        shells = [
            _shell("a", "m1", offset=(0.1, 0.1)),
            _shell("b", "m1", offset=(0.1, 0.1)),
            _shell("c", "m2", offset=(0.1, 0.1)),
            _shell("d", "m3", size=0.5, offset=(0.2, 0.2)),
        ]
        layout = ptk.UvCombine.pack(shells, map_size=512, padding=4)
        np.testing.assert_allclose(layout.uvs[0], layout.uvs[1], atol=1e-9)
        boxes = self._boxes(layout, [0, 2, 3])
        for i, (lo_a, hi_a) in enumerate(boxes):
            self.assertTrue((lo_a >= -1e-6).all() and (hi_a <= 1 + 1e-6).all())
            for lo_b, hi_b in boxes[i + 1 :]:
                apart = (hi_a <= lo_b + 1e-6).any() or (hi_b <= lo_a + 1e-6).any()
                self.assertTrue(apart)

    def test_planned_proportions_come_out_exactly(self):
        shells = [
            _shell("a", "m1", size=0.4, area_3d=1.0),
            _shell("b", "m1", size=0.2, offset=(0.5, 0.5), area_3d=1.0),
            _shell("c", "m2", size=0.3, area_3d=1.0, scale=0.5),
        ]
        plan = ptk.UvCombine.plan(shells, sizing="proportional")
        layout = ptk.UvCombine.pack(shells, plan, map_size=512, padding=2)
        # The engine keeps relative scale to sub-texel chart rounding
        # (measured ~0.15% on 512 px pages), not to float precision.
        scale = [t[0] for t in layout.transforms]
        self.assertAlmostEqual(scale[0] / scale[1], 1.0, delta=0.01)
        expected = plan.scales[0] / plan.scales[2]
        self.assertAlmostEqual(scale[0] / scale[2] / expected, 1.0, delta=0.01)
        self.assertAlmostEqual(
            sum(g.share for g in layout.plan.groups.values()), 1.0, places=6
        )
        self.assertGreater(layout.utilization, 0.0)

    def test_partial_overlap_cluster_packs_as_hull(self):
        shells = [
            _shell("a", "m", offset=(0.0, 0.0)),
            _shell("b", "m", offset=(0.1, 0.1)),
            _shell("c", "n", offset=(0.5, 0.5)),
        ]
        layout = ptk.UvCombine.pack(shells, map_size=256, padding=2)
        # One rigid move for the overlapping pair: their offset is preserved.
        delta = layout.uvs[1][0] - layout.uvs[0][0]
        self.assertAlmostEqual(
            float(np.hypot(*delta)),
            layout.transforms[0][0] * float(np.hypot(0.1, 0.1)),
            places=6,
        )


class TestModuleInvariant(unittest.TestCase):
    """Helpers live on classes, not at module scope (package standard)."""

    def test_no_top_level_functions(self):
        import ast
        import inspect

        from pythontk.geo_utils.uv import combine

        tree = ast.parse(inspect.getsource(combine))
        offenders = [
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.assertEqual(offenders, [], f"module-level def(s): {offenders}")


if __name__ == "__main__":
    unittest.main()
