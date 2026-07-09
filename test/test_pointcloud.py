# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.geo_utils.pointcloud (PointCloud) — unordered point-set
geometry: PCA alignment, proximity clustering, positional hashing. Relocated
from test_math.py when the geometry cluster moved out of MathUtils.
"""

import unittest

from pythontk.geo_utils.pointcloud import PointCloud


class TestPcaTransform(unittest.TestCase):
    def test_pca_transform_identity(self):
        """Identical points -> a valid 4x4 (16-element) matrix."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        try:
            result = PointCloud.pca_transform(pts, pts, tolerance=0.001)
        except (ImportError, ValueError) as e:
            if "binary incompatibility" in str(e) or "numpy" in str(e).lower():
                self.skipTest(f"scipy/numpy incompatibility: {e}")
            raise
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 16)  # 4x4 matrix

    def test_pca_transform_translated(self):
        """Finds a translation."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        pts_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        pts_b = pts_a + np.array([10, 20, 30])  # Translated copy
        try:
            result = PointCloud.pca_transform(pts_a, pts_b, tolerance=0.1)
        except (ImportError, ValueError) as e:
            if "binary incompatibility" in str(e) or "numpy" in str(e).lower():
                self.skipTest(f"scipy/numpy incompatibility: {e}")
            raise
        self.assertIsNotNone(result)

    def test_pca_transform_rotated(self):
        """Finds a rotation alignment."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        # Create a shape
        pts_a = np.array(
            [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [1, 0.5, 0.5]], dtype=float
        )

        # Rotate 90 degrees around Z axis
        theta = np.pi / 2
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        pts_b = pts_a @ R.T

        try:
            result = PointCloud.pca_transform(pts_a, pts_b, tolerance=0.1)
        except (ImportError, ValueError) as e:
            if "binary incompatibility" in str(e) or "numpy" in str(e).lower():
                self.skipTest(f"scipy/numpy incompatibility: {e}")
            raise
        self.assertIsNotNone(result)

    def test_pca_transform_insufficient_points(self):
        """Too few points -> None."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        try:
            result = PointCloud.pca_transform(pts, pts, tolerance=0.1)
        except (ImportError, ValueError) as e:
            if "binary incompatibility" in str(e) or "numpy" in str(e).lower():
                self.skipTest(f"scipy/numpy incompatibility: {e}")
            raise
        self.assertIsNone(result)

    def test_pca_transform_no_match(self):
        """None when shapes don't match."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        pts_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        pts_b = np.array(
            [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float
        )  # Different scale
        try:
            result = PointCloud.pca_transform(pts_a, pts_b, tolerance=0.001)
        except (ImportError, ValueError) as e:
            if "binary incompatibility" in str(e) or "numpy" in str(e).lower():
                self.skipTest(f"scipy/numpy incompatibility: {e}")
            raise
        self.assertIsNone(result)

    def test_pca_transform_robust_mode(self):
        """Robust mode handles different point counts."""
        try:
            import numpy as np
            from scipy.spatial import KDTree  # noqa: F401
        except (ImportError, ValueError) as e:
            self.skipTest(f"numpy or scipy not available: {e}")

        np.random.seed(42)
        pts_a = np.random.rand(100, 3)
        pts_b = np.random.rand(80, 3)  # Different count

        try:
            # Robust mode should handle different point counts
            result = PointCloud.pca_transform(
                pts_a, pts_b, tolerance=10.0, robust=True
            )
        except (ImportError, ValueError) as e:
            if "binary incompatibility" in str(e) or "numpy" in str(e).lower():
                self.skipTest(f"scipy/numpy incompatibility: {e}")
            raise
        # May or may not find a match depending on data, but shouldn't crash
        self.assertTrue(result is None or len(result) == 16)

    def test_pca_transform_caching(self):
        """Caches base rotations on the class."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

        try:
            # First call creates cache
            PointCloud.pca_transform(pts, pts, tolerance=0.1)
        except (ImportError, ValueError) as e:
            if "binary incompatibility" in str(e) or "numpy" in str(e).lower():
                self.skipTest(f"scipy/numpy incompatibility: {e}")
            raise

        # Verify cache exists
        self.assertTrue(hasattr(PointCloud, "_pca_base_rotations"))
        self.assertEqual(len(PointCloud._pca_base_rotations), 24)


class TestClusterByDistance(unittest.TestCase):
    @staticmethod
    def _normalize_clusters(clusters):
        """Order-independent comparison form: a set of frozensets of indices."""
        return {frozenset(c) for c in clusters}

    def test_cluster_by_distance_two_groups(self):
        """Two tight groups separated by a large gap -> two clusters."""
        points = [(0, 0, 0), (1, 0, 0), (50, 0, 0), (51, 0, 0)]
        clusters = PointCloud.cluster_by_distance(points, threshold=5)
        self.assertEqual(
            self._normalize_clusters(clusters),
            {frozenset({0, 1}), frozenset({2, 3})},
        )

    def test_cluster_by_distance_transitive_chain(self):
        """Single-link: a chain of near-neighbours forms ONE cluster even though its ends
        are farther apart than the threshold."""
        points = [(0, 0, 0), (4, 0, 0), (8, 0, 0), (12, 0, 0)]
        clusters = PointCloud.cluster_by_distance(points, threshold=5)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(set(clusters[0]), {0, 1, 2, 3})

    def test_cluster_by_distance_threshold_is_inclusive(self):
        """Points exactly ``threshold`` apart link (<= comparison)."""
        points = [(0, 0, 0), (10, 0, 0)]
        self.assertEqual(len(PointCloud.cluster_by_distance(points, threshold=10)), 1)
        self.assertEqual(len(PointCloud.cluster_by_distance(points, threshold=9.9)), 2)

    def test_cluster_by_distance_matches_naive(self):
        """The spatial-hash result equals the naive O(N^2) single-link grouping on a
        scattered set (the grid is an optimization, not a behavior change)."""
        import random

        rng = random.Random(7)
        points = [
            (rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(0, 100))
            for _ in range(60)
        ]
        threshold = 12.0

        # Naive union-find single-link reference.
        parent = list(range(len(points)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        t_sq = threshold * threshold
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = sum((points[i][k] - points[j][k]) ** 2 for k in range(3))
                if d <= t_sq:
                    parent[find(i)] = find(j)
        naive = {}
        for i in range(len(points)):
            naive.setdefault(find(i), []).append(i)

        self.assertEqual(
            self._normalize_clusters(PointCloud.cluster_by_distance(points, threshold)),
            self._normalize_clusters(naive.values()),
        )

    def test_cluster_by_distance_edge_cases(self):
        """Empty -> []; single point -> one singleton cluster; 2D points supported."""
        self.assertEqual(PointCloud.cluster_by_distance([], threshold=5), [])
        self.assertEqual(PointCloud.cluster_by_distance([(1, 2, 3)], threshold=5), [[0]])
        clusters = PointCloud.cluster_by_distance([(0, 0), (1, 0), (40, 40)], threshold=5)
        self.assertEqual(
            self._normalize_clusters(clusters), {frozenset({0, 1}), frozenset({2})}
        )


class TestHashPoints(unittest.TestCase):
    def test_flat_list_hashes_per_point(self):
        h = PointCloud.hash_points([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
        self.assertEqual(len(h), 2)
        self.assertNotEqual(h[0], h[1])

    def test_position_stable_within_precision(self):
        # Two points equal to the retained precision hash identically.
        a = PointCloud.hash_points([(1.00001, 2.0, 3.0)], precision=4)
        b = PointCloud.hash_points([(1.00002, 2.0, 3.0)], precision=4)
        self.assertEqual(a, b)

    def test_nested_lists_preserve_structure(self):
        h = PointCloud.hash_points([[(1.0, 2.0, 3.0)], [(4.0, 5.0, 6.0)]])
        self.assertEqual(len(h), 2)
        self.assertEqual(len(h[0]), 1)


def _np():
    try:
        import numpy as np

        return np
    except ImportError:
        return None


def _asym_cloud(np, n=60, seed=7):
    """Deterministic asymmetric jittered cloud (no rotational symmetry)."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1.0, 1.0, size=(n, 3))
    pts[:, 0] *= 3.0  # stretch: distinct eigenvalues
    pts[:, 1] *= 1.7
    return pts


def _rot(np, axis, theta):
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    c, s, t = np.cos(theta), np.sin(theta), 1 - np.cos(theta)
    return np.array(
        [
            [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
            [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
            [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
        ]
    )


def _apply_row_matrix(np, pts, flat16):
    """Row-vector convention: p_h @ M."""
    m = np.array(flat16, dtype=float).reshape(4, 4)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    return (pts_h @ m)[:, :3]


class TestMatchClouds(unittest.TestCase):
    """The shared auto-instancer verification pipeline."""

    def setUp(self):
        self.np = _np()
        if self.np is None:
            self.skipTest("numpy not available")

    def test_identical_ordered(self):
        np = self.np
        pts = _asym_cloud(np)
        ok, m = PointCloud.match_clouds(pts, pts.copy())
        self.assertTrue(ok)
        self.assertIsNone(m)  # identity match carries no matrix

    def test_reordered_identity(self):
        np = self.np
        pts = _asym_cloud(np)
        rng = np.random.default_rng(1)
        perm = rng.permutation(len(pts))
        ok, m = PointCloud.match_clouds(pts, pts[perm])
        self.assertTrue(ok)
        self.assertIsNone(m)

    def test_count_mismatch_rejects(self):
        np = self.np
        pts = _asym_cloud(np)
        ok, m = PointCloud.match_clouds(pts, pts[:-1])
        self.assertFalse(ok)
        self.assertIsNone(m)

    def test_empty_clouds_are_identical(self):
        """Two zero-point clouds match as identity (stage-1 short-circuit)."""
        np = self.np
        empty = np.empty((0, 3))
        ok, m = PointCloud.match_clouds(empty, empty)
        self.assertTrue(ok)
        self.assertIsNone(m)

    def test_rotated_copy_matches_with_matrix(self):
        np = self.np
        pts_a = _asym_cloud(np)
        R = _rot(np, (0.3, 1.0, 0.2), 0.9)
        offset = np.array([5.0, -2.0, 3.0])
        pts_b = pts_a @ R.T + offset
        ok, m = PointCloud.match_clouds(pts_a, pts_b, tolerance=0.001)
        self.assertTrue(ok)
        self.assertIsNotNone(m)
        # The matrix maps cloud-a geometry onto cloud-b.
        mapped = _apply_row_matrix(np, pts_a, m)
        dists, _ = PointCloud.nn_query(pts_b, mapped, k=1)
        self.assertLess(float(dists.max()), 0.01)

    def test_uniform_scaled_copy(self):
        np = self.np
        pts_a = _asym_cloud(np)
        R = _rot(np, (0, 0, 1), 0.5)
        pts_b = (pts_a @ R.T) * 0.6 + np.array([10.0, 0.0, 0.0])
        # Strict mode rejects a scaled copy...
        ok_strict, _ = PointCloud.match_clouds(pts_a, pts_b, tolerance=0.001)
        self.assertFalse(ok_strict)
        # ...scale mode matches and the matrix carries the true size.
        ok, m = PointCloud.match_clouds(
            pts_a, pts_b, tolerance=0.001, scale_tolerance=1.0
        )
        self.assertTrue(ok)
        mapped = _apply_row_matrix(np, pts_a, m)
        dists, _ = PointCloud.nn_query(pts_b, mapped, k=1)
        self.assertLess(float(dists.max()), 0.01)

    def test_uvs_identical_callback_rejects(self):
        np = self.np
        pts = _asym_cloud(np)
        ok, m = PointCloud.match_clouds(pts, pts.copy(), uvs_identical=lambda: False)
        self.assertFalse(ok)

    def test_half_flipped_normals_reject(self):
        np = self.np
        pts = _asym_cloud(np)
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(pts), 1))
        flipped = normals.copy()
        flipped[: len(pts) // 2] *= -1.0  # no rigid rotation aligns these
        ok, _ = PointCloud.match_clouds(
            pts, pts.copy(), normals_a=normals, normals_b=flipped
        )
        self.assertFalse(ok)

    def test_matching_normals_pass(self):
        np = self.np
        pts = _asym_cloud(np)
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(pts), 1))
        ok, _ = PointCloud.match_clouds(
            pts, pts.copy(), normals_a=normals, normals_b=normals.copy()
        )
        self.assertTrue(ok)

    def test_rotated_copy_matches_without_scipy(self):
        """The brute-force fallback must still solve a rotated copy."""
        import sys
        from unittest import mock

        np = self.np
        pts_a = _asym_cloud(np)
        R = _rot(np, (0, 0, 1), np.pi / 2)
        pts_b = pts_a @ R.T
        with mock.patch.dict(sys.modules, {"scipy": None, "scipy.spatial": None}):
            ok, m = PointCloud.match_clouds(pts_a, pts_b, tolerance=0.001)
        self.assertTrue(ok)
        self.assertIsNotNone(m)
        mapped = _apply_row_matrix(np, pts_a, m)
        dists, _ = PointCloud.nn_query(pts_b, mapped, k=1)
        self.assertLess(float(dists.max()), 0.01)


class TestNnQuery(unittest.TestCase):
    def setUp(self):
        self.np = _np()
        if self.np is None:
            self.skipTest("numpy not available")

    def test_fallback_matches_scipy(self):
        import sys
        from unittest import mock

        np = self.np
        target = _asym_cloud(np, n=40, seed=3)
        query = _asym_cloud(np, n=25, seed=4)
        d_scipy, i_scipy = PointCloud.nn_query(target, query, k=3)
        with mock.patch.dict(sys.modules, {"scipy": None, "scipy.spatial": None}):
            d_brute, i_brute = PointCloud.nn_query(target, query, k=3)
        self.assertTrue(np.allclose(d_scipy, d_brute, atol=1e-9))
        self.assertTrue((i_scipy == i_brute).all())

    def test_k1_shape(self):
        np = self.np
        target = _asym_cloud(np, n=10)
        d, i = PointCloud.nn_query(target, target, k=1)
        self.assertEqual(d.shape, (10, 1))
        self.assertEqual(i.shape, (10, 1))
        self.assertLess(float(d.max()), 1e-12)


class TestPcaBasis(unittest.TestCase):
    def setUp(self):
        self.np = _np()
        if self.np is None:
            self.skipTest("numpy not available")

    def test_too_few_points(self):
        self.assertIsNone(PointCloud.pca_basis([[0, 0, 0], [1, 1, 1]]))

    def test_canonicalization_consistency(self):
        """Copies canonicalize to the same local geometry under their bases."""
        np = self.np
        pts_a = _asym_cloud(np)
        R = _rot(np, (1.0, 0.4, -0.2), 1.1)
        pts_b = pts_a @ R.T + np.array([4.0, 4.0, -1.0])

        ba = np.array(PointCloud.pca_basis(pts_a)).reshape(4, 4)[:3, :3]
        bb = np.array(PointCloud.pca_basis(pts_b)).reshape(4, 4)[:3, :3]
        ca = pts_a - pts_a.mean(axis=0)
        cb = pts_b - pts_b.mean(axis=0)
        local_a = ca @ ba.T  # coords in the stabilized frame (rows = axes)
        local_b = cb @ bb.T
        self.assertTrue(np.allclose(local_a, local_b, atol=1e-6))


class TestPcaEigenvalueSignature(unittest.TestCase):
    def setUp(self):
        self.np = _np()
        if self.np is None:
            self.skipTest("numpy not available")

    def test_rotation_and_scale_invariant(self):
        np = self.np
        pts = _asym_cloud(np)
        R = _rot(np, (0.2, 0.9, 0.1), 0.7)
        rotated_scaled = (pts @ R.T) * 2.5 + np.array([1.0, 2.0, 3.0])
        self.assertEqual(
            PointCloud.pca_eigenvalue_signature(pts),
            PointCloud.pca_eigenvalue_signature(rotated_scaled),
        )

    def test_different_shapes_differ(self):
        np = self.np
        pts = _asym_cloud(np)
        squashed = pts * np.array([1.0, 1.0, 0.05])
        self.assertNotEqual(
            PointCloud.pca_eigenvalue_signature(pts),
            PointCloud.pca_eigenvalue_signature(squashed),
        )

    def test_too_few_points_empty(self):
        self.assertEqual(
            PointCloud.pca_eigenvalue_signature([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            (),
        )


if __name__ == "__main__":
    unittest.main()
