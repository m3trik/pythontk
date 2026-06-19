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


if __name__ == "__main__":
    unittest.main()
