# !/usr/bin/python
# coding=utf-8
"""Point-cloud geometry — analyze and group unordered sets of points.

Operations on a *cloud* (an unordered set of points) rather than an ordered
polyline: shape alignment by principal axes, proximity clustering, positional
hashing. DCC-neutral pure Python (PCA alignment optionally accelerated by
numpy/scipy when available); the adapters supply world-space points. For ordered
point sequences see :class:`pythontk.geo_utils.polyline.Polyline`; for the scalar
primitives these compose, :class:`pythontk.MathUtils`.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, TYPE_CHECKING

from pythontk.core_utils._core_utils import CoreUtils
from pythontk.iter_utils._iter_utils import IterUtils

if TYPE_CHECKING:
    import numpy as np


class PointCloud:
    """Stateless point-cloud geometry (alignment / clustering / hashing)."""

    @staticmethod
    def pca_transform(
        points_a: "np.ndarray",
        points_b: "np.ndarray",
        tolerance: float = 0.001,
        robust: bool = False,
        sample_size: int = 500,
        symmetry_threshold: float = 0.1,
    ) -> Optional[List[float]]:
        """Transform that aligns ``points_b`` onto ``points_a`` via PCA axis alignment.

        Robust against vertex reordering — it aligns the principal axes of the two
        shapes, testing all 24 orthogonal alignments of the principal axes for the
        best fit.

        Parameters:
            points_a: (N, 3) array of points for the target shape.
            points_b: (N, 3) array of points for the source shape (to transform).
            tolerance: Maximum allowed average distance for a valid match.
            robust: If True, enables enhanced handling for:
                - Cylindrical symmetry (eigenvalue degeneracy)
                - Large point clouds (via sampling)
                - Arbitrary rotations (tests spin around symmetric axis)
                Note: robust=True requires scipy.spatial.KDTree.
            sample_size: Max points to use for KDTree queries when robust=True.
            symmetry_threshold: Relative eigenvalue difference to detect cylindrical
                symmetry. If two eigenvalues are within this ratio of each other,
                treat as symmetric.

        Returns:
            A 16-element list representing the 4x4 transformation matrix
            (row-major). Returns None if no alignment is found within tolerance or
            if dependencies are unavailable.

        Note:
            Requires numpy. scipy.spatial.KDTree is optional but improves
            performance. When robust=True, scipy.spatial.KDTree is required.
        """
        try:
            import numpy as np
            import itertools
        except ImportError:
            return None

        try:
            from scipy.spatial import KDTree
        except ImportError:
            KDTree = None
            if robust:
                return None  # robust mode requires KDTree

        pts_a = np.array(points_a)
        pts_b = np.array(points_b)

        if len(pts_a) < 3 or len(pts_b) < 3:
            return None
        if not robust and len(pts_a) != len(pts_b):
            return None

        # 1. Centroids
        c_a = np.mean(pts_a, axis=0)
        c_b = np.mean(pts_b, axis=0)

        # 2. Center points
        p_a = pts_a - c_a
        p_b = pts_b - c_b

        # 3. PCA (Eigenvectors)
        cov_a = np.cov(p_a, rowvar=False)
        cov_b = np.cov(p_b, rowvar=False)

        val_a, vec_a = np.linalg.eigh(cov_a)
        val_b, vec_b = np.linalg.eigh(cov_b)

        # Sort by eigenvalue (descending)
        idx_a = np.argsort(val_a)[::-1]
        idx_b = np.argsort(val_b)[::-1]

        val_a = val_a[idx_a]
        val_b = val_b[idx_b]
        vec_a = vec_a[:, idx_a]
        vec_b = vec_b[:, idx_b]

        # Ensure right-handed coordinate systems
        if np.linalg.det(vec_a) < 0:
            vec_a[:, 2] *= -1
        if np.linalg.det(vec_b) < 0:
            vec_b[:, 2] *= -1

        # 4. Generate all 24 base rotations (cached for performance)
        if not hasattr(PointCloud, "_pca_base_rotations"):
            axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
            rotations = []
            for p in itertools.permutations([0, 1, 2]):
                for sx in [-1, 1]:
                    for sy in [-1, 1]:
                        col0 = axes[p[0]] * sx
                        col1 = axes[p[1]] * sy
                        col2 = np.cross(col0, col1)
                        P = np.column_stack((col0, col1, col2))
                        rotations.append(P)
            PointCloud._pca_base_rotations = rotations
        base_rotations = PointCloud._pca_base_rotations

        # 5. Symmetry detection and spin angles (robust mode only)
        sym_axis_b = None
        spin_angles = [0.0]

        if robust:
            # Detect cylindrical symmetry
            def detect_symmetry(eigenvalues):
                e = eigenvalues
                max_e = max(abs(e[0]), abs(e[-1]))
                if max_e < 1e-9:
                    return None
                if abs(e[1] - e[2]) < symmetry_threshold * max_e:
                    return 0  # Symmetry around axis 0
                if abs(e[0] - e[1]) < symmetry_threshold * max_e:
                    return 2  # Symmetry around axis 2
                return None

            sym_axis_b = detect_symmetry(val_b)
            if sym_axis_b is not None:
                spin_angles = [
                    i * np.pi / 12 for i in range(24)
                ]  # 15-degree increments

            # Subsample for performance
            p_a_work = (
                p_a[np.random.choice(len(p_a), sample_size, replace=False)]
                if len(p_a) > sample_size
                else p_a
            )
            p_b_work = (
                p_b[np.random.choice(len(p_b), sample_size, replace=False)]
                if len(p_b) > sample_size
                else p_b
            )
        else:
            p_a_work = p_a
            p_b_work = p_b

        # 6. Build KDTree or prepare fallback
        tree = KDTree(p_a_work) if KDTree else None

        if not tree:
            a_sq = np.sum(p_a_work**2, axis=1)
            b_sq = np.sum(p_b_work**2, axis=1)

        def axis_angle_matrix(axis, angle):
            c = np.cos(angle)
            s = np.sin(angle)
            t = 1 - c
            x, y, z = axis / np.linalg.norm(axis)
            return np.array(
                [
                    [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                    [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                    [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
                ]
            )

        best_diff = float("inf")
        best_matrix = None

        for P in base_rotations:
            R_base = vec_a @ P @ vec_b.T

            # Generate spins to try
            if sym_axis_b is not None:
                spin_axis = vec_b[:, sym_axis_b]
                spins = [axis_angle_matrix(spin_axis, angle) for angle in spin_angles]
            else:
                spins = [np.eye(3)]

            for R_spin in spins:
                R = R_base @ R_spin
                p_b_rot = p_b_work @ R.T

                # Measure distance
                if tree:
                    dists, _ = tree.query(p_b_rot, k=1)
                    avg_dist = np.mean(dists)
                else:
                    dists_sq = (
                        b_sq[:, np.newaxis]
                        + a_sq[np.newaxis, :]
                        - 2 * np.dot(p_b_rot, p_a_work.T)
                    )
                    dists_sq = np.maximum(dists_sq, 0)
                    min_dists = np.sqrt(np.min(dists_sq, axis=1))
                    avg_dist = np.mean(min_dists)

                if avg_dist < best_diff:
                    best_diff = avg_dist
                    best_matrix = R
                    # Early exit on near-perfect match
                    if best_diff < tolerance * 0.01:
                        break
            if best_diff < tolerance * 0.01:
                break

        if best_diff > tolerance:
            return None

        # Construct 4x4 Matrix (Row-Major)
        R = best_matrix
        T = c_a - R @ c_b

        M = np.eye(4)
        M[:3, :3] = R.T
        M[3, :3] = T

        return M.flatten().tolist()

    @staticmethod
    def cluster_by_distance(
        points: Sequence[Sequence[float]], threshold: float
    ) -> List[List[int]]:
        """Group points into clusters linked by proximity (threshold flood-fill).

        Two points join the same cluster when they are within ``threshold`` of each
        other (Euclidean); clusters grow transitively (single-link), so a chain of
        near-neighbours forms one cluster even when its ends are far apart. A spatial
        hash grid sized to ``threshold`` keeps this ~O(N) — each point only compares
        against the 3**d surrounding cells — while giving the same result as the
        naive O(N^2) pairwise scan.

        Parameters:
            points: Sequence of equal-length coordinate tuples (any dimensionality).
            threshold: Maximum gap between two points for them to link.

        Returns:
            List of clusters, each a list of indices into ``points`` (clusters in
            first-seen order). An empty input yields ``[]``.

        Example:
            cluster_by_distance([(0, 0, 0), (1, 0, 0), (50, 0, 0)], threshold=5)
            # [[0, 1], [2]]
        """
        pts = [tuple(p) for p in points]
        n = len(pts)
        if n == 0:
            return []
        if n == 1:
            return [[0]]

        dims = len(pts[0])
        threshold_sq = threshold * threshold
        # Cell size == threshold guarantees any two points within threshold land in
        # the same or an adjacent cell, so the 3**d neighbourhood is exact.
        cell = threshold if threshold > 0 else 1.0

        def cell_of(p):
            return tuple(int(c // cell) for c in p)

        grid = {}
        for i, p in enumerate(pts):
            grid.setdefault(cell_of(p), []).append(i)

        # Offsets for the 3**d cells surrounding (and including) a point's own cell.
        offsets = [()]
        for _ in range(dims):
            offsets = [o + (d,) for o in offsets for d in (-1, 0, 1)]

        clusters = []
        processed = set()
        for i in range(n):
            if i in processed:
                continue
            cluster = [i]
            processed.add(i)
            queue = [i]
            while queue:
                cur = queue.pop()
                p1 = pts[cur]
                base = cell_of(p1)
                for off in offsets:
                    key = tuple(base[d] + off[d] for d in range(dims))
                    for cand in grid.get(key, ()):
                        if cand in processed:
                            continue
                        p2 = pts[cand]
                        dist_sq = sum((p1[d] - p2[d]) ** 2 for d in range(dims))
                        if dist_sq <= threshold_sq:
                            processed.add(cand)
                            cluster.append(cand)
                            queue.append(cand)
            clusters.append(cluster)
        return clusters

    @staticmethod
    def hash_points(points, precision=4):
        """Hash the given list of point values (fixed-point, position-stable).

        Parameters:
            points (list): A list of point values as tuples.
            precision (int): Number of decimal places retained in the fixed-point
                representation (e.g. 4 retains 4 decimals).

        Returns:
            (list) list(s) of hashed tuples.

        Example:
            hash_points([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])  # [hash values]
            hash_points([[(1.0, 2.0, 3.0)], [(4.0, 5.0, 6.0)]])  # [[..], [..]]
        """
        nested = IterUtils.nested_depth(points) > 1
        sets = points if nested else [points]

        def clamp(p):
            return int(p * 10**precision)

        result = []
        for pset in sets:
            result.append([hash(tuple(map(clamp, i))) for i in pset])
        return CoreUtils.format_return(result, nested)
