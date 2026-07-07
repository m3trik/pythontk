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
    def _refine_rotation(
        r_stack,
        avg_dists,
        p_a,
        p_b,
        n_a,
        n_b,
        tolerance,
        normal_threshold,
        tree,
        top_k: int = 8,
        iterations: int = 4,
    ):
        """Kabsch-refine the best coarse rotations; return an accepted one or None.

        Starting from each of the *top_k* candidates by point fit, iterate
        nearest-neighbor pairing → orthogonal-Procrustes solve. For a true
        rigid copy this converges from the nearest discrete candidate onto
        the exact rotation. Acceptance mirrors the discrete gates: mean
        distance within *tolerance* and, when normals are given, flip-free
        best-twin agreement ≥ *normal_threshold*. The SVD solution is
        constrained to PROPER rotations (det +1) so a reflected twin can
        never slip through as a "refinement".
        """
        import numpy as np

        use_normals = n_a is not None and n_b is not None
        best = None  # (score, rotation)
        for ci in np.argsort(avg_dists)[:top_k]:
            R = r_stack[int(ci)]
            for _ in range(iterations):
                _, nn = tree.query(p_b @ R.T, k=1)
                H = p_b.T @ p_a[nn]
                U, _s, Vt = np.linalg.svd(H)
                d = np.sign(np.linalg.det(Vt.T @ U.T))
                R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

            k = min(4, len(p_a)) if use_normals else 1
            dists, nn = tree.query(p_b @ R.T, k=k)
            if k == 1:
                dists, nn = dists.reshape(-1, 1), nn.reshape(-1, 1)
            if float(dists[:, 0].mean()) > tolerance:
                continue
            if use_normals:
                cand = np.einsum("pki,pi->pk", n_a[nn], n_b @ R.T)
                eligible = dists <= tolerance
                eligible[:, 0] = True
                dots = np.where(eligible, cand, -np.inf).max(axis=1)
                if float(dots.mean()) < normal_threshold or bool((dots < 0.0).any()):
                    continue
                score = float(dots.mean())
            else:
                score = -float(dists[:, 0].mean())
            if best is None or score > best[0]:
                best = (score, R)
        return None if best is None else best[1]

    @staticmethod
    def pca_transform(
        points_a: "np.ndarray",
        points_b: "np.ndarray",
        tolerance: float = 0.001,
        robust: bool = False,
        sample_size: int = 500,
        symmetry_threshold: float = 0.1,
        normals_a: Optional["np.ndarray"] = None,
        normals_b: Optional["np.ndarray"] = None,
        normal_threshold: float = 0.8,
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
            normals_a: Optional (N, 3) unit normals paired with ``points_a``.
            normals_b: Optional (N, 3) unit normals paired with ``points_b``.
                When both normal arrays are given, candidate rotations must also
                align the normals: among rotations whose point fit is within
                ``tolerance``, the one with the best normal agreement wins, and
                a winner whose mean normal dot falls below ``normal_threshold``
                is rejected. Point positions alone cannot distinguish a
                symmetric shape from its flipped twin (a flat plate maps onto
                itself under a 180° flip while its normals invert).
            normal_threshold: Minimum mean dot product for a normal-verified
                match (only used when normals are provided).

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

        use_normals = normals_a is not None and normals_b is not None
        if use_normals:
            n_a = np.array(normals_a)
            n_b = np.array(normals_b)
            if len(n_a) != len(pts_a) or len(n_b) != len(pts_b):
                use_normals = False

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

            # Subsample ONLY the query side for performance (deterministic
            # stride — a random choice made matching nondeterministic).
            # The KDTree target side must stay dense: subsampling BOTH sides
            # removes a query point's true twin from the target set, so even
            # an exact copy at the exact rotation scores a nearest-neighbor
            # floor around the inter-vertex spacing and no candidate can
            # ever pass a tight tolerance (broke every >sample_size match).
            def subsample_idx(n):
                if n <= sample_size:
                    return np.arange(n)
                return np.linspace(0, n - 1, sample_size).astype(int)

            idx_b = subsample_idx(len(p_b))
            p_a_work = p_a
            p_b_work = p_b[idx_b]
            if use_normals:
                n_a_work = n_a
                n_b_work = n_b[idx_b]
        else:
            p_a_work = p_a
            p_b_work = p_b
            if use_normals:
                n_a_work = n_a
                n_b_work = n_b

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

        # 6. Assemble every candidate rotation (base orientations × spins).
        # The identity is always included explicitly: for DEGENERATE shapes
        # (near-equal eigenvalues — a cube, a sphere) eigh returns an
        # arbitrary basis per cloud, so no eigenvector-derived candidate is
        # guaranteed to align two already-aligned clouds.
        candidate_rotations = [np.eye(3)]
        for P in base_rotations:
            R_base = vec_a @ P @ vec_b.T
            if sym_axis_b is not None:
                spin_axis = vec_b[:, sym_axis_b]
                candidate_rotations.extend(
                    R_base @ axis_angle_matrix(spin_axis, angle)
                    for angle in spin_angles
                )
            else:
                candidate_rotations.append(R_base)

        # 7. Score all candidates in one vectorized pass. Symmetric shapes
        # produce 24 × 24 = 576 candidates; querying them per-rotation in a
        # Python loop dominated the whole instancing pipeline, while a single
        # stacked KDTree query is one C call.
        r_stack = np.array(candidate_rotations)  # (K, 3, 3)
        # out[k, n] = R_k @ p_b[n]  ==  p_b @ R_k.T
        rotated = np.einsum("kij,nj->kni", r_stack, p_b_work)

        # Hard edges duplicate a position with different normals; pairing a
        # rotated point with the single nearest neighbor can then pick the
        # wrong coincident twin and veto a true rigid copy. Each point
        # therefore scores by its best-agreeing neighbor among the K nearest
        # that sit within the positional tolerance (the nearest always
        # counts, so non-duplicated points behave exactly as before).
        agreements = None
        flip_fracs = None
        k_twins = min(4, len(p_a_work)) if use_normals else 1
        if KDTree:
            tree = KDTree(p_a_work)
            dists, nn = tree.query(rotated.reshape(-1, 3), k=k_twins)
            if k_twins == 1:
                dists = dists.reshape(-1, 1)
                nn = nn.reshape(-1, 1)
            avg_dists = dists[:, 0].reshape(len(r_stack), -1).mean(axis=1)
            if use_normals:
                rot_n = np.einsum("kij,nj->kni", r_stack, n_b_work).reshape(-1, 3)
                cand = np.einsum("pki,pi->pk", n_a_work[nn], rot_n)
                eligible = dists <= tolerance
                eligible[:, 0] = True
                dots = np.where(eligible, cand, -np.inf).max(axis=1)
                dots = dots.reshape(len(r_stack), -1)
                agreements = dots.mean(axis=1)
                flip_fracs = (dots < 0.0).mean(axis=1)
        else:
            # Brute-force fallback: loop per rotation to bound memory at
            # (N, M) instead of (K, N, M).
            a_sq = np.sum(p_a_work**2, axis=1)
            b_sq = np.sum(p_b_work**2, axis=1)
            avg_dists = np.empty(len(r_stack))
            if use_normals:
                agreements = np.empty(len(r_stack))
                flip_fracs = np.empty(len(r_stack))
            for k in range(len(r_stack)):
                dists_sq = (
                    b_sq[:, np.newaxis]
                    + a_sq[np.newaxis, :]
                    - 2 * np.dot(rotated[k], p_a_work.T)
                )
                nn_k = dists_sq.argmin(axis=1)
                min_dists = np.sqrt(
                    np.maximum(dists_sq[np.arange(len(nn_k)), nn_k], 0)
                )
                avg_dists[k] = min_dists.mean()
                if use_normals:
                    rot_n_k = n_b_work @ r_stack[k].T
                    kk = min(k_twins, dists_sq.shape[1])
                    part = np.argpartition(dists_sq, kk - 1, axis=1)[:, :kk]
                    d_part = np.sqrt(
                        np.maximum(np.take_along_axis(dists_sq, part, axis=1), 0)
                    )
                    cand = np.einsum("pki,pi->pk", n_a_work[part], rot_n_k)
                    eligible = d_part <= tolerance
                    rows = np.arange(len(d_part))
                    eligible[rows, d_part.argmin(axis=1)] = True
                    dots_k = np.where(eligible, cand, -np.inf).max(axis=1)
                    agreements[k] = dots_k.mean()
                    flip_fracs[k] = (dots_k < 0.0).mean()

        best_matrix = None
        within = avg_dists <= tolerance
        if within.any():
            if agreements is not None:
                # Among geometric fits, the rotation must also align shading.
                # Point positions alone cannot tell a symmetric shape from its
                # flipped twin — the flip matches every vertex while inverting
                # every normal. A true rigid copy aligns EVERY normal (float
                # noise cannot drive a ~1 dot below zero), so a single flipped
                # normal (e.g. a small asymmetric feature on an otherwise
                # symmetric part) rejects the candidate outright.
                valid = (
                    within
                    & (agreements >= normal_threshold)
                    & (flip_fracs <= 0.0)
                )
                if valid.any():
                    candidates_idx = np.flatnonzero(valid)
                    best_matrix = r_stack[
                        int(candidates_idx[np.argmax(agreements[candidates_idx])])
                    ]
            else:
                best_matrix = r_stack[int(np.argmin(avg_dists))]

        if best_matrix is None and robust and KDTree is not None:
            # The discrete search quantizes spin around a symmetric axis to
            # 15° — a copy rotated by an arbitrary angle lands NEAR a
            # candidate but outside a tight tolerance. Kabsch-refine the
            # best coarse candidates onto the exact rotation; acceptance
            # still applies the strict tolerance and normals gates.
            best_matrix = PointCloud._refine_rotation(
                r_stack,
                avg_dists,
                p_a_work,
                p_b_work,
                n_a_work if use_normals else None,
                n_b_work if use_normals else None,
                tolerance,
                normal_threshold,
                tree,
            )
        if best_matrix is None:
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
