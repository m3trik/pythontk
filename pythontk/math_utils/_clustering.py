# !/usr/bin/python
# coding=utf-8
"""Optimal assignment (Hungarian / Jonker-Volgenant) and k-means clustering,
1-D and N-D (the bodies behind the :class:`MathUtils` facade).
"""

from __future__ import annotations

from typing import List, Tuple, Sequence, Optional


class _MathClusteringInternal:
    """Bodies of :class:`MathUtils`' assignment and clustering (a ``MathUtils`` base).

    The public signatures and docstrings stay on the facade; each delegates
    here.
    """

    @staticmethod
    def linear_sum_assignment(
        cost_matrix: Sequence[Sequence[float]],
        maximize: bool = False,
    ) -> Tuple[List[int], List[int]]:
        """Body of :meth:`MathUtils.linear_sum_assignment`."""
        # Fast path: use scipy if available
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment as scipy_lsa

            arr = np.asarray(cost_matrix, dtype=float)
            if arr.size == 0:
                return ([], [])
            row_ind, col_ind = scipy_lsa(arr, maximize=maximize)
            return (row_ind.tolist(), col_ind.tolist())
        except ImportError:
            pass

        def _hungarian_square(cost_sq: List[List[float]]) -> List[int]:
            """Return assignment list where assignment[i] = j for square matrix."""
            n = len(cost_sq)
            if n == 0:
                return []

            u = [0.0] * (n + 1)
            v = [0.0] * (n + 1)
            p = [0] * (n + 1)
            way = [0] * (n + 1)

            for i in range(1, n + 1):
                p[0] = i
                j0 = 0
                minv = [float("inf")] * (n + 1)
                used = [False] * (n + 1)

                while True:
                    used[j0] = True
                    i0 = p[j0]
                    delta = float("inf")
                    j1 = 0
                    for j in range(1, n + 1):
                        if used[j]:
                            continue
                        cur = cost_sq[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j

                    for j in range(0, n + 1):
                        if used[j]:
                            u[p[j]] += delta
                            v[j] -= delta
                        else:
                            minv[j] -= delta

                    j0 = j1
                    if p[j0] == 0:
                        break

                while True:
                    j1 = way[j0]
                    p[j0] = p[j1]
                    j0 = j1
                    if j0 == 0:
                        break

            assignment = [0] * n
            for j in range(1, n + 1):
                assignment[p[j] - 1] = j - 1
            return assignment

        # Validate matrix and extract dimensions
        if not cost_matrix:
            return ([], [])

        n_rows = len(cost_matrix)
        n_cols = len(cost_matrix[0]) if n_rows else 0
        if n_cols == 0:
            return ([], [])

        for row in cost_matrix:
            if len(row) != n_cols:
                raise ValueError(
                    "cost_matrix must be rectangular (all rows same length)"
                )

        # Convert to a mutable list-of-lists and optionally transform for maximize
        costs: List[List[float]] = [list(map(float, row)) for row in cost_matrix]

        flat = [c for row in costs for c in row]
        if not flat:
            return ([], [])

        if maximize:
            max_val = max(flat)
            costs = [[max_val - c for c in row] for row in costs]
            flat = [c for row in costs for c in row]

        # Pad to square with a large dummy cost
        max_cost = max(flat)
        pad_cost = max_cost + abs(max_cost) + 1.0

        n = max(n_rows, n_cols)
        square = [[pad_cost] * n for _ in range(n)]
        for i in range(n_rows):
            for j in range(n_cols):
                square[i][j] = costs[i][j]

        assignment = _hungarian_square(square)

        row_ind: List[int] = []
        col_ind: List[int] = []
        for i in range(n_rows):
            j = assignment[i]
            if j < n_cols:
                row_ind.append(i)
                col_ind.append(j)

        return (row_ind, col_ind)

    @staticmethod
    def kmeans_clustering(
        points: Sequence[Sequence[float]],
        k: int,
        max_iterations: int = 30,
        seed_indices: Optional[List[int]] = None,
    ) -> List[List[int]]:
        """Body of :meth:`MathUtils.kmeans_clustering`."""
        try:
            import numpy as np
        except ImportError:
            np = None

        n = len(points)
        if n == 0:
            return []
        if k <= 1:
            return [list(range(n))]
        k = min(k, n)
        # At least one assignment pass must run: with zero iterations the
        # numpy path's labels stay None and the fallback path never builds
        # its groups (NameError).
        max_iterations = max(1, max_iterations)

        if np is not None:
            pts = np.asarray(points, dtype=float)

            # Initialization
            centers = []
            if seed_indices and len(seed_indices) >= k:
                centers = pts[seed_indices[:k]]
            else:
                # Farthest Point Sampling
                centers_list = []
                first_idx = 0
                centers_list.append(pts[first_idx])

                while len(centers_list) < k:
                    c = np.vstack(centers_list)
                    d2 = np.min(
                        np.sum((pts[:, None, :] - c[None, :, :]) ** 2, axis=2), axis=1
                    )
                    next_idx = np.argmax(d2)
                    centers_list.append(pts[next_idx])
                centers = np.vstack(centers_list)

            labels = None
            pts_sq = np.sum(pts**2, axis=1)

            for _ in range(max_iterations):
                # Assign to nearest center
                centers_sq = np.sum(centers**2, axis=1)
                d2 = (
                    pts_sq[:, np.newaxis]
                    + centers_sq[np.newaxis, :]
                    - 2 * np.dot(pts, centers.T)
                )
                new_labels = np.argmin(d2, axis=1)

                if labels is not None and np.array_equal(new_labels, labels):
                    break
                labels = new_labels

                # Update centers
                new_centers = centers.copy()
                for i in range(k):
                    mask = labels == i
                    if np.any(mask):
                        new_centers[i] = pts[mask].mean(axis=0)

                if np.allclose(new_centers, centers):
                    centers = new_centers
                    break
                centers = new_centers

            groups = [np.where(labels == i)[0].tolist() for i in range(k)]
            return [g for g in groups if g]

        else:
            # Fallback without numpy
            # Initialization (Farthest Point)
            centers = [points[0]]
            while len(centers) < k:
                max_dist = -1
                farthest_pt = None

                for p in points:
                    min_d_to_center = float("inf")
                    for c in centers:
                        d = sum((p[i] - c[i]) ** 2 for i in range(len(p)))
                        if d < min_d_to_center:
                            min_d_to_center = d

                    if min_d_to_center > max_dist:
                        max_dist = min_d_to_center
                        farthest_pt = p

                if farthest_pt:
                    centers.append(farthest_pt)
                else:
                    break

            labels = [-1] * n
            for _ in range(max_iterations):
                changes = 0
                groups = [[] for _ in range(k)]
                for i, p in enumerate(points):
                    best_idx = -1
                    min_dist = float("inf")
                    for c_idx, c in enumerate(centers):
                        d = sum((p[j] - c[j]) ** 2 for j in range(len(p)))
                        if d < min_dist:
                            min_dist = d
                            best_idx = c_idx

                    if labels[i] != best_idx:
                        changes += 1
                    labels[i] = best_idx
                    groups[best_idx].append(i)

                if changes == 0:
                    break

                # Recompute centers
                for i in range(k):
                    g_indices = groups[i]
                    if g_indices:
                        dim = len(points[0])
                        mean_pt = [0.0] * dim
                        for idx in g_indices:
                            for d in range(dim):
                                mean_pt[d] += points[idx][d]
                        mean_pt = [x / len(g_indices) for x in mean_pt]
                        centers[i] = mean_pt

            return [g for g in groups if g]

    @staticmethod
    def kmeans_1d(
        values: Sequence[float],
        k: int = 3,
        max_iterations: int = 10,
    ) -> Tuple[List[float], List[List[float]]]:
        """Body of :meth:`MathUtils.kmeans_1d`."""
        if not values:
            return [], []

        vals = list(values)
        n = len(vals)
        n_unique = len(set(vals))

        if n_unique == 1:
            return [vals[0]], [vals]

        k = min(k, n_unique)

        # Try numpy fast path
        try:
            import numpy as np

            arr = np.asarray(vals, dtype=float)
            sorted_vals = np.sort(arr)

            # Initialize centers using quantiles
            indices = ((np.arange(k) + 0.5) * n / k).astype(int)
            indices = np.clip(indices, 0, n - 1)
            centers = sorted_vals[indices].copy()

            labels = np.zeros(n, dtype=int)
            for _ in range(max_iterations):
                # Vectorized distance calculation: |arr - centers|
                dists = np.abs(arr[:, np.newaxis] - centers[np.newaxis, :])
                new_labels = np.argmin(dists, axis=1)

                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels

                # Recompute centers
                for i in range(k):
                    mask = labels == i
                    if np.any(mask):
                        centers[i] = arr[mask].mean()

            # Sort by center value
            order = np.argsort(centers)
            centers = centers[order].tolist()

            # Build groups in sorted order
            groups = [arr[labels == order[i]].tolist() for i in range(k)]
            return centers, groups

        except ImportError:
            pass

        # Pure Python fallback
        sorted_vals = sorted(vals)
        centers = []
        for i in range(k):
            idx = int((i + 0.5) * n / k)
            idx = min(idx, n - 1)
            centers.append(sorted_vals[idx])

        labels = [-1] * n
        for _ in range(max_iterations):
            new_labels = []
            for v in vals:
                dists = [abs(v - c) for c in centers]
                new_labels.append(dists.index(min(dists)))

            if new_labels == labels:
                break
            labels = new_labels

            new_centers = []
            for i in range(k):
                cluster_vals = [vals[j] for j in range(n) if labels[j] == i]
                if cluster_vals:
                    new_centers.append(sum(cluster_vals) / len(cluster_vals))
                else:
                    new_centers.append(centers[i])
            centers = new_centers

        # Build groups and sort by center value
        groups = [[] for _ in range(k)]
        for j, lbl in enumerate(labels):
            groups[lbl].append(vals[j])

        sorted_indices = sorted(range(k), key=lambda i: centers[i])
        centers = [centers[i] for i in sorted_indices]
        groups = [groups[i] for i in sorted_indices]

        return centers, groups

    @classmethod
    def get_kmeans_threshold(
        cls,
        values: Sequence[float],
        k: int = 3,
    ) -> float:
        """Body of :meth:`MathUtils.get_kmeans_threshold`."""
        if not values:
            return 0.0

        if len(set(values)) < 2:
            return values[0] * 0.5 if values else 0.0

        centers, groups = cls.kmeans_1d(values, k=k)

        if len(centers) < 2:
            return centers[0] * 0.5 if centers else 0.0

        # Merge logic: if middle cluster is close to small cluster (ratio < 3.0),
        # treat them as the same class and threshold between merged and large.
        if len(centers) >= 3:
            c0, c1, c2 = centers[0], centers[1], centers[2]
            if c1 < c0 * 3.0:
                # Merge small + medium; threshold between c1 and c2
                return (c1 + c2) / 2.0
            else:
                # Threshold between c0 and c1
                return (c0 + c1) / 2.0
        else:
            # Only 2 clusters
            return (centers[0] + centers[1]) / 2.0
