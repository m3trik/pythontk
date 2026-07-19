# !/usr/bin/python
# coding=utf-8
from bisect import bisect_left
import math
from typing import Callable, List, Tuple, Union, Sequence, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# from this package:
from pythontk.core_utils._core_utils import CoreUtils
from pythontk.core_utils.help_mixin import HelpMixin


class MathUtils(HelpMixin):
    """ """

    # Centimeter-relative length factors (1 unit -> N centimeters).
    LENGTH_UNIT_FACTORS = {
        "mm": 0.1,
        "cm": 1.0,
        "m": 100.0,
        "km": 100000.0,
        "in": 2.54,
        "ft": 30.48,
        "yd": 91.44,
        "mi": 160934.4,
    }

    @staticmethod
    def eval_expression(expression: str) -> str:
        """Evaluate a math expression string (calculator engine).

        Exposes the ``math`` module's functions/constants plus ``abs/round/min/max/pow``.
        The expression is parsed to an AST and only arithmetic nodes are evaluated
        (numeric literals; the binary operators ``+ - * / // % **``; unary ``+``/``-``;
        and calls to the exposed names). Attribute access, subscripting, comprehensions
        and every other construct return ``"Error"`` -- so a nulled-``__builtins__``
        escape such as ``(1).__class__.__base__.__subclasses__()`` cannot reach the
        object graph. Integer-valued floats are returned without a trailing ``.0``.

        Parameters:
            expression (str): e.g. ``"sin(pi/2) + 2**3"``.

        Returns:
            str: the formatted result, ``""`` for an empty expression, or ``"Error"``
                when the expression is invalid or uses a disallowed construct.

        Example:
            MathUtils.eval_expression("2+2")        -> "4"
            MathUtils.eval_expression("10/4")       -> "2.5"
            MathUtils.eval_expression("sqrt(16)")   -> "4"
        """
        if not expression:
            return ""
        import ast
        import operator

        allowed_names = {
            name: getattr(math, name) for name in dir(math) if not name.startswith("__")
        }
        allowed_names.update(
            {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        )
        binops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        unaryops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

        def _ev(node):
            if isinstance(node, ast.Expression):
                return _ev(node.body)
            if isinstance(node, ast.Constant) and isinstance(
                node.value, (int, float, complex)
            ):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in binops:
                return binops[type(node.op)](_ev(node.left), _ev(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in unaryops:
                return unaryops[type(node.op)](_ev(node.operand))
            if isinstance(node, ast.Name) and node.id in allowed_names:
                return allowed_names[node.id]
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and not node.keywords
                and node.func.id in allowed_names
            ):
                fn = allowed_names[node.func.id]
                if callable(fn):
                    return fn(*[_ev(a) for a in node.args])
            raise ValueError("disallowed expression")

        try:
            result = _ev(ast.parse(expression, mode="eval"))
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return str(result)
        except Exception:
            return "Error"

    @classmethod
    def convert_length_unit(cls, value: float, from_unit: str, to_unit: str) -> str:
        """Convert a length ``value`` between units (mm, cm, m, km, in, ft, yd, mi).

        Parameters:
            value (float): the numeric value to convert.
            from_unit (str): source unit key (one of ``LENGTH_UNIT_FACTORS``).
            to_unit (str): target unit key.

        Returns:
            str: the converted value rounded to 6 decimals, or ``"Error"`` for an
                unknown unit / non-numeric value.

        Example:
            MathUtils.convert_length_unit(100, "cm", "m")  -> "1.0"
            MathUtils.convert_length_unit(1, "in", "cm")   -> "2.54"
        """
        try:
            factors = cls.LENGTH_UNIT_FACTORS
            if from_unit not in factors or to_unit not in factors:
                return "Error"
            cm_value = float(value) * factors[from_unit]
            return str(round(cm_value / factors[to_unit], 6))
        except (TypeError, ValueError):
            return "Error"

    @staticmethod
    def linear_sum_assignment(
        cost_matrix: Sequence[Sequence[float]],
        maximize: bool = False,
    ) -> Tuple[List[int], List[int]]:
        """Solve the linear sum assignment problem (Hungarian algorithm).

        Uses scipy.optimize.linear_sum_assignment when available (10-100x faster),
        otherwise falls back to a pure Python implementation.

        Parameters:
            cost_matrix: Rectangular or square cost matrix (rows x cols). Each entry
                is the cost of assigning row i to column j.
            maximize: If True, finds the assignment with maximum total score instead
                of minimum total cost.

        Returns:
            Tuple[List[int], List[int]]: (row_indices, col_indices) such that
                pairing row_indices[k] -> col_indices[k] yields the optimal assignment.
                The number of pairs is ``min(n_rows, n_cols)``.

        Example:
            cost = [[4, 1, 3],
                    [2, 0, 5],
                    [3, 2, 2]]
            rows, cols = MathUtils.linear_sum_assignment(cost)
        """
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
        """
        Perform K-Means clustering on a set of points.

        Parameters:
            points: List of (x, y, z) tuples or list of lists.
            k: Number of clusters.
            max_iterations: Maximum number of iterations.
            seed_indices: Optional list of indices to use as initial centers.
                          If provided, the first k indices will be used as seeds.
                          If None, uses farthest-point sampling.

        Returns:
            List of lists, where each inner list contains the indices of points in that cluster.
        """
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
        """
        Perform 1D K-Means clustering to find natural breakpoints in scalar data.

        Useful for separating "small", "medium", and "large" items based on size.
        Uses numpy for vectorized operations when available.

        Parameters:
            values: Sequence of scalar values to cluster.
            k: Number of clusters (default 3 for small/medium/large classification).
            max_iterations: Maximum iterations for convergence.

        Returns:
            Tuple of (centers, groups) where:
                - centers: List of cluster center values (sorted ascending).
                - groups: List of lists, each containing ALL input values belonging to that cluster.

        Example:
            centers, groups = kmeans_1d([1, 2, 3, 50, 55, 60], k=2)
            # centers ≈ [2.0, 55.0]
            # groups ≈ [[1, 2, 3], [50, 55, 60]]

            # Duplicates are preserved:
            centers, groups = kmeans_1d([1, 1, 1, 50, 50], k=2)
            # groups ≈ [[1, 1, 1], [50, 50]]
        """
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
        """
        Use K-Means to find an adaptive threshold separating "parts" from "bodies".

        For assembly classification (e.g., separating small lids from large containers),
        this finds the natural breakpoint between the smallest cluster(s) and the largest.

        Parameters:
            values: Sequence of size values (e.g., volumes, areas).
            k: Number of clusters to use (default 3 for small/medium/large).

        Returns:
            A threshold value such that items below it are "parts" and above are "bodies".
            Returns 0.0 if values is empty.

        Example:
            threshold = get_kmeans_threshold([0.8, 1.2, 2.1, 12.4, 15.0], k=3)
            # threshold ≈ 7.25 (midpoint between medium and large clusters)
        """
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

    @staticmethod
    @CoreUtils.listify(threading=True)
    def move_decimal_point(num, places):
        """Move the decimal place in a given number.

        Parameters:
            num (int/float): The number in which you are modifying.
            places (int): The number of decimal places to move.

        Returns:
            (float)

        Example:
            move_decimal_point(11.05, -2) #returns: 0.1105
        """
        from decimal import Decimal

        num_decimal = Decimal(str(num))  # Convert the input number to a Decimal object
        scaling_factor = (
            Decimal(10) ** places
        )  # Create a scaling factor as a Decimal object (exact for negative places)

        result = (
            num_decimal * scaling_factor
        )  # Perform the operation using Decimal objects
        return float(result)  # Convert the result back to a float

    @staticmethod
    def get_vector_from_two_points(
        a: List[float], b: List[float]
    ) -> Tuple[float, float, float]:
        """Get a directional vector from a given start and end point.

        Parameters:
            a (List[float]): A start point given as [x, y, z].
            b (List[float]): An end point given as [x, y, z].

        Returns:
            Tuple[float, float, float]: The directional vector from the start point to the end point.

        Example:
            get_vector_from_two_points([1, 2, 3], [1, 1, -1]) #returns: (0, -1, -4)
        """
        return (b[0] - a[0], b[1] - a[1], b[2] - a[2])

    @staticmethod
    @CoreUtils.listify(threading=True)
    def clamp(n=0.0, minimum=0.0, maximum=1.0):
        """Clamps the value x between min and max.

        Parameters:
            n (float)(tuple): The numeric value to clamp.
            minimum (float): Clamp min value.
            maximum (float): Clamp max value.

        Returns:
            (float)

        Example:
            clamp(range(10), 3, 7) #returns: [3, 3, 3, 3, 4, 5, 6, 7, 7, 7]
        """
        return max(minimum, min(n, maximum))

    @staticmethod
    def clamp_range(
        start,
        end,
        clamp_start=None,
        clamp_end=None,
        validate=True,
    ):
        """Clamp a numeric range (start, end) to optional boundaries with validation.

        Parameters:
            start (float): Range start value.
            end (float): Range end value.
            clamp_start (float, optional): Optional minimum boundary for start.
            clamp_end (float, optional): Optional maximum boundary for end.
            validate (bool): If True, validates that start < end and returns None if invalid.
                           If False, no validation is performed (use with caution).

        Returns:
            tuple: Clamped (start, end) tuple, or None if validation fails.

        Example:
            clamp_range(5, 15) #returns: (5, 15)
            clamp_range(5, 15, clamp_start=10) #returns: (10, 15)
            clamp_range(5, 15, clamp_end=12) #returns: (5, 12)
            clamp_range(5, 15, clamp_start=10, clamp_end=12) #returns: (10, 12)
            clamp_range(15, 5, clamp_start=10, clamp_end=12) #returns: None (invalid: start >= end)
            clamp_range(5, 15, clamp_start=20) #returns: None (clamping makes it invalid)
            clamp_range(None, 15) #returns: None (None values not allowed)
        """
        if start is None or end is None:
            return None

        # Clamp start and end to their respective boundaries
        final_start = max(start, clamp_start) if clamp_start is not None else start
        final_end = min(end, clamp_end) if clamp_end is not None else end

        # Validate that start < end
        if validate and final_start >= final_end:
            return None

        return (final_start, final_end)

    @classmethod
    def normalize(cls, vector, amount=1):
        """Normalize a 2 or 3 dimensional vector.

        Parameters:
            vector (tuple): A two or three point vector. ie. (-0.03484, 0.0, -0.55195)
            amount (float): (1) Normalize standard. (value other than 0 or 1) Normalize using the given float value as desired length.

        Returns:
            (tuple)

        Example:
            normalize((2, 3, 4)) #returns: (0.3713906763541037, 0.5570860145311556, 0.7427813527082074)
            normalize((2, 3)) #returns: (0.5547001962252291, 0.8320502943378437)
            normalize((2, 3, 4), 2) #returns: (0.7427813527082074, 1.1141720290623112, 1.4855627054164149)
        """
        n = len(vector)  # determine 2 or 3d vector.
        length = cls.get_magnitude(vector)
        return tuple(vector[i] / length * amount for i in range(n))

    @staticmethod
    def get_magnitude(vector):
        """Get the magnatude (length) of a given vector.

        Parameters:
            vector (tuple): A two or three point vector. ie. (-0.03484, 0.0, -0.55195)

        Returns:
            (float)

        Example:
            get_magnitude((2, 3, 4)) #returns: 5.385164807134504
            get_magnitude((2, 3)) #returns: 3.605551275463989
        """
        from math import sqrt

        n = len(vector)  # determine 2 or 3d vector.
        return sqrt(sum(vector[i] * vector[i] for i in range(n)))

    @classmethod
    def dot_product(cls, v1, v2, normalize_input=False):
        """Returns the dot product of two 3D float arrays.  If $normalize_input
        is set then the vectors are normalized before the dot product is calculated.

        Parameters:
            v1 (tuple): The first 3 point vector.
            v2 (tuple): The second 3 point vector.
            normalize_input (int): Normalize v1, v2 before calculating the point float list.

        Returns:
            (float) Dot product of the two vectors.

        Example:
            dot_product((1, 2, 3), (1, 1, -1)) #returns: 0
            dot_product((1, 2), (1, 1)) #returns: 3
        """
        if normalize_input:  # normalize the input vectors
            v1 = cls.normalize(v1)
            v2 = cls.normalize(v2)

        return sum((a * b) for a, b in zip(v1, v2))  # the dot product

    @classmethod
    def cross_product(cls, a, b, c=None, normalize=0):
        """Get the cross product of two vectors, using two 3d vectors, or 3 points.

        Parameters:
            a (tuple): A point represented as x,y,z or a 3 point vector.
            b (tuple): A point represented as x,y,z or a 3 point vector.
            c (tuple): A point represented as x,y,z (used only when working with 3 point values instead of 2 vectors).
            normalize (float): (0) Do not normalize. (1) Normalize standard. (value other than 0 or 1) Normalize using the given float value as desired length.

        Returns:
            (tuple)

        Example:
            cross_product((1, 2, 3), (1, 1, -1)) #returns: (-5, 4, -1),
            cross_product((3, 1, 1), (1, 4, 2), (1, 3, 4)) #returns: (7, 4, 2),
            cross_product((1, 2, 3), (1, 1, -1), None, 1) #returns: (-0.7715167498104595, 0.6172133998483676, -0.1543033499620919)
        """
        if c is not None:  # convert points to vectors and unpack.
            a = cls.get_vector_from_two_points(a, b)
            b = cls.get_vector_from_two_points(b, c)

        ax, ay, az = a
        bx, by, bz = b

        result = ((ay * bz) - (az * by), (az * bx) - (ax * bz), (ax * by) - (ay * bx))

        if normalize:
            # safe_normalize returns the (zero) result unchanged for a degenerate
            # cross product (parallel/collinear inputs) instead of dividing by 0.
            result = cls.safe_normalize(result, result, normalize)

        return result

    @classmethod
    def move_point_relative(cls, p, d, v=None):
        """Move a point relative to it's current position.

        Parameters:
            p (tuple): A points x, y, z values.
            d (tuple)(float): The distance to move. Use a float value when moving along a vector,
                                    and a point value to move in a given distance.
            v (tuple): Optional: A vectors x, y, z values can be given to move the point along that vector.

        Returns:
            (tuple) point.

        Example:
            move_point_relative((0, 5, 0), (0, 5, 0)) #returns: (0, 10, 0)
            move_point_relative((0, 5, 0), 5, (0, 1, 0)) #returns: (0, 10, 0)
        """
        x, y, z = p

        if v is not None:  # move along a vector if one is given.
            assert isinstance(
                d, (float, int)
            ), "# Error: {}\n  The distance parameter requires an integer or float value when moving along a vector.\n  {} {} given. #".format(
                __file__, type(d), d
            )
            dx, dy, dz = cls.normalize(v, d)
        else:
            assert isinstance(
                d, (list, tuple, set)
            ), "# Error: {}\n  The distance parameter requires an list, tuple, set value when not moving along a vector.\n  {} {} given. #".format(
                __file__, type(d), d
            )
            dx, dy, dz = d

        result = (x + dx, y + dy, z + dz)

        return result

    @classmethod
    def move_point_relative_along_vector(cls, a, b, vect, dist, toward=True):
        """Move a point (a) along a given vector toward or away from a given point (b).

        Parameters:
            a (tuple): The point to move given as (x,y,z).
            b (tuple): The point to move toward.
            vect (tuple): A vector to move the point along.
            dist (float): The linear amount to move the point.
            toward (bool): Move the point toward or away from.

        Returns:
            (tuple) point.

        Example: move_point_relative_along_vector((0, 0, 0), (0, 10, 0), (0, 1, 0), 5) #returns: (0.0, 5.0, 0.0)
        Example: move_point_relative_along_vector((0, 0, 0), (0, 10, 0), (0, 1, 0), 5, False) #returns: (0.0, -15.0, 0.0)
        """
        lowest = None
        for i in [
            dist,
            -dist,
        ]:  # move in pos and neg direction, and determine which is moving closer to the reference point.
            p = cls.move_point_relative(a, i, vect)
            d = cls.distance_between_points(p, b)
            if lowest is None or (d < lowest if toward else d > lowest):
                result, lowest = (p, d)

        return result

    @staticmethod
    def distance_between_points(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
        """Calculates the Euclidean distance between two points in N-dimensional space.

        Parameters:
            a (Tuple[float, ...]): Coordinates of the first point, can be of any dimension.
            b (Tuple[float, ...]): Coordinates of the second point, must have the same dimension as `a`.

        Returns:
            float: The Euclidean distance between the two points.
        """
        import math

        # Utilize zip to pair corresponding coordinates and compute their squared differences
        squared_diffs = ((p1 - p2) ** 2 for p1, p2 in zip(a, b))

        # Sum the squared differences and take the square root to get the distance
        return math.sqrt(sum(squared_diffs))

    @staticmethod
    def get_center_of_two_points(
        a: List[float], b: List[float]
    ) -> Tuple[float, float, float]:
        """Get the point in the middle of two given points.

        Parameters:
            a (List[float]): A point given as [x, y, z].
            b (List[float]): A point given as [x, y, z].

        Returns:
            Tuple[float, float, float]: The center point between the two input points.

        Example:
            get_center_of_two_points([0, 10, 0], [0, 5, 0]) #returns: (0.0, 7.5, 0.0)
        """
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)

    @classmethod
    def get_angle_from_two_vectors(cls, v1, v2, degree=False):
        """Get an angle from two given vectors.

        Parameters:
            v1 (tuple): A vectors xyz values as a tuple.
            v2 (tuple): A vectors xyz values as a tuple.
            degree (bool): Convert the radian result to degrees.

        Returns:
            (float)

        Example:
            get_angle_from_two_vectors((1, 2, 3), (1, 1, -1)) #returns: 1.5707963267948966,
            get_angle_from_two_vectors((1, 2, 3), (1, 1, -1), True) #returns: 90
        """
        from math import acos, degrees

        def length(v):
            return (cls.dot_product(v, v)) ** 0.5

        cosine = cls.dot_product(v1, v2) / (length(v1) * length(v2))
        # Clamp to acos's valid domain; float rounding can push identical/parallel
        # vectors just past +-1 (e.g. 1.0000000000000002) -> math domain error.
        result = acos(max(-1.0, min(1.0, cosine)))

        if degree:
            result = round(degrees(result), 2)
        return result

    @staticmethod
    def get_angle_from_three_points(a, b, c, degree=False):
        """Get the opposing angle from 3 given points.

        Parameters:
            a (tuple): A point given as (x,y,z).
            b (tuple): A point given as (x,y,z).
            c (tuple): A point given as (x,y,z).
            degree (bool): Convert the radian result to degrees.

        Returns:
            (float)

        Example:
            get_angle_from_three_points((1, 1, 1), (-1, 2, 3), (1, 4, -3)) #returns: 0.7904487543360762,
            get_angle_from_three_points((1, 1, 1), (-1, 2, 3), (1, 4, -3), True) #returns: 45.29
        """
        from math import sqrt, acos, degrees

        ba = [aa - bb for aa, bb in zip(a, b)]  # create vectors from points
        bc = [cc - bb for cc, bb in zip(c, b)]

        nba = sqrt(sum((x**2.0 for x in ba)))  # normalize vector
        ba = [x / nba for x in ba]

        nbc = sqrt(sum((x**2.0 for x in bc)))
        bc = [x / nbc for x in bc]

        scalar = sum(
            (aa * bb for aa, bb in zip(ba, bc))
        )  # get scalar from normalized vectors
        scalar = max(-1.0, min(1.0, scalar))  # guard float rounding past +-1 (collinear)

        angle = acos(scalar)  # get the angle in radian

        if degree:
            angle = round(degrees(angle), 2)
        return angle

    @staticmethod
    def get_two_sides_of_asa_triangle(a1, a2, s, unit="degrees"):
        """Get the length of two sides of a triangle, given two angles, and the length of the side in-between.

        Parameters:
            a1 (float): Angle in radians or degrees. (unit flag must be set if value given in radians)
            a2 (float): Angle in radians or degrees. (unit flag must be set if value given in radians)
            s (float): The distance of the side between the two given angles.
            unit (str): Specify whether the angle values are given in degrees or radians. (valid: 'radians', 'degrees')(default: degrees)

        Returns:
            (tuple)

        Raises:
            ValueError: If the two angles sum to >= 180° — no triangle exists
                (the two side rays are parallel or diverge).

        Example:
            get_two_sides_of_asa_triangle(60, 60, 100) #returns: (100.0, 100.0)
        """
        from math import sin, radians, degrees, pi

        if unit == "degrees":
            a1, a2 = radians(a1), radians(a2)

        a3 = pi - a1 - a2

        if a3 <= 0 or abs(sin(a3)) < 1e-9:
            raise ValueError(
                "Degenerate ASA triangle: the two angles sum to "
                f"{degrees(a1 + a2):.4f}° — the side rays never meet."
            )

        result = ((s / sin(a3)) * sin(a1), (s / sin(a3)) * sin(a2))

        return result

    @classmethod
    def xyz_rotation(cls, theta, axis, degree=False):
        """Get the rotation about the X,Y,Z axes given an angle for rotation
        (in radians) and an axis about which to do the rotation.

        Parameters:
            theta (float):The angular position of a vector in radians.
            axis (tuple): The rotation axis given as float values (x,y,z).
            degree (bool): Convert the radian result to degrees.

        Returns:
            (tuple) 3 point rotation.

        Example:
            xyz_rotation(2, (0, 1, 0)) #returns: (3.589792907376932e-09, 1.9999999964102069, 3.589792907376932e-09)
            xyz_rotation(2, (0, 1, 0), True) #returns: (0.0, 114.59, 0.0)
        """
        from math import cos, sin, sqrt, atan2, degrees, pi

        # set up the xyzw quaternion values
        theta *= 0.5
        w = cos(theta)
        factor = sin(theta)
        axisLen2 = cls.dot_product(axis, axis, 0)

        if axisLen2 != 1.0 and axisLen2 != 0.0:
            factor /= sqrt(axisLen2)
        x, y, z = factor * axis[0], factor * axis[1], factor * axis[2]

        # setup rotation in a matrix
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        s = 2.0 / (ww + xx + yy + zz)
        xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z
        matrix = [
            1.0 - s * (yy + zz),
            s * (xy + wz),
            s * (xz - wy),
            None,
            None,
            1.0 - s * (xx + zz),
            s * (yz + wx),
            None,
            None,
            s * (yz - wx),
            1.0 - s * (xx + yy),
        ]

        # get x,y,z values for rotation
        cosB = sqrt(matrix[0] * matrix[0] + matrix[1] * matrix[1])
        if cosB > 1.0e-10:
            a, b, c = solution1 = [
                atan2(matrix[6], matrix[10]),
                atan2(-matrix[2], cosB),
                atan2(matrix[1], matrix[0]),
            ]

            solution2 = [
                a + (pi if a < pi else -pi),
                (pi if b > -pi else -pi) - b,
                c + (pi if c < pi else -pi),
            ]

            if sum([abs(solution2[0]), abs(solution2[1]), abs(solution2[2])]) < sum(
                [abs(solution1[0]), abs(solution1[1]), abs(solution1[2])]
            ):
                rotation = solution2
            else:
                rotation = solution1

        else:
            rotation = [atan2(-matrix[9], matrix[5]), atan2(-matrix[2], cosB), 0.0]

        if degree:
            rotation = [round(degrees(r), 2) for r in rotation]
        return tuple(rotation)

    @staticmethod
    def lerp(start, end, t: float):
        """Linear interpolation between two values or two equal-length points.

        Scalars interpolate to a scalar; equal-length point/vector sequences
        interpolate component-wise to a tuple.

        Parameters:
            start: Start value (scalar) or point/vector (sequence).
            end: End value/point, matching ``start``.
            t (float): Interpolation factor (0.0 to 1.0).

        Returns:
            float | tuple: Interpolated value, or component-wise tuple for
            sequence inputs.
        """
        if isinstance(start, (list, tuple)):
            return tuple(a + t * (b - a) for a, b in zip(start, end))
        return start + t * (end - start)

    @classmethod
    def safe_normalize(cls, vector, fallback, amount: float = 1):
        """:meth:`normalize`, returning ``fallback`` for a ~zero-length vector.

        Guards the degenerate (coincident / zero) case so callers don't have to
        repeat the magnitude check before normalizing.
        """
        return (
            cls.normalize(vector, amount)
            if cls.get_magnitude(vector) > 1e-9
            else fallback
        )

    @staticmethod
    def smoothstep(x: float, edge0: float = 0.0, edge1: float = 1.0) -> float:
        """Canonical clamped Hermite smoothstep (``3t² − 2t³``).

        Eases ``x`` within ``[edge0, edge1]`` to a smooth ``0``..``1`` with zero
        slope at both ends; ``x`` outside the range clamps to ``0``/``1`` (so a
        raw ratio is safe to pass). Distinct from
        :meth:`ProgressionCurves.smooth_step`, which applies a weight exponent
        and does not clamp.

        Parameters:
            x (float): Input value.
            edge0 (float): Lower edge (maps to 0).
            edge1 (float): Upper edge (maps to 1).

        Returns:
            float: Smoothed value in ``0``..``1``.
        """
        if edge0 == edge1:
            return 0.0 if x < edge0 else 1.0
        t = (x - edge0) / (edge1 - edge0)
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def resolve_falloff_profile(
        profile: Union[str, Callable],
    ) -> Callable[[float], float]:
        """Resolve a falloff profile to a callable ``f(t) -> w`` over t in [0, 1].

        Parameters:
            profile (str/callable): A callable is passed through unchanged.
                "smoothstep" maps to :meth:`smoothstep` (clamped Hermite); any
                name in ``ProgressionCurves.CALCULATION_MODES`` (e.g. "linear",
                "ease_in_out", "sine") maps to that curve function.

        Returns:
            (callable) The profile function.
        """
        # Deferred: progression.py imports MathUtils, so a module-level import
        # here would be circular.
        from pythontk.math_utils.progression import ProgressionCurves

        if callable(profile):
            return profile
        if isinstance(profile, str):
            name = profile.lower()
            if name == "smoothstep":
                return MathUtils.smoothstep
            if name in ProgressionCurves.CALCULATION_MODES:
                fn = getattr(ProgressionCurves, name)
                return lambda t: fn(t)
        valid = ["smoothstep"] + list(ProgressionCurves.CALCULATION_MODES)
        raise ValueError(
            f"Invalid falloff profile: {profile!r}. "
            f"Expected a callable or one of {valid}."
        )

    @staticmethod
    def bspline_clamped_knots(stations: List[float], degree: int) -> List[float]:
        """Clamped knot vector over *stations* via knot averaging (de Boor).

        Interior knots average consecutive stations so each control point's
        basis function peaks near its own station; the ``degree + 1`` end
        multiplicities pin the ends (basis exactly 1 at the end stations).

        Parameters:
            stations (list): Strictly increasing parameter values, one per
                control point.
            degree (int): B-spline degree (>= 1, <= len(stations) - 1).

        Returns:
            (list) The clamped knot vector, length ``len(stations) + degree + 1``.
        """
        n = len(stations)
        interior = [
            sum(stations[j : j + degree]) / degree for j in range(1, n - degree)
        ]
        return (
            [stations[0]] * (degree + 1)
            + interior
            + [stations[-1]] * (degree + 1)
        )

    @staticmethod
    def bspline_basis(
        knots: List[float], span: int, degree: int, s: float
    ) -> List[float]:
        """The non-zero B-spline basis values ``N[span - degree .. span]`` at
        *s* (Cox-de Boor recurrence, The NURBS Book A2.2).

        Parameters:
            knots (list): Knot vector (e.g. from :meth:`bspline_clamped_knots`).
            span (int): Knot span index containing *s*.
            degree (int): B-spline degree.
            s (float): Parameter value.

        Returns:
            (list) ``degree + 1`` basis values; they sum to 1 (partition of
            unity) inside the domain.
        """
        N = [1.0] + [0.0] * degree
        left = [0.0] * (degree + 1)
        right = [0.0] * (degree + 1)
        for j in range(1, degree + 1):
            left[j] = s - knots[span + 1 - j]
            right[j] = knots[span + j] - s
            saved = 0.0
            for r in range(j):
                temp = N[r] / (right[r + 1] + left[j - r])
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            N[j] = saved
        return N

    @staticmethod
    def ricker(x: float) -> float:
        """Ricker (Mexican-hat) wavelet — a unit ridge flanked by two balanced
        troughs (the negated 2nd derivative of a gaussian).

        It integrates to zero, so as a bump/crease/fold cross-section it
        displaces *out* at the crest and *in* to either side — mean-preserving,
        rather than one-sided. Peak is ``1`` at ``x == 0``; zero crossings at
        ``x == ±1``.

        Parameters:
            x (float): Distance from the crest, in half-width units.

        Returns:
            float: Wavelet value.
        """
        xx = x * x
        return (1.0 - xx) * math.exp(-0.5 * xx)

    @staticmethod
    def catenary(t: float, tension: float) -> float:
        """Normalized catenary (``cosh``) profile across a span.

        Returns ``1`` at the span center (``t == 0``) and ``0`` at the supports
        (``|t| == 1``) — the true curve a chain/cable/cloth assumes hung between
        two points. ``tension`` is the shape parameter: ``→0`` degenerates to a
        parabola, larger values give the deeper, more V-shaped sag of a slack,
        heavy line. The peak is always ``1`` so absolute depth is controlled by
        the caller.

        Parameters:
            t (float): Centered span coordinate (``-1``..``1``; ``0`` = center).
            tension (float): Catenary shape parameter (clamped to ``0``..``50``).

        Returns:
            float: Profile value in ``0``..``1``.
        """
        t = -1.0 if t < -1.0 else (1.0 if t > 1.0 else t)
        tension = 0.0 if tension < 0.0 else (50.0 if tension > 50.0 else tension)
        if tension <= 1e-6:
            return 1.0 - t * t  # parabolic limit
        ct = math.cosh(tension)
        return (ct - math.cosh(tension * t)) / (ct - 1.0)

    @classmethod
    def catenary_sag(
        cls,
        t: float,
        tension: float,
        round_amount: float = 0.0,
        gather: float = 0.0,
    ) -> float:
        """Catenary sag profile, optionally rounded / gathered at the supports.

        A pure catenary meets each support with a non-zero slope, so adjacent
        spans form a sharp cusp there. Blending toward ``sin²(πs)`` (zero slope
        at both ends) rounds that cusp into a smooth dome. The center peak stays
        ``1`` either way.

        ``gather`` adds an orthogonal *push-pull* lobe at each support — the
        ``ring²·(2.4 − 3.4·ring)`` shape (``ring = cos²(πs)``): the profile
        lifts **above** the baseline right at the support (a gathered-header
        pucker rising above the rail — the *push*) and **dips** just inside it
        as the slack falls off (the *pull*), easing back to ``0`` by mid-span so
        the center sag is untouched. Independent of ``round_amount``.

        Parameters:
            t (float): Centered span coordinate (``-1``..``1``; ``0`` = center).
            tension (float): Catenary shape parameter (see :meth:`catenary`).
            round_amount (float): ``0``..``1`` blend — ``0`` keeps the crisp
                catenary, ``1`` is fully rounded.
            gather (float): ``≥0`` push-pull overshoot at the supports (``0`` =
                none). At ``1`` the support lifts a full sag-unit above the
                baseline with a ``~0.18`` sag dip just inside.

        Returns:
            float: Profile value (``0``..``1`` for ``gather == 0``; dips below
                ``0`` near the supports when ``gather > 0``).
        """
        cat = cls.catenary(t, tension)
        if round_amount > 0.0:
            s = (t + 1.0) * 0.5  # back to 0..1 within the span
            rounded = math.sin(math.pi * s) ** 2
            cat += (rounded - cat) * min(1.0, round_amount)
        if gather > 0.0:
            ring = math.cos(math.pi * (t + 1.0) * 0.5) ** 2  # 1 at supports, 0 center
            cat += gather * ring * ring * (2.4 - 3.4 * ring)
        return cat

    @staticmethod
    def evaluate_sampled_progress(
        time_value: float,
        sample_times: Sequence[float],
        progress: Sequence[float],
        tolerance: float = 1e-6,
    ) -> float:
        """Interpolate normalized progress from sampled time/progress pairs.

        Parameters:
            time_value (float): The time at which to evaluate progress.
            sample_times (Sequence[float]): Monotonically increasing sample times.
            progress (Sequence[float]): Normalized progress values mapped to sample times.
            tolerance (float): Comparison tolerance for matching sample times exactly.

        Returns:
            float: The interpolated progress value.
        """
        if not sample_times or not progress:
            return 0.0

        limit = min(len(sample_times), len(progress))
        if limit == 0:
            return 0.0

        if time_value <= sample_times[0]:
            return float(progress[0])
        if time_value >= sample_times[limit - 1]:
            return float(progress[limit - 1])

        index = bisect_left(sample_times, time_value, 0, limit)

        if index < limit and math.isclose(
            sample_times[index], time_value, abs_tol=tolerance
        ):
            return float(progress[index])

        prev_index = max(0, index - 1)
        next_index = min(limit - 1, index)

        t0 = sample_times[prev_index]
        t1 = sample_times[next_index]
        p0 = progress[prev_index]
        p1 = progress[next_index]

        if math.isclose(t1, t0, abs_tol=tolerance):
            return float(p1)

        ratio = (time_value - t0) / (t1 - t0)
        return float(p0 + (p1 - p0) * ratio)

    @staticmethod
    def generate_geometric_sequence(
        base_value: int, terms: int, common_ratio: float = 2.0
    ) -> List[int]:
        """Generate a geometric sequence.

        Parameters:
            base_value (int): The initial value of the sequence.
            terms (int): The number of terms in the sequence.
            common_ratio (float, optional): The common ratio of the geometric sequence. Defaults to 2.0.

        Returns:
            List[int]: A list of values in the geometric sequence.

        Example:
            >>> generate_geometric_sequence(2, 5)
            [2, 4, 8, 16, 32]

            >>> generate_geometric_sequence(3, 4, 3.0)
            [3, 9, 27, 81]

        A geometric sequence is a sequence of numbers where each term after the first is
        found by multiplying the previous term by a fixed, non-zero number called the
        common ratio. In this function, the base value is the starting number of the
        sequence, the terms parameter specifies the number of elements in the sequence,
        and the common ratio is the fixed multiplier.
        """
        sequence = [base_value * common_ratio ** (n - 1) for n in range(1, terms + 1)]
        return [int(round(value)) for value in sequence]

    @staticmethod
    def remap(
        value: Union[float, List[Any], Tuple[Any, ...], "np.ndarray"],
        old_range: Tuple[float, float],
        new_range: Tuple[float, float],
        clamp: bool = False,
    ) -> Union[float, List[Any], Tuple[Any, ...], "np.ndarray"]:
        """Remaps a value, list, or tuple of varying sizes from one range to another.

        Parameters:
            value: The value to remap.
            old_range: The original range of the value.
            new_range: The new range to remap the value to.
            clamp: Whether to clamp the remapped value within the new range.

        Returns:
            The remapped value, list, or tuple.
        """
        import numpy as np

        old_min, old_max = map(float, old_range)
        new_min, new_max = map(float, new_range)

        if old_min == old_max:
            raise ValueError(
                "[remap] old_range min and max cannot be the same (division by zero)."
            )

        scale = (new_max - new_min) / (old_max - old_min)
        offset = new_min - old_min * scale

        # Ordered clamp bounds so a descending target range (e.g. (255, 0)) still
        # clamps to the actual interval instead of collapsing to one endpoint.
        clamp_lo, clamp_hi = (
            (new_min, new_max) if new_min <= new_max else (new_max, new_min)
        )

        def process_element(element: Any) -> Any:
            """Recursively remaps individual elements while preserving structure."""
            if isinstance(element, (int, float)):  # Single number
                remapped = element * scale + offset
                return max(min(remapped, clamp_hi), clamp_lo) if clamp else remapped
            elif isinstance(element, np.ndarray):  # NumPy array fix
                remapped = element.astype(np.float64) * scale + offset
                return np.clip(remapped, clamp_lo, clamp_hi) if clamp else remapped
            elif isinstance(element, (list, tuple)):  # Nested list or tuple
                return type(element)(process_element(e) for e in element)
            return element  # Return unchanged if not a number

        return process_element(value)

    @staticmethod
    def point_segment_distance(
        p: Sequence[float], a: Sequence[float], b: Sequence[float]
    ) -> float:
        """Perpendicular distance from point ``p`` to the segment ``a``-``b``.

        Works in any dimension (uses the shortest common length of the inputs);
        clamps to the segment endpoints, so a point past an end measures to that
        end rather than the infinite line.

        Parameters:
            p (Sequence[float]): The query point.
            a (Sequence[float]): Segment start.
            b (Sequence[float]): Segment end.

        Returns:
            float: The Euclidean distance from ``p`` to the closest point on the
            segment.
        """
        n = min(len(p), len(a), len(b))
        ab = [b[i] - a[i] for i in range(n)]
        ap = [p[i] - a[i] for i in range(n)]
        denom = sum(c * c for c in ab)
        if denom < 1e-24:  # degenerate segment -> distance to the point ``a``
            return sum(c * c for c in ap) ** 0.5
        t = sum(ap[i] * ab[i] for i in range(n)) / denom
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        return sum((ap[i] - t * ab[i]) ** 2 for i in range(n)) ** 0.5

    @staticmethod
    def nearest_power_of_two(value: int) -> int:
        """Finds the nearest power of two for a given integer without using the math module.

        Parameters:
            value (int): The input value.

        Returns:
            int: The nearest power of two.
        """
        if value <= 0:
            return 1  # Default to smallest power of two (avoid errors)

        # Start with the smallest power of two
        lower_power = 1
        while lower_power * 2 <= value:
            lower_power *= 2  # Double until reaching or exceeding the value

        # Find the next higher power of two
        upper_power = lower_power * 2

        # Return the closest of the two
        return (
            lower_power
            if (value - lower_power) < (upper_power - value)
            else upper_power
        )

    @staticmethod
    def is_close_to_whole(value: float, tolerance: float = 1e-4) -> bool:
        """Check if a float value is close to a whole number within tolerance.

        Parameters:
            value (float): The value to check.
            tolerance (float): The maximum difference from a whole number to be considered close.
                Default is 1e-4 (0.0001).

        Returns:
            bool: True if the value is within tolerance of a whole number, False otherwise.

        Example:
            is_close_to_whole(10.0) #returns: True
            is_close_to_whole(10.00001) #returns: True
            is_close_to_whole(10.5) #returns: False
            is_close_to_whole(9.9999) #returns: True
        """
        return abs(value - round(value)) <= tolerance

    @staticmethod
    def round_value(
        value: float,
        mode: str = "none",
        max_distance: float = 1.5,
    ) -> Union[int, float]:
        """General-purpose rounding function with multiple modes.

        Provides a unified interface for various rounding strategies, from precise
        preservation to aesthetic snapping.

        Parameters:
            value (float): The value to round.
            mode (str): Rounding mode to use. Options:
                - "none": No rounding, return value as-is (default)
                - "nearest": Round to nearest whole number (standard rounding)
                - "floor": Always round down
                - "ceil": Always round up
                - "half_up": Round .5 and above up, below .5 down
                - "preferred": Round to aesthetically pleasing numbers when very close.
                  Examples: 24→25, 19→20, 99→100. Conservative approach.
                - "aggressive_preferred": Round to preferred numbers even when farther away.
                  Examples: 48.x→50, 73.x→75, 88.x→90, 23.x→25, 7.x→10.
            max_distance (float): Maximum distance for "preferred" mode to consider a
                preferred number. Only used when mode="preferred". Default is 1.5.

        Returns:
            Union[int, float]: The rounded value (int for most modes, float for "none").

        Example:
            round_value(10.7, mode="nearest") #returns: 11
            round_value(10.7, mode="floor") #returns: 10
            round_value(10.5, mode="half_up") #returns: 11
            round_value(24.8, mode="preferred") #returns: 25
            round_value(48.5, mode="aggressive_preferred") #returns: 50
            round_value(10.7, mode="none") #returns: 10.7
        """
        if mode == "none":
            return value
        elif mode == "nearest":
            return int(round(value))
        elif mode == "floor":
            return int(math.floor(value))
        elif mode == "ceil":
            return int(math.ceil(value))
        elif mode == "half_up":
            return int(math.floor(value + 0.5))
        elif mode == "preferred":
            return MathUtils.round_to_preferred(value, max_distance)
        elif mode == "aggressive_preferred":
            return MathUtils.round_to_aggressive_preferred(value)
        else:
            raise ValueError(
                f"Invalid rounding mode: '{mode}'. "
                f"Valid options: 'none', 'nearest', 'floor', 'ceil', 'half_up', 'preferred', 'aggressive_preferred'"
            )

    @staticmethod
    def round_to_preferred(value: float, max_distance: float = 1.5) -> int:
        """Round to aesthetically pleasing 'round' numbers (conservative approach).

        Only rounds to preferred numbers if they're within max_distance frames.
        Preferred numbers in order of priority:
        - Multiples of 100 (0, 100, 200, ...)
        - Multiples of 50 (50, 150, 250, ...)
        - Multiples of 25 (25, 75, 125, ...)
        - Multiples of 20 (20, 40, 60, 80, ...)
        - Multiples of 10 (10, 30, 70, 90, ...)
        - Multiples of 5 (5, 15, 35, 45, ...)

        Parameters:
            value (float): The value to round
            max_distance (float): Maximum distance from value to consider a preferred number.
                Default is 1.5, meaning only round if a preferred number is very close.

        Returns:
            int: The rounded value to the nearest preferred number

        Example:
            round_to_preferred(24.8) #returns: 25
            round_to_preferred(19.3) #returns: 20
            round_to_preferred(99.5) #returns: 100
            round_to_preferred(48.5) #returns: 49 (not 50, too far away with default max_distance)
        """
        # Handle exact integers
        rounded = round(value)
        if value == rounded:
            return int(rounded)

        # Candidate preferred numbers: the bracketing multiples of each tier.
        floor_val = math.floor(value)
        candidates = set()
        for step in (100, 50, 25, 20, 10, 5):
            candidates.add((floor_val // step) * step)
            candidates.add((floor_val // step + 1) * step)

        candidates = [c for c in candidates if abs(c - value) <= max_distance]
        if not candidates:
            return int(round(value))  # no preferred number close enough

        # Closest candidate; ties prefer the "rounder" number
        # (100 > 50 > 25 > 20 > 10 > 5), then the higher value.
        min_distance = min(abs(c - value) for c in candidates)
        tied = [c for c in candidates if abs(c - value) == min_distance]
        if len(tied) > 1:
            for multiple in (100, 50, 25, 20, 10, 5):
                for t in tied:
                    if t % multiple == 0:
                        return int(t)
            return int(max(tied))
        return int(tied[0])

    @classmethod
    def round_to_aggressive_preferred(cls, value: float) -> int:
        """Round to aesthetically pleasing 'round' numbers (aggressive approach).

        :meth:`round_to_preferred` with a wide search radius (10) — rounds to
        preferred numbers even when farther away.
        Examples: 48.x→50, 73.x→75, 88.x→90, 23.x→25, 7.x→10

        Parameters:
            value (float): The value to round

        Returns:
            int: The rounded value to the nearest preferred number

        Example:
            round_to_aggressive_preferred(48.5) #returns: 50
            round_to_aggressive_preferred(73.2) #returns: 75
            round_to_aggressive_preferred(88.9) #returns: 90
            round_to_aggressive_preferred(23.4) #returns: 25
            round_to_aggressive_preferred(7.8) #returns: 10
        """
        return cls.round_to_preferred(value, max_distance=10)

    @staticmethod
    def calculate_rotation_distance(
        r1_vals: Tuple[float, float, float],
        r2_vals: Tuple[float, float, float],
        bbox_points: Optional[List[Any]] = None,
        om_module: Optional[Any] = None,
    ) -> float:
        """Calculate the effective rotation distance between two Euler rotations.

        This method calculates the arc length traveled by the furthest point on an object's
        bounding box as it rotates from r1 to r2. This provides a "surface speed" equivalent
        for rotation, useful for motion-based retiming.

        If OpenMaya (om_module) and bbox_points are provided, it uses accurate quaternion
        math and axis-projection to find the maximum arc length.
        If not, it falls back to a simple Euler distance approximation.

        Parameters:
            r1_vals: Start rotation in degrees (x, y, z).
            r2_vals: End rotation in degrees (x, y, z).
            bbox_points: List of object-space bounding box points (MVector or similar).
                         Required for accurate surface speed calculation.
            om_module: The OpenMaya module or equivalent (passed to avoid dependency).
                       Required for accurate quaternion math.

        Returns:
            float: The calculated rotation distance (arc length).
        """
        dist_rot = 0.0
        deg_to_rad = math.pi / 180.0

        if om_module and bbox_points:
            try:
                # Convert Euler (degrees) to MQuaternion
                e1 = om_module.MEulerRotation(
                    math.radians(r1_vals[0]),
                    math.radians(r1_vals[1]),
                    math.radians(r1_vals[2]),
                )
                e2 = om_module.MEulerRotation(
                    math.radians(r2_vals[0]),
                    math.radians(r2_vals[1]),
                    math.radians(r2_vals[2]),
                )

                q1 = e1.asQuaternion()
                q2 = e2.asQuaternion()

                # Calculate relative rotation: q_diff = q1.inverse() * q2
                # This gives the rotation needed to go from q1 to q2, in the local frame of q1
                q_diff = q1.inverse() * q2

                # Extract axis and angle
                # getAxisAngle returns (MVector axis, float angle)
                axis, angle_rad = q_diff.getAxisAngle()

                # Normalize angle to [-pi, pi] for shortest path
                if angle_rad > math.pi:
                    angle_rad -= 2 * math.pi
                angle_rad = abs(angle_rad)

                if angle_rad > 1e-6:
                    # Calculate effective radius: max distance of any corner from the rotation axis
                    # Distance from point P to line (origin, axis) is |P x axis| if axis is normalized
                    max_radius = 0.0
                    for point in bbox_points:
                        # Cross product magnitude gives distance to axis
                        # (assuming axis passes through origin, which is object center)
                        dist = (point ^ axis).length()
                        if dist > max_radius:
                            max_radius = dist

                    dist_rot = angle_rad * max_radius
                return dist_rot

            except Exception:
                # Fall through to fallback if OM fails
                pass

        # Fallback to simple Euler distance
        d_rx = abs(r2_vals[0] - r1_vals[0])
        d_ry = abs(r2_vals[1] - r1_vals[1])
        d_rz = abs(r2_vals[2] - r1_vals[2])
        angle_dist_deg = math.sqrt(d_rx * d_rx + d_ry * d_ry + d_rz * d_rz)

        # Fallback radius (average of dimensions or unit radius)
        avg_radius = 1.0
        if bbox_points:
            # Rough estimate from first point if available (assuming it's a vector-like object)
            try:
                avg_radius = bbox_points[0].length()
            except AttributeError:
                pass

        dist_rot = angle_dist_deg * deg_to_rad * avg_radius
        return dist_rot


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass

    # -----------------------------------------------------------------------------
    # Notes
    # -----------------------------------------------------------------------------
