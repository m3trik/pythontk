# !/usr/bin/python
# coding=utf-8
from bisect import bisect_left
import math
from typing import List, Tuple, Union, Callable, Sequence, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# from this package:
from pythontk.core_utils._core_utils import CoreUtils
from pythontk.core_utils.help_mixin import HelpMixin


class MathUtils(HelpMixin):
    """ """

    @staticmethod
    def linear_sum_assignment(
        cost_matrix: Sequence[Sequence[float]],
        maximize: bool = False,
    ) -> Tuple[List[int], List[int]]:
        """Solve the linear sum assignment problem (Hungarian algorithm).

        This is a lightweight, dependency-free alternative to
        ``scipy.optimize.linear_sum_assignment``.

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
        scaling_factor = Decimal(
            10**places
        )  # Create a scaling factor as a Decimal object

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
            result = cls.normalize(result, normalize)

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

        result = acos(cls.dot_product(v1, v2) / (length(v1) * length(v2)))

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

        Example:
            get_two_sides_of_asa_triangle(60, 60, 100) #returns: (100.00015320566493, 100.00015320566493)
        """
        from math import sin, radians

        if unit == "degrees":
            a1, a2 = radians(a1), radians(a2)

        a3 = 3.14159 - a1 - a2

        result = ((s / sin(a3)) * sin(a1), (s / sin(a3)) * sin(a2))

        return result

    @classmethod
    def xyz_rotation(cls, theta, axis, rotation=[], degree=False):
        """Get the rotation about the X,Y,Z axes (in rotation) given
        an angle for rotation (in radians) and an axis about which to
        do the rotation.

        Parameters:
            theta (float):The angular position of a vector in radians.
            axis (tuple): The rotation axis given as float values (x,y,z).
            rotation (list):
            degree (bool): Convert the radian result to degrees.

        Returns:
            (tuple) 3 point rotation.

        Example:
            xyz_rotation(2, (0, 1, 0)) #returns: [3.589792907376932e-09, 1.9999999964102069, 3.589792907376932e-09]
            xyz_rotation(2, (0, 1, 0), [], True) #returns: [0.0, 114.59, 0.0]
        """
        from math import cos, sin, sqrt, atan2, degrees

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
            pi = 3.14159265

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
    def lerp(start: float, end: float, t: float) -> float:
        """Linear interpolation between two values.

        Parameters:
            start (float): Start value
            end (float): End value
            t (float): Interpolation factor (0.0 to 1.0)

        Returns:
            float: Interpolated value
        """
        return start + t * (end - start)

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

        def process_element(element: Any) -> Any:
            """Recursively remaps individual elements while preserving structure."""
            if isinstance(element, (int, float)):  # Single number
                remapped = element * scale + offset
                return max(min(remapped, new_max), new_min) if clamp else remapped
            elif isinstance(element, np.ndarray):  # NumPy array fix
                remapped = element.astype(np.float64) * scale + offset
                return np.clip(remapped, new_min, new_max) if clamp else remapped
            elif isinstance(element, (list, tuple)):  # Nested list or tuple
                return type(element)(process_element(e) for e in element)
            return element  # Return unchanged if not a number

        return process_element(value)

    @staticmethod
    def calculate_curve_length(centerline_points: List[List[float]]) -> float:
        """Calculates the total length of the centerline path.

        Parameters:
            centerline_points (List[List[float]]): The list of points along the centerline.

        Returns:
            float: The total length of the centerline path.
        """
        length = 0
        for i in range(1, len(centerline_points)):
            p1 = centerline_points[i - 1]
            p2 = centerline_points[i]
            length += sum([(p2[j] - p1[j]) ** 2 for j in range(3)]) ** 0.5
        return length

    @staticmethod
    def get_point_on_centerline(
        centerline_points: List[List[float]], param: float
    ) -> List[float]:
        """Returns the interpolated point along the centerline.

        Parameters:
            centerline_points (List[List[float]]): The list of points along the centerline.
            param (float): The parameter value along the centerline path.

        Returns:
            List[float]: The interpolated point along the centerline.
        """
        total_length = len(centerline_points) - 1  # Total number of segments
        param = max(0, min(param, 1))  # Clamp the param value between 0 and 1
        index = int(param * total_length)

        if index == total_length:
            return centerline_points[-1]

        p1 = centerline_points[index]
        p2 = centerline_points[index + 1]
        interp_point = [
            p1[0] + (p2[0] - p1[0]) * (param * (len(centerline_points) - 1) - index),
            p1[1] + (p2[1] - p1[1]) * (param * (len(centerline_points) - 1) - index),
            p1[2] + (p2[2] - p1[2]) * (param * (len(centerline_points) - 1) - index),
        ]
        return interp_point

    @classmethod
    def dist_points_along_centerline(
        cls,
        centerline: List[List[float]],
        num_points: int,
        reverse: bool = False,
        interpolation: Callable[[List[List[float]], float], List[float]] = None,
        start_offset: float = 0.0,
        end_offset: float = 0.0,
    ) -> List[List[float]]:
        """Distributes points evenly along the centerline with optional offsets and custom interpolation.

        Parameters:
            centerline (List[List[float]]): The list of points along the centerline.
            num_points (int): The number of points to distribute along the centerline.
            reverse (bool): Reverse the order of the points.
            interpolation (Callable): The interpolation function to use.
            start_offset (float): The offset from the start of the centerline.
            end_offset (float): The offset from the end of the centerline.

        Returns:
            List[List[float]]: The evenly distributed points along the centerline.
        """
        if start_offset < 0 or end_offset < 0 or start_offset + end_offset >= 1:
            raise ValueError("Invalid start or end offset values.")

        if interpolation is None:
            interpolation = cls.get_point_on_centerline

        positions = [
            interpolation(
                centerline,
                cls.remap(i, (0, num_points - 1), (start_offset, 1 - end_offset)),
            )
            for i in range(num_points)
        ]
        return positions[::-1] if reverse else positions

    @staticmethod
    def arrange_points_as_path(
        points: List[List[float]],
        closed_path: bool = False,
        distance_metric: Optional[Callable[[List[float], List[float]], float]] = None,
    ) -> List[List[float]]:
        """Orders a list of points to form a continuous path.

        Parameters:
            points (List): The list of points to order.
            closed_path (bool): Whether to treat the path as a closed loop.
            distance_metric

        Returns:
            List[pm.datatypes.Point]: Ordered list of points forming a continuous path.
        """
        if not points:
            return []

        if distance_metric is None:
            distance_metric = lambda p1, p2: (p1 - p2).length()

        sorted_points = [points.pop(0)]
        while points:
            last_point = sorted_points[-1]
            next_point = min(points, key=lambda p: distance_metric(p, last_point))
            sorted_points.append(next_point)
            points.remove(next_point)

        if closed_path:
            sorted_points.append(sorted_points[0])

        return sorted_points

    @staticmethod
    def smooth_points(
        points: Sequence[Union[tuple, object]], window_size: int = 1
    ) -> list:
        """Apply a moving average to smooth a sequence of 3D points.

        Parameters:
            points: A sequence of (x, y, z) tuples or dt.Point objects.
            window_size: The number of points to include in each averaging window.

        Returns:
            A list of smoothed points in the same format as the input.
        """
        if not points or window_size <= 1:
            return list(points)

        n = len(points)
        window_size = min(window_size, n)
        smoothed_points = []

        is_dt_point = hasattr(points[0], "__add__") and hasattr(
            points[0], "__truediv__"
        )

        # Initialize running sum
        running_sum = points[0] * 0 if is_dt_point else (0.0, 0.0, 0.0)
        count = 0

        for i in range(n):
            # Add new point to the running sum
            new_point = points[i]
            running_sum = (
                running_sum + new_point
                if is_dt_point
                else tuple(running_sum[j] + new_point[j] for j in range(3))
            )
            count += 1

            # Remove the old point when window slides forward
            if i >= window_size:
                old_point = points[i - window_size]
                running_sum = (
                    running_sum - old_point
                    if is_dt_point
                    else tuple(running_sum[j] - old_point[j] for j in range(3))
                )
                count -= 1

            # Compute and store the smoothed point
            avg_point = (
                running_sum / count
                if is_dt_point
                else tuple(v / count for v in running_sum)
            )
            smoothed_points.append(avg_point)

        return smoothed_points

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
        import math

        # Handle exact integers
        rounded = round(value)
        if value == rounded:
            return int(rounded)

        # Define preferred numbers and their intervals
        candidates = []

        # Generate candidates based on scale
        floor_val = math.floor(value)
        ceil_val = math.ceil(value)

        # Add multiples of 100
        hundred_floor = (floor_val // 100) * 100
        hundred_ceil = ((floor_val // 100) + 1) * 100
        candidates.extend([hundred_floor, hundred_ceil])

        # Add multiples of 50
        fifty_floor = (floor_val // 50) * 50
        fifty_ceil = ((floor_val // 50) + 1) * 50
        candidates.extend([fifty_floor, fifty_ceil])

        # Add multiples of 25
        twentyfive_floor = (floor_val // 25) * 25
        twentyfive_ceil = ((floor_val // 25) + 1) * 25
        candidates.extend([twentyfive_floor, twentyfive_ceil])

        # Add multiples of 20
        twenty_floor = (floor_val // 20) * 20
        twenty_ceil = ((floor_val // 20) + 1) * 20
        candidates.extend([twenty_floor, twenty_ceil])

        # Add multiples of 10
        ten_floor = (floor_val // 10) * 10
        ten_ceil = ((floor_val // 10) + 1) * 10
        candidates.extend([ten_floor, ten_ceil])

        # Add multiples of 5
        five_floor = (floor_val // 5) * 5
        five_ceil = ((floor_val // 5) + 1) * 5
        candidates.extend([five_floor, five_ceil])

        # Remove duplicates and filter to conservative range
        candidates = list(set(candidates))
        candidates = [c for c in candidates if abs(c - value) <= max_distance]

        if not candidates:
            # Fallback to simple rounding
            return int(round(value))

        # Find the closest candidate
        closest = min(candidates, key=lambda x: abs(x - value))

        # If there's a tie, prefer the higher number for round numbers
        distances = [(c, abs(c - value)) for c in candidates]
        min_distance = min(d[1] for d in distances)
        tied = [c for c, d in distances if d == min_distance]

        if len(tied) > 1:
            # Prefer rounder numbers in ties
            # Priority: 100 > 50 > 25 > 20 > 10 > 5
            for multiple in [100, 50, 25, 20, 10, 5]:
                for t in tied:
                    if t % multiple == 0:
                        return int(t)
            # If still tied, prefer the higher number
            return int(max(tied))

        return int(closest)

    @staticmethod
    def round_to_aggressive_preferred(value: float) -> int:
        """Round to aesthetically pleasing 'round' numbers (aggressive approach).

        Rounds to preferred numbers even when farther away (within ~10 frames).
        Examples: 48.x→50, 73.x→75, 88.x→90, 23.x→25, 7.x→10

        Preferred numbers in order of priority:
        - Multiples of 100 (0, 100, 200, ...)
        - Multiples of 50 (50, 150, 250, ...)
        - Multiples of 25 (25, 75, 125, ...)
        - Multiples of 20 (20, 40, 60, 80, ...)
        - Multiples of 10 (10, 30, 70, 90, ...)
        - Multiples of 5 (5, 15, 35, 45, ...)

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
        import math

        # Handle exact integers
        rounded = round(value)
        if value == rounded:
            return int(rounded)

        # Define preferred numbers and their intervals
        candidates = []

        # Generate candidates based on scale
        floor_val = math.floor(value)
        ceil_val = math.ceil(value)

        # Add multiples of 100
        hundred_floor = (floor_val // 100) * 100
        hundred_ceil = ((floor_val // 100) + 1) * 100
        candidates.extend([hundred_floor, hundred_ceil])

        # Add multiples of 50
        fifty_floor = (floor_val // 50) * 50
        fifty_ceil = ((floor_val // 50) + 1) * 50
        candidates.extend([fifty_floor, fifty_ceil])

        # Add multiples of 25
        twentyfive_floor = (floor_val // 25) * 25
        twentyfive_ceil = ((floor_val // 25) + 1) * 25
        candidates.extend([twentyfive_floor, twentyfive_ceil])

        # Add multiples of 20
        twenty_floor = (floor_val // 20) * 20
        twenty_ceil = ((floor_val // 20) + 1) * 20
        candidates.extend([twenty_floor, twenty_ceil])

        # Add multiples of 10
        ten_floor = (floor_val // 10) * 10
        ten_ceil = ((floor_val // 10) + 1) * 10
        candidates.extend([ten_floor, ten_ceil])

        # Add multiples of 5
        five_floor = (floor_val // 5) * 5
        five_ceil = ((floor_val // 5) + 1) * 5
        candidates.extend([five_floor, five_ceil])

        # Remove duplicates and filter to aggressive range (within 10 frames)
        candidates = list(set(candidates))
        candidates = [c for c in candidates if abs(c - value) <= 10]

        if not candidates:
            # Fallback to simple rounding
            return int(round(value))

        # Find the closest candidate
        closest = min(candidates, key=lambda x: abs(x - value))

        # If there's a tie, prefer the higher number for round numbers
        distances = [(c, abs(c - value)) for c in candidates]
        min_distance = min(d[1] for d in distances)
        tied = [c for c, d in distances if d == min_distance]

        if len(tied) > 1:
            # Prefer rounder numbers in ties
            # Priority: 100 > 50 > 25 > 20 > 10 > 5
            for multiple in [100, 50, 25, 20, 10, 5]:
                for t in tied:
                    if t % multiple == 0:
                        return int(t)
            # If still tied, prefer the higher number
            return int(max(tied))

        return int(closest)

    @staticmethod
    def hash_points(points, precision=4):
        """Hash the given list of point values.

        Parameters:
            points (list): A list of point values as tuples.
            precision (int): Determines the number of decimal places that are retained
                    in the fixed-point representation. For example, with a value of 4, the
                    fixed-point representation would retain 4 decimal places.

        Returns:
            (list) list(s) of hashed tuples.

        Example:
            hash_points([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]) #returns: [hash values]
            hash_points([[(1.0, 2.0, 3.0)], [(4.0, 5.0, 6.0)]]) #returns: [[hash values], [hash values]]
        """
        nested = CoreUtils.nested_depth(points) > 1
        sets = points if nested else [points]

        def clamp(p):
            return int(p * 10**precision)

        result = []
        for pset in sets:
            result.append([hash(tuple(map(clamp, i))) for i in pset])
        return CoreUtils.format_return(result, nested)

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
            om_module: The maya.api.OpenMaya module (passed to avoid dependency).
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
