# !/usr/bin/python
# coding=utf-8
from typing import List, Tuple

# from this package:
from pythontk.core_utils import CoreUtils


class MathUtils:
    """ """

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
            d = cls.get_distance(p, b)
            if lowest is None or (d < lowest if toward else d > lowest):
                result, lowest = (p, d)

        return result

    @staticmethod
    def get_distance(a: List[float], b: List[float]) -> float:
        """
        Calculate the distance between two points.

        Parameters:
            a (List[float]): A list of the first point's coordinates [x, y, z].
            b (List[float]): A list of the second point's coordinates [x, y, z].

        Returns:
            float: The distance between the two points.
        """
        return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5

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
    def lerp(a, b, t):
        """Perform linear interpolation between two points.

        Linear interpolation is a method of curve fitting using linear
        polynomials to construct new data points within the range of
        a discrete set of known data points.

        Parameters:
            a (float or int): The first point, corresponds to the value at t=0.
            b (float or int): The second point, corresponds to the value at t=1.
            t (float): The interpolation parameter, should be in range [0.0, 1.0].
                       If t=0, it will return `a`, if t=1 it will return `b`.
                       Intermediate values of `t` will return a value somewhere between `a` and `b`.

        Returns:
            float: The interpolated value between `a` and `b`.
        """
        return a + (b - a) * t


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------


# deprecated ---------------------
# @classmethod
# def normalize(cls, vector, amount=1):
#   '''Normalize a vector

#   Parameters:
#       vector (vector) = The vector to normalize.
#       amount (float) = (1) Normalize standard. (value other than 0 or 1) Normalize using the given float value as desired length.

#   Returns:
#       (tuple)
#   '''
#   length = cls.get_magnitude(vector)
#   x, y, z = vector

#   result = (
#       x /length *amount,
#       y /length *amount,
#       z /length *amount
#   )

#   return result

# @classmethod
# def cross_product(cls, v1, v2, normalize_input=False, normalizeResult=False):
#   '''Given two float arrays of 3 values each, this procedure returns
#   the cross product of the two arrays as a float array of size 3.

#   :Parmeters:
#       v1 (list): The first 3 point vector.
#       v2 (list): The second 3 point vector.
#       normalize_input (bool): Normalize v1, v2 before calculating the point float list.
#       normalizeResult (bool): Normalize the return value.

#   Returns:
#       (tuple) The cross product of the two vectors.

#   Example: cross_product((1, 2, 3), (1, 1, -1)) #returns: (-5, 4, -1)
#   Example: cross_product((1, 2, 3), (1, 1, -1), True) #returns: (-0.7715167498104597, 0.6172133998483678, -0.15430334996209194)
#   Example: cross_product((1, 2, 3), (1, 1, -1), False, True) #returns: (-0.7715167498104595, 0.6172133998483676, -0.1543033499620919)
#   '''
#   if normalize_input: #normalize the input vectors
#       v1 = cls.normalize(v1)
#       v2 = cls.normalize(v2)

#   cross = [ #the cross product
#       v1[1]*v2[2] - v1[2]*v2[1],
#       v1[2]*v2[0] - v1[0]*v2[2],
#       v1[0]*v2[1] - v1[1]*v2[0]
#   ]

#   if normalizeResult: #normalize the cross product result
#       cross = cls.normalize(cross)

#   return tuple(cross)
