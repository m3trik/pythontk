#!/usr/bin/python
# coding=utf-8
"""Quaternion: one rotation convention -- (x, y, z, w), ``a (x) b`` applies b
first, rows for column vectors, Euler degrees in Maya's rotate order."""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythontk import Quaternion as Q  # noqa: E402

ORDERS = ("xyz", "yzx", "zxy", "xzy", "yxz", "zyx")


def turn(a, b):
    """Degrees between two unit quaternions as rotations (sign-agnostic): the
    relative rotation's angle, by ``atan2`` -- ``acos`` of a dot product
    cannot resolve a turn under ~1e-6 degrees."""
    x, y, z, w = Q.multiply(Q.conjugate(a), b)
    return math.degrees(2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), abs(w)))


class QuaternionTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(31)

    def random_unit(self):
        return Q.normalize([self.rng.gauss(0.0, 1.0) for _ in range(4)])

    def assertClose(self, got, want, places=9):
        self.assertEqual(len(got), len(want))
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=places, msg=f"{got} != {want}")

    # ------------------------------------------------------------ convention
    def test_a_product_applies_its_right_hand_rotation_first(self):
        for _ in range(50):
            a, b = self.random_unit(), self.random_unit()
            v = [self.rng.uniform(-3, 3) for _ in range(3)]
            self.assertClose(Q.rotate(Q.multiply(a, b), v), Q.rotate(a, Q.rotate(b, v)))

    def test_xyz_applies_x_first_as_maya_does(self):
        # (0, 1, 0) -rx 90-> (0, 0, 1) -ry 90-> (1, 0, 0).
        q = Q.from_euler("xyz", (90.0, 90.0, 0.0))
        self.assertClose(Q.rotate(q, (0.0, 1.0, 0.0)), (1.0, 0.0, 0.0))
        # The same angles in zyx apply y first: (0, 1, 0) stays, then x.
        q = Q.from_euler("zyx", (90.0, 90.0, 0.0))
        self.assertClose(Q.rotate(q, (0.0, 1.0, 0.0)), (0.0, 0.0, 1.0))

    def test_a_matrix_is_rows_for_column_vectors(self):
        q = Q.from_axis_angle(2, 90.0)  # +X -> +Y
        m = Q.to_matrix(q)
        self.assertClose([m[r][0] for r in range(3)], (0.0, 1.0, 0.0))

    def test_an_axis_index_is_exact_and_a_vector_is_normalized(self):
        self.assertEqual(Q.from_axis_angle(1, -40.0)[0::2], (0.0, 0.0))
        self.assertClose(
            Q.from_axis_angle((0.0, 5.0, 0.0), -40.0), Q.from_axis_angle(1, -40.0)
        )
        self.assertEqual(Q.from_axis_angle((0.0, 0.0, 0.0), 30.0), Q.IDENTITY)

    def test_a_conjugate_undoes_and_a_degenerate_quaternion_is_no_turn(self):
        q = self.random_unit()
        self.assertClose(Q.multiply(q, Q.conjugate(q)), Q.IDENTITY)
        self.assertEqual(Q.normalize((0.0, 0.0, 0.0, 1e-13)), Q.IDENTITY)
        self.assertClose(Q.normalize((0.0, 0.0, 0.0, 2.0)), Q.IDENTITY)

    # ---------------------------------------------------------- round trips
    def test_every_order_round_trips_through_its_euler(self):
        for order in ORDERS:
            for _ in range(200):
                q = self.random_unit()
                back = Q.from_euler(order, Q.matrix_euler(order, Q.to_matrix(q)))
                self.assertLess(turn(q, back), 1e-6, order)

    def test_the_middle_angle_reads_in_range_and_its_twin_is_the_same_turn(self):
        for order in ORDERS:
            a, b, c = ("xyz".index(axis) for axis in order[::-1])
            e = Q.matrix_euler(order, Q.to_matrix(self.random_unit()))
            self.assertLessEqual(abs(e[b]), 90.0)
            twin = [0.0, 0.0, 0.0]
            twin[a], twin[b], twin[c] = e[a] + 180.0, 180.0 - e[b], e[c] + 180.0
            self.assertLess(
                turn(Q.from_euler(order, e), Q.from_euler(order, twin)), 1e-6
            )

    def test_a_matrix_round_trips_on_every_branch(self):
        """Shepperd's four branches: a positive trace, and each diagonal the
        largest -- half turns about x, y and z, and turns near them."""
        cases = [self.random_unit() for _ in range(100)]
        for axis in range(3):
            for degrees in (180.0, 179.0, -178.5):
                cases.append(Q.from_axis_angle(axis, degrees))
        for q in cases:
            back = Q.from_matrix(Q.to_matrix(q))
            self.assertLess(turn(q, back), 1e-6, q)
            self.assertAlmostEqual(math.sqrt(sum(v * v for v in back)), 1.0)

    # -------------------------------------------------------------- gimbal
    def test_gimbal_lock_zeroes_the_innermost_and_keeps_the_turn(self):
        for order in ORDERS:
            for middle in (90.0, -90.0):
                e = [self.rng.uniform(-170, 170) for _ in range(3)]
                e["xyz".index(order[1])] = middle
                q = Q.from_euler(order, e)
                got = Q.matrix_euler(order, Q.to_matrix(q))
                self.assertEqual(got["xyz".index(order[0])], 0.0, order)
                self.assertAlmostEqual(got["xyz".index(order[1])], middle, places=4)
                self.assertLess(turn(q, Q.from_euler(order, got)), 1e-4, order)

    # ------------------------------------------------------------ the filter
    def test_euler_near_takes_the_spelling_nearest_the_last_key(self):
        """The second triple and the 360 wraps; a first key as given."""
        self.assertEqual(
            Q.euler_near("xyz", (180.0, 90.0, 180.0), (0.0, 0.0, 0.0)), (0.0, 90.0, 0.0)
        )
        self.assertEqual(
            Q.euler_near("xyz", (-170.0, 0.0, 0.0), (175.0, 0.0, 0.0)), (190.0, 0.0, 0.0)
        )
        self.assertEqual(Q.euler_near("zxy", (400.0, 10.0, -5.0)), (400.0, 10.0, -5.0))

    def test_euler_near_keeps_a_sweep_through_gimbal_lock_on_its_axis(self):
        """At the lock only the inner and outer angles' sum or difference is
        fixed, and a reader puts the whole turn on one of them (this one's
        matrix_euler on the outer, Maya's MEulerRotation on the inner): a key
        takes the split the key before it carried, in every order, whichever
        axis the sweep rides and whichever one the reader chose."""
        for order in ORDERS:
            inner, middle, outer = ("xyz".index(c) for c in order)
            for lock in (90.0, -90.0):
                for carried in (inner, outer):
                    for reader_axis in (inner, outer):
                        with self.subTest(order=order, lock=lock, carried=carried, reader=reader_axis):
                            previous = None
                            # From inside the middle's range: a first key is
                            # taken as read.
                            for step in (-6.0, -3.0, 0.0, 3.0, 6.0)[:: 1 if lock > 0 else -1]:
                                want = [0.0, 0.0, 0.0]
                                want[middle], want[carried] = lock + step, 30.0
                                q = Q.from_euler(order, want)
                                read = list(Q.matrix_euler(order, Q.to_matrix(q)))
                                if step == 0.0 and reader_axis == inner:
                                    # Maya's split: the turn on the inner angle.
                                    k = (1 if lock > 0 else -1) * (
                                        1 if (middle - inner) % 3 == 1 else -1
                                    )
                                    read[inner], read[outer] = -k * read[outer], 0.0
                                    self.assertLess(turn(q, Q.from_euler(order, read)), 1e-9)
                                previous = Q.euler_near(order, read, previous)
                                self.assertClose(previous, want, places=6)

    # ------------------------------------------------------------- slerp
    def test_slerp_takes_the_shorter_arc_whichever_sign_it_is_given(self):
        a = Q.IDENTITY
        b = Q.from_axis_angle(2, 350.0)  # the long way round to -10 degrees
        half = Q.slerp(a, b, 0.5)
        self.assertLess(turn(half, Q.from_axis_angle(2, -5.0)), 1e-9)
        self.assertEqual(Q.slerp(a, b, 0.5), Q.slerp(a, tuple(-v for v in b), 0.5))

    def test_slerp_runs_end_to_end_at_unit_length(self):
        for _ in range(100):
            a, b = self.random_unit(), self.random_unit()
            self.assertLess(turn(Q.slerp(a, b, 0.0), a), 1e-6)
            self.assertLess(turn(Q.slerp(a, b, 1.0), b), 1e-6)
            mid = Q.slerp(a, b, 0.3)
            self.assertAlmostEqual(math.sqrt(sum(v * v for v in mid)), 1.0)
            # A third of the way: the turns either side split 3 : 7.
            self.assertAlmostEqual(turn(a, mid) * 7, turn(mid, b) * 3, places=5)

    def test_nearly_parallel_rotations_interpolate_linearly_at_unit_length(self):
        a = Q.from_axis_angle(0, 10.0)
        b = Q.from_axis_angle(0, 10.5)  # |dot| over PARALLEL
        self.assertGreater(sum(x * y for x, y in zip(a, b)), Q.PARALLEL)
        mid = Q.slerp(a, b, 0.5)
        self.assertLess(turn(mid, Q.from_axis_angle(0, 10.25)), 1e-4)
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in mid)), 1.0)

    def test_an_arc_reads_as_slerp_does_bit_for_bit(self):
        for _ in range(50):
            a, b = self.random_unit(), self.random_unit()
            arc = Q.arc(a, b)
            for u in (0.0, 0.25, 0.5, 0.9, 1.0):
                self.assertEqual(arc(u), Q.slerp(a, b, u))


if __name__ == "__main__":
    unittest.main()
