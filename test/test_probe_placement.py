# !/usr/bin/python
# coding=utf-8
"""``ProbePlacement`` / ``ProbeRays`` -- where a lightmap bake's reflection probe stands.

Pure Python, no DCC: a fake host of axis-aligned boxes (analytic ray casts,
two-sided, each face's normal by its winding) stands in for a DCC's meshes,
through the two adapters a host injects (``ProbeHost``, ``ProbeMesh``). The
DCCs' own suites run the twelve golden scenes against real geometry (mayatk's
``TestProbePlacement``, blendertk's ``test_lightmap_baker``); these pin the
host-free half and its seam: the axes and units a host declares, the lights it
reads, and the nearest-hit walk.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pythontk as ptk  # noqa: E402
from pythontk import ProbePlacement, ProbeRays, ProbeSite  # noqa: E402

ptk.TestSandbox.activate()


class _BoxMesh:
    """A fake :class:`ProbeMesh`: analytic hits on a box ``(lo, hi, inward)``;
    *inward* faces turn in."""

    def __init__(self):
        self.intersected = 0

    def intersect(self, payload, origin, direction, far):
        self.intersected += 1
        lo, hi, inward = payload
        best = None
        for axis in range(3):
            d = float(direction[axis])
            if abs(d) < 1e-12:
                continue
            for side, plane in ((0, lo[axis]), (1, hi[axis])):
                t = (plane - float(origin[axis])) / d
                if t <= 1e-9 or t > far:
                    continue
                p = [float(origin[i]) + float(direction[i]) * t for i in range(3)]
                if all(
                    lo[i] - 1e-9 <= p[i] <= hi[i] + 1e-9 for i in range(3) if i != axis
                ) and (best is None or t < best[0]):
                    n = [0.0, 0.0, 0.0]
                    n[axis] = (-1.0 if side == 0 else 1.0) * (-1.0 if inward else 1.0)
                    best = (t, tuple(p), tuple(n))
        return best

    def normal(self, payload, face):
        return face


def _Boxes(boxes):
    """Rays against boxes ``{name: (lo, hi, inward)}``, through :class:`_BoxMesh`."""
    rays = ProbeRays(_BoxMesh())
    for name, (lo, hi, inward) in boxes.items():
        rays.add(name, lo, hi, (lo, hi, inward))
    return rays


class _BoxHost:
    """A fake :class:`ProbeHost`: a Z-up scene of boxes in *unit* metres."""

    up = 2

    def __init__(self, boxes, lights=(), sky=False, unit=1.0):
        self.internal_m = unit
        self._boxes = boxes
        self._scene_lights = ([(tuple(p), None) for p in lights], sky)

    @staticmethod
    def nodes(objects):
        return list(objects)

    def rays(self, nodes):
        return _Boxes({n: self._boxes[n] for n in nodes})

    def bounds(self, nodes):
        return [
            [min(self._boxes[n][0][a] for n in nodes) for a in range(3)],
            [max(self._boxes[n][1][a] for n in nodes) for a in range(3)],
        ]

    def lights(self):
        return self._scene_lights

    @staticmethod
    def to_scene(value):
        return value

    @staticmethod
    def to_internal(point):
        return [float(c) for c in point]


def _Placement(boxes, baked, unbaked=(), lights=(), sky=False, unit=1.0, host=None):
    """The one ProbePlacement over a fake box host."""
    host = host or _BoxHost(boxes, lights, sky, unit)
    return ProbePlacement(baked, unbaked, host=host)


def _box(size, at, inward=False):
    lo = [c - s / 2.0 for c, s in zip(at, size)]
    hi = [c + s / 2.0 for c, s in zip(at, size)]
    return lo, hi, inward


class TestProbePlacement(unittest.TestCase):
    def assertNear(self, got, want, delta=1e-6):
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, delta=delta, msg=(got, want))

    def test_a_room_takes_the_probe_to_its_middle_at_half_its_height(self):
        boxes = {"room": _box((10, 10, 4), (0, 0, 2), inward=True)}
        site = _Placement(boxes, ["room"], lights=[(3, -3, 3.5)]).place()
        self.assertIsInstance(site, ProbeSite)
        self.assertEqual(site.reason, "the room's middle")
        self.assertNear(site.position, (0.0, 0.0, 2.0))
        self.assertEqual(site.box, [[-5.0, -5.0, 0.0], [5.0, 5.0, 4.0]])
        self.assertEqual(site.open_faces, 0)

    def test_a_solid_on_open_ground_is_flown_over_at_eye_level(self):
        """The host's up axis is the one the probe rises along: Z here, Y in
        Maya -- a car on a ground plane, the probe 2 m over the ground."""
        boxes = {
            "ground": _box((40, 40, 0.1), (0, 0, -0.05)),
            "car": _box((4.5, 1.8, 1.4), (0, 0, 0.7)),
        }
        site = _Placement(boxes, ["ground", "car"], lights=[(0, 0, 8)]).place()
        self.assertNear(site.position, (0.0, 0.0, 2.0), delta=1e-4)
        self.assertEqual(site.open_faces, 5)
        self.assertAlmostEqual(site.box[0][2], 0.0, delta=1e-6)
        self.assertGreaterEqual(abs(site.box[1][2]), 500.0, "the sky, read as distant")

    def test_what_moves_is_seen_through_and_the_probe_stands_over_it(self):
        boxes = {
            "room": _box((10, 10, 4), (0, 0, 2), inward=True),
            "prop": _box((0.2, 0.2, 0.2), (-3, 1.5, 1.0)),
        }
        site = _Placement(boxes, ["room"], ["prop"], lights=[(3, -3, 3.5)]).place()
        self.assertEqual(site.hide, ["prop"])
        self.assertEqual(site.reason, "over what moves")
        self.assertNear(site.position, (-3.0, 1.5, 2.0), delta=1e-4)

    def test_a_lone_object_in_empty_space_is_read_as_distant(self):
        boxes = {"decal": _box((0.1, 0.1, 0.01), (0, 0, 5))}
        site = _Placement(boxes, ["decal"]).place()
        self.assertIsNone(site.box)
        self.assertEqual(site.open_faces, 6)

    def test_the_hosts_unit_scales_every_metre_rule(self):
        """The rules are in metres (eye level 2 m, clearance 0.3 m): a host in
        centimetres declares ``INTERNAL_M = 0.01`` and gets them in its own."""
        boxes = {
            "room": _box((1000, 1000, 400), (0, 0, 200), inward=True),
            "prop": _box((20, 20, 20), (-300, 150, 100)),
        }
        site = _Placement(
            boxes, ["room"], ["prop"], lights=[(300, -300, 350)], unit=0.01
        ).place()
        self.assertNear(site.position, (-300.0, 150.0, 200.0), delta=1e-2)
        self.assertNear(site.box[0], (-500.0, -500.0, 0.0), delta=1e-6)
        self.assertNear(site.box[1], (500.0, 500.0, 400.0), delta=1e-6)

    def test_nothing_baked_places_nothing(self):
        self.assertIsNone(_Placement({}, []).place())

    def test_a_tie_between_two_meshes_goes_to_the_first_one_a_ray_met(self):
        """REGRESSION: the mesh around a point was the most-hit owner taken over
        a SET, so a tie went to the set's hash order -- for node names a new
        order in every process, one scene placing its probe two ways. A tie
        goes to the owner the first ray met."""
        from pythontk.geo_utils.probe_placement import ProbeHit

        class Tied(ProbeRays):
            """Every ray lands, the two meshes taking turns: 13 hits each."""

            def __init__(self, owners):
                super().__init__()
                for owner in owners:
                    self.add(owner, (-5, -5, -5), (5, 5, 5), None)
                self.casts = 0

            def cast(self, origin, direction, far=1.0e7):
                self.casts += 1
                normal = tuple(-c for c in direction)
                return ProbeHit(1.0, tuple(origin), normal, self.casts % 2)

        class Host(_BoxHost):
            def rays(self, nodes):
                return Tied([3, 5])  # the first ray meets owners[1]

        boxes = {n: _box((10, 10, 10), (0, 0, 0)) for n in (3, 5)}
        placement = _Placement(boxes, [3, 5], host=Host(boxes))
        self.assertEqual(placement._look((0.0, 0.0, 0.0)).solid, 5)


class TestProbeHostSeam(unittest.TestCase):
    """The scene comes through one injected adapter, never a subclass."""

    def test_no_host_is_refused(self):
        with self.assertRaises(TypeError):
            ProbePlacement(["room"])

    def test_a_bound_class_builds_a_fresh_host_per_placement(self):
        boxes = {"room": _box((10, 10, 4), (0, 0, 2), inward=True)}
        made = []

        class Bound(ProbePlacement):
            @staticmethod
            def HOST_CLASS():
                made.append(_BoxHost(boxes, lights=[(3, -3, 3.5)]))
                return made[-1]

        first, second = Bound(["room"]), Bound(["room"])
        self.assertIs(first.host, made[0])
        self.assertIs(second.host, made[1])
        self.assertEqual(first.place().reason, "the room's middle")

    def test_the_hosts_axes_units_and_names_are_what_it_declares(self):
        class Named(_BoxHost):
            up = 1

            @staticmethod
            def nodes(objects):
                return [str(o).upper() for o in objects]

        boxes = {"ROOM": _box((10, 4, 10), (0, 2, 0), inward=True)}
        placement = _Placement(boxes, ["room"], host=Named(boxes, unit=0.01))
        self.assertEqual(placement.baked, ["ROOM"])
        self.assertEqual((placement.up, placement.flat), (1, [0, 2]))
        self.assertEqual(placement.INTERNAL_M, 0.01)


class TestProbeRays(unittest.TestCase):
    def test_the_nearest_hit_wins_and_names_its_mesh(self):
        rays = _Boxes(
            {
                "far": _box((1, 1, 1), (10, 0, 0)),
                "near": _box((1, 1, 1), (4, 0, 0)),
            }
        )
        hit = rays.cast((0, 0, 0), (1, 0, 0))
        self.assertEqual(rays.owners[hit.mesh], "near")
        self.assertAlmostEqual(hit.distance, 3.5)
        self.assertEqual(hit.normal, (-1.0, 0.0, 0.0))

    def test_a_mesh_whose_bounds_start_past_the_best_hit_is_never_intersected(self):
        rays = _Boxes(
            {
                "near": _box((1, 1, 1), (4, 0, 0)),
                "far": _box((1, 1, 1), (10, 0, 0)),
                "aside": _box((1, 1, 1), (0, 10, 0)),
            }
        )
        rays.cast((0, 0, 0), (1, 0, 0))
        self.assertEqual(rays.mesh.intersected, 1, "the bounds test culls the rest")

    def test_a_ray_that_meets_nothing_within_reach_is_none(self):
        rays = _Boxes({"box": _box((1, 1, 1), (4, 0, 0))})
        self.assertIsNone(rays.cast((0, 0, 0), (1, 0, 0), far=3.0))
        self.assertIsNone(rays.cast((0, 0, 0), (0, 1, 0)))
        self.assertFalse(ProbeRays())


if __name__ == "__main__":
    unittest.main()
