# !/usr/bin/python
# coding=utf-8
"""``SpotPlacement``: the one placement every tool uses for a first pose, over boxes (``BoxRays``), and the golden cases
every port of it is held to (``ptk.Conformance.cases("spot_placement")``)."""

import math
import unittest

from pythontk.core_utils.conformance import Conformance
from pythontk.geo_utils.spot_placement import (
    BoxRays,
    SpotPlacement,
    SpotPlacementConformance,
    SpotRays,
)

CASES = {c["name"]: c for c in SpotPlacementConformance.scenes()}


def _answer(name):
    return SpotPlacementConformance.answer(CASES[name])


def _scene(boxes):
    rays = BoxRays()
    for owner, low, high in boxes:
        rays.add(owner, low, high)
    return rays


def _gap(point, low, high):
    return math.sqrt(
        sum(max(low[a] - point[a], 0.0, point[a] - high[a]) ** 2 for a in range(3))
    )


class SpotPlacementTest(unittest.TestCase):
    # A cabinet's door (2 m tall, on the floor at the origin) with a latch on its front (+Z); a panel BEHIND the door
    # carries a set of dials on its back face (-Z). The real task's dials camera stood behind the panel.
    DOOR = SpotPlacementConformance.DOOR
    LATCH = SpotPlacementConformance.LATCH
    PANEL = SpotPlacementConformance.PANEL
    DIALS = SpotPlacementConformance.DIALS

    def test_a_camera_never_stands_behind_the_panel_its_part_is_on(self):
        dials = (self.DIALS[1], self.DIALS[2])
        delivery = (
            (-3.0, 0.0, -6.0),
            (3.0, 2.5, -0.6),
        )  # most of the room out at -Z: by bounds, outer is +Z
        plain = SpotPlacement().view(dials, delivery)
        self.assertGreater(
            plain[0][2],
            -0.43,
            "with bounds alone: on the outer side, through the panel",
        )
        place = SpotPlacement(rays=_scene([self.DOOR, self.PANEL, self.DIALS]))
        eye, aim = place.view(dials, delivery, owners=["DIALS"])
        self.assertEqual(
            place.seen(eye, dials, ["DIALS"], fine=True),
            1.0,
            "the whole of the dials seen",
        )
        self.assertLess(
            eye[2], -0.46, "from the side the dials face, never through the panel"
        )

    def test_a_viewpoint_stands_on_the_floor_where_its_eyes_see_the_part(self):
        case = CASES["a viewpoint stands on the floor where its eyes see the part"]
        feet = _answer(case["name"])["position"]
        self.assertEqual(feet[1], 0.0, "on the floor's top")
        self.assertLess(
            feet[2] + SpotPlacement.BODY_RADIUS,
            1.0,
            "with room for a body short of the wall",
        )
        place = SpotPlacement(
            rays=_scene([(b["owner"], b["low"], b["high"]) for b in case["boxes"]])
        )
        eyes = (feet[0], SpotPlacement.EYE_HEIGHT, feet[2])
        self.assertGreater(
            place.seen(eyes, (self.LATCH[1], self.LATCH[2]), ["LATCH"]),
            0.5,
            "its eyes see the latch",
        )

    def test_a_named_side_outranks_sight_and_a_viewpoint_keeps_out_of_what_comes_to_stand_there(
        self,
    ):
        name = (
            "a named side outranks sight, and a viewpoint keeps out of a door's swing"
        )
        feet = _answer(name)["position"]
        self.assertGreater(feet[2], 0.5, "on the named side (+Z)")
        swing = CASES[name]["keep_out"][0]
        self.assertGreaterEqual(
            _gap(feet, swing["low"], swing["high"]),
            SpotPlacement.BODY_RADIUS - 1e-9,
            "a body's width clear of the cart's place",
        )

    def test_a_viewpoint_stands_back_for_a_tall_part_and_near_where_it_stood(self):
        feet = _answer(
            "a viewpoint stands back for a tall part to fit its view, and near where it stood"
        )["position"]
        self.assertGreater(
            math.hypot(feet[0], feet[2]), 2.2, "the 2 m door fits a 60 degree view"
        )
        self.assertAlmostEqual(
            math.atan2(feet[2], feet[0]),
            math.radians(45.0),
            places=6,
            msg="toward where it stood",
        )

    def test_a_viewpoint_that_sees_the_other_parts_too_wins(self):
        feet = _answer("a viewpoint that sees the other parts too wins")["position"]
        self.assertLess(
            feet[0], -0.5, "the side the gauge on the door's -X edge is seen from"
        )

    def test_a_viewpoint_never_stands_where_it_has_no_room(self):
        feet = _answer("a viewpoint never stands under a shelf it has no room under")[
            "position"
        ]
        table = [-1.5, 0.9, 0.5], [1.5, 1.0, 2.5]
        self.assertGreaterEqual(
            _gap((feet[0], 0.95, feet[2]), *table), SpotPlacement.BODY_RADIUS - 1e-9
        )

    def test_a_viewpoint_stands_on_the_floor_under_a_ceiling_never_on_it(self):
        """REGRESSION: the floor was read by rays cast down from a metre over the part's top, so under a ceiling lower
        than that every ray landed on the ceiling's top: the floor read as the ceiling, and the viewpoint stood on the
        roof, seeing nothing. A ray now lands on every top it passes, and the floor is the lowest."""
        case = CASES["a viewpoint stands on the floor under a ceiling, never on it"]
        answer = _answer(case["name"])
        self.assertTrue(answer["found"], "a spot that sees the handle")
        self.assertEqual(answer["position"][1], 0.0, "on the floor's top")
        place = SpotPlacement(
            rays=_scene([(b["owner"], b["low"], b["high"]) for b in case["boxes"]])
        )
        handle = (case["target"]["low"], case["target"]["high"])
        self.assertEqual(place.floor_near(handle), (0.0, True))

    def test_a_capsule_s_room_is_read_along_its_axis_a_quarter_radius_apart(self):
        """The one rule every ray set reads a capsule by (:meth:`SpotPlacement.stations`): its axis end to end."""
        points = SpotPlacement.stations((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.35)
        self.assertEqual((points[0], points[-1]), ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        gaps = [b[1] - a[1] for a, b in zip(points, points[1:])]
        self.assertLessEqual(max(gaps), 0.35 * SpotPlacement.STATION + 1e-12)
        self.assertEqual(
            SpotPlacement.stations((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), 0.35),
            [(1.0, 2.0, 3.0)],
        )

    def test_a_picture_of_a_part_behind_a_grille_stands_where_it_is_seen_whole(self):
        case = CASES[
            "a picture of a part behind a grille stands where it is seen whole"
        ]
        eye = _answer(case["name"])["position"]
        place = SpotPlacement(
            rays=_scene([(b["owner"], b["low"], b["high"]) for b in case["boxes"]])
        )
        fan = (case["target"]["low"], case["target"]["high"])
        self.assertEqual(place.seen(eye, fan, ["FAN"], fine=True), 1.0)

    def test_a_picture_shows_a_flat_part_face_on_among_spots_that_see_it_alike(self):
        # The real dials' first camera saw them whole from 59 degrees off their face: a dial seen along it does not read.
        name = "a picture shows a flat part face-on, not from the eyes' side along it"
        answer = _answer(name)
        way = [p - a for p, a in zip(answer["position"], answer["aim"])]
        cos = way[2] / math.sqrt(sum(w * w for w in way))
        self.assertGreater(
            cos,
            math.cos(math.radians(25.0)),
            "across the plate's face (+Z), whatever side the eyes are on",
        )
        plain = dict(
            CASES[name], target={"low": [-0.15, 1.2, 0.0], "high": [0.15, 1.5, 0.3]}
        )  # a block, no face
        block = SpotPlacementConformance.answer(plain)
        bway = [p - a for p, a in zip(block["position"], block["aim"])]
        self.assertGreater(
            bway[0] / math.sqrt(sum(w * w for w in bway)),
            0.5,
            "a block is no face: the eyes' side (+X)",
        )

    def test_an_arrow_stands_clear_of_a_shelf_and_never_end_on_to_the_eyes(self):
        answer = _answer(
            "an arrow stands clear of a shelf and never end-on to the eyes"
        )
        tip, aim = answer["position"], answer["aim"]
        self.assertLess(tip[1], 1.17, "not above: the shelf is there")
        way = [t - a for t, a in zip(tip, aim)]
        sight = [c - e for c, e in zip(aim, (0.2, 1.6, 2.0))]
        cos = abs(sum(w * s for w, s in zip(way, sight))) / (
            math.sqrt(sum(w * w for w in way)) * math.sqrt(sum(s * s for s in sight))
        )
        self.assertLess(
            cos,
            math.cos(math.radians(SpotPlacement.END_ON)),
            "never along the line of sight",
        )

    def test_an_arrow_points_down_from_above_where_free_and_its_bob_needs_its_sweep(
        self,
    ):
        above = _answer(
            "an arrow points down at its part from above where that is free"
        )
        self.assertEqual(
            [round(c, 9) for c in above["position"]],
            [0.0, round(1.04 + SpotPlacement.GAP, 9), 0.0],
        )
        bobbing = _answer("an arrow's bob needs its whole sweep free")
        self.assertTrue(bobbing["found"])
        self.assertGreater(
            bobbing["position"][2],
            0.0,
            "the lid over it is in the bob's way: out toward the eyes",
        )

    def test_where_nothing_is_free_an_arrow_comes_straight_out_of_what_its_part_is_mounted_on(
        self,
    ):
        answer = _answer(
            "where nothing is free, an arrow comes straight out of what its part is mounted on"
        )
        self.assertFalse(answer["found"])
        way = [t - a for t, a in zip(answer["position"], answer["aim"])]
        self.assertEqual(
            [round(c, 9) for c in way],
            [0.0, 0.0, round(0.01 + SpotPlacement.GAP, 9)],
            "out of the door",
        )

    def test_with_no_rays_a_first_pose_is_the_plain_one(self):
        box = ((-0.1, 1.0, 0.4), (0.1, 1.2, 0.6))
        room = ((-1.0, 0.0, -1.0), (1.0, 2.5, 1.0))
        feet, aim = SpotPlacement().stand(box, room)
        self.assertEqual(feet, (0.0, 0.0, 0.5 + 0.1 + SpotPlacement.STAND_OFF))
        tip, _ = SpotPlacement().point(box, room)
        self.assertEqual(tip, (0.0, 1.2 + SpotPlacement.GAP, 0.5))

    def test_a_z_up_centimetre_scene_places_as_the_y_up_metre_one(self):
        metres = _answer(
            "a picture stands where its part is seen, never behind the panel it is on"
        )["position"]
        cm = _answer("a Z-up centimetre scene places as a Y-up metre one")["position"]
        turned = (metres[0] * 100.0, -metres[2] * 100.0, metres[1] * 100.0)
        for a, b in zip(cm, turned):
            self.assertAlmostEqual(a, b, places=6)

    def test_box_rays_pass_a_box_they_start_in_and_read_room_exactly(self):
        rays = _scene([("A", (0, 0, 0), (1, 1, 1))])
        self.assertIsInstance(rays, SpotRays)
        self.assertIsNone(
            rays.cast((0.5, 0.5, 0.5), (1, 0, 0), 5.0),
            "a ray starting inside passes it",
        )
        hit = rays.cast((-1, 0.5, 0.5), (1, 0, 0), 5.0)
        self.assertEqual((hit.distance, hit.normal), (1.0, (-1.0, 0.0, 0.0)))
        self.assertIsNone(rays.cast((-1, 0.5, 0.5), (1, 0, 0), 5.0, skip={"A"}))
        self.assertTrue(rays.occupied_sphere((1.2, 0.5, 0.5), 0.25))
        self.assertFalse(rays.occupied_sphere((1.3, 0.5, 0.5), 0.25))
        self.assertTrue(rays.occupied_capsule((2, 0.5, 0.5), (1.2, 0.5, 0.5), 0.25))

    def test_the_golden_cases_are_the_registry_s_and_the_same_for_a_seed(self):
        doc = Conformance.cases("spot_placement", seed=3, per_kind=2)
        self.assertEqual(doc, SpotPlacementConformance.cases(seed=3, per_kind=2))
        names = [c["name"] for c in doc["cases"]]
        self.assertEqual(len(names), len(CASES) + 6)
        self.assertTrue(all("expected" in c for c in doc["cases"]))
        self.assertTrue(
            all(
                c["expected"]["found"]
                for c in doc["cases"]
                if c["name"].startswith("room")
            ),
            "every random room has a spot",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
