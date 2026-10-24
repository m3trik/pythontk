# !/usr/bin/python
# coding=utf-8
"""BakeBaseline -- what a lightmap re-bake must redo, decided without a DCC."""

import json
import unittest

import numpy as np

from pythontk.core_utils.engines.scene_export.bake_baseline import (
    BakeBaseline as BB,
    StaleReport,
)
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords

R = BB.Reason

# A cube: 8 points, x-major (index = 4x + 2y + z), six quads.
COUNTS = [4] * 6
INDICES = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
UVS = [[0.1 + 0.2 * (i % 4), 0.1 + 0.4 * (i // 4)] for i in range(8)]
INFO = {
    "map": "Room_Lightmap.exr",
    "cell": [0.5, 0.5, 0.0, 0.0],
    "extent": [1.0, 1.0],
    "size": 1024,
    "packing": "atlas",
}


def cube(size=1.0):
    return [
        [size * (x - 0.5), size * (y - 0.5), size * (z - 0.5)]
        for x in (0, 1)
        for y in (0, 1)
        for z in (0, 1)
    ]


def state(at=(0.0, 0.0, 0.0), points=None, indices=INDICES, **fields):
    """A DCC's ``scene_state`` row for a mesh whose object-space *points* sit
    at world offset *at* (a translate-only matrix, translation in [12:15])."""
    pts = np.asarray(cube() if points is None else points, dtype=np.float64)
    world = pts + np.asarray(at, dtype=np.float64)
    out = {
        "matrix": [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, at[0], at[1], at[2], 1.0],
        "bbox": [world.min(axis=0).tolist(), world.max(axis=0).tolist()],
        "geometry": BB.signature(pts, [4] * (len(indices) // 4), indices),
        "layout": BB.signature(UVS, COUNTS, INDICES),
        "materials": "lambert1",
        "group": "roomSG",
        "rendered": True,
    }
    out.update(fields)
    return out


def part(st, role="baked", map_name="Room_Lightmap.exr", so=(0.5, 0.5, 0.0, 0.0)):
    marker = {"map": map_name, "scaleOffset": list(so)} if map_name else None
    return {"state": st, "role": role, "marker": marker}


def room(**changes):
    """A row of unit cubes along x: A at 0, B at 1.5 (0.5 from A), C at 12
    (far), and an Exclude-set X at 20 that stretches the room to ~21 units --
    reach for a unit cube is then its own diagonal (1.73), inside the room's
    floor (0.21) and cap (5.25)."""
    scene = {
        "|grp|A": part(state((0.0, 0.0, 0.0)), so=(0.5, 0.5, 0.0, 0.0)),
        "|grp|B": part(state((1.5, 0.0, 0.0)), so=(0.5, 0.5, 0.5, 0.0)),
        "|grp|C": part(state((12.0, 0.0, 0.0)), so=(0.5, 0.5, 0.0, 0.5)),
        "|X": part(state((20.0, 0.0, 0.0)), role="excluded", map_name=None),
    }
    for key, value in changes.items():
        key = key if key.startswith("|") else f"|grp|{key}"
        if value is None:
            scene.pop(key, None)
        else:
            scene[key] = value
    return scene


def lights(key=2.0, fill=0.5):
    return BB.lighting_state(
        {
            "|key": {"intensity": key, "color": [1.0, 0.9, 0.8], "visible": True},
            "|fill": {"intensity": fill, "color": [0.6, 0.7, 1.0], "visible": True},
        },
        settings={"gi_depth": 2, "scene_lights": True},
    )


def scoped(lamp, power=2.0, **rule):
    """A Scope-lit bake's lighting: *lamp* alone, its rule recorded."""
    return BB.lighting_state(
        {lamp: {"intensity": power, "on": True}},
        settings={"gi_depth": 2},
        rule=dict(
            {
                "include_scene_lights": False,
                "include_environment": True,
                "include_hidden_lights": False,
                "scope": [lamp],
            },
            **rule,
        ),
    )


def by_rule(powers):
    """A DCC's lighting reader: the scene now, read by a recorded *rule* --
    its Scope's one lamp at the power *powers* holds for it now."""

    def read(rule):
        lamp = rule["scope"][0]
        return scoped(lamp, powers[lamp])

    return read


BAKED = ("|grp|A", "|grp|B", "|grp|C")


def first_bake(scene=None, lighting=None):
    scene = room() if scene is None else scene
    return BB.advance(
        None,
        scene,
        baked={k: dict(INFO) for k in BAKED if k in scene},
        lighting=lights() if lighting is None else lighting,
    )


def seven_digits(points):
    """What a save + reopen through ``.ma`` does to float32 vertex data:
    seven significant digits on the way out, float32 on the way in."""
    return np.asarray(
        [[np.float32(float(f"{v:.7g}")) for v in p] for p in points],
        dtype=np.float64,
    )


class TestSignature(unittest.TestCase):
    def test_topology_is_hashed_exactly_and_by_order(self):
        a = BB.signature(cube(), COUNTS, INDICES)
        self.assertEqual(a, BB.signature(np.asarray(cube()), COUNTS, INDICES))
        swapped = list(INDICES)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        self.assertNotEqual(
            a["topology"], BB.signature(cube(), COUNTS, swapped)["topology"]
        )
        self.assertEqual(len(a["topology"]), 32)  # blake2b, 16 bytes
        self.assertEqual(a["points"], 8)

    def test_moments_are_centroid_then_variances_then_covariances(self):
        geo = BB.signature(
            [[0, 0, 0], [2, 0, 0], [0, 4, 0], [2, 4, 0]], [4], [0, 1, 3, 2]
        )
        self.assertEqual(geo["moments"], [1.0, 2.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        uv = BB.signature(UVS, COUNTS, INDICES)
        self.assertEqual(len(uv["moments"]), 5)
        flat = BB.signature(np.asarray(UVS).ravel(), COUNTS, INDICES, dims=2)
        self.assertEqual(flat, uv)


class TestDiff(unittest.TestCase):
    def test_an_unchanged_state_is_the_same(self):
        self.assertTrue(BB.same(state(), state()))
        self.assertEqual(BB.diff(state(), state()), [])

    def test_float_noise_of_a_saved_scene_stays_within_tolerance(self):
        """``.ma`` keeps ~7 significant digits of float32 vertex data. A
        quantised point hash flips on that; moments with tolerances relative
        to the object's size must not -- near the origin, and on an object
        whose frozen points sit 1234 units out, where the noise is ~1e-4."""
        rng = np.random.default_rng(7)
        for centre in ((0.1, -0.2, 0.3), (1234.5678, -987.6543, 456.789)):
            pts = np.asarray(centre) + rng.uniform(-1.0, 1.0, (200, 3))
            idx = list(range(200))
            before = state(points=pts, indices=idx)
            after = state(points=seven_digits(pts), indices=idx)
            self.assertNotEqual(before["geometry"], after["geometry"])
            self.assertTrue(BB.same(before, after), (centre, BB.diff(before, after)))

    def test_a_move_is_detected(self):
        self.assertEqual(BB.diff(state(), state((0.002, 0.0, 0.0))), [R.MOVED])

    def test_a_deform_is_detected(self):
        pts = cube()
        pts[7] = [p * 1.02 for p in pts[7]]
        self.assertEqual(BB.diff(state(), state(points=pts)), [R.DEFORMED])

    def test_a_deform_inside_the_bounds_is_detected_by_the_moments(self):
        """A point pushed in, not out: the box cannot see it."""
        inner = cube() + [[0.0, 0.0, 0.0]] * 4
        pushed = cube() + [[0.0, 0.0, 0.0]] * 3 + [[0.2, 0.1, 0.0]]
        idx = list(range(12))
        before, after = (
            state(points=inner, indices=idx),
            state(points=pushed, indices=idx),
        )
        self.assertEqual(before["bbox"], after["bbox"])
        self.assertEqual(BB.diff(before, after), [R.DEFORMED])

    def test_a_topology_change_is_detected(self):
        swapped = list(INDICES)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        self.assertEqual(BB.diff(state(), state(indices=swapped)), [R.REMODELLED])
        more = cube() + [[0.0, 0.0, 0.0]]
        self.assertIn(R.REMODELLED, BB.diff(state(), state(points=more)))

    def test_layout_material_and_visibility_each_have_a_reason(self):
        relaid = dict(
            state(),
            layout=BB.signature([[u, v * 0.5] for u, v in UVS], COUNTS, INDICES),
        )
        self.assertEqual(BB.diff(state(), relaid), [R.RELAID])
        self.assertEqual(BB.diff(state(), state(group="otherSG")), [R.MATERIAL])
        self.assertEqual(BB.diff(state(), state(rendered=False)), [R.VISIBILITY])

    def test_tolerance_multiplier_loosens_every_test(self):
        moved = state((0.002, 0.0, 0.0))
        self.assertFalse(BB.same(state(), moved))
        self.assertTrue(BB.same(state(), moved, tol=1e6))


class TestNeighbours(unittest.TestCase):
    OLD = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    NEW = [[5.5, -0.5, -0.5], [6.5, 0.5, 0.5]]
    ROOM = [[-0.5, -0.5, -0.5], [20.5, 0.5, 0.5]]

    def _box(self, x):
        return [[x - 0.5, -0.5, -0.5], [x + 0.5, 0.5, 0.5]]

    def test_a_neighbour_of_the_old_or_the_new_box_is_named(self):
        boxes = {
            "near_old": self._box(1.5),
            "near_new": self._box(7.5),
            "between": self._box(3.0),
            "far": self._box(12.0),
        }
        found = BB.neighbours({"A": [self.OLD, self.NEW]}, boxes, room=self.ROOM)
        self.assertEqual(found, {"near_old": ["A"], "near_new": ["A"]})

    def test_reach_is_clamped_to_the_room(self):
        """A pebble still reaches 1% of the room; a wall at most a quarter."""
        pebble = [[0.0, 0.0, 0.0], [0.001, 0.001, 0.001]]
        boxes = {"b": [[0.15, 0.0, 0.0], [0.2, 0.001, 0.001]]}
        self.assertEqual(
            BB.neighbours({"p": [pebble]}, boxes, room=self.ROOM), {"b": ["p"]}
        )
        wall = [[0.0, -50.0, -50.0], [1.0, 50.0, 50.0]]
        big_room = [[0.0, -50.0, -50.0], [100.0, 50.0, 50.0]]
        boxes = {"in": self._box(30.0), "out": self._box(60.0)}
        self.assertEqual(
            BB.neighbours({"w": [wall]}, boxes, room=big_room), {"in": ["w"]}
        )

    def test_a_zero_multiplier_names_no_neighbour(self):
        boxes = {"touching": self._box(1.0)}
        self.assertEqual(
            BB.neighbours({"A": [self.OLD]}, boxes, room=self.ROOM), {"touching": ["A"]}
        )
        self.assertEqual(
            BB.neighbours({"A": [self.OLD]}, boxes, room=self.ROOM, reach=0), {}
        )

    def test_a_cause_never_names_itself(self):
        self.assertEqual(BB.neighbours({"A": [self.OLD]}, {"A": self.OLD}), {})


class TestAssess(unittest.TestCase):
    def test_an_unchanged_scene_is_not_stale(self):
        report = BB.assess(first_bake(), room(), lighting=lights())
        self.assertEqual(report.targets, ())
        self.assertEqual((report.stale, report.neighbours), ({}, {}))
        self.assertEqual(report.untracked, ())

    def test_a_reopened_scene_is_not_stale(self):
        rng = np.random.default_rng(3)
        pts = np.asarray([3.0, 0.0, 0.0]) + rng.uniform(-0.5, 0.5, (64, 3))
        idx = list(range(64))
        before = room(D=part(state((0.0, 0.0, 0.0), points=pts, indices=idx)))
        after = room(
            D=part(state((0.0, 0.0, 0.0), points=seven_digits(pts), indices=idx))
        )
        record = first_bake(before)
        self.assertEqual(BB.assess(record, after, lighting=lights()).targets, ())

    def test_a_moved_object_is_stale_and_names_its_neighbours_old_and_new(self):
        moved = room(A=part(state((6.0, 0.0, 0.0))), D=part(state((7.5, 0.0, 0.0))))
        record = first_bake(room(D=part(state((7.5, 0.0, 0.0)))))
        report = BB.assess(record, moved)
        self.assertEqual(report.stale, {"|grp|A": (R.MOVED,)})
        # B sat beside A's old place, D beside its new one; C is far from both.
        self.assertEqual(
            report.neighbours, {"|grp|B": ("|grp|A",), "|grp|D": ("|grp|A",)}
        )
        self.assertEqual(report.targets, ("|grp|A", "|grp|B", "|grp|D"))

    def test_a_zero_reach_rebakes_only_what_changed(self):
        report = BB.assess(first_bake(), room(A=part(state((0.0, 0.3, 0.0)))), reach=0)
        self.assertEqual(report.targets, ("|grp|A",))

    def test_an_isolated_cause_reaches_no_neighbour(self):
        isolated = room(A=part(state(), role="isolated"))
        record = first_bake(isolated)
        report = BB.assess(
            record, room(A=part(state((0.0, 0.3, 0.0)), role="isolated"))
        )
        self.assertEqual(report.stale, {"|grp|A": (R.MOVED,)})
        self.assertEqual(report.neighbours, {})

    def test_an_object_placed_after_a_neighbours_bake_stales_it_until_it_rebakes(self):
        """N joins the Exclude set beside B while a later bake re-bakes only
        far-off C: B's map was rendered without N's shadow."""
        placed = room(N=part(state((3.0, 0.0, 0.0)), role="excluded", map_name=None))
        before = BB.assess(first_bake(), placed)
        self.assertEqual(before.neighbours, {"|grp|B": ("|grp|N",)})
        later = BB.advance(first_bake(), placed, baked={"|grp|C": dict(INFO)})
        report = BB.assess(later, placed)
        self.assertEqual(report.neighbours, {"|grp|B": ("|grp|N",)})
        self.assertEqual(report.targets, ("|grp|B",))
        rebaked = BB.advance(later, placed, baked={"|grp|B": dict(INFO)})
        self.assertEqual(BB.assess(rebaked, placed).targets, ())

    def test_a_removal_stales_neighbours_until_they_rebake_then_is_pruned(self):
        gone = room(A=None)
        self.assertEqual(
            BB.assess(first_bake(), gone).neighbours, {"|grp|B": ("|grp|A",)}
        )
        later = BB.advance(first_bake(), gone, baked={"|grp|C": dict(INFO)})
        self.assertEqual(BB.decode(later)["objects"]["|grp|A"]["removed_at"], 2)
        self.assertEqual(BB.assess(later, gone).neighbours, {"|grp|B": ("|grp|A",)})
        rebaked = BB.advance(later, gone, baked={"|grp|B": dict(INFO)})
        self.assertNotIn("|grp|A", BB.decode(rebaked)["objects"])
        self.assertEqual(BB.assess(rebaked, gone).targets, ())

    def test_a_non_target_change_is_kept_stale_across_a_commit(self):
        """The commit records the CURRENT state, so the reasons it differed
        are kept, and so is the neighbour its old place would have lost."""
        moved = room(A=part(state((6.0, 0.0, 0.0))))
        later = BB.advance(first_bake(), moved, baked={"|grp|C": dict(INFO)})
        report = BB.assess(later, moved)
        self.assertEqual(report.stale["|grp|A"], (R.MOVED,))
        self.assertEqual(report.stale["|grp|B"], (R.NEIGHBOUR,))
        self.assertEqual(report.targets, ("|grp|A", "|grp|B"))

    def test_a_rename_rematches(self):
        renamed = room(A=None, A_renamed=room()["|grp|A"])
        report = BB.assess(first_bake(), renamed)
        self.assertEqual((report.targets, report.untracked), ((), ()))
        objects = BB.decode(BB.advance(first_bake(), renamed))["objects"]
        self.assertNotIn("|grp|A", objects)
        self.assertEqual(objects["|grp|A_renamed"]["baked_at"], 1)

    def test_an_in_place_duplicate_is_stale(self):
        duplicated = room(A_copy=room()["|grp|A"])
        report = BB.assess(first_bake(), duplicated)
        self.assertEqual(report.stale["|grp|A_copy"], (R.DUPLICATE,))
        self.assertIn("|grp|A_copy", report.targets)
        self.assertEqual(report.untracked, ())
        later = BB.advance(first_bake(), duplicated)
        self.assertEqual(
            BB.assess(later, duplicated).stale["|grp|A_copy"], (R.DUPLICATE,)
        )

    def test_a_lighting_change_stales_everything_baked_under_it(self):
        report = BB.assess(first_bake(), room(), lighting=lights(key=3.0))
        self.assertEqual(report.lighting, ("|key",))
        self.assertEqual(report.targets, BAKED)
        self.assertTrue(all(r == (R.LIGHTING,) for r in report.stale.values()))
        self.assertEqual(BB.light_changes(lights(), lights(key=3.0)), ["|key"])

    def test_lighting_noise_and_the_opt_out_stale_nothing(self):
        """2.0000051 rounds to another digest (6 digits), so it is the
        tolerance, not the rounding, that keeps it from reading as changed."""
        record = first_bake()
        noisy = lights(key=2.0000051)
        self.assertNotEqual(BB.lighting_digest(noisy), BB.lighting_digest(lights()))
        self.assertEqual(BB.assess(record, room(), lighting=noisy).targets, ())
        self.assertEqual(BB.assess(record, room(), lighting=None).targets, ())

    def test_each_recorded_rule_is_read_again_and_stales_only_what_it_lit(self):
        """Two Scope-lit bakes, each by its own lamp: the dry run reads the
        scene again by each bake's recorded rule, so one lamp's change stales
        what it lit, and nothing else."""
        record = BB.advance(
            None,
            room(),
            baked={k: dict(INFO) for k in ("|grp|A", "|grp|B")},
            lighting=scoped("|lampAB"),
        )
        record = BB.advance(
            record, room(), baked={"|grp|C": dict(INFO)}, lighting=scoped("|lampC")
        )
        powers = {"|lampAB": 2.0, "|lampC": 2.0}
        self.assertEqual(
            BB.assess(record, room(), lighting=by_rule(powers)).targets, ()
        )
        powers["|lampC"] = 3.0
        report = BB.assess(record, room(), lighting=by_rule(powers))
        self.assertEqual(report.targets, ("|grp|C",))
        self.assertEqual(report.lighting, ("|lampC",))

    def test_a_lighting_recorded_without_its_rule_reads_changed_once(self):
        """A baseline written before rules were: its lighting reads as changed
        once, and settles once re-baked under a rule."""
        asked = []

        def read(rule):
            asked.append(rule)
            return scoped("|key")

        report = BB.assess(first_bake(), room(), lighting=read)
        self.assertEqual(asked, [None], "read once, by no rule")
        self.assertEqual(report.targets, BAKED)
        self.assertIn(BB.RULE, report.lighting)
        rebaked = BB.advance(
            first_bake(),
            room(),
            baked={k: dict(INFO) for k in BAKED},
            lighting=scoped("|key"),
        )
        settled = BB.assess(rebaked, room(), lighting=by_rule({"|key": 2.0}))
        self.assertEqual(settled.targets, ())

    def test_a_missing_map_is_stale(self):
        report = BB.assess(first_bake(), room(), missing={"|grp|C"})
        self.assertEqual(report.stale, {"|grp|C": (R.MISSING_MAP,)})

    def test_an_untracked_missing_map_is_stale(self):
        """Untracked is "only the artist knows whether it matches the scene";
        a map on disk nowhere matches nothing. Flagged only for a tracked
        object, a production room with no baseline listed its missing map as
        "Fix All bakes it again", and Fix All left it. Added: 2026-10-23"""
        legacy = room(L=part(state((16.0, 0.0, 0.0)), so=(1, 1, 0, 0)))
        report = BB.assess(first_bake(), legacy, missing={"|grp|L"})
        self.assertEqual(report.stale, {"|grp|L": (R.MISSING_MAP,)})
        self.assertEqual(report.targets, ("|grp|L",))
        report = BB.assess(None, room(), missing={"|grp|C"})
        self.assertEqual(report.targets, ("|grp|C",))

    def test_a_hidden_object_cannot_bake_but_its_neighbour_rebakes(self):
        report = BB.assess(
            first_bake(), room(B=part(state((1.5, 0.0, 0.0), rendered=False)))
        )
        self.assertEqual(report.stale["|grp|B"], (R.VISIBILITY,))
        self.assertEqual(report.unbakeable, ("|grp|B",))
        self.assertEqual(report.neighbours, {"|grp|A": ("|grp|B",)})
        self.assertEqual(report.targets, ("|grp|A",))

    def test_an_untracked_legacy_marker_is_reported_not_rebaked(self):
        report = BB.assess(
            first_bake(), room(L=part(state((16.0, 0.0, 0.0)), so=(1, 1, 0, 0)))
        )
        self.assertEqual(report.untracked, ("|grp|L",))
        self.assertEqual(report.targets, ())

    def test_a_commit_of_others_leaves_an_untracked_map_untracked(self):
        """A commit records every participant, and it took an untracked
        marked object in as tracked, its map vouched for by a bake of
        something else: tracking two maps in production turned every other
        untracked row "up to date". Its arrival is recorded, its map stays
        untracked until a commit bakes (or tracks) it. Added: 2026-10-23"""
        scene = room(L=part(state((16.0, 0.0, 0.0)), so=(1, 1, 0, 0)))
        once = BB.advance(first_bake(), scene, baked={"|grp|A": {}})
        self.assertEqual(BB.rematch(once, scene)[2], ["|grp|L"])
        report = BB.assess(once, scene)
        self.assertEqual((report.untracked, report.targets), (("|grp|L",), ()))
        tracked = BB.advance(once, scene, baked={"|grp|L": {}})
        self.assertEqual(BB.rematch(tracked, scene)[2], [])

    def test_with_no_baseline_nothing_is_stale(self):
        report = BB.assess(None, room())
        self.assertEqual(report.targets, ())
        self.assertEqual(report.untracked, BAKED)

    def test_objects_limits_the_report_not_the_causes(self):
        moved = room(A=part(state((0.0, 0.3, 0.0))))
        report = BB.assess(first_bake(), moved, objects=["|grp|B"])
        self.assertEqual(report.targets, ("|grp|B",))
        self.assertEqual(report.neighbours, {"|grp|B": ("|grp|A",)})


class TestAdvance(unittest.TestCase):
    def test_a_commit_records_targets_at_the_next_serial(self):
        record = BB.decode(
            BB.advance(
                first_bake(),
                room(),
                baked={"|grp|C": dict(INFO, cell=[0.25, 0.25, 0.5, 0.5])},
                lighting=lights(),
            )
        )
        self.assertEqual(record["serial"], 2)
        c = record["objects"]["|grp|C"]
        self.assertEqual(
            (c["baked_at"], c["cell"], c["map"]),
            (2, [0.25, 0.25, 0.5, 0.5], "room_lightmap.exr"),
        )
        self.assertEqual(c["lighting"], BB.lighting_digest(lights()))
        self.assertEqual(record["objects"]["|grp|A"]["baked_at"], 1)
        self.assertEqual(record["objects"]["|X"]["role"], "excluded")
        self.assertEqual(list(record["lightings"]), [BB.lighting_digest(lights())])

    def test_a_lighting_no_entry_reads_any_more_is_dropped(self):
        record = first_bake()
        for k in BAKED:
            record = BB.advance(
                record, room(), baked={k: dict(INFO)}, lighting=lights(key=5.0)
            )
        self.assertEqual(
            list(BB.decode(record)["lightings"]), [BB.lighting_digest(lights(key=5.0))]
        )


class TestLightingRecord(unittest.TestCase):
    """What a bake's recorded lighting tells a re-bake and an export: the rule
    each map was lit by, and the lights that lit it."""

    def test_lighting_rules_name_the_rule_each_map_was_lit_by(self):
        record = BB.advance(
            None, room(), baked={"|grp|A": dict(INFO)}, lighting=scoped("|lampA")
        )
        record = BB.advance(
            record, room(), baked={"|grp|B": dict(INFO)}, lighting=lights()
        )
        rules = BB.lighting_rules(record)
        self.assertEqual(rules["|grp|A"]["scope"], ["|lampA"])
        self.assertIs(rules["|grp|A"]["include_scene_lights"], False)
        self.assertIsNone(rules["|grp|B"], "recorded without a rule")
        self.assertIsNone(rules["|grp|C"], "never baked")

    def test_lights_used_are_the_lights_on_in_each_maps_lighting(self):
        """A hidden light the bake switched on is on; one it left off is not;
        the environment and the settings are no light."""
        lit = BB.lighting_state(
            {
                "|lamp": {"on": True},
                "|switched": {"on": True},
                "|left_off": {"on": False},
            },
            environment={"files": ["sky.exr"]},
            settings={"gi_depth": 2},
            rule={"include_scene_lights": True, "scope": None},
        )
        record = BB.advance(None, room(), baked={"|grp|A": dict(INFO)}, lighting=lit)
        record = BB.advance(
            record, room(), baked={"|grp|B": dict(INFO)}, lighting=scoped("|lampB")
        )
        self.assertEqual(BB.lights_used(record), ["|lamp", "|lampB", "|switched"])
        self.assertEqual(BB.lights_used(record, ["|grp|B"]), ["|lampB"])
        self.assertIsNone(
            BB.lights_used(record, ["|grp|A", "|grp|C"]),
            "C's lighting is unknown, so the lights its map saw are",
        )
        self.assertEqual(BB.lights_used(None, []), [])


class TestCellFits(unittest.TestCase):
    def test_ten_percent_either_way_fits_and_no_more(self):
        self.assertTrue(BB.cell_fits([100, 50], [110, 45]))
        self.assertTrue(BB.cell_fits([100, 50], [90, 55]))
        self.assertFalse(BB.cell_fits([100, 50], [110.01, 50]))
        self.assertFalse(BB.cell_fits([100, 50], [100, 44.99]))
        self.assertFalse(BB.cell_fits(None, [100, 50]))
        self.assertTrue(BB.cell_fits([100, 50], [120, 50], tolerance=0.2))


class TestRecord(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        record = BB.advance(first_bake(), room(A=None), baked={"|grp|B": dict(INFO)})
        text = json.dumps(BB.encode(record))
        self.assertEqual(BB.decode(text), BB.decode(record))
        self.assertEqual(BB.encode(BB.decode(text)), BB.encode(record))
        self.assertEqual(BB.encode(record)["version"], BB.FORMAT)

    def test_an_unreadable_or_newer_record_reads_empty(self):
        empty = {"version": BB.FORMAT, "serial": 0, "objects": {}, "lightings": {}}
        self.assertEqual(BB.decode(None), empty)
        self.assertEqual(BB.decode("not json"), empty)
        newer = dict(BB.encode(first_bake()), version=BB.FORMAT + 1)
        self.assertEqual(BB.decode(newer), empty)

    def test_the_format_is_the_scene_records_version(self):
        self.assertEqual(BB.FORMAT, SceneRecords.LIGHTMAP_BAKE_STATE.version)


class TestStaleReport(unittest.TestCase):
    def test_summary_and_lines_name_what_and_why(self):
        report = BB.assess(first_bake(), room(A=part(state((0.0, 0.3, 0.0)))))
        self.assertEqual(report.summary(), "2 to re-bake: 1 changed, 1 neighbour.")
        self.assertEqual(report.lines(), ["A: moved", "B: neighbour of A"])
        self.assertEqual(report.lines(name=str.upper)[0], "|GRP|A: moved")

    def test_an_empty_report_says_so(self):
        self.assertEqual(
            StaleReport().summary(), "Nothing changed since the last bake."
        )
        lit = BB.assess(first_bake(), room(), lighting=lights(key=3.0))
        self.assertIn("lighting changed (key)", lit.summary())
        self.assertIn("Lighting: key changed", lit.lines())

    def test_the_lighting_clause_says_what_it_stales(self):
        """One lamp's change stales the maps it lit, and the footer counts
        them: "the whole room" only when that is every baked object."""
        whole = BB.assess(first_bake(), room(), lighting=lights(key=3.0))
        self.assertEqual(whole.baked, 3)
        self.assertEqual(
            whole.summary(),
            "3 to re-bake: 3 changed; lighting changed (key) -- the whole room.",
        )
        record = BB.advance(
            None,
            room(),
            baked={k: dict(INFO) for k in ("|grp|A", "|grp|B")},
            lighting=scoped("|lampAB"),
        )
        record = BB.advance(
            record, room(), baked={"|grp|C": dict(INFO)}, lighting=scoped("|lampC")
        )
        one = BB.assess(
            record, room(), lighting=by_rule({"|lampAB": 2.0, "|lampC": 3.0})
        )
        self.assertEqual(one.targets, ("|grp|C",))
        self.assertEqual(
            one.summary(),
            "1 to re-bake: 1 changed; lighting changed (lampC) -- 1 of 3 baked objects.",
        )
        # A scope narrows what is counted, never the room it is counted in.
        scoped_whole = BB.assess(
            first_bake(), room(), lighting=lights(key=3.0), objects=["|grp|A"]
        )
        self.assertIn("-- 1 of 3 baked objects", scoped_whole.summary())
        # A report built without the room's size names the count alone.
        bare = StaleReport(
            targets=("|grp|C",), stale={"|grp|C": (R.LIGHTING,)}, lighting=("|lampC",)
        )
        self.assertIn("lighting changed (lampC) -- 1 baked object.", bare.summary())

    def test_the_summary_counts_what_a_repack_drags_in_and_what_is_out_of_the_bake(
        self,
    ):
        """A re-pack re-bakes its atlas's every member: the footer said "2 to
        re-bake" for a bake of 97 (production PLAYGROUND_wStands)."""
        report = StaleReport(
            targets=("|A",),
            stale={"|A": (R.MOVED,)},
            repack={"Room_Lightmap.exr": "A: its cell no longer fits"},
            joining=("|B", "|C"),
            forked={"Desk_Lightmap.exr": "scenes/source.ma"},
            out_of_bake=("|STAND",),
        )
        self.assertEqual(
            report.summary(),
            "1 to re-bake: 1 changed; 1 atlas re-packed whole (+2 more objects); "
            "1 atlas copied from another scene first; "
            "1 visible mesh out of the bake.",
        )
        lines = report.lines()
        self.assertIn("B: re-bakes with its atlas, which re-packs whole", lines)
        self.assertIn(
            "Desk_Lightmap.exr: copied first -- written by scenes/source.ma", lines
        )
        self.assertIn(
            "STAND: visible, never baked and not excluded -- out of every map's "
            "lighting",
            lines,
        )
        self.assertEqual(
            StaleReport(out_of_bake=("|STAND",)).summary(),
            "Nothing to re-bake; 1 visible mesh out of the bake.",
        )


class TestRequestedReport(unittest.TestCase):
    def test_targets_asked_for_are_neither_changed_nor_neighbours(self):
        """A Re-bake Selected's dry run names its targets with no reason:
        production PLAYGROUND.ma read "96 to re-bake: 96 neighbours"."""
        report = StaleReport(targets=("|A", "|B"))
        self.assertEqual(report.summary(), "2 to re-bake: 2 requested.")
        self.assertEqual(report.lines(), ["A: requested", "B: requested"])


class TestStaleReportDoc(unittest.TestCase):
    """:meth:`StaleReport.doc` -- the Bake State breakdown: one row per map,
    what re-bakes and why, and what is out of the bake."""

    def setUp(self):
        self.report = StaleReport(
            targets=("|grp|A",),
            stale={"|grp|A": (R.MOVED,)},
            in_place=("|grp|A",),
            repack={"Desk_Lightmap.exr": "it sits in another folder"},
            joining=("|grp|D",),
            out_of_bake=("|STAND",),
        )
        self.maps = [
            {
                "map": "Room_Lightmap.exr",
                "path": "C:/proj/sourceimages/lightmaps/Room_Lightmap.exr",
                "folder": "sourceimages/lightmaps",
                "objects": ["|grp|A", "|grp|B"],
                "writer": None,
                "size": (2048, 2048),
            },
            {
                "map": "Desk_Lightmap.exr",
                "path": "C:/proj/sourceimages/Desk_Lightmap.exr",
                "folder": "sourceimages",
                "objects": ["|grp|D"],
                "writer": "scenes/source.ma",
                "size": (1024, 1024),
            },
            {
                "map": "Lost_Lightmap.exr",
                "path": None,
                "folder": "sourceimages/lightmaps",
                "objects": ["|grp|E"],
                "writer": None,
                "size": None,
            },
        ]

    def test_each_map_is_a_row_with_its_state(self):
        text = self.report.doc(self.maps).to_text()
        self.assertIn("Lightmap Bake State", text)
        self.assertIn(self.report.summary(), text)
        for expected in (
            "Room_Lightmap.exr",
            "2048 x 2048",
            "1 changed",
            "group changed",
            "scenes/source.ma",
            "missing",
            # What a state means and how to fix it, as the panel's tooltip
            # says it: the re-pack's reason rides here, not in the state.
            "it sits in another folder",
            "Re-Bake",
        ):
            self.assertIn(expected, text)

    def test_its_links_select_and_open(self):
        html = self.report.doc(self.maps).to_html()
        # The map's readers in one link; its folder opens.
        self.assertIn("action://select?node=%7Cgrp%7CA&amp;node=%7Cgrp%7CB", html)
        self.assertIn("action://open?path=C%3A%2Fproj%2Fsourceimages%2Flightmaps", html)
        # Each object to re-bake, and each one out of the bake, selects itself.
        self.assertIn("action://select?node=%7CSTAND", html)

    def test_what_rebakes_says_why_and_how(self):
        text = self.report.doc(self.maps).to_text()
        self.assertIn("moved", text)
        self.assertIn("in place", text)
        self.assertIn("re-bakes with its atlas, which re-packs whole", text)

    def test_a_map_waiting_for_the_save_shows_its_folder_and_why(self):
        """Production: the Bake State listed two maps in a temp folder outside
        the project, unexplained. A map written since the last save shows the
        folder it reaches at the save, marked in italics, and says why. Added:
        2026-10-23"""
        self.maps[0] = dict(
            self.maps[0],
            path="C:/Temp/ptk/pending_files/pending_files_1-2/Room_Lightmap.exr",
            saves_to="C:/proj/sourceimages/lightmaps/Room_Lightmap.exr",
        )
        doc = self.report.doc(self.maps)
        html, text = doc.to_html(), doc.to_text()
        self.assertIn("action://open?path=C%3A%2Fproj%2Fsourceimages%2Flightmaps", html)
        self.assertNotIn("action://open?path=C%3A%2FTemp", html, "never the temp folder")
        self.assertIn("font-style:italic", html)
        self.assertIn(f"1 map {StaleReport.ON_SAVE}", text)
        rows = {r["map"]: r for r in self.report.map_rows(self.maps)}
        self.assertIn(StaleReport.UNSAVED, rows["Room_Lightmap.exr"]["hint"])
        self.assertNotIn(StaleReport.UNSAVED, rows["Desk_Lightmap.exr"]["hint"])

    def test_nothing_baked_says_so(self):
        text = StaleReport().doc([]).to_text()
        self.assertIn("Nothing in this scene is baked yet.", text)

    def test_map_rows_carry_each_maps_state_for_a_table(self):
        """The rows a panel's table shows read the state the breakdown does.
        A state names the map's condition now, never what a re-bake will do
        to it ("re-packs whole" read as an action nothing offered); the hint
        says why and which button remedies it. Changed: 2026-10-22"""
        rows = {row["map"]: row for row in self.report.map_rows(self.maps)}
        self.assertEqual(set(rows), {m["map"] for m in self.maps})
        room = rows["Room_Lightmap.exr"]
        self.assertEqual((room["state"], room["level"], room["stale"]), ("1 changed", "warn", 1))
        self.assertIn("Re-Bake", room["hint"])
        self.assertIn("own cell", room["hint"], "written in place")
        self.assertEqual(rows["Room_Lightmap.exr"]["objects"], ["|grp|A", "|grp|B"])
        desk = rows["Desk_Lightmap.exr"]
        self.assertEqual((desk["state"], desk["level"]), ("group changed", "warn"))
        self.assertIn("it sits in another folder", desk["hint"])
        self.assertIn("Re-Bake", desk["hint"])
        lost = rows["Lost_Lightmap.exr"]
        self.assertEqual((lost["state"], lost["level"]), ("missing", "error"))
        self.assertIn("sourceimages/lightmaps", lost["hint"])
        self.assertIn("Re-Bake", lost["hint"])
        # Its every reader hidden or excluded: no re-bake renders it, so Fix
        # All reverts it (fix_plan). A production room's missing maps sat on
        # hidden props brought in from other projects, their rows promising
        # "Fix All bakes it again", and Fix All left them. Added: 2026-10-23
        blocked = StaleReport(unbakeable=("|grp|C",)).map_rows(
            [{"map": "Lost_Lightmap.exr", "objects": ["|grp|C"], "path": None}]
        )[0]
        self.assertEqual(blocked["state"], "missing")
        self.assertNotIn("bakes it again", blocked["hint"])
        for words in ("hidden or excluded", "Fix All", "Revert"):
            self.assertIn(words, blocked["hint"])
        fresh =StaleReport().map_rows(self.maps[:1])[0]
        self.assertEqual((fresh["state"], fresh["level"]), ("up to date", "ok"))
        self.assertTrue(fresh["hint"])
        # A row with no readers and no probe flag is not a map the scene reads.
        self.assertEqual(StaleReport().map_rows([{"map": "x.exr", "objects": []}]), [])

    def test_a_split_untracked_or_copied_map_says_so_and_how_to_fix_it(self):
        """The remaining conditions, each with its remedy. A split names its
        materials -- duplicates still in the scene are merged by the fix, one
        material has nothing to merge -- and is no re-bake target: Combine
        mends it with no render. Changed: 2026-10-23 (it said the duplicates
        were "merged since" whether or not they were, and a split re-baked)."""
        room = self.maps[0]
        files = ("Room_Lightmap.exr", "Room_Lightmap_1.exr")
        dup = StaleReport(
            split={"Room_Lightmap.exr": files},
            split_materials={"Room_Lightmap.exr": ("Room", "Room1")},
        ).map_rows([room])[0]
        self.assertEqual((dup["state"], dup["level"], dup["stale"]), ("split", "warn", 0))
        for words in ("Room_Lightmap_1.exr", "Room, Room1", "Combine", "merges"):
            self.assertIn(words, dup["hint"])
        one = StaleReport(
            split={"Room_Lightmap.exr": files},
            split_materials={"Room_Lightmap.exr": ("Room",)},
        ).map_rows([room])[0]
        self.assertIn("one material (Room)", one["hint"])
        self.assertNotIn("merges", one["hint"])
        untracked = StaleReport(untracked=("|grp|B",)).map_rows([room])[0]
        self.assertEqual((untracked["state"], untracked["level"]), ("untracked", "info"))
        self.assertIn("Start Tracking", untracked["hint"])
        self.assertIn("Re-Bake", untracked["hint"])
        copied = StaleReport(
            targets=("|grp|A",),
            stale={"|grp|A": (R.MOVED,)},
            in_place=("|grp|A",),
            forked={"Room_Lightmap.exr": "scenes/source.ma"},
        ).map_rows([room])[0]
        self.assertEqual(copied["state"], "1 changed")
        self.assertIn("scenes/source.ma", copied["hint"])
        self.assertIn("copy", copied["hint"])

    def test_tally_counts_the_rows_states_worst_first(self):
        """The footer's line: short, worst first, every changed state one
        count. Added: 2026-10-23"""
        rows = [{"state": s} for s in ("up to date", "untracked", "2 changed",
                                       "split", "group changed", "split")]
        self.assertEqual(
            StaleReport.tally(rows), "2 split, 2 changed, 1 untracked, 1 up to date"
        )

    def test_fix_all_leaves_no_row_needing_a_fix(self):
        """Fix All's plan: a split no re-bake mends combines, the targets
        re-bake, a missing map no re-bake reaches comes off its hidden reader,
        and the untracked rest is tracked as it stands. It left untracked and
        missing rows in production: "Fix All is still not fixing all".
        Added: 2026-10-23"""
        rows = [
            {"map": "S.exr", "state": "split", "objects": ["|s"]},
            {"map": "J.exr", "state": "split", "objects": ["|m", "|j"]},
            {"map": "U.exr", "state": "untracked", "objects": ["|u"]},
            {"map": "H.exr", "state": "missing", "objects": ["|h"]},
            {"map": "M.exr", "state": "missing", "objects": ["|m"]},
        ]
        report = StaleReport(
            targets=("|m",),
            unbakeable=("|h",),
            untracked=("|s", "|u", "|h", "|m"),
        )
        self.assertEqual(
            report.fix_plan(rows),
            {
                "combine": ["S.exr"],  # J.exr re-packs with |m's re-bake
                "rebake": ["|m"],
                "revert": ["|h"],
                "track": ["|s", "|u"],
            },
        )
        nothing = StaleReport().fix_plan([{"map": "A.exr", "state": "up to date", "objects": ["|a"]}])
        self.assertFalse(any(nothing.values()))
        self.assertEqual(StaleReport.tally([]), "No maps.")

    def test_can_bake_is_the_rule_assess_leaves_out_unbakeable_by(self):
        """A DCC naming targets of its own asks the same rule ``assess``
        does. Added: 2026-10-22"""
        shown = {"role": BB.Role.BAKED, "state": {"rendered": True}}
        self.assertTrue(BB.can_bake(shown))
        self.assertFalse(BB.can_bake(dict(shown, state={"rendered": False})))
        self.assertFalse(BB.can_bake(dict(shown, role=BB.Role.EXCLUDED)))
        self.assertTrue(BB.can_bake(dict(shown, role=BB.Role.ISOLATED)))


if __name__ == "__main__":
    unittest.main()
