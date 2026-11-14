# !/usr/bin/python
# coding=utf-8
"""The articulated-rig math: ``ArticulationModel`` (the joint model and the grab
solver every runtime ports), ``ArticulationAnalysis`` (a rig proposed from the
parts' geometry) and ``ArticulationConformance`` (the golden cases the ports
are held to).

Pure Python, no DCC. The geometry fixtures are built here -- tubes, housings,
knobs and a ball as point shells -- in the shape the production magnifier
has: parts meeting at knobs that stand off the arm's plane, one tube running
inside another, a small sphere under the head.
"""

import copy
import json
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pythontk as ptk  # noqa: E402
from pythontk import ArticulationAnalysis, ArticulationConformance, ArticulationModel  # noqa: E402

ptk.TestSandbox.activate()


def _chain(*joints):
    return {"name": "rig", "joints": list(joints), "grab": []}


_IDENTITY_ANCHOR = {"t": [0.0, 0.0, 0.0], "q": [0.0, 0.0, 0.0, 1.0]}


def _joint(
    name, parent, t, q=(0.0, 0.0, 0.0, 1.0), order="xyz", anchor=None, **channels
):
    return {
        "name": name,
        "parent": parent,
        "t": list(t),
        "q": list(q),
        "anchor": copy.deepcopy(anchor or _IDENTITY_ANCHOR),
        "rotate_order": order,
        "channels": [
            {"channel": c, "min": lim[0], "max": lim[1], "weight": 1.0}
            for c, lim in channels.items()
        ],
    }


def _qdist(a, b):
    """Sign-insensitive quaternion distance."""
    return min(
        math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))),
        math.sqrt(sum((x + y) ** 2 for x, y in zip(a, b))),
    )


class TestModelForward(unittest.TestCase):
    def test_a_hinge_turns_its_child_about_the_rest_frames_z(self):
        model = ArticulationModel(
            _chain(
                _joint("a", None, (0, 0, 0), rz=(None, None)),
                _joint("b", 0, (10, 0, 0)),
            )
        )
        world = model.world([90.0])
        self.assertAlmostEqual(world[1][0][0], 0.0, places=9)
        self.assertAlmostEqual(world[1][0][1], 10.0, places=9)

    def test_the_rotate_order_is_mayas_x_first_for_xyz(self):
        # (0, 1, 0) -rx 90-> (0, 0, 1) -ry 90-> (1, 0, 0): x applied first.
        model = ArticulationModel(
            _chain(_joint("a", None, (0, 0, 0), rx=(None, None), ry=(None, None)))
        )
        q = model.world([90.0, 90.0])[0][1]
        v = ptk.Quaternion.rotate(q, (0.0, 1.0, 0.0))
        for got, want in zip(v, (1.0, 0.0, 0.0)):
            self.assertAlmostEqual(got, want, places=9)

    def test_a_slide_travels_along_the_rest_x_in_the_parents_space(self):
        half = math.sqrt(0.5)
        model = ArticulationModel(
            _chain(
                _joint("a", None, (0, 0, 0)),
                _joint("s", 0, (5, 0, 0), q=(0, 0, half, half), tx=(-1, 3)),
            )
        )
        (t, _q) = model.local([2.0], 1)
        # rest X of the slide is the parent's +Y (a 90 degree orient about Z)
        for got, want in zip(t, (5.0, 2.0, 0.0)):
            self.assertAlmostEqual(got, want, places=9)

    def test_every_rotate_order_round_trips_through_its_euler(self):
        rng = random.Random(4)
        for order in ("xyz", "yzx", "zxy", "xzy", "yxz", "zyx"):
            for _ in range(200):
                angles = [rng.uniform(-179, 179) for _ in range(3)]
                angles["xyz".index(order[1])] = rng.uniform(-89, 89)
                q = ptk.Quaternion.from_euler(order, angles)
                back = ptk.Quaternion.matrix_euler(order, ptk.Quaternion.to_matrix(q))
                self.assertLess(
                    _qdist(q, ptk.Quaternion.from_euler(order, back)), 1e-12, order
                )

    def test_validation_refuses_what_the_runtimes_cannot_pose(self):
        with self.assertRaises(ValueError):
            ArticulationModel(_chain(_joint("a", 0, (0, 0, 0))))  # parent not earlier
        with self.assertRaises(ValueError):
            ArticulationModel(_chain(_joint("a", None, (0, 0, 0), qx=(None, None))))
        with self.assertRaises(ValueError):
            ArticulationModel(_chain(_joint("a", None, (0, 0, 0), rz=(10, -10))))
        with self.assertRaises(ValueError):
            ArticulationModel(_chain(_joint("a", None, (0, 0, 0), order="xxz")))

    def test_a_rig_its_shape_refuses_is_refused_naming_its_problems(self):
        """The model builds only from a rig its declared shape admits
        (``ArticulationRig``), as unitytk's C# model does: a reference and a
        runtime that defaulted what the other refused would pose one rig two
        ways. A channel with no ``weight``, a joint with no ``parent``, a grab
        that is no object -- each is a ValueError naming its problem, never
        defaulted. A rig handed without its grab -- the hosts' IK solvers build
        from the joints alone -- grabs nothing. Added: 2026-10-20"""
        rig = _chain(
            _joint("a", None, (0, 0, 0), rz=(None, None)), _joint("b", 0, (10, 0, 0))
        )
        self.assertEqual(len(ArticulationModel(rig).channels), 1)
        cases = {
            "missing required key 'weight'": lambda r: r["joints"][0]["channels"][
                0
            ].pop("weight"),
            "missing required key 'parent'": lambda r: r["joints"][1].pop("parent"),
            "expected a mapping, got str": lambda r: r.update(grab=["head"]),
        }
        for problem, cut in cases.items():
            with self.subTest(problem=problem):
                broken = copy.deepcopy(rig)
                cut(broken)
                with self.assertRaises(ValueError) as raised:
                    ArticulationModel(broken)
                self.assertIn(problem, str(raised.exception))
        joints_alone = {"name": rig["name"], "joints": rig["joints"]}
        self.assertEqual(len(ArticulationModel(joints_alone).channels), 1)

    def test_from_record_finds_a_rig_by_name(self):
        payload = {
            "rigs": [
                dict(_chain(_joint("a", None, (0, 0, 0))), name="one"),
                dict(_chain(_joint("b", None, (0, 0, 0))), name="two"),
            ]
        }
        self.assertEqual(
            ArticulationModel.from_record(payload, "two").joints[0]["name"], "b"
        )
        self.assertEqual(ArticulationModel.from_record(payload).name, "one")
        with self.assertRaises(KeyError):
            ArticulationModel.from_record(payload, "three")


class TestModelRead(unittest.TestCase):
    def test_a_hinge_on_its_orders_middle_axis_reads_past_ninety(self):
        # rz is the MIDDLE of "xzy": a plain Euler read gives the other triple
        # past 90 degrees, and dropping its two 180s would read the hinge wrong.
        model = ArticulationModel(
            _chain(_joint("a", None, (0, 0, 0), order="xzy", rz=(None, None)))
        )
        for angle in (-170.0, -120.0, -91.0, 95.0, 135.0, 179.0):
            got = model.read(model.pose([angle]))
            self.assertAlmostEqual(got[0], angle, places=9)

    def test_a_ball_reads_the_triple_nearest_its_hint(self):
        model = ArticulationModel(
            _chain(
                _joint(
                    "a",
                    None,
                    (0, 0, 0),
                    rx=(None, None),
                    ry=(None, None),
                    rz=(None, None),
                )
            )
        )
        # The middle axis past 90: its two triples are (40, 120, -30) and
        # (-140, 60, 150). Each read picks the one nearer its hint, and either
        # is the same rotation.
        state = [40.0, 120.0, -30.0]
        other = [-140.0, 60.0, 150.0]
        for hint in (state, other, [0.0, 0.0, 0.0]):
            got = model.read(model.pose(state), hint=hint)
            self.assertLess(
                _qdist(model.world(got)[0][1], model.world(state)[0][1]), 1e-12
            )
        for hint in (state, other):
            got = model.read(model.pose(state), hint=hint)
            for g, w in zip(got, hint):
                self.assertAlmostEqual(g, w, places=9)

    def test_a_wound_hinge_reads_on_toward_its_hint(self):
        model = ArticulationModel(_chain(_joint("a", None, (0, 0, 0), rz=(None, None))))
        self.assertAlmostEqual(
            model.read(model.pose([30.0]), hint=[380.0])[0], 390.0, places=9
        )

    def test_a_half_turn_rounds_up_in_every_port(self):
        # floor(x + 0.5): what JS and C# spell identically, unlike round().
        self.assertEqual(ArticulationModel._unwrap(-180.0, 0.0), 180.0)
        self.assertEqual(ArticulationModel._unwrap(180.0, 0.0), 180.0)

    def test_no_unit_is_measured_the_record_is_in_metres(self):
        """Articulation v2 (2026-11-08) states the record in glTF's frame,
        in metres, and every reader's import already lands there (a glTF is
        in metres; an engine's FBX importer converts the file's unit), so the
        unit a runtime measured off the rest pose (``scale_of``) retired --
        it was no truth an anchored hinge keeps either: its translation moves
        as it turns. Added: 2026-11-08"""
        self.assertFalse(hasattr(ArticulationModel, "scale_of"))
        self.assertTrue(ptk.SceneRecords.ARTICULATION.in_frame)
        self.assertEqual(ptk.SceneRecords.FRAME.length, "metre")


class TestAnchor(unittest.TestCase):
    """Articulation v2's anchor frame: a part turns as ``rest . A .
    pose(channels) . A^-1``, its channels the anchor's -- a hinge about the
    anchor's axes through its point, a slide along them. The identity anchor
    is today's model, bit for bit. Added: 2026-11-08"""

    def setUp(self):
        self.rng = random.Random(23)

    def test_the_identity_anchor_is_the_model_without_one_bit_for_bit(self):
        """``rotation = q (x) Euler``, ``translation = t + q.(tx, ty, tz)``
        spelled out here, compared with ``==``: an identity anchor leaves the
        model exactly as v1 posed it."""
        q = ptk.Quaternion.normalize((0.1, -0.2, 0.3, 0.9))
        spec = _joint(
            "a", None, (1.0, 2.0, 3.0), q=q, order="zxy",
            rx=(None, None), rz=(None, None), tx=(None, None), ty=(None, None),
        )
        model = ArticulationModel(_chain(spec))
        for _ in range(50):
            state = [self.rng.uniform(-170, 170) for _ in range(2)] + [
                self.rng.uniform(-3, 3) for _ in range(2)
            ]
            rx, rz, tx, ty = state
            want_q = ptk.Quaternion.multiply(
                q, ptk.Quaternion.from_euler("zxy", (rx, 0.0, rz))
            )
            rest_t = tuple(float(v) for v in spec["t"])
            slide = ptk.Quaternion.rotate(q, (tx, ty, 0.0))
            want_t = (rest_t[0] + slide[0], rest_t[1] + slide[1], rest_t[2] + slide[2])
            self.assertEqual(model.local(state, 0), (want_t, want_q))

    def test_an_anchored_hinge_turns_about_the_anchors_axis_through_its_point(self):
        half = math.sqrt(0.5)
        # The anchor at (2, 0, 0) of the rest frame, its z the rest frame's -y.
        anchor = {"t": [2.0, 0.0, 0.0], "q": [half, 0.0, 0.0, half]}
        model = ArticulationModel(
            _chain(
                _joint("a", None, (0, 0, 0), anchor=anchor, rz=(None, None)),
                _joint("b", 0, (4, 0, 0)),
            )
        )
        state = [90.0]
        # The anchor's point stays where it is; the child's origin, 2 along
        # the rest x from it, turns 90 degrees about the rest -y: to +z.
        pivot = model.point(state, 0, (2.0, 0.0, 0.0))
        for got, want in zip(pivot, (2.0, 0.0, 0.0)):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(model.world(state)[1][0], (2.0, 0.0, 2.0)):
            self.assertAlmostEqual(got, want, places=12)

    def test_an_anchored_slide_runs_along_the_anchors_axes(self):
        half = math.sqrt(0.5)
        anchor = {"t": [0.5, 0.5, 0.5], "q": [0.0, 0.0, half, half]}  # x -> +y
        model = ArticulationModel(
            _chain(_joint("s", None, (1, 0, 0), anchor=anchor, tx=(-2, 2)))
        )
        (t, q) = model.local([1.5], 0)
        for got, want in zip(t, (1.0, 1.5, 0.0)):
            self.assertAlmostEqual(got, want, places=12)
        self.assertLess(_qdist(q, (0.0, 0.0, 0.0, 1.0)), 1e-12)

    @staticmethod
    def _nested(rig):
        """``(rig, anchors)``: *rig* rebuilt with every joint ON its anchor --
        each rest the anchor frame's, the anchor the identity -- the same
        motion a DCC's nested rig makes (its frame is the anchored joint's
        times its anchor)."""
        model = ArticulationModel(rig)
        anchors = [
            (j["anchor"]["t"], ptk.Quaternion.normalize(j["anchor"]["q"]))
            for j in rig["joints"]
        ]
        frames = []
        for (p, q), (at, aq) in zip(model.world(model.rest_state()), anchors):
            point = ptk.Quaternion.rotate(q, at)
            frames.append(
                (tuple(a + b for a, b in zip(p, point)), ptk.Quaternion.multiply(q, aq))
            )
        nested = copy.deepcopy(rig)
        for index, joint in enumerate(nested["joints"]):
            parent = joint["parent"]
            pp, pq = (
                ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
                if parent is None
                else frames[parent]
            )
            p, q = frames[index]
            back = ptk.Quaternion.conjugate(pq)
            joint["t"] = list(ptk.Quaternion.rotate(back, [a - b for a, b in zip(p, pp)]))
            joint["q"] = list(ptk.Quaternion.multiply(back, q))
            joint["anchor"] = copy.deepcopy(_IDENTITY_ANCHOR)
        return nested, anchors

    def test_read_and_solve_respect_the_anchor(self):
        """Every anchored rig is the rig nested on its anchors: it poses
        alike, reads its own poses back, and a grab solves it to the same
        state -- the anchor's columns, pivots, reach and wrist are the nested
        rig's -- for a grab dragged from rest, and lands one that starts
        near its target."""
        rigs = ArticulationConformance.rigs()
        for name in ("anchored", "flat", "anchored_tree"):
            model = ArticulationModel(rigs[name])
            nested_rig, anchors = self._nested(rigs[name])
            nested = ArticulationModel(nested_rig)
            joint = len(model.joints) - 1
            at, aq = anchors[joint]
            with self.subTest(rig=name):
                for _ in range(20):
                    state = ArticulationConformance._random_state(model, self.rng)
                    read = model.read(model.pose(state), hint=state)
                    for got, want in zip(read, state):
                        self.assertAlmostEqual(got, want, places=7)
                    for (p, q), (np_, nq), (pt, pq) in zip(
                        model.world(state), nested.world(state), anchors
                    ):
                        on = [a + b for a, b in zip(p, ptk.Quaternion.rotate(q, pt))]
                        self.assertLess(math.dist(on, np_), 1e-12)
                        self.assertLess(_qdist(ptk.Quaternion.multiply(q, pq), nq), 1e-12)
                for _ in range(20):
                    goal = ArticulationConformance._random_state(model, self.rng)
                    local = [self.rng.uniform(-1, 1) for _ in range(3)]
                    # The same held point, in the nested joint's frame.
                    there = ptk.Quaternion.rotate(
                        ptk.Quaternion.conjugate(aq), [a - b for a, b in zip(local, at)]
                    )
                    target = list(model.point(goal, joint, local))
                    state, twin = model.rest_state(), nested.rest_state()
                    start = model.point(state, joint, local)
                    for f in range(1, 31):
                        step = [a + (b - a) * f / 30 for a, b in zip(start, target)]
                        state = model.solve(state, joint, local, step)
                        twin = nested.solve(twin, joint, there, step)
                    self.assertEqual(state, model.clamp(state))
                    for got, want in zip(state, twin):
                        self.assertAlmostEqual(got, want, delta=1e-4)
                    near = model.clamp([v + self.rng.uniform(-3, 3) for v in goal])
                    landed = model.solve(near, joint, local, target)
                    reach = model._reach(model.chain(joint), local)
                    miss = math.dist(model.point(landed, joint, local), target)
                    self.assertLess(miss / reach, 0.01)

    def test_an_anchored_ball_takes_the_hands_rotation(self):
        """The wrist split under an anchor: the ball turns about its anchor's
        point, so the chain above places THAT point, the held point lands,
        and -- wherever no limit holds the chain -- the ball's world rotation
        is the hand's, over a hold of a few frames from near the pose the
        hand asks for. The ball's own turn never moves its anchor's point. Pinned 2026-11-08: the point landing alone let a
        solve that dropped the hand's rotation, levered the whole held point
        or skipped the anchor's frame pass."""
        model = ArticulationModel(ArticulationConformance.rigs()["anchored"])
        joint = len(model.joints) - 1
        ball = [s for s, (j, _c) in enumerate(model.channels) if j == joint]
        self.assertEqual({model.channels[s][1] for s in ball}, {"rx", "ry", "rz"})
        pivot = model.joints[joint]["anchor"][0]
        self.assertGreater(math.dist(pivot, (0.0, 0.0, 0.0)), 0.1, "off the pivot")
        chain = [s for j in model.chain(joint) for s, (jj, _c) in enumerate(model.channels) if jj == j]

        def free(state):
            return all(
                (lo is None or state[s] > lo + 1e-9) and (hi is None or state[s] < hi - 1e-9)
                for s in chain
                for lo, hi in [model.limits(s)]
            )

        turned_to = 0
        for _ in range(20):
            goal = ArticulationConformance._random_state(model, self.rng)
            local = [self.rng.uniform(-1, 1) for _ in range(3)]
            target = list(model.point(goal, joint, local))
            rotation = model.world(goal)[joint][1]
            # A hand holding the part a few frames, from a pose near the goal.
            state = model.clamp([v + self.rng.uniform(-10, 10) for v in goal])
            for _frame in range(10):
                state = model.solve(state, joint, local, target, rotation)
            self.assertLess(math.dist(model.point(state, joint, local), target), 1e-2)
            if free(state):  # no limit held the chain: the ball took the turn
                turned_to += 1
                self.assertLess(
                    _qdist(model.world(state)[joint][1], rotation),
                    1e-9,
                    "the ball takes the hand's rotation",
                )
            # The ball's own turn leaves its anchor's point where it was.
            turned = list(goal)
            for s in ball:
                turned[s] = model.clamp(
                    [v + (self.rng.uniform(-30, 30) if i == s else 0.0) for i, v in enumerate(goal)]
                )[s]
            self.assertLess(
                math.dist(
                    model.point(goal, joint, pivot), model.point(turned, joint, pivot)
                ),
                1e-12,
            )
        self.assertGreater(turned_to, 10, "most holds leave every channel free")

    def test_a_flat_model_chains_by_its_joints_alone(self):
        """A flat model -- every part a child of the root, its pivot at the
        root's origin -- is a rig of identity rests, its hinges in its
        anchors: each part's pose is its parent JOINT's times its own turn,
        whatever its node hangs under, so the second hinge rides the first."""
        rig = ArticulationConformance.rigs()["flat"]
        model = ArticulationModel(rig)
        self.assertTrue(
            all(j["t"] == [0.0, 0.0, 0.0] and j["q"] == [0.0, 0.0, 0.0, 1.0]
                for j in rig["joints"])
        )
        self.assertTrue(all(j["anchor"] != _IDENTITY_ANCHOR for j in rig["joints"]))
        state = ArticulationConformance._random_state(model, self.rng)
        world = model.world(state)
        for index, joint in enumerate(model.joints):
            p, q = model.local(state, index)
            if joint["parent"] is not None:
                pp, pq = world[joint["parent"]]
                p = tuple(a + b for a, b in zip(pp, ptk.Quaternion.rotate(pq, p)))
                q = ptk.Quaternion.multiply(pq, q)
            self.assertLess(math.dist(p, world[index][0]), 1e-12)
            self.assertLess(_qdist(q, world[index][1]), 1e-12)
        # The child's anchor point rides the parent's turn.
        child = next(i for i, j in enumerate(model.joints) if j["parent"] is not None)
        point = rig["joints"][child]["anchor"]["t"]
        parent = model.joints[child]["parent"]
        self.assertLess(
            math.dist(
                model.point(state, child, point), model.point(state, parent, point)
            ),
            1e-12,
        )

    def test_a_rig_with_no_anchor_is_refused_naming_its_fix(self):
        """A version 1 rig -- joints without an anchor -- is not v2's shape:
        refused, naming the fix (a re-export), never defaulted."""
        rig = _chain(_joint("a", None, (0, 0, 0), rz=(None, None)))
        del rig["joints"][0]["anchor"]
        with self.assertRaises(ValueError) as raised:
            ArticulationModel(rig)
        self.assertIn("missing required key 'anchor'", str(raised.exception))
        self.assertIn("re-export", str(raised.exception))


class TestModelSolve(unittest.TestCase):
    def setUp(self):
        self.rigs = ArticulationConformance.rigs()
        self.rng = random.Random(11)

    def _reachable(self, model, joint):
        state = ArticulationConformance._random_state(model, self.rng)
        local = [self.rng.uniform(-1, 1) for _ in range(3)]
        return state, local, list(model.point(state, joint, local))

    def test_a_grab_followed_frame_by_frame_lands_on_its_target(self):
        model = ArticulationModel(self.rigs["arm"])
        joint = len(model.joints) - 1
        for _ in range(20):
            goal, local, target = self._reachable(model, joint)
            state = model.rest_state()
            start = model.point(state, joint, local)
            for f in range(1, 31):
                step = [a + (b - a) * f / 30 for a, b in zip(start, target)]
                state = model.solve(state, joint, local, step)
            reach = model._reach(model.chain(joint), local)
            miss = math.dist(model.point(state, joint, local), target) / reach
            self.assertLess(miss, 0.03)
            self.assertEqual(state, model.clamp(state))

    def test_a_ball_takes_the_hands_rotation_and_the_point_still_lands(self):
        model = ArticulationModel(self.rigs["arm"])
        joint = len(model.joints) - 1
        hits = 0
        for _ in range(20):
            goal, local, target = self._reachable(model, joint)
            rotation = model.world(goal)[joint][1]
            state = model.solve(goal, joint, local, target, rotation)  # already there
            self.assertLess(math.dist(model.point(state, joint, local), target), 1e-3)
            near = [v + self.rng.uniform(-2, 2) for v in goal]
            state = model.solve(model.clamp(near), joint, local, target, rotation)
            reach = model._reach(model.chain(joint), local)
            if math.dist(model.point(state, joint, local), target) / reach < 0.01:
                hits += 1
            self.assertEqual(state, model.clamp(state))
        self.assertGreaterEqual(hits, 18)

    def test_the_links_past_the_held_one_ride_along_untouched(self):
        model = ArticulationModel(self.rigs["tree"])
        state = ArticulationConformance._random_state(model, self.rng)
        # hold "left" (joint 1): only hub and left may move; right and tip keep theirs
        solved = model.solve(state, 1, [0.5, 0.5, 0.0], [3.0, 4.0, 1.0])
        for slot, (joint, _c) in enumerate(model.channels):
            if joint in (2, 3):
                self.assertEqual(solved[slot], state[slot])

    def test_a_target_out_of_reach_keeps_every_channel_inside_its_limits(self):
        model = ArticulationModel(self.rigs["arm"])
        state = model.solve(model.rest_state(), 4, [0, 0, 0], [500.0, 500.0, 0.0])
        self.assertEqual(state, model.clamp(state))
        self.assertTrue(any(state[s] in model.limits(s) for s in range(len(state))))

    def test_a_target_dragged_out_of_reach_moves_the_arm_smoothly(self):
        """Bug: past the reach the damped least squares overshot and thrashed
        -- every solve, from rest (Maya's end control) or from the last
        frame's pose (a runtime's hand grab), landed on another flail: joints
        jumped 12-29 units on a 0.46 step of the target (this arm), and 15-34
        on the production magnifier. A step that would leave the held point
        further from its target is halved until it helps, so the arm settles
        where it comes nearest and follows a dragged target. (Solved from rest
        every time, a redundant arm may still switch once between two poses
        that both land -- a base swivel turning the other way -- so a cold
        drag is allowed a jump or two, a warm one none.) Fixed: 2026-10-04."""
        model = ArticulationModel(self.rigs["arm"])
        joint = len(model.joints) - 1
        local = [0.5, 0.2, -0.3]
        reach = model._reach(model.chain(joint), local)
        start = model.point(model.rest_state(), joint, local)
        rotation = model.world(model.rest_state())[joint][1]
        steps = 60  # the old step: 14-32 jumps per drag here
        for direction in ((1, 0, 0), (-1, 0.4, 0.2)):
            length = math.sqrt(sum(c * c for c in direction))
            for warm in (False, True):
                state, before, jumps = model.rest_state(), None, 0
                for i in range(steps + 1):
                    far = 1.5 * reach * i / steps / length
                    target = [s + c * far for s, c in zip(start, direction)]
                    seed = state if warm else model.rest_state()
                    state = model.solve(seed, joint, local, target, rotation)
                    if before is not None:
                        moved = zip(model.world(before), model.world(state))
                        # the target steps 0.77: a joint moving 4 is a jump
                        jumps += max(math.dist(a, b) for (a, _), (b, _) in moved) > 4.0
                    before = state
                self.assertLessEqual(
                    jumps,
                    0 if warm else 2,
                    f"{direction}, {'warm' if warm else 'cold'}",
                )

    def test_a_rig_scaled_by_a_thousand_solves_alike(self):
        base = self.rigs["arm"]
        big = json.loads(json.dumps(base))
        for joint in big["joints"]:
            joint["t"] = [v * 1000.0 for v in joint["t"]]
            for ch in joint["channels"]:
                if ch["channel"].startswith("t"):
                    ch["min"], ch["max"] = ch["min"] * 1000.0, ch["max"] * 1000.0
        small, large = ArticulationModel(base), ArticulationModel(big)
        target = [3.0, 12.0, 4.0]
        a = small.solve(small.rest_state(), 4, [1, 0, 0], target)
        b = large.solve(
            large.rest_state(), 4, [1000, 0, 0], [v * 1000.0 for v in target]
        )
        for slot, (x, y) in enumerate(zip(a, b)):
            if small.channels[slot][1].startswith("t"):
                y /= 1000.0
            self.assertAlmostEqual(x, y, places=6)


# --------------------------------------------------------------- analysis
def _tube(p0, p1, radius, rings=6, sides=10):
    """Points on a capped tube from p0 to p1."""
    p0, p1 = [float(v) for v in p0], [float(v) for v in p1]
    axis = [b - a for a, b in zip(p0, p1)]
    length = math.sqrt(sum(v * v for v in axis))
    u = [v / length for v in axis]
    helper = (0.0, 0.0, 1.0) if abs(u[2]) < 0.9 else (1.0, 0.0, 0.0)
    e1 = ArticulationModel._cross(u, helper)
    n1 = math.sqrt(sum(v * v for v in e1))
    e1 = [v / n1 for v in e1]
    e2 = ArticulationModel._cross(u, e1)
    pts = []
    for i in range(rings):
        c = [a + (b - a) * i / (rings - 1) for a, b in zip(p0, p1)]
        for k in range(sides):
            ang = 2 * math.pi * k / sides
            pts.append(
                [
                    c[j] + radius * (math.cos(ang) * e1[j] + math.sin(ang) * e2[j])
                    for j in range(3)
                ]
            )
    return pts


def _sphere(centre, radius, n=6):
    pts = []
    for i in range(1, n):
        lat = math.pi * i / n - math.pi / 2
        for k in range(2 * n):
            lon = math.pi * k / n
            pts.append(
                [
                    centre[0] + radius * math.cos(lat) * math.cos(lon),
                    centre[1] + radius * math.sin(lat),
                    centre[2] + radius * math.cos(lat) * math.sin(lon),
                ]
            )
    pts += [
        [centre[0], centre[1] + radius, centre[2]],
        [centre[0], centre[1] - radius, centre[2]],
    ]
    return pts


def _knob(centre, span=6.0, thick=1.0):
    """A wing knob: long across the arm's plane's X, standing off it in Z."""
    cx, cy, cz = centre
    return [
        [cx + x, cy + y, cz + z]
        for x in (-span / 2, span / 2)
        for y in (-thick, thick)
        for z in (-thick / 2, thick / 2)
    ] + [[cx, cy, cz + thick]]


def _arm():
    """The production magnifier's shape, in its plane z = 0: plate + post, a
    pole on a knob swivel, an elbow knob, an upper tube and its collar knob,
    an outer rod with an inner rod inside it, and a head on a small ball."""
    base = [
        [[x, 0.0, z] for x in (-7, 7) for z in (-7, 7)]
        + [[x, 0.9, z] for x in (-7, 7) for z in (-7, 7)],
        _tube((0, 0.9, 0), (0, 7, 0), 2.0),  # post
    ]
    pole = [
        _tube((0, 4, 0), (0, 10, 0), 2.5),  # bottom housing
        _tube((0, 10, 0), (0, 29, 0), 2.47, rings=10),  # pole
        _knob((0, 8, 3.4)),  # swivel's tilt knob
    ]
    elbow = (0.0, 32.0, 0.0)
    d = (0.8, 0.6, 0.0)
    collar = tuple(e + 25 * k for e, k in zip(elbow, d))
    upper = [
        _tube(elbow, collar, 2.47, rings=10),
        _knob((elbow[0], elbow[1], 3.4)),  # elbow knob
        _knob((collar[0], collar[1], -3.4)),  # collar knob
    ]
    r = (0.966, 0.259, 0.0)
    rod0 = collar
    rod1 = tuple(c + 32 * k for c, k in zip(collar, r))
    outer = [_tube(rod0, rod1, 2.0, rings=12)]
    inner0 = tuple(c + 20 * k for c, k in zip(collar, r))
    inner1 = tuple(c + 40 * k for c, k in zip(collar, r))
    inner = [_tube(inner0, inner1, 1.1, rings=10)]
    ball = tuple(c + 0.7 * k for c, k in zip(inner1, r))
    head = [
        _sphere(ball, 0.7),
        _tube(
            tuple(b + 12 * k for b, k in zip(ball, r)),
            tuple(b + 12.3 * k for b, k in zip(ball, r)),
            11.0,
            rings=2,
            sides=24,
        ),
        _tube(
            ball, tuple(b + 1.5 * k for b, k in zip(ball, r)), 0.35
        ),  # a stub into the ring
    ]
    names = ["BASE", "LEG_1", "LEG_2", "LEG_3", "LEG_4", "HEAD"]
    parts = [
        {"name": n, "shells": s}
        for n, s in zip(names, [base, pole, upper, outer, inner, head])
    ]
    return parts, {
        "elbow": elbow,
        "collar": collar,
        "rod1": rod1,
        "inner0": inner0,
        "ball": ball,
    }


class TestAnalysis(unittest.TestCase):
    def test_the_magnifier_shape_proposes_its_five_joints(self):
        parts, at = _arm()
        proposal = ArticulationAnalysis.propose(parts)
        self.assertEqual(proposal["order"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(
            [round(abs(v), 3) for v in proposal["normal"]], [0.0, 0.0, 1.0]
        )
        kinds = [j["type"] for j in proposal["joints"]]
        self.assertEqual(kinds, ["universal", "hinge", "hinge", "slide", "ball"])
        joints = {j["part"]: j for j in proposal["joints"]}
        close = lambda a, b, tol=0.25: self.assertLess(math.dist(a, b), tol, (a, b))  # noqa: E731
        close(
            joints[1]["position"], (0.0, 8.0, 0.0)
        )  # the tilt knob, on the post's axis
        close(joints[2]["position"], at["elbow"])
        close(joints[3]["position"], at["collar"])
        close(joints[4]["position"], at["rod1"])  # the outer rod's far end
        close(joints[5]["position"], at["ball"])
        lo, hi = joints[4]["limits"]["tx"]
        self.assertAlmostEqual(
            hi, 12.0 * (1 - ArticulationAnalysis.MIN_INSERTION), places=1
        )
        self.assertLess(lo, 0.0)
        # the slide runs along the rods; a knob's hinge axis is the plane's normal
        for got, want in zip(joints[4]["aim"], (0.966, 0.259, 0.0)):
            self.assertAlmostEqual(got, want, places=2)

    def test_a_selection_order_is_taken_as_the_chain(self):
        parts, _at = _arm()
        swapped = [parts[i] for i in (0, 1, 2, 3, 4, 5)]
        proposal = ArticulationAnalysis.propose(swapped, order=[0, 1, 2, 3, 4, 5])
        self.assertEqual(proposal["parents"], {1: 0, 2: 1, 3: 2, 4: 3, 5: 4})

    def test_a_part_nothing_touches_is_left_out(self):
        parts, _at = _arm()
        parts.append({"name": "STRAY", "shells": [_sphere((500, 500, 500), 1.0)]})
        proposal = ArticulationAnalysis.propose(parts)
        self.assertEqual(proposal["unreached"], [6])
        self.assertNotIn(6, proposal["order"])

    def test_a_part_without_geometry_is_refused(self):
        parts, _at = _arm()
        parts[2]["shells"] = []
        with self.assertRaises(ValueError):
            ArticulationAnalysis.propose(parts)

    def test_a_straight_arm_proposes_no_plane(self):
        """Centroids on one line span no plane: its "normal" would be the
        SVD's rounding noise, so each hinge falls back to an axis across its
        own aim instead."""
        d = [0.3, 0.5, 0.81]
        n = math.sqrt(sum(v * v for v in d))
        d = [v / n for v in d]
        at = lambda k: [k * v for v in d]  # noqa: E731
        parts = [
            {"name": f"P{i}", "shells": [_tube(at(10 * i), at(10 * i + 10), r)]}
            for i, r in enumerate((2.0, 1.5, 1.0, 0.6))
        ]
        proposal = ArticulationAnalysis.propose(parts)
        self.assertIsNone(proposal["normal"])
        for joint in proposal["joints"]:
            self.assertAlmostEqual(
                sum(a * b for a, b in zip(joint["aim"], joint["normal"])), 0.0
            )

    def test_a_ball_is_the_smallest_round_shell_where_the_parts_meet(self):
        facts = ArticulationAnalysis.shell(_sphere((0, 0, 0), 1.0))
        self.assertGreater(facts.isotropy, 0.9)
        self.assertLess(facts.sphericity, 0.05)

    def test_a_tubes_ends_are_its_caps_not_its_centroid_plus_half(self):
        pts = (
            _tube((0, 0, 0), (10, 0, 0), 1.0) + [[0.0, 0.0, 0.0]] * 40
        )  # detail at one end
        facts = ArticulationAnalysis.shell(pts)
        ends = sorted(p[0] for p in facts.ends)
        self.assertAlmostEqual(ends[0], 0.0, places=6)
        self.assertAlmostEqual(ends[1], 10.0, places=6)


class TestConformance(unittest.TestCase):
    def test_the_document_is_plain_json_and_repeatable(self):
        a = ArticulationConformance.cases(seed=3, per_rig=2)
        b = ArticulationConformance.cases(seed=3, per_rig=2)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertEqual(set(a["tolerance"]), {"pose", "world", "read", "solve"})
        self.assertEqual(len(a["cases"]), 2 * len(ArticulationConformance.rigs()))
        self.assertTrue(all("scale" not in case for case in a["cases"]))

    def test_the_anchored_cases_are_appended_so_every_older_index_stands(self):
        """v2's anchored rigs come after v1's eight, so a port's case ``i``
        is the case it was: the same rig at the same state. Added:
        2026-11-08"""
        names = list(ArticulationConformance.rigs())
        v1 = ["arm", "tree"] + [
            f"ball_{o}" for o in ("xyz", "yzx", "zxy", "xzy", "yxz", "zyx")
        ]
        self.assertEqual(names[: len(v1)], v1)
        self.assertEqual(
            names[len(v1) :],
            [
                "anchored",
                "flat",
                "anchored_tree",
                "offset_only",
                "turn_only",
                "hair_off_t",
                "hair_off_q",
                "negated_identity",
                "unnormalised_identity",
                "unnormalised_anchor",
            ],
        )
        doc = ArticulationConformance.cases()
        self.assertEqual(
            [c["rig"] for c in doc["cases"]][: 4 * len(v1)],
            [n for n in v1 for _ in range(4)],
        )

    def test_the_edge_rigs_sit_on_either_side_of_the_identity(self):
        """The anchored arm with its anchors on the identity's edge: a pure
        offset (an engine's default joint: the point off the pivot, the rest's axes)
        and a pure turn, a hair off the identity in each -- each anchored --
        and the identity spelt as -q and unnormalised, which take the
        anchorless path; and the arm's own anchors unnormalised. A port that
        ignores t or q, treats the identity by a tolerance, or reads an
        anchor's q unnormalised, fails these. Added: 2026-11-08"""
        rigs = ArticulationConformance.rigs()
        base = rigs["anchored"]
        for name, anchored in (
            ("offset_only", True),
            ("turn_only", True),
            ("hair_off_t", True),
            ("hair_off_q", True),
            ("negated_identity", False),
            ("unnormalised_identity", False),
            ("unnormalised_anchor", True),
        ):
            with self.subTest(rig=name):
                rig = rigs[name]
                self.assertEqual(
                    [{k: v for k, v in j.items() if k != "anchor"} for j in rig["joints"]],
                    [{k: v for k, v in j.items() if k != "anchor"} for j in base["joints"]],
                )
                model = ArticulationModel(rig)
                self.assertEqual(
                    [j["anchor"] is not None for j in model.joints],
                    [anchored] * len(model.joints),
                )
        offset = rigs["offset_only"]["joints"]
        self.assertTrue(all(j["anchor"]["q"] == [0.0, 0.0, 0.0, 1.0] for j in offset))
        self.assertTrue(all(any(j["anchor"]["t"]) for j in offset))
        turn = rigs["turn_only"]["joints"]
        self.assertTrue(all(j["anchor"]["t"] == [0.0, 0.0, 0.0] for j in turn))
        self.assertTrue(all(j["anchor"]["q"] != [0.0, 0.0, 0.0, 1.0] for j in turn))

    def test_the_anchored_rigs_cover_hinges_slides_a_ball_and_a_branch(self):
        rigs = ArticulationConformance.rigs()
        identity = {"t": [0.0, 0.0, 0.0], "q": [0.0, 0.0, 0.0, 1.0]}
        for name in ("arm", "tree") + tuple(n for n in rigs if n.startswith("ball_")):
            self.assertTrue(all(j["anchor"] == identity for j in rigs[name]["joints"]))
        anchored = [
            j
            for name in ("anchored", "flat", "anchored_tree")
            for j in rigs[name]["joints"]
            if j["anchor"] != identity
        ]
        kinds = {c["channel"] for j in anchored for c in j["channels"]}
        self.assertEqual(kinds, {"rx", "ry", "rz", "tx", "ty", "tz"})
        parents = [j["parent"] for j in rigs["anchored_tree"]["joints"]]
        self.assertGreater(len(parents), len(set(parents)), "a joint with two children")

    def test_the_reference_passes_its_own_cases(self):
        doc = ArticulationConformance.cases(seed=5, per_rig=3)
        for case in doc["cases"]:
            model = ArticulationModel(doc["rigs"][case["rig"]])
            for (t, q), (et, eq) in zip(model.pose(case["state"]), case["pose"]):
                self.assertLess(math.dist(t, et), doc["tolerance"]["pose"])
                self.assertLess(_qdist(q, eq), doc["tolerance"]["pose"])
            read = model.read(case["read"]["locals"], case["read"]["hint"])
            for got, want in zip(read, case["read"]["expect"]):
                self.assertAlmostEqual(got, want, delta=doc["tolerance"]["read"])
            for solve in case["solve"]:
                got = model.solve(
                    solve["from"],
                    solve["joint"],
                    solve["local"],
                    solve["target"],
                    solve["rotation"],
                )
                for x, y in zip(got, solve["expect"]):
                    self.assertAlmostEqual(x, y, delta=doc["tolerance"]["solve"])

    def test_every_joint_type_and_rotate_order_is_covered(self):
        rigs = ArticulationConformance.rigs()
        orders = {j["rotate_order"] for r in rigs.values() for j in r["joints"]}
        self.assertEqual(orders, {"xyz", "yzx", "zxy", "xzy", "yxz", "zyx"})
        channels = {
            c["channel"]
            for r in rigs.values()
            for j in r["joints"]
            for c in j["channels"]
        }
        self.assertEqual(channels, {"rx", "ry", "rz", "tx", "ty", "tz"})


class TestRecordShape(unittest.TestCase):
    """``ArticulationRecord`` -- the record's payload, declared once. The rigs
    every port is pinned with are records of that shape, so the declaration is
    held to real payloads rather than to itself."""

    def test_the_conformance_rigs_are_records_of_the_declared_shape(self):
        payload = {"version": 2, "rigs": list(ArticulationConformance.rigs().values())}
        res = ptk.ArticulationRecord.validate(payload)
        self.assertEqual(res.errors, [])
        self.assertEqual(res.warnings, [])

    def test_the_shape_is_the_records_declared_one(self):
        self.assertIs(ptk.SceneRecords.shape("articulation"), ptk.ArticulationRecord)
        # One layout, two containers: a GLB carries the record itself.
        self.assertTrue(ptk.SceneRecords.ARTICULATION.glb_verbatim)

    def test_v2_is_in_glTFs_frame_with_an_anchor_on_every_joint(self):
        """Articulation v2 (2026-11-08): every joint carries ``anchor: {t,
        q}`` (required; the identity by default), the record is in glTF's
        frame (metres), a GLB carries it as it stands, and a version 1 record
        is refused, its version's help naming the fix. Added: 2026-11-08"""
        spec = ptk.SceneRecords.ARTICULATION
        self.assertEqual(spec.version, 2)
        self.assertTrue(spec.in_frame)
        self.assertTrue(spec.glb_verbatim)
        schema = ptk.ArticulationRecord.json_schema()["$defs"]
        self.assertIn("anchor", schema["ArticulationJoint"]["required"])
        self.assertEqual(
            set(schema["ArticulationAnchor"]["required"]), {"t", "q"}
        )
        joint = ptk.ArticulationRecord.skeleton()["rigs"][0]["joints"][0]
        self.assertEqual(
            joint["anchor"], {"t": [0.0, 0.0, 0.0], "q": [0.0, 0.0, 0.0, 1.0]}
        )
        v1 = {"version": 1, "rigs": []}
        self.assertFalse(spec.is_current(v1))
        (version,) = [
            f for f in ptk.ArticulationRecord.describe() if f.name == "version"
        ]
        self.assertIn("version 1 record", version.help)
        self.assertIn("re-exported", version.help)

    def test_the_retired_block_shapes_name_their_new_home(self):
        """The GLB block (``ArticulationBlock``, its parts and the retired
        ``ArticulationWeb`` aliases) left with v2: a GLB carries the record
        itself. Reading one at the root names where it went. The
        ``ArticulationWeb`` aliases, noticed for 0.18.0 on 2026-10-24, are
        held, deprecated, until their window closes (``HELD_UNTIL_0_22``;
        ``test_scene_records.TestRetiredUntil022``). Added: 2026-11-08"""
        with self.assertRaises(AttributeError) as raised:
            getattr(ptk, "ArticulationBlock")
        self.assertIn("ArticulationRecord", str(raised.exception))
        self.assertEqual(
            ptk.HELD_UNTIL_0_22["ArticulationWeb"], ("pythontk.ArticulationRecord", "0.22.0")
        )

    def test_a_drifted_payload_is_named_where_it_drifted(self):
        rig = json.loads(json.dumps(ArticulationConformance.rigs()["arm"]))
        rig["joints"][1]["rotateOrder"] = rig["joints"][1].pop("rotate_order")
        rig["joints"][2]["channels"][0]["min"] = "-90"
        res = ptk.ArticulationRecord.validate({"version": 2, "rigs": [rig]})
        self.assertEqual(
            res.errors,
            [
                "rigs[0].joints[1].missing required key 'rotate_order'",
                "rigs[0].joints[2].channels[0].min: expected number, got string",
            ],
        )

    def test_the_vocabularies_are_the_models(self):
        from pythontk.geo_utils.articulation.model import CHANNELS, ROTATE_ORDERS

        joint = ptk.ArticulationRecord.json_schema()["$defs"]["ArticulationJoint"]
        channel = ptk.ArticulationRecord.json_schema()["$defs"]["ArticulationChannel"]
        self.assertEqual(
            joint["properties"]["rotate_order"]["enum"], list(ROTATE_ORDERS)
        )
        self.assertEqual(channel["properties"]["channel"]["enum"], list(CHANNELS))


if __name__ == "__main__":
    unittest.main()
