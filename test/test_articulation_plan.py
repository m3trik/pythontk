# !/usr/bin/python
# coding=utf-8
"""``ArticulationPlan`` -- the plan a DCC's ``ArticulatedRig`` builds from.

Pure Python, no DCC. The plan here is made the way mayatk's
``ArticulatedRig.create`` makes it: the production magnifier's parts (the
``test_articulation`` point shells) placed in a world that scales them by 2,
turns them and moves them, analysed, each joint moved into the rig group's
space, checked, and given an end control on the head. The cases are the ones
mayatk's ``test_articulated_rig.py`` pins on the built rig, stated of the
plan: the proposal, the checks, the rest the record ships (the model the
runtimes pose), the end control, the post-rig edits and the pivots adjusted,
and the panel's limits text. That mayatk's numbers are Maya's own -- the
joint orient read exactly as ``MTransformationMatrix`` reads it -- is pinned
by cases measured in Maya.
"""

import copy
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pythontk as ptk  # noqa: E402
from pythontk import ArticulationAnalysis, ArticulationModel, ArticulationPlan  # noqa: E402

from test_articulation import _arm, _sphere  # noqa: E402

ptk.TestSandbox.activate()

PARTS = ["BASE", "LEG_1", "LEG_2", "LEG_3", "LEG_4", "HEAD"]
CONTROL_SCALE = 2.5
SCALE = 2.0
TURN = 30.0  # degrees about Y
MOVE = (5.0, 1.0, -3.0)


def _world(turn=0.0):
    """``(world, inverse)``: the group-to-world matrix the rig group sits at
    (scale, then *turn* degrees about Y, then a move) and its inverse, built
    by hand. The magnifier's is unturned, as the production prop sits (its
    analysis reads the arm's plane off the world axes)."""
    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    rows = [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]  # row vectors
    world = [v * SCALE for v in rows[0]] + [0.0]
    world += [v * SCALE for v in rows[1]] + [0.0]
    world += [v * SCALE for v in rows[2]] + [0.0]
    world += list(MOVE) + [1.0]
    # inverse: (p - t) R^T / s
    rt = [[rows[j][i] / SCALE for j in range(3)] for i in range(3)]
    t = [-sum(MOVE[k] * rt[k][i] for k in range(3)) for i in range(3)]
    inverse = rt[0] + [0.0] + rt[1] + [0.0] + rt[2] + [0.0] + t + [1.0]
    return world, inverse


def _apply(p, m):
    return [
        p[0] * m[c] + p[1] * m[4 + c] + p[2] * m[8 + c] + m[12 + c] for c in range(3)
    ]


def _rotate(v, m):
    return [v[0] * m[c] + v[1] * m[4 + c] + v[2] * m[8 + c] for c in range(3)]


def _mul(a, b):
    return [
        sum(a[r * 4 + k] * b[k * 4 + c] for k in range(4))
        for r in range(4)
        for c in range(4)
    ]


def _close(test, a, b, tol=1e-9, msg=None):
    test.assertLess(max(abs(x - y) for x, y in zip(a, b)), tol, msg or (a, b))


def _same_turn(a, b, tol=1e-9):
    return (
        min(
            max(abs(x - y) for x, y in zip(a, b)), max(abs(x + y) for x, y in zip(a, b))
        )
        < tol
    )


def _magnifier(links=None):
    """``(plan, world parts, proposal plan)``: the magnifier planned in its
    rig group's space, as ``ArticulatedRig.create`` plans it -- one link per
    part unless *links* (part indices per link, root first) says otherwise."""
    world, inverse = _world()
    parts, _at = _arm()
    placed = [
        {
            "name": p["name"],
            "shells": [[_apply(q, world) for q in s] for s in p["shells"]],
        }
        for p in parts
    ]
    links = links or [[i] for i in range(len(placed))]
    facts = [
        {
            "name": placed[link[0]]["name"],
            "shells": [s for i in link for s in placed[i]["shells"]],
        }
        for link in links
    ]
    names = [[placed[i]["name"] for i in link] for link in links]
    proposal = ArticulationAnalysis.propose(facts, order=list(range(len(links))))
    planned = ArticulationPlan.from_proposal(proposal, names, CONTROL_SCALE)
    spec = {
        "version": ArticulationPlan.VERSION,
        "name": "arm",
        "links": [[{"uuid": f"uuid-{n}", "name": n} for n in link] for link in names],
        "joints": [
            ArticulationPlan.plan_joint(j, names[j["link"]][0], inverse)
            for j in planned["joints"]
        ],
    }
    ArticulationPlan.check(spec)
    points = {i: [q for s in facts[i]["shells"] for q in s] for i in range(len(links))}
    spec["end"] = ArticulationPlan.default_end(spec, inverse, lambda i: points[i])
    return spec, placed, planned


class TestFromProposal(unittest.TestCase):
    def test_the_magnifier_shape_proposes_its_five_joints(self):
        world, _inverse = _world()
        parts, at = _arm()
        placed = [
            {
                "name": p["name"],
                "shells": [[_apply(q, world) for q in s] for s in p["shells"]],
            }
            for p in parts
        ]
        proposal = ArticulationAnalysis.propose(placed)
        plan = ArticulationPlan.from_proposal(
            proposal, [[p["name"]] for p in placed], CONTROL_SCALE
        )
        self.assertEqual([link[0] for link in plan["links"]], PARTS)
        self.assertEqual(
            [j["type"] for j in plan["joints"]],
            ["universal", "hinge", "hinge", "slide", "ball"],
        )
        self.assertEqual([j["link"] for j in plan["joints"]], [1, 2, 3, 4, 5])
        self.assertLess(
            math.dist(plan["joints"][1]["position"], _apply(at["elbow"], world)),
            0.3 * SCALE,
        )
        by_part = {j["part"]: j for j in proposal["joints"]}
        for joint in plan["joints"]:
            source = by_part[proposal["order"][joint["link"]]]
            self.assertAlmostEqual(joint["size"], CONTROL_SCALE * source["radius"])
        lo, hi = plan["joints"][3]["limits"]["tx"]
        self.assertLess(lo, 0)
        self.assertAlmostEqual(hi / SCALE, 12 * 0.8, delta=0.5)

    def test_a_part_nothing_reaches_is_named_by_its_first_member(self):
        parts, _at = _arm()
        parts.append({"name": "STRAY", "shells": [_sphere((500, 500, 500), 1.0)]})
        proposal = ArticulationAnalysis.propose(parts)
        links = [[p["name"], p["name"] + "_twin"] for p in parts]
        plan = ArticulationPlan.from_proposal(proposal, links, CONTROL_SCALE)
        self.assertEqual(plan["unreached"], ["STRAY"])
        self.assertEqual(plan["links"][0], ["BASE", "BASE_twin"])

    def test_a_selection_order_is_taken_as_given(self):
        _spec, placed, planned = _magnifier()
        self.assertEqual([j["parent"] for j in planned["joints"]], [0, 1, 2, 3, 4])
        self.assertEqual([link[0] for link in planned["links"]], PARTS)


class TestPlanJoint(unittest.TestCase):
    def test_a_joint_moves_into_the_rig_groups_space(self):
        world, inverse = _world(TURN)
        joint = {
            "link": 1,
            "parent": 0,
            "type": "slide",
            "position": _apply((1.0, 2.0, 3.0), world),
            "aim": _rotate((0.0, 3.0, 0.0), world),
            "normal": _rotate((0.0, 0.0, 1.0), world),
            "limits": {"tx": [-2.0, 4.0], "rz": [-10.0, 10.0]},
            "weights": {"tx": 0.5, "rz": 2.0},
            "size": 6.0,
        }
        planned = ArticulationPlan.plan_joint(joint, "TUBE", inverse)
        self.assertEqual(planned["id"], "TUBE")
        _close(self, planned["position"], (1.0, 2.0, 3.0))
        _close(self, planned["aim"], (0.0, 1.0, 0.0))
        _close(self, planned["normal"], (0.0, 0.0, 1.0))
        # a slide's travel and the control's size by the prop's scale; the
        # channels a slide lacks are dropped
        _close(self, planned["limits"]["tx"], (-2.0 / SCALE, 4.0 / SCALE))
        self.assertEqual(list(planned["limits"]), ["tx"])
        self.assertEqual(planned["weights"], {"tx": 0.5})
        self.assertAlmostEqual(planned["size"], 6.0 / SCALE)
        self.assertEqual(planned["reason"], "as planned")
        # a rotation's limits are degrees, whatever the scale
        joint["type"] = "hinge"
        hinge = ArticulationPlan.plan_joint(joint, "TUBE", inverse)
        self.assertEqual(hinge["limits"], {"rz": [-10.0, 10.0]})

    def test_an_unknown_or_fixed_type_is_refused(self):
        _world_m, inverse = _world()
        for kind in ("fixed", "twist"):
            joint = {
                "link": 1,
                "parent": 0,
                "type": kind,
                "position": [0, 0, 0],
                "aim": [1, 0, 0],
                "normal": [0, 0, 1],
            }
            with self.assertRaises(ValueError, msg=kind):
                ArticulationPlan.plan_joint(joint, "X", inverse)


class TestCheck(unittest.TestCase):
    def test_the_joints_are_ordered_parent_first(self):
        spec, _placed, _planned = _magnifier()
        want = [j["id"] for j in spec["joints"]]
        shuffled = copy.deepcopy(spec)
        random.Random(3).shuffle(shuffled["joints"])
        ArticulationPlan.check(shuffled)
        self.assertEqual([j["id"] for j in shuffled["joints"]], want)

    def test_a_newcomer_repeating_a_joints_id_is_numbered(self):
        """A production assembly repeats part names: the joint listed first
        (already stamped) keeps its id -- the key its parked keys are filed
        under -- and the newcomer takes a fresh one."""
        spec, _placed, _planned = _magnifier()
        spec["joints"][-1]["id"] = "LEG_2"
        ArticulationPlan.check(spec)
        ids = [j["id"] for j in spec["joints"]]
        self.assertEqual(len(set(ids)), 5, ids)
        self.assertEqual(ids[1], "LEG_2")
        self.assertEqual(ids[-1], "LEG_2_5")

    def test_a_plan_the_build_cannot_keep_is_refused(self):
        spec, _placed, _planned = _magnifier()
        cases = {
            "A part is named in two links": lambda s: s["links"][2].append(
                dict(s["links"][1][0])
            ),
            "Link 2 has no single joint slot": lambda s: s["joints"][2].update(link=2),
            "Links \\[5\\] have no joint": lambda s: s["joints"].pop(),
            "do not join every link to the root": lambda s: s["joints"][0].update(
                parent=2
            ),
            "runs along its aim": lambda s: s["joints"][1].update(
                normal=list(s["joints"][1]["aim"])
            ),
            "has no aim": lambda s: s["joints"][1].update(aim=[0.0, 0.0, 0.0]),
        }
        for message, break_it in cases.items():
            broken = copy.deepcopy(spec)
            break_it(broken)
            with self.assertRaisesRegex(ValueError, message):
                ArticulationPlan.check(broken)

    def test_the_hosts_part_check_runs_once_each_part_is_named_once(self):
        """The host's check (a part gone from the scene, a driven part) is
        the plan's to call: after its own part check, before the joints'."""
        spec, _placed, _planned = _magnifier()
        seen = []

        def gone(plan):
            seen.append(len(plan["links"]))
            raise ValueError("Parts no longer in the scene: ['HEAD'].")

        with self.assertRaisesRegex(ValueError, "no longer in the scene.*HEAD"):
            ArticulationPlan.check(copy.deepcopy(spec), parts=gone)
        self.assertEqual(seen, [6])
        doubled = copy.deepcopy(spec)
        doubled["links"][2].append(dict(doubled["links"][1][0]))
        with self.assertRaisesRegex(ValueError, "named in two links"):
            ArticulationPlan.check(doubled, parts=gone)
        self.assertEqual(seen, [6], "the host check ran on a refused plan")


class TestRest(unittest.TestCase):
    def test_a_frame_runs_x_along_the_aim_with_the_normal_its_z(self):
        m = ArticulationPlan.frame((1.0, 2.0, 3.0), (0.0, 2.0, 0.0), (1.0, 1.0, 1.0))
        _close(self, m[0:3], (0.0, 1.0, 0.0))  # X: the aim
        _close(self, m[8:11], (1 / math.sqrt(2), 0.0, 1 / math.sqrt(2)))  # Z
        _close(self, m[4:7], (-1 / math.sqrt(2), 0.0, 1 / math.sqrt(2)))  # Y = Z x X
        _close(self, m[12:16], (1.0, 2.0, 3.0, 1.0))
        with self.assertRaisesRegex(ValueError, "has no aim"):
            ArticulationPlan.frame((0, 0, 0), (0, 0, 0), (0, 0, 1))
        with self.assertRaisesRegex(ValueError, "runs along its aim"):
            ArticulationPlan.frame((0, 0, 0), (0, 0, 2), (0, 0, -1))

    def test_the_record_poses_the_plans_frames_at_rest(self):
        """What a build sets on the joints and what the record ships are one
        computation: the model of the record, at rest, stands every joint on
        its link's rest frame."""
        spec, _placed, _planned = _magnifier()
        frames = ArticulationPlan.frames(spec)
        model = ArticulationModel(ArticulationPlan.record(spec))
        for joint, (p, q) in zip(spec["joints"], model.world(model.rest_state())):
            frame = frames[joint["link"]]
            _close(self, p, frame[12:15], 1e-9, joint["id"])
            rotation = ArticulationModel._qmatrix(q)  # column vectors
            for axis in range(3):
                _close(
                    self,
                    [rotation[r][axis] for r in range(3)],
                    frame[axis * 4 : axis * 4 + 3],
                    1e-9,
                    (joint["id"], axis),
                )

    def test_the_joint_orient_is_the_rest_rotation(self):
        spec, _placed, _planned = _magnifier()
        for pose in ArticulationPlan.rest(spec):
            q = ArticulationModel._euler_quat(
                "xyz", {"rx": pose["jo"][0], "ry": pose["jo"][1], "rz": pose["jo"][2]}
            )
            self.assertTrue(_same_turn(q, pose["q"]), (pose["jo"], pose["q"]))

    def test_a_slides_axes_are_its_rest_frame_in_its_parents_space(self):
        spec, _placed, _planned = _magnifier()
        frames = ArticulationPlan.frames(spec)
        slide = next(j for j in spec["joints"] if j["type"] == "slide")
        pose = ArticulationPlan.rest(spec)[spec["joints"].index(slide)]
        parent = frames[slide["parent"]]
        x_world = _rotate(pose["axes"][0], parent)
        _close(self, x_world, slide["aim"], 1e-9)
        self.assertEqual(
            [p["parent"] for p in ArticulationPlan.rest(spec)], [None, 0, 1, 2, 3]
        )

    def test_the_joint_orient_reads_as_maya_reads_it(self):
        """Measured in Maya 2025 (``MTransformationMatrix.rotation``, which is
        ``MEulerRotation.decompose``): the triple with its middle angle in
        [-90, 90], unless the one half a turn round each axis sums smaller --
        then that one, unwrapped; at gimbal lock z is 0. Its quaternion is
        always the first triple's."""
        measured = [
            (
                [
                    0.12624743664592053,
                    -0.3527725358335947,
                    0.9271532358255941,
                    -0.5223422045598632,
                    -0.8182085363931433,
                    -0.24019453013949532,
                    0.8433387255876945,
                    -0.4539673214394689,
                    -0.287564714436938,
                ],
                [-140.12896378403474, -67.99531736157218, -70.30913016443859],
                [
                    -0.7469974376589255,
                    0.2928774237620043,
                    -0.5925361561352783,
                    -0.0715440175972803,
                ],
            ),
            (
                [
                    -0.34077970154233084,
                    -0.6745817299590658,
                    -0.6548348529378646,
                    -0.4690056395984266,
                    0.7256478391616068,
                    -0.5034569728834449,
                    0.8148023717697387,
                    0.13555332207483914,
                    -0.5636686897751944,
                ],
                [41.77055548458766, 139.09287118330678, 63.19837935824421],
                [
                    -0.3525766402752939,
                    0.8108785714395113,
                    -0.11342747970127182,
                    0.4531002780412086,
                ],
            ),
        ]
        for rows, jo, q in measured:
            m = (
                rows[0:3]
                + [0.0]
                + rows[3:6]
                + [0.0]
                + rows[6:9]
                + [0.0, 1.0, 2.0, 3.0, 1.0]
            )
            t, got_jo, got_q = ArticulationPlan._decompose(m)
            _close(self, t, (1.0, 2.0, 3.0))
            _close(self, got_jo, jo, 1e-9)
            _close(self, got_q, q, 1e-9)
        # gimbal lock: (30, 90, -15) reads (45, 90, 0); (30, -90, -15) reads (15, -90, 0)
        for y, x in ((90.0, 45.0), (-90.0, 15.0)):
            q = ArticulationModel._euler_quat("xyz", {"rx": 30.0, "ry": y, "rz": -15.0})
            c = ArticulationModel._qmatrix(q)
            m = [c[0][0], c[1][0], c[2][0], 0.0, c[0][1], c[1][1], c[2][1], 0.0]
            m += [c[0][2], c[1][2], c[2][2], 0.0, 0.0, 0.0, 0.0, 1.0]
            _t, got_jo, _q = ArticulationPlan._decompose(m)
            _close(self, got_jo, (x, y, 0.0), 1e-6)


class TestRecord(unittest.TestCase):
    def test_the_record_is_the_model_the_runtimes_pose(self):
        spec, _placed, _planned = _magnifier()
        entry = ArticulationPlan.record(spec)
        self.assertEqual([j["parent"] for j in entry["joints"]], [None, 0, 1, 2, 3])
        self.assertEqual(
            [j["name"] for j in entry["joints"]], [f"arm_{p}_jnt" for p in PARTS[1:]]
        )
        self.assertEqual(sorted(g["node"] for g in entry["grab"]), sorted(PARTS[1:]))
        self.assertEqual(len(ArticulationModel(entry).channels), 2 + 1 + 1 + 1 + 3)
        res = ptk.ArticulationRecord.validate({"version": 1, "rigs": [entry]})
        self.assertEqual(res.errors, [])

    def test_the_grab_names_a_part_as_the_host_names_it_now(self):
        """A runtime binds a grab by the part's name: the record carries the
        name the part has now, not the one it was built with."""
        spec, _placed, _planned = _magnifier()
        renamed = {"uuid-HEAD": "LENS"}
        entry = ArticulationPlan.record(
            spec, lambda m: renamed.get(m["uuid"], m["name"])
        )
        grabbed = [g["node"] for g in entry["grab"]]
        self.assertIn("LENS", grabbed)
        self.assertNotIn("HEAD", grabbed)

    def test_riders_and_slots_follow_the_models_order(self):
        spec, _placed, _planned = _magnifier()
        riders = ArticulationPlan.riders(spec)
        self.assertNotIn("uuid-BASE", riders)  # the root link stays put
        self.assertEqual(riders["uuid-HEAD"], 4)
        model = ArticulationModel(ArticulationPlan.record(spec))
        slots = ArticulationPlan.slots(spec)
        self.assertEqual([c for _j, c in slots], [c for _i, c in model.channels])
        self.assertEqual(
            [spec["joints"][i]["id"] for i, _c in model.channels],
            [j for j, _c in slots],
        )


class TestEndControl(unittest.TestCase):
    def test_the_end_control_boxes_the_head_aligned_with_its_joint(self):
        spec, placed, _planned = _magnifier()
        self.assertEqual(spec["end"]["part"], "uuid-HEAD")
        index = ArticulationPlan.end_joint(spec)
        self.assertEqual(spec["joints"][index]["id"], "HEAD")
        head = spec["joints"][index]
        frame = ArticulationPlan.frames(spec)[head["link"]]
        end = spec["end"]["frame"]
        _close(self, end[0:12], frame[0:12], 1e-12)  # aligned with the joint
        # centred on the head's geometry in that frame
        _world_m, inverse = _world()
        to_frame = _mul(inverse, ArticulationPlan._inverse(frame))
        local = [_apply(q, to_frame) for s in placed[5]["shells"] for q in s]
        centre = [
            0.5 * (min(p[i] for p in local) + max(p[i] for p in local))
            for i in range(3)
        ]
        _close(self, end[12:15], _apply(centre, frame), 1e-9)

    def test_the_end_control_goes_where_it_is_asked_but_never_on_the_root(self):
        spec, placed, _planned = _magnifier()
        _world_m, inverse = _world()
        self.assertEqual(ArticulationPlan.link_of(spec, "uuid-LEG_3"), 3)
        self.assertIsNone(ArticulationPlan.link_of(spec, "uuid-NOPE"))
        points = lambda i: [q for s in placed[i]["shells"] for q in s]  # noqa: E731
        moved = ArticulationPlan.default_end(spec, inverse, points, link=3)
        self.assertEqual(moved["part"], "uuid-LEG_3")
        self.assertIsNone(ArticulationPlan.default_end(spec, inverse, points, link=0))
        spec["end"] = {"part": "uuid-BASE", "frame": list(spec["end"]["frame"])}
        self.assertIsNone(ArticulationPlan.end_joint(spec))
        del spec["end"]
        self.assertIsNone(ArticulationPlan.end_joint(spec))

    def test_the_end_link_is_the_end_of_the_longest_chain(self):
        spec = {
            "joints": [
                {"link": 1, "parent": 0},
                {"link": 2, "parent": 1},
                {"link": 3, "parent": 0},
                {"link": 4, "parent": 2},
                {"link": 5, "parent": 3},
            ]
        }
        self.assertEqual(ArticulationPlan.end_link(spec), 4)
        self.assertIsNone(ArticulationPlan.end_link({"joints": []}))

    def test_the_end_link_rides_the_end_control_as_it_did_at_rest(self):
        spec, _placed, _planned = _magnifier()
        index = ArticulationPlan.end_joint(spec)
        link = ArticulationPlan.frames(spec)[spec["joints"][index]["link"]]
        offset = ArticulationPlan.end_offset(spec, index)
        _close(self, _mul(offset, spec["end"]["frame"]), link, 1e-9)
        pivot, turn = ArticulationPlan.end_pivot_turn(spec, index)
        # aligned at rest: no turn; the pivot is the control's origin on the link
        self.assertTrue(_same_turn(turn, (0.0, 0.0, 0.0, 1.0)), turn)
        _close(
            self,
            _apply(pivot, link),
            spec["end"]["frame"][12:15],
            1e-9,
        )


class TestEdits(unittest.TestCase):
    def test_give_reaches_the_record(self):
        spec, _placed, _planned = _magnifier()
        ArticulationPlan.set_weight(spec, "LEG_1", 0.25)
        base = next(
            j
            for j in ArticulationPlan.joint_records(spec)
            if j["name"] == "arm_LEG_1_jnt"
        )
        self.assertEqual([c["weight"] for c in base["channels"]], [0.25, 0.25])
        joint = ArticulationPlan.set_weight(spec, "LEG_1", 2.0, channel="rz")
        self.assertEqual(joint["weights"], {"rx": 0.25, "rz": 2.0})
        self.assertEqual(ArticulationPlan.give_of(joint), 0.25)
        with self.assertRaises(ValueError):
            ArticulationPlan.set_weight(spec, "LEG_1", -1.0)
        with self.assertRaises(KeyError):
            ArticulationPlan.set_weight(spec, "LEG_1", 1.0, channel="ry")
        with self.assertRaises(KeyError):
            ArticulationPlan.set_weight(spec, "NOPE", 1.0)

    def test_limits_from_the_pose_bound_the_record(self):
        spec, _placed, _planned = _magnifier()
        lo, hi = ArticulationPlan.side_limits(spec, "LEG_2", "rz", "max", 72.0)
        self.assertEqual((lo, hi), (None, 72.0))
        ArticulationPlan.set_limits(spec, "LEG_2", "rz", lo, hi)
        lo, hi = ArticulationPlan.side_limits(spec, "LEG_2", "rz", "min", -35.0)
        joint = ArticulationPlan.set_limits(spec, "LEG_2", "rz", lo, hi)
        self.assertEqual(joint["limits"]["rz"], [-35.0, 72.0])
        elbow = next(
            j
            for j in ArticulationPlan.joint_records(spec)
            if j["name"] == "arm_LEG_2_jnt"
        )
        self.assertEqual(
            (elbow["channels"][0]["min"], elbow["channels"][0]["max"]), (-35.0, 72.0)
        )
        ArticulationPlan.set_limits(spec, "LEG_2", "rz", None, None)
        self.assertEqual(joint["limits"]["rz"], [None, None])
        with self.assertRaises(ValueError):
            ArticulationPlan.set_limits(spec, "LEG_2", "rz", 10.0, -10.0)
        with self.assertRaises(KeyError):
            ArticulationPlan.set_limits(spec, "LEG_2", "rx", -1.0, 1.0)
        with self.assertRaises(ValueError):
            ArticulationPlan.side_limits(spec, "LEG_2", "rz", "middle", 1.0)

    def test_folding_a_joint_back_rides_its_parts_on_the_link_above(self):
        spec, _placed, _planned = _magnifier()
        ArticulationPlan.remove_joint(spec, "LEG_4")
        self.assertEqual(
            [j["id"] for j in spec["joints"]], ["LEG_1", "LEG_2", "LEG_3", "HEAD"]
        )
        self.assertEqual([m["name"] for m in spec["links"][3]], ["LEG_3", "LEG_4"])
        head = spec["joints"][-1]
        self.assertEqual((head["link"], head["parent"]), (4, 3))
        ArticulationPlan.check(spec)
        with self.assertRaises(KeyError):
            ArticulationPlan.remove_joint(spec, "LEG_4")

    def test_a_retyped_joint_keeps_its_id_and_the_channels_both_types_share(self):
        spec, _placed, _planned = _magnifier()
        world, _inverse = _world()
        ArticulationPlan.set_limits(spec, "LEG_2", "rz", -20.0, 45.0)
        before = copy.deepcopy(ArticulationPlan.joint_entry(spec, "LEG_2"))
        ArticulationPlan.edit_joint(spec, "LEG_2", world, type="ball")
        after = ArticulationPlan.joint_entry(spec, "LEG_2")
        self.assertEqual(after["type"], "ball")
        self.assertEqual(after["limits"], {"rz": [-20.0, 45.0]})
        for key in ("position", "aim", "normal"):
            _close(self, after[key], before[key], 1e-9, key)
        self.assertAlmostEqual(after["size"], before["size"])
        ArticulationPlan.check(spec)
        with self.assertRaises(KeyError):
            ArticulationPlan.edit_joint(spec, "NOPE", world, type="ball")

    def test_a_slides_travel_survives_an_edit_in_world_units(self):
        spec, _placed, _planned = _magnifier()
        world, _inverse = _world()
        slide = ArticulationPlan.joint_entry(spec, "LEG_4")
        travel = list(slide["limits"]["tx"])
        moved = _apply(
            _apply(slide["position"], world),
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.5, 0, 0, 1],
        )
        ArticulationPlan.edit_joint(spec, "LEG_4", world, position=moved)
        slide = ArticulationPlan.joint_entry(spec, "LEG_4")
        _close(self, slide["limits"]["tx"], travel, 1e-9)
        _close(self, _apply(slide["position"], world), moved, 1e-9)

    def test_a_hinge_axis_along_its_link_is_refused(self):
        spec, _placed, _planned = _magnifier()
        world, _inverse = _world()
        aim = _rotate(ArticulationPlan.joint_entry(spec, "LEG_2")["aim"], world)
        ArticulationPlan.edit_joint(spec, "LEG_2", world, normal=aim)
        with self.assertRaisesRegex(ValueError, "runs along its aim"):
            ArticulationPlan.check(spec)

    def test_a_telescope_added_after_the_fact_carries_the_head(self):
        """The rods built as one link, then split: the inner rod gets a slide
        of its own, and the head -- nearer the inner rod -- now hangs off it."""
        spec, placed, _planned = _magnifier(links=[[0], [1], [2], [3, 4], [5]])
        self.assertEqual(len(spec["joints"]), 4)
        world, inverse = _world()
        host, keep, split = ArticulationPlan.split_link(spec, ["uuid-LEG_4"])
        self.assertEqual(
            (host, [m["name"] for m in keep], [m["name"] for m in split]),
            (3, ["LEG_3"], ["LEG_4"]),
        )
        shells = lambda i: [s for s in placed[i]["shells"]]  # noqa: E731
        proposal = ArticulationAnalysis.propose(
            [
                {"name": "LEG_3", "shells": shells(3)},
                {"name": "LEG_4", "shells": shells(4)},
            ],
            order=[0, 1],
        )["joints"][0]
        proposal.update(link=len(spec["links"]), parent=host)
        joint = ArticulationPlan.plan_joint(proposal, "LEG_4", inverse)
        self.assertEqual(joint["type"], "slide")
        points = lambda i: [q for s in placed[i]["shells"] for q in s]  # noqa: E731
        ArticulationPlan.insert_joint(
            spec, ["uuid-LEG_4"], joint, world, points(3), points(4)
        )
        self.assertEqual([m["name"] for m in spec["links"][3]], ["LEG_3"])
        self.assertEqual([m["name"] for m in spec["links"][5]], ["LEG_4"])
        head = ArticulationPlan.joint_entry(spec, "HEAD")
        self.assertEqual(head["parent"], 5)
        ArticulationPlan.check(spec)
        self.assertEqual(
            [j["id"] for j in spec["joints"]],
            ["LEG_1", "LEG_2", "LEG_3", "LEG_4", "HEAD"],
        )

    def test_a_split_must_leave_one_link_and_keep_some_of_it(self):
        spec, _placed, _planned = _magnifier(links=[[0], [1], [2], [3, 4], [5]])
        with self.assertRaisesRegex(ValueError, "not all in one link"):
            ArticulationPlan.split_link(spec, ["uuid-LEG_4", "uuid-HEAD"])
        with self.assertRaisesRegex(ValueError, "not all in one link"):
            ArticulationPlan.split_link(spec, [])
        with self.assertRaisesRegex(ValueError, "adds no joint"):
            ArticulationPlan.split_link(spec, ["uuid-HEAD"])

    def test_a_retyped_plan_joint_keeps_only_its_types_limits(self):
        joint = {"type": "slide", "limits": {"tx": [-1.0, 1.0]}}
        ArticulationPlan.retype(joint, "hinge")
        self.assertEqual(joint, {"type": "hinge", "limits": {}})
        joint = {"type": "hinge", "limits": {"rz": [-5.0, 5.0]}}
        ArticulationPlan.retype(joint, "universal")
        self.assertEqual(joint["limits"], {"rz": [-5.0, 5.0]})


class TestAdjust(unittest.TestCase):
    """The pivots put out for adjusting and read back."""

    def test_unmoved_pivots_change_nothing(self):
        spec, _placed, _planned = _magnifier()
        pivots = ArticulationPlan.pivots(spec)
        self.assertEqual(
            set(pivots), {j["id"] for j in spec["joints"]} | {ArticulationPlan.END_KEY}
        )
        self.assertIsNone(ArticulationPlan.adjusted(spec, pivots))
        self.assertIsNone(ArticulationPlan.adjusted(spec, {}))
        nudged = dict(pivots)
        nudged["LEG_2"] = list(pivots["LEG_2"])
        nudged["LEG_2"][12] += 0.5 * ArticulationPlan.ADJUST_EPS
        self.assertIsNone(ArticulationPlan.adjusted(spec, nudged))

    def test_a_moved_pivot_moves_its_joint_and_leaves_its_childs_where_it_was(self):
        spec, _placed, _planned = _magnifier()
        pivots = ArticulationPlan.pivots(spec)
        moved = list(pivots["LEG_2"])
        # (2, 1, 0) in the handle's own axes, as cmds.move(objectSpace=True)
        for i in range(3):
            moved[12 + i] += 2.0 * moved[i] + 1.0 * moved[4 + i]
        out = ArticulationPlan.adjusted(spec, dict(pivots, LEG_2=moved))
        self.assertIsNotNone(out)
        self.assertIsNot(out, spec)
        before = ArticulationPlan.joint_entry(spec, "LEG_2")
        after = ArticulationPlan.joint_entry(out, "LEG_2")
        shift = [
            2.0 * a + 1.0 * b
            for a, b in zip(
                before["aim"],
                ArticulationPlan.frame(
                    before["position"], before["aim"], before["normal"]
                )[4:7],
            )
        ]
        _close(
            self,
            after["position"],
            [p + d for p, d in zip(before["position"], shift)],
            1e-9,
        )
        _close(self, after["aim"], before["aim"], 1e-9)
        for key in ("position", "aim", "normal"):
            _close(
                self,
                ArticulationPlan.joint_entry(out, "LEG_3")[key],
                ArticulationPlan.joint_entry(spec, "LEG_3")[key],
                1e-9,
                key,
            )
        # an unmoved pivot reads back as it was (the copy carries its
        # rounding, as a handle's read-back does)
        _close(self, out["end"]["frame"], spec["end"]["frame"], 1e-9)

    def test_a_turned_end_handle_reorients_the_end_control(self):
        spec, _placed, _planned = _magnifier()
        pivots = ArticulationPlan.pivots(spec)
        handle = list(pivots[ArticulationPlan.END_KEY])
        # 90 degrees about the handle's own Z: X -> Y, Y -> -X
        x, y = handle[0:3], handle[4:7]
        handle[0:3], handle[4:7] = y, [-v for v in x]
        out = ArticulationPlan.adjusted(
            spec, dict(pivots, **{ArticulationPlan.END_KEY: handle})
        )
        index = ArticulationPlan.end_joint(spec)
        link = ArticulationPlan.frames(spec)[spec["joints"][index]["link"]]
        _close(self, out["end"]["frame"], _mul(handle, link), 1e-12)
        for a, b in zip(out["joints"], spec["joints"]):
            self.assertEqual(a["id"], b["id"])
            for key in ("position", "aim", "normal"):
                _close(self, a[key], b[key], 1e-9, (a["id"], key))


class TestText(unittest.TestCase):
    """The panel's Limits and Give cells."""

    def test_limits_typed_as_text_parse(self):
        parse = ArticulationPlan.parse_limits
        self.assertEqual(parse("rz -30..95", ("rz",)), {"rz": (-30.0, 95.0)})
        self.assertEqual(parse("-..12.5", ("tx",)), {"tx": (None, 12.5)})
        self.assertEqual(
            parse("rx -10..10, rz ..45", ("rx", "rz")),
            {"rx": (-10.0, 10.0), "rz": (None, 45.0)},
        )
        self.assertEqual(
            parse("free", ("rx", "rz")), {"rx": (None, None), "rz": (None, None)}
        )
        self.assertEqual(parse("", ("rx",)), {"rx": (None, None)})
        self.assertEqual(parse("rz: 1e1..2E1", ("rz",)), {"rz": (10.0, 20.0)})
        for bad in ("rz 5", "ry -1..1", "-1..1", "rz 9..3"):
            with self.assertRaises(ValueError, msg=bad):
                parse(bad, ("rx", "rz"))

    def test_limits_text_reads_back_as_typed(self):
        text = ArticulationPlan.limits_text
        self.assertEqual(text({}), "free")
        self.assertEqual(text(None), "free")
        shown = text({"rx": [-10.0, None], "rz": [0.5, 95.0]})
        self.assertEqual(shown, "rx -10..-, rz 0.5..95")
        self.assertEqual(
            ArticulationPlan.parse_limits(shown, ("rx", "rz")),
            {"rx": (-10.0, None), "rz": (0.5, 95.0)},
        )

    def test_a_limit_survives_the_cell_it_is_shown_in(self):
        """Bug: the Limits cell showed three significant digits, and an edit
        re-applies every channel the cell reads, so a limit read off a pose
        (-37.84623) came back -37.8 and a long slide's 1234.5 came back 1230.
        Added: 2026-10-19"""
        limits = {"rx": [-37.84623, 1234.5], "tz": [0.012345, None]}
        back = ArticulationPlan.parse_limits(
            ArticulationPlan.limits_text(limits), ("rx", "tz")
        )
        for channel, bounds in limits.items():
            for want, got in zip(bounds, back[channel]):
                if want is None:
                    self.assertIsNone(got)
                else:
                    self.assertAlmostEqual(
                        got, want, delta=1e-4 * max(1.0, abs(want)), msg=channel
                    )

    def test_a_joints_give_is_its_stiffest_channel(self):
        give = ArticulationPlan.give_of
        self.assertEqual(give({"type": "ball"}), 1.0)
        self.assertEqual(give({"type": "ball", "weights": {"rx": 0.5, "ry": 2.0}}), 0.5)
        self.assertEqual(give({"type": "hinge", "weights": {"rx": 0.1}}), 1.0)


if __name__ == "__main__":
    unittest.main()
