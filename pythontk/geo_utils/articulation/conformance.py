# !/usr/bin/python
# coding=utf-8
"""The golden cases every port of :class:`ArticulationModel` is held to.

The model runs in three places besides Python -- unitytk's C#
(``Features/ArticulatedRig/Model/ArticulationModel.cs``), the WebXR preview's JavaScript
(``articulated_rig.js``) and whatever production app vendors that script --
and a port is only as good as the cases that pin it. These are generated
from the Python reference at test time, so they can never drift from it: a
port's test asks for the document (``Conformance.cases("articulation")``,
the registry every ported model's cases are listed in, or
:meth:`ArticulationConformance.cases` directly), runs every case through its
own model and compares within :attr:`ArticulationConformance.TOLERANCE`.

Every case is one rig (:meth:`rigs` -- a magnifier-shaped arm with every
joint type, a branching tree, and a ball in each of the six rotate orders,
every joint on its pivot; then, appended so every older case keeps its index,
the anchored rigs of articulation v2: an arm whose hinges, slides and ball
turn about anchors, a flat model -- every part's rest the root's, its hinges
in its anchors -- and a branching tree under anchors; then that arm with its
anchors on the identity's edge: a pure offset, a pure turn, a hair off the
identity in each, the identity as -q and unnormalised, and its own anchors
unnormalised) at one random state, and
what the model says of it: the local and rig-space poses, the state read back
from those locals and a grab solved from that state (with and without a
hand's rotation). A runtime whose parts all hang under the root (a flat
import) places each at its ``world`` pose: the pose the joint chain gives,
whatever node a part hangs under.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List

from pythontk.geo_utils.articulation.model import ArticulationModel


class ArticulationConformance:
    """Golden cases for the ports of :class:`ArticulationModel`."""

    #: Per quantity, the largest difference a port may show: lengths in the
    #: rig's units (the rigs are ~10-30 units long), angles in degrees,
    #: quaternion components (sign-insensitive: q and -q are one rotation).
    #: The closed forms agree to rounding (measured 1e-15 in the browser).
    #: The SOLVE is iterative, and it branches -- the early exit at the
    #: solver's own tolerance, a channel meeting its limit -- where the last
    #: bit of another runtime's sin/atan2 can land one iteration apart; each
    #: iteration there moves a channel by far less than a thousandth of a
    #: degree (measured 3.8e-5 in the browser).
    TOLERANCE: Dict[str, float] = {
        "pose": 1.0e-9,
        "world": 1.0e-9,
        "read": 1.0e-7,
        "solve": 1.0e-3,
    }

    @staticmethod
    def rigs() -> Dict[str, Dict[str, Any]]:
        """The rigs the cases pose, by name."""

        def ch(channel, lo=None, hi=None, weight=1.0):
            return {"channel": channel, "min": lo, "max": hi, "weight": weight}

        def unit(*q):
            n = math.sqrt(sum(v * v for v in q))
            return [v / n for v in q]

        def anchor(t, q=(0.0, 0.0, 0.0, 1.0)):
            return {"t": [float(v) for v in t], "q": unit(*q)}

        half = math.sqrt(0.5)
        arm = {
            "name": "arm",
            "joints": [
                {
                    "name": "base",
                    "parent": None,
                    "t": [0.0, 2.0, 0.0],
                    "q": [0.0, 0.0, half, half],
                    "rotate_order": "zyx",
                    "channels": [ch("rx", -170, 170), ch("rz", -60, 80)],
                },
                {
                    "name": "elbow",
                    "parent": 0,
                    "t": [10.0, 0.0, 0.0],
                    "q": [0.0, 0.0, -0.3826834323650898, 0.9238795325112867],
                    "rotate_order": "xyz",
                    "channels": [ch("rz", -150, 150)],
                },
                {
                    "name": "collar",
                    "parent": 1,
                    "t": [9.0, 0.0, 0.0],
                    "q": [0.0, 0.0, -0.13052619222005157, 0.9914448613738104],
                    "rotate_order": "xyz",
                    "channels": [ch("rz", -90, 90)],
                },
                {
                    "name": "extend",
                    "parent": 2,
                    "t": [8.0, 0.0, 0.0],
                    "q": [0.0, 0.0, 0.0, 1.0],
                    "rotate_order": "xyz",
                    "channels": [ch("tx", -2.5, 4.0, 0.5)],
                },
                {
                    "name": "head",
                    "parent": 3,
                    "t": [3.0, 0.0, 0.0],
                    "q": [0.0, 0.0, 0.0, 1.0],
                    "rotate_order": "xyz",
                    "channels": [
                        ch("rx", -45, 45),
                        ch("ry", -70, 70),
                        ch("rz", -70, 70),
                    ],
                },
            ],
            "grab": [{"node": "head_geo", "joint": 4}],
        }
        tree = {
            "name": "tree",
            "joints": [
                {
                    "name": "hub",
                    "parent": None,
                    "t": [1.0, 0.0, -1.0],
                    "q": [0.1, 0.2, 0.3, 0.9273618495495703],
                    "rotate_order": "yxz",
                    "channels": [ch("ry")],
                },
                {
                    "name": "left",
                    "parent": 0,
                    "t": [0.0, 5.0, 0.0],
                    "q": [0.0, 0.0, 0.0, 1.0],
                    "rotate_order": "xzy",
                    "channels": [ch("rx", -30, 30), ch("rz", -120, 10)],
                },
                {
                    "name": "right",
                    "parent": 0,
                    "t": [0.0, -5.0, 2.0],
                    "q": [0.5, 0.5, 0.5, 0.5],
                    "rotate_order": "zxy",
                    "channels": [ch("tx", -1, 1), ch("ty", 0, 3), ch("tz", -2, 0)],
                },
                {
                    "name": "tip",
                    "parent": 2,
                    "t": [4.0, 0.0, 0.0],
                    "q": [0.0, half, 0.0, half],
                    "rotate_order": "yzx",
                    "channels": [ch("rx"), ch("ry", -80, 80, 2.0), ch("rz")],
                },
            ],
            "grab": [],
        }
        rigs = {"arm": arm, "tree": tree}
        for order in ("xyz", "yzx", "zxy", "xzy", "yxz", "zyx"):
            rigs[f"ball_{order}"] = {
                "name": f"ball_{order}",
                "joints": [
                    {
                        "name": "post",
                        "parent": None,
                        "t": [0.0, 0.0, 0.0],
                        "q": [0.0, 0.0, 0.0, 1.0],
                        "rotate_order": "xyz",
                        "channels": [ch("rz", -90, 90)],
                    },
                    {
                        "name": "ball",
                        "parent": 0,
                        "t": [6.0, 0.0, 0.0],
                        "q": [0.0, 0.0, 0.0, 1.0],
                        "rotate_order": order,
                        "channels": [ch("rx"), ch("ry", -80, 80), ch("rz")],
                    },
                ],
                "grab": [],
            }
        # Version 1's rigs: every joint on its pivot (the identity anchor).
        for rig in rigs.values():
            for joint in rig["joints"]:
                joint["anchor"] = anchor((0.0, 0.0, 0.0))
        # Articulation v2's, appended: hinges, slides and a ball turning about
        # anchors off their pivots, a flat model and a branch under anchors.
        rigs["anchored"] = {
            "name": "anchored",
            "joints": [
                {
                    "name": "base",
                    "parent": None,
                    "t": [0.0, 1.0, 0.0],
                    "q": unit(0.0, 0.0, 0.2, 0.98),
                    "anchor": anchor((0.5, 0.0, 0.2), (0.0, 0.3, 0.0, 0.95)),
                    "rotate_order": "zyx",
                    "channels": [ch("ry", -170, 170), ch("rz", -60, 80)],
                },
                {
                    "name": "elbow",
                    "parent": 0,
                    "t": [8.0, 0.0, 0.0],
                    "q": unit(0.0, 0.0, -0.38, 0.92),
                    "anchor": anchor((0.0, 0.0, 1.0), (half, 0.0, 0.0, half)),
                    "rotate_order": "xyz",
                    "channels": [ch("rz", -150, 150)],
                },
                {
                    "name": "extend",
                    "parent": 1,
                    "t": [7.0, 0.0, 0.0],
                    "q": [0.0, 0.0, 0.0, 1.0],
                    "anchor": anchor((1.0, 0.5, 0.0), (0.0, 0.0, 0.38, 0.92)),
                    "rotate_order": "xyz",
                    "channels": [ch("tx", -2.0, 3.0, 0.5), ch("ty", -1.0, 1.0)],
                },
                {
                    "name": "head",
                    "parent": 2,
                    "t": [3.0, 0.0, 0.0],
                    "q": [0.0, 0.0, 0.0, 1.0],
                    "anchor": anchor((0.5, 0.25, -0.25), (0.1, 0.2, -0.3, 0.9)),
                    "rotate_order": "yzx",
                    "channels": [
                        ch("rx", -45, 45),
                        ch("ry", -70, 70),
                        ch("rz", -70, 70),
                    ],
                },
            ],
            "grab": [{"node": "head_geo", "joint": 3}],
        }
        rest = [0.0, 0.0, 0.0]
        still = [0.0, 0.0, 0.0, 1.0]
        rigs["flat"] = {
            "name": "flat",
            "joints": [
                {
                    "name": "hub",
                    "parent": None,
                    "t": list(rest),
                    "q": list(still),
                    "anchor": anchor((0.0, 1.0, 0.0), (half, 0.0, 0.0, half)),
                    "rotate_order": "xyz",
                    "channels": [ch("ry")],
                },
                {
                    "name": "upper",
                    "parent": 0,
                    "t": list(rest),
                    "q": list(still),
                    "anchor": anchor((2.0, 1.0, 0.0), (half, 0.0, 0.0, half)),
                    "rotate_order": "xyz",
                    "channels": [ch("rz", -100, 100)],
                },
                {
                    "name": "lower",
                    "parent": 1,
                    "t": list(rest),
                    "q": list(still),
                    "anchor": anchor((6.0, 1.0, 0.0), (half, 0.0, 0.0, half)),
                    "rotate_order": "xyz",
                    "channels": [ch("rz", -120, 30)],
                },
                {
                    "name": "slider",
                    "parent": 2,
                    "t": list(rest),
                    "q": list(still),
                    "anchor": anchor((9.0, 1.0, 0.0), (0.0, 0.26, 0.0, 0.97)),
                    "rotate_order": "xyz",
                    "channels": [ch("tz", -1.0, 2.0)],
                },
            ],
            "grab": [{"node": "slider_geo", "joint": 3}],
        }
        rigs["anchored_tree"] = {
            "name": "anchored_tree",
            "joints": [
                {
                    "name": "hub",
                    "parent": None,
                    "t": [1.0, 0.0, -1.0],
                    "q": unit(0.1, 0.2, 0.3, 0.9),
                    "anchor": anchor((0.3, 0.0, 0.0), (0.0, 0.38, 0.0, 0.92)),
                    "rotate_order": "xzy",
                    "channels": [ch("rx")],
                },
                {
                    "name": "left",
                    "parent": 0,
                    "t": [0.0, 5.0, 0.0],
                    "q": [0.0, 0.0, 0.0, 1.0],
                    "anchor": anchor((0.0, 1.0, 0.0), (half, 0.0, 0.0, half)),
                    "rotate_order": "yxz",
                    "channels": [ch("ry", -30, 30), ch("rz", -90, 40)],
                },
                {
                    "name": "right",
                    "parent": 0,
                    "t": [0.0, -5.0, 2.0],
                    "q": [0.5, 0.5, 0.5, 0.5],
                    "anchor": anchor((1.0, 0.0, 0.0), (0.0, 0.0, half, half)),
                    "rotate_order": "zyx",
                    "channels": [ch("tx", -1, 1), ch("ry", -60, 60)],
                },
                {
                    "name": "tip",
                    "parent": 2,
                    "t": [4.0, 0.0, 0.0],
                    "q": [0.0, half, 0.0, half],
                    "anchor": anchor((0.5, -0.5, 0.0), (0.2, 0.1, 0.0, 0.97)),
                    "rotate_order": "xyz",
                    "channels": [ch("rx"), ch("ry", -80, 80, 2.0), ch("rz")],
                },
            ],
            "grab": [],
        }
        # Appended again: the anchored arm with every anchor on the edge of the
        # identity -- a pure offset (its point off the pivot, its axes the
        # rest's: an engine's default joint, its anchor at the part's pivot),
        # a pure turn, each a hair off the identity, and the identity spelled
        # two other ways, -q and an unnormalised q, which every port reads as
        # the identity (ArticulationModel.is_identity); and the anchors as
        # stated, their q unnormalised, which every port normalises.
        edges = {
            "offset_only": lambda a: {"t": list(a["t"]), "q": list(still)},
            "turn_only": lambda a: {"t": list(rest), "q": list(a["q"])},
            "hair_off_t": lambda a: {"t": [1.0e-9, 0.0, 0.0], "q": list(still)},
            "hair_off_q": lambda a: {"t": list(rest), "q": [1.0e-9, 0.0, 0.0, 1.0]},
            "negated_identity": lambda a: {"t": list(rest), "q": [0.0, 0.0, 0.0, -1.0]},
            "unnormalised_identity": lambda a: {
                "t": list(rest),
                "q": [0.0, 0.0, 0.0, 2.0],
            },
            "unnormalised_anchor": lambda a: {
                "t": list(a["t"]),
                "q": [2.0 * v for v in a["q"]],
            },
        }
        for name, edge in edges.items():
            rig = copy.deepcopy(rigs["anchored"])
            rig["name"] = name
            for joint in rig["joints"]:
                joint["anchor"] = edge(joint["anchor"])
            rigs[name] = rig
        return rigs

    @classmethod
    def cases(cls, seed: int = 0, per_rig: int = 4) -> Dict[str, Any]:
        """The conformance document: ``{"tolerance", "rigs", "cases"}``.

        Parameters:
            seed: The random stream's seed -- the same seed, the same cases.
            per_rig: Cases per rig.

        Returns:
            Plain JSON-able values. Each case: ``{"rig", "state", "pose",
            "world", "read": {"locals", "hint", "expect"}, "solve": [{"from",
            "joint", "local", "target", "rotation", "expect"}, ...]}``.
        """
        rng = random.Random(seed)
        rigs = cls.rigs()
        cases: List[Dict[str, Any]] = []
        for name, rig in rigs.items():
            model = ArticulationModel(rig)
            for _ in range(per_rig):
                state = cls._random_state(model, rng)
                pose = model.pose(state)
                world = model.world(state)
                # A wound hint: reading must unwrap toward it, whole turns off.
                hint = [
                    v
                    + (
                        360.0 * rng.choice((-1, 1))
                        if model.channels[s][1][0] == "r"
                        else 0.0
                    )
                    for s, v in enumerate(state)
                ]
                solves = []
                start = (
                    model.rest_state()
                    if rng.random() < 0.5
                    else cls._random_state(model, rng)
                )
                for joint in (len(model.joints) - 1, rng.randrange(len(model.joints))):
                    local = [rng.uniform(-1.5, 1.5) for _ in range(3)]
                    target = list(model.point(state, joint, local))
                    for rotation in (None, list(world[joint][1])):
                        solves.append(
                            {
                                "from": start,
                                "joint": joint,
                                "local": local,
                                "target": target,
                                "rotation": rotation,
                                "expect": model.solve(
                                    start, joint, local, target, rotation
                                ),
                            }
                        )
                # The draw the retired unit-scale case took (a reader reads
                # metres now), kept so every later case is the one it was.
                rng.uniform(0.005, 3.0)
                cases.append(
                    {
                        "rig": name,
                        "state": state,
                        "pose": [[list(t), list(q)] for t, q in pose],
                        "world": [[list(p), list(q)] for p, q in world],
                        "read": {
                            "locals": [[list(t), list(q)] for t, q in pose],
                            "hint": hint,
                            "expect": model.read(pose, hint),
                        },
                        "solve": solves,
                    }
                )
        return {"tolerance": dict(cls.TOLERANCE), "rigs": rigs, "cases": cases}

    @staticmethod
    def _random_state(model: ArticulationModel, rng: random.Random) -> List[float]:
        state = []
        for slot, (_joint, channel) in enumerate(model.channels):
            lo, hi = model.limits(slot)
            span = 170.0 if channel[0] == "r" else 3.0
            lo = -span if lo is None else lo
            hi = span if hi is None else hi
            state.append(rng.uniform(lo, hi))
        return state
