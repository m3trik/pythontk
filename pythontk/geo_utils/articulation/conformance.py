# !/usr/bin/python
# coding=utf-8
"""The golden cases every port of :class:`ArticulationModel` is held to.

The model runs in three places besides Python -- unitytk's C#
(``ArticulatedRigController.cs``), the WebXR preview's JavaScript
(``articulated_rig.js``) and whatever production app vendors that script --
and a port is only as good as the cases that pin it. These are generated
from the Python reference at test time, so they can never drift from it: a
port's test asks :meth:`ArticulationConformance.cases` for the document,
runs every case through its own model and compares within
:attr:`ArticulationConformance.TOLERANCE`.

Every case is one rig (:meth:`rigs` -- a magnifier-shaped arm with every
joint type, a branching tree, and a ball in each of the six rotate orders)
at one random state, and what the model says of it: the local and rig-space
poses, the state read back from those locals, a grab solved from that state
(with and without a hand's rotation) and the unit scale measured off a
rescaled copy of the pose.
"""

from __future__ import annotations

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
        "scale": 1.0e-12,
    }

    @staticmethod
    def rigs() -> Dict[str, Dict[str, Any]]:
        """The rigs the cases pose, by name."""

        def ch(channel, lo=None, hi=None, weight=1.0):
            return {"channel": channel, "min": lo, "max": hi, "weight": weight}

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
            "joint", "local", "target", "rotation", "expect"}, ...], "scale":
            {"locals", "expect"}}``.
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
                k = rng.uniform(0.005, 3.0)
                scaled = [[[v * k for v in t], list(q)] for t, q in pose]
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
                        "scale": {"locals": scaled, "expect": model.scale_of(scaled)},
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
