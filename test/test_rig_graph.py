#!/usr/bin/python
# coding=utf-8
"""Tests for the rig-graph engine: the document, capability negotiation, the planner.

The planner is a pure function, so everything here runs with no DCC at all —
which is the property the pre-flight depends on.
"""

import unittest
from contextlib import contextmanager

from pythontk import (
    RigCapability,
    RigGraph,
    RigPlanner,
    RigPlanRefused,
)
from pythontk.core_utils.engines.rig_graph.rig_model import SCHEMA_VERSION


def _graph(records, nodes=None, policy=None):
    """A minimal well-formed graph around *records*.

    Nodes are synthesised from whatever the records reference, so a test only
    has to state the relationship it is about.
    """
    referenced = []
    for record in records:
        for node_id in list(_ids(record, "target")) + list(_ids(record, "source")):
            if node_id not in referenced:
                referenced.append(node_id)
    data = {
        "version": SCHEMA_VERSION,
        "source": {"app": "test", "linear_unit": "cm"},
        "nodes": [{"id": n} for n in (nodes if nodes is not None else referenced)],
        "records": records,
    }
    if policy is not None:
        data["policy"] = policy
    return RigGraph.from_dict(data)


def _ids(record, which):
    from pythontk.core_utils.engines.rig_graph.rig_model import RigRecord

    parsed = RigRecord.from_dict(record)
    return parsed.target_ids() if which == "target" else parsed.source_ids()


def _capability(ops, shapes=("transform", "channel"), target="test"):
    return RigCapability.from_dict(
        {"target": target, "shapes": list(shapes), "ops": ops}
    )


BLEND_OK = {
    "transform/blend": {
        "fidelity": "exact",
        "channels": ["translate", "rotate"],
        "roles": ["space"],
        "params": {"maintain_offset": True},
        "plugs": ["sources.weight"],
    }
}


def _blend(record_id="r1", target="/a", sources=("/b",), **over):
    record = {
        "id": record_id,
        "shape": "transform",
        "op": "blend",
        "target": {"id": target, "channels": ["translate"]},
        "sources": [{"id": s, "role": "space", "weight": 1.0} for s in sources],
        "params": {"maintain_offset": True},
    }
    record.update(over)
    return record


class TestPlugsAndPaths(unittest.TestCase):
    """Identity is a prim path, so the grammar has to survive real paths."""

    def test_channel_splits_at_the_first_dot(self):
        # A sanitised prim name cannot contain '.', so the first one ends the
        # path -- that is WHY ids are paths, and it must hold for nested channels.
        self.assertEqual(
            RigGraph.split_plug("/rig/ctrls/ctrl_L.translate.x"),
            ("/rig/ctrls/ctrl_L", "translate.x"),
        )
        self.assertEqual(
            RigGraph.split_plug("/rig/ctrl_L.stretch"), ("/rig/ctrl_L", "stretch")
        )

    def test_a_bare_node_reference_has_no_channel(self):
        self.assertEqual(RigGraph.split_plug("/rig/ctrl_L"), ("/rig/ctrl_L", ""))

    def test_parent_comes_from_the_path(self):
        self.assertEqual(RigGraph.parent_of("/rig/ctrls/ctrl_L"), "/rig/ctrls")
        self.assertIsNone(RigGraph.parent_of("/rig"))


class TestExpressionGrammar(unittest.TestCase):
    """The restricted grammar is the line that stops this becoming a language."""

    def test_arithmetic_comparison_and_whitelisted_calls_pass(self):
        self.assertEqual(
            RigGraph.validate_expression("1.0 + a * clamp(b / 10.0, 0, 1)", ["a", "b"]),
            [],
        )
        self.assertEqual(
            RigGraph.validate_expression("select(a > b, a, b)", ["a", "b"]), []
        )

    def test_an_unknown_variable_is_refused(self):
        self.assertTrue(RigGraph.validate_expression("a + zz", ["a"]))

    def test_attribute_access_and_subscripting_cannot_appear(self):
        # The escape a nulled-__builtins__ evaluator would still allow.
        for hostile in (
            "(1).__class__.__base__.__subclasses__()",
            "a[0]",
            "[x for x in a]",
            "lambda: 1",
        ):
            self.assertTrue(
                RigGraph.validate_expression(hostile, ["a", "x"]),
                f"{hostile} was allowed",
            )

    def test_a_call_outside_the_whitelist_is_refused(self):
        self.assertTrue(RigGraph.validate_expression("open('f')", []))
        self.assertTrue(
            RigGraph.validate_expression("exp(a)", ["a"])
        )  # math, but not listed


class TestValidation(unittest.TestCase):
    """The schema validates its five shapes and nothing else — ops stay open."""

    def test_a_well_formed_graph_validates(self):
        self.assertEqual(_graph([_blend()]).validate(), [])

    def test_an_unknown_op_is_NOT_a_validation_error(self):
        # Openness is the point: an unknown op is a planner outcome
        # (`no_builder`), never a malformed document.
        self.assertEqual(_graph([_blend(op="studio:rivet")]).validate(), [])

    def test_an_unknown_shape_is_an_error(self):
        errors = _graph([_blend(shape="wobble")]).validate()
        self.assertTrue(any("unknown shape" in e for e in errors))

    def test_duplicate_record_ids_are_caught(self):
        errors = _graph([_blend("r1"), _blend("r1", target="/c")]).validate()
        self.assertTrue(any("duplicate record id" in e for e in errors))

    def test_a_reference_to_an_unlisted_node_is_caught(self):
        graph = _graph([_blend()], nodes=["/a"])  # '/b' is referenced, not listed
        self.assertTrue(any("unknown node" in e for e in graph.validate()))

    def test_an_expr_record_is_validated_against_its_own_roles(self):
        record = {
            "id": "r1",
            "shape": "channel",
            "op": "expr",
            "target": "/a.scale.y",
            "sources": [{"plug": "/b.stretch", "role": "a"}],
            "params": {"expr": "a * 2 + q"},  # 'q' is not a role
        }
        self.assertTrue(
            any("unknown variable" in e for e in _graph([record]).validate())
        )

    def test_an_unknown_schema_version_is_refused_not_guessed(self):
        graph = RigGraph.from_dict({"version": 99, "nodes": [], "records": []})
        self.assertTrue(any("unknown schema version" in e for e in graph.validate()))

    def test_round_trip_through_plain_values_is_stable(self):
        graph = _graph([_blend()])
        self.assertEqual(RigGraph.from_dict(graph.to_dict()).to_dict(), graph.to_dict())


class TestPolicyResolution(unittest.TestCase):
    """A policy field must distinguish UNSET from 'set to the default'.

    Filling concrete defaults at parse time made a record that says nothing
    override the graph with a default it never asked for, so the graph-level
    default applied to nothing at all.
    """

    def test_the_graphs_default_reaches_a_record_that_says_nothing(self):
        graph = _graph([_blend()], policy={"fallback": "drop"})
        self.assertEqual(graph.effective_policy(graph.records[0]).fallback, "drop")

    def test_a_record_still_overrides_the_graph(self):
        graph = _graph(
            [_blend(policy={"fallback": "fail"})], policy={"fallback": "drop"}
        )
        self.assertEqual(graph.effective_policy(graph.records[0]).fallback, "fail")

    def test_with_neither_stated_the_schemas_default_applies(self):
        graph = _graph([_blend()])
        policy = graph.effective_policy(graph.records[0])
        self.assertEqual((policy.fallback, policy.enabled), ("bake", True))

    def test_a_graph_wide_disable_reaches_its_records(self):
        # `enabled` has the same trap as `fallback`: False is a real value and
        # True was the parse-time default, so this direction had to be proven.
        graph = _graph([_blend()], policy={"enabled": False})
        self.assertFalse(graph.effective_policy(graph.records[0]).enabled)
        self.assertEqual(RigPlanner.plan(graph, _capability({})).report, [])

    def test_an_unset_policy_serialises_as_absent_not_as_a_default(self):
        # Round-tripping must not turn "nobody said" into "someone said bake".
        graph = _graph([_blend()], policy={"fallback": "drop"})
        again = RigGraph.from_dict(graph.to_dict())
        self.assertEqual(again.records[0].policy.to_dict(), {})
        self.assertEqual(again.effective_policy(again.records[0]).fallback, "drop")


class TestChannelShapeExemption(unittest.TestCase):
    """A `channel` op's roles are its expression's OWN variable names, and its
    channel is whatever plug it was pointed at -- neither is a fixed vocabulary
    the target declares, so neither may be validated against one."""

    EXPR_CAP = {
        "channel/expr": {"fidelity": "exact", "params": {"expr": True}},
    }

    def _expr_record(self, target="/a.scale.y"):
        return _record(
            {
                "id": "r1",
                "shape": "channel",
                "op": "expr",
                "target": target,
                "sources": [
                    {"plug": "/b.stretch", "role": "a"},
                    {"plug": "/b.translate.y", "role": "b"},
                ],
                "params": {"expr": "a * b"},
            }
        )

    def test_user_named_roles_are_not_checked_against_a_vocabulary(self):
        cap = _capability(self.EXPR_CAP)
        self.assertEqual(cap.rejects(self._expr_record()), [])

    def test_a_channel_op_may_still_opt_into_a_channel_restriction(self):
        # A target that drives transforms but not blend-shape weights says so.
        cap = _capability(
            {
                "channel/expr": dict(
                    self.EXPR_CAP["channel/expr"], channels=["scale.y", "translate.x"]
                )
            }
        )
        self.assertEqual(cap.rejects(self._expr_record()), [])
        reasons = dict(cap.rejects(self._expr_record("/a.blendshape.smile")))
        self.assertEqual(reasons.get("unsupported_channel"), "blendshape.smile")

    def test_a_transform_op_still_gets_its_vocabulary_enforced(self):
        # The exemption is for `channel` alone -- it must not weaken the shapes
        # that DO have a fixed vocabulary.
        cap = _capability(BLEND_OK)
        record = _record(
            {
                "id": "r1",
                "shape": "transform",
                "op": "blend",
                "target": {"id": "/a", "channels": ["translate"]},
                "sources": [{"id": "/b", "role": "not_a_role"}],
            }
        )
        self.assertEqual(
            dict(cap.rejects(record)).get("unsupported_role"), "not_a_role"
        )


class TestCapability(unittest.TestCase):
    """A target's answer to 'can you build this?', without planning a graph."""

    def test_a_supported_record_is_rejected_for_nothing(self):
        self.assertEqual(_capability(BLEND_OK).rejects(_record(_blend())), [])

    def test_an_absent_shape_rejects_before_the_op_is_consulted(self):
        cap = _capability(BLEND_OK, shapes=("channel",))
        self.assertEqual(cap.rejects(_record(_blend()))[0][0], "unsupported_shape")

    def test_an_unregistered_op_is_no_builder(self):
        cap = _capability({})
        self.assertEqual(cap.rejects(_record(_blend()))[0][0], "no_builder")

    def test_an_undeclared_channel_role_and_param_are_each_named(self):
        cap = _capability(BLEND_OK)
        record = _record(
            {
                "id": "r1",
                "shape": "transform",
                "op": "blend",
                "target": {"id": "/a", "channels": ["scale"]},  # not declared
                "sources": [{"id": "/b", "role": "aim"}],  # not a blend role
                "params": {"compose": "matrix"},  # not a declared param
            }
        )
        reasons = dict(cap.rejects(record))
        self.assertIn("unsupported_channel", reasons)
        self.assertIn("unsupported_role", reasons)
        self.assertIn("unsupported_param", reasons)

    def test_an_enum_value_outside_the_declared_list_is_refused(self):
        cap = _capability(
            {
                "transform/spline_ik": {
                    "fidelity": "approximate",
                    "roles": ["curve"],
                    "params": {"twist.distribution": ["linear"]},
                }
            }
        )
        record = _record(
            {
                "id": "r1",
                "shape": "transform",
                "op": "spline_ik",
                "target": {"id": "/a"},
                "sources": [{"id": "/c", "role": "curve"}],
                "params": {"twist": {"distribution": "smooth"}},
            }
        )
        self.assertEqual(cap.rejects(record)[0][0], "unsupported_param")

    def test_a_plug_on_an_undrivable_path_is_named(self):
        cap = _capability(
            {
                "transform/blend": dict(
                    BLEND_OK["transform/blend"],
                    plugs=(),  # nothing may be driven
                )
            }
        )
        record = _record(
            _blend(sources=())
            | {
                "sources": [
                    {"id": "/b", "role": "space", "weight": {"plug": "/c.follow"}}
                ]
            }
        )
        reasons = dict(cap.rejects(record))
        self.assertIn("undrivable", reasons)
        self.assertEqual(reasons["undrivable"], "sources.weight")

    def test_a_declared_plug_path_passes(self):
        record = _record(
            _blend()
            | {
                "sources": [
                    {"id": "/b", "role": "space", "weight": {"plug": "/c.follow"}}
                ]
            }
        )
        self.assertEqual(_capability(BLEND_OK).rejects(record), [])

    def test_manifest_round_trips_and_sorts_its_ops(self):
        cap = RigCapability.from_dict(
            {
                "target": "t",
                "shapes": ["transform"],
                "ops": {
                    "transform/z": {"fidelity": "exact"},
                    "transform/a": {"fidelity": "exact"},
                },
            }
        )
        # Sorted so a regenerated manifest diffs cleanly against the committed one.
        self.assertEqual(list(cap.to_dict()["ops"]), ["transform/a", "transform/z"])


def _record(data):
    from pythontk.core_utils.engines.rig_graph.rig_model import RigRecord

    return RigRecord.from_dict(data)


class TestPlanner(unittest.TestCase):
    """The planner decides; every loss it causes, it reports."""

    def test_a_buildable_record_is_native_and_needs_no_verification(self):
        result = RigPlanner.plan(_graph([_blend()]), _capability(BLEND_OK))
        self.assertEqual(result.build, ["r1"])
        self.assertEqual(result.bake, [])
        self.assertEqual(result.verify, {})
        self.assertEqual(result.counts(), {"native": 1})
        self.assertEqual(result.worst_severity(), "info")

    def test_an_unbuildable_record_bakes_its_target_and_says_why(self):
        result = RigPlanner.plan(_graph([_blend()]), _capability({}))
        self.assertEqual(result.build, [])
        self.assertEqual(result.bake, ["/a"])
        entry = result.report[0]
        self.assertEqual((entry.kind, entry.reason), ("baked", "no_builder"))
        # Baked is INFO: the motion survives, only editability is lost.
        self.assertEqual(entry.severity, "info")
        self.assertTrue(entry.recoverable)

    def test_drop_removes_it_and_warns_where_bake_would_not(self):
        record = _blend(policy={"fallback": "drop"})
        result = RigPlanner.plan(_graph([record]), _capability({}))
        self.assertEqual(result.bake, [])
        self.assertEqual(result.report[0].kind, "dropped")
        self.assertEqual(result.worst_severity(), "warn")

    def test_fail_refuses_the_whole_plan(self):
        record = _blend(policy={"fallback": "fail"})
        with self.assertRaises(RigPlanRefused):
            RigPlanner.plan(_graph([record]), _capability({}))

    def test_a_disabled_record_is_skipped_silently(self):
        record = _blend(policy={"enabled": False})
        result = RigPlanner.plan(_graph([record]), _capability({}))
        # Switched off by its author: not a loss, so not a report line.
        self.assertEqual((result.build, result.bake, result.report), ([], [], []))

    def test_an_approximate_op_always_gets_verified(self):
        cap = _capability(
            {
                "transform/blend": dict(
                    BLEND_OK["transform/blend"], fidelity="approximate"
                )
            }
        )
        graph = _graph([_blend()], policy={"verify": {"tolerance": 0.1}})
        result = RigPlanner.plan(graph, cap)
        self.assertEqual(result.build, ["r1"])
        self.assertEqual(result.verify["r1"]["tolerance"], 0.1)
        self.assertEqual(result.verify["r1"]["fidelity"], "approximate")

    def test_a_record_overrides_the_graphs_verify_default(self):
        cap = _capability(
            {
                "transform/blend": dict(
                    BLEND_OK["transform/blend"], fidelity="approximate"
                )
            }
        )
        graph = _graph(
            [_blend(policy={"verify": {"tolerance": 0.01}})],
            policy={"verify": {"tolerance": 5.0}},
        )
        self.assertEqual(RigPlanner.plan(graph, cap).verify["r1"]["tolerance"], 0.01)


class TestNodeRule(unittest.TestCase):
    """A half-built node is the one outcome worse than a baked one."""

    def test_one_unbuildable_driver_bakes_the_others_on_that_node(self):
        # Two records drive '/a'; only `blend` has a builder.
        records = [
            _blend("r1", target="/a"),
            {
                "id": "r2",
                "shape": "transform",
                "op": "aim",
                "target": {"id": "/a", "channels": ["translate"]},
                "sources": [{"id": "/c", "role": "target"}],
            },
        ]
        result = RigPlanner.plan(_graph(records), _capability(BLEND_OK))
        self.assertEqual(result.build, [])
        self.assertEqual(result.bake, ["/a"])
        demoted = [e for e in result.report if e.reason == "node_baked"]
        self.assertEqual([e.record for e in demoted], ["r1"])

    def test_demotion_cascades_to_a_records_other_targets(self):
        # r2 is unbuildable and bakes '/b'. r1 drives BOTH '/a' and '/b', so it
        # is demoted -- which must bake '/a' as well, and that takes r3 with it.
        records = [
            {
                "id": "r1",
                "shape": "transform",
                "op": "blend",
                "target": {"chain": ["/a", "/b"], "channels": ["translate"]},
                "sources": [{"id": "/s", "role": "space", "weight": 1.0}],
                "params": {"maintain_offset": True},
            },
            {
                "id": "r2",
                "shape": "transform",
                "op": "unbuildable",
                "target": {"id": "/b", "channels": ["translate"]},
                "sources": [],
            },
            _blend("r3", target="/a", sources=("/s",)),
        ]
        result = RigPlanner.plan(_graph(records), _capability(BLEND_OK))
        self.assertEqual(result.build, [])
        self.assertEqual(sorted(result.bake), ["/a", "/b"])
        self.assertEqual(
            sorted(e.record for e in result.report if e.reason == "node_baked"),
            ["r1", "r3"],
        )

    def test_a_demoted_records_verification_is_dropped_with_it(self):
        cap = _capability(
            {
                "transform/blend": dict(
                    BLEND_OK["transform/blend"], fidelity="approximate"
                )
            }
        )
        records = [
            _blend("r1", target="/a"),
            {
                "id": "r2",
                "shape": "transform",
                "op": "nope",
                "target": {"id": "/a"},
                "sources": [],
            },
        ]
        graph = _graph(records, policy={"verify": {"tolerance": 1.0}})
        self.assertEqual(RigPlanner.plan(graph, cap).verify, {})


class TestComponents(unittest.TestCase):
    """A rig component is the unit a consumer can honestly call 'a rig'."""

    def test_records_sharing_a_node_are_one_component_and_strangers_are_not(self):
        graph = _graph(
            [
                _blend("r1", target="/a", sources=("/ctrl",)),
                _blend("r2", target="/b", sources=("/a",)),  # reads r1's target
                _blend("r3", target="/x", sources=("/y",)),  # touches neither
            ]
        )
        self.assertEqual(graph.components(), [["r1", "r2"], ["r3"]])

    def test_an_opaque_record_joins_through_its_target_ids(self):
        graph = _graph(
            [
                _blend("r1", target="/a", sources=("/ctrl",)),
                {
                    "id": "r2",
                    "shape": "opaque",
                    "op": "opaque",
                    "target": {"ids": ["/ctrl"]},
                    "sources": [],
                },
            ]
        )
        self.assertEqual(graph.components(), [["r1", "r2"]])

    def test_a_restriction_leaves_the_excluded_record_out_of_every_component(self):
        graph = _graph(
            [
                _blend("r1", target="/a", sources=("/ctrl",)),
                _blend("r2", target="/ctrl", sources=("/root",)),  # binds r1 to r3
                _blend("r3", target="/b", sources=("/root",)),
            ]
        )
        self.assertEqual(graph.components(), [["r1", "r2", "r3"]])
        self.assertEqual(graph.components(["r1", "r3"]), [["r1"], ["r3"]])


class TestComponentRule(unittest.TestCase):
    """The planner's component rule: a rig component is all-or-nothing.

    Measured on a production module: 119 records built and verified, none of
    them driving anything usable, because every chain's spline IK and math
    had baked underneath them. A half rig is a bake with clutter.
    """

    APPROX = {
        "transform/blend": dict(BLEND_OK["transform/blend"], fidelity="approximate")
    }

    @staticmethod
    def _spline_ik(record_id, chain, curve, **over):
        record = {
            "id": record_id,
            "shape": "transform",
            "op": "spline_ik",
            "target": {"chain": list(chain), "channels": ["rotate"]},
            "sources": [{"id": curve, "role": "curve"}],
        }
        record.update(over)
        return record

    def _loom(self, **spline_over):
        # ctrl -> driver joint -> curve (stand-in for the skin) -> spline IK chain
        return [
            _blend("r1", target="/drv_jnt", sources=("/ctrl",)),
            _blend("r0", target="/curve", sources=("/drv_jnt",)),
            self._spline_ik("r2", ["/jnt", "/jnt/tip"], "/curve", **spline_over),
            _blend("r3", target="/other", sources=("/elsewhere",)),  # its own rig
        ]

    def test_one_unbuildable_link_bakes_its_whole_component_and_names_the_blocker(self):
        result = RigPlanner.plan(_graph(self._loom()), _capability(self.APPROX))
        self.assertEqual(result.build, ["r3"])
        kinds = {
            e.record: (e.kind, e.reason) for e in result.report if e.kind != "native"
        }
        self.assertEqual(kinds["r2"], ("baked", "no_builder"))
        self.assertEqual(kinds["r1"], ("baked", "component"))
        self.assertEqual(kinds["r0"], ("baked", "component"))
        entry = next(e for e in result.report if e.record == "r1")
        self.assertEqual(entry.detail["blocker"], "r2")
        self.assertEqual(entry.detail["blocker_reason"], "no_builder")
        self.assertEqual(entry.detail["component"], 3)
        for node in ("/drv_jnt", "/curve", "/jnt"):
            self.assertIn(node, result.bake)
        # the cascaded records are no longer measured; the complete rig still is
        self.assertEqual(set(result.verify), {"r3"})

    def test_a_disabled_record_neither_blocks_nor_binds(self):
        records = self._loom(policy={"enabled": False})
        result = RigPlanner.plan(_graph(records), _capability(self.APPROX))
        self.assertEqual(result.build, ["r1", "r0", "r3"])

    def test_a_dropped_link_still_breaks_its_component(self):
        records = self._loom(policy={"fallback": "drop"})
        result = RigPlanner.plan(_graph(records), _capability(self.APPROX))
        self.assertEqual(result.build, ["r3"])
        kinds = {e.record: e.kind for e in result.report}
        self.assertEqual(kinds["r2"], "dropped")
        self.assertEqual(kinds["r1"], "baked")

    def test_a_component_that_bakes_entirely_by_itself_is_not_reported_twice(self):
        records = [self._spline_ik("r2", ["/jnt"], "/curve")]
        result = RigPlanner.plan(_graph(records), _capability(self.APPROX))
        self.assertEqual([e.reason for e in result.report], ["no_builder"])

    def test_the_default_tolerance_is_one_centimetre_in_the_graphs_unit(self):
        for unit, want in (("cm", 1.0), ("m", 0.01), ("mm", 10.0)):
            graph = _graph([_blend("r1")])
            graph.source["linear_unit"] = unit
            result = RigPlanner.plan(graph, _capability(self.APPROX))
            self.assertAlmostEqual(result.verify["r1"]["tolerance"], want, msg=unit)

    def test_a_stated_tolerance_is_kept_whatever_the_unit(self):
        graph = _graph([_blend("r1")], policy={"verify": {"tolerance": 0.25}})
        graph.source["linear_unit"] = "m"
        result = RigPlanner.plan(graph, _capability(self.APPROX))
        self.assertEqual(result.verify["r1"]["tolerance"], 0.25)


class TestCycles(unittest.TestCase):
    """A cycle the rigger meant still plays once baked — it is only frozen."""

    def test_a_two_node_cycle_bakes_both_and_reports_once_per_record(self):
        records = [
            _blend("r1", target="/a", sources=("/b",)),
            _blend("r2", target="/b", sources=("/a",)),
        ]
        result = RigPlanner.plan(_graph(records), _capability(BLEND_OK))
        self.assertEqual(result.build, [])
        self.assertEqual(sorted(result.bake), ["/a", "/b"])
        self.assertEqual(sorted(result.counts()), ["cyclic"])
        self.assertEqual(result.worst_severity(), "warn")

    def test_a_self_edge_is_a_cycle(self):
        result = RigPlanner.plan(
            _graph([_blend("r1", target="/a", sources=("/a",))]), _capability(BLEND_OK)
        )
        self.assertEqual(result.report[0].kind, "cyclic")

    def test_a_long_chain_is_not_a_cycle_and_does_not_exhaust_the_stack(self):
        # Iterative Tarjan: a deep rig must not hit the recursion limit.
        depth = 3000
        records = [
            _blend(f"r{i}", target=f"/n{i + 1}", sources=(f"/n{i}",))
            for i in range(depth)
        ]
        result = RigPlanner.plan(_graph(records), _capability(BLEND_OK))
        self.assertEqual(len(result.build), depth)
        self.assertEqual(result.bake, [])

    def test_a_diamond_is_not_a_cycle(self):
        records = [
            _blend("r1", target="/b", sources=("/a",)),
            _blend("r2", target="/c", sources=("/a",)),
            _blend("r3", target="/d", sources=("/b",)),
            _blend("r4", target="/d2", sources=("/c",)),
        ]
        result = RigPlanner.plan(_graph(records), _capability(BLEND_OK))
        self.assertEqual(len(result.build), 4)

    def test_order_and_opaque_records_create_no_dependency_edges(self):
        # `order` states sequence and `opaque` states that nothing is known;
        # neither is a dependency, so neither can manufacture a cycle.
        records = [
            {
                "id": "r1",
                "shape": "order",
                "op": "order",
                "target": {"id": "/a"},
                "params": {"records": ["r2"]},
            },
            {
                "id": "r2",
                "shape": "opaque",
                "op": "opaque",
                "target": {"ids": ["/a"]},
                "params": {},
            },
        ]
        graph = _graph(records, nodes=["/a"])
        self.assertEqual(graph.validate(), [])
        result = RigPlanner.plan(graph, _capability(BLEND_OK))
        self.assertNotIn("cyclic", result.counts())


class TestModelExtras(unittest.TestCase):
    """The parts of the document contract nothing else exercises."""

    def test_a_nodes_parent_is_derived_from_its_path(self):
        # The spec has no `parent` field on purpose -- a second spelling of the
        # hierarchy is a second thing to keep consistent.
        graph = RigGraph.from_dict(
            {
                "version": SCHEMA_VERSION,
                "nodes": [{"id": "/rig/ctrls/ctrl_L"}],
                "records": [],
            }
        )
        self.assertEqual(graph.node("/rig/ctrls/ctrl_L").parent, "/rig/ctrls")

    def test_an_order_record_naming_a_missing_record_is_caught(self):
        # A dangling reference orders nothing, silently -- the failure mode this
        # schema exists to prevent.
        records = [
            {
                "id": "r1",
                "shape": "order",
                "op": "order",
                "target": {"id": "/a"},
                "params": {"records": ["r2", "ghost"]},
            },
            {
                "id": "r2",
                "shape": "opaque",
                "op": "opaque",
                "target": {"ids": ["/a"]},
                "params": {},
            },
        ]
        errors = _graph(records, nodes=["/a"]).validate()
        self.assertTrue(any("ghost" in e for e in errors), errors)


class TestCoverage(unittest.TestCase):
    """An omitted driver is indistinguishable from "this node is free" — so an
    extractor has to PROVE it accounted for what it saw, not promise it."""

    CENSUS = {"parentConstraint": 217, "multMatrix": 196, "ikHandle": 7}

    def _graph_with(self, provenance):
        return RigGraph.from_dict(
            {
                "version": SCHEMA_VERSION,
                "source": {"census": self.CENSUS},
                "nodes": [{"id": "/a"}],
                "records": [
                    {
                        "id": f"r{i}",
                        "shape": "transform",
                        "op": "blend",
                        "target": {"id": "/a"},
                        "provenance": [entry],
                    }
                    for i, entry in enumerate(provenance)
                ],
            }
        )

    def test_an_extractor_that_saw_more_than_it_emitted_is_caught(self):
        # The measured production shape: every constraint read, the whole
        # matrix graph silently skipped.
        graph = self._graph_with(
            [f"parentConstraint:c{i}" for i in range(217)]
            + [f"ikHandle:h{i}" for i in range(7)]
        )
        coverage = graph.coverage()
        self.assertEqual(coverage["seen"], 420)
        self.assertEqual(coverage["accounted"], 224)
        self.assertEqual(coverage["unaccounted"], 196)
        self.assertEqual(coverage["by_type"], {"multMatrix": 196})

    def test_full_coverage_leaves_nothing_unaccounted(self):
        graph = self._graph_with(
            [f"parentConstraint:c{i}" for i in range(217)]
            + [f"multMatrix:m{i}" for i in range(196)]
            + [f"ikHandle:h{i}" for i in range(7)]
        )
        self.assertEqual(graph.coverage()["unaccounted"], 0)
        self.assertEqual(graph.coverage()["by_type"], {})

    def test_an_opaque_record_counts_as_accounted_for(self):
        # Opaque is the whole point: it says "I saw this and cannot describe
        # it", which is a world away from saying nothing.
        graph = RigGraph.from_dict(
            {
                "version": SCHEMA_VERSION,
                "source": {"census": {"rivet": 2}},
                "nodes": [{"id": "/a"}],
                "records": [
                    {
                        "id": "r1",
                        "shape": "opaque",
                        "op": "opaque",
                        "target": {"ids": ["/a"]},
                        "provenance": ["rivet:rv1", "rivet:rv2"],
                    }
                ],
            }
        )
        self.assertEqual(graph.coverage()["unaccounted"], 0)

    def test_no_census_means_the_graph_cannot_make_the_promise(self):
        graph = _graph([_blend()])
        self.assertEqual(graph.coverage()["seen"], 0)

    def test_two_records_reading_one_node_do_not_cover_a_second(self):
        # Counting occurrences instead of distinct nodes let one constraint
        # stand in for another and reported unaccounted=0 while a real driver
        # was never read -- the exact false negative this function exists for.
        graph = RigGraph.from_dict(
            {
                "version": SCHEMA_VERSION,
                "source": {"census": {"parentConstraint": 2}},
                "nodes": [{"id": "/a"}],
                "records": [
                    {
                        "id": "r1",
                        "shape": "transform",
                        "op": "blend",
                        "target": {"id": "/a"},
                        "provenance": ["parentConstraint:c1"],
                    },
                    {
                        "id": "r2",
                        "shape": "transform",
                        "op": "blend",
                        "target": {"id": "/a"},
                        "provenance": ["parentConstraint:c1"],
                    },
                ],
            }
        )
        coverage = graph.coverage()
        self.assertEqual(coverage["accounted"], 1)
        self.assertEqual(coverage["by_type"], {"parentConstraint": 1})

    def test_provenance_for_a_type_the_census_never_mentioned_is_surfaced(self):
        # An extractor contradicting itself: neither number can be trusted.
        graph = self._graph_with(["rivet:rv1"])
        self.assertEqual(graph.coverage()["uncensused"], {"rivet": 1})

    def test_a_namespaced_node_keeps_its_colons(self):
        # `ns:con1` -- only the FIRST colon separates the type.
        graph = self._graph_with(["parentConstraint:ns:con1"])
        self.assertEqual(graph.coverage()["by_type"]["parentConstraint"], 216)

    def test_provenance_survives_a_round_trip(self):
        graph = self._graph_with(["parentConstraint:c0"])
        again = RigGraph.from_dict(graph.to_dict())
        self.assertEqual(again.records[0].provenance, ["parentConstraint:c0"])


class TestPreFlight(unittest.TestCase):
    """The whole point: an honest summary before anything is written."""

    def test_the_report_summarises_without_a_dcc(self):
        records = [
            _blend("r1", target="/a"),
            {
                "id": "r2",
                "shape": "points",
                "op": "wire",
                "target": {"id": "/m"},
                "sources": [{"id": "/w", "role": "wire"}],
            },
        ]
        result = RigPlanner.plan(_graph(records), _capability(BLEND_OK))
        self.assertEqual(result.counts(), {"native": 1, "baked": 1})
        baked = result.entries("info")
        self.assertTrue(any(e.reason == "unsupported_shape" for e in baked))

    def test_the_plan_serialises_to_plain_values(self):
        result = RigPlanner.plan(_graph([_blend()]), _capability({}))
        data = result.to_dict()
        self.assertEqual(data["bake"], ["/a"])
        self.assertEqual(data["report"][0]["kind"], "baked")
        self.assertEqual(data["report"][0]["severity"], "info")


class TestRigVerify(unittest.TestCase):
    """§9.4: verification is a step, not a hope -- and ONE implementation.

    The exporter, the importer and the tests all compare sampled world points
    through this, so the number a ``diverged`` report entry carries is the same
    number everywhere. Pure math over point arrays: the DCC contributes only a
    sampler.
    """

    def setUp(self):
        from pythontk import RigVerify

        self.verify = RigVerify
        self.tri = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

    def test_identical_points_measure_zero_everywhere(self):
        m = self.verify.compare(self.tri, list(self.tri))
        self.assertEqual(
            (m["worst"], m["rigid"], m["residual"], m["count"]), (0.0, 0.0, 0.0, 3)
        )

    def test_a_pure_translation_is_all_placement_and_no_shape(self):
        # Every point moved by the same vector: the rigid part IS the move and
        # the residual (shape) part is nothing -- the split the return leg's
        # 0.0126 / 0.0605 mm reading depends on.
        moved = [(x + 3.0, y, z) for x, y, z in self.tri]
        m = self.verify.compare(self.tri, moved)
        self.assertAlmostEqual(m["worst"], 3.0)
        self.assertAlmostEqual(m["rigid"], 3.0)
        self.assertAlmostEqual(m["residual"], 0.0)

    def test_one_moved_point_shows_up_as_shape_not_placement(self):
        moved = list(self.tri)
        moved[2] = (0.0, 1.0, 3.0)
        m = self.verify.compare(self.tri, moved)
        self.assertAlmostEqual(m["worst"], 3.0)
        self.assertAlmostEqual(m["rigid"], 1.0)  # the mean delta is (0, 0, 1)
        self.assertGreater(m["residual"], 1.9)

    def test_a_point_count_mismatch_is_a_topology_failure_not_a_distance(self):
        with self.assertRaises(ValueError):
            self.verify.compare(self.tri, self.tri[:2])

    def test_frames_take_the_worst_and_keep_each_frame(self):
        far = [(x + 5.0, y, z) for x, y, z in self.tri]
        m = self.verify.compare_frames(
            {0: self.tri, 10: self.tri}, {0: self.tri, 10: far}
        )
        self.assertAlmostEqual(m["worst"], 5.0)
        self.assertEqual(sorted(m["per_frame"]), [0, 10])
        self.assertAlmostEqual(m["per_frame"][10], 5.0)
        self.assertAlmostEqual(m["per_frame"][0], 0.0)

    def test_a_verdict_carries_both_numbers(self):
        ok = self.verify.verdict(0.05, tolerance=0.1)
        bad = self.verify.verdict(1.24, tolerance=0.1)
        self.assertTrue(ok["passed"])
        self.assertFalse(bad["passed"])
        self.assertEqual((bad["measured"], bad["tolerance"]), (1.24, 0.1))


class TestRigVerifyPlan(unittest.TestCase):
    """The verify-and-demote loop is ONE implementation; a DCC contributes only
    a sampler. A record with nothing measurable is reported, never passed."""

    def _result(self):
        return {
            "built": ["r1", "r2", "r3"],
            "baked": [],
            "verify": {
                "r1": {"tolerance": 0.5},
                "r2": {"tolerance": 0.5},
                "r3": {"tolerance": 0.5},
            },
            "report": [
                {"kind": "native", "record": "r1", "nodes": ["/a"]},
                {"kind": "native", "record": "r2", "nodes": ["/b"]},
                {"kind": "native", "record": "r3", "nodes": ["/skel/bone"]},
            ],
        }

    def test_a_miss_demotes_with_numbers_and_takes_the_build_back(self):
        from pythontk import RigVerify

        removed = []
        # Source is cm, Y-up; the target is m, Z-up: (x, y, z)cm -> (x, -z, y)/100 m.
        samples = {"/a": {"1": [100.0, 0.0, 0.0]}, "/b": {"1": [0.0, 100.0, 0.0]}}
        got = {
            "/a": (1.0, 0.0, 0.0),
            "/b": (0.0, 0.0, 0.0),
        }  # /b is 1 m off (want (0, 0, 1))
        result = self._result()
        demoted = RigVerify.verify_plan(
            result,
            samples,
            lambda node_id, frame: got.get(node_id),
            source_unit="cm",
            source_up_axis="y",
            remove=removed.append,
        )
        self.assertEqual(demoted, ["r2"])
        self.assertEqual(result["built"], ["r1", "r3"])
        self.assertIn("/b", result["baked"])
        self.assertEqual(removed, ["r2"])
        entry = next(e for e in result["report"] if e["kind"] == "diverged")
        self.assertEqual(entry["record"], "r2")
        self.assertAlmostEqual(entry["detail"]["measured"], 1.0)
        self.assertAlmostEqual(entry["detail"]["tolerance"], 0.005)  # 0.5 cm in metres

    def test_nothing_measurable_is_reported_never_passed_silently(self):
        from pythontk import RigVerify

        result = self._result()
        samples = {"/skel/bone": {"1": [0.0, 0.0, 0.0]}}
        RigVerify.verify_plan(
            result,
            samples,
            lambda node_id, frame: None,
            source_unit="cm",
            source_up_axis="y",
            remove=lambda rid: None,
        )
        kinds = {
            e["record"]: e["kind"]
            for e in result["report"]
            if e.get("kind") in ("diverged", "unverified")
        }
        self.assertEqual(kinds.get("r3"), "unverified")
        self.assertIn(
            "r3", result["built"]
        )  # built stays built; it is the CHECK that did not happen

    def test_frame_offset_and_matching_axes(self):
        from pythontk import RigVerify

        seen = []
        result = self._result()
        RigVerify.verify_plan(
            result,
            {"/a": {"10": [1.0, 2.0, 3.0]}},
            lambda node_id, frame: (seen.append(frame), (1.0, 2.0, 3.0))[1],
            source_unit="m",
            source_up_axis="z",
            frame_offset=1.0,
            remove=lambda rid: None,
        )
        self.assertEqual(seen, [11])
        self.assertNotIn("diverged", {e["kind"] for e in result["report"]})

    def _component_graph(self):
        return _graph(
            [
                _blend("r1", target="/a", sources=("/ctrl",)),
                _blend("r2", target="/b", sources=("/a",)),  # one rig with r1
                _blend("r3", target="/skel/bone", sources=("/z",)),  # its own
            ]
        )

    def test_a_diverged_link_takes_its_whole_component_back(self):
        from pythontk import RigVerify

        removed = []
        result = self._result()
        samples = {
            "/a": {"1": [100.0, 0.0, 0.0]},
            "/b": {"1": [0.0, 100.0, 0.0]},
            "/skel/bone": {"1": [0.0, 0.0, 0.0]},
        }
        got = {
            "/a": (1.0, 0.0, 0.0),
            "/b": (0.0, 0.0, 0.0),
            "/skel/bone": (0.0, 0.0, 0.0),
        }
        demoted = RigVerify.verify_plan(
            result,
            samples,
            lambda node_id, frame: got.get(node_id),
            source_unit="cm",
            source_up_axis="y",
            remove=removed.append,
            graph=self._component_graph(),
        )
        self.assertEqual(demoted, ["r2", "r1"])  # the miss, then its sibling
        self.assertEqual(result["built"], ["r3"])
        self.assertEqual(removed, ["r2", "r1"])
        entry = next(e for e in result["report"] if e["kind"] == "cascaded")
        self.assertEqual((entry["record"], entry["reason"]), ("r1", "component"))
        self.assertEqual(entry["detail"], {"blocker": "r2", "component": 2})
        self.assertIn("/a", result["baked"])

    def test_a_failed_build_is_a_dead_link_too_and_the_graph_may_be_plain(self):
        from pythontk import RigVerify

        removed = []
        result = {
            "built": ["r1", "r3"],
            "baked": ["/b"],
            "verify": {},
            "report": [
                {"kind": "native", "record": "r1", "nodes": ["/a"]},
                {"kind": "failed", "record": "r2", "nodes": ["/b"]},
                {"kind": "native", "record": "r3", "nodes": ["/skel/bone"]},
            ],
        }
        demoted = RigVerify.verify_plan(
            result,
            {},
            lambda node_id, frame: None,
            remove=removed.append,
            graph=self._component_graph().to_dict(),
        )
        self.assertEqual(demoted, ["r1"])
        self.assertEqual(result["built"], ["r3"])
        self.assertEqual(removed, ["r1"])

    def test_without_the_graph_demotion_stays_per_record(self):
        from pythontk import RigVerify

        result = self._result()
        samples = {"/a": {"1": [100.0, 0.0, 0.0]}, "/b": {"1": [0.0, 100.0, 0.0]}}
        got = {"/a": (1.0, 0.0, 0.0), "/b": (0.0, 0.0, 0.0)}
        demoted = RigVerify.verify_plan(
            result,
            samples,
            lambda node_id, frame: got.get(node_id),
            source_unit="cm",
            source_up_axis="y",
            remove=lambda rid: None,
        )
        self.assertEqual(demoted, ["r2"])
        self.assertEqual(result["built"], ["r1", "r3"])

    def test_the_default_tolerance_is_a_physical_centimetre_whatever_the_unit(self):
        from pythontk import RigVerify

        self.assertAlmostEqual(RigVerify.default_tolerance("cm"), 1.0)
        self.assertAlmostEqual(RigVerify.default_tolerance("m"), 0.01)
        self.assertAlmostEqual(RigVerify.default_tolerance("unknown"), 1.0)
        # A metre-scaled source that states no tolerance: a 5 cm miss demotes.
        result = {
            "built": ["r1"],
            "baked": [],
            "verify": {"r1": {}},
            "report": [{"kind": "native", "record": "r1", "nodes": ["/a"]}],
        }
        RigVerify.verify_plan(
            result,
            {"/a": {"1": [0.0, 0.0, 0.0]}},
            lambda node_id, frame: (0.05, 0.0, 0.0),
            source_unit="m",
            source_up_axis="z",
            target_unit="m",
            target_up_axis="z",
            remove=lambda rid: None,
        )
        self.assertEqual(result["built"], [])
        entry = next(e for e in result["report"] if e["kind"] == "diverged")
        self.assertAlmostEqual(entry["detail"]["tolerance"], 0.01)

    def test_summary_and_demoted_speak_one_vocabulary_for_every_consumer(self):
        from pythontk import RigVerify

        result = {
            "built": ["r1"],
            "baked": ["/b", "/c"],
            "report": [
                {"kind": "native", "record": "r1"},
                {"kind": "failed", "record": "r2"},
                {"kind": "cascaded", "record": "r3", "reason": "component"},
                {"kind": "baked", "record": "r4", "reason": "component"},
                {"kind": "baked", "record": "r5", "reason": "no_builder"},
                {"kind": "unverified", "record": "r6"},
            ],
        }
        self.assertEqual([e["record"] for e in RigVerify.demoted(result)], ["r2", "r3"])
        line = RigVerify.summary(result)
        self.assertTrue(
            line.startswith("1 record(s) built, 2 node(s) left to the bake, 2 demoted")
        )
        self.assertIn("'component': 1", line)
        self.assertIn("'no_builder': 1", line)
        self.assertEqual(
            RigVerify.summary({"built": [], "baked": [], "report": []}),
            "0 record(s) built, 0 node(s) left to the bake, 0 demoted.",
        )


class _FakeBuilder:
    """The builder protocol, recording the ORDER it was driven in."""

    def __init__(self, result, points=None):
        self._result = result
        self._points = points or {}
        self.calls = []

    def build(self, graph, imported, is_usd=False):
        self.calls.append(("build", is_usd))
        return self._result

    def commit(self, record_id):
        self.calls.append(("commit", record_id))
        return 2  # two key curves deleted, every time

    def remove(self, record_id):
        self.calls.append(("remove", record_id))
        return 1

    def sample_world(self, node_id, frame):
        self.calls.append(("sample", node_id))
        return self._points.get(node_id)

    @contextmanager
    def scope(self):
        self.calls.append(("scope_in", None))
        try:
            yield
        finally:
            self.calls.append(("scope_out", None))

    @staticmethod
    def linear_unit():
        return "m"

    @staticmethod
    def up_axis():
        return "z"


class _FakeLogger:
    def __init__(self):
        self.info_lines, self.warning_lines = [], []

    def info(self, message):
        self.info_lines.append(str(message))

    def warning(self, message):
        self.warning_lines.append(str(message))


class TestRigTransfer(unittest.TestCase):
    """The ONE orchestration both bridges run: build, verify, commit, report."""

    def _section(self, samples=None):
        graph = _graph([_blend("r1", target="/a", sources=("/ctrl",))]).to_dict()
        return {"graph": graph, "verify_samples": samples or {}}

    def _result(self, built=("r1",)):
        return {
            "built": list(built),
            "baked": [],
            "verify": {"r1": {"tolerance": 1.0}},
            "report": [{"kind": "native", "record": "r1", "nodes": ["/a"]}],
            "edits": [],
        }

    def test_no_graph_does_nothing_at_all(self):
        from pythontk import RigTransfer

        builder = _FakeBuilder(self._result())
        for section in (None, {}, {"graph": None}):
            self.assertIsNone(RigTransfer.apply(section, builder, []))
        self.assertEqual(builder.calls, [])

    def test_commit_happens_AFTER_the_measurement_and_inside_no_scope(self):
        # Committing first would delete the payload's keys before anything was
        # measured, leaving a record that then FAILS verification with no
        # motion at all -- worse than either outcome on its own.
        from pythontk import RigTransfer

        builder = _FakeBuilder(self._result(), {"/a": (0.0, 0.0, 0.0)})
        RigTransfer.apply(self._section({"/a": {"1": [0.0, 0.0, 0.0]}}), builder, [])
        kinds = [c[0] for c in builder.calls]
        self.assertLess(kinds.index("scope_out"), kinds.index("commit"))
        self.assertLess(kinds.index("build"), kinds.index("scope_in"))
        self.assertEqual(kinds.count("scope_in"), 1)

    def test_a_diverged_record_is_never_committed(self):
        from pythontk import RigTransfer

        # Sampled at the origin, measured a metre away: a miss at 1 cm.
        builder = _FakeBuilder(self._result(), {"/a": (1.0, 0.0, 0.0)})
        result = RigTransfer.apply(
            self._section({"/a": {"1": [0.0, 0.0, 0.0]}}), builder, []
        )
        self.assertEqual(result["built"], [])
        self.assertNotIn("commit", [c[0] for c in builder.calls])
        self.assertIn(("remove", "r1"), builder.calls)

    def test_the_destructive_edits_are_reported_not_silent(self):
        from pythontk import RigTransfer

        result = self._result()
        result["edits"] = ["3 connected bone(s) unglued"]
        builder = _FakeBuilder(result, {"/a": (0.0, 0.0, 0.0)})
        logger = _FakeLogger()
        RigTransfer.apply(
            self._section({"/a": {"1": [0.0, 0.0, 0.0]}}), builder, [], logger=logger
        )
        joined = " ".join(logger.info_lines)
        self.assertIn("2 baked key curve(s) deleted", joined)
        self.assertIn("3 connected bone(s) unglued", joined)
        self.assertIn("1 record(s) built", joined)

    def test_nothing_destroyed_says_nothing_about_destruction(self):
        from pythontk import RigTransfer

        builder = _FakeBuilder(self._result(built=()), {"/a": (0.0, 0.0, 0.0)})
        logger = _FakeLogger()
        RigTransfer.apply(self._section(), builder, [], logger=logger)
        self.assertNotIn("deleted", " ".join(logger.info_lines))

    def test_the_source_fallback_is_the_PRODUCERS_and_differs_by_direction(self):
        # A graph that omits its unit is assumed to be the OTHER app's
        # convention, and the two directions disagree: Maya samples in
        # centimetres Y-up, Blender in metres Z-up. One shared constant here
        # scales every sample by 100 and swaps two axes for one of them --
        # which diverges every record and reads as "the rig would not transfer".
        from pythontk import RigTransfer

        graph = _graph([_blend("r1", target="/a", sources=("/ctrl",))]).to_dict()
        graph["source"] = {"app": "blender"}  # states no unit at all
        section = {"graph": graph, "verify_samples": {"/a": {"1": [0.0, 1.0, 0.0]}}}

        def run(**over):
            # Sampled 1 unit up in the SOURCE's frame; the builder reports the
            # metres/Z-up position of that point. Only the right assumption
            # converts them onto each other.
            builder = _FakeBuilder(self._result(), {"/a": (0.0, 0.0, 1.0)})
            RigTransfer.apply(section, builder, [], **over)
            return builder

        # Blender producer: 1 m up in Y-up source space -> (0, 0, 1) m. Passes.
        kept = run(source_unit="m", source_up_axis="y")
        self.assertIn(("commit", "r1"), kept.calls)
        # Maya's centimetre default on the same graph is 100x off: demoted.
        lost = run(source_unit="cm", source_up_axis="y")
        self.assertNotIn(("commit", "r1"), lost.calls)

    def test_a_graph_that_states_no_unit_is_warned_about_not_absorbed(self):
        from pythontk import RigTransfer

        graph = _graph([_blend("r1", target="/a", sources=("/ctrl",))]).to_dict()
        graph["source"] = {"app": "maya"}
        logger = _FakeLogger()
        RigTransfer.apply(
            {"graph": graph},
            _FakeBuilder(self._result()),
            [],
            logger=logger,
        )
        self.assertTrue(
            any("states no linear unit" in w for w in logger.warning_lines),
            logger.warning_lines,
        )

    def test_a_graph_that_states_its_unit_overrides_the_fallback(self):
        from pythontk import RigTransfer

        graph = _graph([_blend("r1", target="/a", sources=("/ctrl",))]).to_dict()
        graph["source"] = {"app": "maya", "linear_unit": "m", "up_axis": "y"}
        logger = _FakeLogger()
        builder = _FakeBuilder(self._result(), {"/a": (0.0, 0.0, 1.0)})
        RigTransfer.apply(
            {"graph": graph, "verify_samples": {"/a": {"1": [0.0, 1.0, 0.0]}}},
            builder,
            [],
            source_unit="cm",  # the graph's own "m" must win over this
            source_up_axis="y",
            logger=logger,
        )
        self.assertIn(("commit", "r1"), builder.calls)
        self.assertEqual(logger.warning_lines, [])

    def test_the_capability_key_tracks_content_not_dict_order(self):
        from pythontk import RigTransfer

        a = RigTransfer.capability_key({"target": "x", "ops": {"a": 1, "b": 2}})
        b = RigTransfer.capability_key({"ops": {"b": 2, "a": 1}, "target": "x"})
        c = RigTransfer.capability_key({"target": "x", "ops": {"a": 1, "b": 3}})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 12)


class TestMachineryClassify(unittest.TestCase):
    """What a baked rig leaves behind -- the rule, with no DCC in it."""

    # A rig beside the scene's own furniture: one control under its wrapper
    # group with a constraint of its own, an up-vector locator nothing names, a
    # spline handle and its curve, a driver joint, a deform joint, and content.
    SCENE = {
        "|grp": "group",
        "|grp|cube": "content",
        "|grp|cube|cube_parentConstraint1": "constraint",
        "|rig": "group",
        "|rig|ctrl_GRP": "group",
        "|rig|ctrl_GRP|ctrl": "control",
        "|rig|ctrl_GRP|ctrl|ctrl_pointConstraint1": "constraint",
        "|rig|up_loc": "locator",
        "|rig|ik": "ik",
        "|rig|ik_curve": "control",
        "|skel": "joint",
        "|skel|j1": "joint",
        "|driver_jnt": "joint",
        "|artist_null": "group",
        "|data_export": "locator",
        "|cam": "content",
    }
    SEEDS = (
        "|grp|cube|cube_parentConstraint1",
        "|rig|ctrl_GRP|ctrl",
        "|rig|ctrl_GRP|ctrl|ctrl_pointConstraint1",
        "|rig|ik",
        "|rig|ik_curve",
        "|driver_jnt",
    )
    # |skel|j1 deforms the cube; |driver_jnt only skins the rig's own curve.
    PROTECTED = ("|skel|j1",)

    def named(self, **kwargs):
        from pythontk import RigMachinery

        kwargs.setdefault("seeds", self.SEEDS)
        kwargs.setdefault("protected", self.PROTECTED)
        return RigMachinery.classify(dict(self.SCENE), **kwargs)

    def test_the_rigs_apparatus_by_kind_wrapper_groups_included(self):
        self.assertEqual(
            self.named(),
            {
                "|driver_jnt": "joint",
                "|grp|cube|cube_parentConstraint1": "constraint",
                "|rig": "group",
                "|rig|ctrl_GRP": "group",
                "|rig|ctrl_GRP|ctrl": "control",
                "|rig|ctrl_GRP|ctrl|ctrl_pointConstraint1": "constraint",
                "|rig|ik": "ik",
                "|rig|ik_curve": "control",
                "|rig|up_loc": "locator",
            },
        )

    def test_content_keeps_every_ancestor_that_holds_it(self):
        # |grp holds the cube, so the group survives while the constraint node
        # sitting UNDER the cube does not: protection propagates up, never down.
        named = self.named()
        self.assertNotIn("|grp", named)
        self.assertNotIn("|grp|cube", named)
        self.assertIn("|grp|cube|cube_parentConstraint1", named)

    def test_an_influence_of_content_and_its_ancestors_survive(self):
        named = self.named()
        self.assertNotIn("|skel|j1", named)
        self.assertNotIn("|skel", named)

    def test_a_joint_only_the_rigs_own_curve_needs_is_apparatus(self):
        self.assertEqual(self.named().get("|driver_jnt"), "joint")

    def test_apparatus_means_connected_to_a_rig_not_draws_nothing(self):
        # The half that matters. A scene's own marker and an empty null draw
        # nothing either; taking them would be a loss nobody asked for.
        named = self.named()
        self.assertNotIn("|data_export", named)
        self.assertNotIn("|artist_null", named)

    def test_a_group_holding_only_rig_is_swept_all_the_way_down(self):
        # |rig|up_loc is named by nothing and nothing hangs off it: it is reached
        # only because the group above it turned out to hold rig and nothing else.
        self.assertIn("|rig|up_loc", self.named())

    def test_an_unclassifiable_kind_counts_as_content(self):
        from pythontk import RigMachinery

        scene = dict(self.SCENE)
        scene["|rig|ctrl_GRP|ctrl"] = "nparticle"  # a shape nobody taught it
        named = RigMachinery.classify(scene, seeds=self.SEEDS, protected=self.PROTECTED)
        self.assertNotIn("|rig|ctrl_GRP|ctrl", named)
        self.assertNotIn("|rig|ctrl_GRP", named)  # and it protects its holder

    def test_protected_beats_a_seed(self):
        named = self.named(protected=self.PROTECTED + ("|rig|ctrl_GRP|ctrl",))
        self.assertNotIn("|rig|ctrl_GRP|ctrl", named)
        self.assertNotIn("|rig|ctrl_GRP", named)
        self.assertIn("|rig|ik", named)  # the rest of the rig still goes

    def test_no_seeds_names_nothing(self):
        self.assertEqual(self.named(seeds=()), {})

    def test_an_empty_scene_is_not_an_error(self):
        from pythontk import RigMachinery

        self.assertEqual(RigMachinery.classify({}, seeds=("|a",)), {})

    def test_the_separator_is_the_callers(self):
        from pythontk import RigMachinery

        scene = {"/rig": "group", "/rig/ctrl": "control", "/mesh": "content"}
        self.assertEqual(
            RigMachinery.classify(scene, seeds=["/rig/ctrl"], separator="/"),
            {"/rig": "group", "/rig/ctrl": "control"},
        )


class TestMachineryUnambiguous(unittest.TestCase):
    """A carrier keeps names and loses paths, so a shared leaf is a hazard."""

    def test_a_leaf_shared_with_a_survivor_is_given_up_on(self):
        from pythontk import RigMachinery

        kinds = {"|rig|up_loc": "locator", "|rig|ctrl": "control"}
        kept, dropped = RigMachinery.unambiguous(
            kinds, ["|rig|up_loc", "|rig|ctrl", "|elsewhere|up_loc", "|mesh"]
        )
        self.assertEqual(kept, {"|rig|ctrl": "control"})
        self.assertEqual(dropped, ("|rig|up_loc",))

    def test_a_node_under_named_apparatus_is_not_a_survivor(self):
        # A carrier may land a SHAPE as its own object; it travels with its
        # transform, so it must not make that transform ambiguous.
        from pythontk import RigMachinery

        kinds = {"|rig|ctrl": "control"}
        kept, dropped = RigMachinery.unambiguous(
            kinds, ["|rig|ctrl", "|rig|ctrl|ctrlShape", "|other|ctrlShape"]
        )
        self.assertEqual(kept, kinds)
        self.assertEqual(dropped, ())

    def test_two_apparatus_nodes_may_share_a_leaf(self):
        # Both go: deleting either by name is right.
        from pythontk import RigMachinery

        kinds = {"|a|ctrl": "control", "|b|ctrl": "control"}
        kept, dropped = RigMachinery.unambiguous(kinds, list(kinds) + ["|mesh"])
        self.assertEqual(kept, kinds)
        self.assertEqual(dropped, ())


class TestMachinerySelect(unittest.TestCase):
    """The far side's half: name-matched, subtree-checked, refused out loud."""

    SECTION = {
        "|rig|ctrl": "control",
        "|rig|up_loc": "locator",
        "|skel": "joint",
        "|rig|driven": "control",
    }
    SUBTREES = {
        "body": ["body"],
        "skel": ["skel"],
        "ctrl": ["ctrl", "ctrlShape", "ctrlShape1"],
        "up_loc.001": ["up_loc.001"],
        "driven": ["driven"],
        "data_export": ["data_export"],
    }

    def select(self, **kwargs):
        from pythontk import RigMachinery

        kwargs.setdefault("protected", ("body", "skel", "driven"))
        return RigMachinery.select(self.SECTION, dict(self.SUBTREES), **kwargs)

    def test_the_named_apparatus_goes_with_what_hangs_under_it(self):
        doomed, _ = self.select()
        self.assertEqual(
            doomed,
            {
                "ctrl": "control",
                "ctrlShape": "control",
                "ctrlShape1": "control",
                "up_loc.001": "locator",
            },
        )

    def test_a_carriers_collision_suffix_still_matches(self):
        self.assertIn("up_loc.001", self.select()[0])

    def test_a_protected_subtree_is_refused_by_name(self):
        _, refused = self.select()
        self.assertEqual(refused, ("driven", "skel"))

    def test_nothing_the_section_does_not_name_is_touched(self):
        doomed, _ = self.select()
        self.assertNotIn("data_export", doomed)
        self.assertNotIn("body", doomed)

    def test_an_empty_section_is_a_no_op(self):
        from pythontk import RigMachinery

        self.assertEqual(RigMachinery.select({}, self.SUBTREES), ({}, ()))

    def test_the_producers_separator_is_the_callers(self):
        # The other direction spells its paths with "/" (Blender's prim paths),
        # and taking the whole path for the leaf would match nothing at all --
        # a strip that silently does nothing is the failure this guards.
        from pythontk import RigMachinery

        doomed, _ = RigMachinery.select(
            {"/rig/ctrl": "control", "/rig/up_loc": "locator"},
            {"ctrl": ["ctrl"], "up_loc": ["up_loc"], "body": ["body"]},
            protected=("body",),
            separator="/",
        )
        self.assertEqual(doomed, {"ctrl": "control", "up_loc": "locator"})

    def test_the_tally_counts_what_went_by_kind(self):
        from pythontk import RigMachinery

        doomed, _ = self.select()
        self.assertEqual(RigMachinery.tally(doomed), {"control": 3, "locator": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
