# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.engines.instancing.instance_grouping
(InstanceGrouping) -- the DCC-free half of auto-instancing's grouping pass that
mayatk's and blendertk's ``AutoInstancer`` both delegate to.

Every expectation was recorded from the two DCC ``AutoInstancer`` copies
(``_merge_similar_signatures`` / ``default_summary`` / ``format_summary``)
before they were routed here, so these pin the behavior the hoist preserved.
"""

import logging
import unittest

from pythontk import InstanceGrouping

# ``(verts, edges, faces, pca_sig, materials, uv_signature)`` -- the
# GeometryMatcher mesh-signature layout both DCC matchers produce.
_A = (8, 12, 6, (1.0, 0.5, 0.25), ("m1",), ())


class TestMergeSimilarSignatures(unittest.TestCase):
    def _merged(self):
        signature_map = {
            _A: ["a"],
            # Same topology, PCA within the 0.1 absolute tolerance: merges.
            (8, 12, 6, (1.05, 0.5, 0.25), ("m1",), ()): ["b"],
            # Same topology, PCA 0.2 apart: stays its own bucket.
            (8, 12, 6, (1.2, 0.5, 0.25), ("m1",), ()): ["c"],
            # Different material (sig[4:]): never merges.
            (8, 12, 6, (1.0, 0.5, 0.25), ("m2",), ()): ["d"],
            # Topology differs but the PCA is near-identical (relative diff
            # < 0.005): the combine-mode merge.
            (9, 12, 6, (1.0, 0.5, 0.2501), ("m1",), ()): ["e"],
            # No PCA on either side of a same-topology pair: only equal
            # signatures merge, so empty vs. None stay apart.
            (8, 12, 6, (), ("m1",), ()): ["f"],
            (8, 12, 6, None, ("m1",), ()): ["g"],
        }
        return InstanceGrouping.merge_similar_signatures(signature_map)

    def test_near_identical_buckets_merge_into_the_first(self):
        merged = self._merged()
        self.assertEqual(merged[_A], ["a", "b", "e"])

    def test_distinct_buckets_survive_in_topology_order(self):
        merged = self._merged()
        self.assertEqual(
            list(merged.values()), [["a", "b", "e"], ["c"], ["d"], ["f"], ["g"]]
        )

    def test_input_map_is_not_mutated(self):
        signature_map = {_A: ["a"], (8, 12, 6, (1.05, 0.5, 0.25), ("m1",), ()): ["b"]}
        InstanceGrouping.merge_similar_signatures(signature_map)
        self.assertEqual(signature_map[_A], ["a"])

    def test_logs_a_cross_topology_merge_to_the_given_logger(self):
        signature_map = {_A: ["a"], (9, 12, 6, (1.0, 0.5, 0.2501), ("m1",), ()): ["e"]}
        logger = logging.getLogger("test_instance_grouping")
        with self.assertLogs(logger, level="DEBUG") as captured:
            InstanceGrouping.merge_similar_signatures(signature_map, logger=logger)
        self.assertIn("Merging near-identical signature", captured.output[0])

    def test_empty(self):
        self.assertEqual(dict(InstanceGrouping.merge_similar_signatures({})), {})


class TestSummary(unittest.TestCase):
    def test_default_summary_carries_the_micro_threshold(self):
        self.assertEqual(
            InstanceGrouping.default_summary(300),
            {
                "matched_groups": 0,
                "instanced_groups": 0,
                "instances_created": 0,
                "simple_groups": 0,
                "kept_separate_groups": 0,
                "micro_threshold": 300,
                "details": [],
            },
        )

    def test_default_summaries_are_independent(self):
        first = InstanceGrouping.default_summary(300)
        first["details"].append({"name": "x"})
        self.assertEqual(InstanceGrouping.default_summary(300)["details"], [])

    def test_nothing_matched(self):
        summary = InstanceGrouping.default_summary(300)
        self.assertEqual(
            InstanceGrouping.format_summary(summary, 0, 300),
            "Auto Instance: no geometrically identical meshes were found.",
        )
        self.assertEqual(
            InstanceGrouping.format_summary(summary, 3, 300),
            "Auto Instance: no geometrically identical meshes to instance; "
            "combined loose geometry into 3 mesh(es).",
        )

    def test_full_report(self):
        summary = dict(
            InstanceGrouping.default_summary(300),
            matched_groups=3,
            instanced_groups=1,
            instances_created=4,
            details=[
                {"name": "Bolt", "reason": "too_simple", "count": 5, "tris": 12},
                {"name": "Door", "reason": "kept_separate", "count": 2, "tris": 900},
            ],
        )
        self.assertEqual(
            InstanceGrouping.format_summary(summary, 2, 300),
            "Auto Instance: 3 matching group(s) found.\n"
            "- Instanced 1 group(s) -> 4 new instance(s).\n"
            "- 1 group(s) too simple to instance (< 300 tris); combined where "
            "possible: Bolt (x5, 12 tris).\n"
            "- 1 group(s) left separate (flagged individual / non-static): "
            "Door (x2, 900 tris).",
        )

    def test_matched_but_nothing_done(self):
        self.assertEqual(
            InstanceGrouping.format_summary(
                {"matched_groups": 1, "details": None}, 0, 300
            ),
            "Auto Instance: 1 matching group(s) found.\n- Nothing was instanced.",
        )

    def test_summary_threshold_wins_over_the_fallback(self):
        summary = dict(
            InstanceGrouping.default_summary(50),
            matched_groups=1,
            details=[{"name": "Nut", "reason": "too_simple", "count": 9, "tris": 4}],
        )
        self.assertIn("(< 50 tris)", InstanceGrouping.format_summary(summary, 1, 300))
        del summary["micro_threshold"]
        self.assertIn("(< 300 tris)", InstanceGrouping.format_summary(summary, 1, 300))

    def test_report_is_ascii(self):
        # Maya's script editor / Windows consoles: the DCC tests assert this.
        summary = dict(InstanceGrouping.default_summary(300), matched_groups=2)
        InstanceGrouping.format_summary(summary, 1, 300).encode("ascii")


if __name__ == "__main__":
    unittest.main()
