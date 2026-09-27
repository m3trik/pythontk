# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.engines.instancing.instancing_strategy.

The decision tree both DCC ``AutoInstancer`` bindings (mayatk / blendertk
``core_utils/auto_instancer/instancing_strategy.py``) inherit: the cases pin
the behavior it had while each DCC carried its own copy. The one scene read,
the prototype's triangle count, is a hook the bindings override.
"""

import unittest

import pythontk as ptk
from pythontk import InstancingStrategy, StrategyConfig, StrategyType


class _CountingStrategy(InstancingStrategy):
    """A host binding in miniature: the 'mesh' is its own triangle count."""

    def _get_triangle_count(self, mesh_node):
        return int(mesh_node)


class TestStrategyType(unittest.TestCase):
    def test_all_members_present(self):
        self.assertEqual(
            {s.name for s in StrategyType},
            {"BAKE", "COMBINE", "GPU_INSTANCE", "KEEP_SEPARATE"},
        )

    def test_value_round_trip(self):
        self.assertIs(StrategyType("BAKE"), StrategyType.BAKE)


class TestStrategyConfig(unittest.TestCase):
    def test_defaults(self):
        c = StrategyConfig()
        self.assertTrue(c.is_static)
        self.assertFalse(c.needs_individual)
        self.assertFalse(c.will_be_lightmapped)
        self.assertTrue(c.can_gpu_instance)


class TestInstancingStrategyDecisions(unittest.TestCase):
    def _strat(self, **overrides):
        return InstancingStrategy(StrategyConfig(**overrides))

    def test_needs_individual_overrides_everything(self):
        s = self._strat(needs_individual=True)
        self.assertIs(
            s.evaluate(group_size=100, triangle_count=10000), StrategyType.KEEP_SEPARATE
        )

    def test_dynamic_prefers_gpu_instance(self):
        s = self._strat(is_static=False)
        self.assertIs(
            s.evaluate(group_size=2, triangle_count=10), StrategyType.GPU_INSTANCE
        )

    def test_dynamic_without_gpu_keeps_separate(self):
        s = self._strat(is_static=False, can_gpu_instance=False)
        self.assertIs(
            s.evaluate(group_size=2, triangle_count=10), StrategyType.KEEP_SEPARATE
        )

    def test_micro_repeats_combine(self):
        s = self._strat()
        self.assertIs(
            s.evaluate(group_size=20, triangle_count=100), StrategyType.COMBINE
        )
        self.assertIs(
            s.evaluate(group_size=3, triangle_count=100), StrategyType.COMBINE
        )

    def test_lone_micro_keeps_separate(self):
        s = self._strat()
        self.assertIs(
            s.evaluate(group_size=1, triangle_count=100), StrategyType.KEEP_SEPARATE
        )

    def test_static_no_gpu_instancing_combines(self):
        s = self._strat(can_gpu_instance=False)
        self.assertIs(
            s.evaluate(group_size=10, triangle_count=2000), StrategyType.COMBINE
        )

    def test_worth_instancing_threshold_standard(self):
        s = self._strat()
        self.assertIs(
            s.evaluate(group_size=10, triangle_count=800), StrategyType.GPU_INSTANCE
        )
        self.assertIs(
            s.evaluate(group_size=10, triangle_count=799), StrategyType.COMBINE
        )
        self.assertIs(
            s.evaluate(group_size=9, triangle_count=800), StrategyType.COMBINE
        )

    def test_worth_instancing_threshold_lightmap_stricter(self):
        s = self._strat(will_be_lightmapped=True)
        self.assertIs(
            s.evaluate(group_size=10, triangle_count=900), StrategyType.COMBINE
        )
        self.assertIs(
            s.evaluate(group_size=10, triangle_count=1500), StrategyType.GPU_INSTANCE
        )

    def test_heavy_mesh_exception(self):
        s = self._strat()
        self.assertIs(
            s.evaluate(group_size=3, triangle_count=5000), StrategyType.GPU_INSTANCE
        )
        self.assertIs(
            s.evaluate(group_size=3, triangle_count=4999), StrategyType.COMBINE
        )
        self.assertIs(
            s.evaluate(group_size=2, triangle_count=9000), StrategyType.COMBINE
        )

    def test_default_fallback_combine(self):
        s = self._strat()
        self.assertIs(
            s.evaluate(group_size=4, triangle_count=600), StrategyType.COMBINE
        )

    def test_thresholds_are_overridable_per_instance(self):
        s = self._strat()
        s.MICRO_TRI_THRESHOLD = 50
        self.assertIs(
            s.evaluate(group_size=1, triangle_count=100), StrategyType.COMBINE
        )
        self.assertEqual(InstancingStrategy.MICRO_TRI_THRESHOLD, 300)


class TestTriangleCountHook(unittest.TestCase):
    def test_hook_supplies_the_count_from_mesh_node(self):
        s = _CountingStrategy(StrategyConfig())
        self.assertIs(
            s.evaluate(group_size=10, mesh_node=800), StrategyType.GPU_INSTANCE
        )
        self.assertIs(
            s.evaluate(group_size=1, mesh_node=100), StrategyType.KEEP_SEPARATE
        )

    def test_explicit_triangle_count_wins_over_mesh_node(self):
        s = _CountingStrategy(StrategyConfig())
        self.assertIs(
            s.evaluate(group_size=10, mesh_node=10, triangle_count=5000),
            StrategyType.GPU_INSTANCE,
        )

    def test_no_mesh_node_counts_zero_triangles(self):
        s = _CountingStrategy(StrategyConfig())
        # 0 tris is micro: a repeated group combines, a lone one stays.
        self.assertIs(s.evaluate(group_size=5), StrategyType.COMBINE)
        self.assertIs(s.evaluate(group_size=1), StrategyType.KEEP_SEPARATE)

    def test_base_reads_no_scene(self):
        s = InstancingStrategy(StrategyConfig())
        self.assertEqual(s._get_triangle_count(object()), 0)

    def test_root_registration(self):
        self.assertIs(ptk.InstancingStrategy, InstancingStrategy)


if __name__ == "__main__":
    unittest.main()
