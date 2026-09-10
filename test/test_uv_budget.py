# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.geo_utils.uv_budget (UvBudget — texture-budget planning).

Pure arithmetic with no optional dependency, so everything here runs
unconditionally. The properties worth pinning are the ones a caller reasons
with: the solvers agree in both directions, demand rises with density, the
approximation bounds hold, and an impossible request reports rather than
returning a plausible-looking wrong answer.
"""

import unittest

import pythontk as ptk


def items(n=8, area=1.0, perimeter=4.0, charts=1):
    return [
        ptk.BudgetItem(key=f"i{i}", area=area, perimeter=perimeter, charts=charts)
        for i in range(n)
    ]


class TestDemandModel(unittest.TestCase):
    def test_demand_terms(self):
        """Each term of the polynomial contributes as documented."""
        it = ptk.BudgetItem(key="a", area=2.0, perimeter=3.0, charts=5)
        # area*tpu^2 + (1-share)*pad*perimeter*tpu + 4*pad^2*charts
        self.assertAlmostEqual(it.demand(10.0, 4.0), 2 * 100 + 4 * 3 * 10 + 4 * 16 * 5)

    def test_per_item_density_scales_demand_quadratically(self):
        coarse = ptk.BudgetItem(key="coarse", area=4.0, density=10.0)
        fine = ptk.BudgetItem(key="fine", area=4.0, density=100.0)
        self.assertAlmostEqual(fine.demand(1.0, 0.0) / coarse.demand(1.0, 0.0), 100.0)

    def test_the_solve_argument_scales_every_item_together(self):
        """With per-item densities, `density` is a uniform scale, not a value."""
        pool = [
            ptk.BudgetItem(key="coarse", area=4.0, density=10.0),
            ptk.BudgetItem(key="fine", area=4.0, density=100.0),
        ]
        flat = [ptk.BudgetItem(key=i.key, area=i.area) for i in pool]
        # Scale 1.0 on authored densities must cost more than a flat 1.0.
        pinned = ptk.UvBudget.pages_at_density(pool, 1.0, 256, fill=1.0)
        plain = ptk.UvBudget.pages_at_density(flat, 1.0, 256, fill=1.0)
        self.assertGreater(
            sum(p.demand for p in pinned.page_list),
            sum(p.demand for p in plain.page_list),
        )

    def test_preserved_scale_round_trips(self):
        """Solving for pages at authored densities, then back, returns a scale."""
        pool = [
            ptk.BudgetItem(
                key=f"g{i}", area=2.0, perimeter=6.0, charts=3, density=10.0 * (i + 1)
            )
            for i in range(6)
        ]
        fwd = ptk.UvBudget.pages_at_density(pool, 1.0, 2048)
        self.assertTrue(fwd.feasible)
        inv = ptk.UvBudget.density_at_pages(pool, fwd.pages, 2048)
        self.assertGreaterEqual(inv.density, 1.0 - 1e-6)

    def test_demand_rises_with_density(self):
        it = ptk.BudgetItem(key="a", area=1.0, perimeter=4.0, charts=2)
        values = [it.demand(t, 8.0) for t in (1, 10, 100, 1000)]
        self.assertEqual(values, sorted(values))

    def test_charts_cost_even_at_zero_density(self):
        """Gutter corners are a fixed cost, which is why chart count matters."""
        one = ptk.BudgetItem(key="a", area=1.0, charts=1)
        many = ptk.BudgetItem(key="b", area=1.0, charts=400)
        self.assertGreater(many.demand(0.0, 8.0), one.demand(0.0, 8.0))


class TestPadding(unittest.TestCase):
    def test_ratio_rule_matches_the_ecosystem_primitive(self):
        for size in (512, 1024, 2048, 4096):
            pad, floored = ptk.UvBudget.padding_for(size)
            self.assertEqual(pad, ptk.MathUtils.calculate_uv_padding(size))
            self.assertFalse(floored)

    def test_mip_floor_binds_on_small_maps(self):
        pad, floored = ptk.UvBudget.padding_for(512, mip_levels=6)
        self.assertEqual(pad, 64.0)
        self.assertTrue(floored)

    def test_mip_floor_yields_to_the_ratio_when_larger(self):
        pad, floored = ptk.UvBudget.padding_for(8192, mip_levels=2)
        self.assertEqual(pad, ptk.MathUtils.calculate_uv_padding(8192))
        self.assertFalse(floored)


class TestBinPacking(unittest.TestCase):
    def test_ffd_places_everything_within_capacity(self):
        sizes = [(f"k{i}", 10.0 + i) for i in range(20)]
        pages = ptk.UvBudget.first_fit_decreasing(sizes, 100.0)
        by_key = dict(sizes)
        self.assertEqual(
            sorted(k for p in pages for k in p), sorted(k for k, _ in sizes)
        )
        for page in pages:
            self.assertLessEqual(sum(by_key[k] for k in page), 100.0 + 1e-9)

    def test_ffd_refuses_an_item_larger_than_a_page(self):
        self.assertIsNone(ptk.UvBudget.first_fit_decreasing([("big", 200.0)], 100.0))

    def test_ffd_beats_the_worst_case_bound(self):
        sizes = [(f"k{i}", v) for i, v in enumerate([50, 40, 30, 20, 10] * 4)]
        pages = ptk.UvBudget.first_fit_decreasing(sizes, 100.0)
        optimal = sum(v for _, v in sizes) / 100.0
        self.assertLessEqual(len(pages), 11 / 9 * optimal + 6 / 9)

    def test_lpt_uses_every_page_and_levels_them(self):
        sizes = [(f"k{i}", float(i + 1)) for i in range(12)]
        buckets = ptk.UvBudget.partition_lpt(sizes, 3)
        self.assertEqual(len(buckets), 3)
        by_key = dict(sizes)
        loads = [sum(by_key[k] for k in b) for b in buckets]
        # LPT's guarantee is on the fullest bucket, not on perfect balance.
        self.assertLessEqual(max(loads), (4 / 3 - 1 / 9) * (sum(loads) / 3))


class TestSolveDirections(unittest.TestCase):
    def test_pages_at_density_is_monotone(self):
        pool = items(30, area=2.0)
        counts = [
            ptk.UvBudget.pages_at_density(pool, d, 1024).pages
            for d in (10, 40, 80, 160)
        ]
        self.assertEqual(counts, sorted(counts))

    def test_density_at_pages_is_monotone(self):
        pool = items(30, area=2.0)
        found = [
            ptk.UvBudget.density_at_pages(pool, p, 1024).density for p in range(1, 6)
        ]
        self.assertEqual(found, sorted(found))

    def test_directions_round_trip(self):
        """Solving for pages, then back for density, never loses ground."""
        pool = items(24, area=1.5, perimeter=5.0, charts=3)
        fwd = ptk.UvBudget.pages_at_density(pool, 50.0, 1024)
        inv = ptk.UvBudget.density_at_pages(pool, fwd.pages, 1024)
        self.assertGreaterEqual(inv.density, 50.0 - 1e-6)

    def test_solved_density_actually_fits(self):
        pool = items(24, area=1.5, perimeter=5.0, charts=3)
        row = ptk.UvBudget.density_at_pages(pool, 3, 1024)
        self.assertLessEqual(row.pages, 3)
        self.assertTrue(row.feasible)

    def test_every_item_is_placed_exactly_once(self):
        pool = items(17, area=0.7)
        row = ptk.UvBudget.pages_at_density(pool, 30.0, 1024)
        placed = [k for p in row.page_list for k in p.keys]
        self.assertEqual(sorted(placed), sorted(i.key for i in pool))
        self.assertEqual(len(placed), len(set(placed)))

    def test_fill_scales_capacity(self):
        pool = items(20, area=3.0)
        loose = ptk.UvBudget.density_at_pages(pool, 2, 1024, fill=1.0).density
        tight = ptk.UvBudget.density_at_pages(pool, 2, 1024, fill=0.5).density
        self.assertGreater(loose, tight)


class TestInfeasible(unittest.TestCase):
    def test_one_oversized_item_reports_rather_than_paginates(self):
        row = ptk.UvBudget.pages_at_density(
            [ptk.BudgetItem(key="huge", area=1e9)], 500.0, 1024
        )
        self.assertFalse(row.feasible)
        self.assertIn("huge", row.note)

    def test_gutters_alone_can_exhaust_the_budget(self):
        pool = [ptk.BudgetItem(key=f"c{i}", area=1e-9, charts=5000) for i in range(50)]
        row = ptk.UvBudget.density_at_pages(pool, 1, 1024)
        self.assertFalse(row.feasible)
        self.assertIn("gutters", row.note)

    def test_no_area_is_not_a_plan(self):
        row = ptk.UvBudget.density_at_pages([], 2, 1024)
        self.assertFalse(row.feasible)


class TestRowMetrics(unittest.TestCase):
    def test_worst_fill_catches_a_near_empty_page(self):
        """Two items that cannot share a page, one of them small."""
        # At 100 px/unit on a 1024 page (capacity 1024^2 * 0.65), these price
        # at ~95% and ~40% of a page: FFD must use two, and the second is the
        # wasted-map case worth surfacing.
        pool = [
            ptk.BudgetItem(key="big", area=64.7, perimeter=0.0),
            ptk.BudgetItem(key="crumb", area=27.2, perimeter=0.0),
        ]
        row = ptk.UvBudget.pages_at_density(pool, 100.0, 1024)
        self.assertTrue(row.feasible)
        self.assertEqual(row.pages, 2)
        self.assertLess(row.worst_fill, 0.5)
        self.assertEqual(row.underfilled(), [1])

    def test_leveling_keeps_the_page_count(self):
        pool = items(21, area=2.0)
        plain = ptk.UvBudget.pages_at_density(pool, 60.0, 1024, level=False)
        even = ptk.UvBudget.pages_at_density(pool, 60.0, 1024, level=True)
        self.assertEqual(plain.pages, even.pages)
        self.assertGreaterEqual(even.worst_fill, plain.worst_fill)

    def test_levelling_never_overflows_a_page(self):
        """LPT minimizes the fullest page, which is not the same as fitting.

        5,5,4,4,3,3,3 fits three pages of 9 exactly, while LPT puts 5+3+3 on
        one and overflows to 11. Levelling is a preference, so a layout that
        does not fit must be discarded rather than reported as feasible.
        """
        # Areas chosen so demand at tpu=1 with no gutter is the raw number.
        pool = [
            ptk.BudgetItem(key=f"j{i}", area=float(v), perimeter=0.0, charts=0)
            for i, v in enumerate([5, 5, 4, 4, 3, 3, 3])
        ]
        capacity = 9.0
        size = int(round((capacity / 1.0) ** 0.5))
        row = ptk.UvBudget.pages_at_density(
            pool, 1.0, size, fill=capacity / (size * size), level=True
        )
        self.assertTrue(row.feasible)
        self.assertEqual(row.pages, 3)
        for page in row.page_list:
            self.assertLessEqual(page.fill, 1.0 + 1e-9)
        self.assertIn("uneven", row.note)

    def test_levelling_applies_when_it_fits(self):
        pool = items(21, area=2.0)
        plain = ptk.UvBudget.pages_at_density(pool, 60.0, 1024, level=False)
        even = ptk.UvBudget.pages_at_density(pool, 60.0, 1024, level=True)
        self.assertEqual(even.note, "")
        self.assertGreaterEqual(even.worst_fill, plain.worst_fill)

    def test_payload_does_not_decide_identity(self):
        """An opaque payload must not make a frozen value type unhashable."""
        a = ptk.BudgetItem(key="a", area=1.0, payload={"anything": 1})
        b = ptk.BudgetItem(key="a", area=1.0, payload=None)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)

    def test_a_row_renders_its_own_currency(self):
        """A scaled solve must not print itself as an absolute density."""
        flat = ptk.UvBudget.pages_at_density(items(6, area=1.0), 40.0, 1024)
        self.assertFalse(flat.scaled)
        self.assertIn("px/unit", str(flat))

        pinned = [ptk.BudgetItem(key=f"g{i}", area=1.0, density=40.0) for i in range(6)]
        scaled = ptk.UvBudget.pages_at_density(pinned, 1.0, 1024)
        self.assertTrue(scaled.scaled)
        self.assertIn("authored", str(scaled))
        self.assertNotIn("px/unit", str(scaled))

    def test_texels_counts_every_page(self):
        row = ptk.UvBudget.pages_at_density(items(9), 40.0, 512)
        self.assertEqual(row.texels, row.pages * 512 * 512)


class TestPlan(unittest.TestCase):
    def test_requires_exactly_one_direction(self):
        pool = items(5)
        with self.assertRaises(ValueError):
            ptk.UvBudget.plan(pool)
        with self.assertRaises(ValueError):
            ptk.UvBudget.plan(pool, density=10.0, pages=2)

    def test_rejects_a_zero_page_budget(self):
        """Feeding an infeasible row's page count back must error, not plan."""
        with self.assertRaises(ValueError):
            ptk.UvBudget.plan(items(5), pages=0)

    def test_alternates_are_feasible_and_distinct(self):
        plan = ptk.UvBudget.plan(items(30, area=2.0), map_size=1024, density=40.0)
        self.assertTrue(plan.chosen.feasible)
        self.assertTrue(plan.alternates)
        for row in plan.alternates:
            self.assertTrue(row.feasible)
        configs = [(r.map_size, r.pages) for r in plan.rows]
        self.assertEqual(len(set(configs)), len(configs))

    def test_an_alternate_never_repeats_the_chosen_configuration(self):
        """Asking for one page more often re-solves to the same count."""
        pool = items(9, area=1.0, perimeter=4.0, charts=2)
        plan = ptk.UvBudget.plan(pool, map_size=4096, density=40.0)
        chosen = (plan.chosen.map_size, plan.chosen.pages)
        self.assertNotIn(chosen, [(r.map_size, r.pages) for r in plan.alternates])

    def test_rows_order_by_texels_spent(self):
        plan = ptk.UvBudget.plan(items(30, area=2.0), map_size=1024, density=40.0)
        spent = [r.texels for r in plan.rows]
        self.assertEqual(spent, sorted(spent))

    def test_density_ratio_reports_against_the_target(self):
        plan = ptk.UvBudget.plan(items(12), map_size=1024, density=25.0)
        self.assertAlmostEqual(plan.density_ratio, 1.0)
        self.assertIsNone(
            ptk.UvBudget.plan(items(12), map_size=1024, pages=2).density_ratio
        )

    def test_alternates_can_be_skipped(self):
        plan = ptk.UvBudget.plan(
            items(12), map_size=1024, density=25.0, alternates=False
        )
        self.assertEqual(plan.alternates, [])


class TestDeterminism(unittest.TestCase):
    def test_same_input_plans_identically(self):
        pool = items(40, area=1.3, perimeter=6.0, charts=4)
        a = ptk.UvBudget.plan(pool, map_size=2048, density=35.0)
        b = ptk.UvBudget.plan(pool, map_size=2048, density=35.0)
        self.assertEqual(a.chosen.assignment, b.chosen.assignment)
        self.assertEqual([str(r) for r in a.alternates], [str(r) for r in b.alternates])


if __name__ == "__main__":
    unittest.main()
