# !/usr/bin/python
# coding=utf-8
"""Tests for ``LodChain``: the one LOD convention every carrier writes.

A LOD group is a node ``<base>`` whose children ``<base>_LOD0..n`` are its
levels, most detailed first (the ``_LOD<n>`` naming engines' model
importer reads; LOD0 the most detailed in every source). Switch points are
screen-relative heights, one per level, the last the cull height.
"""

import unittest

from pythontk.file_utils.lod_chain import LodChain


def _node(path, triangles=None):
    parent, _, _ = path.rpartition("|")
    return {"path": path, "parent": parent or None, "triangles": triangles}


class ParseTest(unittest.TestCase):
    def test_a_level_name_splits_into_base_and_index(self):
        self.assertEqual(LodChain.parse("chair_LOD0"), ("chair", 0))
        self.assertEqual(LodChain.parse("chair_LOD12"), ("chair", 12))
        self.assertEqual(LodChain.parse("Big_Chair_LOD2"), ("Big_Chair", 2))

    def test_the_suffix_reads_in_any_case(self):
        """Epic's tools strip ``_lod0`` as well as ``_LOD0``; read both, write ``_LOD``."""
        self.assertEqual(LodChain.parse("chair_lod1"), ("chair", 1))
        self.assertEqual(LodChain.parse("chair_Lod3"), ("chair", 3))

    def test_a_name_without_a_level_number_is_no_level(self):
        for name in ("chair", "chair_LOD", "chair_LODx", "LOD0", "_LOD0", "chairLOD0", "chair_LOD0_geo"):
            self.assertIsNone(LodChain.parse(name), name)

    def test_a_path_parses_by_its_leaf(self):
        self.assertEqual(LodChain.parse("|props|chair|chair_LOD1"), ("chair", 1))
        self.assertEqual(LodChain.parse("ns:chair_LOD1"), ("ns:chair", 1))

    def test_a_container_names_its_level(self):
        """DCCs' LOD generators hold each level in
        a ``LOD_<n>`` node; USD's variants are ``LOD<n>``."""
        self.assertEqual(LodChain.parse_container("LOD_0"), 0)
        self.assertEqual(LodChain.parse_container("LOD_2"), 2)
        self.assertEqual(LodChain.parse_container("LOD3"), 3)
        self.assertEqual(LodChain.parse_container("lod_1"), 1)
        self.assertEqual(LodChain.parse_container("|LOD_Group_1|LOD_1"), 1)
        for name in ("LOD", "LOD_", "chair_LOD0", "LOD_Group_1", "xLOD_1"):
            self.assertIsNone(LodChain.parse_container(name), name)

    def test_a_level_is_named_with_the_upper_case_suffix(self):
        self.assertEqual(LodChain.level_name("chair", 0), "chair_LOD0")
        self.assertEqual(LodChain.level_name("chair", 7), "chair_LOD7")

    def test_a_level_index_past_the_limit_is_refused(self):
        with self.assertRaises(ValueError):
            LodChain.level_name("chair", LodChain.MAX_LEVELS)
        with self.assertRaises(ValueError):
            LodChain.level_name("chair", -1)


class GroupsTest(unittest.TestCase):
    def test_siblings_sharing_a_base_form_one_group(self):
        nodes = [_node(p) for p in ("|chair", "|chair|chair_LOD1", "|chair|chair_LOD0", "|chair|chair_LOD2", "|lamp")]
        groups = LodChain.groups(nodes)
        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group["base"], "chair")
        self.assertEqual(group["parent"], "|chair")
        self.assertEqual(group["form"], "names")
        self.assertEqual(group["levels"], ["|chair|chair_LOD0", "|chair|chair_LOD1", "|chair|chair_LOD2"])
        self.assertEqual(group["issues"], [])

    def test_groups_are_kept_apart_by_parent_and_base(self):
        """An engine's FBX import merges every ``_LOD`` mesh in one file into one group
        (measured 2026-11-10); the convention keeps each group to its parent."""
        nodes = [
            _node(p)
            for p in (
                "|a|stool_LOD0",
                "|a|stool_LOD1",
                "|b|stool_LOD0",
                "|b|stool_LOD1",
                "|a|table_LOD0",
                "|a|table_LOD1",
            )
        ]
        groups = LodChain.groups(nodes)
        self.assertEqual(
            [(g["parent"], g["base"]) for g in groups],
            [("|a", "stool"), ("|a", "table"), ("|b", "stool")],
        )
        self.assertIn("holds other nodes", " ".join(groups[0]["issues"]), "two groups under one node")
        self.assertEqual(groups[2]["issues"], [])

    def test_a_base_groups_without_regard_to_the_suffix_case(self):
        groups = LodChain.groups([_node("|g|chair_LOD0"), _node("|g|chair_lod1")])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["levels"], ["|g|chair_LOD0", "|g|chair_lod1"])

    def test_a_lone_level_is_no_group(self):
        """One ``_LOD0`` beside nothing is a plain mesh with a suffix: it ships as
        authored and is reported, never grouped."""
        self.assertEqual(LodChain.groups([_node("|prop_LOD0"), _node("|lamp")]), [])
        self.assertEqual(LodChain.lone_levels([_node("|prop_LOD0"), _node("|lamp")]), ["|prop_LOD0"])

    def test_a_missing_lod0_or_a_gap_is_an_issue(self):
        groups = LodChain.groups([_node("|g|chair_LOD1"), _node("|g|chair_LOD2")])
        self.assertEqual(groups[0]["levels"], [None, "|g|chair_LOD1", "|g|chair_LOD2"])
        self.assertIn("no LOD0", " ".join(groups[0]["issues"]))
        groups = LodChain.groups([_node("|g|chair_LOD0"), _node("|g|chair_LOD2")])
        self.assertIn("LOD1 is missing", " ".join(groups[0]["issues"]))

    def test_two_nodes_on_one_level_is_an_issue(self):
        groups = LodChain.groups([_node("|g|chair_LOD0"), _node("|g|chair_lod0"), _node("|g|chair_LOD1")])
        self.assertIn("LOD0 is named twice", " ".join(groups[0]["issues"]))

    def test_more_levels_than_the_limit_is_an_issue(self):
        nodes = [_node(f"|g|chair_LOD{i}") for i in range(LodChain.MAX_LEVELS + 1)]
        self.assertIn(f"{LodChain.MAX_LEVELS}-level limit", " ".join(LodChain.groups(nodes)[0]["issues"]))

    def test_triangles_must_fall_level_by_level(self):
        nodes = [_node("|g|chair_LOD0", 1000), _node("|g|chair_LOD1", 400), _node("|g|chair_LOD2", 400)]
        self.assertIn("LOD2 has no fewer triangles than LOD1", " ".join(LodChain.groups(nodes)[0]["issues"]))
        nodes = [_node("|g|chair_LOD0", 1000), _node("|g|chair_LOD1", 400), _node("|g|chair_LOD2", None)]
        self.assertEqual(LodChain.groups(nodes)[0]["issues"], [], "an unknown count is not judged")

    def test_level_containers_form_a_group_of_their_parent(self):
        """A DCC's own LOD generator: ``LOD_Group_1|LOD_0|wheel``,
        ``LOD_1|wheel``. Read, and named as the form engines do not read."""
        nodes = [_node(p) for p in ("|LOD_Group_1|LOD_0", "|LOD_Group_1|LOD_1", "|LOD_Group_1|LOD_0|wheel", "|LOD_Group_1|LOD_1|wheel")]
        groups = LodChain.groups(nodes)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["form"], "containers")
        self.assertEqual(groups[0]["base"], "LOD_Group_1")
        self.assertEqual(groups[0]["levels"], ["|LOD_Group_1|LOD_0", "|LOD_Group_1|LOD_1"])
        self.assertEqual(groups[0]["issues"], [], "a valid group in another form")
        self.assertIn("game engines do not read", " ".join(groups[0]["notes"]))

    def test_a_group_needs_a_node_of_its_own(self):
        """FBX's LodGroup sits on the node whose children are the levels, USD's
        variant set on that prim, and an engine's LOD group on that object: levels
        sharing their parent with other nodes, or at the scene root, have none."""
        shared = LodChain.groups([_node("|props|chair_LOD0"), _node("|props|chair_LOD1"), _node("|props|lamp")])
        self.assertIn("holds other nodes", " ".join(shared[0]["issues"]))
        rooted = LodChain.groups([_node("|chair_LOD0"), _node("|chair_LOD1")])
        self.assertIn("scene root", " ".join(rooted[0]["issues"]))

    def test_the_level_paths_of_a_group_are_listed_most_detailed_first(self):
        nodes = [_node(p) for p in ("|g|chair_LOD0", "|g|chair_LOD1", "|g|chair_LOD2")]
        self.assertEqual(LodChain.lower_levels(LodChain.groups(nodes)), ["|g|chair_LOD1", "|g|chair_LOD2"])

    def test_the_result_does_not_depend_on_input_order(self):
        paths = ["|b|x_LOD1", "|a|y_LOD0", "|b|x_LOD0", "|a|y_LOD1"]
        self.assertEqual(LodChain.groups([_node(p) for p in paths]), LodChain.groups([_node(p) for p in reversed(paths)]))


class PlanTest(unittest.TestCase):
    """``plan``: what an export ships of the LOD groups -- the one decision
    both hosts' Scene Exporters apply (each only lists its nodes)."""

    def _nodes(self):
        paths = ("|props", "|props|chair", "|props|chair|chair_LOD0", "|props|chair|chair_LOD1",
                 "|props|chair|chair_LOD2", "|props|lamp", "|bad", "|bad|x_LOD1", "|bad|x_LOD2", "|prop_LOD0")
        return [_node(p) for p in paths]

    def test_as_authored_drops_nothing_and_reports_every_group(self):
        plan = LodChain.plan(self._nodes(), "authored")
        self.assertEqual([g["base"] for g in plan["groups"]], ["chair"])
        self.assertEqual(plan["drop"], [])
        self.assertEqual(plan["listing"], ["chair: 3 levels"])
        self.assertTrue(any("x: no LOD0" in w for w in plan["warnings"]), plan["warnings"])
        self.assertTrue(any("prop_LOD0" in n for n in plan["notes"]), plan["notes"])

    def test_most_detailed_only_drops_every_lower_level_of_a_valid_group(self):
        plan = LodChain.plan(self._nodes(), "lod0")
        self.assertEqual(plan["drop"], ["|props|chair|chair_LOD1", "|props|chair|chair_LOD2"])

    def test_no_level_name_no_plan_work(self):
        plan = LodChain.plan([_node("|a"), _node("|a|b")], "lod0")
        self.assertEqual((plan["groups"], plan["drop"], plan["warnings"], plan["notes"]), ([], [], [], []))

    def test_generate_missing_ships_the_authored_groups_as_authored(self):
        plan = LodChain.plan(self._nodes(), "generate")
        self.assertEqual(([g["base"] for g in plan["groups"]], plan["drop"]), (["chair"], []))

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            LodChain.plan([], "bogus")

    def test_names_nothing_reads_as_no_level(self):
        """The cheap gate a host runs before it counts a triangle."""
        self.assertTrue(LodChain.names_levels(["|a", "|a|chair_LOD0"]))
        self.assertTrue(LodChain.names_levels(["|g|LOD_1"]))
        self.assertFalse(LodChain.names_levels(["|a", "|a|chair", "|LOD_Group_1"]))


class VariantPlanTest(unittest.TestCase):
    def test_each_valid_group_becomes_a_lod_variant_set(self):
        nodes = [_node(p) for p in ("|props|chair", "|props|chair|chair_LOD0", "|props|chair|chair_LOD1",
                                    "|bad", "|bad|x_LOD1", "|bad|x_LOD2")]
        plan = LodChain.variant_plan(LodChain.groups(nodes))
        self.assertEqual(
            plan,
            [{"group": "chair", "levels": ["chair_LOD0", "chair_LOD1"], "variants": ["LOD0", "LOD1"]}],
        )
        self.assertEqual(LodChain.USD_VARIANT_SET, "LOD")


class RefineSelectionTest(unittest.TestCase):
    """``refine_selection``: a writer that carries each selected node's whole
    subtree (a DCC's FBX and USD export) ships everything selected but the
    dropped levels once a selected node above one is replaced by its children.
    Measured 2026-11-10: a DCC's FBX export of a selected leaf writes its
    ancestors (a lodGroup's attribute included) and not its siblings."""

    CHILDREN = {
        "|props": ["|props|chair", "|props|lamp"],
        "|props|chair": ["|props|chair|chair_LOD0", "|props|chair|chair_LOD1", "|props|chair|chair_LOD2"],
        "|props|chair|chair_LOD0": ["|props|chair|chair_LOD0|chair_LOD0Shape"],
    }
    DROP = ["|props|chair|chair_LOD1", "|props|chair|chair_LOD2"]

    def test_a_selected_root_is_replaced_down_to_the_kept_levels(self):
        self.assertEqual(
            LodChain.refine_selection(["|props"], self.DROP, self.CHILDREN),
            ["|props|chair|chair_LOD0", "|props|lamp"],
        )

    def test_a_selected_dropped_level_leaves_and_a_kept_one_stays(self):
        self.assertEqual(
            LodChain.refine_selection(
                ["|props|chair|chair_LOD1", "|props|chair|chair_LOD0", "|other"], self.DROP, self.CHILDREN
            ),
            ["|props|chair|chair_LOD0", "|other"],
        )

    def test_nothing_to_drop_leaves_the_selection_as_it_is(self):
        self.assertEqual(LodChain.refine_selection(["|props", "|props"], [], self.CHILDREN), ["|props"])


class LadderTest(unittest.TestCase):
    """``ladder``: the triangle target of each generated level -- half the last
    (engines' generated mesh LODs' own step), stopping before a level would fall below the
    floor, at most ``levels`` levels in all."""

    def test_each_level_halves_the_last(self):
        self.assertEqual(LodChain.ladder(4000), [2000, 1000])

    def test_the_floor_stops_the_ladder(self):
        self.assertEqual(LodChain.ladder(200, levels=4, floor=64), [100])
        self.assertEqual(LodChain.ladder(100, floor=64), [], "a mesh too light for a level")

    def test_ratio_and_levels_are_parameters(self):
        self.assertEqual(LodChain.ladder(10000, levels=4, ratio=0.25), [2500, 625, 156])
        with self.assertRaises(ValueError):
            LodChain.ladder(1000, ratio=1.0)
        with self.assertRaises(ValueError):
            LodChain.ladder(1000, levels=LodChain.MAX_LEVELS + 1)


class HeightsTest(unittest.TestCase):
    def test_the_default_ladder_halves_and_ends_at_the_cull(self):
        self.assertEqual(LodChain.default_heights(3), [0.5, 0.25, 0.0])
        self.assertEqual(LodChain.default_heights(3, cull=0.01), [0.5, 0.25, 0.01])
        self.assertEqual(LodChain.default_heights(1), [0.0])

    def test_heights_must_fall_and_stay_in_range(self):
        LodChain.check_heights([0.5, 0.25, 0.0])
        for bad in ([0.25, 0.5, 0.0], [1.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.1], []):
            with self.assertRaises(ValueError, msg=bad):
                LodChain.check_heights(bad)

    def test_fbx_states_the_transitions_in_percent(self):
        """FbxLODGroup: one threshold fewer than its children, in percent when
        ``ThresholdsUsedAsPercentage`` (a DCC writer's default, measured)."""
        self.assertEqual(LodChain.to_fbx_thresholds([0.5, 0.2, 0.0]), [50.0, 20.0])
        self.assertEqual(LodChain.from_fbx_thresholds([50.0, 20.0]), [0.5, 0.2, 0.0])
        self.assertEqual(LodChain.from_fbx_thresholds([50.0, 20.0], cull=0.01), [0.5, 0.2, 0.01])

    def test_kept_levels_keep_their_own_switch_and_the_groups_cull(self):
        """A write that ships some levels of a group: each kept level but the
        last switches where it did, and the last kept draws down to the
        group's cull -- a dropped middle level's range goes to the next kept
        one."""
        heights = [0.6, 0.3, 0.1, 0.01]
        self.assertEqual(LodChain.kept_heights(heights, [0, 1, 2, 3]), heights)
        self.assertEqual(LodChain.kept_heights(heights, [0, 1]), [0.6, 0.01])
        self.assertEqual(LodChain.kept_heights(heights, [0, 2]), [0.6, 0.01])
        self.assertEqual(LodChain.kept_heights(heights, [0, 1, 3]), [0.6, 0.3, 0.01])


class BoundsTest(unittest.TestCase):
    """What a name states is bounded before anything is sized by it: the
    reading runs on every export and every FBX-to-GLB conversion."""

    def test_a_level_number_past_the_limit_is_one_issue_read_at_once(self):
        import time

        nodes = [_node(p) for p in ("|x", "|x|x_LOD0", "|x|x_LOD20000000")]
        start = time.perf_counter()
        (group,) = LodChain.groups(nodes)
        self.assertLess(time.perf_counter() - start, 1.0)
        self.assertEqual(len(group["issues"]), 1, group["issues"])
        self.assertIn("20000000", group["issues"][0])

    def test_numbered_names_that_are_no_levels_say_so_once(self):
        """``Wall_LOD200`` / ``Wall_LOD300`` (a building model's level of
        development) is one issue, not one per missing level."""
        nodes = [_node(p) for p in ("|w", "|w|Wall_LOD200", "|w|Wall_LOD300")]
        (group,) = LodChain.groups(nodes)
        self.assertEqual(len(group["issues"]), 1, group["issues"])

    def test_containers_at_the_scene_root_name_what_they_are(self):
        nodes = [_node(p) for p in ("|LOD_0", "|LOD_1")]
        (group,) = LodChain.groups(nodes)
        self.assertFalse(any(issue.startswith(":") for issue in group["issues"]), group["issues"])


class RecordGroupsTest(unittest.TestCase):
    """``record_groups``: the ``lods`` record's entries both hosts publish."""

    def test_each_shipped_group_with_its_kept_levels_and_switches(self):
        nodes = [_node(p, t) for p, t in (("|seat", None), ("|seat|seat_LOD0", 90), ("|seat|seat_LOD1", 40), ("|seat|seat_LOD2", 9),
                                          ("|lamp", None), ("|lamp|lamp_LOD0", 9), ("|lamp|lamp_LOD1", 3))]
        entries = LodChain.record_groups(
            LodChain.groups(nodes),
            ships=lambda name: name != "seat_LOD1" and not name.startswith("lamp_LOD1"),
            heights_of=lambda path, count: [0.6, 0.3, 0.01][:count - 1] + [0.01] if count == 3 else [0.5, 0.0],
        )
        self.assertEqual(entries, [{"name": "seat", "levels": ["seat_LOD0", "seat_LOD2"], "heights": [0.6, 0.01]}])


class CostTest(unittest.TestCase):
    """The reading runs on every export: thousands of groups stay linear
    (2026-11-10 review: 3000 groups took 8 s to refine, 6000 under one
    parent 12 s to plan)."""

    def test_thousands_of_groups_read_and_refine_at_once(self):
        import time

        nodes, selected, drop = [_node("|props")], [], []
        for i in range(3000):
            group = f"|props|g{i}"
            nodes.append(_node(group))
            selected.append(group)
            for k in range(3):
                nodes.append(_node(f"{group}|g{i}_LOD{k}", 100 - k))
                if k:
                    drop.append(f"{group}|g{i}_LOD{k}")
        children = {g: [f"{g}|{g.rsplit('|', 1)[-1]}_LOD{k}" for k in range(3)] for g in selected}
        flat = [_node("|p")] + [_node(f"|p|g{i}_LOD{k}", 9 - k) for i in range(6000) for k in range(2)]
        start = time.perf_counter()
        kept = LodChain.refine_selection(selected, drop, children)
        LodChain.plan(nodes, "lod0")
        LodChain.plan(flat)
        self.assertLess(time.perf_counter() - start, 2.0)
        self.assertEqual(len(kept), 3000)


class LevelBoundsTest(unittest.TestCase):
    """A level whose world bounds stray from LOD0's pops at its switch (it
    moved, or lost or gained a part): said as a note, never a block -- the
    group still ships as authored."""

    @staticmethod
    def _group(*bounds):
        nodes = [_node("|seat")]
        for index, box in enumerate(bounds):
            node = _node(f"|seat|seat_LOD{index}", 100 - index)
            node["bounds"] = box
            nodes.append(node)
        (group,) = LodChain.groups(nodes)
        return group

    def test_levels_that_agree_say_nothing(self):
        group = self._group(((0, 0, 0), (1, 1, 1)), ((0.01, 0, 0), (0.99, 1, 1)))
        self.assertEqual((group["issues"], group["notes"]), ([], []))

    def test_a_level_that_moved_is_a_note_naming_it(self):
        group = self._group(((0, 0, 0), (1, 1, 1)), ((0, 0, 0), (1, 1, 1)), ((2, 0, 0), (3, 1, 1)))
        self.assertEqual(group["issues"], [])
        (note,) = group["notes"]
        self.assertIn("LOD2", note)
        self.assertNotIn("LOD1", note)

    def test_a_level_that_lost_a_part_is_a_note(self):
        group = self._group(((0, 0, 0), (4, 1, 1)), ((0, 0, 0), (1, 1, 1)))
        self.assertEqual(len(group["notes"]), 1, group["notes"])

    def test_unknown_bounds_are_not_measured(self):
        group = self._group(((0, 0, 0), (1, 1, 1)), None)
        self.assertEqual(group["notes"], [])


if __name__ == "__main__":
    unittest.main()
