# !/usr/bin/python
# coding=utf-8
"""HierarchyBaseline -- the per-scene hierarchy diff's set algebra."""

import unittest

from pythontk.core_utils.engines.scene_export.hierarchy_baseline import (
    HierarchyBaseline as HB,
)


A = {"assetA_grp", "assetA_grp|meshA1", "assetA_grp|meshA2"}
B = {"assetB_grp", "assetB_grp|meshB1"}


class TestScoping(unittest.TestCase):
    def test_top_level_keeps_only_the_shallowest(self):
        self.assertEqual(HB.top_level({"a|b", "a", "c", "a|b|d"}), ["a", "c"])

    def test_in_scope_matches_whole_components_only(self):
        """`assetA_grp` must not claim `assetA_grp_old` -- a bare startswith
        would merge two unrelated assets into one scope."""
        self.assertEqual(
            HB.in_scope({"assetA_grp|m", "assetA_grp_old|m"}, ["assetA_grp"]),
            {"assetA_grp|m"},
        )

    def test_an_unrelated_root_is_not_in_play(self):
        self.assertEqual(HB.relevant_roots(A, B), [])

    def test_a_moved_root_is_still_in_play(self):
        """Wrapping an export in a new group rewrites every path. Scoping off
        the CURRENT roots alone would find nothing to diff and report a wrapped
        scene as a clean first export -- the exact accident being checked for."""
        wrapped = {"Wrapper", "Wrapper|assetA_grp", "Wrapper|assetA_grp|meshA1"}
        self.assertEqual(HB.relevant_roots(A, wrapped), ["assetA_grp"])

    def test_an_empty_export_puts_the_whole_baseline_in_play(self):
        """ "No scope" must not read as "nothing to check": an export set that
        collapsed to empty is the loudest structural change there is."""
        self.assertEqual(HB.relevant_roots(A, set()), ["assetA_grp"])


class TestCompare(unittest.TestCase):
    def test_a_first_export_of_a_scope_is_new_not_extra(self):
        match, missing, extra, new = HB.compare(set(), A)
        self.assertTrue(match)
        self.assertTrue(new)
        self.assertEqual((missing, extra), ([], []))

    def test_an_unchanged_scope_matches(self):
        match, missing, extra, new = HB.compare(A, A)
        self.assertTrue(match)
        self.assertFalse(new)
        self.assertEqual((missing, extra), ([], []))

    def test_a_deletion_is_reported_missing(self):
        match, missing, _, new = HB.compare(A, A - {"assetA_grp|meshA2"})
        self.assertFalse(match)
        self.assertFalse(new)
        self.assertEqual(missing, ["assetA_grp|meshA2"])

    def test_another_asset_is_not_reported_missing(self):
        """The whole point of one-baseline-per-scene: exporting B must not
        report every path of A as gone."""
        match, missing, extra, new = HB.compare(A | B, B)
        self.assertTrue(match, f"unexpected diff: missing={missing} extra={extra}")

    def test_a_wrapper_group_is_still_detected(self):
        wrapped = {"Wrapper", "Wrapper|assetA_grp", "Wrapper|assetA_grp|meshA1"}
        match, missing, extra, _ = HB.compare(A, wrapped)
        self.assertFalse(match)
        self.assertIn("assetA_grp|meshA1", missing)
        self.assertIn("Wrapper|assetA_grp|meshA1", extra)

    def test_an_emptied_export_reports_everything_missing(self):
        match, missing, _, new = HB.compare(A, set())
        self.assertFalse(match)
        self.assertFalse(new)
        self.assertEqual(missing, sorted(A))


class TestMerge(unittest.TestCase):
    def test_merge_leaves_other_scopes_alone(self):
        merged = HB.merge(A | B, A - {"assetA_grp|meshA2"})
        self.assertEqual(merged, (A - {"assetA_grp|meshA2"}) | B)

    def test_merge_uses_the_same_scope_compare_did(self):
        """Or it would strand the very paths it just reported missing."""
        shrunk = A - {"assetA_grp|meshA2"}
        _, missing, _, _ = HB.compare(A | B, shrunk)
        merged = HB.merge(A | B, shrunk)
        for path in missing:
            self.assertNotIn(path, merged)

    def test_an_empty_export_does_not_wipe_the_baseline(self):
        """REGRESSION: relevant_roots puts the WHOLE baseline in play for an
        empty set -- right for the diff (everything is missing), catastrophic
        for the merge, which replaced all of it with nothing and erased every
        scope. The next export would then see a clean slate, hiding the very
        collapse that had just happened."""
        self.assertEqual(HB.merge(A | B, set()), A | B)

    def test_one_record_accumulates_every_scope(self):
        base = HB.merge(HB.merge(set(), A), B)
        self.assertEqual(base, A | B)


class TestAdopt(unittest.TestCase):
    """A deliverable's sidecar fills the scope the scene's record lacks."""

    def test_a_scene_with_no_record_takes_what_the_deliverable_shipped(self):
        self.assertEqual(HB.adopt(set(), A), A)

    def test_a_second_deliverable_is_adopted_beside_the_first(self):
        """REGRESSION: adoption ran only into an EMPTY record. A scene
        exporting A and B, its baseline set aside (a legacy unstamped record),
        adopted A's sidecar at A's export and then refused B's -- B's next
        export had nothing to diff and passed a deleted child."""
        record = HB.adopt(set(), A)
        record = HB.adopt(record, B)
        self.assertEqual(record, A | B)
        _, missing, _, new = HB.compare(record, B - {"assetB_grp|meshB1"})
        self.assertFalse(new)
        self.assertEqual(missing, ["assetB_grp|meshB1"])

    def test_a_scope_the_record_holds_is_never_overwritten(self):
        """The scene's record is newer than the sidecar: a stale sidecar merged
        over it would resurrect paths the scene has since dropped."""
        dropped = A - {"assetA_grp|meshA2"}
        self.assertIsNone(HB.adopt(dropped | B, A))

    def test_a_moved_scope_counts_as_held(self):
        """The same scope rule compare uses: a record holding the root wrapped
        in a new group still holds it."""
        wrapped = {"Wrapper", "Wrapper|assetA_grp", "Wrapper|assetA_grp|meshA1"}
        self.assertIsNone(HB.adopt(A, wrapped))

    def test_nothing_shipped_adopts_nothing(self):
        for shipped in (None, set(), []):
            with self.subTest(shipped=shipped):
                self.assertIsNone(HB.adopt(A, shipped))


class TestRecord(unittest.TestCase):
    def test_encode_decode_round_trips(self):
        self.assertEqual(HB.decode(HB.encode(A)), A)

    def test_decode_is_empty_for_anything_unreadable(self):
        for value in (None, "", "not json", [], {}, {"format": 99, "paths": ["a"]}):
            with self.subTest(value=value):
                self.assertEqual(HB.decode(value), set())

    def test_is_record_separates_no_paths_from_unreadable(self):
        """REGRESSION: a valid record holding no paths decodes to an empty set
        exactly as a corrupt one does, so a consumer using decode() alone
        reported an intact baseline as LOST."""
        self.assertTrue(HB.is_record(HB.encode(set())))
        self.assertTrue(HB.is_record(HB.encode(A)))
        for value in (None, "", "not json", [], {}, {"format": 99, "paths": []}):
            with self.subTest(value=value):
                self.assertFalse(HB.is_record(value))

    def test_decode_accepts_a_raw_json_string(self):
        import json

        self.assertEqual(HB.decode(json.dumps(HB.encode(A))), A)

    def test_the_hash_covers_the_paths_and_nothing_else(self):
        self.assertEqual(HB.encode(A)["hash"], HB.paths_hash(A))
        self.assertNotEqual(HB.paths_hash(A), HB.paths_hash(B))

    def test_a_record_names_the_scene_that_recorded_it(self):
        """A Save As copy carries its source's record verbatim, so only the
        record itself can say which scene file recorded it (2026-09-24: a
        module scene saved as a new module failed its first export against
        the SOURCE's hierarchy)."""
        record = HB.encode(A, scene="scenes/room.ma")
        self.assertEqual(HB.recorded_by(record), "scenes/room.ma")
        # The stamp is neither a path nor hashed: it changes no diff.
        self.assertEqual(HB.decode(record), A)
        self.assertEqual(record["hash"], HB.paths_hash(A))
        self.assertTrue(HB.is_record(record))

    def test_an_unsaved_scene_stamps_empty_and_no_stamp_is_none(self):
        """``""`` is a stamp -- recorded while unsaved; ``None`` is its absence,
        a record written before records were stamped."""
        self.assertEqual(HB.recorded_by(HB.encode(A, scene="")), "")
        self.assertIsNone(HB.recorded_by(HB.encode(A)))

    def test_recorded_by_is_none_for_anything_unreadable(self):
        unknown = {"format": 99, "paths": [], "scene": "x.ma"}
        for value in (None, "", "not json", [], {}, unknown):
            with self.subTest(value=value):
                self.assertIsNone(HB.recorded_by(value))

    def test_recorded_by_accepts_a_raw_json_string(self):
        import json

        raw = json.dumps(HB.encode(A, scene="a.ma"))
        self.assertEqual(HB.recorded_by(raw), "a.ma")


class _FakeStore:
    """A scene store over a dict: the two channel primitives plus the writer
    stamp pair (``SceneStoreBase``'s contract, minus a real scene file)."""

    channels = {}
    scene = "shot_a.ma"
    #: Scene files still on disk -- a stamp naming one of these that is not
    #: ``scene`` is a Save As source, so its record is not this scene's.
    on_disk = {"shot_a.ma"}

    @classmethod
    def read(cls, scope, key):
        return cls.channels.get((scope, key))

    @classmethod
    def write(cls, scope, key, text):
        if text is None:
            cls.channels.pop((scope, key), None)
            return None
        cls.channels[(scope, key)] = text
        return "data_internal"

    @classmethod
    def writer_stamp(cls):
        return cls.scene

    @classmethod
    def written_here(cls, stamp):
        if stamp is None:
            return False
        return stamp in ("", cls.scene) or stamp not in cls.on_disk


class _FakeSidecar:
    """What each deliverable last shipped, by export path."""

    shipped = {}
    migrated = []

    @classmethod
    def migrate_legacy(cls, export_path, base_stem=False):
        cls.migrated.append(export_path)

    @classmethod
    def read_manifest(cls, export_path, base_stem=False):
        return cls.shipped.get(export_path)


class TestStore(unittest.TestCase):
    """HierarchyBaselineStore: the storage both DCC subclasses inherit."""

    def setUp(self):
        from pythontk.core_utils.engines.scene_export.hierarchy_baseline import (
            HierarchyBaselineStore,
        )

        class Store(_FakeStore):
            channels = {}
            scene = "shot_a.ma"
            on_disk = {"shot_a.ma"}

        class Sidecar(_FakeSidecar):
            shipped = {}
            migrated = []

        class Baseline(HierarchyBaselineStore):
            STORE = Store
            SIDECAR = Sidecar

        self.Store, self.Sidecar, self.Baseline = Store, Sidecar, Baseline

    def test_no_record_reads_empty_and_is_not_unreadable(self):
        self.assertEqual(self.Baseline.read(), set())
        self.assertFalse(self.Baseline.is_unreadable())
        self.assertIsNone(self.Baseline.inherited_from())

    def test_write_then_read_round_trips_stamped_with_this_scene(self):
        self.assertTrue(self.Baseline.write(A))
        self.assertEqual(self.Baseline.read(), A)
        from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords

        raw = SceneRecords.HIERARCHY_BASELINE.read_text(self.Store)
        self.assertEqual(HB.recorded_by(raw), "shot_a.ma")

    def test_write_rolls_one_scope_forward_and_keeps_the_others(self):
        self.Baseline.write(A)
        self.Baseline.write(B)
        self.assertEqual(self.Baseline.read(), A | B)
        match, missing, extra, new = self.Baseline.compare(B - {"assetB_grp|meshB1"})
        self.assertEqual((match, missing, new), (False, ["assetB_grp|meshB1"], False))

    def test_an_empty_write_records_nothing(self):
        self.assertTrue(self.Baseline.write(set()))
        self.assertEqual(self.Store.channels, {})

    def test_a_save_as_copy_sets_its_source_record_aside(self):
        self.Baseline.write(A)
        # The copy: same record, another scene file, the source still on disk.
        self.Store.scene, self.Store.on_disk = "shot_b.ma", {"shot_a.ma", "shot_b.ma"}
        self.assertEqual(self.Baseline.read(), set())
        self.assertEqual(self.Baseline.inherited_from(), "shot_a.ma")
        # Its first export replaces the source's record rather than merging.
        self.Baseline.write(B)
        self.assertEqual(self.Baseline.read(), B)
        self.assertIsNone(self.Baseline.inherited_from())

    def test_unreadable_is_flagged_and_reads_empty(self):
        from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords

        SceneRecords.HIERARCHY_BASELINE.write_text(self.Store, "{not json")
        self.assertTrue(self.Baseline.is_unreadable())
        self.assertEqual(self.Baseline.read(), set())

    def test_close_hook_runs_on_the_way_out(self):
        class Closed(self.Baseline):
            @classmethod
            def _close(cls, paths):
                return paths | {"closed"}

        Closed.write(A)
        self.assertEqual(Closed.read(), A | {"closed"})

    def test_adopt_sidecar_takes_this_deliverable_only(self):
        self.Sidecar.shipped = {"a.fbx": A, "b.fbx": B}
        self.assertTrue(self.Baseline.adopt_sidecar("a.fbx"))
        self.assertEqual(self.Baseline.read(), A)
        self.assertEqual(self.Sidecar.migrated, ["a.fbx"])
        # A scope the record holds is never adopted over.
        self.assertFalse(self.Baseline.adopt_sidecar("a.fbx"))
        self.assertTrue(self.Baseline.adopt_sidecar("b.fbx"))
        self.assertEqual(self.Baseline.read(), A | B)

    def test_adopt_sidecar_never_raises(self):
        class Broken(self.Sidecar):
            @classmethod
            def read_manifest(cls, export_path, base_stem=False):
                raise OSError("disk gone")

        class Baseline(self.Baseline):
            SIDECAR = Broken

        self.assertFalse(Baseline.adopt_sidecar("a.fbx"))

    def test_the_dcc_classes_keep_the_channel_name(self):
        self.assertEqual(self.Baseline.ATTR_NAME, "hierarchy_baseline")


if __name__ == "__main__":
    unittest.main()
