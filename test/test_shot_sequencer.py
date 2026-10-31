# !/usr/bin/python
# coding=utf-8
"""Tests for ``pythontk.core_utils.engines.shots.shot_sequencer`` (ShotSequencer).

The class is the ripple-editing orchestration mayatk's and blendertk's
``ShotSequencer`` subclass.  Used directly it is a bounds-only sequencer, and
every bounds expectation below was RECORDED from mayatk's ``ShotSequencer``
running headless (``maya.cmds`` unavailable, so bounds-only) before the
orchestration was hoisted here -- these pin that the hoist preserved it.

The second half drives a recording subclass to pin the hook contract: which
scene hooks each operation reaches, in which order, with which arguments.
"""

import os
import shutil
import sys
import tempfile
import unittest

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pythontk import SceneRecords, ShotPlanner, ShotSequencer, ShotStore  # noqa: E402


class _SequencerTest(unittest.TestCase):
    """Three shots A [1, 10], B [15, 30], C [35, 50]; prefs sandboxed."""

    SEQUENCER = ShotSequencer

    def setUp(self):
        self._prefs_tmp = tempfile.mkdtemp(prefix="shot_sequencer_prefs_")
        ShotStore._prefs_dir_override = self._prefs_tmp
        ShotStore._active = None
        ShotStore._persistence = None
        self.seq = self.SEQUENCER()
        for name, (start, end) in zip("ABC", ((1, 10), (15, 30), (35, 50))):
            self.seq.define_shot(name, start, end, objects=[name.lower()])

    def tearDown(self):
        ShotStore._prefs_dir_override = None
        ShotStore._active = None
        shutil.rmtree(self._prefs_tmp, ignore_errors=True)

    def sid(self, name):
        return self.seq.shot_by_name(name).shot_id

    def bounds(self):
        return [(s.name, s.start, s.end) for s in self.seq.sorted_shots()]

    def assertBounds(self, *expected):
        self.assertEqual(self.bounds(), list(expected))


class TestBoundsOnlyEdits(_SequencerTest):
    def test_expand_ripples_downstream(self):
        self.assertEqual(self.seq.expand_shot(self.sid("A"), 14), 4.0)
        self.assertBounds(("A", 1, 14), ("B", 19, 34), ("C", 39, 54))

    def test_expand_never_shrinks(self):
        self.assertEqual(self.seq.expand_shot(self.sid("A"), 5), 0.0)
        self.assertBounds(("A", 1, 10), ("B", 15, 30), ("C", 35, 50))

    def test_apply_gap_all(self):
        self.assertTrue(self.seq.apply_gap(3))
        self.assertBounds(("A", 1, 10), ("B", 13, 28), ("C", 31, 46))

    def test_apply_gap_scoped(self):
        self.assertTrue(self.seq.apply_gap(2, scope="start", shot_id=self.sid("B")))
        self.assertBounds(("A", 1, 10), ("B", 12, 27), ("C", 32, 47))

    def test_apply_gap_end_moves_the_successor(self):
        self.assertTrue(self.seq.apply_gap(8, scope="end", shot_id=self.sid("A")))
        self.assertBounds(("A", 1, 10), ("B", 18, 33), ("C", 38, 53))

    def test_apply_gap_start_end(self):
        self.seq.apply_gap(1, scope="start_end", shot_id=self.sid("B"))
        self.assertBounds(("A", 1, 10), ("B", 11, 26), ("C", 27, 42))

    def test_move_shot_ripples_downstream_both_ways(self):
        self.seq.move_shot(self.sid("B"), 20)
        self.assertBounds(("A", 1, 10), ("B", 20, 35), ("C", 40, 55))
        self.seq.move_shot(self.sid("B"), 17)
        self.assertBounds(("A", 1, 10), ("B", 17, 32), ("C", 37, 52))

    def test_slide_upstream_moves_the_shots_before(self):
        self.seq.slide_shot(self.sid("B"), 11, direction="upstream")
        self.assertBounds(("A", -3, 6), ("B", 11, 26), ("C", 35, 50))

    def test_slide_without_ripple_is_clamped_by_the_neighbour(self):
        self.seq.slide_shot(self.sid("B"), 2, direction=None)
        self.assertBounds(("A", 1, 10), ("B", 10, 25), ("C", 35, 50))

    def test_slide_to_a_fractional_frame_lands_on_the_grid(self):
        self.seq.slide_shot(self.sid("B"), 17.4, direction="downstream")
        self.assertBounds(("A", 1, 10), ("B", 17, 32), ("C", 37, 52))

    def test_set_shot_start(self):
        self.seq.set_shot_start(self.sid("B"), 18)
        self.assertBounds(("A", 1, 10), ("B", 18, 33), ("C", 38, 53))

    def test_set_shot_start_without_ripple(self):
        self.seq.set_shot_start(self.sid("B"), 17, ripple=False)
        self.assertBounds(("A", 1, 10), ("B", 17, 32), ("C", 35, 50))

    def test_set_shot_duration_ripples_by_the_delta(self):
        self.seq.set_shot_duration(self.sid("A"), 20)
        self.assertBounds(("A", 1, 21), ("B", 26, 41), ("C", 46, 61))

    def test_resize_shot_ripples_both_sides(self):
        self.seq.resize_shot(self.sid("B"), 13, 33)
        self.assertBounds(("A", -1, 8), ("B", 13, 33), ("C", 38, 53))

    def test_resize_bounds_grow_and_shrink_keep_the_gaps(self):
        self.seq.resize_shot_bounds(self.sid("B"), 13, 33)
        self.assertBounds(("A", -1, 8), ("B", 13, 33), ("C", 38, 53))

    def test_resize_bounds_shrink_pulls_neighbours_in(self):
        self.seq.resize_shot_bounds(self.sid("B"), 18, 25)
        self.assertBounds(("A", 4, 13), ("B", 18, 25), ("C", 30, 45))

    def test_resize_bounds_normalizes_an_inverted_range(self):
        self.seq.resize_shot_bounds(self.sid("B"), 30, 16)
        self.assertBounds(("A", 2, 11), ("B", 16, 30), ("C", 35, 50))

    def test_resize_object_grows_the_shot_and_ripples(self):
        self.seq.resize_object(self.sid("B"), "b", 15, 30, 15, 36)
        self.assertBounds(("A", 1, 10), ("B", 15, 36), ("C", 41, 56))

    def test_move_object_in_shot_grows_to_enclose_the_landing(self):
        self.seq.move_object_in_shot(self.sid("B"), "b", 15, 30, 25)
        self.assertBounds(("A", 1, 10), ("B", 15, 40), ("C", 45, 60))

    def test_missing_ids_raise(self):
        for call in (
            lambda: self.seq.expand_shot(999, 3),
            lambda: self.seq.slide_shot(999, 3),
            lambda: self.seq.resize_shot_bounds(999, 1, 2),
            lambda: self.seq.delete_shot(999),
            lambda: self.seq.add_shot_space(999, 2),
            lambda: self.seq.move_shot_to_position(999, 1),
            lambda: self.seq.swap_shots(999, self.sid("A")),
            lambda: self.seq.swap_shots(self.sid("A"), 999),
            lambda: self.seq.copy_shot_content(999, self.sid("A")),
            lambda: self.seq.copy_shot_content(self.sid("A"), 999),
            lambda: self.seq.fit_shot_to_content(999),
            lambda: self.seq.move_object_in_shot(999, "x", 1, 2, 3),
        ):
            with self.assertRaisesRegex(ValueError, "No shot with id 999"):
                call()


class TestTopology(_SequencerTest):
    def test_move_to_position(self):
        self.seq.move_shot_to_position(self.sid("C"), 1)  # at the default gap, 10
        self.assertBounds(("C", 1, 16), ("A", 26, 35), ("B", 45, 60))

    def test_swap_trades_places_whole_by_default(self):
        self.seq.swap_shots(self.sid("A"), self.sid("C"))  # the default gap, 10
        self.assertBounds(("C", 1, 16), ("B", 26, 41), ("A", 51, 60))
        self.assertEqual(self.seq.shot_by_name("A").objects, ["a"])

    def test_swap_retains_fields_in_place(self):
        store = self.seq.store
        a, c = self.sid("A"), self.sid("C")
        store.update_shot(a, description="intro", locked=True, metadata={"step": "1"})
        store.update_shot(c, description="outro", metadata={"step": "3"})
        self.seq.swap_shots(a, c, retain=("name", "description", "locked", "metadata"))
        # C's animation and members now play first, under the first place's
        # name, notes, lock and step; the shot ids travel with the animation.
        self.assertBounds(("A", 1, 16), ("B", 26, 41), ("C", 51, 60))
        first, last = self.seq.sorted_shots()[0], self.seq.sorted_shots()[-1]
        self.assertEqual((first.shot_id, first.objects), (c, ["c"]))
        self.assertEqual(
            (first.description, first.locked, first.metadata),
            ("intro", True, {"step": "1"}),
        )
        self.assertEqual(
            (last.description, last.locked, last.metadata),
            ("outro", False, {"step": "3"}),
        )

    def test_swap_keeps_the_system_member_tags_with_their_members(self):
        store = self.seq.store
        a, c = self.sid("A"), self.sid("C")
        store.add_system_members(a, ["a2"])
        store.update_shot(c, metadata={"step": "3"})
        self.seq.swap_shots(a, c, retain=("metadata",))
        meta_a = self.seq.shot_by_id(a).metadata
        self.assertEqual(meta_a.get("step"), "3")
        self.assertEqual(store.system_members(self.seq.shot_by_id(a)), {"a2"})
        self.assertNotIn(store.SYSTEM_MEMBERS, self.seq.shot_by_id(c).metadata)

    def test_swap_neighbours(self):
        self.seq.swap_shots(self.sid("B"), self.sid("A"), retain=("name",))
        self.assertBounds(("A", 1, 16), ("B", 26, 35), ("C", 45, 60))

    def test_swap_refuses_a_field_it_cannot_retain(self):
        with self.assertRaisesRegex(ValueError, "objects"):
            self.seq.swap_shots(self.sid("A"), self.sid("C"), retain=("objects",))
        self.assertBounds(("A", 1, 10), ("B", 15, 30), ("C", 35, 50))

    def test_insert_after(self):
        # The store's default gap, 10 (maintainer 2026-10-11), after A.
        block = self.seq.insert_shot("N", 5, after_shot_id=self.sid("A"))
        self.assertEqual((block.name, block.start, block.end), ("N", 20, 25))
        self.assertBounds(("A", 1, 10), ("N", 20, 25), ("B", 30, 45), ("C", 50, 65))

    def test_insert_at_front(self):
        self.seq.insert_shot("N", 4, at_position=1, gap=2)
        self.assertBounds(("N", 1, 5), ("A", 7, 16), ("B", 21, 36), ("C", 41, 56))

    def test_insert_appends_by_default(self):
        self.seq.insert_shot("N", 6)  # the default gap, 10, after C
        self.assertBounds(("A", 1, 10), ("B", 15, 30), ("C", 35, 50), ("N", 60, 66))

    def test_new_shot_takes_the_next_name_and_the_initial_length(self):
        self.seq.store.initial_shot_length = 20
        block = self.seq.new_shot(after_shot_id=self.sid("A"))
        self.assertEqual((block.name, block.objects), ("Shot_4", []))
        self.assertBounds(  # the default gap, 10, after A
            ("A", 1, 10), ("Shot_4", 20, 40), ("B", 45, 60), ("C", 65, 80)
        )

    def test_insert_moves_the_shots_after_it_by_position(self):
        """A zero-length shot on the next shot's start frame stays before the
        shot inserted after it.

        Bug: the push took every shot starting at the next shot's frame, so
        Z moved past N: A, N, Z, B.
        Fixed: 2026-10-12
        """
        store = self.seq.store
        for name in "BC":
            store.remove_shot(self.sid(name))
        store.define_shot("Z", 15, 15)
        store.define_shot("B", 15, 30)
        self.seq.insert_shot("N", 5, after_shot_id=self.sid("Z"))
        self.assertBounds(("A", 1, 10), ("Z", 15, 15), ("N", 25, 30), ("B", 30, 45))

    def test_overlapping_shots_sharing_a_start_move_together(self):
        """Two overlapping shots on one start frame move together, the shot
        before the insert with its keys, as the frame rule always moved them.

        Bug: moved by position, B1 stayed while its keys -- in the window B2
        owns, B1's own one empty -- went with B2: keys left their shot.
        Fixed: 2026-10-13
        """
        store = self.seq.store
        for name in "BC":
            store.remove_shot(self.sid(name))
        store.define_shot("B1", 15, 30)
        store.define_shot("B2", 15, 40)
        moved = []

        def apply_plan(plan, retime_gaps=False):
            moved.extend(sid for sid in plan.sequence)
            ShotSequencer._apply_plan(self.seq, plan, retime_gaps)

        self.seq._apply_plan = apply_plan
        self.seq.insert_shot("N", 5, after_shot_id=self.sid("B1"))
        self.assertIn(self.sid("B1"), moved, "B1 and its keys move")
        self.assertBounds(("A", 1, 10), ("B1", 30, 45), ("B2", 30, 55), ("N", 40, 45))

    def test_insert_placement_is_where_insert_shot_puts_it(self):
        """The one placement rule, read before anything moves: the new
        shot's bounds and how far the shots from its position on move."""
        self.assertEqual(self.seq.insert_placement(1, 5), (20, 25, 15, (1, 2)))
        self.assertEqual(self.seq.insert_placement(0, 4, gap=2), (1, 5, 6, (0, 1, 2)))
        self.assertEqual(self.seq.insert_placement(3, 6), (60, 66, 0.0, ()))
        # A timeline as a plan has it: rows moved by a planned ripple.
        c = self.seq.shot_by_name("C")
        line = [(1, 10, None), (100, 115, c)]
        self.assertEqual(
            self.seq.insert_placement(2, 6, line=line), (125, 131, 0.0, ())
        )
        # A start the caller knows (content the store does not hold).
        self.assertEqual(
            self.seq.insert_placement(1, 5, start=12), (12, 17, 15, (1, 2))
        )

    def test_insert_at_a_start_the_caller_knows(self):
        """``start`` places the new shot where the caller says; the shots
        after it still move by its length plus the gap."""
        self.seq.insert_shot("N", 5, after_shot_id=self.sid("A"), start=12)
        self.assertBounds(("A", 1, 10), ("N", 12, 17), ("B", 30, 45), ("C", 50, 65))
        self.seq.insert_shot("T", 6, start=90)  # appended: nothing moves
        self.assertEqual(self.bounds()[-1], ("T", 90, 96))

    def test_insert_refuses_a_taken_name_before_anything_moves(self):
        with self.assertRaisesRegex(ValueError, "another shot is named 'B'"):
            self.seq.insert_shot("B", 6, after_shot_id=self.sid("A"))
        self.assertBounds(("A", 1, 10), ("B", 15, 30), ("C", 35, 50))

    def test_delete_closes_the_vacated_span(self):
        result = self.seq.delete_shot(self.sid("B"))
        self.assertEqual(result, {"curves_cut": 0, "closed": 20.0, "name": "B"})
        self.assertBounds(("A", 1, 10), ("C", 15, 30))

    def test_delete_can_leave_the_hole(self):
        result = self.seq.delete_shot(self.sid("B"), close_gap=False)
        self.assertEqual(result["closed"], 0.0)
        self.assertBounds(("A", 1, 10), ("C", 35, 50))

    def test_delete_last_closes_nothing(self):
        self.assertEqual(self.seq.delete_shot(self.sid("C"))["closed"], 0.0)
        self.assertBounds(("A", 1, 10), ("B", 15, 30))

    def test_merge_keeps_the_earliest_and_unions(self):
        self.seq.shot_by_name("A").description = "one"
        self.seq.shot_by_name("B").description = "two"
        merged = self.seq.merge_shots([self.sid("B"), self.sid("A")])
        self.assertEqual((merged.name, merged.start, merged.end), ("A", 1, 30))
        self.assertEqual(merged.objects, ["a", "b"])
        self.assertEqual(merged.description, "one / two")
        self.assertBounds(("A", 1, 30), ("C", 35, 50))

    def test_merge_can_rename(self):
        merged = self.seq.merge_shots([self.sid("B"), self.sid("C")], name="BC")
        self.assertEqual((merged.name, merged.start, merged.end), ("BC", 15, 50))

    def test_merge_needs_two_shots(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            self.seq.merge_shots([self.sid("A"), 999])

    def test_split(self):
        tail = self.seq.split_shot(self.sid("B"), 20)
        self.assertEqual((tail.name, tail.start, tail.end), ("B_2", 20, 30))
        self.assertEqual(tail.objects, ["b"])
        self.assertBounds(("A", 1, 10), ("B", 15, 20), ("B_2", 20, 30), ("C", 35, 50))

    def test_split_with_a_gap_ripples_the_tail(self):
        self.seq.split_shot(self.sid("B"), 22, name="B2", gap=4)
        self.assertBounds(("A", 1, 10), ("B", 15, 22), ("B2", 26, 34), ("C", 39, 54))

    def test_split_on_a_bound_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not inside B"):
            self.seq.split_shot(self.sid("B"), 15)

    def test_split_refuses_a_taken_tail_name_before_trimming(self):
        with self.assertRaises(ValueError):
            self.seq.split_shot(self.sid("B"), 20, name="C")
        self.assertBounds(("A", 1, 10), ("B", 15, 30), ("C", 35, 50))

    def test_a_split_links_the_tail_to_its_heads_step(self):
        """Bug: the tail got a name and a description and nothing relating
        it to its head's step (2026-10-11). The split writes a non-binding
        link (``ShotStore.STEP_LINK``), never the head's binding: a step has
        one shot. The export ships no section (2026-10-20, no reader read
        it): the link is the store's (``ShotStore.step_links``)."""
        b = self.sid("B")
        self.seq.store.update_shot(b, metadata={"step": "B01", "section": "B"})
        tail = self.seq.split_shot(b, 20, name="B01_2")
        self.assertEqual(tail.metadata, {ShotStore.STEP_LINK: "B01"})
        self.assertEqual(self.seq.store.shot_by_id(b).metadata["step"], "B01")
        again = self.seq.split_shot(tail.shot_id, 25)  # a tail split again
        self.assertEqual(again.metadata, {ShotStore.STEP_LINK: "B01"})
        store = self.seq.store
        self.assertEqual(
            {store.shot_by_id(i).name: step for i, step in store.step_links().items()},
            {"B01_2": "B01", "B01_2_2": "B01"},
        )
        clips = store.to_export_view()[SceneRecords.SHOTS.key]["shots"]
        self.assertFalse([c for c in clips if "section" in c])
        self.assertEqual(self.seq.split_shot(self.sid("A"), 5).metadata, {})


class TestSpacing(_SequencerTest):
    def test_add_trailing_space(self):
        self.assertEqual(
            self.seq.add_shot_space(self.sid("A"), 5, edge="trailing"), (0.0, 5.0)
        )
        self.assertBounds(("A", 1, 15), ("B", 20, 35), ("C", 40, 55))

    def test_add_leading_space_holds_the_start(self):
        self.assertEqual(
            self.seq.add_shot_space(self.sid("A"), 3, edge="leading"), (0.0, 3.0)
        )
        self.assertBounds(("A", 1, 13), ("B", 18, 33), ("C", 38, 53))

    def test_add_space_both(self):
        self.assertEqual(
            self.seq.add_shot_space(self.sid("B"), 2, edge="both"), (0.0, 4.0)
        )
        self.assertBounds(("A", 1, 10), ("B", 15, 34), ("C", 39, 54))

    def test_removing_head_room_of_an_empty_shot_is_a_no_op(self):
        self.assertEqual(
            self.seq.add_shot_space(self.sid("B"), -3, edge="leading"), (0.0, 0.0)
        )
        self.assertEqual(self.seq.add_shot_space(self.sid("B"), 0), (0.0, 0.0))
        self.assertBounds(("A", 1, 10), ("B", 15, 30), ("C", 35, 50))

    def test_respace(self):
        self.seq.respace(gap=2, start_frame=5)
        self.assertBounds(("A", 5, 14), ("B", 16, 31), ("C", 33, 48))

    def test_respace_honours_a_locked_gap(self):
        self.seq.store.lock_gap(self.sid("A"), self.sid("B"))
        self.seq.respace(gap=1, start_frame=1)
        self.assertBounds(("A", 1, 10), ("B", 15, 30), ("C", 31, 46))

    def test_fit_trim_extend_without_content_do_nothing(self):
        b = self.sid("B")
        self.assertEqual(self.seq.fit_shot_to_content(b), (0.0, 0.0))
        self.assertEqual(self.seq.trim_shot_to_content(b), (0.0, 0.0))
        self.assertEqual(self.seq.extend_shot_to_fit(b), (0.0, 0.0))
        self.assertBounds(("A", 1, 10), ("B", 15, 30), ("C", 35, 50))


class TestPureQueries(_SequencerTest):
    def test_sequence_separation_is_one_frame_past_the_gap(self):
        self.assertEqual(self.seq.sequence_separation(), 11.0)  # default gap 10
        self.seq.store.gap = 0
        self.assertEqual(self.seq.sequence_separation(), 1.0)
        self.seq.store.gap = 4
        self.assertEqual(self.seq.sequence_separation(), 5.0)

    def test_envelopes(self):
        self.assertEqual(
            [self.seq._shot_envelope(self.sid(n)) for n in "ABC"],
            [
                (1.0, 15.0, False, False),
                (15.0, 35.0, False, False),
                (35.0, 1e9, False, False),
            ],
        )
        self.assertIsNone(self.seq._shot_envelope(999))

    def test_source_shot_by_range(self):
        self.assertEqual(
            self.seq.source_shot_id({"start": 16, "end": 20}), self.sid("B")
        )
        self.assertIsNone(self.seq.source_shot_id({"start": 11, "end": 12}))

    def test_point_segment_is_a_marker(self):
        seg = ShotSequencer._point_segment("x", 4.0)
        self.assertEqual((seg["start"], seg["end"], seg["duration"]), (4.0, 4.0, 0.0))
        self.assertTrue(seg["marker"] and seg["is_stepped"])

    def test_round_trip_builds_the_store_class(self):
        self.seq.set_object_hidden("a", True)
        again = ShotSequencer.from_dict(self.seq.to_dict())
        self.assertIs(type(again.store), ShotSequencer.STORE_CLASS)
        self.assertEqual(
            [(s.name, s.start, s.end) for s in again.sorted_shots()], self.bounds()
        )
        self.assertTrue(again.is_object_hidden("a"))

    def test_reconcile_without_a_scene(self):
        self.assertEqual(
            self.seq.reconcile_system_edits(),
            {"keys_moved": 0, "keys_removed": 0, "holds": 0},
        )

    def test_reconcile_forgets_behavior_claims_on_a_deleted_curve(self):
        """A curve holding only behavior keys (a highlight, a fade) was swept
        by neither the gap-hold nor the boundary pass, so its claims outlived
        it -- SOL_REPLACE_ASSEMBLY held 399 on 24 curves the scene no longer
        had -- and a key later set on a curve of that name at a claimed frame
        read as the tool's.  Renamed claims are followed first."""

        class _Scene(ShotSequencer):
            live = {"lid_highlight"}
            followed = []

            def _curve_exists(self, curve):
                return curve in self.live

            def _follow_unresolved_claims(self, curves):
                self.followed.append(sorted(curves))
                return False

        seq = _Scene()
        led = seq.ledger
        led.record_authored("lid_highlight", 10, 0, "highlight", "lid")
        led.record_authored("gone_highlight", 10, 0, "highlight", "gone")

        seq.reconcile_system_edits()

        self.assertEqual(led.authored(), [("lid_highlight", 10.0)])
        self.assertEqual(seq.followed, [["gone_highlight", "lid_highlight"]])

    def test_define_shot_discovers_members_through_the_hook(self):
        shot = self.seq.define_shot("D", 60, 70)
        self.assertEqual(shot.objects, [])

    def test_detect_next_prefers_a_candidate_after_every_shot(self):
        cands = [
            {"name": "x", "start": 5, "end": 9, "objects": []},
            {"name": "y", "start": 52, "end": 60, "objects": []},
        ]
        self.seq.detect_shots = lambda **kw: cands
        self.assertEqual(self.seq.detect_next_shot()["name"], "y")

    def test_detect_next_falls_back_to_a_non_overlapping_candidate(self):
        cands = [
            {"name": "x", "start": 5, "end": 9, "objects": []},
            {"name": "y", "start": 31, "end": 33, "objects": []},
            {"name": "z", "start": 40, "end": 45, "objects": []},
        ]
        self.seq.detect_shots = lambda **kw: cands
        self.assertEqual(self.seq.detect_next_shot()["name"], "y")

    def test_detect_next_without_detection(self):
        self.assertIsNone(self.seq.detect_next_shot())


# ---------------------------------------------------------------- hook contract


class _SceneStore(ShotStore):
    """A store subclass, to prove ``STORE_CLASS`` is what gets built."""


class _Recording(ShotSequencer):
    """Scene present; every hook records its call."""

    STORE_CLASS = _SceneStore

    def __init__(self, *a, **kw):
        self.calls = []
        self.segments = {}
        self.times = []
        self.scans = []  # what each seam scan reads, in turn (the last repeats)
        self.pin_results = []  # what each pin call reports, in turn (then 0)
        super().__init__(*a, **kw)

    def _scene_available(self):
        return True

    class _Batch:
        def __init__(self, calls):
            self.calls = calls

        def __enter__(self):
            self.calls.append(("batch", "enter"))

        def __exit__(self, *exc):
            self.calls.append(("batch", "exit"))

    def _content_batch(self):
        return self._Batch(self.calls)

    def _apply_plan(self, plan, retime_gaps=False):
        self.calls.append(("apply_plan", retime_gaps))
        super()._apply_plan(plan, retime_gaps)

    def _reconcile_boundaries(self, plan):
        self.calls.append(("reconcile_boundaries",))
        return lambda: self.calls.append(("finish",))

    def _move_content_keys(
        self, objects, lo, hi, delta, lo_open=False, hi_closed=False
    ):
        self.calls.append(("move_content", list(objects), lo, hi, delta))

    def _shift_audio(self, old_start, old_end, delta):
        self.calls.append(("shift_audio", old_start, old_end, delta))

    def _keyed_transform_times(self):
        return {"stray": [1.0]}

    def _move_audio_sequence(self, seq, delta):
        self.calls.append(("move_audio", seq["obj"], delta))

    def move_attribute_keys(self, obj, attr, delta, times=None, window=None):
        self.calls.append(("move_attr", obj, attr, delta, window))
        return 0

    def collect_object_segments(self, shot_id, **kwargs):
        return list(self.segments.get(shot_id, []))

    def _animator_times_in(self, obj, attr, window):
        self.calls.append(("animator_times", obj, attr, window))
        return [t for t in self.times if window[0] <= t <= window[1]]

    def _object_key_probe(self, obj):
        self.calls.append(("probe", obj))
        return lambda frame: False

    def _gap_hold_seams(self):
        self.calls.append(("seams",))
        if len(self.scans) > 1:
            return self.scans.pop(0)
        return dict(self.scans[0]) if self.scans else {"s": []}

    def _pin_bound_keys(self, pins):
        self.calls.append(("pin", pins))
        return self.pin_results.pop(0) if self.pin_results else 0

    def _release_gap_holds(self, seams):
        self.calls.append(("release", seams))
        return 0

    def _apply_gap_holds(self, seams):
        self.calls.append(("hold", seams))
        return 0


class TestHookContract(_SequencerTest):
    SEQUENCER = _Recording

    def names(self):
        return [c[0] for c in self.seq.calls]

    def test_store_class_is_built(self):
        self.assertIsInstance(self.seq.store, _SceneStore)
        self.assertIsInstance(
            _Recording.from_dict(self.seq.to_dict()).store, _SceneStore
        )

    def test_whole_shot_move_reaches_content_audio_and_holds(self):
        self.seq.calls.clear()
        self.seq._move_shot_content(self.seq.shot_by_name("B"), 18)
        self.assertEqual(
            self.seq.calls,
            [
                ("reconcile_boundaries",),
                ("move_content", ["a", "b", "c", "stray"], 15.0, 35.0, 3.0),
                ("shift_audio", 15.0, 30.0, 3.0),
                ("finish",),
            ],
        )
        self.assertBounds(("A", 1, 10), ("B", 18, 33), ("C", 35, 50))

    def test_gap_holds_are_enforced_with_a_scene(self):
        self.seq.calls.clear()
        self.seq.reconcile_system_edits()
        self.assertEqual(self.names(), ["seams", "release", "hold"])

    def test_respace_asks_for_gap_retiming(self):
        self.seq.calls.clear()
        self.seq.respace(gap=2)
        self.assertIn(("apply_plan", True), self.seq.calls)

    def test_ripples_do_not_retime_gaps(self):
        self.seq.calls.clear()
        self.seq.expand_shot(self.sid("A"), 12)
        self.assertEqual(
            [c for c in self.seq.calls if c[0] == "apply_plan"], [("apply_plan", False)]
        )

    def test_scene_edits_are_batched(self):
        for op in (
            lambda: self.seq.set_shot_start(self.sid("B"), 18),
            lambda: self.seq.add_shot_space(self.sid("A"), 2, edge="trailing"),
        ):
            self.seq.calls.clear()
            op()
            batch = [c for c in self.seq.calls if c[0] == "batch"]
            self.assertEqual(batch, [("batch", "enter"), ("batch", "exit")])

    def test_sequence_moves_dispatch_by_kind(self):
        self.seq.calls.clear()
        self.seq._move_sequence(
            {"kind": "anim", "obj": "b", "start": 15, "end": 20}, 17
        )
        self.seq._move_sequence(
            {"kind": "audio", "obj": "trk", "start": 15, "end": 20}, 12
        )
        self.seq._move_sequence(
            {"kind": "anim", "obj": "b", "attr": "tx", "start": 15, "end": 20}, 16
        )
        self.assertEqual(
            self.seq.calls,
            [
                ("move_attr", "b", None, 2, (15, 20)),
                ("move_audio", "trk", -3),
                ("move_attr", "b", "tx", 1, (15, 20)),
            ],
        )

    def test_gap_overhang_is_read_through_the_hook(self):
        self.seq.times = [31.0, 33.0]
        seq = {"kind": "anim", "obj": "b", "start": 20, "end": 30}
        self.seq._absorb_gap_overhang(seq, 35)
        self.assertEqual(seq["end"], 33.0)
        self.assertIn(("animator_times", "b", None, (30.001, 34.999)), self.seq.calls)

    def test_an_overhang_inside_a_shot_is_not_taken(self):
        self.seq.times = [36.0]
        seq = {"kind": "anim", "obj": "b", "start": 20, "end": 30}
        self.seq._absorb_gap_overhang(seq, 40)
        self.assertEqual(seq["end"], 30)

    def test_move_object_in_shot_probes_the_object(self):
        self.seq.calls.clear()
        self.seq.move_object_in_shot(self.sid("B"), "b", 15, 30, 16)
        self.assertEqual(self.seq.calls[0], ("probe", "b"))

    def test_membership_follows_the_segments_with_a_scene(self):
        b = self.sid("B")
        self.seq.segments[b] = [{"obj": "x", "start": 16, "end": 20}]
        self.seq._recompute_shot_objects(b)
        self.assertEqual(self.seq.shot_by_id(b).objects, ["x"])

    def test_leading_room_reads_the_sequences(self):
        b = self.sid("B")
        self.seq.segments[b] = [{"obj": "b", "start": 19, "end": 25}]
        self.assertEqual(self.seq._leading_room(b), 4.0)

    def test_fit_trims_to_the_sequences(self):
        b = self.sid("B")
        self.seq.segments[b] = [{"obj": "b", "start": 18, "end": 25}]
        self.assertEqual(self.seq.fit_shot_to_content(b, mode="trim"), (3.0, -5.0))
        self.assertBounds(("A", 4, 13), ("B", 18, 25), ("C", 30, 45))

    def test_a_run_ending_on_the_next_shots_pin_lands_in_front(self):
        """Found 2026-10-11 by the SOL gap-0 acceptance: at gap 0 a run's
        drawn extent ended on the next shot's claimed start pin -- the
        SYSTEM's sample, on the shared frame -- so Move to Shot read a block
        from BEFORE the destination as one from after it and appended it
        behind the destination's content, where the shot after played it
        (KEYS 78 off over all 81 frames).  The run is the animator's keys."""
        seq = _Recording()
        for name, (start, end) in zip("PQR", ((0, 50), (50, 100), (100, 150))):
            seq.define_shot(name, start, end, objects=["p"])
        q = seq.shot_by_name("Q").shot_id
        seq.segments[q] = [{"obj": "p", "start": 50, "end": 90}]
        seq.times = [0.0, 20.0, 40.0]  # the animator's keys; 50 is Q's pin
        seq.calls.clear()
        seq.move_sequences_to_shot(
            [{"kind": "anim", "obj": "p", "start": 0.0, "end": 50.0}], q
        )
        moves = [c for c in seq.calls if c[0] == "move_attr"]
        self.assertEqual(moves, [("move_attr", "p", None, 50.0, (0.0, 40.0))])

    def test_markers_are_not_sequences(self):
        b = self.sid("B")
        self.seq.segments[b] = [ShotSequencer._point_segment("b", 20.0)]
        self.assertEqual(self.seq.collect_shot_sequences(b), [])


class TestGapHoldPins(_SequencerTest):
    """A gap hold stops where the next shot starts: before any hold is written
    that shot's start is pinned through the ``_pin_bound_keys`` hook.

    Bug (BACKLOG 2026-10-04, measured in a fresh Blender 5.1 and mayapy): the
    hold steps the last key before the next shot, and with no key ON that
    shot's start the held value ran on to its first key -- a no-op
    ``respace(gap=20)`` over gaps already 20 changed 4 frames of one shot and
    6 of the next, by up to 7.9 units.  The pin that stops it ran only inside
    a respace that changed a gap's width.
    """

    SEQUENCER = _Recording

    def pins(self):
        return [call[1] for call in self.seq.calls if call[0] == "pin"]

    def test_a_hold_is_stopped_on_the_next_shots_start_before_it_is_written(self):
        self.seq.scans = [{"crv": [12.0]}]  # a key in the A-B gap
        self.seq.calls.clear()
        self.seq._enforce_gap_holds()
        self.assertEqual(self.pins(), [{"crv": [(15.0, self.sid("B"), "start")]}])
        self.assertEqual(
            [call[0] for call in self.seq.calls], ["seams", "pin", "release", "hold"]
        )

    def test_a_seam_inside_its_shot_still_holds_from_there(self):
        """No key in the gap: the seam is A's own last key and the hold starts
        on it, as it always has -- only where the hold STOPS is pinned."""
        self.seq.scans = [{"crv": [8.0]}]
        self.seq.calls.clear()
        self.seq._enforce_gap_holds()
        self.assertEqual(self.pins(), [{"crv": [(15.0, self.sid("B"), "start")]}])
        self.assertEqual(self.seq.calls[-1], ("hold", {"crv": [8.0]}))

    def test_the_seams_are_read_again_while_a_pin_pass_changes_anything(self):
        """A DCC may put a key a hair off the bound ON it instead of pinning
        beside it, which moves a seam: holds go on the seams the last pass
        read."""
        b, c = self.sid("B"), self.sid("C")
        self.seq.scans = [{"crv": [8.0]}, {"crv": [8.0, 32.0]}]
        self.seq.pin_results = [1, 0]
        self.seq.calls.clear()
        self.seq._enforce_gap_holds()
        self.assertEqual(
            self.pins(),
            [
                {"crv": [(15.0, b, "start")]},
                {"crv": [(15.0, b, "start"), (35.0, c, "start")]},
            ],
        )
        self.assertEqual(
            self.seq.calls[-2:],
            [("release", {"crv": [8.0, 32.0]}), ("hold", {"crv": [8.0, 32.0]})],
        )

    def test_a_shots_start_pin_is_never_held(self):
        """A curve that passes through B keyless is pinned on B's start to
        stop the A-B hold, and that pin is then its only key in B -- the
        seam of the B-C gap.  Held, it would hold B still from its first
        frame; the curve has nothing in B to hold."""
        b, c = self.sid("B"), self.sid("C")
        self.seq.ledger.record_key("crv", 15.0, b, "start")
        self.seq.ledger.record_key("thru", 15.0, b, "start")
        self.seq.scans = [{"crv": [8.0, 15.0], "thru": [15.0]}]
        self.seq.calls.clear()
        self.seq._enforce_gap_holds()
        # Not held, but the crossing it opens is ended on B's own end and the
        # next shot's start pinned: the gap after B holds from there on the
        # next pass, and a ripple of C no longer re-times the segment.
        self.assertEqual(
            self.pins()[0],
            {
                "crv": [(15.0, b, "start"), (30.0, b, "end"), (35.0, c, "start")],
                "thru": [(30.0, b, "end"), (35.0, c, "start")],
            },
        )
        self.assertEqual(self.seq.calls[-1], ("hold", {"crv": [8.0]}))

    def test_a_seam_the_system_already_holds_is_not_pinned_again(self):
        """Its key steps, so the span it opens is a hold no pin splits."""
        self.seq.ledger.record_step("crv", 12.0, "auto", "auto")
        self.seq.scans = [{"crv": [12.0], "other": [32.0]}]
        self.seq.calls.clear()
        self.seq._enforce_gap_holds()
        self.assertEqual(self.pins(), [{"other": [(35.0, self.sid("C"), "start")]}])

    def test_without_a_scene_nothing_is_pinned(self):
        self.assertEqual(ShotSequencer()._pin_bound_keys({"crv": [(1.0, 0, "")]}), 0)


class TestGapHoldPinsKeepNoMemory(_SequencerTest):
    """Every gap-hold pass offers every unheld seam its pin, reading the scene
    afresh: it keeps no memory of the last pass.

    A memo of settled seams (tried 2026-10-11 for the New Shot cost) went
    stale whenever the keys changed by any path but the sequencer's own: an
    undo put the curve back without its pin while the memo still called the
    seam settled, and the next edit held the gap on past the next shot's
    start.  The cost is the DCC hook's to keep low where there is nothing to
    write (mayatk ``_pinnable_frames``, blendertk ``_pinnable``).
    """

    SEQUENCER = _Recording

    def pins(self):
        return [call[1] for call in self.seq.calls if call[0] == "pin"]

    def enforce(self):
        self.seq.calls.clear()
        self.seq._enforce_gap_holds()
        return self.pins()

    def test_a_second_pass_offers_the_same_pins_again(self):
        self.seq.scans = [{"crv": [8.0], "other": [32.0]}]
        first = self.enforce()
        self.assertEqual(
            first,
            [
                {
                    "crv": [(15.0, self.sid("B"), "start")],
                    "other": [(35.0, self.sid("C"), "start")],
                }
            ],
        )
        self.assertEqual(self.enforce(), first, "nothing remembered between passes")

    def test_a_new_shot_offers_every_unheld_gap(self):
        """The hook decides what an untouched gap needs (nothing, cheaply)."""
        self.seq.scans = [{"crv": [8.0], "other": [32.0, 48.0]}]
        self.enforce()
        self.seq.calls.clear()
        new = self.seq.insert_shot("D", 10, gap=10)
        self.assertEqual(
            self.pins(),
            [
                {
                    "crv": [(15.0, self.sid("B"), "start")],
                    "other": [
                        (35.0, self.sid("C"), "start"),
                        (60.0, new.shot_id, "start"),
                    ],
                }
            ],
        )


class TestRespaceHoldsTailsFirst(_SequencerTest):
    """BACKLOG 2026-10-05 (maintainer 2026-10-10: tails keep HOLDING): a
    respace pinned every bound shape-preservingly, so a shot tail the hold had
    not reached yet was pinned where the curve PLAYED ON toward the next
    shot's first key -- after a respace it played on, after any other op it
    held.  The respace now holds every gap BEFORE its plan pins anything, so
    a held tail is a hold segment the pin declines."""

    SEQUENCER = _Recording

    def test_every_gap_is_held_before_the_plan_is_applied(self):
        self.seq.calls.clear()
        self.seq.respace(gap=7)
        names = [call[0] for call in self.seq.calls]
        applied = names.index("apply_plan")
        self.assertEqual(names[:applied], ["seams", "release", "hold"])
        self.assertIn("hold", names[applied:], "and again on the new seams")


class _SeamScene(_Recording):
    """Records which contiguous seams each whole-shot move leaves open --
    through the plan writer as a DCC's does, and the pivot mover."""

    def _apply_plan(self, plan, retime_gaps=False):
        self.calls.append(("opened", self._opened_seams(plan)))
        super()._apply_plan(plan, retime_gaps)

    def _reconcile_boundaries(self, plan):
        self.calls.append(("opened", self._opened_seams(plan)))
        return super()._reconcile_boundaries(plan)


class TestPendingBounds(_SequencerTest):
    """A seam is pinned where a move LEAVES a gap (BACKLOG 2026-10-07): an
    edit that moves its neighbours before it writes its own new bound opens a
    gap for a moment that its next write closes, and declares where it ends
    (``pending_bounds``) so that moment is no split."""

    SEQUENCER = _SeamScene

    def opened(self):
        return [call[1] for call in self.seq.calls if call[0] == "opened"]

    def flush(self, *names):
        """Make each named shot end where the next one starts."""
        for name in names:
            shots = self.seq.sorted_shots()
            i = next(k for k, s in enumerate(shots) if s.name == name)
            self.seq.store.update_shot(shots[i].shot_id, end=shots[i + 1].start)

    def test_nested_blocks_merge_with_the_outer_side_winning(self):
        seq = self.seq
        with seq.pending_bounds({1: (5.0, None)}):
            with seq.pending_bounds({1: (7.0, 9.0), 2: (1.0, 2.0)}):
                self.assertEqual(
                    seq._pending_bounds_map, {1: (5.0, 9.0), 2: (1.0, 2.0)}
                )
            self.assertEqual(seq._pending_bounds_map, {1: (5.0, None)})
        self.assertEqual(seq._pending_bounds_map, {})

    def test_a_declared_bound_closes_the_ripples_moment_of_gap(self):
        """Move to Shot's expansion: the neighbour ripples, then the bound."""
        self.flush("A")
        a, b = self.sid("A"), self.sid("B")
        plan = ShotPlanner.plan_ripple_downstream(self.seq.store, a, 20.0, 5.0)
        self.assertEqual(self.seq._opened_seams(plan), [(a, b, 15.0, 20.0)])
        with self.seq.pending_bounds({a: (1.0, 20.0)}):
            self.assertEqual(self.seq._opened_seams(plan), [])

    def test_a_slide_splits_only_the_seam_it_really_opens(self):
        """B slides +5 flush with C: C ripples first, then B follows -- the
        B|C seam moves whole and is never split; A|B, flush too, is, from
        the FIRST plan on: C's ripple already re-times a segment crossing
        A|B to a key in C (2026-10-11, a production ramp through 30 shots)."""
        self.flush("A", "B")
        a, b = self.sid("A"), self.sid("B")
        self.seq.calls.clear()
        self.seq.move_shot(b, 20)
        reports = self.opened()
        self.assertEqual(reports[0], [(a, b, 15.0, 20.0)], "before C moves")
        self.assertTrue(all(r in ([], [(a, b, 15.0, 20.0)]) for r in reports))

    def test_a_leading_pad_still_opens_its_head_seam(self):
        """Room at B's head is room A's content does not fill -- reported
        from the first plan of the pad on, B|C never."""
        self.flush("A", "B")
        a, b = self.sid("A"), self.sid("B")
        self.seq.calls.clear()
        self.seq.add_shot_space(b, 5, edge="leading")
        reports = self.opened()
        self.assertEqual(reports[0], [(a, b, 15.0, 20.0)])
        self.assertTrue(all(r in ([], [(a, b, 15.0, 20.0)]) for r in reports))


class _MergeScene(_Recording):
    """Records which bounds each boundary-sample pass resolves."""

    def _reconcile_boundary_keys(self, bounds=None, follow=True):
        self.calls.append(("bound_keys", bounds, follow))
        return 0, 0


class TestMergeRetiresItsInnerBounds(_SequencerTest):
    """Every bound a merge makes interior has its claimed samples retired IN
    PLACE -- the keeper's own end included.

    Bug (found 2026-10-05 with the gap-hold pins): the keeper's end sample
    was left to the reconcile, which FOLLOWED it to the merged end -- across
    every key the merge took in.  Measured in a fresh Blender 5.1: a pin on
    A's end carried 40 frames on reshaped 84 frames of the curve.
    """

    SEQUENCER = _MergeScene

    def test_the_keepers_end_is_retired_in_place_not_followed(self):
        a, b = self.sid("A"), self.sid("B")
        self.seq.calls.clear()
        self.seq.merge_shots([b, a])
        self.assertEqual(
            [call for call in self.seq.calls if call[0] == "bound_keys"],
            [
                ("bound_keys", {b: (None, None)}, True),
                ("bound_keys", {a: (1.0, None)}, True),
                ("bound_keys", None, True),  # the reconcile after it
            ],
        )
        self.assertBounds(("A", 1, 30), ("C", 35, 50))


class _VacateScene(_Recording):
    """Records the content cut and the neighbour pose pins put before it."""

    def _cut_shot_content(self, shot_id):
        self.calls.append(("cut", shot_id))
        return 0

    def _pin_bound_poses(self, objects, window, pins, attr=None, crossing=False):
        self.calls.append(("pose_pins", sorted(objects), window, pins, attr))
        return 0

    def _freeze_left_behind(self, objects, window, attr=None):
        self.calls.append(("freeze", sorted(objects), window, attr))
        return 0


class _ClashScene(_Recording):
    """A scene whose named seams hold two different poses: the DCC hook
    answers a read of one shot's end against another's start (as they are
    when it is asked, before anything moves) by name."""

    def __init__(self, *a, **kw):
        self.clashing = set()  # {(preceding name, following name)}
        self.differ = set()  # {(frame_a, frame_b)} read as two poses
        super().__init__(*a, **kw)

    def _poses_differ(self, pairs, objects=None):
        self.calls.append(("clash_check", list(pairs)))
        out = []
        for a, b in pairs:
            if (a, b) in self.differ:
                out.append(True)
                continue
            ends = {s.name for s in self.store.shots if abs(s.end - a) < 1e-6}
            starts = {s.name for s in self.store.shots if abs(s.start - b) < 1e-6}
            out.append(any((x, y) in self.clashing for x in ends for y in starts))
        return out

    def copy_attribute_keys(self, obj, attr, delta, times=None, window=None):
        self.calls.append(("copy_attr", obj, attr, delta, window))
        return 1


class TestAClosingSeamStaysOneFrameApartWhereItsPosesDiffer(_SequencerTest):
    """The maintainer's rule (2026-10-11) for every op that closes a seam to
    gap 0 -- delete, the scoped gap collapse, respace to 0: a seam closes
    flush where its two poses (keyed or held) agree and stays exactly one
    frame apart where they differ, never refused and never re-timing a
    shot.  Replaces the respace's refusal (``ShotBoundaryConflict``) and the
    delete's silent hand-over of the next shot's opening (BACKLOG
    2026-10-11, S1 and S2)."""

    SEQUENCER = _ClashScene

    def test_respace_to_zero_leaves_a_disagreeing_seam_one_frame_apart(self):
        self.seq.clashing = {("A", "B")}
        self.seq.respace(gap=0, start_frame=1)
        self.assertBounds(("A", 1, 10), ("B", 11, 26), ("C", 26, 41))

    def test_an_agreeing_seam_closes_flush(self):
        self.seq.respace(gap=0, start_frame=1)
        self.assertBounds(("A", 1, 10), ("B", 10, 25), ("C", 25, 40))

    def test_respacing_twice_is_idempotent(self):
        self.seq.clashing = {("A", "B")}
        self.seq.respace(gap=0, start_frame=1)
        self.seq.respace(gap=0, start_frame=1)
        self.assertBounds(("A", 1, 10), ("B", 11, 26), ("C", 26, 41))
        self.seq.clashing = set()  # the poses now agree: it closes flush
        self.seq.respace(gap=0, start_frame=1)
        self.assertBounds(("A", 1, 10), ("B", 10, 25), ("C", 25, 40))

    def test_a_delete_closing_onto_a_disagreeing_seam_keeps_one_frame(self):
        seq = _ClashScene()
        for name, (start, end) in zip("PQR", ((0, 50), (50, 100), (100, 150))):
            seq.define_shot(name, start, end, objects=[name.lower()])
        seq.clashing = {("P", "R")}
        seq.delete_shot(seq.shot_by_name("Q").shot_id)
        self.assertEqual(
            [(s.name, s.start, s.end) for s in seq.sorted_shots()],
            [("P", 0, 50), ("R", 51, 101)],
        )

    def test_a_scoped_collapse_keeps_one_frame_where_the_poses_differ(self):
        self.seq.clashing = {("A", "B")}
        self.seq.apply_gap(0, scope="end", shot_id=self.sid("A"))
        self.assertBounds(("A", 1, 10), ("B", 11, 26), ("C", 31, 46))

    def test_the_seams_kept_as_straight_cuts_are_reported(self):
        """The maintainer's words (2026-10-12): a seam kept a frame apart is
        a straight cut -- no empty frame between the shots, the gap hold
        stepping the transition."""
        self.seq.clashing = {("A", "B")}
        with self.assertLogs(
            "pythontk.core_utils.engines.shots.shot_sequencer", "INFO"
        ) as logs:
            self.seq.respace(gap=0, start_frame=1)
        self.assertEqual(len(logs.records), 1)
        self.assertIn(
            "1 seam kept as a straight cut (poses differ): A|B", logs.output[0]
        )

    def test_a_band_move_lands_a_frame_short_where_the_poses_differ(self):
        """A body drag is ``move_shot``: B dropped onto A's end."""
        self.seq.clashing = {("A", "B")}
        self.seq.move_shot(self.sid("B"), 10)
        self.assertBounds(("A", 1, 10), ("B", 11, 26), ("C", 31, 46))

    def test_a_band_move_lands_flush_where_the_poses_agree(self):
        self.seq.move_shot(self.sid("B"), 10)
        self.assertBounds(("A", 1, 10), ("B", 10, 25), ("C", 30, 45))

    def test_a_slide_stops_a_frame_short_of_either_neighbour(self):
        """A gap handle's plain drag: the shot alone, clamped by both sides."""
        self.seq.clashing = {("A", "B"), ("B", "C")}
        self.seq.slide_shot(self.sid("B"), 5, direction=None)
        self.assertBounds(("A", 1, 10), ("B", 11, 26), ("C", 35, 50))
        self.seq.slide_shot(self.sid("B"), 40, direction=None)
        self.assertBounds(("A", 1, 10), ("B", 19, 34), ("C", 35, 50))

    def test_a_slide_lands_flush_where_the_poses_agree(self):
        self.seq.slide_shot(self.sid("B"), 20, direction=None)
        self.assertBounds(("A", 1, 10), ("B", 20, 35), ("C", 35, 50))

    def test_a_ctrl_edge_stops_a_frame_short_where_the_poses_differ(self):
        self.seq.clashing = {("A", "B"), ("B", "C")}
        self.seq.set_shot_edge(self.sid("A"), new_end=15, push=True)
        self.seq.set_shot_edge(self.sid("C"), new_start=30, push=True)
        self.assertBounds(("A", 1, 14), ("B", 15, 30), ("C", 31, 50))

    def test_a_ctrl_edge_lands_flush_where_the_poses_agree(self):
        self.seq.set_shot_edge(self.sid("A"), new_end=15, push=True)
        self.seq.set_shot_edge(self.sid("C"), new_start=30, push=True)
        self.assertBounds(("A", 1, 15), ("B", 15, 30), ("C", 30, 50))

    def test_a_ctrl_edge_pushed_past_the_neighbour_is_no_closing(self):
        """The frames a push takes are the neighbour's content, one pose
        each: the neighbour's facing bound gives way to the edge, as before."""
        self.seq.clashing = {("A", "B")}
        self.seq.set_shot_edge(self.sid("A"), new_end=20, push=True)
        self.assertBounds(("A", 1, 20), ("B", 20, 30), ("C", 35, 50))

    def test_a_plan_leaving_no_seam_flush_reads_no_poses(self):
        """Only a flush seam can put two poses on one frame, so a plan that
        leaves none asks the DCC nothing (each ask reads every curve the plan
        moves: a second of a production delete)."""
        self.seq.clashing = {("A", "B")}
        self.seq.calls.clear()
        self.seq.respace(gap=10, start_frame=1)
        self.assertNotIn("clash_check", [c[0] for c in self.seq.calls])
        self.seq.respace(gap=0, start_frame=1)
        self.assertIn("clash_check", [c[0] for c in self.seq.calls])


class TestMoveToShotCopiesASharedSample(unittest.TestCase):
    """Move to Shot copies, not carries, a run's sample on a FLUSH seam of
    its shot: it is the neighbour's pose too (the fencepost gives the shot
    before the shared frame).  Bug (2026-10-11, SOL_REPLACE_ASSEMBLY gap 0):
    "B12"'s last run opened on B11's end, the curve's first key; carried
    away, every shot before played the new first key's extrapolation --
    SOLENOID 4.85 off over 1734 frames."""

    def setUp(self):
        self._prefs_tmp = tempfile.mkdtemp(prefix="shot_sequencer_prefs_")
        ShotStore._prefs_dir_override = self._prefs_tmp
        ShotStore._active = None
        ShotStore._persistence = None
        self.seq = _ClashScene()
        for name, (start, end) in zip(
            "PQRS", ((0, 50), (50, 100), (100, 150), (160, 200))
        ):
            self.seq.define_shot(name, start, end, objects=[name.lower()])

    def tearDown(self):
        ShotStore._prefs_dir_override = None
        ShotStore._active = None
        shutil.rmtree(self._prefs_tmp, ignore_errors=True)

    def _send(self, run, dest):
        self.seq.times = [run["start"], run["end"]]
        self.seq.move_sequences_to_shot([run], self.seq.shot_by_name(dest).shot_id)
        return [c for c in self.seq.calls if c[0] == "copy_attr"]

    def test_a_run_opening_on_a_flush_start_leaves_a_copy(self):
        got = self._send({"kind": "anim", "obj": "q", "start": 50.0, "end": 80.0}, "S")
        # Landed at S's start (nothing of q's there): the first key goes back.
        self.assertEqual(got, [("copy_attr", "q", None, -110.0, (160.0, 160.0))])

    def test_a_run_closing_on_a_flush_end_leaves_a_copy(self):
        got = self._send({"kind": "anim", "obj": "q", "start": 70.0, "end": 100.0}, "S")
        self.assertEqual(got, [("copy_attr", "q", None, -90.0, (190.0, 190.0))])

    def test_a_run_off_the_seams_is_carried_whole(self):
        got = self._send({"kind": "anim", "obj": "q", "start": 60.0, "end": 80.0}, "S")
        self.assertEqual(got, [])

    def test_a_head_landing_keeps_a_frame_where_the_poses_differ(self):
        """P's run lands in front of Q, flush behind P: the block owns Q's
        start (maintainer 2026-09-23), a frame after P's end where P closes
        on a different pose (the closing rule, 2026-10-11)."""
        self.seq.differ = {(50.0, 10.0)}  # P's end vs the block's first key
        self.seq.times = [10.0, 30.0]
        self.seq.move_sequences_to_shot(
            [{"kind": "anim", "obj": "p", "start": 10.0, "end": 30.0}],
            self.seq.shot_by_name("Q").shot_id,
        )
        q = self.seq.shot_by_name("Q")
        self.assertEqual((q.start, q.end), (51.0, 101.0))
        moved = [c for c in self.seq.calls if c[0] == "move_attr" and c[1] == "p"]
        self.assertEqual(moved[-1][3], 41.0, "the block lands on Q's new start")

    def test_a_head_landing_is_flush_where_the_poses_agree(self):
        self.seq.times = [10.0, 30.0]
        self.seq.move_sequences_to_shot(
            [{"kind": "anim", "obj": "p", "start": 10.0, "end": 30.0}],
            self.seq.shot_by_name("Q").shot_id,
        )
        self.assertEqual(self.seq.shot_by_name("Q").start, 50.0)

    def test_the_destinations_own_seam_is_the_move_s_to_edit(self):
        """Q's run ending on R's start, sent INTO R: R takes it."""
        got = self._send({"kind": "anim", "obj": "q", "start": 70.0, "end": 100.0}, "R")
        self.assertEqual(got, [])


class _ShrinkScene(_ClashScene):
    """A clash scene that records its pose pins too."""

    def _pin_bound_poses(self, objects, window, pins, attr=None, crossing=False):
        self.calls.append(("pose_pins", window, pins, crossing))
        return 0


class TestAPlainShrinkClosesItsFlushSeam(_SequencerTest):
    """A plain edge shrink (``resize_shot_bounds``) of a shot FLUSH with its
    neighbour pins both poses first -- the shot's on its new bound, the
    neighbour's on the old seam -- and ripples the neighbour onto the new
    bound under the closing rule: flush where they agree, a frame short
    where they differ (BACKLOG 2026-10-11: a ramp crossing the seam keyless
    was compressed between the two shots' keys, SOL "C02_2" 0.096 off)."""

    SEQUENCER = _ShrinkScene

    def setUp(self):
        super().setUp()
        self.seq.shot_by_name("B").start = 10.0  # A [1, 10] | B [10, 30] flush

    def pins(self):
        return [c[2] for c in self.seq.calls if c[0] == "pose_pins"]

    def test_the_next_shot_lands_a_frame_short_where_the_poses_differ(self):
        self.seq.differ = {(6.0, 10.0)}  # A's pose at 6 vs B's at 10
        self.seq.calls.clear()
        self.seq.resize_shot_bounds(self.sid("A"), 1, 6)
        self.assertBounds(("A", 1, 6), ("B", 7, 27), ("C", 32, 47))
        a, b = self.sid("A"), self.sid("B")
        self.assertEqual(self.pins(), [[(6.0, a, "end")], [(10.0, b, "start")]])
        first_move = next(
            i for i, c in enumerate(self.seq.calls) if c[0] == "apply_plan"
        )
        pinned = [i for i, c in enumerate(self.seq.calls) if c[0] == "pose_pins"]
        self.assertLess(pinned[-1], first_move, "pinned before anything moves")

    def test_the_next_shot_lands_flush_where_the_poses_agree(self):
        self.seq.resize_shot_bounds(self.sid("A"), 1, 6)
        self.assertBounds(("A", 1, 6), ("B", 6, 26), ("C", 31, 46))

    def test_a_head_shrink_brings_the_shot_before_on(self):
        self.seq.differ = {(10.0, 14.0)}  # A's pose at 10 vs B's at 14
        self.seq.calls.clear()
        self.seq.resize_shot_bounds(self.sid("B"), 14, 30)
        self.assertBounds(("A", 4, 13), ("B", 14, 30), ("C", 35, 50))
        a, b = self.sid("A"), self.sid("B")
        self.assertEqual(self.pins(), [[(10.0, a, "end")], [(14.0, b, "start")]])

    def test_the_old_seams_claim_is_the_neighbours(self):
        a, b = self.sid("A"), self.sid("B")
        self.seq.ledger.record_key("crv", 10.0, a, "end", stays=True)
        self.seq.differ = {(6.0, 10.0)}
        self.seq.resize_shot_bounds(a, 1, 6)
        self.assertEqual(self.seq.ledger.key_records("crv"), [(10.0, b, "start")])
        self.assertTrue(self.seq.ledger.key_stays("crv", 10.0))

    def test_a_shrink_beside_a_gap_pins_both_sides_and_keeps_the_gap(self):
        """Across a gap nothing closes: the gap keeps its width, and both
        sides of it are pinned (a ramp crossing it keyless was re-timed)."""
        b, c = self.sid("B"), self.sid("C")
        self.seq.calls.clear()
        self.seq.resize_shot_bounds(b, 10, 26)
        self.assertBounds(("A", 1, 10), ("B", 10, 26), ("C", 31, 46))
        self.assertEqual(self.pins(), [[(26.0, b, "end")], [(35.0, c, "start")]])

    def test_a_grow_beside_a_gap_pins_where_the_shot_played_to(self):
        b, c = self.sid("B"), self.sid("C")
        self.seq.calls.clear()
        self.seq.resize_shot_bounds(b, 10, 33)
        self.assertBounds(("A", 1, 10), ("B", 10, 33), ("C", 38, 53))
        self.assertEqual(self.pins(), [[(30.0, b, "end")], [(35.0, c, "start")]])

    def test_a_flush_grow_is_the_splits_to_pin(self):
        self.seq.calls.clear()
        self.seq.resize_shot_bounds(self.sid("A"), 1, 14)
        self.assertEqual(self.pins(), [])


class TestContentLeavingAShotKeepsItsNeighboursPoses(_SequencerTest):
    """Content leaving a shot -- deleted, or sent to another shot -- first
    pins the pose each shot BESIDE it plays at its facing bound, so what the
    removed keys gave those shots (a held last value, the ramp into or out
    of them) stays theirs.

    Bug (found 2026-10-11 by the SOL_REPLACE_ASSEMBLY acceptance, gap 10):
    deleting a shot whose last key held a curve's value for the rest of the
    timeline left the next shot playing the shot BEFORE's value instead --
    KEYS 78 units off over all 81 frames of "A02", SOLENOID 7.2 over 256 of
    "C02"; Move to Shot of a run out of "A01_2" did the same to "A02"."""

    SEQUENCER = _VacateScene

    def pins(self):
        return [c for c in self.seq.calls if c[0] in ("pose_pins", "cut")]

    def test_a_delete_pins_both_neighbours_before_the_cut(self):
        a, b, c = self.sid("A"), self.sid("B"), self.sid("C")
        self.seq.calls.clear()
        self.seq.delete_shot(b)
        self.assertEqual(
            self.pins(),
            [
                (
                    "pose_pins",
                    ["a", "b", "c", "stray"],  # every curve the closing re-times
                    (14.999, 34.999),  # the cut's window: 35 is C's
                    [(10.0, a, "end"), (35.0, c, "start")],
                    None,
                ),
                ("cut", b),
            ],
        )

    def test_a_delete_keeping_its_content_pins_nothing(self):
        self.seq.calls.clear()
        self.seq.delete_shot(self.sid("B"), delete_contents=False)
        self.assertEqual(self.pins(), [])

    def test_a_retime_pins_its_flush_seams_first(self):
        """A Shift retime holds what a FLUSH neighbour plays on the moved
        edge: every content curve is pinned on that seam (keyless crossings
        too), claimed as the earlier shot's end, before anything moves
        (BACKLOG 2026-10-11: keyless, the next shot opened 0.08 off)."""
        b = self.seq.shot_by_name("B")
        b.start, b.end = 10.0, 35.0  # A|B and B|C flush
        self.seq.calls.clear()
        self.seq.resize_shot(self.sid("B"), 5, 40)
        pins = [p[3] for p in self.pins()]
        self.assertEqual(
            pins, [[(35.0, self.sid("B"), "end")], [(10.0, self.sid("A"), "end")]]
        )
        first_move = next(
            i for i, x in enumerate(self.seq.calls) if x[0] in ("apply_plan", "cut")
        )
        self.assertLess(self.seq.calls.index(self.pins()[-1]), first_move)

    def test_a_delete_pins_its_flush_neighbours_too(self):
        """At gap 0 the bounds a deleted shot shares are its neighbours' once
        it is gone (maintainer 2026-10-11): both are pinned, and where the
        next shot lands is the closing rule's (it was left unpinned, and
        "A02" played the deleted shot's closing pose, KEYS 80 off)."""
        seq = _VacateScene()
        for name, (start, end) in zip("PQR", ((0, 50), (50, 100), (100, 150))):
            seq.define_shot(name, start, end, objects=[name.lower()])
        p, q, r = (seq.shot_by_name(n).shot_id for n in "PQR")
        seq.calls.clear()
        seq.delete_shot(q)
        calls = [c for c in seq.calls if c[0] in ("pose_pins", "cut")]
        self.assertEqual([c[0] for c in calls], ["pose_pins", "cut"])
        self.assertEqual(calls[0][3], [(50.0, p, "end"), (100.0, r, "start")])

    def test_a_next_shot_one_gap_away_is_pinned_whatever_closes(self):
        seq = _VacateScene()
        for name, (start, end) in zip("PQR", ((0, 50), (50, 100), (110, 150))):
            seq.define_shot(name, start, end, objects=[name.lower()])
        p, q, r = (seq.shot_by_name(n).shot_id for n in "PQR")
        seq.calls.clear()
        seq.delete_shot(q)
        self.assertEqual(
            [c[3] for c in seq.calls if c[0] == "pose_pins"],
            [[(50.0, p, "end"), (110.0, r, "start")]],
        )

    def test_move_to_shot_pins_the_source_neighbour_that_is_not_the_destination(
        self,
    ):
        a, c = self.sid("A"), self.sid("C")
        self.seq.calls.clear()
        self.seq.move_sequences_to_shot(
            [{"kind": "anim", "obj": "b", "start": 17.0, "end": 28.0}], a
        )
        pins = [x for x in self.seq.calls if x[0] == "pose_pins"]
        self.assertEqual(
            pins,
            [("pose_pins", ["b"], (16.999, 28.001), [(35.0, c, "start")], None)],
        )
        first_move = next(
            i
            for i, x in enumerate(self.seq.calls)
            if x[0] in ("move_attr", "apply_plan")
        )
        self.assertLess(self.seq.calls.index(pins[0]), first_move)

    def test_a_copy_leaves_its_source_and_pins_nothing(self):
        self.seq.calls.clear()
        self.seq.copy_sequences_to_shot(
            [{"kind": "anim", "obj": "b", "start": 17.0, "end": 28.0}],
            self.sid("A"),
        )
        self.assertFalse([x for x in self.seq.calls if x[0] in ("pose_pins", "freeze")])

    def test_move_to_shot_holds_what_the_run_leaves_behind_first(self):
        """The keys a run leaves in its source keep the slopes they play
        into their own content (BACKLOG 2026-10-11: the source re-timed up
        to its last stayed key, SOL "A01_2" / "C02")."""
        self.seq.calls.clear()
        self.seq.move_sequences_to_shot(
            [{"kind": "anim", "obj": "b", "start": 17.0, "end": 28.0}], self.sid("A")
        )
        frozen = [x for x in self.seq.calls if x[0] == "freeze"]
        self.assertEqual(frozen, [("freeze", ["b"], (16.999, 28.001), None)])
        first_move = next(
            i
            for i, x in enumerate(self.seq.calls)
            if x[0] in ("move_attr", "apply_plan")
        )
        self.assertLess(self.seq.calls.index(frozen[0]), first_move)


class _PointClipScene(_Recording):
    """The point-clip hooks over a scripted curve list."""

    def __init__(self, *a, **kw):
        self.mark_curves = []
        super().__init__(*a, **kw)

    def _point_clip_curves(self, obj, t, attr_name=None, shot=None, eps=1e-3):
        self.calls.append(("curves", obj, t, attr_name, shot and shot.name))
        return list(self.mark_curves)

    def _move_point_key(self, obj, curve, old_time, new_time, eps=1e-3):
        self.calls.append(("move_key", obj, curve, old_time, new_time))


class TestPointClipDrag(_SequencerTest):
    """A point (stepped) clip is a MARK -- an animator's key on a member that
    holds still in its shot, drawn from its content channels whether the key
    steps or not -- so its drag moves exactly what the mark was drawn from.

    Bug (BACKLOG 2026-10-04): the rule lived in each DCC and they disagreed.
    mayatk moved only keys with a ``step`` out-tangent, so a mark on a plain
    key (a Move to Shot arrival) dragged nothing while the shot still grew to
    enclose the drop; blendertk moved every fcurve keyed on that frame,
    channels the mark never drew included.
    """

    SEQUENCER = _PointClipScene

    def test_the_drag_moves_every_curve_the_mark_was_drawn_from(self):
        self.seq.mark_curves = ["tx", "vis"]
        self.seq.calls.clear()
        self.seq.move_stepped_keys("hold", 20.0, 24.0)
        self.assertEqual(
            self.seq.calls,
            [
                ("curves", "hold", 20.0, None, "B"),  # read in the shot it sits in
                ("move_key", "hold", "tx", 20.0, 24.0),
                ("move_key", "hold", "vis", 20.0, 24.0),
            ],
        )

    def test_a_sub_row_clip_names_its_channel(self):
        self.seq.mark_curves = ["tx"]
        self.seq.calls.clear()
        self.seq.move_stepped_keys("hold", 20.0, 18.0, attr_name="translateX")
        self.assertEqual(self.seq.calls[0], ("curves", "hold", 20.0, "translateX", "B"))

    def test_a_drag_that_goes_nowhere_reads_nothing(self):
        self.seq.mark_curves = ["tx"]
        self.seq.calls.clear()
        self.seq.move_stepped_keys("hold", 20.0, 20.0)
        self.assertEqual(self.seq.calls, [])

    def test_without_a_scene_nothing_moves(self):
        ShotSequencer().move_stepped_keys("hold", 20.0, 24.0)  # no hook, no error


class TestShotEdge(_SequencerTest):
    """``set_shot_edge``: a bound moves and nothing ripples (the Ctrl drag).

    With *push* the edge may cross the neighbour's facing bound, which gives
    way (2026-10-06: Ctrl "would not let me drag to encompass keys in the
    adjacent shot"); a flush neighbour follows a shrink, so the seam moves.
    """

    def test_without_push_the_edge_stops_at_the_neighbour(self):
        self.assertTrue(self.seq.set_shot_edge(self.sid("A"), new_end=20))
        self.assertBounds(("A", 1, 15), ("B", 15, 30), ("C", 35, 50))

    def test_at_a_flush_neighbour_without_push_nothing_moves(self):
        self.seq.set_shot_edge(self.sid("A"), new_end=15)
        self.assertFalse(self.seq.set_shot_edge(self.sid("A"), new_end=22))
        self.assertBounds(("A", 1, 15), ("B", 15, 30), ("C", 35, 50))

    def test_push_crosses_the_neighbour_whose_bound_gives_way(self):
        self.assertTrue(self.seq.set_shot_edge(self.sid("A"), new_end=20, push=True))
        self.assertBounds(("A", 1, 20), ("B", 20, 30), ("C", 35, 50))

    def test_push_leaves_the_neighbour_one_frame(self):
        self.seq.set_shot_edge(self.sid("A"), new_end=45, push=True)
        self.assertBounds(("A", 1, 29), ("B", 29, 30), ("C", 35, 50))

    def test_push_on_the_start_edge(self):
        self.seq.set_shot_edge(self.sid("B"), new_start=5, push=True)
        self.assertBounds(("A", 1, 5), ("B", 5, 30), ("C", 35, 50))

    def test_a_flush_neighbour_follows_a_shrink(self):
        self.seq.set_shot_edge(self.sid("A"), new_end=15)
        self.seq.set_shot_edge(self.sid("A"), new_end=12, push=True)
        self.assertBounds(("A", 1, 12), ("B", 12, 30), ("C", 35, 50))

    def test_a_shrink_beside_a_gap_leaves_the_neighbour(self):
        self.seq.set_shot_edge(self.sid("A"), new_end=6, push=True)
        self.assertBounds(("A", 1, 6), ("B", 15, 30), ("C", 35, 50))

    def test_a_retime_never_pushes(self):
        self.seq.set_shot_edge(self.sid("A"), new_end=20, scale=True, push=True)
        self.assertBounds(("A", 1, 15), ("B", 15, 30), ("C", 35, 50))

    def test_an_edge_past_the_opposite_one_stops_at_zero_duration(self):
        self.seq.set_shot_edge(self.sid("B"), new_end=3)
        self.assertBounds(("A", 1, 10), ("B", 15, 15), ("C", 35, 50))

    def test_both_edges_given_inverted_are_swapped_not_stored_inverted(self):
        """As resize_shot_bounds normalises: start 28 / end 20 is [20, 28]."""
        self.seq.set_shot_edge(self.sid("B"), new_start=28, new_end=20)
        self.assertBounds(("A", 1, 10), ("B", 20, 28), ("C", 35, 50))

    def test_both_edges_clamped_past_each_other_stop_at_zero_duration(self):
        """Bug: a span asked for wholly past a neighbour was clamped one edge
        at a time, and the clamp carried that edge past the other: B stored
        as [10, 8]."""
        self.assertTrue(self.seq.set_shot_edge(self.sid("B"), new_start=3, new_end=8))
        self.assertBounds(("A", 1, 10), ("B", 10, 10), ("C", 35, 50))

    def test_both_edges_clamped_past_the_next_shot_stop_on_its_start(self):
        """Same bug on the far side: B stored as [40, 35], sorted after C."""
        self.seq.set_shot_edge(self.sid("B"), new_start=40, new_end=60)
        self.assertBounds(("A", 1, 10), ("B", 35, 35), ("C", 35, 50))

    def test_both_edges_pushed_past_each_other_stop_at_zero_duration(self):
        """Same bug with *push*: B stored as [2, 1]."""
        self.seq.set_shot_edge(self.sid("B"), new_start=0, new_end=1, push=True)
        self.assertBounds(("A", 1, 2), ("B", 2, 2), ("C", 35, 50))

    def test_a_change_marks_the_store_dirty(self):
        dirty = []
        self.seq.store.mark_dirty = lambda: dirty.append(True)
        self.seq.set_shot_edge(self.sid("A"), new_end=12)
        self.assertTrue(dirty)


class _EdgeScene(_Recording):
    """Objects that MOVE over a frame window each (``movers``)."""

    def __init__(self, *a, **kw):
        self.movers = {}
        super().__init__(*a, **kw)

    def _find_keyed_transforms(self, start, end, **kwargs):
        return sorted(
            name for name, (lo, hi) in self.movers.items() if lo < end and hi > start
        )

    def collect_object_segments(self, shot_id, **kwargs):
        shot = self.shot_by_id(shot_id)
        return [
            {"obj": name}
            for name, (lo, hi) in self.movers.items()
            if name in shot.objects and lo < shot.end and hi > shot.start
        ]


class TestShotEdgeMembers(_SequencerTest):
    """The keys a bound hands over take their objects' membership with them."""

    SEQUENCER = _EdgeScene

    def objects(self, name):
        return sorted(self.seq.shot_by_name(name).objects)

    def test_keyed_transforms_reads_the_hosts_hook(self):
        """The public reader a panel lists a range's movers with is the
        adapter's ``_find_keyed_transforms``, keywords and all."""
        self.seq.movers = {"b": (21, 30), "a": (1, 5)}
        self.assertEqual(self.seq.keyed_transforms(20, 40), ["b"])
        self.assertEqual(self.seq.keyed_transforms(0, 40, evaluated=True), ["a", "b"])

    def test_a_push_adopts_what_moves_in_the_frames_taken(self):
        self.seq.movers = {"b": (21, 30), "b_head": (16, 19)}
        self.seq.store.update_shot(self.sid("B"), objects=["b", "b_head"])
        self.seq.set_shot_edge(self.sid("A"), new_end=20, push=True)
        self.assertEqual(self.objects("A"), ["a", "b_head"])
        self.assertEqual(self.objects("B"), ["b"], "it no longer moves in B")

    def test_a_member_moving_on_both_sides_belongs_to_both(self):
        self.seq.movers = {"b": (15, 30)}
        self.seq.set_shot_edge(self.sid("A"), new_end=20, push=True)
        self.assertEqual(self.objects("A"), ["a", "b"])
        self.assertEqual(self.objects("B"), ["b"])

    def test_a_flush_shrink_hands_the_tail_to_the_neighbour(self):
        self.seq.movers = {"a": (1, 9), "a_tail": (12, 15)}
        self.seq.store.update_shot(self.sid("A"), objects=["a", "a_tail"])
        self.seq.set_shot_edge(self.sid("A"), new_end=15)
        self.seq.set_shot_edge(self.sid("A"), new_end=11, push=True)
        self.assertEqual(self.objects("A"), ["a"])
        self.assertEqual(self.objects("B"), ["a_tail", "b"])

    def test_a_grow_into_the_gap_adopts_what_moves_there(self):
        self.seq.movers = {"lead_out": (11, 13)}
        self.seq.set_shot_edge(self.sid("A"), new_end=14)
        self.assertEqual(self.objects("A"), ["a", "lead_out"])

    def test_a_retime_leaves_membership_alone(self):
        self.seq.movers = {"lead_out": (11, 13)}
        self.seq.set_shot_edge(self.sid("A"), new_end=14, scale=True)
        self.assertEqual(self.objects("A"), ["a"])


class _CopyScene(_Recording):
    """Records the copy hook beside the move hooks."""

    def copy_attribute_keys(self, obj, attr, delta, times=None, window=None):
        self.calls.append(("copy_attr", obj, attr, delta, window))
        return 0


class TestCopySequences(_SequencerTest):
    """``copy_sequences_to_shot``: Move to Shot's placement, the keys copied --
    the source keeps its content and its members, and audio is not copied."""

    SEQUENCER = _CopyScene

    def test_a_copy_lands_where_a_move_would_and_leaves_the_source(self):
        a, c = self.sid("A"), self.sid("C")
        self.seq.calls.clear()
        self.seq.copy_sequences_to_shot(
            [
                {"kind": "anim", "obj": "a", "start": 2, "end": 8},
                {"kind": "audio", "obj": "trk", "start": 2, "end": 8},
            ],
            c,
        )
        writes = ("copy_attr", "move_attr", "move_audio")
        self.assertEqual(
            [call for call in self.seq.calls if call[0] in writes],
            [("copy_attr", "a", None, 33.0, (2, 8))],
        )
        self.assertEqual(self.seq.shot_by_id(a).objects, ["a"], "the source keeps it")
        self.assertEqual(self.seq.shot_by_id(c).objects, ["a", "c"])

    def test_a_copy_of_audio_alone_does_nothing(self):
        self.seq.calls.clear()
        self.seq.copy_sequences_to_shot(
            [{"kind": "audio", "obj": "trk", "start": 2, "end": 8}], self.sid("C")
        )
        self.assertEqual(self.seq.calls, [])


class _ReplaceScene(_VacateScene):
    """Runs that ride their shots (``runs``: ``{shot id: [(obj, from, to)]}``,
    offsets from the shot's start), and the copy hook recorded."""

    def __init__(self, *a, **kw):
        self.runs = {}
        super().__init__(*a, **kw)

    def collect_object_segments(self, shot_id, **kwargs):
        shot = self.shot_by_id(shot_id)
        return [
            {"obj": obj, "start": shot.start + lo, "end": shot.start + hi}
            for obj, lo, hi in self.runs.get(shot_id, ())
        ]

    def copy_attribute_keys(self, obj, attr, delta, times=None, window=None):
        self.calls.append(("copy_attr", obj, attr, delta, window))
        return 1

    def _cut_shot_content(self, shot_id):
        self.runs.pop(shot_id, None)
        return super()._cut_shot_content(shot_id)


class TestCopyShotContent(_SequencerTest):
    """``copy_shot_content``: the destination's animation is replaced by the
    source's -- its own keys cut, the source's length taken, the source's
    runs copied at the same offsets from its start -- and its record stays."""

    SEQUENCER = _ReplaceScene

    def writes(self):
        return [c for c in self.seq.calls if c[0] in ("cut", "copy_attr")]

    def test_a_later_source_replaces_an_earlier_shot(self):
        a, c = self.sid("A"), self.sid("C")
        self.seq.store.update_shot(
            a, description="intro", metadata={"step": "1"}, objects=["a", "doc_a"]
        )
        self.seq.runs = {a: [("a", 0, 9)], c: [("c", 0, 15)]}
        self.seq.calls.clear()
        self.assertEqual(self.seq.copy_shot_content(c, a), 1)
        # A takes C's 15 frames; B and C ripple by 6 -- and the copy reads C
        # where the ripple left it.
        self.assertBounds(("A", 1, 16), ("B", 21, 36), ("C", 41, 56))
        self.assertEqual(
            self.writes(), [("cut", a), ("copy_attr", "c", None, -40.0, (41, 56))]
        )
        shot = self.seq.shot_by_id(a)
        self.assertEqual(
            (shot.name, shot.description, shot.metadata, shot.objects),
            # Its animated member gave way; the manifest's one stayed.
            ("A", "intro", {"step": "1"}, ["c", "doc_a"]),
        )
        self.assertEqual(self.seq.shot_by_id(c).objects, ["c"], "the source keeps it")

    def test_an_earlier_source_replaces_a_later_shot(self):
        a, c = self.sid("A"), self.sid("C")
        self.seq.runs = {a: [("a", 2, 8)]}
        self.seq.calls.clear()
        self.seq.copy_shot_content(a, c)
        self.assertBounds(("A", 1, 10), ("B", 15, 30), ("C", 35, 44))
        self.assertEqual(
            self.writes(), [("cut", c), ("copy_attr", "a", None, 34.0, (3, 9))]
        )

    def test_a_shot_cannot_replace_itself(self):
        with self.assertRaisesRegex(ValueError, "its own"):
            self.seq.copy_shot_content(self.sid("A"), self.sid("A"))
        self.assertEqual(self.writes(), [])


class TestSwapPlaceMembers(_SequencerTest):
    """With metadata retained, a swap leaves the members a shot does not
    animate in the place (they are the step's), and the animated ones travel
    with their keys."""

    SEQUENCER = _Recording

    def test_unanimated_members_stay_with_the_retained_metadata(self):
        store = self.seq.store
        a, c = self.sid("A"), self.sid("C")
        store.update_shot(a, objects=["a", "doc_a"], metadata={"step": "1"})
        store.update_shot(c, objects=["c", "doc_c"], metadata={"step": "3"})
        self.seq.segments = {a: [{"obj": "a"}], c: [{"obj": "c"}]}
        self.seq.swap_shots(a, c, retain=("metadata",))
        shot_a, shot_c = self.seq.shot_by_id(a), self.seq.shot_by_id(c)
        self.assertEqual(
            (shot_a.objects, shot_a.metadata), (["a", "doc_c"], {"step": "3"})
        )
        self.assertEqual(
            (shot_c.objects, shot_c.metadata), (["c", "doc_a"], {"step": "1"})
        )

    def test_a_member_keyed_only_with_holds_travels_with_its_keys(self):
        """A lone flat key (a render-effect off-state) is content: motion-only
        segments would leave its member behind while its key moved."""
        store = self.seq.store
        a, c = self.sid("A"), self.sid("C")
        store.update_shot(a, objects=["a", "flat_a"])
        holds = {a: [{"obj": "flat_a"}]}
        base = self.seq.collect_object_segments

        def segments(shot_id, ignore_holds=True, **kw):
            extra = [] if ignore_holds else holds.get(shot_id, [])
            return base(shot_id, **kw) + extra

        self.seq.collect_object_segments = segments
        self.seq.swap_shots(a, c, retain=("metadata",))
        # flat_a went with its key; the keyless "a" and "c" stayed in place.
        self.assertEqual(self.seq.shot_by_id(a).objects, ["c", "flat_a"])
        self.assertEqual(self.seq.shot_by_id(c).objects, ["a"])

    def test_without_metadata_retained_every_member_travels(self):
        store = self.seq.store
        a, c = self.sid("A"), self.sid("C")
        store.update_shot(a, objects=["a", "doc_a"])
        self.seq.segments = {a: [{"obj": "a"}], c: [{"obj": "c"}]}
        self.seq.swap_shots(a, c, retain=("name",))
        self.assertEqual(self.seq.shot_by_id(a).objects, ["a", "doc_a"])


class TestDescribeKeySelection(unittest.TestCase):
    """The Shot Sequencer footer's line for a key selection (2026-10-10:
    "footer should show key specific info when keys are selected like frame
    range, etc")."""

    describe = staticmethod(ShotSequencer.describe_key_selection)

    def test_no_keys_says_nothing(self):
        self.assertEqual(self.describe([]), "")
        self.assertEqual(self.describe([("|a|Box01", "tx", [])]), "")

    def test_one_key_names_its_frame_and_channel(self):
        self.assertEqual(
            self.describe([("|grp|Box01", "translateX", [120.0])]),
            "1 key · frame 120 · translateX on Box01",
        )

    def test_many_keys_give_the_range_and_how_many_channels_and_objects(self):
        rows = [
            ("Box01", "translateX", [100, 140]),
            ("Box01", "translateY", [100, 140]),
            ("Lid", "rotateZ", [180.5]),
            ("Lid", None, [100]),  # an object-level clip: no channel
        ]
        self.assertEqual(
            self.describe(rows),
            "6 keys · frames 100–180.50 (80.50f, 3 frames keyed) "
            "· 3 channels on 2 objects",
        )

    def test_keys_of_no_named_object_name_none(self):
        self.assertEqual(
            self.describe([("", None, [3, 4])]),
            "2 keys · frames 3–4 (1f)",
        )

    def test_object_level_keys_on_one_object(self):
        self.assertEqual(
            self.describe([("Box01", "", [1, 10])]),
            "2 keys · frames 1–10 (9f) · on Box01",
        )


class TestPastePairs(unittest.TestCase):
    """``ShotSequencer.paste_pairs``: which copied key row each destination
    row of the Shot Sequencer's Ctrl+V receives -- lifted from mayatk's key
    menu so blendertk's pastes the same way (2026-10-11)."""

    pair = staticmethod(ShotSequencer.paste_pairs)

    def test_a_lone_row_goes_on_every_destination(self):
        self.assertEqual(
            self.pair([("A", "tx", "k")], [("B", "ty"), ("C", "tz")]),
            [("B", "ty", "k"), ("C", "tz", "k")],
        )

    def test_a_row_lands_on_the_same_objects_attribute(self):
        rows = [("A", "tx", "ka"), ("B", "tx", "kb"), ("A", "ty", "kay")]
        self.assertEqual(
            self.pair(rows, [("B", "tx"), ("A", "ty")]),
            [("B", "tx", "kb"), ("A", "ty", "kay")],
        )

    def test_other_objects_take_the_copied_ones_in_order_cycling(self):
        rows = [("A", "tx", "ka"), ("B", "tx", "kb")]
        self.assertEqual(
            self.pair(rows, [("C", "tx"), ("D", "tx"), ("E", "tx")]),
            [("C", "tx", "ka"), ("D", "tx", "kb"), ("E", "tx", "ka")],
        )

    def test_an_attribute_not_copied_gets_nothing(self):
        rows = [("A", "tx", "ka"), ("A", "ty", "kay")]
        self.assertEqual(self.pair(rows, [("A", "rz")]), [])

    def test_the_injected_leaf_matches_two_spellings_of_one_object(self):
        rows = [("|grp|A", "tx", "ka"), ("|grp|B", "tx", "kb")]
        leaf = lambda name: name.split("|")[-1]  # noqa: E731
        self.assertEqual(self.pair(rows, [("B", "tx")], leaf=leaf), [("B", "tx", "kb")])
        self.assertEqual(self.pair(rows, [("B", "tx")]), [("B", "tx", "ka")])


if __name__ == "__main__":
    unittest.main()
