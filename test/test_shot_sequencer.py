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

from pythontk import ShotSequencer, ShotStore  # noqa: E402


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
            lambda: self.seq.fit_shot_to_content(999),
            lambda: self.seq.move_object_in_shot(999, "x", 1, 2, 3),
        ):
            with self.assertRaisesRegex(ValueError, "No shot with id 999"):
                call()


class TestTopology(_SequencerTest):
    def test_move_to_position(self):
        self.seq.move_shot_to_position(self.sid("C"), 1)
        self.assertBounds(("C", 1, 16), ("A", 16, 25), ("B", 25, 40))

    def test_insert_after(self):
        block = self.seq.insert_shot("N", 5, after_shot_id=self.sid("A"))
        self.assertEqual((block.name, block.start, block.end), ("N", 10, 15))
        self.assertBounds(("A", 1, 10), ("N", 10, 15), ("B", 20, 35), ("C", 40, 55))

    def test_insert_at_front(self):
        self.seq.insert_shot("N", 4, at_position=1, gap=2)
        self.assertBounds(("N", 1, 5), ("A", 7, 16), ("B", 21, 36), ("C", 41, 56))

    def test_insert_appends_by_default(self):
        self.seq.insert_shot("N", 6)
        self.assertBounds(("A", 1, 10), ("B", 15, 30), ("C", 35, 50), ("N", 50, 56))

    def test_new_shot_takes_the_next_name_and_the_initial_length(self):
        self.seq.store.initial_shot_length = 20
        block = self.seq.new_shot(after_shot_id=self.sid("A"))
        self.assertEqual((block.name, block.objects), ("Shot_4", []))
        self.assertBounds(
            ("A", 1, 10), ("Shot_4", 10, 30), ("B", 35, 50), ("C", 55, 70)
        )

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
            self.seq._source_shot_id_for({"start": 16, "end": 20}), self.sid("B")
        )
        self.assertIsNone(self.seq._source_shot_id_for({"start": 11, "end": 12}))

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
        b = self.sid("B")
        self.seq.ledger.record_key("crv", 15.0, b, "start")
        self.seq.ledger.record_key("thru", 15.0, b, "start")
        self.seq.scans = [{"crv": [8.0, 15.0], "thru": [15.0]}]
        self.seq.calls.clear()
        self.seq._enforce_gap_holds()
        self.assertEqual(self.pins(), [{"crv": [(15.0, b, "start")]}])
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


if __name__ == "__main__":
    unittest.main()
