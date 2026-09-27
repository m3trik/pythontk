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
        return {"s": []}

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


if __name__ == "__main__":
    unittest.main()
