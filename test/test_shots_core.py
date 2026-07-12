# !/usr/bin/python
# coding=utf-8
"""Tests for the DCC-agnostic shot model core (``pythontk.core_utils.engines.shots``).

Pure-Python, DCC-free.  Covers:

- the planner (``shot_plan``) — collision-safe ordering, envelope computation,
  pivot handling, round-trips, and cycle parking (adapted from mayatk's
  ``test_shot_plan.py``; the assertions are the correctness gate and hold
  unchanged in logic);
- :class:`ShotBlock` — duration and ``classify_objects``;
- :class:`ShotStore` — CRUD, observer, gap-locking, snap, ``compute_gap``, and
  ``to_dict`` / ``from_dict`` round-trip;
- the pure detection math — ``cluster_segments_by_gap`` and
  ``boundaries_from_key_entries``;
- the ``shot_apply.apply`` skeleton — bounds-only default plus the three-phase
  writer-backed path (park / ordered / land) and the +INF envelope cap.
"""
import os
import sys
import shutil
import tempfile
import unittest

# Make ``import pythontk`` resolvable when run directly (pytest / unittest) from
# any cwd: the dir two levels up contains the ``pythontk`` package.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pythontk.core_utils.engines.shots import shot_model
from pythontk.core_utils.engines.shots.shot_model import ShotBlock, ShotStore
from pythontk.core_utils.engines.shots.shot_plan import (
    ShotMove,
    plan_respace,
    plan_ripple_downstream,
    plan_ripple_upstream,
    plan_reorder,
)
from pythontk.core_utils.engines.shots.shot_apply import apply
from pythontk.core_utils.engines.shots.shot_detection import (
    STANDARD_TRANSFORM_ATTRS,
    cluster_segments_by_gap,
    boundaries_from_key_entries,
)


class _ShotTest(unittest.TestCase):
    """Base case: isolate the cross-scene prefs file and class state.

    ``mark_dirty`` → ``save`` → ``_save_user_prefs`` would otherwise write to the
    user's real prefs JSON under ``user_config_root``; pointing
    ``ShotStore._prefs_dir_override`` at a temp dir keeps every case sandboxed.
    Class-level singleton state is reset so cases can't leak into one another.
    """

    def setUp(self):
        self._prefs_tmp = tempfile.mkdtemp(prefix="shots_prefs_")
        ShotStore._prefs_dir_override = self._prefs_tmp
        ShotStore._active = None
        ShotStore._persistence = None
        ShotStore._auto_export_disabled = False
        ShotStore._invalidation_listeners = []

    def tearDown(self):
        ShotStore._prefs_dir_override = None
        shutil.rmtree(self._prefs_tmp, ignore_errors=True)
        ShotStore._active = None
        ShotStore._persistence = None
        ShotStore._invalidation_listeners = []


def _store(shots):
    s = ShotStore(list(shots))
    s.snap_whole_frames = False
    return s


# ===========================================================================
# Planner  (adapted from mayatk/test/test_shot_plan.py)
# ===========================================================================


class TestPlanRespace(_ShotTest):
    def test_forward_shift_orders_back_to_front(self):
        """Gap growth shifts shots forward by varying deltas.  Executing
        back-to-front prevents a shot's move range from overlapping the
        still-unmoved next shot's source window."""
        store = _store(
            [
                ShotBlock(1, "A", 0, 10, []),
                ShotBlock(2, "B", 11, 20, []),
                ShotBlock(3, "C", 21, 30, []),
            ]
        )
        plan = plan_respace(store, gap=20, start_frame=1)
        self.assertEqual(plan.sequence, [3, 2, 1])

    def test_backward_shift_orders_front_to_back(self):
        store = _store(
            [
                ShotBlock(1, "A", 10, 20, []),
                ShotBlock(2, "B", 40, 50, []),
                ShotBlock(3, "C", 70, 80, []),
            ]
        )
        plan = plan_respace(store, gap=0, start_frame=0)
        self.assertEqual(plan.sequence, [1, 2, 3])

    def test_non_moving_shots_absent_from_sequence(self):
        """Shots whose new position equals their old position do not
        need to be executed and must be omitted from ``sequence``."""
        store = _store(
            [
                ShotBlock(1, "A", 0, 10, []),
                ShotBlock(2, "B", 20, 30, []),
            ]
        )
        # gap=10, start=0 → A stays, B stays.
        plan = plan_respace(store, gap=10, start_frame=0)
        self.assertEqual(plan.sequence, [])
        self.assertFalse(plan.moves[1].moves)
        self.assertFalse(plan.moves[2].moves)

    def test_envelope_extends_to_next_shot_start(self):
        """Each shot's envelope must cover up to the next shot's start
        so fade tails in the trailing gap travel with the owning shot."""
        store = _store(
            [
                ShotBlock(1, "A", 0, 10, []),
                ShotBlock(2, "B", 20, 30, []),
            ]
        )
        plan = plan_respace(store, gap=5, start_frame=0)
        a = plan.moves[1]
        b = plan.moves[2]
        self.assertEqual(a.env_start, 0)
        self.assertEqual(a.env_end, 20)  # up to B's old start, not A's old end
        self.assertEqual(b.env_start, 20)
        self.assertGreater(b.env_end, 1e8)  # last shot is unbounded

    def test_locked_gap_preserves_width(self):
        """A locked gap must keep its current width when respacing."""
        store = _store(
            [
                ShotBlock(1, "A", 0, 10, []),
                ShotBlock(2, "B", 25, 35, []),  # gap of 15
                ShotBlock(3, "C", 40, 50, []),  # gap of 5
            ]
        )
        store.lock_gap(1, 2)  # preserve 15-frame gap between A and B
        plan = plan_respace(store, gap=0, start_frame=0)
        self.assertAlmostEqual(plan.moves[1].new_start, 0)
        self.assertAlmostEqual(plan.moves[1].new_end, 10)
        self.assertAlmostEqual(plan.moves[2].new_start, 25)  # 10 + 15 (locked)
        self.assertAlmostEqual(plan.moves[3].new_start, 35)  # 25+10 (gap=0)

    def test_empty_store_returns_empty_plan(self):
        plan = plan_respace(_store([]), gap=5, start_frame=0)
        self.assertEqual(plan.moves, {})
        self.assertEqual(plan.sequence, [])

    def test_snap_applied_when_enabled(self):
        """store.snap must round new positions when snap_whole_frames=True."""
        store = ShotStore(
            [
                ShotBlock(1, "A", 0, 10.4, []),
                ShotBlock(2, "B", 15.7, 22.3, []),
            ]
        )
        store.snap_whole_frames = True
        plan = plan_respace(store, gap=3.6, start_frame=0.4)
        self.assertEqual(plan.moves[1].new_start, 0.0)
        # duration not snapped in-place but new_end is
        for m in plan.moves.values():
            self.assertEqual(m.new_start, round(m.new_start))
            self.assertEqual(m.new_end, round(m.new_end))


class TestPlanRipple(_ShotTest):
    def test_downstream_excludes_pivot_and_earlier(self):
        store = _store(
            [
                ShotBlock(1, "A", 0, 10, []),
                ShotBlock(2, "B", 20, 30, []),  # pivot
                ShotBlock(3, "C", 40, 50, []),
                ShotBlock(4, "D", 60, 70, []),
            ]
        )
        plan = plan_ripple_downstream(
            store, pivot_shot_id=2, after_frame=30, delta=5
        )
        self.assertNotIn(1, plan.moves)  # upstream of after_frame
        self.assertNotIn(2, plan.moves)  # pivot excluded
        self.assertIn(3, plan.moves)
        self.assertIn(4, plan.moves)
        # Forward shift → back-to-front order.
        self.assertEqual(plan.sequence, [4, 3])

    def test_upstream_excludes_pivot_and_later(self):
        store = _store(
            [
                ShotBlock(1, "A", 0, 10, []),
                ShotBlock(2, "B", 20, 30, []),
                ShotBlock(3, "C", 40, 50, []),  # pivot
                ShotBlock(4, "D", 60, 70, []),
            ]
        )
        plan = plan_ripple_upstream(
            store, pivot_shot_id=3, before_frame=40, delta=-5
        )
        self.assertIn(1, plan.moves)
        self.assertIn(2, plan.moves)
        self.assertNotIn(3, plan.moves)  # pivot excluded
        self.assertNotIn(4, plan.moves)  # downstream of before_frame
        # Backward shift → front-to-back order.
        self.assertEqual(plan.sequence, [1, 2])

    def test_zero_delta_returns_empty_plan(self):
        store = _store(
            [
                ShotBlock(1, "A", 0, 10, []),
                ShotBlock(2, "B", 20, 30, []),
            ]
        )
        plan = plan_ripple_downstream(store, 1, 10, 0)
        self.assertEqual(plan.moves, {})
        self.assertEqual(plan.sequence, [])


class TestShotMove(_ShotTest):
    def test_moves_flag_ignores_sub_epsilon_deltas(self):
        m = ShotMove(
            shot_id=1,
            old_start=0,
            old_end=10,
            new_start=1e-9,
            new_end=10 + 1e-9,
            env_start=0,
            env_end=20,
        )
        self.assertFalse(m.moves)

    def test_delta_reflects_new_minus_old(self):
        m = ShotMove(1, 5, 15, 8, 18, 5, 20)
        self.assertAlmostEqual(m.delta, 3.0)


class TestRespaceRoundTrip(_ShotTest):
    """Gap up then back down must restore original shot positions."""

    def test_gap_round_trip_restores_positions(self):
        store = _store(
            [
                ShotBlock(1, "A", 0, 10, []),
                ShotBlock(2, "B", 20, 30, []),
                ShotBlock(3, "C", 40, 50, []),
            ]
        )
        orig = {s.shot_id: (s.start, s.end) for s in store.sorted_shots()}

        def _apply(plan):
            for sid in plan.sequence:
                shot = store.shot_by_id(sid)
                m = plan.moves[sid]
                shot.start = m.new_start
                shot.end = m.new_end

        _apply(plan_respace(store, gap=30, start_frame=0))
        _apply(plan_respace(store, gap=10, start_frame=0))

        restored = {s.shot_id: (s.start, s.end) for s in store.sorted_shots()}
        self.assertEqual(restored, orig)


class TestPlanReorder(_ShotTest):
    """`plan_reorder` lifts a shot to a new 1-based position and re-lays the
    whole set out sequentially, reusing the shared collision resolver."""

    def _abc(self):
        return _store(
            [
                ShotBlock(1, "A", 0, 10, []),
                ShotBlock(2, "B", 20, 30, []),
                ShotBlock(3, "C", 40, 50, []),
            ]
        )

    def _apply_bounds(self, store, plan):
        """Commit a plan's new bounds (bounds-only, mirrors headless apply)."""
        for sid in plan.sequence + plan.parked:
            shot = store.shot_by_id(sid)
            m = plan.moves[sid]
            shot.start, shot.end = m.new_start, m.new_end

    def test_move_first_to_last_reorders_and_relays_out(self):
        store = self._abc()  # A[0,10] B[20,30] C[40,50], gap 10
        plan = plan_reorder(store, shot_id=1, target_pos=3, gap=10)
        self._apply_bounds(store, plan)
        order = [(s.name, s.start, s.end) for s in store.sorted_shots()]
        # B first (anchored at old first-start 0), then C, then A appended.
        self.assertEqual(order, [("B", 0, 10), ("C", 20, 30), ("A", 40, 50)])

    def test_move_last_to_first(self):
        store = self._abc()
        plan = plan_reorder(store, shot_id=3, target_pos=1, gap=10)
        self._apply_bounds(store, plan)
        order = [s.name for s in store.sorted_shots()]
        self.assertEqual(order, ["C", "A", "B"])

    def test_durations_preserved(self):
        store = _store(
            [
                ShotBlock(1, "A", 0, 5, []),      # dur 5
                ShotBlock(2, "B", 20, 40, []),    # dur 20
                ShotBlock(3, "C", 50, 58, []),    # dur 8
            ]
        )
        plan = plan_reorder(store, shot_id=1, target_pos=3, gap=10)
        self._apply_bounds(store, plan)
        durs = {s.name: s.duration for s in store.shots}
        self.assertEqual(durs, {"A": 5, "B": 20, "C": 8})

    def test_unchanged_position_is_empty_plan(self):
        plan = plan_reorder(self._abc(), shot_id=2, target_pos=2, gap=10)
        self.assertEqual(plan.moves, {})

    def test_unknown_id_is_empty_plan(self):
        plan = plan_reorder(self._abc(), shot_id=999, target_pos=1, gap=10)
        self.assertEqual(plan.moves, {})

    def test_single_shot_is_empty_plan(self):
        plan = plan_reorder(_store([ShotBlock(1, "A", 0, 10, [])]), 1, 1, 10)
        self.assertEqual(plan.moves, {})

    def test_target_pos_clamped(self):
        store = self._abc()
        # position 99 clamps to last slot -> A ends up last, same as pos 3.
        plan = plan_reorder(store, shot_id=1, target_pos=99, gap=10)
        self._apply_bounds(store, plan)
        self.assertEqual([s.name for s in store.sorted_shots()], ["B", "C", "A"])

    def test_reordered_plan_partition_is_valid(self):
        """The move set must split cleanly into sequence ∪ parked (disjoint)."""
        store = self._abc()
        plan = plan_reorder(store, shot_id=1, target_pos=3, gap=10)
        moving = {sid for sid, m in plan.moves.items() if m.moves}
        self.assertEqual(set(plan.sequence) | set(plan.parked), moving)
        self.assertEqual(set(plan.sequence) & set(plan.parked), set())


class TestRespaceCollisionParking(_ShotTest):
    """Mixed-sign respace deltas form an ordering cycle (a forward mover
    and a backward mover each land in the other's unread envelope).  The
    planner must park the cycle members instead of raising."""

    def _cycle_store(self):
        # gap 30: B moves forward (+29), C moves backward (-21) — the
        # exact reproduction that used to raise "collision cycle".
        return _store(
            [
                ShotBlock(1, "A", 0, 10, []),
                ShotBlock(2, "B", 11, 20, []),
                ShotBlock(3, "C", 100, 110, []),
            ]
        )

    def test_cycle_members_are_parked_not_raised(self):
        plan = plan_respace(self._cycle_store(), gap=30, start_frame=0)
        moving = {sid for sid, m in plan.moves.items() if m.moves}
        self.assertTrue(plan.parked, "cycle members must be parked")
        self.assertEqual(set(plan.sequence) | set(plan.parked), moving)
        self.assertEqual(
            set(plan.sequence) & set(plan.parked), set(), "disjoint partition"
        )
        # The park offset must clear every old/new envelope edge.
        self.assertGreater(plan.park_offset, 110)


# ===========================================================================
# shot_apply.apply skeleton  (pure — writer callables injected)
# ===========================================================================


class TestApplySkeleton(_ShotTest):
    def _cycle_store(self):
        s = ShotStore(
            [
                ShotBlock(1, "A", 0, 10, []),
                ShotBlock(2, "B", 11, 20, []),
                ShotBlock(3, "C", 100, 110, []),
            ]
        )
        s.snap_whole_frames = False
        return s

    def test_bounds_only_commits_final_positions(self):
        """No ``move_keys`` → in-memory bounds-only commit (headless path)."""
        store = self._cycle_store()
        apply(plan_respace(store, gap=30, start_frame=0), store)
        by = {s.name: s for s in store.shots}
        self.assertEqual((by["A"].start, by["A"].end), (0, 10))
        self.assertEqual((by["B"].start, by["B"].end), (40, 49))
        self.assertEqual((by["C"].start, by["C"].end), (79, 89))

    def test_empty_plan_is_noop(self):
        store = _store([ShotBlock(1, "A", 0, 10, [])])
        calls = []
        apply(
            plan_respace(store, gap=0, start_frame=0),
            store,
            move_keys=lambda *a, **k: calls.append(a),
        )
        self.assertEqual(calls, [])

    def test_three_phase_writer_calls_and_inf_cap(self):
        """Parked cycle → phase-0 park + phase-2 land, both ``over=True``, and
        the last shot's +INF envelope must be capped (never propagated raw)."""
        store = self._cycle_store()
        plan = plan_respace(store, gap=30, start_frame=0)
        self.assertEqual(set(plan.parked), {2, 3})  # B and C park
        self.assertEqual(plan.sequence, [])

        key_calls = []
        audio_calls = []

        def move_keys(objects, lo, hi, delta, over=False):
            key_calls.append(
                {"objects": list(objects), "lo": lo, "hi": hi,
                 "delta": delta, "over": over}
            )

        def shift_audio(lo, hi, delta):
            audio_calls.append((lo, hi, delta))

        apply(plan, store, move_keys=move_keys, shift_audio=shift_audio)

        # Bounds still committed.
        by = {s.name: s for s in store.shots}
        self.assertEqual((by["B"].start, by["B"].end), (40, 49))
        self.assertEqual((by["C"].start, by["C"].end), (79, 89))

        # 2 parked shots × (phase 0 + phase 2) = 4 key writes; sequence empty.
        self.assertEqual(len(key_calls), 4)
        self.assertEqual(len(audio_calls), 4)
        self.assertTrue(
            all(c["over"] for c in key_calls),
            "park/land moves must use over=True",
        )
        # INF cap: no upper bound may carry the raw +INF sentinel (1e9).
        self.assertTrue(
            all(c["hi"] < 1e9 / 2 for c in key_calls),
            "the +INF last-shot envelope must be capped when parking",
        )


# ===========================================================================
# ShotBlock
# ===========================================================================


class TestShotBlock(_ShotTest):
    def test_duration(self):
        self.assertEqual(ShotBlock(0, "S", 5, 20).duration, 15)

    def test_classify_all_valid_without_metadata(self):
        shot = ShotBlock(0, "S", 0, 10, objects=["a", "b"])
        self.assertEqual(shot.classify_objects(), {"a": "valid", "b": "valid"})

    def test_classify_uses_object_status(self):
        shot = ShotBlock(
            0, "S", 0, 10, objects=["a", "b"],
            metadata={"object_status": {"a": "missing_object", "b": "valid"}},
        )
        self.assertEqual(
            shot.classify_objects(), {"a": "missing_object", "b": "valid"}
        )

    def test_classify_leaf_name_fallback(self):
        """Status keyed by short name still matches long DAG-path objects."""
        shot = ShotBlock(
            0, "S", 0, 10, objects=["|grp|a", "|grp|b"],
            metadata={"object_status": {"a": "user_animated"}},
        )
        result = shot.classify_objects()
        self.assertEqual(result["|grp|a"], "user_animated")
        self.assertEqual(result["|grp|b"], "valid")

    def test_classify_scene_discovered_for_non_csv(self):
        shot = ShotBlock(
            0, "S", 0, 10, objects=["a", "z"],
            metadata={"csv_objects": ["a"]},
        )
        result = shot.classify_objects()
        self.assertEqual(result["a"], "valid")
        self.assertEqual(result["z"], "scene_discovered")


# ===========================================================================
# ShotStore — CRUD / observer / gap-lock / snap / compute_gap / (de)serialise
# ===========================================================================


class TestShotStoreCrud(_ShotTest):
    def test_define_shot_assigns_incrementing_ids(self):
        s = ShotStore()
        a = s.define_shot("A", 0, 10)
        b = s.define_shot("B", 20, 30)
        self.assertEqual(a.shot_id, 0)
        self.assertEqual(b.shot_id, 1)
        self.assertEqual(len(s.shots), 2)

    def test_define_shot_snaps_and_dedupes_objects(self):
        s = ShotStore()  # snap on by default
        shot = s.define_shot("A", 0.4, 10.6, objects=["b", "a", "a"])
        self.assertEqual(shot.start, 0.0)
        self.assertEqual(shot.end, 11.0)
        self.assertEqual(shot.objects, ["a", "b"])  # sorted, de-duplicated

    def test_update_shot(self):
        s = _store([])
        shot = s.define_shot("A", 0, 10)
        s.update_shot(shot.shot_id, start=5, end=25, name="A2")
        self.assertEqual((shot.start, shot.end, shot.name), (5, 25, "A2"))

    def test_update_shot_inverted_bounds_clamped(self):
        s = _store([])
        shot = s.define_shot("A", 0, 10)
        s.update_shot(shot.shot_id, end=-5)  # end below start
        self.assertEqual(shot.end, shot.start)

    def test_update_missing_shot_returns_none(self):
        self.assertIsNone(ShotStore().update_shot(999, start=1))

    def test_remove_shot(self):
        s = ShotStore()
        shot = s.define_shot("A", 0, 10)
        self.assertTrue(s.remove_shot(shot.shot_id))
        self.assertFalse(s.remove_shot(shot.shot_id))
        self.assertEqual(s.shots, [])

    def test_append_shot_gap_placement(self):
        s = _store([])
        a = s.append_shot("A", duration=10)  # start 0
        b = s.append_shot("B", duration=5, gap=3)  # start a.end + 3 = 13
        self.assertEqual((a.start, a.end), (0, 10))
        self.assertEqual((b.start, b.end), (13, 18))

    def test_lookups(self):
        s = ShotStore()
        a = s.define_shot("A", 0, 10)
        self.assertIs(s.shot_by_id(a.shot_id), a)
        self.assertIs(s.shot_by_name("A"), a)
        self.assertIsNone(s.shot_by_id(42))
        self.assertIsNone(s.shot_by_name("X"))

    def test_sorted_shots(self):
        s = ShotStore()
        s.define_shot("B", 20, 30)
        s.define_shot("A", 0, 10)
        self.assertEqual([sh.name for sh in s.sorted_shots()], ["A", "B"])

    def test_remove_object_from_shots(self):
        s = ShotStore()
        s.define_shot("A", 0, 10, objects=["a", "b"])
        s.define_shot("B", 20, 30, objects=["a"])
        s.set_object_pinned("a")
        s.remove_object_from_shots("a")
        self.assertNotIn("a", s.shots[0].objects)
        self.assertNotIn("a", s.shots[1].objects)
        self.assertNotIn("a", s.pinned_objects)


class TestShotStoreObserver(_ShotTest):
    def test_listener_receives_typed_events(self):
        s = ShotStore()
        events = []
        s.add_listener(events.append)
        shot = s.define_shot("A", 0, 10)
        s.update_shot(shot.shot_id, name="A2")
        s.remove_shot(shot.shot_id)
        self.assertEqual(
            [type(e).__name__ for e in events],
            ["ShotDefined", "ShotUpdated", "ShotRemoved"],
        )

    def test_remove_listener(self):
        s = ShotStore()
        events = []
        s.add_listener(events.append)
        s.remove_listener(events.append)
        s.define_shot("A", 0, 10)
        self.assertEqual(events, [])

    def test_batch_update_coalesces_to_single_batch_complete(self):
        s = ShotStore()
        events = []
        s.add_listener(events.append)
        with s.batch_update():
            s.define_shot("A", 0, 10)
            s.define_shot("B", 20, 30)
        self.assertEqual([type(e).__name__ for e in events], ["BatchComplete"])

    def test_broken_listener_does_not_break_store(self):
        s = ShotStore()
        ok = []

        def boom(evt):
            raise RuntimeError("nope")

        s.add_listener(boom)
        s.add_listener(ok.append)
        # A raising listener must not break the store, but it must be logged
        # (asserting via assertLogs also keeps the warning off the console).
        with self.assertLogs(shot_model._log, level="WARNING"):
            s.define_shot("A", 0, 10)
        self.assertEqual(len(ok), 1)  # the healthy listener still fired

    def test_set_active_shot_fires_once(self):
        s = ShotStore()
        shot = s.define_shot("A", 0, 10)
        events = []
        s.add_listener(events.append)
        s.set_active_shot(shot.shot_id)
        self.assertEqual(s.active_shot_id, shot.shot_id)
        s.set_active_shot(shot.shot_id)  # same id → no second event
        self.assertEqual([type(e).__name__ for e in events], ["ActiveShotChanged"])


class TestShotStoreGapLock(_ShotTest):
    def test_gap_lock_unlock(self):
        s = ShotStore()
        self.assertFalse(s.is_gap_locked(1, 2))
        s.lock_gap(1, 2)
        self.assertTrue(s.is_gap_locked(1, 2))
        s.unlock_gap(1, 2)
        self.assertFalse(s.is_gap_locked(1, 2))

    def test_lock_all_unlock_all(self):
        s = ShotStore()
        a = s.define_shot("A", 0, 10)
        b = s.define_shot("B", 20, 30)
        c = s.define_shot("C", 40, 50)
        s.lock_all_gaps()
        self.assertTrue(s.is_gap_locked(a.shot_id, b.shot_id))
        self.assertTrue(s.is_gap_locked(b.shot_id, c.shot_id))
        s.unlock_all_gaps()
        self.assertEqual(s.locked_gaps, set())


class TestShotStoreDerived(_ShotTest):
    def test_snap_rounds_when_enabled(self):
        s = ShotStore()  # snap on
        self.assertEqual(s.snap(10.4), 10.0)
        self.assertEqual(s.snap(10.6), 11.0)

    def test_snap_passthrough_when_disabled(self):
        s = _store([])
        self.assertEqual(s.snap(10.4), 10.4)

    def test_compute_gap_median(self):
        s = _store([])
        s.define_shot("A", 0, 10)
        s.define_shot("B", 20, 30)  # gap 10
        s.define_shot("C", 50, 60)  # gap 20
        self.assertEqual(s.compute_gap(), 15.0)  # mean of [10, 20]

    def test_compute_gap_fewer_than_two_shots(self):
        s = ShotStore()
        s.gap = 7.0
        s.define_shot("A", 0, 10)
        self.assertEqual(s.compute_gap(), 7.0)


class TestShotStoreSerialisation(_ShotTest):
    def test_to_from_dict_round_trip(self):
        s = _store([])
        s.define_shot(
            "A", 0, 10, objects=["a"], description="first",
            metadata={"section": "intro"},
        )
        s.define_shot("B", 20, 30, objects=["b"], locked=True)
        s.set_object_hidden("a")
        s.set_object_pinned("b")
        s.gap = 5.0
        s.detection_mode = "skip_zero"
        s.detection_threshold = 8.0
        s.lock_gap(0, 1)
        s.source_csv = "x.csv"
        s.clip_name_strategy = "sequence"

        data = s.to_dict()
        restored = ShotStore.from_dict(data)
        # A restored store must serialise back to the identical dict.
        self.assertEqual(restored.to_dict(), data)

    def test_to_export_view_shape(self):
        s = _store([])
        s.define_shot("Intro Shot", 0, 10, objects=["|grp|hero"], description="d")
        view = s.to_export_view()
        self.assertEqual(len(view["fbx_takes"]), 1)
        take = view["fbx_takes"][0]
        self.assertEqual((take["start"], take["end"]), (0, 10))
        # Sanitised, join-key consistent between takes and metadata.
        self.assertEqual(view["shot_metadata"]["shots"][0]["clip"], take["name"])
        # Objects are reduced to leaf names in the metadata channel.
        self.assertEqual(view["shot_metadata"]["shots"][0]["objects"], ["hero"])


class TestUserPrefs(_ShotTest):
    """Cross-scene prefs persist to the zero-dep JSON file (no Qt) and restore."""

    def test_save_restore_round_trip(self):
        s1 = ShotStore([])
        s1.detection_mode = "skip_zero"
        s1.detection_threshold = 12.0
        s1.fit_mode = "fit_contents"
        s1.initial_shot_length = 150.0
        s1.snap_whole_frames = False
        s1.select_on_load = True
        s1._save_user_prefs()

        # A fresh store starts at defaults (ShotStore() does NOT auto-restore —
        # only ShotStore.active() does), then restores the saved values from disk.
        s2 = ShotStore([])
        s2.detection_mode = "auto"  # differs from the saved value
        s2._restore_user_prefs()
        self.assertEqual(s2.detection_mode, "skip_zero")
        self.assertEqual(s2.detection_threshold, 12.0)
        self.assertEqual(s2.fit_mode, "fit_contents")
        self.assertEqual(s2.initial_shot_length, 150.0)
        self.assertFalse(s2.snap_whole_frames)
        self.assertTrue(s2.select_on_load)

    def test_restore_missing_file_is_noop(self):
        # No file written yet -> restore leaves the store's values untouched.
        s = ShotStore([])
        s.detection_mode = "all"
        s.detection_threshold = 7.0
        s._restore_user_prefs()
        self.assertEqual(s.detection_mode, "all")
        self.assertEqual(s.detection_threshold, 7.0)

    def test_prefs_path_uses_override_dir(self):
        # The temp override (set in setUp) keeps prefs out of the real config root.
        path = ShotStore._prefs_path()
        self.assertEqual(str(path.parent), self._prefs_tmp)


class TestStoreHooks(_ShotTest):
    """The overridable DCC hooks resolve with pure defaults on BOTH the class
    and an instance."""

    def test_has_animation_callable_on_class(self):
        # Regression: has_animation is a @staticmethod (mirroring Maya), so a
        # controller's class-level ``ShotStore.has_animation()`` call resolves.
        # An instance-method port would raise TypeError: missing 'self'.
        self.assertFalse(ShotStore.has_animation())
        self.assertFalse(ShotStore([]).has_animation())

    def test_pure_hook_defaults(self):
        s = ShotStore([ShotBlock(1, "A", 0, 10, ["x"])])
        self.assertEqual(s.detect_regions(), [])
        self.assertEqual(s.assess(), {1: "valid"})
        self.assertEqual(s._resolve_long_names(["a", "b"]), ["a", "b"])
        self.assertEqual(s._scene_fps(), s.scene_fps)


# ===========================================================================
# Pure detection math
# ===========================================================================


class TestClusterSegmentsByGap(_ShotTest):
    def test_two_clusters_split_by_gap(self):
        segments = [
            {"start": 0, "end": 10, "obj": "a"},
            {"start": 12, "end": 20, "obj": "b"},  # 12-10=2 <= gap → same cluster
            {"start": 40, "end": 50, "obj": "c"},  # 40-20=20 > gap → new cluster
        ]
        out = cluster_segments_by_gap(segments, gap_threshold=5, min_duration=2)
        self.assertEqual(len(out), 2)
        self.assertEqual((out[0]["start"], out[0]["end"]), (0, 20))
        self.assertEqual(out[0]["objects"], ["a", "b"])
        self.assertEqual(out[0]["name"], "Shot 1")
        self.assertEqual((out[1]["start"], out[1]["end"]), (40, 50))
        self.assertEqual(out[1]["objects"], ["c"])

    def test_min_duration_filter(self):
        segments = [
            {"start": 0, "end": 1, "obj": "a"},  # duration 1 < 2 → dropped
            {"start": 30, "end": 40, "obj": "b"},  # kept
        ]
        out = cluster_segments_by_gap(segments, gap_threshold=5, min_duration=2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["objects"], ["b"])

    def test_empty(self):
        self.assertEqual(cluster_segments_by_gap([]), [])

    def test_does_not_mutate_input(self):
        segments = [
            {"start": 30, "end": 40, "obj": "b"},
            {"start": 0, "end": 10, "obj": "a"},
        ]
        snapshot = [dict(seg) for seg in segments]
        cluster_segments_by_gap(segments)
        self.assertEqual(segments, snapshot)  # sorted a copy, not in place


class TestBoundariesFromKeyEntries(_ShotTest):
    def test_all_mode_contiguous(self):
        entries = [(0, 1, "a"), (20, 1, "b"), (40, 1, "c")]
        out = boundaries_from_key_entries(entries, gap_threshold=5, key_filter="all")
        self.assertEqual(len(out), 3)
        self.assertEqual((out[0]["start"], out[0]["end"]), (0, 20))
        self.assertEqual((out[1]["start"], out[1]["end"]), (20, 40))
        self.assertEqual((out[2]["start"], out[2]["end"]), (40, 41))  # last +1

    def test_all_mode_merges_within_gap(self):
        entries = [(0, 1, "a"), (3, 1, "b"), (40, 1, "c")]  # 0 & 3 merge (<=5)
        out = boundaries_from_key_entries(entries, gap_threshold=5, key_filter="all")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["objects"], ["a", "b"])
        self.assertEqual((out[0]["start"], out[0]["end"]), (0, 40))

    def test_skip_zero_drops_zero_keys(self):
        entries = [(0, 0, "a"), (20, 1, "b"), (40, 0, "c")]
        out = boundaries_from_key_entries(
            entries, gap_threshold=5, key_filter="skip_zero"
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["start"], 20)
        self.assertEqual(out[0]["objects"], ["b"])

    def test_zero_as_end_pairs_starts_with_ends(self):
        entries = [(0, 1, "a"), (10, 0, "a"), (20, 1, "b"), (35, 0, "b")]
        out = boundaries_from_key_entries(entries, key_filter="zero_as_end")
        self.assertEqual(len(out), 2)
        self.assertEqual((out[0]["start"], out[0]["end"]), (0, 10))
        self.assertEqual((out[1]["start"], out[1]["end"]), (20, 35))

    def test_empty(self):
        self.assertEqual(boundaries_from_key_entries([]), [])


class TestDetectionConstants(_ShotTest):
    def test_standard_transform_attrs(self):
        self.assertIn("translateX", STANDARD_TRANSFORM_ATTRS)
        self.assertIn("visibility", STANDARD_TRANSFORM_ATTRS)
        self.assertEqual(len(STANDARD_TRANSFORM_ATTRS), 10)


if __name__ == "__main__":
    unittest.main()
