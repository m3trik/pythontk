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
from pythontk.core_utils.scene_records import SceneRecords
from pythontk.core_utils.engines.shots.shot_plan import ShotPlanner, ShotMove
from pythontk.core_utils.engines.shots.shot_apply import ShotApply
from pythontk.core_utils.engines.shots.shot_detection import (
    ShotDetection,
    STANDARD_TRANSFORM_ATTRS,
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
        plan = ShotPlanner.plan_respace(store, gap=20, start_frame=1)
        self.assertEqual(plan.sequence, [3, 2, 1])

    def test_backward_shift_orders_front_to_back(self):
        store = _store(
            [
                ShotBlock(1, "A", 10, 20, []),
                ShotBlock(2, "B", 40, 50, []),
                ShotBlock(3, "C", 70, 80, []),
            ]
        )
        plan = ShotPlanner.plan_respace(store, gap=0, start_frame=0)
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
        plan = ShotPlanner.plan_respace(store, gap=10, start_frame=0)
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
        plan = ShotPlanner.plan_respace(store, gap=5, start_frame=0)
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
        plan = ShotPlanner.plan_respace(store, gap=0, start_frame=0)
        self.assertAlmostEqual(plan.moves[1].new_start, 0)
        self.assertAlmostEqual(plan.moves[1].new_end, 10)
        self.assertAlmostEqual(plan.moves[2].new_start, 25)  # 10 + 15 (locked)
        self.assertAlmostEqual(plan.moves[3].new_start, 35)  # 25+10 (gap=0)

    def test_empty_store_returns_empty_plan(self):
        plan = ShotPlanner.plan_respace(_store([]), gap=5, start_frame=0)
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
        plan = ShotPlanner.plan_respace(store, gap=3.6, start_frame=0.4)
        self.assertEqual(plan.moves[1].new_start, 0.0)
        # duration not snapped in-place but new_end is
        for m in plan.moves.values():
            self.assertEqual(m.new_start, round(m.new_start))
            self.assertEqual(m.new_end, round(m.new_end))


class TestPlanGapRetimes(_ShotTest):
    """Which gaps change width -- the half of a respace that is NOT a rigid move.

    A shot keeps its duration, so its content can travel with it unchanged. A
    gap's whole purpose in a respace is to change width, so content living in
    one has to be retimed instead. Measured on a 12-shot production assembly,
    carrying it rigidly instead left a gap's own key 13 frames PAST the
    following shot's content, and cost the PRECEDING shot -- which never moved
    -- 42 of its 109 frames to the tangent change that caused.
    """

    def _shots(self):
        # Two 100-frame shots with a 135-frame gap, then a 15-frame one: the
        # shape that produced the report.
        return [
            ShotBlock(shot_id=1, name="A", start=0, end=100),
            ShotBlock(shot_id=2, name="B", start=235, end=335),
            ShotBlock(shot_id=3, name="C", start=350, end=450),
        ]

    def test_a_shrinking_gap_is_reported_with_its_new_width(self):
        store = _store(self._shots())
        plan = ShotPlanner.plan_respace(store, gap=15, start_frame=0)
        gaps = ShotPlanner.plan_gap_retimes(store, plan)
        self.assertEqual([(g.left_id, g.right_id) for g in gaps], [(1, 2)])
        gap = gaps[0]
        self.assertEqual((gap.lo, gap.hi), (100, 235))
        self.assertEqual(gap.new_width, 15)
        self.assertTrue(gap.shrinks)
        self.assertFalse(gap.grows)
        self.assertAlmostEqual(gap.scale, 15.0 / 135.0)

    def test_a_growing_gap_is_distinguished_from_a_shrinking_one(self):
        """They cannot run at the same moment: a scale is only safe while the
        timeline it writes into is empty, and that is before the moves for one
        and after them for the other."""
        store = _store(self._shots())
        plan = ShotPlanner.plan_respace(store, gap=60, start_frame=0)
        by_pair = {
            (g.left_id, g.right_id): g
            for g in ShotPlanner.plan_gap_retimes(store, plan)
        }
        self.assertTrue(by_pair[(1, 2)].shrinks)  # 135 -> 60
        self.assertTrue(by_pair[(2, 3)].grows)  # 15 -> 60

    def test_a_pure_translation_retimes_nothing(self):
        """Every shot moving by the same delta keeps every gap, so a plain slide
        of the whole sequence must not pay for -- or be changed by -- any of
        this."""
        store = _store(
            [
                ShotBlock(shot_id=1, name="A", start=0, end=100),
                ShotBlock(shot_id=2, name="B", start=115, end=215),
                ShotBlock(shot_id=3, name="C", start=230, end=330),
            ]
        )
        plan = ShotPlanner.plan_respace(store, gap=15, start_frame=40)
        self.assertEqual(ShotPlanner.plan_gap_retimes(store, plan), [])

    def test_a_ripple_that_leaves_its_pivot_behind_still_changes_a_gap(self):
        """The pivot is excluded from a ripple by design, so the gap between it
        and the first shot that DID move is stretched by the full delta -- and
        content in that gap has to be retimed exactly as in a respace. Not an
        edge case: it is what a downstream ripple always does."""
        store = _store(self._shots())
        plan = ShotPlanner.plan_ripple_downstream(store, 1, after_frame=0, delta=50)
        gaps = ShotPlanner.plan_gap_retimes(store, plan)
        self.assertEqual([(g.left_id, g.right_id) for g in gaps], [(1, 2)])
        self.assertTrue(gaps[0].grows)
        self.assertEqual(gaps[0].new_width, 185)
        self.assertEqual(gaps[0].left_delta, 0.0)

    def test_the_left_delta_is_how_far_the_gap_itself_travels(self):
        """The gap's content rides the PRECEDING shot, so its left edge moves by
        that shot's delta -- which is where the post-move scale is anchored."""
        store = _store(self._shots())
        plan = ShotPlanner.plan_respace(store, gap=15, start_frame=50)
        gap = ShotPlanner.plan_gap_retimes(store, plan)[0]
        self.assertEqual(gap.left_delta, 50)

    def test_a_collapsed_gap_scales_to_zero_rather_than_dividing(self):
        store = _store(self._shots())
        plan = ShotPlanner.plan_respace(store, gap=0, start_frame=0)
        for gap in ShotPlanner.plan_gap_retimes(store, plan):
            self.assertEqual(gap.new_width, 0)
            self.assertEqual(gap.scale, 0.0)

    def test_contiguous_shots_have_no_gap_to_retime(self):
        store = _store(
            [
                ShotBlock(shot_id=1, name="A", start=0, end=100),
                ShotBlock(shot_id=2, name="B", start=100, end=200),
            ]
        )
        plan = ShotPlanner.plan_respace(store, gap=30, start_frame=0)
        self.assertEqual(ShotPlanner.plan_gap_retimes(store, plan), [])


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
        plan = ShotPlanner.plan_ripple_downstream(
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
        plan = ShotPlanner.plan_ripple_upstream(
            store, pivot_shot_id=3, before_frame=40, delta=-5
        )
        self.assertIn(1, plan.moves)
        self.assertIn(2, plan.moves)
        self.assertNotIn(3, plan.moves)  # pivot excluded
        self.assertNotIn(4, plan.moves)  # downstream of before_frame
        # Backward shift → front-to-back order.
        self.assertEqual(plan.sequence, [1, 2])

    def test_carry_gap_hands_the_pivots_trailing_gap_to_the_first_moved_shot(self):
        """A bound change ripples EVERYTHING beyond the bound -- the content
        parked in the gap after the pivot included (the user's rule, 2026-09-06:
        "only the current shot changes, everything else ripples").  Without it
        a fade tail in that gap stayed put while the next shot moved, and a
        grow swallowed it or a shrink landed the neighbour on it.  The pivot's
        own bound sample stays: the window opens just past ``after_frame``."""
        store = _store(
            [
                ShotBlock(1, "A", 0, 10, []),  # pivot
                ShotBlock(2, "B", 20, 30, []),
                ShotBlock(3, "C", 40, 50, []),
            ]
        )
        plan = ShotPlanner.plan_ripple_downstream(store, 1, 10, 5, carry_gap=True)
        b, c = plan.moves[2], plan.moves[3]
        self.assertEqual(b.env_start, 10, "B's window opens at the pivot's bound")
        self.assertTrue(b.env_lo_open, "...but the bound sample itself is A's")
        self.assertEqual((c.env_start, c.env_lo_open), (40, False), "only the first")

    def test_the_default_ripple_moves_shots_alone(self):
        """A whole-shot move carries its own trailing gap, so the ripple it
        asks for must not carry it too (the keys would move twice)."""
        store = _store([ShotBlock(1, "A", 0, 10, []), ShotBlock(2, "B", 20, 30, [])])
        plan = ShotPlanner.plan_ripple_downstream(store, 1, 10, 5)
        self.assertEqual(
            (plan.moves[2].env_start, plan.moves[2].env_lo_open), (20, False)
        )

    def test_carry_gap_never_cuts_into_the_shot_it_moves(self):
        """A bound moved ONTO the next shot's start leaves that shot its own
        opening sample.  The key-drag handlers ripple BEFORE the dragged keys
        land, so the sample on that frame is the NEIGHBOUR's opening pose, not
        the pivot's: a window opening just past it stranded the pose in the
        gap, one frame past the landed key, with the neighbour opening on
        nothing (measured 2026-09-22).  The carried window trims the gap and
        never the shot it moves."""
        store = _store([ShotBlock(1, "A", 0, 10, []), ShotBlock(2, "B", 20, 30, [])])
        plan = ShotPlanner.plan_ripple_downstream(store, 1, 20, 5, carry_gap=True)
        self.assertEqual(
            (plan.moves[2].env_start, plan.moves[2].env_lo_open), (20, False)
        )
        # Short of the neighbour it still trims the gap: the sample on the
        # bound stays with the pivot and the rest of the gap rides.
        gap = ShotPlanner.plan_ripple_downstream(store, 1, 15, 5, carry_gap=True)
        self.assertEqual((gap.moves[2].env_start, gap.moves[2].env_lo_open), (15, True))

    def test_carry_gap_upstream_caps_the_last_window_at_the_pivots_bound(self):
        store = _store(
            [
                ShotBlock(1, "A", 0, 10, []),
                ShotBlock(2, "B", 20, 30, []),
                ShotBlock(3, "C", 40, 50, []),  # pivot
            ]
        )
        plan = ShotPlanner.plan_ripple_upstream(store, 3, 35, -5, carry_gap=True)
        self.assertEqual(
            (plan.moves[2].env_end, plan.moves[2].env_hi_closed), (35, False)
        )
        self.assertEqual(
            plan.moves[1].env_end, 20, "only the last moved shot is capped"
        )
        plain = ShotPlanner.plan_ripple_upstream(store, 3, 35, -5)
        self.assertEqual(plain.moves[2].env_end, 40, "the default reaches the pivot")

    def test_carry_gap_upstream_never_cuts_into_the_shot_it_moves(self):
        """The mirror: a key dragged onto the PREVIOUS shot's keyed end grows
        the pivot's head to that end, and capping the window there cut the
        neighbour's own closing sample off -- stranded inside the gap it left
        (measured 2026-09-22).  Only a bound in the gap caps the window."""
        store = _store([ShotBlock(1, "P", 0, 10, []), ShotBlock(2, "A", 20, 30, [])])
        onto = ShotPlanner.plan_ripple_upstream(store, 2, 10, -10, carry_gap=True)
        self.assertEqual(
            (onto.moves[1].env_end, onto.moves[1].env_hi_closed), (20, False)
        )
        inside = ShotPlanner.plan_ripple_upstream(store, 2, 5, -15, carry_gap=True)
        self.assertEqual(inside.moves[1].env_end, 20, "into the shot: still whole")

    def test_a_bound_dragged_into_the_neighbour_still_ripples_it(self):
        """A key dragged INSIDE the next shot grows the pivot past that shot's
        start; the neighbour is downstream by ORDER and must still move, and it
        moves WHOLE.  The carried window opening at the new bound stripped the
        keys the bound now covered off it -- the neighbour played torn.  The
        drag grammar keeps neighbours intact (keeping covered keys is Ctrl's
        job), and the sample on a CONTIGUOUS start stays the pivot's
        fencepost."""
        store = _store([ShotBlock(1, "A", 0, 30, []), ShotBlock(2, "B", 30, 60, [])])
        plan = ShotPlanner.plan_ripple_downstream(store, 1, 35, 5, carry_gap=True)
        b = plan.moves[2]
        self.assertEqual((b.new_start, b.env_start, b.env_lo_open), (35, 30, True))
        gapped = _store([ShotBlock(1, "A", 0, 20, []), ShotBlock(2, "B", 30, 60, [])])
        plan = ShotPlanner.plan_ripple_downstream(gapped, 1, 35, 15, carry_gap=True)
        b = plan.moves[2]
        self.assertEqual((b.new_start, b.env_start, b.env_lo_open), (45, 30, False))

    def test_insert_and_delete_ripple_without_carry(self):
        """Insert and delete ripple from a shot's own start with pivot -1 and
        no carry, so the sample on that start moves with its shot as before.
        Carry cannot change that: it trims a gap, never the shot it moves."""
        store = _store([ShotBlock(1, "A", 0, 10, []), ShotBlock(2, "B", 20, 30, [])])
        plain = ShotPlanner.plan_ripple_downstream(store, -1, 20, 5)
        self.assertEqual(
            (plain.moves[2].env_start, plain.moves[2].env_lo_open), (20, False)
        )
        carried = ShotPlanner.plan_ripple_downstream(store, -1, 20, 5, carry_gap=True)
        self.assertEqual(
            (carried.moves[2].env_start, carried.moves[2].env_lo_open), (20, False)
        )

    def test_zero_delta_returns_empty_plan(self):
        store = _store(
            [
                ShotBlock(1, "A", 0, 10, []),
                ShotBlock(2, "B", 20, 30, []),
            ]
        )
        plan = ShotPlanner.plan_ripple_downstream(store, 1, 10, 0)
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

        _apply(ShotPlanner.plan_respace(store, gap=30, start_frame=0))
        _apply(ShotPlanner.plan_respace(store, gap=10, start_frame=0))

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
        plan = ShotPlanner.plan_reorder(store, shot_id=1, target_pos=3, gap=10)
        self._apply_bounds(store, plan)
        order = [(s.name, s.start, s.end) for s in store.sorted_shots()]
        # B first (anchored at old first-start 0), then C, then A appended.
        self.assertEqual(order, [("B", 0, 10), ("C", 20, 30), ("A", 40, 50)])

    def test_move_last_to_first(self):
        store = self._abc()
        plan = ShotPlanner.plan_reorder(store, shot_id=3, target_pos=1, gap=10)
        self._apply_bounds(store, plan)
        order = [s.name for s in store.sorted_shots()]
        self.assertEqual(order, ["C", "A", "B"])

    def test_durations_preserved(self):
        store = _store(
            [
                ShotBlock(1, "A", 0, 5, []),  # dur 5
                ShotBlock(2, "B", 20, 40, []),  # dur 20
                ShotBlock(3, "C", 50, 58, []),  # dur 8
            ]
        )
        plan = ShotPlanner.plan_reorder(store, shot_id=1, target_pos=3, gap=10)
        self._apply_bounds(store, plan)
        durs = {s.name: s.duration for s in store.shots}
        self.assertEqual(durs, {"A": 5, "B": 20, "C": 8})

    def test_unchanged_position_is_empty_plan(self):
        plan = ShotPlanner.plan_reorder(self._abc(), shot_id=2, target_pos=2, gap=10)
        self.assertEqual(plan.moves, {})

    def test_unknown_id_is_empty_plan(self):
        plan = ShotPlanner.plan_reorder(self._abc(), shot_id=999, target_pos=1, gap=10)
        self.assertEqual(plan.moves, {})

    def test_single_shot_is_empty_plan(self):
        plan = ShotPlanner.plan_reorder(
            _store([ShotBlock(1, "A", 0, 10, [])]), 1, 1, 10
        )
        self.assertEqual(plan.moves, {})

    def test_target_pos_clamped(self):
        store = self._abc()
        # position 99 clamps to last slot -> A ends up last, same as pos 3.
        plan = ShotPlanner.plan_reorder(store, shot_id=1, target_pos=99, gap=10)
        self._apply_bounds(store, plan)
        self.assertEqual([s.name for s in store.sorted_shots()], ["B", "C", "A"])

    def test_reordered_plan_partition_is_valid(self):
        """The move set must split cleanly into sequence ∪ parked (disjoint)."""
        store = self._abc()
        plan = ShotPlanner.plan_reorder(store, shot_id=1, target_pos=3, gap=10)
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
        plan = ShotPlanner.plan_respace(self._cycle_store(), gap=30, start_frame=0)
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
        ShotApply.apply(ShotPlanner.plan_respace(store, gap=30, start_frame=0), store)
        by = {s.name: s for s in store.shots}
        self.assertEqual((by["A"].start, by["A"].end), (0, 10))
        self.assertEqual((by["B"].start, by["B"].end), (40, 49))
        self.assertEqual((by["C"].start, by["C"].end), (79, 89))

    def test_empty_plan_is_noop(self):
        store = _store([ShotBlock(1, "A", 0, 10, [])])
        calls = []
        ShotApply.apply(
            ShotPlanner.plan_respace(store, gap=0, start_frame=0),
            store,
            move_keys=lambda *a, **k: calls.append(a),
        )
        self.assertEqual(calls, [])

    def test_objects_for_replaces_the_member_list_in_every_phase(self):
        """A host hands over its keyed CONTENT per envelope; the writer never
        sees the shot's own list then.  Membership stays a label."""
        store = self._cycle_store()
        for s in store.shots:
            s.objects = [f"member_of_{s.name}"]
        plan = ShotPlanner.plan_respace(store, gap=30, start_frame=0)
        seen = []
        ShotApply.apply(
            plan,
            store,
            move_keys=lambda objects, *a, **k: seen.append(sorted(objects)),
            objects_for=lambda sid: ["everything", f"keyed_in_{sid}"],
        )
        self.assertEqual(len(seen), 4)
        self.assertTrue(
            all(
                "everything" in objs and not any("member_of" in o for o in objs)
                for objs in seen
            ),
            seen,
        )
        self.assertEqual(
            {o for objs in seen for o in objs if o.startswith("keyed")},
            {"keyed_in_2", "keyed_in_3"},
        )

    def test_three_phase_writer_calls_and_inf_cap(self):
        """Parked cycle → phase-0 park + phase-2 land, both ``over=True``, and
        the last shot's +INF envelope must be capped (never propagated raw)."""
        store = self._cycle_store()
        plan = ShotPlanner.plan_respace(store, gap=30, start_frame=0)
        self.assertEqual(set(plan.parked), {2, 3})  # B and C park
        self.assertEqual(plan.sequence, [])

        key_calls = []
        audio_calls = []

        def move_keys(objects, lo, hi, delta, over=False, **window):
            key_calls.append(
                {
                    "objects": list(objects),
                    "lo": lo,
                    "hi": hi,
                    "delta": delta,
                    "over": over,
                }
            )

        def shift_audio(lo, hi, delta, **window):
            audio_calls.append((lo, hi, delta))

        ShotApply.apply(plan, store, move_keys=move_keys, shift_audio=shift_audio)

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
            0,
            "S",
            0,
            10,
            objects=["a", "b"],
            metadata={"object_status": {"a": "missing_object", "b": "valid"}},
        )
        self.assertEqual(shot.classify_objects(), {"a": "missing_object", "b": "valid"})

    def test_classify_leaf_name_fallback(self):
        """Status keyed by short name still matches long DAG-path objects."""
        shot = ShotBlock(
            0,
            "S",
            0,
            10,
            objects=["|grp|a", "|grp|b"],
            metadata={"object_status": {"a": "user_animated"}},
        )
        result = shot.classify_objects()
        self.assertEqual(result["|grp|a"], "user_animated")
        self.assertEqual(result["|grp|b"], "valid")

    def test_classify_scene_discovered_for_non_csv(self):
        shot = ShotBlock(
            0,
            "S",
            0,
            10,
            objects=["a", "z"],
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

    def test_a_lock_survives_the_shots_around_its_gap_changing(self):
        """A lock names its gap by the flanking ids, and those went stale on
        every insert or delete beside it -- the gaps that replaced the locked
        one opened unlocked, silently (2026-09-07: "I locked all gaps and
        they automatically became unlocked a few operations later")."""
        s = ShotStore()
        a = s.define_shot("A", 0, 10)
        b = s.define_shot("B", 20, 30)
        c = s.define_shot("C", 40, 50)
        s.lock_all_gaps()
        ab, bc = (a.shot_id, b.shot_id), (b.shot_id, c.shot_id)

        n = s.define_shot("N", 12, 18)  # inserted into the locked gap A-B
        self.assertEqual(
            s.locked_gaps,
            {(a.shot_id, n.shot_id), (n.shot_id, b.shot_id), bc},
            "both gaps that replaced A-B inherit its lock",
        )
        s.remove_shot(n.shot_id)
        self.assertEqual(
            s.locked_gaps, {ab, bc}, "the gap that replaced two locked ones is locked"
        )
        s.remove_shot(b.shot_id)
        self.assertEqual(s.locked_gaps, {(a.shot_id, c.shot_id)})
        s.remove_shot(a.shot_id)
        self.assertEqual(s.locked_gaps, set(), "no shot before C: nothing to lock")

    def test_undo_and_redo_put_the_locks_of_their_moment_back(self):
        s = ShotStore()
        a = s.define_shot("A", 0, 10)
        b = s.define_shot("B", 20, 30)
        c = s.define_shot("C", 40, 50)
        s.lock_all_gaps()
        s.push_boundary_snapshot()
        s.remove_shot(b.shot_id)
        self.assertEqual(s.locked_gaps, {(a.shot_id, c.shot_id)})
        self.assertTrue(s.restore_boundary_snapshot())
        self.assertEqual(
            s.locked_gaps,
            {(a.shot_id, b.shot_id), (b.shot_id, c.shot_id)},
            "undoing the delete brings B back with its two locks",
        )
        self.assertTrue(s.redo_boundary_snapshot())
        self.assertEqual(s.locked_gaps, {(a.shot_id, c.shot_id)})


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
            "A",
            0,
            10,
            objects=["a"],
            description="first",
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
        """One record, one join key: each clip carries its own range, so
        there is no second take list to disagree with it; an empty
        description or section is left out rather than written as ""."""
        s = _store([])
        s.define_shot("Intro_Shot", 0, 10, objects=["|grp|hero"], description="d")
        s.define_shot("Plain", 20, 30)
        view = s.to_export_view()
        self.assertEqual(list(view), ["shot_metadata"])
        meta = view["shot_metadata"]
        self.assertNotIn("takes", meta)
        self.assertEqual(
            meta["shots"][0],
            {
                "clip": "Intro_Shot",
                "start": 0,
                "end": 10,
                "description": "d",
                # Objects are reduced to leaf names in the metadata channel.
                "objects": ["hero"],
            },
        )
        # Nothing to say: no description / section keys, objects still an array.
        self.assertEqual(
            meta["shots"][1], {"clip": "Plain", "start": 20, "end": 30, "objects": []}
        )


class TestShotNames(_ShotTest):
    """A shot's name is its clip name on every carrier, so the store refuses a
    name the export would have to respell -- instead of respelling it silently
    on the way out (2026-09-18: names typed as "Step 9.1" shipped as
    "Step_9_1", and a second "Shot 1" as "Shot_1_1")."""

    #: Each would reach the FBX take / Unity clip / join key as something else.
    ILLEGAL = ("Shot 1", "Step 9.1", "Fade-In!", "café", "a/b", "")

    def _clips(self, store):
        view = store.to_export_view()
        return [r["clip"] for r in view["shot_metadata"]["shots"]]

    def test_define_refuses_a_name_the_carrier_would_respell(self):
        s = _store([])
        for name in self.ILLEGAL:
            with self.subTest(name=name):
                self.assertIsNotNone(s.name_error(name))
                with self.assertRaises(ValueError):
                    s.define_shot(name, 0, 10)
        self.assertEqual(s.shots, [])

    def test_a_refused_rename_edits_nothing(self):
        s = _store([])
        shot = s.define_shot("Intro", 0, 10, description="d")
        with self.assertRaises(ValueError):
            s.update_shot(shot.shot_id, start=5, name="Intro 2", description="x")
        self.assertEqual((shot.name, shot.start, shot.description), ("Intro", 0, "d"))

    def test_names_must_differ_by_more_than_case(self):
        """Unity joins clip names ignoring case: Intro and intro are one clip."""
        s = _store([])
        intro = s.define_shot("Intro", 0, 10)
        with self.assertRaises(ValueError):
            s.define_shot("intro", 20, 30)
        # A shot's own name is no collision: re-casing it is a rename.
        s.update_shot(intro.shot_id, name="INTRO")
        self.assertEqual(intro.name, "INTRO")

    def test_every_accepted_name_reaches_the_carrier_verbatim(self):
        """Including the forms the legacy respelling collapsed or stripped."""
        names = ["Intro", "A__B", "Shot_", "_lead", "01_x", "b"]
        s = _store([])
        for i, name in enumerate(names):
            s.define_shot(name, i * 10, i * 10 + 5)
        self.assertEqual(self._clips(s), names)
        # ... and through the stored JSON record the deliverable carries, and
        # the take list every reader derives from it.
        (shots_rec,) = s.export_records()
        decoded = shots_rec.spec.decode(shots_rec.text)
        self.assertEqual([r["clip"] for r in decoded["shots"]], names)
        takes = SceneRecords.declared_takes({SceneRecords.SHOTS.key: decoded}.get)
        self.assertEqual([t["name"] for t in takes], names)

    def test_descriptions_travel_verbatim(self):
        """JSON holds any text, so a description is never refused or respelled."""
        text = 'café — "quoted" \\ back\nslash\ttab ✓'
        s = _store([])
        s.define_shot("A", 0, 10, description=text)
        (shots_rec,) = s.export_records()
        decoded = shots_rec.spec.decode(shots_rec.text)
        self.assertEqual(decoded["shots"][0]["description"], text)

    def test_a_legacy_name_still_exports_and_says_so(self):
        """A store loaded from before the rule: respelled as before, but said."""
        s = _store(
            [
                ShotBlock(0, "Shot 1", 0, 10),
                ShotBlock(1, "Intro", 20, 30),
                ShotBlock(2, "intro", 40, 50),
            ]
        )
        with self.assertLogs(shot_model._log, "WARNING") as logs:
            clips = self._clips(s)
        self.assertEqual(clips, ["Shot_1", "Intro", "intro_1"])
        self.assertIn("'Shot 1' -> 'Shot_1'", logs.output[0])
        self.assertIn("'intro' -> 'intro_1'", logs.output[0])

    def test_legal_names_export_without_a_warning(self):
        s = _store([])
        s.define_shot("Intro", 0, 10)
        with self.assertLogs(shot_model._log, "WARNING") as logs:
            shot_model._log.warning("sentinel")  # assertLogs needs one record
            self._clips(s)
        self.assertEqual(len(logs.output), 1, logs.output)

    def test_unique_name_derives_a_legal_unused_name(self):
        s = _store([])
        self.assertEqual(s.unique_name("Shot", first=1), "Shot_1")
        s.define_shot("Shot_1", 0, 10)
        self.assertEqual(s.unique_name("Shot", first=1), "Shot_2")
        self.assertEqual(s.unique_name("shot_1"), "shot_1_2")
        # Derived from a legacy name: legal all the same.
        self.assertEqual(s.unique_name("Step 9.1_2"), "Step_9_1_2")
        self.assertIsNone(s.name_error(s.unique_name("Step 9.1_2")))

    def test_a_name_cannot_take_what_a_legacy_name_exports_as(self):
        """Accepted beside a legacy "Shot 1", ``Shot_1`` would ship as
        ``Shot_1_1``: the legal name respelled, the very thing refused."""
        s = _store([ShotBlock(0, "Shot 1", 0, 10)])
        error = s.name_error("Shot_1")
        self.assertIsNotNone(error)
        self.assertIn("exported as 'Shot_1'", error)
        with self.assertRaises(ValueError):
            s.define_shot("shot_1", 20, 30)
        self.assertEqual(s.unique_name("Shot", first=1), "Shot_2")

    def test_a_name_cannot_take_a_legacy_twins_suffixed_clip(self):
        """Legacy case twins export de-duplicated (``Intro`` / ``intro_1``);
        a new ``Intro_1`` would ship as ``Intro_1_1`` -- respelled, the very
        thing name validation exists to prevent."""
        s = _store([ShotBlock(0, "Intro", 0, 10), ShotBlock(1, "intro", 20, 30)])
        clips = [c for c, _s, _e in s.resolve_clip_specs(s.sorted_shots())]
        self.assertEqual(clips, ["Intro", "intro_1"])
        self.assertIn("exports as 'intro_1'", s.name_error("Intro_1"))
        self.assertIsNone(s.name_error("Intro_2"))
        # Renaming the twin itself is not a collision with its own clip.
        self.assertIsNone(s.name_error("intro_9", shot_id=1))

    def test_detected_candidates_take_the_next_free_default_name(self):
        """Detection numbers its candidates from 1; beside ``Shot_1`` /
        ``Shot_2`` the new shots are ``Shot_3`` / ``Shot_4``, not
        ``Shot_1_2`` / ``Shot_2_2``."""

        class Detecting(ShotStore):
            def detect_regions(self):
                return [
                    {"name": "Shot_1", "start": 100, "end": 110},
                    {"name": "Shot_2", "start": 120, "end": 130},
                ]

        s = Detecting([])
        s.define_shot("Shot_1", 0, 10)
        s.define_shot("Shot_2", 20, 30)
        created = s.detect_and_define()
        self.assertEqual([c.name for c in created], ["Shot_3", "Shot_4"])
        self.assertEqual(s.default_name("Walk"), "Walk")
        self.assertEqual(s.default_name("Walk Cycle"), "Shot_5")
        self.assertEqual(s.default_name(None), "Shot_5")

    def test_a_restore_puts_names_back_without_revalidating(self):
        """Undoing a swap passes through a moment where two shots share a name."""
        s = _store([])
        a = s.define_shot("P", 0, 10)
        b = s.define_shot("Q", 20, 30)
        s.push_boundary_snapshot()
        s.update_shot(a.shot_id, name="R")
        s.update_shot(b.shot_id, name="P")
        s.update_shot(a.shot_id, name="Q")
        self.assertTrue(s.restore_boundary_snapshot())
        self.assertEqual((a.name, b.name), ("P", "Q"))


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

    def test_flush_pending_stores_what_the_active_store_holds_unwritten(self):
        """A DCC store writes on idle (Maya coalesces a mutation into one
        deferred write), so its record can be an edit behind; a crossing
        reads the record, and asks for what the live store holds first."""

        class Backend:
            data, saves = None, 0

            def load(self):
                return self.data

            def save(self, data):
                self.data, self.saves = data, self.saves + 1

        class Deferred(ShotStore):
            def _schedule_flush(self):
                """Writes on idle, as a DCC store does."""

        backend = Backend()
        Deferred._persistence, Deferred._active = backend, None
        try:
            Deferred.active().define_shot("A", 0.0, 10.0)
            self.assertEqual(backend.saves, 0)
            Deferred.flush_pending()
            self.assertEqual([s["name"] for s in backend.data["shots"]], ["A"])
            Deferred.flush_pending()
            self.assertEqual(backend.saves, 1)  # nothing pending: nothing written
        finally:
            Deferred._persistence, Deferred._active = None, None


class _RecordingStore(ShotStore):
    """Stand-in for a DCC subclass: records what the publish hook was handed.

    The pure core's :meth:`publish_export_view` writes nothing and returns
    ``None``, so overriding it is the only way to observe that the hook fired
    at all — which is exactly what a real DCC subclass does to serialise the
    view onto its carrier node.
    """

    def __init__(self, shots=None):
        super().__init__(shots)
        self.published = []

    def publish_export_view(self, strategy=None):
        self.published.append(self.to_export_view(strategy or "name"))
        return "carrier"


class TestExportViewRefresh(_ShotTest):
    """``refresh_export_view`` -> ``publish_export_view`` — the pre-export path.

    Note the active store is always driven through ``ShotStore`` (never through
    ``_RecordingStore``): ``_active`` is a ``ShotStore`` class attribute, so
    setting it on the subclass would shadow rather than share it.
    """

    def test_pure_publish_hook_is_a_noop_returning_none(self):
        # No DCC carrier in the pure core — the hook resolves and does nothing,
        # populated or empty, so refresh_export_view is safe to call anywhere.
        populated = ShotStore([ShotBlock(1, "A", 0, 10, ["x"])])
        self.assertIsNone(populated.publish_export_view())
        self.assertIsNone(ShotStore([]).publish_export_view())

    def test_refresh_without_active_store_creates_no_shots(self):
        # ``active()`` materialises an empty in-memory store on first access;
        # the refresh must ride that without inventing content to publish.
        ShotStore.refresh_export_view()
        self.assertEqual(ShotStore.active().shots, [])

    def test_empty_store_export_view_is_the_clearing_payload(self):
        # Pinned literally: this is what a DCC subclass writes onto its carrier
        # for an empty store, and empty channels are what clear it.  ``fps``
        # qualifies the frame numbers the clips' ranges are expressed in, so
        # it rides the envelope even when there are none.
        store = ShotStore([])
        self.assertEqual(
            store.to_export_view(),
            {
                "shot_metadata": {
                    "version": 1,
                    "fps": store.scene_fps,
                    "shots": [],
                },
            },
        )

    def test_empty_store_still_publishes(self):
        # Contract: an empty store MUST reach the hook.  Guarding on
        # ``store.shots`` skipped it, so deleting the last shot left the
        # previous publish's takes riding into the next export.
        store = _RecordingStore([])
        ShotStore.set_active(store)

        ShotStore.refresh_export_view()

        self.assertEqual(len(store.published), 1)
        self.assertEqual(
            store.published[0],
            {
                "shot_metadata": {
                    "version": 1,
                    "fps": store.scene_fps,
                    "shots": [],
                },
            },
        )

    def test_populated_store_publishes_its_takes(self):
        store = _RecordingStore([])
        store.snap_whole_frames = False
        store.define_shot("Intro", 0, 10, objects=["|grp|hero"], description="d")
        ShotStore.set_active(store)

        ShotStore.refresh_export_view()

        self.assertEqual(len(store.published), 1)
        view = store.published[0]
        self.assertEqual(
            view["shot_metadata"],
            {
                "version": 1,
                # The rate the clip frames below are counted in -- without it
                # they are unitless to every consumer that did not author the
                # scene (MeshConvert.apply_glb_animations reads this back out
                # of the delivered GLB).
                "fps": store.scene_fps,
                # Each clip carries its own range: the take list IS the clips
                # (one record, one join key); no section, so none is written.
                "shots": [
                    {
                        "clip": "Intro",
                        "start": 0,
                        "end": 10,
                        "description": "d",
                        "objects": ["hero"],
                    }
                ],
            },
        )
        self.assertNotIn("fbx_takes", view)

    def test_export_records_declare_the_context_clip_mode(self):
        """The producer form: the exporter's Animation Clips decision is an
        INPUT on the record, never patched on afterwards, and an empty store
        yields nothing -- which a commit turns into a clear."""
        from pythontk.core_utils.scene_records import ExportContext

        store = _RecordingStore([])
        self.assertIsNone(store.export_records(ExportContext(clip_mode="full")))
        store.snap_whole_frames = False
        store.define_shot("Intro", 0, 10)
        (record,) = store.export_records(ExportContext(clip_mode="full"))
        self.assertEqual(record.spec, SceneRecords.SHOTS)
        self.assertEqual(record.payload["clip_mode"], "full")
        self.assertEqual(record.payload["shots"][0]["clip"], "Intro")
        self.assertNotIn("clip_mode", store.export_records()[0].payload)

    def test_refresh_republishes_on_every_call(self):
        # The refresh is the canonical *pre-export* step: it must reproject
        # each time, not memoise, or a mutation between exports is lost.
        store = _RecordingStore([])
        ShotStore.set_active(store)

        ShotStore.refresh_export_view()
        store.define_shot("Intro", 0, 10, objects=["hero"])
        ShotStore.refresh_export_view()

        self.assertEqual(len(store.published), 2)
        self.assertEqual(store.published[0]["shot_metadata"]["shots"], [])
        self.assertEqual(len(store.published[1]["shot_metadata"]["shots"]), 1)


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
        out = ShotDetection.cluster_segments_by_gap(
            segments, gap_threshold=5, min_duration=2
        )
        self.assertEqual(len(out), 2)
        self.assertEqual((out[0]["start"], out[0]["end"]), (0, 20))
        self.assertEqual(out[0]["objects"], ["a", "b"])
        # A name the store accepts as it is: the clip every carrier writes.
        self.assertEqual(out[0]["name"], "Shot_1")
        self.assertEqual((out[1]["start"], out[1]["end"]), (40, 50))
        self.assertEqual(out[1]["objects"], ["c"])

    def test_min_duration_filter(self):
        segments = [
            {"start": 0, "end": 1, "obj": "a"},  # duration 1 < 2 → dropped
            {"start": 30, "end": 40, "obj": "b"},  # kept
        ]
        out = ShotDetection.cluster_segments_by_gap(
            segments, gap_threshold=5, min_duration=2
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["objects"], ["b"])

    def test_empty(self):
        self.assertEqual(ShotDetection.cluster_segments_by_gap([]), [])

    def test_does_not_mutate_input(self):
        segments = [
            {"start": 30, "end": 40, "obj": "b"},
            {"start": 0, "end": 10, "obj": "a"},
        ]
        snapshot = [dict(seg) for seg in segments]
        ShotDetection.cluster_segments_by_gap(segments)
        self.assertEqual(segments, snapshot)  # sorted a copy, not in place


class TestBoundariesFromKeyEntries(_ShotTest):
    def test_all_mode_contiguous(self):
        entries = [(0, 1, "a"), (20, 1, "b"), (40, 1, "c")]
        out = ShotDetection.boundaries_from_key_entries(
            entries, gap_threshold=5, key_filter="all"
        )
        self.assertEqual(len(out), 3)
        self.assertEqual((out[0]["start"], out[0]["end"]), (0, 20))
        self.assertEqual((out[1]["start"], out[1]["end"]), (20, 40))
        self.assertEqual((out[2]["start"], out[2]["end"]), (40, 41))  # last +1

    def test_all_mode_merges_within_gap(self):
        entries = [(0, 1, "a"), (3, 1, "b"), (40, 1, "c")]  # 0 & 3 merge (<=5)
        out = ShotDetection.boundaries_from_key_entries(
            entries, gap_threshold=5, key_filter="all"
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["objects"], ["a", "b"])
        self.assertEqual((out[0]["start"], out[0]["end"]), (0, 40))

    def test_skip_zero_drops_zero_keys(self):
        entries = [(0, 0, "a"), (20, 1, "b"), (40, 0, "c")]
        out = ShotDetection.boundaries_from_key_entries(
            entries, gap_threshold=5, key_filter="skip_zero"
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["start"], 20)
        self.assertEqual(out[0]["objects"], ["b"])

    def test_zero_as_end_pairs_starts_with_ends(self):
        entries = [(0, 1, "a"), (10, 0, "a"), (20, 1, "b"), (35, 0, "b")]
        out = ShotDetection.boundaries_from_key_entries(
            entries, key_filter="zero_as_end"
        )
        self.assertEqual(len(out), 2)
        self.assertEqual((out[0]["start"], out[0]["end"]), (0, 10))
        self.assertEqual((out[1]["start"], out[1]["end"]), (20, 35))

    def test_empty(self):
        self.assertEqual(ShotDetection.boundaries_from_key_entries([]), [])


class TestDetectionConstants(_ShotTest):
    def test_standard_transform_attrs(self):
        self.assertIn("translateX", STANDARD_TRANSFORM_ATTRS)
        self.assertIn("visibility", STANDARD_TRANSFORM_ATTRS)
        self.assertEqual(len(STANDARD_TRANSFORM_ATTRS), 10)


class TestSnapEnclosesContent(unittest.TestCase):
    """``ShotStore.snap(frame, direction)``: a bound that must ENCLOSE
    content snaps outward.

    Rounded to the nearest frame, a start could land up to half a frame PAST
    a shot's first key -- a retime leaves keys on fractional frames: measured
    on the production assembly, a trim moved "Step 4.1"'s start to 985 over
    its fade's first key at 984.556, leaving that key in the span the trim
    gave up, where the neighbour's envelope carries it on the next edit.
    Added: 2026-09-19
    """

    def test_nearest_is_unchanged_and_down_up_enclose(self):
        store = ShotStore()
        self.assertEqual(store.snap(984.556), 985.0)
        self.assertEqual(store.snap(984.556, "down"), 984.0)
        self.assertEqual(store.snap(1031.222, "up"), 1032.0)
        self.assertEqual(store.snap(40.0, "down"), 40.0, "a whole frame stays")
        # Float noise off a whole frame is not a frame of content.
        self.assertEqual(store.snap(40.0000001, "up"), 40.0)
        self.assertEqual(store.snap(39.9999999, "down"), 40.0)

    def test_direction_is_ignored_when_snapping_is_off(self):
        store = ShotStore()
        store.snap_whole_frames = False
        self.assertEqual(store.snap(984.556, "down"), 984.556)

    def test_an_unknown_direction_is_refused(self):
        with self.assertRaises(ValueError):
            ShotStore().snap(984.556, "outward")


class TestEditLedgerRemap(unittest.TestCase):
    """``ShotEditLedger.remap`` moves every claim ONCE, all pairs at once.

    Bug: the pairs were applied one after another, so a claim moved onto the
    source frame of a LATER pair was moved a second time whenever that later
    key was not claimed itself.  Every mover hands ``remap`` the pairs of ALL
    the keys it moved, claimed or not, so any ripple whose delta equals a key
    spacing hit it: measured in Maya, a respace moving a shot +10 over its
    start pin at 60 and an animator key at 70 left the pin at 70 and its
    claim at 80 -- on the animator's key, where the next reconcile could cut
    or disown it while the real pin went unclaimed.
    Fixed: 2026-09-19
    """

    def test_a_claim_moves_once_whatever_it_lands_on(self):
        from pythontk.core_utils.engines.shots.shot_ledger import ShotEditLedger

        led = ShotEditLedger()
        led.record_key("c", 60.0, 1, "start")
        led.record_step("c", 60.0, "auto", "auto")
        led.remap("c", [(60.0, 70.0), (70.0, 80.0), (90.0, 100.0)])
        self.assertEqual(led.key_records("c"), [(70.0, 1, "start")])
        self.assertEqual(led.step_times("c"), [70.0])

    def test_a_scale_moves_a_claim_once_past_an_unclaimed_key(self):
        from pythontk.core_utils.engines.shots.shot_ledger import ShotEditLedger

        led = ShotEditLedger()
        led.record_key("c", 25.0, 0, "end")
        led.remap("c", [(t, t * 2.0) for t in (10.0, 25.0, 50.0)])  # x2 about 0
        self.assertEqual(led.key_records("c"), [(50.0, 0, "end")])

    def test_claims_that_trade_places_each_move_once(self):
        from pythontk.core_utils.engines.shots.shot_ledger import ShotEditLedger

        led = ShotEditLedger()
        led.record_key("c", 10.0, 1, "a")
        led.record_key("c", 20.0, 2, "b")
        self.assertEqual(led.remap("c", [(10.0, 20.0), (20.0, 10.0)]), 2)
        self.assertEqual(led.key_records("c"), [(10.0, 2, "b"), (20.0, 1, "a")])

    def test_unclaimed_and_unmoved_pairs_are_ignored(self):
        from pythontk.core_utils.engines.shots.shot_ledger import ShotEditLedger

        led = ShotEditLedger()
        led.record_key("c", 5.0, 0, "start")
        self.assertEqual(led.remap("c", [(5.0, 5.0), (7.0, 9.0)]), 0)
        self.assertEqual(led.remap("other", [(5.0, 6.0)]), 0)
        self.assertEqual(led.key_records("c"), [(5.0, 0, "start")])


class TestBoundaryLedger(unittest.TestCase):
    """The store-scoped boundary-snapshot ledger.

    Scene keys ride the DCC's undo queue; shot bounds live here, outside
    it.  One ledger per store (= per scene) is shared by every panel that
    mutates boundaries, so an undo after ANY panel's edit restores the
    right snapshot — per-controller stacks desynced from each other.
    """

    def _store(self):
        store = ShotStore()
        store.define_shot("A", 0, 50)
        store.define_shot("B", 60, 100)
        return store

    def test_the_edit_ledger_rides_the_restore_point(self):
        """Undo puts the claims back where the keys go back to; redo moves
        them forward again.  A claim left at the post-edit frame after an
        undo names an animator key as the system's sample."""
        store = self._store()
        store.edit_ledger.record_key("crv", 10.0, 0, "end")
        store.edit_ledger.record_step("crv", 10.0, "auto", "auto")
        store.push_boundary_snapshot()
        store.edit_ledger.remap("crv", [(10.0, 20.0)])
        store.edit_ledger.record_key("other", 5.0, 1, "start")
        self.assertTrue(store.restore_boundary_snapshot())
        self.assertEqual(store.edit_ledger.key_times("crv"), [10.0])
        self.assertEqual(store.edit_ledger.step_times("crv"), [10.0])
        self.assertEqual(store.edit_ledger.key_times("other"), [])
        self.assertTrue(store.redo_boundary_snapshot())
        self.assertEqual(store.edit_ledger.key_times("crv"), [20.0])
        self.assertEqual(store.edit_ledger.key_times("other"), [5.0])

    def test_restore_returns_the_pre_edit_bounds(self):
        store = self._store()
        store.push_boundary_snapshot()
        store.update_shot(0, start=10, end=70)
        self.assertTrue(store.restore_boundary_snapshot())
        self.assertEqual(
            (store.shot_by_id(0).start, store.shot_by_id(0).end), (0.0, 50.0)
        )

    def test_restore_removes_shots_created_after_the_snapshot(self):
        """Undo of an insert must not leave a phantom overlapping shot."""
        store = self._store()
        store.push_boundary_snapshot()
        store.update_shot(1, start=180, end=220)  # the ripple
        store.define_shot("Inserted", 60, 170)  # the insert
        store.restore_boundary_snapshot()
        self.assertIsNone(store.shot_by_name("Inserted"))
        self.assertEqual(
            (store.shot_by_id(1).start, store.shot_by_id(1).end), (60.0, 100.0)
        )

    def test_redo_reapplies_what_undo_stepped_back_from(self):
        """Without the redo side, a DCC redo re-applies the scene keys while
        the bounds stay restored — the keys-outside-their-shot state."""
        store = self._store()
        store.push_boundary_snapshot()
        store.update_shot(0, start=0, end=80)
        store.update_shot(1, start=90, end=130)
        store.restore_boundary_snapshot()
        self.assertEqual(store.shot_by_id(0).end, 50.0)
        self.assertTrue(store.redo_boundary_snapshot())
        self.assertEqual(store.shot_by_id(0).end, 80.0)
        self.assertEqual(store.shot_by_id(1).start, 90.0)
        # And undo works again after the redo.
        self.assertTrue(store.restore_boundary_snapshot())
        self.assertEqual(store.shot_by_id(0).end, 50.0)

    def test_a_new_edit_invalidates_the_redo_branch(self):
        store = self._store()
        store.push_boundary_snapshot()
        store.update_shot(0, end=80)
        store.restore_boundary_snapshot()
        store.push_boundary_snapshot()  # a NEW edit begins
        store.update_shot(0, end=95)
        self.assertFalse(
            store.redo_boundary_snapshot(),
            "a new edit must clear the redo branch, mirroring every undo queue",
        )
        self.assertEqual(store.shot_by_id(0).end, 95.0, "and leave the edit intact")

    def test_discard_drops_a_noop_restore_point(self):
        store = self._store()
        store.push_boundary_snapshot()
        store.discard_boundary_snapshot()  # the edit turned out to be a no-op
        self.assertFalse(store.restore_boundary_snapshot())

    def test_restore_on_empty_ledger_is_a_safe_noop(self):
        store = self._store()
        self.assertFalse(store.restore_boundary_snapshot())
        self.assertFalse(store.redo_boundary_snapshot())

    def test_ledger_is_capped(self):
        store = self._store()
        for _ in range(store._BOUNDARY_LEDGER_CAP + 10):
            store.push_boundary_snapshot()
        self.assertEqual(len(store._boundary_undo), store._BOUNDARY_LEDGER_CAP)

    def test_clear_drops_both_sides(self):
        store = self._store()
        store.push_boundary_snapshot()
        store.update_shot(0, end=80)
        store.restore_boundary_snapshot()
        store.push_boundary_snapshot()
        store.clear_boundary_snapshots()
        self.assertFalse(store._boundary_undo)
        self.assertFalse(store._boundary_redo)

    def test_restore_recreates_shots_deleted_after_the_snapshot(self):
        """Undo of a delete: the keys were never deleted with the store
        record, so re-creating it from the snapshot is a faithful restore
        — full identity included, not just bounds."""
        store = self._store()
        store.update_shot(
            0, name="Hero", description="the money shot", objects=["cubeA"]
        )
        store.shot_by_id(0).metadata["camera"] = "cam1"
        store.push_boundary_snapshot()
        store.remove_shot(0)
        self.assertIsNone(store.shot_by_id(0))
        store.restore_boundary_snapshot()
        shot = store.shot_by_id(0)
        self.assertIsNotNone(shot, "the deleted shot must be re-created")
        self.assertEqual(
            (shot.name, shot.description, shot.objects, shot.metadata),
            ("Hero", "the money shot", ["cubeA"], {"camera": "cam1"}),
            "with its FULL identity, not just bounds",
        )

    def test_redo_of_an_insert_recreates_the_inserted_shot(self):
        """Redo re-applies the scene keys (the ripple); without membership
        re-creation the inserted shot's record stayed gone while the keys
        moved — the asymmetric half of the phantom-shot fix."""
        store = self._store()
        store.push_boundary_snapshot()
        store.update_shot(1, start=180, end=220)  # the ripple
        inserted = store.define_shot("Inserted", 60, 170)
        store.restore_boundary_snapshot()  # undo removes it
        self.assertIsNone(store.shot_by_id(inserted.shot_id))
        store.redo_boundary_snapshot()
        again = store.shot_by_id(inserted.shot_id)
        self.assertIsNotNone(again, "redo must bring the inserted shot back")
        self.assertEqual(
            (again.name, again.start, again.end), ("Inserted", 60.0, 170.0)
        )
        self.assertEqual(
            (store.shot_by_id(1).start, store.shot_by_id(1).end), (180.0, 220.0)
        )

    def test_snapshot_includes_objects(self):
        store = self._store()
        store.update_shot(0, objects=["cubeA"])
        store.push_boundary_snapshot()
        store.update_shot(0, objects=["cubeA", "cubeB"])
        store.restore_boundary_snapshot()
        self.assertEqual(store.shot_by_id(0).objects, ["cubeA"])


class TestBoundaryLedgerTags(unittest.TestCase):
    """Restore points carry an opaque DCC tag.

    A restore point and the DCC undo step that accompanies it are two
    different stacks and do NOT always come in pairs — an edit that moved
    only BOUNDS records nothing natively.  The store keeps whatever marker
    the DCC layer hands it so that layer can tell the cases apart.
    """

    def _store(self):
        store = ShotStore()
        store.define_shot("A", 0, 50)
        store.define_shot("B", 60, 100)
        return store

    def test_untagged_push_peeks_none(self):
        store = self._store()
        store.push_boundary_snapshot()
        self.assertIsNone(store.peek_boundary_tag())
        self.assertTrue(store.has_boundary_snapshot())

    def test_peek_is_none_when_empty_and_has_is_false(self):
        store = self._store()
        self.assertIsNone(store.peek_boundary_tag())
        self.assertFalse(store.has_boundary_snapshot())
        self.assertFalse(store.has_boundary_snapshot(redo=True))

    def test_tag_stamps_the_newest_restore_point(self):
        store = self._store()
        store.push_boundary_snapshot()
        self.assertTrue(store.tag_boundary_snapshot((False, "other_op")))
        self.assertEqual(store.peek_boundary_tag(), (False, "other_op"))

    def test_tagging_an_empty_ledger_reports_failure(self):
        self.assertFalse(self._store().tag_boundary_snapshot((True, "x")))

    def test_tag_only_touches_the_top_entry(self):
        store = self._store()
        store.push_boundary_snapshot(tag=(True, "first"))
        store.push_boundary_snapshot()
        store.tag_boundary_snapshot((False, "second"))
        self.assertEqual(store.peek_boundary_tag(), (False, "second"))
        store.restore_boundary_snapshot()
        self.assertEqual(store.peek_boundary_tag(), (True, "first"))

    def test_tag_rides_across_to_the_redo_side(self):
        """Redoing an edit re-applies the same DCC step (or, unpaired, none)."""
        store = self._store()
        store.push_boundary_snapshot(tag=(False, "bounds_only"))
        store.update_shot(0, end=70)
        store.restore_boundary_snapshot()
        self.assertEqual(store.peek_boundary_tag(redo=True), (False, "bounds_only"))
        self.assertTrue(store.has_boundary_snapshot(redo=True))
        store.redo_boundary_snapshot()
        self.assertEqual(store.peek_boundary_tag(), (False, "bounds_only"))

    def test_tagged_entries_still_restore_bounds(self):
        store = self._store()
        store.push_boundary_snapshot(tag=(True, "chunk_1"))
        store.update_shot(0, start=10, end=70)
        self.assertTrue(store.restore_boundary_snapshot())
        self.assertEqual(
            (store.shot_by_id(0).start, store.shot_by_id(0).end), (0.0, 50.0)
        )


class TestOrderedPhaseOverwrites(unittest.TestCase):
    """Phase 1 must let keys pass their neighbours.

    Its topological order only orders the moves the PLAN contains.  A shared
    curve can carry keys no shot owns (keys inside a shot that does not list
    that object); those never move, and a clamping "move" semantic collapses
    the travelling key onto the first one it meets — landing it at the wrong
    frame, duplicated.  Reproduced on a real production scene.
    """

    def test_ordered_moves_pass_over(self):
        store = ShotStore()
        store.define_shot("A", 0, 50, objects=["obj"])
        store.define_shot("B", 60, 100, objects=["obj"])
        calls = []

        def move_keys(objects, lo, hi, delta, over=False, **window):
            calls.append(over)

        ShotApply.apply(
            ShotPlanner.plan_respace(store, gap=20, start_frame=0),
            store,
            move_keys=move_keys,
        )
        self.assertTrue(calls, "the plan must have moved something")
        self.assertTrue(
            all(calls), "every phase must pass over=True (see the MoveKeys protocol)"
        )


class TestDiscardRestoresTheRedoBranch(unittest.TestCase):
    """``discard_boundary_snapshot`` is the inverse of the push before it.

    A push clears the redo branch, which is right for a real edit.  But
    several paths push UP FRONT and only then discover the edit was a no-op
    (a drag that scaled nothing, a trim with zero delta, a delete whose every
    cutKey failed on a locked attr) — and those were costing the user their
    redo branch with nothing on screen to explain it.
    """

    def _store_with_redo(self):
        store = ShotStore()
        store.define_shot("A", 0, 50)
        store.define_shot("B", 60, 100)
        store.push_boundary_snapshot(tag=(False, "edit"))
        store.update_shot(0, end=40)
        store.restore_boundary_snapshot()  # -> a redo branch now exists
        return store

    def test_a_no_op_edit_keeps_the_redo_branch(self):
        store = self._store_with_redo()
        store.push_boundary_snapshot(tag=(False, "noop"))
        store.discard_boundary_snapshot()
        self.assertTrue(store.has_boundary_snapshot(redo=True))
        self.assertEqual(store.peek_boundary_tag(redo=True), (False, "edit"))

    def test_the_restored_redo_branch_still_applies(self):
        store = self._store_with_redo()
        store.push_boundary_snapshot()
        store.discard_boundary_snapshot()
        self.assertTrue(store.redo_boundary_snapshot())
        self.assertEqual(store.shot_by_id(0).end, 40.0)

    def test_a_real_edit_still_invalidates_the_redo_branch(self):
        store = self._store_with_redo()
        store.push_boundary_snapshot(tag=(True, "real"))
        store.update_shot(1, end=90)
        self.assertFalse(store.has_boundary_snapshot(redo=True))

    def test_discard_still_drops_the_undo_entry(self):
        store = self._store_with_redo()
        depth = len(store._boundary_undo)
        store.push_boundary_snapshot()
        store.discard_boundary_snapshot()
        self.assertEqual(len(store._boundary_undo), depth)

    def test_a_second_discard_cannot_resurrect_a_consumed_branch(self):
        """The stash is a single slot; consuming the redo branch clears it."""
        store = self._store_with_redo()
        store.redo_boundary_snapshot()  # consumes the branch
        store.push_boundary_snapshot()
        store.discard_boundary_snapshot()
        self.assertFalse(store.has_boundary_snapshot(redo=True))

    def test_discard_on_an_empty_ledger_is_harmless(self):
        store = ShotStore()
        store.define_shot("A", 0, 50)
        store.discard_boundary_snapshot()
        self.assertFalse(store.has_boundary_snapshot())
        self.assertFalse(store.has_boundary_snapshot(redo=True))


class TestObjectsToAdopt(unittest.TestCase):
    """The rule deciding what a moving shot may carry.

    A mover shifts a shot's own object list within a window, so anything
    keyed in that window but missing from the list is left behind — the shot
    moves and part of its animation does not.  Membership goes stale for
    ordinary reasons (a renamed object, an object animated after the shots
    were authored), so the window has to be consulted; but claiming purely
    on time steals keys at a shared boundary.
    """

    def test_an_object_no_shot_owns_is_adopted(self):
        self.assertEqual(
            ShotPlanner.objects_to_adopt({"orphan": [10.0, 20.0]}, set(), 0.0, 50.0),
            ["orphan"],
        )

    def test_an_object_the_shot_already_lists_is_skipped(self):
        self.assertEqual(
            ShotPlanner.objects_to_adopt({"mine": [10.0]}, {"mine"}, 0.0, 50.0), []
        )

    def test_keys_outside_the_window_do_not_qualify(self):
        self.assertEqual(
            ShotPlanner.objects_to_adopt({"late": [80.0]}, set(), 0.0, 50.0), []
        )

    def test_a_gapped_window_is_half_open(self):
        """With a gap, the upper bound is the NEXT shot's own opening sample."""
        self.assertEqual(
            ShotPlanner.objects_to_adopt({"edge": [50.0]}, set(), 0.0, 50.0), []
        )
        self.assertEqual(
            ShotPlanner.objects_to_adopt({"edge": [0.0]}, set(), 0.0, 50.0), ["edge"]
        )

    def test_a_closed_upper_bound_takes_the_shared_sample(self):
        """Contiguous: the shared sample is this shot's closing fencepost."""
        self.assertEqual(
            ShotPlanner.objects_to_adopt(
                {"edge": [50.0]}, set(), 0.0, 50.0, hi_closed=True
            ),
            ["edge"],
        )

    def test_an_open_lower_bound_declines_the_shared_sample(self):
        """The same sample seen from the FOLLOWING shot: not its to move."""
        self.assertEqual(
            ShotPlanner.objects_to_adopt(
                {"edge": [40.0]}, set(), 40.0, 60.0, lo_open=True
            ),
            [],
        )

    def test_the_two_flags_partition_a_shared_sample(self):
        """Exactly one of the two adjacent shots claims it — never both, never
        neither.  This is what replaced the owner exemption: the resized-shot
        case (B ends where C starts) now falls out of the window itself."""
        keyed = {"B": [20.0, 30.0, 40.0]}
        preceding = ShotPlanner.objects_to_adopt(
            keyed, set(), 20.0, 40.0, hi_closed=True
        )
        following = ShotPlanner.objects_to_adopt(keyed, set(), 40.0, 60.0, lo_open=True)
        self.assertEqual(preceding, ["B"], "the shared sample is the owner's")
        self.assertEqual(following, [], "and the next shot must not claim it too")

    def test_stale_membership_far_from_any_boundary_is_still_adopted(self):
        """An object animated after the shots were authored: no flag involved,
        it simply has a key inside the window being moved."""
        self.assertEqual(
            ShotPlanner.objects_to_adopt(
                {"shared": [10.0, 500.0]}, set(), 480.0, 520.0
            ),
            ["shared"],
        )

    def test_the_result_is_sorted_and_deduplicated(self):
        got = ShotPlanner.objects_to_adopt(
            {"b": [1.0, 2.0], "a": [3.0]}, set(), 0.0, 10.0
        )
        self.assertEqual(got, ["a", "b"])

    def test_empty_inputs_are_harmless(self):
        self.assertEqual(ShotPlanner.objects_to_adopt({}, set(), 0.0, 10.0), [])
        self.assertEqual(ShotPlanner.objects_to_adopt({"x": []}, None, 0.0, 10.0), [])


class TestFencepostEnvelope(unittest.TestCase):
    """Contiguous shots share one sample; the preceding shot owns it.

    A shot spans the samples ``start..end``, so at gap 0 shot N's closing
    sample and shot N+1's opening sample are the same frame number.  The
    envelope's two closure flags decide it once, for every consumer, and
    the result must PARTITION: never both shots, never neither.
    """

    def _shots(self, *pairs):
        return [
            ShotBlock(i, chr(65 + i), lo, hi, []) for i, (lo, hi) in enumerate(pairs)
        ]

    def test_contiguous_shots_close_the_upper_bound_and_open_the_lower(self):
        shots = self._shots((0, 40), (40, 80), (80, 120))
        self.assertEqual(ShotPlanner.envelope_for(shots, 0), (0, 40, False, True))
        self.assertEqual(ShotPlanner.envelope_for(shots, 1), (40, 80, True, True))
        env = ShotPlanner.envelope_for(shots, 2)
        self.assertEqual(env[0], 80)
        self.assertTrue(env[2], "the last shot still declines its opening sample")
        self.assertFalse(env[3], "and its unbounded end closes nothing")

    def test_a_gapped_boundary_keeps_the_plain_half_open_window(self):
        shots = self._shots((0, 40), (50, 90))
        self.assertEqual(ShotPlanner.envelope_for(shots, 0), (0, 50, False, False))
        self.assertEqual(ShotPlanner.envelope_for(shots, 1)[:3], (50, 1.0e9, False))

    def test_every_shared_sample_belongs_to_exactly_one_shot(self):
        shots = self._shots((0, 40), (40, 80), (80, 120))

        def owners(t):
            return [
                s.name
                for i, s in enumerate(shots)
                if ShotPlanner.in_window(t, *ShotPlanner.envelope_for(shots, i))
            ]

        self.assertEqual(owners(40.0), ["A"], "shared sample -> preceding shot")
        self.assertEqual(owners(80.0), ["B"])
        self.assertEqual(owners(0.0), ["A"], "the very first sample still lands")
        for t in (0.5, 39.5, 40.5, 79.5, 119.0):
            self.assertEqual(len(owners(t)), 1, f"frame {t} must have one owner")

    def test_trailing_gap_content_still_travels_with_the_preceding_shot(self):
        shots = self._shots((0, 40), (50, 90))
        self.assertTrue(
            ShotPlanner.in_window(45.0, *ShotPlanner.envelope_for(shots, 0)),
            "a fade tail in the gap belongs to the shot it trails",
        )
        self.assertFalse(
            ShotPlanner.in_window(45.0, *ShotPlanner.envelope_for(shots, 1))
        )


class TestBoundarySplits(unittest.TestCase):
    """Opening a gap pulls a shared sample apart; the following shot needs a copy."""

    def _store(self, *pairs):
        st = ShotStore()
        st.shots = [
            ShotBlock(i, chr(65 + i), lo, hi, []) for i, (lo, hi) in enumerate(pairs)
        ]
        return st

    def test_a_respace_that_opens_a_gap_reports_the_following_shot(self):
        st = self._store((0, 40), (40, 80))
        plan = ShotPlanner.plan_respace(st, gap=10, start_frame=0)
        self.assertEqual(ShotPlanner.boundary_splits(st, plan), [(0, 1, 40.0, 50.0)])

    def test_a_respace_that_keeps_the_gap_at_zero_reports_nothing(self):
        st = self._store((0, 40), (40, 80))
        plan = ShotPlanner.plan_respace(st, gap=0, start_frame=0)
        self.assertEqual(ShotPlanner.boundary_splits(st, plan), [])

    def test_an_already_gapped_boundary_is_not_a_split(self):
        st = self._store((0, 40), (50, 90))
        plan = ShotPlanner.plan_respace(st, gap=20, start_frame=0)
        self.assertEqual(ShotPlanner.boundary_splits(st, plan), [])

    def test_a_shot_absent_from_the_plan_counts_as_stationary(self):
        """Only the following shot moves: the boundary still splits."""
        st = self._store((0, 40), (40, 80))
        plan = ShotPlanner.plan_ripple_downstream(st, 0, 40.0, 15.0)
        self.assertNotIn(0, plan.moves, "the pivot is excluded by construction")
        self.assertEqual(ShotPlanner.boundary_splits(st, plan), [(0, 1, 40.0, 55.0)])


class TestKeyCollisions(unittest.TestCase):
    """Collapsing a gap converges two samples on one frame.

    Maya neither refuses nor overwrites there — it stacks a duplicate a
    fraction of a frame away — so the caller has to know beforehand.
    """

    def test_a_mover_landing_on_a_stationary_key_is_reported(self):
        windows = [(50.0, 90.0, False, False, -10.0)]  # B collapses onto A
        self.assertEqual(
            ShotPlanner.key_collisions(windows, [10.0, 40.0, 50.0, 70.0]),
            [(40.0, [50.0], [40.0])],
        )

    def test_a_clear_destination_is_not_reported(self):
        windows = [(50.0, 90.0, False, False, -10.0)]
        self.assertEqual(ShotPlanner.key_collisions(windows, [10.0, 50.0, 70.0]), [])

    def test_keys_that_travel_together_never_collide(self):
        """The whole window shifts, so its keys keep their spacing."""
        windows = [(0.0, 100.0, False, False, 25.0)]
        self.assertEqual(ShotPlanner.key_collisions(windows, [10.0, 20.0, 30.0]), [])

    def test_two_movers_converging_on_one_frame_are_reported(self):
        windows = [
            (0.0, 45.0, False, False, 10.0),
            (45.0, 90.0, False, False, -10.0),
        ]
        got = ShotPlanner.key_collisions(windows, [30.0, 50.0])
        self.assertEqual(got, [(40.0, [30.0, 50.0], [])])

    def test_a_group_with_no_mover_is_not_a_collision(self):
        """Two stationary keys cannot already share a frame."""
        self.assertEqual(ShotPlanner.key_collisions([], [10.0, 20.0]), [])


class TestPivotMovePlan(unittest.TestCase):
    """One shot moving outside a multi-shot plan still uses the engine's window.

    Both DCC pivot movers derive their envelope from this, so a shared
    sample cannot be assigned one way by the plan path and another by the
    single-shot path — which is exactly how the two movers drifted apart.
    """

    def _store(self, *pairs):
        st = ShotStore()
        st.shots = [
            ShotBlock(i, chr(65 + i), lo, hi, []) for i, (lo, hi) in enumerate(pairs)
        ]
        st.snap_whole_frames = False
        return st

    def test_the_move_carries_the_shots_own_envelope_and_flags(self):
        st = self._store((0, 40), (40, 80))
        plan = ShotPlanner.plan_pivot_move(st, 1, 60)
        mv = plan.moves[1]
        self.assertEqual((mv.old_start, mv.old_end), (40, 80))
        self.assertEqual((mv.new_start, mv.new_end), (60, 100))
        self.assertTrue(mv.env_lo_open, "A closes on 40, so B must decline it")
        self.assertEqual(mv.env_start, 40)

    def test_only_the_named_shot_is_in_the_plan(self):
        st = self._store((0, 40), (40, 80), (80, 120))
        plan = ShotPlanner.plan_pivot_move(st, 1, 60)
        self.assertEqual(list(plan.moves), [1], "nothing is rippled")
        self.assertEqual(plan.parked, [])

    def test_a_zero_delta_move_is_planned_but_not_sequenced(self):
        st = self._store((0, 40), (40, 80))
        plan = ShotPlanner.plan_pivot_move(st, 1, 40)
        self.assertIn(1, plan.moves)
        self.assertFalse(plan.moves[1].moves)
        self.assertEqual(plan.sequence, [], "a shot that does not move is not applied")

    def test_an_unknown_shot_yields_an_empty_plan(self):
        st = self._store((0, 40))
        self.assertEqual(ShotPlanner.plan_pivot_move(st, 99, 10).moves, {})

    def test_the_duration_is_preserved(self):
        st = self._store((10, 55))
        mv = ShotPlanner.plan_pivot_move(st, 0, 100).moves[0]
        self.assertEqual(mv.new_end - mv.new_start, 45)


class TestMoveWindows(unittest.TestCase):
    """The window tuples a collision test consumes come from the engine."""

    def _store(self, *pairs):
        st = ShotStore()
        st.shots = [
            ShotBlock(i, chr(65 + i), lo, hi, []) for i, (lo, hi) in enumerate(pairs)
        ]
        return st

    def test_only_moving_shots_contribute_a_window(self):
        st = self._store((0, 40), (40, 80))
        plan = ShotPlanner.plan_respace(st, gap=10, start_frame=0)
        # A stays at [0,40]; only B moves.
        self.assertEqual(len(ShotPlanner.move_windows(plan)), 1)

    def test_each_window_carries_bounds_flags_and_delta(self):
        st = self._store((0, 40), (40, 80))
        plan = ShotPlanner.plan_respace(st, gap=10, start_frame=0)
        lo, hi, lo_open, hi_closed, delta = ShotPlanner.move_windows(plan)[0]
        self.assertEqual((lo, delta), (40, 10.0))
        self.assertTrue(lo_open, "B declines the sample A closes on")
        self.assertIsInstance(hi_closed, bool)
        self.assertGreater(hi, lo)

    def test_a_plan_that_moves_nothing_has_no_windows(self):
        st = self._store((0, 40), (50, 90))
        plan = ShotPlanner.plan_respace(st, gap=10, start_frame=0)
        self.assertEqual(ShotPlanner.move_windows(plan), [])

    def test_the_windows_feed_key_collisions_directly(self):
        """The two are a matched pair — pin the shape by using it."""
        st = self._store((0, 40), (50, 90))
        plan = ShotPlanner.plan_respace(st, gap=0, start_frame=0)
        windows = ShotPlanner.move_windows(plan)
        self.assertEqual(
            ShotPlanner.key_collisions(windows, [10.0, 40.0, 50.0]),
            [(40.0, [50.0], [40.0])],
            "B's opening sample collapses onto A's closing one",
        )


if __name__ == "__main__":
    unittest.main()
