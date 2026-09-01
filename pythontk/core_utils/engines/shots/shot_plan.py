# coding=utf-8
"""Pure planning layer for multi-shot topology transformations.

NO DCC IMPORTS — this module is part of the shot *model* layer and must not
import ``maya.cmds`` / ``bpy``.  It computes WHAT should move WHERE given a
:class:`ShotStore`, producing a :class:`MovePlan` dataclass with no side
effects.  Anything that actually writes to a DCC scene lives in the sibling
module :mod:`shot_apply` (via injected writer callables).

Two-stage discipline for every multi-shot operation:
    1. Build a :class:`MovePlan` from the current :class:`ShotStore`.
    2. Hand the plan to :func:`shot_apply.apply`.

The split exists because interleaved resolve → mutate loops (the old
``respace`` / ``_ripple_*`` shape) corrupted keyframes when a shot's
new envelope overlapped an unmoved neighbor's old envelope.  Keeping
planning pure makes that bug unwritable here.

The core shots layer is therefore complete on its own: :mod:`shot_model`
models the topology, :mod:`shot_plan` resolves transformations, and
:mod:`shot_apply` commits them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pythontk.core_utils.engines.shots.shot_model import ShotStore


# Sentinel used for an unbounded envelope edge on the last shot.
_INF = 1.0e9
_EPS = 1.0e-6


class ShotBoundaryConflict(RuntimeError):
    """Two poses would be forced onto one sample by collapsing a gap.

    A frame holds one key per curve, so when shots become contiguous the
    preceding shot's closing sample and the following shot's opening sample
    have to become the same key.  That is lossless only if they already
    agree; a hard cut between the shots does not, and no ownership rule can
    make one frame hold both poses.  The operation is refused before it
    writes anything rather than silently dropping a pose.

    ``conflicts`` is ``[(curve, frame, [value, ...]), ...]`` — enough for a
    caller to name the curves and let the user widen the gap instead.
    """

    def __init__(self, conflicts: Sequence[Tuple[str, float, Sequence[float]]]):
        self.conflicts = list(conflicts)
        names = sorted({str(c[0]) for c in self.conflicts})
        shown = ", ".join(names[:5]) + (" ..." if len(names) > 5 else "")
        super().__init__(
            f"Collapsing this gap would force {len(self.conflicts)} pair(s) of "
            f"different poses onto one frame ({shown}). Use a gap of at least "
            "1 frame so each shot keeps its own opening and closing pose."
        )


class _ShotPlannerInternal(object):
    """Internal helpers for ShotPlanner."""

    @staticmethod
    def _overlaps(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
        return a_lo < b_hi and b_lo < a_hi

    @staticmethod
    def _plan_sequence(moves: Dict[int, ShotMove]) -> tuple:
        """Topo-sort shot_ids so each shot moves before any shot whose new
        envelope lands inside its current (old) envelope.

        If j's *new* envelope overlaps i's *old* envelope then executing j
        first would deposit j's keys inside i's source window and corrupt
        i's subsequent read.  The correct order is therefore i → j.

        Only shots whose ``moves`` flag is True are considered.  Returns
        ``(sequence, parked)``: shots left over when the dependency graph
        contains a cycle (mixed-sign deltas, e.g. one shot moving forward
        while another moves backward through it) can't be safely ordered
        and are returned in ``parked`` for the executor's temp-parking
        pass.  ``sequence`` alone is valid provided parked shots are out
        of the way while it runs.
        """
        active = [m for m in moves.values() if m.moves]
        if not active:
            return [], []

        incoming: Dict[int, int] = {m.shot_id: 0 for m in active}
        outgoing: Dict[int, List[int]] = {m.shot_id: [] for m in active}

        for i in active:
            for j in active:
                if i.shot_id == j.shot_id:
                    continue
                j_new_lo = j.env_start + j.delta
                j_new_hi = j.env_end + j.delta
                if _ShotPlannerInternal._overlaps(
                    j_new_lo, j_new_hi, i.env_start, i.env_end
                ):
                    outgoing[i.shot_id].append(j.shot_id)
                    incoming[j.shot_id] += 1

        ready = [sid for sid, deg in incoming.items() if deg == 0]
        out: List[int] = []
        while ready:
            sid = ready.pop()
            out.append(sid)
            for nxt in outgoing[sid]:
                incoming[nxt] -= 1
                if incoming[nxt] == 0:
                    ready.append(nxt)

        emitted = set(out)
        parked = [m.shot_id for m in active if m.shot_id not in emitted]
        return out, parked

    @staticmethod
    def _content_top(moves: Dict[int, ShotMove]) -> float:
        """Highest finite content edge across all moves — the top of real
        keyframe/envelope territory, excluding the +INF last-shot sentinel.

        Both the park offset (which must clear it) and the applier's INF-
        envelope cap (which must stay below the park zone that sits above it)
        derive from this single high-water mark, so they can never disagree.
        """
        hi = 0.0
        for m in moves.values():
            hi = max(hi, m.old_end, m.new_end)
            for v in (m.env_end, m.env_end + m.delta):
                if v < _INF / 2:  # skip the unbounded last-shot sentinel
                    hi = max(hi, v)
        return hi

    @staticmethod
    def _park_offset(moves: Dict[int, ShotMove], parked: List[int]) -> float:
        """Offset that places parked envelopes beyond every other envelope.

        Any offset that clears the highest old/new envelope edge works —
        parked shots translate rigidly, so their relative layout (and thus
        their mutual disjointness) is preserved at the parked location.  The
        +1000 headroom leaves a wide, precision-safe gap between real content
        and the park zone — :func:`shot_apply.apply` caps the last shot's
        +INF envelope just above ``_content_top`` so no move window ever
        reaches into that zone.
        """
        lo = min(moves[sid].env_start for sid in parked)
        return float(round((_ShotPlannerInternal._content_top(moves) - lo) + 1000.0))

    @staticmethod
    def _finalize_plan(moves: Dict[int, ShotMove]) -> MovePlan:
        """Assemble a :class:`MovePlan` from resolved moves (shared tail of
        every plan constructor)."""
        sequence, parked = _ShotPlannerInternal._plan_sequence(moves)
        park_offset = (
            _ShotPlannerInternal._park_offset(moves, parked) if parked else 0.0
        )
        return MovePlan(
            moves=moves, sequence=sequence, parked=parked, park_offset=park_offset
        )

    @staticmethod
    def _envelope_for(sorted_shots: List, index: int) -> tuple:
        """Return ``(env_start, env_end, lo_open, hi_closed)`` for a shot.

        Envelope rule: ``env_start`` is the shot's own ``start``; ``env_end``
        is the next shot's ``start`` if one exists, otherwise ``+INF`` so a
        final shot's trailing content (including fade tails) travels with it.

        **Fencepost rule.**  A shot spans the samples ``start..end``, so two
        contiguous shots SHARE one sample: the preceding shot's closing
        sample IS the following shot's opening sample (it is the same frame
        number).  That sample belongs to the PRECEDING shot — the convention
        every other site already follows (the drag path treats
        ``t > shot.end`` as outside, ``scaleKey`` lands the last key ON
        ``end``, cluster detection sets ``end`` to the last key, and
        ``fbx_takes`` bake ``[start, end]`` inclusive).  The two returned
        flags carry that decision to every consumer:

        ``hi_closed``
            The next shot starts exactly on this shot's ``end`` — the upper
            bound is INCLUSIVE, so the shared sample moves with this shot.
        ``lo_open``
            The previous shot ends exactly on this shot's ``start`` — the
            lower bound is EXCLUSIVE, because that shared sample is the
            previous shot's closing fencepost, not this shot's to move.

        Adjacent shots always agree (both flags derive from one test on the
        same pair), so every whole frame is inside exactly one envelope —
        none is claimed twice and none falls between.  (Strictly, that holds
        outside the writers' float tolerance: a key authored within an
        epsilon of a bound sits in the tolerance band both sides pad by.
        Whole-frame snapping, the default, keeps keys out of it.)  With a gap
        the bounds keep their old half-open shape: ``[start, next.start)``
        still carries trailing-gap fade tails with the shot they belong to.
        """
        shot = sorted_shots[index]
        env_start = shot.start
        has_next = index + 1 < len(sorted_shots)
        env_end = sorted_shots[index + 1].start if has_next else _INF
        hi_closed = has_next and abs(env_end - shot.end) <= _EPS
        lo_open = index > 0 and abs(sorted_shots[index - 1].end - shot.start) <= _EPS
        return env_start, env_end, lo_open, hi_closed


class ShotPlanner(_ShotPlannerInternal):
    """ShotPlanner — module namespace."""

    @staticmethod
    def envelope_for(sorted_shots: List, index: int) -> tuple:
        """``(env_start, env_end, lo_open, hi_closed)`` for one shot's window.

        Public entry point to the fencepost rule, for callers that move a
        single shot outside a :class:`MovePlan` and still have to agree with
        the planner about which shot owns a shared sample.

        Parameters:
            sorted_shots: Shots in timeline order.
            index: Position of the shot in *sorted_shots*.

        Returns:
            The shot's owned key window and its two boundary-closure flags.
        """
        return _ShotPlannerInternal._envelope_for(sorted_shots, index)

    @staticmethod
    def in_window(
        t: float,
        lo: float,
        hi: float,
        lo_open: bool = False,
        hi_closed: bool = False,
        eps: float = _EPS,
    ) -> bool:
        """Is sample *t* inside the window ``[lo, hi)`` as shaped by the flags?

        The single definition of envelope membership.  ``lo_open`` makes the
        lower bound exclusive and ``hi_closed`` makes the upper bound
        inclusive — see :meth:`_ShotPlannerInternal._envelope_for` for the
        fencepost rule that sets them.  Because adjacent shots always set
        them consistently, this predicate partitions the timeline: every
        sample is in exactly one shot's window.

        Parameters:
            t: The sample (frame) to test.
            lo: Window start.
            hi: Window end.
            lo_open: Exclude a sample sitting exactly on *lo*.
            hi_closed: Include a sample sitting exactly on *hi*.
            eps: Float tolerance on both bounds.

        Returns:
            ``True`` when *t* belongs to this window.
        """
        if lo_open:
            if t <= lo + eps:
                return False
        elif t < lo - eps:
            return False
        return t <= hi + eps if hi_closed else t < hi - eps

    @staticmethod
    def objects_to_adopt(
        keyed: Dict[str, Sequence[float]],
        owned: Optional[Iterable[str]],
        lo: float,
        hi: float,
        lo_open: bool = False,
        hi_closed: bool = False,
        eps: float = _EPS,
    ) -> List[str]:
        """Objects a moving shot may claim over the window it is about to move.

        A mover shifts a shot's OWN object list within a window, so anything
        keyed inside that window but missing from the list is left behind:
        the shot moves and part of its animation does not.  Membership goes
        stale for ordinary reasons invisible beforehand — a renamed object
        leaves an entry that resolves to nothing, and an object animated
        after the shots were authored belongs to no shot at all.

        Adoption is purely the mover's own window: a key inside it is going
        to be moved, so the object holding it must be listed or it is left
        behind.  No ownership exemption is needed — :meth:`in_window`
        partitions the timeline, so a key at a shared boundary is inside
        exactly one shot's window and can never be claimed twice.

        Parameters:
            keyed: ``{object_name: [key_time, ...]}`` for every candidate.
            owned: Names the shot already lists (adopted objects exclude these).
            lo: Window start.
            hi: Window end.
            lo_open: Exclude a key sitting exactly on *lo* (the preceding
                shot's closing fencepost).
            hi_closed: Include a key sitting exactly on *hi* (this shot's own
                closing fencepost, shared with the next shot's start).
            eps: Float tolerance on the window bounds.

        Returns:
            Sorted names to add to the shot.
        """
        owned = set(owned or ())
        add = []
        for name, times in keyed.items():
            if name in owned:
                continue
            if any(
                ShotPlanner.in_window(t, lo, hi, lo_open, hi_closed, eps) for t in times
            ):
                add.append(name)
        return sorted(add)

    @staticmethod
    def move_windows(
        plan: MovePlan,
    ) -> List[Tuple[float, float, bool, bool, float]]:
        """The windows *plan* will actually move, shaped for :meth:`key_collisions`.

        Keeps the tuple's shape an engine detail instead of a contract each
        DCC re-spells; a field added here reaches both writers at once.

        Parameters:
            plan: The resolved plan about to be applied.

        Returns:
            ``[(lo, hi, lo_open, hi_closed, delta), ...]`` — one per shot
            that moves.  Non-moving shots are omitted: their keys stay put,
            which is what a collision test means by "stationary".
        """
        return [
            (m.env_start, m.env_end, m.env_lo_open, m.env_hi_closed, m.delta)
            for m in plan.moves.values()
            if m.moves
        ]

    @staticmethod
    def plan_pivot_move(
        store: ShotStore,
        shot_id: int,
        new_start: float,
    ) -> MovePlan:
        """A plan that moves one shot to ``new_start``, rippling nothing.

        For the movers that shift a single shot outside a multi-shot plan:
        they still need that shot's envelope — fencepost flags included —
        and a plan to reconcile boundaries against.  Deriving those
        separately per DCC is how the two key movers drifted apart in the
        first place, so the derivation lives here.

        The envelope comes from the layout as it stands NOW, which is the
        point: when a ripple has already moved a neighbour away there is no
        shared sample left to decline, and when it has not, the neighbour
        still owns its own fencepost.

        Parameters:
            store: Store holding the current layout.
            shot_id: The shot to move.
            new_start: Desired first frame (snapped by the store).

        Returns:
            A :class:`MovePlan` with that single move, or an empty plan when
            the store does not hold the shot.
        """
        shots = store.sorted_shots()
        idx = next((i for i, s in enumerate(shots) if s.shot_id == shot_id), None)
        if idx is None:
            return MovePlan()
        shot = shots[idx]
        env_start, env_end, lo_open, hi_closed = _ShotPlannerInternal._envelope_for(
            shots, idx
        )
        start = store.snap(new_start)
        move = ShotMove(
            shot_id=shot_id,
            old_start=shot.start,
            old_end=shot.end,
            new_start=start,
            new_end=store.snap(start + (shot.end - shot.start)),
            env_start=env_start,
            env_end=env_end,
            env_lo_open=lo_open,
            env_hi_closed=hi_closed,
        )
        return MovePlan(moves={shot_id: move}, sequence=[shot_id] if move.moves else [])

    @staticmethod
    def boundary_splits(
        store: ShotStore,
        plan: MovePlan,
        eps: float = _EPS,
    ) -> List[Tuple[int, int, float, float]]:
        """Shared samples this plan pulls apart.

        Two contiguous shots share one sample, which the fencepost rule gives
        to the PRECEDING shot.  When a plan opens a gap between them that
        sample stops being shared: it stays with the preceding shot, and the
        following shot — whose opening pose it also was — is left starting on
        nothing.  Each entry names both shots, the frame the shared sample
        sat on, and the frame the following shot now opens at, so the caller
        can carry an opening pose across.  The inverse, two samples
        converging onto one, is :meth:`key_collisions`.

        Both ids are reported because the split is only a *split* on curves
        BOTH shots animate.  Where only the following shot has keys, that
        sample was never shared — it is that shot's opening pose alone and
        should travel with it, not stay behind on a curve its neighbour has
        no stake in.  The caller decides that per curve; this reports the
        boundary.

        Shots absent from the plan are treated as stationary.

        Parameters:
            store: Store holding the pre-move layout.
            plan: The resolved plan about to be applied.
            eps: Float tolerance for contiguity and gap tests.

        Returns:
            ``[(preceding_id, following_id, boundary_frame, new_start), ...]``
        """
        shots = store.sorted_shots()
        out: List[Tuple[int, int, float, float]] = []
        for i in range(len(shots) - 1):
            prev, nxt = shots[i], shots[i + 1]
            if abs(nxt.start - prev.end) > eps:
                continue  # not sharing a sample to begin with
            prev_mv, next_mv = plan.moves.get(prev.shot_id), plan.moves.get(nxt.shot_id)
            new_prev_end = prev_mv.new_end if prev_mv else prev.end
            new_next_start = next_mv.new_start if next_mv else nxt.start
            if new_next_start - new_prev_end > eps:
                out.append(
                    (
                        prev.shot_id,
                        nxt.shot_id,
                        float(nxt.start),
                        float(new_next_start),
                    )
                )
        return out

    @staticmethod
    def key_collisions(
        windows: Sequence[Tuple[float, float, bool, bool, float]],
        times: Iterable[float],
        eps: float = 1.0e-3,
    ) -> List[Tuple[float, List[float], List[float]]]:
        """Samples of one curve that a plan would land on the same frame.

        Two keys cannot share a frame: a mover that lands on an occupied one
        neither refuses nor overwrites, it stacks a near-duplicate a
        fraction of a frame away (measured in Maya), which then travels as a
        pair forever.  Collapsing a gap to zero does exactly that — the
        preceding shot's closing sample and the following shot's opening
        sample converge — so callers pre-flight with this and either merge
        (the samples agree, one survives) or refuse (they disagree: a hard
        cut cannot live at gap 0).

        Each window is ``(lo, hi, lo_open, hi_closed, delta)``.  Windows
        partition the timeline, so a key matches at most one; keys in none
        are stationary.

        Parameters:
            windows: The plan's move windows for this curve's owners.
            times: The curve's key times.
            eps: Frames within which two destinations count as the same.

        Returns:
            ``[(destination, moving_sources, stationary_sources), ...]`` for
            every frame that would end up holding more than one key, ordered
            by destination.  A group always has at least one moving source.
        """
        landings: List[Tuple[float, float, bool]] = []
        for t in times:
            delta = 0.0
            for lo, hi, lo_open, hi_closed, d in windows:
                # Same tolerance the writer will use, so what this predicts
                # a key does is what the writer actually does to it.
                if ShotPlanner.in_window(t, lo, hi, lo_open, hi_closed, eps):
                    delta = d
                    break
            landings.append((float(t) + delta, float(t), abs(delta) > _EPS))

        groups: List[Tuple[float, List[float], List[float]]] = []
        for dest, src, moving in sorted(landings):
            for dst, movers, still in groups:
                if abs(dst - dest) <= eps:
                    (movers if moving else still).append(src)
                    break
            else:
                groups.append((dest, [src] if moving else [], [] if moving else [src]))
        return [g for g in groups if len(g[1]) + len(g[2]) > 1 and g[1]]

    @staticmethod
    def plan_respace(store: ShotStore, gap: float, start_frame: float) -> MovePlan:
        """Build a plan that lays shots out sequentially with uniform gaps.

        Locked gaps preserve their current width.  Durations are preserved;
        only start frames change.  All new positions are snapped through
        ``store.snap`` so the in-memory model stays integer-clean.
        """
        shots = store.sorted_shots()
        if not shots:
            return MovePlan()

        locked_widths: dict = {}
        for i in range(len(shots) - 1):
            if store.is_gap_locked(shots[i].shot_id, shots[i + 1].shot_id):
                locked_widths[i] = max(0.0, shots[i + 1].start - shots[i].end)

        moves: Dict[int, ShotMove] = {}
        cursor = start_frame
        for i, shot in enumerate(shots):
            duration = shot.end - shot.start
            new_start = store.snap(cursor)
            new_end = store.snap(new_start + duration)
            env_start, env_end, lo_open, hi_closed = _ShotPlannerInternal._envelope_for(
                shots, i
            )
            moves[shot.shot_id] = ShotMove(
                shot_id=shot.shot_id,
                old_start=shot.start,
                old_end=shot.end,
                new_start=new_start,
                new_end=new_end,
                env_start=env_start,
                env_end=env_end,
                env_lo_open=lo_open,
                env_hi_closed=hi_closed,
            )
            effective_gap = locked_widths.get(i, gap)
            cursor = new_end + effective_gap

        return _ShotPlannerInternal._finalize_plan(moves)

    @staticmethod
    def plan_gap_retimes(store: ShotStore, plan: MovePlan) -> List[GapRetime]:
        """Every gap in *plan* whose width changes, as a :class:`GapRetime`.

        Derived from a finished plan rather than built alongside one, so every
        constructor here (respace, ripple, slide, reorder) gets the same answer
        from the same rule and a pure translation -- where each gap keeps its
        width -- correctly yields nothing.

        A shot the plan does not mention does not move, which is exactly the
        ``delta = 0`` this reads: an unmoved neighbour is what makes a gap
        change width in the first place.

        Returns them in timeline order; the caller decides when to run each,
        which matters because the safe moment differs (see the two-stage rule
        in ``mayatk.anim_utils.shots._shot_apply``).
        """
        shots = store.sorted_shots()
        retimes: List[GapRetime] = []
        for left, right in zip(shots, shots[1:]):
            lo, hi = left.end, right.start
            if hi - lo <= _EPS:
                continue  # contiguous: no gap, and nothing in one to retime
            left_move = plan.moves.get(left.shot_id)
            right_move = plan.moves.get(right.shot_id)
            left_delta = left_move.delta if left_move else 0.0
            new_lo = left_move.new_end if left_move else left.end
            new_hi = right_move.new_start if right_move else right.start
            new_width = max(0.0, new_hi - new_lo)
            if abs(new_width - (hi - lo)) <= _EPS:
                continue
            retimes.append(
                GapRetime(
                    left_id=left.shot_id,
                    right_id=right.shot_id,
                    lo=lo,
                    hi=hi,
                    new_width=new_width,
                    left_delta=left_delta,
                )
            )
        return retimes

    @staticmethod
    def plan_ripple_downstream(
        store: ShotStore,
        pivot_shot_id: int,
        after_frame: float,
        delta: float,
    ) -> MovePlan:
        """Build a plan that shifts every shot starting at or after
        ``after_frame`` by ``delta`` frames.

        The pivot shot is excluded — the caller's primary edit already
        placed it.  Snapping is applied to the resulting bounds.
        """
        shots = store.sorted_shots()
        if not shots or abs(delta) < _EPS:
            return MovePlan()

        moves: Dict[int, ShotMove] = {}
        for i, shot in enumerate(shots):
            if shot.shot_id == pivot_shot_id:
                continue
            if shot.start < after_frame:
                continue
            env_start, env_end, lo_open, hi_closed = _ShotPlannerInternal._envelope_for(
                shots, i
            )
            moves[shot.shot_id] = ShotMove(
                shot_id=shot.shot_id,
                old_start=shot.start,
                old_end=shot.end,
                new_start=store.snap(shot.start + delta),
                new_end=store.snap(shot.end + delta),
                env_start=env_start,
                env_end=env_end,
                env_lo_open=lo_open,
                env_hi_closed=hi_closed,
            )

        return _ShotPlannerInternal._finalize_plan(moves)

    @staticmethod
    def plan_reorder(
        store: ShotStore,
        shot_id: int,
        target_pos: int,
        gap: float,
    ) -> MovePlan:
        """Build a plan that moves ``shot_id`` to 1-based timeline position ``target_pos``.

        The shot is lifted from its current slot and re-inserted at ``target_pos``
        (clamped to ``[1, n]``); the whole set is then laid out sequentially from the
        earliest current start, each shot preserving its duration.  Inter-shot gaps
        use ``gap``, except a gap that is *locked* **and** whose two shots were already
        adjacent keeps its current width.  Because a reorder makes one shot's move
        cross others', the resulting ``moves`` almost always form a collision cycle —
        :func:`_finalize_plan` resolves that into the park / ordered / land structure
        that :func:`shot_apply.apply` executes, so no bespoke park/land loop is needed.

        Envelopes describe each shot's *current* (pre-move) owned key window, so the
        executor reads keys from where they are now and lands them at the reordered
        positions.  Returns an empty plan when there are fewer than two shots, the id
        is unknown, or the position is unchanged.
        """
        shots = store.sorted_shots()
        if len(shots) < 2:
            return MovePlan()
        ids = [s.shot_id for s in shots]
        if shot_id not in ids:
            return MovePlan()

        cur_idx = ids.index(shot_id)
        target_idx = max(0, min(int(target_pos) - 1, len(shots) - 1))
        if target_idx == cur_idx:
            return MovePlan()

        # Current owned key windows + adjacency widths, captured before any move.
        old_env = {
            s.shot_id: _ShotPlannerInternal._envelope_for(shots, i)
            for i, s in enumerate(shots)
        }
        old_width = {
            (shots[i].shot_id, shots[i + 1].shot_id): max(
                0.0, shots[i + 1].start - shots[i].end
            )
            for i in range(len(shots) - 1)
        }

        new_order = list(shots)
        moving = new_order.pop(cur_idx)
        new_order.insert(target_idx, moving)

        moves: Dict[int, ShotMove] = {}
        cursor = shots[0].start
        for i, shot in enumerate(new_order):
            duration = shot.end - shot.start
            new_start = store.snap(cursor)
            new_end = store.snap(new_start + duration)
            env_start, env_end, lo_open, hi_closed = old_env[shot.shot_id]
            moves[shot.shot_id] = ShotMove(
                shot_id=shot.shot_id,
                old_start=shot.start,
                old_end=shot.end,
                new_start=new_start,
                new_end=new_end,
                env_start=env_start,
                env_end=env_end,
                env_lo_open=lo_open,
                env_hi_closed=hi_closed,
            )
            if i < len(new_order) - 1:
                pair = (shot.shot_id, new_order[i + 1].shot_id)
                if store.is_gap_locked(*pair) and pair in old_width:
                    effective_gap = old_width[pair]
                else:
                    effective_gap = gap
                cursor = new_end + effective_gap

        return _ShotPlannerInternal._finalize_plan(moves)

    @staticmethod
    def plan_ripple_upstream(
        store: ShotStore,
        pivot_shot_id: int,
        before_frame: float,
        delta: float,
    ) -> MovePlan:
        """Build a plan that shifts every shot ending at or before
        ``before_frame`` by ``delta`` frames.
        """
        shots = store.sorted_shots()
        if not shots or abs(delta) < _EPS:
            return MovePlan()

        moves: Dict[int, ShotMove] = {}
        for i, shot in enumerate(shots):
            if shot.shot_id == pivot_shot_id:
                continue
            if shot.end > before_frame + _EPS:
                continue
            env_start, env_end, lo_open, hi_closed = _ShotPlannerInternal._envelope_for(
                shots, i
            )
            moves[shot.shot_id] = ShotMove(
                shot_id=shot.shot_id,
                old_start=shot.start,
                old_end=shot.end,
                new_start=store.snap(shot.start + delta),
                new_end=store.snap(shot.end + delta),
                env_start=env_start,
                env_end=env_end,
                env_lo_open=lo_open,
                env_hi_closed=hi_closed,
            )

        return _ShotPlannerInternal._finalize_plan(moves)


@dataclass
class ShotMove:
    """A single shot's source and destination ranges.

    ``env_start`` / ``env_end`` describe the *owned* keyframe window —
    typically ``[old_start, next_shot.old_start)``.  Extending the
    window to the next shot's start ensures fade tails that live in
    the trailing gap travel with their owning shot rather than being
    stranded by a tight ``[old_start, old_end]`` key window.

    ``env_lo_open`` / ``env_hi_closed`` carry the fencepost decision for a
    SHARED sample — see :meth:`_ShotPlannerInternal._envelope_for`.  Both
    default to the plain half-open window, so a move built by hand (rather
    than from a shot's neighbours) behaves exactly as before.
    """

    shot_id: int
    old_start: float
    old_end: float
    new_start: float
    new_end: float
    env_start: float
    env_end: float
    env_lo_open: bool = False
    env_hi_closed: bool = False

    @property
    def delta(self) -> float:
        return self.new_start - self.old_start

    @property
    def moves(self) -> bool:
        return abs(self.delta) > _EPS


@dataclass
class MovePlan:
    """Resolved multi-shot timeline mutation.

    ``moves`` is keyed by ``shot_id`` and covers every shot considered
    by the planner.  ``sequence`` is the execution order the executor
    must honour to avoid transient envelope collisions.  Only shots
    that actually move appear in ``sequence``.

    ``parked`` lists shots whose moves form a collision cycle (each
    would deposit keys inside another's unread source window — e.g. a
    respace where one shot moves forward while another moves backward).
    The executor must first shift their keys out of the way by
    ``park_offset``, run ``sequence``, then land them at their final
    positions (``delta - park_offset`` from the parked location).
    """

    moves: Dict[int, ShotMove] = field(default_factory=dict)
    sequence: List[int] = field(default_factory=list)
    parked: List[int] = field(default_factory=list)
    park_offset: float = 0.0


@dataclass
class GapRetime:
    """One inter-shot gap whose WIDTH changes, and where its content must land.

    A shot move is rigid: the shot keeps its duration, so its content can
    travel with it unchanged.  A gap is the opposite -- respacing is defined by
    changing its width -- and content living in one therefore has to be
    RETIMED, not carried.  Moving it rigidly is what breaks a respace: measured
    on a 12-shot production assembly, collapsing a 135-frame gap to 15 left the
    gap's own key sitting 13 frames PAST the following shot's content, and the
    preceding shot -- which had not moved at all -- lost 42 of its 109 frames
    to the tangent change that key's new neighbour caused.

    The retime is anchored at the gap's LEFT edge, because that edge is the
    preceding shot's end and travels with it: content at ``lo + d`` lands at
    ``lo + d * scale``.

    Fields are in the ORIGINAL timeline; :attr:`left_delta` is how far the
    preceding shot moves, which is also how far this gap's left edge moves.
    """

    left_id: int
    right_id: int
    lo: float
    hi: float
    new_width: float
    left_delta: float

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def scale(self) -> float:
        """Time factor about the gap's left edge (0.0 collapses the gap)."""
        return 0.0 if self.width <= _EPS else self.new_width / self.width

    @property
    def shrinks(self) -> bool:
        return self.new_width < self.width - _EPS

    @property
    def grows(self) -> bool:
        return self.new_width > self.width + _EPS


# ---------------------------------------------------------------------------
# Sequence (collision-safe topo sort)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plan constructors
# ---------------------------------------------------------------------------
