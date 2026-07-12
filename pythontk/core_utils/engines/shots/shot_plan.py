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
from dataclasses import dataclass, field
from typing import Dict, List

from pythontk.core_utils.engines.shots.shot_model import ShotStore


# Sentinel used for an unbounded envelope edge on the last shot.
_INF = 1.0e9
_EPS = 1.0e-6


@dataclass
class ShotMove:
    """A single shot's source and destination ranges.

    ``env_start`` / ``env_end`` describe the *owned* keyframe window —
    typically ``[old_start, next_shot.old_start)``.  Extending the
    window to the next shot's start ensures fade tails that live in
    the trailing gap travel with their owning shot rather than being
    stranded by a tight ``[old_start, old_end]`` key window.
    """

    shot_id: int
    old_start: float
    old_end: float
    new_start: float
    new_end: float
    env_start: float
    env_end: float

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


# ---------------------------------------------------------------------------
# Sequence (collision-safe topo sort)
# ---------------------------------------------------------------------------


def _overlaps(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    return a_lo < b_hi and b_lo < a_hi


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
            if _overlaps(j_new_lo, j_new_hi, i.env_start, i.env_end):
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
    return float(round((_content_top(moves) - lo) + 1000.0))


def _finalize_plan(moves: Dict[int, ShotMove]) -> MovePlan:
    """Assemble a :class:`MovePlan` from resolved moves (shared tail of
    every plan constructor)."""
    sequence, parked = _plan_sequence(moves)
    park_offset = _park_offset(moves, parked) if parked else 0.0
    return MovePlan(
        moves=moves, sequence=sequence, parked=parked, park_offset=park_offset
    )


# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------


def _envelope_for(sorted_shots: List, index: int) -> tuple:
    """Return ``(env_start, env_end)`` for the shot at ``sorted_shots[index]``.

    Envelope rule: ``env_start`` is the shot's own ``start``; ``env_end``
    is the next shot's ``start`` if one exists, otherwise ``+INF`` so a
    final shot's trailing content (including fade tails) travels with it.
    """
    shot = sorted_shots[index]
    env_start = shot.start
    env_end = (
        sorted_shots[index + 1].start if index + 1 < len(sorted_shots) else _INF
    )
    return env_start, env_end


# ---------------------------------------------------------------------------
# Plan constructors
# ---------------------------------------------------------------------------


def plan_respace(
    store: ShotStore, gap: float, start_frame: float
) -> MovePlan:
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
        env_start, env_end = _envelope_for(shots, i)
        moves[shot.shot_id] = ShotMove(
            shot_id=shot.shot_id,
            old_start=shot.start,
            old_end=shot.end,
            new_start=new_start,
            new_end=new_end,
            env_start=env_start,
            env_end=env_end,
        )
        effective_gap = locked_widths.get(i, gap)
        cursor = new_end + effective_gap

    return _finalize_plan(moves)


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
        env_start, env_end = _envelope_for(shots, i)
        moves[shot.shot_id] = ShotMove(
            shot_id=shot.shot_id,
            old_start=shot.start,
            old_end=shot.end,
            new_start=store.snap(shot.start + delta),
            new_end=store.snap(shot.end + delta),
            env_start=env_start,
            env_end=env_end,
        )

    return _finalize_plan(moves)


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
    old_env = {s.shot_id: _envelope_for(shots, i) for i, s in enumerate(shots)}
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
        env_start, env_end = old_env[shot.shot_id]
        moves[shot.shot_id] = ShotMove(
            shot_id=shot.shot_id,
            old_start=shot.start,
            old_end=shot.end,
            new_start=new_start,
            new_end=new_end,
            env_start=env_start,
            env_end=env_end,
        )
        if i < len(new_order) - 1:
            pair = (shot.shot_id, new_order[i + 1].shot_id)
            if store.is_gap_locked(*pair) and pair in old_width:
                effective_gap = old_width[pair]
            else:
                effective_gap = gap
            cursor = new_end + effective_gap

    return _finalize_plan(moves)


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
        env_start, env_end = _envelope_for(shots, i)
        moves[shot.shot_id] = ShotMove(
            shot_id=shot.shot_id,
            old_start=shot.start,
            old_end=shot.end,
            new_start=store.snap(shot.start + delta),
            new_end=store.snap(shot.end + delta),
            env_start=env_start,
            env_end=env_end,
        )

    return _finalize_plan(moves)
