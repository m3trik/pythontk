# coding=utf-8
"""Commit a resolved :class:`MovePlan` via injected writer callables.

Pure orchestration skeleton that walks a plan built by
:mod:`pythontk.core_utils.engines.shots.shot_plan` in its predetermined
execution order.  It contains NO DCC imports: the actual keyframe / audio writes are
supplied by the caller as *strategy callables* (``move_keys`` / ``shift_audio``).
mayatk and blendertk pass closures that write to their scene; the pure core
falls back to a bounds-only commit when no ``move_keys`` is given.

Three-phase discipline (only when ``move_keys`` is supplied):
    Phase 0 — park cycle members' keys beyond every envelope.
    Phase 1 — apply the collision-safe ordered moves, committing bounds.
    Phase 2 — land the parked shots at their final positions.

The +INF last-shot envelope is capped just above the plan's real content top
whenever any shot is parked, so an unbounded move window can never sweep
already-parked content into a second shift.
"""
from typing import Callable, Iterable, Optional

from pythontk.core_utils.engines.shots.shot_model import ShotStore
from pythontk.core_utils.engines.shots.shot_plan import MovePlan, _INF, _content_top


# ``move_keys`` protocol:
#     move_keys(objects, env_lo, env_hi, delta, over=False) -> None
# Shift the keys of *objects* whose time falls in the half-open envelope
# ``[env_lo, env_hi)`` by *delta* frames.  ``over=True`` (used by the park /
# land phases) must let keys pass neighbouring keys on the same curve — those
# phases teleport a shot's keys across other shots' content, and a clamping
# "move" semantic would strand them.  The ordered phase uses ``over=False``:
# the plan's topological order guarantees those moves never cross.
MoveKeys = Callable[..., None]

# ``shift_audio`` protocol:
#     shift_audio(env_lo, env_hi, delta) -> None
# Shift audio keys within the envelope by *delta*.  Optional; ``None`` skips
# audio entirely (the common pure case).
ShiftAudio = Callable[[float, float, float], None]


def apply(
    plan: MovePlan,
    store: ShotStore,
    move_keys: Optional[MoveKeys] = None,
    shift_audio: Optional[ShiftAudio] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    """Execute ``plan`` against ``store`` (and, via ``move_keys``, a scene).

    When ``move_keys`` is ``None`` (the pure default) only the in-memory shot
    bounds are committed — the graceful-degradation path shared with headless
    callers.  When a ``move_keys`` strategy is supplied, the full three-phase
    park / ordered / land treatment runs, invoking ``move_keys`` (and, if
    given, ``shift_audio``) for each shot's owned envelope before committing
    its new bounds.

    Parameters:
        plan: The resolved :class:`MovePlan` to execute.
        store: The :class:`ShotStore` whose shot bounds are committed.
        move_keys: Scene keyframe writer (see :data:`MoveKeys`).  ``None`` →
            bounds-only commit.
        shift_audio: Scene audio writer (see :data:`ShiftAudio`).  ``None`` →
            no audio shifting.
        progress_callback: Optional ``(current, total, message)`` reporter,
            invoked once per shot plus a final "Done".
    """
    if not plan.sequence and not plan.parked:
        return

    total = len(plan.sequence) + len(plan.parked)
    shots_by_id = {s.shot_id: s for s in store.shots}

    # ---- bounds-only path (no scene writer) ------------------------------
    if move_keys is None:
        for i, shot_id in enumerate(plan.sequence + plan.parked):
            if progress_callback:
                progress_callback(i, total, f"Applying shot: {shot_id}")
            move = plan.moves[shot_id]
            shot = shots_by_id.get(shot_id)
            if shot is not None:
                shot.start = move.new_start
                shot.end = move.new_end
        store.mark_dirty()
        if progress_callback and total:
            progress_callback(total, total, "Done")
        return

    # ---- writer-backed three-phase path ----------------------------------
    park = plan.park_offset

    # The timeline-last shot's envelope end is the +INF sentinel so its
    # trailing content travels with it.  Propagating that literal 1e9 into a
    # move window — and, after parking, into ``1e9 + park`` — is hazardous:
    # the unbounded window would sweep already-parked content into a second
    # shift.  When any shot is parked, cap +INF at the plan's real content top
    # plus one frame; the park zone sits ``+1000`` frames above content, so a
    # capped window clears it by a wide, precision-safe margin while still
    # covering all of the last shot's own content.
    cap = _content_top(plan.moves) + 1.0 if plan.parked else _INF

    def _capped(env_end: float) -> float:
        return env_end if env_end < _INF / 2 else cap

    def _audio(env_lo: float, env_hi: float, delta: float) -> None:
        if shift_audio is not None:
            shift_audio(env_lo, env_hi, delta)

    # Phase 0 — park cycle members' keys beyond every envelope.
    for shot_id in plan.parked:
        move = plan.moves[shot_id]
        shot = shots_by_id.get(shot_id)
        if shot is None:
            continue
        move_keys(shot.objects, move.env_start, _capped(move.env_end), park, over=True)
        _audio(move.env_start, _capped(move.env_end), park)

    # Phase 1 — ordered moves.
    for i, shot_id in enumerate(plan.sequence):
        if progress_callback:
            progress_callback(i, total, f"Applying shot: {shot_id}")
        move = plan.moves[shot_id]
        shot = shots_by_id.get(shot_id)
        if shot is None:
            continue
        move_keys(shot.objects, move.env_start, _capped(move.env_end), move.delta)
        _audio(move.env_start, _capped(move.env_end), move.delta)
        shot.start = move.new_start
        shot.end = move.new_end

    # Phase 2 — land parked shots at their final positions.
    for i, shot_id in enumerate(plan.parked):
        if progress_callback:
            progress_callback(
                len(plan.sequence) + i, total, f"Applying shot: {shot_id}"
            )
        move = plan.moves[shot_id]
        shot = shots_by_id.get(shot_id)
        if shot is None:
            continue
        move_keys(
            shot.objects,
            move.env_start + park,
            _capped(move.env_end) + park,
            move.delta - park,
            over=True,
        )
        _audio(move.env_start + park, _capped(move.env_end) + park, move.delta - park)
        shot.start = move.new_start
        shot.end = move.new_end

    # Bounds were written directly (not via update_shot) — flag the
    # store so the mutation survives a scene save.
    store.mark_dirty()

    if progress_callback and total:
        progress_callback(total, total, "Done")


# Unused in the pure skeleton but part of the public surface: callers that
# want to type-annotate their strategies can import these protocol aliases.
__all__ = ["apply", "MoveKeys", "ShiftAudio"]
