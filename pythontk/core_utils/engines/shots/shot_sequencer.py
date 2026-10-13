# !/usr/bin/python
# coding=utf-8
"""Shot sequencer core -- ripple-editing orchestration over a :class:`ShotStore`.

The timeline operations every sequencer performs -- define, expand, resize,
slide, ripple, insert, delete, merge, split, pad, respace, apply a gap, move
sequences between shots, fit a shot to its content -- decided once, here, in
plain Python.  What the operations DO to a scene (move keys, retime them, cut
them, shift audio, hold gaps, read where content sits) is a set of narrow
hooks a DCC adapter overrides (Template Method): mayatk's and blendertk's
``ShotSequencer`` subclass this class and supply their scene I/O, so the two
share one orchestration instead of two hand-kept copies.

With no adapter the hooks are pure no-ops and the class is a complete
**bounds-only** sequencer: every operation plans and commits the shot bounds
exactly as a DCC would, and no scene is touched -- the same degradation the
DCC adapters fall back to headless, and what the tests here exercise.

Hook contract (override in a DCC subclass; defaults describe an empty scene):

====================================  ========================================
``STORE_CLASS``                       the :class:`ShotStore` subclass built
                                      when no store is passed
``_scene_available()``                a scene can be read/written
``_content_batch()``                  context grouping a multi-step scene edit
``_apply_plan(plan, retime_gaps)``    commit a :class:`MovePlan` (keys + bounds)
``_find_keyed_transforms(start, end)``  objects keyed in a range
``_member_names(shot, names)``        *names* split into listed / unlisted
``collect_object_segments(shot_id)``  per-object anim segments in a shot
``_collect_audio_sequences(lo, hi)``  audio sequences in a range
``_key_extent(shot, probe, reach)``   a shot's key extent (see DCC docs)
``_cut_passed_bound_keys(...)``       cut redundant keys a moving bound passes
``_reconcile_boundary_keys(...)``     re-home the system's boundary samples
``_gap_hold_seams()`` / ``_release_gap_holds(s)`` / ``_apply_gap_holds(s)``
                                      the inter-shot gap holds
``_pin_bound_keys(pins)``             key curves on shot bounds, changing nothing
``_pin_bound_poses(objs, w, pins)``   key the pose played on bounds, holds too
``_freeze_left_behind(objs, w)``      hold the slopes a leaving run's keys gave
``_poses_differ(pairs, objects)``    whether two frames play different poses
``_gaps_keyed(windows)``              whether the animator keyed inside a gap
``_retime_closing_gaps(retimes)``     retime a closing gap's keys, before moves
``_curve_exists(curve)``              the scene still has a ledger key's curve
``_follow_unresolved_claims(c)``      re-key claims whose owner was renamed
``_reconcile_boundaries(plan)``       fencepost reconcile around a plan write
``_move_content_keys(...)``           move keyed content inside an envelope
``_shift_audio(lo, hi, delta)``       shift audio inside a range
``_move_audio_sequence(seq, delta)``  move one audio sequence
``move_attribute_keys(...)``          move one object's (attribute's) keys
``copy_attribute_keys(...)``          copy them, the originals left in place
``_point_clip_curves(obj, t, ...)``   the curves a point clip at *t* stands for
``_move_point_key(obj, curve, ...)``  move one curve's key at a frame
``scale_shot_keys`` / ``scale_object_keys``  retime keys into a new span
``_cut_shot_content(shot_id)``        delete a shot's keys
``_keyed_transform_times()``          the scene's keyed content objects
``_trailing_content_extent(shot)``    how far a shot's content trails its end
``_drop_claimed_seam_copy(...)``      drop a source's copy of a claimed seam
``_object_key_probe(obj)``            ``frame -> bool``: *obj* keyed there
``_animator_times_in(obj, attr, w)``  an object's animator key times in *w*
``detect_shots(...)``                 shot regions detected in the scene
====================================  ========================================
"""

from __future__ import annotations

import bisect
import contextlib
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from pythontk.core_utils.engines.shots.shot_apply import ShotApply
from pythontk.core_utils.engines.shots.shot_model import ShotBlock, ShotStore
from pythontk.core_utils.engines.shots.shot_plan import GapRetime, ShotPlanner

# Below this a frame delta is no move at all.
_EPS = 1e-6
# How far past a shot's bounds an unbounded envelope (the first shot's lead-in,
# the last one's tail) is read for content -- the DCC cut's own cap.
_CONTENT_REACH = 1.0e7
# Frames within which a key is ON a frame: the DCC writers' own tolerance.
_KEY_TOL = 1.0e-3

_log = logging.getLogger(__name__)

__all__ = ["ShotSequencer"]


class _ShotSequencerHooks(object):
    """The scene hooks :class:`ShotSequencer` orchestrates, as empty-scene defaults.

    A DCC adapter overrides the ones its scene supports; each default is what
    that hook reports for a scene with no keys, no audio and nothing to hold.
    """

    #: The :class:`ShotStore` class a sequencer builds when handed no store.
    STORE_CLASS = ShotStore

    def _scene_available(self) -> bool:
        """Whether a scene can be read and written (``False``: bounds only)."""
        return False

    def _content_batch(self):
        """Context manager grouping a multi-step scene edit (audio re-sync etc.)."""
        return contextlib.nullcontext()

    def _apply_plan(self, plan, retime_gaps: bool = False) -> None:
        """Commit *plan*: move each shot's content, then its bounds.

        The single chokepoint for every whole-shot mutation (respace, reorder,
        ripple, slide).  *retime_gaps* asks for each changed gap's content to
        be retimed into its new width (:meth:`respace` asks); an adapter
        without gap retiming may ignore it.  The default commits the bounds
        only (``ShotApply.apply`` with no writers).
        """
        ShotApply.apply(plan, self.store)

    @staticmethod
    def _find_keyed_transforms(start: float, end: float, **kwargs) -> List[str]:
        """Objects MOVING inside ``[start, end]`` (``define_shot``'s default
        members).  ``evaluated=True`` also reads each curve's values at the
        window's ends, so one key of a curve moving through it counts."""
        return []

    def _member_names(self, shot, names: List[str]) -> Tuple[List[str], List[str]]:
        """``(listed, unlisted)``: *shot*'s own spelling of the *names* it
        lists, and the *names* it does not list.  A DCC whose members can be
        spelled more than one way (a short name and its long path) matches by
        the object they name; the default compares the names as given."""
        listed = [n for n in names if n in shot.objects]
        return listed, [n for n in names if n not in shot.objects]

    def collect_object_segments(self, shot_id: int, **kwargs) -> List[Dict[str, Any]]:
        """Per-object animation segments inside a shot (none without a scene)."""
        return []

    def _collect_audio_sequences(
        self, start: float, end: float
    ) -> List[Dict[str, Any]]:
        """Audio sequences overlapping ``[start, end]``."""
        return []

    def _key_extent(self, shot, probe_outside: bool, reach: Optional[float] = None):
        """``(inner_start, inner_end, outer_start, outer_end, on_bound)`` of *shot*'s keys."""
        return None, None, None, None, []

    def _cut_passed_bound_keys(self, on_bound, old_start, old_end, new_start, new_end):
        """Cut the redundant on-bound keys a bound moving inward passes."""
        return 0

    def _reconcile_boundary_keys(
        self, bounds: Optional[Dict[int, tuple]] = None, follow: bool = True
    ) -> Tuple[int, int]:
        """Re-home the system's boundary samples; ``(moved, removed)`` counts."""
        return 0, 0

    def _gap_hold_seams(self) -> Dict[str, list]:
        """The keys that should hold each inter-shot gap."""
        return {}

    def _release_gap_holds(self, seams: Dict[str, list]) -> int:
        """Release system-written holds that are no longer seams."""
        return 0

    def _apply_gap_holds(self, seams: Dict[str, list]) -> int:
        """Hold the gaps at *seams*."""
        return 0

    def _pin_bound_keys(self, pins: Dict[str, List[Tuple[float, int, str]]]) -> int:
        """Key each curve at the bound frames *pins* names, changing nothing
        it plays: ``{curve: [(frame, shot_id, edge), ...]}``.

        A key goes in only where the curve has shape to keep (inside its key
        range, not on a key, not inside a hold), and every key inserted is
        claimed in :attr:`ledger` for its ``(shot_id, edge)`` bound -- a pin is
        the system's sample, never content.  Returns the number of keys
        inserted (or otherwise moved onto a bound).
        """
        return 0

    def _pin_bound_poses(
        self,
        objects: List[str],
        window: Tuple[float, float],
        pins: List[Tuple[float, int, str]],
        attr: Optional[str] = None,
        crossing: bool = False,
    ) -> int:
        """Key the pose each of *objects*' (*attr*'s) curves plays on every
        bound frame *pins* names (``[(frame, shot_id, edge), ...]``), on the
        curves with a key inside *window* (closed, tolerance already in it)
        -- content about to LEAVE, so the
        shots on those bounds keep what it gave them.  *crossing*: the curves
        whose segment runs across *window* keyless too -- the window is about
        to close (a delete), which re-times that segment.  Unlike
        :meth:`_pin_bound_keys` a held or extrapolated pose is keyed too: the
        key making the hold is the one leaving.  Every key put in is claimed
        for its bound, ``stays``.  Returns the number of keys put in.
        """
        return 0

    def _freeze_left_behind(
        self,
        objects: List[str],
        window: Tuple[float, float],
        attr: Optional[str] = None,
    ) -> int:
        """Hold, on each of *objects*' (*attr*'s) curves keyed inside
        *window* (a run about to LEAVE), the derived slope of the keys it
        leaves behind: the last key before it keeps its IN half, the first
        after it its OUT half, each at the angle it plays now.  A derived
        slope is computed from both neighbours, so with the run gone those
        keys re-derive and the content they close or open plays differently
        (:meth:`move_sequences_to_shot`; a gap hold's stepped key's IN half
        excepted, BACKLOG 2026-10-11, the held tail).  Returns the halves
        held.  Pure default: nothing to hold."""
        return 0

    def _poses_differ(
        self,
        pairs: List[Tuple[float, float]],
        objects: Optional[List[Tuple[str, Optional[str]]]] = None,
    ) -> List[bool]:
        """Per ``(frame_a, frame_b)`` of *pairs*: whether the content curves
        -- or only *objects*' (``[(obj, attr or None), ...]``) -- play a
        different value at the two frames, keyed or held (the value a curve
        plays where no key is: a hold, an interpolation, its extrapolation).
        The pose test behind the gap-0 closing rule
        (:meth:`ShotSequencer._close_flush_seams`), read before anything
        moves.  Pure default: every pose agrees."""
        return [False] * len(pairs)

    def _gaps_keyed(self, windows: List[Tuple[float, float]]) -> List[bool]:
        """Per ``(start, end)`` window -- a gap's interior -- whether the
        animator keyed anything inside it now (the system's bound samples
        are its bookkeeping, not content).  What :meth:`respace`'s closing
        rule asks before it closes a gap flush (:meth:`_seams_hold_keys`).
        Pure default: nothing is keyed."""
        return [False] * len(windows)

    def _retime_closing_gaps(self, retimes: List[GapRetime]) -> int:
        """Retime each gap of *retimes* -- one a placement narrows to the
        frame the closing rule kept -- BEFORE anything moves: its keys
        compress toward its left edge, inside the gap they are in, their
        claims with them (the respace's shrink stage, the DCC's own gap
        retime).  Returns the curves retimed.  Pure default: none."""
        return 0

    def _curve_exists(self, curve: str) -> bool:
        """Whether the scene still has the curve a ledger key names.  Pure
        default: ``True`` -- without a scene nothing is known gone, so nothing
        is forgotten."""
        return True

    def _follow_unresolved_claims(self, curves) -> bool:
        """Re-key the claims among *curves* whose owner was renamed, before a
        pass forgets what it cannot resolve (blendertk's keys embed the
        owner's name).  Pure default: nothing to follow."""
        return False

    def _reconcile_boundaries(self, plan) -> Callable[[], None]:
        """Reconcile shared samples around a plan write; returns the finisher.

        The chokepoint every whole-shot move passes (:meth:`_apply_plan` and
        :meth:`_move_shot_content`), so it is where a split seam is made whole
        whatever op opened it: a keyed shared sample is copied onto the
        following shot's new start, and a KEYLESS one is first pinned with
        the preceding shot's own pose (:meth:`_opened_seams`,
        :meth:`ShotPlanner.keyless_seam`)."""
        return lambda: None

    def _move_content_keys(
        self,
        objects: List[str],
        env_lo: float,
        env_hi: float,
        delta: float,
        lo_open: bool = False,
        hi_closed: bool = False,
    ) -> None:
        """Move every key of *objects* inside the envelope by *delta*."""

    def _shift_audio(self, old_start: float, old_end: float, delta: float) -> None:
        """Shift the audio inside ``[old_start, old_end]`` by *delta* -- and the
        claims :attr:`ledger` holds on what moved."""

    def _move_audio_sequence(self, seq: Dict[str, Any], delta: float) -> None:
        """Move one audio sequence (``seq["kind"] == "audio"``) by *delta*."""

    def move_attribute_keys(
        self,
        obj: str,
        attr: Optional[str],
        delta: float,
        times: Optional[List[float]] = None,
        window: Optional[tuple] = None,
    ) -> int:
        """Move *obj*'s keys (one attribute's, or all) by *delta*; keys moved."""
        return 0

    def copy_attribute_keys(
        self,
        obj: str,
        attr: Optional[str],
        delta: float,
        times: Optional[List[float]] = None,
        window: Optional[tuple] = None,
    ) -> int:
        """Copy *obj*'s keys (one attribute's, or all) *delta* frames on, the
        originals left where they are; keys copied."""
        return 0

    def _point_clip_curves(
        self,
        obj: str,
        t: float,
        attr_name: Optional[str] = None,
        shot=None,
        eps: float = 1e-3,
    ) -> list:
        """The curves a point clip of *obj* at *t* stands for: *attr_name*'s,
        or the content curves whose key at *t* *shot* shows as a mark
        (``None``: the animator's keys at *t*).  See :meth:`move_stepped_keys`."""
        return []

    def _move_point_key(
        self, obj: str, curve, old_time: float, new_time: float, eps: float = 1e-3
    ) -> None:
        """Move *curve*'s key at *old_time* to *new_time*, its claims with it."""

    def scale_shot_keys(
        self, old_start: float, old_end: float, new_start: float, new_end: float
    ) -> None:
        """Retime every content key inside ``[old_start, old_end]`` into the new span."""

    def scale_object_keys(
        self,
        obj: str,
        old_start: float,
        old_end: float,
        new_start: float,
        new_end: float,
    ) -> None:
        """Retime *obj*'s keys inside ``[old_start, old_end]`` into the new span."""

    def _cut_shot_content(self, shot_id: int) -> int:
        """Delete every key inside the shot's owned window; curves cut."""
        return 0

    @staticmethod
    def _keyed_transform_times() -> dict:
        """``{object: key times}`` for the scene's keyed content."""
        return {}

    def _trailing_content_extent(self, shot) -> float:
        """The frame *shot*'s own content trails out to (at least its end)."""
        return shot.end

    def _drop_claimed_seam_copy(self, seq, target: float, seam: float, **kwargs):
        """Drop a source shot's copy of the seam sample a moved block claims."""

    def _object_key_probe(self, obj: str) -> Callable[[float], bool]:
        """``frame -> bool``: whether *obj* holds a key at that frame."""
        return lambda frame: False

    def _animator_times_in(
        self, obj: str, attr: Optional[str], window: Tuple[float, float]
    ) -> List[float]:
        """The animator's (not the system's) key times of *obj* inside *window*."""
        return []

    def detect_shots(self, **kwargs) -> List[Dict[str, Any]]:
        """Shot-region candidates detected in the scene (none without one)."""
        return []


class ShotSequencer(_ShotSequencerHooks):
    """Manages a :class:`ShotStore` and provides ripple editing on top of it.

    The DCC-free half of mayatk's / blendertk's ``ShotSequencer``: every
    timeline operation is decided here and reaches the scene only through
    the hooks on :class:`_ShotSequencerHooks` (see the module docstring).
    Used directly it is a bounds-only sequencer.

    Parameters:
        shots: Initial shot list (creates an internal store of
            :attr:`STORE_CLASS`).
        store: Existing store to wrap.  Takes precedence over *shots*.
    """

    #: ``{shot_id: (start, end)}`` an enclosing edit is about to write
    #: (:meth:`pending_bounds`).  Replaced, never mutated in place.
    _pending_bounds_map: Dict[int, Tuple[Optional[float], Optional[float]]] = {}

    def __init__(
        self,
        shots: Optional[List[ShotBlock]] = None,
        store: Optional[ShotStore] = None,
    ):
        if store is not None:
            self.store = store
        else:
            self.store = self.STORE_CLASS(shots)

    @contextlib.contextmanager
    def pending_bounds(
        self, bounds: Dict[int, Tuple[Optional[float], Optional[float]]]
    ):
        """Declare the bounds an edit is about to write, for every whole-shot
        move made inside the block.

        A split seam is pinned where the move LEAVES a gap (the one rule,
        maintainer 2026-10-10: an op on one shot never changes what another
        plays inside its own span -- a keyless seam pulled apart re-timed the
        shot before it from its first frame).  An edit that moves its
        neighbours BEFORE it writes its own new bound opens a gap that its
        next write closes, and judged move by move that transient gap is a
        split: Move to Shot growing its destination around what it is about
        to land, a slide rippling its neighbour out of the way before the
        pivot follows.  Pinned there, the seam is a key the landing has to
        fight.  Inside this block :meth:`_opened_seams` judges each pair on
        the bounds declared here (:meth:`ShotPlanner.boundary_splits`
        ``pending=``).

        Nested blocks merge, the OUTER declaration winning per side: the
        outermost edit knows where its whole sequence ends (Move to Shot's
        head room is a slide whose own declaration moves the destination's
        start, which the landing then fills).

        Parameters:
            bounds: ``{shot_id: (start, end)}``; ``None`` on a side leaves
                that side to the moves themselves.
        """
        outer = self._pending_bounds_map
        merged = {k: tuple(v) for k, v in (bounds or {}).items()}
        for sid, (o_start, o_end) in outer.items():
            i_start, i_end = merged.get(sid, (None, None))
            merged[sid] = (
                o_start if o_start is not None else i_start,
                o_end if o_end is not None else i_end,
            )
        self._pending_bounds_map = merged
        try:
            yield
        finally:
            self._pending_bounds_map = outer

    def _opened_seams(self, plan) -> List[Tuple[int, int, float, float]]:
        """The contiguous seams *plan* leaves OPEN once the enclosing edit is
        done (:meth:`pending_bounds`): where a DCC's
        :meth:`_reconcile_boundaries` pins a keyless seam with the preceding
        shot's own pose (:meth:`ShotPlanner.keyless_seam`).  The KEYED
        shared sample keeps :meth:`ShotPlanner.boundary_splits` without
        *pending*: its copy is how a landing on the seam leaves the
        neighbour its opening pose (``ShotStore.enclosing_bounds``)."""
        return ShotPlanner.boundary_splits(
            self.store, plan, pending=self._pending_bounds_map
        )

    # ---- delegated properties -------------------------------------------

    @property
    def shots(self) -> List[ShotBlock]:
        return self.store.shots

    @shots.setter
    def shots(self, value: List[ShotBlock]):
        self.store.shots = value

    @property
    def hidden_objects(self) -> set:
        return self.store.hidden_objects

    @hidden_objects.setter
    def hidden_objects(self, value: set):
        self.store.hidden_objects = value

    @property
    def markers(self) -> List[Dict[str, Any]]:
        return self.store.markers

    @markers.setter
    def markers(self, value: List[Dict[str, Any]]):
        self.store.markers = value

    def is_object_hidden(self, obj_name: str) -> bool:
        return self.store.is_object_hidden(obj_name)

    def set_object_hidden(self, obj_name: str, hidden: bool = True) -> None:
        self.store.set_object_hidden(obj_name, hidden)

    def sorted_shots(self) -> List[ShotBlock]:
        return self.store.sorted_shots()

    @staticmethod
    def describe_key_selection(
        rows: Iterable[Tuple[Any, Optional[str], Optional[Iterable[float]]]],
    ) -> str:
        """The footer line for a key selection: how many keys, on which
        frames, on how many channels of how many objects.

        Parameters:
            rows: ``[(obj, attr, times)]`` -- one per clip holding selected
                keys; *attr* ``None`` (or ``""``) for an object-level clip.

        Returns:
            ``""`` when no key is selected, else e.g.
            ``"12 keys · frames 100–180 (80f, 3 frames keyed) · 4 channels
            on 2 objects"``
            or ``"1 key · frame 120 · translateX on Box01"``.
        """

        def frame(t: float) -> str:
            return f"{t:.0f}" if abs(t - round(t)) < 1e-6 else f"{t:.2f}"

        times, channels, objects, n = set(), [], [], 0
        for obj, attr, keys in rows:
            keys = list(keys or ())
            if not keys:
                continue
            n += len(keys)
            times.update(keys)
            if obj and obj not in objects:
                objects.append(obj)
            if attr and (obj, attr) not in channels:
                channels.append((obj, attr))
        if not n:
            return ""
        parts = [f"{n} key{'s' if n != 1 else ''}"]
        lo, hi = min(times), max(times)
        if len(times) == 1:
            parts.append(f"frame {frame(lo)}")
        else:
            # How many frames the keys sit on -- said only when they share some.
            keyed = f", {len(times)} frames keyed" if len(times) != n else ""
            parts.append(f"frames {frame(lo)}–{frame(hi)} ({frame(hi - lo)}f{keyed})")
        if not objects:
            return " · ".join(parts)
        where = (
            str(objects[0]).split("|")[-1]
            if len(objects) == 1
            else f"{len(objects)} objects"
        )
        if len(channels) == 1:
            parts.append(f"{channels[0][1]} on {where}")
        elif channels:
            parts.append(f"{len(channels)} channels on {where}")
        else:
            parts.append(f"on {where}")
        return " · ".join(parts)

    @staticmethod
    def paste_pairs(
        rows: Iterable[Tuple[Any, str, Any]],
        destinations: Iterable[Tuple[Any, str]],
        leaf: Optional[Callable[[Any], Any]] = None,
    ) -> List[Tuple[Any, str, Any]]:
        """Which copied row each destination row of a key paste receives --
        the Shot Sequencer's key clipboard, in both DCCs.

        A lone copied row goes on every destination, as a Graph Editor pastes
        one curve onto each selected one.  Otherwise a row lands on the
        destination of its own attribute: on the same object when that is
        among them, else the n-th destination object of an attribute takes
        the n-th copied object of it (cycling), so copying two objects and
        pasting onto two others pairs them in order.  A destination whose
        attribute was not copied gets nothing.

        Parameters:
            rows: ``[(obj, attr, keys)]`` -- the copied rows; *keys* is
                carried through untouched.
            destinations: ``[(obj, attr)]`` -- the rows pasted onto.
            leaf: Maps an object name to the name two rows' objects are
                compared by (mayatk: the DAG leaf, as a copy and a paste may
                spell one node long and short); identity by default.

        Returns:
            ``[(dest_obj, dest_attr, keys)]`` in destination order.
        """
        rows = list(rows)
        destinations = list(destinations)
        if len(rows) == 1:
            return [(obj, attr, rows[0][2]) for obj, attr in destinations]
        same_name = leaf or (lambda name: name)
        by_attr: Dict[str, List[Tuple[Any, Any]]] = {}
        for obj, attr, keys in rows:
            by_attr.setdefault(attr, []).append((obj, keys))
        seen: Dict[str, int] = {}
        pairs = []
        for obj, attr in destinations:
            sources = by_attr.get(attr)
            if not sources:
                continue
            keys = next((k for o, k in sources if same_name(o) == same_name(obj)), None)
            if keys is None:
                i = seen.get(attr, 0)
                keys = sources[i % len(sources)][1]
                seen[attr] = i + 1
            pairs.append((obj, attr, keys))
        return pairs

    def shot_by_id(self, shot_id: int) -> Optional[ShotBlock]:
        return self.store.shot_by_id(shot_id)

    def shot_by_name(self, name: str) -> Optional[ShotBlock]:
        return self.store.shot_by_name(name)

    @property
    def ledger(self):
        """The store's :class:`~pythontk.ShotEditLedger` (system-write claims)."""
        return self.store.edit_ledger

    # ---- definition -------------------------------------------------------

    def define_shot(
        self,
        name: str,
        start: float,
        end: float,
        objects: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        locked: bool = False,
        description: str = "",
    ) -> ShotBlock:
        """Define a shot manually from a name and range.

        Parameters:
            name: Human-readable label.
            start: First frame.
            end: Last frame.
            objects: Object names.  If ``None``, automatically discovers all
                objects with keyframes in [start, end]
                (:meth:`_find_keyed_transforms`).
            metadata: Arbitrary key/value pairs to persist with the shot.
            locked: Mark this shot as user-finalized.
            description: Human-readable description of the shot.

        Returns:
            The newly created :class:`ShotBlock`.
        """
        if objects is None:
            objects = self._find_keyed_transforms(start, end)
        return self.store.define_shot(
            name=name,
            start=start,
            end=end,
            objects=objects,
            metadata=metadata,
            locked=locked,
            description=description,
        )

    def _refuse_name(self, name: str) -> None:
        """Raise ``ValueError`` when the store would refuse *name* for a new
        shot (``ShotStore.name_error``).

        For an operation that edits the scene BEFORE it defines the shot (a
        ripple, a trim): ``define_shot`` refuses only once it is reached, which
        is after the edit it cannot undo.
        """
        error = self.store.name_error(name)
        if error:
            raise ValueError(error)

    def detect_next_shot(
        self,
        gap_threshold: float = 5.0,
        ignore: Optional[str] = None,
        motion_rate: float = 1e-3,
    ) -> Optional[Dict[str, Any]]:
        """Detect the first animation cluster after all existing shots.

        Useful for incremental shot building — discovers the next
        unregistered animation region without re-scanning the entire
        timeline.

        Parameters:
            gap_threshold: Minimum gap (frames) between clusters.
            ignore: Attribute pattern(s) to exclude.
            motion_rate: Per-frame rate threshold (see :meth:`detect_shots`).

        Returns:
            A candidate shot dict (``name``, ``start``, ``end``,
            ``objects``) or ``None`` if no uncovered animation remains.
        """
        candidates = self.detect_shots(
            gap_threshold=gap_threshold,
            ignore=ignore,
            motion_rate=motion_rate,
        )
        if not candidates:
            return None

        existing = self.store.sorted_shots()
        if not existing:
            return candidates[0]

        # Find the first candidate whose start is beyond all existing shots
        last_end = max(s.end for s in existing)
        for cand in candidates:
            if cand["start"] >= last_end:
                return cand

        # Fall back: find candidates that don't overlap any existing shot
        for cand in candidates:
            if not any(
                cand["start"] < shot.end and cand["end"] > shot.start
                for shot in existing
            ):
                return cand

        return None

    # ---- sequences --------------------------------------------------------

    @staticmethod
    def _point_segment(obj: str, t: float) -> Dict[str, Any]:
        """The zero-length, stepped segment the widget draws as a key marker."""
        return {
            "obj": obj,
            "curves": [],
            "keyframes": [t],
            "start": t,
            "end": t,
            "duration": 0.0,
            "segment_range": (t, t),
            "is_stepped": True,
            "stepped_key_time": t,
            # A mark, not motion: drawn, but never content for a trim, a fit
            # or a membership backfill (see collect_shot_sequences).
            "marker": True,
        }

    def collect_shot_sequences(
        self,
        shot_id: int,
        include_audio: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return all sequences (anim + audio) inside a shot's range.

        Each item is a dict with ``"kind"`` (``"anim"`` or ``"audio"``),
        ``"obj"`` (object name or audio track id), ``"start"``, ``"end"``.
        Anim segments come from :meth:`collect_object_segments`; audio
        comes from :meth:`_collect_audio_sequences`.
        """
        anim = self.collect_object_segments(shot_id)
        # A value-less key drawn as a marker is not a sequence: counting it
        # made a flat member's key on the bound untrimmable again.
        result: List[Dict[str, Any]] = [
            {
                "kind": "anim",
                "obj": seg["obj"],
                "start": seg["start"],
                "end": seg["end"],
            }
            for seg in anim
            if not seg.get("marker")
        ]
        if include_audio:
            shot = self.shot_by_id(shot_id)
            if shot is not None:
                result.extend(self._collect_audio_sequences(shot.start, shot.end))
        return result

    def _move_sequence(self, seq: Dict[str, Any], new_start: float) -> None:
        """Dispatch a sequence move based on ``seq["kind"]``.

        Anim sequences re-use :meth:`move_object_keys` (or, for a key
        selection, :meth:`move_attribute_keys`); audio sequences go to
        :meth:`_move_audio_sequence`.  Caller is responsible for any wrapping
        :meth:`_content_batch` / ``store.batch_update()`` context.
        """
        delta = new_start - seq["start"]
        if abs(delta) < _EPS:
            return
        if seq["kind"] == "anim":
            if seq.get("attr") or seq.get("times"):
                # A key selection: one attribute, and only the keys named.
                self.move_attribute_keys(
                    seq["obj"],
                    seq.get("attr"),
                    delta,
                    times=seq.get("times"),
                    window=(seq["start"], seq["end"]),
                )
            else:
                self.move_object_keys(seq["obj"], seq["start"], seq["end"], new_start)
        elif seq["kind"] == "audio":
            self._move_audio_sequence(seq, delta)

    def _copy_sequence(self, seq: Dict[str, Any], new_start: float) -> None:
        """Copy an anim sequence so the copy begins at *new_start* -- the twin of
        :meth:`_move_sequence`, through :meth:`copy_attribute_keys` (a whole
        object's run is every attribute's keys in its window)."""
        delta = new_start - seq["start"]
        if abs(delta) < _EPS or seq["kind"] != "anim":
            return
        self.copy_attribute_keys(
            seq["obj"],
            seq.get("attr"),
            delta,
            times=seq.get("times"),
            window=(seq["start"], seq["end"]),
        )

    def _recompute_shot_objects(self, shot_id: int, only=None, keep=()) -> None:
        """Rebuild ``shot.objects`` from animation that actually lives in the shot.

        Scans every anim sequence inside the shot's frame range and keeps
        only the objects that contribute keys.  Locked / pinned objects
        are preserved even when they have no remaining keys.  Audio is
        out of scope — audio tracks are not part of ``shot.objects``.
        Without a scene (:meth:`_scene_available`) membership stands.

        *only* narrows the re-decision to those objects: every other member
        stands as it was.  A move re-decides what it moved and nothing else --
        a member keyed nowhere in the shot (manifest-authored, or pinned by
        hand before its keys existed) is not the move's to drop.  Measured on
        a production assembly: one clip moved out of an 8-member shot left it
        with 2, and the 10-member destination with 5.

        *keep* names objects this shot owns whatever the scan finds -- what a
        caller just placed here.  Membership is derived from MOTION, so a lone
        flat key (a render-effect off-state, an anim bookend) moved into a
        shot was disowned by it the moment it arrived: no track, no clip,
        nothing to show the user the key had landed.
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            return
        if not self._scene_available():
            return
        anim_objs = {seg["obj"] for seg in self.collect_object_segments(shot_id)}
        keep = self.store.pinned_objects | self.store.locked_objects | set(keep)
        current = set(shot.objects)
        if only is None:
            new_objs = current & (anim_objs | keep) | anim_objs
        else:
            subject = set(only)
            new_objs = (current - subject) | (subject & (anim_objs | keep))
        new_objs = sorted(new_objs)
        if new_objs != sorted(shot.objects):
            self.store.update_shot(shot_id, objects=new_objs)

    def _source_shot_id_for(self, seq: Dict[str, Any]) -> Optional[int]:
        """Return the shot_id that currently contains *seq* (by frame range)."""
        for sh in self.store.shots:
            if sh.start - _EPS <= seq["start"] and seq["end"] <= sh.end + _EPS:
                return sh.shot_id
        return None

    def sequence_separation(self) -> float:
        """Room to leave between a moved sequence and what it lands after.

        Strictly MORE than the inter-shot gap, and never less than a frame.
        Both halves matter for how the result reads: at zero the arriving clip
        butts against the existing one and the two draw as a single merged
        run, which makes a non-destructive move look like it overwrote
        something; at exactly the shot gap the seam inside a shot is
        indistinguishable from a seam BETWEEN shots.  One frame past the gap
        is the tightest spacing that is unambiguously neither.
        """
        return self.store.snap(max(float(self.store.gap) + 1.0, 1.0))

    def move_sequences_to_shot(
        self,
        sequences: List[Dict[str, Any]],
        dest_shot_id: int,
    ) -> None:
        """Move *sequences* (anim and/or audio) into *dest_shot_id*.

        Sequences are grouped by source shot so each subgroup moves as a unit,
        preserving internal offsets — a multi-object selection keeps its
        shape.  Content keeps its ORDER: a group that sat before the
        destination is the head of what is already there, one that sat after
        it is the tail.

            - From before, onto objects the destination already animates: the
              shot's time is inserted at its start (:meth:`add_shot_space`,
              leading) -- every object's content and every downstream shot
              move later by the room the block needs -- and the group lands
              at the destination start, in front.  The block carries its own
              overhang (:meth:`_absorb_gap_overhang`), and when it sat within
              a shot-gap of the destination the room is its own displacement,
              so the block and the content it joins shift as one run.
              Measured on a production assembly: the on-ramp of a highlight
              pulse sat one frame inside the previous shot, its partner in
              the gap, the rest of the train in the destination; appending
              sent the key to the far end of the shot, and moving it alone
              flipped the ramp.
            - From after, onto objects the destination already animates: the
              group lands AFTER the last of that content, separated by
              :meth:`sequence_separation`.
            - Onto objects with nothing there: anchored to the destination
              start, nothing pushed.

        Neither direction can reach outside the destination: the head case
        makes its room by insertion rather than by placing content ahead of
        the shot's start (which put it on top of the previous shot), and the
        tail case grows the shot.

        Nothing is overwritten and nothing is trimmed: the destination grows
        to enclose whatever landed (:meth:`extend_shot_to_fit`), rippling its
        neighbours so their spacing is preserved.

        After the move, ``shot.objects`` is recomputed for the destination and
        every source shot that lost content.

        Parameters:
            sequences: dicts with ``"kind"``, ``"obj"``, ``"start"``,
                ``"end"`` (as produced by :meth:`collect_shot_sequences`).
            dest_shot_id: Target shot's id.
        """
        self._send_sequences_to_shot(sequences, dest_shot_id, copy=False)

    def copy_sequences_to_shot(
        self,
        sequences: List[Dict[str, Any]],
        dest_shot_id: int,
    ) -> None:
        """Copy *sequences* into *dest_shot_id*, the originals left in place.

        :meth:`move_sequences_to_shot`'s placement exactly -- in front of the
        destination's content for what came from before it, after it for what
        came from after, the room opened and the shot grown to hold it -- with
        the keys copied (:meth:`copy_attribute_keys`, the animator's own:
        tangents, no claims) instead of moved.  The sources keep their content
        and their members; the destination takes the copied objects as
        members.  Audio sequences are skipped: a clip is one placement of its
        track, not keys to duplicate.

        Parameters:
            sequences: as :meth:`move_sequences_to_shot` takes them.
            dest_shot_id: Target shot's id.
        """
        self._send_sequences_to_shot(sequences, dest_shot_id, copy=True)

    def _send_sequences_to_shot(
        self,
        sequences: List[Dict[str, Any]],
        dest_shot_id: int,
        copy: bool,
    ) -> None:
        """Move or *copy* *sequences* into *dest_shot_id* -- the one placement
        behind :meth:`move_sequences_to_shot` and :meth:`copy_sequences_to_shot`."""
        dest = self.shot_by_id(dest_shot_id)
        if dest is None:
            raise ValueError(f"No shot with id {dest_shot_id}")
        if copy:
            sequences = [s for s in sequences if s["kind"] == "anim"]
        if not sequences:
            return
        for s in sequences:
            self._animator_run(s)

        def content_by_obj() -> Dict[str, List[Dict[str, Any]]]:
            """The destination's sequences keyed by object, as they sit NOW."""
            by_obj: Dict[str, List[Dict[str, Any]]] = {}
            for s in self.collect_shot_sequences(dest_shot_id):
                by_obj.setdefault(s["obj"], []).append(s)
            return by_obj

        dest_seqs_by_obj = content_by_obj()

        groups: Dict[Optional[int], List[Dict[str, Any]]] = {}
        for seq in sequences:
            sid = self._source_shot_id_for(seq)
            if sid == dest_shot_id:
                continue  # already in destination — skip
            groups.setdefault(sid, []).append(seq)

        if not groups:
            return

        affected_shots: set = {dest_shot_id}

        seam_samples: Dict[int, List[str]] = {}  # id(seq) -> ["start", "end"]
        if not copy:
            # What leaves a source gave the shots beside it their poses on the
            # facing bounds; pinned before anything moves (the destination is
            # edited by the move itself).  A FLUSH bound's sample is the
            # neighbour's pose too, and is copied, not carried: its facing
            # slope is held here, and a copy is put back once the run has
            # moved (:meth:`_seam_samples_of`).  The keys the run leaves
            # behind keep the slopes they play into their own content first:
            # the source plays the same up to its last stayed key (BACKLOG
            # 2026-10-11, "A01_2" / "C02").
            for sid, grp in groups.items():
                for seq in grp:
                    if sid is not None and seq["kind"] == "anim":
                        self._freeze_left_behind(
                            [seq["obj"]],
                            (seq["start"] - _KEY_TOL, seq["end"] + _KEY_TOL),
                            attr=seq.get("attr"),
                        )
                        self._hold_neighbour_poses(
                            sid,
                            [seq["obj"]],
                            (seq["start"] - _KEY_TOL, seq["end"] + _KEY_TOL),
                            attr=seq.get("attr"),
                            skip=(dest_shot_id,),
                            flush=True,
                            crossing=False,
                        )
                        sides = self._seam_samples_of(seq, sid, dest_shot_id)
                        if sides:
                            seam_samples[id(seq)] = sides

        # Pre-register moved anim objects on dest so that the post-move
        # _recompute_shot_objects pass actually scans them.  Without this,
        # collect_object_segments would only see dest's existing objects
        # and the newly-moved objects would never make it into
        # dest.objects.
        dest_anim_additions = {
            seq["obj"]
            for grp in groups.values()
            for seq in grp
            if seq["kind"] == "anim"
        }
        if dest_anim_additions:
            merged = sorted(set(dest.objects) | dest_anim_additions)
            if merged != sorted(dest.objects):
                self.store.update_shot(dest_shot_id, objects=merged)

        separation = self.sequence_separation()

        # ---- 1. the head block: insert its room at the start ---------------
        # Groups that lie entirely before the destination land in front, as
        # ONE block in their own order.  Where their objects already have
        # content there, the shot's time is inserted at its start so the
        # arrivals meet empty timeline and nothing is reordered; the pad is a
        # real move of the destination's content and every downstream shot,
        # so what it carried is re-read before any landing spot is resolved.
        head_groups = {
            sid: grp
            for sid, grp in groups.items()
            if max(s["end"] for s in grp) < dest.start - _EPS
        }
        block_min = 0.0
        # What lands ON the destination's start owns the seam there
        # (maintainer 2026-09-23): the head block, and a group from after it
        # whose objects have nothing there (anchored to the start, step 2).
        # Where the shot before closes flush on a different pose, the
        # destination opens a frame later and keeps it (the closing rule,
        # 2026-10-11): the frame the shot before ends on stays its own.  An
        # anchored group took that frame from it -- SOL gap 0, B11's first
        # run sent to B10_2 played B10_1's last frame 98.35 off.
        head_objs = {s["obj"] for grp in head_groups.values() for s in grp}
        floor = max(
            [
                self._head_landing_floor(dest, grp, min(s["start"] for s in grp))
                for sid, grp in groups.items()
                if sid not in head_groups
                and not any(
                    dest_seqs_by_obj.get(s["obj"]) or s["obj"] in head_objs for s in grp
                )
            ]
            or [0.0]
        )
        room = 0.0
        if head_groups:
            head_seqs = [s for grp in head_groups.values() for s in grp]
            for s in head_seqs:
                self._absorb_gap_overhang(s, dest.start)
            block_min = min(s["start"] for s in head_seqs)
            block_max = max(s["end"] for s in head_seqs)
            floor = max(floor, self._head_landing_floor(dest, head_seqs, block_min))
            if any(dest_seqs_by_obj.get(s["obj"]) for s in head_seqs):
                # The block's own displacement when it already sat within a
                # shot-gap of the destination -- it and the content it joins
                # then shift as ONE run, spacing intact -- and otherwise its
                # span plus the standard clip separation.  Up, never nearest:
                # a room short of a fractional block lands the block's last
                # key after the first key it was put in front of.
                room = self.store.snap(
                    block_max - block_min + min(dest.start - block_max, separation),
                    "up",
                )
                # The room is the block's to fill, so the destination opens
                # where it opens now: its seam with the shot before is not
                # split, and is not pinned (:meth:`pending_bounds`).
                with self.pending_bounds({dest_shot_id: (dest.start, None)}):
                    self.add_shot_space(dest_shot_id, room, edge="leading")
        if floor > _EPS:
            self.set_shot_start(dest_shot_id, dest.start + floor, ripple=True)
        if room + floor > _EPS:
            # Everything from the destination's start onward travelled with
            # the pad -- its own content, the gap behind it, the downstream
            # shots -- and so did the tail groups' keys.
            for sid, grp in groups.items():
                if sid in head_groups:
                    continue
                if min(s["start"] for s in grp) >= dest.start - floor - _EPS:
                    for s in grp:
                        s["start"] += room + floor
                        s["end"] += room + floor
            dest_seqs_by_obj = content_by_obj()

        # ---- 2. resolve every landing spot BEFORE anything moves ----------
        # Purely arithmetic, against the destination's CURRENT content: what
        # is already in the destination never moves, so these targets stay
        # valid across the room-making below.
        placements: List[tuple] = []  # (seq, source_shot_id, target_start)
        needed_end = dest.end
        # Earliest group first, so a multi-source move stacks in the order the
        # content sat on the timeline rather than in dict order.
        for source_id, group in sorted(
            groups.items(), key=lambda kv: min(s["start"] for s in kv[1])
        ):
            base = min(s["start"] for s in group)

            existing: List[Dict[str, Any]] = []
            for seq in group:
                existing.extend(dest_seqs_by_obj.get(seq["obj"], []))

            if source_id in head_groups:
                # In front, keeping the block's own spacing.
                anchor = dest.start + (base - block_min)
            elif existing:
                anchor = self.store.snap(max(e["end"] for e in existing) + separation)
            else:
                # Nothing of this group's to clear, so nothing is pushed.
                anchor = dest.start

            for seq in group:
                target = anchor + (seq["start"] - base)
                span = seq["end"] - seq["start"]
                placements.append((seq, source_id, target))
                needed_end = max(needed_end, target + span)
                # Stack later groups behind this one instead of stomping it.
                dest_seqs_by_obj.setdefault(seq["obj"], []).append(
                    {
                        "kind": seq["kind"],
                        "obj": seq["obj"],
                        "start": target,
                        "end": target + span,
                    }
                )
            if source_id is not None and not copy:
                # A source loses content only to a move.
                affected_shots.add(source_id)

        # ---- 3. open the room, THEN land in it ----------------------------
        # Growing the destination afterwards cannot work: content that lands
        # past its end sits inside the NEXT shot's span, where the extend
        # probe disowns it (it cannot tell a neighbour's keys from its own)
        # and the following ripple would drag it straight back out again.
        # Making room first means the arriving content only ever lands on
        # empty timeline -- which is also what makes the operation read as
        # non-destructive rather than as an overwrite.
        with self._content_batch(), self.store.batch_update():
            # Up, never nearest: the destination must enclose what lands in it
            # (fractional content would otherwise end past its end).
            room = self.store.snap(needed_end, "up") - dest.end
            if room > _EPS:
                old_end = dest.end
                # Source shots at or after the destination's end travel with
                # the ripple, and so does the content still sitting in them;
                # their recorded positions move by the same delta.
                travelled = {
                    sid
                    for sid in groups
                    if sid is not None
                    and (self.shot_by_id(sid) or dest).start >= old_end - _EPS
                }
                # The ripple keeps the next shot where it was relative to the
                # destination's end -- flush stays flush -- which lands the
                # last run's closing pose on that shot's opening one: a
                # frame apart where they differ (the closing rule).
                shift = room + self._landing_floor(
                    dest, self.store.snap(needed_end, "up"), room, placements, groups
                )
                dest.end = self.store.snap(needed_end, "up")
                self.ripple_downstream(dest_shot_id, old_end, shift)
                for seq, source_id, _target in placements:
                    if source_id in travelled:
                        seq["start"] += shift
                        seq["end"] += shift

            # The moved block owns the seam it lands on (maintainer decision,
            # 2026-09-23). Room inserted at the head of a destination the
            # source touched splits their shared sample: the source keeps a
            # copy of its closing pose ON the destination's start, exactly
            # where the block's first key lands, and pushed aside by the
            # landing it survived as a stray key a hair into the destination's
            # own content (measured: 290.1 beside the true opening pose at 290).
            # The block's key takes the frame; the source ends on it.
            for seq, source_id, target in placements:
                if source_id in head_groups and seq["kind"] == "anim":
                    self._drop_claimed_seam_copy(seq, target, dest.start)
            # A run landing on the destination's last frame -- or a head block
            # becoming the last key before the next shot -- is the next shot's
            # opening pose's new neighbour: pinned first, that pose keeps its
            # value and the slope its own content plays (a landing a frame away
            # re-derived it -- the next shot opened 0.97 off, 2026-10-11).
            self._hold_landing_neighbour(dest, placements, groups, head_groups)
            send = self._copy_sequence if copy else self._move_sequence
            for seq, _source_id, target in placements:
                send(seq, target)
                for side in seam_samples.get(id(seq), ()):
                    # The run's sample on a flush seam stays the neighbour's:
                    # a copy of the landed key goes back to its frame (where
                    # the run sat when it left, a ripple included).
                    frame = seq["start"] if side == "start" else seq["end"]
                    landed = target + (frame - seq["start"])
                    self.copy_attribute_keys(
                        seq["obj"],
                        seq.get("attr"),
                        frame - landed,
                        window=(landed, landed),
                    )

            moved_objs = {
                seq["obj"] for seq, _s, _t in placements if seq["kind"] == "anim"
            }
            for sid in affected_shots:
                # What landed here is this shot's, motion or not.
                self._recompute_shot_objects(
                    sid,
                    only=moved_objs,
                    keep=moved_objs if sid == dest_shot_id else (),
                )

        # A safety net for anything the arithmetic could not predict (audio
        # whose carrier resolved differently, a curve that refused a move):
        # extend-to-fit is implicit, never a separate user action.  Normally a
        # no-op now, because the room was already opened to size.
        self.extend_shot_to_fit(dest_shot_id)
        # The room above was opened by writing the destination's END by hand,
        # and the keys that left changed the source shot's seam: both are
        # what the system's own samples answer to, and extend reconciles
        # only when IT moved a bound.  Measured on a production assembly: one
        # Move to Shot grew "Shot 3.4-5" 748 -> 781 and left 38 of its 39
        # claimed end samples -- stepped -- at 748, inside the shot.
        self.reconcile_system_edits()

    @staticmethod
    def _plan_bounds(plan) -> Dict[int, Tuple[float, float]]:
        """``{shot_id: (start, end)}`` of *plan* -- a :class:`MovePlan`'s moves'
        new bounds, or a bounds dict as it is."""
        if isinstance(plan, dict):
            return plan
        return {sid: (m.new_start, m.new_end) for sid, m in plan.moves.items()}

    def _seam_samples_of(
        self, seq: Dict[str, Any], source_id: int, dest_id: int
    ) -> List[str]:
        """Which ends of *seq* (``"start"``, ``"end"``), about to leave
        *source_id*, hold the sample of a FLUSH seam with a shot other than
        *dest_id*: its first key on the source's start, its last on the
        source's end.

        That sample is the neighbour's pose too -- the closing pose of the
        shot before, by the fencepost, or the opening one of the shot after
        -- so it is copied, not carried (the rule a shot move follows).
        Carried, the curve's first key left with the run and every shot
        before played its new first key's extrapolation: measured
        2026-10-11 on a production assembly, SOLENOID 4.85 off over 1734
        frames.
        """
        shots = self.sorted_shots()
        idx = next((i for i, s in enumerate(shots) if s.shot_id == source_id), None)
        if idx is None:
            return []
        src = shots[idx]
        out: List[str] = []
        if idx > 0:
            prev = shots[idx - 1]
            if (
                prev.shot_id != dest_id
                and abs(prev.end - src.start) < _KEY_TOL
                and abs(seq["start"] - src.start) <= _KEY_TOL
            ):
                out.append("start")
        if idx + 1 < len(shots):
            nxt = shots[idx + 1]
            if (
                nxt.shot_id != dest_id
                and abs(nxt.start - src.end) < _KEY_TOL
                and abs(seq["end"] - src.end) <= _KEY_TOL
                and abs(seq["end"] - seq["start"]) > _KEY_TOL
            ):
                out.append("end")
        return out

    def _head_landing_floor(self, dest, head_seqs, block_min: float) -> float:
        """The frame *dest* opens later by when a Move to Shot head block --
        *head_seqs*, landing from *block_min* on *dest*'s start -- would land
        flush behind a shot closing on a different pose; otherwise 0
        (:meth:`_close_flush_seams`).  The runs read are those opening the
        block, each against the shot before's closing pose."""
        shots = self.sorted_shots()
        idx = next(i for i, s in enumerate(shots) if s.shot_id == dest.shot_id)
        if idx == 0:
            return 0.0
        prev = shots[idx - 1]
        if abs(prev.end - dest.start) > _KEY_TOL:
            return 0.0  # a gap: the block's first pose has its own frame
        first = [
            seq
            for seq in head_seqs
            if seq["kind"] == "anim" and abs(seq["start"] - block_min) <= _KEY_TOL
        ]
        if not first:
            return 0.0

        def layout(floors: Dict[int, float]) -> Dict[int, Tuple[float, float]]:
            shift = floors.get(dest.shot_id, 0.0)
            return {dest.shot_id: (dest.start + shift, dest.end + shift)}

        def seams(plan) -> List[Tuple[int, int]]:
            flush = abs(plan[dest.shot_id][0] - prev.end) < _KEY_TOL
            return [(prev.shot_id, dest.shot_id)] if flush else []

        def poses(_prev_id, _next_id) -> List[tuple]:
            return [
                (prev.end, seq["start"], [(seq["obj"], seq.get("attr"))])
                for seq in first
            ]

        _plan, apart = self._close_flush_seams(layout, seams=seams, poses=poses)
        return apart.get(dest.shot_id, 0.0)

    def _hold_landing_neighbour(self, dest, placements, sources, heads=()) -> int:
        """Pin the pose and facing slope the shot after *dest* opens on
        (:meth:`_pin_bound_poses`) for each anim run of *placements* that
        becomes that pose's new neighbour: one landing on *dest*'s last
        frame, or a head block (its source in *heads*) landing past the last
        key of its curve that stays before the next shot -- that shot played
        the stayed key's hold or extrapolation, and would play the block's.
        Measured 2026-10-11 on a production assembly: a head block landing
        clear of its destination's end became its curve's last key, and
        "D05" played its pose (SOCKET_WRENCH 1.7 off over all 61 frames).
        A next shot that is itself a source of the move (*sources*) is the
        move's to edit.  Returns the number of keys pinned."""
        shots = self.sorted_shots()
        idx = next((i for i, s in enumerate(shots) if s.shot_id == dest.shot_id), None)
        if idx is None or idx + 1 >= len(shots) or shots[idx + 1].shot_id in sources:
            return 0
        nxt = shots[idx + 1]
        pinned = 0
        for seq, sid, target in placements:
            if seq["kind"] != "anim":
                continue
            end = target + (seq["end"] - seq["start"])
            if sid in heads:
                keys = self._animator_times_in(
                    seq["obj"], seq.get("attr"), (-_CONTENT_REACH, nxt.start - _KEY_TOL)
                )
                lo, hi = seq["start"] - _KEY_TOL, seq["end"] + _KEY_TOL
                still = [t for t in keys if not lo <= t <= hi]
                if not still or max(still) > end + _KEY_TOL:
                    continue  # the next shot's left neighbour is not the block
            elif end < dest.end - 1.0:
                continue  # lands clear of the destination's end
            pinned += self._pin_bound_poses(
                [seq["obj"]],
                (seq["start"] - _KEY_TOL, seq["end"] + _KEY_TOL),
                [(float(nxt.start), nxt.shot_id, "start")],
                attr=seq.get("attr"),
            )
        return pinned

    def _landing_floor(
        self, dest, new_end: float, room: float, placements, sources
    ) -> float:
        """The frame the destination's next shot keeps from a Move to Shot
        landing, when it is flush with the destination and the run(s)
        landing on its new end close on a different pose than its opening
        one (:meth:`_close_flush_seams`); otherwise 0.  A next shot that is
        itself a source of the move (*sources*) opens on whatever the move
        leaves it, which is the move's own edit of both shots."""
        shots = self.sorted_shots()
        idx = next(i for i, s in enumerate(shots) if s.shot_id == dest.shot_id)
        if idx + 1 >= len(shots):
            return 0.0
        nxt = shots[idx + 1]
        if abs(nxt.start - dest.end) > _KEY_TOL or nxt.shot_id in sources:
            return 0.0  # a gap the ripple keeps, or a shot the move edits
        enders = [
            seq
            for seq, _sid, target in placements
            if seq["kind"] == "anim"
            and abs(target + (seq["end"] - seq["start"]) - new_end) <= 1.0 - _KEY_TOL
        ]
        if not enders:
            return 0.0

        def layout(floors: Dict[int, float]) -> Dict[int, Tuple[float, float]]:
            shift = room + floors.get(nxt.shot_id, 0.0)
            out = {dest.shot_id: (dest.start, new_end)}
            for s in shots[idx + 1 :]:
                out[s.shot_id] = (s.start + shift, s.end + shift)
            return out

        def seams(plan) -> List[Tuple[int, int]]:
            lo, hi = plan[dest.shot_id][1], plan[nxt.shot_id][0]
            return [(dest.shot_id, nxt.shot_id)] if abs(hi - lo) < _KEY_TOL else []

        def poses(_prev_id, _next_id) -> List[tuple]:
            # Each run's own closing pose, read where it is now, against the
            # opening pose of the shot it lands beside.
            return [
                (seq["end"], nxt.start, [(seq["obj"], seq.get("attr"))])
                for seq in enders
            ]

        _plan, apart = self._close_flush_seams(layout, seams=seams, poses=poses)
        return apart.get(nxt.shot_id, 0.0)

    def _close_flush_seams(
        self,
        make_plan: Callable[[Dict[int, float]], Any],
        seams: Optional[Callable[[Any], List[Tuple[int, int]]]] = None,
        poses: Optional[Callable[[int, int], tuple]] = None,
        keep: Optional[Callable[[List[Tuple[int, int]]], List[bool]]] = None,
    ):
        """Build an op's plan with every seam it closes decided: flush where
        the two poses agree, kept as a straight cut where they differ --
        exactly one frame apart, no empty frame between the shots, the gap
        hold stepping the transition (the user's words, 2026-10-12).

        The maintainer's rule (2026-10-11) for every op that closes a seam to
        gap 0 -- a delete, the scoped gap collapse, a respace to 0, a band
        move, a slide, a Ctrl edge, Move to Shot's landing: one frame holds
        one pose, so a seam whose two poses differ keeps a frame of its own,
        and the op is never refused and never re-times a shot.  "Poses" are
        keyed AND held -- the value each curve plays there, read by the DCC
        (:meth:`_poses_differ`) before anything moves.  A widened seam can
        move what follows it, so the plan is asked again until no new seam
        closes; a seam left apart re-closes on a later op only once its poses
        agree.  The seams kept apart -- kept as straight cuts -- are reported in
        one info line (:meth:`_report_straight_cuts`).

        Parameters:
            make_plan: ``{shot_id: frames} -> plan``, the op's own planner
                (a :class:`MovePlan`, or ``{shot_id: (start, end)}``) with a
                floor on the gap BEFORE each named shot.
            seams: ``plan -> [(prev_id, next_id), ...]``, the seams the op
                closes; default :meth:`ShotPlanner.closing_seams` -- flush
                once the plan applies, apart now.
            poses: ``(prev_id, next_id) -> [(frame_a, frame_b, objects),
                ...]``, where each side's pose is read now (any pair that
                differs keeps the seam apart); default the previous shot's
                end and the next one's start, every content curve
                (``objects`` ``None``).
            keep: ``[(prev_id, next_id), ...] -> [bool, ...]``, the op's own
                further reason a seam keeps its frame though its poses agree
                (:meth:`respace`: its gap holds keys, :meth:`_seams_hold_keys`);
                default none.

        Returns:
            ``(plan, {shot_id: frames})`` -- the plan to apply and the seams
            it keeps apart.
        """
        one = self.store.snap(1.0) or 1.0
        widen: Dict[int, float] = {}
        why: Dict[int, str] = {}  # each kept seam's reason, for the report
        plan = make_plan(widen)
        if not self._scene_available():
            return plan, widen
        asked: set = set()
        for _ in range(len(self.store.shots)):
            if seams is not None:
                found = seams(plan)
            else:
                found = ShotPlanner.closing_seams(self.store, self._plan_bounds(plan))
            todo = [pair for pair in found if pair not in asked]
            if not todo:
                break
            asked.update(todo)
            clash = []
            kept = keep(todo) if keep is not None else [False] * len(todo)
            for (prev_id, next_id), differ, held in zip(
                todo, self._seam_poses_differ(todo, poses), kept
            ):
                if differ or held:
                    clash.append(next_id)
                    why[next_id] = "poses differ" if differ else "gap holds keys"
            if not clash:
                break
            for sid in clash:
                widen[sid] = one
            plan = make_plan(widen)
        if widen:
            self._report_straight_cuts(widen, why)
        return plan, widen

    def _report_straight_cuts(self, widen: Dict[int, float], why: Dict[int, str]):
        """The one info line naming the seams :meth:`_close_flush_seams` kept
        apart, by reason -- "kept as straight cuts": no empty frame sits
        between the two shots, the gap hold steps the transition (maintainer
        2026-10-12)."""
        shots = self.sorted_shots()
        groups: Dict[str, List[str]] = {}
        for i, shot in enumerate(shots):
            if shot.shot_id in widen and i > 0:
                groups.setdefault(why.get(shot.shot_id, "poses differ"), []).append(
                    f"{shots[i - 1].name}|{shot.name}"
                )
        count = sum(len(v) for v in groups.values())
        _log.info(
            "%d seam%s kept as %s %s",
            count,
            "" if count == 1 else "s",
            "a straight cut" if count == 1 else "straight cuts",
            "; ".join(
                f"({reason}): {', '.join(names)}" for reason, names in groups.items()
            ),
        )

    def _seams_hold_keys(self, pairs: List[Tuple[int, int]]) -> List[bool]:
        """Per seam ``(prev_id, next_id)``: whether the gap between the two
        shots holds the animator's keys now (:meth:`_gaps_keyed`).

        The further reason :meth:`respace` and a slide
        (:meth:`_closing_slide_start`) keep a seam's frame.  A gap's keys are
        retimed into its new width, and a width of 0 is refused -- that would
        cut or stack the animator's keys -- so a seam closed flush over a
        keyed gap left the keys where they were, and the shot after played
        them: SOL_REPLACE_ASSEMBLY pulled into Blender with its stepped keys,
        a respace to 0 played A04 0.0012 off.  One frame apart, the keys keep
        a gap to be retimed into (a slide's: :meth:`_retime_closed_gap`).
        """
        windows = []
        for prev_id, next_id in pairs:
            prev, nxt = self.shot_by_id(prev_id), self.shot_by_id(next_id)
            windows.append((prev.end + _KEY_TOL, nxt.start - _KEY_TOL))
        return list(self._gaps_keyed(windows)) if windows else []

    def _seam_poses_differ(self, pairs, poses=None) -> List[bool]:
        """:meth:`_poses_differ` for seams ``[(prev_id, next_id), ...]``: each
        read where *poses* says (``(frame_a, frame_b, objects)``), default the
        previous shot's end and the next shot's start over the whole content.
        One DCC read per object set."""
        asks: List[tuple] = []  # (seam index, object key, (frame_a, frame_b))
        for n, (prev_id, next_id) in enumerate(pairs):
            if poses is not None:
                reads = poses(prev_id, next_id)
            else:
                prev, nxt = self.shot_by_id(prev_id), self.shot_by_id(next_id)
                reads = [(prev.end, nxt.start, None)]
            for frame_a, frame_b, objects in reads:
                key = None if objects is None else tuple(sorted(objects))
                asks.append((n, key, (float(frame_a), float(frame_b))))
        out: List[bool] = [False] * len(pairs)
        for key in {k for _n, k, _f in asks}:
            mine = [a for a in asks if a[1] == key]
            got = self._poses_differ(
                [f for _n, _k, f in mine], None if key is None else list(key)
            )
            for (n, _k, _f), differ in zip(mine, got):
                out[n] = out[n] or bool(differ)
        return out

    def _hold_neighbour_poses(
        self,
        shot_id: int,
        objects: Iterable[str],
        window: Tuple[float, float],
        attr: Optional[str] = None,
        skip: Iterable[int] = (),
        flush: bool = False,
        crossing: Optional[bool] = None,
    ) -> int:
        """Pin, before content of *objects* inside *window* leaves *shot_id*,
        the pose each shot beside it plays on its facing bound
        (:meth:`_pin_bound_poses`): the previous shot's end, the next shot's
        start -- neither in *skip*, and only across a gap unless *flush*.
        *flush* (a delete): a seam shared with *shot_id* is pinned too -- the
        frame is the neighbour's once *shot_id* is gone (the delete's cut
        leaves a flush closing sample to the next shot), and where the next
        shot then lands is the closing rule's (:meth:`_close_flush_seams`) --
        and so is every curve crossing *window* keyless, which the closing
        would re-time (*crossing*, default *flush*: a Move to Shot closes
        nothing, so it leaves those alone).

        The one rule (maintainer 2026-10-10): an op on one shot never changes
        what another plays inside its own span.  The keys leaving were the
        neighbours' too wherever a curve's segment crossed into them -- the
        ramp out of the previous shot's last key, the value the leaving shot's
        last key held over every shot after it.  Measured 2026-10-11 on a
        production assembly at gap 10: deleting "A01_2" left "A02" playing
        KEYS 78 units off over all 81 of its frames, deleting "C01" moved
        "C02"'s SOLENOID by 7.2 over 256; at gap 0 the delete handed "A02"
        the deleted shot's closing pose -- KEYS 80 off.

        Returns the number of keys pinned.
        """
        shots = self.sorted_shots()
        idx = next((i for i, s in enumerate(shots) if s.shot_id == shot_id), None)
        objects = list(objects or ())
        if idx is None or not objects:
            return 0
        shot, skip = shots[idx], set(skip)
        prev = shots[idx - 1] if idx > 0 else None
        nxt = shots[idx + 1] if idx + 1 < len(shots) else None
        pins: List[Tuple[float, int, str]] = []
        if prev is not None and prev.shot_id not in skip:
            if flush or prev.end < shot.start - _EPS:
                pins.append((float(prev.end), prev.shot_id, "end"))
        if nxt is not None and nxt.shot_id not in skip:
            if flush or nxt.start > shot.end + _EPS:
                pins.append((float(nxt.start), nxt.shot_id, "start"))
        if not pins:
            return 0
        return self._pin_bound_poses(
            objects,
            (float(window[0]), float(window[1])),
            pins,
            attr=attr,
            crossing=flush if crossing is None else crossing,
        )

    def _animator_run(self, seq: Dict[str, Any]) -> None:
        """Narrow *seq*, in place, to the animator's own keys inside it.

        A run's drawn extent is its MOTION, and motion can run on into a
        sample the system planted for a bound -- a pin follows its bound, so
        it is never content to send (:meth:`_animator_times_in`).  Measured
        2026-10-11 on a production assembly at gap 0: a clip ending on the
        next shot's claimed start pin, on the shared frame, read as content
        from AFTER the destination; Move to Shot appended it behind the
        destination's own, the curve's last pose became the moved block's
        and the shot after played it -- KEYS 78 units off over all 81 frames.

        A key selection (``"times"``), an audio clip and a run with no key of
        the animator's in it are left as they are, and a run is only ever
        narrowed: read a key-time tolerance wide, so a key ON an end (stored
        single-precision in Blender) still holds that end.  In place, like
        :meth:`_absorb_gap_overhang`: the caller's dict reports what moved.
        """
        if seq["kind"] != "anim" or seq.get("times"):
            return
        lo, hi = seq["start"], seq["end"]
        tol = self.ledger.eps
        own = self._animator_times_in(seq["obj"], seq.get("attr"), (lo - tol, hi + tol))
        if own:
            seq["start"], seq["end"] = max(lo, min(own)), min(hi, max(own))

    def _absorb_gap_overhang(self, seq: Dict[str, Any], dest_start: float) -> None:
        """Widen a head-bound *seq* over its own keys hanging in the gap.

        A run's keys that continue past its shot's end into the gap facing
        the destination are the run's own overhang -- the on-ramp of a pulse
        whose train sits in the destination, a fade-out -- and the panel
        cannot even select them (nothing draws a gap key).  Left behind, they
        would sit AFTER the landed run in time and the ramp between them
        would flip: measured on a production assembly, an on-ramp
        ``2279 (0) -> 2288 (1)`` moved by its first key alone read
        ``2288 (1) -> 2295 (0)``, a flicker at the shot start.

        Only a pure overhang is taken: if any key of the run's curves lies
        between the run and the destination INSIDE a shot (the rest of the
        source shot, a shot in between), those gap keys belong to that later
        run and *seq* is left as it is.  Samples the system itself wrote for
        a shot bound (the ledger's claimed keys -- a seam hold at the source's
        end) are nobody's run: they neither pin the overhang nor travel
        (:meth:`_animator_times_in` reports the animator's keys only).
        """
        if seq.get("kind") != "anim":
            return
        eps = 1e-3
        lo, hi = seq["end"] + eps, dest_start - eps
        if hi <= lo:
            return
        found: List[float] = list(
            self._animator_times_in(seq["obj"], seq.get("attr"), (lo, hi))
        )
        if not found:
            return
        shots = self.store.shots
        if any(sh.start - eps <= t <= sh.end + eps for t in found for sh in shots):
            return
        seq["end"] = max(found)
        if seq.get("times") is not None:
            seq["times"] = sorted(set(seq["times"]) | set(found))

    # ---- fit / trim / extend ----------------------------------------------

    def fit_shot_to_content(
        self,
        shot_id: int,
        mode: str = "fit",
        edge: str = "both",
        reach: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Resize a shot's boundaries to its sequence content, rippling neighbors.

        Mode controls direction:
            ``"fit"`` — boundaries snap exactly to content (both expand and
                contract as needed).
            ``"trim"`` — only contract empty space; boundaries move *inward*
                and never past content.
            ``"extend"`` — only expand to enclose out-of-range content;
                boundaries move *outward* and never inward.

        *edge* restricts which end may move: ``"both"`` (default),
        ``"leading"`` (head only) or ``"trailing"`` (tail only).

        *reach* bounds how far outside the shot the ``"extend"`` / ``"fit"``
        probe looks, in frames, and widens it to BOTH gaps: the user's
        "extend to the keys I set" gesture, whose keys sit just past either
        bound.  ``None`` (the default, the implicit auto-extend) keeps the
        envelope rule -- the trailing gap only, the leading gap belonging to
        the previous shot (see :meth:`_key_extent`).  A neighbour's span is
        never read either way.

        Neighbouring shots ripple by the head/tail deltas so spacing is
        preserved.  Scene edits are batched (:meth:`_content_batch`).

        Returns:
            ``(head_delta, tail_delta)`` — the amount the start/end moved.
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            raise ValueError(f"No shot with id {shot_id}")

        sequences = self.collect_shot_sequences(shot_id)

        # ``sequences`` reports MOTION: ``collect_object_segments`` drops hold
        # spans so a clip reads as the animation it plays rather than as the
        # frames it occupies.  A BOUND may not move past a key, though — so
        # the in-bounds KEY extent is folded in beside the motion extent.
        # Without it a shot whose tail is a long hold read as empty at the
        # end: the trim put the bound in front of keys that then stayed
        # behind while the downstream ripple pulled the next shot's content
        # back on top of them.  Measured on a production assembly, "Step 4.1"
        # [991, 1590] collected motion out to 1313 while two of its members
        # hold keys to 1533, and one trailing trim moved "Step 4.8" from 1605
        # to 1328 — interleaving two shots' animation on six curves.
        #
        # For "extend" / "fit" the probe must ALSO look outside the current
        # range, which ``collect_shot_sequences`` clips away, to reveal
        # overrun.
        probe_outside = mode in ("extend", "fit")
        inner_start, inner_end, outer_start, outer_end, on_bound = self._key_extent(
            shot, probe_outside, reach=reach
        )

        probes = (inner_start, outer_start, outer_end)
        if not sequences and all(v is None for v in probes):
            return 0.0, 0.0

        seq_start = min(s["start"] for s in sequences) if sequences else None
        seq_end = max(s["end"] for s in sequences) if sequences else None

        def _combine(agg, *vals):
            present = [v for v in vals if v is not None]
            return agg(present) if present else None

        content_start = _combine(min, seq_start, inner_start, outer_start)
        content_end = _combine(max, seq_end, inner_end, outer_end)

        if mode == "extend":
            # One-sided rescue: content that drifted entirely past ONE edge
            # leaves the other side None — substituting the shot's own
            # boundary keeps extend usable in exactly the case it exists
            # for (enclosing out-of-range content).
            if content_start is None:
                content_start = shot.start
            if content_end is None:
                content_end = shot.end

        if content_start is None or content_end is None:
            return 0.0, 0.0

        if mode == "trim":
            new_start = max(shot.start, content_start)
            new_end = min(shot.end, content_end)
        elif mode == "extend":
            new_start = min(shot.start, content_start)
            new_end = max(shot.end, content_end)
        else:  # "fit"
            new_start = content_start
            new_end = content_end

        # Outward: rounded to the NEAREST frame, a start could land past the
        # first key and an end short of the last -- fractional content is what
        # a retime leaves (a trim put "Step 4.1"'s start at 985 over a key at
        # 984.556, handing that key to the neighbour's envelope).  The edge
        # *edge* excludes is not snapped at all: it does not move.
        new_start = (
            shot.start if edge == "trailing" else self.store.snap(new_start, "down")
        )
        new_end = shot.end if edge == "leading" else self.store.snap(new_end, "up")
        head_delta = new_start - shot.start
        tail_delta = new_end - shot.end
        if abs(head_delta) < _EPS and abs(tail_delta) < _EPS:
            return 0.0, 0.0

        old_start, old_end = shot.start, shot.end

        with self._content_batch():
            # A redundant key sitting ON a bound that is moving inward goes
            # first (see _key_extent), then the shot's own samples follow the
            # shrinking bounds BEFORE the neighbours ripple onto the frames it
            # is giving up.  A GROWING edge waits for the reconcile after the
            # ripple: with carry_gap the frames it uncovers are still the
            # gap's until the ripple has moved that content away.
            self._cut_passed_bound_keys(
                on_bound, old_start, old_end, new_start, new_end
            )
            # What each side plays across a moved edge, pinned (and a flush
            # shrink's closing), as a plain edge edit does it -- a grow from
            # its new bound (below).
            closing = self._hold_edge_neighbours(
                shot, new_start, new_end, grow_from_new=True
            )
            self._reconcile_pending_bounds(
                shot_id,
                new_start if head_delta > _EPS else old_start,
                new_end if tail_delta < -_EPS else old_end,
            )
            shot.start = new_start
            shot.end = new_end
            # A GROWING edge ripples from the NEW bound: everything beyond
            # what the shot now covers moves, and the keys it grew over stay
            # where they are -- enclosed, which is the whole point of an
            # extend.  Rippled from the OLD bound (carry_gap's reading of "the
            # pivot's trailing gap rides"), the very keys the grow reached for
            # rode away with the neighbour and landed outside the shot again:
            # measured, extending "S0" [0, 50] over its gap key at 55 grew the
            # shot to 55 and moved the key to 60.  A SHRINKING edge still
            # ripples from the old bound -- nothing of the shot's is left
            # between the two (a trim stops at content).
            if "end" in closing:
                closing["end"]()
            elif abs(tail_delta) > _EPS:
                self.ripple_downstream(
                    shot_id, new_end if tail_delta > 0 else old_end, tail_delta
                )
            if "start" in closing:
                closing["start"]()
            elif abs(head_delta) > _EPS:
                self.ripple_upstream(
                    shot_id, new_start if head_delta < 0 else old_start, head_delta
                )

        # reconcile, not just enforce: this moved a shot BOUND, which is exactly
        # when a boundary sample the system created has to follow it or be
        # cleaned up.
        self.reconcile_system_edits()
        self.store.mark_dirty()
        return head_delta, tail_delta

    def trim_shot_to_content(
        self, shot_id: int, edge: str = "both"
    ) -> Tuple[float, float]:
        """Shrink shot boundaries inward so they exactly enclose content.

        Empty leading/trailing space is removed; downstream/upstream shots
        ripple to preserve their spacing.  *edge* narrows the operation to
        one end — ``"leading"`` or ``"trailing"`` — leaving the other where
        the animator put it.
        """
        return self.fit_shot_to_content(shot_id, mode="trim", edge=edge)

    def extend_shot_to_fit(
        self, shot_id: int, edge: str = "both", reach: Optional[float] = None
    ) -> Tuple[float, float]:
        """Expand shot boundaries outward to enclose all of its sequences.

        If sequences extend past the current head or tail, the shot grows
        to cover them and neighbouring shots ripple outward.  *edge* limits
        the growth to one end; *reach* (frames) is the user's "extend to the
        keys I set" form -- keys within *reach* of either bound, in the gaps
        only (see :meth:`fit_shot_to_content`).
        """
        return self.fit_shot_to_content(shot_id, mode="extend", edge=edge, reach=reach)

    # ---- per-object key motion --------------------------------------------

    def move_object_keys(
        self,
        obj: str,
        old_start: float,
        old_end: float,
        new_start: float,
    ) -> None:
        """Offset all keyframes of *obj* that fall within [old_start, old_end]
        so the segment begins at *new_start*.

        Parameters:
            obj: Object name.
            old_start: Original first frame of the segment.
            old_end: Original last frame of the segment.
            new_start: Desired first frame after the move.
        """
        self.move_attribute_keys(
            obj, None, new_start - old_start, window=(old_start, old_end)
        )

    def move_stepped_keys(
        self,
        obj: str,
        old_time: float,
        new_time: float,
        attr_name: Optional[str] = None,
        eps: float = 1e-3,
    ) -> None:
        """Move a point (stepped) clip: the key(s) at *old_time* it stands for.

        A main-track point clip is a MARK (:meth:`_point_segment`): an
        animator's key on a member that holds still in its shot, drawn from
        that object's content channels whether the key steps or not.  So the
        drag moves exactly what the mark was drawn from -- each content
        curve whose own key at *old_time* the shot shows as a mark
        (:meth:`_point_clip_curves`) -- and nothing it did not draw.  The rule
        used to live in each DCC, and they disagreed: mayatk moved only keys
        with a ``step`` out-tangent, so a mark on a plain key (a Move to Shot
        arrival) dragged nothing while the shot still grew to enclose the
        drop; blendertk moved every fcurve keyed on that frame, channels the
        mark never drew included.

        *attr_name* (a sub-row clip) narrows the move to that channel, whose
        keys at *old_time* all go.  Each key moves through
        :meth:`_move_point_key`, its record and its claims with it.
        """
        if abs(new_time - old_time) < _EPS:
            return
        shot = next(
            (
                s
                for s in self.sorted_shots()
                if s.start - eps <= old_time <= s.end + eps
            ),
            None,
        )
        for curve in self._point_clip_curves(obj, old_time, attr_name, shot, eps):
            self._move_point_key(obj, curve, old_time, new_time, eps)

    def move_object_in_shot(
        self,
        shot_id: int,
        obj: str,
        old_start: float,
        old_end: float,
        new_start: float,
    ) -> None:
        """Move one object's keys within a shot, expanding the shot and
        rippling downstream shots when the clip exceeds shot boundaries.

        The shot grows to ENCLOSE the landing, by the rule a key or sub-row
        clip drag grows by (``ShotStore.enclosing_bounds``): outward to whole
        frames, and one frame past a contiguous seam the landing would sit on
        when the object holds a key there -- rounded and unstepped, a clip
        onto the seam opened the neighbour on the dragged pose.

        Parameters:
            shot_id: Shot the object belongs to.
            obj: Object name to move.
            old_start: Original first frame of the object segment.
            old_end: Original last frame of the object segment.
            new_start: Desired first frame after the move.
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            raise ValueError(f"No shot with id {shot_id}")

        new_start = self.store.snap(new_start)
        grown_start, grown_end = self.store.enclosing_bounds(
            shot_id,
            new_start,
            new_start + (old_end - old_start),
            self._object_key_probe(obj),
        )

        # Boundaries FIRST, keys second.  Expanding past the next shot's
        # start ripples that shot through its envelope, and a key already
        # landed at/past the envelope's start is swept a SECOND time -- the
        # clip travels the drag distance plus the ripple.  Measured on a
        # production assembly: a segment dragged +20 past the shot end came
        # out +38, its keys interleaved with the next shot's and their
        # auto tangents recomputed flat in the new neighbourhood.  Before the
        # move the keys still sit at their old times inside the pivot, which
        # the ripple plan excludes, so nothing can reach them.  (The per-key
        # drag path orders it the same way.)
        prior_start = shot.start
        prior_end = shot.end
        start_expanded = False
        end_expanded = False

        if grown_start < shot.start:
            shot.start = grown_start
            start_expanded = True

        if grown_end > shot.end:
            shot.end = grown_end
            end_expanded = True

        # Ripple upstream by however much the shot head grew
        if start_expanded:
            start_delta = shot.start - prior_start  # negative
            if abs(start_delta) > _EPS:
                self.ripple_upstream(shot_id, prior_start, start_delta)

        # Ripple downstream by however much the shot tail grew
        if end_expanded:
            end_delta = shot.end - prior_end  # positive
            if abs(end_delta) > _EPS:
                self.ripple_downstream(shot_id, prior_end, end_delta)
        if start_expanded or end_expanded:
            self.store.mark_dirty()

        # Move the object's keys into the room just opened for them.
        self.move_object_keys(obj, old_start, old_end, new_start)
        # A bound may have moved and the seam may have: a boundary sample
        # follows its bound or is cleaned up, and a hold the system stepped
        # is released where it no longer belongs and applied where it now
        # does.
        self.reconcile_system_edits()

    def _move_shot_content(
        self,
        shot: ShotBlock,
        new_start: float,
    ) -> None:
        """Shift all content (object keys and audio) for *shot* to *new_start*.

        Moves the keyed content through :meth:`_move_content_keys` and the
        audio through :meth:`_shift_audio`, then updates the shot boundaries.
        Without a scene (:meth:`_scene_available`) only the boundaries are
        updated.

        Parameters:
            shot: The shot to move.
            new_start: Desired first frame after the move.
        """
        new_start = self.store.snap(new_start)
        old_start = shot.start
        old_end = shot.end
        delta = new_start - old_start
        if abs(delta) < _EPS:
            return

        duration = old_end - old_start

        if self._scene_available():
            # The engine derives the pivot's envelope and one-move plan, so
            # this mover and the plan path cannot disagree about a shared
            # sample.  Falls back to the shot's own span only if the store
            # somehow does not hold it.
            plan = ShotPlanner.plan_pivot_move(self.store, shot.shot_id, new_start)
            move = plan.moves.get(shot.shot_id)
            if move is None:
                env_lo, env_hi, lo_open, hi_closed = old_start, old_end, False, True
            else:
                env_lo, env_hi = move.env_start, move.env_end
                lo_open, hi_closed = move.env_lo_open, move.env_hi_closed

            # A shot moving must carry everything keyed inside it, not just
            # what membership happens to list — stale entries (a renamed rig)
            # and objects animated after the shots were authored are both
            # invisible until the move strands them.  The envelope is the
            # contract, so the whole keyed content goes; the list stays the
            # label it is (see :meth:`_content_objects`).
            finish = self._reconcile_boundaries(plan)
            self._move_content_keys(
                self._content_objects(),
                env_lo,
                env_hi,
                delta,
                lo_open=lo_open,
                hi_closed=hi_closed,
            )
            # Audio keeps its own inclusive [old_start, old_end] window: clips
            # are a separate store with no fencepost sharing.
            self._shift_audio(old_start, old_end, delta)
            finish()

        shot.start = new_start
        shot.end = self.store.snap(new_start + duration)

    # ---- whole-shot moves -------------------------------------------------

    def move_shot(self, shot_id: int, new_start: float) -> None:
        """Move an entire shot (all object keys) to *new_start*, rippling downstream.

        The shot's duration is preserved.  All keyframes belonging to
        the shot's objects are shifted by the same delta.  Downstream
        shots are then shifted to maintain their original spacing
        relative to this shot's end.

        Parameters:
            shot_id: The shot to move.
            new_start: Desired new start frame.
        """
        self.slide_shot(shot_id, new_start, direction="downstream")

    def _clamp_slide_start(self, shot, new_start: float, rippled: Optional[str]):
        """Hold *new_start* inside the room the neighbours are NOT making.

        A slide moves the shot whole and ripples ONE side, or neither — so
        the other side has to hold.  Without that the pivot slides straight
        over its neighbour and the store ends up with two shots claiming one
        span, which makes key ownership (and every envelope derived from it)
        ambiguous: exactly the corruption the inner gap-edge drag already
        guards against, reached instead through the gesture that moves a
        whole shot.  Measured on a production assembly: an outer gap drag
        pulled "Step 4.8" 212 frames earlier, from [1605, 2180] to
        [1393, 1968], while "Step 4.4" [1373, 1605] — which the downstream
        ripple never touches — stayed put, so the two shots shared 212 frames
        and the panel drew "Step 4.4" ending mid-content.

        Parameters:
            shot: The shot being slid.
            new_start: The requested start frame.
            rippled: ``"downstream"``, ``"upstream"``, or ``None`` when
                neither side moves and both therefore hold.

        Returns:
            *new_start*, clamped against whichever side is not rippling.
        """
        sorted_s = self.sorted_shots()
        idx = next(
            (i for i, s in enumerate(sorted_s) if s.shot_id == shot.shot_id), None
        )
        if idx is None:
            return new_start
        # Tail first, head second: the head clamp wins a tie, so a shot with
        # nowhere to go stays put rather than swapping which neighbour it
        # overlaps.  (A shot that started inside valid bounds always has
        # room, so the two cannot actually conflict.)
        if rippled != "downstream" and idx + 1 < len(sorted_s):
            nxt = sorted_s[idx + 1]
            # Not just the span: the shot's content -- its trailing tail in
            # the gap included -- must stop short of the neighbour.  Measured
            # 2026-09-07: "Step 6" slid onto "Step 7" arrived with its
            # highlight lead-out (2131.2) inside Step 7's span, and the
            # landing-zone push then moved Step 7's keys by a different
            # delta, tearing the pulse.  A tail lands before the neighbour's
            # start; a shot with no tail may still close the gap entirely
            # (contiguous shots share their boundary sample).
            # The shot's own envelope, read by the ONE content scan
            # (:meth:`_key_extent`): its trailing gap up to the neighbour.
            # A second walk here would be the same question asked twice, and
            # would answer it differently -- it counted a flat bake's keys,
            # which are not content and block no slide.
            _s, _e, _os, outer_end, _ob = self._key_extent(shot, True)
            reach = shot.end if outer_end is None else max(shot.end, outer_end)
            limit = nxt.start - (reach - shot.start)
            if reach > shot.end + _EPS:
                limit -= 1.0
            new_start = min(new_start, limit)
        if rippled != "upstream" and idx > 0:
            new_start = max(new_start, sorted_s[idx - 1].end)
        return new_start

    def slide_shot(
        self,
        shot_id: int,
        new_start: float,
        direction: str = "downstream",
        _enforce: bool = True,
    ) -> None:
        """Slide a shot intact to *new_start*, rippling only in *direction*.

        Unlike :meth:`move_shot` which always ripples downstream, this
        method lets the caller choose which side of the timeline absorbs
        the displacement.  The shot's duration and internal keyframes
        are preserved (translated, not scaled).

        Parameters:
            shot_id: The shot to slide.
            new_start: Desired new start frame.
            direction: ``"downstream"`` ripples shots after this one;
                ``"upstream"`` ripples shots before this one; ``None``
                ripples neither -- the shot slides between its neighbours,
                clamped by both (a gap handle's plain drag).
            _enforce: If True (default), call :meth:`_enforce_gap_holds`
                after the operation.  Pass False when batching multiple
                edits and calling it once at the end.
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            raise ValueError(f"No shot with id {shot_id}")

        old_start = shot.start
        old_end = shot.end
        new_start = self._clamp_slide_start(shot, new_start, direction)
        new_start = self._closing_slide_start(shot, new_start, direction)
        delta = new_start - old_start
        if abs(delta) < _EPS:
            return
        self._slide(shot, old_start, old_end, new_start, delta, direction, _enforce)

    def _closing_slide_start(
        self, shot: ShotBlock, new_start: float, direction: Optional[str]
    ) -> float:
        """*new_start* for a slide, one frame short of a neighbour it would
        land flush on with a different pose (:meth:`_close_flush_seams`): the
        band move (``move_shot``), a gap handle's slide, the scoped gap
        collapse.  The slide's own layout (:meth:`_slid_bounds`) is the plan:
        the side that ripples keeps its gap, so only the other can close.
        A gap holding the animator's keys keeps its frame too, as a
        respace's does (:meth:`_seams_hold_keys`; maintainer 2026-10-12):
        the keys are retimed into it (:meth:`_retime_closed_gap`)."""
        delta = new_start - shot.start
        if abs(delta) < _EPS:
            return new_start
        shots = self.sorted_shots()
        idx = next(i for i, s in enumerate(shots) if s.shot_id == shot.shot_id)
        nxt = shots[idx + 1] if idx + 1 < len(shots) else None

        def layout(floors: Dict[int, float]) -> Dict[int, Tuple[float, float]]:
            start = new_start + floors.get(shot.shot_id, 0.0)
            if nxt is not None:
                start -= floors.get(nxt.shot_id, 0.0)
            return self._slid_bounds(shot, start - shot.start, direction)

        _plan, apart = self._close_flush_seams(layout, keep=self._seams_hold_keys)
        out = new_start + apart.get(shot.shot_id, 0.0)
        if nxt is not None:
            out -= apart.get(nxt.shot_id, 0.0)
        return out

    def _retime_closed_gap(
        self, shot: ShotBlock, new_start: float, direction: Optional[str]
    ) -> int:
        """Before a slide lands *shot* a frame from the shot before it, that
        gap's keys -- the stationary shot's, which it does not carry -- are
        retimed into the frame the closing rule kept (:meth:`_seams_hold_keys`),
        as a respace retimes them (maintainer 2026-10-12, "same as respace").
        Left where they were, the slid shot landed over them and played them
        (A [0, 60] still, B [100, 160] moved to 60: 0.5 for 1.0).  The poses
        both bounds play are pinned first (:meth:`_pin_bound_poses`), so
        neither shot plays the retime.  Rippled upstream, the shot before
        moves with *shot* and the gap keeps its width.  Returns the curves
        retimed."""
        if direction == "upstream":
            return 0
        shots = self.sorted_shots()
        idx = next(i for i, s in enumerate(shots) if s.shot_id == shot.shot_id)
        if idx == 0:
            return 0
        prev = shots[idx - 1]
        return self._retime_closing_gap(prev, shot, prev.end, new_start)

    def _retime_closing_gap(
        self, left: ShotBlock, right: ShotBlock, new_lo: float, new_hi: float
    ) -> int:
        """The gap between *left* and *right* is about to become ``(new_lo,
        new_hi)``, the frame the closing rule kept for its keys
        (:meth:`_seams_hold_keys`): the poses both bounds play are pinned
        (:meth:`_pin_bound_poses`), the keys retimed into that width at the
        gap's left edge (:meth:`_retime_closing_gaps`), then carried to
        *new_lo* when that edge moves -- a Shift edge growing *left*'s end
        (:meth:`set_shot_edge`).  Before anything else moves; the one step
        behind a slide's and a Shift edge's closed gap.  Returns the curves
        retimed (0: no gap closing to a kept frame, or no keys in it)."""
        if not self._scene_available():
            return 0
        lo, hi = left.end, right.start
        width = new_hi - new_lo
        one = self.store.snap(1.0) or 1.0
        if hi - lo <= one + _KEY_TOL or not -_EPS < width <= one + _EPS:
            return 0  # no gap to close, or one the edit keeps open
        if not self._seams_hold_keys([(left.shot_id, right.shot_id)])[0]:
            return 0
        content = self._content_objects()
        self._pin_bound_poses(
            content,
            (lo + _KEY_TOL, hi - _KEY_TOL),
            [(lo, left.shot_id, "end"), (hi, right.shot_id, "start")],
        )
        done = self._retime_closing_gaps(
            [GapRetime(left.shot_id, right.shot_id, lo, hi, max(0.0, width), 0.0)]
        )
        if done and abs(new_lo - lo) > _EPS:
            # Strictly inside the retimed width: neither bound's key goes.
            self._move_content_keys(content, lo, lo + width, new_lo - lo, lo_open=True)
        return done

    def _slid_bounds(self, shot, delta: float, direction: Optional[str]) -> dict:
        """``{shot_id: (start, end)}`` once *shot* has slid by *delta* and the
        *direction* side rippled with it: the :meth:`pending_bounds` of a
        slide.  Its two steps -- the ripple, then the pivot -- each leave the
        pivot and its rippled neighbour a gap apart for a moment, and the
        seam between them is not split: both sides' content moves by the
        same delta."""
        shots = self.sorted_shots()
        idx = next(i for i, s in enumerate(shots) if s.shot_id == shot.shot_id)
        moving = [shot]
        if direction == "downstream":
            moving += shots[idx + 1 :]
        elif direction == "upstream":
            moving += shots[:idx]
        snap = self.store.snap  # as the planner and the pivot mover land them
        return {s.shot_id: (snap(s.start + delta), snap(s.end + delta)) for s in moving}

    def _slide(self, shot, old_start, old_end, new_start, delta, direction, enforce):
        """The ordered ripple + content move behind :meth:`slide_shot`, a
        gap it closes over the shot before retimed first
        (:meth:`_retime_closed_gap`)."""
        self._retime_closed_gap(shot, new_start, direction)
        with self.pending_bounds(self._slid_bounds(shot, delta, direction)):
            self._slide_moves(shot, old_start, old_end, new_start, delta, direction)
        if enforce:
            self._enforce_gap_holds()
        self.store.mark_dirty()

    def _slide_moves(self, shot, old_start, old_end, new_start, delta, direction):
        """:meth:`_slide`'s ripple and pivot move, in their collision-safe order."""
        shot_id = shot.shot_id
        # Vacate the destination side before moving the pivot: when the
        # pivot moves TOWARD the shots about to ripple, moving it first
        # would deposit its keys inside a neighbor's not-yet-read
        # envelope and shared-object curves would be shifted twice.
        if direction == "downstream":
            # The pivot carries its own trailing gap; the ripple must not.
            if delta > 0:
                self.ripple_downstream(shot_id, old_end, delta, carry_gap=False)
                self._move_shot_content(shot, new_start)
            else:
                self._move_shot_content(shot, new_start)
                self.ripple_downstream(shot_id, old_end, delta, carry_gap=False)
        elif direction == "upstream":
            if delta < 0:
                self.ripple_upstream(shot_id, old_start, delta, carry_gap=False)
                self._move_shot_content(shot, new_start)
            else:
                self._move_shot_content(shot, new_start)
                self.ripple_upstream(shot_id, old_start, delta, carry_gap=False)
        else:
            self._move_shot_content(shot, new_start)

    def ripple_downstream(
        self, shot_id: int, after_frame: float, delta: float, carry_gap: bool = True
    ) -> None:
        """Shift all shots starting at or after *after_frame* by *delta*.

        Routes through :class:`ShotPlanner` and :meth:`_apply_plan` so the
        whole downstream topology is resolved before any keyframe is touched
        -- preventing envelope collisions between moved and not-yet-moved
        shots.  The public ripple entry point for callers outside the
        sequencer (settings panel, clip motion).

        Every moved shot's window is its envelope, so it moves WHOLE, and
        with *carry_gap* (the default) so does whatever is keyed between
        *after_frame* and the first moved shot -- the pivot's trailing gap.
        That is the user's rule for a BOUND change (2026-09-06: "only the
        current shot changes, everything else ripples"): a fade tail parked
        in the gap is "everything else".  Left behind, a grow swallowed it
        into the shot and a shrink landed the neighbour on it; a retime's
        stretched keys landed on it too (measured: one Shift drag merged 80
        keys away on every baked curve of the production assembly).  A
        WHOLE-SHOT move already carries its gap inside its own envelope
        (:meth:`_move_shot_content`) and passes ``carry_gap=False``, or the
        same keys would move twice.  A sample ON *after_frame* in the gap
        stays with the pivot; the carry never cuts into a shot it moves, so a
        bound on or past the neighbour's start moves the neighbour whole.
        """
        plan = ShotPlanner.plan_ripple_downstream(
            self.store, shot_id, after_frame, delta, carry_gap=carry_gap
        )
        self._apply_plan(plan)

    def ripple_upstream(
        self, shot_id: int, before_frame: float, delta: float, carry_gap: bool = True
    ) -> None:
        """Shift all shots ending at or before *before_frame* by *delta*.

        Routes through the plan/executor pair; see :meth:`ripple_downstream`
        for *carry_gap* (here: the last moved shot's window is capped at a
        *before_frame* in its trailing gap, so the pivot keeps the sample on
        its bound; a bound on or before that shot's end moves it whole).
        """
        plan = ShotPlanner.plan_ripple_upstream(
            self.store, shot_id, before_frame, delta, carry_gap=carry_gap
        )
        self._apply_plan(plan)

    # ---- system-edit maintenance ------------------------------------------

    def _enforce_gap_holds(self) -> None:
        """Hold every inter-shot gap, and release the holds that no longer are.

        A gap must not contain interpolated motion, so the last key before it
        is held (a stepped / constant segment).  The release half is what
        keeps those from accumulating: a key the system held that is no
        longer the seam — because the boundary moved, the gap closed, the
        shot was deleted, or the animator dragged the key inward — gets its
        original interpolation back.  Only keys this system held are ever
        restored (see :meth:`_release_gap_holds`).

        A hold must stop where the next shot starts, so before one is written
        the next shot's start is pinned on the held curve
        (:meth:`_hold_pin_bounds`, written by the :meth:`_pin_bound_keys`
        hook): a key at its current value there, so its animation cannot
        change.  Without it the hold ran on to that shot's first key and
        replaced its opening frames -- measured in a fresh Blender 5.1 and
        mayapy alike, a no-op respace changed 4 frames of one shot and 6 of
        the next, by up to 7.9 units, because only a respace that changed a
        gap's width pinned.  A shot's start pin is never held itself
        (:meth:`_holdable_seams`): the only key a curve has in that shot, it
        would hold the shot still from its first frame.

        Called automatically after every timeline-modifying operation, and
        idempotent: a second run finds every seam already held and every
        claim still wanted, so it writes nothing.

        Every pass offers every unheld seam its pin, and keeps no memory of
        the last: what a seam needs is read from the scene each time, so an
        edit made by any other path (a Graph Editor key, an undo, a reload)
        can never leave one skipped.  The hook is what has to be cheap where
        there is nothing to write -- measured 2026-10-11 on a production
        assembly, a New Shot spent 1.0 of its 1.5 s in 190 pin writes that
        inserted nothing; the DCC hooks now decline those from one read of
        the curve (mayatk ``_pinnable_frames``, blendertk ``_pinnable``).
        """
        if not self._scene_available():
            return
        raw = self._gap_hold_seams()
        seams = self._holdable_seams(raw)
        # A pin moves no seam, but a key a DCC puts ON a bound instead of
        # pinning beside it (its merge distance) can: read the seams again
        # while a pass changed anything.  Each pass also carries a keyless
        # crossing one gap on (:meth:`_through_pins`).
        for _pass in range(len(self.store.shots)):
            pins = self._hold_pin_bounds(seams)
            through = self._through_pins(raw, seams)
            for curve, want in through.items():
                pins[curve] = sorted(set(pins.get(curve, ())) | set(want))
            claimed = {c: self.ledger.key_times(c) for c in through}
            if not pins or not self._pin_bound_keys(pins):
                break
            self._keep_through_ends(through, claimed)
            raw = self._gap_hold_seams()
            seams = self._holdable_seams(raw)
        self._release_gap_holds(seams)
        self._apply_gap_holds(seams)

    def _through_pins(
        self, raw: Dict[str, list], held: Dict[str, list]
    ) -> Dict[str, List[Tuple[float, int, str]]]:
        """``{curve: [(frame, shot_id, edge), ...]}`` for each seam of *raw*
        that :meth:`_holdable_seams` dropped from *held*: a shot's start pin,
        the curve running on through that shot keyless and across the gap
        after it.  That shot's END and the next shot's START are pinned
        (:meth:`_pin_bound_keys`), so the crossing ends on the shot's own
        pose, the gap holds from there like any other (the next pass), and a
        ripple of what follows no longer re-times the segment.

        Unpinned, a ramp running on from a start pin -- a curve no member of
        the shot keys, held from the gap before -- was re-timed by every
        ripple past it: a trim played the next shot 1.2 off, and on SOL
        pulled into Blender a band move played the shots it crossed about 1.0
        off (the LCKR door ramp through 30 shots, a SCREWDRIVER ramp).
        """
        eps = self.ledger.eps
        shots = self.sorted_shots()
        out: Dict[str, List[Tuple[float, int, str]]] = {}
        for curve, times in raw.items():
            kept = held.get(curve, ())
            for t in times:
                if any(abs(t - k) <= eps for k in kept):
                    continue
                i = next(
                    (j for j, s in enumerate(shots) if abs(s.start - t) <= eps), None
                )
                if i is None or i + 1 >= len(shots):
                    continue
                shot, nxt = shots[i], shots[i + 1]
                if nxt.start - shot.end < _EPS:
                    continue  # flush: no gap to hold
                out.setdefault(curve, []).extend(
                    [
                        (float(shot.end), shot.shot_id, "end"),
                        (float(nxt.start), nxt.shot_id, "start"),
                    ]
                )
        return out

    def _keep_through_ends(
        self, through: Dict[str, list], claimed: Dict[str, List[float]]
    ) -> None:
        """Mark the END pins :meth:`_through_pins` just put in ``stays``: each
        is the shot's own pose on its end, so a later move of that end leaves
        it where it is (disowned) instead of carrying it off -- followed, a
        grow re-timed the shot's whole crossing (0.44 off).  *claimed*: each
        curve's claimed times before the pins went in; only new ones are
        marked."""
        led = self.ledger
        for curve, want in through.items():
            before = claimed.get(curve, ())
            for frame, shot_id, edge in want:
                if edge != "end" or any(abs(frame - t) <= led.eps for t in before):
                    continue
                for t, owner, e in led.key_records(curve):
                    if abs(t - frame) <= led.eps and (owner, e) == (shot_id, edge):
                        led.release_key(curve, t)
                        led.record_key(curve, t, owner, e, stays=True)

    def _hold_runs_through_shots(
        self, key_time: float, next_key: Optional[float]
    ) -> bool:
        """Whether a hold from a key at *key_time* to the curve's next key at
        *next_key* (``None``: none, the hold is the curve's own extrapolation)
        runs through the start of a shot -- what that shot plays, so a DCC's
        :meth:`_release_gap_holds` keeps it.

        A gap hold stops at the next shot's start, where it is pinned before
        it is written (:meth:`_hold_pin_bounds`).  A scene held before that
        pin existed has holds running on to the curve's next key, through
        every shot between, and those shots play the held pose: released,
        the restored interpolation re-timed all of them (measured
        2026-10-11 on a production assembly: a respace 10 -> 0 played a held
        door off its hold in 15 shots, by up to 1.22 in "C12", and the
        layout settled only on a second respace).  Kept, the claim still
        goes: the step is the shots' pose now, not a seam's.
        """
        if next_key is None:
            return False
        lo, hi = key_time + _KEY_TOL, next_key - _KEY_TOL
        return any(lo < s.start < hi for s in self.store.shots)

    def _holdable_seams(self, seams: Dict[str, list]) -> Dict[str, list]:
        """*seams* without the ones that are a shot's claimed START sample.

        The seam is the last key before the next shot, so a start sample is
        one only when it is the curve's sole key in its shot: the curve passes
        through that shot keyless and was pinned on its start to stop the
        previous gap's hold.  It has no content there to hold, and stepped it
        would hold the shot still from its first frame.
        """
        eps = self.ledger.eps
        out: Dict[str, list] = {}
        for curve, times in seams.items():
            starts = [
                t
                for t, _owner, edge in self.ledger.key_records(curve)
                if edge == "start"
            ]
            kept = [t for t in times if not any(abs(t - s) <= eps for s in starts)]
            if kept:
                out[curve] = kept
        return out

    def _hold_pin_bounds(
        self, seams: Dict[str, list]
    ) -> Dict[str, List[Tuple[float, int, str]]]:
        """``{curve: [(frame, shot_id, "start"), ...]}`` -- where each seam's
        hold has to stop: the start of the shot after its gap, the first shot
        to start after the seam.

        A seam the ledger already holds is skipped: its key steps, so the span
        it opens is a hold, which no pin splits.
        """
        shots = self.sorted_shots()
        starts = [s.start for s in shots]
        eps = self.ledger.eps
        pins: Dict[str, set] = {}
        for curve, times in seams.items():
            for t in times:
                if self.ledger.owns_step(curve, t):
                    continue
                i = bisect.bisect_right(starts, t + eps)
                if not 0 < i < len(shots):
                    continue
                before, after = shots[i - 1], shots[i]
                if after.start - before.end < _EPS:
                    continue  # no gap: nothing is held here
                pins.setdefault(curve, set()).add(
                    (float(after.start), after.shot_id, "start")
                )
        return {curve: sorted(want) for curve, want in pins.items()}

    def _reconcile_pending_bounds(
        self, shot_id: int, new_start: float, new_end: float
    ) -> None:
        """Resolve *shot_id*'s claimed samples against the bounds it is ABOUT
        to have, before anything else moves onto the frames they sit on.

        A bound move on this shot is a ripple on its neighbours, and the
        ripple lands their keys on exactly the frames a shrinking bound just
        gave up — where this shot's own end/start samples still sit.  Run
        afterwards, the reconcile finds those samples wedged between the
        neighbour's arrivals: the frame the bound moved to is occupied, and
        the plateau test that would call the sample redundant now reads the
        NEIGHBOUR's poses, so it is disowned and left behind as content of a
        shot that never authored it.

        Measured 2026-09-06 on the production assembly, and the report that
        found it ("when i trimmed step 3.1, Step 3.3 became broken -- namely
        the plug animation now jumps"): Trim Trailing Space on "Step 3.1"
        [81, 431] moved the end to 401 and rippled "Shot 3.3" to [416, 604];
        three plug curves kept the claimed 431 sample (0.0) between the landed
        421 (0.0) and 441 (-5.86), so a 20-frame ramp played as a hold and a
        10-frame ramp — 19 frames off on each of the three.  Run first, the
        same sample reads as the flat plateau it is (401 .. 446, both 0.0,
        stepped) and is cut.

        The four-way resolution is :meth:`_reconcile_boundary_keys`'s, called
        with the pending bounds; only this shot's claims are touched.
        """
        self._reconcile_boundary_keys(bounds={shot_id: (new_start, new_end)})

    def reconcile_system_edits(self, follow: bool = True) -> Dict[str, int]:
        """Release every shot-system write whose boundary has moved on.

        The single maintenance entry point, safe to call after any mutation:
        boundary samples follow their bound (or are cleaned up), then gap
        holds are released and re-applied at the current seams.

        Parameters:
            follow: ``False`` for an edit that moves a bound and NOTHING else
                (the Ctrl edge drag): a sample whose bound moved stays where
                it is instead of following (see :meth:`_reconcile_boundary_keys`).

        Returns:
            ``{"keys_moved", "keys_removed", "holds"}`` counts.
        """
        moved, removed = self._reconcile_boundary_keys(follow=follow)
        self._enforce_gap_holds()
        self._forget_deleted_behavior_curves()
        return {
            "keys_moved": moved,
            "keys_removed": removed,
            "holds": self.ledger.step_count,
        }

    def _forget_deleted_behavior_curves(self) -> int:
        """Forget the claims on every behavior curve the scene no longer has.

        The gap-hold and boundary passes forget their own registers' deleted
        curves; a curve holding only behavior keys (a highlight, a fade) is
        neither's, so its claims outlived it, and a key later set on a curve of
        that name at a claimed frame read as the tool's -- the next build
        could replace it.  Renamed claims are followed first
        (:meth:`_follow_unresolved_claims`), so a rename re-keys rather than
        forgets.  Returns how many curves were forgotten.
        """
        led = self.ledger
        self._follow_unresolved_claims(led.authored_curves())
        gone = [c for c in led.authored_curves() if not self._curve_exists(c)]
        for curve in gone:
            led.forget_curve(curve)
        return len(gone)

    # ---- resize -------------------------------------------------------------

    def expand_shot(
        self,
        shot_id: int,
        new_end: float,
    ) -> float:
        """Expand a shot's end frame and ripple downstream shots.

        Only expands — if *new_end* is not greater than the current end,
        no change is made.

        Parameters:
            shot_id: ID of the shot to expand.
            new_end: Desired new end frame.

        Returns:
            The delta by which the shot was expanded (0 if unchanged).
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            raise ValueError(f"No shot with id {shot_id}")
        if new_end <= shot.end:
            return 0.0
        delta = new_end - shot.end
        old_end = shot.end
        # Ripple first, then the bound (resize_shot_bounds' order): written
        # first, a flush neighbour read as no neighbour and its seam was
        # never split -- see :meth:`add_shot_space`.
        self.ripple_downstream(shot_id, old_end, delta)
        shot.end = self.store.snap(new_end)
        self.reconcile_system_edits()
        return delta

    def resize_object(
        self,
        shot_id: int,
        obj: str,
        old_start: float,
        old_end: float,
        new_start: float,
        new_end: float,
    ) -> None:
        """Scale one object's keys and ripple-shift all downstream shots.

        Only the named *obj* is scaled.  Other objects in the same shot
        are untouched.  Downstream shots are shifted by the end-frame
        delta so the gap is preserved.

        Parameters:
            shot_id: Shot the object belongs to.
            obj: Object name to resize.
            old_start: Original first frame of the object segment.
            old_end: Original last frame of the object segment.
            new_start: Desired first frame after the resize.
            new_end: Desired last frame after the resize.
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            raise ValueError(f"No shot with id {shot_id}")

        new_start = self.store.snap(new_start)
        new_end = self.store.snap(new_end)

        # Ripple FIRST by however much the shot's envelope has to grow, then
        # scale: a key scaled past the old end into the neighbour's window
        # rode the ripple (the order :meth:`resize_shot` fixed the same way).
        prior_start = shot.start
        prior_end = shot.end
        grown_start = min(shot.start, new_start)
        grown_end = max(shot.end, new_end)
        head_delta = grown_start - prior_start
        if abs(head_delta) > _EPS:
            self.ripple_upstream(shot_id, prior_start, head_delta)
        delta = grown_end - prior_end
        if abs(delta) > _EPS:
            self.ripple_downstream(shot_id, prior_end, delta)

        # Scale only this object's keys
        self.scale_object_keys(obj, old_start, old_end, new_start, new_end)
        shot.start = grown_start
        shot.end = grown_end
        self.reconcile_system_edits()
        self.store.mark_dirty()

    def set_shot_duration(self, shot_id: int, new_duration: float) -> None:
        """Change a shot's duration and ripple-shift all downstream shots.

        The shot's *start* stays fixed; its *end* moves, and every
        downstream shot shifts by the same delta.

        Parameters:
            shot_id: ID of the shot to resize.
            new_duration: Desired duration in frames.
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            raise ValueError(f"No shot with id {shot_id}")

        delta = new_duration - shot.duration
        if abs(delta) < _EPS:
            return

        old_end = shot.end
        new_end = self.store.snap(shot.start + new_duration)

        # A growing end ripples FIRST and a shrinking one AFTER the scale --
        # see :meth:`resize_shot` (and its pending bounds).
        with self.pending_bounds({shot_id: (shot.start, new_end)}):
            if delta > 0:
                self.ripple_downstream(shot_id, old_end, delta)
            self.scale_shot_keys(shot.start, old_end, shot.start, new_end)
            if delta < 0:
                self.ripple_downstream(shot_id, old_end, delta)
        shot.end = new_end
        self.reconcile_system_edits()
        self.store.mark_dirty()

    def resize_shot(
        self,
        shot_id: int,
        new_start: float,
        new_end: float,
        _enforce: bool = True,
    ) -> None:
        """Resize a shot to [new_start, new_end], scaling all keys and rippling.

        Both edges may move.  Keyframes are scaled from the old range
        into the new one.  Downstream shots are shifted by any change
        in the tail, and upstream shots are shifted by any change in
        the head.

        Parameters:
            shot_id: ID of the shot to resize.
            new_start: Desired start frame.
            new_end: Desired end frame.
            _enforce: If True (default), call :meth:`_enforce_gap_holds`
                after the operation.  Pass False when batching multiple
                edits and calling it once at the end.
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            raise ValueError(f"No shot with id {shot_id}")

        new_start = self.store.snap(new_start)
        new_end = self.store.snap(new_end)
        # Same inversion normalization as resize_shot_bounds — inverted
        # bounds would store start > end and hand the retime an inverted
        # target range.
        if new_end < new_start:
            new_start, new_end = new_end, new_start
        old_start, old_end = shot.start, shot.end
        if abs(new_start - old_start) < _EPS and abs(new_end - old_end) < _EPS:
            return

        # Every GROWING edge ripples FIRST, then the scale, then every
        # SHRINKING edge ripples: the scale must write into an empty span and
        # never reach a neighbour's keys.
        #
        # A grow vacates the room first.  A neighbour's move window is read
        # from ITS start, so a key scaled past the old end into that window
        # rode away with the ripple: measured, doubling [0, 50] to [0, 100]
        # beside a shot at 60 sent the key scaled to 80 on to 130.
        #
        # A shrink closes the room AFTER.  Rippled first, the neighbours --
        # and this shot's own trailing-gap content -- land inside the span its
        # content still occupies whenever the shrink is wider than the gap,
        # and the scale then retimes THEIR keys with its own.  Measured
        # 2026-09-19 on the production assembly: a -6 duration change pulled
        # the gap's baked keys onto the shot's own (the equal-pose merge cut
        # 3 keys from each of 80 curves) and compressed content into the
        # shot that belonged beyond it.
        tail_delta = new_end - old_end
        head_delta = new_start - old_start
        # A flush neighbour on a retimed edge keeps its own pose and slope
        # there: a curve crossing that seam keyless is pinned first, so the
        # DCC's retime (:meth:`scale_shot_keys`) has a key to hold the
        # neighbour's half on (BACKLOG 2026-10-11: keyless, the next shot
        # opened 0.08 off).
        self._pin_retimed_seams(shot, head_delta, tail_delta)
        # The retimed content fills the new span, its closing pose landing on
        # the new seam: the ripple's moment of gap is no split to pin
        # (:meth:`pending_bounds`).
        with self.pending_bounds({shot_id: (new_start, new_end)}):
            if tail_delta > _EPS:
                self.ripple_downstream(shot_id, old_end, tail_delta)
            if head_delta < -_EPS:
                self.ripple_upstream(shot_id, old_start, head_delta)

            self.scale_shot_keys(old_start, old_end, new_start, new_end)

            if tail_delta < -_EPS:
                self.ripple_downstream(shot_id, old_end, tail_delta)
            if head_delta > _EPS:
                self.ripple_upstream(shot_id, old_start, head_delta)
        shot.start = new_start
        shot.end = new_end

        if _enforce:
            # reconcile, not just enforce: this moved a shot BOUND, which
            # is exactly when a boundary sample the system created has to
            # follow it or be cleaned up.
            self.reconcile_system_edits()
        self.store.mark_dirty()

    def _pin_retimed_seams(self, shot, head_delta: float, tail_delta: float) -> int:
        """Pin the pose every content curve plays on each FLUSH seam of *shot*
        whose edge a retime moves (:meth:`_pin_bound_poses`, keyless curves
        crossing it included), claimed as the shot before's end -- the
        sample the fencepost gives that shot.  Returns the keys pinned."""
        shots = self.sorted_shots()
        idx = next(i for i, s in enumerate(shots) if s.shot_id == shot.shot_id)
        pins = []
        if abs(tail_delta) > _EPS and idx + 1 < len(shots):
            if abs(shots[idx + 1].start - shot.end) < _KEY_TOL:
                pins.append((float(shot.end), shot.shot_id, "end"))
        if abs(head_delta) > _EPS and idx > 0:
            prev = shots[idx - 1]
            if abs(prev.end - shot.start) < _KEY_TOL:
                pins.append((float(shot.start), prev.shot_id, "end"))
        if not pins:
            return 0
        content = self._content_objects()
        return sum(
            self._pin_bound_poses(
                content, (pin[0] - _KEY_TOL, pin[0] + _KEY_TOL), [pin], crossing=True
            )
            for pin in pins
        )

    def _hold_edge_neighbours(
        self, shot, new_start: float, new_end: float, grow_from_new: bool = False
    ) -> Dict[str, Any]:
        """Before a plain edge edit of *shot* to ``[new_start, new_end]``
        ripples its neighbours: what each side plays across every moved edge,
        pinned; and the ripple of each edge a FLUSH neighbour shrinks onto
        (``{"start": apply, "end": apply}``, each a callable running it; an
        edge left out takes the plain ripple).

        On each moved edge, every content curve is keyed
        (:meth:`_pin_bound_poses`, keyless crossings included) where *shot*
        keeps playing to -- its new bound for a shrink; for a grow its old
        one, or the new one when the ripple runs from there
        (*grow_from_new*, :meth:`fit_shot_to_content`: the keys the shot grew
        over stay with it) -- and on the neighbour's facing bound, each
        claimed for that bound.  A curve crossing the edge keyless ran from a key that stays
        to one the ripple moves, and both shots played it re-timed (BACKLOG
        2026-10-11: SOL "C02_2" 0.096 off over 55 frames after a flush
        shrink; across a gap, a ramp no member of the shot keys -- the gap
        hold never steps a shot's start pin -- played the next shot 1.2 off
        after a trim, "A02" 1.5 in the Blender pull).  A flush GROW is the
        split's to pin (:meth:`_reconcile_boundaries`).

        A flush shrink closes the seam again on the new bound: flush where
        *shot*'s pose there and the neighbour's on the old seam agree, a
        frame short where they differ -- the closing rule
        (:meth:`_close_flush_seams`, maintainer 2026-10-11), so there the
        neighbour moves one frame less than the edge.  The old seam's sample
        is the neighbour's from here: its claims are re-named for it, and the
        neighbour's window reaches over the frames *shot* gave up, so it
        rides."""
        shots = self.sorted_shots()
        idx = next(i for i, s in enumerate(shots) if s.shot_id == shot.shot_id)
        # (edge, before, after, where before closes, where after opens, flush)
        sides = []
        if idx + 1 < len(shots) and abs(new_end - shot.end) > _EPS:
            nxt = shots[idx + 1]
            flush = abs(nxt.start - shot.end) < _KEY_TOL
            close = new_end if grow_from_new else min(new_end, shot.end)
            if not (flush and new_end > shot.end):
                sides.append(("end", shot, nxt, close, nxt.start, flush))
        if idx > 0 and abs(new_start - shot.start) > _EPS:
            prev = shots[idx - 1]
            flush = abs(prev.end - shot.start) < _KEY_TOL
            opening = new_start if grow_from_new else max(new_start, shot.start)
            if not (flush and new_start < shot.start):
                sides.append(("start", prev, shot, prev.end, opening, flush))
        plans: Dict[str, Any] = {}
        content = self._content_objects() if sides else []
        for edge, before, after, close, opening, flush in sides:
            # Flush, the old seam's sample changes sides: *shot*'s claim on it
            # is the neighbour's from here.
            if flush and edge == "end":
                self._rename_seam_claims(
                    opening, (shot.shot_id, "end"), (after.shot_id, "start")
                )
            elif flush:
                self._rename_seam_claims(
                    close, (shot.shot_id, "start"), (before.shot_id, "end")
                )
            for pin in (
                (close, before.shot_id, "end"),
                (opening, after.shot_id, "start"),
            ):
                self._pin_bound_poses(
                    content,
                    (pin[0] - _KEY_TOL, pin[0] + _KEY_TOL),
                    [pin],
                    crossing=True,
                )
            if not flush:
                continue  # the plain ripple keeps the gap: nothing closes

            def layout(floors, edge=edge, after=after, close=close, opening=opening):
                floor = floors.get(after.shot_id, 0.0)
                if edge == "end":  # what follows comes back onto the new end
                    return ShotPlanner.plan_ripple_downstream(
                        self.store,
                        shot.shot_id,
                        close,
                        min(0.0, close - opening + floor),
                        carry_gap=True,
                    )
                return ShotPlanner.plan_ripple_upstream(  # what precedes comes on
                    self.store,
                    shot.shot_id,
                    opening,
                    max(0.0, opening - close - floor),
                    carry_gap=True,
                )

            pair = (before.shot_id, after.shot_id)
            plan, apart = self._close_flush_seams(
                layout,
                seams=lambda _plan, pair=pair: [pair],
                poses=lambda _p, _n, ab=(close, opening): [(ab[0], ab[1], None)],
            )

            # Landed flush, the seam's one sample is the shot before's (the
            # fencepost), whichever pin survived the merge.
            landing = (
                None
                if after.shot_id in apart
                else (close if edge == "end" else opening)
            )

            def apply(plan=plan, pair=pair, landing=landing):
                self._apply_plan(plan)
                if landing is not None:
                    self._rename_seam_claims(
                        landing, (pair[1], "start"), (pair[0], "end")
                    )

            plans[edge] = apply
        return plans

    def _rename_seam_claims(
        self, frame: float, old: tuple, new: tuple, stays: Optional[bool] = None
    ) -> int:
        """Re-name every claim on *frame* made for ``old = (shot_id, edge)`` as
        ``new``'s, keeping whether it stays unless *stays* says: a seam sample
        changing sides (:meth:`_hold_edge_neighbours`), or a bound's sample
        that is the content's pose now (:meth:`add_shot_space`).  Returns the
        claims re-named."""
        renamed = 0
        for crv in self.ledger.keyed_curves():
            for t, owner, edge in self.ledger.key_records(crv):
                if abs(t - frame) > _KEY_TOL or (owner, edge) != tuple(old):
                    continue
                keep = self.ledger.key_stays(crv, t) if stays is None else stays
                self.ledger.release_key(crv, t)
                self.ledger.record_key(crv, t, new[0], new[1], stays=keep)
                renamed += 1
        return renamed

    def resize_shot_bounds(
        self,
        shot_id: int,
        new_start: float,
        new_end: float,
        _enforce: bool = True,
        clamp: bool = True,
    ) -> None:
        """Move a shot's boundaries to ``[new_start, new_end]`` WITHOUT
        touching its keyframes, rippling neighbours by the edge deltas.

        The counterpart to :meth:`resize_shot`: same envelope bookkeeping,
        but the shot's own content stays exactly where the animator put it.
        Dragging a shot edge means "this shot covers a different span",
        which is a far more common intent than retiming everything inside
        it — retiming is the Shift-modified gesture.

        Because content is left alone, a shrink can leave keys outside the
        new bounds.  They are not deleted; they simply stop being counted
        as this shot's content until a boundary covers them again.

        EVERY edge move ripples the neighbours on that side by the same
        delta, so a gap keeps its width unless it is the thing being
        dragged: growing pushes them away, shrinking pulls them in behind
        the bound.  A shrink used to leave them where they were, which
        silently widened the adjacent gap on every resize — the one place
        a gap changed width without anyone asking it to.

        The ripple runs BEFORE the pivot's bounds are written, and that
        order is load-bearing in BOTH directions: a neighbour's move window
        is bounded by the pivot's boundary, so while that boundary is still
        the OLD one the window cannot reach the keys a shrink is about to
        strand.  Write the new bounds first and the window widens over them:
        measured on a two-shot scene sharing one curve, shrinking B's head
        from 60 to 80 swept B's own stranded key at 70 along with the
        upstream ripple, to 90.  Landing positions are unaffected — the
        neighbour ends one gap-width from the pivot's NEW bound either way,
        so a ripple can never run it onto the pivot.  Earlier still, this
        shot's own claimed bound samples are resolved against the bounds it
        is about to have (:meth:`_reconcile_pending_bounds`): the ripple lands
        the neighbour on exactly the frames a shrink gave up, and a sample
        still sitting there afterwards is stranded mid-content.

        Everything beyond the moved bound moves WHOLE: a neighbour's window
        is its envelope (its span and the gap after it), so its head keys and
        the lead-out hanging in the gap before a head grow ride with it, and
        a grow never claims what it covers.  The user's rule (2026-09-06):
        "we should only be modifying the current shot, everything else
        ripples unless ctrl is held" -- keeping the keys a bound is dragged
        over is Ctrl's gesture (the bound-only edge drag, which moves nothing
        else).  Measured on the production assembly with a claim in this
        path: Step 7's start dragged -30 moved Step 6 -30 but left its
        highlight lead-out (2131.2 .. 2142) where it was, inside Step 7; the
        end dragged +30 moved Step 8 +30 but left its first two keys behind.

        A shrink STOPS at the shot's content (*clamp*, the default): the bound
        never passes an animator's key of a curve that moves in the shot --
        the same rule :meth:`fit_shot_to_content` trims by, read from the same
        scan (:meth:`_key_extent`).  Keys a bound passed used to be stranded
        where the neighbour's ripple then landed, interleaving two shots'
        animation; cutting content off to the neighbour is Ctrl's gesture,
        which moves nothing else.  The one key a shrink may pass is a
        redundant, unclaimed key sitting ON the bound (a disowned pin), which
        is cut as the bound goes by.

        Parameters:
            shot_id: ID of the shot to resize.
            new_start: Desired start frame.
            new_end: Desired end frame.
            _enforce: If True (default), call :meth:`_enforce_gap_holds`
                after the operation.
            clamp: Hold a shrinking bound at the shot's content (see above).
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            raise ValueError(f"No shot with id {shot_id}")

        new_start = self.store.snap(new_start)
        new_end = self.store.snap(new_end)
        if new_end < new_start:
            new_start, new_end = new_end, new_start
        old_start, old_end = shot.start, shot.end
        on_bound: list = []
        if clamp and (new_start > old_start + _EPS or new_end < old_end - _EPS):
            first, last, _o1, _o2, on_bound = self._key_extent(shot, False)
            # Snapped OUTWARD, so the clamp never stops inside fractional
            # content (see fit_shot_to_content).
            if new_start > old_start + _EPS and first is not None:
                new_start = self.store.snap(min(new_start, first), "down")
            if new_end < old_end - _EPS and last is not None:
                new_end = self.store.snap(max(new_end, last), "up")
        if abs(new_start - old_start) < _EPS and abs(new_end - old_end) < _EPS:
            return

        tail_delta = new_end - old_end
        head_delta = new_start - old_start

        # A redundant key ON a bound moving inward goes first, then the
        # shot's own samples follow the SHRINKING bounds BEFORE the
        # neighbours ripple onto the frames it is giving up
        # (:meth:`_reconcile_pending_bounds`).  A growing edge waits for the
        # reconcile after the ripple: with carry_gap the frames it uncovers
        # are still the gap's until the ripple has moved that content away.
        self._cut_passed_bound_keys(on_bound, old_start, old_end, new_start, new_end)
        # What each side plays across a moved edge is pinned first, and a
        # shrinking edge a neighbour is FLUSH with closes that seam again on
        # the new bound, under the closing rule -- before the pending
        # reconcile, which would move the old seam's sample with this shot's
        # bound when it is the neighbour's now.
        closing = self._hold_edge_neighbours(shot, new_start, new_end)
        self._reconcile_pending_bounds(
            shot_id,
            new_start if head_delta > _EPS else old_start,
            new_end if tail_delta < -_EPS else old_end,
        )

        # Ripple BEFORE the pivot's bounds are written, in both directions
        # (see the docstring).
        if "end" in closing:
            closing["end"]()
        elif abs(tail_delta) > _EPS:
            self.ripple_downstream(shot_id, old_end, tail_delta)
        if "start" in closing:
            closing["start"]()
        elif abs(head_delta) > _EPS:
            self.ripple_upstream(shot_id, old_start, head_delta)

        shot.start = new_start
        shot.end = new_end

        if _enforce:
            # reconcile, not just enforce: this moved a shot BOUND, which
            # is exactly when a boundary sample the system created has to
            # follow it or be cleaned up.
            self.reconcile_system_edits()
        self.store.mark_dirty()

    def set_shot_edge(
        self,
        shot_id: int,
        new_start: Optional[float] = None,
        new_end: Optional[float] = None,
        scale: bool = False,
        push: bool = False,
    ) -> bool:
        """Move a shot's edge(s) with NOTHING rippling: the bound-only edit.

        Behind every edge gesture that moves no other shot -- Ctrl and Shift
        on a gap edge, Ctrl on the active shot's own edge, the inner edge of
        a gap-body drag.  The value is snapped through the store and clamped
        at the shot's opposite edge (zero duration is the floor): an edge
        dragged past the far one would store inverted bounds and hand
        :meth:`scale_shot_keys` an inverted range.

        ``scale=False`` moves the bound and leaves every key where it is, so
        the frames that change hands take their keys with them: an object
        moving in frames the shot gained becomes its member, and one whose
        motion there was its only motion in the shot that gave them up stops
        being that shot's (:meth:`_hand_over_members`).  ``scale=True``
        (Shift) retimes the shot's keys into the new span instead, and
        membership stands.

        Without *push* an edge stops at the neighbour on its side: the shot
        grows into the gap and no further, so at a gap of zero it cannot grow
        at all.  With *push* (the Ctrl drag of a shot's own edge, 2026-10-06:
        neither modifier "allows for me to drag to encompass keys in the
        adjacent shot so they become members of the current shot") the edge
        may cross the neighbour's facing bound, which gives way -- the
        neighbour shrinks, keeping at least one frame, and the keys it gave
        up are this shot's.  A neighbour FLUSH with the edge follows a shrink
        too, so the seam moves and the keys uncovered go to it rather than
        into a gap that was never there.  A retime never pushes: it writes
        keys across its new span, and the neighbour's would be among them.

        Callers reconcile the system's samples afterwards
        (:meth:`reconcile_system_edits`): the gap-body drag edits two edges
        before one reconcile.

        Parameters:
            shot_id: The shot whose edge moves.
            new_start: The start's new frame (``None``: it stays).
            new_end: The end's new frame (``None``: it stays).
            scale: Retime the shot's keys into the new span (Shift).
            push: Let the edge cross the neighbour's facing bound (Ctrl).

        Returns:
            ``True`` when any bound changed.

        Raises:
            ValueError: If *shot_id* does not exist.
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            raise ValueError(f"No shot with id {shot_id}")
        snap = self.store.snap
        push = push and not scale
        old_s, old_e = shot.start, shot.end
        ns = old_s if new_start is None else snap(new_start)
        ne = old_e if new_end is None else snap(new_end)
        if ns > ne:
            if new_start is not None and new_end is not None:
                # An inverted range, normalised as resize_shot_bounds does.
                ns, ne = ne, ns
            elif new_start is not None:
                ns = ne  # one edge past the other: zero duration is the floor
            else:
                ne = ns

        ordered = self.sorted_shots()
        idx = next(i for i, s in enumerate(ordered) if s.shot_id == shot_id)
        prev = ordered[idx - 1] if new_start is not None and idx > 0 else None
        nxt = (
            ordered[idx + 1] if new_end is not None and idx + 1 < len(ordered) else None
        )
        wanted_s = ns
        if prev is not None:
            ns = max(ns, prev.end if not push else min(prev.start + 1.0, prev.end))
        if nxt is not None:
            ne = min(ne, nxt.start if not push else max(nxt.end - 1.0, nxt.start))
        if ns > ne:
            # Both edges given, the span asked for lying wholly past a
            # neighbour: the clamp that held one edge carried it past the
            # other.  Zero duration at that neighbour's bound is the floor.
            if ns > wanted_s:
                ne = ns
            else:
                ns = ne
        prev_end = nxt_start = None  # the neighbours' new facing bounds
        if push and prev is not None:
            if ns < prev.end - _EPS or abs(prev.end - old_s) < _EPS:
                prev_end = ns
        if push and nxt is not None:
            if ne > nxt.start + _EPS or abs(nxt.start - old_e) < _EPS:
                nxt_start = ne
        if not scale:
            # An edge closing a GAP onto the neighbour's facing bound stops a
            # frame short where the two poses differ (the closing rule).  A
            # push past that bound is not a closing: the frames it takes are
            # the neighbour's content, one pose each.
            ns, ne = self._closing_edges(
                shot,
                prev if prev_end is None else None,
                nxt if nxt_start is None else None,
                ns,
                ne,
            )
        else:
            # A retime closing a gap that holds the animator's keys stops a
            # frame short too, and the keys are retimed into that frame
            # before the shot's own are (maintainer 2026-10-12, "same rule"):
            # left there, the retimed shot was written across them.
            ns, ne = self._closing_edges(shot, prev, nxt, ns, ne, retime=True)
        if abs(ns - old_s) < _EPS and abs(ne - old_e) < _EPS:
            return False

        if scale:
            if prev is not None and ns < old_s - _EPS:
                self._retime_closing_gap(prev, shot, prev.end, ns)
            if nxt is not None and ne > old_e + _EPS:
                self._retime_closing_gap(shot, nxt, ne, nxt.start)
            self.scale_shot_keys(old_s, old_e, ns, ne)
        # (giver, receiver, lo, hi): the frames that changed hands, the
        # neighbour's side ``None`` where they came from or went to a gap.
        before = prev if prev_end is not None else None
        after = nxt if nxt_start is not None else None
        handed = []
        if ns < old_s - _EPS:
            handed.append((before, shot, ns, old_s))
        elif ns > old_s + _EPS:
            handed.append((shot, before, old_s, ns))
        if ne > old_e + _EPS:
            handed.append((after, shot, old_e, ne))
        elif ne < old_e - _EPS:
            handed.append((shot, after, ne, old_e))
        shot.start, shot.end = ns, ne
        if prev_end is not None:
            prev.end = prev_end
        if nxt_start is not None:
            nxt.start = nxt_start
        if not scale:
            self._hand_over_members(handed)
        self.store.mark_dirty()
        return True

    def _closing_edges(
        self, shot, prev, nxt, ns: float, ne: float, retime: bool = False
    ):
        """``(ns, ne)`` for an edge edit of *shot*, each a frame short of the
        neighbour (*prev* / *nxt*, ``None`` where it moves or is absent) it
        would close a gap onto with a different pose
        (:meth:`_close_flush_seams`).  The edge's own new bounds are the
        plan; the poses are read where each shot ends and starts now.
        *retime* (a Shift edge, which writes its keys across the new span):
        the poses are not asked, only whether the gap holds the animator's
        keys (:meth:`_seams_hold_keys`; maintainer 2026-10-12, "same rule")."""
        closes_lo = (
            prev is not None
            and abs(ns - prev.end) < _EPS
            and shot.start > prev.end + _EPS
        )
        closes_hi = (
            nxt is not None
            and abs(ne - nxt.start) < _EPS
            and shot.end < nxt.start - _EPS
        )
        if not (closes_lo or closes_hi):
            return ns, ne

        def layout(floors: Dict[int, float]) -> Dict[int, Tuple[float, float]]:
            lo = ns + (floors.get(shot.shot_id, 0.0) if closes_lo else 0.0)
            hi = ne - (floors.get(nxt.shot_id, 0.0) if closes_hi else 0.0)
            return {shot.shot_id: (lo, max(lo, hi))}

        if retime:
            plan, _apart = self._close_flush_seams(
                layout, poses=lambda _p, _n: [], keep=self._seams_hold_keys
            )
        else:
            plan, _apart = self._close_flush_seams(layout)
        return plan[shot.shot_id]

    def _hand_over_members(self, handed) -> None:
        """Move membership with the frames a bound handed over.

        *handed* is ``[(giver, receiver, lo, hi), ...]``, either shot ``None``
        for a gap.  An object moving inside a span joins the receiver;
        the giver re-decides it from what is left of its own content
        (:meth:`_recompute_shot_objects`), so it stays wherever it still
        moves.
        """
        if not self._scene_available():
            return
        for giver, receiver, lo, hi in handed:
            if hi - lo < _EPS or (giver is None and receiver is None):
                continue
            # Evaluated: the span usually holds ONE key of what moves through
            # it -- the one it moves to or from, the motion running on to the
            # cut -- which a keys-only read never calls motion.
            movers = self._find_keyed_transforms(lo, hi, evaluated=True)
            if not movers:
                continue
            if receiver is not None:
                _listed, unlisted = self._member_names(receiver, movers)
                if unlisted:
                    self.store.update_shot(
                        receiver.shot_id,
                        objects=sorted(set(receiver.objects) | set(unlisted)),
                    )
            if giver is not None:
                listed, _unlisted = self._member_names(giver, movers)
                if listed:
                    self._recompute_shot_objects(giver.shot_id, only=listed)

    def set_shot_start(
        self, shot_id: int, new_start: float, ripple: bool = True
    ) -> None:
        """Move a shot to a new start time.

        Parameters:
            shot_id: ID of the shot to move.
            new_start: New start frame.
            ripple: If True, downstream shots shift by the same delta.
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            raise ValueError(f"No shot with id {shot_id}")

        direction = "downstream" if ripple else None
        new_start = self._clamp_slide_start(shot, new_start, direction)
        new_start = self._closing_slide_start(shot, new_start, direction)
        delta = new_start - shot.start
        if abs(delta) < _EPS:
            return

        old_end = shot.end
        slid = self._slid_bounds(shot, delta, direction)
        self._retime_closed_gap(shot, new_start, direction)

        with self._content_batch(), self.pending_bounds(slid):
            # Vacate the destination side first for forward moves (see
            # slide_shot) so the pivot's keys can't be double-shifted.
            # The pivot carries its own trailing gap; the ripple must not.
            if ripple and delta > 0:
                self.ripple_downstream(shot_id, old_end, delta, carry_gap=False)
                self._move_shot_content(shot, new_start)
            else:
                self._move_shot_content(shot, new_start)
                if ripple:
                    self.ripple_downstream(shot_id, old_end, delta, carry_gap=False)
        # reconcile, not just enforce: this moved a shot BOUND, which is exactly
        # when a boundary sample the system created has to follow it or be
        # cleaned up.
        self.reconcile_system_edits()
        self.store.mark_dirty()

    # ---- topology -----------------------------------------------------------

    def move_shot_to_position(self, shot_id: int, target_pos: int) -> None:
        """Move a shot to a new 1-based position in the timeline order.

        Other shots shift to accommodate.  Keyframes move with their
        shots.  Durations are preserved; gaps use the store's current
        gap setting (locked gaps are honoured).

        Parameters:
            shot_id:    The shot to relocate.
            target_pos: Desired 1-based position (clamped to valid range).

        Raises:
            ValueError: If *shot_id* does not exist.
        """
        if self.shot_by_id(shot_id) is None:
            raise ValueError(f"No shot with id {shot_id}")

        # Routed through the planner rather than a local park/land loop.
        # A hand-rolled version moved each shot's keys over the window
        # ``[shot.start, shot.end]``, which loses anything sitting in the
        # trailing gap (fade tails) and re-derives the collision ordering
        # the planner already solves.  The planner also honours locked gaps
        # by ADJACENCY, so a lock only survives between shots that stay
        # neighbours.
        plan = ShotPlanner.plan_reorder(self.store, shot_id, target_pos, self.store.gap)
        if not plan.sequence and not plan.parked:
            return
        self._apply_plan(plan)
        self._enforce_gap_holds()
        self.store.mark_dirty()

    def new_shot(
        self,
        after_shot_id: Optional[int] = None,
        at_position: Optional[int] = None,
    ) -> ShotBlock:
        """Make a new, empty shot -- what every panel's New Shot button makes.

        Named ``Shot_<n>`` (the next number the store accepts), as long as the
        store's ``initial_shot_length`` (the Shots window's Initial Length),
        and placed as :meth:`insert_shot` places it, one store gap from its
        neighbour.  One definition for the Shot Sequencer's and the Shots
        window's buttons, so the two cannot drift.

        Parameters:
            after_shot_id: Insert directly after this shot.
            at_position: 1-based slot to occupy instead (1 = first).  With
                neither, the shot is appended after the last one.

        Returns:
            The new :class:`ShotBlock`.
        """
        store = self.store
        name = store.unique_name("Shot", first=len(self.sorted_shots()) + 1)
        duration = float(store.initial_shot_length or store.DEFAULT_INITIAL_SHOT_LENGTH)
        return self.insert_shot(
            name=name,
            duration=duration,
            after_shot_id=after_shot_id,
            at_position=at_position,
        )

    def insert_placement(
        self,
        index: int,
        duration: float,
        gap: Optional[float] = None,
        line: Optional[List[tuple]] = None,
        start: Optional[float] = None,
    ) -> Tuple[float, float, float, Tuple[int, ...]]:
        """Where a shot inserted at *index* lands, and which rows it moves
        how far -- the one placement rule of :meth:`insert_shot`, read before
        anything moves (``ShotManifest.plan_insertions`` plans with it).

        The new shot opens *gap* after the shot before it; before every shot
        it takes the timeline's head frame; appended after the last one it
        also clears that shot's trailing content (fade tails past its end,
        trailing audio: :meth:`_trailing_content_extent`, the +INF-envelope
        content every ripple elsewhere protects).  The rows after it move by
        the push, the new shot's length plus *gap*: by position -- every row
        from *index* on -- when the shot before it ends at or before the next
        one's start, so a zero-length shot on the next one's start frame
        stays before the insert; by frame -- every row starting at or after
        the next shot's start -- when the two overlap, so overlapping shots
        sharing a start move together, each with its keys.

        A row may stand for something that is no shot yet when the insert is
        made (a range a plan will give a shot): it moves with the rows after
        the insert, and a new shot right after it starts *gap* after its end,
        but the shots alone decide the rest -- the head, the last shot and
        its trailing content, the rule the shots move by.

        Parameters:
            index: The 0-based position the new shot takes; ``len(line)``
                appends it.
            duration: Its length in frames.
            gap: Frames after the row before it (default: the store's).
            line: The timeline to insert into, as ``(start, end, shot)`` rows
                in order -- default the store's shots -- or ``(start, end,
                shot, is_shot)``, *is_shot* ``False`` for a row that is no
                shot yet.  A row's bounds may be where a planned ripple moves
                it: its shot (``None`` for one not built yet) is read only for
                the content trailing its end, which moves with it.
            start: Where the new shot starts instead, for a caller that knows
                of content the store does not hold (a shot a plan gives a
                range); the rows move all the same.

        Returns:
            ``(start, end, push, moved)``: the new shot's bounds, how far the
            rows it moves move (``0.0`` when nothing follows it), and the
            indices of those rows in *line*.
        """
        store = self.store
        gap = store.gap if gap is None else gap
        if line is None:
            line = [(s.start, s.end, s) for s in self.sorted_shots()]
        rows = [(r[0], r[1], r[2], r[3] if len(r) > 3 else True) for r in line]
        index = max(0, min(int(index), len(rows)))
        shots = [i for i, row in enumerate(rows) if row[3]]
        prev = max((i for i in shots if i < index), default=None)
        nxt = min((i for i in shots if i >= index), default=None)
        if start is not None:
            start = store.snap(start)
        elif index and not rows[index - 1][3]:  # right after a row no shot yet
            start = store.snap(rows[index - 1][1] + gap)
        elif prev is None:
            # Before every shot: keep the timeline's head frame and push the
            # whole sequence out of the way.
            start = rows[nxt][0] if nxt is not None else store.snap(1.0)
        elif nxt is None:
            _prev_start, prev_end, shot, _is = rows[prev]
            tail = prev_end
            if shot is not None:  # its content, where the row's ripple takes it
                moved = prev_end - shot.end  # 0.0 for the shot where it is
                tail = max(prev_end, self._trailing_content_extent(shot) + moved)
            start = store.snap(tail + gap)
        else:
            start = store.snap(rows[prev][1] + gap)
        end = store.snap(start + duration)
        push = (end - start) + gap
        if index >= len(rows) or abs(push) < _EPS:
            return start, end, (0.0 if index >= len(rows) else push), ()
        if nxt is None or prev is None or rows[prev][1] <= rows[nxt][0]:
            moved = tuple(range(index, len(rows)))
        else:  # the frame rule: what ``plan_ripple_downstream`` moves by frame
            frame = rows[nxt][0]
            moved = tuple(i for i, row in enumerate(rows) if not row[0] < frame)
        return start, end, push, moved

    def insert_shot(
        self,
        name: str,
        duration: float,
        after_shot_id: Optional[int] = None,
        at_position: Optional[int] = None,
        gap: Optional[float] = None,
        objects: Optional[List[str]] = None,
        description: str = "",
        start: Optional[float] = None,
    ) -> ShotBlock:
        """Create a shot BETWEEN existing shots, pushing later ones downstream.

        Appending was the only way to add a shot, so making room in the
        middle meant hand-rippling every following shot.  This opens the
        space first -- the shots after the insertion point (and their
        keyframes and audio) move by ``duration + gap``
        (:meth:`insert_placement`) -- then defines the new shot in the hole.

        Parameters:
            name: Human-readable label.
            duration: Length of the new shot in frames.
            after_shot_id: Insert directly after this shot.  ``None`` with
                *at_position* unset appends at the end.
            at_position: 1-based slot the new shot should occupy, as an
                alternative to *after_shot_id* (1 = before every shot).
            gap: Frames between the preceding shot's content and the new
                shot (defaults to the store's gap).  Downstream shots ripple
                rigidly by ``duration + gap``, so the spacing between the
                new shot and its follower stays whatever the preceding↔
                follower gap was; at position 1 there is no preceding shot
                and the gap falls after the new shot instead.
            objects: Object names to seed the shot with.
            description: Optional description.
            start: Where the new shot starts instead of where the placement
                rule puts it -- for a caller that knows of content the store
                does not hold (``ShotManifest.insert_planned``: a shot a
                plan gives a typed range).  The shots after it still move by
                its length plus *gap*.

        Returns:
            The newly created :class:`ShotBlock`.

        Raises:
            ValueError: If *after_shot_id* does not exist, or the store refuses
                *name* (``ShotStore.name_error``) -- checked BEFORE anything
                moves, so a refused name leaves every shot and key in place.
        """
        self._refuse_name(name)
        gap = self.store.gap if gap is None else gap
        shots = self.sorted_shots()

        if after_shot_id is not None:
            idx = next(
                (i for i, s in enumerate(shots) if s.shot_id == after_shot_id), None
            )
            if idx is None:
                raise ValueError(f"No shot with id {after_shot_id}")
            insert_idx = idx + 1
        elif at_position is not None:
            insert_idx = max(0, min(int(at_position) - 1, len(shots)))
        else:
            insert_idx = len(shots)

        start, new_end, push, moved = self.insert_placement(
            insert_idx, duration, gap=gap, start=start
        )

        # Open the hole before the shot exists, so the ripple can't pick up
        # the new shot as one of the shots it should move.  By position, the
        # shot before the insert is the pivot (at the head every shot is at or
        # after the first one's frame); by frame, no shot is exempt (pivot -1):
        # each moves the shots insert_placement says.
        if moved:
            by_position = moved == tuple(range(insert_idx, len(shots)))
            pivot = shots[insert_idx - 1].shot_id if by_position and insert_idx else -1
            plan = ShotPlanner.plan_ripple_downstream(
                self.store, pivot, shots[insert_idx].start, push
            )
            self._apply_plan(plan)

        block = self.define_shot(
            name=name,
            start=start,
            end=new_end,
            objects=objects if objects is not None else [],
            description=description,
        )
        self._enforce_gap_holds()
        self.store.mark_dirty()
        return block

    def _shot_envelope(self, shot_id: int) -> Optional[tuple]:
        """``(lo, hi, lo_open, hi_closed)`` — the key window a shot owns.

        The planner's fencepost rule, resolved for one shot, so every
        lifecycle operation here reads the same window the movers write.
        """
        shots = self.sorted_shots()
        idx = next((i for i, s in enumerate(shots) if s.shot_id == shot_id), None)
        if idx is None:
            return None
        return ShotPlanner.envelope_for(shots, idx)

    def delete_shot(
        self,
        shot_id: int,
        delete_contents: bool = True,
        close_gap: bool = True,
    ) -> Dict[str, Any]:
        """Remove a shot — by default with its keys, and closing up behind it.

        Removing only the RECORD leaves the shot's animation orphaned in the
        middle of the timeline and a hole where the shot was, which is almost
        never what "delete this shot" means.  The default therefore cuts the
        shot's own content (:meth:`_cut_shot_content`) and slides everything
        downstream back by the span the shot occupied — its own range plus
        the gap that followed it — so the next shot lands where this one
        started.

        Both halves are opt-out for the caller that really does want just the
        record gone (``delete_contents=False``) or the timeline left alone
        (``close_gap=False``).

        Parameters:
            shot_id: The shot to remove.
            delete_contents: Cut the keys the shot owns.
            close_gap: Ripple later shots upstream into the vacated span.

        Returns:
            ``{"curves_cut", "closed", "name"}`` — curves the cut reached,
            frames the timeline closed by, and the removed shot's name.

        Raises:
            ValueError: If *shot_id* does not exist.
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            raise ValueError(f"No shot with id {shot_id}")

        shots = self.sorted_shots()
        idx = next(i for i, s in enumerate(shots) if s.shot_id == shot_id)
        nxt = shots[idx + 1] if idx + 1 < len(shots) else None
        # The span the shot occupies is its range PLUS the gap after it: the
        # next shot slides into both, or the hole simply moves downstream.
        span_end = nxt.start if nxt is not None else shot.end
        vacated = max(0.0, span_end - shot.start)
        name = shot.name

        curves_cut = 0
        if delete_contents:
            lo, hi, lo_open, _hi_closed = ShotPlanner.envelope_for(shots, idx)
            self._hold_neighbour_poses(
                shot_id,
                # Every keyed object: what the closing re-times is any curve
                # crossing the span, the shot's members or not.
                self._content_objects(),
                # The cut's own window: a flush closing sample stays, the
                # next shot's opening pose (the DCC ``_cut_shot_content``).
                (
                    max(lo, shot.start - _CONTENT_REACH)
                    + (_KEY_TOL if lo_open else -_KEY_TOL),
                    min(hi, shot.end + _CONTENT_REACH) - _KEY_TOL,
                ),
                flush=True,
            )
            curves_cut = self._cut_shot_content(shot_id)

        # The shot's own bound samples go with it -- cut where provably
        # redundant, released otherwise -- BEFORE the gap closes.  Disowned
        # in place they sat exactly where the ripple landed the next shot:
        # measured 2026-09-06 (an empty inserted shot, a respace, a delete),
        # "Shot 3.3" played 33 frames differently on four curves because the
        # deleted shot's end pin now stood mid-ramp with a recomputed tangent.
        # The removal disowns what is left (ShotStore.remove_shot).
        self._reconcile_boundary_keys(bounds={shot_id: (None, None)})
        self.store.remove_shot(shot_id)

        closed = 0.0
        if close_gap and nxt is not None and vacated > _EPS:
            # Pivot -1: no shot is exempt, everything at or after the vacated
            # span comes back by its width.  The shot's record is already gone
            # so the plan cannot pick it up as one of the shots to move.
            # Landing flush on the shot before, the next shot keeps a frame
            # of its own where the two poses differ (maintainer 2026-10-11).
            plan, apart = self._close_flush_seams(
                lambda floors: ShotPlanner.plan_ripple_downstream(
                    self.store,
                    -1,
                    span_end,
                    -(vacated - floors.get(nxt.shot_id, 0.0)),
                )
            )
            if plan.sequence or plan.parked:
                self._apply_plan(plan)
                closed = vacated - apart.get(nxt.shot_id, 0.0)

        self.reconcile_system_edits()
        self.store.mark_dirty()
        return {"curves_cut": curves_cut, "closed": closed, "name": name}

    def merge_shots(self, shot_ids: List[int], name: Optional[str] = None) -> ShotBlock:
        """Fuse two or more shots into one spanning all of them.

        The earliest shot is kept and grown to the union range; the others are
        removed and their objects folded in.  Nothing MOVES — a merge is a
        statement about how the timeline is divided, not about where content
        sits — so any gap between the merged shots becomes ordinary empty
        space inside the result.  The holds that were guarding those gaps stop
        being seams and are released by :meth:`reconcile_system_edits`, which
        is exactly right: there is no longer a cut there.

        Parameters:
            shot_ids: Two or more ids.  Unknown ids are ignored; order does
                not matter (the earliest START wins).
            name: Name for the merged shot.  Defaults to the keeper's.

        Returns:
            The surviving :class:`ShotBlock`.

        Raises:
            ValueError: If fewer than two of *shot_ids* resolve to shots.
        """
        shots = [s for s in (self.shot_by_id(i) for i in shot_ids) if s is not None]
        if len(shots) < 2:
            raise ValueError("merge_shots needs at least two existing shots")
        shots.sort(key=lambda s: (s.start, s.shot_id))
        keeper = shots[0]

        new_start = min(s.start for s in shots)
        new_end = max(s.end for s in shots)
        objects: List[str] = []
        for s in shots:  # union, first-seen order, so track order is stable
            for obj in s.objects:
                if obj not in objects:
                    objects.append(obj)
        notes = [s.description for s in shots if s.description]

        with self.store.batch_update():
            for s in shots[1:]:
                # Inner bounds vanish with the merge; their pins go too where
                # provably redundant (see delete_shot).
                self._reconcile_boundary_keys(bounds={s.shot_id: (None, None)})
                self.store.remove_shot(s.shot_id)
            # The keeper's own end is inner now too, and is retired the same
            # way.  Left to the reconcile, its sample FOLLOWED the merged end
            # across every key the merge took in: measured 2026-10-05, a pin
            # on A's end carried 40 frames on reshaped 84 frames of the curve.
            if new_end > keeper.end + _EPS:
                self._reconcile_boundary_keys(
                    bounds={keeper.shot_id: (new_start, None)}
                )
            self.store.update_shot(
                keeper.shot_id,
                name=name or keeper.name,
                start=new_start,
                end=new_end,
                objects=objects,
                description=" / ".join(notes),
            )

        self.reconcile_system_edits()
        self.store.mark_dirty()
        return self.shot_by_id(keeper.shot_id)

    def split_shot(
        self,
        shot_id: int,
        at_frame: float,
        name: Optional[str] = None,
        gap: float = 0.0,
    ) -> ShotBlock:
        """Cut a shot in two at *at_frame*, leaving its content where it is.

        The head keeps the original record (name, description, id) and ends at
        the cut; the tail is a new shot from the cut to the original end.  The
        two are contiguous and therefore SHARE the sample on the cut frame,
        which is the same fencepost convention every other operation uses.

        With ``gap`` the tail (and everything after it) ripples downstream by
        that many frames, so the split lands as a real cut with room between
        the halves rather than an invisible division.

        Membership is recomputed per side from the keys actually there, so
        neither half claims objects that only animate in the other.

        Parameters:
            shot_id: The shot to split.
            at_frame: Frame to cut on.  Must lie strictly inside the shot.
            name: Name for the tail.  Defaults to a unique ``<name>_2``.
            gap: Frames to open between the halves (0 = contiguous).

        Returns:
            The newly created tail :class:`ShotBlock`.

        Raises:
            ValueError: If *shot_id* does not exist, *at_frame* is not
                strictly inside it (a cut on a bound divides nothing), or the
                store refuses the tail's *name* -- checked before the head is
                trimmed, so a refused name leaves the shot whole.
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            raise ValueError(f"No shot with id {shot_id}")
        at = self.store.snap(float(at_frame))
        if not (shot.start + _EPS < at < shot.end - _EPS):
            raise ValueError(
                f"Split frame {at:g} is not inside {shot.name} "
                f"[{shot.start:g}-{shot.end:g}]"
            )

        tail_end = shot.end
        tail_name = name or self.store.unique_name(f"{shot.name}_2")
        self._refuse_name(tail_name)
        return self._split_at(shot, at, tail_name, tail_end, gap)

    def _split_at(
        self, shot, at: float, tail_name: str, tail_end: float, gap
    ) -> ShotBlock:
        """The validated half of :meth:`split_shot`: trim, define, ripple."""
        shot_id = shot.shot_id
        self.store.update_shot(shot_id, end=at)
        # The tail INHERITS the shot's object list and is then narrowed to
        # what actually animates in it.  Seeding it empty instead makes
        # ``collect_object_segments`` fall back to a scene-wide probe for
        # keyed objects, which adopts objects this shot never claimed.
        # The tail belongs with the step its head is -- by the head's binding,
        # or its own link -- through a NON-binding link (``ShotStore.step_links``):
        # a step has one shot, but the export sections the tail with it.
        step = (shot.metadata or {}).get("step") or self.store.step_links().get(shot_id)
        tail = self.define_shot(
            name=tail_name,
            start=at,
            end=tail_end,
            objects=list(shot.objects),
            description=shot.description,
            metadata={self.store.STEP_LINK: step} if step else None,
        )
        self._recompute_shot_objects(shot_id)
        self._recompute_shot_objects(tail.shot_id)

        gap = float(gap)
        if abs(gap) > _EPS:
            plan = ShotPlanner.plan_ripple_downstream(self.store, -1, at, gap)
            if plan.sequence or plan.parked:
                self._apply_plan(plan)

        self.reconcile_system_edits()
        self.store.mark_dirty()
        return self.shot_by_id(tail.shot_id)

    # ---- padding / spacing ----------------------------------------------------

    def _leading_room(self, shot_id: int) -> float:
        """Empty frames between a shot's start and its first piece of content.

        Zero when the shot is empty — there is no room to reclaim from a shot
        that holds nothing, and treating its whole span as slack would let a
        pad silently resize it to a point.
        """
        sequences = self.collect_shot_sequences(shot_id)
        if not sequences:
            return 0.0
        shot = self.shot_by_id(shot_id)
        return max(0.0, min(s["start"] for s in sequences) - shot.start)

    def add_shot_space(
        self, shot_id: int, frames: float, edge: str = "leading"
    ) -> Tuple[float, float]:
        """Insert empty room at a shot's head and/or tail, rippling downstream.

        Both edges open room *forward in time* — the shot's start is an
        anchor, never something padding drags backwards:

        * ``"leading"`` — the start stays exactly where it is and everything
          from it onward shifts later by *frames*: this shot's own keys and
          audio, its end, and every downstream shot.  The new room lands at
          the head, in front of the content.
        * ``"trailing"`` — the end moves later by *frames* and the downstream
          shots follow.  The shot's own content stays put; the new room lands
          behind it.
        * ``"both"`` — each of the above, so the shot grows by ``2 * frames``
          and its content sits *frames* further along.

        Spacing between shots is preserved throughout, so the padding is
        genuinely new room rather than an existing gap being eaten.

        A negative *frames* removes that much room, which is how the same
        control does both directions.

        Parameters:
            shot_id: The shot to pad.
            frames: Frames of room to add (negative removes).
            edge: ``"leading"``, ``"trailing"`` or ``"both"``.

        Returns:
            ``(head_delta, tail_delta)`` — how far each bound actually moved.
            The head delta is always 0 for a leading pad: that is the point.

        Raises:
            ValueError: If *shot_id* does not exist.
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            raise ValueError(f"No shot with id {shot_id}")
        frames = float(frames)
        if abs(frames) < _EPS:
            return 0.0, 0.0

        head = frames if edge in ("leading", "both") else 0.0
        tail = frames if edge in ("trailing", "both") else 0.0
        if head == 0.0 and tail == 0.0:
            return 0.0, 0.0

        old_start, old_end = shot.start, shot.end
        if head < 0:
            # Removing head room pulls the content back toward the anchored
            # start, so it may only reclaim room that is actually EMPTY --
            # past that it would drag keys out through the head and into the
            # upstream gap, which is a delete dressed up as a pad.
            head = -min(-head, self._leading_room(shot_id))
            if abs(head) < _EPS and abs(tail) < _EPS:
                return 0.0, 0.0
        # A shot may not be padded into nothing; both ends push the tail out,
        # so the guard is on the end alone (removing room is the negative
        # -frames case).
        if self.store.snap(old_end + head + tail) <= old_start:
            return 0.0, 0.0

        with self._content_batch():
            if abs(head) > _EPS:
                # Slide the shot bodily downstream, then put the start back:
                # the content and every following shot end up *frames* later
                # while the head holds, which is exactly "empty room at the
                # front".  ``slide_shot`` owns the ordering that keeps the
                # pivot's keys out of a neighbour's not-yet-read envelope.
                self.slide_shot(
                    shot_id, old_start + head, direction="downstream", _enforce=False
                )
                # The start's claimed samples rode with the content and are
                # its opening pose now: left to follow the start back, they
                # stretched the shot's first segment (SOL harness: "A01_2"
                # KEYS_translateY 1.65 off over 16 frames).  Kept where they
                # landed (``stays``).
                self._rename_seam_claims(
                    shot.start, (shot_id, "start"), (shot_id, "start"), stays=True
                )
                shot.start = old_start
            if abs(tail) > _EPS:
                pre_tail_end = shot.end
                new_end = self.store.snap(pre_tail_end + tail)
                # Ripple by what the END ACTUALLY moved rather than by the
                # amount asked for -- the same value ``fit_shot_to_content``
                # passes.  The plan snaps its own destinations, so the two
                # agree today; deriving it from the bound is what keeps them
                # agreeing if either side's rounding ever changes.
                if tail > 0:
                    # Room AFTER the content: ripple first, from the end as
                    # it stands, then write it (resize_shot_bounds' order).
                    # Written first, a flush neighbour read as no neighbour:
                    # its envelope took the seam sample the shot ends on, and
                    # a keyless seam was never seen to split -- both shots
                    # played differently (2026-10-11: a 10-frame pad moved
                    # the shot's own frames 0.16 and the next 1.10).
                    self.ripple_downstream(
                        shot_id, pre_tail_end, new_end - pre_tail_end
                    )
                    shot.end = new_end
                else:
                    shot.end = new_end
                    self.ripple_downstream(
                        shot_id, pre_tail_end, shot.end - pre_tail_end
                    )

        head_delta = shot.start - old_start
        tail_delta = shot.end - old_end
        # Holds are deliberately left to the reconcile below rather than
        # enforced per step: the head slide is asked NOT to enforce so the
        # seams are read once, after the start has been put back.
        self.reconcile_system_edits()
        self.store.mark_dirty()
        return head_delta, tail_delta

    def _content_objects(self) -> list:
        """Every object a whole-shot move can reach: what the movers, the pin
        and the retime all act on.

        :meth:`_keyed_transform_times` is the sequencer's OWN definition of
        content (standard transform/visibility and the render-effect
        channels, so a marker attribute never makes an object look animated),
        and this reuses it rather than deciding again — a second rule that
        drifted would pin one set and retime another.  Without a scene
        (:meth:`_scene_available`) it is what the shots claim.

        Union with what the shots claim, which covers the one case the keyed
        walk excludes by design: a shot object animated only on a non-standard
        channel. Stale entries are harmless — the resolution downstream drops
        names nothing resolves.

        Deliberately NOT the shots' object lists: a node no shot claims still
        has its curve cut by the shots' bounds and carried by their envelopes.
        The lists used to be backfilled with everything keyed in a moving
        envelope so the movers would carry it -- which is how a baked rig's
        forty proxy joints became members of all twelve shots of a production
        assembly (531 flat-keyed memberships, each drawing a track).  Handing
        the movers this set keeps the guarantee and leaves membership a label.
        """
        claimed = {obj for shot in self.store.shots for obj in shot.objects}
        if not self._scene_available():
            return sorted(claimed)
        return sorted(claimed | set(self._keyed_transform_times()))

    def respace(
        self, gap: float = 0, start_frame: float = 1, respect_locks: bool = True
    ) -> None:
        """Redistribute all shots sequentially with uniform gaps.

        Each shot keeps its current duration but is repositioned so the
        first shot starts at *start_frame* and subsequent shots follow
        with *gap* frames between them.  Locked gaps preserve their
        current width instead of using the uniform *gap* value.
        Keyframes are moved with their shots when a scene is available.

        Delegates to :meth:`ShotPlanner.plan_respace` and :meth:`_apply_plan`
        so the full topology is resolved in memory before any scene write,
        eliminating envelope collisions between moved and not-yet-moved shots.

        Each gap's own content is RETIMED into the gap's new width rather than
        carried rigidly with the shot before it: a gap's width is the thing a
        respace changes, so content living in one has nowhere to be carried to
        (``retime_gaps``; see :meth:`_apply_plan`).

        Every gap is HELD before anything is pinned (:meth:`_enforce_gap_holds`
        first, a no-op where the holds already stand).  A shot's frames after
        its last key hold that key's pose -- the hold rule every other op
        applies -- and the respace pins every bound shape-preservingly, which
        on a tail the hold had not reached yet pinned the curve PLAYED ON
        toward the next shot's first key: the same scene played its tail on
        after a respace and held it after any other op (BACKLOG 2026-10-05;
        maintainer 2026-10-10: tails keep holding, an op on one shot never
        changes what another plays inside its own span).  Held first, the
        tail is a hold segment, which the pin declines, so it stays held at
        its held pose.  Pinning the end instead was measured and rejected:
        padded tails drift toward the next shot's first pose.

        Parameters:
            gap: Frames of gap between consecutive shots.
            start_frame: Timeline frame for the first shot.
            respect_locks: When False, spend *gap* on locked gaps too.  The
                locks are left set either way.
        """
        self._enforce_gap_holds()
        # A seam it closes to 0 stays a frame apart where its poses differ
        # (maintainer 2026-10-11): never refused, never re-timing a shot --
        # and where its gap holds keys, which a zero-width retime would leave
        # inside the next shot (:meth:`_seams_hold_keys`).
        plan, _apart = self._close_flush_seams(
            lambda floors: ShotPlanner.plan_respace(
                self.store,
                gap,
                start_frame,
                respect_locks=respect_locks,
                seam_gaps=floors,
            ),
            keep=self._seams_hold_keys,
        )
        self._apply_plan(plan, retime_gaps=True)
        self._enforce_gap_holds()

    def apply_gap(
        self,
        gap: float,
        scope: str = "all",
        shot_id: Optional[int] = None,
        respect_locks: bool = True,
    ) -> bool:
        """Re-space shots so the given *gap* separates them, per *scope*.

        Parameters:
            gap: Desired gap width in frames.
            scope: ``"all"`` respaces every shot from the first shot's
                current start.  ``"start"`` moves *shot_id* so it sits
                *gap* after its predecessor; ``"end"`` moves the
                successor so it sits *gap* after *shot_id*;
                ``"start_end"`` does both.
            shot_id: Anchor shot for the scoped modes (typically the
                active shot).  Ignored for ``"all"``.
            respect_locks: When False, a locked gap is re-spaced like any
                other.  Only ``"all"`` consults the lock table at all -- the
                scoped modes move one shot through ``move_shot``, whose
                ripple never asks -- so it changes nothing for those.

        Returns:
            ``True`` when any shot was repositioned.
        """
        sorted_s = self.sorted_shots()
        if not sorted_s:
            return False

        if scope == "all":
            self.respace(
                gap=gap, start_frame=sorted_s[0].start, respect_locks=respect_locks
            )
            return True

        if shot_id is None:
            return False
        idx = next((i for i, s in enumerate(sorted_s) if s.shot_id == shot_id), None)
        if idx is None:
            return False

        moved = False
        if scope in ("start", "start_end") and idx > 0:
            self.move_shot(shot_id, sorted_s[idx - 1].end + gap)
            moved = True
            # move_shot ripples neighbors — re-derive the ordering before
            # positioning the successor.
            sorted_s = self.sorted_shots()
            idx = next(
                (i for i, s in enumerate(sorted_s) if s.shot_id == shot_id),
                idx,
            )
        if scope in ("end", "start_end") and idx < len(sorted_s) - 1:
            self.move_shot(sorted_s[idx + 1].shot_id, sorted_s[idx].end + gap)
            moved = True
        return moved

    # ---- serialisation --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise shots and settings to a plain dict."""
        return self.store.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShotSequencer":
        """Restore from serialised data (a store of :attr:`STORE_CLASS`)."""
        return cls(store=cls.STORE_CLASS.from_dict(data))
