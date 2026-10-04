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
``collect_object_segments(shot_id)``  per-object anim segments in a shot
``_collect_audio_sequences(lo, hi)``  audio sequences in a range
``_key_extent(shot, probe, reach)``   a shot's key extent (see DCC docs)
``_cut_passed_bound_keys(...)``       cut redundant keys a moving bound passes
``_reconcile_boundary_keys(...)``     re-home the system's boundary samples
``_gap_hold_seams()`` / ``_release_gap_holds(s)`` / ``_apply_gap_holds(s)``
                                      the inter-shot gap holds
``_reconcile_boundaries(plan)``       fencepost reconcile around a plan write
``_move_content_keys(...)``           move keyed content inside an envelope
``_shift_audio(lo, hi, delta)``       shift audio inside a range
``_move_audio_sequence(seq, delta)``  move one audio sequence
``move_attribute_keys(...)``          move one object's (attribute's) keys
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

import contextlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from pythontk.core_utils.engines.shots.shot_apply import ShotApply
from pythontk.core_utils.engines.shots.shot_model import ShotBlock, ShotStore
from pythontk.core_utils.engines.shots.shot_plan import ShotPlanner

# Below this a frame delta is no move at all.
_EPS = 1e-6

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
        """Objects keyed inside ``[start, end]`` (``define_shot``'s default members)."""
        return []

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

    def _reconcile_boundaries(self, plan) -> Callable[[], None]:
        """Reconcile shared samples around a plan write; returns the finisher."""
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

    def __init__(
        self,
        shots: Optional[List[ShotBlock]] = None,
        store: Optional[ShotStore] = None,
    ):
        if store is not None:
            self.store = store
        else:
            self.store = self.STORE_CLASS(shots)

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
        dest = self.shot_by_id(dest_shot_id)
        if dest is None:
            raise ValueError(f"No shot with id {dest_shot_id}")
        if not sequences:
            return

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
        if head_groups:
            head_seqs = [s for grp in head_groups.values() for s in grp]
            for s in head_seqs:
                self._absorb_gap_overhang(s, dest.start)
            block_min = min(s["start"] for s in head_seqs)
            block_max = max(s["end"] for s in head_seqs)
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
                self.add_shot_space(dest_shot_id, room, edge="leading")
                # Everything from the destination's start onward travelled
                # with the pad -- its own content, the gap behind it, the
                # downstream shots -- and so did the tail groups' keys.
                for sid, grp in groups.items():
                    if sid in head_groups:
                        continue
                    if min(s["start"] for s in grp) >= dest.start - _EPS:
                        for s in grp:
                            s["start"] += room
                            s["end"] += room
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
            if source_id is not None:
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
                dest.end = self.store.snap(needed_end, "up")
                self.ripple_downstream(dest_shot_id, old_end, room)
                for seq, source_id, _target in placements:
                    if source_id in travelled:
                        seq["start"] += room
                        seq["end"] += room

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
            for seq, _source_id, target in placements:
                self._move_sequence(seq, target)

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
            if abs(tail_delta) > _EPS:
                self.ripple_downstream(
                    shot_id, new_end if tail_delta > 0 else old_end, tail_delta
                )
            if abs(head_delta) > _EPS:
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
        delta = new_start - old_start
        if abs(delta) < _EPS:
            return
        self._slide(shot, old_start, old_end, new_start, delta, direction, _enforce)

    def _slide(self, shot, old_start, old_end, new_start, delta, direction, enforce):
        """The ordered ripple + content move behind :meth:`slide_shot`."""
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

        if enforce:
            self._enforce_gap_holds()
        self.store.mark_dirty()

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

        Called automatically after every timeline-modifying operation, and
        idempotent: a second run finds every seam already held and every
        claim still wanted, so it writes nothing.
        """
        if not self._scene_available():
            return
        seams = self._gap_hold_seams()
        self._release_gap_holds(seams)
        self._apply_gap_holds(seams)

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
        return {
            "keys_moved": moved,
            "keys_removed": removed,
            "holds": self.ledger.step_count,
        }

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
        shot.end = self.store.snap(new_end)
        self.ripple_downstream(shot_id, old_end, delta)
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
        # see :meth:`resize_shot`.
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
        self._reconcile_pending_bounds(
            shot_id,
            new_start if head_delta > _EPS else old_start,
            new_end if tail_delta < -_EPS else old_end,
        )

        # Ripple BEFORE the pivot's bounds are written, in both directions
        # (see the docstring).
        if abs(tail_delta) > _EPS:
            self.ripple_downstream(shot_id, old_end, tail_delta)
        if abs(head_delta) > _EPS:
            self.ripple_upstream(shot_id, old_start, head_delta)

        shot.start = new_start
        shot.end = new_end

        if _enforce:
            # reconcile, not just enforce: this moved a shot BOUND, which
            # is exactly when a boundary sample the system created has to
            # follow it or be cleaned up.
            self.reconcile_system_edits()
        self.store.mark_dirty()

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

        new_start = self._clamp_slide_start(
            shot, new_start, "downstream" if ripple else None
        )
        delta = new_start - shot.start
        if abs(delta) < _EPS:
            return

        old_end = shot.end

        with self._content_batch():
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

    def insert_shot(
        self,
        name: str,
        duration: float,
        after_shot_id: Optional[int] = None,
        at_position: Optional[int] = None,
        gap: Optional[float] = None,
        objects: Optional[List[str]] = None,
        description: str = "",
    ) -> ShotBlock:
        """Create a shot BETWEEN existing shots, pushing later ones downstream.

        Appending was the only way to add a shot, so making room in the
        middle meant hand-rippling every following shot.  This opens the
        space first — every shot at or after the insertion point (and its
        keyframes and audio) moves by ``duration + gap`` — then defines the
        new shot in the hole.

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

        if not shots:
            start = self.store.snap(1.0)
        elif insert_idx == 0:
            # Before everything: keep the timeline's existing head frame and
            # push the whole sequence out of the way.
            start = shots[0].start
        elif insert_idx == len(shots):
            # Appending after the LAST shot: its trailing envelope content
            # (fade tails past .end, trailing audio — the +INF-envelope
            # content every ripple elsewhere protects) must not be built
            # over.
            prev = shots[-1]
            start = self.store.snap(
                max(prev.end, self._trailing_content_extent(prev)) + gap
            )
        else:
            start = self.store.snap(shots[insert_idx - 1].end + gap)

        new_end = self.store.snap(start + duration)

        # Open the hole before the shot exists, so the ripple can't pick up
        # the new shot as one of the shots it should move.  A pivot id no
        # shot owns means "shift everything at or after the frame".
        if insert_idx < len(shots):
            delta = (new_end - start) + gap
            if abs(delta) > _EPS:
                plan = ShotPlanner.plan_ripple_downstream(
                    self.store, -1, shots[insert_idx].start, delta
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

        curves_cut = self._cut_shot_content(shot_id) if delete_contents else 0

        # The shot's own bound samples go with it -- cut where provably
        # redundant, released otherwise -- BEFORE the gap closes.  Disowned
        # in place they sat exactly where the ripple landed the next shot:
        # measured 2026-09-06 (an empty inserted shot, a respace, a delete),
        # "Shot 3.3" played 33 frames differently on four curves because the
        # deleted shot's end pin now stood mid-ramp with a recomputed tangent.
        self._reconcile_boundary_keys(bounds={shot_id: (None, None)})
        self.ledger.disown_shot(shot_id)
        self.store.remove_shot(shot_id)

        closed = 0.0
        if close_gap and nxt is not None and vacated > _EPS:
            # Pivot -1: no shot is exempt, everything at or after the vacated
            # span comes back by its width.  The shot's record is already gone
            # so the plan cannot pick it up as one of the shots to move.
            plan = ShotPlanner.plan_ripple_downstream(
                self.store, -1, span_end, -vacated
            )
            if plan.sequence or plan.parked:
                self._apply_plan(plan)
                closed = vacated

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
                self.ledger.disown_shot(s.shot_id)
                self.store.remove_shot(s.shot_id)
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
        tail = self.define_shot(
            name=tail_name,
            start=at,
            end=tail_end,
            objects=list(shot.objects),
            description=shot.description,
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
                shot.start = old_start
            if abs(tail) > _EPS:
                pre_tail_end = shot.end
                shot.end = self.store.snap(pre_tail_end + tail)
                # Ripple by what the END ACTUALLY moved rather than by the
                # amount asked for -- the same value ``fit_shot_to_content``
                # passes.  The plan snaps its own destinations, so the two
                # agree today; deriving it from the bound is what keeps them
                # agreeing if either side's rounding ever changes.
                self.ripple_downstream(shot_id, pre_tail_end, shot.end - pre_tail_end)

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

        Parameters:
            gap: Frames of gap between consecutive shots.
            start_frame: Timeline frame for the first shot.
            respect_locks: When False, spend *gap* on locked gaps too.  The
                locks are left set either way.
        """
        plan = ShotPlanner.plan_respace(
            self.store, gap, start_frame, respect_locks=respect_locks
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
