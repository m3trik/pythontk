# !/usr/bin/python
# coding=utf-8
"""Shot Manifest engine — pure planning/orchestration core with scene hooks.

DCC-agnostic.  The :class:`ShotManifest` engine turns parsed
:class:`~pythontk.core_utils.engines.shots.manifest.manifest_model.BuilderStep`
graphs into a :class:`~pythontk.core_utils.engines.shots.shot_model.ShotStore`
build (compute-then-commit) and assesses the result — all without importing any
DCC.  Every place the original reached into a live scene (fps query, audio clip
measurement, name resolution, animation walks, key application, existence
checks) is exposed as an **overridable hook with a pure default**; the DCC
toolkits (mayatk, blendertk) subclass this and override those hooks.

Mirrors the :class:`ShotStore` hook pattern: the pure core never imports
``maya`` / ``bpy`` / Qt, and the module-level duration helpers
(:func:`resolve_duration`, :func:`_audio_placeholder_dur`) take an optional
``measure_audio`` callable so audio-clip probing stays a DCC concern.
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from pythontk.core_utils.engines.shots.shot_model import ShotStore
from pythontk.core_utils.engines.shots.manifest.manifest_model import (
    Action,
    AUDIO_PLACEHOLDER_DURATION,
    BuilderObject,
    BuilderStep,
    ColumnMap,
    DEFAULT_FIT_MODE,
    DEFAULT_INITIAL_SHOT_LENGTH,
    FitMode,
    ObjectStatus,
    PlannedShot,
    StepStatus,
    parse_csv,
)

log = logging.getLogger(__name__)

__all__ = ["ShotManifest", "resolve_duration"]


# ---------------------------------------------------------------------------
# Duration resolution  (pure — audio probing injected via ``measure_audio``)
# ---------------------------------------------------------------------------


def resolve_duration(
    step: BuilderStep,
    initial_shot_length: float,
    fit_mode: FitMode,
    fps: float,
    measure_audio: Optional[Callable[[BuilderObject], Optional[float]]] = None,
) -> Tuple[float, float, float]:
    """Compute final shot duration for *step* under the given fit policy.

    Probes behavior templates and (via *measure_audio*) audio-clip lengths to
    determine the minimum content-driven length, then applies *fit_mode*
    against the user-specified *initial_shot_length* (default 200f).

    Parameters:
        step: The step whose objects drive the duration.
        initial_shot_length: Baseline length the fit policy is applied against.
        fit_mode: ``"extend_only"`` or ``"fit_contents"``.
        fps: Scene frame rate.  Retained for signature compatibility — the
            pure core does not probe audio itself; a DCC layer binds its
            frame-rate into *measure_audio* when it needs one.
        measure_audio: Optional callable ``(BuilderObject) -> Optional[float]``
            returning an audio clip's length in frames, or ``None`` when it is
            unresolvable.  Injected by the DCC layer.  ``None`` (or a ``None`` /
            non-positive return) contributes no audio length, so a purely
            behavior-template step is sized entirely by its templates.

    Returns:
        ``(duration, behavior_span, audio_span)`` — the resolved shot
        length plus the individual content measurements that drove it.
    """
    from pythontk.core_utils.engines.shots.manifest.behaviors import load_behavior

    audio_span = 0.0
    max_obj_total = 0.0
    global_max_in = 0.0
    global_max_out = 0.0

    for obj in step.objects:
        # ---- behavior template durations (phase-aware) ----
        obj_in = 0.0
        obj_out = 0.0
        for b in obj.behaviors:
            if not b:
                continue
            try:
                tmpl = load_behavior(b)
            except FileNotFoundError:
                continue
            dur_field = tmpl.get("duration")
            if dur_field == "from_source":
                continue  # handled via audio_span below
            for _attr_name, attr_def in tmpl.get("attributes", {}).items():
                for phase in ("in", "out"):
                    block = attr_def.get(phase)
                    if not block:
                        continue
                    d = float(block.get("duration", 0) or 0)
                    if phase == "in":
                        obj_in += d
                    else:
                        obj_out += d
        max_obj_total = max(max_obj_total, obj_in + obj_out)
        global_max_in = max(global_max_in, obj_in)
        global_max_out = max(global_max_out, obj_out)

        # ---- audio clip length ----
        if obj.kind == "audio" and measure_audio is not None:
            try:
                frames = measure_audio(obj)
            except Exception as exc:
                log.debug("audio duration probe failed for %r: %s", obj.name, exc)
                frames = None
            if frames and frames > 0:
                audio_span = max(audio_span, float(frames))

    behavior_span = max(max_obj_total, global_max_in + global_max_out)
    content_min = max(behavior_span, audio_span)

    if fit_mode == "extend_only":
        duration = max(initial_shot_length, content_min)
    elif fit_mode == "fit_contents":
        duration = content_min if content_min > 0 else initial_shot_length
    else:
        raise ValueError(f"Unknown fit_mode: {fit_mode!r}")

    return duration, behavior_span, audio_span


def _audio_placeholder_dur(
    step: BuilderStep,
    measure_audio: Optional[Callable[[BuilderObject], Optional[float]]] = None,
) -> Optional[float]:
    """Return ``AUDIO_PLACEHOLDER_DURATION`` if *step* is an audio step with
    no resolvable source — i.e. the shot should grow to the clip length
    once it loads, but currently has nothing to size by.  Returns ``None``
    when a regular fit-mode + initial_shot_length policy should be used.

    Source resolvability is probed through *measure_audio*: a positive
    return means the clip is measurable (regular policy applies), so ``None``
    is returned.  Absent a resolver, an audio step with no ``source_path``
    yields the placeholder.
    """
    has_audio = False
    for obj in step.objects:
        if obj.kind != "audio":
            continue
        has_audio = True
        if obj.source_path:
            return None
        if measure_audio is not None:
            try:
                dur = measure_audio(obj)
            except Exception:
                dur = None
            if dur and dur > 0:
                return None
    return AUDIO_PLACEHOLDER_DURATION if has_audio else None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ShotManifest:
    """Creates shot store entries from parsed steps and applies behaviors.

    Duration for each step is derived entirely from behavior templates.
    Layout is computed from the current store state (new shots append
    after the last existing shot; frame 1 when empty).

    The pure, DCC-agnostic core.  Scene-reaching behaviour lives in the
    overridable hooks (:meth:`_resolve_fps`, :meth:`_measure_audio`,
    :meth:`_resolve_names_keep_missing`, :meth:`_filter_to_animated`,
    :meth:`_discover_scene_objects`, :meth:`rewire_audio`,
    :meth:`apply_behaviors`, :meth:`_object_exists`, :meth:`_verify_behavior`,
    :meth:`_keyframe_range`, :meth:`_audio_exists`,
    :meth:`_audio_grow_duration`), each with a pure default; the DCC toolkits
    subclass this and override them.

    Parameters:
        store: Target ``ShotStore`` instance to populate.
    """

    def __init__(self, store: ShotStore):
        self.store = store
        self._fps_cache: Optional[float] = None
        # Per-cycle caches (cleared at the top of update()/assess()):
        # transform → standard-attr curves, and per-curve key data.
        # Maintained here so a DCC subclass's scene-walk hooks share a single
        # per-build/-assess cache lifecycle without re-implementing the clears.
        self._animated_transforms: Optional[Dict[str, List[str]]] = None
        self._curve_data: Optional[Dict[str, Tuple[list, list]]] = None

    @staticmethod
    def _step_metadata(
        step: BuilderStep,
        pass_through: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Build a metadata dict from a parsed step."""
        meta: Dict[str, Any] = {
            "section": step.section,
            "section_title": step.section_title,
            "csv_objects": [{"name": o.name, "kind": o.kind} for o in step.objects],
            "behaviors": [
                {
                    "name": o.name,
                    "behavior": b,
                    "kind": o.kind,
                    "source_path": o.source_path,
                }
                for o in step.objects
                for b in o.behaviors
            ],
        }
        if step.audio and step.audio.upper() != "N/A":
            meta["voice_text"] = step.audio
        if pass_through:
            meta.update(pass_through)
        return meta

    # ---- scene hooks (overridable; pure defaults) ------------------------

    def _resolve_fps(self) -> float:
        """Return the scene FPS (overridable hook).

        Pure default: the store's ``scene_fps`` (or 24.0).  Cached per
        instance and cleared at the top of :meth:`update` so a single build
        call resolves fps once.  DCC subclasses override to query the live
        scene's time unit.
        """
        if self._fps_cache is not None:
            return self._fps_cache
        self._fps_cache = float(getattr(self.store, "scene_fps", None) or 24.0)
        return self._fps_cache

    def _measure_audio(self, obj: BuilderObject) -> Optional[float]:
        """Return *obj*'s audio-clip length in frames, or ``None`` (hook).

        Pure default: ``None`` — the pure core has no audio backend.  DCC
        subclasses override to normalize the track / resolve the source path
        and probe the file, returning its length in frames (or ``None`` when
        unresolvable).  Consumed by :func:`resolve_duration` and
        :func:`_audio_placeholder_dur`.
        """
        return None

    def _resolve_names_keep_missing(self, names: List[str]) -> List[str]:
        """Resolve object names, keeping the caller's form for missing ones (hook).

        Pure default: identity — pure / Blender names are already unique.  The
        Maya subclass overrides to long-name-resolve via the scene while
        keeping unresolved (missing / ambiguous) names in their original form
        so the pinned-object system can surface them instead of dropping them.
        """
        return list(names)

    def _filter_to_animated(
        self, names: List[str], start: float, end: float
    ) -> List[str]:
        """Return only objects with standard-attribute animation in [start, end] (hook).

        Pure default: identity (all *names* pass) — the pure core cannot walk
        anim curves.  DCC subclasses override to drop objects animated only on
        custom attributes (boundary markers) or with flat keys in the range.
        """
        return list(names)

    def _discover_scene_objects(
        self,
        start: float,
        end: float,
        exclude_names: set,
    ) -> List[str]:
        """Find animated scene objects in [start, end] not in *exclude_names* (hook).

        Pure default: ``[]`` — the pure core has no scene to walk.  DCC
        subclasses override to return transform nodes with non-flat standard-
        attribute animation whose leaf names are not already excluded.
        """
        return []

    def rewire_audio(
        self, tracks: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """Reconcile managed audio nodes with keyed track state (hook).

        Pure default: ``{"created": [], "updated": [], "deleted": []}`` — no
        audio backend in the pure core.  DCC subclasses override to sync their
        audio compositor.  Safe to call any time — after a build, after marker
        edits, or standalone from the UI.

        Parameters:
            tracks: When provided, limit reconciliation to these ``track_id``
                values.  Default: full scan.

        Returns:
            ``{"created": [...], "updated": [...], "deleted": [...]}`` of audio
            node names, or empty lists when unavailable.
        """
        return {"created": [], "updated": [], "deleted": []}

    def apply_behaviors(self) -> Dict[str, list]:
        """Apply detected behaviors to the store's shots (hook).

        Pure default: ``{"applied": [], "skipped": []}`` — behavior keying
        reaches a scene, so the pure core no-ops.  DCC subclasses override to
        key fades / audio onto each shot's objects (the mayatk/blendertk
        applier that maps a behavior template's ``resolve_keys`` output onto
        scene keyframes) and return the applied/skipped tallies.
        """
        return {"applied": [], "skipped": []}

    def _object_exists(self, name: str) -> bool:
        """Return whether *name* exists in the scene (hook).

        Pure default: ``True`` — the pure core cannot check a scene, so nothing
        is flagged missing.  DCC subclasses override (e.g. ``cmds.objExists``).
        """
        return True

    def _verify_behavior(
        self,
        obj: str,
        behavior: str,
        start: float,
        end: float,
        anchor_override: Optional[float] = None,
    ) -> bool:
        """Return whether *behavior*'s expected keys exist on *obj* (hook).

        Pure default: ``True`` — no scene to verify against.  DCC subclasses
        override to check the behaviour template's keyframes within
        ``[start, end]`` (honouring *anchor_override* for distributed anchors).
        """
        return True

    def _keyframe_range(self, obj_name: str) -> Optional[Tuple[float, float]]:
        """Return the full ``(min, max)`` keyframe extent of *obj_name* (hook).

        Pure default: ``None`` — no keys to query.  DCC subclasses override to
        return the object's keyframe time range, or ``None`` when it has none.
        """
        return None

    def _audio_exists(self, name: str) -> bool:
        """Return whether an audio node / track named *name* exists (hook).

        Pure default: ``False`` — no audio backend.  DCC subclasses override to
        check registered tracks and scene audio nodes.
        """
        return False

    def _audio_grow_duration(self, audio_objs: List[BuilderObject]) -> float:
        """Content-driven duration for an existing audio step (hook).

        Drives the audio-grow pass in :meth:`_compute_plan`.  Pure default:
        the larger of the behavior-template duration and the measured clip
        length — measurement routes through the same :meth:`_measure_audio`
        hook the new-shot path uses (via :func:`resolve_duration`), so an
        existing audio shot re-grows when its clip loads/changes (the pure
        ``_measure_audio`` returns ``None`` → no growth).  A DCC layer may
        override to route through its own ``compute_duration`` binding
        (e.g. one that resolves registered track paths and probes files).
        """
        from pythontk.core_utils.engines.shots.manifest.behaviors import (
            compute_duration,
        )

        tmpl_dur = compute_duration(audio_objs, fallback=0.0)
        measured = [self._measure_audio(o) for o in audio_objs]
        return max([tmpl_dur] + [m for m in measured if m])

    # ---- sync (thin orchestrator) ----------------------------------------

    def sync(
        self,
        steps: List[BuilderStep],
        apply_behaviors: bool = True,
        ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        remove_missing: bool = True,
        zero_duration_fallback: bool = False,
        fit_mode: FitMode = DEFAULT_FIT_MODE,
        initial_shot_length: float = DEFAULT_INITIAL_SHOT_LENGTH,
        skip_scene_discovery: bool = False,
    ) -> Tuple[Dict[str, str], Dict[str, list], List[StepStatus]]:
        """Full build pipeline: plan -> commit -> apply behaviors -> assess.

        Parameters:
            steps: Parsed steps to build.
            apply_behaviors: If True (default), detected behaviors are
                applied to scene objects after updating the store (via the
                :meth:`apply_behaviors` hook).
            ranges: Optional mapping of ``step_id`` → ``(start, end)``
                frame ranges.  When provided, shots are placed at these
                positions instead of being sequentially appended.
            remove_missing: If True (default), shots in the store that
                are absent from *steps* are removed.  Set to False for
                scene-detection mode where existing shots should be
                preserved.
            zero_duration_fallback: If True, new shots without an
                explicit range are created with zero duration instead
                of using ``compute_duration``.  Used during incremental
                builds to avoid disrupting existing shot positions.
            skip_scene_discovery: Forwarded to :meth:`assess` so a
                selected-keys build only considers the selected keys'
                objects instead of discovering every animated scene
                object in each shot's range.

        Returns:
            ``(actions, behavior_result, assessment)`` tuple.
        """
        actions = self.update(
            steps,
            ranges=ranges,
            remove_missing=remove_missing,
            zero_duration_fallback=zero_duration_fallback,
            fit_mode=fit_mode,
            initial_shot_length=initial_shot_length,
        )

        behavior_result: Dict[str, list] = {"applied": [], "skipped": []}
        if apply_behaviors:
            behavior_result = self.apply_behaviors()

        # Rewire managed audio nodes so the sequencer/timeline reflects any
        # key changes authored above.  Idempotent.
        self.rewire_audio()

        assessment = self.assess(steps, skip_scene_discovery=skip_scene_discovery)
        return actions, behavior_result, assessment

    # ---- update (data-only sync) ----------------------------------------

    def update(
        self,
        steps: List[BuilderStep],
        ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        remove_missing: bool = True,
        zero_duration_fallback: bool = False,
        fit_mode: FitMode = DEFAULT_FIT_MODE,
        initial_shot_length: float = DEFAULT_INITIAL_SHOT_LENGTH,
    ) -> Dict[str, str]:
        """Sync parsed steps to the ShotStore (data only, no behaviors).

        Computes a full build plan, then commits it to the store in a
        single pass.  All position arithmetic (cursor placement,
        audio-grow, ripple deltas) happens on :class:`PlannedShot`
        objects before any store mutation occurs.

        Returns:
            Dict mapping ``step_id`` -> action taken
            (``"created"`` | ``"patched"`` | ``"skipped"``
            | ``"locked"`` | ``"removed"``).
        """
        self._fps_cache = None
        self._animated_transforms = None
        self._curve_data = None
        plan = self._compute_plan(
            steps,
            ranges=ranges,
            remove_missing=remove_missing,
            zero_duration_fallback=zero_duration_fallback,
            fit_mode=fit_mode,
            initial_shot_length=initial_shot_length,
        )
        return self._execute_plan(plan, remove_missing=remove_missing)

    # ---- compute-then-commit internals -----------------------------------

    def _compute_plan(
        self,
        steps: List[BuilderStep],
        ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        remove_missing: bool = True,
        zero_duration_fallback: bool = False,
        fit_mode: FitMode = DEFAULT_FIT_MODE,
        initial_shot_length: float = DEFAULT_INITIAL_SHOT_LENGTH,
    ) -> List[PlannedShot]:
        """Pure planning pass: compute final positions without touching the store.

        Reads the current store state once, then builds a list of
        :class:`PlannedShot` objects that describe every mutation
        (create, patch, skip, lock, remove).  All cursor advancement,
        audio-grow, and ripple arithmetic happens here on plan data.

        Returns:
            Ordered list of :class:`PlannedShot` instructions.
        """
        sorted_shots = self.store.sorted_shots()
        built_map = {s.name: s for s in sorted_shots}
        csv_ids = {step.step_id for step in steps}
        plan: List[PlannedShot] = []

        # Track removals
        if remove_missing:
            for name, shot in list(built_map.items()):
                if name not in csv_ids:
                    dummy_step = BuilderStep(
                        step_id=name,
                        section="",
                        section_title="",
                        description="",
                    )
                    plan.append(
                        PlannedShot(
                            step=dummy_step,
                            action="removed",
                            existing_shot_id=shot.shot_id,
                        )
                    )

        # Cursor for new shots (after all existing shots).
        # We maintain a virtual cursor that advances as we plan
        # new shots, independent of the store.
        cursor = sorted_shots[-1].end if sorted_shots else 1.0

        # Accumulate ripple deltas from audio-grow so downstream
        # planned positions account for earlier expansions.
        cumulative_ripple = 0.0

        for step in steps:
            existing = built_map.get(step.step_id)
            meta = self._step_metadata(
                step, pass_through=getattr(step, "_pass_through", None)
            )

            if existing is None:
                # ---- NEW SHOT ----
                # Placement comes from ``rng[0]`` (when provided) or the
                # cursor.  Duration: a provided range pins the end
                # (grown only when measured content exceeds it);
                # otherwise ``fit_mode`` / ``initial_shot_length``
                # govern.  ``zero_duration_fallback`` is the one opt-out
                # (incremental/selected-keys flows).
                rng = ranges.get(step.step_id) if ranges else None
                if zero_duration_fallback and rng is not None:
                    start = rng[0] + cumulative_ripple
                    end = rng[1] + cumulative_ripple
                elif zero_duration_fallback:
                    start = cursor + cumulative_ripple
                    end = start
                else:
                    adjusted_cursor = cursor + cumulative_ripple
                    if rng is not None:
                        # ``rng[0]`` is the preferred placement, but if
                        # the fit-driven duration would overlap the
                        # previous shot, ripple forward to the cursor.
                        start = max(rng[0] + cumulative_ripple, adjusted_cursor)
                    else:
                        start = adjusted_cursor
                    fps = self._resolve_fps()
                    if rng is not None:
                        # Range pinned by caller (rebuild / explicit user
                        # ranges) — respect rng[1] as the end, but still
                        # grow if measured content exceeds it.
                        _content_dur, _beh, _aud = resolve_duration(
                            step,
                            initial_shot_length=0.0,
                            fit_mode="fit_contents",
                            fps=fps,
                            measure_audio=self._measure_audio,
                        )
                        dur = max(rng[1] - rng[0], _content_dur)
                    else:
                        # Audio steps with no resolvable source get a
                        # small placeholder so the shot grows once the
                        # clip loads.
                        placeholder = _audio_placeholder_dur(
                            step, measure_audio=self._measure_audio
                        )
                        if placeholder is not None:
                            dur = placeholder
                        else:
                            dur, _beh, _aud = resolve_duration(
                                step,
                                initial_shot_length,
                                fit_mode,
                                fps,
                                measure_audio=self._measure_audio,
                            )
                    end = start + dur

                scene_objs = [o for o in step.objects if o.kind != "audio"]
                obj_names = [o.name for o in scene_objs]

                plan.append(
                    PlannedShot(
                        step=step,
                        action="created",
                        start=start,
                        end=end,
                        objects=obj_names,
                        metadata=meta,
                        description=step.display_text,
                    )
                )
                # Advance virtual cursor
                if end == start:
                    cursor = (end - cumulative_ripple) + (
                        self.store.gap if self.store.gap > 0 else 1
                    )
                else:
                    cursor = end - cumulative_ripple
                continue

            # ---- EXISTING SHOT ----
            if existing.locked:
                plan.append(
                    PlannedShot(
                        step=step,
                        action="locked",
                        start=existing.start + cumulative_ripple,
                        end=existing.end + cumulative_ripple,
                        existing_shot_id=existing.shot_id,
                    )
                )
                continue

            # Apply cumulative ripple to existing position
            ex_start = existing.start + cumulative_ripple
            ex_end = existing.end + cumulative_ripple

            # Reposition from user-provided range
            repositioned = False
            rng = ranges.get(step.step_id) if ranges else None
            if rng is not None:
                new_start = rng[0] + cumulative_ripple
                new_end = rng[1] + cumulative_ripple
                if abs(ex_start - new_start) > 1e-6 or abs(ex_end - new_end) > 1e-6:
                    ex_start, ex_end = new_start, new_end
                    repositioned = True

            # Audio-grow: compute whether audio extends the shot
            range_is_noop = rng is None or (
                abs(rng[0] - existing.start) < 1e-6
                and abs(rng[1] - existing.end) < 1e-6
            )
            ripple_delta = 0.0
            new_audio = {o.name for o in step.objects if o.kind == "audio"}
            if range_is_noop and new_audio:
                audio_objs = [o for o in step.objects if o.kind == "audio"]
                new_dur = self._audio_grow_duration(audio_objs)
                current_dur = ex_end - ex_start
                if new_dur > current_dur + 1e-6:
                    ripple_delta = (ex_start + new_dur) - ex_end
                    ex_end = ex_start + new_dur
                    repositioned = True
                    cumulative_ripple += ripple_delta

            # Diff CSV objects vs previous
            csv_obj_map = {
                o.name: sorted(o.behaviors) for o in step.objects if o.kind != "audio"
            }
            csv_objs = set(csv_obj_map)
            raw_csv = existing.metadata.get("csv_objects", existing.objects)
            old_csv_objs = set(
                (e["name"] if isinstance(e, dict) else e)
                for e in raw_csv
                if not (isinstance(e, dict) and e.get("kind") == "audio")
            )
            scene_discovered = set(existing.objects) - old_csv_objs

            old_behaviors: Dict[str, List[str]] = {}
            for entry in existing.metadata.get("behaviors", []):
                old_behaviors.setdefault(entry["name"], []).append(
                    entry.get("behavior", "")
                )
            for k in old_behaviors:
                old_behaviors[k] = sorted(old_behaviors[k])

            new_objs = csv_objs - old_csv_objs
            changed_beh = {
                name
                for name in csv_objs & old_csv_objs
                if csv_obj_map.get(name, []) != old_behaviors.get(name, [])
            }

            old_audio = {
                e["name"]
                for e in raw_csv
                if isinstance(e, dict) and e.get("kind") == "audio"
            }
            audio_changed = new_audio != old_audio

            has_content_change = bool(
                new_objs or (old_csv_objs - csv_objs) or changed_beh
            )

            if has_content_change or repositioned or audio_changed:
                action: Action = "patched"
            else:
                action = "skipped"

            # Compute merged objects for patched shots
            merged_objects = sorted(csv_objs | scene_discovered)

            plan.append(
                PlannedShot(
                    step=step,
                    action=action,
                    start=ex_start,
                    end=ex_end,
                    objects=merged_objects,
                    metadata=meta,
                    description=step.display_text or "",
                    existing_shot_id=existing.shot_id,
                    ripple_delta=ripple_delta,
                )
            )

        return plan

    def _execute_plan(
        self,
        plan: List[PlannedShot],
        remove_missing: bool = True,
    ) -> Dict[str, str]:
        """Commit a build plan to the store in a single pass.

        Applies removals first, then creates/patches in plan order.
        All positional data comes from the plan -- no re-reading of
        the store is needed.

        Returns:
            Dict mapping ``step_id`` -> action string.
        """
        actions: Dict[str, str] = {}

        # Coalesce per-shot mutations into a single flush/save and a
        # single BatchComplete event for UI listeners.
        with self.store.batch_update():
            # Phase 1: removals
            for ps in plan:
                if ps.action == "removed" and ps.existing_shot_id is not None:
                    self.store.remove_shot(ps.existing_shot_id)
                    actions[ps.step.step_id] = "removed"

            # Phase 2: creates / patches / skips / locks (order matters)
            for ps in plan:
                if ps.action == "removed":
                    continue

                if ps.action == "created":
                    # Store resolved (long / unique) names; missing objects
                    # keep their CSV form so pinning can surface them later.
                    obj_names = self._resolve_names_keep_missing(ps.objects)
                    self.store.define_shot(
                        name=ps.step.step_id,
                        start=ps.start,
                        end=ps.end,
                        objects=obj_names,
                        metadata=ps.metadata,
                        description=ps.description,
                    )
                    for n in obj_names:
                        self.store.set_object_pinned(n)
                    actions[ps.step.step_id] = "created"

                elif ps.action == "locked":
                    # Locked shots are content-protected, but still need
                    # repositioning if an upstream ripple displaced them.
                    if ps.existing_shot_id is not None:
                        existing = self._find_shot(ps.existing_shot_id)
                        if existing and (
                            abs(existing.start - ps.start) > 1e-6
                            or abs(existing.end - ps.end) > 1e-6
                        ):
                            self.store.update_shot(
                                existing.shot_id,
                                start=ps.start,
                                end=ps.end,
                            )
                    actions[ps.step.step_id] = "locked"

                elif ps.action == "skipped":
                    # Still update metadata/description from CSV, and
                    # reposition if an upstream ripple displaced this shot.
                    # All writes go through update_shot so the store is
                    # dirtied/notified (direct attribute writes are
                    # silently lost on save).
                    if ps.existing_shot_id is not None:
                        existing = self._find_shot(ps.existing_shot_id)
                        if existing:
                            kwargs = {}
                            if (
                                abs(existing.start - ps.start) > 1e-6
                                or abs(existing.end - ps.end) > 1e-6
                            ):
                                kwargs.update(start=ps.start, end=ps.end)
                            if existing.metadata != ps.metadata:
                                kwargs["metadata"] = ps.metadata
                            if existing.description != ps.description:
                                kwargs["description"] = ps.description
                            if kwargs:
                                self.store.update_shot(existing.shot_id, **kwargs)
                    actions[ps.step.step_id] = "skipped"

                elif ps.action == "patched":
                    if ps.existing_shot_id is not None:
                        existing = self._find_shot(ps.existing_shot_id)
                        if existing:
                            kwargs = {}
                            # Apply absolute position from plan — no
                            # ripple pass needed because the plan already
                            # computed final positions for every shot.
                            if (
                                abs(existing.start - ps.start) > 1e-6
                                or abs(existing.end - ps.end) > 1e-6
                            ):
                                kwargs.update(start=ps.start, end=ps.end)
                            if existing.metadata != ps.metadata:
                                kwargs["metadata"] = ps.metadata
                            if existing.description != ps.description:
                                kwargs["description"] = ps.description

                            # Merge CSV objects with scene-discovered
                            # extras.  Resolve the CSV names; missing
                            # objects keep their CSV form so pinning can
                            # surface them instead of silently dropping them.
                            csv_objs = {
                                o.name for o in ps.step.objects if o.kind != "audio"
                            }
                            scene_objs = set(ps.objects) - csv_objs
                            if scene_objs:
                                scene_objs = set(
                                    self._filter_to_animated(
                                        sorted(scene_objs), ps.start, ps.end
                                    )
                                )
                            csv_resolved = self._resolve_names_keep_missing(
                                sorted(csv_objs)
                            )
                            merged = sorted(set(csv_resolved) | scene_objs)
                            if set(existing.objects) != set(merged):
                                kwargs["objects"] = merged
                            # Route every write through update_shot so
                            # the store is dirtied/notified.
                            if kwargs:
                                self.store.update_shot(existing.shot_id, **kwargs)

                            for n in csv_resolved:
                                self.store.set_object_pinned(n)

                    actions[ps.step.step_id] = "patched"

        return actions

    def _find_shot(self, shot_id: int):
        """Return the ShotBlock with *shot_id*, or None."""
        return self.store.shot_by_id(shot_id)

    # ---- assess ----------------------------------------------------------

    def assess(
        self,
        steps: List[BuilderStep],
        exists_fn: Optional[Callable[[str], bool]] = None,
        verify_fn: Optional[Callable] = None,
        keyframe_range_fn: Optional[
            Callable[[str], Optional[Tuple[float, float]]]
        ] = None,
        audio_exists_fn: Optional[Callable[[str], bool]] = None,
        skip_scene_discovery: bool = False,
    ) -> List[StepStatus]:
        """Compare parsed steps against the current store state.

        For each step, checks whether a matching shot has been built in
        :attr:`store`, whether every referenced object exists in the
        host application, and whether expected behavior keyframes are
        present.

        User-animated objects (no detected behavior) are checked for
        keyframe extent within the step range.  If their keys exceed the
        step boundaries, the step is flagged for expansion.

        Parameters:
            steps: Parsed steps from the CSV.
            exists_fn: Callable that returns ``True`` when an object name
                exists in the scene.  Defaults to :meth:`_object_exists`
                (pure default: always ``True``).
            verify_fn: Callable ``(obj, behavior, start, end) -> bool``
                that returns ``True`` when the expected behaviour keys
                exist.  Defaults to :meth:`_verify_behavior` (pure default:
                always ``True``).
            keyframe_range_fn: Callable ``(obj) -> (min_time, max_time)``
                returning the full keyframe extent for a user-animated
                object, or ``None`` if no keys exist.  Defaults to
                :meth:`_keyframe_range` (pure default: ``None``).
            audio_exists_fn: Callable that returns ``True`` when an audio
                node with the given name exists.  Defaults to
                :meth:`_audio_exists` (pure default: ``False``).

        Returns:
            One :class:`StepStatus` per step with per-object results.
        """
        if exists_fn is None:
            exists_fn = self._object_exists

        if verify_fn is None:
            verify_fn = self._verify_behavior

        if audio_exists_fn is None:
            audio_exists_fn = self._audio_exists

        # Invalidate per-assess caches
        self._animated_transforms = None
        self._curve_data = None

        if keyframe_range_fn is None:
            keyframe_range_fn = self._keyframe_range

        built_map = {s.name: s for s in self.store.sorted_shots()}

        results: List[StepStatus] = []
        for step in steps:
            shot = built_map.get(step.step_id)
            built = shot is not None
            is_locked = built and shot.locked

            # Locked shots are user-finalized — skip detailed checking
            if is_locked:
                obj_statuses = [
                    ObjectStatus(
                        name=o.name,
                        exists=True,
                        status="valid",
                    )
                    for o in step.objects
                ]
                results.append(
                    StepStatus(
                        step_id=step.step_id,
                        built=True,
                        objects=obj_statuses,
                        locked=True,
                    )
                )
                continue

            obj_statuses = []
            for obj in step.objects:
                if obj.kind == "audio":
                    exists = audio_exists_fn(obj.name)
                    broken = []
                    if not exists:
                        status = "missing_object"
                    elif built and obj.behaviors:
                        broken = [
                            b
                            for b in obj.behaviors
                            if not verify_fn(obj.name, b, shot.start, shot.end)
                        ]
                        status = "missing_behavior" if broken else "valid"
                    else:
                        status = "valid" if exists else "missing_object"
                    obj_statuses.append(
                        ObjectStatus(
                            name=obj.name,
                            exists=exists,
                            status=status,
                            behaviors=list(obj.behaviors),
                            broken_behaviors=broken,
                        )
                    )
                    continue
                exists = exists_fn(obj.name)
                key_range = None
                broken = []
                if not exists:
                    status = "missing_object"
                elif built and obj.behaviors:
                    # Check each declared behavior individually, modeling
                    # the distributed anchors apply_to_shots used to place
                    # multi-behavior objects (idx / (total-1)) — exact-mode
                    # verify against the template's default anchors would
                    # permanently flag them right after a successful build.
                    total = len(obj.behaviors)
                    for idx, b in enumerate(obj.behaviors):
                        anchor = idx / max(total - 1, 1) if total > 1 else None
                        try:
                            ok = verify_fn(
                                obj.name,
                                b,
                                shot.start,
                                shot.end,
                                anchor_override=anchor,
                            )
                        except TypeError:
                            # Caller-supplied 4-arg verify_fn (old seam).
                            ok = verify_fn(obj.name, b, shot.start, shot.end)
                        if not ok:
                            broken.append(b)
                    status = "missing_behavior" if broken else "valid"
                elif built and not obj.behaviors:
                    # User-animated: query actual keyframe extent
                    key_range = keyframe_range_fn(obj.name)
                    status = "user_animated" if key_range else "valid"
                else:
                    status = "valid"
                obj_statuses.append(
                    ObjectStatus(
                        name=obj.name,
                        exists=exists,
                        status=status,
                        behaviors=list(obj.behaviors),
                        broken_behaviors=broken,
                        key_range=key_range,
                    )
                )

            # Detect additional objects (in shot but not in CSV)
            additional = []
            if shot is not None:
                from pythontk.core_utils.engines.shots.shot_model import (
                    leaf_name as _short,
                )

                csv_short = {_short(o.name) for o in step.objects}
                stored_extra = [n for n in shot.objects if _short(n) not in csv_short]
                # Filter stored extras to only those with actual motion
                # (removes flat-key objects from previous builds).
                if stored_extra:
                    stored_extra = self._filter_to_animated(
                        stored_extra, shot.start, shot.end
                    )
                additional = stored_extra
                # Also discover scene objects with keys in this shot's
                # range that aren't tracked in the CSV or the store.
                # Skip in selected-keys mode: only the explicitly
                # selected keys' objects are relevant.
                if not skip_scene_discovery:
                    known = csv_short | {_short(n) for n in shot.objects}
                    scene_extra = self._discover_scene_objects(
                        shot.start, shot.end, known
                    )
                    additional.extend(scene_extra)
                    # Merge discovered objects into the shot so the sequencer
                    # can display them (it reads shot.objects).  Mark dirty —
                    # this mutates persisted state outside update_shot.
                    if scene_extra:
                        shot.objects = sorted(set(shot.objects) | set(scene_extra))
                        self.store.mark_dirty()

            # Compute shrinkable frames (unused tail)
            shrinkable = 0.0
            if built and shot is not None:
                content_end = self._compute_content_end(step, shot, obj_statuses)
                if content_end < shot.end:
                    shrinkable = shot.end - content_end

            results.append(
                StepStatus(
                    step_id=step.step_id,
                    built=built,
                    objects=obj_statuses,
                    additional_objects=additional,
                    shrinkable_frames=shrinkable,
                )
            )
        return results

    @staticmethod
    def _compute_content_end(
        step: BuilderStep,
        scene,
        obj_statuses: List[ObjectStatus],
    ) -> float:
        """Return the latest frame used by content in this step."""
        from pythontk.core_utils.engines.shots.manifest.behaviors import load_behavior

        latest = scene.start  # at minimum, content starts at scene start
        for obj, obj_st in zip(step.objects, obj_statuses):
            for beh in obj.behaviors:
                try:
                    tmpl = load_behavior(beh)
                except FileNotFoundError:
                    continue
                for _attr, attr_def in tmpl.get("attributes", {}).items():
                    for phase in ("in", "out"):
                        block = attr_def.get(phase)
                        if not block:
                            continue
                        anchor = block.get(
                            "anchor", "start" if phase == "in" else "end"
                        )
                        offset = block.get("offset", 0)
                        dur = block.get("duration", 0)
                        if anchor == "end":
                            end_t = scene.end - offset
                        else:
                            end_t = scene.start + offset + dur
                        if end_t > latest:
                            latest = end_t
            if obj_st.key_range:
                if obj_st.key_range[1] > latest:
                    latest = obj_st.key_range[1]
        return latest

    # ---- from_csv --------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        store: Optional[ShotStore] = None,
        columns: Optional[ColumnMap] = None,
        post_process: Optional[Callable[[BuilderStep], None]] = None,
    ) -> Tuple["ShotManifest", List[BuilderStep]]:
        """Convenience: parse a CSV and return a ready-to-build engine.

        Parameters:
            filepath: Path to the CSV file.
            store: Optional existing ``ShotStore`` to populate.
                If ``None``, a fresh instance is created.
            columns: Column index mapping.
            post_process: Optional callable invoked on each step after
                assembly.

        Returns:
            ``(builder, steps)`` tuple. Call ``builder.sync(steps)`` to
            execute.
        """
        steps = parse_csv(filepath, columns, post_process=post_process)
        st = store or ShotStore.active()
        builder = cls(st)
        return builder, steps
