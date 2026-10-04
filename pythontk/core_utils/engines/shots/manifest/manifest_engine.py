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

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from pythontk.core_utils.engines.shots.shot_model import ShotStore
from pythontk.core_utils.engines.shots.manifest.manifest_model import (
    ManifestModel,
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
    ShotPairing,
    StepStatus,
)

log = logging.getLogger(__name__)

__all__ = ["ShotManifest"]


# ---------------------------------------------------------------------------
# Duration resolution  (pure — audio probing injected via ``measure_audio``)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class _ShotManifestInternal(object):
    """Internal helpers for ShotManifest."""

    @staticmethod
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


class ShotManifest(_ShotManifestInternal):
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
    :meth:`_audio_grow_duration`, :meth:`_placed_clip_keys`), each with a pure
    default; the DCC toolkits subclass this and override them.

    The scene's :class:`~pythontk.EffectRecipe` (``store.effect_recipe``) is
    what every recipe-driven behavior is keyed, verified and sized from: a
    Build stamps the keys it writes with the recipe they were made under, and
    Assess flags those an older recipe made (``stale_behavior``).

    Parameters:
        store: Target ``ShotStore`` instance to populate.
        match: How a step finds its shot (:meth:`pair`): ``"name"`` (the
            stored binding, then the name) or ``"name_then_order"`` (then
            the remaining steps and shots in timeline order).
    """

    #: Metadata keys the manifest writes on a shot (:meth:`_step_metadata`);
    #: a build replaces these and the doc's pass-through columns, and keeps
    #: every other key it finds.
    MANIFEST_METADATA = frozenset(
        ("section", "section_title", "csv_objects", "behaviors", "voice_text", "step")
    )

    def __init__(self, store: ShotStore, match: str = "name"):
        self.store = store
        self.match = match
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
        # The binding: which doc step this shot is, whatever it is named.
        meta["step"] = step.step_id
        # A blank cell writes nothing (and, being the doc's column, takes out
        # the value it held -- _content_kwargs).
        meta.update({k: v for k, v in (pass_through or {}).items() if v})
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

    @property
    def recipe(self):
        """The scene's :class:`~pythontk.EffectRecipe` -- the store's."""
        return self.store.effect_recipe

    def _stamp(self, behavior: str) -> str:
        """The recipe stamp *behavior*'s keys carry: the fingerprint of the
        effect it keys, ``""`` for a template keying its own channels."""
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

        effect = Behaviors.effect_of(behavior)
        return self.recipe.fingerprint(effect) if effect else ""

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
        """Resolve object names, keeping the caller's form for missing ones.

        Through :meth:`_resolve_object` (the store's one resolver), so a build
        stores the node a namespaced reference really is; an unresolved
        (missing / ambiguous) name stays as written so the pinned-object system
        can surface it instead of dropping it.
        """
        out = []
        for name in names:
            node, reason = self._resolve_object(name)
            out.append(node if reason == "found" else name)
        return out

    def _resolve_object(self, name: str) -> Tuple[str, str]:
        """``(node, reason)`` for a doc object name: ``"found"`` / ``"missing"``
        / ``"ambiguous"`` -- the store's :meth:`~ShotStore.resolve_member`,
        confirmed by :meth:`_object_exists`."""
        node, reason = self.store.resolve_member(name)
        if reason == "found" and not self._object_exists(node):
            return name, "missing"
        return node, reason

    def _key_samples(
        self, obj: str, behavior: str, start: float, end: float
    ) -> List[Tuple[str, float]]:
        """``(curve, time)`` of every key in ``[start, end]`` on the channels
        *behavior* keys on *obj* (hook).

        Pure default: ``[]`` -- no scene curves.  DCC subclasses return the
        keys on the template's channels (and their mirrors), naming curves
        the way their ledger does.
        """
        return []

    def _delete_keys(self, curve: str, times: List[float]) -> None:
        """Delete the keys at *times* on *curve* (hook).  Pure default: no-op."""

    def _placed_clip_keys(
        self, name: str, start: float, end: float
    ) -> List[Tuple[str, float]]:
        """``(curve, time)`` of audio clip *name*'s keys placed exactly where a
        build places them over ``[start, end]`` (hook).

        A build before the manifest claimed its clip keys left them unclaimed,
        and a build no longer clears a whole track, so these are adopted
        (:meth:`adopt_placed_clips`) before a shot moves -- or the clip would
        play at its old place too.  Pure default: ``[]`` (no audio keys).
        """
        return []

    def _is_asset_candidate(self, name: str) -> bool:
        """Whether shot member *name* reads as a doc asset (hook) -- what
        auto-fill offers.  Pure default: ``True``; DCC subclasses leave out
        cameras, lights, joints and the like."""
        return True

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

    def rewire_audio(self, tracks: Optional[List[str]] = None) -> Dict[str, List[str]]:
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
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

        tmpl_dur = Behaviors.compute_duration(audio_objs, fallback=0.0)
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
        if apply_behaviors:
            # Before any shot moves: a clip keyed by a build that did not claim
            # it is the manifest's, and must move with its claim.
            self.adopt_placed_clips()
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
            self.release_dropped(steps)
            behavior_result = self.apply_behaviors()
            self._record_authored(behavior_result)

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
            | ``"locked"`` | ``"removed"`` | ``"refused"``).  ``"refused"``
            is a new step whose id the store will not take as a shot name
            (another shot is that clip ignoring case -- ``ShotStore.name_error``);
            it is logged and left unbuilt rather than aborting the batch.
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

    # ---- pairing -----------------------------------------------------------

    def pair(self, steps: List[BuilderStep]) -> ShotPairing:
        """Which shot each step is -- the one place a step finds its shot.

        In order: the shot whose ``metadata["step"]`` binding names the step
        (written by every build, so a renamed shot stays paired and an
        inserted step shifts nothing); then the shot of the step's name
        unless that shot is bound to another step; then, with ``match="name_then_order"``, the remaining
        steps and shots in doc and timeline order.  A shot nothing pairs with
        is an orphan -- reported, never removed by a build.
        """
        shots = self.store.sorted_shots()
        by_binding: Dict[str, Any] = {}
        for shot in shots:
            bound = (shot.metadata or {}).get("step")
            if bound and bound not in by_binding:
                by_binding[bound] = shot
        by_name = {shot.name: shot for shot in shots}
        pairing = ShotPairing()
        used: set = set()

        def take(sid: str, shot, how: str) -> None:
            pairing.shots[sid] = shot
            pairing.how[sid] = how
            used.add(shot.shot_id)

        ids = [step.step_id for step in steps]
        for sid in ids:
            shot = by_binding.get(sid)
            if shot is not None and shot.shot_id not in used:
                take(sid, shot, "binding")
        for sid in ids:
            shot = by_name.get(sid)
            if sid in pairing.shots or shot is None or shot.shot_id in used:
                continue
            if (shot.metadata or {}).get("step") not in (None, sid):
                continue  # bound to another step
            take(sid, shot, "name")
        if self.match == "name_then_order":
            todo = [sid for sid in ids if sid not in pairing.shots]
            free = [shot for shot in shots if shot.shot_id not in used]
            for sid, shot in zip(todo, free):
                take(sid, shot, "order")
        pairing.orphans = [shot for shot in shots if shot.shot_id not in used]
        return pairing

    # ---- behavior-key ownership (the ledger's ``authored`` register) -------

    def unowned_keys(
        self, obj: str, behavior: str, start: float, end: float
    ) -> List[Tuple[str, float]]:
        """Keys on *behavior*'s channels of *obj* in ``[start, end]`` that the
        system did not write -- the animator's, which a build never touches."""
        led = self.store.edit_ledger
        return [
            (curve, t)
            for curve, t in self._key_samples(obj, behavior, start, end)
            if not led.owns_any(curve, t)
        ]

    def release_authored(self, shot_id: int, obj: str, behavior: str) -> int:
        """Delete the keys *behavior* wrote on *obj* for shot *shot_id* and drop
        their claims -- what re-applying it does first.  Returns how many."""
        led = self.store.edit_ledger
        records = led.authored(owner=shot_id, obj=obj, behavior=behavior)
        by_curve: Dict[str, List[float]] = {}
        for curve, t in records:
            by_curve.setdefault(curve, []).append(t)
        for curve, times in by_curve.items():
            self._delete_keys(curve, times)
            for t in times:
                led.release_authored(curve, t)
        if records:
            self.store.mark_dirty()
        return len(records)

    def release_dropped(self, steps: List[BuilderStep]) -> int:
        """Delete the keys of behaviors a step's doc no longer lists.

        The manifest's own output, so a build takes it out: a behavior
        unticked or dropped from the sheet, or an object dropped from the step,
        leaves claims on its shot that nothing in the doc answers.  Locked
        shots keep theirs, and so does a shot no step pairs with (an orphan is
        never edited by a build -- removing it is the user's call, and leaves
        its keys).  Returns how many keys were released.
        """
        released = 0
        for pairs, shot in self._dropped_pairs(steps):
            for obj, behavior in pairs:
                released += self.release_authored(shot.shot_id, obj, behavior)
        return released

    def _dropped_pairs(self, steps: List[BuilderStep]):
        """``[(sorted [(obj, behavior)], shot)]`` -- per paired, unlocked
        shot, the claims its step's doc no longer lists."""
        led = self.store.edit_ledger
        pairing = self.pair(steps)
        out = []
        for step in steps:
            shot = pairing.shots.get(step.step_id)
            if shot is None or shot.locked:
                continue
            listed = {(o.name, b) for o in step.objects for b in o.behaviors or ()}
            dropped = sorted(led.authored_pairs(shot.shot_id) - listed)
            if dropped:
                out.append((dropped, shot))
        return out

    def adopt_placed_clips(self) -> int:
        """Claim the audio keys a build placed before it claimed them.

        For every unlocked shot's audio behavior with no claim, the keys
        :meth:`_placed_clip_keys` finds exactly where a build puts them are
        recorded as the manifest's -- so the next build moves its clip rather
        than leaving it to play at its old place too.  Returns how many keys
        were adopted.
        """
        led = self.store.edit_ledger
        n = 0
        for shot in self.store.sorted_shots():
            if shot.locked:
                continue
            for entry in (shot.metadata or {}).get("behaviors", ()):
                name, behavior = entry.get("name", ""), entry.get("behavior", "")
                is_audio = entry.get("kind") == "audio" or entry.get("source_path")
                if not (name and behavior and is_audio):
                    continue
                if led.authored(owner=shot.shot_id, obj=name, behavior=behavior):
                    continue
                for curve, t in self._placed_clip_keys(name, shot.start, shot.end):
                    if led.owns_any(curve, t):
                        continue
                    n += led.record_authored(
                        curve, t, shot.shot_id, behavior, name, self._stamp(behavior)
                    )
        if n:
            self.store.mark_dirty()
        return n

    def is_stale(self, shot_id: int, obj: str, behavior: str) -> bool:
        """Whether *behavior*'s keys on shot *shot_id*'s *obj* were made under
        an older effect recipe than the scene's.

        Only a recipe-keyed ramp (a fade or a pulse) can be stale, and only
        keys the manifest claims say what made them: a key claimed before
        stamps existed reads as stale (an older recipe made it), and a
        behavior with no claims has nothing to judge.
        """
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

        effect = Behaviors.effect_of(behavior)
        if effect not in ("fade_in", "fade_out", "pulse"):
            return False
        stamps = self.store.edit_ledger.authored_stamps(shot_id, obj, behavior)
        current = self.recipe.fingerprint(effect)
        return any(stamp != current for stamp in stamps)

    def reapply_object(self, shot, obj: BuilderObject) -> bool:
        """Re-key every behavior of one doc object over *shot*'s range.

        The per-object "Apply" action: an explicit request, so the
        animator-keys guard does not apply -- but ownership does.  Every
        behavior's previous keys are released before any is re-keyed (one at a
        time, a later release deleted the key an earlier behavior had just
        written on a shared frame) and the new ones recorded, each placed where
        a build places it (:meth:`Behaviors.anchor_overrides`) -- an audio
        clip on its track, by name.  Wrap the call in ``store.scene_edit`` for
        one undo step.  The keys of a behavior the doc dropped for the object
        go too.  Returns whether anything was applied or released.
        """
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

        behaviors = list(obj.behaviors or [])
        claimed = {
            b
            for o, b in self.store.edit_ledger.authored_pairs(shot.shot_id)
            if o == obj.name
        }
        if not behaviors and not claimed:
            return False
        if obj.kind == "audio" or obj.source_path:
            # A clip is keyed by its track name, as a build's audio pass keys
            # it: no scene node answers to it, so resolving read it as missing.
            node = obj.name
        else:
            node, reason = self._resolve_object(obj.name)
            if reason != "found":
                return False
        # Its listed behaviors' keys AND those of behaviors the doc dropped for
        # it: the Apply re-keys exactly what the doc lists now.
        for behavior in sorted(claimed | set(behaviors)):
            self.release_authored(shot.shot_id, obj.name, behavior)
        anchors = Behaviors.anchor_overrides(behaviors)
        for behavior, anchor in zip(behaviors, anchors):
            kwargs: Dict[str, Any] = {
                "source_path": obj.source_path or "",
                "recipe": self.recipe,
                "fps": self._resolve_fps(),
            }
            if anchor is not None:
                kwargs["anchor_override"] = anchor
            written = self._apply_one(node, behavior, shot.start, shot.end, **kwargs)
            self._record_authored(
                {
                    "applied": [
                        {
                            "object": obj.name,
                            "behavior": behavior,
                            "shot_id": shot.shot_id,
                            "keys": list(written or ()),
                        }
                    ]
                }
            )
        return True

    def _apply_one(
        self, node: str, behavior: str, start: float, end: float, **kwargs
    ) -> List[Tuple[str, float]]:
        """Key one *behavior* on *node* over ``[start, end]`` and return the
        ``(curve, time)`` keys written (hook).  Pure default: nothing keyed."""
        return []

    def _record_authored(self, behavior_result: Dict[str, list]) -> int:
        """Claim the keys the applier reports writing (``applied[*]["keys"]``),
        each stamped with the recipe it was keyed under (:meth:`_stamp`)."""
        led = self.store.edit_ledger
        n = 0
        stamps: Dict[str, str] = {}
        for entry in behavior_result.get("applied", ()):
            shot_id = entry.get("shot_id")
            if shot_id is None:
                continue
            behavior = entry.get("behavior", "")
            if behavior not in stamps:
                stamps[behavior] = self._stamp(behavior)
            for curve, t in entry.get("keys") or ():
                n += led.record_authored(
                    curve,
                    t,
                    shot_id,
                    behavior,
                    entry.get("object", ""),
                    stamps[behavior],
                )
        if n:
            self.store.mark_dirty()
        return n

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
        pairing = self.pair(steps)
        key = ShotStore.member_key
        plan: List[PlannedShot] = []

        # Removals: only when asked, and only of shots no step pairs with.
        if remove_missing:
            for shot in pairing.orphans:
                dummy_step = BuilderStep(
                    step_id=shot.name,
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
            existing = pairing.shots.get(step.step_id)
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
                        _content_dur, _beh, _aud = ShotManifest.resolve_duration(
                            step,
                            initial_shot_length=0.0,
                            fit_mode="fit_contents",
                            fps=fps,
                            measure_audio=self._measure_audio,
                            recipe=self.recipe,
                        )
                        dur = max(rng[1] - rng[0], _content_dur)
                    else:
                        # Audio steps with no resolvable source get a
                        # small placeholder so the shot grows once the
                        # clip loads.
                        placeholder = _ShotManifestInternal._audio_placeholder_dur(
                            step, measure_audio=self._measure_audio
                        )
                        if placeholder is not None:
                            dur = placeholder
                        else:
                            dur, _beh, _aud = ShotManifest.resolve_duration(
                                step,
                                initial_shot_length,
                                fit_mode,
                                fps,
                                measure_audio=self._measure_audio,
                                recipe=self.recipe,
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
            # Doc names vs stored (long, namespaced) members: one identity rule.
            csv_keys = {key(n) for n in csv_objs}
            old_keys = {key(n) for n in old_csv_objs}
            member_keys = {key(n) for n in existing.objects}
            scene_discovered = {n for n in existing.objects if key(n) not in old_keys}

            old_behaviors: Dict[str, List[str]] = {}
            for entry in existing.metadata.get("behaviors", []):
                old_behaviors.setdefault(key(entry["name"]), []).append(
                    entry.get("behavior", "")
                )
            for k in old_behaviors:
                old_behaviors[k] = sorted(old_behaviors[k])

            new_objs = {n for n in csv_objs if key(n) not in old_keys}
            dropped = {n for n in old_csv_objs if key(n) not in csv_keys}
            changed_beh = {
                name
                for name in csv_objs
                if key(name) in old_keys
                and csv_obj_map.get(name, []) != old_behaviors.get(key(name), [])
            }
            # A doc object in the scene but not among the members (removed from
            # the shot by hand, or never added): a build puts it back.
            absent = {
                n
                for n in csv_objs
                if key(n) not in member_keys and self._resolve_object(n)[1] == "found"
            }

            old_audio = {
                e["name"]
                for e in raw_csv
                if isinstance(e, dict) and e.get("kind") == "audio"
            }
            audio_changed = new_audio != old_audio

            has_content_change = bool(new_objs or dropped or changed_beh or absent)

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

    @staticmethod
    def _reposition_kwargs(existing, ps: PlannedShot) -> Dict[str, Any]:
        """``start``/``end`` kwargs when the plan displaced *existing*, else empty."""
        if abs(existing.start - ps.start) > 1e-6 or abs(existing.end - ps.end) > 1e-6:
            return {"start": ps.start, "end": ps.end}
        return {}

    @classmethod
    def _content_kwargs(cls, existing, ps: PlannedShot) -> Dict[str, Any]:
        """``metadata``/``description`` kwargs where the doc differs from the store.

        The manifest's own keys (:attr:`MANIFEST_METADATA`, plus every
        pass-through column the step's doc has -- a blank cell included, so
        clearing it takes the value out) are replaced; every other key --
        Assess's ``object_status``, another tool's, a column the doc does not
        have -- is kept.
        """
        kwargs: Dict[str, Any] = {}
        columns = getattr(ps.step, "_pass_through", None) or ()
        owned = cls.MANIFEST_METADATA | set(columns)
        merged = {k: v for k, v in (existing.metadata or {}).items() if k not in owned}
        merged.update(ps.metadata)
        if existing.metadata != merged:
            kwargs["metadata"] = merged
        if existing.description != ps.description:
            kwargs["description"] = ps.description
        return kwargs

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
                    # Checked HERE, not in the plan: phase 1 has already
                    # removed the shots it planned to, and one of those may
                    # be the name this step would have collided with.
                    refused = self.store.name_error(ps.step.step_id)
                    if refused:
                        log.warning("Step %s not built: %s", ps.step.step_id, refused)
                        actions[ps.step.step_id] = "refused"
                        continue
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

                elif ps.action in ("locked", "skipped", "patched"):
                    # All writes go through update_shot so the store is
                    # dirtied/notified (direct attribute writes are
                    # silently lost on save).  Positions are absolute —
                    # the plan already computed final positions for every
                    # shot, so no ripple pass is needed here.
                    existing = (
                        self.store.shot_by_id(ps.existing_shot_id)
                        if ps.existing_shot_id is not None
                        else None
                    )
                    if existing is not None:
                        # Every action repositions if an upstream ripple
                        # displaced the shot; locked shots are content-
                        # protected, so metadata/description stay untouched.
                        kwargs = self._reposition_kwargs(existing, ps)
                        if ps.action != "locked":
                            kwargs.update(self._content_kwargs(existing, ps))

                        csv_resolved: List[str] = []
                        if ps.action == "patched":
                            # Merge CSV objects with scene-discovered
                            # extras.  Resolve the CSV names; missing
                            # objects keep their CSV form so pinning can
                            # surface them instead of silently dropping them.
                            csv_objs = {
                                o.name for o in ps.step.objects if o.kind != "audio"
                            }
                            key = ShotStore.member_key
                            csv_keys = {key(n) for n in csv_objs}
                            scene_objs = {
                                n for n in ps.objects if key(n) not in csv_keys
                            }
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

                        if kwargs:
                            self.store.update_shot(existing.shot_id, **kwargs)
                        for n in csv_resolved:
                            self.store.set_object_pinned(n)

                    actions[ps.step.step_id] = ps.action

        return actions

    # ---- assess ----------------------------------------------------------

    def fill_missing_assets(self, steps: List[BuilderStep]) -> Dict[str, List[str]]:
        """Give every step that lists no scene objects what its paired shot holds.

        Only a paired shot (:meth:`pair`) links a step to scene time: its
        members, plus what animates inside its range (what Assess reports as
        additional), are the step's objects.  A step without a shot stays
        empty -- its range would be a guess.  Members the scene no longer
        holds (:meth:`_object_exists`) and non-assets
        (:meth:`_is_asset_candidate`: cameras, lights, ...) are left out.
        Names are member keys -- the leaf, namespace dropped -- as a doc
        writes them, added in place with ``origin="shot"`` and no behaviors.

        Returns:
            ``{step_id: [names added]}`` for every step that was filled.
        """
        pairing = self.pair(steps)
        key = ShotStore.member_key
        self._animated_transforms = None  # one discovery cycle per call
        self._curve_data = None
        filled: Dict[str, List[str]] = {}
        for step in steps:
            if any(o.kind != "audio" for o in step.objects):
                continue
            shot = pairing.shots.get(step.step_id)
            if shot is None:
                continue
            members = [n for n in shot.objects if self._object_exists(n)]
            animated = self._discover_scene_objects(
                shot.start, shot.end, {key(n) for n in members}
            )
            names = list(
                dict.fromkeys(
                    key(n) for n in members + animated if self._is_asset_candidate(n)
                )
            )
            if not names:
                continue
            step.objects.extend(BuilderObject(name=n, origin="shot") for n in names)
            filled[step.step_id] = names
        return filled

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
        if verify_fn is None:
            verify_fn = self._verify_behavior

        if audio_exists_fn is None:
            audio_exists_fn = self._audio_exists

        # Invalidate per-assess caches
        self._animated_transforms = None
        self._curve_data = None

        if keyframe_range_fn is None:
            keyframe_range_fn = self._keyframe_range

        pairing = self.pair(steps)
        key = ShotStore.member_key
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

        known = set(Behaviors.list_behaviors())
        dropped_by_step = {
            shot.shot_id: pairs for pairs, shot in self._dropped_pairs(steps)
        }

        if exists_fn is not None:  # caller seam: a plain exists check

            def resolve(name):
                return name, ("found" if exists_fn(name) else "missing")

        else:
            resolve = self._resolve_object

        results: List[StepStatus] = []
        for step in steps:
            shot = pairing.shots.get(step.step_id)
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

            members = {key(n) for n in shot.objects} if built else set()
            dropped = dropped_by_step.get(shot.shot_id, []) if built else []
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
                        status = "valid"
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
                node, reason = resolve(obj.name)
                exists = reason == "found"
                key_range = None
                unknown = [b for b in obj.behaviors if b not in known]
                broken = list(unknown)
                stale: List[str] = []
                if reason == "missing":
                    status = "missing_object"
                elif reason == "ambiguous":
                    status = "ambiguous_object"
                elif built and key(node) not in members:
                    status = "not_in_shot"
                elif built and obj.behaviors:
                    # Check each declared behavior individually, at the anchor
                    # the build placed it (Behaviors.anchor_overrides) —
                    # exact-mode verify against the template's default anchors
                    # would permanently flag a multi-behavior object right
                    # after a successful build.
                    failed = []
                    anchors = Behaviors.anchor_overrides(obj.behaviors)
                    for b, anchor in zip(obj.behaviors, anchors):
                        if b in unknown:
                            continue
                        try:
                            ok = verify_fn(
                                node,
                                b,
                                shot.start,
                                shot.end,
                                anchor_override=anchor,
                            )
                        except TypeError:
                            # Caller-supplied 4-arg verify_fn (old seam).
                            ok = verify_fn(node, b, shot.start, shot.end)
                        if not ok:
                            failed.append(b)
                    broken.extend(failed)
                    # Verified keys an older recipe made: Build re-keys them.
                    stale = [
                        b
                        for b in obj.behaviors
                        if b not in failed
                        and b not in unknown
                        and self.is_stale(shot.shot_id, obj.name, b)
                    ]
                    # Over keys the animator owns, Build's guard leaves a
                    # behavior as it is -- unsatisfied or stale alike, so it
                    # is a conflict, not a fix.
                    conflicts = [
                        b
                        for b in failed + stale
                        if self.unowned_keys(node, b, shot.start, shot.end)
                    ]
                    stale = [b for b in stale if b not in conflicts]
                    if unknown:
                        status = "unknown_behavior"
                    elif conflicts:
                        status = "behavior_conflict"
                    elif failed:
                        status = "missing_behavior"
                    else:
                        status = "stale_behavior" if stale else "valid"
                elif built:
                    # User-animated: query actual keyframe extent
                    key_range = keyframe_range_fn(node)
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
                        stale_behaviors=stale,
                    )
                )

            # Additional objects: members, or animation in the shot's range,
            # the doc does not list.  Reported only -- Assess writes nothing
            # (the sequencer's own discovery maintains membership).
            additional = []
            if shot is not None:
                doc_keys = {key(o.name) for o in step.objects}
                stored_extra = [n for n in shot.objects if key(n) not in doc_keys]
                # Filter stored extras to only those with actual motion
                # (removes flat-key objects from previous builds).
                if stored_extra:
                    stored_extra = self._filter_to_animated(
                        stored_extra, shot.start, shot.end
                    )
                additional = stored_extra
                # Skip in selected-keys mode: only the explicitly
                # selected keys' objects are relevant.
                if not skip_scene_discovery:
                    additional.extend(
                        self._discover_scene_objects(
                            shot.start, shot.end, doc_keys | members
                        )
                    )

            # Compute shrinkable frames (unused tail)
            shrinkable = 0.0
            if built and shot is not None:
                content_end = self._compute_content_end(
                    step, shot, obj_statuses, self.recipe, self._resolve_fps()
                )
                if content_end < shot.end:
                    shrinkable = shot.end - content_end

            results.append(
                StepStatus(
                    step_id=step.step_id,
                    built=built,
                    objects=obj_statuses,
                    additional_objects=additional,
                    shrinkable_frames=shrinkable,
                    dropped_behaviors=[list(pair) for pair in dropped],
                )
            )
        return results

    @staticmethod
    def _compute_content_end(
        step: BuilderStep,
        scene,
        obj_statuses: List[ObjectStatus],
        recipe=None,
        fps: Optional[float] = None,
    ) -> float:
        """Return the latest frame used by content in this step (an effect's
        length is the *recipe*'s, at *fps*)."""
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

        latest = scene.start  # at minimum, content starts at scene start
        for obj, obj_st in zip(step.objects, obj_statuses):
            for beh in obj.behaviors:
                try:
                    tmpl = Behaviors.keyed(beh, recipe, fps)
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
            filepath: Path to the CSV file, or an ``http(s)`` URL.
            store: Optional existing ``ShotStore`` to populate.
                If ``None``, a fresh instance is created.
            columns: Column index mapping.
            post_process: Optional callable invoked on each step after
                assembly.

        Returns:
            ``(builder, steps)`` tuple. Call ``builder.sync(steps)`` to
            execute.
        """
        steps = ManifestModel.parse_csv(filepath, columns, post_process=post_process)
        st = store or ShotStore.active()
        builder = cls(st)
        return builder, steps

    @staticmethod
    def resolve_duration(
        step: BuilderStep,
        initial_shot_length: float,
        fit_mode: FitMode,
        fps: float,
        measure_audio: Optional[Callable[[BuilderObject], Optional[float]]] = None,
        recipe=None,
    ) -> Tuple[float, float, float]:
        """Compute final shot duration for *step* under the given fit policy.

        Probes behavior templates and (via *measure_audio*) audio-clip lengths to
        determine the minimum content-driven length, then applies *fit_mode*
        against the user-specified *initial_shot_length* (default 200f).

        Parameters:
            step: The step whose objects drive the duration.
            initial_shot_length: Baseline length the fit policy is applied against.
            fit_mode: ``"extend_only"`` or ``"fit_contents"``.
            fps: Scene frame rate -- an effect template's lengths (the pulse's
                in seconds) become frames at it.  Audio is not probed here: a
                DCC layer binds its rate into *measure_audio*.
            measure_audio: Optional callable ``(BuilderObject) -> Optional[float]``
                returning an audio clip's length in frames, or ``None`` when it is
                unresolvable.  Injected by the DCC layer.  ``None`` (or a ``None`` /
                non-positive return) contributes no audio length, so a purely
                behavior-template step is sized entirely by its templates.
            recipe: The scene's :class:`~pythontk.EffectRecipe` (the defaults
                when omitted) -- what an effect template is sized from.

        Returns:
            ``(duration, behavior_span, audio_span)`` — the resolved shot
            length plus the individual content measurements that drove it.
        """
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

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
                    tmpl = Behaviors.keyed(b, recipe, fps)
                except FileNotFoundError:
                    continue
                if tmpl.get("duration") == "from_source":
                    continue  # handled via audio_span below
                d_in, d_out = Behaviors.phase_durations(tmpl)
                obj_in += d_in
                obj_out += d_out
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
