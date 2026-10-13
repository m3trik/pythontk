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
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

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
    PlannedInsert,
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
    :meth:`_audio_grow_duration`, :meth:`_placed_clip_keys`,
    :meth:`_sequencer`), each with a pure
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

    #: The owner tag on the timeline markers a build places
    #: (``ShotStore.marker_owner``): how it finds its own again.
    MARKER_OWNER = "shot_manifest"
    #: How a marked behavior's marker looks: set apart from a hand's (the
    #: sequencer's default triangle), and fixed -- the next build puts a
    #: moved one back.
    MARKER_LOOK = {
        "color": "#6FB1E0",
        "draggable": False,
        "style": "diamond",
        "line_style": "dotted",
        "opacity": 1.0,
    }

    #: Metadata keys the manifest writes on a shot (:meth:`_step_metadata`);
    #: a build replaces these and the doc's pass-through columns, and keeps
    #: every other key it finds.
    MANIFEST_METADATA = frozenset(
        ("section", "section_title", "csv_objects", "behaviors", "voice_text", "step")
    )

    def __init__(
        self,
        store: ShotStore,
        match: str = "name",
        rebuild_edited: bool = False,
        disabled_behaviors: Iterable[str] = (),
        marked_behaviors: Iterable[str] = (),
    ):
        """
        Parameters:
            store: The shots to build into and assess.
            match: How a step finds its shot (:meth:`pair`).
            rebuild_edited: Rebuild a shot edited by hand since its last build
                like any other (:meth:`edited_by_hand`).  Off by default: a
                build keeps such a shot as the hand left it.
            disabled_behaviors: Behavior names turned off: a build neither keys
                nor releases them and Assess does not report them; the keys
                they have stay.
            marked_behaviors: Behavior names a build marks instead of keying:
                a timeline marker where each would be keyed
                (:meth:`place_markers`), which Assess checks for.  Like a
                turned-off behavior it is neither keyed nor released.  A name
                also turned off is off.
        """
        self.store = store
        self.match = match
        self.rebuild_edited = bool(rebuild_edited)
        self.disabled_behaviors = frozenset(disabled_behaviors)
        self.marked_behaviors = frozenset(marked_behaviors) - self.disabled_behaviors
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

    @property
    def unkeyed_behaviors(self) -> frozenset:
        """The behavior names a build does not key: turned off or marked."""
        return self.disabled_behaviors | self.marked_behaviors

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

    def _sequencer(self):
        """The ``ShotSequencer`` over :attr:`store` that :meth:`insert_planned`
        inserts through (hook).

        Pure default: pythontk's, which moves shot bounds only.  DCC
        subclasses return their own, so the keys and audio of the shots an
        insert ripples move with them.
        """
        from pythontk.core_utils.engines.shots.shot_sequencer import ShotSequencer

        return ShotSequencer(store=self.store)

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
            ``(actions, behavior_result, assessment)`` tuple; the behavior
            result's ``"marked"`` lists what :meth:`place_markers` marked.
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
            behavior_result["marked"] = self.place_markers()

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
            | ``"locked"`` | ``"kept"`` | ``"removed"`` | ``"refused"``).
            ``"kept"`` is a shot edited by hand since its last build
            (:meth:`edited_by_hand`), left as it is.  ``"refused"``
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
        inserted step shifts nothing); then the shot of the step's ID -- or of
        its :attr:`~BuilderStep.shot_name`, when it was renamed -- unless that
        shot is bound to another step; then, with ``match="name_then_order"``,
        the remaining steps and shots in doc and timeline order -- a shot
        split off a step's shot left out.  A shot nothing pairs with is an
        orphan -- reported, never removed by a build -- and one split off a
        step's shot is linked to that step (``ShotPairing.links``,
        ``ShotStore.step_links``) without pairing.
        """
        shots = self.store.sorted_shots()
        by_binding: Dict[str, Any] = {}
        for shot in shots:
            bound = (shot.metadata or {}).get("step")
            if bound and bound not in by_binding:
                by_binding[bound] = shot
        by_name = {shot.name: shot for shot in shots}
        pairing = ShotPairing(timeline=list(shots))
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
        for step in steps:
            sid = step.step_id
            for name in dict.fromkeys((sid, step.shot_name)):
                shot = by_name.get(name)
                if sid in pairing.shots or shot is None or shot.shot_id in used:
                    continue
                if (shot.metadata or {}).get("step") not in (None, sid):
                    continue  # bound to another step
                take(sid, shot, "name")
        # Where a shot nothing pairs with belongs: a tail split off a step's
        # shot is in that step's section -- a link, never a pairing.
        links = self.store.step_links()
        if self.match == "name_then_order":
            # A linked shot is its step's tail, not another step's shot: the
            # order pass passes it by (maintainer 2026-10-12).
            todo = [sid for sid in ids if sid not in pairing.shots]
            free = [
                shot
                for shot in shots
                if shot.shot_id not in used and shot.shot_id not in links
            ]
            for sid, shot in zip(todo, free):
                take(sid, shot, "order")
        pairing.orphans = [shot for shot in shots if shot.shot_id not in used]
        pairing.links = {
            s.shot_id: links[s.shot_id] for s in pairing.orphans if s.shot_id in links
        }
        return pairing

    def bind(self, shot, step_id: Optional[str]) -> None:
        """Make *shot* step *step_id*'s shot -- the binding :meth:`pair`
        reads first -- or, with ``None``, bind it to no step.

        Another shot bound to *step_id* loses its binding: a step has one
        shot.  The user's call ("Pair with Step" on a shot the doc does not
        list): the shot is then the step's, built and assessed as it, under
        the name it has.  Wrap the call in ``store.scene_edit`` for one undo
        step.
        """
        for other in self.store.sorted_shots():
            bound = (other.metadata or {}).get("step")
            if other.shot_id == shot.shot_id:
                wanted = step_id
            elif step_id is not None and bound == step_id:
                wanted = None  # a step has one shot
            else:
                continue
            if bound == wanted:
                continue
            meta = dict(other.metadata or {})
            if wanted:
                meta["step"] = wanted
            else:
                meta.pop("step", None)
            self.store.update_shot(other.shot_id, metadata=meta)

    # ---- inserting into a built timeline -----------------------------------

    def _new_shot_duration(
        self, step: BuilderStep, initial_shot_length: float, fit_mode: FitMode
    ) -> float:
        """The length a build gives *step*'s new shot when no range is given:
        an audio step with no resolvable source takes a placeholder that
        grows once the clip loads; any other step takes *fit_mode* /
        *initial_shot_length* (:meth:`resolve_duration`)."""
        placeholder = _ShotManifestInternal._audio_placeholder_dur(
            step, measure_audio=self._measure_audio
        )
        if placeholder is not None:
            return placeholder
        dur, _beh, _aud = ShotManifest.resolve_duration(
            step,
            initial_shot_length,
            fit_mode,
            self._resolve_fps(),
            measure_audio=self._measure_audio,
            recipe=self.recipe,
        )
        return dur

    def plan_insertions(
        self,
        steps: List[BuilderStep],
        typed: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        fit_mode: FitMode = DEFAULT_FIT_MODE,
        initial_shot_length: float = DEFAULT_INITIAL_SHOT_LENGTH,
    ) -> List[PlannedInsert]:
        """Where each step with no shot goes on a built timeline: in sheet
        order, rippling every later shot.

        A new shot follows its previous step's shot and the shots after that
        no step pairs with (the step's split tails, shots the doc does not
        list); a step before every built one takes the timeline's head.  A
        step with a typed range is no insert: a Build gives it its shot at
        that range, in the timeline's current frames, so it is a row of the
        timeline there -- moved by the inserts before it, as a shot is
        (:meth:`ride_inserts`) -- and the next new step follows it.  Each
        insert is placed by :meth:`_sequencer`'s ``insert_placement`` -- the
        rule its ``insert_shot`` places by -- on the timeline the earlier
        inserts leave, so the plan's bounds are the ones
        :meth:`insert_planned`'s inserts leave.  Nothing is changed here: a
        caller can say what will move before it does.

        Parameters:
            steps: The steps to build, in sheet order.
            typed: ``{step_id: (start, end)}``, the ranges typed for steps
                with no shot, in the timeline's current frames (an end of
                ``None``: zero-length at the start).  Those steps are not
                inserted; a range for a step with a shot, or not in *steps*,
                is ignored.
            fit_mode: How a new shot is sized (:meth:`resolve_duration`).
            initial_shot_length: A new shot's length before fitting.

        Returns:
            One :class:`PlannedInsert` per step to insert, in sheet order, its
            bounds where it lands once every insert is in; none when no step
            has a shot -- a first build places every step itself.  A step
            whose name the store refuses (``ShotStore.name_error``) is left
            out for the build to report.
        """
        pairing = self.pair(steps)
        if not pairing.shots:
            return []
        store = self.store
        seq = self._sequencer()
        paired = {shot.shot_id for shot in pairing.shots.values()}
        taken = {shot.name.lower() for shot in store.sorted_shots()}
        # The timeline as the inserts will leave it, rows [key, start, end,
        # shot, at, rank]: key ("shot", shot_id), ("typed", step_id) or
        # ("insert", step_id); shot None but for an existing one; at the
        # current start of a row that is there now (a shot, a typed range),
        # else None.  Kept in the store's order -- by start, a tie by rank,
        # the order the shots were made in -- so each index is the one the
        # store gives the shot when the inserts are made.
        made_rank = len(store.shots)
        line = [
            [("shot", s.shot_id), s.start, s.end, s, s.start, n]
            for n, s in enumerate(store.shots)
        ]
        ids = {step.step_id for step in steps}
        for sid, (lo, hi) in (typed or {}).items():
            if sid in ids and sid not in pairing.shots and lo is not None:
                hi = lo if hi is None else hi
                line.append([("typed", sid), lo, hi, None, lo, float("inf")])
        line.sort(key=lambda row: (row[1], row[5]))

        def row_of(key) -> int:
            return next(i for i, row in enumerate(line) if row[0] == key)

        plans: List[PlannedInsert] = []
        by_step: Dict[str, PlannedInsert] = {}
        anchor = None  # the key of the row the next insert follows (None: the head)
        for step in steps:
            shot = pairing.shots.get(step.step_id)
            if shot is not None:
                at = row_of(("shot", shot.shot_id))
                while (
                    at + 1 < len(line)
                    and line[at + 1][0][0] == "shot"
                    and line[at + 1][0][1] not in paired
                ):
                    at += 1
                anchor = line[at][0]
                continue
            if any(row[0] == ("typed", step.step_id) for row in line):
                anchor = ("typed", step.step_id)  # its shot is made at its range
                continue
            name = step.shot_name
            if name.lower() in taken or store.name_error(name):
                continue
            taken.add(name.lower())
            dur = self._new_shot_duration(step, initial_shot_length, fit_mode)
            idx = 0 if anchor is None else row_of(anchor) + 1
            start, end, push, moved = seq.insert_placement(
                idx,
                dur,
                gap=store.gap,
                line=[(r[1], r[2], r[3], r[0][0] != "typed") for r in line],
            )
            fixed = [line[i][4] for i in moved if line[i][4] is not None]
            # What it goes after when it is made: the rows before it now,
            # nearest first, back to an existing shot.
            follows = []
            for prev in reversed(line[:idx]):
                follows.append(prev[0])
                if prev[0][0] == "shot":
                    break
            for i in moved:
                line[i][1] = store.snap(line[i][1] + push)
                line[i][2] = store.snap(line[i][2] + push)
            plan = PlannedInsert(
                step=step,
                start=start,
                end=end,
                duration=dur,
                follows=tuple(follows),
                pivot=min(fixed) if fixed else None,
                push=push,
            )
            plans.append(plan)
            by_step[step.step_id] = plan
            line.insert(
                idx, [("insert", step.step_id), start, end, None, None, made_rank]
            )
            made_rank += 1
            line.sort(key=lambda row: (row[1], row[5]))
            anchor = ("insert", step.step_id)
        # Each where it lands once every insert is in (a later one placed
        # before it moved it).
        for row in line:
            if row[0][0] == "insert":
                plan = by_step[row[0][1]]
                plan.start, plan.end = row[1], row[2]
        return plans

    def insert_planned(
        self,
        plans: List[PlannedInsert],
        typed: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> Dict[str, str]:
        """Insert :meth:`plan_insertions`' shots, each bound to its step and
        recorded as a build's (the step's metadata and :attr:`BUILD_NAME`,
        as a first build writes them).

        Each goes through :meth:`_sequencer`'s ``insert_shot``: the shots
        after it -- with their keys and audio, on a DCC host -- move out of
        its way.  Each goes after the nearest shot before it on the plan's
        timeline that exists by then (``PlannedInsert.follows``), so any of a
        plan's inserts, given in sheet order, go where the plan put them; one
        that directly follows a typed range starts after where that range
        lands, as the inserts made so far move it (:meth:`ride_inserts`).
        Wrap the call in ``store.scene_edit`` for one undo step.

        Parameters:
            plans: Inserts from :meth:`plan_insertions` (all, or some of
                them), in sheet order.
            typed: The typed ranges the plan was made with
                (:meth:`plan_insertions`' *typed*); without them an insert
                after a typed range follows the shot before it.

        Returns:
            ``{step_id: "created" | "refused"}``; a refused insert (the store
            would not take its name, or the shot it follows is gone) moves
            nothing.
        """
        seq = self._sequencer()
        made_ids: Dict[str, int] = {}  # step id -> the shot its insert made
        made: List[PlannedInsert] = []
        actions: Dict[str, str] = {}
        for plan in plans:
            step = plan.step
            after, start = None, None
            for kind, key in plan.follows:
                if kind == "shot":
                    after = key
                    break
                if kind == "insert":
                    if key in made_ids:
                        after = made_ids[key]
                        break
                    continue  # not made: what comes before it
                if start is None and typed and typed.get(key):
                    lo, hi = self.ride_inserts({key: typed[key]}, made)[key]
                    if lo is not None:
                        start = (lo if hi is None else hi) + self.store.gap
            try:
                block = seq.insert_shot(
                    name=step.shot_name,
                    duration=plan.duration,
                    after_shot_id=after,
                    at_position=1 if after is None else None,  # the head
                    description=step.display_text or "",
                    start=start,
                )
            except ValueError as exc:
                log.warning("Step %s not built: %s", step.shot_name, exc)
                actions[step.step_id] = "refused"
                continue
            # One write binds it (the metadata names its step) and records it
            # as built; bind() then frees any other shot claiming the step.
            meta = {**(block.metadata or {}), **self._built_metadata(step)}
            self.store.update_shot(block.shot_id, metadata=meta)
            self.bind(block, step.step_id)
            made_ids[step.step_id] = block.shot_id
            made.append(plan)
            actions[step.step_id] = "created"
        return actions

    @staticmethod
    def ride_inserts(
        ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
        plans: List[PlannedInsert],
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """*ranges* where they stand once *plans*' inserts are in.

        A range typed in the timeline's current frames is content like any
        other: one starting at or after an insert's pivot
        (``PlannedInsert.pivot``) moves by its push, insert by insert
        (maintainer 2026-10-12).

        Parameters:
            ranges: ``{key: (start, end)}`` in the timeline's current frames;
                an end may be ``None`` (a start alone), a start ``None`` (no
                place), which stays.
            plans: :meth:`plan_insertions`' inserts.

        Returns:
            A new dict of the moved ranges; *ranges* is left as it is.
        """
        moved: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for key, (start, end) in ranges.items():
            push = 0.0
            if start is not None:
                push = sum(
                    p.push
                    for p in plans
                    if p.pivot is not None and start >= p.pivot - 1e-6
                )
            if not push:
                moved[key] = (start, end)
                continue
            moved[key] = (start + push, None if end is None else end + push)
        return moved

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
            if shot is None or self._kept(shot):
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
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

        led = self.store.edit_ledger
        n = 0
        for shot in self._behavior_shots():
            for entry in (shot.metadata or {}).get("behaviors", ()):
                name, behavior = entry.get("name", ""), entry.get("behavior", "")
                if behavior in self.unkeyed_behaviors:
                    continue
                if not (name and behavior and Behaviors.is_audio_entry(entry)):
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

        listed = list(obj.behaviors or [])
        off = self.unkeyed_behaviors
        behaviors = [b for b in listed if b not in off]
        claimed = {
            b
            for o, b in self.store.edit_ledger.authored_pairs(shot.shot_id)
            if o == obj.name and b not in off
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
        # Anchors over the whole list: a behavior not keyed keeps its place.
        anchors = Behaviors.anchor_overrides(listed)
        for behavior, anchor in zip(listed, anchors):
            if behavior in off:
                continue
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

    # ---- behavior markers ------------------------------------------------

    def _mark_frame(self, shot, behavior: str, anchor: Any = None) -> Optional[float]:
        """The frame *behavior* is marked at on *shot*: where its keys would
        start (:meth:`Behaviors.key_window`); ``None`` for one keying nothing."""
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

        window = Behaviors.key_window(
            behavior, shot.start, shot.end, anchor, self.recipe
        )
        return None if window is None else window[0]

    def _is_mark(self, marker: Dict[str, Any], shot_ids: Collection[int]) -> bool:
        """Whether *marker* is one this manifest placed on one of *shot_ids*."""
        return (
            ShotStore.marker_owner(marker) == self.MARKER_OWNER
            and marker["data"].get("shot_id") in shot_ids
        )

    def planned_markers(self, shots: Optional[list] = None) -> List[Dict[str, Any]]:
        """The markers a build places for :attr:`marked_behaviors`: per shot,
        one per behavior and frame -- where its keys would start, each object
        placed as a build places it (:meth:`Behaviors.anchor_overrides`) --
        naming the objects it marks.

        Parameters:
            shots: The shots to mark; default the ones a build reaches
                (:meth:`_behavior_shots`).  A zero-length shot gets none, as
                it gets no keys.
        """
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

        marked = self.marked_behaviors
        if not marked:
            return []
        out: List[Dict[str, Any]] = []
        for shot in self._behavior_shots() if shots is None else shots:
            if abs(shot.end - shot.start) < 1e-6:
                continue
            entries = [
                e
                for e in (shot.metadata or {}).get("behaviors", ())
                if e.get("name") and e.get("behavior")
            ]
            groups: Dict[Tuple[float, str], List[str]] = {}
            # Placed as the build places them (Behaviors.entry_anchors).
            for entry, anchor in zip(entries, Behaviors.entry_anchors(entries)):
                name, behavior = entry["name"], entry["behavior"]
                if behavior not in marked:
                    continue
                frame = self._mark_frame(shot, behavior, anchor)
                if frame is None:
                    continue
                names = groups.setdefault((frame, behavior), [])
                if name not in names:
                    names.append(name)
            for (frame, behavior), names in sorted(groups.items()):
                out.append(
                    dict(
                        self.MARKER_LOOK,
                        time=frame,
                        note=f"{Behaviors.label(behavior)}: {', '.join(names)}",
                        data={
                            "owner": self.MARKER_OWNER,
                            "shot_id": shot.shot_id,
                            "behavior": behavior,
                            "objects": names,
                        },
                    )
                )
        return out

    def place_markers(self) -> List[Dict[str, Any]]:
        """Mark where the :attr:`marked_behaviors` would be keyed, on the
        shots a build reaches (:meth:`_behavior_shots`).

        This manifest's markers on those shots go first -- all but a
        turned-off behavior's, which a build leaves as it leaves its keys --
        then :meth:`planned_markers` are placed: a behavior keyed again,
        dropped from the doc or moved with its shot loses its old marker.  A
        hand's markers, and those on the shots a build keeps, are never
        touched.

        Returns:
            One ``{object, behavior, shot, shot_id, time}`` record per object
            marked.
        """
        shots = self._behavior_shots()
        ids = {s.shot_id for s in shots}
        off = self.disabled_behaviors
        planned = self.planned_markers(shots)
        self.store.set_markers(
            [
                m
                for m in self.store.markers
                if not (self._is_mark(m, ids) and m["data"].get("behavior") not in off)
            ]
            + planned
        )
        names = {s.shot_id: s.name for s in shots}
        return [
            {
                "object": name,
                "behavior": m["data"]["behavior"],
                "shot": names[m["data"]["shot_id"]],
                "shot_id": m["data"]["shot_id"],
                "time": m["time"],
            }
            for m in planned
            for name in m["data"]["objects"]
        ]

    def _marker_placed(self, shot, obj: str, behavior: str, anchor: Any = None) -> bool:
        """Whether *behavior*'s marker naming *obj* is on *shot* where a build
        places it (:meth:`planned_markers`): Assess's check of a marked
        behavior.  A zero-length shot has nothing to mark."""
        if abs(shot.end - shot.start) < 1e-6:
            return True
        frame = self._mark_frame(shot, behavior, anchor)
        if frame is None:
            return False
        return any(
            self._is_mark(m, (shot.shot_id,))
            and m["data"].get("behavior") == behavior
            and obj in (m["data"].get("objects") or ())
            and abs(float(m.get("time", 0.0)) - frame) < 0.5
            for m in self.store.markers
        )

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
                        dur = self._new_shot_duration(
                            step, initial_shot_length, fit_mode
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
            if self._kept(existing):
                plan.append(
                    PlannedShot(
                        step=step,
                        action="locked" if existing.locked else "kept",
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
            old_csv_objs, old_audio = self._recorded_doc(existing)
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
            if "csv_objects" not in (existing.metadata or {}):
                # A shot no build made (the Shot Sequencer's, or split off
                # there): its members are the scene's, none a previous doc's
                # to drop -- read as doc objects, a step paired with it
                # stripped every member it did not list.
                dropped = set()
                scene_discovered = {
                    n for n in existing.objects if key(n) not in csv_keys
                }
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

            audio_changed = new_audio != old_audio
            renamed = self._new_name(existing, step) is not None

            has_content_change = bool(new_objs or dropped or changed_beh or absent)

            if has_content_change or repositioned or audio_changed or renamed:
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

    #: Shot metadata recording the name a build last gave the shot.  A shot
    #: still so named is the build's to rename -- back to its step's ID once
    #: the step's :attr:`~BuilderStep.name` is gone; one renamed by hand since
    #: keeps the hand's name.
    BUILD_NAME = "build_name"

    #: Shot metadata recording the shot as the last build left it
    #: (:meth:`_build_record`): what tells a shot edited by hand since.
    BUILT = "built"

    @staticmethod
    def _build_record(shot) -> Dict[str, Any]:
        """What a build leaves on *shot* that only a hand changes: its length
        (a ripple moves a shot, it does not resize it), its members, its
        description.  The name has its own record (:attr:`BUILD_NAME`)."""
        return {
            "length": round(shot.end - shot.start, 3),
            "members": sorted({ShotStore.member_key(n) for n in shot.objects}),
            "description": shot.description or "",
        }

    def edited_by_hand(self, shot) -> bool:
        """Whether *shot* was edited since a build last left it: renamed,
        resized, its members or its description changed.

        A member the shot system added on its own is no hand's edit
        (``ShotStore.SYSTEM_MEMBERS``: the sequencer filling an empty shot
        when it shows it, an object keyed in the active shot).  Read as one,
        opening a built step with no -- or missing -- objects in the Shot
        Sequencer made the shot "edited", and every later Build kept it, doc
        changes and all (BACKLOG 2026-10-10).  A member a hand adds or
        removes still counts.

        A shot a build made before this record existed reads as unedited,
        save a rename (the name has had its record longer).
        """
        meta = shot.metadata or {}
        built_name = meta.get(self.BUILD_NAME)
        if built_name is not None and shot.name != built_name:
            return True
        record = meta.get(self.BUILT)
        if record is None:
            return False
        current = self._build_record(shot)
        tagged = set(meta.get(ShotStore.SYSTEM_MEMBERS) or ())
        if tagged:
            built = set(record.get("members") or ())
            current["members"] = [
                k for k in current["members"] if k not in tagged or k in built
            ]
        return record != current

    def _built_metadata(
        self, step: BuilderStep, meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """What a build writes on a shot it makes for *step*: the step's
        metadata (*meta*, else :meth:`_step_metadata` with the step's
        pass-through columns) and the name it gives the shot
        (:attr:`BUILD_NAME`) -- a first build's and an insert's alike."""
        if meta is None:
            meta = self._step_metadata(
                step, pass_through=getattr(step, "_pass_through", None)
            )
        return {**meta, self.BUILD_NAME: step.shot_name}

    def _kept(self, shot) -> bool:
        """Whether a build leaves *shot* as it is: locked, or edited by hand
        and not asked to rebuild edited shots (2026-10-07: "make sure that
        the manifest does not override edited shots on re-build by
        default")."""
        return shot.locked or (not self.rebuild_edited and self.edited_by_hand(shot))

    def _behavior_shots(self) -> list:
        """The shots a build's behavior pass reaches: every one it does not
        keep (:meth:`_kept`)."""
        return [s for s in self.store.sorted_shots() if not self._kept(s)]

    def _record_build(self, shot_ids: Iterable[int]) -> None:
        """Write :attr:`BUILT` on each of *shot_ids*, as the build left it."""
        for sid in shot_ids:
            shot = self.store.shot_by_id(sid)
            if shot is None:
                continue
            record = self._build_record(shot)
            if (shot.metadata or {}).get(self.BUILT) != record:
                meta = dict(shot.metadata or {})
                meta[self.BUILT] = record
                self.store.update_shot(sid, metadata=meta)

    def _new_name(self, existing, step: BuilderStep) -> Optional[str]:
        """The name a build gives *existing*, *step*'s shot, or ``None`` to
        leave it: a renamed step's name always; the step's ID only while the
        shot still has the name a build gave it (:attr:`BUILD_NAME`), so a
        rename made in the sequencer survives a build -- unless the build
        rebuilds edited shots, which names a hand-renamed shot back.  A shot
        no build named (one bound by "Pair with Step") keeps its own name."""
        target = step.shot_name
        if existing.name == target:
            return None
        built = (existing.metadata or {}).get(self.BUILD_NAME)
        if not step.name and existing.name != built:
            if built is None or not self.rebuild_edited:
                return None
        return target

    def _rename_kwargs(self, existing, ps: PlannedShot) -> Dict[str, Any]:
        """``name`` kwarg when the build renames the shot (:meth:`_new_name`),
        else empty.  A name the store refuses (``ShotStore.name_error``:
        another shot holds it) is logged and the shot keeps its name."""
        name = self._new_name(existing, ps.step)
        if name is None:
            return {}
        refused = self.store.name_error(name, shot_id=existing.shot_id)
        if refused:
            log.warning("Shot %s not renamed to %s: %s", existing.name, name, refused)
            return {}
        return {"name": name}

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
                    refused = self.store.name_error(ps.step.shot_name)
                    if refused:
                        log.warning("Step %s not built: %s", ps.step.shot_name, refused)
                        actions[ps.step.step_id] = "refused"
                        continue
                    # Store resolved (long / unique) names; missing objects
                    # keep their CSV form so pinning can surface them later.
                    obj_names = self._resolve_names_keep_missing(ps.objects)
                    self.store.define_shot(
                        name=ps.step.shot_name,
                        start=ps.start,
                        end=ps.end,
                        objects=obj_names,
                        metadata=self._built_metadata(ps.step, ps.metadata),
                        description=ps.description,
                    )
                    for n in obj_names:
                        self.store.set_object_pinned(n)
                    actions[ps.step.step_id] = "created"

                elif ps.action in ("locked", "kept", "skipped", "patched"):
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
                        # displaced the shot; locked and kept shots are
                        # content-protected, so metadata/description stay.
                        kwargs = self._reposition_kwargs(existing, ps)
                        if ps.action not in ("locked", "kept"):
                            kwargs.update(self._content_kwargs(existing, ps))
                            kwargs.update(self._rename_kwargs(existing, ps))
                        if "name" in kwargs:
                            meta = dict(kwargs.get("metadata", existing.metadata or {}))
                            meta[self.BUILD_NAME] = kwargs["name"]
                            kwargs["metadata"] = meta

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

            # What this build left on every shot it built: the next one tells
            # a hand's edit by it (edited_by_hand).
            built = []
            for ps in plan:
                if actions.get(ps.step.step_id) not in (
                    "created",
                    "patched",
                    "skipped",
                ):
                    continue
                if ps.existing_shot_id is not None:
                    built.append(ps.existing_shot_id)
                else:
                    shot = self.store.shot_by_name(ps.step.shot_name)
                    if shot is not None:
                        built.append(shot.shot_id)
            self._record_build(built)

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

        off = self.disabled_behaviors
        marked = self.marked_behaviors
        results: List[StepStatus] = []
        for step in steps:
            shot = pairing.shots.get(step.step_id)
            built = shot is not None
            is_locked = built and shot.locked
            edited = built and not is_locked and self._kept(shot)

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
                            if b not in off
                            and not (
                                self._marker_placed(shot, obj.name, b)
                                if b in marked
                                else verify_fn(obj.name, b, shot.start, shot.end)
                            )
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
                        if b in unknown or b in off:
                            continue
                        if b in marked:
                            if not self._marker_placed(shot, obj.name, b, anchor):
                                failed.append(b)
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
                        and b not in off
                        and b not in marked
                        and self.is_stale(shot.shot_id, obj.name, b)
                    ]
                    # Over keys the animator owns, Build's guard leaves a
                    # behavior as it is -- unsatisfied or stale alike, so it
                    # is a conflict, not a fix.
                    conflicts = [
                        b
                        for b in failed + stale
                        if b not in marked
                        and self.unowned_keys(node, b, shot.start, shot.end)
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
                    step, shot, obj_statuses, self.recipe
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
                    edited=edited,
                    updates=self._updates(shot, step) if built and not edited else [],
                )
            )
        return results

    def _updates(self, shot, step: BuilderStep) -> List[str]:
        """What a build rewrites on *shot*, *step*'s, that no object status
        shows (``StepStatus.updates``): a new name the store takes
        (:meth:`_new_name`), a new description, an object or audio clip the
        last build recorded and the step no longer lists."""
        updates = []
        name = self._new_name(shot, step)
        if name is not None and not self.store.name_error(name, shot.shot_id):
            updates.append("name")
        if (shot.description or "") != (step.display_text or ""):
            updates.append("description")
        if "csv_objects" in (shot.metadata or {}):
            key = ShotStore.member_key
            objs, audio = self._recorded_doc(shot)
            listed = {key(o.name) for o in step.objects if o.kind != "audio"}
            clips = {o.name for o in step.objects if o.kind == "audio"}
            if any(key(n) not in listed for n in objs) or audio != clips:
                updates.append("objects")
        return updates

    @staticmethod
    def _recorded_doc(shot) -> Tuple[Set[str], Set[str]]:
        """``(objects, audio)``: the doc's names the last build recorded on
        *shot* (``csv_objects``), its members when no build did."""
        objs, audio = set(), set()
        for entry in (shot.metadata or {}).get("csv_objects", shot.objects):
            if isinstance(entry, dict) and entry.get("kind") == "audio":
                audio.add(entry["name"])
            else:
                objs.add(entry["name"] if isinstance(entry, dict) else entry)
        return objs, audio

    @staticmethod
    def _compute_content_end(
        step: BuilderStep,
        scene,
        obj_statuses: List[ObjectStatus],
        recipe=None,
    ) -> float:
        """Return the latest frame used by content in this step: each
        behavior's last key where the build puts it -- an object's point
        behaviors spread over the range (:meth:`Behaviors.anchor_overrides`,
        the one placement rule) and each placed by
        :meth:`Behaviors.key_window` (an effect by the *recipe*)."""
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

        latest = scene.start  # at minimum, content starts at scene start
        for obj, obj_st in zip(step.objects, obj_statuses):
            names = list(obj.behaviors)
            for beh, anchor in zip(names, Behaviors.anchor_overrides(names)):
                window = Behaviors.key_window(
                    beh, scene.start, scene.end, anchor, recipe
                )
                if window is not None and window[1] > latest:
                    latest = window[1]
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
