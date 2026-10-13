# coding=utf-8
"""DCC-agnostic shot data model and persistent store.

Provides :class:`ShotBlock` (the fundamental shot data structure) and
:class:`ShotStore` (CRUD + observer + pluggable persistence).  This is the
shared foundation both mayatk and blendertk consume: the model is engine-free,
and every place that would otherwise reach into a DCC scene is exposed as an
**overridable hook with a pure default** (see :meth:`ShotStore._scene_fps`,
:meth:`ShotStore.has_animation`, :meth:`ShotStore.detect_regions`,
:meth:`ShotStore.assess`, :meth:`ShotStore.publish_export_view`,
:meth:`ShotStore._resolve_long_names`, :meth:`ShotStore._existing_objects`,
:meth:`ShotStore._keyed_windows` and :meth:`ShotStore._curve_ledger_keys`).
DCC toolkits subclass ``ShotStore`` and override those hooks; the pure core
never imports ``maya`` or ``bpy``.

Persistence is pluggable: call :meth:`ShotStore.set_persistence` with a backend
that implements ``save(data)`` / ``load() -> dict | None``.  Absent a backend
the store runs in pure in-memory mode.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)
from contextlib import contextmanager

from pythontk.core_utils.engines.shots.effect_recipe import EffectRecipe
from pythontk.core_utils.engines.shots.shot_ledger import ShotEditLedger
from pythontk.core_utils.engines.scene_export.export_snapshot import ExportContext
from pythontk.core_utils.engines.scene_export.scene_records import Record, SceneRecords
from pythontk.str_utils._str_utils import StrUtils

_log = logging.getLogger(__name__)

_DEFAULT_FPS = 24.0

#: How far inside its bounds a key has to sit to count as a shot's own when
#: telling an empty shot (``ShotStore.empty_shots``): at a zero gap the frame a
#: shot starts on is the last of the shot before it, keys and all.
_BOUND_TOL = 1e-3


__all__ = [
    "SHOT_PALETTE",
    "SHOT_STATUS_LEVELS",
    "ShotBlock",
    "ShotStore",
    "StoreEvent",
    "ShotDefined",
    "ShotUpdated",
    "ShotRemoved",
    "ActiveShotChanged",
    "SettingsChanged",
    "BatchComplete",
    "ShotsEdited",
    "MarkersChanged",
    "StoreInvalidated",
    "NativeStepFollowed",
    "ScenePersistence",
    "ShotEditLedger",
]


# ---------------------------------------------------------------------------
# Persistence protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ScenePersistence(Protocol):
    """Interface for saving / loading ShotStore data."""

    def save(self, data: Dict[str, Any]) -> None: ...

    def load(self) -> Optional[Dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Shared shot palette (single source of truth for both UIs)
# ---------------------------------------------------------------------------

#: The severity each shot status is shown at -- a level of
#: ``Palette.status()`` (valid, locked, info, warn, error) -- so a UI can
#: mark it by shape as well as colour.  ``SHOT_PALETTE`` colours by it.
SHOT_STATUS_LEVELS: Dict[str, str] = {
    "csv_object": "valid",  # expected — no color
    "scene_discovered": "info",  # found in scene, not in CSV
    "missing_object": "error",  # referenced but missing
    "missing_behavior": "warn",  # expected behaviour keys absent
    "stale_behavior": "info",  # keyed under an older effect recipe
    "user_animated": "info",  # custom user animation detected
    "additional": "warn",  # unexpected scene objects
    "collision": "error",  # timing overlap
    "missing_shot": "info",  # shot not yet built
    "not_in_shot": "warn",  # in the scene, not a member of its shot
    "ambiguous_object": "warn",  # the doc name matches several nodes
    "behavior_conflict": "error",  # animator keys where a behavior keys
    "unknown_behavior": "warn",  # no behavior template by that name
    "not_in_doc": "info",  # a shot no doc step pairs with
    "no_objects": "info",  # the doc lists no objects for the step
    "edited": "locked",  # edited by hand since its build: Build keeps it
}

try:
    from pythontk import Palette

    SHOT_PALETTE = Palette.status().alias(SHOT_STATUS_LEVELS)
except ImportError:
    SHOT_PALETTE = {}  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data Structure
# ---------------------------------------------------------------------------


@dataclass
class ShotBlock:
    """Represents a single shot (contiguous animation range).

    Attributes:
        shot_id: Unique identifier for the shot.
        name: Human-readable label (e.g. "Intro", "Shot_1").
        start: First frame of the shot.
        end: Last frame of the shot.
        objects: Transform node names that belong to this shot.
        metadata: Arbitrary key/value pairs (section, behaviors, …).
        locked: If True, the shot has been finalized by the user and
            should not be flagged/modified by automated assessment.
        description: Free-text description of the shot content.
    """

    shot_id: int
    name: str
    start: float
    end: float
    objects: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    locked: bool = False
    description: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def classify_objects(self) -> Dict[str, str]:
        """Return ``{obj_name: status_key}`` using stored metadata.

        Resolution order per object:

        1. ``metadata["object_status"][obj]`` — written by manifest
           assessment (richest: missing_object / missing_behavior /
           user_animated / valid).
        2. ``metadata["csv_objects"]`` membership — if present and the
           object is *not* listed → ``"scene_discovered"``.
        3. Fallback → ``"valid"``.

        Both the manifest and sequencer use this method so that
        classification logic lives in one place.
        """
        statuses = self.metadata.get("object_status", {})
        raw_csv = self.metadata.get("csv_objects", [])
        csv_objs = set((e["name"] if isinstance(e, dict) else e) for e in raw_csv)
        # Metadata holds the doc's names; shot.objects hold resolved (long,
        # possibly namespaced) nodes -- compare through the one identity rule.
        key = ShotStore.member_key
        status_by_key = {key(k): v for k, v in statuses.items()}
        csv_keys = {key(n) for n in csv_objs}
        result: Dict[str, str] = {}
        for obj in self.objects:
            if obj in statuses:
                result[obj] = statuses[obj]
            elif key(obj) in status_by_key:
                result[obj] = status_by_key[key(obj)]
            elif csv_objs and key(obj) not in csv_keys:
                result[obj] = "scene_discovered"
            else:
                result[obj] = "valid"
        return result


# ---------------------------------------------------------------------------
# Export-view helpers (shot → FBX takes + Unity metadata)
# ---------------------------------------------------------------------------

CLIP_NAME_STRATEGIES: Dict[str, Callable[[int, "ShotBlock"], str]] = {
    "name": lambda i, shot: shot.name,
    "sequence": lambda i, shot: f"{(i + 1) * 10:03d}_{shot.name}",
}


# ---------------------------------------------------------------------------
# Store events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoreEvent:
    """Base class for typed :class:`ShotStore` events.

    Each subclass carries event-specific payload as typed fields.
    The ``name`` class variable matches the legacy string event name
    for backward compatibility with the Qt ``app_event`` signal.
    """

    name: ClassVar[str] = ""


@dataclass(frozen=True)
class ShotDefined(StoreEvent):
    """A new shot was created and added to the store."""

    name: ClassVar[str] = "shot_defined"
    shot: ShotBlock


@dataclass(frozen=True)
class ShotUpdated(StoreEvent):
    """An existing shot's fields were modified."""

    name: ClassVar[str] = "shot_updated"
    shot: ShotBlock


@dataclass(frozen=True)
class ShotRemoved(StoreEvent):
    """A shot was removed from the store."""

    name: ClassVar[str] = "shot_removed"
    shot_id: int = 0


@dataclass(frozen=True)
class ActiveShotChanged(StoreEvent):
    """The active (selected) shot changed.

    *previous* is the shot it replaced (``None``: there was none) -- what
    tells a pick that moved from one shot to another from one that only
    filled an empty slot (a panel selecting the first shot once the active
    one was deleted).
    """

    name: ClassVar[str] = "active_shot_changed"
    shot_id: Optional[int] = None
    previous: Optional[int] = None


@dataclass(frozen=True)
class SettingsChanged(StoreEvent):
    """Detection-relevant settings were modified."""

    name: ClassVar[str] = "settings_changed"


@dataclass(frozen=True)
class BatchComplete(StoreEvent):
    """A :meth:`ShotStore.batch_update` context has exited."""

    name: ClassVar[str] = "batch_complete"


@dataclass(frozen=True)
class ShotsEdited(StoreEvent):
    """A :meth:`ShotStore.scene_edit` -- one step of the scene's undo --
    changed the shot records.

    The one event every panel can follow for an edit made anywhere: the
    engine's bound edits (trim, resize, slide, ripple, respace, add space)
    write the shots directly, so this is the only word a panel that did not
    make the edit gets of them.  *label* names the edit.
    """

    name: ClassVar[str] = "shots_edited"
    label: str = ""


@dataclass(frozen=True)
class MarkersChanged(StoreEvent):
    """:meth:`ShotStore.set_markers` changed the timeline markers -- a tool's
    write (the manifest's Build), not a marker edit the sequencer made itself.
    """

    name: ClassVar[str] = "markers_changed"


@dataclass(frozen=True)
class StoreInvalidated(StoreEvent):
    """The active store was discarded (scene change / new scene).

    Listeners should rebind to the new :meth:`ShotStore.active` instance.
    Fired on class-level invalidation listeners registered via
    :meth:`ShotStore.add_invalidation_listener`.
    """

    name: ClassVar[str] = "store_invalidated"


@dataclass(frozen=True)
class NativeStepFollowed(StoreEvent):
    """The DCC's own undo (or redo, *redo*) ran and the active *store*
    followed it (:meth:`ShotStore.follow_native_step`), putting back *applied*
    restore points -- none when the step was somebody else's (*store* is
    ``None`` when no store was active).

    Fired on the class-level listeners of
    :meth:`ShotStore.add_native_step_listener`, after the store's own events:
    the one word a panel needs to redraw after a Ctrl+Z made anywhere.
    """

    name: ClassVar[str] = "native_step_followed"
    redo: bool = False
    applied: int = 0
    store: Any = field(default=None, compare=False)


# ---------------------------------------------------------------------------
# ShotStore
# ---------------------------------------------------------------------------


class _ShotStoreInternal(object):
    """Internal helpers for ShotStore."""

    # ---- the claim hooks (``ShotStore.release_replaced`` / ``remap_moved``) --

    def _release_through(self, keys_of) -> Callable[[Any, Any], int]:
        """The ``on_replace(curve, frames)`` :meth:`ShotStore.release_replaced`
        returns, resolving a curve's ledger keys through *keys_of*."""

        def on_replace(curve, frames) -> int:
            ledger = self.edit_ledger
            if not ledger:  # nothing claimed: no curve's keys need resolving
                return 0
            if frames is not None:
                frames = list(frames)  # read once per ledger key, not once in all
            if frames is None:  # the curve is gone: all of it
                released = sum(
                    ledger.release(key, -math.inf, math.inf) for key in keys_of(curve)
                )
            else:
                released = sum(
                    ledger.release(key, t) for key in keys_of(curve) for t in frames
                )
            if released:
                self.mark_dirty()
            return released

        return on_replace

    def _remap_through(self, keys_of) -> Callable[[Any, Any], int]:
        """The ``on_move(curve, pairs)`` :meth:`ShotStore.remap_moved` returns,
        resolving a curve's ledger keys through *keys_of*."""

        def on_move(curve, pairs) -> int:
            ledger = self.edit_ledger
            if not ledger:  # nothing claimed: no curve's keys need resolving
                return 0
            pairs = list(pairs)
            moved = sum(ledger.remap(key, pairs) for key in keys_of(curve))
            if moved:
                self.mark_dirty()
            return moved

        return on_move

    @staticmethod
    def _spans_overlap(
        spans: Iterable[Tuple[float, Optional[float]]],
        windows: List[Tuple[float, float]],
    ) -> List[bool]:
        """Per window, whether any ``(start, stop)`` span overlaps it -- the DCC
        audio hooks' shared half (:meth:`ShotStore._audio_windows`); a
        ``stop`` of ``None`` plays on past its start, so only its start has to
        fall before the window's end."""
        spans = list(spans)
        return [
            any(s <= b and (e is None or e >= a) for s, e in spans) for a, b in windows
        ]

    @staticmethod
    def _sanitize_clip_name(name: str) -> str:
        """*name* as the clip name every carrier writes.

        A legal name (:attr:`ShotStore.NAME_PATTERN`) is returned VERBATIM --
        that is the contract the store's name validation rests on: whatever
        the tools accept arrives unchanged.  Only a name the store holds from
        before names were validated (a legacy scene, a hand-edited record) is
        respelled, by the legacy rule (alphanumeric + ``_``, case preserved,
        runs of ``_`` collapsed, trailing ones stripped) so those scenes keep
        the clip names their earlier deliverables shipped with.
        """
        if StrUtils.is_legal_name(name):
            return name
        clean = StrUtils.sanitize(name or "", preserve_case=True)
        return clean or "shot"

    @staticmethod
    def _clip_key(name: Any) -> str:
        """The identity two names collide on: their clip spelling, case-folded.

        Compared this way -- not raw -- a legal name cannot be accepted beside
        a legacy one that EXPORTS as it (``Shot_1`` beside ``"Shot 1"``), and
        ``Intro`` / ``intro`` are the one clip Unity's case-insensitive join
        makes them.
        """
        return _ShotStoreInternal._sanitize_clip_name(str(name)).casefold()

    def _check_name(self, name: Any, shot_id: Optional[int] = None) -> None:
        """Raise ``ValueError`` when :meth:`ShotStore.name_error` has a reason."""
        error = self.name_error(name, shot_id)
        if error:
            raise ValueError(error)


class ShotStore(_ShotStoreInternal):
    """Central store for shot data with pluggable persistence.

    The pure, DCC-agnostic core.  Scene-reaching behaviour lives in the
    overridable hooks (:meth:`_scene_fps`, :meth:`has_animation`,
    :meth:`detect_regions`, :meth:`assess`, :meth:`publish_export_view`,
    :meth:`_resolve_long_names`, :meth:`_existing_objects`,
    :meth:`_keyed_windows`, :meth:`_members_keyed_windows`,
    :meth:`_audio_windows`, :meth:`_curve_ledger_keys`, :meth:`member_nodes`,
    :meth:`scene_edit`, :meth:`_schedule_flush`), each with a pure default;
    mayatk / blendertk subclass this and override them.

    Parameters:
        shots: Initial shot list.  Copied on construction.
    """

    _active: Optional["ShotStore"] = None
    _persistence: Optional[ScenePersistence] = None
    _invalidation_listeners: ClassVar[List[Callable[["StoreInvalidated"], None]]] = []
    # Class-level, like the invalidation listeners: they outlive a store swap.
    _native_step_listeners: ClassVar[List[Callable[["NativeStepFollowed"], None]]] = []
    # Depth of :meth:`native_undo_sync_held` blocks.  Held on THIS class, not
    # per subclass: a process has one native undo queue, whichever store
    # class listens to it.
    _native_sync_hold: ClassVar[int] = 0
    # Directory for the cross-scene prefs JSON. ``None`` → pythontk's user-config
    # root (see :meth:`_prefs_path`); tests point it at a temp dir so they never
    # touch the user's real config. A DCC adapter that wants its own store (e.g.
    # QSettings) overrides :meth:`_restore_user_prefs` / :meth:`_save_user_prefs`.
    _prefs_dir_override: ClassVar[Optional[str]] = None
    #: What a shot name may hold.  The name IS the clip every carrier writes --
    #: the FBX take, the Unity AnimationClip, the glTF animation and the
    #: ``shot_metadata`` join key -- and this is the set all of them keep
    #: verbatim (proven through Unity by unitytk's clip-name contract test).
    #: The store refuses anything else rather than let an export respell it.
    #: It is the ecosystem's one legal-name rule
    #: (``StrUtils.LEGAL_NAME_PATTERN``), so a field validating a shot name as
    #: the user types -- uitk's ``set_validator("name")`` -- enforces this set.
    NAME_PATTERN: ClassVar[str] = StrUtils.LEGAL_NAME_PATTERN
    #: :attr:`NAME_PATTERN` in words, for tooltips and refusals.
    NAME_RULE: ClassVar[str] = StrUtils.LEGAL_NAME_RULE
    DETECTION_MODES = ("auto", "all", "skip_zero", "zero_as_end")
    FIT_MODES = ("extend_only", "fit_contents")
    DEFAULT_INITIAL_SHOT_LENGTH: float = 200.0
    DEFAULT_FIT_MODE: str = "extend_only"
    DEFAULT_SNAP_WHOLE_FRAMES: bool = True
    #: The gap a NEW store starts with: the production spacing, so a new or
    #: split-off shot does not share its seam frame with its neighbour
    #: (maintainer 2026-10-11).  A saved scene keeps the gap it saved.
    DEFAULT_GAP: float = 10.0
    #: The gap a saved payload without one reads as -- the default it was
    #: written under, so an old scene loads exactly as it did.
    LEGACY_GAP: float = 0.0

    def __init__(
        self,
        shots: Optional[List[ShotBlock]] = None,
    ):
        self.shots: List[ShotBlock] = list(shots) if shots else []
        self.hidden_objects: set = set()
        self.pinned_objects: set = set()
        self.markers: List[Dict[str, Any]] = []
        self.gap: float = self.DEFAULT_GAP
        self.detection_threshold: float = 5.0
        self.detection_mode: str = "auto"  # "auto", "all", "skip_zero", "zero_as_end"
        # Shot construction policy (applies to any caller that builds shots —
        # manifest, sequencer, future tools).  ``fit_mode`` governs whether a
        # shot may shrink below ``initial_shot_length`` to fit its contents.
        self.initial_shot_length: float = self.DEFAULT_INITIAL_SHOT_LENGTH
        self.fit_mode: str = self.DEFAULT_FIT_MODE
        # When enabled, every frame value written through ``snap()`` is
        # rounded to the nearest integer.  Applied at mutation sites so the
        # in-memory model is always valid (see ``ShotStore.snap``).
        self.snap_whole_frames: bool = self.DEFAULT_SNAP_WHOLE_FRAMES
        self.select_on_load: bool = False
        self.frame_on_shot_change: bool = True
        self.locked_gaps: set = set()  # {(left_shot_id, right_shot_id), ...}
        # What the shot system itself wrote on the scene's curves (gap holds,
        # boundary samples).  Persisted with the store because the writes it
        # tracks are persisted with the scene -- a claim that did not survive a
        # reopen would leave its edit behind permanently.
        self.edit_ledger: ShotEditLedger = ShotEditLedger()
        # How each render effect and audio clip is keyed in this scene -- the
        # Render Effects / Audio Clips panels edit it, the manifest's Build
        # keys with it.  A scene setting like ``fit_mode``: saved with the
        # scene and carried by the hand-off, never seeded from user prefs.
        self.effect_recipe: EffectRecipe = EffectRecipe()
        self.locked_objects: set = set()  # object names locked in the sequencer
        self.scene_fps: float = self._scene_fps()
        # Source CSV path (when the store was populated from a manifest CSV).
        # Purely informational — lets the user retrace provenance on reopen.
        self.source_csv: str = ""
        # The Shot Manifest's edit layers, one per source they edit (a sheet's
        # path, or "" for the scene's own shots): ``ManifestEdits.to_dict``
        # payloads, kept apart from the sheet and composed over it on load.
        self.manifest_edits: Dict[str, Dict[str, Any]] = {}
        # The Shot Manifest's colour tags, ``{step_id: palette slot}`` per
        # source, keyed as ``manifest_edits``: how the user organises the
        # table, never an edit (``ManifestTags``).
        self.manifest_tags: Dict[str, Dict[str, str]] = {}
        # How the scene reads each sheet, keyed as ``manifest_edits``:
        # ``{"template": name, "values": {option: value}}`` -- the mapping
        # template and its options travel with the scene, so a reopen reads
        # the sheet the way it was built (``ManifestEdits.sheet_options``).
        self.manifest_options: Dict[str, Dict[str, Any]] = {}
        # A stable id per name the store holds an object by (Maya: the node
        # UUID), captured as it is saved (:meth:`_capture_member_ids`) -- what
        # finds a member a rename the store was not told of left behind
        # (:meth:`_current_names`).  Additive: a reader that does not know
        # the key reads the names as before.
        self.member_ids: Dict[str, str] = {}
        # Export-view projection (opt-in).  When True, every save publishes the
        # FBX-takes + Unity-metadata channels via the ``publish_export_view``
        # hook so any DCC export can carry them.  Off by default.
        self.auto_publish_export: bool = False
        self.clip_name_strategy: str = "name"
        self._active_shot_id: Optional[int] = None  # session-only, not persisted
        self._listeners: List[Callable[[StoreEvent], None]] = []
        self._batch_depth: int = 0
        self._batch_events: List[tuple] = []
        self._dirty: bool = False
        # Boundary-snapshot ledger (session-only, never persisted).  One
        # stack per store — i.e. per scene — shared by every panel that
        # mutates shot boundaries, so an undo after ANY panel's edit
        # restores the right snapshot.  See push_boundary_snapshot().
        # Entries are ``(state, tag)`` — see push_boundary_snapshot().
        self._boundary_undo: List[tuple] = []
        self._boundary_redo: List[tuple] = []
        # Redo branch cleared by the newest push, held so a discard can put
        # it back (see :meth:`discard_boundary_snapshot`).  None = nothing to
        # restore; anything that consumes the redo branch invalidates it --
        # and so does a discard: not None means "the newest push may still be
        # discarded", which keeps a discard to one per push.
        self._boundary_redo_stash: Optional[List[tuple]] = None
        # True while follow_native_step applies restore points.
        self._following_native: bool = False

    # ---- boundary-snapshot ledger ----------------------------------------
    #
    # Scene keyframes ride the DCC's native undo queue; shot BOUNDS live in
    # this store, outside it.  Consumers therefore snapshot bounds before a
    # boundary-mutating edit and restore them when the DCC's undo fires.
    # The ledger lives here — not on a UI controller — because more than one
    # panel mutates boundaries, and per-controller stacks desync from each
    # other and from the DCC queue.

    _BOUNDARY_LEDGER_CAP = 50

    def snapshot_bounds(self) -> list:
        """Return the current shot state for the boundary ledger.

        One record per shot — the FULL identity, not just bounds, so a
        restore can faithfully re-create a shot the mutation removed (redo
        of an insert, undo of a delete).  A shot's keys live in the scene
        and survive both, so reconstruction loses nothing.
        """
        return [
            {
                "shot_id": s.shot_id,
                "name": s.name,
                "start": s.start,
                "end": s.end,
                "objects": list(s.objects),
                "description": s.description,
                "locked": s.locked,
                "metadata": dict(s.metadata),
            }
            for s in self.shots
        ]

    def push_boundary_snapshot(self, tag: Any = None) -> None:
        """Record the current bounds as an undo restore point.

        Call BEFORE mutating (a snapshot taken after records the post-edit
        bounds, so undo would re-apply the very edit it should reverse).
        A new edit invalidates any redo branch, mirroring every undo queue.

        *tag* is an opaque marker the DCC layer may attach (usually via
        :meth:`tag_boundary_snapshot`, once the edit has run and its effect
        on the native queue is known) — see the pairing note below.
        """
        self._boundary_undo.append((self._restore_point(), tag))
        if len(self._boundary_undo) > self._BOUNDARY_LEDGER_CAP:
            self._boundary_undo.pop(0)
        self._boundary_redo_stash = list(self._boundary_redo)
        self._boundary_redo.clear()

    def _restore_point(self) -> dict:
        """Everything a restore has to put back: the shot records, the edit
        ledger AND the gap locks.

        The ledger is a set of ``(curve, frame)`` claims on samples the shot
        system wrote.  An edit moves those claims with the keys; the DCC's
        undo moves the keys back but knows nothing of the claims, so without
        this the ledger points at frames where the animator's own keys now
        sit -- and the next reconcile, believing them its samples, moves or
        cuts them.  Measured on a production assembly: one key drag, undone,
        and the following drag cut 7 of 22 keys off a curve.
        """
        return {
            "shots": self.snapshot_bounds(),
            "ledger": self.edit_ledger.to_dict(),
            # A lock names its gap by the flanking shot ids, so undoing a
            # delete (which re-creates the shot) or an insert has to put the
            # locks of that moment back with it; see _rekey_gap_locks.
            "locked_gaps": sorted(self.locked_gaps),
            # The shot the user was on: undoing a new shot returns to it (and
            # redo to the new one) rather than to whatever survives the
            # restore.
            "active": self._active_shot_id,
            # The markers a tool placed (a Build's), so undoing the build
            # takes them out with its keys.  A hand's markers stay out of it:
            # no marker edit pushes a restore point, so an undo would drop
            # every one placed since.
            "owned_markers": [dict(m) for m in self.markers if self.marker_owner(m)],
        }

    # ---- pairing with the DCC's own undo queue ---------------------------
    #
    # A restore point and the native undo step that accompanies it live in
    # two different stacks, and they do NOT always come in pairs: an edit
    # that moved only BOUNDS (a bounds-only resize, a shot delete) touches
    # no scene data, so the DCC records nothing — and a consumer that then
    # fires a native undo unconditionally pops the user's PREVIOUS,
    # unrelated operation.  The store cannot see the DCC's queue, so it
    # stores whatever opaque marker the DCC layer hands it and gives it
    # back at undo time; interpreting it is that layer's job.

    def tag_boundary_snapshot(self, tag: Any) -> bool:
        """Attach *tag* to the newest restore point.  False if there is none."""
        if not self._boundary_undo:
            return False
        state, _old = self._boundary_undo[-1]
        self._boundary_undo[-1] = (state, tag)
        return True

    def peek_boundary_tag(self, redo: bool = False) -> Any:
        """Return the tag of the restore point the next apply would consume."""
        stack = self._boundary_redo if redo else self._boundary_undo
        return stack[-1][1] if stack else None

    def has_boundary_snapshot(self, redo: bool = False) -> bool:
        """True when a restore point is available in that direction."""
        return bool(self._boundary_redo if redo else self._boundary_undo)

    def changed_since_snapshot(self) -> bool:
        """Whether the store differs from its newest restore point: the shot
        records, the edit ledger, the gap locks or the active shot.  ``False``
        with no restore point.

        What a DCC's :meth:`scene_edit` asks to tell an edit that changed what
        only the store holds -- which the DCC's undo queue cannot see -- from
        one that changed nothing at all.
        """
        if not self._boundary_undo:
            return False
        return self._boundary_undo[-1][0] != self._restore_point()

    def discard_boundary_snapshot(self) -> bool:
        """Drop the most recent restore point without applying it.

        For an edit that pushed up front and then turned out to be a no-op:
        leaving the entry would hand the next undo a restore point for an
        operation that never happened.

        The push also cleared the redo branch — correct for a real edit, but
        a no-op must not cost the user their redo — so this puts it back.

        At most once per push: a second discard, or one after the point was
        applied or the ledger cleared, drops nothing.  A DCC's
        :meth:`scene_edit` drops its own point when the edit changed nothing
        at all, and the caller that then discards "its" no-op point would
        otherwise drop the restore point of the edit before it -- the undo of
        THAT edit then left its shot state behind.

        Returns:
            Whether a restore point was dropped.
        """
        if self._boundary_redo_stash is None:
            return False
        dropped = bool(self._boundary_undo)
        if dropped:
            self._boundary_undo.pop()
        self._boundary_redo = self._boundary_redo_stash
        self._boundary_redo_stash = None
        return dropped

    def restore_boundary_snapshot(self) -> bool:
        """Apply the most recent restore point (the undo direction).

        The pre-restore state moves to the redo side.  Membership is
        restored SYMMETRICALLY: a shot absent from the snapshot is removed
        (undoing an insert must not leave a phantom overlapping the
        restored layout), and a shot the snapshot names but the store no
        longer holds is re-created from its record (undoing a delete —
        the keys were never deleted with the store entry).

        Returns ``True`` when a snapshot was applied.
        """
        if not self._boundary_undo:
            return False
        state, tag = self._boundary_undo.pop()
        # The tag rides across: redoing this edit re-applies the same DCC
        # step (or, when unpaired, again touches nothing).
        self._boundary_redo.append((self._restore_point(), tag))
        self._boundary_redo_stash = None
        self._apply_boundary_snapshot(state)
        return True

    def redo_boundary_snapshot(self) -> bool:
        """Re-apply the state a restore stepped back from (the redo direction).

        Without this, a DCC redo re-applies the scene keys while the bounds
        stay restored — resurrecting the keys-outside-their-shot state.
        Returns ``True`` when a snapshot was applied.
        """
        if not self._boundary_redo:
            return False
        state, tag = self._boundary_redo.pop()
        self._boundary_undo.append((self._restore_point(), tag))
        self._boundary_redo_stash = None
        self._apply_boundary_snapshot(state)
        return True

    def clear_boundary_snapshots(self) -> None:
        """Drop both ledger sides (scene swap: colliding shot_ids across
        scenes would write one scene's boundaries onto another's shots)."""
        self._boundary_undo.clear()
        self._boundary_redo.clear()
        self._boundary_redo_stash = None

    def _apply_boundary_snapshot(self, state: dict) -> None:
        """Put back a :meth:`_restore_point`: the ledger, the shots, then the
        active shot."""
        self.edit_ledger = ShotEditLedger.from_dict(state.get("ledger"))
        owned = state.get("owned_markers")
        locks = state.get("locked_gaps")
        has_active = "active" in state
        active = state.get("active")
        state = state["shots"]
        snap_ids = {rec["shot_id"] for rec in state}
        with self.batch_update():
            if owned is not None:
                # In the batch: one redraw with the shots, never one before.
                self.set_markers(
                    [m for m in self.markers if not self.marker_owner(m)] + owned
                )
            for shot in list(self.shots):
                if shot.shot_id not in snap_ids:
                    self.remove_shot(shot.shot_id)
            for rec in state:
                if self.shot_by_id(rec["shot_id"]) is None:
                    # Re-create a shot the mutation removed — its keys live
                    # in the scene and were never deleted with it.
                    block = ShotBlock(
                        shot_id=rec["shot_id"],
                        name=rec["name"],
                        start=rec["start"],
                        end=rec["end"],
                        objects=list(rec["objects"]),
                        metadata=dict(rec["metadata"]),
                        locked=rec["locked"],
                        description=rec["description"],
                    )
                    self.shots.append(block)
                    self._notify(ShotDefined(shot=block))
                    self.mark_dirty()
                else:
                    # The name is put back as recorded, not re-validated: a
                    # restore returns to a state the store already held, and
                    # part-way through one two shots can briefly share a name
                    # (undoing a swap).  update_shot below notifies the rename.
                    self.shot_by_id(rec["shot_id"]).name = rec["name"]
                    self.update_shot(
                        rec["shot_id"],
                        start=rec["start"],
                        end=rec["end"],
                        objects=rec["objects"],
                        description=rec["description"],
                        locked=rec["locked"],
                        metadata=rec["metadata"],
                    )
            if locks is not None:
                restored = {tuple(pair) for pair in locks}
                if restored != self.locked_gaps:
                    self.locked_gaps = restored
                    self.mark_dirty()
            if has_active and (active is None or self.shot_by_id(active)):
                self.set_active_shot(active)

    # ---- following the DCC's own undo --------------------------------------
    #
    # The DCC's undo puts the scene's keys back; the restore points put back
    # what only the store holds.  So the STORE follows every native undo and
    # redo -- one listener per store class, installed by :meth:`active` -- and
    # Ctrl+Z restores the shot state whichever panel is open, or none.  When
    # only a panel listened, an edit made with no Sequencer this session (a
    # Shots-window resize, a key edit's claims, a manifest Build) was undone in
    # its keys and kept in its shots.  A panel redraws off
    # :meth:`add_native_step_listener`.

    def _native_step_crossed(self, redo: bool = False, first: bool = True) -> bool:
        """Whether the native undo (or redo) that just ran passed over the edit
        of the restore point next in that direction (overridable hook).

        Pure default: ``False`` -- the pure model has no native queue.  The DCC
        stores read the pairing their :meth:`scene_edit` tagged the point with
        (mayatk: the undo chunk's name; blendertk: an edit serial).  *first*:
        no point has been applied for this step yet -- an untagged point (a
        legacy push) counts as crossed for the first one only.
        """
        return False

    def follow_native_step(self, redo: bool = False) -> int:
        """Apply every restore point the DCC's own undo (or redo) that just
        ran passed over: newest first for an undo, oldest first for a redo --
        the order the stacks hold them.  One step crosses one edit; an undo
        history jump can cross several.

        The store's events fire as for any restore, inside one batch;
        :attr:`following_native_step` is ``True`` while they do.

        Parameters:
            redo: Follow a redo instead of an undo.

        Returns:
            The number of restore points applied (0: the step was not a shot
            edit's).
        """
        apply = self.redo_boundary_snapshot if redo else self.restore_boundary_snapshot
        applied = 0
        self._following_native = True
        try:
            with self.batch_update():
                while self._native_step_crossed(redo, first=applied == 0):
                    if not apply():
                        break
                    applied += 1
        finally:
            self._following_native = False
        return applied

    @property
    def following_native_step(self) -> bool:
        """``True`` while :meth:`follow_native_step` applies restore points.

        A panel that redraws once on :class:`NativeStepFollowed` skips the store
        events raised meanwhile rather than redraw for them too.
        """
        return self._following_native

    @classmethod
    def _on_native_step(cls, redo: bool = False) -> None:
        """What the DCC's undo / redo listener runs: the active store follows
        the step (:meth:`follow_native_step`), then every native-step listener
        hears of it -- also when nothing was applied, so a panel can redraw
        after any undo.  Nothing at all while :meth:`native_undo_sync_held`.
        """
        if ShotStore._native_sync_hold:
            return
        store = cls._active
        applied = 0
        if store is not None:
            try:
                applied = store.follow_native_step(redo)
            except Exception:
                _log.warning(
                    "shot store: native %s not followed",
                    "redo" if redo else "undo",
                    exc_info=True,
                )
        event = NativeStepFollowed(redo=redo, applied=applied, store=store)
        for cb in list(ShotStore._native_step_listeners):
            try:
                cb(event)
            except Exception:
                _log.warning("native-step listener failed", exc_info=True)

    @classmethod
    @contextmanager
    def native_undo_sync_held(cls):
        """Hold the store's native undo sync for the block.

        For a caller that steps the DCC's undo queue itself AND applies the
        restore points it decided on (the Maya Sequencer's Undo / Redo
        buttons, which step past bookkeeping entries first): the store must
        not follow those steps a second time.  Process-wide, and nests.
        """
        ShotStore._native_sync_hold += 1
        try:
            yield
        finally:
            ShotStore._native_sync_hold -= 1

    @classmethod
    def add_native_step_listener(
        cls, callback: Callable[["NativeStepFollowed"], None]
    ) -> None:
        """Register *callback* for every native undo / redo the store followed
        (:class:`NativeStepFollowed`).  Class-level: it survives store swaps,
        as :meth:`add_invalidation_listener` does."""
        if callback not in ShotStore._native_step_listeners:
            ShotStore._native_step_listeners.append(callback)

    @classmethod
    def remove_native_step_listener(
        cls, callback: Callable[["NativeStepFollowed"], None]
    ) -> None:
        """Remove a callback :meth:`add_native_step_listener` registered."""
        try:
            ShotStore._native_step_listeners.remove(callback)
        except ValueError:
            pass

    @classmethod
    def _install_native_undo_sync(cls) -> None:
        """Subscribe this store class to the DCC's undo and redo (overridable
        hook): each runs :meth:`_on_native_step`.

        Pure default: nothing -- no native queue.  :meth:`active` and
        :meth:`set_active` call it every time, so an override must be cheap
        and idempotent; :meth:`clear_active` tears it down
        (:meth:`_remove_native_undo_sync`).
        """

    @classmethod
    def _remove_native_undo_sync(cls) -> None:
        """Tear down what :meth:`_install_native_undo_sync` subscribed
        (overridable hook; pure default: nothing)."""

    # ---- scene hooks (overridable; pure defaults) ------------------------

    def _scene_fps(self) -> float:
        """Return the current scene framerate (overridable hook).

        Pure default: the last-known :attr:`scene_fps` or :data:`_DEFAULT_FPS`
        (24.0).  DCC subclasses override to query the live scene's time unit.
        Reads via ``getattr`` so it is safe to call before ``scene_fps`` is
        first assigned in ``__init__``.
        """
        return getattr(self, "scene_fps", None) or _DEFAULT_FPS

    @staticmethod
    def has_animation() -> bool:
        """Return whether the scene contains animation (overridable hook).

        Pure default: ``False`` — the pure model has no scene to inspect.
        DCC subclasses override to detect animCurves / actions driving transforms.
        ``@staticmethod`` (mirroring the Maya original) so a controller's
        class-level ``ShotStore.has_animation()`` call — no instance needed, it
        queries the live scene — resolves on the class, not just an instance.
        """
        return False

    def detect_regions(self) -> List[Dict[str, Any]]:
        """Detect shot-region candidates from the scene (overridable hook).

        Pure default: ``[]`` — region detection needs scene acquisition.
        DCC subclasses override to gather segments / selected keys and feed
        them to :func:`~pythontk.core_utils.engines.shots.shot_detection.cluster_segments_by_gap`
        or :func:`~pythontk.core_utils.engines.shots.shot_detection.boundaries_from_key_entries`.

        Returns:
            List of candidate dicts with ``"name"``, ``"start"``,
            ``"end"``, and ``"objects"`` keys.
        """
        return []

    def assess(self) -> Dict[int, str]:
        """Assess whether each shot's objects exist in the scene (hook).

        Pure default: every shot ``"valid"`` — the pure model can't query a
        scene, so nothing is flagged missing.  DCC subclasses override to
        resolve object existence and return ``"missing_object"`` where
        appropriate.

        Returns:
            Dict mapping ``shot_id`` → status string.
        """
        return {s.shot_id: "valid" for s in self.shots}

    def _resolve_long_names(self, names):
        """Resolve object names to unique scene paths (overridable hook).

        Pure default: identity — pure / Blender names are already unique, so
        no disambiguation is needed.  The Maya subclass overrides to resolve
        to long DAG paths (``cmds.ls(names, long=True)``).  This is the single
        chokepoint for name disambiguation so per-site copies stay out.
        """
        return list(names) if names else []

    def _existing_objects(self, names: Iterable[str]) -> set:
        """Which of *names* the scene still holds (overridable hook).

        Pure default: every one -- the pure model has no scene to lose an
        object from, so :meth:`stale_shots` never finds one here.  DCC
        subclasses override to look each name up as the export view names
        it: a Maya member is a long DAG path, and one whose parent was
        renamed or reparented is still the object its leaf names.
        """
        return set(names)

    def _keyed_windows(self, windows: List[Tuple[float, float]]) -> List[bool]:
        """Per ``(start, end)`` window, whether anything in the scene is keyed
        inside it (overridable hook).

        Pure default: every window is -- the pure model cannot tell, and a
        shot is never called stale on a guess.  DCC subclasses override to
        ask every animation curve in the scene, whatever it drives.
        """
        return [True] * len(windows)

    def _members_keyed_windows(
        self, entries: List[Tuple[List[str], Tuple[float, float]]]
    ) -> List[Optional[bool]]:
        """Per ``(names, (start, end))``, whether a curve of the named objects
        keys a frame inside the window (overridable hook, :meth:`empty_shots`).

        ``None`` for an entry none of whose names the scene holds -- the caller
        then asks the whole scene (:meth:`_keyed_windows`).  The shot system's
        own bound samples (:attr:`edit_ledger` keys) are bookkeeping, not
        content, and do not count.  Pure default: every window is keyed -- a
        shot is never called empty on a guess.
        """
        return [True] * len(entries)

    def _audio_windows(self, windows: List[Tuple[float, float]]) -> List[bool]:
        """Per ``(start, end)`` window, whether an audio clip plays in it
        (overridable hook, :meth:`empty_shots`).

        Pure default: none -- the pure model holds no audio.  DCC subclasses
        override to read every clip's span (``AudioUtils.read_all_events``).
        """
        return [False] * len(windows)

    def _curve_ledger_keys(self, objects=None) -> Callable[[Any], Iterable[str]]:
        """``keys_of(curve)``: the :attr:`edit_ledger` keys a DCC curve is
        claimed under (overridable hook; see :meth:`release_replaced`).

        Pure default: a curve is its own key (``str(curve)``) -- Maya claims
        by anim-curve node name.  A DCC whose curves have no name overrides it:
        blendertk keys an fcurve by its owner and channel, so it maps through
        the owners among *objects*.
        """
        return lambda curve: (str(curve),)

    def publish_export_view(self, strategy: Optional[str] = None) -> Optional[str]:
        """Project the export view onto a DCC carrier (overridable hook).

        Pure default: no-op returning ``None`` — there is no DCC node to write
        to.  DCC subclasses override to serialise :meth:`to_export_view` onto
        their shared export node so any FBX export carries the shot takes and
        metadata.
        """
        return None

    def _schedule_flush(self) -> None:
        """Flush the dirty store to persistence (overridable hook).

        Pure default: flush immediately.  DCC subclasses may override to
        coalesce rapid mutations into a single deferred write (e.g. Maya's
        ``cmds.evalDeferred``).
        """
        self._flush_dirty()

    @contextmanager
    def scene_edit(self, label: str = "edit", snapshot: bool = True):
        """Run a shot edit as one step of the scene's undo (overridable hook).

        Pure default: push a boundary restore point (*snapshot*) and run the
        body -- the pure model has no native undo queue to pair the point
        with.  The DCC stores override it to run the body in one named undo
        step and tag the point with that step (``mayatk.ShotStore`` /
        ``blendertk.BlenderShotStore``).  Every one ends by emitting
        :class:`ShotsEdited` when the body changed the shot records.

        Parameters:
            label: The edit's name (the undo step's, and the event's).
            snapshot: Push a restore point before the body; ``False`` for an
                edit that must not record one.
        """
        if snapshot:
            self.push_boundary_snapshot()
        yield None
        # Every store's scene_edit ends the same way: an edit that changed the
        # records since its restore point says so (ShotsEdited) -- only after
        # a body that did not raise, and never without a point to compare to.
        if snapshot and self.changed_since_snapshot():
            self._notify(ShotsEdited(label=label))

    # ---- claims on keys a DCC key edit replaced or moved ------------------

    def release_replaced(
        self, objects: Optional[Iterable[Any]] = None
    ) -> Callable[[Any, Optional[Iterable[float]]], int]:
        """An ``on_replace(curve, frames)`` for the DCC key edits that land a
        key on an occupied frame (the ``on_replace`` of mayatk's and
        blendertk's ``AnimUtils`` key edits: Align, Invert, Paste, ...).

        Such an edit REPLACES the key that stood on the frame, and every claim
        the shot system held on that key -- a gap hold's step, a bound sample,
        a behavior's authored key -- goes with it, as with every key a system
        cut drops (:meth:`ShotEditLedger.release`).  Left on the frame, the
        claims passed to the key that landed there, and the next Build deleted
        that animator key as a dropped behavior.  A key a delete edit removes
        goes the same way, so the same callable is that edit's ``on_delete``
        (Delete Keys, Remove Intermediate Keys, Simplify, Optimize, Untie,
        Repair) -- and *frames* ``None`` says the CURVE itself is gone (a
        static curve dropped, a corrupted one deleted): every claim on it goes.

        Parameters:
            objects: The objects the edit runs over, for a DCC that keys a
                claim by the curve's owner (:meth:`_curve_ledger_keys`);
                ``None``: whatever owns the curve.

        Returns:
            ``on_replace(curve, frames)``: drops every claim at each of
            *frames* on *curve* (every claim on it, *frames* ``None``) --
            from the ledger the store holds when it runs, which an undo
            restore may have swapped -- marks the store dirty when any went,
            and returns how many did.
        """
        return self._release_through(self._curve_ledger_keys(objects))

    def remap_moved(
        self, objects: Optional[Iterable[Any]] = None
    ) -> Callable[[Any, Iterable[Tuple[float, float]]], int]:
        """An ``on_move(curve, pairs)`` for the DCC key edits that move keys
        (the ``on_move`` of mayatk's and blendertk's ``AnimUtils`` key edits).

        A claimed key the edit MOVES carries its claims to its new frame, as
        the shot sequencer's own key moves do (:meth:`ShotEditLedger.remap`).
        Left on the old frame, a claim passed to whatever key arrived there
        while the key that moved read as the animator's: the next Build
        deleted the one and keyed the behavior again beside the other.  A
        moved bound sample then follows its bound at the sequencer's next
        reconcile, as a dragged one does.

        Parameters:
            objects: As :meth:`release_replaced`.

        Returns:
            ``on_move(curve, pairs)``: moves every claim on *curve* from the
            ``old`` frame of each ``(old, new)`` pair to its ``new`` -- every
            pair matched against the claims as they stood, so two keys that
            trade frames trade claims -- marks the store dirty when any moved,
            and returns how many did.
        """
        return self._remap_through(self._curve_ledger_keys(objects))

    def claim_hooks(
        self, objects: Optional[Iterable[Any]] = None
    ) -> Tuple[Callable, Callable]:
        """``(on_replace, on_move)`` over *objects*: :meth:`release_replaced`
        and :meth:`remap_moved` sharing ONE key resolution
        (:meth:`_curve_ledger_keys`), so a DCC that resolves a curve's owners
        by scanning the scene pays that once per edit, however many of the
        two hooks the edit calls (and nothing for a hook it never calls).

        Parameters:
            objects: As :meth:`release_replaced`.
        """
        keys_of = self._curve_ledger_keys(objects)
        return self._release_through(keys_of), self._remap_through(keys_of)

    @contextmanager
    def key_edit(self, label: str = "edit", objects: Optional[Iterable[Any]] = None):
        """Run a DCC key edit that may replace or move claimed keys.

        Yields ``(on_replace, on_move)`` -- :meth:`release_replaced` and
        :meth:`remap_moved` over *objects* -- for the edit to take.  When the
        store claims anything, the edit runs inside :meth:`scene_edit`: one
        undo step with a restore point taken before it, so undoing the edit
        puts the claims it released or moved back with its keys.  A store that
        claims nothing records no restore point (a scene with no shots gets
        none), and the edit keeps the undo step its caller gives it.

        Parameters:
            label: The undo step's name.
            objects: As :meth:`release_replaced`.
        """
        hooks = self.claim_hooks(objects)
        if not self.edit_ledger:
            yield hooks
            return
        with self.scene_edit(label):
            yield hooks

    # ---- active shot (session state, not persisted) ----------------------

    @property
    def active_shot_id(self) -> Optional[int]:
        """The currently selected shot, or ``None``."""
        return self._active_shot_id

    def set_active_shot(self, shot_id: Optional[int]) -> None:
        """Set the active shot and notify listeners."""
        previous = self._active_shot_id
        if shot_id == previous:
            return
        self._active_shot_id = shot_id
        self._notify(ActiveShotChanged(shot_id=shot_id, previous=previous))

    def is_empty(self) -> bool:
        """``True`` when the store declares no shots.

        The scene-persistence adapters ask this before reacting to a scene
        event (a frame-rate change) so an empty store is never marked dirty
        and flushed onto a scene that had no store data.
        """
        return not self.shots

    # ---- observer --------------------------------------------------------

    def update_effect_recipe(self, **changes: Any) -> bool:
        """Change fields of the scene's :attr:`effect_recipe`.

        The one way a panel writes the recipe: the store is dirtied (so the
        change saves with the scene) and :class:`SettingsChanged` fires, so
        every panel bound to it -- and the manifest, whose Assess compares
        keys against it -- re-reads it.

        Parameters:
            **changes: :class:`EffectRecipe` field values.

        Returns:
            Whether anything changed (an equal value writes nothing).

        Raises:
            TypeError: A name that is not a recipe field.
        """
        recipe = self.effect_recipe.replace(**changes)
        if recipe == self.effect_recipe:
            return False
        self.effect_recipe = recipe
        self.mark_dirty()
        self.notify_settings_changed()
        return True

    def notify_settings_changed(self) -> None:
        """Fire a ``"settings_changed"`` event.

        Call after modifying detection-relevant settings such as
        ``detection_mode``, ``detection_threshold``, or ``gap`` so
        downstream consumers can invalidate cached results and re-detect.
        """
        self._notify(SettingsChanged())

    def add_listener(self, callback: Callable[[StoreEvent], None]) -> None:
        """Register a listener called on store mutations.

        The callback receives a single :class:`StoreEvent` instance.
        Use ``isinstance()`` to dispatch on event type::

            def on_event(event: StoreEvent) -> None:
                if isinstance(event, ShotDefined):
                    print(event.shot)

        Event types: :class:`ShotDefined`, :class:`ShotUpdated`,
        :class:`ShotRemoved`, :class:`ActiveShotChanged`,
        :class:`SettingsChanged`, :class:`BatchComplete`,
        :class:`ShotsEdited`, :class:`MarkersChanged`.
        """
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[StoreEvent], None]) -> None:
        """Remove a previously registered listener."""
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass

    def _notify(self, event: StoreEvent) -> None:
        """Fire all registered listeners (deferred during :meth:`batch_update`)."""
        if self._batch_depth > 0:
            self._batch_events.append(event)
            return
        for cb in self._listeners:
            try:
                cb(event)
            except Exception:
                # A broken listener must not break the store, but a
                # silent swallow leaves the UI dead with no trace.
                _log.warning("shot store listener failed", exc_info=True)

    @contextmanager
    def batch_update(self):
        """Defer listener notifications until the block exits.

        On exit a single ``"batch_complete"`` event is fired instead of
        the individual events that were accumulated.
        """
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                if self._batch_events:
                    self._batch_events.clear()
                    self._notify(BatchComplete())
                # Synchronous flush — batch = single atomic write.
                # Runs even with no accumulated events: mark_dirty
                # inside a batch defers its flush to here, and mutators
                # that don't notify (pin/hide/gap-lock) would otherwise
                # leave the dirty store unscheduled for save.
                self._flush_dirty()

    # ---- markers -----------------------------------------------------------

    @staticmethod
    def marker_owner(marker: Dict[str, Any]) -> str:
        """The tool that placed *marker* (its ``data["owner"]``), ``""`` for
        one placed by hand.

        A tool tags the markers it places so it can find them again -- to
        move them or take them out -- and leave everyone else's alone.
        """
        data = marker.get("data") if isinstance(marker, dict) else None
        return str((data or {}).get("owner") or "")

    def set_markers(self, markers: Iterable[Dict[str, Any]]) -> bool:
        """Replace the timeline markers; on a change, mark the store dirty
        and emit :class:`MarkersChanged` so a sequencer showing them redraws.

        For a tool writing markers: the sequencer's own marker edits start in
        its widget and write :attr:`markers` directly.

        Returns:
            Whether the markers changed.
        """
        markers = [dict(m) for m in markers]
        if markers == self.markers:
            return False
        self.markers = markers
        self.mark_dirty()
        self._notify(MarkersChanged())
        return True

    # ---- gap locking -----------------------------------------------------

    def is_gap_locked(self, left_id: int, right_id: int) -> bool:
        """Return whether the gap between two adjacent shots is locked."""
        return (left_id, right_id) in self.locked_gaps

    def lock_gap(self, left_id: int, right_id: int) -> None:
        """Lock a gap so its width is preserved during global respace."""
        key = (left_id, right_id)
        if key not in self.locked_gaps:
            self.locked_gaps.add(key)
            self.mark_dirty()

    def unlock_gap(self, left_id: int, right_id: int) -> None:
        """Unlock a gap so it follows the global gap value."""
        if (left_id, right_id) in self.locked_gaps:
            self.locked_gaps.discard((left_id, right_id))
            self.mark_dirty()

    def lock_all_gaps(self) -> None:
        """Lock every adjacent gap."""
        sorted_shots = self.sorted_shots()
        before = len(self.locked_gaps)
        for i in range(len(sorted_shots) - 1):
            self.locked_gaps.add((sorted_shots[i].shot_id, sorted_shots[i + 1].shot_id))
        if len(self.locked_gaps) != before:
            self.mark_dirty()

    def unlock_all_gaps(self) -> None:
        """Unlock every gap."""
        if self.locked_gaps:
            self.locked_gaps.clear()
            self.mark_dirty()

    # ---- singleton / persistence -----------------------------------------

    @classmethod
    def set_persistence(cls, backend: Optional[ScenePersistence]) -> None:
        """Set the persistence backend used by :meth:`active` and :meth:`save`.

        Pass ``None`` to disable persistence (pure in-memory mode).
        Call *before* :meth:`active` to ensure load picks up the backend.
        """
        cls._persistence = backend

    @classmethod
    def active(cls) -> "ShotStore":
        """Return the current active store, creating one if needed.

        If a persistence backend is configured, saved data is loaded
        automatically on first access.  With no backend the store runs in
        pure in-memory mode.
        """
        if cls._active is None:
            persistence = cls._persistence
            data = persistence.load() if persistence is not None else None
            if data:
                cls._active = cls.from_dict(data)
                # Reconcile FPS: if the scene was saved at a different
                # framerate, rescale shot timings to match the current one.
                current_fps = cls._active._scene_fps()
                if (
                    cls._active.shots
                    and abs(cls._active.scene_fps - current_fps) > 0.01
                ):
                    cls._active.rescale_to_fps(current_fps)
            else:
                cls._active = cls()
                cls._active._restore_user_prefs()
        # The active store follows the DCC's own undo and redo -- whichever
        # panel is open, or none (see follow_native_step).
        cls._install_native_undo_sync()
        return cls._active

    @classmethod
    def set_active(cls, store: "ShotStore") -> None:
        """Replace the active store instance (which follows the DCC's undo
        from then on, as :meth:`active`'s does)."""
        cls._active = store
        cls._install_native_undo_sync()

    @classmethod
    def clear_active(cls) -> None:
        """Reset the active store and persistence backend, and stop following
        the DCC's undo (:meth:`active` subscribes again)."""
        cls._active = None
        cls._remove_native_undo_sync()
        if cls._persistence is not None:
            # Tear down the backend's scene callbacks — dropping the
            # reference alone leaks them, and a leaked before-save
            # callback can write the OLD store's data into a new scene.
            remove = getattr(cls._persistence, "remove_callbacks", None)
            if remove is not None:
                try:
                    remove()
                except Exception:
                    pass
            cls._persistence = None

    # ---- invalidation listeners (class-level) ----------------------------

    @classmethod
    def add_invalidation_listener(
        cls, callback: Callable[["StoreInvalidated"], None]
    ) -> None:
        """Register a callback fired when the active store is discarded.

        Unlike instance-level :meth:`add_listener`, these survive across
        store instances — useful for UI controllers that need to rebind
        after a scene change.
        """
        if callback not in cls._invalidation_listeners:
            cls._invalidation_listeners.append(callback)

    @classmethod
    def remove_invalidation_listener(
        cls, callback: Callable[["StoreInvalidated"], None]
    ) -> None:
        """Remove a previously registered invalidation listener."""
        try:
            cls._invalidation_listeners.remove(callback)
        except ValueError:
            pass

    @classmethod
    def watch_settings(cls, callback: Callable[[], None]) -> Callable[[], None]:
        """Call *callback* whenever the active store's settings change -- and
        when the active store itself is replaced (a scene opened) -- until the
        returned function is called.

        What a panel showing a store setting (the effect recipe, the fit mode)
        needs, once: a listener on the active store for :class:`SettingsChanged`
        and :class:`BatchComplete`, moved to the next store when the scene
        changes. ``callback`` takes no arguments; it re-reads what it shows.

        Returns:
            ``unsubscribe()`` -- call it when the panel goes away (the
            invalidation registry is class-level and holds the callback).
        """
        state: Dict[str, Any] = {"store": None}

        def on_event(event: StoreEvent) -> None:
            if isinstance(event, (SettingsChanged, BatchComplete)):
                callback()

        def detach() -> None:
            if state["store"] is not None:
                state["store"].remove_listener(on_event)
                state["store"] = None

        def attach() -> None:
            try:
                store = cls.active()
            except Exception:
                return
            store.add_listener(on_event)
            state["store"] = store

        def on_invalidated(_event: StoreInvalidated) -> None:
            detach()
            attach()
            callback()

        attach()
        cls.add_invalidation_listener(on_invalidated)

        def unsubscribe() -> None:
            cls.remove_invalidation_listener(on_invalidated)
            detach()

        return unsubscribe

    @classmethod
    def invalidate(cls) -> None:
        """Drop the active store (the scene changed) and fire the invalidation listeners.

        The backend keeps its subscriptions; the next :meth:`active` reloads
        from it.  This is what a scene-persistence adapter calls from its
        scene-opened job — the same entry point ``KeyStash`` exposes, so one
        adapter can serve either store.
        """
        cls._active = None
        cls._notify_invalidated()

    @classmethod
    def flush_pending(cls) -> None:
        """Store what the active store holds but has not written yet.

        A DCC store coalesces its writes (Maya's on idle), so the scene record
        can be an edit behind the live store.  A crossing reads the record
        (the owner hook ``SceneStoreBase`` calls first), and so does any
        caller that must see the store as the user does.
        """
        if cls._active is not None:
            cls._active._flush_dirty()

    @classmethod
    def _notify_invalidated(cls) -> None:
        """Fire all invalidation listeners."""
        event = StoreInvalidated()
        for cb in cls._invalidation_listeners:
            try:
                cb(event)
            except Exception:
                _log.warning("shot store invalidation listener failed", exc_info=True)

    # ---- cross-scene user preferences ------------------------------------

    @classmethod
    def _prefs_path(cls):
        """Cross-scene prefs JSON location — pythontk's user-config root by default
        (zero-dep, no Qt/QSettings), or ``_prefs_dir_override`` when set (tests)."""
        from pathlib import Path

        if cls._prefs_dir_override is not None:
            return Path(cls._prefs_dir_override) / "shots_prefs.json"
        from pythontk.core_utils.user_config import UserConfig

        return UserConfig.user_config_root() / "shots" / "prefs.json"

    def _restore_user_prefs(self) -> None:
        """Apply detection preferences from the cross-scene prefs file.

        Called when a fresh store is created (no per-scene persistence) so that
        ``detection_mode`` etc. survive scene changes.  Silent no-op when the
        file is absent or unreadable.  A DCC adapter may override this (and
        :meth:`_save_user_prefs`) to use its own store — pythontk stays zero-dep.
        """
        import json

        try:
            path = self._prefs_path()
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        dm = data.get("detection_mode")
        if isinstance(dm, str) and dm in self.DETECTION_MODES:
            self.detection_mode = dm
        if data.get("select_on_load"):
            self.select_on_load = True
        dt = data.get("detection_threshold")
        if dt is not None:
            try:
                self.detection_threshold = float(dt)
            except (TypeError, ValueError):
                pass
        fm = data.get("fit_mode")
        if isinstance(fm, str) and fm in self.FIT_MODES:
            self.fit_mode = fm
        isl = data.get("initial_shot_length")
        if isl is not None:
            try:
                self.initial_shot_length = float(isl)
            except (TypeError, ValueError):
                pass
        if "snap_whole_frames" in data:
            self.snap_whole_frames = bool(data.get("snap_whole_frames"))

    def _save_user_prefs(self) -> None:
        """Persist detection preferences to the cross-scene prefs file (zero-dep JSON).

        Called by the panel that changes a preference, never by :meth:`save`.
        Atomic (``UserConfig.save_file``): the whole file is rewritten, and a
        plain ``write_text`` cut off mid-write left a truncated file the reader
        discards -- the preferences with it.
        """
        from pythontk.core_utils.user_config import UserConfig

        try:
            UserConfig.save_file(
                self._prefs_path(),
                {
                    "detection_mode": self.detection_mode,
                    "select_on_load": self.select_on_load,
                    "detection_threshold": self.detection_threshold,
                    "fit_mode": self.fit_mode,
                    "initial_shot_length": self.initial_shot_length,
                    "snap_whole_frames": self.snap_whole_frames,
                },
            )
        except Exception:
            pass

    # ---- frame snapping --------------------------------------------------

    def snap(self, frame: float, direction: str = "nearest") -> float:
        """Return *frame* on a whole frame when snapping is on.

        Single chokepoint for the ``snap_whole_frames`` policy.  Call at
        any site that writes a frame value to a shot, keyframe, or
        timeline range to guarantee the in-memory model stays valid.

        Parameters:
            frame: The frame to snap.
            direction: ``"nearest"`` (the default) rounds.  ``"down"`` /
                ``"up"`` floor / ceil, for a bound that must ENCLOSE content
                -- a start snaps down, an end up.  Rounded, a start could land
                up to half a frame past a shot's first key (a retime leaves
                keys on fractional frames: 984.556 -> 985), putting that key
                in the span the bound gave up and in the neighbour's envelope
                from the next edit on.

        Returns:
            The snapped frame, or *frame* unchanged when snapping is off.

        Raises:
            ValueError: *direction* is none of the three (a typo would
                otherwise round silently).
        """
        if direction not in ("nearest", "down", "up"):
            raise ValueError(
                f"snap direction must be 'nearest', 'down' or 'up', not {direction!r}"
            )
        if not self.snap_whole_frames:
            return float(frame)
        if direction == "down":
            return float(math.floor(frame + 1e-6))
        if direction == "up":
            return float(math.ceil(frame - 1e-6))
        return float(round(frame))

    def enclosing_bounds(
        self,
        shot_id: int,
        lo: float,
        hi: float,
        seam_keyed: Optional[Callable[[float], bool]] = None,
    ) -> Tuple[float, float]:
        """The bounds *shot_id* takes so content landing on ``[lo, hi]`` sits
        INSIDE it -- the grow a key or clip dragged past a bound asks for.

        Outward to whole frames (:meth:`snap`: a start down, an end up) and
        never shrinking. And never onto a contiguous seam: two shots that touch
        share ONE sample, the preceding shot's, so content landing on the frame
        the grown bound shares with a touching neighbour would take that
        neighbour's opening pose (measured 2026-09-23: A [0,50], B [50,100],
        A's key dragged to 60 -- B opened on A's pose and its own slipped a
        frame). The bound goes one frame past such a landing instead
        (maintainer decision, 2026-09-23): the shots still touch, and the
        neighbour keeps its pose. Upstream mirrors it.

        Parameters:
            shot_id: The shot that grows.
            lo, hi: Where the moved content lands, earliest and latest.
            seam_keyed: Whether the moving content holds a key ON a seam
                frame (called with the frame) -- the shared sample a landing
                would take. With none there nothing is displaced, and a step
                would only shift the neighbour a frame for no pose. ``None``
                assumes a key: the side that never loses a pose.

        Returns:
            ``(start, end)`` for the shot; its current bounds when the content
            already fits. An unknown *shot_id* returns ``(lo, hi)`` snapped.
        """
        eps = 1e-6
        shot = self.shot_by_id(shot_id)
        start, end = self.snap(lo, "down"), self.snap(hi, "up")
        if shot is None:
            return start, end
        start, end = min(shot.start, start), max(shot.end, end)
        ordered = self.sorted_shots()
        index = next(i for i, s in enumerate(ordered) if s.shot_id == shot_id)
        before = ordered[index - 1] if index > 0 else None
        after = ordered[index + 1] if index + 1 < len(ordered) else None
        keyed = seam_keyed or (lambda _frame: True)
        if after is not None and abs(after.start - shot.end) <= eps:
            if abs(hi - end) <= eps and keyed(shot.end):
                end += 1.0
        if before is not None and abs(shot.start - before.end) <= eps:
            if abs(lo - start) <= eps and keyed(shot.start):
                start -= 1.0
        return start, end

    # ---- derived queries --------------------------------------------------

    def compute_gap(self) -> float:
        """Derive the predominant inter-shot gap from current shot positions.

        Returns the median gap between consecutive shots (rounded to the
        nearest integer).  When fewer than two shots exist the current
        ``self.gap`` value is returned unchanged.
        """
        shots = self.sorted_shots()
        if len(shots) < 2:
            return self.gap
        gaps = [
            max(0, round(shots[i + 1].start - shots[i].end))
            for i in range(len(shots) - 1)
        ]
        gaps.sort()
        mid = len(gaps) // 2
        median = gaps[mid] if len(gaps) % 2 else round((gaps[mid - 1] + gaps[mid]) / 2)
        return float(median)

    # ---- CRUD ------------------------------------------------------------

    def sorted_shots(self) -> List[ShotBlock]:
        """Return shots ordered by start time."""
        return sorted(self.shots, key=lambda s: s.start)

    def shot_by_id(self, shot_id: int) -> Optional[ShotBlock]:
        for s in self.shots:
            if s.shot_id == shot_id:
                return s
        return None

    def shot_by_name(self, name: str) -> Optional[ShotBlock]:
        """Return the first shot whose name matches *name*, or ``None``."""
        for s in self.shots:
            if s.name == name:
                return s
        return None

    # ---- names -----------------------------------------------------------

    def name_error(self, name: Any, shot_id: Optional[int] = None) -> Optional[str]:
        """Why *name* cannot name a shot in this store, or ``None`` when it can.

        A shot's name is its clip name on every carrier, so it has to be
        written as it will arrive: non-empty, :attr:`NAME_PATTERN` only, and
        unique among the other shots IGNORING CASE -- Unity joins clip names
        case-insensitively (``ShotMetadataController``), so ``Intro`` and
        ``intro`` would be one clip there.  :meth:`define_shot` and
        :meth:`update_shot` refuse what this refuses; a UI asks it first to
        say why.

        Parameters:
            name: The candidate name.
            shot_id: The shot being renamed, whose own name is no collision.

        Returns:
            A sentence for the user, or ``None``.
        """
        if not isinstance(name, str) or not name:
            return "A shot needs a name."
        illegal = StrUtils.name_error(
            name,
            subject="shot names",
            reason="because the name is the exported clip name",
        )
        if illegal:
            return illegal
        key = self._clip_key(name)
        for shot in self.shots:
            if shot.shot_id == shot_id or self._clip_key(shot.name) != key:
                continue
            spelled = self._sanitize_clip_name(shot.name)
            other = repr(shot.name)
            if spelled != shot.name:  # a legacy name, respelled on export
                other += f" (exported as {spelled!r})"
            return (
                f"{name!r}: another shot is named {other} -- clip names must "
                "differ by more than case."
            )
        # Legacy case twins export de-duplicated (``Intro`` / ``intro`` ship as
        # ``Intro`` / ``intro_1``), so a name can also collide with a SUFFIXED
        # clip no shot is named.
        others = [s for s in self.sorted_shots() if s.shot_id != shot_id]
        for clip, _start, _end in self.resolve_clip_specs(others):
            if clip.casefold() == name.casefold():
                return (
                    f"{name!r}: another shot already exports as {clip!r} -- "
                    "clip names must differ by more than case."
                )
        return None

    @staticmethod
    def unique_among(
        name: str, taken: Iterable[str], first: Optional[int] = None
    ) -> str:
        """*name* in its clip spelling, or the first ``name_<n>`` that collides
        with nothing in *taken* -- compared as :meth:`name_error` compares, by
        clip spelling ignoring case.

        The spelling comes first, so the result is always a legal name: a
        legacy one (``"Shot 1"``, from a merged store or a split tail) becomes
        ``Shot_1``, never ``Shot 1_2``.

        Parameters:
            name: The wanted name.
            taken: Names already in use.
            first: Number from ``name_<first>`` even when *name* itself is
                free (the ``Shot_<n>`` defaults).  ``None`` keeps *name* when
                free and otherwise numbers from 2.

        Returns:
            A name :attr:`NAME_PATTERN` accepts that collides with nothing in
            *taken*.
        """
        name = _ShotStoreInternal._sanitize_clip_name(name)
        key = _ShotStoreInternal._clip_key
        used = {key(t) for t in taken}
        if first is None:
            if key(name) not in used:
                return name
            n = 2
        else:
            n = int(first)
        while key(f"{name}_{n}") in used:
            n += 1
        return f"{name}_{n}"

    def unique_name(self, base: str = "Shot", first: Optional[int] = None) -> str:
        """A name :meth:`name_error` accepts, derived from *base* by
        :meth:`unique_among` against this store's shots.

        Returns:
            The derived name.
        """
        return self.unique_among(base, (s.name for s in self.shots), first)

    def default_name(self, wanted: Optional[str] = None) -> str:
        """*wanted* when :meth:`name_error` accepts it, else the store's next
        free default ``Shot_<n>``.

        For a name a TOOL proposes rather than the user types: detection
        numbers its candidates from 1, and a store that already has shots may
        hold those numbers -- the new shot is ``Shot_3``, not ``Shot_1_2``.

        Returns:
            A name this store accepts.
        """
        if wanted and self.name_error(wanted) is None:
            return wanted
        return self.unique_name("Shot", first=1)

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
        """Create a new shot and add it to the store.

        Parameters:
            name: The shot's name, which is also its exported clip name --
                see :meth:`name_error` (:meth:`unique_name` derives one).
            start: First frame.
            end: Last frame.
            objects: Transform node names.  ``None`` → empty list.
            metadata: Arbitrary key/value pairs.
            locked: Mark this shot as user-finalized.

        Returns:
            The newly created :class:`ShotBlock`.

        Raises:
            ValueError: *name* is not one the carriers write verbatim, or
                another shot already has it (:meth:`name_error` says which).
        """
        self._check_name(name)
        if objects is None:
            objects = []
        else:
            # Preserve the caller's name form (short or long).  Live path
            # reconciliation is a DCC concern handled by subclasses.
            objects = list(objects)
        new_id = max((s.shot_id for s in self.shots), default=-1) + 1
        block = ShotBlock(
            shot_id=new_id,
            name=name,
            start=self.snap(start),
            end=self.snap(end),
            objects=sorted(set(objects)),
            metadata=dict(metadata) if metadata else {},
            locked=locked,
            description=description,
        )
        self.shots.append(block)
        self._rekey_gap_locks()
        self._notify(ShotDefined(shot=block))
        self.mark_dirty()
        return block

    def update_shot(
        self,
        shot_id: int,
        *,
        start: Optional[float] = None,
        end: Optional[float] = None,
        name: Optional[str] = None,
        objects: Optional[List[str]] = None,
        description: Optional[str] = None,
        locked: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ShotBlock]:
        """Update fields on an existing shot.  Returns the shot, or ``None``.

        Raises:
            ValueError: A new *name* that :meth:`name_error` refuses.  Checked
                before any field changes, so a refused rename edits nothing.
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            return None
        if name is not None and name != shot.name:
            self._check_name(name, shot_id)
        if start is not None:
            shot.start = self.snap(start)
        if end is not None:
            shot.end = self.snap(end)
        # Guard against inverted bounds (e.g. an end spinner typed below
        # the shot's start): clamp to a zero-duration shot at the edited
        # edge — downstream envelope/respace math assumes start <= end.
        if shot.end < shot.start:
            if start is not None and end is None:
                shot.start = shot.end
            else:
                shot.end = shot.start
        if name is not None:
            shot.name = name
        if objects is not None:
            # Keep the caller's name form; lazy reconciliation handles renames.
            shot.objects = sorted(set(objects))
        if description is not None:
            shot.description = description
        if locked is not None:
            shot.locked = locked
        if metadata is not None:
            shot.metadata = dict(metadata)
        if objects is not None:
            self._prune_system_members(shot)
        self._notify(ShotUpdated(shot=shot))
        self.mark_dirty()
        return shot

    #: Shot metadata naming the members the shot system added on its own
    #: (:meth:`add_system_members`), by :meth:`member_key`: an empty shot's
    #: members filled in when the sequencer shows it, an object keyed in the
    #: active shot.  A tool comparing a shot's members with what it made -- the
    #: manifest's ``edited_by_hand`` -- reads past them: a member a hand added
    #: or removed is an edit, one the system filled in is not.
    SYSTEM_MEMBERS = "system_members"

    def add_system_members(self, shot_id: int, names: Iterable[str]) -> List[str]:
        """Add *names* to a shot's members as the shot system's own additions
        (:attr:`SYSTEM_MEMBERS`), keeping every member it has -- one the scene
        no longer resolves too (inert; reconciliation re-points, never drops).

        Parameters:
            shot_id: The shot to add to.
            names: The members to add; ones already members are skipped.

        Returns:
            The names added, sorted (none: nothing changed, no event).
        """
        shot = self.shot_by_id(shot_id)
        if shot is None:
            return []
        added = sorted({str(n) for n in names} - set(shot.objects))
        if not added:
            return []
        meta = dict(shot.metadata or {})
        tagged = set(meta.get(self.SYSTEM_MEMBERS) or ())
        meta[self.SYSTEM_MEMBERS] = sorted(tagged | {self.member_key(n) for n in added})
        self.update_shot(shot_id, objects=list(shot.objects) + added, metadata=meta)
        return added

    def system_members(self, shot: ShotBlock) -> set:
        """The :meth:`member_key` of every member of *shot* the shot system
        added (:meth:`add_system_members`) that the shot still holds."""
        tagged = (shot.metadata or {}).get(self.SYSTEM_MEMBERS) or ()
        keys = {self.member_key(n) for n in shot.objects}
        return {k for k in tagged if k in keys}

    def _prune_system_members(self, shot: ShotBlock) -> None:
        """Drop the :attr:`SYSTEM_MEMBERS` of members *shot* no longer holds:
        one a hand removes and adds back is the hand's."""
        tagged = (shot.metadata or {}).get(self.SYSTEM_MEMBERS)
        if not tagged:
            return
        kept = sorted(self.system_members(shot))
        if kept == sorted(tagged):
            return
        meta = dict(shot.metadata)
        if kept:
            meta[self.SYSTEM_MEMBERS] = kept
        else:
            meta.pop(self.SYSTEM_MEMBERS)
        shot.metadata = meta

    def remove_shot(self, shot_id: int) -> bool:
        """Remove a shot by ID.  Returns ``True`` if found.

        Records only: the keys the system wrote for the shot stay in the
        scene, so their claims stay too, owned by no shot
        (:meth:`ShotEditLedger.disown_shot`).  Left on the id, they passed to
        the next shot :meth:`define_shot` gave it -- whose Build then deleted
        them as behaviors its doc had dropped.  A restore point holds the
        ledger as it was, so undoing the removal gives them back.  Removing
        the active shot leaves none active (:class:`ActiveShotChanged`).
        """
        for i, s in enumerate(self.shots):
            if s.shot_id == shot_id:
                self.shots.pop(i)
                self.edit_ledger.disown_shot(shot_id)
                self._rekey_gap_locks()
                self._notify(ShotRemoved(shot_id=shot_id))
                if shot_id == self._active_shot_id:
                    # An active id naming no shot is a dangling reference
                    # every reader would have to guard.
                    self.set_active_shot(None)
                self.mark_dirty()
                return True
        return False

    def _rekey_gap_locks(self) -> None:
        """Re-key the gap locks onto the current adjacency after a shot is
        added or removed.

        A lock names its gap by the two shot ids flanking it, and the pair
        went stale the moment the shots around it changed: a shot inserted
        into a locked gap left both gaps that replaced it unlocked, and a
        shot deleted from between two locked gaps took both pairs with it
        while the gap that took their place opened unlocked -- with nothing
        said either time (2026-09-07: "I locked all gaps and they
        automatically became unlocked a few operations later").

        The lock is on the GAP, and the gap is wherever the two shots'
        content now meets.  A locked pair that is still adjacent stands.
        One whose shots both exist but are no longer adjacent becomes every
        adjacent pair between them, so an inserted or split-off shot inherits
        the lock on both sides.  One that lost a shot becomes the gap its
        survivor now flanks on that side -- the gap after a surviving left
        shot, before a surviving right one -- and one that lost both is
        dropped.
        """
        if not self.locked_gaps:
            return
        ordered = self.sorted_shots()
        pos = {shot.shot_id: i for i, shot in enumerate(ordered)}
        adjacent = {(a.shot_id, b.shot_id) for a, b in zip(ordered, ordered[1:])}
        rekeyed: set = set()
        for left, right in self.locked_gaps:
            if (left, right) in adjacent:
                rekeyed.add((left, right))
                continue
            i, j = pos.get(left), pos.get(right)
            if i is not None and j is not None:
                span = range(min(i, j), max(i, j))
            elif i is not None:
                span = range(i, min(i + 1, len(ordered) - 1))
            elif j is not None:
                span = range(max(j - 1, 0), j)
            else:
                continue
            for k in span:
                rekeyed.add((ordered[k].shot_id, ordered[k + 1].shot_id))
        if rekeyed != self.locked_gaps:
            self.locked_gaps = rekeyed
            self.mark_dirty()

    def append_shot(
        self,
        name: str,
        duration: float,
        gap: float = 0,
        start_frame: Optional[float] = None,
        objects: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        locked: bool = False,
        description: str = "",
    ) -> ShotBlock:
        """Append a shot after the last existing shot, with gap-aware placement.

        Parameters:
            name: The shot's name -- its clip name on every carrier, so it
                must pass :meth:`name_error`.
            duration: Shot duration in frames.
            gap: Gap frames after the previous shot.
            start_frame: Explicit start frame.  If ``None``, computed
                from the last shot's end + *gap*.
            objects: Transform node names.
            metadata: Arbitrary key/value pairs.
            locked: Mark this shot as user-finalized.

        Returns:
            The newly created :class:`ShotBlock`.

        Raises:
            ValueError: *name* is one :meth:`name_error` refuses.
        """
        if start_frame is None:
            sorted_s = self.sorted_shots()
            start_frame = (sorted_s[-1].end + gap) if sorted_s else 0.0
        return self.define_shot(
            name=name,
            start=start_frame,
            end=start_frame + duration,
            objects=objects,
            metadata=metadata,
            locked=locked,
            description=description,
        )

    # ---- visibility ------------------------------------------------------

    def is_object_hidden(self, obj_name: str) -> bool:
        """Return True if *obj_name* is hidden in the sequencer UI."""
        return obj_name in self.hidden_objects

    def set_object_hidden(self, obj_name: str, hidden: bool = True) -> None:
        """Show or hide *obj_name* in the sequencer UI."""
        if hidden == (obj_name in self.hidden_objects):
            return
        if hidden:
            self.hidden_objects.add(obj_name)
        else:
            self.hidden_objects.discard(obj_name)
        self.mark_dirty()

    # ---- pinning ---------------------------------------------------------

    def is_object_pinned(self, obj_name: str) -> bool:
        """Return True if *obj_name* is pinned (kept even when missing)."""
        return obj_name in self.pinned_objects

    def set_object_pinned(self, obj_name: str, pinned: bool = True) -> None:
        """Pin or unpin *obj_name*.

        Pinned objects remain visible in the sequencer with a
        'missing' indicator when they no longer exist in the scene.
        Non-pinned objects are silently removed from tracks.
        """
        if pinned == (obj_name in self.pinned_objects):
            return
        if pinned:
            self.pinned_objects.add(obj_name)
        else:
            self.pinned_objects.discard(obj_name)
        self.mark_dirty()

    # ---- object removal --------------------------------------------------

    def remove_object_from_shots(self, obj_name: str) -> None:
        """Remove *obj_name* from every shot's object list."""
        changed = False
        for shot in self.shots:
            if obj_name in shot.objects:
                shot.objects.remove(obj_name)
                changed = True
        if obj_name in self.pinned_objects:
            self.pinned_objects.discard(obj_name)
            changed = True
        if obj_name in self.hidden_objects:
            self.hidden_objects.discard(obj_name)
            changed = True
        if changed:
            self.mark_dirty()

    # ---- renaming --------------------------------------------------------

    #: The store's object-name sets, re-pointed with the members they name.
    OBJECT_SETS = ("hidden_objects", "pinned_objects", "locked_objects")

    def held_names(self) -> set:
        """Every name the store holds an object by -- shot members and the
        :attr:`OBJECT_SETS` -- which is what :meth:`rename_members` re-points."""
        names = {o for shot in self.shots for o in shot.objects}
        for attr in self.OBJECT_SETS:
            names |= getattr(self, attr)
        return names

    def rename_members(self, renames: Dict[str, str]) -> bool:
        """Re-point every name the store holds an object by: ``{old: new}``.

        Shot members and the :attr:`OBJECT_SETS` alike, so a re-pointed member
        keeps its pin (a pin left on the old name no longer keeps the member)
        and its hidden / locked state.  Two names re-pointed at one node become
        one member: a doc name a build stored while it could not resolve it,
        once the scene scan has added the node beside it, is the same object
        twice.  The host decides what each name now is -- a followed rename,
        a stale path re-resolved, a doc name now found; this applies it.

        Parameters:
            renames: ``{old name: new name}``; an entry naming itself is
                ignored.

        Returns:
            ``True`` when anything was re-pointed.
        """
        renames = {old: new for old, new in renames.items() if old != new}
        if not renames:
            return False
        changed = False
        with self.batch_update():
            for shot in self.shots:
                if not renames.keys() & set(shot.objects):
                    continue
                # update_shot sorts and de-duplicates: two names, one member.
                self.update_shot(
                    shot.shot_id, objects=[renames.get(o, o) for o in shot.objects]
                )
                changed = True
            for attr in self.OBJECT_SETS:
                held = getattr(self, attr)
                if renames.keys() & held:
                    setattr(self, attr, {renames.get(n, n) for n in held})
                    changed = True
        if changed:
            self.mark_dirty()
        return changed

    # ---- serialisation ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise shots and settings to a plain dict."""
        return {
            "shots": [
                {
                    "shot_id": s.shot_id,
                    "name": s.name,
                    "start": s.start,
                    "end": s.end,
                    "objects": list(s.objects),
                    "metadata": dict(s.metadata) if s.metadata else {},
                    "locked": s.locked,
                    "description": s.description,
                }
                for s in self.sorted_shots()
            ],
            "hidden_objects": sorted(self.hidden_objects),
            "pinned_objects": sorted(self.pinned_objects),
            "markers": list(self.markers),
            "gap": self.gap,
            "detection_threshold": self.detection_threshold,
            "detection_mode": self.detection_mode,
            "initial_shot_length": self.initial_shot_length,
            "fit_mode": self.fit_mode,
            "snap_whole_frames": self.snap_whole_frames,
            "select_on_load": self.select_on_load,
            "frame_on_shot_change": self.frame_on_shot_change,
            "locked_gaps": [list(pair) for pair in sorted(self.locked_gaps)],
            "scene_fps": self.scene_fps,
            "source_csv": self.source_csv,
            "manifest_edits": dict(self.manifest_edits),
            "manifest_tags": {k: dict(v) for k, v in self.manifest_tags.items()},
            "manifest_options": {
                k: {
                    "template": v.get("template"),
                    "values": dict(v.get("values") or {}),
                }
                for k, v in self.manifest_options.items()
            },
            "auto_publish_export": self.auto_publish_export,
            "clip_name_strategy": self.clip_name_strategy,
            "edit_ledger": self.edit_ledger.to_dict(),
            "effect_recipe": self.effect_recipe.to_dict(),
            "member_ids": self._capture_member_ids(),
        }

    def stale_shots(self) -> List[ShotBlock]:
        """The shots that no longer describe anything in the scene, in order.

        A shot names the objects it animates.  One whose every member is gone
        from the scene (:meth:`_existing_objects`), and inside whose window
        nothing at all is keyed (:meth:`_keyed_windows`), holds what the scene
        USED to animate: a Save As copy whose animated objects were deleted
        keeps its source's shots verbatim (2026-09-24: a module forked from an
        18-shot assembly declared all 18 as takes, and its GLB -- nothing in
        it animated -- carried none of them).

        Both halves must hold.  A keyed shot whose members were renamed away
        keeps its keys in the window, so it is not stale.  A hold shot keys
        nothing there, so only its members say it is live -- and members are
        stored by name: a rename the store is not told about (by
        :meth:`update_shot`, as a host's own renaming pass must) leaves it
        naming nothing, and it reads as stale unless the members' lookup
        still finds them (:meth:`_existing_objects`).  A shot that names no
        members is never stale: nothing tells it from a hold.  Asking changes
        nothing: :meth:`export_records` and :meth:`declared_range` leave stale
        shots out of what an export declares, and only
        :meth:`remove_stale_shots` -- the user's call -- takes them out of the
        store.
        """
        stale, _present = self._scene_scope()
        return [s for s in self.sorted_shots() if s.shot_id in stale]

    def remove_stale_shots(self) -> List[ShotBlock]:
        """Remove every stale shot (:meth:`stale_shots`); return what went.

        Records only.  A stale shot keys nothing in its window and names no
        object the scene holds, so there is nothing of it to cut and nothing
        moves -- where a Shot Sequencer delete cuts a shot's keys and ripples
        every later shot upstream into the space it leaves, which would retime
        the shots that ARE live.  The claims each removed shot held -- its
        bound samples and its behaviors' keys -- go with it rather than being
        disowned: they name keys the scene no longer holds, and a claim left
        behind is inherited by any key later set on that curve and frame.  The
        gap locks re-key onto the shots that remain (:meth:`remove_shot`), and
        listeners hear one batch.
        """
        stale = self.stale_shots()
        if not stale:
            return []
        ids = {s.shot_id for s in stale}
        ledger = self.edit_ledger
        with self.batch_update():
            for curve in ledger.keyed_curves():
                for time, owner, _edge in ledger.key_records(curve):
                    if owner in ids:
                        ledger.release_key(curve, time)
            for shot_id in ids:
                for curve, time in ledger.authored(owner=shot_id):
                    ledger.release_authored(curve, time)
            for shot in stale:
                self.remove_shot(shot.shot_id)
        return stale

    def empty_shots(self) -> List[ShotBlock]:
        """The shots that hold nothing, in order: nothing they name is keyed
        inside their frames -- and, for a shot that names nothing (or nothing
        the scene still holds), nothing in the scene is -- and no audio plays
        in them (:meth:`_audio_windows`).

        What the Shot Sequencer draws as an empty band (2026-10-10: "delete
        empty is leaving behind empty shots").  Naming objects used to keep a
        shot out of this scope, as a hold -- but a new shot names the objects
        keyed on its start (at a zero gap, the shot before's last frame), and
        a shot whose keys were deleted keeps its members: both held nothing
        and stayed.  A hold that keys nothing in its own frames is empty by
        this measure too; the deletes ask first, naming every shot.  A stale
        shot is empty as well (:meth:`stale_shots` is the narrower scope).

        A key ON a bound counts only where no neighbour shares the frame: at
        a zero gap it is the neighbour's as much as this shot's
        (:data:`_BOUND_TOL`).  The pure model cannot see keys
        (:meth:`_members_keyed_windows`, :meth:`_keyed_windows`), so it never
        calls a shot empty.
        """
        shots = self.sorted_shots()
        windows = []
        for i, s in enumerate(shots):
            shared_lo = i > 0 and shots[i - 1].end >= s.start - _BOUND_TOL
            shared_hi = i + 1 < len(shots) and shots[i + 1].start <= s.end + _BOUND_TOL
            lo = s.start + (_BOUND_TOL if shared_lo else -_BOUND_TOL)
            hi = s.end + (-_BOUND_TOL if shared_hi else _BOUND_TOL)
            windows.append((lo, hi) if lo <= hi else (s.start, s.end))
        named = [i for i, s in enumerate(shots) if s.objects]
        keyed: List[Optional[bool]] = [None] * len(shots)
        if named:
            answers = self._members_keyed_windows(
                [(list(shots[i].objects), windows[i]) for i in named]
            )
            for i, answer in zip(named, answers):
                keyed[i] = answer
        # Naming nothing the scene holds: the scene says.
        rest = [i for i, k in enumerate(keyed) if k is None]
        if rest:
            for i, answer in zip(rest, self._keyed_windows([windows[i] for i in rest])):
                keyed[i] = answer
        # Audio is a shot's content too (the Sequencer draws its clips), and a
        # clip can play through a shot with no key of its own inside it.
        quiet = [i for i, k in enumerate(keyed) if not k]
        if quiet:
            for i, plays in zip(
                quiet, self._audio_windows([windows[i] for i in quiet])
            ):
                keyed[i] = plays
        return [s for s, k in zip(shots, keyed) if not k]

    #: What :meth:`shots_in_scope` and :meth:`remove_shots_in_scope` take, in
    #: the order a host offers them: the narrowest first.
    DELETE_SCOPES = ("stale", "empty", "all")

    def shots_in_scope(self, scope: str) -> List[ShotBlock]:
        """The shots a delete of *scope* takes, in order.

        Parameters:
            scope: ``"stale"`` (:meth:`stale_shots`), ``"empty"``
                (:meth:`empty_shots`) or ``"all"``.

        Raises:
            ValueError: *scope* is none of :attr:`DELETE_SCOPES`.
        """
        if scope == "stale":
            return self.stale_shots()
        if scope == "empty":
            return self.empty_shots()
        if scope == "all":
            return self.sorted_shots()
        raise ValueError(
            f"shot delete scope {scope!r} is not one of {self.DELETE_SCOPES}"
        )

    def remove_shots_in_scope(self, scope: str) -> List[ShotBlock]:
        """Remove the shots of *scope* (:meth:`shots_in_scope`); return what
        went.

        Records only, like :meth:`remove_stale_shots` (which the ``"stale"``
        scope is): no key is touched and no other shot moves.  An empty shot
        keys nothing to cut; "all" leaves every shot's animation in the scene,
        the claims it held disowned (:meth:`remove_shot`).  Listeners hear one
        batch.
        """
        if scope == "stale":
            return self.remove_stale_shots()
        doomed = self.shots_in_scope(scope)
        if doomed:
            with self.batch_update():
                for shot in doomed:
                    self.remove_shot(shot.shot_id)
        return doomed

    def _scene_scope(self) -> Tuple[set, Optional[set]]:
        """``(stale shot ids, members the scene still holds)`` -- each hook
        asked once, for :meth:`stale_shots` and the export view it scopes.

        ``None`` for the members means "keep them all".  A hook that raises
        leaves the export as it was before the check -- nothing stale, every
        member kept -- and says so: the check must never cost the shot record,
        or the stale one on the carrier ships in its place.
        """
        listed = [s for s in self.shots if s.objects]
        if not listed:
            return set(), None
        try:
            present = self._existing_objects({o for s in listed for o in s.objects})
            orphaned = [s for s in listed if not any(o in present for o in s.objects)]
            keyed = (
                self._keyed_windows([(s.start, s.end) for s in orphaned])
                if orphaned
                else []
            )
        except Exception:  # a DCC lookup; the record must still be produced
            _log.warning(
                "Shots: could not check the shots against the scene; every shot "
                "is declared as held.",
                exc_info=True,
            )
            return set(), None
        return {s.shot_id for s, k in zip(orphaned, keyed) if not k}, present

    @staticmethod
    def _stale_note(stale: List[ShotBlock]) -> str:
        """One sentence naming the shots an export left out as stale."""
        named = [f"{s.name} ({s.start:g}-{s.end:g})" for s in stale]
        shown = ", ".join(named[:8])
        if len(named) > 8:
            shown += f" and {len(named) - 8} more"
        # The remedy is named by what it does, not where a host offers it: a
        # host links its own remedy onto the note (``ExportSnapshot.noted``).
        return (
            f"{len(stale)} shot(s) left out of the export -- every object they "
            f"name is gone from the scene and nothing is keyed in their frames: "
            f"{shown}. A scene saved from another keeps that scene's shots; if "
            "these no longer apply, remove their records "
            "(ShotStore.remove_stale_shots) -- no key or other shot moves."
        )

    def to_export_view(self, strategy: str = "name") -> Dict[str, Any]:
        """Build the FBX/Unity export view from the current shots.

        Returns ``{"shot_metadata": {...}}``: one entry per clip -- its name
        (the FBX take and the ``clip`` join key, from a single
        :func:`resolve_clip_specs` pass, so they cannot drift), its
        ``start`` / ``end`` frames, its objects, and a description or section
        only when the shot has one.  One record, one join key: the ranges
        are the take list, so there is no second list to disagree with it
        (``fbx_takes``, dropped in 0.11.0 -- readers of older files go through
        ``SceneRecords.declared_takes``).

        ``fps`` rides on the metadata envelope rather than per shot: the ranges
        it qualifies are the store's, not any one shot's, and without it every
        frame number in this view is unitless to a consumer that did not author
        the scene (``MeshConvert.apply_glb_animations`` is one -- it reads the
        published view back out of the deliverable).  Additive, so the schema
        version stands: a reader that does not know the key is unaffected.

        Every shot, as the store holds it.  What an export publishes is
        :meth:`export_records`, scoped to the scene: stale shots
        (:meth:`stale_shots`) and members the scene no longer holds left out.
        """
        return self._export_view(strategy)

    #: Metadata key of a shot's STEP LINK: the doc step a shot split off a
    #: step's shot belongs with (``ShotSequencer.split_shot`` writes it).
    #: Non-binding -- the step's shot is the one ``metadata["step"]`` binds,
    #: and a step has one -- so the link only says where the shot belongs:
    #: the export gives it the step's section, the manifest shows it there
    #: (:meth:`step_links`).
    STEP_LINK = "step_link"

    def step_links(self) -> Dict[int, str]:
        """``{shot_id: step_id}``: the doc step each shot no step is bound to
        belongs with.  Read only -- asking writes nothing.

        A shot bound to a step (``metadata["step"]``) is that step's own and
        is not listed.  Any other shot is linked by the link the split wrote
        (:attr:`STEP_LINK`), or -- a tail split before the split wrote one --
        by its name: the split names a tail ``<name>_2`` (numbered on, when
        taken), so ``<step>_<n>`` where ``<step>`` is a step some shot is
        bound to links it there, one ``_<n>`` stripped at a time (a tail
        split again, ``A01_2_2``, finds ``A01``).  Derived on every read,
        never stamped on the scene, so opening a scene changes nothing
        (maintainer 2026-10-11; SOL_REPLACE_ASSEMBLY's 8 tails, A01_2 ...
        E02_2, carry no link).

        Returns:
            ``{shot_id: step_id}`` for every linked shot.
        """
        shots = self.sorted_shots()
        bound = {
            str((s.metadata or {}).get("step"))
            for s in shots
            if (s.metadata or {}).get("step")
        }
        links: Dict[int, str] = {}
        for shot in shots:
            meta = shot.metadata or {}
            if meta.get("step"):
                continue
            link = meta.get(self.STEP_LINK)
            if not link:
                name = str(shot.name)
                while True:
                    head, sep, n = name.rpartition("_")
                    if not (sep and head and n.isdigit()):
                        break
                    if head in bound:
                        link = head
                        break
                    name = head
            if link:
                links[shot.shot_id] = str(link)
        return links

    def _export_view(
        self,
        strategy: str,
        stale: Iterable[int] = (),
        present: Optional[set] = None,
    ) -> Dict[str, Any]:
        """:meth:`to_export_view`, less the *stale* shot ids and -- when
        *present* is given -- every member it does not hold.

        Every shot is NAMED before any is left out: the ``sequence`` strategy
        numbers clips by position, and leaving one out first would renumber
        every clip after it.

        A shot's section is its own (a build writes it), else that of the
        shot bound to the step it is linked to (:meth:`step_links`): a shot
        split off a step's shot is in that step's section.
        """
        stale = set(stale)
        sorted_s = self.sorted_shots()
        links = self.step_links()
        step_sections = {}
        for shot in sorted_s:
            meta = shot.metadata or {}
            if meta.get("step") and meta.get("section"):
                step_sections.setdefault(str(meta["step"]), meta["section"])
        # A member renamed behind the store's back ships by its node's name
        # now (:meth:`_current_names`); the store is not changed.
        try:
            current = self._current_names({o for s in sorted_s for o in s.objects})
        except Exception:  # a DCC lookup; the record must still be produced
            _log.debug("Shots: renamed members not looked up.", exc_info=True)
            current = {}
        specs = ShotStore.resolve_clip_specs(sorted_s, strategy=strategy)
        # The store refuses a name the carrier would respell, so a respelling
        # here is a name held from before that rule.  Said, never silent.
        fn = CLIP_NAME_STRATEGIES.get(strategy, CLIP_NAME_STRATEGIES["name"])
        wanted = [fn(i, shot) for i, shot in enumerate(sorted_s)]
        respelled = [
            f"{want!r} -> {clip!r}"
            for want, (clip, _s, _e), shot in zip(wanted, specs, sorted_s)
            if clip != want and shot.shot_id not in stale
        ]
        if respelled:
            _log.warning(
                "Exporting %d shot name(s) as different clip names: %s. Rename "
                "the shot(s) (%s, unique ignoring case) to ship the names shown.",
                len(respelled),
                ", ".join(respelled),
                self.NAME_RULE,
            )
        shots_meta = []
        for (clip, start, end), shot in zip(specs, sorted_s):
            if shot.shot_id in stale:
                continue
            entry: Dict[str, Any] = {"clip": clip, "start": start, "end": end}
            if shot.description:
                entry["description"] = shot.description
            members = shot.objects
            if present is not None:
                members = [o for o in members if o in present]
            # Always present, even empty: Unity's ShotRecord reads it as an array.
            entry["objects"] = [ShotStore.leaf_name(current.get(o, o)) for o in members]
            section = (shot.metadata or {}).get("section") or step_sections.get(
                links.get(shot.shot_id)
            )
            if section:
                entry["section"] = section
            shots_meta.append(entry)
        return {
            "shot_metadata": {
                "version": SceneRecords.SHOTS.version,
                "fps": self.scene_fps,
                "shots": shots_meta,
            },
        }

    def _export_set_members(self, names: Iterable[str], scope: Iterable[str]) -> set:
        """Which member *names* an export of *scope* carries (overridable hook).

        *scope* is ``ExportContext.scope``: the export set's roots, as the
        host spells its nodes; a member is carried when it IS a root or sits
        under one.  Pure default: DAG-path spelling (``"|grp|obj"`` sits
        under ``"|grp"``), and a bare name is its own root.  DCC stores
        override to resolve both sides in the scene (a short Maya name, a
        Blender parent).
        """
        roots = [str(r) for r in scope]
        return {
            n for n in names if any(n == r or str(n).startswith(r + "|") for r in roots)
        }

    def _member_ids(self, names: Iterable[str]) -> Dict[str, str]:
        """``{name: stable id}`` for each of *names* that names ONE node now
        (overridable hook; Maya: its node UUID).  Pure default: none -- the
        pure model has no scene to give an id."""
        return {}

    def _capture_member_ids(self) -> Dict[str, str]:
        """:attr:`member_ids` brought up to date for :meth:`to_dict`: every
        name the store holds an object by keeps an id -- the one its node has
        now (:meth:`_member_ids`), else the one saved for it (a name a rename
        left behind: the id is what finds the node again).  Names the store
        no longer holds are dropped.  Never raises: a failing lookup keeps
        the ids saved."""
        held = self.held_names()
        ids = {name: uid for name, uid in self.member_ids.items() if name in held}
        try:
            ids.update(self._member_ids(sorted(held)))
        except Exception:  # a DCC lookup; the save must go on
            _log.debug("Shots: member ids not refreshed.", exc_info=True)
        self.member_ids = ids
        return dict(ids)

    def _current_names(self, names: Iterable[str]) -> Dict[str, str]:
        """``{name: current name}`` for each of *names* no object answers to
        any more but whose node is found another way -- a rename the store
        was not told of (overridable hook; read-only, the store is not
        changed).  Maya finds the node by its saved UUID
        (:attr:`member_ids`), Blender by the rename it follows.  Pure default:
        none.  A HOLD shot -- nothing keyed in its frames -- is live only
        through its members, so without this a renamed member made it stale
        and its take left the export (2026-09-26)."""
        return {}

    #: Key on the shot record under which the exporter's Animation Clips mode
    #: is declared (``full`` / ``shots`` / ``both``).  The clips carry their
    #: ranges in every mode, so only this says whether the file was meant to
    #: carry them as clips.  ``MeshConvert.SHOT_CLIP_MODE_KEY`` is
    #: the reader's name for the same key.
    CLIP_MODE_KEY = "clip_mode"

    def export_records(
        self, ctx: Optional[ExportContext] = None, strategy: Optional[str] = None
    ) -> Optional[List[Record]]:
        """This store's shot records, ready to store -- ``None`` when empty.

        The producer form of :meth:`to_export_view`: the shot record, with the
        context's clip mode declared on it when the exporter decided one.  An
        empty store yields ``None``, which a commit turns into a CLEAR --
        deleting the last shot must not leave the previous takes riding into
        the next export.  Either way the commit also clears a legacy
        ``fbx_takes`` a scene still holds (its successor was produced).

        Scoped to the scene as it is NOW, since the record is the deliverable's
        take list: a stale shot (:meth:`stale_shots`) is left out -- its take
        would carry nothing -- and so is each member the scene no longer holds.
        What was left out is said once, on *ctx* (``ExportContext.note``) or,
        publishing without one, in the log; a store whose every shot is stale
        yields ``None`` like an empty one.

        Scoped to the export set too (``SceneRecords.SHOTS`` is
        ``export_scoped``): with ``ctx.scope`` each shot lists only the
        members the write carries (:meth:`_export_set_members`) -- a selected
        or visible-only export no longer names nodes its file leaves out.
        Every take still ships: a shot is a span of time, and the frames are
        the file's whatever it carries.  The scene keeps its whole record
        (``ExportSnapshot.restore_scene_wide``).

        Returns:
            ``[shot record]``, or ``None`` for an empty or wholly stale store.
        """
        if not self.shots:
            return None
        stale, present = self._scene_scope()
        if ctx is not None and ctx.scope is not None:
            listed = {o for s in self.shots for o in s.objects}
            held = listed if present is None else listed & present
            try:
                present = self._export_set_members(held, ctx.scope)
            except Exception:  # a DCC lookup; the record must still be produced
                _log.warning(
                    "Shots: could not narrow the shot members to the export set; "
                    "every member the scene holds is listed.",
                    exc_info=True,
                )
        view = self._export_view(
            strategy or self.clip_name_strategy, stale=stale, present=present
        )
        if stale:
            note = self._stale_note(
                [s for s in self.sorted_shots() if s.shot_id in stale]
            )
            if ctx is not None:
                ctx.note(note)
            else:  # an authoring publish, after every shot edit: not a warning
                _log.info(note)
        meta = view["shot_metadata"]
        if not meta["shots"]:
            return None
        if ctx is not None and ctx.clip_mode:
            meta[self.CLIP_MODE_KEY] = ctx.clip_mode
        return [SceneRecords.SHOTS.make(meta)]

    @classmethod
    def produce_export_records(
        cls, ctx: Optional[ExportContext] = None
    ) -> Optional[List[Record]]:
        """The Shots PRODUCER: the active store's records, or ``None``.

        What a DCC's export producer table names for ``SceneRecords.SHOTS``.
        ``active()`` loads the scene's store when none is live, so an empty
        scene yields ``None`` and the commit clears whatever an earlier scene
        state published -- the same "an empty store still publishes" rule
        :meth:`refresh_export_view` has always had.

        Returns:
            :meth:`export_records` of the active store, or ``None``.
        """
        store = cls.active()
        return store.export_records(ctx) if store is not None else None

    #: How many :meth:`transient_edit` brackets are open in this process --
    #: on ``ShotStore`` itself, so every DCC subclass reads the one count.
    _transient_depth: int = 0

    @staticmethod
    @contextmanager
    def transient_edit():
        """Bracket a batch that edits the scene and puts it back -- an
        export's bake, snap and tie, a bridge send's staged proxies: none of
        its key edits is the user's, so a shot listener credits none of them
        to a shot (:meth:`in_transient_edit`).

        Measured 2026-10-05 (blendertk, a Unity bridge send): the send's
        Action burst banked every animated object, and the Shot Sequencer's
        keying refresh added a non-member keyed in the active shot's frames
        to its members -- after the send had restored the scene.  Re-entrant;
        held while any bracket is open, released if the body raises.
        """
        ShotStore._transient_depth += 1
        try:
            yield
        finally:
            ShotStore._transient_depth = max(ShotStore._transient_depth - 1, 0)

    @staticmethod
    def in_transient_edit() -> bool:
        """Whether a :meth:`transient_edit` bracket is open: a listener that
        follows the user's key edits (a Shot Sequencer's keying refresh)
        ignores the edits made inside one."""
        return ShotStore._transient_depth > 0

    @classmethod
    def refresh_export_view(cls) -> None:
        """Republish the active store's export view.

        The canonical, no-arg pre-export refresh for the Shots system.  A no-op
        when no store is active, and — via the :meth:`publish_export_view`
        hook — a no-op in the pure core until a DCC subclass supplies a real
        projection.  An *empty* store still publishes: the projection clears
        any previously published channels (without ever creating a carrier),
        so deleting the last shot can't leave stale takes riding into the
        next export.
        """
        store = cls.active()
        if store is not None:
            store.publish_export_view()

    #: Explicit user opt-out (``disable_auto_export``) — session-global; wins
    #: over the automatic registration that authoring a shot performs.
    _auto_export_disabled = False

    @classmethod
    def enable_auto_export(cls) -> None:
        """Opt every export this session into carrying the active store's shots.

        Registers a before-export preparer (via the :meth:`_register_export_preparer`
        hook) that republishes the export view fresh from whatever store is
        active.  Session-global and idempotent.  Call :meth:`disable_auto_export`
        to remove it.  In the pure core the preparer registration is a no-op
        until a DCC subclass overrides the hook.
        """
        cls._auto_export_disabled = False
        cls._register_export_preparer()

    @classmethod
    def disable_auto_export(cls) -> None:
        """Remove the before-export preparer for the rest of the session.

        An explicit opt-out: the automatic registration performed by
        :meth:`save` respects it and won't re-install the hook.
        """
        cls._auto_export_disabled = True
        cls._unregister_export_preparer()

    @classmethod
    def _register_export_preparer(cls) -> None:
        """Install the before-export preparer (overridable hook).

        Pure default: honour the opt-out flag, then no-op — the pure core has
        no DCC exporter to hook.  DCC subclasses override to register
        :meth:`refresh_export_view` with their FBX exporter.
        """
        if cls._auto_export_disabled:
            return
        # No DCC exporter in the pure core — subclasses override.

    @classmethod
    def _unregister_export_preparer(cls) -> None:
        """Remove the before-export preparer (overridable hook).

        Pure default: no-op.  DCC subclasses override to unregister from their
        FBX exporter.
        """
        # No DCC exporter in the pure core — subclasses override.

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShotStore":
        """Restore from serialised data.

        Parameters:
            data: Dict with ``"shots"`` list and optional
                ``"hidden_objects"`` / ``"markers"`` keys.
        """
        shot_list = data.get("shots", [])
        hidden = data.get("hidden_objects", [])
        pinned = data.get("pinned_objects", [])
        shots = [
            ShotBlock(
                shot_id=d["shot_id"],
                name=d["name"],
                start=d["start"],
                end=d["end"],
                objects=d.get("objects", []),
                metadata=d.get("metadata", {}),
                locked=d.get("locked", False),
                description=d.get("description", ""),
            )
            for d in shot_list
        ]
        store = cls(shots)
        store.hidden_objects = set(hidden)
        store.pinned_objects = set(pinned)
        store.markers = data.get("markers", [])
        store.gap = float(data.get("gap", cls.LEGACY_GAP))
        store.detection_threshold = float(data.get("detection_threshold", 5.0))
        # Migrate legacy use_selected_keys + key_filter_mode if present
        dm = data.get("detection_mode")
        if dm is not None:
            store.detection_mode = str(dm)
        elif data.get("use_selected_keys"):
            kf = data.get("key_filter_mode", "all")
            store.detection_mode = (
                str(kf) if kf in ("all", "skip_zero", "zero_as_end") else "all"
            )
        store.select_on_load = bool(data.get("select_on_load", False))
        try:
            store.initial_shot_length = float(
                data.get("initial_shot_length", cls.DEFAULT_INITIAL_SHOT_LENGTH)
            )
        except (TypeError, ValueError):
            store.initial_shot_length = cls.DEFAULT_INITIAL_SHOT_LENGTH
        fm = data.get("fit_mode")
        store.fit_mode = str(fm) if fm in cls.FIT_MODES else cls.DEFAULT_FIT_MODE
        snap = data.get("snap_whole_frames")
        store.snap_whole_frames = (
            bool(snap) if snap is not None else cls.DEFAULT_SNAP_WHOLE_FRAMES
        )
        store.frame_on_shot_change = bool(data.get("frame_on_shot_change", True))
        store.locked_gaps = {tuple(pair) for pair in data.get("locked_gaps", [])}
        stored_fps = data.get("scene_fps")
        if stored_fps is not None:
            store.scene_fps = float(stored_fps)
        store.source_csv = str(data.get("source_csv", "") or "")
        edits = data.get("manifest_edits")
        store.manifest_edits = (
            {str(k): v for k, v in edits.items() if isinstance(v, dict)}
            if isinstance(edits, dict)
            else {}
        )
        tags = data.get("manifest_tags")
        store.manifest_tags = {
            str(source): {
                str(step): slot
                for step, slot in layer.items()
                if isinstance(slot, str) and slot
            }
            for source, layer in (tags.items() if isinstance(tags, dict) else ())
            if isinstance(layer, dict)
        }
        options = data.get("manifest_options")
        store.manifest_options = {
            str(source): {
                "template": rec.get("template")
                if isinstance(rec.get("template"), str)
                else None,
                "values": dict(rec["values"])
                if isinstance(rec.get("values"), dict)
                else {},
            }
            for source, rec in (options.items() if isinstance(options, dict) else ())
            if isinstance(rec, dict)
        }
        ids = data.get("member_ids")
        store.member_ids = {
            str(name): uid
            for name, uid in (ids.items() if isinstance(ids, dict) else ())
            if isinstance(uid, str) and uid
        }
        store.auto_publish_export = bool(data.get("auto_publish_export", False))
        strat = data.get("clip_name_strategy")
        if strat in CLIP_NAME_STRATEGIES:
            store.clip_name_strategy = str(strat)
        store.edit_ledger = ShotEditLedger.from_dict(data.get("edit_ledger"))
        # Claims a removed shot left on its id (data saved before removal
        # disowned them) would pass to the next shot given that id.
        store.edit_ledger.disown_absent(shot.shot_id for shot in store.shots)
        store.effect_recipe = EffectRecipe.from_dict(data.get("effect_recipe"))
        return store

    # ---- persistence convenience -----------------------------------------

    def rescale_to_fps(self, new_fps: float) -> None:
        """Scale all shot timings from the current ``scene_fps`` to *new_fps*.

        Called automatically when the scene framerate changes.  Updates
        ``scene_fps``, rescales shot boundaries, gap, markers and the edit
        ledger's claims (the keys they name moved with the clock), then
        fires a :class:`BatchComplete` so the UI repaints.
        """
        old_fps = self.scene_fps
        if not old_fps or abs(new_fps - old_fps) < 0.01:
            return
        ratio = new_fps / old_fps
        # Route through snap() so sub-frame bounds survive a time-unit
        # change when snap_whole_frames is off (bare round() quantized
        # them unconditionally).
        for shot in self.shots:
            shot.start = self.snap(shot.start * ratio)
            shot.end = self.snap(shot.end * ratio)
        self.gap = round(self.gap * ratio, 2)
        for marker in self.markers:
            if "time" in marker:
                marker["time"] = self.snap(marker["time"] * ratio)
        # Never snapped: a claim follows its key, wherever the clock put it.
        self.edit_ledger.retime(ratio)
        self.scene_fps = new_fps
        self.mark_dirty()
        self._notify(BatchComplete())

    def mark_dirty(self) -> None:
        """Flag the store as needing a save.

        Inside a :meth:`batch_update` block the flush is deferred to the block
        exit.  Otherwise the flush is scheduled via :meth:`_schedule_flush`
        (immediate in the pure core; DCC subclasses may coalesce).
        """
        self._dirty = True
        if self._batch_depth > 0:
            return
        self._schedule_flush()

    def _flush_dirty(self) -> None:
        """Write to the persistence backend if the dirty flag is set."""
        if not self._dirty:
            return
        self.save()

    def save(self) -> None:
        """Persist via the configured backend (no-op if none set).

        Since saving shots means the user is authoring them, also installs the
        before-export preparer so any DCC export ships the current export view
        (see :meth:`enable_auto_export`).

        The cross-scene prefs file is NOT written here: a save follows every
        edit from every caller (a build, a sequencer drag, a headless script, a
        test), and writing it here copied whatever store saved last over the
        user's own preferences -- every test run reset them to defaults. The
        panel that changes a preference saves it (:meth:`_save_user_prefs`).
        """
        if self._persistence is not None:
            self._persistence.save(self.to_dict())
        # Clear only after a successful write — clearing first would
        # silently discard the pending changes if the backend raises.
        self._dirty = False
        if self.auto_publish_export:
            try:
                self.publish_export_view()
            except Exception:  # never let export projection break a save
                pass
        if self.shots:
            type(self)._register_export_preparer()

    # ---- detection convenience -------------------------------------------

    @property
    def is_detection_relevant(self) -> bool:
        """True when detection settings are actionable.

        Returns False when shots already exist in the store (detection
        settings would have no effect — shots are already defined).
        """
        return not bool(self.shots)

    def _overlaps_existing(self, candidate: Dict[str, Any]) -> bool:
        """True if *candidate* overlaps any existing shot's range."""
        c_start = candidate["start"]
        c_end = candidate["end"]
        for shot in self.shots:
            if c_start < shot.end and c_end > shot.start:
                return True
        return False

    def detect_and_define(self, overwrite: bool = False) -> List[ShotBlock]:
        """Detect shot regions and define them in the store.

        Convenience method that calls :meth:`detect_regions` and
        :meth:`define_shot` for each candidate.  Wraps all mutations
        in :meth:`batch_update` for a single ``BatchComplete`` event.

        Parameters:
            overwrite: If False (default), candidates that overlap
                existing shots are skipped.

        Returns:
            List of newly created :class:`ShotBlock` instances.  Empty in the
            pure core (``detect_regions`` returns ``[]``) until a DCC subclass
            supplies scene acquisition.
        """
        candidates = self.detect_regions()
        created: List[ShotBlock] = []
        with self.batch_update():
            for cand in candidates:
                if not overwrite and self._overlaps_existing(cand):
                    continue
                shot = self.define_shot(
                    name=self.default_name(cand["name"]),
                    start=cand["start"],
                    end=cand["end"],
                    objects=cand.get("objects", []),
                )
                created.append(shot)
        return created

    @staticmethod
    def leaf_name(node) -> str:
        """Leaf name with namespace preserved: ``"|grp|ns:obj"`` -> ``"ns:obj"``.

        Pure, DCC-agnostic ``"|"``-split — the only naming primitive the shot model
        needs.  Mirrors ``mayatk.core_utils.leaf_name`` so classification behaves
        identically whether names are short (Blender / pure) or long DAG paths (Maya).
        """
        return str(node).split("|")[-1]

    @staticmethod
    def member_key(node) -> str:
        """The identity a doc name and a shot member are compared by: the leaf
        with its namespace dropped (``"|set|AC:door_geo"`` -> ``"door_geo"``).

        The one rule every consumer uses -- the manifest's planner and Assess,
        the sequencer's classification -- so a doc that writes ``door_geo``
        matches the member a build stored as a long, referenced path.
        """
        return ShotStore.leaf_name(node).rsplit(":", 1)[-1]

    def member_nodes(self, name: str) -> List[str]:
        """Every scene node answering to a doc object *name* (scene hook).

        Several is a duplicate name the scene must fix, none a missing object.
        Pure default: the name itself; the DCC stores look it up, namespaces
        included.
        """
        return [str(name)]

    def resolve_member(self, name: str) -> Tuple[str, str]:
        """Resolve a doc object *name* to one scene node.

        Returns ``(node, reason)`` -- *reason* ``"found"`` (*node* is the
        scene's name for it), ``"missing"`` or ``"ambiguous"`` (*node* is
        *name* unchanged): several nodes answering to it (:meth:`member_nodes`)
        is a finding, never a silent pick.
        """
        hits = self.member_nodes(name)
        if len(hits) == 1:
            return hits[0], "found"
        return str(name), ("ambiguous" if hits else "missing")

    @staticmethod
    def resolve_clip_specs(
        shots: List["ShotBlock"], strategy: str = "name"
    ) -> List[Tuple[str, int, int]]:
        """Resolve ``[(clip_name, start, end), …]`` — the single source of truth for
        clip naming.  A legal, unique name passes VERBATIM (every name the store
        accepts is one); a legacy one is respelled and de-duplicated -- ignoring
        case, as Unity joins clips -- *in the given order*, so callers pass
        ``sorted_shots()`` for deterministic, stable results.
        """
        fn = CLIP_NAME_STRATEGIES.get(strategy, CLIP_NAME_STRATEGIES["name"])
        seen: set = set()
        specs: List[Tuple[str, int, int]] = []
        for i, shot in enumerate(shots):
            clip = _ShotStoreInternal._sanitize_clip_name(fn(i, shot))
            if clip.casefold() in seen:
                n = 1
                while f"{clip}_{n}".casefold() in seen:
                    n += 1
                clip = f"{clip}_{n}"
            seen.add(clip.casefold())
            specs.append((clip, int(round(shot.start)), int(round(shot.end))))
        return specs

    @classmethod
    def declared_range(cls, strategy: str = "name") -> Optional[Tuple[int, int]]:
        """The ``(start, end)`` frames the active store's shots span, or None.

        Read through :meth:`resolve_clip_specs` rather than off ``ShotBlock``
        directly, so this range and the ``fbx_takes`` an export publishes are
        rounded by the same code and cannot disagree about a fractional
        boundary.

        The authority is the STORE, not the published ``data_export`` carrier:
        an exporter asking "what do the shots span?" is computing a number, and
        must not stamp a metadata node to find out (the carrier is a
        projection, and publishing it is a scene mutation the user may have
        deliberately switched off).  Stale shots (:meth:`stale_shots`) are no
        part of it, as they are no part of what :meth:`export_records`
        declares: the range covers the takes the export ships.

        Returns:
            The union of every shot the export declares, or None when no store
            is active or it declares none -- so a caller can fall back rather
            than treat an empty scene as the range ``(0, 0)``.
        """
        store = cls.active()
        if store is None:
            return None
        shots = store.sorted_shots()
        stale, _present = store._scene_scope()
        specs = [
            spec
            for spec, shot in zip(cls.resolve_clip_specs(shots, strategy), shots)
            if shot.shot_id not in stale
        ]
        if not specs:
            return None
        return min(s for _, s, _ in specs), max(e for _, _, e in specs)
