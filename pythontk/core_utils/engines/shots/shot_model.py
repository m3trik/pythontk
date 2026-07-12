# coding=utf-8
"""DCC-agnostic shot data model and persistent store.

Provides :class:`ShotBlock` (the fundamental shot data structure) and
:class:`ShotStore` (CRUD + observer + pluggable persistence).  This is the
shared foundation both mayatk and blendertk consume: the model is engine-free,
and every place that would otherwise reach into a DCC scene is exposed as an
**overridable hook with a pure default** (see :meth:`ShotStore._scene_fps`,
:meth:`ShotStore.has_animation`, :meth:`ShotStore.detect_regions`,
:meth:`ShotStore.assess`, :meth:`ShotStore.publish_export_view`, and
:meth:`ShotStore._resolve_long_names`).  DCC toolkits subclass ``ShotStore``
and override those hooks; the pure core never imports ``maya`` or ``bpy``.

Persistence is pluggable: call :meth:`ShotStore.set_persistence` with a backend
that implements ``save(data)`` / ``load() -> dict | None``.  Absent a backend
the store runs in pure in-memory mode.
"""
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)
from contextlib import contextmanager


_log = logging.getLogger(__name__)

_DEFAULT_FPS = 24.0


def leaf_name(node) -> str:
    """Leaf name with namespace preserved: ``"|grp|ns:obj"`` -> ``"ns:obj"``.

    Pure, DCC-agnostic ``"|"``-split — the only naming primitive the shot model
    needs.  Mirrors ``mayatk.core_utils.leaf_name`` so classification behaves
    identically whether names are short (Blender / pure) or long DAG paths (Maya).
    """
    return str(node).split("|")[-1]


__all__ = [
    "SHOT_PALETTE",
    "ShotBlock",
    "ShotStore",
    "StoreEvent",
    "ShotDefined",
    "ShotUpdated",
    "ShotRemoved",
    "ActiveShotChanged",
    "SettingsChanged",
    "BatchComplete",
    "StoreInvalidated",
    "ScenePersistence",
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

try:
    from pythontk import Palette

    SHOT_PALETTE = Palette.status().alias(
        {
            "csv_object": "valid",  # expected — no color
            "scene_discovered": "info",  # found in scene, not in CSV
            "missing_object": "error",  # referenced but missing
            "missing_behavior": "warn",  # expected behaviour keys absent
            "user_animated": "info",  # custom user animation detected
            "additional": "warn",  # unexpected scene objects
            "collision": "error",  # timing overlap
            "missing_shot": "info",  # shot not yet built
        }
    )
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
        # Metadata is keyed by CSV (short) names while shot.objects hold
        # long DAG paths after a manifest sync — fall back to leaf-name
        # comparison so every object doesn't degrade to
        # "scene_discovered" on the first re-sync.
        status_by_leaf = {leaf_name(k): v for k, v in statuses.items()}
        csv_leaves = {leaf_name(n) for n in csv_objs}
        result: Dict[str, str] = {}
        for obj in self.objects:
            leaf = leaf_name(obj)
            if obj in statuses:
                result[obj] = statuses[obj]
            elif leaf in status_by_leaf:
                result[obj] = status_by_leaf[leaf]
            elif csv_objs and obj not in csv_objs and leaf not in csv_leaves:
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


def _sanitize_clip_name(name: str) -> str:
    """Return a Unity/FBX-legal clip name (alphanumeric + ``_``, case preserved)."""
    import pythontk as ptk

    clean = ptk.StrUtils.sanitize(name or "", preserve_case=True)
    return clean or "shot"


def resolve_clip_specs(
    shots: List["ShotBlock"], strategy: str = "name"
) -> List[Tuple[str, int, int]]:
    """Resolve ``[(clip_name, start, end), …]`` — the single source of truth for
    clip naming.  Names are sanitized and made unique *in the given order*, so
    callers pass ``sorted_shots()`` for deterministic, stable results.
    """
    fn = CLIP_NAME_STRATEGIES.get(strategy, CLIP_NAME_STRATEGIES["name"])
    seen: set = set()
    specs: List[Tuple[str, int, int]] = []
    for i, shot in enumerate(shots):
        clip = _sanitize_clip_name(fn(i, shot))
        if clip in seen:
            n = 1
            while f"{clip}_{n}" in seen:
                n += 1
            clip = f"{clip}_{n}"
        seen.add(clip)
        specs.append((clip, int(round(shot.start)), int(round(shot.end))))
    return specs


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
    """The active (selected) shot changed."""

    name: ClassVar[str] = "active_shot_changed"
    shot_id: Optional[int] = None


@dataclass(frozen=True)
class SettingsChanged(StoreEvent):
    """Detection-relevant settings were modified."""

    name: ClassVar[str] = "settings_changed"


@dataclass(frozen=True)
class BatchComplete(StoreEvent):
    """A :meth:`ShotStore.batch_update` context has exited."""

    name: ClassVar[str] = "batch_complete"


@dataclass(frozen=True)
class StoreInvalidated(StoreEvent):
    """The active store was discarded (scene change / new scene).

    Listeners should rebind to the new :meth:`ShotStore.active` instance.
    Fired on class-level invalidation listeners registered via
    :meth:`ShotStore.add_invalidation_listener`.
    """

    name: ClassVar[str] = "store_invalidated"


# ---------------------------------------------------------------------------
# ShotStore
# ---------------------------------------------------------------------------


class ShotStore:
    """Central store for shot data with pluggable persistence.

    The pure, DCC-agnostic core.  Scene-reaching behaviour lives in the
    overridable hooks (:meth:`_scene_fps`, :meth:`has_animation`,
    :meth:`detect_regions`, :meth:`assess`, :meth:`publish_export_view`,
    :meth:`_resolve_long_names`, :meth:`_schedule_flush`), each with a pure
    default; mayatk / blendertk subclass this and override them.

    Parameters:
        shots: Initial shot list.  Copied on construction.
    """

    _active: Optional["ShotStore"] = None
    _persistence: Optional[ScenePersistence] = None
    _invalidation_listeners: ClassVar[List[Callable[["StoreInvalidated"], None]]] = []
    # Directory for the cross-scene prefs JSON. ``None`` → pythontk's user-config
    # root (see :meth:`_prefs_path`); tests point it at a temp dir so they never
    # touch the user's real config. A DCC adapter that wants its own store (e.g.
    # QSettings) overrides :meth:`_restore_user_prefs` / :meth:`_save_user_prefs`.
    _prefs_dir_override: ClassVar[Optional[str]] = None
    DETECTION_MODES = ("auto", "all", "skip_zero", "zero_as_end")
    FIT_MODES = ("extend_only", "fit_contents")
    DEFAULT_INITIAL_SHOT_LENGTH: float = 200.0
    DEFAULT_FIT_MODE: str = "extend_only"
    DEFAULT_SNAP_WHOLE_FRAMES: bool = True

    def __init__(
        self,
        shots: Optional[List[ShotBlock]] = None,
    ):
        self.shots: List[ShotBlock] = list(shots) if shots else []
        self.hidden_objects: set = set()
        self.pinned_objects: set = set()
        self.markers: List[Dict[str, Any]] = []
        self.gap: float = 0.0
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
        self.locked_objects: set = set()  # object names locked in the sequencer
        self.scene_fps: float = self._scene_fps()
        # Source CSV path (when the store was populated from a manifest CSV).
        # Purely informational — lets the user retrace provenance on reopen.
        self.source_csv: str = ""
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

    # ---- active shot (session state, not persisted) ----------------------

    @property
    def active_shot_id(self) -> Optional[int]:
        """The currently selected shot, or ``None``."""
        return self._active_shot_id

    def set_active_shot(self, shot_id: Optional[int]) -> None:
        """Set the active shot and notify listeners."""
        if shot_id == self._active_shot_id:
            return
        self._active_shot_id = shot_id
        self._notify(ActiveShotChanged(shot_id=shot_id))

    # ---- observer --------------------------------------------------------

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
        :class:`SettingsChanged`, :class:`BatchComplete`.
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
                    _evt = BatchComplete()
                    for cb in self._listeners:
                        try:
                            cb(_evt)
                        except Exception:
                            _log.warning(
                                "shot store listener failed", exc_info=True
                            )
                # Synchronous flush — batch = single atomic write.
                # Runs even with no accumulated events: mark_dirty
                # inside a batch defers its flush to here, and mutators
                # that don't notify (pin/hide/gap-lock) would otherwise
                # leave the dirty store unscheduled for save.
                self._flush_dirty()

    # ---- gap locking -----------------------------------------------------

    def is_gap_locked(self, left_id: str, right_id: str) -> bool:
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
            self.locked_gaps.add(
                (sorted_shots[i].shot_id, sorted_shots[i + 1].shot_id)
            )
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
            if persistence is not None:
                data = persistence.load()
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
            else:
                cls._active = cls()
                cls._active._restore_user_prefs()
        return cls._active

    @classmethod
    def set_active(cls, store: "ShotStore") -> None:
        """Replace the active store instance."""
        cls._active = store

    @classmethod
    def clear_active(cls) -> None:
        """Reset the active store and persistence backend."""
        cls._active = None
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
    def _notify_invalidated(cls) -> None:
        """Fire all invalidation listeners."""
        event = StoreInvalidated()
        for cb in cls._invalidation_listeners:
            try:
                cb(event)
            except Exception:
                _log.warning(
                    "shot store invalidation listener failed", exc_info=True
                )

    # ---- cross-scene user preferences ------------------------------------

    @classmethod
    def _prefs_path(cls):
        """Cross-scene prefs JSON location — pythontk's user-config root by default
        (zero-dep, no Qt/QSettings), or ``_prefs_dir_override`` when set (tests)."""
        from pathlib import Path

        if cls._prefs_dir_override is not None:
            return Path(cls._prefs_dir_override) / "shots_prefs.json"
        from pythontk.core_utils.user_config import user_config_root

        return user_config_root() / "shots" / "prefs.json"

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
        """Persist detection preferences to the cross-scene prefs file (zero-dep JSON)."""
        import json

        try:
            path = self._prefs_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "detection_mode": self.detection_mode,
                        "select_on_load": self.select_on_load,
                        "detection_threshold": self.detection_threshold,
                        "fit_mode": self.fit_mode,
                        "initial_shot_length": self.initial_shot_length,
                        "snap_whole_frames": self.snap_whole_frames,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ---- frame snapping --------------------------------------------------

    def snap(self, frame: float) -> float:
        """Return *frame* rounded to the nearest integer when snapping is on.

        Single chokepoint for the ``snap_whole_frames`` policy.  Call at
        any site that writes a frame value to a shot, keyframe, or
        timeline range to guarantee the in-memory model stays valid.
        """
        if self.snap_whole_frames:
            return float(round(frame))
        return float(frame)

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
            name: Human-readable label.
            start: First frame.
            end: Last frame.
            objects: Transform node names.  ``None`` → empty list.
            metadata: Arbitrary key/value pairs.
            locked: Mark this shot as user-finalized.

        Returns:
            The newly created :class:`ShotBlock`.
        """
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
        """Update fields on an existing shot.  Returns the shot, or ``None``."""
        shot = self.shot_by_id(shot_id)
        if shot is None:
            return None
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
        self._notify(ShotUpdated(shot=shot))
        self.mark_dirty()
        return shot

    def remove_shot(self, shot_id: int) -> bool:
        """Remove a shot by ID.  Returns ``True`` if found."""
        for i, s in enumerate(self.shots):
            if s.shot_id == shot_id:
                self.shots.pop(i)
                self._notify(ShotRemoved(shot_id=shot_id))
                self.mark_dirty()
                return True
        return False

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
            name: Human-readable label.
            duration: Shot duration in frames.
            gap: Gap frames after the previous shot.
            start_frame: Explicit start frame.  If ``None``, computed
                from the last shot's end + *gap*.
            objects: Transform node names.
            metadata: Arbitrary key/value pairs.
            locked: Mark this shot as user-finalized.

        Returns:
            The newly created :class:`ShotBlock`.
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
            "auto_publish_export": self.auto_publish_export,
            "clip_name_strategy": self.clip_name_strategy,
        }

    def to_export_view(self, strategy: str = "name") -> Dict[str, Any]:
        """Build the FBX/Unity export view from the current shots.

        Returns ``{"fbx_takes": [...], "shot_metadata": {...}}``.  Both payloads
        derive from a single :func:`resolve_clip_specs` pass, so the FBX take
        name and the metadata ``clip`` join-key cannot drift.  Minimal overlap:
        ranges live only in ``fbx_takes``; ``shot_metadata`` carries the extras
        a clip can't (description, objects, section), keyed by clip.
        """
        sorted_s = self.sorted_shots()
        specs = resolve_clip_specs(sorted_s, strategy=strategy)
        fbx_takes = [{"name": c, "start": s, "end": e} for c, s, e in specs]
        shots_meta = [
            {
                "clip": clip,
                "description": shot.description or "",
                "objects": [leaf_name(o) for o in shot.objects],
                "section": (shot.metadata or {}).get("section", ""),
            }
            for (clip, _s, _e), shot in zip(specs, sorted_s)
        ]
        return {
            "fbx_takes": fbx_takes,
            "shot_metadata": {"version": 1, "shots": shots_meta},
        }

    @classmethod
    def refresh_export_view(cls) -> None:
        """Republish the active store's export view when it has shots.

        The canonical, no-arg pre-export refresh for the Shots system.  A no-op
        when no store is active or it has no shots (leaving no empty carrier
        behind), and — via the :meth:`publish_export_view` hook — a no-op in the
        pure core until a DCC subclass supplies a real projection.
        """
        store = cls.active()
        if store is not None and store.shots:
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
        store.gap = float(data.get("gap", 0.0))
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
        store.auto_publish_export = bool(data.get("auto_publish_export", False))
        strat = data.get("clip_name_strategy")
        if strat in CLIP_NAME_STRATEGIES:
            store.clip_name_strategy = str(strat)
        return store

    # ---- persistence convenience -----------------------------------------

    def rescale_to_fps(self, new_fps: float) -> None:
        """Scale all shot timings from the current ``scene_fps`` to *new_fps*.

        Called automatically when the scene framerate changes.  Updates
        ``scene_fps``, rescales shot boundaries, gap, and markers,
        then fires a :class:`BatchComplete` so the UI repaints.
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

        Also writes detection preferences to the cross-scene prefs file so they
        survive across scenes, and — since saving shots means the user is authoring
        them — installs the before-export preparer so any DCC export ships the
        current export view (see :meth:`enable_auto_export`).
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
        self._save_user_prefs()

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
                    name=cand["name"],
                    start=cand["start"],
                    end=cand["end"],
                    objects=cand.get("objects", []),
                )
                created.append(shot)
        return created
