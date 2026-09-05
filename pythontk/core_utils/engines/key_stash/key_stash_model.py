# !/usr/bin/python
# coding=utf-8
"""Key stash — park keyframes outside the working animation, retrieve later.

The pure, DCC-agnostic core.  A :class:`StashedClip` records WHERE a set of
keys came from and WHEN (per-curve key times, in scene frames); the DCC
adapter owns the payload that says where the keys physically live now (a
locked, unconnected animCurve in Maya; an orphan Action in Blender).  The
engine never interprets that payload — it only requires each curve record to
carry a ``"times"`` list so ranges, envelopes and frame-rate rescales are
computable without a scene.

Design constraints the adapters honour:

* A stashed clip must have **zero effect** on evaluation and export until it
  is retrieved.  That is a property of the adapter's storage mechanism, not
  of this model — but the model is where the record survives across
  sessions, so it is persisted through the same :class:`ScenePersistence`
  protocol the shots engine uses, on a channel of its own.
* **Copy before cut.**  ``stash`` persists the manifest BEFORE removing the
  keys from the live curves, so a crash between the two leaves duplicate
  keys, never lost ones.
* The **preview** (a transient override that lets the user scrub the stored
  range without retrieving it) is recorded here too, so a scene reopened
  mid-preview can be cleaned up rather than left silently overridden.
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

from pythontk.core_utils.engines.shots.shot_model import ScenePersistence

_log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_DEFAULT_FPS = 24.0


# ---------------------------------------------------------------------------
# Records / events
# ---------------------------------------------------------------------------


@dataclass
class StashedClip:
    """One parked set of keys.

    Parameters:
        clip_id: Store-unique integer id.
        label: User-facing name (defaults to the object and range at stash time).
        objects: Scene objects the keys came from (long names where the DCC has them).
        curves: One record per source curve.  Adapter-owned, except that every
            record carries ``"times"`` — the stashed key times in scene frames.
        created: ISO-8601 timestamp of the stash.
        source_shot_id: The shot the keys were stashed from, when the stash was
            driven from the shots system (lets the shot menu list its own clips).
        metadata: Free-form extension point (mirrors ``ShotBlock.metadata``).
    """

    clip_id: int
    label: str
    objects: List[str]
    curves: List[Dict[str, Any]]
    created: str = ""
    source_shot_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- derived ---------------------------------------------------------

    @property
    def times(self) -> List[float]:
        """Every stashed key time across all curves, sorted, de-duplicated."""
        seen = set()
        for rec in self.curves:
            seen.update(float(t) for t in rec.get("times", ()))
        return sorted(seen)

    @property
    def start(self) -> Optional[float]:
        """Earliest stashed key time, or ``None`` for an empty clip."""
        t = self.times
        return t[0] if t else None

    @property
    def end(self) -> Optional[float]:
        """Latest stashed key time, or ``None`` for an empty clip."""
        t = self.times
        return t[-1] if t else None

    @property
    def duration(self) -> float:
        """``end - start`` in frames (``0.0`` for a single-frame or empty clip)."""
        t = self.times
        return (t[-1] - t[0]) if len(t) > 1 else 0.0

    @property
    def key_count(self) -> int:
        """Total stashed keys over every curve."""
        return sum(len(rec.get("times", ())) for rec in self.curves)

    def rescale(self, ratio: float) -> None:
        """Multiply every recorded key time by *ratio* (frame-rate change)."""
        for rec in self.curves:
            if "times" in rec:
                rec["times"] = [float(t) * ratio for t in rec["times"]]

    # ---- serialization ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "label": self.label,
            "objects": list(self.objects),
            # Deep: a record's "times" list (and any nested adapter ref) must
            # not be shared between the clip and its serialized form.
            "curves": copy.deepcopy(self.curves),
            "created": self.created,
            "source_shot_id": self.source_shot_id,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StashedClip":
        return cls(
            clip_id=int(data["clip_id"]),
            label=str(data.get("label", "")),
            objects=list(data.get("objects", [])),
            curves=copy.deepcopy(list(data.get("curves", []))),
            created=str(data.get("created", "")),
            source_shot_id=data.get("source_shot_id"),
            metadata=copy.deepcopy(dict(data.get("metadata", {}))),
        )


@dataclass
class StashChanged:
    """Fired on every store mutation.

    ``kind`` is one of ``"stashed"``, ``"retrieved"``, ``"dropped"``,
    ``"preview"`` (preview started or ended) or ``"reloaded"`` (the store
    changed wholesale — a batch, a frame-rate rescale, a scene change).
    """

    kind: str
    clip_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class _KeyStashInternal(object):
    """Internal helpers for :class:`KeyStash`."""

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _fmt_frame(value: float) -> str:
        value = float(value)
        return f"{int(value)}" if value.is_integer() else f"{value:g}"

    @classmethod
    def _default_label(cls, objects: List[str], times: List[float]) -> str:
        """``"<leaf> 10-30"`` — the first object's leaf name and the frame span."""
        leaf = ""
        if objects:
            leaf = str(objects[0]).replace("|", "/").rstrip("/").rsplit("/", 1)[-1]
            leaf = leaf.rsplit(":", 1)[-1]
            if len(objects) > 1:
                leaf += f" +{len(objects) - 1}"
        if not times:
            return leaf or "clip"
        lo, hi = times[0], times[-1]
        span = (
            cls._fmt_frame(lo)
            if lo == hi
            else f"{cls._fmt_frame(lo)}-{cls._fmt_frame(hi)}"
        )
        return f"{leaf} {span}".strip()


class KeyStash(_KeyStashInternal):
    """Store of parked key clips with pluggable persistence.

    Pure core: CRUD, observer, serialization, the preview record and the
    class-level active-store singleton.  Scene-reaching behaviour is left to
    the DCC subclasses (``stash`` / ``retrieve`` / ``drop`` / ``preview`` /
    ``end_preview``), each of which calls back into these primitives.

    Parameters:
        clips: Initial clip list.  Copied on construction.
        scene_fps: Frame rate the recorded times are expressed in.  ``None``
            asks :meth:`_scene_fps` (the DCC hook; ``24.0`` in the pure core).
    """

    _active: ClassVar[Optional["KeyStash"]] = None
    _persistence: ClassVar[Optional[ScenePersistence]] = None
    _invalidation_listeners: ClassVar[List[Callable[[StashChanged], None]]] = []

    def __init__(
        self,
        clips: Optional[List[StashedClip]] = None,
        scene_fps: Optional[float] = None,
    ):
        self.clips: List[StashedClip] = list(clips or [])
        self.scene_fps: float = float(scene_fps) if scene_fps else self._scene_fps()
        #: The active preview, or ``None``: ``{"clip_id": int, **adapter payload}``.
        self.active_preview: Optional[Dict[str, Any]] = None
        self._listeners: List[Callable[[StashChanged], None]] = []
        self._dirty = False
        self._batch_depth = 0
        self._batch_pending = False
        self._next_id = max((c.clip_id for c in self.clips), default=0) + 1

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"<{type(self).__name__} clips={len(self.clips)} "
            f"preview={bool(self.active_preview)}>"
        )

    # ---- CRUD ------------------------------------------------------------

    def add_clip(
        self,
        objects: List[str],
        curves: List[Dict[str, Any]],
        label: Optional[str] = None,
        source_shot_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StashedClip:
        """Record a new clip and return it.

        Parameters:
            objects: Source objects.
            curves: Per-curve adapter records; each must carry ``"times"``.
            label: Display name; defaults to ``"<object> <start>-<end>"``.
            source_shot_id: Owning shot when stashed from the shots system.
            metadata: Free-form extras.

        Raises:
            ValueError: When no curve record carries any key time.
        """
        clip = StashedClip(
            clip_id=self._next_id,
            label=label or "",
            objects=[str(o) for o in objects],
            curves=copy.deepcopy(list(curves)),
            created=self._now(),
            source_shot_id=source_shot_id,
            metadata=dict(metadata or {}),
        )
        if clip.key_count == 0:
            raise ValueError("a stashed clip must hold at least one key")
        if not clip.label:
            clip.label = self._default_label(clip.objects, clip.times)
        self._next_id += 1
        self.clips.append(clip)
        self.mark_dirty()
        self._notify(StashChanged("stashed", clip.clip_id))
        return clip

    def get_clip(self, clip_id: int) -> Optional[StashedClip]:
        """The clip with *clip_id*, or ``None``."""
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        return None

    def remove_clip(self, clip_id: int, kind: str = "dropped") -> Optional[StashedClip]:
        """Forget *clip_id* and return the removed record (``None`` if absent).

        *kind* names the reason in the fired event: ``"dropped"`` (discarded)
        or ``"retrieved"`` (keys put back).  A clip under preview loses its
        preview record too (the adapter tears the scene side down first).
        """
        clip = self.get_clip(clip_id)
        if clip is None:
            return None
        self.clips.remove(clip)
        if self.active_preview and self.active_preview.get("clip_id") == clip_id:
            self.active_preview = None
        self.mark_dirty()
        self._notify(StashChanged(kind, clip_id))
        return clip

    def clips_for_object(self, name: str) -> List[StashedClip]:
        """Clips whose ``objects`` include *name* (exact or leaf-name match)."""
        leaf = str(name).rsplit("|", 1)[-1]
        out = []
        for clip in self.clips:
            for obj in clip.objects:
                if obj == name or obj.rsplit("|", 1)[-1] == leaf:
                    out.append(clip)
                    break
        return out

    def clips_for_shot(self, shot_id: int) -> List[StashedClip]:
        """Clips stashed from shot *shot_id*."""
        return [c for c in self.clips if c.source_shot_id == shot_id]

    def is_empty(self) -> bool:
        """``True`` when the store holds no clips and no preview."""
        return not self.clips and not self.active_preview

    # ---- preview record --------------------------------------------------

    def set_preview(
        self, clip_id: int, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record that *clip_id* is being previewed (adapter *payload* rides along).

        Raises:
            KeyError: When *clip_id* is not a stashed clip.
        """
        if self.get_clip(clip_id) is None:
            raise KeyError(f"no stashed clip {clip_id}")
        self.active_preview = {"clip_id": clip_id, **(payload or {})}
        self.mark_dirty()
        self._notify(StashChanged("preview", clip_id))

    def clear_preview(self) -> Optional[Dict[str, Any]]:
        """Forget the preview record; return what it held (``None`` if none)."""
        prev, self.active_preview = self.active_preview, None
        if prev is not None:
            self.mark_dirty()
            self._notify(StashChanged("preview", prev.get("clip_id")))
        return prev

    # ---- observer --------------------------------------------------------

    def add_listener(self, callback: Callable[[StashChanged], None]) -> None:
        """Register *callback* for :class:`StashChanged` events."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[StashChanged], None]) -> None:
        """Remove a previously registered listener."""
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass

    def _notify(self, event: StashChanged) -> None:
        if self._batch_depth > 0:
            self._batch_pending = True
            return
        for cb in list(self._listeners):
            try:
                cb(event)
            except Exception:
                _log.warning("key stash listener failed", exc_info=True)

    @contextmanager
    def batch_update(self):
        """Defer notifications and the flush until the block exits."""
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                if self._batch_pending:
                    self._batch_pending = False
                    self._notify(StashChanged("reloaded"))
                self._flush_dirty()

    # ---- frame rate ------------------------------------------------------

    def rescale_to_fps(self, new_fps: float) -> None:
        """Rescale every recorded key time from :attr:`scene_fps` to *new_fps*.

        The pure default assumes a frame-rate change MOVES keys in frame
        space (Maya: curve keys live in ticks, so the frame numbers change).
        An adapter whose keys stay put in frame space (Blender) overrides this
        to update :attr:`scene_fps` only.
        """
        old_fps = self.scene_fps
        if not old_fps or abs(float(new_fps) - old_fps) < 0.01:
            return
        ratio = float(new_fps) / old_fps
        for clip in self.clips:
            clip.rescale(ratio)
        self.scene_fps = float(new_fps)
        self.mark_dirty()
        self._notify(StashChanged("reloaded"))

    def _scene_fps(self) -> float:
        """DCC hook: the scene frame rate. Pure default ``24.0``."""
        return _DEFAULT_FPS

    # ---- persistence -----------------------------------------------------

    def mark_dirty(self) -> None:
        """Flag the store for saving; the flush is deferred inside :meth:`batch_update`."""
        self._dirty = True
        if self._batch_depth > 0:
            return
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        """DCC hook: coalesce writes. Pure default flushes immediately."""
        self._flush_dirty()

    def _flush_dirty(self) -> None:
        if self._dirty:
            self.save()

    def save(self) -> None:
        """Persist through the configured backend (no-op without one)."""
        backend = type(self)._persistence
        if backend is not None:
            backend.save(self.to_dict())
        # Clear only after a successful write -- clearing first would silently
        # discard the pending changes if the backend raises, and the next
        # _flush_dirty() would no-op. Same rule, same reason, as
        # ShotStore.save. It matters more here: the Maya adapter flushes under
        # evalDeferred, so a raise is printed and swallowed by the host.
        self._dirty = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "scene_fps": self.scene_fps,
            "clips": [c.to_dict() for c in self.clips],
            "preview": dict(self.active_preview) if self.active_preview else None,
            # The id high-water mark. Without it a reload recomputes
            # max(clip_id) + 1, which walks BACKWARDS once the highest clip
            # has been removed and hands a retired id to a new clip -- so a
            # reference captured before the reload (a tree item's UserRole, a
            # queued StashChanged, the restored active_preview) silently
            # addresses different keys. Optional on read, so documents written
            # before this key still load.
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyStash":
        store = cls(
            clips=[StashedClip.from_dict(c) for c in data.get("clips", [])],
            scene_fps=data.get("scene_fps"),
        )
        preview = data.get("preview")
        store.active_preview = dict(preview) if preview else None
        # max(), never assignment: a document written before "next_id" existed
        # falls back to the computed mark, and a corrupt low value can never
        # drag the counter below the ids already in the document.
        try:
            store._next_id = max(store._next_id, int(data.get("next_id", 0)))
        except (TypeError, ValueError):
            pass
        return store

    # ---- singleton -------------------------------------------------------

    @classmethod
    def set_persistence(cls, backend: Optional[ScenePersistence]) -> None:
        """Set the backend :meth:`active` loads from and :meth:`save` writes to."""
        cls._persistence = backend

    @classmethod
    def active(cls) -> "KeyStash":
        """The active store, loaded from the backend on first access.

        Reconciles frame rate the way the shot store does: a store saved at a
        different frame rate is rescaled to the current one on load.
        """
        if cls._active is None:
            backend = cls._persistence
            data = backend.load() if backend is not None else None
            if data:
                store = cls.from_dict(data)
                current = store._scene_fps()
                if store.clips and abs(store.scene_fps - current) > 0.01:
                    store.rescale_to_fps(current)
            else:
                store = cls()
            cls._active = store
            store._on_activated()
        return cls._active

    def _on_activated(self) -> None:
        """DCC hook: run once when this instance becomes the active store.

        Adapters reconcile the record against the scene here — prune clips
        whose storage nodes are gone, tear down a preview left over from a
        save made mid-preview.  Pure default does nothing.
        """

    @classmethod
    def invalidate(cls) -> None:
        """Drop the active store (scene changed); tell the invalidation listeners."""
        cls._active = None
        for cb in list(cls._invalidation_listeners):
            try:
                cb(StashChanged("reloaded"))
            except Exception:
                _log.warning("key stash invalidation listener failed", exc_info=True)

    @classmethod
    def add_invalidation_listener(
        cls, callback: Callable[[StashChanged], None]
    ) -> None:
        """Register a callback that survives store instances (UI rebinding)."""
        if callback not in cls._invalidation_listeners:
            cls._invalidation_listeners.append(callback)

    @classmethod
    def remove_invalidation_listener(
        cls, callback: Callable[[StashChanged], None]
    ) -> None:
        """Remove a previously registered invalidation listener."""
        try:
            cls._invalidation_listeners.remove(callback)
        except ValueError:
            pass

    # ---- helpers shared by adapters --------------------------------------

    @staticmethod
    def offset_for(clip: StashedClip, at: Optional[float]) -> float:
        """Frame offset that lands *clip*'s first key on *at* (``0.0`` for ``None``)."""
        if at is None or clip.start is None:
            return 0.0
        return float(at) - clip.start

    @staticmethod
    def gate_range(clip: StashedClip) -> Optional[Tuple[float, float]]:
        """``(start, end)`` of *clip*, or ``None`` when it holds no keys."""
        if clip.start is None or clip.end is None:
            return None
        return (clip.start, clip.end)
