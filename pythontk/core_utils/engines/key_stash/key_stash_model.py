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

The store is one class on every host: everything the hosts do alike (the
record's lifecycle, the frame-rate rule, the undo subscription, the crossing's
reload) is written once on :class:`KeyStash`, and what a host genuinely differs
by comes through :attr:`KeyStash.HOST`, a short fixed :class:`KeyStashHost`
(:class:`PureKeyStashHost` holds the pure defaults; a fake stands in for a
DCC in a plain-Python test).  mayatk's and blendertk's ``KeyStash`` bind their
host's adapter and add the scene operations (``stash`` / ``retrieve`` /
``drop`` / ``preview``), overriding nothing.

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
* **Undo.**  Each scene operation is ONE undo step whose record rides
  inside it (:meth:`KeyStash._undo_step`), and :meth:`KeyStash.active`
  re-reads a record an undo or redo moved under the loaded store.
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    ClassVar,
    ContextManager,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
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
# The host adapter
# ---------------------------------------------------------------------------


@runtime_checkable
class KeyStashHost(Protocol):
    """What the key stash asks of the host it runs in.

    Each DCC implements it (mayatk's ``MayaKeyStashHost``, blendertk's
    ``BlenderKeyStashHost``: each library's ``KeyStash.HOST``);
    :class:`PureKeyStashHost` holds the pure defaults, and a fake stands in for
    either in a plain-Python test.  Each member answers what the pure default
    does when the host's scene cannot be reached.
    """

    #: Whether a scene's keys keep real TIME when its frame rate changes --
    #: the stashed times rescale to follow them (Maya: curve keys live in
    #: ticks) -- or stay on their FRAMES, so the store records the new rate and
    #: moves nothing (Blender): :meth:`KeyStash.rescale_to_fps`.
    keys_keep_time: bool

    def persistence(self, store_cls: type) -> Optional[ScenePersistence]:
        """A new backend for *store_cls*'s record (its ``ATTR_NAME``;
        :meth:`KeyStash.active` installs it once), or ``None``."""

    def store_state(self) -> Any:
        """What the host keeps per store, made with each store
        (:attr:`KeyStash.host_state`; Blender: the deferred write's pending
        flag) -- the shot store's ``ShotsHost.store_state``."""

    def scene_fps(self) -> Optional[float]:
        """The scene's frame rate, or ``None`` with no scene clock."""

    def schedule_flush(self, store: "KeyStash") -> None:
        """Have *store*'s pending write made (``store._flush_dirty``): at once,
        or coalesced into one deferred write."""

    def undo_chunk(self, name: str) -> ContextManager[Any]:
        """A context manager making its block ONE step of the host's undo
        queue, labelled *name*."""

    def save_in_step(self, backend: ScenePersistence, data: Dict[str, Any]) -> bool:
        """Write the record *data* through *backend* so the open undo step
        records it; ``False`` when the backend's ordinary ``save`` already does
        (the store then saves as usual)."""

    def reconcile(self, store: "KeyStash") -> List[int]:
        """Bring *store*'s record in line with the scene -- prune the clips
        whose parked keys are gone, end a preview the record says runs
        (:meth:`KeyStash._reconcile_with`) -- and return the pruned ids."""

    def script_jobs(self) -> Any:
        """The host's event manager (``subscribe(event, callback, owner=)``,
        ``connect_cleanup(widget, owner=)``) :meth:`KeyStash.watch_undo`
        subscribes through, or ``None`` with no event system."""


class PureKeyStashHost:
    """The :class:`KeyStashHost` with no host behind it: the pure defaults.

    What a store runs on in plain Python, and the base a fake host overrides a
    few members of.  It never guesses: no clip is pruned on a lookup it cannot
    make.
    """

    keys_keep_time = True

    def persistence(self, store_cls: type) -> Optional[ScenePersistence]:
        return None

    def store_state(self) -> Any:
        return None

    def scene_fps(self) -> Optional[float]:
        return None

    def schedule_flush(self, store: "KeyStash") -> None:
        store._flush_dirty()

    def undo_chunk(self, name: str) -> ContextManager[Any]:
        # No undo queue: the block is the step.
        return nullcontext()

    def save_in_step(self, backend: ScenePersistence, data: Dict[str, Any]) -> bool:
        return False

    def reconcile(self, store: "KeyStash") -> List[int]:
        return []

    def script_jobs(self) -> Any:
        return None


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

    One class on every host: CRUD, observer, serialization, the preview
    record, the class-level active-store singleton, the frame-rate rule, the
    undo subscription and the crossing's reload are written here once, and
    every place that would reach into a scene is a call on :attr:`HOST` (a
    :class:`KeyStashHost`).  A host library binds its adapter on a class of its
    own (``mtk.KeyStash``, ``btk.KeyStash``: configuration) and adds the scene
    operations (``stash`` / ``retrieve`` / ``drop`` / ``preview`` /
    ``end_preview``), each of which calls back into these primitives and runs
    as ONE :meth:`_undo_step`.

    Parameters:
        clips: Initial clip list.  Copied on construction.
        scene_fps: Frame rate the recorded times are expressed in.  ``None``
            asks the host (:meth:`KeyStashHost.scene_fps`; ``24.0`` with none).
    """

    #: The host adapter every scene call goes through; a host library's
    #: binding sets its own (``MayaKeyStashHost``, ``BlenderKeyStashHost``).
    HOST: ClassVar[KeyStashHost] = PureKeyStashHost()
    #: The scene record the clip manifest rides (its declaration's key).
    ATTR_NAME: ClassVar[str] = SceneRecords.KEY_STASH.key
    #: The host events after which the record may have moved under the loaded
    #: store (:meth:`watch_undo`).
    UNDO_EVENTS: ClassVar[Tuple[str, ...]] = ("Undo", "Redo")

    _active: ClassVar[Optional["KeyStash"]] = None
    _persistence: ClassVar[Optional[ScenePersistence]] = None
    _invalidation_listeners: ClassVar[List[Callable[[StashChanged], None]]] = []

    def __init__(
        self,
        clips: Optional[List[StashedClip]] = None,
        scene_fps: Optional[float] = None,
    ):
        self.clips: List[StashedClip] = list(clips or [])
        #: What the host keeps for this store (:meth:`KeyStashHost.store_state`).
        self.host_state: Any = self.HOST.store_state()
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

    def is_previewing(self, clip_id: Optional[int] = None) -> bool:
        """Whether a preview is active (for *clip_id*, when given)."""
        if not self.active_preview:
            return False
        return clip_id is None or self.active_preview.get("clip_id") == clip_id

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
        """Bring the record from :attr:`scene_fps` to *new_fps*, the scene's
        new frame rate, so the stashed times stay with their keys.

        On a host whose keys keep real time (:attr:`KeyStashHost.keys_keep_time`:
        Maya, whose curve keys live in ticks, and the pure default) the frame
        numbers moved: every recorded key time rescales and the listeners are
        told.  On one whose keys stay on their frames (Blender) the new rate is
        recorded and nothing moves -- the shot store's rule
        (``ShotStore.rescale_to_fps``).
        """
        new_fps = float(new_fps)
        if not self.HOST.keys_keep_time:
            if abs(new_fps - self.scene_fps) < 0.01:
                return
            self.scene_fps = new_fps
            self.mark_dirty()
            return
        old_fps = self.scene_fps
        if not old_fps or abs(new_fps - old_fps) < 0.01:
            return
        ratio = new_fps / old_fps
        for clip in self.clips:
            clip.rescale(ratio)
        self.scene_fps = new_fps
        self.mark_dirty()
        self._notify(StashChanged("reloaded"))

    def _scene_fps(self) -> float:
        """The scene frame rate (:meth:`KeyStashHost.scene_fps`); ``24.0``
        with no scene clock."""
        fps = self.HOST.scene_fps()
        return float(fps) if fps else _DEFAULT_FPS

    # ---- persistence -----------------------------------------------------

    def mark_dirty(self) -> None:
        """Flag the store for saving; the flush is deferred inside :meth:`batch_update`."""
        self._dirty = True
        if self._batch_depth > 0:
            return
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        """Have the pending write made, at once or coalesced
        (:meth:`KeyStashHost.schedule_flush`; the pure default writes at once)."""
        self.HOST.schedule_flush(self)

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
    def merge_record(
        cls,
        own: Optional[Dict[str, Any]],
        other: Optional[Dict[str, Any]],
        ctx: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """The ``key_stash`` record's merge (``SceneRecords.CODECS``).

        Another scene's parked clips join this scene's under fresh ids; this
        scene's clips and preview stand.  Their times are rescaled to this
        scene's frame rate when the two differ, and a clip's source shot
        follows ``ctx.remaps["shot_id"]`` when the shot store merged first --
        a source shot that did not arrive is forgotten rather than pointed at
        an unrelated shot of this scene.  *other* is already respelled.
        """
        if not other or not other.get("clips"):
            return own
        theirs = cls.from_dict(other)
        mine = cls.from_dict(own) if own else cls(scene_fps=theirs.scene_fps)
        ratio = 1.0
        if mine.scene_fps and theirs.scene_fps:
            ratio = float(mine.scene_fps) / float(theirs.scene_fps)
        shot_map = None if ctx is None else ctx.remaps.get("shot_id")
        for clip in theirs.clips:
            clip = copy.deepcopy(clip)
            clip.clip_id = mine._next_id
            mine._next_id += 1
            if abs(ratio - 1.0) > 1e-9:
                clip.rescale(ratio)
            if shot_map is not None and clip.source_shot_id is not None:
                clip.source_shot_id = shot_map.get(int(clip.source_shot_id))
            mine.clips.append(clip)
        if ctx is not None and abs(ratio - 1.0) > 1e-9:
            ctx.note(
                f"Key Stash: {len(theirs.clips)} clip(s) rescaled from "
                f"{theirs.scene_fps:g} to {mine.scene_fps:g} fps."
            )
        return mine.to_dict()

    @classmethod
    def respell_record(cls, data: Optional[Dict[str, Any]], ctx: Any) -> Any:
        """*data* (a stash dict another scene saved) with the names it holds
        put through ``ctx`` -- the ``key_stash`` record's respelling
        (``RecordTransfer.respell``).

        A clip's ``objects`` and its curve records -- adapter-owned names and
        node references beside the numeric key times -- are respelled; its
        label, the preview and the counters are not (a label reading like a
        renamed node keeps its text).
        """
        if not data or not data.get("clips"):
            return data
        out = copy.deepcopy(data)
        for clip in out["clips"]:
            clip["objects"] = [ctx.spell(str(o)) for o in clip.get("objects") or []]
            clip["curves"] = ctx.respell(clip.get("curves") or [])
        return out

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

    # ---- undo ------------------------------------------------------------

    @contextmanager
    def _undo_step(self, name: str):
        """Run a scene operation as ONE undo step that carries its record.

        The store's mutations inside the block are batched, and the record is
        written inside the step on the way out (:meth:`_save_in_step`), so an
        undo or redo moves the record with the scene edit it describes and
        :meth:`active` re-reads it.  Nothing may flush mid-step: a host that
        runs deferred work at once (mayapy's ``evalDeferred``, background
        Blender) wrote the new record outside the step first, where no undo
        could take it back (measured 2026-09-15 in both adapters).

        Parameters:
            name: The undo step's label.
        """
        with self._undo_chunk(name), self.batch_update():
            try:
                yield
            finally:
                # Also when the block raises: the batch's exit flush would
                # write through save(), outside an adapter's undo queue.
                if self._dirty:
                    self._save_in_step()

    def _undo_chunk(self, name: str) -> ContextManager[Any]:
        """A context manager making its block ONE step of the host's undo
        queue (:meth:`KeyStashHost.undo_chunk`; the pure default has none)."""
        return self.HOST.undo_chunk(name)

    def _save_in_step(self) -> None:
        """Write the record so the open undo step records it.

        Through the host (:meth:`KeyStashHost.save_in_step`) when its backend
        writes outside the undo queue unless told otherwise (Maya: an undone
        Retrieve left its keys in a stash node the record no longer named,
        measured 2026-09-15); else :meth:`save`.
        """
        backend = type(self)._persistence
        if backend is not None and self.HOST.save_in_step(backend, self.to_dict()):
            self._dirty = False  # after the write, as save() clears it
            return
        self.save()

    def _reread_record(self) -> None:
        """Load the backend's record into THIS store and tell its listeners.

        In place rather than by replacing the active store: activation runs
        :meth:`reconcile`, where the host tears down a recorded preview --
        right for a scene just opened, wrong here, where it would end the very
        preview a redo restored.
        """
        fresh = type(self)._load_record()
        self.clips = fresh.clips
        self.active_preview = fresh.active_preview
        self.scene_fps = fresh.scene_fps
        self._next_id = fresh._next_id
        self._dirty = False
        self._notify(StashChanged("reloaded"))

    # ---- singleton -------------------------------------------------------

    @classmethod
    def set_persistence(cls, backend: Optional[ScenePersistence]) -> None:
        """Set the backend :meth:`active` loads from and :meth:`save` writes to."""
        cls._persistence = backend

    @classmethod
    def active(cls) -> "KeyStash":
        """The active store, loaded from the backend on first access.

        With no backend set, the host's (:meth:`KeyStashHost.persistence`) is
        installed once.  Reconciles frame rate the way the shot store does: a
        store saved at a different frame rate is rescaled to the current one
        on load.  A backend that can tell its record moved under the loaded
        store (an optional ``record_changed()``: an undo or redo of a scene
        operation) has it re-read in place (:meth:`_reread_record`).
        """
        if cls._active is None and cls._persistence is None:
            backend = cls.HOST.persistence(cls)
            if backend is not None:
                cls.set_persistence(backend)
        if cls._active is None:
            store = cls._load_record()
            if store.clips:
                current = store._scene_fps()
                if abs(store.scene_fps - current) > 0.01:
                    store.rescale_to_fps(current)
            cls._active = store
            store._on_activated()
        else:
            record_changed = getattr(cls._persistence, "record_changed", None)
            if record_changed is not None and record_changed():
                cls._active._reread_record()
        return cls._active

    @classmethod
    def _load_record(cls) -> "KeyStash":
        """A store holding the backend's record, or an empty one without it.

        Activation rescales what this returns; the in-place re-read does not,
        since a rescaled throwaway store would queue a flush of its own.
        """
        backend = cls._persistence
        data = backend.load() if backend is not None else None
        return cls.from_dict(data) if data else cls()

    def _on_activated(self) -> None:
        """Run once when this instance becomes the active store: the record is
        reconciled against the scene (:meth:`reconcile`)."""
        self.reconcile()

    def reconcile(self) -> List[int]:
        """Bring the record in line with the scene (the host's half,
        :meth:`KeyStashHost.reconcile`).

        Drops every clip none of whose parked keys still exist (an undone
        stash, a manual delete) and ends a preview the record says is active
        (a scene saved mid-preview must not reopen silently overridden).  The
        pure default touches nothing.

        Returns:
            The ids of the pruned clips.
        """
        return list(self.HOST.reconcile(self))

    def _reconcile_with(
        self,
        stash_alive: Callable[[Dict[str, Any]], Any],
        end_preview: Callable[[Dict[str, Any]], None],
    ) -> List[int]:
        """The reconcile every host runs (:meth:`KeyStashHost.reconcile`):
        drop each clip none of whose curve records *stash_alive* answers truthy
        for, end the recorded preview's scene side through *end_preview*
        (given the preview record), and tell the listeners once.

        Returns:
            The ids of the pruned clips.
        """
        gone: List[int] = []
        for clip in list(self.clips):
            if not any(stash_alive(rec) for rec in clip.curves):
                self.clips.remove(clip)
                gone.append(clip.clip_id)
        changed = bool(gone)
        if self.active_preview:
            end_preview(self.active_preview)
            self.active_preview = None
            changed = True
        if changed:
            self.mark_dirty()
            self._notify(StashChanged("reloaded"))
        return gone

    @classmethod
    def watch_undo(cls, owner: Any, widget: Any = None) -> List[int]:
        """Keep the loaded store in step with undo and redo, for *owner*.

        Each operation writes the record inside its undo step, so an undo or
        redo can move the record under the loaded store.  This subscribes
        :meth:`active` -- which re-reads a moved record, and the store's change
        event tells its listeners -- to the host's :attr:`UNDO_EVENTS` through
        its event manager (:meth:`KeyStashHost.script_jobs`), under *owner*;
        with *widget*, the subscriptions end when it is destroyed.  An event
        the host lacks is skipped, never raised: a panel watching the store
        must still open.  With no event manager nothing is subscribed.

        Parameters:
            owner: The subscriptions' owner (``ScriptJobManager.unsubscribe_all``).
            widget: A Qt widget whose destruction ends them.

        Returns:
            The subscription tokens.
        """
        jobs = cls.HOST.script_jobs()
        if jobs is None:
            return []
        tokens: List[int] = []
        for event in cls.UNDO_EVENTS:
            try:
                tokens.append(jobs.subscribe(event, cls.active, owner=owner))
            except Exception as error:  # noqa: BLE001 -- see the docstring
                _log.debug("KeyStash.watch_undo: %r unavailable (%s)", event, error)
        if widget is not None:
            jobs.connect_cleanup(widget, owner=owner)
        return tokens

    @classmethod
    def merge_carrier(cls, carriers: Any, other: Dict[Any, Any], ctx: Any) -> None:
        """Another scene's parked clips merged into the record (an owner hook,
        ``SceneStoreBase.OWNERS``): reload the active stash from it -- their
        parked keys came along with the import."""
        if SceneRecords.KEY_STASH in other:
            cls.invalidate()

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
    def flush_pending(cls) -> None:
        """Store what the active stash holds but has not written yet -- a DCC
        stash writes on idle, and a crossing reads the record (the owner hook
        ``SceneStoreBase`` calls first).  Mirror of ``ShotStore``'s."""
        if cls._active is not None:
            cls._active._flush_dirty()

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

    # ---- helpers the scene operations share ------------------------------

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


# ---------------------------------------------------------------------------
# The golden case
# ---------------------------------------------------------------------------


class KeyStashConformance:
    """The ``key_stash`` record's golden case: :meth:`fill` and :meth:`payload`.

    The store is one class on every host (:class:`KeyStash` over a
    :class:`KeyStashHost`), and what it saves -- the ``key_stash`` record a
    scene carries -- is a contract older scenes hold, so its spelling and bytes
    never move with the code.  This is that contract once, as data: one store
    holding one of everything :meth:`KeyStash.to_dict` saves (a clip with a
    source shot and metadata, one over two objects, a retired id, a running
    preview) and the exact text it is saved as.  pythontk's tests pin the
    reference store to it; mayatk's and blendertk's pin their host's store,
    saved through the scene's own persistence.  Mirror of the shot store's
    ``ShotStoreConformance``.
    """

    #: The objects the case's clips name.
    OBJECTS = ("pin_cube", "pin_sphere")

    #: The record :meth:`fill` saves, byte for byte, every key in the order
    #: the store writes them.
    _PAYLOAD = (
        '{"schema": 1, "scene_fps": 24.0, "clips": [{"clip_id": 1, '
        '"label": "walk in", "objects": ["pin_cube"], "curves": [{"times": '
        '[10.0, 20.0], "plug": {"name": "pin_cube", "attr": "translateX"}}], '
        '"created": "2026-09-02T10:00:00", "source_shot_id": 3, "metadata": '
        '{"take": "walk"}}, {"clip_id": 2, "label": "pin_cube +1 30-40.5", '
        '"objects": ["pin_cube", "pin_sphere"], "curves": [{"times": '
        '[30.0, 40.5], "plug": {"name": "pin_cube", "attr": "rotateY"}}, '
        '{"times": [35.0], "plug": {"name": "pin_sphere", "attr": "scaleX"}}], '
        '"created": "2026-09-02T10:00:00", "source_shot_id": null, '
        '"metadata": {}}], "preview": {"clip_id": 2, "in_context": true, '
        '"playback": [1.0, 100.0]}, "next_id": 4}'
    )

    #: When the case's clips were stashed (``add_clip`` stamps the clock).
    CREATED = "2026-09-02T10:00:00"

    @classmethod
    def fill(cls, store: "KeyStash") -> "KeyStash":
        """Give *store* -- a ``KeyStash`` on any host -- one of everything
        ``to_dict`` saves, as the case does.

        Returns:
            *store*.
        """
        a, b = cls.OBJECTS
        store.scene_fps = 24.0
        first = store.add_clip(
            [a],
            [{"times": [10.0, 20.0], "plug": {"name": a, "attr": "translateX"}}],
            label="walk in",
            source_shot_id=3,
            metadata={"take": "walk"},
        )
        second = store.add_clip(
            [a, b],
            [
                {"times": [30.0, 40.5], "plug": {"name": a, "attr": "rotateY"}},
                {"times": [35.0], "plug": {"name": b, "attr": "scaleX"}},
            ],
        )
        # A retired id: the saved high-water mark stays past it.
        store.remove_clip(store.add_clip([b], [{"times": [5.0]}]).clip_id)
        first.created = second.created = cls.CREATED
        store.set_preview(
            second.clip_id, {"in_context": True, "playback": [1.0, 100.0]}
        )
        return store

    @classmethod
    def payload(cls) -> str:
        """The exact ``key_stash`` text a :meth:`fill`-ed store saves."""
        return cls._PAYLOAD
