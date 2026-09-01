# !/usr/bin/python
# coding=utf-8
"""Prefix-scoped temp artifacts with an explicit lifetime policy.

The single home for the "write a temp payload for another process" pattern the app
hand-off bridges (and any exporter staging files for an external tool) share. Each
instance owns a filename *prefix* inside one directory and hands out
``<prefix>_<tag><ext>`` paths; the *policy* names the only three lifetimes that are
actually sound for inter-process payloads:

* ``"scoped"`` — the producer outlives the consumer (a blocking conversion, a test):
  delete on :meth:`cleanup` / clean ``with``-exit, **keep on exception** so failures
  stay debuggable.
* ``"session"`` — a detached consumer reads the file during this process's lifetime
  (a launched DCC): removed at interpreter exit via ``atexit``.
* ``"detached"`` — no deterministic delete exists (the consumer may outlive us and
  there is no completion signal). Allocation instead garbage-collects *stale* files
  of the same prefix (:meth:`sweep_stale`) — amortized cleanup with no risk to a
  payload another app may still be reading.

Every policy runs that stale sweep on its first allocation: detached because
nothing else deletes, scoped/session because keep-on-failure and hard-crash
leftovers have no other reclamation path.
"""
from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import tempfile
import threading
import time
from typing import Any, Callable, List, NamedTuple, Optional, Sequence

from pythontk.core_utils.logging_mixin import LoggingMixin


class TempArtifacts(LoggingMixin):
    """Allocate and lifecycle-manage ``<prefix>_*`` temp files/dirs in one directory.

    Use this instead of ``tempfile.mkstemp`` / ``mkdtemp`` / ``NamedTemporaryFile``
    for anything written under the system temp dir. A raw allocation has no owner:
    if the ``finally`` is forgotten, an exception escapes, or the process dies (a
    DCC crash is routine here), the artifact leaks with nothing left to reclaim it.
    Every allocation through this class joins a prefix namespace that a later run
    sweeps by age, so the worst case is delayed collection rather than a permanent
    leak. ``m3trik/scripts/check_temp_artifacts.py`` enforces this.

    Example (scoped — a synchronous convert-then-import round-trip):
        >>> with TempArtifacts("maya_to_btk", policy="scoped") as tmp:
        ...     fbx = tmp.path(extension=".fbx")
        ...     convert(src, fbx)   # on exception the fbx is kept + logged
        ...     import_fbx(fbx)     # clean exit removes it

    Example (scratch directory — the ``mkdtemp`` replacement):
        >>> with TempArtifacts("glb_export", policy="scoped") as tmp:
        ...     work = tmp.dir_path()      # created, tracked, swept if abandoned
        ...     build_into(work)

    Parameters:
        prefix: Filename stem prefix; also the sweep scope. Required, non-empty.
        policy: ``"scoped"`` | ``"session"`` | ``"detached"`` (default).
        dir: Base directory (default: the system temp dir).
        max_age_days: Age threshold for :meth:`sweep_stale` (the first-allocation
            stale-leftover GC every policy runs).
        on_cleanup: Optional callback invoked with the list of existing tracked
            paths just before they are removed (both :meth:`cleanup` and the
            context-manager exit). Exceptions are logged, never propagated.
    """

    POLICIES = ("scoped", "session", "detached")

    # Process-wide monotonic tag source. Deliberately NOT per-instance: distinct
    # stores routinely share one dir+prefix namespace (``CachedArtifact`` builds a
    # fresh scoped store per ``get``), and ``time_ns`` resolution on Windows
    # (~15ms) is far coarser than the gap between two allocations - so a
    # per-instance counter let two stores mint the same tag and hand back the same
    # path. The lock makes the read-modify-write atomic for concurrent producers.
    _tag_lock = threading.Lock()
    _last_tag_ns = 0

    def __init__(
        self,
        prefix: str,
        *,
        policy: str = "detached",
        dir: Optional[str] = None,  # noqa: A002 - matches tempfile's own param name
        max_age_days: float = 7,
        on_cleanup: Optional[Callable[[List[str]], None]] = None,
        log_level: str = "WARNING",
    ):
        super().__init__()
        if not prefix:
            raise ValueError("TempArtifacts requires a non-empty prefix.")
        if policy not in self.POLICIES:
            raise ValueError(
                f"Unknown policy {policy!r}. Expected one of {self.POLICIES}."
            )
        self.logger.setLevel(log_level)
        self.prefix = prefix
        self.policy = policy
        self.dir = dir or tempfile.gettempdir()
        self.max_age_days = max_age_days
        self.on_cleanup = on_cleanup
        self._tracked: List[str] = []
        self._atexit_registered = False
        self._swept = False

    # ------------------------------------------------------------------ allocation
    def path(self, extension: str = ".tmp", name: Optional[str] = None) -> str:
        """Return a tracked ``<prefix>_<tag><extension>`` path in :attr:`dir`.

        *name* fixes the tag (deterministic, self-overwriting — the rizom-style
        fixed-name pattern); otherwise a unique time-based tag is used. GC is
        amortized here: the first allocation sweeps stale same-prefix files
        (once per instance — a directory scan per allocation would be wasted;
        and never fresh files, since a recent payload may still be read by a
        launched app). Detached needs this because nothing else ever deletes;
        scoped/session need it because their keep-on-failure / hard-crash
        leftovers would otherwise accumulate forever.
        """
        if not self._swept:
            self._swept = True
            self.sweep_stale()
        tag = name if name is not None else f"{self._next_tag_ns():x}"
        return self.register(os.path.join(self.dir, f"{self.prefix}_{tag}{extension}"))

    def dir_path(self, name: Optional[str] = None, create: bool = True) -> str:
        """Return a tracked ``<prefix>_<tag>/`` DIRECTORY path in :attr:`dir`.

        The directory twin of :meth:`path`, and the reason it exists: without it
        every caller needing scratch space hand-rolled ``tempfile.mkdtemp`` plus a
        ``finally: shutil.rmtree``, so the lifecycle guarantees this class exists
        to provide (age-gated sweep of leftovers, keep-on-failure, one prefix
        namespace) had to be re-implemented per site -- and were simply missing
        wherever the ``finally`` was forgotten or the process died first.

        Tracked exactly like a file: :meth:`cleanup` removes it recursively and
        :meth:`sweep_stale` reclaims stale ones, so an abandoned directory is
        collected on a later run instead of leaking forever.
        """
        target = self.path(extension="", name=name)
        if create:
            os.makedirs(target, exist_ok=True)
        return target

    @classmethod
    def _next_tag_ns(cls) -> int:
        """A strictly-increasing ns-scale tag, unique process-wide.

        ``time.time_ns`` alone cannot carry this: its resolution is coarser than
        the interval between two allocations, so the counter floor supplies the
        distinctness the clock can't. Writes target :class:`TempArtifacts`
        explicitly rather than ``cls`` - assigning through a subclass would bind a
        *shadowing* class attribute and hand that subclass its own counter,
        reintroducing the very collision across the base/subclass pair.
        """
        with TempArtifacts._tag_lock:
            ns = max(time.time_ns(), TempArtifacts._last_tag_ns + 1)
            TempArtifacts._last_tag_ns = ns
            return ns

    def register(self, path: str) -> str:
        """Adopt *path* (e.g. a side artifact a tool wrote) into this lifecycle."""
        if path not in self._tracked:
            self._tracked.append(path)
        if self.policy == "session" and not self._atexit_registered:
            atexit.register(self.cleanup, force=True)
            self._atexit_registered = True
        return path

    def release(self, path: str) -> bool:
        """Delete ONE tracked *path* now, whatever the policy; did it go?

        The counterpart of :meth:`register`, for the artifact whose consumer is
        provably finished with it before the store's own lifetime ends. The
        motivating case is a ``detached`` store: its policy is right in general
        (a launched app reads the payload after we return, so nothing may
        delete) yet wrong for one intermediate a BLOCKING run consumes inside
        itself -- the WebXR preview's FBX, converted to GLB and never read
        again. Without this it waited out :attr:`max_age_days`: measured on a
        production assembly, 324 MB per push and 3.1 GB left in the temp dir.

        Only paths this store MINTED (or adopted via :meth:`register`) are
        removed, and the refusal is silent-and-false rather than an exception.
        The guarantee matters because the caller is typically a *strategy*
        object holding a path it did not allocate: a deliverer mounted on a
        bridge whose producer returns a durable file must not be able to delete
        it, and an ownership test the caller has to remember is one it can
        forget. Untracks only on SUCCESS: a path that could not be removed --
        a file another process still holds open, the ordinary reason this
        fails -- stays tracked so :meth:`cleanup` still retries it at scope
        exit. Dropping it on failure would quietly demote a recoverable miss
        to a leak only the age sweep can reach.

        Parameters:
            path: The artifact to remove.

        Returns:
            True when this store owned *path* and it is now gone; False when
            the path is untracked, was never created, or could not be removed.
        """
        if path not in self._tracked:
            return False
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)
            else:
                # Allocation only reserves a name, so a path never written has
                # nothing to remove -- and nothing left to track either.
                self._tracked.remove(path)
                return False
        except OSError as e:
            self.logger.warning(f"Could not release temp artifact {path}: {e}")
            return False
        self._tracked.remove(path)
        return True

    # ------------------------------------------------------------------ lifecycle
    def cleanup(self, force: bool = False) -> List[str]:
        """Remove tracked files per the policy; return the paths removed.

        ``scoped`` / ``session`` remove on every call; ``detached`` only when
        *force* is True (its payloads must outlive this process — stale ones are
        reclaimed by :meth:`sweep_stale` instead). Fires *on_cleanup* with the
        existing paths about to be removed.
        """
        if self.policy == "detached" and not force:
            return []
        existing = [p for p in self._tracked if os.path.exists(p)]
        if existing and self.on_cleanup is not None:
            try:
                self.on_cleanup(list(existing))
            except Exception as e:  # noqa: BLE001 - a callback must never block removal
                self.logger.warning(f"on_cleanup callback failed: {e}")
        removed = []
        for p in existing:
            try:
                # Directories are first-class here (see dir_path): a scratch dir
                # left behind leaks just as hard as a file, and harder to notice.
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
                removed.append(p)
            except OSError as e:
                self.logger.warning(f"Could not remove temp artifact {p}: {e}")
        self._tracked = [p for p in self._tracked if p not in removed]
        return removed

    def sweep_stale(self) -> List[str]:
        """Best-effort delete of ``<prefix>_*`` files in :attr:`dir` older than
        :attr:`max_age_days`; return the paths removed.

        Conservative by design: age-gated (never a fresh payload another app may
        still be reading), prefix-scoped (never another producer's files), and
        errors are swallowed (a locked file just waits for the next sweep). A
        directory is as fresh as its NEWEST entry, not its own mtime: a file
        rewritten in place (how Maya saves a ``.ma``) never touches the parent's
        mtime, so a scratch dir someone keeps saving into would otherwise read as
        abandoned and be reclaimed with their work in it.
        """
        cutoff = time.time() - self.max_age_days * 86400
        removed = []
        try:
            entries = os.scandir(self.dir)
        except OSError:
            return removed
        with entries:
            for entry in entries:
                if not entry.name.startswith(f"{self.prefix}_"):
                    continue
                try:
                    if self._newest_mtime(entry) >= cutoff:
                        continue
                    if entry.is_dir():
                        shutil.rmtree(entry.path)
                    else:
                        os.remove(entry.path)
                    removed.append(entry.path)
                except OSError:
                    continue
        return removed

    @staticmethod
    def _newest_mtime(entry: os.DirEntry) -> float:
        """The entry's mtime, or for a directory the newest of it and its files."""
        newest = entry.stat().st_mtime
        if entry.is_dir():
            try:
                with os.scandir(entry.path) as children:
                    for child in children:
                        try:
                            newest = max(newest, child.stat().st_mtime)
                        except OSError:
                            continue
            except OSError:
                pass
        return newest

    # ------------------------------------------------------------------ context manager
    def __enter__(self) -> "TempArtifacts":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.cleanup(force=True)
        else:  # keep everything for debugging; say where it is
            kept = [p for p in self._tracked if os.path.exists(p)]
            if kept:  # warning: visible at the default log level — files were left behind
                self.logger.warning(f"Keeping temp artifacts after failure: {kept}")


class CachedArtifact(LoggingMixin):
    """Produce-once / reuse-forever artifact behind a content-addressed cache slot.

    The lifecycle every headless-conversion pipeline otherwise re-implements: hash the
    inputs into a stable *key*, hand back the cached file when one exists, else produce it
    into a **scoped scratch** store and only then atomically promote it into the cache
    slot. Built on :class:`TempArtifacts` — the ``"detached"`` policy owns the cache
    (nothing can safely delete a payload another process may still read; stale entries are
    age-swept), ``"scoped"`` owns the scratch (kept + logged on failure).

    The scratch hop is load-bearing, not ceremony: producing straight into the cache slot
    lets a timeout-killed partial write poison every later run, and lets two concurrent
    producers of the same key interleave into one file. ``os.replace`` is atomic on both
    POSIX and Windows, so a slot is only ever empty or complete.

    Example (a headless DCC conversion, cached on scene + template identity):
        >>> cache = CachedArtifact("maya_to_btk", extension=".fbx")
        >>> key = CachedArtifact.key(sorted(opts.items()), files=[src, template])
        >>> got = cache.get(key, lambda out: convert(src, out), sidecars=(".manifest.json",))
        >>> import_fbx(got.path)
        >>> if got.scratch:          # a miss produced it — clean up on success
        ...     got.scratch.cleanup()

    Parameters:
        name: Prefix shared by the cache (``<name>_cache_*``) and scratch (``<name>_*``)
            stores; also the sweep scope of each.
        extension: Artifact file extension, e.g. ``".fbx"``.
        dir: Base directory for both stores (default: the system temp dir).
        max_age_days: Stale-sweep threshold for the cache store.
    """

    class Result(NamedTuple):
        """What :meth:`CachedArtifact.get` hands back.

        ``scratch`` is the scoped store on a miss (``cleanup()`` it once the artifact has
        been consumed) and ``None`` on a hit — persistence is the cache's whole point, so
        a hit must never be cleaned up.
        """

        path: str
        hit: bool
        scratch: Optional[TempArtifacts]

    def __init__(
        self,
        name: str,
        *,
        extension: str,
        dir: Optional[str] = None,  # noqa: A002 - matches TempArtifacts' own param name
        max_age_days: float = 7,
        log_level: str = "WARNING",
    ):
        super().__init__()
        if not name:
            raise ValueError("CachedArtifact requires a non-empty name.")
        self.logger.setLevel(log_level)
        self.name = name
        self.extension = extension
        self.dir = dir
        self.max_age_days = max_age_days

    @staticmethod
    def key(*parts: Any, files: Sequence[str] = (), length: int = 16) -> str:
        """A deterministic tag over *parts* and the identity of each path in *files*.

        A file contributes its path + mtime + size, so editing an input invalidates the
        key. Pass every file the artifact's content depends on — the source **and** the
        script/template that produces it: a template fix must invalidate stale payloads,
        or a retry after an upgrade silently replays the old bug.
        """
        blob = "|".join(repr(p) for p in parts)
        for f in files:
            stat = os.stat(f)
            blob += f"|{f}|{stat.st_mtime_ns}|{stat.st_size}"
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]

    def get(
        self,
        key: str,
        produce: Callable[[str], Any],
        *,
        sidecars: Sequence[str] = (),
        use_cache: bool = True,
    ) -> "CachedArtifact.Result":
        """The artifact for *key*, produced by ``produce(out_path)`` on a miss.

        Parameters:
            key: Content-addressed tag from :meth:`key` (or any stable string).
            produce: Called with the scratch path it must write; its return value is
                ignored (the artifact's *existence* is the contract). An exception
                propagates with the partial output kept + logged for debugging.
            sidecars: Suffixes appended to the artifact path that ``produce`` may also
                write (e.g. ``".manifest.json"``). Promoted with the artifact; a slot's
                stale sidecar is removed when the new run produced none, so a sidecar can
                never outlive the artifact it describes.
            use_cache: When False, produce into scratch every time and skip the cache
                entirely (the result's ``path`` is then the scratch path).

        Returns:
            CachedArtifact.Result: ``(path, hit, scratch)``.
        """
        cache_path = None
        if use_cache:
            store = TempArtifacts(
                f"{self.name}_cache",
                policy="detached",
                dir=self.dir,
                max_age_days=self.max_age_days,
            )
            cache_path = store.path(extension=self.extension, name=key)
            if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
                self.logger.info(
                    f"Cache hit ({os.path.basename(cache_path)}) -- skipping production."
                )
                return self.Result(cache_path, True, None)

        scratch = TempArtifacts(self.name, policy="scoped", dir=self.dir)
        out_path = scratch.path(extension=self.extension)
        for suffix in sidecars:
            scratch.register(out_path + suffix)
        try:
            produce(out_path)
        except Exception:
            if os.path.isfile(out_path):
                self.logger.warning(f"Keeping partial artifact for debugging: {out_path}")
            raise
        if cache_path is None:
            return self.Result(out_path, False, scratch)

        os.replace(out_path, cache_path)
        for suffix in sidecars:
            if os.path.isfile(out_path + suffix):
                os.replace(out_path + suffix, cache_path + suffix)
            elif os.path.isfile(cache_path + suffix):
                os.remove(cache_path + suffix)  # stale sidecar from a partial promote
        return self.Result(cache_path, False, scratch)


class ScratchTwins(LoggingMixin):
    """Per-source scratch twins of foreign files, discarded only while untouched.

    The "open a foreign scene as a new document" pattern the DCC Reference Managers
    share: a converted payload is COPIED to a deterministic, per-source scratch path
    the host opens (so a second Open resolves the same file and can close it), the
    copy's identity is stamped, and once the host has moved on the copy is discarded
    only if it is still that untouched copy — one the user saved into is real work
    and stays, its location logged. Built on :class:`TempArtifacts`: one tracked,
    age-swept ``<prefix>_<hash-of-source-path>/`` directory per source (same-named
    sources in different folders never share a twin), the twin named
    ``<stem>_<ext><extension>`` so the title bar and any Save-As default carry the
    provenance (``scene.ma`` → ``scene_ma.blend``) and it can never shadow a sibling
    ``scene<extension>``.

    Example (a Blender panel opening a baked Maya scene):
        >>> twins = ScratchTwins("btk_opened", extension=".blend")
        >>> scratch = twins.create(source_ma, baked_blend)   # copy + stamp
        >>> open_scene(scratch)
        ...                                                  # later, on close / next open
        >>> twins.discard_except(current_file)               # untouched twins go, saved ones stay

    Parameters:
        prefix: Directory prefix; also the sweep scope (one family per host).
        extension: The twin's file extension — the HOST's native format.
        dir: Base directory (default: the system temp dir).
        max_age_days: Stale-sweep threshold for abandoned twin directories. Generous by
            default: a twin is a working document until the user saves it elsewhere.
    """

    def __init__(
        self,
        prefix: str,
        *,
        extension: str,
        dir: Optional[str] = None,  # noqa: A002 - matches TempArtifacts' own param name
        max_age_days: float = 30,
        log_level: str = "WARNING",
    ):
        super().__init__()
        self.logger.setLevel(log_level)
        self.extension = extension
        self._store = TempArtifacts(
            prefix, policy="detached", dir=dir, max_age_days=max_age_days
        )
        # Twin path (normalized) -> (size, mtime_ns) as written by create().
        self._stamps: dict = {}
        # Same key -> the path as create() handed it out. The key is normcased, so
        # on Windows it is lowercased and is NOT what a caller can compare against
        # path_for(); discard_except reports these instead.
        self._paths: dict = {}

    @staticmethod
    def _key(path: str) -> str:
        return os.path.normcase(os.path.normpath(os.path.abspath(path)))

    @staticmethod
    def _stamp(path: str) -> Optional[tuple]:
        """``(size, mtime_ns)`` of *path*, or None when absent — the cheap identity that
        tells an untouched copy from one saved into since."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_size, st.st_mtime_ns)

    @staticmethod
    def _stamp_file(path: str) -> str:
        """Sidecar holding the stamp ``create`` wrote, beside the twin it describes."""
        return path + ".stamp"

    def _record(self, path: str) -> None:
        """Remember the stamp of the copy just written, in memory AND on disk.

        The on-disk half is what makes the untouched-test survive a host restart: a
        twin outlives the process that made it, so an in-memory map alone reports
        every twin from a previous session as unrecognized — which is precisely the
        case where overwriting it would destroy saved work.
        """
        stamp = self._stamp(path)
        key = self._key(path)
        self._stamps[key] = stamp
        self._paths[key] = path
        if stamp is None:
            return
        try:
            with open(self._stamp_file(path), "w", encoding="utf-8") as f:
                f.write(f"{stamp[0]} {stamp[1]}")
        except OSError:  # unwritable sidecar; the in-memory stamp still covers today
            self.logger.debug(f"Could not write scratch stamp beside: {path}")

    def _recorded(self, path: str) -> Optional[tuple]:
        """The stamp ``create`` last wrote for *path*, from memory or the sidecar."""
        key = self._key(path)
        if key in self._stamps:
            return self._stamps[key]
        try:
            with open(self._stamp_file(path), encoding="utf-8") as f:
                size, mtime_ns = f.read().split()
            return (int(size), int(mtime_ns))
        except (OSError, ValueError):
            return None

    def _is_untouched(self, path: str) -> bool:
        """True when *path* is still byte-for-byte the copy this store wrote."""
        recorded = self._recorded(path)
        return recorded is not None and self._stamp(path) == recorded

    def _forget_stamp(self, path: str) -> None:
        try:
            os.remove(self._stamp_file(path))
        except OSError:
            pass

    def _preserve(self, path: str) -> Optional[str]:
        """Move a twin that is NOT our untouched copy aside, and return its new path.

        Called only from ``create``: the twin path is deterministic per source, so
        re-opening the same source lands on a file the user may have saved real work
        into. Renaming beside it keeps that work, keeps ``path_for`` deterministic
        (the panels recompute it to answer "is this row the current scene?"), and
        still lets the fresh conversion be written.
        """
        stem, ext = os.path.splitext(path)
        n = 1
        while os.path.exists(f"{stem}_saved{n}{ext}"):
            n += 1
        kept = f"{stem}_saved{n}{ext}"
        try:
            os.replace(path, kept)
        except OSError:
            self.logger.warning(
                f"Could not move a modified scratch aside; leaving it untouched "
                f"and NOT overwriting: {path}"
            )
            return None
        self._forget_stamp(path)
        self.logger.warning(f"Scratch had unsaved-elsewhere changes; kept as: {kept}")
        return kept

    def path_for(self, source: str) -> str:
        """The deterministic twin path for *source* (nothing is created)."""
        key = self._key(source)
        stem, ext = os.path.splitext(os.path.basename(source))
        folder = self._store.dir_path(
            name=hashlib.sha1(key.encode("utf-8")).hexdigest()[:10], create=False
        )
        return os.path.join(folder, f"{stem}_{ext.lstrip('.').lower()}{self.extension}")

    def create(self, source: str, payload: str) -> str:
        """Copy *payload* to *source*'s twin path, stamp it, and return the path.
        Raises ``OSError`` when the copy cannot be written."""
        path = self.path_for(source)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not self._is_untouched(path):
            # Not the copy we wrote: the user saved into it, or a previous session
            # left it. Either way it is real work — move it aside, never over it.
            if self._preserve(path) is None:
                return path
        shutil.copyfile(payload, path)
        self._record(path)
        return path

    def is_twin(self, path: str) -> bool:
        """True if *path* is a twin this store created and still tracks."""
        return bool(path) and self._key(path) in self._stamps

    def discard(self, path: str) -> bool:
        """Delete the twin at *path* if it is still the untouched copy (stamp unchanged);
        a twin the user saved into is kept, its location logged. Either way the twin is
        forgotten. Returns True when a file was removed."""
        if not path:
            return False
        key = self._key(path)
        stamp = self._recorded(path)
        self._stamps.pop(key, None)
        self._paths.pop(key, None)
        if stamp is None:
            return False
        if self._stamp(path) != stamp:
            self.logger.info(f"Keeping saved scratch: {path}")
            return False
        try:
            os.remove(path)
        except OSError:
            return False
        self._forget_stamp(path)
        self.logger.info(f"Removed scratch: {path}")
        return True

    def discard_except(self, keep: Optional[str] = None) -> List[str]:
        """Discard every tracked twin other than *keep* (the host's current file) — the
        rest were replaced by a close or another open, so an untouched one has nothing
        left to represent.

        Returns:
            The twins removed, spelled as ``create`` handed them out -- not the
            normcased internal key, which on Windows is lowercased and so compares
            unequal to :meth:`path_for`.
        """
        keep_key = self._key(keep) if keep else None
        removed = []
        for key in list(self._stamps):
            if key == keep_key:
                continue
            path = self._paths.get(key, key)
            if self.discard(path):
                removed.append(path)
        return removed


__all__ = ["TempArtifacts", "CachedArtifact", "ScratchTwins"]
