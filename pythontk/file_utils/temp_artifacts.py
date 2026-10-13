# !/usr/bin/python
# coding=utf-8
"""Prefix-scoped temp artifacts with an explicit lifetime policy.

The single home for every temp file and folder the ecosystem writes: the app
hand-off payloads, an exporter's staging, a conversion's scratch, a cache of
regenerable results. Everything lives under ONE root (:meth:`TempArtifacts.root`:
``$PYTHONTK_TEMP_ROOT``, else ``<system temp>/ptk``), and a store with no ``dir``
owns one folder in it, ``<root>/<prefix>/``, holding ``<prefix>_<tag><ext>``
entries. A store's sweep reads only its own folder, so one store can never reach
another's entries -- a prefix that extends another (``glb_pipeline`` /
``glb_pipeline_media``) is simply a different folder.

The *policy* names the only three lifetimes that are actually sound for
inter-process payloads:

* ``"scoped"`` — the producer outlives the consumer (a blocking conversion, a test):
  delete on :meth:`~TempArtifacts.cleanup` / clean ``with``-exit, **keep on
  exception** so failures stay debuggable. :meth:`TempArtifacts.scoped` spells it.
* ``"session"`` — a detached consumer reads the file during this process's lifetime
  (a launched DCC): removed at interpreter exit via ``atexit``, and at
  :meth:`ProcessExit.hard_exit <pythontk.ProcessExit.hard_exit>`, which skips it.
  Its tags name the owner (``<prefix>_<tag>-<pid>``), so what outlives the owner --
  killed, or still held when it exited -- goes at the next same-prefix allocation.
* ``"detached"`` (the default) — no deterministic delete exists (the consumer may
  outlive us and there is no completion signal). Allocation instead
  garbage-collects *stale* entries of the same prefix
  (:meth:`~TempArtifacts.sweep_stale`) — amortized cleanup with no risk to a
  payload another app may still be reading.

Every policy runs that stale sweep on its first allocation: detached because
nothing else deletes, scoped/session because keep-on-failure and hard-crash
leftovers have no other reclamation path. A store folder records the age its
owner declared, so :meth:`TempArtifacts.sweep_all` -- run on its own about once
a day -- also reclaims a prefix that never allocates again.

Removal is best-effort, never all-or-nothing: an entry something still holds
stays (for a later cleanup or sweep) and everything else goes now.

:class:`CachedArtifact` is the produce-once cache built on it (temp-resident, or
project-resident with ``dir`` and ``max_age_days=None``); :class:`ScratchTwins`
the per-source scratch copies a Reference Manager opens.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
import threading
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.core_utils.process_exit import ProcessExit

#: "Never written": distinct from a declared ``None`` (never age-swept).
_UNSET = object()


class TempArtifacts(LoggingMixin):
    """Allocate and lifecycle-manage ``<prefix>_*`` temp files/dirs.

    Use this instead of ``tempfile.mkstemp`` / ``mkdtemp`` / ``NamedTemporaryFile``
    for anything written to temp. A raw allocation has no owner: if the
    ``finally`` is forgotten, an exception escapes, or the process dies (a DCC
    crash is routine here), the artifact leaks with nothing left to reclaim it.
    Every allocation through this class joins a prefix namespace that a later run
    sweeps by age, so the worst case is delayed collection rather than a permanent
    leak. ``m3trik/scripts/check_temp_artifacts.py`` enforces this.

    **Choose the policy explicitly.** The default is ``"detached"`` for
    compatibility, and it is right only for a payload another process reads after
    you return: nothing deletes it but the age sweep. Scratch the caller consumes
    itself is ``TempArtifacts.scoped(prefix)`` -- under ``with`` it goes on a
    clean exit and stays for debugging on an exception; :meth:`keep` disowns the
    one result that must outlive the scope.

    Example (scoped — a synchronous convert-then-import round-trip):
        >>> with TempArtifacts.scoped("maya_to_btk") as tmp:
        ...     fbx = tmp.path(extension=".fbx")
        ...     convert(src, fbx)   # on exception the fbx is kept + logged
        ...     import_fbx(fbx)     # clean exit removes it

    Example (scratch directory — the ``mkdtemp`` replacement):
        >>> with TempArtifacts.scoped("glb_export") as tmp:
        ...     work = tmp.dir_path()      # created, tracked, swept if abandoned
        ...     build_into(work)

    Parameters:
        prefix: Filename stem prefix; also the store's folder name under the root
            and the sweep scope. Required, non-empty.
        policy: ``"scoped"`` | ``"session"`` | ``"detached"`` (default).
        dir: Base directory. ``None`` (the norm) gives the store its own folder,
            ``<root>/<prefix>/`` (:meth:`root`). A directory the caller names holds
            the entries flat, ``<dir>/<prefix>_<tag>``, for an artifact that must
            sit beside another file (:meth:`FileUtils.atomic_write`'s ``.part``,
            a project cache); its sweep matches the prefix there, so give it one
            no neighbour extends.
        max_age_days: Age past which :meth:`sweep_stale` reclaims an entry, judged
            by the newest file in it. ``None``: never by age (a project-resident
            cache a saved scene references).
        on_cleanup: Optional callback invoked with the list of existing tracked
            paths just before they are removed (both :meth:`cleanup` and the
            context-manager exit). Exceptions are logged, never propagated.
        sweep_interval: Seconds within which a repeat automatic sweep of the same
            folder + prefix in this process is skipped. ``None`` (default): every
            new store's first allocation sweeps. For a store built per call over
            a folder the caller does not own (``atomic_write``: one store per
            write, into the target's folder), where a scan per call costs more
            than the leftovers it reclaims.
    """

    POLICIES = ("scoped", "session", "detached")

    #: Names the temp root; else it is :attr:`ROOT_NAME` in the system temp dir.
    #: :meth:`TestSandbox.temp <pythontk.TestSandbox.temp>` sets it to its own root.
    ROOT_ENV_VAR = "PYTHONTK_TEMP_ROOT"
    #: The root's folder name in the system temp dir.
    ROOT_NAME = "ptk"
    #: A store folder under the root holds this file: the age its owner declared,
    #: read by :meth:`sweep_all`. A folder without one is no store and is never
    #: touched.
    STORE_MARKER = ".ptk_store.json"
    #: The age :meth:`sweep_all` reads a declaration it cannot parse as.
    UNKNOWN_MAX_AGE_DAYS = 30
    #: Seconds between the automatic :meth:`sweep_all` runs (a stamp in the root).
    SWEEP_ALL_INTERVAL = 86400.0
    _SWEEP_STAMP = ".ptk_swept"
    #: The pre-root layout kept every entry flat in the system temp dir; a store's
    #: first sweep also reclaims its prefix's stale flat leftovers there, for one
    #: release. ``test_temp_artifacts`` fails once this window closes.
    _LEGACY_SWEEP_REMOVE_IN = "0.16.0"
    _LEGACY_SWEEP_SINCE = "2026-10-13"
    #: Fresh tags tried before :meth:`dir_path` gives up on an exclusive folder.
    _MINT_ATTEMPTS = 64

    # Process-wide monotonic tag source. Deliberately NOT per-instance: distinct
    # stores routinely share one dir+prefix namespace (``CachedArtifact`` builds a
    # fresh scoped store per ``get``), and ``time_ns`` resolution on Windows
    # (~15ms) is far coarser than the gap between two allocations - so a
    # per-instance counter let two stores mint the same tag and hand back the same
    # path. The lock makes the read-modify-write atomic for concurrent producers.
    _tag_lock = threading.Lock()
    _last_tag_ns = 0
    #: A session tag after the prefix: the allocation clock, then its owner's pid.
    _OWNED_TAG = re.compile(r"[0-9a-f]{12,}-([0-9]{1,10})(?:\.|$)")

    # Process-wide memos, written through ``TempArtifacts`` by name (a subclass
    # must share them, never shadow them). Guarded where a read-modify-write is.
    _state_lock = threading.Lock()
    #: ``(env value, system temp dir)`` -> the resolved root.
    _roots: Dict[Tuple[str, str], str] = {}
    #: Normalized system temp dir -> its entry names, listed once a process.
    _legacy_listings: Dict[str, List[str]] = {}
    #: ``(normalized system temp dir, prefix)`` already given its legacy sweep.
    _legacy_swept: Set[Tuple[str, str]] = set()
    #: Normalized roots whose automatic :meth:`sweep_all` was considered.
    _sweep_all_checked: Set[str] = set()
    #: ``(normalized folder, prefix)`` -> monotonic time of its last sweep.
    _swept_at: Dict[Tuple[str, str], float] = {}
    #: Normalized marker path -> the age this process last declared in it.
    _markers: Dict[str, Any] = {}

    def __init__(
        self,
        prefix: str,
        *,
        policy: str = "detached",
        dir: Optional[str] = None,  # noqa: A002 - matches tempfile's own param name
        max_age_days: Optional[float] = 7,
        on_cleanup: Optional[Callable[[List[str]], None]] = None,
        sweep_interval: Optional[float] = None,
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
        #: True when the store owns ``<root>/<prefix>/`` (no ``dir`` was named).
        self.managed = not dir
        self.dir = os.path.join(self.root(), prefix) if self.managed else os.fspath(dir)
        self.max_age_days = max_age_days
        self.on_cleanup = on_cleanup
        self.sweep_interval = sweep_interval
        self._tracked: List[str] = []
        self._atexit_registered = False
        self._swept = False

    @classmethod
    def scoped(cls, prefix: str, **kwargs: Any) -> "TempArtifacts":
        """A ``"scoped"`` store: ``TempArtifacts(prefix, policy="scoped", ...)``.

        The explicit spelling for scratch the caller consumes itself -- the
        default policy is ``"detached"``, whose entries nothing deletes but the
        age sweep.

        Parameters:
            prefix: The store's prefix.
            **kwargs: Any other constructor keyword (``dir``, ``max_age_days``,
                ``on_cleanup``, ...).

        Returns:
            The new store.
        """
        return cls(prefix, policy="scoped", **kwargs)

    # ------------------------------------------------------------------ the root
    @classmethod
    def root(cls) -> str:
        """The ONE temp root every store with no ``dir`` lives under, created.

        ``$PYTHONTK_TEMP_ROOT`` (``~`` and ``%VAR%`` expanded) when set, else
        ``<tempfile.gettempdir()>/ptk``. Spelled so a program that reads paths in
        the ANSI code page can open it (:meth:`AppLauncher.ansi_safe_path
        <pythontk.AppLauncher.ansi_safe_path>`): Maya, under a TEMP holding a
        letter its code page lacks, opens the 8.3 form and nothing else -- every
        payload minted under the root inherits it.

        Returns:
            The root directory (absolute).
        """
        override = os.environ.get(cls.ROOT_ENV_VAR) or ""
        system = tempfile.gettempdir()
        key = (override, system)
        cached = TempArtifacts._roots.get(key)
        if cached is not None and os.path.isdir(cached):
            return cached
        if override:
            raw = os.path.abspath(os.path.expandvars(os.path.expanduser(override)))
        else:
            raw = os.path.join(system, cls.ROOT_NAME)
        os.makedirs(raw, exist_ok=True)
        from pythontk.core_utils.app_launcher import AppLauncher

        safe = os.fspath(AppLauncher.ansi_safe_path(raw))
        TempArtifacts._roots[key] = safe
        return safe

    @classmethod
    def contains(cls, path: str) -> bool:
        """Whether *path* is scratch this primitive holds: under :meth:`root`
        -- or, for the one-release legacy window
        (``_LEGACY_SWEEP_REMOVE_IN``), the system temp dir the pre-root flat
        layout used, where an entry minted before the root (a scene's scratch
        twin) still lives.

        Both sides resolved (``os.path.realpath``: a junction or an 8.3
        spelling of the root reads the same), compared on separator
        boundaries (:meth:`FileUtils.is_under`); the root itself counts.

        Parameters:
            path: The path to test (absolute).

        Returns:
            True when it lies there; False for an empty or unresolvable path.
        """
        from pythontk.file_utils._file_utils import FileUtils

        try:
            real = os.path.realpath(os.fspath(path)) if path else ""
        except (OSError, TypeError, ValueError):
            return False
        if not real or "\0" in real:
            return False
        bases = (cls.root(), tempfile.gettempdir())  # legacy: drop at 0.16.0
        return any(FileUtils.is_under(real, os.path.realpath(b)) for b in bases)

    # ------------------------------------------------------------------ allocation
    def path(self, extension: str = ".tmp", name: Optional[str] = None) -> str:
        """Return a tracked ``<prefix>_<tag><extension>`` path in :attr:`dir`.

        *name* fixes the tag (deterministic, self-overwriting — the rizom-style
        fixed-name pattern); otherwise a unique tag is minted: the allocation
        clock, this process's id and a random part (two processes in one ~15 ms
        clock tick never share a path), which a ``session`` store spells
        ``<clock>-<pid>`` to name its owner. Only a name is reserved -- the file
        is the caller's to write -- but a store with no ``dir`` creates its
        folder first, so the write lands.

        GC is amortized here: the first allocation sweeps stale same-prefix
        entries (once per instance — a directory scan per allocation would be
        wasted; and never fresh ones, since a recent payload may still be read
        by a launched app — unless a session tag names an owner that has
        exited). Detached needs this because nothing else ever deletes;
        scoped/session need it because their keep-on-failure / hard-crash
        leftovers would otherwise accumulate forever.
        """
        self._ready()
        tag = name if name is not None else self._mint_tag()
        return self.register(self._entry(tag, extension))

    def dir_path(self, name: Optional[str] = None, create: bool = True) -> str:
        """Return a tracked ``<prefix>_<tag>/`` DIRECTORY path in :attr:`dir`.

        The directory twin of :meth:`path`, and the reason it exists: without it
        every caller needing scratch space hand-rolled ``tempfile.mkdtemp`` plus a
        ``finally: shutil.rmtree``, so the lifecycle guarantees this class exists
        to provide (age-gated sweep of leftovers, keep-on-failure, one prefix
        namespace) had to be re-implemented per site -- and were simply missing
        wherever the ``finally`` was forgotten or the process died first.

        A minted directory is created EXCLUSIVELY: one that already exists
        belongs to someone else, and a fresh tag is tried instead -- never a
        shared folder. A *name* is a fixed, shared folder by design (created if
        missing). Tracked exactly like a file: :meth:`cleanup` removes it
        recursively and :meth:`sweep_stale` reclaims stale ones.

        Parameters:
            name: A fixed tag (a shared, self-overwriting folder).
            create: False reserves the name only.

        Returns:
            The directory path.

        Raises:
            FileExistsError: No fresh name could be minted (never in practice).
        """
        if name is not None or not create:
            target = self.path(extension="", name=name)
            if create:
                os.makedirs(target, exist_ok=True)
            return target
        for _ in range(self._MINT_ATTEMPTS):
            target = self.path(extension="")
            os.makedirs(self.dir, exist_ok=True)
            try:
                os.mkdir(target)
                return target
            except FileExistsError:
                self._forget(target)  # someone else's: never ours to delete
        raise FileExistsError(f"Could not mint a fresh directory in {self.dir}.")

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

    def _mint_tag(self) -> str:
        """A fresh tag: the clock, then this process -- the owner, for a
        session store -- and, past a session, a random part, all one lowercase
        hex token (``<clock:016x><pid:x><random:04x>``).

        The counter (:meth:`_next_tag_ns`) is unique only inside this process:
        two processes in one clock tick minted one path. The pid separates
        processes alive together, the random part one whose id was reused.
        """
        ns = self._next_tag_ns()
        if self.policy == "session":
            return f"{ns:x}-{os.getpid()}"
        return f"{ns:016x}{os.getpid():x}{secrets.randbits(16):04x}"

    def _entry(self, tag: str, extension: str) -> str:
        """The path of the entry named *tag* -- the one naming rule."""
        return os.path.join(self.dir, f"{self.prefix}_{tag}{extension}")

    def register(self, path: str) -> str:
        """Adopt *path* (e.g. a side artifact a tool wrote) into this lifecycle."""
        if path not in self._tracked:
            self._tracked.append(path)
        if self.policy == "session" and not self._atexit_registered:
            atexit.register(self.cleanup, force=True)
            # hard_exit skips atexit, and it is how every DCC child and several
            # test runners end -- each leaked its whole store until this.
            ProcessExit.register(self.cleanup, force=True)
            self._atexit_registered = True
        return path

    def keep(self, path: str) -> bool:
        """Disown *path*: :meth:`cleanup` and the scope exit leave it.

        For the one result of a scoped run that must outlive the scope -- a
        stage that holds the only copy of a deliverable whose move failed. It
        replaces the ``stage_dir = None`` + ``finally: rmtree`` idiom: the rest
        of the store still goes, and so does a run that fails before the
        decision. A kept entry stays in the store's folder, where the age sweep
        reclaims it once stale: move it out of temp to keep it for good.

        Parameters:
            path: A path this store minted or adopted.

        Returns:
            True when it was tracked and is now disowned.
        """
        found = self._tracked_spelling(path)
        if found is None:
            return False
        self._tracked.remove(found)
        return True

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
        found = self._tracked_spelling(path)
        if found is None:
            return False
        if not os.path.exists(found):
            # Allocation only reserves a name, so a path never written has
            # nothing to remove -- and nothing left to track either.
            self._tracked.remove(found)
            return False
        left = self._remove(found)
        if left:
            self.logger.warning(
                f"Could not release temp artifact {found}: {self._left_note(left)}"
            )
            return False
        self._tracked.remove(found)
        return True

    def _tracked_spelling(self, path: str) -> Optional[str]:
        """*path* as this store tracks it (the same file in any spelling), or None."""
        if path in self._tracked:
            return path
        key = self._norm(path)
        return next((p for p in self._tracked if self._norm(p) == key), None)

    def _forget(self, path: str) -> None:
        """Untrack *path* without touching it."""
        if path in self._tracked:
            self._tracked.remove(path)

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
            # Directories are first-class here (see dir_path): a scratch dir
            # left behind leaks just as hard as a file, and harder to notice.
            left = self._remove(p)
            if left:
                self.logger.warning(
                    f"Could not remove temp artifact {p}: {self._left_note(left)}"
                )
            else:
                removed.append(p)
        self._tracked = [p for p in self._tracked if p not in removed]
        return removed

    def sweep_stale(self) -> List[str]:
        """Best-effort delete of this store's stale ``<prefix>_*`` entries in
        :attr:`dir`; return the paths removed.

        Conservative by design: age-gated (never a fresh payload another app may
        still be reading), scoped to this store (its own folder under the root;
        the prefix in a folder the caller named), and errors are swallowed (a
        locked file just waits for the next sweep). A directory is as fresh as
        its NEWEST entry, not its own mtime: a file rewritten in place (how Maya
        saves a ``.ma``) never touches the parent's mtime, so a scratch dir
        someone keeps saving into would otherwise read as abandoned and be
        reclaimed with their work in it. ``max_age_days=None`` never ages.

        One exception to the age gate: a ``session`` tag names its owner, and an
        entry whose owner has exited is used by nobody, so it goes at any age. A
        live owner, or an id another process has taken since, leaves it to the
        age gate (the sweep assumes :attr:`dir` is on this machine, as the
        system temp dir is).

        A store under the root also reclaims, for one release, its prefix's
        stale entries the pre-root layout left flat in the system temp dir.
        """
        removed = self._sweep_folder(self.dir, self.prefix, self.max_age_days)
        if self.managed:
            removed += self._sweep_legacy()
        TempArtifacts._swept_at[self._sweep_key()] = time.monotonic()
        return removed

    @classmethod
    def sweep_all(cls, root: Optional[str] = None) -> List[str]:
        """Sweep every store folder under the root by the age its owner declared.

        A store sweeps only on its own allocations, so a prefix that never
        allocates again -- a retired tool, a suite that skips -- was swept by no
        one. Each store folder records its declared ``max_age_days``
        (:attr:`STORE_MARKER`); an entry younger than that is never removed,
        ``None`` is never aged, and a declaration that cannot be read counts as
        :attr:`UNKNOWN_MAX_AGE_DAYS`. A folder with no declaration is no store
        and is never touched, nor is anything loose in the root. Runs on its own
        about once a day (:attr:`SWEEP_ALL_INTERVAL`), at the first allocation a
        process makes under the root.

        Parameters:
            root: The root to sweep (default: :meth:`root`).

        Returns:
            The entries removed.
        """
        root = cls.root() if root is None else os.fspath(root)
        removed: List[str] = []
        try:
            with os.scandir(root) as entries:
                folders = [e for e in entries if e.is_dir(follow_symlinks=False)]
        except OSError:
            return removed
        for folder in folders:
            marker = os.path.join(folder.path, cls.STORE_MARKER)
            if not os.path.isfile(marker):
                continue
            removed += cls._sweep_folder(
                folder.path, folder.name, cls._declared_age(marker)
            )
        try:
            with open(os.path.join(root, cls._SWEEP_STAMP), "w"):
                pass
        except OSError:
            pass
        return removed

    def _ready(self) -> None:
        """Before an allocation: the store's folder exists (one with no ``dir``),
        and the first allocation has swept."""
        if self.managed:
            os.makedirs(self.dir, exist_ok=True)
        if self._swept:
            return
        self._swept = True
        if self.managed:
            self._declare()
        recent = TempArtifacts._swept_at.get(self._sweep_key())
        skip = (
            self.sweep_interval is not None
            and recent is not None
            and time.monotonic() - recent < self.sweep_interval
        )
        if not skip:
            self.sweep_stale()
        if self.managed:
            self._maybe_sweep_all(os.path.dirname(self.dir))

    def _sweep_key(self) -> Tuple[str, str]:
        return (self._norm(self.dir), self.prefix)

    def _declare(self) -> None:
        """Record this store's declared age in its folder (:attr:`STORE_MARKER`),
        for :meth:`sweep_all`. Best-effort: unrecorded, the folder is skipped.

        The LONGEST age any store of the folder declared stands (``None``,
        never aged, the longest): a store spelled with a shorter default must
        not let :meth:`sweep_all` age out a longer-lived one's entries.
        """
        marker = os.path.join(self.dir, self.STORE_MARKER)
        key = self._norm(marker)
        want = self.max_age_days
        memo = TempArtifacts._markers.get(key, _UNSET)
        if memo is not _UNSET and not self._outlives(want, memo):
            return
        on_disk = self._declared_age(marker, missing=_UNSET)
        if on_disk is not _UNSET and not self._outlives(want, on_disk):
            TempArtifacts._markers[key] = on_disk
            return
        from pythontk.file_utils._file_utils import FileUtils

        try:
            FileUtils.write_json(marker, {"prefix": self.prefix, "max_age_days": want})
        except OSError:
            return
        TempArtifacts._markers[key] = want

    @staticmethod
    def _outlives(age: Optional[float], other: Optional[float]) -> bool:
        """Whether *age* keeps an entry strictly longer than *other* (``None``:
        never aged)."""
        return other is not None and (age is None or age > other)

    @classmethod
    def _declared_age(cls, marker: str, missing: Any = None) -> Any:
        """The ``max_age_days`` a store folder declares; *missing* when it has
        no declaration, :attr:`UNKNOWN_MAX_AGE_DAYS` when it cannot be read."""
        try:
            with open(marker, encoding="utf-8") as fh:
                age = json.load(fh)["max_age_days"]
        except FileNotFoundError:
            return missing
        except (OSError, ValueError, KeyError, TypeError):
            return cls.UNKNOWN_MAX_AGE_DAYS
        if age is None or (
            isinstance(age, (int, float)) and not isinstance(age, bool) and age >= 0
        ):
            return age
        return cls.UNKNOWN_MAX_AGE_DAYS

    @classmethod
    def _maybe_sweep_all(cls, root: str) -> None:
        """:meth:`sweep_all` once a process, when the root's last one is a day old."""
        key = cls._norm(root)
        with TempArtifacts._state_lock:
            if key in TempArtifacts._sweep_all_checked:
                return
            TempArtifacts._sweep_all_checked.add(key)
        try:
            last = os.stat(os.path.join(root, cls._SWEEP_STAMP)).st_mtime
            if time.time() - last < cls.SWEEP_ALL_INTERVAL:
                return
        except OSError:
            pass
        try:
            cls.sweep_all(root)
        except Exception as e:  # noqa: BLE001 - housekeeping never fails an allocation
            cls.logger.debug(f"sweep_all failed: {e}")

    def _sweep_legacy(self) -> List[str]:
        """The one-release sweep of this prefix's flat pre-root leftovers.

        Lists the system temp dir once a process (it can hold tens of thousands
        of entries) and sweeps each prefix from that list once. A folder that is
        a store (a root placed IN the system temp dir) is never a leftover.
        """
        legacy = tempfile.gettempdir()
        legacy_key = self._norm(legacy)
        with TempArtifacts._state_lock:
            if (legacy_key, self.prefix) in TempArtifacts._legacy_swept:
                return []
            TempArtifacts._legacy_swept.add((legacy_key, self.prefix))
            names = TempArtifacts._legacy_listings.get(legacy_key)
        if names is None:
            try:
                with os.scandir(legacy) as entries:
                    names = [e.name for e in entries]
            except OSError:
                names = []
            TempArtifacts._legacy_listings[legacy_key] = names
        root = self._norm(os.path.dirname(self.dir))
        cutoff = self._cutoff(self.max_age_days)
        removed = []
        for name in names:
            if not name.startswith(f"{self.prefix}_"):
                continue
            path = os.path.join(legacy, name)
            try:
                if self._norm(path) == root or os.path.isfile(
                    os.path.join(path, self.STORE_MARKER)
                ):
                    continue
                if not self._owner_exited(name, self.prefix) and (
                    cutoff is None or self._newest_mtime(path) >= cutoff
                ):
                    continue
            except OSError:
                continue
            if not self._remove(path):
                removed.append(path)
        return removed

    @classmethod
    def _sweep_folder(
        cls, folder: str, prefix: str, max_age_days: Optional[float]
    ) -> List[str]:
        """Remove *folder*'s stale or ownerless ``<prefix>_*`` entries."""
        cutoff = cls._cutoff(max_age_days)
        removed = []
        try:
            entries = os.scandir(folder)
        except OSError:
            return removed
        with entries:
            for entry in entries:
                if not entry.name.startswith(f"{prefix}_"):
                    continue
                try:
                    if not cls._owner_exited(entry.name, prefix) and (
                        cutoff is None or cls._newest_mtime(entry) >= cutoff
                    ):
                        continue
                except OSError:
                    continue
                # A held entry waits for the next sweep; the rest goes now.
                if not cls._remove(entry.path):
                    removed.append(entry.path)
        return removed

    @staticmethod
    def _cutoff(max_age_days: Optional[float]) -> Optional[float]:
        """The mtime below which an entry is stale; None = never by age."""
        if max_age_days is None:
            return None
        return time.time() - max_age_days * 86400

    @classmethod
    def _owner_exited(cls, name: str, prefix: str) -> bool:
        """True when *name* is a session tag whose owning process has exited."""
        match = cls._OWNED_TAG.match(name[len(prefix) + 1 :])
        if not match:
            return False
        from pythontk.core_utils.execution_monitor._sidecar import process_alive

        try:
            return not process_alive(int(match.group(1)))
        except Exception:  # noqa: BLE001 - an id no process could have
            return False

    @staticmethod
    def _norm(path: str) -> str:
        """A comparable spelling of *path*."""
        return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(path))))

    @staticmethod
    def _remove(path: str) -> List[str]:
        """Delete *path* -- a file, or a directory and all in it -- as far as it
        will go; return what stayed as ``"<entry>: <error>"`` lines (``[]`` = gone).

        Best-effort where ``shutil.rmtree`` is all-or-nothing. rmtree stops at
        the first entry it cannot delete, and on Windows that is routine: a
        helper a DCC spawns outlives it holding a file in the DCC's temp dir
        (Autodesk's ADPClientService, observed). One held file then kept a whole
        test sandbox where only that file had to stay -- with the hard-exit leak
        in :meth:`register`, 791 sandboxes and 572 MB in the system temp dir
        after a week of runs. A read-only entry (a git object), which Windows
        refuses to delete, is made writable and retried; anything else is
        skipped and the walk carries on, so only what is still held remains.
        """
        left: List[str] = []

        def skip(func, target, exc):
            if isinstance(exc, tuple):  # onerror's exc_info, before 3.12
                exc = exc[1]
            # Never chmod through a link: that would edit a file outside the tree.
            if func in (os.unlink, os.remove, os.rmdir) and not os.path.islink(target):
                try:
                    os.chmod(target, os.stat(target).st_mode | stat.S_IWRITE)
                    func(target)
                    return
                except OSError as e:
                    exc = e
            if os.path.lexists(target):  # gone meanwhile is not a leftover
                left.append(f"{target}: {exc}")

        if os.path.isdir(path):
            if sys.version_info >= (3, 12):
                shutil.rmtree(path, onexc=skip)
            else:
                shutil.rmtree(path, onerror=skip)
        else:
            try:
                os.remove(path)
            except OSError as e:
                skip(os.remove, path, e)
        return left

    @staticmethod
    def _left_note(left: List[str]) -> str:
        """One log line for :meth:`_remove`'s leftovers: the first, and a count.

        The first is the held entry itself; the rest are mostly its ancestors,
        which could not go because it did not."""
        return left[0] + (f" (+{len(left) - 1} more)" if len(left) > 1 else "")

    @staticmethod
    def _newest_mtime(entry: Any) -> float:
        """The mtime of *entry* (a ``DirEntry`` or a path), or for a directory
        the newest of it and its files."""
        if isinstance(entry, os.DirEntry):  # its stat is already cached
            path, newest, is_dir = entry.path, entry.stat().st_mtime, entry.is_dir()
        else:
            path = os.fspath(entry)
            newest, is_dir = os.stat(path).st_mtime, os.path.isdir(path)
        if is_dir:
            try:
                with os.scandir(path) as children:
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
            if (
                kept
            ):  # warning: visible at the default log level — files were left behind
                self.logger.warning(f"Keeping temp artifacts after failure: {kept}")


class CachedArtifact(LoggingMixin):
    """Produce-once / reuse-forever artifact behind a content-addressed cache slot.

    The lifecycle every headless-conversion pipeline otherwise re-implements: hash the
    inputs into a stable *key*, hand back the cached entry when one exists, else produce it
    into a **scoped scratch** store and only then promote it into the cache slot. Built on
    :class:`TempArtifacts` — a ``"detached"`` store (``<name>_cache``) owns the cache
    (nothing can safely delete a payload another process may still read; stale entries are
    age-swept), a ``"scoped"`` one (``<name>``) the scratch (kept + logged on failure).
    Scratch ALWAYS lives under the temp root; only the promoted result lands in ``dir``.

    The scratch hop is load-bearing, not ceremony: producing straight into the cache slot
    lets a timeout-killed partial write poison every later run, and lets two concurrent
    producers of the same key interleave into one file. The promotion is one rename
    (a copy into a ``.part`` beside the slot first, where the slot is on another volume),
    so a slot is only ever empty or complete.

    A slot is a file (``extension``) or a whole directory (``directory=True``: a
    produce that writes several files, or a file whose name the producer decides).
    The slot API is public, so a caller never rebuilds the naming by hand:
    :meth:`slot` (where a key's entry is, there or not), :meth:`peek` (it, when
    complete), :meth:`touch` (mark it in use), :meth:`promote` (adopt an entry a
    batch produced itself).

    Example (a headless DCC conversion, cached on scene + template identity):
        >>> cache = CachedArtifact("maya_to_btk", extension=".fbx")
        >>> key = CachedArtifact.key(sorted(opts.items()), files=[src, template])
        >>> got = cache.get(key, lambda out: convert(src, out), sidecars=(".manifest.json",))
        >>> import_fbx(got.path)
        >>> if got.scratch:          # a miss produced it — clean up on success
        ...     got.scratch.cleanup()

    Example (a project-resident cache a saved scene references -- never stale):
        >>> bakes = CachedArtifact("bake", extension=".ma",
        ...                        dir=ws.cache_dir("bake"), max_age_days=None)

    Parameters:
        name: Prefix of the cache (``<name>_cache``) and scratch (``<name>``) stores;
            also the sweep scope of each.
        extension: A file slot's extension, e.g. ``".fbx"``. Required unless
            *directory*.
        directory: Each slot is a directory (no *extension*).
        dir: Where the cache slots live (default: the cache store's own folder
            under the temp root, :meth:`TempArtifacts.root`). Scratch is never here.
        max_age_days: Stale-sweep threshold for the cache store; ``None`` never
            ages (a project-resident cache).
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
        extension: Optional[str] = None,
        directory: bool = False,
        dir: Optional[str] = None,  # noqa: A002 - matches TempArtifacts' own param name
        max_age_days: Optional[float] = 7,
        log_level: str = "WARNING",
    ):
        super().__init__()
        if not name:
            raise ValueError("CachedArtifact requires a non-empty name.")
        if directory and extension is not None:
            raise ValueError("A directory slot takes no extension.")
        if not directory and extension is None:
            raise ValueError("A file slot needs an extension (or directory=True).")
        self.logger.setLevel(log_level)
        self.name = name
        self.extension = extension
        self.directory = directory
        self.dir = dir
        self.max_age_days = max_age_days
        self._store: Optional[TempArtifacts] = None

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

    # ------------------------------------------------------------------ the slots
    @property
    def cache_dir(self) -> str:
        """The folder the slots live in (nothing is created by asking)."""
        return self._cache().dir

    def slot(self, key: str) -> str:
        """The path *key*'s entry lives at -- there or not; nothing is created
        but the temp store's own folder.

        Parameters:
            key: A :meth:`key` tag (or any stable string).

        Returns:
            The slot's path: ``<cache dir>/<name>_cache_<key><extension>`` (a
            directory slot has no extension).
        """
        store = self._cache()
        store._ready()
        return store._entry(key, "" if self.directory else self.extension)

    def peek(self, key: str) -> Optional[str]:
        """*key*'s entry when the slot holds a complete one, else None.

        A file slot is complete when non-empty, a directory slot when it exists
        (it arrives by one rename). Reads only; :meth:`touch` marks it in use.
        """
        slot = self.slot(key)
        if self.directory:
            return slot if os.path.isdir(slot) else None
        try:
            return slot if os.path.isfile(slot) and os.path.getsize(slot) > 0 else None
        except OSError:
            return None

    def touch(self, key: str, sidecars: Sequence[str] = ()) -> bool:
        """Mark *key*'s entry in use: its age starts again.

        The age sweep goes by modification time, and a read writes nothing: an
        entry a scene references every day would be collected at its age.
        :meth:`get` touches on every hit.

        Parameters:
            key: The entry's key.
            sidecars: Suffixes of files beside the slot to refresh with it.

        Returns:
            True when the slot exists and was refreshed.
        """
        slot = self.slot(key)
        try:
            os.utime(slot, None)
        except OSError:
            return False
        for suffix in sidecars:
            try:
                os.utime(slot + suffix, None)
            except OSError:
                pass
        return True

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
            produce: Called with the scratch path it must write -- a file path, or
                an existing empty directory for a directory slot; its return value
                is ignored (the artifact's *existence* is the contract). An
                exception propagates with the partial output kept + logged for
                debugging.
            sidecars: Suffixes appended to the artifact path that ``produce`` may also
                write (e.g. ``".manifest.json"``). Promoted with the artifact; a slot's
                stale sidecar is removed when the new run produced none, so a sidecar can
                never outlive the artifact it describes.
            use_cache: When False, produce into scratch every time and skip the cache
                entirely (the result's ``path`` is then the scratch path).

        Returns:
            CachedArtifact.Result: ``(path, hit, scratch)``. A hit refreshes the
            entry's age (:meth:`touch`).
        """
        if use_cache:
            hit = self.peek(key)
            if hit is not None:
                self.touch(key, sidecars)
                self.logger.info(
                    f"Cache hit ({os.path.basename(hit)}) -- skipping production."
                )
                return self.Result(hit, True, None)

        scratch = TempArtifacts.scoped(self.name)
        out_path = (
            scratch.dir_path()
            if self.directory
            else scratch.path(extension=self.extension)
        )
        for suffix in sidecars:
            scratch.register(out_path + suffix)
        try:
            produce(out_path)
        except Exception:
            if os.path.exists(out_path):
                self.logger.warning(
                    f"Keeping partial artifact for debugging: {out_path}"
                )
            raise
        if not use_cache:
            return self.Result(out_path, False, scratch)
        return self.Result(
            self.promote(key, out_path, sidecars=sidecars), False, scratch
        )

    def promote(self, key: str, produced: str, *, sidecars: Sequence[str] = ()) -> str:
        """Move a finished artifact into *key*'s slot; return the slot.

        What :meth:`get` does on a miss, public for a producer that runs its own
        batch (a parallel encode pass) and only then hands each result to the
        cache. *produced* is a file, or a directory for a directory slot, and is
        moved (it is gone afterwards). An existing entry is replaced whole.

        Parameters:
            key: The slot's key.
            produced: The finished artifact.
            sidecars: As in :meth:`get`: files beside *produced* moved with it; a
                slot's sidecar the new artifact lacks is removed.

        Returns:
            The slot path.

        Raises:
            OSError: The artifact could not be moved (the slot is unchanged).
        """
        slot = self.slot(key)
        os.makedirs(os.path.dirname(slot), exist_ok=True)
        if self.directory:
            self._promote_dir(produced, slot)
        else:
            self._move_file(produced, slot)
        for suffix in sidecars:
            if os.path.isfile(produced + suffix):
                self._move_file(produced + suffix, slot + suffix)
            elif os.path.isfile(slot + suffix):
                os.remove(slot + suffix)  # stale sidecar from a partial promote
        return slot

    def _cache(self) -> TempArtifacts:
        """The cache store, built once: its first access sweeps it."""
        if self._store is None:
            self._store = TempArtifacts(
                f"{self.name}_cache",
                policy="detached",
                dir=self.dir,
                max_age_days=self.max_age_days,
            )
        return self._store

    @staticmethod
    def _same_volume(path: str, folder: str) -> bool:
        """Whether a rename can carry *path* into *folder*."""
        try:
            return os.stat(path).st_dev == os.stat(folder).st_dev
        except OSError:
            return True  # let the rename itself say why not

    @classmethod
    def _move_file(cls, src: str, dst: str) -> None:
        """*src* to *dst* atomically: a rename, or across volumes a copy into a
        ``.part`` beside *dst* promoted by one (:meth:`FileUtils.atomic_write`)."""
        from pythontk.file_utils._file_utils import FileUtils

        if cls._same_volume(src, os.path.dirname(dst)):
            FileUtils.replace_file(src, dst)
            return
        FileUtils.copy_file(src, os.path.dirname(dst), os.path.basename(dst))
        try:
            os.remove(src)
        except OSError:
            pass

    @classmethod
    def _promote_dir(cls, src: str, slot: str) -> None:
        """Put the directory *src* at *slot* by one rename, replacing what is there.

        Across volumes the tree is copied beside the slot first. A slot another
        producer filled meanwhile is complete too: ours is dropped for it.
        """
        staged = src
        if not cls._same_volume(src, os.path.dirname(slot)):
            staged = f"{slot}.part{secrets.token_hex(4)}"
            shutil.copytree(src, staged)
        aside = None
        if os.path.lexists(slot):
            aside = f"{slot}.old{secrets.token_hex(4)}"
            os.replace(slot, aside)
        try:
            os.replace(staged, slot)
        except OSError:
            if aside is not None and not os.path.lexists(slot):
                os.replace(aside, slot)
                aside = None
            if not os.path.isdir(slot):
                raise
            TempArtifacts._remove(staged)  # another producer's complete entry won
        if aside is not None:
            TempArtifacts._remove(aside)
        if staged != src:
            TempArtifacts._remove(src)


class ScratchTwins(LoggingMixin):
    """Per-source scratch twins of foreign files, discarded only while untouched.

    The "open a foreign scene as a new document" pattern the DCC Reference Managers
    share: a converted payload is COPIED to a deterministic, per-source scratch path
    the host opens (so a second Open resolves the same file and can close it), the
    copy's identity is stamped, and once the host has moved on the copy is discarded
    only if it is still that untouched copy — one the user saved into is real work
    and stays, its location logged. Built on :class:`TempArtifacts`: one tracked,
    age-swept ``<prefix>_<hash-of-source-path>/`` directory per source (same-named
    sources in different folders never share a twin), in the store's own folder
    under the temp root, the twin named ``<stem>_<ext><extension>`` so the title bar
    and any Save-As default carry the provenance (``scene.ma`` → ``scene_ma.blend``)
    and it can never shadow a sibling ``scene<extension>``.

    Example (a Blender panel opening a baked Maya scene):
        >>> twins = ScratchTwins("btk_opened", extension=".blend")
        >>> scratch = twins.create(source_ma, baked_blend)   # copy + stamp
        >>> open_scene(scratch)
        ...                                                  # later, on close / next open
        >>> twins.discard_except(current_file)               # untouched twins go, saved ones stay

    Parameters:
        prefix: Directory prefix; also the sweep scope (one family per host).
        extension: The twin's file extension — the HOST's native format.
        dir: Base directory (default: the store's own folder under the temp root).
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
        from pythontk.file_utils._file_utils import FileUtils

        try:
            FileUtils.replace_file(path, kept)
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
