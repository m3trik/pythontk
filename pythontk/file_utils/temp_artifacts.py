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
import os
import tempfile
import time
from typing import Callable, List, Optional

from pythontk.core_utils.logging_mixin import LoggingMixin


class TempArtifacts(LoggingMixin):
    """Allocate and lifecycle-manage ``<prefix>_*`` temp files in one directory.

    Example (scoped — a synchronous convert-then-import round-trip):
        >>> with TempArtifacts("maya_to_btk", policy="scoped") as tmp:
        ...     fbx = tmp.path(extension=".fbx")
        ...     convert(src, fbx)   # on exception the fbx is kept + logged
        ...     import_fbx(fbx)     # clean exit removes it

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
        self._last_ns = 0

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
        if name is not None:
            tag = name
        else:  # monotonic per instance — Windows' time_ns is too coarse to be unique
            ns = max(time.time_ns(), self._last_ns + 1)
            self._last_ns = ns
            tag = f"{ns:x}"
        return self.register(os.path.join(self.dir, f"{self.prefix}_{tag}{extension}"))

    def register(self, path: str) -> str:
        """Adopt *path* (e.g. a side artifact a tool wrote) into this lifecycle."""
        if path not in self._tracked:
            self._tracked.append(path)
        if self.policy == "session" and not self._atexit_registered:
            atexit.register(self.cleanup, force=True)
            self._atexit_registered = True
        return path

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
        existing = [p for p in self._tracked if os.path.isfile(p)]
        if existing and self.on_cleanup is not None:
            try:
                self.on_cleanup(list(existing))
            except Exception as e:  # noqa: BLE001 - a callback must never block removal
                self.logger.warning(f"on_cleanup callback failed: {e}")
        removed = []
        for p in existing:
            try:
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
        errors are swallowed (a locked file just waits for the next sweep).
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
                    if entry.is_file() and entry.stat().st_mtime < cutoff:
                        os.remove(entry.path)
                        removed.append(entry.path)
                except OSError:
                    continue
        return removed

    # ------------------------------------------------------------------ context manager
    def __enter__(self) -> "TempArtifacts":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.cleanup(force=True)
        else:  # keep everything for debugging; say where it is
            kept = [p for p in self._tracked if os.path.isfile(p)]
            if kept:  # warning: visible at the default log level — files were left behind
                self.logger.warning(f"Keeping temp artifacts after failure: {kept}")


__all__ = ["TempArtifacts"]
