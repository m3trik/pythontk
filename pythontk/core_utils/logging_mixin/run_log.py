# !/usr/bin/python
# coding=utf-8
"""``RunLog`` -- one bounded log file per source, holding its newest runs.

A tool that runs over the same thing again and again (an export over a scene,
a bake over a project) wants ONE log for that thing: a file per run piles up
wherever it lands, and one file per tool mixes every source and grows without
bound.  ``RunLog`` names the file after its source's path, brackets each run
with a header and a footer written straight to the file (so a run at a quiet
level still records that it ran and how it ended), trims the file to its
newest runs once it passes a size cap, and sweeps the logs a folder has not
written for a while.

Each step is its own classmethod, so a caller composes what it needs::

    path = os.path.join(folder, RunLog.file_name(scene_path))
    RunLog.sweep(folder, keep=[path])
    RunLog.trim(path)
    handler = RunLog.attach(logger, path, header="scene: hero.ma")
    ...  # the run logs through *logger*
    RunLog.detach(logger, handler, footer="Export succeeded")
"""

import datetime
import hashlib
import logging
import os
import re
import time
from typing import Iterable, Optional

from pythontk.core_utils.logging_mixin.logger_ext import LoggerExt, StripHtmlFormatter


class _RunLogHandler(logging.FileHandler):
    """A run's file handler: appends, remembers when the run began, and what
    :meth:`RunLog.attach` changed on the logger to capture more than it shows."""

    def __init__(self, path: str):
        super().__init__(path, mode="a", encoding=LoggerExt.FILE_ENCODING)
        self.started = time.time()
        #: ``(logger level, [(handler, level, pin)])`` to put back, or ``None``.
        self.restore = None


class RunLog:
    """One append-only log per source, trimmed to its newest runs."""

    #: Starts every run's header line; :meth:`trim` cuts the file at one, so
    #: a trimmed log always opens on a whole run.
    RUN_MARKER = "===== "
    #: Starts every run's footer line.
    END_MARKER = "----- "
    EXTENSION = ".log"
    #: Record layout: plain text, a timestamp and the level on every line.
    FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    #: Past this size :meth:`trim` keeps the newest half -- hysteresis, so a
    #: log at the cap is not rewritten on every run.
    MAX_BYTES = 1024 * 1024
    #: :meth:`sweep` removes logs not written for this many days.
    MAX_AGE_DAYS = 30

    @classmethod
    def file_name(cls, source: Optional[str], fallback: str = "untitled") -> str:
        """The log's file name for *source*: ``<stem>_<hash>.log``.

        The stem keeps the file findable by eye; the hash of the full path
        keeps two same-named sources apart (every shot's ``layout.ma``).  A
        renamed or moved source gets a new name, and its old log ages out
        through :meth:`sweep`.

        Parameters:
            source: The path the runs work on; blank for one never saved.
            fallback: The name a blank *source* logs under, shared by every
                unsaved source.

        Returns:
            str: A file name (no folder).
        """
        if not source:
            return f"{fallback}{cls.EXTENSION}"
        stem = os.path.splitext(os.path.basename(source))[0]
        label = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:60] or fallback
        key = os.path.normcase(os.path.abspath(source)).encode("utf-8")
        return f"{label}_{hashlib.sha256(key).hexdigest()[:8]}{cls.EXTENSION}"

    @classmethod
    def attach(
        cls,
        logger: logging.Logger,
        path: str,
        header: Optional[str] = None,
        formatter: Optional[logging.Formatter] = None,
        capture: Optional[int] = None,
    ) -> logging.FileHandler:
        """Open a run in *path* and tee *logger* into it.

        The header -- the time, then *header* -- is written straight to the
        file, past every level, and the folder is created when missing.

        Parameters:
            logger: The logger the run logs through.
            path: The log file; appended to.
            header: What the run is (its source, its target); optional.
            formatter: The record format; default :attr:`FORMAT`, markup
                stripped.
            capture: The level the FILE records from, when it is below what
                *logger* passes: the logger is lowered for the run and every
                other handler that follows it is held at the level it showed,
                so the file gains the detail and nothing else does.
                :meth:`detach` puts both back. ``None`` records what the
                logger passes.

        Returns:
            logging.FileHandler: The handler to pass to :meth:`detach`.
        """
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        handler = _RunLogHandler(path)
        try:
            handler.setFormatter(formatter or StripHtmlFormatter(cls.FORMAT))
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"{stamp} | {header}" if header else stamp
            handler.stream.write(f"{cls.RUN_MARKER}{line}\n")
            handler.flush()
        except Exception:
            handler.close()  # never leave the file open (locked on Windows)
            raise
        shown = logger.getEffectiveLevel()
        if capture is not None and capture < shown:
            # Held by PINNING (``_pinned_level``, LoggerExt's convention): a
            # patched logger's setLevel syncs every unpinned handler to it, so
            # a level set on the panel's handler alone was carried down too.
            # A handler already pinned keeps its own level: it filters for
            # itself (a DEBUG ring buffer stays DEBUG), and no sync reaches it.
            held = []
            for other in logger.handlers:
                if getattr(other, "_pinned_level", False):
                    continue
                held.append((other, other.level, getattr(other, "_pinned_level", None)))
                if other.level < shown:
                    other.setLevel(shown)
                other._pinned_level = True
            handler.restore = (logger.level, held)
            handler.setLevel(capture)
            handler._pinned_level = True
            logger.setLevel(capture)
        logger.addHandler(handler)
        return handler

    @classmethod
    def detach(
        cls,
        logger: logging.Logger,
        handler: logging.FileHandler,
        footer: Optional[str] = None,
    ) -> None:
        """Close the run :meth:`attach` opened.

        Detached first -- the logger and its other handlers back at the levels
        a *capture* changed -- so no record lands after the footer, and written under
        the handler's lock, so a record another thread was mid-way through
        emitting finishes first; the footer -- *footer* and the run's
        duration -- is written past every level.

        Parameters:
            logger: The logger *handler* was attached to.
            handler: What :meth:`attach` returned.
            footer: How the run ended; ``None`` writes no footer.
        """
        logger.removeHandler(handler)
        restore = getattr(handler, "restore", None)
        if restore:
            level, held = restore
            logger.setLevel(level)  # before the pins lift: nothing re-syncs
            for other, other_level, pinned in held:
                other.setLevel(other_level)
                if pinned is None:
                    other.__dict__.pop("_pinned_level", None)
                else:
                    other._pinned_level = pinned
        try:
            if footer is not None and handler.stream is not None:
                started = getattr(handler, "started", None)
                took = f" ({time.time() - started:.1f} s)" if started else ""
                handler.acquire()
                try:
                    handler.stream.write(f"{cls.END_MARKER}{footer}{took}\n")
                    handler.flush()
                finally:
                    handler.release()
        finally:
            handler.close()

    @classmethod
    def trim(cls, path: str, max_bytes: Optional[int] = None) -> bool:
        """Cut *path* to its newest whole runs once it is past *max_bytes*.

        Keeps the last half of the cap from the first run header in it.
        Best-effort: a file another process holds stays as it is.

        Parameters:
            path: The log file.
            max_bytes: The cap; default :attr:`MAX_BYTES`.

        Returns:
            bool: True when the file was cut.
        """
        cap = cls.MAX_BYTES if max_bytes is None else max_bytes
        try:
            if os.path.getsize(path) <= cap:
                return False
            with open(path, "rb") as fh:
                fh.seek(-max(cap // 2, 1), os.SEEK_END)
                tail = fh.read()
            cut = tail.find(b"\n" + cls.RUN_MARKER.encode("utf-8"))
            tail = tail[cut + 1 :] if cut >= 0 else tail[tail.find(b"\n") + 1 :]
            with open(path, "wb") as fh:
                fh.write(tail)
            return True
        except OSError:
            return False

    @classmethod
    def sweep(
        cls,
        folder: str,
        max_age_days: Optional[float] = None,
        keep: Iterable[str] = (),
    ) -> int:
        """Remove the logs in *folder* not written for *max_age_days*.

        Only ``*.log`` files, and only in a folder that holds nothing but
        run logs: a log nobody re-ran in that long belongs to a source that
        was renamed, moved or finished.  Best-effort, per file.

        Parameters:
            folder: The run logs' own folder.
            max_age_days: The age; default :attr:`MAX_AGE_DAYS`.
            keep: Paths never removed (the log about to be written).

        Returns:
            int: How many were removed.
        """
        days = cls.MAX_AGE_DAYS if max_age_days is None else max_age_days
        if not days or not os.path.isdir(folder):
            return 0
        cutoff = time.time() - days * 86400
        kept = {os.path.normcase(os.path.abspath(p)) for p in keep}
        removed = 0
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if not name.endswith(cls.EXTENSION):
                continue
            if os.path.normcase(os.path.abspath(path)) in kept:
                continue
            try:
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except OSError:
                continue
        return removed
