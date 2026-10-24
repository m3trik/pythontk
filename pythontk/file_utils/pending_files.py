# !/usr/bin/python
# coding=utf-8
"""Files written for the open scene, held out of their folders until it is saved."""
import os
import shutil
from typing import Callable, Dict, Iterable, List, Optional

from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.temp_artifacts import TempArtifacts


class PendingFiles(LoggingMixin):
    """The files a tool wrote for the open scene, kept where the scene file on
    disk cannot see them until that scene is saved.

    A write over a file the saved scene reads, by a session that is then not
    saved, leaves that scene reading bytes laid out for records it never got:
    an unsaved Combine wrote one atlas over a room's two, and the next session
    opened with one map missing and three objects lit from the wrong cells
    (2026-10-23). So a tool writes each file into a session folder instead,
    under its own name -- one folder per home folder (:meth:`stage`) -- and
    records that folder as where the scene reads it. A save of the open scene
    moves every file it still reads home (:meth:`commit`, through the tools'
    :meth:`on_save` hooks); a new or opened scene drops them all
    (:meth:`discard`), and so does the end of the session -- a crashed one's
    by the store's sweep. Nothing reaches the project, or a synced folder,
    before the scene that reads it is saved, and an unsaved change leaves
    nothing behind.

    One ledger per process (:meth:`session`): a DCC has one open scene. Its
    save hook calls :meth:`saving`, its open and new hooks :meth:`discard`.
    """

    #: The temp store's prefix (``TempArtifacts``, session policy).
    PREFIX = "pending_files"

    _session: Optional["PendingFiles"] = None

    def __init__(self, prefix: str = PREFIX):
        self._prefix = prefix
        self._store: Optional[TempArtifacts] = None
        self._homes: Dict[str, str] = {}  # session folder key -> home folder
        self._folders: Dict[str, str] = {}  # home folder key -> session folder
        self._retiring: List[str] = []
        self._hooks: Dict[str, Callable[[str, bool], None]] = {}
        self._failed: set = set()  # files a save's commit could not move

    @classmethod
    def session(cls) -> "PendingFiles":
        """The process's ledger, for its open scene."""
        if cls._session is None:
            cls._session = cls()
        return cls._session

    def __bool__(self) -> bool:
        """Whether anything waits for the save: a file staged, or retiring."""
        return bool(self.staged() or self._retiring)

    @staticmethod
    def _key(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    # ------------------------------------------------------------ the session
    def folder(self, home: str) -> str:
        """The session folder standing in for the folder *home* -- made on
        first ask, the same one after. A session folder is its own."""
        key = self._key(home)
        if key in self._homes:
            return os.path.abspath(home)
        folder = self._folders.get(key)
        if folder is None or not os.path.isdir(folder):
            if self._store is None:
                self._store = TempArtifacts(self._prefix, policy="session")
            folder = self._store.dir_path()
            self._folders[key] = folder
            self._homes[self._key(folder)] = os.path.abspath(home)
        return folder

    def stage(self, path: str) -> str:
        """Where to write the file *path* names: its name, in the session
        folder for its folder (:meth:`folder`). A session path is its own."""
        if self.holds(path):
            return os.path.abspath(path)
        return os.path.join(
            self.folder(os.path.dirname(os.path.abspath(path))), os.path.basename(path)
        )

    def holds(self, path: str) -> bool:
        """Whether *path* is a session folder or a file in one."""
        key = self._key(path)
        return key in self._homes or os.path.dirname(key) in self._homes

    def home(self, path: str) -> str:
        """Where a session folder, or a file in one, belongs: the folder it
        stands in for, joined with the name. Any other path is its own."""
        key = self._key(path)
        if key in self._homes:
            return self._homes[key]
        folder = self._homes.get(os.path.dirname(key))
        return os.path.join(folder, os.path.basename(path)) if folder else path

    def staged(self) -> List[str]:
        """Every file in the session folders, sorted."""
        out: List[str] = []
        for folder in self._folders.values():
            try:
                names = os.listdir(folder)
            except OSError:
                continue
            out.extend(
                os.path.join(folder, n)
                for n in names
                if os.path.isfile(os.path.join(folder, n))
            )
        return sorted(out)

    def retire(self, paths: Iterable[str]) -> None:
        """Hold *paths* -- home files a write superseded -- for the save to
        set aside: until then the scene file on disk still reads them."""
        known = {self._key(p) for p in self._retiring}
        for path in paths:
            if self._key(path) not in known:
                known.add(self._key(path))
                self._retiring.append(os.path.abspath(path))

    def take_retiring(self) -> List[str]:
        """The files :meth:`retire` held, handed over once."""
        out, self._retiring = self._retiring, []
        return out

    # --------------------------------------------------------------- the save
    def on_save(self, key: str, hook: Callable[[str, bool], None]) -> None:
        """Run *hook(target, fork)* at every save of the open scene, before
        the file is written (:meth:`saving`): the tool's half of the commit --
        which of its staged files the scene still reads, and the records that
        name their folder. *key* names the tool; a second hook under it (a
        reloaded module's) replaces the first."""
        self._hooks[key] = hook

    def saving(self, target: str, saved_from: str = "") -> None:
        """The open scene is about to be written to *target*: each tool's
        :meth:`on_save` hook commits its files. *saved_from* is the file the
        scene was opened from or last saved to, ``""`` before its first save;
        another one still on disk is a Save As -- a *fork*: the file it came
        from still reads every home file as it is, so a hook must move nothing
        over one and retire nothing. One that is gone (the scene was renamed
        or moved) reads nothing: no fork (``FileDependencies.written_here``).
        Never raises: a save never fails on this."""
        if not self:
            return
        fork = (
            bool(saved_from)
            and self._key(saved_from) != self._key(target)
            and os.path.isfile(saved_from)
        )
        self._failed = set()
        hook_failed = False
        for key, hook in list(self._hooks.items()):
            try:
                hook(target, fork)
            except Exception:  # noqa: BLE001 -- see above
                hook_failed = True
                self.logger.warning(
                    "Pending %s files were not committed for the save; every "
                    "staged file waits for the next save.",
                    key,
                    exc_info=True,
                )
        if hook_failed:
            # Which staged files the failed hook's scene still reads is
            # unknown -- no file names its tool -- so none is dropped.
            self._failed = set()
            return
        # Once every hook has moved its own: the rest nothing reads (an undone
        # write), but a file a hook could not move, which the scene still does.
        for staged in self.staged():
            if self._key(staged) not in self._failed:
                try:
                    os.remove(staged)
                except OSError:
                    pass
        self._failed = set()

    def commit(
        self, keep: Iterable[str], rename: Optional[Callable[[str], str]] = None
    ) -> Dict[str, str]:
        """Move each session file in *keep* home -- a hook's half of
        :meth:`saving`, which then drops whatever no hook kept.

        Copied in whole (``FileUtils.atomic_write``: a reader of the home file
        sees the old one or the new, never part; across drives, where a
        rename cannot go). *rename* may give a file another home path -- a
        fork's, never over a file on disk. A file that cannot be written
        (held open) stays staged and is logged: the scene still reads it from
        here, and the next save tries again -- the session store is swept when
        the process ends, so one no save moves is lost then.

        Returns:
            ``{session path: home path}`` for each file moved.
        """
        kept = {self._key(p) for p in keep}
        moved: Dict[str, str] = {}
        for staged in self.staged():
            if self._key(staged) not in kept:
                continue
            home = self.home(staged)
            if rename is not None:
                home = rename(home)
            try:
                os.makedirs(os.path.dirname(home), exist_ok=True)
                FileUtils.atomic_write(
                    home, lambda part, src=staged: shutil.copyfile(src, part)
                )
            except OSError as error:
                self.logger.error(
                    "%s could not be written to %s (%s); the scene reads it from "
                    "%s until a save can, and it is lost when this session "
                    "ends if none does.",
                    os.path.basename(staged),
                    os.path.dirname(home),
                    error,
                    os.path.dirname(staged),
                )
                self._failed.add(self._key(staged))
                continue
            moved[staged] = home
            try:
                os.remove(staged)
            except OSError:
                pass
        return moved

    def discard(self) -> None:
        """Drop every staged file and retirement: the scene they were for is
        gone unsaved (a new or opened scene). The hooks stay."""
        if self._store is not None:
            self._store.cleanup(force=True)
        self._store = None
        self._homes.clear()
        self._folders.clear()
        self._retiring = []
