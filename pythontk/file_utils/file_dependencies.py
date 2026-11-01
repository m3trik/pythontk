# !/usr/bin/python
# coding=utf-8
"""File dependencies -- files a record names by *name*, found where they are NOW.

Plenty of scene data names a file the way a file path cannot survive: as a file
name plus the folder it was written from. A baked-lightmap marker does, an audio
event does, a baked shadow silhouette does. That folder is history, not a
contract -- a reorganised project, a scene migrated to another module, a
teammate's machine with the project on another drive -- so every tool that reads
such a record ends up needing the same four answers:

* which file each name means NOW (:meth:`FileDependencies.resolve`): its own
  recorded folder, else the one file of that name the search folders and the
  walk hold (:meth:`FileDependencies.candidates`). Several different files of
  one name are AMBIGUOUS and come back with their candidates
  (:meth:`FileDependencies.choose`), never a silent first or newest pick;
* where a consumer that can only join a name against a list of folders should
  look first (:meth:`FileDependencies.search_dirs`);
* which names a new batch of outputs must not take, because something else
  still reads them (:meth:`FileDependencies.claims`, the input to
  :meth:`FileUtils.unique_path`);
* which files a batch's owners stopped reading, now that nothing reads them
  -- set aside, never deleted (:meth:`FileDependencies.remove_superseded`,
  :meth:`FileDependencies.set_aside`);
* how to gather the files into one folder (:meth:`FileDependencies.relocate`,
  over :class:`~pythontk.file_utils.file_collect.FileCollect`'s plan);
* which names the files under a folder still reference, read from their bytes
  without the app that wrote them (:meth:`FileDependencies.scan_references`,
  :meth:`FileDependencies.reference_bytes`) -- what a purge of derived files
  (``BakeCache.purge``) keeps.

They take plain values -- ``(owner, name, recorded folder)`` references -- and
callables where a host does something its own way (how a stored folder resolves,
how a tree is searched, how a file is copied), so a DCC reads and writes its own
records and hands the rules the data.
"""

import errno
import fnmatch
import os
import posixpath
import re
import shutil
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Pattern,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.tiled_path import TiledPath


class FileDependencies(LoggingMixin):
    """The rules over files that records name by name; see the module docstring.

    A dependency record, as :meth:`resolve` returns it::

        {"name": file name, "dir": recorded folder, "owners": [owner, ...],
         "path": absolute path or None, "found_by": "hint" | "search" | None,
         "note": "" | why an unresolved file stayed unresolved,
         "candidates": [] | the different same-named files that made it ambiguous}
    """

    #: How a dependency was located. ``"hint"``: its own recorded folder still
    #: holds it. ``"search"``: found elsewhere (the search folders, then the
    #: walk), so the recorded folder is stale and worth healing. ``None``: on
    #: disk nowhere.
    FOUND_BY_HINT: str = "hint"
    FOUND_BY_SEARCH: str = "search"

    #: The folder :meth:`set_aside` used to move a file into, beside it, where
    #: its volume keeps no trash (before 2026-10-13: now the
    #: :attr:`SUPERSEDED_STORE`). :meth:`find_files` still never walks one: a
    #: project may hold one from then, and what was set aside is out of
    #: circulation, as a file in the Recycle Bin is.
    SUPERSEDED_DIR: str = "_superseded"

    #: The ``TempArtifacts`` store :meth:`set_aside` moves a file into where its
    #: volume keeps no trash, and the days it stays there to be restored.
    SUPERSEDED_STORE: str = "superseded"
    SUPERSEDED_DAYS: float = 30
    #: Seconds within which a repeat set-aside skips re-sweeping the store.
    _SUPERSEDED_SWEEP_INTERVAL: float = 600.0

    #: Folder names a search for files by name never enters (:meth:`walk`),
    #: compared without case: sync-client caches, the OS's trash and volume
    #: folders, version control, dependency and bytecode caches, and
    #: :attr:`SUPERSEDED_DIR`. Each holds stale same-named copies -- an older
    #: texture, a map a re-bake set aside -- that a walk would bind as the
    #: current file. A freedesktop ``.Trash-<uid>`` is skipped by its prefix.
    WALK_SKIP_DIRS: FrozenSet[str] = frozenset(
        {
            ".dropbox.cache",
            ".dropbox",
            "$recycle.bin",
            "system volume information",
            ".trash",
            ".trashes",
            ".git",
            ".svn",
            ".hg",
            "node_modules",
            "__pycache__",
            SUPERSEDED_DIR,
        }
    )

    @staticmethod
    def file_name(path: Any) -> str:
        """The file name *path* ends in, whichever system spelled it: a record
        written on Windows names ``C:\\tex\\wood.png``, whose POSIX
        ``os.path.basename`` is the whole string -- a name no file carries.
        ``""`` for none."""
        return posixpath.basename(str(path or "").replace("\\", "/"))

    @staticmethod
    def spelled(path: str) -> str:
        """*path* with forward slashes -- except a UNC path's leading ``\\\\``:
        the spelling every path this class and :class:`FileCollect` hand back
        takes.

        ``//server/share`` is a valid spelling to Windows, but Blender reads a
        leading ``//`` as RELATIVE TO THE .blend, so a UNC folder spelled that
        way resolved under the project instead of on the share.
        """
        path = str(path or "")
        if path.startswith("\\\\"):
            return "\\\\" + path[2:].replace("\\", "/")
        return path.replace("\\", "/")

    @staticmethod
    def claims(refs: Iterable[Sequence[str]]) -> Dict[str, FrozenSet[str]]:
        """``{file name: owners}`` -- which owners read which file name.

        What :meth:`FileUtils.unique_path` takes as ``claims``, with the owners a
        write replaces as its ``owners``: a name only they read stays theirs to
        overwrite, and a name anything else reads is not. Names compare
        without case -- the filesystems these records live on do.

        Parameters:
            refs: ``(owner, file name[, ...])`` references; a name may be a path.

        Returns:
            ``{lower-case file name: frozenset of owners}``.
        """
        readers: Dict[str, set] = {}
        for owner, name, *_rest in refs:
            key = FileDependencies.file_name(name).lower()
            if key:
                readers.setdefault(key, set()).add(owner)
        return {key: frozenset(owners) for key, owners in readers.items()}

    @staticmethod
    def strangers(
        folder: str,
        stem: str,
        ext: str,
        claims: Mapping[str, Any],
        wrote: Callable[[str], bool],
    ) -> Set[str]:
        """The files in *folder* a write naming ``<stem>`` must not replace:
        ``<stem><ext>`` / ``<stem>_<k><ext>`` on disk (any case) that nothing
        the host knows reads -- no key of *claims* (:meth:`claims`) -- and the
        host did not write itself (*wrote*, handed the lower-case name).

        :meth:`claims` holds only one scene's readers: two scenes in one
        project whose names agree write the same files, and the second's write
        would replace the first's. A file the host wrote and no longer reads
        (a revert) stays its own to replace.

        Parameters:
            folder: The folder the write lands in.
            stem: The name the write takes, without its ``_<k>`` or extension.
            ext: The extension, with its dot.
            claims: ``{lower-case file name: readers}``.
            wrote: Whether the host wrote a file, by its lower-case name.

        Returns:
            Absolute paths; empty when *folder* cannot be listed.
        """
        try:
            names = os.listdir(folder)
        except OSError:
            return set()
        pattern = re.compile(rf"{re.escape(stem)}(_\d+)?{re.escape(ext)}", re.IGNORECASE)
        return {
            os.path.abspath(os.path.join(folder, name))
            for name in names
            if pattern.fullmatch(name)
            and name.lower() not in claims
            and not wrote(name.lower())
        }

    @classmethod
    def remove_superseded(
        cls, before: Iterable[str], after: Iterable[Sequence[Any]]
    ) -> List[str]:
        """Set aside the files of *before* that nothing in *after* reads; return those.

        The other half of :meth:`claims`: a write that gives its owners new
        files -- another folder, another name, several maps folded into one
        -- leaves the old ones behind, read by nobody, and a same-named
        leftover is what a reader that joins names against folders then finds.
        *before* is the files the owners read before the write, narrowed by
        the host to the ones it wrote itself: which files are a scene's own
        is the host's to know, never this rule's. *after* is every reference
        the host holds once the write is recorded, each with the file its
        owner finds NOW.

        A file stays when a reference reads it -- by the filesystem's answer,
        not the spelling's: a junction, a ``subst`` or mapped drive names one
        file two ways (:meth:`FileUtils.is_same_file`) -- and when a
        reference naming a file of the same name found its file nowhere, or
        names it by a relative spelling only the host could place: a search
        the host did not run -- a walk of its texture tree -- could still
        land on this one. Names compare without case. Only an absolute path
        ever goes (a relative one would resolve against the process CWD),
        never a folder.

        Nothing is deleted: each file goes to the Recycle Bin, or into the
        :attr:`SUPERSEDED_STORE` temp store (:meth:`set_aside`). A
        host sees only its own scene's references, so a file another scene
        file still reads -- a Save As copy's source, a copy made in Explorer
        -- can look unread here; set aside, it is one restore away. A file
        already gone is skipped; one that cannot be moved (held open) is
        logged and left out of the result.

        Parameters:
            before: Absolute paths the owners read before the write.
            after: ``(owner, file name, path)`` references after it; *path*
                is ``None`` or ``""`` for a file found nowhere.

        Returns:
            The files set aside, as *before* spelled them, in its order.
        """
        removed: List[str] = []
        for path in cls.superseded(before, after):
            try:
                cls.set_aside(path)
            except FileNotFoundError:
                continue
            except OSError as error:
                cls.logger.warning(
                    "Could not set aside %s, which nothing reads any more (%s); "
                    "move or delete it by hand.",
                    path,
                    error,
                )
                continue
            removed.append(path)
        return removed

    @classmethod
    def superseded(
        cls, before: Iterable[str], after: Iterable[Sequence[Any]]
    ) -> List[str]:
        """The files of *before* that nothing in *after* reads -- what
        :meth:`remove_superseded` sets aside, by its rule, touching nothing:
        for a host that sets them aside later (``PendingFiles.retire``).

        Returns:
            Those files, as *before* spelled them, in its order.
        """

        def key(path: str) -> str:
            return os.path.normcase(os.path.abspath(path))

        read: Dict[str, str] = {}
        lost = set()
        for _owner, name, path, *_rest in after:
            path = str(path or "")
            if FileUtils.is_rooted_path(path):
                read.setdefault(key(path), path)
            else:
                lost.add(cls.file_name(name or path).lower())
        identities: Optional[set] = None
        out: List[str] = []
        seen: set = set()
        for path in before:
            path = str(path or "")
            if not FileUtils.is_rooted_path(path):
                continue
            spelled = key(path)
            if spelled in seen:
                continue
            seen.add(spelled)
            if spelled in read or os.path.basename(path).lower() in lost:
                continue
            if os.path.isdir(path) or not os.path.exists(path):
                continue
            if identities is None:  # one stat per file read, and only if needed
                identities = {cls._identity(p) for p in read.values()} - {None}
            if cls._identity(path) in identities:
                continue
            out.append(path)
        return out

    @classmethod
    def set_aside(cls, path: str) -> str:
        """Take the file *path* out of its folder without destroying it.

        To the platform's trash (:meth:`FileUtils.move_to_trash`: the Recycle
        Bin), else -- a network share, a removable drive, a bin that would
        delete it at once -- under its own name into a fresh folder of the
        :attr:`SUPERSEDED_STORE` temp store, kept :attr:`SUPERSEDED_DAYS`
        days. Either way it keeps its content and is one restore away, and
        neither place is in the project a search walks -- nothing is left
        behind in the user's folders (a ``_superseded`` folder beside the maps
        was, until 2026-10-13).

        The file keeps its modification time, so the store's age clock starts
        at its last WRITE, not the set-aside: "30 days" is pythontk's own
        sweep guarantee (:class:`TempArtifacts`, the store's first allocation
        in a process and the daily :meth:`TempArtifacts.sweep_all`), not the
        Recycle Bin's or the OS temp cleaner's. Only a cross-device rename
        (temp on another volume) falls back to a copy; a copy cut short is
        removed with its folder.

        Parameters:
            path: The file to set aside.

        Returns:
            Where it is now: inside the trash (``""`` when Windows could not
            say where), or the temp store.

        Raises:
            FileNotFoundError: *path* does not exist.
            OSError: It could not be moved (held open); it is where it was.
        """
        trashed = FileUtils.move_to_trash(path)
        if trashed is not None:
            return trashed
        if not os.path.isfile(path):
            raise FileNotFoundError(2, "No such file", path)
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        folder = TempArtifacts(
            cls.SUPERSEDED_STORE,
            max_age_days=cls.SUPERSEDED_DAYS,
            # A batch builds one store per file: sweep the store once, not N
            # times. The first set-aside in a process still sweeps it, and so
            # does the daily ``sweep_all`` (its marker declares the 30 days).
            sweep_interval=cls._SUPERSEDED_SWEEP_INTERVAL,
        ).dir_path()
        target = os.path.join(folder, os.path.basename(path))
        try:
            os.rename(path, target)  # same volume: a move, nothing copied
        except OSError as e:
            if not cls._cross_device(e):
                cls._remove_empty_folder(folder)
                raise  # held open, denied: the file is where it was
            # Another volume (a share has no trash, temp is local): copy, then
            # take the original away -- and when it is held open, take the
            # copy back so the file is where it was, once.
            try:
                shutil.copy2(path, target)
            except BaseException:
                cls._discard_partial(target, folder)
                raise
            try:
                os.remove(path)
            except OSError:
                cls._discard_partial(target, folder)
                raise
        cls.logger.info(
            "No trash on %s's volume: set it aside in %s.",
            os.path.basename(path),
            folder,
        )
        return cls.spelled(target)

    @staticmethod
    def _cross_device(error: OSError) -> bool:
        """Whether a rename failed only because its ends are on two volumes
        (``EXDEV``; Windows' ``ERROR_NOT_SAME_DEVICE``, 17)."""
        return error.errno == errno.EXDEV or getattr(error, "winerror", None) == 17

    @classmethod
    def _discard_partial(cls, target: str, folder: str) -> None:
        """Remove a set-aside copy cut short, and its folder once empty."""
        try:
            os.remove(target)
        except OSError:
            pass
        cls._remove_empty_folder(folder)

    @staticmethod
    def _remove_empty_folder(folder: str) -> None:
        try:
            os.rmdir(folder)  # only ever an empty one
        except OSError:
            pass

    @staticmethod
    def _identity(path: str) -> Optional[Tuple[int, int]]:
        """``(device, file id)`` of *path* -- which file a spelling names, the
        test ``os.path.samefile`` makes -- or ``None`` when it cannot be read."""
        try:
            stat = os.stat(path)
        except (OSError, ValueError):
            return None
        return (stat.st_dev, stat.st_ino)

    @staticmethod
    def written_here(writer: Optional[str], scene: str, base: Optional[str]) -> bool:
        """Whether a file whose writer record says *writer* is *scene*'s own to delete.

        What narrows :meth:`remove_superseded`'s *before* to a host's own
        files. It is when *writer* names *scene*; when it is ``""`` -- written
        while unsaved -- and the scene is STILL unsaved: once saved it can be
        copied, a Save As writes the same ``""`` into the copy, and nothing
        tells the two files apart; and when the file it names is gone: the
        scene was renamed or moved, and nothing can open the old one to read
        the file (:meth:`writer_gone`). Another scene file that is still there
        (the source of a Save As copy) reads what it wrote; ``None`` -- no
        record -- a relative entry with no *base* to read it from, and one on
        a drive this machine cannot reach are nobody's to delete.

        Parameters:
            writer: The record's entry (a :meth:`FileUtils.portable_path`
                spelling), ``""``, or ``None`` for none.
            scene: The open scene's file, ``""`` while unsaved.
            base: The scene's own project, which *writer* is spelled from.

        Returns:
            bool: True when the file may be deleted as the scene's own.
        """
        if writer is None:
            return False
        if not writer:
            return not scene
        written = FileUtils.resolve_portable_path(writer, base)
        if (
            scene
            and FileUtils.is_rooted_path(written)
            and os.path.normcase(os.path.abspath(scene))
            == os.path.normcase(os.path.abspath(written))
        ):
            return True
        return FileDependencies.writer_gone(writer, base)

    @staticmethod
    def writer_gone(writer: Optional[str], base: Optional[str]) -> bool:
        """Whether *writer* -- a writer record's entry -- names a scene file
        that is gone: renamed or moved, so nothing can open it again.

        Not gone: ``""`` and ``None`` (no file named), a relative entry with
        no *base* to read it from, and a file on a drive or share this
        machine cannot reach -- offline is not renamed, and the scene there
        still reads what it wrote once it is back.

        Parameters:
            writer: The record's entry, spelled from *base*.
            base: The project *writer* is spelled from.

        Returns:
            bool: True when no file is where *writer* names, on a reachable drive.
        """
        if not writer:
            return False
        written = FileUtils.resolve_portable_path(writer, base)
        if not FileUtils.is_rooted_path(written):
            return False  # unresolvable, which is not gone
        written = os.path.abspath(written)
        drive = os.path.splitdrive(written)[0]
        if drive and not os.path.isdir(drive + os.sep):
            return False  # unreachable, which is not gone
        return not os.path.isfile(written)

    @staticmethod
    def find_files(names: Iterable[str], root: str) -> List[str]:
        """Every file under *root*, recursively, whose name is one of *names*.

        Names compare without case. A name carrying a tile / frame token
        (``rock.<UDIM>.png``, :class:`TiledPath`) matches each file of its
        set, as strictly as :meth:`TiledPath.tiles` globs one. The default
        search :meth:`resolve`, :meth:`candidates` and :meth:`relocate` walk
        with; a host with a texture walk of its own passes that instead,
        walking with :meth:`walk` too. Links are not followed, and no
        :attr:`WALK_SKIP_DIRS` folder is entered: a file set aside in
        ``_superseded`` keeps its name, and a walk that found it would bind
        it again.

        Parameters:
            names: File names (a path's base name is used).
            root: The folder to walk.

        Returns:
            The matching paths, in walk order.
        """
        wanted: Set[str] = set()
        patterns: List[str] = []
        for name in names:
            base = FileDependencies.file_name(name).lower()
            if TiledPath.has_token(base):
                patterns.append(TiledPath.wildcard(base, None).lower())
            elif base:
                wanted.add(base)
        hits: List[str] = []
        for folder, _dirs, files in FileDependencies.walk(root):
            for name in files:
                low = name.lower()
                if low in wanted or any(fnmatch.fnmatchcase(low, p) for p in patterns):
                    hits.append(os.path.join(folder, name))
        return hits

    @classmethod
    def walk(cls, root: str) -> Iterator[Tuple[str, List[str], List[str]]]:
        """``os.walk(root)``, top-down, that never enters a folder of stale
        copies (:attr:`WALK_SKIP_DIRS`).

        The one pruning every search for files by name shares: :meth:`find_files`,
        and each DCC's texture walks (resolve a missing texture, find and copy),
        so a folder one of them skips is one all of them skip.

        Parameters:
            root: The folder to walk.

        Yields:
            ``(folder, subfolders, files)``, as ``os.walk`` does.
        """
        for folder, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not cls._skips_dir(d)]
            yield folder, dirs, files

    @classmethod
    def _skips_dir(cls, name: str) -> bool:
        """Whether :meth:`walk` leaves the folder *name* out."""
        name = str(name).lower()
        return name in cls.WALK_SKIP_DIRS or name.startswith(".trash-")

    @staticmethod
    def on_disk(path: str) -> bool:
        """Whether a file is behind *path* -- a tile / frame pattern when any
        file of its set is (:meth:`TiledPath.tiles`)."""
        return bool(path) and bool(TiledPath.tiles(path))

    @classmethod
    def same_content(cls, path_a: str, path_b: str) -> bool:
        """Whether two existing files -- or two tile sets -- hold the same bytes.

        A set is the same as another when both hold the same tiles (by name,
        any case) and each tile pair is byte-identical
        (:meth:`FileUtils.has_same_content`). A set is never the same as a
        single file.
        """
        token_a, token_b = TiledPath.has_token(path_a), TiledPath.has_token(path_b)
        if token_a != token_b:
            return False
        if not token_a:
            return FileUtils.has_same_content(path_a, path_b)
        tiles_a = {os.path.basename(t).lower(): t for t in TiledPath.tiles(path_a)}
        tiles_b = {os.path.basename(t).lower(): t for t in TiledPath.tiles(path_b)}
        return (
            bool(tiles_a)
            and tiles_a.keys() == tiles_b.keys()
            and all(FileUtils.has_same_content(tiles_a[k], tiles_b[k]) for k in tiles_a)
        )

    @classmethod
    def choose(cls, paths: Iterable[str]) -> Tuple[Optional[str], List[str]]:
        """The one file *paths* mean, or the different files that make them ambiguous.

        THE rule every lookup by file name here and in the hosts shares
        (:meth:`resolve`, a texture tool's Resolve Missing, a Collect's
        search): one file reached through several spellings is one
        candidate, and so are files whose contents are identical
        (:meth:`same_content`; the first spelling is kept) -- either answer
        reads the same bytes. Several different files are never settled by
        their order or their age: the caller gets them back to show.

        Parameters:
            paths: Candidate files (or tile patterns), in priority order.

        Returns:
            ``(path, [])`` when they come down to one, ``(None, candidates)``
            when they do not -- the different files, in order -- and
            ``(None, [])`` for none.
        """
        distinct: List[str] = []
        spelled: Set[str] = set()
        for path in paths:
            path = str(path or "")
            key = os.path.normcase(os.path.abspath(path)) if path else ""
            if not key or key in spelled:
                continue
            spelled.add(key)
            if any(
                FileUtils.is_same_file(path, kept) or cls.same_content(path, kept)
                for kept in distinct
            ):
                continue
            distinct.append(cls.spelled(os.path.abspath(path)))
        if len(distinct) == 1:
            return distinct[0], []
        return None, distinct

    @classmethod
    def candidates(
        cls,
        names: Iterable[str],
        search_dirs: Iterable[str] = (),
        walk_root: str = "",
        find_files: Optional[Callable[[List[str], str], List[str]]] = None,
    ) -> Dict[str, List[str]]:
        """Every file of each name in *search_dirs* and under *walk_root*.

        The search behind every lookup by name: each search folder is a plain
        join, in order (a consumer that joins names would use that order), then
        ONE walk of *walk_root* for all the names (*find_files*). A tile /
        frame name is a set: each folder holding a tile of it is one
        candidate, spelled as the pattern in that folder. Nothing is chosen
        here -- :meth:`choose` is the rule that does.

        Parameters:
            names: File names (a path's base name is used).
            search_dirs: Folders to join each name against.
            walk_root: The folder walked for them; ``""`` for no walk.
            find_files: ``(names, root) -> paths``; default :meth:`find_files`.

        Returns:
            ``{lower-case name: [candidate path, ...]}`` -- every name asked
            for, in search-folder then walk order (one spelling may repeat; a
            caller hands the list to :meth:`choose`).
        """
        asked: Dict[str, str] = {}
        for name in names:
            base = cls.file_name(name)
            if base:
                asked.setdefault(base.lower(), base)
        found: Dict[str, List[str]] = {key: [] for key in asked}
        for folder in search_dirs:
            if not folder:
                continue
            for key, base in asked.items():
                path = os.path.join(folder, base)
                if cls.on_disk(path):
                    found[key].append(cls.spelled(os.path.abspath(path)))
        if asked and walk_root and os.path.isdir(walk_root):
            sets = {
                key: TiledPath.wildcard(key, None).lower()
                for key in asked
                if TiledPath.has_token(key)
            }
            for hit in (find_files or cls.find_files)(list(asked.values()), walk_root):
                low = os.path.basename(hit).lower()
                if low in found and low not in sets:
                    found[low].append(cls.spelled(os.path.abspath(hit)))
                    continue
                for key, pattern in sets.items():
                    if fnmatch.fnmatchcase(low, pattern):
                        spelled = os.path.join(os.path.dirname(hit), asked[key])
                        spelled = cls.spelled(os.path.abspath(spelled))
                        if spelled not in found[key]:
                            found[key].append(spelled)
        return found

    @staticmethod
    def _ambiguous_note(candidates: Sequence[str]) -> str:
        """The ``note`` of a record :meth:`choose` would not settle."""
        folders = ", ".join(os.path.dirname(c) for c in candidates)
        return (
            f"ambiguous: {len(candidates)} same-named files ({folders}) -- "
            "not guessing; pick one"
        )

    @classmethod
    def resolve(
        cls,
        refs: Iterable[Sequence[str]],
        search_dirs: Iterable[str] = (),
        walk_root: str = "",
        find_files: Optional[Callable[[List[str], str], List[str]]] = None,
        resolve_hint: Optional[Callable[[str, str], str]] = None,
        key: Optional[Callable[[Any, str], Hashable]] = None,
    ) -> List[Dict[str, Any]]:
        """Every file *refs* name, resolved on disk NOW; one record per unique name.

        Each name is looked for in its recorded folders first (the hint -- every
        distinct folder its owners recorded, in first-seen order: after a
        partial repath the owners of one file can disagree, and the stale one
        must not hide the one that holds it). A name no recorded folder holds
        is SEARCHED: every same-named file in *search_dirs* (each a plain join)
        and under *walk_root* (recursively, with *find_files*) is a candidate
        (:meth:`candidates`), and :meth:`choose` decides -- one file (or
        identical copies) resolves it, ``found_by`` = ``"search"``; several
        different files leave it unresolved, its ``candidates`` listed and the
        reason in ``note``. A stale same-named file in a search folder is
        never taken over a newer one elsewhere because the folder was searched
        first. A tile / frame name (``rock.<UDIM>.png``) is a set, present
        when any of its tiles is.

        Parameters:
            refs: ``(owner, file name, recorded folder)`` references. A name may
                be a path (its base name is used); owners are kept in order.
            search_dirs: Folders to join each name against after the hint.
            walk_root: The folder a missing file is searched for under; ``""``
                for no walk.
            find_files: ``(names, root) -> paths``; default :meth:`find_files`.
            resolve_hint: ``(recorded folder, name) -> absolute folder``, for a
                host that stores folders in a portable spelling (relative to a
                project); default: the recorded folder as it is. Handed the
                folder exactly as recorded -- a respelling here is what turned a
                UNC share into a Blender-relative ``//`` path.
            key: ``(owner, file name) -> key``: which references share a
                record. Default: the lower-case name -- a record names its file
                by name. A reader that names its file by PATH (a texture node)
                passes the owner, so two same-named files keep a record each.

        Returns:
            The dependency records (see the class docstring), in first-seen order.
        """
        key_of = key or (lambda _owner, name: name.lower())
        deps: Dict[Hashable, Dict[str, Any]] = {}
        recorded: Dict[Hashable, List[str]] = {}  # record key -> distinct folders
        for owner, name, folder, *_rest in refs:
            name = cls.file_name(name)
            if not name:
                continue
            rec = key_of(owner, name)
            dep = deps.get(rec)
            if dep is None:
                dep = deps[rec] = {
                    "name": name,
                    "dir": str(folder or ""),
                    "owners": [],
                    "path": None,
                    "found_by": None,
                    "note": "",
                    "candidates": [],
                }
                recorded[rec] = []
            dep["owners"].append(owner)
            folder = str(folder or "")
            if folder and all(
                os.path.normcase(folder) != os.path.normcase(f) for f in recorded[rec]
            ):
                recorded[rec].append(folder)

        for rec, dep in deps.items():
            for folder in recorded[rec]:
                where = resolve_hint(folder, dep["name"]) if resolve_hint else folder
                candidate = os.path.join(where, dep["name"]) if where else ""
                if candidate and cls.on_disk(candidate):
                    dep["path"] = cls.spelled(os.path.abspath(candidate))
                    dep["found_by"] = cls.FOUND_BY_HINT
                    dep["dir"] = folder  # the recorded folder that held it
                    break

        pending = [d for d in deps.values() if d["path"] is None]
        if not pending:
            return list(deps.values())
        found = cls.candidates(
            [d["name"] for d in pending],
            search_dirs=search_dirs,
            walk_root=walk_root,
            find_files=find_files,
        )
        for dep in pending:
            path, several = cls.choose(found.get(dep["name"].lower()) or [])
            if path:
                dep["path"] = path
                dep["found_by"] = cls.FOUND_BY_SEARCH
            elif several:
                dep["candidates"] = several
                dep["note"] = cls._ambiguous_note(several)
        return list(deps.values())

    @classmethod
    def search_dirs(
        cls, deps: Iterable[Dict[str, Any]], then: Iterable[str] = ()
    ) -> List[str]:
        """The folders a consumer that joins names should try, in priority order.

        The folders the dependencies resolved to FIRST -- the most-read first
        (by their owners), ties broken on the path -- then *then*, whatever the
        host searches by default. A consumer that can only join a name against
        a list takes the first folder holding a file of that name, so the order
        is the contract: a host's texture folder routinely holds a same-named
        file from an earlier run, and reaching it first bound a 17-day-old
        lightmap on a production room. Existing folders only, de-duplicated.

        A folder that holds a file under a resolved dependency's name -- the
        consumer's own join, so a case-folding file system folds the case as
        it will -- without being that dependency's folder is a SHADOW, and is
        moved after the dependency's folder whatever its reads: on a production
        room the texture root (105 readers) held a 512 px atlas from an earlier
        bake under the name a current 2048 px one carried in its own folder (46
        readers), and the GLB shipped the stale map through the current rects.
        Two folders that shadow each other cannot both lead: the read order
        stands and the names no order can serve are warned.

        Parameters:
            deps: Dependency records (:meth:`resolve`).
            then: Folders to append, in order, after the resolved ones.

        Returns:
            The folders, in the order a consumer should try them.
        """
        deps = list(deps)
        named: Dict[str, int] = {}
        spelled: Dict[str, str] = {}
        for dep in deps:
            folder = os.path.dirname(dep["path"]) if dep.get("path") else ""
            if not folder or not os.path.isdir(folder):
                continue
            key = os.path.normcase(os.path.abspath(folder))
            named[key] = named.get(key, 0) + max(1, len(dep.get("owners") or ()))
            spelled.setdefault(key, folder)
        order = [
            key for key, _n in sorted(named.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        for folder in then:
            if not folder or not os.path.isdir(folder):
                continue
            key = os.path.normcase(os.path.abspath(folder))
            if key not in spelled:
                spelled[key] = folder
                order.append(key)
        order = cls._unshadowed(order, spelled, deps)
        return [spelled[key] for key in order]

    @classmethod
    def _unshadowed(
        cls, order: List[str], spelled: Dict[str, str], deps: List[Dict[str, Any]]
    ) -> List[str]:
        """*order* (folder keys) with each dependency's folder ahead of its shadows.

        A stable topological sort: the first folder in *order* with no
        unplaced folder it must follow goes next. Where none is free (folders
        shadowing each other) the first unplaced one goes anyway, and the names
        that order cannot serve are warned once.
        """
        must_follow: Dict[str, set] = {key: set() for key in order}
        shadows: List[Tuple[str, str, str]] = []  # (name, own folder, shadow)
        # Each folder listed once, its names case-folded: a hit is then stated
        # through the consumer's own join, which folds case only where the file
        # system does -- not one stat per dependency per folder.
        listings: Dict[str, Set[str]] = {}
        for key in order:
            try:
                listings[key] = {n.casefold() for n in os.listdir(spelled[key])}
            except OSError:
                listings[key] = set()
        for dep in deps:
            path = dep.get("path")
            if not path:
                continue
            own = os.path.normcase(os.path.abspath(os.path.dirname(path)))
            if own not in must_follow:
                continue
            name = os.path.basename(path)
            folded = name.casefold()
            for key in order:
                if key == own or folded not in listings[key]:
                    continue
                twin = os.path.join(spelled[key], name)
                if os.path.isfile(twin) and not FileUtils.is_same_file(twin, path):
                    must_follow[key].add(own)
                    shadows.append((name, own, key))
        if not shadows:
            return order
        placed: List[str] = []
        left = list(order)
        while left:
            key = next((k for k in left if must_follow[k] <= set(placed)), left[0])
            placed.append(key)
            left.remove(key)
        rank = {key: i for i, key in enumerate(placed)}
        lost = sorted({name for name, own, key in shadows if rank[key] < rank[own]})
        if lost:
            cls.logger.warning(
                "No folder order puts every file ahead of its same-named twins; "
                "a consumer joining names onto this list binds the twin for: "
                "%s (folders: %s)",
                ", ".join(lost),
                ", ".join(spelled[key] for key in placed),
            )
        return placed

    @staticmethod
    def keep_read_elsewhere(
        deps: Iterable[Dict[str, Any]], claims: Mapping[str, Any], reader: Any
    ) -> List[str]:
        """Mark each record *reader* also reads ``keep_source`` -- a move copies it
        instead (:meth:`relocate`) -- and return their names.

        A host's records hold only its own scene's readers; one that reads a
        file another scene does too (*reader*, in the host's :meth:`claims`)
        must not take the file away from it.

        Parameters:
            deps: Dependency records (``name``, or a host's ``map``).
            claims: ``{lower-case file name: readers}`` (:meth:`claims`).
            reader: The reader that marks a file read elsewhere.
        """
        kept = []
        for dep in deps:
            name = FileDependencies.file_name(dep.get("name") or dep.get("map"))
            if reader in claims.get(name.lower(), ()):
                dep["keep_source"] = True
                kept.append(name)
        return kept

    @classmethod
    def copy_files(
        cls,
        sources: Iterable[str],
        dest_dir: str,
        mode: str = "copy",
        progress: Optional[Callable[[int, int], Any]] = None,
    ) -> List[Tuple[str, str]]:
        """Copy (or, ``mode="move"``, move) *sources* into *dest_dir*, without clobbering.

        The default :meth:`relocate` copies with. A source that already IS
        the destination -- the same file however spelled, a junction or a
        mapped drive included (:meth:`FileUtils.is_same_file`) -- is left
        alone. A file already at the destination is reused when its CONTENT is
        the source's (:meth:`FileUtils.has_same_content`; a move then drops
        the redundant source) and SKIPPED when it is not: a same-named
        different file is someone else's, and replacing it would rebind every
        reader of that name to the wrong content. Size is not the test -- two
        bakes of one map are the same size, and a move that trusted it deleted
        the new one and kept the stale. Failures are logged and skipped.

        Parameters:
            sources: The files to copy or move.
            dest_dir: The folder they go into.
            mode: ``"copy"`` or ``"move"``.
            progress: ``progress(done, total)`` after each file; ``False``
                back stops it (the files so far are returned).

        Returns:
            ``[(source, destination)]`` for each file now at the destination.
        """
        landed: List[Tuple[str, str]] = []
        sources = list(sources)
        for done, src in enumerate(sources, 1):
            if progress is not None and done > 1 and progress(done - 1, len(sources)) is False:
                break
            dst = os.path.join(dest_dir, os.path.basename(src))
            try:
                if FileUtils.is_same_file(src, dst):
                    landed.append((src, dst))
                    continue
                if os.path.exists(dst):
                    if not FileUtils.has_same_content(src, dst):
                        cls.logger.warning(
                            "%s already exists in %s with different content; "
                            "skipped rather than overwritten.",
                            os.path.basename(dst),
                            dest_dir,
                        )
                        continue
                    if mode == "move":
                        os.remove(src)
                    landed.append((src, dst))
                    continue
                # Never overwrite: a file that appeared since the check above
                # is someone else's too.
                transfer = (
                    FileUtils.move_file if mode == "move" else FileUtils.copy_file
                )
                transfer(src, dest_dir, overwrite=False)
                landed.append((src, dst))
            except (OSError, shutil.Error) as e:
                cls.logger.warning("Could not %s %s: %s", mode, src, e)
        return landed

    @classmethod
    def relocate(
        cls,
        deps: Sequence[Dict[str, Any]],
        dest_dir: str,
        source_dir: str = "",
        mode: str = "copy",
        dry_run: bool = False,
        find_files: Optional[Callable[[List[str], str], List[str]]] = None,
        copy: Optional[Callable[[List[str], str, str], List[Tuple[str, str]]]] = None,
    ) -> Dict[str, Any]:
        """Gather *deps*' files into *dest_dir* -- a Collect over records by name.

        A resolved dependency is its own source. An unresolved one is searched
        for under *source_dir* (:meth:`candidates`), with its own candidates
        when it has some; :meth:`choose` decides, so several different
        same-named files leave it in ``ambiguous`` rather than guessed at. Where
        each file lands is :meth:`FileCollect.plan`'s: its name kept, flat in
        *dest_dir* unless a different file of that name already is (or is
        gathered with it) -- then under the shortest run of its own parent
        folders that tells them apart -- and a source already under
        *dest_dir* stays where it is. The caller repoints each record at the
        folder its file ended up in: every ``copied`` destination's and every
        ``in_place`` source's. With *dry_run* nothing is created or copied --
        the plan comes back as it would run.

        A dependency carrying ``"keep_source": True`` is copied in a move (a
        file another reader still needs).

        Parameters:
            deps: Dependency records (:meth:`resolve`).
            dest_dir: The folder to gather into (created when needed).
            source_dir: Where to look for the files that did not resolve.
            mode: ``"copy"`` or ``"move"``.
            dry_run: Plan only.
            find_files: ``(names, root) -> paths``; default :meth:`find_files`.
            copy: ``(sources, dest_dir, mode) -> [(src, dst)]``, the host's own
                copier (:meth:`FileCollect.apply`); default :meth:`copy_files`.

        Returns:
            ``{"relocate": [(src, dst)], "in_place": [src], "missing": [deps],
            "ambiguous": [deps, each with its "candidates"], "blocked": [deps
            no folder under the destination is free for], "copied": [(src,
            dst)], "plan": [rows]}`` (rows: :meth:`FileCollect.plan`).
        """
        # Deferred: the Collect sits above this module -- it resolves through it.
        from pythontk.file_utils.file_collect import FileCollect

        result: Dict[str, Any] = {
            "relocate": [],
            "in_place": [],
            "missing": [],
            "ambiguous": [],
            "blocked": [],
            "copied": [],
            "plan": [],
        }
        if not deps:
            return result
        dest_dir = cls.spelled(dest_dir)

        pending = [d["name"] for d in deps if not d.get("path")]
        found = (
            cls.candidates(pending, walk_root=source_dir, find_files=find_files)
            if pending and source_dir
            else {}
        )
        entries = []
        for dep in deps:
            source, several = (dep["path"], []) if dep.get("path") else cls.choose(
                list(dep.get("candidates") or [])
                + found.get(dep["name"].lower(), [])
            )
            entries.append(
                {
                    "owner": dep["name"],
                    "source": source,
                    "candidates": several,
                    "keep_source": bool(dep.get("keep_source")),
                    "note": dep.get("note") or "",
                    "dep": dep,
                }
            )
        rows = FileCollect.plan(entries, dest_dir)
        result["plan"] = rows
        for row in rows:
            status, dep = row["status"], row["dep"]
            if status == FileCollect.IN_PLACE:
                result["in_place"].append(row["source"])
            elif status in FileCollect.TRANSFERS:
                result["relocate"].append((row["source"], row["destination"]))
            elif status == FileCollect.AMBIGUOUS:
                result["ambiguous"].append(dict(dep, candidates=row["candidates"]))
            elif status == FileCollect.BLOCKED:
                result["blocked"].append(dep)
            else:
                result["missing"].append(dep)
        if dry_run or not result["relocate"]:
            return result

        landed = FileCollect.apply(rows, mode=mode, copy=copy)
        result["copied"] = [
            (row["source"], row["destination"])
            for row in landed
            if row["status"] in FileCollect.TRANSFERS
        ]
        return result

    # ------------------------------------------------------------ references
    #: How :meth:`reference_bytes` reads each file type, by extension: the
    #: name of the reader method. A Maya ASCII scene names its references in
    #: its header (the ``file -r`` lines precede the first ``createNode``); a
    #: ``.blend`` is read decompressed. Any other type -- a Maya binary scene
    #: among them -- is read whole. A host adds a type with one entry.
    REFERENCE_READERS: Dict[str, str] = {
        ".ma": "_header_bytes",
        ".blend": "_decompressed_bytes",
    }

    @classmethod
    def reference_bytes(cls, path: str) -> Optional[bytes]:
        """The bytes of the file at *path* that can name the files it
        references, read without the app that wrote it
        (:attr:`REFERENCE_READERS`).

        Parameters:
            path: The file.

        Returns:
            The bytes, or None when they cannot be read (missing, truncated, a
            zstd file with no decoder in this interpreter).
        """
        ext = os.path.splitext(str(path))[1].lower()
        reader = getattr(cls, cls.REFERENCE_READERS.get(ext, "_whole_bytes"))
        try:
            return reader(path)
        except (OSError, EOFError, ValueError, *_ZstdErrors.classes()):
            return None

    @classmethod
    def scan_references(
        cls,
        root: str,
        pattern: Union[bytes, Pattern[bytes]],
        extensions: Iterable[str],
        skip: Iterable[str] = (),
    ) -> Optional[Set[str]]:
        """What the files under *root* name: every match of *pattern* (its
        first group, else the whole match) in the :meth:`reference_bytes` of
        each file with one of *extensions* -- no file is opened in its app.

        The question a purge of derived files asks (which of them does
        nothing read any more), so a file that cannot be read stops the
        answer: None, logged -- never a partial set a caller would delete by.
        The walk is :meth:`walk`'s (no folder of stale copies is entered).

        Parameters:
            root: The folder to walk.
            pattern: A bytes regular expression.
            extensions: The file types to read (``".ma"``, ``".blend"``).
            skip: Folders not entered (the derived files' own).

        Returns:
            The matches, decoded, or None.
        """
        regex = re.compile(pattern) if isinstance(pattern, bytes) else pattern
        exts = {str(e).lower() for e in extensions}
        skipped = {os.path.normcase(os.path.normpath(s)) for s in skip if s}
        found: Set[str] = set()
        for folder, dirs, files in cls.walk(root):
            if os.path.normcase(os.path.normpath(folder)) in skipped:
                dirs[:] = []
                continue
            for name in files:
                if os.path.splitext(name)[1].lower() not in exts:
                    continue
                path = os.path.join(folder, name)
                data = cls.reference_bytes(path)
                if data is None:
                    cls.logger.warning(
                        "%s cannot be scanned for the files it references.", path
                    )
                    return None
                for match in regex.finditer(data):
                    hit = match.group(1) if regex.groups else match.group(0)
                    found.add(hit.decode("utf-8", "replace"))
        return found

    @staticmethod
    def open_decompressed(path: str) -> Optional[BinaryIO]:
        """*path* as a readable binary stream of its uncompressed bytes.

        By its magic: gzip through the stdlib; zstd through
        ``compression.zstd`` (Python 3.14+), else the optional ``zstandard``
        (Blender bundles it; that stream reads forward only), across frames;
        anything else as it is.

        Parameters:
            path: The file.

        Returns:
            The stream (the caller closes it), or None for a zstd file with no
            decoder in this interpreter. A zstd stream raises ``OSError`` for
            a corrupt frame, as gzip's does (the decoders' own ``ZstdError``
            is an ``Exception``, which a reader's ``except OSError`` misses).

        Raises:
            OSError: The file could not be opened.
        """
        import gzip

        with open(path, "rb") as fh:
            magic = fh.read(4)
        if magic[:2] == b"\x1f\x8b":
            return gzip.open(path, "rb")
        if magic == b"\x28\xb5\x2f\xfd":
            try:
                from compression import zstd  # Python 3.14+
            except ImportError:
                pass
            else:
                return _ZstdErrors.as_os_errors(
                    _ZstdErrors.call(zstd.open, path, "rb")
                )
            try:
                import zstandard
            except ImportError:
                return None
            raw = open(path, "rb")
            try:
                return _ZstdErrors.as_os_errors(
                    _ZstdErrors.call(
                        zstandard.ZstdDecompressor().stream_reader,
                        raw,
                        read_across_frames=True,
                        closefd=True,
                    )
                )
            except BaseException:
                raw.close()
                raise
        return open(path, "rb")

    @staticmethod
    def _whole_bytes(path: str) -> bytes:
        with open(path, "rb") as fh:
            return fh.read()

    @staticmethod
    def _header_bytes(path: str) -> bytes:
        """A Maya ASCII scene's lines before its first ``createNode``."""
        chunks = []
        with open(path, "rb") as fh:
            for line in fh:
                if line.startswith(b"createNode"):
                    break
                chunks.append(line)
        return b"".join(chunks)

    @classmethod
    def _decompressed_bytes(cls, path: str) -> Optional[bytes]:
        stream = cls.open_decompressed(path)
        if stream is None:
            return None
        with stream:
            return b"".join(iter(lambda: stream.read(1 << 20), b""))


class _ZstdErrors:
    """The zstd decoders' own error classes, read as ``OSError``.

    ``compression.zstd.ZstdError`` and ``zstandard.ZstdError`` derive from
    ``Exception``, so a corrupt frame escaped every ``except OSError`` a
    reader of :meth:`FileDependencies.open_decompressed` wrote. Each class is
    resolved only when its module imports.
    """

    @staticmethod
    def classes() -> Tuple[type, ...]:
        found = []
        try:
            from compression import zstd  # Python 3.14+
        except ImportError:
            pass
        else:
            found.append(getattr(zstd, "ZstdError", None))
        try:
            import zstandard
        except ImportError:
            pass
        else:
            found.append(getattr(zstandard, "ZstdError", None))
        return tuple(c for c in found if isinstance(c, type))

    @classmethod
    def call(cls, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """``fn(*args, **kwargs)``, a decoder error raised as ``OSError``."""
        errors = cls.classes()
        if not errors:
            return fn(*args, **kwargs)
        try:
            return fn(*args, **kwargs)
        except errors as e:
            raise OSError(f"corrupt zstd data: {e}") from e

    @classmethod
    def as_os_errors(cls, stream: Any) -> Any:
        """*stream*, its decoder errors raised as ``OSError``."""
        errors = cls.classes()
        return _OSErrorStream(stream, errors) if errors else stream


class _OSErrorStream:
    """A decompressing stream whose calls raise ``OSError`` for its decoder's
    own errors; everything else passes through."""

    def __init__(self, stream: Any, errors: Tuple[type, ...]):
        self._stream = stream
        self._errors = errors

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._stream, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            try:
                return attr(*args, **kwargs)
            except self._errors as e:
                raise OSError(f"corrupt zstd data: {e}") from e

        return call

    def __enter__(self) -> "_OSErrorStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stream.close()

    def __iter__(self) -> Iterator[bytes]:
        return iter(self.readline, b"")
