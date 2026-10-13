# !/usr/bin/python
# coding=utf-8
"""File dependencies -- files a record names by *name*, found where they are NOW.

Plenty of scene data names a file the way a file path cannot survive: as a file
name plus the folder it was written from. A baked-lightmap marker does, an audio
event does, a baked shadow silhouette does. That folder is history, not a
contract -- a reorganised project, a scene migrated to another module, a
teammate's machine with the project on another drive -- so every tool that reads
such a record ends up needing the same four answers:

* which file each name means NOW (:meth:`FileDependencies.resolve`);
* where a consumer that can only join a name against a list of folders should
  look first (:meth:`FileDependencies.search_dirs`);
* which names a new batch of outputs must not take, because something else
  still reads them (:meth:`FileDependencies.claims`, the input to
  :meth:`FileUtils.unique_path`);
* which files a batch's owners stopped reading, now that nothing reads them
  -- set aside, never deleted (:meth:`FileDependencies.remove_superseded`,
  :meth:`FileDependencies.set_aside`);
* how to gather the files into one folder (:meth:`FileDependencies.relocate`);
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
import os
import re
import shutil
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Pattern,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils._file_utils import FileUtils


class FileDependencies(LoggingMixin):
    """The rules over files that records name by name; see the module docstring.

    A dependency record, as :meth:`resolve` returns it::

        {"name": file name, "dir": recorded folder, "owners": [owner, ...],
         "path": absolute path or None, "found_by": "hint" | "search" | None,
         "note": "" | why an unresolved file stayed unresolved}
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
    def _spelled(path: str) -> str:
        """*path* with forward slashes -- except a UNC path's leading ``\\\\``.

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
            key = os.path.basename(str(name or "")).lower()
            if key:
                readers.setdefault(key, set()).add(owner)
        return {key: frozenset(owners) for key, owners in readers.items()}

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

        def key(path: str) -> str:
            return os.path.normcase(os.path.abspath(path))

        read: Dict[str, str] = {}
        lost = set()
        for _owner, name, path, *_rest in after:
            path = str(path or "")
            if FileUtils.is_rooted_path(path):
                read.setdefault(key(path), path)
            else:
                lost.add(os.path.basename(str(name or path)).lower())
        identities: Optional[set] = None
        removed: List[str] = []
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
            if os.path.isdir(path):
                continue
            if identities is None:  # one stat per file read, and only if needed
                identities = {cls._identity(p) for p in read.values()} - {None}
            if cls._identity(path) in identities:
                continue
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
        return cls._spelled(target)

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
        the file. Another scene file that is still there (the source of a Save
        As copy) reads what it wrote; ``None`` -- no record -- and a relative
        entry with no *base* to read it from are nobody's to delete.

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
        if not FileUtils.is_rooted_path(written):
            return False  # unresolvable, which is not gone
        written = os.path.normcase(os.path.abspath(written))
        if scene and os.path.normcase(os.path.abspath(scene)) == written:
            return True
        return not os.path.isfile(written)

    @staticmethod
    def find_files(names: Iterable[str], root: str) -> List[str]:
        """Every file under *root*, recursively, whose name is one of *names*.

        Names compare without case. The default search :meth:`resolve` and
        :meth:`relocate` walk with; a host with a texture walk of its own (tile
        tokens) passes that instead, walking with :meth:`walk` too. Links are
        not followed, and no :attr:`WALK_SKIP_DIRS` folder is entered: a file
        set aside in ``_superseded`` keeps its name, and a walk that found it
        would bind it again.

        Parameters:
            names: File names (a path's base name is used).
            root: The folder to walk.

        Returns:
            The matching paths, in walk order.
        """
        wanted = {os.path.basename(str(n)).lower() for n in names}
        hits: List[str] = []
        for folder, _dirs, files in FileDependencies.walk(root):
            hits.extend(os.path.join(folder, n) for n in files if n.lower() in wanted)
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

    @classmethod
    def resolve(
        cls,
        refs: Iterable[Sequence[str]],
        search_dirs: Iterable[str] = (),
        walk_root: str = "",
        find_files: Optional[Callable[[List[str], str], List[str]]] = None,
        resolve_hint: Optional[Callable[[str, str], str]] = None,
    ) -> List[Dict[str, Any]]:
        """Every file *refs* name, resolved on disk NOW; one record per unique name.

        Each name is looked for in its recorded folders first (the hint -- every
        distinct folder its owners recorded, in first-seen order: after a
        partial repath the owners of one file can disagree, and the stale one
        must not hide the one that holds it), then in *search_dirs* in order -- each a plain join, the order a
        consumer that joins names would use, so the two cannot disagree about
        a file. A name still missing is then looked for under *walk_root*
        (recursively, with *find_files*): a UNIQUE hit resolves it
        (``found_by`` = ``"search"``); several same-named files leave it
        unresolved with the count in ``note``, rather than guessed at -- the
        rule a texture tool uses for rebinding a file by name.

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

        Returns:
            The dependency records (see the class docstring), in first-seen order.
        """
        deps: Dict[str, Dict[str, Any]] = {}
        recorded: Dict[str, List[str]] = {}  # name key -> distinct folders
        for owner, name, folder, *_rest in refs:
            name = os.path.basename(str(name or ""))
            if not name:
                continue
            key = name.lower()
            dep = deps.get(key)
            if dep is None:
                dep = deps[key] = {
                    "name": name,
                    "dir": str(folder or ""),
                    "owners": [],
                    "path": None,
                    "found_by": None,
                    "note": "",
                }
                recorded[key] = []
            dep["owners"].append(owner)
            folder = str(folder or "")
            if folder and all(
                os.path.normcase(folder) != os.path.normcase(f) for f in recorded[key]
            ):
                recorded[key].append(folder)

        search_dirs = list(search_dirs)
        for key, dep in deps.items():
            attempts = [
                (
                    cls.FOUND_BY_HINT,
                    resolve_hint(folder, dep["name"]) if resolve_hint else folder,
                    folder,
                )
                for folder in recorded[key]
            ]
            attempts.extend((cls.FOUND_BY_SEARCH, d, None) for d in search_dirs)
            for found_by, folder, spelling in attempts:
                candidate = os.path.join(folder, dep["name"]) if folder else ""
                if candidate and os.path.isfile(candidate):
                    dep["path"] = cls._spelled(os.path.abspath(candidate))
                    dep["found_by"] = found_by
                    if spelling is not None:
                        dep["dir"] = spelling  # the recorded folder that held it
                    break

        pending = [d for d in deps.values() if d["path"] is None]
        if pending and walk_root and os.path.isdir(walk_root):
            find = find_files or cls.find_files
            by_name: Dict[str, List[str]] = {}
            for hit in find([d["name"] for d in pending], walk_root):
                by_name.setdefault(os.path.basename(hit).lower(), []).append(hit)
            for dep in pending:
                candidates = by_name.get(dep["name"].lower()) or []
                if len(candidates) == 1:
                    dep["path"] = cls._spelled(os.path.abspath(candidates[0]))
                    dep["found_by"] = cls.FOUND_BY_SEARCH
                elif candidates:
                    dep["note"] = (
                        f"ambiguous: {len(candidates)} same-named files under "
                        f"{walk_root} -- not guessing"
                    )
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

    @classmethod
    def copy_files(
        cls, sources: Iterable[str], dest_dir: str, mode: str = "copy"
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

        Returns:
            ``[(source, destination)]`` for each file now at the destination.
        """
        landed: List[Tuple[str, str]] = []
        for src in sources:
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
        """Gather *deps*' files into *dest_dir*.

        A resolved dependency is its own source; an unresolved one is searched
        for under *source_dir* (recursively; the NEWEST same-named file wins --
        the one most likely to be the current version). A source already in
        *dest_dir* needs no file operation. The caller repoints its records at
        the files that ended up there: every ``copied`` destination and every
        ``in_place`` source. With *dry_run* nothing is created or copied -- the
        plan comes back as it would run.

        Parameters:
            deps: Dependency records (:meth:`resolve`).
            dest_dir: The folder to gather into (created when needed).
            source_dir: Where to look for the files that did not resolve.
            mode: ``"copy"`` or ``"move"``.
            dry_run: Plan only.
            find_files: ``(names, root) -> paths``; default :meth:`find_files`.
            copy: ``(sources, dest_dir, mode) -> [(src, dst)]``, the host's own
                collision policy; default :meth:`copy_files`.

        Returns:
            ``{"relocate": [(src, dst)], "in_place": [src], "missing": [deps],
            "copied": [(src, dst)]}``.
        """
        result: Dict[str, Any] = {
            "relocate": [],
            "in_place": [],
            "missing": [],
            "copied": [],
        }
        if not deps:
            return result
        dest_dir = cls._spelled(dest_dir)

        sources = {d["name"].lower(): d["path"] for d in deps if d.get("path")}
        pending = [d["name"] for d in deps if d["name"].lower() not in sources]
        if pending and source_dir and os.path.isdir(source_dir):
            newest: Dict[str, Tuple[float, str]] = {}
            for hit in (find_files or cls.find_files)(pending, source_dir):
                key = os.path.basename(hit).lower()
                try:
                    mtime = os.path.getmtime(hit)
                except OSError:
                    mtime = 0.0
                if key not in newest or mtime > newest[key][0]:
                    newest[key] = (mtime, hit)
            for key, (_mtime, hit) in newest.items():
                sources[key] = cls._spelled(os.path.abspath(hit))
        result["missing"] = [d for d in deps if d["name"].lower() not in sources]

        for src in sources.values():
            # The filesystem's answer, not the spellings': a dest_dir reached
            # through a junction or a mapped drive IS the source folder, and
            # planning a move there deleted the only copy.
            if FileUtils.is_same_file(os.path.dirname(os.path.abspath(src)), dest_dir):
                result["in_place"].append(src)
            else:
                dst = cls._spelled(os.path.join(dest_dir, os.path.basename(src)))
                result["relocate"].append((src, dst))
        if dry_run or not result["relocate"]:
            return result

        os.makedirs(dest_dir, exist_ok=True)
        result["copied"] = [
            (cls._spelled(s), cls._spelled(d))
            for s, d in (copy or cls.copy_files)(
                [src for src, _dst in result["relocate"]], dest_dir, mode
            )
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
