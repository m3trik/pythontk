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
* how to gather the files into one folder (:meth:`FileDependencies.relocate`).

They take plain values -- ``(owner, name, recorded folder)`` references -- and
callables where a host does something its own way (how a stored folder resolves,
how a tree is searched, how a file is copied), so a DCC reads and writes its own
records and hands the rules the data.
"""

import os
import shutil
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
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

    @staticmethod
    def find_files(names: Iterable[str], root: str) -> List[str]:
        """Every file under *root*, recursively, whose name is one of *names*.

        Names compare without case. The default search :meth:`resolve` and
        :meth:`relocate` walk with; a host with a texture walk of its own (a skip
        list, tile tokens) passes that instead. Links are not followed.

        Parameters:
            names: File names (a path's base name is used).
            root: The folder to walk.

        Returns:
            The matching paths, in walk order.
        """
        wanted = {os.path.basename(str(n)).lower() for n in names}
        hits: List[str] = []
        for folder, _dirs, files in os.walk(root):
            hits.extend(os.path.join(folder, n) for n in files if n.lower() in wanted)
        return hits

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

    @staticmethod
    def search_dirs(
        deps: Iterable[Dict[str, Any]], then: Iterable[str] = ()
    ) -> List[str]:
        """The folders a consumer that joins names should try, in priority order.

        The folders the dependencies resolved to FIRST -- the most-read first
        (by their owners), ties broken on the path -- then *then*, whatever the
        host searches by default. A consumer that can only join a name against
        a list takes the first folder holding a file of that name, so the order
        is the contract: a host's texture folder routinely holds a same-named
        file from an earlier run, and reaching it first bound a 17-day-old
        lightmap on a production room. Existing folders only, de-duplicated.

        Parameters:
            deps: Dependency records (:meth:`resolve`).
            then: Folders to append, in order, after the resolved ones.

        Returns:
            The folders, in the order a consumer should try them.
        """
        named: Dict[str, int] = {}
        spelled: Dict[str, str] = {}
        for dep in deps:
            folder = os.path.dirname(dep["path"]) if dep.get("path") else ""
            if not folder or not os.path.isdir(folder):
                continue
            key = os.path.normcase(os.path.abspath(folder))
            named[key] = named.get(key, 0) + max(1, len(dep.get("owners") or ()))
            spelled.setdefault(key, folder)
        dirs = [
            spelled[key]
            for key, _n in sorted(named.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        seen = set(named)
        for folder in then:
            if not folder or not os.path.isdir(folder):
                continue
            key = os.path.normcase(os.path.abspath(folder))
            if key not in seen:
                seen.add(key)
                dirs.append(folder)
        return dirs

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
