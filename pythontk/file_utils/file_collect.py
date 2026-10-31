# !/usr/bin/python
# coding=utf-8
"""Collect: gather the files a set of readers use into one folder, keyed by reader.

A texture node, a lightmap marker, an audio event: each READER names its own
file, and a Collect brings every reader's file into one destination folder and
says where each one landed, so the host can repoint its readers there. It is
two steps, each a plain-value call:

* **Resolve** (:meth:`FileCollect.entries`): each reader's own file when its
  stored path resolves, else the files of its name under a search folder --
  :meth:`FileDependencies.resolve`'s rule, so several different same-named
  files leave the reader AMBIGUOUS with its candidates, never a guess.
* **Plan** (:meth:`FileCollect.plan`): where each file lands. File names never
  change. A file goes flat into the destination; a name two DIFFERENT files
  carry -- among the readers, or already in the destination -- puts each of
  them under the shortest run of its own parent folders that tells it apart
  (``dest/wood/diffuse.png``, ``dest/metal/diffuse.png``). Two different
  files always differ somewhere in their full paths, so such a run exists.
  Files with identical contents share one copy, a file already under the
  destination stays where it is, and a tile set (``rock.<UDIM>.png``) is
  planned, and moved, as one.

:meth:`FileCollect.apply` does the plan's file work; repointing the readers is
the host's. The preview and the commit derive every destination through the one
:meth:`FileCollect.plan` call, so a preview cannot promise a path the commit
would not write.

A host reaches its readers through :class:`CollectHost` (the stored path and
how it resolves), with :class:`PureCollectHost` the answer for plain paths.
"""

import os
import posixpath
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

from pythontk.core_utils.cancel_scope import OperationCancelled
from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.file_dependencies import FileDependencies
from pythontk.file_utils.tiled_path import TiledPath


class CollectHost(Protocol):
    """What a Collect asks of the host whose readers it gathers files for.

    Each DCC's Texture Path Editor implements it (its readers are file nodes /
    image datablocks); :class:`PureCollectHost` holds it for plain paths.
    """

    def stored_path(self, node: Any) -> str:
        """The path *node* stores, as stored (``""`` for none)."""

    def resolve_path(self, path: str, file_node: Any = None) -> Tuple[bool, str]:
        """``(on disk, absolute path)`` for *path* as *file_node* reads it --
        tile / frame tokens kept in the path, the verdict over the set."""

    def stored_form(self, path: str, file_node: Any = None) -> str:
        """The spelling *file_node* would store for the absolute *path* (a
        project-relative one where the host keeps those)."""

    def repath(self, rows: List[Dict[str, Any]]) -> int:
        """Point the reader of each landed row at its ``destination`` (one undo
        step; a reader already storing its final form left alone). Returns how
        many readers changed."""

    def read_elsewhere(self, nodes: Sequence[Any]) -> Iterable[str]:
        """The files the scene reads through readers OTHER than *nodes* --
        every reader the host knows (a node a scope or a filter leaves out, a
        linked one), absolute; a tile set by its pattern or a tile. A move
        copies these instead: taken away, they would strand those readers."""


class PureCollectHost:
    """The :class:`CollectHost` with no host: a reader is a key of *paths*, its
    stored path the value, read relative to *base* (or the process folder).
    A repath writes the new path back into *paths*.

    Parameters:
        paths: ``{reader: stored path}``.
        base: The folder a relative stored path is read from.
    """

    def __init__(self, paths: Optional[Mapping[Any, str]] = None, base: str = ""):
        self.paths = dict(paths or {})
        self.base = base

    def stored_path(self, node: Any) -> str:
        return str(self.paths.get(node) or "")

    def resolve_path(self, path: str, file_node: Any = None) -> Tuple[bool, str]:
        if not path:
            return False, ""
        absolute = os.path.abspath(os.path.join(self.base, path) if self.base else path)
        return FileDependencies.on_disk(absolute), absolute

    def stored_form(self, path: str, file_node: Any = None) -> str:
        if self.base and FileUtils.is_under(path, os.path.abspath(self.base)):
            return os.path.relpath(path, os.path.abspath(self.base)).replace("\\", "/")
        return path

    def repath(self, rows: List[Dict[str, Any]]) -> int:
        changed = 0
        for row in rows:
            final = self.stored_form(row["destination"], row["owner"])
            if self.paths.get(row["owner"]) != final:
                self.paths[row["owner"]] = final
                changed += 1
        return changed

    @staticmethod
    def read_elsewhere(nodes: Sequence[Any]) -> Iterable[str]:
        return ()  # plain paths: no readers but the ones handed over


class FileCollect(LoggingMixin):
    """Resolve, plan and apply a Collect; see the module docstring.

    A row of the plan is a plain dict -- the entry's own keys, plus::

        {"owner": reader, "source": the file it reads now (None: unresolved),
         "candidates": [] | the different files an ambiguous reader could mean,
         "destination": where it lands (None: nowhere), "folder": the parent
         run it was kept apart under ("" when flat or in place),
         "status": one of :attr:`STATUSES`}

    Parameters:
        host: The :class:`CollectHost` its readers are read through
            (default :class:`PureCollectHost`).
    """

    #: The source is copied (or moved) to its destination.
    RELOCATE: str = "relocate"
    #: The destination already holds -- or another row writes -- identical
    #: bytes; nothing is copied, and a move only drops the redundant source.
    REUSE: str = "reuse"
    #: The source already is its destination: nothing to do on disk.
    IN_PLACE: str = "in_place"
    #: Several different files of the reader's name: pick one (``choices``).
    AMBIGUOUS: str = "ambiguous"
    #: No file found for the reader.
    MISSING: str = "missing"
    #: No folder under the destination is free for it (a different file at
    #: every run of its parents); left where it is.
    BLOCKED: str = "blocked"
    STATUSES: Tuple[str, ...] = (RELOCATE, REUSE, IN_PLACE, AMBIGUOUS, MISSING, BLOCKED)
    #: The statuses whose row has file work in :meth:`apply`.
    TRANSFERS: Tuple[str, ...] = (RELOCATE, REUSE)
    #: The statuses whose reader reads its destination once the plan is applied.
    LANDS: Tuple[str, ...] = (RELOCATE, REUSE, IN_PLACE)
    #: Lines of a listing shown in full before it collapses to a count: long
    #: enough to recognise the operation, short enough that the plan stays one
    #: screenful beside the form that produced it.
    PREVIEW_ROWS: int = 12

    def __init__(
        self,
        host: Optional[CollectHost] = None,
        report: Optional[Callable[..., Any]] = None,
    ):
        self.host = host if host is not None else PureCollectHost()
        #: ``report(message, level, items)`` -- where :meth:`run` says what it
        #: did (a host's own channel, a panel's pane); default this class's logger.
        self.report = report or self._log

    # ------------------------------------------------------------------ resolve
    def entries(
        self,
        nodes: Iterable[Any],
        search_dir: str = "",
        find_files: Optional[Callable[[List[str], str], List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """The Resolve step: what file each of *nodes* reads, or could.

        A reader whose stored path resolves IS its source -- a search could
        only rediscover the same bytes. One that does not is searched for by
        its file name under *search_dir* (:meth:`FileDependencies.resolve`,
        one record per reader: two readers of ``wood/diffuse.png`` and
        ``metal/diffuse.png`` are never merged), and a name several different
        files carry comes back with its ``candidates``.

        Parameters:
            nodes: The readers, in order (each once).
            search_dir: Where an unresolved reader's file is searched for;
                ``""``: not searched.
            find_files: ``(names, root) -> paths`` for the walk (a host's own,
                with its progress); default :meth:`FileDependencies.find_files`.

        Returns:
            ``[{"owner", "path" (stored), "source", "candidates", "note"}]``.
        """
        entries: List[Dict[str, Any]] = []
        refs = []
        for node in dict.fromkeys(nodes):
            stored = self.host.stored_path(node) or ""
            exists, absolute = (
                self.host.resolve_path(stored, node) if stored else (False, "")
            )
            entry = {
                "owner": node,
                "path": stored,
                "source": FileDependencies._spelled(absolute) if exists else None,
                "candidates": [],
                "note": "" if stored else "no path",
            }
            entries.append(entry)
            if stored and not exists:
                name = os.path.basename(absolute) or posixpath.basename(
                    stored.replace("\\", "/")
                )
                refs.append((node, name, ""))
        if refs and search_dir:
            deps = FileDependencies.resolve(
                refs,
                walk_root=search_dir,
                find_files=find_files,
                key=lambda owner, _name: owner,
            )
            by_owner = {dep["owners"][0]: dep for dep in deps}
            for entry in entries:
                dep = by_owner.get(entry["owner"])
                if dep is not None:
                    entry["source"] = dep["path"]
                    entry["candidates"] = list(dep["candidates"])
                    entry["note"] = dep["note"]
        return entries

    # ------------------------------------------------------------------ plan
    @classmethod
    def plan(
        cls,
        entries: Iterable[Mapping[str, Any]],
        dest_dir: str,
        choices: Optional[Mapping[Hashable, str]] = None,
        keep: Iterable[str] = (),
    ) -> List[Dict[str, Any]]:
        """The Collect step: where each entry's file lands in *dest_dir*.

        THE one call a preview and its commit derive destinations through.
        Every entry keeps its name; see the module docstring for where it goes.
        An ambiguous entry stays unplaced until *choices* names its file.

        What a move keeps is decided here, once, per FILE: every row reading a
        file in *keep* -- or one any entry marks ``keep_source`` -- is marked
        ``keep_source``, so the rows of one file can never disagree (what the
        plan shows is what :meth:`apply` does).

        Parameters:
            entries: ``{"owner", "source", "candidates", ...}`` dicts
                (:meth:`entries`, or a host's own -- a lightmap record's); a
                previous plan's rows are entries too. Extra keys ride along.
            dest_dir: The folder everything is gathered into.
            choices: ``{row key: path}`` -- the file picked for an entry,
                keyed by :meth:`row_key` (an owner of one kind is never another
                kind's: a texture and a lightmap may share a name); normally
                one of its candidates, any existing file is taken.
            keep: Files readers outside the plan read
                (:meth:`CollectHost.read_elsewhere`): a move copies them.

        Returns:
            The rows, in entry order (see the class docstring).
        """
        choices = dict(choices or {})
        rows = []
        for entry in entries:
            row = dict(entry)
            row.update(destination=None, folder="", status=cls.MISSING)
            row.setdefault("source", None)
            row["candidates"] = list(row.get("candidates") or [])
            picked = choices.get(cls.row_key(row))
            if picked and FileDependencies.on_disk(picked):
                row["source"] = FileDependencies._spelled(os.path.abspath(picked))
            if not row["source"]:
                row["status"] = cls.AMBIGUOUS if row["candidates"] else cls.MISSING
            rows.append(row)
        _Destinations(FileDependencies._spelled(os.path.abspath(dest_dir))).place(
            [row for row in rows if row["source"]]
        )
        kept = {f for path in keep for f in cls.source_files(path)}
        kept.update(
            f
            for row in rows
            if row.get("keep_source") and row["source"]
            for f in cls.source_files(row["source"])
        )
        for row in rows:
            row["keep_source"] = bool(row["source"]) and not kept.isdisjoint(
                cls.source_files(row["source"])
            )
        return rows

    def plan_nodes(
        self,
        nodes: Iterable[Any],
        dest_dir: str,
        search_dir: str = "",
        find_files: Optional[Callable[[List[str], str], List[str]]] = None,
        choices: Optional[Mapping[Hashable, str]] = None,
        kind: str = "file",
        extra: Iterable[Mapping[str, Any]] = (),
    ) -> List[Dict[str, Any]]:
        """Resolve *nodes* and plan them, with *extra* entries beside them.

        :meth:`entries` (each row's ``kind`` set), then :meth:`plan`, with
        every file a reader outside the plan reads kept by a move
        (:meth:`CollectHost.read_elsewhere` over *nodes*).

        Parameters:
            nodes: The readers, in order.
            dest_dir: The folder everything is gathered into.
            search_dir: Where an unresolved reader's file is searched for.
            find_files: The walk (:meth:`entries`).
            choices: The picks (:meth:`plan`).
            kind: The kind of *nodes*' rows (``"texture"``).
            extra: More entries -- a host's own records (lightmaps).

        Returns:
            :meth:`plan`'s rows.
        """
        nodes = list(dict.fromkeys(nodes))
        entries = [
            dict(e, kind=kind) for e in self.entries(nodes, search_dir, find_files)
        ]
        entries.extend(extra)
        return self.plan(
            entries, dest_dir, choices, keep=self.host.read_elsewhere(nodes) or ()
        )

    @staticmethod
    def row_key(row: Mapping[str, Any]) -> Tuple[str, Any]:
        """``(kind, owner)`` -- what a pick (``plan(choices=)``) names a row by."""
        return str(row.get("kind") or "file"), row.get("owner")

    # ------------------------------------------------------------------ apply
    @classmethod
    def apply(
        cls,
        rows: Iterable[Dict[str, Any]],
        mode: str = "copy",
        copy: Optional[Callable[[List[str], str, str], List[Tuple[str, str]]]] = None,
        progress: Optional[Callable[[int, int], Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Do a plan's file work; return the rows whose reader now reads its destination.

        The :attr:`RELOCATE` rows are copied first, a folder at a time through
        *copy*, then the :attr:`REUSE` rows are checked against what is at
        their destination. A file the copier did not write -- one already at
        the destination when the copy began, which a copier reuses on its own
        test -- counts only when its bytes are the source's: a different bake
        of the name that appeared since the preview is never taken for it. A
        move is a copy, then the source taken away -- only once the row's every
        file is at its destination (a tile set is never split between two
        folders), never for a source that IS its destination however spelled,
        and never for a file any row marks ``keep_source`` (one another reader
        still needs). A row whose copy failed keeps its source and is left out
        of the result; the host repoints only the rows returned (and the
        :attr:`IN_PLACE` ones). In a move every file is compared whole with
        its source before the source goes -- a file that appeared at the
        destination after the copier looked is never taken for it. A cancel
        (*progress* answering False) stops before anything is taken away or
        repointed: what was copied stays, and the error says how much.

        Parameters:
            rows: :meth:`plan`'s rows (the others are ignored).
            mode: ``"copy"`` or ``"move"``.
            copy: ``(sources, dest_dir, "copy") -> [(src, dst)]`` -- the host's
                copier, which never overwrites a different file; handed
                ``progress=callback(done, total)`` over those *sources* too when
                *progress* is given. Default :meth:`FileDependencies.copy_files`.
            progress: ``progress(done, total)`` over every file copied;
                ``False`` back cancels (:class:`OperationCancelled`).

        Returns:
            The rows that landed, :attr:`IN_PLACE` ones included, in order.

        Raises:
            OperationCancelled: *progress* answered False.
        """
        rows = list(rows)
        copier = copy or FileDependencies.copy_files
        pairs = {id(row): cls._files(row) for row in rows if row["status"] in cls.TRANSFERS}
        landed: Set[Tuple[str, str]] = set()

        def key(src, dst):
            norm = lambda p: os.path.normcase(os.path.abspath(p))  # noqa: E731
            return norm(src), norm(dst)

        # Each source once: two readers of one file are one copy, and a copier
        # running its files in parallel must not race itself on one target.
        batches: Dict[str, Dict[str, str]] = {}
        there: Set[Tuple[str, str]] = set()  # a destination before the copy
        for row in rows:
            if row["status"] == cls.RELOCATE:
                folder = os.path.dirname(row["destination"])
                batch = batches.setdefault(folder, {})
                for src, dst in pairs[id(row)]:
                    batch.setdefault(os.path.normcase(os.path.abspath(src)), src)
                    if os.path.exists(dst):
                        there.add(key(src, dst))
        total = sum(len(batch) for batch in batches.values())
        done = 0
        for folder, sources in batches.items():
            os.makedirs(folder, exist_ok=True)
            # ``progress=`` only when there is one: a host's three-argument
            # copier (``relocate``'s contract) keeps working.
            ticking = {}
            if progress is not None:
                ticking["progress"] = lambda n, _t, base=done: progress(base + n, total)
            for src, dst in copier(list(sources.values()), folder, "copy", **ticking) or []:
                pair = key(src, dst)
                if pair not in there or FileUtils.has_same_content(src, dst):
                    landed.add(pair)
            done += len(sources)
            if progress is not None and progress(done, total) is False:
                raise OperationCancelled(
                    f"Cancelled after {len(landed)} of {total} file(s) were copied "
                    "-- they stay at the destination; no source was taken away "
                    "and no reader repathed."
                )
        kept = cls.kept_files(rows)

        out = []
        for row in rows:
            status = row["status"]
            if status == cls.IN_PLACE:
                out.append(row)
                continue
            if status not in cls.TRANSFERS:
                continue
            files = pairs[id(row)]
            if status == cls.RELOCATE:
                ok = bool(files) and all(key(s, d) in landed for s, d in files)
                if ok and mode == "move":
                    # Whole bytes before a source goes: a destination that
                    # appeared after the copier looked passed its own test.
                    ok = all(cls._holds(s, d) for s, d in files)
            else:
                ok = bool(files) and all(cls._holds(s, d) for s, d in files)
            if not ok:
                continue
            if mode == "move":
                cls._drop_sources(
                    [(s, d) for s, d in files if os.path.normcase(os.path.abspath(s)) not in kept]
                )
            out.append(row)
        return out

    @classmethod
    def kept_files(cls, rows: Iterable[Mapping[str, Any]]) -> Set[str]:
        """The files a move of *rows* copies instead (``keep_source``, decided per
        file by :meth:`plan`), normalized -- the one set :meth:`apply` and a
        view of the plan read."""
        return {
            f
            for row in rows
            if row.get("keep_source") and row.get("source")
            for f in cls.source_files(row["source"])
        }

    @staticmethod
    def source_files(path: str) -> Set[str]:
        """The files *path* names (a set: its tiles), normalized."""
        files = TiledPath.tiles(path) if TiledPath.has_token(path) else [path]
        return {os.path.normcase(os.path.abspath(f)) for f in files}

    @staticmethod
    def _holds(src: str, dst: str) -> bool:
        """Whether *dst* holds *src*'s bytes (or IS it, however spelled)."""
        if FileUtils.is_same_file(src, dst):
            return True
        return os.path.isfile(dst) and FileUtils.has_same_content(src, dst)

    # ------------------------------------------------------------------ run
    def run(
        self,
        rows: List[Dict[str, Any]],
        dest_dir: str,
        mode: str = "copy",
        dry_run: bool = False,
        search_dir: str = "",
        copy: Optional[Callable[[List[str], str, str], List[Tuple[str, str]]]] = None,
        progress: Optional[Callable[[str], Any]] = None,
    ) -> bool:
        """Report a plan, or commit it and repath: a whole Collect over the host.

        What cannot be placed is named first, by row (a run that found 46 of
        48 read as a success, and the export then failed on the two). A
        *dry_run* reports what a commit would do and writes nothing -- no
        folder created, no file copied, no reader repathed. A commit applies
        the plan (:meth:`apply`), has the host repath what landed
        (:meth:`CollectHost.repath`), and closes honestly: the readers whose
        stored path STILL resolves nowhere, re-read from the host.

        Parameters:
            rows: :meth:`plan`'s rows -- as shown, picks applied.
            dest_dir: The folder they were planned into.
            mode: ``"copy"`` or ``"move"``.
            dry_run: Report only.
            search_dir: The folder the unresolved were searched under (for the
                report; ``""`` when there was none).
            copy: The host's copier (:meth:`apply`).
            progress: ``progress(text)`` -> a context manager yielding a
                ``callback(done, total)``; ``None``: none.

        Returns:
            bool: whether there was anything to do -- what arms a preview's
            Apply, so a preview that found nothing offers nothing.
        """
        rows = list(rows)
        self._say(self.problems(rows, search_dir))
        if dry_run:
            work, lines = self.preview(rows, dest_dir, mode)
            self._say(lines)
            return work
        if not any(row["status"] in self.LANDS for row in rows):
            self.report("Nothing found to collect.", "warning", [])
            return False
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except OSError as error:
            self.report(f"Cannot create '{dest_dir}': {error}", "error", [])
            return False
        verb = "Moving" if mode == "move" else "Copying"
        with self._progress(progress, f"{verb} files…") as tick:
            try:
                landed = self.apply(rows, mode=mode, copy=copy, progress=tick)
            except OperationCancelled as cancel:
                self.report(str(cancel), "warning", [])
                raise
        self._say(self.outcome(rows, landed, mode))
        self.host.repath(landed)
        self._say(self.unresolved(rows))
        return True

    @classmethod
    def problems(
        cls, rows: Sequence[Mapping[str, Any]], search_dir: str = ""
    ) -> List[Tuple[str, str, List[str]]]:
        """``[(level, message, items)]`` naming the rows a plan cannot place."""
        out = []
        for row in rows:
            if row["status"] == cls.AMBIGUOUS:
                name = os.path.basename(row["candidates"][0])
                out.append(
                    (
                        "warning",
                        f"{row['owner']}: {len(row['candidates'])} different files "
                        f"named {name} — pick one; it keeps its current path until then",
                        list(row["candidates"]),
                    )
                )
        for kind, missing in cls._by_kind(r for r in rows if r["status"] == cls.MISSING):
            items = [cls._label(row) for row in missing]
            if search_dir:
                out.append(
                    (
                        "warning",
                        f"{len(missing)} unresolved {kind}(s) not found under "
                        f"{search_dir} — they keep their current path",
                        items,
                    )
                )
            else:
                placed = sum(
                    r["status"] in cls.LANDS for r in rows if cls._kind(r) == kind
                )
                out.append(
                    (
                        "warning",
                        f"No search folder — {len(missing)} unresolved {kind}(s) "
                        f"skipped; continuing with {placed} that resolve.",
                        items,
                    )
                )
        blocked = [r for r in rows if r["status"] == cls.BLOCKED]
        if blocked:
            out.append(
                (
                    "warning",
                    f"{len(blocked)} file(s) have no free folder under the "
                    "destination — a different file of the name holds every one; "
                    "left where they are",
                    [f"{r['owner']}:  {r['source']}" for r in blocked],
                )
            )
        return out

    def preview(
        self, rows: Sequence[Mapping[str, Any]], dest_dir: str, mode: str
    ) -> Tuple[bool, List[Tuple[str, str, List[str]]]]:
        """``(anything to do, [(level, message, items)])`` -- what a commit of
        *rows* would do, every line from the rows themselves."""
        verb = "move" if mode == "move" else "copy"
        repaths, settled = [], 0
        for row in rows:
            if "path" in row and row["status"] in self.LANDS:
                final = self.host.stored_form(row["destination"], row["owner"])
                if final == row["path"]:
                    settled += 1
                else:
                    repaths.append(
                        (row, f"{row['owner']}:  {row['path']}  →  {final}")
                    )
        moving = [r for r in rows if r["status"] == self.RELOCATE]
        other = [r for r in rows if "path" not in r and r["status"] in self.LANDS]
        if not moving and not repaths and not other:
            return False, [
                (
                    "warning",
                    "Dry run — nothing would change: every file is already at "
                    "the destination and every path already stored in its final "
                    "form.",
                    [],
                )
            ]
        lines = []
        for kind, group in self._by_kind(moving):
            there = sum(
                r["status"] in (self.REUSE, self.IN_PLACE)
                for r in rows
                if self._kind(r) == kind
            )
            lines.append(
                (
                    "info",
                    f"Dry run — would {verb} {len(group)} {kind}(s) into {dest_dir}"
                    + (f" ({there} already there)" if there else ""),
                    self.listing(self._line(r) for r in group),
                )
            )
        if repaths:
            kind = self._kind(repaths[0][0])
            lines.append(
                (
                    "info",
                    f"Would repath {len(repaths)} {kind}(s)",
                    self.listing(line for _row, line in repaths),
                )
            )
        if settled:
            lines.append(("info", f"{settled} already point at the destination.", []))
        if other and not any(self._kind(r) == self._kind(other[0]) for r in moving):
            lines.append(
                (
                    "info",
                    f"{len(other)} {self._kind(other[0])}(s) already at the "
                    "destination; only their records would change.",
                    [],
                )
            )
        lines.append(
            (
                "warning",
                f"Nothing has been written — press Apply to {verb} and repath "
                "exactly this.",
                [],
            )
        )
        return True, lines

    @classmethod
    def outcome(
        cls,
        rows: Sequence[Mapping[str, Any]],
        landed: Sequence[Mapping[str, Any]],
        mode: str,
    ) -> List[Tuple[str, str, List[str]]]:
        """``[(level, message, items)]`` -- what a commit did, per kind."""
        out = []
        verb = "moved" if mode == "move" else "copied"
        for kind, group in cls._by_kind(landed):
            relocated = sum(r["status"] in cls.TRANSFERS for r in group)
            apart = {r["destination"] for r in group if r["folder"]}
            out.append(
                (
                    "success",
                    f"{kind.capitalize()}s — {relocated} {verb}, "
                    f"{len(group) - relocated} already at the destination"
                    + (f"; {len(apart)} kept apart in their own folder(s)" if apart else "")
                    + ".",
                    [],
                )
            )
        arrived = {id(r) for r in landed}
        failed = [r for r in rows if r["status"] in cls.TRANSFERS and id(r) not in arrived]
        if failed:
            out.append(
                (
                    "warning",
                    f"{len(failed)} file(s) did not land — they keep their current path",
                    [f"{r['owner']}:  {r['source']}" for r in failed],
                )
            )
        return out

    def unresolved(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> List[Tuple[str, str, List[str]]]:
        """The honest close: the readers (of *rows* that store a path) whose
        stored path resolves nowhere NOW, re-read from the host."""
        out = []
        readers = [row for row in rows if "path" in row]
        for kind, group in self._by_kind(readers):
            still = []
            for row in group:
                stored = self.host.stored_path(row["owner"])
                if not (stored and self.host.resolve_path(stored, row["owner"])[0]):
                    still.append(f"{row['owner']}:  {stored or '<no path>'}")
            if still:
                out.append(
                    (
                        "warning",
                        f"{len(still)} {kind}(s) still unresolved after this run — "
                        "the export's path check will fail on them",
                        still,
                    )
                )
        return out

    @classmethod
    def listing(cls, lines: Iterable[str]) -> List[str]:
        """*lines* capped at :attr:`PREVIEW_ROWS`, the remainder counted: a cut
        listing that does not SAY it was cut reads as the whole plan."""
        listed = list(lines)
        if len(listed) <= cls.PREVIEW_ROWS:
            return listed
        hidden = len(listed) - cls.PREVIEW_ROWS
        return listed[: cls.PREVIEW_ROWS] + [f"… and {hidden} more"]

    @staticmethod
    def _kind(row: Mapping[str, Any]) -> str:
        return str(row.get("kind") or "file")

    @classmethod
    def _by_kind(cls, rows: Iterable[Mapping[str, Any]]):
        """``[(kind, rows)]`` in first-seen order."""
        groups: Dict[str, List[Mapping[str, Any]]] = {}
        for row in rows:
            groups.setdefault(cls._kind(row), []).append(row)
        return list(groups.items())

    @staticmethod
    def _label(row: Mapping[str, Any]) -> str:
        """``owner:  file name  (note)`` for a report line."""
        name = posixpath.basename(str(row.get("path") or "").replace("\\", "/"))
        note = row.get("note") or ""
        return f"{row['owner']}" + (f":  {name}" if name else "") + (
            f"  ({note})" if note else ""
        )

    @staticmethod
    def _line(row: Mapping[str, Any]) -> str:
        """``name  ←  source folder`` -- and the folder it is kept apart in."""
        line = f"{os.path.basename(row['source'])}  ←  {os.path.dirname(row['source'])}"
        return line + (f"  (kept apart in {row['folder']}/)" if row["folder"] else "")

    def _say(self, lines: Iterable[Tuple[str, str, List[str]]]) -> None:
        for level, message, items in lines:
            self.report(message, level, items)

    @classmethod
    def _log(cls, message: str, level: str = "info", items=None) -> None:
        """The default :attr:`report`: this class's logger, a list indented under
        its title as one record."""
        if items:
            message = "\n".join([message, *(f"  {item}" for item in items)])
        channels = {"warning": cls.logger.warning, "error": cls.logger.error}
        channels.get(level, cls.logger.info)(message)

    @staticmethod
    def _progress(progress, text):
        """``progress(text)``, or a context yielding no callback when there is none."""
        if progress is not None:
            return progress(text)
        import contextlib

        return contextlib.nullcontext(None)

    @staticmethod
    def _files(row: Mapping[str, Any]) -> List[Tuple[str, str]]:
        """``[(source file, destination file)]`` of a row -- a set's every tile."""
        source, destination = row["source"], row["destination"]
        if not TiledPath.has_token(source):
            return [(source, destination)]
        folder = os.path.dirname(destination)
        return [
            (tile, FileDependencies._spelled(os.path.join(folder, os.path.basename(tile))))
            for tile in TiledPath.tiles(source)
        ]

    @classmethod
    def _drop_sources(cls, files: Sequence[Tuple[str, str]]) -> None:
        """Take away each source a move copied -- compared whole by
        :meth:`apply` -- unless it IS its destination however spelled."""
        for src, dst in files:
            try:
                if not os.path.exists(src) or FileUtils.is_same_file(src, dst):
                    continue  # another reader of this file already moved it
                os.remove(src)
            except OSError as error:
                cls.logger.warning(
                    "Moved %s, but could not remove the original (%s).", src, error
                )


class _Destinations:
    """Where each distinct file of a plan lands: :meth:`FileCollect.plan`'s
    placement, one instance per plan (the claims it has made so far)."""

    def __init__(self, dest_dir: str):
        self.dest = dest_dir
        #: lower-case destination file -> the member that claimed it.
        self.claimed: Dict[str, "_Member"] = {}

    def place(self, rows: List[Dict[str, Any]]) -> None:
        """Fill each row's ``destination`` / ``folder`` / ``status``."""
        members = self._members(rows)
        groups: Dict[str, List[_Member]] = {}
        for member in members:
            groups.setdefault(member.key, []).append(member)
        # The files already under the destination hold their place first: a
        # newcomer is placed around them, never over them.
        for member in members:
            if member.in_place:
                self._claim(member, os.path.dirname(member.source), "")
        for group in groups.values():
            newcomers = [m for m in group if not m.in_place]
            for member in newcomers:
                self._place(member, newcomers)
        for member in members:
            member.write()

    def _members(self, rows: List[Dict[str, Any]]) -> List["_Member"]:
        """One member per distinct content of each name: two readers of one
        file are one member, and so are identical files of one name -- the
        one already under the destination standing for them, so its readers
        stay where they are and the others reuse it."""
        members: List[_Member] = []
        by_file: Dict[str, _Member] = {}
        inside = lambda row: _Member.under(row["source"], self.dest)  # noqa: E731
        for row in sorted(rows, key=lambda row: not inside(row)):
            spelled = os.path.normcase(os.path.abspath(row["source"]))
            member = by_file.get(spelled)
            if member is None and not inside(row):
                # Only a newcomer shares an identical file: two already inside
                # each stay where they are (merged, a move deleted one).
                member = self._twin(row["source"], members)
            if member is None:
                member = _Member(row["source"], self.dest)
                members.append(member)
            by_file[spelled] = member
            member.rows.append(row)
        return members

    @staticmethod
    def _twin(source: str, members: List["_Member"]) -> Optional["_Member"]:
        """The member of *source*'s name holding identical bytes, if any."""
        key = os.path.basename(source).lower()
        for member in members:
            if member.key == key and (
                FileUtils.is_same_file(source, member.source)
                or FileDependencies.same_content(source, member.source)
            ):
                return member
        return None

    def _place(self, member: "_Member", newcomers: List["_Member"]) -> None:
        """Put *member* under the shortest run of its parents that no other
        newcomer of its name shares and whose folder is free. Flat (no run)
        when it is its name's one newcomer and the name is free there -- or
        whenever the flat file already holds its bytes; two newcomers of one
        name never race for the flat place."""
        for depth in range(0, len(member.parents) + 1):
            run = member.run(depth)
            if depth and any(
                other is not member and other.run(depth) == run for other in newcomers
            ):
                continue
            folder = os.path.join(self.dest, *member.parents[:depth][::-1])
            fits = self._fits(member, folder)
            if fits is None or (
                not depth and len(newcomers) > 1 and fits != FileCollect.REUSE
            ):
                continue
            self._claim(member, folder, "/".join(member.parents[:depth][::-1]), fits)
            return
        member.status = FileCollect.BLOCKED

    def _fits(self, member: "_Member", folder: str) -> Optional[str]:
        """How *member* lands in *folder*: :attr:`FileCollect.RELOCATE`, or
        :attr:`FileCollect.REUSE` when every file it writes is there already
        with identical bytes; None when another file holds the place -- on
        disk, or claimed by another member of this plan. A tile set also needs
        the folder to hold no OTHER tile of its pattern: the set read there
        would not be the set copied."""
        target = os.path.join(folder, member.name)
        if member.is_set:
            there = {os.path.basename(t).lower() for t in TiledPath.tiles(target)}
            if not there.issubset(member.tile_names):
                return None
        reused = True
        for name in member.file_names:
            path = os.path.join(folder, name)
            owner = self.claimed.get(os.path.normcase(os.path.abspath(path)))
            if owner is not None and owner is not member:
                return None
            if os.path.exists(path):
                source = member.file_for(name)
                if not FileUtils.has_same_content(source, path):
                    return None
            else:
                reused = False
        return FileCollect.REUSE if reused else FileCollect.RELOCATE

    def _claim(
        self,
        member: "_Member",
        folder: str,
        run: str,
        status: str = FileCollect.IN_PLACE,
    ) -> None:
        member.destination = FileDependencies._spelled(
            os.path.join(folder, member.name)
        )
        member.folder = run
        member.status = status
        for name in member.file_names:
            path = os.path.normcase(os.path.abspath(os.path.join(folder, name)))
            self.claimed.setdefault(path, member)


class _Member:
    """One distinct file (or tile set) of a plan and the rows that read it."""

    def __init__(self, source: str, dest_dir: str):
        self.source = source
        self.name = os.path.basename(source)
        self.key = self.name.lower()
        self.rows: List[Dict[str, Any]] = []
        self.is_set = TiledPath.has_token(source)
        tiles = TiledPath.tiles(source) if self.is_set else [source]
        self._tiles = {os.path.basename(t).lower(): t for t in tiles}
        self.tile_names = set(self._tiles)
        self.file_names = [os.path.basename(t) for t in tiles]
        folder = os.path.dirname(os.path.abspath(source))
        self.in_place = self.under(source, dest_dir)
        #: Parent folder names, innermost first (a drive's colon dropped).
        self.parents = [
            part.replace(":", "")
            for part in reversed(folder.replace("\\", "/").split("/"))
            if part.replace(":", "")
        ]
        self.destination: Optional[str] = None
        self.folder = ""
        self.status = FileCollect.MISSING

    @staticmethod
    def under(source: str, dest_dir: str) -> bool:
        """Whether *source* already lies in *dest_dir* or below it -- by the
        spelling, or the filesystem's answer for the folder itself (a junction
        or mapped drive naming the destination)."""
        folder = os.path.dirname(os.path.abspath(source))
        return FileUtils.is_under(folder, dest_dir) or FileUtils.is_same_file(
            folder, dest_dir
        )

    def run(self, depth: int) -> Tuple[str, ...]:
        """The innermost *depth* parents, case-folded: what tells it apart."""
        return tuple(p.casefold() for p in self.parents[:depth])

    def file_for(self, name: str) -> str:
        return self._tiles.get(name.lower(), self.source)

    def write(self) -> None:
        """Fill the member's rows: each reader of the member's own file takes
        its status; a reader of an identical file at another path reuses the
        member's copy (:attr:`FileCollect.REUSE`)."""
        for row in self.rows:
            row["destination"] = self.destination
            row["folder"] = self.folder
            own = FileUtils.is_same_file(row["source"], self.source)
            row["status"] = (
                self.status
                if own or self.status not in FileCollect.LANDS
                else FileCollect.REUSE
            )
