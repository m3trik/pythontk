# !/usr/bin/python
# coding=utf-8
"""A project's files baked from other files: one slot per source, kept with
the project, purged by reference.

The bookkeeping mayatk's ``BlenderSceneImport`` and blendertk's
``MayaSceneImport`` both keep for the native stand-in a foreign scene is
referenced (or linked) through, hoisted here from their twin copies
(2026-10-13). A bake is a permanent project file -- a saved scene reads it
-- in the project's cache tree (:meth:`Workspace.cache_dir`), never
age-swept. ONE slot per source, so a re-bake replaces the old file; the bake
records which source it holds and that source's content key INSIDE itself,
which is how a current bake is told from a stale one without a conversion;
and a bake no file of the project references any more is purged.

What a host does its own way is handed in: how a record is read from a bake
(a ``.ma``'s ``fileInfo`` header, a ``.blend``'s scene properties), how one is
produced (a headless DCC run), the session's live references. The rules take
plain values and those callables, so neither DCC branches on the other.
"""

import hashlib
import os
import re
import shutil
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Pattern, Set

from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.file_dependencies import FileDependencies
from pythontk.file_utils.temp_artifacts import CachedArtifact
from pythontk.file_utils.workspace import Workspace


class BakeCache(LoggingMixin):
    """One kind of bake in a project's cache tree; see the module docstring.

    A slot is ``<cache dir>/<name>_cache_<slot key><extension>`` (the
    :class:`CachedArtifact` spelling), its slot key ``<source stem>_<hash8>``
    over the source's project-relative path (:meth:`slot_key`). Methods take
    the project root explicitly: resolving "the open scene's project" is the
    host's.

    Parameters:
        name: The store's name (``mtk_bake``): the slots' prefix.
        extension: A bake's extension (``".ma"``, ``".blend"``).
        read_info: ``read_info(bake path) -> {field: value}``: the record a
            bake carries, ``{}`` when it carries none or cannot be read.
        source_field: The record field naming the source a bake holds.
        key_field: The record field holding that source's content key.
        kind: The project cache kind (:meth:`Workspace.cache_dir`: a
            ``<kind>Cache`` rule, else ``cache/<kind>``).
        legacy_suffix: The sidecar beside a bake that named its source before
            the record moved inside it (read by :meth:`source_of`), if any.
        log_level: The store's logging level.
    """

    #: The project cache kind both DCC bake stores share.
    KIND = "bridge"
    #: A slot key: the source's stem (word characters) and an 8-digit hash.
    SLOT_KEY = r"[0-9A-Za-z_]+?_[0-9a-f]{8}"

    def __init__(
        self,
        name: str,
        extension: str,
        *,
        read_info: Callable[[str], Mapping[str, str]],
        source_field: str,
        key_field: str,
        kind: str = KIND,
        legacy_suffix: Optional[str] = None,
        log_level: str = "WARNING",
    ):
        super().__init__()
        self.logger.setLevel(log_level)
        self.name = name
        self.extension = extension
        self.read_info = read_info
        self.source_field = source_field
        self.key_field = key_field
        self.kind = kind
        self.legacy_suffix = legacy_suffix
        #: Normalized cache folder -> its slots' store, built once: the slot
        #: calls run per listed reference.
        self._stores: Dict[str, CachedArtifact] = {}

    # ------------------------------------------------------------------ slots
    @property
    def prefix(self) -> str:
        """What every file of this store is named with: ``<name>_cache_``."""
        return f"{self.name}_cache_"

    @property
    def name_pattern(self) -> Pattern[str]:
        """A file name of this store -- a bake or anything named for it (a
        stream it reads, a legacy sidecar) -- its slot key captured."""
        return re.compile(rf"^{re.escape(self.prefix)}({self.SLOT_KEY})(?:\.|$)")

    def is_bake(self, path: str) -> bool:
        """Whether *path* is named as one of this store's files."""
        return os.path.basename(str(path or "")).startswith(self.prefix)

    def cache_dir(self, project: str, create: bool = False) -> str:
        """Where *project*'s bakes live (absolute; ``create`` makes it)."""
        return Workspace.load(project).cache_dir(self.kind, create=create)

    def store(self, project: str, create: bool = False) -> CachedArtifact:
        """*project*'s slots, as a never-aged :class:`CachedArtifact` (one per
        cache folder: a rule that moves the folder gets another). Each
        :meth:`CachedArtifact.get` still produces into its own scoped scratch,
        so a shared store never shares a write."""
        folder = self.cache_dir(project, create=create)
        key = os.path.normcase(os.path.abspath(folder))
        store = self._stores.get(key)
        if store is None:
            store = self._stores.setdefault(
                key,
                CachedArtifact(
                    self.name,
                    extension=self.extension,
                    dir=folder,
                    max_age_days=None,
                ),
            )
        return store

    @staticmethod
    def slot_key(src: str, project: str) -> str:
        """The slot name of *src*'s bake: ``<stem>_<hash8>``, the hash over its
        project-relative path -- the same source always lands in the same slot,
        and a same-named one elsewhere in another. ASCII word characters only
        (the name is a file name, and a pid-tag spelling would read as a
        session's)."""
        rel = Workspace(project).relative(src)
        digest = hashlib.sha1(
            os.path.normcase(rel).replace("\\", "/").encode("utf-8")
        ).hexdigest()[:8]
        stem = re.sub(r"[^0-9A-Za-z_]", "_", os.path.splitext(os.path.basename(src))[0])
        return f"{stem[:40] or 'scene'}_{digest}"

    def slot(self, src: str, project: str) -> str:
        """Where *project*'s bake of *src* lives -- there or not."""
        return self.store(project).slot(self.slot_key(src, project))

    @staticmethod
    def source_key(src: str, *parts: Any, companions: Iterable[str] = ()) -> str:
        """The content key of a bake of *src* as it is now: *parts* (the bake
        format, the options that shape it) and the size + mtime of *src* and
        of each of *companions* that exists (a manifest beside it: a payload
        can land before its sidecar on a synced drive). No path: a moved
        project keeps its bakes current.

        Raises:
            OSError: *src* cannot be read.
        """
        stamps = []
        for path in [src, *(c for c in companions if os.path.isfile(c))]:
            stat = os.stat(path)
            stamps.append((os.path.basename(path), stat.st_size, stat.st_mtime_ns))
        return CachedArtifact.key(*parts, stamps)

    def current(self, src: str, project: str, key: str) -> Optional[str]:
        """*project*'s bake of *src* when it holds the source version *key*
        (its record says so), else None."""
        slot = self.store(project).peek(self.slot_key(src, project))
        if slot is None:
            return None
        return slot if self.read_info(slot).get(self.key_field) == key else None

    # ----------------------------------------------------------------- records
    def spell_source(self, src: str, project: str) -> str:
        """*src* as a bake records it: relative to the bakes' folder when it is
        in the project (a moved project keeps the mapping), else absolute."""
        if FileUtils.is_under(src, project):
            return os.path.relpath(src, self.cache_dir(project)).replace("\\", "/")
        return src.replace("\\", "/")

    def record(
        self, src: str, project: str, key: Optional[str] = None
    ) -> Dict[str, str]:
        """The record a bake of *src* carries: its source and, given, its key."""
        info = {self.source_field: self.spell_source(src, project)}
        if key is not None:
            info[self.key_field] = key
        return info

    def source_of(self, baked_path: str) -> Optional[str]:
        """The source *baked_path* was baked from (absolute), or None if it is
        not a bake.

        Read from the bake's record; only a file named as a bake is read (a
        browser asks this of every reference, and a user's scene is not a
        bake). A bake from before the record falls back to its
        :attr:`legacy_suffix` sidecar.
        """
        if not baked_path:
            return None
        recorded = (
            self.read_info(baked_path).get(self.source_field)
            if self.is_bake(baked_path)
            else None
        )
        if recorded:
            if not os.path.isabs(recorded):
                recorded = os.path.join(os.path.dirname(baked_path), recorded)
            return os.path.normpath(recorded)
        if not self.legacy_suffix:
            return None
        data = FileUtils.read_json(baked_path + self.legacy_suffix, default=None)
        return (data.get("source") or None) if isinstance(data, dict) else None

    # ------------------------------------------------------------------ writes
    def write(
        self,
        src: str,
        project: str,
        produce: Callable[[str, Dict[str, str]], Any],
        key: Optional[str] = None,
    ) -> str:
        """Bake *src* into its slot: ``produce(out, record)`` writes the bake,
        carrying *record* (:meth:`record`), to the scratch path *out*, which
        then replaces the slot in one move -- a killed bake never leaves a
        partial slot.

        Parameters:
            src: The source.
            project: Its project root.
            produce: The host's bake.
            key: The source version's content key (:meth:`source_key`); None
                records none (the next bake of the source replaces it).

        Returns:
            The slot.

        Raises:
            Whatever *produce* raises (its partial output is kept + logged);
            OSError when the bake cannot be moved into the slot.
        """
        store = self.store(project, create=True)
        slot_key = self.slot_key(src, project)
        info = self.record(src, project, key)
        got = store.get(slot_key, lambda out: produce(out, info), use_cache=False)
        try:
            return store.promote(slot_key, got.path)
        finally:
            got.scratch.cleanup()

    def rehome(
        self,
        baked_path: str,
        project: str,
        produce: Callable[[str, str, Dict[str, str]], Any],
    ) -> Optional[str]:
        """The project bake standing in for *baked_path*, a bake that lives
        elsewhere (temp, before bakes were project files), or None when it
        names no source.

        The project's slot for its source when that already holds a bake,
        else ``produce(baked_path, out, record)`` -- the host's copy carrying
        the source's record and no key -- written into it (:meth:`write`).
        """
        source = self.source_of(baked_path)
        if not source or not os.path.isfile(baked_path):
            return None
        existing = self.store(project, create=True).peek(self.slot_key(source, project))
        if existing is not None:
            return existing
        return self.write(
            source, project, lambda out, info: produce(baked_path, out, info)
        )

    # ------------------------------------------------------------------- purge
    def referenced_keys(
        self, project: str, extensions: Iterable[str], live: Iterable[str] = ()
    ) -> Optional[Set[str]]:
        """The slot keys any file of *extensions* under *project* names (read
        from its bytes: :meth:`FileDependencies.scan_references`; the bakes'
        own folder is not read), or a *live* session reference does; None
        when a file cannot be scanned."""
        pattern = rf"{re.escape(self.prefix)}({self.SLOT_KEY})\.".encode("ascii")
        used = FileDependencies.scan_references(
            project, pattern, extensions, skip=[self.cache_dir(project)]
        )
        if used is None:
            return None
        names = self.name_pattern
        for path in live:
            match = names.match(os.path.basename(str(path or "")))
            if match:
                used.add(match.group(1))
        return used

    def purge(
        self,
        project: str,
        extensions: Iterable[str],
        live: Iterable[str] = (),
        dry_run: bool = False,
    ) -> List[str]:
        """Remove the bakes in *project*'s cache that nothing references, with
        anything named for them.

        A bake is a project file a saved scene reads, so nothing ages it out;
        a source dropped from every scene leaves its bake behind until this
        runs. Every file of *extensions* in the project is scanned for the
        bake names (:meth:`referenced_keys`; no file is opened in its app),
        and so are the *live* session's references; a bake any of them names
        stays. A file that cannot be scanned stops the purge: a bake it
        references would otherwise go.

        Parameters:
            project: The project root.
            extensions: The scene types that can reference a bake.
            live: The open session's referenced files.
            dry_run: Report what would go without removing it.

        Returns:
            The paths removed (or that would be).
        """
        home = self.cache_dir(project)
        if not os.path.isdir(home):
            return []
        names = self.name_pattern
        entries = []
        for name in os.listdir(home):
            match = names.match(name)
            if match:
                entries.append((name, match.group(1)))
        if not entries:
            return []
        used = self.referenced_keys(project, extensions, live)
        if used is None:
            self.logger.warning(f"Not purging {home}: a scene could not be scanned.")
            return []
        removed = []
        for name, slot_key in entries:
            if slot_key in used:
                continue
            path = os.path.join(home, name)
            if not dry_run:
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except OSError as e:
                    self.logger.warning(f"Could not remove unused bake {path}: {e}")
                    continue
            removed.append(path)
        if removed:
            self.logger.info(
                ("Would remove " if dry_run else "Removed ")
                + f"{len(removed)} unused bake file(s) from {home}."
            )
        return removed
