# !/usr/bin/python
# coding=utf-8
"""Whole-root view over the ecosystem's preset stores.

A :class:`~pythontk.core_utils.presets.store.PresetStore` manages ONE tool's
presets; :class:`PresetLibrary` manages all of them at once: browse every store,
lock presets, group them into named collections, and export / import them as a
bundle -- the backup and the studio-sharing format are the same file. It is the
layer behind uitk's Preset Editor and behind any installer that ships a studio's
presets, so it is Qt-free and never imports a tool.

Discovery follows the UI Browser's lazy pattern: a store announces itself with a
``.domain`` marker in its own folder (written by ``PresetStore`` on first use),
the scan reads only file names and those markers, and a preset's metadata is read
from its sidecar when asked. Nothing here writes during a scan.

Layout under the root (``UserConfig.user_config_root()``)::

    .collections/<id>.json       one header per collection (never a shared index)
    .backups/auto-*.zip          automatic pre-import backups (newest kept)
    <pkg>/<tool>[/<mode>]/       a store: .domain, .active, payloads, .<name>.preset

Bundle (a ``.zip``)::

    collection.json              header: format, kind, id, name, version, packages
    presets/<key>/<payload>      the payload bytes, verbatim
    presets/<key>/.<name>.preset its sidecar
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from pythontk.core_utils.presets.store import (
    DOMAIN_MARKER,
    INFO_EXT,
    JSON_CODEC,
    Codec,
    PresetStore,
    _PresetStoreInternal,
)
from pythontk.core_utils.user_config import UserConfig

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = 1
BUNDLE_HEADER = "collection.json"
BUNDLE_ROOT = "presets"
COLLECTIONS_DIR = ".collections"
BACKUPS_DIR = ".backups"
#: Automatic backups kept; older ``auto-*.zip`` files are pruned. Manual backups
#: are never pruned.
AUTO_BACKUP_KEEP = 10
#: A collection id: lower-case ASCII words joined by single dashes. It names a
#: file under ``.collections/``, so an imported header's id must match it.
_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
#: Shell-style store-key patterns: one string (commas separate several) or many.
Patterns = Union[str, Iterable[str]]


def _opaque_load(_text: str) -> Any:
    raise ValueError(
        "this store's payload format needs its own codec; the library copies "
        "these files verbatim but cannot parse them"
    )


def _opaque_dump(_data: Any) -> str:
    raise ValueError("the library never re-serializes a payload")


@dataclass(frozen=True)
class PresetDomain:
    """One preset store as the library sees it (a folder under the root).

    Parameters:
        key: Folder path relative to the root, posix (``"mayatk/scene_exporter"``).
        path: The folder.
        ext: Payload extension (``".json"``, or a codec's own, e.g. ``".yaml"``).
        builtin_dir: The store's shipped read-only presets, when resolvable here.
    """

    key: str
    path: Path
    ext: str = ".json"
    builtin_dir: Optional[Path] = None

    @property
    def package(self) -> str:
        """First key segment: the owning package (``"mayatk"``) or shared area."""
        return self.key.split("/", 1)[0]

    def store(self) -> PresetStore:
        """A non-marking store over this folder (the library never writes a marker).

        Non-JSON payloads get an opaque codec: listing, copying, locking and
        tagging need no parsing, so a YAML store is fully manageable here without
        its codec -- only :meth:`PresetStore.load` / ``save`` would raise.
        """
        codec = (
            JSON_CODEC
            if self.ext == JSON_CODEC.ext
            else Codec(self.ext, _opaque_load, _opaque_dump)
        )
        return PresetStore(
            Path(self.key).name,
            builtin_dir=self.builtin_dir,
            user_dir=self.path,
            codec=codec,
            mark=False,
        )


@dataclass(frozen=True)
class PresetEntry:
    """One preset: an immutable record, the ``HandlerEntry`` of presets.

    Parameters:
        domain: The owning store's key.
        name: The preset name (file stem) -- the key every store API takes.
        tier: ``"user"`` or ``"builtin"``.
        path: The payload file.
        info: The sidecar metadata (read-only mapping; ``{}`` for a built-in).
    """

    domain: str
    name: str
    tier: str
    path: Path
    info: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), compare=False, hash=False
    )

    @property
    def id(self) -> Optional[str]:
        return self.info.get("id") or None

    @property
    def label(self) -> str:
        """The name as typed (file names lose punctuation), else the name."""
        return self.info.get("label") or self.name

    @property
    def collection(self) -> Optional[str]:
        return self.info.get("collection") or None

    @property
    def tags(self) -> Tuple[str, ...]:
        return tuple(self.info.get("tags") or ())

    @property
    def read_only(self) -> bool:
        return self.tier == "builtin" or bool(self.info.get("read_only"))

    @property
    def modified(self) -> Optional[float]:
        """Payload mtime (epoch seconds): right even after an older install saves."""
        try:
            return self.path.stat().st_mtime
        except OSError:
            return None


@dataclass
class ImportItem:
    """One row of an :class:`ImportPlan`. Change :attr:`action` before applying.

    Statuses and their allowed actions (first = default):

    * ``new`` -- ``add`` | ``skip``
    * ``identical`` -- ``skip``
    * ``update`` -- ``replace`` | ``skip`` (same preset id, not edited locally)
    * ``conflict`` -- ``skip`` | ``replace`` | ``keep_both`` (a different preset
      holds the name, or the local copy was edited since it was installed)
    * ``removed`` -- ``remove`` | ``untag`` | ``skip`` (a local member of the
      collection that the bundle no longer contains; ``untag`` is the default
      when it was edited locally)
    """

    domain: str
    name: str
    status: str
    action: str
    reason: str = ""
    member: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)

    ACTIONS = {
        "new": ("add", "skip"),
        "identical": ("skip",),
        "update": ("replace", "skip"),
        "conflict": ("skip", "replace", "keep_both"),
        "removed": ("remove", "untag", "skip"),
    }

    @property
    def label(self) -> str:
        return self.info.get("label") or self.name

    def choices(self) -> Tuple[str, ...]:
        """The actions valid for this item's status."""
        return self.ACTIONS[self.status]


@dataclass
class ImportPlan:
    """What importing a bundle would do; built by :meth:`PresetLibrary.plan_import`."""

    path: Path
    header: Dict[str, Any]
    items: List[ImportItem]
    warnings: List[str] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return self.header.get("kind", "backup")

    def counts(self) -> Dict[str, int]:
        """``{status: n}`` for a summary line."""
        out: Dict[str, int] = {}
        for item in self.items:
            out[item.status] = out.get(item.status, 0) + 1
        return out

    def pending(self) -> List[ImportItem]:
        """Items whose action changes something."""
        return [i for i in self.items if i.action != "skip"]


@dataclass
class ImportResult:
    """What :meth:`PresetLibrary.apply` did."""

    applied: Dict[str, int]
    domains: List[str]
    backup: Optional[Path] = None


class _PresetLibraryInternal(object):
    """Internal helpers for PresetLibrary."""

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()

    @staticmethod
    def _write_bytes(path: Path, data: bytes) -> None:
        """Write *data* to *path* atomically; bytes verbatim (no newline mapping)."""
        from pythontk.file_utils._file_utils import FileUtils

        path.parent.mkdir(parents=True, exist_ok=True)
        FileUtils.atomic_write(str(path), lambda part: Path(part).write_bytes(data))

    @staticmethod
    def _slug(text: str) -> str:
        """*text* in the :data:`_ID_PATTERN` grammar (``"Acme_Std!"`` -> ``"acme-std"``).

        Accents fold to ASCII (``"Ünïcode"`` -> ``"unicode"``); text with no
        ASCII left gives ``"collection"``.
        """
        import unicodedata

        ascii_text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore")
        slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.decode().lower()).strip("-")
        return slug or "collection"

    @staticmethod
    def _version_tuple(version: str) -> Tuple[int, ...]:
        parts = []
        for chunk in str(version).split("."):
            m = re.match(r"\d+", chunk)
            if not m:
                break
            parts.append(int(m.group()))
        return tuple(parts)

    @staticmethod
    def _installed_version(package: str) -> Optional[str]:
        try:
            from importlib.metadata import PackageNotFoundError, version
        except ImportError:  # pragma: no cover - py<3.8
            return None
        try:
            return version(package)
        except PackageNotFoundError:
            return None
        except Exception:
            return None

    @staticmethod
    def _safe_parts(parts: Iterable[str]) -> bool:
        """True when bundle path *parts* can only name a file under the root.

        A bundle comes from another machine, so its member names are untrusted:
        ``..``, a drive (``C:``), a backslash or a leading dot (the library's own
        ``.backups`` / ``.collections``) could otherwise steer a write out of a
        store folder -- the classic zip-slip.
        """
        for part in parts:
            if (
                not part
                or part in (".", "..")
                or part.startswith(".")
                or any(c in part for c in ("\\", ":", "\0"))
            ):
                return False
        return True

    @staticmethod
    def _stem_of_sidecar(filename: str) -> Optional[str]:
        if filename.startswith(".") and filename.endswith(INFO_EXT):
            return filename[1 : -len(INFO_EXT)] or None
        return None


class PresetLibrary(_PresetLibraryInternal):
    """Every preset store under one root: browse, lock, collect, back up, share.

    Parameters:
        root: The presets root. Defaults to ``UserConfig.user_config_root()`` --
            the same root every ``PresetStore`` and uitk ``PresetManager``
            resolves under (``$UITK_PRESETS_ROOT`` honoured).

    Example:
        >>> lib = PresetLibrary()
        >>> studio = lib.create_collection("Acme Standard")
        >>> lib.assign(lib.entries("mayatk/scene_exporter"), studio["id"])
        >>> lib.set_read_only(lib.members(studio["id"]))
        >>> lib.export("Acme Standard.presets.zip", collection=studio["id"])
        >>> plan = lib.plan_import("Acme Standard.presets.zip")  # elsewhere
        >>> lib.apply(plan)
    """

    def __init__(self, root: Optional[Union[str, os.PathLike]] = None):
        self._root = Path(root) if root else None

    @property
    def root(self) -> Path:
        return self._root or Path(UserConfig.user_config_root())

    # ------------------------------------------------------------------ scan
    def domains(
        self, inc: Optional[Patterns] = None, exc: Optional[Patterns] = None
    ) -> List[PresetDomain]:
        """Every preset store under the root, sorted by key. Writes nothing.

        A folder is a store when it holds a ``.domain`` marker or a preset
        sidecar -- or, for a folder written before markers existed, a JSON
        payload with a ``_meta`` block (the codebase's existing test for "a
        preset file"). That keeps settings files such as ``shots/prefs.json`` out.
        Dot-folders (``.collections``, ``.backups``) are never descended into.

        Parameters:
            inc: Keep only stores under a folder matching one of these
                shell-style patterns (see :meth:`in_scope`).
            exc: Drop stores under a folder matching one of these.
        """
        root = self.root
        if not root.is_dir():
            return []
        found: List[PresetDomain] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            here = Path(dirpath)
            if here == root:
                continue
            domain = self._domain_at(here, filenames)
            if domain is not None:
                found.append(domain)
        if inc or exc:
            kept = set(self.in_scope([d.key for d in found], inc, exc))
            found = [d for d in found if d.key in kept]
        return sorted(found, key=lambda d: d.key.lower())

    @staticmethod
    def in_scope(
        keys: Iterable[str],
        inc: Optional[Patterns] = None,
        exc: Optional[Patterns] = None,
    ) -> List[str]:
        """The store *keys* inside the scope *inc* / *exc* describe, in order.

        A pattern names a FOLDER and takes every store under it: it is matched
        (shell-style, case-insensitive) against a key and each of its parent
        folders, so ``"mayatk"`` is every mayatk store, ``"mayatk/scene_*"``
        one tool's, and ``"*/fbx_*"`` one kind of tool in every package. A
        pattern is never a bare prefix -- ``"maya"`` matches no ``mayatk`` key.
        A key is kept when some folder of it matches *inc* (or *inc* is empty)
        and none matches *exc*. Strings may hold several comma-separated
        patterns.
        """
        from pythontk.iter_utils._iter_utils import IterUtils

        chains = []
        for key in keys:
            parts = key.split("/")
            chains.append(tuple("/".join(parts[:i]) for i in range(1, len(parts) + 1)))
        kept = IterUtils.filter_list(
            chains, inc, exc, nested_as_unit=True, ignore_case=True
        )
        return [chain[-1] for chain in kept]

    def domain(self, key: str) -> PresetDomain:
        """The store at *key* (whether or not the folder exists yet)."""
        path = self.root.joinpath(*[p for p in key.split("/") if p])
        filenames = os.listdir(path) if path.is_dir() else []
        return self._domain_at(path, filenames, force=True)

    def _domain_at(
        self, path: Path, filenames: Iterable[str], force: bool = False
    ) -> Optional[PresetDomain]:
        filenames = list(filenames)
        key = path.relative_to(self.root).as_posix()
        if DOMAIN_MARKER in filenames:
            marker = _PresetStoreInternal._read_json(path / DOMAIN_MARKER)
            ext = marker.get("ext") or JSON_CODEC.ext
            builtin = PresetStore.resolve_builtin_spec(marker.get("builtin"))
            return PresetDomain(key, path, ext, builtin)
        stems = {s for s in map(self._stem_of_sidecar, filenames) if s is not None}
        payloads = [f for f in filenames if not f.startswith(".")]
        if stems:
            exts = [Path(f).suffix for f in payloads if Path(f).stem in stems]
            return PresetDomain(key, path, exts[0] if exts else JSON_CODEC.ext)
        for f in payloads:
            if not f.endswith(JSON_CODEC.ext):
                continue
            data = _PresetStoreInternal._read_json(path / f)
            if "_meta" in data:
                return PresetDomain(key, path, JSON_CODEC.ext)
        return PresetDomain(key, path) if force else None

    def entries(
        self,
        key: Optional[str] = None,
        *,
        builtin: bool = True,
        collection: Optional[str] = None,
        inc: Optional[Patterns] = None,
        exc: Optional[Patterns] = None,
    ) -> List[PresetEntry]:
        """Presets of one store (*key*) or of all, user tier first then built-ins.

        Parameters:
            key: Limit to one store.
            builtin: Include shipped built-ins not shadowed by a user preset.
            collection: Limit to members of this collection id (user tier only).
            inc / exc: Limit to the stores in this scope (see :meth:`in_scope`).
                An out-of-scope store's sidecars are never read.
        """
        if key:
            domains = [self.domain(key)]
            if (inc or exc) and not self.in_scope([key], inc, exc):
                domains = []
        else:
            domains = self.domains(inc, exc)
        out: List[PresetEntry] = []
        for domain in domains:
            store = domain.store()
            user = store.list("user")
            for name in user:
                entry = self._entry(domain, store, name, "user")
                if collection is None or entry.collection == collection:
                    out.append(entry)
            if builtin and collection is None and domain.builtin_dir:
                user_set = set(user)
                for name in store.list("builtin"):
                    if name not in user_set:
                        out.append(self._entry(domain, store, name, "builtin"))
        return out

    @staticmethod
    def _entry(
        domain: PresetDomain, store: PresetStore, name: str, tier: str
    ) -> PresetEntry:
        info = store.info(name) if tier == "user" else {}
        return PresetEntry(
            domain.key, name, tier, store.path(name, tier), MappingProxyType(info)
        )

    def entry(self, key: str, name: str) -> Optional[PresetEntry]:
        """The preset *name* of store *key* (user shadows built-in), or ``None``."""
        domain = self.domain(key)
        store = domain.store()
        tier = store.source(name)
        return self._entry(domain, store, name, tier) if tier else None

    def find(self, preset_id: str) -> Optional[PresetEntry]:
        """The user preset with sidecar id *preset_id*, or ``None``."""
        for entry in self.entries(builtin=False):
            if entry.id == preset_id:
                return entry
        return None

    # ------------------------------------------------------------ batch edits
    def _user_entries(self, entries: Iterable[PresetEntry]) -> List[PresetEntry]:
        return [e for e in entries if e.tier == "user"]

    def _stores(self) -> Callable[[str], PresetStore]:
        """A per-operation ``key -> store`` getter: one folder read per store, not
        per preset (:meth:`domain` lists the folder and reads its marker)."""
        cache: Dict[str, PresetStore] = {}

        def store_of(key: str) -> PresetStore:
            if key not in cache:
                cache[key] = self.domain(key).store()
            return cache[key]

        return store_of

    def set_read_only(self, entries: Iterable[PresetEntry], flag: bool = True) -> int:
        """Lock (or unlock) user presets. Built-ins are skipped. Returns the count."""
        count = 0
        store_of = self._stores()
        for e in self._user_entries(entries):
            store = store_of(e.domain)
            store.ensure_info(e.name)
            store.set_info(e.name, read_only=True if flag else None)
            count += 1
        return count

    def assign(self, entries: Iterable[PresetEntry], collection: Optional[str]) -> int:
        """Tag user presets with *collection* (``None`` untags). Returns the count.

        A preset belongs to at most one collection -- that is what keeps a
        collection update unambiguous; use ``tags`` for any other grouping.
        """
        if collection is not None and self.collection(collection) is None:
            raise KeyError(f"no collection {collection!r}")
        count = 0
        store_of = self._stores()
        for e in self._user_entries(entries):
            store = store_of(e.domain)
            store.ensure_info(e.name)
            # The publish baseline belongs to the collection it was taken for.
            origin = e.info.get("origin_hash") if collection == e.collection else None
            store.set_info(e.name, collection=collection, origin_hash=origin)
            count += 1
        return count

    def set_tags(self, entries: Iterable[PresetEntry], tags: Iterable[str]) -> int:
        """Replace the tags of user presets. Returns the count."""
        clean = sorted({str(t).strip() for t in tags if str(t).strip()})
        count = 0
        store_of = self._stores()
        for e in self._user_entries(entries):
            store = store_of(e.domain)
            store.ensure_info(e.name)
            store.set_info(e.name, tags=clean or None)
            count += 1
        return count

    def duplicate(self, entry: PresetEntry, name: Optional[str] = None) -> PresetEntry:
        """Copy a preset (user or built-in) to a new, unlocked, untagged user preset.

        The "duplicate to edit" flow for a locked or shipped preset. *name*
        defaults to ``"<label> copy"`` (made unique).
        """
        domain = self.domain(entry.domain)
        store = domain.store()
        new = name or store.unique_name(f"{entry.label} copy")
        if store.exists(new):
            raise ValueError(f"a preset named {new!r} already exists in {entry.domain}")
        self._write_bytes(store.path(new, "user"), entry.path.read_bytes())
        info = store._new_info(new)
        if entry.tags:
            info["tags"] = list(entry.tags)
        store.write_info(new, info)
        return self._entry(domain, store, new, "user")

    def rename(self, entry: PresetEntry, new_name: str) -> bool:
        """Rename a user preset (refused when locked or the name is taken)."""
        if entry.tier != "user":
            return False
        return self.domain(entry.domain).store().rename(entry.name, new_name)

    def delete(self, entries: Iterable[PresetEntry], *, force: bool = False) -> int:
        """Delete user presets (locked ones only with *force*). Returns the count."""
        count = 0
        store_of = self._stores()
        for e in self._user_entries(entries):
            if store_of(e.domain).delete(e.name, force=force):
                count += 1
        return count

    # ------------------------------------------------------------ collections
    @property
    def _collections_dir(self) -> Path:
        return self.root / COLLECTIONS_DIR

    def collections(self) -> List[Dict[str, Any]]:
        """Every collection header, sorted by name."""
        d = self._collections_dir
        if not d.is_dir():
            return []
        out = []
        for f in d.glob("*.json"):
            header = _PresetStoreInternal._read_json(f)
            if header.get("id"):
                out.append(header)
        return sorted(out, key=lambda h: str(h.get("name", "")).lower())

    def collection(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """The header of collection *collection_id*, or ``None``."""
        path = self._collections_dir / f"{collection_id}.json"
        header = _PresetStoreInternal._read_json(path)
        return header or None

    def _write_collection(self, header: Dict[str, Any]) -> None:
        self._collections_dir.mkdir(parents=True, exist_ok=True)
        _PresetStoreInternal._atomic_write_text(
            self._collections_dir / f"{header['id']}.json",
            json.dumps(header, indent=4, sort_keys=True),
        )

    def create_collection(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create an empty collection named *name*.

        The name is what people pick by, so it must be unique here (ignoring
        case). The id is the lineage an update follows -- a slug of the name
        plus a random suffix, so two authors' same-named collections never share
        one (an install would otherwise read as an update of the other and
        offer to remove its presets).

        Raises:
            ValueError: *name* is empty or another collection already has it.
        """
        name = str(name).strip()
        if not name:
            raise ValueError("a collection needs a name")
        if self.collection_named(name) is not None:
            raise ValueError(f"a collection named {name!r} already exists")
        header = {
            "id": f"{self._slug(name)}-{uuid.uuid4().hex[:8]}",
            "name": name,
            "description": description,
            "version": 0,
            "author": _PresetStoreInternal._author(),
            "created": _PresetStoreInternal._now(),
        }
        self._write_collection(header)
        return header

    def collection_named(
        self, name: str, exclude: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """The header of the collection called *name* -- ignoring case and edge
        spaces, the rule that makes a name taken -- other than the one with id
        *exclude*, or ``None``. A form checks a name with it before submitting."""
        wanted = str(name).strip().lower()
        for header in self.collections():
            if header.get("id") == exclude:
                continue
            if str(header.get("name", "")).strip().lower() == wanted:
                return header
        return None

    def update_collection(self, collection_id: str, **fields: Any) -> Dict[str, Any]:
        """Merge *fields* (``name``, ``description``) into a collection header.

        Raises:
            KeyError: no such collection.
            ValueError: a new ``name`` is empty or another collection has it.
        """
        header = self.collection(collection_id)
        if header is None:
            raise KeyError(f"no collection {collection_id!r}")
        if "name" in fields:
            name = str(fields["name"]).strip()
            if not name:
                raise ValueError("a collection needs a name")
            if self.collection_named(name, exclude=collection_id) is not None:
                raise ValueError(f"a collection named {name!r} already exists")
            fields["name"] = name
        header.update({k: v for k, v in fields.items() if k != "id"})
        self._write_collection(header)
        return header

    def members(self, collection_id: str) -> List[PresetEntry]:
        """The user presets tagged with *collection_id*."""
        return self.entries(builtin=False, collection=collection_id)

    def is_modified(self, entry: PresetEntry) -> bool:
        """True when a collection member's payload changed since install/publish.

        A member without a recorded ``origin_hash`` counts as modified: with no
        baseline, "unchanged" cannot be proven, and the safe answer is the one
        that never deletes or overwrites the user's work.
        """
        origin = entry.info.get("origin_hash")
        if not origin:
            return True
        try:
            return self.hash_bytes(entry.path.read_bytes()) != origin
        except OSError:
            return True

    def delete_collection(
        self, collection_id: str, *, delete_members: bool = False
    ) -> Dict[str, int]:
        """Remove a collection header and untag (or delete) its members.

        With *delete_members*, members unchanged since install are deleted (even
        if locked) and edited ones are untagged, never lost.
        """
        counts = {"deleted": 0, "untagged": 0}
        store_of = self._stores()
        for e in self.members(collection_id):
            store = store_of(e.domain)
            if delete_members and not self.is_modified(e):
                if store.delete(e.name, force=True):
                    counts["deleted"] += 1
                continue
            store.set_info(e.name, collection=None, origin_hash=None)
            counts["untagged"] += 1
        try:
            (self._collections_dir / f"{collection_id}.json").unlink()
        except FileNotFoundError:
            pass
        return counts

    # ----------------------------------------------------------------- export
    def export(
        self,
        path: Union[str, os.PathLike],
        *,
        collection: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        name: Optional[str] = None,
    ) -> Path:
        """Write a bundle of user presets to *path* (a ``.zip``).

        With *collection*: its members only, the header's ``version`` bumped,
        and each member's ``origin_hash`` stamped with the published content (so
        a later local edit reads as one) -- both only once the bundle is written.
        Without: a backup of every user preset (optionally only the stores in
        *keys*), tags and ids kept as they are. Every exported preset is given an
        id first, so a later import of the same bundle recognises it.
        """
        if collection is not None:
            header = self.collection(collection)
            if header is None:
                raise KeyError(f"no collection {collection!r}")
            entries = self.members(collection)
        else:
            header = {"id": None, "name": name or "Backup"}
            entries = self.entries(builtin=False)
        if keys is not None:
            keyset = set(keys)
            entries = [e for e in entries if e.domain in keyset]
        return self._write_bundle(Path(path), entries, header, collection is not None)

    def _write_bundle(
        self,
        path: Path,
        entries: List[PresetEntry],
        header: Dict[str, Any],
        is_collection: bool,
    ) -> Path:
        """Zip *entries* under *header*; a collection's version + baselines after."""
        store_of = self._stores()
        members: List[Tuple[str, bytes]] = []
        published: List[Tuple[PresetStore, str, bytes]] = []
        for e in entries:
            store = store_of(e.domain)
            payload = e.path.read_bytes()
            info = store.ensure_info(e.name)
            base = f"{BUNDLE_ROOT}/{e.domain}"
            members.append((f"{base}/{e.path.name}", payload))
            members.append(
                (
                    f"{base}/{_PresetStoreInternal._sidecar_name(e.path.stem)}",
                    json.dumps(info, indent=4, sort_keys=True).encode("utf-8"),
                )
            )
            published.append((store, e.name, payload))

        version = header.get("version")
        if is_collection:
            version = int(version or 0) + 1
        bundle_header = {
            "format": BUNDLE_FORMAT,
            "kind": "collection" if is_collection else "backup",
            "id": header.get("id"),
            "name": header.get("name"),
            "description": header.get("description", ""),
            "version": version,
            "author": header.get("author") or _PresetStoreInternal._author(),
            "created": header.get("created"),
            "exported": _PresetStoreInternal._now(),
            "count": len(entries),
            "packages": self._package_versions({e.domain for e in entries}),
        }

        def _write(part: str) -> None:
            with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(BUNDLE_HEADER, json.dumps(bundle_header, indent=4))
                for arcname, data in members:
                    zf.writestr(arcname, data)

        from pythontk.file_utils._file_utils import FileUtils

        path.parent.mkdir(parents=True, exist_ok=True)
        FileUtils.atomic_write(str(path), _write)
        if is_collection:
            for store, name, payload in published:
                store.set_info(name, origin_hash=self.hash_bytes(payload))
            self._write_collection({**header, "version": version})
        return path

    def _package_versions(self, keys: Iterable[str]) -> Dict[str, str]:
        """Installed versions of the packages owning *keys*, plus pythontk."""
        out: Dict[str, str] = {}
        for package in {"pythontk", *(k.split("/", 1)[0] for k in keys)}:
            version = self._installed_version(package)
            if version:
                out[package] = version
        return out

    def backup(
        self, path: Optional[Union[str, os.PathLike]] = None, *, reason: str = "manual"
    ) -> Optional[Path]:
        """Back up every user preset; ``None`` when there is nothing to back up.

        Without *path* the bundle goes to ``<root>/.backups/`` under a name no
        earlier backup holds (two in one second must not overwrite each other --
        the first may hold the only copy of what the second action deleted). A
        manual backup is kept forever; an automatic one (any other *reason*, e.g.
        ``"import"``) is named ``auto-*`` and only the newest
        :data:`AUTO_BACKUP_KEEP` are kept.
        """
        entries = self.entries(builtin=False)
        if not entries:
            return None
        auto = reason != "manual"
        if path is None:
            stamp = _PresetStoreInternal._now().replace(":", "").replace("-", "")
            prefix = "auto-" if auto else ""
            stem = f"{prefix}{stamp}-{self._slug(reason)}"
            path = self.root / BACKUPS_DIR / f"{stem}.zip"
            n = 2
            while path.exists():
                path = self.root / BACKUPS_DIR / f"{stem}-{n}.zip"
                n += 1
        written = self._write_bundle(
            Path(path), entries, {"id": None, "name": f"Backup ({reason})"}, False
        )
        if auto:
            self._prune_backups()
        return written

    def backups(self) -> List[Path]:
        """Bundles in ``<root>/.backups/``, newest first."""
        d = self.root / BACKUPS_DIR
        if not d.is_dir():
            return []
        return sorted(d.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)

    def _prune_backups(self) -> None:
        autos = [p for p in self.backups() if p.name.startswith("auto-")]
        for stale in autos[AUTO_BACKUP_KEEP:]:
            try:
                stale.unlink()
            except OSError as e:
                logger.debug("PresetLibrary: could not prune %s: %s", stale, e)

    # ----------------------------------------------------------------- import
    @staticmethod
    def read_header(path: Union[str, os.PathLike]) -> Dict[str, Any]:
        """The ``collection.json`` header of bundle *path*.

        The header is untrusted (it comes from another machine): its collection
        id names a file under ``.collections/``, so only the slug form
        :meth:`create_collection` makes is accepted.

        Raises:
            ValueError: not a preset bundle, a malformed or unsafe header, or
                written by a newer format.
        """
        try:
            with zipfile.ZipFile(path) as zf:
                header = json.loads(zf.read(BUNDLE_HEADER).decode("utf-8"))
        except (KeyError, zipfile.BadZipFile, ValueError) as e:
            raise ValueError(f"{path} is not a preset bundle: {e}") from e
        if not isinstance(header, dict):
            raise ValueError(f"{path} has a malformed header")
        try:
            fmt = int(header.get("format") or 0)
            if header.get("version") is not None:
                header["version"] = int(header["version"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path} has a malformed header: {e}") from e
        if fmt > BUNDLE_FORMAT:
            raise ValueError(
                f"{path} was written by a newer version (format "
                f"{header.get('format')}); update pythontk to import it"
            )
        cid = header.get("id")
        if cid is not None and not _ID_PATTERN.fullmatch(str(cid)):
            raise ValueError(f"{path} has an unsafe collection id {cid!r}")
        return header

    def plan_import(self, path: Union[str, os.PathLike]) -> ImportPlan:
        """Compare bundle *path* against the local presets; change nothing.

        Returns an :class:`ImportPlan` whose items carry a status and a default
        action (see :class:`ImportItem`); edit actions, then :meth:`apply`.
        """
        path = Path(path)
        header = self.read_header(path)
        store_of = self._stores()
        items: List[ImportItem] = []
        incoming_ids = set()
        incoming_names = set()
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            for member in sorted(names):
                parts = member.split("/")
                if len(parts) < 3 or parts[0] != BUNDLE_ROOT:
                    continue
                filename = parts[-1]
                if not filename or filename.startswith("."):
                    continue
                if not self._safe_parts(parts[1:]):
                    logger.warning("PresetLibrary: skipped unsafe member %r", member)
                    continue
                key = "/".join(parts[1:-1])
                stem = Path(filename).stem
                sidecar = "/".join(
                    [BUNDLE_ROOT, key, _PresetStoreInternal._sidecar_name(stem)]
                )
                info: Dict[str, Any] = {}
                if sidecar in names:
                    try:
                        loaded = json.loads(zf.read(sidecar).decode("utf-8"))
                        info = loaded if isinstance(loaded, dict) else {}
                    except ValueError:
                        info = {}
                if info.get("id"):
                    incoming_ids.add(info["id"])
                incoming_names.add((key, stem))
                store = self._import_store(store_of, key, member)
                items.append(
                    self._classify(store, key, stem, zf.read(member), info, member)
                )

        if header.get("kind") == "collection" and header.get("id"):
            for e in self.members(header["id"]):
                if (e.domain, e.name) in incoming_names or (
                    e.id and e.id in incoming_ids
                ):
                    continue
                modified = self.is_modified(e)
                items.append(
                    ImportItem(
                        e.domain,
                        e.name,
                        "removed",
                        "untag" if modified else "remove",
                        "no longer in the collection"
                        + (" (edited locally)" if modified else ""),
                        None,
                        dict(e.info),
                    )
                )
        return ImportPlan(path, header, items, self._import_warnings(header))

    @staticmethod
    def _import_store(
        store_of: Callable[[str], PresetStore], key: str, member: Optional[str]
    ) -> PresetStore:
        """The store bundle *member* of store *key* is compared with and written to.

        A folder without a ``.domain`` marker -- a fresh root, or a tool never
        run on this machine -- cannot say its payload format, and the scan's
        default (``.json``) would land a YAML payload as ``*.json``, which its
        own store never lists. There the member's own extension is the format;
        a marker, when present, is the tool's word and wins.

        Parameters:
            store_of: The operation's ``key -> store`` getter (:meth:`_stores`).
            key: The store key.
            member: The bundle member name, or ``None`` for an item that has
                none (a ``removed`` one acts on the local preset).

        Returns:
            The store to classify and install the member through.
        """
        store = store_of(key)
        ext = Path(member).suffix if member else ""
        if not ext or ext == store.ext or (store.user_dir / DOMAIN_MARKER).is_file():
            return store
        return PresetDomain(key, store.user_dir, ext, store.builtin_dir).store()

    def _classify(
        self,
        store: PresetStore,
        key: str,
        name: str,
        payload: bytes,
        info: Dict[str, Any],
        member: str,
    ) -> ImportItem:
        local_path = store.path(name, "user")
        if not local_path.is_file():
            reason = (
                "replaces the built-in of this name for you"
                if store.source(name) == "builtin"
                else ""
            )
            return ImportItem(key, name, "new", "add", reason, member, info)
        local = local_path.read_bytes()
        local_info = store.info(name)
        if local == payload:
            # Same content: nothing of the user's can be lost, whatever the ids
            # say -- only the metadata (lock, collection, tags) may differ.
            same_meta = all(
                local_info.get(k) == info.get(k)
                for k in ("read_only", "collection", "tags", "label")
            )
            status, action = (
                ("identical", "skip") if same_meta else ("update", "replace")
            )
            return ImportItem(key, name, status, action, "", member, info)
        same_id = bool(info.get("id")) and local_info.get("id") == info.get("id")
        origin = local_info.get("origin_hash")
        if same_id and origin and self.hash_bytes(local) == origin:
            return ImportItem(key, name, "update", "replace", "", member, info)
        reason = (
            "you edited this preset since it was installed"
            if same_id
            else "a different preset already has this name"
        )
        return ImportItem(key, name, "conflict", "skip", reason, member, info)

    def _import_warnings(self, header: Dict[str, Any]) -> List[str]:
        warnings = []
        for package, theirs in sorted((header.get("packages") or {}).items()):
            mine = self._installed_version(package)
            if mine and self._version_tuple(mine) < self._version_tuple(theirs):
                warnings.append(
                    f"Made with {package} {theirs}; you have {mine}. Settings "
                    "added since may not apply until you update."
                )
        if header.get("kind") == "collection" and header.get("id"):
            installed = self.collection(header["id"])
            namesake = (
                None if installed else self.collection_named(header.get("name") or "")
            )
            if namesake:
                warnings.append(
                    f"You already have a different collection named "
                    f"{namesake.get('name')}; this one is installed beside it."
                )
            if installed and int(installed.get("version") or 0) > (
                header.get("version") or 0
            ):
                warnings.append(
                    f"You have version {installed.get('version')} of "
                    f"{installed.get('name')}; this bundle is version "
                    f"{header.get('version')}."
                )
        return warnings

    def apply(self, plan: ImportPlan, *, backup: bool = True) -> ImportResult:
        """Carry out *plan*'s actions. Backs up first when anything changes.

        Every action is checked against its item's :meth:`ImportItem.choices`
        before anything is written, so a bad plan fails whole rather than half
        applied. A collection bundle also installs (or updates) its header, so
        the collection shows up and a later version can be imported over it.

        Raises:
            ValueError: an item's action is not one its status allows.
        """
        pending = plan.pending()
        applied: Dict[str, int] = {}
        if not pending:
            return ImportResult(applied, [])
        invalid = [i for i in pending if i.action not in i.choices()]
        if invalid:
            raise ValueError(
                "invalid import action(s): "
                + ", ".join(f"{i.domain}/{i.name}: {i.action!r}" for i in invalid)
            )
        backup_path = self.backup(reason="import") if backup else None
        is_collection = plan.kind == "collection" and plan.header.get("id")
        store_of = self._stores()
        touched = set()
        with zipfile.ZipFile(plan.path) as zf:
            for item in pending:
                store = self._import_store(store_of, item.domain, item.member)
                if item.action in ("add", "replace"):
                    payload = zf.read(item.member)
                    self._install(store, item.name, payload, item.info, plan.header)
                elif item.action == "keep_both":
                    payload = zf.read(item.member)
                    suffix = plan.header.get("name") or "imported"
                    new = store.unique_name(f"{item.label} ({suffix})")
                    self._write_bytes(store.path(new, "user"), payload)
                    info = store._new_info(new)
                    if item.info.get("tags"):
                        info["tags"] = list(item.info["tags"])
                    store.write_info(new, info)
                elif item.action == "remove":
                    store.delete(item.name, force=True)
                else:  # "untag" -- the only choice left (validated above)
                    store.set_info(item.name, collection=None, origin_hash=None)
                applied[item.action] = applied.get(item.action, 0) + 1
                touched.add(item.domain)
        if is_collection:
            header = {
                k: plan.header.get(k)
                for k in ("id", "name", "description", "version", "author", "created")
            }
            header["installed"] = _PresetStoreInternal._now()
            header["source"] = plan.path.name
            self._write_collection(header)
        return ImportResult(applied, sorted(touched), backup_path)

    def _install(
        self,
        store: PresetStore,
        name: str,
        payload: bytes,
        info: Dict[str, Any],
        header: Dict[str, Any],
    ) -> None:
        """Write an incoming payload + its sidecar over whatever is there."""
        self._write_bytes(store.path(name, "user"), payload)
        info = dict(info)
        info.setdefault("id", uuid.uuid4().hex)
        info.setdefault("label", name)
        if header.get("kind") == "collection" and header.get("id"):
            info["collection"] = header["id"]
            info["origin_hash"] = self.hash_bytes(payload)
        store.write_info(name, info)

    # -------------------------------------------------------------- upkeep
    def clean_up(self) -> Dict[str, int]:
        """Remove sidecars whose preset is gone, and empty folders.

        Stray sidecars are what an older install leaves when it deletes or
        renames a preset (it doesn't know about them).
        """
        counts = {"sidecars": 0, "folders": 0}
        root = self.root
        if not root.is_dir():
            return counts
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            here = Path(dirpath)
            rel_parts = here.relative_to(root).parts
            if any(p.startswith(".") for p in rel_parts):
                continue
            payload_stems = {Path(f).stem for f in filenames if not f.startswith(".")}
            for f in filenames:
                stem = self._stem_of_sidecar(f)
                if stem is not None and stem not in payload_stems:
                    try:
                        (here / f).unlink()
                        counts["sidecars"] += 1
                    except OSError:
                        pass
            if here != root:
                try:
                    if not any(here.iterdir()):
                        here.rmdir()
                        counts["folders"] += 1
                except OSError:
                    pass
        return counts
