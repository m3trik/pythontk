# !/usr/bin/python
# coding=utf-8
"""Qt-free, zero-dependency named-preset *store* for the ecosystem.

Where :class:`UserConfig` resolves a *single* config doc (deep-merged over a
default), a :class:`PresetStore` manages a *collection of named presets* across
two tiers:

* **built-in** — read-only presets shipped in the repo next to the module (a
  ``presets/`` dir by convention; pass any dir, or ``None`` to skip the tier).
* **user** — writable presets under :func:`user_config_root` — the *same*
  consolidated root uitk's ``PresetManager`` uses, so the headless and GUI paths
  resolve to one location.

A user preset **shadows** a built-in of the same name (it replaces, not merges —
"duplicate to edit" covers tweaking a shipped default), so :meth:`list` shows each
name once and :meth:`load` returns the user copy when present.

This is the storage/resolution SSoT: it deals only in plain JSON dicts and never
imports Qt, so headless engines (e.g. photogrammetry runners in Metashape's
bundled Python 3.9) use it directly. uitk's ``PresetManager`` layers Qt
widget-state (de)serialization + combo wiring on top of an instance of this
class, so both front-ends share one set of directories, names, and rules.

Management metadata (a stable id, a lock, a collection tag, ...) lives in a
per-preset sidecar ``.<name>.preset`` beside the payload, never in the payload.
A payload is handed verbatim to its tool -- some splat it into keyword arguments
-- and older installs of this package read the same folders (pythontk is
installed once per host Python, so Maya and Blender on one machine routinely run
different versions against one store). A key added to the payload would reach
every one of those readers; a sidecar reaches none of them.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from pythontk.core_utils.user_config import UserConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Codec:
    """Pluggable (de)serialiser for a :class:`PresetStore`'s on-disk format.

    *load* parses file text into a dict, *dump* renders a dict back to text, and
    *ext* is the file extension (with leading dot).  The default
    :data:`JSON_CODEC` preserves the historical JSON behaviour; a caller needing
    another format (e.g. mayatk's YAML behavior templates) injects its own, so
    pythontk itself stays dependency-free.
    """

    ext: str
    load: Callable[[str], Any]
    dump: Callable[[Any], str]

    def __post_init__(self):
        # Honour the documented "with leading dot" contract leniently: a caller
        # injecting ``Codec("yaml", …)`` gets ``.yaml`` rather than a dotless
        # extension that would silently break path-building / discovery globs.
        if self.ext and not self.ext.startswith("."):
            object.__setattr__(self, "ext", "." + self.ext)


JSON_CODEC = Codec(
    ext=".json",
    load=json.loads,
    dump=lambda data: json.dumps(data, indent=4),
)

# Sidecar (in ``user_dir``) recording the last-selected preset name, so a GUI
# can restore the *active* preset across sessions and a headless runner can
# honour "last used". A dotfile with no ``.json`` extension so the ``*.json``
# discovery glob never picks it up (same convention as ``.migrated``).
ACTIVE_SENTINEL = ".active"

# Per-preset management sidecar: ``.<stem>.preset`` beside ``<stem><ext>``. Always
# JSON whatever the payload codec, dot-prefixed and without a payload extension so
# no discovery glob picks it up. Not a hidden file on Windows on purpose: a plain
# ``open(path, "w")`` on a hidden file raises there, and ``os.replace`` onto one
# silently drops the attribute -- the dot prefix is the whole convention, as for
# ``.active``.
INFO_EXT = ".preset"

# Folder marker announcing a preset store to ``PresetLibrary``'s scan: what the
# payload extension is and where the shipped built-ins live. Written by the store
# itself (see :meth:`PresetStore._mark`), so a scan finds every store without
# importing the tool that owns it.
DOMAIN_MARKER = ".domain"

# The BUILT-IN presets hidden from their tool's dropdown (``{"names": [<stem>]}``,
# in ``user_dir``). A built-in has no sidecar to carry the flag, and a hide is the
# user's view of a shipped file, so it lives beside ``.active``; a user preset
# carries its own flag in its sidecar (``hidden``), which moves with a rename and
# goes with a delete -- in older installs too, which keep a sidecar's unknown
# keys. Dot-prefixed without a payload extension: no discovery glob matches it.
HIDDEN_BUILTINS = ".hidden"


class PresetReadOnlyError(PermissionError):
    """Raised when saving over a locked (read-only) user preset.

    A ``PermissionError`` so an existing ``except OSError`` around a save still
    catches it. Read-only is a guard against accidents, not a security boundary:
    the lock is a flag in the preset's sidecar that the Preset Editor can clear.
    """


class _PresetStoreInternal(object):
    """Internal helpers for PresetStore."""

    # ``"<user_dir>|<root>"`` of every store whose ``.domain`` marker this process
    # has already written or confirmed -- the marker is checked once per process,
    # not on every ``list()`` (uitk rebuilds a store per access). The root is part
    # of the token because a redirected root (``$UITK_PRESETS_ROOT``) changes
    # whether the folder is inside it at all.
    _marked_dirs: Set[str] = set()

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        """Write *text* to *path* atomically (temp file + ``os.replace``).

        Preset files and the ``.active`` sidecar are read by other processes —
        another DCC session applying its startup preset while this one saves — and
        a plain ``write_text`` lets such a reader see a torn/partial file (and a
        crash mid-write corrupt an existing preset permanently). Delegates to
        ``FileUtils.atomic_write_text``; imported lazily because ``file_utils``
        itself imports from ``core_utils`` (module-level would risk a cycle).
        """
        from pythontk.file_utils._file_utils import FileUtils

        FileUtils.atomic_write_text(path, text)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        """Parse a small JSON sidecar; ``{}`` when missing, unreadable or not a dict."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _sidecar_name(stem: str) -> str:
        """File name of the management sidecar for payload stem *stem*."""
        return f".{stem}{INFO_EXT}"

    @staticmethod
    def _meta_description(path: Path, load: Callable[[str], Any]) -> str:
        """The ``_meta.description`` a shipped payload carries, or ``""``.

        Parameters:
            path: The payload file.
            load: The payload codec's parser.

        Returns:
            The description, stripped; ``""`` when the file is unreadable, not
            parseable by *load* (an opaque codec raises for every file) or says
            nothing.
        """
        try:
            data = load(path.read_text(encoding="utf-8"))
        except Exception:  # any codec's parse error: a description is optional
            return ""
        meta = data.get("_meta") if isinstance(data, dict) else None
        text = meta.get("description") if isinstance(meta, dict) else None
        return text.strip() if isinstance(text, str) else ""

    @staticmethod
    def _now() -> str:
        """UTC timestamp, second precision, ISO-8601 with a ``Z`` suffix."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _author() -> str:
        """The OS user name, or ``""`` when the platform can't say."""
        try:
            return getpass.getuser()
        except Exception:  # getpass raises a bare OSError/KeyError per platform
            return ""

    @staticmethod
    def _builtin_spec(builtin_dir: Optional[Path]) -> Optional[Dict[str, str]]:
        """Describe *builtin_dir* host-independently for the ``.domain`` marker.

        A shipped ``presets/`` dir lives inside a package, whose install location
        differs per host Python (Maya's site-packages vs Blender's vs an editable
        checkout). Recording ``{"package", "path"}`` relative to the OUTERMOST
        package dir lets every host resolve its own copy (see
        :meth:`PresetStore.resolve_builtin_spec`) without importing the package.
        A dir outside any package falls back to ``{"dir": <absolute path>}``.
        """
        if builtin_dir is None:
            return None
        d = Path(builtin_dir).resolve()
        top = None
        for candidate in (d, *d.parents):
            if (candidate / "__init__.py").is_file():
                top = candidate
            elif top is not None:
                break
        if top is None:
            return {"dir": str(d)}
        return {"package": top.name, "path": d.relative_to(top).as_posix()}


class PresetStore(_PresetStoreInternal):
    """Named-preset collection with a read-only built-in tier and a writable
    user tier. Qt-free; deals in plain dicts (JSON by default, or any injected
    :class:`Codec` — e.g. YAML).

    Parameters:
        name: Collection name. With no explicit *user_dir*, the user tier is
            ``user_config_root()/<package>/<name>/``.
        package: Owning package, used only to build the default *user_dir*.
        builtin_dir: Directory of shipped, read-only presets (the repo
            ``presets/`` dir). ``None`` or a missing dir ⇒ the built-in tier is
            simply absent.
        user_dir: Explicit writable directory. When omitted it is derived from
            *package*/*name* under :func:`user_config_root`. (uitk passes its own
            already-migrated dir here so both layers agree.)
        codec: On-disk (de)serialiser (see :class:`Codec`). Defaults to
            :data:`JSON_CODEC`; pass a YAML codec to back ``*.yaml`` templates.
        mark: Write the ``.domain`` marker that lets ``PresetLibrary`` find
            this store (once per process, only when the folder exists).
            ``PresetLibrary`` passes ``False`` for its own stores: a scan never
            writes.
    """

    def __init__(
        self,
        name: str,
        package: str = "",
        *,
        builtin_dir: Optional[Union[str, os.PathLike]] = None,
        user_dir: Optional[Union[str, os.PathLike]] = None,
        codec: Codec = JSON_CODEC,
        mark: bool = True,
    ):
        self.name = name
        self.package = package
        self._builtin_dir = Path(builtin_dir) if builtin_dir else None
        self._user_dir = Path(user_dir) if user_dir else None
        self._codec = codec
        self._ext = codec.ext
        self._mark_enabled = mark

    @property
    def ext(self) -> str:
        """File extension this store reads/writes (from its :class:`Codec`)."""
        return self._codec.ext

    # ------------------------------------------------------------------ dirs
    @property
    def user_dir(self) -> Path:
        """Writable preset directory (created lazily on first :meth:`save`)."""
        if self._user_dir is None:
            root = UserConfig.user_config_root()
            self._user_dir = (
                root / self.package / self.name if self.package else root / self.name
            )
        return self._user_dir

    @property
    def builtin_dir(self) -> Optional[Path]:
        """Read-only shipped preset directory, or ``None`` when not configured."""
        return self._builtin_dir

    @property
    def key(self) -> Optional[str]:
        """This store's folder relative to the ecosystem presets root, or ``None``.

        E.g. ``"mayatk/rizom_bridge/unwrap_hard"``: the id ``PresetLibrary`` and
        exported bundles use for a store. ``None`` for a store whose user dir sits
        outside :meth:`UserConfig.user_config_root` (a test dir, a custom path) --
        such a store is simply not part of the managed collection.
        """
        try:
            rel = self.user_dir.resolve().relative_to(
                Path(UserConfig.user_config_root()).resolve()
            )
        except (ValueError, OSError):
            return None
        posix = rel.as_posix()
        return None if posix in ("", ".") else posix

    def _mark(self) -> None:
        """Write/refresh the ``.domain`` marker (see :data:`DOMAIN_MARKER`).

        Once per process per folder, only when the folder already exists (a read
        never creates one), and only rewritten when its content changed -- the
        built-in spec is host-independent, so every host that can import the
        owning package writes the same file. Best-effort: a failure only means
        the library cannot see this store until the next attempt.
        """
        if not self._mark_enabled:
            return
        user_dir = self.user_dir
        token = f"{user_dir}|{UserConfig.user_config_root()}"
        if token in _PresetStoreInternal._marked_dirs or not user_dir.is_dir():
            return
        key = self.key
        if key is None:
            _PresetStoreInternal._marked_dirs.add(token)
            return
        builtin = _PresetStoreInternal._builtin_spec(self._builtin_dir)
        if builtin and "package" in builtin:
            # Keep the portable form only if it finds THIS dir here: a package
            # dir named like another importable one (a repo's ``test`` package
            # vs the stdlib's) would otherwise resolve somewhere else entirely.
            resolved = PresetStore.resolve_builtin_spec(builtin)
            if (
                resolved is None
                or resolved.resolve() != Path(self._builtin_dir).resolve()
            ):
                builtin = {"dir": str(Path(self._builtin_dir).resolve())}
        content = json.dumps(
            {
                "key": key,
                "ext": self.ext,
                "builtin": builtin,
            },
            indent=4,
            sort_keys=True,
        )
        path = user_dir / DOMAIN_MARKER
        try:
            current = path.read_text(encoding="utf-8") if path.is_file() else None
            if current != content:
                _PresetStoreInternal._atomic_write_text(path, content)
            _PresetStoreInternal._marked_dirs.add(token)
        except OSError as e:
            logger.debug("PresetStore: could not write %s: %s", path, e)

    @staticmethod
    def resolve_builtin_spec(spec: Optional[Dict[str, str]]) -> Optional[Path]:
        """Resolve a ``.domain`` marker's built-in spec to a directory, or ``None``.

        ``{"package", "path"}`` resolves through ``importlib.util.find_spec``,
        which locates a top-level package WITHOUT importing it, so a host lists
        another tool's shipped presets without running that tool. ``None`` when
        the package is not installed in this Python.
        """
        if not spec:
            return None
        if spec.get("dir"):
            return Path(spec["dir"])
        package, rel = spec.get("package"), spec.get("path", "")
        if not package:
            return None
        import importlib.util

        try:
            found = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            return None
        locations = list(getattr(found, "submodule_search_locations", None) or [])
        if not locations:
            return None
        return Path(locations[0]).joinpath(*[p for p in rel.split("/") if p])

    # ------------------------------------------------------------------ info
    def info_path(self, name: str) -> Path:
        """Path of *name*'s management sidecar (user tier; see :data:`INFO_EXT`)."""
        return self.user_dir / _PresetStoreInternal._sidecar_name(
            PresetStore.sanitize_preset_name(name)
        )

    def info(self, name: str) -> Dict[str, Any]:
        """Management metadata of the user preset *name* (``{}`` when it has none).

        Keys (all optional): ``id`` (uuid, survives renames), ``label`` (the name
        as typed -- file names lose punctuation), ``collection``, ``read_only``,
        ``created``, ``author``, ``tags``, ``description`` (what the preset is
        for), ``hidden`` (left out of its tool's dropdown; see
        :meth:`set_hidden`), ``origin_hash`` (payload hash when installed from /
        published to a collection). Built-ins carry none; they are read-only by
        tier (see :meth:`description` and :meth:`is_hidden` for theirs).
        """
        return _PresetStoreInternal._read_json(self.info_path(name))

    def write_info(self, name: str, info: Dict[str, Any]) -> None:
        """Replace *name*'s whole sidecar with *info* (the user preset must exist)."""
        if not self.path(name, "user").is_file():
            raise KeyError(f"no user preset {name!r} in {self.name}")
        _PresetStoreInternal._atomic_write_text(
            self.info_path(name), json.dumps(info, indent=4, sort_keys=True)
        )
        self._mark()

    def set_info(self, name: str, **fields: Any) -> Dict[str, Any]:
        """Merge *fields* into *name*'s sidecar; a ``None`` value removes the key.

        Returns the merged metadata. Raises ``KeyError`` when *name* is not a
        user preset (built-ins are never stamped).
        """
        info = self.info(name)
        for key, value in fields.items():
            if value is None:
                info.pop(key, None)
            else:
                info[key] = value
        self.write_info(name, info)
        return info

    def ensure_info(self, name: str) -> Dict[str, Any]:
        """*name*'s metadata, stamped with an ``id`` (and label/created/author) if missing."""
        info = self.info(name)
        if info.get("id"):
            return info
        info = {**self._new_info(name), **{k: v for k, v in info.items() if v}}
        self.write_info(name, info)
        return info

    def is_read_only(self, name: str) -> bool:
        """True for a shipped built-in, or a user preset whose sidecar is locked."""
        tier = self.source(name)
        if tier == "builtin":
            return True
        return tier == "user" and bool(self.info(name).get("read_only"))

    def description(self, name: str) -> str:
        """What preset *name* is for, or ``""``.

        A user preset's is the ``description`` in its sidecar (written with
        :meth:`set_info`, so it follows a rename); a built-in's is read-only,
        from its shipped payload's ``_meta.description``.
        """
        tier = self.source(name)
        if tier == "user":
            return str(self.info(name).get("description") or "")
        if tier == "builtin":
            return _PresetStoreInternal._meta_description(
                self.path(name, "builtin"), self._codec.load
            )
        return ""

    # ----------------------------------------------------------------- hidden
    def is_hidden(self, name: str) -> bool:
        """True when *name* is left out of its tool's dropdown (:meth:`set_hidden`)."""
        tier = self.source(name)
        if tier == "user":
            return bool(self.info(name).get("hidden"))
        return (
            tier == "builtin"
            and PresetStore.sanitize_preset_name(name) in self._hidden_builtins()
        )

    def set_hidden(self, name: str, hidden: bool = True) -> bool:
        """Leave preset *name* out of its tool's dropdown, or list it again.

        Nothing is deleted: the preset stays on disk and loads as before, and a
        panel keeps showing the one it is on. A user preset carries the flag in
        its sidecar, so it follows a rename and goes with a delete; a built-in,
        which has no sidecar, is listed in the store's ``.hidden`` file
        (:data:`HIDDEN_BUILTINS`). The flag is the preset's, not the name's: a
        user copy saved over a hidden built-in shows until it is hidden itself.

        Parameters:
            name: The preset (any spelling of its name).
            hidden: ``False`` lists it again.

        Returns:
            ``True`` when the flag changed.

        Raises:
            KeyError: *name* is in neither tier.
        """
        tier = self.source(name)
        if tier is None:
            raise KeyError(f"preset {name!r} not found in {self.name}")
        hidden = bool(hidden)
        if self.is_hidden(name) == hidden:
            return False
        if tier == "user":
            self.ensure_info(name)
            self.set_info(name, hidden=True if hidden else None)
            return True
        names = self._hidden_builtins()
        stem = PresetStore.sanitize_preset_name(name)
        if hidden:
            names.add(stem)
        else:
            names.discard(stem)
        self._write_hidden_builtins(names)
        return True

    def _hidden_builtins(self) -> Set[str]:
        """The stems in the ``.hidden`` list: built-ins left out of the dropdown."""
        data = _PresetStoreInternal._read_json(self.user_dir / HIDDEN_BUILTINS)
        names = data.get("names")
        if not isinstance(names, list):
            return set()
        return {
            PresetStore.sanitize_preset_name(n)
            for n in names
            if isinstance(n, str) and n
        }

    def _write_hidden_builtins(self, names: Set[str]) -> None:
        """Replace the ``.hidden`` list atomically; an empty one removes the file."""
        path = self.user_dir / HIDDEN_BUILTINS
        if names:
            path.parent.mkdir(parents=True, exist_ok=True)
            _PresetStoreInternal._atomic_write_text(
                path, json.dumps({"names": sorted(names)}, indent=4)
            )
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self._mark()

    def unique_name(self, base: str) -> str:
        """*base*, or ``"<base> 2"``, ``"<base> 3"``... -- the first free in either tier."""
        candidate, n = str(base), 2
        while self.exists(candidate):
            candidate = f"{base} {n}"
            n += 1
        return candidate

    def _locked(self, name: str) -> bool:
        """True when the USER copy of *name* exists and is locked."""
        return self.path(name, "user").is_file() and bool(
            self.info(name).get("read_only")
        )

    @staticmethod
    def _new_info(name: str) -> Dict[str, Any]:
        """Fresh metadata for a preset first written under *name*."""
        return {
            "id": uuid.uuid4().hex,
            "label": str(name),
            "created": _PresetStoreInternal._now(),
            "author": _PresetStoreInternal._author(),
        }

    # ----------------------------------------------------------------- active
    @property
    def _active_path(self) -> Path:
        """Path to the ``.active`` sidecar (last-selected preset pointer)."""
        return self.user_dir / ACTIVE_SENTINEL

    @property
    def active(self) -> Optional[str]:
        """The last-selected preset's file stem, or ``None`` when unset/unreadable.

        Stored as ``{"name": <stem>}`` in the ``.active`` sidecar. The stem is
        the name :meth:`list` shows, a combo holds and the Preset Editor renames
        and deletes by -- a name with punctuation loses it on disk (``"a (b)"``
        is stored as ``"a _b_"``). A pointer an older release wrote holds the
        name as typed; it reads back as its stem, so it names the same preset.
        The name is returned even if it no longer resolves to a file, so
        callers can decide how to handle a stale pointer (a GUI combo simply
        falls back to no-selection); :meth:`delete` / :meth:`rename` keep it
        consistent when *they* are the cause of the change.
        """
        path = self._active_path
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return None
        name = data.get("name") if isinstance(data, dict) else None
        if not (isinstance(name, str) and name):
            return None
        return PresetStore.sanitize_preset_name(name)

    @active.setter
    def active(self, name: Optional[str]) -> None:
        """Set (or clear, with ``None``) the active-preset pointer.

        Any spelling of the preset's name may be given (the name as typed, or
        its stem); the file stem is what is stored (see :attr:`active`).
        """
        path = self._active_path
        if not name:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.debug("PresetStore: could not clear .active: %s", e)
            return
        stem = PresetStore.sanitize_preset_name(name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _PresetStoreInternal._atomic_write_text(path, json.dumps({"name": stem}))
        except OSError as e:
            logger.debug("PresetStore: could not write .active: %s", e)

    # ------------------------------------------------------------------ query
    def _names_in(self, directory: Optional[Path]) -> List[str]:
        if not directory or not Path(directory).is_dir():
            return []
        return [p.stem for p in Path(directory).glob("*" + self._ext) if p.is_file()]

    def list(self, tier: Optional[str] = None) -> List[str]:
        """Sorted preset names.

        *tier* ``None`` (default) returns the union of both tiers (a user name
        shadows the built-in of the same name — listed once); ``"user"`` /
        ``"builtin"`` restrict to one tier.
        """
        self._mark()
        if tier == "user":
            names = set(self._names_in(self.user_dir))
        elif tier == "builtin":
            names = set(self._names_in(self._builtin_dir))
        elif tier is None:
            names = set(self._names_in(self._builtin_dir)) | set(
                self._names_in(self.user_dir)
            )
        else:
            raise ValueError(f"tier must be None, 'user', or 'builtin'; got {tier!r}")
        return sorted(names)

    def source(self, name: str) -> Optional[str]:
        """Which tier *name* resolves from: ``"user"``, ``"builtin"``, or ``None``."""
        if self.path(name, "user").is_file():
            return "user"
        if self._builtin_dir and self.path(name, "builtin").is_file():
            return "builtin"
        return None

    def exists(self, name: str) -> bool:
        return self.source(name) is not None

    def path(self, name: str, tier: str = "user") -> Path:
        """Sanitized file path for *name* in *tier* (``"user"`` or ``"builtin"``)."""
        if tier == "user":
            base = self.user_dir
        elif tier == "builtin":
            if self._builtin_dir is None:
                raise ValueError("no built-in directory configured")
            base = self._builtin_dir
        else:
            raise ValueError(f"tier must be 'user' or 'builtin'; got {tier!r}")
        return base / f"{PresetStore.sanitize_preset_name(name)}{self._ext}"

    # ------------------------------------------------------------------ io
    def load(self, name: str) -> dict:
        """Return the preset dict for *name* (user tier shadows built-in).

        Raises ``KeyError`` when *name* exists in neither tier, propagates the
        codec's own parse error for a malformed file (e.g. ``json``'s
        ``ValueError`` / a YAML codec's ``YAMLError``), and raises ``ValueError``
        when the parsed top-level value is not a mapping.
        """
        tier = self.source(name)
        if tier is None:
            raise KeyError(
                f"preset {name!r} not found in {self.name} "
                f"(available: {self.list() or '(none)'})"
            )
        path = self.path(name, tier)
        data = self._codec.load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"preset {name!r} is not a mapping: {path}")
        return data

    def save(self, name: str, data: dict, *, force: bool = False) -> Path:
        """Write *data* as a user preset *name* (built-ins stay read-only).

        Creates the user dir on demand, and the preset's sidecar (id, label,
        created, author) on its first save; a re-save keeps the existing sidecar,
        so the preset keeps its identity. A preset that did not exist gets a
        FRESH sidecar even if a stray one sits under its name (an older install
        deletes a payload without knowing its sidecar) -- a new preset must not
        inherit a dead one's id, lock or collection. Returns the path written.

        Raises:
            PresetReadOnlyError: the user preset *name* is locked. *force* writes
                anyway (a collection update replacing its own locked member).
        """
        path = self.path(name, "user")
        if not force and self._locked(name):
            raise PresetReadOnlyError(
                f"preset {name!r} in {self.name} is read-only; save it under "
                "another name, or unlock it in the Preset Editor"
            )
        existed = path.is_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        _PresetStoreInternal._atomic_write_text(path, self._codec.dump(data))
        logger.debug("PresetStore: saved %r -> %s", name, path)
        try:
            if existed:
                self.ensure_info(name)
            else:
                self.write_info(name, self._new_info(name))
        except OSError as e:  # the payload is written; metadata is best-effort
            logger.debug("PresetStore: could not stamp %r: %s", name, e)
        return path

    def delete(self, name: str, *, force: bool = False) -> bool:
        """Delete the *user* preset *name*. Returns ``True`` if a file was removed.

        Built-ins are never deleted; deleting a name that exists only as a
        built-in is a no-op returning ``False``, and so is deleting a locked
        user preset unless *force*. The preset's sidecar goes with it.
        """
        path = self.path(name, "user")
        if path.is_file():
            if not force and self._locked(name):
                return False
            path.unlink()
            try:
                self.info_path(name).unlink()
            except FileNotFoundError:
                pass
            logger.debug("PresetStore: deleted %r -> %s", name, path)
            # Drop a now-dangling active pointer (a user shadow that fell back
            # to a built-in of the same name is still valid, so guard on exists).
            # The pointer holds the stem; *name* may be spelled as typed.
            stem = PresetStore.sanitize_preset_name(name)
            if self.active == stem and not self.exists(name):
                self.active = None
            return True
        return False

    def rename(self, old: str, new: str, *, force: bool = False) -> bool:
        """Rename a *user* preset. Returns ``True`` on success.

        Fails (``False``) when *old* isn't a user preset, is locked (unless
        *force*), or *new* already exists in either tier (no silent shadowing of
        a built-in). The sidecar follows the payload and its ``label`` becomes
        *new*; the ``id`` is kept, which is what lets a collection update find a
        renamed member.
        """
        src = self.path(old, "user")
        if not src.is_file() or self.exists(new):
            return False
        if not force and self._locked(old):
            return False
        info = self.info(old)
        src.rename(self.path(new, "user"))
        old_info = self.info_path(old)
        try:
            self.write_info(new, {**(info or self._new_info(new)), "label": str(new)})
            if old_info.is_file():
                old_info.unlink()
        except OSError as e:  # payload moved; a stale sidecar is harmless
            logger.debug("PresetStore: could not move %r's sidecar: %s", old, e)
        # The pointer holds the stem; *old* may be spelled as typed.
        if self.active == PresetStore.sanitize_preset_name(old):
            self.active = new
        return True

    @staticmethod
    def sanitize_preset_name(name: str) -> str:
        """Filesystem-safe filename stem for a preset *name*.

        Keeps alphanumerics, ``-``, ``_`` and spaces; every other character becomes
        ``_``. Shared by both tiers (and by uitk's ``PresetManager``) so a name maps
        to the same file everywhere.
        """
        return "".join(
            c if c.isalnum() or c in ("-", "_", " ") else "_" for c in str(name)
        )
