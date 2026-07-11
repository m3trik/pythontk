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
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from pythontk.core_utils.user_config import user_config_root

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


def sanitize_preset_name(name: str) -> str:
    """Filesystem-safe filename stem for a preset *name*.

    Keeps alphanumerics, ``-``, ``_`` and spaces; every other character becomes
    ``_``. Shared by both tiers (and by uitk's ``PresetManager``) so a name maps
    to the same file everywhere.
    """
    return "".join(c if c.isalnum() or c in ("-", "_", " ") else "_" for c in str(name))


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


class PresetStore:
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
    """

    def __init__(
        self,
        name: str,
        package: str = "",
        *,
        builtin_dir: Optional[Union[str, os.PathLike]] = None,
        user_dir: Optional[Union[str, os.PathLike]] = None,
        codec: Codec = JSON_CODEC,
    ):
        self.name = name
        self.package = package
        self._builtin_dir = Path(builtin_dir) if builtin_dir else None
        self._user_dir = Path(user_dir) if user_dir else None
        self._codec = codec
        self._ext = codec.ext

    @property
    def ext(self) -> str:
        """File extension this store reads/writes (from its :class:`Codec`)."""
        return self._codec.ext

    # ------------------------------------------------------------------ dirs
    @property
    def user_dir(self) -> Path:
        """Writable preset directory (created lazily on first :meth:`save`)."""
        if self._user_dir is None:
            root = user_config_root()
            self._user_dir = root / self.package / self.name if self.package else root / self.name
        return self._user_dir

    @property
    def builtin_dir(self) -> Optional[Path]:
        """Read-only shipped preset directory, or ``None`` when not configured."""
        return self._builtin_dir

    # ----------------------------------------------------------------- active
    @property
    def _active_path(self) -> Path:
        """Path to the ``.active`` sidecar (last-selected preset pointer)."""
        return self.user_dir / ACTIVE_SENTINEL

    @property
    def active(self) -> Optional[str]:
        """The last-selected preset name, or ``None`` when unset/unreadable.

        Stored as ``{"name": <preset>}`` in the ``.active`` sidecar. The raw
        name is returned even if it no longer resolves to a file, so callers
        can decide how to handle a stale pointer (a GUI combo simply falls back
        to no-selection); :meth:`delete` / :meth:`rename` keep it consistent
        when *they* are the cause of the change.
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
        return name if isinstance(name, str) and name else None

    @active.setter
    def active(self, name: Optional[str]) -> None:
        """Set (or clear, with ``None``) the active-preset pointer."""
        path = self._active_path
        if not name:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.debug("PresetStore: could not clear .active: %s", e)
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(path, json.dumps({"name": name}))
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
        if tier == "user":
            names = set(self._names_in(self.user_dir))
        elif tier == "builtin":
            names = set(self._names_in(self._builtin_dir))
        elif tier is None:
            names = set(self._names_in(self._builtin_dir)) | set(self._names_in(self.user_dir))
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
        return base / f"{sanitize_preset_name(name)}{self._ext}"

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

    def save(self, name: str, data: dict) -> Path:
        """Write *data* as a user preset *name* (built-ins stay read-only).

        Creates the user dir on demand. Returns the path written.
        """
        path = self.path(name, "user")
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, self._codec.dump(data))
        logger.debug("PresetStore: saved %r -> %s", name, path)
        return path

    def delete(self, name: str) -> bool:
        """Delete the *user* preset *name*. Returns ``True`` if a file was removed.

        Built-ins are never deleted; deleting a name that exists only as a
        built-in is a no-op returning ``False``.
        """
        path = self.path(name, "user")
        if path.is_file():
            path.unlink()
            logger.debug("PresetStore: deleted %r -> %s", name, path)
            # Drop a now-dangling active pointer (a user shadow that fell back
            # to a built-in of the same name is still valid, so guard on exists).
            if self.active == name and not self.exists(name):
                self.active = None
            return True
        return False

    def rename(self, old: str, new: str) -> bool:
        """Rename a *user* preset. Returns ``True`` on success.

        Fails (``False``) when *old* isn't a user preset or *new* already exists
        in either tier (no silent shadowing of a built-in).
        """
        src = self.path(old, "user")
        if not src.is_file() or self.exists(new):
            return False
        src.rename(self.path(new, "user"))
        if self.active == old:
            self.active = new
        return True
