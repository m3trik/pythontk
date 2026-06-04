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
from pathlib import Path
from typing import List, Optional, Union

from pythontk.core_utils.user_config import user_config_root

logger = logging.getLogger(__name__)

EXT = ".json"


def sanitize_preset_name(name: str) -> str:
    """Filesystem-safe filename stem for a preset *name*.

    Keeps alphanumerics, ``-``, ``_`` and spaces; every other character becomes
    ``_``. Shared by both tiers (and by uitk's ``PresetManager``) so a name maps
    to the same file everywhere.
    """
    return "".join(c if c.isalnum() or c in ("-", "_", " ") else "_" for c in str(name))


class PresetStore:
    """Named-preset collection with a read-only built-in tier and a writable
    user tier. Qt-free; deals in plain JSON dicts.

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
    """

    EXT = EXT

    def __init__(
        self,
        name: str,
        package: str = "",
        *,
        builtin_dir: Optional[Union[str, os.PathLike]] = None,
        user_dir: Optional[Union[str, os.PathLike]] = None,
    ):
        self.name = name
        self.package = package
        self._builtin_dir = Path(builtin_dir) if builtin_dir else None
        self._user_dir = Path(user_dir) if user_dir else None

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

    # ------------------------------------------------------------------ query
    @staticmethod
    def _names_in(directory: Optional[Path]) -> List[str]:
        if not directory or not Path(directory).is_dir():
            return []
        return [p.stem for p in Path(directory).glob("*" + EXT) if p.is_file()]

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
        return base / f"{sanitize_preset_name(name)}{EXT}"

    # ------------------------------------------------------------------ io
    def load(self, name: str) -> dict:
        """Return the preset dict for *name* (user tier shadows built-in).

        Raises ``KeyError`` when *name* exists in neither tier and ``ValueError``
        when the file is present but not a JSON object.
        """
        tier = self.source(name)
        if tier is None:
            raise KeyError(
                f"preset {name!r} not found in {self.name} "
                f"(available: {self.list() or '(none)'})"
            )
        path = self.path(name, tier)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"preset {name!r} is not a JSON object: {path}")
        return data

    def save(self, name: str, data: dict) -> Path:
        """Write *data* as a user preset *name* (built-ins stay read-only).

        Creates the user dir on demand. Returns the path written.
        """
        path = self.path(name, "user")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)
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
        return True
