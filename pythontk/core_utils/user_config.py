# !/usr/bin/python
# coding=utf-8
"""Qt-free, zero-dependency user-config resolution for the ecosystem.

Keeps **personal / site-specific** values out of package source. A package ships
a generic *default* (and optionally an ``*.example.json`` template); the user's
real values live in a JSON file under :func:`user_config_root` (or an
env-pointed path), and only the keys they override need be present —
:meth:`UserConfig.resolve` deep-merges the user doc over the default.

Why here (and not via uitk's ``SettingsManager`` / ``PresetManager``): those are
the ecosystem's GUI settings/template stores, but both import ``qtpy`` and resolve
their directory through ``QtCore.QStandardPaths``. This module must be usable from
**headless, Qt-free** contexts — notably the photogrammetry engines running inside
Metashape's bundled Python 3.9 (where Qt is absent and engine code must not import
it). So it reproduces the *same* consolidated location uitk uses
(``<per-user-config>/uitk/<package>/``) with plain ``os``/``pathlib`` instead of
Qt, and honors the same :data:`CONFIG_ROOT_ENV_VAR` override — so a power user who
redirects uitk's store redirects this too, and the Qt and Qt-free paths stay in
lockstep without this module depending on uitk.

JSON (not TOML) is deliberate: ``tomllib`` is 3.11+ and won't import under
Metashape's 3.9; ``json`` is stdlib everywhere.
"""
from __future__ import annotations

import json
import logging
import os
import platform
from pathlib import Path
from typing import Any, Mapping, Optional, Union

logger = logging.getLogger(__name__)

# Env var that redirects the ecosystem user-config root wholesale. Shared *by
# name* (a documented string convention, not an import) with uitk's
# ``preset_manager.get_presets_root()`` so one override moves both stores.
CONFIG_ROOT_ENV_VAR = "UITK_PRESETS_ROOT"
_ECOSYSTEM_WRAPPER = "uitk"


def user_config_root() -> Path:
    """The ecosystem per-user config directory, resolved **without Qt**.

    Honors ``$UITK_PRESETS_ROOT`` (used as given; ``~`` and ``%VAR%`` expanded).
    Otherwise the host-independent per-user config dir plus a ``uitk`` wrapper
    folder — matching uitk ``preset_manager.get_presets_root()`` so this Qt-free
    path and uitk's ``QStandardPaths`` path resolve to the same location:

    * Windows: ``%LOCALAPPDATA%/uitk``
    * macOS:   ``~/Library/Preferences/uitk``
    * Linux:   ``$XDG_CONFIG_HOME/uitk`` (else ``~/.config/uitk``)
    """
    override = os.environ.get(CONFIG_ROOT_ENV_VAR)
    if override:
        p = Path(os.path.expandvars(override)).expanduser()
        return p if p.is_absolute() else p.absolute()

    system = platform.system().lower()
    if system == "windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local"
        )
    elif system == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Preferences")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
            os.path.expanduser("~"), ".config"
        )
    return Path(base) / _ECOSYSTEM_WRAPPER


class UserConfig:
    """Resolve a JSON user-config doc with discovery + deep-merge over a default.

    Typical use — a package exposes a thin accessor::

        DEFAULT = {"root": "${TEMP}/myapp", "tuning": {"quality": 1}}

        def get_config(path=None):
            return UserConfig.resolve(
                "myapp", package="mypkg", env="MYAPP_CONFIG",
                default=DEFAULT, path=path,
            )

    The default ships in source (generic, non-personal); the user drops a
    partial ``<user_config_root>/mypkg/myapp.json`` overriding only what differs.
    """

    @staticmethod
    def path_for(name: str, package: str) -> Path:
        """Default on-disk location: ``<user_config_root>/<package>/<name>.json``."""
        return user_config_root() / package / f"{name}.json"

    @staticmethod
    def load_file(path: Union[str, os.PathLike]) -> dict:
        """Load a JSON object from *path*.

        Returns ``{}`` when the file is missing, unreadable, invalid JSON, or not
        a JSON object — resolution stays robust (falls back to the default)
        rather than raising at import/startup.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            logger.warning(f"UserConfig: could not read {path}: {e}")
            return {}
        if not isinstance(data, dict):
            logger.warning(f"UserConfig: {path} is not a JSON object; ignoring.")
            return {}
        return data

    @classmethod
    def resolve(
        cls,
        name: str,
        *,
        package: str,
        env: Optional[str] = None,
        default: Optional[Mapping[str, Any]] = None,
        path: Optional[Union[str, os.PathLike]] = None,
    ) -> dict:
        """Resolve config *name* for *package*, deep-merged over *default*.

        Source-file discovery (first match wins):

        1. explicit *path*
        2. ``$env`` (when that env var is set; ``~`` / ``%VAR%`` expanded)
        3. ``<user_config_root>/<package>/<name>.json`` (when it exists)
        4. none — *default* is returned as-is

        :returns: ``deep_merge(default, user)`` — the user doc need only carry
                  the keys it overrides.
        """
        result = cls.deep_merge({}, default or {})

        src: Optional[Union[str, os.PathLike]] = None
        if path:
            src = path
        elif env and os.environ.get(env):
            src = os.path.expanduser(os.path.expandvars(os.environ[env]))
        else:
            cand = cls.path_for(name, package)
            if cand.is_file():
                src = cand

        if src is not None:
            result = cls.deep_merge(result, cls.load_file(src))
        return result

    @staticmethod
    def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
        """Recursively merge *override* into a copy of *base* (override wins).

        Nested dicts merge key-by-key; scalars and lists replace wholesale
        (a list in *override* is taken as the intended value, not appended).
        """
        out = dict(base)
        for k, v in (override or {}).items():
            if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
                out[k] = UserConfig.deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    @staticmethod
    def expand(value: Any) -> Any:
        """Expand ``~`` and ``${ENV}`` / ``%VAR%`` in string values.

        Recurses into dicts/lists/tuples; non-strings pass through. Apply to
        path-valued config entries so a profile can reference ``${TEMP}`` or
        ``~`` portably. (Intra-document ``{token}`` interpolation is left to the
        schema-aware consumer, which knows which keys are the bases.)
        """
        if isinstance(value, str):
            return os.path.expanduser(os.path.expandvars(value))
        if isinstance(value, Mapping):
            return {k: UserConfig.expand(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(UserConfig.expand(v) for v in value)
        return value
