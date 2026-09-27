# !/usr/bin/python
# coding=utf-8
"""Qt-free, zero-dependency user-config resolution for the ecosystem.

Keeps **personal / site-specific** values out of package source. A package ships
a generic *default* (and optionally an ``*.example.json`` template); the user's
real values live in a JSON file under :func:`user_config_root` (or an
env-pointed path), and only the keys they override need be present —
:meth:`UserConfig.resolve` deep-merges the user doc over the default.

Why here (and not in uitk's ``SettingsManager`` / ``PresetManager``): those are
the ecosystem's GUI settings/template stores and import ``qtpy``, but this module
must be usable from **headless, Qt-free** contexts -- notably the photogrammetry
engines running inside Metashape's bundled Python 3.9 (where Qt is absent and
engine code must not import it). So the consolidated location
(``<per-user-config>/uitk/<package>/``) is resolved HERE, with plain
``os``/``pathlib``, and uitk's ``PresetManager.get_presets_root`` delegates to it:
one owner, so one :data:`CONFIG_ROOT_ENV_VAR` override moves every store.

JSON (not TOML) is deliberate: ``tomllib`` is 3.11+ and won't import under
Metashape's 3.9; ``json`` is stdlib everywhere.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Union

logger = logging.getLogger(__name__)

# Env var that redirects the ecosystem user-config root wholesale. uitk's
# ``PresetManager.get_presets_root()`` delegates to :meth:`UserConfig.user_config_root`,
# so one override moves every store.
CONFIG_ROOT_ENV_VAR = "UITK_PRESETS_ROOT"
_ECOSYSTEM_WRAPPER = "uitk"


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

    #: Env var that redirects the ecosystem user-config root wholesale -- the
    #: public spelling of the module constant of the same name, so another
    #: package reaches it through the root (``ptk.UserConfig.CONFIG_ROOT_ENV_VAR``)
    #: instead of this module's path.
    CONFIG_ROOT_ENV_VAR: str = CONFIG_ROOT_ENV_VAR

    @staticmethod
    def path_for(name: str, package: str) -> Path:
        """Default on-disk location: ``<user_config_root>/<package>/<name>.json``."""
        return UserConfig.user_config_root() / package / f"{name}.json"

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

    @staticmethod
    def save_file(path: Union[str, os.PathLike], data: Mapping[str, Any]) -> None:
        """Write *data* to *path* as a JSON object, atomically.

        The counterpart :meth:`load_file` never had. Without it every caller
        that needed to persist a config hand-rolled a writer, and none of them
        were atomic -- so an interrupted save left a half-written file that
        ``load_file`` then silently discarded as unreadable, taking the user's
        settings with it.

        Refuses a non-mapping rather than writing one: ``load_file`` returns
        ``{}`` for anything that is not a JSON object, so a list written here
        would round-trip to nothing and look like data loss at read time.

        Parameters:
            path: Destination; missing parent directories are created.
            data: The config object to store.

        Raises:
            TypeError: *data* is not a mapping, or is not JSON-serialisable.
        """
        from pythontk.file_utils._file_utils import FileUtils

        if not isinstance(data, Mapping):
            raise TypeError(
                f"UserConfig.save_file expects a JSON object (mapping), got "
                f"{type(data).__name__}; load_file would read it back as {{}}."
            )
        FileUtils.write_json(path, dict(data), indent=4)

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
            src = cls.expand(os.environ[env])
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

    #: A variable reference in any spelling a profile may carry -- ``${VAR}``,
    #: ``$VAR`` or Windows' ``%VAR%`` -- or the ``%%`` escape for a literal
    #: ``%``. ``os.path.expandvars`` reads ``%VAR%`` only on Windows, so a
    #: profile written there kept it literal on Linux. A braced or ``%`` name
    #: runs to its delimiter, so ``%ProgramFiles(x86)%`` reads; a ``%`` name
    #: never spans whitespace or a path separator, so a stray ``%`` cannot
    #: swallow a path segment. A quote is text, as a path spells it
    #: (``Bob's Files``), not Windows' ``expandvars`` no-expansion span.
    _VAR_REF = re.compile(r"(%%)|\$\{([^}]+)\}|\$(\w+)|%([^%\s/\\]+)%")

    @staticmethod
    def expand(value: Any) -> Any:
        """Expand ``~`` and ``${VAR}`` / ``$VAR`` / ``%VAR%`` in string values.

        Every spelling on every OS, so a profile written on one machine reads
        the same on another; an unset variable is left as written, and ``%%``
        is a literal ``%``. ``TEMP`` and ``TMP`` fall back to the system temp
        dir where the OS sets neither (Linux), so a profile can reference
        ``${TEMP}`` portably.

        Recurses into dicts/lists/tuples; non-strings pass through. Apply to
        path-valued config entries. (Intra-document ``{token}`` interpolation
        is left to the schema-aware consumer, which knows which keys are the
        bases.)
        """
        if isinstance(value, str):

            def lookup(match):
                if match.group(1):  # the %% escape
                    return "%"
                name = next(group for group in match.groups() if group)
                found = os.environ.get(name)
                if found is None and name in ("TEMP", "TMP"):
                    found = tempfile.gettempdir()
                return match.group(0) if found is None else found

            return os.path.expanduser(UserConfig._VAR_REF.sub(lookup, value))
        if isinstance(value, Mapping):
            return {k: UserConfig.expand(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(UserConfig.expand(v) for v in value)
        return value

    #: The XDG base-directory defaults, under the home folder.
    _XDG_DEFAULTS = {"config": (".config",), "data": (".local", "share")}

    @staticmethod
    def xdg_home(kind: str = "config") -> str:
        """An XDG base directory: ``$XDG_CONFIG_HOME`` / ``$XDG_DATA_HOME`` when it
        holds an absolute path, else the spec's default (``~/.config`` /
        ``~/.local/share``).

        The spec calls a relative value invalid, and Qt ignores one; used as
        given, the directory moved with the working directory. It is the
        freedesktop rule -- whether it applies on this OS is the caller's call.

        Parameters:
            kind (str): ``"config"`` or ``"data"``.

        Returns:
            str: An absolute directory path (not created).

        Raises:
            ValueError: *kind* is neither.
        """
        try:
            default = UserConfig._XDG_DEFAULTS[kind]
        except KeyError:
            raise ValueError(f"kind is 'config' or 'data', not {kind!r}") from None
        base = os.environ.get(f"XDG_{kind.upper()}_HOME", "")
        if os.path.isabs(base):
            return base
        return os.path.join(os.path.expanduser("~"), *default)

    @staticmethod
    def user_config_root() -> Path:
        """The ecosystem per-user config directory, resolved **without Qt**.

        Honors ``$UITK_PRESETS_ROOT`` (used as given; ``~`` and ``%VAR%`` expanded).
        Otherwise the host-independent per-user config dir plus a ``uitk`` wrapper
        folder. The one owner of the root: uitk's ``PresetManager.get_presets_root()``
        delegates here.

        * Windows: ``%LOCALAPPDATA%/uitk``
        * macOS:   ``~/Library/Preferences/uitk``
        * Linux:   ``$XDG_CONFIG_HOME/uitk`` (else ``~/.config/uitk``)

        Always absolute: a relative ``$XDG_CONFIG_HOME`` is ignored, as the XDG
        spec requires (and Qt does) -- used as given, the root moved with the
        working directory.
        """
        override = os.environ.get(CONFIG_ROOT_ENV_VAR)
        if override:
            p = Path(UserConfig.expand(override))
            return p if p.is_absolute() else p.absolute()

        system = platform.system().lower()
        if system == "windows":
            base = os.environ.get("LOCALAPPDATA") or os.path.join(
                os.path.expanduser("~"), "AppData", "Local"
            )
        elif system == "darwin":
            base = os.path.join(os.path.expanduser("~"), "Library", "Preferences")
        else:
            base = UserConfig.xdg_home("config")
        root = Path(base) / _ECOSYSTEM_WRAPPER
        return root if root.is_absolute() else root.absolute()
