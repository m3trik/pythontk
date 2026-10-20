# !/usr/bin/python
# coding=utf-8
"""A look-match session file, declared once: the parts a session names, and where.

Read by :meth:`LookMatch.open`, :class:`LookLoop`, ``LookCapture.open_session``
and the Look Match panel, through :meth:`LookSession.read` and
:meth:`LookSession.resolve`. A path is relative to the session file, so a
session travels with its project; one spelled ``cache:<path>`` lives in the
session's own folder in the per-user cache instead (:meth:`LookSession.cache_dir`):
derived data -- captured layers, bakes, rounds -- that is rebuilt from its
sources and kept off a synced project drive::

    {"version": 1, "id": "soldering_moody",
     "layers": "cache:layers", "target": "target.json", "mix": "mixes/h4.json",
     "template": "cache:template/ROOM.glb", "size": [820, 480],
     "transform": {"operator": "aces_filmic", "exposure": 1.0},
     "capture": {"camera": {...}, "size": [1638, 960], "black": ["PROPS"]},
     "loop": {"queue": "cache:bake_queue", "service": "http://127.0.0.1:8770/",
              "worker": {"module": "mayatk.light_utils.light_layer_baker",
                         "scene": "../scenes/ROOM.ma"}}}
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from pythontk.core_utils.schema_spec import SchemaSpec
from pythontk.core_utils.user_config import UserConfig

_field = SchemaSpec.spec_field


@dataclass
class LookCaptureSettings(SchemaSpec):
    """How the capture service shoots the session's template."""

    TYPED = True

    camera: Optional[Dict[str, Any]] = _field(
        help="{pos, target, up, vfov} in model space; none keeps the page's own framing."
    )
    size: Optional[List[int]] = _field(help="The viewport's [width, height].")
    black: Optional[List[str]] = _field(
        help="Material or object name patterns whose reflections a layer capture leaves out ('!pattern' keeps a match)."
    )
    emissive: Optional[Dict[str, Any]] = _field(
        help="{pattern, intensity}: an emissive level an exact frame previews."
    )


@dataclass
class LookWorker(SchemaSpec):
    """The host process that bakes the session's layers: a module run as
    ``python -m <module> launch --queue <queue> --scene <scene>``."""

    TYPED = True

    module: str = _field(
        help="The host's layer-baker module (mayatk's: mayatk.light_utils.light_layer_baker).",
        required=True,
    )
    scene: str = _field(help="The scene it bakes (a session path).", required=True)


@dataclass
class LookLoopSettings(SchemaSpec):
    """Where a round's bakes and captures go (:class:`LookLoop`)."""

    TYPED = True

    queue: Optional[str] = _field(help="The bake worker's queue folder (a session path).")
    service: Optional[str] = _field(help="The capture service's URL.")
    worker: Optional[LookWorker] = _field(help="What serves the queue (LookWorker).")


def _check_id(value: Any) -> List[str]:
    """An id names a cache folder: one plain path component, nothing to climb out by
    (``null`` is no id)."""
    if value is None or (
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
    ):
        return []
    return [f"{value!r} is not a plain folder name (letters, digits, _ . -)"]


@dataclass
class LookSession(SchemaSpec):
    """A look-match session file (see the module docstring)."""

    TYPED = True
    VERSION = 1
    #: A path starting with this lives in the session's cache folder (:meth:`cache_dir`).
    CACHE = "cache:"

    version: int = _field(help="The file's version (LookSession.VERSION).", required=True)
    id: Optional[str] = _field(
        help="Its folder in the per-user cache, where its 'cache:' paths live.",
        validate=_check_id,
    )
    layers: str = _field(help="The layer store's folder.", required=True)
    target: str = _field(help="The reference target's JSON file.", required=True)
    mix: Optional[str] = _field(help="The mix the session opens with.")
    template: Optional[str] = _field(help="The lightmapped GLB the capture service relights.")
    size: Optional[List[int]] = _field(help="The [width, height] it works at.")
    transform: Optional[Dict[str, Any]] = _field(
        help="The viewer's display transform: {operator, exposure}."
    )
    capture: Optional[LookCaptureSettings] = _field(
        help="How the capture service shoots it (LookCaptureSettings)."
    )
    loop: Optional[LookLoopSettings] = _field(
        help="Where a round's bakes and captures go (LookLoopSettings)."
    )

    @classmethod
    def read(cls, path: str) -> Dict[str, Any]:
        """The session file at *path*.

        Raises:
            ValueError: it is newer than this version reads, or does not fit.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a look-match session: expected a JSON object.")
        version = data.get("version", 1)
        if version > cls.VERSION:
            raise ValueError(
                f"Session version {version} is newer than this pythontk reads ({cls.VERSION})."
            )
        checked = cls.validate(dict(data, version=version))
        if checked.errors:
            raise ValueError(
                f"{path} is not a look-match session: " + "; ".join(checked.errors)
            )
        return data

    @classmethod
    def cache_dir(cls, data: Mapping[str, Any]) -> str:
        """The session's folder in the per-user cache (``look_match/<id>``).

        Raises:
            ValueError: the session has no ``id``.
        """
        if not data.get("id"):
            raise ValueError(
                "This session names a 'cache:' path but no id: give it one."
            )
        return str(UserConfig.cache_dir("look_match", data["id"]))

    @classmethod
    def resolve(
        cls, path: str, data: Mapping[str, Any], value: Optional[str]
    ) -> Optional[str]:
        """*value*, a path the session file at *path* (read as *data*) names, as an
        absolute path: under its cache folder when spelled ``cache:``, else
        beside the file. ``None`` stays ``None``."""
        if value is None:
            return None
        if value.startswith(cls.CACHE):
            base = cls.cache_dir(data)
            value = value[len(cls.CACHE) :]
        else:
            base = os.path.dirname(os.path.abspath(path))
        return os.path.normpath(os.path.join(base, value))
