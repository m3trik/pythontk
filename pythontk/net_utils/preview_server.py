# !/usr/bin/python
# coding=utf-8
"""Deprecated import path: the live preview moved to :mod:`pythontk.net_utils.preview`.

Kept for one release so ``from pythontk.net_utils.preview_server import X``
keeps resolving, with a warning naming the new home. The public names have
always been on the root (``ptk.PreviewServer`` / ``ptk.PreviewDeliverer`` /
``ptk.PreviewBridge`` / ``ptk.PreviewPassContext``) and are unaffected.
"""

import importlib
import warnings

_HOMES = {
    "PreviewServer": "server",
    "VIEWER_CLOSED_PATH": "server",
    "SETTINGS_PATH": "server",
    "PreviewDeliverer": "deliverer",
    "PreviewPassContext": "deliverer",
    "PreviewBridge": "bridge",
}


def __getattr__(name):
    home = _HOMES.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"pythontk.net_utils.preview_server.{name} moved to "
        f"pythontk.net_utils.preview.{home}.{name}; this alias goes next release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(f"pythontk.net_utils.preview.{home}"), name)
