# !/usr/bin/python
# coding=utf-8
"""Deprecated import path: the live preview moved to :mod:`pythontk.net_utils.preview`.

Kept for one release so ``from pythontk.net_utils.preview_server import X``
keeps resolving, with a warning naming the new home. The public names have
always been on the root (``ptk.PreviewServer`` / ``ptk.PreviewDeliverer`` /
``ptk.PreviewBridge``) and are unaffected.
"""

from pythontk.core_utils.deprecation import Deprecation

#: The hand-rolled ``__getattr__`` this replaces got the warning right and the
#: deadline wrong -- "goes next release" named no release, so nothing could
#: check it. ``Deprecation.attributes`` also chains onto any ``__getattr__``
#: already on the module, which matters wherever a lazy loader owns one.
Deprecation.attributes(
    globals(),
    {
        "PreviewServer": "pythontk.net_utils.preview.server.PreviewServer",
        "VIEWER_CLOSED_PATH": "pythontk.net_utils.preview.server.VIEWER_CLOSED_PATH",
        "SETTINGS_PATH": "pythontk.net_utils.preview.server.SETTINGS_PATH",
        "PreviewDeliverer": "pythontk.net_utils.preview.deliverer.PreviewDeliverer",
        "PreviewBridge": "pythontk.net_utils.preview.bridge.PreviewBridge",
    },
    remove_in="0.11.0",
)
