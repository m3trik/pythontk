# !/usr/bin/python
# coding=utf-8
"""Moved: this module is ``mesh_convert/glb/fades.py`` since 2026-09-26.

A one-release stub for the names other packages imported from this path. The
channel table is ``GlbFades.CHANNELS`` on the root name: ``from pythontk import
GlbFades``.
"""

from pythontk.core_utils.deprecation import Deprecation

Deprecation.attributes(
    globals(),
    {
        "CHANNELS": "pythontk.GlbFades.CHANNELS",
        "GlbFades": "pythontk.GlbFades",
        "PointerChannel": "pythontk.file_utils.mesh_convert.glb.fades.PointerChannel",
    },
    remove_in="0.13.0",
    since="2026-09-26",
)
