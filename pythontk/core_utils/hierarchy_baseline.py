# !/usr/bin/python
# coding=utf-8
"""Moved to :mod:`pythontk.core_utils.engines.scene_export.hierarchy_baseline` (2026-09-26).

A one-release stub for code importing the old path; import the names from the
pythontk root instead (``from pythontk import ...``).
"""

from pythontk.core_utils.deprecation import Deprecation

Deprecation.attributes(
    globals(),
    {
        "HierarchyBaseline": "pythontk.core_utils.engines.scene_export.hierarchy_baseline.HierarchyBaseline",
        "HierarchyBaselineStore": "pythontk.core_utils.engines.scene_export.hierarchy_baseline.HierarchyBaselineStore",
    },
    remove_in="0.13.0",
    since="2026-09-26",
)
