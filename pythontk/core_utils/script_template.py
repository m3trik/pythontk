# !/usr/bin/python
# coding=utf-8
"""Moved to :mod:`pythontk.core_utils.handoff.script_template` (2026-09-26).

A one-release stub for code importing the old path; import the names from the
pythontk root instead (``from pythontk import ...``).
"""
from pythontk.core_utils.deprecation import Deprecation

Deprecation.attributes(
    globals(),
    {
        "SEND_TO": "pythontk.core_utils.handoff.script_template.SEND_TO",
        "SAVE_AS": "pythontk.core_utils.handoff.script_template.SAVE_AS",
        "ROUND_TRIP": "pythontk.core_utils.handoff.script_template.ROUND_TRIP",
        "ScriptTemplate": "pythontk.core_utils.handoff.script_template.ScriptTemplate",
    },
    remove_in="0.13.0",
    since="2026-09-26",
)
