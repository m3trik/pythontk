# !/usr/bin/python
# coding=utf-8
"""Moved to :mod:`pythontk.core_utils.handoff.app_handoff` (2026-09-26).

A one-release stub for code importing the old path; import the names from the
pythontk root instead (``from pythontk import ...``).
"""
from pythontk.core_utils.deprecation import Deprecation

Deprecation.attributes(
    globals(),
    {
        "SEND_TO": "pythontk.core_utils.handoff.app_handoff.SEND_TO",
        "SAVE_AS": "pythontk.core_utils.handoff.app_handoff.SAVE_AS",
        "ROUND_TRIP": "pythontk.core_utils.handoff.app_handoff.ROUND_TRIP",
        "CARRIER_PARAM": "pythontk.core_utils.handoff.app_handoff.CARRIER_PARAM",
        "CARRIER_EXTENSIONS": "pythontk.core_utils.handoff.app_handoff.CARRIER_EXTENSIONS",
        "CARRIER_BY_EXTENSION": "pythontk.core_utils.handoff.app_handoff.CARRIER_BY_EXTENSION",
        "RIG_MODE_PARAM": "pythontk.core_utils.handoff.app_handoff.RIG_MODE_PARAM",
        "RIG_MODES": "pythontk.core_utils.handoff.app_handoff.RIG_MODES",
        "AppSpec": "pythontk.core_utils.handoff.app_handoff.AppSpec",
        "HandoffRequest": "pythontk.core_utils.handoff.app_handoff.HandoffRequest",
        "Payload": "pythontk.core_utils.handoff.app_handoff.Payload",
        "Deliverer": "pythontk.core_utils.handoff.app_handoff.Deliverer",
        "HandoffBridge": "pythontk.core_utils.handoff.app_handoff.HandoffBridge",
        "ScriptLaunchSpec": "pythontk.core_utils.handoff.app_handoff.ScriptLaunchSpec",
        "ScriptLaunchDeliverer": "pythontk.core_utils.handoff.app_handoff.ScriptLaunchDeliverer",
        "ScriptRunDeliverer": "pythontk.core_utils.handoff.app_handoff.ScriptRunDeliverer",
        "ScriptRoundTripDeliverer": "pythontk.core_utils.handoff.app_handoff.ScriptRoundTripDeliverer",
        "ScriptLaunchBridge": "pythontk.core_utils.handoff.app_handoff.ScriptLaunchBridge",
    },
    remove_in="0.13.0",
    since="2026-09-26",
)
