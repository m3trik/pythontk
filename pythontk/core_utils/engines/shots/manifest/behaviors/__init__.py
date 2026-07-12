# coding=utf-8
"""Behaviors — pure loading, schema, and keying math for shot keying recipes.

A behavior template defines attribute keyframe patterns (e.g. fade-in,
fade-out) anchored to a time range's start or end.  This package is the
DCC-agnostic core shared by the DCC toolkits: template discovery +
loading (:func:`load_behavior`, :func:`list_behaviors`, :func:`templates`),
the schema (:class:`BehaviorSpec` and its validators), the anchor/offset/
duration → absolute-keyframe math (:func:`resolve_keys`), and the pure
duration summation (:func:`compute_duration`, with audio resolution injected).

The scene-touching appliers (``apply_behavior``, ``verify_behavior``,
``apply_audio_clip``, ``apply_to_shots``) are **not** here — they live in the
DCC toolkits, which import this module for the pure core.

Package facade: the implementation lives in :mod:`._behaviors`. The public API
is re-exported here so ``from ...behaviors import X`` keeps working and
``mock.patch`` of ``...behaviors.X`` still takes effect for callers that read
the name off this package.  To intercept an *intra-module* call, patch
``...behaviors._behaviors.<name>`` instead, where the call is actually resolved.
"""
from pythontk.core_utils.engines.shots.manifest.behaviors._behaviors import (  # noqa: F401
    templates,
    load_behavior,
    list_behaviors,
    resolve_keys,
    compute_duration,
)
from pythontk.core_utils.engines.shots.manifest.behaviors._spec import (  # noqa: F401
    BehaviorSpec,
    KNOWN_VERIFY_MODES,
    validate_duration,
    validate_verify,
    validate_attributes,
    format_markdown,
)

__all__ = [
    "templates",
    "load_behavior",
    "list_behaviors",
    "resolve_keys",
    "compute_duration",
    "BehaviorSpec",
    "KNOWN_VERIFY_MODES",
    "validate_duration",
    "validate_verify",
    "validate_attributes",
    "format_markdown",
]
