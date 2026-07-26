# coding=utf-8
"""Behaviors — pure loading, schema, and keying math for shot keying recipes.

A behavior template defines attribute keyframe patterns (e.g. fade-in,
fade-out) anchored to a time range's start or end.  This package is the
DCC-agnostic core shared by the DCC toolkits: template discovery + loading,
the schema (:class:`BehaviorSpec`), the anchor/offset/duration → absolute
keyframe math, and the pure duration summation — all exposed as staticmethods
of :class:`Behaviors` (with audio resolution injected).

The scene-touching appliers (``apply_behavior``, ``verify_behavior``,
``apply_audio_clip``, ``apply_to_shots``) are **not** here — they live in the
DCC toolkits, which import this module for the pure core.

Package facade: the implementation lives in :mod:`._behaviors` / :mod:`._spec`.
The public classes are re-exported here so ``from ...behaviors import Behaviors``
keeps working; ``mock.patch`` of ``...behaviors.Behaviors.<method>`` takes
effect for callers that read the class off this package.  To intercept an
*intra-module* call, patch ``...behaviors._behaviors.Behaviors.<method>``.
"""

from pythontk.core_utils.engines.shots.manifest.behaviors._behaviors import (  # noqa: F401
    Behaviors,
)
from pythontk.core_utils.engines.shots.manifest.behaviors._spec import (  # noqa: F401
    BehaviorSpec,
    KNOWN_VERIFY_MODES,
)

__all__ = [
    "Behaviors",
    "BehaviorSpec",
    "KNOWN_VERIFY_MODES",
]
