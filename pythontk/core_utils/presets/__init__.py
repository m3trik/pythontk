# !/usr/bin/python
# coding=utf-8
"""Presets -- named snapshots of a tool's settings, and the library over all of them.

* :mod:`.store` -- ``PresetStore``: ONE tool's presets (a folder of files, a codec,
  read-only built-ins beside the user's own), ``PresetReadOnlyError``, ``Codec`` /
  ``JSON_CODEC``.
* :mod:`.library` -- ``PresetLibrary``: every store under the presets root at once
  (lock, collections, bundles, import/export).

The classes and codecs are registered at the pythontk root (``ptk.PresetStore``,
``ptk.PresetLibrary``, ``ptk.Codec``); another package imports those, not these paths.
"""
