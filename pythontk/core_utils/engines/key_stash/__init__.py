# !/usr/bin/python
# coding=utf-8
"""DCC-agnostic key-stash model: clips of keyframes parked outside the working
animation, retrievable later, inert until then.

- :mod:`~pythontk.core_utils.engines.key_stash.key_stash_model` models the
  store (:class:`KeyStash`, :class:`StashedClip`, the :class:`StashChanged`
  event) and its persistence / observer / singleton plumbing.

Scene-reaching behaviour (moving keys on and off the scene's curves, the
transient preview) lives in the mayatk / blendertk adapters, which subclass
:class:`KeyStash` and add ``stash`` / ``retrieve`` / ``drop`` / ``preview``.
"""

# Lazy-loaded via parent package - no explicit imports needed
