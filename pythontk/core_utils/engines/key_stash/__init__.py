# !/usr/bin/python
# coding=utf-8
"""DCC-agnostic key-stash model: clips of keyframes parked outside the working
animation, retrievable later, inert until then.

- :mod:`~pythontk.core_utils.engines.key_stash.key_stash_model` models the
  store (:class:`KeyStash`, :class:`StashedClip`, the :class:`StashChanged`
  event), its persistence / observer / singleton plumbing, the host adapter
  seam (:class:`KeyStashHost`, :class:`PureKeyStashHost`) and the saved
  record's golden case (:class:`KeyStashConformance`).

Scene-reaching behaviour (moving keys on and off the scene's curves, the
transient preview) lives in the mayatk / blendertk bindings, which set
:attr:`KeyStash.HOST` to their adapter and add ``stash`` / ``retrieve`` /
``drop`` / ``preview``, overriding nothing.
"""

# Lazy-loaded via parent package - no explicit imports needed
