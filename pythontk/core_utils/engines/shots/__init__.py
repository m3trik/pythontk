# !/usr/bin/python
# coding=utf-8
"""DCC-agnostic shot model, planner, detection math, and apply skeleton.

The core shots layer is complete on its own:

- :mod:`~pythontk.core_utils.engines.shots.shot_model` models the topology
  (:class:`ShotBlock`, :class:`ShotStore`, typed store events, the
  :class:`ScenePersistence` protocol).
- :mod:`~pythontk.core_utils.engines.shots.shot_plan` resolves multi-shot
  timeline transformations into a pure :class:`MovePlan` (collision-safe topo
  sort).
- :mod:`~pythontk.core_utils.engines.shots.shot_detection` supplies the pure
  boundary / clustering math that a DCC's scene-acquisition feeds.
- :mod:`~pythontk.core_utils.engines.shots.shot_apply` commits a plan via
  injected writer callables (bounds-only by default).

Scene-reaching behaviour (framerate, animation queries, region detection,
export projection, name resolution) is exposed as overridable hooks with pure
defaults; mayatk and blendertk subclass :class:`ShotStore` and override them.
"""

# Lazy-loaded via parent package - no explicit imports needed
