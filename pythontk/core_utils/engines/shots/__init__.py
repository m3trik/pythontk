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
  boundary / clustering math that a DCC's scene-acquisition feeds, including
  :meth:`ShotDetection.cluster_spans`, the one key-timing overlap grouping.
- :mod:`~pythontk.core_utils.engines.shots.shot_sequencer` is the sequencer's
  ripple-editing orchestration (:class:`ShotSequencer`); the DCC sequencers
  subclass it and supply scene hooks (bounds-only on its own).
- :mod:`~pythontk.core_utils.engines.shots.shot_apply` commits a plan via
  injected writer callables (bounds-only by default).
- :mod:`~pythontk.core_utils.engines.shots.shot_transfer` encodes a store as
  the hand-off manifest's ``shots`` section and decodes it against another
  scene (the DCC adapters' ``export_transfer`` / ``apply_transfer``).
- :mod:`~pythontk.core_utils.engines.shots.shot_ledger` records the edits the
  shot system authors on the animator's curves (gap holds, boundary keys), so
  a moved boundary can take them back (:class:`ShotEditLedger`).
- :mod:`~pythontk.core_utils.engines.shots.effect_recipe` is how each render
  effect and audio clip is keyed, once per scene (:class:`EffectRecipe`, the
  store's ``effect_recipe``): the panels edit it and the manifest keys with it.
- :mod:`~pythontk.core_utils.engines.shots.shot_report` words what a Shots
  panel says about the store -- the sequence at a glance and the outcome of a
  trim / pad -- once for both hosts (:class:`ShotReport`).
- :mod:`~pythontk.core_utils.engines.shots.manifest` is the production-CSV ->
  shot-plan pipeline (model, mapping files, keying behaviours, range
  resolution, the compute-then-commit :class:`ShotManifest`).

A store's export record is declared in the scene-export engine
(:mod:`~pythontk.core_utils.engines.scene_export`): :class:`ShotStore` produces
its ``SceneRecords`` entry from the export's ``ExportContext``.

Scene-reaching behaviour (framerate, animation queries, region detection,
export projection, name resolution) is exposed as overridable hooks with pure
defaults; mayatk and blendertk subclass :class:`ShotStore` and override them.
"""

# Lazy-loaded via parent package - no explicit imports needed
