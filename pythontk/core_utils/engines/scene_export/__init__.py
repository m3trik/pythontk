# !/usr/bin/python
# coding=utf-8
"""DCC-agnostic scene-export core: what an export records, decides and ships.

The pure half of both DCC Scene Exporters, layered model -> planner -> hooks:

- :mod:`~pythontk.core_utils.engines.scene_export.scene_records` declares every
  tool-authored scene record once (:class:`Scope`, :class:`Kind`,
  :class:`Merge`, :class:`RecordSpec`, the :class:`SceneRecords` catalogue).
- :mod:`~pythontk.core_utils.engines.scene_export.scene_store` is the storage
  contract each DCC's ``DataNodes`` implements (:class:`SceneStoreBase`) and
  the crossings written on top of it.
- :mod:`~pythontk.core_utils.engines.scene_export.export_snapshot` assembles
  the records an export ships from their producers (:class:`ExportContext`,
  :class:`ExportSnapshot`): compute, then commit, once.
- :mod:`~pythontk.core_utils.engines.scene_export.record_transfer` crosses
  records into another scene by their declarations (:class:`RecordTransfer`,
  :class:`TransferContext`).
- :mod:`~pythontk.core_utils.engines.scene_export.export_profile` is the
  panels' shared contract: widget values -> a run configuration
  (:class:`ExportProfile`, :class:`ExportRun`).
- :mod:`~pythontk.core_utils.engines.scene_export.hierarchy_baseline` is the
  change-detection baseline an export diffs the scene's hierarchy against
  (:class:`HierarchyBaseline`) and its scene-stored half
  (:class:`HierarchyBaselineStore`).
- :mod:`~pythontk.core_utils.engines.scene_export.scene_exporter` is the
  exporter's orchestration shell (:class:`SceneExporterBase`: progress stream,
  check override, run config) and
  :mod:`~pythontk.core_utils.engines.scene_export.scene_data_sidecar` the
  scene-data sidecar's file format (:class:`SceneDataSidecarBase`); mayatk and
  blendertk subclass both and supply the scene I/O through hooks.

Scene-reaching behaviour (reading carriers, writing the file, the DCC's
dialogs) is the DCC adapters' -- mayatk / blendertk ``env_utils`` -- behind
overridable hooks with pure defaults.
"""

# Lazy-loaded via parent package - no explicit imports needed
