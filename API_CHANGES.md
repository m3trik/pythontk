# pythontk — API Changes

_Diff vs the last release (origin/main @ a0076c9)._

## Added (1)

- `file_utils/mesh_convert/glb/pipeline.py::GlbPipeline.publishes_as_authored(source: str) -> bool`

## Deprecations (6)

_Live retirement debt, earliest deadline first. An **EXPIRED** row has outlived its window: delete the alias and its tests rather than moving the date. A **HELD** row is due by version, but its notice has not yet had its calendar window._

- **HELD** `core_utils/engines/shots/manifest/manifest_model.py::ManifestModel.asset_column` — remove in 0.14.0, not before 2026-11-09
- **HELD** `core_utils/engines/shots/manifest/manifest_model.py::ManifestModel.column_clipboard` — remove in 0.14.0, not before 2026-11-09
- **HELD** `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.confirm_check_override` — remove in 0.14.0, not before 2026-11-03
- `core_utils/engines/scene_export/export_profile.py::ExportProfile.baked_reflections_default` — remove in 0.17.0, not before 2026-11-18
- `core_utils/engines/scene_export/export_profile.py::ExportRun.baked_reflection_level` — remove in 0.17.0, not before 2026-11-18
- `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.generate_log_file_path` — remove in 0.17.0, not before 2026-11-13
