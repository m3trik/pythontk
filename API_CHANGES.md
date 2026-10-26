# pythontk — API Changes

_Diff vs the last release (origin/main @ 19ef6ab)._

## Added (2)

- `core_utils/module_resolver.py::import_footprint(package: str) -> Dict[str, List[str]]`
- `img_utils/_img_utils.py::ImageFormat(class)`

## Deprecations (36)

_Live retirement debt, earliest deadline first. An **EXPIRED** row has outlived its window: delete the alias and its tests rather than moving the date. A **HELD** row is due by version, but its notice has not yet had its calendar window._

- **HELD** `core_utils/engines/shots/manifest/manifest_model.py::ManifestModel.asset_column` — remove in 0.14.0, not before 2026-11-09
- **HELD** `core_utils/engines/shots/manifest/manifest_model.py::ManifestModel.column_clipboard` — remove in 0.14.0, not before 2026-11-09
- **HELD** `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.confirm_check_override` — remove in 0.14.0, not before 2026-11-03
- **HELD** `core_utils/engines/scene_export/export_profile.py::ExportProfile.baked_reflections_default` — remove in 0.17.0, not before 2026-11-18
- **HELD** `core_utils/engines/scene_export/export_profile.py::ExportRun.baked_reflection_level` — remove in 0.17.0, not before 2026-11-18
- **HELD** `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.generate_log_file_path` — remove in 0.17.0, not before 2026-11-13
- `geo_utils/articulation/record.py::ArticulationWeb` — remove in 0.18.0, not before 2026-11-23
- `geo_utils/articulation/record.py::ArticulationWebGrab` — remove in 0.18.0, not before 2026-11-23
- `geo_utils/articulation/record.py::ArticulationWebJoint` — remove in 0.18.0, not before 2026-11-23
- `geo_utils/articulation/record.py::ArticulationWebRig` — remove in 0.18.0, not before 2026-11-23
- `file_utils/mesh_convert/export_verify.py::ExportVerifier.check_glb_envelope` — remove in 0.18.0, not before 2026-11-23
- `core_utils/engines/scene_export/scene_records.py::GlbBlock.key` — remove in 0.18.0, not before 2026-11-23
- `file_utils/mesh_convert/glb/pipeline.py::GlbPipeline.envelope` — remove in 0.18.0, not before 2026-11-23
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.SHADOW_PLANE_DEFAULTS` — remove in 0.18.0, not before 2026-11-23
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.apply_scene_sidecar` — remove in 0.18.0, not before 2026-11-23
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.build_fbx_handoff` — remove in 0.18.0, not before 2026-11-23
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.build_scene_sidecar` — remove in 0.18.0, not before 2026-11-23
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.embed_scene_sidecar` — remove in 0.18.0, not before 2026-11-23
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.fix_glb_lightmap_metadata` — remove in 0.18.0, not before 2026-11-23
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.read_glb_lightmap_manifest` — remove in 0.18.0, not before 2026-11-23
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.read_scene_sidecar` — remove in 0.18.0, not before 2026-11-23
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.rendering_policy` — remove in 0.18.0, not before 2026-11-23
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.sidecar_foreign_packings` — remove in 0.18.0, not before 2026-11-23
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.strip_fbx_handoff` — remove in 0.18.0, not before 2026-11-23
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.without_locate_hints` — remove in 0.18.0, not before 2026-11-23
- `net_utils/preview/bridge.py::PreviewBridge.sidecar_summary` — remove in 0.18.0, not before 2026-11-23
- `core_utils/engines/scene_export/scene_records.py::RecordSpec.version_key` — remove in 0.18.0, not before 2026-11-23
- `core_utils/engines/scene_export/scene_records.py::RecordSpec.web` — remove in 0.18.0, not before 2026-11-23
- `core_utils/engines/textures/region_masks.py::RegionMaskManifest.schema` — remove in 0.18.0, not before 2026-11-23
- `core_utils/engines/scene_export/scene_records.py::SceneRecords.HANDOFF_ASSEMBLY` — remove in 0.18.0, not before 2026-11-23
- `core_utils/engines/scene_export/scene_records.py::SceneRecords.HANDOFF_INSTRUCTIONS` — remove in 0.18.0, not before 2026-11-23
- `core_utils/engines/scene_export/scene_records.py::SceneRecords.handoff_block` — remove in 0.18.0, not before 2026-11-23
- `core_utils/engines/scene_export/scene_records.py::SceneRecords.rendering_policy` — remove in 0.18.0, not before 2026-11-23
- `core_utils/engines/scene_export/scene_records.py::SceneRecords.web_projected` — remove in 0.18.0, not before 2026-11-23
- `core_utils/engines/scene_export/scene_records.py::SceneRecords.web_shape` — remove in 0.18.0, not before 2026-11-23
- `core_utils/engines/scene_export/scene_records.py::WebProjection` — remove in 0.18.0, not before 2026-11-23
