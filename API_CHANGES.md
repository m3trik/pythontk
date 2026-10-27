# pythontk — API Changes

_Diff vs the last release (origin/main @ f97442f)._

## Added (19)

- `__init__.py::MOVED_NAMES(constant)`
- `core_utils/engines/scene_export/scene_records.py::SceneRecords.forget_registered_hooks(cls) -> None`
- `core_utils/engines/scene_export/scene_records.py::SceneRecords.hook_row(cls, row: Optional[Tuple[str, str, str]], host: str, role: str) -> Optional[Tuple[str, str, str]]`
- `core_utils/engines/scene_export/scene_records.py::SceneRecords.producer_rows(cls, table: Mapping[Any, Tuple[str, str, str]], host: str) -> Dict[Any, Tuple[str, str, str]]`
- `core_utils/engines/scene_export/scene_records.py::SceneRecords.registered_hooks(cls, host: str, role: str) -> Dict[str, Tuple[str, str, Optional[str]]]`
- `core_utils/engines/scene_export/scene_records.py::SceneRecords.stager_rows(cls, table: Mapping[str, Tuple[str, str, str, Optional[str]]], host: str) -> Dict[str, Tuple[str, str, str, Optional[str]]]`
- `core_utils/engines/scene_export/scene_store.py::SceneStoreBase.owner_rows(cls) -> Dict[str, Tuple[str, str]]`
- `core_utils/module_resolver.py::MovedImportError(class)`
- `core_utils/module_resolver.py::MovedNameError(class)`
- `core_utils/package_manager.py::PackageManager.entry_points(group: str) -> list`
- `file_utils/mesh_convert/export_fingerprint.py::ExportFingerprint(class)`
- `file_utils/mesh_convert/export_fingerprint.py::ExportFingerprint.against(cls, path: str, baseline: str, update: bool = False, tolerance: float = 0.001) -> List[str]`
- `file_utils/mesh_convert/export_fingerprint.py::ExportFingerprint.diff(cls, baseline: Any, current: Any, tolerance: float = 0.001) -> List[str]`
- `file_utils/mesh_convert/export_fingerprint.py::ExportFingerprint.of(cls, path: str) -> Dict[str, Any]`
- `file_utils/mesh_convert/export_fingerprint.py::ExportFingerprint.of_fbx(cls, path: str) -> Dict[str, Any]`
- `file_utils/mesh_convert/export_fingerprint.py::ExportFingerprint.of_glb(cls, path: str) -> Dict[str, Any]`
- `file_utils/mesh_convert/export_fingerprint.py::ExportFingerprint.read(path: str) -> Dict[str, Any]`
- `file_utils/mesh_convert/export_fingerprint.py::ExportFingerprint.write(fingerprint: Dict[str, Any], path: str) -> str`
- `file_utils/mesh_convert/fbx_file.py::FbxFile.object_name(self, record: Dict[str, Any]) -> Optional[str]`

## Deprecations (36)

_Live retirement debt, earliest deadline first. An **EXPIRED** row has outlived its window: delete the alias and its tests rather than moving the date. A **HELD** row is due by version, but its notice has not yet had its calendar window._

- **HELD** `core_utils/engines/shots/manifest/manifest_model.py::ManifestModel.asset_column` — remove in 0.14.0, not before 2026-11-09
- **HELD** `core_utils/engines/shots/manifest/manifest_model.py::ManifestModel.column_clipboard` — remove in 0.14.0, not before 2026-11-09
- **HELD** `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.confirm_check_override` — remove in 0.14.0, not before 2026-11-03
- **HELD** `core_utils/engines/scene_export/export_profile.py::ExportProfile.baked_reflections_default` — remove in 0.17.0, not before 2026-11-18
- **HELD** `core_utils/engines/scene_export/export_profile.py::ExportRun.baked_reflection_level` — remove in 0.17.0, not before 2026-11-18
- **HELD** `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.generate_log_file_path` — remove in 0.17.0, not before 2026-11-13
- **HELD** `geo_utils/articulation/record.py::ArticulationWeb` — remove in 0.18.0, not before 2026-11-23
- **HELD** `geo_utils/articulation/record.py::ArticulationWebGrab` — remove in 0.18.0, not before 2026-11-23
- **HELD** `geo_utils/articulation/record.py::ArticulationWebJoint` — remove in 0.18.0, not before 2026-11-23
- **HELD** `geo_utils/articulation/record.py::ArticulationWebRig` — remove in 0.18.0, not before 2026-11-23
- **HELD** `file_utils/mesh_convert/export_verify.py::ExportVerifier.check_glb_envelope` — remove in 0.18.0, not before 2026-11-23
- **HELD** `core_utils/engines/scene_export/scene_records.py::GlbBlock.key` — remove in 0.18.0, not before 2026-11-23
- **HELD** `file_utils/mesh_convert/glb/pipeline.py::GlbPipeline.envelope` — remove in 0.18.0, not before 2026-11-23
- **HELD** `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.SHADOW_PLANE_DEFAULTS` — remove in 0.18.0, not before 2026-11-23
- **HELD** `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.apply_scene_sidecar` — remove in 0.18.0, not before 2026-11-23
- **HELD** `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.build_fbx_handoff` — remove in 0.18.0, not before 2026-11-23
- **HELD** `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.build_scene_sidecar` — remove in 0.18.0, not before 2026-11-23
- **HELD** `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.embed_scene_sidecar` — remove in 0.18.0, not before 2026-11-23
- **HELD** `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.fix_glb_lightmap_metadata` — remove in 0.18.0, not before 2026-11-23
- **HELD** `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.read_glb_lightmap_manifest` — remove in 0.18.0, not before 2026-11-23
- **HELD** `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.read_scene_sidecar` — remove in 0.18.0, not before 2026-11-23
- **HELD** `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.rendering_policy` — remove in 0.18.0, not before 2026-11-23
- **HELD** `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.sidecar_foreign_packings` — remove in 0.18.0, not before 2026-11-23
- **HELD** `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.strip_fbx_handoff` — remove in 0.18.0, not before 2026-11-23
- **HELD** `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.without_locate_hints` — remove in 0.18.0, not before 2026-11-23
- **HELD** `net_utils/preview/bridge.py::PreviewBridge.sidecar_summary` — remove in 0.18.0, not before 2026-11-23
- **HELD** `core_utils/engines/scene_export/scene_records.py::RecordSpec.version_key` — remove in 0.18.0, not before 2026-11-23
- **HELD** `core_utils/engines/scene_export/scene_records.py::RecordSpec.web` — remove in 0.18.0, not before 2026-11-23
- **HELD** `core_utils/engines/textures/region_masks.py::RegionMaskManifest.schema` — remove in 0.18.0, not before 2026-11-23
- **HELD** `core_utils/engines/scene_export/scene_records.py::SceneRecords.HANDOFF_ASSEMBLY` — remove in 0.18.0, not before 2026-11-23
- **HELD** `core_utils/engines/scene_export/scene_records.py::SceneRecords.HANDOFF_INSTRUCTIONS` — remove in 0.18.0, not before 2026-11-23
- **HELD** `core_utils/engines/scene_export/scene_records.py::SceneRecords.handoff_block` — remove in 0.18.0, not before 2026-11-23
- **HELD** `core_utils/engines/scene_export/scene_records.py::SceneRecords.rendering_policy` — remove in 0.18.0, not before 2026-11-23
- **HELD** `core_utils/engines/scene_export/scene_records.py::SceneRecords.web_projected` — remove in 0.18.0, not before 2026-11-23
- **HELD** `core_utils/engines/scene_export/scene_records.py::SceneRecords.web_shape` — remove in 0.18.0, not before 2026-11-23
- **HELD** `core_utils/engines/scene_export/scene_records.py::WebProjection` — remove in 0.18.0, not before 2026-11-23

## Signature changed (1)

- `core_utils/module_resolver.py::bootstrap_package`
  - was: `(module_globals: MutableMapping[str, Any], *, include: Optional[IncludeMapping] = None, module_to_parent: Optional[Mapping[str, str]] = None, eager: bool = False, allow_getattr: bool = True, install_legacy_helpers: bool = True, on_import_error: Optional[Callable[[str, Exception], None]] = None, method_predicate: Optional[Callable[[str], bool]] = None, custom_getattr: Optional[Callable[[str], Any]] = None, lazy_import: Optional[bool] = None, set_all: bool = True) -> PackageResolverHandle`
  - now: `(module_globals: MutableMapping[str, Any], *, include: Optional[IncludeMapping] = None, module_to_parent: Optional[Mapping[str, str]] = None, eager: bool = False, allow_getattr: bool = True, install_legacy_helpers: bool = True, on_import_error: Optional[Callable[[str, Exception], None]] = None, method_predicate: Optional[Callable[[str], bool]] = None, custom_getattr: Optional[Callable[[str], Any]] = None, lazy_import: Optional[bool] = None, set_all: bool = True, moved: Optional[MovedMapping] = None) -> PackageResolverHandle`
