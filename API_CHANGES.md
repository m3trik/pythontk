# pythontk — API Changes

_Diff vs the last release (origin/main @ 892506a)._

## Removed (1)

_A break: each owes a `ptk.Deprecation` alias, or a `MOVED_NAMES` entry when the name left for another package._

- `core_utils/engines/shots/manifest/mapping/_mapping.py::Mapping.retired` — was `(name: str) -> Optional[Tuple[str, Dict[str, Any]]]`

## Retired (2)

_Deprecated aliases whose window ran out._

- `core_utils/engines/shots/manifest/manifest_model.py::ManifestModel.asset_column`
- `core_utils/engines/shots/manifest/manifest_model.py::ManifestModel.column_clipboard`

## Added (53)

- `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.definition_tables(self) -> Tuple[Dict[str, Any], Dict[str, Any]]`
- `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.export_cancelled(self) -> bool`
- `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.migrate_legacy_regex(self, store: Any, name_row: str, save: Callable[[str], Any]) -> Optional[str]`
- `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.note_retired_panel_state(self, store: Any) -> None`
- `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.note_retired_preset_rows(self, name: str, stored: Optional[Dict[str, Any]]) -> None`
- `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.resolve_export_dir(self, export_dir: Optional[str]) -> str`
- `core_utils/engines/shots/shot_report.py::ShotReport.removal_question(cls, shots: Sequence, scope: str) -> Tuple[str, str]`
- `core_utils/engines/shots/shot_report.py::ShotReport.scope_count(cls, shots: Sequence, scope: str) -> str`
- `core_utils/engines/shots/shot_report.py::ShotReport.scope_label(cls, scope: str) -> str`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter(class)`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.add_members(self, store: str, face_map: FaceMap) -> None`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.carrier_weights(self, prefix: str) -> List[str]`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.delete_weight(self, attr: str) -> bool`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.discard_stores(self, stores: List[str], ctx) -> None`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.ensure_weight(self, attr: str, default: float) -> str`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.extra_warnings(self) -> List[str]`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.face_count(self, obj: str) -> int`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.faces(self, faces=None) -> FaceMap`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.foreign_color_sets(self, objects: List[str], color_set: str) -> Dict[str, List[str]]`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.key_weight(self, attr: str, value: Optional[float], frame: Optional[float]) -> str`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.mask_path(self, suffix: str) -> str`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.members(self, store: str) -> Optional[FaceMap]`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.membership_stores(self, prefix: str) -> Iterable[Tuple[str, str]]`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.merge_stores(self, stores: List[str], ctx) -> None`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.mesh(self, name: str) -> Optional[str]`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.object_names(self, objects) -> List[str]`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.publish(self, records: Mapping[Any, Any]) -> None`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.remove_members(self, store: str) -> None`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.select(self, face_map: FaceMap) -> None`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.sync_weight(self, attr: str, value: float) -> None`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.uv_triangles(self, obj: str, indices: List[int], uv_set: Optional[str]) -> list`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.weight_exists(self, attr: str) -> bool`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsAdapter.write_colors(self, obj: str, channels: Mapping[int, List[float]], color_set: str) -> None`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel(class)`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.add_group(cls, name: str, faces=None, default: float = 1.0) -> str`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.bake_mask(cls, output_path: Optional[str] = None, resolution: int = 512, padding_px: int = 4, uv_set: Optional[str] = None) -> dict`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.bake_vertex_colors(cls, force: bool = False) -> dict`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.compact_slots(cls) -> List[int]`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.discard_carrier(cls, carriers, other, ctx) -> None`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.export_record(cls, ctx: 'ptk.ExportContext') -> Optional['ptk.Record']`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.key_weight(cls, name: str, value: Optional[float] = None, frame: Optional[float] = None, auto_keyable: bool = True) -> str`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.list_groups(cls) -> Dict[str, dict]`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.make_weights_keyable(cls, names=None) -> Dict[str, str]`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.merge_carrier(cls, carriers, other, ctx) -> None`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.refresh_export_metadata(cls) -> Optional[str]`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.remove_group(cls, name: str) -> None`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.remove_keyable_weights(cls, names=None) -> List[str]`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.select_group(cls, name: str) -> None`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.set_default(cls, name: str, default: float) -> None`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.transfer_in(cls, payload: dict, ctx: 'ptk.TransferContext') -> None`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.transfer_out(cls, ctx: 'ptk.TransferContext') -> Optional[dict]`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.validate(cls) -> List[str]`
- `core_utils/engines/textures/emissive_groups.py::EmissiveGroupsModel.weight_attrs(cls) -> Dict[str, str]`

## Deprecations (34)

_Live retirement debt, earliest deadline first. An **EXPIRED** row has outlived its window: delete the alias and its tests rather than moving the date. A **HELD** row is due by version, but its notice has not yet had its calendar window._

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
