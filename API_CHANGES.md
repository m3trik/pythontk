# pythontk — API Changes

_Diff vs the last release (origin/main @ 412df2e)._

## Added (36)

- `core_utils/app_handoff.py::HandoffBridge.child_sys_path(entries: Optional[Sequence[str]] = None) -> List[str]`
- `core_utils/engines/textures/map_factory/_map_factory.py::MapFactory.dominant_texture_set(cls, paths: Iterable[str]) -> Optional[Tuple[str, str]]`
- `core_utils/export_profile.py::ExportProfile.baked_reflections_default(cls) -> str`
- `core_utils/export_profile.py::ExportProfile.glb_defaults(cls) -> Dict[str, Any]`
- `core_utils/export_profile.py::ExportProfile.glb_options(cls) -> Dict[str, Dict[str, Any]]`
- `core_utils/export_profile.py::ExportProfile.optimize_textures_tasks(choice: Any, template: Optional[str] = None) -> Dict[str, Any]`
- `core_utils/export_profile.py::ExportRun.baked_reflection_level(cls, value: Any) -> Optional[float]`
- `core_utils/export_profile.py::ExportRun.for_glb(cls, values: Mapping[str, Any]) -> Tuple['ExportRun', List[Tuple[str, str]]]`
- `core_utils/export_profile.py::ExportRun.glb_max_size(self, logger: Any = None) -> int`
- `core_utils/export_profile.py::ExportRun.glb_texture_params(self, logger: Any = None) -> Dict[str, Any]`
- `core_utils/export_profile.py::ExportRun.rendering(self) -> Dict[str, Dict[str, Any]]`
- `core_utils/upstream_patch.py::UpstreamPatch(class)`
- `core_utils/upstream_patch.py::UpstreamPatch.applied(self)`
- `core_utils/upstream_patch.py::UpstreamPatch.available(self) -> bool`
- `core_utils/upstream_patch.py::UpstreamPatch.detects(self, func: Callable[[], bool]) -> Callable[[], bool]`
- `core_utils/upstream_patch.py::UpstreamPatch.registry(cls) -> List['UpstreamPatch']`
- `core_utils/upstream_patch.py::UpstreamPatch.replaces(self, func: Callable) -> Callable`
- `core_utils/upstream_patch.py::UpstreamPatch.still_needed(self) -> bool`
- `file_utils/_file_utils.py::FileUtils.has_same_content(path_a: str, path_b: str) -> bool`
- `file_utils/_file_utils.py::FileUtils.is_same_file(path_a: str, path_b: str) -> bool`
- `file_utils/_file_utils.py::FileUtils.portable_path(cls, path: str, base: Optional[str]) -> str`
- `file_utils/_file_utils.py::FileUtils.resolve_portable_path(cls, stored: str, base: Optional[str]) -> str`
- `file_utils/_file_utils.py::FileUtils.unique_path(folder: str, stem: str, ext: str, taken: Optional[set] = None, claims: Optional[Union[Mapping, Iterable[str]]] = None, owners: Iterable[str] = (), avoid: Iterable[str] = ()) -> str`
- `file_utils/file_dependencies.py::FileDependencies(class)`
- `file_utils/file_dependencies.py::FileDependencies.claims(refs: Iterable[Sequence[str]]) -> Dict[str, FrozenSet[str]]`
- `file_utils/file_dependencies.py::FileDependencies.copy_files(cls, sources: Iterable[str], dest_dir: str, mode: str = 'copy') -> List[Tuple[str, str]]`
- `file_utils/file_dependencies.py::FileDependencies.find_files(names: Iterable[str], root: str) -> List[str]`
- `file_utils/file_dependencies.py::FileDependencies.relocate(cls, deps: Sequence[Dict[str, Any]], dest_dir: str, source_dir: str = '', mode: str = 'copy', dry_run: bool = False, find_files: Optional[Callable[[List[str], str], List[str]]] = None, copy: Optional[Callable[[List[str], str, str], List[Tuple[str, str]]]] = None) -> Dict[str, Any]`
- `file_utils/file_dependencies.py::FileDependencies.resolve(cls, refs: Iterable[Sequence[str]], search_dirs: Iterable[str] = (), walk_root: str = '', find_files: Optional[Callable[[List[str], str], List[str]]] = None, resolve_hint: Optional[Callable[[str, str], str]] = None) -> List[Dict[str, Any]]`
- `file_utils/file_dependencies.py::FileDependencies.search_dirs(deps: Iterable[Dict[str, Any]], then: Iterable[str] = ()) -> List[str]`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.rendering_policy(cls, overrides: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]`
- `img_utils/_img_utils.py::ImgUtils.compose_rect(outer: Optional[Sequence[float]], inner: Sequence[float]) -> List[float]`
- `img_utils/_img_utils.py::ImgUtils.denoise_image(cls, image: 'np.ndarray', mask: Optional['np.ndarray'] = None, radius: int = 2, strength: float = 3.0, noise: Optional[float] = None, outliers: float = 5.0) -> 'np.ndarray'`
- `net_utils/preview/deliverer.py::PreviewDeliverer.preflight(self, bridge, request: HandoffRequest) -> bool`
- `net_utils/preview/server.py::PreviewServer.save_snapshot(self, data: bytes, content_type: str = 'image/png') -> Dict[str, Any]`
- `net_utils/preview/server.py::SNAPSHOT_PATH(constant)`

## Signature changed (5)

- `core_utils/scene_records.py::SceneRecords.handoff_block`
  - was: `(cls, channels: Union[Iterable[str], Mapping[str, Any]], source: Optional[Mapping[str, str]] = None) -> Dict[str, Any]`
  - now: `(cls, channels: Union[Iterable[str], Mapping[str, Any]], source: Optional[Mapping[str, str]] = None, rendering: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]`
- `core_utils/scene_records.py::SceneRecords.rendering_policy`
  - was: `() -> Dict[str, Any]`
  - now: `(overrides: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.build_scene_sidecar`
  - was: `(cls, sections: Optional[Dict[str, Any]], source: Dict[str, str], asset: Optional[str] = None) -> Dict[str, Any]`
  - now: `(cls, sections: Optional[Dict[str, Any]], source: Dict[str, str], asset: Optional[str] = None, rendering: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]`
- `file_utils/mesh_convert/glb_pipeline.py::GlbPipeline.envelope`
  - was: `(cls, read_sections: Callable[[], Optional[Dict[str, Any]]], *, source: Dict[str, str], asset: Optional[str] = None, logger: Any = None) -> Dict[str, Any]`
  - now: `(cls, read_sections: Callable[[], Optional[Dict[str, Any]]], *, source: Dict[str, str], asset: Optional[str] = None, rendering: Optional[Dict[str, Dict[str, Any]]] = None, logger: Any = None) -> Dict[str, Any]`
- `net_utils/preview/bridge.py::PreviewBridge.push`
  - was: `(self, objects: Optional[List[Any]] = None, scope: str = 'selected', open_browser: Union[bool, str, None] = None, texture_format: Optional[str] = None, scripts: Optional[Union[Dict[str, Any], List[str], tuple]] = None, progress: Optional[Callable[[str], Any]] = None, data_export: Optional[Dict[str, Any]] = None, **params: Any) -> Optional[Dict[str, Any]]`
  - now: `(self, objects: Optional[List[Any]] = None, scope: str = 'selected', open_browser: Union[bool, str, None] = None, glb_options: Optional[Dict[str, Any]] = None, scripts: Optional[Union[Dict[str, Any], List[str], tuple]] = None, progress: Optional[Callable[[str], Any]] = None, data_export: Optional[Dict[str, Any]] = None, **params: Any) -> Optional[Dict[str, Any]]`
