# pythontk — API Changes

_Diff vs the last release (origin/main @ 60d43ac). Generated 2026-08-19._

## Added (37)

- `core_utils/app_handoff.py::AppSpec.available(self) -> bool`
- `core_utils/app_handoff.py::AppSpec.path(self) -> Optional[str]`
- `core_utils/app_handoff.py::AppSpec.refresh(self) -> Optional[str]`
- `core_utils/engines/textures/map_optimizer.py::MapOptimizer.describe_size_clamp(cls, max_size: Any, template: Optional[str] = None, logger: Optional[Any] = None) -> str`
- `core_utils/engines/textures/map_optimizer.py::MapOptimizer.resolve_size_clamp(cls, max_size: Any, template: Optional[str] = None, logger: Optional[Any] = None) -> Dict[str, Any]`
- `core_utils/engines/textures/map_registry.py::MapRegistry.split_duplicate_token(cls, name_only: str) -> Tuple[str, str]`
- `core_utils/status_badge.py::StatusBadge.discover_module_names(test_dir: PathLike) -> set`
- `core_utils/status_badge.py::StatusBadge.gate(cls, expected, ran, passed: int, failed: int) -> Tuple[bool, str]`
- `core_utils/status_badge.py::StatusBadge.is_import_standin(cls, test) -> bool`
- `core_utils/status_badge.py::StatusBadge.module_of(cls, test) -> str`
- `file_utils/_file_utils.py::FileUtils.relativize_output_dir(cls, path: str, base: Optional[str]) -> str`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.suspect_orm_materials(cls, glb: GlbTarget, *, described: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, str]]`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.verify_glb(cls, glb: GlbTarget) -> Dict[str, Any]`
- `geo_utils/uv_transfer.py::TransferTable(class)`
- `geo_utils/uv_transfer.py::TransferTable.coverage(self) -> 'np.ndarray'`
- `geo_utils/uv_transfer.py::TransferTable.frames(self) -> 'np.ndarray'`
- `geo_utils/uv_transfer.py::TransferTable.mask(self) -> 'np.ndarray'`
- `geo_utils/uv_transfer.py::TransferTable.nbytes(self) -> int`
- `geo_utils/uv_transfer.py::TransferTable.passes(self) -> int`
- `geo_utils/uv_transfer.py::UvTransfer(class)`
- `geo_utils/uv_transfer.py::UvTransfer.build(cls, src_tris, dst_tris, size: Union[int, Tuple[int, int]], *, supersample: int = 2, source_ids=None) -> TransferTable`
- `geo_utils/uv_transfer.py::UvTransfer.load_map(path: str) -> Tuple['np.ndarray', float]`
- `geo_utils/uv_transfer.py::UvTransfer.merge_layouts(cls, jobs: Dict[str, Dict[str, Any]], name: str, *, probe_size: int = 256) -> Dict[str, Dict[str, Any]]`
- `geo_utils/uv_transfer.py::UvTransfer.normal_convention(cls, path: str, override: Optional[str] = None) -> str`
- `geo_utils/uv_transfer.py::UvTransfer.pad(cls, image, coverage, width: int = -1) -> 'np.ndarray'`
- `geo_utils/uv_transfer.py::UvTransfer.save_map(path: str, arr: 'np.ndarray', value_max: float = 255.0) -> str`
- `geo_utils/uv_transfer.py::UvTransfer.transfer(cls, table: TransferTable, sources, *, source_masks=None, bilinear: bool = True) -> Tuple['np.ndarray', 'np.ndarray']`
- `geo_utils/uv_transfer.py::UvTransfer.transfer_materials(cls, jobs: Dict[str, Dict[str, Any]], *, output_dir: str, channels: Optional[Sequence[str]] = None, size: Optional[int] = None, supersample: int = 2, padding: int = -1, name_format: str = '{material}_{channel}', normal_convention: Optional[str] = None, source_mask_from_uvs: bool = True, log=None) -> Dict[str, Dict[str, str]]`
- `geo_utils/uv_transfer.py::UvTransfer.transfer_normals(cls, table: TransferTable, sources, *, convention: str = 'opengl', source_masks=None, bilinear: bool = True, value_range: Tuple[float, float] = (0.0, 255.0)) -> Tuple['np.ndarray', 'np.ndarray']`
- `geo_utils/uv_transfer.py::UvTransfer.triangle_frames(cls, src_tris, dst_tris) -> 'np.ndarray'`
- `net_utils/preview_server.py::PreviewPassContext(class)`
- `net_utils/preview_server.py::PreviewPassContext.logger(self)`
- `net_utils/preview_server.py::PreviewPassContext.sidecar(self) -> Optional[Dict[str, Any]]`
- `net_utils/preview_server.py::PreviewServer.add_script(self, name: str, path: Optional[Union[str, Path]] = None) -> 'PreviewServer'`
- `net_utils/preview_server.py::PreviewServer.remove_script(self, name: str) -> 'PreviewServer'`
- `net_utils/preview_server.py::PreviewServer.scripts(self) -> tuple`
- `net_utils/preview_server.py::PreviewServer.set_scripts(self, scripts: Optional[Union[Dict[str, Any], List[str], tuple]]) -> 'PreviewServer'`

## Signature changed (2)

- `core_utils/cli.py::CLI.add_connection_args`
  - was: `(parser: argparse.ArgumentParser, default_host: str = DEFAULT_HOST, default_user: str = DEFAULT_USER, default_target: str = DEFAULT_CRED_TARGET) -> argparse.ArgumentParser`
  - now: `(parser: argparse.ArgumentParser, default_host: Optional[str] = None, default_user: Optional[str] = None, default_target: Optional[str] = None) -> argparse.ArgumentParser`
- `net_utils/preview_server.py::PreviewBridge.push`
  - was: `(self, objects: Optional[List[Any]] = None, whole_scene: bool = False, open_browser: Union[bool, str] = 'auto', texture_format: Optional[str] = None, **params: Any) -> Optional[Dict[str, Any]]`
  - now: `(self, objects: Optional[List[Any]] = None, whole_scene: bool = False, open_browser: Union[bool, str] = 'auto', texture_format: Optional[str] = None, scripts: Optional[Union[Dict[str, Any], List[str], tuple]] = None, **params: Any) -> Optional[Dict[str, Any]]`
