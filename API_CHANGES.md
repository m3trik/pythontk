# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-08-06._

## Added (28)

- `core_utils/engines/textures/output_template.py::DeliveryBudget(class)`
- `core_utils/engines/textures/output_template.py::DeliveryBudget.check(self, width: int, height: int) -> List[str]`
- `core_utils/engines/textures/output_template.py::DeliveryBudget.from_dict(cls, d: dict) -> 'DeliveryBudget'`
- `core_utils/engines/textures/output_template.py::DeliveryBudget.to_dict(self) -> dict`
- `core_utils/engines/textures/output_template.py::OutputTemplates.budget(cls, profile: Optional[str]) -> DeliveryBudget`
- `core_utils/package_manager.py::PackageManager.latest_versions(self, package_names, timeout=None)`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.set_glb_base_color(cls, glb_path: str, base_color: Dict[str, Dict[str, Any]]) -> List[Dict]`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.set_glb_emissive(cls, glb_path: str, emissive: Dict[str, Dict[str, Any]]) -> List[Dict]`
- `net_utils/preview_server.py::PreviewBridge(class)`
- `net_utils/preview_server.py::PreviewBridge.params_defaults(self) -> Dict[str, Any]`
- `net_utils/preview_server.py::PreviewBridge.push(self, objects: Optional[List[Any]] = None, whole_scene: bool = False, open_browser: Union[bool, str] = 'auto', **params: Any) -> Optional[Dict[str, Any]]`
- `net_utils/preview_server.py::PreviewBridge.sidecar_summary(result: Optional[Dict[str, Any]]) -> str`
- `net_utils/preview_server.py::PreviewBridge.stop(self) -> None`
- `net_utils/preview_server.py::PreviewBridge.url(self) -> Optional[str]`
- `net_utils/preview_server.py::PreviewDeliverer(class)`
- `net_utils/preview_server.py::PreviewDeliverer.deliver(self, bridge, payload: Payload, request: HandoffRequest) -> Optional[Dict[str, Any]]`
- `net_utils/preview_server.py::PreviewDeliverer.ensure_server(self) -> PreviewServer`
- `net_utils/preview_server.py::PreviewServer(class)`
- `net_utils/preview_server.py::PreviewServer.has_viewer(self) -> bool`
- `net_utils/preview_server.py::PreviewServer.is_running(self) -> bool`
- `net_utils/preview_server.py::PreviewServer.manifest(self) -> Dict[str, Any]`
- `net_utils/preview_server.py::PreviewServer.open_in_browser(self) -> bool`
- `net_utils/preview_server.py::PreviewServer.port(self) -> Optional[int]`
- `net_utils/preview_server.py::PreviewServer.publish(self, src: Union[str, Path], name: Optional[str] = None, move: bool = False) -> int`
- `net_utils/preview_server.py::PreviewServer.start(self) -> 'PreviewServer'`
- `net_utils/preview_server.py::PreviewServer.stop(self) -> None`
- `net_utils/preview_server.py::PreviewServer.url(self) -> Optional[str]`
- `net_utils/preview_server.py::PreviewServer.version(self) -> int`

## Signature changed (4)

- `core_utils/engines/textures/map_optimizer.py::MapOptimizer.assess`
  - was: `(cls, texture_path: str, max_size: int = None, force_pot: bool = False, optimize_bit_depth: bool = True, map_type: str = None, allow_palette: bool = False, image: 'Image.Image' = None, output_type: str = None, output_profile: str = None, predict_size: bool = False) -> Dict[str, Any]`
  - now: `(cls, texture_path: str, max_size: int = None, force_pot: Optional[bool] = None, optimize_bit_depth: bool = True, map_type: str = None, allow_palette: bool = False, image: 'Image.Image' = None, output_type: str = None, output_profile: str = None, predict_size: bool = False, enforce_budget: bool = False) -> Dict[str, Any]`
- `core_utils/engines/textures/map_optimizer.py::MapOptimizer.optimize_map`
  - was: `(cls, texture_path: str, output_dir: str = None, output_type: str = None, max_size: int = None, force_pot: bool = False, suffix_old: str = None, suffix_opt: str = None, old_files_folder: str = None, optimize_bit_depth: bool = True, check_existing: bool = False, map_type: str = None, allow_palette: bool = False, output_profile: str = None) -> str`
  - now: `(cls, texture_path: str, output_dir: str = None, output_type: str = None, max_size: int = None, force_pot: Optional[bool] = None, suffix_old: str = None, suffix_opt: str = None, old_files_folder: str = None, optimize_bit_depth: bool = True, check_existing: bool = False, map_type: str = None, allow_palette: bool = False, output_profile: str = None, enforce_budget: bool = False) -> str`
- `core_utils/engines/textures/map_optimizer.py::MapOptimizer.plan`
  - was: `(cls, image: 'Image.Image', max_size: Optional[int] = None, force_pot: bool = False, optimize_bit_depth: bool = True, map_type_key: Optional[str] = None, allow_palette: bool = False) -> List[Op]`
  - now: `(cls, image: 'Image.Image', max_size: Optional[int] = None, force_pot: bool = False, optimize_bit_depth: bool = True, map_type_key: Optional[str] = None, allow_palette: bool = False, pot_mode: str = 'nearest') -> List[Op]`
- `core_utils/package_manager.py::PackageManager.latest_version`
  - was: `(self, package_name)`
  - now: `(self, package_name, timeout=None)`
