# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-08-05._

## Added (15)

- `core_utils/app_handoff.py::HandoffBridge.headless_app_path(self) -> Optional[str]`
- `core_utils/app_handoff.py::HandoffBridge.import_roots(*packages: str) -> List[str]`
- `core_utils/app_handoff.py::ScriptLaunchBridge.resolve_save_path(cls, out_path: str) -> str`
- `core_utils/app_handoff.py::ScriptLaunchBridge.save_as(self, out_path: str, objects: Optional[List[Any]] = None, *, template: Optional[str] = None, params: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **extras: Any) -> Optional[Dict[str, Any]]`
- `core_utils/app_handoff.py::ScriptRunDeliverer(class)`
- `core_utils/app_handoff.py::ScriptRunDeliverer.deliver(self, bridge: HandoffBridge, payload: Payload, request: HandoffRequest) -> Optional[Dict[str, Any]]`
- `core_utils/app_handoff.py::ScriptRunDeliverer.run(app_exe, script_text, *, artifact, launch_args, timeout, env=None)`
- `core_utils/app_launcher.py::AppLauncher.process_environ()`
- `core_utils/engines/textures/map_factory/_map_factory.py::MapFactory.get_tile_token(cls, filepath_or_filename: str) -> str`
- `core_utils/engines/textures/map_factory/_map_factory.py::MapFactory.resolve_normal_maps(cls, sorted_maps: Dict[str, Any], target_format: Optional[str] = None, convert: bool = True) -> Dict[str, Dict[str, str]]`
- `core_utils/engines/textures/map_registry.py::MapRegistry.resolve_type_from_channel(cls, channel: str) -> Optional[str]`
- `core_utils/engines/textures/map_registry.py::MapRegistry.select_normal_type(cls, available) -> Optional[str]`
- `core_utils/engines/textures/map_registry.py::MapRegistry.split_tile_token(cls, name_only: str) -> Tuple[str, str]`
- `core_utils/script_template.py::ScriptTemplate.declared_modes(template_path, field: str = 'BRIDGE_MODES') -> Optional[Tuple[str, ...]]`
- `file_utils/temp_artifacts.py::TempArtifacts.dir_path(self, name: Optional[str] = None, create: bool = True) -> str`
