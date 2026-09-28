# pythontk — API Changes

_Diff vs the last release (origin/main @ 0ab5a6f)._

## Added (55)

- `core_utils/app_launcher/_app_launcher.py::AppLauncher.ansi_safe_path(path, ascii_only=False)`
- `core_utils/app_launcher/_app_launcher.py::AppLauncher.python_args_via_env(argv, env=None)`
- `core_utils/engines/scene_export/scene_records.py::RecordSpec.stamp_keys(self) -> Optional[Tuple[str, ...]]`
- `core_utils/engines/scene_export/scene_records.py::SceneRecords.stamp_unsaved(cls, store, stamp: str) -> int`
- `core_utils/engines/scene_export/scene_records.py::SceneRecords.with_stamps(cls) -> List[RecordSpec]`
- `core_utils/engines/scene_export/scene_store.py::SceneStoreBase.path_records(cls) -> Dict[Tuple[Scope, str], Optional[str]]`
- `core_utils/engines/scene_export/scene_store.py::SceneStoreBase.respell_for_write(cls, target: str, old_base: Optional[str], first_save: bool = False) -> Dict[Tuple[Scope, str], Optional[str]]`
- `core_utils/engines/scene_export/scene_store.py::SceneStoreBase.restore_path_records(cls, snapshot: Mapping[Tuple[Scope, str], Optional[str]]) -> int`
- `core_utils/engines/scene_export/scene_store.py::SceneStoreBase.writer_stamp_of(cls, scene_path: Optional[str]) -> str`
- `core_utils/handoff/script_template.py::ScriptTemplate.child_path(path) -> str`
- `core_utils/presets/library.py::PresetEntry.description(self) -> str`
- `core_utils/presets/library.py::PresetLibrary.set_description(self, entries: Iterable[PresetEntry], text: Optional[str]) -> int`
- `core_utils/presets/library.py::PresetLibrary.set_hidden(self, entries: Iterable[PresetEntry], flag: bool = True) -> int`
- `core_utils/presets/store.py::HIDDEN_BUILTINS(constant)`
- `core_utils/presets/store.py::PresetStore.description(self, name: str) -> str`
- `core_utils/presets/store.py::PresetStore.is_hidden(self, name: str) -> bool`
- `core_utils/presets/store.py::PresetStore.set_hidden(self, name: str, hidden: bool = True) -> bool`
- `core_utils/process_exit.py::ProcessExit.register(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Callable[..., Any]`
- `core_utils/process_exit.py::ProcessExit.unregister(func: Callable[..., Any]) -> None`
- `core_utils/test_sandbox.py::TestSandbox.real_trash(cls) -> Iterator[None]`
- `core_utils/test_sandbox.py::TestSandbox.trash(cls) -> None`
- `file_utils/_file_utils.py::FileUtils.can_trash(path: str) -> bool`
- `file_utils/_file_utils.py::FileUtils.move_to_trash(path: str) -> Optional[str]`
- `file_utils/_file_utils.py::FileUtils.trash_name() -> str`
- `file_utils/file_dependencies.py::FileDependencies.set_aside(cls, path: str) -> str`
- `file_utils/file_dependencies.py::FileDependencies.walk(cls, root: str) -> Iterator[Tuple[str, List[str], List[str]]]`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.apply_glb_articulation(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]`
- `geo_utils/articulation/analysis.py::ArticulationAnalysis(class)`
- `geo_utils/articulation/analysis.py::ArticulationAnalysis.propose(cls, parts: Sequence[Dict[str, Any]], root: Optional[int] = None, order: Optional[Sequence[int]] = None, up: Sequence[float] = (0.0, 1.0, 0.0)) -> Dict[str, Any]`
- `geo_utils/articulation/analysis.py::ArticulationAnalysis.shell(points: Sequence[Sequence[float]]) -> Optional[ShellFacts]`
- `geo_utils/articulation/analysis.py::ShellFacts(class)`
- `geo_utils/articulation/analysis.py::ShellFacts.isotropy(self) -> float`
- `geo_utils/articulation/analysis.py::ShellFacts.length(self) -> float`
- `geo_utils/articulation/conformance.py::ArticulationConformance(class)`
- `geo_utils/articulation/conformance.py::ArticulationConformance.cases(cls, seed: int = 0, per_rig: int = 4) -> Dict[str, Any]`
- `geo_utils/articulation/conformance.py::ArticulationConformance.rigs() -> Dict[str, Dict[str, Any]]`
- `geo_utils/articulation/model.py::ArticulationModel(class)`
- `geo_utils/articulation/model.py::ArticulationModel.chain(self, joint: int) -> List[int]`
- `geo_utils/articulation/model.py::ArticulationModel.clamp(self, state: Sequence[float]) -> List[float]`
- `geo_utils/articulation/model.py::ArticulationModel.from_record(cls, payload: Mapping[str, Any], name: Optional[str] = None) -> 'ArticulationModel'`
- `geo_utils/articulation/model.py::ArticulationModel.joint_index(self, name: str) -> int`
- `geo_utils/articulation/model.py::ArticulationModel.limits(self, slot: int) -> Tuple[Optional[float], Optional[float]]`
- `geo_utils/articulation/model.py::ArticulationModel.local(self, state: Sequence[float], joint: int) -> Tuple[Vec, Quat]`
- `geo_utils/articulation/model.py::ArticulationModel.point(self, state: Sequence[float], joint: int, local_point: Sequence[float]) -> Vec`
- `geo_utils/articulation/model.py::ArticulationModel.pose(self, state: Sequence[float]) -> List[Tuple[Vec, Quat]]`
- `geo_utils/articulation/model.py::ArticulationModel.read(self, locals_: Sequence[Tuple[Sequence[float], Sequence[float]]], hint: Optional[Sequence[float]] = None) -> List[float]`
- `geo_utils/articulation/model.py::ArticulationModel.rest_state(self) -> List[float]`
- `geo_utils/articulation/model.py::ArticulationModel.scale_of(self, locals_: Sequence[Tuple[Sequence[float], Sequence[float]]]) -> float`
- `geo_utils/articulation/model.py::ArticulationModel.solve(self, state: Sequence[float], joint: int, local_point: Sequence[float], target: Sequence[float], rotation: Optional[Sequence[float]] = None) -> List[float]`
- `geo_utils/articulation/model.py::ArticulationModel.to_local(self, state: Sequence[float], joint: int, point: Sequence[float]) -> Vec`
- `geo_utils/articulation/model.py::ArticulationModel.world(self, state: Sequence[float]) -> List[Tuple[Vec, Quat]]`
- `geo_utils/articulation/model.py::CHANNELS(constant)`
- `geo_utils/articulation/model.py::ROTATE_CHANNELS(constant)`
- `geo_utils/articulation/model.py::ROTATE_ORDERS(constant)`
- `geo_utils/articulation/model.py::TRANSLATE_CHANNELS(constant)`

## Signature changed (1)

- `core_utils/presets/library.py::PresetLibrary.backup`
  - was: `(self, path: Optional[Union[str, os.PathLike]] = None, *, reason: str = 'manual') -> Optional[Path]`
  - now: `(self, path: Optional[Union[str, os.PathLike]] = None, *, reason: str = 'manual', keys: Optional[Iterable[str]] = None, name: Optional[str] = None) -> Optional[Path]`
