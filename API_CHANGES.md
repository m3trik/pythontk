# pythontk — API Changes

_Diff vs the last release (origin/main @ de61369)._

## Removed (15)

- `core_utils/engines/textures/map_factory/conversions.py::ConversionRegistry.register_from_class` — was `(self, cls)`
- `core_utils/git.py::Git` — was `(class)`
- `core_utils/git.py::Git.checkout` — was `(self, branch: str)`
- `core_utils/git.py::Git.current_branch` — was `(self) -> str`
- `core_utils/git.py::Git.execute` — was `(self, cmd: Union[str, List[str]], desc: str = None, check: bool = True) -> Optional[str]`
- `core_utils/git.py::Git.fetch` — was `(self, remote: str = 'origin')`
- `core_utils/git.py::Git.merge` — was `(self, source_branch: str)`
- `core_utils/git.py::Git.pull` — was `(self, remote: str = 'origin', branch: str = None)`
- `core_utils/git.py::Git.push` — was `(self, remote: str = 'origin', branch: str = None)`
- `core_utils/git.py::Git.run` — was `(self, cmd: Union[str, List[str]], desc: str = None, check: bool = True) -> Optional[str]`
- `core_utils/git.py::Git.status` — was `(self) -> str`
- `file_utils/_file_utils.py::FileUtils.get_json` — was `(cls, key, file=None)`
- `file_utils/_file_utils.py::FileUtils.get_json_file` — was `(cls)`
- `file_utils/_file_utils.py::FileUtils.set_json` — was `(cls, key, value, file=None)`
- `file_utils/_file_utils.py::FileUtils.set_json_file` — was `(cls, file)`

## Added (58)

- `core_utils/engines/shots/shot_model.py::ShotStore.default_name(self, wanted: Optional[str] = None) -> str`
- `core_utils/engines/shots/shot_model.py::ShotStore.export_records(self, ctx: Optional[ExportContext] = None, strategy: Optional[str] = None) -> Optional[List[Record]]`
- `core_utils/engines/shots/shot_model.py::ShotStore.name_error(self, name: Any, shot_id: Optional[int] = None) -> Optional[str]`
- `core_utils/engines/shots/shot_model.py::ShotStore.produce_export_records(cls, ctx: Optional[ExportContext] = None) -> Optional[List[Record]]`
- `core_utils/engines/shots/shot_model.py::ShotStore.unique_among(name: str, taken: Iterable[str], first: Optional[int] = None) -> str`
- `core_utils/engines/shots/shot_model.py::ShotStore.unique_name(self, base: str = 'Shot', first: Optional[int] = None) -> str`
- `core_utils/scene_records.py::ExportContext(class)`
- `core_utils/scene_records.py::ExportContext.record(self, spec: Union[RecordSpec, str], store=None, default: Any = None) -> Any`
- `core_utils/scene_records.py::ExportContext.refreshes(self, spec: RecordSpec) -> bool`
- `core_utils/scene_records.py::ExportSnapshot(class)`
- `core_utils/scene_records.py::ExportSnapshot.assemble(cls, producers: Mapping[Union[RecordSpec, str], Producer], ctx: Optional[ExportContext] = None, only: Optional[Iterable[Union[RecordSpec, str]]] = None) -> 'ExportSnapshot'`
- `core_utils/scene_records.py::ExportSnapshot.channels(self, scope: Scope = Scope.DELIVERABLE) -> Dict[str, Any]`
- `core_utils/scene_records.py::ExportSnapshot.commit(self, store) -> Dict[str, Optional[str]]`
- `core_utils/scene_records.py::ExportSnapshot.publish(cls, store, records: Mapping[Union[RecordSpec, str], Any], ctx: Optional[ExportContext] = None) -> 'ExportSnapshot'`
- `core_utils/scene_records.py::ExportSnapshot.record(self, spec: Union[RecordSpec, str], default: Any = None) -> Any`
- `core_utils/scene_records.py::ExportSnapshot.records(self) -> Dict[str, Record]`
- `core_utils/scene_records.py::ExportSnapshot.summary(self) -> str`
- `core_utils/scene_records.py::Kind(class)`
- `core_utils/scene_records.py::Record(class)`
- `core_utils/scene_records.py::Record.key(self) -> str`
- `core_utils/scene_records.py::Record.save(self, store) -> Optional[str]`
- `core_utils/scene_records.py::Record.text(self) -> str`
- `core_utils/scene_records.py::RecordSpec(class)`
- `core_utils/scene_records.py::RecordSpec.clear(self, store) -> Optional[str]`
- `core_utils/scene_records.py::RecordSpec.decode(self, text: Optional[str], default: Any = None) -> Any`
- `core_utils/scene_records.py::RecordSpec.encode(self, payload: Any) -> str`
- `core_utils/scene_records.py::RecordSpec.is_present(self, store) -> bool`
- `core_utils/scene_records.py::RecordSpec.load(self, store, default: Any = None) -> Any`
- `core_utils/scene_records.py::RecordSpec.make(self, payload: Any) -> Record`
- `core_utils/scene_records.py::RecordSpec.read_text(self, store) -> Optional[str]`
- `core_utils/scene_records.py::RecordSpec.save(self, store, payload: Any) -> Optional[str]`
- `core_utils/scene_records.py::RecordSpec.write_text(self, store, text: Optional[str]) -> Optional[str]`
- `core_utils/scene_records.py::SceneRecords(class)`
- `core_utils/scene_records.py::SceneRecords.all(cls) -> List[RecordSpec]`
- `core_utils/scene_records.py::SceneRecords.by_key(cls, key: str, scope: Optional[Scope] = None) -> Optional[RecordSpec]`
- `core_utils/scene_records.py::SceneRecords.check_producers(cls, table: Mapping[Any, Any]) -> List[RecordSpec]`
- `core_utils/scene_records.py::SceneRecords.declared_takes(cls, read: Callable[[str], Any]) -> List[Dict[str, Any]]`
- `core_utils/scene_records.py::SceneRecords.deliverable(cls) -> List[RecordSpec]`
- `core_utils/scene_records.py::SceneRecords.describe(cls) -> List[Dict[str, Any]]`
- `core_utils/scene_records.py::SceneRecords.handoff_block(cls, channels: Union[Iterable[str], Mapping[str, Any]], source: Optional[Mapping[str, str]] = None) -> Dict[str, Any]`
- `core_utils/scene_records.py::SceneRecords.ordered(cls, specs: Iterable[RecordSpec]) -> List[RecordSpec]`
- `core_utils/scene_records.py::SceneRecords.private(cls) -> List[RecordSpec]`
- `core_utils/scene_records.py::SceneRecords.rendering_policy() -> Dict[str, Any]`
- `core_utils/scene_records.py::SceneRecords.resolve(cls, item: Union[RecordSpec, str]) -> RecordSpec`
- `core_utils/scene_records.py::SceneStoreBase(class)`
- `core_utils/scene_records.py::SceneStoreBase.channels(cls, scope: Scope) -> Dict[str, str]`
- `core_utils/scene_records.py::SceneStoreBase.dump(cls, decode: bool = True) -> Dict[str, Dict[str, Any]]`
- `core_utils/scene_records.py::SceneStoreBase.format_dump(cls, decode: bool = True) -> str`
- `core_utils/scene_records.py::SceneStoreBase.keys(cls, scope: Scope) -> List[str]`
- `core_utils/scene_records.py::SceneStoreBase.name(cls, scope: Scope) -> str`
- `core_utils/scene_records.py::SceneStoreBase.read(cls, scope: Scope, key: str) -> Optional[str]`
- `core_utils/scene_records.py::SceneStoreBase.values(cls, scope: Scope) -> Dict[str, Any]`
- `core_utils/scene_records.py::SceneStoreBase.write(cls, scope: Scope, key: str, text: Optional[str]) -> Optional[str]`
- `core_utils/scene_records.py::Scope(class)`
- `str_utils/_str_utils.py::StrUtils.illegal_name_chars(name: str) -> List[str]`
- `str_utils/_str_utils.py::StrUtils.is_legal_name(name) -> bool`
- `str_utils/_str_utils.py::StrUtils.legal_name_matcher(legal_name: str) -> 're.Pattern'`
- `str_utils/_str_utils.py::StrUtils.name_error(cls, name, subject: str = 'names', reason: str = '') -> Optional[str]`

## Signature changed (1)

- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.build_fbx_handoff`
  - was: `(cls, channels: Iterable[str], source: Optional[Dict[str, str]] = None) -> Dict[str, Any]`
  - now: `(cls, channels: Union[Iterable[str], Mapping[str, Any]], source: Optional[Dict[str, str]] = None) -> Dict[str, Any]`
