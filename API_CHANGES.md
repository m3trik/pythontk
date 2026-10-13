# pythontk — API Changes

_Diff vs the last release (origin/main @ 8a51fb8)._

## Added (47)

- `audio_utils/_audio_utils.py::AudioUtils.cache_dir_for(cls, audio_path: str, create: bool = False) -> str`
- `core_utils/user_config.py::UserConfig.cache_dir(*parts: str, create: bool = False) -> Path`
- `file_utils/bake_cache.py::BakeCache(class)`
- `file_utils/bake_cache.py::BakeCache.cache_dir(self, project: str, create: bool = False) -> str`
- `file_utils/bake_cache.py::BakeCache.current(self, src: str, project: str, key: str) -> Optional[str]`
- `file_utils/bake_cache.py::BakeCache.is_bake(self, path: str) -> bool`
- `file_utils/bake_cache.py::BakeCache.name_pattern(self) -> Pattern[str]`
- `file_utils/bake_cache.py::BakeCache.prefix(self) -> str`
- `file_utils/bake_cache.py::BakeCache.purge(self, project: str, extensions: Iterable[str], live: Iterable[str] = (), dry_run: bool = False) -> List[str]`
- `file_utils/bake_cache.py::BakeCache.record(self, src: str, project: str, key: Optional[str] = None) -> Dict[str, str]`
- `file_utils/bake_cache.py::BakeCache.referenced_keys(self, project: str, extensions: Iterable[str], live: Iterable[str] = ()) -> Optional[Set[str]]`
- `file_utils/bake_cache.py::BakeCache.rehome(self, baked_path: str, project: str, produce: Callable[[str, str, Dict[str, str]], Any]) -> Optional[str]`
- `file_utils/bake_cache.py::BakeCache.slot(self, src: str, project: str) -> str`
- `file_utils/bake_cache.py::BakeCache.slot_key(src: str, project: str) -> str`
- `file_utils/bake_cache.py::BakeCache.source_key(src: str, *parts: Any, companions: Iterable[str] = ()) -> str`
- `file_utils/bake_cache.py::BakeCache.source_of(self, baked_path: str) -> Optional[str]`
- `file_utils/bake_cache.py::BakeCache.spell_source(self, src: str, project: str) -> str`
- `file_utils/bake_cache.py::BakeCache.store(self, project: str, create: bool = False) -> CachedArtifact`
- `file_utils/bake_cache.py::BakeCache.write(self, src: str, project: str, produce: Callable[[str, Dict[str, str]], Any], key: Optional[str] = None) -> str`
- `file_utils/file_dependencies.py::FileDependencies.open_decompressed(path: str) -> Optional[BinaryIO]`
- `file_utils/file_dependencies.py::FileDependencies.reference_bytes(cls, path: str) -> Optional[bytes]`
- `file_utils/file_dependencies.py::FileDependencies.scan_references(cls, root: str, pattern: Union[bytes, Pattern[bytes]], extensions: Iterable[str], skip: Iterable[str] = ()) -> Optional[Set[str]]`
- `file_utils/scene_notes.py::SceneNotes(class)`
- `file_utils/scene_notes.py::SceneNotes.get(self, scene: str) -> str`
- `file_utils/scene_notes.py::SceneNotes.import_notes(self, notes: Mapping[str, str]) -> Dict[str, str]`
- `file_utils/scene_notes.py::SceneNotes.move(self, old: str, new: str) -> None`
- `file_utils/scene_notes.py::SceneNotes.remove(self, scene: str) -> None`
- `file_utils/scene_notes.py::SceneNotes.set(self, scene: str, text: str) -> None`
- `file_utils/scene_notes.py::SceneNotes.store_path(self, scene: str) -> Optional[str]`
- `file_utils/temp_artifacts.py::CachedArtifact.cache_dir(self) -> str`
- `file_utils/temp_artifacts.py::CachedArtifact.peek(self, key: str) -> Optional[str]`
- `file_utils/temp_artifacts.py::CachedArtifact.promote(self, key: str, produced: str, *, sidecars: Sequence[str] = ()) -> str`
- `file_utils/temp_artifacts.py::CachedArtifact.slot(self, key: str) -> str`
- `file_utils/temp_artifacts.py::CachedArtifact.touch(self, key: str, sidecars: Sequence[str] = ()) -> bool`
- `file_utils/temp_artifacts.py::TempArtifacts.contains(cls, path: str) -> bool`
- `file_utils/temp_artifacts.py::TempArtifacts.keep(self, path: str) -> bool`
- `file_utils/temp_artifacts.py::TempArtifacts.root(cls) -> str`
- `file_utils/temp_artifacts.py::TempArtifacts.scoped(cls, prefix: str, **kwargs: Any) -> 'TempArtifacts'`
- `file_utils/temp_artifacts.py::TempArtifacts.sweep_all(cls, root: Optional[str] = None) -> List[str]`
- `file_utils/workspace.py::Workspace.absolute(self, stored: str) -> str`
- `file_utils/workspace.py::Workspace.backup_dir(scene_path: str, create: bool = False) -> str`
- `file_utils/workspace.py::Workspace.cache_dir(self, kind: str, create: bool = False) -> str`
- `file_utils/workspace.py::Workspace.export_dir(self, create: bool = False) -> str`
- `file_utils/workspace.py::Workspace.for_scene(cls, scene_path: Optional[str], session_root: Optional[str] = None) -> Optional['Workspace']`
- `file_utils/workspace.py::Workspace.images_dir(self, create: bool = False) -> str`
- `file_utils/workspace.py::Workspace.movie_dir(self, create: bool = False) -> str`
- `file_utils/workspace.py::Workspace.relative(self, path: str) -> str`

## Deprecations (3)

_Live retirement debt, earliest deadline first. An **EXPIRED** row has outlived its window: delete the alias and its tests rather than moving the date. A **HELD** row is due by version, but its notice has not yet had its calendar window._

- **HELD** `core_utils/engines/shots/manifest/manifest_model.py::ManifestModel.asset_column` — remove in 0.14.0, not before 2026-11-09
- **HELD** `core_utils/engines/shots/manifest/manifest_model.py::ManifestModel.column_clipboard` — remove in 0.14.0, not before 2026-11-09
- **HELD** `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.confirm_check_override` — remove in 0.14.0, not before 2026-11-03

## Signature changed (2)

- `core_utils/engines/textures/region_masks.py::RegionMaskPacker.write`
  - was: `(self, mask_path: str, manifest_path: Optional[str] = None, uv_channel: int = 0) -> RegionMaskManifest`
  - now: `(self, mask_path: str, manifest_path: Optional[str] = None, uv_channel: int = 0, write_manifest: bool = False) -> RegionMaskManifest`
- `file_utils/file_dependencies.py::FileDependencies.search_dirs`
  - was: `(deps: Iterable[Dict[str, Any]], then: Iterable[str] = ()) -> List[str]`
  - now: `(cls, deps: Iterable[Dict[str, Any]], then: Iterable[str] = ()) -> List[str]`
