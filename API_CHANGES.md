# pythontk — API Changes

_Diff vs the last release (origin/main @ 7f9d41f)._

## Added (10)

- `audio_utils/_audio_utils.py::AudioUtils.memo_duration(cls, file_path: str, fps: float, probe: Callable[[str, float], Tuple[float, str]]) -> Tuple[float, str]`
- `core_utils/engines/shots/manifest/manifest_engine.py::ShotManifest.insert_planned(self, plans: List[PlannedInsert], typed: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None) -> Dict[str, str]`
- `core_utils/engines/shots/manifest/manifest_engine.py::ShotManifest.plan_insertions(self, steps: List[BuilderStep], typed: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None, fit_mode: FitMode = DEFAULT_FIT_MODE, initial_shot_length: float = DEFAULT_INITIAL_SHOT_LENGTH) -> List[PlannedInsert]`
- `core_utils/engines/shots/manifest/manifest_engine.py::ShotManifest.ride_inserts(ranges: Dict[str, Tuple[Optional[float], Optional[float]]], plans: List[PlannedInsert]) -> Dict[str, Tuple[Optional[float], Optional[float]]]`
- `core_utils/engines/shots/manifest/manifest_model.py::PlannedInsert(class)`
- `core_utils/engines/shots/manifest/manifest_model.py::PlannedInsert.after_shot_id(self) -> Optional[int]`
- `core_utils/engines/shots/manifest/manifest_model.py::PlannedInsert.after_step_id(self) -> Optional[str]`
- `core_utils/engines/shots/manifest/mapping/_mapping.py::Mapping.option_sections(mapping: Optional[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]`
- `core_utils/engines/shots/shot_model.py::SHOT_STATUS_LEVELS(constant)`
- `core_utils/engines/shots/shot_sequencer.py::ShotSequencer.insert_placement(self, index: int, duration: float, gap: Optional[float] = None, line: Optional[List[tuple]] = None, start: Optional[float] = None) -> Tuple[float, float, float, Tuple[int, ...]]`

## Deprecations (3)

_Live retirement debt, earliest deadline first. An **EXPIRED** row has outlived its window: delete the alias and its tests rather than moving the date. A **HELD** row is due by version, but its notice has not yet had its calendar window._

- **HELD** `core_utils/engines/shots/manifest/manifest_model.py::ManifestModel.asset_column` — remove in 0.14.0, not before 2026-11-09
- **HELD** `core_utils/engines/shots/manifest/manifest_model.py::ManifestModel.column_clipboard` — remove in 0.14.0, not before 2026-11-09
- **HELD** `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.confirm_check_override` — remove in 0.14.0, not before 2026-11-03

## Signature changed (2)

- `core_utils/engines/shots/manifest/manifest_edits.py::ManifestEdits.hidden_objects`
  - was: `(self, step_id: str) -> List[Tuple[str, str]]`
  - now: `(self, step_id: Optional[str] = None) -> List[Tuple[str, str]]`
- `core_utils/engines/shots/shot_sequencer.py::ShotSequencer.insert_shot`
  - was: `(self, name: str, duration: float, after_shot_id: Optional[int] = None, at_position: Optional[int] = None, gap: Optional[float] = None, objects: Optional[List[str]] = None, description: str = '') -> ShotBlock`
  - now: `(self, name: str, duration: float, after_shot_id: Optional[int] = None, at_position: Optional[int] = None, gap: Optional[float] = None, objects: Optional[List[str]] = None, description: str = '', start: Optional[float] = None) -> ShotBlock`
