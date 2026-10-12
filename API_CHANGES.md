# pythontk — API Changes

_Diff vs the last release (origin/main @ a1d084a)._

## Added (56)

- `core_utils/engines/shots/manifest/behaviors/_behaviors.py::Behaviors.anchored(block: Dict[str, Any], phase: str, anchor_override: Any = None) -> Dict[str, Any]`
- `core_utils/engines/shots/manifest/behaviors/_behaviors.py::Behaviors.entry_anchors(entries: List[Dict[str, Any]]) -> List[Optional[float]]`
- `core_utils/engines/shots/manifest/behaviors/_behaviors.py::Behaviors.is_audio_entry(entry: Dict[str, Any]) -> bool`
- `core_utils/engines/shots/manifest/behaviors/_behaviors.py::Behaviors.key_window(behavior: Any, start: float, end: float, anchor_override: Any = None, recipe: Optional[EffectRecipe] = None) -> Optional[Tuple[float, float]]`
- `core_utils/engines/shots/manifest/behaviors/_behaviors.py::Behaviors.label(name: str) -> str`
- `core_utils/engines/shots/manifest/manifest_edits.py::ManifestEdits.adopt_layers(cls, store: Any, source: str) -> bool`
- `core_utils/engines/shots/manifest/manifest_edits.py::ManifestEdits.layer_key(cls, store: Any, source: str) -> Optional[str]`
- `core_utils/engines/shots/manifest/manifest_edits.py::ManifestEdits.read(cls, store: Any, source: str) -> 'ManifestEdits'`
- `core_utils/engines/shots/manifest/manifest_edits.py::ManifestEdits.same_sheet(cls, a: str, b: str) -> bool`
- `core_utils/engines/shots/manifest/manifest_edits.py::ManifestEdits.set_sheet_options(cls, store: Any, source: str, template: Optional[str], values: Optional[Dict[str, Any]]) -> bool`
- `core_utils/engines/shots/manifest/manifest_edits.py::ManifestEdits.sheet_options(cls, store: Any, source: str) -> Optional[Dict[str, Any]]`
- `core_utils/engines/shots/manifest/manifest_edits.py::ManifestEdits.write(self, store: Any, source: str) -> bool`
- `core_utils/engines/shots/manifest/manifest_engine.py::ShotManifest.place_markers(self) -> List[Dict[str, Any]]`
- `core_utils/engines/shots/manifest/manifest_engine.py::ShotManifest.planned_markers(self, shots: Optional[list] = None) -> List[Dict[str, Any]]`
- `core_utils/engines/shots/manifest/manifest_engine.py::ShotManifest.unkeyed_behaviors(self) -> frozenset`
- `core_utils/engines/shots/manifest/mapping/_mapping.py::Mapping.infer_options(csv_path: str, mapping: Dict[str, Any], built: Dict[str, Iterable[str]], values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`
- `core_utils/engines/shots/shot_ledger.py::ShotEditLedger.key_stays(self, curve: str, time: float) -> bool`
- `core_utils/engines/shots/shot_ledger.py::StepClaim(class)`
- `core_utils/engines/shots/shot_model.py::MarkersChanged(class)`
- `core_utils/engines/shots/shot_model.py::NativeStepFollowed(class)`
- `core_utils/engines/shots/shot_model.py::ShotStore.add_native_step_listener(cls, callback: Callable[['NativeStepFollowed'], None]) -> None`
- `core_utils/engines/shots/shot_model.py::ShotStore.add_system_members(self, shot_id: int, names: Iterable[str]) -> List[str]`
- `core_utils/engines/shots/shot_model.py::ShotStore.follow_native_step(self, redo: bool = False) -> int`
- `core_utils/engines/shots/shot_model.py::ShotStore.following_native_step(self) -> bool`
- `core_utils/engines/shots/shot_model.py::ShotStore.in_transient_edit() -> bool`
- `core_utils/engines/shots/shot_model.py::ShotStore.marker_owner(marker: Dict[str, Any]) -> str`
- `core_utils/engines/shots/shot_model.py::ShotStore.native_undo_sync_held(cls)`
- `core_utils/engines/shots/shot_model.py::ShotStore.remove_native_step_listener(cls, callback: Callable[['NativeStepFollowed'], None]) -> None`
- `core_utils/engines/shots/shot_model.py::ShotStore.set_markers(self, markers: Iterable[Dict[str, Any]]) -> bool`
- `core_utils/engines/shots/shot_model.py::ShotStore.step_links(self) -> Dict[int, str]`
- `core_utils/engines/shots/shot_model.py::ShotStore.system_members(self, shot: ShotBlock) -> set`
- `core_utils/engines/shots/shot_model.py::ShotStore.transient_edit()`
- `core_utils/engines/shots/shot_plan.py::ShotPlanner.closing_seams(store: ShotStore, bounds: Dict[int, Tuple[float, float]], eps: float = 0.001) -> List[Tuple[int, int]]`
- `core_utils/engines/shots/shot_plan.py::ShotPlanner.keyless_seam(times: Iterable[float], boundary: float, eps: float = 0.001) -> bool`
- `core_utils/engines/shots/shot_sequencer.py::ShotSequencer.paste_pairs(rows: Iterable[Tuple[Any, str, Any]], destinations: Iterable[Tuple[Any, str]], leaf: Optional[Callable[[Any], Any]] = None) -> List[Tuple[Any, str, Any]]`
- `core_utils/engines/shots/shot_sequencer.py::ShotSequencer.pending_bounds(self, bounds: Dict[int, Tuple[Optional[float], Optional[float]]])`
- `file_utils/mesh_convert/fbx_file.py::FbxFile.take_layout(self, tolerance: float = 0.0001) -> Dict[str, Any]`
- `file_utils/mesh_convert/fbx_media.py::FbxMedia.merge_takes(cls, src: str, dst: Optional[str] = None, *, takes: Optional[Iterable[str]] = None, name: Optional[str] = MERGED_TAKE, tolerance: float = 0.0001, keep: Iterable[str] = ()) -> Dict[str, Any]`
- `math_utils/key_curve.py::CurveKey(class)`
- `math_utils/key_curve.py::INFINITY_MODES(constant)`
- `math_utils/key_curve.py::INTERPOLATIONS(constant)`
- `math_utils/key_curve.py::KeyCurve(class)`
- `math_utils/key_curve.py::KeyCurve.bezier_parameter(x: float, x0: float, x1: float, x2: float, x3: float) -> float`
- `math_utils/key_curve.py::KeyCurve.control_points(self, i: int) -> Tuple[Vec, Vec, Vec, Vec]`
- `math_utils/key_curve.py::KeyCurve.copy(self) -> 'KeyCurve'`
- `math_utils/key_curve.py::KeyCurve.evaluate(self, t: float) -> float`
- `math_utils/key_curve.py::KeyCurve.index_at(self, t: float, eps: float = TIME_EPS) -> int`
- `math_utils/key_curve.py::KeyCurve.normalization(self, t0: Optional[float] = None, t1: Optional[float] = None) -> Tuple[float, float]`
- `math_utils/key_curve.py::KeyCurve.segments(self, lo: Optional[int] = None, hi: Optional[int] = None) -> List[dict]`
- `math_utils/key_curve.py::KeyCurve.times(self) -> List[float]`
- `math_utils/key_curve.py::KeyCurve.to_preview(self, t_start: float, t_end: float, hidden_times: Iterable[float] = ()) -> Optional[dict]`
- `math_utils/key_curve.py::KeyCurve.value_bounds(self, t0: Optional[float] = None, t1: Optional[float] = None, handles: bool = False) -> Tuple[float, float]`
- `math_utils/key_curve.py::KeyCurve.with_handle(self, index: int, side: str, vector: Vec, unify: Optional[bool] = None) -> 'KeyCurve'`
- `math_utils/key_curve.py::KeyCurve.with_infinity(self, pre: Optional[str] = None, post: Optional[str] = None) -> 'KeyCurve'`
- `math_utils/key_curve.py::KeyCurve.with_keys_moved(self, indices: Iterable[int], dt: float = 0.0, dv: float = 0.0) -> 'KeyCurve'`
- `math_utils/key_curve.py::TIME_EPS(constant)`

## Deprecations (3)

_Live retirement debt, earliest deadline first. An **EXPIRED** row has outlived its window: delete the alias and its tests rather than moving the date. A **HELD** row is due by version, but its notice has not yet had its calendar window._

- **HELD** `core_utils/engines/shots/manifest/manifest_model.py::ManifestModel.asset_column` — remove in 0.14.0, not before 2026-11-09
- **HELD** `core_utils/engines/shots/manifest/manifest_model.py::ManifestModel.column_clipboard` — remove in 0.14.0, not before 2026-11-09
- **HELD** `core_utils/engines/scene_export/scene_exporter.py::SceneExporterBase.confirm_check_override` — remove in 0.14.0, not before 2026-11-03

## Signature changed (7)

- `core_utils/engines/shots/shot_ledger.py::ShotEditLedger.claimed_step`
  - was: `(self, curve: str, time: float) -> Optional[Tuple[str, str, Optional[float]]]`
  - now: `(self, curve: str, time: float) -> Optional[StepClaim]`
- `core_utils/engines/shots/shot_ledger.py::ShotEditLedger.is_same_key`
  - was: `(claim: Tuple[str, str, Optional[float]], in_type: str, value: float, tol: float = 1e-06) -> bool`
  - now: `(claim: Tuple[str, str, Optional[float]], in_type: str, value: float, tol: float = 1e-06, in_half: Optional[Iterable[float]] = None, half_tol: float = 0.0001) -> bool`
- `core_utils/engines/shots/shot_ledger.py::ShotEditLedger.record_key`
  - was: `(self, curve: str, time: float, owner: int = NO_OWNER, edge: str = '') -> bool`
  - now: `(self, curve: str, time: float, owner: int = NO_OWNER, edge: str = '', stays: bool = False) -> bool`
- `core_utils/engines/shots/shot_ledger.py::ShotEditLedger.record_step`
  - was: `(self, curve: str, time: float, in_type: str, out_type: str, value: Optional[float] = None) -> bool`
  - now: `(self, curve: str, time: float, in_type: str, out_type: str, value: Optional[float] = None, in_half: Optional[Iterable[float]] = None, out_half: Optional[Iterable[float]] = None) -> bool`
- `core_utils/engines/shots/shot_model.py::ShotStore.discard_boundary_snapshot`
  - was: `(self) -> None`
  - now: `(self) -> bool`
- `core_utils/engines/shots/shot_plan.py::ShotPlanner.boundary_splits`
  - was: `(store: ShotStore, plan: MovePlan, eps: float = _EPS) -> List[Tuple[int, int, float, float]]`
  - now: `(store: ShotStore, plan: MovePlan, eps: float = _EPS, pending: Optional[Dict[int, Tuple[Optional[float], Optional[float]]]] = None) -> List[Tuple[int, int, float, float]]`
- `core_utils/engines/shots/shot_plan.py::ShotPlanner.plan_respace`
  - was: `(store: ShotStore, gap: float, start_frame: float, respect_locks: bool = True) -> MovePlan`
  - now: `(store: ShotStore, gap: float, start_frame: float, respect_locks: bool = True, seam_gaps: Optional[Dict[int, float]] = None) -> MovePlan`
