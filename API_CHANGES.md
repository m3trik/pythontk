# pythontk — API Changes

_Diff vs the last release (origin/main @ a03683a)._

## Removed (21)

- `geo_utils/shadow_horizon.py::HorizonMap` — was `(class)`
- `geo_utils/shadow_horizon.py::HorizonMap.alpha` — was `(self, points, light=None, *, direction=None, source_size: float = 0.0, source_angle: float = 0.0, intensity: float = 1.0) -> np.ndarray`
- `geo_utils/shadow_horizon.py::HorizonMap.decode_cot` — was `(self, value: np.ndarray) -> np.ndarray`
- `geo_utils/shadow_horizon.py::HorizonMap.encode_angle` — was `(self, angle: np.ndarray) -> np.ndarray`
- `geo_utils/shadow_horizon.py::HorizonMap.from_rgba` — was `(cls, pixels: np.ndarray, *, bins: int, size: Sequence[int], r_min: float, r_max: float, ground: float = 0.0, up: int = 1, max_stretch: Optional[float] = None) -> 'HorizonMap'`
- `geo_utils/shadow_horizon.py::HorizonMap.layers` — was `(self) -> int`
- `geo_utils/shadow_horizon.py::HorizonMap.layout` — was `(self) -> Tuple[int, int]`
- `geo_utils/shadow_horizon.py::HorizonMap.mask_bits` — was `(values: np.ndarray) -> np.ndarray`
- `geo_utils/shadow_horizon.py::HorizonMap.taps` — was `(self, layer: int, k, u, v) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
- `geo_utils/shadow_horizon.py::HorizonMap.texel_positions` — was `(self) -> np.ndarray`
- `geo_utils/shadow_horizon.py::HorizonMap.tile_index` — was `(self, layer: int, k: int) -> int`
- `geo_utils/shadow_horizon.py::HorizonMap.tile_rects` — was `(self) -> List[Tuple[float, float, float, float]]`
- `geo_utils/shadow_horizon.py::HorizonMap.tiles` — was `(self) -> int`
- `geo_utils/shadow_horizon.py::HorizonMap.to_rgba` — was `(self) -> np.ndarray`
- `geo_utils/shadow_horizon.py::HorizonMap.uv` — was `(self, horizontal) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
- `geo_utils/shadow_horizon.py::ShadowHorizon.layout` — was `(tiles: int) -> Tuple[int, int]`
- `geo_utils/shadow_horizon.py::ShadowHorizon.range_for` — was `(cls, radius: float, height: float, max_stretch: Optional[float] = None) -> Tuple[float, float]`
- `net_utils/preview/deliverer.py::PreviewPassContext` — was `(class)`
- `net_utils/preview/deliverer.py::PreviewPassContext.lightmap_search_dirs` — was `(self) -> Sequence[str]`
- `net_utils/preview/deliverer.py::PreviewPassContext.logger` — was `(self)`
- `net_utils/preview/deliverer.py::PreviewPassContext.sidecar` — was `(self) -> Optional[Dict[str, Any]]`

## Added (25)

- `core_utils/export_profile.py::ExportProfile(class)`
- `core_utils/export_profile.py::ExportProfile.legal_name(name: str) -> str`
- `core_utils/export_profile.py::ExportProfile.read_values(cls, widgets: Mapping[str, Any], *tables: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]`
- `core_utils/export_profile.py::ExportProfile.run_config(cls, values: Mapping[str, Any], task_definitions: Mapping[str, Mapping[str, Any]], check_definitions: Mapping[str, Mapping[str, Any]], override_checks: bool = False, ignore_groups_case_sensitive: bool = False, default_export_mode: str = 'visible') -> Dict[str, Any]`
- `core_utils/export_profile.py::ExportProfile.value_method(cls, spec: Mapping[str, Any]) -> str`
- `core_utils/export_profile.py::ExportProfile.widget_key(cls, name: str, spec: Mapping[str, Any]) -> str`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.lightmap_report(coverage: Dict[str, List[str]], bound: Sequence[Dict[str, Any]]) -> Dict[str, Any]`
- `file_utils/mesh_convert/glb_pipeline.py::GlbPipeline(class)`
- `file_utils/mesh_convert/glb_pipeline.py::GlbPipeline.build(cls, src: str, dst: Optional[str] = None, *, sidecar: Optional[Dict[str, Any]] = None, lightmap_dirs: Sequence[str] = (), texture_params: Optional[Dict[str, Any]] = None, downsize: bool = True, scratch_path: Optional[Callable[[str], str]] = None, release_source: Optional[Callable[[str], Any]] = None, progress: Optional[Callable[[str], Any]] = None, logger: Any = None) -> Dict[str, Any]`
- `file_utils/mesh_convert/glb_pipeline.py::GlbPipeline.envelope(cls, read_sections: Callable[[], Optional[Dict[str, Any]]], *, source: Dict[str, str], asset: Optional[str] = None, logger: Any = None) -> Dict[str, Any]`
- `geo_utils/shadow_horizon.py::HeightFieldMap(class)`
- `geo_utils/shadow_horizon.py::HeightFieldMap.alpha(self, points, light=None, *, direction=None, source_size: float = 0.0, source_angle: float = 0.0, intensity: float = 1.0) -> np.ndarray`
- `geo_utils/shadow_horizon.py::HeightFieldMap.aspect(self) -> float`
- `geo_utils/shadow_horizon.py::HeightFieldMap.from_rgba(cls, rgba: np.ndarray, *, size: int, spans: int, bounds: Sequence[float], ground: float, up: int, height_scale: float) -> 'HeightFieldMap'`
- `geo_utils/shadow_horizon.py::HeightFieldMap.hull(self) -> Tuple[np.ndarray, np.ndarray]`
- `geo_utils/shadow_horizon.py::HeightFieldMap.levels(self) -> int`
- `geo_utils/shadow_horizon.py::HeightFieldMap.pixel(self) -> Tuple[float, float]`
- `geo_utils/shadow_horizon.py::HeightFieldMap.pyramid(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]`
- `geo_utils/shadow_horizon.py::HeightFieldMap.tiles(self) -> int`
- `geo_utils/shadow_horizon.py::HeightFieldMap.to_rgba(self) -> np.ndarray`
- `geo_utils/shadow_horizon.py::ShadowHorizon.record(cls, *, texture: str, size: int, spans: int, levels: int, bounds: Sequence[float], height_scale: float, frame_a: Sequence[float], frame_b: Sequence[float], rect: Sequence[float]) -> Dict[str, object]`
- `img_utils/_img_utils.py::ImgUtils.rasterize_height_spans(cls, meshes, *, up: int = 1, size: int = 64, ground: float = 0.0, bounds=None, padding: float = 0.02, spans: int = 1)`
- `math_utils/_math_utils.py::MathUtils.evaluate_hermite(times: Sequence[float], values: Sequence[float], keep_indices: Sequence[int], in_slopes: Sequence[float], out_slopes: Sequence[float], at: Optional[Sequence[float]] = None) -> 'np.ndarray'`
- `math_utils/_math_utils.py::MathUtils.reduce_samples(times: Sequence[float], values: Sequence[float], value_tolerance: float = 1e-05, max_error: Optional[float] = None) -> Tuple[List[int], List[float], List[float]]`
- `vid_utils/_vid_utils.py::VidUtils.ensure_ffmpeg(cls, prompt: Union[bool, Callable[[str], bool]] = True) -> Optional[str]`

## Signature changed (8)

- `core_utils/engines/shots/shot_apply.py::ShotApply.apply`
  - was: `(plan: MovePlan, store: ShotStore, move_keys: Optional[MoveKeys] = None, shift_audio: Optional[ShiftAudio] = None, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> None`
  - now: `(plan: MovePlan, store: ShotStore, move_keys: Optional[MoveKeys] = None, shift_audio: Optional[ShiftAudio] = None, progress_callback: Optional[Callable[[int, int, str], None]] = None, objects_for: Optional[Callable[[int], Iterable[str]]] = None) -> None`
- `core_utils/engines/shots/shot_plan.py::ShotPlanner.plan_ripple_downstream`
  - was: `(store: ShotStore, pivot_shot_id: int, after_frame: float, delta: float) -> MovePlan`
  - now: `(store: ShotStore, pivot_shot_id: int, after_frame: float, delta: float, carry_gap: bool = False) -> MovePlan`
- `core_utils/engines/shots/shot_plan.py::ShotPlanner.plan_ripple_upstream`
  - was: `(store: ShotStore, pivot_shot_id: int, before_frame: float, delta: float) -> MovePlan`
  - now: `(store: ShotStore, pivot_shot_id: int, before_frame: float, delta: float, carry_gap: bool = False) -> MovePlan`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.fbx_to_glb`
  - was: `(cls, src: str, dst: Optional[str] = None, *, overwrite: bool = False, auto_install: bool = True, prompt: Union[bool, Callable[[str], bool]] = True, timeout: Optional[float] = AUTO_TIMEOUT, extra_args: Optional[List[str]] = None, sidecar: Optional[Dict[str, Any]] = None, lightmaps: bool = True, lightmap_dirs: Sequence[str] = (), shadow_dirs: Sequence[str] = ()) -> str`
  - now: `(cls, src: str, dst: Optional[str] = None, *, overwrite: bool = False, auto_install: bool = True, prompt: Union[bool, Callable[[str], bool]] = True, timeout: Optional[float] = AUTO_TIMEOUT, extra_args: Optional[List[str]] = None, sidecar: Optional[Dict[str, Any]] = None, lightmaps: bool = True, lightmap_dirs: Sequence[str] = (), shadow_dirs: Sequence[str] = (), report: Optional[Dict[str, Any]] = None) -> str`
- `geo_utils/shadow_horizon.py::ShadowHorizon.bake`
  - was: `(cls, meshes, *, ground: float = 0.0, up: int = 1, radius: Optional[float] = None, height: Optional[float] = None, bins: int = DEFAULT_BINS, size: Sequence[int] = DEFAULT_SIZE, r_min: Optional[float] = None, r_max: Optional[float] = None, max_stretch: Optional[float] = None, footprint: int = DEFAULT_FOOTPRINT, threads: Optional[int] = None) -> HorizonMap`
  - now: `(cls, meshes, *, ground: float = 0.0, up: int = 1, size: int = DEFAULT_SIZE, spans: int = DEFAULT_SPANS, bounds=None, padding: float = DEFAULT_PADDING) -> HeightFieldMap`
- `geo_utils/shadow_horizon.py::ShadowHorizon.bake_adaptive`
  - was: `(cls, meshes, *, threshold: float = 0.05, max_bins: int = 64, measure_samples: int = 6, **kwargs) -> Tuple[HorizonMap, Dict[str, float]]`
  - now: `(cls, meshes, *, threshold: float = 0.05, max_size: int = 256, measure_samples: int = 6, **kwargs) -> Tuple[HeightFieldMap, Dict[str, float]]`
- `geo_utils/shadow_horizon.py::ShadowHorizon.measure`
  - was: `(cls, hmap: HorizonMap, meshes, *, samples: int = 8, size: int = 256, seed: int = 0, max_stretch: Optional[float] = None, radius: Optional[float] = None, height: Optional[float] = None) -> Dict[str, float]`
  - now: `(cls, hmap: HeightFieldMap, meshes, *, samples: int = 8, size: int = 256, seed: int = 0, max_stretch: Optional[float] = None, radius: Optional[float] = None, height: Optional[float] = None) -> Dict[str, float]`
- `vid_utils/_vid_utils.py::VidUtils.resolve_ffmpeg`
  - was: `(cls, required: bool = True, auto_install: bool = False) -> Optional[str]`
  - now: `(cls, required: bool = True, auto_install: bool = False, prompt: Union[bool, Callable[[str], bool]] = False) -> Optional[str]`
