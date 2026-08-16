# pythontk — API Changes

_Diff vs the last release (origin/main @ 30eea1f). Generated 2026-08-15._

## Removed (3)

- `file_utils/mesh_cleaner.py::MeshCleaner` — was `(class)`
- `file_utils/mesh_cleaner.py::MeshCleaner.clean` — was `(self, input_path: str, output_path: Optional[str] = None, merge_distance: float = 1e-05, remove_isolated_pieces_diameter_percent: float = 5.0, fill_holes_max_edge_count: int = 500, decimate_target_faces: int = 0) -> Optional[str]`
- `file_utils/mesh_cleaner.py::MeshCleaner.is_available` — was `(self) -> bool`

## Added (30)

- `core_utils/doc_audit.py::DocAudit(class)`
- `core_utils/doc_audit.py::DocAudit.audit_code(cls, code: str, roots: Optional[Mapping[str, Any]] = None) -> List[str]`
- `core_utils/doc_audit.py::DocAudit.audit_markdown(cls, markdown: str, roots: Optional[Mapping[str, Any]] = None, lang: str = 'python') -> List[str]`
- `core_utils/doc_audit.py::DocAudit.default_roots(cls) -> dict`
- `core_utils/doc_audit.py::DocAudit.extract_code_blocks(cls, markdown: str, lang: str = 'python') -> List[str]`
- `core_utils/engines/textures/map_optimizer.py::MapOptimizer.resolve_compression(cls, map_type_key: Optional[str], output_type: Optional[str], spec: Optional[OutputSpec] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]`
- `file_utils/mesh_ops.py::MeshOps(class)`
- `file_utils/mesh_ops.py::MeshOps.apply(cls, input_path: str, filter_name: str, output_path: Optional[str] = None, **params) -> str`
- `file_utils/mesh_ops.py::MeshOps.available(cls) -> bool`
- `file_utils/mesh_ops.py::MeshOps.bake_vertex_color(cls, input_path: str, output_path: Optional[str] = None, texture_size: int = 1024, border: int = 2) -> Tuple[str, str]`
- `file_utils/mesh_ops.py::MeshOps.clean(cls, input_path: str, output_path: Optional[str] = None, merge_distance: float = 1e-05, remove_isolated_pieces_diameter_percent: float = 5.0, fill_holes_max_edge_count: int = 500, decimate_target_faces: int = 0) -> str`
- `file_utils/mesh_ops.py::MeshOps.compare(cls, input_path: str, reference_path: str, sample_num: int = 10000) -> Dict[str, Any]`
- `file_utils/mesh_ops.py::MeshOps.decimate(cls, input_path: str, output_path: Optional[str] = None, target_faces: int = 0, target_pct: float = 0.0, curvature_weighted: bool = False, preserve_boundary: bool = True, preserve_normal: bool = True, preserve_topology: bool = False, quality_threshold: float = 0.3) -> str`
- `file_utils/mesh_ops.py::MeshOps.measure(cls, input_path: str) -> Dict[str, Any]`
- `file_utils/mesh_ops.py::MeshOps.remesh(cls, input_path: str, output_path: Optional[str] = None, target_edge_pct: float = 1.0, iterations: int = 10, adaptive: bool = False) -> str`
- `file_utils/mesh_ops.py::MeshOps.resolve(cls, required: bool = True)`
- `file_utils/mesh_ops.py::MeshOps.session(cls, input_path: str) -> _MeshSession`
- `file_utils/mesh_ops.py::OpSpec(class)`
- `geo_utils/polyline.py::Polyline.cumulative_lengths(cls, points: Sequence[Vec]) -> List[float]`
- `geo_utils/polyline.py::Polyline.point_at_arc(cls, points: Sequence[Sequence[float]], t: float) -> List[float]`
- `img_utils/_img_utils.py::ImgUtils.convert_f_to_l(cls, image)`
- `img_utils/_img_utils.py::ImgUtils.ktx2_available(cls) -> bool`
- `img_utils/_img_utils.py::ImgUtils.register_ktx2_encoder(cls, encoder) -> None`
- `img_utils/_img_utils.py::ImgUtils.resolve_ktx2_encoder(cls, required: bool = False)`
- `img_utils/ktx2_encoder.py::Ktx2Encoder(class)`
- `img_utils/ktx2_encoder.py::Ktx2Encoder.args_for(self, source: str, output: str, codec: str = 'UASTC', srgb: bool = True, mipmaps: bool = True, quality: Optional[int] = None) -> List[str]`
- `img_utils/ktx2_encoder.py::Ktx2Encoder.available(cls) -> bool`
- `img_utils/ktx2_encoder.py::Ktx2Encoder.encode(self, source: Union[str, 'Image.Image'], output: str, codec: str = 'UASTC', srgb: bool = True, mipmaps: bool = True, quality: Optional[int] = None) -> str`
- `img_utils/ktx2_encoder.py::Ktx2Encoder.read_header(cls, path: str) -> Dict[str, int]`
- `img_utils/ktx2_encoder.py::Ktx2Encoder.resolve_toktx(cls, required: bool = False) -> Optional[str]`

## Signature changed (7)

- `core_utils/engines/textures/map_optimizer.py::MapOptimizer.assess`
  - was: `(cls, texture_path: str, max_size: int = None, force_pot: Optional[bool] = None, optimize_bit_depth: bool = True, map_type: str = None, allow_palette: bool = False, image: 'Image.Image' = None, output_type: str = None, output_profile: str = None, predict_size: bool = False, enforce_budget: bool = False, lossy_quality: int = None) -> Dict[str, Any]`
  - now: `(cls, texture_path: str, max_size: int = None, force_pot: Optional[bool] = None, optimize_bit_depth: bool = True, map_type: str = None, allow_palette: bool = False, image: 'Image.Image' = None, output_type: str = None, output_profile: str = None, predict_size: bool = False, enforce_budget: bool = False, lossy_quality: int = None, pot_mode: Optional[str] = None) -> Dict[str, Any]`
- `core_utils/engines/textures/map_optimizer.py::MapOptimizer.optimize_map`
  - was: `(cls, texture_path: str, output_dir: str = None, output_type: str = None, max_size: int = None, force_pot: Optional[bool] = None, suffix_old: str = None, suffix_opt: str = None, old_files_folder: str = None, optimize_bit_depth: bool = True, check_existing: bool = False, map_type: str = None, allow_palette: bool = False, output_profile: str = None, enforce_budget: bool = False, lossy_quality: int = None) -> str`
  - now: `(cls, texture_path: str, output_dir: str = None, output_type: str = None, max_size: int = None, force_pot: Optional[bool] = None, suffix_old: str = None, suffix_opt: str = None, old_files_folder: str = None, optimize_bit_depth: bool = True, check_existing: bool = False, map_type: str = None, allow_palette: bool = False, output_profile: str = None, enforce_budget: bool = False, lossy_quality: int = None, pot_mode: Optional[str] = None) -> str`
- `core_utils/engines/textures/map_optimizer.py::MapOptimizer.plan`
  - was: `(cls, image: 'Image.Image', max_size: Optional[int] = None, force_pot: bool = False, optimize_bit_depth: bool = True, map_type_key: Optional[str] = None, allow_palette: bool = False, pot_mode: str = 'nearest') -> List[Op]`
  - now: `(cls, image: 'Image.Image', max_size: Optional[int] = None, force_pot: bool = False, optimize_bit_depth: bool = True, map_type_key: Optional[str] = None, allow_palette: bool = False, pot_mode: str = 'nearest', output_profile: Optional[str] = None, output_type: Optional[str] = None) -> List[Op]`
- `geo_utils/polyline.py::Polyline.frames`
  - was: `(points: Sequence[Vec], segments: int, closed: bool, up: Vec = (0.0, 1.0, 0.0)) -> List[Tuple[Vec, Vec, Vec]]`
  - now: `(cls, points: Sequence[Vec], segments: int, closed: bool, up: Vec = (0.0, 1.0, 0.0)) -> List[Tuple[Vec, Vec, Vec]]`
- `geo_utils/polyline.py::Polyline.length`
  - was: `(points: Sequence[Vec], closed: bool = False) -> float`
  - now: `(cls, points: Sequence[Vec], closed: bool = False) -> float`
- `img_utils/_img_utils.py::ImgUtils.save_image`
  - was: `(cls, image: Union[str, Image.Image], name: str, mode: str = None, bit_depth: int = None, compression: str = None, quality: int = None, **kwargs)`
  - now: `(cls, image: Union[str, Image.Image], name: str, mode: str = None, bit_depth: int = None, compression: str = None, quality: int = None, colorspace: str = None, **kwargs)`
- `net_utils/preview_server.py::PreviewBridge.push`
  - was: `(self, objects: Optional[List[Any]] = None, whole_scene: bool = False, open_browser: Union[bool, str] = 'auto', **params: Any) -> Optional[Dict[str, Any]]`
  - now: `(self, objects: Optional[List[Any]] = None, whole_scene: bool = False, open_browser: Union[bool, str] = 'auto', texture_format: Optional[str] = None, **params: Any) -> Optional[Dict[str, Any]]`
