# pythontk — API Changes

_Diff vs the last release (origin/main @ 95d4939)._

## Added (4)

- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.compact_glb_animations(cls, glb: GlbTarget) -> Dict[str, int]`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.drop_glb_texture_fallbacks(cls, glb: GlbTarget) -> Dict[str, int]`
- `file_utils/mesh_convert/export_verify.py::ExportVerifier.check_clip_origin(self) -> List[Finding]`
- `str_utils/_str_utils.py::StrUtils.to_legal_name(name: str) -> str`

## Signature changed (4)

- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.apply_glb_clips`
  - was: `(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]`
  - now: `(cls, glb: GlbTarget, *, mode: str = 'both') -> Optional[Dict[str, Any]]`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.fbx_to_glb`
  - was: `(cls, src: str, dst: Optional[str] = None, *, overwrite: bool = False, auto_install: bool = True, prompt: Union[bool, Callable[[str], bool]] = True, timeout: Optional[float] = AUTO_TIMEOUT, extra_args: Optional[List[str]] = None, sidecar: Optional[Dict[str, Any]] = None, lightmaps: bool = True, lightmap_dirs: Sequence[str] = (), shadow_dirs: Sequence[str] = (), report: Optional[Dict[str, Any]] = None) -> str`
  - now: `(cls, src: str, dst: Optional[str] = None, *, overwrite: bool = False, auto_install: bool = True, prompt: Union[bool, Callable[[str], bool]] = True, timeout: Optional[float] = AUTO_TIMEOUT, extra_args: Optional[List[str]] = None, sidecar: Optional[Dict[str, Any]] = None, lightmaps: bool = True, lightmap_dirs: Sequence[str] = (), shadow_dirs: Sequence[str] = (), clip_mode: str = 'both', report: Optional[Dict[str, Any]] = None) -> str`
- `file_utils/mesh_convert/glb_clips.py::GlbClips.rebuild`
  - was: `(cls, edit: Any, takes: Sequence[Dict[str, Any]], fps: float, source_zero: float = 0.0) -> Optional[Dict[str, Any]]`
  - now: `(cls, edit: Any, takes: Sequence[Dict[str, Any]], fps: float, source_zero: float = 0.0, *, cut_shots: bool = True, keep_sequence: bool = True) -> Optional[Dict[str, Any]]`
- `file_utils/mesh_convert/glb_pipeline.py::GlbPipeline.build`
  - was: `(cls, src: str, dst: Optional[str] = None, *, sidecar: Optional[Dict[str, Any]] = None, lightmap_dirs: Sequence[str] = (), texture_params: Optional[Dict[str, Any]] = None, downsize: bool = True, scratch_path: Optional[Callable[[str], str]] = None, release_source: Optional[Callable[[str], Any]] = None, progress: Optional[Callable[[str], Any]] = None, logger: Any = None) -> Dict[str, Any]`
  - now: `(cls, src: str, dst: Optional[str] = None, *, sidecar: Optional[Dict[str, Any]] = None, lightmap_dirs: Sequence[str] = (), texture_params: Optional[Dict[str, Any]] = None, clip_mode: str = 'both', downsize: bool = True, scratch_path: Optional[Callable[[str], str]] = None, release_source: Optional[Callable[[str], Any]] = None, progress: Optional[Callable[[str], Any]] = None, logger: Any = None) -> Dict[str, Any]`
