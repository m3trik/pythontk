# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-05-16._

## Added (5)

- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.fix_glb_phantom_opaque_alpha(cls, glb_path: str) -> List[Dict]`
- `img_utils/_img_utils.py::ImgUtils.gaussian_blur(cls, image: Union[str, 'Image.Image', 'np.ndarray'], radius: float = 2.0, channel: Optional[str] = None) -> Union['Image.Image', 'np.ndarray']`
- `img_utils/_img_utils.py::ImgUtils.radial_gradient(size: Tuple[int, int], center: Tuple[float, float] = (0.5, 0.5), max_radius: Optional[float] = None, falloff_power: float = 1.0, invert: bool = False, dtype: type = None) -> 'np.ndarray'`
- `str_utils/_str_utils.py::StrUtils.apply_affix(string: str, prefix: str = '', suffix: str = '') -> str`
- `str_utils/_str_utils.py::StrUtils.strip_known_affix(string: str, prefix: str = '', suffix: str = '') -> str`

## Signature changed (6)

- `core_utils/app_launcher.py::AppLauncher.launch`
  - was: `(app_identifier, args=None, cwd=None, detached=True)`
  - now: `(app_identifier, args=None, cwd=None, detached=True, env=None)`
- `img_utils/_img_utils.py::ImgUtils.get_base_texture_name`
  - was: `(cls, filepath_or_filename: str) -> str`
  - now: `(cls, filepath_or_filename: str, prefix: str = '', suffix: str = '') -> str`
- `img_utils/map_factory.py::MapFactory.detect_normal_map_format`
  - was: `(cls, image: Union[str, 'Image.Image'], threshold: float = 0.1) -> Optional[str]`
  - now: `(cls, image: Union[str, 'Image.Image'], threshold: float = 0.25, min_gradient_std: float = 1.0) -> Optional[str]`
- `img_utils/map_factory.py::MapFactory.get_base_texture_name`
  - was: `(cls, filepath_or_filename: str) -> str`
  - now: `(cls, filepath_or_filename: str, prefix: str = '', suffix: str = '') -> str`
- `img_utils/map_factory.py::MapFactory.group_textures_by_set`
  - was: `(cls, image_paths: List[str]) -> Dict[str, List[str]]`
  - now: `(cls, image_paths: List[str], prefix: str = '', suffix: str = '') -> Dict[str, List[str]]`
- `img_utils/map_factory.py::MapFactory.prepare_maps`
  - was: `(cls, source: Union[str, List[str]], output_dir: str = None, group_by_set: bool = True, max_workers: int = 1, progress_callback: Callable = None, **kwargs) -> Union[List[str], Dict[str, List[str]]]`
  - now: `(cls, source: Union[str, List[str]], output_dir: str = None, group_by_set: bool = True, max_workers: int = 1, progress_callback: Callable = None, prefix: str = '', suffix: str = '', **kwargs) -> Union[List[str], Dict[str, List[str]]]`
