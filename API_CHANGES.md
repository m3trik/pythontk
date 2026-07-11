# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-07-10._

## Added (7)

- `img_utils/_img_utils.py::ImgUtils.atlas_pixel_rects(rects: Sequence[Tuple[float, float, float, float]], size: Union[int, Tuple[int, int]]) -> List[Tuple[int, int, int, int]]`
- `img_utils/_img_utils.py::ImgUtils.inset_atlas_rects(rects: Sequence[Tuple[float, float, float, float]], size: Union[int, Tuple[int, int]], gutter: int) -> List[Tuple[float, float, float, float]]`
- `img_utils/_img_utils.py::ImgUtils.unique_dir_stems(dirs)`
- `math_utils/_math_utils.py::MathUtils.bspline_basis(knots: List[float], span: int, degree: int, s: float) -> List[float]`
- `math_utils/_math_utils.py::MathUtils.bspline_clamped_knots(stations: List[float], degree: int) -> List[float]`
- `math_utils/_math_utils.py::MathUtils.resolve_falloff_profile(profile: Union[str, Callable]) -> Callable[[float], float]`
- `net_utils/_net_utils.py::NetUtils.is_port_bindable(port: int, host: str = '127.0.0.1') -> bool`

## Signature changed (2)

- `img_utils/exposure_equalizer.py::ExposureEqualizer.equalize_directories`
  - was: `(self, source_dirs: Sequence[str], output_root: str, reference_dir: Optional[str] = None, suffix: str = '_eq', sample_count: int = 20, strength: float = 1.0, reference_strategy: str = 'first', quality: int = 100, preserve_exif: bool = True) -> List[str]`
  - now: `(self, source_dirs: Sequence[str], output_root: str, reference_dir: Optional[str] = None, suffix: str = '_eq', sample_count: int = 20, strength: float = 1.0, reference_strategy: str = 'first', quality: int = 100, preserve_exif: bool = True, per_image: bool = False, overwrite_output: bool = True) -> List[str]`
- `img_utils/image_curator.py::ImageCurator.curate`
  - was: `(self, source_dirs: Sequence[str], output_root: str, hash_threshold: int = 5, sharpness_floor: float = 0.0, sharpness_floor_percentile: Optional[float] = None, min_sharpness_fraction_of_median: float = 0.0, keep_per_cluster: int = 1, suffix: str = '_curated', progress: Optional[callable] = None, overwrite_output: bool = True) -> List[str]`
  - now: `(self, source_dirs: Sequence[str], output_root: str, hash_threshold: int = 0, sharpness_floor: float = 0.0, sharpness_floor_percentile: Optional[float] = None, min_sharpness_fraction_of_median: float = 0.0, keep_per_cluster: int = 1, suffix: str = '_curated', progress: Optional[callable] = None, overwrite_output: bool = True) -> List[str]`
