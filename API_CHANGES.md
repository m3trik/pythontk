# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-05-22._

## Removed (2)

- `img_utils/_img_utils.py::ImgUtils.batch_optimize_textures` — was `(cls, directory: str, **kwargs)`
- `img_utils/_img_utils.py::ImgUtils.optimize_texture` — was `(cls, texture_path: str, output_dir: str = None, output_type: str = None, max_size: int = None, force_pot: bool = False, suffix_old: str = None, suffix_opt: str = None, old_files_folder: str = None, generate_mipmaps: bool = False, optimize_bit_depth: bool = True, check_existing: bool = False, map_type: str = None, allow_palette: bool = False) -> str`

## Added (9)

- `img_utils/texture_optimizer.py::Op(class)`
- `img_utils/texture_optimizer.py::TextureOptimizer(class)`
- `img_utils/texture_optimizer.py::TextureOptimizer.apply(cls, image: 'Image.Image', plan: List[Op]) -> 'Image.Image'`
- `img_utils/texture_optimizer.py::TextureOptimizer.assess(cls, texture_path: str, max_size: int = None, force_pot: bool = False, optimize_bit_depth: bool = True, map_type: str = None, allow_palette: bool = False, generate_mipmaps: bool = False, image: 'Image.Image' = None) -> Dict[str, Any]`
- `img_utils/texture_optimizer.py::TextureOptimizer.batch_optimize_textures(cls, directory: str, **kwargs)`
- `img_utils/texture_optimizer.py::TextureOptimizer.optimize_texture(cls, texture_path: str, output_dir: str = None, output_type: str = None, max_size: int = None, force_pot: bool = False, suffix_old: str = None, suffix_opt: str = None, old_files_folder: str = None, generate_mipmaps: bool = False, optimize_bit_depth: bool = True, check_existing: bool = False, map_type: str = None, allow_palette: bool = False) -> str`
- `img_utils/texture_optimizer.py::TextureOptimizer.plan(cls, image: 'Image.Image', max_size: Optional[int] = None, force_pot: bool = False, optimize_bit_depth: bool = True, map_type_key: Optional[str] = None, allow_palette: bool = False, generate_mipmaps: bool = False) -> List[Op]`
- `str_utils/_str_utils.py::StrUtils.infer_affix_mode(text: str, delimiter: str = '_', *, default: str = 'prefix') -> str`
- `str_utils/_str_utils.py::StrUtils.split_affix(text: str, mode: str = 'auto', *, default: str = 'prefix', delimiter: str = '_') -> Tuple[str, str]`
