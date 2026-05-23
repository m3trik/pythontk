# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-05-23._

## Removed (7)

- `img_utils/texture_optimizer.py::Op` — was `(class)`
- `img_utils/texture_optimizer.py::TextureOptimizer` — was `(class)`
- `img_utils/texture_optimizer.py::TextureOptimizer.apply` — was `(cls, image: 'Image.Image', plan: List[Op]) -> 'Image.Image'`
- `img_utils/texture_optimizer.py::TextureOptimizer.assess` — was `(cls, texture_path: str, max_size: int = None, force_pot: bool = False, optimize_bit_depth: bool = True, map_type: str = None, allow_palette: bool = False, generate_mipmaps: bool = False, image: 'Image.Image' = None) -> Dict[str, Any]`
- `img_utils/texture_optimizer.py::TextureOptimizer.batch_optimize_textures` — was `(cls, directory: str, **kwargs)`
- `img_utils/texture_optimizer.py::TextureOptimizer.optimize_texture` — was `(cls, texture_path: str, output_dir: str = None, output_type: str = None, max_size: int = None, force_pot: bool = False, suffix_old: str = None, suffix_opt: str = None, old_files_folder: str = None, generate_mipmaps: bool = False, optimize_bit_depth: bool = True, check_existing: bool = False, map_type: str = None, allow_palette: bool = False) -> str`
- `img_utils/texture_optimizer.py::TextureOptimizer.plan` — was `(cls, image: 'Image.Image', max_size: Optional[int] = None, force_pot: bool = False, optimize_bit_depth: bool = True, map_type_key: Optional[str] = None, allow_palette: bool = False, generate_mipmaps: bool = False) -> List[Op]`

## Added (7)

- `img_utils/map_optimizer.py::MapOptimizer(class)`
- `img_utils/map_optimizer.py::MapOptimizer.apply(cls, image: 'Image.Image', plan: List[Op]) -> 'Image.Image'`
- `img_utils/map_optimizer.py::MapOptimizer.assess(cls, texture_path: str, max_size: int = None, force_pot: bool = False, optimize_bit_depth: bool = True, map_type: str = None, allow_palette: bool = False, generate_mipmaps: bool = False, image: 'Image.Image' = None) -> Dict[str, Any]`
- `img_utils/map_optimizer.py::MapOptimizer.batch_optimize_maps(cls, directory: str, **kwargs)`
- `img_utils/map_optimizer.py::MapOptimizer.optimize_map(cls, texture_path: str, output_dir: str = None, output_type: str = None, max_size: int = None, force_pot: bool = False, suffix_old: str = None, suffix_opt: str = None, old_files_folder: str = None, generate_mipmaps: bool = False, optimize_bit_depth: bool = True, check_existing: bool = False, map_type: str = None, allow_palette: bool = False) -> str`
- `img_utils/map_optimizer.py::MapOptimizer.plan(cls, image: 'Image.Image', max_size: Optional[int] = None, force_pot: bool = False, optimize_bit_depth: bool = True, map_type_key: Optional[str] = None, allow_palette: bool = False, generate_mipmaps: bool = False) -> List[Op]`
- `img_utils/map_optimizer.py::Op(class)`
