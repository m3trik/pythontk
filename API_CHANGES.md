# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-05-14._

## Added (5)

- `file_utils/_file_utils.py::FileUtils.next_version_path(filepath: str, format: str = '{stem}_v{n:03d}{ext}', start: int = 1) -> str`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.check_glb_materials(cls, glb_path: str) -> List[Dict[str, str]]`
- `img_utils/map_registry.py::MapRegistry.get_resolution_critical_types(self) -> List[str]`
- `img_utils/map_registry.py::MapRegistry.is_resolution_critical(self, name: str) -> bool`
- `str_utils/_str_utils.py::StrUtils.sequential_suffixes(count: int, switch_at: int = 26, lowercase: bool = False) -> List[str]`

## Signature changed (3)

- `img_utils/_img_utils.py::ImgUtils.enforce_mode`
  - was: `(cls, image: Image.Image, target_mode: str, allow_compatible: bool = True) -> Image.Image`
  - now: `(cls, image: Image.Image, target_mode: str, allow_compatible: bool = False) -> Image.Image`
- `img_utils/_img_utils.py::ImgUtils.optimize_texture`
  - was: `(cls, texture_path: str, output_dir: str = None, output_type: str = None, max_size: int = None, force_pot: bool = False, suffix_old: str = None, suffix_opt: str = None, old_files_folder: str = None, generate_mipmaps: bool = False, optimize_bit_depth: bool = True, check_existing: bool = False, map_type: str = None) -> str`
  - now: `(cls, texture_path: str, output_dir: str = None, output_type: str = None, max_size: int = None, force_pot: bool = False, suffix_old: str = None, suffix_opt: str = None, old_files_folder: str = None, generate_mipmaps: bool = False, optimize_bit_depth: bool = True, check_existing: bool = False, map_type: str = None, allow_palette: bool = False) -> str`
- `img_utils/_img_utils.py::ImgUtils.set_bit_depth`
  - was: `(cls, image, map_type: str) -> object`
  - now: `(cls, image, map_type: str, allow_palette: bool = False) -> object`
