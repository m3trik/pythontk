# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-07-29._

## Added (4)

- `core_utils/engines/textures/map_compositor.py::MapCompositor.written_paths(self) -> List[str]`
- `math_utils/_math_utils.py::MathUtils.fit_into_tile(bounds: Tuple[float, float, float, float], tile: Tuple[int, int], margin: float = 0.0) -> Tuple[float, float]`
- `math_utils/_math_utils.py::MathUtils.majority_tile(bounds) -> Optional[Tuple[int, int]]`
- `math_utils/_math_utils.py::MathUtils.uv_tile_margin(cls, map_size: int, factor: int = 256) -> float`

## Signature changed (1)

- `core_utils/engines/textures/map_compositor.py::MapCompositor.apply_output_template`
  - was: `(self, output_dir: str) -> List[str]`
  - now: `(self, output_dir: str, files: Optional[List[str]] = None) -> List[str]`
