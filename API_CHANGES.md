# pythontk — API Changes

_Diff vs the last release (origin/main @ 4c19a97). Generated 2026-08-20._

## Added (1)

- `core_utils/logging_mixin.py::LoggingMixin.use_logger(self, logger: Optional[internal_logging.Logger]) -> None`

## Signature changed (1)

- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.optimize_glb_textures`
  - was: `(cls, glb: GlbTarget, max_size: int = 2048, image_format: str = 'WEBP', quality: int = 85, workers: Optional[int] = None) -> Dict[str, Any]`
  - now: `(cls, glb: GlbTarget, max_size: int = 2048, image_format: str = 'WEBP', quality: int = 85, workers: Optional[int] = None, ktx2_fallback: bool = True) -> Dict[str, Any]`
