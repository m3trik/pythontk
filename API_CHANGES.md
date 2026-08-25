# pythontk — API Changes

_Diff vs the last release (origin/main @ 3e8a339)._

## Added (3)

- `core_utils/engines/textures/output_template.py::OutputTemplates.profile_outline(cls, profile: str, *, delivery: bool = True, base_name: Optional[str] = None) -> Dict[str, Any]`
- `core_utils/engines/textures/output_template.py::OutputTemplates.profile_outlines(cls, **kwargs) -> List[Tuple[str, Dict[str, Any]]]`
- `core_utils/package_manager.py::PackageManager.install_targeted(self, specs, target_dir, upgrade=False) -> List[str]`
