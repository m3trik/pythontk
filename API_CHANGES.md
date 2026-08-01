# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-08-01._

## Removed (21)

- `core_utils/engines/shots/manifest/behaviors/_spec.py::BehaviorSpec.validate_attributes` — was `(value: Any) -> List[str]`
- `core_utils/engines/shots/manifest/behaviors/_spec.py::BehaviorSpec.validate_duration` — was `(value: Any) -> List[str]`
- `core_utils/engines/shots/manifest/behaviors/_spec.py::BehaviorSpec.validate_verify` — was `(value: Any) -> List[str]`
- `core_utils/engines/shots/manifest/mapping/_spec.py::MappingSpec.validate_audio_resolve` — was `(value: Any) -> List[str]`
- `core_utils/engines/shots/manifest/mapping/_spec.py::MappingSpec.validate_default_behaviors` — was `(value: Any) -> List[str]`
- `core_utils/package_manager.py::PackageManager.check_version` — was `(self, package_name=None, python_path=None) -> None`
- `core_utils/package_manager.py::PackageManager.install` — was `(self, package_name)`
- `core_utils/package_manager.py::PackageManager.installed_ver` — was `(self) -> str`
- `core_utils/package_manager.py::PackageManager.installed_version` — was `(self, package_name)`
- `core_utils/package_manager.py::PackageManager.is_outdated` — was `(self, package_name: str) -> bool`
- `core_utils/package_manager.py::PackageManager.latest_ver` — was `(self) -> str`
- `core_utils/package_manager.py::PackageManager.latest_version` — was `(self, package_name)`
- `core_utils/package_manager.py::PackageManager.list_outdated_packages` — was `(self)`
- `core_utils/package_manager.py::PackageManager.list_packages` — was `(self)`
- `core_utils/package_manager.py::PackageManager.new_version_available` — was `(self) -> bool`
- `core_utils/package_manager.py::PackageManager.package_details` — was `(self, package_name)`
- `core_utils/package_manager.py::PackageManager.start_version_check` — was `(self, package_name=None, python_path=None) -> None`
- `core_utils/package_manager.py::PackageManager.uninstall` — was `(self, package_name)`
- `core_utils/package_manager.py::PackageManager.update` — was `(self, package_name)`
- `core_utils/package_manager.py::PackageManager.update_requirements` — was `(file_path=None, inc=None, exc=None) -> list`
- `core_utils/package_manager.py::PackageManager.update_version` — was `(filepath: str, change: str = 'increment', version_part: str = 'patch', max_version_parts: tuple = (99, 99), version_regex: str = '__version__\\s*=\\s*[\'\\"](\\d+)\\.(\\d+)\\.(\\d+)[\'\\"]') -> str`
