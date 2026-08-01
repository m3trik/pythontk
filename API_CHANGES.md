# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-08-01._

## Added (21)

- `core_utils/engines/shots/manifest/behaviors/_spec.py::BehaviorSpec.validate_attributes(value: Any) -> List[str]`
- `core_utils/engines/shots/manifest/behaviors/_spec.py::BehaviorSpec.validate_duration(value: Any) -> List[str]`
- `core_utils/engines/shots/manifest/behaviors/_spec.py::BehaviorSpec.validate_verify(value: Any) -> List[str]`
- `core_utils/engines/shots/manifest/mapping/_spec.py::MappingSpec.validate_audio_resolve(value: Any) -> List[str]`
- `core_utils/engines/shots/manifest/mapping/_spec.py::MappingSpec.validate_default_behaviors(value: Any) -> List[str]`
- `core_utils/package_manager.py::PackageManager.check_version(self, package_name=None, python_path=None) -> None`
- `core_utils/package_manager.py::PackageManager.install(self, package_name)`
- `core_utils/package_manager.py::PackageManager.installed_ver(self) -> str`
- `core_utils/package_manager.py::PackageManager.installed_version(self, package_name)`
- `core_utils/package_manager.py::PackageManager.is_outdated(self, package_name: str) -> bool`
- `core_utils/package_manager.py::PackageManager.latest_ver(self) -> str`
- `core_utils/package_manager.py::PackageManager.latest_version(self, package_name)`
- `core_utils/package_manager.py::PackageManager.list_outdated_packages(self)`
- `core_utils/package_manager.py::PackageManager.list_packages(self)`
- `core_utils/package_manager.py::PackageManager.new_version_available(self) -> bool`
- `core_utils/package_manager.py::PackageManager.package_details(self, package_name)`
- `core_utils/package_manager.py::PackageManager.start_version_check(self, package_name=None, python_path=None) -> None`
- `core_utils/package_manager.py::PackageManager.uninstall(self, package_name)`
- `core_utils/package_manager.py::PackageManager.update(self, package_name)`
- `core_utils/package_manager.py::PackageManager.update_requirements(file_path=None, inc=None, exc=None) -> list`
- `core_utils/package_manager.py::PackageManager.update_version(filepath: str, change: str = 'increment', version_part: str = 'patch', max_version_parts: tuple = (99, 99), version_regex: str = '__version__\\s*=\\s*[\'\\"](\\d+)\\.(\\d+)\\.(\\d+)[\'\\"]') -> str`
