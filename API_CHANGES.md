# pythontk — API Changes

_Diff vs the last release (origin/main @ ccdb96e)._

## Added (1)

- `core_utils/engines/textures/map_registry.py::MapRegistry.split_map_suffix(self, name_only: str) -> Tuple[str, str]`

## Signature changed (4)

- `file_utils/file_naming.py::FileNaming.find`
  - was: `(cls, paths, fltr: str, regex: bool = False, ignore_case: bool = False) -> List[str]`
  - now: `(cls, paths, fltr: str, regex: bool = False, ignore_case: bool = False, base_names: bool = False) -> List[str]`
- `file_utils/file_naming.py::FileNaming.rename`
  - was: `(cls, paths, to: str, fltr: str = '', regex: bool = False, ignore_case: bool = False, retain_suffix: bool = False, valid_suffixes: Optional[List[str]] = None, dry_run: bool = False, logger=None) -> List[Tuple[str, str]]`
  - now: `(cls, paths, to: str, fltr: str = '', regex: bool = False, ignore_case: bool = False, retain_suffix: bool = False, valid_suffixes: Optional[List[str]] = None, base_names: bool = False, dry_run: bool = False, logger=None) -> List[Tuple[str, str]]`
- `file_utils/file_naming.py::FileNaming.set_case`
  - was: `(cls, paths, case: str = 'capitalize', dry_run: bool = False, logger=None) -> List[Tuple[str, str]]`
  - now: `(cls, paths, case: str = 'capitalize', base_names: bool = False, dry_run: bool = False, logger=None) -> List[Tuple[str, str]]`
- `file_utils/file_naming.py::FileNaming.strip_chars`
  - was: `(cls, paths, num_chars: int = 1, trailing: bool = False, dry_run: bool = False, logger=None) -> List[Tuple[str, str]]`
  - now: `(cls, paths, num_chars: int = 1, trailing: bool = False, base_names: bool = False, dry_run: bool = False, logger=None) -> List[Tuple[str, str]]`
