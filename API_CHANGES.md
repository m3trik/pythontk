# pythontk — API Changes

_Diff vs the last release (origin/main @ c5a481d). Generated 2026-08-23._

## Added (26)

- `core_utils/app_handoff.py::HandoffBridge.carrier(self, request: HandoffRequest) -> str`
- `core_utils/app_handoff.py::HandoffBridge.carrier_of(path: str) -> str`
- `core_utils/app_handoff.py::HandoffBridge.payload_extension(self, request: HandoffRequest) -> str`
- `core_utils/process_stream.py::TeeStream(class)`
- `core_utils/process_stream.py::TeeStream.flush(self) -> None`
- `core_utils/process_stream.py::TeeStream.write(self, text: str) -> int`
- `core_utils/process_stream.py::TeeStream.writelines(self, lines) -> None`
- `file_utils/file_naming.py::FileNaming(class)`
- `file_utils/file_naming.py::FileNaming.expand(paths) -> List[str]`
- `file_utils/file_naming.py::FileNaming.find(cls, paths, fltr: str, regex: bool = False, ignore_case: bool = False) -> List[str]`
- `file_utils/file_naming.py::FileNaming.rename(cls, paths, to: str, fltr: str = '', regex: bool = False, ignore_case: bool = False, retain_suffix: bool = False, valid_suffixes: Optional[List[str]] = None, dry_run: bool = False, logger=None) -> List[Tuple[str, str]]`
- `file_utils/file_naming.py::FileNaming.set_case(cls, paths, case: str = 'capitalize', dry_run: bool = False, logger=None) -> List[Tuple[str, str]]`
- `file_utils/file_naming.py::FileNaming.stem(path: str) -> str`
- `file_utils/file_naming.py::FileNaming.strip_chars(cls, paths, num_chars: int = 1, trailing: bool = False, dry_run: bool = False, logger=None) -> List[Tuple[str, str]]`
- `file_utils/file_naming.py::RenamePlan(class)`
- `file_utils/file_naming.py::RenamePlan.apply(cls, plan: Sequence[PlanEntry], rename: Callable[[object, str], str], title: str = 'Rename', dry_run: bool = False, logger=None, link: Optional[Callable[[object, str], str]] = None, unit: str = 'item') -> List[Tuple[str, str]]`
- `file_utils/temp_artifacts.py::ScratchTwins(class)`
- `file_utils/temp_artifacts.py::ScratchTwins.create(self, source: str, payload: str) -> str`
- `file_utils/temp_artifacts.py::ScratchTwins.discard(self, path: str) -> bool`
- `file_utils/temp_artifacts.py::ScratchTwins.discard_except(self, keep: Optional[str] = None) -> List[str]`
- `file_utils/temp_artifacts.py::ScratchTwins.is_twin(self, path: str) -> bool`
- `file_utils/temp_artifacts.py::ScratchTwins.path_for(self, source: str) -> str`
- `iter_utils/_iter_utils.py::IterUtils.find_extrema_indices(values: Sequence[float], value_tolerance: float = 1e-05) -> 'np.ndarray'`
- `math_utils/_math_utils.py::MathUtils.fit_hermite_slopes(times: Sequence[float], values: Sequence[float], keep_indices: Sequence[int], flat_tolerance: float = 0.0) -> Tuple[List[float], List[float]]`
- `str_utils/_str_utils.py::StrUtils.retain_suffix(old_name: str, new_name: str, valid_suffixes: Optional[List[str]] = None) -> str`
- `str_utils/_str_utils.py::StrUtils.strip_suffix(name: str, suffixes: Iterable[str]) -> str`

## Signature changed (1)

- `file_utils/_file_utils.py::FileUtils.get_file_contents`
  - was: `(filepath: str, as_list=False, encoding='utf-8') -> None`
  - now: `(filepath: str, as_list: bool = False, encoding: str = 'utf-8') -> Optional[Union[str, List[str]]]`
