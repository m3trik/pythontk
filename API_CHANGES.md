# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-05-07._

## Added (18)

- `core_utils/namespace_handler.py::NamespaceHandler.peek(self, key: str, default: Any = None) -> Any`
- `file_utils/_file_utils.py::FileUtils.atomic_write_text(filepath: str, content: str, encoding: str = 'utf-8') -> None`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert(class)`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.fbx_to_glb(cls, src: str, dst: Optional[str] = None, *, overwrite: bool = False, auto_install: bool = True, prompt: bool = True, timeout: Optional[float] = DEFAULT_TIMEOUT, extra_args: Optional[List[str]] = None) -> str`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.resolve_binary(cls, required: bool = True, auto_install: bool = False, prompt: bool = True) -> Optional[str]`
- `file_utils/mesh_convert/slots.py::MeshConvertSlots(class)`
- `file_utils/mesh_convert/slots.py::MeshConvertSlots.fbx_provider(self, fn: Optional[Callable[[], Iterable[str]]]) -> None`
- `file_utils/mesh_convert/slots.py::MeshConvertSlots.header_init(self, widget) -> None`
- `file_utils/mesh_convert/slots.py::MeshConvertSlots.source_dir(self, value: str) -> None`
- `file_utils/mesh_convert/slots.py::MeshConvertSlots.tb000(self, widget) -> None`
- `file_utils/mesh_convert/slots.py::MeshConvertSlots.tb000_init(self, widget) -> None`
- `file_utils/mesh_convert/slots.py::MeshConvertUi(class)`
- `img_utils/map_converter.py::MapConverterSlots.header_init(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.texture_provider(self, fn)`
- `str_utils/_str_utils.py::StrUtils.alpha_sequence(index: int) -> str`
- `str_utils/_str_utils.py::StrUtils.resolve_name_collisions(names: Iterable[str], strip: Union[str, List[str]] = '', strip_trailing_ints: bool = False, strip_trailing_alpha: bool = False, collision_suffix: Union[str, Callable[[int, int], str], None] = 'alpha', suffix_separator: str = '_') -> Dict[str, str]`
- `str_utils/fuzzy_matcher.py::FuzzyMatcher.find_unique_match(target: str, candidates: List[str], score_threshold: float = 0.5, ambiguity_delta: float = 0.05, use_base_name: bool = True, use_substring: bool = True, use_prefix: bool = True, use_ratio: bool = False) -> Tuple[Optional[str], float, str]`
- `str_utils/fuzzy_matcher.py::FuzzyMatcher.find_with_fallbacks(target: str, candidates: List[str], strategies: List[Union[str, Callable]], score_threshold: float = 0.5, ambiguity_delta: float = 0.05, stop_on_ambiguous: bool = True) -> Tuple[Optional[str], float, str, str]`
