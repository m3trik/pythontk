# pythontk — API Changes

_Diff vs the last release (origin/main @ 81f3387)._

## Added (33)

- `core_utils/app_installer.py::AppInstaller.consent(prompt: Union[bool, Callable[[str], bool], None], question: str) -> Optional[bool]`
- `core_utils/engines/textures/map_factory/_map_factory.py::MapFactory.extract_channels(cls, packed_type: str, packed_path: Optional[str], targets: List[str], config: Dict[str, Any] = None) -> Optional[Dict[str, str]]`
- `core_utils/naming_convention.py::AffixRule(class)`
- `core_utils/naming_convention.py::AffixRule.apply(self, name: str, *, default: str = 'suffix') -> str`
- `core_utils/naming_convention.py::AffixRule.as_dict(self) -> Dict[str, str]`
- `core_utils/naming_convention.py::AffixRule.parts(self, *, default: str = 'suffix') -> Tuple[str, str]`
- `core_utils/naming_convention.py::NamingConvention(class)`
- `core_utils/naming_convention.py::NamingConvention.affix(cls, key: str) -> str`
- `core_utils/naming_convention.py::NamingConvention.affix_parts(cls, key: str, *, default: str = 'suffix') -> Tuple[str, str]`
- `core_utils/naming_convention.py::NamingConvention.all_affixes(cls) -> List[str]`
- `core_utils/naming_convention.py::NamingConvention.apply(cls, name: str, key: str, *, default: str = 'suffix') -> str`
- `core_utils/naming_convention.py::NamingConvention.bind(cls, bindings: Iterable[Tuple[str, str, str]], overrides: Optional[Dict[str, str]] = None, modes: Optional[Dict[str, str]] = None) -> Dict[str, AffixRule]`
- `core_utils/naming_convention.py::NamingConvention.config_path(cls)`
- `core_utils/naming_convention.py::NamingConvention.get(cls, key: str, fallback: Optional[AffixRule] = None) -> AffixRule`
- `core_utils/naming_convention.py::NamingConvention.items(cls) -> List[Tuple[str, AffixRule]]`
- `core_utils/naming_convention.py::NamingConvention.keys(cls) -> List[str]`
- `core_utils/naming_convention.py::NamingConvention.label(cls, key: str) -> str`
- `core_utils/naming_convention.py::NamingConvention.mode(cls, key: str) -> str`
- `core_utils/naming_convention.py::NamingConvention.reload(cls) -> Dict[str, AffixRule]`
- `core_utils/naming_convention.py::NamingConvention.reset(cls, key: Optional[str] = None) -> Dict[str, AffixRule]`
- `core_utils/naming_convention.py::NamingConvention.resolve(cls, *, refresh: bool = False) -> Dict[str, AffixRule]`
- `core_utils/naming_convention.py::NamingConvention.set(cls, key: str, text: str, mode: str = 'auto', label: str = '') -> AffixRule`
- `core_utils/naming_convention.py::NamingConvention.update(cls, mapping: Dict[str, object]) -> Dict[str, AffixRule]`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.build_fbx_handoff(cls, channels: Iterable[str], source: Optional[Dict[str, str]] = None) -> Dict[str, Any]`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.describe_texture_pass(cls, summary: Dict[str, Any], image_format: str, max_size: int = 0) -> str`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.set_glb_alpha_mode(cls, glb: GlbTarget, alpha_mode: Dict[str, Dict[str, Any]]) -> List[Dict]`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.strip_fbx_handoff(cls, gltf: dict) -> int`
- `img_utils/ktx2_encoder.py::Ktx2Encoder.not_installed_error(cls, detail: str = '') -> FileNotFoundError`
- `net_utils/preview_server.py::PreviewBridge.lightmap_search_dirs(self) -> Sequence[str]`
- `net_utils/preview_server.py::PreviewBridge.lightmap_summary(result: Optional[Dict[str, Any]]) -> str`
- `net_utils/preview_server.py::PreviewPassContext.lightmap_search_dirs(self) -> Sequence[str]`
- `str_utils/_str_utils.py::StrUtils.delimit_affix(text: str, mode: str = 'suffix', *, delimiter: str = '_') -> str`
- `str_utils/_str_utils.py::StrUtils.strip_any_affix(string: str, known, *, exclude=(), one: bool = True, case_sensitive: bool = True) -> str`

## Signature changed (7)

- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.fbx_to_glb`
  - was: `(cls, src: str, dst: Optional[str] = None, *, overwrite: bool = False, auto_install: bool = True, prompt: bool = True, timeout: Optional[float] = DEFAULT_TIMEOUT, extra_args: Optional[List[str]] = None, sidecar: Optional[Dict[str, Any]] = None, lightmaps: bool = True) -> str`
  - now: `(cls, src: str, dst: Optional[str] = None, *, overwrite: bool = False, auto_install: bool = True, prompt: Union[bool, Callable[[str], bool]] = True, timeout: Optional[float] = DEFAULT_TIMEOUT, extra_args: Optional[List[str]] = None, sidecar: Optional[Dict[str, Any]] = None, lightmaps: bool = True, lightmap_dirs: Sequence[str] = ()) -> str`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.resolve_binary`
  - was: `(cls, required: bool = True, auto_install: bool = False, prompt: bool = True) -> Optional[str]`
  - now: `(cls, required: bool = True, auto_install: bool = False, prompt: Union[bool, Callable[[str], bool]] = True) -> Optional[str]`
- `file_utils/uv_unwrap/_uv_unwrap.py::UvUnwrap.resolve_engine`
  - was: `(cls, engine: str, required: bool = True, auto_install: bool = False, prompt: bool = True) -> Optional[str]`
  - now: `(cls, engine: str, required: bool = True, auto_install: bool = False, prompt: Union[bool, Callable[[str], bool]] = True) -> Optional[str]`
- `file_utils/uv_unwrap/_uv_unwrap.py::UvUnwrap.unwrap`
  - was: `(cls, obj_in: str, obj_out: Optional[str] = None, *, engine: str = 'mof', overwrite: bool = False, auto_install: bool = True, prompt: bool = True, timeout: Optional[float] = DEFAULT_TIMEOUT, **params) -> str`
  - now: `(cls, obj_in: str, obj_out: Optional[str] = None, *, engine: str = 'mof', overwrite: bool = False, auto_install: bool = True, prompt: Union[bool, Callable[[str], bool]] = True, timeout: Optional[float] = DEFAULT_TIMEOUT, **params) -> str`
- `img_utils/_img_utils.py::ImgUtils.resolve_ktx2_encoder`
  - was: `(cls, required: bool = False)`
  - now: `(cls, required: bool = False, auto_install: bool = False, prompt: Union[bool, Callable[[str], bool]] = True)`
- `img_utils/ktx2_encoder.py::Ktx2Encoder.resolve_toktx`
  - was: `(cls, required: bool = False) -> Optional[str]`
  - now: `(cls, required: bool = False, auto_install: bool = False, prompt: Union[bool, Callable[[str], bool]] = True) -> Optional[str]`
- `str_utils/_str_utils.py::StrUtils.strip_known_affix`
  - was: `(string: str, prefix: str = '', suffix: str = '') -> str`
  - now: `(string: str, prefix: str = '', suffix: str = '', *, case_sensitive: bool = False) -> str`
