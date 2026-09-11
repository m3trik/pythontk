# pythontk — API Changes

_Diff vs the last release (origin/main @ d1a0b2b)._

## Added (7)

- `core_utils/process_exit.py::ProcessExit(class)`
- `core_utils/process_exit.py::ProcessExit.hard_exit(code: int = 0) -> NoReturn`
- `geo_utils/polyline.py::Polyline.transport_frames(cls, points: Sequence[Vec], up: Optional[Vec] = None) -> List[Tuple[Vec, Vec, Vec]]`
- `img_utils/_img_utils.py::ImgUtils.channels_carrying_data(cls, image, bands) -> Tuple[str, ...]`
- `net_utils/preview/bridge.py::FilePreviewBridge(class)`
- `net_utils/preview/bridge.py::FilePreviewBridge.lightmap_search_dirs(self) -> Sequence[str]`
- `net_utils/preview/server.py::PreviewServer.webxr_browser(cls) -> Optional[str]`

## Signature changed (3)

- `file_utils/_file_utils.py::FileUtils.get_file_contents`
  - was: `(filepath: str, as_list: bool = False, encoding: str = 'utf-8') -> Optional[Union[str, List[str]]]`
  - now: `(filepath: str, as_list: bool = False, encoding: str = 'utf-8', default: Optional[Union[str, List[str]]] = None) -> Optional[Union[str, List[str]]]`
- `img_utils/_img_utils.py::ImgUtils.dropped_channels`
  - was: `(cls, mode: str, ext: str) -> Tuple[str, ...]`
  - now: `(cls, mode: str, ext: str = '', *, target_mode: str = '') -> Tuple[str, ...]`
- `net_utils/preview/bridge.py::PreviewBridge.push`
  - was: `(self, objects: Optional[List[Any]] = None, scope: str = 'selected', open_browser: Union[bool, str, None] = None, texture_format: Optional[str] = None, scripts: Optional[Union[Dict[str, Any], List[str], tuple]] = None, **params: Any) -> Optional[Dict[str, Any]]`
  - now: `(self, objects: Optional[List[Any]] = None, scope: str = 'selected', open_browser: Union[bool, str, None] = None, texture_format: Optional[str] = None, scripts: Optional[Union[Dict[str, Any], List[str], tuple]] = None, progress: Optional[Callable[[str], Any]] = None, **params: Any) -> Optional[Dict[str, Any]]`
