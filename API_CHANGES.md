# pythontk — API Changes

_Diff vs the last release (origin/main @ 3d7df22)._

## Added (27)

- `core_utils/engines/shots/shot_model.py::ShotStore.remove_stale_shots(self) -> List[ShotBlock]`
- `core_utils/engines/shots/shot_model.py::ShotStore.stale_shots(self) -> List[ShotBlock]`
- `core_utils/engines/textures/map_registry.py::MapRegistry.estimate_gpu_bytes(self, width: int, height: int, name: Optional[str] = None, tiles: int = 1) -> Tuple[float, float]`
- `core_utils/hierarchy_baseline.py::HierarchyBaseline.adopt(cls, baseline: Iterable[str], shipped: Optional[Iterable[str]]) -> Optional[Set[str]]`
- `core_utils/hierarchy_baseline.py::HierarchyBaseline.recorded_by(cls, record) -> Optional[str]`
- `core_utils/scene_records.py::ExportContext.note(self, text: str) -> None`
- `core_utils/scene_records.py::RecordSpec.path_keys(self) -> Optional[Tuple[str, ...]]`
- `core_utils/scene_records.py::SceneStoreBase.scene_path(cls) -> str`
- `core_utils/scene_records.py::SceneStoreBase.writer_stamp(cls) -> str`
- `core_utils/scene_records.py::SceneStoreBase.written_here(cls, stamp: Optional[str]) -> bool`
- `net_utils/preview/bridge.py::PreviewBridge.timing_summary(cls, result: Optional[Dict[str, Any]]) -> str`
- `net_utils/share_tunnel.py::ShareTunnel.cancel(self) -> None`
- `str_utils/report_doc.py::ReportDoc(class)`
- `str_utils/report_doc.py::ReportDoc.action(cls, text: Any, verb: str, /, **params: Any) -> Inline`
- `str_utils/report_doc.py::ReportDoc.color(cls, tone: Optional[str]) -> Optional[str]`
- `str_utils/report_doc.py::ReportDoc.extend(self, other: 'ReportDoc') -> 'ReportDoc'`
- `str_utils/report_doc.py::ReportDoc.fields(self, rows: Iterable[Tuple[Any, Any]]) -> 'ReportDoc'`
- `str_utils/report_doc.py::ReportDoc.file(cls, path: str, text: Optional[Any] = None) -> Inline`
- `str_utils/report_doc.py::ReportDoc.heading(self, text: Any, level: int = 2, tone: Optional[str] = 'heading') -> 'ReportDoc'`
- `str_utils/report_doc.py::ReportDoc.items(self, items: Iterable[Any], tone: Optional[str] = None) -> 'ReportDoc'`
- `str_utils/report_doc.py::ReportDoc.join(cls, parts: Iterable[Any], sep: str = ', ') -> Inline`
- `str_utils/report_doc.py::ReportDoc.link(cls, text: Any, href: str, tone: Optional[str] = 'link') -> Inline`
- `str_utils/report_doc.py::ReportDoc.span(cls, text: Any, tone: Optional[str] = None, bold: bool = False) -> Inline`
- `str_utils/report_doc.py::ReportDoc.table(self, headers: Sequence[Any], rows: Iterable[Sequence[Any]], align: Optional[Union[str, Sequence[str]]] = None, title: Optional[Any] = None, footer: Optional[Any] = None, wrap: Optional[Iterable[int]] = None) -> 'ReportDoc'`
- `str_utils/report_doc.py::ReportDoc.text(self, text: Any, tone: Optional[str] = None) -> 'ReportDoc'`
- `str_utils/report_doc.py::ReportDoc.to_html(self) -> str`
- `str_utils/report_doc.py::ReportDoc.to_text(self) -> str`

## Signature changed (8)

- `core_utils/hierarchy_baseline.py::HierarchyBaseline.encode`
  - was: `(cls, paths: Iterable[str]) -> Dict`
  - now: `(cls, paths: Iterable[str], scene: Optional[str] = None) -> Dict`
- `core_utils/logging_mixin.py::TableMixin.format_table`
  - was: `(self, data: List[List[Any]], headers: List[str], title: Optional[str] = None, col_max_width: int = 60, max_width: int = 160) -> str`
  - now: `(self, data: List[List[Any]], headers: List[str], title: Optional[str] = None, col_max_width: int = 60, max_width: int = 160, wrap: bool = False, markup: bool = True) -> str`
- `core_utils/scene_records.py::RecordTransfer.absolute_paths`
  - was: `(payload: Any, ctx: TransferContext) -> Any`
  - now: `(payload: Any, ctx: TransferContext, keys: Optional[Tuple[str, ...]] = None) -> Any`
- `core_utils/scene_records.py::RecordTransfer.arriving_paths`
  - was: `(payload: Any, ctx: TransferContext) -> Any`
  - now: `(payload: Any, ctx: TransferContext, keys: Optional[Tuple[str, ...]] = None) -> Any`
- `core_utils/scene_records.py::SceneRecords.map_paths`
  - was: `(payload: Any, spell: Callable[[str], str]) -> Any`
  - now: `(payload: Any, spell: Callable[[str], str], keys: Optional[Tuple[str, ...]] = None) -> Any`
- `net_utils/preview/bridge.py::PreviewBridge.lightmap_summary`
  - was: `(result: Optional[Dict[str, Any]]) -> str`
  - now: `(cls, result: Optional[Dict[str, Any]]) -> str`
- `net_utils/preview/bridge.py::PreviewBridge.share`
  - was: `(self, provider: Optional[str] = None, alias: Union[str, os.PathLike, Callable[[Optional[str]], Any], bool, None] = None, alias_url: Optional[str] = None) -> Dict[str, Any]`
  - now: `(self, provider: Optional[str] = None, alias: Union[str, os.PathLike, Callable[[Optional[str]], Any], bool, None] = None, alias_url: Optional[str] = None, on_step: Optional[Callable[[Any], Any]] = None) -> Dict[str, Any]`
- `net_utils/preview/server.py::PreviewServer.share`
  - was: `(self, provider: Optional[str] = None, alias: Union[str, os.PathLike, Callable[[Optional[str]], Any], bool, None] = None, alias_url: Optional[str] = None, timeout: Optional[float] = None) -> Dict[str, Any]`
  - now: `(self, provider: Optional[str] = None, alias: Union[str, os.PathLike, Callable[[Optional[str]], Any], bool, None] = None, alias_url: Optional[str] = None, timeout: Optional[float] = None, on_step: Optional[Callable[[Any], Any]] = None) -> Dict[str, Any]`
