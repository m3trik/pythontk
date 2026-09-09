# pythontk — API Changes

_Diff vs the last release (origin/main @ dcd93fe)._

## Added (27)

- `net_utils/preview/playblast.py::PreviewPlayblast(class)`
- `net_utils/preview/playblast.py::PreviewPlayblast.active(self) -> List[Dict[str, Any]]`
- `net_utils/preview/playblast.py::PreviewPlayblast.add_frame(self, token: str, index: int, data: bytes) -> Dict[str, Any]`
- `net_utils/preview/playblast.py::PreviewPlayblast.begin(self, name: str, fps: float, start_frame: int = 1, frames: int = 0, content_type: str = 'image/png') -> Dict[str, Any]`
- `net_utils/preview/playblast.py::PreviewPlayblast.cancel(self, token: str) -> bool`
- `net_utils/preview/playblast.py::PreviewPlayblast.clip_name(self, token: str) -> str`
- `net_utils/preview/playblast.py::PreviewPlayblast.finish(self, token: str, output_dir: str, target: Optional[str] = None, stem: Optional[str] = None) -> Dict[str, Any]`
- `net_utils/preview/playblast.py::PreviewPlayblast.resolve_output_dir(source: Optional[Path], fallback: Path) -> Path`
- `net_utils/preview/server.py::PreviewServer.begin_playblast(self, **kwargs: Any) -> Dict[str, Any]`
- `net_utils/preview/server.py::PreviewServer.finish_playblast(self, token: str, target: Optional[str] = None) -> Dict[str, Any]`
- `net_utils/preview/server.py::PreviewServer.playblast(self) -> 'PreviewPlayblast'`
- `net_utils/preview/server.py::PreviewServer.recording_path(self, token: str) -> Optional[Path]`
- `vid_utils/sequence_exporter.py::CaptureResult(class)`
- `vid_utils/sequence_exporter.py::CaptureResult.pattern(self) -> str`
- `vid_utils/sequence_exporter.py::ExportResult(class)`
- `vid_utils/sequence_exporter.py::ExportResult.ok(self) -> bool`
- `vid_utils/sequence_exporter.py::ExportTarget(class)`
- `vid_utils/sequence_exporter.py::SequenceEncoder(class)`
- `vid_utils/sequence_exporter.py::SequenceEncoder.encode_sequence(self, capture: Union[CaptureResult, str], output_filepath: str, fps: Optional[float] = None, audio: Optional[Union[bool, str]] = None, quality: Optional[int] = None, **ffmpeg_options: Any) -> str`
- `vid_utils/sequence_exporter.py::SequenceEncoder.sequence_fps(self) -> float`
- `vid_utils/sequence_exporter.py::SequenceEncoder.sequence_name(self) -> str`
- `vid_utils/sequence_exporter.py::SequenceExporter(class)`
- `vid_utils/sequence_exporter.py::SequenceExporter.available_targets(cls) -> List[Tuple[str, str]]`
- `vid_utils/sequence_exporter.py::SequenceExporter.capture_sequence(self, directory: str, prefix: Optional[str] = None, start: Optional[int] = None, end: Optional[int] = None, camera: Optional[str] = None, image_format: str = 'png', **overrides: Any) -> CaptureResult`
- `vid_utils/sequence_exporter.py::SequenceExporter.capture_still(self, filepath: str, frame: Optional[int] = None, camera: Optional[str] = None, image_format: str = 'png', **overrides: Any) -> str`
- `vid_utils/sequence_exporter.py::SequenceExporter.export(self, output_dir: str, name: Optional[str] = None, targets: Union[str, Sequence[str]] = ('mp4',), range_mode: Optional[str] = None, start: Optional[int] = None, end: Optional[int] = None, camera: Optional[str] = None, keep_frames: bool = False, progress_callback: Optional[Callable[[int, int, str], None]] = None, **overrides: Any) -> List[ExportResult]`
- `vid_utils/sequence_exporter.py::SequenceExporter.resolve_frame_range(cls, mode: str = 'custom', start: Optional[int] = None, end: Optional[int] = None) -> Tuple[int, int]`
