# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-05-19._

## Removed (43)

- `file_utils/mesh_convert/slots.py::MeshConvertSlots` — was `(class)`
- `file_utils/mesh_convert/slots.py::MeshConvertSlots.fbx_provider` — was `(self, fn: Optional[Callable[[], Iterable[str]]]) -> None`
- `file_utils/mesh_convert/slots.py::MeshConvertSlots.header_init` — was `(self, widget) -> None`
- `file_utils/mesh_convert/slots.py::MeshConvertSlots.source_dir` — was `(self, value: str) -> None`
- `file_utils/mesh_convert/slots.py::MeshConvertSlots.tb000` — was `(self, widget) -> None`
- `file_utils/mesh_convert/slots.py::MeshConvertSlots.tb000_init` — was `(self, widget) -> None`
- `file_utils/mesh_convert/slots.py::MeshConvertUi` — was `(class)`
- `img_utils/map_converter.py::MapConverterSlots` — was `(class)`
- `img_utils/map_converter.py::MapConverterSlots.b000` — was `(self)`
- `img_utils/map_converter.py::MapConverterSlots.b001` — was `(self)`
- `img_utils/map_converter.py::MapConverterSlots.b004` — was `(self)`
- `img_utils/map_converter.py::MapConverterSlots.b005` — was `(self)`
- `img_utils/map_converter.py::MapConverterSlots.b006` — was `(self)`
- `img_utils/map_converter.py::MapConverterSlots.b007` — was `(self)`
- `img_utils/map_converter.py::MapConverterSlots.b008` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b008_init` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b009` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b009_init` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b010` — was `(self)`
- `img_utils/map_converter.py::MapConverterSlots.b011` — was `(self)`
- `img_utils/map_converter.py::MapConverterSlots.b012` — was `(self)`
- `img_utils/map_converter.py::MapConverterSlots.b013` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b013_init` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b014` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b014_init` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b015` — was `(self)`
- `img_utils/map_converter.py::MapConverterSlots.b016` — was `(self)`
- `img_utils/map_converter.py::MapConverterSlots.header_init` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.source_dir` — was `(self, value)`
- `img_utils/map_converter.py::MapConverterSlots.tb000` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.tb000_init` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.tb001` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.tb001_init` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.tb003` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.tb003_init` — was `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.texture_provider` — was `(self, fn)`
- `img_utils/map_converter.py::MapConverterUi` — was `(class)`
- `img_utils/map_packer.py::MapPackerSlots` — was `(class)`
- `img_utils/map_packer.py::MapPackerSlots.b000` — was `(self)`
- `img_utils/map_packer.py::MapPackerSlots.b001` — was `(self)`
- `img_utils/map_packer.py::MapPackerSlots.header_init` — was `(self, widget)`
- `img_utils/map_packer.py::MapPackerSlots.source_dir` — was `(self, value)`
- `img_utils/map_packer.py::MapPackerUi` — was `(class)`

## Added (13)

- `img_utils/map_compositor.py::BatchResult(class)`
- `img_utils/map_compositor.py::MapCompositor(class)`
- `img_utils/map_compositor.py::MapCompositor.apply_output_template(self, output_dir: str) -> List[str]`
- `img_utils/map_compositor.py::MapCompositor.composite_images(self, sorted_images: SortedImages, output_dir: str, name: str = '') -> SortedImages`
- `img_utils/map_compositor.py::MapCompositor.process_batch(self, sorted_images: SortedImages, output_dir: str, name: str = '') -> BatchResult`
- `img_utils/map_compositor.py::MapCompositor.removeNormalMap(self, value: bool) -> None`
- `img_utils/map_compositor.py::MapCompositor.reset(self) -> None`
- `img_utils/map_compositor.py::MapCompositor.retry_failed(self, failed: SortedImages, name: str) -> SortedImages`
- `img_utils/map_compositor.py::NormalOutputMode(class)`
- `vid_utils/frame_extractor.py::FrameExtractor(class)`
- `vid_utils/frame_extractor.py::FrameExtractor.extract_frames(self, video_path: str, output_folder: str, step: int = 5, quality: int = 95, prefix: str = 'frame', max_frames: Optional[int] = None) -> List[str]`
- `vid_utils/frame_extractor.py::FrameExtractor.get_video_info(self, video_path: str) -> dict`
- `vid_utils/frame_extractor.py::extract_frames(video_path: str, output_folder: str, step: int = 5) -> List[str]`
