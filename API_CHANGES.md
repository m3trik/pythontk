# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-05-19._

## Added (35)

- `img_utils/map_converter.py::MapConverterSlots.b008_init(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b009_init(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b013(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b013_init(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b014(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b014_init(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b015(self)`
- `img_utils/map_converter.py::MapConverterSlots.b016(self)`
- `img_utils/map_factory.py::MRAOMapHandler(class)`
- `img_utils/map_factory.py::MRAOMapHandler.can_handle(self, context: TextureProcessor) -> bool`
- `img_utils/map_factory.py::MRAOMapHandler.get_consumed_types(self) -> List[str]`
- `img_utils/map_factory.py::MRAOMapHandler.process(self, context: TextureProcessor) -> Optional[str]`
- `img_utils/map_factory.py::MapFactory.pack_mrao_texture(cls, metallic_map_path: Optional[str], roughness_map_path: Optional[str], ao_map_path: Optional[str], detail_map_path: Optional[str] = None, output_dir: str = None, suffix: str = '_MRAO', invert_roughness: bool = False, output_path: str = None, save: bool = True, layout: str = 'rgb') -> Union[str, 'Image.Image']`
- `img_utils/map_factory.py::MapFactory.pack_orm_texture(cls, ao_map_path: Optional[str], roughness_map_path: Optional[str], metallic_map_path: Optional[str], output_dir: str = None, suffix: str = '_ORM', invert_roughness: bool = False, output_path: str = None, save: bool = True) -> Union[str, 'Image.Image']`
- `img_utils/map_factory.py::MapFactory.unpack_mrao_texture(cls, mrao_map_path: str, output_dir: str = None, metallic_suffix: str = '_Metallic', roughness_suffix: str = '_Roughness', ao_suffix: str = '_AO', invert_roughness: bool = False, save: bool = True, layout: Optional[str] = None, **kwargs) -> Union[Tuple[str, str, str], Tuple['Image.Image', 'Image.Image', 'Image.Image']]`
- `img_utils/map_factory.py::TextureProcessor.create_mrao_map(self, inventory: Dict[str, Union[str, 'Image.Image']]) -> 'Image.Image'`
- `img_utils/map_factory.py::TextureProcessor.get_ao_from_mrao(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
- `img_utils/map_factory.py::TextureProcessor.get_metallic_from_mrao(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
- `img_utils/map_factory.py::TextureProcessor.get_roughness_from_mrao(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
- `img_utils/map_factory.py::TextureProcessor.get_smoothness_from_mrao(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
- `img_utils/map_factory.py::TextureProcessor.unpack_mrao(self, source_path: Union[str, 'Image.Image']) -> None`
- `net_utils/rpc/client.py::RpcClient(class)`
- `net_utils/rpc/client.py::RpcClient.connect(self, exe: Optional[str] = None, timeout: float = 30.0, force_new: bool = False, poll_interval: float = 0.5, auto_cleanup: bool = False) -> bool`
- `net_utils/rpc/client.py::RpcClient.describe(self, op: str = '', timeout: float = 5.0) -> Any`
- `net_utils/rpc/client.py::RpcClient.invoke(self, op: str, timeout: float = 60.0, **kwargs: Any) -> Any`
- `net_utils/rpc/client.py::RpcClient.list_ops(self) -> list`
- `net_utils/rpc/client.py::RpcClient.ping(self, timeout: float = 1.0) -> bool`
- `net_utils/rpc/client.py::RpcClient.shutdown(self, force: bool = False) -> None`
- `net_utils/rpc/client.py::RpcClient.url(self) -> str`
- `net_utils/rpc/installer.py::install_plugin(plugin_src: Union[str, Path], dest: Union[str, Path], force: bool = False) -> Optional[Path]`
- `net_utils/rpc/installer.py::is_plugin_installed(dest: Union[str, Path]) -> bool`
- `net_utils/rpc/installer.py::uninstall_plugin(dest: Union[str, Path]) -> bool`
- `net_utils/rpc/job.py::Call(class)`
- `net_utils/rpc/job.py::Result(class)`
- `net_utils/rpc/job.py::run_batch(calls: List[Call], client: RpcClient, stop_on_error: bool = False) -> List[Result]`

## Signature changed (4)

- `img_utils/map_converter.py::MapConverterSlots.b008`
  - was: `(self)`
  - now: `(self, widget)`
- `img_utils/map_converter.py::MapConverterSlots.b009`
  - was: `(self)`
  - now: `(self, widget)`
- `img_utils/map_factory.py::MapFactory.pack_msao_texture`
  - was: `(cls, metallic_map_path: str, ao_map_path: Optional[str], alpha_map_path: Optional[str], detail_map_path: Optional[str] = None, output_dir: str = None, suffix: str = '_MSAO', invert_alpha: bool = False, output_path: str = None, save: bool = True) -> Union[str, 'Image.Image']`
  - now: `(cls, metallic_map_path: str, ao_map_path: Optional[str], alpha_map_path: Optional[str], detail_map_path: Optional[str] = None, output_dir: str = None, suffix: str = '_MSAO', invert_alpha: bool = False, output_path: str = None, save: bool = True, layout: str = 'rgba') -> Union[str, 'Image.Image']`
- `img_utils/map_factory.py::MapFactory.unpack_msao_texture`
  - was: `(cls, msao_map_path: str, output_dir: str = None, metallic_suffix: str = '_Metallic', ao_suffix: str = '_AO', smoothness_suffix: str = '_Smoothness', invert_smoothness: bool = False, save: bool = True, **kwargs) -> Union[Tuple[str, str, str], Tuple['Image.Image', 'Image.Image', 'Image.Image']]`
  - now: `(cls, msao_map_path: str, output_dir: str = None, metallic_suffix: str = '_Metallic', ao_suffix: str = '_AO', smoothness_suffix: str = '_Smoothness', invert_smoothness: bool = False, save: bool = True, layout: Optional[str] = None, **kwargs) -> Union[Tuple[str, str, str], Tuple['Image.Image', 'Image.Image', 'Image.Image']]`
