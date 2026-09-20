# pythontk — API Changes

_Diff vs the last release (origin/main @ 8881740)._

## Added (46)

- `core_utils/engines/key_stash/key_stash_model.py::KeyStash.flush_pending(cls) -> None`
- `core_utils/engines/key_stash/key_stash_model.py::KeyStash.merge_record(cls, own: Optional[Dict[str, Any]], other: Optional[Dict[str, Any]], ctx: Any = None) -> Optional[Dict[str, Any]]`
- `core_utils/engines/key_stash/key_stash_model.py::KeyStash.respell_record(cls, data: Optional[Dict[str, Any]], ctx: Any) -> Any`
- `core_utils/engines/shots/shot_model.py::ShotStore.flush_pending(cls) -> None`
- `core_utils/engines/shots/shot_transfer.py::ShotTransfer.merge_record(cls, own: Optional[Dict[str, Any]], other: Optional[Dict[str, Any]], ctx: Any = None) -> Optional[Dict[str, Any]]`
- `core_utils/engines/shots/shot_transfer.py::ShotTransfer.respell_record(cls, state: Optional[Dict[str, Any]], ctx: Any) -> Any`
- `core_utils/engines/shots/shot_transfer.py::ShotTransfer.section_in(cls, section: Dict[str, Any], ctx: Any) -> Dict[str, Any]`
- `core_utils/engines/shots/shot_transfer.py::ShotTransfer.section_out(cls, state: Dict[str, Any], ctx: Any) -> Optional[Dict[str, Any]]`
- `core_utils/engines/textures/map_optimizer.py::MapOptimizer.resolve_uastc_rdo(cls, uastc_rdo: Optional[float], map_type_key: Optional[str], output_type: Optional[str], compression: Optional[str]) -> Tuple[Optional[float], Optional[str]]`
- `core_utils/engines/textures/region_masks.py::RegionGroupRegistry.merge_record(cls, own: Optional[dict], other: Optional[dict], ctx: Any = None) -> Optional[dict]`
- `core_utils/export_profile.py::ExportRun.clip_mode(value: Any) -> str`
- `core_utils/scene_records.py::Merge(class)`
- `core_utils/scene_records.py::RecordTransfer(class)`
- `core_utils/scene_records.py::RecordTransfer.apply(self, store, ctx: Optional[TransferContext] = None) -> TransferContext`
- `core_utils/scene_records.py::RecordTransfer.between(cls, store, other: Mapping[Any, Mapping[str, Any]]) -> 'RecordTransfer'`
- `core_utils/scene_records.py::RecordTransfer.incoming(self) -> List[Tuple[Scope, str]]`
- `core_utils/scene_records.py::RecordTransfer.is_empty(self) -> bool`
- `core_utils/scene_records.py::RecordTransfer.merge_record(cls, store, spec: RecordSpec, other: Any, ctx: TransferContext, respelled: bool = False) -> Any`
- `core_utils/scene_records.py::RecordTransfer.payloads(self, ctx: Optional[TransferContext] = None) -> Dict[RecordSpec, Any]`
- `core_utils/scene_records.py::RecordTransfer.receive(cls, manifest: Mapping[str, Any], store, ctx: TransferContext, owners: Optional[Mapping[str, Any]] = None) -> TransferContext`
- `core_utils/scene_records.py::RecordTransfer.rederive(self) -> List[RecordSpec]`
- `core_utils/scene_records.py::RecordTransfer.respell(spec: RecordSpec, payload: Any, ctx: TransferContext) -> Any`
- `core_utils/scene_records.py::RecordTransfer.sections(cls, store, ctx: TransferContext, owners: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]`
- `core_utils/scene_records.py::RecordTransfer.summary(self) -> List[str]`
- `core_utils/scene_records.py::RecordTransfer.union(own: Any, other: Any, spec: RecordSpec, ctx: TransferContext) -> Any`
- `core_utils/scene_records.py::SceneRecords.codec(cls, spec: RecordSpec) -> Optional[Any]`
- `core_utils/scene_records.py::SceneRecords.portable(cls) -> List[RecordSpec]`
- `core_utils/scene_records.py::SceneRecords.resolve_class(module: str, name: str) -> Any`
- `core_utils/scene_records.py::SceneStoreBase.discard_carriers(cls, carriers: Mapping[Any, Any], rename=None, source: str = '', adapters: Optional[Mapping[str, Any]] = None) -> 'TransferContext'`
- `core_utils/scene_records.py::SceneStoreBase.flush_owners(cls) -> None`
- `core_utils/scene_records.py::SceneStoreBase.merge_carriers(cls, carriers: Mapping[Any, Any], rename=None, source: str = '', adapters: Optional[Mapping[str, Any]] = None) -> 'TransferContext'`
- `core_utils/scene_records.py::SceneStoreBase.merge_plan(cls, carriers: Mapping[Any, Any]) -> 'RecordTransfer'`
- `core_utils/scene_records.py::SceneStoreBase.owners(cls) -> Dict[str, Any]`
- `core_utils/scene_records.py::SceneStoreBase.receive_sections(cls, manifest: Optional[Mapping[str, Any]], resolve: Optional[Callable[[str], Optional[str]]] = None, source: str = '', **adapters: Any) -> 'TransferContext'`
- `core_utils/scene_records.py::SceneStoreBase.transfer_sections(cls, spell: Optional[Callable[[str], str]] = None, objects=None) -> Dict[str, Any]`
- `core_utils/scene_records.py::TransferContext(class)`
- `core_utils/scene_records.py::TransferContext.adapter(self, name: str, default: Any = None) -> Any`
- `core_utils/scene_records.py::TransferContext.note(self, text: str) -> None`
- `core_utils/scene_records.py::TransferContext.respell(self, value: Any) -> Any`
- `core_utils/scene_records.py::TransferContext.spell(self, name: str) -> str`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.UASTC_RDO_NORMAL_MAX(cls) -> float`
- `file_utils/mesh_convert/fbx_media.py::FbxMedia.drop_apparatus(cls, src: str, dst: Optional[str] = None, *, section: Mapping[str, str], separator: str = '|') -> Dict[str, Any]`
- `img_utils/_img_utils.py::ImgUtils.settle_ktx2_encoder(cls, prompt: Union[bool, Callable[[str], bool]], refused: Callable[[str], Any], installed: Optional[Callable[[str], Any]] = None) -> bool`
- `img_utils/ktx2_encoder.py::Ktx2Encoder.rdo_dictionary(cls, value: Optional[int]) -> Optional[int]`
- `img_utils/ktx2_encoder.py::Ktx2Encoder.rdo_for(cls, uastc_rdo: Optional[float], normal_map: bool = False) -> Optional[float]`
- `img_utils/ktx2_encoder.py::Ktx2Encoder.rdo_kwargs(uastc_rdo: Optional[float] = None, uastc_rdo_dictionary: Optional[int] = None) -> Dict[str, Union[float, int]]`

## Signature changed (8)

- `core_utils/engines/shots/shot_model.py::ShotStore.snap`
  - was: `(self, frame: float) -> float`
  - now: `(self, frame: float, direction: str = 'nearest') -> float`
- `core_utils/engines/shots/shot_transfer.py::ShotTransfer.merge`
  - was: `(cls, existing: Optional[Dict[str, Any]], incoming: Dict[str, Any]) -> Dict[str, Any]`
  - now: `(cls, existing: Optional[Dict[str, Any]], incoming: Dict[str, Any], id_map: Optional[Dict[int, int]] = None) -> Dict[str, Any]`
- `core_utils/engines/textures/map_optimizer.py::MapOptimizer.assess`
  - was: `(cls, texture_path: str, max_size: int = None, force_pot: Optional[bool] = None, optimize_bit_depth: bool = True, map_type: str = None, allow_palette: bool = False, image: 'Image.Image' = None, output_type: str = None, output_profile: str = None, predict_size: bool = False, enforce_budget: bool = False, lossy_quality: int = None, pot_mode: Optional[str] = None) -> Dict[str, Any]`
  - now: `(cls, texture_path: str, max_size: int = None, force_pot: Optional[bool] = None, optimize_bit_depth: bool = True, map_type: str = None, allow_palette: bool = False, image: 'Image.Image' = None, output_type: str = None, output_profile: str = None, predict_size: bool = False, enforce_budget: bool = False, lossy_quality: int = None, pot_mode: Optional[str] = None, uastc_rdo: Optional[float] = None, uastc_rdo_dictionary: Optional[int] = None) -> Dict[str, Any]`
- `core_utils/engines/textures/map_optimizer.py::MapOptimizer.optimize_map`
  - was: `(cls, texture_path: str, output_dir: str = None, output_type: str = None, max_size: int = None, force_pot: Optional[bool] = None, suffix_old: str = None, suffix_opt: str = None, old_files_folder: str = None, optimize_bit_depth: bool = True, check_existing: bool = False, map_type: str = None, allow_palette: bool = False, output_profile: str = None, enforce_budget: bool = False, lossy_quality: int = None, pot_mode: Optional[str] = None) -> str`
  - now: `(cls, texture_path: str, output_dir: str = None, output_type: str = None, max_size: int = None, force_pot: Optional[bool] = None, suffix_old: str = None, suffix_opt: str = None, old_files_folder: str = None, optimize_bit_depth: bool = True, check_existing: bool = False, map_type: str = None, allow_palette: bool = False, output_profile: str = None, enforce_budget: bool = False, lossy_quality: int = None, pot_mode: Optional[str] = None, uastc_rdo: Optional[float] = None, uastc_rdo_dictionary: Optional[int] = None) -> str`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.describe_texture_pass`
  - was: `(cls, summary: Dict[str, Any], image_format: str, max_size: int = 0, secondary_max_size: int = 0, uastc_rdo: Optional[float] = None) -> str`
  - now: `(cls, summary: Dict[str, Any], image_format: str, max_size: int = 0, secondary_max_size: int = 0, uastc_rdo: Optional[float] = None, uastc_rdo_dictionary: Optional[int] = None) -> str`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.optimize_glb_textures`
  - was: `(cls, glb: GlbTarget, max_size: int = WEB_DELIVERY_MAX_SIZE, image_format: str = WEB_DELIVERY_FORMAT, quality: int = 85, workers: Optional[int] = None, ktx2_fallback: bool = WEB_DELIVERY_KTX2_FALLBACK, secondary_max_size: int = WEB_DELIVERY_SECONDARY_MAX_SIZE, uastc_rdo: Optional[float] = WEB_DELIVERY_UASTC_RDO) -> Dict[str, Any]`
  - now: `(cls, glb: GlbTarget, max_size: int = WEB_DELIVERY_MAX_SIZE, image_format: str = WEB_DELIVERY_FORMAT, quality: int = 85, workers: Optional[int] = None, ktx2_fallback: bool = WEB_DELIVERY_KTX2_FALLBACK, secondary_max_size: int = WEB_DELIVERY_SECONDARY_MAX_SIZE, uastc_rdo: Optional[float] = WEB_DELIVERY_UASTC_RDO, uastc_rdo_dictionary: Optional[int] = WEB_DELIVERY_UASTC_RDO_DICTIONARY) -> Dict[str, Any]`
- `file_utils/mesh_convert/_mesh_convert.py::MeshConvert.web_delivery_texture_params`
  - was: `(cls, image_format: Optional[str] = None, max_size: Optional[int] = None, ktx2_fallback: Optional[bool] = None, secondary_max_size: Optional[int] = None, uastc_rdo: Optional[float] = None) -> Dict[str, Any]`
  - now: `(cls, image_format: Optional[str] = None, max_size: Optional[int] = None, ktx2_fallback: Optional[bool] = None, secondary_max_size: Optional[int] = None, uastc_rdo: Optional[float] = None, uastc_rdo_dictionary: Optional[int] = None) -> Dict[str, Any]`
- `img_utils/_img_utils.py::ImgUtils.save_image`
  - was: `(cls, image: Union[str, Image.Image], name: str, mode: str = None, bit_depth: int = None, compression: str = None, quality: int = None, colorspace: str = None, **kwargs)`
  - now: `(cls, image: Union[str, Image.Image], name: str, mode: str = None, bit_depth: int = None, compression: str = None, quality: int = None, colorspace: str = None, uastc_rdo: float = None, uastc_rdo_dictionary: int = None, **kwargs)`
