# pythontk — API Index

_Auto-generated. Do not edit by hand. Compact symbol index — grep this for a name; for full signatures/docs, slice [API_REGISTRY.md](API_REGISTRY.md) (never Read it whole)._

_Generated: 2026-06-27_

### `audio_utils/_audio_utils.py`
- `class AudioUtils(HelpMixin)`
  - methods: resolve_ffmpeg, is_playable_extension, is_supported_source_extension, ensure_playable_path, build_composite_wav, resolve_playable_path, build_audio_map, build_audio_map_from_file_map, build_audio_map_from_files, trim_silence, compute_waveform_envelope

### `color_utils/_color_utils.py` — Lightweight, DCC-agnostic color primitives.
- `class Color`
  - methods: from_hex, from_rgbf, hex, rgb, rgba, rgbf, rgbaf, luminance, lighter, darker, with_alpha, blend, subtle_bg
- `class ColorPair`
  - methods: auto
- `class Palette(dict)`
  - methods: alias, override, status, axes, channels, ui, diff

### `core_utils/_core_utils.py`
- `class CoreUtils(HelpMixin)`
  - methods: cached_property, listify, format_return, set_attributes, get_attributes, has_attribute, get_derived_type, cycle, are_similar, randomize, parse_method_args

### `core_utils/app_handoff.py` — Generic, Qt-free / DCC-free engine for "export something and hand it to an app".
- `class AppSpec`
  - methods: resolve, not_found_message
- `class HandoffRequest`
  - methods: get
- `class Payload`
- `class Deliverer`
  - methods: preflight, deliver
- `class HandoffBridge(LoggingMixin)`
  - methods: app_path, app_path, params_defaults, merge_params, send
- `class ScriptLaunchSpec`
- `class ScriptLaunchDeliverer(Deliverer)`
  - methods: preflight, deliver, render
- `class ScriptLaunchBridge(HandoffBridge)`
  - methods: render_context, render_template, list_template_modes, list_templates

### `core_utils/app_installer.py`
- `class AppInstaller`
  - methods: ensure, get_path

### `core_utils/app_launcher.py`
- `class AppLauncher`
  - methods: launch, run, current_session_id, active_console_session_id, is_interactive_session, find_session_launcher, launch_in_session, wait_for_ready, get_window_titles, append_to_path, scan_for_executables, is_path_persisted, scan_install_dirs, resolve_app_path, find_app, get_running_processes, close_process

### `core_utils/class_property.py`
- `class ClassProperty`

### `core_utils/cli.py`
- `class CLI`
  - methods: get_parser, add_connection_args, get_connection_kwargs

### `core_utils/execution_monitor/_dialog_viewer.py` — Subprocess-based dialog viewer for custom button labels.
- `run(title: str, message: str, force_label: str | None = None)`

### `core_utils/execution_monitor/_execution_monitor.py`
- `class ExecutionMonitor`
  - methods: is_escape_pressed, set_interpreter, on_long_execution, show_long_execution_dialog, execution_monitor, external_watchdog

### `core_utils/execution_monitor/_gif_viewer.py`
- `run(gif_path, target_size=DEFAULT_SIZE, pos=None)`

### `core_utils/execution_monitor/_spinner.py` — Lightweight canvas-based spinner for task-indicator overlay.
- `run(size=DEFAULT_SIZE, pos=None)`

### `core_utils/git.py`
- `class Git`
  - methods: execute, run, checkout, pull, push, merge, fetch, status, current_branch

### `core_utils/help_mixin.py` — HelpMixin - Enhanced help system leveraging Python's built-in help infrastructure.
- `class HelpMixin`
  - methods: help, source, where, show_mro, signature, classify, list_members, about

### `core_utils/hierarchy_utils/hierarchy_analyzer.py`
- `class DifferenceType(Enum)`
- `class HierarchyDifference`
- `class HierarchyAnalyzer`
  - methods: compare_path_sets, analyze_hierarchy_differences, detect_moved_items, categorize_differences, generate_diff_report, export_differences_to_dict, filter_differences

### `core_utils/hierarchy_utils/hierarchy_diff.py`
- `class HierarchyDiff`
  - methods: is_valid, has_differences, total_issues, as_dict, as_json, save_to_file, load_from_file, clear, merge, get_summary, filter_by_pattern, add_metadata

### `core_utils/hierarchy_utils/hierarchy_indexer.py`
- `class HierarchyIndexer`
  - methods: build_path_index, find_by_path, find_by_tail_path, get_path_components_index, get_depth_index

### `core_utils/hierarchy_utils/hierarchy_matching.py`
- `class HierarchyMatching`
  - methods: exact_path_match, tail_path_match, fuzzy_name_match, multi_strategy_match

### `core_utils/logging_mixin.py`
- `class StripHtmlFormatter(internal_logging.Formatter)`
  - methods: format
- `class LevelAwareFormatter(internal_logging.Formatter)`
  - methods: format
- `class LoggerExt`
  - methods: patch, get_color, register_html_preset, get_html_preset, format_message_as_html
- `class DefaultTextLogHandler(internal_logging.Handler)`
  - methods: emit, get_color
- `class RingBufferHandler(internal_logging.Handler)`
  - methods: emit, clear, format_records
- `class TableMixin`
  - methods: format_table, log_table
- `class LoggingMixin(TableMixin)`
  - methods: logger, class_logger, logging, set_log_level, set_log_file, enable_log_buffer, disable_log_buffer, clear_log_buffer, dump_log

### `core_utils/module_reloader.py` — Helpers for hot-reloading packages and their submodules.
- `reload_package(package: ModuleRef, **kwargs) -> List[ModuleType]`
- `class ModuleReloader`
  - methods: reload

### `core_utils/module_resolver.py` — Reusable module attribute resolver for package-style imports.
- `bootstrap_package(module_globals: MutableMapping[str, Any], *, include: Optional[IncludeMapping] = None, module_to_parent: Optional[Mapping[str, str]] = None, eager: bool = False, allow_getattr: bool = True, install_legacy_helpers: bool = True, on_import_error: Optional[Callable[[str, Exception], None]] = None, method_predicate: Optional[Callable[[str], bool]] = None, custom_getattr: Optional[Callable[[str], Any]] = None, lazy_import: Optional[bool] = None) -> PackageResolverHandle`
- `create_namespace_aliases(module_globals: MutableMapping[str, Any], aliases: Mapping[str, Union[str, Sequence[str]]], *, include_spec: Optional[IncludeMapping] = None) -> None`
- `class ModuleAttributeResolver`
  - methods: build, rebuild, resolve, get_module, bind_to, iter_registered_names, clear_module_cache
- `class PackageResolverHandle`
  - methods: install, configure, build_dictionaries, import_module, get_attribute_from_module, export_all

### `core_utils/namedtuple_container.py`
- `class NamedTupleContainer(LoggingMixin)`
  - methods: extend, get, filter, map, modify, remove, clear, to_dict_list, to_csv, from_csv

### `core_utils/namespace_handler.py`
- `class Placeholder`
  - methods: info, create
- `class NamespaceHandler(LoggingMixin)`
  - methods: placeholders, is_placeholder, get_placeholder, set_placeholder, resolve_all_placeholders, has_placeholder, keys, items, values, setdefault, has, peek, raw, resolve, is_resolving

### `core_utils/package_manager.py`
- `class PackageManager(_PkgVersionCheck, _PkgVersionUtils, _PackageManagerHelperMixin, help_mixin.HelpMixin)`
  - methods: pip, get_local_dependency_order

### `core_utils/preset_store.py` — Qt-free, zero-dependency named-preset *store* for the ecosystem.
- `sanitize_preset_name(name: str) -> str`
- `class PresetStore`
  - methods: user_dir, builtin_dir, active, active, list, source, exists, path, load, save, delete, rename

### `core_utils/qc_log.py` — Structured run logs and threshold-based acceptance gates for pipeline
- `class GateError(RuntimeError)`
- `class QcLog`
  - methods: stage, warn, set, finalize
- `class QcGate`
  - methods: check

### `core_utils/script_template.py` — Generic on-disk script-template discovery + ``__KEY__`` rendering.
- `list_templates(template_dir, extension: str = '.py') -> List[Path]`
- `template_modes(template_path, allowed: Sequence[str] = (SEND_TO,), field: str = 'BRIDGE_MODES') -> Tuple[str, ...]`
- `list_template_modes(template_dir, extension: str = '.py', allowed: Sequence[str] = (SEND_TO,), field: str = 'BRIDGE_MODES') -> List[Tuple[str, str]]`
- `render_template(template_path, context: Dict[str, str]) -> str`

### `core_utils/singleton_mixin.py`
- `class SingletonMixin`
  - methods: instance, has_instance, reset_instance

### `core_utils/user_config.py` — Qt-free, zero-dependency user-config resolution for the ecosystem.
- `user_config_root() -> Path`
- `class UserConfig`
  - methods: path_for, load_file, resolve, deep_merge, expand

### `file_utils/_file_utils.py`
- `class FileUtils(HelpMixin)`
  - methods: is_valid, is_cloud_placeholder, free_space, create_dir, next_version_path, get_dir_contents, open_explorer, get_file_contents, write_to_file, atomic_write_text, copy_file, move_file, reveal_in_file_manager, get_file_info, format_path, convert_to_relative_path, remap_file_paths, append_path, get_object_path, get_classes_from_path, set_json_file, get_json_file, set_json, get_json

### `file_utils/mesh_cleaner.py` — Mesh repair / cleanup via PyMeshLab (optional dependency).
- `class MeshCleaner`
  - methods: is_available, clean

### `file_utils/mesh_convert/_mesh_convert.py`
- `class MeshConvert(HelpMixin)`
  - methods: resolve_binary, fbx_to_glb, check_glb_materials, fix_glb_phantom_opaque_alpha

### `file_utils/metadata.py`
- `class MetadataInternal`
- `class Metadata(MetadataInternal)`
  - methods: get, set

### `geo_utils/drape.py` — Procedural draped-cloth (curtain) generator — pure geometry, no DCC.
- `class CurtainDrape(LoggingMixin)`
  - methods: prepare, grid_points, drape

### `geo_utils/pointcloud.py` — Point-cloud geometry — analyze and group unordered sets of points.
- `class PointCloud`
  - methods: pca_transform, cluster_by_distance, hash_points

### `geo_utils/polyline.py` — Pure polyline / curve geometry — generate, measure, sample, reshape.
- `class Polyline`
  - methods: make, from_point_cloud, order_points, length, point_at, resample, smooth, simplify, frames

### `img_utils/_img_utils.py`
- `class ImgUtils(HelpMixin)`
  - methods: im_help, allow_large_images, ensure_image, enforce_mode, assert_pathlike, validate_image_integrity, create_image, register_dds_codec, save_image, load_image, get_images, get_image_size, get_image_info, are_identical, resize_image, ensure_pot, format_bit_depth, set_bit_depth, invert_grayscale_image, invert_channels, swizzle_channels, create_mask, fill_masked_area, fill, get_background, replace_color, set_contrast, gaussian_blur, dilate_image, compute_atlas_layout, assemble_atlas, radial_gradient, rasterize_silhouette, convert_rgb_to_gray, convert_rgb_to_hsv, convert_i_to_l, pack_channels, pack_channel_into_alpha, srgb_to_linear, linear_to_srgb, generate_mipmaps, depalettize_image, is_image_constant, get_base_texture_name, extract_channels

### `img_utils/exposure_equalizer.py` — Cross-set exposure / white-balance equalization.
- `class ExposureEqualizer`
  - methods: is_available, equalize_directories

### `img_utils/image_curator.py` — Perceptual-hash + sharpness curation for large image sets.
- `class ImageCurator`
  - methods: is_available, dhash, hamming, sharpness, curate, preview

### `img_utils/map_compositor.py` — Pure image-compositing engine — alpha-composite layered texture maps
- `class BatchResult(Enum)`
- `class NormalOutputMode(Enum)`
- `class MapCompositor(ptk.LoggingMixin)`
  - methods: removeNormalMap, removeNormalMap, reset, process_batch, apply_output_template, composite_images, retry_failed

### `img_utils/map_factory/_map_factory.py` — ``MapFactory`` -- the texture-map workflow orchestrator.
- `class MapFactory(LoggingMixin)`
  - methods: register_conversions, resolve_map_type, resolve_color_space, resolve_texture_filename, get_base_texture_name, group_textures_by_set, filter_images_by_type, sort_images_by_type, contains_map_types, is_normal_map, register_handler, register_conversion, get_map_fallbacks, get_precedence_rules, filter_redundant_maps, prepare_maps, pack_transparency_into_albedo, pack_smoothness_into_metallic, detect_normal_map_format, convert_normal_map_format, convert_bump_to_normal, extract_gloss_from_spec, convert_spec_gloss_to_pbr, create_base_color_from_spec, create_metallic_from_spec, create_roughness_from_spec, convert_base_color_to_albedo, get_converted_map, pack_orm_texture, pack_msao_texture, pack_mrao_texture, convert_smoothness_to_roughness, convert_roughness_to_smoothness, unpack_orm_texture, unpack_msao_texture, unpack_mrao_texture, unpack_albedo_transparency, unpack_metallic_smoothness, unpack_specular_gloss

### `img_utils/map_factory/conversions.py` — Map-conversion registry primitives for the texture MapFactory.
- `class MapConversion`
- `class ConversionRegistry`
  - methods: add_plugin, register, register_from_class, get_conversions_for

### `img_utils/map_factory/handlers.py` — Workflow handlers (Strategy pattern) for the texture MapFactory.
- `class WorkflowHandler(ABC)`
  - methods: can_handle, process, get_consumed_types, is_explicitly_requested
- `class ORMMapHandler(WorkflowHandler)`
  - methods: can_handle, process, get_consumed_types
- `class MRAOMapHandler(WorkflowHandler)`
  - methods: can_handle, process, get_consumed_types
- `class MaskMapHandler(WorkflowHandler)`
  - methods: can_handle, process, get_consumed_types
- `class MetallicSmoothnessHandler(WorkflowHandler)`
  - methods: can_handle, process, get_consumed_types
- `class SeparateMetallicRoughnessHandler(WorkflowHandler)`
  - methods: can_handle, process, get_consumed_types
- `class BaseColorHandler(WorkflowHandler)`
  - methods: can_handle, process, get_consumed_types
- `class NormalMapHandler(WorkflowHandler)`
  - methods: can_handle, process, get_consumed_types
- `class OutputFallbackHandler(WorkflowHandler)`
  - methods: can_handle, process, get_consumed_types

### `img_utils/map_factory/processor.py` — ``TextureProcessor`` -- shared processing context for the MapFactory.
- `class TextureProcessor`
  - methods: get_cached_image, save_map, resolve_map, mark_used, convert_specular_to_metallic, convert_smoothness_to_roughness, convert_roughness_to_smoothness, convert_specular_to_roughness, convert_dx_to_gl, convert_gl_to_dx, convert_bump_to_normal, extract_gloss_from_spec, copy_map, unpack_metallic_smoothness, get_metallic_from_packed, get_smoothness_from_packed, get_roughness_from_packed, unpack_msao, get_metallic_from_msao, get_smoothness_from_msao, get_roughness_from_msao, get_ao_from_msao, unpack_mrao, get_metallic_from_mrao, get_roughness_from_mrao, get_smoothness_from_mrao, get_ao_from_mrao, unpack_orm, get_ao_from_orm, get_roughness_from_orm, get_smoothness_from_orm, get_metallic_from_orm, unpack_albedo_transparency, get_base_color_from_albedo_transparency, get_opacity_from_albedo_transparency, create_orm_map, create_mrao_map, create_mask_map, create_metallic_smoothness_map

### `img_utils/map_optimizer.py` — Plan, assess, and apply map (texture) optimizations.
- `class Op`
- `class MapOptimizer(HelpMixin)`
  - methods: plan, apply, optimize_map, batch_optimize_maps, assess

### `img_utils/map_registry.py`
- `class WF`
- `class MapType`
- `class MapRegistry(SingletonMixin)`
  - methods: get, resolve_type_from_path, get_workflow_presets, get_map_types, get_fallbacks, get_output_fallbacks, get_precedence_rules, get_scale_as_mask_types, get_resolution_critical_types, is_resolution_critical, get_passthrough_maps, get_map_backgrounds, get_map_modes, resolve_config

### `img_utils/mask_generator.py` — Background mask generation via rembg (optional dependency).
- `class MaskGenerator`
  - methods: is_available, generate_masks

### `img_utils/mat_report.py` — DCC-agnostic formatters for material / texture info reports.
- `class MatReport`
  - methods: format_texture_info_text, format_texture_info_html, format_mat_info_text, format_mat_info_html

### `img_utils/output_template.py` — Per-map output-format templates — the "export preset" layer.
- `class OutputSpec`
  - methods: to_dict, from_dict
- `class OutputTemplate`
  - methods: resolve, to_dict, from_dict
- `class OutputTemplates`
  - methods: get, resolve

### `iter_utils/_iter_utils.py`
- `class IterUtils(HelpMixin)`
  - methods: make_iterable, nested_depth, flatten, collapse_integer_sequence, bit_array_to_list, insert_into_dict, rindex, indices, remove_duplicates, filter_results, filter_list, filter_dict, split_list, find_flat_interior_indices

### `math_utils/_math_utils.py`
- `class MathUtils(HelpMixin)`
  - methods: eval_expression, convert_length_unit, linear_sum_assignment, kmeans_clustering, kmeans_1d, get_kmeans_threshold, move_decimal_point, get_vector_from_two_points, clamp, clamp_range, normalize, get_magnitude, dot_product, cross_product, move_point_relative, move_point_relative_along_vector, distance_between_points, get_center_of_two_points, get_angle_from_two_vectors, get_angle_from_three_points, get_two_sides_of_asa_triangle, xyz_rotation, lerp, safe_normalize, smoothstep, ricker, catenary, catenary_sag, evaluate_sampled_progress, generate_geometric_sequence, remap, point_segment_distance, nearest_power_of_two, is_close_to_whole, round_value, round_to_preferred, round_to_aggressive_preferred, calculate_rotation_distance

### `math_utils/noise.py`
- `class BandLimitedNoise`
  - methods: at

### `math_utils/progression.py`
- `class ProgressionCurves`
  - methods: linear, exponential, logarithmic, sine, ease_in, ease_out, ease_in_out, smooth_step, bounce, elastic, weighted, calculate_progression_factor, get_curve_function, generate_curve_samples

### `net_utils/_net_utils.py`
- `class NetUtils`
  - methods: connect_rdp, is_port_open, get_local_ip

### `net_utils/credentials.py`
- `class Credentials`
  - methods: get_password, get_credential, set_credential

### `net_utils/rpc/client.py` — Generic HTTP JSON-RPC client for plugin-hosted RPC servers.
- `class RpcClient`
  - methods: url, ping, invoke, list_ops, describe, connect, shutdown

### `net_utils/rpc/installer.py` — Generic DCC plugin installer (symlink-first, copytree fallback).
- `install_plugin(plugin_src: Union[str, Path], dest: Union[str, Path], force: bool = False) -> Optional[Path]`
- `uninstall_plugin(dest: Union[str, Path]) -> bool`
- `is_plugin_installed(dest: Union[str, Path]) -> bool`

### `net_utils/rpc/job.py` — One-shot batch pipeline over :class:`RpcClient`.
- `run_batch(calls: List[Call], client: RpcClient, stop_on_error: bool = False) -> List[Result]`
- `class Call`
- `class Result`

### `net_utils/ssh_client.py`
- `class SSHClient`
  - methods: connect, disconnect, execute, upload_file, download_file

### `str_utils/_str_utils.py`
- `class StrUtils(CoreUtils)`
  - methods: sanitize, replace_placeholders, replace_delimited, set_case, get_mangled_name, get_matching_hierarchy_items, split_delimited_string, get_text_between_delimiters, insert, rreplace, truncate, get_trailing_integers, find_str, find_str_and_format, format_suffix, strip_known_affix, infer_affix_mode, split_affix, apply_affix, alpha_sequence, sequential_suffixes, resolve_name_collisions, time_stamp

### `str_utils/fuzzy_matcher.py`
- `class FuzzyMatcher`
  - methods: get_base_name, find_best_match, find_all_matches, find_trailing_digit_matches, find_unique_match, find_with_fallbacks, calculate_levenshtein_distance, similarity_from_distance

### `vid_utils/_vid_utils.py`
- `class VidUtils(HelpMixin)`
  - methods: get_frame_rate, resolve_ffmpeg, get_video_frame_rate, compress_video

### `vid_utils/frame_extractor.py` — Extract still frames from a video file via OpenCV.
- `extract_frames(video_path: str, output_folder: str, step: int = 5) -> List[str]`
- `class FrameExtractor`
  - methods: score_sharpness, extract_frames, extract_frames_sharpest, get_video_info
