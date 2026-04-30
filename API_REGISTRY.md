# pythontk — API Registry

_Auto-generated. Do not edit by hand. Refresh via `m3trik/scripts/generate_api_registry.py`._

_Generated: 2026-04-29_

## Index

- [`audio_utils/_audio_utils.py`](#audio_utils--_audio_utils)
- [`color_utils/_color_utils.py`](#color_utils--_color_utils) — Lightweight, DCC-agnostic color primitives.
- [`core_utils/_core_utils.py`](#core_utils--_core_utils)
- [`core_utils/app_installer.py`](#core_utils--app_installer)
- [`core_utils/app_launcher.py`](#core_utils--app_launcher)
- [`core_utils/class_property.py`](#core_utils--class_property)
- [`core_utils/cli.py`](#core_utils--cli)
- [`core_utils/execution_monitor/_dialog_viewer.py`](#core_utils--execution_monitor--_dialog_viewer) — Subprocess-based dialog viewer for custom button labels.
- [`core_utils/execution_monitor/_execution_monitor.py`](#core_utils--execution_monitor--_execution_monitor)
- [`core_utils/execution_monitor/_gif_viewer.py`](#core_utils--execution_monitor--_gif_viewer)
- [`core_utils/execution_monitor/_spinner.py`](#core_utils--execution_monitor--_spinner) — Lightweight canvas-based spinner for task-indicator overlay.
- [`core_utils/git.py`](#core_utils--git)
- [`core_utils/help_mixin.py`](#core_utils--help_mixin) — HelpMixin - Enhanced help system leveraging Python's built-in help infrastructure.
- [`core_utils/hierarchy_diff.py`](#core_utils--hierarchy_diff)
- [`core_utils/hierarchy_utils/backup/hierarchy_analyzer.py`](#core_utils--hierarchy_utils--backup--hierarchy_analyzer)
- [`core_utils/hierarchy_utils/backup/hierarchy_indexer.py`](#core_utils--hierarchy_utils--backup--hierarchy_indexer)
- [`core_utils/hierarchy_utils/backup/hierarchy_matching.py`](#core_utils--hierarchy_utils--backup--hierarchy_matching)
- [`core_utils/hierarchy_utils/hierarchy_analyzer.py`](#core_utils--hierarchy_utils--hierarchy_analyzer)
- [`core_utils/hierarchy_utils/hierarchy_indexer.py`](#core_utils--hierarchy_utils--hierarchy_indexer)
- [`core_utils/hierarchy_utils/hierarchy_matching.py`](#core_utils--hierarchy_utils--hierarchy_matching)
- [`core_utils/logging_mixin.py`](#core_utils--logging_mixin)
- [`core_utils/module_reloader.py`](#core_utils--module_reloader) — Helpers for hot-reloading packages and their submodules.
- [`core_utils/module_resolver.py`](#core_utils--module_resolver) — Reusable module attribute resolver for package-style imports.
- [`core_utils/namedtuple_container.py`](#core_utils--namedtuple_container)
- [`core_utils/namespace_handler.py`](#core_utils--namespace_handler)
- [`core_utils/package_manager.py`](#core_utils--package_manager)
- [`core_utils/singleton_mixin.py`](#core_utils--singleton_mixin)
- [`file_utils/_file_utils.py`](#file_utils--_file_utils)
- [`file_utils/metadata.py`](#file_utils--metadata)
- [`img_utils/_img_utils.py`](#img_utils--_img_utils)
- [`img_utils/map_converter.py`](#img_utils--map_converter)
- [`img_utils/map_factory.py`](#img_utils--map_factory) — Texture Map Factory for PBR workflow preparation - Refactored.
- [`img_utils/map_packer.py`](#img_utils--map_packer)
- [`img_utils/map_registry.py`](#img_utils--map_registry)
- [`iter_utils/_iter_utils.py`](#iter_utils--_iter_utils)
- [`math_utils/_math_utils.py`](#math_utils--_math_utils)
- [`math_utils/progression.py`](#math_utils--progression)
- [`net_utils/_net_utils.py`](#net_utils--_net_utils)
- [`net_utils/credentials.py`](#net_utils--credentials)
- [`net_utils/ssh_client.py`](#net_utils--ssh_client)
- [`str_utils/_str_utils.py`](#str_utils--_str_utils)
- [`str_utils/fuzzy_matcher.py`](#str_utils--fuzzy_matcher)
- [`vid_utils/_vid_utils.py`](#vid_utils--_vid_utils)

---

<a id="audio_utils--_audio_utils"></a>
### `audio_utils/_audio_utils.py`

- **[`class AudioUtils(HelpMixin)`](pythontk/pythontk/audio_utils/_audio_utils.py#L17)** — Utility helpers for portable audio-file preparation.
  - `AudioUtils.resolve_ffmpeg(cls, required: bool = True, auto_install: bool = False) -> Optional[str]` *(class)* — Resolve ffmpeg executable from PATH or managed installs.
  - `AudioUtils.is_playable_extension(cls, file_path: str) -> bool` *(class)* — Return True if extension is already timeline-playable.
  - `AudioUtils.is_supported_source_extension(cls, file_path: str) -> bool` *(class)* — Return True if extension is accepted as conversion source.
  - `AudioUtils.ensure_playable_path(cls, audio_path: str, cache_dir: Optional[str] = None) -> str` *(class)* — Return a playable audio path, converting with ffmpeg if required.
  - `AudioUtils.build_composite_wav(cls, events: list, audio_map: dict, fps: float, output_path: str, logger=None) -> Optional[str]` *(class)* — Mix source WAV clips into one composite WAV.
  - `AudioUtils.resolve_playable_path(cls, audio_path: str, cache_dir: Optional[str] = None, logger=None) -> Optional[str]` *(class)* — Return a playable path, converting to WAV when required.
  - `AudioUtils.build_audio_map(cls, search_dir: str, extensions: Optional[Set[str]] = None, cache_dir: Optional[str] = None, logger=None) -> Dict[str, str]` *(class)* — Recursively scan a directory for audio files.
  - `AudioUtils.build_audio_map_from_file_map(cls, file_map: Dict[str, str], cache_dir: Optional[str] = None, logger=None) -> Dict[str, str]` *(class)* — Build an audio map from a ``{stem: path}`` dict.
  - `AudioUtils.build_audio_map_from_files(cls, audio_files: List[str], cache_dir: Optional[str] = None, logger=None) -> Dict[str, str]` *(class)* — Build an audio map from an explicit list of file paths.
  - `AudioUtils.trim_silence(cls, wav_path: str, output_path: Optional[str] = None, threshold: int = 8) -> str` *(class)* — Trim leading and trailing silence from a 16-bit PCM WAV file.
  - `AudioUtils.compute_waveform_envelope(wav_path: str, num_bins: int = 512) -> List[tuple]` *(static)* — Read a WAV file and return a downsampled min/max envelope.

<a id="color_utils--_color_utils"></a>
### `color_utils/_color_utils.py`

Lightweight, DCC-agnostic color primitives.

- **[`class Color`](pythontk/pythontk/color_utils/_color_utils.py#L18)** — Immutable RGBA color stored as 0–255 integers.
  - `Color.from_hex(cls, hex_str: str) -> 'Color'` *(class)* — Parse ``#RGB``, ``#RRGGBB``, or ``#RRGGBBAA``.
  - `Color.from_rgbf(cls, r: float, g: float, b: float, a: float = 1.0) -> 'Color'` *(class)* — Create from 0.0–1.0 float components (Maya API convention).
  - `Color.hex(self) -> str` *(property)* — ``'#RRGGBB'`` (or ``'#RRGGBBAA'`` when alpha < 255).
  - `Color.rgb(self) -> Tuple[int, int, int]` *(property)* — ``(r, g, b)`` in 0–255.
  - `Color.rgba(self) -> Tuple[int, int, int, int]` *(property)* — ``(r, g, b, a)`` in 0–255.
  - `Color.rgbf(self) -> Tuple[float, float, float]` *(property)* — ``(r, g, b)`` in 0.0–1.0 (Maya API format).
  - `Color.rgbaf(self) -> Tuple[float, float, float, float]` *(property)* — ``(r, g, b, a)`` in 0.0–1.0.
  - `Color.luminance(self) -> float` *(property)* — Perceived luminance (ITU-R BT.709, linear approximation).
  - `Color.lighter(self, factor: float = 0.2) -> 'Color'` — Return a lighter colour.
  - `Color.darker(self, factor: float = 0.2) -> 'Color'` — Return a darker colour.
  - `Color.with_alpha(self, a: Union[int, float]) -> 'Color'` — Return a copy with a new alpha (int 0–255 or float 0.0–1.0).
  - `Color.blend(self, other: 'Color', t: float = 0.5) -> 'Color'` — Linear interpolation towards *other* by *t* (0.0 = self, 1.0 = other).
  - `Color.subtle_bg(self, value: float = 0.24, sat_factor: float = 1.0) -> 'Color'` — Derive a tinted dark-theme background from this colour.
- **[`class ColorPair`](pythontk/pythontk/color_utils/_color_utils.py#L177)** — Foreground / background pair for themed UIs.
  - `ColorPair.auto(cls, fg: Union[str, 'Color'], value: float = 0.24, sat_factor: float = 1.0) -> 'ColorPair'` *(class)* — Derive background automatically from foreground for dark themes.
- **[`class Palette(dict)`](pythontk/pythontk/color_utils/_color_utils.py#L254)** — Named color collection with auto-wrapping and alias support.
  - `Palette.alias(self, mapping: Dict[str, str]) -> 'Palette'` — Return a new Palette with additional keys pointing to existing values.
  - `Palette.override(self, **kwargs: object) -> 'Palette'` — Return a new Palette with selected entries replaced.
  - `Palette.status(cls) -> 'Palette'` *(class)* — Standard severity palette for dark-theme UIs.
  - `Palette.axes(cls) -> 'Palette'` *(class)* — Standard XYZ / RGB axis colours (Maya / 3D convention).
  - `Palette.channels(cls) -> 'Palette'` *(class)* — Standard transform-attribute colours for animation editors.
  - `Palette.ui(cls) -> 'Palette'` *(class)* — Common UI element colours for dark themes.
  - `Palette.diff(cls) -> 'Palette'` *(class)* — Comparison / diff palette for dark-theme tree views.

<a id="core_utils--_core_utils"></a>
### `core_utils/_core_utils.py`

- **[`class CoreUtils(HelpMixin)`](pythontk/pythontk/core_utils/_core_utils.py#L14)**
  - `CoreUtils.cached_property(func: Callable) -> Any` *(static)* — Decorator that converts a method with a single self argument into a property
  - `CoreUtils.listify(func=None, arg_name=None, threading=False)` *(static)* — A decorator to make a function accept list-like arguments and return a list of results.
  - `CoreUtils.format_return(cls, lst, orig=None)` *(class)* — Return the list element if the given iterable only contains a single element.
  - `CoreUtils.set_attributes(obj, **attributes)` *(static)* — Set attributes for a given object.
  - `CoreUtils.get_attributes(obj, inc=[], exc=[])` *(static)* — Get attributes for a given object.
  - `CoreUtils.has_attribute(cls, attr)` *(static)* — This function checks whether a class has a specific static attribute by using `inspect.getattr_stat…
  - `CoreUtils.get_derived_type(obj, return_name=False, module=None, include=[], exclude=[], filter_by_base_type=False)` *(static)* — Get the base class of a custom object.
  - `CoreUtils.cycle(cls, sequence, name=None, query=False)` *(class)* — Toggle between numbers in a given sequence.
  - `CoreUtils.are_similar(cls, a, b, tolerance=0.0)` *(class)* — Check if the two numberical values are within a given tolerance.
  - `CoreUtils.randomize(lst, ratio=1.0)` *(static)* — Random elements from the given list will be returned with a quantity determined by the given ratio.
  - `CoreUtils.parse_method_args(args: Tuple) -> Tuple[Union[Any, None], Tuple]` — Parse method arguments to determine if the function is an instance method or static method.

<a id="core_utils--app_installer"></a>
### `core_utils/app_installer.py`

- **[`class AppInstaller`](pythontk/pythontk/core_utils/app_installer.py#L40)** — Download, extract, and manage external OS-level tool binaries.
  - `AppInstaller.ensure(cls, name: str, platforms: Dict[str, dict], executable: Optional[str] = None, version: Optional[str] = None, sha256: Optional[Dict[str, str]] = None, location: str = 'user', update: bool = False, add_to_path: bool = True, progress_callback: Optional[Callable[[int, int], None]] = None) -> str` *(class)* — Ensure a tool is available locally, downloading if necessary.
  - `AppInstaller.get_path(cls, name: str, location: str = 'user', executable: Optional[str] = None, add_to_path: bool = False) -> Optional[str]` *(class)* — Find a tool without installing.

<a id="core_utils--app_launcher"></a>
### `core_utils/app_launcher.py`

- **[`class AppLauncher`](pythontk/pythontk/core_utils/app_launcher.py#L13)** — A utility class for launching applications on Windows and Linux.
  - `AppLauncher.launch(app_identifier, args=None, cwd=None, detached=True)` *(static)* — Launches an application.
  - `AppLauncher.run(app_identifier, args=None, cwd=None, timeout=None)` *(static)* — Execute an application synchronously and return its result.
  - `AppLauncher.wait_for_ready(process, timeout=15, check_fn=None)` *(static)* — Waits until the application is ready.
  - `AppLauncher.get_window_titles(pid)` *(static)* — Returns a list of window titles owned by the given PID.
  - `AppLauncher.append_to_path(path, user_scope=True)` *(static)* — Appends a directory to the system PATH.
  - `AppLauncher.scan_for_executables(root_paths, executable_name, depth=3)` *(static)* — Scans directories for a specific executable.
  - `AppLauncher.is_path_persisted(path)` *(static)* — Checks if the path is permanently stored in the system configuration (e.g.
  - `AppLauncher.find_app(app_identifier)` *(static)* — Attempts to locate the executable for the given application identifier.
  - `AppLauncher.get_running_processes(process_name)` *(static)* — Returns a list of PIDs of running processes matching the given name.
  - `AppLauncher.close_process(pid, force=False)` *(static)* — Terminates the process with the given PID.

<a id="core_utils--class_property"></a>
### `core_utils/class_property.py`

- **[`class ClassProperty`](pythontk/pythontk/core_utils/class_property.py#L6)** — A descriptor for class-level properties (replaces @classmethod @property).

<a id="core_utils--cli"></a>
### `core_utils/cli.py`

- **[`class CLI`](pythontk/pythontk/core_utils/cli.py#L10)** — Utilities for standardizing Command Line Interfaces across scripts.
  - `CLI.get_parser(description: str = None) -> argparse.ArgumentParser` *(static)* — Create a standard ArgumentParser.
  - `CLI.add_connection_args(parser: argparse.ArgumentParser, default_host: str = DEFAULT_HOST, default_user: str = DEFAULT_USER, default_target: str = DEFAULT_CRED_TARGET) -> argparse.ArgumentParser` *(static)* — Add standard SSH connection arguments (host, user, password, cred-target).
  - `CLI.get_connection_kwargs(args: argparse.Namespace) -> Dict[str, Any]` *(static)* — Convert parsed arguments into a dictionary suitable for SSHClient.__init__.

<a id="core_utils--execution_monitor--_dialog_viewer"></a>
### `core_utils/execution_monitor/_dialog_viewer.py`

Subprocess-based dialog viewer for custom button labels.

- [`run(title: str, message: str, force_label: str | None = None)`](pythontk/pythontk/core_utils/execution_monitor/_dialog_viewer.py#L12) — Display a dialog with custom buttons matching VS Code style.

<a id="core_utils--execution_monitor--_execution_monitor"></a>
### `core_utils/execution_monitor/_execution_monitor.py`

- **[`class ExecutionMonitor`](pythontk/pythontk/core_utils/execution_monitor/_execution_monitor.py#L14)** — Utilities for monitoring and handling long-running executions.
  - `ExecutionMonitor.is_escape_pressed()` *(static)* — Check if the Escape key is currently pressed (Windows & Linux).
  - `ExecutionMonitor.set_interpreter(cls, path)` *(class)* — Set a custom Python interpreter to use for subprocesses.
  - `ExecutionMonitor.on_long_execution(threshold, callback, interval=None, allow_escape_cancel=False, indicator=None)` *(static)* — Decorator that triggers a callback if the decorated function
  - `ExecutionMonitor.show_long_execution_dialog(title, message, force_action=None)` *(static)* — Show a dialog to ask the user how to proceed with a long operation.
  - `ExecutionMonitor.execution_monitor(threshold, message, logger=None, allow_escape_cancel=False, show_dialog: bool = True, force_action: str | None = None, watchdog_timeout: float | None = None, watchdog_heartbeat_interval: float = 1.0, watchdog_check_interval: float = 1.0, watchdog_kill_tree: bool = True, watchdog_heartbeat_path: str | None = None, indicator: bool | str | None = None)` *(static)* — Decorator that monitors execution time and (optionally) prompts the user via a native
  - `ExecutionMonitor.external_watchdog(timeout: float, message: str = 'Operation appears to have stalled', heartbeat_interval: float = 1.0, check_interval: float = 1.0, kill_tree: bool = True, logger=None, heartbeat_path: str | None = None)` *(static)* — Decorator that starts an OS-level watchdog for the current process.

<a id="core_utils--execution_monitor--_gif_viewer"></a>
### `core_utils/execution_monitor/_gif_viewer.py`

- [`run(gif_path, target_size=DEFAULT_SIZE, pos=None)`](pythontk/pythontk/core_utils/execution_monitor/_gif_viewer.py#L48)

<a id="core_utils--execution_monitor--_spinner"></a>
### `core_utils/execution_monitor/_spinner.py`

Lightweight canvas-based spinner for task-indicator overlay.

- [`run(size=DEFAULT_SIZE, pos=None)`](pythontk/pythontk/core_utils/execution_monitor/_spinner.py#L17)

<a id="core_utils--git"></a>
### `core_utils/git.py`

- **[`class Git`](pythontk/pythontk/core_utils/git.py#L10)** — A wrapper around git subprocess commands for a specific repository.
  - `Git.execute(self, cmd: Union[str, List[str]], desc: str = None, check: bool = True) -> Optional[str]` — Run a generic shell command in the repository directory.
  - `Git.run(self, cmd: Union[str, List[str]], desc: str = None, check: bool = True) -> Optional[str]` — Run a git command in the repository.
  - `Git.checkout(self, branch: str)` — Checkout a branch.
  - `Git.pull(self, remote: str = 'origin', branch: str = None)` — Pull changes.
  - `Git.push(self, remote: str = 'origin', branch: str = None)` — Push changes.
  - `Git.merge(self, source_branch: str)` — Merge a branch into the current branch.
  - `Git.fetch(self, remote: str = 'origin')` — Fetch remote.
  - `Git.status(self) -> str` — Get status output.
  - `Git.current_branch(self) -> str` — Get current branch name.

<a id="core_utils--help_mixin"></a>
### `core_utils/help_mixin.py`

HelpMixin - Enhanced help system leveraging Python's built-in help infrastructure.

- **[`class HelpMixin`](pythontk/pythontk/core_utils/help_mixin.py#L14)** — A mixin providing enhanced help() functionality with filtering and sorting.
  - `HelpMixin.help(cls, name: Optional[str] = None, *, members: Optional[str] = None, inherited: bool = True, brief: bool = False, sort: bool = False, private: bool = False, returns: bool = False) -> Optional[str]` *(class)* — Display or return help information for this class or a specific member.
  - `HelpMixin.source(cls, name: Optional[str] = None, *, returns: bool = False) -> Optional[str]` *(class)* — Get source code for the class or a specific member.
  - `HelpMixin.where(cls, name: Optional[str] = None, *, returns: bool = False) -> Optional[str]` *(class)* — Get the file and line number where the class or member is defined.
  - `HelpMixin.mro(cls, *, brief: bool = False, returns: bool = False) -> Optional[str]` *(class)* — Show the method resolution order (inheritance chain) for this class.
  - `HelpMixin.signature(cls, name: str, *, returns: bool = False) -> Optional[str]` *(class)* — Get detailed signature information for a method.
  - `HelpMixin.classify(cls, name: Optional[str] = None, *, returns: bool = False) -> Optional[str]` *(class)* — Classify a member or list all members with their classifications.
  - `HelpMixin.list_members(cls, members: Optional[str] = None, *, inherited: bool = True, private: bool = False, sort: bool = True, returns: bool = False) -> Optional[List[str]]` *(class)* — Get a list of member names.
  - `HelpMixin.about(target, name=None, *, brief=False, returns=False)` *(static)* — Get help for any Python object (class, function, module, method, etc.).

<a id="core_utils--hierarchy_diff"></a>
### `core_utils/hierarchy_diff.py`

- **[`class HierarchyDiff`](pythontk/pythontk/core_utils/hierarchy_diff.py#L7)** — Generic data class to hold hierarchical difference results.
  - `HierarchyDiff.is_valid(self) -> bool` — Check if hierarchy has no significant differences.
  - `HierarchyDiff.has_differences(self) -> bool` — Check if any differences exist (including extra items).
  - `HierarchyDiff.total_issues(self) -> int` — Get total count of all issues.
  - `HierarchyDiff.as_dict(self) -> Dict[str, List[str]]` — Convert to dictionary representation.
  - `HierarchyDiff.as_json(self, indent: Optional[int] = 2) -> str` — Convert to JSON string.
  - `HierarchyDiff.save_to_file(self, filepath: str, indent: Optional[int] = 2) -> None` — Save diff result to JSON file.
  - `HierarchyDiff.load_from_file(cls, filepath: str) -> 'HierarchyDiff'` *(class)* — Load diff result from JSON file.
  - `HierarchyDiff.clear(self) -> None` — Clear all diff results.
  - `HierarchyDiff.merge(self, other: 'HierarchyDiff') -> None` — Merge another diff result into this one.
  - `HierarchyDiff.get_summary(self) -> Dict[str, int]` — Get summary counts of all difference types.
  - `HierarchyDiff.filter_by_pattern(self, pattern: str, field: str = 'missing') -> List[str]` — Filter items by regex pattern.
  - `HierarchyDiff.add_metadata(self, key: str, value: Any) -> None` — Add metadata to the diff result.

<a id="core_utils--hierarchy_utils--backup--hierarchy_analyzer"></a>
### `core_utils/hierarchy_utils/backup/hierarchy_analyzer.py`

- **[`class DifferenceType(Enum)`](pythontk/pythontk/core_utils/hierarchy_utils/backup/hierarchy_analyzer.py#L8)** — Types of differences that can be found between hierarchies.
- **[`class HierarchyDifference`](pythontk/pythontk/core_utils/hierarchy_utils/backup/hierarchy_analyzer.py#L18)** — Represents a single difference between hierarchies.
- **[`class HierarchyAnalyzer`](pythontk/pythontk/core_utils/hierarchy_utils/backup/hierarchy_analyzer.py#L30)** — Analyzer for comparing hierarchical structures and identifying differences.
  - `HierarchyAnalyzer.compare_path_sets(current_paths: Set[str], reference_paths: Set[str], path_separator: str = '|') -> Dict[str, Set[str]]` *(static)* — Compare two sets of hierarchical paths and categorize differences.
  - `HierarchyAnalyzer.analyze_hierarchy_differences(current_items: List[Any], reference_items: List[Any], path_extractor: Callable[[Any], str], attribute_extractors: Dict[str, Callable[[Any], Any]] = None, path_separator: str = '|') -> List[HierarchyDifference]` *(static)* — Perform comprehensive analysis of differences between hierarchies.
  - `HierarchyAnalyzer.detect_moved_items(differences: List[HierarchyDifference], similarity_threshold: float = 0.8, path_separator: str = '|') -> List[HierarchyDifference]` *(static)* — Detect items that may have been moved rather than deleted/added.
  - `HierarchyAnalyzer.categorize_differences(differences: List[HierarchyDifference], path_separator: str = '|') -> Dict[str, Dict[str, List[HierarchyDifference]]]` *(static)* — Categorize differences by type and hierarchy level.
  - `HierarchyAnalyzer.generate_diff_report(differences: List[HierarchyDifference], include_details: bool = True, max_items_per_category: int = 10) -> str` *(static)* — Generate a human-readable report of differences.
  - `HierarchyAnalyzer.export_differences_to_dict(differences: List[HierarchyDifference]) -> Dict[str, Any]` *(static)* — Export differences to a dictionary format for serialization.
  - `HierarchyAnalyzer.filter_differences(differences: List[HierarchyDifference], types: List[DifferenceType] = None, path_patterns: List[str] = None, exclude_patterns: List[str] = None) -> List[HierarchyDifference]` *(static)* — Filter differences based on type and path patterns.

<a id="core_utils--hierarchy_utils--backup--hierarchy_indexer"></a>
### `core_utils/hierarchy_utils/backup/hierarchy_indexer.py`

- **[`class HierarchyIndexer`](pythontk/pythontk/core_utils/hierarchy_utils/backup/hierarchy_indexer.py#L6)** — Generic utilities for building and querying tree indices.
  - `HierarchyIndexer.build_path_index(items: List[Any], get_path_func: Callable[[Any], str], path_separator: str = '|', clean_namespaces: bool = True, namespace_separator: str = ':') -> Dict[str, List[Any]]` *(static)* — Build an index mapping cleaned paths to items.
  - `HierarchyIndexer.find_by_path(index: Dict[str, List[Any]], target_path: str, clean_namespaces: bool = True, path_separator: str = '|', namespace_separator: str = ':') -> List[Any]` *(static)* — Find items in index by path.
  - `HierarchyIndexer.find_by_tail_path(index: Dict[str, List[Any]], target_tail: str, num_components: int = 2, path_separator: str = '|') -> List[Any]` *(static)* — Find items by matching the tail portion of their paths.
  - `HierarchyIndexer.get_path_components_index(items: List[Any], get_path_func: Callable[[Any], str], path_separator: str = '|') -> Dict[str, List[Any]]` *(static)* — Build an index mapping individual path components to items.
  - `HierarchyIndexer.get_depth_index(items: List[Any], get_path_func: Callable[[Any], str], path_separator: str = '|') -> Dict[int, List[Any]]` *(static)* — Build an index mapping path depths to items.

<a id="core_utils--hierarchy_utils--backup--hierarchy_matching"></a>
### `core_utils/hierarchy_utils/backup/hierarchy_matching.py`

- **[`class HierarchyMatching`](pythontk/pythontk/core_utils/hierarchy_utils/backup/hierarchy_matching.py#L7)** — Generic matching strategies for hierarchical data.
  - `HierarchyMatching.exact_path_match(source_items: List[Any], target_items: List[Any], get_path_func: Callable[[Any], str], path_separator: str = '|', clean_namespaces: bool = True, namespace_separator: str = ':') -> Dict[Any, List[Any]]` *(static)* — Find exact path matches between source and target items.
  - `HierarchyMatching.tail_path_match(source_items: List[Any], target_items: List[Any], get_path_func: Callable[[Any], str], num_components: int = 2, path_separator: str = '|', clean_namespaces: bool = True, namespace_separator: str = ':') -> Dict[Any, List[Any]]` *(static)* — Find matches by comparing tail portions of paths.
  - `HierarchyMatching.fuzzy_name_match(source_items: List[Any], target_items: List[Any], get_name_func: Callable[[Any], str], similarity_threshold: float = 0.8) -> Dict[Any, List[Any]]` *(static)* — Find matches using fuzzy string matching on names.
  - `HierarchyMatching.multi_strategy_match(source_items: List[Any], target_items: List[Any], get_path_func: Callable[[Any], str], get_name_func: Optional[Callable[[Any], str]] = None, strategies: List[str] = None, path_separator: str = '|', clean_namespaces: bool = True, namespace_separator: str = ':', fuzzy_threshold: float = 0.8) -> Dict[Any, List[Any]]` *(static)* — Apply multiple matching strategies in order of preference.

<a id="core_utils--hierarchy_utils--hierarchy_analyzer"></a>
### `core_utils/hierarchy_utils/hierarchy_analyzer.py`

- **[`class DifferenceType(Enum)`](pythontk/pythontk/core_utils/hierarchy_utils/hierarchy_analyzer.py#L8)** — Types of differences that can be found between hierarchies.
- **[`class HierarchyDifference`](pythontk/pythontk/core_utils/hierarchy_utils/hierarchy_analyzer.py#L18)** — Represents a single difference between hierarchies.
- **[`class HierarchyAnalyzer`](pythontk/pythontk/core_utils/hierarchy_utils/hierarchy_analyzer.py#L30)** — Analyzer for comparing hierarchical structures and identifying differences.
  - `HierarchyAnalyzer.compare_path_sets(current_paths: Set[str], reference_paths: Set[str], path_separator: str = '|') -> Dict[str, Set[str]]` *(static)* — Compare two sets of hierarchical paths and categorize differences.
  - `HierarchyAnalyzer.analyze_hierarchy_differences(current_items: List[Any], reference_items: List[Any], path_extractor: Callable[[Any], str], attribute_extractors: Dict[str, Callable[[Any], Any]] = None, path_separator: str = '|') -> List[HierarchyDifference]` *(static)* — Perform comprehensive analysis of differences between hierarchies.
  - `HierarchyAnalyzer.detect_moved_items(differences: List[HierarchyDifference], similarity_threshold: float = 0.8, path_separator: str = '|') -> List[HierarchyDifference]` *(static)* — Detect items that may have been moved rather than deleted/added.
  - `HierarchyAnalyzer.categorize_differences(differences: List[HierarchyDifference], path_separator: str = '|') -> Dict[str, Dict[str, List[HierarchyDifference]]]` *(static)* — Categorize differences by type and hierarchy level.
  - `HierarchyAnalyzer.generate_diff_report(differences: List[HierarchyDifference], include_details: bool = True, max_items_per_category: int = 10) -> str` *(static)* — Generate a human-readable report of differences.
  - `HierarchyAnalyzer.export_differences_to_dict(differences: List[HierarchyDifference]) -> Dict[str, Any]` *(static)* — Export differences to a dictionary format for serialization.
  - `HierarchyAnalyzer.filter_differences(differences: List[HierarchyDifference], types: List[DifferenceType] = None, path_patterns: List[str] = None, exclude_patterns: List[str] = None) -> List[HierarchyDifference]` *(static)* — Filter differences based on type and path patterns.

<a id="core_utils--hierarchy_utils--hierarchy_indexer"></a>
### `core_utils/hierarchy_utils/hierarchy_indexer.py`

- **[`class HierarchyIndexer`](pythontk/pythontk/core_utils/hierarchy_utils/hierarchy_indexer.py#L6)** — Generic utilities for building and querying tree indices.
  - `HierarchyIndexer.build_path_index(items: List[Any], get_path_func: Callable[[Any], str], path_separator: str = '|', clean_namespaces: bool = True, namespace_separator: str = ':') -> Dict[str, List[Any]]` *(static)* — Build an index mapping cleaned paths to items.
  - `HierarchyIndexer.find_by_path(index: Dict[str, List[Any]], target_path: str, clean_namespaces: bool = True, path_separator: str = '|', namespace_separator: str = ':') -> List[Any]` *(static)* — Find items in index by path.
  - `HierarchyIndexer.find_by_tail_path(index: Dict[str, List[Any]], target_tail: str, num_components: int = 2, path_separator: str = '|') -> List[Any]` *(static)* — Find items by matching the tail portion of their paths.
  - `HierarchyIndexer.get_path_components_index(items: List[Any], get_path_func: Callable[[Any], str], path_separator: str = '|') -> Dict[str, List[Any]]` *(static)* — Build an index mapping individual path components to items.
  - `HierarchyIndexer.get_depth_index(items: List[Any], get_path_func: Callable[[Any], str], path_separator: str = '|') -> Dict[int, List[Any]]` *(static)* — Build an index mapping path depths to items.

<a id="core_utils--hierarchy_utils--hierarchy_matching"></a>
### `core_utils/hierarchy_utils/hierarchy_matching.py`

- **[`class HierarchyMatching`](pythontk/pythontk/core_utils/hierarchy_utils/hierarchy_matching.py#L7)** — Generic matching strategies for hierarchical data.
  - `HierarchyMatching.exact_path_match(source_items: List[Any], target_items: List[Any], get_path_func: Callable[[Any], str], path_separator: str = '|', clean_namespaces: bool = True, namespace_separator: str = ':') -> Dict[Any, List[Any]]` *(static)* — Find exact path matches between source and target items.
  - `HierarchyMatching.tail_path_match(source_items: List[Any], target_items: List[Any], get_path_func: Callable[[Any], str], num_components: int = 2, path_separator: str = '|', clean_namespaces: bool = True, namespace_separator: str = ':') -> Dict[Any, List[Any]]` *(static)* — Find matches by comparing tail portions of paths.
  - `HierarchyMatching.fuzzy_name_match(source_items: List[Any], target_items: List[Any], get_name_func: Callable[[Any], str], similarity_threshold: float = 0.8) -> Dict[Any, List[Any]]` *(static)* — Find matches using fuzzy string matching on names.
  - `HierarchyMatching.multi_strategy_match(source_items: List[Any], target_items: List[Any], get_path_func: Callable[[Any], str], get_name_func: Optional[Callable[[Any], str]] = None, strategies: List[str] = None, path_separator: str = '|', clean_namespaces: bool = True, namespace_separator: str = ':', fuzzy_threshold: float = 0.8) -> Dict[Any, List[Any]]` *(static)* — Apply multiple matching strategies in order of preference.

<a id="core_utils--logging_mixin"></a>
### `core_utils/logging_mixin.py`

- **[`class StripHtmlFormatter(internal_logging.Formatter)`](pythontk/pythontk/core_utils/logging_mixin.py#L11)** — Formatter that strips HTML tags from the message.
  - `StripHtmlFormatter.format(self, record)`
- **[`class LevelAwareFormatter(internal_logging.Formatter)`](pythontk/pythontk/core_utils/logging_mixin.py#L33)** — Formatter that dynamically selects format per-record based on log level.
  - `LevelAwareFormatter.format(self, record)`
- **[`class LoggerExt`](pythontk/pythontk/core_utils/logging_mixin.py#L73)**
  - `LoggerExt.patch(cls, logger: internal_logging.Logger) -> None` *(class)* — Patch the logger with additional methods and setup.
  - `LoggerExt.get_color(cls, level: str) -> str` *(class)* — Get the color code for a given log level.
  - `LoggerExt.register_html_preset(cls, name: str, format_str: str) -> None` *(class)* — Register a new HTML preset.
  - `LoggerExt.get_html_preset(cls, name: str) -> str` *(class)* — Get an HTML preset by name.
  - `LoggerExt.format_message_as_html(cls, message: str, level: str, preset: str = None) -> str` *(class)* — Format a message using HTML presets.
- **[`class DefaultTextLogHandler(internal_logging.Handler)`](pythontk/pythontk/core_utils/logging_mixin.py#L1004)** — A generic thread-safe logging handler that writes logs to any widget
  - `DefaultTextLogHandler.emit(self, record: internal_logging.LogRecord) -> None`
  - `DefaultTextLogHandler.get_color(self, level: str) -> str`
- **[`class TableMixin`](pythontk/pythontk/core_utils/logging_mixin.py#L1056)** — Mixin for formatting data as ASCII tables.
  - `TableMixin.format_table(self, data: List[List[Any]], headers: List[str], title: Optional[str] = None, col_max_width: int = 60, max_width: int = 160) -> str` — Formats a list of lists as an ASCII table.
  - `TableMixin.log_table(self, data: List[List[Any]], headers: List[str], title: Optional[str] = None, level: str = 'info') -> None` — Logs a formatted table.
- **[`class LoggingMixin(TableMixin)`](pythontk/pythontk/core_utils/logging_mixin.py#L1210)** — Mixin class for logging utilities.
  - `LoggingMixin.logger(cls) -> internal_logging.Logger`
  - `LoggingMixin.class_logger(cls) -> internal_logging.Logger`
  - `LoggingMixin.logging(cls)` — Access to Python's internal logging module (aliased).
  - `LoggingMixin.set_log_level(cls, level: int | str)` *(class)* — Set log level for the class logger and its handlers.

<a id="core_utils--module_reloader"></a>
### `core_utils/module_reloader.py`

Helpers for hot-reloading packages and their submodules.

- [`reload_package(package: ModuleRef, **kwargs) -> List[ModuleType]`](pythontk/pythontk/core_utils/module_reloader.py#L274) — Convenience wrapper around :class:`ModuleReloader`.
- **[`class ModuleReloader`](pythontk/pythontk/core_utils/module_reloader.py#L17)** — Flexible controller for reloading packages and related modules.
  - `ModuleReloader.reload(self, package: ModuleRef, *, include_submodules: Optional[bool] = None, dependencies_first: Optional[Iterable[ModuleRef]] = None, dependencies_last: Optional[Iterable[ModuleRef]] = None, predicate: Optional[Callable[[ModuleType], bool]] = None, before_reload: Optional[Callable[[ModuleType], None]] = None, after_reload: Optional[Callable[[ModuleType], None]] = None, import_missing: Optional[bool] = None, verbose: Optional[Union[bool, int]] = None, max_passes: Optional[int] = None, exclude_modules: Optional[Iterable[str]] = None) -> List[ModuleType]` — Reload a package and return the modules processed.

<a id="core_utils--module_resolver"></a>
### `core_utils/module_resolver.py`

Reusable module attribute resolver for package-style imports.

- [`bootstrap_package(module_globals: MutableMapping[str, Any], *, include: Optional[IncludeMapping] = None, module_to_parent: Optional[Mapping[str, str]] = None, eager: bool = False, allow_getattr: bool = True, install_legacy_helpers: bool = True, on_import_error: Optional[Callable[[str, Exception], None]] = None, method_predicate: Optional[Callable[[str], bool]] = None, custom_getattr: Optional[Callable[[str], Any]] = None, lazy_import: Optional[bool] = None) -> PackageResolverHandle`](pythontk/pythontk/core_utils/module_resolver.py#L745) — Bootstrap a package's ``__init__`` module with dynamic attribute resolution.
- [`create_namespace_aliases(module_globals: MutableMapping[str, Any], aliases: Mapping[str, Union[str, Sequence[str]]], *, include_spec: Optional[IncludeMapping] = None) -> None`](pythontk/pythontk/core_utils/module_resolver.py#L824) — Create multi-inheritance namespace classes from groups of related classes.
- **[`class ModuleAttributeResolver`](pythontk/pythontk/core_utils/module_resolver.py#L34)** — Discover and resolve attributes exposed from package submodules lazily.
  - `ModuleAttributeResolver.build(self) -> 'ModuleAttributeResolver'` — Populate resolver dictionaries based on current include spec.
  - `ModuleAttributeResolver.rebuild(self, include: Optional[Mapping[str, Union[Sequence[str], str]]] = None) -> 'ModuleAttributeResolver'` — Reset include spec (optional) and rebuild dictionaries.
  - `ModuleAttributeResolver.resolve(self, name: str)` — Resolve an attribute using the registered dictionaries.
  - `ModuleAttributeResolver.get_module(self, module_name: str) -> ModuleType` — Import a module managed by the resolver, using the local cache.
  - `ModuleAttributeResolver.bind_to(self, module_globals: MutableMapping[str, object], *, install_getattr: bool = True, eager: bool = False, names: Optional[Iterable[str]] = None) -> None` — Bind resolver helpers into a module's globals dictionary.
  - `ModuleAttributeResolver.iter_registered_names(self) -> Iterable[str]` — Return an iterable of attribute names known to the resolver.
  - `ModuleAttributeResolver.clear_module_cache(self) -> None` — Drop cached module imports managed by the resolver.
- **[`class PackageResolverHandle`](pythontk/pythontk/core_utils/module_resolver.py#L431)** — Facade that wires a :class:`ModuleAttributeResolver` into a package module.
  - `PackageResolverHandle.install(self, *, expose_maps: bool = True, install_helpers: bool = True, allow_getattr: Optional[bool] = None, eager: Optional[bool] = None, custom_getattr: Optional[Callable[[str], Any]] = None) -> None` — Publish resolver artifacts into the target module globals.
  - `PackageResolverHandle.configure(self, *, include: Optional[IncludeMapping] = None, module_to_parent: Optional[Mapping[str, str]] = None, merge: bool = True, eager: Optional[bool] = None, custom_getattr: Optional[Callable[[str], Any]] = None) -> ModuleAttributeResolver` — Reconfigure the underlying resolver and optionally re-export symbols.
  - `PackageResolverHandle.build_dictionaries(self, include_override: Optional[IncludeMapping] = None, *, eager: bool = False, custom_getattr: Optional[Callable[[str], Any]] = None) -> ModuleAttributeResolver` — Compatibility wrapper mirroring the legacy ``build_dictionaries`` helper.
  - `PackageResolverHandle.import_module(self, module_name: str) -> ModuleType`
  - `PackageResolverHandle.get_attribute_from_module(self, module: ModuleType, attribute_name: str, class_name: Optional[str] = None)`
  - `PackageResolverHandle.export_all(self) -> None` — Eagerly publish resolver-managed attributes into the module globals.

<a id="core_utils--namedtuple_container"></a>
### `core_utils/namedtuple_container.py`

- **[`class NamedTupleContainer(LoggingMixin)`](pythontk/pythontk/core_utils/namedtuple_container.py#L8)** — A generic container class for managing collections of named tuples.
  - `NamedTupleContainer.extend(self, objects: Union[List[namedtuple], List[tuple], Any], **metadata) -> None` — Extend the container with new objects while handling duplicates properly.
  - `NamedTupleContainer.get(self, return_field: Optional[str] = None, **conditions) -> Union[List[Any], Any, None]` — Query the named tuples based on specified conditions.
  - `NamedTupleContainer.filter(self, predicate: Callable[[namedtuple], bool]) -> 'NamedTupleContainer'` — Filter the container based on a predicate function.
  - `NamedTupleContainer.map(self, func: Callable[[namedtuple], namedtuple]) -> 'NamedTupleContainer'` — Apply a function to all named tuples in the container.
  - `NamedTupleContainer.modify(self, index: int, **kwargs) -> namedtuple` — Modify a named tuple at a specific index within the container.
  - `NamedTupleContainer.remove(self, index: int) -> namedtuple` — Remove a named tuple at a specific index within the container.
  - `NamedTupleContainer.clear(self) -> None` — Clear all named tuples from the container.
  - `NamedTupleContainer.to_dict_list(self) -> List[Dict[str, Any]]` — Convert all named tuples to a list of dictionaries.
  - `NamedTupleContainer.to_csv(self, filename: str, **kwargs) -> None` — Export the container to a CSV file.
  - `NamedTupleContainer.from_csv(cls, filename: str, tuple_class_name: str = 'CsvRow', **kwargs) -> 'NamedTupleContainer'` *(class)* — Create a NamedTupleContainer from a CSV file.

<a id="core_utils--namespace_handler"></a>
### `core_utils/namespace_handler.py`

- **[`class Placeholder`](pythontk/pythontk/core_utils/namespace_handler.py#L7)**
  - `Placeholder.info(self) -> dict`
  - `Placeholder.create(self, *args, **kwargs)`
- **[`class NamespaceHandler(LoggingMixin)`](pythontk/pythontk/core_utils/namespace_handler.py#L40)** — A NamespaceHandler that manages its own internal dictionary without attaching
  - `NamespaceHandler.placeholders(self) -> dict[str, Any]` *(property)*
  - `NamespaceHandler.is_placeholder(self, value: Any) -> bool`
  - `NamespaceHandler.get_placeholder(self, key: str) -> Optional[Placeholder]`
  - `NamespaceHandler.set_placeholder(self, key: str, placeholder: Placeholder)`
  - `NamespaceHandler.resolve_all_placeholders(self)`
  - `NamespaceHandler.has_placeholder(self, key: str) -> bool`
  - `NamespaceHandler.keys(self, inc_placeholders=False)`
  - `NamespaceHandler.items(self, inc_placeholders=False)`
  - `NamespaceHandler.values(self, inc_placeholders=False)`
  - `NamespaceHandler.setdefault(self, key: str, default: Any = None) -> Any`
  - `NamespaceHandler.has(self, key: str) -> bool`
  - `NamespaceHandler.raw(self, key: str) -> Optional[Any]`
  - `NamespaceHandler.resolve(self, key: str, default: Any = None, resolve_placeholders: bool = True) -> Any`
  - `NamespaceHandler.is_resolving(self, key: str) -> bool` — Returns True if this key is currently being resolved (to prevent recursion).

<a id="core_utils--package_manager"></a>
### `core_utils/package_manager.py`

- **[`class PackageManager(_PkgVersionCheck, _PkgVersionUtils, _PackageManagerHelperMixin, help_mixin.HelpMixin)`](pythontk/pythontk/core_utils/package_manager.py#L424)** — A class that encapsulates package management functionalities using pip.
  - `PackageManager.pip(self, command, output_as_string=False)` — Execute a pip command and return the output.
  - `PackageManager.get_local_dependency_order(paths: List[Union[str, Path]]) -> List[Path]` *(static)* — Sort a list of local repository paths based on their pyproject.toml dependencies.

<a id="core_utils--singleton_mixin"></a>
### `core_utils/singleton_mixin.py`

- **[`class SingletonMixin`](pythontk/pythontk/core_utils/singleton_mixin.py#L6)** — Reusable singleton mixin that supports optional key-based instances.
  - `SingletonMixin.instance(cls, *args: Any, **kwargs: Any) -> Any` *(class)*
  - `SingletonMixin.has_instance(cls, singleton_key: Optional[Any] = None) -> bool` *(class)*
  - `SingletonMixin.reset_instance(cls, singleton_key: Optional[Any] = None) -> None` *(class)*

<a id="file_utils--_file_utils"></a>
### `file_utils/_file_utils.py`

- **[`class FileUtils(HelpMixin)`](pythontk/pythontk/file_utils/_file_utils.py#L16)**
  - `FileUtils.is_valid(filepath: str, expected_type: Optional[str] = None) -> bool` *(static)* — Check if a path is valid, optionally requiring a specific type ('file' or 'dir').
  - `FileUtils.create_dir(filepath: str) -> None` *(static)* — Create a directory if one doesn't already exist.
  - `FileUtils.get_dir_contents(dirPath, content='file', recursive=False, num_threads=1, inc_files=[], exc_files=[], inc_dirs=[], exc_dirs=[], group_by_type=False)` *(static)* — Get the contents of a directory and any of its children.
  - `FileUtils.open_explorer(path: str, create_dir: bool = False, logger=None) -> bool` *(static)* — Open the file explorer at the given path.
  - `FileUtils.get_file(filepath, mode='a+')` *(static)* — Return a file object with the given mode.
  - `FileUtils.get_file_contents(filepath: str, as_list=False, encoding='utf-8') -> None` *(static)* — Get each line of a text file as indices of a list.
  - `FileUtils.write_to_file(filepath, lines)` *(static)* — Write the given list contents to the given file.
  - `FileUtils.copy_file(file_path: str, destination: str, new_name: Optional[str] = None, overwrite: bool = True, create_dir: bool = True) -> str` *(static)* — Copies a file to a specified folder, ensuring the folder exists.
  - `FileUtils.move_file(cls, file_path: Union[str, List[Union[str, Tuple[str, str]]]], destination: str, new_name: Optional[str] = None, overwrite: bool = True, create_dir: bool = True, verbose: bool = False) -> Union[str, List[str]]` *(class)* — Moves one or more files to a specified folder.
  - `FileUtils.get_file_info(cls, paths, info, hash_algo=None, force_tuples=False)` *(class)* — Returns file and directory information for a list of file strings based on specified parameters.
  - `FileUtils.format_path(p: Union[str, List[str]], section: Union[str, None] = None, replace: Union[str, None] = None) -> Union[str, List[str]]` *(static)* — Format a given filepath(s).
  - `FileUtils.convert_to_relative_path(file_path: str, base_dir: str, prepend_base: bool = True, check_existence: bool = False) -> str` *(static)* — Convert an absolute file path to a relative path based on the given base directory.
  - `FileUtils.remap_file_paths(source_paths: List[str], target_dir: str, base_dir: str) -> List[Tuple[str, str, str]]` *(static)* — Remap a list of file paths to a new directory while preserving their relative
  - `FileUtils.append_path(cls, path, **kwargs)` *(class)* — Append a directory to the python path.
  - `FileUtils.get_object_path(obj, inc_filename: bool = False) -> str` *(static)* — Retrieve the absolute file path associated with a Python object.
  - `FileUtils.get_classes_from_path(cls, path, returned_type=['classname', 'filepath'], inc=[], exc=[], top_level_only=True, force_tuples=False)` *(class)* — Scans the specified directory or Python file, loads each file as a module, and retrieves classes fr…
  - `FileUtils.set_json_file(cls, file)` *(class)* — Set the current json filepath.
  - `FileUtils.get_json_file(cls)` *(class)* — Get the current json filepath.
  - `FileUtils.set_json(cls, key, value, file=None)` *(class)* — Parameters:
  - `FileUtils.get_json(cls, key, file=None)` *(class)* — Parameters:

<a id="file_utils--metadata"></a>
### `file_utils/metadata.py`

- **[`class MetadataInternal`](pythontk/pythontk/file_utils/metadata.py#L9)** — Internal utilities for handling file metadata on Windows and Linux.
- **[`class Metadata(MetadataInternal)`](pythontk/pythontk/file_utils/metadata.py#L409)** — Public interface for metadata operations.
  - `Metadata.get(cls, file_path: Any, *keys: str, mode: str = 'metadata') -> Any` *(class)* — Unified get method for metadata and tags.
  - `Metadata.set(cls, file_path: Any, mode: str = 'metadata', **kwargs) -> None` *(class)* — Unified set method for metadata and tags.

<a id="img_utils--_img_utils"></a>
### `img_utils/_img_utils.py`

- **[`class ImgUtils(HelpMixin)`](pythontk/pythontk/img_utils/_img_utils.py#L33)** — Helper methods for working with image file formats.
  - `ImgUtils.im_help(a=None)` *(static)* — Get help documentation on a specific PIL image attribute
  - `ImgUtils.allow_large_images(cls)` *(class)* — Context manager to safely load very large images.
  - `ImgUtils.ensure_image(cls, input_image: Union[str, Image.Image], mode: str = None, *, max_pixels: Optional[int] = 268435456) -> Image.Image` *(class)* — Ensures the input is a valid PIL Image.
  - `ImgUtils.enforce_mode(cls, image: Image.Image, target_mode: str, allow_compatible: bool = True) -> Image.Image` *(class)* — Converts image to target_mode, optionally allowing compatible modes to preserve file size.
  - `ImgUtils.assert_pathlike(obj: object, name: str = 'argument') -> None` *(static)* — Assert that the given object is a valid path-like object.
  - `ImgUtils.create_image(mode, size=(4096, 4096), color=None)` *(static)* — Create a new image.
  - `ImgUtils.save_image(cls, image: Union[str, Image.Image], name: str, mode: str = None, **kwargs)` *(class)* — Saves an image to the specified path, with optional mode conversion.
  - `ImgUtils.load_image(cls, filepath)` *(class)* — Load an image from the given file path and return a copy of the image object.
  - `ImgUtils.get_images(cls, directory, inc=None, exc='')` *(class)* — Get bitmap images from a given directory as PIL images.
  - `ImgUtils.get_image_info(cls, file_paths: Union[str, List[str]]) -> List[Dict[str, Any]]` *(class)* — Get information about image files.
  - `ImgUtils.are_identical(cls, imageA, imageB)` *(class)* — Check if two images are the same.
  - `ImgUtils.resize_image(cls, image, x, y)` *(class)* — Returns a resized copy of an image.
  - `ImgUtils.ensure_pot(cls, image: Union[str, Image.Image]) -> Image.Image` *(class)* — Resizes an image to the nearest Power of Two dimensions.
  - `ImgUtils.set_bit_depth(cls, image, map_type: str) -> object` *(class)* — Sets the bit depth and image mode of an image according to the map type.
  - `ImgUtils.invert_grayscale_image(cls, image: Union[str, Image.Image]) -> Image.Image` *(class)* — Inverts a grayscale image.
  - `ImgUtils.invert_channels(cls, image, channels='RGBA')` *(class)* — Invert specified channels in an image.
  - `ImgUtils.create_mask(cls, image, mask, background=(0, 0, 0, 255), foreground=(255, 255, 255, 255))` *(class)* — Create mask(s) from the given image(s).
  - `ImgUtils.fill_masked_area(cls, image, color, mask)` *(class)* — Parameters:
  - `ImgUtils.fill(cls, image, color=(0, 0, 0, 0))` *(class)* — Parameters:
  - `ImgUtils.get_background(cls, image, mode=None, average=False)` *(class)* — Sample the pixel values of each corner of an image and if they are uniform, return the result.
  - `ImgUtils.replace_color(cls, image, from_color=(0, 0, 0, 0), to_color=(0, 0, 0, 0), mode=None)` *(class)* — Parameters:
  - `ImgUtils.set_contrast(cls, image, level=255)` *(class)* — Parameters:
  - `ImgUtils.convert_rgb_to_gray(data)` *(static)* — Convert an RGB Image data array to grayscale.
  - `ImgUtils.convert_rgb_to_hsv(cls, image)` *(class)* — Manually convert the image to a NumPy array, iterate over the pixels
  - `ImgUtils.convert_i_to_l(cls, image)` *(class)* — Convert to 8 bit 'L' grayscale.
  - `ImgUtils.pack_channels(cls, channel_files: dict[str, str | Image.Image], channels: list[str] = None, out_mode: str = None, fill_values: dict[str, int] = None, output_path: str = None, output_format: str = 'PNG', grayscale_to_rgb: bool = False, invert_channels: list[str] = None, **kwargs) -> str | Image.Image` *(class)* — Packs up to 4 grayscale images into R, G, B, A channels of a single image.
  - `ImgUtils.pack_channel_into_alpha(cls, image: Union[str, Image.Image], alpha: Union[str, Image.Image], output_path: Optional[str] = None, invert_alpha: bool = False, resize_alpha: bool = True, preserve_existing_alpha: bool = False) -> str | Image.Image` *(class)* — Packs a channel from the alpha source image into the alpha channel of the base image.
  - `ImgUtils.srgb_to_linear(cls, data)` *(class)* — Friendly wrapper: accepts PIL Image, numpy array, or list/tuple.
  - `ImgUtils.linear_to_srgb(cls, data)` *(class)* — Friendly wrapper: accepts PIL Image, numpy array, or list/tuple.
  - `ImgUtils.generate_mipmaps(cls, image: Image.Image) -> Image.Image` *(class)* — Generates mipmaps for an image.
  - `ImgUtils.depalettize_image(cls, image: Image.Image) -> Image.Image` *(class)* — Converts a paletted image (Mode P) to RGB or RGBA.
  - `ImgUtils.batch_optimize_textures(cls, directory: str, **kwargs)` *(class)* — Batch optimizes all textures in a directory.
  - `ImgUtils.optimize_texture(cls, texture_path: str, output_dir: str = None, output_type: str = None, max_size: int = None, force_pot: bool = False, suffix_old: str = None, suffix_opt: str = None, old_files_folder: str = None, generate_mipmaps: bool = False, optimize_bit_depth: bool = True, check_existing: bool = False, map_type: str = None) -> str` *(class)* — Optimizes a texture by resizing, setting bit depth, and adjusting image type.
  - `ImgUtils.is_image_constant(cls, image: Union[str, PILImage.Image], tolerance: int = 0) -> Tuple[bool, Optional[Tuple[int, ...]]]` *(class)* — Check if an image is constant color.
  - `ImgUtils.get_base_texture_name(cls, filepath_or_filename: str) -> str` *(class)* — Extracts the base texture name from a filename or path,
  - `ImgUtils.extract_channels(cls, image_path: Union[str, 'Image.Image'], channel_config: Dict[str, Dict[str, Any]], output_dir: str = None, base_name: str = None, save: bool = True, **kwargs) -> Dict[str, Union[str, 'Image.Image']]` *(class)* — Generic channel extraction utility.

<a id="img_utils--map_converter"></a>
### `img_utils/map_converter.py`

- **[`class MapConverterSlots(ImgUtils)`](pythontk/pythontk/img_utils/map_converter.py#L11)**
  - `MapConverterSlots.source_dir(self)` *(property)* — Get the starting directory for file dialogs.
  - `MapConverterSlots.source_dir(self, value)` — Set the starting directory for file dialogs.
  - `MapConverterSlots.tb000_init(self, widget)`
  - `MapConverterSlots.tb000(self, widget)` — Optimize a texture map(s)
  - `MapConverterSlots.tb001_init(self, widget)`
  - `MapConverterSlots.tb001(self, widget)` — Batch converts Spec/Gloss maps to PBR Metal/Rough using MapFactory.
  - `MapConverterSlots.tb003_init(self, widget)` — Initialize a 'Bump to Normal' toolbutton with options.
  - `MapConverterSlots.tb003(self, widget)` — Bump/Height to Normal converter (single entry point with options).
  - `MapConverterSlots.b000(self)` — Convert DirectX to OpenGL
  - `MapConverterSlots.b001(self)` — Convert OpenGL to DirectX
  - `MapConverterSlots.b004(self)` — Batch pack Transparency into Albedo across texture sets.
  - `MapConverterSlots.b005(self)` — Batch pack Smoothness or Roughness into Metallic across texture sets.
  - `MapConverterSlots.b006(self)` — Unpack Metallic and Smoothness maps from MetallicSmoothness textures.
  - `MapConverterSlots.b007(self)` — Unpack Specular and Gloss maps from SpecularGloss textures.
  - `MapConverterSlots.b008(self)` — Batch pack Metallic (R), AO (G), and Smoothness (A) across texture sets.
  - `MapConverterSlots.b009(self)` — Unpack Metallic, AO, and Smoothness maps from MSAO textures.
  - `MapConverterSlots.b010(self)` — Convert Smoothness maps to Roughness maps.
  - `MapConverterSlots.b011(self)` — Convert Roughness maps to Smoothness maps.
  - `MapConverterSlots.b012(self)` — Batch prepare textures for PBR workflow using MapFactory.
- **[`class MapConverterUi`](pythontk/pythontk/img_utils/map_converter.py#L678)**

<a id="img_utils--map_factory"></a>
### `img_utils/map_factory.py`

Texture Map Factory for PBR workflow preparation - Refactored.

- **[`class MapConversion`](pythontk/pythontk/img_utils/map_factory.py#L58)** — Defines a single map conversion operation.
- **[`class ConversionRegistry`](pythontk/pythontk/img_utils/map_factory.py#L67)** — Central registry for all map type conversions.
  - `ConversionRegistry.add_plugin(self, cls)` — Register a class to be scanned for conversions later.
  - `ConversionRegistry.register(self, target_type: Union[str, MapConversion], source_types: Union[str, List[str]] = None, converter: Callable = None, priority: int = 0)` — Register a new conversion strategy.
  - `ConversionRegistry.register_from_class(self, cls)` — Register all decorated conversion methods from a class.
  - `ConversionRegistry.get_conversions_for(self, target_type: str) -> List[MapConversion]` — Get all conversions that can produce target type.
- **[`class TextureProcessor`](pythontk/pythontk/img_utils/map_factory.py#L172)** — Shared context and processor for all map operations.
  - `TextureProcessor.get_cached_image(self, path: str) -> 'Image.Image'` — Load an image with caching to avoid redundant disk I/O.
  - `TextureProcessor.save_map(self, image: Union[str, Any], map_type: str, suffix: str = None, optimize: bool = None, source_images: List[Union[str, Any]] = None) -> str` — Saves and optimizes a map, enforcing mode and naming conventions.
  - `TextureProcessor.resolve_map(self, *preferred_types: str, allow_conversion: bool = True) -> Optional[Union[str, 'Image.Image']]` — Intelligently resolve a map from inventory with fallback conversions.
  - `TextureProcessor.mark_used(self, *map_types: str)` — Mark map types as consumed.
  - `TextureProcessor.convert_specular_to_metallic(self, specular_path: Union[str, 'Image.Image']) -> 'Image.Image'`
  - `TextureProcessor.convert_smoothness_to_roughness(self, smoothness_path: Union[str, 'Image.Image']) -> 'Image.Image'`
  - `TextureProcessor.convert_roughness_to_smoothness(self, roughness_path: Union[str, 'Image.Image']) -> 'Image.Image'`
  - `TextureProcessor.convert_specular_to_roughness(self, specular_path: Union[str, 'Image.Image']) -> 'Image.Image'`
  - `TextureProcessor.convert_dx_to_gl(self, dx_path: Union[str, 'Image.Image']) -> 'Image.Image'`
  - `TextureProcessor.convert_gl_to_dx(self, gl_path: Union[str, 'Image.Image']) -> 'Image.Image'`
  - `TextureProcessor.convert_bump_to_normal(self, bump_path: Union[str, 'Image.Image']) -> 'Image.Image'`
  - `TextureProcessor.extract_gloss_from_spec(self, specular_path: Union[str, 'Image.Image']) -> 'Image.Image'`
  - `TextureProcessor.copy_map(self, source_path: Union[str, 'Image.Image'], target_type: str) -> Union[str, 'Image.Image']` — Simple copy/rename for compatible maps (e.g.
  - `TextureProcessor.unpack_metallic_smoothness(self, source_path: Union[str, 'Image.Image']) -> None` — Helper to unpack and cache results.
  - `TextureProcessor.get_metallic_from_packed(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
  - `TextureProcessor.get_smoothness_from_packed(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
  - `TextureProcessor.get_roughness_from_packed(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
  - `TextureProcessor.unpack_msao(self, source_path: Union[str, 'Image.Image']) -> None` — Helper to unpack MSAO and cache results.
  - `TextureProcessor.get_metallic_from_msao(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
  - `TextureProcessor.get_smoothness_from_msao(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
  - `TextureProcessor.get_roughness_from_msao(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
  - `TextureProcessor.get_ao_from_msao(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
  - `TextureProcessor.unpack_orm(self, source_path: Union[str, 'Image.Image']) -> None` — Helper to unpack ORM and cache results.
  - `TextureProcessor.get_ao_from_orm(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
  - `TextureProcessor.get_roughness_from_orm(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
  - `TextureProcessor.get_smoothness_from_orm(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
  - `TextureProcessor.get_metallic_from_orm(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
  - `TextureProcessor.unpack_albedo_transparency(self, source_path: Union[str, 'Image.Image']) -> None` — Helper to unpack Albedo+Transparency and cache results.
  - `TextureProcessor.get_base_color_from_albedo_transparency(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
  - `TextureProcessor.get_opacity_from_albedo_transparency(self, source_path: Union[str, 'Image.Image']) -> Union[str, 'Image.Image']`
  - `TextureProcessor.create_orm_map(self, inventory: Dict[str, Union[str, 'Image.Image']]) -> 'Image.Image'` — Create ORM map from components.
  - `TextureProcessor.create_mask_map(self, inventory: Dict[str, Union[str, 'Image.Image']]) -> 'Image.Image'` — Create Mask Map (MSAO) from components.
  - `TextureProcessor.create_metallic_smoothness_map(self, inventory: Dict[str, Union[str, 'Image.Image']]) -> 'Image.Image'` — Create Metallic-Smoothness map from components.
- **[`class WorkflowHandler(ABC)`](pythontk/pythontk/img_utils/map_factory.py#L941)** — Abstract base for workflow-specific map processing.
  - `WorkflowHandler.can_handle(self, context: TextureProcessor) -> bool` — Check if this handler should process the workflow.
  - `WorkflowHandler.process(self, context: TextureProcessor) -> Optional[str]` — Process and return the output map path.
  - `WorkflowHandler.get_consumed_types(self) -> List[str]` — Return list of map types this handler consumes.
  - `WorkflowHandler.is_explicitly_requested(self, context: TextureProcessor, map_type: str) -> bool` — Check if a map type is explicitly requested in the config.
- **[`class ORMMapHandler(WorkflowHandler)`](pythontk/pythontk/img_utils/map_factory.py#L972)** — Handles Unreal Engine / glTF ORM packing.
  - `ORMMapHandler.can_handle(self, context: TextureProcessor) -> bool`
  - `ORMMapHandler.process(self, context: TextureProcessor) -> Optional[str]`
  - `ORMMapHandler.get_consumed_types(self) -> List[str]`
- **[`class MaskMapHandler(WorkflowHandler)`](pythontk/pythontk/img_utils/map_factory.py#L1061)** — Handles Unity HDRP Mask Map (MSAO).
  - `MaskMapHandler.can_handle(self, context: TextureProcessor) -> bool`
  - `MaskMapHandler.process(self, context: TextureProcessor) -> Optional[str]`
  - `MaskMapHandler.get_consumed_types(self) -> List[str]`
- **[`class MetallicSmoothnessHandler(WorkflowHandler)`](pythontk/pythontk/img_utils/map_factory.py#L1172)** — Handles packed Metallic+Smoothness.
  - `MetallicSmoothnessHandler.can_handle(self, context: TextureProcessor) -> bool`
  - `MetallicSmoothnessHandler.process(self, context: TextureProcessor) -> Optional[str]`
  - `MetallicSmoothnessHandler.get_consumed_types(self) -> List[str]`
- **[`class SeparateMetallicRoughnessHandler(WorkflowHandler)`](pythontk/pythontk/img_utils/map_factory.py#L1257)** — Handles separate metallic and roughness maps.
  - `SeparateMetallicRoughnessHandler.can_handle(self, context: TextureProcessor) -> bool`
  - `SeparateMetallicRoughnessHandler.process(self, context: TextureProcessor) -> List[str]` — Returns list since this produces multiple maps.
  - `SeparateMetallicRoughnessHandler.get_consumed_types(self) -> List[str]`
- **[`class BaseColorHandler(WorkflowHandler)`](pythontk/pythontk/img_utils/map_factory.py#L1298)** — Handles base color / albedo with optional packing.
  - `BaseColorHandler.can_handle(self, context: TextureProcessor) -> bool`
  - `BaseColorHandler.process(self, context: TextureProcessor) -> Optional[str]`
  - `BaseColorHandler.get_consumed_types(self) -> List[str]`
- **[`class NormalMapHandler(WorkflowHandler)`](pythontk/pythontk/img_utils/map_factory.py#L1438)** — Handles normal map format conversion.
  - `NormalMapHandler.can_handle(self, context: TextureProcessor) -> bool`
  - `NormalMapHandler.process(self, context: TextureProcessor) -> Optional[str]`
  - `NormalMapHandler.get_consumed_types(self) -> List[str]`
- **[`class OutputFallbackHandler(WorkflowHandler)`](pythontk/pythontk/img_utils/map_factory.py#L1601)** — Handles outputting fallback maps for failed requests.
  - `OutputFallbackHandler.can_handle(self, context: TextureProcessor) -> bool`
  - `OutputFallbackHandler.process(self, context: TextureProcessor) -> List[str]`
  - `OutputFallbackHandler.get_consumed_types(self) -> List[str]`
- **[`class MapFactory(LoggingMixin)`](pythontk/pythontk/img_utils/map_factory.py#L1669)** — Refactored factory with pluggable workflow system.
  - `MapFactory.register_conversions(cls, registry: ConversionRegistry)` *(class)* — Register all standard PBR conversions.
  - `MapFactory.resolve_map_type(cls, file: str, key: bool = True, validate: str = None) -> str` *(class)* — Resolves the map type from a filename or alias using `map_types`.
  - `MapFactory.resolve_texture_filename(cls, texture_path: str, map_type: str, prefix: str = None, suffix: str = None, ext: str = None) -> str` *(class)* — Generates a correctly formatted filename while preserving the original suffix and file extension.
  - `MapFactory.get_base_texture_name(cls, filepath_or_filename: str) -> str` *(class)* — Extracts the base texture name from a filename or path,
  - `MapFactory.group_textures_by_set(cls, image_paths: List[str]) -> Dict[str, List[str]]` *(class)* — Groups texture maps into sets based on matching base names.
  - `MapFactory.filter_images_by_type(cls, files, types='')` *(class)* — Parameters:
  - `MapFactory.sort_images_by_type(cls, files: Union[List[Union[str, Tuple[str, Any]]], Dict[str, Any]]) -> Dict[str, List[Union[str, Tuple[str, Any]]]]` *(class)* — Sort image files by map type based on the input format.
  - `MapFactory.contains_map_types(cls, files, map_types)` *(class)* — Check if the given images contain the given map types.
  - `MapFactory.is_normal_map(cls, file)` *(class)* — Check the map type for one of the normal values in map_types.
  - `MapFactory.register_handler(cls, handler_class: Type[WorkflowHandler])` *(class)* — Register a custom workflow handler (extensibility).
  - `MapFactory.register_conversion(cls, conversion: MapConversion)` *(class)* — Register a custom map conversion (extensibility).
  - `MapFactory.get_map_fallbacks(cls, map_type: str) -> Tuple[str, ...]` *(class)* — Get fallback map types for a given map type.
  - `MapFactory.get_precedence_rules(cls) -> Dict[str, List[str]]` *(class)* — Returns a dictionary of map precedence rules.
  - `MapFactory.filter_redundant_maps(cls, sorted_maps: Dict[str, List[str]]) -> None` *(class)* — Filters out maps that are rendered redundant by other present maps (e.g.
  - `MapFactory.prepare_maps(cls, source: Union[str, List[str]], output_dir: str = None, group_by_set: bool = True, max_workers: int = 1, progress_callback: Callable = None, **kwargs) -> Union[List[str], Dict[str, List[str]]]` *(class)* — Main factory method.
  - `MapFactory.pack_transparency_into_albedo(cls, albedo_map_path: str, alpha_map_path: str, output_dir: Optional[str] = None, suffix: Optional[str] = '_AlbedoTransparency', invert_alpha: bool = False, output_path: Optional[str] = None, save: bool = True) -> Union[str, 'Image.Image']` *(class)* — Combines an albedo texture with a transparency map by packing the transparency into the alpha chann…
  - `MapFactory.pack_smoothness_into_metallic(cls, metallic_map_path: str, alpha_map_path: str, output_dir: str = None, suffix: str = '_MetallicSmoothness', invert_alpha: bool = False, output_path: str = None, save: bool = True) -> Union[str, 'Image.Image']` *(class)* — Packs a smoothness (or inverted roughness) texture into the alpha channel of a metallic texture map.
  - `MapFactory.detect_normal_map_format(cls, image: Union[str, 'Image.Image'], threshold: float = 0.1) -> Optional[str]` *(class)* — Detects if a normal map is OpenGL (Y+) or DirectX (Y-) based on surface integrability.
  - `MapFactory.convert_normal_map_format(cls, file: str, target_format: str, output_path: str = None, save: bool = True, **kwargs) -> Union[str, 'Image.Image']` *(class)* — Converts a normal map between OpenGL (Y+) and DirectX (Y-) formats by inverting the green channel.
  - `MapFactory.convert_bump_to_normal(cls, bump_map: Union[str, 'Image.Image'], output_path: str = None, intensity: float = 1.0, output_format: str = 'opengl', smooth_filter: bool = True, filter_radius: float = 0.5, edge_wrap: bool = False, save: bool = True, **kwargs) -> Union[str, 'Image.Image']` *(class)* — Convert a bump/height map to a tangent-space normal map.
  - `MapFactory.extract_gloss_from_spec(cls, specular_map: str, channel: str = 'A') -> Union['Image.Image', None]` *(class)* — Extracts gloss from a specific channel in the specular map.
  - `MapFactory.convert_spec_gloss_to_pbr(cls, specular_map: Union[str, 'Image.Image'], glossiness_map: Union[str, 'Image.Image'], diffuse_map: Union[str, 'Image.Image'] = None, output_dir: str = None, convert_diffuse_to_albedo: bool = False, output_type: str = None, image_size: Optional[int] = None, optimize_bit_depth: bool = True, write_files: bool = False) -> Union[Tuple['Image.Image', 'Image.Image', 'Image.Image'], Tuple[str, str, str]]` *(class)* — Converts Specular/Glossiness maps to PBR Metal/Rough.
  - `MapFactory.create_base_color_from_spec(cls, diffuse: Union[str, 'Image.Image'], spec: Union[str, 'Image.Image'], metalness: Union[str, 'Image.Image'], conserve_energy: bool = True, metal_darkening: float = 0.22) -> 'Image.Image'` *(class)* — Computes Base Color from Specular workflow with better metal handling.
  - `MapFactory.create_metallic_from_spec(cls, specular_map: Union[str, 'Image.Image'], glossiness_map: Union[str, 'Image.Image'] = None, threshold: int = 55, softness: float = 0.2) -> 'Image.Image'` *(class)* — Creates a metallic map from a specular (and optional glossiness) map.
  - `MapFactory.create_roughness_from_spec(cls, specular_map: Union[str, 'Image.Image'], glossiness_map: Union[str, 'Image.Image'] = None) -> 'Image.Image'` *(class)* — Estimates roughness from a specular map.
  - `MapFactory.convert_base_color_to_albedo(cls, base_color: 'Image.Image', metalness: 'Image.Image') -> 'Image.Image'` *(class)* — Converts a Base Color map to a true Albedo map by:
  - `MapFactory.get_converted_map(map_type: str, available: dict) -> Optional[Any]` *(static)* — Get the converted map based on the given map type and available maps.
  - `MapFactory.pack_msao_texture(cls, metallic_map_path: str, ao_map_path: Optional[str], alpha_map_path: Optional[str], detail_map_path: Optional[str] = None, output_dir: str = None, suffix: str = '_MSAO', invert_alpha: bool = False, output_path: str = None, save: bool = True) -> Union[str, 'Image.Image']` *(class)* — Packs Metallic (R), AO (G), Detail (B), and Smoothness/Roughness (A) into a single MSAO texture.
  - `MapFactory.convert_smoothness_to_roughness(cls, smoothness_path: str, output_dir: str = None, save: bool = True, **kwargs) -> Union[str, 'Image.Image']` *(class)* — Convert a Smoothness map to a Roughness map by inverting the grayscale values.
  - `MapFactory.convert_roughness_to_smoothness(cls, roughness_path: str, output_dir: str = None, save: bool = True, **kwargs) -> Union[str, 'Image.Image']` *(class)* — Convert a Roughness map to a Smoothness map by inverting the grayscale values.
  - `MapFactory.unpack_orm_texture(cls, orm_map_path: str, output_dir: str = None, ao_suffix: str = '_AO', roughness_suffix: str = '_Roughness', metallic_suffix: str = '_Metallic', invert_roughness: bool = False, save: bool = True, **kwargs) -> Union[Tuple[str, str, str], Tuple['Image.Image', 'Image.Image', 'Image.Image']]` *(class)* — Unpacks AO (R), Roughness (G), and Metallic (B) maps from a combined ORM texture.
  - `MapFactory.unpack_msao_texture(cls, msao_map_path: str, output_dir: str = None, metallic_suffix: str = '_Metallic', ao_suffix: str = '_AO', smoothness_suffix: str = '_Smoothness', invert_smoothness: bool = False, save: bool = True, **kwargs) -> Union[Tuple[str, str, str], Tuple['Image.Image', 'Image.Image', 'Image.Image']]` *(class)* — Unpacks Metallic (R), AO (G), and Smoothness (A) maps from a combined MSAO texture.
  - `MapFactory.unpack_albedo_transparency(cls, albedo_map_path: str, output_dir: str = None, base_color_suffix: str = '_BaseColor', opacity_suffix: str = '_Opacity', save: bool = True, **kwargs) -> Union[Tuple[str, str], Tuple['Image.Image', 'Image.Image']]` *(class)* — Unpacks Base Color (RGB) and Opacity (A) from an Albedo+Transparency map.
  - `MapFactory.unpack_metallic_smoothness(cls, map_path: str, output_dir: str = None, metallic_suffix: str = '_Metallic', smoothness_suffix: str = '_Smoothness', invert_smoothness: bool = False, save: bool = True, **kwargs) -> Union[Tuple[str, str], Tuple['Image.Image', 'Image.Image']]` *(class)* — Unpacks Metallic (RGB) and Smoothness (A) from a combined map.
  - `MapFactory.unpack_specular_gloss(cls, map_path: str, output_dir: str = None, specular_suffix: str = '_Specular', gloss_suffix: str = '_Glossiness', invert_gloss: bool = False, save: bool = True, **kwargs) -> Union[Tuple[str, str], Tuple['Image.Image', 'Image.Image']]` *(class)* — Unpacks Specular (RGB) and Glossiness (A) from a combined map.

<a id="img_utils--map_packer"></a>
### `img_utils/map_packer.py`

- **[`class MapPackerSlots(ImgUtils)`](pythontk/pythontk/img_utils/map_packer.py#L9)**
  - `MapPackerSlots.header_init(self, widget)` — Configure the header menu with presets for common packed map types.
  - `MapPackerSlots.source_dir(self)` *(property)*
  - `MapPackerSlots.source_dir(self, value)`
  - `MapPackerSlots.b000(self)` — Batch pack up to 4 channels into RGBA maps across texture sets.
  - `MapPackerSlots.b001(self)` — Open the last output directory in the system file explorer.
- **[`class MapPackerUi`](pythontk/pythontk/img_utils/map_packer.py#L253)**

<a id="img_utils--map_registry"></a>
### `img_utils/map_registry.py`

- **[`class WF`](pythontk/pythontk/img_utils/map_registry.py#L7)** — Workflow identifiers.
- **[`class MapType`](pythontk/pythontk/img_utils/map_registry.py#L43)** — Defines the properties of a texture map type.
- **[`class MapRegistry(SingletonMixin)`](pythontk/pythontk/img_utils/map_registry.py#L65)** — Central registry for map type definitions.
  - `MapRegistry.get(self, name: str) -> Optional[MapType]` — Get a map type by name.
  - `MapRegistry.resolve_type_from_path(self, path: str) -> Optional[str]` — Resolve the map type key from a file path.
  - `MapRegistry.get_workflow_presets(self) -> Dict[str, Dict[str, Any]]` — Generate the workflow presets dictionary.
  - `MapRegistry.get_map_types(self) -> Dict[str, Tuple[str, ...]]` — Generate the dictionary format for ImgUtils.map_types.
  - `MapRegistry.get_fallbacks(self) -> Dict[str, Tuple[str, ...]]` — Generate the input fallback dictionary.
  - `MapRegistry.get_output_fallbacks(self) -> Dict[str, Tuple[str, ...]]` — Generate the output fallback dictionary.
  - `MapRegistry.get_precedence_rules(self) -> Dict[str, List[str]]` — Generate the precedence rules dictionary.
  - `MapRegistry.get_scale_as_mask_types(self) -> List[str]` — Get list of map types that should be scaled as masks.
  - `MapRegistry.get_passthrough_maps(self) -> List[str]` — Get list of maps that should be passed through if not consumed.
  - `MapRegistry.get_map_backgrounds(self) -> Dict[str, Tuple[int, int, int, int]]` — Generate the map backgrounds dictionary.
  - `MapRegistry.get_map_modes(self) -> Dict[str, str]` — Generate the map modes dictionary.
  - `MapRegistry.resolve_config(self, config: Union[str, Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]` — Resolve configuration from presets, dicts, and kwargs.

<a id="iter_utils--_iter_utils"></a>
### `iter_utils/_iter_utils.py`

- **[`class IterUtils(HelpMixin)`](pythontk/pythontk/iter_utils/_iter_utils.py#L9)**
  - `IterUtils.make_iterable(x: Any, snapshot: bool = False) -> Iterable` *(static)* — Convert the given object to an iterable, unless it's a string, bytes, or bytearray.
  - `IterUtils.nested_depth(cls, lst, typ=(list, set, tuple))` *(class)* — Get the maximum nested depth of any sub-lists of the given list.
  - `IterUtils.flatten(cls, lst, return_type: Optional[type] = None)` *(class)* — Flatten arbitrarily nested lists.
  - `IterUtils.collapse_integer_sequence(lst, limit=None, compress=True, to_string=True)` *(static)* — Converts a list of integers into a compressed string representation of sequences.
  - `IterUtils.bit_array_to_list(bit_array)` *(static)* — Convert a binary bit_array to a python list.
  - `IterUtils.insert_into_dict(original_dict: Dict[Any, Any], key: Any, value: Any, index: int = 0, in_place: bool = False) -> Dict[Any, Any]` *(static)* — Insert a key-value pair at a specified index in a dictionary.
  - `IterUtils.rindex(itr, item)` *(static)* — Get the index of the first item to match the given item
  - `IterUtils.indices(itr, value)` *(static)* — Get the index of each element of a list matching the given value.
  - `IterUtils.remove_duplicates(lst, trailing=True)` *(static)* — Removes duplicate entries from a list while maintaining the original order of the remaining items.
  - `IterUtils.filter_results(func: Callable) -> Callable` — Decorator to filter the results of a function that returns a list or dictionary.
  - `IterUtils.filter_list(cls, lst: List, inc: Optional[Union[str, List]] = None, exc: Optional[Union[str, List]] = None, map_func: Optional[Callable] = None, check_unmapped: bool = False, nested_as_unit: bool = False, basename_only: bool = False, ignore_case: bool = False, delimiter: str = ',', match_all: bool = False) -> List` *(class)* — Filters the given list based on inclusion/exclusion criteria using shell-style wildcards.
  - `IterUtils.filter_dict(cls, dct: Dict, keys: bool = False, values: bool = False, **kwargs) -> Dict` *(class)* — Filter the given dictionary.
  - `IterUtils.split_list(lst, into)` *(static)* — Split a list into parts.
  - `IterUtils.find_flat_interior_indices(values, value_tolerance=1e-05)` *(static)* — Return indices of redundant interior keys in flat segments.

<a id="math_utils--_math_utils"></a>
### `math_utils/_math_utils.py`

- **[`class MathUtils(HelpMixin)`](pythontk/pythontk/math_utils/_math_utils.py#L15)**
  - `MathUtils.linear_sum_assignment(cost_matrix: Sequence[Sequence[float]], maximize: bool = False) -> Tuple[List[int], List[int]]` *(static)* — Solve the linear sum assignment problem (Hungarian algorithm).
  - `MathUtils.get_pca_transform(points_a: 'np.ndarray', points_b: 'np.ndarray', tolerance: float = 0.001, robust: bool = False, sample_size: int = 500, symmetry_threshold: float = 0.1) -> Optional[List[float]]` *(static)* — Calculate the transformation matrix to align points_b to points_a using PCA axis alignment.
  - `MathUtils.kmeans_clustering(points: Sequence[Sequence[float]], k: int, max_iterations: int = 30, seed_indices: Optional[List[int]] = None) -> List[List[int]]` *(static)* — Perform K-Means clustering on a set of points.
  - `MathUtils.kmeans_1d(values: Sequence[float], k: int = 3, max_iterations: int = 10) -> Tuple[List[float], List[List[float]]]` *(static)* — Perform 1D K-Means clustering to find natural breakpoints in scalar data.
  - `MathUtils.get_kmeans_threshold(cls, values: Sequence[float], k: int = 3) -> float` *(class)* — Use K-Means to find an adaptive threshold separating "parts" from "bodies".
  - `MathUtils.move_decimal_point(num, places)` *(static)* — Move the decimal place in a given number.
  - `MathUtils.get_vector_from_two_points(a: List[float], b: List[float]) -> Tuple[float, float, float]` *(static)* — Get a directional vector from a given start and end point.
  - `MathUtils.clamp(n=0.0, minimum=0.0, maximum=1.0)` *(static)* — Clamps the value x between min and max.
  - `MathUtils.clamp_range(start, end, clamp_start=None, clamp_end=None, validate=True)` *(static)* — Clamp a numeric range (start, end) to optional boundaries with validation.
  - `MathUtils.normalize(cls, vector, amount=1)` *(class)* — Normalize a 2 or 3 dimensional vector.
  - `MathUtils.get_magnitude(vector)` *(static)* — Get the magnatude (length) of a given vector.
  - `MathUtils.dot_product(cls, v1, v2, normalize_input=False)` *(class)* — Returns the dot product of two 3D float arrays.
  - `MathUtils.cross_product(cls, a, b, c=None, normalize=0)` *(class)* — Get the cross product of two vectors, using two 3d vectors, or 3 points.
  - `MathUtils.move_point_relative(cls, p, d, v=None)` *(class)* — Move a point relative to it's current position.
  - `MathUtils.move_point_relative_along_vector(cls, a, b, vect, dist, toward=True)` *(class)* — Move a point (a) along a given vector toward or away from a given point (b).
  - `MathUtils.distance_between_points(a: Tuple[float, ...], b: Tuple[float, ...]) -> float` *(static)* — Calculates the Euclidean distance between two points in N-dimensional space.
  - `MathUtils.get_center_of_two_points(a: List[float], b: List[float]) -> Tuple[float, float, float]` *(static)* — Get the point in the middle of two given points.
  - `MathUtils.get_angle_from_two_vectors(cls, v1, v2, degree=False)` *(class)* — Get an angle from two given vectors.
  - `MathUtils.get_angle_from_three_points(a, b, c, degree=False)` *(static)* — Get the opposing angle from 3 given points.
  - `MathUtils.get_two_sides_of_asa_triangle(a1, a2, s, unit='degrees')` *(static)* — Get the length of two sides of a triangle, given two angles, and the length of the side in-between.
  - `MathUtils.xyz_rotation(cls, theta, axis, rotation=[], degree=False)` *(class)* — Get the rotation about the X,Y,Z axes (in rotation) given
  - `MathUtils.lerp(start: float, end: float, t: float) -> float` *(static)* — Linear interpolation between two values.
  - `MathUtils.evaluate_sampled_progress(time_value: float, sample_times: Sequence[float], progress: Sequence[float], tolerance: float = 1e-06) -> float` *(static)* — Interpolate normalized progress from sampled time/progress pairs.
  - `MathUtils.generate_geometric_sequence(base_value: int, terms: int, common_ratio: float = 2.0) -> List[int]` — Generate a geometric sequence.
  - `MathUtils.remap(value: Union[float, List[Any], Tuple[Any, ...], 'np.ndarray'], old_range: Tuple[float, float], new_range: Tuple[float, float], clamp: bool = False) -> Union[float, List[Any], Tuple[Any, ...], 'np.ndarray']` *(static)* — Remaps a value, list, or tuple of varying sizes from one range to another.
  - `MathUtils.calculate_curve_length(centerline_points: List[List[float]]) -> float` *(static)* — Calculates the total length of the centerline path.
  - `MathUtils.get_point_on_centerline(centerline_points: List[List[float]], param: float) -> List[float]` *(static)* — Returns the interpolated point along the centerline.
  - `MathUtils.dist_points_along_centerline(cls, centerline: List[List[float]], num_points: int, reverse: bool = False, interpolation: Callable[[List[List[float]], float], List[float]] = None, start_offset: float = 0.0, end_offset: float = 0.0) -> List[List[float]]` *(class)* — Distributes points evenly along the centerline with optional offsets and custom interpolation.
  - `MathUtils.arrange_points_as_path(points: List[List[float]], closed_path: bool = False, distance_metric: Optional[Callable[[List[float], List[float]], float]] = None) -> List[List[float]]` *(static)* — Orders a list of points to form a continuous path.
  - `MathUtils.smooth_points(points: Sequence[Union[tuple, object]], window_size: int = 1) -> list` *(static)* — Apply a moving average to smooth a sequence of 3D points.
  - `MathUtils.nearest_power_of_two(value: int) -> int` *(static)* — Finds the nearest power of two for a given integer without using the math module.
  - `MathUtils.is_close_to_whole(value: float, tolerance: float = 0.0001) -> bool` *(static)* — Check if a float value is close to a whole number within tolerance.
  - `MathUtils.round_value(value: float, mode: str = 'none', max_distance: float = 1.5) -> Union[int, float]` *(static)* — General-purpose rounding function with multiple modes.
  - `MathUtils.round_to_preferred(value: float, max_distance: float = 1.5) -> int` *(static)* — Round to aesthetically pleasing 'round' numbers (conservative approach).
  - `MathUtils.round_to_aggressive_preferred(value: float) -> int` *(static)* — Round to aesthetically pleasing 'round' numbers (aggressive approach).
  - `MathUtils.hash_points(points, precision=4)` *(static)* — Hash the given list of point values.
  - `MathUtils.calculate_rotation_distance(r1_vals: Tuple[float, float, float], r2_vals: Tuple[float, float, float], bbox_points: Optional[List[Any]] = None, om_module: Optional[Any] = None) -> float` *(static)* — Calculate the effective rotation distance between two Euler rotations.

<a id="math_utils--progression"></a>
### `math_utils/progression.py`

- **[`class ProgressionCurves`](pythontk/pythontk/math_utils/progression.py#L10)** — A collection of mathematical progression curves for animations and transformations.
  - `ProgressionCurves.linear(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float` *(static)* — Simple linear progression f(x) = x.
  - `ProgressionCurves.exponential(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float` *(static)* — Exponential progression f(x) = x^curve.
  - `ProgressionCurves.logarithmic(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float` *(static)* — Logarithmic progression for smooth acceleration.
  - `ProgressionCurves.sine(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float` *(static)* — Sine-based progression for smooth curves.
  - `ProgressionCurves.ease_in(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float` *(static)* — Ease-in: slow start, fast end (acceleration).
  - `ProgressionCurves.ease_out(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float` *(static)* — Ease-out: fast start, slow end (deceleration).
  - `ProgressionCurves.ease_in_out(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float` *(static)* — Ease-in-out: slow start and end, fast middle (S-curve).
  - `ProgressionCurves.smooth_step(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float` *(static)* — Smooth step function using Hermite interpolation (3x² - 2x³).
  - `ProgressionCurves.bounce(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float` *(static)* — Bouncing effect using sine waves.
  - `ProgressionCurves.elastic(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float` *(static)* — Elastic effect with overshoot.
  - `ProgressionCurves.weighted(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float` *(static)* — Advanced weighted progression with bias control.
  - `ProgressionCurves.calculate_progression_factor(cls, index: int, total_count: int, weight_bias: float = 0.5, weight_curve: float = 1.0, calculation_mode: str = 'linear') -> float` *(class)* — Calculate a progression factor using various mathematical functions.
  - `ProgressionCurves.get_curve_function(cls, calculation_mode: str)` *(class)* — Get the curve function by name.
  - `ProgressionCurves.generate_curve_samples(cls, calculation_mode: str, num_samples: int = 100, weight_bias: float = 0.5, weight_curve: float = 1.0) -> List[float]` *(class)* — Generate a list of samples from a curve for visualization or analysis.

<a id="net_utils--_net_utils"></a>
### `net_utils/_net_utils.py`

- **[`class NetUtils`](pythontk/pythontk/net_utils/_net_utils.py#L16)** — General purpose network utilities.
  - `NetUtils.connect_rdp(host: str, username: str = None, password: str = None, width: int = None, height: int = None, fullscreen: bool = True, extra_settings: Dict[str, str] = None, save_credentials: bool = True)` *(static)* — Connect to a remote desktop using Windows RDP (mstsc.exe).
  - `NetUtils.is_port_open(host: str, port: int, timeout: float = 1.0) -> bool` *(static)* — Check if a TCP port is open on a host.
  - `NetUtils.get_local_ip() -> Optional[str]` *(static)* — Get the local IP address of this machine.

<a id="net_utils--credentials"></a>
### `net_utils/credentials.py`

- **[`class Credentials`](pythontk/pythontk/net_utils/credentials.py#L19)** — Abstractions for OS-level secure credential storage.
  - `Credentials.get_password(target_name: str) -> str` *(static)* — Retrieve a password from the OS secure store or environment.
  - `Credentials.get_credential(target_name: str) -> dict | None` *(static)* — Retrieve full credentials (username and password).
  - `Credentials.set_credential(target_name: str, username: str, password: str, persist: str = 'local_machine') -> bool` *(static)* — Save credentials to the OS secure store.

<a id="net_utils--ssh_client"></a>
### `net_utils/ssh_client.py`

- **[`class SSHClient`](pythontk/pythontk/net_utils/ssh_client.py#L15)** — A unified SSH Client wrapper around Paramiko that handles:
  - `SSHClient.connect(self)` — Establish the SSH connection.
  - `SSHClient.disconnect(self)` — Close the connection.
  - `SSHClient.execute(self, command: str, stream: bool = False, use_pty: bool = False, timeout: float = None) -> Union[Tuple[str, str, int], int]` — Execute a command on the remote server.
  - `SSHClient.upload_file(self, local_path: str, remote_path: str)` — Upload a file via SFTP.
  - `SSHClient.download_file(self, remote_path: str, local_path: str)` — Download a file via SFTP.

<a id="str_utils--_str_utils"></a>
### `str_utils/_str_utils.py`

- **[`class StrUtils(CoreUtils)`](pythontk/pythontk/str_utils/_str_utils.py#L10)**
  - `StrUtils.sanitize(text: Union[str, List[str]], replacement_char: str = '_', char_map: Optional[Dict[str, str]] = None, preserve_trailing: bool = False, preserve_case: bool = False, allow_consecutive: bool = False, return_original: bool = False) -> Union[str, Tuple[str, str], List[str], List[Tuple[str, str]]]` *(static)* — Sanitizes a string or a list of strings by replacing invalid characters.
  - `StrUtils.replace_placeholders(text: str, **kwargs) -> str` *(static)* — Replace placeholders in a string with provided values.
  - `StrUtils.replace_delimited(text: str, context: dict, prefix: str = '__', suffix: str = '__') -> str` *(static)* — Replace delimited placeholders in *text* using *context*.
  - `StrUtils.set_case(string, case='title')` *(static)* — Format the given string(s) in the given case.
  - `StrUtils.get_mangled_name(class_input, attribute_name)` *(static)* — Returns the mangled name for a private attribute of a class.
  - `StrUtils.get_matching_hierarchy_items(hierarchy_items, target, upstream=False, exact=False, downstream=False, reverse=False, delimiters='|')` *(static)* — Find the closest match(es) for a given 'target' string in a list of hierarchical strings.
  - `StrUtils.split_delimited_string(string: str, delimiter: str = '|', max_split: Optional[int] = None, occurrence: Optional[int] = None, strip_whitespace: bool = False, remove_empty: bool = False, func: Optional[Callable] = None) -> Union[List[str], Tuple[str, str]]` *(static)* — Split a delimited string with flexible control over the result format.
  - `StrUtils.get_text_between_delimiters(string, start_delim, end_delim, as_string=False)` *(static)* — Get any text between the specified start and end delimiters in the given string.
  - `StrUtils.insert(cls, src, ins, at, occurrence=1, before=False)` *(class)* — Insert character(s) into a string at a given location.
  - `StrUtils.rreplace(string, old, new='', count=None)` *(static)* — Replace occurrances in a string from right to left.
  - `StrUtils.truncate(string, length=75, mode='start', insert='..')` *(static)* — Shorten the given string to the given length.
  - `StrUtils.get_trailing_integers(string, inc=0, as_string=False)` *(static)* — Returns any integers from the end of the given string.
  - `StrUtils.find_str(find, strings, regex=False, ignore_case=False)` *(static)* — Filter for elements that containing the given string in a list of strings.
  - `StrUtils.find_str_and_format(cls, strings, to, fltr='', regex=False, ignore_case=False, return_orig_strings=False)` *(class)* — Expanding on the 'find_str' function: Find matches of a string in a list of strings and re-format t…
  - `StrUtils.format_suffix(string: str, suffix: str = '', strip: Union[str, List[str]] = '', strip_trailing_ints: bool = False, strip_trailing_alpha: bool = False) -> str` *(static)* — Re-format the suffix for the given string.
  - `StrUtils.time_stamp(filepath, stamp='%m-%d-%Y  %H:%M')` *(static)* — Attach or detach a modified timestamp and date to/from a given file path.

<a id="str_utils--fuzzy_matcher"></a>
### `str_utils/fuzzy_matcher.py`

- **[`class FuzzyMatcher`](pythontk/pythontk/str_utils/fuzzy_matcher.py#L7)** — Fuzzy matching utilities for object names and hierarchical structures.
  - `FuzzyMatcher.get_base_name(name: str) -> str` *(static)* — Remove trailing digits from name to get base name.
  - `FuzzyMatcher.find_best_match(target_name: str, available_names: List[str], score_threshold: float = 0.5) -> Optional[Tuple[str, float]]` *(static)* — Find best fuzzy match for target name from available candidates.
  - `FuzzyMatcher.find_all_matches(target_names: List[str], available_names: List[str], score_threshold: float = 0.5) -> Dict[str, Tuple[str, float]]` *(static)* — Find fuzzy matches for multiple target names.
  - `FuzzyMatcher.find_trailing_digit_matches(missing_paths: List[str], extra_paths: List[str], path_separator: str = '|') -> Tuple[List[Dict[str, str]], List[str], List[str]]` *(static)* — Find fuzzy matches specifically for trailing digit differences in hierarchical paths.
  - `FuzzyMatcher.calculate_levenshtein_distance(s1: str, s2: str) -> int` *(static)* — Calculate Levenshtein (edit) distance between two strings.
  - `FuzzyMatcher.similarity_from_distance(s1: str, s2: str) -> float` *(static)* — Calculate similarity score from Levenshtein distance.

<a id="vid_utils--_vid_utils"></a>
### `vid_utils/_vid_utils.py`

- **[`class VidUtils(HelpMixin)`](pythontk/pythontk/vid_utils/_vid_utils.py#L13)**
  - `VidUtils.get_frame_rate(cls, value: Union[str, float, int]) -> Union[float, str]` *(class)* — Converts between frame rate names and values.
  - `VidUtils.resolve_ffmpeg(cls, required: bool = True, auto_install: bool = False) -> Optional[str]` *(class)* — Finds FFmpeg executable path in system path or managed installs.
  - `VidUtils.get_video_frame_rate(cls, filepath: str) -> float` *(class)* — Extracts frame rate from a video file using FFmpeg.
  - `VidUtils.compress_video(cls, input_filepath: str, output_filepath: str = None, frame_rate: Union[float, int] = None, delete_original: bool = False, **ffmpeg_options) -> Union[str, None]` *(class)* — Compresses a video file using FFmpeg.
