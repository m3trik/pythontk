# core_utils

Shared infrastructure — the non-data-type half of pythontk. Where the `*_utils` siblings hold general primitives placed by data type (strings, images, math, …), this package holds the *mechanisms* the ecosystem runs on: mixins, package bootstrap, app orchestration, config stores, pipeline primitives, and the shared domain engines.

Deliberately flat: these module paths are import contracts for every downstream package (uitk, mayatk, blendertk, tentacle, extapps), so modules are not re-nested for cosmetics.

Everything here is reachable from the package root (`import pythontk as ptk; ptk.LoggingMixin`) unless marked otherwise below. Full signatures: [`API_INDEX.md`](../../API_INDEX.md) / [`API_REGISTRY.md`](../../API_REGISTRY.md).

## Class infrastructure

| Module | Key symbols | What it does |
|---|---|---|
| `_core_utils` | `CoreUtils` | Decorators and reflection helpers: `cached_property`, `listify` (broadcast a function over list-like args, optionally threaded), attribute get/set, `format_return`. |
| `logging_mixin` | `LoggingMixin` | Class-scoped logging: extra levels (SUCCESS/RESULT/NOTICE/PROGRESS), spam-guarded `*_once` variants, boxes/groups/tables, managed file tee, capped ring buffer. |
| `help_mixin` | `HelpMixin` | `.help()` / `.source()` / `.signature()` introspection on any class; the *dynamic* producer of `SymbolRecord`s (the registry generator is the static one). |
| `class_property` | `ClassProperty` | Class-level properties (replacement for the removed `@classmethod @property` stacking). |
| `singleton_mixin` | `SingletonMixin` | Singletons keyed per `(class, singleton_key)` — subclasses sharing a key never collide. |
| `namedtuple_container` | `NamedTupleContainer` | Generic named-tuple collection with querying, dynamic field access, and CSV round-trip. |
| `namespace_handler` | `NamespaceHandler` | Lazily-resolved attribute namespace for an owner object, with `Placeholder` deferred construction. |
| `color` | `Color`, `ColorPair`, `Palette` | Stdlib-only color primitives (a lone value primitive lodges here — see the placement note in `__init__.py`). |

## App / process orchestration

| Module | Key symbols | What it does |
|---|---|---|
| `app_launcher` | `AppLauncher` | Cross-platform app discovery + launching: detached `launch` vs blocking `run`, executable scanning, session-aware launching, process queries/termination. |
| `app_installer` | `AppInstaller` | Download, extract, and version-track external tool binaries (ffmpeg, toktx, …) from caller-supplied platform definitions; stdlib-only, `.installed.json` catalog. |
| `app_handoff` | `HandoffBridge`, `Deliverer`, `AppSpec` | The Qt-free/DCC-free "export something and hand it to an app" backbone — one Template-Method flow (`resolve → preflight → produce → deliver → ingest`) with a per-mode `Deliverer` strategy. Base of the ecosystem's Maya/Blender/Marmoset/Substance bridges. |
| `script_run` | `ScriptRunner`, `ScriptRunResult` | Blocking script-run-to-artifact: success is judged by the produced artifact, not the exit code (DCC batch interpreters routinely crash in teardown after succeeding). |
| `script_template` | `ScriptTemplate` *(module-path import)* | On-disk script-template discovery + `__KEY__` rendering; declares the ecosystem-wide hand-off mode vocabulary (`SEND_TO`, `SAVE_AS`, `ROUND_TRIP`) that every bridge imports — it is an on-disk contract, so there is exactly one copy. |
| `process_stream` | `OutputStream`, `ProcessReader`, `LogTailer` | Composable line-stream primitives: thread-safe multi-consumer pub/sub with replayable history + `wait_for`, subprocess PIPE reader, rotation-aware log tailer. |
| `execution_monitor` | `ExecutionMonitor` | Wraps long-running operations with threshold-escalated dialogs, spinner/task indicators, and an external watchdog; cancellation is delegated to a `CancelScope`. |
| `cancel_scope` | `CancelScope`, `OperationCancelled` | One cooperative cancellation object shared by every cancel affordance: *push* (`cancel()` from any thread) + *pull* (sources polled at the operation's own checkpoints), consumable bool-style (`tick()`) or ambient exception-style (`CancelScope.check()` via `ContextVar`). |

**The three hand-off shapes.** `HandoffBridge` owns one invariant flow — `resolve → preflight → produce → deliver → ingest` — with the delivery step a per-mode `Deliverer` strategy, so three hand-off shapes come off one export pipeline: `send()` (`SEND_TO`, detached launch), `save_as()` (`SAVE_AS`, blocking run that keeps a native file of the target's format), and `round_trip()` (`ROUND_TRIP`, blocking run whose intermediate artifact is folded back onto the host's own objects — the target either edits the payload in place, as RizomUV does, or writes a new artifact, as mayatk's Blender lightmap bake does). The mode strings are declared once, in `script_template.py`, and every bridge imports them: they are an on-disk contract (each template's `BRIDGE_MODES` tuple), and a local copy would be a second dialect of a file format.

## Config & persistence

| Module | Key symbols | What it does |
|---|---|---|
| `user_config` | `UserConfig` | Resolve one JSON config doc by deep-merging a user file over a shipped default — Qt-free twin of uitk's per-user config root. |
| `preset_store` | `PresetStore` | Two-tier named-preset store (read-only built-in dir + writable user dir; user shadows built-in). uitk's `PresetManager` is a GUI over it. |
| `schema_spec` | `SchemaSpec` | Dataclass-declared schema for JSON/YAML template files: one definition derives `validate` (errors vs tolerated warnings), `skeleton`, and `to_markdown` reference docs. |
| `template_set` | `TemplateSet` | Binds a `PresetStore` (storage SSoT) to a `SchemaSpec` (shape SSoT): a discoverable, user-extensible set of schema-validated template files. |

## Package / dev infrastructure

| Module | Key symbols | What it does |
|---|---|---|
| `module_resolver` | `bootstrap_package` *(direct import)* | The lazy attribute-resolution machinery behind every ecosystem package root (`DEFAULT_INCLUDE` → lazy `__getattr__` → derived `__all__`). |
| `module_reloader` | `ModuleReloader`, `reload_package` | Hot-reload a package and its submodules in dependency order; returns a `ReloadReport`. |
| `package_manager` | `PackageManager` | pip wrapper: install/uninstall/list/update, version checks, concurrent latest-index lookups. |
| `git` | `Git` | Thin subprocess git wrapper scoped to one repo, with uniform dry-run/logging/error checking. |
| `cli` | `CLI` | Consistent `argparse` builders shared across scripts (e.g. the standard SSH connection argument group). |
| `symbol_record` | `SymbolRecord` | The shared public-API symbol shape produced by both the static registry generator and the dynamic `HelpMixin`; compared in the runtime-vs-static drift gate. |
| `doc_audit` | `DocAudit` | Markdown-example rot gate: extracts fenced code blocks and validates attribute chains + keyword arguments against the live package. Backs the README gates in `test/test_doc_audit.py`; downstream repos can gate their own docs with it. |
| `status_badge` | `StatusBadge` | Shields.io badges embedded in a markdown file; sole writer of the ecosystem's README test badges (see `m3trik/docs/TEST_BADGE_STANDARD.md`). |
| `test_sandbox` | `TestSandbox` | Process-level test isolation, activated once from a conftest/runner: refuses every real browser launch (loudly, and records it) and routes the process temp dir into one throwaway root that children inherit. `uitk.testing.TestSandbox` extends it with the Qt-side stores. |

## Pipeline primitives

| Module | Key symbols | What it does |
|---|---|---|
| `task_factory` | `TaskFactory` | Reflection-based task/check runner: discovers `task_*`/`check_*` methods, orders by `TASK_ORDER`, runs with LIFO set/revert state management (the DCC scene exporters subclass it). |
| `qc_log` | `QcLog`, `QcGate` | Append-only structured run logs with stage timing, plus threshold-based acceptance gates for batch pipelines. |
| `step_toggle` | `StepToggle` | Timed multi-step press toggles: rapid repeats step deeper, a pause decays to plain on/off; injected clock for testability. |

## Hierarchy toolkit (`hierarchy_utils/`)

DCC-agnostic delimited-path hierarchy comparison. `HierarchyPath` is the single home for path-string primitives (namespace cleaning, split/join, leaf/parent/tail — Maya-style `|`/`:` defaults, but parameterized); `HierarchyIndexer` builds/queries tree indices; `HierarchyMatching` supplies exact / tail-path / fuzzy match strategies; `HierarchyAnalyzer` diffs two hierarchies into typed difference records (including *moved* detection via deterministic best-pair assignment); `HierarchyDiff.from_differences` turns them into a JSON-serializable report.

## Domain engines (`engines/`)

A chartered namespace for **host-agnostic domain engines** — complete application cores (model + planner + DI seams) that downstream packages which cannot import each other (mayatk, blendertk) must share. Same hard rules as the rest of pythontk (no DCC imports, zero-dep preferred); the *generality* requirement is relaxed by charter. Placement rules live in [`CLAUDE.md`](../../CLAUDE.md).

### `engines/shots/` — shot timeline engine

Layered so everything pure is separable from everything that touches a scene:

- `shot_model` — `ShotBlock` + `ShotStore`: CRUD, typed observer events, pluggable `ScenePersistence`; every scene-reaching operation is an overridable hook with a pure default.
- `shot_plan` — pure planner: multi-shot timeline transformations resolved into a side-effect-free `MovePlan` via collision-safe topological ordering.
- `shot_apply` *(module-path import — `apply` is too generic for the root)* — commits a plan through injected `move_keys`/`shift_audio` writers in a park/move/land discipline.
- `shot_detection` — pure boundary/clustering math over already-gathered animation segments.
- `manifest/` — production-CSV → shot-plan pipeline: `manifest_model` (step/object graph, `ColumnMap`, CSV parsing), `mapping/` (declarative JSON column-mapping files) and `behaviors/` (keying-recipe schema + anchor/offset/duration → keyframe math) *(both module-path imports — `Mapping`/`Behaviors` are too generic for the root)*, `range_resolver`, and `ShotManifest` (compute-then-commit planner with overridable scene hooks).

### `engines/instancing/`

`AssemblySorter` — clusters separated mesh parts into copies of a repeated assembly, operating purely on per-part feature dicts (bbox, topology counts, area, center, material key) supplied by the DCC adapter. The general PCA/NN math stays in `geo_utils.PointCloud`.

### `engines/textures/` — PBR texture engine

- `map_registry` — the domain model: `MapType` taxonomy, runtime-extensible `MapRegistry`, `WF` workflow identifiers (Unity URP/HDRP, UE, glTF, Godot, spec-gloss), alias/fallback/precedence rules.
- `map_factory/` — planner + strategies: `ConversionRegistry`, `TextureProcessor` (shared per-set context), `WorkflowHandler` strategies (ORM, MRAO, mask map, metallic-smoothness, …), and the `MapFactory.prepare_maps` orchestrator.
- `map_optimizer` — pure `plan(image) → [Op]` decision pass with an `apply` executor and a read-only `assess` twin, so predictions can never drift from mutations.
- `map_compositor` — alpha-composites layered maps into channel-packed outputs; auto-generates the complementary DirectX/OpenGL normal.
- `output_template` — the export-preset layer: `OutputSpec` is hard (container, bit depth, compression), `DeliveryBudget` is advisory (size ceiling, POT); `OutputTemplates` owns the per-workflow profile catalogue.
- `region_masks` — named face-group masks that gate texture regions at runtime: registry, JSON manifest schema, and `RegionMaskPacker` (rasterizes UV triangles into RGBA slot channels).
- `mat_report` — pure text/HTML formatters over the material/texture record schema both DCCs produce, so they share one report surface.

The Qt panels driving the texture engine interactively (Map Converter / Packer / Compositor) live in [extapps](https://github.com/m3trik/extapps).

## Links

- Package overview: [`docs/README.md`](../../docs/README.md)
- Full API: [`API_REGISTRY.md`](../../API_REGISTRY.md)
- Placement charter (`*_utils` vs engines): [`CLAUDE.md`](../../CLAUDE.md)
