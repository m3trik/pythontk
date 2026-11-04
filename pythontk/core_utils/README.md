# core_utils

Shared infrastructure — the non-data-type half of pythontk. Where the `*_utils` siblings hold general primitives placed by data type (strings, images, math, …), this package holds the *mechanisms* the ecosystem runs on: mixins, package bootstrap, app orchestration, config stores, pipeline primitives, and the shared domain engines.

Flat by default, never re-nested for cosmetics: a module moves into a subpackage only when a CODE_STANDARD §3 trigger fires (a second cohesive module, private internals, an owned asset directory). Downstream packages still import many of these module paths directly, so every move follows CODE_STANDARD §14: grep the workspace, and alias an externally imported path for one release.

Everything here is reachable from the package root (`import pythontk as ptk; ptk.LoggingMixin`) unless marked otherwise below. Full signatures: `python -m pythontk ops run api.search "query=<words>"`.

## Class infrastructure

| Module | Key symbols | What it does |
|---|---|---|
| `_core_utils` | `CoreUtils` | Decorators and reflection helpers: `cached_property`, `listify` (broadcast a function over list-like args, optionally threaded), attribute get/set, `format_return`. |
| `logging_mixin/` | `LoggingMixin`, `LoggerExt`, `TableMixin` | Class-scoped logging: extra levels (SUCCESS/RESULT/NOTICE/PROGRESS), spam-guarded `*_once` variants, boxes/groups/tables, managed file tee, capped ring buffer. The column math of every block (display widths, wrapping, box and table geometry) is one internal `TextLayout` (`_text_layout.py`). |
| `help_mixin` | `HelpMixin` | `.help()` / `.source()` / `.signature()` on any class: the rendering, over `ApiSurface`'s live reading. |
| `class_property` | `ClassProperty` | Class-level properties (replacement for the removed `@classmethod @property` stacking). |
| `singleton_mixin` | `SingletonMixin` | Singletons keyed per `(class, singleton_key)` — subclasses sharing a key never collide. |
| `namedtuple_container` | `NamedTupleContainer` | Generic named-tuple collection with querying, dynamic field access, and CSV round-trip. |
| `namespace_handler` | `NamespaceHandler` | Lazily-resolved attribute namespace for an owner object, with `Placeholder` deferred construction. |
| `color` | `Color`, `ColorPair`, `Palette` | Stdlib-only color primitives (a lone value primitive lodges here — see the placement note in `__init__.py`). |

## App / process orchestration

| Module | Key symbols | What it does |
|---|---|---|
| `app_launcher/` | `AppLauncher` | Cross-platform app discovery + launching: detached `launch`, blocking `run`, bound `spawn`; install discovery (incl. `companion_python` / `windowless_python`: the interpreter a host binary pairs with, and its console-free `pythonw` twin), the live process environment, session-aware launching and window lookup, process queries/termination, and desktop shortcuts (`create_shortcut`: a Unicode `IShellLinkW` `.lnk` stamped with an AppUserModelID, or a freedesktop `.desktop` entry). One private mixin per job (`_discovery`, `_environment`, `_desktop`, `_processes`, `_shortcuts`) behind the facade in `_app_launcher.py`; every member resolves on `AppLauncher`. |
| `app_installer` | `AppInstaller` | Download, extract, and version-track external tool binaries (ffmpeg, toktx, …) from caller-supplied platform definitions; stdlib-only, `.installed.json` catalog. |
| `handoff/app_handoff` | `HandoffBridge`, `Deliverer`, `AppSpec` | The Qt-free/DCC-free "export something and hand it to an app" backbone — one Template-Method flow (`resolve → preflight → produce → deliver → ingest`) with a per-mode `Deliverer` strategy. Base of the ecosystem's Maya/Blender/Marmoset/Substance bridges. |
| `handoff/handoff_scope` | `HandoffScope` | The Scope words a hand-off offers (`selected` / `all` / `visible`) and their precedence: `resolve(scope, selected=..., all=..., visible=...)` over lookups the host supplies -- unknown is `selected`, and a widening lookup that cannot answer narrows to the selection. Every bridge panel, the preview push and the DCC bakers resolve through it. |
| `handoff/manifest` | `HandoffManifest` | The hand-off sidecar's contract (`<payload>.manifest.json`, what the FBX / USD carrier has no place for): suffix, version, section vocabulary and read/write rules, imported by both sides. Reading is tolerant (a damaged sidecar is an empty manifest, flagged `unreadable`); a section's contents stay its codec's (`ShotTransfer` for `shots`). |
| `handoff/manifest_plan` | `ManifestPlan` | Ordered, gated replay of what a manifest asks for (`HandoffManifest.plan`): a consumer names its steps once, in order; a step runs only when its section was sent, progress and stop requests are honoured, and a failing step raises unless declared `best_effort`. |
| `handoff/script_run` | `ScriptRunner`, `ScriptRunResult` | Blocking script-run-to-artifact: success is judged by the produced artifact, not the exit code (DCC batch interpreters routinely crash in teardown after succeeding). |
| `handoff/script_template` | `ScriptTemplate`, `SEND_TO` / `SAVE_AS` / `ROUND_TRIP` | On-disk script-template discovery + `__KEY__` rendering; declares the ecosystem-wide hand-off mode vocabulary (`SEND_TO`, `SAVE_AS`, `ROUND_TRIP`) that every bridge imports — it is an on-disk contract, so there is exactly one copy. |
| `process_stream` | `OutputStream`, `ProcessReader`, `LogTailer` | Composable line-stream primitives: thread-safe multi-consumer pub/sub with replayable history + `wait_for`, subprocess PIPE reader, rotation-aware log tailer. |
| `execution_monitor` | `ExecutionMonitor` | Wraps long-running operations with threshold-escalated dialogs, spinner/task indicators, and an external watchdog; cancellation is delegated to a `CancelScope`. |
| `agent_instructions` | `AgentInstructions` | A codebase's agent instructions as a checked tree: `AGENTS.md` rules with a `CLAUDE.md` stub at every level, shared packs vendored under a drift check, a SessionStart hook for a repo cloned alone; `check`, `sync`, `measure` (what a session loads), `duplicates` -- also pythontk's `instructions.*` operations. |
| `operation_catalog/` | `Operation`, `OperationEntry`, `OperationCatalogDocument`, `OpContext`, `OpRequest`, `Outcome`, `Issue`, `Change`, `OpCall`, `OperationHost`, `PureOperationHost`, `BatchOperationHost`, `SceneFiles`, `PureSceneFiles`, `SessionOperationHost`, `OperationServer`, `OperationProvider`, `OperationCatalog` | Named, typed actions every front end calls alike (the `python -m pythontk ops` CLI, MCP, panels), whatever language declares them: a declaration and its published forms (`operation`), the run context with its dry run (`op_context`), the request and result payloads (`outcome`), the host seam with this process as its default (`operation_host`), the runner a host library configures to give each call a fresh instance of its application, request file in and outcome file out (`batch_operation_host`), the runner that keeps one instance per session instead and serves the same run folders from a `JobQueue`, for a front end that lives long (`session_operation_host`), the catalog as an MCP server (`operation_server`, `python -m pythontk ops mcp`), one package's declarations, committed catalog and upkeep, onboarding and checks included (`operation_provider`), and the facade (`_operation_catalog`) that discovers providers through the `pythontk.operations` entry points (or a catalog file named by `--provider`), validates a call against its JSON Schema and runs it. |
| `python_console` | `PythonConsole` | Code run a snippet at a time in one namespace that lasts, as in a console: the last expression's value and what it printed come back; a dry run lists the code. Behind the `python.exec` (a plain Python, the `python` host) / `maya.python.exec` / `blender.python.exec` operations. |
| `cancel_scope` | `CancelScope`, `OperationCancelled` | One cooperative cancellation object shared by every cancel affordance: *push* (`cancel()` from any thread) + *pull* (sources polled at the operation's own checkpoints), consumable bool-style (`tick()`) or ambient exception-style (`CancelScope.check()` via `ContextVar`). |

**The three hand-off shapes.** `HandoffBridge` owns one invariant flow — `resolve → preflight → produce → deliver → ingest` — with the delivery step a per-mode `Deliverer` strategy, so three hand-off shapes come off one export pipeline: `send()` (`SEND_TO`, detached launch), `save_as()` (`SAVE_AS`, blocking run that keeps a native file of the target's format), and `round_trip()` (`ROUND_TRIP`, blocking run whose intermediate artifact is folded back onto the host's own objects — the target either edits the payload in place, as RizomUV does, or writes a new artifact, as mayatk's Blender lightmap bake does). The mode strings are declared once, in `handoff/script_template.py`, and every bridge imports them from the root: they are an on-disk contract (each template's `BRIDGE_MODES` tuple), and a local copy would be a second dialect of a file format.

## Config & persistence

| Module | Key symbols | What it does |
|---|---|---|
| `user_config` | `UserConfig` | Resolve one JSON config doc by deep-merging a user file over a shipped default — Qt-free twin of uitk's per-user config root. |
| `presets/store` | `PresetStore` | Two-tier named-preset store (read-only built-in dir + writable user dir; user shadows built-in). uitk's `PresetManager` is a GUI over it. |
| `presets/library` | `PresetLibrary` | Every store under the presets root at once: lock, collections, bundles, import/export (uitk's Preset Editor). |
| `schema_spec` | `SchemaSpec` | Dataclass-declared schema for JSON/YAML template files: one definition derives `validate` (errors vs tolerated warnings), `skeleton`, and `to_markdown` reference docs. |
| `template_set` | `TemplateSet` | Binds a `PresetStore` (storage SSoT) to a `SchemaSpec` (shape SSoT): a discoverable, user-extensible set of schema-validated template files. |

## Package / dev infrastructure

| Module | Key symbols | What it does |
|---|---|---|
| `module_resolver` | `bootstrap_package` *(direct import)* | The lazy attribute-resolution machinery behind every ecosystem package root (`DEFAULT_INCLUDE` → lazy `__getattr__` → derived `__all__`). |
| `module_reloader` | `ModuleReloader` | Hot-reload a package and its submodules in dependency order; returns a `ReloadReport`. |
| `package_manager` | `PackageManager` | pip wrapper: install/uninstall/list/update, version checks, concurrent latest-index lookups. |
| `cli` | `CLI` | Consistent `argparse` builders shared across scripts (e.g. the standard SSH connection argument group). |
| `api_surface/` | `ApiSurface`, `PackageSurface`, `ModuleSurface`, `ClassSurface` | The one reader of a package's public API: from source with `ast` (no import; what the release check reads), the C# a package declares (`read_csharp`: no editor, nothing compiled), from live objects (what `HelpMixin` renders and the runtime drift gate compares), one docstring-summary rule, and `search` / `describe` over every registered package (pythontk's `api.search` / `api.describe` operations); a package read once is kept in the per-user cache until its source or the reader changes. Standard library only, so the registry generator runs on a bare CI box. |
| `symbol_record` | `SymbolRecord` | The shared public-API symbol shape `ApiSurface` produces from source and from live objects; compared in the runtime-vs-static drift gate. |
| `doc_audit` | `DocAudit` | Markdown-example rot gate: extracts fenced code blocks and validates attribute chains + keyword arguments against the live package. Backs the README gates in `test/test_doc_audit.py`; downstream repos can gate their own docs with it. |
| `status_badge` | `StatusBadge` | Shields.io badges embedded in a markdown file; sole writer of the ecosystem's README test badges (see `m3trik/docs/TEST_BADGE_STANDARD.md`). |
| `test_sandbox` | `TestSandbox` | Process-level test isolation, activated once from a conftest/runner: refuses every real browser launch (loudly, and records it) and routes the process temp dir into one throwaway root that waited children inherit (a detached `AppLauncher.launch` gets the host's environment instead, `host_environ`, since it outlives the root). `uitk.testing.TestSandbox` extends it with the Qt-side stores. |

## Pipeline primitives

| Module | Key symbols | What it does |
|---|---|---|
| `task_factory` | `TaskFactory` | Reflection-based task/check runner: discovers `task_*`/`check_*` methods, orders by `TASK_ORDER`, runs with LIFO set/revert state management (the DCC scene exporters subclass it). |
| `qc_log` | `QcLog`, `QcGate` | Append-only structured run logs with stage timing, plus threshold-based acceptance gates for batch pipelines. |
| `step_toggle` | `StepToggle` | Timed multi-step press toggles: rapid repeats step deeper, a pause decays to plain on/off; injected clock for testability. |

## Hierarchy toolkit (`hierarchy_utils/`)

DCC-agnostic delimited-path hierarchy comparison. `HierarchyPath` is the single home for path-string primitives (namespace cleaning, split/join, leaf/parent/tail — Maya-style `|`/`:` defaults, but parameterized); `HierarchyIndexer` builds/queries tree indices; `HierarchyMatching` supplies exact / tail-path / fuzzy match strategies; `HierarchyAnalyzer` diffs two hierarchies into typed difference records (including *moved* detection via deterministic best-pair assignment) and runs the pairing passes both DCC Hierarchy Syncs chain over missing/extra path lists (`detect_reparented`, `detect_fuzzy_renames`, `detect_suffix_flattening`); `HierarchyDiff.from_differences` turns them into a JSON-serializable report.

## Domain engines (`engines/`)

A chartered namespace for **host-agnostic domain engines** — complete application cores (model + planner + DI seams) that downstream packages which cannot import each other (mayatk, blendertk) must share. Same hard rules as the rest of pythontk (no DCC imports, zero-dep preferred); the *generality* requirement is relaxed by charter. Placement rules live in [`AGENTS.md`](../../AGENTS.md).

### `engines/shots/` — shot timeline engine

Layered so everything pure is separable from everything that touches a scene:

- `shot_model` — `ShotBlock` + `ShotStore`: CRUD, typed observer events, pluggable `ScenePersistence`; every scene-reaching operation is an overridable hook with a pure default.
- `shot_plan` — pure planner: multi-shot timeline transformations resolved into a side-effect-free `MovePlan` via collision-safe topological ordering.
- `shot_apply` *(module-path import — `apply` is too generic for the root)* — commits a plan through injected `move_keys`/`shift_audio` writers in a park/move/land discipline.
- `shot_detection` — pure boundary/clustering math over already-gathered animation segments.
- `shot_sequencer` — `ShotSequencer`: every ripple-editing operation (define, slide, ripple, insert, merge, split, pad, respace, …) decided once over narrow scene hooks the DCC sequencers override; with no adapter it is a complete bounds-only sequencer.
- `shot_transfer` — `ShotTransfer`: the store as the hand-off manifest's `shots` section, the one codec both producers and both consumers run, so shots cross Maya ↔ Blender 1:1 (scene-name spelling injected).
- `shot_ledger` — `ShotEditLedger`: the edits the shot system authors on the animator's curves (gap holds, boundary keys, a behavior's own keys), so a moved boundary can take them back.
- `effect_recipe` — `EffectRecipe`: how each render effect and audio clip is keyed, once per scene, riding the store (`ShotStore.effect_recipe`); the panels edit it and the manifest's Build keys with it.
- `shot_report` — `ShotReport`: what a shots panel says about the store (the sequence at a glance, a trim / pad's outcome, the Delete Shots scopes and their question), worded once for both hosts.
- `animation_record` — `AnimationRecord` / `AnimationBlock`: the `animation` record the store publishes on the `data_export` carrier and the GLB's `extras.animation` block, declared once.
- `shot_conformance` *(module-path import)* — `ShotStoreConformance`: the golden case every host's store is held to, one store holding one of everything `ShotStore.to_dict` saves and its exact saved text.
- `manifest/` — production-CSV → shot-plan pipeline: `manifest_model` (step/object graph, `ColumnMap`, CSV parsing), `mapping/` (declarative JSON column-mapping files) and `behaviors/` (keying-recipe schema + anchor/offset/duration → keyframe math) *(both module-path imports — `Mapping`/`Behaviors` are too generic for the root)*, `range_resolver`, `manifest_engine` (`ShotManifest`, the compute-then-commit planner with overridable scene hooks), `manifest_edits` (`ManifestEdits`: a user's edits to a sheet, kept apart from it and composed over it), `manifest_export` (`ManifestExport`: the manifest's rows as CSV, JSON or Markdown, for a file or the clipboard) and `manifest_tags` (`ManifestTags`: a palette-slot colour tag per step, the user's over the mapping's automatic `color_by` rule).

### `engines/scene_export/` — scene-export engine

The pure half of both DCC Scene Exporters. The data a deliverable carries --
every record and GLB block, its versions and fields -- is
[`DELIVERABLE_FORMAT.md`](engines/scene_export/DELIVERABLE_FORMAT.md),
generated from these declarations.

- `scene_records` — every tool-authored scene record declared once (`Scope`, `Kind`, `Merge`, `RecordSpec`, the `SceneRecords` catalogue); `scene_store` — `SceneStoreBase`, the string-per-scope contract each DCC's `DataNodes` implements, plus the crossings built on it; `export_snapshot` — `ExportContext` / `ExportSnapshot`, compute every record then commit once; `record_transfer` — `TransferContext` / `RecordTransfer`, records crossing into another scene by declaration.
- `export_profile` — `ExportProfile` / `ExportRun`: the panels' widget values -> one run configuration (task order, naming, texture and GLB policy).
- `hierarchy_baseline` — `HierarchyBaseline` (per-scene set algebra) and `HierarchyBaselineStore` (its scene-stored half).
- `scene_exporter` / `scene_data_sidecar` — `SceneExporterBase` (progress stream, check-override consent, run config, log file) and `SceneDataSidecarBase` (the `.scene_data.json` format, naming, migration, diff report): the DCC `SceneExporter` / `SceneDataSidecar` bind them to their host -- the sidecar's `HOST` adapter (`SidecarHost`), the exporter's `STORE` (the host's `SceneStoreBase`) -- and override nothing (`check_dcc_twins.py`, host bindings).

### `engines/instancing/`

`AssemblySorter` — clusters separated mesh parts into copies of a repeated assembly, operating purely on per-part feature dicts (bbox, topology counts, area, center, material key) supplied by the DCC adapter. The general PCA/NN math stays in `geo_utils.PointCloud`. `InstanceGrouping` merges signature buckets and summarizes a run (stateless; the adapters pass their own logger and threshold), and `InstancingStrategy` picks the instancing strategy for a group under a config, its thresholds overridable per instance.

### `engines/key_stash/`

`KeyStash` — clips of keyframes parked outside the working animation (`StashedClip`), inert until retrieved: one store on every host over a `KeyStashHost` adapter (`PureKeyStashHost`: the pure defaults), which mayatk's and blendertk's `KeyStash` bind, adding the scene operations (`stash` / `retrieve` / `drop` / `preview`).

### `engines/rig_graph/` — rig intent

What a rigger meant, carried between hosts instead of a node-type-to-node-type converter: `RigGraph` (the document and its validation), `RigCapability` (what one target can build), `RigPlanner` (a pure `PlanResult`: what to build, what to bake, a `ReportEntry` per loss), `RigVerify` (what was built, measured against the source's samples), `RigTransfer` (the one orchestration both bridges run) and `RigMachinery` (what a baked rig leaves behind). Nothing reaches a scene, so a pre-flight runs with no DCC open.

### `engines/lighting/` — light layers and look match

`LightLayers` / `LightMix` (one linear viewer frame per light, and a weighted, tinted mix of them), `LightPredictor` (a mix's frame through the viewer's display transform), `LightRigPayload` (the rig a mix becomes, which a host's rig tool builds), and `look_match/`: `LookMatch` and its parts, holding a mix to a reference image region by region (`python -m pythontk.core_utils.engines.lighting.look_match --help`). The guide: [Look match](../../docs/look_match.md).

### `engines/textures/` — PBR texture engine

- `map_registry` — the domain model: `MapType` taxonomy, runtime-extensible `MapRegistry`, `WF` workflow identifiers (Unity URP/HDRP, UE, glTF, Godot, spec-gloss), alias/fallback/precedence rules.
- `map_factory/` — planner + strategies: `ConversionRegistry`, `TextureProcessor` (shared per-set context), `WorkflowHandler` strategies (ORM, MRAO, mask map, metallic-smoothness, …), and the `MapFactory.prepare_maps` orchestrator.
- `map_optimizer` — pure `plan(image) → [Op]` decision pass with an `apply` executor and a read-only `assess` twin, so predictions can never drift from mutations.
- `map_compositor` — alpha-composites layered maps into channel-packed outputs; auto-generates the complementary DirectX/OpenGL normal.
- `output_template` — the export-preset layer: `OutputSpec` is hard (container, bit depth, compression), `DeliveryBudget` is advisory (size ceiling, POT); `OutputTemplates` owns the per-workflow profile catalogue.
- `region_masks` — named face-group masks that gate texture regions at runtime: registry, JSON manifest schema, and `RegionMaskPacker` (rasterizes UV triangles into RGBA slot channels).
- `emissive_groups` — `EmissiveGroupsModel`: named face groups whose emissive texels a game engine toggles or dims at runtime (vertex-colour or `_EMask` channel encoding, over the region-mask model); every scene read and write goes through an `EmissiveGroupsHost` that mayatk and blendertk bind (`PureEmissiveGroupsHost`: no scene).
- `mat_report` — pure text/HTML formatters over the material/texture record schema both DCCs produce, so they share one report surface.

The Qt panels driving the texture engine interactively (Map Converter / Packer / Compositor) are [extapps](https://github.com/m3trik/extapps)' `texture_maps` apps.

## Links

- Package overview: [`docs/README.md`](../../docs/README.md)
- Full API: `python -m pythontk ops run api.search "query=<words>"`
- Placement charter (`*_utils` vs engines): [`AGENTS.md`](../../AGENTS.md)
