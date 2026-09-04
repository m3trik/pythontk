# !/usr/bin/python
# coding=utf-8
from pythontk.core_utils.module_resolver import bootstrap_package

__package__ = "pythontk"
__version__ = "0.9.34"

"""Expose toolkit utilities with explicit resolver include maps for clarity."""


DEFAULT_INCLUDE = {
    "audio_utils._audio_utils": "AudioUtils",
    "img_utils._img_utils": "*",
    "img_utils.mask_generator": ["MaskGenerator"],
    "img_utils.ktx2_encoder": ["Ktx2Encoder"],
    "img_utils.exposure_equalizer": ["ExposureEqualizer"],
    "img_utils.image_curator": ["ImageCurator"],
    "file_utils.mesh_ops": ["MeshOps"],
    "str_utils._str_utils": "*",
    "vid_utils._vid_utils": "*",
    "vid_utils.frame_extractor": ["FrameExtractor"],
    "file_utils._file_utils": "*",
    "file_utils.metadata": "Metadata",
    "file_utils.mesh_convert._mesh_convert": "MeshConvert",
    # Deliverable inspection/verification (read-only siblings of the
    # editors above): structured GLB access, a zero-dep binary-FBX
    # reader, and the gate runner the export pipelines point at files.
    "file_utils.mesh_convert.glb_reader": ["GlbReader"],
    "file_utils.mesh_convert.fbx_file": ["FbxFile"],
    "file_utils.mesh_convert.fbx_media": ["FbxMedia"],
    "file_utils.mesh_convert.export_verify": ["ExportVerifier"],
    "file_utils.uv_unwrap._uv_unwrap": "UvUnwrap",
    # Zero-dep USD primitives: sniffing, spec-compliant USDZ packaging, and a
    # usda mesh author + OBJ converters (the no-DCC publish path). DCC-native
    # USD I/O lives downstream in mayatk/blendertk ``env_utils.usd``.
    "file_utils.usd": [
        "USD_EXTENSIONS",
        "UsdFile",
        "UsdzPackager",
        "UsdMeshWriter",
    ],
    "file_utils.temp_artifacts": ["TempArtifacts", "CachedArtifact", "ScratchTwins"],
    # Batch-rename executor (dry-run + one-record report) shared by the file
    # tenant here and the mayatk / blendertk scene-naming engines.
    "file_utils.file_naming": ["RenamePlan", "FileNaming"],
    # Shared project-workspace model + workspace.mel codec (zero-dep). One
    # project folder serves Maya (which parses the marker natively) and
    # blendertk (whose current-workspace resolver builds on this); one
    # unnamespaced template store defines how BOTH build a new one.
    "file_utils.workspace": [
        "WORKSPACE_MARKER",
        "DEFAULT_FILE_RULES",
        "RULE_NICE_NAMES",
        "Workspace",
        "WorkspaceTemplates",
    ],
    "iter_utils._iter_utils": "*",
    "math_utils._math_utils": "*",
    "math_utils.progression": "ProgressionCurves",
    "math_utils.noise": "BandLimitedNoise",
    "math_utils.weights": "Weights",
    "geo_utils.polyline": "Polyline",
    "geo_utils.pointcloud": "PointCloud",
    "geo_utils.rail_surface": "RailSurface",
    "geo_utils.plate_emitter": "PlateEmitter",
    "geo_utils.shadow_projection": ["ShadowProjection", "ShadowModel", "ShadowRaster"],
    "geo_utils.shadow_horizon": ["ShadowHorizon", "HorizonMap"],
    "img_utils.shadow_atlas": "ShadowAtlas",
    "geo_utils.uv_pack": ["UvPack", "PackIslandsResult"],
    "geo_utils.uv_transfer": ["UvTransfer", "TransferTable"],
    # Shots engine — DCC-agnostic shot model core shared by mayatk / blendertk
    "core_utils.engines.shots.shot_model": [
        "ShotStore",
        "ShotBlock",
        "ScenePersistence",
        "StoreEvent",
        "ShotDefined",
        "ShotUpdated",
        "ShotRemoved",
        "ActiveShotChanged",
        "SettingsChanged",
        "BatchComplete",
        "StoreInvalidated",
        "SHOT_PALETTE",
    ],
    # The shot system's record of what IT wrote on scene curves, so those
    # writes can be released when the boundary that caused them moves.
    "core_utils.engines.shots.shot_ledger": [
        "ShotEditLedger",
    ],
    # Key stash engine — clips of keys parked outside the working animation,
    # inert until retrieved; the DCC adapters (mayatk / blendertk ``KeyStash``)
    # subclass it and own where the keys physically live.
    "core_utils.engines.key_stash.key_stash_model": [
        "KeyStash",
        "StashedClip",
        "StashChanged",
    ],
    "core_utils.engines.shots.shot_plan": [
        "ShotMove",
        "MovePlan",
        "ShotPlanner",
    ],
    "core_utils.engines.shots.shot_detection": [
        "STANDARD_TRANSFORM_ATTRS",
        "ShotDetection",
    ],
    # shot_apply's `apply` is deliberately NOT root-registered — same policy as
    # the comment below: generic names stay off the top level (consumers import
    # it via its module path).
    # Shot Manifest engine — pure CSV → shot-plan core (mapping/behaviors are
    # imported via their module path to keep generic names off the top level).
    "core_utils.engines.shots.manifest.manifest_model": [
        "ManifestModel",
        "ColumnMap",
        "BuilderStep",
        "BuilderObject",
        "PlannedShot",
        "ObjectStatus",
        "StepStatus",
    ],
    "core_utils.engines.shots.manifest.manifest_engine": [
        "ShotManifest",
    ],
    "core_utils.engines.shots.manifest.range_resolver": [
        "RangeResolver",
    ],
    # Auto-instancing engine — separated-part clustering core (mayatk / blendertk)
    "core_utils.engines.instancing.assembly_sorter": "AssemblySorter",
    # PBR texture engine — map taxonomy / preparation / packaging
    # (mayatk / blendertk material tools + extapps texture apps)
    "core_utils.engines.textures.map_factory": ["MapFactory"],
    "core_utils.engines.textures.map_registry": ["MapRegistry", "MapType"],
    "core_utils.engines.textures.output_template": [
        "DeliveryBudget",
        "OutputSpec",
        "OutputTemplate",
        "OutputTemplates",
    ],
    "core_utils.engines.textures.map_compositor": [
        "MapCompositor",
        "BatchResult",
        "NormalOutputMode",
    ],
    "core_utils.engines.textures.map_optimizer": ["MapOptimizer", "Op"],
    "core_utils.engines.textures.region_masks": [
        "RegionGroup",
        "RegionGroupRegistry",
        "RegionMaskManifest",
        "RegionMaskPacker",
    ],
    "core_utils.engines.textures.mat_report": ["MatReport"],
    "core_utils._core_utils": "*",
    "core_utils.help_mixin": "HelpMixin",
    "core_utils.symbol_record": "SymbolRecord",
    # Markdown-example rot gate — validates doc code blocks against the live
    # surface (README gates here and downstream build on it).
    "core_utils.doc_audit": "DocAudit",
    "core_utils.package_manager": "PackageManager",
    "core_utils.git": "Git",
    "core_utils.class_property": "ClassProperty",
    # LoggerExt / LevelAwareFormatter are reached by raw module path today
    # (uitk's example wiring, mayatk's scene_audit), which is a published
    # dependency on a private path -- register them so the surface says so.
    "core_utils.logging_mixin": [
        "LoggingMixin",
        "TableMixin",
        "LoggerExt",
        "LevelAwareFormatter",
    ],
    # Same case: mayatk's blender_bridge imports ScriptTemplate and the three
    # mode constants straight from this module.
    "core_utils.script_template": [
        "ScriptTemplate",
        "SEND_TO",
        "SAVE_AS",
        "ROUND_TRIP",
    ],
    "core_utils.namespace_handler": "NamespaceHandler",
    "core_utils.namedtuple_container": "NamedTupleContainer",
    "core_utils.color": ["Color", "ColorPair", "Palette"],
    "core_utils.hierarchy_utils.hierarchy_diff": "HierarchyDiff",
    "core_utils.singleton_mixin": "SingletonMixin",
    "core_utils.step_toggle": ["StepToggle"],
    "core_utils.module_reloader": ["ModuleReloader", "ReloadReport", "reload_package"],
    "core_utils.execution_monitor._execution_monitor": "ExecutionMonitor",
    "core_utils.cancel_scope": ["CancelScope", "OperationCancelled"],
    "core_utils.qc_log": ["QcLog", "QcGate", "GateError"],
    "core_utils.status_badge": ["StatusBadge"],
    "core_utils.test_sandbox": ["TestSandbox"],
    "core_utils.app_launcher": "AppLauncher",
    "core_utils.app_installer": "AppInstaller",
    "core_utils.app_handoff": [
        # The carrier vocabulary beside USD_EXTENSIONS: one spelling for every
        # panel (uitk) and every producer (the DCC mixins).
        "CARRIER_PARAM",
        "CARRIER_EXTENSIONS",
        "CARRIER_BY_EXTENSION",
        "HandoffBridge",
        "ScriptLaunchBridge",
        "ScriptLaunchDeliverer",
        "ScriptRunDeliverer",
        "ScriptRoundTripDeliverer",
        "ScriptLaunchSpec",
        "AppSpec",
        "Deliverer",
        "HandoffRequest",
        "Payload",
    ],
    # Blocking run-script-collect-artifact counterpart of ScriptLaunchDeliverer
    # (backs pull-direction bridges, e.g. blendertk's Maya-scene import).
    "core_utils.script_run": ["ScriptRunResult", "ScriptRunner"],
    # Process/log line-stream primitives (composed by the app-specific
    # connection shells in mayatk/blendertk, e.g. SubstanceConnection).
    "core_utils.process_stream": [
        "OutputStream",
        "ProcessReader",
        "LogTailer",
        "TeeStream",
    ],
    "core_utils.user_config": ["UserConfig"],
    # Ecosystem SSoT for type-based name affixes (mesh -> _GEO, material -> _MAT).
    # Qt-free and DCC-free on purpose: uitk's affix picker resolves against it and
    # cannot import a toolkit, and extapps has no DCC at all.
    "core_utils.naming_convention": [
        "AffixRule",
        "NamingConvention",
    ],
    "core_utils.preset_store": [
        "PresetStore",
        "Codec",
        "JSON_CODEC",
    ],
    "core_utils.schema_spec": [
        "SchemaSpec",
        "ValidationResult",
        "SchemaError",
        "FieldDoc",
    ],
    "core_utils.template_set": ["TemplateSet"],
    # Generic task/check pipeline base (shared by mayatk/blendertk scene exporters)
    "core_utils.task_factory": "TaskFactory",
    "core_utils.cli": "CLI",
    # Hierarchy utils
    "core_utils.hierarchy_utils.hierarchy_path": "HierarchyPath",
    "core_utils.hierarchy_utils.hierarchy_indexer": "HierarchyIndexer",
    "core_utils.hierarchy_utils.hierarchy_matching": "HierarchyMatching",
    "core_utils.hierarchy_utils.hierarchy_analyzer": [
        "HierarchyDifference",
        "HierarchyAnalyzer",
        "DifferenceType",
    ],
    "net_utils.ssh_client": "SSHClient",
    "net_utils.credentials": "Credentials",
    "net_utils._net_utils": "NetUtils",
    "net_utils.rpc.client": "RpcClient",
    # Bytes behind an http(s) URL (share links normalized); the seam every
    # path-or-URL loader (manifest CSV, installer download) fetches through.
    "net_utils.remote_file": "RemoteFile",
    # Loopback static server + live manifest behind the WebXR/browser preview
    # loop. Localhost is a secure context, so this is all `navigator.xr` needs.
    "net_utils.preview.server": "PreviewServer",
    # What a post-conversion pass is handed; a consumer registering one
    # against `PreviewDeliverer.EDIT_PASSES` / `FILE_PASSES` annotates it.
    "net_utils.preview.deliverer": ["PreviewDeliverer", "PreviewPassContext"],
    "net_utils.preview.bridge": "PreviewBridge",
    "net_utils.rpc.installer": [
        "PluginInstaller",
    ],
    "net_utils.rpc.job": ["Call", "Result", "RpcJob"],
    # The in-application half of the RPC pair (`client` drives it from outside).
    # Exposed here for the plugin that loads in place and can import pythontk;
    # installed plugins stage `plugin_core.py` into their payload instead.
    "net_utils.rpc.plugin_core": ["OpRegistry", "MainThreadMarshaller", "RpcPlugin"],
    "str_utils.fuzzy_matcher": "FuzzyMatcher",
    "str_utils.hotkey_utils": "HotkeyUtils",
}


# ``bootstrap_package`` derives ``__all__`` automatically from the resolved
# public surface (DEFAULT_INCLUDE), so it never has to be hand-maintained here.
bootstrap_package(globals(), include=DEFAULT_INCLUDE)
# Test: 222117
