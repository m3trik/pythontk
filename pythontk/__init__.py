# !/usr/bin/python
# coding=utf-8
import os as _os

from pythontk.core_utils.module_resolver import bootstrap_package

__package__ = "pythontk"
__version__ = "0.21.0"

"""Expose toolkit utilities with explicit resolver include maps for clarity."""

# OpenCV reads this once, when its EXR codec first initializes (often at the
# first cv2 import). Every module here and in every consumer is imported after
# this package, so setting it here enables EXR/HDR IO however a process first
# reaches cv2. It sat in ``img_utils/_img_utils.py``, where the call made the
# resolver load the image stack (PIL, numpy) at every ``import pythontk``.
# ``setdefault`` respects an explicit opt-out.
_os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


DEFAULT_INCLUDE = {
    "audio_utils._audio_utils": "AudioUtils",
    # The ``audio`` record's declared shape.
    "audio_utils.audio_record": ["AudioRecord"],
    "img_utils._img_utils": "*",
    "img_utils.mask_generator": ["MaskGenerator"],
    "img_utils.ktx2_encoder": ["Ktx2Encoder"],
    "img_utils.exposure_equalizer": ["ExposureEqualizer"],
    "img_utils.image_curator": ["ImageCurator"],
    "file_utils.mesh_ops": ["MeshOps"],
    # STEP / IGES to meshes, an assembled GLB and the record of the product's
    # truth (OpenCascade, the optional ``pythontk[cad]`` extra).
    "file_utils.cad_import._cad_import": ["CadImport"],
    "file_utils.cad_import.cad_record": ["CadRecord"],
    "str_utils._str_utils": "*",
    "vid_utils._vid_utils": "*",
    "vid_utils.frame_extractor": ["FrameExtractor"],
    # Image sequence -> movie, with the producer of the frames left open:
    # the Maya playblast exporter and the WebXR preview's page recorder are
    # both hosts of this, so the target registry and the encode are written
    # once (see ``vid_utils/sequence_exporter.py``).
    "vid_utils.sequence_exporter": [
        "ExportTarget",
        "CaptureResult",
        "ExportResult",
        "SequenceEncoder",
        "SequenceExporter",
        # Its host adapter seam, and the pure host a DCC's adapter builds on.
        "SequenceHost",
        "PureSequenceHost",
    ],
    "file_utils._file_utils": "*",
    "file_utils.metadata": "Metadata",
    "file_utils.mesh_convert._mesh_convert": "MeshConvert",
    # Deliverable inspection/verification (read-only siblings of the
    # editors above): structured GLB access, a zero-dep binary-FBX
    # reader, and the gate runner the export pipelines point at files.
    "file_utils.mesh_convert.glb.reader": ["GlbReader"],
    "file_utils.mesh_convert.fbx_file": ["FbxFile"],
    "file_utils.mesh_convert.fbx_media": ["FbxMedia"],
    "file_utils.mesh_convert.export_verify": ["ExportVerifier"],
    "file_utils.mesh_convert.export_fingerprint": ["ExportFingerprint"],
    "file_utils.mesh_convert.gltf_validator": ["GltfValidator"],
    # FBX -> GLB: the one build the Scene Exporters and the WebXR preview share
    "file_utils.mesh_convert.glb.pipeline": ["GlbPipeline"],
    # The fade/highlight channel table (``GlbFades.CHANNELS``) the DCC render
    # effects join their own channel rows against.
    "file_utils.mesh_convert.glb.fades": ["GlbFades"],
    # The ``render_effects`` record's declared shape.
    "file_utils.mesh_convert.render_effects_record": ["RenderEffectsRecord"],
    "file_utils.uv_unwrap": "UvUnwrap",
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
    # Files a record names by name + recorded folder, found where they are NOW
    # (resolve, search order, claims, relocation) -- the lightmap markers of
    # both DCC bakers today; any record that names files that way.
    "file_utils.file_dependencies": ["FileDependencies"],
    # Gather the files a set of readers use into one folder, keyed by reader:
    # names kept, a clash kept apart by its parent folders, identical files
    # shared (the Texture Path Editor's Find & Copy / Collect, both DCCs).
    "file_utils.file_collect": ["FileCollect", "CollectHost", "PureCollectHost"],
    "file_utils.pending_files": ["PendingFiles"],
    # The tile / frame token vocabulary of a path (<UDIM>, <uvtile>, <u>_<v>,
    # <f>): detect, collapse to a representative, expand to the set.
    "file_utils.tiled_path": ["TiledPath"],
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
    # The workspaces under a root, measured and shaped as a tree (the Workspaces
    # panel's map); each DCC binds its host adapter (``WorkspaceMapBase.HOST``).
    "file_utils.workspace_map": [
        "WorkspaceMapBase",
        "WorkspaceMapHost",
        "PureWorkspaceMapHost",
    ],
    # Per-scene notes kept with the project: one file both DCCs' Reference
    # Manager panels read and write.
    "file_utils.scene_notes": ["SceneNotes"],
    # The Reference Manager's listing rules and Naming conventions (the save
    # path a script reproduces), over a project's Workspace.
    "file_utils.scene_listing": ["SceneListing"],
    # A project's bakes of foreign scenes: one slot per source, the record
    # inside each bake (host-read), purged by reference -- both DCC bridges.
    "file_utils.bake_cache": ["BakeCache"],
    # A folder of JSON jobs one worker process takes in order: work inside a
    # DCC opened once on a slow scene, without a port anything else could answer.
    "file_utils.job_queue": ["JobQueue"],
    "iter_utils._iter_utils": "*",
    "math_utils._math_utils": "*",
    "math_utils.progression": "ProgressionCurves",
    "math_utils.noise": "BandLimitedNoise",
    "math_utils.weights": "Weights",
    "math_utils.ramp_keys": "RampKeys",
    "math_utils.key_curve": ["KeyCurve", "CurveKey"],
    "math_utils.quaternion": "Quaternion",
    "geo_utils.polyline": "Polyline",
    "geo_utils.pointcloud": "PointCloud",
    "geo_utils.rail_surface": "RailSurface",
    # The curtain drape over RailSurface: both hosts' CurtainMesh subclass it.
    "geo_utils.curtain_drape": "CurtainDrape",
    "geo_utils.plate_emitter": "PlateEmitter",
    "geo_utils.shadow_projection": [
        "ShadowProjection",
        "ShadowModel",
        "ShadowRaster",
        "ShadowConformance",
    ],
    "geo_utils.shadow_horizon": [
        "ShadowHorizon",
        "HeightFieldMap",
        "HorizonMap",
        "ShadowHorizonConformance",
    ],
    # Where a lightmap bake's reflection probe stands, by ray casts a DCC
    # hands it through its host adapter (mayatk's and blendertk's
    # ProbePlacement bind theirs).
    "geo_utils.probe_placement": [
        "ProbePlacement",
        "ProbeRays",
        "ProbeSite",
        "ProbeHost",
        "ProbeMesh",
    ],
    # Articulated rigs: the joint model + grab solver every runtime ports, and
    # the geometry rules that propose a rig from the parts alone.
    "geo_utils.articulation.model": "ArticulationModel",
    "geo_utils.articulation.analysis": "ArticulationAnalysis",
    "geo_utils.articulation.conformance": "ArticulationConformance",
    # The plan a DCC's ArticulatedRig builds from (mayatk + blendertk share it).
    "geo_utils.articulation.plan": "ArticulationPlan",
    # The ``articulation`` record's payload, declared once
    # (SceneRecords.ARTICULATION names it; a GLB carries it as it stands).
    "geo_utils.articulation.record": [
        "ArticulationRecord",
        # Held until 0.22.0 (see HELD_UNTIL_0_22 below).
        "ArticulationWeb",
        "ArticulationWebRig",
        "ArticulationWebJoint",
        "ArticulationWebGrab",
    ],
    "img_utils.shadow_atlas": "ShadowAtlas",
    "img_utils.rect_packer": ["RectPacker", "RectPackResult"],
    "geo_utils.uv.pack": ["UvPack", "PackIslandsResult"],
    "geo_utils.uv.combine": [
        "UvCombine",
        "CombineShell",
        "CombineGroup",
        "CombinePlan",
        "CombineLayout",
    ],
    "geo_utils.uv.budget": [
        "UvBudget",
        "BudgetItem",
        "BudgetPlan",
        "BudgetRow",
        "BudgetPage",
    ],
    "geo_utils.uv.transfer": ["UvTransfer", "TransferTable"],
    # The Texture Combine panel's Transfer run (UVs or textures), written once
    # over each library's host adapter (``TextureCombine.run_transfer``).
    "geo_utils.uv.transfer_run": [
        "TransferRun",
        "TransferResult",
        "TransferHost",
        "PureTransferHost",
    ],
    "geo_utils.uv.cylinder_seams": "CylinderSeams",
    # Scene-export engine -- the pure half of both DCC Scene Exporters.
    # Declarative widget definitions -> run configuration (the two Scene
    # Exporter panels' export button, once).
    "core_utils.engines.scene_export.export_profile": ["ExportProfile", "ExportRun"],
    # The hierarchy change-detection baseline's set algebra -- ONE baseline per
    # scene, scoped at compare time (shared by both DCC scene exporters) -- and
    # its storage half, which each DCC subclasses with its scene store.
    "core_utils.engines.scene_export.hierarchy_baseline": [
        "HierarchyBaseline",
        "HierarchyBaselineStore",
    ],
    # The lightmap bake baseline: what each map was baked against, the stale
    # rule a re-bake of what changed runs (moved / deformed / neighbours /
    # lighting) and its dry-run report -- one rule for both DCC bakers.
    "core_utils.engines.scene_export.bake_baseline": ["BakeBaseline", "StaleReport"],
    # Scene records -- every tool-authored scene channel declared ONCE (key,
    # scope, version, kind, dependencies); the store contract each DCC's
    # DataNodes implements; the export context + snapshot that assemble the
    # producers in dependency order and commit the carrier in one pass; the
    # crossing of records into another scene.
    "core_utils.engines.scene_export.scene_records": [
        "Scope",
        "RecordSpec",
        "Record",
        "SceneRecords",
        # Held until 0.22.0 (see HELD_UNTIL_0_22 below).
        "GlbBlock",
        "WebProjection",
    ],
    # Held until 0.22.0 (see HELD_UNTIL_0_22 below).
    "core_utils.engines.textures.region_masks": ["RegionMaskManifest"],
    "core_utils.engines.scene_export.scene_store": ["SceneStoreBase"],
    # How a deliverable states itself, for every carrier: the handoff block's
    # instructions, the approved scene's parts, the lighting it was approved
    # under -- one declaration each carrier's writer builds from.
    "core_utils.engines.scene_export.handoff_record": [
        "HandoffRecord",
        "HandoffCarrier",
        "HandoffBlock",
    ],
    # The ``animation`` record's declared shape and its GLB block.
    "core_utils.engines.shots.animation_record": ["AnimationRecord"],
    # The ``shadows`` record's declared shape and its GLB block.
    "geo_utils.shadow_record": ["ShadowRecord"],
    # The lightmap record's model: what both hosts' bakers publish and every
    # reader reads, in one shape -- an FBX's and a GLB's alike.
    "core_utils.engines.scene_export.lightmap_record": ["LightmapRecord"],
    # The materials record's model: a material in its shading family's own
    # terms, stated as glTF states it -- what each host's producer and the
    # FBX reader of a record-less file share.
    "core_utils.engines.scene_export.material_record": ["MaterialFamilies"],
    "core_utils.engines.scene_export.export_snapshot": [
        "ExportContext",
        "ExportSnapshot",
    ],
    "core_utils.engines.scene_export.record_transfer": [
        "TransferContext",
        "RecordTransfer",
    ],
    # The scene-data sidecar's format / naming / migration / diff report, and
    # the exporter's orchestration shell -- each DCC binds both to its host
    # (the sidecar's HOST adapter, the exporter's STORE), overriding nothing.
    "core_utils.engines.scene_export.scene_data_sidecar": [
        "SceneDataSidecarBase",
        # Its host adapter seam, and the pure host a DCC's adapter builds on.
        "SidecarHost",
        "PureSidecarHost",
    ],
    "core_utils.engines.scene_export.scene_exporter": ["SceneExporterBase", "SceneExportHandoff"],
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
        "ShotsEdited",
        "MarkersChanged",
        "StoreInvalidated",
        "NativeStepFollowed",
        "SHOT_PALETTE",
        "SHOT_STATUS_LEVELS",
        # The store's host adapter seam: the protocol each DCC implements
        # (``mtk.MayaShotsHost``, ``btk.BlenderShotsHost``) and its pure defaults,
        # and the hand-off codec its ``transfer()`` returns.
        "ShotsHost",
        "PureShotsHost",
        "ShotTransferHost",
    ],
    # The shot system's record of what IT wrote on scene curves, so those
    # writes can be released when the boundary that caused them moves.
    "core_utils.engines.shots.shot_ledger": [
        "ShotEditLedger",
    ],
    # The store as a hand-off manifest section: mayatk / blendertk encode on
    # the way out and decode on the way in, so shots cross the FBX / USD hop
    # 1:1 in either direction.
    "core_utils.engines.shots.shot_transfer": [
        "ShotTransfer",
    ],
    # How each render effect and audio clip is keyed, once per scene: the
    # panels edit it, the manifest's Build keys with it (a ShotStore field).
    "core_utils.engines.shots.effect_recipe": [
        "EffectRecipe",
    ],
    # Key stash engine — clips of keys parked outside the working animation,
    # inert until retrieved; one store on every host, over the host adapter
    # each DCC implements (``mtk.KeyStash`` / ``btk.KeyStash`` bind theirs and
    # own where the keys physically live) and its pure defaults.
    "core_utils.engines.key_stash.key_stash_model": [
        "KeyStash",
        "StashedClip",
        "StashChanged",
        "KeyStashHost",
        "PureKeyStashHost",
    ],
    # Look match: light a baked scene to match a reference (the engine's own
    # command line is `python -m pythontk.core_utils.engines.lighting.look_match`).
    "core_utils.engines.lighting.light_layers": [
        "LightLayers",
        "LightMix",
        "LayerKind",
        "LightLayerInfo",
        "LightLayerTarget",
        "LightLayerScene",
    ],
    # The rig payload a look match writes and a host's rig tool builds.
    "core_utils.engines.lighting.light_rig_payload": [
        "LightRigPayload",
        "LightRigLight",
        "LightRigFixture",
    ],
    "core_utils.engines.lighting.light_predictor": ["LightPredictor", "ViewerTransform"],
    "core_utils.engines.lighting.look_match.target": ["LookTarget", "LookRegion", "RegionShape"],
    "core_utils.engines.lighting.look_match.measures": ["LookMeasures", "LookScore"],
    "core_utils.engines.lighting.look_match.fit": "LookFit",
    "core_utils.engines.lighting.look_match.sheet": "LookSheet",
    "core_utils.engines.lighting.look_match._look_match": "LookMatch",
    # The session file every look-match reader opens.
    "core_utils.engines.lighting.look_match.session": "LookSession",
    # One round of look matching, from a rig by role to a judged frame: bakes
    # only the roles that changed (a DCC worker), captures, fits, judges.
    "core_utils.engines.lighting.look_match.loop": "LookLoop",
    "core_utils.engines.shots.shot_plan": [
        "ShotMove",
        "MovePlan",
        "GapRetime",
        "ShotPlanner",
        "ShotBoundaryConflict",
    ],
    "core_utils.engines.shots.shot_detection": [
        "TRANSFORM_CHANNELS",
        "STANDARD_TRANSFORM_ATTRS",
        "ShotDetection",
    ],
    # The Shots panels' one-line summaries (sequence stats, trim/pad deltas).
    "core_utils.engines.shots.shot_report": "ShotReport",
    # The sequencer's ripple-editing orchestration; mayatk / blendertk
    # ``ShotSequencer`` subclass it and supply the scene I/O hooks.
    "core_utils.engines.shots.shot_sequencer": [
        "ShotSequencer",
    ],
    # Rig-graph engine — DCC-agnostic model of a rig's INTENT, plus the pure
    # planner that decides what ONE target can build from it. mayatk extracts
    # into it and blendertk builds from it; neither can import the other, and
    # the planner runs a pre-flight with no DCC open.
    "core_utils.engines.rig_graph.rig_model": [
        "RigGraph",
        "RigNode",
        "RigRecord",
        "RigPolicy",
    ],
    "core_utils.engines.rig_graph.rig_capability": [
        "RigCapability",
        "RigOpCapability",
    ],
    "core_utils.engines.rig_graph.rig_verify": ["RigVerify"],
    "core_utils.engines.rig_graph.rig_transfer": ["RigTransfer"],
    "core_utils.engines.rig_graph.rig_machinery": ["RigMachinery"],
    "core_utils.engines.rig_graph.rig_plan": [
        "RigPlanner",
        "PlanResult",
        "ReportEntry",
        "RigPlanRefused",
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
        "PlannedInsert",
        "ObjectStatus",
        "StepStatus",
        "ShotPairing",
    ],
    "core_utils.engines.shots.manifest.manifest_engine": [
        "ShotManifest",
    ],
    "core_utils.engines.shots.manifest.manifest_edits": [
        "ManifestEdits",
    ],
    "core_utils.engines.shots.manifest.manifest_export": [
        "ManifestExport",
    ],
    "core_utils.engines.shots.manifest.manifest_tags": [
        "ManifestTags",
    ],
    "core_utils.engines.shots.manifest.range_resolver": [
        "RangeResolver",
    ],
    # Auto-instancing engine — separated-part clustering core (mayatk / blendertk)
    "core_utils.engines.instancing.assembly_sorter": "AssemblySorter",
    # ...and the grouping pass's signature merge + run summary (both AutoInstancers)
    "core_utils.engines.instancing.instance_grouping": "InstanceGrouping",
    # ...and the strategy decision tree (host bindings supply the triangle count)
    "core_utils.engines.instancing.instancing_strategy": [
        "InstancingStrategy",
        "StrategyConfig",
        "StrategyType",
    ],
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
    "core_utils.engines.textures.map_optimizer": ["MapOptimizer"],
    # Emissive groups as materials: the host-free model each host library
    # drives through its own half (an ``EmissiveGroupsHost``).
    "core_utils.engines.textures.emissive_groups": [
        "EmissiveGroupsModel",
        "EmissiveGroupsHost",
        "PureEmissiveGroupsHost",
    ],
    "core_utils.engines.textures.mat_report": ["MatReport"],
    "core_utils._core_utils": "*",
    "core_utils.help_mixin": "HelpMixin",
    "core_utils.symbol_record": "SymbolRecord",
    # The one engine every reader of the public API goes through: the
    # registries, HelpMixin, the api.* operations, the drift gate.
    "core_utils.api_surface.api_surface": "ApiSurface",
    "core_utils.api_surface.package_surface": [
        "PackageSurface",
        "ModuleSurface",
        "ClassSurface",
    ],
    # One mechanism for retiring public surface, and the only thing in the
    # ecosystem that records WHICH release an alias stops working in.
    "core_utils.deprecation": ["Deprecation", "DeprecationRecord"],
    "core_utils.upstream_patch": ["UpstreamPatch"],
    # Markdown-example rot gate — validates doc code blocks against the live
    # surface (README gates here and downstream build on it).
    "core_utils.doc_audit": "DocAudit",
    "core_utils.package_manager": "PackageManager",
    "core_utils.package_requirement": "PackageRequirement",
    "core_utils.class_property": "ClassProperty",
    # LoggerExt / LevelAwareFormatter are reached by raw module path today
    # (uitk's example wiring, mayatk's scene_audit), which is a published
    # dependency on a private path -- register them so the surface says so.
    "core_utils.logging_mixin": [
        "LoggingMixin",
        "TableMixin",
        "LoggerExt",
        "LevelAwareFormatter",
        "RunLog",
    ],
    # Same case: mayatk's blender_bridge imports ScriptTemplate and the three
    # mode constants straight from this module.
    "core_utils.handoff.script_template": [
        "ScriptTemplate",
        "SEND_TO",
        "SAVE_AS",
        "ROUND_TRIP",
    ],
    "core_utils.namespace_handler": "NamespaceHandler",
    "core_utils.namedtuple_container": "NamedTupleContainer",
    "core_utils.color": ["Color", "ColorPair", "ColorStops", "Palette"],
    "core_utils.hierarchy_utils.hierarchy_diff": "HierarchyDiff",
    "core_utils.singleton_mixin": "SingletonMixin",
    "core_utils.step_toggle": ["StepToggle"],
    "core_utils.module_reloader": ["ModuleReloader", "ReloadReport"],
    "core_utils.execution_monitor._execution_monitor": "ExecutionMonitor",
    "core_utils.cancel_scope": ["CancelScope", "OperationCancelled"],
    # The named, typed actions every front end calls alike (the ops CLI, MCP,
    # panels): a declaration and its published forms, its run context and
    # result payloads, the host seam, and the catalog that discovers,
    # validates and runs them.
    "core_utils.agent_instructions": "AgentInstructions",
    "core_utils.operation_catalog.operation": [
        "Operation",
        "OperationEntry",
        "OperationCatalogDocument",
    ],
    "core_utils.operation_catalog.op_context": ["OpContext"],
    "core_utils.operation_catalog.outcome": ["Outcome", "Issue", "Change", "OpCall", "OpRequest"],
    "core_utils.operation_catalog.batch_operation_host": [
        "BatchOperationHost",
        "SceneFiles",
        "PureSceneFiles",
    ],
    "core_utils.operation_catalog.session_operation_host": "SessionOperationHost",
    "core_utils.python_console": "PythonConsole",
    "core_utils.operation_catalog.operation_host": ["OperationHost", "PureOperationHost"],
    "core_utils.operation_catalog.operation_server": "OperationServer",
    "core_utils.operation_catalog._operation_catalog": "OperationCatalog",
    "core_utils.operation_catalog.operation_provider": "OperationProvider",
    "core_utils.qc_log": ["QcLog", "QcGate", "GateError"],
    "core_utils.status_badge": ["StatusBadge"],
    "core_utils.test_sandbox": ["TestSandbox"],
    # Golden-case documents for every model ported to another language, by
    # name -- what each port's test is held to.
    "core_utils.conformance": ["Conformance"],
    # Exit a DCC-hosted interpreter without running DLL_PROCESS_DETACH, whose
    # static destructors fault in Maya's own teardown and file a crash report
    # on every ordinary exit. os._exit does NOT skip it on Windows, so every
    # entry point that exits from inside a DCC host shares this one.
    "core_utils.process_exit": ["ProcessExit"],
    "core_utils.app_launcher._app_launcher": "AppLauncher",
    "core_utils.x11": "X11",
    "core_utils.app_installer": "AppInstaller",
    "core_utils.handoff.app_handoff": [
        # The carrier vocabulary beside USD_EXTENSIONS: one spelling for every
        # panel (uitk) and every producer (the DCC mixins).
        "CARRIER_PARAM",
        "RIG_MODE_PARAM",
        "RIG_MODES",
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
    # A file delivered by copying it into a folder another application watches
    # (a game engine's project, a hot folder): the deliverer an app shapes, and
    # the bridge for a file already on disk.
    "core_utils.handoff.folder_delivery": ["FolderDeliverer", "FileDeliveryBridge"],
    # The `<payload>.manifest.json` sidecar every conversion ships beside its
    # FBX / USD intermediate: the suffix, the section vocabulary and the
    # read/write rules, so both producers and both consumers spell one contract
    # instead of eighty string literals. A section's CONTENTS stay with the
    # codec that owns them (e.g. ShotTransfer for `shots`).
    "core_utils.handoff.manifest": ["HandoffManifest"],
    # The Scope words (selected / all / visible) and their precedence: every
    # bridge panel, the preview push and the bakers resolve through the one
    # rule, each host passing its own scene reads in as lookups.
    "core_utils.handoff.handoff_scope": "HandoffScope",
    # Replay of those sections: an ordered, gated, best-effort-per-step plan
    # with one progress/cancel protocol, so adding a section is a line rather
    # than an edit to a hand-written chain in every consumer.
    "core_utils.handoff.manifest_plan": ["ManifestPlan"],
    # Blocking run-script-collect-artifact counterpart of ScriptLaunchDeliverer
    # (backs pull-direction bridges, e.g. blendertk's Maya-scene import).
    "core_utils.handoff.script_run": [
        "ScriptRunResult",
        "ScriptRunner",
        "ProgressRelay",
    ],
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
    "core_utils.naming_convention._naming_convention": [
        "AffixRule",
        "NamingConvention",
    ],
    "core_utils.presets.store": [
        "PresetStore",
        "PresetReadOnlyError",
        "Codec",
        "JSON_CODEC",
    ],
    # Every preset store under the root at once: lock, collections, bundles.
    "core_utils.presets.library": [
        "PresetLibrary",
        "PresetDomain",
        "PresetEntry",
        "ImportPlan",
        "ImportItem",
        "ImportResult",
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
    # A loopback port at a public HTTPS link, through a tunnel CLI (Cloudflare,
    # Tailscale) -- how a preview, or any local server, is shared by link.
    "net_utils.share_tunnel": "ShareTunnel",
    # Loopback static server + live manifest behind the WebXR/browser preview
    # loop. Localhost is a secure context, so this is all `navigator.xr` needs.
    "net_utils.preview.server": "PreviewServer",
    # Builds the GLB through the shared GlbPipeline and publishes it.
    "net_utils.preview.deliverer": "PreviewDeliverer",
    # ``PreviewBridge`` is the skeleton a DCC binds its export mixin to;
    # ``FilePreviewBridge`` is the host-free peer that previews an FBX or GLB
    # already on disk, so a panel outside any DCC drives the same chain.
    "net_utils.preview.bridge": [
        "PreviewBridge",
        "FilePreviewBridge",
    ],
    # The viewer, headless, for lighting work: a mix's bake swapped into a
    # template and shot -- the ground truth the look-match prediction is held to.
    "net_utils.preview.look_capture": "LookCapture",
    "net_utils.rpc.installer": [
        "PluginInstaller",
        "RpcPluginSpec",
    ],
    "net_utils.rpc.job": ["RpcJob"],
    # The in-application half of the RPC pair (`client` drives it from outside).
    # Exposed here for the plugin that loads in place and can import pythontk;
    # installed plugins stage `plugin_core.py` into their payload instead.
    "net_utils.rpc.plugin_core": ["OpRegistry", "MainThreadMarshaller", "RpcInstances", "RpcPlugin"],
    "str_utils.fuzzy_matcher": "FuzzyMatcher",
    "str_utils.hotkey_utils": "HotkeyUtils",
    # Pure-string rich-tooltip DSL; uitk's TooltipPresenter shows the result.
    "str_utils.tooltip_format": "TooltipFormat",
    "str_utils.report_doc": ["ReportDoc"],
}


# ``bootstrap_package`` derives ``__all__`` automatically from the resolved
# public surface (DEFAULT_INCLUDE), so it never has to be hand-maintained here.
#: Public names that left this package (CODE_STANDARD section 5): no alias, so a
#: read raises MovedNameError naming the new home. ``{name: (new home, version)}``;
#: a move adds its names here in the same change set.
MOVED_NAMES = {
    # A flat wrapper over the class it fronted (CODE_STANDARD section 4).
    "reload_package": ("pythontk.ModuleReloader().reload", "0.19.0"),
    # Generic names off the root, on the class that owns them (the policy above).
    "Kind": ("pythontk.SceneRecords.Kind", "0.19.0"),
    "Merge": ("pythontk.SceneRecords.Merge", "0.19.0"),
    "Op": ("pythontk.MapOptimizer.Op", "0.19.0"),
    # Removed with the "speedrun" translation (2026-10-27, no replacement): a
    # retired template name now raises FileNotFoundError in load_mapping.
    "Mapping.retired": (
        "pythontk.core_utils.engines.shots.manifest.mapping.Mapping.load_mapping",
        "0.19.0",
    ),
    # The Scene Exporter panel reads its own rows (pythontk knows no Qt).
    "ExportProfile.read_values": (
        "tentacle.slots.tools.scene_exporter.SceneExporterSlots",
        "0.19.0",
    ),
    "ExportProfile.value_method": (
        "tentacle.slots.tools.scene_exporter.SceneExporterSlots",
        "0.19.0",
    ),
    "ExportProfile.CHECKBOX": (
        "tentacle.slots.tools.scene_exporter.SceneExporterSlots",
        "0.19.0",
    ),
    # Where a node-keyed record lists its nodes, declared on the record itself.
    "FbxMedia.NODE_ENTRIES": ("pythontk.RecordSpec.nodes", "0.19.0"),
    # The scene sidecar's material sections: the FBX's materials record states
    # every material (removed 2026-10-31).
    "MeshConvert.apply_scene_sidecar": ("pythontk.MeshConvert.apply_glb_materials", "0.19.0"),
    "MeshConvert.SIDECAR_APPLIERS": ("pythontk.MeshConvert.apply_glb_materials", "0.19.0"),
    "MeshConvert.embed_scene_sidecar": ("pythontk.MeshConvert.embed_glb_handoff", "0.19.0"),
    "MeshConvert.sidecar_foreign_packings": ("pythontk.MapFactory.foreign_packings", "0.19.0"),
    # The per-channel writers those sections ran (removed 2026-10-31).
    "MeshConvert.set_glb_base_color": ("pythontk.MeshConvert.apply_glb_materials", "0.19.0"),
    "MeshConvert.set_glb_emissive": ("pythontk.MeshConvert.apply_glb_materials", "0.19.0"),
    "MeshConvert.set_glb_metallic_roughness": ("pythontk.MeshConvert.apply_glb_materials", "0.19.0"),
    "MeshConvert.set_glb_alpha_mode": ("pythontk.MeshConvert.apply_glb_materials", "0.19.0"),
    "MeshConvert.set_glb_unlit": ("pythontk.MeshConvert.apply_glb_materials", "0.19.0"),
    # The host seam's one convention: a <Domain>Host Protocol beside the model,
    # a Pure<Domain>Host default, a HOST class attribute (2026-11-01).
    "EmissiveGroupsAdapter": ("pythontk.EmissiveGroupsHost", "0.19.0"),
    "EmissiveGroupsModel.adapter": ("pythontk.EmissiveGroupsModel.HOST", "0.19.0"),
    # The removed tasks["version"] pattern it carried: the Output Filename's own
    # {n} counter states a version (removed 2026-11-01).
    "ExportRun.version_format": ("pythontk.ExportProfile.VERSION_TOKEN", "0.19.0"),
    # One take per shot (2026-11-05): a file's takes are its shots, named after
    # them, so the Animation Clips modes (full / shots / both), the shot record's
    # frame ranges and the whole-timeline clip bookkeeping retire. Take ranges
    # at export time are the store's; after the write, the file's own.
    "ExportProfile.ANIMATION_CLIPS_OPTIONS": ("pythontk.ShotStore.declared_takes", "0.21.0"),
    "ExportRun.clip_mode": ("pythontk.ShotStore.declared_takes", "0.21.0"),
    "ExportRun.animation_clips_mode": ("pythontk.ExportRun.splits_takes", "0.21.0"),
    "ExportContext.clip_mode": ("pythontk.ShotStore.declared_takes", "0.21.0"),
    "ExportContext.clip_span": ("pythontk.FbxFile.take_spans", "0.21.0"),
    "SceneRecords.declared_takes": ("pythontk.ShotStore.declared_takes", "0.21.0"),
    "ShotStore.CLIP_MODE_KEY": ("pythontk.ShotStore.declared_takes", "0.21.0"),
    "MeshConvert.apply_glb_clips": ("pythontk.MeshConvert.apply_glb_animations", "0.21.0"),
    "MeshConvert.clip_spans": ("pythontk.FbxFile.take_spans", "0.21.0"),
    "MeshConvert.SHOT_CLIP_MODE_KEY": ("pythontk.ShotStore.declared_takes", "0.21.0"),
    "MeshConvert.ANIMATION_CLIP_MODES": ("pythontk.ShotStore.declared_takes", "0.21.0"),
    "MeshConvert.DEFAULT_CLIP_SPAN": ("pythontk.FbxFile.take_spans", "0.21.0"),
    "MeshConvert.CLIP_SPAN_DRIFT_FRAMES": ("pythontk.FbxFile.take_spans", "0.21.0"),
    "GlbClips.rebuild": ("pythontk.MeshConvert.apply_glb_animations", "0.21.0"),
    "ExportVerifier.check_clip_origin": (
        "pythontk.ExportVerifier.check_clips_vs_takes",
        "0.21.0",
    ),
    # One layout, two containers (2026-11-05): a GLB carries the 'animation'
    # record as it stands, each shot joined to the animation of its clip's
    # name, so the block that bound the shots to animation indices retires.
    "AnimationBlock": ("pythontk.AnimationRecord", "0.21.0"),
    "AnimationBlockClip": ("pythontk.AnimationRecord", "0.21.0"),
    # The 'audio' record v3 (2026-11-05): cues per clip in seconds, each with
    # its sound by the delivered file's name, its offset and its stop -- so
    # the frame scoping of one event per start retires, and the FBX reader
    # names the delivered sound files rather than track names.
    "AudioUtils.scope_events": ("pythontk.AudioUtils.clip_cues", "0.21.0"),
    "FbxMedia.audio_events": ("pythontk.FbxMedia.audio_sounds", "0.21.0"),
    # The 'render_effects' record v2 (2026-11-05): it names channels and the
    # curves ride the takes on their proxies, so the writers placing record
    # ramps per clip by frame retire for the one reading the file's curves.
    "GlbFades.apply": ("pythontk.GlbFades.apply_curves", "0.21.0"),
    "GlbFades.apply_channels": ("pythontk.GlbFades.apply_curves", "0.21.0"),
    # Emissive groups become materials (2026-11-07): a group's faces wear a
    # copy of their material per group, its emission keyed by the weight, so
    # the masks (vertex colours, the _EMask texture), the slots and the
    # emissive_groups record retire; an old scene converts in one action.
    "RegionGroup": ("pythontk.EmissiveGroupsModel", "0.21.0"),
    "RegionGroupRegistry": ("pythontk.EmissiveGroupsModel", "0.21.0"),
    "RegionMaskPacker": ("pythontk.EmissiveGroupsModel.convert_legacy", "0.21.0"),
    "EmissiveGroupsRecord": ("pythontk.RenderEffectsRecord", "0.21.0"),
    "EmissiveGroupsModel.set_default": ("pythontk.EmissiveGroupsModel.set_weight", "0.21.0"),
    "EmissiveGroupsModel.make_weights_keyable": ("pythontk.EmissiveGroupsModel.key_weight", "0.21.0"),
    "EmissiveGroupsModel.remove_keyable_weights": ("pythontk.EmissiveGroupsModel.remove_keys", "0.21.0"),
    "EmissiveGroupsModel.weight_attrs": ("pythontk.EmissiveGroupsModel.list_groups", "0.21.0"),
    "EmissiveGroupsModel.compact_slots": ("pythontk.EmissiveGroupsModel.convert_legacy", "0.21.0"),
    "EmissiveGroupsModel.bake_vertex_colors": ("pythontk.EmissiveGroupsModel.convert_legacy", "0.21.0"),
    "EmissiveGroupsModel.bake_mask": ("pythontk.EmissiveGroupsModel.convert_legacy", "0.21.0"),
    "EmissiveGroupsModel.export_record": ("pythontk.EmissiveGroupsModel.render_effect_materials", "0.21.0"),
    "EmissiveGroupsModel.refresh_export_metadata": ("pythontk.EmissiveGroupsModel.render_effect_materials", "0.21.0"),
    "EmissiveGroupsModel.merge_carrier": ("pythontk.EmissiveGroupsModel.transfer_in", "0.21.0"),
    "EmissiveGroupsModel.discard_carrier": ("pythontk.EmissiveGroupsModel.transfer_in", "0.21.0"),
    "SceneRecords.EMISSIVE_GROUPS": ("pythontk.SceneRecords.VISIBILITY", "0.21.0"),
    # The host seam those masks spoke (stores, carrier weights, colour sets,
    # mask bakes) gives way to the material seam: each old member to the one
    # that now does its job, or, for the masks, to the conversion's removal.
    "EmissiveGroupsHost.add_members": ("pythontk.EmissiveGroupsHost.assign", "0.21.0"),
    "EmissiveGroupsHost.remove_members": ("pythontk.EmissiveGroupsHost.delete_material", "0.21.0"),
    "EmissiveGroupsHost.members": ("pythontk.EmissiveGroupsHost.material_faces", "0.21.0"),
    "EmissiveGroupsHost.membership_stores": ("pythontk.EmissiveGroupsHost.legacy_store_names", "0.21.0"),
    "EmissiveGroupsHost.carrier_weights": ("pythontk.EmissiveGroupsHost.copies", "0.21.0"),
    "EmissiveGroupsHost.weight_exists": ("pythontk.EmissiveGroupsHost.copies", "0.21.0"),
    "EmissiveGroupsHost.ensure_weight": ("pythontk.EmissiveGroupsHost.set_emission", "0.21.0"),
    "EmissiveGroupsHost.delete_weight": ("pythontk.EmissiveGroupsHost.clear_emission_keys", "0.21.0"),
    "EmissiveGroupsHost.sync_weight": ("pythontk.EmissiveGroupsHost.set_emission", "0.21.0"),
    "EmissiveGroupsHost.key_weight": ("pythontk.EmissiveGroupsHost.key_emission", "0.21.0"),
    "EmissiveGroupsHost.foreign_color_sets": ("pythontk.EmissiveGroupsHost.remove_legacy", "0.21.0"),
    "EmissiveGroupsHost.write_colors": ("pythontk.EmissiveGroupsHost.remove_legacy", "0.21.0"),
    "EmissiveGroupsHost.uv_triangles": ("pythontk.EmissiveGroupsHost.remove_legacy", "0.21.0"),
    "EmissiveGroupsHost.mask_path": ("pythontk.EmissiveGroupsHost.remove_legacy", "0.21.0"),
    "EmissiveGroupsHost.object_names": ("pythontk.EmissiveGroupsHost.faces", "0.21.0"),
    "EmissiveGroupsHost.face_count": ("pythontk.EmissiveGroupsHost.material_faces", "0.21.0"),
    "EmissiveGroupsHost.mesh": ("pythontk.EmissiveGroupsHost.material", "0.21.0"),
    "EmissiveGroupsHost.merge_stores": ("pythontk.EmissiveGroupsHost.copies", "0.21.0"),
    "EmissiveGroupsHost.discard_stores": ("pythontk.EmissiveGroupsHost.remove_legacy", "0.21.0"),
    "EmissiveGroupsHost.publish": ("pythontk.EmissiveGroupsHost.stage_emission_proxy", "0.21.0"),
    "PureEmissiveGroupsHost.add_members": ("pythontk.PureEmissiveGroupsHost.assign", "0.21.0"),
    "PureEmissiveGroupsHost.remove_members": ("pythontk.PureEmissiveGroupsHost.delete_material", "0.21.0"),
    "PureEmissiveGroupsHost.members": ("pythontk.PureEmissiveGroupsHost.material_faces", "0.21.0"),
    "PureEmissiveGroupsHost.membership_stores": ("pythontk.PureEmissiveGroupsHost.legacy_store_names", "0.21.0"),
    "PureEmissiveGroupsHost.carrier_weights": ("pythontk.PureEmissiveGroupsHost.copies", "0.21.0"),
    "PureEmissiveGroupsHost.weight_exists": ("pythontk.PureEmissiveGroupsHost.copies", "0.21.0"),
    "PureEmissiveGroupsHost.ensure_weight": ("pythontk.PureEmissiveGroupsHost.set_emission", "0.21.0"),
    "PureEmissiveGroupsHost.delete_weight": ("pythontk.PureEmissiveGroupsHost.clear_emission_keys", "0.21.0"),
    "PureEmissiveGroupsHost.sync_weight": ("pythontk.PureEmissiveGroupsHost.set_emission", "0.21.0"),
    "PureEmissiveGroupsHost.key_weight": ("pythontk.PureEmissiveGroupsHost.key_emission", "0.21.0"),
    "PureEmissiveGroupsHost.foreign_color_sets": ("pythontk.PureEmissiveGroupsHost.remove_legacy", "0.21.0"),
    "PureEmissiveGroupsHost.write_colors": ("pythontk.PureEmissiveGroupsHost.remove_legacy", "0.21.0"),
    "PureEmissiveGroupsHost.uv_triangles": ("pythontk.PureEmissiveGroupsHost.remove_legacy", "0.21.0"),
    "PureEmissiveGroupsHost.mask_path": ("pythontk.PureEmissiveGroupsHost.remove_legacy", "0.21.0"),
    "PureEmissiveGroupsHost.object_names": ("pythontk.PureEmissiveGroupsHost.faces", "0.21.0"),
    "PureEmissiveGroupsHost.face_count": ("pythontk.PureEmissiveGroupsHost.material_faces", "0.21.0"),
    "PureEmissiveGroupsHost.mesh": ("pythontk.PureEmissiveGroupsHost.material", "0.21.0"),
    "PureEmissiveGroupsHost.merge_stores": ("pythontk.PureEmissiveGroupsHost.copies", "0.21.0"),
    "PureEmissiveGroupsHost.discard_stores": ("pythontk.PureEmissiveGroupsHost.remove_legacy", "0.21.0"),
    "PureEmissiveGroupsHost.publish": ("pythontk.PureEmissiveGroupsHost.stage_emission_proxy", "0.21.0"),
    "ENCODING_CHANNELS": ("pythontk.EmissiveGroupsModel.convert_legacy", "0.21.0"),
    "ENCODING_ID": ("pythontk.EmissiveGroupsModel.convert_legacy", "0.21.0"),
    "ENCODING_VERTEX_COLOR": ("pythontk.EmissiveGroupsModel.convert_legacy", "0.21.0"),
    "SLOT_CHANNELS": ("pythontk.EmissiveGroupsModel.convert_legacy", "0.21.0"),
    "EmissiveGroupsEntry": ("pythontk.RenderEffectsRecord", "0.21.0"),
    # The 'lightmaps' record v3 (2026-11-07): one layout, two containers -- a
    # GLB carries the record as it stands, its files by bufferView name
    # (SceneRecords.GLB_MEDIA), so the index-bound block retires; the probe's
    # files are the delivery's in both containers, named for what they are.
    "LightmapsBlock": ("pythontk.LightmapRecord", "0.21.0"),
    "LightmapsBlockMaterial": ("pythontk.LightmapRecord", "0.21.0"),
    "LightmapsBlockProbe": ("pythontk.LightmapRecord", "0.21.0"),
    "LightmapsBlockHighlight": ("pythontk.LightmapRecord", "0.21.0"),
    "MeshConvert.FBX_HIGHLIGHT_NAME": ("pythontk.MeshConvert.HIGHLIGHT_COPY_NAME", "0.21.0"),
    # The GLB build carries that record and its files (2026-11-07): it binds
    # no material and clones none, so the block's builder, the clone marker
    # and the probe's separate embed go; the WebXR viewer binds at load.
    "MeshConvert.lightmaps_block": ("pythontk.SceneRecords.LIGHTMAPS", "0.21.0"),
    "MeshConvert.embed_glb_lightmap_probe": ("pythontk.MeshConvert.apply_glb_lightmaps", "0.21.0"),
    "MeshConvert.LIGHTMAP_CLONE_SUFFIX": ("pythontk.MeshConvert.lit_nodes", "0.21.0"),
    # The probe's highlight copies are the lightmaps record's own row's on
    # every carrier (2026-11-07), so the per-carrier text that said them --
    # empty on both since -- goes with its lighting clause.
    "HandoffCarrier.highlights": ("pythontk.SceneRecords.LIGHTMAPS.clauses", "0.21.0"),
    # The 'shadows' record v3 (2026-11-07): one layout, two containers -- a
    # GLB carries the record as it stands, its maps by bufferView name
    # (SceneRecords.GLB_MEDIA), so the index-bound block retires; the plane's
    # fields are one shape (no carrier shares them any more).
    "ShadowBlock": ("pythontk.ShadowRecord", "0.21.0"),
    "ShadowBlockPlane": ("pythontk.ShadowRecord", "0.21.0"),
    "ShadowBlockHorizon": ("pythontk.ShadowRecord", "0.21.0"),
    "ShadowPlaneModel": ("pythontk.ShadowRecord", "0.21.0"),
    # The GLB build carries that record and its files (2026-11-07): nothing is
    # bound by index and no glTF texture is made, so the sampler the horizon
    # map's texture took goes; a reader samples the map as the record says it
    # is read (whole texels: the WebXR viewer's texture, no mipmaps, clamped).
    "MeshConvert.SHADOW_DATA_SAMPLER": ("pythontk.SceneRecords.SHADOWS", "0.21.0"),
    # The 'articulation' record v2 (2026-11-08): one layout, two containers --
    # a GLB carries the record as it stands, its joints and parts bound by
    # name, so the index-bound block (and the aliases it was renamed from)
    # retires; the record is in metres (glTF's frame), so no runtime measures
    # its unit off the rest pose.
    "ArticulationBlock": ("pythontk.ArticulationRecord", "0.21.0"),
    "ArticulationBlockRig": ("pythontk.ArticulationRecord", "0.21.0"),
    "ArticulationBlockJoint": ("pythontk.ArticulationRecord", "0.21.0"),
    "ArticulationBlockGrab": ("pythontk.ArticulationRecord", "0.21.0"),
    "ArticulationModel.scale_of": ("pythontk.SceneRecords.FRAME", "0.21.0"),
    # The GLB block machinery retires whole with it (2026-11-08): articulation
    # was the last record a GLB carried bound to its indices, so a GLB carries
    # each record it carries as it stands (RecordSpec.glb_verbatim; False for
    # the deliverables a GLB states otherwise -- materials, as glTF's own; the
    # handoff, the GLB's own -- and every private record) -- nothing declares
    # a block, its node fields or its version apart from its record's, and no
    # build pass renumbers one when it removes nodes. A block version is its
    # record's: SceneRecords.resolve(item).version.
    "RecordSpec.glb": ("pythontk.RecordSpec.glb_verbatim", "0.21.0"),
    "SceneRecords.glb_shape": ("pythontk.SceneRecords.shape", "0.21.0"),
    "SceneRecords.glb_version": ("pythontk.SceneRecords.resolve", "0.21.0"),
    "SceneRecords.block_problems": ("pythontk.SceneRecords.problems", "0.21.0"),
}

#: Held until 0.22.0: names v0.20.0 noticed for removal in 0.18.0 on
#: 2026-10-24, whose window closes 2026-11-23, after 0.21.0 ships -- restored
#: as working deprecated aliases (``scene_export/_retired.py``,
#: ``geo_utils/articulation/_retired.py``, ``textures/region_masks.py``,
#: ``RecordSpec.web``, ``SceneRecords.web_shape``, the three
#: ``MeshConvert.*_WEB_*`` class attributes; the test class
#: ``TestRetiredUntil022``). At 0.22.0 delete them and their DEFAULT_INCLUDE
#: entries, and move these rows into MOVED_NAMES.
HELD_UNTIL_0_22 = {
    "GlbBlock": ("pythontk.RecordSpec.glb_verbatim", "0.22.0"),
    "WebProjection": ("pythontk.RecordSpec.glb_verbatim", "0.22.0"),
    "RecordSpec.web": ("pythontk.RecordSpec.glb_verbatim", "0.22.0"),
    "SceneRecords.web_shape": ("pythontk.SceneRecords.shape", "0.22.0"),
    "ArticulationWeb": ("pythontk.ArticulationRecord", "0.22.0"),
    "ArticulationWebRig": ("pythontk.ArticulationRecord", "0.22.0"),
    "ArticulationWebJoint": ("pythontk.ArticulationRecord", "0.22.0"),
    "ArticulationWebGrab": ("pythontk.ArticulationRecord", "0.22.0"),
    "RegionMaskManifest": ("pythontk.EmissiveGroupsModel.render_effect_materials", "0.22.0"),
    "RegionMaskManifest.SCHEMA_VERSION": ("pythontk.EmissiveGroupsModel.render_effect_materials", "0.22.0"),
    "MeshConvert.ANIMATION_WEB_VERSION": ("pythontk.SceneRecords.SHOTS.version", "0.22.0"),
    "MeshConvert.SHADOW_WEB_NODE_FIELDS": ("pythontk.SceneRecords.SHADOWS.nodes", "0.22.0"),
    "MeshConvert.ARTICULATION_WEB_NODE_FIELDS": ("pythontk.SceneRecords.ARTICULATION.nodes", "0.22.0"),
}

bootstrap_package(globals(), include=DEFAULT_INCLUDE, moved=MOVED_NAMES)

# Generic names off the root, like Kind / Merge / Op above, but aliased for a window:
# the released extapps imports these two at module top and publishes off the
# cascade, so a moved-name error would break every install that upgrades
# pythontk alone. Chains onto the bootstrap's ``__getattr__``.
from pythontk.core_utils.deprecation import Deprecation as _Deprecation  # noqa: E402

_Deprecation.attributes(
    globals(),
    {"Call": "pythontk.RpcJob.Call", "Result": "pythontk.RpcJob.Result"},
    remove_in="0.20.0",
    since="2026-10-29",
)
del _Deprecation
