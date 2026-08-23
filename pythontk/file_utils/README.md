# file_utils

Filesystem primitives plus the file-*format* utilities built on them: atomic writes, policy-scoped temp artifacts, mesh conversion and processing, zero-dependency USD authoring, UV unwrapping, project workspaces, and embedded metadata.

Everything is reachable from the package root (`import pythontk as ptk`). Full signatures: [`API_INDEX.md`](../../API_INDEX.md) / [`API_REGISTRY.md`](../../API_REGISTRY.md).

## `FileUtils` (`_file_utils.py`)

The path-plumbing workhorse, in five clusters:

- **Validation & environment sanity** — `is_valid`, `is_under`, `is_rooted_path`, `exceeds_path_length`, `free_space` / `format_bytes`, and `is_cloud_placeholder`, which reads the Windows OFFLINE / RECALL_ON_OPEN file attributes to flag dehydrated OneDrive/Dropbox/Drive files *without triggering a recall*.
- **Traversal & filesystem ops** — `get_dir_contents` (content-type selection `file`/`filename`/`filepath`/`dir`/`dirpath`, recursion, include/exclude wildcard filters for files and dirs), `create_dir`, `next_version_path`, `copy_file` / `move_file`, `open_explorer` / `reveal_in_file_manager`.
- **Reads / writes** — `get_file_contents`, `write_to_file`, and `atomic_write_text`: temp file in the same directory then `os.replace`, so a concurrent reader (or a cloud-sync client) sees old-or-complete, never partial.
- **Path formatting / remapping** — `format_path`, `convert_to_relative_path`, `remap_file_paths`, `append_path`.
- **Introspection & JSON** — `get_classes_from_path` (plugin discovery from a directory or `.py` file — uses the canonical package import when an `__init__.py` chain exists so returned class objects match a normal import), plus a small JSON key/value store (`set_json` / `get_json`).

## Temp artifacts (`temp_artifacts.py`)

`TempArtifacts` replaces raw `mkstemp`/`mkdtemp`: a raw allocation has no owner and leaks when a `finally` never runs (routine when a DCC process dies). Allocations are prefix-scoped with an explicit **lifetime policy** — `"scoped"` (deleted on clean exit, *kept on exception* so failures stay debuggable), `"session"` (removed at interpreter exit, for detached consumers like a launched DCC), `"detached"` (no deterministic delete exists, so allocation age-sweeps stale same-prefix files instead). Every policy runs the stale sweep on first allocation, so the worst case is delayed collection, never a permanent leak. Workspace-wide use is enforced by `m3trik/scripts/check_temp_artifacts.py`.

`CachedArtifact` layers produce-once/reuse-forever on top: inputs hash to a content-addressed key; a miss produces into a *scoped scratch* store and atomically `os.replace`s into the cache slot, so a timeout-killed partial write can never poison later runs.

`ScratchTwins` is the third: a per-source scratch **twin** of a foreign file (a `.blend` opened in Maya arrives as an `.ma`). The twin path is deterministic per source, so a panel can recompute it to answer "is this row the current scene?", and each source gets its own directory so two projects' `shot.ma` cannot collide. A twin is discarded only while it is still byte-for-byte the copy that was written — one the user saved into is real work, so it is kept and its location logged, and `create` moves it aside rather than over it. The untouched-test is stamped to a sidecar so it survives a host restart.

## Mesh conversion (`mesh_convert/`)

`MeshConvert` wraps the godotengine **FBX2glTF** CLI (pinned, auto-installed into `~/.pythontk/tools/` on first use — override with `PYTHONTK_TOOLS_DIR`) for static-mesh FBX → GLB, then acts as a glTF/GLB repair-and-enrichment toolkit where multiple passes share one edit session:

- **Scene sidecar** — `build_scene_sidecar` / `apply_scene_sidecar` / `read_scene_sidecar`: a versioned envelope embedded into the glTF root `extras`, making the deliverable self-describing to any glTF tool with no side files. The schema has exactly one home here, so producers (DCC exporters, WebXR bridges) cannot fork it.
- **Lightmaps** — `apply_glb_lightmaps` encodes baked HDR EXRs for web and binds them as `occlusionTexture` on `TEXCOORD_1` (glTF has no lightmap slot; occlusion is the shared convention, and naive viewers degrade to grey AO). No manifest = clean no-op.
- **GLB passes** — `optimize_glb_textures` (WebP default, or KTX2/Basis via `KHR_texture_basisu`; KTX2 needs the external `toktx` encoder), `check_glb_materials`, `fix_glb_phantom_opaque_alpha`, `set_glb_base_color` / `set_glb_metallic_roughness` / `set_glb_emissive`.

The end-to-end DCC → GLB → headset pipeline this serves is documented in [Live WebXR preview](../../docs/webxr_preview.md).

## USD (`usd.py`) — zero-dependency

Pure Python, explicitly no `pxr` import. `UsdFile` sniffs format by magic bytes and inspects USDZ packages; `UsdzPackager` is a spec-compliant `.usdz` writer/verifier over stdlib `zipfile` (stored-never-compressed, 64-byte alignment, default layer first); `UsdMeshWriter` authors a textured mesh as a `.usda` layer with `UsdPreviewSurface`, composed by `obj_to_usd` / `obj_to_usdz` — the no-DCC, no-license publish path (e.g. Metashape OBJ → QuickLook-ready USDZ). The dependency-free floor beneath `mayatk`/`blendertk`'s own USD adapters.

## Mesh processing (`mesh_ops.py`)

`MeshOps`: file-level mesh processing over optional **PyMeshLab** (`pip install pythontk[mesh]`; gate: `MeshOps.available()`), path in → path out — the ecosystem's headless mesh floor beneath the DCC adapters. Three tiers, the `UvUnwrap` registry design applied to filters:

- **Typed methods** — `measure` (flat, gate-safe metrics dict: values are numeric or `None`, never a fabricated 0), `compare` (Hausdorff deviation between two meshes — keys say `hausdorff_peak`, not `max`, because `QcGate` strips rule prefixes with an unanchored replace), `clean` (the repair chain: weld → prune islands → non-manifold repair → hole fill → optional decimate), `remesh` (isotropic — the evenness pass scans need *before* collapse), `decimate` (quadric collapse; `curvature_weighted=True` bakes ABS curvature into vertex quality for Decimation-Master-style adaptive density), and `bake_vertex_color` (per-vertex color → texture on trivial per-wedge UVs — a bake carrier, not production UVs; use `UvUnwrap` for real layouts).
- **The `OPS` registry** — curated single-filter operators with validated params and `PercentageValue`/`PureValue` wrapping declared per-spec; adding one is one `OpSpec` entry.
- **`apply()` / `session()`** — the escape hatch to any PyMeshLab filter by name (unvalidated), and a context-managed `MeshSet` session so composed ops skip the load → save → reload round-trip.

## UV unwrapping (`uv_unwrap/`)

`UvUnwrap`: automatic unwrapping via external CLI engines, OBJ in → OBJ out, behind product-facing names — `hard_surface` (**Ministry of Flat**; discovery-only, its license forbids redistribution so the binary is never downloaded on your behalf) and `organic` (**Boundary First Flattening**; MIT, pinned + checksum-verified, auto-installable). Both preserve input topology exactly so callers can map UVs back by component index. Success is judged by the output file, not the exit code (MoF returns 1 even on success).

## Workspaces (`workspace.py`)

Shared project-workspace model plus a `workspace.mel` codec — pure Python, no DCC import. Despite the extension, `workspace.mel` is a flat, order-tolerant rule store that Maya round-trips losslessly *including rules it doesn't recognize*, which makes it a legitimate shared format: one project folder serves Maya natively and Blender via blendertk. The writer is merge-preserving (managed rules update in place; unrecognized rules, comments, and variables survive verbatim). `Workspace` resolves directories semantically (rule → existing conventional folder → default); `WorkspaceTemplates` holds named rule sets for new-project scaffolding.

## Batch renaming (`file_naming.py`)

`RenamePlan` is the host-agnostic executor both DCC naming panels share: it takes a plan of `(item, new_name)` and a `rename` **strategy callable**, applies it entry by entry, and reports what changed — collision handling, dry-run, per-entry failure isolation and the log/link formatting in one place, with no filesystem or DCC call of its own. `FileNaming` is its filesystem tenant: `expand` (paths or directories to files), `find`, `rename`, `set_case` and `strip_chars`, driving `RenamePlan` with an `os.rename` strategy. A DCC layer supplies its own strategy instead (`cmds.rename`, `object.name = …`) and gets the same behaviour for free.

## Others

- **`metadata.py`** — `Metadata`: cross-platform file metadata/tag read-write, native where possible (Windows property system via pywin32) with an opt-in hidden JSON sidecar; works sidecar-only without pywin32.

## Dependency gating

No third-party dependencies: `usd`, `workspace`, `temp_artifacts`. Optional, feature-gated: `mesh_ops` (PyMeshLab via the `pythontk[mesh]` extra), `metadata` (pywin32, sidecar fallback), `mesh_convert` (FBX2glTF binary auto-installed; `toktx` for KTX2), `uv_unwrap` (external CLIs, one auto-installable).

## Links

- Package overview: [`docs/README.md`](../../docs/README.md)
- WebXR preview pipeline: [`docs/webxr_preview.md`](../../docs/webxr_preview.md)
