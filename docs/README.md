[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/pythontk.svg)](https://pypi.org/project/pythontk/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-4188%20passed-brightgreen.svg)](../test/)

# pythontk

<!-- short_description_start -->
*Composable Python primitives for files, strings, iteration, math, geometry, images, video, audio, and networking — the DCC-agnostic foundation of a game-art tooling ecosystem.*
<!-- short_description_end -->

Pure Python: no Qt, no DCC imports, two hard dependencies (`numpy`, `Pillow`). Everything heavier — FFmpeg, OpenCV, rembg, PyMeshLab — is optional and feature-gated, so the core runs identically in `mayapy`, Blender's Python, a CI runner, or a bare venv.

## Why

pythontk is the bottom of the chain `pythontk → uitk → mayatk / blendertk → tentacle`: everything above imports it; it imports nothing above it. The environment-independent 80% of every tool lives at this layer — in effect, the ecosystem's standard library.

Two rules shape it:

- **Placed by data type, not domain.** Sharpest-frame extraction lives in `vid_utils`, perceptual-hash curation in `img_utils` — not in a "photogrammetry" package — so each primitive stays independently reusable. Domain pipelines (PBR conversion, photogrammetry ingest, timeline audio events) are compositions of these, assembled downstream.
- **Shared code moves down.** When two downstream tools need the same helper, it moves here and becomes the single source of truth — Maya and Blender panels share one calculator engine, one material-report formatter, one point-clustering routine, instead of drifting copies.

## Install

```bash
pip install pythontk
```

**Installing into a DCC that ships its own `numpy`.** The two hard dependencies are declared floor-only (`numpy>=1.24`, `Pillow`) and deliberately carry no ceiling: a Python environment marker sees the interpreter, never the application hosting it, and the hosts want opposite bounds — Maya 2025 ships numpy 1.24.4 with extensions compiled against 1.x, Blender 5.1 ships 2.3.4. Installing *into the host* (`mayapy -m pip install pythontk`) is fine either way: pip leaves a numpy that already satisfies the floor alone. Installing to a **fresh directory** is the case that bites, because pip resolves numpy 2.x there and a 1.x host then refuses to load it (*"module compiled using NumPy 1.x cannot be run in NumPy 2.x"*) — so let the host supply its own:

```bash
mayapy -m pip install --target <dir> --no-deps pythontk
```

**Optional dependencies**, each gating a specific feature (guarded by `is_available()`-style checks — nothing else breaks without them):

- `FFmpeg` (on PATH) — audio conversion / compositing, video compression
- `OpenCV` — video frame extraction, image curation, exposure equalization, a few `ImgUtils` ops
- `rembg` — background mask generation (`MaskGenerator`)
- `PyMeshLab` — file-level mesh measure/repair/remesh/decimate/bake (`MeshOps`; declared as the `pythontk[mesh]` extra)
- `xatlas` — UV island packing (`UvPack`)
- `toktx` (KTX-Software binary) — KTX2 / Basis Universal encoding (`Ktx2Encoder`)
- `paramiko` — SSH client · `keyring` — credential storage (falls back to Windows Credential Manager, then environment variables)

## Packages

Everything is exposed at the package root via the lazy-loading resolver — bare or class-qualified:

```python
import pythontk as ptk

ptk.filter_list(...)                # bare form — wildcard-exposed
ptk.ImgUtils.pack_channels(...)     # class-qualified — explicit, collision-proof
```

| Package | What it covers |
|---|---|
| `audio_utils` | FFmpeg-backed conversion, composite WAV building, silence trimming, waveform envelopes |
| [`core_utils`](../pythontk/core_utils/README.md) | The infrastructure layer: mixins (`LoggingMixin`, `HelpMixin`, `SingletonMixin`), `listify`, package bootstrap, hot-reload, app orchestration (`AppLauncher`, `HandoffBridge`), cooperative cancellation (`CancelScope`), task pipeline, process output streaming, config/template stores, QC gates, `ExecutionMonitor`, hierarchy diffing, color primitives — plus the shared domain engines (`shots`, `instancing`, `textures`) |
| [`file_utils`](../pythontk/file_utils/README.md) | Filtered directory traversal, atomic writes, policy-scoped temp artifacts (`TempArtifacts`), cloud-placeholder detection, FBX→GLB conversion + glTF repair (`MeshConvert`), zero-dependency USD/USDZ authoring, UV unwrapping, project workspaces, embedded metadata |
| `geo_utils` | Pure geometry — `Polyline` (order/resample/smooth/simplify, arc-length sampling), `PointCloud` (PCA, clustering), `RailSurface` (line-pair framing), `PlateEmitter`, UV island packing (`UvPack`) |
| `img_utils` | Pillow-backed image ops, channel packing, atlas layout/assembly, KTX2 encoding (`Ktx2Encoder`), exposure equalization, image curation, mask generation |
| `iter_utils` | Flatten, dedupe, wildcard filtering of lists/dicts, integer-sequence collapse |
| `math_utils` | Vectors, clustering, remap/lerp/clamp, easing curves (`ProgressionCurves`), band-limited noise, morph-weight math (`Weights`), safe expression evaluation |
| [`net_utils`](../pythontk/net_utils/README.md) | SSH client, both ends of the plugin-hosted JSON-RPC protocol + DCC plugin installer, credentials, port/RDP helpers, URL file reads (`RemoteFile`, Google Sheets share links), [live WebXR preview server](webxr_preview.md) |
| `str_utils` | Sanitizing, batch rename, affix handling, `FuzzyMatcher`, hotkey-token parsing |
| `vid_utils` | Frame rate probing, compression, sharpest-frame extraction |

The three linked packages carry their own READMEs — a table row can't hold their surface. Full public surface (every class, method, signature — auto-generated): [`API_REGISTRY.md`](../API_REGISTRY.md); compact index: [`API_INDEX.md`](../API_INDEX.md).

---

## Tour

A curated subset — one example per idea, not per function.

### LoggingMixin

Structured logging for any class — custom levels, spam prevention, file tee, ring-buffer dump:

```python
import pythontk as ptk

class MyProcessor(ptk.LoggingMixin):
    def process(self):
        self.logger.info("Starting process")
        self.logger.success("Task completed")      # custom level
        self.logger.error_once("Connection failed") # logs once per 5 min, not per retry
        self.logger.log_box("Summary", ["Files: 10", "Errors: 0"])

MyProcessor.logger.setLevel("DEBUG")
MyProcessor.set_log_file("process.log")             # continuous tee
MyProcessor.enable_log_buffer(2000)                 # O(1) ring buffer, dump on demand
```

### @listify

Make any function accept a single item or a list, with optional multi-threading:

```python
@ptk.CoreUtils.listify(threading=True)
def process_texture(filepath):
    return expensive_operation(filepath)

process_texture("texture.png")                  # single result
process_texture(["a.png", "b.png", "c.png"])    # list, parallelized
```

### One filtering language

The same include/exclude wildcard language runs through the whole library — lists, dicts, directory traversal, image sets:

```python
ptk.filter_list(
    ["mesh_main", "mesh_backup", "mesh_LOD0", "cube_old"],
    inc=["mesh_*", "cube_*"],
    exc=["*_backup", "*_old"],
)
# ['mesh_main', 'mesh_LOD0']

files = ptk.get_dir_contents(
    "/path/to/project",
    content="filepath",              # file | filename | filepath | dir | dirpath
    recursive=True,
    inc_files=["*.py", "*.pyw"],
    exc_files=["*test*", "*_backup*"],
    exc_dirs=["__pycache__", ".git", "venv"],
)
```

### Texture maps — pack, convert, identify

```python
# Pack grayscale maps into RGBA channels for game engines
ptk.ImgUtils.pack_channels(
    channel_files={"R": "ao.png", "G": "roughness.png", "B": "metallic.png"},
    output_path="packed_ORM.png",
)

# Spec/Gloss → Metal/Rough PBR conversion
base_color, metallic, roughness = ptk.MapFactory.convert_spec_gloss_to_pbr(
    specular_map="specular.png", glossiness_map="gloss.png", diffuse_map="diffuse.png",
)

# Bump/height → normal map
ptk.MapFactory.convert_bump_to_normal("height.png", output_format="opengl", intensity=1.5)

# Identify map types from filenames (100+ naming conventions)
ptk.MapFactory.resolve_map_type("character_Normal_DirectX.png")  # "Normal_DirectX"
ptk.MapFactory.resolve_map_type("material_BC.tga")               # "Base_Color"
```

All three surfaces are runtime-extensible — new map types, workflow handlers, and conversions register without touching the engine ([worked example](../examples/texture_factory_extensibility_example.py)). The Qt panels that drive these engines interactively — **Map Converter**, **Map Packer**, **Map Compositor** — ship in the [extapps](https://github.com/m3trik/extapps) repo; pythontk itself stays UI-agnostic.

### Capture ingest — sharpest-frame extraction & image curation

Video-to-photogrammetry primitives. Fixed-step frame extraction wastes frames when the camera is still and starves overlap when it moves — sharpest-of-window picks the best frame from every part of the timeline instead. Then perceptual-hash curation collapses near-duplicates:

```python
frames = ptk.FrameExtractor().extract_frames_sharpest(
    "capture.mp4", "frames/", window_sec=1.0,   # sharpest frame per second of footage
)

curated_dirs = ptk.ImageCurator().curate(
    ["frames/"], "curated/",
    hash_threshold=5,                 # Hamming distance on dHash — near-dupes cluster
    sharpness_floor_percentile=10,    # drop the blurriest tenth of the survivors
)
```

`ExposureEqualizer` (cross-set exposure / white-balance matching) and `MaskGenerator` (rembg-backed background masks) round out the ingest cluster.

### Batch rename & fuzzy matching

```python
ptk.find_str_and_format(["mesh_old", "cube_old"], to="*_new", fltr="*_old")
# ['mesh_new', 'cube_new']

matches, matched_missing, matched_extra = ptk.FuzzyMatcher.find_trailing_digit_matches(
    missing_paths=["group1|mesh_01", "group1|mesh_02"],
    extra_paths=["group1|mesh_03", "group1|mesh_05"],
)
```

### Geometry & math

```python
from pythontk import Polyline, ProgressionCurves

ordered = Polyline.order_points(scattered_points, closed_path=True)
smoothed = Polyline.smooth(ordered, window_size=3)

factor = ProgressionCurves.ease_in_out(0.5)      # also: bounce, elastic, weighted, ...
ptk.remap(50, old_range=(0, 100), new_range=(0, 1))   # 0.5

ptk.collapse_integer_sequence([1, 2, 3, 5, 7, 8, 9, 15])   # "1-3, 5, 7-9, 15"
```

### Long-running task escape hatch

```python
@ptk.ExecutionMonitor.execution_monitor(threshold=30, message="Processing")
def batch_process():
    ...  # shows an abort dialog if it runs past 30s
```

### Plugin discovery

```python
plugins = ptk.get_classes_from_path(
    "plugins/", returned_type=["classobj", "filepath"], inc=["*Plugin"], exc=["*Base"],
)
```

AST enumerates the classes, then each module is imported to resolve the class objects (via the canonical package import where one exists, so identities match a normal import) — a module that fails to import is skipped, not fatal.

---

## Infrastructure the ecosystem is built on

Beyond the data-type utilities, `core_utils` supplies the machinery the layers above are built on — the full module map is [`core_utils/README.md`](../pythontk/core_utils/README.md). Highlights:

- **`bootstrap_package`** (`module_resolver`) — the lazy-loading package root. Every ecosystem package (`uitk`, `mayatk`, `blendertk`, …) exposes its public surface through it.
- **`PresetStore` / `TemplateSet` / `SchemaSpec` / `UserConfig`** — Qt-free named-preset and schema-validated-template stores with built-in + user tiers; uitk's `PresetManager` is a GUI over them.
- **`AppLauncher` / `AppInstaller` / `HandoffBridge`** — find, launch, and hand work to external applications; the base of the ecosystem's Maya/Blender/Marmoset/Substance bridges. Three hand-off shapes — `send()`, `save_as()`, `round_trip()` — come off one export pipeline with a per-mode `Deliverer` strategy.
- **`CancelScope` / `ExecutionMonitor`** — one cooperative cancellation object shared by every cancel affordance (push from any thread, pull at the operation's own checkpoints), plus threshold-escalated dialogs and watchdogs for long-running operations.
- **`RpcClient` + `RpcPlugin`** — both ends of the plugin-hosted JSON-RPC protocol, shipped together so the wire format cannot drift; the in-app half is stdlib-only so installed plugin payloads can carry a verbatim copy ([`net_utils/README.md`](../pythontk/net_utils/README.md)).
- **`QcLog` / `QcGate`** — structured run logs and threshold-based acceptance gates for batch pipelines.
- **Hierarchy toolkit** — delimited-path indexing, exact / tail-path / fuzzy matching, moved-item detection, and JSON-serializable diffs.
- **`HelpMixin`** — `.help()`, `.source()`, `.signature()` introspection on any class that mixes it in. Reachable from a shell (or an agent) without a REPL snippet via `python -m pythontk <dotted.path> [member] [--json|--source|--where|--signature|--brief|--members]`; it reads the *live* object, so it answers what the static [`API_REGISTRY.md`](../API_REGISTRY.md) cannot. `python -m pythontk --index` lists the whole resolved surface — both the `__all__` tier and the bare wildcard aliases — with each row a valid target to feed back in; it's the live twin of [`API_INDEX.md`](../API_INDEX.md), available wherever the wheel is installed.
- **`DocAudit`** — the documentation rot gate: extracts fenced code blocks from markdown and validates every attribute chain and keyword argument against the live package. This README's own examples are gated by it (`test/test_doc_audit.py`), which also pins the literal outputs shown above — an API rename or a stale doc claim fails the suite, not a user's session.

## Guides

- **[Live WebXR preview](webxr_preview.md)** — the shared DCC → glTF → headset pipeline (`PreviewServer` / `PreviewDeliverer` / `PreviewBridge` + `MeshConvert` + the bundled three.js viewer): how a baked lightmap is carried through a format that has no lightmap slot, what the scene sidecar repairs and how to read it back out of the deliverable, and the measured size/memory budget.

## Links

- **Full API:** [`API_REGISTRY.md`](../API_REGISTRY.md) · [`API_CHANGES.md`](../API_CHANGES.md)
- **Changelog:** [`CHANGELOG.md`](../CHANGELOG.md)
- **Contributor / AI-agent guide:** [`CLAUDE.md`](../CLAUDE.md)
- **PyPI:** https://pypi.org/project/pythontk/
- **Issues:** https://github.com/m3trik/pythontk/issues

## License

MIT — see [LICENSE](../LICENSE).
