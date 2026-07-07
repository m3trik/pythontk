[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/pythontk.svg)](https://pypi.org/project/pythontk/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-1654%20passed-brightgreen.svg)](../test/)

# pythontk

<!-- short_description_start -->
*The foundation layer of a DCC-tooling ecosystem — composable Python primitives for files, strings, iteration, math, geometry, images, video, audio, and networking, plus the class mixins and package infrastructure the layers above are built on.*
<!-- short_description_end -->

Pure Python: no Qt, no DCC imports, two small hard dependencies (`numpy`, `Pillow`). Everything heavier — FFmpeg, OpenCV, rembg, PyMeshLab — is optional and feature-gated, so the core runs identically in `mayapy`, Blender's Python, a CI runner, or a bare venv.

## Why

pythontk is the bottom of an ecosystem chain (`pythontk → uitk → mayatk / blendertk → tentacle`) built for game-art and DCC pipeline tooling. Everything above imports it; it imports nothing above it — no `maya`, no `bpy`, no Qt. That constraint is the point: a helper written here works everywhere the ecosystem runs, and the layers above stay thin because the environment-independent 80% of every tool lives at this layer.

Two rules shape the library:

- **Primitives are placed by data type, not by domain.** Sharpest-frame extraction lives in `vid_utils` and perceptual-hash image curation in `img_utils` — not in a "photogrammetry" package — so each piece stays independently reusable. Domain pipelines (PBR texture conversion, photogrammetry ingest, timeline audio events) are *compositions* of these primitives, assembled downstream.
- **Shared code moves down.** When two downstream tools need the same helper, it is extracted here and becomes the single source of truth — Maya and Blender panels share one calculator engine, one material-report formatter, one point-clustering routine, instead of drifting copies.

The result is less a grab-bag of utilities than the standard library of its ecosystem: [uitk](https://github.com/m3trik/uitk) builds its preset and logging UIs on pythontk stores and mixins, [mayatk](https://github.com/m3trik/mayatk) launches Maya through `AppLauncher` and hands scenes to other apps through `HandoffBridge`, and every package in the chain boots its lazy-loading root through `module_resolver`.

## Install

```bash
pip install pythontk
```

**Optional dependencies**, each gating a specific feature (guarded by `is_available()` checks — nothing else breaks without them):

- `FFmpeg` (on PATH) — audio conversion / compositing, video compression
- `OpenCV` — video frame extraction, image curation, exposure equalization, a few `ImgUtils` ops
- `rembg` — background mask generation (`MaskGenerator`)
- `PyMeshLab` — mesh repair (`MeshCleaner`)

## Packages

Everything is exposed at the package root via the lazy-loading resolver — bare or class-qualified:

```python
import pythontk as ptk

ptk.filter_list(...)                # bare form — wildcard-exposed
ptk.ImgUtils.pack_channels(...)     # class-qualified — explicit, collision-proof
```

| Package | What it covers |
|---|---|
| `audio_utils` | FFmpeg-backed conversion, composite WAV building, waveform envelopes |
| `color_utils` | `Color` / `Palette` primitives — hex/rgb/luminance, blending, themed palettes |
| `core_utils` | The infrastructure layer: mixins (`LoggingMixin`, `HelpMixin`, `SingletonMixin`), `listify`, package bootstrap (`module_resolver`), app orchestration (`AppLauncher`, `HandoffBridge`), config/template stores (`PresetStore`, `TemplateSet`, `SchemaSpec`, `UserConfig`), QC gates, `ExecutionMonitor`, hierarchy diffing |
| `file_utils` | Filtered directory traversal, atomic writes, JSON helpers, cloud-placeholder detection, mesh format conversion, embedded metadata |
| `geo_utils` | Pure geometry — `Polyline` (order/resample/smooth/simplify), `PointCloud` (PCA, clustering), procedural drape |
| `img_utils` | Pillow-backed image ops, channel packing, `MapFactory` (PBR map conversion/packing), map optimizer & compositor, exposure equalization, image curation, mask generation |
| `iter_utils` | Flatten, dedupe, wildcard filtering of lists/dicts, integer-sequence collapse |
| `math_utils` | Vectors, clustering, remap/lerp/clamp, easing curves (`ProgressionCurves`), band-limited noise, safe expression evaluation |
| `net_utils` | SSH client, generic JSON-RPC client + DCC plugin installer, credentials, port/RDP helpers |
| `str_utils` | Sanitizing, batch rename, affix handling, `FuzzyMatcher`, hotkey-token parsing |
| `vid_utils` | Frame rate probing, compression, sharpest-frame extraction |

Full public surface (every class, method, signature — auto-generated): [`API_REGISTRY.md`](../API_REGISTRY.md); compact index: [`API_INDEX.md`](../API_INDEX.md).

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

### Filtered directory traversal

```python
files = ptk.get_dir_contents(
    "/path/to/project",
    content="filepath",              # file | filename | filepath | dir | dirpath
    recursive=True,
    inc_files=["*.py", "*.pyw"],
    exc_files=["*test*", "*_backup*"],
    exc_dirs=["__pycache__", ".git", "venv"],
)
```

### Wildcard filtering

The same include/exclude pattern language runs through the whole library:

```python
ptk.filter_list(
    ["mesh_main", "mesh_backup", "mesh_LOD0", "cube_old"],
    inc=["mesh_*", "cube_*"],
    exc=["*_backup", "*_old"],
)
# ['mesh_main', 'mesh_LOD0']
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

The Qt panels that drive these engines interactively — **Map Converter**, **Map Packer**, **Map Compositor** — ship in the [extapps](https://github.com/m3trik/extapps) repo; pythontk itself stays UI-agnostic.

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

### Plugin discovery (AST-based — never executes plugin code)

```python
plugins = ptk.get_classes_from_path(
    "plugins/", returned_type=["classobj", "filepath"], inc=["*Plugin"], exc=["*Base"],
)
```

---

## Infrastructure the ecosystem is built on

Beyond the data-type utilities, `core_utils` supplies the machinery every package in the chain shares:

- **`bootstrap_package`** (`module_resolver`) — the lazy-loading package root. Every ecosystem package (`uitk`, `mayatk`, `blendertk`, …) exposes its public surface through it.
- **`PresetStore` / `TemplateSet` / `SchemaSpec` / `UserConfig`** — Qt-free named-preset and schema-validated-template stores with built-in + user tiers; uitk's `PresetManager` is a GUI over them.
- **`AppLauncher` / `AppInstaller` / `HandoffBridge`** — find, launch, and hand work to external applications; the base of the ecosystem's Maya/Blender/Marmoset/Substance bridges and of mayatk's `MayaConnection`.
- **`QcLog` / `QcGate`** — structured run logs and threshold-based acceptance gates for batch pipelines.
- **`HelpMixin`** — `.help()`, `.source()`, `.signature()` introspection on any class that mixes it in.

## Links

- **Full API:** [`API_REGISTRY.md`](../API_REGISTRY.md) · [`API_CHANGES.md`](../API_CHANGES.md)
- **Changelog:** [`CHANGELOG.md`](../CHANGELOG.md)
- **Contributor / AI-agent guide:** [`CLAUDE.md`](../CLAUDE.md)
- **PyPI:** https://pypi.org/project/pythontk/
- **Issues:** https://github.com/m3trik/pythontk/issues

## License

MIT — see [LICENSE](../LICENSE).
