[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/Version-0.7.59-blue.svg)](https://pypi.org/project/pythontk/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-895%20passed-brightgreen.svg)](test/)

# pythontk

<!-- short_description_start -->
*A Python utility library for game development, DCC pipelines, and technical art workflows. Features texture processing, PBR conversion, batch processing, structured logging, and utilities designed for Maya/3ds Max pipelines.*
<!-- short_description_end -->

## Installation

```bash
pip install pythontk
```

**Optional Dependencies:**
- `Pillow` – Required for image/texture operations
- `numpy` – Required for math and image operations
- `FFmpeg` – Required for video utilities

---

## Core Features

### LoggingMixin

Add structured logging to any class with custom log levels, spam prevention, and formatted output:

```python
import pythontk as ptk

class MyProcessor(ptk.LoggingMixin):
    def process(self):
        self.logger.info("Starting process")
        self.logger.success("Task completed")      # Custom level
        self.logger.result("Output: 42")           # Custom level
        self.logger.notice("Check results")        # Custom level
        
        # Prevent log spam – only logs once per 5 minutes
        self.logger.error_once("Connection failed - retrying")
        
        # Formatted box output
        self.logger.log_box("Summary", ["Files: 10", "Errors: 0"])
        # ╔══════════════════╗
        # ║     Summary      ║
        # ╟──────────────────╢
        # ║ Files: 10        ║
        # ║ Errors: 0        ║
        # ╚══════════════════╝

# Configure
MyProcessor.logger.setLevel("DEBUG")
MyProcessor.logger.add_file_handler("process.log")
MyProcessor.logger.set_log_prefix("[MyApp] ")
MyProcessor.logger.log_timestamp = "%H:%M:%S"
```

### @listify Decorator

Make any function handle both single items and lists, with optional multi-threading:

```python
@ptk.CoreUtils.listify(threading=True)
def process_texture(filepath):
    return expensive_operation(filepath)

# Automatically handles single or multiple inputs
process_texture("texture.png")                    # Returns single result
process_texture(["a.png", "b.png", "c.png"])      # Returns list (parallelized)
```

### Directory Traversal with Filtering

```python
files = ptk.get_dir_contents(
    "/path/to/project",
    content="filepath",                           # Options: file, filename, filepath, dir, dirpath
    recursive=True,
    inc_files=["*.py", "*.pyw"],
    exc_files=["*test*", "*_backup*"],
    exc_dirs=["__pycache__", ".git", "venv"]
)

# Group results by type
contents = ptk.get_dir_contents(
    "/textures",
    content=["filepath", "dirpath"],
    group_by_type=True
)
# Returns: {'filepath': [...], 'dirpath': [...]}
```

### Pattern Filtering

Filter lists or dictionaries using shell-style wildcards:

```python
ptk.filter_list(
    ["mesh_main", "mesh_backup", "mesh_LOD0", "cube_old"],
    inc=["mesh_*", "cube_*"],
    exc=["*_backup", "*_old"]
)
# Returns: ['mesh_main', 'mesh_LOD0']

ptk.filter_dict(
    {"mesh_body": obj1, "mesh_head": obj2, "light_key": obj3},
    inc=["mesh_*"],
    keys=True
)
# Returns: {'mesh_body': obj1, 'mesh_head': obj2}
```

---

## Texture & Image Processing

### Channel Packing

Pack grayscale maps into RGBA channels for game engines:

```python
ptk.ImgUtils.pack_channels(
    channel_files={"R": "ao.png", "G": "roughness.png", "B": "metallic.png"},
    output_path="packed_ORM.png"
)

ptk.ImgUtils.pack_channels(
    channel_files={"R": "metallic.png", "G": "roughness.png", "B": "ao.png", "A": "height.png"},
    output_path="packed_MRAH.tga",
    output_format="TGA"
)
```

### PBR Conversion

Convert Specular/Glossiness to Metal/Roughness:

```python
base_color, metallic, roughness = ptk.ImgUtils.convert_spec_gloss_to_pbr(
    specular_map="specular.png",
    glossiness_map="gloss.png",
    diffuse_map="diffuse.png"
)
```

### Normal Map Generation

```python
ptk.ImgUtils.convert_bump_to_normal(
    "height.png",
    output_format="opengl",  # or "directx"
    intensity=1.5,
    edge_wrap=True
)
```

### Texture Type Detection

Identify texture types from filenames (100+ naming conventions):

```python
ptk.TextureMapFactory.resolve_map_type("character_Normal_DirectX.png")  # "Normal_DirectX"
ptk.TextureMapFactory.resolve_map_type("material_BC.tga")               # "Base_Color"
ptk.TextureMapFactory.resolve_map_type("metal_AO.jpg")                  # "Ambient_Occlusion"
```

---

## DCC Pipeline Utilities

### Fuzzy Matching

Match objects when numbering differs:

```python
from pythontk import FuzzyMatcher

matches = FuzzyMatcher.find_trailing_digit_matches(
    missing_paths=["group1|mesh_01", "group1|mesh_02"],
    extra_paths=["group1|mesh_03", "group1|mesh_05"]
)
# Matches mesh_01→mesh_03, mesh_02→mesh_05
```

### Batch Rename

```python
ptk.find_str_and_format(
    ["mesh_old", "cube_old", "sphere_old"],
    to="*_new",
    fltr="*_old"
)
# Returns: ['mesh_new', 'cube_new', 'sphere_new']

ptk.find_str_and_format(
    ["body", "head", "hands"],
    to="character_**",  # ** = append
    fltr="*"
)
# Returns: ['character_body', 'character_head', 'character_hands']
```

### Integer Sequence Compression

```python
ptk.collapse_integer_sequence([1, 2, 3, 5, 7, 8, 9, 15])
# Returns: "1-3, 5, 7-9, 15"

ptk.collapse_integer_sequence([1, 2, 3, 5, 7, 8, 9, 15], limit=3)
# Returns: "1-3, 5, 7-9, ..."
```

### Point Path Operations

```python
ordered = ptk.arrange_points_as_path(scattered_points, closed_path=True)
smoothed = ptk.smooth_points(ordered, window_size=3)
```

---

## Animation & Math

### Easing Curves

```python
from pythontk import ProgressionCurves

# Available: linear, exponential, logarithmic, ease_in, ease_out,
# ease_in_out, bounce, elastic, sine, smooth_step, weighted

factor = ProgressionCurves.ease_in_out(0.5)
factor = ProgressionCurves.bounce(0.5)
factor = ProgressionCurves.elastic(0.5)
factor = ProgressionCurves.weighted(0.5, weight_curve=2.0, weight_bias=0.3)
factor = ProgressionCurves.calculate_progression_factor(5, 10, "ease_in_out")
```

### Range Remapping

```python
ptk.remap(50, old_range=(0, 100), new_range=(0, 1))  # 0.5
ptk.remap([[0.5, 0.5], [0.0, 1.0]], old_range=(0, 1), new_range=(-1, 1))
ptk.remap(150, old_range=(0, 100), new_range=(0, 1), clamp=True)  # 1.0
```

---

## Advanced Features

### Execution Monitor

```python
from pythontk import ExecutionMonitor

@ExecutionMonitor.execution_monitor(threshold=30, message="Processing")
def batch_process():
    # Shows dialog after 30s, allowing user to abort
    ...
```

### Lazy Module Loading

```python
from pythontk.core_utils.module_resolver import bootstrap_package

bootstrap_package(globals(), lazy_import=True, include={
    "heavy_module": "*",
    "optional_feature": ["SpecificClass"],
})
```

### Plugin Discovery

```python
plugins = ptk.get_classes_from_path(
    "plugins/",
    returned_type=["classobj", "filepath"],
    inc=["*Plugin"],
    exc=["*Base"]
)
# Uses AST parsing – never executes plugin code
```

### Color Space Conversion

```python
linear_data = ptk.ImgUtils.srgb_to_linear(srgb_data)
display_data = ptk.ImgUtils.linear_to_srgb(linear_data)
```

---

## Reference

| Function | Description |
|----------|-------------|
| `filter_list(lst, inc, exc)` | Filter with wildcards |
| `filter_dict(d, inc, exc, keys)` | Filter dict keys/values |
| `get_dir_contents(path, ...)` | Directory traversal |
| `flatten(nested)` | Flatten nested lists |
| `make_iterable(obj)` | Ensure iterable |
| `remove_duplicates(lst)` | Dedupe preserving order |
| `sanitize(text)` | Clean for filenames |
| `remap(val, old, new)` | Remap ranges |
| `clamp(val, min, max)` | Constrain to range |
| `lerp(a, b, t)` | Linear interpolation |

| Module | Classes |
|--------|---------|
| `core_utils` | `CoreUtils`, `LoggingMixin`, `HelpMixin` |
| `str_utils` | `StrUtils`, `FuzzyMatcher` |
| `file_utils` | `FileUtils` |
| `iter_utils` | `IterUtils` |
| `math_utils` | `MathUtils`, `ProgressionCurves` |
| `img_utils` | `ImgUtils` |
| `vid_utils` | `VidUtils` |

## License

MIT License

<!-- Test update: 2025-12-02 20:53 -->

<!-- Test update: 2025-12-02 21:05:10 -->
