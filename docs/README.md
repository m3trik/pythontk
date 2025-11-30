[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/Version-0.7.34-blue.svg)](https://pypi.org/project/pythontk/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-423%20passed-brightgreen.svg)](test/)


# PYTHONTK

---
<!-- short_description_start -->
*A Python utility library for game development, DCC pipelines, and technical art workflows. Features texture processing, PBR conversion, batch processing, structured logging, and utilities designed for Maya/3ds Max pipelines.*
<!-- short_description_end -->

## Installation

```bash
pip install pythontk
```

**Optional Dependencies:**
- `Pillow` - Required for image/texture operations
- `numpy` - Required for math and image operations
- `FFmpeg` - Required for video utilities

---

## Core Features

### LoggingMixin - Structured Logging for Classes

Add powerful logging to any class with custom log levels, spam prevention, and formatted output:

```python
import pythontk as ptk

class MyProcessor(ptk.LoggingMixin):
    def process(self):
        self.logger.info("Starting process")
        self.logger.success("Task completed")      # Custom level
        self.logger.result("Output: 42")           # Custom level
        self.logger.notice("Check results")        # Custom level
        
        # Prevent log spam - only logs once per 5 minutes
        self.logger.error_once("Connection failed - retrying")
        
        # Formatted output
        self.logger.log_box("Summary", ["Files: 10", "Errors: 0"])
        # ╔══════════════════╗
        # ║     Summary      ║
        # ╟──────────────────╢
        # ║ Files: 10        ║
        # ║ Errors: 0        ║
        # ╚══════════════════╝

# Configure logging
MyProcessor.logger.setLevel("DEBUG")
MyProcessor.logger.add_file_handler("process.log")
MyProcessor.logger.set_log_prefix("[MyApp] ")
MyProcessor.logger.log_timestamp = "%H:%M:%S"
```

### @listify Decorator - Automatic Batch Processing

Make any function handle both single items and lists, with optional multi-threading:

```python
@ptk.CoreUtils.listify(threading=True)
def process_texture(filepath):
    return expensive_operation(filepath)

# Automatically handles single or multiple inputs:
process_texture("texture.png")                    # Returns: single result
process_texture(["a.png", "b.png", "c.png"])      # Returns: [result, result, result]
# With threading=True, list operations are parallelized
```

### Directory Contents with Advanced Filtering

Traverse directories with include/exclude patterns for files and folders:

```python
# Find all Python files, excluding tests and cache
files = ptk.get_dir_contents(
    "/path/to/project",
    content="filepath",               # Return full paths (also: 'file', 'filename', 'dir', 'dirpath')
    recursive=True,                   # Include subdirectories
    inc_files=["*.py", "*.pyw"],      # Only Python files
    exc_files=["*test*", "*_backup*"],# Exclude test and backup files
    exc_dirs=["__pycache__", ".git", "node_modules", "venv"]
)

# Get both files and directories grouped
contents = ptk.get_dir_contents(
    "/textures",
    content=["filepath", "dirpath"],
    group_by_type=True
)
# Returns: {'filepath': [...], 'dirpath': [...]}
```

### Shell-Style Pattern Filtering

Filter any list using Unix wildcards with include/exclude patterns:

```python
# Filter Maya objects
ptk.filter_list(
    ["mesh_main", "mesh_backup", "mesh_LOD0", "cube_old", "helper_ctrl"],
    inc=["mesh_*", "cube_*"],        # Include patterns
    exc=["*_backup", "*_old"],       # Exclude patterns
    ignore_case=True
)
# Returns: ['mesh_main', 'mesh_LOD0']

# Filter dictionary by keys
ptk.filter_dict(
    {"mesh_body": obj1, "mesh_head": obj2, "light_key": obj3, "helper": obj4},
    inc=["mesh_*"],
    keys=True  # Filter by keys (vs values)
)
# Returns: {'mesh_body': obj1, 'mesh_head': obj2}
```

---

## Texture & Image Processing

### Channel Packing for Game Engines

Pack grayscale maps into RGBA channels (Unity, Unreal ORM/MRAO workflows):

```python
# Create an ORM map (Occlusion, Roughness, Metallic)
ptk.ImgUtils.pack_channels(
    channel_files={
        "R": "occlusion.png",
        "G": "roughness.png",
        "B": "metallic.png"
    },
    output_path="packed_ORM.png"
)

# With alpha channel for height
ptk.ImgUtils.pack_channels(
    channel_files={
        "R": "metallic.png",
        "G": "roughness.png",
        "B": "ao.png",
        "A": "height.png"
    },
    output_path="packed_MRAH.tga",
    output_format="TGA"
)
```

### PBR Workflow Conversion

Convert legacy Specular/Glossiness textures to modern Metal/Roughness:

```python
base_color, metallic, roughness = ptk.ImgUtils.convert_spec_gloss_to_pbr(
    specular_map="old_specular.png",
    glossiness_map="old_gloss.png",
    diffuse_map="old_diffuse.png"
)
```

### Normal Map Generation

Convert height/bump maps to tangent-space normal maps:

```python
ptk.ImgUtils.convert_bump_to_normal(
    "height.png",
    output_format="opengl",     # or "directx"
    intensity=1.5,
    edge_wrap=True              # For tileable textures
)
```

### Automatic Texture Type Detection

Identify texture types from filenames (100+ naming conventions):

```python
ptk.ImgUtils.resolve_map_type("character_arm_Normal_DirectX.png")  # "Normal_DirectX"
ptk.ImgUtils.resolve_map_type("material_BC.tga")                   # "Base_Color"
ptk.ImgUtils.resolve_map_type("wood_floor_Roughness.png")          # "Roughness"
ptk.ImgUtils.resolve_map_type("metal_plate_AO.jpg")                # "Ambient_Occlusion"
```

---

## DCC Pipeline Utilities

### Fuzzy Matching for Object Hierarchies

Match objects when numbering differs (essential for Maya, 3ds Max retargeting):

```python
from pythontk import FuzzyMatcher

# Objects were renamed/renumbered between versions
matches = FuzzyMatcher.find_trailing_digit_matches(
    missing_paths=["group1|mesh_01", "group1|mesh_02"],
    extra_paths=["group1|mesh_03", "group1|mesh_05"]
)
# Matches mesh_01→mesh_03, mesh_02→mesh_05 based on hierarchy position
```

### Batch Rename with Wildcards

Search and replace with pattern matching for batch renaming:

```python
# Replace suffix
ptk.find_str_and_format(
    ["mesh_old", "cube_old", "sphere_old"],
    to="*_new",           # Replace suffix
    fltr="*_old"          # Find pattern
)
# Returns: ['mesh_new', 'cube_new', 'sphere_new']

# Add prefix
ptk.find_str_and_format(
    ["body", "head", "hands"],
    to="character_**",    # Append prefix (** = append, * = replace)
    fltr="*"
)
# Returns: ['character_body', 'character_head', 'character_hands']
```

### Integer Sequence Compression

Collapse integer lists into readable range strings (frames, vertex IDs):

```python
ptk.collapse_integer_sequence([1, 2, 3, 5, 7, 8, 9, 15])
# Returns: "1-3, 5, 7-9, 15"

# With limit for long sequences
ptk.collapse_integer_sequence([1, 2, 3, 5, 7, 8, 9, 15, 20, 21, 22], limit=3)
# Returns: "1-3, 5, 7-9, ..."
```

### 3D Point Operations

Arrange unordered points into a continuous path and smooth the result:

```python
# Sort scattered points into a path
ordered = ptk.arrange_points_as_path(scattered_points, closed_path=True)

# Smooth a point sequence (moving average)
smoothed = ptk.smooth_points(ordered, window_size=3)
```

---

## Animation & Math

### Progression Curves (Easing Functions)

Easing curves for animation, procedural generation, and non-linear distributions:

```python
from pythontk import ProgressionCurves

# Available: linear, exponential, logarithmic, ease_in, ease_out,
# ease_in_out, bounce, elastic, sine, smooth_step, weighted

t = 0.5  # Progress from 0.0 to 1.0

# Smooth acceleration/deceleration
factor = ProgressionCurves.ease_in_out(t)

# Bouncy effect
factor = ProgressionCurves.bounce(t)

# Elastic overshoot
factor = ProgressionCurves.elastic(t)

# Weighted easing with custom curve/bias
factor = ProgressionCurves.weighted(t, weight_curve=2.0, weight_bias=0.3)

# Get any curve by name
factor = ProgressionCurves.calculate_progression_factor(5, 10, "ease_in_out")
```

### Range Remapping

Remap values between ranges (supports nested structures):

```python
# Simple remap
ptk.remap(50, old_range=(0, 100), new_range=(0, 1))  # 0.5

# Remap UV coordinates
ptk.remap([[0.5, 0.5], [0.0, 1.0]], old_range=(0, 1), new_range=(-1, 1))
# Returns: [[0.0, 0.0], [-1.0, 1.0]]

# With clamping
ptk.remap(150, old_range=(0, 100), new_range=(0, 1), clamp=True)  # 1.0
```

---

## Advanced Features

### Execution Monitor

Monitor long-running functions with native OS dialogs:

```python
from pythontk import ExecutionMonitor

@ExecutionMonitor.execution_monitor(
    threshold=30,                  # Show dialog after 30 seconds
    message="Processing textures",
    allow_escape_cancel=True       # Allow ESC key to abort
)
def batch_process_textures():
    # Long operation - user gets dialog to continue/abort
    for texture in textures:
        process(texture)
```

### Lazy Module Loading

Speed up package imports with deferred loading:

```python
# In your package's __init__.py:
from pythontk.core_utils.module_resolver import bootstrap_package

bootstrap_package(globals(), lazy_import=True, include={
    "heavy_module": "*",
    "optional_feature": ["SpecificClass"],
})
# Modules only import when actually accessed
```

### Plugin Discovery (AST-based)

Discover plugin classes without executing code:

```python
plugins = ptk.get_classes_from_path(
    "plugins/",
    returned_type=["classobj", "filepath"],
    inc=["*Plugin", "*Handler"],    # Pattern matching
    exc=["*Base", "*Abstract"]
)
# Returns: [(PluginClass, "/path/to/plugin.py"), ...]
# Safe: uses AST parsing, never executes plugin code
```

### Color Space Conversion

Proper gamma-correct color space conversion:

```python
# Convert for linear lighting calculations
linear_data = ptk.ImgUtils.srgb_to_linear(srgb_image_data)

# Convert back for display
display_data = ptk.ImgUtils.linear_to_srgb(linear_data)
```

---

## Quick Reference

### Commonly Used Functions

| Function | Description |
|----------|-------------|
| `filter_list(items, inc, exc)` | Filter with wildcards |
| `filter_dict(d, inc, exc, keys)` | Filter dict keys/values |
| `get_dir_contents(path, ...)` | Directory traversal with filtering |
| `flatten(nested_list)` | Flatten arbitrarily nested lists |
| `make_iterable(obj)` | Ensure object is iterable |
| `remove_duplicates(lst)` | Dedupe preserving order |
| `sanitize(text)` | Clean strings for filenames |
| `remap(val, old_range, new_range)` | Remap value between ranges |
| `clamp(val, min, max)` | Constrain value to range |
| `lerp(a, b, t)` | Linear interpolation |
| `collapse_integer_sequence(lst)` | Compress to range string |

### Module Reference

| Module | Classes | Purpose |
|--------|---------|---------|
| `core_utils` | `CoreUtils`, `LoggingMixin`, `HelpMixin` | Decorators, logging, introspection |
| `str_utils` | `StrUtils`, `FuzzyMatcher` | String ops, pattern matching, fuzzy search |
| `file_utils` | `FileUtils` | File I/O, directory ops, class discovery |
| `iter_utils` | `IterUtils` | List/dict filtering, flattening, grouping |
| `math_utils` | `MathUtils`, `ProgressionCurves` | Numeric ops, easing, vectors |
| `img_utils` | `ImgUtils` | Texture processing, PBR, channel packing |
| `vid_utils` | `VidUtils` | Video frame extraction, compression |

## License

MIT License
