[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/Version-0.7.34-blue.svg)](https://pypi.org/project/pythontk/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-423%20passed-brightgreen.svg)](test/)


# PYTHONTK

---
<!-- short_description_start -->
*A Python utility library for game development, DCC pipelines, and technical art workflows. Features texture processing, PBR conversion, fuzzy matching, progression curves, and batch processing utilities.*
<!-- short_description_end -->

## Installation

```bash
pip install pythontk
```

**Optional Dependencies:**
- `Pillow` - Required for image/texture operations
- `numpy` - Required for math and image operations
- `FFmpeg` - Required for video utilities

## Key Features

### Texture Channel Packing

Pack grayscale maps into RGBA channels for game engine workflows (Unity, Unreal):

```python
import pythontk as ptk

# Create an ORM map (Occlusion, Roughness, Metallic)
ptk.ImgUtils.pack_channels(
    channel_files={
        "R": "occlusion.png",
        "G": "roughness.png",
        "B": "metallic.png"
    },
    output_path="packed_ORM.png"
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
    output_format="opengl",  # or "directx"
    intensity=1.5,
    edge_wrap=True  # for tileable textures
)
```

### Automatic Texture Type Detection

Identify texture types from filenames (100+ naming conventions):

```python
ptk.ImgUtils.resolve_map_type("character_arm_Normal_DirectX.png")
# Returns: "Normal_DirectX"

ptk.ImgUtils.resolve_map_type("material_BC.tga")
# Returns: "Base_Color"
```

### @listify Decorator

Make any function automatically handle both single items and lists, with optional threading:

```python
@ptk.CoreUtils.listify(threading=True)
def process_texture(filepath):
    return expensive_operation(filepath)

# Works with single item OR list - parallelized automatically:
process_texture("texture.png")                    # Single result
process_texture(["a.png", "b.png", "c.png"])      # List of results
```

### Progression Curves

11 easing functions for animation, procedural generation, and non-linear distributions:

```python
from pythontk import ProgressionCurves

# Available: linear, exponential, logarithmic, ease_in, ease_out,
# ease_in_out, ease_in_out_weighted, slow_in_out, bell_curve, s_curve, smooth_step

for i in range(steps):
    factor = ProgressionCurves.calculate_progression_factor(
        i, steps, calculation_mode="ease_in_out"
    )
    # Smooth acceleration/deceleration curve
```

### Fuzzy Matching for DCC Pipelines

Match objects with different numbering (essential for Maya, 3ds Max workflows):

```python
from pythontk import FuzzyMatcher

# Find matching objects when numbering differs
matches = FuzzyMatcher.find_trailing_digit_matches(
    missing_paths=["group1|mesh_01", "group1|mesh_02"],
    extra_paths=["group1|mesh_03", "group1|mesh_05"]
)
# Matches mesh_01→mesh_03, mesh_02→mesh_05 based on hierarchy
```

### Shell-Style Pattern Filtering

Filter lists using Unix wildcards with include/exclude patterns:

```python
ptk.filter_list(
    ["mesh_main", "mesh_backup", "cube_LOD0", "cube_old"],
    inc=["mesh_*", "cube_*"],    # Include patterns
    exc=["*_backup", "*_old"],   # Exclude patterns
    ignore_case=True
)
# Returns: ['mesh_main', 'cube_LOD0']
```

### Integer Sequence Compression

Collapse integer lists into readable range strings (frames, vertex IDs):

```python
ptk.collapse_integer_sequence([1, 2, 3, 5, 7, 8, 9, 15])
# Returns: "1-3, 5, 7-9, 15"

ptk.collapse_integer_sequence([1, 2, 3, 5, 7, 8, 9, 15], limit=3)
# Returns: "1-3, 5, 7-9, ..."
```

### Execution Monitor

Monitor long-running functions with native OS dialogs:

```python
from pythontk import ExecutionMonitor

@ExecutionMonitor.execution_monitor(
    threshold=30,  # Show dialog after 30 seconds
    message="Processing textures",
    allow_escape_cancel=True
)
def batch_process():
    # Long operation - user can abort via dialog
    ...
```

### Lazy Module Loading

Speed up package imports with deferred loading:

```python
# In your package's __init__.py:
from pythontk.core_utils.module_resolver import bootstrap_package

bootstrap_package(globals(), lazy_import=True, include={
    "heavy_module": "*",
})
# Modules only import when actually accessed
```

### Plugin Discovery (AST-based)

Discover classes without executing code (safe for plugin systems):

```python
plugins = ptk.get_classes_from_path(
    "plugins/",
    returned_type=["classobj", "filepath"],
    inc=["*Plugin"],  # Only classes ending with "Plugin"
)
# Returns: [(PluginClass, "/path/to/plugin.py"), ...]
```

## Additional Utilities

### Math
- **`remap`**: Remap values between ranges (supports nested structures)
- **`lerp`**: Linear interpolation
- **`clamp`**: Constrain values to range
- **`normalize`**: Normalize 2D/3D vectors
- **`get_vector_from_two_points`**: Direction vector between points
- **`order_points_by_distance`**: Sort 3D points into continuous path
- **`smooth_points`**: Moving average smoothing for point sequences

### String
- **`sanitize`**: Clean strings for filenames with custom rules
- **`find_str_and_format`**: Batch rename with wildcards
- **`get_text_between_delimiters`**: Extract text between markers
- **`truncate`**: Shorten strings with configurable ellipsis position

### Iteration
- **`flatten`**: Flatten arbitrarily nested lists
- **`filter_dict`**: Filter dictionaries by key/value patterns
- **`remove_duplicates`**: Dedupe while preserving order
- **`nested_depth`**: Get maximum nesting level

### Image
- **`are_identical`**: Compare images for equality
- **`linear_to_srgb` / `srgb_to_linear`**: Color space conversion
- **`resize_image`**: Resize with various resampling modes

## Module Reference

| Module | Class | Purpose |
|--------|-------|---------|
| `core_utils` | `CoreUtils` | Decorators (`@listify`, `@cached_property`), attribute helpers |
| `str_utils` | `StrUtils`, `FuzzyMatcher` | String manipulation, pattern matching |
| `file_utils` | `FileUtils` | File operations, class discovery |
| `iter_utils` | `IterUtils` | List/dict filtering, flattening, grouping |
| `math_utils` | `MathUtils`, `ProgressionCurves` | Numeric ops, easing curves, vectors |
| `img_utils` | `ImgUtils` | Texture processing, PBR conversion |
| `vid_utils` | `VidUtils` | Video frame extraction, compression |

## License

MIT License
