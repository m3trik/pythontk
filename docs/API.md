# API Reference

## Core Utils

### HelpMixin
Base mixin providing auto-documentation functionality.

#### Methods
- `get_help()`: Get formatted help documentation for the class

### LoggingMixin
Standardized logging functionality for classes.

#### Properties
- `logger`: Returns configured logger instance

### SingletonMixin
Implements singleton pattern for classes.

### Decorators

#### @cached_property
```python
@cached_property
def expensive_property(self):
    """Property that caches its result after first computation"""
    return expensive_computation()
```

#### @listify
```python
@listify(threading=True)
def process_item(item):
    """Function that can process single items or lists of items"""
    return processed_item
```

---

## File Utils

### FileUtils Class

#### Methods

##### `is_valid(filepath, expected_type=None)`
Check if a path is valid.

**Parameters:**
- `filepath` (str): Path to check
- `expected_type` (str, optional): 'file' or 'dir'

**Returns:** bool

##### `create_dir(filepath)`
Create directory if it doesn't exist.

**Parameters:**
- `filepath` (str): Directory path to create

##### `get_dir_contents(directory, return_type='filepath', inc=None, exc=None, recursive=False)`
Get directory contents with filtering.

**Parameters:**
- `directory` (str): Directory to scan
- `return_type` (str): 'filepath', 'filename', or 'basename'
- `inc` (list/str): Include patterns (wildcards supported)
- `exc` (list/str): Exclude patterns (wildcards supported)
- `recursive` (bool): Scan subdirectories

**Returns:** list

##### `get_file_contents(filepath, encoding='auto')`
Read file contents with encoding detection.

**Parameters:**
- `filepath` (str): File to read
- `encoding` (str): Encoding to use ('auto' for detection)

**Returns:** str

##### `write_file(filepath, content, encoding='utf-8')`
Write content to file with directory creation.

**Parameters:**
- `filepath` (str): File path to write
- `content` (str): Content to write
- `encoding` (str): Text encoding

##### `format_path(path, style='forward')`
Convert path separators for cross-platform compatibility.

**Parameters:**
- `path` (str): Path to format
- `style` (str): 'forward', 'backward', or 'native'

**Returns:** str

---

## String Utils

### StrUtils Class

#### Methods

##### `sanitize(text, replacement_char='_', char_map=None, preserve_trailing=False, preserve_case=False, allow_consecutive=False, return_original=False)`
Sanitize strings for use as filenames or identifiers.

**Parameters:**
- `text` (str/list): Text to sanitize
- `replacement_char` (str): Character for replacing invalid chars
- `char_map` (dict): Custom character replacements
- `preserve_trailing` (bool): Keep trailing characters
- `preserve_case` (bool): Maintain original case
- `allow_consecutive` (bool): Allow consecutive replacement chars
- `return_original` (bool): Return tuple with original

**Returns:** str/tuple/list

##### `get_text_between_delimiters(text, start_delimiter, end_delimiter, as_string=False)`
Extract text between delimiters.

**Parameters:**
- `text` (str): Source text
- `start_delimiter` (str): Start marker
- `end_delimiter` (str): End marker
- `as_string` (bool): Return as single string vs list

**Returns:** str/list

##### `camel_to_snake(text)`
Convert camelCase to snake_case.

**Parameters:**
- `text` (str): Text to convert

**Returns:** str

##### `snake_to_camel(text)`
Convert snake_case to camelCase.

**Parameters:**
- `text` (str): Text to convert

**Returns:** str

---

## Image Utils

### ImgUtils Class

#### Basic Operations

##### `ensure_image(input_image, mode=None)`
Convert file path or ensure PIL Image object.

**Parameters:**
- `input_image` (str/Image): File path or PIL Image
- `mode` (str): Convert to specific mode

**Returns:** PIL.Image

##### `resize_image(image, width, height)`
Resize image to specified dimensions.

**Parameters:**
- `image` (str/Image): Image to resize
- `width` (int): Target width
- `height` (int): Target height

**Returns:** PIL.Image

##### `create_image(mode, size=(4096, 4096), color=None)`
Create new blank image.

**Parameters:**
- `mode` (str): Image mode ('RGB', 'RGBA', etc.)
- `size` (tuple): Image dimensions
- `color` (tuple): Fill color

**Returns:** PIL.Image

#### Channel Operations

##### `pack_channels(channel_files, channels=None, out_mode=None, output_path=None)`
Pack multiple single-channel images into multi-channel image.

**Parameters:**
- `channel_files` (dict): {'R': 'red.jpg', 'G': 'green.jpg', ...}
- `channels` (list): Channel order ['R', 'G', 'B', 'A']
- `out_mode` (str): Output mode ('RGB', 'RGBA')
- `output_path` (str): Save path

**Returns:** str/PIL.Image

##### `pack_channel_into_alpha(image, alpha, output_path=None, invert_alpha=False)`
Add alpha channel to image.

**Parameters:**
- `image` (str/Image): Base image
- `alpha` (str/Image): Alpha channel
- `output_path` (str): Save path
- `invert_alpha` (bool): Invert alpha values

**Returns:** str

#### Texture Processing

##### `convert_spec_gloss_to_metal_rough(specular_map, glossiness_map, output_dir=None, write_files=False)`
Convert specular/glossiness to metallic/roughness workflow.

**Parameters:**
- `specular_map` (str/Image): Specular map
- `glossiness_map` (str/Image): Glossiness map
- `output_dir` (str): Output directory
- `write_files` (bool): Save to files

**Returns:** tuple(Image, Image, Image)

##### `create_metallic_from_spec(specular_map, glossiness_map=None, threshold=55, softness=0.2)`
Generate metallic map from specular.

**Parameters:**
- `specular_map` (str/Image): Specular map
- `glossiness_map` (str/Image): Optional glossiness
- `threshold` (int): Metallic threshold
- `softness` (float): Transition softness

**Returns:** PIL.Image

##### `create_roughness_from_spec(specular_map, glossiness_map=None)`
Generate roughness map from specular/glossiness.

**Parameters:**
- `specular_map` (str/Image): Specular map
- `glossiness_map` (str/Image): Optional glossiness

**Returns:** PIL.Image

##### `batch_optimize_textures(directory, max_size=4096, output_dir=None)`
Batch optimize textures in directory.

**Parameters:**
- `directory` (str): Directory containing textures
- `max_size` (int): Maximum dimension
- `output_dir` (str): Output directory

---

## Video Utils

### VidUtils Class

#### Methods

##### `resolve_ffmpeg()`
Find FFmpeg executable in system.

**Returns:** str (path to ffmpeg)

##### `extract_frames(video_path, output_folder, step=5, quality=95, prefix='frame', max_frames=None)`
Extract frames from video file.

**Parameters:**
- `video_path` (str): Video file path
- `output_folder` (str): Output directory
- `step` (int): Extract every nth frame
- `quality` (int): JPEG quality (1-100)
- `prefix` (str): Frame filename prefix
- `max_frames` (int): Maximum frames to extract

**Returns:** list (frame file paths)

##### `get_video_frame_rate(filepath)`
Get video frame rate.

**Parameters:**
- `filepath` (str): Video file path

**Returns:** float

---

## Math Utils

### MathUtils Class

#### Methods

##### `move_decimal_point(num, places)`
Move decimal point with precision.

**Parameters:**
- `num` (float): Number to modify
- `places` (int): Places to move (negative = left)

**Returns:** float

##### `get_vector_from_two_points(a, b)`
Calculate directional vector between points.

**Parameters:**
- `a` (list): Start point [x, y, z]
- `b` (list): End point [x, y, z]

**Returns:** tuple(float, float, float)

##### `clamp(value, min_val, max_val)`
Clamp value within range.

**Parameters:**
- `value` (float): Value to clamp
- `min_val` (float): Minimum value
- `max_val` (float): Maximum value

**Returns:** float

##### `lerp(start, end, t)`
Linear interpolation between values.

**Parameters:**
- `start` (float): Start value
- `end` (float): End value
- `t` (float): Interpolation factor (0-1)

**Returns:** float

---

## Iterator Utils

### IterUtils Class

#### Methods

##### `filter_list(lst, inc=None, exc=None, map_func=None, check_unmapped=False, nested_as_unit=False, basename_only=False, ignore_case=False)`
Advanced list filtering with wildcards.

**Parameters:**
- `lst` (list): List to filter
- `inc` (str/list): Include patterns
- `exc` (str/list): Exclude patterns
- `map_func` (callable): Transform function for comparison
- `check_unmapped` (bool): Check original values
- `nested_as_unit` (bool): Treat nested items as units
- `basename_only` (bool): Compare basenames only
- `ignore_case` (bool): Case-insensitive comparison

**Returns:** list

##### `filter_dict(dictionary, inc=None, exc=None, keys=False, values=False)`
Filter dictionary by keys or values.

**Parameters:**
- `dictionary` (dict): Dictionary to filter
- `inc` (str/list): Include patterns
- `exc` (str/list): Exclude patterns
- `keys` (bool): Filter by keys
- `values` (bool): Filter by values

**Returns:** dict

##### `make_iterable(x, snapshot=False)`
Convert object to iterable safely.

**Parameters:**
- `x` (any): Object to make iterable
- `snapshot` (bool): Return list snapshot

**Returns:** iterable

##### `nested_depth(lst, typ=(list, set, tuple))`
Get maximum nesting depth.

**Parameters:**
- `lst` (list): List to analyze
- `typ` (tuple): Types to consider nested

**Returns:** int

---

## Error Handling

All functions include comprehensive error handling:

```python
try:
    result = ptk.function_name(parameters)
except FileNotFoundError:
    # Handle missing files
    pass
except ValueError:
    # Handle invalid parameters
    pass
except Exception as e:
    # Handle other errors
    print(f"Error: {e}")
```

---

## Type Hints

All functions include proper type hints for better IDE support:

```python
from typing import Union, List, Optional, Dict, Tuple
from PIL import Image

def resize_image(
    image: Union[str, Image.Image], 
    width: int, 
    height: int
) -> Image.Image:
    pass
```
