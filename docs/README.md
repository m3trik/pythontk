[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/Version-0.7.34-blue.svg)](https://pypi.org/project/pythontk/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-423%20passed-brightgreen.svg)](test/)


# PYTHONTK (Python Toolkit)

---
<!-- short_description_start -->
*A modular Python utility library providing class-based utilities for string manipulation, file operations, iteration helpers, math functions, and image/video processing.*
<!-- short_description_end -->

## Features

pythontk organizes utilities into focused classes that can be used statically or inherited:

- **StrUtils**: String sanitization, case conversion, text extraction, and fuzzy matching
- **FileUtils**: Directory traversal, file I/O, path manipulation, and JSON handling
- **IterUtils**: List flattening, filtering, grouping, and iterable transformations
- **MathUtils**: Clamping, vector operations, interpolation, and numeric utilities
- **ImgUtils**: Image channel packing, resizing, format conversion (requires Pillow)
- **VidUtils**: Video frame extraction and compression (requires FFmpeg)
- **CoreUtils**: Decorators (`@listify`, `@cached_property`) and helper mixins

## Installation

```bash
pip install pythontk
```

**Development Installation:**
```bash
git clone https://github.com/m3trik/pythontk.git
cd pythontk
pip install -e .
```

**Optional Dependencies:**
- `Pillow` - Required for image operations
- `numpy` - Required for some math and image operations
- `FFmpeg` - Required for video utilities (must be in PATH)

## Usage

pythontk uses a dynamic import system that exposes all utility methods at the package level:

```python
import pythontk as ptk

# Access methods directly
ptk.sanitize("Hello World!")  # 'hello_world'

# Or via the utility class
ptk.StrUtils.sanitize("Hello World!")  # 'hello_world'
```

## Examples

### String Utilities

```python
import pythontk as ptk

# Sanitize strings for filenames (removes special chars, lowercases)
ptk.sanitize("My File Name!@#.txt")
# Returns: 'my_file_name_txt'

# Preserve case
ptk.sanitize("CamelCase Name", preserve_case=True)
# Returns: 'CamelCase_Name'

# Case conversion
ptk.set_case("hello world", case="title")  # 'Hello World'
ptk.set_case("hello world", case="upper")  # 'HELLO WORLD'
ptk.set_case("HelloWorld", case="camel")   # 'helloWorld'

# Truncate long strings (mode='end' keeps start, trims end)
ptk.truncate("This is a very long string", 20, mode="end")
# Returns: 'This is a very lo..'

# Find text between delimiters
list(ptk.get_text_between_delimiters("Hello <world> and <python>", "<", ">"))
# Returns: ['world', 'python']
```

### File Utilities

```python
import pythontk as ptk

# Get directory contents with filtering
files = ptk.get_dir_contents(
    "/path/to/project",
    content="filepath",           # Return full paths
    recursive=True,               # Include subdirectories
    inc_files=["*.py"],           # Only Python files
    exc_files=["*test*"],         # Exclude test files
    exc_dirs=["__pycache__", ".git"]
)

# Read file contents
content = ptk.get_file_contents("/path/to/file.txt")

# Read specific lines
lines = ptk.get_file_contents("/path/to/file.txt", as_list=True)

# Format/normalize paths
ptk.format_path("C:\\Users\\name\\file.txt")
# Returns: 'C:/Users/name/file.txt'

# Create directories (like mkdir -p)
ptk.create_dir("/path/to/new/directory")
```

### Iteration Utilities

```python
import pythontk as ptk

# Flatten nested lists
ptk.flatten([[1, 2], [3, [4, 5]]], return_type=list)
# Returns: [1, 2, 3, 4, 5]

# Filter lists with include/exclude patterns
files = ["test.py", "main.py", "test_utils.py", "config.json"]
ptk.filter_list(files, inc=["*.py"], exc=["test*"])
# Returns: ['main.py']

# Make anything iterable (strings stay as single items)
ptk.make_iterable("hello")      # Returns: ('hello',)
ptk.make_iterable([1, 2, 3])    # Returns: [1, 2, 3]
ptk.make_iterable(None)         # Returns: ()

# Remove duplicates while preserving order
ptk.remove_duplicates([1, 2, 2, 3, 1, 4])
# Returns: [1, 2, 3, 4]
```

### Math Utilities

```python
import pythontk as ptk

# Clamp values to a range
ptk.clamp(15, minimum=0, maximum=10)  # Returns: 10
ptk.clamp(-5, minimum=0, maximum=10)  # Returns: 0

# Works with lists via @listify decorator
ptk.clamp([1, 5, 15, -3], minimum=0, maximum=10)
# Returns: [1, 5, 10, 0]

# Linear interpolation
ptk.lerp(0, 100, 0.5)   # Returns: 50.0
ptk.lerp(0, 100, 0.25)  # Returns: 25.0

# Remap values from one range to another
ptk.remap(50, old_range=(0, 100), new_range=(0, 1))  # Returns: 0.5
ptk.remap(0.5, old_range=(0, 1), new_range=(0, 255))  # Returns: 127.5

# Vector operations
ptk.get_vector_from_two_points([0, 0, 0], [1, 2, 3])
# Returns: (1, 2, 3)

# Normalize a vector
ptk.normalize((2, 3, 4))
# Returns: (0.371..., 0.557..., 0.743...)
```

### Image Utilities

```python
import pythontk as ptk

# Pack grayscale images into RGBA channels
# Useful for game engine texture packing (e.g., ORM maps)
packed = ptk.ImgUtils.pack_channels({
    "R": "metallic.png",
    "G": "roughness.png",
    "B": "ao.png",
    "A": "height.png"
})
ptk.ImgUtils.save_image(packed, "packed_orm.png")

# Resize images
img = ptk.ImgUtils.load_image("texture.png")
resized = ptk.ImgUtils.resize_image(img, (512, 512))

# Check if images are identical
ptk.ImgUtils.are_identical("img1.png", "img2.png")
```

### Core Utilities & Decorators

```python
import pythontk as ptk

# @listify - Make functions accept lists and return lists
@ptk.CoreUtils.listify()
def double(x):
    return x * 2

double(5)           # Returns: 10
double([1, 2, 3])   # Returns: [2, 4, 6]

# @cached_property - Cache expensive computations
class MyClass:
    @ptk.CoreUtils.cached_property
    def expensive_data(self):
        # Only computed once, then cached
        return load_expensive_data()
```

## Module Reference

| Module | Class | Description |
|--------|-------|-------------|
| `core_utils` | `CoreUtils` | Decorators, format helpers, attribute utilities |
| `str_utils` | `StrUtils` | String manipulation, sanitization, case conversion |
| `file_utils` | `FileUtils` | File I/O, directory operations, path handling |
| `iter_utils` | `IterUtils` | List operations, filtering, grouping, flattening |
| `math_utils` | `MathUtils` | Numeric operations, clamping, interpolation, vectors |
| `img_utils` | `ImgUtils` | Image loading, channel packing, resizing |
| `vid_utils` | `VidUtils` | Video frame extraction, compression |

### Additional Classes

- **`HelpMixin`**: Adds `help()` method to print class documentation
- **`FuzzyMatcher`**: Fuzzy string matching with configurable thresholds
- **`ProgressionCurves`**: Easing functions for animations/interpolation
- **`SingletonMixin`**: Mixin for singleton pattern
- **`LoggingMixin`**: Structured logging support

## Contributing

Contributions welcome! Please ensure new utilities:
- Follow the existing class-based pattern
- Include docstrings with Parameters/Returns/Example sections
- Add corresponding tests

## License

MIT License
