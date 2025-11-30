[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/Version-0.7.34-blue.svg)](https://pypi.org/project/pythontk/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-389%20passed-brightgreen.svg)](test/)


# PYTHONTK (Python Toolkit)

---
<!-- short_description_start -->
*A collection of Python utility functions for file operations, text processing, and basic image/video manipulation. Provides helper classes and convenience functions for common programming tasks.*
<!-- short_description_end -->

## Features

pythontk provides utility functions organized into focused modules:

- **File Operations**: Directory listing, file reading/writing, and path utilities
- **Text Processing**: String sanitization, formatting, and text manipulation
- **Data Filtering**: List and dictionary filtering with pattern matching
- **Image Utilities**: Basic image operations and texture map processing (requires Pillow)
- **Video Utilities**: Simple video operations using FFmpeg
- **Math Utilities**: Basic mathematical helper functions
- **Core Utilities**: Decorators and helper classes for common patterns

## Installation

**Python Requirements:**
- Python 3.7+

**Installation:**
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
- `PIL` (Pillow) for image operations
- `numpy` for mathematical operations  
- `FFmpeg` for video utilities

## Usage Examples

### File Operations
```python
import pythontk as ptk

# Get directory contents with filtering
files = ptk.get_dir_contents(
    '/path/to/directory',
    inc_files=['*.py', '*.txt'],
    exc_files=['*temp*']
)

# Path formatting (normalizes slashes, extracts path sections)
clean_path = ptk.format_path('/path\\to/file.txt')  # Returns: 'path/to/file.txt'
filename = ptk.format_path('/path/to/file.txt', section='file')  # Returns: 'file.txt'
```

### Text Processing
```python
# String sanitization
clean_name = ptk.sanitize('My Asset Name!@#', replacement_char='_')
# Result: 'my_asset_name'

# Text case formatting
formatted = ptk.set_case('hello world', case='title')
# Result: 'Hello World'
```

### Basic Image Operations
```python
# Requires PIL/Pillow
from pythontk import ImgUtils

# Pack grayscale images into RGB/RGBA channels
img = ImgUtils.pack_channels({
    'R': 'red_channel.png',
    'G': 'green_channel.png',
    'B': 'blue_channel.png'
})

# Save image
ImgUtils.save_image(img, 'output.png')
```

### Data Processing
```python
# Filter lists and dictionaries
filtered_data = ptk.filter_dict(
    data, inc=['name', 'version'], keys=True
)

# List filtering with patterns
python_files = ptk.filter_list(
    file_list, inc=['*.py'], exc=['*test*']
)
```

## Module Overview

### Core Utilities (`core_utils`)
Basic helper classes and decorators:
- `HelpMixin`: Documentation helper
- `cached_property`: Property caching decorator
- `@listify`: Convert returns to lists

### File Utilities (`file_utils`)  
File system operations:
- `get_dir_contents()`: Directory listing with filtering
- `get_file_contents()`: Read file contents
- `create_dir()`: Directory creation
- `format_path()`: Path formatting

### String Utilities (`str_utils`)
Text processing functions:
- `sanitize()`: Clean strings for filenames
- `set_case()`: Text case formatting
- `find_str()`: Search strings with wildcards
- `truncate()`: Shorten strings with ellipsis

### Image Utilities (`img_utils`)
Basic image operations (requires PIL):
- `ensure_image()`: Load and validate images
- `pack_channels()`: Combine image channels
- `save_image()`: Save images to file
- `resize_image()`: Resize images

### Video Utilities (`vid_utils`)
Simple video operations (requires FFmpeg):
- `get_video_frame_rate()`: Get video frame rate
- `compress_video()`: Compress video files

### Math Utilities (`math_utils`)
Mathematical helper functions for common calculations.

## Dynamic Imports

pythontk uses a dynamic attribute resolution system allowing direct access to utility functions:

```python
import pythontk as ptk

# These are equivalent:
ptk.sanitize('text')           # Direct access
ptk.str_utils.sanitize('text') # Module-specific access
```

## Contributing

This is a utility collection for common Python tasks. Contributions that add useful, focused utility functions are welcome.

## License

This project is licensed under the MIT License.
