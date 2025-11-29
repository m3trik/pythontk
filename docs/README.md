<p align="center">
  <h1 align="center">pythontk</h1>
  <p align="center">A modular Python utility toolkit for everyday programming tasks</p>
</p>

<p align="center">
  <a href="https://github.com/m3trik/pythontk/actions/workflows/tests.yml"><img src="https://github.com/m3trik/pythontk/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/pythontk/"><img src="https://img.shields.io/badge/Version-0.7.34-blue.svg" alt="Version"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
</p>

---

<!-- short_description_start -->
*Utility functions and mixin classes for file operations, text processing, data filtering, image/video manipulation, and common programming patterns.*
<!-- short_description_end -->

## Installation

```bash
pip install pythontk
```

<details>
<summary>Development installation</summary>

```bash
git clone https://github.com/m3trik/pythontk.git
cd pythontk
pip install -e .
```
</details>

## Quick Example

```python
import pythontk as ptk

# Filter files by pattern
files = ptk.get_dir_contents('/project', recursive=True)
scripts = ptk.filter_list(files, inc='*.py', exc='*test*')

# Clean strings for filenames
safe_name = ptk.sanitize('My File (v2)!.txt')  # 'My_File_v2_.txt'

# Format paths consistently
path = ptk.format_path('C:\\Users\\docs/file.txt', 'forward')  # 'C:/Users/docs/file.txt'
```

## Modules

### File Operations
```python
ptk.get_dir_contents(path, recursive=True)    # List directory contents
ptk.get_file_contents(path)                    # Read file
ptk.write_to_file(path, content)               # Write file
ptk.format_path(path, 'forward')               # Normalize path separators
```

### Text Processing
```python
ptk.sanitize('text!@#', replacement_char='_')  # Clean for filenames
ptk.set_case('hello world', 'title')           # 'Hello World'
ptk.set_case('hello world', 'camel')           # 'helloWorld'

# Fuzzy string matching
from pythontk import FuzzyMatcher
FuzzyMatcher.find_best_match('mesh_03', ['mesh_01', 'mesh_02'])
```

### Data Filtering
```python
# Lists - supports wildcards
ptk.filter_list(['a.py', 'b.txt', 'c.py'], inc='*.py')  # ['a.py', 'c.py']

# Dictionaries
ptk.filter_dict({'a': 1, 'b': 2, 'c': 3}, inc=['a', 'b'])  # {'a': 1, 'b': 2}
```

### Image Utilities
Requires `Pillow`
```python
from pythontk import ImgUtils

ImgUtils.get_image_size('texture.png')
ImgUtils.pack_channels(output='packed.png', R='rough.png', G='metal.png', B='ao.png')
ImgUtils.get_channels('image.png')
```

### Video Utilities
Requires `FFmpeg` in PATH
```python
from pythontk import VidUtils

VidUtils.extract_frames('video.mp4', output_dir='frames/')
VidUtils.get_video_info('video.mp4')
```

### Math Utilities
```python
ptk.lerp(0, 100, 0.5)      # 50.0 - Linear interpolation
ptk.clamp(150, 0, 100)     # 100  - Clamp to range

# Easing curves for animation
from pythontk import ProgressionCurves
ProgressionCurves.calculate_progression_factor(i, total, calculation_mode='ease_in_out')
```

## Mixin Classes

Reusable base classes for common patterns:

```python
from pythontk import SingletonMixin, LoggingMixin, HelpMixin

class Config(SingletonMixin):
    """Only one instance ever created."""
    pass

class Processor(LoggingMixin):
    """Automatic logger attribute."""
    def run(self):
        self.logger.info("Processing...")

class Tool(HelpMixin):
    """Self-documenting with .help() method."""
    pass
```

## Additional Utilities

| Class | Purpose |
|-------|---------|
| `ExecutionMonitor` | Profile execution time and memory |
| `PackageManager` | Pip operations from Python |
| `ModuleReloader` | Hot-reload modules during development |
| `NamespaceHandler` | Dynamic attribute containers |
| `NamedTupleContainer` | Manage collections of named tuples |
| `HierarchyDiff` | Compare hierarchical structures |

## Testing

```bash
cd test
pytest -v
```

## License

MIT
