# Quick Reference

## Installation
```bash
pip install pythontk
```

## Basic Import
```python
import pythontk as ptk
```

## File Operations
```python
# Check existence
ptk.is_valid('/path/file.txt', 'file')  # Check file
ptk.is_valid('/path/dir', 'dir')        # Check directory

# Directory operations
ptk.create_dir('/new/path')             # Create directory
files = ptk.get_dir_contents('/path')   # List contents
py_files = ptk.get_dir_contents('/path', inc='*.py')  # Filter

# File I/O
content = ptk.get_file_contents('file.txt')    # Read file
ptk.write_file('output.txt', 'content')        # Write file
```

## String Operations
```python
# Sanitization
clean = ptk.sanitize('My File!@#')      # → 'my_file'
clean = ptk.sanitize('text', replacement_char='-')  # Custom char

# Text extraction
html = '<div>content</div>'
text = ptk.get_text_between_delimiters(html, '<div>', '</div>')

# Case conversion
snake = ptk.camel_to_snake('myVariable')  # → 'my_variable'
camel = ptk.snake_to_camel('my_variable') # → 'myVariable'
```

## List/Dict Filtering
```python
# List filtering with wildcards
files = ['script.py', 'test.py', 'readme.txt']
py_files = ptk.filter_list(files, inc='*.py')           # Include
source = ptk.filter_list(files, inc='*.py', exc='test*') # Exclude

# Dictionary filtering
data = {'user_name': 'john', 'admin_role': True}
user_data = ptk.filter_dict(data, inc='user_*', keys=True)
```

## Image Processing
```python
# Basic operations
ptk.resize_image('input.jpg', 1024, 1024)
blank = ptk.create_image('RGB', (512, 512), (255, 255, 255))

# Channel packing
ptk.pack_channels({
    'R': 'red.jpg',
    'G': 'green.jpg',
    'B': 'blue.jpg'
}, output_path='packed.png')

# PBR conversion
base, metal, rough = ptk.convert_spec_gloss_to_metal_rough(
    'spec.jpg', 'gloss.jpg', write_files=True
)

# Batch optimization
ptk.batch_optimize_textures('/textures/', max_size=2048)
```

## Video Processing
```python
# Frame extraction
frames = ptk.extract_frames(
    'video.mp4', 'frames/', 
    step=30, quality=95
)

# Video info
info = ptk.get_video_info('video.mp4')
print(f"Duration: {info['duration']}s")
```

## Math Operations
```python
# Decimal manipulation
result = ptk.move_decimal_point(123.45, -2)  # → 1.2345

# Vector operations
vector = ptk.get_vector_from_two_points([0,0,0], [1,1,1])

# Value operations
clamped = ptk.clamp(150, 0, 100)     # → 100
lerped = ptk.lerp(0, 100, 0.5)       # → 50.0
```

## Common Patterns

### File Processing Pipeline
```python
# Find all Python files, process them
py_files = ptk.get_dir_contents('/project', inc='*.py', recursive=True)
for file in py_files:
    content = ptk.get_file_contents(file)
    # Process content
    processed = process_code(content)
    # Save to new location
    output_path = file.replace('/project', '/processed')
    ptk.write_file(output_path, processed)
```

### Texture Workflow
```python
# Convert texture workflow from spec/gloss to metal/rough
texture_dir = '/game_assets/textures'
spec_files = ptk.get_dir_contents(texture_dir, inc='*spec*')

for spec_file in spec_files:
    gloss_file = spec_file.replace('_spec', '_gloss')
    if ptk.is_valid(gloss_file, 'file'):
        ptk.convert_spec_gloss_to_metal_rough(
            spec_file, gloss_file, 
            output_dir=texture_dir + '/converted',
            write_files=True
        )
```

### Batch Image Resize
```python
# Resize all images in directory
images = ptk.get_dir_contents('/images', inc=['*.jpg', '*.png'])
for img in images:
    resized = ptk.resize_image(img, 1024, 1024)
    output_path = img.replace('/images', '/resized')
    ptk.save_image(resized, output_path)
```

## Error Handling
```python
try:
    result = ptk.process_operation(input_data)
except FileNotFoundError:
    print("File not found")
except ValueError:
    print("Invalid input")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

### Threading
```python
from pythontk.core_utils import CoreUtils

@CoreUtils.listify(threading=True)
def process_item(item):
    return expensive_operation(item)

# Automatically uses threading for lists
results = process_item(large_list)
```

### Memory Management
```python
# Process large batches efficiently
for i in range(0, len(large_list), 100):
    batch = large_list[i:i+100]
    results = process_batch(batch)
    save_results(results)
    # Explicit cleanup for large objects
    del batch, results
```

### Caching
```python
from pythontk.core_utils import CoreUtils

class Processor:
    @CoreUtils.cached_property
    def expensive_resource(self):
        return create_expensive_resource()
    
    def process(self, data):
        return process_with_resource(data, self.expensive_resource)
```

## Configuration

### Logging
```python
import logging
logging.getLogger('pythontk').setLevel(logging.DEBUG)
```

### Environment Variables
```python
# FFmpeg path for video processing
export MAYA_SCRIPT_PATH="/path/to/ffmpeg/bin"

# Default encoding
export PYTHONIOENCODING="utf-8"
```

## Common Use Cases

| Task | Function | Example |
|------|----------|---------|
| Find files | `get_dir_contents()` | `ptk.get_dir_contents('/dir', inc='*.py')` |
| Clean filenames | `sanitize()` | `ptk.sanitize('My File!@#')` |
| Filter lists | `filter_list()` | `ptk.filter_list(items, inc='*pattern*')` |
| Resize images | `resize_image()` | `ptk.resize_image('img.jpg', 512, 512)` |
| Pack channels | `pack_channels()` | `ptk.pack_channels({'R': 'r.jpg'})` |
| Extract frames | `extract_frames()` | `ptk.extract_frames('vid.mp4', 'frames/')` |
| Vector math | `get_vector_from_two_points()` | `ptk.get_vector_from_two_points([0,0,0], [1,1,1])` |

## Module Reference
- `core_utils`: Base classes, decorators, mixins
- `file_utils`: File/directory operations  
- `str_utils`: String manipulation
- `img_utils`: Image processing
- `vid_utils`: Video processing
- `math_utils`: Mathematical operations
- `iter_utils`: List/dict filtering

## Links
- [Full Documentation](README.md)
- [API Reference](API.md)
- [Examples](EXAMPLES.md)
- [Developer Guide](DEVELOPER.md)
- [Changelog](CHANGELOG.md)
