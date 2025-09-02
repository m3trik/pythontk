# Usage Examples

## File Operations

### Basic File Operations
```python
import pythontk as ptk

# Check if file/directory exists
if ptk.is_valid('/path/to/file.txt', 'file'):
    print("File exists")

if ptk.is_valid('/path/to/directory', 'dir'):
    print("Directory exists")

# Create directory structure
ptk.create_dir('/path/to/new/structure')

# Read file with encoding detection
content = ptk.get_file_contents('config.txt')
print(content)

# Write file with directory creation
ptk.write_file('/new/path/output.txt', 'Hello World')
```

### Directory Scanning
```python
import pythontk as ptk

# Get all files in directory
all_files = ptk.get_dir_contents('/project')

# Get only Python files
python_files = ptk.get_dir_contents('/project', inc='*.py')

# Get files recursively, excluding tests
source_files = ptk.get_dir_contents(
    '/project', 
    inc='*.py', 
    exc='*test*',
    recursive=True
)

# Get just filenames instead of full paths
filenames = ptk.get_dir_contents(
    '/project', 
    return_type='filename',
    inc='*.py'
)
```

---

## String Manipulation

### Text Sanitization
```python
import pythontk as ptk

# Basic sanitization
clean = ptk.sanitize("My File Name!@#$")
# Returns: "my_file_name"

# Custom replacement character
clean = ptk.sanitize("My File Name!@#$", replacement_char="-")
# Returns: "my-file-name"

# Preserve case
clean = ptk.sanitize("My File Name!@#$", preserve_case=True)
# Returns: "My_File_Name"

# Custom character mapping
char_map = {'@': 'at', '#': 'hash', '$': 'dollar'}
clean = ptk.sanitize("user@domain#tag$", char_map=char_map)
# Returns: "user_at_domain_hash_tag_dollar"

# Return both original and cleaned
original, cleaned = ptk.sanitize("My File!", return_original=True)
```

### Text Extraction
```python
import pythontk as ptk

# Extract text between delimiters
html = "<div>Hello World</div>"
content = ptk.get_text_between_delimiters(html, "<div>", "</div>")
# Returns: "Hello World"

# Multiple occurrences
text = "Start<!-- content -->First<!-- /content -->Middle<!-- content -->Second<!-- /content -->End"
contents = ptk.get_text_between_delimiters(text, "<!-- content -->", "<!-- /content -->")
# Returns: ["First", "Second"]

# As single string
content = ptk.get_text_between_delimiters(text, "<!-- content -->", "<!-- /content -->", as_string=True)
# Returns: "First Second"
```

### Case Conversion
```python
import pythontk as ptk

# camelCase to snake_case
snake = ptk.camel_to_snake("myVariableName")
# Returns: "my_variable_name"

# snake_case to camelCase
camel = ptk.snake_to_camel("my_variable_name")
# Returns: "myVariableName"
```

---

## List and Dictionary Filtering

### List Filtering
```python
import pythontk as ptk

# Basic filtering with wildcards
files = ['script.py', 'test.py', 'readme.txt', 'config.json']

# Include only Python files
python_files = ptk.filter_list(files, inc='*.py')
# Returns: ['script.py', 'test.py']

# Exclude test files
source_files = ptk.filter_list(files, inc='*.py', exc='test*')
# Returns: ['script.py']

# Multiple include patterns
code_files = ptk.filter_list(files, inc=['*.py', '*.json'])
# Returns: ['script.py', 'test.py', 'config.json']
```

### Advanced List Filtering
```python
import pythontk as ptk

# Filter with transformation function
names = ['Apple', 'banana', 'Cherry', 'date']
lowercase_a = ptk.filter_list(
    names, 
    inc='*a*', 
    map_func=str.lower,  # Compare lowercase versions
    check_unmapped=True  # Return original case
)
# Returns: ['Apple', 'banana']

# Filter file paths by basename only
paths = ['/home/user/script.py', '/tmp/test.py', '/var/log/error.log']
py_files = ptk.filter_list(paths, inc='*.py', basename_only=True)
# Returns: ['/home/user/script.py', '/tmp/test.py']

# Case-insensitive filtering
mixed_case = ['FILE.TXT', 'script.py', 'README.md']
txt_files = ptk.filter_list(mixed_case, inc='*.txt', ignore_case=True)
# Returns: ['FILE.TXT']
```

### Dictionary Filtering
```python
import pythontk as ptk

# Filter by keys
data = {
    'user_name': 'john',
    'user_age': 30,
    'admin_role': True,
    'system_version': '1.0'
}

# Get user-related data
user_data = ptk.filter_dict(data, inc='user_*', keys=True)
# Returns: {'user_name': 'john', 'user_age': 30}

# Exclude admin data
public_data = ptk.filter_dict(data, exc='admin_*', keys=True)
# Returns: {'user_name': 'john', 'user_age': 30, 'system_version': '1.0'}

# Filter by values
numbers = {'a': 1, 'b': 'text', 'c': 2, 'd': 'more text'}
numeric_data = ptk.filter_dict(numbers, inc=[1, 2], values=True)
# Returns: {'a': 1, 'c': 2}
```

---

## Image Processing

### Basic Image Operations
```python
import pythontk as ptk

# Resize image
ptk.resize_image('input.jpg', 1024, 1024)

# Create blank image
blank = ptk.create_image('RGB', (512, 512), (255, 255, 255))

# Convert image format
ptk.save_image('input.png', 'output.jpg')

# Load and process
image = ptk.load_image('texture.png')
resized = ptk.resize_image(image, 2048, 2048)
ptk.save_image(resized, 'texture_2k.png')
```

### Channel Packing
```python
import pythontk as ptk

# Pack separate channels into RGBA texture
channel_files = {
    'R': 'metallic.jpg',     # Red channel
    'G': 'roughness.jpg',    # Green channel  
    'B': 'ambient_occlusion.jpg',  # Blue channel
    'A': 'height.jpg'        # Alpha channel
}

packed = ptk.pack_channels(
    channel_files, 
    output_path='packed_texture.png'
)

# Pack with custom channel order
custom_packed = ptk.pack_channels(
    {'R': 'red.jpg', 'G': 'green.jpg'}, 
    channels=['R', 'G', 'B'],  # Fill B with default
    out_mode='RGB',
    fill_values={'B': 128}  # Fill B channel with gray
)
```

### PBR Texture Conversion
```python
import pythontk as ptk

# Convert old spec/gloss workflow to modern metal/rough
base_color, metallic, roughness = ptk.convert_spec_gloss_to_metal_rough(
    specular_map='diffuse_spec.jpg',
    glossiness_map='gloss.jpg',
    write_files=True,
    output_dir='converted_textures/'
)

# Just create metallic map
metallic_map = ptk.create_metallic_from_spec(
    'specular.jpg',
    threshold=60,  # Adjust metallic threshold
    softness=0.3   # Adjust transition softness
)

# Create roughness from glossiness
roughness_map = ptk.create_roughness_from_spec(
    'specular.jpg',
    'glossiness.jpg'
)
```

### Batch Processing
```python
import pythontk as ptk

# Optimize all textures in directory
ptk.batch_optimize_textures(
    '/textures/original/',
    max_size=2048,
    output_dir='/textures/optimized/'
)

# Process multiple texture sets
import os

texture_dir = '/game_assets/textures/'
for subfolder in os.listdir(texture_dir):
    subfolder_path = os.path.join(texture_dir, subfolder)
    if os.path.isdir(subfolder_path):
        ptk.batch_optimize_textures(subfolder_path, max_size=1024)
```

---

## Video Processing

### Frame Extraction
```python
import pythontk as ptk

# Extract every 30th frame
frames = ptk.extract_frames(
    video_path='input.mp4',
    output_folder='frames/',
    step=30,
    quality=95,
    prefix='scene_'
)

# Extract first 100 frames only
limited_frames = ptk.extract_frames(
    video_path='long_video.mp4',
    output_folder='preview_frames/',
    step=1,
    max_frames=100
)

# Get video information first
info = ptk.get_video_info('video.mp4')
print(f"Video duration: {info.get('duration', 'unknown')} seconds")
print(f"Frame rate: {info.get('fps', 'unknown')} fps")

# Extract based on duration
if info.get('duration', 0) > 60:  # If longer than 1 minute
    frames = ptk.extract_frames('video.mp4', 'frames/', step=60)  # Every 60th frame
```

---

## Mathematical Operations

### Decimal Precision
```python
import pythontk as ptk

# Move decimal point
result = ptk.move_decimal_point(123.456, 2)   # Returns: 12345.6
result = ptk.move_decimal_point(123.456, -2)  # Returns: 1.23456

# Process multiple values
values = [10.5, 20.3, 30.7]
moved = [ptk.move_decimal_point(v, -1) for v in values]
# Returns: [1.05, 2.03, 3.07]
```

### Vector Operations
```python
import pythontk as ptk

# Calculate direction vector
start_point = [0, 0, 0]
end_point = [10, 5, -3]
direction = ptk.get_vector_from_two_points(start_point, end_point)
# Returns: (10.0, 5.0, -3.0)

# Multiple vectors
points = [
    ([0, 0, 0], [1, 0, 0]),
    ([0, 0, 0], [0, 1, 0]),
    ([0, 0, 0], [0, 0, 1])
]
vectors = [ptk.get_vector_from_two_points(p1, p2) for p1, p2 in points]
```

### Value Clamping and Interpolation
```python
import pythontk as ptk

# Clamp values within range
clamped = ptk.clamp(150, 0, 100)  # Returns: 100
clamped = ptk.clamp(-10, 0, 100)  # Returns: 0
clamped = ptk.clamp(50, 0, 100)   # Returns: 50

# Linear interpolation
mid_point = ptk.lerp(0, 100, 0.5)    # Returns: 50.0
quarter = ptk.lerp(0, 100, 0.25)     # Returns: 25.0

# Interpolate between colors
red = [255, 0, 0]
blue = [0, 0, 255]
purple = [ptk.lerp(red[i], blue[i], 0.5) for i in range(3)]
# Returns: [127.5, 0.0, 127.5]
```

---

## Advanced Patterns

### Combining Operations
```python
import pythontk as ptk
import os

# Complex file processing workflow
def process_project_textures(project_dir, output_dir):
    # Find all texture directories
    texture_dirs = ptk.get_dir_contents(
        project_dir, 
        'filepath',
        inc='*texture*',
        recursive=True
    )
    
    for tex_dir in texture_dirs:
        if not os.path.isdir(tex_dir):
            continue
            
        # Get all images
        images = ptk.get_dir_contents(tex_dir, inc=['*.jpg', '*.png', '*.tga'])
        
        # Filter for spec/gloss pairs
        spec_files = ptk.filter_list(images, inc='*spec*', basename_only=True)
        
        for spec_file in spec_files:
            # Find corresponding gloss file
            base_name = os.path.splitext(os.path.basename(spec_file))[0]
            base_name = base_name.replace('_spec', '').replace('_specular', '')
            
            gloss_candidates = ptk.filter_list(
                images, 
                inc=[f'*{base_name}*gloss*', f'*{base_name}*smooth*'],
                basename_only=True
            )
            
            if gloss_candidates:
                gloss_file = gloss_candidates[0]
                
                # Convert to metal/rough workflow
                ptk.convert_spec_gloss_to_metal_rough(
                    spec_file,
                    gloss_file,
                    output_dir=output_dir,
                    write_files=True
                )

# Use the function
process_project_textures('/game_project/', '/converted_textures/')
```

### Custom Processing Pipeline
```python
import pythontk as ptk

class TextureProcessor:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.processed = []
    
    def find_texture_sets(self):
        """Find all complete texture sets"""
        all_images = ptk.get_dir_contents(
            self.base_dir, 
            inc=['*.jpg', '*.png', '*.tga'],
            recursive=True
        )
        
        # Group by base name
        texture_sets = {}
        for img in all_images:
            basename = ptk.sanitize(os.path.basename(img).split('_')[0])
            if basename not in texture_sets:
                texture_sets[basename] = []
            texture_sets[basename].append(img)
        
        return texture_sets
    
    def process_set(self, texture_list):
        """Process a complete texture set"""
        # Find different map types
        diffuse = ptk.filter_list(texture_list, inc='*diffuse*', basename_only=True)
        normal = ptk.filter_list(texture_list, inc='*normal*', basename_only=True)
        spec = ptk.filter_list(texture_list, inc='*spec*', basename_only=True)
        
        results = {}
        if diffuse:
            results['diffuse'] = ptk.resize_image(diffuse[0], 2048, 2048)
        if normal:
            results['normal'] = ptk.resize_image(normal[0], 2048, 2048)
        if spec:
            results['metallic'] = ptk.create_metallic_from_spec(spec[0])
            results['roughness'] = ptk.create_roughness_from_spec(spec[0])
        
        return results
    
    def process_all(self):
        """Process all texture sets"""
        texture_sets = self.find_texture_sets()
        
        for set_name, textures in texture_sets.items():
            print(f"Processing {set_name}...")
            results = self.process_set(textures)
            self.processed.append((set_name, results))
            
        return self.processed

# Use the processor
processor = TextureProcessor('/asset_library/')
processed_sets = processor.process_all()
```

---

## Performance Tips

### Memory Management
```python
import pythontk as ptk

# Process large batches efficiently
def process_large_image_batch(image_paths):
    for i, path in enumerate(image_paths):
        # Process in chunks to manage memory
        if i % 10 == 0:
            print(f"Processing {i}/{len(image_paths)}")
        
        # Load, process, save, and release
        img = ptk.load_image(path)
        resized = ptk.resize_image(img, 1024, 1024)
        output_path = path.replace('.jpg', '_resized.jpg')
        ptk.save_image(resized, output_path)
        
        # Explicitly delete to free memory
        del img, resized
```

### Threading with @listify
```python
import pythontk as ptk
from pythontk.core_utils import CoreUtils

# Functions decorated with @listify support threading
@CoreUtils.listify(threading=True)
def process_single_file(filepath):
    # This will automatically work with lists and use threading
    return ptk.resize_image(filepath, 512, 512)

# Process multiple files in parallel
file_list = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
results = process_single_file(file_list)  # Processes in parallel
```

### Caching Expensive Operations
```python
import pythontk as ptk
from pythontk.core_utils import CoreUtils

class ImageProcessor:
    @CoreUtils.cached_property
    def expensive_filter(self):
        # This will only be computed once
        return self.create_complex_filter()
    
    def create_complex_filter(self):
        # Expensive operation
        return complex_computation()
    
    def process_image(self, image):
        # Use cached filter
        return apply_filter(image, self.expensive_filter)
```
