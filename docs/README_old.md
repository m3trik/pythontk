[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/Version-0.7.28-blue.svg)](https://pypi.org/project/pythontk/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)

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
files = ptk.filter_list(
    ptk.get_dir_contents('/path/to/directory'),
    inc=['*.py', '*.txt'], exc='*temp*'
)

# Path formatting
clean_path = ptk.format_path('/path\\to/file.txt', style='forward')
```

### Text Processing
```python
# String sanitization
clean_name = ptk.sanitize('My Asset Name!@#', replacement_char='_')
# Result: 'My_Asset_Name'

# Text formatting
formatted = ptk.format_string('hello world', style='title')
# Result: 'Hello World'
```

### Basic Image Operations
```python
# Requires PIL/Pillow
from pythontk import ImgUtils

# Create image from channels
img = ImgUtils.pack_channels({
    'R': red_channel_path,
    'G': green_channel_path,
    'B': blue_channel_path
})

# Get image dimensions
width, height = ImgUtils.get_image_size('image.jpg')
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
- `format_string()`: Text case formatting
- `remove_chars()`: Character removal

### Image Utilities (`img_utils`)
Basic image operations (requires PIL):
- `get_image_size()`: Get image dimensions
- `pack_channels()`: Combine image channels
- `get_channels()`: Extract image channels

### Video Utilities (`vid_utils`)
Simple video operations (requires FFmpeg):
- `extract_frames()`: Extract video frames
- `get_video_info()`: Video metadata

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

import os

class AssetPipeline:
    def __init__(self, project_root, department='modeling'):
        self.project_root = project_root
        self.department = department
        self.processed_assets = []
    
    def discover_assets(self, asset_types=['ma', 'mb', 'fbx']):
        """Discover assets with department-specific filtering."""
        patterns = [f'*.{ext}' for ext in asset_types]
        department_filter = f'*{self.department}*' if self.department else '*'
        
        return ptk.filter_list(
            ptk.get_dir_contents(self.project_root, recursive=True),
            inc=patterns + [department_filter],
            exc=['*_backup*', '*_temp*', '*_archive*'],
            basename_only=False
        )
    
    def validate_naming_convention(self, asset_paths):
        """Enforce studio naming conventions."""
        validated_assets = []
        
        for asset_path in asset_paths:
            filename = os.path.basename(asset_path)
            clean_name = ptk.sanitize(
                filename,
                replacement_char='_',
                char_map={'(': '_', ')': '_', ' ': '_'},
                preserve_case=False
            )
            
            if clean_name != filename:
                # Log naming violations for review
                self.log_naming_violation(asset_path, clean_name)
            
            validated_assets.append((asset_path, clean_name))
        
        return validated_assets
    
    def process_texture_assets(self, texture_directory):
        """Automated texture processing for modern workflows."""
        texture_sets = self.group_texture_sets(texture_directory)
        
        for set_name, textures in texture_sets.items():
            # Legacy workflow conversion
            if self.has_spec_gloss_workflow(textures):
                self.convert_to_pbr_workflow(textures, set_name)
            
            # Optimization for target platform
            self.optimize_texture_set(textures, set_name)
        
        return self.processed_assets

# Usage in production environment
pipeline = AssetPipeline('/project/assets', 'modeling')
discovered_assets = pipeline.discover_assets(['ma', 'mb'])
validated_assets = pipeline.validate_naming_convention(discovered_assets)
```

### Batch Processing with Error Recovery
```python
# Enterprise-grade batch processing with comprehensive error handling
import pythontk as ptk
from concurrent.futures import ThreadPoolExecutor
import logging

def process_asset_batch(asset_list, max_workers=None):
    """Process assets with parallel execution and error recovery."""
    successful_processes = []
    failed_processes = []
    
    @ptk.CoreUtils.listify(threading=True)
    def process_single_asset(asset_path):
        try:
            # Validate asset integrity
            if not ptk.is_valid(asset_path, 'file'):
                raise FileNotFoundError(f"Asset not found: {asset_path}")
            
            # Process based on asset type
            if asset_path.endswith(('.exr', '.tiff', '.tx')):
                return process_texture_asset(asset_path)
            elif asset_path.endswith(('.ma', '.mb')):
                return process_scene_asset(asset_path)
            else:
                return process_generic_asset(asset_path)
                
        except Exception as e:
            logging.error(f"Failed to process {asset_path}: {e}")
            failed_processes.append((asset_path, str(e)))
            return None
    
    # Execute batch processing
    results = process_single_asset(asset_list)
    successful_processes = [r for r in results if r is not None]
    
    return {
        'successful': len(successful_processes),
        'failed': len(failed_processes),
        'results': successful_processes,
        'errors': failed_processes
    }
```

### Configuration Management
```python
# Centralized configuration system for pipeline tools
import pythontk as ptk
import json

class PipelineConfig:
    def __init__(self, config_path='/pipeline/config/settings.json'):
        self.config_path = config_path
        self._config_cache = None
    
    @ptk.CoreUtils.cached_property
    def settings(self):
        """Load and cache pipeline configuration."""
        if ptk.is_valid(self.config_path, 'file'):
            content = ptk.get_file_contents(self.config_path)
            return json.loads(content)
        return self.get_default_config()
    
    def get_department_settings(self, department):
        """Extract department-specific configuration."""
        return ptk.filter_dict(
            self.settings,
            inc=[f'{department}_*', 'global_*'],
            keys=True
        )
    
    def get_texture_specifications(self):
        """Get texture processing specifications."""
        texture_config = ptk.filter_dict(
            self.settings,
            inc=['texture_*', 'quality_*', 'format_*'],
            keys=True
        )
        
        return {
            'max_resolution': texture_config.get('texture_max_resolution', 4096),
            'compression_quality': texture_config.get('quality_compression', 95),
            'output_format': texture_config.get('format_output', 'EXR')
        }
```

## Enterprise Considerations

### Error Handling and Logging
Production environments require robust error handling and comprehensive logging:

```python
import pythontk as ptk
import logging

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

try:
    # Asset processing with comprehensive error context
    processed_assets = ptk.batch_process_textures(asset_directory)
    logging.info(f"Successfully processed {len(processed_assets)} assets")
    
except FileNotFoundError as e:
    logging.error(f"Asset directory not accessible: {e}")
    # Implement fallback or notification system
    
except Exception as e:
    logging.critical(f"Unexpected pipeline failure: {e}")
    # Trigger incident response procedures
```

### Performance Optimization
Memory management and processing efficiency for large-scale operations:

```python
# Memory-efficient processing for large asset libraries
import pythontk as ptk

def process_large_asset_library(asset_paths, batch_size=50):
    """Process assets in memory-managed batches."""
    total_processed = 0
    
    for i in range(0, len(asset_paths), batch_size):
        batch = asset_paths[i:i + batch_size]
        
        # Process batch with automatic threading
        results = ptk.process_asset_batch(batch)
        
        # Memory cleanup between batches
        del batch, results
        total_processed += len(batch)
        
        # Progress reporting for long-running operations
        logging.info(f"Processed {total_processed}/{len(asset_paths)} assets")
    
    return total_processed

# Threading optimization for I/O intensive operations
from pythontk.core_utils import CoreUtils

@CoreUtils.listify(threading=True)
def validate_asset_integrity(asset_path):
    """Validate individual assets with automatic threading support."""
    return ptk.comprehensive_asset_validation(asset_path)

# Automatic parallel execution for large asset sets
validation_results = validate_asset_integrity(large_asset_list)
```

### Scalability and Integration
Design patterns for studio-scale deployment:

```python
# Modular pipeline architecture
class StudioPipeline:
    def __init__(self, studio_config):
        self.config = studio_config
        self.processors = self.initialize_processors()
    
    def initialize_processors(self):
        """Initialize department-specific processors."""
        return {
            'modeling': ModelingProcessor(self.config),
            'texturing': TextureProcessor(self.config),
            'animation': AnimationProcessor(self.config),
            'lighting': LightingProcessor(self.config)
        }
    
    def process_department_assets(self, department, asset_list):
        """Route assets to appropriate department processors."""
        processor = self.processors.get(department)
        if not processor:
            raise ValueError(f"Unknown department: {department}")
        
        return processor.process_assets(asset_list)

# Integration with external systems
class AssetDatabase:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def sync_processed_assets(self, processed_assets):
        """Update asset database with processing results."""
        for asset_info in processed_assets:
            standardized_path = ptk.format_path(asset_info['path'])
            clean_metadata = ptk.filter_dict(
                asset_info['metadata'],
                exc=['temp_*', 'cache_*'],
                keys=True
            )
            self.db.update_asset_record(standardized_path, clean_metadata)
```

## Technical Requirements

**Core Dependencies:**
- Python 3.7+ (3.9+ recommended for production)
- Cross-platform compatibility (Windows, Linux, macOS)

**Optional Enhancement Libraries:**
- `Pillow`: Required for texture processing and image manipulation
- `numpy`: Advanced mathematical operations and image calculations
- `FFmpeg`: Video processing and frame extraction capabilities

**Enterprise Integrations:**
- Compatible with major DCC applications (Maya, 3ds Max, Houdini, Blender)
- Supports studio pipeline frameworks and asset management systems
- Integrates with version control systems and build automation

## Development and Contribution

### Professional Development Standards
1. Fork the repository and create feature branches
2. Implement comprehensive unit tests for new functionality  
3. Follow PEP 8 coding standards and type annotations
4. Submit pull requests with detailed technical documentation

### Code Quality Requirements
- Comprehensive error handling and logging
- Performance benchmarking for batch operations
- Memory efficiency testing for large-scale processing
- Cross-platform compatibility validation

## License and Support

**License:** MIT License - see [LICENSE](../LICENSE) for complete terms

**Professional Support:**
- **Technical Issues:** [GitHub Issues](https://github.com/m3trik/pythontk/issues)
- **Feature Requests:** [GitHub Discussions](https://github.com/m3trik/pythontk/discussions)
- **Documentation:** [Complete API Reference](API.md)
- **Integration Examples:** [Production Workflows](EXAMPLES.md)

## Version Information

### Current Release: v0.7.28
- Enhanced texture processing pipeline with PBR workflow automation
- Improved enterprise-grade error handling and logging systems  
- Performance optimizations for large-scale batch operations
- Extended cross-platform compatibility and path handling

### Previous Releases
- **v0.7.27:** Video processing utilities and FFmpeg integration
- **v0.7.26:** Advanced image channel manipulation and texture optimization
- **v0.7.25:** Mathematical utilities and precision calculation tools

**Migration Documentation:** [Complete Changelog](CHANGELOG.md)

---

*This framework is actively maintained and deployed in professional production environments. Technical documentation is updated with each release to ensure accuracy and completeness.*
