# Developer Guide

## Architecture Overview

Pythontk is organized into utility modules that can be used independently or together. The package uses dynamic module loading to provide a unified interface while keeping individual modules decoupled.

### Module Structure
```
pythontk/
├── __init__.py              # Dynamic loading and attribute resolution
├── core_utils/              # Base utilities and mixins
│   ├── _core_utils.py      # Core utility functions and decorators
│   ├── help_mixin.py       # Auto-documentation functionality
│   ├── logging_mixin.py    # Standardized logging
│   ├── singleton_mixin.py  # Singleton pattern implementation
│   └── ...
├── file_utils/             # File and directory operations
├── str_utils/              # String manipulation and processing
├── img_utils/              # Image processing with PIL
├── vid_utils/              # Video processing with FFmpeg
├── math_utils/             # Mathematical operations
└── iter_utils/             # Advanced iteration and filtering
```

## Dynamic Loading System

The package implements a sophisticated dynamic loading system that allows accessing any utility function directly from the main package:

### How It Works

1. **Discovery Phase**: At import time, `build_dictionaries()` scans all modules
2. **Mapping Creation**: Creates mappings of class names, methods, and functions to their modules
3. **Lazy Loading**: Modules are only imported when their functions are first accessed
4. **Attribute Resolution**: `__getattr__` resolves function calls to appropriate modules

```python
# In __init__.py
def __getattr__(name):
    if name in CLASS_TO_MODULE:
        module = import_module(CLASS_TO_MODULE[name])
        return get_attribute_from_module(module, name)
    # ... similar for methods and functions
```

This means:
```python
import pythontk as ptk
ptk.filter_list([1, 2, 3], inc=[1, 2])  # Automatically routes to IterUtils.filter_list
```

## Base Classes and Mixins

### HelpMixin
Provides automatic documentation generation for classes.

```python
class MyClass(HelpMixin):
    def my_method(self):
        """This method does something."""
        pass

# Automatic help generation
obj = MyClass()
obj.get_help()  # Shows formatted documentation
```

### LoggingMixin
Standardized logging across all modules.

```python
class MyProcessor(LoggingMixin):
    def process_data(self, data):
        self.logger.info("Starting data processing")
        # ... processing logic
        self.logger.debug(f"Processed {len(data)} items")
```

### SingletonMixin
Ensures only one instance of a class exists.

```python
class ConfigManager(SingletonMixin):
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.settings = {}

# Always returns same instance
config1 = ConfigManager()
config2 = ConfigManager()
assert config1 is config2  # True
```

## Core Decorators

### @cached_property
Caches the result of expensive property computations.

```python
class DataProcessor:
    @cached_property
    def expensive_data(self):
        """This computation only runs once."""
        return sum(range(1000000))
    
    def use_data(self):
        # First call computes and caches
        result1 = self.expensive_data
        # Subsequent calls use cached value
        result2 = self.expensive_data
```

**Implementation:**
```python
@staticmethod
def cached_property(func: Callable) -> Any:
    attr_name = f"_cached_{func.__name__}"
    
    @property
    @wraps(func)
    def _cached_property(self: Any) -> Any:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return _cached_property
```

### @listify
Makes functions work seamlessly with single items or lists.

```python
@listify(threading=True)
def process_item(item):
    return item * 2

# Works with single items
result = process_item(5)  # Returns: 10

# Works with lists (threaded)
results = process_item([1, 2, 3, 4])  # Returns: [2, 4, 6, 8]
```

**Key Features:**
- Automatic threading for list processing
- Maintains original input type
- Handles edge cases (None, empty lists, etc.)

## Error Handling Patterns

### Consistent Error Handling
All modules follow consistent error handling patterns:

```python
def safe_operation(input_param):
    try:
        # Validate input
        if not input_param:
            raise ValueError("Input parameter is required")
        
        # Perform operation
        result = risky_operation(input_param)
        
        return result
        
    except SpecificError as e:
        # Handle specific errors with context
        raise SpecificError(f"Operation failed for {input_param}: {e}")
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error in safe_operation: {e}")
        raise
```

### Type Validation
Input validation with helpful error messages:

```python
def validate_image_input(image_input):
    """Ensure input is valid for image operations."""
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
    elif hasattr(image_input, 'mode'):  # PIL Image
        # Valid PIL Image
        pass
    else:
        raise TypeError(
            "Input must be a file path (str) or PIL Image object, "
            f"got {type(image_input)}"
        )
```

## Threading and Performance

### Thread-Safe Operations
The `@listify` decorator provides built-in threading:

```python
@listify(threading=True)
def cpu_intensive_task(item):
    # This will run in parallel for list inputs
    return complex_computation(item)

# Automatically uses ThreadPoolExecutor
results = cpu_intensive_task(large_list)
```

### Memory Management
Best practices for handling large datasets:

```python
def process_large_image_batch(image_paths):
    """Process images while managing memory usage."""
    for i, path in enumerate(image_paths):
        try:
            # Load and process
            image = load_image(path)
            processed = process_image(image)
            save_image(processed, get_output_path(path))
            
            # Explicit cleanup
            del image, processed
            
            # Progress reporting
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(image_paths)} images")
                
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            continue
```

## Extension Patterns

### Adding New Utility Modules

1. **Create Module Structure**:
```python
# new_utils/_new_utils.py
from pythontk.core_utils import HelpMixin

class NewUtils(HelpMixin):
    @staticmethod
    def new_function(param):
        """New utility function."""
        return processed_param
```

2. **Create Module Init**:
```python
# new_utils/__init__.py
from ._new_utils import NewUtils
```

3. **Register in Main Package**:
The dynamic loading system will automatically discover the new module.

### Custom Decorators
Following the package patterns:

```python
def retry(max_attempts=3, delay=1.0):
    """Decorator for retrying failed operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                        continue
                    
            raise last_exception
        return wrapper
    return decorator
```

## Testing Patterns

### Unit Testing Structure
```python
import unittest
import pythontk as ptk

class TestFileUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, 'test.txt')
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_file_operations(self):
        # Test file creation
        ptk.write_file(self.test_file, "test content")
        self.assertTrue(ptk.is_valid(self.test_file, 'file'))
        
        # Test file reading
        content = ptk.get_file_contents(self.test_file)
        self.assertEqual(content, "test content")
```

### Mock Testing for External Dependencies
```python
from unittest.mock import patch, MagicMock

class TestImageUtils(unittest.TestCase):
    @patch('pythontk.img_utils.Image')
    def test_image_processing(self, mock_image):
        # Mock PIL Image operations
        mock_img = MagicMock()
        mock_image.open.return_value = mock_img
        
        result = ptk.resize_image('test.jpg', 512, 512)
        
        mock_image.open.assert_called_once_with('test.jpg')
        mock_img.resize.assert_called_once_with((512, 512))
```

## Configuration and Settings

### Environment Variables
The package respects common environment variables:

```python
# Video utilities look for FFmpeg in MAYA_SCRIPT_PATH
maya_paths = os.getenv("MAYA_SCRIPT_PATH", "").split(";")

# File operations respect system encoding
default_encoding = os.getenv("PYTHONIOENCODING", "utf-8")
```

### Runtime Configuration
```python
# Enable debug logging
import logging
logging.getLogger('pythontk').setLevel(logging.DEBUG)

# Configure threading behavior
import pythontk.core_utils as core
core.DEFAULT_THREAD_COUNT = 8
```

## Performance Optimization

### Profiling Integration
```python
import cProfile
import pythontk as ptk

def profile_operation():
    # Profile specific operations
    pr = cProfile.Profile()
    pr.enable()
    
    # Your operation here
    results = ptk.batch_process_images(image_list)
    
    pr.disable()
    pr.print_stats(sort='cumulative')
```

### Memory Profiling
```python
from memory_profiler import profile

@profile
def memory_intensive_operation():
    # This will show line-by-line memory usage
    return ptk.process_large_dataset(dataset)
```

## Best Practices

### Function Design
1. **Single Responsibility**: Each function should do one thing well
2. **Type Hints**: Always include proper type annotations
3. **Documentation**: Comprehensive docstrings with examples
4. **Error Handling**: Validate inputs and provide meaningful errors

### API Consistency
1. **Parameter Naming**: Use consistent parameter names across functions
2. **Return Types**: Return types should be predictable and documented
3. **Optional Parameters**: Use sensible defaults for optional parameters

### Example of Well-Designed Function
```python
def process_image_batch(
    image_paths: List[str],
    output_dir: str,
    size: Tuple[int, int] = (1024, 1024),
    quality: int = 95,
    format: str = 'JPEG',
    overwrite: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[str]:
    """
    Process a batch of images with consistent sizing and quality.
    
    Args:
        image_paths: List of input image file paths
        output_dir: Directory for processed images
        size: Target dimensions as (width, height)
        quality: JPEG quality (1-100)
        format: Output format ('JPEG', 'PNG', etc.)
        overwrite: Whether to overwrite existing files
        progress_callback: Optional callback for progress updates
    
    Returns:
        List of output file paths
        
    Raises:
        FileNotFoundError: If input images don't exist
        ValueError: If parameters are invalid
        
    Example:
        >>> paths = ['img1.jpg', 'img2.jpg']
        >>> results = process_image_batch(paths, 'output/', (512, 512))
        >>> print(f"Processed {len(results)} images")
    """
    # Implementation with proper error handling, validation, etc.
```

This comprehensive approach ensures that pythontk maintains high code quality, performance, and usability across all its utility modules.
