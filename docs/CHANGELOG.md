# Changelog

All notable changes to pythontk will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.28] - 2024-08-22

### Added
- Comprehensive documentation suite with API reference, examples, and developer guide
- Enhanced image processing utilities for PBR workflow support
- Advanced texture conversion functions (spec/gloss to metal/rough)
- Batch texture optimization capabilities
- Video frame extraction utilities with FFmpeg integration
- Dynamic module loading system for better package organization
- Threading support via `@listify` decorator
- Memory-efficient image processing for large batches

### Enhanced
- Improved error handling across all modules
- Better type hints for IDE support
- More robust file operations with encoding detection
- Advanced filtering capabilities for lists and dictionaries
- Cross-platform path handling
- Performance optimizations for image operations

### Fixed
- Memory leaks in large image processing batches
- Unicode handling in file operations
- Path separator issues on different platforms
- Error propagation in threaded operations

## [0.7.27] - 2024-07-15

### Added
- Video processing utilities
- FFmpeg integration for frame extraction
- Enhanced string sanitization options
- Custom character mapping in sanitization
- Case preservation options
- Multiple delimiter support in text extraction

### Enhanced
- File operation robustness
- Directory scanning performance
- String processing capabilities
- Error messages and logging

### Fixed
- File encoding detection issues
- Memory usage in large file operations
- String sanitization edge cases

## [0.7.26] - 2024-06-20

### Added
- Image channel packing utilities
- Alpha channel manipulation
- Transparency handling in textures
- Batch image processing
- Map type detection for textures

### Enhanced
- PIL integration improvements
- Image format support
- Memory management for large images
- Processing pipeline efficiency

### Fixed
- Image mode conversion issues
- Channel order in packed textures
- Memory cleanup in batch operations

## [0.7.25] - 2024-05-18

### Added
- Mathematical utilities for vector operations
- Decimal point manipulation functions
- Value clamping and interpolation
- 3D vector calculations
- Precision handling for mathematical operations

### Enhanced
- Core utility decorators
- Cached property implementation
- Singleton pattern support
- Help system documentation

### Fixed
- Floating point precision issues
- Vector calculation accuracy
- Property caching edge cases

## [0.7.24] - 2024-04-22

### Added
- Advanced list filtering with wildcards
- Dictionary filtering by keys/values
- Nested structure analysis
- Map function support in filtering
- Case-insensitive filtering options

### Enhanced
- Iterator utilities performance
- Memory usage in large list operations
- Filtering accuracy and speed
- Error handling in iteration operations

### Fixed
- Memory leaks in large list processing
- Wildcard matching edge cases
- Nested structure handling

## [0.7.23] - 2024-03-25

### Added
- File utilities with directory scanning
- Recursive directory operations
- Pattern-based file filtering
- Safe directory creation
- Cross-platform path handling

### Enhanced
- File operation safety
- Path normalization
- Directory traversal performance
- Error handling in file operations

### Fixed
- Path separator issues on Windows
- Unicode filename handling
- Directory creation race conditions

## [0.7.22] - 2024-02-20

### Added
- Core utility mixins and base classes
- HelpMixin for auto-documentation
- LoggingMixin for standardized logging
- SingletonMixin for singleton pattern
- Base utility functions

### Enhanced
- Package architecture
- Module organization
- Import system
- Documentation generation

### Fixed
- Import resolution issues
- Module loading performance
- Documentation formatting

## [0.7.21] - 2024-01-15

### Added
- String utilities for text processing
- Text sanitization for filenames
- Case conversion utilities
- Delimiter-based text extraction
- Advanced string formatting

### Enhanced
- Text processing performance
- Unicode support
- String manipulation accuracy
- Memory usage in text operations

### Fixed
- Unicode encoding issues
- Regular expression edge cases
- Memory leaks in string processing

## [0.7.20] - 2023-12-10

### Added
- Initial package structure
- Basic utility framework
- Module loading system
- Core functionality base
- Testing infrastructure

### Enhanced
- Package organization
- Import performance
- Error handling
- Documentation structure

## Development Roadmap

### Planned for v0.8.0
- **Audio Processing**: Audio file manipulation utilities
- **3D Utilities**: Basic 3D math and geometry functions
- **Database Utils**: Simple database operation utilities
- **Network Utils**: HTTP request and URL utilities
- **Config Management**: Configuration file handling
- **Async Support**: Async/await support for I/O operations

### Planned for v0.9.0
- **Plugin System**: Extensible plugin architecture
- **GUI Utilities**: Basic GUI helper functions
- **Data Processing**: Pandas integration utilities
- **Machine Learning**: Basic ML utility functions
- **Cloud Integration**: Cloud storage utilities
- **Performance Profiling**: Built-in profiling tools

### Planned for v1.0.0
- **API Stabilization**: Finalize all public APIs
- **Comprehensive Testing**: 100% test coverage
- **Documentation**: Complete API documentation
- **Performance**: Optimized performance benchmarks
- **Backward Compatibility**: Stable backward compatibility
- **Security**: Security audit and hardening

## Migration Guide

### Upgrading from 0.7.27 to 0.7.28

**New Features Available:**
```python
# New PBR texture conversion
base_color, metallic, roughness = ptk.convert_spec_gloss_to_metal_rough(
    specular_map='spec.jpg',
    glossiness_map='gloss.jpg'
)

# New batch optimization
ptk.batch_optimize_textures('/textures/', max_size=2048)

# New video frame extraction
frames = ptk.extract_frames('video.mp4', 'frames/', step=30)
```

**Breaking Changes:**
- None in this release

**Deprecations:**
- None in this release

### Upgrading from 0.7.26 to 0.7.27

**New Features Available:**
```python
# Enhanced string sanitization
clean = ptk.sanitize("text", char_map={'@': 'at'}, preserve_case=True)

# Video processing
info = ptk.get_video_info('video.mp4')
```

**Breaking Changes:**
- None in this release

## Contributors

- **Ryan Simpson (m3trik)** - Original author and maintainer
- **Community Contributors** - Various bug fixes and improvements

## Support and Feedback

- **Issues**: [GitHub Issues](https://github.com/m3trik/pythontk/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/m3trik/pythontk/discussions)
- **Documentation**: [Repository Wiki](https://github.com/m3trik/pythontk/wiki)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
