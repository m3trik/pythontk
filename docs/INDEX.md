# Documentation Index

Welcome to the pythontk documentation! This comprehensive guide covers everything you need to know about using and contributing to the Python Toolkit.

## 📚 Documentation Structure

### Getting Started
- **[README](README.md)** - Main documentation with installation, features, and usage examples
- **[Quick Reference](QUICKREF.md)** - Cheat sheet for common operations and patterns

### Detailed Guides  
- **[API Reference](API.md)** - Complete function and class documentation
- **[Usage Examples](EXAMPLES.md)** - Comprehensive examples and use cases
- **[Developer Guide](DEVELOPER.md)** - Architecture and development patterns

### Project Information
- **[Changelog](CHANGELOG.md)** - Version history and breaking changes
- **[License](../LICENSE)** - MIT License details

## 🚀 Quick Navigation

### By Use Case
| I want to... | Go to |
|--------------|-------|
| Get started quickly | [README](README.md#quick-start) |
| See code examples | [Examples](EXAMPLES.md) |
| Look up a function | [API Reference](API.md) |
| Find a quick snippet | [Quick Reference](QUICKREF.md) |
| Understand the architecture | [Developer Guide](DEVELOPER.md) |
| See what's new | [Changelog](CHANGELOG.md) |

### By Module
| Module | Description | Key Functions |
|--------|-------------|---------------|
| **core_utils** | Base utilities, decorators | `@cached_property`, `@listify`, mixins |
| **file_utils** | File operations | `get_dir_contents()`, `get_file_contents()` |
| **str_utils** | String processing | `sanitize()`, `camel_to_snake()` |
| **img_utils** | Image processing | `resize_image()`, `pack_channels()` |
| **vid_utils** | Video processing | `extract_frames()`, `get_video_info()` |
| **math_utils** | Math operations | `get_vector_from_two_points()`, `lerp()` |
| **iter_utils** | List/dict filtering | `filter_list()`, `filter_dict()` |

## 📖 Documentation Types

### 🎯 **Quick Reference** - [QUICKREF.md](QUICKREF.md)
*Perfect for: Experienced users who need a quick lookup*
- Concise syntax examples
- Common patterns
- Function signatures
- Quick troubleshooting

### 📘 **Complete Guide** - [README.md](README.md)  
*Perfect for: New users getting started*
- Installation instructions
- Feature overview
- Basic usage examples
- Module descriptions

### 📋 **API Reference** - [API.md](API.md)
*Perfect for: Developers who need detailed specifications*
- Complete function documentation
- Parameter descriptions
- Return types
- Error conditions

### 💡 **Examples** - [EXAMPLES.md](EXAMPLES.md)
*Perfect for: Learning through practical examples*
- Real-world use cases
- Complete code examples
- Best practices
- Advanced patterns

### 🔧 **Developer Guide** - [DEVELOPER.md](DEVELOPER.md)
*Perfect for: Contributors and advanced users*
- Architecture overview
- Extension patterns
- Performance optimization
- Testing strategies

## 🎨 Common Workflows

### Image Processing Workflow
1. [Basic Operations](EXAMPLES.md#basic-image-operations) - Resize, convert, save
2. [Channel Manipulation](EXAMPLES.md#channel-packing) - Pack/unpack channels  
3. [PBR Conversion](EXAMPLES.md#pbr-texture-conversion) - Spec/gloss to metal/rough
4. [Batch Processing](EXAMPLES.md#batch-processing) - Process multiple images

### File Management Workflow  
1. [Directory Scanning](EXAMPLES.md#directory-scanning) - Find files with patterns
2. [Content Processing](EXAMPLES.md#basic-file-operations) - Read/write files
3. [Batch Operations](EXAMPLES.md#combining-operations) - Process multiple files
4. [Path Handling](API.md#format_pathpath-styleforward) - Cross-platform paths

### Data Processing Workflow
1. [List Filtering](EXAMPLES.md#list-filtering) - Filter with wildcards
2. [Dictionary Operations](EXAMPLES.md#dictionary-filtering) - Key/value filtering  
3. [String Processing](EXAMPLES.md#text-sanitization) - Clean and format text
4. [Math Operations](EXAMPLES.md#mathematical-operations) - Vector and decimal math

## 🎓 Learning Path

### Beginner
1. Read [Installation](README.md#installation)
2. Try [Quick Start](README.md#quick-start) examples
3. Explore [Basic Examples](EXAMPLES.md#file-operations)
4. Use [Quick Reference](QUICKREF.md) for syntax

### Intermediate  
1. Study [Module Overview](README.md#module-overview)
2. Practice [Advanced Usage](README.md#advanced-usage) patterns
3. Read [API Reference](API.md) for specific functions
4. Try [Complex Examples](EXAMPLES.md#advanced-patterns)

### Advanced
1. Understand [Architecture](DEVELOPER.md#architecture-overview)
2. Learn [Extension Patterns](DEVELOPER.md#extension-patterns)
3. Study [Performance Tips](DEVELOPER.md#performance-optimization)
4. Contribute improvements

## 📋 Checklists

### Installation Checklist
- [ ] Python 3.7+ installed
- [ ] Run `pip install pythontk`
- [ ] Test basic import: `import pythontk as ptk`
- [ ] Optional: Install PIL for images (`pip install Pillow`)
- [ ] Optional: Install FFmpeg for video processing

### First Use Checklist
- [ ] Read [Quick Start](README.md#quick-start)
- [ ] Try a basic file operation
- [ ] Test list filtering
- [ ] Explore image processing (if PIL available)
- [ ] Check [Examples](EXAMPLES.md) for your use case

### Development Checklist
- [ ] Read [Developer Guide](DEVELOPER.md)
- [ ] Understand module structure
- [ ] Set up testing environment
- [ ] Follow coding patterns
- [ ] Add tests for new features

## 🔍 Search Tips

### Finding Functions
1. **By purpose**: Check module descriptions in [README](README.md#module-overview)
2. **By name**: Use [API Reference](API.md) index
3. **By example**: Browse [Examples](EXAMPLES.md) 
4. **Quick lookup**: Use [Quick Reference](QUICKREF.md) table

### Finding Examples
1. **Basic usage**: [README Quick Start](README.md#quick-start)
2. **Specific modules**: [Examples by module](EXAMPLES.md#file-operations)
3. **Advanced patterns**: [Complex workflows](EXAMPLES.md#advanced-patterns)
4. **Performance**: [Optimization examples](EXAMPLES.md#performance-tips)

## 🤝 Contributing

Interested in contributing? See:
- [Developer Guide](DEVELOPER.md) - Architecture and patterns
- [GitHub Issues](https://github.com/m3trik/pythontk/issues) - Bug reports and features
- [Testing Patterns](DEVELOPER.md#testing-patterns) - How to add tests

## 📞 Support

Need help?
- **Documentation**: You're reading it!
- **Examples**: Check [EXAMPLES.md](EXAMPLES.md)  
- **Issues**: [GitHub Issues](https://github.com/m3trik/pythontk/issues)
- **Quick Help**: [QUICKREF.md](QUICKREF.md)

---

*This documentation is actively maintained. Last updated: August 2024*
