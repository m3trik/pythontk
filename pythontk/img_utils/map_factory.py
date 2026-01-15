# !/usr/bin/python
# coding=utf-8
"""Texture Map Factory for PBR workflow preparation - Refactored.

Provides a dynamic, extensible factory for processing and preparing
texture maps for various PBR workflows (Unity, Unreal, glTF, etc.).

Key improvements:
- Strategy pattern for workflow handlers
- Pluggable conversion system
- DRY source resolution
- Declarative workflow configuration
"""
import os
from dataclasses import dataclass, field, asdict, fields
from abc import ABC, abstractmethod
from typing import (
    List,
    Dict,
    Optional,
    Callable,
    Type,
    Any,
    Union,
    Tuple,
    TYPE_CHECKING,
)
from collections import defaultdict
import inspect

try:
    import numpy as np
except ImportError:
    np = None
try:
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageChops
except ImportError:
    Image = None

if TYPE_CHECKING:
    from PIL import Image

# From this package:
from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.img_utils._img_utils import ImgUtils
from pythontk.file_utils._file_utils import FileUtils
from pythontk.iter_utils._iter_utils import IterUtils
from pythontk.str_utils._str_utils import StrUtils
from pythontk.img_utils.map_registry import MapRegistry
import re

# Constants
DEFAULT_EXTENSION = "png"  # Default extension for saved maps
ALPHA_EXTENSION = "png"  # Default extension for maps requiring alpha channel


@dataclass
class MapConversion:
    """Defines a single map conversion operation."""

    target_type: str
    source_types: List[str]
    converter: Callable
    priority: int = 0


class ConversionRegistry:
    """Central registry for all map type conversions.

    This eliminates duplicate conversion logic across methods.
    """

    def __init__(self):
        self._conversions: Dict[str, List[MapConversion]] = defaultdict(list)
        self._registered_classes = set()
        self._pending_plugins = set()

    def add_plugin(self, cls):
        """Register a class to be scanned for conversions later."""
        self._pending_plugins.add(cls)

    def _scan_pending(self):
        """Scan any pending plugins."""
        while self._pending_plugins:
            cls = self._pending_plugins.pop()
            if hasattr(cls, "register_conversions"):
                cls.register_conversions(self)
            else:
                # Fallback for backward compatibility or mixed usage
                self.register_from_class(cls)

    def register(
        self,
        target_type: Union[str, MapConversion],
        source_types: Union[str, List[str]] = None,
        converter: Callable = None,
        priority: int = 0,
    ):
        """Register a new conversion strategy.

        Can be called with a MapConversion object or with individual arguments.
        """
        if isinstance(target_type, MapConversion):
            conversion = target_type
        else:
            if source_types is None or converter is None:
                raise ValueError(
                    "source_types and converter are required when registering by arguments"
                )

            if isinstance(source_types, str):
                source_types = [source_types]

            conversion = MapConversion(
                target_type=target_type,
                source_types=source_types,
                converter=converter,
                priority=priority,
            )

        # Prevent duplicate registrations
        current_list = self._conversions[conversion.target_type]
        for existing in current_list:
            if (
                existing.converter == conversion.converter
                and existing.source_types == conversion.source_types
            ):
                return

        self._conversions[conversion.target_type].append(conversion)
        # Sort by priority (higher first)
        self._conversions[conversion.target_type].sort(
            key=lambda c: c.priority, reverse=True
        )

    def register_from_class(self, cls):
        """Register all decorated conversion methods from a class."""
        if cls in self._registered_classes:
            return

        for name, method in inspect.getmembers(cls):
            if hasattr(method, "_conversion_info"):
                infos = method._conversion_info
                if isinstance(infos, dict):
                    infos = [infos]
                for info in infos:
                    self.register(
                        target_type=info["target_type"],
                        source_types=info["source_types"],
                        converter=method,
                        priority=info["priority"],
                    )

        self._registered_classes.add(cls)

    def get_conversions_for(self, target_type: str) -> List[MapConversion]:
        """Get all conversions that can produce target type."""
        self._scan_pending()
        return self._conversions.get(target_type, [])

    def __getattr__(self, name):
        """Allow property-style access to conversions (e.g. registry.Metallic)."""
        return self.get_conversions_for(name)


# =============================================================================
# Processing Context - Shared state and utilities
# =============================================================================


@dataclass
class TextureProcessor:
    """Shared context and processor for all map operations."""

    inventory: Dict[str, Union[str, "Image.Image"]]
    config: Dict[str, Any]
    output_dir: str
    base_name: str
    ext: Optional[str]
    conversion_registry: ConversionRegistry
    logger: Any = None
    used_maps: set = field(default_factory=set)
    created_files: set = field(default_factory=set)

    def save_map(
        self,
        image: Union[str, Any],
        map_type: str,
        suffix: str = None,
        optimize: bool = None,
        source_images: List[Union[str, Any]] = None,
    ) -> str:
        """Saves and optimizes a map, enforcing mode and naming conventions.

        Args:
            image: PIL Image or path to image.
            map_type: Type of map (e.g., 'Metallic', 'Normal').
            suffix: Optional suffix override. If None, uses map_type.
            optimize: Whether to run optimization (including mode enforcement).
                      If None, uses the value from self.config.
            source_images: List of source images (paths or PIL images) used to create this map.
                           Used for logging comparison stats.

        Returns:
            Path to the saved file.
        """
        # Determine optimization setting
        if optimize is None:
            optimize = self.config.get("optimize", True)

        # Determine suffix
        suffix = suffix or map_type
        suffix = f"_{suffix.lstrip('_')}"

        # Determine extension
        ext = self.ext
        if not ext:
            if isinstance(image, str):
                ext = os.path.splitext(image)[1].lstrip(".")
            else:
                ext = DEFAULT_EXTENSION  # Fallback for generated images

        # Force PNG for maps requiring alpha if source was JPG
        if ext.lower() in ["jpg", "jpeg"] and map_type in [
            "MaskMap",
            "MSAO",
            "ORM",
            "Albedo_Transparency",
        ]:
            ext = ALPHA_EXTENSION

        # Generate output path
        filename = StrUtils.replace_placeholders(
            "{base_name}{suffix}.{ext}",
            base_name=self.base_name,
            suffix=suffix,
            ext=ext,
        )
        output_path = os.path.join(self.output_dir, filename)

        # Ensure output directory exists
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating output directory: {e}")

        # Check for dry run
        if self.config.get("dry_run", False):
            details = []

            # Source Info
            if source_images:
                total_mb = 0
                count = 0
                for src in source_images:
                    if isinstance(src, str) and os.path.exists(src):
                        total_mb += os.path.getsize(src) / (1024 * 1024)
                        count += 1

                if count > 0:
                    details.append(f"Sources: {count} files ({total_mb:.2f} MB)")

            elif isinstance(image, str) and os.path.exists(image):
                try:
                    size_mb = os.path.getsize(image) / (1024 * 1024)
                    from PIL import Image

                    with Image.open(image) as img:
                        details.append(
                            f"Source: {img.size[0]}x{img.size[1]} {img.mode} ({size_mb:.2f} MB)"
                        )
                except Exception:
                    pass

            if hasattr(image, "size") and hasattr(image, "mode"):
                details.append(
                    f"Generated: {image.size[0]}x{image.size[1]} {image.mode}"
                )

            # Target Info
            max_size = self.config.get("max_size")
            if max_size:
                if map_type in MapFactory.packed_grayscale_maps:
                    scale = self.config.get("mask_map_scale", 1.0)
                    max_size = int(max_size * scale)
                details.append(f"Limit: {max_size}px")

            info = (
                f" <hl style='color:rgb(150, 150, 150)'>[{', '.join(details)}]</hl>"
                if details
                else ""
            )
            self.logger.info(
                f"[Dry Run] Would save {map_type} to {output_path}{info}",
                extra={"preset": "highlight"},
            )
            return output_path

        # Check if we can skip (only for file-based inputs)
        force = self.config.get("force", False)
        if (
            not force
            and isinstance(image, str)
            and os.path.exists(output_path)
            and os.path.exists(image)
        ):
            out_mtime = os.path.getmtime(output_path)
            in_mtime = os.path.getmtime(image)
            # Check if output is newer than input
            if out_mtime > in_mtime:
                self.logger.info(f"Skipping {map_type} (up to date)")
                return output_path

        # Determine if we should optimize
        # Legacy support: 'optimize' implies resize
        # New support: 'resize' explicitly controls resizing
        allow_resize = self.config.get("resize", self.config.get("optimize", True))
        should_optimize = optimize and allow_resize

        max_size = self.config.get("max_size")
        if max_size and map_type in MapFactory.packed_grayscale_maps:
            scale = self.config.get("mask_map_scale", 1.0)
            max_size = int(max_size * scale)

        # Smart Copy: If input is a file and no optimization is needed, just copy
        if isinstance(image, str) and os.path.exists(image):
            # Check if format conversion is needed
            _, src_ext = os.path.splitext(image)
            _, dst_ext = os.path.splitext(output_path)
            format_changed = src_ext.lower() != dst_ext.lower()

            # Legacy support: 'convert' implies format conversion
            # New support: 'convert_format' explicitly controls format conversion
            allow_format_conversion = self.config.get(
                "convert_format", self.config.get("convert", True)
            )

            if format_changed and not allow_format_conversion:
                # If format changed but conversion not allowed, revert extension
                output_path = os.path.splitext(output_path)[0] + src_ext
                dst_ext = src_ext
                format_changed = False

            if should_optimize or format_changed:
                try:
                    # Check if optimization is actually needed
                    from PIL import Image

                    with Image.open(image) as img:
                        width, height = img.size
                        # Check resize
                        needs_resize = (
                            allow_resize and max_size and max(width, height) > max_size
                        )

                        # Check mode enforcement
                        needs_mode_change = False
                        map_def = MapRegistry().get(map_type)
                        if map_def:
                            if map_def.mode == "RGB" and img.mode != "RGB":
                                needs_mode_change = True
                            elif map_def.mode == "RGBA" and img.mode != "RGBA":
                                needs_mode_change = True
                            elif map_def.mode == "L" and img.mode == "P":
                                needs_mode_change = True

                        # If we don't need resize or mode change, we can likely skip re-encoding
                        if (
                            not needs_resize
                            and not needs_mode_change
                            and not format_changed
                        ):
                            should_optimize = False
                except Exception:
                    pass  # Fallback to full processing if check fails

            if not should_optimize:
                import shutil

                # Ensure extensions match before copying to avoid invalid files (e.g. jpg -> png)
                _, src_ext = os.path.splitext(image)
                _, dst_ext = os.path.splitext(output_path)

                if src_ext.lower() == dst_ext.lower():
                    # Check if we should rename/copy
                    if not self.config.get("rename", False):
                        # If not renaming, and extensions match, return original path
                        return image

                    if os.path.abspath(image) != os.path.abspath(output_path):
                        try:
                            shutil.copy2(image, output_path)
                        except shutil.SameFileError:
                            pass  # Source and dest are the same file
                    self.created_files.add(output_path)
                    return output_path
                else:
                    # Extensions differ, must re-encode
                    should_optimize = True

        # IN-MEMORY OPTIMIZATION PIPELINE
        # Load image
        img_obj = ImgUtils.ensure_image(image)

        if should_optimize:
            # Determine if resizing is needed
            will_resize = False
            if allow_resize and max_size:
                width, height = img_obj.size
                if max(width, height) > max_size:
                    will_resize = True

            # 1. Depalettize ONLY if resizing
            if will_resize:
                img_obj = ImgUtils.depalettize_image(img_obj)

            # 2. Enforce Mode
            map_def = MapRegistry().get(map_type)
            if map_def:
                img_obj = ImgUtils.enforce_mode(img_obj, map_def.mode)

            # 3. Resize
            if will_resize:
                from PIL import Image

                img_obj.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # 4. Save with optimization
            ImgUtils.save_image(img_obj, output_path, optimize=True)

        else:
            # Just save (re-encode) without resizing/mode change
            ImgUtils.save_image(img_obj, output_path, optimize=True)

        self.created_files.add(output_path)
        return output_path

    def resolve_map(
        self, *preferred_types: str, allow_conversion: bool = True
    ) -> Optional[Union[str, "Image.Image"]]:
        """Intelligently resolve a map from inventory with fallback conversions.

        This is the DRY replacement for repeated "get X or convert from Y" logic.

        Args:
            *preferred_types: Ordered list of preferred map types
            allow_conversion: Whether to attempt conversions if direct match fails

        Returns:
            Path to resolved map, PIL Image, or None
        """
        # 1. Try direct matches first (in priority order)
        for map_type in preferred_types:
            if map_type in self.inventory:
                return self.inventory[map_type]

        # 2. Try conversions if allowed
        if allow_conversion and self.config.get("convert", True):
            for target_type in preferred_types:
                conversions = self.conversion_registry.get_conversions_for(target_type)
                for conversion in conversions:
                    # Check if we have all required source types
                    if all(self.inventory.get(src) for src in conversion.source_types):
                        try:
                            result = conversion.converter(self.inventory, self)
                            # Cache the result in inventory
                            self.inventory[target_type] = result
                            return result
                        except Exception as e:
                            if self.logger:
                                self.logger.error(
                                    f"Error converting to {target_type}: {str(e)}"
                                )

        # 3. Try registry fallbacks (Safe Substitutes)
        # Only use fallbacks that DON'T have a registered conversion (otherwise step 2 would have caught them)
        if self.config.get("use_input_fallbacks", True):
            for map_type in preferred_types:
                map_def = MapRegistry().get(map_type)
                if map_def:
                    for fb in map_def.input_fallbacks:
                        if fb in self.inventory:
                            # Check if a conversion exists for this fallback
                            # We can check if any conversion for map_type uses fb as source
                            conversions = self.conversion_registry.get_conversions_for(
                                map_type
                            )
                            has_conversion = any(
                                fb in c.source_types for c in conversions
                            )

                            if not has_conversion:
                                return self.inventory[fb]

        return None

    def mark_used(self, *map_types: str):
        """Mark map types as consumed."""
        self.used_maps.update(map_types)

    def convert_specular_to_metallic(
        self, specular_path: Union[str, "Image.Image"]
    ) -> "Image.Image":
        if not specular_path:
            raise ValueError(
                "Cannot convert Specular to Metallic: Input map is missing"
            )
        metallic_img = MapFactory.create_metallic_from_spec(specular_path)
        if self.logger:
            self.logger.info(
                "Created metallic from specular", extra={"preset": "highlight"}
            )
        return metallic_img

    def convert_smoothness_to_roughness(
        self, smoothness_path: Union[str, "Image.Image"]
    ) -> "Image.Image":
        if not smoothness_path:
            raise ValueError(
                "Cannot convert Smoothness to Roughness: Input map is missing"
            )
        roughness_img = MapFactory.convert_smoothness_to_roughness(
            smoothness_path, self.output_dir, save=False
        )
        if self.logger:
            self.logger.info(
                "Converted smoothness to roughness", extra={"preset": "highlight"}
            )
        return roughness_img

    def convert_roughness_to_smoothness(
        self, roughness_path: Union[str, "Image.Image"]
    ) -> "Image.Image":
        if not roughness_path:
            raise ValueError(
                "Cannot convert Roughness to Smoothness: Input map is missing"
            )
        smooth_img = MapFactory.convert_roughness_to_smoothness(
            roughness_path, self.output_dir, save=False
        )
        if self.logger:
            self.logger.info(
                "Converted roughness to smoothness", extra={"preset": "highlight"}
            )
        return smooth_img

    def convert_specular_to_roughness(
        self, specular_path: Union[str, "Image.Image"]
    ) -> "Image.Image":
        if not specular_path:
            raise ValueError(
                "Cannot convert Specular to Roughness: Input map is missing"
            )
        rough_img = MapFactory.create_roughness_from_spec(specular_path)
        if self.logger:
            self.logger.info(
                "Created roughness from specular", extra={"preset": "highlight"}
            )
        return rough_img

    def convert_dx_to_gl(self, dx_path: Union[str, "Image.Image"]) -> "Image.Image":
        if not dx_path:
            raise ValueError(
                "Cannot convert DirectX Normal to OpenGL: Input map is missing"
            )
        gl_img = MapFactory.convert_normal_map_format(
            dx_path, target_format="opengl", save=False
        )
        if self.logger:
            self.logger.info(
                "Converted DirectX normal to OpenGL", extra={"preset": "highlight"}
            )
        return gl_img

    def convert_gl_to_dx(self, gl_path: Union[str, "Image.Image"]) -> "Image.Image":
        if not gl_path:
            raise ValueError(
                "Cannot convert OpenGL Normal to DirectX: Input map is missing"
            )
        dx_img = MapFactory.convert_normal_map_format(
            gl_path, target_format="directx", save=False
        )
        if self.logger:
            self.logger.info(
                "Converted OpenGL normal to DirectX", extra={"preset": "highlight"}
            )
        return dx_img

    def convert_bump_to_normal(
        self, bump_path: Union[str, "Image.Image"]
    ) -> "Image.Image":
        if not bump_path:
            raise ValueError("Cannot convert Bump to Normal: Input map is missing")
        normal_img = MapFactory.convert_bump_to_normal(
            bump_path,
            output_format=self.config.get("normal_type", "opengl").lower(),
            save=False,
        )
        if self.logger:
            self.logger.info(
                "Generated normal from bump/height", extra={"preset": "highlight"}
            )
        return normal_img

    def extract_gloss_from_spec(
        self, specular_path: Union[str, "Image.Image"]
    ) -> "Image.Image":
        if not specular_path:
            raise ValueError(
                "Cannot extract Glossiness from Specular: Input map is missing"
            )
        gloss_img = MapFactory.extract_gloss_from_spec(specular_path)
        if not gloss_img:
            raise ValueError("Could not extract gloss from specular map")

        if self.logger:
            self.logger.info(
                "Extracted glossiness from specular", extra={"preset": "highlight"}
            )
        return gloss_img

    def copy_map(
        self, source_path: Union[str, "Image.Image"], target_type: str
    ) -> Union[str, "Image.Image"]:
        """Simple copy/rename for compatible maps (e.g. Smoothness -> Glossiness)."""
        if self.logger:
            self.logger.info(
                f"Created {target_type} from source map", extra={"preset": "highlight"}
            )
        return source_path

    def unpack_metallic_smoothness(
        self, source_path: Union[str, "Image.Image"]
    ) -> None:
        """Helper to unpack and cache results."""
        # Return cached if available
        if "Metallic" in self.inventory and "Smoothness" in self.inventory:
            return

        metallic_img, smoothness_img = MapFactory.unpack_metallic_smoothness(
            source_path, self.output_dir, optimize=False, save=False
        )

        self.inventory["Metallic"] = metallic_img
        self.inventory["Smoothness"] = smoothness_img
        if self.logger:
            self.logger.info(
                "Unpacked Metallic and Smoothness from packed map",
                extra={"preset": "highlight"},
            )

    def get_metallic_from_packed(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_metallic_smoothness(source_path)
        return self.inventory["Metallic"]

    def get_smoothness_from_packed(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_metallic_smoothness(source_path)
        return self.inventory["Smoothness"]

    def get_roughness_from_packed(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_metallic_smoothness(source_path)

        if not self.inventory.get("Smoothness"):
            return None

        # Convert S -> R
        return self.convert_smoothness_to_roughness(self.inventory["Smoothness"])

    def unpack_msao(self, source_path: Union[str, "Image.Image"]) -> None:
        """Helper to unpack MSAO and cache results."""
        if (
            "Metallic" in self.inventory
            and "AO" in self.inventory
            and "Smoothness" in self.inventory
        ):
            return

        metallic_img, ao_img, smoothness_img = MapFactory.unpack_msao_texture(
            source_path, self.output_dir, optimize=False, save=False
        )

        self.inventory["Metallic"] = metallic_img
        self.inventory["AO"] = ao_img
        self.inventory["Ambient_Occlusion"] = self.inventory["AO"]
        self.inventory["Smoothness"] = smoothness_img
        if self.logger:
            self.logger.info(
                "Unpacked Metallic, AO, and Smoothness from MSAO map",
                extra={"preset": "highlight"},
            )

    def get_metallic_from_msao(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_msao(source_path)
        return self.inventory["Metallic"]

    def get_smoothness_from_msao(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_msao(source_path)
        return self.inventory["Smoothness"]

    def get_roughness_from_msao(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_msao(source_path)

        if not self.inventory.get("Smoothness"):
            return None

        return self.convert_smoothness_to_roughness(self.inventory["Smoothness"])

    def get_ao_from_msao(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_msao(source_path)
        return self.inventory["AO"]

    def unpack_orm(self, source_path: Union[str, "Image.Image"]) -> None:
        """Helper to unpack ORM and cache results."""
        if (
            "AO" in self.inventory
            and "Roughness" in self.inventory
            and "Metallic" in self.inventory
        ):
            return

        ao_img, roughness_img, metallic_img = MapFactory.unpack_orm_texture(
            source_path, self.output_dir, optimize=False, save=False
        )

        self.inventory["AO"] = ao_img
        self.inventory["Ambient_Occlusion"] = self.inventory["AO"]
        self.inventory["Roughness"] = roughness_img
        self.inventory["Metallic"] = metallic_img
        if self.logger:
            self.logger.info(
                "Unpacked AO, Roughness, and Metallic from ORM map",
                extra={"preset": "highlight"},
            )

    def get_ao_from_orm(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_orm(source_path)
        return self.inventory["AO"]

    def get_roughness_from_orm(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_orm(source_path)
        return self.inventory["Roughness"]

    def get_smoothness_from_orm(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_orm(source_path)
        # Convert R -> S
        return self.convert_roughness_to_smoothness(self.inventory["Roughness"])

    def get_metallic_from_orm(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_orm(source_path)
        return self.inventory["Metallic"]

    def unpack_albedo_transparency(
        self, source_path: Union[str, "Image.Image"]
    ) -> None:
        """Helper to unpack Albedo+Transparency and cache results."""
        if "Base_Color" in self.inventory and "Opacity" in self.inventory:
            return

        base_color_img, opacity_img = MapFactory.unpack_albedo_transparency(
            source_path, self.output_dir, optimize=False, save=False
        )

        self.inventory["Base_Color"] = base_color_img
        self.inventory["Opacity"] = opacity_img
        if self.logger:
            self.logger.info(
                "Unpacked Base Color and Opacity from Albedo+Transparency map",
                extra={"preset": "highlight"},
            )

    def get_base_color_from_albedo_transparency(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_albedo_transparency(source_path)
        return self.inventory["Base_Color"]

    def get_opacity_from_albedo_transparency(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_albedo_transparency(source_path)
        return self.inventory["Opacity"]

    def create_orm_map(
        self, inventory: Dict[str, Union[str, "Image.Image"]]
    ) -> "Image.Image":
        """Create ORM map from components."""
        # Resolve required components
        ao = self.resolve_map("Ambient_Occlusion", "AO", allow_conversion=False)
        roughness = self.resolve_map(
            "Roughness", "Smoothness", "Glossiness", allow_conversion=True
        )
        metallic = self.resolve_map("Metallic", "Specular", allow_conversion=True)

        if not (ao or roughness or metallic):
            raise ValueError("Missing components for ORM map")

        orm_img = ImgUtils.pack_channels(
            channel_files={"R": ao, "G": roughness, "B": metallic},
            output_path=None,
            fill_values={"R": 255} if not ao else None,
        )
        if self.logger:
            self.logger.info(
                "Created ORM map from components", extra={"preset": "highlight"}
            )
        return orm_img

    def create_mask_map(
        self, inventory: Dict[str, Union[str, "Image.Image"]]
    ) -> "Image.Image":
        """Create Mask Map (MSAO) from components."""
        # Resolve required components
        metallic = self.resolve_map("Metallic", "Specular", allow_conversion=True)
        ao = self.resolve_map("Ambient_Occlusion", "AO", allow_conversion=False) or None
        detail = (
            self.resolve_map("Detail_Mask", "Detail", allow_conversion=False) or None
        )

        # Get smoothness with inversion tracking
        smoothness = None
        invert = False

        if "Smoothness" in inventory:
            smoothness = inventory["Smoothness"]
        elif "Glossiness" in inventory:
            smoothness = inventory["Glossiness"]
        elif "Roughness" in inventory:
            smoothness = inventory["Roughness"]
            invert = True
        else:
            smoothness = None

        if not metallic and not ao and not smoothness:
            raise ValueError(
                "Missing components for Mask Map (need at least Metallic, AO, or Smoothness)"
            )

        mask_map = MapFactory.pack_msao_texture(
            metallic_map_path=metallic,
            ao_map_path=ao,
            alpha_map_path=smoothness,
            detail_map_path=detail,
            output_dir=self.output_dir,
            suffix="_MaskMap",
            invert_alpha=invert,
            save=False,
        )
        if self.logger:
            self.logger.info(
                "Created Mask Map from components", extra={"preset": "highlight"}
            )
        return mask_map

    def create_metallic_smoothness_map(
        self, inventory: Dict[str, Union[str, "Image.Image"]]
    ) -> "Image.Image":
        """Create Metallic-Smoothness map from components."""
        metallic = self.resolve_map("Metallic", "Specular", allow_conversion=True)

        smoothness = None
        invert = False

        if "Smoothness" in inventory:
            smoothness = inventory["Smoothness"]
        elif "Glossiness" in inventory:
            smoothness = inventory["Glossiness"]
        elif "Roughness" in inventory:
            smoothness = inventory["Roughness"]
            invert = True

        if not metallic or not smoothness:
            raise ValueError("Missing components for Metallic-Smoothness map")

        ms_map = MapFactory.pack_smoothness_into_metallic(
            metallic_map_path=metallic,
            alpha_map_path=smoothness,
            output_dir=self.output_dir,
            suffix="_MetallicSmoothness",
            invert_alpha=invert,
            save=False,
        )
        if self.logger:
            self.logger.info(
                "Packed smoothness into metallic", extra={"preset": "highlight"}
            )
        return ms_map


# =============================================================================
# Workflow Handlers - Strategy Pattern
# =============================================================================


class WorkflowHandler(ABC):
    """Abstract base for workflow-specific map processing."""

    @abstractmethod
    def can_handle(self, context: TextureProcessor) -> bool:
        """Check if this handler should process the workflow."""
        pass

    @abstractmethod
    def process(self, context: TextureProcessor) -> Optional[str]:
        """Process and return the output map path."""
        pass

    @abstractmethod
    def get_consumed_types(self) -> List[str]:
        """Return list of map types this handler consumes."""
        pass

    def is_explicitly_requested(self, context: TextureProcessor, map_type: str) -> bool:
        """Check if a map type is explicitly requested in the config.

        Checks for:
        1. Exact key match (e.g. "orm_map")
        2. Lowercase key match (e.g. "orm")
        """
        key = map_type.lower()
        key_map = f"{key}_map"

        return context.config.get(key, False) or context.config.get(key_map, False)


class ORMMapHandler(WorkflowHandler):
    """Handles Unreal Engine / glTF ORM packing."""

    def can_handle(self, context: TextureProcessor) -> bool:
        # Legacy support: 'convert' implies packing
        # New support: 'pack' explicitly controls packing
        if not context.config.get("pack", context.config.get("convert", True)):
            return False

        # Explicitly disabled?
        if context.config.get("orm_map") is False:
            return False

        # Explicit Logic: Config requests ORM
        if self.is_explicitly_requested(context, "ORM"):
            return True

        # Implicit Logic: Disabled to prevent greedy generation
        return False

    def process(self, context: TextureProcessor) -> Optional[str]:
        # Resolve required components with smart fallbacks
        ao = context.resolve_map("Ambient_Occlusion", allow_conversion=False)
        roughness = context.resolve_map("Roughness", allow_conversion=True)
        metallic = context.resolve_map("Metallic", allow_conversion=True)

        if not ao:
            if context.logger:
                context.logger.warning("No AO map, using white for ORM red channel")

        if not roughness:
            if not context.config.get("force_packed_maps", False):
                if context.logger:
                    context.logger.warning("No roughness map for ORM green channel")
                return None
            if context.logger:
                context.logger.warning(
                    "No roughness map for ORM green channel, using black (forced)"
                )

        if not metallic:
            if not context.config.get("force_packed_maps", False):
                if context.logger:
                    context.logger.warning("No metallic map for ORM blue channel")
                return None
            if context.logger:
                context.logger.warning(
                    "No metallic map for ORM blue channel, using black (forced)"
                )

        try:
            fill_values = {}
            if not ao:
                fill_values["R"] = 255
            if not roughness:
                fill_values["G"] = 0  # Default to smooth? Or rough? Using 0 (Black)
            if not metallic:
                fill_values["B"] = 0  # Default to non-metal

            orm_map = ImgUtils.pack_channels(
                channel_files={"R": ao, "G": roughness, "B": metallic},
                output_path=None,
                fill_values=fill_values if fill_values else None,
                optimize=False,
            )
            if context.logger:
                context.logger.info(
                    "Created Unreal/glTF ORM map", extra={"preset": "highlight"}
                )
            sources = [img for img in [ao, roughness, metallic] if img]
            return context.save_map(orm_map, "ORM", source_images=sources)
        except Exception as e:
            if context.logger:
                context.logger.error(f"Error creating ORM map: {str(e)}")
            return None

    def get_consumed_types(self) -> List[str]:
        return [
            "Metallic",
            "Roughness",
            "Smoothness",
            "Glossiness",
            "Ambient_Occlusion",
            "AO",
            "Specular",
            "ORM",
        ]


class MaskMapHandler(WorkflowHandler):
    """Handles Unity HDRP Mask Map (MSAO)."""

    def can_handle(self, context: TextureProcessor) -> bool:
        # Legacy support: 'convert' implies packing
        # New support: 'pack' explicitly controls packing
        if not context.config.get("pack", context.config.get("convert", True)):
            return False

        # Explicit Logic: Config requests Mask Map
        if self.is_explicitly_requested(
            context, "Mask"
        ) or self.is_explicitly_requested(context, "MSAO"):
            return True

        # Implicit Logic: Disabled to prevent greedy generation
        return False

    def process(self, context: TextureProcessor) -> Optional[str]:
        if "MSAO" in context.inventory:
            return context.save_map(context.inventory["MSAO"], "MSAO")

        metallic = context.resolve_map("Metallic", "Specular", allow_conversion=True)
        ao = context.resolve_map("Ambient_Occlusion", allow_conversion=False)

        # Get smoothness with inversion tracking
        smoothness = None
        invert = False

        # Try to resolve smoothness/roughness explicitly
        smoothness_map = context.resolve_map(
            "Smoothness", "Glossiness", allow_conversion=False
        )
        if smoothness_map:
            smoothness = smoothness_map
        else:
            roughness_map = context.resolve_map(
                "Roughness", "Specular", allow_conversion=True
            )
            if roughness_map:
                smoothness = roughness_map
                invert = True
            else:
                smoothness = metallic

        # Ensure we have at least one component
        if not any([metallic, ao, smoothness]):
            return None

        # Special case: If only AO is present, we proceed but log a message.
        if ao and not metallic and not smoothness:
            if not context.config.get("force_packed_maps", False):
                return None
            if context.logger:
                context.logger.info(
                    "Only AO map present, generating Mask Map with default Metallic/Smoothness"
                )

        if not metallic:
            if context.logger:
                context.logger.info("No metallic map for Mask Map, using black")

        if not ao:
            if context.logger:
                context.logger.info("No AO map, using white for Mask Map green channel")

        try:
            # Use pack_channels directly to allow missing AO (defaults to white)
            # Pass output_path=None to get the PIL Image object back instead of saving to disk
            mask_map_image = ImgUtils.pack_channels(
                channel_files={
                    "R": metallic,
                    "G": ao,
                    "B": None,
                    "A": smoothness,
                },
                channels=["R", "G", "B", "A"],
                out_mode="RGBA",
                fill_values={"R": 0, "G": 255, "B": 0, "A": 0 if invert else 255},
                output_path=None,  # Return Image object
            )

            if invert:
                # Invert the alpha channel after packing if needed
                mask_map_image = ImgUtils.invert_channels(mask_map_image, "A")

            if context.logger:
                context.logger.info(
                    "Created Unity HDRP Mask Map", extra={"preset": "highlight"}
                )
            # Pass the PIL Image directly to save_map, which handles optimization and saving
            sources = [img for img in [metallic, ao, smoothness] if img]
            return context.save_map(mask_map_image, "MSAO", source_images=sources)
        except Exception as e:
            if context.logger:
                context.logger.error(f"Error creating Mask Map: {str(e)}")
            return None

    def get_consumed_types(self) -> List[str]:
        return [
            "Metallic",
            "Metallic_Smoothness",
            "MSAO",
            "Roughness",
            "Smoothness",
            "Glossiness",
            "Ambient_Occlusion",
            "Specular",
        ]


class MetallicSmoothnessHandler(WorkflowHandler):
    """Handles packed Metallic+Smoothness."""

    def can_handle(self, context: TextureProcessor) -> bool:
        # Legacy support: 'convert' implies packing
        # New support: 'pack' explicitly controls packing
        if not context.config.get("pack", context.config.get("convert", True)):
            return False

        # Explicitly disabled?
        if context.config.get("metallic_smoothness") is False:
            return False

        # Explicit Logic: Config requests Metallic Smoothness
        if self.is_explicitly_requested(context, "Metallic_Smoothness"):
            return True

        # Fallback Logic: Check if we are a fallback for any failed requested map
        registry = MapRegistry()
        for map_name, map_def in registry._maps.items():
            if self.is_explicitly_requested(context, map_name):
                if map_name not in context.used_maps:
                    if "Metallic_Smoothness" in map_def.output_fallbacks:
                        return True

        # Implicit Logic: Disabled to prevent greedy generation
        return False

    def process(self, context: TextureProcessor) -> Optional[str]:
        if "Metallic_Smoothness" in context.inventory:
            return context.save_map(
                context.inventory["Metallic_Smoothness"], "Metallic_Smoothness"
            )

        metallic = context.resolve_map("Metallic", "Specular", allow_conversion=True)
        if not metallic:
            return None

        # Get smoothness/roughness with inversion tracking
        alpha_map = None
        invert = False
        smoothness = context.resolve_map(
            "Smoothness", "Glossiness", allow_conversion=False
        )
        if smoothness:
            alpha_map = smoothness
        else:
            roughness = context.resolve_map(
                "Roughness", "Specular", allow_conversion=True
            )
            if roughness:
                alpha_map = roughness
                invert = True

        if not alpha_map:
            return context.save_map(metallic, "Metallic", source_images=[metallic])

        try:
            combined = MapFactory.pack_smoothness_into_metallic(
                metallic, alpha_map, invert_alpha=invert, save=False
            )
            if context.logger:
                context.logger.info(
                    "Packed smoothness into metallic", extra={"preset": "highlight"}
                )
            sources = [img for img in [metallic, alpha_map] if img]
            return context.save_map(
                combined, "Metallic_Smoothness", source_images=sources
            )
        except Exception as e:
            if context.logger:
                context.logger.error(f"Error packing metallic/smoothness: {str(e)}")
            return context.save_map(metallic, "Metallic")

    def get_consumed_types(self) -> List[str]:
        return [
            "Metallic",
            "Metallic_Smoothness",
            "Roughness",
            "Smoothness",
            "Glossiness",
            "Specular",
        ]


class SeparateMetallicRoughnessHandler(WorkflowHandler):
    """Handles separate metallic and roughness maps."""

    def can_handle(self, context: TextureProcessor) -> bool:
        # This is the default/fallback handler
        return True

    def process(self, context: TextureProcessor) -> List[str]:
        """Returns list since this produces multiple maps."""
        output_maps = []

        if "Metallic" not in context.used_maps:
            metallic = context.resolve_map(
                "Metallic", "Specular", allow_conversion=True
            )
            if metallic:
                output_maps.append(
                    context.save_map(metallic, "Metallic", source_images=[metallic])
                )
                context.mark_used("Metallic", "Specular")

        if "Roughness" not in context.used_maps:
            roughness = context.resolve_map(
                "Roughness",
                "Smoothness",
                "Glossiness",
                "Specular",
                allow_conversion=True,
            )
            if roughness:
                output_maps.append(
                    context.save_map(roughness, "Roughness", source_images=[roughness])
                )
                context.mark_used("Roughness", "Smoothness", "Glossiness")

        return output_maps

    def get_consumed_types(self) -> List[str]:
        return ["Metallic", "Roughness", "Smoothness", "Glossiness", "Specular"]


class BaseColorHandler(WorkflowHandler):
    """Handles base color / albedo with optional packing."""

    def can_handle(self, context: TextureProcessor) -> bool:
        return True  # Always processes if base color exists

    def process(self, context: TextureProcessor) -> Optional[str]:
        # Check if we have a pre-existing Albedo_Transparency map
        if "Albedo_Transparency" in context.inventory:
            if context.logger:
                context.logger.info(
                    "Processing existing Albedo_Transparency map",
                    extra={"preset": "highlight"},
                )
            context.mark_used("Albedo_Transparency")
            return context.save_map(
                context.inventory["Albedo_Transparency"], "Albedo_Transparency"
            )

        # Resolve base color
        base_color = context.resolve_map("Base_Color", allow_conversion=False)
        if not base_color:
            return None

        # Check if we should create Albedo_Transparency
        # Only if explicitly requested AND we have opacity data (separate or packed)
        create_albedo_transparency = False
        opacity = None

        if self.is_explicitly_requested(context, "Albedo_Transparency"):
            # Check for separate opacity map
            opacity = context.resolve_map("Opacity", allow_conversion=False)
            if opacity:
                create_albedo_transparency = True
            else:
                # Check if base color has alpha channel
                try:
                    img = ImgUtils.ensure_image(base_color)
                    if "A" in img.getbands():
                        # Check if alpha is not fully opaque
                        # This might be expensive for large images, but necessary
                        # Optimization: Just assume if A exists, it's intended?
                        # Or check extrema?
                        extrema = img.getextrema()
                        if len(extrema) == 4:  # RGBA
                            alpha_min, alpha_max = extrema[3]
                            if alpha_min < 255:
                                create_albedo_transparency = True
                except Exception:
                    pass

        if create_albedo_transparency:
            if context.logger:
                context.logger.info(
                    "Processing albedo with transparency", extra={"preset": "highlight"}
                )
            if opacity:
                try:
                    combined = MapFactory.pack_transparency_into_albedo(
                        base_color, opacity, save=False
                    )
                    if context.logger:
                        context.logger.info(
                            "Packed transparency into albedo",
                            extra={"preset": "highlight"},
                        )
                    context.mark_used(
                        "Base_Color",
                        "Diffuse",
                        "Opacity",
                        "Transparency",
                        "Albedo_Transparency",
                    )
                    sources = [img for img in [base_color, opacity] if img]
                    return context.save_map(
                        combined, "Albedo_Transparency", source_images=sources
                    )
                except Exception as e:
                    if context.logger:
                        context.logger.error(
                            f"Error packing transparency: {str(e)}",
                            extra={"preset": "error"},
                        )
            else:
                # Base color already has alpha, just save it as Albedo_Transparency
                # Practical Lens: We only rename Base_Color to Albedo_Transparency if it
                # actually contains transparency data. Otherwise, we keep it as Base_Color
                # to avoid misleading filenames.
                context.mark_used("Base_Color", "Diffuse", "Albedo_Transparency")
                return context.save_map(
                    base_color, "Albedo_Transparency", source_images=[base_color]
                )

        # Fallback to standard Base Color processing
        if context.logger:
            context.logger.info("Processing base color", extra={"preset": "highlight"})

        # Clean up base color if requested
        if context.config.get("cleanup_base_color", False):
            metallic = context.resolve_map("Metallic", allow_conversion=False)
            if metallic:
                try:
                    base_img = ImgUtils.ensure_image(base_color)
                    metallic_img = ImgUtils.ensure_image(metallic)
                    cleaned = MapFactory.convert_base_color_to_albedo(
                        base_img, metallic_img
                    )
                    if context.logger:
                        context.logger.info(
                            "Cleaned base color to true albedo",
                            extra={"preset": "highlight"},
                        )
                    context.mark_used("Base_Color", "Diffuse")
                    sources = [img for img in [base_color, metallic] if img]
                    return context.save_map(
                        cleaned, "Base_Color", source_images=sources
                    )
                except Exception as e:
                    if context.logger:
                        context.logger.error(
                            f"Error cleaning base color: {str(e)}",
                            extra={"preset": "error"},
                        )

        context.mark_used("Base_Color", "Diffuse")
        return context.save_map(base_color, "Base_Color", source_images=[base_color])

    def get_consumed_types(self) -> List[str]:
        return [
            "Base_Color",
            "Diffuse",
            "Opacity",
            "Transparency",
            "Albedo_Transparency",
        ]


class NormalMapHandler(WorkflowHandler):
    """Handles normal map format conversion."""

    def can_handle(self, context: TextureProcessor) -> bool:
        return True  # Always processes if normal exists

    def process(self, context: TextureProcessor) -> Optional[str]:
        target_format = context.config.get("normal_type", "OpenGL")
        target_key = f"Normal_{target_format}"

        # 1. Try exact match (e.g. Normal_OpenGL)
        if target_key in context.inventory:
            if context.logger:
                context.logger.info(
                    f"Processing existing {target_format} normal map",
                    extra={"preset": "highlight"},
                )
            context.mark_used(target_key)
            return context.save_map(
                context.inventory[target_key],
                target_key,
                source_images=[context.inventory[target_key]],
            )

        # 2. Try opposite match (e.g. Normal_DirectX) -> Convert
        opposite_key = f"Normal_{self._opposite(target_format)}"
        if opposite_key in context.inventory:
            # Check if conversion is allowed
            # Legacy support: 'convert' implies type conversion
            # New support: 'convert_type' explicitly controls type conversion
            allow_conversion = context.config.get(
                "convert_type", context.config.get("convert", True)
            )

            if allow_conversion:
                # Let resolve_map handle the conversion via registry
                normal = context.resolve_map(target_key, allow_conversion=True)
                if normal:
                    if context.logger:
                        context.logger.info(
                            f"Converted {self._opposite(target_format)} to {target_format}",
                            extra={"preset": "highlight"},
                        )
                    context.mark_used(opposite_key)
                    return context.save_map(
                        normal,
                        target_key,
                        source_images=[context.inventory[opposite_key]],
                    )
            else:
                # If conversion not allowed, just save the opposite map as is (or skip?)
                # Usually we want to save it even if wrong format if conversion is disabled
                if context.logger:
                    context.logger.info(
                        f"Skipping conversion of {self._opposite(target_format)} to {target_format} (convert_type=False)",
                        extra={"preset": "highlight"},
                    )
                context.mark_used(opposite_key)
                return context.save_map(
                    context.inventory[opposite_key],
                    opposite_key,
                    source_images=[context.inventory[opposite_key]],
                )

        # 3. Try generic Normal
        if "Normal" in context.inventory:
            normal_map = context.inventory["Normal"]

            # Attempt to detect format
            detected_format = MapFactory.detect_normal_map_format(normal_map)

            if detected_format:
                if context.logger:
                    context.logger.info(
                        f"Detected {detected_format} format from generic normal map",
                        extra={"preset": "highlight"},
                    )

                # If detected format matches target, save as target
                if detected_format == target_format:
                    context.mark_used("Normal")
                    return context.save_map(
                        normal_map, target_key, source_images=[normal_map]
                    )

                # If detected format is opposite, convert
                else:
                    # Check if conversion is allowed
                    allow_conversion = context.config.get(
                        "convert_type", context.config.get("convert", True)
                    )

                    if allow_conversion:
                        if context.logger:
                            context.logger.info(
                                f"Converting detected {detected_format} to {target_format}",
                                extra={"preset": "highlight"},
                            )
                        # We can use the registry converters if we temporarily treat it as the detected type
                        # Or just call MapFactory directly
                        if detected_format == "DirectX":  # Target is OpenGL
                            converted = MapFactory.convert_normal_map_format(
                                normal_map, target_format="opengl", save=False
                            )
                        else:  # Detected OpenGL, Target is DirectX
                            converted = MapFactory.convert_normal_map_format(
                                normal_map, target_format="directx", save=False
                            )

                        context.mark_used("Normal")
                        return context.save_map(
                            converted, target_key, source_images=[normal_map]
                        )
                    else:
                        if context.logger:
                            context.logger.info(
                                f"Skipping conversion of detected {detected_format} to {target_format} (convert_type=False)",
                                extra={"preset": "highlight"},
                            )
                        # Save as generic Normal or target key?
                        # If we know it's wrong, maybe save as generic Normal to preserve it?
                        # Or save as target key but warn?
                        # Let's save as generic Normal
                        context.mark_used("Normal")
                        return context.save_map(
                            normal_map, "Normal", source_images=[normal_map]
                        )

            # Fallback: Format unknown
            # Practical Lens: If we can't detect the format, keep it generic "Normal"
            if context.logger:
                context.logger.info(
                    "Processing generic normal map (format indeterminate)",
                    extra={"preset": "highlight"},
                )
            context.mark_used("Normal")
            return context.save_map(normal_map, "Normal", source_images=[normal_map])

        # 4. Fallback (Bump, Height, etc) -> Convert to target_key
        # These are safe to convert because we know they are NOT normal maps
        normal = context.resolve_map(target_key, allow_conversion=True)

        if normal:
            if context.logger:
                context.logger.info(
                    f"Generated normal map from Bump/Height",
                    extra={"preset": "highlight"},
                )
            context.mark_used("Bump", "Height")
            source = context.inventory.get("Bump") or context.inventory.get("Height")
            sources = [source] if source else []
            return context.save_map(normal, target_key, source_images=sources)

        return None

    @staticmethod
    def _opposite(format_type: str) -> str:
        return "DirectX" if format_type == "OpenGL" else "OpenGL"

    def get_consumed_types(self) -> List[str]:
        return ["Normal", "Normal_OpenGL", "Normal_DirectX", "Bump", "Height"]


class OutputFallbackHandler(WorkflowHandler):
    """Handles outputting fallback maps for failed requests."""

    def can_handle(self, context: TextureProcessor) -> bool:
        return True

    def process(self, context: TextureProcessor) -> List[str]:
        if not context.config.get("use_output_fallbacks", True):
            return []

        output_maps = []
        registry = MapRegistry()

        # Identify all requested maps
        requested_maps = []
        for map_name in registry._maps:
            if self.is_explicitly_requested(context, map_name):
                requested_maps.append(map_name)

        # Helper to recursively get fallbacks
        def get_all_fallbacks(map_type):
            fallbacks = []

            def _recurse(mt):
                m = registry.get(mt)
                if m and m.output_fallbacks:
                    for fb in m.output_fallbacks:
                        if fb not in fallbacks:
                            fallbacks.append(fb)
                            _recurse(fb)

            _recurse(map_type)
            return fallbacks

        for req in requested_maps:
            # If the requested map was generated (or its inputs consumed), we assume success.
            # But checking inputs is hard for packed maps.
            # Instead, we just check if the fallbacks are present and unused.

            fallbacks = get_all_fallbacks(req)
            for fb in fallbacks:
                if fb in context.inventory and fb not in context.used_maps:
                    # Special check: Don't output if it's a packed map that wasn't generated
                    # (Packed maps in inventory are usually just paths, but if they exist they should be used)
                    # If it's a component map (Metallic, Smoothness), output it.

                    path = context.save_map(
                        context.inventory[fb], fb, source_images=[context.inventory[fb]]
                    )
                    output_maps.append(path)
                    context.mark_used(fb)
                    if context.logger:
                        context.logger.info(
                            f"Outputting fallback map: {fb} (fallback for {req})",
                            extra={"preset": "highlight"},
                        )

        return output_maps

    def get_consumed_types(self) -> List[str]:
        return []  # Dynamic


# =============================================================================
# Main Factory - Simplified and extensible
# =============================================================================


class MapFactory(LoggingMixin):
    """Refactored factory with pluggable workflow system."""

    DEFAULT_CONFIG = {
        "convert": True,
        "optimize": True,
        "dry_run": False,
        "force": False,
        "max_size": None,
        "old_files_folder": None,
        "rename": False,
        "mask_map_scale": 1.0,
        "output_extension": None,
        "use_input_fallbacks": True,
        "use_output_fallbacks": True,
        # Workflow flags
        "albedo_transparency": False,
        "metallic_smoothness": False,
        "mask_map": False,
        "orm_map": False,
        "convert_specgloss_to_pbr": False,
        "normal_type": "OpenGL",
        "cleanup_base_color": False,
        "ignored_patterns": ["specular_cube", "diffuse_cube", "ibl_brdf_lut"],
    }

    _conversion_registry = ConversionRegistry()
    _map_registry = MapRegistry()
    map_types = _map_registry.get_map_types()
    _workflow_handlers: List[Type[WorkflowHandler]] = [
        BaseColorHandler,
        NormalMapHandler,
        ORMMapHandler,
        MaskMapHandler,
        MetallicSmoothnessHandler,
        OutputFallbackHandler,
        SeparateMetallicRoughnessHandler,
    ]

    passthrough_maps = _map_registry.get_passthrough_maps()
    packed_grayscale_maps = _map_registry.get_scale_as_mask_types()
    map_fallbacks = _map_registry.get_fallbacks()

    # Conversion implementations
    @classmethod
    def register_conversions(cls, registry: ConversionRegistry):
        """Register all standard PBR conversions."""
        # Metallic conversions
        registry.register(
            "Metallic",
            "Specular",
            lambda inv, ctx: ctx.convert_specular_to_metallic(inv["Specular"]),
            priority=5,
        )

        # Roughness conversions
        registry.register(
            "Roughness",
            "Smoothness",
            lambda inv, ctx: ctx.convert_smoothness_to_roughness(inv["Smoothness"]),
            priority=10,
        )
        registry.register(
            "Roughness",
            "Glossiness",
            lambda inv, ctx: ctx.convert_smoothness_to_roughness(inv["Glossiness"]),
            priority=9,
        )
        registry.register(
            "Roughness",
            "Specular",
            lambda inv, ctx: ctx.convert_specular_to_roughness(inv["Specular"]),
            priority=5,
        )

        # Glossiness conversions
        registry.register(
            "Glossiness",
            "Specular",
            lambda inv, ctx: ctx.extract_gloss_from_spec(inv["Specular"]),
            priority=5,
        )
        registry.register(
            "Glossiness",
            "Roughness",
            lambda inv, ctx: ctx.convert_roughness_to_smoothness(
                inv["Roughness"]
            ),  # Inverted Roughness = Smoothness ≈ Glossiness
            priority=9,
        )
        registry.register(
            "Glossiness",
            "Smoothness",
            lambda inv, ctx: ctx.copy_map(inv["Smoothness"], "Glossiness"),
            priority=10,
        )

        # Smoothness conversions
        registry.register(
            "Smoothness",
            "Roughness",
            lambda inv, ctx: ctx.convert_roughness_to_smoothness(inv["Roughness"]),
            priority=10,
        )

        # Normal conversions
        registry.register(
            "Normal_OpenGL",
            "Normal_DirectX",
            lambda inv, ctx: ctx.convert_dx_to_gl(inv["Normal_DirectX"]),
            priority=10,
        )
        registry.register(
            "Normal_DirectX",
            "Normal_OpenGL",
            lambda inv, ctx: ctx.convert_gl_to_dx(inv["Normal_OpenGL"]),
            priority=10,
        )

        # Bump/Height to Normal conversions
        for target in ["Normal_OpenGL", "Normal_DirectX", "Normal"]:
            for source in ["Bump", "Height"]:
                registry.register(
                    target,
                    source,
                    lambda inv, ctx: ctx.convert_bump_to_normal(inv[source]),
                    priority=5,
                )
        registry.register(
            "Normal",
            ["Bump", "Height"],
            lambda inv, ctx: ctx.convert_bump_to_normal(
                inv.get("Bump") or inv["Height"]
            ),
            priority=5,
        )

        # Packing conversions (ORM)
        # Priority 10: All components present, native Roughness
        registry.register(
            "ORM",
            ["Metallic", "Roughness", "Ambient_Occlusion"],
            lambda inv, ctx: ctx.create_orm_map(inv),
            priority=10,
        )
        # Priority 9: All components present, converted Smoothness
        registry.register(
            "ORM",
            ["Metallic", "Smoothness", "Ambient_Occlusion"],
            lambda inv, ctx: ctx.create_orm_map(inv),
            priority=9,
        )
        # Priority 8: Missing AO, native Roughness
        registry.register(
            "ORM",
            ["Metallic", "Roughness"],
            lambda inv, ctx: ctx.create_orm_map(inv),
            priority=8,
        )
        # Priority 7: Missing AO, converted Smoothness
        registry.register(
            "ORM",
            ["Metallic", "Smoothness"],
            lambda inv, ctx: ctx.create_orm_map(inv),
            priority=7,
        )

        # Packing conversions (MSAO/MaskMap)
        # Priority 10: All components present, native Smoothness
        registry.register(
            "MSAO",
            ["Metallic", "Ambient_Occlusion", "Smoothness"],
            lambda inv, ctx: ctx.create_mask_map(inv),
            priority=10,
        )
        # Priority 9: All components present, converted Roughness
        registry.register(
            "MSAO",
            ["Metallic", "Ambient_Occlusion", "Roughness"],
            lambda inv, ctx: ctx.create_mask_map(inv),
            priority=9,
        )
        # Priority 8: Missing AO, native Smoothness
        registry.register(
            "MSAO",
            ["Metallic", "Smoothness"],
            lambda inv, ctx: ctx.create_mask_map(inv),
            priority=8,
        )
        # Priority 7: Missing AO, converted Roughness
        registry.register(
            "MSAO",
            ["Metallic", "Roughness"],
            lambda inv, ctx: ctx.create_mask_map(inv),
            priority=7,
        )
        # Priority 6: Missing Smoothness (Metallic + AO)
        registry.register(
            "MSAO",
            ["Metallic", "Ambient_Occlusion"],
            lambda inv, ctx: ctx.create_mask_map(inv),
            priority=6,
        )
        # Priority 5: Missing Metallic (Smoothness + AO)
        registry.register(
            "MSAO",
            ["Ambient_Occlusion", "Smoothness"],
            lambda inv, ctx: ctx.create_mask_map(inv),
            priority=5,
        )

        # Packing conversions (Metallic_Smoothness)
        registry.register(
            "Metallic_Smoothness",
            ["Metallic", "Smoothness"],
            lambda inv, ctx: ctx.create_metallic_smoothness_map(inv),
            priority=10,
        )
        registry.register(
            "Metallic_Smoothness",
            ["Metallic", "Roughness"],
            lambda inv, ctx: ctx.create_metallic_smoothness_map(inv),
            priority=9,
        )

        # Unpacking conversions (Metallic_Smoothness)
        registry.register(
            "Metallic",
            "Metallic_Smoothness",
            lambda inv, ctx: ctx.get_metallic_from_packed(inv["Metallic_Smoothness"]),
            priority=8,
        )

        registry.register(
            "Smoothness",
            "Metallic_Smoothness",
            lambda inv, ctx: ctx.get_smoothness_from_packed(inv["Metallic_Smoothness"]),
            priority=8,
        )
        registry.register(
            "Roughness",
            "Metallic_Smoothness",
            lambda inv, ctx: ctx.get_roughness_from_packed(inv["Metallic_Smoothness"]),
            priority=8,
        )

        # Unpacking conversions (MSAO)
        registry.register(
            "Metallic",
            "MSAO",
            lambda inv, ctx: ctx.get_metallic_from_msao(inv["MSAO"]),
            priority=8,
        )
        registry.register(
            "Smoothness",
            "MSAO",
            lambda inv, ctx: ctx.get_smoothness_from_msao(inv["MSAO"]),
            priority=8,
        )
        registry.register(
            "Roughness",
            "MSAO",
            lambda inv, ctx: ctx.get_roughness_from_msao(inv["MSAO"]),
            priority=8,
        )
        registry.register(
            "Ambient_Occlusion",
            "MSAO",
            lambda inv, ctx: ctx.get_ao_from_msao(inv["MSAO"]),
            priority=8,
        )
        registry.register(
            "AO",
            "MSAO",
            lambda inv, ctx: ctx.get_ao_from_msao(inv["MSAO"]),
            priority=8,
        )

        # Unpacking conversions (ORM)
        registry.register(
            "Ambient_Occlusion",
            "ORM",
            lambda inv, ctx: ctx.get_ao_from_orm(inv["ORM"]),
            priority=8,
        )
        registry.register(
            "AO",
            "ORM",
            lambda inv, ctx: ctx.get_ao_from_orm(inv["ORM"]),
            priority=8,
        )
        registry.register(
            "Roughness",
            "ORM",
            lambda inv, ctx: ctx.get_roughness_from_orm(inv["ORM"]),
            priority=8,
        )
        registry.register(
            "Smoothness",
            "ORM",
            lambda inv, ctx: ctx.get_smoothness_from_orm(inv["ORM"]),
            priority=8,
        )
        registry.register(
            "Metallic",
            "ORM",
            lambda inv, ctx: ctx.get_metallic_from_orm(inv["ORM"]),
            priority=8,
        )

        # Unpacking conversions (Albedo_Transparency)
        registry.register(
            "Base_Color",
            "Albedo_Transparency",
            lambda inv, ctx: ctx.get_base_color_from_albedo_transparency(
                inv["Albedo_Transparency"]
            ),
            priority=8,
        )
        registry.register(
            "Opacity",
            "Albedo_Transparency",
            lambda inv, ctx: ctx.get_opacity_from_albedo_transparency(
                inv["Albedo_Transparency"]
            ),
            priority=8,
        )

    @classmethod
    def resolve_map_type(cls, file: str, key: bool = True, validate: str = None) -> str:
        """Resolves the map type from a filename or alias using `map_types`.

        Parameters:
            file (str): Image filename, full path, or map type suffix.
            key (bool): If True, get the corresponding key from 'map_types'.
                        If False, get the abbreviation from 'map_types'.
            validate (str, optional): If provided, validate the map type against this expected type.

        Returns:
            str: The map type.

        Raises:
            ValueError: If the map type is not the expected type when 'validate' is provided.
        """
        ImgUtils.assert_pathlike(file, "file")
        filename = FileUtils.format_path(file, "name")

        if key:
            result = cls._map_registry.resolve_type_from_path(file)
        else:
            result = next(
                (
                    i
                    for v in cls.map_types.values()
                    for i in v
                    if filename.lower().endswith(i.lower())
                ),
                (
                    StrUtils.split_delimited_string(filename, "_", occurrence=-1)[1]
                    or None
                ),
            )

        if validate:
            # Check both keys and values for validation
            valid_types = [validate] + list(cls.map_types[validate])
            if result not in valid_types:
                raise ValueError(
                    f"Invalid map type '{result}'. Expected type is one of: {valid_types}"
                )

        return result

    @classmethod
    def resolve_texture_filename(
        cls,
        texture_path: str,
        map_type: str,
        prefix: str = None,
        suffix: str = None,
        ext: str = None,
    ) -> str:
        """Generates a correctly formatted filename while preserving the original suffix and file extension.

        Parameters:
            texture_path (str): Path to the original texture.
            map_type (str): The type of map being generated.
            prefix (str, optional): Extra prefix for renaming, e.g., "Optimized_".
            suffix (str, optional): Extra suffix for renaming, e.g., "_old" or "_optimized".
            ext (str, optional): The desired file extension (e.g., "png", "tga").
                                If None, keeps the original format.
        Returns:
            str: The resolved output file path.
        """
        ImgUtils.assert_pathlike(texture_path, "texture_path")

        # Extract sections from the given path
        directory = FileUtils.format_path(texture_path, "path")
        base_name = cls.get_base_texture_name(texture_path)
        original_ext = FileUtils.format_path(texture_path, "ext")

        # Ensure map_type does not start with an underscore
        map_type = map_type.lstrip("_")

        # Ensure suffix formatting (prevents double underscores)
        suffix = f"_{suffix.lstrip('_')}" if suffix else ""

        # Determine output file extension (preserve original unless explicitly changed)
        ext = f".{ext.lower().lstrip('.')}" if ext else f".{original_ext}"

        # Construct the final filename correctly
        new_name = StrUtils.replace_placeholders(
            "{prefix}{base_name}_{map_type}{suffix}{ext}",
            prefix=prefix or "",
            base_name=base_name,
            map_type=map_type,
            suffix=suffix,
            ext=ext,
        )

        return os.path.join(directory, new_name)

    @classmethod
    def get_base_texture_name(cls, filepath_or_filename: str) -> str:
        """Extracts the base texture name from a filename or path,
        removing known suffixes (e.g., _normal, _roughness).

        Logic:
        - Long suffixes (>3 chars): Case-insensitive.
        - Short suffixes (<=3 chars): Must start with a capital letter (rest case-insensitive) to avoid false positives.

        Parameters:
            filepath_or_filename (str): A texture path or name.

        Returns:
            str: The base name without map-type suffix.
        """
        ImgUtils.assert_pathlike(filepath_or_filename, "filepath_or_filename")

        filename = os.path.basename(str(filepath_or_filename))
        base_name, _ = os.path.splitext(filename)

        all_suffixes = set()
        for suffixes in cls.map_types.values():
            all_suffixes.update(suffixes)

        if not all_suffixes:
            return base_name

        sorted_suffixes = sorted(list(all_suffixes), key=len, reverse=True)

        # 1. Build Pattern for Underscore-Delimited Suffixes (Loose/Case-Insensitive)
        # Matches: _AO, _ao, _Normal, _normal at end of string
        # (?i:...) makes the group case-insensitive
        p_underscore_inner = "|".join(re.escape(s) for s in sorted_suffixes)
        pattern_underscore = f"_(?i:{p_underscore_inner})$"

        # 2. Build Pattern for Attached Suffixes (Strict for Short)
        # Long (>3): Case Insensitive
        # Short (<=3): Capitalized First Letter
        short_suffixes = [s for s in sorted_suffixes if len(s) <= 3]
        long_suffixes = [s for s in sorted_suffixes if len(s) > 3]

        attached_parts = []

        if long_suffixes:
            p_long = "|".join(re.escape(s) for s in long_suffixes)
            attached_parts.append(f"(?i:{p_long})")

        if short_suffixes:
            p_short_parts = []
            for s in short_suffixes:
                if s and s[0].isalpha():
                    first = s[0].upper()
                    rest = re.escape(s[1:])
                    p_short_parts.append(f"{first}(?i:{rest})")
                else:
                    p_short_parts.append(re.escape(s))
            attached_parts.append("|".join(p_short_parts))

        pattern_attached = f"(?:{'|'.join(attached_parts)})$"

        # Combine: Try Underscore first (greedy), then Attached
        full_pattern = f"(?:{pattern_underscore}|{pattern_attached})"

        base_name = StrUtils.format_suffix(base_name, strip=full_pattern)

        return base_name.rstrip("_")

    @classmethod
    def group_textures_by_set(cls, image_paths: List[str]) -> Dict[str, List[str]]:
        """Groups texture maps into sets based on matching base names.

        Parameters:
            image_paths (List[str]): A list of full image file paths.

        Returns:
            Dict[str, List[str]]: A dictionary where:
                - Keys are unique base texture names.
                - Values are lists of associated texture files.
        """
        texture_sets = {}
        for path in image_paths:
            base_name = cls.get_base_texture_name(path)  # Extract base texture name
            # print(f"[grouping] {path} → {base_name}")
            if base_name not in texture_sets:
                texture_sets[base_name] = []

            texture_sets[base_name].append(path)

        return texture_sets

    @classmethod
    def filter_images_by_type(cls, files, types=""):
        """
        Parameters:
            files (list): A list of image filenames, fullpaths, or map type suffixes.
            types (str/list): Any of the keys in the 'map_types' dict.
                    A single string or a list of strings representing the types. ex. 'Base_Color','Roughness','Metallic','Ambient_Occlusion','Normal',
                        'Normal_DirectX','Normal_OpenGL','Height','Emissive','Diffuse','Specular',
                        'Glossiness','Displacement','Refraction','Reflection'
        Returns:
            (list)
        """
        types = IterUtils.make_iterable(types)
        return [f for f in files if cls.resolve_map_type(f) in types]

    @classmethod
    def sort_images_by_type(
        cls, files: Union[List[Union[str, Tuple[str, Any]]], Dict[str, Any]]
    ) -> Dict[str, List[Union[str, Tuple[str, Any]]]]:
        """Sort image files by map type based on the input format.

        Parameters:
            files (Union[List[Union[str, Tuple[str, Any]]], Dict[str, Any]]): A list of image filenames, full paths, tuples of (filename, image file),
                    or a dictionary with filenames as keys and image files as values.
        Returns:
            Dict[str, List[Union[str, Tuple[str, Any]]]]: A dictionary where each key is a map type. The values are lists that match the input format,
                    containing either just the paths or tuples of (path, file data).
        """
        if isinstance(files, dict):
            # Convert dictionary to list of tuples
            files = list(files.items())

        sorted_images = {}
        for file in files:
            # Determine if the input is a path or a tuple of (path, file data)
            is_tuple = isinstance(file, tuple)

            file_path = file[0] if is_tuple else file
            map_type = cls.resolve_map_type(file_path)
            if not map_type:
                continue

            if map_type not in sorted_images:
                sorted_images[map_type] = []

            # Add the file to the sorted list according to its input format
            sorted_images[map_type].append(file if is_tuple else file_path)

        return sorted_images

    @classmethod
    def contains_map_types(cls, files, map_types):
        """Check if the given images contain the given map types.

        Parameters:
            files (list)(dict): filenames, fullpaths, or map type suffixes as the first element
                of two-element tuples or keys in a dictionary. ex. [('file', <image>)] or {'file': <image>} or {'type': ('file', <image>)}
            map_types (str/list): The map type(s) to query. Any of the keys in the 'map_types' dict.
                A single string or a list of strings representing the types. ex. 'Base_Color','Roughness','Metallic','Ambient_Occlusion','Normal',
                    'Normal_DirectX','Normal_OpenGL','Height','Emissive','Diffuse','Specular',
                    'Glossiness','Displacement','Refraction','Reflection'
        Returns:
            (bool)
        """
        if isinstance(files, (list, set, tuple)):
            # convert list to dict of the correct format.
            files = cls.sort_images_by_type(files)

        map_types = IterUtils.make_iterable(map_types)

        result = next(
            (True for i in files.keys() if cls.resolve_map_type(i) in map_types),
            False,
        )

        return True if result else False

    @classmethod
    def is_normal_map(cls, file):
        """Check the map type for one of the normal values in map_types.

        Parameters:
            file (str): Image filename, fullpath, or map type suffix.

        Returns:
            (bool)
        """
        typ = cls.resolve_map_type(file)
        return any(
            (
                typ in cls.map_types["Normal_DirectX"],
                typ in cls.map_types["Normal_OpenGL"],
                typ in cls.map_types["Normal"],
            )
        )

    @classmethod
    def register_handler(cls, handler_class: Type[WorkflowHandler]):
        """Register a custom workflow handler (extensibility)."""
        cls._workflow_handlers.insert(-2, handler_class)  # Before default handlers

    @classmethod
    def register_conversion(cls, conversion: MapConversion):
        """Register a custom map conversion (extensibility)."""
        cls._conversion_registry.register(conversion)

    @classmethod
    def get_map_fallbacks(cls, map_type: str) -> Tuple[str, ...]:
        """Get fallback map types for a given map type.

        Parameters:
            map_type (str): The map type to get fallbacks for.

        Returns:
            Tuple[str, ...]: A tuple of fallback map types.
        """
        return cls.map_fallbacks.get(map_type, ())

    @classmethod
    def get_precedence_rules(cls) -> Dict[str, List[str]]:
        """Returns a dictionary of map precedence rules.

        Format: { "DominantMap": ["RedundantMap1", "RedundantMap2"] }
        """
        return cls._map_registry.get_precedence_rules()

    @classmethod
    def filter_redundant_maps(cls, sorted_maps: Dict[str, List[str]]) -> None:
        """Filters out maps that are rendered redundant by other present maps (e.g. MSAO).

        Modifies the sorted_maps dictionary in-place.

        Parameters:
            sorted_maps: Dictionary of map types to file paths.
        """
        precedence_rules = cls.get_precedence_rules()

        for dominant, redundants in precedence_rules.items():
            if dominant in sorted_maps and sorted_maps[dominant]:
                for redundant in redundants:
                    if redundant in sorted_maps:
                        cls.logger.info(
                            f"Skipping {redundant} map (replaced by {dominant})",
                            extra={"preset": "highlight"},
                        )
                        del sorted_maps[redundant]

    @classmethod
    def prepare_maps(
        cls,
        source: Union[str, List[str]],
        output_dir: str = None,
        group_by_set: bool = True,
        max_workers: int = 1,
        **kwargs,
    ) -> Union[List[str], Dict[str, List[str]]]:
        """
        Main factory method. Automatically handles batch processing.

        Parameters:
            source: A directory path (str), a single file path (str), or a list of file paths.
            output_dir: Optional output directory.
            group_by_set: Whether to automatically group textures into sets (default: True).
                          If False, all input files are treated as a single set.
            max_workers: Number of threads for parallel processing.
            **kwargs: Configuration options overriding DEFAULT_CONFIG.
                      Key options:
                      - use_input_fallbacks (bool): Allow generating maps from alternative inputs (e.g. Diffuse -> Base Color).
                      - use_output_fallbacks (bool): Allow substituting missing maps with alternatives (e.g. AO -> Mask).
                      - convert (bool): Enable format conversion/renaming.
                      - optimize (bool): Enable image optimization.
                      - force_packed_maps (bool): Force generation of packed maps even if components are missing.

        Returns:
            List[str] if a single asset was processed.
            Dict[str, List[str]] if multiple assets were processed (keyed by asset name).
        """
        # Normalize config
        workflow_config = cls.DEFAULT_CONFIG.copy()
        workflow_config.update(kwargs)

        # Extract logger if provided, else use class logger
        logger = kwargs.get("logger", cls.logger)

        if Image is None:
            logger.warning(
                "Pillow (PIL) is not installed. Image processing operations will be limited."
            )

        # Resolve input files
        files = []
        if isinstance(source, str):
            if os.path.isdir(source):
                files = FileUtils.get_dir_contents(
                    source,
                    "filepath",
                    inc_files=[f"*.{ext}" for ext in ImgUtils.texture_file_types],
                )
            elif os.path.isfile(source):
                files = [source]
        else:
            files = source

        if not files:
            if logger:
                logger.warning("No input files found.")
            return []

        # Filter ignored files
        ignored_patterns = workflow_config.get("ignored_patterns", [])
        if ignored_patterns:
            files = [
                f
                for f in files
                if not any(
                    pat.lower() in os.path.basename(f).lower()
                    for pat in ignored_patterns
                )
            ]
            if not files:
                if logger:
                    logger.warning(
                        "All input files were filtered out by ignored_patterns."
                    )
                return []

        if group_by_set:
            # Group by texture set
            texture_sets = cls.group_textures_by_set(files)
        else:
            # Treat all files as a single set
            # Use the common prefix or just the first file's base name as the key
            base_name = cls.get_base_texture_name(files[0])
            texture_sets = {base_name: files}

        results = {}
        total_sets = len(texture_sets)

        if total_sets > 1:
            if logger:
                logger.info(f"Found {total_sets} texture sets. Processing batch...")

        if max_workers > 1 and total_sets > 1:
            import concurrent.futures

            def process_set(args):
                i, base_name, textures = args
                try:
                    if total_sets > 1 and logger:
                        logger.info(f"Processing set {i}/{total_sets}: {base_name}")

                    generated = cls._process_map_set(
                        textures,
                        workflow_config,
                        output_dir=output_dir,
                        logger=logger,
                    )
                    return base_name, generated
                except Exception as e:
                    if logger:
                        logger.error(f"Error processing set {base_name}: {e}")
                    import traceback

                    traceback.print_exc()
                    return base_name, []

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                tasks = [
                    (i, base_name, textures)
                    for i, (base_name, textures) in enumerate(texture_sets.items(), 1)
                ]
                future_to_set = {
                    executor.submit(process_set, task): task for task in tasks
                }

                for future in concurrent.futures.as_completed(future_to_set):
                    base_name, generated = future.result()
                    if generated:
                        results[base_name] = generated
        else:
            for i, (base_name, textures) in enumerate(texture_sets.items(), 1):
                if total_sets > 1 and logger:
                    logger.info(f"Processing set {i}/{total_sets}: {base_name}")

                try:
                    generated = cls._process_map_set(
                        textures,
                        workflow_config,
                        output_dir=output_dir,
                        logger=logger,
                    )
                    results[base_name] = generated
                except Exception as e:
                    if logger:
                        logger.error(f"Error processing set {base_name}: {e}")
                    import traceback

                    traceback.print_exc()

        # Smart return: if single set, return list directly
        if len(results) == 1:
            return next(iter(results.values()))

        return results

    @classmethod
    def _process_map_set(
        cls,
        textures: List[str],
        workflow_config: dict,
        output_dir: str = None,
        logger: Any = None,
    ) -> List[str]:
        """Internal method to process a single set of textures (one asset)."""
        # Build inventory
        map_inventory = MapFactory._build_map_inventory(textures)

        convert = workflow_config.get("convert", True)

        # Pre-process: Spec/Gloss conversion
        if convert:
            map_inventory = MapFactory._convert_specgloss_workflow(
                map_inventory, workflow_config
            )

        # Create processing context
        # Use the first input texture as a reference for directory and naming
        # This ensures we have a valid path even if the inventory contains Image objects
        reference_path = textures[0] if textures else None

        if not reference_path:
            return []

        context = TextureProcessor(
            inventory=map_inventory,
            config=workflow_config,
            output_dir=output_dir or os.path.dirname(reference_path),
            base_name=MapFactory.get_base_texture_name(reference_path),
            ext=workflow_config.get("output_extension", "png"),
            conversion_registry=MapFactory._conversion_registry,
            logger=logger or MapFactory.logger,
        )

        # Process through workflow handlers
        output_maps = []
        if convert:
            for handler_class in MapFactory._workflow_handlers:
                handler = handler_class()
                if handler.can_handle(context):
                    result = handler.process(context)
                    if result:
                        if isinstance(result, list):
                            output_maps.extend(result)
                        else:
                            output_maps.append(result)

                        consumed = handler.get_consumed_types()
                        context.mark_used(*consumed)
                        # Handlers are no longer mutually exclusive - explicit output required
                        # if handler_class not in [
                        #     SeparateMetallicRoughnessHandler,
                        #     BaseColorHandler,
                        #     NormalMapHandler,
                        # ]:
                        #     break  # Stop after first match for packed workflows

        # Pass through unconsumed maps
        for map_type in MapFactory.passthrough_maps:
            if map_type in map_inventory and map_type not in context.used_maps:
                path = context.save_map(
                    map_inventory[map_type],
                    map_type,
                    source_images=[map_inventory[map_type]],
                )
                output_maps.append(path)
                if context.logger:
                    context.logger.info(f"Passing through {map_type} map")

        # Cleanup intermediate files
        # We normalize paths to ensure reliable comparison
        normalized_outputs = {os.path.normpath(p) for p in output_maps}

        for created_file in context.created_files:
            if os.path.normpath(created_file) not in normalized_outputs:
                try:
                    if os.path.exists(created_file):
                        os.remove(created_file)
                        # callback(f"Removed intermediate file: {os.path.basename(created_file)}")
                except OSError as e:
                    if context.logger:
                        context.logger.warning(f"Error removing intermediate file: {e}")

        return output_maps if output_maps else textures

    @staticmethod
    def _build_map_inventory(textures: List[str]) -> Dict[str, str]:
        """Build map inventory using ImgUtils."""
        inventory = {}
        # Sort textures by length descending to prefer more specific names (e.g. Mixed_AO over AO)
        for texture in sorted(textures, key=len, reverse=True):
            map_type = MapFactory.resolve_map_type(texture)
            if map_type and map_type not in inventory:
                inventory[map_type] = texture
        return inventory

    @classmethod
    def _convert_specgloss_workflow(
        cls,
        inventory: Dict[str, Union[str, "Image.Image"]],
        config: dict,
    ) -> Dict[str, Union[str, "Image.Image"]]:
        """Convert Spec/Gloss workflow to PBR."""
        spec_map = inventory.get("Specular")
        gloss_map = inventory.get("Glossiness") or inventory.get("Smoothness")
        diffuse_map = inventory.get("Diffuse")

        # Attempt to extract Glossiness from Specular Alpha if missing
        if spec_map and not gloss_map:
            try:
                img = ImgUtils.ensure_image(spec_map)
                if "A" in img.getbands():
                    cls.logger.info(
                        "Found Alpha in Specular map, using as Glossiness.",
                        extra={"preset": "highlight"},
                    )
                    gloss_map = img.getchannel(
                        "A"
                    )  # Use extracted channel as Image object
            except Exception as e:
                cls.logger.warning(f"Error checking Specular alpha: {e}")

        # Require both Specular and Glossiness (file or extracted) to attempt conversion
        if not (spec_map and gloss_map):
            return inventory

        try:
            # Get output params from config
            first_map = next(iter(inventory.values()))
            if isinstance(first_map, str):
                output_dir = os.path.dirname(first_map)
            else:
                output_dir = None

            base_color_img, metallic_img, roughness_img = (
                MapFactory.convert_spec_gloss_to_pbr(
                    specular_map=spec_map,
                    glossiness_map=gloss_map,
                    diffuse_map=diffuse_map,
                    output_dir=output_dir,
                    write_files=False,
                )
            )

            new_inventory = inventory.copy()
            new_inventory["Base_Color"] = base_color_img
            new_inventory["Metallic"] = metallic_img
            new_inventory["Roughness"] = roughness_img

            # Remove converted maps
            for key in ["Specular", "Glossiness", "Smoothness", "Diffuse"]:
                new_inventory.pop(key, None)

            cls.logger.info(
                "Converted Spec/Gloss workflow to PBR Metal/Rough",
                extra={"preset": "highlight"},
            )
            return new_inventory

        except Exception as e:
            cls.logger.error(f"Error converting Spec/Gloss: {str(e)}")
            return inventory

    @classmethod
    def pack_transparency_into_albedo(
        cls,
        albedo_map_path: str,
        alpha_map_path: str,
        output_dir: Optional[str] = None,
        suffix: Optional[str] = "_AlbedoTransparency",
        invert_alpha: bool = False,
        output_path: Optional[str] = None,
        save: bool = True,
    ) -> Union[str, "Image.Image"]:
        """Combines an albedo texture with a transparency map by packing the transparency into the alpha channel.

        Parameters:
            albedo_map_path (str): Path to the albedo (base color) texture map.
            alpha_map_path (str): Path to the transparency (alpha) texture map.
            output_dir (str, optional): Output directory. If None, uses the albedo map directory.
            suffix (str, optional): Suffix for the output file name. Defaults to '_AlbedoTransparency'.
            invert_alpha (bool, optional): If True, inverts the alpha texture.
            output_path (str, optional): Explicit output path. Overrides output_dir/suffix logic.
            save (bool, optional): If True, saves to disk. If False, returns PIL Image.

        Returns:
            str | Image.Image: The output file path or PIL Image object.
        """
        if isinstance(albedo_map_path, str):
            ImgUtils.assert_pathlike(albedo_map_path, "albedo_map_path")
        if isinstance(alpha_map_path, str):
            ImgUtils.assert_pathlike(alpha_map_path, "alpha_map_path")

        if save and output_path is None:
            if not isinstance(albedo_map_path, str):
                raise ValueError(
                    "Cannot determine output path from Image object. Please provide output_path or output_dir."
                )

            base_name = ImgUtils.get_base_texture_name(albedo_map_path)

            if output_dir is None:
                output_dir = os.path.dirname(albedo_map_path)
            elif not os.path.isdir(output_dir):
                raise ValueError(
                    f"The specified output directory '{output_dir}' is not valid."
                )

            output_path = os.path.join(
                output_dir, f"{base_name}{suffix}.{ALPHA_EXTENSION}"
            )
        elif not save:
            output_path = None

        return ImgUtils.pack_channel_into_alpha(
            albedo_map_path,
            alpha_map_path,
            output_path,
            invert_alpha=invert_alpha,
        )

    @classmethod
    def pack_smoothness_into_metallic(
        cls,
        metallic_map_path: str,
        alpha_map_path: str,
        output_dir: str = None,
        suffix: str = "_MetallicSmoothness",
        invert_alpha: bool = False,
        output_path: str = None,
        save: bool = True,
    ) -> Union[str, "Image.Image"]:
        """Packs a smoothness (or inverted roughness) texture into the alpha channel of a metallic texture map.

        Parameters:
            metallic_map_path (str): Path to the metallic texture map.
            alpha_map_path (str): Path to the smoothness or roughness texture map.
            output_dir (str, optional): Directory path for the output. If None, the output directory will be the same as the metallic map path.
            invert_alpha (bool, optional): If True, the alpha (smoothness/roughness) texture will be inverted.
            suffix (str, optional): Suffix for the output file name, defaulting to '_MetallicSmoothness'.
            output_path (str, optional): Explicit output path. Overrides output_dir/suffix logic.
            save (bool, optional): If True, saves to disk. If False, returns PIL Image.

        Returns:
            str | Image.Image: The file path of the newly created metallic-smoothness texture map or PIL Image.
        """
        if isinstance(metallic_map_path, str):
            ImgUtils.assert_pathlike(metallic_map_path, "metallic_map_path")
        if isinstance(alpha_map_path, str):
            ImgUtils.assert_pathlike(alpha_map_path, "alpha_map_path")

        if save and output_path is None:
            if not isinstance(metallic_map_path, str):
                raise ValueError(
                    "Cannot determine output path from Image object. Please provide output_path or output_dir."
                )

            base_name = ImgUtils.get_base_texture_name(metallic_map_path)
            if output_dir is None:
                output_dir = os.path.dirname(metallic_map_path)
            elif not os.path.isdir(output_dir):
                raise ValueError(
                    f"The specified output directory '{output_dir}' is not valid."
                )

            output_path = os.path.join(
                output_dir, f"{base_name}{suffix}.{ALPHA_EXTENSION}"
            )
        elif not save:
            output_path = None

        return ImgUtils.pack_channel_into_alpha(
            metallic_map_path, alpha_map_path, output_path, invert_alpha=invert_alpha
        )

    @classmethod
    def detect_normal_map_format(
        cls, image: Union[str, "Image.Image"], threshold: float = 0.1
    ) -> Optional[str]:
        """Detects if a normal map is OpenGL (Y+) or DirectX (Y-) based on surface integrability.

        Theory:
        If a normal map represents a continuous height field H(x,y):
        Red channel R ~ dH/dx
        Green channel G ~ dH/dy (OpenGL) or -dH/dy (DirectX)

        The cross derivatives must be equal: d(dH/dx)/dy = d(dH/dy)/dx
        Therefore: dR/dy = dG/dx (OpenGL)
        Or:        dR/dy = -dG/dx (DirectX)

        Parameters:
            image (str | PIL.Image.Image): Input normal map.
            threshold (float): Correlation threshold (0.0 to 1.0).

        Returns:
            str | None: "OpenGL", "DirectX", or None if indeterminate.
        """
        try:
            img = ImgUtils.ensure_image(image)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize for speed if too large (we just need statistics)
            # 512x512 is plenty for statistical analysis
            if max(img.size) > 512:
                img.thumbnail((512, 512))

            arr = np.array(img, dtype=np.float32)

            # Extract R and G channels
            R = arr[:, :, 0]
            G = arr[:, :, 1]

            # Calculate gradients
            # dR/dy: Gradient of Red in Y direction (axis 0)
            dRy = np.gradient(R, axis=0)

            # dG/dx: Gradient of Green in X direction (axis 1)
            dGx = np.gradient(G, axis=1)

            # Flatten arrays
            dRy_flat = dRy.flatten()
            dGx_flat = dGx.flatten()

            # Calculate correlation coefficient
            correlation = np.corrcoef(dRy_flat, dGx_flat)[0, 1]

            if correlation > threshold:
                return "OpenGL"
            elif correlation < -threshold:
                return "DirectX"
            else:
                return None

        except Exception as e:
            print(f"Error detecting normal map format: {e}")
            return None

    @classmethod
    def convert_normal_map_format(
        cls,
        file: str,
        target_format: str,
        output_path: str = None,
        save: bool = True,
        **kwargs,
    ) -> Union[str, "Image.Image"]:
        """
        Converts a normal map between OpenGL (Y+) and DirectX (Y-) formats by inverting the green channel.

        Parameters:
            file (str): Path to the input normal map.
            target_format (str): The target format ('opengl' or 'directx').
            output_path (str, optional): Path to save the converted map. If None, a new name is generated.
            save (bool): Whether to save the image to disk.
            **kwargs: Additional arguments for Image.save().

        Returns:
            Union[str, Image.Image]: The path to the saved image or the PIL Image object.
        """
        ImgUtils.assert_pathlike(file, "file")

        target_format = target_format.lower()
        if target_format not in ("opengl", "directx"):
            raise ValueError("target_format must be 'opengl' or 'directx'")

        # Determine source format for validation and naming
        if target_format == "opengl":
            source_type_key = "Normal_DirectX"
            target_type_key = "Normal_OpenGL"
        else:
            source_type_key = "Normal_OpenGL"
            target_type_key = "Normal_DirectX"

        try:
            typ = cls.resolve_map_type(file, key=False, validate=source_type_key)
        except ValueError:
            try:
                typ = cls.resolve_map_type(file, key=False, validate="Normal")
            except ValueError:
                typ = ""

        inverted_image = ImgUtils.invert_channels(file, "g")

        if not save:
            return inverted_image

        if output_path is None:
            output_dir = FileUtils.format_path(file, "path")
            name = FileUtils.format_path(file, "name")
            ext = FileUtils.format_path(file, "ext")

            # Try to find corresponding suffix
            new_suffix = cls.map_types[target_type_key][0]  # Default
            if typ:
                try:
                    # If we found a specific source suffix, try to map it to target suffix by index
                    if typ in cls.map_types[source_type_key]:
                        index = cls.map_types[source_type_key].index(typ)
                        if index < len(cls.map_types[target_type_key]):
                            new_suffix = cls.map_types[target_type_key][index]
                except (ValueError, IndexError, KeyError):
                    pass

                name = name.removesuffix(typ)

            output_path = f"{output_dir}/{name}{new_suffix}.{ext}"

        output_path = os.path.abspath(output_path)
        inverted_image.save(output_path, **kwargs)
        return output_path

    @classmethod
    def convert_bump_to_normal(
        cls,
        bump_map: Union[str, "Image.Image"],
        output_path: str = None,
        intensity: float = 1.0,
        output_format: str = "opengl",
        smooth_filter: bool = True,
        filter_radius: float = 0.5,
        edge_wrap: bool = False,
        save: bool = True,
        **kwargs,
    ) -> Union[str, "Image.Image"]:
        """Convert a bump/height map to a tangent-space normal map.

        This method follows industry best practices from Substance, Marmoset, and V-Ray
        for generating high-quality normal maps from height data.

        Parameters:
            bump_map (str | PIL.Image.Image): Input bump/height map file path or image.
            output_path (str, optional): Output file path. If None, generates based on input.
            intensity (float): Height depth multiplier (0.1 = subtle, 2.0+ = dramatic).
                               Controls how "deep" the height values are interpreted.
            output_format (str): Target normal map format - "opengl" or "directx".
                               Affects Y-channel (green) orientation.
            smooth_filter (bool): Apply smoothing to reduce aliasing artifacts.
            filter_radius (float): Radius for smoothing filter (0.1-2.0 range).
            edge_wrap (bool): Whether to wrap edges for seamless tiling.
            save (bool): Whether to save the image to disk. Defaults to True.
            **kwargs: Additional keyword arguments passed to the image save method (e.g., optimize=True).

        Returns:
            str | PIL.Image.Image: Path to the generated normal map file if saved, else the PIL Image object.

        Notes:
            - Uses Sobel operator for gradient calculation (industry standard)
            - OpenGL: Y+ points up (green channel positive = surface pointing up)
            - DirectX: Y+ points down (green channel inverted from OpenGL)
            - Intensity should be scaled based on real-world height units
            - Pre-filtering reduces mipmap artifacts in final rendering
        """
        # Load and ensure grayscale; validate path only when a path is provided
        if isinstance(bump_map, str):
            ImgUtils.assert_pathlike(bump_map, "bump_map")
        image = ImgUtils.ensure_image(bump_map, "L")

        # Apply smoothing filter to reduce aliasing if requested
        if smooth_filter and filter_radius > 0:
            # Use Gaussian blur to smooth height data before gradient calculation
            image = image.filter(ImageFilter.GaussianBlur(radius=filter_radius))

        # Convert to numpy array for gradient calculations
        height_srgb = np.asarray(image, dtype=np.float32) / 255.0

        # Convert sRGB grayscale to linear before computing derivatives (safer filtering/derivatives)
        height_lin = ImgUtils._srgb_to_linear_np(height_srgb)

        # Calculate gradients using Sobel operator (industry standard)
        # Sobel X kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        # Sobel Y kernel: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        if edge_wrap:
            # Pad with wrapped edges for seamless tiling
            padded = np.pad(height_lin, 1, mode="wrap")
        else:
            # Pad with edge values
            padded = np.pad(height_lin, 1, mode="edge")

        # Sobel X gradient (horizontal edges)
        grad_x = (
            -1 * padded[:-2, :-2]
            + 1 * padded[:-2, 2:]
            + -2 * padded[1:-1, :-2]
            + 2 * padded[1:-1, 2:]
            + -1 * padded[2:, :-2]
            + 1 * padded[2:, 2:]
        ) / 8.0

        # Sobel Y gradient (vertical edges)
        grad_y = (
            -1 * padded[:-2, :-2]
            + -2 * padded[:-2, 1:-1]
            + -1 * padded[:-2, 2:]
            + 1 * padded[2:, :-2]
            + 2 * padded[2:, 1:-1]
            + 1 * padded[2:, 2:]
        ) / 8.0

        # Scale gradients by intensity
        grad_x *= intensity
        grad_y *= intensity

        # Calculate normal vectors
        # The cross product of tangent (1,0,grad_x) and bitangent (0,1,grad_y)
        # gives us the surface normal (-grad_x, -grad_y, 1)
        normal_x = -grad_x
        normal_y = -grad_y
        normal_z = np.ones_like(grad_x)

        # Normalize the normal vectors (with epsilon to avoid division by zero)
        length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        length = np.maximum(length, 1e-8)
        normal_x /= length
        normal_y /= length
        normal_z /= length

        # Handle DirectX vs OpenGL Y-channel orientation
        if output_format.lower() == "directx":
            # DirectX expects Y+ to point down, so invert Y component
            normal_y = -normal_y
        # OpenGL is the default (Y+ points up)

        # Convert from [-1,1] to [0,255] range for RGB channels
        # R = X component, G = Y component, B = Z component
        red_f = (normal_x + 1.0) * 127.5
        green_f = (normal_y + 1.0) * 127.5
        blue_f = (normal_z + 1.0) * 127.5

        # Clamp to valid [0,255] range before casting
        red = np.clip(red_f, 0, 255).astype(np.uint8)
        green = np.clip(green_f, 0, 255).astype(np.uint8)
        blue = np.clip(blue_f, 0, 255).astype(np.uint8)

        # Create RGB image from normal components
        normal_array = np.stack([red, green, blue], axis=-1)
        normal_image = Image.fromarray(normal_array, "RGB")

        if not save:
            return normal_image

        # Generate output path if not provided
        if output_path is None:
            if isinstance(bump_map, str):
                base_path = bump_map
            else:
                # If PIL Image was passed, create generic output name
                base_path = f"bump_map.{DEFAULT_EXTENSION}"

            format_suffix = (
                "DirectX" if output_format.lower() == "directx" else "OpenGL"
            )
            output_path = cls.resolve_texture_filename(
                base_path,
                f"Normal_{format_suffix}",
                suffix=(
                    f"_intensity{intensity}".replace(".", "p")
                    if intensity != 1.0
                    else None
                ),
            )

        # Save the normal map
        normal_image.save(output_path, **kwargs)

        return output_path

    @classmethod
    def extract_gloss_from_spec(
        cls, specular_map: str, channel: str = "A"
    ) -> Union["Image.Image", None]:
        """Extracts gloss from a specific channel in the specular map.

        Attempts:
        1. Extracts specified channel (default: Alpha).
        2. If missing or empty, normalizes grayscale and enhances contrast.

        Parameters:
            specular_map: File path to the specular map.
            channel: One of "R", "G", "B", "A".

        Returns:
            Grayscale gloss map (L mode) if extracted, else None.
        """
        spec = ImgUtils.ensure_image(specular_map)

        # Attempt channel extraction
        if channel.upper() in spec.getbands():
            gloss = spec.getchannel(channel.upper())
            if gloss.getextrema() != (0, 0):  # Ensure non-empty
                return gloss.convert("L")

        print(
            f"// Warning: No gloss found in '{channel}' channel; using normalized grayscale..."
        )
        spec_gray = spec.convert("L")
        spec_gray = ImageEnhance.Brightness(spec_gray).enhance(1.2)
        gloss = ImageOps.autocontrast(spec_gray)

        return gloss.convert("L")

    @classmethod
    def convert_spec_gloss_to_pbr(
        cls,
        specular_map: Union[str, "Image.Image"],
        glossiness_map: Union[str, "Image.Image"],
        diffuse_map: Union[str, "Image.Image"] = None,
        output_dir: str = None,
        convert_diffuse_to_albedo: bool = False,
        output_type: str = None,
        image_size: Optional[int] = None,
        optimize_bit_depth: bool = True,
        write_files: bool = False,
    ) -> Union[
        Tuple["Image.Image", "Image.Image", "Image.Image"], Tuple[str, str, str]
    ]:
        """Converts Specular/Glossiness maps to PBR Metal/Rough.

        Parameters:
            specular_map: File path or loaded Image of the specular texture.
            glossiness_map: File path or loaded Image of the glossiness (or estimated roughness).
            diffuse_map: (Optional) File path or loaded Image of the diffuse texture.
            output_dir: (Optional) Directory where converted textures will be saved.
            convert_diffuse_to_albedo: (Optional) If True, generates a true Albedo map.
            output_type: (Optional) Desired output format (e.g., PNG, TGA). If None, keeps original.
            image_size: (Optional[int]) Target max dimension for output maps. If set and
                larger than current, images will be downscaled to this size while preserving aspect.
                If None, maintain original sizes.
            optimize_bit_depth: (Optional) If True, adjusts bit depth based on the map type.
            write_files: (Optional) If True, saves the images and returns file paths.

        Returns:
            Tuple of (BaseColor, Metallic, Roughness) images or file paths depending on `write_files`.
        """
        spec = ImgUtils.ensure_image(specular_map, "RGB")
        gloss = ImgUtils.ensure_image(glossiness_map, "L")
        diffuse = ImgUtils.ensure_image(diffuse_map, "RGB") if diffuse_map else None

        metallic = cls.create_metallic_from_spec(specular_map)
        base_color = cls.create_base_color_from_spec(diffuse, spec, metallic)
        roughness = cls.create_roughness_from_spec(spec, gloss)

        if convert_diffuse_to_albedo:
            base_color = cls.convert_base_color_to_albedo(base_color, metallic)

        if optimize_bit_depth:
            base_color = ImgUtils.set_bit_depth(base_color, "Base_Color")
            metallic = ImgUtils.set_bit_depth(metallic, "Metallic")
            roughness = ImgUtils.set_bit_depth(roughness, "Roughness")

        # Optional downscale to target max dimension while preserving original if not requested
        if isinstance(image_size, int) and image_size > 0:
            if max(base_color.size) > image_size:
                base_color = ImgUtils.resize_image(base_color, image_size, image_size)
            if max(metallic.size) > image_size:
                metallic = ImgUtils.resize_image(metallic, image_size, image_size)
            if max(roughness.size) > image_size:
                roughness = ImgUtils.resize_image(roughness, image_size, image_size)

        if not write_files:
            return base_color, metallic, roughness

        if output_dir is None:
            output_dir = (
                os.path.dirname(specular_map)
                if isinstance(specular_map, str)
                else os.getcwd()
            )
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        base_color_type = "Albedo" if convert_diffuse_to_albedo else "Base_Color"
        base_color_file = cls.resolve_texture_filename(
            specular_map, base_color_type, ext=output_type
        )
        metallic_file = cls.resolve_texture_filename(
            specular_map, "Metallic", ext=output_type
        )
        roughness_file = cls.resolve_texture_filename(
            specular_map, "Roughness", ext=output_type
        )

        base_color.save(base_color_file)
        metallic.save(metallic_file)
        roughness.save(roughness_file)

        print(
            f"PBR Conversion complete. Files saved:\n- {base_color_file}\n- {metallic_file}\n- {roughness_file}"
        )
        return base_color_file, metallic_file, roughness_file

    @classmethod
    def create_base_color_from_spec(
        cls,
        diffuse: Union[str, "Image.Image"],
        spec: Union[str, "Image.Image"],
        metalness: Union[str, "Image.Image"],
        conserve_energy: bool = True,
        metal_darkening: float = 0.22,
    ) -> "Image.Image":
        """Computes Base Color from Specular workflow with better metal handling.

        Parameters:
            diffuse (str/Image.Image): Diffuse map (RGB) or None.
            spec (str/Image.Image): Specular map (RGB).
            metalness (str/Image.Image): Metalness map (L mode grayscale).
            conserve_energy (bool, optional): Adjusts base color to balance PBR energy conservation.
            metal_darkening (float, optional): Strength of metal darkening (higher = darker metals).

        Returns:
            Image.Image: Base Color map (RGB).
        """
        spec = np.array(ImgUtils.ensure_image(spec, "RGB"), dtype=np.float32) / 255.0
        metalness = (
            np.array(ImgUtils.ensure_image(metalness, "L"), dtype=np.float32) / 255.0
        )

        if diffuse:
            diffuse = (
                np.array(ImgUtils.ensure_image(diffuse, "RGB"), dtype=np.float32)
                / 255.0
            )
            base_color = (
                diffuse * (1 - metalness[..., None]) + spec * metalness[..., None]
            )
        else:
            base_color = spec * (1 - metalness[..., None])

        # Darken metal areas (Reduce brightness in metals)
        # NOTE: Standard PBR does not require darkening metals, but this can help
        # if the source specular map is too bright or contains baked lighting.
        if metal_darkening > 0:
            base_color = np.where(
                metalness[..., None] > 0.5,
                base_color * (1.0 - metal_darkening),
                base_color,
            )

        # Apply energy conservation fix
        # NOTE: This is an artistic tweak to boost metal brightness, not strict PBR.
        if conserve_energy:
            base_color = np.clip(
                base_color / (1.0 - 0.08 * metalness[..., None] + 1e-6), 0.0, 1.0
            )

        return Image.fromarray((base_color * 255).astype(np.uint8), mode="RGB")

    @classmethod
    def create_metallic_from_spec(
        cls,
        specular_map: Union[str, "Image.Image"],
        glossiness_map: Union[str, "Image.Image"] = None,
        threshold: int = 55,
        softness: float = 0.2,
    ) -> "Image.Image":
        """Creates a metallic map from a specular (and optional glossiness) map.

        Steps:
        1. Use gloss map if provided, or extract from spec.
        2. Compute metallic from spec using soft threshold.
        3. Refine metallic using gloss (if available).

        Returns:
            Image.Image: Metallic map (L mode).
        """
        spec_rgb = ImgUtils.ensure_image(specular_map, "RGB")
        spec_lum = np.array(spec_rgb.convert("L"), dtype=np.float32) / 255.0

        # Step 1: Get gloss
        if glossiness_map:
            gloss = (
                np.array(ImgUtils.ensure_image(glossiness_map, "L"), dtype=np.float32)
                / 255.0
            )
            print("// Using gloss map to refine metallic computation.")
        else:
            gloss_img = cls.extract_gloss_from_spec(specular_map)
            gloss = np.array(gloss_img, dtype=np.float32) / 255.0 if gloss_img else None
            if gloss is not None:
                print("// Extracted gloss from specular map.")
            else:
                print("// No valid gloss map found; using spec only.")

        # Step 2: Base metallic estimate
        metallic = np.clip((spec_lum - (threshold / 255.0)) / softness, 0.0, 1.0)

        # Step 3: Refine with gloss
        if gloss is not None:
            metallic *= 1.0 - gloss  # Reduce metallic in high-gloss regions

        return Image.fromarray((metallic * 255).astype(np.uint8), mode="L")

    @classmethod
    def create_roughness_from_spec(
        cls,
        specular_map: Union[str, "Image.Image"],
        glossiness_map: Union[str, "Image.Image"] = None,
    ) -> "Image.Image":
        """Estimates roughness from a specular map.

        Steps:
        1. **If glossiness_map is provided, use it directly**.
        2. **If gloss is missing, attempt to extract it from the spec map**.
        3. **Convert gloss to roughness following industry PBR standards**.

        Parameters:
            specular_map (str/Image.Image): Specular texture file or image.
            glossiness_map (str/Image.Image, optional): Glossiness texture file or image.

        Returns:
            Image.Image: Roughness map (L mode grayscale).
        """
        spec = ImgUtils.ensure_image(specular_map, "RGB")

        # Step 1: Use provided gloss map or extract from specular
        gloss = (
            ImgUtils.ensure_image(glossiness_map, "L")
            if glossiness_map
            else cls.extract_gloss_from_spec(specular_map)
        )
        if not gloss:
            print(
                "// No valid gloss map found; estimating roughness directly from spec."
            )
            spec_gray = spec.convert("L")
            gloss = ImageOps.autocontrast(spec_gray)

        # Step 2: Convert glossiness to roughness
        gloss = np.array(gloss, dtype=np.float32) / 255.0
        roughness = 1.0 - gloss  # Direct inversion

        # Step 3: Apply gamma correction (for perceptual accuracy)
        gamma = 2.2  # Industry standard
        roughness = roughness**gamma

        # Step 4: Normalize roughness to maintain balanced shading
        roughness = np.clip(roughness, 0.0, 1.0)

        return Image.fromarray((roughness * 255).astype(np.uint8), mode="L")

    @classmethod
    def convert_base_color_to_albedo(
        cls, base_color: "Image.Image", metalness: "Image.Image"
    ) -> "Image.Image":
        """Converts a Base Color map to a true Albedo map by:

        - Removing baked reflections.
        - Setting metallic areas to black.
        - Normalizing colors for PBR consistency.

        Parameters:
            base_color: PIL Image (Base Color map).
            metalness: PIL Image (Grayscale Metalness map).

        Returns:
            albedo: PIL Image (True Albedo map).
        """
        base_color = ImgUtils.ensure_image(base_color)
        original_mode = base_color.mode

        # Ensure we have at least RGB
        if base_color.mode not in ["RGB", "RGBA"]:
            base_color = base_color.convert("RGB")

        metalness = ImgUtils.ensure_image(metalness, "L")

        # Convert metalness to grayscale and threshold (Metal = 1, Non-Metal = 0)
        # Metal (>128) -> 255 (White)
        # Non-Metal (<=128) -> 0 (Black)
        metal_mask = metalness.point(lambda p: 255 if p > 128 else 0)

        # Create a black image for metals
        # Match base color mode (RGB or RGBA)
        black_image = Image.new(
            base_color.mode,
            base_color.size,
            (0, 0, 0, 0) if "A" in base_color.mode else (0, 0, 0),
        )
        # Mask 0 (Non-Metal) -> Uses base_color
        albedo = Image.composite(black_image, base_color, metal_mask)

        return albedo

    @staticmethod
    def get_converted_map(map_type: str, available: dict) -> Optional[Any]:
        """Get the converted map based on the given map type and available maps.

        Parameters:
            map_type (str): The type of map to convert.
            available (dict): A dictionary of available maps.
                Keys are map types and values are the corresponding images.
                Example: {"Base_Color": image, "Roughness": image, ...}
        Returns:
            Optional[Any]: The converted map or None if not available.
        """
        # Smoothness <-> Roughness
        if map_type == "Smoothness" and "Roughness" in available:
            rough = available["Roughness"]
            return ImgUtils.invert_grayscale_image(rough)
        if map_type == "Roughness" and "Smoothness" in available:
            smooth = available["Smoothness"]
            return ImgUtils.invert_grayscale_image(smooth)
        # Glossiness <-> Roughness
        if map_type == "Glossiness" and "Roughness" in available:
            rough = available["Roughness"]
            return ImgUtils.invert_grayscale_image(rough)
        if map_type == "Roughness" and "Glossiness" in available:
            gloss = available["Glossiness"]
            return ImgUtils.invert_grayscale_image(gloss)
        # Glossiness <-> Smoothness
        if map_type == "Smoothness" and "Glossiness" in available:
            gloss = available["Glossiness"]
            return ImgUtils.invert_grayscale_image(gloss)
        if map_type == "Glossiness" and "Smoothness" in available:
            smooth = available["Smoothness"]
            return ImgUtils.invert_grayscale_image(smooth)
        # AO from Base_Color
        if map_type == "Ambient_Occlusion" and "Base_Color" in available:
            color = available["Base_Color"]
            return ImgUtils.ensure_image(color, "L")
        # Normal DirectX <-> OpenGL
        if map_type == "Normal_DirectX" and "Normal_OpenGL" in available:
            return MapFactory.convert_normal_map_format(
                available["Normal_OpenGL"], target_format="directx", save=False
            )
        if map_type == "Normal_OpenGL" and "Normal_DirectX" in available:
            return MapFactory.convert_normal_map_format(
                available["Normal_DirectX"], target_format="opengl", save=False
            )
        return None

    @classmethod
    def pack_msao_texture(
        cls,
        metallic_map_path: str,
        ao_map_path: Optional[str],
        alpha_map_path: Optional[str],
        detail_map_path: Optional[str] = None,
        output_dir: str = None,
        suffix: str = "_MSAO",
        invert_alpha: bool = False,
        output_path: str = None,
        save: bool = True,
    ) -> Union[str, "Image.Image"]:
        """Packs Metallic (R), AO (G), Detail (B), and Smoothness/Roughness (A) into a single MSAO texture.

        Parameters:
            metallic_map_path (str): Path to the metallic texture map.
            ao_map_path (str): Path to the ambient occlusion texture map. Can be None (fills with white).
            alpha_map_path (str): Path to the smoothness or roughness texture map. Can be None (fills with white).
            detail_map_path (str, optional): Path to the detail mask map. Can be None (fills with black).
            output_dir (str, optional): Output directory. If None, uses metallic map directory.
            suffix (str, optional): Suffix for the output file name.
            invert_alpha (bool, optional): If True, inverts the alpha channel (roughness to smoothness).
            output_path (str, optional): Explicit output path. Overrides output_dir/suffix logic.
            save (bool, optional): If True, saves to disk. If False, returns PIL Image.

        Returns:
            str | Image.Image: Path to the packed MSAO texture or PIL Image.
        """
        if isinstance(metallic_map_path, str):
            ImgUtils.assert_pathlike(metallic_map_path, "metallic_map_path")
        if ao_map_path and isinstance(ao_map_path, str):
            ImgUtils.assert_pathlike(ao_map_path, "ao_map_path")
        if alpha_map_path and isinstance(alpha_map_path, str):
            ImgUtils.assert_pathlike(alpha_map_path, "alpha_map_path")
        if detail_map_path and isinstance(detail_map_path, str):
            ImgUtils.assert_pathlike(detail_map_path, "detail_map_path")

        if save and output_path is None:
            # Derive base name from the first available map
            source_map = (
                metallic_map_path or ao_map_path or alpha_map_path or detail_map_path
            )
            if not source_map:
                raise ValueError("No source maps provided to derive output name")

            base_name = cls.get_base_texture_name(source_map)

            if output_dir is None:
                if isinstance(source_map, str):
                    output_dir = os.path.dirname(source_map)
                else:
                    raise ValueError(
                        "Cannot derive output directory from Image object; provide output_dir explicitly"
                    )
            elif not os.path.isdir(output_dir):
                raise ValueError(
                    f"The specified output directory '{output_dir}' is not valid."
                )

            output_path = os.path.join(
                output_dir, f"{base_name}{suffix}.{DEFAULT_EXTENSION}"
            )
        elif not save:
            output_path = None

        # Pack channels using the existing pack_channels method
        return ImgUtils.pack_channels(
            channel_files={
                "R": metallic_map_path,
                "G": ao_map_path,
                "B": detail_map_path,
                "A": alpha_map_path,
            },
            output_path=output_path,
            out_mode="RGBA",
            invert_channels=["A"] if invert_alpha else None,
            fill_values={
                "G": 255,
                "B": 0,
                "A": 255,
            },  # AO=White, Detail=Black, Alpha=White
            save=save,
        )

    @classmethod
    def convert_smoothness_to_roughness(
        cls, smoothness_path: str, output_dir: str = None, save: bool = True, **kwargs
    ) -> Union[str, "Image.Image"]:
        """Convert a Smoothness map to a Roughness map by inverting the grayscale values.

        Smoothness (0=rough, 255=smooth) becomes Roughness (0=smooth, 255=rough).

        Parameters:
            smoothness_path (str): Path to the smoothness texture map.
            output_dir (str, optional): Output directory. If None, uses smoothness map directory.
            save (bool): Whether to save the image to disk. Defaults to True.
            **kwargs: Additional arguments passed to PIL.Image.save (e.g., optimize=True).

        Returns:
            str | PIL.Image.Image: Path to the converted roughness map if saved, else the PIL Image object.
        """
        if isinstance(smoothness_path, str):
            ImgUtils.assert_pathlike(smoothness_path, "smoothness_path")
            if not os.path.exists(smoothness_path):
                raise FileNotFoundError(f"Input file not found: {smoothness_path}")

        # Load and invert the smoothness map
        smoothness_image = ImgUtils.ensure_image(smoothness_path, "L")
        roughness_image = ImgUtils.invert_grayscale_image(smoothness_image)

        if not save:
            return roughness_image

        if not isinstance(smoothness_path, str):
            raise ValueError(
                "Input must be a file path when save=True, or provide output_dir/name handling (not implemented for Image input)."
            )

        # Generate output path
        base_name = cls.get_base_texture_name(smoothness_path)

        if output_dir is None:
            output_dir = os.path.dirname(smoothness_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        # Get original extension
        original_ext = os.path.splitext(smoothness_path)[1]
        output_path = os.path.join(output_dir, f"{base_name}_Roughness{original_ext}")

        # Save the roughness map
        roughness_image.save(output_path, **kwargs)

        return output_path

    @classmethod
    def convert_roughness_to_smoothness(
        cls, roughness_path: str, output_dir: str = None, save: bool = True, **kwargs
    ) -> Union[str, "Image.Image"]:
        """Convert a Roughness map to a Smoothness map by inverting the grayscale values.

        Roughness (0=smooth, 255=rough) becomes Smoothness (0=rough, 255=smooth).

        Parameters:
            roughness_path (str): Path to the roughness texture map.
            output_dir (str, optional): Output directory. If None, uses roughness map directory.
            save (bool): Whether to save the image to disk. Defaults to True.
            **kwargs: Additional arguments passed to PIL.Image.save (e.g., optimize=True).

        Returns:
            str | PIL.Image.Image: Path to the converted smoothness map if saved, else the PIL Image object.
        """
        if isinstance(roughness_path, str):
            ImgUtils.assert_pathlike(roughness_path, "roughness_path")
            if not os.path.exists(roughness_path):
                raise FileNotFoundError(f"Input file not found: {roughness_path}")

        # Load and invert the roughness map
        roughness_image = ImgUtils.ensure_image(roughness_path, "L")
        smoothness_image = ImgUtils.invert_grayscale_image(roughness_image)

        if not save:
            return smoothness_image

        if not isinstance(roughness_path, str):
            raise ValueError(
                "Input must be a file path when save=True, or provide output_dir/name handling (not implemented for Image input)."
            )

        # Generate output path
        base_name = cls.get_base_texture_name(roughness_path)

        if output_dir is None:
            output_dir = os.path.dirname(roughness_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        # Get original extension
        original_ext = os.path.splitext(roughness_path)[1]
        output_path = os.path.join(output_dir, f"{base_name}_Smoothness{original_ext}")

        # Save the smoothness map
        smoothness_image.save(output_path, **kwargs)

        return output_path

    @classmethod
    def unpack_orm_texture(
        cls,
        orm_map_path: str,
        output_dir: str = None,
        ao_suffix: str = "_AO",
        roughness_suffix: str = "_Roughness",
        metallic_suffix: str = "_Metallic",
        invert_roughness: bool = False,
        save: bool = True,
        **kwargs,
    ) -> Union[
        Tuple[str, str, str], Tuple["Image.Image", "Image.Image", "Image.Image"]
    ]:
        """Unpacks AO (R), Roughness (G), and Metallic (B) maps from a combined ORM texture."""
        channel_config = {
            "R": {"suffix": ao_suffix},
            "G": {"suffix": roughness_suffix, "invert": invert_roughness},
            "B": {"suffix": metallic_suffix},
        }

        results = ImgUtils.extract_channels(
            orm_map_path, channel_config, output_dir=output_dir, save=save, **kwargs
        )

        if save:
            return results.get("R"), results.get("G"), results.get("B")
        else:
            return results.get("R"), results.get("G"), results.get("B")

    @classmethod
    def unpack_msao_texture(
        cls,
        msao_map_path: str,
        output_dir: str = None,
        metallic_suffix: str = "_Metallic",
        ao_suffix: str = "_AO",
        smoothness_suffix: str = "_Smoothness",
        invert_smoothness: bool = False,
        save: bool = True,
        **kwargs,
    ) -> Union[
        Tuple[str, str, str], Tuple["Image.Image", "Image.Image", "Image.Image"]
    ]:
        """Unpacks Metallic (R), AO (G), and Smoothness (A) maps from a combined MSAO texture."""
        channel_config = {
            "R": {"suffix": metallic_suffix},
            "G": {"suffix": ao_suffix},
            "A": {"suffix": smoothness_suffix, "invert": invert_smoothness},
        }

        results = ImgUtils.extract_channels(
            msao_map_path, channel_config, output_dir=output_dir, save=save, **kwargs
        )

        if save:
            return results.get("R"), results.get("G"), results.get("A")
        else:
            return results.get("R"), results.get("G"), results.get("A")

    @classmethod
    def unpack_albedo_transparency(
        cls,
        albedo_map_path: str,
        output_dir: str = None,
        base_color_suffix: str = "_BaseColor",
        opacity_suffix: str = "_Opacity",
        save: bool = True,
        **kwargs,
    ) -> Union[Tuple[str, str], Tuple["Image.Image", "Image.Image"]]:
        """Unpacks Base Color (RGB) and Opacity (A) from an Albedo+Transparency map."""
        channel_config = {
            "RGB": {"suffix": base_color_suffix},
            "A": {"suffix": opacity_suffix},
        }

        results = ImgUtils.extract_channels(
            albedo_map_path, channel_config, output_dir=output_dir, save=save, **kwargs
        )

        if save:
            return results.get("RGB"), results.get("A")
        else:
            return results.get("RGB"), results.get("A")

    @classmethod
    def unpack_metallic_smoothness(
        cls,
        map_path: str,
        output_dir: str = None,
        metallic_suffix: str = "_Metallic",
        smoothness_suffix: str = "_Smoothness",
        invert_smoothness: bool = False,
        save: bool = True,
        **kwargs,
    ) -> Union[Tuple[str, str], Tuple["Image.Image", "Image.Image"]]:
        """Unpacks Metallic (RGB) and Smoothness (A) from a combined map."""
        channel_config = {
            "RGB": {"suffix": metallic_suffix},
            "A": {"suffix": smoothness_suffix, "invert": invert_smoothness},
        }

        results = ImgUtils.extract_channels(
            map_path, channel_config, output_dir=output_dir, save=save, **kwargs
        )

        if save:
            return results.get("RGB"), results.get("A")
        else:
            return results.get("RGB"), results.get("A")

    @classmethod
    def unpack_specular_gloss(
        cls,
        map_path: str,
        output_dir: str = None,
        specular_suffix: str = "_Specular",
        gloss_suffix: str = "_Glossiness",
        invert_gloss: bool = False,
        save: bool = True,
        **kwargs,
    ) -> Union[Tuple[str, str], Tuple["Image.Image", "Image.Image"]]:
        """Unpacks Specular (RGB) and Glossiness (A) from a combined map."""
        channel_config = {
            "RGB": {"suffix": specular_suffix},
            "A": {"suffix": gloss_suffix, "invert": invert_gloss},
        }

        results = ImgUtils.extract_channels(
            map_path, channel_config, output_dir=output_dir, save=save, **kwargs
        )

        if save:
            return results.get("RGB"), results.get("A")
        else:
            return results.get("RGB"), results.get("A")


# Initialize the registry with the factory class
MapFactory._conversion_registry.add_plugin(MapFactory)
