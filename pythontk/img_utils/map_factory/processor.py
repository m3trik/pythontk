# !/usr/bin/python
# coding=utf-8
"""``TextureProcessor`` -- shared processing context for the MapFactory.

Holds the per-set inventory/config plus the cached image IO and save pipeline
that the workflow handlers operate on. Split out of the monolithic
``map_factory`` module.

``MapFactory`` is late-bound by this package's ``__init__`` (see the note there):
the processor calls MapFactory's stateless conversion primitives at runtime,
while MapFactory lists the handler classes at class-definition time, so a
top-level import here would form a cycle.
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

try:
    from PIL import Image
except ImportError:
    Image = None

if TYPE_CHECKING:
    from PIL import Image

# From this package:
from pythontk.img_utils._img_utils import ImgUtils
from pythontk.str_utils._str_utils import StrUtils
from pythontk.img_utils.map_registry import MapRegistry
from pythontk.img_utils.output_template import OutputTemplates
from .conversions import ConversionRegistry

# Constants -- single source of truth for the package (imported by _map_factory).
DEFAULT_EXTENSION = "png"  # Default extension for saved maps
ALPHA_EXTENSION = "png"  # Default extension for maps requiring alpha channel

# Late-bound by the package __init__ to break the runtime import cycle
# with MapFactory's primitive library.
MapFactory = None  # type: ignore


@dataclass
class TextureProcessor:
    """Shared context and processor for all map operations."""

    inventory: Dict[str, Union[str, "Image.Image"]]
    config: Dict[str, Any]
    output_dir: str
    base_name: str
    ext: Optional[str]
    conversion_registry: ConversionRegistry
    # When set (a WF profile key), per-map output format/bit-depth/compression is
    # resolved from the profile's template instead of the single global ``ext``.
    output_profile: Optional[str] = None
    logger: Any = None
    used_maps: set = field(default_factory=set)
    created_files: set = field(default_factory=set)
    _image_cache: dict = field(default_factory=dict)

    def get_cached_image(self, path: str) -> "Image.Image":
        """Load an image with caching to avoid redundant disk I/O.

        The cache is keyed by absolute path. Images loaded once are reused
        for subsequent calls (e.g. alpha checks then save_map pipeline).

        Parameters:
            path: File path to the image.

        Returns:
            PIL.Image.Image: The loaded image.
        """
        abs_path = os.path.abspath(path)
        if abs_path not in self._image_cache:
            self._image_cache[abs_path] = ImgUtils.ensure_image(path)
        # Return a copy to avoid mutation issues between consumers
        return self._image_cache[abs_path].copy()

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

        # Resolve per-map output format. When a profile is active, its template is
        # authoritative for ext / bit depth / compression; otherwise behave exactly
        # as before (single global ``self.ext``).
        spec = (
            OutputTemplates.resolve(map_type, self.output_profile)
            if self.output_profile
            else None
        )
        target_bit_depth = spec.bit_depth if spec else None
        target_compression = spec.compression if spec else None

        # Determine extension
        ext = spec.ext if spec else self.ext
        if not ext:
            if isinstance(image, str):
                ext = os.path.splitext(image)[1].lstrip(".")
            else:
                ext = DEFAULT_EXTENSION  # Fallback for generated images

        # Force PNG for maps requiring alpha if source was JPG
        if ext.lower() in ["jpg", "jpeg"] and map_type in [
            "MaskMap",
            "MSAO",
            "MRAO",
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
            if self.logger:
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
                if self.logger:
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

        # A profile demanding 16-bit or block compression can't be byte-copied from
        # an 8-bit/uncompressed source — force the re-encode path.
        needs_reencode = bool(target_compression) or bool(
            target_bit_depth and target_bit_depth >= 16
        )

        # Smart Copy: If input is a file and no optimization is needed, just copy
        if isinstance(image, str) and os.path.exists(image) and not needs_reencode:
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

            if should_optimize and not format_changed:
                # Only worth checking if we might skip optimization.
                # When format_changed is True, re-encoding is mandatory
                # so opening the file just to check is wasted I/O.
                try:
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

                        # If we don't need resize or mode change, skip re-encoding
                        if not needs_resize and not needs_mode_change:
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
        # Use image cache to avoid redundant disk I/O
        if should_optimize:
            if isinstance(image, str):
                img_obj = self.get_cached_image(image)
            else:
                img_obj = ImgUtils.ensure_image(image)

            # Determine if resizing is needed
            will_resize = False
            if allow_resize and max_size:
                width, height = img_obj.size
                if max(width, height) > max_size:
                    will_resize = True

            # 1. Depalettize ONLY if resizing
            if will_resize:
                img_obj = ImgUtils.depalettize_image(img_obj)

            # 2. Enforce Mode (skipped when the map type has no fixed mode,
            # e.g. MRAO which supports both 3- and 4-channel layouts).
            # When 16-bit output is requested for a grayscale map, don't downcast
            # an already-single-channel high-bit source ("I"/"I;16") to 8-bit "L":
            # that quantizes away the precision the profile asked for. It is already
            # single-channel and save_image's 16-bit path consumes it directly.
            map_def = MapRegistry().get(map_type)
            if map_def and map_def.mode:
                keep_high_bit_gray = (
                    target_bit_depth
                    and target_bit_depth >= 16
                    and map_def.mode == "L"
                    and img_obj.mode in ("I", "I;16")
                )
                if not keep_high_bit_gray:
                    img_obj = ImgUtils.enforce_mode(img_obj, map_def.mode)

            # 3. Resize
            if will_resize:
                from PIL import Image

                img_obj.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # 4. Save with optimization
            ImgUtils.save_image(
                img_obj,
                output_path,
                optimize=True,
                bit_depth=target_bit_depth,
                compression=target_compression,
            )

        else:
            # Re-encode needed (e.g. PIL Image input or extension mismatch)
            if isinstance(image, str):
                img_obj = self.get_cached_image(image)
            else:
                img_obj = ImgUtils.ensure_image(image)
            ImgUtils.save_image(
                img_obj,
                output_path,
                optimize=True,
                bit_depth=target_bit_depth,
                compression=target_compression,
            )

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
        # 1. Try direct matches first (in priority order). Truthiness, not
        # membership — unpack helpers cache None for a missing channel, and
        # that must read as absent rather than short-circuit resolution.
        for map_type in preferred_types:
            if self.inventory.get(map_type):
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
                            # Only cache a usable result — caching None would
                            # poison later lookups ("Roughness" in inventory
                            # reads as present) and shadow lower-priority
                            # conversions that could still succeed.
                            if result:
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
                        if self.inventory.get(fb):
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
        # Return cached if available (truthy — a failed unpack must not stick)
        if self.inventory.get("Metallic") and self.inventory.get("Smoothness"):
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
            self.inventory.get("Metallic")
            and self.inventory.get("AO")
            and self.inventory.get("Smoothness")
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

    def unpack_mrao(self, source_path: Union[str, "Image.Image"]) -> None:
        """Helper to unpack MRAO and cache results.

        Layout is detected from the source image mode (RGB → 3-channel,
        RGBA → 4-channel mirror of MSAO).
        """
        if (
            self.inventory.get("Metallic")
            and self.inventory.get("Roughness")
            and self.inventory.get("AO")
        ):
            return

        metallic_img, roughness_img, ao_img = MapFactory.unpack_mrao_texture(
            source_path, self.output_dir, optimize=False, save=False
        )

        self.inventory["Metallic"] = metallic_img
        self.inventory["Roughness"] = roughness_img
        self.inventory["AO"] = ao_img
        self.inventory["Ambient_Occlusion"] = self.inventory["AO"]
        if self.logger:
            self.logger.info(
                "Unpacked Metallic, Roughness, and AO from MRAO map",
                extra={"preset": "highlight"},
            )

    def get_metallic_from_mrao(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_mrao(source_path)
        return self.inventory["Metallic"]

    def get_roughness_from_mrao(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_mrao(source_path)
        return self.inventory["Roughness"]

    def get_smoothness_from_mrao(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_mrao(source_path)
        if not self.inventory.get("Roughness"):
            return None
        return self.convert_roughness_to_smoothness(self.inventory["Roughness"])

    def get_ao_from_mrao(
        self, source_path: Union[str, "Image.Image"]
    ) -> Union[str, "Image.Image"]:
        self.unpack_mrao(source_path)
        return self.inventory["AO"]

    def unpack_orm(self, source_path: Union[str, "Image.Image"]) -> None:
        """Helper to unpack ORM and cache results."""
        if (
            self.inventory.get("AO")
            and self.inventory.get("Roughness")
            and self.inventory.get("Metallic")
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
        if self.inventory.get("Base_Color") and self.inventory.get("Opacity"):
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
        # Resolve required components. Target types only — the conversion
        # registry derives them (Smoothness -> inverted Roughness, Specular ->
        # Metallic). Listing sources as preferred types would return the raw
        # files verbatim under the wrong semantics.
        ao = self.resolve_map("Ambient_Occlusion", "AO", allow_conversion=False)
        roughness = self.resolve_map("Roughness", allow_conversion=True)
        metallic = self.resolve_map("Metallic", allow_conversion=True)

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

    def create_mrao_map(
        self, inventory: Dict[str, Union[str, "Image.Image"]]
    ) -> "Image.Image":
        """Create MRAO (Metallic R / Roughness G / AO B) map from components.

        Honours ``mrao_layout`` in config: ``"rgb"`` (default, 3-channel
        industry-standard order) or ``"rgba"`` (4-channel mirror of MSAO with
        roughness in alpha).
        """
        metallic = self.resolve_map("Metallic", allow_conversion=True)
        ao = self.resolve_map("Ambient_Occlusion", "AO", allow_conversion=False)

        # Get roughness with inversion tracking. Truthiness (not `in`) so a
        # cached failed unpack (None) can't shadow a usable alternative.
        roughness = None
        invert = False
        if inventory.get("Roughness"):
            roughness = inventory["Roughness"]
        elif inventory.get("Smoothness"):
            roughness = inventory["Smoothness"]
            invert = True
        elif inventory.get("Glossiness"):
            roughness = inventory["Glossiness"]
            invert = True

        if not (metallic or roughness or ao):
            raise ValueError("Missing components for MRAO map")

        layout = self.config.get("mrao_layout", "rgb")
        detail = None
        if layout == "rgba":
            detail = self.resolve_map("Detail_Mask", "Detail", allow_conversion=False)

        mrao_img = MapFactory.pack_mrao_texture(
            metallic_map_path=metallic,
            roughness_map_path=roughness,
            ao_map_path=ao,
            detail_map_path=detail,
            output_dir=self.output_dir,
            suffix="_MRAO",
            invert_roughness=invert,
            layout=layout,
            save=False,
        )
        if self.logger:
            self.logger.info(
                f"Created MRAO map from components (layout={layout})",
                extra={"preset": "highlight"},
            )
        return mrao_img

    def create_mask_map(
        self, inventory: Dict[str, Union[str, "Image.Image"]]
    ) -> "Image.Image":
        """Create Mask Map (MSAO) from components.

        Honours ``mask_map_layout`` in config: ``"rgba"`` (default; HDRP Mask
        Map: R=Metallic, G=AO, B=Detail, A=Smoothness) or ``"rgb"`` (3-channel
        parallel to MRAO: R=Metallic, G=Smoothness, B=AO).
        """
        metallic = self.resolve_map("Metallic", allow_conversion=True)
        ao = self.resolve_map("Ambient_Occlusion", "AO", allow_conversion=False)

        # Get smoothness with inversion tracking. Truthiness (not `in`) so a
        # cached failed unpack (None) can't shadow a usable alternative.
        smoothness = None
        invert = False
        if inventory.get("Smoothness"):
            smoothness = inventory["Smoothness"]
        elif inventory.get("Glossiness"):
            smoothness = inventory["Glossiness"]
        elif inventory.get("Roughness"):
            smoothness = inventory["Roughness"]
            invert = True

        if not metallic and not ao and not smoothness:
            raise ValueError(
                "Missing components for Mask Map (need at least Metallic, AO, or Smoothness)"
            )

        layout = self.config.get("mask_map_layout", "rgba")
        detail = None
        if layout == "rgba":
            detail = self.resolve_map("Detail_Mask", "Detail", allow_conversion=False)

        mask_map = MapFactory.pack_msao_texture(
            metallic_map_path=metallic,
            ao_map_path=ao,
            alpha_map_path=smoothness,
            detail_map_path=detail,
            output_dir=self.output_dir,
            suffix="_MaskMap",
            invert_alpha=invert,
            layout=layout,
            save=False,
        )
        if self.logger:
            self.logger.info(
                f"Created Mask Map from components (layout={layout})",
                extra={"preset": "highlight"},
            )
        return mask_map

    def create_metallic_smoothness_map(
        self, inventory: Dict[str, Union[str, "Image.Image"]]
    ) -> "Image.Image":
        """Create Metallic-Smoothness map from components."""
        metallic = self.resolve_map("Metallic", allow_conversion=True)

        smoothness = None
        invert = False

        if inventory.get("Smoothness"):
            smoothness = inventory["Smoothness"]
        elif inventory.get("Glossiness"):
            smoothness = inventory["Glossiness"]
        elif inventory.get("Roughness"):
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
