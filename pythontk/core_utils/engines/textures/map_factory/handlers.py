# !/usr/bin/python
# coding=utf-8
"""Workflow handlers (Strategy pattern) for the texture MapFactory.

Each handler decides whether it applies to a given ``TextureProcessor`` context
and, if so, produces one or more output maps. Split out of the monolithic
``map_factory`` module.

``MapFactory`` is late-bound by this package's ``__init__`` (handlers call its
stateless primitives at runtime).
"""
from abc import ABC, abstractmethod
from typing import List, Optional

# From this package:
from pythontk.img_utils._img_utils import ImgUtils
from pythontk.core_utils.engines.textures.map_registry import MapRegistry
from .processor import TextureProcessor

# Late-bound by the package __init__ to break the runtime import cycle
# with MapFactory's primitive library.
MapFactory = None  # type: ignore


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


class MRAOMapHandler(WorkflowHandler):
    """Handles MRAO packing (Metallic R, Roughness G, AO B by default).

    Mirror of :class:`ORMMapHandler` for pipelines that prefer the Unity-style
    M/R/AO channel order. Honours ``mrao_layout`` config (``"rgb"`` /
    ``"rgba"``) to optionally produce a 4-channel layout parallel to MSAO.
    """

    def can_handle(self, context: TextureProcessor) -> bool:
        # Legacy support: 'convert' implies packing
        # New support: 'pack' explicitly controls packing
        if not context.config.get("pack", context.config.get("convert", True)):
            return False

        # Explicitly disabled?
        if context.config.get("mrao_map") is False:
            return False

        if self.is_explicitly_requested(context, "MRAO"):
            return True

        return False

    def process(self, context: TextureProcessor) -> Optional[str]:
        if "MRAO" in context.inventory:
            return context.save_map(context.inventory["MRAO"], "MRAO")

        # Request the target type only — the conversion registry derives it
        # (e.g. Specular -> Metallic). Listing "Specular" as a preferred type
        # would return the raw spec file verbatim as metallic data.
        metallic = context.resolve_map("Metallic", allow_conversion=True)
        ao = context.resolve_map("Ambient_Occlusion", allow_conversion=False)

        # Resolve roughness with inversion tracking
        roughness = None
        invert = False
        roughness_map = context.resolve_map("Roughness", allow_conversion=False)
        if roughness_map:
            roughness = roughness_map
        else:
            smoothness = context.resolve_map(
                "Smoothness", "Glossiness", allow_conversion=False
            )
            if smoothness:
                roughness = smoothness
                invert = True
            else:
                # Last resort: derive via the broader conversion system
                roughness = context.resolve_map("Roughness", allow_conversion=True)

        if not any([metallic, roughness, ao]):
            return None

        if not metallic and not context.config.get("force_packed_maps", False):
            if context.logger:
                context.logger.warning(
                    "No metallic map for MRAO red channel"
                )
            return None

        if not roughness and not context.config.get("force_packed_maps", False):
            if context.logger:
                context.logger.warning(
                    "No roughness map for MRAO green channel"
                )
            return None

        layout = context.config.get("mrao_layout", "rgb")
        detail = None
        if layout == "rgba":
            detail = context.resolve_map(
                "Detail_Mask", "Detail", allow_conversion=False
            )

        try:
            mrao_image = MapFactory.pack_mrao_texture(
                metallic_map_path=metallic,
                roughness_map_path=roughness,
                ao_map_path=ao,
                detail_map_path=detail,
                layout=layout,
                invert_roughness=invert,
                save=False,
            )
            if context.logger:
                context.logger.info(
                    f"Created MRAO map (layout={layout})",
                    extra={"preset": "highlight"},
                )
            sources = [img for img in [metallic, roughness, ao, detail] if img]
            return context.save_map(mrao_image, "MRAO", source_images=sources)
        except Exception as e:
            if context.logger:
                context.logger.error(f"Error creating MRAO map: {str(e)}")
            return None

    def get_consumed_types(self) -> List[str]:
        return [
            "Metallic",
            "MRAO",
            "Roughness",
            "Smoothness",
            "Glossiness",
            "Ambient_Occlusion",
            "AO",
            "Specular",
            "Detail",
            "Detail_Mask",
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

        # Request the target type only — the conversion registry derives it
        # (e.g. Specular -> Metallic). Listing "Specular" as a preferred type
        # would return the raw spec file verbatim as metallic data.
        metallic = context.resolve_map("Metallic", allow_conversion=True)
        ao = context.resolve_map("Ambient_Occlusion", allow_conversion=False)

        # Get smoothness with inversion tracking
        smoothness = None
        invert = False

        # Try to resolve smoothness/roughness explicitly. When neither exists,
        # leave None — pack_msao_texture fills the alpha with neutral white;
        # substituting another map's data (e.g. metallic) would bake wrong
        # smoothness values into the Mask Map.
        smoothness_map = context.resolve_map(
            "Smoothness", "Glossiness", allow_conversion=False
        )
        if smoothness_map:
            smoothness = smoothness_map
        else:
            roughness_map = context.resolve_map("Roughness", allow_conversion=True)
            if roughness_map:
                smoothness = roughness_map
                invert = True

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

        layout = context.config.get("mask_map_layout", "rgba")
        detail = None
        if layout == "rgba":
            detail = context.resolve_map(
                "Detail_Mask", "Detail", allow_conversion=False
            )

        try:
            mask_map_image = MapFactory.pack_msao_texture(
                metallic_map_path=metallic,
                ao_map_path=ao,
                alpha_map_path=smoothness,
                detail_map_path=detail,
                invert_alpha=invert,
                layout=layout,
                save=False,
            )
            if context.logger:
                context.logger.info(
                    f"Created Unity HDRP Mask Map (layout={layout})",
                    extra={"preset": "highlight"},
                )
            sources = [img for img in [metallic, ao, smoothness, detail] if img]
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
        for map_name in registry.get_map_types():
            if self.is_explicitly_requested(context, map_name):
                if map_name not in context.used_maps:
                    map_def = registry.get(map_name)
                    if map_def and "Metallic_Smoothness" in map_def.output_fallbacks:
                        return True

        # Implicit Logic: Disabled to prevent greedy generation
        return False

    def process(self, context: TextureProcessor) -> Optional[str]:
        if "Metallic_Smoothness" in context.inventory:
            return context.save_map(
                context.inventory["Metallic_Smoothness"], "Metallic_Smoothness"
            )

        # Target type only — conversions derive Metallic from Specular etc.;
        # a raw spec file must never stand in for metallic data verbatim.
        metallic = context.resolve_map("Metallic", allow_conversion=True)
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
            roughness = context.resolve_map("Roughness", allow_conversion=True)
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
            return context.save_map(metallic, "Metallic", source_images=[metallic])

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

        # Request the target type only. Listing "Smoothness"/"Specular" as
        # preferred types would return those raw files verbatim under the
        # target name (an un-inverted Smoothness saved as "_Roughness");
        # the conversion registry performs the correct derivations.
        if "Metallic" not in context.used_maps:
            metallic = context.resolve_map("Metallic", allow_conversion=True)
            if metallic:
                output_maps.append(
                    context.save_map(metallic, "Metallic", source_images=[metallic])
                )
                context.mark_used("Metallic", "Specular")

        if "Roughness" not in context.used_maps:
            roughness = context.resolve_map("Roughness", allow_conversion=True)
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
                    if isinstance(base_color, str):
                        img = context.get_cached_image(base_color)
                    else:
                        img = base_color
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
        # Opacity/Transparency are deliberately absent: they are only consumed
        # when actually packed into Albedo_Transparency (marked in process());
        # otherwise a separate Opacity map must pass through to its own slot.
        return [
            "Base_Color",
            "Diffuse",
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
                    "Generated normal map from Bump/Height",
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
        # Bump/Height are deliberately absent: they are only consumed when a
        # normal map is actually generated from them (marked in process()).
        # Height drives its own engine slot (parallax/displacement), so
        # processing an existing normal map must not swallow it.
        return ["Normal", "Normal_OpenGL", "Normal_DirectX"]


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
        for map_name in registry.get_map_types():
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
