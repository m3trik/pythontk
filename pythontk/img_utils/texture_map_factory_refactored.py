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
from typing import List, Dict, Optional, Callable, Type, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from pythontk.img_utils._img_utils import ImgUtils


# =============================================================================
# Map Conversion Registry - DRY conversion logic
# =============================================================================


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
        self._conversions: List[MapConversion] = []
        self._register_default_conversions()

    def register(self, conversion: MapConversion):
        """Register a new conversion strategy."""
        self._conversions.append(conversion)
        # Sort by priority (higher first)
        self._conversions.sort(key=lambda c: c.priority, reverse=True)

    def get_conversions_for(self, target_type: str) -> List[MapConversion]:
        """Get all conversions that can produce target type."""
        return [c for c in self._conversions if c.target_type == target_type]

    def _register_default_conversions(self):
        """Register all standard PBR conversions."""
        # Metallic conversions
        self.register(
            MapConversion(
                target_type="Metallic",
                source_types=["Specular"],
                converter=lambda inv, ctx: self._convert_specular_to_metallic(
                    inv["Specular"], ctx
                ),
                priority=5,
            )
        )

        # Roughness conversions
        self.register(
            MapConversion(
                target_type="Roughness",
                source_types=["Smoothness"],
                converter=lambda inv, ctx: self._convert_smoothness_to_roughness(
                    inv["Smoothness"], ctx
                ),
                priority=10,
            )
        )
        self.register(
            MapConversion(
                target_type="Roughness",
                source_types=["Glossiness"],
                converter=lambda inv, ctx: self._convert_smoothness_to_roughness(
                    inv["Glossiness"], ctx
                ),
                priority=9,
            )
        )
        self.register(
            MapConversion(
                target_type="Roughness",
                source_types=["Specular"],
                converter=lambda inv, ctx: self._convert_specular_to_roughness(
                    inv["Specular"], ctx
                ),
                priority=5,
            )
        )

        # Smoothness conversions
        self.register(
            MapConversion(
                target_type="Smoothness",
                source_types=["Roughness"],
                converter=lambda inv, ctx: self._convert_roughness_to_smoothness(
                    inv["Roughness"], ctx
                ),
                priority=10,
            )
        )

        # Normal conversions
        self.register(
            MapConversion(
                target_type="Normal_OpenGL",
                source_types=["Normal_DirectX"],
                converter=lambda inv, ctx: self._convert_dx_to_gl(
                    inv["Normal_DirectX"], ctx
                ),
                priority=10,
            )
        )
        self.register(
            MapConversion(
                target_type="Normal_DirectX",
                source_types=["Normal_OpenGL"],
                converter=lambda inv, ctx: self._convert_gl_to_dx(
                    inv["Normal_OpenGL"], ctx
                ),
                priority=10,
            )
        )
        self.register(
            MapConversion(
                target_type="Normal",
                source_types=["Bump", "Height"],
                converter=lambda inv, ctx: self._convert_bump_to_normal(
                    inv.get("Bump") or inv["Height"], ctx
                ),
                priority=5,
            )
        )

    # Conversion implementations
    @staticmethod
    def _convert_specular_to_metallic(
        specular_path: str, context: "ProcessingContext"
    ) -> str:
        metallic_img = ImgUtils.create_metallic_from_spec(specular_path)
        metallic_path = os.path.join(
            context.output_dir, f"{context.base_name}_Metallic.{context.ext}"
        )
        metallic_img.save(metallic_path)
        context.log("Created metallic from specular")
        return metallic_path

    @staticmethod
    def _convert_smoothness_to_roughness(
        smoothness_path: str, context: "ProcessingContext"
    ) -> str:
        roughness_path = ImgUtils.convert_smoothness_to_roughness(
            smoothness_path, context.output_dir
        )
        context.log("Converted smoothness to roughness")
        return roughness_path

    @staticmethod
    def _convert_roughness_to_smoothness(
        roughness_path: str, context: "ProcessingContext"
    ) -> str:
        # Smoothness = 1 - Roughness, use same function with result awareness
        smoothness_path = roughness_path.replace("Roughness", "Smoothness")
        rough_img = ImgUtils.load_image(roughness_path)
        smooth_img = ImgUtils.invert_grayscale_image(rough_img)
        ImgUtils.save_image(smooth_img, smoothness_path)
        context.log("Converted roughness to smoothness")
        return smoothness_path

    @staticmethod
    def _convert_specular_to_roughness(
        specular_path: str, context: "ProcessingContext"
    ) -> str:
        rough_img = ImgUtils.create_roughness_from_spec(specular_path)
        roughness_path = os.path.join(
            context.output_dir, f"{context.base_name}_Roughness.{context.ext}"
        )
        rough_img.save(roughness_path)
        context.log("Created roughness from specular")
        return roughness_path

    @staticmethod
    def _convert_dx_to_gl(dx_path: str, context: "ProcessingContext") -> str:
        gl_path = ImgUtils.create_gl_from_dx(dx_path)
        context.log("Converted DirectX normal to OpenGL")
        return gl_path

    @staticmethod
    def _convert_gl_to_dx(gl_path: str, context: "ProcessingContext") -> str:
        dx_path = ImgUtils.create_dx_from_gl(gl_path)
        context.log("Converted OpenGL normal to DirectX")
        return dx_path

    @staticmethod
    def _convert_bump_to_normal(bump_path: str, context: "ProcessingContext") -> str:
        normal_path = ImgUtils.convert_bump_to_normal(
            bump_path, output_format=context.config.get("normal_type", "opengl").lower()
        )
        context.log("Generated normal from bump/height")
        return normal_path


# =============================================================================
# Processing Context - Shared state and utilities
# =============================================================================


@dataclass
class ProcessingContext:
    """Shared context for all processing operations."""

    inventory: Dict[str, str]
    config: Dict[str, Any]
    output_dir: str
    base_name: str
    ext: str
    callback: Callable
    conversion_registry: ConversionRegistry
    used_maps: set = field(default_factory=set)

    def log(self, message: str, level: str = "success"):
        """Unified logging with color coding."""
        colors = {
            "success": "rgb(100, 160, 100)",
            "warning": "rgb(200, 200, 100)",
            "error": "rgb(255, 100, 100)",
        }
        color = colors.get(level, colors["success"])
        self.callback(f'<br><hl style="color:{color};">{message}</hl>')

    def resolve_map(
        self, *preferred_types: str, allow_conversion: bool = True
    ) -> Optional[str]:
        """Intelligently resolve a map from inventory with fallback conversions.

        This is the DRY replacement for repeated "get X or convert from Y" logic.

        Args:
            *preferred_types: Ordered list of preferred map types
            allow_conversion: Whether to attempt conversions if direct match fails

        Returns:
            Path to resolved map or None
        """
        # Try direct matches first (in priority order)
        for map_type in preferred_types:
            if map_type in self.inventory:
                return self.inventory[map_type]

        # Try conversions if allowed
        if allow_conversion:
            for target_type in preferred_types:
                conversions = self.conversion_registry.get_conversions_for(target_type)
                for conversion in conversions:
                    # Check if we have all required source types
                    if all(src in self.inventory for src in conversion.source_types):
                        try:
                            result = conversion.converter(self.inventory, self)
                            # Cache the result in inventory
                            self.inventory[target_type] = result
                            return result
                        except Exception as e:
                            self.log(
                                f"Error converting to {target_type}: {str(e)}", "error"
                            )

        return None

    def mark_used(self, *map_types: str):
        """Mark map types as consumed."""
        self.used_maps.update(map_types)


# =============================================================================
# Workflow Handlers - Strategy Pattern
# =============================================================================


class WorkflowHandler(ABC):
    """Abstract base for workflow-specific map processing."""

    @abstractmethod
    def can_handle(self, config: Dict[str, Any]) -> bool:
        """Check if this handler should process the workflow."""
        pass

    @abstractmethod
    def process(self, context: ProcessingContext) -> Optional[str]:
        """Process and return the output map path."""
        pass

    @abstractmethod
    def get_consumed_types(self) -> List[str]:
        """Return list of map types this handler consumes."""
        pass


class ORMMapHandler(WorkflowHandler):
    """Handles Unreal Engine / glTF ORM packing."""

    def can_handle(self, config: Dict[str, Any]) -> bool:
        return config.get("orm_map", False)

    def process(self, context: ProcessingContext) -> Optional[str]:
        # Resolve required components with smart fallbacks
        ao = context.resolve_map("Ambient_Occlusion", "AO", allow_conversion=False)
        roughness = context.resolve_map(
            "Roughness", "Smoothness", "Glossiness", allow_conversion=True
        )
        metallic = context.resolve_map("Metallic", "Specular", allow_conversion=True)

        if not ao:
            context.log("No AO map, using white for ORM red channel", "warning")

        if not roughness:
            context.log("No roughness map for ORM green channel", "warning")
            return None

        if not metallic:
            context.log("No metallic map for ORM blue channel", "warning")
            return None

        try:
            orm_map = ImgUtils.pack_channels(
                channel_files={"R": ao, "G": roughness, "B": metallic},
                output_path=os.path.join(
                    context.output_dir, f"{context.base_name}_ORM.{context.ext}"
                ),
                fill_values={"R": 255} if not ao else None,
            )
            context.log("Created Unreal/glTF ORM map")
            return orm_map
        except Exception as e:
            context.log(f"Error creating ORM map: {str(e)}", "error")
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

    def can_handle(self, config: Dict[str, Any]) -> bool:
        return config.get("mask_map", False)

    def process(self, context: ProcessingContext) -> Optional[str]:
        if "MSAO" in context.inventory:
            return context.inventory["MSAO"]

        metallic = context.resolve_map("Metallic", "Specular", allow_conversion=True)
        if not metallic:
            context.log("No metallic map for Mask Map", "warning")
            return None

        ao = (
            context.resolve_map("Ambient_Occlusion", allow_conversion=False) or metallic
        )

        # Get smoothness with inversion tracking
        smoothness = None
        invert = False
        if "Smoothness" in context.inventory:
            smoothness = context.inventory["Smoothness"]
        elif "Glossiness" in context.inventory:
            smoothness = context.inventory["Glossiness"]
        elif "Roughness" in context.inventory:
            smoothness = context.inventory["Roughness"]
            invert = True
        else:
            smoothness = metallic

        try:
            mask_map = ImgUtils.pack_msao_texture(
                metallic_map_path=metallic,
                ao_map_path=ao,
                alpha_map_path=smoothness,
                output_dir=context.output_dir,
                suffix="_MaskMap",
                invert_alpha=invert,
            )
            context.log("Created Unity HDRP Mask Map")
            return mask_map
        except Exception as e:
            context.log(f"Error creating Mask Map: {str(e)}", "error")
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

    def can_handle(self, config: Dict[str, Any]) -> bool:
        return config.get("metallic_smoothness", False)

    def process(self, context: ProcessingContext) -> Optional[str]:
        if "Metallic_Smoothness" in context.inventory:
            return context.inventory["Metallic_Smoothness"]

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
            return metallic

        try:
            combined = ImgUtils.pack_smoothness_into_metallic(
                metallic, alpha_map, invert_alpha=invert
            )
            context.log("Packed smoothness into metallic")
            return combined
        except Exception as e:
            context.log(f"Error packing metallic/smoothness: {str(e)}", "error")
            return metallic

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

    def can_handle(self, config: Dict[str, Any]) -> bool:
        # This is the default/fallback handler
        return True

    def process(self, context: ProcessingContext) -> List[str]:
        """Returns list since this produces multiple maps."""
        output_maps = []

        metallic = context.resolve_map("Metallic", "Specular", allow_conversion=True)
        if metallic:
            output_maps.append(metallic)
            context.mark_used("Metallic", "Specular")

        roughness = context.resolve_map(
            "Roughness", "Smoothness", "Glossiness", "Specular", allow_conversion=True
        )
        if roughness:
            output_maps.append(roughness)
            context.mark_used("Roughness", "Smoothness", "Glossiness")

        return output_maps

    def get_consumed_types(self) -> List[str]:
        return ["Metallic", "Roughness", "Smoothness", "Glossiness", "Specular"]


class BaseColorHandler(WorkflowHandler):
    """Handles base color / albedo with optional packing."""

    def can_handle(self, config: Dict[str, Any]) -> bool:
        return True  # Always processes if base color exists

    def process(self, context: ProcessingContext) -> Optional[str]:
        if "Albedo_Transparency" in context.inventory:
            context.mark_used(
                "Albedo_Transparency",
                "Base_Color",
                "Diffuse",
                "Albedo",
                "Opacity",
                "Transparency",
            )
            return context.inventory["Albedo_Transparency"]

        base_color = context.resolve_map(
            "Base_Color", "Diffuse", "Albedo", allow_conversion=False
        )
        if not base_color:
            return None

        # Pack transparency if requested
        if context.config.get("albedo_transparency", False):
            opacity = context.resolve_map(
                "Opacity", "Transparency", allow_conversion=False
            )
            if opacity:
                try:
                    combined = ImgUtils.pack_transparency_into_albedo(
                        base_color, opacity
                    )
                    context.log("Packed transparency into albedo")
                    context.mark_used(
                        "Base_Color",
                        "Diffuse",
                        "Albedo",
                        "Opacity",
                        "Transparency",
                        "Albedo_Transparency",
                    )
                    return combined
                except Exception as e:
                    context.log(f"Error packing transparency: {str(e)}", "error")

        # Clean up base color if requested
        if context.config.get("cleanup_base_color", False):
            metallic = context.resolve_map("Metallic", allow_conversion=False)
            if metallic:
                try:
                    base_img = ImgUtils.load_image(base_color)
                    metallic_img = ImgUtils.load_image(metallic)
                    cleaned = ImgUtils.convert_base_color_to_albedo(
                        base_img, metallic_img
                    )
                    cleaned_path = os.path.join(
                        context.output_dir, f"{context.base_name}_Albedo.{context.ext}"
                    )
                    ImgUtils.save_image(cleaned, cleaned_path)
                    context.log("Cleaned base color to true albedo")
                    context.mark_used("Base_Color", "Diffuse", "Albedo")
                    return cleaned_path
                except Exception as e:
                    context.log(f"Error cleaning base color: {str(e)}", "error")

        context.mark_used("Base_Color", "Diffuse", "Albedo")
        return base_color

    def get_consumed_types(self) -> List[str]:
        return [
            "Base_Color",
            "Diffuse",
            "Albedo",
            "Opacity",
            "Transparency",
            "Albedo_Transparency",
        ]


class NormalMapHandler(WorkflowHandler):
    """Handles normal map format conversion."""

    def can_handle(self, config: Dict[str, Any]) -> bool:
        return True  # Always processes if normal exists

    def process(self, context: ProcessingContext) -> Optional[str]:
        target_format = context.config.get("normal_type", "OpenGL")
        target_key = f"Normal_{target_format}"

        # Try exact match, generic, converted, or generated
        normal = context.resolve_map(
            target_key,
            "Normal",
            f"Normal_{self._opposite(target_format)}",
            "Bump",
            "Height",
            allow_conversion=True,
        )

        if normal:
            context.mark_used(
                "Normal", "Normal_OpenGL", "Normal_DirectX", "Bump", "Height"
            )

        return normal

    @staticmethod
    def _opposite(format_type: str) -> str:
        return "DirectX" if format_type == "OpenGL" else "OpenGL"

    def get_consumed_types(self) -> List[str]:
        return ["Normal", "Normal_OpenGL", "Normal_DirectX", "Bump", "Height"]


# =============================================================================
# Main Factory - Simplified and extensible
# =============================================================================


class TextureMapFactory:
    """Refactored factory with pluggable workflow system."""

    _conversion_registry = ConversionRegistry()
    _workflow_handlers: List[Type[WorkflowHandler]] = [
        BaseColorHandler,
        ORMMapHandler,
        MaskMapHandler,
        MetallicSmoothnessHandler,
        SeparateMetallicRoughnessHandler,
        NormalMapHandler,
    ]

    @classmethod
    def register_handler(cls, handler_class: Type[WorkflowHandler]):
        """Register a custom workflow handler (extensibility)."""
        cls._workflow_handlers.insert(-2, handler_class)  # Before default handlers

    @classmethod
    def register_conversion(cls, conversion: MapConversion):
        """Register a custom map conversion (extensibility)."""
        cls._conversion_registry.register(conversion)

    @staticmethod
    def prepare_maps(
        textures: List[str],
        workflow_config: dict,
        callback: Callable = print,
    ) -> List[str]:
        """Main factory method - now much simpler."""
        # Build inventory
        map_inventory = TextureMapFactory._build_map_inventory(textures)

        # Pre-process: Spec/Gloss conversion
        if workflow_config.get("convert_specgloss_to_pbr", False):
            map_inventory = TextureMapFactory._convert_specgloss_workflow(
                map_inventory, workflow_config, callback
            )

        # Create processing context
        first_map = next(iter(map_inventory.values()), None)
        if not first_map:
            return textures if textures else []

        context = ProcessingContext(
            inventory=map_inventory,
            config=workflow_config,
            output_dir=os.path.dirname(first_map),
            base_name=ImgUtils.get_base_texture_name(first_map),
            ext=workflow_config.get("output_extension", "png"),
            callback=callback,
            conversion_registry=TextureMapFactory._conversion_registry,
        )

        # Process through workflow handlers
        output_maps = []
        for handler_class in TextureMapFactory._workflow_handlers:
            handler = handler_class()
            if handler.can_handle(workflow_config):
                result = handler.process(context)
                if result:
                    if isinstance(result, list):
                        output_maps.extend(result)
                    else:
                        output_maps.append(result)
                    context.mark_used(*handler.get_consumed_types())
                    # Most handlers are mutually exclusive (except default)
                    if handler_class not in [
                        SeparateMetallicRoughnessHandler,
                        BaseColorHandler,
                        NormalMapHandler,
                    ]:
                        break  # Stop after first match for packed workflows

        # Pass through unconsumed maps
        passthrough_types = [
            "Emissive",
            "Emission",
            "Ambient_Occlusion",
            "Height",
            "Displacement",
        ]
        for map_type in passthrough_types:
            if map_type in map_inventory and map_type not in context.used_maps:
                output_maps.append(map_inventory[map_type])

        return output_maps if output_maps else textures

    @staticmethod
    def _build_map_inventory(textures: List[str]) -> Dict[str, str]:
        """Build map inventory using ImgUtils."""
        inventory = {}
        for map_type in ImgUtils.map_types.keys():
            maps = ImgUtils.filter_images_by_type(textures, map_type)
            if maps:
                inventory[map_type] = maps[0]
        return inventory

    @staticmethod
    def _convert_specgloss_workflow(
        inventory: Dict[str, str], config: dict, callback: Callable
    ) -> Dict[str, str]:
        """Convert Spec/Gloss workflow to PBR."""
        has_specular = "Specular" in inventory
        has_glossiness = "Glossiness" in inventory or "Smoothness" in inventory
        has_diffuse = "Diffuse" in inventory

        if not (has_specular and (has_glossiness or has_diffuse)):
            return inventory

        try:
            # Get output params from config
            first_map = next(iter(inventory.values()))
            output_dir = os.path.dirname(first_map)

            base_color_path, metallic_path, roughness_path = (
                ImgUtils.convert_spec_gloss_to_pbr(
                    specular_map=inventory["Specular"],
                    glossiness_map=inventory.get("Glossiness")
                    or inventory.get("Smoothness"),
                    diffuse_map=inventory.get("Diffuse"),
                    output_dir=output_dir,
                    write_files=True,
                )
            )

            new_inventory = inventory.copy()
            new_inventory["Base_Color"] = base_color_path
            new_inventory["Metallic"] = metallic_path
            new_inventory["Roughness"] = roughness_path

            # Remove converted maps
            for key in ["Specular", "Glossiness", "Smoothness", "Diffuse"]:
                new_inventory.pop(key, None)

            callback(
                '<br><hl style="color:rgb(100, 160, 100);">Converted Spec/Gloss workflow to PBR Metal/Rough</hl>'
            )
            return new_inventory

        except Exception as e:
            callback(
                f'<br><hl style="color:rgb(255, 100, 100);">Error converting Spec/Gloss: {str(e)}</hl>'
            )
            return inventory
