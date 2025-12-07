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
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Callable, Type, Any, Union, Tuple
from collections import defaultdict

from pythontk.img_utils._img_utils import ImgUtils
from pythontk.file_utils._file_utils import FileUtils


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
        self._conversions: Dict[str, List[MapConversion]] = defaultdict(list)
        self._register_default_conversions()

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

        self._conversions[conversion.target_type].append(conversion)
        # Sort by priority (higher first)
        self._conversions[conversion.target_type].sort(
            key=lambda c: c.priority, reverse=True
        )

    def get_conversions_for(self, target_type: str) -> List[MapConversion]:
        """Get all conversions that can produce target type."""
        return self._conversions.get(target_type, [])

    def _register_default_conversions(self):
        """Register all standard PBR conversions."""
        # Metallic conversions
        self.register(
            "Metallic",
            "Specular",
            lambda inv, ctx: self._convert_specular_to_metallic(inv["Specular"], ctx),
            priority=5,
        )

        # Roughness conversions
        self.register(
            "Roughness",
            "Smoothness",
            lambda inv, ctx: self._convert_smoothness_to_roughness(
                inv["Smoothness"], ctx
            ),
            priority=10,
        )
        self.register(
            "Roughness",
            "Glossiness",
            lambda inv, ctx: self._convert_smoothness_to_roughness(
                inv["Glossiness"], ctx
            ),
            priority=9,
        )
        self.register(
            "Roughness",
            "Specular",
            lambda inv, ctx: self._convert_specular_to_roughness(inv["Specular"], ctx),
            priority=5,
        )

        # Glossiness conversions
        self.register(
            "Glossiness",
            "Specular",
            lambda inv, ctx: self._extract_gloss_from_spec(inv["Specular"], ctx),
            priority=5,
        )
        self.register(
            "Glossiness",
            "Roughness",
            lambda inv, ctx: self._convert_roughness_to_smoothness(
                inv["Roughness"], ctx
            ),  # Inverted Roughness = Smoothness ≈ Glossiness
            priority=9,
        )
        self.register(
            "Glossiness",
            "Smoothness",
            lambda inv, ctx: self._copy_map(inv["Smoothness"], "Glossiness", ctx),
            priority=10,
        )

        # Smoothness conversions
        self.register(
            "Smoothness",
            "Roughness",
            lambda inv, ctx: self._convert_roughness_to_smoothness(
                inv["Roughness"], ctx
            ),
            priority=10,
        )

        # Normal conversions
        self.register(
            "Normal_OpenGL",
            "Normal_DirectX",
            lambda inv, ctx: self._convert_dx_to_gl(inv["Normal_DirectX"], ctx),
            priority=10,
        )
        self.register(
            "Normal_DirectX",
            "Normal_OpenGL",
            lambda inv, ctx: self._convert_gl_to_dx(inv["Normal_OpenGL"], ctx),
            priority=10,
        )
        self.register(
            "Normal",
            ["Bump", "Height"],
            lambda inv, ctx: self._convert_bump_to_normal(
                inv.get("Bump") or inv["Height"], ctx
            ),
            priority=5,
        )

        # Packing conversions (ORM)
        # Priority 10: All components present, native Roughness
        self.register(
            "ORM",
            ["Metallic", "Roughness", "Ambient_Occlusion"],
            lambda inv, ctx: self._create_orm_map(inv, ctx),
            priority=10,
        )
        # Priority 9: All components present, converted Smoothness
        self.register(
            "ORM",
            ["Metallic", "Smoothness", "Ambient_Occlusion"],
            lambda inv, ctx: self._create_orm_map(inv, ctx),
            priority=9,
        )
        # Priority 8: Missing AO, native Roughness
        self.register(
            "ORM",
            ["Metallic", "Roughness"],
            lambda inv, ctx: self._create_orm_map(inv, ctx),
            priority=8,
        )
        # Priority 7: Missing AO, converted Smoothness
        self.register(
            "ORM",
            ["Metallic", "Smoothness"],
            lambda inv, ctx: self._create_orm_map(inv, ctx),
            priority=7,
        )

        # Packing conversions (MSAO/MaskMap)
        # Priority 10: All components present, native Smoothness
        self.register(
            "MSAO",
            ["Metallic", "Ambient_Occlusion", "Smoothness"],
            lambda inv, ctx: self._create_mask_map(inv, ctx),
            priority=10,
        )
        # Priority 9: All components present, converted Roughness
        self.register(
            "MSAO",
            ["Metallic", "Ambient_Occlusion", "Roughness"],
            lambda inv, ctx: self._create_mask_map(inv, ctx),
            priority=9,
        )
        # Priority 8: Missing AO, native Smoothness
        self.register(
            "MSAO",
            ["Metallic", "Smoothness"],
            lambda inv, ctx: self._create_mask_map(inv, ctx),
            priority=8,
        )
        # Priority 7: Missing AO, converted Roughness
        self.register(
            "MSAO",
            ["Metallic", "Roughness"],
            lambda inv, ctx: self._create_mask_map(inv, ctx),
            priority=7,
        )

        # Packing conversions (Metallic_Smoothness)
        self.register(
            "Metallic_Smoothness",
            ["Metallic", "Smoothness"],
            lambda inv, ctx: self._create_metallic_smoothness_map(inv, ctx),
            priority=10,
        )
        self.register(
            "Metallic_Smoothness",
            ["Metallic", "Roughness"],
            lambda inv, ctx: self._create_metallic_smoothness_map(inv, ctx),
            priority=9,
        )

        # Unpacking conversions (Metallic_Smoothness)
        self.register(
            "Metallic",
            "Metallic_Smoothness",
            lambda inv, ctx: self._get_metallic_from_packed(
                inv["Metallic_Smoothness"], ctx
            ),
            priority=8,
        )

        self.register(
            "Smoothness",
            "Metallic_Smoothness",
            lambda inv, ctx: self._get_smoothness_from_packed(
                inv["Metallic_Smoothness"], ctx
            ),
            priority=8,
        )
        self.register(
            "Roughness",
            "Metallic_Smoothness",
            lambda inv, ctx: self._get_roughness_from_packed(
                inv["Metallic_Smoothness"], ctx
            ),
            priority=8,
        )

        # Unpacking conversions (MSAO)
        self.register(
            "Metallic",
            "MSAO",
            lambda inv, ctx: self._get_metallic_from_msao(inv["MSAO"], ctx),
            priority=8,
        )
        self.register(
            "Smoothness",
            "MSAO",
            lambda inv, ctx: self._get_smoothness_from_msao(inv["MSAO"], ctx),
            priority=8,
        )
        self.register(
            "Roughness",
            "MSAO",
            lambda inv, ctx: self._get_roughness_from_msao(inv["MSAO"], ctx),
            priority=8,
        )
        self.register(
            "Ambient_Occlusion",
            "MSAO",
            lambda inv, ctx: self._get_ao_from_msao(inv["MSAO"], ctx),
            priority=8,
        )
        self.register(
            "AO",
            "MSAO",
            lambda inv, ctx: self._get_ao_from_msao(inv["MSAO"], ctx),
            priority=8,
        )

        # Unpacking conversions (ORM)
        self.register(
            "Ambient_Occlusion",
            "ORM",
            lambda inv, ctx: self._get_ao_from_orm(inv["ORM"], ctx),
            priority=8,
        )
        self.register(
            "AO",
            "ORM",
            lambda inv, ctx: self._get_ao_from_orm(inv["ORM"], ctx),
            priority=8,
        )
        self.register(
            "Roughness",
            "ORM",
            lambda inv, ctx: self._get_roughness_from_orm(inv["ORM"], ctx),
            priority=8,
        )
        self.register(
            "Smoothness",
            "ORM",
            lambda inv, ctx: self._get_smoothness_from_orm(inv["ORM"], ctx),
            priority=8,
        )
        self.register(
            "Metallic",
            "ORM",
            lambda inv, ctx: self._get_metallic_from_orm(inv["ORM"], ctx),
            priority=8,
        )

        # Unpacking conversions (Albedo_Transparency)
        self.register(
            "Base_Color",
            "Albedo_Transparency",
            lambda inv, ctx: self._get_base_color_from_albedo_transparency(
                inv["Albedo_Transparency"], ctx
            ),
            priority=8,
        )
        self.register(
            "Opacity",
            "Albedo_Transparency",
            lambda inv, ctx: self._get_opacity_from_albedo_transparency(
                inv["Albedo_Transparency"], ctx
            ),
            priority=8,
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

    @staticmethod
    def _extract_gloss_from_spec(
        specular_path: str, context: "ProcessingContext"
    ) -> str:
        gloss_img = ImgUtils.extract_gloss_from_spec(specular_path)
        if not gloss_img:
            raise ValueError("Could not extract gloss from specular map")

        gloss_path = os.path.join(
            context.output_dir, f"{context.base_name}_Glossiness.{context.ext}"
        )
        gloss_img.save(gloss_path)
        context.log("Extracted glossiness from specular")
        return gloss_path

    @staticmethod
    def _copy_map(
        source_path: str, target_type: str, context: "ProcessingContext"
    ) -> str:
        """Simple copy/rename for compatible maps (e.g. Smoothness -> Glossiness)."""
        target_path = os.path.join(
            context.output_dir, f"{context.base_name}_{target_type}.{context.ext}"
        )
        ImgUtils.save_image(source_path, target_path)
        context.log(f"Created {target_type} from source map")
        return target_path

    @staticmethod
    def _unpack_metallic_smoothness(
        source_path: str, context: "ProcessingContext"
    ) -> None:
        """Helper to unpack and cache results."""
        # Return cached if available
        if "Metallic" in context.inventory and "Smoothness" in context.inventory:
            return

        metallic, smoothness = ImgUtils.unpack_metallic_smoothness(
            source_path, context.output_dir
        )
        context.inventory["Metallic"] = metallic
        context.inventory["Smoothness"] = smoothness
        context.log("Unpacked Metallic and Smoothness from packed map")

    @staticmethod
    def _get_metallic_from_packed(
        source_path: str, context: "ProcessingContext"
    ) -> str:
        ConversionRegistry._unpack_metallic_smoothness(source_path, context)
        return context.inventory["Metallic"]

    @staticmethod
    def _get_smoothness_from_packed(
        source_path: str, context: "ProcessingContext"
    ) -> str:
        ConversionRegistry._unpack_metallic_smoothness(source_path, context)
        return context.inventory["Smoothness"]

    @staticmethod
    def _get_roughness_from_packed(
        source_path: str, context: "ProcessingContext"
    ) -> str:
        ConversionRegistry._unpack_metallic_smoothness(source_path, context)
        # Convert S -> R
        return ConversionRegistry._convert_smoothness_to_roughness(
            context.inventory["Smoothness"], context
        )

    @staticmethod
    def _unpack_msao(source_path: str, context: "ProcessingContext") -> None:
        """Helper to unpack MSAO and cache results."""
        if (
            "Metallic" in context.inventory
            and "AO" in context.inventory
            and "Smoothness" in context.inventory
        ):
            return

        metallic, ao, smoothness = ImgUtils.unpack_msao_texture(
            source_path, context.output_dir
        )
        context.inventory["Metallic"] = metallic
        context.inventory["AO"] = ao
        context.inventory["Ambient_Occlusion"] = ao
        context.inventory["Smoothness"] = smoothness
        context.log("Unpacked Metallic, AO, and Smoothness from MSAO map")

    @staticmethod
    def _get_metallic_from_msao(source_path: str, context: "ProcessingContext") -> str:
        ConversionRegistry._unpack_msao(source_path, context)
        return context.inventory["Metallic"]

    @staticmethod
    def _get_smoothness_from_msao(
        source_path: str, context: "ProcessingContext"
    ) -> str:
        ConversionRegistry._unpack_msao(source_path, context)
        return context.inventory["Smoothness"]

    @staticmethod
    def _get_roughness_from_msao(source_path: str, context: "ProcessingContext") -> str:
        ConversionRegistry._unpack_msao(source_path, context)
        return ConversionRegistry._convert_smoothness_to_roughness(
            context.inventory["Smoothness"], context
        )

    @staticmethod
    def _get_ao_from_msao(source_path: str, context: "ProcessingContext") -> str:
        ConversionRegistry._unpack_msao(source_path, context)
        return context.inventory["AO"]

    @staticmethod
    def _unpack_orm(source_path: str, context: "ProcessingContext") -> None:
        """Helper to unpack ORM and cache results."""
        if (
            "AO" in context.inventory
            and "Roughness" in context.inventory
            and "Metallic" in context.inventory
        ):
            return

        ao, roughness, metallic = ImgUtils.unpack_orm_texture(
            source_path, context.output_dir
        )
        context.inventory["AO"] = ao
        context.inventory["Ambient_Occlusion"] = ao
        context.inventory["Roughness"] = roughness
        context.inventory["Metallic"] = metallic
        context.log("Unpacked AO, Roughness, and Metallic from ORM map")

    @staticmethod
    def _get_ao_from_orm(source_path: str, context: "ProcessingContext") -> str:
        ConversionRegistry._unpack_orm(source_path, context)
        return context.inventory["AO"]

    @staticmethod
    def _get_roughness_from_orm(source_path: str, context: "ProcessingContext") -> str:
        ConversionRegistry._unpack_orm(source_path, context)
        return context.inventory["Roughness"]

    @staticmethod
    def _get_smoothness_from_orm(source_path: str, context: "ProcessingContext") -> str:
        ConversionRegistry._unpack_orm(source_path, context)
        # Convert R -> S
        return ConversionRegistry._convert_roughness_to_smoothness(
            context.inventory["Roughness"], context
        )

    @staticmethod
    def _get_metallic_from_orm(source_path: str, context: "ProcessingContext") -> str:
        ConversionRegistry._unpack_orm(source_path, context)
        return context.inventory["Metallic"]

    @staticmethod
    def _unpack_albedo_transparency(
        source_path: str, context: "ProcessingContext"
    ) -> None:
        """Helper to unpack Albedo+Transparency and cache results."""
        if "Base_Color" in context.inventory and "Opacity" in context.inventory:
            return

        base_color, opacity = ImgUtils.unpack_albedo_transparency(
            source_path, context.output_dir
        )
        context.inventory["Base_Color"] = base_color
        context.inventory["Opacity"] = opacity
        context.log("Unpacked Base Color and Opacity from Albedo+Transparency map")

    @staticmethod
    def _get_base_color_from_albedo_transparency(
        source_path: str, context: "ProcessingContext"
    ) -> str:
        ConversionRegistry._unpack_albedo_transparency(source_path, context)
        return context.inventory["Base_Color"]

    @staticmethod
    def _create_orm_map(inventory: Dict[str, str], context: "ProcessingContext") -> str:
        """Create ORM map from components."""
        # Resolve required components
        ao = context.resolve_map("Ambient_Occlusion", "AO", allow_conversion=False)
        roughness = context.resolve_map(
            "Roughness", "Smoothness", "Glossiness", allow_conversion=True
        )
        metallic = context.resolve_map("Metallic", "Specular", allow_conversion=True)

        if not (ao or roughness or metallic):
            raise ValueError("Missing components for ORM map")

        orm_map = ImgUtils.pack_channels(
            channel_files={"R": ao, "G": roughness, "B": metallic},
            output_path=os.path.join(
                context.output_dir, f"{context.base_name}_ORM.{context.ext}"
            ),
            fill_values={"R": 255} if not ao else None,
        )
        context.log("Created ORM map from components")
        return orm_map

    @staticmethod
    def _create_mask_map(
        inventory: Dict[str, str], context: "ProcessingContext"
    ) -> str:
        """Create Mask Map (MSAO) from components."""
        # Resolve required components
        metallic = context.resolve_map("Metallic", "Specular", allow_conversion=True)
        ao = (
            context.resolve_map("Ambient_Occlusion", "AO", allow_conversion=False)
            or metallic
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
            smoothness = metallic

        if not metallic:
            raise ValueError("Missing Metallic map for Mask Map")

        mask_map = ImgUtils.pack_msao_texture(
            metallic_map_path=metallic,
            ao_map_path=ao,
            alpha_map_path=smoothness,
            output_dir=context.output_dir,
            suffix="_MaskMap",
            invert_alpha=invert,
        )
        context.log("Created Mask Map from components")
        return mask_map

    @staticmethod
    def _create_metallic_smoothness_map(
        inventory: Dict[str, str], context: "ProcessingContext"
    ) -> str:
        """Create Metallic-Smoothness map from components."""
        metallic = context.resolve_map("Metallic", "Specular", allow_conversion=True)

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

        ms_map = ImgUtils.pack_smoothness_into_metallic(
            metallic_map_path=metallic,
            alpha_map_path=smoothness,
            output_dir=context.output_dir,
            suffix="_MetallicSmoothness",
            invert_alpha=invert,
        )
        context.log("Packed smoothness into metallic")
        return ms_map

    @staticmethod
    def _get_opacity_from_albedo_transparency(
        source_path: str, context: "ProcessingContext"
    ) -> str:
        ConversionRegistry._unpack_albedo_transparency(source_path, context)
        return context.inventory["Opacity"]


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

    _LOG_COLORS = {
        "success": "rgb(100, 160, 100)",
        "warning": "rgb(200, 200, 100)",
        "error": "rgb(255, 100, 100)",
    }

    def log(self, message: str, level: str = "success"):
        """Unified logging with color coding."""
        color = self._LOG_COLORS.get(level, self._LOG_COLORS["success"])
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

        ao = context.resolve_map("Ambient_Occlusion", allow_conversion=False)
        if not ao:
            context.log("No AO map, using white for Mask Map green channel", "warning")

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
            # Use pack_channels directly to allow missing AO (defaults to white)
            output_path = os.path.join(
                context.output_dir, f"{context.base_name}_MaskMap.{context.ext}"
            )

            mask_map = ImgUtils.pack_channels(
                channel_files={
                    "R": metallic,
                    "G": ao,
                    "B": None,
                    "A": smoothness,
                },
                channels=["R", "G", "B", "A"],
                out_mode="RGBA",
                fill_values={"R": 0, "G": 255, "B": 0, "A": 0 if invert else 255},
                output_path=output_path,
            )

            if invert:
                # Invert the alpha channel after packing if needed
                img = ImgUtils.ensure_image(mask_map)
                img = ImgUtils.invert_channels(img, "A")
                img.save(output_path)

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
            context.log("Processing albedo with transparency")
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

        context.log("Processing base color")

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
            context.log(f"Processing normal map ({target_format} format)")
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
        NormalMapHandler,
        ORMMapHandler,
        MaskMapHandler,
        MetallicSmoothnessHandler,
        SeparateMetallicRoughnessHandler,
    ]

    passthrough_maps = [
        "Emissive",
        "Emission",
        "Ambient_Occlusion",
        "AO",
        "Height",
        "Displacement",
    ]

    map_fallbacks = {
        "Base_Color": ("Diffuse", "Albedo_Transparency"),
        "Diffuse": ("Base_Color", "Albedo_Transparency"),
        "Albedo_Transparency": ("Base_Color", "Diffuse"),
        "Normal": ("Normal_OpenGL", "Normal_DirectX", "Bump", "Height"),
        "Normal_OpenGL": ("Normal", "Normal_DirectX", "Bump", "Height"),
        "Normal_DirectX": ("Normal", "Normal_OpenGL", "Bump", "Height"),
        "Bump": ("Normal", "Normal_OpenGL", "Normal_DirectX", "Height"),
        "Height": ("Displacement", "Bump", "Normal"),
        "Roughness": ("Glossiness", "Smoothness"),
        "Glossiness": ("Roughness", "Smoothness"),
        "Smoothness": ("Roughness", "Glossiness"),
        "Metallic": ("Specular", "Metalness"),
        "Specular": ("Metallic", "Metalness"),
        "Ambient_Occlusion": ("AO", "Occlusion"),
        "Opacity": ("Transparency", "Alpha"),
        "Emissive": ("Emission",),
    }

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
    def prepare_maps(
        cls,
        source: Union[str, List[str]],
        workflow_config: dict,
        output_dir: str = None,
        callback: Callable = print,
        group_by_set: bool = True,
    ) -> Union[List[str], Dict[str, List[str]]]:
        """
        Main factory method. Automatically handles batch processing.

        Parameters:
            source: A directory path (str), a single file path (str), or a list of file paths.
            workflow_config: Configuration dictionary.
            output_dir: Optional output directory.
            callback: Logging callback.
            group_by_set: Whether to automatically group textures into sets (default: True).
                          If False, all input files are treated as a single set.

        Returns:
            List[str] if a single asset was processed.
            Dict[str, List[str]] if multiple assets were processed (keyed by asset name).
        """
        # Resolve input files
        files = []
        if isinstance(source, str):
            if os.path.isdir(source):
                files = FileUtils.get_dir_contents(
                    source,
                    "filepath",
                    inc_files=["*.png", "*.jpg", "*.tga", "*.tif", "*.tiff", "*.exr"],
                )
            elif os.path.isfile(source):
                files = [source]
        else:
            files = source

        if not files:
            callback("No input files found.")
            return []

        if group_by_set:
            # Group by texture set
            texture_sets = ImgUtils.group_textures_by_set(files)
        else:
            # Treat all files as a single set
            # Use the common prefix or just the first file's base name as the key
            base_name = ImgUtils.get_base_texture_name(files[0])
            texture_sets = {base_name: files}

        results = {}
        total_sets = len(texture_sets)

        if total_sets > 1:
            callback(f"Found {total_sets} texture sets. Processing batch...")

        for i, (base_name, textures) in enumerate(texture_sets.items(), 1):
            if total_sets > 1:
                callback(f"Processing set {i}/{total_sets}: {base_name}")

            try:
                generated = cls._process_map_set(
                    textures,
                    workflow_config,
                    output_dir=output_dir,
                    callback=callback,
                )
                results[base_name] = generated
            except Exception as e:
                callback(f"Error processing set {base_name}: {e}", "error")
                import traceback

                traceback.print_exc()

        # Smart return: if single set, return list directly
        if len(results) == 1:
            return next(iter(results.values()))

        return results

    @staticmethod
    def _process_map_set(
        textures: List[str],
        workflow_config: dict,
        output_dir: str = None,
        callback: Callable = print,
    ) -> List[str]:
        """Internal method to process a single set of textures (one asset)."""
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
            output_dir=output_dir or os.path.dirname(first_map),
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
        for map_type in TextureMapFactory.passthrough_maps:
            if map_type in map_inventory and map_type not in context.used_maps:
                output_maps.append(map_inventory[map_type])
                callback(f"Passing through {map_type} map")

        return output_maps if output_maps else textures

    @staticmethod
    def _build_map_inventory(textures: List[str]) -> Dict[str, str]:
        """Build map inventory using ImgUtils."""
        inventory = {}
        for texture in textures:
            map_type = ImgUtils.resolve_map_type(texture)
            if map_type and map_type not in inventory:
                inventory[map_type] = texture
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
