import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Union
from pythontk.core_utils.singleton_mixin import SingletonMixin


class WF:
    """Workflow identifiers."""

    STD = "PBR Metallic/Roughness"
    URP = "Unity URP Lit"
    HDRP = "Unity HDRP"
    UE = "Unreal Engine"
    GLTF = "glTF 2.0"
    GODOT = "Godot"
    SPEC = "PBR Specular/Glossiness"

    # Groups
    ALL_ENGINES = [URP, HDRP, UE, GLTF, GODOT]


@dataclass
class _WorkflowPreset:
    """Internal configuration for workflow presets."""

    albedo_transparency: bool = False
    metallic_smoothness: bool = False
    mask_map: bool = False
    orm_map: bool = False
    opacity: bool = False
    emissive: bool = False
    ambient_occlusion: bool = False
    convert_specgloss_to_pbr: bool = False
    normal_type: str = "OpenGL"
    cleanup_base_color: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MapType:
    """Defines the properties of a texture map type."""

    name: str
    aliases: List[str]
    color_space: str = "Linear"  # "sRGB" or "Linear"
    mode: str = "RGB"  # "RGB", "RGBA", "L"
    default_background: Tuple[int, ...] = (0, 0, 0, 255)  # Default background color
    is_packed: bool = False  # Is this a packed map (e.g. ORM, MSAO)?
    scale_as_mask: bool = False  # Should this map be scaled down by mask_map_scale?
    input_fallbacks: List[str] = field(
        default_factory=list
    )  # Safe substitutes for INPUT (e.g. Bump -> Normal)
    output_fallbacks: List[str] = field(
        default_factory=list
    )  # Safe substitutes for OUTPUT (e.g. MSAO -> AO)
    replaces: List[str] = field(
        default_factory=list
    )  # Maps that this map renders redundant
    workflows: List[str] = field(default_factory=list)  # Workflows that use this map


class MapRegistry(SingletonMixin):
    """Central registry for map type definitions."""

    _precedence_rules = None
    _workflow_settings = {
        WF.STD: {
            "description": "Standard PBR workflow (Metallic/Roughness) with separate Opacity. Best for general use."
        },
        WF.URP: {
            "description": "Unity Universal Render Pipeline. Packs Metallic (R) and Smoothness (A)."
        },
        WF.HDRP: {
            "description": "Unity High Definition Render Pipeline. Uses Mask Map (Metallic, AO, Detail, Smoothness)."
        },
        WF.UE: {
            "normal_type": "DirectX",
            "description": "Unreal Engine. Uses ORM (Occlusion, Roughness, Metallic) and DirectX Normals.",
        },
        WF.GLTF: {
            "description": "glTF 2.0 standard. Uses ORM (Occlusion, Roughness, Metallic)."
        },
        WF.GODOT: {
            "normal_type": "OpenGL",
            "description": "Godot Engine. Uses ORM and OpenGL Normals.",
        },
        WF.SPEC: {
            "convert_specgloss_to_pbr": True,
            "description": "Converts Specular/Glossiness maps to PBR Metallic/Roughness.",
        },
    }
    _maps: Dict[str, MapType] = {
        "Base_Color": MapType(
            name="Base_Color",
            aliases=[
                "BaseColor",
                "BaseColour",
                "Base_Map",
                "BaseMap",
                "BaseColorMap",
                "Base_ColorMap",
                "Albedo",
                "AlbedoMap",
                "BaseColorTexture",
                "BaseMapTexture",
                "ColorMap",
                "Color",
                "BC",
            ],
            color_space="sRGB",
            mode="RGB",
            default_background=(127, 127, 127, 255),
            input_fallbacks=["Albedo_Transparency", "Diffuse"],
        ),
        "Diffuse": MapType(
            name="Diffuse",
            aliases=[
                "DiffuseMap",
                "Diff",
                "D",
            ],
            color_space="sRGB",
            mode="RGB",
            default_background=(127, 127, 127, 255),
            input_fallbacks=["Base_Color"],
        ),
        "Albedo_Transparency": MapType(
            name="Albedo_Transparency",
            aliases=[
                "AlbedoTransparency",
                "AlbedoAlpha",
                "AlbedoOpacity",
                "BaseColorTransparency",
                "BaseColorAlpha",
                "BaseMapAlpha",
                "AT",
            ],
            color_space="sRGB",
            mode="RGBA",
            default_background=(0, 0, 0, 255),
            input_fallbacks=["Base_Color"],
            workflows=WF.ALL_ENGINES,
        ),
        "Roughness": MapType(
            name="Roughness",
            aliases=[
                "RoughnessMap",
                "Rough",
                "RoughMap",
                "Ruff",
                "Rgh",
                "RGH",
                "R",
            ],
            color_space="Linear",
            mode="L",
            default_background=(255, 255, 255, 255),
            input_fallbacks=["Glossiness", "Smoothness"],
        ),
        "Metallic": MapType(
            name="Metallic",
            aliases=[
                "MetallicMap",
                "Metal",
                "MetalMap",
                "Metalness",
                "Met",
                "MTL",
                "M",
            ],
            color_space="Linear",
            mode="L",
            default_background=(0, 0, 0, 255),
            input_fallbacks=["Specular", "Metalness"],
        ),
        "Normal": MapType(
            name="Normal",
            aliases=[
                "NormalMap",
                "Normal_Map",
                "Norm",
                "NRM",
                "N",
                "TangentSpaceNormal",
                "TSN",
            ],
            color_space="Linear",
            mode="RGB",
            default_background=(127, 127, 255, 255),
            input_fallbacks=["Normal_OpenGL", "Normal_DirectX", "Bump", "Height"],
        ),
        "Normal_OpenGL": MapType(
            name="Normal_OpenGL",
            aliases=[
                "NormalGL",
                "Normal_GL",
                "Normal_Tangent_GL",
                "NormalMap_GL",
                "NGL",
                "GL",
            ],
            color_space="Linear",
            mode="RGB",
            default_background=(127, 127, 255, 255),
            input_fallbacks=["Normal", "Normal_DirectX", "Bump", "Height"],
        ),
        "Normal_DirectX": MapType(
            name="Normal_DirectX",
            aliases=[
                "NormalDX",
                "Normal_DX",
                "Normal_Tangent_DX",
                "NormalMap_DX",
                "NDX",
                "DX",
                "DXN",
            ],
            color_space="Linear",
            mode="RGB",
            default_background=(127, 127, 255, 255),
            input_fallbacks=["Normal", "Normal_OpenGL", "Bump", "Height"],
        ),
        "ORM": MapType(
            name="ORM",
            aliases=[
                "OcclusionRoughnessMetallic",
                "Occlusion_Roughness_Metallic",
                "ORMMap",
            ],
            color_space="Linear",
            mode="RGB",
            default_background=(255, 255, 0, 255),
            is_packed=True,
            scale_as_mask=True,
            output_fallbacks=["Ambient_Occlusion", "Roughness", "Metallic"],
            replaces=["Metallic", "Ambient_Occlusion", "Roughness"],
            workflows=[WF.UE, WF.GLTF],
        ),
        "MSAO": MapType(
            name="MSAO",
            aliases=[
                "Metallic_SmoothnessAO",
                "MetallicSmoothnessAO",
                "MetallicSmoothAO",
                "MetallicSmoothness_AO",
                "MetallicSmoothnessAmbientOcclusion",
                "MetallicSmoothnessOcclusion",
                "MaskMap",
                "Mask_Map",
                "MSA",
            ],
            color_space="Linear",
            mode="RGBA",
            default_background=(0, 255, 0, 255),
            is_packed=True,
            scale_as_mask=True,
            output_fallbacks=[
                "Metallic_Smoothness",
                "Ambient_Occlusion",
                "Detail_Mask",
            ],
            replaces=[
                "Metallic",
                "Ambient_Occlusion",
                "Roughness",
                "Specular",
                "Glossiness",
                "Detail",
                "Detail_Mask",
                "Metallic_Smoothness",
            ],
            workflows=[WF.HDRP],
        ),
        "Metallic_Smoothness": MapType(
            name="Metallic_Smoothness",
            aliases=[
                "MetallicSmoothness",
                "MetalSmooth",
                "Metal_Smooth",
                "Metal_Smoothness",
                "MetallicSmoothnessMap",
                "Metallic_SmoothnessMap",
                "MetallicGloss",
                "MetalGloss",
                "MetallicGlossMap",
                "MS",
            ],
            color_space="Linear",
            mode="RGBA",
            default_background=(255, 255, 255, 255),
            is_packed=True,
            scale_as_mask=True,
            output_fallbacks=["Metallic", "Smoothness"],
            workflows=[WF.URP, WF.SPEC],
        ),
        "Ambient_Occlusion": MapType(
            name="Ambient_Occlusion",
            aliases=[
                "AmbientOcclusion",
                "Ambient",
                "Amb",
                "AO",
                "Occlusion",
                "Occ",
                "AO_Map",
                "AOMap",
                "Mixed_AO",
                "MixedAO",
            ],
            color_space="Linear",
            mode="L",
            default_background=(255, 255, 255, 255),
            input_fallbacks=["AO", "Occlusion"],
            workflows=[WF.STD],
        ),
        "Height": MapType(
            name="Height",
            aliases=[
                "HeightMap",
                "Height_Map",
                "High",
                "HGT",
                "Parallax",
                "ParallaxMap",
                "ParallaxOcclusion",
                "POM",
                "H",
            ],
            color_space="Linear",
            mode="L",
            default_background=(128, 128, 128, 255),
            input_fallbacks=["Displacement", "Bump", "Normal"],
        ),
        "Bump": MapType(
            name="Bump",
            aliases=[
                "BumpMap",
                "Bump_Map",
                "Bumpiness",
                "BumpinessMap",
                "BP",
                "B",
            ],
            color_space="Linear",
            mode="L",
            default_background=(128, 128, 128, 255),
            input_fallbacks=["Normal", "Normal_OpenGL", "Normal_DirectX", "Height"],
        ),
        "Emissive": MapType(
            name="Emissive",
            aliases=[
                "EmissiveMap",
                "Emission",
                "EmissionMap",
                "Emit",
                "Glow",
                "GlowMap",
                "EMI",
                "E",
                "EM",
            ],
            color_space="sRGB",
            mode="RGB",
            default_background=(0, 0, 0, 255),
            input_fallbacks=["Emission"],
            workflows=[WF.STD],
        ),
        "Detail_Mask": MapType(
            name="Detail_Mask",
            aliases=[
                "DetailMask",
                "Detail_Map",
                "DetailMap",
                "Detail",
                "DTL",
            ],
            color_space="Linear",
            mode="L",
            default_background=(0, 0, 0, 255),
            scale_as_mask=True,
        ),
        "Mask": MapType(
            name="Mask",
            aliases=["Mask"],
            color_space="Linear",
            mode="L",
            default_background=(255, 255, 255, 255),
            scale_as_mask=True,
            output_fallbacks=["Metallic_Smoothness", "Ambient_Occlusion"],
        ),
        "Specular": MapType(
            name="Specular",
            aliases=["SpecularMap", "Spec", "SPC", "S"],
            color_space="sRGB",
            mode="RGB",
            default_background=(0, 0, 0, 255),
            input_fallbacks=["Metallic", "Metalness"],
        ),
        "Glossiness": MapType(
            name="Glossiness",
            aliases=[
                "GlossinessMap",
                "Gloss",
                "Gls",
                "G",
            ],
            color_space="Linear",
            mode="L",
            default_background=(0, 0, 0, 255),
            input_fallbacks=["Roughness", "Smoothness"],
        ),
        "Smoothness": MapType(
            name="Smoothness",
            aliases=["SmoothnessMap", "Smooth"],
            color_space="Linear",
            mode="L",
            default_background=(0, 0, 0, 255),
            input_fallbacks=["Roughness", "Glossiness"],
        ),
        "Opacity": MapType(
            name="Opacity",
            aliases=["OpacityMap", "Transparency", "Alpha"],
            color_space="Linear",
            mode="L",
            default_background=(255, 255, 255, 255),
            input_fallbacks=["Transparency", "Alpha"],
            workflows=[WF.STD],
        ),
        "Displacement": MapType(
            name="Displacement",
            aliases=["DisplacementMap", "Disp", "DSP"],
            color_space="Linear",
            mode="L",
            default_background=(128, 128, 128, 255),
            input_fallbacks=["Height"],
        ),
        "Refraction": MapType(
            name="Refraction",
            aliases=["RefractionMap", "Refr"],
            color_space="Linear",
            mode="L",
            default_background=(0, 0, 0, 255),
        ),
        "Reflection": MapType(
            name="Reflection",
            aliases=["ReflectionMap", "Refl"],
            color_space="Linear",
            mode="L",
            default_background=(0, 0, 0, 255),
        ),
        "Thickness": MapType(
            name="Thickness",
            aliases=["ThicknessMap", "Thick"],
            color_space="Linear",
            mode="L",
            default_background=(0, 0, 0, 255),
            scale_as_mask=True,
        ),
        "Anisotropy": MapType(
            name="Anisotropy",
            aliases=["AnisotropyMap", "Aniso"],
            color_space="Linear",
            mode="L",
            default_background=(127, 127, 127, 255),
        ),
        "Subsurface_Scattering": MapType(
            name="Subsurface_Scattering",
            aliases=["SSS", "Subsurface", "Scattering"],
            color_space="sRGB",
            mode="RGB",
            default_background=(255, 255, 255, 255),
        ),
        "Sheen": MapType(
            name="Sheen",
            aliases=["SheenMap"],
            color_space="Linear",
            mode="L",
            default_background=(127, 127, 127, 255),
            scale_as_mask=True,
        ),
        "Clearcoat": MapType(
            name="Clearcoat",
            aliases=["ClearcoatMap", "Coat"],
            color_space="Linear",
            mode="L",
            default_background=(127, 127, 127, 255),
            scale_as_mask=True,
        ),
    }

    def get(self, name: str) -> Optional[MapType]:
        """Get a map type by name."""
        return self._maps.get(name)

    def resolve_type_from_path(self, path: str) -> Optional[str]:
        """Resolve the map type key from a file path.

        Prioritizes longer matches to prevent short aliases (e.g. 'S') from
        matching longer names (e.g. 'Smoothness').
        """
        filename = os.path.basename(path)
        name_only, _ = os.path.splitext(filename)

        # Collect all candidates: (alias, map_name)
        all_candidates = []
        for name, m in self._maps.items():
            # Add main name
            all_candidates.append((name, name))
            # Add aliases
            for alias in m.aliases:
                all_candidates.append((alias, name))

        # Sort by length descending to ensure longest match first
        all_candidates.sort(key=lambda x: len(x[0]), reverse=True)

        for alias, map_name in all_candidates:
            # Logic for short aliases (<= 3 chars)
            if len(alias) <= 3:
                # Must match case for first letter, rest case-insensitive
                # And must be at the end of the string
                if name_only.lower().endswith(alias.lower()):
                    # Check case sensitivity for short aliases
                    suffix_start_index = len(name_only) - len(alias)
                    suffix_in_name = name_only[suffix_start_index:]

                    # If alias starts with uppercase, require uppercase in filename
                    if alias[0].isupper():
                        if suffix_in_name[0] == alias[0]:
                            return map_name
                    else:
                        return map_name
            else:
                # Long aliases: Case-insensitive
                if name_only.lower().endswith(alias.lower()):
                    return map_name

        return None

    def get_workflow_presets(self) -> Dict[str, Dict[str, Any]]:
        """Generate the workflow presets dictionary."""
        presets = {}

        for workflow_name, settings in self._workflow_settings.items():
            # Create preset with defaults
            preset = _WorkflowPreset(**settings)

            # Check maps for this workflow
            for m in self._maps.values():
                if workflow_name in m.workflows:
                    # Infer config field from map name
                    name_lower = m.name.lower()
                    if hasattr(preset, name_lower):
                        setattr(preset, name_lower, True)
                    elif hasattr(preset, f"{name_lower}_map"):
                        setattr(preset, f"{name_lower}_map", True)
                    elif m.name == "MSAO":
                        setattr(preset, "mask_map", True)

            presets[workflow_name] = preset.to_dict()

        return presets

    def get_map_types(self) -> Dict[str, Tuple[str, ...]]:
        """Generate the dictionary format for ImgUtils.map_types."""
        return {name: tuple([name] + m.aliases) for name, m in self._maps.items()}

    def get_fallbacks(self) -> Dict[str, Tuple[str, ...]]:
        """Generate the input fallback dictionary."""
        return {
            name: tuple(m.input_fallbacks)
            for name, m in self._maps.items()
            if m.input_fallbacks
        }

    def get_output_fallbacks(self) -> Dict[str, Tuple[str, ...]]:
        """Generate the output fallback dictionary."""
        return {
            name: tuple(m.output_fallbacks)
            for name, m in self._maps.items()
            if m.output_fallbacks
        }

    def get_precedence_rules(self) -> Dict[str, List[str]]:
        """Generate the precedence rules dictionary."""
        if self._precedence_rules is None:
            self._precedence_rules = {
                name: m.replaces for name, m in self._maps.items() if m.replaces
            }
        return self._precedence_rules

    def get_scale_as_mask_types(self) -> List[str]:
        """Get list of map types that should be scaled as masks."""
        return [name for name, m in self._maps.items() if m.scale_as_mask]

    def get_passthrough_maps(self) -> List[str]:
        """Get list of maps that should be passed through if not consumed."""
        # Return all registered maps so anything not consumed by a handler is passed through
        return list(self._maps.keys())

    def get_map_backgrounds(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Generate the map backgrounds dictionary."""
        return {
            name: m.default_background
            for name, m in self._maps.items()
            if m.default_background is not None
        }

    def get_map_modes(self) -> Dict[str, str]:
        """Generate the map modes dictionary."""
        return {name: m.mode for name, m in self._maps.items() if m.mode is not None}

    def resolve_config(
        self, config: Union[str, Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Resolve configuration from presets, dicts, and kwargs.

        Args:
            config: Configuration preset name (str) or dictionary.
            **kwargs: Configuration overrides.

        Returns:
            Dict[str, Any]: Fully resolved configuration dictionary.
        """
        cfg = {}
        presets = self.get_workflow_presets()

        if isinstance(config, str):
            if config in presets:
                cfg = presets[config].copy()
        elif isinstance(config, dict):
            # Check for preset inheritance
            preset_name = config.get("preset")
            if preset_name and preset_name in presets:
                cfg = presets[preset_name].copy()

            # Apply overrides from dict
            overrides = {k: v for k, v in config.items() if v is not None}
            cfg.update(overrides)

        # Apply kwargs overrides
        overrides = {k: v for k, v in kwargs.items() if v is not None}
        cfg.update(overrides)

        # --- Standardization Logic (DRY) ---

        # Handle aliases
        if "output_type" in cfg:
            cfg["output_extension"] = cfg.pop("output_type")

        # Derive resize from max_size
        if "max_size" in cfg and "resize" not in cfg:
            cfg["resize"] = cfg["max_size"] is not None

        # Derive convert_format from output_extension
        if "output_extension" in cfg and "convert_format" not in cfg:
            cfg["convert_format"] = cfg["output_extension"] is not None

        return cfg


if __name__ == "__main__":
    registry = MapRegistry()

    print("Map Types:")
    for name, m in registry._maps.items():
        print(f"{name}: {m}")

    print("\nWorkflow Presets:")
    presets = registry.get_workflow_presets()
    for wf_name, config in presets.items():
        print(f"{wf_name}: {config}")
