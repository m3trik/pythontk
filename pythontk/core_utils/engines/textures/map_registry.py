import os
import re
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
    mrao_map: bool = False
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
    mode: Optional[str] = "RGB"  # "RGB", "RGBA", "L"; None = preserve natural mode
    default_background: Optional[Tuple[int, ...]] = (0, 0, 0, 255)  # Default background color
    is_packed: bool = False  # Is this a packed map (e.g. ORM, MSAO)?
    scale_as_mask: bool = False  # Should this map be scaled down by mask_map_scale?
    resolution_critical: bool = False  # Surface detail depends on full resolution (color, normals, emissive). Others may be downscaled as a fraction.
    input_fallbacks: List[str] = field(
        default_factory=list
    )  # Safe substitutes for INPUT (e.g. Bump -> Normal)
    output_fallbacks: List[str] = field(
        default_factory=list
    )  # Safe substitutes for OUTPUT (e.g. MSAO -> AO)
    replaces: List[str] = field(
        default_factory=list
    )  # Maps that this map renders redundant
    config_key: Optional[str] = (
        None  # SSoT for the config flag gating this packed map as a desired OUTPUT (note MSAO -> "mask_map", not "msao_map"). filter_redundant_maps consults it (where the map declares `replaces`) to choose packed vs. separate maps.
    )
    channels: Dict[str, str] = field(
        default_factory=dict
    )  # Packed maps only: what each channel carries in the CANONICAL layout, e.g. ORM {"R": "Ambient_Occlusion", "G": "Roughness", "B": "Metallic"}. A trailing "?" marks an optional/filler channel (MSAO's Detail) that redundancy coverage must not demand. SSoT for coverage-aware filtering and channel extraction.
    workflows: List[str] = field(default_factory=list)  # Workflows that use this map

    def carried_types(self, include_optional: bool = False) -> List[str]:
        """The map types this packed map's channels carry.

        Parameters:
            include_optional: Include channels marked optional (trailing ``"?"``).

        Returns:
            list[str]: Canonical map-type names, in channel order.
        """
        types = []
        for carried in self.channels.values():
            optional = carried.endswith("?")
            if optional and not include_optional:
                continue
            types.append(carried.rstrip("?"))
        return types

    def __post_init__(self):
        # A partial packed-map contract is a latent wiring bug, not a valid
        # definition. `filter_redundant_maps` keys its precedence rules off
        # `replaces` (not `is_packed`), reads `config_key` for direction, and
        # `channels` for per-channel coverage/extraction — so each omission
        # fails differently, and silently:
        #   - `is_packed` without `replaces`/`config_key`: nothing ever retires
        #     the separate maps the packed map absorbed; both wire into the same
        #     material slots (the shipped example: packing opacity into
        #     Albedo_Transparency left the old Opacity map connected).
        #   - `replaces` without `config_key`: the direction gate is skipped and
        #     the map supersedes its components in EVERY preset, unpacked ones
        #     included.
        #   - `replaces` without `is_packed`: the marker lies to sweeps/tools
        #     that enumerate packed maps.
        #   - missing `channels`: coverage-aware filtering can't know what the
        #     map carries, so dropping it in an unpacked workflow silently
        #     loses the channels no loose map covers.
        # Fail at definition time so the next packed map type can't repeat any
        # of them. Loose maps (none of these fields) are untouched.
        if self.is_packed or self.replaces or self.channels:
            missing = [
                f
                for f, v in (
                    ("is_packed", self.is_packed),
                    ("replaces", self.replaces),
                    ("config_key", self.config_key),
                    ("channels", self.channels),
                )
                if not v
            ]
            if missing:
                raise ValueError(
                    f"Map type '{self.name}' declares a partial packed-map contract "
                    f"(missing {' and '.join(missing)}): a map that packs others "
                    "must set `is_packed=True`, declare the maps it `replaces`, "
                    "name the `config_key` that directs packed-vs-separate per "
                    "workflow, and declare its per-channel `channels` layout."
                )
            # Consistency: every carried type must be in `replaces`, or the
            # loose map a channel duplicates would never be retired when the
            # packed map wins (found live: MSAO carried Smoothness in its alpha
            # but didn't replace it, so a loose Smoothness stayed wired).
            unreplaced = [
                t for t in self.carried_types() if t not in self.replaces
            ]
            if unreplaced:
                raise ValueError(
                    f"Map type '{self.name}' carries {unreplaced} in its channels "
                    "but does not list them in `replaces` — the loose map a "
                    "channel duplicates must be retired when the packed map wins."
                )


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
            resolution_critical=True,
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
            resolution_critical=True,
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
            is_packed=True,
            input_fallbacks=["Base_Color"],
            # The albedo it packs and the alpha it packs both drive slots the
            # separate maps drive; without this, packing opacity into the albedo
            # leaves the old Opacity map wired alongside it.
            replaces=["Base_Color", "Diffuse", "Opacity"],
            channels={"RGB": "Base_Color", "A": "Opacity"},
            config_key="albedo_transparency",
            workflows=WF.ALL_ENGINES,
            resolution_critical=True,
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
            resolution_critical=True,
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
            resolution_critical=True,
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
            resolution_critical=True,
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
            channels={"R": "Ambient_Occlusion", "G": "Roughness", "B": "Metallic"},
            config_key="orm_map",
            workflows=[WF.UE, WF.GLTF, WF.GODOT],
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
            # mode=None: preserve the natural mode produced by pack_msao_texture
            # (RGBA for the default HDRP Mask Map layout, RGB for the 3-channel parallel layout).
            mode=None,
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
                # Carried in the alpha channel — omitting it left a loose
                # Smoothness map wired alongside the mask map (the channels
                # consistency check in __post_init__ now enforces this).
                "Smoothness",
                "Specular",
                "Glossiness",
                "Detail",
                "Detail_Mask",
                "Metallic_Smoothness",
            ],
            # Canonical HDRP Mask Map layout; Detail is a filler channel
            # ("?") — coverage must not demand a loose Detail_Mask for it.
            channels={
                "R": "Metallic",
                "G": "Ambient_Occlusion",
                "B": "Detail_Mask?",
                "A": "Smoothness",
            },
            config_key="mask_map",
            workflows=[WF.HDRP],
        ),
        "MRAO": MapType(
            name="MRAO",
            aliases=[
                "Metallic_RoughnessAO",
                "MetallicRoughnessAO",
                "MetallicRoughAO",
                "MetallicRoughness_AO",
                "MetallicRoughnessAmbientOcclusion",
                "MetallicRoughnessOcclusion",
                "MetalRoughAO",
                "MetalRoughAmbientOcclusion",
                "MRA",
            ],
            color_space="Linear",
            # mode=None: preserve the natural mode produced by pack_mrao_texture
            # (RGB for the default 3-channel layout, RGBA for the MSAO mirror).
            mode=None,
            default_background=(0, 0, 255, 255),
            is_packed=True,
            scale_as_mask=True,
            output_fallbacks=[
                "ORM",
                "Ambient_Occlusion",
                "Roughness",
                "Metallic",
            ],
            replaces=[
                "Metallic",
                "Ambient_Occlusion",
                "Roughness",
                "Smoothness",
                "Glossiness",
            ],
            channels={"R": "Metallic", "G": "Roughness", "B": "Ambient_Occlusion"},
            config_key="mrao_map",
            workflows=[],
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
            replaces=["Metallic", "Roughness", "Smoothness", "Glossiness"],
            channels={"RGB": "Metallic", "A": "Smoothness"},
            config_key="metallic_smoothness",
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
            resolution_critical=True,
        ),
        "Emissive_Mask": MapType(
            name="Emissive_Mask",
            # Region-group gate mask (see engines/textures/region_masks.py):
            # per-channel group coverage, not color data — linear, never packed
            # into other maps, and safe to scale down with the other masks.
            aliases=[
                "EmissiveMask",
                "EmissiveGroups",
                "EMask",
            ],
            color_space="Linear",
            mode="RGBA",
            default_background=(0, 0, 0, 0),
            scale_as_mask=True,
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
            # The canonical name is always a resolution candidate; no aliases.
            aliases=[],
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

    # Derived-view caches. Built lazily on first call and held until the next
    # register() call invalidates them (see _invalidate_caches).
    _sorted_candidates: Optional[list] = None
    _resolve_cache: Optional[dict] = None
    _suffix_strip_pattern: Optional[str] = None
    _map_types_cache: Optional[dict] = None
    _aliases_by_len_desc: Optional[list] = None

    def get(self, name: str) -> Optional[MapType]:
        """Get a map type by name."""
        return self._maps.get(name)

    @classmethod
    def _invalidate_caches(cls) -> None:
        """Reset every derived-view cache after the map table changes."""
        cls._sorted_candidates = None
        cls._resolve_cache = None
        cls._suffix_strip_pattern = None
        cls._precedence_rules = None
        cls._map_types_cache = None
        cls._aliases_by_len_desc = None

    def register(self, map_type: MapType, overwrite: bool = False) -> MapType:
        """Register a new map type (or replace an existing one) at runtime.

        The extension point that completes the factory's plug-in story: a
        custom :class:`MapType` registered here is picked up everywhere the
        engine consults the taxonomy — filename resolution, base-name suffix
        stripping, inventory building in ``MapFactory.prepare_maps``, and
        passthrough — so custom conversions/handlers can receive inputs the
        built-in table doesn't know about.

        Registration is process-wide (the registry is a singleton backed by
        class state) and invalidates all derived caches. Longer names/aliases
        win over shorter ones during filename resolution, exactly as with the
        built-in types.

        Idempotent under module reload: re-registering a definition equal to
        the current one is a no-op (dataclass value equality), so module-level
        ``register()`` calls survive the reload cycles DCC tooling lives by.
        A *different* definition under an existing name raises unless
        ``overwrite`` is set, so two tools can't silently fight over a name.

        Parameters:
            map_type: The map type definition to add.
            overwrite: Allow replacing an already-registered type of the same
                name with a different definition.

        Returns:
            MapType: The registered definition (for chaining).

        Raises:
            TypeError: ``map_type`` is not a :class:`MapType`.
            ValueError: A different definition is already registered under
                this name and ``overwrite`` is False.
        """
        if not isinstance(map_type, MapType):
            raise TypeError(
                f"Expected a MapType, got {type(map_type).__name__!r}"
            )
        existing = self._maps.get(map_type.name)
        if existing is not None and not overwrite:
            if existing == map_type:
                return existing  # no-op: identical definition, caches stay warm
            raise ValueError(
                f"Map type {map_type.name!r} is already registered with a "
                "different definition. Pass overwrite=True to replace it."
            )
        self._maps[map_type.name] = map_type
        self._invalidate_caches()
        return map_type

    def _get_sorted_candidates(self):
        """Return the pre-computed sorted alias→map_name list."""
        if self._sorted_candidates is None:
            candidates = []
            for name, m in self._maps.items():
                candidates.append((name, name))
                for alias in m.aliases:
                    candidates.append((alias, name))
            candidates.sort(key=lambda x: len(x[0]), reverse=True)
            self.__class__._sorted_candidates = candidates
            self.__class__._resolve_cache = {}
        return self._sorted_candidates

    # Logical PBR channel -> canonical map type. The join between "what socket a
    # shader read this texture from" (the DCCs' vocabulary) and "what kind of map
    # it is" (this registry's). Both live above pythontk and cannot import each
    # other, so the table belongs here.
    #
    # Deliberately conservative: a channel says only which INPUT consumed the
    # file, never how it is packed. An MSAO wired into a metallic slot is still an
    # MSAO, and only its FILENAME reveals that. So this is the fallback for files
    # that classify to nothing -- never a replacement for filename classification.
    LOGICAL_CHANNEL_TYPES = {
        "baseColor": "Base_Color",
        "emission": "Emissive",
        "specular": "Specular",
        "roughness": "Roughness",
        "metallic": "Metallic",
        "opacity": "Opacity",
        "normal": "Normal",
        "ambientOcclusion": "Ambient_Occlusion",
        # Blender-side only: a Bump node and a Normal Map node feed the SAME
        # Principled input, so that producer can tell the two apart where Maya's
        # single ``normalCamera`` plug cannot. Kept in the shared vocabulary so
        # one table serves both directions.
        "bump": "Bump",
        "height": "Height",
    }

    # Every tangent-space normal map type, in the order a shader should PREFER
    # them. An explicitly tagged map outranks the ambiguous generic one: its
    # convention is known, so it can be corrected, while "Normal" can only be
    # guessed at (and flipping a guess inverts a map that may already be right).
    #
    # Lives here for the same reason LOGICAL_CHANNEL_TYPES does: mayatk and
    # blendertk both need this exact ordering and cannot import each other, so
    # two hardcoded copies would be free to drift (measured: they already had —
    # blendertk tried "Normal" FIRST, shadowing a labeled map with an unlabeled
    # one). Also the membership set for "is this a normal map type".
    NORMAL_TYPES = ("Normal_OpenGL", "Normal_DirectX", "Normal")

    @classmethod
    def select_normal_type(cls, available) -> Optional[str]:
        """The single normal map type a shader should wire, out of those present.

        `Normal`, `Normal_OpenGL` and `Normal_DirectX` are three distinct map
        types, so redundancy filtering never collapses them — a set carrying two
        drives one shader input twice and the last connection silently wins.
        Callers use this to pick exactly one.

        Parameters:
            available: Any container of map-type names (a dict keyed by type
                works — membership is all that is tested).

        Returns:
            str | None: The winning type, or None when none is present.
        """
        return next((t for t in cls.NORMAL_TYPES if t in available), None)

    @classmethod
    def resolve_type_from_channel(cls, channel: str) -> Optional[str]:
        """Canonical map type for a logical shader channel, or None if unmapped.

        Case-insensitive. Use ONLY for a file whose filename carries no map-type
        token -- see :attr:`LOGICAL_CHANNEL_TYPES` for why this must not override
        a successful filename classification.
        """
        if not channel:
            return None
        wanted = str(channel).strip().lower()
        for name, map_type in cls.LOGICAL_CHANNEL_TYPES.items():
            if name.lower() == wanted:
                return map_type
        return None

    # A trailing UDIM / UV-tile token, with its leading separator. Three forms:
    #   .1001        4-digit UDIM, restricted to the real range 1001-1999
    #                (1000 + u+1 + v*10, u 0-9). DOT-delimited only -- see below.
    #   .<UDIM>      unexpanded token as Maya/Substance write it (also <UVTILE>)
    #   .u1_v1       Mari/Mudbox explicit UV tile
    # The bare-digit form is deliberately dot-only: `_1024` / `_2048` is the
    # everyday resolution tag, and 1024 sits squarely inside the UDIM range, so
    # accepting `_####` would silently rename every `wall_Normal_1024.png`.
    # The token forms that cannot be confused with anything else (`<UDIM>`,
    # `u#_v#`) are accepted after either separator.
    _TILE_TOKEN_PATTERN = re.compile(
        r"(?:"
        r"\.1[0-9]{3}"
        r"|[._]<(?:UDIM|UVTILE)>"
        r"|[._]u[0-9]+_v[0-9]+"
        r")$",
        re.IGNORECASE,
    )

    @classmethod
    def split_tile_token(cls, name_only: str) -> Tuple[str, str]:
        """Split a trailing UDIM / UV-tile token off an extension-less filename.

        ``os.path.splitext`` removes only the extension, so a tiled filename
        reaches alias matching as ``rock_Normal.1001`` — ending in no map-type
        alias at all. Every consumer of the taxonomy (classification, base-name
        grouping, the factory's inventory) therefore has to strip the tile token
        first, and re-append it when naming outputs so two tiles can't collide.

        Parameters:
            name_only: A filename with its extension already removed.

        Returns:
            tuple[str, str]: ``(stem_without_token, token)``. The token keeps its
            leading separator so ``stem + token`` is the input verbatim; it is
            ``""`` when the name carries no tile token.
        """
        match = cls._TILE_TOKEN_PATTERN.search(name_only)
        if not match:
            return name_only, ""
        return name_only[: match.start()], match.group(0)

    @staticmethod
    def _short_alias_boundary(name_only: str, index: int) -> Optional[str]:
        """What kind of word boundary a short alias starting at *index* sits on.

        Returns:
            str | None: ``"separator"`` for an explicit delimiter (``rock_AO``)
            or the alias standing alone (``N.png``); ``"camel"`` for a
            lowercase->uppercase step (``rockN``); ``None`` for a digit or
            uppercase predecessor, which is what a model/part number looks like
            (``Agilent_E4419B``, ``Agilent_PSG``, ``Agilent_8757D``).

        The two boundary kinds carry different evidence, and the caller demands
        different things of each. A separator is an explicit authoring decision,
        so the alias may be spelled in any case (``rock_ao``). A CamelCase step
        is inferred from case alone, so it only counts when the alias actually
        starts with a capital -- otherwise every ordinary word ending in an
        alias letter would classify (``wood_green`` -> ``n`` -> Normal).
        """
        if index <= 0:
            return "separator"
        prev = name_only[index - 1]
        if prev in "_-. ":
            return "separator"
        return "camel" if prev.islower() else None

    def resolve_type_from_path(self, path: str) -> Optional[str]:
        """Resolve the map type key from a file path.

        Prioritizes longer matches to prevent short aliases (e.g. 'S') from
        matching longer names (e.g. 'Smoothness').

        A trailing UDIM / UV-tile token is stripped first (see
        :meth:`split_tile_token`) so a tiled map classifies exactly like its
        untiled twin.
        """
        filename = os.path.basename(path)
        name_only, _ = os.path.splitext(filename)
        name_only, _tile = self.split_tile_token(name_only)

        # Check cache first
        if self._resolve_cache is not None and name_only in self._resolve_cache:
            return self._resolve_cache[name_only]

        all_candidates = self._get_sorted_candidates()
        result = None

        for alias, map_name in all_candidates:
            # Logic for short aliases (<= 3 chars)
            if len(alias) <= 3:
                # Must sit at the end of the string, on a real word boundary.
                if name_only.lower().endswith(alias.lower()):
                    suffix_start_index = len(name_only) - len(alias)
                    suffix_in_name = name_only[suffix_start_index:]

                    # A short alias glued to the tail of a model/part number is a
                    # false positive that silently wires a color map into the
                    # wrong socket ("Agilent_E4419B" -> "B" -> Bump, measured on
                    # a production scene). Require a real boundary first; the
                    # sibling MapFactory.resolve_map_type(key=False) path has
                    # always demanded one, so this makes the two agree.
                    boundary = self._short_alias_boundary(
                        name_only, suffix_start_index
                    )
                    if boundary is None:
                        continue

                    # After a separator the suffix is explicit, so honor the
                    # everyday lowercase spellings ("rock_ao", "rock_nrm", and
                    # the classic _d/_n/_s convention) — get_suffix_strip_pattern
                    # has always stripped those case-insensitively, so requiring
                    # a capital here made base names and classification disagree.
                    # A CamelCase boundary is inferred from case alone and keeps
                    # demanding the capital: without it every word ending in an
                    # alias letter would classify ("wood_green" -> "n").
                    if boundary == "separator" or suffix_in_name[0].isupper():
                        result = map_name
                        break
            else:
                # Long aliases: Case-insensitive
                if name_only.lower().endswith(alias.lower()):
                    result = map_name
                    break

        # Cache the result (including None for misses)
        if self._resolve_cache is not None:
            self._resolve_cache[name_only] = result
        return result

    def get_suffix_strip_pattern(self) -> Optional[str]:
        """Regex matching one trailing map-type suffix (any registered alias).

        Single source of truth for base-name resolution — both
        ``MapFactory.get_base_texture_name`` and ``ImgUtils.get_base_texture_name``
        consume this pattern (they once carried drifted copies of it).

        Matching rules:
        - Underscore-delimited suffixes match case-insensitively at any length
          (``brick_ao`` → ``brick``) — the explicit ``_`` boundary makes false
          positives unlikely.
        - Attached suffixes are case-insensitive only when longer than 3 chars;
          short ones require a capital first letter AND a lowercase character
          immediately before them (``brickAO``, not ``brickao`` and not
          ``Agilent_E4419B``) so ordinary words and model numbers aren't misread
          as map types. That lowercase-boundary rule is the same one
          :meth:`resolve_type_from_path` applies, so base names and classification
          agree about what counts as a suffix.

        Returns:
            str | None: The compiled-ready pattern, or None when no maps are
            registered.
        """
        if self._suffix_strip_pattern is None:
            all_aliases = sorted(
                {a for aliases in self.get_map_types().values() for a in aliases},
                key=len,
                reverse=True,
            )
            if not all_aliases:
                return None

            p_underscore = "|".join(re.escape(s) for s in all_aliases)
            pattern_underscore = f"_(?i:{p_underscore})$"

            short_suffixes = [s for s in all_aliases if len(s) <= 3]
            long_suffixes = [s for s in all_aliases if len(s) > 3]

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
                # Require a CamelCase step down from lowercase ("rockN"), matching
                # resolve_type_from_path's boundary rule. Without it a short alias
                # glued to a model number is stripped as a suffix
                # ("Agilent_E4419B" -> "Agilent_E4419") while the resolver -- which
                # does enforce the boundary -- reports no map type at all, so base
                # names and classification disagree and texture SETS mis-group.
                attached_parts.append(f"(?<=[a-z])(?:{'|'.join(p_short_parts)})")

            pattern_attached = f"(?:{'|'.join(attached_parts)})$"

            self.__class__._suffix_strip_pattern = (
                f"(?:{pattern_underscore}|{pattern_attached})"
            )
        return self._suffix_strip_pattern

    def get_workflow_presets(self) -> Dict[str, Dict[str, Any]]:
        """Generate the workflow presets dictionary."""
        presets = {}

        for workflow_name, settings in self._workflow_settings.items():
            # Create preset with defaults
            preset = _WorkflowPreset(**settings)

            # Enable the flag for every map this workflow uses. config_key is
            # the SSoT for the flag gating a packed map (e.g. MSAO ->
            # "mask_map"); loose maps fall back to name inference.
            for m in self._maps.values():
                if workflow_name in m.workflows:
                    key = m.config_key or m.name.lower()
                    if hasattr(preset, key):
                        setattr(preset, key, True)
                    elif hasattr(preset, f"{key}_map"):
                        setattr(preset, f"{key}_map", True)

            presets[workflow_name] = preset.to_dict()

        return presets

    def get_map_types(self) -> Dict[str, Tuple[str, ...]]:
        """Return ``{canonical_key: (canonical, *aliases)}`` for every registered map."""
        if self._map_types_cache is None:
            self.__class__._map_types_cache = {
                name: tuple([name] + m.aliases) for name, m in self._maps.items()
            }
        return self._map_types_cache

    def get_aliases_by_len_desc(self) -> List[str]:
        """Every registered canonical name and alias, sorted longest-first.

        Cached alongside the other derived views (and invalidated with them);
        consumed by ``MapFactory.resolve_map_type(key=False)`` for verbatim
        suffix matching.
        """
        if self._aliases_by_len_desc is None:
            self.__class__._aliases_by_len_desc = sorted(
                {a for v in self.get_map_types().values() for a in v},
                key=len,
                reverse=True,
            )
        return self._aliases_by_len_desc

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
            # Write through the class: an instance attribute would shadow the
            # class-level cache and survive _invalidate_caches().
            self.__class__._precedence_rules = {
                name: m.replaces for name, m in self._maps.items() if m.replaces
            }
        return self._precedence_rules

    def get_scale_as_mask_types(self) -> List[str]:
        """Get list of map types that should be scaled as masks."""
        return [name for name, m in self._maps.items() if m.scale_as_mask]

    def get_resolution_critical_types(self) -> List[str]:
        """Get list of map types whose surface detail requires full resolution."""
        return [name for name, m in self._maps.items() if m.resolution_critical]

    def is_resolution_critical(self, name: str) -> bool:
        """True when surface detail for ``name`` requires full resolution.

        Unknown names default to True (treat as critical) so callers don't
        silently downscale maps the registry doesn't recognise.
        """
        m = self._maps.get(name)
        return True if m is None else m.resolution_critical

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

            # Apply overrides from dict. "preset" is consumed above — it is
            # not itself a config option and must not leak into the result.
            overrides = {
                k: v for k, v in config.items() if v is not None and k != "preset"
            }
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
