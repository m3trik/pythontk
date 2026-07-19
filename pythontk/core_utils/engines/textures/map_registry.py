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
            input_fallbacks=["Base_Color"],
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
                "Specular",
                "Glossiness",
                "Detail",
                "Detail_Mask",
                "Metallic_Smoothness",
            ],
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

    def resolve_type_from_path(self, path: str) -> Optional[str]:
        """Resolve the map type key from a file path.

        Prioritizes longer matches to prevent short aliases (e.g. 'S') from
        matching longer names (e.g. 'Smoothness').
        """
        filename = os.path.basename(path)
        name_only, _ = os.path.splitext(filename)

        # Check cache first
        if self._resolve_cache is not None and name_only in self._resolve_cache:
            return self._resolve_cache[name_only]

        all_candidates = self._get_sorted_candidates()
        result = None

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
                            result = map_name
                            break
                    else:
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
          short ones require a capital first letter (``brickAO``, not
          ``brickao``) so ordinary words aren't misread as map types.

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
                attached_parts.append("|".join(p_short_parts))

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
