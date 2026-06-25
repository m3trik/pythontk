# !/usr/bin/python
# coding=utf-8
"""``MapFactory`` -- the texture-map workflow orchestrator.

Public surface is unchanged: ``from pythontk import MapFactory`` and
``from pythontk.img_utils.map_factory import MapFactory`` resolve here via the
package ``__init__``. Split out of the original single-file module; the
conversion registry, processing context, and workflow handlers now live in
sibling modules.
"""
import os
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    TYPE_CHECKING,
)

try:
    import numpy as np
except ImportError:
    np = None
try:
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter
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
from .conversions import MapConversion, ConversionRegistry
from .processor import TextureProcessor, DEFAULT_EXTENSION, ALPHA_EXTENSION
from .handlers import (
    WorkflowHandler,
    BaseColorHandler,
    NormalMapHandler,
    ORMMapHandler,
    MRAOMapHandler,
    MaskMapHandler,
    MetallicSmoothnessHandler,
    OutputFallbackHandler,
    SeparateMetallicRoughnessHandler,
)


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
        # When set (a WF profile key), per-map output format is resolved from the
        # profile's output template instead of the single global output_extension.
        "output_profile": None,
        "use_input_fallbacks": True,
        "use_output_fallbacks": True,
        # Workflow flags
        "albedo_transparency": False,
        "metallic_smoothness": False,
        "mask_map": False,
        "mask_map_layout": "rgba",  # "rgba" (HDRP default) or "rgb" (3-channel parallel to MRAO)
        "orm_map": False,
        "mrao_map": False,
        "mrao_layout": "rgb",  # "rgb" (industry default) or "rgba" (mirror of MSAO)
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
        MRAOMapHandler,
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

        # Bump/Height to Normal conversions. Bind the loop var as a default
        # arg — a plain closure late-binds, leaving every registration
        # reading inv["Height"] (KeyError when only a Bump map exists).
        for target in ["Normal_OpenGL", "Normal_DirectX", "Normal"]:
            for source in ["Bump", "Height"]:
                registry.register(
                    target,
                    source,
                    lambda inv, ctx, s=source: ctx.convert_bump_to_normal(inv[s]),
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

        # Packing conversions (MRAO)
        # Priority 10: All components present, native Roughness
        registry.register(
            "MRAO",
            ["Metallic", "Roughness", "Ambient_Occlusion"],
            lambda inv, ctx: ctx.create_mrao_map(inv),
            priority=10,
        )
        # Priority 9: All components present, converted Smoothness
        registry.register(
            "MRAO",
            ["Metallic", "Smoothness", "Ambient_Occlusion"],
            lambda inv, ctx: ctx.create_mrao_map(inv),
            priority=9,
        )
        # Priority 8: Missing AO, native Roughness
        registry.register(
            "MRAO",
            ["Metallic", "Roughness"],
            lambda inv, ctx: ctx.create_mrao_map(inv),
            priority=8,
        )
        # Priority 7: Missing AO, converted Smoothness
        registry.register(
            "MRAO",
            ["Metallic", "Smoothness"],
            lambda inv, ctx: ctx.create_mrao_map(inv),
            priority=7,
        )

        # Unpacking conversions (MRAO)
        registry.register(
            "Metallic",
            "MRAO",
            lambda inv, ctx: ctx.get_metallic_from_mrao(inv["MRAO"]),
            priority=8,
        )
        registry.register(
            "Roughness",
            "MRAO",
            lambda inv, ctx: ctx.get_roughness_from_mrao(inv["MRAO"]),
            priority=8,
        )
        registry.register(
            "Smoothness",
            "MRAO",
            lambda inv, ctx: ctx.get_smoothness_from_mrao(inv["MRAO"]),
            priority=8,
        )
        registry.register(
            "Ambient_Occlusion",
            "MRAO",
            lambda inv, ctx: ctx.get_ao_from_mrao(inv["MRAO"]),
            priority=8,
        )
        registry.register(
            "AO",
            "MRAO",
            lambda inv, ctx: ctx.get_ao_from_mrao(inv["MRAO"]),
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

    _aliases_by_len_desc: List[str] = None

    @classmethod
    def _get_aliases_by_len_desc(cls) -> List[str]:
        """Cached list of every alias across all map types, sorted longest-first.

        Used by `resolve_map_type(key=False)`; mirrors the caching the registry
        already does for `key=True`.
        """
        if cls._aliases_by_len_desc is None:
            cls._aliases_by_len_desc = sorted(
                {a for v in cls.map_types.values() for a in v},
                key=len,
                reverse=True,
            )
        return cls._aliases_by_len_desc

    @classmethod
    def resolve_map_type(cls, file: str, key: bool = True, validate: str = None) -> str:
        """Resolves the map type from a filename or alias using `map_types`.

        Parameters:
            file (str): Image filename, full path, or map type suffix.
            key (bool): If True, return the canonical key from `map_types`
                (e.g. "Ambient_Occlusion").
                If False, return the matched alias **verbatim from the filename**
                so a round-trip through `resolve_texture_filename` does not
                rename the file. Requires an underscore boundary (or full
                filename equality) to avoid mid-word matches like
                "diffuse_cube" matching the single-letter alias "E".
            validate (str, optional): If provided, validate the resolved map
                type against this expected key. Comparison is case-insensitive
                so non-canonical filename casing does not falsely fail.

        Returns:
            str: The map type. None when no alias matched.

        Raises:
            ValueError: If the map type is not the expected type when 'validate' is provided.
        """
        ImgUtils.assert_pathlike(file, "file")
        filename = FileUtils.format_path(file, "name")

        if key:
            result = cls._map_registry.resolve_type_from_path(file)
        else:
            filename_lower = filename.lower()
            result = None
            for alias in cls._get_aliases_by_len_desc():
                alias_lower = alias.lower()
                if filename_lower == alias_lower:
                    result = filename
                    break
                needle = "_" + alias_lower
                if filename_lower.endswith(needle):
                    # Slice the alias out of the original filename to preserve case
                    result = filename[len(filename) - len(alias):]
                    break

        if validate:
            # Case-insensitive: `result` may carry filename casing (key=False)
            # which won't match the canonical-cased registry entries verbatim.
            valid_types_lower = {validate.lower()} | {
                a.lower() for a in cls.map_types[validate]
            }
            if (result or "").lower() not in valid_types_lower:
                raise ValueError(
                    f"Invalid map type '{result}'. Expected type is one of: "
                    f"{[validate] + list(cls.map_types[validate])}"
                )

        return result

    @classmethod
    def resolve_color_space(cls, file: str, default: str = "Linear") -> str:
        """Resolve the working color space ("sRGB" or "Linear") for a texture by filename.

        Looks up the resolved map type's declared ``color_space`` — the SSoT in the map
        registry (Base Color / Albedo / Emissive are sRGB; Normal / Roughness / Metallic and
        the other data maps are Linear). DCC-agnostic: callers translate "Linear" to their own
        raw/data label (Maya's *Raw*, Blender's *Non-Color*).

        Parameters:
            file (str): Image filename, full path, or map-type suffix.
            default (str): Returned when the map type cannot be resolved from the name.

        Returns:
            str: "sRGB" or "Linear" (or ``default`` when unresolved).
        """
        map_type = cls._map_registry.resolve_type_from_path(file)
        entry = cls._map_registry.get(map_type) if map_type else None
        return entry.color_space if entry else default

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

        # If no map type was resolved, we can't safely synthesize a "<base>_<type>"
        # filename without dropping naming detail. Preserve the original path
        # (changing extension if explicitly requested via `ext`).
        if not map_type:
            directory = FileUtils.format_path(texture_path, "path")
            stem, original_ext = os.path.splitext(os.path.basename(texture_path))
            ext_out = (
                f".{ext.lower().lstrip('.')}" if ext else original_ext
            )
            # Idempotent affix application: strip the configured prefix/suffix from
            # the existing stem before re-applying, so "Optimized_foo" + prefix
            # "Optimized_" stays "Optimized_foo" (not "Optimized_Optimized_foo").
            stem_core = StrUtils.strip_known_affix(
                stem, prefix=prefix or "", suffix=suffix or ""
            )
            prefix_str = prefix or ""
            suffix_str = f"_{suffix.lstrip('_')}" if suffix else ""
            return os.path.join(
                directory, f"{prefix_str}{stem_core}{suffix_str}{ext_out}"
            )

        # Extract sections from the given path
        directory = FileUtils.format_path(texture_path, "path")
        # Strip the configured prefix/suffix from the base so we can re-apply
        # them idempotently below.
        base_name = cls.get_base_texture_name(
            texture_path, prefix=prefix or "", suffix=suffix or ""
        )
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
    def get_base_texture_name(
        cls,
        filepath_or_filename: str,
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        """Extracts the base texture name from a filename or path,
        removing known suffixes (e.g., _normal, _roughness).

        Logic:
        - Long suffixes (>3 chars): Case-insensitive.
        - Short suffixes (<=3 chars): Must start with a capital letter (rest case-insensitive) to avoid false positives.

        Parameters:
            filepath_or_filename (str): A texture path or name.
            prefix (str): Optional user-defined prefix to strip from the resolved base
                (case-insensitive). Lets callers safely re-apply it without producing
                e.g. ``Mat_Mat_brick`` when the source filename already had ``Mat_``.
            suffix (str): Optional user-defined suffix to strip from the resolved base.

        Returns:
            str: The base name without map-type suffix, with any configured user prefix/suffix removed.
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

        # Strip any configured user prefix/suffix so callers can re-apply them
        # idempotently, then collapse a trailing underscore (preserves the
        # original behavior for filenames like 'foo_.png' even when no affix
        # was supplied).
        return StrUtils.strip_known_affix(
            base_name, prefix=prefix, suffix=suffix
        ).rstrip("_")

    @classmethod
    def group_textures_by_set(
        cls,
        image_paths: List[str],
        prefix: str = "",
        suffix: str = "",
    ) -> Dict[str, List[str]]:
        """Groups texture maps into sets based on matching base names.

        Parameters:
            image_paths (List[str]): A list of full image file paths.
            prefix (str): Optional prefix to strip from set keys so files like
                ``Mat_brick_Albedo.png`` and ``brick_Normal.png`` group together
                when the caller's affix is ``Mat_``.
            suffix (str): Optional suffix to strip from set keys (same rationale).

        Returns:
            Dict[str, List[str]]: A dictionary where:
                - Keys are unique base texture names.
                - Values are lists of associated texture files.
        """
        texture_sets = {}
        for path in image_paths:
            base_name = cls.get_base_texture_name(
                path, prefix=prefix, suffix=suffix
            )
            if base_name not in texture_sets:
                texture_sets[base_name] = []

            texture_sets[base_name].append(path)

        return texture_sets

    @classmethod
    def _supplement_sets_from_dir(
        cls,
        texture_sets: Dict[str, List[str]],
        directory: str,
        prefix: str = "",
        suffix: str = "",
        logger=None,
    ) -> Dict[str, List[str]]:
        """Gap-fill each texture set with same-base-name siblings from ``directory``.

        For every set, scans ``directory`` for files that resolve to the same base
        name (honoring ``prefix``/``suffix``) and a recognized map type, then appends
        any whose map type is missing from the set. Provided files always win — an
        existing map slot is never replaced and files already present are not
        duplicated. Lets callers pull in required maps that live next to the inputs
        but weren't part of the supplied list (e.g. a Normal sitting in a project's
        ``sourceimages`` that was never wired into the material).

        Parameters:
            texture_sets: Mapping of base name -> file paths (mutated in place).
            directory: Directory to scan for sibling textures.
            prefix: Prefix stripped during base-name resolution (must match set keys).
            suffix: Suffix stripped during base-name resolution.
            logger: Optional logger for reporting discovered files.

        Returns:
            Dict[str, List[str]]: The same ``texture_sets`` mapping, supplemented.
        """
        log = logger or cls.logger
        if not (directory and os.path.isdir(directory)):
            return texture_sets

        dir_files = FileUtils.get_dir_contents(
            directory,
            "filepath",
            inc_files=[f"*.{ext}" for ext in ImgUtils.texture_file_types],
        )
        if not dir_files:
            return texture_sets

        dir_by_set = cls.group_textures_by_set(dir_files, prefix=prefix, suffix=suffix)

        for base_name, files in texture_sets.items():
            siblings = dir_by_set.get(base_name)
            if not siblings:
                continue

            present_types = {cls.resolve_map_type(f) for f in files}
            present_paths = {os.path.normcase(os.path.abspath(f)) for f in files}

            for sib in siblings:
                key = os.path.normcase(os.path.abspath(sib))
                if key in present_paths:
                    continue
                map_type = cls.resolve_map_type(sib)
                if not map_type or map_type in present_types:
                    continue

                files.append(sib)
                present_types.add(map_type)
                present_paths.add(key)
                if log:
                    log.info(
                        f"Discovered {os.path.basename(sib)} ({map_type}) "
                        f"for set '{base_name}'"
                    )

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
    def filter_redundant_maps(
        cls, sorted_maps: Dict[str, List[str]], config: Dict[str, Any] = None
    ) -> None:
        """Resolve packed/loose map redundancy in-place.

        A packed map (ORM/MSAO/MRAO) and its loose components (Metallic,
        Roughness, AO, …) are mutually redundant — wiring both fights over the
        same material slots. Which one wins depends on the target workflow:

        - **Packed workflow** — the packed map is a requested output, or no
          ``config`` is supplied (legacy behavior): the packed map supersedes
          its loose components, which are dropped.
        - **Unpacked workflow** — ``config`` is supplied and the packed map is
          *not* requested (e.g. the "PBR Metallic/Roughness" preset with
          ``mask_map=False``): the packed map is itself the redundant one and is
          dropped in favor of the present loose components, so the separate
          Metallic/Roughness/AO maps connect instead of the mask map.

        Modifies the sorted_maps dictionary in-place.

        Parameters:
            sorted_maps: Dictionary of map types to file path(s). Mutated in place.
            config: Optional workflow config. When provided, redundancy direction
                follows each packed map's ``config_key`` flag (plus
                ``force_packed_maps``). When omitted, packed maps always win.
        """
        precedence_rules = cls.get_precedence_rules()
        registry = cls._map_registry

        for dominant, redundants in precedence_rules.items():
            if not (dominant in sorted_maps and sorted_maps[dominant]):
                continue

            # Does the target workflow actually want this packed map as output?
            # Default True keeps legacy "packed wins" behavior when no config.
            packed_requested = True
            if config is not None:
                map_def = registry.get(dominant)
                key = map_def.config_key if map_def else None
                if key:
                    packed_requested = bool(config.get(key)) or bool(
                        config.get("force_packed_maps")
                    )

            if packed_requested:
                # Packed map supersedes its loose components.
                for redundant in redundants:
                    if redundant in sorted_maps:
                        cls.logger.info(
                            f"Skipping {redundant} map (replaced by {dominant})",
                            extra={"preset": "highlight"},
                        )
                        del sorted_maps[redundant]
            elif any(r in sorted_maps and sorted_maps[r] for r in redundants):
                # Unpacked workflow with loose components present: drop the packed
                # map so the separate maps win the material slots.
                cls.logger.info(
                    f"Skipping {dominant} map (separate maps present)",
                    extra={"preset": "highlight"},
                )
                del sorted_maps[dominant]

    @classmethod
    def prepare_maps(
        cls,
        source: Union[str, List[str]],
        output_dir: str = None,
        group_by_set: bool = True,
        max_workers: int = 1,
        progress_callback: Callable = None,
        prefix: str = "",
        suffix: str = "",
        discover_dir: str = None,
        **kwargs,
    ) -> Union[List[str], Dict[str, List[str]]]:
        """
        Main factory method. Automatically handles batch processing.

        Parameters:
            source: A directory path (str), a single file path (str), or a list of file paths.
            output_dir: Optional output directory.
            group_by_set: Whether to automatically group textures into sets (default: True).
                          If False, all input files are treated as a single set.
            discover_dir: Optional directory to scan for same-base-name sibling
                          textures that aren't in ``source``. Any whose map type is
                          missing from a set is pulled in (gap-fill); provided files
                          always win — a present map type is never replaced. Honors
                          ``prefix``/``suffix`` when matching base names.
            max_workers: Number of threads for parallel processing.
            progress_callback: Optional callback(current, total, message) for reporting progress.
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
            texture_sets = cls.group_textures_by_set(
                files, prefix=prefix, suffix=suffix
            )
        else:
            # Treat all files as a single set
            # Use the common prefix or just the first file's base name as the key
            base_name = cls.get_base_texture_name(
                files[0], prefix=prefix, suffix=suffix
            )
            # Copy so the working set never aliases the caller's input list
            # (discovery and downstream steps append/edit it).
            texture_sets = {base_name: list(files)}

        # Gap-fill each set with same-base-name siblings found on disk.
        if discover_dir:
            texture_sets = cls._supplement_sets_from_dir(
                texture_sets,
                discover_dir,
                prefix=prefix,
                suffix=suffix,
                logger=logger,
            )

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

                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_set):
                    completed_count += 1
                    # Retrieve the original task arguments
                    _, base_name_task, _ = future_to_set[future]
                    
                    if progress_callback:
                        progress_callback(
                            completed_count, total_sets, f"Processed {base_name_task}"
                        )

                    base_name, generated = future.result()
                    if generated:
                        results[base_name] = generated
        else:
            for i, (base_name, textures) in enumerate(texture_sets.items(), 1):
                if progress_callback:
                    progress_callback(i, total_sets, f"Processing {base_name}")

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

        # Pre-process: Spec/Gloss conversion (only if explicitly requested)
        if convert and workflow_config.get("convert_specgloss_to_pbr", False):
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
            output_profile=workflow_config.get("output_profile"),
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
        cls,
        image: Union[str, "Image.Image"],
        threshold: float = 0.25,
        min_gradient_std: float = 1.0,
    ) -> Optional[str]:
        """Detects if a normal map is OpenGL (Y+) or DirectX (Y-) based on surface integrability.

        Theory:
        If a normal map represents a continuous height field H over image
        coordinates (x = column, y = row, row increasing DOWNWARD):
        Red channel   R ~ -dH/dx              (both formats)
        Green channel G ~ +dH/dy (OpenGL)     (image top = V max, so the
                      Y-up green component equals the row-down derivative)
                      G ~ -dH/dy (DirectX)

        Cross derivatives of a real height field are equal
        (d²H/dxdy = d²H/dydx), therefore:
        corr(dR/dy, dG/dx) < 0  -> OpenGL
        corr(dR/dy, dG/dx) > 0  -> DirectX
        (Verified against a labeled real-world OpenGL map: r = -0.19.)

        Parameters:
            image (str | PIL.Image.Image): Input normal map.
            threshold (float): Correlation magnitude required to call a format.
                0.25 is empirically conservative — small biases on near-flat
                inputs (e.g. baked maps with large neutral backgrounds) can
                still produce |r| around 0.1, so anything looser is noise.
            min_gradient_std (float): Per-channel gradient std-dev floor
                (8-bit units). When both dR/dy and dG/dx are below this floor
                the image is effectively flat and correlation is meaningless;
                returns None rather than emitting a confident-looking guess.

        Returns:
            str | None: "OpenGL", "DirectX", or None if indeterminate.
        """
        try:
            img = ImgUtils.ensure_image(image)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 512x512 is plenty for statistical analysis
            if max(img.size) > 512:
                img.thumbnail((512, 512))

            arr = np.array(img, dtype=np.float32)
            R = arr[:, :, 0]
            G = arr[:, :, 1]

            # dR/dy along image rows; dG/dx along image cols.
            dRy = np.gradient(R, axis=0).ravel()
            dGx = np.gradient(G, axis=1).ravel()

            # Variance floor: flat or near-flat inputs produce meaningless
            # correlations (often NaN, often spuriously signed).
            if dRy.std() < min_gradient_std or dGx.std() < min_gradient_std:
                return None

            correlation = np.corrcoef(dRy, dGx)[0, 1]
            if not np.isfinite(correlation):
                return None

            if correlation < -threshold:
                return "OpenGL"
            if correlation > threshold:
                return "DirectX"
            return None

        except Exception as e:
            cls.logger.warning(f"Error detecting normal map format: {e}")
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
        ImgUtils.save_image(inverted_image, output_path, **kwargs)
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

        # Calculate normal vectors. The surface normal of z=H is
        # (-dH/dx, -dH/dy_up, 1). grad_y is the IMAGE-ROW derivative
        # (row increases downward), and textures display right side up
        # (image top = V max), so dH/dy_up = -grad_y and the green (Y-up)
        # component is +grad_y. The old `-grad_y` silently produced
        # DirectX orientation under an OpenGL label (verified against a
        # labeled real-world map via the integrability correlation).
        normal_x = -grad_x
        normal_y = grad_y
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
        ImgUtils.save_image(normal_image, output_path, **kwargs)

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

        ImgUtils.save_image(base_color, base_color_file)
        ImgUtils.save_image(metallic, metallic_file)
        ImgUtils.save_image(roughness, roughness_file)

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
    def pack_orm_texture(
        cls,
        ao_map_path: Optional[str],
        roughness_map_path: Optional[str],
        metallic_map_path: Optional[str],
        output_dir: str = None,
        suffix: str = "_ORM",
        invert_roughness: bool = False,
        output_path: str = None,
        save: bool = True,
    ) -> Union[str, "Image.Image"]:
        """Pack AO (R) + Roughness (G) + Metallic (B) into a single ORM texture.

        Parameters:
            ao_map_path (str): AO texture. Can be None (fills white).
            roughness_map_path (str): Roughness texture. Can be None (fills black).
            metallic_map_path (str): Metallic texture. Can be None (fills black).
            output_dir (str, optional): Output directory. Defaults to the first source's directory.
            suffix (str, optional): Suffix for the output file name.
            invert_roughness (bool, optional): Treat ``roughness_map_path`` as Smoothness and invert it.
            output_path (str, optional): Explicit output path. Overrides output_dir/suffix logic.
            save (bool, optional): If True, saves to disk. If False, returns PIL Image.

        Returns:
            str | Image.Image: Path to the packed ORM texture or PIL Image.
        """
        if ao_map_path and isinstance(ao_map_path, str):
            ImgUtils.assert_pathlike(ao_map_path, "ao_map_path")
        if roughness_map_path and isinstance(roughness_map_path, str):
            ImgUtils.assert_pathlike(roughness_map_path, "roughness_map_path")
        if metallic_map_path and isinstance(metallic_map_path, str):
            ImgUtils.assert_pathlike(metallic_map_path, "metallic_map_path")

        if save and output_path is None:
            source_map = ao_map_path or roughness_map_path or metallic_map_path
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

        return ImgUtils.pack_channels(
            channel_files={
                "R": ao_map_path,
                "G": roughness_map_path,
                "B": metallic_map_path,
            },
            output_path=output_path,
            out_mode="RGB",
            invert_channels=["G"] if invert_roughness else None,
            fill_values={"R": 255, "G": 0, "B": 0},
            save=save,
        )

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
        layout: str = "rgba",
    ) -> Union[str, "Image.Image"]:
        """Pack Metallic + AO + Smoothness (and optional Detail) into a single MSAO texture.

        Parameters:
            metallic_map_path (str): Path to the metallic texture map.
            ao_map_path (str): Path to the ambient occlusion texture map. Can be None (fills with white).
            alpha_map_path (str): Path to the smoothness/roughness texture map. Can be None (fills with white).
            detail_map_path (str, optional): Path to the detail mask map (RGBA layout only).
            output_dir (str, optional): Output directory. If None, uses the first source map's directory.
            suffix (str, optional): Suffix for the output file name.
            invert_alpha (bool, optional): If True, inverts the smoothness channel (roughness → smoothness).
            layout (str, optional): ``"rgba"`` (default; HDRP Mask Map: R=M, G=AO, B=Detail, A=S) or
                ``"rgb"`` (3-channel parallel to MRAO: R=M, G=S, B=AO).
            output_path (str, optional): Explicit output path. Overrides output_dir/suffix logic.
            save (bool, optional): If True, saves to disk. If False, returns PIL Image.

        Returns:
            str | Image.Image: Path to the packed MSAO texture or PIL Image.
        """
        layout = (layout or "rgba").lower()
        if layout not in ("rgba", "rgb"):
            raise ValueError(f"Unsupported MSAO layout: {layout!r}")

        if isinstance(metallic_map_path, str):
            ImgUtils.assert_pathlike(metallic_map_path, "metallic_map_path")
        if ao_map_path and isinstance(ao_map_path, str):
            ImgUtils.assert_pathlike(ao_map_path, "ao_map_path")
        if alpha_map_path and isinstance(alpha_map_path, str):
            ImgUtils.assert_pathlike(alpha_map_path, "alpha_map_path")
        if detail_map_path and isinstance(detail_map_path, str):
            ImgUtils.assert_pathlike(detail_map_path, "detail_map_path")

        if save and output_path is None:
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

        if layout == "rgb":
            # 3-channel parallel layout: R=Metallic, G=Smoothness, B=AO
            return ImgUtils.pack_channels(
                channel_files={
                    "R": metallic_map_path,
                    "G": alpha_map_path,
                    "B": ao_map_path,
                },
                output_path=output_path,
                out_mode="RGB",
                invert_channels=["G"] if invert_alpha else None,
                fill_values={"R": 0, "G": 255, "B": 255},
                save=save,
            )

        # Default HDRP Mask Map layout: R=Metallic, G=AO, B=Detail, A=Smoothness
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
            fill_values={"G": 255, "B": 0, "A": 255},
            save=save,
        )

    @classmethod
    def pack_mrao_texture(
        cls,
        metallic_map_path: Optional[str],
        roughness_map_path: Optional[str],
        ao_map_path: Optional[str],
        detail_map_path: Optional[str] = None,
        output_dir: str = None,
        suffix: str = "_MRAO",
        invert_roughness: bool = False,
        output_path: str = None,
        save: bool = True,
        layout: str = "rgb",
    ) -> Union[str, "Image.Image"]:
        """Pack Metallic + Roughness + AO (and optional Detail) into a single MRAO texture.

        Parameters:
            metallic_map_path (str): Metallic texture. Can be None (fills black).
            roughness_map_path (str): Roughness texture. Can be None (fills black).
            ao_map_path (str): AO texture. Can be None (fills white).
            detail_map_path (str, optional): Detail mask (RGBA layout only).
            output_dir (str, optional): Output directory. If None, uses the first source map's directory.
            suffix (str, optional): Suffix for the output file name.
            invert_roughness (bool, optional): Treat ``roughness_map_path`` as Smoothness and invert it.
            layout (str, optional): ``"rgb"`` (default; industry standard: R=M, G=R, B=AO) or
                ``"rgba"`` (mirror of MSAO: R=M, G=AO, B=Detail, A=R).
            output_path (str, optional): Explicit output path. Overrides output_dir/suffix logic.
            save (bool, optional): If True, saves to disk. If False, returns PIL Image.

        Returns:
            str | Image.Image: Path to the packed MRAO texture or PIL Image.
        """
        layout = (layout or "rgb").lower()
        if layout not in ("rgb", "rgba"):
            raise ValueError(f"Unsupported MRAO layout: {layout!r}")

        if metallic_map_path and isinstance(metallic_map_path, str):
            ImgUtils.assert_pathlike(metallic_map_path, "metallic_map_path")
        if roughness_map_path and isinstance(roughness_map_path, str):
            ImgUtils.assert_pathlike(roughness_map_path, "roughness_map_path")
        if ao_map_path and isinstance(ao_map_path, str):
            ImgUtils.assert_pathlike(ao_map_path, "ao_map_path")
        if detail_map_path and isinstance(detail_map_path, str):
            ImgUtils.assert_pathlike(detail_map_path, "detail_map_path")

        if save and output_path is None:
            source_map = (
                metallic_map_path or roughness_map_path or ao_map_path or detail_map_path
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

        if layout == "rgba":
            # Mirror of MSAO: R=Metallic, G=AO, B=Detail, A=Roughness
            return ImgUtils.pack_channels(
                channel_files={
                    "R": metallic_map_path,
                    "G": ao_map_path,
                    "B": detail_map_path,
                    "A": roughness_map_path,
                },
                output_path=output_path,
                out_mode="RGBA",
                invert_channels=["A"] if invert_roughness else None,
                fill_values={"R": 0, "G": 255, "B": 0, "A": 0},
                save=save,
            )

        # Default 3-channel industry layout: R=Metallic, G=Roughness, B=AO
        return ImgUtils.pack_channels(
            channel_files={
                "R": metallic_map_path,
                "G": roughness_map_path,
                "B": ao_map_path,
            },
            output_path=output_path,
            out_mode="RGB",
            invert_channels=["G"] if invert_roughness else None,
            fill_values={"R": 0, "G": 0, "B": 255},
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
        ImgUtils.save_image(roughness_image, output_path, **kwargs)

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
        ImgUtils.save_image(smoothness_image, output_path, **kwargs)

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

    @staticmethod
    def _detect_packed_layout(source: Union[str, "Image.Image"]) -> str:
        """Return ``"rgba"`` if ``source`` has an alpha channel, else ``"rgb"``."""
        try:
            img = ImgUtils.ensure_image(source)
            return "rgba" if "A" in img.getbands() else "rgb"
        except Exception:
            return "rgba"

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
        layout: Optional[str] = None,
        **kwargs,
    ) -> Union[
        Tuple[str, str, str], Tuple["Image.Image", "Image.Image", "Image.Image"]
    ]:
        """Unpack Metallic, AO, and Smoothness from a combined MSAO texture.

        Layout is auto-detected from the image mode when not specified:
        - ``"rgba"`` (HDRP Mask Map): R=Metallic, G=AO, B=Detail, A=Smoothness.
        - ``"rgb"`` (3-channel parallel to MRAO): R=Metallic, G=Smoothness, B=AO.

        Returns the (metallic, ao, smoothness) tuple regardless of layout.
        """
        resolved_layout = (
            (layout or "").lower() or cls._detect_packed_layout(msao_map_path)
        )
        if resolved_layout not in ("rgba", "rgb"):
            raise ValueError(f"Unsupported MSAO layout: {layout!r}")

        if resolved_layout == "rgb":
            channel_config = {
                "R": {"suffix": metallic_suffix},
                "G": {"suffix": smoothness_suffix, "invert": invert_smoothness},
                "B": {"suffix": ao_suffix},
            }
            results = ImgUtils.extract_channels(
                msao_map_path,
                channel_config,
                output_dir=output_dir,
                save=save,
                **kwargs,
            )
            # (metallic, ao, smoothness)
            return results.get("R"), results.get("B"), results.get("G")

        channel_config = {
            "R": {"suffix": metallic_suffix},
            "G": {"suffix": ao_suffix},
            "A": {"suffix": smoothness_suffix, "invert": invert_smoothness},
        }
        results = ImgUtils.extract_channels(
            msao_map_path, channel_config, output_dir=output_dir, save=save, **kwargs
        )
        return results.get("R"), results.get("G"), results.get("A")

    @classmethod
    def unpack_mrao_texture(
        cls,
        mrao_map_path: str,
        output_dir: str = None,
        metallic_suffix: str = "_Metallic",
        roughness_suffix: str = "_Roughness",
        ao_suffix: str = "_AO",
        invert_roughness: bool = False,
        save: bool = True,
        layout: Optional[str] = None,
        **kwargs,
    ) -> Union[
        Tuple[str, str, str], Tuple["Image.Image", "Image.Image", "Image.Image"]
    ]:
        """Unpack Metallic, Roughness, and AO from a combined MRAO texture.

        Layout is auto-detected from the image mode when not specified:
        - ``"rgb"`` (industry default): R=Metallic, G=Roughness, B=AO.
        - ``"rgba"`` (mirror of MSAO): R=Metallic, G=AO, B=Detail, A=Roughness.

        Returns the (metallic, roughness, ao) tuple regardless of layout.
        """
        resolved_layout = (
            (layout or "").lower() or cls._detect_packed_layout(mrao_map_path)
        )
        if resolved_layout not in ("rgb", "rgba"):
            raise ValueError(f"Unsupported MRAO layout: {layout!r}")

        if resolved_layout == "rgba":
            channel_config = {
                "R": {"suffix": metallic_suffix},
                "G": {"suffix": ao_suffix},
                "A": {"suffix": roughness_suffix, "invert": invert_roughness},
            }
            results = ImgUtils.extract_channels(
                mrao_map_path,
                channel_config,
                output_dir=output_dir,
                save=save,
                **kwargs,
            )
            # (metallic, roughness, ao)
            return results.get("R"), results.get("A"), results.get("G")

        channel_config = {
            "R": {"suffix": metallic_suffix},
            "G": {"suffix": roughness_suffix, "invert": invert_roughness},
            "B": {"suffix": ao_suffix},
        }
        results = ImgUtils.extract_channels(
            mrao_map_path, channel_config, output_dir=output_dir, save=save, **kwargs
        )
        return results.get("R"), results.get("G"), results.get("B")

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
