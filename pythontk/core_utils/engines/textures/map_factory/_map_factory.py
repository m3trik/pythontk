# !/usr/bin/python
# coding=utf-8
"""``MapFactory`` -- the texture-map workflow orchestrator.

Public API is unchanged: ``from pythontk import MapFactory`` resolves through the
lazy root as before; the internal path is now
``from pythontk.core_utils.engines.textures.map_factory import MapFactory`` (relocated
from ``img_utils`` into the ``core_utils/engines/textures`` engine namespace),
resolving here via the package ``__init__``. Split out of the original single-file
module; the conversion registry, processing context, and workflow handlers now
live in sibling modules.
"""

import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
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
    # Every imported name gets bound -- see the note on the same guard in
    # ``pythontk/img_utils/_img_utils.py``. An unbound name is a ``NameError`` at the
    # call site, and it cannot be repaired by a late Pillow install.
    Image = ImageOps = ImageEnhance = ImageFilter = None

if TYPE_CHECKING:
    from PIL import Image

# From this package:
from pythontk.core_utils.cancel_scope import CancelScope, OperationCancelled
from pythontk.core_utils.class_property import ClassProperty
from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.img_utils._img_utils import ImgUtils
from pythontk.file_utils._file_utils import FileUtils
from pythontk.iter_utils._iter_utils import IterUtils
from pythontk.str_utils._str_utils import StrUtils
from pythontk.core_utils.engines.textures.map_registry import MapRegistry
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
        # What a packed map does when its source channels aren't all resolvable
        # (MapRegistry.MISSING_SKIP / _MULTI / _FORCE). None defers to the legacy
        # ``force_packed_maps`` bool, then to MISSING_SKIP — see
        # MapRegistry.resolve_missing_map_rule.
        "missing_map_rule": None,
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

    # Live views over the registry (not import-time snapshots) so map types
    # added via MapRegistry.register() are honored everywhere — filename
    # resolution, inventory building, passthrough, and mask scaling.
    @ClassProperty
    def map_types(cls) -> Dict[str, Tuple[str, ...]]:
        """``{canonical_key: (canonical, *aliases)}`` for every registered map."""
        return cls._map_registry.get_map_types()

    @ClassProperty
    def passthrough_maps(cls) -> List[str]:
        """Maps passed through to the output when no handler consumes them."""
        return cls._map_registry.get_passthrough_maps()

    @ClassProperty
    def packed_grayscale_maps(cls) -> List[str]:
        """Maps that scale down by ``mask_map_scale`` (packed/mask data)."""
        return cls._map_registry.get_scale_as_mask_types()

    @ClassProperty
    def map_fallbacks(cls) -> Dict[str, Tuple[str, ...]]:
        """Safe input substitutes per map type (e.g. Bump -> Normal)."""
        return cls._map_registry.get_fallbacks()

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

    @classmethod
    def _get_aliases_by_len_desc(cls) -> List[str]:
        """Every alias across all map types, longest-first.

        Used by `resolve_map_type(key=False)`. The cache lives on the registry
        (the alias owner) so registration invalidates it with the other views.
        """
        return cls._map_registry.get_aliases_by_len_desc()

    @classmethod
    def resolve_map_type(cls, file: str, key: bool = True, validate: str = None) -> str:
        """Resolves the map type from a filename or alias using `map_types`.

        Parameters:
            file (str): Image filename, full path, or map type suffix.
            key (bool): If True, return the canonical key from `map_types`
                (e.g. "Ambient_Occlusion").
                If False, return the matched alias **verbatim from the filename**
                so a round-trip through `resolve_texture_filename` does not
                rename the file. Requires a separator boundary
                (``MapRegistry.SEPARATORS``) or full filename equality, to avoid
                mid-word matches like "diffuse_cube" matching the single-letter
                alias "E".

                The two forms deliberately DISAGREE about a trailing duplicate
                marker: ``key=True`` retries past it (see
                ``MapRegistry.split_duplicate_token``) because classifying is
                what a caller wants there, while ``key=False`` stays strict
                because its answer is spliced back into a FILENAME. Making it
                tolerant would have ``resolve_texture_filename`` append rather
                than replace -- ``X_Base_Color_1.png`` + ``"Base_Color"`` is
                ``X_Base_Color_1_Base_Color.png``, and the strict ``None``
                yields the empty suffix that leaves the name untouched. Do not
                "harmonize" them without changing that splice first.
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
            separators = cls._map_registry.SEPARATORS
            result = None
            for alias in cls._get_aliases_by_len_desc():
                alias_lower = alias.lower()
                if filename_lower == alias_lower:
                    result = filename
                    break
                # Any registry separator counts, not `_` alone: this path fed
                # `resolve_texture_filename`, so a `-`/`.`/space-delimited suffix
                # was invisible here while the other two suffix implementations
                # accepted it, and the round-trip renamed the file.
                start = len(filename) - len(alias)
                if (
                    start > 0
                    and filename_lower.endswith(alias_lower)
                    and filename[start - 1] in separators
                ):
                    # Slice the alias out of the original filename to preserve case
                    result = filename[start:]
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
            ext_out = f".{ext.lower().lstrip('.')}" if ext else original_ext
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

        # A tile token trails the map type (matching TextureProcessor.output_path_for)
        # so the output is still a readable tile sequence AND two tiles of the same
        # material resolve to different paths instead of overwriting each other.
        tile_token = cls.get_tile_token(texture_path)

        # Construct the final filename correctly
        new_name = StrUtils.replace_placeholders(
            "{prefix}{base_name}_{map_type}{suffix}{tile}{ext}",
            prefix=prefix or "",
            base_name=base_name,
            map_type=map_type,
            suffix=suffix,
            tile=tile_token,
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

        Delegates to ``ImgUtils.get_base_texture_name``, which owns the single
        implementation — see it for the matching rules. The two were parallel
        implementations documented as unable to drift, and drifted anyway (the
        UDIM/UV-tile token came off on this side only), so one now calls the other
        rather than mirroring it.

        Parameters:
            filepath_or_filename (str): A texture path or name.
            prefix (str): Optional user-defined prefix to strip from the resolved base
                (case-insensitive). Lets callers safely re-apply it without producing
                e.g. ``Mat_Mat_brick`` when the source filename already had ``Mat_``.
            suffix (str): Optional user-defined suffix to strip from the resolved base.

        Returns:
            str: The base name without map-type suffix, with any configured user prefix/suffix removed.
        """
        return ImgUtils.get_base_texture_name(
            filepath_or_filename, prefix=prefix, suffix=suffix
        )

    @classmethod
    def get_tile_token(cls, filepath_or_filename: str) -> str:
        """The UDIM / UV-tile token on a texture filename, or ``""``.

        The counterpart to :meth:`get_base_texture_name`, which drops the token:
        together they split a tiled filename into the part every tile of a
        material shares and the part that distinguishes one tile from the next.
        Output naming re-appends this so two tiles cannot resolve to one path.

        Parameters:
            filepath_or_filename (str): A texture path or name.

        Returns:
            str: The token including its leading separator (``".1001"``,
            ``"_<UDIM>"``), or ``""`` when the name carries none.
        """
        ImgUtils.assert_pathlike(filepath_or_filename, "filepath_or_filename")

        filename = os.path.basename(str(filepath_or_filename))
        name_only, _ = os.path.splitext(filename)
        return cls._map_registry.split_tile_token(name_only)[1]

    @classmethod
    def group_textures_by_set(
        cls,
        image_paths: List[str],
        prefix: str = "",
        suffix: str = "",
    ) -> Dict[str, List[str]]:
        """Groups texture maps into sets based on matching base names.

        A UDIM/UV-tile token keeps its own set: one tile is one complete,
        independently processable collection of maps (its own image data, its
        own output file), so ``rock.1001`` and ``rock.1002`` are separate keys
        while the maps WITHIN each tile group together. Collapsing tiles into
        one set instead would overflow the factory's one-path-per-map-type
        inventory and silently drop every tile but the last.

        Parameters:
            image_paths (List[str]): A list of full image file paths.
            prefix (str): Optional prefix to strip from set keys so files like
                ``Mat_brick_Albedo.png`` and ``brick_Normal.png`` group together
                when the caller's affix is ``Mat_``.
            suffix (str): Optional suffix to strip from set keys (same rationale).

        Returns:
            Dict[str, List[str]]: A dictionary where:
                - Keys are unique base texture names (``<base><tile token>``).
                - Values are lists of associated texture files.
        """
        texture_sets = {}
        for path in image_paths:
            base_name = cls.get_base_texture_name(path, prefix=prefix, suffix=suffix)
            key = f"{base_name}{cls.get_tile_token(path)}"
            if key not in texture_sets:
                texture_sets[key] = []

            texture_sets[key].append(path)

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

        For every set, scans ``directory`` **and its subdirectories** for files
        that resolve to the same base
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

        # Recursive: the caller hands over a texture ROOT (a Maya project's
        # ``sourceImages`` rule, say), and that root is routinely one folder
        # per asset rather than a flat pile. A root-only scan finds nothing in
        # that layout -- the maps sitting beside the ones already wired are one
        # level down -- so discovery silently no-ops exactly where it is needed.
        # Matching stays base-name + map-type, so depth widens the search
        # without widening what can match.
        dir_files = FileUtils.get_dir_contents(
            directory,
            "filepath",
            recursive=True,
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
        """Register a custom workflow handler (extensibility).

        Idempotent under re-registration: a handler with the same
        module+qualname — including the *new* class object a module reload
        produces — replaces its previous registration in place instead of
        duplicating it in the pipeline.
        """
        key = (handler_class.__module__, handler_class.__qualname__)
        for i, existing in enumerate(cls._workflow_handlers):
            if (existing.__module__, existing.__qualname__) == key:
                cls._workflow_handlers[i] = handler_class
                return
        # Before the trailing fallback/default handlers.
        cls._workflow_handlers.insert(-2, handler_class)

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
    def resolve_normal_maps(
        cls,
        sorted_maps: Dict[str, Any],
        target_format: Optional[str] = None,
        convert: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        """Reduce an inventory to exactly ONE normal map — optionally in a given convention.

        The sibling of :meth:`filter_redundant_maps`, for the redundancy that
        one cannot see. `Normal`, `Normal_OpenGL` and `Normal_DirectX` are three
        distinct map types with no ``replaces`` relationship, so redundancy
        filtering never collapses them — yet they all drive the SAME shader
        input. A set carrying two wires that input twice and the last connection
        silently wins.

        ``target_format`` additionally guarantees the survivor's handedness.
        Every consumer needs the same reduction but corrects the convention
        differently: a renderer whose shading graph can flip green does it
        there and passes ``None``; one that cannot has to correct the FILE and
        names its own convention. Neither is knowledge this layer has, which is
        exactly why the convention is a parameter rather than a branch.

        An explicitly tagged map of the opposite convention is converted (green
        inverted, written beside the source under the target's name). The
        ambiguous generic ``Normal`` is NEVER converted: its convention is
        unknown, so flipping it would invert a map that may already be correct.

        Modifies ``sorted_maps`` in place, like :meth:`filter_redundant_maps`.
        Values may be paths or lists of paths (both caller shapes preserved).

        Parameters:
            sorted_maps: ``{map_type: path-or-[paths]}``. Mutated in place.
            target_format: ``"OpenGL"`` / ``"DirectX"`` (matched
                case-insensitively) to guarantee the survivor's convention, or
                None to keep whatever wins as-is. An unrecognised convention
                warns and leaves the map untouched rather than inventing a type.
            convert: Allow writing the converted sibling. When False a
                mismatched map is kept unchanged rather than converted.

        Returns:
            dict: ``{"dropped": {map_type: reason}, "converted": {map_type: path}}``
            — ``converted`` names the NEW type and path when a conversion ran,
            so a caller tracking real files can pick it up.
        """
        report: Dict[str, Dict[str, str]] = {"dropped": {}, "converted": {}}
        registry = cls._map_registry

        present = [t for t in registry.NORMAL_TYPES if t in sorted_maps]
        winner_type = registry.select_normal_type(sorted_maps)
        if winner_type is None:
            return report

        for map_type in present:
            if map_type == winner_type:
                continue
            cls.logger.info(
                f"Skipping {map_type} map (superseded by the {winner_type} normal map)",
                extra={"preset": "highlight"},
            )
            del sorted_maps[map_type]
            report["dropped"][map_type] = f"superseded by the {winner_type} normal map"

        if not target_format or not convert:
            return report

        # Convention name -> canonical map type, derived from the registry so a
        # newly registered tagged type is understood without editing this, and
        # so the caller's spelling is normalized. `convert_normal_map_format`
        # takes its format case-insensitively, so a caller passing "opengl" is
        # reasonable -- and would otherwise build the map type "Normal_opengl",
        # putting a type nothing in the taxonomy recognises into the inventory.
        conventions = {
            t[len("Normal_") :].lower(): t
            for t in registry.NORMAL_TYPES
            if t.startswith("Normal_")
        }
        target_type = conventions.get(str(target_format).strip().lower())
        if target_type is None:
            cls.logger.warning(
                f"Unknown normal map convention {target_format!r} "
                f"(expected one of {sorted(conventions)}); leaving the map as-is."
            )
            return report

        # Only an explicitly tagged opposite is convertible; the generic map's
        # convention is unknown and a match needs no work.
        if winner_type == target_type or winner_type not in conventions.values():
            return report

        value = sorted_maps[winner_type]
        is_list = isinstance(value, (list, tuple))
        source = (value[0] if value else None) if is_list else value
        if not source:
            return report  # nothing to convert (empty entry)
        try:
            converted = cls.convert_normal_map_format(
                source, target_format=target_format.lower()
            )
        except Exception as error:
            cls.logger.warning(
                f"Could not convert {winner_type} to {target_type} ({error}); "
                "keeping the source map."
            )
            return report

        if not converted:
            return report

        del sorted_maps[winner_type]
        sorted_maps[target_type] = [converted] if is_list else converted
        # The source type leaves the inventory, so it belongs in `dropped` too:
        # a caller mapping the report back onto its own list of real file paths
        # would otherwise keep the original file alongside the converted one and
        # wire the normal slot twice — the exact double-wire this method exists
        # to prevent. Every type that leaves the inventory is reported.
        report["dropped"][winner_type] = f"converted to {target_type}"
        report["converted"][target_type] = converted
        return report

    @classmethod
    def filter_redundant_maps(
        cls,
        sorted_maps: Dict[str, Any],
        config: Dict[str, Any] = None,
        extract_missing: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        """Resolve packed-map redundancy in-place — losslessly.

        A packed map (ORM/MSAO/MRAO/…) is redundant against two different
        things, and both are resolved here because either one left standing
        wires the same material slots twice.

        **Rival packings** run first (:meth:`_resolve_packed_conflicts`). Two
        packings can carry the same channels — ORM and the HDRP mask map both
        drive metallic / roughness / AO — and ``replaces`` cannot express that,
        so the loser is chosen by :meth:`MapRegistry.packed_precedence` and its
        uncovered channels extracted before it drops.

        **Loose components** (Metallic, Roughness, AO, …) are then weighed
        against the surviving packing. Which side wins depends on the target
        workflow:

        - **Packed workflow** — the packed map is a requested output, or no
          ``config`` is supplied (legacy behavior): the packed map supersedes
          its loose components, which are dropped.
        - **Unpacked workflow** — ``config`` is supplied and the packed map is
          *not* requested (e.g. the "PBR Metallic/Roughness" preset with
          ``mask_map=False``): the packed map is redundant *where its channels
          are covered*. Coverage is judged per channel against the map's
          declared ``channels`` layout, dynamically: a channel counts as
          covered when its type — or any loose type the conversion registry can
          derive it from (Roughness covers a Smoothness channel) — survives the
          drop. Channels nothing covers are **extracted to real loose maps**
          from the packed source before it is dropped, so no data is ever lost
          (the shipped example: an MSAO beside loose Metallic/Roughness but no
          separate AO used to lose its AO channel here). When extraction is
          unavailable (``extract_missing=False``, missing Pillow, or the packed
          entry is not a readable file), the packed map is kept and its loose
          components retire instead — the lossless direction.
        - A packed map with **no loose components at all** is the same case with
          nothing covered: every channel it carries is extracted and it retires.
          A preset has to mean the same thing whatever the material happened to
          start with, and this used to be the one shape that quietly kept the
          packing — so one material came out packed and its neighbour unpacked
          purely on whether a stray loose map sat beside it. The lossless
          fallback above still applies: a packing whose channels cannot be
          recovered is kept, because dropping it would lose all of them, as is
          one whose channels the registry does not describe at all and that has
          no loose component to take the slots over.

        Modifies ``sorted_maps`` in place. Values may be file paths or lists of
        paths (both caller shapes are preserved).

        Parameters:
            sorted_maps: ``{map_type: path-or-[paths]}``. Mutated in place.
            config: Optional workflow config. When provided, redundancy
                direction follows each packed map's ``config_key`` flag (plus
                any ``missing_map_rule`` past ``skip``); ``dry_run`` plans
                extractions without writing them. When omitted, a packed map
                always wins against its LOOSE components — rival packings are
                still reduced to one, since two of them driving the same slots
                was never a legitimate outcome to preserve.
            extract_missing: Allow extracting uncovered channels to files.

        Returns:
            dict: ``{"dropped": {map_type: reason}, "extracted": {map_type: path}}``.
        """
        report: Dict[str, Dict[str, str]] = {"dropped": {}, "extracted": {}}
        precedence_rules = cls.get_precedence_rules()
        registry = cls._map_registry

        def drop(map_type: str, reason: str) -> None:
            cls._drop_map(sorted_maps, report, map_type, reason)

        # Rival PACKED maps first: `replaces` cannot express that conflict, so
        # leaving it to the loop below lets the requested packing retire the
        # loose components and the rival then read as a sole source.
        cls._resolve_packed_conflicts(sorted_maps, config, extract_missing, report)

        for dominant, declared_redundants in precedence_rules.items():
            if not (dominant in sorted_maps and sorted_maps[dominant]):
                continue

            # LOOSE components only. A `replaces` entry naming another PACKING
            # (MSAO lists Metallic_Smoothness) would let this pass retire a
            # rival on name alone — no ranking, no coverage check — and so
            # overturn the packed-vs-packed pass above, which may have kept
            # both deliberately because dropping one would lose a channel.
            redundants = [r for r in declared_redundants if cls._is_loose(r)]

            # Does the target workflow actually want this packed map as output?
            # Default True keeps legacy "packed wins" behavior when no config.
            packed_requested = True
            map_def = registry.get(dominant)
            if config is not None:
                key = map_def.config_key if map_def else None
                if key:
                    packed_requested = (
                        bool(config.get(key))
                        or registry.resolve_missing_map_rule(config)
                        != registry.MISSING_SKIP
                    )

            if packed_requested:
                # Packed map supersedes its loose components.
                for redundant in redundants:
                    if redundant in sorted_maps:
                        drop(redundant, f"superseded by {dominant}")
                continue

            # Unpacked workflow. Judge coverage per declared channel: covered
            # when the carried type — or a loose type a registered conversion
            # derives it from — survives the drop. With no loose components at
            # all nothing is covered, so this extracts the whole packing, which
            # is the point: the preset decides the shape, not the input set.
            present = {
                t
                for t, v in sorted_maps.items()
                if v and t != dominant and cls._is_loose(t)
            }

            carried = map_def.carried_types() if map_def else []
            if not (
                carried or any(r in sorted_maps and sorted_maps[r] for r in redundants)
            ):
                # A packing that carries nothing checkable, with no loose
                # component standing by to take the slots over: coverage cannot
                # be judged and there is nothing to extract, so the drop below
                # would be a pure loss. ``MapType.__post_init__`` already
                # refuses a packing with no ``channels`` at all, which leaves
                # one shape — every channel marked OPTIONAL, so
                # ``carried_types()`` skips them all. Rare, and only reachable
                # through a caller-registered type, but the failure is a map
                # silently vanishing, and keeping it is this function's standing
                # answer to "cannot be shown redundant". A packing that carries
                # real channels needs no guard: they are extracted first.
                continue

            uncovered = [t for t in carried if not cls._channel_covered(t, present)]

            if uncovered:
                extracted = None
                if extract_missing:
                    extracted = cls.extract_channels(
                        dominant,
                        cls._first_path(sorted_maps[dominant]),
                        uncovered,
                        config,
                    )
                if extracted is None:
                    # Can't recover the uncovered channels — keeping the packed
                    # map is the only lossless direction; its loose components
                    # retire so the slots still have one source each.
                    reason = (
                        f"superseded by {dominant} (its "
                        f"{', '.join(uncovered)} channel has no loose source; "
                        "extraction unavailable)"
                    )
                    for redundant in redundants:
                        if redundant in sorted_maps:
                            drop(redundant, reason)
                    continue

                cls._absorb_extracted(sorted_maps, report, dominant, extracted)

            drop(dominant, "superseded by separate maps")

        return report

    # --- redundancy internals, shared by both passes -----------------------

    @staticmethod
    def _first_path(value) -> Optional[str]:
        """The single path behind an inventory value (path or list of paths)."""
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        return value if isinstance(value, str) else None

    @classmethod
    def _drop_map(
        cls,
        sorted_maps: Dict[str, Any],
        report: Dict[str, Dict[str, str]],
        map_type: str,
        reason: str,
    ) -> None:
        """Remove ``map_type`` from the inventory and record why."""
        cls.logger.info(
            f"Skipping {map_type} map ({reason})",
            extra={"preset": "highlight"},
        )
        del sorted_maps[map_type]
        report["dropped"][map_type] = reason

    @classmethod
    def _absorb_extracted(
        cls,
        sorted_maps: Dict[str, Any],
        report: Dict[str, Dict[str, str]],
        source_type: str,
        extracted: Dict[str, str],
    ) -> None:
        """Add extracted loose maps to the inventory, keeping its value shape."""
        as_list = isinstance(sorted_maps[source_type], (list, tuple))
        for map_type, path in extracted.items():
            sorted_maps[map_type] = [path] if as_list else path
            report["extracted"][map_type] = path
            cls.logger.info(
                f"Extracted {map_type} from {source_type} ({path})",
                extra={"preset": "highlight"},
            )

    @classmethod
    def _is_loose(cls, map_type: str) -> bool:
        """Is ``map_type`` a separate map rather than a packing?

        Unknown types count as loose: a type the registry does not define
        cannot be asserted to pack anything.
        """
        map_def = cls._map_registry.get(map_type)
        return not (map_def and map_def.is_packed)

    @classmethod
    def _channel_covered(cls, carried_type: str, present: Set[str]) -> bool:
        """Does anything in ``present`` source ``carried_type``?

        Covered when the type is present outright, or when a single-source
        conversion derives it from something that is. That second arm is what
        makes the judgement work across packings as well as loose maps: the
        registry declares ``Roughness <- Smoothness`` *and*
        ``Roughness <- ORM``, so a surviving ORM demonstrably covers an MSAO's
        metallic / AO / smoothness channels. Shared by both redundancy passes
        so that "covered" means one thing.
        """
        if carried_type in present:
            return True
        for conv in cls._conversion_registry.get_conversions_for(carried_type):
            if len(conv.source_types) == 1 and conv.source_types[0] in present:
                return True
        return False

    @classmethod
    def _resolve_packed_conflicts(
        cls,
        sorted_maps: Dict[str, Any],
        config: Optional[Dict[str, Any]],
        extract_missing: bool,
        report: Dict[str, Dict[str, str]],
    ) -> None:
        """Reduce rival PACKED maps to the one this workflow wants.

        The packed-vs-loose pass cannot do this. Precedence is keyed off
        :attr:`MapType.replaces`, which lists the LOOSE maps a packing absorbs
        — no packing lists another — so two packings never meet there. Worse,
        the loose pass actively hides the conflict: the requested packing
        retires the loose Metallic/Roughness/AO first, after which the rival
        finds no loose components left and takes the "sole source of its
        channels" branch. Measured live on a glTF 2.0 conversion, which
        connected an ORM *and* an HDRP MSAO to the same three slots.

        Rivalry is judged by channel coverage, not by name: a packing is a
        rival only when a more-preferred one covers at least one channel it
        carries, so ``Albedo_Transparency`` (base colour + opacity) is never
        weighed against an ORM. Preference is
        :meth:`MapRegistry.packed_precedence` — a total order, so the survivor
        does not depend on which rival was judged first.

        Lossless, like the loose pass: channels nothing else supplies are
        extracted from the loser before it is dropped, and if extraction is
        unavailable the loser is KEPT (with a warning) rather than taking its
        only copy of a channel with it. "Nothing else" counts surviving LOOSE
        maps as well as the winners — otherwise a channel the caller already
        listed loose is extracted anyway, and their entry is replaced by
        derived data.

        Modifies ``sorted_maps`` and ``report`` in place.
        """
        registry = cls._map_registry
        order = [
            t
            for t in registry.packed_precedence(config)
            if t in sorted_maps and sorted_maps[t]
        ]
        if len(order) < 2:
            return

        # Least-preferred first: a loser is judged against the survivors that
        # outrank it, so a three-way pile-up collapses in one pass. Descending
        # is what makes `order[:rank]` safe to use unfiltered — only indices
        # ABOVE the current one have been dropped, so every more-preferred
        # entry is still in `sorted_maps`.
        for rank in range(len(order) - 1, 0, -1):
            loser = order[rank]
            winners = order[:rank]

            map_def = registry.get(loser)
            carried = map_def.carried_types() if map_def else []

            # Rivalry is decided against the WINNERS alone. A channel some
            # loose map happens to cover says nothing about whether two
            # PACKINGS collide — an ORM beside a loose Metallic is the loose
            # pass's business, and crediting that here would make the ORM look
            # like a rival of whatever packing outranked it.
            rivals = set(winners)
            uncovered = [t for t in carried if not cls._channel_covered(t, rivals)]
            if len(uncovered) == len(carried):
                continue  # shares no channel with any winner — not a rival

            # For what must be EXTRACTED, a surviving loose map counts too:
            # re-extracting a channel one already supplies swaps the caller's
            # own entry for derived data (measured: an `asset_Mixed_AO.png`
            # replaced by an extracted `asset_Ambient_Occlusion.png` — the
            # canonical-name guard in `extract_channels` cannot
            # catch it, since the caller's file need not use that name).
            # Excluded are any the winners will retire below: `replaces` may
            # name a type its packing does not carry (MSAO lists Specular),
            # which would leave that channel with no source at all.
            retired = {r for w in winners for r in registry.get(w).replaces}
            loose = {
                t
                for t, v in sorted_maps.items()
                if v and t != loser and t not in retired and cls._is_loose(t)
            }
            uncovered = [t for t in uncovered if not cls._channel_covered(t, loose)]

            if uncovered:
                extracted = (
                    cls.extract_channels(
                        loser,
                        cls._first_path(sorted_maps[loser]),
                        uncovered,
                        config,
                    )
                    if extract_missing
                    else None
                )
                if extracted is None:
                    # Keeping both double-wires the shared slots, but dropping
                    # the loser would lose a channel outright. Say so instead
                    # of picking silently.
                    cls.logger.warning(
                        f"{loser} loses to {winners[0]} as a rival packing, but "
                        f"its {', '.join(uncovered)} channel has no other "
                        "source and extraction is unavailable — keeping both. "
                        "They will drive the same material slots."
                    )
                    continue
                cls._absorb_extracted(sorted_maps, report, loser, extracted)

            cls._drop_map(
                sorted_maps,
                report,
                loser,
                f"superseded by {winners[0]} (rival packing for the same channels)",
            )

    @classmethod
    def extract_channels(
        cls,
        packed_type: str,
        packed_path: Optional[str],
        targets: List[str],
        config: Dict[str, Any] = None,
    ) -> Optional[Dict[str, str]]:
        """Extract ``targets`` from a packed map into loose files beside it.

        Reuses the conversion registry (the same unpack conversions
        ``prepare_maps`` runs on) and the ``TextureProcessor`` save pipeline —
        so naming, mode enforcement, and ``dry_run`` behave exactly like every
        other generated map. All-or-nothing: if any target cannot be derived,
        returns None so the caller can fall back to keeping the packed map.
        A target already on disk under its canonical name is REUSED, never
        overwritten — the caller's own maps outrank derived data.

        Public because a shader that cannot sample a channel needs exactly this:
        StingrayPBS binds a ``TEX_*`` slot only through the compound plug, so a
        packed map can drive ONE slot and its other channels have to become
        images of their own (``mayatk.GameShader``).

        Parameters:
            packed_type: The packed map's canonical type (e.g. "MSAO").
            packed_path: Path to the packed texture file.
            targets: Map types to extract (uncovered channels only).
            config: Workflow config (``dry_run`` is honored by ``save_map``).

        Returns:
            dict | None: ``{map_type: saved_path}``, or None when unavailable.
        """
        if Image is None or not (packed_path and os.path.isfile(packed_path)):
            return None

        context = TextureProcessor(
            inventory={packed_type: packed_path},
            config=dict(config or {}),
            output_dir=os.path.dirname(packed_path),
            base_name=cls.get_base_texture_name(packed_path),
            tile_token=cls.get_tile_token(packed_path),
            ext=(config or {}).get("output_extension")
            or os.path.splitext(packed_path)[1].lstrip("."),
            conversion_registry=cls._conversion_registry,
            logger=cls.logger,
        )

        extracted: Dict[str, str] = {}
        for target in targets:
            # A real loose map already on disk under the canonical name wins —
            # overwriting it with extracted channel data would destroy user
            # files the caller simply didn't list.
            candidate = context.output_path_for(target)
            if os.path.isfile(candidate):
                cls.logger.info(
                    f"Reusing existing {target} map instead of extracting "
                    f"from {packed_type}: {candidate}",
                    extra={"preset": "highlight"},
                )
                extracted[target] = candidate
                continue

            try:
                image = context.resolve_map(target, allow_conversion=True)
            except Exception as e:
                cls.logger.warning(f"Extracting {target} from {packed_type}: {e}")
                return None
            if not image:
                return None
            extracted[target] = context.save_map(
                image, target, source_images=[packed_path]
            )
        return extracted

    @staticmethod
    def _checkpoint(progress_result: Any = None) -> None:
        """Cooperative cancel point for the batch loops.

        Raises :class:`~pythontk.OperationCancelled` on either signal: the
        ambient :class:`~pythontk.CancelScope` -- what a mayatk/blendertk
        ``@Cancelable`` slot arms -- or a ``progress_callback`` that returned
        ``False``, the progress-bar ``update()`` contract ``CancelScope.tick``
        is written to match. ``None`` is deliberately NOT a cancel: a callback
        that only prints returns it.

        Call this on the thread that owns the scope. The ambient scope is a
        ``ContextVar``, and a ``ThreadPoolExecutor`` worker starts with an empty
        context, so a checkpoint inside a submitted task never fires -- the
        parallel branch checks as it submits and collects instead.
        """
        if progress_result is False:
            raise OperationCancelled(
                "Texture batch cancelled from the progress callback"
            )
        CancelScope.check()

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
            discover_dir: Optional directory tree to scan for same-base-name
                          sibling textures that aren't in ``source``. Scanned
                          RECURSIVELY, so a per-asset subfolder layout is
                          reached. Any whose map type is
                          missing from a set is pulled in (gap-fill); provided files
                          always win — a present map type is never replaced. Honors
                          ``prefix``/``suffix`` when matching base names.
            max_workers: Number of threads for parallel processing.
            progress_callback: Optional callback(current, total, message) for
                reporting progress. Returning ``False`` cancels the batch --
                the progress-bar ``update()`` contract; any other return value,
                ``None`` included, continues.
            **kwargs: Configuration options overriding DEFAULT_CONFIG.
                      Key options:
                      - use_input_fallbacks (bool): Allow generating maps from alternative inputs (e.g. Diffuse -> Base Color).
                      - use_output_fallbacks (bool): Allow substituting missing maps with alternatives (e.g. AO -> Mask).
                      - convert (bool): Enable format conversion/renaming.
                      - optimize (bool): Enable image optimization.
                      - missing_map_rule (str): What a packed map does when its source
                        channels aren't all resolvable - "skip" (default), "multi"
                        (pack once 2+ channels resolved), or "force" (always pack).
                      - force_packed_maps (bool): Legacy alias for missing_map_rule="force".

        Returns:
            List[str] if a single asset was processed.
            Dict[str, List[str]] if multiple assets were processed (keyed by asset name).

        Raises:
            OperationCancelled: The ambient :class:`~pythontk.CancelScope` was
                cancelled, or *progress_callback* returned ``False``. Checked
                between sets, so the run stops at a set boundary rather than
                returning a half-finished batch a caller would wire up as
                complete.
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

        # Before anything is queued: an already-cancelled scope must not spin
        # up a pool or touch the first set.
        cls._checkpoint()

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

                    reported = (
                        progress_callback(
                            completed_count, total_sets, f"Processed {base_name_task}"
                        )
                        if progress_callback
                        else None
                    )
                    try:
                        cls._checkpoint(reported)
                    except OperationCancelled:
                        # Drop everything still queued before unwinding: the
                        # executor's __exit__ waits for shutdown, and would
                        # otherwise run the whole remaining batch anyway.
                        for pending in future_to_set:
                            pending.cancel()
                        raise

                    base_name, generated = future.result()
                    if generated:
                        results[base_name] = generated
        else:
            for i, (base_name, textures) in enumerate(texture_sets.items(), 1):
                reported = (
                    progress_callback(i, total_sets, f"Processing {base_name}")
                    if progress_callback
                    else None
                )
                cls._checkpoint(reported)

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
            tile_token=MapFactory.get_tile_token(reference_path),
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

        result = output_maps if output_maps else textures

        # Retire the inputs this run replaced. Opt-in: absent `old_files_folder`
        # the sources are left exactly where they were (the long-standing
        # default). When `result is textures` nothing was superseded, so the
        # loop below finds no candidates and the folder is never created.
        old_files_folder = workflow_config.get("old_files_folder")
        if old_files_folder and not workflow_config.get("dry_run", False):
            cls._archive_superseded(
                textures,
                result,
                old_files_folder,
                output_dir or os.path.dirname(reference_path),
                logger=logger,
            )

        return result

    @classmethod
    def _archive_superseded(
        cls,
        sources: List[str],
        outputs: List[str],
        old_files_folder: str,
        output_dir: str,
        logger: Any = None,
    ) -> List[str]:
        """Move each source the output set replaced into the archive folder.

        A source is *superseded* when it is not itself part of the result — it
        was consumed into a packed map, re-encoded under a new extension, or
        (the common case) canonicalized from an alias, which
        :meth:`TextureProcessor.process_map` performs as a COPY. Without this
        the alias survives beside its canonical twin and the folder accumulates
        a duplicate per run.

        Parameters:
            sources: The set's input paths.
            outputs: The paths this run is returning.
            old_files_folder: Archive folder; relative names resolve against
                ``output_dir``.
            output_dir: Directory the run wrote to.
            logger: Optional logger for per-file reporting.

        Returns:
            list[str]: The source paths that were archived.
        """
        kept = {
            os.path.normcase(os.path.normpath(p)) for p in outputs if isinstance(p, str)
        }
        archive_dir = (
            old_files_folder
            if os.path.isabs(old_files_folder)
            else os.path.join(output_dir, old_files_folder)
        )
        archive_norm = os.path.normcase(os.path.normpath(archive_dir))

        archived = []
        for src in sources:
            if not isinstance(src, str) or not os.path.isfile(src):
                continue
            src_norm = os.path.normcase(os.path.normpath(src))
            if src_norm in kept:
                continue
            # Never re-archive something already sitting in the archive folder
            # (a re-run over a directory that includes it).
            if os.path.normcase(os.path.dirname(src_norm)) == archive_norm:
                continue
            try:
                FileUtils.move_file(src, archive_dir, overwrite=True, create_dir=True)
                archived.append(src)
            except OSError as e:  # shutil.Error subclasses OSError
                if logger:
                    logger.warning(f"Could not archive '{os.path.basename(src)}': {e}")

        if archived and logger:
            logger.info(
                f"Archived {len(archived)} superseded map(s) to "
                f"'{os.path.basename(archive_dir.rstrip(os.sep))}'"
            )
        return archived

    @staticmethod
    def _build_map_inventory(textures: List[str]) -> Dict[str, str]:
        """Build map inventory using ImgUtils."""
        inventory = {}
        # Prefer more specific FILENAMES (Mixed_AO over AO) when two files
        # resolve to the same type — basename length, not full-path length,
        # which would let a longer directory name decide the winner.
        for texture in sorted(
            textures, key=lambda t: len(os.path.basename(t)), reverse=True
        ):
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

        Measured behavior (synthetic height fields x {clean, JPEG q40-70,
        quarter-res}, 42 cases): 40 correct, 2 indeterminate, 0 wrong-sign.
        Non-normal inputs (photographs, random noise, flat fills, OBJECT-space
        normals) all fall below the threshold and return None rather than
        guessing.

        How strong the evidence is varies far more by map than "|r| ~ 0.64-0.95"
        once suggested here: measured across four real production OpenGL bakes,
        |r| ranges 0.19 to 0.77. Deep, high-contrast relief lands near the top
        (a turret bake: 0.77); shallow relief over a large neutral field lands
        near the bottom (a 4096 hook/pin bake: 0.19) and legitimately abstains
        at the default threshold. The SIGN was correct in all four, which is
        what the statistic is really good for -- it is much better at "not
        backwards" than at "confident".

        Known blind spot: the statistic measures the RELATIVE handedness of the
        two channels, so it cannot tell "G is inverted" from "R is inverted". A
        map whose RED channel was flipped (an X- bake, or a mirrored-UV export)
        reports the opposite convention with full confidence. Filename evidence
        outranks this function wherever both exist -- which is why the caller
        only consults it for a map classified as the ambiguous generic
        ``Normal`` (see NormalMapHandler), never to override a ``Normal_OpenGL``
        or ``Normal_DirectX`` tag.

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
            # convert("RGB") always returns our own copy (PIL copies even when
            # the mode already matches), so the in-place thumbnail() below never
            # mutates a caller-supplied Image.
            img = ImgUtils.ensure_image(image).convert("RGB")

            # Reducing first keeps the common case cheap, but it is a LOW-PASS
            # over exactly the gradients this statistic reads, so it cannot be
            # the last word: on maps whose relief is fine and shallow it
            # averages the signal flat (measured on real OpenGL bakes: r fell
            # -0.368 -> -0.105 and -0.19 -> -0.09, both under the threshold, a
            # correct answer downgraded to "don't know"). So the reduction is a
            # FAST PATH -- taken when it answers, re-read at native resolution
            # when it does not. Full res costs ~68 ms on a 2048 map against the
            # ~50 ms already spent decoding it, so the escalation is cheap and
            # only the indeterminate minority pays it.
            reduced = img
            if max(img.size) > 512:
                # `resize` rather than `copy() + thumbnail()`: thumbnail is
                # in-place, and the full-size image has to survive for the
                # re-read below, so taking it that way costs a full-size copy
                # first (48 MB on a 4k map). Same `reducing_gap` two-step, same
                # aspect rule, byte-identical output, and measurably faster.
                width, height = img.size
                scale = 512 / max(width, height)
                reduced = img.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    reducing_gap=2.0,
                )

            correlation = cls._normal_handedness_correlation(reduced, min_gradient_std)
            if (correlation is None or abs(correlation) <= threshold) and (
                reduced is not img
            ):
                full = cls._normal_handedness_correlation(img, min_gradient_std)
                if full is not None:
                    correlation = full

            if correlation is None:
                return None
            if correlation < -threshold:
                return "OpenGL"
            if correlation > threshold:
                return "DirectX"
            return None

        except Exception as e:
            cls.logger.warning(f"Error detecting normal map format: {e}")
            return None

    @staticmethod
    def _normal_handedness_correlation(
        img: "Image.Image", min_gradient_std: float
    ) -> Optional[float]:
        """``corr(dR/dy, dG/dx)`` for *img*, or ``None`` if it is meaningless.

        The integrability statistic behind :meth:`detect_normal_map_format`,
        split out so the same computation serves both the reduced fast path and
        the native-resolution re-read. Negative = OpenGL, positive = DirectX;
        the caller owns the threshold.

        ``None`` means "no usable signal here", not "flat": either channel's
        gradient falling under *min_gradient_std* (8-bit units) makes the
        correlation noise, and a non-finite result (a constant channel) is the
        same answer arrived at by division.
        """
        # Only R and G carry the signal, so only R and G are materialized --
        # `np.array(img)` would build the blue plane too, a third of the
        # allocation for nothing (measured on a 2048 map: 218 -> 201 MB peak,
        # and marginally faster). Identical correlation to 0e+00.
        dRy = np.gradient(  # dR/dy along image rows
            np.asarray(img.getchannel("R"), dtype=np.float32), axis=0
        ).ravel()
        dGx = np.gradient(  # dG/dx along image cols
            np.asarray(img.getchannel("G"), dtype=np.float32), axis=1
        ).ravel()
        # Variance floor: flat or near-flat inputs produce meaningless
        # correlations (often NaN, often spuriously signed).
        if dRy.std() < min_gradient_std or dGx.std() < min_gradient_std:
            return None
        correlation = np.corrcoef(dRy, dGx)[0, 1]
        return float(correlation) if np.isfinite(correlation) else None

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

            # Keep the source file's naming style by swapping only the
            # convention tag. This used to pair the two alias tuples by INDEX,
            # which silently depended on them being the same length and in
            # lockstep order.
            new_suffix = target_type_key
            if typ:
                if typ in cls.map_types[source_type_key]:
                    new_suffix = cls._map_registry.counterpart_normal_spelling(
                        typ, target_type_key
                    )

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

        # Output filenames are derived from the specular_map path; a PIL Image
        # has no path, so fail clearly here instead of raising an obscure
        # TypeError from resolve_texture_filename's assert_pathlike below.
        if not isinstance(specular_map, (str, os.PathLike)):
            raise ValueError(
                "convert_spec_gloss_to_pbr(write_files=True) needs a file-path "
                "specular_map to derive output filenames; pass path inputs, or "
                "call with write_files=False to receive Image objects."
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
                Keys are map types and values are the corresponding source
                file paths. The Normal_OpenGL/Normal_DirectX branches require a
                path; the grayscale-inversion branches also accept a PIL Image.
                Example: {"Base_Color": path, "Roughness": path, ...}
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

        Any of the three may name a **packed** map (MSAO, ORM, MRAO, ...) rather
        than a loose one; it is decomposed first and supplies every channel it
        carries, with smoothness inverted to roughness on the way. See
        :meth:`_resolve_orm_sources` for why that is not the caller's job.
        """
        if ao_map_path and isinstance(ao_map_path, str):
            ImgUtils.assert_pathlike(ao_map_path, "ao_map_path")
        if roughness_map_path and isinstance(roughness_map_path, str):
            ImgUtils.assert_pathlike(roughness_map_path, "roughness_map_path")
        if metallic_map_path and isinstance(metallic_map_path, str):
            ImgUtils.assert_pathlike(metallic_map_path, "metallic_map_path")

        # Expanded after the assertions (which reject an Image) but the ORIGINAL
        # arguments still name the output below: expansion replaces a packed
        # path with in-memory channels, and deriving the name from those would
        # turn every packed input into the "cannot derive from Image" error.
        originals = (ao_map_path, roughness_map_path, metallic_map_path)
        ao_map_path, roughness_map_path, metallic_map_path = cls._resolve_orm_sources(
            *originals
        )

        if save and output_path is None:
            source_map = next((src for src in originals if src), None)
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
                metallic_map_path
                or roughness_map_path
                or ao_map_path
                or detail_map_path
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

    #: Packed map type -> (unpacker, the canonical map types it returns, in order).
    #: The dispatch table :meth:`unpack_to_channels` reads.
    #:
    #: Keyed off the registry's canonical names but dispatched to the specific
    #: unpackers rather than derived from :attr:`MapType.channels`, because the
    #: layout a packed map actually ships in is not always the canonical one:
    #: MSAO and MRAO each have two in the wild, auto-detected per image from the
    #: presence of an alpha channel. ``channels`` names the canonical layout
    #: only, so driving the split from it would silently mis-read the other.
    #:
    #: Spec/Gloss is deliberately absent: recovering metallic/roughness from it
    #: is a PBR *conversion* (see ``convert_specgloss_to_pbr``), not a channel
    #: split, so listing it here would promise a decomposition that is wrong.
    PACKED_UNPACKERS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
        "ORM": ("unpack_orm_texture", ("Ambient_Occlusion", "Roughness", "Metallic")),
        "MRAO": (
            "unpack_mrao_texture",
            ("Metallic", "Roughness", "Ambient_Occlusion"),
        ),
        "MSAO": (
            "unpack_msao_texture",
            ("Metallic", "Ambient_Occlusion", "Smoothness"),
        ),
        "Metallic_Smoothness": (
            "unpack_metallic_smoothness",
            ("Metallic", "Smoothness"),
        ),
        "Albedo_Transparency": (
            "unpack_albedo_transparency",
            ("Base_Color", "Opacity"),
        ),
    }

    @classmethod
    def foreign_packings(
        cls,
        sources: Iterable[Any],
        target: str = "ORM",
        workflow: Optional[str] = None,
    ) -> Dict[str, str]:
        """``{path: map type}`` for **packed** sources belonging to another engine.

        The one predicate behind every "is this source set right for what I'm
        writing?" question in the pipeline -- :meth:`pack_orm_texture`'s
        per-map warning, the GLB writer's highlighted summary, and both DCCs'
        Scene Exporter gate all read it, so a mismatch is defined once.

        Two callers, two ways of naming what "right" means, one judgement:

        * A **writer** knows what it is emitting, not which engine this run
          serves -- the GLB channel writer emits an ORM whether the deliverable
          is for three.js, UE or Godot. It names *target* as a map type, and
          the judgement is :meth:`MapRegistry.shares_workflow` -- so no engine
          name is hardcoded, and a packing that declares no workflows
          (``MRAO``) is never accused of anything.
        * An **exporter** knows the opposite: the user chose a texture
          template, i.e. a registry workflow, and the packing follows from it.
          It names *workflow* (which then takes precedence over *target*), and
          a packed source is foreign when it does not declare that workflow.
          An unknown workflow name -- a stale persisted UI value after a
          registry rename -- reports nothing rather than everything, because a
          wrong name must never turn into "every mask map in the scene is
          foreign" and block an export.

        **Only packed maps are eligible**, and that restriction is the whole
        contract, not an optimisation. A *packing* belongs to an engine family
        -- MSAO is how HDRP wants a mask, ORM is how glTF/UE/Godot want one --
        so comparing two packings' declared workflows is meaningful. A LOOSE
        map's ``workflows`` answers a different question: which presets *emit*
        it. ``Ambient_Occlusion`` declares only the Standard preset and
        ``Emissive`` likewise, so a general form reported both as foreign to
        glTF -- flagging an ordinary AO map as an engine mismatch, which would
        have made the exporter gate fire on almost every scene.

        Parameters:
            sources: Texture paths (non-strings and falsy entries are skipped,
                so a mixed list of paths and in-memory images is fine).
            target: The packing being written. Sources are reported when they
                share none of its declared workflows.

        Returns:
            ``{path: map type}``, one entry per distinct offending path, in
            first-seen order. Empty when every source is loose, appropriate, or
            undeclared.
        """
        registry = MapRegistry.instance()
        if workflow is not None and workflow not in registry.get_workflow_presets():
            cls.logger.warning(
                "foreign_packings: unknown workflow %r - reporting nothing. "
                "Known workflows: %s.",
                workflow,
                ", ".join(registry.get_workflow_presets()),
            )
            return {}
        found: Dict[str, str] = {}
        for src in sources:
            if not isinstance(src, str) or not src or src in found:
                continue
            map_type = cls.resolve_map_type(src)
            entry = registry.get(map_type) if map_type else None
            if not entry or not entry.is_packed or not entry.workflows:
                continue
            if workflow is not None:
                if workflow not in entry.workflows:
                    found[src] = map_type
            elif registry.shares_workflow(map_type, target) is False:
                found[src] = map_type
        return found

    @classmethod
    def unpack_to_channels(
        cls,
        source: Union[str, "Image.Image"],
        map_type: Optional[str] = None,
        save: bool = False,
    ) -> Dict[str, "Image.Image"]:
        """The loose maps a packed source map carries, keyed by canonical type.

        The generic front door to the ``unpack_*`` family: given *any* texture,
        return ``{canonical map type: image}`` for what it actually carries, or
        ``{}`` when it is a loose map with nothing to decompose. That lets a
        consumer ask "what channels can I get out of this file?" without first
        knowing which packing scheme it is -- which is what every caller that
        wires a source set into a fixed set of engine slots needs.

        Reports what the map *carries*, not what a caller wants: an MSAO map
        yields ``Smoothness``, never ``Roughness``. Converting between the two
        is the consumer's call (:meth:`convert_smoothness_to_roughness`) and
        folding it in here would make the return type a lie in the one case a
        caller genuinely wants smoothness.

        Parameters:
            source: Texture path, or an already-loaded image (then *map_type*
                is required -- there is no filename to classify).
            map_type: Canonical map type, when it is already known or cannot be
                resolved from the name. Defaults to classifying *source*.
            save: Write the extracted channels to disk instead of returning
                in-memory images.

        Returns:
            ``{canonical map type: image}``; empty when *source* is not a
            packed map this can decompose.
        """
        if map_type is None and isinstance(source, str):
            map_type = cls.resolve_map_type(source)
        entry = cls.PACKED_UNPACKERS.get(map_type or "")
        if not entry:
            return {}
        method, carried = entry
        unpacked = getattr(cls, method)(source, save=save)
        return {
            name: image for name, image in zip(carried, unpacked) if image is not None
        }

    @classmethod
    def _resolve_orm_sources(
        cls,
        ao: Optional[Union[str, "Image.Image"]],
        roughness: Optional[Union[str, "Image.Image"]],
        metallic: Optional[Union[str, "Image.Image"]],
    ) -> Tuple[Any, Any, Any]:
        """Expand any packed map among the three ORM sources into loose channels.

        A packed source map in *any* of the three slots supplies **every**
        channel it carries, not just the slot it happened to arrive in -- that
        is what packing means. Without this, a caller holding only an MSAO map
        can describe it to :meth:`pack_orm_texture` in exactly one way (as one
        of the three slots), and every way is wrong: the packed RGBA is
        flattened to luminance for that one channel and the other two fall back
        to their fill values. Measured on a production room, MSAO named as the
        metallic source produced roughness **0** (mirror-smooth) and metallic
        0.43 against a true 0.016 -- worse than the unrepaired conversion,
        because it overwrote a roughly-correct ORM with a confidently wrong one.

        A loose map the caller passed explicitly always wins over the same
        channel recovered from a packed one: naming both means "use the packed
        map for what the loose maps don't cover".
        """
        # Classified once and threaded through: `resolve_map_type` parses the
        # filename against the whole alias list, and `unpack_to_channels` would
        # otherwise repeat it per source.
        packed = {}
        for src in (ao, roughness, metallic):
            if isinstance(src, str) and src not in packed:
                map_type = cls.resolve_map_type(src)
                if map_type in cls.PACKED_UNPACKERS:
                    packed[src] = map_type
        if not packed:
            return ao, roughness, metallic

        slots = {
            "Ambient_Occlusion": ao,
            "Roughness": roughness,
            "Metallic": metallic,
        }
        # Handled, but say so. A mask map from another engine family unpacks and
        # repacks correctly, and staying silent means the mismatch is never fixed
        # at the source -- while every push pays for a full-resolution channel
        # split and an 8-bit round trip (MSAO carries smoothness, so roughness is
        # reconstructed by inversion rather than read from an authored map), and
        # any channel ORM has no slot for is dropped.
        registry = MapRegistry.instance()
        foreign = cls.foreign_packings(packed, target="ORM")
        for src, map_type in foreign.items():
            cls.logger.warning(
                "pack_orm_texture: %r is a %s map (targets %s), not an "
                "ORM-family packing (targets %s). Unpacked and repacked it, "
                "but re-exporting the source set for an ORM target avoids "
                "the conversion.",
                os.path.basename(src),
                map_type,
                ", ".join(registry.get(map_type).workflows) or "unspecified",
                ", ".join(registry.get("ORM").workflows),
            )

        for src, map_type in packed.items():  # one unpack per distinct map
            carried = cls.unpack_to_channels(src, map_type=map_type)
            if (
                not any(name in slots for name in carried)
                and "Smoothness" not in carried
            ):
                # A packed map that carries no ORM channel at all (an
                # Albedo_Transparency, say) cannot fill any slot, and its path
                # is cleared below -- so if it was the only source, the pack
                # then fails with "no input images provided", which names
                # nothing useful. Say what was actually wrong here.
                cls.logger.warning(
                    "pack_orm_texture: %r is a %s map, which carries none of "
                    "occlusion/roughness/metallic - ignoring it.",
                    os.path.basename(src),
                    map_type,
                )
            if "Roughness" not in carried and "Smoothness" in carried:
                # glTF and every ORM consumer want roughness; the mask-map
                # family stores its inverse. Dropping the inversion is the
                # subtler half of the same bug -- it previews as "everything
                # is shiny where it should be matte", which reads as a bad
                # material rather than as a lost conversion.
                carried["Roughness"] = cls.convert_smoothness_to_roughness(
                    carried.pop("Smoothness"), save=False
                )
            for name, image in carried.items():
                if name not in slots:
                    continue  # a channel ORM has no slot for (Detail, Opacity)
                # Equality, not identity: the same packed map named in
                # several slots arrives as EQUAL strings that are DISTINCT
                # objects whenever the caller's spec crossed a JSON boundary
                # (the export sidecar). Identity recognised only the first
                # slot; the others were then cleared below and took their
                # black fills -- roughness 0, metallic 0, AO intact.
                if slots[name] == src or not slots[name]:
                    slots[name] = image
        # A slot still holding a packed path is one that map carries no channel
        # for; leaving the path would flatten the whole packed image into it.
        for name, value in slots.items():
            if isinstance(value, str) and value in packed:
                slots[name] = None
        return slots["Ambient_Occlusion"], slots["Roughness"], slots["Metallic"]

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
        resolved_layout = (layout or "").lower() or cls._detect_packed_layout(
            msao_map_path
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
        resolved_layout = (layout or "").lower() or cls._detect_packed_layout(
            mrao_map_path
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
        return results.get("RGB"), results.get("A")


# Initialize the registry with the factory class
MapFactory._conversion_registry.add_plugin(MapFactory)
