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
from pythontk.core_utils.engines.textures.map_registry import MapRegistry
from .conversions import MapConversion, ConversionRegistry
from .processor import TextureProcessor
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

# The facade's bodies, split by concept (CODE_STANDARD section 3): each is a
# MapFactory base, so ``cls`` stays the factory and the public methods
# (signatures + docstrings) stay here.
from ._texture_sets import _TextureSetInternal
from ._map_inventory import _MapInventoryInternal
from ._converters import _MapConverterInternal
from ._channel_packing import _ChannelPackingInternal


class MapFactory(
    _TextureSetInternal,
    _MapInventoryInternal,
    _MapConverterInternal,
    _ChannelPackingInternal,
    LoggingMixin,
):
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
        return super().register_conversions(registry)

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
        return super().resolve_map_type(file, key, validate)

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
        return super().resolve_color_space(file, default)

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
        return super().resolve_texture_filename(
            texture_path, map_type, prefix, suffix, ext
        )

    @classmethod
    def get_base_texture_name(
        cls,
        filepath_or_filename: str,
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        """Extracts the base texture name from a filename or path,
        removing known suffixes (e.g., _normal, _roughness).

        The single implementation (``ImgUtils.get_base_texture_name`` is a
        facade over this one): what counts as a map suffix is this engine's
        taxonomy. The two were parallel implementations once, and drifted (the
        UDIM/UV-tile token came off on one side only), so one calls the other.

        Logic: ``MapRegistry.split_map_suffix`` (which composes the tile split
        with ``get_suffix_strip_pattern`` — the SSoT) decides what a suffix is:
        - Delimited suffixes (``MapRegistry.SEPARATORS``): case-insensitive at
          any length (``_ao``, ``-ao``, ``.ao``).
        - Attached suffixes: must start with a capital letter preceded by a
          lowercase one (``brickAO``, ``mat_SpecularGloss``), at EVERY alias
          length — so ordinary words and model numbers aren't misread as maps.

        Parameters:
            filepath_or_filename (str): A texture path or name.
            prefix (str): Optional user-defined prefix to strip from the resolved base
                (case-insensitive). Lets callers safely re-apply it without producing
                e.g. ``Mat_Mat_brick`` when the source filename already had ``Mat_``.
            suffix (str): Optional user-defined suffix to strip from the resolved base.

        Returns:
            str: The base name without map-type suffix, with any configured user prefix/suffix removed.
        """
        return super().get_base_texture_name(filepath_or_filename, prefix, suffix)

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
        return super().get_tile_token(filepath_or_filename)

    @classmethod
    def get_tile_paths(cls, filepath: str) -> List[str]:
        """Every tile on disk of the tile set a texture path names, sorted.

        *filepath* is one tile (``rock.1001.png``) or the set's pattern
        (``rock.<UDIM>.png``). Its tiles are the files beside it whose names
        differ from it only in the tile, spelled in its scheme: four digits for
        a UDIM, ``u#_v#`` for a UV tile. The question a host asks before it
        TILES an image, since a lone tile-numbered file is not a set: tiled,
        ``wall.1024.png`` leaves 0-1 UVs for tile 1024 and renders black, while
        read as one image it renders wherever the UVs sit, by wrapping.

        Parameters:
            filepath (str): A texture path.

        Returns:
            List[str]: The tiles' paths, spelled with *filepath*'s own
            directory; empty when the name carries no tile token or its
            directory does not exist.
        """
        return super().get_tile_paths(filepath)

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
        return super().group_textures_by_set(image_paths, prefix, suffix)

    @classmethod
    def collapse_tile_sets(
        cls, texture_sets: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """One set per MATERIAL from per-tile sets: each map once, as its lowest tile.

        The shader builder's view over :meth:`group_textures_by_set`, which keeps
        a set per tile because the factory converts each tile's images on their
        own. A material is ONE shader however many tiles it spans, so its tiles
        merge here under the tokenless base (``rock.1001`` and ``rock.1002`` ->
        ``rock``), and each map keeps a single path -- a real tile rather than a
        ``<UDIM>`` pattern, because the hosts tile from one: Maya's
        ``uvTilingMode`` and Blender's ``TILED`` image source both find every
        sibling tile from the file they are given, while Blender loads a literal
        ``<UDIM>`` path as a one-tile image (measured, 5.1). The lowest tile is
        kept, so the choice is stable. An untiled set passes through unchanged,
        and an untiled map of a tiled material joins it.

        Parameters:
            texture_sets: ``{set key: paths}`` as :meth:`group_textures_by_set`
                or :meth:`prepare_maps` return them.

        Returns:
            Dict[str, List[str]]: ``{material base: sorted paths}``.
        """
        return super().collapse_tile_sets(texture_sets)

    @classmethod
    def dominant_texture_set(cls, paths: Iterable[str]) -> Optional[Tuple[str, str]]:
        """``(base, folder)`` of the texture set most of *paths* belong to, or ``None``.

        What names a map derived from a material's textures -- a baked
        lightmap is ``<base>_Lightmap`` after the set it lights, not after a
        long import-namespaced node name -- and, with *folder*, where that set
        lives. Only a real MATERIAL MAP votes: a name carrying no map-type
        token (:meth:`resolve_map_type`) is an environment cube, a lookup or a
        stray, and a base taken from a map like that is shared by everything
        that wears it -- measured on a production scene, the object ``TABLE``
        baked as ``diffuse_cube_LightMap.exr`` (Maya's StingrayPBS environment
        texture) beside 46 correctly named objects. The most common base wins,
        a tie going to the name that sorts first, so neither one stray map nor
        the order the paths arrive in decides; *folder* is the one most of
        THAT set's maps sit in, ``""`` for a bare file name (an image embedded
        in the scene keeps only its name).

        One rule for every host: mayatk's and blendertk's lightmap bakers each
        carried their own, and the copies disagreed about both the vote and
        the tie.

        Parameters:
            paths: Texture paths or bare file names, in any order.

        Returns:
            ``(base, folder)``, or ``None`` when no path is a material map.
        """
        return super().dominant_texture_set(paths)

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
        return super().filter_images_by_type(files, types)

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
        return super().sort_images_by_type(files)

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
        return super().contains_map_types(files, map_types)

    @classmethod
    def is_normal_map(cls, file):
        """Check the map type for one of the normal values in map_types.

        Parameters:
            file (str): Image filename, fullpath, or map type suffix.

        Returns:
            (bool)
        """
        return super().is_normal_map(file)

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
        return super().resolve_normal_maps(sorted_maps, target_format, convert)

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
        return super().filter_redundant_maps(sorted_maps, config, extract_missing)

    # --- redundancy internals, shared by both passes -----------------------

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
        return super().extract_channels(packed_type, packed_path, targets, config)

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
                          If False, all input files are treated as a single set --
                          one ASSET, whose UDIM tiles still convert a tile at a
                          time and come back together as one list.
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
            List[str] if a single set was processed -- and always for
            ``group_by_set=False``, whose tiles' maps come back as one list.
            Dict[str, List[str]] if several were, keyed as
            :meth:`group_textures_by_set` keys them (each tile of a UDIM
            material is a set of its own).

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
            # All files are ONE asset, keyed by the first file's base name -- but
            # its UDIM tiles still split: an inventory holds one path per map
            # type, so a merged 2-tile set converted tile 1001 alone and dropped
            # 1002 without a word. The tiles rejoin as one list below. (Fresh
            # lists, so the working set never aliases the caller's input list.)
            base_name = cls.get_base_texture_name(
                files[0], prefix=prefix, suffix=suffix
            )
            texture_sets = {}
            for path in files:
                texture_sets.setdefault(
                    f"{base_name}{cls.get_tile_token(path)}", []
                ).append(path)

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
                finished = {}
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
                    finished[base_name] = generated
            # In the sets' own order, not completion order: the serial branch
            # returns that order, and a caller iterating the batch must not get
            # a different one, run to run, because some worker finished first.
            for base_name in texture_sets:
                if finished.get(base_name):
                    results[base_name] = finished[base_name]
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

        # A named asset is one list, however many tiles it spans.
        if not group_by_set:
            return [path for generated in results.values() for path in generated]
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
        return super().pack_transparency_into_albedo(
            albedo_map_path,
            alpha_map_path,
            output_dir,
            suffix,
            invert_alpha,
            output_path,
            save,
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
        return super().pack_smoothness_into_metallic(
            metallic_map_path,
            alpha_map_path,
            output_dir,
            suffix,
            invert_alpha,
            output_path,
            save,
        )

    @classmethod
    def detect_normal_map_format(
        cls,
        image: Union[str, "Image.Image"],
        threshold: float = 0.25,
        min_gradient_std: float = 1.0,
    ) -> Optional[str]:
        """Whether a normal map is OpenGL (Y+) or DirectX (Y-), read off its content.

        :meth:`ImgUtils.detect_normal_map_format` -- the integrability
        statistic, its measured reliability and its blind spot are documented
        there. It is image analysis, so it lives in ``img_utils``, where code
        below the engines (the UV transfer) reads it too. Here filename
        evidence outranks it: the normal handler asks it only for a map
        classified as the generic ``Normal`` (see NormalMapHandler), never to
        override a ``Normal_OpenGL`` or ``Normal_DirectX`` tag.

        Parameters:
            image (str | PIL.Image.Image): Input normal map.
            threshold (float): Correlation magnitude required to call a format.
            min_gradient_std (float): Per-channel gradient std-dev floor
                (8-bit units) under which the map reads as flat.

        Returns:
            str | None: "OpenGL", "DirectX", or None if indeterminate.
        """
        return ImgUtils.detect_normal_map_format(image, threshold, min_gradient_std)

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
        return super().convert_normal_map_format(
            file, target_format, output_path, save, **kwargs
        )

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
        return super().convert_bump_to_normal(
            bump_map,
            output_path,
            intensity,
            output_format,
            smooth_filter,
            filter_radius,
            edge_wrap,
            save,
            **kwargs,
        )

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
        return super().extract_gloss_from_spec(specular_map, channel)

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
        return super().convert_spec_gloss_to_pbr(
            specular_map,
            glossiness_map,
            diffuse_map,
            output_dir,
            convert_diffuse_to_albedo,
            output_type,
            image_size,
            optimize_bit_depth,
            write_files,
        )

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
        return super().create_base_color_from_spec(
            diffuse, spec, metalness, conserve_energy, metal_darkening
        )

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
        return super().create_metallic_from_spec(
            specular_map, glossiness_map, threshold, softness
        )

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
        return super().create_roughness_from_spec(specular_map, glossiness_map)

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
        return super().convert_base_color_to_albedo(base_color, metalness)

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
        return _MapConverterInternal.get_converted_map(map_type, available)

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
        return super().pack_orm_texture(
            ao_map_path,
            roughness_map_path,
            metallic_map_path,
            output_dir,
            suffix,
            invert_roughness,
            output_path,
            save,
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
        return super().pack_msao_texture(
            metallic_map_path,
            ao_map_path,
            alpha_map_path,
            detail_map_path,
            output_dir,
            suffix,
            invert_alpha,
            output_path,
            save,
            layout,
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
        return super().pack_mrao_texture(
            metallic_map_path,
            roughness_map_path,
            ao_map_path,
            detail_map_path,
            output_dir,
            suffix,
            invert_roughness,
            output_path,
            save,
            layout,
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
        return super().convert_smoothness_to_roughness(
            smoothness_path, output_dir, save, **kwargs
        )

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
        return super().convert_roughness_to_smoothness(
            roughness_path, output_dir, save, **kwargs
        )

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
        return super().foreign_packings(sources, target, workflow)

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
        return super().unpack_to_channels(source, map_type, save)

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
        return super().unpack_orm_texture(
            orm_map_path,
            output_dir,
            ao_suffix,
            roughness_suffix,
            metallic_suffix,
            invert_roughness,
            save,
            **kwargs,
        )

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
        return super().unpack_msao_texture(
            msao_map_path,
            output_dir,
            metallic_suffix,
            ao_suffix,
            smoothness_suffix,
            invert_smoothness,
            save,
            layout,
            **kwargs,
        )

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
        return super().unpack_mrao_texture(
            mrao_map_path,
            output_dir,
            metallic_suffix,
            roughness_suffix,
            ao_suffix,
            invert_roughness,
            save,
            layout,
            **kwargs,
        )

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
        return super().unpack_albedo_transparency(
            albedo_map_path,
            output_dir,
            base_color_suffix,
            opacity_suffix,
            save,
            **kwargs,
        )

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
        return super().unpack_metallic_smoothness(
            map_path,
            output_dir,
            metallic_suffix,
            smoothness_suffix,
            invert_smoothness,
            save,
            **kwargs,
        )

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
        return super().unpack_specular_gloss(
            map_path,
            output_dir,
            specular_suffix,
            gloss_suffix,
            invert_gloss,
            save,
            **kwargs,
        )


# Initialize the registry with the factory class
MapFactory._conversion_registry.add_plugin(MapFactory)
