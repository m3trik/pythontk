# !/usr/bin/python
# coding=utf-8
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import List, Tuple, Dict, Union, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage

try:
    import numpy as np
except ImportError as e:
    print(f"# ImportError: {__file__}\n\t{e}")
try:
    from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageChops, ImageDraw
except ImportError as e:
    print(f"# ImportError: {__file__}\n\t{e}")
    Image = None  # type: ignore# from this package:
from pythontk.core_utils._core_utils import CoreUtils
from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.iter_utils._iter_utils import IterUtils


class ImgUtils(HelpMixin):
    """Helper methods for working with image file formats."""

    map_types: Dict[str, Tuple[str, ...]] = {
        "Base_Color": (
            "Base_Color",
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
            "_BC",
            "_BaseColor",
            "_BaseMap",
        ),
        "Albedo_Transparency": (
            "Albedo_Transparency",
            "AlbedoTransparency",
            "AlbedoAlpha",
            "AlbedoOpacity",
            "BaseColorTransparency",
            "BaseColorAlpha",
            "BaseMapAlpha",
            "_AT",
        ),
        "Roughness": (
            "Roughness",
            "RoughnessMap",
            "Rough",
            "RoughMap",
            "Ruff",
            "Rgh",
            "RGH",
            "_R",
            "_Roughness",
        ),
        # glTF/Unreal-style packed Occlusion (R) + Roughness (G) + Metallic (B)
        "ORM": (
            "ORM",
            "OcclusionRoughnessMetallic",
            "Occlusion_Roughness_Metallic",
            "ORMMap",
            "_ORM",
        ),
        "Metallic": (
            "Metallic",
            "MetallicMap",
            "Metal",
            "MetalMap",
            "Metalness",
            "Met",
            "MTL",
            "_M",
            "_Metallic",
        ),
        "Metallic_Smoothness": (
            "Metallic_Smoothness",
            "MetallicSmoothness",
            "MetalSmooth",
            "Metal_Smooth",
            "Metal_Smoothness",
            "MetallicSmoothnessMap",
            "Metallic_SmoothnessMap",
            "MetallicGloss",
            "MetalGloss",
            "MetallicGlossMap",
            "_MS",
            "_MetalSmooth",
            "_MetallicSmoothness",
            "_MetallicGloss",
        ),
        # Unity-style packed metallic + AO (+ optional detail) + smoothness
        "MSAO": (
            "MSAO",
            "Metallic_SmoothnessAO",
            "MetallicSmoothnessAO",
            "MetallicSmoothAO",
            "MetallicSmoothness_AO",
            "MetallicSmoothnessAmbientOcclusion",
            "MetallicSmoothnessOcclusion",
            "MaskMap",  # Unity HDRP
            "Mask_Map",
            "_MSA",
            "_MaskMap",
        ),
        "Normal": (
            "Normal",
            "NormalMap",
            "Normal_Map",
            "Norm",
            "NRM",
            "_N",
            "_Normal",
            "TangentSpaceNormal",
            "TSN",
            "_TSN",
        ),
        "Normal_DirectX": (
            "Normal_DirectX",
            "NormalDX",
            "Normal_DX",
            "Normal_Tangent_DX",
            "NormalMap_DX",
            "_NDX",
            "_DX",
            "_DXN",
        ),
        "Normal_OpenGL": (
            "Normal_OpenGL",
            "NormalGL",
            "Normal_GL",
            "Normal_Tangent_GL",
            "NormalMap_GL",
            "_NGL",
            "_GL",
        ),
        # Grayscale height/bump
        "Bump": (
            "Bump",
            "BumpMap",
            "Bump_Map",
            "Bumpiness",
            "BumpinessMap",
            "_Bump",
            "_BP",
            "_B",
        ),
        "Height": (
            "Height",
            "HeightMap",
            "Height_Map",
            "High",
            "HGT",
            "Parallax",
            "ParallaxMap",
            "ParallaxOcclusion",
            "POM",
            "_H",
            "_Height",
        ),
        "Emissive": (
            "Emissive",
            "EmissiveMap",
            "Emission",
            "EmissionMap",
            "Emit",
            "Glow",
            "GlowMap",
            "EMI",
            "_E",
            "_EM",
            "_Emissive",
        ),
        # Legacy/spec-gloss diffuse
        "Diffuse": (
            "Diffuse",
            "DiffuseMap",
            "Diff",
            "DIF",
            "DiffuseColor",
            "Diffuse_Color",
            "_DF",
            "_DIF",
            "_Diffuse",
        ),
        "Specular": (
            "Specular",
            "SpecularMap",
            "Spec",
            "SpecMap",
            "SpecColor",
            "SpecularColor",
            "_S",
            "_Spec",
            "_Specular",
        ),
        "Glossiness": (
            "Glossiness",
            "GlossinessMap",
            "Gloss",
            "GlossMap",
            "Glos",
            "Gls",
            "Glo",
            "SpecGloss",
            "SpecularGlossiness",
            "_G",
            "_Gloss",
        ),
        "Displacement": (
            "Displacement",
            "DisplacementMap",
            "Displacement_Map",
            "Disp",
            "Displace",
            "Displ",
            "Dis",
            "Height_Displacement",
            "_DP",
            "_DISP",
        ),
        "Refraction": (
            "Refraction",
            "RefractionMap",
            "Refraction_Index",
            "IndexOfRefraction",
            "Index_of_Refraction",
            "IOR",
            "_IOR",
        ),
        "Reflection": (
            "Reflection",
            "ReflectionMap",
            "Reflect",
            "Refl",
            "ReflMap",
            "_RF",
            "_Refl",
        ),
        "Opacity": (
            "Opacity",
            "OpacityMap",
            "OpacityMask",
            "Transparency",
            "TransparencyMap",
            "Translucency",
            "Alpha",
            "Alpha_Mask",
            "AlphaMask",
            "_O",
            "_OP",
            "_Opacity",
            "_Alpha",
        ),
        "Smoothness": (
            "Smoothness",
            "SmoothnessMap",
            "Smooth",
            "SpecularSmoothness",
            "_SM",
            "_Smooth",
        ),
        "Thickness": (
            "Thickness",
            "ThicknessMap",
            "VolumetricThickness",
            "TransmissionDepth",
            "_TH",
        ),
        "Anisotropy": (
            "Anisotropy",
            "AnisotropyMap",
            "Aniso",
            "Anisotropic",
            "_AN",
        ),
        "Subsurface_Scattering": (
            "Subsurface_Scattering",
            "SubsurfaceScattering",
            "Subsurface",
            "SubsurfaceMap",
            "SubsurfaceColor",
            "SubsurfaceRadius",
            "SSS",
            "_SSS",
        ),
        "Sheen": (
            "Sheen",
            "SheenMap",
            "SheenColor",
            "SheenRoughness",
            "_SH",
        ),
        "Clearcoat": (
            "Clearcoat",
            "ClearCoat",
            "Clear_Coat",
            "ClearcoatMap",
            "ClearCoatMap",
            "ClearCoatMask",
            "Coat",
            "CoatMask",
            "_CC",
            "_Coat",
        ),
        "Ambient_Occlusion": (
            "Ambient_Occlusion",
            "AmbientOcclusion",
            "AO",
            "AOMap",
            "Occlusion",
            "OcclusionMap",
            "MixedAO",
            "Mixed_AO",
            "Occ",
            "Occl",
            "Cavity",
            "CavityMap",
            "_AO",
            "_Occlusion",
        ),
    }

    map_backgrounds = {  # Default map backgrounds in RGBA format by map type.
        "Base_Color": (127, 127, 127, 255),
        "Albedo_Transparency": (0, 0, 0, 255),
        "Roughness": (255, 255, 255, 255),
        "Metallic": (0, 0, 0, 255),
        "Metallic_Smoothness": (255, 255, 255, 255),
        "Metallic_SmoothnessAO": (
            0,
            255,
            0,
            255,
        ),  # R=metallic(black), G=AO(white), B=empty(black), A=smoothness(white)
        "ORM": (
            255,
            255,
            0,
            255,
        ),  # R=AO(white), G=Roughness(white), B=Metallic(black)
        "Normal": (127, 127, 255, 255),
        "Normal_DirectX": (127, 127, 255, 255),
        "Normal_OpenGL": (127, 127, 255, 255),
        "Bump": (127, 127, 127, 255),
        "Height": (127, 127, 127, 255),
        "Emissive": (0, 0, 0, 255),
        "Diffuse": (0, 0, 0, 255),
        "Specular": (0, 0, 0, 255),
        "Glossiness": (0, 0, 0, 255),
        "Displacement": (0, 0, 0, 255),
        "Refraction": (0, 0, 0, 255),
        "Reflection": (0, 0, 0, 255),
        "Opacity": (0, 0, 0, 255),
        "Smoothness": (255, 255, 255, 255),
        "Thickness": (0, 0, 0, 255),
        "Anisotropy": (127, 127, 127, 255),
        "Subsurface_Scattering": (255, 255, 255, 255),
        "Sheen": (127, 127, 127, 255),
        "Clearcoat": (127, 127, 127, 255),
        "Ambient_Occlusion": (255, 255, 255, 255),
    }

    map_modes = {  # Default map mode by map type with comments.
        "Base_Color": "RGB",  # Full color map representing the object's base color.
        "Albedo_Transparency": "RGBA",  # Color map with transparency in the alpha channel.
        "Roughness": "L",  # Grayscale map defining surface roughness.
        "Metallic": "L",  # Grayscale map defining metallic properties.
        "Metallic_Smoothness": "RGBA",  # Multi-channel map for metallic and smoothness.
        "Metallic_SmoothnessAO": "RGBA",  # Multi-channel map for metallic, smoothness, and ambient occlusion.
        "ORM": "RGB",  # Multi-channel map for occlusion, roughness, and metallic.
        "Normal": "RGB",  # Full color normal map.
        "Normal_DirectX": "RGB",  # DirectX normal map with Y-axis inversion.
        "Normal_OpenGL": "RGB",  # OpenGL normal map with standard Y-axis.
        "Bump": "L",  # Grayscale bump map.
        "Height": "I",  # Integer mode for height, often 16 or 32-bit.
        "Emissive": "RGB",  # Full color map for self-illumination.
        "Diffuse": "RGB",  # Full color map for diffuse properties.
        "Specular": "RGB",  # Full color map for specular highlights.
        "Glossiness": "L",  # Grayscale map for surface glossiness.
        "Displacement": "L",  # Grayscale map for displacement mapping.
        "Refraction": "L",  # Grayscale map for light refraction.
        "Reflection": "L",  # Grayscale map for reflection intensity.
        "Opacity": "L",  # Grayscale map for transparency.
        "Smoothness": "L",  # Grayscale map for smoothness level.
        "Thickness": "L",  # Grayscale map for thickness of subsurface scattering.
        "Anisotropy": "L",  # Grayscale map for anisotropic reflections.
        "Subsurface_Scattering": "RGB",  # Multi-channel for SSS color and depth.
        "Sheen": "L",  # Grayscale map for fabric-like reflection layer.
        "Clearcoat": "L",  # Grayscale map for extra clear coating.
        "Ambient_Occlusion": "L",  # Grayscale map for ambient occlusion shading.
    }

    bit_depth = {  # Get bit depth from mode.
        "1": 1,
        "L": 8,
        "P": 8,
        "I;16": 16,
        "I;16B": 16,
        "I;16L": 16,
        "I;16S": 16,
        "I;16BS": 16,
        "I;16LS": 16,
        "RGB": 24,
        "RGBA": 32,
        "CMYK": 32,
        "YCbCr": 24,
        "LAB": 24,
        "HSV": 24,
        "F": 32,
        "I": 32,
        "I;32": 32,
        "I;32B": 32,
        "I;32L": 32,
        "I;32S": 32,
        "I;32BS": 32,
        "I;32LS": 32,
    }

    @staticmethod
    def im_help(a=None):
        """Get help documentation on a specific PIL image attribute
        or list all available attributes.

        Parameters:
            a (str): A specific PIL image attribute (ie. 'resize')
                or if None given; list all available attributes.
        """
        im = Image.new("RGB", (32, 32))

        if a is None:
            for i in dir(im):
                if i.startswith("_"):
                    continue
                print(i)
        else:
            print(help(getattr(im, a)))

        del im

    @classmethod
    @contextmanager
    def allow_large_images(cls):
        """Context manager to safely load very large images.

        Temporarily disables Pillow's MAX_IMAGE_PIXELS guard and suppresses
        DecompressionBombWarning only within the context.
        Restores original settings afterward.
        """
        import warnings

        # Localize warning filters to this context only
        with warnings.catch_warnings():
            if hasattr(Image, "DecompressionBombWarning"):
                warnings.simplefilter("ignore", category=Image.DecompressionBombWarning)

            orig_max_pixels = getattr(Image, "MAX_IMAGE_PIXELS", None)
            if hasattr(Image, "MAX_IMAGE_PIXELS"):
                Image.MAX_IMAGE_PIXELS = None
            try:
                yield
            finally:
                if hasattr(Image, "MAX_IMAGE_PIXELS"):
                    Image.MAX_IMAGE_PIXELS = orig_max_pixels

    @classmethod
    def ensure_image(
        cls,
        input_image: Union[str, Image.Image],
        mode: str = None,
        *,
        max_pixels: Optional[int] = 268_435_456,
    ) -> Image.Image:
        """Ensures the input is a valid PIL Image. Supports optional mode conversion.

        Parameters:
            input_image (str | PIL.Image.Image): Image file path or loaded Image.
            mode (str, optional): Converts the image to the given mode (e.g., "L", "RGB").
            max_pixels (int | None, optional): Combined control for large-image behavior.
                - > 0: Temporarily set Pillow's MAX_IMAGE_PIXELS to this value and suppress
                  DecompressionBombWarning while loading (enables large image handling).
                - 0: Do not override MAX_IMAGE_PIXELS and do not suppress warnings.
                - None: Keep current global behavior unchanged.

        Returns:
            PIL.Image.Image: Valid image object, optionally converted to `mode`.
        """
        if isinstance(input_image, str):
            try:
                # Manage large image safety at call-site granularity
                import warnings

                with warnings.catch_warnings():
                    if (max_pixels is not None and max_pixels > 0) and hasattr(
                        Image, "DecompressionBombWarning"
                    ):
                        warnings.simplefilter(
                            "ignore", category=Image.DecompressionBombWarning
                        )

                    orig_max = getattr(Image, "MAX_IMAGE_PIXELS", None)
                    try:
                        if max_pixels is not None and hasattr(
                            Image, "MAX_IMAGE_PIXELS"
                        ):
                            # 0 means no override (keep current guard), >0 apply the provided cap
                            if max_pixels > 0:
                                Image.MAX_IMAGE_PIXELS = max_pixels
                            # else leave as-is
                        image = Image.open(input_image)
                        image.load()  # Force read the image (PIL is lazy)
                    finally:
                        if hasattr(Image, "MAX_IMAGE_PIXELS"):
                            Image.MAX_IMAGE_PIXELS = orig_max
            except IOError as e:
                raise IOError(
                    f"Unable to load image from path '{input_image}'. Error: {e}"
                )
        elif isinstance(input_image, Image.Image):
            image = input_image
        else:
            raise TypeError(
                "Input must be a file path (str) or a PIL.Image.Image object."
            )

        return image.convert(mode) if mode else image

    @staticmethod
    def assert_pathlike(obj: object, name: str = "argument") -> None:
        """Assert that the given object is a valid path-like object.

        Parameters:
            obj (object): The object to check.
            name (str): The name of the argument for error messages.

        Raises:
            TypeError: If obj is not str, bytes, or os.PathLike.
        """
        if not isinstance(obj, (str, bytes, os.PathLike)):
            raise TypeError(
                f"Expected {name} as str, bytes, or os.PathLike, got {type(obj).__name__}"
            )

    @staticmethod
    def create_image(mode, size=(4096, 4096), color=None):
        """Create a new image.

        Parameters:
            mode (str): Image color mode. ex. 'I', 'L', 'RGBA'
            size (tuple): Size as x and y coordinates.
            color (int)(tuple): Color values.
                    'I' mode image color must be int or single-element tuple.
        Returns:
            (obj) image.
        """
        return Image.new(mode, size, color)

    @classmethod
    def save_image(cls, image: Union[str, Image.Image], name: str, mode: str = None):
        """
        Saves an image to the specified path, with optional mode conversion.

        Parameters:
            image (str | PIL.Image.Image): Image object or file path.
            name (str): Output path including filename and extension (e.g., "output.png").
            mode (str, optional): Converts the image to the specified mode before saving (e.g., "RGB", "L").
        """
        im = cls.ensure_image(image, mode)  # Now allows optional mode conversion
        im.save(name)

    @classmethod
    def load_image(cls, filepath):
        """
        Load an image from the given file path and return a copy of the image object.

        Parameters:
            filepath (str): The full path to the image file.

        Returns:
            (PIL.Image.Image) A copy of the loaded image object.
        """
        cls.assert_pathlike(filepath, "filepath")

        with Image.open(filepath) as im:
            return im.copy()

    @classmethod
    def get_images(
        cls,
        directory,
        inc=["*.png", "*.jpg", "*.bmp", "*.tga", "*.tiff", "*.gif"],
        exc="",
    ):
        """Get bitmap images from a given directory as PIL images.

        Parameters:
            directory (string) = A full path to a directory containing images with the given file_types.
            inc (str): The files to include.
                    supports using the '*' operator: startswith*, *endswith, *contains*
            exc (str): The files to exclude.
                    (exlude take precidence over include)
        Returns:
            (dict) {<full file path>:<image object>}
        """
        cls.assert_pathlike(directory, "directory")

        images = {}
        for f in FileUtils.get_dir_contents(
            directory, "filepath", inc_files=inc, exc_files=exc
        ):
            im = cls.load_image(f)
            images[f] = im

        return images

    @classmethod
    def resolve_map_type(cls, file: str, key: bool = True, validate: str = None) -> str:
        """Resolves the map type from a filename or alias using `map_types`.

        Parameters:
            file (str): Image filename, full path, or map type suffix.
            key (bool): If True, get the corresponding key from 'map_types'.
                        If False, get the abbreviation from 'map_types'.
            validate (str, optional): If provided, validate the map type against this expected type.

        Returns:
            str: The map type.

        Raises:
            ValueError: If the map type is not the expected type when 'validate' is provided.
        """
        cls.assert_pathlike(file, "file")
        filename = FileUtils.format_path(file, "name")

        if key:
            result = next(
                (
                    k
                    for k, v in cls.map_types.items()
                    for i in v
                    if filename.lower().endswith(i.lower())
                ),
                None,
            )
        else:
            result = next(
                (
                    i
                    for v in cls.map_types.values()
                    for i in v
                    if filename.lower().endswith(i.lower())
                ),
                ("".join(filename.split("_")[-1]) if "_" in filename else None),
            )

        if validate:
            # Check both keys and values for validation
            valid_types = [validate] + list(cls.map_types[validate])
            if result not in valid_types:
                raise ValueError(
                    f"Invalid map type '{result}'. Expected type is one of: {valid_types}"
                )

        return result

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
        cls.assert_pathlike(texture_path, "texture_path")

        # Extract sections from the given path
        directory = FileUtils.format_path(texture_path, "path")
        base_name = cls.get_base_texture_name(texture_path)
        original_ext = FileUtils.format_path(texture_path, "ext")

        # Ensure map_type does not start with an underscore
        map_type = map_type.lstrip("_")

        # Ensure suffix formatting (prevents double underscores)
        def clean_suffix(sfx: str) -> str:
            if sfx:
                return sfx if sfx.startswith("_") else f"_{sfx}"
            return ""

        # Determine output file extension (preserve original unless explicitly changed)
        ext = f".{ext.lower()}" if ext else f".{original_ext}"

        # Construct the final filename correctly
        new_name = f"{prefix or ''}{base_name}_{map_type}{clean_suffix(suffix)}{ext}"

        return os.path.join(directory, new_name)

    @classmethod
    def get_base_texture_name(cls, filepath_or_filename: str) -> str:
        """Extracts the base texture name from a filename or path,
        removing known suffixes (e.g., _normal, _roughness) case-insensitively.

        Parameters:
            filepath_or_filename (str): A texture path or name.

        Returns:
            str: The base name without map-type suffix.
        """
        import re

        cls.assert_pathlike(filepath_or_filename, "filepath_or_filename")

        filename = os.path.basename(str(filepath_or_filename))
        base_name, _ = os.path.splitext(filename)

        suffixes_pattern = "|".join(
            re.escape(suffix)
            for suffixes in cls.map_types.values()
            for suffix in suffixes
        )

        pattern = re.compile(
            f"(?:_{suffixes_pattern}|{suffixes_pattern})$", re.IGNORECASE
        )

        base_name = pattern.sub("", base_name)
        return base_name.rstrip("_")

    @classmethod
    def group_textures_by_set(cls, image_paths: List[str]) -> Dict[str, List[str]]:
        """Groups texture maps into sets based on matching base names.

        Parameters:
            image_paths (List[str]): A list of full image file paths.

        Returns:
            Dict[str, List[str]]: A dictionary where:
                - Keys are unique base texture names.
                - Values are lists of associated texture files.
        """
        texture_sets = {}
        for path in image_paths:
            base_name = cls.get_base_texture_name(path)  # Extract base texture name
            print(f"[grouping] {path} → {base_name}")
            if base_name not in texture_sets:
                texture_sets[base_name] = []

            texture_sets[base_name].append(path)

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
    def are_identical(cls, imageA, imageB):
        """Check if two images are the same.

        Parameters:
            imageA (str/obj): An image or path to an image.
            imageB (str/obj): An image or path to an image.

        Returns:
            (bool)
        """
        imA = cls.ensure_image(imageA)
        imB = cls.ensure_image(imageB)

        if np.sum(np.array(ImageChops.difference(imA, imB).getdata())) == 0:
            return True
        return False

    @classmethod
    def resize_image(cls, image, x, y):
        """Returns a resized copy of an image. It doesn't modify the original.

        Parameters:
            image (str/obj): An image or path to an image.
            x (int): Size in the x coordinate.
            y (int): Size in the y coordinate.

        Returns:
            (obj) new image of the given size.
        """
        im = cls.ensure_image(image)
        return im.resize((x, y), Image.Resampling.LANCZOS)

    @classmethod
    def set_bit_depth(cls, image, map_type: str) -> object:
        """Sets the bit depth and image mode of an image according to the map type.

        Parameters:
            image (PIL.Image.Image): The input image.
            map_type (str): The type of the map to determine the mode and bit depth.

        Returns:
            PIL.Image.Image: The image with the specified or recommended bit depth and mode.
        """
        # Determine the target mode based on map type
        if map_type in cls.map_modes:
            target_mode = cls.map_modes[map_type]
            image = image.convert(target_mode)

        # Adjust bit depth
        bit_depth_mapping = {v: k for k, v in cls.bit_depth.items()}
        depth = cls.bit_depth.get(image.mode, 8)

        if depth not in bit_depth_mapping:
            raise ValueError(f"Unsupported bit depth: {depth}")

        if image.mode != bit_depth_mapping[depth]:
            image = image.convert(bit_depth_mapping[depth])

        # Handle unsupported modes specifically
        unsupported_modes = ["HSV", "LAB", "CMYK", "YCbCr"]
        if image.mode in unsupported_modes:
            image = image.convert("RGB" if image.mode != "CMYK" else "RGBA")

        return image

    @classmethod
    def invert_grayscale_image(cls, image: Union[str, Image.Image]) -> Image.Image:
        """Inverts a grayscale image. This method ensures the input is a grayscale image before inverting.

        Parameters:
            image (str/PIL.Image.Image): An image or path to an image to invert.

        Returns:
            PIL.Image.Image: The inverted grayscale image.
        """
        image = cls.ensure_image(image, "L")
        return ImageOps.invert(image)

    @classmethod
    def invert_channels(cls, image, channels="RGBA"):
        """Invert specified channels in an image.

        Parameters:
            image (str/PIL.Image.Image): An image or path to an image.
            channels (str): Specify which channels to invert, e.g., 'R', 'G', 'B', 'A' for red, green, blue, and alpha channels respectively. Case insensitive.

        Returns:
            PIL.Image.Image: The image with specified channels inverted.
        """
        im = cls.ensure_image(image)
        split_channels = im.split()

        # Dictionary to hold the inverted channels
        inverted_channels = {}

        # Loop through each channel in the image
        for i, channel in enumerate("RGBA"[: len(split_channels)]):
            if channel.upper() in channels.upper():
                inverted_channels[channel] = ImageChops.invert(split_channels[i])
            else:
                inverted_channels[channel] = split_channels[i]

        # Handling different image modes
        if len(split_channels) == 1:  # Grayscale image
            return inverted_channels["R"]  # 'R' channel holds the grayscale data
        elif len(split_channels) == 2:  # Grayscale image with alpha
            return Image.merge("LA", (inverted_channels["R"], inverted_channels["A"]))
        else:  # RGB or RGBA image
            return Image.merge(
                "RGBA",
                (
                    inverted_channels["R"],
                    inverted_channels["G"],
                    inverted_channels["B"],
                    inverted_channels.get("A", split_channels[-1]),
                ),
            )

    @classmethod
    @CoreUtils.listify(threading=True)
    def create_mask(
        cls, image, mask, background=(0, 0, 0, 255), foreground=(255, 255, 255, 255)
    ):
        """Create mask(s) from the given image(s).

        Parameters:
            images (str/obj/list): Image(s) or path(s) to an image.
            mask (tuple)(image) = The color to isolate as a mask. (RGB) or (RGBA)
                            or an Image(s) or path(s) to an image. The image's background color will be used.
            background (tuple): Mask background color. (RGB) or (RGBA)
            foreground (tuple): Mask foreground color. (RGB) or (RGBA)

        Returns:
            (obj/list) 'L' mode images. list if 'images' given as a list. else; single image.
        """
        if not isinstance(mask, (tuple, list, set)):
            mask = cls.get_background(mask)

        im = cls.ensure_image(image)
        # mode = im.mode
        im = im.convert("RGBA")
        width, height = im.size
        data = np.array(im)

        r1, g1, b1, a1 = mask if len(mask) == 4 else mask + (None,)

        r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

        bool_list = (
            ((r == r1) & (g == g1) & (b == b1) & (a == a1))
            if len(mask) == 4
            else ((r == r1) & (g == g1) & (b == b1))
        )

        data[:, :, :4][bool_list.any()] = foreground
        data[:, :, :4][bool_list] = background

        # Set the border to background color:
        data[0, 0] = background  # Get the pixel value at top left coordinate.
        data[width - 1, 0] = background  # Top right coordinate.
        data[0, height - 1] = background  # Bottom right coordinate.
        data[width - 1, height - 1] = background  # Bottom left coordinate.

        mask = Image.fromarray(data).convert("L")
        return mask

    @classmethod
    def fill_masked_area(cls, image, color, mask):
        """
        Parameters:
            image (str/obj): An image or path to an image.
            color (list): RGB or RGBA color values.
            mask () =

        Returns:
            (obj) image.
        """
        im = cls.ensure_image(image)
        mode = im.mode
        im = im.convert("RGBA")

        background = cls.create_image(mode=im.mode, size=im.size, color=color)

        return Image.composite(im, background, mask).convert(mode)

    @classmethod
    def fill(cls, image, color=(0, 0, 0, 0)):
        """
        Parameters:
            image (str/obj): An image or path to an image.
            color (list): RGB or RGBA color values.

        Returns:
            (obj) image.
        """
        im = cls.ensure_image(image)

        draw = ImageDraw.Draw(im)
        draw.rectangle([(0, 0), im.size], fill=color)

        return im

    @classmethod
    def get_background(cls, image, mode=None, average=False):
        """Sample the pixel values of each corner of an image and if they are uniform, return the result.

        Parameters:
            image (str/obj): An image or path to an image.
            mode (str): The returned image color mode. ex. 'RGBA'
                    If None is given, the original mode will be returned.
            average (bool): Average the sampled pixel values.

        Returns:
            (int)(tuple) dependant on mode. ex. 32767 for mode 'I' or (211, 211, 211, 255) for 'RGBA'
        """
        im = cls.ensure_image(image)

        if mode and not im.mode == mode:
            im = im.convert(mode)

        width, height = im.size

        tl = im.getpixel((0, 0))  # get the pixel value at top left coordinate.
        tr = im.getpixel((width - 1, 0))  #             ""   top right coordinate.
        br = im.getpixel((0, height - 1))  #            ""   bottom right coordinate.
        bl = im.getpixel((width - 1, height - 1))  #        ""   bottom left coordinate.

        if len(set([tl, tr, br, bl])) == 1:  # list of pixel values are all identical.
            return tl

        elif average:
            return tuple(int(np.mean(i)) for i in zip(*[tl, tr, br, bl]))

        else:
            return None  # non-uniform background.

    @classmethod
    def replace_color(
        cls, image, from_color=(0, 0, 0, 0), to_color=(0, 0, 0, 0), mode=None
    ):
        """
        Parameters:
            image (str/obj): An image or path to an image.
            from_color (tuple): The starting color. (RGB) or (RGBA)
            to_color (tuple): The ending color. (RGB) or (RGBA)
            mode (str): The image is converted to rgba for the operation specify the returned image mode.
                The original image mode will be returned if None is given. ex. 'RGBA' to return in rgba format.
        Returns:
            (obj) image.
        """
        im = cls.ensure_image(image)
        mode = mode if mode else im.mode
        im = im.convert("RGBA")
        data = np.array(im)

        r1, g1, b1, a1 = from_color if len(from_color) == 4 else from_color + (None,)

        r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

        mask = (
            ((r == r1) & (g == g1) & (b == b1) & (a == a1))
            if len(from_color) == 4
            else ((r == r1) & (g == g1) & (b == b1))
        )
        data[:, :, :4][mask] = to_color if len(to_color) == 4 else to_color + (255,)

        return Image.fromarray(data).convert("RGBA")

    @classmethod
    def set_contrast(cls, image, level=255):
        """
        Parameters:
            image (str/obj): An image or path to an image.
            level (int): Contrast level from 0-255.

        Returns:
            (obj) image.
        """
        im = cls.ensure_image(image)

        factor = (259 * (level + 255)) / (255 * (259 - level))

        def adjust_contrast(c):
            # make sure the contrast filter only return values within the range [0-255].
            return int(max(0, min(255, 128 + factor * (c - 128))))

        return im.point(adjust_contrast)  # Pass the contrast filter to im.point.

    @staticmethod
    def convert_rgb_to_gray(data):
        """Convert an RGB Image data array to grayscale.

        :Paramters:
            data (str/obj)(array) = An image, path to an image, or
                    image data as numpy array.
        Returns:
            (array)

        # gray_data = np.average(data, weights=[0.299, 0.587, 0.114], axis=2)
        # gray_data = (data[:,:,:3] * [0.2989, 0.5870, 0.1140]).sum(axis=2)
        """
        if not isinstance(data, np.ndarray):
            im = data.open(data) if (isinstance(data, str)) else data
            data = np.array(im)

        gray_data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])

        # array = gray_data.reshape(gray_data.shape[0], gray_data.shape[1], 1)
        # print (array.shape)

        return gray_data

    @classmethod
    def convert_rgb_to_hsv(cls, image):
        """Manually convert the image to a NumPy array, iterate over the pixels
        and use the colorsys module to convert the colors from RGB to HSV.
        PIL images can be converted usin: image.convert("HSV")
        PNG files cannot be saved as HSV.

        Parameters:
            image (str/obj): An image or path to an image.

        Returns:
            (obj) image.
        """
        import colorsys

        im = cls.ensure_image(image)
        data = np.array(im)

        # Convert the colors from RGB to HSV
        hsv = np.empty_like(data)
        for i in range(data.shape[0]):
            for ii in range(data.shape[1]):
                r, g, b = data[i, ii]
                h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
                hsv[i, ii] = (int(h * 360), int(s * 100), int(v * 100))

        return Image.fromarray(hsv, mode="HSV")

    @classmethod
    def convert_i_to_l(cls, image):
        """Convert to 8 bit 'L' grayscale.

        Parameters:
            image (str/obj): An image or path to an image.

        Returns:
            (obj) image.
        """
        im = cls.ensure_image(image)
        data = np.array(im)

        data = np.asarray(data, np.uint8)  # np.uint8(data / 256)
        return Image.fromarray(data)

    @classmethod
    def pack_channels(
        cls,
        channel_files: dict[str, str | Image.Image],
        channels: list[str] = None,
        out_mode: str = None,
        fill_values: dict[str, int] = None,
        output_path: str = None,
        output_format: str = "PNG",
        grayscale_to_rgb: bool = False,
    ) -> str | Image.Image:
        """Packs up to 4 grayscale images into R, G, B, A channels of a single image.

        Parameters:
            channel_files (dict): {"R": image, "G": image, "B": image, "A": image} (values can be None).
            channels (list): Channel order, default ["R","G","B","A"].
            out_mode (str): "RGB" or "RGBA". If None, uses "RGBA" if "A" present, else "RGB".
            fill_values (dict): Per-channel fallback, default: 0 for RGB, 255 for A.
            output_path (str): If given, saves image and returns path.
            output_format (str): Save format, e.g., "png", "tga".
            grayscale_to_rgb (bool): If True and only one RGB channel is assigned,
                                    its image will be duplicated across R, G, B.

        Returns:
            str | Image.Image: Output path if saving, else the PIL image object.
        """
        if channels is None:
            channels = ["R", "G", "B", "A"]
        if fill_values is None:
            fill_values = {ch: 0 for ch in "RGB"}
            fill_values["A"] = 255

        has_alpha = bool(channel_files.get("A"))
        out_mode = out_mode or ("RGBA" if has_alpha else "RGB")
        n_channels = 4 if out_mode == "RGBA" else 3

        # Get first valid image for sizing
        first_file = next(
            (f for f in (channel_files.get(ch) for ch in channels) if f), None
        )
        if first_file is None:
            raise ValueError("No input images provided")
        size = cls.ensure_image(first_file).size

        # Determine if we should replicate grayscale to RGB (duplicate if only one RGB channel is used)
        used_rgb_channels = [ch for ch in "RGB" if channel_files.get(ch)]
        allow_duplicate = grayscale_to_rgb and len(used_rgb_channels) == 1
        r_img = (
            cls.ensure_image(channel_files.get("R"), mode="L").resize(size)
            if channel_files.get("R")
            else None
        )

        bands = []
        for ch in channels[:n_channels]:
            img = channel_files.get(ch)
            if img:
                band = cls.ensure_image(img, mode="L").resize(size)
            elif ch == "G" and not channel_files.get("G"):
                # Ensure the green channel stays empty
                band = cls.create_image("L", size, color=fill_values.get("G", 0))
            elif ch == "B" and not channel_files.get("B"):
                # Ensure the blue channel stays empty
                band = cls.create_image("L", size, color=fill_values.get("B", 0))
            elif ch in "GB" and allow_duplicate and r_img is not None:
                # Duplicate R into G/B if only R is used
                band = r_img
            else:
                band = cls.create_image("L", size, color=fill_values.get(ch, 0))
            bands.append(band)

        img = Image.merge(out_mode, bands)

        if output_path:
            img.save(output_path, format=output_format)
            return output_path
        return img

    @classmethod
    def pack_channel_into_alpha(
        cls,
        image: Union[str, Image.Image],
        alpha: Union[str, Image.Image],
        output_path: Optional[str] = None,
        invert_alpha: bool = False,
        resize_alpha: bool = True,
        preserve_existing_alpha: bool = False,
    ) -> str:
        """Packs a channel from the alpha source image into the alpha channel of the base image.

        Parameters:
            image (str | Image.Image): Base texture (albedo).
            alpha (str | Image.Image): Transparency map to pack into the alpha channel.
            output_path (str, optional): Output path. If None, overwrites the input image path.
            invert_alpha (bool): Invert the alpha source before packing.
            resize_alpha (bool): Resize the alpha to match the base if needed.
            preserve_existing_alpha (bool): If True, multiply existing alpha with the new alpha.

        Returns:
            str: Path to the saved image.
        """
        base_img = cls.ensure_image(image, mode="RGB")
        alpha_img = cls.ensure_image(alpha)

        if invert_alpha:
            alpha_img = cls.invert_grayscale_image(alpha_img)

        alpha_img = alpha_img.convert("L")

        if resize_alpha and base_img.size != alpha_img.size:
            print(
                f"// Resizing alpha from {alpha_img.size} to match base {base_img.size}"
            )
            alpha_img = alpha_img.resize(base_img.size, Image.Resampling.LANCZOS)
        elif base_img.size != alpha_img.size:
            raise ValueError(
                f"Alpha image size {alpha_img.size} does not match base {base_img.size} and resize is disabled."
            )

        base_img = base_img.convert("RGBA")
        r, g, b, existing_alpha = base_img.split()

        if preserve_existing_alpha:
            alpha_combined = ImageChops.multiply(existing_alpha, alpha_img)
        else:
            alpha_combined = alpha_img

        combined_img = Image.merge("RGBA", (r, g, b, alpha_combined))

        if not output_path:
            if isinstance(image, str):
                output_path = image
            else:
                raise ValueError(
                    "Output path must be provided when using Image objects directly."
                )

        combined_img.save(output_path)
        return output_path

    @classmethod
    def pack_transparency_into_albedo(
        cls,
        albedo_map_path: str,
        alpha_map_path: str,
        output_dir: Optional[str] = None,
        suffix: Optional[str] = "_AlbedoTransparency",
        invert_alpha: bool = False,
    ) -> str:
        """Combines an albedo texture with a transparency map by packing the transparency into the alpha channel.

        Parameters:
            albedo_map_path (str): Path to the albedo (base color) texture map.
            alpha_map_path (str): Path to the transparency (alpha) texture map.
            output_dir (str, optional): Output directory. If None, uses the albedo map directory.
            suffix (str, optional): Suffix for the output file name. Defaults to '_AlbedoTransparency'.
            invert_alpha (bool, optional): If True, inverts the alpha texture.

        Returns:
            str: The output file path.
        """
        cls.assert_pathlike(albedo_map_path, "albedo_map_path")
        cls.assert_pathlike(alpha_map_path, "alpha_map_path")

        base_name = cls.get_base_texture_name(albedo_map_path)

        if output_dir is None:
            output_dir = os.path.dirname(albedo_map_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        output_path = os.path.join(output_dir, f"{base_name}{suffix}.png")

        success = cls.pack_channel_into_alpha(
            albedo_map_path,
            alpha_map_path,
            output_path,
            invert_alpha=invert_alpha,
        )

        if success:
            return output_path
        else:
            raise Exception("Failed to pack transparency into albedo map.")

    @classmethod
    def pack_smoothness_into_metallic(
        cls,
        metallic_map_path: str,
        alpha_map_path: str,
        output_dir: str = None,
        suffix: str = "_MetallicSmoothness",
        invert_alpha: bool = False,
    ) -> str:
        """Packs a smoothness (or inverted roughness) texture into the alpha channel of a metallic texture map.

        Parameters:
            metallic_map_path (str): Path to the metallic texture map.
            alpha_map_path (str): Path to the smoothness or roughness texture map.
            output_dir (str, optional): Directory path for the output. If None, the output directory will be the same as the metallic map path.
            invert_alpha (bool, optional): If True, the alpha (smoothness/roughness) texture will be inverted.
            suffix (str, optional): Suffix for the output file name, defaulting to '_MetallicSmoothness'.

        Returns:
            str: The file path of the newly created metallic-smoothness texture map.
        """
        cls.assert_pathlike(metallic_map_path, "metallic_map_path")
        cls.assert_pathlike(alpha_map_path, "alpha_map_path")

        base_name = cls.get_base_texture_name(metallic_map_path)
        if output_dir is None:
            output_dir = os.path.dirname(metallic_map_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        output_path = os.path.join(output_dir, f"{base_name}{suffix}.png")

        success = cls.pack_channel_into_alpha(
            metallic_map_path, alpha_map_path, output_path, invert_alpha=invert_alpha
        )

        if success:
            return output_path
        else:
            raise Exception("Failed to pack smoothness into metallic map.")

    @classmethod
    def unpack_metallic_smoothness(
        cls,
        metallic_smoothness_map_path: str,
        output_dir: str = None,
        metallic_suffix: str = "_Metallic",
        smoothness_suffix: str = "_Smoothness",
        invert_smoothness: bool = False,
    ) -> Tuple[str, str]:
        """Unpacks metallic and smoothness maps from a combined metallic-smoothness texture.

        The metallic channel is extracted from RGB channels (typically Red channel),
        and the smoothness channel is extracted from the Alpha channel.

        Parameters:
            metallic_smoothness_map_path (str): Path to the metallic-smoothness texture map.
            output_dir (str, optional): Directory path for the output. If None, the output directory will be the same as the input map path.
            metallic_suffix (str, optional): Suffix for the metallic output file name, defaulting to '_Metallic'.
            smoothness_suffix (str, optional): Suffix for the smoothness output file name, defaulting to '_Smoothness'.
            invert_smoothness (bool, optional): If True, the smoothness channel will be inverted to create a roughness map.

        Returns:
            Tuple[str, str]: A tuple containing the file paths of the extracted metallic and smoothness texture maps.

        Raises:
            ValueError: If the input path is invalid.
            FileNotFoundError: If the input file does not exist.
        """
        cls.assert_pathlike(
            metallic_smoothness_map_path, "metallic_smoothness_map_path"
        )

        if not os.path.exists(metallic_smoothness_map_path):
            raise FileNotFoundError(
                f"Input file not found: {metallic_smoothness_map_path}"
            )

        # Load the combined texture
        combined_image = cls.ensure_image(metallic_smoothness_map_path)

        # Get base name for output files
        base_name = cls.get_base_texture_name(metallic_smoothness_map_path)

        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(metallic_smoothness_map_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        # Extract metallic channel (typically from RGB channels)
        # For metallic maps, we usually use the red channel or convert to grayscale
        if combined_image.mode in ("RGBA", "RGB"):
            metallic_channel = combined_image.getchannel("R").convert("L")
        else:
            metallic_channel = combined_image.convert("L")

        # Extract smoothness channel from alpha
        if combined_image.mode in ("RGBA", "LA"):
            smoothness_channel = combined_image.getchannel("A").convert("L")
        else:
            # If no alpha channel, use a default smoothness (white = smooth)
            smoothness_channel = Image.new("L", combined_image.size, 255)
            print(
                f"// Warning: No alpha channel found in {metallic_smoothness_map_path}, using default smoothness."
            )

        # Invert smoothness if requested (to create roughness)
        if invert_smoothness:
            smoothness_channel = ImageOps.invert(smoothness_channel)
            # Update suffix if inverted
            if smoothness_suffix == "_Smoothness":
                smoothness_suffix = "_Roughness"

        # Create output file paths
        metallic_output_path = os.path.join(
            output_dir, f"{base_name}{metallic_suffix}.png"
        )
        smoothness_output_path = os.path.join(
            output_dir, f"{base_name}{smoothness_suffix}.png"
        )

        # Save the extracted maps
        metallic_channel.save(metallic_output_path, format="PNG")
        smoothness_channel.save(smoothness_output_path, format="PNG")

        return metallic_output_path, smoothness_output_path

    @classmethod
    def create_dx_from_gl(cls, file: str, output_path: str = None) -> str:
        cls.assert_pathlike(file, "file")

        try:
            typ = cls.resolve_map_type(file, key=False, validate="Normal_OpenGL")
        except ValueError:
            typ = cls.resolve_map_type(file, key=False, validate="Normal")

        inverted_image = cls.invert_channels(file, "g")

        if output_path is None:
            output_dir = FileUtils.format_path(file, "path")
            name = FileUtils.format_path(file, "name")
            ext = FileUtils.format_path(file, "ext")

            try:
                index = cls.map_types["Normal_OpenGL"].index(typ)
                new_type = cls.map_types["Normal_DirectX"][index]
            except (ValueError, IndexError):
                new_type = cls.map_types["Normal_DirectX"][0]

            name = name.removesuffix(typ)
            output_path = f"{output_dir}/{name}{new_type}.{ext}"

        output_path = os.path.abspath(output_path)
        inverted_image.save(output_path)
        return output_path

    @classmethod
    def create_gl_from_dx(cls, file: str, output_path: str = None) -> str:
        cls.assert_pathlike(file, "file")

        try:
            typ = cls.resolve_map_type(file, key=False, validate="Normal_DirectX")
        except ValueError:
            typ = cls.resolve_map_type(file, key=False, validate="Normal")

        inverted_image = cls.invert_channels(file, "g")

        if output_path is None:
            output_dir = FileUtils.format_path(file, "path")
            name = FileUtils.format_path(file, "name")
            ext = FileUtils.format_path(file, "ext")

            try:
                index = cls.map_types["Normal_DirectX"].index(typ)
                new_type = cls.map_types["Normal_OpenGL"][index]
            except (ValueError, IndexError):
                new_type = cls.map_types["Normal_OpenGL"][0]

            name = name.removesuffix(typ)
            output_path = f"{output_dir}/{name}{new_type}.{ext}"

        output_path = os.path.abspath(output_path)
        inverted_image.save(output_path)
        return output_path

    @classmethod
    def convert_bump_to_normal(
        cls,
        bump_map: Union[str, Image.Image],
        output_path: str = None,
        intensity: float = 1.0,
        output_format: str = "opengl",
        smooth_filter: bool = True,
        filter_radius: float = 0.5,
        edge_wrap: bool = False,
    ) -> str:
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

        Returns:
            str: Path to the generated normal map file.

        Notes:
            - Uses Sobel operator for gradient calculation (industry standard)
            - OpenGL: Y+ points up (green channel positive = surface pointing up)
            - DirectX: Y+ points down (green channel inverted from OpenGL)
            - Intensity should be scaled based on real-world height units
            - Pre-filtering reduces mipmap artifacts in final rendering
        """
        # Load and ensure grayscale; validate path only when a path is provided
        if isinstance(bump_map, str):
            cls.assert_pathlike(bump_map, "bump_map")
        image = cls.ensure_image(bump_map, "L")

        # Apply smoothing filter to reduce aliasing if requested
        if smooth_filter and filter_radius > 0:
            # Use Gaussian blur to smooth height data before gradient calculation
            image = image.filter(ImageFilter.GaussianBlur(radius=filter_radius))

        # Convert to numpy array for gradient calculations
        height_srgb = np.asarray(image, dtype=np.float32) / 255.0

        # Convert sRGB grayscale to linear before computing derivatives (safer filtering/derivatives)
        height_lin = cls._srgb_to_linear_np(height_srgb)

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

        # Calculate normal vectors
        # The cross product of tangent (1,0,grad_x) and bitangent (0,1,grad_y)
        # gives us the surface normal (-grad_x, -grad_y, 1)
        normal_x = -grad_x
        normal_y = -grad_y
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

        # Generate output path if not provided
        if output_path is None:
            if isinstance(bump_map, str):
                base_path = bump_map
            else:
                # If PIL Image was passed, create generic output name
                base_path = "bump_map.png"

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
        normal_image.save(output_path)

        return output_path

    @staticmethod
    def _srgb_to_linear_np(arr):
        """Convert sRGB values to linear.

        Accepts a NumPy array or array-like. Values can be either 0-255 or 0-1.
        Returns float32 in [0,1]. Alpha channel (if present) is preserved.
        """
        a = np.asarray(arr)
        # Convert to float32 for calculation
        if a.dtype != np.float32 and a.dtype != np.float64:
            a = a.astype(np.float32)

        alpha = None
        if a.ndim == 3 and a.shape[-1] == 4:
            alpha = a[..., 3:4]
            a = a[..., :3]

        # Normalize to [0,1] if needed
        if a.max() > 1.0:
            a = a / 255.0

        a = np.clip(a, 0.0, 1.0)
        k0 = 0.04045
        out = np.empty_like(a, dtype=np.float32)
        low = a <= k0
        out[low] = a[low] / 12.92
        out[~low] = ((a[~low] + 0.055) / 1.055) ** 2.4

        if alpha is not None:
            out = np.concatenate([out, alpha], axis=-1)
        return out

    @staticmethod
    def _linear_to_srgb_np(arr):
        """Convert linear values to sRGB.

        Accepts a NumPy array in [0,1] and returns float32 in [0,1].
        """
        a = np.asarray(arr)
        if a.dtype != np.float32 and a.dtype != np.float64:
            a = a.astype(np.float32)

        alpha = None
        if a.ndim == 3 and a.shape[-1] == 4:
            alpha = a[..., 3:4]
            a = a[..., :3]

        a = np.clip(a, 0.0, 1.0)
        k1 = 0.0031308
        out = np.empty_like(a, dtype=np.float32)
        low = a <= k1
        out[low] = a[low] * 12.92
        out[~low] = 1.055 * (a[~low] ** (1.0 / 2.4)) - 0.055

        if alpha is not None:
            out = np.concatenate([out, alpha], axis=-1)
        return out

    @classmethod
    def _srgb_to_linear_image(cls, img: Image.Image) -> Image.Image:
        """Convert a PIL image (L/RGB/RGBA) from sRGB to linear, returned as 8-bit per channel.

        Alpha channel (if present) is preserved untouched.
        """
        arr = np.array(img, dtype=np.float32)
        if img.mode in ("L", "RGB", "RGBA"):
            lin = cls._srgb_to_linear_np(arr)
            lin_8 = np.clip(lin * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(lin_8, mode=img.mode)
        # For other modes, fall back to converting to RGB
        return cls._srgb_to_linear_image(img.convert("RGBA"))

    @classmethod
    def _linear_to_srgb_image(cls, img: Image.Image) -> Image.Image:
        """Convert a PIL image (L/RGB/RGBA) from linear to sRGB, returned as 8-bit per channel.

        Alpha channel (if present) is preserved untouched.
        """
        arr = np.array(img, dtype=np.float32)
        if img.mode in ("L", "RGB", "RGBA"):
            srgb = cls._linear_to_srgb_np(arr / 255.0)
            srgb_8 = np.clip(srgb * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(srgb_8, mode=img.mode)
        return cls._linear_to_srgb_image(img.convert("RGBA"))

    @classmethod
    def srgb_to_linear(cls, data):
        """Friendly wrapper: accepts PIL Image, numpy array, or list/tuple.

        - If Image: returns Image in the same mode (8-bit), converted to linear.
        - Otherwise: converts input to numpy, applies sRGB->linear, returns numpy array float32 in [0,1].
        """
        if isinstance(data, Image.Image):
            return cls._srgb_to_linear_image(data)
        # Accept lists/tuples/arrays
        return cls._srgb_to_linear_np(data)

    @classmethod
    def linear_to_srgb(cls, data):
        """Friendly wrapper: accepts PIL Image, numpy array, or list/tuple.

        - If Image: returns Image in the same mode (8-bit), converted to sRGB.
        - Otherwise: expects data in [0,1], returns numpy array float32 in [0,1].
        """
        if isinstance(data, Image.Image):
            return cls._linear_to_srgb_image(data)
        return cls._linear_to_srgb_np(data)

    @classmethod
    def extract_gloss_from_spec(
        cls, specular_map: str, channel: str = "A"
    ) -> Union[Image.Image, None]:
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
        spec = cls.ensure_image(specular_map)

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
    def unpack_specular_gloss(
        cls,
        specular_gloss_map_path: str,
        output_dir: str = None,
        specular_suffix: str = "_Specular",
        gloss_suffix: str = "_Gloss",
        gloss_channel: str = "A",
    ) -> Tuple[str, str]:
        """Unpacks specular and gloss maps from a combined specular-gloss texture.

        The specular data is extracted from RGB channels, and the gloss data is
        extracted from the specified channel (typically Alpha channel).

        Parameters:
            specular_gloss_map_path (str): Path to the specular-gloss texture map.
            output_dir (str, optional): Directory path for the output. If None, the output directory will be the same as the input map path.
            specular_suffix (str, optional): Suffix for the specular output file name, defaulting to '_Specular'.
            gloss_suffix (str, optional): Suffix for the gloss output file name, defaulting to '_Gloss'.
            gloss_channel (str, optional): Channel to extract gloss from ("R", "G", "B", "A"), defaulting to "A".

        Returns:
            Tuple[str, str]: A tuple containing the file paths of the extracted specular and gloss texture maps.

        Raises:
            ValueError: If the input path is invalid.
            FileNotFoundError: If the input file does not exist.
        """
        cls.assert_pathlike(specular_gloss_map_path, "specular_gloss_map_path")

        if not os.path.exists(specular_gloss_map_path):
            raise FileNotFoundError(f"Input file not found: {specular_gloss_map_path}")

        # Load the combined texture
        combined_image = cls.ensure_image(specular_gloss_map_path)

        # Get base name for output files
        base_name = cls.get_base_texture_name(specular_gloss_map_path)

        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(specular_gloss_map_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        # Extract specular channel (RGB channels)
        if combined_image.mode in ("RGBA", "RGB"):
            # Convert to RGB to remove alpha for specular map
            specular_channel = combined_image.convert("RGB")
        else:
            specular_channel = combined_image.convert("RGB")

        # Extract gloss channel using existing method logic
        gloss_channel_upper = gloss_channel.upper()
        if gloss_channel_upper in combined_image.getbands():
            gloss_map = combined_image.getchannel(gloss_channel_upper)
            if gloss_map.getextrema() != (0, 0):  # Ensure non-empty
                gloss_map = gloss_map.convert("L")
            else:
                # Use normalized grayscale fallback
                print(
                    f"// Warning: No gloss found in '{gloss_channel}' channel; using normalized grayscale..."
                )
                spec_gray = combined_image.convert("L")
                spec_gray = ImageEnhance.Brightness(spec_gray).enhance(1.2)
                gloss_map = ImageOps.autocontrast(spec_gray)
        else:
            # Use normalized grayscale fallback
            print(
                f"// Warning: '{gloss_channel}' channel not found; using normalized grayscale..."
            )
            spec_gray = combined_image.convert("L")
            spec_gray = ImageEnhance.Brightness(spec_gray).enhance(1.2)
            gloss_map = ImageOps.autocontrast(spec_gray)

        # Create output file paths
        specular_output_path = os.path.join(
            output_dir, f"{base_name}{specular_suffix}.png"
        )
        gloss_output_path = os.path.join(output_dir, f"{base_name}{gloss_suffix}.png")

        # Save the extracted maps
        specular_channel.save(specular_output_path, format="PNG")
        gloss_map.save(gloss_output_path, format="PNG")

        return specular_output_path, gloss_output_path

    @classmethod
    def convert_spec_gloss_to_pbr(
        cls,
        specular_map: Union[str, Image.Image],
        glossiness_map: Union[str, Image.Image],
        diffuse_map: Union[str, Image.Image] = None,
        output_dir: str = None,
        convert_diffuse_to_albedo: bool = False,
        output_type: str = None,
        image_size: Optional[int] = None,
        optimize_bit_depth: bool = True,
        write_files: bool = False,
    ) -> Union[Tuple[Image.Image, Image.Image, Image.Image], Tuple[str, str, str]]:
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
        spec = cls.ensure_image(specular_map, "RGB")
        gloss = cls.ensure_image(glossiness_map, "L")
        diffuse = cls.ensure_image(diffuse_map, "RGB") if diffuse_map else None

        metallic = cls.create_metallic_from_spec(specular_map)
        base_color = cls.create_base_color_from_spec(diffuse, spec, metallic)
        roughness = cls.create_roughness_from_spec(spec, gloss)

        if convert_diffuse_to_albedo:
            base_color = cls.convert_base_color_to_albedo(base_color, metallic)

        if optimize_bit_depth:
            base_color = cls.set_bit_depth(base_color, "Base_Color")
            metallic = cls.set_bit_depth(metallic, "Metallic")
            roughness = cls.set_bit_depth(roughness, "Roughness")

        # Optional downscale to target max dimension while preserving original if not requested
        if isinstance(image_size, int) and image_size > 0:
            if max(base_color.size) > image_size:
                base_color = cls.resize_image(base_color, image_size, image_size)
            if max(metallic.size) > image_size:
                metallic = cls.resize_image(metallic, image_size, image_size)
            if max(roughness.size) > image_size:
                roughness = cls.resize_image(roughness, image_size, image_size)

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

        base_color.save(base_color_file)
        metallic.save(metallic_file)
        roughness.save(roughness_file)

        print(
            f"PBR Conversion complete. Files saved:\n- {base_color_file}\n- {metallic_file}\n- {roughness_file}"
        )
        return base_color_file, metallic_file, roughness_file

    @classmethod
    def create_base_color_from_spec(
        cls,
        diffuse: Union[str, Image.Image],
        spec: Union[str, Image.Image],
        metalness: Union[str, Image.Image],
        conserve_energy: bool = True,
        metal_darkening: float = 0.22,
    ) -> Image.Image:
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
        spec = np.array(cls.ensure_image(spec, "RGB"), dtype=np.float32) / 255.0
        metalness = np.array(cls.ensure_image(metalness, "L"), dtype=np.float32) / 255.0

        if diffuse:
            diffuse = (
                np.array(cls.ensure_image(diffuse, "RGB"), dtype=np.float32) / 255.0
            )
            base_color = (
                diffuse * (1 - metalness[..., None]) + spec * metalness[..., None]
            )
        else:
            base_color = spec * (1 - metalness[..., None])

        # Darken metal areas (Reduce brightness in metals)
        base_color = np.where(
            metalness[..., None] > 0.5,
            base_color * (1.0 - metal_darkening),  # Apply metal darkening factor
            base_color,
        )
        # Apply energy conservation fix
        if conserve_energy:
            base_color = np.clip(
                base_color / (1.0 - 0.08 * metalness[..., None] + 1e-6), 0.0, 1.0
            )

        return Image.fromarray((base_color * 255).astype(np.uint8), mode="RGB")

    @classmethod
    def create_metallic_from_spec(
        cls,
        specular_map: Union[str, Image.Image],
        glossiness_map: Union[str, Image.Image] = None,
        threshold: int = 55,
        softness: float = 0.2,
    ) -> Image.Image:
        """Creates a metallic map from a specular (and optional glossiness) map.

        Steps:
        1. Use gloss map if provided, or extract from spec.
        2. Compute metallic from spec using soft threshold.
        3. Refine metallic using gloss (if available).

        Returns:
            Image.Image: Metallic map (L mode).
        """
        spec_rgb = cls.ensure_image(specular_map, "RGB")
        spec_lum = np.array(spec_rgb.convert("L"), dtype=np.float32) / 255.0

        # Step 1: Get gloss
        if glossiness_map:
            gloss = (
                np.array(cls.ensure_image(glossiness_map, "L"), dtype=np.float32)
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
        specular_map: Union[str, Image.Image],
        glossiness_map: Union[str, Image.Image] = None,
    ) -> Image.Image:
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
        spec = cls.ensure_image(specular_map, "RGB")

        # Step 1: Use provided gloss map or extract from specular
        gloss = (
            cls.ensure_image(glossiness_map, "L")
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
        cls, base_color: Image.Image, metalness: Image.Image
    ) -> Image.Image:
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
        base_color = cls.ensure_image(base_color, "RGB")
        metalness = cls.ensure_image(metalness, "L")

        # Convert metalness to grayscale and threshold (Metal = 1, Non-Metal = 0)
        metal_mask = metalness.point(lambda p: 0 if p > 128 else 255)

        # Create a black image for metals
        black_image = Image.new("RGB", base_color.size, (0, 0, 0))

        # Composite to replace metallic areas with black
        albedo = Image.composite(black_image, base_color, metal_mask)

        # Normalize colors to prevent artifacts
        albedo = ImageOps.autocontrast(albedo)

        return albedo

    @classmethod
    def generate_mipmaps(cls, image: Image.Image) -> Image.Image:
        """Generates mipmaps for an image.

        Parameters:
            image (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The image with mipmaps applied.
        """
        base = image.copy()
        mipmaps = [base]

        while min(base.size) > 1:
            base = base.resize(
                (base.size[0] // 2, base.size[1] // 2), Image.Resampling.LANCZOS
            )
            mipmaps.append(base)

        return mipmaps[0]  # Return the highest-resolution mipmap

    @classmethod
    def batch_optimize_textures(cls, directory: str, **kwargs):
        """Batch optimizes all textures in a directory.

        Parameters:
            directory (str): Directory containing the textures to optimize.
            output_dir (str, optional): Directory path for the optimized textures. If None, the textures will be saved next to the originals.
            max_size (int, optional): Maximum size for the longest dimension of the textures. Defaults to 4096
        """
        cls.assert_pathlike(directory, "directory")

        textures = cls.get_images(directory)
        print(f"Optimizing textures in: {directory}")
        for texture_path in textures.keys():
            cls.optimize_texture(texture_path, **kwargs)
        print(f"{len(textures)} textures optimized.")

    @classmethod
    def optimize_texture(
        cls,
        texture_path: str,
        output_dir: str = None,
        output_type: str = None,
        max_size: int = None,
        suffix_old: str = None,
        suffix_opt: str = None,
        old_files_folder: str = None,
        generate_mipmaps: bool = False,
        optimize_bit_depth: bool = True,
    ) -> str:
        """Optimizes a texture by resizing, setting bit depth, and adjusting image type.

        Parameters:
            texture_path (str): Path to the texture file.
            output_dir (str, optional): Directory for the optimized texture. Defaults to same directory.
            output_type (str, optional): Output image format (e.g., PNG, TGA). If None, keeps original.
            max_size (int, optional): Maximum size for the longest dimension. Only applies if the image is larger.
            suffix_old (str, optional): Suffix to rename the original file before optimization.
            suffix_opt (str, optional): Suffix to append to the optimized file (None = overwrite).
            old_files_folder (str, optional): Name of the folder to store old files.
            generate_mipmaps (bool): Generates mipmaps if enabled.
            optimize_bit_depth (bool): Adjusts bit depth to match the map type.

        Returns:
            str: Path to the optimized texture.
        """
        cls.assert_pathlike(texture_path, "texture_path")

        if output_dir is None:
            output_dir = os.path.dirname(texture_path)
        os.makedirs(output_dir, exist_ok=True)

        # Determine correct map suffix format
        map_type_suffix = cls.resolve_map_type(texture_path, key=False)

        # Load the image first (before renaming)
        image = cls.ensure_image(texture_path)

        # Get current dimensions
        width, height = image.size

        # Resize if the image is larger than max_size
        if max_size and max(width, height) > max_size:
            print(
                f"Resizing {texture_path} from {width}x{height} to {max_size}x{max_size} .."
            )
            image = cls.resize_image(image, max_size, max_size)

        # Optimize bit depth
        if optimize_bit_depth:
            image = cls.set_bit_depth(image, map_type_suffix)

        if generate_mipmaps:
            image = cls.generate_mipmaps(image)

        # Format filenames
        old_texture_path = (
            cls.resolve_texture_filename(
                texture_path, map_type_suffix, suffix=suffix_old
            )
            if suffix_old
            else None
        )

        optimized_texture_path = cls.resolve_texture_filename(
            texture_path, map_type_suffix, suffix=suffix_opt, ext=output_type
        )

        # Move the old file to an archive folder if enabled
        if old_files_folder:
            old_folder = os.path.join(output_dir, old_files_folder)
            FileUtils.move_file(
                texture_path,
                old_folder,
                new_name=(
                    os.path.basename(old_texture_path) if old_texture_path else None
                ),
            )

        # Save the optimized image
        image.save(optimized_texture_path, format=output_type or image.format)

        print(
            f"Saved optimized texture: {optimized_texture_path} ({image.size[0]}x{image.size[1]})"
        )
        return optimized_texture_path

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
            return ImgUtils.create_dx_from_gl(available["Normal_OpenGL"])
        if map_type == "Normal_OpenGL" and "Normal_DirectX" in available:
            return ImgUtils.create_gl_from_dx(available["Normal_DirectX"])
        return None

    @classmethod
    def pack_msao_texture(
        cls,
        metallic_map_path: str,
        ao_map_path: str,
        alpha_map_path: str,
        output_dir: str = None,
        suffix: str = "_MSAO",
        invert_alpha: bool = False,
    ) -> str:
        """Packs Metallic (R), AO (G), and Smoothness/Roughness (A) into a single MSAO texture.

        Parameters:
            metallic_map_path (str): Path to the metallic texture map.
            ao_map_path (str): Path to the ambient occlusion texture map.
            alpha_map_path (str): Path to the smoothness or roughness texture map.
            output_dir (str, optional): Output directory. If None, uses metallic map directory.
            suffix (str, optional): Suffix for the output file name.
            invert_alpha (bool, optional): If True, inverts the alpha channel (roughness to smoothness).

        Returns:
            str: Path to the packed MSAO texture.
        """
        cls.assert_pathlike(metallic_map_path, "metallic_map_path")
        cls.assert_pathlike(ao_map_path, "ao_map_path")
        cls.assert_pathlike(alpha_map_path, "alpha_map_path")

        base_name = cls.get_base_texture_name(metallic_map_path)

        if output_dir is None:
            output_dir = os.path.dirname(metallic_map_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        output_path = os.path.join(output_dir, f"{base_name}{suffix}.png")

        # Pack channels using the existing pack_channels method
        packed_image = cls.pack_channels(
            channel_files={
                "R": metallic_map_path,
                "G": ao_map_path,
                "B": None,  # Keep blue channel empty/black
                "A": alpha_map_path,
            },
            channels=["R", "G", "B", "A"],
            out_mode="RGBA",
            fill_values={"R": 0, "G": 255, "B": 0, "A": 255 if not invert_alpha else 0},
            output_path=output_path,
            output_format="PNG",
        )

        if invert_alpha:
            # Invert the alpha channel after packing
            img = cls.ensure_image(packed_image)
            img = cls.invert_channels(img, "A")
            img.save(output_path)

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
    ) -> Tuple[str, str, str]:
        """Unpacks AO (R), Roughness (G), and Metallic (B) maps from a combined ORM texture.

        Parameters:
            orm_map_path (str): Path to the ORM texture map.
            output_dir (str, optional): Directory path for the output. If None, uses input map directory.
            ao_suffix (str, optional): Suffix for the AO output file name.
            roughness_suffix (str, optional): Suffix for the roughness output file name.
            metallic_suffix (str, optional): Suffix for the metallic output file name.
            invert_roughness (bool, optional): If True, inverts the roughness to create smoothness.

        Returns:
            Tuple[str, str, str]: Paths to the extracted AO, roughness/smoothness, and metallic maps.
        """
        cls.assert_pathlike(orm_map_path, "orm_map_path")

        if not os.path.exists(orm_map_path):
            raise FileNotFoundError(f"Input file not found: {orm_map_path}")

        # Load the combined texture
        combined_image = cls.ensure_image(orm_map_path)
        if combined_image.mode != "RGB" and combined_image.mode != "RGBA":
            combined_image = combined_image.convert("RGB")

        # Get base name for output files
        base_name = cls.get_base_texture_name(orm_map_path)

        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(orm_map_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        # Extract channels
        # R = Occlusion
        ao_channel = combined_image.getchannel("R").convert("L")
        # G = Roughness
        roughness_channel = combined_image.getchannel("G").convert("L")
        # B = Metallic
        metallic_channel = combined_image.getchannel("B").convert("L")

        # Invert roughness if requested (to create smoothness)
        if invert_roughness:
            roughness_channel = ImageOps.invert(roughness_channel)
            if roughness_suffix == "_Roughness":
                roughness_suffix = "_Smoothness"

        # Create output file paths
        ao_output_path = os.path.join(output_dir, f"{base_name}{ao_suffix}.png")
        roughness_output_path = os.path.join(
            output_dir, f"{base_name}{roughness_suffix}.png"
        )
        metallic_output_path = os.path.join(
            output_dir, f"{base_name}{metallic_suffix}.png"
        )

        # Save the extracted maps
        ao_channel.save(ao_output_path, format="PNG")
        roughness_channel.save(roughness_output_path, format="PNG")
        metallic_channel.save(metallic_output_path, format="PNG")

        return ao_output_path, roughness_output_path, metallic_output_path

    @classmethod
    def unpack_albedo_transparency(
        cls,
        albedo_map_path: str,
        output_dir: str = None,
        base_color_suffix: str = "_BaseColor",
        opacity_suffix: str = "_Opacity",
    ) -> Tuple[str, str]:
        """Unpacks Base Color (RGB) and Opacity (A) from an Albedo+Transparency map.

        Parameters:
            albedo_map_path (str): Path to the Albedo texture map.
            output_dir (str, optional): Directory path for the output. If None, uses input map directory.
            base_color_suffix (str, optional): Suffix for the base color output file name.
            opacity_suffix (str, optional): Suffix for the opacity output file name.

        Returns:
            Tuple[str, str]: Paths to the extracted base color and opacity maps.
        """
        cls.assert_pathlike(albedo_map_path, "albedo_map_path")

        if not os.path.exists(albedo_map_path):
            raise FileNotFoundError(f"Input file not found: {albedo_map_path}")

        # Load the combined texture
        combined_image = cls.ensure_image(albedo_map_path)

        # Check for alpha channel
        if "A" not in combined_image.getbands():
            # If no alpha, just return the original as base color and None for opacity
            # But to be consistent with unpacking, we might want to save a copy?
            # For now, let's assume the user knows what they are doing if they call this.
            # Or better, just return the original path and a white opacity map?
            # Let's stick to extraction.
            pass

        if combined_image.mode != "RGBA":
            combined_image = combined_image.convert("RGBA")

        # Get base name for output files
        base_name = cls.get_base_texture_name(albedo_map_path)

        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(albedo_map_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        # Extract channels
        base_color_image = combined_image.convert("RGB")
        opacity_channel = combined_image.getchannel("A").convert("L")

        # Create output file paths
        base_color_output_path = os.path.join(
            output_dir, f"{base_name}{base_color_suffix}.png"
        )
        opacity_output_path = os.path.join(
            output_dir, f"{base_name}{opacity_suffix}.png"
        )

        # Save the extracted maps
        base_color_image.save(base_color_output_path, format="PNG")
        opacity_channel.save(opacity_output_path, format="PNG")

        return base_color_output_path, opacity_output_path

    @classmethod
    def unpack_msao_texture(
        cls,
        msao_map_path: str,
        output_dir: str = None,
        metallic_suffix: str = "_Metallic",
        ao_suffix: str = "_AO",
        smoothness_suffix: str = "_Smoothness",
        invert_smoothness: bool = False,
    ) -> Tuple[str, str, str]:
        """Unpacks Metallic (R), AO (G), and Smoothness (A) maps from a combined MSAO texture.

        Parameters:
            msao_map_path (str): Path to the MSAO texture map.
            output_dir (str, optional): Directory path for the output. If None, uses input map directory.
            metallic_suffix (str, optional): Suffix for the metallic output file name.
            ao_suffix (str, optional): Suffix for the AO output file name.
            smoothness_suffix (str, optional): Suffix for the smoothness output file name.
            invert_smoothness (bool, optional): If True, inverts the smoothness to create roughness.

        Returns:
            Tuple[str, str, str]: Paths to the extracted metallic, AO, and smoothness/roughness maps.

        Raises:
            ValueError: If the input path is invalid.
            FileNotFoundError: If the input file does not exist.
        """
        cls.assert_pathlike(msao_map_path, "msao_map_path")

        if not os.path.exists(msao_map_path):
            raise FileNotFoundError(f"Input file not found: {msao_map_path}")

        # Load the combined texture
        combined_image = cls.ensure_image(msao_map_path)

        # Get base name for output files
        base_name = cls.get_base_texture_name(msao_map_path)

        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(msao_map_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        # Extract channels
        if combined_image.mode not in ("RGBA", "RGB"):
            combined_image = combined_image.convert("RGBA")

        # Extract metallic channel (Red)
        metallic_channel = combined_image.getchannel("R").convert("L")

        # Extract AO channel (Green)
        ao_channel = combined_image.getchannel("G").convert("L")

        # Extract smoothness channel (Alpha)
        if combined_image.mode == "RGBA":
            smoothness_channel = combined_image.getchannel("A").convert("L")
        else:
            # If no alpha channel, create a default smoothness (white = smooth)
            smoothness_channel = Image.new("L", combined_image.size, 255)
            print(
                f"// Warning: No alpha channel found in {msao_map_path}, using default smoothness."
            )

        # Invert smoothness if requested (to create roughness)
        if invert_smoothness:
            smoothness_channel = ImageOps.invert(smoothness_channel)
            # Update suffix if inverted
            if smoothness_suffix == "_Smoothness":
                smoothness_suffix = "_Roughness"

        # Create output file paths
        metallic_output_path = os.path.join(
            output_dir, f"{base_name}{metallic_suffix}.png"
        )
        ao_output_path = os.path.join(output_dir, f"{base_name}{ao_suffix}.png")
        smoothness_output_path = os.path.join(
            output_dir, f"{base_name}{smoothness_suffix}.png"
        )

        # Save the extracted maps
        metallic_channel.save(metallic_output_path, format="PNG")
        ao_channel.save(ao_output_path, format="PNG")
        smoothness_channel.save(smoothness_output_path, format="PNG")

        return metallic_output_path, ao_output_path, smoothness_output_path

    @classmethod
    def convert_smoothness_to_roughness(
        cls, smoothness_path: str, output_dir: str = None
    ) -> str:
        """Convert a Smoothness map to a Roughness map by inverting the grayscale values.

        Smoothness (0=rough, 255=smooth) becomes Roughness (0=smooth, 255=rough).

        Parameters:
            smoothness_path (str): Path to the smoothness texture map.
            output_dir (str, optional): Output directory. If None, uses smoothness map directory.

        Returns:
            str: Path to the converted roughness map.
        """
        cls.assert_pathlike(smoothness_path, "smoothness_path")

        if not os.path.exists(smoothness_path):
            raise FileNotFoundError(f"Input file not found: {smoothness_path}")

        # Load and invert the smoothness map
        smoothness_image = cls.ensure_image(smoothness_path, "L")
        roughness_image = cls.invert_grayscale_image(smoothness_image)

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
        roughness_image.save(output_path)

        return output_path

    @classmethod
    def convert_roughness_to_smoothness(
        cls, roughness_path: str, output_dir: str = None
    ) -> str:
        """Convert a Roughness map to a Smoothness map by inverting the grayscale values.

        Roughness (0=smooth, 255=rough) becomes Smoothness (0=rough, 255=smooth).

        Parameters:
            roughness_path (str): Path to the roughness texture map.
            output_dir (str, optional): Output directory. If None, uses roughness map directory.

        Returns:
            str: Path to the converted smoothness map.
        """
        cls.assert_pathlike(roughness_path, "roughness_path")

        if not os.path.exists(roughness_path):
            raise FileNotFoundError(f"Input file not found: {roughness_path}")

        # Load and invert the roughness map
        roughness_image = cls.ensure_image(roughness_path, "L")
        smoothness_image = cls.invert_grayscale_image(roughness_image)

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
        smoothness_image.save(output_path)

        return output_path


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
