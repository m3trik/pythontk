# !/usr/bin/python
# coding=utf-8
import os
from typing import List, Tuple, Dict, Union, Any

try:
    import numpy as np
except ImportError as e:
    print(f"# ImportError: {__file__}\n\t{e}")
try:
    from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageChops, ImageDraw
except ImportError as e:
    print(f"# ImportError: {__file__}\n\t{e}")

# from this package:
from pythontk import core_utils
from pythontk import file_utils
from pythontk import iter_utils


class ImgUtils(core_utils.HelpMixin):
    """Helper methods for working with image file formats."""

    map_types = {
        "Base_Color": ("Base_Color", "BaseColor", "Color", "_BC"),
        "Albedo_Transparency": ("Albedo_Transparency", "AlbedoTransparency", "_AT"),
        "Roughness": ("Roughness", "Rough", "RGH", "_R"),
        "Metallic": ("Metallic", "Metal", "Metalness", "MTL", "_M"),
        "Metallic_Smoothness": ("Metallic_Smoothness", "MetallicSmoothness", "_MS"),
        "Normal": ("Normal", "Norm", "NRM", "_N"),
        "Normal_DirectX": ("Normal_DirectX", "NormalDX", "_NDX"),
        "Normal_OpenGL": ("Normal_OpenGL", "NormalGL", "_NGL"),
        "Height": ("Height", "High", "HGT", "_H"),
        "Emissive": ("Emissive", "Emit", "EMI", "_E"),
        "Diffuse": ("Diffuse", "_DF", "Diff", "DIF", "_D"),
        "Specular": ("Specular", "Spec", "_S"),
        "Glossiness": ("Glossiness", "Gloss", "Glos", "Gls", "Glo", "_G"),
        "Displacement": ("Displacement", "_DP", "Displace", "Disp", "Dis", "_D"),
        "Refraction": ("Refraction", "IndexofRefraction", "_IOR"),
        "Reflection": ("Reflection", "_RF"),
        "Opacity": ("Opacity", "Transparency", "Alpha", "Alpha_Mask", "_O"),
        "Smoothness": ("Smoothness", "Smooth", "_SM"),
        "Thickness": ("Thickness", "_TH"),
        "Anisotropy": ("Anisotropy", "_AN"),
        "Subsurface_Scattering": ("Subsurface_Scattering", "SSS", "_SSS"),
        "Sheen": ("Sheen", "_SH"),
        "Clearcoat": ("Clearcoat", "_CC"),
        "Ambient_Occlusion": (
            "Ambient_Occlusion",
            "AmbientOcclusion",
            "Mixed_AO",
            "_AO",
        ),
    }

    map_backgrounds = {  # Default map backgrounds in RGBA format by map type.
        "Base_Color": (127, 127, 127, 255),
        "Albedo_Transparency": (0, 0, 0, 255),
        "Roughness": (255, 255, 255, 255),
        "Metallic": (0, 0, 0, 255),
        "Metallic_Smoothness": (255, 255, 255, 255),
        "Normal": (127, 127, 255, 255),
        "Normal_DirectX": (127, 127, 255, 255),
        "Normal_OpenGL": (127, 127, 255, 255),
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
        "Metallic_Smoothness": "RGB",  # Multi-channel map for metallic and smoothness.
        "Normal": "RGB",  # Full color normal map.
        "Normal_DirectX": "RGB",  # DirectX normal map with Y-axis inversion.
        "Normal_OpenGL": "RGB",  # OpenGL normal map with standard Y-axis.
        "Height": "I",  # Integer mode for height, often 16 or 32-bit.
        "Emissive": "RGB",  # Full color map for self-illumination.
        "Diffuse": "RGB",  # Full color map for diffuse properties.
        "Specular": "L",  # Grayscale map for specular highlights.
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
        "RGBA": 32,
        "CMYK": 32,
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
    def ensure_image(cls, input_image):
        """Ensures that the input is a PIL.Image.Image object. If the input is a string (file path),
        it loads the image from the path. Otherwise, it assumes the input is already an image object.

        Parameters:
            input_image (str/PIL.Image.Image): The input image path or image object.

        Returns:
            PIL.Image.Image: The ensured image object.
        """
        if isinstance(input_image, str):
            # Load the image from the file path
            try:
                image = Image.open(input_image)
                # Ensure the file is read, as Image.open is lazy
                image.load()
                return image
            except IOError as e:
                raise IOError(
                    f"Unable to load image from path '{input_image}'. Error: {e}"
                )
        elif isinstance(input_image, Image.Image):
            # Input is already an image object
            return input_image
        else:
            raise TypeError(
                "Input must be a file path (str) or a PIL.Image.Image object."
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
    def save_image(cls, image, name):
        """
        Parameters:
            image (obj): PIL image object.
            name (str): Path + filename including extension. ie. new_image.png
        """
        im = cls.ensure_image(image)
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
        images = {}
        for f in file_utils.FileUtils.get_dir_contents(
            directory, "filepath", inc_files=inc, exc_files=exc
        ):
            im = cls.load_image(f)
            images[f] = im

        return images

    @classmethod
    def get_map_type_from_filename(
        cls, file: str, key: bool = True, validate: str = None
    ) -> str:
        """Determine the map type from the filename and optionally validate it.

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
        filename = file_utils.FileUtils.format_path(file, "name")

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
    def get_base_texture_name(cls, filepath_or_filename: str) -> str:
        """Extracts the base texture name from a given filename or full filepath,
        removing known suffixes based on the class attribute `map_types` dynamically,
        case-insensitively.

        Parameters:
            filepath_or_filename (str): The full file path or just the filename.

        Returns:
            str: The base name of the texture without the map type suffix.
        """
        import re

        filename = os.path.basename(filepath_or_filename)

        # Extract the base name without the extension
        base_name, extension = os.path.splitext(filename)

        # Compile a single regex pattern that matches any known suffix
        suffixes_pattern = "|".join(
            re.escape(suffix)
            for suffixes in cls.map_types.values()
            for suffix in suffixes
        )

        # Create a regex to match and remove the suffixes at the end of the base name
        pattern = re.compile(
            f"(?:_{suffixes_pattern}|{suffixes_pattern})$", re.IGNORECASE
        )

        # Remove the matched suffix, if any
        base_name = pattern.sub("", base_name)

        # Remove any trailing underscores
        base_name = base_name.rstrip("_")

        return base_name

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
        types = iter_utils.IterUtils.make_iterable(types)
        return [f for f in files if cls.get_map_type_from_filename(f) in types]

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
            map_type = cls.get_map_type_from_filename(file_path)
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

        map_types = iter_utils.IterUtils.make_iterable(map_types)

        result = next(
            (
                True
                for i in files.keys()
                if cls.get_map_type_from_filename(i) in map_types
            ),
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
        typ = cls.get_map_type_from_filename(file)
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
    def invert_grayscale_image(cls, image):
        """Inverts a grayscale image. This method ensures the input is a grayscale image before inverting.

        Parameters:
            image (str/PIL.Image.Image): An image or path to an image to invert.

        Returns:
            PIL.Image.Image: The inverted grayscale image.
        """
        im = cls.ensure_image(image)
        if im.mode != "L":
            raise ValueError("Image must be in grayscale ('L') mode to invert.")
        return ImageOps.invert(im)

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
    @core_utils.CoreUtils.listify(threading=True)
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
    def pack_channel_into_alpha(
        cls, image, alpha, output_path=None, invert_alpha=False
    ):
        """Packs a channel from the alpha source image into the alpha channel of the base image.
        Optionally inverts the alpha source image.

        Parameters:
            image (str/PIL.Image.Image): File path or image object for the base texture map.
            alpha (str/PIL.Image.Image): File path or image object for the texture map to pack into the alpha channel.
            output_path (str, optional): File path to save the modified base texture map with the alpha packed.
            invert_alpha (bool, optional): If True, inverts the alpha source image before packing.

        Returns:
            PIL.Image.Image: The modified base image with the alpha packed into its alpha channel.
        """
        base_img = cls.ensure_image(image)
        alpha_img = cls.ensure_image(alpha)

        # Optionally invert the alpha source image
        if invert_alpha:
            alpha_img = cls.invert_grayscale_image(alpha_img)

        alpha_img = alpha_img.convert("L")  # Ensure alpha is in grayscale for packing

        # Merge the alpha channel into the base image
        if base_img.mode in ["L", "LA"]:
            base_img = base_img.convert("LA")
            combined_img = Image.merge("LA", (base_img.getchannel(0), alpha_img))
        else:
            base_img = base_img.convert("RGBA")
            combined_img = Image.merge("RGBA", (*base_img.split()[:3], alpha_img))

        # Determine the output path if not provided
        if not output_path:
            if isinstance(image, str):
                output_path = image  # Use the base image path
            else:
                raise ValueError(
                    "Output path must be provided when using Image objects directly"
                )

        combined_img.save(output_path)
        return output_path

    @classmethod
    def pack_transparency_into_albedo(
        cls,
        albedo_map_path: str,
        alpha_map_path: str,
        output_dir: str = None,
        suffix: str = "_AlbedoTransparency",
        invert_alpha: bool = False,
    ) -> str:
        """Combines an albedo texture with a transparency map by packing the transparency information into the alpha channel of the albedo texture.

        Parameters:
            albedo_map_path (str): Path to the albedo (base color) texture map.
            alpha_map_path (str): Path to the transparency texture map.
            output_dir (str, optional): Directory path for the output. If None, the output directory will be the same as the albedo map path.
            invert_alpha (bool, optional): If True, the alpha (transparency) texture will be inverted.
            suffix (str, optional): Suffix for the output file name, defaulting to '_AlbedoTransparency'.

        Returns:
            str: The file path of the newly created albedo-transparency texture map.
        """
        base_name = cls.get_base_texture_name(albedo_map_path)
        if output_dir is None:
            output_dir = os.path.dirname(albedo_map_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        output_path = os.path.join(output_dir, f"{base_name}{suffix}.png")

        success = cls.pack_channel_into_alpha(
            albedo_map_path, alpha_map_path, output_path, invert_alpha=invert_alpha
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
    def create_dx_from_gl(cls, file: str, output_path: str = None) -> str:
        """Create and save a DirectX map using a given OpenGL image.
        The new map will be saved next to the existing map if no output path is provided.

        Parameters:
            file (str): The full path to an OpenGL normal map file.
            output_path (str, optional): Path to save the resulting DirectX map. If None, saves next to the input map.

        Returns:
            (str): filepath of the new image.

        Raises:
            ValueError: If the map type is not found in the expected map types.
        """
        # Get and validate the map type from the filename
        typ = cls.get_map_type_from_filename(file, key=False, validate="Normal_OpenGL")

        inverted_image = cls.invert_channels(file, "g")

        if output_path is None:
            output_dir = file_utils.FileUtils.format_path(file, "path")
            name = file_utils.FileUtils.format_path(file, "name")
            ext = file_utils.FileUtils.format_path(file, "ext")

            try:
                index = cls.map_types["Normal_OpenGL"].index(typ)
                new_type = cls.map_types["Normal_DirectX"][index]
            except ValueError:
                raise ValueError(
                    f"Type '{typ}' not found in 'Normal_OpenGL' map types."
                )

            name = name.removesuffix(typ)
            output_path = f"{output_dir}/{name}{new_type}.{ext}"

        output_path = os.path.abspath(output_path)
        inverted_image.save(output_path)
        return output_path

    @classmethod
    def create_gl_from_dx(cls, file: str, output_path: str = None) -> str:
        """Create and save an OpenGL map using a given DirectX image.
        The new map will be saved next to the existing map if no output path is provided.

        Parameters:
            file (str): The full path to a DirectX normal map file.
            output_path (str, optional): Path to save the resulting OpenGL map. If None, saves next to the input map.

        Returns:
            (str): filepath of the new image.

        Raises:
            ValueError: If the map type is not found in the expected map types.
        """
        # Get and validate the map type from the filename
        typ = cls.get_map_type_from_filename(file, key=False, validate="Normal_DirectX")

        inverted_image = cls.invert_channels(file, "g")

        if output_path is None:
            output_dir = file_utils.FileUtils.format_path(file, "path")
            name = file_utils.FileUtils.format_path(file, "name")
            ext = file_utils.FileUtils.format_path(file, "ext")

            try:
                index = cls.map_types["Normal_DirectX"].index(typ)
                new_type = cls.map_types["Normal_OpenGL"][index]
            except ValueError:
                raise ValueError(
                    f"Type '{typ}' not found in 'Normal_DirectX' map types."
                )

            name = name.removesuffix(typ)
            output_path = f"{output_dir}/{name}{new_type}.{ext}"

        output_path = os.path.abspath(output_path)
        inverted_image.save(output_path)
        return output_path

    @classmethod
    def create_roughness_from_spec(
        cls,
        specular_map_path: str,
        output_dir: str = None,
        suffix: str = "_Roughness",
        apply_gamma: bool = True,
        gamma_value: float = 2.2,
        apply_normalization: bool = True,
        adjust_contrast: bool = True,
        contrast_factor: float = 1.2,
        apply_blur: bool = True,
        blur_radius: float = 1.0,
    ) -> str:
        """Creates a roughness map from a specular map by converting it to grayscale, inverting it, and applying necessary adjustments.

        Parameters:
            specular_map_path (str): Path to the specular map.
            output_dir (str, optional): Directory path for the output. If None, the output directory will be the same as the specular map path.
            suffix (str, optional): Suffix for the output file name, defaulting to '_Roughness'.
            apply_gamma (bool, optional): Apply gamma correction.
            gamma_value (float, optional): Gamma value to use.
            apply_normalization (bool, optional): Apply normalization.
            adjust_contrast (bool, optional): Adjust contrast.
            contrast_factor (float, optional): Contrast adjustment factor.
            apply_blur (bool, optional): Apply blurring.
            blur_radius (float, optional): Radius for blurring.

        Returns:
            str: Path to the saved roughness map.

        Raises:
            ValueError: If the map type is not found in the expected map types.
        """
        # Validate the map type
        cls.get_map_type_from_filename(
            specular_map_path, key=False, validate="Specular"
        )

        if output_dir is None:
            output_dir = os.path.dirname(specular_map_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        base_name = cls.get_base_texture_name(specular_map_path)
        output_path = os.path.join(output_dir, f"{base_name}{suffix}.png")

        # Load and convert the specular image to grayscale
        specular_image = cls.ensure_image(specular_map_path).convert("L")

        if apply_gamma:
            specular_image = ImageEnhance.Brightness(specular_image).enhance(
                gamma_value
            )

        if apply_normalization:
            specular_array = np.array(specular_image)
            normalized_array = (
                (specular_array - specular_array.min())
                / (specular_array.max() - specular_array.min())
                * 255
            )
            specular_image = Image.fromarray(normalized_array.astype(np.uint8))

        if adjust_contrast:
            enhancer = ImageEnhance.Contrast(specular_image)
            specular_image = enhancer.enhance(contrast_factor)

        if apply_blur:
            specular_image = specular_image.filter(
                ImageFilter.GaussianBlur(blur_radius)
            )

        # Invert the specular image to create the roughness map
        roughness_image = ImageOps.invert(specular_image)

        roughness_image.save(output_path)
        return output_path

    @classmethod
    def create_metallic_from_spec(
        cls,
        specular_map_path: str,
        output_dir: str = None,
        suffix: str = "_Metallic",
        apply_gamma: bool = True,
        gamma_value: float = 2.2,
        apply_normalization: bool = True,
        adjust_contrast: bool = True,
        contrast_factor: float = 1.2,
        apply_blur: bool = True,
        blur_radius: float = 1.0,
    ) -> str:
        """Creates a metallic map from a specular map by converting it to grayscale and applying necessary adjustments.

        Parameters:
            specular_map_path (str): Path to the specular map.
            output_dir (str, optional): Directory path for the output. If None, the output directory will be the same as the specular map path.
            suffix (str, optional): Suffix for the output file name, defaulting to '_Metallic'.
            apply_gamma (bool, optional): Apply gamma correction.
            gamma_value (float, optional): Gamma value to use.
            apply_normalization (bool, optional): Apply normalization.
            adjust_contrast (bool, optional): Adjust contrast.
            contrast_factor (float, optional): Contrast adjustment factor.
            apply_blur (bool, optional): Apply blurring.
            blur_radius (float, optional): Radius for blurring.

        Returns:
            str: Path to the saved metallic map.

        Raises:
            ValueError: If the map type is not found in the expected map types.
        """
        # Validate the map type
        cls.get_map_type_from_filename(
            specular_map_path, key=False, validate="Specular"
        )

        if output_dir is None:
            output_dir = os.path.dirname(specular_map_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        base_name = cls.get_base_texture_name(specular_map_path)
        output_path = os.path.join(output_dir, f"{base_name}{suffix}.png")

        # Load and convert the specular image to grayscale
        specular_image = cls.ensure_image(specular_map_path).convert("L")

        if apply_gamma:
            specular_image = ImageEnhance.Brightness(specular_image).enhance(
                gamma_value
            )

        if apply_normalization:
            specular_array = np.array(specular_image)
            normalized_array = (
                (specular_array - specular_array.min())
                / (specular_array.max() - specular_array.min())
                * 255
            )
            specular_image = Image.fromarray(normalized_array.astype(np.uint8))

        if adjust_contrast:
            enhancer = ImageEnhance.Contrast(specular_image)
            specular_image = enhancer.enhance(contrast_factor)

        if apply_blur:
            specular_image = specular_image.filter(
                ImageFilter.GaussianBlur(blur_radius)
            )

        specular_image.save(output_path)
        return output_path

    @classmethod
    def optimize_texture(
        cls,
        texture_path: str,
        output_dir: str = None,
        output_type="PNG",
        max_size: int = None,
        suffix: str = None,
    ) -> str:
        """Optimizes a texture by resizing, setting bit depth, and adjusting image type according to the map type.

        Parameters:
            texture_path (str): Path to the texture file.
            output_dir (str, optional): Directory path for the optimized texture. If None, the texture will be saved next to the original.
            output_type (str): The output image type for the optimized texture.
            max_size (int, optional): Maximum size for the longest dimension of the texture. Defaults to None.
            suffix (str, optional): Suffix to add to the optimized file name. Defaults to None.
                        e.g., '_opt' will result in '<name>_opt_<map_type>.<extension>'
        Returns:
            str: Path to the optimized texture.
        """
        if output_dir is None:
            output_dir = os.path.dirname(texture_path)
        elif not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # Determine the map type
        map_type = cls.get_map_type_from_filename(texture_path)
        base_name = cls.get_base_texture_name(texture_path)
        suffix = suffix if suffix else ""
        output_file_name = f"{base_name}{suffix}_{map_type}.{output_type.lower()}"
        output_path = os.path.join(output_dir, output_file_name)

        # Load the image
        image = cls.ensure_image(texture_path)

        # Resize the image if larger than max size
        if max_size and max(image.size) > max_size:
            image = cls.resize_image(image, max_size, max_size)

        # Adjust bit depth and image mode
        image = cls.set_bit_depth(image, map_type)

        # Save the optimized image
        image.save(output_path, format=output_type)
        return output_path

    @classmethod
    def batch_optimize_textures(cls, directory: str, **kwargs):
        """Batch optimizes all textures in a directory.

        Parameters:
            directory (str): Directory containing the textures to optimize.
            output_dir (str, optional): Directory path for the optimized textures. If None, the textures will be saved next to the originals.
            max_size (int, optional): Maximum size for the longest dimension of the textures. Defaults to 4096
        """
        textures = cls.get_images(directory)
        print(f"Optimizing textures in: {directory}")
        for texture_path in textures.keys():
            cls.optimize_texture(texture_path, **kwargs)
        print(f"{len(textures)} textures optimized.")


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
