# !/usr/bin/python
# coding=utf-8
from __future__ import annotations

import os
import math
import re

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

# From this package:
from pythontk.core_utils._core_utils import CoreUtils
from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.str_utils._str_utils import StrUtils
from pythontk.img_utils.map_registry import MapRegistry


class ImgUtils(HelpMixin):
    """Helper methods for working with image file formats."""

    map_types: Dict[str, Tuple[str, ...]] = MapRegistry().get_map_types()
    map_backgrounds: Dict[str, Tuple[int, int, int, int]] = (
        MapRegistry().get_map_backgrounds()
    )
    map_modes: Dict[str, str] = MapRegistry().get_map_modes()

    texture_file_types = ["png", "jpg", "bmp", "tga", "tiff", "gif", "exr", "hdr"]

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
        if Image is None:
            raise ImportError(
                "Pillow (PIL) is not installed. Image operations are unavailable."
            )

        if isinstance(input_image, (str, os.PathLike)):
            input_image = str(input_image)
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

    @classmethod
    def enforce_mode(
        cls, image: Image.Image, target_mode: str, allow_compatible: bool = True
    ) -> Image.Image:
        """Converts image to target_mode, optionally allowing compatible modes to preserve file size.

        Compatible modes:
        - Target RGB: Allows P (Indexed) and L (Grayscale)
        - Target RGBA: Allows P (Indexed)
        - Target L: Strict conversion (P is converted to L)

        Parameters:
            image (PIL.Image.Image): Input image.
            target_mode (str): Desired mode (RGB, RGBA, L).
            allow_compatible (bool): If True, allows smaller compatible modes.

        Returns:
            PIL.Image.Image: The converted (or original) image.
        """
        if not allow_compatible:
            return image.convert(target_mode) if image.mode != target_mode else image

        if target_mode == "RGB":
            if image.mode in ["RGB", "P", "L"]:
                return image
            return image.convert("RGB")
        elif target_mode == "RGBA":
            if image.mode in ["RGBA", "P"]:
                return image
            return image.convert("RGBA")
        elif target_mode == "L":
            # Always enforce L for grayscale maps to ensure single channel
            if image.mode != "L":
                return image.convert("L")
            return image

        return image.convert(target_mode) if image.mode != target_mode else image

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
    def save_image(
        cls, image: Union[str, Image.Image], name: str, mode: str = None, **kwargs
    ):
        """
        Saves an image to the specified path, with optional mode conversion.

        Parameters:
            image (str | PIL.Image.Image): Image object or file path.
            name (str): Output path including filename and extension (e.g., "output.png").
            mode (str, optional): Converts the image to the specified mode before saving (e.g., "RGB", "L").
            **kwargs: Additional arguments passed to PIL.Image.save (e.g., optimize=True, compress_level=9).
        """
        im = cls.ensure_image(image, mode)  # Now allows optional mode conversion

        # Auto-convert RGBA to RGB if saving as JPEG to prevent OSError
        if name.lower().endswith((".jpg", ".jpeg")) and im.mode == "RGBA":
            im = im.convert("RGB")

        # Handle EXR using OpenCV if available
        if name.lower().endswith(".exr"):
            try:
                # Enable OpenEXR in OpenCV (disabled by default in newer versions)
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                import cv2

                # Convert PIL to Numpy
                img_np = np.array(im)

                # Handle Color Space (PIL is RGB/RGBA, OpenCV expects BGR/BGRA)
                if im.mode == "RGB":
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                elif im.mode == "RGBA":
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA)
                elif im.mode == "L":
                    pass  # Grayscale is fine

                # EXR requires float32
                img_np = img_np.astype(np.float32) / 255.0

                # Save
                cv2.imwrite(name, img_np)
                return
            except ImportError:
                print(
                    "Warning: OpenCV (cv2) not found. Cannot save EXR. Falling back to PIL (likely to fail)."
                )
            except Exception as e:
                print(f"Error saving EXR with OpenCV: {e}. Falling back to PIL.")

        im.save(name, **kwargs)

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
        inc=None,
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
        if inc is None:
            inc = [f"*.{ext}" for ext in cls.texture_file_types]

        cls.assert_pathlike(directory, "directory")

        images = {}
        for f in FileUtils.get_dir_contents(
            directory, "filepath", inc_files=inc, exc_files=exc
        ):
            im = cls.load_image(f)
            images[f] = im

        return images

    @classmethod
    def get_image_info(cls, file_paths: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Get information about image files.

        Parameters:
            file_paths (str or list): Path(s) to image files.

        Returns:
            list[dict]: List of dictionaries containing image info.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        info_list = []
        for path in file_paths:
            if not path:
                continue

            if not os.path.exists(path):
                print(f"Warning: Image path not found: {path}")
                continue

            try:
                size_bytes = os.path.getsize(path)

                with cls.allow_large_images():
                    img = cls.ensure_image(path)
                    width, height = img.size
                    mode = img.mode
                    img_format = img.format

                info = {
                    "path": path,
                    "name": os.path.basename(path),
                    "size": size_bytes,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "format": img_format,
                }
                info_list.append(info)
            except Exception as e:
                print(f"Error getting info for {path}: {e}")

        return info_list

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
    def ensure_pot(cls, image: Union[str, Image.Image]) -> Image.Image:
        """Resizes an image to the nearest Power of Two dimensions.

        Parameters:
            image (str/PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The resized image.
        """
        im = cls.ensure_image(image)
        width, height = im.size

        if width <= 0 or height <= 0:
            return im

        new_width = 2 ** round(math.log2(width))
        new_height = 2 ** round(math.log2(height))

        if (width, height) == (new_width, new_height):
            return im

        print(f"Resizing to POT: {width}x{height} -> {new_width}x{new_height}")
        return im.resize((new_width, new_height), Image.Resampling.LANCZOS)

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

            # Smart conversion: Avoid up-sampling channels if not necessary
            # If target is RGB, allow L (Grayscale) and P (Indexed)
            if target_mode == "RGB" and image.mode in ("L", "P"):
                pass  # Keep as is
            # If target is RGBA, allow PA (Indexed Alpha) and P (if transparent)
            elif target_mode == "RGBA" and image.mode in ("PA", "P"):
                if image.mode == "P" and "transparency" in image.info:
                    pass  # Keep as is
                elif image.mode == "PA":
                    pass  # Keep as is
                else:
                    image = image.convert(target_mode)
            else:
                image = image.convert(target_mode)

        # If the image is already in a standard mode, don't mess with it based on bit depth
        if image.mode in ("RGB", "RGBA", "L", "1", "P"):
            return image

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
        elif len(split_channels) == 3:  # RGB image
            return Image.merge(
                "RGB",
                (
                    inverted_channels["R"],
                    inverted_channels["G"],
                    inverted_channels["B"],
                ),
            )
        else:  # RGBA image
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
        if mode is None:
            if len(to_color) == 4:
                mode = "RGBA"
            elif len(to_color) == 3:
                mode = "RGB"
            else:
                mode = im.mode
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

        return Image.fromarray(data).convert(mode)

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
        invert_channels: list[str] = None,
        **kwargs,
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
            invert_channels (list): List of channels to invert (e.g. ["A"]).
            **kwargs: Additional arguments passed to PIL.Image.save (e.g., optimize=True).

        Returns:
            str | Image.Image: Output path if saving, else the PIL image object.
        """
        if channels is None:
            channels = ["R", "G", "B", "A"]
        if fill_values is None:
            fill_values = {ch: 0 for ch in "RGB"}
            fill_values["A"] = 255
        if invert_channels is None:
            invert_channels = []

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
            img_input = channel_files.get(ch)
            if img_input:
                # Load image once to avoid double I/O
                img_obj = cls.ensure_image(img_input)

                # Optimization: Check if image is constant
                # This avoids expensive resizing artifacts for small constant maps
                is_const, const_color = cls.is_image_constant(img_obj)

                if is_const:
                    # Convert constant color to grayscale
                    # Create 1x1 temp image to handle color conversion correctly
                    temp_img = Image.new(img_obj.mode, (1, 1), const_color)
                    gray_val = temp_img.convert("L").getpixel((0, 0))
                    band = cls.create_image("L", size, color=gray_val)
                else:
                    band = img_obj.convert("L").resize(size)
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

            if ch in invert_channels:
                band = ImageOps.invert(band)

            bands.append(band)

        img = Image.merge(out_mode, bands)

        if output_path:
            img.save(output_path, format=output_format, **kwargs)
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
    ) -> str | Image.Image:
        """Packs a channel from the alpha source image into the alpha channel of the base image.

        Parameters:
            image (str | Image.Image): Base texture (albedo).
            alpha (str | Image.Image): Transparency map to pack into the alpha channel.
            output_path (str, optional): Output path. If None, returns the PIL Image object.
            invert_alpha (bool): Invert the alpha source before packing.
            resize_alpha (bool): Resize the alpha to match the base if needed.
            preserve_existing_alpha (bool): If True, multiply existing alpha with the new alpha.

        Returns:
            str | Image.Image: Path to the saved image or the PIL Image object.
        """
        base_img = cls.ensure_image(image).convert("RGBA")
        r, g, b, existing_alpha_channel = base_img.split()

        alpha_img = cls.ensure_image(alpha)

        final_alpha = alpha_img
        invert_list = ["A"] if invert_alpha else []

        if preserve_existing_alpha:
            # Pre-process alpha for multiplication
            if invert_alpha:
                alpha_img = cls.invert_grayscale_image(alpha_img)
                invert_list = []  # Already inverted

            alpha_img = alpha_img.convert("L")

            # Handle resizing for multiplication
            if alpha_img.size != base_img.size:
                if resize_alpha:
                    # Optimization: Check if alpha is constant
                    is_const, const_color = cls.is_image_constant(alpha_img)
                    if is_const:
                        alpha_img = cls.create_image(
                            "L", base_img.size, color=const_color[0]
                        )
                    else:
                        alpha_img = alpha_img.resize(
                            base_img.size, Image.Resampling.LANCZOS
                        )
                else:
                    raise ValueError(
                        f"Alpha image size {alpha_img.size} does not match base {base_img.size} and resize is disabled."
                    )

            final_alpha = ImageChops.multiply(existing_alpha_channel, alpha_img)

        return cls.pack_channels(
            channel_files={"R": r, "G": g, "B": b, "A": final_alpha},
            output_path=output_path,
            invert_channels=invert_list,
        )

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
    def depalettize_image(cls, image: Image.Image) -> Image.Image:
        """Converts a paletted image (Mode P) to RGB or RGBA.

        Parameters:
            image (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The converted image (RGB or RGBA).
        """
        if image.mode == "P":
            # Check if the palette has transparency
            if "transparency" in image.info:
                return image.convert("RGBA")
            else:
                return image.convert("RGB")
        elif image.mode == "PA":
            return image.convert("RGBA")
        return image

    @classmethod
    def batch_optimize_textures(cls, directory: str, **kwargs):
        """Batch optimizes all textures in a directory.

        Parameters:
            directory (str): Directory containing the textures to optimize.
            output_dir (str, optional): Directory path for the optimized textures. If None, the textures will be saved next to the originals.
            max_size (int, optional): Maximum size for the longest dimension of the textures. Defaults to None (no resizing).
            force_pot (bool, optional): Force Power of Two dimensions. Defaults to False.
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
        force_pot: bool = False,
        suffix_old: str = None,
        suffix_opt: str = None,
        old_files_folder: str = None,
        generate_mipmaps: bool = False,
        optimize_bit_depth: bool = True,
        check_existing: bool = False,
        map_type: str = None,
    ) -> str:
        """Optimizes a texture by resizing, setting bit depth, and adjusting image type.

        Parameters:
            texture_path (str): Path to the texture file.
            output_dir (str, optional): Directory for the optimized texture. Defaults to same directory.
            output_type (str, optional): Output image format (e.g., PNG, TGA). If None, keeps original.
            max_size (int, optional): Maximum size for the longest dimension. Only applies if the image is larger. Defaults to None.
            force_pot (bool): Force Power of Two dimensions.
            suffix_old (str, optional): Suffix to rename the original file before optimization.
            suffix_opt (str, optional): Suffix to append to the optimized file (None = overwrite).
            old_files_folder (str, optional): Name of the folder to store old files.
            generate_mipmaps (bool): Generates mipmaps if enabled.
            optimize_bit_depth (bool): Adjusts bit depth to match the map type.
            check_existing (bool): If True, returns existing optimized file if it exists and is newer.
            map_type (str, optional): The type of map (e.g., "Normal", "MaskMap") to enforce specific modes.

        Returns:
            str: Path to the optimized texture.
        """
        cls.assert_pathlike(texture_path, "texture_path")

        if output_dir is None:
            output_dir = os.path.dirname(texture_path)
        os.makedirs(output_dir, exist_ok=True)

        from pythontk.img_utils.map_factory import MapFactory as TextureMapFactory

        # Determine correct map suffix format
        map_type_suffix = TextureMapFactory.resolve_map_type(texture_path, key=False)
        if map_type_suffix is None:
            map_type_suffix = ""
        map_type_key = map_type or TextureMapFactory.resolve_map_type(
            texture_path, key=True
        )

        # Calculate output path early to check for existence
        temp_path = TextureMapFactory.resolve_texture_filename(
            texture_path,
            map_type_suffix,
            suffix=suffix_opt,
            ext=output_type,
        )
        final_output_path = os.path.join(output_dir, os.path.basename(temp_path))

        if check_existing and os.path.exists(final_output_path):
            # Check if output is newer than input
            if os.path.getmtime(final_output_path) > os.path.getmtime(texture_path):
                print(
                    f"Skipping optimization (existing/newer): {os.path.basename(final_output_path)}"
                )
                return final_output_path

        # Load the image first (before renaming)
        image = cls.ensure_image(texture_path)

        # Get current dimensions
        width, height = image.size

        # Determine if resizing will occur
        will_resize = False
        if max_size and max(width, height) > max_size:
            will_resize = True
        elif force_pot:
            if width > 0 and height > 0:
                nw = 2 ** round(math.log2(width))
                nh = 2 ** round(math.log2(height))
                if (width, height) != (nw, nh):
                    will_resize = True

        # Fix paletted images only if resizing is needed (to ensure high quality resampling)
        if will_resize:
            image = cls.depalettize_image(image)

        # Enforce mode based on map_type
        if map_type_key:
            map_def = MapRegistry().get(map_type_key)
            if map_def:
                # Enforce mode from registry
                if map_def.mode == "RGB":
                    # Allow P (Indexed) and L (Grayscale) for RGB maps to preserve size
                    if image.mode not in ["RGB", "P", "L"]:
                        image = image.convert("RGB")
                elif map_def.mode == "RGBA":
                    # Allow PA (Indexed with Alpha) for RGBA maps
                    if image.mode not in ["RGBA", "PA", "P"]:
                        image = image.convert("RGBA")
                elif map_def.mode == "L":
                    if image.mode not in ["L", "P"]:
                        image = image.convert("L")
            else:
                # Fallback for unknown types (legacy behavior)
                if map_type_key in ["Normal", "Normal_OpenGL", "Normal_DirectX"]:
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                elif map_type_key in ["MSAO", "MaskMap", "ORM"]:
                    if map_type_key in ["MSAO", "MaskMap"] and image.mode != "RGBA":
                        image = image.convert("RGBA")
                    elif map_type_key == "ORM" and image.mode not in ["RGB", "RGBA"]:
                        image = image.convert("RGB")
                elif map_type_key in [
                    "Ambient_Occlusion",
                    "Roughness",
                    "Metallic",
                    "Smoothness",
                    "Height",
                    "Bump",
                ]:
                    # Grayscale maps should be L or RGB, never P
                    if image.mode == "P":
                        image = image.convert("L")
        else:
            # If no map type is known, ensure we don't have a paletted image with transparency issues
            # Only depalettize if we are resizing (handled above) or if we really need to?
            # If no map type, we probably want to preserve original as much as possible.
            pass

        # Resize if the image is larger than max_size
        if max_size and max(width, height) > max_size:
            print(
                f"Resizing {texture_path} from {width}x{height} to {max_size}x{max_size} .."
            )
            image = cls.resize_image(image, max_size, max_size)

        # Force POT if requested
        if force_pot:
            image = cls.ensure_pot(image)

        # Optimize bit depth
        if optimize_bit_depth:
            # Skip bit depth optimization for Paletted images to preserve index
            if image.mode != "P":
                image = cls.set_bit_depth(image, map_type_suffix)

        if generate_mipmaps:
            image = cls.generate_mipmaps(image)

        # Format filenames
        old_texture_path = (
            TextureMapFactory.resolve_texture_filename(
                texture_path, map_type_suffix, suffix=suffix_old
            )
            if suffix_old
            else None
        )

        optimized_texture_path = TextureMapFactory.resolve_texture_filename(
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
        save_kwargs = {"optimize": True}

        image.save(final_output_path, format=output_type or image.format, **save_kwargs)

        print(
            f"Saved optimized texture: {final_output_path} ({image.size[0]}x{image.size[1]})"
        )
        return final_output_path

    @classmethod
    def is_image_constant(
        cls, image: Union[str, PILImage.Image], tolerance: int = 0
    ) -> Tuple[bool, Optional[Tuple[int, ...]]]:
        """Check if an image is constant color.

        Parameters:
            image: Path to image or PIL Image object.
            tolerance: Max difference between min/max values per channel (0-255).

        Returns:
            Tuple of (is_constant, color_value).
            color_value is a tuple of channel values (e.g. (255, 0, 0) for red).
        """
        try:
            img = cls.ensure_image(image)
            extrema = img.getextrema()

            # Handle single channel (L) vs multi-channel (RGB/RGBA)
            # Single channel returns (min, max)
            # Multi-channel returns [(min, max), (min, max), ...]
            if extrema and isinstance(extrema[0], (int, float)):
                extrema = [extrema]

            is_constant = True
            color = []

            for min_val, max_val in extrema:
                if (max_val - min_val) > tolerance:
                    is_constant = False
                    break
                color.append(int((min_val + max_val) / 2))

            if is_constant:
                return True, tuple(color)
            return False, None

        except Exception as e:
            print(f"Error checking image constancy: {e}")
            return False, None

    @classmethod
    def get_base_texture_name(cls, filepath_or_filename: str) -> str:
        """Extracts the base texture name from a filename or path,
        removing known suffixes (e.g., _normal, _roughness).

        Logic:
        - Long suffixes (>3 chars): Case-insensitive.
        - Short suffixes (<=3 chars): Must start with a capital letter (rest case-insensitive) to avoid false positives.

        Parameters:
            filepath_or_filename (str): A texture path or name.

        Returns:
            str: The base name without map-type suffix.
        """
        cls.assert_pathlike(filepath_or_filename, "filepath_or_filename")

        filename = os.path.basename(str(filepath_or_filename))
        base_name, _ = os.path.splitext(filename)

        short_suffixes = []
        long_suffixes = []

        for suffixes in cls.map_types.values():
            for suffix in suffixes:
                if len(suffix) <= 3:
                    short_suffixes.append(suffix)
                else:
                    long_suffixes.append(suffix)

        # Sort by length descending to ensure longest match first
        short_suffixes.sort(key=len, reverse=True)
        long_suffixes.sort(key=len, reverse=True)

        patterns = []

        # Long suffixes: Case insensitive
        if long_suffixes:
            p = "|".join(re.escape(s) for s in long_suffixes)
            patterns.append(f"(?i:{p})")

        # Short suffixes: Start with capital, rest case insensitive
        if short_suffixes:
            short_parts = []
            for s in short_suffixes:
                if s and s[0].isalpha():
                    # Enforce first char case (assuming registry has it capitalized)
                    first = s[0].upper()
                    rest = re.escape(s[1:])
                    short_parts.append(f"{first}(?i:{rest})")
                else:
                    short_parts.append(re.escape(s))

            p = "|".join(short_parts)
            patterns.append(p)

        suffixes_pattern = "|".join(patterns)

        # Pattern: (underscore + suffix) OR (suffix) at end
        pattern = f"(?:_{suffixes_pattern}|{suffixes_pattern})$"
        base_name = StrUtils.format_suffix(base_name, strip=pattern)

        return base_name.rstrip("_")

    @classmethod
    def extract_channels(
        cls,
        image_path: Union[str, "Image.Image"],
        channel_config: Dict[str, Dict[str, Any]],
        output_dir: str = None,
        base_name: str = None,
        save: bool = True,
        **kwargs,
    ) -> Dict[str, Union[str, "Image.Image"]]:
        """Generic channel extraction utility.

        Extracts specific channels (or combinations like 'RGB') from an image,
        optionally processes them (invert), and saves them.

        Parameters:
            image_path (str | Image.Image): Source image path or object.
            channel_config (dict): Mapping of source channel to configuration.
                Keys: 'R', 'G', 'B', 'A', 'RGB', 'L'.
                Values (dict):
                    - 'suffix' (str): Output filename suffix (e.g. '_AO').
                    - 'invert' (bool, optional): Whether to invert the result.
                    - 'default' (int, optional): Default value (0-255) if channel missing.
            output_dir (str, optional): Output directory. If None, uses source directory.
            base_name (str, optional): Base name for output files. If None, derived from image path.
            save (bool): Whether to save to disk. Defaults to True.
            **kwargs: Additional arguments for Image.save().
                - output_format (str): Format to save as (default: "PNG").
                - ext (str): Extension to use (default: "png").

        Returns:
            Dict[str, str | Image.Image]: Dictionary mapping source channel keys to
            the resulting file path (if save=True) or PIL Image object (if save=False).
        """
        # Load image
        img = cls.ensure_image(image_path)

        # Determine output directory and base name
        if base_name is None:
            if isinstance(image_path, str):
                base_name = cls.get_base_texture_name(image_path)
            else:
                base_name = "texture"

        if output_dir is None:
            if isinstance(image_path, str):
                output_dir = os.path.dirname(image_path)
            else:
                output_dir = os.getcwd()

        if save and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Extract format/extension from kwargs
        output_format = kwargs.pop("output_format", "PNG")
        ext = kwargs.pop("ext", "png")
        if not ext.startswith("."):
            ext = f".{ext}"

        results = {}

        # Helper to get channel safely
        def get_channel_data(source_mode, channel_name, default_val=None):
            # Handle RGB extraction
            if channel_name == "RGB":
                return img.convert("RGB")

            # Handle single channel extraction
            # Check if channel exists in image
            if channel_name in img.getbands():
                return img.getchannel(channel_name)

            # Handle fallback/default
            if default_val is not None:
                # Create constant image
                return Image.new("L", img.size, default_val)

            # If requesting R/G/B from L image, return the L image
            if source_mode == "L" and channel_name in "RGB":
                return img.copy()

            return None

        for src_chan, config in channel_config.items():
            suffix = config.get("suffix", f"_{src_chan}")
            invert = config.get("invert", False)
            default = config.get("default", None)

            extracted = get_channel_data(img.mode, src_chan, default)

            if extracted is None:
                # print(f"// Warning: Channel '{src_chan}' not found in image.")
                continue

            # Ensure L mode for single channels if they aren't already (getchannel returns L)
            if len(src_chan) == 1 and src_chan in "RGBA" and extracted.mode != "L":
                extracted = extracted.convert("L")

            # Invert if requested
            if invert:
                extracted = ImageOps.invert(extracted)

            if not save:
                results[src_chan] = extracted
                continue

            # Save
            out_path = os.path.join(output_dir, f"{base_name}{suffix}{ext}")
            extracted.save(out_path, format=output_format, **kwargs)
            results[src_chan] = out_path

        return results


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
