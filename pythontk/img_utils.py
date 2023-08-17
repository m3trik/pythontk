# !/usr/bin/python
# coding=utf-8
import sys

try:
    import numpy as np
except ImportError as error:
    print("{}\n # Error: {} #".format(__file__, error))
try:
    from PIL import Image, ImageChops, ImageDraw
except ImportError as error:
    print("{}\n # Error: {} #".format(__file__, error))
try:
    from PySide2 import QtWidgets
except ImportError as error:
    print("{}\n # Error: {} #".format(__file__, error))

# from this package:
from pythontk import core_utils
from pythontk import file_utils
from pythontk import iter_utils


class ImgUtils:
    """Helper methods for working with image file formats."""

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        sys.argv
    )  # return the existing QApplication object, or create a new one if none exists.

    map_types = {  # Get map type from filename suffix.
        "Base_Color": ("Base_Color", "BaseColor", "_BC"),
        "Roughness": ("Roughness", "Rough", "_R"),
        "Metallic": ("Metallic", "Metal", "Metalness", "_M"),
        "Ambient_Occlusion": (
            "Mixed_AO",
            "AmbientOcclusion",
            "Ambient_Occlusion",
            "_AO",
        ),
        "Normal": ("Normal", "Norm", "_N"),
        "Normal_DirectX": ("Normal_DirectX", "NormalDirectX", "NormalDX", "_NDX"),
        "Normal_OpenGL": ("Normal_OpenGL", "NormalOpenGL", "NormalGL", "_NGL"),
        "Height": ("Height", "High", "_H"),
        "Emissive": ("Emissive", "Emit", "_E"),
        "Diffuse": ("Diffuse", "_DF", "Diff", "Dif", "_D"),
        "Specular": ("Specular", "Spec", "_S"),
        "Glossiness": ("Glossiness", "Gloss", "Glos", "Glo", "_G"),
        "Displacement": ("Displacement", "_DP", "Displace", "Disp", "Dis", "_D"),
        "Refraction": ("Refraction", "IndexofRefraction", "_IOR"),
        "Reflection": ("Reflection", "_RF"),
        "Opacity": ("Opacity", "Transparancy", "Alpha", "Alpha_Mask", "_O"),
    }

    map_backgrounds = {  # Get default map backgrounds in RGBA format from map type.
        "Base_Color": (127, 127, 127, 255),
        "Roughness": (255, 255, 255, 255),
        "Metallic": (0, 0, 0, 255),
        "Ambient_Occlusion": (255, 255, 255, 255),
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
    }

    map_modes = {  # Get default map mode from map type.
        "Base_Color": "RGB",
        "Roughness": "L",
        "Metallic": "L",
        "Ambient_Occlusion": "L",
        "Normal": "RGB",
        "Normal_DirectX": "RGB",
        "Normal_OpenGL": "RGB",
        "Height": "I",  # I 32bit mode conversion from rgb not currently working.
        "Emissive": "L",
        "Diffuse": "RGB",
        "Specular": "L",
        "Glossiness": "L",
        "Displacement": "L",
        "Refraction": "L",
        "Reflection": "L",
    }

    bit_depth = {  # Get bit depth from mode.
        "1": 1,
        "L": 8,
        "P": 8,
        "RGB": 24,
        "RGBA": 32,
        "CMYK": 32,
        "YCbCr": 24,
        "LAB": 24,
        "HSV": 24,
        "I": 32,
        "F": 32,
        "I;16": 16,
        "I;16B": 16,
        "I;16L": 16,
        "I;16S": 16,
        "I;16BS": 16,
        "I;16LS": 16,
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
    def resize_image(cls, image, x, y):
        """Returns a resized copy of an image. It doesn't modify the original.

        Parameters:
                image (str/obj): An image or path to an image.
                x (int): Size in the x coordinate.
                y (int): Size in the y coordinate.

        Returns:
                (obj) new image of the given size.
        """
        im = cls.load_image(image) if isinstance(image, str) else image
        return im.resize((x, y), Image.Resampling.LANCZOS)

    @classmethod
    def save_image(cls, image, name):
        """
        Parameters:
                image (obj): PIL image object.
                name (str): Path + filename including extension. ie. new_image.png
        """
        im = cls.load_image(image) if isinstance(image, str) else image
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

    @staticmethod
    def get_image_files(
        file_types=["*.png", "*.jpg", "*.bmp", "*.tga", "*.tiff", "*.gif"]
    ):
        """Open a dialog prompt to choose image files of the given type(s).

        Parameters:
                file_types (str): The extensions of image types to include.

        Returns:
                (list)
        """
        file_types_list = iter_utils.IterUtils.make_iterable(file_types)

        files = QtWidgets.QFileDialog.getOpenFileNames(
            None,
            "Select one or more image files to open",
            "/home",
            f"Images ({' '.join(file_types_list)})",
        )[0]

        return files

    @staticmethod
    def get_image_dir():
        """Open a dialog prompt to choose a directory.

        Returns:
                (list)
        """
        image_dir = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select a directory containing image files", "/home"
        )

        return image_dir

    @classmethod
    def get_image_type_from_filename(cls, file, key=True):
        """
        Parameters:
                file (str): Image filename, fullpath, or map type suffix.
                key (bool): Get the corresponding key from the type in 'map_types'.
                        ie. Base_Color from <filename>_BC or BC. else: _BC from <filename>_BC.

        Returns:
                (str)
        """
        name = file_utils.FileUtils.format_path(file, "name")

        if key:
            result = next(
                (
                    k
                    for k, v in cls.map_types.items()
                    for i in v
                    if name.lower().endswith(i.lower())
                ),
                None,
            )
        else:
            result = next(
                (
                    i
                    for v in cls.map_types.values()
                    for i in v
                    if name.lower().endswith(i.lower())
                ),
                ("".join(name.split("_")[-1]) if "_" in name else None),
            )
        if result:
            return result

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
        return [f for f in files if cls.get_image_type_from_filename(f) in types]

    @classmethod
    def sort_images_by_type(cls, files):
        """Sort images files by map type.

        Parameters:
            files (list)(dict): filenames, fullpaths, or map type suffixes as the first element
                                of two element tuples or keys in a dictionary. ex. [('file', <image>)] or {'file': <image>}
        Returns:
            (dict): ex. {Height:[('img_height.png', <image>)]}
        """
        if not isinstance(files, (list, tuple, set)):
            files = files.items()

        sorted_images = {}
        for file, image in files:
            typ = cls.get_image_type_from_filename(file)
            if not typ:
                continue

            try:
                sorted_images[typ].append((file, image))
            except KeyError:
                sorted_images[typ] = [(file, image)]

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
                if cls.get_image_type_from_filename(i) in map_types
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
        typ = cls.get_image_type_from_filename(file)
        return any(
            (
                typ in cls.map_types["Normal_DirectX"],
                typ in cls.map_types["Normal_OpenGL"],
                typ in cls.map_types["Normal"],
            )
        )

    @classmethod
    def invert_channels(cls, image, channels="RGB"):
        """Invert RGB channels.

        Parameters:
            image (str/obj): An image or path to an image.
            channels (str): Specify which channels to invert.
                    valid: 'R','G','B' case insensitive.
        Returns:
            (obj): image.
        """
        im = cls.load_image(image) if isinstance(image, str) else image
        alpha = None

        try:
            r, g, b = im.split()
        except ValueError as error:
            r, g, b, a = im.split()

        r = ImageChops.invert(r) if "r" in channels.lower() else r
        g = ImageChops.invert(g) if "g" in channels.lower() else g
        b = ImageChops.invert(b) if "b" in channels.lower() else b

        return Image.merge(
            "RGB", (r, g, b)
        )  # if alpha else Image.merge('RGB', (red, green, blue))

    @classmethod
    def create_dx_from_gl(cls, file):
        """Create and save an OpenGL map using a given DirectX image.
        The new map will be saved next to the existing map.

        Parameters:
            file (str): The fullpath to a DirectX normal map file.

        Returns:
            (str) filepath of the new image.
        """
        inverted_image = cls.invert_channels(file, "g")

        output_dir = file_utils.FileUtils.format_path(file, "path")
        name = file_utils.FileUtils.format_path(file, "name")
        ext = file_utils.FileUtils.format_path(file, "ext")

        typ = cls.get_image_type_from_filename(file, key=False)
        try:
            index = cls.map_types["Normal_OpenGL"].index(typ)
            new_type = cls.map_types["Normal_DirectX"][index]
        except (IndexError, ValueError) as error:
            print("{} in create_dx_from_gl\n\t# Error: {} #".format(__file__, error))
            new_type = "Normal_DirectX"

        name = name.removesuffix(typ)
        filepath = "{}/{}{}.{}".format(output_dir, name, new_type, ext)
        inverted_image.save(filepath)

        return filepath

    @classmethod
    def create_gl_from_dx(cls, file):
        """Create and save an DirectX map using a given OpenGL image.
        The new map will be saved next to the existing map.

        Parameters:
            file (str): The fullpath to a OpenGL normal map file.

        Returns:
            (str): filepath of the new image.
        """
        inverted_image = cls.invert_channels(file, "g")

        output_dir = file_utils.FileUtils.format_path(file, "path")
        name = file_utils.FileUtils.format_path(file, "name")
        ext = file_utils.FileUtils.format_path(file, "ext")

        typ = cls.get_image_type_from_filename(file, key=False)
        try:
            index = cls.map_types["Normal_DirectX"].index(typ)
            new_type = cls.map_types["Normal_OpenGL"][index]
        except IndexError as error:
            print("{} in create_gl_from_dx\n\t# Error: {} #".format(__file__, error))
            new_type = "Normal_OpenGL"

        name = name.removesuffix(typ)
        filepath = "{}/{}{}.{}".format(output_dir, name, new_type, ext)
        inverted_image.save(filepath)

        return filepath

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

        im = cls.load_image(image) if isinstance(image, str) else image
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
        im = cls.load_image(image) if isinstance(image, str) else image
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
        im = cls.load_image(image) if isinstance(image, str) else image

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
        im = cls.load_image(image) if isinstance(image, str) else image

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
                mode (str): The image is converted to rgba for the operation specify the returned image mode. the original image mode will be returned if None is given. ex. 'RGBA' to return in rgba format.

        Returns:
                (obj) image.
        """
        im = cls.load_image(image) if isinstance(image, str) else image
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
        im = cls.load_image(image) if isinstance(image, str) else image

        factor = (259 * (level + 255)) / (255 * (259 - level))
        _contrast = lambda c: int(
            max(0, min(255, 128 + factor * (c - 128)))
        )  # make sure the contrast filter only return values within the range [0-255].

        return im.point(_contrast)

    @staticmethod
    def convert_rgb_to_gray(data):
        """Convert an RGB Image data array to grayscale.

        :Paramters:
                data (str/obj)(array) = An image, path to an image, or
                                Image data as numpy array.
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

        im = cls.load_image(image) if isinstance(image, str) else image
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
        im = cls.load_image(image) if isinstance(image, str) else image
        data = np.array(im)

        data = np.asarray(data, np.uint8)  # np.uint8(data / 256)
        return Image.fromarray(data)

    @staticmethod
    def are_identical(imageA, imageB):
        """Check if two images are the same.

        Parameters:
                imageA (str/obj): An image or path to an image.
                imageB (str/obj): An image or path to an image.

        Returns:
                (bool)
        """
        imA = Image.open(imageA) if (isinstance(imageA, str)) else imageA
        imB = Image.open(imageB) if (isinstance(imageB, str)) else imageB

        if np.sum(np.array(ImageChops.difference(imA, imB).getdata())) == 0:
            return True
        return False


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
