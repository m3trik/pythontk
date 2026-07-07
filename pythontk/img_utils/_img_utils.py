# !/usr/bin/python
# coding=utf-8
from __future__ import annotations

import os
import math
import re
import struct

# OpenCV reads this once, when its EXR codec first initializes (often at the
# first cv2 import). Set it here — pythontk is the ecosystem's EXR/HDR IO entry
# point and imports cv2 only lazily — so EXR/HDR IO works regardless of when a
# consumer first touches cv2. ``setdefault`` respects an explicit opt-out.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from collections import namedtuple
from contextlib import contextmanager
from typing import List, Tuple, Dict, Union, Any, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage

try:
    import numpy as np
except ImportError as e:
    print(f"# ImportError: {__file__}\n\t{e}")
try:
    from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageDraw
except ImportError as e:
    print(f"# ImportError: {__file__}\n\t{e}")
    Image = None  # type: ignore# from this package:

# From this package:
from pythontk.core_utils._core_utils import CoreUtils
from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.str_utils._str_utils import StrUtils


# Per-format IO capability. ``backend`` selects the library used to read/write:
#   "pil" — Pillow (the default).
#   "cv2" — OpenCV, for float formats Pillow cannot handle (EXR, HDR).
ImageFormat = namedtuple("ImageFormat", "read write backend")


class ImgUtils(HelpMixin):
    """Helper methods for working with image file formats."""

    # ------------------------------------------------------------------
    # Image-format capability table — single source of truth for which
    # extensions are textures, whether each can be read/written, and which
    # backend handles it. The ``recognized`` / ``readable`` / ``writable``
    # sets below are *derived* from this table so they cannot drift.
    # ------------------------------------------------------------------
    image_formats: Dict[str, ImageFormat] = {
        "png": ImageFormat(True, True, "pil"),
        "jpg": ImageFormat(True, True, "pil"),
        "jpeg": ImageFormat(True, True, "pil"),
        "bmp": ImageFormat(True, True, "pil"),
        "tga": ImageFormat(True, True, "pil"),
        "tiff": ImageFormat(True, True, "pil"),
        "gif": ImageFormat(True, True, "pil"),
        "dds": ImageFormat(True, True, "pil"),  # DXT-tier; BC7/BC6H unsupported by PIL's writer
        "exr": ImageFormat(True, True, "cv2"),
        "hdr": ImageFormat(True, True, "cv2"),
    }

    recognized = tuple(image_formats)  # discovery / file dialogs
    readable = tuple(e for e, f in image_formats.items() if f.read)  # load / scan
    writable = tuple(e for e, f in image_formats.items() if f.write)  # convert / output menus

    # Backward-compatible alias for the historical flat list (discovery surfaces).
    texture_file_types = list(recognized)

    # Plain photographic raster formats (dotted, lowercase) — the directory-scan
    # set shared by the photogrammetry/SfM ingest cluster (ExposureEqualizer /
    # ImageCurator / MaskGenerator). A deliberate semantic subset of
    # ``image_formats``: capture stills only, no float/texture formats.
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

    # DDS block-compression formats Pillow's writer handles directly (no external
    # tool). BC7 / BC6H are not in this set — they need a registered codec.
    PIL_DDS_PIXEL_FORMATS = ("DXT1", "DXT3", "DXT5", "BC5")

    # Optional external DDS codec for block formats Pillow can't write (BC7, BC6H).
    # Registered via :meth:`register_dds_codec`; ``None`` until an extension installs one.
    _dds_codec = None

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
        "LA": 16,
        "PA": 16,
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
        cls, image: Image.Image, target_mode: str, allow_compatible: bool = False
    ) -> Image.Image:
        """Converts image to target_mode. Strict by default.

        With allow_compatible=True, smaller "compatible" modes are preserved
        instead of being upcast (file-size efficiency):
            - Target RGB: keep P (Indexed) and L (Grayscale)
            - Target RGBA: keep P (Indexed)

        Strict (default) is recommended for textures consumed by DCCs / engines
        that read PNG palette-transparency as alpha (e.g. Maya's file node sets
        fileHasAlpha=True from PNG transparency info even when no pixel is
        actually transparent). Allowing palette mode for RGB targets leaks that
        signal into downstream FBX export and produces unexpected alphaMode=BLEND
        materials in glTF.

        Parameters:
            image (PIL.Image.Image): Input image.
            target_mode (str): Desired mode (RGB, RGBA, L).
            allow_compatible (bool): If True, keep smaller compatible modes
                (P, L) instead of upcasting. Default False (strict).

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
    def validate_image_integrity(filepath: str) -> Tuple[bool, str]:
        """Cheaply check that an image file is complete and decodable.

        Targets the files that crash native texture loaders: empty or
        truncated downloads (a partially-synced cloud file, an interrupted
        export) and stubs whose declared dimensions far exceed the bytes
        actually present. Pure-Python and zero-dependency — it does not fully
        decode pixels, so a clean result is a strong but not absolute
        guarantee, and an unrecognized structure is treated as ok (never a
        false reject).

        Coverage:
            * Radiance HDR (``.hdr`` / ``.pic``) — parses the resolution line
              and walks the RLE/flat scanlines; reports truncation precisely.
            * OpenEXR (``.exr``) — verifies the magic number and a sane
              minimum size (catches empty stubs / wrong-format files).
            * Other formats — only checks the file exists and is non-empty.

        Returns:
            (ok, detail): ``ok`` is False only when the file is provably bad;
            ``detail`` is a short reason ("" when ok).
        """
        fp = os.path.expandvars(filepath)
        try:
            size = os.path.getsize(fp)
        except OSError:
            return False, "file not found"
        if size == 0:
            return False, "file is empty"

        ext = os.path.splitext(fp)[1].lower().lstrip(".")
        try:
            if ext in ("hdr", "pic"):
                return ImgUtils._validate_radiance_hdr(fp)
            if ext == "exr":
                return ImgUtils._validate_exr(fp, size)
        except Exception as e:  # validation must never raise on the caller
            return True, f"unvalidated ({e})"
        return True, ""

    @staticmethod
    def _validate_radiance_hdr(fp: str) -> Tuple[bool, str]:
        """Walk a Radiance HDR's scanlines to detect truncation."""
        with open(fp, "rb") as f:
            data = f.read()
        if not data.startswith(b"#?"):  # #?RADIANCE / #?RGBE
            return True, ""  # not the expected format; don't block
        nl = data.find(b"\n\n")  # header ends at the first blank line
        if nl < 0:
            return False, "incomplete header"
        p = nl + 2
        eol = data.find(b"\n", p)
        if eol < 0:
            return False, "missing resolution line"
        res = data[p:eol].split()  # e.g. b"-Y 4096 +X 8192"
        if len(res) != 4:
            return True, ""  # nonstandard orientation; skip the strict check
        try:
            height, width = int(res[1]), int(res[3])
        except ValueError:
            return True, ""
        off, n = eol + 1, len(data)

        # New-style adaptive RLE: each scanline is 0x02 0x02 <hi> <lo> then four
        # run-length-encoded channels. Old/flat RGBE has no markers.
        if 8 <= width <= 0x7FFF and off + 4 <= n and data[off] == 2 and data[off + 1] == 2:
            rows = 0
            while rows < height and off + 4 <= n:
                if data[off] != 2 or data[off + 1] != 2:
                    break
                if ((data[off + 2] << 8) | data[off + 3]) != width:
                    break
                off += 4
                truncated = False
                for _channel in range(4):
                    x = 0
                    while x < width:
                        if off >= n:
                            truncated = True
                            break
                        run = data[off]
                        off += 1
                        if run > 128:  # a run of (run-128) identical bytes
                            off += 1
                            x += run - 128
                        else:  # (run) literal bytes
                            off += run
                            x += run
                    if truncated or off > n:
                        truncated = True
                        break
                if truncated:
                    break
                rows += 1
            if rows < height:
                return False, f"truncated: {rows}/{height} scanlines"
            return True, ""

        # Flat RGBE fallback: 4 bytes/pixel, no length markers.
        expected = width * height * 4
        if (n - off) < expected:
            return False, f"truncated: {n - off}/{expected} pixel bytes"
        return True, ""

    @staticmethod
    def _validate_exr(fp: str, size: int) -> Tuple[bool, str]:
        """Check an OpenEXR magic number + a sane minimum size."""
        with open(fp, "rb") as f:
            magic = f.read(4)
        if magic != b"\x76\x2f\x31\x01":
            return False, "not an OpenEXR file (bad magic)"
        if size < 64:  # header alone is larger than this
            return False, "EXR too small to be valid"
        return True, ""

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
    def register_dds_codec(cls, codec) -> None:
        """Register an external DDS codec for block formats Pillow can't write.

        Pillow handles :attr:`PIL_DDS_PIXEL_FORMATS` (DXT/BC5) natively; BC7/BC6H
        need an external tool (e.g. texconv/nvtt). An extension installs one here.

        Parameters:
            codec: ``callable(im: PIL.Image, name: str, compression: str) -> None``
                that writes *im* to *name* using the *compression* block format.
        """
        cls._dds_codec = codec

    @classmethod
    def save_image(
        cls,
        image: Union[str, Image.Image],
        name: str,
        mode: str = None,
        bit_depth: int = None,
        compression: str = None,
        **kwargs,
    ):
        """Save an image to ``name``, dispatching on the file extension.

        Routing follows :attr:`image_formats`: most formats use Pillow; float
        formats (EXR, HDR) use OpenCV via :meth:`_save_via_cv2`. A recognized but
        read-only format raises ``ValueError``.

        Parameters:
            image (str | PIL.Image.Image): Image object or file path.
            name (str): Output path including filename and extension (e.g., "output.png").
            mode (str, optional): Converts the image to the specified mode before saving (e.g., "RGB", "L").
            bit_depth (int, optional): Target per-channel bit depth. 16 writes a
                16-bit PNG/TIFF (8-bit sources are promoted); other containers fall
                back to 8-bit with a warning. 32-bit float is the EXR/HDR path.
            compression (str, optional): DDS block format (e.g. "DXT5", "BC7").
                Only honored for ``.dds`` — see :meth:`_save_dds_compressed`.
            **kwargs: Additional arguments forwarded to PIL.Image.save (e.g.,
                optimize=True, compress_level=9). Ignored for OpenCV-backed formats.
        """
        im = cls.ensure_image(image, mode)  # Now allows optional mode conversion

        ext = os.path.splitext(name)[1].lstrip(".").lower()
        fmt = cls.image_formats.get(ext)

        if fmt is not None and not fmt.write:
            raise ValueError(
                f"Cannot save {name!r}: {ext!r} is a read-only format in ImgUtils."
            )

        # GPU block-compressed DDS (DXT/BC5 via Pillow; BC7/BC6H via registered codec).
        if ext == "dds" and compression:
            cls._save_dds_compressed(im, name, compression)
            return

        # Route float formats (EXR, HDR) through OpenCV — Pillow cannot write them.
        if fmt is not None and fmt.backend == "cv2":
            cls._save_via_cv2(im, name)
            return

        # 16-bit precision for PIL container formats (PNG/TIFF). Returns False when
        # the container can't hold it → fall through to the 8-bit path.
        if bit_depth and int(bit_depth) >= 16 and cls._save_high_bit_depth(
            im, name, int(bit_depth)
        ):
            return

        # Auto-convert RGBA to RGB if saving as JPEG to prevent OSError
        if ext in ("jpg", "jpeg") and im.mode == "RGBA":
            im = im.convert("RGB")

        im.save(name, **kwargs)

    @classmethod
    def _save_dds_compressed(cls, im: "Image.Image", name: str, compression: str) -> None:
        """Write *im* to a block-compressed ``.dds``.

        DXT/BC5 use Pillow's ``pixel_format``; BC7/BC6H route to a codec registered
        via :meth:`register_dds_codec`, raising a clear error if none is installed.
        """
        comp = compression.upper()
        if comp in cls.PIL_DDS_PIXEL_FORMATS:
            # BC5 is a two-channel format and only accepts RGB; DXT* want RGB(A).
            if comp == "BC5":
                im = im.convert("RGB") if im.mode != "RGB" else im
            elif im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA")
            im.save(name, pixel_format=comp)
            return

        if cls._dds_codec is not None:
            cls._dds_codec(im, name, comp)
            return

        raise ValueError(
            f"DDS compression {comp!r} requires an external codec. Pillow writes "
            f"{cls.PIL_DDS_PIXEL_FORMATS}; for BC7/BC6H install the DDS codec "
            f"extension and register it via ImgUtils.register_dds_codec()."
        )

    @staticmethod
    def _save_high_bit_depth(im: "Image.Image", name: str, bit_depth: int) -> bool:
        """Write *im* at 16-bit. Returns True when handled, False when the request
        can't be honored (unsupported depth or container) — the caller then falls
        back to an 8-bit save. Either way the degrade is announced, never silent.

        Grayscale uses Pillow's ``I;16``; RGB(A) routes through OpenCV ``uint16``.
        8-bit sources are promoted (value*257); existing 16-bit data is preserved.
        """
        if bit_depth != 16:  # only 16 is supported here; 32-bit float = EXR/HDR.
            print(f"# ImgUtils: {bit_depth}-bit unsupported for {name}; saving as 8-bit.")
            return False

        ext = os.path.splitext(name)[1].lstrip(".").lower()
        if ext not in ("png", "tiff", "tif"):
            print(f"# ImgUtils: '{ext}' cannot store 16-bit; saving {name} as 8-bit.")
            return False

        if im.mode in ("L", "P", "1", "I", "I;16"):
            arr = np.asarray(im.convert("I"), dtype=np.int64)
            if im.mode in ("L", "P", "1"):  # promote 8-bit range to 16-bit
                arr = arr * 257
            arr = np.clip(arr, 0, 65535).astype(np.uint16)
            Image.fromarray(arr).save(name)  # uint16 array → "I;16" natively
            return True

        # RGB / RGBA — Pillow has no 16-bit colour mode, so use OpenCV uint16.
        try:
            import cv2
        except ImportError:
            return False
        rgb = im.convert("RGBA") if im.mode == "RGBA" else im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint16) * 257
        code = cv2.COLOR_RGBA2BGRA if rgb.mode == "RGBA" else cv2.COLOR_RGB2BGR
        cv2.imwrite(name, cv2.cvtColor(arr, code))
        return True

    @staticmethod
    def _save_via_cv2(im: "Image.Image", name: str) -> None:
        """Write a PIL image to a float format (EXR, HDR) via OpenCV.

        Pillow cannot encode these. The source PIL image is 8-bit, so values
        are normalized to 0-1 float32 (the inverse of :meth:`_load_via_cv2`).
        OpenEXR is enabled at module import (``OPENCV_IO_ENABLE_OPENEXR``).
        """
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                f"OpenCV (cv2) is required to save '{os.path.splitext(name)[1]}' files."
            ) from e

        img_np = np.array(im)
        if im.mode == "RGB":
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif im.mode == "RGBA":
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA)
        # "L" (grayscale) passes through unchanged.

        img_np = img_np.astype(np.float32) / 255.0
        cv2.imwrite(name, img_np)

    @classmethod
    def load_image(cls, filepath):
        """Load an image and return a PIL copy, dispatching on the file extension.

        Float formats (EXR, HDR) are read via OpenCV (:meth:`_load_via_cv2`) and
        returned as an 8-bit PIL image (lossy — preview-grade); all others use
        Pillow directly.

        Parameters:
            filepath (str): The full path to the image file.

        Returns:
            (PIL.Image.Image) A copy of the loaded image object.
        """
        cls.assert_pathlike(filepath, "filepath")

        ext = os.path.splitext(str(filepath))[1].lstrip(".").lower()
        fmt = cls.image_formats.get(ext)
        if fmt is not None and fmt.backend == "cv2":
            return cls._load_via_cv2(str(filepath))

        with Image.open(filepath) as im:
            return im.copy()

    @staticmethod
    def _load_via_cv2(filepath: str) -> "Image.Image":
        """Read a float format (EXR, HDR) via OpenCV and return an 8-bit PIL image.

        Values are clipped to 0-1 and scaled to 8-bit, so the result is
        preview-grade — lossy for true HDR data. Consumers needing float
        precision (e.g. lightmap baking) should read via cv2 directly.
        OpenEXR is enabled at module import (``OPENCV_IO_ENABLE_OPENEXR``).
        """
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                f"OpenCV (cv2) is required to read '{os.path.splitext(filepath)[1]}' files."
            ) from e

        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise OSError(f"OpenCV could not read image: {filepath}")

        # Float HDR data → clip to 0-1 and scale to 8-bit for the PIL contract.
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        if img.ndim == 2:
            return Image.fromarray(img, mode="L")
        if img.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA), mode="RGBA")
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mode="RGB")

    @classmethod
    def list_image_files(cls, directory, exts=None, full_paths=False):
        """Sorted image file names in a directory (non-recursive).

        Parameters:
            directory (str): Directory to scan.
            exts (str/tuple, optional): Dotted extension(s) to accept, any
                    case; a bare string is treated as a single extension.
                    Defaults to :attr:`IMAGE_EXTS` (plain photographic formats).
            full_paths (bool): Return joined ``directory/name`` paths instead
                    of bare file names.

        Returns:
            (list) Sorted file names, or joined paths when ``full_paths=True``.
        """
        if exts is None:
            exts = cls.IMAGE_EXTS
        else:
            # A bare string would tuple-ize into single characters and
            # silently match on them; names are compared lowercased.
            if isinstance(exts, str):
                exts = (exts,)
            exts = tuple(e.lower() for e in exts)
        names = sorted(f for f in os.listdir(directory) if f.lower().endswith(exts))
        if full_paths:
            return [os.path.join(directory, f) for f in names]
        return names

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
            inc = [f"*.{ext}" for ext in cls.readable]

        cls.assert_pathlike(directory, "directory")

        images = {}
        for f in FileUtils.get_dir_contents(
            directory, "filepath", inc_files=inc, exc_files=exc
        ):
            im = cls.load_image(f)
            images[f] = im

        return images

    @staticmethod
    def _image_size_from_header(image_path: str) -> Optional[Tuple[int, int]]:
        """``(width, height)`` from a JPEG/PNG header using only the stdlib.

        Reads the dimensions out of the file header — no PIL, numpy, or cv2 — so
        it works in dependency-light interpreters (e.g. Metashape's bundled
        Python). ``None`` for an unrecognized or truncated file.
        """
        try:
            with open(image_path, "rb") as f:
                head = f.read(24)
                # PNG: 8-byte signature, then IHDR chunk (width,height big-endian u32).
                if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
                    w, h = struct.unpack(">II", head[16:24])
                    return int(w), int(h)
                # JPEG: SOI 0xFFD8, then scan segments for a Start-Of-Frame marker.
                if head[:2] == b"\xff\xd8":
                    f.seek(2)
                    while True:
                        b = f.read(1)
                        if not b:
                            return None
                        if b != b"\xff":
                            continue
                        marker = f.read(1)
                        while marker == b"\xff":          # skip fill bytes
                            marker = f.read(1)
                        if not marker:
                            return None
                        m = marker[0]
                        if 0xD0 <= m <= 0xD9:             # RSTn / SOI / EOI: no length
                            continue
                        lb = f.read(2)
                        if len(lb) < 2:
                            return None
                        seglen = struct.unpack(">H", lb)[0]
                        if seglen < 2:  # invalid: length includes its own 2 bytes
                            return None  # guards against a backward-seek infinite loop
                        # SOF0..SOF15 carry the frame size (excl. DHT/JPG/DAC: C4/C8/CC).
                        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                            f.read(1)                     # sample precision
                            hw = f.read(4)
                            if len(hw) < 4:
                                return None
                            h, w = struct.unpack(">HH", hw)
                            return int(w), int(h)
                        f.seek(seglen - 2, 1)             # skip to next segment
        except Exception:
            return None
        return None

    @staticmethod
    def get_image_size(image_path: str) -> Optional[Tuple[int, int]]:
        """``(width, height)`` of an image, read as cheaply as possible.

        Parses the JPEG/PNG header with the **stdlib only** (no PIL/numpy/cv2),
        so it works in dependency-light interpreters such as Metashape's bundled
        Python; falls back to PIL for other formats when available. ``None`` if
        the size can't be determined. Use this (not :meth:`get_image_info`) when
        you need only the dimensions and can't assume PIL is installed.
        """
        size = ImgUtils._image_size_from_header(image_path)
        if size:
            return size
        if Image is not None:
            try:
                with Image.open(image_path) as im:
                    return int(im.size[0]), int(im.size[1])
            except Exception:
                pass
        return None

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

        if np.sum(np.array(ImageChops.difference(imA, imB))) == 0:
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
    def format_bit_depth(cls, mode_or_image) -> str:
        """Format bit depth as e.g. '24bit (8x3)' — total bits with (per-channel x channels) breakdown."""
        mode = mode_or_image.mode if hasattr(mode_or_image, "mode") else mode_or_image
        total = cls.bit_depth.get(mode, 8)
        try:
            channels = Image.getmodebands(mode) if Image else 1
        except (KeyError, ValueError):
            channels = 1
        bpc = total // channels if channels else total
        return f"{total}bit ({bpc}x{channels})"

    @classmethod
    def set_bit_depth(cls, image, map_type: str, allow_palette: bool = False) -> object:
        """Sets the bit depth and image mode of an image according to the map type.

        Parameters:
            image (PIL.Image.Image): The input image.
            map_type (str): The type of the map to determine the mode and bit depth.
            allow_palette (bool): If True, palette (P) and grayscale (L) inputs
                may be preserved when the target mode is RGB/RGBA, trading
                fidelity for smaller file size. Default False (strict) — palette
                images get fully upcast, dropping any palette-transparency info
                that would otherwise be read as alpha by Maya / FBX exporters.

        Returns:
            PIL.Image.Image: The image with the specified or recommended bit depth and mode.
        """
        # Determine the target mode based on map type. MapRegistry is a
        # SingletonMixin so the import + lookup are cheap; the deferred
        # import keeps _img_utils.py free of module-load-time coupling to
        # the map cluster.
        from pythontk.img_utils.map_registry import MapRegistry

        map_modes = MapRegistry().get_map_modes()
        if map_type in map_modes:
            target_mode = map_modes[map_type]
            image = cls.enforce_mode(image, target_mode, allow_compatible=allow_palette)

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
    def swizzle_channels(cls, image, mapping):
        """Reorder, duplicate, or constant-fill an image's channels.

        A general channel-remap primitive: each destination channel pulls from
        a chosen source channel (or a constant), so it covers swaps (R↔B),
        broadcasts (red → grayscale), and alpha fills. Pairs with
        :meth:`invert_channels` for full per-channel control.

        Parameters:
            image (str/PIL.Image.Image): An image or path to an image.
            mapping (str/dict): The channel remap.
                - **str**: destination channels in ``RGBA`` order, each
                  character naming the *source* to pull into that slot — e.g.
                  ``"BGRA"`` swaps red and blue, ``"RRR"`` broadcasts red. The
                  length (1-4) sets the number of output channels
                  (1→``L``, 2→``LA``, 3→``RGB``, 4→``RGBA``).
                - **dict** ``{dest: source}``: only the listed destinations are
                  remapped; the rest keep their original channel. Output is
                  ``RGB``, gaining an ``A`` channel when the input already has
                  one *or* the mapping names an ``A`` destination — so a dict
                  can add alpha to an RGB image (a grayscale input is promoted
                  to ``RGB``).
                Sources are case-insensitive ``R``/``G``/``B``/``A`` or the
                constants ``"0"`` (black) / ``"1"`` (white). A source the input
                lacks (e.g. ``A`` on an RGB image) resolves to white.

        Returns:
            PIL.Image.Image: The remapped image.
        """
        im = cls.ensure_image(image)
        rgba = im.convert("RGBA")
        bands = dict(zip("RGBA", rgba.split()))

        def resolve(token):
            token = str(token).strip().upper()
            if token in ("0", "1"):
                return Image.new("L", rgba.size, 0 if token == "0" else 255)
            if token in bands:
                return bands[token]
            raise ValueError(
                f"swizzle_channels: invalid source '{token}'; expected one of "
                "R, G, B, A, 0, 1."
            )

        if isinstance(mapping, str):
            order = mapping.strip()
            if not 1 <= len(order) <= 4:
                raise ValueError(
                    "swizzle_channels: string mapping must be 1-4 characters."
                )
            out_bands = [resolve(c) for c in order]
            if len(out_bands) == 1:
                return out_bands[0]
            out_mode = {2: "LA", 3: "RGB", 4: "RGBA"}[len(out_bands)]
            return Image.merge(out_mode, tuple(out_bands))

        # dict mapping — output RGB, gaining alpha when the input already has
        # one or the mapping explicitly addresses the ``A`` destination.
        remap = {str(k).strip().upper(): v for k, v in mapping.items()}
        has_alpha = "A" in remap or "A" in im.getbands()
        dest_order = "RGBA" if has_alpha else "RGB"
        out_bands = [resolve(remap.get(dest, dest)) for dest in dest_order]
        return Image.merge(dest_order, tuple(out_bands))

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
        im = im.convert("RGBA")
        width, height = im.size
        data = np.array(im)  # shape (height, width, 4) — rows first.

        r1, g1, b1, a1 = mask if len(mask) == 4 else tuple(mask) + (None,)

        r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

        matched = (
            ((r == r1) & (g == g1) & (b == b1) & (a == a1))
            if len(mask) == 4
            else ((r == r1) & (g == g1) & (b == b1))
        )

        data[~matched] = foreground
        data[matched] = background

        # Force the corners to background color:
        data[0, 0] = background  # top left
        data[0, width - 1] = background  # top right
        data[height - 1, 0] = background  # bottom left
        data[height - 1, width - 1] = background  # bottom right

        return Image.fromarray(data).convert("L")

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

    @classmethod
    def gaussian_blur(
        cls,
        image: Union[str, "Image.Image", "np.ndarray"],
        radius: float = 2.0,
        channel: Optional[str] = None,
    ) -> Union["Image.Image", "np.ndarray"]:
        """Apply a Gaussian blur to an image or 2D/3D numpy array.

        Numpy in → numpy out (same dtype / shape); PIL in → PIL out. Choose
        based on what the caller already holds; no implicit conversion.

        Parameters:
            image: File path, PIL Image, or numpy array (HxW or HxWxC, uint8/float).
            radius: Blur radius (PIL units; ~sigma in pixels). 0 returns a copy.
            channel: For RGBA inputs, optionally restrict the blur to a single
                channel (``"R"``, ``"G"``, ``"B"``, or ``"A"``) and leave the
                others untouched. Useful for softening only the alpha of a cutout.

        Returns:
            Blurred image, in the same form as the input.
        """
        if radius <= 0:
            if isinstance(image, np.ndarray):
                return image.copy()
            im = cls.ensure_image(image)
            return im.copy()

        # Numpy path
        if isinstance(image, np.ndarray):
            return cls._gaussian_blur_array(image, radius, channel)

        # PIL path
        im = cls.ensure_image(image)
        if channel and im.mode in ("RGBA", "LA"):
            bands = list(im.split())
            idx = {"R": 0, "G": 1, "B": 2, "A": 3}.get(channel.upper())
            if idx is None or idx >= len(bands):
                raise ValueError(
                    f"Channel {channel!r} not present in image mode {im.mode!r}"
                )
            bands[idx] = bands[idx].filter(ImageFilter.GaussianBlur(radius=radius))
            return Image.merge(im.mode, bands)
        return im.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def _gaussian_blur_array(
        arr: "np.ndarray", radius: float, channel: Optional[str]
    ) -> "np.ndarray":
        """Numpy-array blur. Uses PIL when present (avoids pulling in scipy/cv2); falls back to a
        pure-numpy separable Gaussian when PIL is unavailable, so dependency-light callers (e.g.
        ``rasterize_silhouette`` under Blender's PIL-less Python) keep working."""
        if Image is None:
            return ImgUtils._gaussian_blur_array_numpy(arr, radius, channel)
        # 2D grayscale
        if arr.ndim == 2:
            src = Image.fromarray(arr if arr.dtype == np.uint8 else arr.astype(np.uint8))
            blurred = src.filter(ImageFilter.GaussianBlur(radius=radius))
            out = np.asarray(blurred)
            return out.astype(arr.dtype, copy=False)

        # 3D: HxWxC
        if arr.ndim == 3:
            chans = arr.shape[2]
            mode = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}.get(chans)
            if mode is None:
                raise ValueError(f"Unsupported channel count: {chans}")
            src = Image.fromarray(
                arr if arr.dtype == np.uint8 else arr.astype(np.uint8), mode=mode
            )
            if channel and mode in ("RGBA", "LA"):
                bands = list(src.split())
                idx = {"R": 0, "G": 1, "B": 2, "A": 3}.get(channel.upper())
                if idx is None or idx >= len(bands):
                    raise ValueError(
                        f"Channel {channel!r} not present in mode {mode!r}"
                    )
                bands[idx] = bands[idx].filter(ImageFilter.GaussianBlur(radius=radius))
                blurred = Image.merge(mode, bands)
            else:
                blurred = src.filter(ImageFilter.GaussianBlur(radius=radius))
            out = np.asarray(blurred)
            return out.astype(arr.dtype, copy=False)

        raise ValueError(f"Unsupported array shape: {arr.shape}")

    @staticmethod
    def _gaussian_blur_array_numpy(
        arr: "np.ndarray", radius: float, channel: Optional[str]
    ) -> "np.ndarray":
        """Pure-numpy separable Gaussian blur (PIL-free fallback for :meth:`_gaussian_blur_array`).

        Treats ``radius`` as the kernel std-dev (sigma), matching PIL's ``GaussianBlur(radius=…)``,
        and pads with reflection so edges don't darken. Returns the input dtype (uint8 inputs are
        rounded). For a 3D RGBA/LA array, ``channel`` (``"R"``/``"G"``/``"B"``/``"A"``) restricts the
        blur to one channel, mirroring the PIL path."""
        sigma = max(float(radius), 1e-6)
        rad = max(1, int(round(3.0 * sigma)))
        x = np.arange(-rad, rad + 1, dtype=np.float64)
        k = np.exp(-(x * x) / (2.0 * sigma * sigma))
        k /= k.sum()

        def blur2d(a2d: "np.ndarray") -> "np.ndarray":
            a2d = a2d.astype(np.float64, copy=False)
            pad = len(k) // 2
            ap = np.pad(a2d, ((0, 0), (pad, pad)), mode="reflect")
            a2d = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, ap)
            ap = np.pad(a2d, ((pad, pad), (0, 0)), mode="reflect")
            return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, ap)

        def cast(out: "np.ndarray") -> "np.ndarray":
            if arr.dtype == np.uint8:
                return (out + 0.5).clip(0, 255).astype(np.uint8)
            return out.astype(arr.dtype, copy=False)

        if arr.ndim == 2:
            return cast(blur2d(arr))
        if arr.ndim == 3:
            chans = arr.shape[2]
            idx = {"R": 0, "G": 1, "B": 2, "A": 3}.get(channel.upper()) if channel else None
            targets = [idx] if (idx is not None and idx < chans) else range(chans)
            out = arr.astype(np.float64, copy=True)
            for c in targets:
                out[:, :, c] = blur2d(arr[:, :, c])
            return cast(out)
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    @staticmethod
    def dilate_image(
        image: "np.ndarray",
        mask: Optional["np.ndarray"] = None,
        iterations: int = -1,
        connectivity: int = 8,
    ) -> "np.ndarray":
        """Extend valid pixels outward into empty (background) regions.

        Texture "edge padding" / "dilation": fills the gutter around UV
        islands so bilinear filtering and mip generation never pull
        background color across an island seam. Pure numpy (works on HDR
        float data); no PIL/cv2 dependency.

        Each pass assigns every still-empty pixel adjacent to filled pixels
        the average of its filled neighbors, then marks it filled.
        ``iterations=-1`` repeats until fully filled.

        Parameters:
            image: HxW or HxWxC numpy array. Not modified -- a copy is returned.
            mask: HxW bool/numeric "valid" mask (truthy = keep & spread from).
                Defaults to "any channel > 0". For baked maps pass the explicit
                coverage/alpha mask -- a luminance heuristic wrongly treats
                dark-but-valid texels (shadowed contact, near-black albedo) as
                empty and overwrites them.
            iterations: Max passes (≈ gutter width in px). -1 = until filled.
            connectivity: 4 or 8 neighbor connectivity.

        Returns:
            Image with empty regions filled; same shape and dtype as input.
        """
        arr = np.asarray(image)
        out = arr.astype(np.float32, copy=True)
        squeeze = out.ndim == 2
        if squeeze:
            out = out[..., None]
        h, w, _ = out.shape

        if mask is None:
            valid = (out > 0).any(axis=2)
        else:
            valid = np.asarray(mask).astype(bool)
            if valid.shape != (h, w):
                raise ValueError(f"mask shape {valid.shape} != image {(h, w)}")
        out[~valid] = 0.0  # empties must not contribute color until filled

        if connectivity == 8:
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                       (0, 1), (1, -1), (1, 0), (1, 1)]
        elif connectivity == 4:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            raise ValueError("connectivity must be 4 or 8")

        def shift(a: "np.ndarray", dy: int, dx: int) -> "np.ndarray":
            s = np.zeros_like(a)
            ys, yd = slice(max(dy, 0), h + min(dy, 0)), slice(max(-dy, 0), h + min(-dy, 0))
            xs, xd = slice(max(dx, 0), w + min(dx, 0)), slice(max(-dx, 0), w + min(-dx, 0))
            s[yd, xd] = a[ys, xs]
            return s

        it = 0
        while not valid.all() and (iterations < 0 or it < iterations):
            color_acc = np.zeros_like(out)
            count_acc = np.zeros((h, w), dtype=np.float32)
            vf = valid.astype(np.float32)
            # `out` is already zero at every invalid pixel (and stays so until
            # the pass that fills it also flips it valid), so out == out*vf --
            # accumulate `out` directly; only the count needs the validity mask.
            for dy, dx in offsets:
                color_acc += shift(out, dy, dx)
                count_acc += shift(vf, dy, dx)
            fillable = (~valid) & (count_acc > 0)
            if not fillable.any():
                break  # remaining empties are unreachable from any valid pixel
            out[fillable] = color_acc[fillable] / count_acc[fillable][..., None]
            valid[fillable] = True
            it += 1

        if squeeze:
            out = out[..., 0]
        return out.astype(arr.dtype, copy=False)

    @staticmethod
    def compute_atlas_layout(
        weights: Sequence[float],
        *,
        rows: Optional[int] = None,
    ) -> List[Tuple[float, float, float, float]]:
        """Lay out N weighted items as non-overlapping rects tiling the unit square.

        Turns per-item importance *weights* into ``(scaleX, scaleY, offsetX,
        offsetY)`` rects in normalized [0, 1] texture space -- exactly the form a
        texture atlas needs, and exactly Unity's ``Renderer.lightmapScaleOffset``
        convention: ``uv' = uv * (scaleX, scaleY) + (offsetX, offsetY)`` places
        an item's 0-1 UVs into its sub-rect. Each item's rect *area* is
        proportional to its weight, so a large object can be given more atlas
        texels than a small one.

        Shelf packing: items are balanced into ``rows`` shelves (longest-
        processing-time, to keep aspect ratios sane); each shelf's height is its
        share of the total weight, and within a shelf each item's width is its
        share of that shelf's weight. Rows cover the full height and items cover
        each row's full width, so the rects tile [0, 1]^2 with no gaps or
        overlaps. Pure Python -- no numpy/PIL.

        Parameters:
            weights: One non-negative importance value per item, in caller order.
                All-zero (or empty-after-clamp) weights fall back to equal
                shares; negative values are clamped to 0.
            rows: Number of shelves. Defaults to ``round(sqrt(n))`` for roughly
                square cells. Clamped to ``1..n``.

        Returns:
            One ``(scaleX, scaleY, offsetX, offsetY)`` tuple per input weight, in
            the same order. ``[]`` for no items; ``[(1, 1, 0, 0)]`` for one.
        """
        n = len(weights)
        if n == 0:
            return []
        if n == 1:
            return [(1.0, 1.0, 0.0, 0.0)]

        w = [max(float(x), 0.0) for x in weights]
        total = sum(w)
        if total <= 0.0:  # no information -> equal shares
            w = [1.0] * n
            total = float(n)

        r = round(math.sqrt(n)) if rows is None else int(rows)
        r = max(1, min(r, n))

        # Balance items across `r` shelves by weight (LPT): assign the heaviest
        # remaining item to the currently-lightest shelf. Keeps shelf weight-sums
        # (and thus heights) even, which keeps rect aspect ratios reasonable.
        # With r <= n and zero-weight shelves being "lightest", the first r items
        # seed distinct shelves, so no shelf is ever left empty.
        shelves: List[List[int]] = [[] for _ in range(r)]
        shelf_w = [0.0] * r
        for i in sorted(range(n), key=lambda k: w[k], reverse=True):
            j = min(range(r), key=lambda k: shelf_w[k])
            shelves[j].append(i)
            shelf_w[j] += w[i]

        rects: List[Tuple[float, float, float, float]] = [(1.0, 1.0, 0.0, 0.0)] * n
        oy = 0.0
        for j, shelf in enumerate(shelves):
            sh = shelf_w[j] / total  # shelf height == its weight share
            ox = 0.0
            for i in sorted(shelf):  # input order within the row, for stable output
                sw = w[i] / shelf_w[j] if shelf_w[j] > 0 else 1.0 / len(shelf)
                rects[i] = (sw, sh, ox, oy)
                ox += sw
            oy += sh
        return rects

    @classmethod
    def assemble_atlas(
        cls,
        images: Sequence["np.ndarray"],
        rects: Sequence[Tuple[float, float, float, float]],
        size: Union[int, Tuple[int, int]],
        *,
        background: float = 0.0,
    ) -> "np.ndarray":
        """Composite per-item images into one atlas at normalized ``scaleOffset`` rects.

        Pairs each image with its ``(scaleX, scaleY, offsetX, offsetY)`` rect (from
        :meth:`compute_atlas_layout`) and resizes it into that sub-rectangle of a
        single atlas canvas. The rect is in UV space (origin bottom-left -- the
        convention :meth:`compute_atlas_layout` and Unity's ``lightmapScaleOffset``
        use), so placement applies the standard vertical flip into image-row space
        (row 0 == top == v 1): an item later bound with the *same* scaleOffset
        samples exactly the pixels written here.

        HDR-safe (works in float32, returns the input dtype). Requires cv2 for the
        resize -- guard call sites / tests with ``cv2`` availability.

        Parameters:
            images: One HxW or HxWxC array per item; all must share channel count.
                The atlas inherits the first image's channels and dtype.
            rects: One ``(sx, sy, ox, oy)`` per image -- same order and length.
            size: Atlas pixel size -- ``int`` for square, or ``(width, height)``.
            background: Fill for any uncovered atlas texels (default 0).

        Returns:
            The atlas as an HxWxC (or HxW) array, dtype matching ``images[0]``.
        """
        if len(images) != len(rects):
            raise ValueError(
                f"images ({len(images)}) and rects ({len(rects)}) length differ"
            )
        if not images:
            raise ValueError("assemble_atlas requires at least one image")

        import cv2

        w_px, h_px = (size, size) if isinstance(size, int) else size
        first = np.asarray(images[0])
        dtype = first.dtype
        squeeze = first.ndim == 2
        channels = 1 if squeeze else first.shape[2]
        canvas = np.full((h_px, w_px, channels), background, dtype=np.float32)

        for img, (sx, sy, ox, oy) in zip(images, rects):
            col0 = int(round(ox * w_px))
            col1 = int(round((ox + sx) * w_px))
            # UV v is bottom-up; image rows are top-down -> flip vertically.
            row0 = int(round((1.0 - (oy + sy)) * h_px))
            row1 = int(round((1.0 - oy) * h_px))
            tw, th = col1 - col0, row1 - row0
            if tw <= 0 or th <= 0:
                continue  # degenerate (e.g. zero-weight) rect -- nothing to place

            a = np.asarray(img, dtype=np.float32)
            if a.ndim == 2:
                a = a[..., None]
            if a.shape[2] != channels:
                raise ValueError(
                    f"image channel count {a.shape[2]} != atlas {channels}"
                )
            # INTER_AREA is the correct downscale kernel (the common atlas case);
            # use bilinear only when a rect happens to be larger than its source.
            interp = (
                cv2.INTER_AREA
                if th <= a.shape[0] and tw <= a.shape[1]
                else cv2.INTER_LINEAR
            )
            resized = cv2.resize(a, (tw, th), interpolation=interp)
            if resized.ndim == 2:
                resized = resized[..., None]
            canvas[row0:row1, col0:col1, :] = resized

        if squeeze:
            canvas = canvas[..., 0]
        return canvas.astype(dtype, copy=False)

    @staticmethod
    def radial_gradient(
        size: Tuple[int, int],
        center: Tuple[float, float] = (0.5, 0.5),
        max_radius: Optional[float] = None,
        falloff_power: float = 1.0,
        invert: bool = False,
        dtype: type = None,
    ) -> "np.ndarray":
        """Generate a normalized radial gradient as a 2D numpy array.

        At the centre the value is 1.0 and it falls toward 0.0 at ``max_radius``
        (or further). Useful for shadow/vignette opacity masks where a single
        contact point should be the brightest part and falloff increases with
        distance.

        Parameters:
            size: ``(width, height)`` in pixels.
            center: Origin of the gradient in *normalized* image coords
                ``(u, v)`` where ``(0,0)`` is top-left and ``(1,1)`` is
                bottom-right. ``(0.5, 1.0)`` = bottom-centre (common for
                ground-contact shadows).
            max_radius: Distance (in pixels) at which the gradient hits 0.
                ``None`` → image diagonal (full coverage).
            falloff_power: Exponent applied to the normalized distance before
                inverting. ``1.0`` = linear; ``<1`` = sharper falloff near the
                centre; ``>1`` = softer, lingers longer.
            invert: If True, return ``1 - result`` (centre dark, edges bright).
            dtype: Output dtype. ``None`` → ``float32``. Pass ``np.uint8`` to
                get a 0-255 mask ready to use as an alpha channel.

        Returns:
            2D numpy array shape ``(height, width)`` with values in ``[0, 1]``
            (or ``[0, 255]`` for uint8).
        """
        w, h = int(size[0]), int(size[1])
        cx = float(center[0]) * (w - 1)
        cy = float(center[1]) * (h - 1)

        if max_radius is None:
            max_radius = math.hypot(w - 1, h - 1)
        max_radius = max(float(max_radius), 1.0)

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        norm = np.clip(dist / max_radius, 0.0, 1.0)
        if falloff_power != 1.0:
            norm = norm ** float(falloff_power)
        result = 1.0 - norm
        if invert:
            result = 1.0 - result

        if dtype is None or dtype == np.float32:
            return result.astype(np.float32, copy=False)
        if dtype == np.uint8:
            return (result * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        return result.astype(dtype, copy=False)

    @staticmethod
    def _fill_triangle(mask, tri):
        """Fill a 2D triangle (3x2 int pixel coords) into ``mask`` with 255 (numpy edge test)."""
        xs, ys = tri[:, 0], tri[:, 1]
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        if x1 < x0 or y1 < y0:
            return
        ax, ay = float(tri[0][0]), float(tri[0][1])
        bx, by = float(tri[1][0]), float(tri[1][1])
        cx, cy = float(tri[2][0]), float(tri[2][1])
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denom) < 1e-9:
            return  # degenerate
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        l1 = ((by - cy) * (xx - cx) + (cx - bx) * (yy - cy)) / denom
        l2 = ((cy - ay) * (xx - cx) + (ax - cx) * (yy - cy)) / denom
        l3 = 1.0 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        sub = mask[y0:y1 + 1, x0:x1 + 1]
        sub[inside] = 255

    @classmethod
    def _contact_falloff(cls, mask, falloff_source, falloff_power, vertical_weight):
        """Radial + vertical contact-falloff weight (0..1) for a silhouette mask (pre-flip coords)."""
        h, w = mask.shape
        rows = np.where(mask.max(axis=1) > 0)[0]
        cols = np.where(mask.max(axis=0) > 0)[0]
        if not (len(rows) and len(cols)):
            return np.ones((h, w), dtype=np.float32)
        top_row, bottom_row = rows[0], rows[-1]
        center_col = (cols[0] + cols[-1]) // 2
        if falloff_source is not None:  # saved-PNG coords -> pre-flip (v mirrored)
            src = (float(falloff_source[0]), 1.0 - float(falloff_source[1]))
        else:
            src = (center_col / max(w - 1, 1), bottom_row / max(h - 1, 1))
        radial_w = max(1.0 - vertical_weight, 0.0)
        vertical_w = max(min(vertical_weight, 1.0), 0.0)
        radial = cls.radial_gradient(
            (w, h), center=src, max_radius=max(bottom_row - top_row, 1), falloff_power=falloff_power
        )
        vertical = np.zeros((h, w), dtype=np.float32)
        span = max(bottom_row - top_row, 1)
        t = np.clip((np.arange(h) - top_row) / span, 0.0, 1.0) ** 0.6
        vertical[:, :] = t[:, None]
        vertical[np.arange(h) < top_row, :] = 0.0
        vertical[np.arange(h) > bottom_row, :] = 1.0
        return radial * radial_w + vertical * vertical_w

    @classmethod
    def rasterize_silhouette(
        cls, meshes, size=512, axis="auto", *, uniform_alpha=False,
        falloff_source=None, falloff_power=0.8, vertical_weight=0.3, blur_amount=1.5,
    ):
        """Rasterize a flattened-silhouette RGBA alpha from world-space mesh triangles.

        DCC-agnostic core of the projected-shadow tools (mayatk / blendertk ``ShadowRig``): the DCC
        supplies world geometry, this projects → fills → composes the contact-falloff alpha. Reuses
        :meth:`gaussian_blur` and :meth:`radial_gradient`; pure numpy triangle fill (no OpenCV/PIL),
        and returns the array rather than writing a file — the caller persists it via its own image
        API (Blender's python ships no PIL, Maya uses cv2), keeping this layer dependency-clean.

        Parameters:
            meshes: iterable of ``(points, tris)`` — ``points`` an ``(N,3)`` world-space float array,
                ``tris`` an ``(M,3)`` int array of vertex indices into it.
            size: square texture resolution.
            axis: projection axis ``'x'`` / ``'y'`` / ``'z'`` / ``'auto'`` (perpendicular to the
                widest XZ span — matches mayatk's auto rule).
            uniform_alpha: flat silhouette (no contact falloff).
            falloff_source: override contact origin in saved-PNG coords ((0,0)=top-left); else auto.
            falloff_power / vertical_weight: contact-falloff shaping. blur_amount: edge Gaussian blur.

        Returns:
            ``(size, size, 4)`` uint8 RGBA array (silhouette in alpha; V flipped for bottom-left UV).
        """
        meshes = [
            (np.asarray(p, dtype=float).reshape(-1, 3), np.asarray(t, dtype=np.int64).reshape(-1, 3))
            for p, t in meshes if len(p) and len(t)
        ]
        if not meshes:
            raise ValueError("rasterize_silhouette: no geometry provided.")

        all_pts = np.concatenate([p for p, _ in meshes], axis=0)
        mn, mx = all_pts.min(axis=0), all_pts.max(axis=0)
        a = axis.lower()
        if a == "auto":
            a = "x" if (mx[2] - mn[2]) > (mx[0] - mn[0]) else "z"
        u_idx, v_idx = {"y": (0, 2), "x": (2, 1)}.get(a, (0, 1))
        # 1.1 = 10% padding; `or 1.0` guards a zero-extent (single-point/degenerate) mesh against /0.
        extent = max(mx[u_idx] - mn[u_idx], mx[v_idx] - mn[v_idx]) * 1.1 or 1.0
        u_c, v_c = (mn[u_idx] + mx[u_idx]) / 2.0, (mn[v_idx] + mx[v_idx]) / 2.0

        mask = np.zeros((size, size), dtype=np.uint8)
        for pts, tris in meshes:
            pu = np.clip(((pts[:, u_idx] - u_c) / extent + 0.5) * size, 0, size - 1).astype(np.int32)
            pv = np.clip((1.0 - ((pts[:, v_idx] - v_c) / extent + 0.5)) * size, 0, size - 1).astype(np.int32)
            proj = np.stack([pu, pv], axis=1)
            for tri in tris:
                cls._fill_triangle(mask, proj[tri])

        if blur_amount and blur_amount > 0:
            mask = cls.gaussian_blur(mask, radius=blur_amount)
        combined = (
            np.ones(mask.shape, dtype=np.float32) if uniform_alpha
            else cls._contact_falloff(mask, falloff_source, falloff_power, vertical_weight)
        )
        alpha = np.flipud((mask.astype(np.float32) / 255.0 * combined * 255).astype(np.uint8))
        result = np.zeros((size, size, 4), dtype=np.uint8)
        result[:, :, 3] = alpha
        return result

    @classmethod
    def convert_rgb_to_gray(cls, data):
        """Convert an RGB Image data array to grayscale (luma weights).

        Parameters:
            data (str/PIL.Image.Image/np.ndarray): An image, path to an image,
                or image data as a numpy array.

        Returns:
            (np.ndarray) 2D float array of luma values.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(cls.ensure_image(data))

        return np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])

    @classmethod
    def convert_rgb_to_hsv(cls, image):
        """Convert an RGB image to HSV mode.

        Uses PIL's native conversion (H/S/V each 0-255, with H scaled from
        0-360°). Note: PNG files cannot be saved as HSV.

        Parameters:
            image (str/obj): An image or path to an image.

        Returns:
            (PIL.Image.Image) image in "HSV" mode.
        """
        return cls.ensure_image(image, mode="RGB").convert("HSV")

    @classmethod
    def convert_i_to_l(cls, image):
        """Convert a high-bit-depth grayscale image to 8-bit 'L'.

        Values above the 8-bit range are treated as 16-bit (0-65535) and
        scaled down (÷257), not truncated.

        Parameters:
            image (str/obj): An image or path to an image.

        Returns:
            (PIL.Image.Image) image in "L" mode.
        """
        im = cls.ensure_image(image)
        data = np.asarray(im)

        if data.dtype != np.uint8:
            if data.max(initial=0) > 255:  # 16-bit range -> scale, don't truncate
                data = np.clip(data, 0, 65535) / 257.0
            data = np.clip(np.round(data), 0, 255).astype(np.uint8)

        return Image.fromarray(data, mode="L")

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
            cls.save_image(img, output_path, format=output_format, **kwargs)
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
    def generate_mipmaps(cls, image: Union[str, Image.Image]) -> List[Image.Image]:
        """Generate a mipmap chain for an image.

        Note: PIL's writers (including DDS) cannot embed mip chains in a file;
        this returns the chain for callers that hand the levels to an external
        codec (see :meth:`register_dds_codec`).

        Parameters:
            image (str | PIL.Image.Image): The input image.

        Returns:
            list[PIL.Image.Image]: ``[base, half, quarter, …]`` down to 1px on
            the shorter side. The base level is a copy of the input.
        """
        base = cls.ensure_image(image).copy()
        chain = [base]

        while min(base.size) > 1:
            base = base.resize(
                (max(base.size[0] // 2, 1), max(base.size[1] // 2, 1)),
                Image.Resampling.LANCZOS,
            )
            chain.append(base)

        return chain

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
        cls.assert_pathlike(filepath_or_filename, "filepath_or_filename")

        filename = os.path.basename(str(filepath_or_filename))
        base_name, _ = os.path.splitext(filename)

        from pythontk.img_utils.map_registry import MapRegistry

        short_suffixes = []
        long_suffixes = []

        for type_aliases in MapRegistry().get_map_types().values():
            for alias in type_aliases:
                if len(alias) <= 3:
                    short_suffixes.append(alias)
                else:
                    long_suffixes.append(alias)

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

            p_short = "|".join(short_parts)
            patterns.append(p_short)

        suffixes_pattern = "|".join(patterns)

        # Pattern: (underscore + suffix) OR (suffix) at end
        pattern = f"(?:_{suffixes_pattern}|{suffixes_pattern})$"
        base_name = StrUtils.format_suffix(base_name, strip=pattern)

        # Strip any configured user prefix/suffix so callers can re-apply them
        # idempotently, then collapse a trailing underscore (preserves the
        # original behavior for filenames like 'foo_.png' even when no affix
        # was supplied).
        return StrUtils.strip_known_affix(
            base_name, prefix=prefix, suffix=suffix
        ).rstrip("_")

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
            cls.save_image(extracted, out_path, format=output_format, **kwargs)
            results[src_chan] = out_path

        return results


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
