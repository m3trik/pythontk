# !/usr/bin/python
# coding=utf-8
from __future__ import annotations

import os
import math
import logging

# OpenCV reads this once, when its EXR codec first initializes (often at the
# first cv2 import). Set it here — pythontk is the ecosystem's EXR/HDR IO entry
# point and imports cv2 only lazily — so EXR/HDR IO works regardless of when a
# consumer first touches cv2. ``setdefault`` respects an explicit opt-out.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from collections import namedtuple
from contextlib import contextmanager
from typing import (
    List,
    Tuple,
    Dict,
    Union,
    Any,
    Callable,
    Optional,
    Sequence,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from PIL import Image as PILImage

try:
    import numpy as np
except ImportError as e:
    logging.getLogger(__name__).debug(f"# ImportError: {__file__}\n\t{e}")
    np = None  # type: ignore
try:
    from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageDraw, ImageMode
except ImportError as e:
    logging.getLogger(__name__).debug(f"# ImportError: {__file__}\n\t{e}")
    # Bind EVERY name the ``try`` imports, not just ``Image``. Two reasons, and the
    # second is the one that bit: (1) an unbound name raises ``NameError`` at its call
    # site instead of taking the intended "no Pillow" branch, and (2) a name that was
    # never created is invisible to late provisioning -- blendertk's
    # ``ensure_image_deps`` installs Pillow into Blender's Python *after* pythontk is
    # already imported and then rebinds these globals, but it can only repair names
    # that exist. Leaving them out silently broke every ImageOps/ImageChops path in
    # Blender even once Pillow was present.
    Image = ImageOps = ImageFilter = ImageChops = ImageDraw = ImageMode = None  # type: ignore

# From this package:
from pythontk.core_utils._core_utils import CoreUtils
from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.core_utils.naming_convention import NamingConvention
from pythontk.file_utils._file_utils import FileUtils

# The facade's bodies, split by concept (CODE_STANDARD section 3): each is an
# ImgUtils base, so helpers resolve through ``cls`` and the public methods
# (signatures + docstrings) stay here for the flat ``ptk.<method>`` surface.
from pythontk.img_utils._image_header import _ImgHeaderInternal
from pythontk.img_utils._codecs import _ImgCodecInternal
from pythontk.img_utils._channels import _ImgChannelInternal
from pythontk.img_utils._filters import _ImgFilterInternal
from pythontk.img_utils._atlas import _ImgAtlasInternal
from pythontk.img_utils._rasterize import _ImgRasterizeInternal
from pythontk.img_utils._color_space import _ImgColorSpaceInternal


# Per-format IO capability. ``backend`` selects the library used to read/write:
#   "pil" — Pillow (the default).
#   "cv2" — OpenCV, for float formats Pillow cannot handle (EXR, HDR).
ImageFormat = namedtuple("ImageFormat", "read write backend")


class ImgUtils(
    _ImgHeaderInternal,
    _ImgCodecInternal,
    _ImgChannelInternal,
    _ImgFilterInternal,
    _ImgAtlasInternal,
    _ImgRasterizeInternal,
    _ImgColorSpaceInternal,
    HelpMixin,
):
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
        "webp": ImageFormat(True, True, "pil"),
        "bmp": ImageFormat(True, True, "pil"),
        "tga": ImageFormat(True, True, "pil"),
        "tiff": ImageFormat(True, True, "pil"),
        "gif": ImageFormat(True, True, "pil"),
        "dds": ImageFormat(
            True, True, "pil"
        ),  # DXT-tier; BC7/BC6H unsupported by PIL's writer
        "exr": ImageFormat(True, True, "cv2"),
        "hdr": ImageFormat(True, True, "cv2"),
    }

    recognized = tuple(image_formats)  # discovery / file dialogs
    readable = tuple(e for e, f in image_formats.items() if f.read)  # load / scan
    writable = tuple(
        e for e, f in image_formats.items() if f.write
    )  # convert / output menus

    # Backward-compatible alias for the historical flat list (discovery surfaces).
    texture_file_types = list(recognized)

    # Encode-only delivery containers, written through an external encoder
    # rather than PIL/cv2. Deliberately NOT folded into ``image_formats`` /
    # ``writable``: nothing here can be read back into an Image, and panels
    # persist their format combos by INDEX (``OutputTemplates.format_choices``
    # documents that position is a compatibility contract), so growing
    # ``writable`` would re-point every saved selection. A surface that wants
    # to offer KTX2 appends this set explicitly, gated on
    # :meth:`ktx2_available`.
    DELIVERY_FORMATS = ("ktx2",)

    # Containers only a DELIVERY carrier (a GLB, the web) may hold — never a
    # DCC scene's texture node or an FBX, whose consumers cannot read them.
    # A superset of DELIVERY_FORMATS: KTX2 is encode-only, while WebP is a
    # perfectly readable image *here* (PIL decodes it, glTF embeds it natively
    # via EXT_texture_webp) and still unreadable *there* — measured 2026-08-25,
    # a Maya `file` node pointed at a 64x64 .webp reports outSize 0x0 where
    # png / tga / jpg all report 64x64, and an FBX exported with webp maps
    # binds no textures in any consumer. So webp must NOT join
    # DELIVERY_FORMATS (that one routes `save_image` to the external KTX2
    # encoder and gates a UI on `ktx2_available`); the distinction is which
    # DESTINATION may carry it, which is what this set names.
    # Consumers: the scene-exporter texture clamp in mayatk / blendertk
    # (`_resolved_output_type` / `_scene_safe_output_type`) — the GLB's own
    # container row (`ExportRun.glb_texture_params`) is deliberately NOT
    # clamped by it.
    DELIVERY_ONLY_FORMATS = DELIVERY_FORMATS + ("webp",)

    # Plain photographic raster formats (dotted, lowercase) — the directory-scan
    # set shared by the photogrammetry/SfM ingest cluster (ExposureEqualizer /
    # ImageCurator / MaskGenerator). A deliberate semantic subset of
    # ``image_formats``: capture stills only, no float/texture formats.
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

    # DDS block-compression formats Pillow's writer handles directly (no external
    # tool). BC7 / BC6H are not in this set — they need a registered codec.
    PIL_DDS_PIXEL_FORMATS = ("DXT1", "DXT3", "DXT5", "BC5")

    # Containers whose writer accepts a lossy quality setting. Everything else in
    # ``image_formats`` is lossless (or float), so a quality request against one
    # is a caller mistake worth naming rather than dropping on the floor.
    LOSSY_FORMATS = ("jpg", "jpeg", "webp")

    # The subset with NO lossless mode at all, so naming one as the output
    # container IS a request to degrade the pixels -- there is no "write it
    # losslessly instead" for these. WebP is deliberately absent: it has a
    # lossless mode and ``_apply_lossy_kwargs`` selects it when no quality is
    # requested, so a WebP target degrades nothing by default.
    ALWAYS_LOSSY_FORMATS = ("jpg", "jpeg")

    # JPEG has no lossless mode, so ``quality=None`` still has to pick a number.
    # 95/4:4:4 rather than Pillow's 75/4:2:0: this is a *texture* library, and
    # chroma subsampling is what turns a normal map's X/Y vectors to mush.
    JPEG_DEFAULT_QUALITY = 95

    # WebP's hard encoder ceiling on either edge. A larger source has to stay PNG
    # or be resized first — the encoder's own error names the limit but not the fix.
    WEBP_MAX_DIMENSION = 16383

    #: Threads for a batch of image encodes (see :meth:`encode_workers`).
    #: Deliberately well below a modern core count: each worker holds a fully
    #: decoded source (a 4096 RGBA is 67 MB) beside its resize and encode
    #: buffers, and these batches routinely run inside a DCC that is already
    #: holding the exported scene -- the ceiling is host memory, not cores.
    ENCODE_WORKERS = 8

    @classmethod
    def encode_workers(cls, requested: Optional[int] = None) -> int:
        """Threads for a batch of encodes: *requested* as given (at least 1),
        else :attr:`ENCODE_WORKERS` capped by the core count.

        The one policy the GLB texture pass (``MeshConvert.optimize_glb_textures``),
        ``MapOptimizer.optimize_maps`` and the DCC exporters' texture passes
        share. Pillow decodes, resizes and encodes in C with the GIL released,
        so threads scale close to linearly (measured: 31.8 s serial for 27
        production images).
        """
        if requested:
            return max(1, int(requested))
        return max(1, min(cls.ENCODE_WORKERS, os.cpu_count() or 1))

    # What each container ACTUALLY stores when it cannot hold the requested mode.
    # Measured against Pillow's writers (save + reopen), not assumed — half of
    # these raise rather than degrade, so a caller cannot discover them by
    # inspecting the result.
    #
    # This exists because a container's limits are invisible until after the
    # write: WebP has no grayscale mode at all, so an "L" roughness map comes
    # back as RGB. Left implicit, the optimizer would report 8-bit for a file
    # that is 24-bit on disk, and its dry run would predict a mode the real run
    # could not produce. Only modes whose stored form DIFFERS are listed.
    _CONTAINER_MODE_FALLBACKS: Dict[str, Dict[str, str]] = {
        # No grayscale and no palette; alpha survives.
        "webp": {"L": "RGB", "P": "RGB", "I;16": "RGB", "LA": "RGBA"},
        # No alpha of any kind, no palette, 8-bit only.
        "jpg": {"LA": "L", "P": "RGB", "RGBA": "RGB", "I;16": "L"},
        "jpeg": {"LA": "L", "P": "RGB", "RGBA": "RGB", "I;16": "L"},
        # Writes 24-bit; alpha is dropped and 16-bit grayscale flattened.
        "bmp": {"LA": "L", "RGBA": "RGB", "I;16": "L"},
        "tga": {"I;16": "RGB"},
        # Always palettised, whatever goes in.
        "gif": {"L": "P", "LA": "P", "RGB": "P", "RGBA": "P", "I;16": "P"},
        # Basis Universal (KTX2) is 8-bit LDR: 16-bit grayscale flattens (a
        # precision loss the optimizer announces), palettes unroll. The encoder
        # stages exactly this stored mode to the PNG toktx reads.
        "ktx2": {"1": "L", "P": "RGB", "PA": "RGBA", "I;16": "L"},
    }

    @classmethod
    def effective_mode(cls, mode: str, ext: str) -> str:
        """The mode *ext* will actually store for an image in *mode*.

        Returns *mode* unchanged when the container can hold it — including for
        every container with no entry in :attr:`_CONTAINER_MODE_FALLBACKS`
        (PNG/TIFF hold everything; EXR/HDR go through the float path).

        Parameters:
            mode: Requested PIL mode.
            ext: Target extension, with or without a leading dot.

        Returns:
            str: The stored mode — use it to predict or report what is on disk
            rather than what was handed to the writer.
        """
        ext = (ext or "").lower().lstrip(".")
        return cls._CONTAINER_MODE_FALLBACKS.get(ext, {}).get(mode, mode)

    @classmethod
    def dropped_channels(
        cls, mode: str, ext: str = "", *, target_mode: str = ""
    ) -> Tuple[str, ...]:
        """Band names a write or a conversion cannot keep from *mode*.

        A widening is not automatically a loss — "L" to WebP's RGB duplicates
        the one channel it had. This names the channels that actually go away,
        which for a PACKED map is the difference between a container note and
        destroyed data: MSAO carries Smoothness in alpha, so writing one to
        JPEG silently discards a material input rather than a transparency.

        Answers for a CONTAINER (*ext*) or for an explicit *target_mode* --
        a registry-declared mode narrows the same way a container does, and
        the band rule below is subtle enough that asking it twice would be
        two chances to get it wrong. Pass exactly one; *target_mode* wins.

        Parameters:
            mode: The source image mode.
            ext: The container being written to. Resolved through
                :meth:`effective_mode`.
            target_mode: An explicit destination mode, used verbatim.

        Returns:
            tuple: Band names present in *mode* but absent from the stored
            mode, in the source's own band order. Empty when nothing is lost.
        """
        stored = target_mode or cls.effective_mode(mode, ext)
        if stored == mode:
            return ()
        try:
            have = ImageMode.getmode(mode).bands
            kept = set(ImageMode.getmode(stored).bands)
        except (KeyError, ValueError):
            return ()
        # Comparing band NAMES alone is wrong: it reports "L" as dropped when
        # WebP stores an "L" map as RGB, which replicates the single channel
        # rather than truncating it. A band only disappears when the stored
        # mode has no counterpart carrying its data, and widening always has
        # one — "L"/"P" -> RGB replicate or unroll, "RGB" -> "P" quantises but
        # keeps colour, "I;16" -> "L" loses precision, not a channel. Across
        # every entry in _CONTAINER_MODE_FALLBACKS that leaves alpha as the one
        # band a container genuinely discards.
        return tuple(b for b in have if b == "A" and b not in kept)

    @classmethod
    def channels_carrying_data(cls, image, bands) -> Tuple[str, ...]:
        """Which of *bands* actually hold varying data in *image*.

        A dropped channel is only a LOSS if something was in it. Alpha that
        is uniformly opaque -- a lightmap, an albedo with no transparency --
        costs nothing, and warning about it trains the reader straight past
        the line that matters.

        Parameters:
            image: A PIL image.
            bands: Band names to test, e.g. the result of
                :meth:`dropped_channels`.

        Returns:
            tuple: Those of *bands* the image both HAS and varies across.
        """
        if not bands:
            return ()
        try:
            ranges = image.getextrema()
        except OSError:  # truncated source -- the writer will surface it
            return ()
        # A SINGLE-band image answers one ``(min, max)`` pair rather than a
        # tuple of them, so zipping it raw pairs the band with the MINIMUM and
        # the test below then reads it as "not a range" and stays quiet.
        if ranges and not isinstance(ranges[0], tuple):
            ranges = (ranges,)
        extrema = dict(zip(image.getbands(), ranges))
        return tuple(
            b
            for b in bands
            if isinstance(extrema.get(b), tuple) and extrema[b][0] != extrema[b][1]
        )

    # Optional external DDS codec for block formats Pillow can't write (BC7, BC6H).
    # Registered via :meth:`register_dds_codec`; ``None`` until an extension installs one.
    _dds_codec = None

    # Optional KTX2/Basis encoder override. ``None`` = discover the built-in
    # ``toktx`` wrapper on demand — see :meth:`resolve_ktx2_encoder`.
    _ktx2_encoder = None

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
    def register_ktx2_encoder(cls, encoder) -> None:
        """Register the KTX2/Basis encoder ``save_image`` uses for ``.ktx2``.

        The same seam shape as :meth:`register_dds_codec`, with one difference:
        pythontk ships a default implementation
        (:class:`~pythontk.img_utils.ktx2_encoder.Ktx2Encoder`, a ``toktx`` CLI
        wrapper) that is picked up automatically whenever the binary is
        discoverable — so registration is only for substituting another
        implementation (or a fake in tests).

        Parameters:
            encoder: Object exposing ``encode(source, output, codec=...,
                srgb=..., mipmaps=..., quality=...)`` — the
                ``Ktx2Encoder.encode`` contract. ``None`` restores built-in
                discovery.
        """
        cls._ktx2_encoder = encoder

    @classmethod
    def resolve_ktx2_encoder(
        cls,
        required: bool = False,
        auto_install: bool = False,
        prompt: Union[bool, Callable[[str], bool]] = True,
    ):
        """The registered KTX2 encoder, or the built-in ``toktx`` wrapper.

        Parameters:
            required: If True, raise ``FileNotFoundError`` (naming the install
                source and the registration seam) instead of returning None.
            auto_install: When no encoder is discoverable, offer the managed
                KTX-Software download (:meth:`Ktx2Encoder.resolve_toktx`) and
                bind the encoder to what it installs.
            prompt: Consent policy for that download -- ``True`` asks on the
                console, ``False`` needs none, a callable ``(question) -> bool``
                is asked instead (a panel passes its dialog). See
                :meth:`AppInstaller.consent`.

        Returns:
            The encoder, or None when unavailable and *required* is False.
        """
        if cls._ktx2_encoder is not None:
            return cls._ktx2_encoder
        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        if Ktx2Encoder.available():
            return Ktx2Encoder()
        if auto_install:
            # The install path owns consent, fetch, and the fix-shaped error.
            # Bind the path it hands back rather than re-discovering, so a
            # just-written catalog entry cannot be missed.
            toktx = Ktx2Encoder.resolve_toktx(
                required=required, auto_install=True, prompt=prompt
            )
            if toktx:
                return Ktx2Encoder(toktx=toktx)
            if required:
                # The same backstop the non-auto_install branch below spells
                # out, and it was missing here: `resolve_toktx(required=True)`
                # is EXPECTED to raise, but relying on a delegate to always
                # raise is what turns `required=True` -- a promise of a usable
                # encoder or an exception -- into a silent None handed to a
                # caller who asked for a guarantee. It then fails much later,
                # inside the encode, with no mention of the missing tool.
                raise Ktx2Encoder.not_installed_error()
            return None
        if required:
            # `Ktx2Encoder.resolve_toktx(required=True)` is expected to raise
            # this same FileNotFoundError -- but relying on the delegate to
            # always raise falls through to `return None` (handing "no
            # encoder" to a caller who explicitly asked for a guaranteed one)
            # if `available()` says False while `resolve_toktx` unexpectedly
            # succeeds. Raise unconditionally instead, with the encoder's own
            # fix-shaped message.
            raise Ktx2Encoder.not_installed_error()
        return None

    @classmethod
    def ktx2_available(cls) -> bool:
        """True when ``.ktx2`` output is currently writable — the capability
        gate a UI checks before offering :attr:`DELIVERY_FORMATS`."""
        # A predicate never raises: a resolver that refuses (no binary, no
        # catalog) answers "not available", which is the question asked.
        try:
            return cls.resolve_ktx2_encoder() is not None
        except FileNotFoundError:
            return False

    @classmethod
    def settle_ktx2_encoder(
        cls,
        prompt: Union[bool, Callable[[str], bool]],
        refused: Callable[[str], Any],
        installed: Optional[Callable[[str], Any]] = None,
    ) -> bool:
        """Whether a KTX2 run may go ahead: :meth:`ensure_ktx2_encoder`, with
        its two outcomes delivered to the host rather than raised at it.

        The step every panel that offers KTX2 was writing for itself -- Maya's
        and Blender's Scene Exporters, the WebXR preview, the Map Converter --
        each with its own copy of the same try/except and its own drift in
        what it told the user.  Nothing here is host-specific: the host says
        how it asks (*prompt*), how it refuses (*refused*: the fix-shaped
        message naming the manual install, to a log or a dialog) and, when it
        cares, how it reports an install it just made (*installed*: the
        binary's path).

        Parameters:
            prompt: As :meth:`ensure_ktx2_encoder`'s -- a consent callable, or
                a bool answering for the user.
            refused: Receives the message when the encoder is not there and
                the install was declined or failed.
            installed: Receives the installed binary's path when the call
                installed it; ``None`` reports nothing.

        Returns:
            True when an encoder is available now.
        """
        try:
            path = cls.ensure_ktx2_encoder(prompt=prompt)
        except FileNotFoundError as error:
            refused(str(error))
            return False
        if path and installed is not None:
            installed(path)
        return True

    @classmethod
    def ensure_ktx2_encoder(
        cls, prompt: Union[bool, Callable[[str], bool]] = True
    ) -> Optional[str]:
        """Guarantee a KTX2 encoder, offering the managed install when none is found.

        The **panel-side** counterpart of :meth:`resolve_ktx2_encoder`: a user
        who picks KTX2 in a UI must be offered the install, never handed a URL
        and an abort. Every such panel was writing the same four steps — probe,
        announce, resolve with consent, then re-discover the binary to report
        it — and by the third copy (Maya's Scene Exporter, Blender's, the WebXR
        preview) the re-discovery had no owner. It lives here because it is a
        property of the ENCODER, not of any host: nothing in it is Maya-, Blender-
        or Qt-specific, and mayatk and blendertk cannot import each other.

        Parameters:
            prompt: Consent policy for the download -- ``True`` asks on the
                console, ``False`` needs none, a callable ``(question) -> bool``
                is asked instead (a panel passes its dialog). See
                :meth:`AppInstaller.consent`.

        Returns:
            The ``toktx`` path this call INSTALLED, or ``None`` when an encoder
            was already available. That distinction is the whole return value:
            a caller reports "installed <path>" only for the first, and both
            answers are one call rather than a flag the caller has to carry
            past its own try/except.

        Raises:
            FileNotFoundError: There is still no encoder -- declined, or the
                install failed. It is the fix-shaped error naming the manual
                install, so a caller can surface it verbatim.
        """
        if cls.ktx2_available():
            return None
        # `required=True` is what makes a decline raise rather than return
        # None -- the caller asked for a guaranteed encoder.
        encoder = cls.resolve_ktx2_encoder(
            required=True, auto_install=True, prompt=prompt
        )
        # The path off the encoder the resolver just BOUND, never a second
        # discovery pass: the install may have written a catalog entry a fresh
        # `resolve_toktx()` would miss, which is the same reason the resolver
        # binds rather than re-discovers.
        #
        # getattr: a CUSTOM encoder registered through `register_ktx2_encoder`
        # need not be a `Ktx2Encoder` and so need not name a binary. It cannot
        # reach here today -- one would have made `ktx2_available()` true above
        # -- so this is about not turning a future seam into an AttributeError
        # in the middle of a consent flow.
        return getattr(encoder, "toktx", None)

    @classmethod
    def save_image(
        cls,
        image: Union[str, Image.Image],
        name: str,
        mode: str = None,
        bit_depth: int = None,
        compression: str = None,
        quality: int = None,
        colorspace: str = None,
        uastc_rdo: float = None,
        uastc_rdo_dictionary: int = None,
        **kwargs,
    ):
        """Save an image to ``name``, dispatching on the file extension.

        Routing follows :attr:`image_formats`: most formats use Pillow; float
        formats (EXR, HDR) use OpenCV via :meth:`_save_via_cv2`; the
        :attr:`DELIVERY_FORMATS` (``.ktx2``) route to the external encoder via
        :meth:`_save_ktx2`. A recognized but read-only format raises
        ``ValueError``.

        Parameters:
            image (str | PIL.Image.Image): Image object or file path.
            name (str): Output path including filename and extension (e.g., "output.png").
            mode (str, optional): Converts the image to the specified mode before saving (e.g., "RGB", "L").
            bit_depth (int, optional): Target per-channel bit depth. 16 writes a
                16-bit PNG/TIFF (8-bit sources are promoted); other containers fall
                back to 8-bit with a warning. 32-bit float is the EXR/HDR path.
            compression (str, optional): GPU compression scheme. For ``.dds`` a
                block format (e.g. "DXT5", "BC7" — see
                :meth:`_save_dds_compressed`); for ``.ktx2`` a Basis codec
                ("UASTC" or "ETC1S" — see :meth:`_save_ktx2`). Ignored elsewhere.
            quality (int, optional): Lossy quality (1-100) for the
                :attr:`LOSSY_FORMATS` containers, and the ETC1S quality dial for
                ``.ktx2``. **None means lossless wherever the container offers
                it** — see :meth:`_apply_lossy_kwargs`. A value passed against a
                lossless container is reported and ignored.
            colorspace (str, optional): Transfer-function label for containers
                that carry one — currently ``.ktx2`` ("sRGB"/"linear"; None =
                sRGB). A label, not a conversion: the loader samples by it, so
                linear data maps must say so. Ignored by the PIL/cv2 paths.
            uastc_rdo (float, optional): ``.ktx2`` UASTC encodes only -- the RDO
                lambda (``Ktx2Encoder`` ``uastc_rdo``); None = off. Which maps
                should take it, and at what cap, is the caller's policy
                (``MapOptimizer.resolve_uastc_rdo``).
            uastc_rdo_dictionary (int, optional): The RDO dictionary size with
                *uastc_rdo*; None = toktx's own.
            **kwargs: Additional arguments forwarded to PIL.Image.save (e.g.,
                optimize=True, compress_level=9). Ignored for OpenCV-backed formats.

        Raises:
            ValueError: *name* names a format this class can read but not write.
            OSError: an EXR/HDR write failed (see :meth:`_save_via_cv2`).
        """
        im = cls.ensure_image(image, mode)  # Now allows optional mode conversion

        ext = os.path.splitext(name)[1].lstrip(".").lower()

        # KTX2/Basis routes to the external encoder before every PIL concern:
        # bit depth (Basis is 8-bit), lossy kwargs, and encoder buffers do not
        # apply, and the mode fixup happens against the staged PNG instead.
        if ext in cls.DELIVERY_FORMATS:
            cls._save_ktx2(
                im,
                name,
                compression,
                quality,
                colorspace,
                uastc_rdo=uastc_rdo,
                uastc_rdo_dictionary=uastc_rdo_dictionary,
            )
            return

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
        if (
            bit_depth
            and int(bit_depth) >= 16
            and cls._save_high_bit_depth(im, name, int(bit_depth))
        ):
            return

        # Widen to what the container can actually hold. Replaces the former
        # RGBA->RGB-for-JPEG special case, which covered only one of the four
        # modes JPEG rejects (P, LA and I;16 raised OSError instead).
        effective = cls.effective_mode(im.mode, ext)
        if effective != im.mode:
            if effective == "P" and im.mode in ("RGB", "L"):
                # ADAPTIVE, not Pillow's default 216-colour WEB palette. This
                # fallback replaced "let GifImagePlugin quantise on the way
                # out", which used an adaptive palette -- a bare convert("P")
                # does not, and measured 49/255 max (19.4 mean) round-trip
                # error on a gradient against the writer's own 17/2.76. A
                # palettising container must not come out worse than the
                # plugin it took the job from.
                #
                # Gated on the source mode because ADAPTIVE routes through
                # ``quantize()``, which accepts only RGB and L and raises
                # ``ValueError: image has wrong mode`` on the LA / RGBA / I;16
                # entries this same table maps to P. Those keep the plain
                # convert, i.e. exactly their previous behaviour -- the
                # measured regression is on the RGB path.
                im = im.convert(effective, palette=Image.Palette.ADAPTIVE)
            else:
                im = im.convert(effective)

        if ext == "webp":
            cls._assert_webp_dimensions(im, name)

        if ext in cls.LOSSY_FORMATS:
            kwargs = cls._apply_lossy_kwargs(ext, quality, kwargs)
        elif quality is not None:
            print(
                f"# ImgUtils: '{ext}' is a lossless container; ignoring "
                f"quality={quality} for {name}."
            )

        with cls._sized_encoder_buffer(im, ext, kwargs):
            im.save(name, **kwargs)

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

    @staticmethod
    def unique_dir_stems(dirs):
        """Unique, order-preserving output stems for a set of directories.

        The stem is the directory basename; when two directories share a
        basename (e.g. ``capA/images`` and ``capB/images``) the parent
        directory name is prepended (``capA_images``) — walking further up
        the path until every stem is unique, with a positional-index suffix
        as the final fallback (identical paths passed twice). Consumers that
        write one output directory per input directory key them by these
        stems so same-basename sources can't clobber each other.

        Parameters:
            dirs (Sequence[str]): Directory paths (need not exist).

        Returns:
            (list) One stem per input, in input order.
        """
        norm = [os.path.normpath(str(d)) for d in dirs]
        stems = [os.path.basename(p) or p for p in norm]
        parents = [os.path.dirname(p) for p in norm]
        while len(set(stems)) != len(stems):
            dupes = {s for s in stems if stems.count(s) > 1}
            changed = False
            for i, s in enumerate(stems):
                parent_name = os.path.basename(parents[i])
                if s in dupes and parent_name:
                    stems[i] = f"{parent_name}_{s}"
                    parents[i] = os.path.dirname(parents[i])
                    changed = True
            if not changed:  # path components exhausted (identical inputs)
                stems = [f"{s}_{i}" if s in dupes else s for i, s in enumerate(stems)]
                break
        return stems

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
    def get_image_size(image_path: str) -> Optional[Tuple[int, int]]:
        """``(width, height)`` of an image, read as cheaply as possible.

        Parses the JPEG/PNG/DDS/TGA/OpenEXR/Radiance HDR header with the
        **stdlib only** (no PIL/numpy/cv2), so it works in dependency-light
        interpreters such as Metashape's bundled Python; falls back to PIL for
        other formats when available. ``None`` if the size can't be determined.
        Use this (not :meth:`get_image_info`) when you need only the dimensions
        and can't assume PIL is installed.

        Reads the file, so on an online-only cloud placeholder it triggers the
        download -- gate on :meth:`FileUtils.is_cloud_placeholder` first when
        sizing many files a user merely browses.
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

    @staticmethod
    def get_image_mode(image_path: str) -> Optional[str]:
        """The image's channel layout, read as cheaply as possible.

        PIL's mode name (``"RGBA"``, ``"RGB"``, ``"L"``, ``"I;16"``, ``"P"``)
        from a lazy ``Image.open`` -- the header only, no pixels decoded. The
        float formats PIL does not read answer from their own headers: OpenEXR
        from its channel list (``"RGBA;16F"`` half, ``";32F"`` float, ``"L"``
        for luminance, ``"6ch"`` for a layered render), Radiance HDR as
        ``"RGB;32F"``. :meth:`format_bit_depth` spells a PIL mode's depth.
        ``None`` when the file is missing or unreadable.

        The :meth:`get_image_size` twin, with its cost: it reads the file, so
        gate on :meth:`FileUtils.is_cloud_placeholder` before probing many
        files a user merely browses.
        """
        mode = ImgUtils._image_mode_from_header(image_path)
        if mode:
            return mode
        if Image is not None:
            try:
                with Image.open(image_path) as im:
                    return str(im.mode)
            except Exception:
                pass
        return None

    #: The fields :meth:`texture_facts` can read.
    TEXTURE_FACTS = ("bytes", "dimensions", "mode")

    #: ``{(normalized file, field): ((mtime_ns, byte size), value)}`` behind
    #: :meth:`texture_facts` -- a table refreshes often; a header is read once
    #: per file version.
    _facts_cache: Dict[tuple, tuple] = {}

    @classmethod
    def texture_facts(
        cls, path: str, fields: Sequence[str] = TEXTURE_FACTS
    ) -> Dict[str, Any]:
        """What a texture listing shows beside a path, each read only if asked.

        *path* may be a tile / frame token pattern (``rock.<UDIM>.png``):
        ``"bytes"`` sums every file of the set, ``"dimensions"`` and
        ``"mode"`` read the first one's header (:meth:`get_image_size`,
        :meth:`get_image_mode` -- never the pixels). A field not in *fields* is
        never read, so a column that is hidden costs nothing. An online-only
        cloud placeholder answers ``"bytes"`` (a stat downloads nothing) but
        not the header fields, which would download it. Cached per file
        version.

        Parameters:
            path: An absolute file or token pattern.
            fields: Any of :attr:`TEXTURE_FACTS`.

        Returns:
            ``{"tiles": n, "online_only": bool, <field>: value}`` -- ``bytes``
            an int, ``dimensions`` ``(w, h)``, ``mode`` a mode string, either
            of the last two ``None`` when unreadable. ``{}`` when nothing is on
            disk.
        """
        from pythontk.file_utils.tiled_path import TiledPath

        files = TiledPath.tiles(path)
        if not files:
            return {}
        first = files[0]
        online_only = FileUtils.is_cloud_placeholder(first)
        facts: Dict[str, Any] = {"tiles": len(files), "online_only": online_only}

        def cached(file, field, read):
            try:
                stat = os.stat(file)
            except OSError:
                return None
            version = (stat.st_mtime_ns, stat.st_size)
            key = (os.path.normcase(os.path.abspath(file)), field)
            hit = cls._facts_cache.get(key)
            if hit and hit[0] == version:
                return hit[1]
            value = read(file)
            cls._facts_cache[key] = (version, value)
            return value

        if "bytes" in fields:
            total = 0
            for file in files:
                try:
                    total += os.path.getsize(file)
                except OSError:
                    pass
            facts["bytes"] = total
        if "dimensions" in fields:
            facts["dimensions"] = (
                None if online_only else cached(first, "size", cls.get_image_size)
            )
        if "mode" in fields:
            facts["mode"] = (
                None if online_only else cached(first, "mode", cls.get_image_mode)
            )
        return facts

    #: Width:height of an equirectangular (latlong) environment map.
    LATLONG_ASPECT = 2.0

    #: ``{normalized path: ((mtime_ns, byte size), (w, h) | None)}`` behind
    #: :meth:`is_equirectangular` -- a file browser re-lists on every open, so a
    #: header is read once per file version.
    _latlong_size_cache: Dict[str, tuple] = {}

    @classmethod
    def is_equirectangular(
        cls, image_path: str, tolerance: float = 0.05
    ) -> Optional[bool]:
        """Whether *image_path* is shaped like a latlong environment map (2:1).

        The projection a dome light / world background maps by default, and the
        shape no baked lightmap or tiling texture has (those are square). Read
        from the header (:meth:`get_image_size`), cached per file version.

        Parameters:
            image_path: The image file.
            tolerance: Relative slack on :attr:`LATLONG_ASPECT` -- stitched
                panoramas land a few rows off (Maya's own ``skyDome.hdr`` is
                4096x2004, 2.044).

        Returns:
            ``None`` when the size is unknown: a missing or unrecognized file,
            or an online-only cloud placeholder, which is never read (the sync
            client would download the whole image). A caller deciding what to
            list should keep an unknown.
        """
        try:
            stat = os.stat(image_path)
        except OSError:
            return None
        if FileUtils.is_cloud_placeholder(image_path):
            return None  # not cached: it sizes normally once synced
        key = os.path.normcase(os.path.normpath(str(image_path)))
        version = (stat.st_mtime_ns, stat.st_size)
        cached = cls._latlong_size_cache.get(key)
        if cached is not None and cached[0] == version:
            size = cached[1]
        else:
            size = cls.get_image_size(image_path)
            cls._latlong_size_cache[key] = (version, size)
        if not size or not size[1]:  # a header may declare a zero height
            return None
        width, height = size
        return abs(width / height - cls.LATLONG_ASPECT) <= (
            cls.LATLONG_ASPECT * tolerance
        )

    @classmethod
    def is_environment_map(
        cls,
        image_path: str,
        *,
        latlong_only: bool = True,
        skip_lightmaps: bool = True,
    ) -> bool:
        """True if *image_path* can light a scene as a dome / world environment.

        What an HDR picker lists from a folder that also holds baked lightmaps
        and EXR textures (a Maya ``sourceimages``). Two tests, each switchable:

        * *skip_lightmaps* -- the file name carries the ``lightmap`` affix of the
          shared :class:`NamingConvention` (``_Lightmap`` unless a user changed
          it), seen through tile indices and light-group tails
          (``room_Lightmap_12``, ``desk_Lightmap.LIGHT_A_areaLight``).
        * *latlong_only* -- :meth:`is_equirectangular`. An unknown size passes:
          never a false reject.
        """
        if skip_lightmaps and NamingConvention.matches(
            os.path.splitext(os.path.basename(str(image_path)))[0], "lightmap"
        ):
            return False
        return not latlong_only or cls.is_equirectangular(image_path) is not False

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
        # Determine the target mode based on map type. The map-type -> mode
        # table is the texture engine's taxonomy (not a general image rule),
        # so it is read from MapRegistry through a deferred import: the engine
        # imports ImgUtils at module load, and a generic ImgUtils consumer
        # should not pay for the map cluster. MapRegistry is a singleton, so
        # the lookup is cheap.
        from pythontk.core_utils.engines.textures.map_registry import MapRegistry

        map_modes = MapRegistry().get_map_modes()
        if map_type in map_modes:
            target_mode = map_modes[map_type]
            image = cls.enforce_mode(image, target_mode, allow_compatible=allow_palette)

        # If the image is already in a standard mode, don't mess with it based on bit depth
        if image.mode in ("RGB", "RGBA", "L", "1", "P"):
            return image

        # Adjust bit depth. Map the source's total bit depth to a standard,
        # widely-convertible mode. The previous {v: k for k, v in bit_depth}
        # inversion was order-dependent and kept only the LAST mode per bit
        # count (16->'PA', 24->'HSV', 32->'I;32LS'), so exotic inputs such as
        # 'LA'/'F'/'YCbCr' converted to nonsensical targets — raising
        # ValueError ('F'->'I;32LS') or silently corrupting pixels
        # ('YCbCr'->'HSV'). A fixed canonical map avoids that.
        canonical = {1: "1", 8: "L", 16: "I;16", 24: "RGB", 32: "RGBA"}
        depth = cls.bit_depth.get(image.mode, 8)
        target = canonical.get(depth, "RGB")

        if image.mode != target:
            try:
                image = image.convert(target)
            except (ValueError, OSError):
                # Not every source mode converts to every canonical target on
                # every PIL build (notably 16-bit -> 'I;16'); fall back to a
                # universally-supported color mode — never re-running the exact
                # conversion that just failed (depth>=32's target IS 'RGBA',
                # which made this fallback dead and let the error escape; the
                # same held for 24-bit's 'RGB' target).
                fallback = "RGB" if target != "RGB" else "RGBA"
                image = image.convert(fallback)

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
        return super().invert_channels(image, channels)

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
        return super().swizzle_channels(image, mapping)

    @classmethod
    @CoreUtils.listify(threading=True)
    def create_mask(
        cls, image, mask, background=(0, 0, 0, 255), foreground=(255, 255, 255, 255)
    ) -> Union[Image.Image, List[Image.Image]]:
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
        return super().gaussian_blur(image, radius, channel)

    @staticmethod
    def dilate_image(
        image: "np.ndarray",
        mask: Optional["np.ndarray"] = None,
        iterations: int = -1,
        connectivity: int = 8,
        return_mask: bool = False,
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
            return_mask: Also return the HxW bool mask of the texels that are
                valid AFTER the passes (the input mask grown by the ring), so a
                caller continuing the fill another way knows where this stopped.

        Returns:
            Image with empty regions filled; same shape and dtype as input --
            or ``(image, mask)`` with *return_mask*.
        """
        return _ImgFilterInternal.dilate_image(
            image, mask, iterations, connectivity, return_mask
        )

    @classmethod
    def denoise_image(
        cls,
        image: "np.ndarray",
        mask: Optional["np.ndarray"] = None,
        radius: int = 2,
        strength: float = 3.0,
        noise: Optional[float] = None,
        outliers: float = 5.0,
    ) -> "np.ndarray":
        """Edge-preserving denoise of a linear-light image (HDR-safe), within *mask*.

        A self-guided filter (He, Sun and Tang, *Guided Image Filtering*) in
        LOG space. Every ``(2 * radius + 1)^2`` window fits the image as a line
        ``a * G + b`` of its own log-brightness ``G`` (the channel mean, so
        RGB and BGR are alike), and the window's spread
        against the noise decides ``a``: a window no wider than the noise gets
        ``a ~ 0`` -- its mean, the noise averaged away -- and one straddling a
        step many times the noise gets ``a ~ 1``, the step kept. A LINEAR ramp
        (a light's falloff) passes through exactly at any ``a``, because a
        window's mean of a ramp is its centre. Each texel then averages the
        lines of every window covering it.

        Log space, because a path tracer's noise is relative -- it scales with
        the signal it rides on -- so in log units one noise level describes a
        whole map, shadow and lit alike. O(texels) at any radius: five box
        filters and one 3x3 median in float32 through cv2 where it is present
        (every DCC Python the bakes run in ships it), integral images in numpy
        where it is not -- the same result either way.

        Only *mask* texels are read or written: every box sum is normalised by
        the count of mask texels in its window, so gutters, background and the
        texels a bake refills are never averaged in, and everything outside
        the mask comes back untouched for the caller to refill from the result.

        Lone spikes (a firefly, a sample-starved black texel) are clamped to
        their 3x3 neighbourhood's median first. The filter alone keeps them:
        one texel far from its neighbours makes every window holding it look
        like an edge. The median decides, not a Laplacian, because beside a
        straight shadow edge the median stays on the texel's own side -- an
        edge texel is not a spike.

        Parameters:
            image: HxW or HxWxC linear values (``>= 0``). Not modified.
            mask: HxW truthy -- the texels that are the image's own content.
                Defaults to every texel with a channel above zero.
            radius: Window radius, in texels of *image*.
            strength: The edge threshold in noise sigmas (``eps = (strength *
                noise) ** 2``): a window whose spread stays within it is
                smoothed as noise, a step well beyond it is kept.
            noise: Per-texel noise as a log-space standard deviation. ``None``
                estimates it from the image: the robust spread (MAD) of the
                Laplacian over the mask's interior, which a smooth signal adds
                little to and an edge -- a thin line of texels -- cannot move.
            outliers: How far from its neighbourhood's median, in noise sigmas,
                a texel must stand to be clamped as a spike; ``0`` disables it.

        Returns:
            The denoised image, same shape and dtype; texels outside *mask*
            unchanged.
        """
        return super().denoise_image(image, mask, radius, strength, noise, outliers)

    @classmethod
    def fill_empty_texels(
        cls,
        image: "np.ndarray",
        mask: Optional["np.ndarray"] = None,
    ) -> "np.ndarray":
        """Fill EVERY empty texel with its nearest valid texel's color.

        The complement of :meth:`dilate_image`: dilation grows a smooth
        averaged gutter a few px wide, but anything beyond it stays
        background. Background texels are what a GPU mip chain averages into
        content at every level -- a black background reads as a dark halo
        around each island/atlas rect the moment the texture is minified
        (distant/grazing views), darkening the border of every instance that
        samples it. After this fill no texel is background, so every mip
        level averages plausible nearby lighting instead.

        Nearest-neighbor via cv2's distance transform when available (one
        O(n) pass -- flood-filling a 2048 map by iteration is hundreds of
        full-image passes). Without cv2 (Blender's Python ships none) the
        near ring is neighbour-averaged and the far field is filled from an
        image pyramid (:meth:`_fill_pyramid`) -- O(log n) passes, where the
        previous iterate-until-filled fallback paid one full-image pass per
        texel of distance.

        Parameters:
            image: HxW or HxWxC array. Not modified -- a copy is returned.
            mask: HxW truthy "valid" mask. Defaults to "any channel > 0"
                (see :meth:`dilate_image` for why baked maps should pass
                their real coverage mask instead).

        Returns:
            Image with every empty texel filled; same shape/dtype as input.
        """
        return super().fill_empty_texels(image, mask)

    @classmethod
    def extrapolate_fill(
        cls,
        image: "np.ndarray",
        mask: "np.ndarray",
        rings: int = 1,
        clamp: float = 2.0,
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """Grow *mask* by *rings* texels, continuing its content's slope.

        Each new texel is the LINEAR extrapolation of the two mask texels
        inward of it along an axis (``2 * v1 - v2``), averaged over the axes
        that have them -- so a ramp continues exactly, where
        :meth:`dilate_image`'s averaging holds the value from further in. That
        difference is a step on every edge two lightmap cells share: an
        averaged refill (or a filter window truncated at the edge) reads the
        light from inside the cell, not at its edge. Use it for the first few
        texels past real content; :meth:`dilate_image` /
        :meth:`fill_empty_texels` remain the tools for the far field.

        Each ring grows along the four axes only (a corner fills on the next
        ring), from texels at least two deep: a one-texel sliver has no slope
        and is left alone. Each extrapolated value is clamped to
        ``[v1 / clamp, v1 * clamp]`` so a shadow edge at the border cannot run
        to zero or blow up; it stays positive for positive content.

        Parameters:
            image: HxW or HxWxC array. Not modified -- a copy is returned.
            mask: HxW truthy -- the texels that are content.
            rings: How many texels to grow.
            clamp: The per-ring bound on the extrapolated value, as a factor of
                the texel it continues from.

        Returns:
            ``(image, mask)``: the filled copy and the grown mask.
        """
        return super().extrapolate_fill(image, mask, rings, clamp)

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

        **Squarified treemap** (the default): rows are grown along the remaining
        rect's SHORTER side, admitting the next item only while that does not
        worsen the row's worst aspect ratio. This is what a resampled bake needs
        -- a rect's two axes are two independent resolutions and the cell is only
        as good as the SMALLER one, and ``bake_atlas`` renders each object AT its
        cell, so a 9:1 cell is a 9x anisotropic loss for that object.

        Neither predecessor bounds aspect. Shelf packing (the ``rows`` path
        below) balances items into shelves whose HEIGHT is a weight share, so
        one dominant item drives every other shelf thin -- and a thin shelf's
        cells are thin at any width (measured: a room of weight 200 among 30
        props of 0.5 gives each prop 341x29 px in a 2048 atlas, 11.9:1).
        Weight-balanced bisection, which replaced it, always cuts the longer
        axis -- provably optimal for TWO items -- but a two-item subtree with
        lopsided weights still hands the lighter one a sliver of the full
        extent, and when that child is a leaf the sliver is final. Measured on
        a production room scene, same bake one algorithm apart, on the
        **delivered** atlas (46 objects, 1024 px): worst cell **101x11 px
        (9.18:1) -> 145x62 (2.34:1)**, the smallest cell's short axis
        **11 px -> 29 px**, median aspect 1.42 -> 1.27, atlas area used 89.1%
        -> 89.3%. Cell-edge uniformity improved with it (the border-vs-interior
        excess halved, 0.051 -> 0.029 median) -- squarer cells are also less
        likely to run a lighting gradient the short way across a few texels.

        Exact tiling, area proportionality and input order are identical under
        all three -- only the shapes differ. Pure Python -- no numpy/PIL.

        Parameters:
            weights: One non-negative importance value per item, in caller order.
                All-zero (or empty-after-clamp) weights fall back to equal
                shares; negative values are clamped to 0.
            rows: Opt in to the legacy shelf packing with this many shelves
                (items balanced across them longest-processing-time first, each
                shelf's height its weight share). ``None`` (default) uses the
                bisection above, which needs no row count.

        Returns:
            One ``(scaleX, scaleY, offsetX, offsetY)`` tuple per input weight, in
            the same order. ``[]`` for no items; ``[(1, 1, 0, 0)]`` for one.
        """
        return _ImgAtlasInternal.compute_atlas_layout(weights, rows=rows)

    @staticmethod
    def atlas_pixel_rects(
        rects: Sequence[Tuple[float, float, float, float]],
        size: Union[int, Tuple[int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        """Convert normalized ``scaleOffset`` rects to integer pixel rects.

        The single source of truth for the UV-rect -> pixel-rect mapping,
        including the vertical flip (UV v is bottom-up, image rows are
        top-down): :meth:`assemble_atlas` places content with exactly this
        mapping, so a consumer that needs the *placed* pixel regions (e.g. to
        build a coverage mask for gutter dilation) must use this instead of
        re-deriving the rounding and inviting off-by-one drift.

        Parameters:
            rects: ``(scaleX, scaleY, offsetX, offsetY)`` rects in [0, 1] UV
                space (:meth:`compute_atlas_layout` / Unity ``lightmapScaleOffset``).
            size: Atlas pixel size -- ``int`` for square, or ``(width, height)``.

        Returns:
            One ``(row0, row1, col0, col1)`` half-open pixel rect per input, in
            input order -- i.e. the content occupies ``image[row0:row1, col0:col1]``.
            Degenerate rects come back with zero (or negative) extent; callers
            skip those the same way :meth:`assemble_atlas` does.
        """
        return _ImgAtlasInternal.atlas_pixel_rects(rects, size)

    @staticmethod
    def flip_rect_v(rect: Sequence[float]) -> List[float]:
        """A ``[sx, sy, ox, oy]`` atlas rect, V-flipped between UV conventions.

        The scaleOffset rects this module produces are bottom-left-origin UV
        space (Unity ``lightmapScaleOffset``); glTF texture space runs top-down
        (``v' = 1 - v``), so a rect published into a glTF texture transform
        (``KHR_texture_transform``) keeps its scales and flips only the V
        offset: ``oy' = 1 - sy - oy``. Involutory -- applying it twice returns
        the input -- so the same helper converts either direction, and being
        THE helper is the point: one V-flip authored twice is how the two
        conventions drift.
        """
        return _ImgAtlasInternal.flip_rect_v(rect)

    @staticmethod
    def compose_rect(
        outer: Optional[Sequence[float]], inner: Sequence[float]
    ) -> List[float]:
        """The one ``[sx, sy, ox, oy]`` rect that applies *inner*, then *outer*.

        ``uv' = uv * s + o`` twice over folded into one: scale ``inner_s *
        outer_s``, offset ``inner_o * outer_s + outer_o``. A rect is a UV
        transform, so a mapping already baked into a layout (UVs an old atlas
        pack squeezed into their cell, say) can move into the rect a consumer
        applies at sample time -- restore the layout, compose, and every texel
        is sampled exactly where it was. ``None`` for *outer* is the identity.

        Parameters:
            outer: The rect applied second (``None``: identity).
            inner: The rect applied first.

        Returns:
            List[float]: ``[sx, sy, ox, oy]``.
        """
        return _ImgAtlasInternal.compose_rect(outer, inner)

    @staticmethod
    def inset_atlas_rects(
        rects: Sequence[Tuple[float, float, float, float]],
        size: Union[int, Tuple[int, int]],
        gutter: int,
    ) -> List[Tuple[float, float, float, float]]:
        """Shrink each atlas rect by a pixel gutter on every side.

        :meth:`compute_atlas_layout` tiles the unit square *exactly*, which
        leaves zero spacing between neighbors -- any mip level or bilinear tap
        near a rect edge then bleeds into the next item. Insetting the content
        rects by ``gutter`` px (and later dilating the assembled atlas into the
        freed border) gives every item a bleed margin while the returned rects
        stay valid ``scaleOffset`` values for sampling the inset content.

        Small rects are protected: the per-axis inset is capped at a quarter of
        the rect's pixel extent, so content never shrinks below half its rect
        (a rect too small to inset is returned unchanged).

        Parameters:
            rects: ``(scaleX, scaleY, offsetX, offsetY)`` rects in [0, 1] UV space.
            size: Atlas pixel size -- ``int`` for square, or ``(width, height)``.
            gutter: Margin in pixels to free on each side of each rect.

        Returns:
            The inset rects, same format and order as the input.
        """
        return _ImgAtlasInternal.inset_atlas_rects(rects, size, gutter)

    @classmethod
    def snap_atlas_rects(
        cls,
        rects: Sequence[Tuple[float, float, float, float]],
        size: Union[int, Tuple[int, int]],
    ) -> List[Tuple[float, float, float, float]]:
        """Snap normalized atlas rects onto the atlas's integer texel grid.

        :meth:`compute_atlas_layout` / :meth:`inset_atlas_rects` produce
        arbitrary float rects, but :meth:`assemble_atlas` writes content at
        ROUNDED pixel edges (:meth:`atlas_pixel_rects`). Publishing the
        un-rounded float as the engine's ``scaleOffset`` therefore samples up
        to half a texel of the neighboring gutter along every rect edge --
        on instanced tiles that sliver lands on the same shared 3D edge from
        both sides and reads as a thin dark border. Snapping re-derives each
        rect FROM its integer pixel rect, so the published rect and the
        written texels agree exactly.

        Degenerate rects (zero pixel extent after rounding) are returned
        unchanged -- :meth:`assemble_atlas` skips placing those, so there is
        nothing to agree with.

        Parameters:
            rects: ``(scaleX, scaleY, offsetX, offsetY)`` rects in [0, 1] UV
                space (bottom-left origin).
            size: Atlas pixel size -- ``int`` for square, or ``(width, height)``.

        Returns:
            The snapped rects, same format and order as the input.
        """
        return super().snap_atlas_rects(rects, size)

    @staticmethod
    def inset_rects_to_texel_centers(
        rects: Sequence[Tuple[float, float, float, float]],
        size: Union[int, Tuple[int, int]],
        bboxes: Optional[Sequence[Optional[Tuple[float, float, float, float]]]] = None,
    ) -> List[Tuple[float, float, float, float]]:
        """Re-aim each rect so its content's edge UVs sample border-texel CENTERS.

        A rect whose content edge lies on a texel BOUNDARY makes every
        bilinear tap along that edge split between the content's border texel
        and the texel beyond it -- in a packed atlas that texel is gutter
        shared with whatever unrelated item landed next door, so up to half
        of an edge tap's weight reads another item's lighting. (A sampling
        defect in its own right; it is not the only thing that can put a line
        on a shared 3D edge -- an island-border falloff baked INTO the source
        map survives this entirely.) The standard lightmap convention maps the content's
        outermost UVs to the CENTER of its outermost texels instead: an edge
        tap then reads the border texel pure, and neighboring gutters only
        matter to minified mips (which dilation already covers). The sub-texel
        stretch this introduces (content spans n texels, sampling spans n-1)
        is invisible on lighting data.

        Per axis, the content span ``[edge0_px, edge1_px]`` is re-mapped to
        ``[floor(edge0_px) + 0.5, ceil(edge1_px) - 0.5]``. A rect whose
        content spans under two texels on either axis is returned unchanged
        (there is no interior to stretch across).

        Parameters:
            rects: ``(scaleX, scaleY, offsetX, offsetY)`` rects in [0, 1] UV
                space (bottom-left origin). Crop-composed rects extending past
                the unit square are fine -- only the content bbox matters.
            size: Atlas pixel size -- ``int`` for square, or ``(width, height)``.
            bboxes: Per-rect content bbox ``(u0, v0, u1, v1)`` in the rect's
                OWN uv space (the UVs the engine will sample through it).
                ``None`` -- or a ``None`` entry -- means full coverage
                ``(0, 0, 1, 1)``.

        Returns:
            The adjusted rects, same format and order as the input.
        """
        return _ImgAtlasInternal.inset_rects_to_texel_centers(rects, size, bboxes)

    @staticmethod
    def uv_crop_extent(
        bbox: Optional[Tuple[float, float, float, float]],
        max_coverage: float = 0.85,
    ) -> Tuple[float, float]:
        """The ``(u, v)`` fraction of a map :meth:`crop_to_uv_bbox` keeps.

        ``(1.0, 1.0)`` when it would take no crop: ``bbox`` ``None`` or
        degenerate, or both axes already at or above *max_coverage* (an
        auto-unwrap's few percent of margin gains nothing from a crop). A
        baker divides a tile's planned size by this so the CROPPED island,
        not the whole map, lands at the cell's density.

        Parameters:
            bbox: The island bounds ``(u0, v0, u1, v1)``; clamped to [0, 1].
            max_coverage: The per-axis coverage at or above which no crop is
                taken (on both axes).

        Returns:
            ``(u_fraction, v_fraction)``.
        """
        return _ImgAtlasInternal.uv_crop_extent(bbox, max_coverage)

    @staticmethod
    def crop_to_uv_bbox(
        img: "np.ndarray",
        bbox: Optional[Tuple[float, float, float, float]],
        cell: Sequence[float],
        max_coverage: float = 0.85,
    ) -> Tuple["np.ndarray", List[float], Tuple[float, float, float, float]]:
        """Crop a baked map to its UV island's bounds and fold the crop into its atlas rect.

        A lightmap whose islands cover only part of the unwrap (an artist's
        layout: production walls at u 0..1/3) wastes its atlas cell on dead
        space. Cropped, the island fills the cell at full density, and the
        returned rect maps ``uv`` so the engine's ``uv * scale + offset``
        still lands exactly where the texels went.

        The crop keeps exactly the texels the island TOUCHES -- no pad. A pad
        admits edge-EXTENSION texels, which are not this object's lighting: a
        point just past a wall panel's edge is coplanar with its neighbour,
        so it renders dark (a padded crop measured a 4.23% mean per-side edge
        error against 1.64%). Touched rather than fully covered texels,
        because the bounds must CONTAIN the island: cropping inside it leaves
        a sub-texel overhang that samples past the cell.

        Parameters:
            img: The map, ``(H, W[, C])``, row 0 at the TOP (v = 1).
            bbox: The island bounds ``(u0, v0, u1, v1)``, or ``None``.
            cell: The atlas cell ``(scaleX, scaleY, offsetX, offsetY)`` the
                map is placed in.
            max_coverage: See :meth:`uv_crop_extent`.

        Returns:
            ``(image, rect, bounds)``: the (possibly) cropped image, the rect
            to publish, and the uv range that maps onto the FULL cell --
            ``(0, 0, 1, 1)`` when no crop was taken. Publish through
            :meth:`inset_rects_to_texel_centers` with ``bboxes=[bounds]``, so
            the cell's edges, not the island's, land on border-texel centers.
        """
        return _ImgAtlasInternal.crop_to_uv_bbox(img, bbox, cell, max_coverage)

    @classmethod
    def resize_into_cell(
        cls,
        image: "np.ndarray",
        size: Tuple[int, int],
        coverage: Optional["np.ndarray"] = None,
        edge_centers: bool = True,
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """Resample a tile into the atlas cell it is published in.

        An exact area (box-filter) resample, with two properties a plain
        resize lacks:

        * **edge_centers** -- the source's edges land on the cell's border-
          texel CENTERS, the mapping :meth:`inset_rects_to_texel_centers`
          publishes. Resized edge to edge instead, every sample along the
          content's edge reads it from half a texel inside, so two coplanar
          tiles meeting at a 3D edge each show their lighting from further
          in than the edge. Each border texel averages the half of its box
          that lies inside the source. Ignored when the cell is under two
          texels on either axis (the inset leaves such a rect unchanged).
        * **coverage** -- a per-texel weight (an island's coverage): only
          covered texels are averaged, so a cell texel the island only
          partly covers holds the island's own value rather than one mixed
          with the gutter -- and needs no refill from the texel inside it.

        Numpy only (no cv2 -- Blender's Python ships none); float64 prefix
        sums, O(texels) at any ratio.

        Parameters:
            image: ``(H, W)`` or ``(H, W, C)`` source tile.
            size: Cell ``(width, height)`` in texels (cv2's order).
            coverage: Optional ``(H, W)`` weights in ``[0, 1]``; ``None`` is
                full coverage.
            edge_centers: Land the source edges on border-texel centers
                (default) rather than on the cell's outer boundary.

        Returns:
            ``(image, coverage)``: the resampled tile (float, the source's
            float dtype or float32) and each cell texel's covered fraction;
            a texel with no coverage is 0 in both.
        """
        return _ImgAtlasInternal.resize_into_cell(image, size, coverage, edge_centers)

    @classmethod
    def stitch_seams(
        cls,
        image: "np.ndarray",
        pairs: "np.ndarray",
        iterations: int = 64,
    ) -> "np.ndarray":
        """Make each pair of atlas positions read one value: a seam between two cells.

        Two atlas cells whose content meets at a 3D edge (two coplanar wall
        panels, two instances of one tile) are sampled from either side of that
        edge, each through its own rect, and each holds its own sampling noise
        -- so however well each cell is finished, they disagree along the edge
        and the edge reads as a step. Given the pixel positions where each side
        samples the same 3D points, this nudges ONLY the texels those bilinear
        taps read until both reads agree, meeting halfway (Jacobi iterations of
        the least-norm correction; a texel several samples share takes the mean
        of their updates). Everything else is untouched. The correspondence --
        which positions are one 3D point -- is the caller's: the image knows
        nothing about geometry.

        Parameters:
            image: ``(H, W)`` or ``(H, W, C)`` atlas.
            pairs: ``(N, 4)`` -- ``(xa, ya, xb, yb)`` pixel positions, texel
                ``i`` centred at ``i`` (``x = u * W - 0.5``,
                ``y = (1 - v) * H - 0.5``), that must read the same value.
            iterations: Solver iterations.

        Returns:
            The stitched copy (same dtype).
        """
        return _ImgAtlasInternal.stitch_seams(image, pairs, iterations)

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
        return super().assemble_atlas(images, rects, size, background=background)

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
        return _ImgRasterizeInternal.radial_gradient(
            size, center, max_radius, falloff_power, invert, dtype
        )

    @classmethod
    def rasterize_uv_triangles(
        cls,
        triangles,
        size: int = 512,
        supersample: int = 4,
    ) -> "np.ndarray":
        """Rasterize filled UV-space triangles into a single-channel coverage image.

        General polygon-coverage primitive (region masks, UV-shell fills,
        coverage checks): the caller supplies triangles in normalized UV
        space, this fills them at ``supersample``× resolution and box-filters
        down, so edges carry the same anti-aliased falloff a baked texture
        has. V is flipped to image coordinates ((0,0) = top-left) to match
        how UV-mapped textures are stored on disk.

        The value is an exact coverage fraction to within the sampling rate,
        so ``== 255`` means "every sample in this texel fell inside the
        geometry" and can be thresholded on as such (what a lightmap bake
        needs to tell an island's own texels from the ones it merely
        overlaps). ``supersample`` sets that rate: 4 resolves coverage to
        1/16 and costs ``(size * 4)²`` bytes of scratch.

        Parameters:
            triangles: (N, 3, 2) array-like of UV coordinates (V up, usually
                in [0,1] — geometry outside the unit square is cropped).
            size: Output square resolution in pixels.
            supersample: Coverage oversampling factor (1 = hard edges).

        Returns:
            (size, size) uint8 coverage array (0 outside, 255 fully inside,
            anti-aliased edges between).
        """
        return super().rasterize_uv_triangles(triangles, size, supersample)

    @classmethod
    def rasterize_silhouette(
        cls,
        meshes,
        size=512,
        axis="auto",
        *,
        uniform_alpha=False,
        falloff_source=None,
        falloff_power=0.8,
        vertical_weight=0.3,
        blur_amount=1.5,
    ):
        """Rasterize a flattened-silhouette RGBA alpha from world-space mesh triangles.

        DCC-agnostic core of the projected-shadow tools (mayatk / blendertk ``ShadowRig``): the DCC
        supplies world geometry, this projects → fills → composes the contact-falloff alpha. Reuses
        :meth:`gaussian_blur` and :meth:`radial_gradient`; pure numpy triangle fill (no OpenCV/PIL),
        and returns the array rather than writing a file — the caller persists it via its own image
        API (Blender via ``bpy``'s image API, Maya via PIL), keeping this layer dependency-clean.

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
        return super().rasterize_silhouette(
            meshes,
            size,
            axis,
            uniform_alpha=uniform_alpha,
            falloff_source=falloff_source,
            falloff_power=falloff_power,
            vertical_weight=vertical_weight,
            blur_amount=blur_amount,
        )

    @classmethod
    def rasterize_height_fields(
        cls,
        meshes,
        *,
        up: int = 1,
        size: int = 64,
        ground: float = 0.0,
        bounds=None,
        padding: float = 0.02,
    ):
        """Top and bottom height fields of world meshes over their footprint.

        A z-buffer pair: per footprint pixel the highest and the lowest
        surface height above *ground* (a solid column is ``[z_bot, z_top]``,
        the hull a horizon-map bake reads — :mod:`pythontk.geo_utils.shadow_horizon`).
        Rows run along the second horizontal axis and columns along the first
        (:meth:`ShadowProjection.horizontal_axes`), with no image flip: this is
        data, indexed ``field[ib, ia]``. Triangles are area-filled at pixel
        centres and their edges are splatted at half-pixel steps, so a member
        thinner than a pixel (a chair leg, a diagonal brace) still registers
        as a one-pixel line instead of vanishing.

        Parameters:
            meshes: Iterable of ``(points, tris)`` — ``(N, 3)`` points,
                ``(M, 3)`` vertex-index triangles.
            up: Vertical axis index.
            size: Pixels per side over the footprint.
            ground: Height of the ground plane along *up*; heights are
                relative to it and clamped at 0 (a surface below the ground
                cannot block a ground texel).
            bounds: ``(a0, a1, b0, b1)`` horizontal extent to rasterize; None
                fits the geometry with *padding*.
            padding: Margin as a fraction of the larger extent when fitting.

        Returns:
            ``(z_top, z_bot, mask, bounds)`` — two ``(size, size)`` float32
            fields (0 where empty), a bool coverage mask, and the bounds used.
        """
        return super().rasterize_height_fields(
            meshes, up=up, size=size, ground=ground, bounds=bounds, padding=padding
        )

    @classmethod
    def rasterize_height_spans(
        cls,
        meshes,
        *,
        up: int = 1,
        size: int = 64,
        ground: float = 0.0,
        bounds=None,
        padding: float = 0.02,
        spans: int = 1,
    ):
        """The solid vertical spans of world meshes per footprint pixel: a
        depth-peeled :meth:`rasterize_height_fields`.

        Per pixel the surface crossings at its centre (every triangle the
        centre falls in, at the triangle's height there) are sorted and
        paired into solid spans -- a closed mesh enters and leaves -- and the
        *spans* thickest are kept, the rest merged into their nearest
        neighbour by gap so nothing solid is ever dropped (``spans=1`` is the
        hull ``rasterize_height_fields`` returns). Triangle edges are splatted
        at half-pixel steps as one hull span where the fill missed, so a
        member thinner than a pixel still registers. A seat over a stretcher
        keeps daylight between them; a plain height field fills it.

        Parameters:
            meshes, up, size, ground, bounds, padding: As :meth:`rasterize_height_fields`.
            spans: Solid spans kept per pixel, ``K``.

        Returns:
            ``(lo, hi, bounds)`` -- two ``(K, size, size)`` float32 arrays,
            ``NaN`` where a pixel has no span at that index, spans sorted by
            height per pixel, and the bounds used.
        """
        return super().rasterize_height_spans(
            meshes,
            up=up,
            size=size,
            ground=ground,
            bounds=bounds,
            padding=padding,
            spans=spans,
        )

    @classmethod
    def rasterize_shadow(
        cls,
        meshes,
        light=None,
        ground=0.0,
        size=512,
        *,
        up=1,
        direction=None,
        source_size=0.0,
        max_stretch=None,
        canvas=None,
        contact=None,
        radius=None,
        height=None,
        padding=0.04,
        uniform_alpha=True,
        falloff_power=0.8,
        vertical_weight=0.3,
        blur_amount=1.0,
    ):
        """Rasterize the shadow world-space meshes cast onto the ground plane.

        The physically projected successor of :meth:`rasterize_silhouette`: every
        triangle is mapped onto the ground through the source
        (:meth:`ShadowProjection.project` — perspective from a position, parallel
        along a direction) and filled where it lands, so an overhead source
        draws the footprint, a low one the long stretched shape, and a near one
        the perspective-grown head. A source with a size draws a penumbra that
        widens with each point's height above the ground — sharp at the contact,
        soft at the tip — as a variable-radius blur.

        The texture covers a canvas rectangle in the shadow's ``(u, w)`` frame
        (``u`` along the bearing away from the light, ``w`` across it; see
        :class:`ShadowProjection`): rows run from the light-side edge at the
        top of the saved image to the far edge at the bottom, columns along
        ``w``. Returned with the :class:`ShadowRaster` that says exactly where
        that canvas sits, so the caller places its plane on it.

        Parameters:
            meshes: iterable of ``(points, tris)`` — ``(N,3)`` world points and
                ``(M,3)`` vertex-index triangles.
            light: World position of a positional source (ignored with
                *direction*).
            ground: Height of the ground plane along the up axis.
            size: Square texture resolution.
            up: Index of the vertical axis (Maya 1, Blender 2).
            direction: Unit direction a directional source shines along.
            source_size: A positional source's diameter (world units), or a
                directional source's angular diameter (radians); 0 = a point
                source, sharp everywhere.
            max_stretch: Reach cap in object heights
                (:attr:`ShadowProjection.DEFAULT_MAX_STRETCH`).
            canvas: ``(u_lo, u_hi, w_lo, w_hi)`` to draw into a given canvas
                (a baked plane keeps its rect); None fits the canvas to the
                projected shadow plus its penumbra and *padding*.
            contact / radius / height: The model's inputs — the base centre,
                footprint radius and height of the occluder's bounding
                cylinder. Pass the values the live expression reads (its
                contact handle, its stamped constants) so the canvas
                fractions are measured in the frame that will place the
                plane; None derives them from the meshes' bounds.
            padding: Margin around a fitted canvas, as a fraction of its
                larger extent.
            uniform_alpha: Physically flat shadow (the default). False adds the
                stylised contact falloff (alpha fading from the footprint to
                the tip, shaped by *falloff_power* / *vertical_weight*).
            blur_amount: Edge anti-aliasing blur (pixels) applied before the
                penumbra.

        Returns:
            ``(rgba, raster)`` — a ``(size, size, 4)`` uint8 array (black RGB,
            shadow in alpha) and the :class:`ShadowRaster` it was drawn into.
        """
        return super().rasterize_shadow(
            meshes,
            light,
            ground,
            size,
            up=up,
            direction=direction,
            source_size=source_size,
            max_stretch=max_stretch,
            canvas=canvas,
            contact=contact,
            radius=radius,
            height=height,
            padding=padding,
            uniform_alpha=uniform_alpha,
            falloff_power=falloff_power,
            vertical_weight=vertical_weight,
            blur_amount=blur_amount,
        )

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

    @staticmethod
    def kelvin_to_linear_rgb(
        kelvin: float, normalize: bool = True
    ) -> Tuple[float, float, float]:
        """Blackbody colour temperature -> LINEAR RGB, normalised to max 1.0.

        Sits with the other colour-space conversions here rather than on the
        ``Color`` value type: it CONVERTS into a colour, it does not describe
        one. Deliberately pure stdlib -- no numpy, no Pillow -- so it still
        answers when this module's optional imports degraded to ``None``.

        For authoring light colour, where the value a renderer wants is linear
        and the number an artist thinks in is Kelvin -- 2700K domestic warm,
        3000K halogen, 3500-4100K office troffer, 5600K daylight, 6500K
        neutral white. Returned linear, not display-referred: a light colour
        picker's "warm white" is an sRGB-encoded value, and feeding that to a
        light makes it markedly too pale.

        Tanner Helland's piecewise fit to the Planckian locus (valid ~1000K to
        40000K, within a few percent over the range that matters here),
        converted from its sRGB output to linear. An **approximation**: hosts
        with their own blackbody conversion -- Blender's ``use_temperature``,
        for one -- will differ slightly, so a scene light authored at 5000K
        here and a host-side light set to 5000K are close but not identical.

        Parameters:
            kelvin: Colour temperature.
            normalize: Scale so the largest channel is 1.0 (the usual want:
                temperature sets the HUE and a separate intensity sets the
                level). ``False`` keeps the fit's own relative magnitudes.

        Returns:
            ``(r, g, b)`` linear floats.
        """
        return _ImgColorSpaceInternal.kelvin_to_linear_rgb(kelvin, normalize)

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
        scaled down (÷257), not truncated. A float source is handed to
        :meth:`convert_f_to_l`, whose 0..1 full scale this rule would flatten.

        Parameters:
            image (str/obj): An image or path to an image.

        Returns:
            (PIL.Image.Image) image in "L" mode.
        """
        im = cls.ensure_image(image)
        data = np.asarray(im)

        # Float data is the twin's job, and taking this name literally would
        # destroy it: 0..1 never exceeds 255, so it would skip the rescale and
        # round straight to 0/1. Dispatch on the dtype rather than trusting the
        # caller to have picked the matching entry point.
        if np.issubdtype(data.dtype, np.floating):
            return cls.convert_f_to_l(im)

        if data.dtype != np.uint8:
            if data.max(initial=0) > 255:  # 16-bit range -> scale, don't truncate
                data = np.clip(data, 0, 65535) / 257.0
            data = np.clip(np.round(data), 0, 255).astype(np.uint8)

        return Image.fromarray(data, mode="L")

    @classmethod
    def convert_f_to_l(cls, image):
        """Convert a float grayscale image to 8-bit 'L', rescaling the unit range.

        The float twin of :meth:`convert_i_to_l`, and the same trap: Pillow
        implements "F" -> "L" as a truncation, so a map whose data lives in
        0..1 -- the float convention, and what an EXR/float-TIFF height or
        displacement map carries -- collapses to two values, everything below
        1.0 becoming 0. 0..1 is that convention's full scale exactly as
        0..65535 is the 16-bit one, so it maps onto the whole 0..255.

        Out-of-range data is CLAMPED, not normalized: dividing by the actual
        maximum would silently tonemap, changing every texel's meaning to make
        one bright one fit. A genuine HDR image that needs its range preserved
        wants :meth:`encode_hdr_for_web` (which keeps the scalar so a viewer
        can multiply it back), not an 8-bit delivery map.

        An integer source is handed to :meth:`convert_i_to_l`, whose full scale
        the 0..1 clamp here would drive to white.

        Parameters:
            image (str/obj): An image or path to an image.

        Returns:
            (PIL.Image.Image) image in "L" mode.
        """
        im = cls.ensure_image(image)
        data = np.asarray(im)

        # Integer data is the twin's job, and taking this name literally would
        # destroy it: the 0..1 clamp below sends every value >= 1 to white.
        # Dispatch on the dtype rather than trusting the caller to have picked
        # the matching entry point.
        if not np.issubdtype(data.dtype, np.floating):
            return cls.convert_i_to_l(im)

        return Image.fromarray(
            np.round(np.clip(data, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L"
        )

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
        return super().pack_channels(
            channel_files,
            channels,
            out_mode,
            fill_values,
            output_path,
            output_format,
            grayscale_to_rgb,
            invert_channels,
            **kwargs,
        )

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
        return super().pack_channel_into_alpha(
            image,
            alpha,
            output_path,
            invert_alpha,
            resize_alpha,
            preserve_existing_alpha,
        )

    @classmethod
    def srgb_to_linear(cls, data):
        """Friendly wrapper: accepts PIL Image, numpy array, or list/tuple.

        - If Image: returns Image in the same mode (8-bit), converted to linear.
        - Otherwise: converts input to numpy, applies sRGB->linear, returns numpy array float32 in [0,1].
        """
        return super().srgb_to_linear(data)

    @classmethod
    def linear_to_srgb(cls, data):
        """Friendly wrapper: accepts PIL Image, numpy array, or list/tuple.

        - If Image: returns Image in the same mode (8-bit), converted to sRGB.
        - Otherwise: expects data in [0,1], returns numpy array float32 in [0,1].
        """
        return super().linear_to_srgb(data)

    @classmethod
    def convert_scene_linear(
        cls, image, src, dst="scene-linear Rec.709-sRGB", bgr=False
    ):
        """Re-express linear light from one colour space's primaries in another's.

        For renders made in a wide-gamut working space -- Maya renders in ACEScg
        by default, and so does what ``arnoldRenderToTexture`` writes -- whose
        consumers read linear Rec.709 (game engines, glTF viewers): left as is,
        an ACEScg pure red reads as (0.61, 0.07, 0.02) and every tint as less
        saturated than it is. White stays white in every space here.

        The spaces are Maya's default OCIO config's scene-linear ones, by their
        matrices to ACES2065-1 there: ``ACEScg``, ``ACES2065-1``,
        ``scene-linear Rec.709-sRGB``, ``scene-linear DCI-P3 D65``,
        ``scene-linear Rec.2020`` -- also under the ACES / OCIO v2 / Blender
        names for the same spaces (``ACES - ACEScg``, ``Utility - Linear -
        sRGB``, ``Linear Rec.709 (sRGB)``, ``lin_ap1``, ...), case-insensitive.

        Parameters:
            image: ``(..., C)`` array, ``C >= 3``; channels past the third
                (alpha) ride through untouched.
            src: The colour space *image* is in.
            dst: The one to return it in. Default linear Rec.709 / sRGB primaries.
            bgr: The first three channels are B, G, R (OpenCV's order).

        Returns:
            A float copy in *dst* -- or *image* itself when the two spaces are
            the same one.

        Raises:
            KeyError: *src* or *dst* is not a known scene-linear space (a log or
                display encoding is not a matrix away from linear light).
        """
        return super().convert_scene_linear(image, src, dst, bgr)

    @classmethod
    def quantize_8bit(cls, unit):
        """Quantize values in 0..1 to uint8 by stochastic rounding.

        ``floor(v * 255 + r)`` with one uniform ``r`` per texel, shared by its
        channels and drawn from a fixed seed: unbiased, so a smooth gradient
        keeps its mean instead of terracing into contour bands; a value already
        on a code stays exactly on it; the grain left is at most half a code
        and has no colour; and the same input always gives the same bytes.
        Measured on a production room's lightmaps at 2K, rounding to nearest
        terraced the walls' gradients into contour bands 0.5-1% apart.

        Parameters:
            unit: ``(..., C)`` values in 0..1, channels last (an ``(H, W, C)``
                image or a flat ``(N, C)`` pixel buffer).

        Returns:
            uint8 array of the same shape.
        """
        return super().quantize_8bit(unit)

    #: Percentile used to normalize an HDR image for 8-bit web encoding. High enough
    #: that only genuine light sources clip, low enough that one hot texel cannot
    #: crush the whole map. Shared contract with blendertk's ``encode_for_web``
    #: (bpy-I/O twin -- Blender ships no cv2); both test suites pin the same golden
    #: values so the implementations cannot drift.
    HDR_WEB_PERCENTILE = 99.5

    @classmethod
    def encode_hdr_for_web(cls, path, percentile=None):
        """Encode a linear-float HDR image (EXR/HDR) as sRGB PNG bytes for the web.

        The divisor is the *percentile* of the image's nonzero values, so the useful
        range fills the 8-bit encoding and only genuine light sources clip; the scalar
        is returned so a viewer can multiply it back and recover linear intensity
        (the ``lightmap_web`` manifest's per-material ``intensity``).

        Reads via cv2 directly, NOT :meth:`load_image` -- that path clips to 0-1 and
        quantizes to 8-bit (its docstring says so), which would destroy exactly the
        range this normalization exists to keep. The divide happens BEFORE
        :meth:`linear_to_srgb`, which clips.

        The 8-bit step is stochastic rounding (a fixed seed, one draw per texel
        for all its channels): unbiased, so a smooth wall keeps its gradient
        instead of terracing into contour bands, with at most half a code of
        colourless grain; a value already on a code encodes exactly.

        Parameters:
            path: An EXR/HDR file (any float image cv2 can read works).
            percentile: Normalization percentile. Default :attr:`HDR_WEB_PERCENTILE`.

        Returns:
            ``(png_bytes, scalar)`` -- multiply the decoded colour by ``scalar`` to
            recover linear intensity.

        Raises:
            ImportError: cv2 unavailable (it is the only float-image reader here).
            ValueError: the file could not be read as an image.
        """
        return super().encode_hdr_for_web(path, percentile)

    @classmethod
    def encode_hdr_radiance(cls, path) -> bytes:
        """Encode a linear-float HDR image (EXR/HDR) as Radiance ``.hdr`` bytes.

        The lossless-in-range sibling of :meth:`encode_hdr_for_web`, for an
        image whose whole range is the point -- an environment map, where a
        light is fifty times its walls and an 8-bit encode would clip exactly
        what a reflection shows. RGBE: three 8-bit mantissas over a shared
        exponent, run-length encoded, which three.js' ``RGBELoader`` and every
        HDR tool read. Values stay as stored (linear; whatever colour space
        and unit the file is in); alpha is dropped, as are non-finite values
        and negatives, which RGBE cannot express.

        Parameters:
            path: An EXR/HDR file (any float image cv2 can read works).

        Returns:
            The Radiance file's bytes.

        Raises:
            ImportError: cv2 unavailable (it is the only float-image reader here).
            ValueError: the file could not be read or encoded.
        """
        return super().encode_hdr_radiance(path)

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

        A facade over :meth:`MapFactory.get_base_texture_name`, which owns the
        implementation: what counts as a map suffix is the texture engine's
        taxonomy, not a general image rule. Kept here because callers reach it
        on ``ImgUtils`` (and subclasses of it). They were twins once, and
        drifted: only the factory dropped the UDIM/UV-tile token first, so a
        tiled filename produced two base names depending on the entry point.

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
        # Deferred, and one of the two upward edges left here: the texture engine
        # imports ImgUtils at module load, so a top-level import would be a
        # cycle, and a generic ImgUtils consumer should not pay for the map
        # taxonomy.
        from pythontk.core_utils.engines.textures.map_factory import MapFactory

        return MapFactory.get_base_texture_name(
            filepath_or_filename, prefix=prefix, suffix=suffix
        )

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
        return super().extract_channels(
            image_path, channel_config, output_dir, base_name, save, **kwargs
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
        reports the opposite convention with full confidence. So a caller that
        must name a map's ABSOLUTE convention ranks filename evidence first
        (:meth:`MapFactory.detect_normal_map_format`'s normal handler), while
        one that needs only the relative handedness -- the UV transfer, which
        turns X and Y with each island -- ranks this first
        (:meth:`UvTransfer.normal_convention`).

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
        return super().detect_normal_map_format(image, threshold, min_gradient_std)


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
