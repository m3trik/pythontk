# !/usr/bin/python
# coding=utf-8
"""The per-format write/read backends ``save_image`` / ``load_image`` dispatch to:
block-compressed DDS, KTX2, 16-bit PNG, OpenCV float formats (EXR, HDR), and
the lossy-encoder settings -- plus OpenCV file IO that opens any path
(``cv2_read`` / ``cv2_write`` / ``cv2_temp``).
"""

from __future__ import annotations

import os
import struct
import threading
from contextlib import contextmanager, suppress
from typing import Optional, Sequence

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore
try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


class _ImgCodecInternal:
    """Per-format encoders and decoders behind :class:`ImgUtils`' IO.

    Reached through :meth:`ImgUtils.save_image`, :meth:`ImgUtils.load_image`
    and the ``ImgUtils.cv2_*`` file IO (an ``ImgUtils`` base, so the helpers
    resolve as ``ImgUtils._<name>`` and read the format table and encoder
    registry off ``cls``); nothing here is called directly.
    """

    @classmethod
    @contextmanager
    def _sized_encoder_buffer(cls, im: "Image.Image", ext: str, kwargs: dict):
        """Widen Pillow's optimize buffer for the duration of one JPEG save.

        In optimize/progressive mode libjpeg needs a single buffer big enough
        for the WHOLE encoded image, and Pillow guesses its size from the pixel
        count -- ``2*w*h`` at quality >= 95, ``w*h`` below. That guess assumes
        4:2:0 chroma. :meth:`_apply_lossy_kwargs` writes **4:4:4**
        (``subsampling=0``) so a normal map's X/Y vectors are not turned to
        mush, and full-resolution chroma on a high-frequency map encodes past
        the guess -- at which point Pillow raises ``OSError: broken data stream
        when writing image file`` rather than growing the buffer.

        Not hypothetical: ``MapOptimizer.optimize_map`` passes ``optimize=True``
        on every save, so this is the .jpg path of the texture optimizer
        failing on exactly the detailed maps it exists to process. Measured on
        random-noise RGB at q95/4:4:4 -- 256^2 encoded to 159 KB against a
        131 KB budget, 1024^2 to 2.48 MB against 2 MB.

        ``ImageFile._save`` takes ``max(MAXBLOCK, bufsize)``, so raising that
        module global is the lever Pillow offers. The bound is the raw pixel
        size (channels * w * h) plus slack for headers: a JPEG that big would
        mean the encoder expanded the image, which it does not do at any
        quality. Restored in ``finally`` -- it is process-wide state and this
        is a library, not an application.
        """
        optimizing = kwargs.get("optimize") or kwargs.get("progressive")
        # ALWAYS_LOSSY_FORMATS is exactly the JPEG pair, and it is the set this
        # buffer problem belongs to: a container with no lossless mode is the
        # one whose writer takes the quality/subsampling that overflows the
        # guess. Naming the pair a second time here would be two lists to keep
        # in step.
        if ext not in cls.ALWAYS_LOSSY_FORMATS or not optimizing or Image is None:
            yield
            return

        from PIL import ImageFile

        width, height = im.size
        needed = len(im.getbands()) * width * height + 65536
        previous = ImageFile.MAXBLOCK
        ImageFile.MAXBLOCK = max(previous, needed)
        try:
            yield
        finally:
            ImageFile.MAXBLOCK = previous

    @classmethod
    def _assert_webp_dimensions(cls, im: "Image.Image", name: str) -> None:
        """Raise a fix-shaped error when *im* exceeds WebP's encoder ceiling.

        Pillow surfaces the libwebp error, which states the limit but not what to
        do about it — and by then the caller has already paid for a full optimize
        pass. Fail here with both.
        """
        width, height = im.size
        if max(width, height) > cls.WEBP_MAX_DIMENSION:
            raise ValueError(
                f"Cannot write {name!r} as WebP: {width}x{height} exceeds the "
                f"format's {cls.WEBP_MAX_DIMENSION}px limit. Resize first "
                f"(e.g. max_size={cls.WEBP_MAX_DIMENSION}) or keep it as PNG."
            )

    @classmethod
    def _apply_lossy_kwargs(
        cls, ext: str, quality: Optional[int], kwargs: dict
    ) -> dict:
        """Resolve writer kwargs for a lossy container. Returns a new dict.

        ``quality is None`` means *lossless wherever the container offers it*.
        That default is the whole point: Pillow writes WebP at *lossy q80* unless
        told otherwise, so a caller who merely picked ".webp" off a format menu
        would silently ship a degraded normal map — the exact failure this
        parameter exists to make impossible to reach by accident.

        JPEG has no lossless mode, so None resolves to
        :attr:`JPEG_DEFAULT_QUALITY` at 4:4:4 instead. Explicit kwargs always
        win, so a caller who really wants ``subsampling=2`` can still say so.
        """
        kwargs = dict(kwargs)
        if ext == "webp":
            if quality is None:
                # In lossless mode Pillow reads ``quality`` as compression EFFORT,
                # not fidelity — 100 is the smallest file, not the best pixels.
                kwargs.setdefault("lossless", True)
                kwargs.setdefault("quality", 100)
            else:
                kwargs.setdefault("lossless", False)
                kwargs.setdefault("quality", int(quality))
        else:  # jpg / jpeg — lossy either way; only the amount is negotiable.
            kwargs.setdefault(
                "quality",
                cls.JPEG_DEFAULT_QUALITY if quality is None else int(quality),
            )
            kwargs.setdefault("subsampling", 0)  # 4:4:4 — see JPEG_DEFAULT_QUALITY
        return kwargs

    @classmethod
    def _save_dds_compressed(
        cls, im: "Image.Image", name: str, compression: str
    ) -> None:
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

    @classmethod
    def _save_ktx2(
        cls,
        im: "Image.Image",
        name: str,
        compression: Optional[str],
        quality: Optional[int],
        colorspace: Optional[str],
        uastc_rdo: Optional[float] = None,
        uastc_rdo_dictionary: Optional[int] = None,
    ) -> None:
        """Write *im* to ``.ktx2`` through the registered / built-in encoder.

        ``compression`` selects the Basis codec. The bare-call default is UASTC
        — with no map-type context the quality-safe codec is the only safe one;
        ``MapOptimizer.resolve_compression`` derives the right codec per map
        type for the texture pipeline. ``colorspace`` labels the transfer
        function (None = sRGB, the common case for a bare save); mip levels are
        always generated — a GPU-compressed texture cannot make its own at
        runtime. The RDO pair rides a UASTC encode only (ETC1S has no RDO
        stage), and only when set, so a registered encoder that models neither
        keyword still encodes everything else.
        """
        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        encoder = cls.resolve_ktx2_encoder(required=True)
        codec = (compression or "UASTC").upper()
        srgb = (colorspace or "sRGB").lower() != "linear"
        rdo = Ktx2Encoder.rdo_kwargs(uastc_rdo, uastc_rdo_dictionary)
        encoder.encode(
            im,
            name,
            codec=codec,
            srgb=srgb,
            mipmaps=True,
            quality=quality,
            **(rdo if codec == "UASTC" else {}),
        )

    @classmethod
    def _save_high_bit_depth(cls, im: "Image.Image", name: str, bit_depth: int) -> bool:
        """Write *im* at 16-bit. Returns True when handled, False when the request
        can't be honored (unsupported depth or container) — the caller then falls
        back to an 8-bit save. Either way the degrade is announced, never silent.

        Grayscale uses Pillow's ``I;16``; a colour PNG is written by
        :meth:`_write_png16` (standard library), a colour TIFF through OpenCV
        ``uint16`` (:meth:`cv2_write`). 8-bit sources are promoted (value*257);
        existing 16-bit data is preserved.

        Raises:
            OSError: the OpenCV TIFF write failed.
        """
        if bit_depth != 16:  # only 16 is supported here; 32-bit float = EXR/HDR.
            print(
                f"# ImgUtils: {bit_depth}-bit unsupported for {name}; saving as 8-bit."
            )
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

        # RGB / RGBA — Pillow has no 16-bit colour mode. A PNG is written with
        # the standard library (a data map a GPU must not sRGB-decode needs 16
        # bits on any machine, OpenCV or not); a TIFF through OpenCV uint16.
        rgb = im.convert("RGBA") if im.mode == "RGBA" else im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint16) * 257
        if ext == "png":
            cls._write_png16(arr, name)
            return True
        try:
            import cv2
        except ImportError:
            return False
        code = cv2.COLOR_RGBA2BGRA if rgb.mode == "RGBA" else cv2.COLOR_RGB2BGR
        # cv2 reports a failed write by RETURNING False: unchecked, a TIFF that
        # was never written read as saved.
        if not cls.cv2_write(name, cv2.cvtColor(arr, code)):
            raise OSError(f"Failed to write image: {name}")
        return True

    @staticmethod
    def _write_png16(arr: "np.ndarray", name: str) -> None:
        """Write a ``(height, width, 3 | 4)`` ``uint16`` array as a 16-bit
        RGB / RGBA PNG with the standard library alone: one IDAT, filter type
        0 on every row, big-endian samples as the format requires.
        """
        import zlib

        height, width, channels = arr.shape
        if channels not in (3, 4):
            raise ValueError(f"_write_png16: {channels} channels; expected 3 or 4.")
        samples = np.ascontiguousarray(arr, dtype=">u2").view(np.uint8)
        rows = samples.reshape(height, width * channels * 2)
        raw = np.concatenate([np.zeros((height, 1), np.uint8), rows], axis=1)

        def chunk(kind: bytes, data: bytes) -> bytes:
            body = kind + data
            crc = zlib.crc32(body) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + body + struct.pack(">I", crc)

        colour_type = 6 if channels == 4 else 2
        header = struct.pack(">IIBBBBB", width, height, 16, colour_type, 0, 0, 0)
        with open(name, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
            fh.write(chunk(b"IHDR", header))
            fh.write(chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)))
            fh.write(chunk(b"IEND", b""))

    @classmethod
    def _save_via_cv2(cls, im: "Image.Image", name: str) -> None:
        """Write a PIL image to a float format (EXR, HDR) via OpenCV.

        Pillow cannot encode these. The source PIL image is 8-bit, so values
        are normalized to 0-1 float32 (the inverse of :meth:`_load_via_cv2`).
        OpenEXR is enabled when pythontk is imported (``OPENCV_IO_ENABLE_OPENEXR``);
        written with :meth:`cv2_write`, so any path will do.

        Raises:
            ImportError: cv2 unavailable (it is the only writer for these).
            OSError: the write failed -- cv2 reports that by RETURNING False
                (no exception), so an unchecked call leaves the caller believing
                a file exists that was never created.
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
        # Half float for EXR: the source is 8-bit, and half's 11-bit mantissa
        # already exceeds that -- cv2's FP32 default would double the bytes for
        # no precision. cv2 RETURNS False rather than raising when the codec is
        # missing or the path is bad, so an unchecked call reports success with
        # no file on disk.
        params = (
            [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]
            if os.path.splitext(name)[1].lower() == ".exr"
            else []
        )
        if not cls.cv2_write(name, img_np, params):
            raise OSError(f"Failed to write image: {name}")

    @classmethod
    def _load_via_cv2(cls, filepath: str) -> "Image.Image":
        """Read a float format (EXR, HDR) via OpenCV and return an 8-bit PIL image.

        Values are clipped to 0-1 and scaled to 8-bit, so the result is
        preview-grade — lossy for true HDR data. Consumers needing float
        precision (e.g. lightmap baking) should read with :meth:`cv2_read`.
        OpenEXR is enabled when pythontk is imported (``OPENCV_IO_ENABLE_OPENEXR``).
        """
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                f"OpenCV (cv2) is required to read '{os.path.splitext(filepath)[1]}' files."
            ) from e

        img = cls.cv2_read(filepath, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
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

    # ------------------------------------------------------------------
    # OpenCV file IO by any path. On Windows cv2 opens files -- and the
    # scratch file its EXR/HDR codecs stage through the temp folder -- by a
    # narrow (ANSI) path, so a path with any character outside ASCII fails.
    # ------------------------------------------------------------------

    #: :meth:`cv2_temp`'s one swap of TMP/TEMP, shared by every block open
    #: at once (nested, or on an encode thread pool) and undone when the last
    #: ends. Written through ``_ImgCodecInternal`` by name: through ``cls`` it
    #: would shadow onto ``ImgUtils`` and leave a second, unguarded count.
    _cv2_temp_lock = threading.Lock()
    _cv2_temp_blocks = 0
    _cv2_temp_saved: Optional[dict] = None

    @staticmethod
    def _cv2_temp_short() -> str:
        """The folder OpenCV stages through for a block, spelled in ASCII.

        A folder of the ecosystem temp root (``TempArtifacts("cv2_temp")``),
        its 8.3 form where a component needs one (``AppLauncher.ansi_safe_path``,
        the one short-name helper): a scratch file a crashed codec leaves is
        then swept like any other temp entry, never stranded in the system
        temp dir. ``""`` when there is nothing to do: not Windows, a temp folder
        OpenCV can already open, or no ASCII spelling (a volume that keeps no
        short names). Asks what OpenCV asks (``GetTempPath``: ``TMP``, then
        ``TEMP``), not Python's ``tempfile``, whose cached answer can name
        another folder.
        """
        if os.name != "nt":
            return ""
        import ctypes

        buffer = ctypes.create_unicode_buffer(32768)
        if not ctypes.windll.kernel32.GetTempPathW(len(buffer), buffer):
            return ""
        if buffer.value.isascii():
            return ""
        from pythontk.core_utils.app_launcher import AppLauncher
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        try:
            folder = TempArtifacts("cv2_temp").dir_path(name="scratch")
        except OSError:
            return ""
        safe = os.fspath(AppLauncher.ansi_safe_path(folder, ascii_only=True))
        return safe if safe.isascii() else ""

    @staticmethod
    @contextmanager
    def cv2_temp():
        """Body of :meth:`ImgUtils.cv2_temp`."""
        shared = _ImgCodecInternal
        with shared._cv2_temp_lock:
            if not shared._cv2_temp_blocks:
                short = shared._cv2_temp_short()
                if short:
                    shared._cv2_temp_saved = {
                        key: os.environ.get(key) for key in ("TEMP", "TMP")
                    }
                    os.environ["TEMP"] = os.environ["TMP"] = short
            shared._cv2_temp_blocks += 1
        try:
            yield
        finally:
            with shared._cv2_temp_lock:
                shared._cv2_temp_blocks -= 1
                saved = shared._cv2_temp_saved
                if not shared._cv2_temp_blocks and saved is not None:
                    shared._cv2_temp_saved = None
                    for key, value in saved.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value

    @classmethod
    def cv2_read(cls, path, flags: Optional[int] = None):
        """Body of :meth:`ImgUtils.cv2_read`."""
        import cv2
        import numpy as np

        try:
            data = np.fromfile(os.fspath(path), dtype=np.uint8)
        except (OSError, ValueError):
            return None
        if not data.size:
            return None
        with cls.cv2_temp():
            try:
                return cv2.imdecode(data, cv2.IMREAD_COLOR if flags is None else flags)
            except cv2.error:
                return None

    @classmethod
    def cv2_write(cls, path, image, params: Optional[Sequence[int]] = None) -> bool:
        """Body of :meth:`ImgUtils.cv2_write`."""
        import cv2

        path = os.fspath(path)
        with cls.cv2_temp():
            ok, data = cv2.imencode(
                os.path.splitext(path)[1], image, [] if params is None else params
            )
        if not ok:
            return False
        try:
            fh = open(path, "wb")
        except OSError:
            return False
        try:
            with fh:
                fh.write(data.tobytes())
        except OSError:
            # A part-written file is no image: remove it, as cv2.imwrite does.
            with suppress(OSError):
                os.remove(path)
            return False
        return True
