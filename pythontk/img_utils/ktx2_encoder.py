#!/usr/bin/python
# coding=utf-8
"""KTX2 / Basis Universal encoding via KTX-Software's ``toktx`` (external binary).

KTX2 is the GPU-delivery container (glTF ``KHR_texture_basisu``, WebXR): unlike
PNG/WebP — which decode to raw RGBA on the GPU — a KTX2/Basis texture is
*transcoded* at load time to whichever block-compressed format the device
supports (ASTC on mobile/standalone VR, BC7/DXT on desktop, ETC2 as fallback)
and stays compressed in GPU memory. One transcoder handles both codecs, so the
per-map codec choice is a quality/size decision, never a compatibility one:

- **UASTC** — high quality (~8 bpp before Zstd supercompression). For normal
  maps, ORM/packed masks, and every linear data map, where ETC1S's palettised
  encoding bands visibly.
- **ETC1S** — low bitrate (~1 bpp). For perceptual sRGB maps (base color,
  emissive), the same set :meth:`MapRegistry.is_lossy_safe` admits to lossy
  container codecs, and for the same reason.

There is no pure-Python Basis encoder, so encoding shells out to ``toktx``
(ships with KTX-Software, https://github.com/KhronosGroup/KTX-Software/releases)
— the same pattern as ``AudioUtils``' ffmpeg dependency. Discovery is PATH,
then the conventional install locations, then the :class:`AppInstaller` managed
catalog; a caller can also pass an explicit binary path or register a custom
encoder via :meth:`ImgUtils.register_ktx2_encoder`.

Both codecs are 8-bit LDR: a 16-bit source is announced and reduced. Mip levels
are generated at encode time by default — a compressed texture cannot generate
its own mips at runtime, and glTF's ``KHR_texture_basisu`` requires level count
1 or a full pyramid.
"""
import logging
import os
import shutil
import struct
import subprocess
from typing import Dict, List, Optional, Union

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

logger = logging.getLogger(__name__)


class Ktx2Encoder:
    """Encode images to ``.ktx2`` (Basis Universal) by shelling out to ``toktx``.

    Deliberately dumb about *which* codec a given map should get — that policy
    lives with the map taxonomy (``MapOptimizer.resolve_compression`` /
    ``MeshConvert.optimize_glb_textures``); this class owns binary discovery,
    argument assembly, and the subprocess.

    Parameters:
        toktx: Explicit path to the ``toktx`` binary. None (default) discovers
            it — see :meth:`resolve_toktx`.
        zstd_level: Zstandard supercompression level applied to UASTC output
            (ETC1S carries its own BasisLZ supercompression). 1-22; 18 is the
            common delivery setting.
        etc1s_qlevel: ETC1S quality when the caller names none (1-255).
        etc1s_clevel: ETC1S compression effort (0-5). 2 trades encode time for
            size the way delivery pipelines usually want.
        uastc_quality: UASTC encode quality (0-4). 2 is the documented
            quality/speed balance.
        extra_args: Additional ``toktx`` arguments appended verbatim before the
            file arguments — the escape hatch for flags this class does not
            model (``--uastc_rdo_l``, ``--normalize``, …).
        timeout: Seconds before a ``toktx`` subprocess is killed and treated
            as a failure. A hung encoder (bad/corrupt input, a stuck child
            process) would otherwise block the calling thread forever — fatal
            for a DCC's single-threaded UI. 300s (5 min) comfortably covers a
            large texture at the slowest quality tier.
    """

    #: Codec vocabulary accepted by :meth:`encode` (and by
    #: ``OutputSpec.compression`` for ``ktx2`` targets).
    CODECS = ("ETC1S", "UASTC")

    #: KTX 2.0 file identifier — first 12 bytes of any valid output.
    KTX2_MAGIC = b"\xabKTX 20\xbb\r\n\x1a\n"

    #: The fixed-layout header fields that follow :attr:`KTX2_MAGIC`, in order:
    #: ``vkFormat``, ``typeSize``, ``pixelWidth``, ``pixelHeight``,
    #: ``pixelDepth`` — five little-endian u32s (KTX 2.0 §3.1). Everything
    #: after them is index/level data this class has no reason to parse.
    _HEADER_FIELDS = ("vk_format", "type_size", "width", "height", "depth")
    _HEADER_STRUCT = "<5I"

    #: Conventional install locations probed after PATH.
    _WINDOWS_INSTALL_PATHS = (
        r"C:\Program Files\KTX-Software\bin\toktx.exe",
        r"C:\Program Files (x86)\KTX-Software\bin\toktx.exe",
    )

    def __init__(
        self,
        toktx: Optional[str] = None,
        zstd_level: int = 18,
        etc1s_qlevel: int = 128,
        etc1s_clevel: int = 2,
        uastc_quality: int = 2,
        extra_args: tuple = (),
        timeout: Optional[float] = 300,
    ) -> None:
        self._toktx = toktx
        self.zstd_level = int(zstd_level)
        self.etc1s_qlevel = int(etc1s_qlevel)
        self.etc1s_clevel = int(etc1s_clevel)
        self.uastc_quality = int(uastc_quality)
        self.extra_args = tuple(extra_args)
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def resolve_toktx(cls, required: bool = False) -> Optional[str]:
        """Resolve the ``toktx`` executable: PATH, conventional install
        locations, then the :class:`AppInstaller` managed catalog.

        Parameters:
            required: If True, raises ``FileNotFoundError`` (with the install
                source) instead of returning None.

        Returns:
            Path to ``toktx``, or None when not found and *required* is False.
        """
        found = shutil.which("toktx")
        if found:
            return found

        for candidate in cls._WINDOWS_INSTALL_PATHS:
            if os.path.isfile(candidate):
                return candidate

        # Managed catalog only — no auto-download: KTX-Software's Windows
        # release is an NSIS installer, not an unzip-and-run archive, so
        # ``AppInstaller.ensure`` has nothing safe to do unattended.
        from pythontk.core_utils.app_installer import AppInstaller

        managed = AppInstaller.get_path("ktx-software", executable="toktx")
        if managed:
            return managed

        if required:
            raise FileNotFoundError(
                "KTX2 encoding requires 'toktx' (KTX-Software). Install it from "
                "https://github.com/KhronosGroup/KTX-Software/releases and ensure "
                "it is on PATH, or register a custom encoder via "
                "ImgUtils.register_ktx2_encoder()."
            )
        return None

    @classmethod
    def available(cls) -> bool:
        """True when a ``toktx`` binary is discoverable."""
        return cls.resolve_toktx() is not None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @classmethod
    def read_header(cls, path: str) -> Dict[str, int]:
        """Read the fixed-layout KTX 2.0 header of *path* — no transcoder needed.

        There is no pure-Python Basis decoder, so PIL cannot open a ``.ktx2``
        at all — yet the geometry of one is exactly what a delivery gate needs
        when it assesses the file an encode just wrote
        (:meth:`MapOptimizer.assess`). Everything up to ``pixelDepth`` sits at a
        fixed offset in every valid KTX2 file, so 32 bytes answer that without
        touching the payload.

        Parameters:
            path: Path to a ``.ktx2`` file.

        Returns:
            dict: ``{"vk_format", "type_size", "width", "height", "depth"}``.
            ``vk_format`` is 0 (``VK_FORMAT_UNDEFINED``) for a Basis-
            supercompressed payload — the real format is the transcode target
            chosen at load time, so it is reported raw rather than interpreted.

        Raises:
            ValueError: *path* is not a KTX2 file (identifier mismatch), or its
                header is truncated. Named rather than left to
                ``struct.error``, for the same reason the encode failures are.
        """
        size = len(cls.KTX2_MAGIC) + struct.calcsize(cls._HEADER_STRUCT)
        with open(path, "rb") as fh:
            head = fh.read(size)
        if not head.startswith(cls.KTX2_MAGIC):
            raise ValueError(
                f"Not a KTX2 file (identifier mismatch): '{path}'. Expected the "
                f"KTX 2.0 file identifier in the first {len(cls.KTX2_MAGIC)} bytes."
            )
        if len(head) < size:
            raise ValueError(
                f"Truncated KTX2 header for '{path}': {len(head)} of {size} bytes."
            )
        values = struct.unpack_from(cls._HEADER_STRUCT, head, len(cls.KTX2_MAGIC))
        return dict(zip(cls._HEADER_FIELDS, values))

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def args_for(
        self,
        source: str,
        output: str,
        codec: str = "UASTC",
        srgb: bool = True,
        mipmaps: bool = True,
        quality: Optional[int] = None,
    ) -> List[str]:
        """Assemble the full ``toktx`` command for one encode.

        Split from :meth:`encode` so the exact flag surface is testable without
        the binary. ``--t2`` is non-negotiable — without it toktx writes KTX1,
        which nothing in the glTF/WebXR chain reads.

        Parameters:
            source: Input image path (PNG/JPEG — what toktx reads).
            output: Output ``.ktx2`` path.
            codec: ``"ETC1S"`` or ``"UASTC"`` (case-insensitive).
            srgb: Label the transfer function sRGB (color maps) or linear
                (normal / data / packed maps). This writes the container's DFD
                — it labels the existing pixel values, it does not convert them
                — and the loader samples accordingly, so a wrong label shifts
                every texel.
            mipmaps: Generate the full mip pyramid at encode time.
            quality: Optional 1-100 quality mapped onto ETC1S ``--qlevel``
                (1-255). Ignored for UASTC, whose quality is the constructor's
                ``uastc_quality`` tier.

        Returns:
            list[str]: The complete argv, binary first.
        """
        codec_key = (codec or "").upper()
        if codec_key not in self.CODECS:
            raise ValueError(
                f"Unknown KTX2 codec {codec!r}: expected one of {self.CODECS}."
            )

        args = [self.resolve_toktx(required=True) if self._toktx is None else self._toktx]
        args += ["--t2", "--encode", codec_key.lower()]
        if mipmaps:
            args.append("--genmipmap")
        # Label only — pixels are already authored in this space. Primaries are
        # BT.709/sRGB for the whole PBR set, both color and data maps.
        args += ["--assign_oetf", "srgb" if srgb else "linear"]
        args += ["--assign_primaries", "srgb"]
        if codec_key == "ETC1S":
            # Integer mapping of the 1-100 dial onto qlevel's 1-255 — float
            # rounding here would make the argv depend on the platform's
            # round-half behavior.
            qlevel = (
                max(1, min(255, (int(quality) * 255 + 50) // 100))
                if quality is not None
                else self.etc1s_qlevel
            )
            args += ["--qlevel", str(qlevel), "--clevel", str(self.etc1s_clevel)]
        else:  # UASTC — fixed quality tier + Zstandard supercompression.
            args += ["--uastc_quality", str(self.uastc_quality)]
            if self.zstd_level:
                args += ["--zcmp", str(self.zstd_level)]
        args += list(self.extra_args)
        args += [output, source]
        return args

    def encode(
        self,
        source: Union[str, "Image.Image"],
        output: str,
        codec: str = "UASTC",
        srgb: bool = True,
        mipmaps: bool = True,
        quality: Optional[int] = None,
    ) -> str:
        """Encode *source* to *output* (``.ktx2``).

        Parameters:
            source: Image file path, or a ``PIL.Image.Image`` (staged to a
                scratch PNG for the encoder — toktx reads files, not pipes).
            output: Destination path; parent directories are created.
            codec, srgb, mipmaps, quality: See :meth:`args_for`.

        Returns:
            str: *output*, for chaining.

        Raises:
            FileNotFoundError: No ``toktx`` binary (message carries the fix).
            RuntimeError: toktx returned non-zero (message carries its stderr),
                or was killed after exceeding :attr:`timeout`.
        """
        out_dir = os.path.dirname(output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if PIL_AVAILABLE and Image is not None and isinstance(source, Image.Image):
            from pythontk.file_utils.temp_artifacts import TempArtifacts

            with TempArtifacts("ktx2_encode", policy="scoped") as tmp:
                staged = tmp.path(extension=".png")
                self._stage_image(source, staged)
                return self._run(staged, output, codec, srgb, mipmaps, quality)
        return self._run(str(source), output, codec, srgb, mipmaps, quality)

    def _stage_image(self, im: "Image.Image", path: str) -> None:
        """Write *im* to *path* as the 8-bit PNG toktx will read.

        Basis is 8-bit LDR, so anything wider has to be reduced here. Two of
        the three routes cannot come from the mode-only fallback table
        (``effective_mode``), because the right answer depends on more than the
        mode name:

        - **High-bit-depth integers** (``I``, ``I;16`` and friends) are
          RANGE-RESCALED. Pillow implements ``I;16`` -> ``L`` as a *clip* at
          255, not a rescale, so a plain ``convert`` turned a smooth 0..65535
          ramp into two values, 99.6% of them pure white (probe-proven
          2026-08-14) — a destroyed map handed to the encoder under a log line
          calling it a precision reduction. ``I`` isn't in the table at all, so
          it wasn't even reduced: a 16-bit PNG went to an 8-bit encoder.
        - **Palettised sources** go through :meth:`ImgUtils.depalettize_image`,
          the SSoT for the tRNS rule (``P`` + ``transparency`` -> RGBA). The
          table's mode-only ``"P" -> "RGB"`` row silently dropped palette
          alpha, and both Basis codecs carry alpha, so nothing forced it.

        Everything else takes the table row, and any reduction is announced
        rather than silent.

        Raises:
            ValueError: *im* is floating-point (mode ``"F"`` — an EXR/HDR
                source). Deliberately NOT a fallback-table row: an integer
                source has a known full-scale range, so it can be rescaled
                into 0-255 with the image intact, but ``F`` carries no such
                range — PIL's ``F`` -> ``L`` clips to 0-255, which on the 0..1
                data an EXR normally carries collapses the whole image to black
                or white. A named error beats both that and the bare
                ``OSError: cannot write mode F as PNG`` this used to hit
                three frames deeper, inside ``Image.save``.
        """
        from pythontk.img_utils._img_utils import ImgUtils

        if im.mode == "F":
            raise ValueError(
                "KTX2/Basis is 8-bit LDR and cannot carry floating-point pixel "
                "data (mode 'F'). Tonemap or normalize the source into an "
                "8-bit mode before encoding — converting automatically would "
                "clip the HDR range and destroy the image silently."
            )

        if im.mode in ("P", "PA"):
            staged = ImgUtils.depalettize_image(im)
        elif im.mode == "I" or im.mode.startswith("I;"):
            # Pillow's own ``I;16`` -> ``L`` is a CLIP at 255, which turns a
            # full-scale height ramp into a 99.6%-white card. ``convert_i_to_l``
            # owns the range rescale package-wide; a private ``point``-based
            # one here also crashed on ``I;16B`` (Pillow refuses ``point`` for
            # byte-order-qualified modes), which is what a big-endian 16-bit
            # TIFF opens as.
            staged = ImgUtils.convert_i_to_l(im)
        else:
            stored = ImgUtils.effective_mode(im.mode, "ktx2")
            staged = im.convert(stored) if stored != im.mode else im

        if staged.mode != im.mode:
            logger.info(
                "Ktx2Encoder: %s source stored as %s (Basis is 8-bit LDR).",
                im.mode,
                staged.mode,
            )
        staged.save(path)

    def _run(
        self,
        source: str,
        output: str,
        codec: str,
        srgb: bool,
        mipmaps: bool,
        quality: Optional[int],
    ) -> str:
        args = self.args_for(source, output, codec, srgb, mipmaps, quality)
        try:
            result = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"toktx timed out for '{source}' -> '{output}' after "
                f"{self.timeout}s (killed)."
            ) from exc
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()
            raise RuntimeError(
                f"toktx failed for '{source}' -> '{output}' "
                f"(exit {result.returncode}): {tail[-1] if tail else 'unknown error'}"
            )
        return output
