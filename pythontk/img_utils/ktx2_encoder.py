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
catalog — and, with ``auto_install=True``, a managed download of the pinned
release (:data:`KTX_SOFTWARE_VERSION`) behind the caller's consent policy; a
caller can also pass an explicit binary path or register a custom encoder via
:meth:`ImgUtils.register_ktx2_encoder`.

Both codecs are 8-bit LDR: a 16-bit source is announced and reduced. Mip levels
are generated at encode time by default — a compressed texture cannot generate
its own mips at runtime, and glTF's ``KHR_texture_basisu`` requires level count
1 or a full pyramid.
"""

import logging
import os
import platform
import shutil
import struct
import subprocess
from typing import Callable, Dict, List, Optional, Union

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

logger = logging.getLogger(__name__)

#: The KTX-Software release the managed install fetches (GitHub release tag
#: ``v<version>``). Bump deliberately: the ``toktx`` flag surface
#: :meth:`Ktx2Encoder.args_for` assembles is pinned to what this release accepts.
KTX_SOFTWARE_VERSION = "4.4.2"
_KTX_RELEASE_URL = (
    "https://github.com/KhronosGroup/KTX-Software/releases/download/"
    f"v{KTX_SOFTWARE_VERSION}/KTX-Software-{KTX_SOFTWARE_VERSION}"
)
_ARM64 = platform.machine().lower() in ("arm64", "aarch64")

#: :class:`AppInstaller` platform table for the managed install. Windows ships
#: only an NSIS installer (run silently into the tools dir -- the ``nsis``
#: type); Linux a plain tarball whose ``bin/toktx`` finds ``lib/libktx.so`` by
#: ``$ORIGIN`` rpath. macOS ships a ``.pkg`` that needs root to install, so it
#: is deliberately absent: a Mac user installs KTX-Software by hand and PATH
#: discovery picks it up.
KTX_SOFTWARE_PLATFORMS = {
    "windows": {
        "url": f"{_KTX_RELEASE_URL}-Windows-{'arm64' if _ARM64 else 'x64'}.exe",
        "type": "nsis",
    },
    "linux": {
        "url": f"{_KTX_RELEASE_URL}-Linux-{'arm64' if _ARM64 else 'x86_64'}.tar.bz2",
        "type": "tar.bz2",
    },
}

#: SHA-256 of each release asset in :data:`KTX_SOFTWARE_PLATFORMS`, keyed the way
#: :meth:`AppInstaller.ensure` reads it -- by PLATFORM. The URLs above are
#: arch-dependent, so each entry mirrors the same ``_ARM64`` conditional; a flat
#: one-digest-per-platform table would hard-fail every install on an arm64 machine.
#:
#: Windows is the reason this exists: its entry is typed ``nsis``, the one installer
#: type that RUNS the download instead of unpacking it, and re-runs it elevated on
#: WinError 740. Consent is obtained for KTX-Software; without a digest, what runs as
#: Administrator was never confirmed to BE KTX-Software. Pinning also means a
#: truncated transfer is rejected rather than executed.
#:
#: Taken from the v4.4.2 release assets (GitHub's published per-asset digest;
#: the Windows x64 entry additionally re-verified by downloading and hashing:
#: 6,417,024 bytes). Bumping KTX_SOFTWARE_VERSION MUST update all four.
_KTX_SHA256 = {
    "windows-x64": "1f323b0fec19794f5e6c0425a61d4b1da396872a10be862d105f4f4b2d2957fe",
    "windows-arm64": "86d6edba47f3df597f3b9bceda6e4da8b4205b43c8386519e1c0d2ce804c4284",
    "linux-x86_64": "a8781bad05f9624edbf910b7f258cd0a4ba7d3e63b49ecc0a0ab440bf6a0a245",
    "linux-arm64": "60382e7b842177b8048bd58ccdc770383f8ef65b94452a25d3afdb55f2405c5a",
}

KTX_SOFTWARE_SHA256 = {
    "windows": _KTX_SHA256["windows-arm64" if _ARM64 else "windows-x64"],
    "linux": _KTX_SHA256["linux-arm64" if _ARM64 else "linux-x86_64"],
}


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
        uastc_rdo: UASTC rate-distortion optimisation lambda
            (``--uastc_rdo_l``), or None for off: the blocks are steered toward
            what the Zstandard stage compresses, at a controlled quality cost.
            Measured on a 4K production set: ORM packs -30% at 1.0 (PSNR
            50/44/48 dB), a noisy normal map only -3.5%, encode 3-4x slower.
            toktx's range is 0.001-10, and at most
            :attr:`UASTC_RDO_NORMAL_MAX` for a normal map -- applying that is
            the caller's per-map policy (:meth:`rdo_for`); a per-call value
            overrides.
        uastc_rdo_dictionary: RDO dictionary size (``--uastc_rdo_d``), or None
            for toktx's own (:attr:`TOKTX_RDO_DICTIONARY`); see
            :meth:`args_for`. A per-call value overrides.
        extra_args: Additional ``toktx`` arguments appended verbatim before the
            file arguments — the escape hatch for flags this class does not
            model (``--normalize``, ``--uastc_rdo_b``, …).
        timeout: Seconds before a ``toktx`` subprocess is killed and treated
            as a failure. A hung encoder (bad/corrupt input, a stuck child
            process) would otherwise block the calling thread forever — fatal
            for a DCC's single-threaded UI. The default (:attr:`AUTO_TIMEOUT`)
            sizes it per encode from the image (:meth:`encode_timeout`); a
            number is used as given, and ``None`` waits forever.
    """

    #: Codec vocabulary accepted by :meth:`encode` (and by
    #: ``OutputSpec.compression`` for ``ktx2`` targets).
    CODECS = ("ETC1S", "UASTC")

    #: The RDO lambda a normal map should be capped at, whatever a caller asks
    #: for -- toktx's guidance: "for normal maps a good range is [.25,.75]".
    #: Which maps ARE normal maps is the caller's taxonomy (:meth:`rdo_for`).
    UASTC_RDO_NORMAL_MAX: float = 0.75
    #: toktx's own RDO dictionary size, used whenever none is passed
    #: (``toktx --help``, pinned with :data:`KTX_SOFTWARE_VERSION`).
    TOKTX_RDO_DICTIONARY: int = 4096
    #: The dictionary sizes toktx documents: ``[64, 65536]``. toktx does NOT
    #: enforce it -- 63 and 65537 encode with exit 0 -- so this is the only
    #: guard against a typo.
    RDO_DICTIONARY_RANGE = (64, 65536)

    #: FLOOR for one encode, in seconds: the historical flat budget, so no small
    #: map encodes on a shorter leash than it did.
    DEFAULT_TIMEOUT = 300
    #: Budget per megapixel of the source above that floor. The flat 300 s
    #: shipped two 4096 normal maps as PNG (2026-09-14): UASTC quality 2 + RDO
    #: 0.75 took 260 s with eight encodes sharing a 20-core host, and past 300 s
    #: with a test run on top -- ~15.5 s/MP under that contention. This is that
    #: with a ~4x margin, the asymmetry ``MeshConvert.conversion_timeout``
    #: settled on: a hung encode costs minutes, a spurious timeout ships the
    #: wrong deliverable.
    SECONDS_PER_MEGAPIXEL = 60.0
    #: ``timeout=AUTO_TIMEOUT`` (the default) derives each encode's budget from
    #: its pixels. Negative so it cannot collide with a real value, and unlike
    #: ``None`` it does not already mean "wait forever".
    AUTO_TIMEOUT = -1.0

    #: :class:`AppInstaller` catalog key of the managed install.
    TOOL_NAME = "ktx-software"

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
        uastc_rdo: Optional[float] = None,
        uastc_rdo_dictionary: Optional[int] = None,
        extra_args: tuple = (),
        timeout: Optional[float] = AUTO_TIMEOUT,
    ) -> None:
        self._toktx = toktx
        self.zstd_level = int(zstd_level)
        self.etc1s_qlevel = int(etc1s_qlevel)
        self.etc1s_clevel = int(etc1s_clevel)
        self.uastc_quality = int(uastc_quality)
        self.uastc_rdo = self._rdo_lambda(uastc_rdo)
        self.uastc_rdo_dictionary = self.rdo_dictionary(uastc_rdo_dictionary)
        self.extra_args = tuple(extra_args)
        self.timeout = timeout

    @classmethod
    def encode_timeout(cls, width: int, height: int) -> float:
        """Seconds to allow one encode of a *width* x *height* image:
        :attr:`DEFAULT_TIMEOUT`, or :attr:`SECONDS_PER_MEGAPIXEL` per megapixel
        when that is more."""
        return max(
            float(cls.DEFAULT_TIMEOUT),
            width * height / 1e6 * cls.SECONDS_PER_MEGAPIXEL,
        )

    def _timeout_for(self, source: str) -> Optional[float]:
        """:attr:`timeout`, or under :attr:`AUTO_TIMEOUT` the budget for the
        file toktx is handed (its size read from the header; an unreadable one
        gets the floor and toktx reports what is wrong with it)."""
        if self.timeout is None or self.timeout >= 0:
            return self.timeout
        if Image is None:
            return float(self.DEFAULT_TIMEOUT)
        try:
            with Image.open(source) as im:
                return self.encode_timeout(*im.size)
        except Exception:  # noqa: BLE001 -- toktx's error is the useful one
            return float(self.DEFAULT_TIMEOUT)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @property
    def toktx(self) -> str:
        """The binary this encoder will actually run — bound path, else discovery.

        The read-only accessor for what was previously an inline
        ``self.resolve_toktx(required=True) if self._toktx is None else
        self._toktx`` at the one call site that needed it, and a private
        ``_toktx`` reach for everyone else. It matters that this prefers the
        BOUND path: an encoder handed back by
        ``ImgUtils.resolve_ktx2_encoder(auto_install=True)`` carries the path
        the install just wrote, and a fresh discovery pass can miss a
        catalog entry written moments earlier — the reason that resolver binds
        rather than re-discovers in the first place.

        Raises:
            FileNotFoundError: Nothing is bound and none can be discovered.
        """
        return (
            self._toktx
            if self._toktx is not None
            else self.resolve_toktx(required=True)
        )

    @classmethod
    def not_installed_error(cls, detail: str = "") -> FileNotFoundError:
        """The fix-shaped error every "no toktx" outcome raises: install
        source, the auto-install switch, and the registration seam."""
        return FileNotFoundError(
            f"KTX2 encoding requires 'toktx' (KTX-Software){detail}. Install it "
            "from https://github.com/KhronosGroup/KTX-Software/releases and "
            "ensure it is on PATH, pass auto_install=True to download it, or "
            "register a custom encoder via ImgUtils.register_ktx2_encoder()."
        )

    @classmethod
    def resolve_toktx(
        cls,
        required: bool = False,
        auto_install: bool = False,
        prompt: Union[bool, Callable[[str], bool]] = True,
    ) -> Optional[str]:
        """Resolve the ``toktx`` executable: PATH, conventional install
        locations, the :class:`AppInstaller` managed catalog -- then, with
        *auto_install*, a managed download of :data:`KTX_SOFTWARE_VERSION`.

        Parameters:
            required: If True, raises ``FileNotFoundError`` (with the install
                source) instead of returning None.
            auto_install: Permit downloading KTX-Software into the managed
                tools dir when discovery finds nothing. On Windows the release
                is an installer that requests administrator rights, so a UAC
                prompt follows the download.
            prompt: Consent policy for that download -- ``True`` asks on the
                console, ``False`` needs none, a callable ``(question) -> bool``
                is asked instead (a panel passes its dialog). See
                :meth:`AppInstaller.consent`.

        Returns:
            Path to ``toktx``, or None when not found and *required* is False.
        """
        found = shutil.which("toktx")
        if found:
            return found

        for candidate in cls._WINDOWS_INSTALL_PATHS:
            if os.path.isfile(candidate):
                return candidate

        from pythontk.core_utils.app_installer import AppInstaller

        managed = AppInstaller.get_path(
            cls.TOOL_NAME, executable="toktx", add_to_path=True
        )
        if managed:
            return managed

        if not auto_install:
            if required:
                raise cls.not_installed_error()
            return None

        question = (
            f"KTX-Software v{KTX_SOFTWARE_VERSION} (toktx) is not installed.\n"
            "Download and install it into the pythontk tools folder now?"
        )
        if platform.system().lower() == "windows":
            question += (
                "\n\nWindows will ask for administrator approval: the "
                "KTX-Software installer requires it."
            )
        answer = AppInstaller.consent(prompt, question)
        if answer is None:
            if required:
                raise cls.not_installed_error(
                    ", and no interactive console is available to confirm the "
                    "download (pass prompt=False to install non-interactively)"
                )
            return None
        if not answer:
            if required:
                raise cls.not_installed_error(" and the download was declined")
            return None

        try:
            return AppInstaller.ensure(
                cls.TOOL_NAME,
                platforms=KTX_SOFTWARE_PLATFORMS,
                executable="toktx",
                version=KTX_SOFTWARE_VERSION,
                sha256=KTX_SOFTWARE_SHA256,
            )
        except (RuntimeError, OSError, LookupError) as exc:
            logger.warning(f"KTX-Software install failed: {exc}")
            if required:
                raise cls.not_installed_error(
                    f" and the managed install failed: {exc}"
                ) from exc
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

    @staticmethod
    def _rdo_lambda(value: Optional[float]) -> Optional[float]:
        """*value* as an RDO lambda, None for off (``None`` / ``0``).

        Raises:
            ValueError: outside toktx's ``[0.001, 10]``.
        """
        if not value:
            return None
        rdo = float(value)
        if not 0.001 <= rdo <= 10.0:
            raise ValueError(
                f"uastc_rdo must be within 0.001-10 (toktx's range), got {value!r}."
            )
        return rdo

    @classmethod
    def rdo_for(
        cls, uastc_rdo: Optional[float], normal_map: bool = False
    ) -> Optional[float]:
        """The RDO lambda one UASTC encode takes: *uastc_rdo*, capped at
        :attr:`UASTC_RDO_NORMAL_MAX` when the map is a normal map; None = off.

        The cap is toktx's; deciding which maps are normal maps is the caller's
        (the texture taxonomy by map type, the GLB pass by material slot).
        """
        rdo = cls._rdo_lambda(uastc_rdo)
        if rdo is None:
            return None
        return min(rdo, cls.UASTC_RDO_NORMAL_MAX) if normal_map else rdo

    @staticmethod
    def rdo_kwargs(
        uastc_rdo: Optional[float] = None, uastc_rdo_dictionary: Optional[int] = None
    ) -> Dict[str, Union[float, int]]:
        """The RDO keywords one :meth:`encode` call should carry -- only those
        that apply: none without an RDO lambda (the dictionary means nothing
        then), and the dictionary only when one is chosen. An encoder
        registered through ``ImgUtils.register_ktx2_encoder`` need not model
        either keyword, so a call that uses neither must not pass them."""
        if not uastc_rdo:
            return {}
        kwargs: Dict[str, Union[float, int]] = {"uastc_rdo": uastc_rdo}
        if uastc_rdo_dictionary:
            kwargs["uastc_rdo_dictionary"] = uastc_rdo_dictionary
        return kwargs

    @classmethod
    def rdo_dictionary(cls, value: Optional[int]) -> Optional[int]:
        """Validate a UASTC RDO dictionary size (``--uastc_rdo_d``); None = toktx's own.

        The RDO pass rewrites UASTC blocks so the supercompressor finds more
        matches, and the dictionary bounds how far back it may look. A smaller
        window is dramatically cheaper and gives up some of the size win.

        Raises:
            ValueError: Outside :attr:`RDO_DICTIONARY_RANGE`, toktx's documented
                range (toktx itself accepts anything, silently).
        """
        if value is None:
            return None
        try:
            size = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"uastc_rdo_dictionary must be an integer, got {value!r}."
            ) from None
        low, high = cls.RDO_DICTIONARY_RANGE
        if not low <= size <= high:
            raise ValueError(
                f"uastc_rdo_dictionary must be within {low}-{high} (toktx's "
                f"documented range), got {value!r}."
            )
        return size

    def args_for(
        self,
        source: str,
        output: str,
        codec: str = "UASTC",
        srgb: bool = True,
        mipmaps: bool = True,
        quality: Optional[int] = None,
        uastc_rdo: Optional[float] = None,
        uastc_rdo_dictionary: Optional[int] = None,
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
            uastc_rdo: UASTC RDO lambda for THIS encode; None takes the
                constructor's, ``0`` switches it off. Ignored for ETC1S.
            uastc_rdo_dictionary: RDO dictionary size (``--uastc_rdo_d``,
                :attr:`RDO_DICTIONARY_RANGE`) for THIS encode; None takes the
                constructor's, and a constructor default of None leaves toktx
                to its own (:attr:`TOKTX_RDO_DICTIONARY`). The encode's dominant
                cost: on a production set's own maps (8 normals at 4K, 8 ORM
                packs at 2K) the GLB pass took 245 s at toktx's 4096 and 129 s
                at 1024, for +2.2% bytes. This class stays at toktx's own; a
                delivery policy that trades size for time chooses its size
                (``MeshConvert.WEB_DELIVERY_UASTC_RDO_DICTIONARY``). Ignored
                for ETC1S.

        Returns:
            list[str]: The complete argv, binary first.
        """
        codec_key = (codec or "").upper()
        if codec_key not in self.CODECS:
            raise ValueError(
                f"Unknown KTX2 codec {codec!r}: expected one of {self.CODECS}."
            )
        # Validated up front rather than where it is emitted: the flag is only
        # USED with an RDO pass on UASTC, but a malformed VALUE is a typo either
        # way, and swallowing it on the branches that ignore the option would
        # mean the constructor rejects `99` while a per-call `99` passes.
        dictionary = self.rdo_dictionary(
            self.uastc_rdo_dictionary
            if uastc_rdo_dictionary is None
            else uastc_rdo_dictionary
        )

        args = [self.toktx]
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
            rdo = self._rdo_lambda(self.uastc_rdo if uastc_rdo is None else uastc_rdo)
            if rdo:
                args += ["--uastc_rdo_l", f"{rdo:g}"]
                # Only meaningful WITH an RDO pass -- toktx rejects the
                # dictionary flag when RDO is off, so it is emitted inside the
                # same branch rather than beside it (validated above).
                if dictionary:
                    args += ["--uastc_rdo_d", str(dictionary)]
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
        uastc_rdo: Optional[float] = None,
        uastc_rdo_dictionary: Optional[int] = None,
    ) -> str:
        """Encode *source* to *output* (``.ktx2``).

        Parameters:
            source: Image file path, or a ``PIL.Image.Image`` (staged to a
                scratch PNG for the encoder — toktx reads files, not pipes).
            output: Destination path; parent directories are created.
            codec, srgb, mipmaps, quality, uastc_rdo, uastc_rdo_dictionary:
                See :meth:`args_for`.

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

        # Gate on the BINDING, not on PIL_AVAILABLE. Blender imports pythontk
        # before ensure_image_deps() can provision Pillow, so this module
        # caches Image = None; blendertk's _rebind_pil_globals repairs that
        # binding afterwards but cannot repair a bool, leaving PIL_AVAILABLE
        # False forever. Gating on it sent a live PIL image down the path
        # branch below, handing toktx str(source) -- an image repr. The bool
        # is left bound rather than deleted: it is part of the optional-dep
        # idiom blendertk's _rebind_pil_globals enumerates across seven
        # modules, and the regression test sets it False on purpose to pin
        # that reading it is no longer load-bearing.
        if Image is not None and isinstance(source, Image.Image):
            from pythontk.file_utils.temp_artifacts import TempArtifacts

            with TempArtifacts("ktx2_encode", policy="scoped") as tmp:
                staged = tmp.path(extension=".png")
                self._stage_image(source, staged)
                return self._run(
                    staged,
                    output,
                    codec,
                    srgb,
                    mipmaps,
                    quality,
                    uastc_rdo,
                    uastc_rdo_dictionary,
                )
        return self._run(
            str(source),
            output,
            codec,
            srgb,
            mipmaps,
            quality,
            uastc_rdo,
            uastc_rdo_dictionary,
        )

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
        uastc_rdo: Optional[float] = None,
        uastc_rdo_dictionary: Optional[int] = None,
    ) -> str:
        args = self.args_for(
            source,
            output,
            codec,
            srgb,
            mipmaps,
            quality,
            uastc_rdo=uastc_rdo,
            uastc_rdo_dictionary=uastc_rdo_dictionary,
        )
        timeout = self._timeout_for(source)
        try:
            result = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"toktx timed out for '{source}' -> '{output}' after "
                f"{timeout}s (killed)."
            ) from exc
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()
            raise RuntimeError(
                f"toktx failed for '{source}' -> '{output}' "
                f"(exit {result.returncode}): {tail[-1] if tail else 'unknown error'}"
            )
        return output
