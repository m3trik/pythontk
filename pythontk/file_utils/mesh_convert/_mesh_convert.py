# !/usr/bin/python
# coding=utf-8
import base64
import json
import logging
import os
import platform as _platform
import shlex
import shutil
import struct
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

from pythontk.core_utils.help_mixin import HelpMixin

logger = logging.getLogger(__name__)

#: What every GLB repair on :class:`MeshConvert` accepts as its target: a path,
#: or an already-open :class:`MeshConvert.GlbEdit` to join. The forward
#: reference is never resolved at runtime -- it exists so the signatures say
#: which of the two they take.
GlbTarget = Union[str, os.PathLike, "MeshConvert.GlbEdit"]

# godotengine/FBX2glTF — single-binary FBX -> glTF/GLB converter, the same
# tool Godot 4 uses internally for FBX import. Pinned to v0.13.1.
FBX2GLTF_VERSION = "0.13.1"
FBX2GLTF_PLATFORMS = {
    "windows": {
        "url": f"https://github.com/godotengine/FBX2glTF/releases/download/v{FBX2GLTF_VERSION}/FBX2glTF-windows-x86_64.zip",
        "type": "zip",
        "executable": "FBX2glTF-windows-x86_64",
    },
    "linux": {
        "url": f"https://github.com/godotengine/FBX2glTF/releases/download/v{FBX2GLTF_VERSION}/FBX2glTF-linux-x86_64.zip",
        "type": "zip",
        "executable": "FBX2glTF-linux-x86_64",
    },
    "darwin": {
        "url": f"https://github.com/godotengine/FBX2glTF/releases/download/v{FBX2GLTF_VERSION}/FBX2glTF-macos-x86_64.zip",
        "type": "zip",
        "executable": "FBX2glTF-macos-x86_64",
    },
}


class MeshConvert(HelpMixin):
    """3D mesh format conversion via the godotengine/FBX2glTF CLI.

    Currently supports static-mesh FBX -> GLB (binary glTF 2.0).
    The FBX2glTF binary is fetched on first use into the pythontk-managed
    tools directory under ``~/.pythontk/tools/`` (overridable via
    ``PYTHONTK_TOOLS_DIR``).

    Note: godotengine/FBX2glTF only ships an x86_64 build for macOS.
    Apple Silicon (arm64) Macs run it transparently via Rosetta 2,
    which must be installed (``softwareupdate --install-rosetta``).
    """

    TOOL_NAME = "fbx2gltf"
    DEFAULT_TIMEOUT = 300  # 5 minutes — enough for very large FBX files
    # Image types glTF 2.0 accepts natively. Anything else (TIFF, EXR, TGA —
    # all common in a DCC source tree) is re-encoded to PNG via Pillow when
    # available (see `_reencode_as_png`), and otherwise rejected by name
    # rather than written as an unloadable data URI.
    IMAGE_MIME_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }

    # ------------------------------------------------------------------ #
    # The open-GLB handle every repair below operates on
    # ------------------------------------------------------------------ #

    class GlbEdit:
        """A GLB parsed once and held open: read the JSON, edit, write once.

        Every repair on :class:`MeshConvert` edits the JSON chunk and nothing
        else, but each used to open, parse and rewrite the file for itself --
        and the preview path runs three of them back to back on the same GLB.
        A single push therefore read a file the size of its geometry three
        times over to change a handful of fields. This is the shared handle
        that collapses that to one read and one write; see
        :meth:`MeshConvert.open_glb` for how a caller joins several repairs
        onto one session.

        Two properties carry most of the saving:

        - :attr:`rest` -- everything after the JSON chunk, which in practice is
          the geometry -- is read **lazily**. The common outcome is a repair
          that finds nothing to change, and that path now never reads past the
          first few kilobytes of a file that is routinely hundreds of
          megabytes.
        - :attr:`bin_data` is a ``memoryview`` into :attr:`rest`, not a slice
          of it. A slice copies, which put peak memory at twice the file size
          to gain nothing: every consumer here only ever reads it.
        """

        #: Byte offset of the JSON chunk's payload -- the 12-byte file header
        #: plus the 8-byte chunk header. Fixed by the GLB spec, and the seek
        #: target for the in-place write in :meth:`MeshConvert._write_glb`.
        JSON_OFFSET = 12 + 8

        def __init__(
            self, path: str, version_bytes: bytes, gltf: dict, json_len: int
        ):
            self.path = path
            self.version_bytes = version_bytes
            self.gltf = gltf
            #: Length of the JSON chunk as it currently sits on disk. Updated
            #: by a full rewrite so a second write in the same session still
            #: knows where the chunk ends.
            self.json_len = json_len
            #: Set by any editor that changed :attr:`gltf`. Without it the
            #: close writes nothing, so a repair that matches nothing costs no
            #: I/O at all -- which is the usual case for the alpha repair that
            #: runs after every single conversion.
            self.dirty = False
            self._rest: Optional[bytes] = None
            self._bin: Optional[memoryview] = None
            self._alpha_extrema: Dict[int, Optional[Tuple[int, int]]] = {}
            #: Texture path -> texture index, shared by every channel writer on
            #: this session so a map used as both base colour and emissive is
            #: embedded once rather than once per channel.
            self.embedded: Dict[str, int] = {}

        @property
        def rest(self) -> bytes:
            """Every byte after the JSON chunk, read on first use and cached."""
            if self._rest is None:
                with open(self.path, "rb") as f:
                    f.seek(self.JSON_OFFSET + self.json_len)
                    self._rest = f.read()
            return self._rest

        @property
        def bin_data(self) -> Optional[memoryview]:
            """The BIN chunk's payload as a read-only view, or ``None``."""
            if self._bin is None:
                rest = self.rest
                if len(rest) >= 8 and rest[4:8] == b"BIN\x00":
                    length = struct.unpack("<I", rest[:4])[0]
                    self._bin = memoryview(rest)[8 : 8 + length]
                else:  # probed and absent; remembered so we don't re-probe
                    self._bin = memoryview(b"")
            return self._bin or None

        @property
        def materials(self) -> list:
            return self.gltf.get("materials", []) or []

        @property
        def textures(self) -> list:
            return self.gltf.get("textures", []) or []

        @property
        def images(self) -> list:
            return self.gltf.get("images", []) or []

        @property
        def buffer_views(self) -> list:
            return self.gltf.get("bufferViews", []) or []

        def base_color_image(self, mat: dict) -> Optional[int]:
            """Index of *mat*'s base-colour image, or ``None`` if it has none.

            The shared front half of both alpha repairs: material -> pbr ->
            baseColorTexture -> texture -> image, bounds-checked at each hop.
            Written out twice it had already drifted -- one copy bounds-checked
            the image index and the other left it to the probe to notice.

            Both bounds are checked, not just the upper one: glTF indices are
            non-negative by spec, so a negative one means a malformed file --
            and left to Python's own indexing it would quietly resolve to the
            *last* texture and report a finding against the wrong material
            rather than being skipped.
            """
            pbr = mat.get("pbrMetallicRoughness") or {}
            bct = pbr.get("baseColorTexture")
            if not bct:
                return None
            tex_idx = bct.get("index")
            textures = self.textures
            if not isinstance(tex_idx, int) or not 0 <= tex_idx < len(textures):
                return None
            img_idx = textures[tex_idx].get("source")
            if not isinstance(img_idx, int) or not 0 <= img_idx < len(self.images):
                return None
            return img_idx

        def image_label(self, img_idx: int) -> str:
            """How a finding or a fix record names an image.

            Its glTF name, else its uri, else its index -- a converter that
            carried none of the first two still has to produce something a
            reader can match against the file.
            """
            images = self.images
            entry = images[img_idx] if 0 <= img_idx < len(images) else {}
            return entry.get("name") or entry.get("uri") or f"image[{img_idx}]"

        def image_bytes(self, img_entry: dict) -> Optional[bytes]:
            """Raw bytes for a glTF image entry, or ``None`` if unavailable.

            Resolves all three ways an image can be stored: inline as a
            ``data:`` URI, as a file beside the GLB, or as a slice of the BIN
            chunk. Only the last touches :attr:`bin_data`, so a GLB whose
            images are all external never reads the geometry.
            """
            uri = img_entry.get("uri")
            if uri:
                if uri.startswith("data:"):
                    try:
                        _, b64 = uri.split(",", 1)
                        return base64.b64decode(b64)
                    except Exception:  # noqa: BLE001 — malformed URI, not fatal
                        return None
                sibling = os.path.join(os.path.dirname(self.path), uri)
                if os.path.isfile(sibling):
                    with open(sibling, "rb") as f:
                        return f.read()
                return None

            bv_idx = img_entry.get("bufferView")
            buffer_views = self.buffer_views
            # Two-sided, as in `base_color_image`: a negative index would slice
            # the wrong bufferView rather than be rejected.
            if not isinstance(bv_idx, int) or not 0 <= bv_idx < len(buffer_views):
                return None
            bin_data = self.bin_data
            if bin_data is None:
                return None
            bv = buffer_views[bv_idx]
            offset = bv.get("byteOffset", 0)
            length = bv.get("byteLength", 0)
            return bin_data[offset : offset + length] or None

        def alpha_extrema(self, img_idx: int) -> Optional[Tuple[int, int]]:
            """``(min, max)`` of an image's alpha channel, or ``None``.

            ``None`` covers every case a caller must not act on: an index out
            of range, bytes that could not be resolved, a decoder that refused
            the file, and an image with no alpha channel at all -- they are
            deliberately not distinguished, because the answer to each is the
            same "leave this material alone".

            Cached on the session, so an atlas shared by twenty materials is
            decoded once -- and now once across *both* alpha repairs rather
            than once inside each, which is what the two private caches this
            replaced could not do.

            Pillow is imported here rather than at module scope; the public
            entry points check for it up front and raise something actionable.
            """
            if img_idx in self._alpha_extrema:
                return self._alpha_extrema[img_idx]

            from io import BytesIO

            from PIL import Image

            result: Optional[Tuple[int, int]] = None
            images = self.images
            if img_idx < len(images):
                raw = self.image_bytes(images[img_idx])
                if raw:
                    try:
                        with Image.open(BytesIO(raw)) as im:
                            im.load()
                            has_alpha = im.mode in ("RGBA", "LA", "PA") or (
                                im.mode == "P" and "transparency" in im.info
                            )
                            if has_alpha:
                                result = (
                                    im.convert("RGBA").getchannel("A").getextrema()
                                )
                    except Exception as exc:  # noqa: BLE001 — varied decoder errors
                        logger.debug(
                            "GLB alpha probe: skipped image %s (%s)", img_idx, exc
                        )
            self._alpha_extrema[img_idx] = result
            return result

    @classmethod
    def _platform_exe_name(cls) -> str:
        """Return the FBX2glTF binary name for the current platform."""
        plat = _platform.system().lower()
        info = FBX2GLTF_PLATFORMS.get(plat)
        if not info:
            raise LookupError(f"FBX2glTF: unsupported platform '{plat}'")
        return info["executable"]

    @classmethod
    def resolve_binary(
        cls,
        required: bool = True,
        auto_install: bool = False,
        prompt: bool = True,
    ) -> Optional[str]:
        """Resolve the FBX2glTF executable from PATH or managed installs.

        Parameters:
            required:      Raise FileNotFoundError when missing.
            auto_install:  Download FBX2glTF if not found.
            prompt:        Ask before downloading (TTY only; non-TTY proceeds).

        Returns:
            Absolute path to FBX2glTF executable, or None.
        """
        platform_exe = cls._platform_exe_name()
        # Try platform-specific binary name first (matches release zip),
        # then plain "FBX2glTF" for users who renamed it.
        for candidate in (platform_exe, "FBX2glTF"):
            on_path = shutil.which(candidate)
            if on_path:
                return on_path

        from pythontk.core_utils.app_installer import AppInstaller

        managed = AppInstaller.get_path(
            cls.TOOL_NAME, executable=platform_exe, add_to_path=True
        )
        if managed:
            return managed

        if not auto_install:
            if required:
                raise FileNotFoundError(
                    f"FBX2glTF not found on PATH (looked for {platform_exe!r}). "
                    "Pass auto_install=True to download it."
                )
            return None

        if prompt:
            if not (sys.stdin and sys.stdin.isatty()):
                # No interactive console (CI, GUI host, pythonw.exe, etc.).
                # Refuse to silently download — caller must opt-in via prompt=False.
                if required:
                    raise FileNotFoundError(
                        "FBX2glTF is not installed and no interactive console "
                        "is available to confirm the download. Pass "
                        "prompt=False to install non-interactively."
                    )
                return None
            sys.stdout.write(
                f"\nFBX2glTF v{FBX2GLTF_VERSION} is not installed. "
                f"Download to ~/.pythontk/tools/ now? [y/N] "
            )
            sys.stdout.flush()
            answer = sys.stdin.readline().strip().lower()
            if answer not in ("y", "yes"):
                if required:
                    raise FileNotFoundError("User declined FBX2glTF installation.")
                return None

        try:
            return AppInstaller.ensure(
                cls.TOOL_NAME,
                platforms=FBX2GLTF_PLATFORMS,
                executable=platform_exe,
                version=FBX2GLTF_VERSION,
            )
        except (RuntimeError, OSError, LookupError) as exc:
            if required:
                raise
            logger.warning("FBX2glTF install failed: %s", exc)
            return None

    @classmethod
    def fbx_to_glb(
        cls,
        src: str,
        dst: Optional[str] = None,
        *,
        overwrite: bool = False,
        auto_install: bool = True,
        prompt: bool = True,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        extra_args: Optional[List[str]] = None,
    ) -> str:
        """Convert an FBX file to a binary glTF 2.0 (GLB) file.

        Parameters:
            src:           Input FBX path.
            dst:           Output GLB path. Defaults to src with .glb extension.
                           ``.glb`` is appended if absent.
            overwrite:     Replace existing destination.
            auto_install:  Download FBX2glTF if missing.
            prompt:        Ask before downloading.
            timeout:       Subprocess timeout in seconds. None disables.
            extra_args:    Extra CLI flags forwarded to FBX2glTF
                           (e.g. ``["--draco"]``, ``["-v"]``).

        Returns:
            Absolute path to the written GLB file.
        """
        src_abs = os.path.abspath(src)
        if not os.path.isfile(src_abs):
            raise FileNotFoundError(f"FBX source not found: {src_abs}")
        if os.path.splitext(src_abs)[1].lower() != ".fbx":
            raise ValueError(f"Expected .fbx input, got: {src_abs}")

        if dst is None:
            dst = os.path.splitext(src_abs)[0] + ".glb"
        elif not dst.lower().endswith(".glb"):
            dst = dst + ".glb"
        dst_abs = os.path.abspath(dst)

        if os.path.exists(dst_abs):
            if not overwrite:
                raise FileExistsError(
                    f"GLB output already exists: {dst_abs}. "
                    "Pass overwrite=True to replace."
                )
            os.remove(dst_abs)

        os.makedirs(os.path.dirname(dst_abs) or ".", exist_ok=True)

        binary = cls.resolve_binary(
            required=True, auto_install=auto_install, prompt=prompt
        )

        # FBX2glTF wants the output base WITHOUT extension; --binary forces .glb
        output_base = os.path.splitext(dst_abs)[0]
        cmd = [binary, "-i", src_abs, "-o", output_base, "--binary"]
        if extra_args:
            cmd.extend(extra_args)

        logger.debug("FBX2glTF: %s", shlex.join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"FBX2glTF timed out after {timeout}s converting {src_abs}"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"FBX2glTF failed (exit={result.returncode}):\n"
                f"  cmd: {shlex.join(cmd)}\n"
                f"  stdout: {result.stdout}\n"
                f"  stderr: {result.stderr}"
            )
        if not os.path.isfile(dst_abs):
            raise RuntimeError(
                f"FBX2glTF exited 0 but {dst_abs} was not created.\n"
                f"  stdout: {result.stdout}"
            )

        try:
            fixes = cls.fix_glb_phantom_opaque_alpha(dst_abs)
            for fx in fixes:
                logger.info(
                    "fix_glb_phantom_opaque_alpha: %s baseColorFactor[3] %.3f -> %.3f (image: %s)",
                    fx["material"],
                    fx["old_alpha"],
                    fx["new_alpha"],
                    fx["image"],
                )
        except Exception as exc:  # noqa: BLE001 — never let post-process kill a successful conversion
            logger.warning("fix_glb_phantom_opaque_alpha skipped: %s", exc)

        return dst_abs

    # ------------------------------------------------------------------ #
    # Post-conversion material sanity check
    # ------------------------------------------------------------------ #

    @classmethod
    def check_glb_materials(cls, glb: GlbTarget) -> List[Dict[str, str]]:
        """Inspect a GLB for materials flagged transparent that should be opaque.

        Catches the Maya/Stingray/OpenPBR/Standard-Surface failure mode where
        a color texture happens to carry an alpha channel (often PNG palette
        transparency) without any actual transparency intent. Maya's FBX
        exporter writes a TransparencyFactor; FBX2glTF then sets
        ``alphaMode: BLEND`` and the renderer disables depth-write —
        producing the "inverted face" / wrong-render-order artifact.

        A material is flagged when its ``alphaMode`` is BLEND or MASK *and*
        its base-color texture's alpha channel is uniformly 255. Genuine
        transparency (varying alpha) is not reported.

        Parameters:
            glb: Path to a binary glTF (.glb) file, or an open
                :class:`GlbEdit` session to inspect. Read-only either way.

        Returns:
            List of findings. Each finding is a dict with keys:
                material   — material name (or '<material[i]>')
                alpha_mode — "BLEND" or "MASK"
                image      — image name / uri / fallback id
                reason     — short human-readable explanation
        """
        try:
            # Presence check only — the session does the decoding. Imported as
            # `from PIL import Image` rather than `import PIL` because the dead
            # original PIL also ships the package name and not the module.
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "check_glb_materials requires Pillow (PIL). Install it with "
                "`pip install pillow`."
            ) from exc

        # Reason text per alpha mode — BLEND and MASK fail in different ways.
        REASONS = {
            "BLEND": (
                "alphaMode=BLEND but base-color alpha is uniformly opaque (255). "
                "Renderers disable depth-write for BLEND, causing render-order "
                "artifacts (faces drawing in the wrong order)."
            ),
            "MASK": (
                "alphaMode=MASK but base-color alpha is uniformly opaque (255). "
                "Every fragment passes the cutoff so alpha-testing is a no-op; "
                "the material should be OPAQUE."
            ),
        }

        findings: List[Dict[str, str]] = []
        with cls.open_glb(glb) as edit:
            for mi, mat in enumerate(edit.materials):
                alpha_mode = mat.get("alphaMode", "OPAQUE")
                if alpha_mode not in REASONS:  # OPAQUE or unknown — skip
                    continue

                # Real transparency can come from the scalar baseColorFactor[3];
                # don't flag those as "accidentally transparent".
                pbr = mat.get("pbrMetallicRoughness") or {}
                bc_factor = pbr.get("baseColorFactor")
                if bc_factor and len(bc_factor) >= 4 and bc_factor[3] < 1.0:
                    continue

                img_idx = edit.base_color_image(mat)
                if img_idx is None:
                    continue

                # Decoded once per source image even if many materials share
                # it, and once across every repair sharing this session.
                if edit.alpha_extrema(img_idx) != (255, 255):
                    continue

                findings.append(
                    {
                        "material": mat.get("name") or f"<material[{mi}]>",
                        "alpha_mode": alpha_mode,
                        "image": edit.image_label(img_idx),
                        "reason": REASONS[alpha_mode],
                    }
                )

        return findings

    @classmethod
    def fix_glb_phantom_opaque_alpha(cls, glb: GlbTarget) -> List[Dict]:
        """Repair the Maya phong → FBX → FBX2glTF transparency translation bug.

        When a Maya phong/lambert/blinn shader has its ``.transparency`` fed
        by a file node's ``.outTransparency``, Maya's FBX exporter writes
        ``TransparencyFactor=1.0`` (the texture is meant to modulate
        per-pixel). FBX2glTF then computes
        ``baseColorFactor[3] = 1 - 1 = 0`` — multiplying every fragment's
        alpha by zero and rendering the mesh fully invisible regardless of
        texture content.

        A material is fixed when ALL of:
            - ``alphaMode`` is BLEND or MASK
            - ``baseColorFactor[3]`` is ~0
            - ``baseColorTexture`` exists and references an image with
              *varying* alpha (a real cutout mask, not uniformly 0 or 255)

        On match, ``baseColorFactor[3]`` is reset to 1.0 so per-pixel alpha
        from the texture controls visibility as intended.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.

        Returns:
            List of fix records. Empty when nothing matched. Each record:
                material   — material name
                old_alpha  — original baseColorFactor[3]
                new_alpha  — 1.0
                image      — the baseColorTexture image identifier
        """
        try:
            # Presence check only — the session does the decoding. Imported as
            # `from PIL import Image` rather than `import PIL` because the dead
            # original PIL also ships the package name and not the module.
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "fix_glb_phantom_opaque_alpha requires Pillow (PIL). "
                "Install it with `pip install pillow`."
            ) from exc

        EPSILON = 1e-4
        fixes: List[Dict] = []
        with cls.open_glb(glb) as edit:
            for mi, mat in enumerate(edit.materials):
                if mat.get("alphaMode") not in ("BLEND", "MASK"):
                    continue
                pbr = mat.get("pbrMetallicRoughness") or {}
                bcf = pbr.get("baseColorFactor")
                if not bcf or len(bcf) < 4 or bcf[3] > EPSILON:
                    continue
                img_idx = edit.base_color_image(mat)
                if img_idx is None:
                    continue
                # Every check above reads the JSON alone; this is the first
                # line that can pull the BIN chunk in, and it is reached only
                # by a material that already looks wrong. A GLB with nothing to
                # fix is therefore never read past its JSON chunk.
                extrema = edit.alpha_extrema(img_idx)
                # Skip uniform alpha (genuinely-transparent or genuinely-opaque
                # textures) — only varying alpha indicates a real cutout mask
                # whose per-pixel control was cancelled by baseColorFactor[3]=0.
                if extrema is None or extrema[0] == extrema[1]:
                    continue

                old_alpha = bcf[3]
                bcf[3] = 1.0
                pbr["baseColorFactor"] = bcf
                mat["pbrMetallicRoughness"] = pbr

                fixes.append(
                    {
                        "material": mat.get("name") or f"<material[{mi}]>",
                        "old_alpha": old_alpha,
                        "new_alpha": 1.0,
                        "image": edit.image_label(img_idx),
                    }
                )

            if fixes:
                edit.dirty = True

        return fixes

    @classmethod
    @contextmanager
    def open_glb(cls, glb: GlbTarget):
        """Yield an open :class:`GlbEdit` for *glb*, writing once on close.

        *glb* is a path, **or an already-open session** -- in which case it is
        yielded as-is and its owner keeps responsibility for the write. That
        second form is what lets these repairs compose: each one takes either,
        so running three against a path costs three read/write cycles, while
        wrapping the same three in one ``open_glb`` costs one::

            with MeshConvert.open_glb(path) as glb:
                MeshConvert.set_glb_base_color(glb, base_color)
                MeshConvert.set_glb_emissive(glb, emissive)

        Nothing is written when the body raises -- a half-applied edit must not
        reach disk -- nor when no editor set :attr:`GlbEdit.dirty`.
        """
        if isinstance(glb, cls.GlbEdit):
            yield glb
            return
        edit = cls._read_glb(os.fspath(glb))
        yield edit
        if edit.dirty:
            cls._write_glb(edit)

    @classmethod
    def _read_glb(cls, glb_path: str) -> "MeshConvert.GlbEdit":
        """Parse a GLB's container and JSON chunk into an open edit session.

        The single owner of GLB container parsing for this class — the JSON
        chunk is what every repair here edits, and re-deriving the offsets per
        function is how one of them ends up with a subtly different idea of
        where the BIN chunk starts.

        Only the JSON chunk is read: everything after it is pulled in on demand
        by :attr:`GlbEdit.rest`, because most repairs decide they have nothing
        to do from the JSON alone and the remainder is the whole geometry.
        """
        if not os.path.isfile(glb_path):
            raise FileNotFoundError(glb_path)

        with open(glb_path, "rb") as f:
            # Read the fixed 12-byte header and the 8-byte chunk header in one
            # go each and length-check them: a file truncated mid-write reaches
            # struct.unpack with a short buffer, and struct.error derives from
            # Exception — it slips past callers' (RuntimeError, ValueError,
            # OSError) per-file handlers and aborts a whole batch.
            header = f.read(12)
            if len(header) < 12 or header[:4] != b"glTF":
                raise ValueError(f"Not a GLB file: {glb_path}")
            version_bytes = header[4:8]  # total length is recomputed on write
            chunk0_header = f.read(8)
            if len(chunk0_header) < 8:
                raise ValueError(f"Malformed GLB: truncated chunk header ({glb_path})")
            chunk0_len = struct.unpack("<I", chunk0_header[:4])[0]
            chunk0_type = chunk0_header[4:]
            if chunk0_type != b"JSON":
                raise ValueError(f"Malformed GLB: first chunk not JSON ({glb_path})")
            payload = f.read(chunk0_len)
            # Same reasoning as the header checks, and newly reachable now that
            # the read stops at the chunk boundary instead of consuming the
            # file: a chunk header promising more JSON than the file holds must
            # surface as the ValueError callers already handle.
            if len(payload) < chunk0_len:
                raise ValueError(f"Malformed GLB: truncated JSON chunk ({glb_path})")
            gltf = json.loads(payload.decode("utf-8"))

        return cls.GlbEdit(glb_path, version_bytes, gltf, chunk0_len)

    @staticmethod
    def _write_glb(edit: "MeshConvert.GlbEdit") -> None:
        """Persist *edit*'s JSON chunk, rewriting the whole file only if it must.

        The JSON chunk is padded to a 4-byte boundary with spaces and the total
        length recomputed — both required by the GLB spec, and both easy to get
        wrong in a way that only some loaders reject.

        When the re-serialized JSON still fits the chunk it came from it is
        padded back out to *exactly* that length and written in place, so every
        offset after it — and the entire BIN chunk — is left untouched. That is
        the common case rather than a lucky one: the alpha repair changes a
        single float, and this writer serializes compactly while most producers
        do not. The alternative is pulling a few hundred megabytes of geometry
        through memory to edit a few bytes of JSON.

        Padding out to the original length rather than shrinking to fit is what
        keeps that safe: the chunk header, the total-length field and every
        byte beyond stay correct only if the chunk keeps its size.
        """
        new_json = json.dumps(edit.gltf, separators=(",", ":")).encode("utf-8")

        if len(new_json) <= edit.json_len:
            new_json += b" " * (edit.json_len - len(new_json))
            with open(edit.path, "r+b") as f:
                f.seek(edit.JSON_OFFSET)
                f.write(new_json)
            return

        new_json += b" " * ((4 - (len(new_json) % 4)) % 4)
        # Resolved before the truncating open, not inside it: `rest` is lazy,
        # and reading it back out of a file we have just emptied would hand the
        # writer nothing.
        rest = edit.rest
        with open(edit.path, "wb") as f:
            f.write(b"glTF")
            f.write(edit.version_bytes)
            f.write(struct.pack("<I", 12 + 8 + len(new_json) + len(rest)))
            f.write(struct.pack("<I", len(new_json)))
            f.write(b"JSON")
            f.write(new_json)
            f.write(rest)
        edit.json_len = len(new_json)

    @staticmethod
    def _match_glb_materials(
        gltf: dict, entries: Dict[str, Dict[str, Any]], caller: str
    ) -> List[Tuple[str, Dict[str, Any], dict]]:
        """Pair sidecar *entries* with the GLB materials they name.

        The shared front half of every by-name channel writer here: resolve
        each entry against the GLB's material list and report the misses in
        one loud line. Loud, with both name sets, because a name the converter
        renamed makes the write a total no-op and "my emissive is missing" is
        indistinguishable from "the channel was never read" unless the
        mismatch says so.
        """
        by_name = {
            m.get("name"): m for m in gltf.get("materials", []) or [] if m.get("name")
        }
        matched = [
            (name, spec, by_name[name])
            for name, spec in entries.items()
            if name in by_name
        ]
        unmatched = sorted(set(entries) - set(by_name))
        if unmatched:
            logger.warning(
                "%s: %s material(s) had no match in the GLB and were skipped: %s. "
                "GLB materials are: %s.",
                caller,
                len(unmatched),
                unmatched,
                sorted(by_name) or "<none>",
            )
        return matched

    @classmethod
    def set_glb_emissive(
        cls, glb: GlbTarget, emissive: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """Write emissive color / texture into a GLB's materials, by name.

        Repairs the channel Maya's FBX exporter simply drops. It maps emissive
        only for its own legacy shading models (lambert / blinn / phong via
        ``incandescence``); ``aiStandardSurface``, ``StingrayPBS`` and openPBR
        emission never reach the FBX at all, so the GLB has no ``emissiveFactor``
        to correct and the surface previews as unlit. Measured against Maya
        2025 + MtoA: lambert and blinn arrive with ``emissiveFactor``, the other
        two arrive with the key absent entirely.

        Emissive intensity above 1.0 is preserved rather than clipped, by
        normalizing the color and carrying the magnitude in
        ``KHR_materials_emissive_strength`` — glTF's base ``emissiveFactor`` is
        LDR, so a Maya emission of 5.0 would otherwise flatten to 1.0 and lose
        exactly the over-bright look that motivates using it. The extension is
        declared in ``extensionsUsed`` only when actually applied; loaders that
        don't implement it still get a sensible clamped color.

        Textures are embedded as ``data:`` URIs rather than appended to the BIN
        chunk. That keeps the edit inside the JSON chunk — no buffer offsets to
        recompute, which is the part of GLB surgery that silently corrupts a
        file — at the cost of base64's ~33% overhead, which a local preview can
        afford. Repeated paths are embedded once per session, so a map used as
        both base colour and emissive costs one copy, not one per channel.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            emissive: ``{material_name: {"color": (r, g, b), "texture": path}}``.
                Both keys optional; a texture with no color implies white.
                Names not present in the GLB are reported, not raised.

        Returns:
            List of records: ``material``, ``factor``, ``strength``, ``texture``.
        """
        if not emissive:
            return []

        records: List[Dict] = []
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            used_strength = False
            for name, spec, mat in cls._match_glb_materials(
                gltf, emissive, "set_glb_emissive"
            ):
                color = list(spec.get("color") or (1.0, 1.0, 1.0))[:3]
                if len(color) < 3:
                    color += [0.0] * (3 - len(color))

                # A zero colour is zero emission -- and paired with a texture it
                # is actively harmful rather than merely redundant: glTF
                # emission is ``emissiveFactor * emissiveTexture``, so writing
                # both multiplies the map to black while still reporting a
                # successful transfer. Tested before embedding, so a skipped
                # material leaves no orphaned image in the file.
                if all(c <= 0.0 for c in color):
                    continue

                peak = max(color)
                strength = None
                if peak > 1.0:
                    color = [c / peak for c in color]
                    strength = peak
                    mat.setdefault("extensions", {})[
                        "KHR_materials_emissive_strength"
                    ] = {"emissiveStrength": strength}
                    used_strength = True

                tex_path = spec.get("texture")
                tex_index = cls._embed_image(edit, tex_path) if tex_path else None
                if tex_index is not None:
                    mat["emissiveTexture"] = {"index": tex_index}

                mat["emissiveFactor"] = color

                records.append(
                    {
                        "material": name,
                        "factor": color,
                        "strength": strength,
                        "texture": tex_path if tex_index is not None else None,
                    }
                )

            if not records:
                # `dirty` is left as found rather than cleared: on a shared
                # session a sibling writer's edits may already be pending, and
                # this one having nothing to say is no reason to drop them.
                return []

            if used_strength:
                used = gltf.setdefault("extensionsUsed", [])
                if "KHR_materials_emissive_strength" not in used:
                    used.append("KHR_materials_emissive_strength")

            cls._prune_empty_containers(gltf)
            edit.dirty = True

        return records

    @staticmethod
    def _prune_empty_containers(gltf: dict) -> None:
        """Drop container arrays an edit left empty.

        Emitting ``"samplers": []`` is invalid per the glTF schema (minItems 1),
        and every channel writer can leave one behind when its textures all
        failed to embed -- so the cleanup belongs in one place rather than
        copied into the tail of each.
        """
        for key in ("images", "textures", "samplers"):
            if not gltf.get(key):
                gltf.pop(key, None)

    @classmethod
    def _embed_image(cls, edit: "MeshConvert.GlbEdit", path: str) -> Optional[int]:
        """Embed *path* as a ``data:`` URI image and return its texture index.

        Shared by every channel writer here. Data URIs keep the whole edit
        inside the JSON chunk -- no buffer offsets to recompute, which is the
        part of GLB surgery that silently corrupts a file -- at the cost of
        base64's ~33% overhead, which a local preview can afford.

        Repeated paths resolve to one image via the session's embed cache, so
        the sharing now spans every writer on that session rather than only the
        one call -- a map assigned as both base colour and emissive used to be
        base64'd into the file twice.
        """
        gltf, cache = edit.gltf, edit.embedded
        if path in cache:
            return cache[path]
        if not os.path.isfile(path):
            logger.warning("GLB texture embed: file not found: %s", path)
            return None
        mime = cls.IMAGE_MIME_TYPES.get(os.path.splitext(path)[1].lower())
        if mime is not None:
            # Guarded like the re-encode branch below, which this asymmetry had
            # left as the only safe one: a PNG that exists but cannot be read
            # (permissions, a network path that dropped, deleted between the
            # isfile check and here) raised straight out of the channel writer,
            # while the exact same failure on a TGA warned and skipped. It also
            # made this the one file operation left inside an applier, so a
            # locked texture could abort a whole sidecar section.
            try:
                with open(path, "rb") as f:
                    raw = f.read()
            except OSError as error:
                logger.warning("GLB texture embed: could not read %s: %s", path, error)
                return None
        else:
            # glTF accepts only PNG/JPEG, but TGA/TIFF/BMP are routine in a
            # DCC source tree -- TGA especially, all over game art -- and
            # written as-is they would be an unloadable data URI rather than a
            # reported failure. Re-encode to PNG when Pillow can read them;
            # only what it cannot (EXR, or no Pillow at all) is rejected.
            raw = cls._reencode_as_png(path)
            if raw is None:
                logger.warning("GLB texture embed: unsupported image type: %s", path)
                return None
            mime = "image/png"

        images = gltf.setdefault("images", [])
        textures = gltf.setdefault("textures", [])
        samplers = gltf.setdefault("samplers", [])
        data = base64.b64encode(raw).decode("ascii")
        images.append(
            {
                "name": os.path.basename(path),
                "uri": f"data:{mime};base64,{data}",
                "mimeType": mime,
            }
        )
        if not samplers:  # one repeat sampler is enough for a preview
            samplers.append({"wrapS": 10497, "wrapT": 10497})
        textures.append({"source": len(images) - 1, "sampler": 0})
        cache[path] = len(textures) - 1
        return cache[path]

    @staticmethod
    def _reencode_as_png(path: str) -> Optional[bytes]:
        """PNG bytes for an image glTF can't hold natively, or ``None``.

        Pillow is deliberately optional (this package's zero-dep rule): without
        it, or for a format it can't read (EXR), the caller falls back to the
        same rejected-by-name warning as before. The bare ``save`` is tried
        first so PNG-representable modes (16-bit gray, palette) pass through
        losslessly; only modes PNG itself refuses (CMYK, float) are converted.
        """
        try:
            from PIL import Image
        except ImportError:
            # Distinguishable in the log from a truly unsupported format: the
            # caller's warning says "unsupported image type", which would send
            # someone hunting a format problem when the fix is `pip install
            # pillow`.
            logger.debug(
                "GLB texture embed: Pillow unavailable; cannot re-encode %s", path
            )
            return None
        import io

        try:
            with Image.open(path) as img:
                buf = io.BytesIO()
                try:
                    img.save(buf, format="PNG")
                except (OSError, ValueError):
                    buf = io.BytesIO()
                    mode = "RGBA" if "A" in img.getbands() else "RGB"
                    img.convert(mode).save(buf, format="PNG")
                return buf.getvalue()
        except (OSError, ValueError) as error:
            logger.warning("GLB texture embed: could not re-encode %s: %s", path, error)
            return None

    @classmethod
    def set_glb_base_color(
        cls, glb: GlbTarget, base_color: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """Write base colour / texture into a GLB's materials, by name.

        The sibling of :meth:`set_glb_emissive`, for the same underlying gap.
        Measured against Maya 2025 + MtoA: ``lambert`` / ``blinn`` / ``phong``
        carry their colour through FBX (scaled by Maya's ``diffuse`` weight),
        while ``aiStandardSurface`` and ``standardSurface`` arrive with
        ``baseColorFactor`` at a flat **[1,1,1,1]** -- Maya's exporter does not
        map them at all, so every modern shader previews as white plastic. That
        also swamps emissive: a surface already at white leaves no headroom for
        an additive term to read against.

        Unlike emissive there is no strength extension and no zero-skip: black
        is a legitimate base colour, so values are clamped into [0,1] rather
        than normalized, and an all-zero colour is written as authored.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            base_color: ``{material_name: {"color": (r, g, b), "texture": path}}``.
                Both keys optional. Names absent from the GLB are reported.

        Returns:
            List of records: ``material``, ``factor``, ``texture``.
        """
        if not base_color:
            return []

        records: List[Dict] = []
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            for name, spec, mat in cls._match_glb_materials(
                gltf, base_color, "set_glb_base_color"
            ):
                pbr = mat.setdefault("pbrMetallicRoughness", {})
                entry_written = False

                tex_path = spec.get("texture")
                tex_index = None
                if tex_path:
                    tex_index = cls._embed_image(edit, tex_path)
                    if tex_index is not None:
                        pbr["baseColorTexture"] = {"index": tex_index}
                        entry_written = True

                color = spec.get("color")
                factor = None
                if color is not None:
                    factor = [min(1.0, max(0.0, float(c))) for c in list(color)[:3]]
                    while len(factor) < 3:
                        factor.append(0.0)
                    # Preserve any alpha the converter already established; this
                    # writer is about colour and must not silently turn a
                    # transparent material opaque.
                    existing = pbr.get("baseColorFactor") or [1.0, 1.0, 1.0, 1.0]
                    factor.append(existing[3] if len(existing) > 3 else 1.0)
                    pbr["baseColorFactor"] = factor
                    entry_written = True

                if entry_written:
                    records.append(
                        {
                            "material": name,
                            "factor": factor,
                            # Gate on the embed RESULT, not the request: an image
                            # that could not be embedded (missing, EXR, no Pillow)
                            # writes no baseColorTexture, so reporting its path
                            # claims a channel the GLB does not carry. Mirrors the
                            # emissive writer.
                            "texture": tex_path if tex_index is not None else None,
                        }
                    )

            if not records:
                # As in `set_glb_emissive`: on a shared session another writer's
                # pending edits are not this one's to discard.
                return []

            cls._prune_empty_containers(gltf)
            edit.dirty = True

        return records
