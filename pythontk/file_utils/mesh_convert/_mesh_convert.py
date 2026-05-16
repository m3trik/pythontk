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
from typing import Dict, List, Optional, Tuple

from pythontk.core_utils.help_mixin import HelpMixin

logger = logging.getLogger(__name__)

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
    def check_glb_materials(cls, glb_path: str) -> List[Dict[str, str]]:
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
            glb_path: Path to a binary glTF (.glb) file.

        Returns:
            List of findings. Each finding is a dict with keys:
                material   — material name (or '<material[i]>')
                alpha_mode — "BLEND" or "MASK"
                image      — image name / uri / fallback id
                reason     — short human-readable explanation
        """
        from io import BytesIO

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "check_glb_materials requires Pillow (PIL). Install it with "
                "`pip install pillow`."
            ) from exc

        if not os.path.isfile(glb_path):
            raise FileNotFoundError(glb_path)

        with open(glb_path, "rb") as f:
            header = f.read(12)
            if len(header) < 12 or header[:4] != b"glTF":
                raise ValueError(f"Not a GLB file: {glb_path}")
            _version, _total = struct.unpack("<II", header[4:])
            chunk0_len, chunk0_type = struct.unpack("<I4s", f.read(8))
            if chunk0_type != b"JSON":
                raise ValueError(f"Malformed GLB: first chunk is not JSON ({glb_path})")
            gltf = json.loads(f.read(chunk0_len).decode("utf-8"))
            bin_data: Optional[bytes] = None
            header = f.read(8)
            if len(header) == 8:
                bin_len, bin_type = struct.unpack("<I4s", header)
                if bin_type == b"BIN\x00":
                    bin_data = f.read(bin_len)

        materials = gltf.get("materials", []) or []
        textures = gltf.get("textures", []) or []
        images = gltf.get("images", []) or []
        buffer_views = gltf.get("bufferViews", []) or []

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

        # Decoded once per source image even if multiple materials reference it.
        # Value is (extrema_min, extrema_max) or None when the image was
        # unreadable / had no alpha channel and is therefore safe to skip.
        alpha_extrema_cache: Dict[int, Optional[Tuple[int, int]]] = {}

        def _image_alpha_extrema(img_idx: int) -> Optional[Tuple[int, int]]:
            if img_idx in alpha_extrema_cache:
                return alpha_extrema_cache[img_idx]
            img_entry = images[img_idx]
            img_bytes = cls._extract_image_bytes(
                img_entry, glb_path, bin_data, buffer_views
            )
            result: Optional[tuple] = None
            if img_bytes:
                try:
                    with Image.open(BytesIO(img_bytes)) as im:
                        im.load()
                        has_alpha_channel = im.mode in ("RGBA", "LA", "PA") or (
                            im.mode == "P" and "transparency" in im.info
                        )
                        if has_alpha_channel:
                            result = im.convert("RGBA").getchannel("A").getextrema()
                except Exception as exc:  # noqa: BLE001 — decoder reports varied errors
                    logger.debug(
                        "check_glb_materials: skipped image %s (%s)", img_idx, exc
                    )
            alpha_extrema_cache[img_idx] = result
            return result

        findings: List[Dict[str, str]] = []
        for mi, mat in enumerate(materials):
            alpha_mode = mat.get("alphaMode", "OPAQUE")
            if alpha_mode not in REASONS:  # OPAQUE or unknown — skip
                continue
            pbr = mat.get("pbrMetallicRoughness") or {}

            # Real transparency can come from the scalar baseColorFactor[3];
            # don't flag those as "accidentally transparent".
            bc_factor = pbr.get("baseColorFactor")
            if bc_factor and len(bc_factor) >= 4 and bc_factor[3] < 1.0:
                continue

            bct = pbr.get("baseColorTexture")
            if not bct:
                continue
            tex_idx = bct.get("index")
            if tex_idx is None or tex_idx >= len(textures):
                continue
            img_idx = textures[tex_idx].get("source")
            if img_idx is None or img_idx >= len(images):
                continue

            extrema = _image_alpha_extrema(img_idx)
            if extrema != (255, 255):
                continue

            img_entry = images[img_idx]
            findings.append(
                {
                    "material": mat.get("name") or f"<material[{mi}]>",
                    "alpha_mode": alpha_mode,
                    "image": (
                        img_entry.get("name")
                        or img_entry.get("uri")
                        or f"image[{img_idx}]"
                    ),
                    "reason": REASONS[alpha_mode],
                }
            )

        return findings

    @classmethod
    def fix_glb_phantom_opaque_alpha(cls, glb_path: str) -> List[Dict]:
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
            glb_path: Path to a binary glTF (.glb) file (modified in place).

        Returns:
            List of fix records. Empty when nothing matched. Each record:
                material   — material name
                old_alpha  — original baseColorFactor[3]
                new_alpha  — 1.0
                image      — the baseColorTexture image identifier
        """
        from io import BytesIO

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "fix_glb_phantom_opaque_alpha requires Pillow (PIL). "
                "Install it with `pip install pillow`."
            ) from exc

        if not os.path.isfile(glb_path):
            raise FileNotFoundError(glb_path)

        with open(glb_path, "rb") as f:
            magic = f.read(4)
            if magic != b"glTF":
                raise ValueError(f"Not a GLB file: {glb_path}")
            version_bytes = f.read(4)
            f.read(4)  # total length — recomputed on write
            chunk0_len = struct.unpack("<I", f.read(4))[0]
            chunk0_type = f.read(4)
            if chunk0_type != b"JSON":
                raise ValueError(f"Malformed GLB: first chunk not JSON ({glb_path})")
            json_bytes = f.read(chunk0_len)
            gltf = json.loads(json_bytes.decode("utf-8"))
            rest = f.read()

        bin_data: Optional[bytes] = None
        if len(rest) >= 8:
            bin_len = struct.unpack("<I", rest[:4])[0]
            if rest[4:8] == b"BIN\x00":
                bin_data = rest[8 : 8 + bin_len]

        materials = gltf.get("materials", []) or []
        textures = gltf.get("textures", []) or []
        images = gltf.get("images", []) or []
        buffer_views = gltf.get("bufferViews", []) or []

        alpha_extrema_cache: Dict[int, Optional[Tuple[int, int]]] = {}

        def _alpha_extrema(img_idx: int) -> Optional[Tuple[int, int]]:
            if img_idx in alpha_extrema_cache:
                return alpha_extrema_cache[img_idx]
            result: Optional[Tuple[int, int]] = None
            if img_idx < len(images):
                img_bytes = cls._extract_image_bytes(
                    images[img_idx], glb_path, bin_data, buffer_views
                )
                if img_bytes:
                    try:
                        with Image.open(BytesIO(img_bytes)) as im:
                            im.load()
                            has_alpha = im.mode in ("RGBA", "LA", "PA") or (
                                im.mode == "P" and "transparency" in im.info
                            )
                            if has_alpha:
                                result = im.convert("RGBA").getchannel("A").getextrema()
                    except Exception as exc:  # noqa: BLE001 — varied decoder errors
                        logger.debug(
                            "fix_glb_phantom_opaque_alpha: skipped image %s (%s)",
                            img_idx,
                            exc,
                        )
            alpha_extrema_cache[img_idx] = result
            return result

        EPSILON = 1e-4
        fixes: List[Dict] = []
        for mi, mat in enumerate(materials):
            if mat.get("alphaMode") not in ("BLEND", "MASK"):
                continue
            pbr = mat.get("pbrMetallicRoughness") or {}
            bcf = pbr.get("baseColorFactor")
            if not bcf or len(bcf) < 4 or bcf[3] > EPSILON:
                continue
            bct = pbr.get("baseColorTexture")
            if not bct:
                continue
            tex_idx = bct.get("index")
            if tex_idx is None or tex_idx >= len(textures):
                continue
            img_idx = textures[tex_idx].get("source")
            if img_idx is None:
                continue
            extrema = _alpha_extrema(img_idx)
            # Skip uniform alpha (genuinely-transparent or genuinely-opaque
            # textures) — only varying alpha indicates a real cutout mask
            # whose per-pixel control was cancelled by baseColorFactor[3]=0.
            if extrema is None or extrema[0] == extrema[1]:
                continue

            old_alpha = bcf[3]
            bcf[3] = 1.0
            pbr["baseColorFactor"] = bcf
            mat["pbrMetallicRoughness"] = pbr

            img_entry = images[img_idx] if img_idx < len(images) else {}
            fixes.append(
                {
                    "material": mat.get("name") or f"<material[{mi}]>",
                    "old_alpha": old_alpha,
                    "new_alpha": 1.0,
                    "image": (
                        img_entry.get("name")
                        or img_entry.get("uri")
                        or f"image[{img_idx}]"
                    ),
                }
            )

        if not fixes:
            return []

        # Re-serialize JSON (compact) and pad to 4-byte align with 0x20.
        new_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        pad = (4 - (len(new_json) % 4)) % 4
        new_json = new_json + (b" " * pad)
        new_total = 12 + 8 + len(new_json) + len(rest)

        with open(glb_path, "wb") as f:
            f.write(b"glTF")
            f.write(version_bytes)
            f.write(struct.pack("<I", new_total))
            f.write(struct.pack("<I", len(new_json)))
            f.write(b"JSON")
            f.write(new_json)
            f.write(rest)

        return fixes

    @staticmethod
    def _extract_image_bytes(
        img_entry: dict,
        glb_path: str,
        bin_data: Optional[bytes],
        buffer_views: list,
    ) -> Optional[bytes]:
        """Return raw bytes for a glTF image entry, or None if unavailable."""
        uri = img_entry.get("uri")
        if uri:
            if uri.startswith("data:"):
                try:
                    _, b64 = uri.split(",", 1)
                    return base64.b64decode(b64)
                except Exception:
                    return None
            sibling = os.path.join(os.path.dirname(glb_path), uri)
            if os.path.isfile(sibling):
                with open(sibling, "rb") as f:
                    return f.read()
            return None
        bv_idx = img_entry.get("bufferView")
        if bv_idx is None or bin_data is None or bv_idx >= len(buffer_views):
            return None
        bv = buffer_views[bv_idx]
        offset = bv.get("byteOffset", 0)
        length = bv.get("byteLength", 0)
        return bin_data[offset : offset + length] or None
