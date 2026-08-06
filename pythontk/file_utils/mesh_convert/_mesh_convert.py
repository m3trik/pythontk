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
from typing import Any, Dict, List, Optional, Tuple

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
    # Image types glTF 2.0 accepts natively. Anything else (TIFF, EXR, TGA —
    # all common in a DCC source tree) is re-encoded to PNG via Pillow when
    # available (see `_reencode_as_png`), and otherwise rejected by name
    # rather than written as an unloadable data URI.
    IMAGE_MIME_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }

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

        _version, gltf, _rest, bin_data = cls._read_glb(glb_path)

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

        version_bytes, gltf, rest, bin_data = cls._read_glb(glb_path)

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

        cls._write_glb(glb_path, version_bytes, gltf, rest)
        return fixes

    @staticmethod
    def _read_glb(glb_path: str) -> Tuple[bytes, dict, bytes, Optional[bytes]]:
        """Split a GLB into ``(version_bytes, gltf, rest, bin_data)``.

        The single owner of GLB container parsing for this class — the JSON
        chunk is what every repair here edits, and re-deriving the offsets per
        function is how one of them ends up with a subtly different idea of
        where the BIN chunk starts.

        *rest* is every byte after the JSON chunk, kept verbatim so a writer can
        put it back untouched; *bin_data* is the BIN chunk's payload alone, for
        callers resolving ``bufferView``-backed images.
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
            gltf = json.loads(f.read(chunk0_len).decode("utf-8"))
            rest = f.read()

        bin_data: Optional[bytes] = None
        if len(rest) >= 8:
            bin_len = struct.unpack("<I", rest[:4])[0]
            if rest[4:8] == b"BIN\x00":
                bin_data = rest[8 : 8 + bin_len]

        return version_bytes, gltf, rest, bin_data

    @staticmethod
    def _write_glb(glb_path: str, version_bytes: bytes, gltf: dict, rest: bytes) -> None:
        """Rewrite *glb_path* with an edited JSON chunk and *rest* verbatim.

        The JSON chunk is padded to a 4-byte boundary with spaces and the total
        length recomputed — both required by the GLB spec, and both easy to get
        wrong in a way that only some loaders reject.
        """
        new_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        new_json += b" " * ((4 - (len(new_json) % 4)) % 4)

        with open(glb_path, "wb") as f:
            f.write(b"glTF")
            f.write(version_bytes)
            f.write(struct.pack("<I", 12 + 8 + len(new_json) + len(rest)))
            f.write(struct.pack("<I", len(new_json)))
            f.write(b"JSON")
            f.write(new_json)
            f.write(rest)

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
        cls, glb_path: str, emissive: Dict[str, Dict[str, Any]]
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
        afford. Repeated paths are embedded once and shared.

        Parameters:
            glb_path: Path to a binary glTF (.glb), modified in place.
            emissive: ``{material_name: {"color": (r, g, b), "texture": path}}``.
                Both keys optional; a texture with no color implies white.
                Names not present in the GLB are reported, not raised.

        Returns:
            List of records: ``material``, ``factor``, ``strength``, ``texture``.
        """
        if not emissive:
            return []

        version_bytes, gltf, rest, _bin = cls._read_glb(glb_path)
        embedded: Dict[str, int] = {}

        def _embed(path: str) -> Optional[int]:
            return cls._embed_image(gltf, path, embedded)

        records: List[Dict] = []
        used_strength = False
        for name, spec, mat in cls._match_glb_materials(
            gltf, emissive, "set_glb_emissive"
        ):
            color = list(spec.get("color") or (1.0, 1.0, 1.0))[:3]
            if len(color) < 3:
                color += [0.0] * (3 - len(color))

            # A zero colour is zero emission -- and paired with a texture it is
            # actively harmful rather than merely redundant: glTF emission is
            # ``emissiveFactor * emissiveTexture``, so writing both multiplies
            # the map to black while still reporting a successful transfer.
            # Tested before embedding, so a skipped material leaves no orphaned
            # image in the file.
            if all(c <= 0.0 for c in color):
                continue

            peak = max(color)
            strength = None
            if peak > 1.0:
                color = [c / peak for c in color]
                strength = peak
                mat.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
                    "emissiveStrength": strength
                }
                used_strength = True

            tex_path = spec.get("texture")
            tex_index = _embed(tex_path) if tex_path else None
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
            return []

        if used_strength:
            used = gltf.setdefault("extensionsUsed", [])
            if "KHR_materials_emissive_strength" not in used:
                used.append("KHR_materials_emissive_strength")

        # Drop containers left empty, rather than emitting `"samplers": []` —
        # an empty array is invalid per the glTF schema (minItems 1).
        for key in ("images", "textures", "samplers"):
            if not gltf.get(key):
                gltf.pop(key, None)

        cls._write_glb(glb_path, version_bytes, gltf, rest)
        return records

    @classmethod
    def _embed_image(
        cls, gltf: dict, path: str, cache: Dict[str, int]
    ) -> Optional[int]:
        """Embed *path* as a ``data:`` URI image and return its texture index.

        Shared by every channel writer here. Data URIs keep the whole edit
        inside the JSON chunk -- no buffer offsets to recompute, which is the
        part of GLB surgery that silently corrupts a file -- at the cost of
        base64's ~33% overhead, which a local preview can afford. Repeated
        paths resolve to one image via *cache*.
        """
        if path in cache:
            return cache[path]
        if not os.path.isfile(path):
            logger.warning("GLB texture embed: file not found: %s", path)
            return None
        mime = cls.IMAGE_MIME_TYPES.get(os.path.splitext(path)[1].lower())
        if mime is not None:
            with open(path, "rb") as f:
                raw = f.read()
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
        cls, glb_path: str, base_color: Dict[str, Dict[str, Any]]
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
            glb_path: Path to a binary glTF (.glb), modified in place.
            base_color: ``{material_name: {"color": (r, g, b), "texture": path}}``.
                Both keys optional. Names absent from the GLB are reported.

        Returns:
            List of records: ``material``, ``factor``, ``texture``.
        """
        if not base_color:
            return []

        version_bytes, gltf, rest, _bin = cls._read_glb(glb_path)
        embedded: Dict[str, int] = {}
        records: List[Dict] = []

        for name, spec, mat in cls._match_glb_materials(
            gltf, base_color, "set_glb_base_color"
        ):
            pbr = mat.setdefault("pbrMetallicRoughness", {})
            entry_written = False

            tex_path = spec.get("texture")
            tex_index = None
            if tex_path:
                tex_index = cls._embed_image(gltf, tex_path, embedded)
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
            return []

        for key in ("images", "textures", "samplers"):
            if not gltf.get(key):
                gltf.pop(key, None)

        cls._write_glb(glb_path, version_bytes, gltf, rest)
        return records

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
