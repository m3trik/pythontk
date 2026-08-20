# !/usr/bin/python
# coding=utf-8
"""Unit tests for MeshConvert.

Network-free — FBX2glTF resolution and subprocess invocation are mocked.
An opt-in integration test triggers a real install when
``PYTHONTK_INTEGRATION_TESTS=1``.

Run with:
    python -m pytest test_mesh_convert.py -v
    python test_mesh_convert.py
"""

import hashlib
import io
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk import ImgUtils, MeshConvert


class TestResolveBinary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_returns_path_when_on_system_path(self):
        with patch("shutil.which", return_value="/usr/bin/FBX2glTF"):
            self.assertEqual(MeshConvert.resolve_binary(), "/usr/bin/FBX2glTF")

    def test_returns_managed_path_when_in_catalog(self):
        managed = os.path.join(self.tmp, "FBX2glTF.exe")
        with (
            patch("shutil.which", return_value=None),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=managed,
            ),
        ):
            self.assertEqual(MeshConvert.resolve_binary(), managed)

    def test_raises_when_missing_and_required(self):
        with (
            patch("shutil.which", return_value=None),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=None,
            ),
        ):
            with self.assertRaises(FileNotFoundError):
                MeshConvert.resolve_binary(required=True, auto_install=False)

    def test_returns_none_when_missing_and_not_required(self):
        with (
            patch("shutil.which", return_value=None),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=None,
            ),
        ):
            self.assertIsNone(
                MeshConvert.resolve_binary(required=False, auto_install=False)
            )

    def test_no_tty_with_prompt_refuses_install(self):
        """prompt=True without a TTY should NOT silently install."""
        with (
            patch("shutil.which", return_value=None),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=None,
            ),
            patch("pythontk.core_utils.app_installer.AppInstaller.ensure") as ensure,
            patch("sys.stdin") as stdin,
        ):
            stdin.isatty.return_value = False
            with self.assertRaises(FileNotFoundError) as cm:
                MeshConvert.resolve_binary(
                    auto_install=True, prompt=True, required=True
                )
            self.assertIn("interactive", str(cm.exception).lower())
            ensure.assert_not_called()

    def test_no_tty_without_prompt_installs_silently(self):
        """prompt=False allows non-interactive install (CI/automation)."""
        installed = os.path.join(self.tmp, "FBX2glTF.exe")
        with (
            patch("shutil.which", return_value=None),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=None,
            ),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.ensure",
                return_value=installed,
            ) as ensure,
            patch("sys.stdin") as stdin,
        ):
            stdin.isatty.return_value = False
            result = MeshConvert.resolve_binary(auto_install=True, prompt=False)
            self.assertEqual(result, installed)
            ensure.assert_called_once()

    def test_prompt_decline_raises_when_required(self):
        with (
            patch("shutil.which", return_value=None),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=None,
            ),
            patch("sys.stdin") as stdin,
        ):
            stdin.isatty.return_value = True
            stdin.readline.return_value = "n\n"
            with self.assertRaises(FileNotFoundError):
                MeshConvert.resolve_binary(
                    auto_install=True, prompt=True, required=True
                )

    def test_prompt_accept_triggers_install(self):
        installed = os.path.join(self.tmp, "FBX2glTF.exe")
        with (
            patch("shutil.which", return_value=None),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=None,
            ),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.ensure",
                return_value=installed,
            ) as ensure,
            patch("sys.stdin") as stdin,
        ):
            stdin.isatty.return_value = True
            stdin.readline.return_value = "y\n"
            result = MeshConvert.resolve_binary(auto_install=True, prompt=True)
            self.assertEqual(result, installed)
            ensure.assert_called_once()

    def test_platform_exe_name_known(self):
        name = MeshConvert._platform_exe_name()
        self.assertTrue(name.startswith("FBX2glTF"))


class TestFbxToGlb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = os.path.join(self.tmp, "model.fbx")
        with open(self.src, "wb") as fh:
            fh.write(b"fake-fbx")
        self.fake_bin = os.path.join(self.tmp, "FBX2glTF.exe")

    def test_missing_src_raises(self):
        with self.assertRaises(FileNotFoundError):
            MeshConvert.fbx_to_glb(os.path.join(self.tmp, "nope.fbx"))

    def test_wrong_extension_raises(self):
        bad = os.path.join(self.tmp, "model.obj")
        with open(bad, "wb") as fh:
            fh.write(b"")
        with self.assertRaises(ValueError):
            MeshConvert.fbx_to_glb(bad)

    def test_existing_dst_without_overwrite_raises(self):
        dst = os.path.join(self.tmp, "model.glb")
        with open(dst, "wb") as fh:
            fh.write(b"existing")
        with self.assertRaises(FileExistsError):
            MeshConvert.fbx_to_glb(self.src, dst, overwrite=False)

    def _run_simulator(self, captured):
        """Return a subprocess.run replacement that records the cmd and
        creates the expected .glb at <output_base>.glb."""

        def _run(cmd, **kw):
            captured["cmd"] = cmd
            captured["kwargs"] = kw
            # FBX2glTF writes to <output_base>.glb
            output_base_idx = cmd.index("-o") + 1
            with open(cmd[output_base_idx] + ".glb", "wb") as fh:
                fh.write(b"glb-bytes")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        return _run

    def test_default_dst_derived_from_src(self):
        expected_dst = os.path.join(self.tmp, "model.glb")
        captured = {}
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=self._run_simulator(captured)),
        ):
            result = MeshConvert.fbx_to_glb(self.src, auto_install=False)
            self.assertEqual(result, expected_dst)
            self.assertTrue(os.path.isfile(expected_dst))

    def test_dst_glb_extension_appended_if_missing(self):
        captured = {}
        dst_no_ext = os.path.join(self.tmp, "out")
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=self._run_simulator(captured)),
        ):
            result = MeshConvert.fbx_to_glb(self.src, dst_no_ext, auto_install=False)
            self.assertEqual(result, dst_no_ext + ".glb")

    def test_command_uses_input_output_binary_flags(self):
        dst = os.path.join(self.tmp, "out.glb")
        captured = {}
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=self._run_simulator(captured)),
        ):
            MeshConvert.fbx_to_glb(self.src, dst, auto_install=False)

        cmd = captured["cmd"]
        self.assertEqual(cmd[0], self.fake_bin)
        self.assertIn("-i", cmd)
        self.assertIn("-o", cmd)
        self.assertIn("--binary", cmd)
        # Default-on (measured v0.13.1 + Maya 2025): without it the DataNodes
        # channels deliberately embedded in the FBX are silently dropped.
        self.assertIn("--user-properties", cmd)
        # -o argument must be the output base WITHOUT .glb suffix
        output_base = cmd[cmd.index("-o") + 1]
        self.assertFalse(output_base.lower().endswith(".glb"))

    def test_extra_args_forwarded(self):
        dst = os.path.join(self.tmp, "out.glb")
        captured = {}
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=self._run_simulator(captured)),
        ):
            MeshConvert.fbx_to_glb(
                self.src, dst, auto_install=False, extra_args=["--draco"]
            )
        self.assertIn("--draco", captured["cmd"])

    def test_subprocess_failure_raises(self):
        dst = os.path.join(self.tmp, "out.glb")
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["x"], 1, stdout="", stderr="boom"
                ),
            ),
        ):
            with self.assertRaises(RuntimeError) as cm:
                MeshConvert.fbx_to_glb(self.src, dst, auto_install=False)
            self.assertIn("boom", str(cm.exception))

    def test_subprocess_zero_exit_but_no_output_raises(self):
        dst = os.path.join(self.tmp, "out.glb")
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["x"], 0, stdout="", stderr=""
                ),
            ),
        ):
            with self.assertRaises(RuntimeError) as cm:
                MeshConvert.fbx_to_glb(self.src, dst, auto_install=False)
            self.assertIn("not created", str(cm.exception))

    def test_timeout_raises_runtime_error(self):
        dst = os.path.join(self.tmp, "out.glb")
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["x"], timeout=1),
            ),
        ):
            with self.assertRaises(RuntimeError) as cm:
                MeshConvert.fbx_to_glb(self.src, dst, auto_install=False, timeout=1)
            self.assertIn("timed out", str(cm.exception))

    def test_timeout_kwarg_forwarded_to_subprocess(self):
        dst = os.path.join(self.tmp, "out.glb")
        captured = {}
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=self._run_simulator(captured)),
        ):
            MeshConvert.fbx_to_glb(self.src, dst, auto_install=False, timeout=42)
        self.assertEqual(captured["kwargs"].get("timeout"), 42)


class TestCheckGlbMaterials(unittest.TestCase):
    """Verify the post-conversion material sanity check."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_check_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    @staticmethod
    def _png_bytes(mode, size=(4, 4), alpha=None):
        """Build a small PNG and return its bytes."""
        from io import BytesIO
        from PIL import Image

        if mode == "RGBA":
            im = Image.new(
                "RGBA", size, (200, 100, 50, alpha if alpha is not None else 255)
            )
        elif mode == "RGB":
            im = Image.new("RGB", size, (200, 100, 50))
        else:
            raise ValueError(mode)
        buf = BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _build_glb(materials, images, textures, image_blobs):
        """Pack a synthetic GLB. image_blobs is a list of bytes objects."""
        import struct

        buffer_views = []
        bin_chunks = []
        offset = 0
        for blob in image_blobs:
            buffer_views.append(
                {"buffer": 0, "byteOffset": offset, "byteLength": len(blob)}
            )
            bin_chunks.append(blob)
            # 4-byte align
            pad = (4 - (len(blob) % 4)) % 4
            if pad:
                bin_chunks.append(b"\x00" * pad)
                offset += pad
            offset += len(blob)

        bin_data = b"".join(bin_chunks)
        gltf = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": len(bin_data)}],
            "bufferViews": buffer_views,
            "images": images,
            "textures": textures,
            "materials": materials,
        }
        json_bytes = json.dumps(gltf).encode("utf-8")
        # Pad JSON chunk to 4-byte boundary with spaces
        pad_json = (4 - (len(json_bytes) % 4)) % 4
        json_bytes += b" " * pad_json

        header = struct.pack(
            "<4sII", b"glTF", 2, 12 + 8 + len(json_bytes) + 8 + len(bin_data)
        )
        json_chunk = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
        bin_chunk = struct.pack("<I4s", len(bin_data), b"BIN\x00") + bin_data
        return header + json_chunk + bin_chunk

    def _write_glb(self, name, **kw):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(self._build_glb(**kw))
        return path

    def test_flags_blend_with_opaque_alpha(self):
        """Texture is RGBA but alpha=255 everywhere → must flag."""
        blob = self._png_bytes("RGBA", alpha=255)
        path = self._write_glb(
            "opaque_blend.glb",
            materials=[
                {
                    "name": "Body_base",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png", "name": "color"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        findings = MeshConvert.check_glb_materials(path)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["material"], "Body_base")
        self.assertEqual(findings[0]["alpha_mode"], "BLEND")

    def test_does_not_flag_genuine_transparency(self):
        """Texture is RGBA with varying alpha → genuine transparency, no flag."""
        from io import BytesIO
        from PIL import Image

        im = Image.new("RGBA", (4, 4))
        for y in range(4):
            for x in range(4):
                im.putpixel((x, y), (200, 100, 50, 30 + 50 * x))
        buf = BytesIO()
        im.save(buf, format="PNG")
        path = self._write_glb(
            "real_blend.glb",
            materials=[
                {
                    "name": "Glass",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[buf.getvalue()],
        )
        self.assertEqual(MeshConvert.check_glb_materials(path), [])

    def test_does_not_flag_opaque_material(self):
        """alphaMode=OPAQUE is never flagged, even if texture is RGBA."""
        blob = self._png_bytes("RGBA", alpha=255)
        path = self._write_glb(
            "opaque.glb",
            materials=[
                {
                    "name": "Plain",
                    "alphaMode": "OPAQUE",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        self.assertEqual(MeshConvert.check_glb_materials(path), [])

    def test_does_not_flag_rgb_texture(self):
        """No alpha channel → can't have leaked transparency, no flag."""
        blob = self._png_bytes("RGB")
        path = self._write_glb(
            "no_alpha.glb",
            materials=[
                {
                    "name": "RGBish",
                    "alphaMode": "BLEND",  # weird but possible
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        self.assertEqual(MeshConvert.check_glb_materials(path), [])

    def test_mask_mode_is_also_checked(self):
        """alphaMode=MASK with uniformly-255 alpha is still wrong."""
        blob = self._png_bytes("RGBA", alpha=255)
        path = self._write_glb(
            "mask.glb",
            materials=[
                {
                    "name": "Leaf",
                    "alphaMode": "MASK",
                    "alphaCutoff": 0.5,
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        findings = MeshConvert.check_glb_materials(path)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["alpha_mode"], "MASK")

    def test_does_not_flag_transparency_from_basecolorfactor(self):
        """Material is legitimately transparent via baseColorFactor[3] < 1.0 —
        even with an opaque texture this is real transparency, not a leak."""
        blob = self._png_bytes("RGBA", alpha=255)
        path = self._write_glb(
            "factor_alpha.glb",
            materials=[
                {
                    "name": "TintedGlass",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 0.4],
                        "baseColorTexture": {"index": 0},
                    },
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        self.assertEqual(MeshConvert.check_glb_materials(path), [])

    def test_basecolorfactor_alpha_1_still_flags(self):
        """Factor alpha = 1.0 is fully opaque so it must NOT exempt the
        material — the texture-alpha leak should still be caught."""
        blob = self._png_bytes("RGBA", alpha=255)
        path = self._write_glb(
            "factor_one.glb",
            materials=[
                {
                    "name": "Body",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                        "baseColorTexture": {"index": 0},
                    },
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        self.assertEqual(len(MeshConvert.check_glb_materials(path)), 1)

    def test_reason_differs_per_alpha_mode(self):
        """BLEND reason mentions depth-write; MASK reason mentions no-op
        alpha-test. They must not be the same boilerplate."""
        blob = self._png_bytes("RGBA", alpha=255)

        blend_path = self._write_glb(
            "blend_reason.glb",
            materials=[
                {
                    "name": "B",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        mask_path = self._write_glb(
            "mask_reason.glb",
            materials=[
                {
                    "name": "M",
                    "alphaMode": "MASK",
                    "alphaCutoff": 0.5,
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        blend_reason = MeshConvert.check_glb_materials(blend_path)[0]["reason"]
        mask_reason = MeshConvert.check_glb_materials(mask_path)[0]["reason"]
        self.assertIn("depth-write", blend_reason)
        self.assertIn("no-op", mask_reason)
        self.assertNotEqual(blend_reason, mask_reason)

    def test_shared_image_decoded_once(self):
        """Two materials referencing the same image should both flag, but
        the underlying image must only be decoded a single time (cache)."""
        from unittest.mock import patch
        from PIL import Image as PILImage

        blob = self._png_bytes("RGBA", alpha=255)
        path = self._write_glb(
            "shared_image.glb",
            materials=[
                {
                    "name": "A",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                },
                {
                    "name": "B",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                },
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )

        real_open = PILImage.open
        calls = {"n": 0}

        def counting_open(*a, **kw):
            calls["n"] += 1
            return real_open(*a, **kw)

        with patch("PIL.Image.open", side_effect=counting_open):
            findings = MeshConvert.check_glb_materials(path)

        self.assertEqual(len(findings), 2, "both materials must be flagged")
        self.assertEqual(calls["n"], 1, "image should be decoded only once")

    def test_raises_on_non_glb(self):
        path = os.path.join(self.tmp, "not_a_glb.bin")
        with open(path, "wb") as f:
            f.write(b"not glTF")
        with self.assertRaises(ValueError):
            MeshConvert.check_glb_materials(path)

    def test_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            MeshConvert.check_glb_materials(os.path.join(self.tmp, "nope.glb"))


class TestFixGlbPhantomOpaqueAlpha(unittest.TestCase):
    """Verify the post-conversion fix for the Maya phong → FBX → glTF
    transparency translation bug, where baseColorFactor[3]=0 cancels the
    per-pixel alpha of a real cutout mask."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_fix_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    @staticmethod
    def _varying_alpha_png():
        from io import BytesIO
        from PIL import Image

        im = Image.new("RGBA", (4, 4))
        for y in range(4):
            for x in range(4):
                im.putpixel((x, y), (200, 100, 50, 30 + 50 * x))
        buf = BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _opaque_alpha_png():
        from io import BytesIO
        from PIL import Image

        im = Image.new("RGBA", (4, 4), (200, 100, 50, 255))
        buf = BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def _write_glb(self, name, materials, images, textures, image_blobs):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(
                TestCheckGlbMaterials._build_glb(
                    materials=materials,
                    images=images,
                    textures=textures,
                    image_blobs=image_blobs,
                )
            )
        return path

    @staticmethod
    def _read_alpha_factor(path):
        with open(path, "rb") as f:
            f.read(12)
            chunk0_len = struct.unpack("<I", f.read(4))[0]
            f.read(4)  # JSON
            gltf = json.loads(f.read(chunk0_len).decode("utf-8"))
        return gltf["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"][3]

    def test_fixes_blend_with_zero_alpha_and_varying_texture(self):
        """The bug pattern: BLEND + baseColorFactor[3]=0 + varying-alpha texture."""
        path = self._write_glb(
            "buggy.glb",
            materials=[
                {
                    "name": "TREELINE_D",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                        "baseColorTexture": {"index": 0},
                    },
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[self._varying_alpha_png()],
        )
        fixes = MeshConvert.fix_glb_phantom_opaque_alpha(path)
        self.assertEqual(len(fixes), 1)
        self.assertEqual(fixes[0]["material"], "TREELINE_D")
        self.assertEqual(fixes[0]["new_alpha"], 1.0)
        self.assertEqual(self._read_alpha_factor(path), 1.0)

    def test_fixes_mask_with_zero_alpha(self):
        """Same bug under alphaMode=MASK."""
        path = self._write_glb(
            "buggy_mask.glb",
            materials=[
                {
                    "name": "FoliageMask",
                    "alphaMode": "MASK",
                    "alphaCutoff": 0.5,
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                        "baseColorTexture": {"index": 0},
                    },
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[self._varying_alpha_png()],
        )
        fixes = MeshConvert.fix_glb_phantom_opaque_alpha(path)
        self.assertEqual(len(fixes), 1)
        self.assertEqual(self._read_alpha_factor(path), 1.0)

    def test_skips_opaque_material(self):
        """alphaMode=OPAQUE never gets touched."""
        path = self._write_glb(
            "opaque.glb",
            materials=[
                {
                    "name": "Plain",
                    "alphaMode": "OPAQUE",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                        "baseColorTexture": {"index": 0},
                    },
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[self._varying_alpha_png()],
        )
        self.assertEqual(MeshConvert.fix_glb_phantom_opaque_alpha(path), [])
        self.assertEqual(self._read_alpha_factor(path), 0.0)

    def test_skips_when_factor_already_nonzero(self):
        """Genuine partial-transparency factor must not be promoted."""
        path = self._write_glb(
            "partial.glb",
            materials=[
                {
                    "name": "Glass",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 0.5],
                        "baseColorTexture": {"index": 0},
                    },
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[self._varying_alpha_png()],
        )
        self.assertEqual(MeshConvert.fix_glb_phantom_opaque_alpha(path), [])
        self.assertEqual(self._read_alpha_factor(path), 0.5)

    def test_skips_when_texture_alpha_uniform(self):
        """Uniformly-opaque alpha (the 'check' bug) is NOT fixed by this pass."""
        path = self._write_glb(
            "uniform.glb",
            materials=[
                {
                    "name": "UniformAlpha",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                        "baseColorTexture": {"index": 0},
                    },
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[self._opaque_alpha_png()],
        )
        self.assertEqual(MeshConvert.fix_glb_phantom_opaque_alpha(path), [])
        self.assertEqual(self._read_alpha_factor(path), 0.0)

    def test_skips_when_no_base_color_texture(self):
        """No texture means no per-pixel alpha to recover; leave alone."""
        path = self._write_glb(
            "no_tex.glb",
            materials=[
                {
                    "name": "NoTex",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                    },
                }
            ],
            images=[],
            textures=[],
            image_blobs=[],
        )
        self.assertEqual(MeshConvert.fix_glb_phantom_opaque_alpha(path), [])

    def test_returns_empty_when_nothing_to_fix(self):
        """No changes → no rewrite, empty list returned, file untouched."""
        path = self._write_glb(
            "clean.glb",
            materials=[
                {
                    "name": "Plain",
                    "alphaMode": "OPAQUE",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[self._opaque_alpha_png()],
        )
        before = open(path, "rb").read()
        self.assertEqual(MeshConvert.fix_glb_phantom_opaque_alpha(path), [])
        self.assertEqual(open(path, "rb").read(), before)


class TestGlbReadGuards(unittest.TestCase):
    """A truncated GLB must fail as a ValueError, not a struct.error.

    Batch callers catch ``(RuntimeError, ValueError, OSError)`` per file.
    ``struct.error`` derives straight from ``Exception``, so a GLB truncated
    mid-write (converter killed, disk full) escaped the per-file handler and
    aborted the whole batch.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_glb_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _truncated(self, payload):
        path = os.path.join(self.tmp, "truncated.glb")
        with open(path, "wb") as f:
            f.write(payload)
        return path

    def test_header_shorter_than_12_bytes_raises_value_error(self):
        path = self._truncated(b"glTF\x02\x00\x00\x00")  # valid magic, no length
        with self.assertRaises(ValueError):
            MeshConvert.check_glb_materials(path)

    def test_missing_chunk_header_raises_value_error(self):
        path = self._truncated(b"glTF" + struct.pack("<II", 2, 12))  # header only
        with self.assertRaises(ValueError):
            MeshConvert.check_glb_materials(path)


class TestBaseColorRecordHonesty(unittest.TestCase):
    """``set_glb_base_color`` must not report a texture it did not embed."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_basecolor_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unembeddable_texture_is_not_reported_as_written(self):
        gltf = {
            "asset": {"version": "2.0"},
            "materials": [{"name": "Body", "pbrMetallicRoughness": {}}],
        }
        json_bytes = json.dumps(gltf).encode("utf-8")
        json_bytes += b" " * ((4 - (len(json_bytes) % 4)) % 4)
        path = os.path.join(self.tmp, "m.glb")
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_bytes)))
            f.write(struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes)

        # A texture path that cannot be embedded (file does not exist), plus a
        # colour so the entry is still written.
        records = MeshConvert.set_glb_base_color(
            path,
            {
                "Body": {
                    "texture": os.path.join(self.tmp, "missing.png"),
                    "color": [1.0, 0.0, 0.0],
                }
            },
        )
        self.assertEqual(len(records), 1)
        self.assertIsNone(
            records[0]["texture"],
            "no baseColorTexture was written — the record must not claim one",
        )


def _write_glb_file(path, gltf, bin_chunk=b""):
    """Pack *gltf* (+ optional BIN payload) into a GLB at *path*."""
    payload = json.dumps(gltf).encode("utf-8")
    payload += b" " * ((4 - (len(payload) % 4)) % 4)
    rest = b""
    if bin_chunk:
        bin_chunk += b"\x00" * ((4 - (len(bin_chunk) % 4)) % 4)
        rest = struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(payload) + len(rest)))
        f.write(struct.pack("<I4s", len(payload), b"JSON") + payload)
        f.write(rest)
    return path


def _converted_orm_glb(tmp, name="converted.glb"):
    """A GLB the way FBX2glTF hands one over: the material samples its
    converted ORM (image 0, bufferView 0 in the BIN) and the geometry
    (accessor -> bufferView 1) sits BEHIND it in the same BIN, so dropping
    the image must move the geometry's view without moving its bytes.

    Returns ``(path, geometry_bytes)``.
    """
    import io as iolib

    from PIL import Image

    buf = iolib.BytesIO()
    Image.new("RGB", (4, 4), (255, 255, 255)).save(buf, format="PNG")
    orm_bytes = buf.getvalue()
    orm_padded = orm_bytes + b"\x00" * ((4 - (len(orm_bytes) % 4)) % 4)
    geometry = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(orm_padded) + len(geometry)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(orm_bytes)},
            {
                "buffer": 0,
                "byteOffset": len(orm_padded),
                "byteLength": len(geometry),
                "target": 34962,
            },
        ],
        "accessors": [
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [1, 1, 0],
            }
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "materials": [
            {
                "name": "Room",
                "pbrMetallicRoughness": {
                    "metallicRoughnessTexture": {"index": 0},
                    "metallicFactor": 1.0,
                },
                "occlusionTexture": {"index": 0},
            }
        ],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{}],
        "images": [
            {"name": "ao_met_rough_Room", "bufferView": 0, "mimeType": "image/png"}
        ],
    }
    path = _write_glb_file(os.path.join(tmp, name), gltf, orm_padded + geometry)
    return path, geometry


def _assert_no_dead_payload(tc, path, geometry):
    """*path* ships no unreferenced image, the converted ORM is gone, and the
    geometry survived the BIN rewrite (view moved, bytes did not)."""
    edit = MeshConvert._read_glb(path)
    gltf = edit.gltf
    referenced = {
        gltf["textures"][t["index"]]["source"]
        for m in gltf["materials"]
        for t in (
            m["pbrMetallicRoughness"]["metallicRoughnessTexture"],
            m["occlusionTexture"],
        )
    }
    tc.assertEqual(
        referenced, set(range(len(gltf["images"]))), "unreferenced image shipped"
    )
    tc.assertNotIn("ao_met_rough_Room", [i.get("name") for i in gltf["images"]])
    view = gltf["bufferViews"][gltf["accessors"][0]["bufferView"]]
    blob = edit.bin_data
    tc.assertEqual(
        bytes(blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]),
        geometry,
    )
    tc.assertEqual(view.get("target"), 34962, "view attributes must survive")
    # The displaced ORM's view must be gone, stated as the invariant that
    # outlives the layout: nothing ships a view no accessor and no image reads.
    # (A count was the proxy for this while embeds lived in the JSON chunk as
    # base64; they are relocated into the BIN on close now, so the packed ORM
    # legitimately owns a view of its own.)
    live_views = {gltf["accessors"][0]["bufferView"]} | {
        i["bufferView"] for i in gltf["images"] if "bufferView" in i
    }
    tc.assertEqual(
        live_views,
        set(range(len(gltf["bufferViews"]))),
        "a bufferView shipped that nothing references",
    )
    tc.assertEqual(gltf["buffers"][0]["byteLength"], len(blob))
    return gltf


class TestGlbEditSession(unittest.TestCase):
    """One open GLB shared by several repairs: read once, write once.

    Every repair edits the JSON chunk and nothing else, but each used to open,
    parse and rewrite the whole file for itself — so a preview push, which runs
    three of them back to back, read a file the size of its geometry three
    times to change a handful of fields. These pin the four properties that
    made collapsing it safe.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_session_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write_glb(self, gltf, bin_chunk=b"", pretty=False, name="s.glb"):
        """Pack a GLB. *pretty* pads the JSON chunk the way a producer that
        indents its output does — which is what leaves room for the in-place
        rewrite, since this module always re-serializes compactly."""
        payload = json.dumps(gltf, indent=4 if pretty else None).encode("utf-8")
        payload += b" " * ((4 - (len(payload) % 4)) % 4)
        rest = b""
        if bin_chunk:
            bin_chunk += b"\x00" * ((4 - (len(bin_chunk) % 4)) % 4)
            rest = struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(payload) + len(rest)))
            f.write(struct.pack("<I4s", len(payload), b"JSON") + payload)
            f.write(rest)
        return path

    def _png(self, name="map.png"):
        from PIL import Image

        path = os.path.join(self.tmp, name)
        Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(path)
        return path

    @staticmethod
    def _counting_write():
        """``(patcher, writes)`` — wraps ``_write_glb`` and records each call."""
        real = MeshConvert._write_glb
        writes = []

        def counting(edit):
            writes.append(edit.path)
            return real(edit)

        return patch.object(MeshConvert, "_write_glb", counting), writes

    def test_one_session_writes_once_for_every_applier(self):
        """Two channel writers sharing a session must produce one write."""
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]},
            bin_chunk=b"GEOMETRY",
        )
        patcher, writes = self._counting_write()
        with patcher:
            with MeshConvert.open_glb(path) as session:
                MeshConvert.set_glb_base_color(
                    session, {"Body": {"color": (0.2, 0.4, 0.8)}}
                )
                MeshConvert.set_glb_emissive(
                    session, {"Body": {"color": (1.0, 0.0, 0.0)}}
                )

        self.assertEqual(len(writes), 1, "the session must write once, not per applier")
        mat = MeshConvert._read_glb(path).gltf["materials"][0]
        self.assertEqual(
            mat["pbrMetallicRoughness"]["baseColorFactor"][:3], [0.2, 0.4, 0.8]
        )
        self.assertEqual(mat["emissiveFactor"], [1.0, 0.0, 0.0])

    def test_a_path_target_still_writes_for_itself(self):
        """The path form stays self-contained — sharing is opt-in, not required."""
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )
        patcher, writes = self._counting_write()
        with patcher:
            MeshConvert.set_glb_base_color(path, {"Body": {"color": (0.2, 0.4, 0.8)}})
            MeshConvert.set_glb_emissive(path, {"Body": {"color": (1.0, 0.0, 0.0)}})
        self.assertEqual(len(writes), 2)

    def test_an_applier_that_matches_nothing_writes_nothing(self):
        """No match, no edit, no I/O — the usual case for the alpha repair."""
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )
        before = open(path, "rb").read()
        patcher, writes = self._counting_write()
        with patcher:
            self.assertEqual(
                MeshConvert.set_glb_emissive(path, {"Absent": {"color": (1, 1, 1)}}), []
            )
        self.assertEqual(writes, [])
        self.assertEqual(open(path, "rb").read(), before)

    def test_a_json_only_edit_leaves_the_file_size_and_bin_chunk_alone(self):
        """A shrinking edit is written in place, not by copying the geometry.

        This module re-serializes compactly while most producers do not, so the
        new JSON usually fits the chunk it came from; it is then padded back to
        exactly that length, leaving the total size and every byte after the
        JSON chunk untouched.
        """
        geometry = b"GEOMETRY" * 8
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]},
            bin_chunk=geometry,
            pretty=True,
        )
        size_before = os.path.getsize(path)

        MeshConvert.set_glb_emissive(path, {"Body": {"color": (1.0, 0.5, 0.0)}})

        self.assertEqual(os.path.getsize(path), size_before)
        edit = MeshConvert._read_glb(path)
        self.assertEqual(edit.gltf["materials"][0]["emissiveFactor"], [1.0, 0.5, 0.0])
        self.assertEqual(bytes(edit.bin_data), geometry)

    def test_a_growing_edit_falls_back_to_a_full_rewrite(self):
        """When the JSON no longer fits, the geometry must still come back whole."""
        geometry = b"GEOMETRYDATA1234"
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]},
            bin_chunk=geometry,
        )
        size_before = os.path.getsize(path)

        MeshConvert.set_glb_emissive(path, {"Body": {"texture": self._png()}})

        # Growth is what distinguishes this from the in-place path, which holds
        # the size fixed — asserting only that the file parses would pass on
        # either, and the fallback is the branch that has to move the geometry.
        self.assertGreater(os.path.getsize(path), size_before)
        edit = MeshConvert._read_glb(path)
        # The geometry keeps its bytes AND its place at the front of the BIN --
        # the embed is appended behind it, never interleaved.
        self.assertEqual(bytes(edit.bin_data)[: len(geometry)], geometry)
        image = edit.gltf["images"][0]
        self.assertNotIn("uri", image, "the embed stayed base64 in the JSON chunk")
        self.assertIn("bufferView", image)

    def test_an_unreadable_texture_is_skipped_not_raised(self):
        """A texture that exists but cannot be read must not abort the writer.

        The re-encode branch already warned and skipped; the direct PNG/JPEG
        read did not, so the same failure aborted the channel writer depending
        only on the file's extension — and it was the one file operation left
        inside an applier, able to take a whole sidecar section with it.
        """
        png = self._png()
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )

        real_open = open

        def refusing_open(file, *args, **kwargs):
            if os.path.abspath(str(file)) == os.path.abspath(png):
                raise PermissionError(13, "Permission denied", str(file))
            return real_open(file, *args, **kwargs)

        with patch("builtins.open", refusing_open):
            records = MeshConvert.set_glb_base_color(
                path, {"Body": {"texture": png, "color": [1.0, 0.0, 0.0]}}
            )

        self.assertEqual(len(records), 1, "the colour must still be written")
        self.assertIsNone(
            records[0]["texture"], "no texture was embedded — don't claim one"
        )
        self.assertNotIn("images", MeshConvert._read_glb(path).gltf)

    def test_a_clean_glb_is_never_read_past_its_json_chunk(self):
        """The repair that runs after *every* conversion must stay cheap.

        Nothing in the alpha check needs the BIN chunk until a material already
        looks wrong, so a GLB with nothing to fix should touch only the first
        few kilobytes of what is routinely a hundreds-of-megabytes file.
        """
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "Plain", "alphaMode": "OPAQUE"}],
            },
            bin_chunk=b"GEOMETRY" * 16,
        )
        with MeshConvert.open_glb(path) as session:
            self.assertEqual(MeshConvert.fix_glb_phantom_opaque_alpha(session), [])
            self.assertIsNone(
                session._rest,
                "a repair with nothing to fix must not pull the geometry in",
            )

    def test_bin_data_is_a_view_rather_than_a_copy(self):
        """Slicing the BIN chunk out put peak memory at twice the file size."""
        path = self._write_glb({"asset": {"version": "2.0"}}, bin_chunk=b"GEOMETRY")
        edit = MeshConvert._read_glb(path)
        self.assertIsInstance(edit.bin_data, memoryview)
        self.assertEqual(bytes(edit.bin_data), b"GEOMETRY")

    def test_a_base_color_texture_neutralises_the_converters_tint(self):
        """A texture with no colour beside it must not stay multiplied by a tint.

        Measured on a production room (StingrayPBS -> FBX2glTF 0.13.1): the FBX
        carries no DiffuseColor for a Stingray material, so every material
        reached the GLB at a flat 0.5 grey baseColorFactor. The sidecar rebound
        the texture correctly on top of it and left the factor alone, and glTF
        multiplies the two -- the whole room shipped at HALF its authored
        albedo, with nothing in the envelope reporting a problem.
        """
        png = self._png()
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [
                    {
                        "name": "Body",
                        "pbrMetallicRoughness": {
                            "baseColorFactor": [0.5, 0.5, 0.5, 1.0]
                        },
                    }
                ],
            }
        )
        MeshConvert.set_glb_base_color(path, {"Body": {"texture": png}})

        pbr = MeshConvert._read_glb(path).gltf["materials"][0]["pbrMetallicRoughness"]
        self.assertEqual(pbr["baseColorFactor"], [1.0, 1.0, 1.0, 1.0])
        self.assertIn("baseColorTexture", pbr)

    def test_a_base_color_texture_keeps_the_materials_alpha(self):
        """Neutralising the tint must not turn a transparent material opaque."""
        png = self._png()
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [
                    {
                        "name": "Body",
                        "pbrMetallicRoughness": {
                            "baseColorFactor": [0.5, 0.5, 0.5, 0.25]
                        },
                    }
                ],
            }
        )
        MeshConvert.set_glb_base_color(path, {"Body": {"texture": png}})

        pbr = MeshConvert._read_glb(path).gltf["materials"][0]["pbrMetallicRoughness"]
        self.assertEqual(pbr["baseColorFactor"], [1.0, 1.0, 1.0, 0.25])

    def test_an_explicit_base_colour_still_wins_over_the_texture(self):
        """An authored tint is intent -- only the converter's fallback is reset."""
        png = self._png()
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )
        MeshConvert.set_glb_base_color(
            path, {"Body": {"texture": png, "color": (0.2, 0.4, 0.8)}}
        )

        pbr = MeshConvert._read_glb(path).gltf["materials"][0]["pbrMetallicRoughness"]
        self.assertEqual(
            [round(c, 3) for c in pbr["baseColorFactor"][:3]], [0.2, 0.4, 0.8]
        )

    def test_a_shared_texture_is_embedded_once_across_channels(self):
        """One file on disk, one embedded copy — the embed cache spans the session."""
        png = self._png()
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.set_glb_base_color(session, {"Body": {"texture": png}})
            MeshConvert.set_glb_emissive(session, {"Body": {"texture": png}})

        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 1)
        mat = gltf["materials"][0]
        self.assertEqual(
            mat["pbrMetallicRoughness"]["baseColorTexture"]["index"],
            mat["emissiveTexture"]["index"],
        )

    def test_an_embed_reuses_bytes_the_glb_already_carries(self):
        """The converter's own embedded media must not be base64'd in a second time.

        The embed cache is keyed by source PATH, so it can only dedupe embeds
        made this session — it cannot see that the FBX->GLB conversion already
        wrote the very same texture into the BIN chunk. Since the sidecar
        re-applies exactly the channels FBX translation drops, the normal case
        was two copies of the same bytes, the second inflated ~33% by base64.
        Measured on a production room: 23 MB duplicated, 31 MB on disk, a
        quarter of the deliverable.
        """
        png = self._png()
        raw = open(png, "rb").read()
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "Body"}],
                "images": [{"bufferView": 0, "mimeType": "image/png", "name": "c.png"}],
                "textures": [{"source": 0}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(raw)}],
                "buffers": [{"byteLength": len(raw)}],
            },
            bin_chunk=raw,
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.set_glb_base_color(session, {"Body": {"texture": png}})

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        self.assertEqual(len(gltf["images"]), 1, "the payload was embedded twice")
        # A "no data: URI" check no longer bites here — the relocation pass
        # would turn a duplicate embed into a second bufferView, not a second
        # base64 blob. The BIN's size is what still catches it.
        self.assertEqual(
            bytes(edit.bin_data).count(raw),
            1,
            "a second copy of the payload reached the BIN",
        )
        index = gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"][
            "index"
        ]
        self.assertEqual(gltf["textures"][index]["source"], 0)

    def test_two_source_files_with_identical_bytes_embed_once(self):
        """Path-keyed dedupe misses this; content-keyed catches it.

        Duplicated textures under different names are routine in a DCC source
        tree, and neither copy need be in the GLB already — so the reuse index
        has to include what this session appends, not only what it opened.
        """
        import shutil

        first = self._png("a.png")
        second = os.path.join(self.tmp, "b.png")
        shutil.copyfile(first, second)
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "One"}, {"name": "Two"}],
            }
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.set_glb_base_color(
                session, {"One": {"texture": first}, "Two": {"texture": second}}
            )

        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 1, "identical bytes embedded twice")
        indices = {
            m["pbrMetallicRoughness"]["baseColorTexture"]["index"]
            for m in gltf["materials"]
        }
        self.assertTrue(
            all(gltf["textures"][i]["source"] == 0 for i in indices),
            "both materials must resolve to the single embedded image",
        )

    def test_an_embedded_image_lands_in_the_bin_not_the_json(self):
        """An embedded map ships as a bufferView, not base64 in the JSON chunk.

        Measured on a delivered asset (TURRETS_WIRES.glb): the sidecar's packed
        ORM rode in as a ``data:`` URI while the converter's own maps sat in the
        BIN, putting 4.0 MB of base64 in the JSON chunk -- 45% of an 8.9 MB
        deliverable, 1.0 MB of it pure base64 overhead, and every byte of it
        parsed before a loader can draw anything. The 33% premium was priced for
        a local preview; ``create_glb`` ships the same path as a deliverable.
        """
        png = self._png()
        raw = open(png, "rb").read()
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.set_glb_base_color(session, {"Body": {"texture": png}})

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        image = gltf["images"][0]
        self.assertNotIn("uri", image, "the payload is still base64 in the JSON")
        view = gltf["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        self.assertEqual(
            bytes(edit.bin_data[start : start + view["byteLength"]]),
            raw,
            "the relocated bytes must be the source PNG, verbatim",
        )
        self.assertEqual(gltf["buffers"][0]["byteLength"], len(bytes(edit.bin_data)))

    def test_relocating_an_embed_preserves_existing_bin_payloads(self):
        """Appending must not disturb a byte or an offset already in the BIN.

        Recomputing existing offsets is the part of GLB surgery that silently
        corrupts a file -- which is why the embed took the JSON in the first
        place. Appending past the end is what makes the BIN safe to touch:
        every prior view keeps its index, offset and bytes.
        """
        geometry = b"GEOMETRY" * 64
        png = self._png()
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "Body"}],
                "bufferViews": [
                    {"buffer": 0, "byteOffset": 0, "byteLength": len(geometry)}
                ],
                "buffers": [{"byteLength": len(geometry)}],
                "accessors": [
                    {
                        "bufferView": 0,
                        "componentType": 5126,
                        "count": 1,
                        "type": "SCALAR",
                    }
                ],
            },
            bin_chunk=geometry,
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.set_glb_base_color(session, {"Body": {"texture": png}})

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        self.assertEqual(gltf["bufferViews"][0]["byteOffset"], 0)
        self.assertEqual(gltf["bufferViews"][0]["byteLength"], len(geometry))
        self.assertEqual(gltf["accessors"][0]["bufferView"], 0)
        self.assertEqual(bytes(edit.bin_data)[: len(geometry)], geometry)
        self.assertGreater(len(gltf["bufferViews"]), 1, "no view was appended")

    def test_an_external_buffer_is_left_alone(self):
        """Buffer 0 is the BIN only when it declares no ``uri``.

        Appending to a GLB whose first buffer is EXTERNAL would strand the new
        views on bytes the file does not carry and overwrite that buffer's
        byteLength. Base64 is a size cost; a corrupt buffer table is not.
        """
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "Body"}],
                "buffers": [{"uri": "geometry.bin", "byteLength": 64}],
            }
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.set_glb_base_color(session, {"Body": {"texture": self._png()}})

        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(gltf["buffers"][0]["uri"], "geometry.bin")
        self.assertEqual(gltf["buffers"][0]["byteLength"], 64)
        self.assertTrue(gltf["images"][0]["uri"].startswith("data:"))
        self.assertNotIn("bufferView", gltf["images"][0])

    def test_an_unknown_trailing_chunk_is_never_dropped(self):
        """No BIN, but bytes after the JSON: an extension chunk, left intact.

        The relocation swaps the whole tail for a BIN (``replace_rest``), so
        writing one here would delete a chunk the spec tells clients to ignore
        rather than discard -- and this class never read it, so it cannot put
        it back. Same trade as the external buffer above: base64 is a size
        cost, silently losing bytes is not.
        """
        payload = b"\x01\x02\x03\x04" * 4
        extra = struct.pack("<I4s", len(payload), b"XTRA") + payload
        path = os.path.join(self.tmp, "extra_chunk.glb")
        gltf = json.dumps(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        ).encode("utf-8")
        gltf += b" " * ((4 - (len(gltf) % 4)) % 4)
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(gltf) + len(extra)))
            f.write(struct.pack("<I4s", len(gltf), b"JSON") + gltf)
            f.write(extra)

        with MeshConvert.open_glb(path) as session:
            MeshConvert.set_glb_base_color(session, {"Body": {"texture": self._png()}})

        edit = MeshConvert._read_glb(path)
        self.assertEqual(edit.rest, extra, "the trailing chunk was rewritten")
        self.assertTrue(edit.gltf["images"][0]["uri"].startswith("data:"))
        self.assertNotIn("bufferView", edit.gltf["images"][0])

    def test_an_embedded_texture_carries_its_name(self):
        """The embed names its image; the texture sampling it went out unnamed.

        FBX2glTF names the textures it writes, so an unnamed one in the middle
        of the list is a tell that a later pass added it -- and it costs the
        only human-readable handle on which slot the map serves.
        """
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.set_glb_base_color(session, {"Body": {"texture": self._png()}})

        gltf = MeshConvert._read_glb(path).gltf
        self.assertTrue(
            gltf["textures"][0].get("name"), "the embedded texture has no name"
        )

    def test_an_embed_of_new_bytes_still_lands(self):
        """The reuse path must not swallow a texture the file does NOT already have."""
        existing = self._png("existing.png")
        fresh = self._png("fresh.png")
        from PIL import Image

        Image.new("RGBA", (2, 2), (0, 255, 0, 255)).save(fresh)
        raw = open(existing, "rb").read()
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "Body"}],
                "images": [{"bufferView": 0, "mimeType": "image/png", "name": "e.png"}],
                "textures": [{"source": 0}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(raw)}],
                "buffers": [{"byteLength": len(raw)}],
            },
            bin_chunk=raw,
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.set_glb_base_color(session, {"Body": {"texture": fresh}})

        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 2)

    def test_optimize_repacks_images_and_preserves_geometry(self):
        """WebP re-encode + BIN repack: image shrinks, geometry bytes survive.

        The geometry view sits AFTER the image view on purpose — its offset
        must be recomputed when the image payload shrinks, and a stale offset
        reads garbage that no schema validator would catch.
        """
        from PIL import Image

        big = Image.new("RGB", (256, 256))
        for y in range(256):
            for x in range(0, 256, 16):
                big.putpixel((x, y), (x, y, 128))
        buffer = io.BytesIO()
        big.save(buffer, format="PNG")
        png = buffer.getvalue()
        geometry = bytes(range(256)) * 4  # recognizable, order-sensitive

        pad = (4 - len(png) % 4) % 4
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"bufferView": 0, "mimeType": "image/png", "name": "c.png"}],
                "textures": [{"source": 0}],
                "bufferViews": [
                    {"buffer": 0, "byteOffset": 0, "byteLength": len(png)},
                    {
                        "buffer": 0,
                        "byteOffset": len(png) + pad,
                        "byteLength": len(geometry),
                    },
                ],
                "buffers": [{"byteLength": len(png) + pad + len(geometry)}],
            },
            bin_chunk=png + b"\x00" * pad + geometry,
        )
        summary = MeshConvert.optimize_glb_textures(path, max_size=64)
        self.assertEqual(summary["images"], 1)
        self.assertLess(summary["bytes_after"], summary["bytes_before"])

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        image = gltf["images"][0]
        self.assertEqual(image["mimeType"], "image/webp")
        blob = edit.bin_data
        img_view = gltf["bufferViews"][image["bufferView"]]
        webp = bytes(
            blob[
                img_view["byteOffset"] : img_view["byteOffset"] + img_view["byteLength"]
            ]
        )
        self.assertEqual(webp[:4], b"RIFF", "payload must be a real WebP container")
        resized = Image.open(io.BytesIO(webp))
        self.assertEqual(max(resized.size), 64)

        geo_view = gltf["bufferViews"][1]
        survived = bytes(
            blob[
                geo_view["byteOffset"] : geo_view["byteOffset"] + geo_view["byteLength"]
            ]
        )
        self.assertEqual(survived, geometry, "geometry bytes corrupted by the repack")
        self.assertIn("EXT_texture_webp", gltf.get("extensionsUsed", []))
        self.assertEqual(
            gltf["textures"][0]["extensions"]["EXT_texture_webp"]["source"], 0
        )

    def test_optimize_relocates_data_uris_and_exempts_lightmaps(self):
        """A data-URI image lands in the BIN; a lightmap resists the resize."""
        import base64 as b64

        from PIL import Image

        def png_bytes(size):
            im = Image.new("RGB", (size, size), (200, 180, 40))
            buffer = io.BytesIO()
            im.save(buffer, format="PNG")
            return buffer.getvalue()

        source = png_bytes(128)
        lightmap = png_bytes(128)
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "extras": {
                    "lightmap_web": {"materials": {"M": {"map": "room_Lightmap.png"}}}
                },
                "images": [
                    {
                        "name": "source.png",
                        "uri": "data:image/png;base64,"
                        + b64.b64encode(source).decode("ascii"),
                        "mimeType": "image/png",
                    },
                    {
                        "name": "room_Lightmap.png",
                        "uri": "data:image/png;base64,"
                        + b64.b64encode(lightmap).decode("ascii"),
                        "mimeType": "image/png",
                    },
                ],
                "textures": [{"source": 0}, {"source": 1}],
            }
        )
        summary = MeshConvert.optimize_glb_textures(path, max_size=64)
        self.assertEqual(summary["images"], 2)

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        for image in gltf["images"]:
            self.assertNotIn("uri", image, "data URI should have moved into the BIN")
            self.assertIn("bufferView", image)
        blob = edit.bin_data

        def decode(image):
            view = gltf["bufferViews"][image["bufferView"]]
            raw = bytes(
                blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
            )
            return Image.open(io.BytesIO(raw))

        self.assertEqual(max(decode(gltf["images"][0]).size), 64, "source must resize")
        self.assertEqual(
            max(decode(gltf["images"][1]).size), 128, "lightmap must keep its size"
        )

    def test_metallic_roughness_summary_counts_only_what_it_wrote(self):
        """The foreign-packing headline must not claim work that never happened.

        It is derived from the sources that actually reached a material, not from
        the input spec: ``_match_glb_materials`` drops entries naming a material
        the GLB does not have, and a pack failure drops more. Counting the input
        made the headline say "2 MSAO maps" when one of the two materials was
        never touched.
        """
        from PIL import Image

        msao_present = os.path.join(self.tmp, "present_MSAO.png")
        msao_absent = os.path.join(self.tmp, "absent_MSAO.png")
        Image.new("RGBA", (8, 8), (200, 60, 0, 100)).save(msao_present)
        Image.new("RGBA", (8, 8), (10, 20, 0, 30)).save(msao_absent)

        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "present"}],
            }
        )
        with self.assertLogs(
            "pythontk.file_utils.mesh_convert._mesh_convert", level="WARNING"
        ) as caught:
            records = MeshConvert.set_glb_metallic_roughness(
                path,
                {
                    "present": {"metallic": msao_present},
                    "absent": {"metallic": msao_absent},
                },
            )
        self.assertEqual(len(records), 1, "only the matched material is written")
        headline = [m for m in caught.output if "non-glTF mask packing" in m]
        self.assertEqual(len(headline), 1, f"one headline expected: {caught.output}")
        self.assertIn("1 MSAO map", headline[0])
        self.assertNotIn("2 MSAO", headline[0])

    def test_optimize_splits_a_shared_bufferview_when_owners_diverge(self):
        """Two images can share ONE bufferView, and then need different bytes.

        FBX2glTF really emits this (measured: 4 such pairs on a production room
        GLB). Co-owners differ in the one thing that decides their encoding --
        a lightmap is exempt from the resize and encodes lossless, its co-owner
        is not -- so recording a single owner per view handed the loser the
        winner's bytes. With the lightmap winning, its co-owner silently kept
        full resolution; with the order reversed, the LIGHTMAP got resized and
        lossy-encoded, which is exactly the corruption the structural exemption
        exists to prevent. Asserted in BOTH orders, because the old bug was
        invisible in one of them.
        """
        import random

        from PIL import Image

        rng = random.Random(11)
        source = Image.new("RGB", (128, 128))
        source.putdata([(rng.randrange(256),) * 3 for _ in range(128 * 128)])
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
        png = buffer.getvalue()

        for names in (
            ["source.png", "room_Lightmap.png"],
            ["room_Lightmap.png", "source.png"],
        ):
            path = self._write_glb(
                {
                    "asset": {"version": "2.0"},
                    "extras": {
                        "lightmap_web": {
                            "materials": {"M": {"map": "room_Lightmap.png"}}
                        }
                    },
                    # Both images point at bufferView 0.
                    "images": [
                        {"bufferView": 0, "mimeType": "image/png", "name": name}
                        for name in names
                    ],
                    "textures": [{"source": i} for i in range(len(names))],
                    "bufferViews": [
                        {"buffer": 0, "byteOffset": 0, "byteLength": len(png)}
                    ],
                    "buffers": [{"byteLength": len(png)}],
                },
                bin_chunk=png + b"\x00" * ((4 - len(png) % 4) % 4),
            )
            MeshConvert.optimize_glb_textures(path, max_size=64)

            edit = MeshConvert._read_glb(path)
            gltf, blob = edit.gltf, edit.bin_data
            by_name = {}
            for image in gltf["images"]:
                view = gltf["bufferViews"][image["bufferView"]]
                by_name[image["name"]] = bytes(
                    blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
                )
            self.assertNotEqual(
                by_name["source.png"],
                by_name["room_Lightmap.png"],
                f"co-owners must not share one encoding (order {names})",
            )
            self.assertEqual(
                max(Image.open(io.BytesIO(by_name["source.png"])).size),
                64,
                f"source must resize (order {names})",
            )
            self.assertEqual(
                max(Image.open(io.BytesIO(by_name["room_Lightmap.png"])).size),
                128,
                f"lightmap must keep its size (order {names})",
            )
            # VP8L is the lossless WebP chunk; VP8 (no L) is the lossy one.
            self.assertEqual(
                by_name["room_Lightmap.png"][12:16],
                b"VP8L",
                f"lightmap must encode lossless (order {names})",
            )

    def test_optimize_is_byte_identical_across_worker_counts(self):
        """The concurrent encode pass must be a pure speedup, not a new variable.

        The encode is ~60% of the pass and Pillow releases the GIL through it,
        so it runs on a thread pool (measured on a production room GLB: 31.8s ->
        7.1s). Threads make the ordering of the per-image work nondeterministic,
        and the pass also *dedupes* identical payloads across images -- so if a
        result were ever attributed to the wrong image index, the artifact would
        differ run to run with nothing failing. Pin the whole file: serial and
        concurrent must produce the same bytes.

        Distinct payloads AND a duplicated one, so both the encode path and the
        dedupe fan-out are covered.
        """
        import random

        from PIL import Image

        def noisy_png(seed, size=96):
            rng = random.Random(seed)
            im = Image.new("RGB", (size, size))
            im.putdata(
                [
                    (rng.randrange(256), rng.randrange(256), rng.randrange(256))
                    for _ in range(size * size)
                ]
            )
            buffer = io.BytesIO()
            im.save(buffer, format="PNG")
            return buffer.getvalue()

        payloads = [noisy_png(seed) for seed in range(5)]
        payloads.append(payloads[0])  # a duplicate: two images, one encode

        views, chunks, offset = [], [], 0
        for png in payloads:
            views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(png)})
            padded = png + b"\x00" * ((4 - len(png) % 4) % 4)
            chunks.append(padded)
            offset += len(padded)
        gltf = {
            "asset": {"version": "2.0"},
            "images": [
                {"bufferView": i, "mimeType": "image/png", "name": f"t{i}.png"}
                for i in range(len(payloads))
            ],
            "textures": [{"source": i} for i in range(len(payloads))],
            "bufferViews": views,
            "buffers": [{"byteLength": offset}],
        }
        bin_chunk = b"".join(chunks)

        digests, summaries = [], []
        for workers in (1, 4):
            path = self._write_glb(json.loads(json.dumps(gltf)), bin_chunk=bin_chunk)
            summaries.append(
                MeshConvert.optimize_glb_textures(path, max_size=48, workers=workers)
            )
            digests.append(hashlib.sha256(open(path, "rb").read()).hexdigest())

        self.assertEqual(
            digests[0],
            digests[1],
            "concurrent encode produced a different GLB than the serial path",
        )
        self.assertEqual(summaries[0], summaries[1])
        self.assertEqual(summaries[0]["images"], len(payloads))

    def test_optimize_encodes_lightmaps_lossless(self):
        """Lightmaps re-encode LOSSLESS (VP8L), sources stay lossy (VP8).

        Lossy WebP is YUV 4:2:0 -- on a lightmap's near-black texels the
        chroma quantization reads as magenta/green blotching, and the 2x2
        chroma blocks smear color across atlas rect borders. The exemption
        must therefore cover the ENCODE, not just the resize.
        """
        import base64 as b64
        import random

        from PIL import Image

        def noisy_png(seed):
            rng = random.Random(seed)
            im = Image.new("RGB", (64, 64))
            im.putdata(
                [
                    (rng.randrange(256), rng.randrange(256), rng.randrange(256))
                    for _ in range(64 * 64)
                ]
            )
            buffer = io.BytesIO()
            im.save(buffer, format="PNG")
            return im, buffer.getvalue()

        lm_img, lm_png = noisy_png(1)
        _src_img, src_png = noisy_png(2)
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "extras": {
                    "lightmap_web": {"materials": {"M": {"map": "room_Lightmap.png"}}}
                },
                "images": [
                    {
                        "name": "room_Lightmap.png",
                        "uri": "data:image/png;base64,"
                        + b64.b64encode(lm_png).decode("ascii"),
                        "mimeType": "image/png",
                    },
                    {
                        "name": "source.png",
                        "uri": "data:image/png;base64,"
                        + b64.b64encode(src_png).decode("ascii"),
                        "mimeType": "image/png",
                    },
                ],
                "textures": [{"source": 0}, {"source": 1}],
            }
        )
        MeshConvert.optimize_glb_textures(path)

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        blob = edit.bin_data

        def payload(image):
            view = gltf["bufferViews"][image["bufferView"]]
            return bytes(
                blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
            )

        lm_bytes = payload(gltf["images"][0])
        self.assertEqual(lm_bytes[12:16], b"VP8L", "lightmap must be lossless WebP")
        self.assertEqual(
            Image.open(io.BytesIO(lm_bytes)).convert("RGB").tobytes(),
            lm_img.tobytes(),
            "lossless lightmap must round-trip pixel-exact",
        )
        self.assertEqual(
            payload(gltf["images"][1])[12:16], b"VP8 ", "source stays lossy"
        )

    def test_optimize_exempts_by_binding_when_the_name_lies(self):
        """Structural exemption: an image bound as a texCoord-1 occlusion map
        is a lightmap whatever its name says (digest dedupe can hand a
        lightmap payload another image's name), so it must keep its size."""
        import base64 as b64

        from PIL import Image

        im = Image.new("RGB", (128, 128), (10, 12, 14))
        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "images": [
                    {
                        "name": "definitely_not_a_lightmap.png",
                        "uri": "data:image/png;base64,"
                        + b64.b64encode(buffer.getvalue()).decode("ascii"),
                        "mimeType": "image/png",
                    }
                ],
                "textures": [{"source": 0}],
                "materials": [
                    {
                        "name": "M",
                        "occlusionTexture": {"index": 0, "texCoord": 1},
                    }
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path, max_size=64)
        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
        raw = bytes(
            edit.bin_data[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
        )
        self.assertEqual(
            max(Image.open(io.BytesIO(raw)).size),
            128,
            "texCoord-1 occlusion image must not resize",
        )

    def test_metallic_roughness_repair_replaces_a_white_orm(self):
        """The packed ORM must carry the REAL maps, blue channel = metallic.

        The production failure this section exists for: FBX2glTF packs a
        solid-white ORM when it cannot resolve the source maps, and glTF reads
        metallic from blue — so the material renders metallic=1, whose diffuse
        response is zero, and a lightmap (diffuse-only) lights nothing.
        """
        from PIL import Image

        rough = os.path.join(self.tmp, "rough.png")
        metal = os.path.join(self.tmp, "metal.png")
        Image.new("L", (4, 4), 128).save(rough)
        Image.new("L", (4, 4), 10).save(metal)
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [
                    {
                        "name": "Room",
                        "pbrMetallicRoughness": {
                            "metallicRoughnessTexture": {"index": 0},
                            "metallicFactor": 1.0,
                        },
                    }
                ],
                "textures": [{"source": 0}],
                "images": [{"name": "white_orm.png"}],
            }
        )
        with MeshConvert.open_glb(path) as session:
            records = MeshConvert.set_glb_metallic_roughness(
                session, {"Room": {"roughness": rough, "metallic": metal}}
            )
        self.assertEqual(len(records), 1)

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        pbr = gltf["materials"][0]["pbrMetallicRoughness"]
        tex = pbr["metallicRoughnessTexture"]["index"]
        img = gltf["images"][gltf["textures"][tex]["source"]]
        import io as iolib

        view = gltf["bufferViews"][img["bufferView"]]
        start = view.get("byteOffset", 0)
        raw = bytes(edit.bin_data[start : start + view["byteLength"]])
        pixels = Image.open(iolib.BytesIO(raw)).convert("RGB").getpixel((1, 1))
        self.assertEqual(pixels[2], 10, "blue channel must be the metallic map")
        self.assertEqual(pixels[1], 128, "green channel must be the roughness map")
        self.assertEqual(pixels[0], 255, "red (occlusion) fills white when absent")
        self.assertEqual(pbr["metallicFactor"], 1.0)

    def test_metallic_roughness_binds_occlusion_to_the_packed_orm(self):
        """The AO packed into R must actually be sampled: glTF reads occlusion
        ONLY from ``occlusionTexture``, so an ORM bound solely as
        ``metallicRoughnessTexture`` ships its AO as dead payload (measured on
        a delivered preview GLB: 0 of 57 materials sampled it). Three cases,
        one rule each:

        * a free slot binds (the spec's packed-ORM idiom -- same image, both
          slots);
        * a slot still naming the converted ORM this write replaces is
          REPOINTED (FBX2glTF binds its own packing there, and the stale
          reference samples the -- often solid-white -- image it emitted);
        * a separate authored AO map is left alone.
        """
        from PIL import Image

        ao = os.path.join(self.tmp, "ao.png")
        Image.new("L", (4, 4), 200).save(ao)
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [
                    {"name": "free"},
                    {
                        "name": "stale",
                        "pbrMetallicRoughness": {
                            "metallicRoughnessTexture": {"index": 0}
                        },
                        "occlusionTexture": {"index": 0},
                    },
                    {
                        "name": "authored",
                        "pbrMetallicRoughness": {
                            "metallicRoughnessTexture": {"index": 0}
                        },
                        "occlusionTexture": {"index": 1},
                    },
                ],
                "textures": [{"source": 0}, {"source": 1}],
                "images": [{"name": "fbx2gltf_orm.png"}, {"name": "real_ao.png"}],
            }
        )
        spec = {"occlusion": ao}
        with MeshConvert.open_glb(path) as session:
            MeshConvert.set_glb_metallic_roughness(
                session,
                {"free": dict(spec), "stale": dict(spec), "authored": dict(spec)},
            )
        gltf = MeshConvert._read_glb(path).gltf
        by_name = {m["name"]: m for m in gltf["materials"]}
        for case in ("free", "stale"):
            mat = by_name[case]
            packed = mat["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]
            self.assertEqual(
                mat["occlusionTexture"]["index"], packed, f"{case}: slot must bind"
            )
        self.assertEqual(
            by_name["authored"]["occlusionTexture"]["index"],
            1,
            "an authored separate AO map must never be displaced",
        )

    def test_prune_unreferenced_textures_drops_the_displaced_converted_orm(self):
        """The image the ORM repack displaces must not ship as dead payload.

        Measured on a production delivery (HOOKS_PINS.glb): after
        ``set_glb_metallic_roughness`` rebound both slots to its packed image,
        FBX2glTF's ``ao_met_rough_<mat>`` stayed in ``images``/``textures`` and
        its full-size PNG in the BIN -- 2 MB the reviewer flagged as an
        orphaned texture, per material, per export.
        """
        from PIL import Image

        rough = os.path.join(self.tmp, "rough.png")
        Image.new("L", (4, 4), 128).save(rough)
        path, geometry = _converted_orm_glb(self.tmp)
        with MeshConvert.open_glb(path) as session:
            MeshConvert.set_glb_metallic_roughness(
                session, {"Room": {"roughness": rough}}
            )
            dropped = MeshConvert.prune_glb_unreferenced_textures(session)
        self.assertEqual(dropped["images"], 1)
        gltf = _assert_no_dead_payload(self, path, geometry)
        # And the repacked ORM is still what both slots sample.
        mat = gltf["materials"][0]
        self.assertEqual(
            mat["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"],
            mat["occlusionTexture"]["index"],
        )

    def test_prune_unreferenced_textures_is_a_no_op_on_a_clean_file(self):
        path, geometry = _converted_orm_glb(self.tmp)
        with open(path, "rb") as f:
            before = f.read()
        with MeshConvert.open_glb(path) as session:
            dropped = MeshConvert.prune_glb_unreferenced_textures(session)
        self.assertEqual(dropped, {"textures": 0, "images": 0, "bytes": 0})
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before, "clean file must not be rewritten")

    def test_prune_unreferenced_textures_keeps_a_view_another_owner_reads(self):
        """A bufferView shared with a live image (FBX2glTF does emit two images
        on one view) or with an accessor stays; only its exclusive owner goes."""
        path, geometry = _converted_orm_glb(self.tmp)
        with MeshConvert.open_glb(path) as session:
            gltf = session.gltf
            # A second, LIVE image on the same view, sampled as base colour.
            gltf["images"].append(
                {"name": "twin", "bufferView": 0, "mimeType": "image/png"}
            )
            gltf["textures"].append({"source": 1, "sampler": 0})
            mat = gltf["materials"][0]
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 1}
            # Unbind the converted ORM from both slots.
            del mat["pbrMetallicRoughness"]["metallicRoughnessTexture"]
            del mat["occlusionTexture"]
            session.dirty = True
            dropped = MeshConvert.prune_glb_unreferenced_textures(session)
        self.assertEqual((dropped["textures"], dropped["images"]), (1, 1))
        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        self.assertEqual([i["name"] for i in gltf["images"]], ["twin"])
        self.assertEqual(len(gltf["bufferViews"]), 2, "the shared view must stay")
        self.assertEqual(
            gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"],
            0,
        )
        view = gltf["bufferViews"][gltf["accessors"][0]["bufferView"]]
        self.assertEqual(
            bytes(
                edit.bin_data[
                    view["byteOffset"] : view["byteOffset"] + view["byteLength"]
                ]
            ),
            geometry,
        )

    def test_prune_unreferenced_textures_bails_under_an_image_referring_extension(self):
        """EXT_lights_image_based names images from the root; pruning under it
        would renumber indices the walk cannot see. Left alone, loudly."""
        path, geometry = _converted_orm_glb(self.tmp)
        with MeshConvert.open_glb(path) as session:
            gltf = session.gltf
            gltf["extensionsUsed"] = ["EXT_lights_image_based"]
            gltf["extensions"] = {
                "EXT_lights_image_based": {"lights": [{"specularImages": [[0]]}]}
            }
            del gltf["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"]
            del gltf["materials"][0]["occlusionTexture"]
            session.dirty = True
            dropped = MeshConvert.prune_glb_unreferenced_textures(session)
        self.assertEqual(dropped, {"textures": 0, "images": 0, "bytes": 0})
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 1)
        self.assertEqual(len(gltf["bufferViews"]), 2)

    def test_prune_unreferenced_textures_bails_when_a_view_lives_outside_buffer_0(self):
        """The BIN rebuild re-slices every kept view out of the embedded
        buffer; a view on an external buffer would keep ``buffer`` but get a
        byteOffset into the rebuilt BIN. Bail whole, like the extension case."""
        path, geometry = _converted_orm_glb(self.tmp)
        with MeshConvert.open_glb(path) as session:
            gltf = session.gltf
            gltf.setdefault("buffers", []).append({"uri": "ext.bin", "byteLength": 4})
            gltf["bufferViews"].append({"buffer": 1, "byteOffset": 0, "byteLength": 4})
            del gltf["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"]
            del gltf["materials"][0]["occlusionTexture"]
            session.dirty = True
            dropped = MeshConvert.prune_glb_unreferenced_textures(session)
        self.assertEqual(dropped, {"textures": 0, "images": 0, "bytes": 0})
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 1)
        self.assertEqual(len(gltf["bufferViews"]), 3)

    def test_metallic_roughness_shared_sources_pack_once(self):
        """N materials naming the same maps must cost one pack + one embed."""
        from PIL import Image

        rough = os.path.join(self.tmp, "r2.png")
        Image.new("L", (4, 4), 90).save(rough)
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "A"}, {"name": "B"}],
            }
        )
        spec = {"roughness": rough}
        with MeshConvert.open_glb(path) as session:
            records = MeshConvert.set_glb_metallic_roughness(
                session, {"A": dict(spec), "B": dict(spec)}
            )
        self.assertEqual(len(records), 2)
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 1, "shared sources embedded twice")

    def test_an_out_of_range_index_is_skipped_not_wrapped(self):
        """A negative glTF index must be rejected, not resolved from the end.

        Indices are non-negative by spec, so a negative one means a malformed
        file — but Python indexing would quietly hand back the *last* texture
        and report a finding against a material that has nothing wrong with it.
        """
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [
                    {
                        "name": "Bad",
                        "alphaMode": "BLEND",
                        "pbrMetallicRoughness": {"baseColorTexture": {"index": -1}},
                    }
                ],
                "textures": [{"source": 0}],
                "images": [{"name": "real.png"}],
            }
        )
        self.assertEqual(MeshConvert.check_glb_materials(path), [])

    def test_a_raising_body_writes_nothing(self):
        """A half-applied edit must not reach disk."""
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )
        before = open(path, "rb").read()
        with self.assertRaises(RuntimeError):
            with MeshConvert.open_glb(path) as session:
                MeshConvert.set_glb_emissive(session, {"Body": {"color": (1, 1, 1)}})
                raise RuntimeError("interrupted mid-edit")
        self.assertEqual(open(path, "rb").read(), before)


@unittest.skipUnless(
    os.environ.get("PYTHONTK_INTEGRATION_TESTS") == "1",
    "Set PYTHONTK_INTEGRATION_TESTS=1 to run network/install integration tests.",
)
class TestRealInstall(unittest.TestCase):
    """End-to-end install. Downloads FBX2glTF (~3.7 MB)."""

    def test_install_and_invoke_help(self):
        binary = MeshConvert.resolve_binary(auto_install=True, prompt=False)
        self.assertTrue(os.path.isfile(binary), f"binary missing: {binary}")
        result = subprocess.run(
            [binary, "--help"], capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("FBX2glTF", result.stdout)


def _looks_like_a_file_path(text):
    """A path-shaped string, as distinct from prose that happens to contain "/".

    HANDOFF_INSTRUCTIONS says "metallic/roughness/occlusion", so a bare
    separator test mistakes the contract text for a texture path and "scrubs"
    it to its last clause.
    """
    if chr(10) in text or len(text) > 260:
        return False
    if os.sep not in text and "/" not in text:
        return False
    return bool(re.match(r"^\.[A-Za-z0-9]{1,6}$", os.path.splitext(text)[1]))


class TestSceneSidecar(unittest.TestCase):
    """The scene-sidecar grid's converter column: build, apply+embed, read."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_sidecar_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write_glb(self, name="scene.glb", materials=None):
        """A JSON-chunk-only GLB with named materials (no BIN needed)."""
        gltf = {"asset": {"version": "2.0"}}
        if materials is not None:
            gltf["materials"] = materials
        payload = json.dumps(gltf).encode("utf-8")
        payload += b" " * ((4 - (len(payload) % 4)) % 4)
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(
                b"glTF"
                + struct.pack("<I", 2)
                + struct.pack("<I", 12 + 8 + len(payload))
                + struct.pack("<I", len(payload))
                + b"JSON"
                + payload
            )
        return path

    def _envelope(self, sections):
        return MeshConvert.build_scene_sidecar(
            sections, source={"application": "test", "version": "0"}, asset="scene.fbx"
        )

    def _assert_embeds(self, glb, envelope):
        """The embedded envelope is the caller's, scrubbed, plus resolution keys.

        Not equality, for two reasons. Only the apply pass can know which glTF
        image each authoring path became, so it embeds a COPY carrying
        ``textures`` and ``validate``. And the shipped copy names textures by
        FILE NAME -- the authoring directory is deliberately not carried, so a
        hand-off does not spell out the folder tree it was built from.

        The caller's dict must survive both untouched: the bridges keep that
        object unscrubbed, and THAT copy is the one that should still carry
        full provenance on the authoring machine.
        """
        embedded = MeshConvert.read_scene_sidecar(glb)
        self.assertEqual(set(embedded) - set(envelope), {"textures", "validate"})

        # Derive the expectation from the CALLER's envelope rather than from
        # the embedded map (whose keys are already scrubbed, which would make
        # the comparison circular): every path-shaped string the caller wrote
        # should come back as its file name, and nothing else should move.
        expected = {k: v for k, v in envelope.items()}
        paths = set()

        def collect(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    collect(k)
                    collect(v)
            elif isinstance(node, list):
                for v in node:
                    collect(v)
            elif isinstance(node, str) and _looks_like_a_file_path(node):
                paths.add(node)

        collect(expected)
        expected["textures"] = {path: None for path in paths}
        MeshConvert._scrub_sidecar_paths(expected)
        expected.pop("textures")
        self.assertEqual({k: v for k, v in embedded.items() if k in envelope}, expected)

        self.assertNotIn("textures", envelope, "caller's envelope was mutated")
        return embedded

    @staticmethod
    def _ref(embedded, path):
        """Look a texture reference up the way a reader must: by file name."""
        return embedded["textures"][os.path.basename(path)]

    def test_apply_scene_sidecar_prunes_what_its_appliers_displace(self):
        """The applier tail sweeps the images the section writers unbound, and
        the embedded texture map is built AFTER the sweep so its image indices
        describe the delivered file."""
        from PIL import Image

        rough = os.path.join(self.tmp, "rough.png")
        Image.new("L", (4, 4), 128).save(rough)
        path, geometry = _converted_orm_glb(self.tmp)
        envelope = self._envelope(
            {"metallic_roughness": {"Room": {"roughness": rough}}}
        )
        MeshConvert.apply_scene_sidecar(path, envelope)
        gltf = _assert_no_dead_payload(self, path, geometry)
        embedded = MeshConvert.read_scene_sidecar(path)
        mat = gltf["materials"][0]
        bound = gltf["textures"][
            mat["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]
        ]["source"]
        self.assertEqual(self._ref(embedded, rough)["image"], bound)

    def test_build_scene_sidecar_owns_the_frozen_top_level(self):
        """Standalone readers parse against exactly these keys."""
        envelope = self._envelope({"emissive": {"m": {"color": [1, 0, 0]}}})
        self.assertEqual(
            set(envelope),
            {"version", "source", "asset", "color_space", "sections", "handoff"},
        )
        self.assertEqual(envelope["version"], MeshConvert.SIDECAR_VERSION)
        self.assertEqual(envelope["color_space"], "linear")
        self.assertEqual(envelope["asset"], "scene.fbx")

    def test_handoff_states_the_contract_a_standalone_reader_needs(self):
        """The rule that only lives in the artifact, not in a doc: a section's
        texture path is provenance, and ``textures`` is what resolves it.

        Asserted on substance rather than wording -- the instructions have to
        name the trap (paths do not resolve) and the escape (the map), or an
        agent handed one .glb will go looking on a filesystem. ``reads`` is
        derived from the registries so a section or manifest added later cannot
        fall out of the contract silently.
        """
        handoff = self._envelope({})["handoff"]
        text = handoff["instructions"].lower()
        self.assertIn("provenance only", text)
        self.assertIn("textures", text)
        self.assertIn("lightmap", text)
        self.assertEqual(handoff["sections"], sorted(MeshConvert.SIDECAR_APPLIERS))
        self.assertIn(f"extras.{MeshConvert.LIGHTMAP_WEB_KEY}", handoff["reads"])

    def test_apply_resolves_every_authoring_path_to_an_embedded_image(self):
        """The content-addressed join: path -> glTF image index + sha256.

        Covers the ORM wrinkle too -- the packer's cache key is the three
        source paths joined by ``|`` because the trio collapses into ONE image,
        so all three must resolve, to the same index, and the digest must match
        the bytes actually embedded.
        """
        from PIL import Image

        rough = os.path.join(self.tmp, "r.png")
        metal = os.path.join(self.tmp, "m.png")
        emit = os.path.join(self.tmp, "e.png")
        Image.new("L", (4, 4), 90).save(rough)
        Image.new("L", (4, 4), 10).save(metal)
        Image.new("RGB", (4, 4), (5, 6, 7)).save(emit)

        glb = self._write_glb(materials=[{"name": "m"}])
        envelope = self._envelope(
            {
                "emissive": {"m": {"texture": emit}},
                "metallic_roughness": {"m": {"roughness": rough, "metallic": metal}},
            }
        )
        MeshConvert.apply_scene_sidecar(glb, envelope)
        embedded = self._assert_embeds(glb, envelope)

        refs = embedded["textures"]
        for path in (rough, metal, emit):
            self.assertIn(
                os.path.basename(path),
                refs,
                "every authoring texture must resolve, by file name",
            )
            self.assertNotIn(path, refs, "the authoring directory must not ship")
        self.assertEqual(
            refs[os.path.basename(rough)]["image"],
            refs[os.path.basename(metal)]["image"],
            "the ORM trio collapses into one image",
        )
        self.assertNotEqual(
            refs[os.path.basename(emit)]["image"],
            refs[os.path.basename(rough)]["image"],
        )
        self.assertNotIn("None", refs, "an absent slot is not a path")

        # Digests must address the bytes really present, resolved the way a
        # standalone reader would: sidecar path -> image index -> payload.
        with MeshConvert.open_glb(glb) as edit:
            for path, ref in refs.items():
                payload = edit._image_payload(edit.images[ref["image"]])
                self.assertEqual(
                    ref["sha256"],
                    hashlib.sha256(payload).hexdigest(),
                    f"{path}: digest must match the embedded payload",
                )
                self.assertEqual(ref["bytes"], len(payload))
        self.assertEqual(
            embedded["validate"]["sections"], {"emissive": 1, "metallic_roughness": 1}
        )
        self.assertEqual(embedded["validate"]["textures"], len(refs))
        # NOT the file's totals: the lightmap applier runs after this in every
        # production path and adds images, so a total stamped here would be
        # stale on arrival and a reader checking it would reject a good file.
        self.assertNotIn("images", embedded["validate"])

    def test_a_path_used_by_two_channels_resolves_deterministically(self):
        """One path CAN become two images -- embedded whole for one channel and
        folded into an ORM for another -- and a single-valued map can only say
        one. First wins, fixed by SIDECAR_APPLIERS order, so the answer does not
        depend on which applier happened to run last."""
        from PIL import Image

        shared = os.path.join(self.tmp, "shared.png")
        Image.new("RGB", (4, 4), (9, 9, 9)).save(shared)
        glb = self._write_glb(materials=[{"name": "m"}])
        envelope = self._envelope(
            {
                "emissive": {"m": {"texture": shared}},
                "metallic_roughness": {"m": {"roughness": shared}},
            }
        )
        MeshConvert.apply_scene_sidecar(glb, envelope)
        refs = MeshConvert.read_scene_sidecar(glb)["textures"]

        with MeshConvert.open_glb(glb) as edit:
            emissive_image = edit.image_for_texture(
                edit.gltf["materials"][0]["emissiveTexture"]["index"]
            )
            packed_image = edit.image_for_texture(
                edit.gltf["materials"][0]["pbrMetallicRoughness"][
                    "metallicRoughnessTexture"
                ]["index"]
            )
        self.assertNotEqual(emissive_image, packed_image, "two distinct embeds")
        self.assertEqual(
            refs[os.path.basename(shared)]["image"],
            emissive_image,
            "the whole-image embed wins over a channel of a pack",
        )

    def test_apply_survives_a_malformed_section_when_sizing_validate(self):
        """A null section is skipped by the dispatch, so sizing it for
        ``validate`` must not be the thing that raises: the container catch
        handles OSError/ValueError only, so a TypeError from ``len(None)`` would
        abort an apply that was otherwise entirely fine."""
        glb = self._write_glb(materials=[{"name": "m"}])
        envelope = self._envelope(
            {"emissive": None, "base_color": {"m": {"color": [0, 1, 0]}}}
        )
        summary = MeshConvert.apply_scene_sidecar(glb, envelope)
        self.assertEqual(summary, {"base_color": "1 of 1"})
        embedded = MeshConvert.read_scene_sidecar(glb)
        self.assertEqual(embedded["validate"]["sections"], {"base_color": 1})

    def test_optimize_restamps_digests_it_invalidated(self):
        """The lifetime bug this guards: ``optimize_glb_textures`` re-encodes
        every image, so a digest taken at apply time describes bytes the
        delivered file no longer holds. Content addressing that does not
        address the delivered content is worse than none -- a reader that
        verifies would reject a perfectly good artifact."""
        from PIL import Image

        big = os.path.join(self.tmp, "big.png")
        Image.new("RGB", (128, 128), (30, 60, 90)).save(big)
        glb = self._write_glb(materials=[{"name": "m"}])
        envelope = self._envelope({"emissive": {"m": {"texture": big}}})
        MeshConvert.apply_scene_sidecar(glb, envelope)
        key = os.path.basename(big)
        before = MeshConvert.read_scene_sidecar(glb)["textures"][key]["sha256"]

        MeshConvert.optimize_glb_textures(glb, max_size=32)
        after = MeshConvert.read_scene_sidecar(glb)["textures"][key]
        self.assertNotEqual(after["sha256"], before, "re-encode must restamp")
        with MeshConvert.open_glb(glb) as edit:
            payload = edit._image_payload(edit.images[after["image"]])
        self.assertEqual(after["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(after["bytes"], len(payload))
        self.assertEqual(after["mimeType"], "image/webp")

    def test_apply_dispatches_sections_and_embeds_the_envelope(self):
        """Apply + embed happen in one session; the artifact self-describes."""
        glb = self._write_glb(materials=[{"name": "m"}])
        envelope = self._envelope({"emissive": {"m": {"color": [1, 0, 0]}}})
        summary = MeshConvert.apply_scene_sidecar(glb, envelope)
        self.assertEqual(summary, {"emissive": "1 of 1"})

        with MeshConvert.open_glb(glb) as edit:
            self.assertEqual(
                edit.gltf["materials"][0]["emissiveFactor"], [1.0, 0.0, 0.0]
            )
            self.assertEqual(edit.gltf["extras"]["scene_sidecar_applied"], summary)
        self._assert_embeds(glb, envelope)

    def test_apply_composes_with_an_open_session(self):
        """Session form: the owner writes once; apply must not write for itself."""
        glb = self._write_glb(materials=[{"name": "m"}])
        with MeshConvert.open_glb(glb) as edit:
            summary = MeshConvert.apply_scene_sidecar(
                edit, self._envelope({"base_color": {"m": {"color": [0, 0.5, 0]}}})
            )
        self.assertEqual(summary, {"base_color": "1 of 1"})
        with MeshConvert.open_glb(glb) as edit:
            self.assertEqual(
                edit.gltf["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"],
                [0.0, 0.5, 0.0, 1.0],
            )

    def test_apply_none_is_a_true_noop(self):
        glb = self._write_glb(materials=[{"name": "m"}])
        before = open(glb, "rb").read()
        self.assertEqual(MeshConvert.apply_scene_sidecar(glb, None), {})
        self.assertEqual(open(glb, "rb").read(), before)
        self.assertIsNone(MeshConvert.read_scene_sidecar(glb))

    def test_apply_empty_sections_still_embeds(self):
        """ "Sidecar on, nothing to carry" must be visible in the artifact."""
        glb = self._write_glb(materials=[{"name": "m"}])
        envelope = self._envelope({})
        self.assertEqual(MeshConvert.apply_scene_sidecar(glb, envelope), {})
        self._assert_embeds(glb, envelope)

    def test_unknown_sections_are_skipped_never_fatal(self):
        """Forward compatibility: a reader skips sections it does not know."""
        glb = self._write_glb(materials=[{"name": "m"}])
        envelope = self._envelope({"lights": {"key": {"intensity": 5}}})
        self.assertEqual(MeshConvert.apply_scene_sidecar(glb, envelope), {})
        self._assert_embeds(glb, envelope)

    def test_fbx_to_glb_sidecar_param_applies_in_the_conversion_pass(self):
        """The one-parameter path every production caller uses."""
        src = os.path.join(self.tmp, "model.fbx")
        with open(src, "wb") as fh:
            fh.write(b"fake-fbx")
        dst = os.path.join(self.tmp, "model.glb")
        envelope = self._envelope({"emissive": {"m": {"color": [0, 1, 0]}}})

        def fake_run(cmd, **kwargs):
            self._write_glb("model.glb", materials=[{"name": "m"}])

            class R:
                returncode = 0
                stdout = ""
                stderr = ""

            return R()

        with (
            patch("shutil.which", return_value=os.path.join(self.tmp, "bin")),
            patch("subprocess.run", side_effect=fake_run),
        ):
            out = MeshConvert.fbx_to_glb(src, dst, overwrite=True, sidecar=envelope)

        self._assert_embeds(out, envelope)
        with MeshConvert.open_glb(out) as edit:
            self.assertEqual(
                edit.gltf["materials"][0]["emissiveFactor"], [0.0, 1.0, 0.0]
            )


class TestVerifyGlb(unittest.TestCase):
    """The reader a RECIPIENT runs against a delivered GLB.

    Everything the envelope promises is checkable from the file alone -- that
    is what `textures` and `validate` are for -- but nothing read either back
    until this existed, so a truncated envelope arrived indistinguishable from
    a good one.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_verify_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _delivered(self, name="scene.glb"):
        """A GLB carrying a real applied envelope: one material, one texture."""
        src = os.path.join(self.tmp, "map.png")
        from PIL import Image

        Image.new("RGB", (4, 4), (10, 200, 30)).save(src)
        path = os.path.join(self.tmp, name)
        _write_glb_file(
            path, {"asset": {"version": "2.0"}, "materials": [{"name": "m"}]}
        )
        envelope = MeshConvert.build_scene_sidecar(
            {"base_color": {"m": {"texture": src}}},
            source={"application": "maya", "version": "2025"},
            asset="scene.fbx",
        )
        MeshConvert.apply_scene_sidecar(path, envelope)
        return path

    def test_a_clean_deliverable_verifies(self):
        """The pass case has to be reachable, or every report reads as noise."""
        report = MeshConvert.verify_glb(self._delivered())

        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["problems"], [])
        self.assertEqual(report["envelope"]["version"], MeshConvert.SIDECAR_VERSION)
        self.assertEqual(report["envelope"]["source"]["application"], "maya")
        self.assertEqual(report["textures"]["verified"], report["textures"]["checked"])
        self.assertGreater(report["textures"]["checked"], 0)
        self.assertIn("pythontk", report["generator"])

    def test_a_payload_swapped_after_stamping_is_caught(self):
        """The digest is the only thing that can catch this.

        Counts still agree, the index still resolves, the envelope still reads
        as complete -- and the bytes a viewer renders are not the bytes that
        were approved.
        """
        path = self._delivered()
        import base64
        from io import BytesIO
        from PIL import Image

        buf = BytesIO()
        Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, format="PNG")
        swapped = base64.b64encode(buf.getvalue()).decode("ascii")
        with MeshConvert.open_glb(path) as edit:
            refs = edit.gltf["extras"]["scene_sidecar"]["textures"]
            index = next(iter(refs.values()))["image"]
            # The appliers embed as data URIs, so re-pointing the uri IS the
            # payload swap -- no BIN surgery needed to stage it.
            image = edit.gltf["images"][index]
            image.pop("bufferView", None)
            image["uri"] = f"data:image/png;base64,{swapped}"
            edit.dirty = True

        report = MeshConvert.verify_glb(path)

        self.assertFalse(report["ok"])
        self.assertEqual(len(report["textures"]["mismatched"]), 1)
        self.assertTrue(any("sha256" in p for p in report["problems"]))

    def test_a_truncated_envelope_is_caught_by_its_own_claim(self):
        """`validate` is the envelope's claim about itself.

        A section dropped out of a hand-edited envelope leaves every remaining
        entry internally consistent; only the recorded count disagrees.
        """
        path = self._delivered()
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["extras"]["scene_sidecar"]["sections"].clear()
            edit.dirty = True

        report = MeshConvert.verify_glb(path)

        self.assertFalse(report["ok"])
        self.assertTrue(any("validate" in p for p in report["problems"]))

    def test_a_partial_apply_is_not_read_as_a_failure(self):
        """ "10 of 20" contains "0 of" -- the outcome test has to be anchored.

        A section that matched most of its entries is a warning the apply pass
        already logged, not a broken deliverable; reporting it as one would
        make the verdict useless on any real scene, where a renamed material
        is routine.
        """
        path = self._delivered()
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["extras"]["scene_sidecar_applied"] = {"base_color": "10 of 20"}
            edit.dirty = True

        report = MeshConvert.verify_glb(path)

        self.assertEqual(
            [p for p in report["problems"] if "did not land" in p],
            [],
            "a partial apply was reported as a section that did not land",
        )

    def test_a_section_that_matched_nothing_is_reported(self):
        """The anchored test must still catch the real thing."""
        path = self._delivered()
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["extras"]["scene_sidecar_applied"] = {
                "base_color": "0 of 3 matched"
            }
            edit.dirty = True

        report = MeshConvert.verify_glb(path)

        self.assertFalse(report["ok"])
        self.assertTrue(any("did not land" in p for p in report["problems"]))

    def test_an_unvalidated_binding_is_a_note_not_a_problem(self):
        """`ok` and `problems` must stay strictly in step.

        An ORM the producer never had to repair is legitimate, so it cannot
        make a deliverable fail -- but a caller doing the obvious
        `if report["problems"]` would read it as a defect if it were filed
        there.
        """
        src = os.path.join(self.tmp, "map.png")
        from PIL import Image

        Image.new("RGB", (4, 4), (10, 200, 30)).save(src)
        path = os.path.join(self.tmp, "undescribed.glb")
        _write_glb_file(
            path, {"asset": {"version": "2.0"}, "materials": [{"name": "m"}]}
        )
        # An envelope that describes a DIFFERENT channel, so the ORM binding
        # added below is real but undescribed.
        envelope = MeshConvert.build_scene_sidecar(
            {"base_color": {"m": {"texture": src}}},
            source={"application": "maya", "version": "2025"},
            asset="scene.fbx",
        )
        MeshConvert.apply_scene_sidecar(path, envelope)
        with MeshConvert.open_glb(path) as edit:
            pbr = edit.gltf["materials"][0].setdefault("pbrMetallicRoughness", {})
            pbr["metallicRoughnessTexture"] = {"index": 0}
            edit.dirty = True

        report = MeshConvert.verify_glb(path)

        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["problems"], [])
        self.assertTrue(any("not described" in n for n in report["notes"]))
        self.assertEqual(
            report["orm"]["m"]["finding"], MeshConvert.ORM_FINDING_UNVALIDATED
        )

    def test_a_malformed_sections_block_is_reported_not_raised(self):
        """A report-only method must not raise on the shape it is reporting on.

        `sections` was already isinstance-guarded where the counts are built,
        then dereferenced unguarded to read `metallic_roughness` -- so a
        hand-edited (or future-schema) envelope whose block is a list raised
        `AttributeError` out of the one method a recipient runs to find that
        out. Reported as a PROBLEM rather than normalised to `{}`: an envelope
        this reader could not parse must never come back `ok`.
        """
        path = self._delivered()
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["extras"]["scene_sidecar"]["sections"] = ["base_color"]
            edit.dirty = True

        report = MeshConvert.verify_glb(path)

        self.assertFalse(report["ok"])
        self.assertTrue(
            any("'sections' is not an object" in p for p in report["problems"])
        )
        self.assertEqual(report["sections"]["declared"], {})

    def test_a_malformed_textures_block_is_reported_not_raised(self):
        """Same shape one key over: `textures` was dereferenced with `.items()`."""
        path = self._delivered()
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["extras"]["scene_sidecar"]["textures"] = ["map.png"]
            edit.dirty = True

        report = MeshConvert.verify_glb(path)

        self.assertFalse(report["ok"])
        self.assertTrue(
            any("'textures' is not an object" in p for p in report["problems"])
        )
        self.assertEqual(report["textures"]["checked"], 0)

    def test_a_foreign_glb_is_reported_not_raised(self):
        """Pointing this at someone else's asset is a legitimate thing to do."""
        path = os.path.join(self.tmp, "foreign.glb")
        _write_glb_file(path, {"asset": {"version": "2.0"}})

        report = MeshConvert.verify_glb(path)

        self.assertFalse(report["ok"])
        self.assertIsNone(report["envelope"])
        self.assertTrue(
            any("no scene-sidecar envelope" in p for p in report["problems"])
        )


class TestSuspectOrmMaterials(unittest.TestCase):
    """The detector for the ORM the sidecar did NOT repair.

    FBX2glTF white-fills a grayscale ("L"-mode) PBR source, and glTF reads
    metallic from the ORM's blue channel -- so the converter's packing renders
    metallic=1: no diffuse response, and pure black under a lightmap (which
    contributes to diffuse alone). `set_glb_metallic_roughness` repairs the
    materials the sidecar names; these tests cover the ones it does not reach,
    which shipped that failure with nothing said.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_orm_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    @staticmethod
    def _png(color, size=(4, 4)):
        from io import BytesIO
        from PIL import Image

        buf = BytesIO()
        Image.new("RGB", size, color).save(buf, format="PNG")
        return buf.getvalue()

    def _glb(self, materials, blobs, name="scene.glb"):
        """A GLB binding one image per texture, in order."""
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(
                TestCheckGlbMaterials._build_glb(
                    materials=materials,
                    images=[
                        {"bufferView": i, "mimeType": "image/png", "name": f"img{i}"}
                        for i in range(len(blobs))
                    ],
                    textures=[{"source": i} for i in range(len(blobs))],
                    image_blobs=blobs,
                )
            )
        return path

    @staticmethod
    def _orm_material(name, texture, **pbr):
        mat = {"name": name, "pbrMetallicRoughness": dict(pbr)}
        mat["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": texture}
        return mat

    def test_flags_only_the_material_whose_metallic_channel_is_full(self):
        """White ORM -> flagged; a real packing -> not. The rest is noise."""
        path = self._glb(
            materials=[
                self._orm_material("white", 0),
                self._orm_material("packed", 1),
                {"name": "no_orm"},
            ],
            blobs=[self._png((255, 255, 255)), self._png((255, 180, 20))],
        )

        found = MeshConvert.suspect_orm_materials(path)

        self.assertEqual(list(found), ["white"])
        self.assertEqual(
            found["white"]["finding"], MeshConvert.ORM_FINDING_METALLIC_FULL
        )
        self.assertEqual(found["white"]["image"], "img0")

    def test_a_zero_metallic_factor_cancels_the_texture(self):
        """metallicFactor 0 multiplies the white channel to nothing.

        The material renders as a non-metal, so flagging it would send an
        artist after a file that is behaving exactly as authored.
        """
        path = self._glb(
            materials=[self._orm_material("inert", 0, metallicFactor=0)],
            blobs=[self._png((255, 255, 255))],
        )

        self.assertEqual(MeshConvert.suspect_orm_materials(path), {})

    def test_excluded_names_are_not_probed(self):
        """A material the repair covers is the packer's business, not this.

        Its ORM comes from the authoring maps rather than from the converter,
        and `pack_orm_texture` has already reported per map. Excluding by name
        is also what keeps a fully-covered export from decoding anything.
        """
        path = self._glb(
            materials=[self._orm_material("covered", 0)],
            blobs=[self._png((255, 255, 255))],
        )

        self.assertEqual(
            MeshConvert.suspect_orm_materials(path, described={"covered"}), {}
        )

    def test_an_undescribed_binding_is_reported_even_when_it_looks_ordinary(self):
        """The case a whiteness test is blind to.

        A mask map packed for another engine (Unity's is R=Metallic,
        G=Occlusion, B=Detail) that reaches the GLB unrepaired is read channel
        for channel as ORM and misinterpreted -- while looking like perfectly
        ordinary image data. Nothing about its pixels says so; what says so is
        that the envelope never described the binding.
        """
        path = self._glb(
            materials=[self._orm_material("foreign", 0)],
            blobs=[self._png((40, 190, 90))],  # nothing uniform, nothing white
        )

        found = MeshConvert.suspect_orm_materials(path, described=set())

        self.assertEqual(
            found["foreign"]["finding"], MeshConvert.ORM_FINDING_UNVALIDATED
        )

    def test_nothing_is_unvalidated_when_the_caller_says_nothing(self):
        """`described=None` must not report every material in the file.

        Only the caller knows what was described; absent that, "unvalidated"
        is not a claim this can make, and making it anyway would fire on every
        material of every asset from another producer.
        """
        path = self._glb(
            materials=[self._orm_material("foreign", 0)],
            blobs=[self._png((40, 190, 90))],
        )

        self.assertEqual(MeshConvert.suspect_orm_materials(path), {})

    def test_glb_orm_layout_matches_the_registry(self):
        """The spec constant and the authored taxonomy must not drift apart.

        `GLTF_ORM_CHANNELS` is held as glTF's own constant rather than looked
        up from `MapRegistry`, so an edit to the registry cannot silently
        change what the check tests. This is the tie that makes the two
        visible to each other -- if the registry's ORM layout ever stops
        matching the spec, that is a finding, not a silent divergence.
        """
        from pythontk import MapRegistry

        self.assertEqual(
            MapRegistry().get("ORM").channels, MeshConvert.GLTF_ORM_CHANNELS
        )
        # And the channel the harm test reads really is the metallic one.
        self.assertEqual(
            MeshConvert.GLTF_ORM_CHANNELS[MeshConvert._ORM_HARMFUL_CHANNEL], "Metallic"
        )

    def test_apply_scene_sidecar_reports_what_its_sections_left_behind(self):
        """The warning has to fire where the deliverable is actually built.

        A producer whose sidecar covers only some materials is the measured
        production case -- the covered room looks right, the omitted material
        ships black -- so the check runs inside the apply pass, against the
        section's own name set.
        """
        path = self._glb(
            materials=[
                self._orm_material("repaired", 0),
                self._orm_material("omitted", 0),
            ],
            blobs=[self._png((255, 255, 255))],
        )
        envelope = MeshConvert.build_scene_sidecar(
            {"metallic_roughness": {"repaired": {"metallic": "m.png"}}},
            source={"application": "test", "version": "0"},
            asset="scene.fbx",
        )

        with self.assertLogs(
            "pythontk.file_utils.mesh_convert._mesh_convert", level="WARNING"
        ) as caught:
            MeshConvert.apply_scene_sidecar(path, envelope)

        flagged = [m for m in caught.output if "Metallic=1 everywhere" in m]
        self.assertEqual(len(flagged), 1)
        self.assertIn("omitted", flagged[0])
        # The excluded name must not be listed -- deliberately not "covered",
        # which is a word the message itself uses.
        self.assertNotIn("repaired", flagged[0])

    def test_the_export_time_warning_stays_off_the_unvalidated_finding(self):
        """Coverage information is not a defect, and must not warn at export.

        An artist cannot act on "nothing described this binding" mid-export,
        and a warning nobody can act on is how the actionable ones stop being
        read. It reaches the recipient through `verify_glb` instead.
        """
        path = self._glb(
            materials=[self._orm_material("undescribed", 0)],
            blobs=[self._png((40, 190, 90))],
        )
        envelope = MeshConvert.build_scene_sidecar(
            {"emissive": {"other": {"color": [0, 1, 0]}}},
            source={"application": "test", "version": "0"},
            asset="scene.fbx",
        )

        with self.assertLogs(
            "pythontk.file_utils.mesh_convert._mesh_convert", level="WARNING"
        ) as caught:
            logging.getLogger("pythontk.file_utils.mesh_convert._mesh_convert").warning(
                "sentinel so assertLogs always has a record"
            )
            MeshConvert.apply_scene_sidecar(path, envelope)

        self.assertEqual(
            [m for m in caught.output if "Metallic=1" in m or "unvalidated" in m], []
        )
        # Not vacuous: the finding really is there, and only the LOG is quiet.
        with MeshConvert.open_glb(path) as edit:
            found = MeshConvert.suspect_orm_materials(edit, described=set())
        self.assertEqual(
            found["undescribed"]["finding"], MeshConvert.ORM_FINDING_UNVALIDATED
        )


class TestAssetGeneratorStamp(unittest.TestCase):
    """`asset.generator` is the one provenance field glTF itself defines.

    Every viewer and inspector displays it, so it reaches a recipient who opens
    the deliverable in a tool that is not ours and reads nothing else we write.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_generator_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _glb(self, gltf, name="scene.glb"):
        path = os.path.join(self.tmp, name)
        _write_glb_file(path, gltf)
        return path

    def _envelope(self):
        return MeshConvert.build_scene_sidecar(
            {"emissive": {"m": {"color": [0, 1, 0]}}},
            source={"application": "maya", "version": "2025"},
            asset="scene.fbx",
        )

    def test_stamp_names_the_authoring_app_and_this_package(self):
        """Coarse on purpose: the app the envelope already names, and us.

        No host, no user, no paths -- a generator string travels to whoever
        gets the file.
        """
        from pythontk import __version__ as ptk_version

        path = self._glb({"asset": {"version": "2.0"}, "materials": [{"name": "m"}]})
        MeshConvert.apply_scene_sidecar(path, self._envelope())

        with MeshConvert.open_glb(path) as edit:
            generator = edit.gltf["asset"]["generator"]
        self.assertIn("maya 2025", generator)
        self.assertIn(f"pythontk {ptk_version}", generator)
        self.assertNotIn(os.getcwd(), generator)

    def test_an_application_without_a_version_stamps_no_literal_None(self):
        """An f-string over a missing version writes "None" mid-string.

        `.strip()` cannot reach it there. Every in-repo producer supplies a
        version (`cmds.about(version=True)` / `bpy.app.version_string`), so this
        reaches only a caller of the PUBLIC `build_scene_sidecar(source=...)` --
        which is exactly the caller who would then ship "someApp None + pythontk
        0.9.24" in the one provenance field glTF itself defines.
        """
        path = self._glb({"asset": {"version": "2.0"}, "materials": [{"name": "m"}]})
        envelope = MeshConvert.build_scene_sidecar(
            {"emissive": {"m": {"color": [0, 1, 0]}}},
            source={"application": "someApp"},
            asset="scene.fbx",
        )
        MeshConvert.apply_scene_sidecar(path, envelope)

        with MeshConvert.open_glb(path) as edit:
            generator = edit.gltf["asset"]["generator"]
        self.assertNotIn("None", generator)
        self.assertIn("someApp", generator)

    def test_a_version_without_an_application_is_not_dropped(self):
        """The inverse gate discarded the only fact the source did carry."""
        path = self._glb({"asset": {"version": "2.0"}, "materials": [{"name": "m"}]})
        envelope = MeshConvert.build_scene_sidecar(
            {"emissive": {"m": {"color": [0, 1, 0]}}},
            source={"version": "2025"},
            asset="scene.fbx",
        )
        MeshConvert.apply_scene_sidecar(path, envelope)

        with MeshConvert.open_glb(path) as edit:
            generator = edit.gltf["asset"]["generator"]
        self.assertIn("2025", generator)
        self.assertNotIn("None", generator)

    def test_the_converters_own_claim_is_kept(self):
        """FBX2glTF really did produce the geometry.

        Replacing that claim would lose the fact that matters most when a mesh
        arrives wrong, so ours is appended to it.
        """
        path = self._glb(
            {
                "asset": {"version": "2.0", "generator": "FBX2glTF"},
                "materials": [{"name": "m"}],
            }
        )
        MeshConvert.apply_scene_sidecar(path, self._envelope())

        with MeshConvert.open_glb(path) as edit:
            self.assertTrue(edit.gltf["asset"]["generator"].startswith("FBX2glTF via "))

    def test_reapplying_stacks_nothing_when_there_was_no_converter_claim(self):
        """A GLB whose only generator claim is a previous stamp of ours.

        There is no separator to split on in that case, so the whole string
        comes back as the "prior" claim and the next pass appends ours to
        itself. Reachable from any producer that writes no generator of its
        own and re-applies an envelope.
        """
        path = self._glb({"asset": {"version": "2.0"}, "materials": [{"name": "m"}]})
        MeshConvert.apply_scene_sidecar(path, self._envelope())
        MeshConvert.apply_scene_sidecar(path, self._envelope())

        with MeshConvert.open_glb(path) as edit:
            generator = edit.gltf["asset"]["generator"]
        self.assertEqual(generator.count("pythontk"), 1, generator)
        self.assertNotIn(" via ", generator)

    def test_reapplying_refreshes_the_stamp_rather_than_stacking(self):
        """Passes compose: a GLB can be re-opened and re-applied.

        A stamp that appended every time would grow the generator string one
        entry per pass, and a reader could not tell which was current.
        """
        path = self._glb(
            {
                "asset": {"version": "2.0", "generator": "FBX2glTF"},
                "materials": [{"name": "m"}],
            }
        )
        MeshConvert.apply_scene_sidecar(path, self._envelope())
        MeshConvert.apply_scene_sidecar(path, self._envelope())

        with MeshConvert.open_glb(path) as edit:
            generator = edit.gltf["asset"]["generator"]
        self.assertEqual(generator.count(" via "), 1)


class TestSidecarPathScrub(unittest.TestCase):
    """The shipped envelope must not spell out the authoring folder tree.

    Section entries and the ``textures`` map both named each texture by its
    ABSOLUTE authoring path, so a GLB handed to an external developer disclosed
    the client directory structure it was built from. The two sides of that join
    are the same string, so both are rewritten to the file name together -- the
    lookup still resolves and no reader has to change.
    """

    CLIENT = r"O:\Dropbox (BigClient)\Job"

    def _envelope(self):
        return {
            "materials": [
                {"name": "body", "baseColor": self.CLIENT + r"\tex\body_BC.png"},
                {"name": "trim", "baseColor": self.CLIENT + r"\alt\body_BC.png"},
            ],
            "textures": {
                self.CLIENT + r"\tex\body_BC.png": {"image": 0},
                self.CLIENT + r"\alt\body_BC.png": {"image": 1},
                self.CLIENT + r"\tex\body_N.png": {"image": 2},
            },
        }

    def test_the_authoring_tree_is_gone(self):
        envelope = self._envelope()
        MeshConvert._scrub_sidecar_paths(envelope)

        blob = json.dumps(envelope)
        self.assertNotIn("Dropbox", blob)
        self.assertNotIn("BigClient", blob)
        self.assertNotIn("\\", blob)

    def test_the_join_still_resolves(self):
        """A scrub that broke the lookup would be worse than the disclosure."""
        envelope = self._envelope()
        MeshConvert._scrub_sidecar_paths(envelope)

        for material in envelope["materials"]:
            self.assertIn(material["baseColor"], envelope["textures"])

    def test_colliding_basenames_stay_distinct(self):
        """Two dirs can hold the same file name; they are different images."""
        envelope = self._envelope()
        MeshConvert._scrub_sidecar_paths(envelope)

        body, trim = envelope["materials"]
        self.assertNotEqual(body["baseColor"], trim["baseColor"])
        self.assertEqual(envelope["textures"][body["baseColor"]]["image"], 0)
        self.assertEqual(envelope["textures"][trim["baseColor"]]["image"], 1)

    def test_it_is_idempotent(self):
        """Already-scrubbed names are left alone (nothing to strip)."""
        envelope = self._envelope()
        MeshConvert._scrub_sidecar_paths(envelope)
        before = json.dumps(envelope, sort_keys=True)

        self.assertEqual(MeshConvert._scrub_sidecar_paths(envelope), 0)
        self.assertEqual(json.dumps(envelope, sort_keys=True), before)

    def test_a_missing_textures_map_is_not_an_error(self):
        envelope = {"materials": [{"name": "body"}]}
        self.assertEqual(MeshConvert._scrub_sidecar_paths(envelope), 0)


class TestGlbLightmaps(unittest.TestCase):
    """The self-feeding lightmap applier: committed bake -> GLB deliverable.

    The manifest travels INSIDE the GLB (FBX user properties -> node extras via
    FBX2glTF's --user-properties, probe-verified), so the applier takes no scene
    knowledge from the caller -- these tests hand-build that GLB shape and only
    ever pass file paths.
    """

    #: Cross-implementation golden value: a constant-0.5 linear EXR must encode
    #: with scalar 0.5 in BOTH this encoder and blendertk's bpy-I/O twin
    #: (test_web_export pins the same number), so the two cannot drift.
    GOLDEN_CONSTANT = 0.5

    @classmethod
    def setUpClass(cls):
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cv2 unavailable; HDR encode untestable")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_lightmap_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ helpers
    def _exr(self, name="room_Lightmap.exr", value=GOLDEN_CONSTANT):
        import cv2
        import numpy as np

        path = os.path.join(self.tmp, name)
        cv2.imwrite(path, np.full((8, 8, 3), value, dtype=np.float32))
        return path

    def _glb(self, gltf, name="scene.glb"):
        json_bytes = json.dumps(gltf).encode("utf-8")
        json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
        blob = (
            struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_bytes))
            + struct.pack("<I4s", len(json_bytes), b"JSON")
            + json_bytes
        )
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(blob)
        return path

    def _scene(
        self,
        manifest,
        objects=("room",),
        texcoord1=True,
        material=0,
        nested=True,
    ):
        """A minimal lit-scene GLB: mesh nodes + the data_export carrier node.

        *nested* picks which of the two real on-disk carrier shapes the manifest
        rides: the FBX2glTF one (``extras.fromFBX.userProperties``, wrapped as
        ``{"type", "value"}``) or a native glTF export's top-level node extras.
        ``None`` writes no carrier at all -- a GLB whose per-node markers
        survived but whose manifest did not.
        """
        attrs = {"POSITION": 0, "TEXCOORD_0": 1}
        if texcoord1:
            attrs["TEXCOORD_1"] = 2
        nodes = [{"name": n, "mesh": i} for i, n in enumerate(objects)]
        if nested is None:
            carrier = {}
        elif nested:
            carrier = {
                "fromFBX": {
                    "userProperties": {
                        "lightmap_metadata": {
                            "type": "eFbxString",
                            "value": json.dumps(manifest),
                        }
                    }
                }
            }
        else:
            carrier = {"lightmap_metadata": json.dumps(manifest)}
        nodes.append({"name": "data_export", "extras": carrier})
        return {
            "asset": {"version": "2.0"},
            "nodes": nodes,
            "meshes": [
                {"primitives": [{"attributes": dict(attrs), "material": material}]}
                for _ in objects
            ],
            "materials": [{"name": "roomMat"}, {"name": "propMat"}],
        }

    def _manifest(self, entries, version=1):
        return {
            "version": version,
            "dir": self.tmp,  # the publisher's locate hint -- no caller paths
            "objects": entries,
        }

    # ------------------------------------------------------------------ encode
    def test_encode_golden_constant(self):
        """The cross-implementation pin: constant 0.5 -> scalar 0.5, texel 255."""
        import cv2
        import numpy as np

        from pythontk import ImgUtils

        png, scalar = ImgUtils.encode_hdr_for_web(self._exr())
        self.assertAlmostEqual(scalar, self.GOLDEN_CONSTANT, places=5)
        arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(int(arr.max()), 255)
        self.assertEqual(int(arr.min()), 255)  # constant in, constant out

    def test_encode_percentile_ignores_zero_texels(self):
        """Unbaked (zero) texels must not drag the divisor toward black."""
        import cv2
        import numpy as np

        from pythontk import ImgUtils

        img = np.zeros((8, 8, 3), dtype=np.float32)
        img[0, 0] = 2.0  # one lit texel in a sea of gutter
        path = os.path.join(self.tmp, "sparse.exr")
        cv2.imwrite(path, img)
        _png, scalar = ImgUtils.encode_hdr_for_web(path)
        self.assertAlmostEqual(scalar, 2.0, places=4)

    # ------------------------------------------------------------------ applier
    def test_binds_carrier_on_texcoord1_and_writes_viewer_manifest(self):
        exr = self._exr()
        glb = self._glb(
            self._scene(
                self._manifest([{"name": "room", "map": os.path.basename(exr)}])
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["material"], "roomMat")

        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        oc = gltf["materials"][0]["occlusionTexture"]
        self.assertEqual(oc["texCoord"], 1)
        img = gltf["images"][gltf["textures"][oc["index"]]["source"]]
        self.assertEqual(img["mimeType"], "image/png")
        self.assertIn("bufferView", img, "the carrier stayed base64 in the JSON")
        web = gltf["extras"]["lightmap_web"]
        # The exact contract preview_viewer.html parses.
        self.assertEqual(web["carrier"], "occlusion")
        self.assertEqual(web["uv"], 1)
        self.assertEqual(web["encoding"], "srgb")
        self.assertAlmostEqual(
            web["materials"]["roomMat"]["intensity"], self.GOLDEN_CONSTANT, places=5
        )

    def test_superseded_carriers_agree_with_the_authoritative_one(self):
        """Every surviving copy of the lightmap metadata must say the same thing.

        The GLB carries the same facts in several places. ``extras.lightmap_web``
        is written by this applier from the FINAL values (the embedded PNG and
        the scalar that restores the bake range); the per-node
        ``fromFBX.userProperties.lightmapInfo`` markers and the data_export
        node are written EARLIER, by the Maya bake pass, before normalisation
        exists. Measured on a client hand-off: lightmap_web said
        ``OFFICE_ENV_LightMap.png`` @ 13.65625 while all the others still said
        ``.exr`` @ 1.0 -- so a consumer trusting one of them rendered the bake
        ~13.7x too dark, and the wrong copies are the ones found FIRST (the
        per-node markers sit next to the mesh).

        NOT fixed by deleting the markers: the applier itself reads them to
        locate the EXR, and ``_reconcile_node_markers`` deliberately keeps
        ``map``/``uv_set``/``intensity``/``scaleOffset`` because a consumer acts
        on them. The defect is that those kept values are STALE, so the fix is
        to correct them in the walk that already rewrites every marker.
        """
        exr = self._exr()
        stale = {
            "map": os.path.basename(exr),
            "uv_set": "lightmap",
            "intensity": 1.0,
        }
        scene = self._scene(
            self._manifest([{"name": "room", "map": os.path.basename(exr)}])
        )
        scene["nodes"][0]["extras"] = {
            "fromFBX": {
                "userProperties": {
                    "lightmapInfo": {"type": "eFbxString", "value": json.dumps(stale)}
                }
            }
        }
        glb = self._glb(scene)

        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(len(records), 1, "the marker must still locate the EXR")

        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
            web = gltf["extras"]["lightmap_web"]["materials"]
            authoritative = next(iter(web.values()))
            marker = json.loads(
                gltf["nodes"][0]["extras"]["fromFBX"]["userProperties"]["lightmapInfo"][
                    "value"
                ]
            )

        # Guard the guard: if the encode ever produced 1.0 the assertions below
        # would pass on a GLB that never disagreed in the first place.
        self.assertNotEqual(
            authoritative["intensity"],
            stale["intensity"],
            "fixture is inert -- seed an intensity the encode will not reproduce",
        )

        self.assertEqual(
            marker["intensity"],
            authoritative["intensity"],
            "the per-node marker still reports the PRE-normalisation intensity, so "
            "a consumer reading it renders the bake at the wrong exposure",
        )
        self.assertEqual(
            marker["map"],
            authoritative["map"],
            "the per-node marker still names the .exr, which ships nowhere -- the "
            "only real copy is the embedded atlas",
        )
        # The payload a consumer acts on is still whole: this corrects values,
        # it does not strip the carrier.
        self.assertEqual(marker["uv_set"], "lightmap")

    def test_markers_are_corrected_on_a_NAMESPACED_node(self):
        """The correction must key off the RESOLVED node, not the manifest name.

        Node lookup here is deliberately namespace-tolerant: a manifest naming
        "room" binds a GLB node "NS:room", because manifests and exports can
        disagree about namespaces without either being wrong (a referenced Maya
        rack arrives namespaced). Keying the marker corrections off the MANIFEST
        name therefore misses on exactly the scenes that tolerance exists for,
        and misses SILENTLY -- the markers just keep their stale values, which
        is indistinguishable from having no correction at all.

        Asserted separately from the plain case because the plain case passes
        either way: in it the two names coincide.
        """
        exr = self._exr()
        stale = {"map": os.path.basename(exr), "uv_set": "lightmap", "intensity": 1.0}
        # Manifest says "room"...
        scene = self._scene(
            self._manifest([{"name": "room", "map": os.path.basename(exr)}])
        )
        # ...the GLB node is namespaced.
        scene["nodes"][0]["name"] = "NS:room"
        scene["nodes"][0]["extras"] = {
            "fromFBX": {
                "userProperties": {
                    "lightmapInfo": {"type": "eFbxString", "value": json.dumps(stale)}
                }
            }
        }
        glb = self._glb(scene)

        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(len(records), 1, "namespace-tolerant binding regressed")

        with MeshConvert.open_glb(glb) as edit:
            authoritative = next(
                iter(edit.gltf["extras"]["lightmap_web"]["materials"].values())
            )
            marker = json.loads(
                edit.gltf["nodes"][0]["extras"]["fromFBX"]["userProperties"][
                    "lightmapInfo"
                ]["value"]
            )
        self.assertEqual(
            marker["intensity"],
            authoritative["intensity"],
            "a namespaced node kept its pre-normalisation intensity -- the "
            "correction keyed off the manifest name instead of the bound node",
        )
        self.assertEqual(marker["map"], authoritative["map"])

    def test_top_level_extras_marker_shape_is_also_corrected(self):
        """There are TWO on-disk shapes of the per-node marker, both real.

        FBX2glTF nests user properties under
        ``extras.fromFBX.userProperties`` (the Maya route). blendertk's NATIVE
        glTF export writes the same marker as a TOP-LEVEL node extra -- verified
        on a real deliverable, ``OFFICE_ENV_both.glb``, whose nodes carry
        ``extras: {currentUVSet, lightmapInfo}`` with the same stale
        ``.exr`` @ ``intensity 1.0``. Walking only the nested shape skipped every
        Blender-authored GLB silently, and this is public API the preview server
        points at whatever GLB it is handed, so the shape cannot be known from
        the call site.
        """
        exr = self._exr()
        stale = {"map": os.path.basename(exr), "uv_set": "lightmap", "intensity": 1.0}
        scene = self._scene(
            self._manifest([{"name": "room", "map": os.path.basename(exr)}])
        )
        # The Blender-native shape: no fromFBX wrapper, no {"type","value"} box.
        scene["nodes"][0]["extras"] = {
            "currentUVSet": "map1",
            "lightmapInfo": json.dumps(stale),
        }
        glb = self._glb(scene)

        self.assertEqual(len(MeshConvert.apply_glb_lightmaps(glb)), 1)
        with MeshConvert.open_glb(glb) as edit:
            authoritative = next(
                iter(edit.gltf["extras"]["lightmap_web"]["materials"].values())
            )
            extras = edit.gltf["nodes"][0]["extras"]
            marker = json.loads(extras["lightmapInfo"])
        self.assertEqual(
            marker["intensity"],
            authoritative["intensity"],
            "a top-level (Blender-native) marker kept its stale intensity",
        )
        self.assertEqual(marker["map"], authoritative["map"])
        self.assertEqual(
            extras["currentUVSet"],
            "map1",
            "unrelated sibling extras must survive untouched",
        )

    def test_a_top_level_manifest_carrier_is_read(self):
        """The marker walk and the manifest probe must know the SAME two shapes.

        Correcting the top-level marker shape is unreachable while the gate that
        decides whether the applier runs at all -- ``_lightmap_manifest`` --
        probes only the nested one: a natively exported GLB carries both halves
        top-level, so the manifest read misses, ``apply_glb_lightmaps`` returns
        early, and the marker walk never happens. Asserted end to end (a binding
        AND a corrected marker) because the manifest probe on its own would pass
        on a file the applier still no-ops.
        """
        exr = self._exr()
        stale = {"map": os.path.basename(exr), "uv_set": "lightmap", "intensity": 1.0}
        scene = self._scene(
            self._manifest([{"name": "room", "map": os.path.basename(exr)}]),
            nested=False,
        )
        scene["nodes"][0]["extras"] = {"lightmapInfo": json.dumps(stale)}
        glb = self._glb(scene)

        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(
            len(records),
            1,
            "a natively exported GLB carries its manifest as a TOP-LEVEL node "
            "extra; probing only the FBX2glTF shape no-ops the whole applier",
        )
        with MeshConvert.open_glb(glb) as edit:
            authoritative = next(
                iter(edit.gltf["extras"]["lightmap_web"]["materials"].values())
            )
            marker = json.loads(edit.gltf["nodes"][0]["extras"]["lightmapInfo"])
        self.assertNotEqual(
            authoritative["intensity"],
            stale["intensity"],
            "fixture is inert -- seed an intensity the encode will not reproduce",
        )
        self.assertEqual(marker["intensity"], authoritative["intensity"])
        self.assertEqual(marker["map"], authoritative["map"])

    def test_markers_without_a_manifest_are_a_clean_no_op(self):
        """The documented limit of what this applier can repair.

        A deliverable can arrive carrying per-node markers and NO manifest at
        all -- blendertk's native glTF export writes exactly that today (checked
        on ``OFFICE_ENV_both.glb``: 7 nodes with a top-level ``lightmapInfo``,
        no ``lightmap_metadata`` anywhere). There is nothing to bind and nothing
        authoritative to correct the markers AGAINST, so they are left as found
        rather than guessed at or stripped, and the repair belongs in the
        producer. Pinned so the no-op is a stated contract rather than an
        accident of the early return.
        """
        exr = self._exr()
        stale = {"map": os.path.basename(exr), "uv_set": "lightmap", "intensity": 1.0}
        scene = self._scene(
            self._manifest([{"name": "room", "map": os.path.basename(exr)}]),
            nested=None,
        )
        scene["nodes"][0]["extras"] = {"lightmapInfo": json.dumps(stale)}
        glb = self._glb(scene)

        self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])
        self.assertIsNone(MeshConvert.read_glb_lightmap_manifest(glb))
        with MeshConvert.open_glb(glb) as edit:
            self.assertNotIn("lightmap_web", edit.gltf.get("extras") or {})
            self.assertEqual(
                json.loads(edit.gltf["nodes"][0]["extras"]["lightmapInfo"]), stale
            )

    def test_authoring_locate_hints_do_not_ship_in_the_glb(self):
        """The ``dir`` hint is build-time only -- use it, then strip it.

        The publisher stamps an ABSOLUTE authoring path into both the manifest
        (``dir``) and every per-object ``lightmapInfo`` marker, so the applier can
        find the EXR on the machine that baked it. Once the PNG is embedded that
        path has no remaining reader, but it used to ride into the deliverable --
        measured on a client hand-off, 49 copies of
        ``O:\\Dropbox (Client)\\...\\sourceimages`` in one shipped GLB, leaking the
        drive layout and the client's name to whoever received it.

        Both halves are asserted together on purpose: strip too early and the
        binding silently breaks (the map is located THROUGH this hint), so
        "no absolute paths" alone would pass on a GLB with no lightmap at all.
        """
        exr = self._exr()
        marker = {
            "map": os.path.basename(exr),
            "dir": self.tmp,
            "uv_set": "lightmap",
            "intensity": 1.0,
        }
        scene = self._scene(
            self._manifest([{"name": "room", "map": os.path.basename(exr)}])
        )
        scene["nodes"][0]["extras"] = {
            "fromFBX": {
                "userProperties": {
                    "lightmapInfo": {"type": "eFbxString", "value": json.dumps(marker)}
                }
            }
        }
        glb = self._glb(scene)

        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(len(records), 1, "the hint must still locate the EXR")

        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
            raw = json.dumps(gltf)
        # It bound -- so the strip below provably ran AFTER the hint was consumed.
        self.assertEqual(gltf["materials"][0]["occlusionTexture"]["texCoord"], 1)

        # Compared with backslashes removed. These paths sit inside JSON strings
        # nested in a JSON document, so the separator arrives re-escaped (``\\``,
        # then ``\\\\``) and a naive substring search for the plain path reports
        # CLEAN on a GLB that is still leaking it -- the exact false negative that
        # hid this in the first place.
        self.assertNotIn(
            self.tmp.replace("\\", ""),
            raw.replace("\\", ""),
            "the authoring directory shipped inside the GLB",
        )
        manifest = MeshConvert.read_glb_lightmap_manifest(glb) or {}
        self.assertNotIn("dir", manifest, "manifest kept its locate hint")
        node_marker = json.loads(
            gltf["nodes"][0]["extras"]["fromFBX"]["userProperties"]["lightmapInfo"][
                "value"
            ]
        )
        self.assertNotIn("dir", node_marker, "per-node marker kept its locate hint")
        # Scrub the leak, not the payload: what a consumer actually reads
        # survives -- and is CORRECTED to what actually shipped. The marker
        # arrives naming the source .exr, which is NOT in the deliverable; the
        # embedded PNG is, and that is what extras.lightmap_web names. Keeping
        # the .exr here would preserve a filename that resolves nowhere. See
        # test_superseded_carriers_agree_with_the_authoritative_one.
        self.assertEqual(
            node_marker["map"],
            os.path.splitext(os.path.basename(exr))[0] + ".png",
        )
        self.assertEqual(node_marker["uv_set"], "lightmap")

    def test_without_locate_hints_scrubs_a_data_export_snapshot(self):
        """The DCC-side scrub both export sidecars delegate to.

        mayatk and blendertk each write a sidecar straight from a
        ``DataNodes.dump(decode=True)`` snapshot; the rule for what may not ship
        lives here, with the key names and the glTF-side twin, so a second hint
        key cannot be added to one container and missed in the other.
        """
        authored = r"O:\Dropbox (Client)\Team Folder\PROD\sourceimages"
        snapshot = {
            "lightmap_metadata": {
                "version": 1,
                "dir": authored,
                "objects": [{"name": "room", "map": "room_Lightmap.exr"}],
            },
            "shot_metadata": {"shots": [1]},
        }
        out = MeshConvert.without_locate_hints(snapshot)
        self.assertNotIn("dir", out["lightmap_metadata"])
        # Scrubbed, not gutted -- and unrelated channels pass through untouched.
        self.assertEqual(
            out["lightmap_metadata"]["objects"],
            [{"name": "room", "map": "room_Lightmap.exr"}],
        )
        self.assertEqual(out["shot_metadata"], {"shots": [1]})
        # The caller's dump is theirs; a scrub for serialization must not edit it.
        self.assertEqual(snapshot["lightmap_metadata"]["dir"], authored)

    def test_without_locate_hints_passes_clean_snapshots_through(self):
        """No hint, no copy -- the common case must not churn the payload."""
        snapshot = {"lightmap_metadata": {"version": 1}, "shot_metadata": {}}
        self.assertIs(MeshConvert.without_locate_hints(snapshot), snapshot)
        empty: dict = {}
        self.assertIs(MeshConvert.without_locate_hints(empty), empty)

    def test_no_manifest_is_a_clean_noop(self):
        glb = self._glb(
            {
                "asset": {"version": "2.0"},
                "nodes": [{"name": "room", "mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "materials": [{"name": "roomMat"}],
            }
        )
        with open(glb, "rb") as f:
            before = f.read()
        self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])
        with open(glb, "rb") as f:
            self.assertEqual(f.read(), before, "no-op must not rewrite")

    def test_namespaced_node_binds_a_stripped_manifest_name(self):
        """Namespace tolerance, the direction that shipped black racks: an
        older publisher stripped ``NS:`` from the manifest while the FBX kept
        it on the node -- exact matching bound nothing and every referenced
        object lost its lighting. The stripped leaf is unambiguous here, so
        it must bind."""
        exr = self._exr()
        glb = self._glb(
            self._scene(
                self._manifest([{"name": "room", "map": os.path.basename(exr)}]),
                objects=("VDATS_DA:room",),
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["object"], "room")

    def test_manifest_namespace_binds_a_stripped_node(self):
        """The reverse direction: a namespaced manifest against an exporter
        that flattened namespaces out of the node names."""
        exr = self._exr()
        glb = self._glb(
            self._scene(
                self._manifest(
                    [{"name": "VDATS_DA:room", "map": os.path.basename(exr)}]
                ),
                objects=("room",),
            )
        )
        self.assertEqual(len(MeshConvert.apply_glb_lightmaps(glb)), 1)

    def test_ambiguous_leaf_is_not_guessed(self):
        """Two nodes stripping to the same leaf: binding either would put one
        object's lighting on the other -- skip loudly instead."""
        exr = self._exr()
        glb = self._glb(
            self._scene(
                self._manifest([{"name": "room", "map": os.path.basename(exr)}]),
                objects=("NS_A:room", "NS_B:room"),
            )
        )
        self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])

    def test_exact_match_outranks_leaf_fallback(self):
        """When the manifest name exists verbatim, a same-leaf sibling under a
        namespace must not attract the binding."""
        exr = self._exr()
        glb = self._glb(
            self._scene(
                self._manifest(
                    [
                        {
                            "name": "room",
                            "map": os.path.basename(exr),
                            "scaleOffset": [0.5, 0.5, 0.0, 0.0],
                        }
                    ]
                ),
                objects=("room", "NS_A:room"),
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(len(records), 1)
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        # Only the exact-name node's primitive gained the per-instance material
        # clone (the rect forces one); the namespaced sibling keeps material 0.
        exact_mat = gltf["meshes"][gltf["nodes"][0]["mesh"]]["primitives"][0][
            "material"
        ]
        sibling_mat = gltf["meshes"][gltf["nodes"][1]["mesh"]]["primitives"][0][
            "material"
        ]
        self.assertGreater(exact_mat, 1, "rect binding must clone the material")
        self.assertEqual(sibling_mat, 0)

    def test_lightmap_texture_samples_clamp_to_edge(self):
        """Atlas rects legally extend past [0,1]; REPEAT would wrap a tap past
        an atlas edge onto the opposite side's unrelated texels. The lightmap's
        texture must clamp -- without disturbing sampler 0 (the file default)."""
        exr = self._exr()
        glb = self._glb(
            self._scene(
                self._manifest([{"name": "room", "map": os.path.basename(exr)}])
            )
        )
        MeshConvert.apply_glb_lightmaps(glb)
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        oc = gltf["materials"][0]["occlusionTexture"]
        sampler = gltf["samplers"][gltf["textures"][oc["index"]]["sampler"]]
        self.assertEqual(sampler, {"wrapS": 33071, "wrapT": 33071})
        self.assertEqual(gltf["samplers"][0], {"wrapS": 10497, "wrapT": 10497})

    def test_missing_texcoord1_is_skipped_loudly_not_bound(self):
        """No second UV set = the FBX shipped without lightmap UVs; binding the
        map anyway would sample it through the texture UVs -- silently wrong."""
        exr = self._exr()
        glb = self._glb(
            self._scene(
                self._manifest([{"name": "room", "map": os.path.basename(exr)}]),
                texcoord1=False,
            )
        )
        self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])
        with MeshConvert.open_glb(glb) as edit:
            self.assertNotIn("occlusionTexture", edit.gltf["materials"][0])
            self.assertNotIn("lightmap_web", edit.gltf.get("extras", {}))

    def test_unresolvable_map_is_skipped(self):
        glb = self._glb(
            self._scene(self._manifest([{"name": "room", "map": "gone.exr"}]))
        )
        self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])

    def test_newer_manifest_version_is_refused(self):
        """Misreading a future schema would bind wrong data; refuse instead."""
        exr = self._exr()
        glb = self._glb(
            self._scene(
                self._manifest(
                    [{"name": "room", "map": os.path.basename(exr)}], version=99
                )
            )
        )
        self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])

    def test_shared_material_with_different_maps_first_claim_wins(self):
        """Per-object maps on one material: the second has nowhere to go. Atlas
        packing prevents this upstream; here it must warn, not mis-bind."""
        # Written for their side effect: the manifest below names them by file.
        _a, _b = self._exr("a.exr"), self._exr("b.exr", value=0.25)
        glb = self._glb(
            self._scene(
                self._manifest(
                    [
                        {"name": "room", "map": "a.exr"},
                        {"name": "prop", "map": "b.exr"},
                    ]
                ),
                objects=("room", "prop"),
                material=0,  # both meshes share materials[0]
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual([r["object"] for r in records], ["room"])
        with MeshConvert.open_glb(glb) as edit:
            self.assertEqual([i["name"] for i in edit.gltf["images"]], ["a.png"])

    def test_shared_atlas_binds_every_object_once_per_material(self):
        """The normal atlas case: two objects, one material, ONE map -- one
        embed, two records."""
        exr = self._exr("atlas.exr")
        glb = self._glb(
            self._scene(
                self._manifest(
                    [
                        {"name": "room", "map": os.path.basename(exr)},
                        {"name": "prop", "map": os.path.basename(exr)},
                    ]
                ),
                objects=("room", "prop"),
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(len(records), 2)
        with MeshConvert.open_glb(glb) as edit:
            self.assertEqual(len(edit.gltf["images"]), 1, "atlas embeds once")

    def test_per_instance_rects_ride_khr_texture_transform(self):
        """Instances (one shared glTF mesh, many nodes -- what FBX2glTF emits,
        probe-measured) each end with their OWN mesh entry over the same
        accessors and a material CLONE carrying the rect as a glTF-standard
        KHR_texture_transform -- so any compliant viewer renders each copy's
        patch of the atlas with no custom code, and no geometry is duplicated.
        """
        from pythontk import ImgUtils

        self._exr("atlas.exr")
        rect_a, rect_b = [0.5, 1.0, 0.0, 0.0], [0.5, 1.0, 0.5, 0.0]
        gltf = self._scene(
            self._manifest(
                [
                    {"name": "wall_a", "map": "atlas.exr", "scaleOffset": rect_a},
                    {"name": "wall_b", "map": "atlas.exr", "scaleOffset": rect_b},
                ]
            ),
            objects=("wall_a", "wall_b"),
        )
        gltf["meshes"] = gltf["meshes"][:1]  # collapse to ONE shared mesh
        for node in gltf["nodes"]:
            if "mesh" in node:
                node["mesh"] = 0
        glb = self._glb(gltf)

        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(
            {r["object"]: tuple(r["scaleOffset"]) for r in records},
            {"wall_a": tuple(rect_a), "wall_b": tuple(rect_b)},
        )
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        self.assertIn("KHR_texture_transform", gltf.get("extensionsUsed", []))
        walls = {
            n["name"]: n for n in gltf["nodes"] if n.get("name", "").startswith("wall")
        }
        self.assertEqual(
            len({n["mesh"] for n in walls.values()}), 2, "own mesh entry each"
        )
        self.assertEqual(len(gltf["meshes"]), 2, "one clone; last user keeps original")
        prims = {
            name: gltf["meshes"][n["mesh"]]["primitives"][0]
            for name, n in walls.items()
        }
        # The clones reference the SAME accessors -- zero geometry duplication.
        self.assertEqual(prims["wall_a"]["attributes"], prims["wall_b"]["attributes"])
        self.assertEqual(len(gltf["images"]), 1, "one shared atlas embed")
        for name, rect in (("wall_a", rect_a), ("wall_b", rect_b)):
            mat = gltf["materials"][prims[name]["material"]]
            tex = mat["occlusionTexture"]
            self.assertEqual(tex["texCoord"], 1)
            flipped = ImgUtils.flip_rect_v(rect)
            khr = tex["extensions"]["KHR_texture_transform"]
            self.assertEqual(khr["scale"], [flipped[0], flipped[1]])
            self.assertEqual(khr["offset"], [flipped[2], flipped[3]])
            # ...and the viewer manifest keys the clone so the rebind still works.
            self.assertIn(mat["name"], gltf["extras"]["lightmap_web"]["materials"])

    def test_displaced_authored_map_warns_once_per_material(self):
        """A shared material warns ONCE, not once per instance clone.

        The carrier slot may already hold an authored map (a real AO), and
        displacing it is worth saying -- but the warning is per SOURCE material,
        so a room whose pieces share one material cannot bury every other line in
        the log under N copies of the same sentence (46 of them, measured on the
        OFFICE_ENV module).
        """
        self._exr("atlas.exr")
        gltf = self._scene(
            self._manifest(
                [
                    {
                        "name": f"wall_{i}",
                        "map": "atlas.exr",
                        "scaleOffset": [0.5, 1.0, 0.5 * (i % 2), 0.0],
                    }
                    for i in range(3)
                ]
            ),
            objects=("wall_0", "wall_1", "wall_2"),
        )
        gltf["meshes"] = gltf["meshes"][:1]  # one shared mesh + one shared material
        for node in gltf["nodes"]:
            if "mesh" in node:
                node["mesh"] = 0
        # An AUTHORED occlusion map already sits on the carrier slot.
        gltf["materials"][gltf["meshes"][0]["primitives"][0]["material"]][
            "occlusionTexture"
        ] = {"index": 0}
        glb = self._glb(gltf)

        with self.assertLogs(
            "pythontk.file_utils.mesh_convert._mesh_convert", level="WARNING"
        ) as caught:
            MeshConvert.apply_glb_lightmaps(glb)
        displaced = [m for m in caught.output if "is dropped on every" in m]
        self.assertEqual(len(displaced), 1, f"one line expected, got {caught.output}")

    def test_lightmap_displaces_a_packed_orm_occlusion_silently(self):
        """A slot holding the material's own metallicRoughnessTexture is the
        packed-ORM occlusion binding (set_glb_metallic_roughness, or FBX2glTF's
        converted packing), not an authored map -- its R channel is AO the bake
        already contains. Displacing it must not fire the authored-map warning:
        with the ORM writer now binding that slot on nearly every repaired
        material, the warning would fire on nearly every lightmapped material
        and bury the real signal it exists to carry."""
        exr = self._exr()
        gltf = self._scene(
            self._manifest([{"name": "room", "map": os.path.basename(exr)}])
        )
        gltf["materials"][0].update(
            {
                "pbrMetallicRoughness": {"metallicRoughnessTexture": {"index": 0}},
                "occlusionTexture": {"index": 0},
            }
        )
        gltf["textures"] = [{"source": 0}]
        gltf["images"] = [{"name": "orm.png"}]
        glb = self._glb(gltf)
        with self.assertNoLogs(
            "pythontk.file_utils.mesh_convert._mesh_convert", level="WARNING"
        ):
            records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(len(records), 1)
        with MeshConvert.open_glb(glb) as edit:
            oc = edit.gltf["materials"][0]["occlusionTexture"]
        self.assertEqual(oc["texCoord"], 1, "the lightmap must take the slot")

    def test_replace_authored_false_keeps_the_authored_carrier(self):
        """The gate: an authored AO map survives and the lightmap stands down.

        Skipping has to happen BEFORE the encode-and-embed, or a material whose
        every instance keeps its authored map would still carry the lightmap
        PNG as an orphan texture nothing samples. And it must warn ONCE for the
        shared material, not once per object/primitive wearing it -- the skip
        path never sets ``claimed``, so without its own once-guard a 46-piece
        room would repeat the line 46 times.
        """
        exr = self._exr()
        basename = os.path.basename(exr)
        gltf = self._scene(
            self._manifest(
                [{"name": "room", "map": basename}, {"name": "prop", "map": basename}]
            ),
            objects=("room", "prop"),  # two objects, one shared material
        )
        gltf["materials"][0]["occlusionTexture"] = {"index": 0}  # authored, no ORM
        gltf["textures"] = [{"source": 0}]
        gltf["images"] = [{"name": "hand_authored_ao.png"}]
        glb = self._glb(gltf)
        with self.assertLogs(
            "pythontk.file_utils.mesh_convert._mesh_convert", level="WARNING"
        ) as caught:
            records = MeshConvert.apply_glb_lightmaps(glb, replace_authored=False)
        self.assertEqual(records, [])
        kept = [m for m in caught.output if "replace_authored=False" in m]
        self.assertEqual(len(kept), 1, f"one line expected, got {caught.output}")
        with MeshConvert.open_glb(glb) as edit:
            gltf_after = edit.gltf
        self.assertEqual(gltf_after["materials"][0]["occlusionTexture"], {"index": 0})
        self.assertEqual(
            len(gltf_after.get("images") or []),
            1,
            "the skipped lightmap must not be embedded as an orphan",
        )
        self.assertNotIn("lightmap_web", gltf_after.get("extras", {}))

    def test_replace_authored_false_skips_instance_clones_without_orphans(self):
        """Same gate on the rect path: no clones, and no orphan atlas embed --
        the authored check has to run before ``_scalar()``."""
        self._exr("atlas.exr")
        gltf = self._scene(
            self._manifest(
                [
                    {
                        "name": f"wall_{i}",
                        "map": "atlas.exr",
                        "scaleOffset": [0.5, 1.0, 0.5 * (i % 2), 0.0],
                    }
                    for i in range(2)
                ]
            ),
            objects=("wall_0", "wall_1"),
        )
        gltf["materials"][0]["occlusionTexture"] = {"index": 0}  # authored
        gltf["textures"] = [{"source": 0}]
        gltf["images"] = [{"name": "hand_authored_ao.png"}]
        glb = self._glb(gltf)
        records = MeshConvert.apply_glb_lightmaps(glb, replace_authored=False)
        self.assertEqual(records, [])
        with MeshConvert.open_glb(glb) as edit:
            gltf_after = edit.gltf
        self.assertEqual(len(gltf_after["materials"]), 2, "no ~lm clones")
        self.assertEqual(len(gltf_after.get("images") or []), 1, "no orphan embed")
        self.assertNotIn("extensionsUsed", gltf_after)

    def test_identity_rects_change_nothing_structural(self):
        """An explicit identity scaleOffset is the plain path: shared material
        binding, no clones, no extension declared."""
        self._exr("atlas.exr")
        glb = self._glb(
            self._scene(
                self._manifest(
                    [
                        {
                            "name": "room",
                            "map": "atlas.exr",
                            "scaleOffset": [1.0, 1.0, 0.0, 0.0],
                        }
                    ]
                )
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(records[0]["scaleOffset"], [1.0, 1.0, 0.0, 0.0])
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        self.assertNotIn("extensionsUsed", gltf)
        self.assertEqual(len(gltf["materials"]), 2, "no clones")
        self.assertEqual(len(gltf["meshes"]), 1)


class _FakeKtx2Encoder:
    """Records encode calls and writes a KTX2-magic marker file."""

    MAGIC = b"\xabKTX 20\xbb\r\n\x1a\n"

    def __init__(self):
        self.calls = []

    def encode(
        self, source, output, codec="UASTC", srgb=True, mipmaps=True, quality=None
    ):
        self.calls.append(
            {
                "codec": codec,
                "srgb": srgb,
                "quality": quality,
                "size": getattr(source, "size", None),
            }
        )
        with open(output, "wb") as fh:
            fh.write(self.MAGIC + codec.encode("ascii"))
        return output


class TestOptimizeGlbKtx2(unittest.TestCase):
    """KTX2 mode: per-slot codecs, basisu bindings, POT snap, lightmap carve-out."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_ktx2_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.fake = _FakeKtx2Encoder()
        ImgUtils.register_ktx2_encoder(self.fake)
        self.addCleanup(ImgUtils.register_ktx2_encoder, None)

    # ------------------------------------------------------------------ helpers
    def _png_uri(self, size=(128, 128), color=(200, 60, 40)):
        import base64 as b64

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", size, color).save(buffer, format="PNG")
        return "data:image/png;base64," + b64.b64encode(buffer.getvalue()).decode(
            "ascii"
        )

    def _glb(self, gltf, name="scene.glb"):
        json_bytes = json.dumps(gltf).encode("utf-8")
        json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
        blob = (
            struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_bytes))
            + struct.pack("<I4s", len(json_bytes), b"JSON")
            + json_bytes
        )
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(blob)
        return path

    def _payload(self, edit, image):
        view = edit.gltf["bufferViews"][image["bufferView"]]
        return bytes(
            edit.bin_data[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
        )

    # -------------------------------------------------------------------- tests
    def test_per_slot_codecs_and_fallback_binding(self):
        """Color -> ETC1S/sRGB, normal -> UASTC/linear; each texture keeps a
        core-readable fallback ``source`` (JPEG for color, PNG for normal), so
        KHR_texture_basisu stays out of extensionsRequired and the GLB is no
        terminal delivery artifact — a stock importer reads the fallbacks."""
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [
                    {"name": "wall_color", "uri": self._png_uri(color=(200, 60, 40))},
                    {
                        "name": "wall_normal",
                        "uri": self._png_uri(color=(127, 127, 255)),
                    },
                ],
                "textures": [{"source": 0}, {"source": 1}],
                "materials": [
                    {
                        "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                        "normalTexture": {"index": 1},
                    }
                ],
            }
        )
        summary = MeshConvert.optimize_glb_textures(
            path, max_size=64, image_format="KTX2", quality=60
        )
        self.assertEqual(summary["images"], 2)

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        by_codec = {c["codec"]: c for c in self.fake.calls}
        self.assertTrue(by_codec["ETC1S"]["srgb"])
        self.assertEqual(by_codec["ETC1S"]["quality"], 60)
        self.assertFalse(by_codec["UASTC"]["srgb"])
        self.assertIsNone(by_codec["UASTC"]["quality"], "quality is the ETC1S dial")

        self.assertEqual(len(gltf["images"]), 4, "one appended fallback per encode")
        fallback_magic = {"image/jpeg": b"\xff\xd8", "image/png": b"\x89PN"}
        for index, codec, fb_mime in (
            (0, b"ETC1S", "image/jpeg"),
            (1, b"UASTC", "image/png"),
        ):
            image = gltf["images"][index]
            self.assertEqual(image["mimeType"], "image/ktx2")
            payload = self._payload(edit, image)
            self.assertEqual(payload[:12], _FakeKtx2Encoder.MAGIC)
            self.assertEqual(payload[12:], codec)
            texture = gltf["textures"][index]
            self.assertEqual(
                texture["extensions"]["KHR_texture_basisu"]["source"], index
            )
            fb_index = texture["source"]
            self.assertGreaterEqual(fb_index, 2, "fallback is an appended image")
            fallback = gltf["images"][fb_index]
            self.assertEqual(fallback["mimeType"], fb_mime)
            self.assertEqual(
                self._payload(edit, fallback)[: len(fallback_magic[fb_mime])],
                fallback_magic[fb_mime],
            )
            self.assertEqual(fallback["name"], f"{image['name']}_fallback")
        self.assertIn("KHR_texture_basisu", gltf["extensionsUsed"])
        self.assertNotIn(
            "KHR_texture_basisu",
            gltf.get("extensionsRequired", []),
            "every binding has a core-readable fallback",
        )

    def test_pure_delivery_mode_requires_basisu(self):
        """ktx2_fallback=False is the old contract: no fallback source, and the
        extension hard-requires a basisu-capable viewer (extensionsRequired)."""
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "c", "uri": self._png_uri()}],
                "textures": [{"source": 0}],
                "materials": [
                    {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                ],
            }
        )
        MeshConvert.optimize_glb_textures(
            path, max_size=64, image_format="KTX2", ktx2_fallback=False
        )
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 1, "no fallback appended")
        texture = gltf["textures"][0]
        self.assertEqual(texture["extensions"]["KHR_texture_basisu"]["source"], 0)
        self.assertNotIn("source", texture, "no fallback is kept")
        self.assertIn("KHR_texture_basisu", gltf["extensionsRequired"])

    def test_fallback_alpha_color_stays_png(self):
        """A color image with alpha cannot fall back to JPEG — the twin must
        keep the channel, so ETC1S + alpha lands on PNG."""
        import base64 as b64

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGBA", (32, 32), (200, 60, 40, 128)).save(buffer, format="PNG")
        uri = "data:image/png;base64," + b64.b64encode(buffer.getvalue()).decode(
            "ascii"
        )
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "decal", "uri": uri}],
                "textures": [{"source": 0}],
                "materials": [
                    {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path, max_size=0, image_format="KTX2")
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(self.fake.calls[0]["codec"], "ETC1S")
        fallback = gltf["images"][gltf["textures"][0]["source"]]
        self.assertEqual(fallback["mimeType"], "image/png")

    def test_shared_image_takes_the_stricter_semantic(self):
        """Bytes sampled as both color and data must encode once, as UASTC."""
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "shared", "uri": self._png_uri()}],
                "textures": [{"source": 0}],
                "materials": [
                    {
                        "pbrMetallicRoughness": {
                            "baseColorTexture": {"index": 0},
                            "metallicRoughnessTexture": {"index": 0},
                        }
                    }
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path, max_size=64, image_format="KTX2")
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(self.fake.calls[0]["codec"], "UASTC")
        self.assertFalse(self.fake.calls[0]["srgb"])

    def test_dimensions_snap_down_to_pot(self):
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "odd", "uri": self._png_uri(size=(96, 48))}],
                "textures": [{"source": 0}],
                "materials": [
                    {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path, max_size=0, image_format="KTX2")
        self.assertEqual(self.fake.calls[0]["size"], (64, 32))

    def test_lightmap_stays_lossless_webp(self):
        """The bake keeps its fidelity path even in KTX2 mode."""
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "extras": {
                    "lightmap_web": {"materials": {"M": {"map": "room_Lightmap.png"}}}
                },
                "images": [
                    {"name": "source.png", "uri": self._png_uri(color=(10, 200, 30))},
                    {"name": "room_Lightmap.png", "uri": self._png_uri()},
                ],
                "textures": [{"source": 0}, {"source": 1}],
                "materials": [
                    {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path, max_size=64, image_format="KTX2")

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        self.assertEqual(gltf["images"][0]["mimeType"], "image/ktx2")
        lightmap = gltf["images"][1]
        self.assertEqual(lightmap["mimeType"], "image/webp")
        self.assertEqual(self._payload(edit, lightmap)[:4], b"RIFF")
        # The carrier texture keeps the standard WebP binding WITH fallback.
        self.assertEqual(
            gltf["textures"][1]["extensions"]["EXT_texture_webp"]["source"], 1
        )
        self.assertEqual(gltf["textures"][1]["source"], 1)
        self.assertIn("EXT_texture_webp", gltf["extensionsUsed"])
        self.assertNotIn("EXT_texture_webp", gltf.get("extensionsRequired", []))
        # Only the encoder saw only the source image — never the lightmap.
        self.assertEqual(len(self.fake.calls), 1)

    def test_webp_then_ktx2_rerun_cleans_the_webp_declaration(self):
        """Switching a deliverable from WebP to KTX2 must not leave a dangling
        EXT_texture_webp binding or declaration behind — a declared extension
        with no user is a validator warning shipped for nothing."""
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "c", "uri": self._png_uri()}],
                "textures": [{"source": 0}],
                "materials": [
                    {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path, max_size=64)  # WebP first
        gltf = MeshConvert._read_glb(path).gltf
        self.assertIn("EXT_texture_webp", gltf["extensionsUsed"])

        MeshConvert.optimize_glb_textures(path, max_size=64, image_format="KTX2")
        gltf = MeshConvert._read_glb(path).gltf
        texture = gltf["textures"][0]
        self.assertNotIn("EXT_texture_webp", texture["extensions"])
        self.assertEqual(texture["extensions"]["KHR_texture_basisu"]["source"], 0)
        self.assertNotIn("EXT_texture_webp", gltf["extensionsUsed"])
        self.assertIn("KHR_texture_basisu", gltf["extensionsUsed"])
        self.assertNotIn("KHR_texture_basisu", gltf.get("extensionsRequired", []))
        self.assertEqual(gltf["images"][0]["mimeType"], "image/ktx2")
        # The WebP bytes the first pass wrote became the decode source for the
        # KTX2 rerun; the plain source now points at the appended core fallback.
        self.assertEqual(gltf["images"][texture["source"]]["mimeType"], "image/jpeg")

    def test_unsampled_image_never_gets_a_non_core_mime_type(self):
        """An image no texture samples cannot be rebound through
        KHR_texture_basisu, so re-encoding it would strand ``image/ktx2`` in a
        file that declares nothing to enable it — and glTF 2.0 core permits
        only image/jpeg and image/png. Both ways a texture fails to resolve:
        a ``source`` out of range, and no textures at all.
        """
        for label, extra in (
            ("out-of-range source", {"textures": [{"source": 7}]}),
            ("no textures", {}),
        ):
            with self.subTest(label):
                path = self._glb(
                    {
                        "asset": {"version": "2.0"},
                        "images": [{"name": "orphan", "uri": self._png_uri()}],
                        **extra,
                    },
                    name=f"{label.replace(' ', '_')}.glb",
                )
                with open(path, "rb") as f:
                    before = f.read()
                MeshConvert.optimize_glb_textures(
                    path, max_size=64, image_format="KTX2"
                )
                gltf = MeshConvert._read_glb(path).gltf
                self.assertNotEqual(
                    gltf["images"][0].get("mimeType"),
                    "image/ktx2",
                    "non-core mime with no extension declaring it",
                )
                self.assertNotIn("KHR_texture_basisu", gltf.get("extensionsUsed", []))
                with open(path, "rb") as f:
                    self.assertEqual(f.read(), before, "nothing to do, nothing written")
                self.assertEqual(self.fake.calls, [], "no encode worth paying for")

    def test_only_a_sampled_image_takes_the_ktx2_path(self):
        """A bound image still re-encodes; its unsampled sibling keeps its
        core-mime bytes instead of riding the declaration the bound one earns.
        """
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [
                    {"name": "bound", "uri": self._png_uri()},
                    {"name": "orphan", "uri": self._png_uri(color=(9, 9, 9))},
                ],
                "textures": [{"source": 0}],
                "materials": [
                    {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                ],
            }
        )
        summary = MeshConvert.optimize_glb_textures(
            path, max_size=64, image_format="KTX2"
        )
        self.assertEqual(summary["images"], 1)
        self.assertEqual(len(self.fake.calls), 1)

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        self.assertEqual(gltf["images"][0]["mimeType"], "image/ktx2")
        self.assertEqual(
            self._payload(edit, gltf["images"][0])[:12], _FakeKtx2Encoder.MAGIC
        )
        self.assertIn("KHR_texture_basisu", gltf["extensionsUsed"])
        self.assertNotIn("KHR_texture_basisu", gltf.get("extensionsRequired", []))
        orphan = gltf["images"][1]
        self.assertNotEqual(orphan.get("mimeType"), "image/ktx2")
        self.assertTrue(orphan["uri"].startswith("data:image/png"))

    def test_missing_encoder_fails_before_touching_the_file(self):
        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "c", "uri": self._png_uri()}],
                "textures": [{"source": 0}],
            }
        )
        with open(path, "rb") as f:
            before = f.read()
        ImgUtils.register_ktx2_encoder(None)
        with (
            patch.object(Ktx2Encoder, "available", return_value=False),
            patch.object(
                Ktx2Encoder,
                "resolve_toktx",
                side_effect=FileNotFoundError("no toktx"),
            ),
        ):
            with self.assertRaises(FileNotFoundError):
                MeshConvert.optimize_glb_textures(path, image_format="KTX2")
        with open(path, "rb") as f:
            after = f.read()
        self.assertEqual(after, before, "no partial rewrite")


if __name__ == "__main__":
    unittest.main()
