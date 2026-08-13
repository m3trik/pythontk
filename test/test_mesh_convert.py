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
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk import MeshConvert


class TestResolveBinary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_returns_path_when_on_system_path(self):
        with patch("shutil.which", return_value="/usr/bin/FBX2glTF"):
            self.assertEqual(MeshConvert.resolve_binary(), "/usr/bin/FBX2glTF")

    def test_returns_managed_path_when_in_catalog(self):
        managed = os.path.join(self.tmp, "FBX2glTF.exe")
        with patch("shutil.which", return_value=None), patch(
            "pythontk.core_utils.app_installer.AppInstaller.get_path",
            return_value=managed,
        ):
            self.assertEqual(MeshConvert.resolve_binary(), managed)

    def test_raises_when_missing_and_required(self):
        with patch("shutil.which", return_value=None), patch(
            "pythontk.core_utils.app_installer.AppInstaller.get_path",
            return_value=None,
        ):
            with self.assertRaises(FileNotFoundError):
                MeshConvert.resolve_binary(required=True, auto_install=False)

    def test_returns_none_when_missing_and_not_required(self):
        with patch("shutil.which", return_value=None), patch(
            "pythontk.core_utils.app_installer.AppInstaller.get_path",
            return_value=None,
        ):
            self.assertIsNone(
                MeshConvert.resolve_binary(required=False, auto_install=False)
            )

    def test_no_tty_with_prompt_refuses_install(self):
        """prompt=True without a TTY should NOT silently install."""
        with patch("shutil.which", return_value=None), patch(
            "pythontk.core_utils.app_installer.AppInstaller.get_path",
            return_value=None,
        ), patch(
            "pythontk.core_utils.app_installer.AppInstaller.ensure"
        ) as ensure, patch("sys.stdin") as stdin:
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
        with patch("shutil.which", return_value=None), patch(
            "pythontk.core_utils.app_installer.AppInstaller.get_path",
            return_value=None,
        ), patch(
            "pythontk.core_utils.app_installer.AppInstaller.ensure",
            return_value=installed,
        ) as ensure, patch("sys.stdin") as stdin:
            stdin.isatty.return_value = False
            result = MeshConvert.resolve_binary(auto_install=True, prompt=False)
            self.assertEqual(result, installed)
            ensure.assert_called_once()

    def test_prompt_decline_raises_when_required(self):
        with patch("shutil.which", return_value=None), patch(
            "pythontk.core_utils.app_installer.AppInstaller.get_path",
            return_value=None,
        ), patch("sys.stdin") as stdin:
            stdin.isatty.return_value = True
            stdin.readline.return_value = "n\n"
            with self.assertRaises(FileNotFoundError):
                MeshConvert.resolve_binary(auto_install=True, prompt=True, required=True)

    def test_prompt_accept_triggers_install(self):
        installed = os.path.join(self.tmp, "FBX2glTF.exe")
        with patch("shutil.which", return_value=None), patch(
            "pythontk.core_utils.app_installer.AppInstaller.get_path",
            return_value=None,
        ), patch(
            "pythontk.core_utils.app_installer.AppInstaller.ensure",
            return_value=installed,
        ) as ensure, patch("sys.stdin") as stdin:
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
        with patch.object(
            MeshConvert, "resolve_binary", return_value=self.fake_bin
        ), patch("subprocess.run", side_effect=self._run_simulator(captured)):
            result = MeshConvert.fbx_to_glb(self.src, auto_install=False)
            self.assertEqual(result, expected_dst)
            self.assertTrue(os.path.isfile(expected_dst))

    def test_dst_glb_extension_appended_if_missing(self):
        captured = {}
        dst_no_ext = os.path.join(self.tmp, "out")
        with patch.object(
            MeshConvert, "resolve_binary", return_value=self.fake_bin
        ), patch("subprocess.run", side_effect=self._run_simulator(captured)):
            result = MeshConvert.fbx_to_glb(self.src, dst_no_ext, auto_install=False)
            self.assertEqual(result, dst_no_ext + ".glb")

    def test_command_uses_input_output_binary_flags(self):
        dst = os.path.join(self.tmp, "out.glb")
        captured = {}
        with patch.object(
            MeshConvert, "resolve_binary", return_value=self.fake_bin
        ), patch("subprocess.run", side_effect=self._run_simulator(captured)):
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
        with patch.object(
            MeshConvert, "resolve_binary", return_value=self.fake_bin
        ), patch("subprocess.run", side_effect=self._run_simulator(captured)):
            MeshConvert.fbx_to_glb(
                self.src, dst, auto_install=False, extra_args=["--draco"]
            )
        self.assertIn("--draco", captured["cmd"])

    def test_subprocess_failure_raises(self):
        dst = os.path.join(self.tmp, "out.glb")
        with patch.object(
            MeshConvert, "resolve_binary", return_value=self.fake_bin
        ), patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                ["x"], 1, stdout="", stderr="boom"
            ),
        ):
            with self.assertRaises(RuntimeError) as cm:
                MeshConvert.fbx_to_glb(self.src, dst, auto_install=False)
            self.assertIn("boom", str(cm.exception))

    def test_subprocess_zero_exit_but_no_output_raises(self):
        dst = os.path.join(self.tmp, "out.glb")
        with patch.object(
            MeshConvert, "resolve_binary", return_value=self.fake_bin
        ), patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(["x"], 0, stdout="", stderr=""),
        ):
            with self.assertRaises(RuntimeError) as cm:
                MeshConvert.fbx_to_glb(self.src, dst, auto_install=False)
            self.assertIn("not created", str(cm.exception))

    def test_timeout_raises_runtime_error(self):
        dst = os.path.join(self.tmp, "out.glb")
        with patch.object(
            MeshConvert, "resolve_binary", return_value=self.fake_bin
        ), patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["x"], timeout=1),
        ):
            with self.assertRaises(RuntimeError) as cm:
                MeshConvert.fbx_to_glb(self.src, dst, auto_install=False, timeout=1)
            self.assertIn("timed out", str(cm.exception))

    def test_timeout_kwarg_forwarded_to_subprocess(self):
        dst = os.path.join(self.tmp, "out.glb")
        captured = {}
        with patch.object(
            MeshConvert, "resolve_binary", return_value=self.fake_bin
        ), patch("subprocess.run", side_effect=self._run_simulator(captured)):
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
            im = Image.new("RGBA", size, (200, 100, 50, alpha if alpha is not None else 255))
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
            buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob)})
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

        header = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_bytes) + 8 + len(bin_data))
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
            materials=[{
                "name": "Body_base",
                "alphaMode": "BLEND",
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
            }],
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
            materials=[{
                "name": "Glass",
                "alphaMode": "BLEND",
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
            }],
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
            materials=[{
                "name": "Plain",
                "alphaMode": "OPAQUE",
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
            }],
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
            materials=[{
                "name": "RGBish",
                "alphaMode": "BLEND",  # weird but possible
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
            }],
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
            materials=[{
                "name": "Leaf",
                "alphaMode": "MASK",
                "alphaCutoff": 0.5,
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
            }],
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
            materials=[{
                "name": "TintedGlass",
                "alphaMode": "BLEND",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 0.4],
                    "baseColorTexture": {"index": 0},
                },
            }],
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
            materials=[{
                "name": "Body",
                "alphaMode": "BLEND",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "baseColorTexture": {"index": 0},
                },
            }],
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
            materials=[{"name": "B", "alphaMode": "BLEND",
                        "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        mask_path = self._write_glb(
            "mask_reason.glb",
            materials=[{"name": "M", "alphaMode": "MASK", "alphaCutoff": 0.5,
                        "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
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
                {"name": "A", "alphaMode": "BLEND",
                 "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}},
                {"name": "B", "alphaMode": "BLEND",
                 "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}},
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
            f.write(TestCheckGlbMaterials._build_glb(
                materials=materials,
                images=images,
                textures=textures,
                image_blobs=image_blobs,
            ))
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
            materials=[{
                "name": "TREELINE_D",
                "alphaMode": "BLEND",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                    "baseColorTexture": {"index": 0},
                },
            }],
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
            materials=[{
                "name": "FoliageMask",
                "alphaMode": "MASK",
                "alphaCutoff": 0.5,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                    "baseColorTexture": {"index": 0},
                },
            }],
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
            materials=[{
                "name": "Plain",
                "alphaMode": "OPAQUE",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                    "baseColorTexture": {"index": 0},
                },
            }],
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
            materials=[{
                "name": "Glass",
                "alphaMode": "BLEND",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 0.5],
                    "baseColorTexture": {"index": 0},
                },
            }],
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
            materials=[{
                "name": "UniformAlpha",
                "alphaMode": "BLEND",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                    "baseColorTexture": {"index": 0},
                },
            }],
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
            materials=[{
                "name": "NoTex",
                "alphaMode": "BLEND",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                },
            }],
            images=[],
            textures=[],
            image_blobs=[],
        )
        self.assertEqual(MeshConvert.fix_glb_phantom_opaque_alpha(path), [])

    def test_returns_empty_when_nothing_to_fix(self):
        """No changes → no rewrite, empty list returned, file untouched."""
        path = self._write_glb(
            "clean.glb",
            materials=[{
                "name": "Plain",
                "alphaMode": "OPAQUE",
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
            }],
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
            {"Body": {"texture": os.path.join(self.tmp, "missing.png"),
                      "color": [1.0, 0.0, 0.0]}},
        )
        self.assertEqual(len(records), 1)
        self.assertIsNone(
            records[0]["texture"],
            "no baseColorTexture was written — the record must not claim one",
        )


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
        self.assertEqual(bytes(edit.bin_data), geometry)
        self.assertTrue(edit.gltf["images"][0]["uri"].startswith("data:image/png"))

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
        """One file on disk, one base64 copy — the embed cache spans the session."""
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

        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 1, "the payload was embedded twice")
        self.assertFalse(
            any(str(i.get("uri", "")).startswith("data:") for i in gltf["images"]),
            "a base64 copy was added alongside the bufferView image",
        )
        index = gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
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
        webp = bytes(blob[img_view["byteOffset"] : img_view["byteOffset"] + img_view["byteLength"]])
        self.assertEqual(webp[:4], b"RIFF", "payload must be a real WebP container")
        resized = Image.open(io.BytesIO(webp))
        self.assertEqual(max(resized.size), 64)

        geo_view = gltf["bufferViews"][1]
        survived = bytes(
            blob[geo_view["byteOffset"] : geo_view["byteOffset"] + geo_view["byteLength"]]
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
            raw = bytes(blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]])
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
            path = self._write_glb(
                json.loads(json.dumps(gltf)), bin_chunk=bin_chunk
            )
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

        gltf = MeshConvert._read_glb(path).gltf
        pbr = gltf["materials"][0]["pbrMetallicRoughness"]
        tex = pbr["metallicRoughnessTexture"]["index"]
        img = gltf["images"][gltf["textures"][tex]["source"]]
        import base64 as b64
        import io as iolib

        raw = b64.b64decode(img["uri"].split(",", 1)[1])
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
        """The embedded envelope is the caller's plus the resolution keys.

        Not equality: only the apply pass can know which glTF image each
        authoring path became, so it embeds a COPY carrying ``textures`` and
        ``validate``. Equality here would also mean the caller's dict had been
        mutated behind their back -- the bridges keep that object and write a
        ``.scene.json`` inspection copy from it.
        """
        embedded = MeshConvert.read_scene_sidecar(glb)
        self.assertEqual(set(embedded) - set(envelope), {"textures", "validate"})
        self.assertEqual(
            {k: v for k, v in embedded.items() if k in envelope}, envelope
        )
        self.assertNotIn("textures", envelope, "caller's envelope was mutated")
        return embedded

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
        self.assertIn(
            f"extras.{MeshConvert.LIGHTMAP_WEB_KEY}", handoff["reads"]
        )

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
            self.assertIn(path, refs, "every authoring path must resolve")
        self.assertEqual(
            refs[rough]["image"],
            refs[metal]["image"],
            "the ORM trio collapses into one image",
        )
        self.assertNotEqual(refs[emit]["image"], refs[rough]["image"])
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
            refs[shared]["image"],
            emissive_image,
            "the whole-image embed wins over a channel of a pack",
        )

    def test_apply_survives_a_malformed_section_when_sizing_validate(self):
        """A null section is skipped by the dispatch, so sizing it for
        ``validate`` must not be the thing that raises: the container catch
        handles OSError/ValueError only, so a TypeError from ``len(None)`` would
        abort an apply that was otherwise entirely fine."""
        glb = self._write_glb(materials=[{"name": "m"}])
        envelope = self._envelope({"emissive": None, "base_color": {"m": {"color": [0, 1, 0]}}})
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
        before = MeshConvert.read_scene_sidecar(glb)["textures"][big]["sha256"]

        MeshConvert.optimize_glb_textures(glb, max_size=32)
        after = MeshConvert.read_scene_sidecar(glb)["textures"][big]
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
            self.assertEqual(
                edit.gltf["extras"]["scene_sidecar_applied"], summary
            )
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
        """"Sidecar on, nothing to carry" must be visible in the artifact."""
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

        with patch("shutil.which", return_value=os.path.join(self.tmp, "bin")), patch(
            "subprocess.run", side_effect=fake_run
        ):
            out = MeshConvert.fbx_to_glb(src, dst, overwrite=True, sidecar=envelope)

        self._assert_embeds(out, envelope)
        with MeshConvert.open_glb(out) as edit:
            self.assertEqual(
                edit.gltf["materials"][0]["emissiveFactor"], [0.0, 1.0, 0.0]
            )


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

    def _scene(self, manifest, objects=("room",), texcoord1=True, material=0):
        """A minimal lit-scene GLB: mesh nodes + the data_export carrier node."""
        attrs = {"POSITION": 0, "TEXCOORD_0": 1}
        if texcoord1:
            attrs["TEXCOORD_1"] = 2
        nodes = [{"name": n, "mesh": i} for i, n in enumerate(objects)]
        nodes.append(
            {
                "name": "data_export",
                "extras": {
                    "fromFBX": {
                        "userProperties": {
                            "lightmap_metadata": {
                                "type": "eFbxString",
                                "value": json.dumps(manifest),
                            }
                        }
                    }
                },
            }
        )
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
        self.assertTrue(img["uri"].startswith("data:image/png"))
        web = gltf["extras"]["lightmap_web"]
        # The exact contract preview_viewer.html parses.
        self.assertEqual(web["carrier"], "occlusion")
        self.assertEqual(web["uv"], 1)
        self.assertEqual(web["encoding"], "srgb")
        self.assertAlmostEqual(
            web["materials"]["roomMat"]["intensity"], self.GOLDEN_CONSTANT, places=5
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
        # Scrub the leak, not the payload: what a consumer actually reads survives.
        self.assertEqual(node_marker["map"], os.path.basename(exr))
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
        a, _b = self._exr("a.exr"), self._exr("b.exr", value=0.25)
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
        self.assertEqual(
            prims["wall_a"]["attributes"], prims["wall_b"]["attributes"]
        )
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
        self.assertEqual(
            gltf_after["materials"][0]["occlusionTexture"], {"index": 0}
        )
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


if __name__ == "__main__":
    unittest.main()
