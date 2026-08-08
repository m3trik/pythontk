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

    def test_build_scene_sidecar_owns_the_frozen_top_level(self):
        """Standalone readers parse against exactly these keys."""
        envelope = self._envelope({"emissive": {"m": {"color": [1, 0, 0]}}})
        self.assertEqual(
            set(envelope),
            {"version", "source", "asset", "color_space", "sections"},
        )
        self.assertEqual(envelope["version"], MeshConvert.SIDECAR_VERSION)
        self.assertEqual(envelope["color_space"], "linear")
        self.assertEqual(envelope["asset"], "scene.fbx")

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
            self.assertEqual(edit.gltf["extras"]["scene_sidecar"], envelope)
            self.assertEqual(
                edit.gltf["extras"]["scene_sidecar_applied"], summary
            )
        self.assertEqual(MeshConvert.read_scene_sidecar(glb), envelope)

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
        self.assertEqual(MeshConvert.read_scene_sidecar(glb), envelope)

    def test_unknown_sections_are_skipped_never_fatal(self):
        """Forward compatibility: a reader skips sections it does not know."""
        glb = self._write_glb(materials=[{"name": "m"}])
        envelope = self._envelope({"lights": {"key": {"intensity": 5}}})
        self.assertEqual(MeshConvert.apply_scene_sidecar(glb, envelope), {})
        self.assertEqual(MeshConvert.read_scene_sidecar(glb), envelope)

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

        self.assertEqual(MeshConvert.read_scene_sidecar(out), envelope)
        with MeshConvert.open_glb(out) as edit:
            self.assertEqual(
                edit.gltf["materials"][0]["emissiveFactor"], [0.0, 1.0, 0.0]
            )


if __name__ == "__main__":
    unittest.main()
