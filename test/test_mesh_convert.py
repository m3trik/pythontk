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


if __name__ == "__main__":
    unittest.main()
