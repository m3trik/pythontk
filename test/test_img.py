#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk ImgUtils.

Comprehensive edge case coverage for:
- Image creation and manipulation
- Resize, save, load operations
- Map type detection and filtering
- Channel operations (invert, etc.)
- Normal map conversions (DX <-> GL)
- Color operations (mask, fill, replace)
- Mode conversions

Run with:
    python -m pytest test_img.py -v
    python test_img.py
"""

import ast
import os
import unittest
import shutil
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from pythontk import (
    FileUtils,
    ImgUtils,
    MapFactory as TextureMapFactory,
    TempArtifacts,
    TestSandbox,
)
from pythontk.img_utils._img_utils import ImageFormat
from pythontk.core_utils.engines.textures.map_optimizer import MapOptimizer

from conftest import BaseTestCase

try:
    import cv2  # noqa: F401

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

REPO_ROOT = Path(__file__).resolve().parents[1]


class ImgTest(BaseTestCase):
    """Image utilities test class with comprehensive edge case coverage."""

    # Class-level test images
    im_h = ImgUtils.create_image("RGB", (1024, 1024), (0, 0, 0))
    im_n = ImgUtils.create_image("RGB", (1024, 1024), (127, 127, 255))

    @classmethod
    def setUpClass(cls):
        # Generated into a throwaway dir: this used to rmtree and regenerate the
        # TRACKED test_files/imgtk_test/ (rewriting im_h.png & co. every run, and
        # as im_H.png on a case-sensitive FS, which test_file then tripped on).
        fixtures = TempArtifacts("pythontk_imgtest", policy="scoped")
        cls.addClassCleanup(fixtures.cleanup)  # runs even if setup fails below
        cls.test_dir = fixtures.dir_path()

        # Create base images
        cls.im_h.save(os.path.join(cls.test_dir, "im_H.png"))
        cls.im_n.save(os.path.join(cls.test_dir, "im_N.png"))

        # Create other expected images
        ImgUtils.create_image("RGB", (1024, 1024), (255, 0, 0)).save(
            os.path.join(cls.test_dir, "im_Base_color.png")
        )
        # Create 16-bit height map
        ImgUtils.create_image("I;16", (1024, 1024), 32767).save(
            os.path.join(cls.test_dir, "im_Height_16.png")
        )
        # Create 8-bit height map
        ImgUtils.create_image("L", (1024, 1024), 128).save(
            os.path.join(cls.test_dir, "im_Height_8.png")
        )
        # Create grayscale AO map (L mode)
        ImgUtils.create_image("L", (1024, 1024), 128).save(
            os.path.join(cls.test_dir, "im_Mixed_AO_L.png")
        )
        ImgUtils.create_image("RGB", (1024, 1024), (128, 128, 255)).save(
            os.path.join(cls.test_dir, "im_Normal_DirectX.png")
        )
        ImgUtils.create_image("RGB", (1024, 1024), (128, 128, 255)).save(
            os.path.join(cls.test_dir, "im_Normal_OpenGL.png")
        )

    # -------------------------------------------------------------------------
    # Image Creation Tests
    # -------------------------------------------------------------------------

    def test_create_image_basic(self):
        """Test create_image creates images with correct properties."""
        img = ImgUtils.create_image("RGB", (1024, 1024), (0, 0, 0))
        self.assertEqual(img.tobytes(), self.im_h.tobytes())

    def test_get_image_size_header_parse(self):
        """get_image_size reads (width, height) from the JPEG/PNG header with the
        stdlib alone (no PIL) — the path Metashape's bundled Python relies on.
        Non-square sizes catch any width/height transpose."""
        # Use a private temp dir — must not pollute the shared imgtk fixture dir
        # (test_get_images_all globs "*" there and would choke on bad.bin).
        with tempfile.TemporaryDirectory() as tmp:
            png = os.path.join(tmp, "size_probe.png")
            jpg = os.path.join(tmp, "size_probe.jpg")
            ImgUtils.create_image("RGB", (800, 600), (10, 20, 30)).save(png, "PNG")
            ImgUtils.create_image("RGB", (640, 480), (10, 20, 30)).save(jpg, "JPEG")
            # stdlib-only header parse, right (width, height) order
            self.assertEqual(ImgUtils._image_size_from_header(png), (800, 600))
            self.assertEqual(ImgUtils._image_size_from_header(jpg), (640, 480))
            # public API agrees
            self.assertEqual(ImgUtils.get_image_size(png), (800, 600))
            self.assertEqual(ImgUtils.get_image_size(jpg), (640, 480))
            # garbage -> None, never raises
            bad = os.path.join(tmp, "bad.bin")
            with open(bad, "wb") as f:
                f.write(b"not an image")
            self.assertIsNone(ImgUtils._image_size_from_header(bad))
            self.assertIsNone(ImgUtils.get_image_size(bad))

    def test_the_header_parse_reads_tga_and_dds(self):
        """The common game formats too: an interpreter with no PIL (Blender's)
        otherwise sizes a TGA or DDS map by decoding the whole image."""
        import struct

        with tempfile.TemporaryDirectory() as tmp:
            tga = os.path.join(tmp, "size_probe.tga")
            ImgUtils.create_image("RGBA", (320, 200), (1, 2, 3, 4)).save(tga, "TGA")
            self.assertEqual(ImgUtils._image_size_from_header(tga), (320, 200))
            dds = os.path.join(tmp, "size_probe.dds")
            with open(dds, "wb") as f:  # "DDS ", size 124, flags, height, width
                f.write(b"DDS " + struct.pack("<IIII", 124, 0x1007, 256, 512))
                f.write(b"\0" * 108)
            self.assertEqual(ImgUtils._image_size_from_header(dds), (512, 256))
            # No signature to go on: a .tga that fails the header's own sanity
            # is refused, never read as a size.
            fake = os.path.join(tmp, "fake.tga")
            with open(fake, "wb") as f:
                f.write(b"not an image at all")
            self.assertIsNone(ImgUtils._image_size_from_header(fake))

    @staticmethod
    def _exr_header(data_window, *, before=()):
        """An OpenEXR header: magic, version, *before* attributes, then dataWindow."""
        import struct

        def attr(name, type_name, value):
            return (
                name + b"\0" + type_name + b"\0" + struct.pack("<i", len(value)) + value
            )

        blob = b"\x76\x2f\x31\x01" + struct.pack("<I", 2)
        for name, type_name, value in before:
            blob += attr(name, type_name, value)
        blob += attr(b"dataWindow", b"box2i", struct.pack("<iiii", *data_window))
        return blob + b"\0"

    def test_the_header_parse_reads_exr_and_radiance_hdr(self):
        """HDR formats too: PIL reads neither, so without this an EXR or .hdr
        environment map had no size at all -- the HDR manager needs its aspect
        to tell a 2:1 latlong from a square lightmap."""
        with tempfile.TemporaryDirectory() as tmp:
            exr = os.path.join(tmp, "env.exr")
            with open(exr, "wb") as f:
                # A window not at the origin, behind a large attribute the parse
                # must seek past rather than read (Arnold stamps render metadata).
                f.write(
                    self._exr_header(
                        (-10, 5, 2037, 1028),
                        before=[(b"comments", b"string", b"x" * 70000)],
                    )
                )
            self.assertEqual(ImgUtils._image_size_from_header(exr), (2048, 1024))
            self.assertEqual(ImgUtils.get_image_size(exr), (2048, 1024))

            no_window = os.path.join(tmp, "no_window.exr")
            with open(no_window, "wb") as f:
                f.write(b"\x76\x2f\x31\x01" + b"\x02\0\0\0" + b"\0")
            self.assertIsNone(ImgUtils._image_size_from_header(no_window))

            hdr = os.path.join(tmp, "env.hdr")
            with open(hdr, "wb") as f:
                f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 512 +X 1024\n")
            self.assertEqual(ImgUtils._image_size_from_header(hdr), (1024, 512))
            # Rotated storage lists X first; each value belongs to its axis.
            rotated = os.path.join(tmp, "rotated.hdr")
            with open(rotated, "wb") as f:
                f.write(b"#?RGBE\n\n+X 300 -Y 100\n")
            self.assertEqual(ImgUtils._image_size_from_header(rotated), (300, 100))
            unended = os.path.join(tmp, "unended.hdr")
            with open(unended, "wb") as f:
                f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n")
            self.assertIsNone(ImgUtils._image_size_from_header(unended))
            # CRLF is not a Radiance header per the format. Size and integrity
            # check read it through ONE parse, so they agree: the sizer used to
            # size it (8, 4) while the check refused it.
            crlf = os.path.join(tmp, "crlf.hdr")
            with open(crlf, "wb") as f:
                f.write(
                    b"#?RADIANCE\r\nFORMAT=32-bit_rle_rgbe\r\n\r\n-Y 4 +X 8\r\n"
                    + b"\x10" * 128
                )
            self.assertIsNone(ImgUtils._image_size_from_header(crlf))
            self.assertEqual(
                ImgUtils.validate_image_integrity(crlf), (False, "incomplete header")
            )

    @staticmethod
    def _exr_channels(*channels):
        """An EXR ``channels`` (chlist) attribute: ``(name, pixel type)`` pairs,
        pixel type 0 uint / 1 half / 2 float."""
        import struct

        value = b"".join(
            name + b"\0" + struct.pack("<iB3xii", kind, 0, 1, 1)
            for name, kind in channels
        )
        return (b"channels", b"chlist", value + b"\0")

    def test_get_image_mode_reads_the_layout_without_decoding(self):
        """The channel layout a texture table shows beside its size: PIL's
        mode name where PIL reads the file (a lazy open, header only), the
        EXR ``channels`` header and Radiance's fixed RGB float where it does
        not -- the lightmaps are EXR. Added: 2026-10-04
        """
        with tempfile.TemporaryDirectory() as tmp:
            for mode, ext in (("RGBA", "png"), ("RGB", "jpg"), ("L", "png")):
                path = os.path.join(tmp, f"{mode}.{ext}")
                ImgUtils.create_image(mode, (8, 4), 0).save(path)
                self.assertEqual(ImgUtils.get_image_mode(path), mode, path)

            def exr(name, *channels):
                path = os.path.join(tmp, name)
                with open(path, "wb") as f:
                    f.write(
                        self._exr_header(
                            (0, 0, 7, 3),
                            before=[
                                (b"comments", b"string", b"x" * 5000),
                                self._exr_channels(*channels),
                            ],
                        )
                    )
                return path

            # EXR lists channels alphabetically; the mode reads in RGBA order.
            half = exr("half.exr", (b"B", 1), (b"G", 1), (b"R", 1))
            self.assertEqual(ImgUtils.get_image_mode(half), "RGB;16F")
            full = exr("full.exr", (b"A", 2), (b"B", 2), (b"G", 2), (b"R", 2))
            self.assertEqual(ImgUtils.get_image_mode(full), "RGBA;32F")
            luma = exr("luma.exr", (b"Y", 1))
            self.assertEqual(ImgUtils.get_image_mode(luma), "L;16F")
            layered = exr("aov.exr", (b"diffuse.R", 1), (b"spec.R", 1))
            self.assertEqual(ImgUtils.get_image_mode(layered), "2ch;16F")
            no_list = os.path.join(tmp, "no_list.exr")
            with open(no_list, "wb") as f:
                f.write(self._exr_header((0, 0, 7, 3)))
            self.assertIsNone(ImgUtils.get_image_mode(no_list))

            hdr = os.path.join(tmp, "env.hdr")
            with open(hdr, "wb") as f:
                f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n")
            self.assertEqual(ImgUtils.get_image_mode(hdr), "RGB;32F")

            bad = os.path.join(tmp, "bad.bin")
            with open(bad, "wb") as f:
                f.write(b"not an image")
            self.assertIsNone(ImgUtils.get_image_mode(bad))
            self.assertIsNone(ImgUtils.get_image_mode(os.path.join(tmp, "gone.png")))

    def test_texture_facts_reads_only_what_is_asked(self):
        """A texture table's optional columns: each fact read only when its
        column shows, a tile set summed / sized by its first tile, and an
        online-only placeholder never opened. Added: 2026-10-04"""
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            for tile in (1001, 1002):
                ImgUtils.create_image("RGBA", (16, 8), 0).save(
                    os.path.join(tmp, f"rock.{tile}.png")
                )
            pattern = os.path.join(tmp, "rock.<UDIM>.png")
            tile_bytes = os.path.getsize(os.path.join(tmp, "rock.1001.png"))

            with mock.patch.object(ImgUtils, "get_image_mode") as mode:
                facts = ImgUtils.texture_facts(pattern, ("bytes",))
            mode.assert_not_called()
            self.assertEqual(facts["tiles"], 2)
            self.assertEqual(
                facts["bytes"],
                tile_bytes + os.path.getsize(os.path.join(tmp, "rock.1002.png")),
            )
            self.assertNotIn("mode", facts)

            facts = ImgUtils.texture_facts(pattern, ("dimensions", "mode"))
            self.assertEqual((facts["dimensions"], facts["mode"]), ((16, 8), "RGBA"))

            self.assertEqual(ImgUtils.texture_facts(os.path.join(tmp, "x.png")), {})

            single = os.path.join(tmp, "flat.png")
            ImgUtils.create_image("RGB", (4, 4), 0).save(single)
            with (
                mock.patch.object(FileUtils, "is_cloud_placeholder", return_value=True),
                mock.patch.object(ImgUtils, "get_image_size") as sizer,
            ):
                facts = ImgUtils.texture_facts(single)
            sizer.assert_not_called()
            self.assertTrue(facts["online_only"])
            self.assertIsNone(facts["dimensions"])
            self.assertGreater(facts["bytes"], 0, "a stat downloads nothing")

    def test_is_environment_map_by_shape_and_name(self):
        """An HDR picker's filter over a folder that also holds baked lightmaps
        (a production sourceimages held 72, every one square). Sandboxed: the
        name half reads the convention, which a developer may have edited."""
        with TestSandbox.user_config(), tempfile.TemporaryDirectory() as tmp:

            def exr(name, size):
                path = os.path.join(tmp, name)
                with open(path, "wb") as f:
                    f.write(self._exr_header((0, 0, size[0] - 1, size[1] - 1)))
                return path

            env = exr("workshop_8k.exr", (2048, 1024))
            pano = exr("skyDome.exr", (4096, 2004))  # 2.044: Maya's own
            lightmap = exr("ROOM_ENV_Lightmap_12.exr", (256, 256))
            texture = exr("crate.exr", (512, 512))
            wide_lightmap = exr("desk_LightMap.LIGHT_A.exr", (2048, 1024))

            self.assertTrue(ImgUtils.is_equirectangular(env))
            self.assertTrue(ImgUtils.is_equirectangular(pano))
            self.assertFalse(ImgUtils.is_equirectangular(texture))
            self.assertIsNone(ImgUtils.is_equirectangular(os.path.join(tmp, "x.hdr")))

            self.assertTrue(ImgUtils.is_environment_map(env))
            self.assertTrue(ImgUtils.is_environment_map(pano))
            for path in (lightmap, texture, wide_lightmap):
                with self.subTest(path=os.path.basename(path)):
                    self.assertFalse(ImgUtils.is_environment_map(path))
            # Each filter switches off on its own.
            self.assertTrue(ImgUtils.is_environment_map(texture, latlong_only=False))
            self.assertTrue(
                ImgUtils.is_environment_map(wide_lightmap, skip_lightmaps=False)
            )
            # Square AND lightmap-named: either filter alone still hides it.
            self.assertFalse(ImgUtils.is_environment_map(lightmap, latlong_only=False))
            self.assertFalse(
                ImgUtils.is_environment_map(lightmap, skip_lightmaps=False)
            )
            # Unknown size is never a false reject.
            self.assertTrue(ImgUtils.is_environment_map(os.path.join(tmp, "x.hdr")))

    def test_is_equirectangular_never_reads_an_online_only_file(self):
        """Even a header read makes the sync client download the whole image --
        100 MB per HDR on every dropdown open."""
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "crate.exr")
            with open(path, "wb") as f:
                f.write(self._exr_header((0, 0, 511, 511)))
            with (
                mock.patch.object(FileUtils, "is_cloud_placeholder", return_value=True),
                mock.patch.object(ImgUtils, "get_image_size") as sizer,
            ):
                self.assertIsNone(ImgUtils.is_equirectangular(path))
                self.assertTrue(ImgUtils.is_environment_map(path))
            sizer.assert_not_called()

    def test_is_equirectangular_rereads_a_rewritten_file(self):
        """Sizes are cached per file version, not per path."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "env.exr")
            with open(path, "wb") as f:
                f.write(self._exr_header((0, 0, 511, 511)))
            self.assertFalse(ImgUtils.is_equirectangular(path))
            with open(path, "wb") as f:  # a different byte size: a new version
                f.write(
                    self._exr_header(
                        (0, 0, 1023, 511), before=[(b"owner", b"string", b"me")]
                    )
                )
            self.assertTrue(ImgUtils.is_equirectangular(path))

    @unittest.skipUnless(HAS_CV2, "cv2 not installed")
    def test_the_header_parse_reads_an_exr_cv2_wrote(self):
        """The lightmap baker writes its EXRs through cv2 -- a real header."""
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            exr = os.path.join(tmp, "desk_Lightmap.exr")
            img = np.zeros((96, 160, 3), np.float32)
            if not cv2.imwrite(
                exr, img, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]
            ):
                self.skipTest("cv2 built without OpenEXR write support")
            self.assertEqual(ImgUtils.get_image_size(exr), (160, 96))

    def test_create_image_rgb(self):
        """Test create_image with RGB mode."""
        img = ImgUtils.create_image("RGB", (100, 100), (255, 0, 0))
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.size, (100, 100))

    def test_create_image_rgba(self):
        """Test create_image with RGBA mode."""
        img = ImgUtils.create_image("RGBA", (100, 100), (255, 0, 0, 128))
        self.assertEqual(img.mode, "RGBA")
        self.assertEqual(img.size, (100, 100))

    def test_create_image_grayscale(self):
        """Test create_image with grayscale mode."""
        img = ImgUtils.create_image("L", (100, 100), 128)
        self.assertEqual(img.mode, "L")
        self.assertEqual(img.size, (100, 100))

    def test_create_image_1x1(self):
        """Test create_image with 1x1 pixel (smallest possible)."""
        img = ImgUtils.create_image("RGB", (1, 1), (128, 128, 128))
        self.assertEqual(img.size, (1, 1))

    def test_create_image_large(self):
        """Test create_image with large dimensions."""
        img = ImgUtils.create_image("RGB", (4096, 4096), (0, 0, 0))
        self.assertEqual(img.size, (4096, 4096))

    def test_create_image_non_square(self):
        """Test create_image with non-square dimensions."""
        img = ImgUtils.create_image("RGB", (100, 200), (0, 0, 0))
        self.assertEqual(img.size, (100, 200))

    # -------------------------------------------------------------------------
    # Resize Tests
    # -------------------------------------------------------------------------

    def test_resize_image_basic(self):
        """Test resize_image changes image dimensions."""
        resized = ImgUtils.resize_image(self.im_h, 32, 32)
        self.assertEqual(resized.size, (32, 32))

    def test_resize_image_upscale(self):
        """Test resize_image upscaling."""
        small = ImgUtils.create_image("RGB", (32, 32), (0, 0, 0))
        resized = ImgUtils.resize_image(small, 128, 128)
        self.assertEqual(resized.size, (128, 128))

    def test_resize_image_downscale(self):
        """Test resize_image downscaling."""
        large = ImgUtils.create_image("RGB", (512, 512), (0, 0, 0))
        resized = ImgUtils.resize_image(large, 64, 64)
        self.assertEqual(resized.size, (64, 64))

    def test_resize_image_non_square(self):
        """Test resize_image to non-square."""
        resized = ImgUtils.resize_image(self.im_h, 100, 50)
        self.assertEqual(resized.size, (100, 50))

    def test_resize_image_to_1x1(self):
        """Test resize_image to minimum size."""
        resized = ImgUtils.resize_image(self.im_h, 1, 1)
        self.assertEqual(resized.size, (1, 1))

    # -------------------------------------------------------------------------
    # Save/Load Tests
    # -------------------------------------------------------------------------

    def test_save_image_file(self):
        """Test save_image writes image to disk."""
        path_h = os.path.join(self.test_dir, "im_h.png")
        path_n = os.path.join(self.test_dir, "im_n.png")
        result_h = ImgUtils.save_image(self.im_h, path_h)
        result_n = ImgUtils.save_image(self.im_n, path_n)
        self.assertIsNone(result_h)
        self.assertIsNone(result_n)

    def test_save_image_overwrites(self):
        """Test save_image can overwrite existing files."""
        path = os.path.join(self.test_dir, "im_h.png")
        img1 = ImgUtils.create_image("RGB", (100, 100), (255, 0, 0))
        ImgUtils.save_image(img1, path)
        img2 = ImgUtils.create_image("RGB", (100, 100), (0, 255, 0))
        result = ImgUtils.save_image(img2, path)
        self.assertIsNone(result)

    # -------------------------------------------------------------------------
    # get_images Tests
    # -------------------------------------------------------------------------

    def test_get_images_pattern(self):
        """Test get_images finds images by pattern."""
        images = ImgUtils.get_images(self.test_dir, "*Normal*")
        expected = [
            os.path.join(self.test_dir, "im_Normal_DirectX.png"),
            os.path.join(self.test_dir, "im_Normal_OpenGL.png"),
        ]
        # Normalize paths for comparison
        found = sorted([os.path.normpath(p) for p in images.keys()])
        expected = sorted([os.path.normpath(p) for p in expected])
        self.assertEqual(found, expected)

    def test_get_images_all(self):
        """Test get_images with wildcard to get all images."""
        images = ImgUtils.get_images(self.test_dir, "*")
        self.assertGreater(len(images), 0)

    def test_get_images_no_match(self):
        """Test get_images with pattern that matches nothing."""
        images = ImgUtils.get_images(self.test_dir, "*NonexistentPattern*")
        self.assertEqual(len(images), 0)

    # -------------------------------------------------------------------------
    # Map Type Resolution Tests
    # -------------------------------------------------------------------------

    def test_resolve_map_type_height(self):
        """Test resolve_map_type identifies height maps."""
        self.assertEqual(
            TextureMapFactory.resolve_map_type(os.path.join(self.test_dir, "im_H.png")),
            "Height",
        )
        self.assertEqual(
            TextureMapFactory.resolve_map_type(
                os.path.join(self.test_dir, "im_H.png"), key=False
            ),
            "H",
        )

    def test_resolve_map_type_normal(self):
        """Test resolve_map_type identifies normal maps."""
        self.assertEqual(
            TextureMapFactory.resolve_map_type(os.path.join(self.test_dir, "im_N.png")),
            "Normal",
        )
        self.assertEqual(
            TextureMapFactory.resolve_map_type(
                os.path.join(self.test_dir, "im_N.png"), key=False
            ),
            "N",
        )

    def test_resolve_map_type_unknown(self):
        """Test resolve_map_type with unknown map type."""
        result = TextureMapFactory.resolve_map_type("random_file.png")
        # Should return None or empty for unknown types
        self.assertTrue(result is None or result == "")

    def test_resolve_map_type_prefers_longest_alias(self):
        """resolve_map_type(key=False) must prefer the longest alias match.

        Regression: previously returned 'AO' for 'Cart_Mixed_AO.png' because
        'AO' iterated before 'Mixed_AO' in the aliases list. Output round-trip
        through resolve_texture_filename then dropped 'Mixed_'.
        """
        self.assertEqual(
            TextureMapFactory.resolve_map_type("Cart_Mixed_AO.png", key=False),
            "Mixed_AO",
        )
        self.assertEqual(
            TextureMapFactory.resolve_map_type("Cart_AmbientOcclusion.PNG", key=False),
            "AmbientOcclusion",
        )

    def test_resolve_map_type_requires_underscore_boundary(self):
        """resolve_map_type(key=False) must not match mid-word.

        Regression: 'diffuse_cube.dds' returned 'E' because filename ends in 'e'
        and 'E' is an Emissive alias. 'ibl_brdf_lut.dds' returned 'lut' via the
        fallback path. Both should return None now.
        """
        for fn in ("diffuse_cube.dds", "specular_cube.dds", "ibl_brdf_lut.dds"):
            self.assertIsNone(
                TextureMapFactory.resolve_map_type(fn, key=False),
                f"expected None for {fn}",
            )

    def test_resolve_map_type_preserves_filename_case(self):
        """The returned alias must match the case in the filename, not the
        canonical alias case in the registry, so re-saving doesn't rename."""
        self.assertEqual(
            TextureMapFactory.resolve_map_type("foo_Base_color.png", key=False),
            "Base_color",
        )

    def test_resolve_texture_filename_preserves_when_map_type_empty(self):
        """When the resolver can't identify a map type, resolve_texture_filename
        must NOT synthesize a renamed path — it should round-trip the original.
        """
        for fn in ("diffuse_cube.dds", "ibl_brdf_lut.dds", "random_file.png"):
            out = TextureMapFactory.resolve_texture_filename(fn, "")
            self.assertEqual(os.path.basename(out), fn)
            out_none = TextureMapFactory.resolve_texture_filename(fn, None or "")
            self.assertEqual(os.path.basename(out_none), fn)

    def test_resolve_map_type_validate_is_case_insensitive(self):
        """validate= must accept filename-cased results.

        Regression: after switching key=False to return filename-cased aliases,
        validate compared against canonical-case registry entries and raised
        ValueError on perfectly valid files with lowercase names.
        """
        # Lowercase filename — alias is "Normal_DirectX" canonically.
        result = TextureMapFactory.resolve_map_type(
            "asset_normal_directx.png", key=False, validate="Normal_DirectX"
        )
        self.assertEqual(result, "normal_directx")
        # Should not raise.
        TextureMapFactory.resolve_map_type(
            "asset_Normal.png", key=False, validate="Normal"
        )
        # Genuinely invalid type still raises.
        with self.assertRaises(ValueError):
            TextureMapFactory.resolve_map_type(
                "asset_Roughness.png", key=False, validate="Normal"
            )

    def test_resolve_texture_filename_round_trip_preserves_suffix_exact(self):
        """End-to-end: detect alias from filename, pass it back to
        resolve_texture_filename. Output filename must equal input.
        """
        cases = [
            "Cart_Mixed_AO.png",
            "Cart_AmbientOcclusion.PNG",
            "foo_Base_color.png",
            "asset_Roughness.png",
        ]
        for fn in cases:
            rt = TextureMapFactory.resolve_map_type(fn, key=False)
            self.assertIsNotNone(rt, f"resolver dropped suffix for {fn}")
            out = TextureMapFactory.resolve_texture_filename(fn, rt)
            self.assertEqual(
                os.path.basename(out),
                fn,
                f"round-trip changed filename: {fn} -> {os.path.basename(out)}",
            )

    # -------------------------------------------------------------------------
    # Filter Images By Type Tests
    # -------------------------------------------------------------------------

    def test_filter_images_by_type_height(self):
        """Test filter_images_by_type filters by texture type."""
        files = FileUtils.get_dir_contents(self.test_dir)
        filtered = TextureMapFactory.filter_images_by_type(files, "Height")
        expected = ["im_H.png"]
        # Sort for comparison
        self.assertEqual(sorted(filtered), sorted(expected))

    def test_filter_images_by_type_empty_list(self):
        """Test filter_images_by_type with empty list."""
        result = TextureMapFactory.filter_images_by_type([], "Height")
        self.assertEqual(result, [])

    def test_filter_images_by_type_no_match(self):
        """Test filter_images_by_type when no images match."""
        result = TextureMapFactory.filter_images_by_type(["random.txt"], "Height")
        self.assertEqual(result, [])

    # -------------------------------------------------------------------------
    # Sort Images By Type Tests
    # -------------------------------------------------------------------------

    def test_sort_images_by_type_list(self):
        """Test sort_images_by_type groups images by texture type."""
        self.assertEqual(
            TextureMapFactory.sort_images_by_type(
                [("im_H.png", "<im_h>"), ("im_N.png", "<im_n>")]
            ),
            {
                "Height": [("im_H.png", "<im_h>")],
                "Normal": [("im_N.png", "<im_n>")],
            },
        )

    def test_sort_images_by_type_dict(self):
        """Test sort_images_by_type with dict input."""
        self.assertEqual(
            TextureMapFactory.sort_images_by_type(
                {"im_H.png": "<im_h>", "im_N.png": "<im_n>"}
            ),
            {
                "Height": [("im_H.png", "<im_h>")],
                "Normal": [("im_N.png", "<im_n>")],
            },
        )

    def test_sort_images_by_type_empty(self):
        """Test sort_images_by_type with empty input."""
        result = TextureMapFactory.sort_images_by_type([])
        self.assertEqual(result, {})

    # -------------------------------------------------------------------------
    # Contains Map Types Tests
    # -------------------------------------------------------------------------

    def test_contains_map_types_list(self):
        """Test contains_map_types with list input."""
        self.assertTrue(
            TextureMapFactory.contains_map_types([("im_H.png", "<im_h>")], "Height")
        )

    def test_contains_map_types_dict(self):
        """Test contains_map_types with dict input."""
        self.assertTrue(
            TextureMapFactory.contains_map_types(
                {"im_H.png": "<im_h>", "im_N.png": "<im_n>"}, "Height"
            )
        )

    def test_contains_map_types_sorted_dict(self):
        """Test contains_map_types with pre-sorted dict."""
        self.assertTrue(
            TextureMapFactory.contains_map_types(
                {"Height": [("im_H.png", "<im_h>")]}, "Height"
            )
        )

    def test_contains_map_types_multiple(self):
        """Test contains_map_types with multiple types."""
        self.assertTrue(
            TextureMapFactory.contains_map_types(
                {"Height": [("im_H.png", "<im_h>")]}, ["Height", "Normal"]
            )
        )

    def test_contains_map_types_not_found(self):
        """Test contains_map_types when type not present."""
        self.assertFalse(
            TextureMapFactory.contains_map_types([("im_H.png", "<im_h>")], "Roughness")
        )

    # -------------------------------------------------------------------------
    # Is Normal Map Tests
    # -------------------------------------------------------------------------

    def test_is_normal_map_false(self):
        """Test is_normal_map returns False for non-normal maps."""
        self.assertFalse(TextureMapFactory.is_normal_map("im_H.png"))

    def test_is_normal_map_true(self):
        """Test is_normal_map returns True for normal maps."""
        self.assertTrue(TextureMapFactory.is_normal_map("im_N.png"))

    def test_is_normal_map_explicit_name(self):
        """Test is_normal_map with explicit normal map name."""
        self.assertTrue(TextureMapFactory.is_normal_map("texture_Normal.png"))

    def test_is_normal_map_directx(self):
        """Test is_normal_map with DirectX normal map."""
        self.assertTrue(TextureMapFactory.is_normal_map("im_Normal_DirectX.png"))

    def test_is_normal_map_opengl(self):
        """Test is_normal_map with OpenGL normal map."""
        self.assertTrue(TextureMapFactory.is_normal_map("im_Normal_OpenGL.png"))

    # -------------------------------------------------------------------------
    # Channel Operations Tests
    # -------------------------------------------------------------------------

    def test_invert_channels_green(self):
        """Test invert_channels inverts green channel."""
        result = ImgUtils.invert_channels(self.im_n, "g")
        channel = result.getchannel("G")
        self.assertEqual(channel.mode, "L")

    def test_invert_channels_red(self):
        """Test invert_channels inverts red channel."""
        result = ImgUtils.invert_channels(self.im_n, "r")
        channel = result.getchannel("R")
        self.assertEqual(channel.mode, "L")

    def test_invert_channels_multiple(self):
        """Test invert_channels with multiple channels - preserves original mode."""
        result = ImgUtils.invert_channels(self.im_n, "rg")
        # The function preserves the original image mode (RGBA for PNG with alpha)
        self.assertIn(result.mode, ["RGB", "RGBA"])

    def test_invert_channels_la_targets_real_alpha(self):
        """Regression: invert_channels on an LA (grayscale+alpha) image must not
        raise KeyError 'A', and channels='A' must invert the true alpha band
        while leaving the luminance band untouched."""
        im = Image.new("LA", (4, 4), (100, 200))  # L=100, A=200
        result = ImgUtils.invert_channels(im, "A")
        self.assertEqual(result.mode, "LA")
        lum, alpha = result.split()
        self.assertEqual(lum.getpixel((0, 0)), 100)  # luminance untouched
        self.assertEqual(alpha.getpixel((0, 0)), 55)  # 255 - 200 inverted

    def test_srgb_to_linear_preserves_rgba_alpha(self):
        """Regression: sRGB->linear on an RGBA image must preserve the alpha
        channel instead of blowing it out to 255."""
        im = Image.new("RGBA", (4, 4), (128, 128, 128, 64))
        result = ImgUtils.srgb_to_linear(im)
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.getpixel((0, 0))[3], 64)

    def test_gaussian_blur_preserves_float_array(self):
        """Regression: blurring a float numpy array in [0,1] must preserve its
        magnitude rather than truncating it to uint8 (all-zero) via the PIL path."""
        arr = np.full((16, 16), 0.7, np.float32)
        out = ImgUtils.gaussian_blur(arr, radius=2.0)
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(float(out.max()), 0.7, places=5)

    # -------------------------------------------------------------------------
    # Channel Swizzle Tests
    # -------------------------------------------------------------------------

    def test_swizzle_channels_dict_swaps(self):
        """A dict mapping swaps the named channels and preserves channel count."""
        im = ImgUtils.create_image("RGB", (4, 4), (10, 20, 30))
        result = ImgUtils.swizzle_channels(im, {"R": "B", "B": "R"})
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (30, 20, 10))

    def test_swizzle_channels_string_sets_output_count(self):
        """A string mapping's length drives the output mode; chars name sources."""
        im = ImgUtils.create_image("RGBA", (4, 4), (10, 20, 30, 40))
        # 'BGR' -> 3-channel output with red/blue swapped, alpha dropped.
        result = ImgUtils.swizzle_channels(im, "BGR")
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (30, 20, 10))

    def test_swizzle_channels_constant_fill(self):
        """Constants '0'/'1' fill a destination with black/white."""
        im = ImgUtils.create_image("RGBA", (4, 4), (10, 20, 30, 40))
        result = ImgUtils.swizzle_channels(im, {"A": "1"})
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.getpixel((0, 0)), (10, 20, 30, 255))

    def test_swizzle_channels_missing_source_resolves_white(self):
        """Pulling alpha from an RGB input (no alpha) resolves to white."""
        im = ImgUtils.create_image("RGB", (4, 4), (10, 20, 30))
        result = ImgUtils.swizzle_channels(im, "RGBA")
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.getpixel((0, 0)), (10, 20, 30, 255))

    def test_swizzle_channels_dict_adds_alpha_to_rgb(self):
        """A dict naming an 'A' destination promotes an RGB input to RGBA."""
        im = ImgUtils.create_image("RGB", (4, 4), (10, 20, 30))
        result = ImgUtils.swizzle_channels(im, {"A": "R"})
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.getpixel((0, 0)), (10, 20, 30, 10))

    def test_swizzle_channels_dict_rgb_untouched_alpha_stays_rgb(self):
        """A dict that doesn't name 'A' leaves an RGB input as RGB."""
        im = ImgUtils.create_image("RGB", (4, 4), (10, 20, 30))
        result = ImgUtils.swizzle_channels(im, {"R": "G"})
        self.assertEqual(result.mode, "RGB")

    def test_swizzle_channels_invalid_source_raises(self):
        """An unknown source token is rejected."""
        im = ImgUtils.create_image("RGB", (4, 4), (10, 20, 30))
        with self.assertRaises(ValueError):
            ImgUtils.swizzle_channels(im, {"R": "Z"})

    # -------------------------------------------------------------------------
    # Normal Map Conversion Tests
    # -------------------------------------------------------------------------

    def test_convert_normal_map_format_gl_to_dx(self):
        """Test convert_normal_map_format converts OpenGL to DirectX normal maps."""
        input_path = os.path.join(self.test_dir, "im_Normal_OpenGL.png")
        dx_path = TextureMapFactory.convert_normal_map_format(
            input_path, target_format="directx"
        )
        expected = os.path.join(self.test_dir, "im_Normal_DirectX.png")
        self.assertEqual(os.path.normpath(dx_path), os.path.normpath(expected))

    def test_convert_normal_map_format_dx_to_gl(self):
        """Test convert_normal_map_format converts DirectX to OpenGL normal maps."""
        input_path = os.path.join(self.test_dir, "im_Normal_DirectX.png")
        gl_path = TextureMapFactory.convert_normal_map_format(
            input_path, target_format="opengl"
        )
        expected = os.path.join(self.test_dir, "im_Normal_OpenGL.png")
        self.assertEqual(os.path.normpath(gl_path), os.path.normpath(expected))

    def test_convert_normal_map_format_keeps_the_source_spelling(self):
        """A non-canonical suffix must convert to its paired spelling.

        The two tests above use canonical names only, which sit at index 0 of
        both alias tuples — so they passed while the suffix was being paired by
        INDEX across two lists of different lengths. Any other spelling mapped to
        an unrelated alias (``im_NDX`` -> ``im_NRMGL``).
        """
        # Its own scratch dir, not the class fixture dir: the conversion writes a
        # sibling file per case, and `test_get_images_pattern` globs `*Normal*`
        # over the shared dir expecting exactly the two fixtures.
        artifacts = TempArtifacts(prefix="test_normal_spelling", policy="scoped")
        scratch = artifacts.dir_path()
        try:
            for src_suffix, target, expected_suffix in (
                ("NormalDX", "opengl", "NormalGL"),
                ("NDX", "opengl", "NGL"),
                ("Normal_DX", "opengl", "Normal_GL"),
                ("NormalGL", "directx", "NormalDX"),
                ("NGL", "directx", "NDX"),
            ):
                input_path = os.path.join(scratch, f"im_{src_suffix}.png")
                ImgUtils.save_image(
                    ImgUtils.create_image("RGB", (8, 8), (127, 127, 255)), input_path
                )
                out = TextureMapFactory.convert_normal_map_format(
                    input_path, target_format=target
                )
                expected = os.path.join(scratch, f"im_{expected_suffix}.png")
                self.assertEqual(
                    os.path.normpath(out),
                    os.path.normpath(expected),
                    f"{src_suffix} -> {target}",
                )
        finally:
            artifacts.cleanup()

    # -------------------------------------------------------------------------
    # Mask Tests
    # -------------------------------------------------------------------------

    def test_create_mask_from_background(self):
        """Test create_mask generates image masks from background color."""
        input_path = os.path.join(self.test_dir, "im_Base_color.png")
        bg = ImgUtils.get_background(input_path, "RGB")
        mask = ImgUtils.create_mask(input_path, bg)
        self.assertEqual(mask.mode, "L")

    def test_create_mask_from_image(self):
        """Test create_mask generates mask from another image."""
        input_path = os.path.join(self.test_dir, "im_Base_color.png")
        mask = ImgUtils.create_mask(
            input_path,
            input_path,
        )
        self.assertEqual(mask.mode, "L")

    def test_create_mask_non_square(self):
        """create_mask must handle non-square images (regression: corner
        indexing transposed numpy's row/column axes and raised IndexError)."""
        im = Image.new("RGB", (32, 8), (10, 20, 30))
        mask = ImgUtils.create_mask(im, (10, 20, 30))
        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.size, (32, 8))

    def test_create_mask_foreground_background(self):
        """Matched (mask-color) pixels go to background; the rest to foreground."""
        im = Image.new("RGB", (4, 4), (10, 20, 30))
        im.putpixel((2, 2), (200, 200, 200))
        mask = ImgUtils.create_mask(im, (10, 20, 30))
        self.assertEqual(mask.getpixel((1, 1)), 0)  # matched -> background (black)
        self.assertEqual(mask.getpixel((2, 2)), 255)  # unmatched -> foreground

    # -------------------------------------------------------------------------
    # Fill Tests
    # -------------------------------------------------------------------------

    def test_fill_masked_area(self):
        """Test fill_masked_area fills masked regions with color."""
        input_path = os.path.join(self.test_dir, "im_Base_color.png")
        bg = ImgUtils.get_background(input_path, "RGB")
        mask = ImgUtils.create_mask(input_path, bg)
        result = ImgUtils.fill_masked_area(input_path, (0, 255, 0), mask)
        self.assertEqual(result.mode, "RGB")

    def test_fill_solid_color(self):
        """Test fill fills image with color."""
        result = ImgUtils.fill(self.im_h.copy(), (127, 127, 127))
        self.assertEqual(result.mode, "RGB")

    def test_fill_black(self):
        """Test fill with black color."""
        result = ImgUtils.fill(self.im_h.copy(), (0, 0, 0))
        self.assertEqual(result.mode, "RGB")

    def test_fill_white(self):
        """Test fill with white color."""
        result = ImgUtils.fill(self.im_h.copy(), (255, 255, 255))
        self.assertEqual(result.mode, "RGB")

    # -------------------------------------------------------------------------
    # Background Detection Tests
    # -------------------------------------------------------------------------

    def test_get_background_i_mode(self):
        """Test get_background with I mode."""
        self.assertEqual(
            ImgUtils.get_background(
                os.path.join(self.test_dir, "im_Height_16.png"), "I"
            ),
            32767,
        )

    def test_get_background_l_mode(self):
        """Test get_background with L mode."""
        self.assertEqual(
            ImgUtils.get_background(
                os.path.join(self.test_dir, "im_Height_8.png"), "L"
            ),
            128,
        )

    def test_get_background_rgb_mode(self):
        """Test get_background with RGB mode."""
        self.assertEqual(
            ImgUtils.get_background(os.path.join(self.test_dir, "im_N.png"), "RGB"),
            (127, 127, 255),
        )

    # -------------------------------------------------------------------------
    # Color Replacement Tests
    # -------------------------------------------------------------------------

    def test_replace_color(self):
        """Test replace_color substitutes colors in image."""
        input_path = os.path.join(self.test_dir, "im_Base_color.png")
        bg = ImgUtils.get_background(input_path, "RGB")
        result = ImgUtils.replace_color(input_path, bg, (255, 0, 0), mode="RGBA")
        self.assertEqual(result.mode, "RGBA")

    def test_replace_color_black_to_white(self):
        """Test replace_color black to white."""
        black_img = ImgUtils.create_image("RGB", (100, 100), (0, 0, 0))
        result = ImgUtils.replace_color(
            black_img, (0, 0, 0), (255, 255, 255), mode="RGBA"
        )
        self.assertEqual(result.mode, "RGBA")

    # -------------------------------------------------------------------------
    # Contrast Tests
    # -------------------------------------------------------------------------

    def test_set_contrast(self):
        """Test set_contrast adjusts image contrast."""
        result = ImgUtils.set_contrast(
            os.path.join(self.test_dir, "im_Mixed_AO_L.png"), 255
        )
        self.assertEqual(result.mode, "L")

    def test_set_contrast_zero(self):
        """Test set_contrast with zero value (no contrast)."""
        result = ImgUtils.set_contrast(
            os.path.join(self.test_dir, "im_Mixed_AO_L.png"), 0
        )
        self.assertEqual(result.mode, "L")

    def test_set_contrast_mid(self):
        """Test set_contrast with mid value."""
        result = ImgUtils.set_contrast(
            os.path.join(self.test_dir, "im_Mixed_AO_L.png"), 128
        )
        self.assertEqual(result.mode, "L")

    # -------------------------------------------------------------------------
    # Color Space Conversion Tests
    # -------------------------------------------------------------------------

    def test_convert_rgb_to_gray(self):
        """Test convert_rgb_to_gray converts to grayscale array."""
        result = ImgUtils.convert_rgb_to_gray(self.im_h)
        self.assertEqual(str(type(result)), "<class 'numpy.ndarray'>")

    def test_convert_rgb_to_hsv(self):
        """Test convert_rgb_to_hsv converts to HSV color space."""
        result = ImgUtils.convert_rgb_to_hsv(self.im_h)
        self.assertEqual(result.mode, "HSV")

    def test_convert_i_to_l(self):
        """Test convert_i_to_l converts I mode to L mode."""
        im_i = ImgUtils.create_image("I", (32, 32))
        result = ImgUtils.convert_i_to_l(im_i)
        self.assertEqual(result.mode, "L")

    def test_convert_i_to_l_scales_16bit(self):
        """16-bit values must be scaled to 8-bit, not truncated modulo 256."""
        arr = np.full((4, 4), 65535, dtype=np.uint16)  # pure white in 16-bit
        result = ImgUtils.convert_i_to_l(Image.fromarray(arr))
        self.assertEqual(result.mode, "L")
        self.assertEqual(result.getpixel((0, 0)), 255)

    def test_convert_f_to_l_rescales_the_unit_range(self):
        """A float map carries 0..1, so Pillow's ``F`` -> ``L`` (a straight
        truncation) collapses it to two values -- everything below 1.0 to 0.
        The unit range is the float convention's full scale, exactly as
        0..65535 is the 16-bit one, so it must map onto the whole 0..255."""
        ramp = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)
        result = ImgUtils.convert_f_to_l(Image.fromarray(ramp, mode="F"))
        self.assertEqual(result.mode, "L")
        self.assertEqual(len(set(result.get_flattened_data())), 256)
        self.assertEqual(min(result.get_flattened_data()), 0)
        self.assertEqual(max(result.get_flattened_data()), 255)

    def test_convert_f_to_l_clamps_out_of_range_hdr(self):
        """Above-unit data is genuine HDR. Clamping it is the honest 8-bit
        answer (and what an LDR delivery map wants); normalizing by the actual
        max instead would be a silent tonemap, and letting it through would
        wrap. Below-zero is clamped for the same reason."""
        hdr = np.linspace(-2.0, 8.0, 256, dtype=np.float32).reshape(16, 16)
        result = ImgUtils.convert_f_to_l(Image.fromarray(hdr, mode="F"))
        data = list(result.get_flattened_data())
        self.assertEqual(min(data), 0)
        self.assertEqual(max(data), 255)

    def test_the_reduction_twins_dispatch_on_dtype(self):
        """Each twin destroys the other's input if it takes the name
        literally: ``convert_f_to_l`` on 8-bit data clamps every value >= 1 to
        white, and ``convert_i_to_l`` on 0..1 floats truncates them to 0/1 --
        both collapsing to two values. They pick the rule from the actual
        dtype, so either entry point is safe."""
        integer = Image.fromarray(
            np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16), mode="L"
        )
        floating = Image.fromarray(
            np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16), mode="F"
        )
        for name in ("convert_f_to_l", "convert_i_to_l"):
            for label, source in (("integer", integer), ("float", floating)):
                with self.subTest(entry_point=name, source=label):
                    result = getattr(ImgUtils, name)(source)
                    data = list(result.get_flattened_data())
                    self.assertEqual(result.mode, "L")
                    self.assertEqual(len(set(data)), 256)
                    self.assertEqual((min(data), max(data)), (0, 255))

    def test_convert_rgb_to_hsv_high_hue(self):
        """Hues above 255 degrees (blues/violets) must not crash or wrap.

        Regression: the old per-pixel implementation stored 0-360 hues in a
        uint8 array (OverflowError on numpy >= 2)."""
        violet = Image.new("RGB", (2, 2), (127, 0, 255))  # hue ~270 degrees
        result = ImgUtils.convert_rgb_to_hsv(violet)
        self.assertEqual(result.mode, "HSV")
        # PIL scales hue to 0-255: 270 degrees -> ~191.
        h = result.getpixel((0, 0))[0]
        self.assertAlmostEqual(h, 270 / 360 * 255, delta=2)

    def test_generate_mipmaps_returns_chain(self):
        """generate_mipmaps returns the full chain down to 1px (regression:
        it used to discard the chain and return only the base level)."""
        chain = ImgUtils.generate_mipmaps(Image.new("RGB", (16, 4)))
        self.assertEqual([im.size for im in chain], [(16, 4), (8, 2), (4, 1)])

    def test_pack_channels_grayscale_to_rgb(self):
        """grayscale_to_rgb replicates a lone R input into G and B (regression:
        the replication branch was unreachable behind the empty-fill branches)."""
        r = Image.new("L", (8, 8), 200)
        out = ImgUtils.pack_channels({"R": r}, grayscale_to_rgb=True)
        self.assertEqual(out.getpixel((0, 0)), (200, 200, 200))

    # -------------------------------------------------------------------------
    # Image Comparison Tests
    # -------------------------------------------------------------------------

    def test_are_identical_different(self):
        """Test are_identical returns False for different images."""
        self.assertFalse(ImgUtils.are_identical(self.im_h, self.im_n))

    def test_are_identical_same(self):
        """Test are_identical returns True for same image."""
        self.assertTrue(ImgUtils.are_identical(self.im_h, self.im_h))

    def test_are_identical_same_content(self):
        """Test are_identical with identical content different objects."""
        img1 = ImgUtils.create_image("RGB", (100, 100), (128, 128, 128))
        img2 = ImgUtils.create_image("RGB", (100, 100), (128, 128, 128))
        self.assertTrue(ImgUtils.are_identical(img1, img2))

    def test_are_identical_different_sizes(self):
        """Test are_identical - WARNING: Only compares overlap region, not full size.

        Note: ImageChops.difference only compares the overlapping region of two images.
        If the overlapping pixels are identical, it returns True even if sizes differ.
        """
        small = ImgUtils.create_image("RGB", (50, 50), (0, 0, 0))
        large = ImgUtils.create_image("RGB", (100, 100), (0, 0, 0))
        # Both are black, so the overlapping 50x50 region is identical
        # This is actual behavior - size difference is NOT detected
        self.assertTrue(ImgUtils.are_identical(small, large))

    def test_are_identical_different_sizes_different_colors(self):
        """Test are_identical with different sizes and colors."""
        small = ImgUtils.create_image("RGB", (50, 50), (255, 0, 0))  # Red
        large = ImgUtils.create_image("RGB", (100, 100), (0, 0, 255))  # Blue
        self.assertFalse(ImgUtils.are_identical(small, large))

    def test_are_identical_different_modes(self):
        """Test are_identical with different modes raises ValueError."""
        rgb = ImgUtils.create_image("RGB", (100, 100), (128, 128, 128))
        gray = ImgUtils.create_image("L", (100, 100), 128)
        # ImageChops.difference raises ValueError when modes don't match
        with self.assertRaises(ValueError):
            ImgUtils.are_identical(rgb, gray)


class TestImgUtilsMemory(unittest.TestCase):
    """Test ImgUtils methods with save=False (in-memory processing)."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        # Create simple test images
        self.size = (64, 64)
        self.rgb_img = Image.new("RGB", self.size, (100, 150, 200))
        self.rgba_img = Image.new("RGBA", self.size, (100, 150, 200, 128))
        self.gray_img = Image.new("L", self.size, 128)

        # Save some to disk for methods that require paths (though most should accept images now)
        self.rgb_path = os.path.join(self.test_dir, "test_rgb.png")
        self.rgb_img.save(self.rgb_path)

        self.rgba_path = os.path.join(self.test_dir, "test_rgba.png")
        self.rgba_img.save(self.rgba_path)

    def test_convert_normal_map_format_dx_to_gl_memory(self):
        """Test convert_normal_map_format returns Image when save=False (DX -> GL)."""
        # Create a mock normal map (OpenGL style)
        # R=128, G=128, B=255 (Flat normal)
        gl_normal = Image.new("RGB", self.size, (128, 128, 255))
        gl_path = os.path.join(self.test_dir, "normal_gl.png")
        gl_normal.save(gl_path)

        result = TextureMapFactory.convert_normal_map_format(
            gl_path, target_format="directx", save=False
        )
        self.assertIsInstance(result, Image.Image)

    def test_convert_normal_map_format_gl_to_dx_memory(self):
        """Test convert_normal_map_format returns Image when save=False (GL -> DX)."""
        dx_normal = Image.new("RGB", self.size, (128, 128, 255))
        dx_path = os.path.join(self.test_dir, "normal_dx.png")
        dx_normal.save(dx_path)

        result = TextureMapFactory.convert_normal_map_format(
            dx_path, target_format="opengl", save=False
        )
        self.assertIsInstance(result, Image.Image)

    def test_convert_bump_to_normal_memory(self):
        """Test convert_bump_to_normal returns Image when save=False."""
        result = TextureMapFactory.convert_bump_to_normal(self.gray_img, save=False)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, "RGB")

    def test_convert_smoothness_to_roughness_memory(self):
        """Test convert_smoothness_to_roughness returns Image when save=False."""
        # Now accepts Image object directly
        result = TextureMapFactory.convert_smoothness_to_roughness(
            self.gray_img, save=False
        )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, "L")
        # 128 inverted is 127
        self.assertEqual(result.getpixel((0, 0)), 127)

    def test_convert_roughness_to_smoothness_memory(self):
        """Test convert_roughness_to_smoothness returns Image when save=False."""
        # Now accepts Image object directly
        result = TextureMapFactory.convert_roughness_to_smoothness(
            self.gray_img, save=False
        )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, "L")

    def test_unpack_specular_gloss_memory(self):
        """Test unpack_specular_gloss returns tuple of Images when save=False."""
        # Specular map usually has gloss in alpha
        spec_gloss = Image.new("RGBA", self.size, (100, 100, 100, 200))
        spec_path = os.path.join(self.test_dir, "spec_gloss.png")
        spec_gloss.save(spec_path)

        spec, gloss = TextureMapFactory.unpack_specular_gloss(spec_path, save=False)
        self.assertIsInstance(spec, Image.Image)
        self.assertIsInstance(gloss, Image.Image)
        self.assertEqual(spec.mode, "RGB")
        self.assertEqual(gloss.mode, "L")

    def test_unpack_orm_texture_memory(self):
        """Test unpack_orm_texture returns tuple of Images when save=False."""
        # ORM: R=AO, G=Roughness, B=Metallic
        orm = Image.new("RGB", self.size, (50, 100, 150))
        orm_path = os.path.join(self.test_dir, "orm.png")
        orm.save(orm_path)

        ao, rough, metal = TextureMapFactory.unpack_orm_texture(orm_path, save=False)
        self.assertIsInstance(ao, Image.Image)
        self.assertIsInstance(rough, Image.Image)
        self.assertIsInstance(metal, Image.Image)

        self.assertEqual(ao.getpixel((0, 0)), 50)
        self.assertEqual(rough.getpixel((0, 0)), 100)
        self.assertEqual(metal.getpixel((0, 0)), 150)

    def test_pack_orm_texture_round_trips_through_unpack(self):
        """pack_orm_texture is the channel-layout SSoT the ORM workflow handler
        delegates to — pin that its output unpacks back to the same channels."""
        ao = Image.new("L", self.size, 50)
        rough = Image.new("L", self.size, 100)
        metal = Image.new("L", self.size, 150)

        orm = TextureMapFactory.pack_orm_texture(ao, rough, metal, save=False)
        self.assertIsInstance(orm, Image.Image)
        self.assertEqual(orm.mode, "RGB")
        self.assertEqual(orm.getpixel((0, 0)), (50, 100, 150))

        ao2, rough2, metal2 = TextureMapFactory.unpack_orm_texture(orm, save=False)
        self.assertEqual(ao2.getpixel((0, 0)), 50)
        self.assertEqual(rough2.getpixel((0, 0)), 100)
        self.assertEqual(metal2.getpixel((0, 0)), 150)

    def test_pack_orm_texture_neutral_fills(self):
        """Absent inputs fill to neutral: white AO (full ambient), black
        roughness/metallic — the same fills the workflow handler warns about."""
        rough = Image.new("L", self.size, 100)

        orm = TextureMapFactory.pack_orm_texture(None, rough, None, save=False)
        self.assertEqual(orm.getpixel((0, 0)), (255, 100, 0))

    def _msao_source(self, name="probe_MSAO.png"):
        """A canonical HDRP mask map: R=Metallic, G=AO, B=Detail, A=Smoothness."""
        path = os.path.join(self.test_dir, name)
        Image.new("RGBA", self.size, (200, 60, 0, 100)).save(path)
        return path

    def test_pack_orm_texture_decomposes_a_packed_source(self):
        """A packed map in ANY slot supplies every channel it carries.

        Regression: the WebXR preview's scene sidecar describes a material whose
        only mask is an MSAO map as ``{"metallic": <MSAO>}`` — the one slot it
        fits. Measured on a production room that produced roughness **0**
        (mirror-smooth) and metallic 0.43 against a true 0.016, because the
        packed RGBA was flattened to luminance for the metallic channel and the
        other two took their fill values. Worse than no repair at all: it
        overwrote a roughly-correct converted ORM with a confidently wrong one.
        """
        orm = TextureMapFactory.pack_orm_texture(
            None, None, self._msao_source(), save=False
        )
        occlusion, roughness, metallic = orm.getpixel((0, 0))
        self.assertEqual(occlusion, 60)  # MSAO's G, not the 255 fill
        self.assertEqual(roughness, 155)  # 255 - smoothness(100), not the 0 fill
        self.assertEqual(metallic, 200)  # MSAO's R, not its luminance

    def test_pack_orm_texture_same_packed_map_in_every_slot_as_distinct_strings(self):
        """The SAME packed map named in all three slots must fill all three.

        Regression: a StingrayPBS material wiring one ORM file node into its
        AO, roughness AND metallic inputs reaches the exporter's sidecar as
        three EQUAL path strings that are three DISTINCT objects (the sidecar
        is a JSON round trip). The resolver matched slots to the packed source
        by identity, so only the first slot was recognised: the other two
        stayed a packed path, were cleared as "carries no channel", and took
        their black fills. Measured on a production delivery: AO intact,
        roughness 0 and metallic 0 at every sampled texel.
        """
        path = os.path.join(self.test_dir, "probe_ORM.png")
        Image.new("RGB", self.size, (50, 100, 150)).save(path)
        # Three equal strings that are NOT the same object, as after JSON.
        ao, rough, metal = (str(path[:1]) + path[1:] for _ in range(3))
        self.assertFalse(ao is rough or ao is metal)

        orm = TextureMapFactory.pack_orm_texture(ao, rough, metal, save=False)
        self.assertEqual(orm.getpixel((0, 0)), (50, 100, 150))

    def test_pack_orm_texture_prefers_a_loose_map_over_the_packed_one(self):
        """Naming both means "use the packed map for what the loose ones don't"."""
        rough = Image.new("L", self.size, 111)

        orm = TextureMapFactory.pack_orm_texture(
            None, rough, self._msao_source(), save=False
        )
        occlusion, roughness, metallic = orm.getpixel((0, 0))
        self.assertEqual(roughness, 111)  # the explicit loose map wins
        self.assertEqual(occlusion, 60)  # still recovered from the MSAO
        self.assertEqual(metallic, 200)

    def test_pack_orm_texture_leaves_loose_sources_untouched(self):
        """The expansion is a no-op for the loose-map case (guards a regression
        in the far more common path)."""
        ao = Image.new("L", self.size, 50)
        rough = Image.new("L", self.size, 100)
        metal = Image.new("L", self.size, 150)

        orm = TextureMapFactory.pack_orm_texture(ao, rough, metal, save=False)
        self.assertEqual(orm.getpixel((0, 0)), (50, 100, 150))

    def test_pack_orm_texture_names_its_output_from_the_packed_path(self):
        """``save=True`` derives the filename from the ORIGINAL argument.

        The expansion replaces a packed path with in-memory channels, so a
        naming pass reading the expanded values would raise "cannot derive
        output directory from Image object" for every packed input.
        """
        out = TextureMapFactory.pack_orm_texture(
            None, None, self._msao_source("named_MSAO.png"), save=True
        )
        self.assertTrue(os.path.isfile(out))
        self.assertIn("named", os.path.basename(out))

    def test_pack_orm_texture_names_a_packed_map_that_carries_no_orm_channel(self):
        """An Albedo+Transparency map fills no ORM slot — say so, don't flatten it.

        Its path is cleared rather than luminance-flattened into a channel it
        has nothing to do with, so when it was the only source the pack fails
        with ``No input images provided`` — which names nothing useful. The
        warning is the only thing that points at the real cause.
        """
        path = os.path.join(self.test_dir, "x_Albedo_Transparency.png")
        Image.new("RGBA", self.size, (10, 20, 30, 40)).save(path)

        # The logger OBJECT, not its name: LoggingMixin builds its logger
        # outside the standard registry, so `getLogger(name)` hands back a
        # different instance and the assertion never sees the record.
        with self.assertLogs(TextureMapFactory.logger, level="WARNING") as captured:
            with self.assertRaises(ValueError):
                TextureMapFactory.pack_orm_texture(None, None, path, save=False)
        self.assertIn("Albedo_Transparency", "\n".join(captured.output))

        # With a loose map alongside it, the pack still succeeds on that map.
        rough = Image.new("L", self.size, 111)
        orm = TextureMapFactory.pack_orm_texture(None, rough, path, save=False)
        self.assertEqual(orm.getpixel((0, 0)), (255, 111, 0))

    def test_pack_orm_texture_passes_an_orm_source_through_unchanged(self):
        """ORM is already the glTF layout, so decomposing and repacking it must
        round-trip exactly — the cheapest path for a caller that has one."""
        path = os.path.join(self.test_dir, "y_ORM.png")
        Image.new("RGB", self.size, (40, 80, 120)).save(path)

        orm = TextureMapFactory.pack_orm_texture(None, None, path, save=False)
        self.assertEqual(orm.getpixel((0, 0)), (40, 80, 120))

    def test_unpack_to_channels_reports_what_a_map_carries(self):
        """The generic front door: canonical map type -> image, and ``{}`` for a
        loose map with nothing to decompose. Reports ``Smoothness`` rather than
        ``Roughness`` — converting is the consumer's call."""
        carried = TextureMapFactory.unpack_to_channels(self._msao_source())
        self.assertEqual(
            sorted(carried), ["Ambient_Occlusion", "Metallic", "Smoothness"]
        )
        self.assertEqual(carried["Metallic"].getpixel((0, 0)), 200)
        self.assertEqual(carried["Ambient_Occlusion"].getpixel((0, 0)), 60)
        self.assertEqual(carried["Smoothness"].getpixel((0, 0)), 100)

        loose = os.path.join(self.test_dir, "probe_Roughness.png")
        Image.new("L", self.size, 100).save(loose)
        self.assertEqual(TextureMapFactory.unpack_to_channels(loose), {})

    def test_unpack_albedo_transparency_memory(self):
        """Test unpack_albedo_transparency returns tuple of Images when save=False."""
        albedo = Image.new("RGBA", self.size, (200, 100, 50, 128))
        albedo_path = os.path.join(self.test_dir, "albedo.png")
        albedo.save(albedo_path)

        base, opacity = TextureMapFactory.unpack_albedo_transparency(
            albedo_path, save=False
        )
        self.assertIsInstance(base, Image.Image)
        self.assertIsInstance(opacity, Image.Image)
        self.assertEqual(base.mode, "RGB")
        self.assertEqual(opacity.mode, "L")
        self.assertEqual(opacity.getpixel((0, 0)), 128)

    def test_unpack_msao_texture_memory(self):
        """Test unpack_msao_texture returns tuple of Images when save=False."""
        # MSAO: R=Metallic, G=AO, B=Detail, A=Smoothness
        msao = Image.new("RGBA", self.size, (200, 50, 0, 100))
        msao_path = os.path.join(self.test_dir, "msao.png")
        msao.save(msao_path)

        metal, ao, smooth = TextureMapFactory.unpack_msao_texture(msao_path, save=False)
        self.assertIsInstance(metal, Image.Image)
        self.assertIsInstance(ao, Image.Image)
        self.assertIsInstance(smooth, Image.Image)

        self.assertEqual(metal.getpixel((0, 0)), 200)
        self.assertEqual(ao.getpixel((0, 0)), 50)
        self.assertEqual(smooth.getpixel((0, 0)), 100)

    def test_unpack_metallic_smoothness_memory(self):
        """Test unpack_metallic_smoothness returns tuple of Images when save=False."""
        # Metallic Smoothness: RGB=Metallic (usually), A=Smoothness
        met_smooth = Image.new("RGBA", self.size, (200, 200, 200, 150))
        path = os.path.join(self.test_dir, "met_smooth.png")
        met_smooth.save(path)

        metal, smooth = TextureMapFactory.unpack_metallic_smoothness(path, save=False)
        self.assertIsInstance(metal, Image.Image)
        self.assertIsInstance(smooth, Image.Image)
        self.assertEqual(smooth.getpixel((0, 0)), 150)


class OpenExrFlagTest(unittest.TestCase):
    """OpenCV reads ``OPENCV_IO_ENABLE_OPENEXR`` once, when cv2 loads: a module
    that imports cv2 at its top without setting it first leaves OpenEXR off
    for the whole process (measured, opencv-python 4.13), and every later EXR
    read or write in the session fails -- a DCC that loaded a photogrammetry
    panel before a lightmap bake could no longer encode its maps.
    Fixed: 2026-10-04."""

    @staticmethod
    def _module_statements(tree):
        """Module-level statements, a top-level ``try``'s body included."""
        for node in tree.body:
            if isinstance(node, ast.Try):
                yield from node.body
            else:
                yield node

    def test_every_top_level_cv2_import_sets_the_exr_flag_first(self):
        import pythontk

        root = Path(pythontk.__file__).parent
        offenders = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            flag = cv2_line = None
            for node in self._module_statements(tree):
                if cv2_line is None and isinstance(node, ast.Import):
                    if any(alias.name == "cv2" for alias in node.names):
                        cv2_line = node.lineno
                if flag is None and "OPENCV_IO_ENABLE_OPENEXR" in ast.unparse(node):
                    flag = node.lineno
            if cv2_line is not None and (flag is None or flag > cv2_line):
                offenders.append(f"{path.relative_to(root).as_posix()}:{cv2_line}")
        self.assertEqual(
            offenders, [], "cv2 imported before OPENCV_IO_ENABLE_OPENEXR is set"
        )


class DilateImageTest(unittest.TestCase):
    """ImgUtils.dilate_image -- texture edge-padding / gutter fill."""

    def test_fills_all_background_from_single_pixel(self):
        img = np.zeros((5, 5, 3), dtype=np.float32)
        img[2, 2] = (1.0, 0.5, 0.25)
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        out = ImgUtils.dilate_image(img, mask)
        # Every pixel is filled with the single source color.
        self.assertTrue(np.allclose(out, (1.0, 0.5, 0.25)))

    def test_explicit_mask_keeps_dark_valid_pixel_as_source(self):
        # Row: [bright-valid, dark-valid, background]. The dark pixel is a
        # legitimate baked texel (e.g. shadow contact) -- it must spread, not
        # be treated as empty. With the default luminance mask it would be
        # background; the explicit coverage mask is the whole point.
        img = np.array(
            [[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32
        )
        mask = np.array([[True, True, False]])
        out = ImgUtils.dilate_image(img, mask)
        np.testing.assert_allclose(out[0, 0], (1, 1, 1))  # untouched
        np.testing.assert_allclose(out[0, 1], (0, 0, 0))  # dark valid preserved
        np.testing.assert_allclose(out[0, 2], (0, 0, 0))  # filled from dark valid

        # Default (luminance) mask instead treats the dark pixel as empty and
        # fills it from the bright neighbor -- demonstrating why callers must
        # pass coverage for baked maps.
        out_default = ImgUtils.dilate_image(img)
        np.testing.assert_allclose(out_default[0, 1], (1, 1, 1))

    def test_iterations_limit_bounds_growth(self):
        img = np.zeros((1, 5, 1), dtype=np.float32)
        img[0, 0, 0] = 1.0
        mask = np.zeros((1, 5), dtype=bool)
        mask[0, 0] = True
        out = ImgUtils.dilate_image(img, mask, iterations=1)
        # One pass fills only the immediate neighbor; the rest stay empty.
        self.assertAlmostEqual(float(out[0, 1, 0]), 1.0)
        self.assertAlmostEqual(float(out[0, 2, 0]), 0.0)

    def test_preserves_shape_dtype_and_2d_input(self):
        img = np.zeros((4, 4), dtype=np.float32)
        img[0, 0] = 0.7
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        out = ImgUtils.dilate_image(img, mask)
        self.assertEqual(out.shape, (4, 4))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.allclose(out, 0.7))

    def test_all_background_is_noop_not_infinite_loop(self):
        img = np.zeros((3, 3, 3), dtype=np.float32)
        mask = np.zeros((3, 3), dtype=bool)
        out = ImgUtils.dilate_image(img, mask)  # must terminate
        self.assertTrue(np.all(out == 0))

    def test_mask_shape_mismatch_raises(self):
        img = np.zeros((3, 3, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            ImgUtils.dilate_image(img, np.zeros((2, 2), dtype=bool))

    def test_return_mask_reports_where_the_ring_stopped(self):
        """``return_mask`` hands back the input mask grown by the passes run."""
        img = np.zeros((8, 8, 3), dtype=np.float32)
        img[3:5, 3:5] = 1.0
        mask = img.any(axis=2)
        out, filled = ImgUtils.dilate_image(img, mask, iterations=1, return_mask=True)
        self.assertEqual(out.shape, img.shape)
        # A 2x2 block grown by one 8-connected ring is the 4x4 around it.
        expected = np.zeros((8, 8), dtype=bool)
        expected[2:6, 2:6] = True
        np.testing.assert_array_equal(filled, expected)
        self.assertTrue((out[filled] == 1.0).all() and (out[~filled] == 0.0).all())
        # Default return shape is unchanged: the image alone.
        self.assertIsInstance(
            ImgUtils.dilate_image(img, mask, iterations=1), np.ndarray
        )


class DenoiseImageTest(unittest.TestCase):
    """ImgUtils.denoise_image -- edge-preserving denoise of a baked (HDR) map.

    Arnold's bake path has no denoiser (RTT ignores imagers), so a lightmap's
    sampling noise shipped as-is: measured on a production floor, 9% per texel
    in the cell after the atlas's own supersample, which read as splotches in
    the WebXR preview. This is the pass that removes it. Added: 2026-09-21
    """

    @staticmethod
    def _noisy(signal, sigma=0.08, seed=7):
        """*signal* (linear, HxWx3) under RELATIVE noise, as a path tracer's."""
        rng = np.random.default_rng(seed)
        return signal * np.exp(rng.normal(0.0, sigma, signal.shape[:2]))[..., None]

    @staticmethod
    def _log_rms(a, b, where=None):
        d = np.log(a.mean(axis=2)) - np.log(b.mean(axis=2))
        return float(
            np.sqrt((d[where] ** 2).mean() if where is not None else (d**2).mean())
        )

    def _ramp_with_a_shadow(self):
        """A light falloff (a ramp in log) with a hard shadow across it."""
        h, w = 64, 64
        ramp = np.exp(np.linspace(0.0, 1.0, w))[None, :].repeat(h, axis=0)
        shadow = np.ones((h, w))
        shadow[:, 40:] = 0.35
        return np.stack([ramp * shadow] * 3, axis=2)

    def test_grain_goes_and_a_falloff_and_a_shadow_edge_stay(self):
        clean = self._ramp_with_a_shadow()
        noisy = self._noisy(clean)
        out = ImgUtils.denoise_image(noisy)
        # Away from the edge, the grain is mostly gone and the falloff is kept
        # (an error against the CLEAN ramp, so a flattened ramp would fail).
        away = np.ones(clean.shape[:2], dtype=bool)
        away[:, 36:44] = False
        self.assertLess(
            self._log_rms(out, clean, away), 0.3 * self._log_rms(noisy, clean, away)
        )

        # The shadow edge. Clear of the windows' reach the step is exact; right
        # at it, a razor-sharp synthetic step keeps most of its height (the
        # guided filter's known softening within a radius of a step -- a
        # baked shadow is soft to begin with).
        def step(left, right):
            return float(np.log(out[:, right].mean() / out[:, left].mean()))

        def truth(left, right):
            return float(np.log(0.35) + (right - left) / 63)

        self.assertLess(abs(step(34, 46) - truth(34, 46)), 0.05)
        self.assertGreater(step(38, 41) / truth(38, 41), 0.8)

    def test_only_the_mask_is_read_or_written(self):
        clean = self._ramp_with_a_shadow()
        noisy = self._noisy(clean)
        mask = np.zeros(noisy.shape[:2], dtype=bool)
        mask[8:56, 8:56] = True
        poisoned = noisy.copy()
        poisoned[~mask] = 1e4  # a gutter the filter must never average in
        out = ImgUtils.denoise_image(poisoned, mask=mask)
        np.testing.assert_array_equal(out[~mask], poisoned[~mask])
        self.assertLess(out[mask].max(), 10.0)

    def test_a_lone_spike_is_clamped_to_its_neighbourhood(self):
        clean = np.ones((32, 32, 3))
        noisy = self._noisy(clean, sigma=0.03)
        noisy[16, 16] *= 50.0  # a firefly
        out = ImgUtils.denoise_image(noisy)
        self.assertLess(out[16, 16].mean(), 1.3)

    def test_scale_and_channel_order_change_nothing(self):
        """Relative noise is filtered in log space: an HDR map 100x brighter
        denoises the same, and BGR (cv2) the same as RGB."""
        clean = self._ramp_with_a_shadow() * np.array([1.0, 0.8, 0.6])
        noisy = self._noisy(clean)
        out = ImgUtils.denoise_image(noisy)
        np.testing.assert_allclose(
            ImgUtils.denoise_image(noisy * 100.0), out * 100.0, rtol=1e-5
        )
        np.testing.assert_allclose(
            ImgUtils.denoise_image(noisy[..., ::-1])[..., ::-1], out, rtol=1e-5
        )

    def test_dtype_and_shape_come_back(self):
        noisy = self._noisy(self._ramp_with_a_shadow()).astype(np.float32)
        out = ImgUtils.denoise_image(noisy)
        self.assertEqual((out.dtype, out.shape), (noisy.dtype, noisy.shape))
        single = ImgUtils.denoise_image(noisy[..., 0])
        self.assertEqual(single.shape, noisy.shape[:2])

    def test_an_8bit_image_rounds_instead_of_truncating(self):
        """Cast straight back to an integer dtype, every denoised texel was
        truncated toward dark: half an LSB low on average, everywhere."""
        noisy = self._noisy(self._ramp_with_a_shadow()) * 90.0
        u8 = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
        out = ImgUtils.denoise_image(u8)
        smooth = ImgUtils.denoise_image(u8.astype(np.float32))
        self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(
            out, np.clip(np.rint(smooth), 0, 255).astype(np.uint8)
        )

    def test_the_cv2_path_and_the_numpy_path_agree(self):
        """cv2 is the fast path (float32 filters), numpy the fallback for a
        Python without it (float64 integral images) -- the same filter, so the
        same answer to float32 precision."""
        from unittest import mock

        clean = self._ramp_with_a_shadow()
        noisy = self._noisy(clean)
        noisy[20, 20] *= 40.0  # a spike, so the median path runs too
        mask = np.ones(noisy.shape[:2], dtype=bool)
        mask[:, 56:] = False
        if ImgUtils._cv2() is None:
            self.skipTest("cv2 unavailable: nothing to compare the fallback with")
        fast = ImgUtils.denoise_image(noisy, mask=mask)
        with mock.patch.object(ImgUtils, "_cv2", staticmethod(lambda: None)):
            slow = ImgUtils.denoise_image(noisy, mask=mask)
        np.testing.assert_allclose(fast, slow, rtol=1e-4)

    def test_a_bad_sample_is_healed_and_poisons_nothing(self):
        """One NaN or inf texel -- a renderer's rare bad sample -- turned the
        WHOLE map NaN: every statistic it touched went with it, 0 * NaN
        included. It now reads as absent and is written from its neighbours."""
        clean = self._ramp_with_a_shadow()
        noisy = self._noisy(clean)
        for bad in (np.nan, np.inf):
            with self.subTest(bad=bad):
                image = noisy.copy()
                image[30, 20] = bad
                out = ImgUtils.denoise_image(image)
                self.assertTrue(np.isfinite(out).all())
                self.assertLess(
                    abs(float(np.log(out[30, 20].mean() / clean[30, 20].mean()))), 0.2
                )

    def test_a_bad_sample_outside_the_mask_stays_outside_it(self):
        """ "Only mask texels are read or written" held for finite gutters
        only: a NaN two texels off an island poisoned the island through the
        zero-weighted window sums."""
        clean = self._ramp_with_a_shadow()
        image = self._noisy(clean)
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[8:56, 8:56] = True
        image[6, 30], image[30, 6] = np.nan, np.inf
        out = ImgUtils.denoise_image(image, mask=mask)
        self.assertTrue(np.isfinite(out[mask]).all())
        self.assertTrue(np.isnan(out[6, 30]).all(), "left for the caller, untouched")
        self.assertTrue(np.isinf(out[30, 6]).all())

    def test_the_two_paths_agree_on_a_bad_sample(self):
        from unittest import mock

        if ImgUtils._cv2() is None:
            self.skipTest("cv2 unavailable: nothing to compare the fallback with")
        image = self._noisy(self._ramp_with_a_shadow())
        image[20, 20], image[40, 10] = np.inf, np.nan
        fast = ImgUtils.denoise_image(image)
        with mock.patch.object(ImgUtils, "_cv2", staticmethod(lambda: None)):
            slow = ImgUtils.denoise_image(image)
        np.testing.assert_allclose(fast, slow, rtol=1e-4)

    def test_a_zero_channel_barely_tints_its_neighbours(self):
        """A channel at exactly zero sat ~14 log units under its lit
        neighbours (floored against the MAP's level) and dragged that
        channel's window means: the texel two over lost 42% of its blue.
        Floored against its own texel's level it drags half as far."""
        image = self._noisy(np.stack([self._ramp_with_a_shadow()[..., 0]] * 3, 2))
        reference = ImgUtils.denoise_image(image)
        image[30, 20, 2] = 0.0
        out = ImgUtils.denoise_image(image)
        ratios = out[30, 21:24, 2] / reference[30, 21:24, 2]
        self.assertGreater(float(ratios.min()), 0.7, ratios)


class ImageFormatCapabilityTest(unittest.TestCase):
    """The per-format capability table is the SSoT for IO routing (read/write/backend)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.img = ImgUtils.create_image("RGB", (8, 8), (128, 64, 32))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --- table integrity -------------------------------------------------
    def test_derived_sets_match_table(self):
        fmts = ImgUtils.image_formats
        self.assertEqual(ImgUtils.recognized, tuple(fmts))
        self.assertEqual(ImgUtils.readable, tuple(e for e, f in fmts.items() if f.read))
        self.assertEqual(
            ImgUtils.writable, tuple(e for e, f in fmts.items() if f.write)
        )

    def test_texture_file_types_alias_preserved(self):
        # Public back-compat name still exists and mirrors the recognized set.
        self.assertIsInstance(ImgUtils.texture_file_types, list)
        self.assertEqual(ImgUtils.texture_file_types, list(ImgUtils.recognized))

    def test_dds_recognized_and_writable_via_pil(self):
        self.assertIn("dds", ImgUtils.recognized)
        self.assertIn("dds", ImgUtils.writable)
        self.assertEqual(ImgUtils.image_formats["dds"].backend, "pil")

    def test_exr_hdr_use_cv2_backend(self):
        self.assertEqual(ImgUtils.image_formats["exr"].backend, "cv2")
        self.assertEqual(ImgUtils.image_formats["hdr"].backend, "cv2")

    # --- write guard -----------------------------------------------------
    def test_save_image_rejects_readonly_format(self):
        """A write=False format must raise a clear ValueError, not a cryptic KeyError."""
        import unittest.mock as mock

        patched = dict(ImgUtils.image_formats)
        patched["rok"] = ImageFormat(True, False, "pil")
        with mock.patch.object(ImgUtils, "image_formats", patched):
            with self.assertRaises(ValueError):
                ImgUtils.save_image(self.img, os.path.join(self.tmp, "x.rok"))

    # --- round trips -----------------------------------------------------
    def test_pil_roundtrip_png(self):
        p = os.path.join(self.tmp, "t.png")
        ImgUtils.save_image(self.img, p)
        self.assertEqual(ImgUtils.load_image(p).size, (8, 8))

    def test_dds_roundtrip(self):
        p = os.path.join(self.tmp, "t.dds")
        ImgUtils.save_image(self.img, p)
        self.assertEqual(ImgUtils.load_image(p).size, (8, 8))

    def test_dds_save_accepts_optimize_kwarg(self):
        # optimize_map forwards optimize=True through save_image; DDS must tolerate it.
        p = os.path.join(self.tmp, "opt.dds")
        ImgUtils.save_image(self.img, p, optimize=True)
        self.assertTrue(os.path.isfile(p))

    @unittest.skipUnless(HAS_CV2, "cv2 required for EXR")
    def test_exr_roundtrip_via_cv2(self):
        p = os.path.join(self.tmp, "t.exr")
        ImgUtils.save_image(self.img, p)
        back = ImgUtils.load_image(p)
        self.assertEqual((back.size, back.mode), ((8, 8), "RGB"))

    @unittest.skipUnless(HAS_CV2, "cv2 required for HDR")
    def test_hdr_roundtrip_via_cv2(self):
        p = os.path.join(self.tmp, "t.hdr")
        ImgUtils.save_image(self.img, p)
        self.assertEqual(ImgUtils.load_image(p).size, (8, 8))

    @unittest.skipUnless(HAS_CV2, "cv2 required for EXR")
    def test_get_images_handles_cv2_format_without_crashing(self):
        """Regression: get_images scans `readable` and an EXR in the dir must
        load via cv2 (pre-fix, load_image used pure PIL and crashed)."""
        ImgUtils.save_image(self.img, os.path.join(self.tmp, "a.png"))
        ImgUtils.save_image(self.img, os.path.join(self.tmp, "b.exr"))
        images = ImgUtils.get_images(self.tmp)
        self.assertEqual(len(images), 2)
        self.assertTrue(all(im.size == (8, 8) for im in images.values()))

    # --- unified save path (optimize_map → save_image) -------------------
    def test_optimize_map_routes_to_dds(self):
        """optimize_map must reach DDS through the unified save_image path."""
        src = os.path.join(self.tmp, "wood_Base_color.png")
        ImgUtils.save_image(self.img, src)
        out = MapOptimizer.optimize_map(src, output_type="dds")
        self.assertTrue(out.lower().endswith(".dds"))
        self.assertEqual(ImgUtils.load_image(out).size, (8, 8))

    @unittest.skipUnless(HAS_CV2, "cv2 required for EXR")
    def test_optimize_map_routes_to_exr_via_cv2(self):
        """optimize_map → EXR previously failed (direct PIL save); now backend-routed."""
        src = os.path.join(self.tmp, "wood_Base_color.png")
        ImgUtils.save_image(self.img, src)
        out = MapOptimizer.optimize_map(src, output_type="exr")
        self.assertTrue(out.lower().endswith(".exr"))
        self.assertTrue(os.path.isfile(out))

    # --- bit depth ------------------------------------------------------
    def test_save_16bit_grayscale_png(self):
        gray = ImgUtils.create_image("L", (8, 8), 128)
        p = os.path.join(self.tmp, "h16.png")
        ImgUtils.save_image(gray, p, bit_depth=16)
        back = Image.open(p)
        self.assertEqual(back.mode, "I;16")
        self.assertEqual(back.getpixel((0, 0)), 128 * 257)  # promoted 8→16

    def test_save_16bit_rgb_png_needs_no_cv2(self):
        """A 16-bit colour PNG comes from the standard library (a data map a
        GPU must not sRGB-decode needs 16 bits on any machine): filter type 0
        on every row, the 8-bit samples promoted, decoded here without OpenCV."""
        import struct
        import zlib

        import numpy as np

        p = os.path.join(self.tmp, "c16.png")
        ImgUtils.save_image(self.img, p, bit_depth=16)
        with open(p, "rb") as fh:
            data = fh.read()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        width, height, depth, colour = struct.unpack(">IIBB", data[16:26])
        self.assertEqual((depth, colour), (16, 2), "16-bit RGB")
        idat, pos = b"", 8
        while pos < len(data):
            length, kind = struct.unpack(">I4s", data[pos : pos + 8])
            if kind == b"IDAT":
                idat += data[pos + 8 : pos + 8 + length]
            pos += 12 + length
        rows = np.frombuffer(zlib.decompress(idat), np.uint8)
        rows = rows.reshape(height, 1 + width * 6)
        self.assertTrue((rows[:, 0] == 0).all(), "filter type 0 on every row")
        samples = rows[:, 1:].copy().view(">u2").reshape(height, width, 3)
        expected = np.asarray(self.img.convert("RGB"), dtype=np.uint16) * 257
        np.testing.assert_array_equal(samples.astype(np.uint16), expected)

        rgba = os.path.join(self.tmp, "c16a.png")
        ImgUtils.save_image(self.img.convert("RGBA"), rgba, bit_depth=16)
        with open(rgba, "rb") as fh:
            head = fh.read(26)
        self.assertEqual(struct.unpack(">IIBB", head[16:26])[2:], (16, 6), "RGBA")

    @unittest.skipUnless(HAS_CV2, "cv2 reads the file back")
    def test_save_16bit_rgb_png_reads_back_through_cv2(self):
        """The standard-library file is a PNG another decoder accepts."""
        import cv2
        import numpy as np

        p = os.path.join(self.tmp, "c16.png")
        ImgUtils.save_image(self.img, p, bit_depth=16)
        arr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        self.assertEqual(arr.dtype.name, "uint16")
        self.assertEqual(arr.shape[2], 3)
        expected = np.asarray(self.img.convert("RGB"), dtype=np.uint16) * 257
        np.testing.assert_array_equal(arr[..., ::-1], expected)

    def test_16bit_unsupported_container_falls_back_to_8bit(self):
        """A 16-bit request on a container that can't hold it must not raise —
        it degrades to 8-bit (with a warning)."""
        gray = ImgUtils.create_image("L", (8, 8), 100)
        p = os.path.join(self.tmp, "x.tga")
        ImgUtils.save_image(gray, p, bit_depth=16)
        self.assertTrue(os.path.isfile(p))
        self.assertEqual(Image.open(p).mode, "L")

    # --- compression ----------------------------------------------------
    def test_dds_dxt5_compression(self):
        p = os.path.join(self.tmp, "c.dds")
        ImgUtils.save_image(self.img, p, compression="DXT5")
        self.assertEqual(ImgUtils.load_image(p).size, (8, 8))

    def test_dds_bc7_without_codec_raises_clearly(self):
        with self.assertRaises(ValueError) as ctx:
            ImgUtils.save_image(
                self.img, os.path.join(self.tmp, "x.dds"), compression="BC7"
            )
        self.assertIn("codec", str(ctx.exception).lower())

    def test_register_dds_codec_is_used_for_block_formats(self):
        calls = {}

        def fake_codec(im, name, compression):
            calls["args"] = (im.size, os.path.basename(name), compression)
            open(name, "wb").close()  # stand in for a real encoder

        original = ImgUtils._dds_codec
        try:
            ImgUtils.register_dds_codec(fake_codec)
            ImgUtils.save_image(
                self.img, os.path.join(self.tmp, "bc7.dds"), compression="BC7"
            )
            self.assertEqual(calls["args"], ((8, 8), "bc7.dds", "BC7"))
        finally:
            ImgUtils._dds_codec = original


class AtlasLayoutTest(unittest.TestCase):
    """ImgUtils.compute_atlas_layout — pure-geometry rect packer for atlasing."""

    @staticmethod
    def _area(rect):
        sx, sy, _, _ = rect
        return sx * sy

    @staticmethod
    def _overlap(a, b):
        ax, ay, aox, aoy = a
        bx, by, box, boy = b
        # rects: [ox, ox+sx) x [oy, oy+sy); overlap iff they intersect on both axes
        eps = 1e-9
        sep_x = aox + ax <= box + eps or box + bx <= aox + eps
        sep_y = aoy + ay <= boy + eps or boy + by <= aoy + eps
        return not (sep_x or sep_y)

    def _assert_tiles_unit_square(self, rects):
        # areas sum to 1 (full coverage) and no two rects overlap
        self.assertAlmostEqual(sum(self._area(r) for r in rects), 1.0, places=6)
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                self.assertFalse(
                    self._overlap(rects[i], rects[j]),
                    f"rects {i} {rects[i]} and {j} {rects[j]} overlap",
                )
        # every rect stays inside [0, 1]^2
        for sx, sy, ox, oy in rects:
            self.assertGreaterEqual(ox, -1e-9)
            self.assertGreaterEqual(oy, -1e-9)
            self.assertLessEqual(ox + sx, 1.0 + 1e-9)
            self.assertLessEqual(oy + sy, 1.0 + 1e-9)

    def test_empty(self):
        self.assertEqual(ImgUtils.compute_atlas_layout([]), [])

    def test_single_is_identity(self):
        self.assertEqual(ImgUtils.compute_atlas_layout([5.0]), [(1.0, 1.0, 0.0, 0.0)])

    def test_equal_weights_tile_and_are_equal_area(self):
        rects = ImgUtils.compute_atlas_layout([1.0] * 4)
        self._assert_tiles_unit_square(rects)
        for r in rects:
            self.assertAlmostEqual(self._area(r), 0.25, places=6)

    def test_area_is_proportional_to_weight(self):
        weights = [4.0, 2.0, 1.0, 1.0]
        rects = ImgUtils.compute_atlas_layout(weights)
        self._assert_tiles_unit_square(rects)
        total = sum(weights)
        for w, r in zip(weights, rects):
            self.assertAlmostEqual(self._area(r), w / total, places=6)

    def test_all_zero_falls_back_to_equal(self):
        rects = ImgUtils.compute_atlas_layout([0.0, 0.0, 0.0])
        self._assert_tiles_unit_square(rects)
        for r in rects:
            self.assertAlmostEqual(self._area(r), 1.0 / 3.0, places=6)

    def test_negative_weights_clamped(self):
        # a negative weight contributes 0 area but still gets a (degenerate) rect
        rects = ImgUtils.compute_atlas_layout([-5.0, 1.0, 1.0])
        self._assert_tiles_unit_square(rects)
        self.assertAlmostEqual(self._area(rects[0]), 0.0, places=6)

    def test_output_order_matches_input(self):
        # ordering is preserved even though packing sorts internally by weight
        rects = ImgUtils.compute_atlas_layout([1.0, 9.0])
        total = 10.0
        self.assertAlmostEqual(self._area(rects[0]), 1.0 / total, places=6)
        self.assertAlmostEqual(self._area(rects[1]), 9.0 / total, places=6)

    def test_zero_weight_shelf_still_tiles(self):
        # fewer positive weights than shelves -> a non-empty shelf with total
        # weight 0 (exercises the divide-by-zero guard). rows=2 forces it.
        rects = ImgUtils.compute_atlas_layout([10.0, 0.0, 0.0], rows=2)
        self._assert_tiles_unit_square(rects)
        self.assertAlmostEqual(self._area(rects[0]), 1.0, places=6)
        self.assertAlmostEqual(self._area(rects[1]), 0.0, places=6)
        self.assertAlmostEqual(self._area(rects[2]), 0.0, places=6)

    def test_single_row_override_tiles(self):
        rects = ImgUtils.compute_atlas_layout([1.0, 2.0, 3.0], rows=1)
        self._assert_tiles_unit_square(rects)
        # one shelf -> every rect is full height
        for sx, sy, ox, oy in rects:
            self.assertAlmostEqual(sy, 1.0, places=6)
            self.assertAlmostEqual(oy, 0.0, places=6)

    def test_many_items_tile_cleanly(self):
        for n in (2, 3, 5, 7, 16, 37):
            with self.subTest(n=n):
                rects = ImgUtils.compute_atlas_layout(list(range(1, n + 1)))
                self.assertEqual(len(rects), n)
                self._assert_tiles_unit_square(rects)

    @staticmethod
    def _worst_aspect(rects):
        return max(
            max(sx, sy) / min(sx, sy)
            for sx, sy, _ox, _oy in rects
            if sx > 1e-9 and sy > 1e-9
        )

    #: The ROOM_ENV production room's own weight spread, recovered from the
    #: cell areas of a delivered atlas (46 objects, ~50:1 range). A synthetic
    #: "one dominant item + N small ones" does NOT stand in for it: the packers
    #: separate by only 2.09 vs 1.91 there, so a test built on one would have
    #: passed under the algorithm it was written to replace.
    ROOM_WEIGHTS = (
        [59.5, 59.3, 59.3, 59.3, 59.3, 59.0, 59.0, 58.9]
        + [18.8, 18.8, 18.7, 18.7, 18.7]
        + [18.6] * 14
        + [18.5, 18.5, 18.5, 18.4, 18.2, 18.2, 18.2]
        + [12.5, 12.5, 12.4, 12.4, 9.8, 9.7]
        + [1.5] * 6
        + [1.4, 1.2]
    )

    def test_a_real_room_spread_keeps_every_cell_near_square(self):
        """A cell's two axes are two independent resolutions, and ``bake_atlas``
        renders each object AT its cell -- so a 2.6:1 cell spends 2.6x of one
        axis' resolution for nothing, whatever the cell's area.

        Measured on this spread: weight-balanced bisection (the predecessor)
        reaches **2.62:1** and squarification **2.02:1**, so the bound below
        fails under bisection rather than passing vacuously.
        """
        rects = ImgUtils.compute_atlas_layout(self.ROOM_WEIGHTS)
        self._assert_tiles_unit_square(rects)
        self.assertLess(self._worst_aspect(rects), 2.25, f"{self._worst_aspect(rects)}")

    def test_the_smallest_cell_keeps_a_usable_short_axis(self):
        """The short axis IS the cell's resolution. On this spread the smallest
        cell goes 27 px -> 32 px in a 1024 atlas -- before the gutter, which
        takes a fixed pixel border and so costs a small cell proportionally
        most."""
        rects = ImgUtils.compute_atlas_layout(self.ROOM_WEIGHTS)
        shortest = min(min(sx, sy) for sx, sy, _ox, _oy in rects) * 1024
        self.assertGreaterEqual(shortest, 30.0, f"{shortest:.1f}px")

    def test_area_stays_exactly_proportional_to_weight(self):
        """The squarification changes each cell's SHAPE, never its share."""
        weights = [7.0, 3.0, 3.0, 1.0, 0.5, 0.25]
        total = sum(weights)
        rects = ImgUtils.compute_atlas_layout(weights)
        for weight, (sx, sy, _ox, _oy) in zip(weights, rects):
            self.assertAlmostEqual(sx * sy, weight / total, places=9)


class AtlasPixelRectsTest(unittest.TestCase):
    """ImgUtils.atlas_pixel_rects — SSoT UV-rect → pixel-rect mapping (with flip)."""

    def test_full_rect_covers_whole_canvas(self):
        ((r0, r1, c0, c1),) = ImgUtils.atlas_pixel_rects([(1.0, 1.0, 0.0, 0.0)], 8)
        self.assertEqual((r0, r1, c0, c1), (0, 8, 0, 8))

    def test_uv_bottom_rect_lands_in_bottom_rows(self):
        # UV oy=0 (bottom half) -> image rows 4..8 (bottom) after the flip.
        ((r0, r1, c0, c1),) = ImgUtils.atlas_pixel_rects([(1.0, 0.5, 0.0, 0.0)], 8)
        self.assertEqual((r0, r1), (4, 8))

    def test_size_tuple_is_width_height(self):
        ((r0, r1, c0, c1),) = ImgUtils.atlas_pixel_rects([(1.0, 1.0, 0.0, 0.0)], (6, 4))
        self.assertEqual((r0, r1, c0, c1), (0, 4, 0, 6))

    def test_shared_edges_have_no_gap_or_overlap(self):
        rects = ImgUtils.compute_atlas_layout([1.0, 2.0, 3.0], rows=1)
        px = ImgUtils.atlas_pixel_rects(rects, 64)
        cols = sorted((c0, c1) for _r0, _r1, c0, c1 in px)
        self.assertEqual(cols[0][1], cols[1][0])  # neighbor boundaries meet
        self.assertEqual(cols[1][1], cols[2][0])  # exactly (same rounding)


class InsetAtlasRectsTest(unittest.TestCase):
    """ImgUtils.inset_atlas_rects — pixel gutter around each atlas rect."""

    def test_inset_frees_gutter_on_all_sides(self):
        ((sx, sy, ox, oy),) = ImgUtils.inset_atlas_rects([(1.0, 1.0, 0.0, 0.0)], 64, 4)
        self.assertAlmostEqual(ox, 4 / 64)
        self.assertAlmostEqual(oy, 4 / 64)
        self.assertAlmostEqual(sx, 1.0 - 8 / 64)
        self.assertAlmostEqual(sy, 1.0 - 8 / 64)

    def test_tiny_rect_is_protected(self):
        # A 4px-wide rect can't afford an 8px gutter: per-axis inset is capped
        # at a quarter of the extent, so content keeps at least half the rect.
        ((sx, _sy, ox, _oy),) = ImgUtils.inset_atlas_rects(
            [(4 / 64, 1.0, 0.0, 0.0)], 64, 8
        )
        self.assertGreaterEqual(sx, (4 / 64) / 2)
        self.assertLess(ox, 4 / 64)  # inset stayed inside the original rect

    def test_inset_rects_remain_inside_originals(self):
        rects = ImgUtils.compute_atlas_layout([1.0, 5.0, 2.0])
        inset = ImgUtils.inset_atlas_rects(rects, 128, 4)
        for (sx, sy, ox, oy), (isx, isy, iox, ioy) in zip(rects, inset):
            self.assertGreaterEqual(iox, ox)
            self.assertGreaterEqual(ioy, oy)
            self.assertLessEqual(iox + isx, ox + sx + 1e-9)
            self.assertLessEqual(ioy + isy, oy + sy + 1e-9)


class SnapAtlasRectsTest(unittest.TestCase):
    """ImgUtils.snap_atlas_rects — published rects agree with written texels."""

    def test_snapped_rect_edges_land_on_texel_grid(self):
        rects = ImgUtils.inset_atlas_rects(
            ImgUtils.compute_atlas_layout([3.0, 1.0, 2.0, 5.0]), 256, 4
        )
        for sx, sy, ox, oy in ImgUtils.snap_atlas_rects(rects, 256):
            for edge in (ox * 256, oy * 256, (ox + sx) * 256, (oy + sy) * 256):
                self.assertAlmostEqual(edge, round(edge), places=6)

    def test_snap_is_idempotent_and_matches_pixel_rects(self):
        # The defect this exists for: assemble_atlas writes at rounded pixel
        # edges while the un-snapped float rect is what got published -- up to
        # half a texel of gutter sampled along every rect edge. Snapping must
        # make atlas_pixel_rects(snapped) == atlas_pixel_rects(original) and
        # a second snap a no-op.
        rects = ImgUtils.inset_atlas_rects(
            ImgUtils.compute_atlas_layout([1.0, 1.0, 1.0, 1.0, 7.0]), 128, 2
        )
        snapped = ImgUtils.snap_atlas_rects(rects, 128)
        self.assertEqual(
            ImgUtils.atlas_pixel_rects(rects, 128),
            ImgUtils.atlas_pixel_rects(snapped, 128),
        )
        self.assertEqual(snapped, ImgUtils.snap_atlas_rects(snapped, 128))

    def test_degenerate_rect_passes_through(self):
        (rect,) = ImgUtils.snap_atlas_rects([(0.0, 0.5, 0.25, 0.25)], 64)
        self.assertEqual(rect, (0.0, 0.5, 0.25, 0.25))


class CropToUvBboxTest(unittest.TestCase):
    """ImgUtils.crop_to_uv_bbox / uv_crop_extent -- a partial island fills its cell."""

    def test_crop_admits_no_edge_extension_texel(self):
        # The production seam: a crop padded a whole texel past the island's
        # high edge, and an edge-extension texel is not this object's lighting
        # (a point just past a wall panel's edge bakes dark). A source whose
        # island region is uniform must crop to island texels ONLY.
        import numpy as np

        img = np.full((24, 24, 3), 9.0, np.float32)  # outside: an extreme value
        img[6:18, 6:18] = 1.0  # island u/v [0.25, 0.75] -> texels 6..18
        cropped, rect, bounds = ImgUtils.crop_to_uv_bbox(
            img, (0.25, 0.25, 0.75, 0.75), [0.5, 0.5, 0.0, 0.0]
        )
        self.assertEqual(float(cropped.max()), 1.0, "an extension texel leaked in")
        self.assertEqual(bounds, (0.25, 0.25, 0.75, 0.75))
        sx, sy, ox, oy = rect  # the cropped region still maps onto the whole cell
        self.assertAlmostEqual(ox + sx * 0.25, 0.0, places=6)
        self.assertAlmostEqual(ox + sx * 0.75, 0.5, places=6)

    def test_a_near_full_or_missing_island_is_not_cropped(self):
        import numpy as np

        img = np.ones((16, 16, 3), np.float32)
        cell = [0.5, 0.5, 0.25, 0.0]
        for bbox in (None, (0.02, 0.03, 0.97, 0.98), (0.4, 0.4, 0.4, 0.9)):
            out, rect, bounds = ImgUtils.crop_to_uv_bbox(img, bbox, cell)
            self.assertIs(out, img, bbox)
            self.assertEqual((rect, bounds), (cell, (0.0, 0.0, 1.0, 1.0)), bbox)

    def test_extent_is_what_the_crop_keeps(self):
        self.assertEqual(ImgUtils.uv_crop_extent(None), (1.0, 1.0))
        self.assertEqual(ImgUtils.uv_crop_extent((0.0, 0.0, 0.9, 0.95)), (1.0, 1.0))
        eu, ev = ImgUtils.uv_crop_extent((0.0, 0.004, 0.332, 0.996))
        self.assertAlmostEqual(eu, 0.332)
        self.assertAlmostEqual(ev, 0.992)


class InsetRectsToTexelCentersTest(unittest.TestCase):
    """ImgUtils.inset_rects_to_texel_centers — edge UVs sample border-texel centers.

    The defect this exists for: a rect edge on a texel BOUNDARY makes every
    bilinear tap along a shared 3D edge split onto the neighboring item's
    gutter texel — up to 50% of the tap's weight reading another item's
    lighting.
    """

    def test_full_span_edges_move_to_border_texel_centers(self):
        # A snapped cell [32..64) x [0..32) px at 128: uv 0/1 must map to the
        # centers of the border texels (32.5 / 63.5), not the boundaries.
        (sx, sy, ox, oy) = ImgUtils.inset_rects_to_texel_centers(
            [(0.25, 0.25, 0.25, 0.0)], 128
        )[0]
        self.assertAlmostEqual(ox * 128, 32.5, places=6)
        self.assertAlmostEqual((ox + sx) * 128, 63.5, places=6)
        self.assertAlmostEqual(oy * 128, 0.5, places=6)
        self.assertAlmostEqual((oy + sy) * 128, 31.5, places=6)

    def test_partial_bbox_maps_island_edges_to_centers(self):
        # Crop-composed production case: island u [0.0013, 0.332] folded so it
        # spans an ~30px cell. The ISLAND edges (not the rect's 0/1) must land
        # on texel centers, and interior mapping stays within half a texel.
        rect = (0.3448275862068966, 0.12890625, 0.48046875, 0.15234375)
        bbox = (0.001292, 0.003876, 0.332041, 0.996124)
        (sx, sy, ox, oy) = ImgUtils.inset_rects_to_texel_centers(
            [rect], 256, bboxes=[bbox]
        )[0]
        for uv, px in ((bbox[0], None), (bbox[2], None)):
            edge = (ox + sx * uv) * 256
            self.assertAlmostEqual(edge % 1.0, 0.5, places=6, msg=f"u={uv} -> {edge}")
        for vv in (bbox[1], bbox[3]):
            edge = (oy + sy * vv) * 256
            self.assertAlmostEqual(edge % 1.0, 0.5, places=6, msg=f"v={vv} -> {edge}")
        # Interior samples shift by less than one texel vs the original rect.
        u_mid = (bbox[0] + bbox[2]) / 2
        self.assertLess(abs((ox + sx * u_mid) - (rect[2] + rect[0] * u_mid)) * 256, 1.0)

    def test_none_bbox_entry_means_full_coverage(self):
        rects = [(0.5, 0.5, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5)]
        with_none = ImgUtils.inset_rects_to_texel_centers(
            rects, 64, bboxes=[None, None]
        )
        without = ImgUtils.inset_rects_to_texel_centers(rects, 64)
        self.assertEqual(with_none, without)

    def test_sub_two_texel_content_passes_through(self):
        # One-texel-wide content has no interior to stretch across.
        (rect,) = ImgUtils.inset_rects_to_texel_centers([(1 / 64, 0.5, 0.0, 0.0)], 64)
        self.assertEqual(rect, (1 / 64, 0.5, 0.0, 0.0))

    def test_exact_boundary_is_float_noise_tolerant(self):
        # Edges arriving a hair past an exact texel boundary must not gain or
        # lose a texel to floor/ceil.
        noisy = (0.25 + 1e-9, 0.25, 0.25 - 1e-9, 0.0)
        clean = (0.25, 0.25, 0.25, 0.0)
        self.assertEqual(
            ImgUtils.inset_rects_to_texel_centers([noisy], 128),
            ImgUtils.inset_rects_to_texel_centers([clean], 128),
        )


class ConvertSceneLinearTest(unittest.TestCase):
    """ImgUtils.convert_scene_linear -- linear light between colour-space primaries.

    The defect this exists for: Arnold renders, and arnoldRenderToTexture
    writes, in the scene's rendering space -- ACEScg by default -- and every
    lightmap consumer reads linear Rec.709. A pure-red sRGB texture baked to
    (0.611, 0.070, 0.021) (measured, mtoa 5.5, 2026-10-02), and a production
    room's lightmap shipped its lighting tint ~30% under-saturated.
    """

    # What Maya's default config makes of a pure-red sRGB texel in ACEScg.
    ACESCG_RED = (0.6131, 0.0702, 0.0206)

    def test_acescg_red_comes_back_rec709_red(self):
        img = np.array([[self.ACESCG_RED]], np.float32)
        out = ImgUtils.convert_scene_linear(img, "ACEScg")
        np.testing.assert_allclose(out[0, 0], (1.0, 0.0, 0.0), atol=1e-3)

    def test_white_is_white_in_every_space(self):
        img = np.ones((2, 2, 3), np.float32)
        for space in (
            "ACEScg",
            "ACES2065-1",
            "scene-linear DCI-P3 D65",
            "scene-linear Rec.2020",
        ):
            np.testing.assert_allclose(
                ImgUtils.convert_scene_linear(img, space), img, atol=1e-5, err_msg=space
            )

    def test_the_same_space_is_untouched(self):
        img = np.random.default_rng(0).random((4, 4, 3)).astype(np.float32)
        out = ImgUtils.convert_scene_linear(img, "scene-linear Rec.709-sRGB")
        np.testing.assert_array_equal(out, img)

    def test_bgr_order_and_alpha_ride_through(self):
        img = np.array([[self.ACESCG_RED[::-1] + (0.25,)]], np.float32)  # BGRA
        out = ImgUtils.convert_scene_linear(img, "ACEScg", bgr=True)
        np.testing.assert_allclose(out[0, 0], (0.0, 0.0, 1.0, 0.25), atol=1e-3)

    def test_an_aces_config_name_resolves_to_the_same_space(self):
        img = np.array([[self.ACESCG_RED]], np.float32)
        a = ImgUtils.convert_scene_linear(img, "ACEScg")
        b = ImgUtils.convert_scene_linear(
            img, "ACES - ACEScg", "Utility - Linear - sRGB"
        )
        np.testing.assert_allclose(a, b, atol=2e-3)
        # Blender 5.1's config (datafiles/colormanagement/config.ocio): its
        # names and aliases for the same spaces. Several raised KeyError.
        for name, space in (
            ("Linear Rec.709", "scene-linear Rec.709-sRGB"),
            ("Linear BT.709", "scene-linear Rec.709-sRGB"),
            ("lin_rec709_srgb", "scene-linear Rec.709-sRGB"),
            ("lin_rec709_scene", "scene-linear Rec.709-sRGB"),
            ("Linear DCI-P3 D65", "scene-linear DCI-P3 D65"),
            ("lin_p3d65_scene", "scene-linear DCI-P3 D65"),
            ("Linear BT.2020", "scene-linear Rec.2020"),
            ("lin_rec2020_scene", "scene-linear Rec.2020"),
            ("Linear ACEScg", "ACEScg"),
            ("lin_ap1_scene", "ACEScg"),
            ("Linear ACES", "ACES2065-1"),
            ("lin_ap0_scene", "ACES2065-1"),
        ):
            with self.subTest(name=name):
                np.testing.assert_array_equal(
                    ImgUtils.convert_scene_linear(img, name),
                    ImgUtils.convert_scene_linear(img, space),
                )

    def test_an_unknown_space_raises(self):
        with self.assertRaises(KeyError):
            ImgUtils.convert_scene_linear(np.ones((1, 1, 3)), "ARRI LogC (v3-EI800)")


class ResizeIntoCellTest(unittest.TestCase):
    """ImgUtils.resize_into_cell -- a tile resampled into the cell it is published in.

    The defect this exists for: atlas tiles were resized EDGE-TO-EDGE into
    their cells (INTER_AREA) while their rects are published with the source
    edges on border-texel CENTERS (:meth:`inset_rects_to_texel_centers`), so
    every edge sample read content from half a texel inside; and the border
    ring a partial coverage left was refilled from the ring inside it. On a
    production room's 18-texel wall cells two coplanar panels met with a 31%
    step under a perfectly smooth light field.
    """

    def _ramp(self, w=64, h=8):
        x = np.arange(w, dtype=np.float32) + 0.5  # texel centres
        return np.tile(x[None, :], (h, 1)), x

    def test_edge_centers_land_source_edges_on_border_texel_centers(self):
        img, _x = self._ramp(64)
        out, cov = ImgUtils.resize_into_cell(img, (9, 4), edge_centers=True)
        self.assertEqual(out.shape, (4, 9))
        step = 64 / 8  # output centres span the source edge to edge
        # Interior texels: a box mean of a ramp is its centre.
        for i in range(1, 8):
            self.assertAlmostEqual(float(out[0, i]), i * step, places=4)
        # Border texels average the half of their footprint inside the source.
        self.assertAlmostEqual(float(out[0, 0]), step / 4, places=4)
        self.assertAlmostEqual(float(out[0, 8]), 64 - step / 4, places=4)
        np.testing.assert_allclose(cov, 1.0)

    def test_without_edge_centers_it_is_a_plain_area_resize(self):
        rng = np.random.default_rng(3)
        img = rng.random((32, 48, 3)).astype(np.float32)
        out, _cov = ImgUtils.resize_into_cell(img, (12, 8), edge_centers=False)
        blocks = img.reshape(8, 4, 12, 4, 3).mean(axis=(1, 3))
        np.testing.assert_allclose(out, blocks, rtol=1e-5, atol=1e-6)

    def test_coverage_weights_out_the_gutter(self):
        # Left half island at 2.0, right half gutter garbage (coverage 0).
        img = np.zeros((8, 8, 3), np.float32)
        img[:, :4] = 2.0
        img[:, 4:] = 50.0
        coverage = np.zeros((8, 8), np.float32)
        coverage[:, :4] = 1.0
        out, cov = ImgUtils.resize_into_cell(
            img, (1, 1), coverage=coverage, edge_centers=False
        )
        np.testing.assert_allclose(out[0, 0], 2.0, rtol=1e-6)
        self.assertAlmostEqual(float(cov[0, 0]), 0.5, places=6)

    def test_uncovered_cell_texels_are_zero_with_zero_coverage(self):
        img = np.full((8, 8), 3.0, np.float32)
        coverage = np.zeros((8, 8), np.float32)
        coverage[:, :2] = 1.0
        out, cov = ImgUtils.resize_into_cell(
            img, (4, 1), coverage=coverage, edge_centers=False
        )
        self.assertEqual(float(cov[0, 3]), 0.0)
        self.assertEqual(float(out[0, 3]), 0.0)
        self.assertAlmostEqual(float(out[0, 0]), 3.0, places=6)

    def test_hdr_peak_does_not_cost_distant_texels_precision(self):
        img = np.ones((1024, 1024), np.float32)
        img[-1, -1] = 65504.0
        out, _cov = ImgUtils.resize_into_cell(img, (64, 64), edge_centers=True)
        np.testing.assert_allclose(out[:32, :32], 1.0, rtol=0, atol=1e-6)

    def test_sub_two_texel_cell_is_never_edge_centred(self):
        # inset_rects_to_texel_centers passes such a rect through unchanged,
        # so the resample must not centre it either (on EITHER axis).
        img, _x = self._ramp(64)
        a, _ = ImgUtils.resize_into_cell(img, (9, 1), edge_centers=True)
        b, _ = ImgUtils.resize_into_cell(img, (9, 1), edge_centers=False)
        np.testing.assert_allclose(a, b)

    def test_two_panels_meet_without_a_step_under_a_smooth_field(self):
        # The production symptom, end to end through the published mapping.
        n, ss, panel = 18, 14, 3.94
        u = (np.arange(n * ss) + 0.5) / (n * ss)

        def field(x):
            return 1.0 + 1.5 * np.exp(-((x - panel - 0.35) ** 2) / (2 * 0.7**2))

        cells = []
        for x0 in (0.0, panel):
            src = np.tile(field(x0 + u * panel)[None, :], (4, 1)).astype(np.float32)
            cells.append(ImgUtils.resize_into_cell(src, (n, 2), edge_centers=True)[0])
        # Published: source edges on border-texel centres -> u=1 / u=0 read the
        # border texels themselves.
        a, b = float(cells[0][0, -1]), float(cells[1][0, 0])
        self.assertLess(abs(a - b) / ((a + b) / 2), 0.06)


class StitchSeamsTest(unittest.TestCase):
    """ImgUtils.stitch_seams -- two cells that meet at a 3D edge read ONE value there.

    The defect this exists for: every wall panel of a production room is its
    own atlas cell with its own sampling noise, so two coplanar panels showed a
    step along every shared edge however well each cell was finished.
    """

    def _two_cells(self):
        img = np.zeros((8, 20, 3), np.float32)
        img[:, 1:9] = 1.0  # cell A, texels 1..8
        img[:, 11:19] = 1.3  # cell B, texels 11..18
        # The shared edge: A's right border-texel centre (x=8) meets B's left (x=11).
        ys = np.linspace(0.0, 7.0, 15)
        pairs = np.stack([np.full_like(ys, 8.0), ys, np.full_like(ys, 11.0), ys], 1)
        return img, pairs

    def test_both_sides_read_one_value(self):
        img, pairs = self._two_cells()
        out = ImgUtils.stitch_seams(img, pairs)
        a = out[:, 8].mean(axis=1)
        b = out[:, 11].mean(axis=1)
        np.testing.assert_allclose(a, b, atol=2e-3)
        # Met halfway: neither side is simply copied onto the other.
        self.assertGreater(float(a.mean()), 1.05)
        self.assertLess(float(a.mean()), 1.25)

    def test_only_the_tapped_texels_move(self):
        img, pairs = self._two_cells()
        out = ImgUtils.stitch_seams(img, pairs)
        untouched = np.ones(img.shape[:2], bool)
        untouched[:, [8, 11]] = False
        np.testing.assert_array_equal(out[untouched], img[untouched])

    def test_a_sample_between_texels_is_matched_as_sampled(self):
        img, _ = self._two_cells()
        img[:, 7] = 0.8  # a ramp inside A
        pairs = np.array([[7.6, 3.0, 11.0, 3.0]])
        out = ImgUtils.stitch_seams(img, pairs)

        def bilinear(a, x, y):
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            fx, fy = x - x0, y - y0
            return (
                a[y0, x0] * (1 - fx) * (1 - fy)
                + a[y0, x0 + 1] * fx * (1 - fy)
                + a[y0 + 1, x0] * (1 - fx) * fy
                + a[y0 + 1, x0 + 1] * fx * fy
            )

        np.testing.assert_allclose(
            bilinear(out, 7.6, 3.0), bilinear(out, 11.0, 3.0), atol=2e-3
        )

    def test_no_pairs_is_a_copy(self):
        img, _ = self._two_cells()
        out = ImgUtils.stitch_seams(img, np.zeros((0, 4)))
        np.testing.assert_array_equal(out, img)
        self.assertIsNot(out, img)

    def test_an_8bit_image_saturates_instead_of_wrapping(self):
        """Solved in float and cast straight back, a tap pushed past 255
        WRAPPED: a bright texel beside dark ones, raised to make its read meet
        the other side's, came back 79 where the solve said 335."""
        img = np.zeros((4, 8), np.uint8)
        img[1, 1] = 250
        img[1, 2] = img[2, 1] = img[2, 2] = 10
        img[1, 5] = 240
        pairs = np.array([[1.5, 1.5, 5.0, 1.0]])  # a 4-tap read against one texel
        out = ImgUtils.stitch_seams(img, pairs)
        solved = ImgUtils.stitch_seams(img.astype(np.float32), pairs)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(int(out[1, 1]), 255)
        np.testing.assert_array_equal(
            out, np.clip(np.rint(solved), 0, 255).astype(np.uint8)
        )


class ExtrapolateFillTest(unittest.TestCase):
    """ImgUtils.extrapolate_fill -- grow a mask by continuing its content's slope.

    The defect this exists for: a lightmap cell's border texels were refilled
    (and its denoise windows truncated) by averaging the texels inside them,
    which reads a light ramp from further in than the edge -- every shared
    panel edge of a production room showed a step.
    """

    def _ramp(self):
        y, x = np.mgrid[0:12, 0:16].astype(np.float32)
        img = np.dstack([1.0 + 0.1 * x + 0.05 * y] * 3)
        mask = np.zeros((12, 16), bool)
        mask[3:9, 4:12] = True
        return img, mask

    def test_a_ramp_continues_exactly(self):
        img, mask = self._ramp()
        seed = np.where(mask[..., None], img, 0.0)
        out, grown = ImgUtils.extrapolate_fill(seed, mask, rings=2)
        new = grown & ~mask
        self.assertTrue(new.any())
        np.testing.assert_allclose(out[new], img[new], rtol=1e-5)
        np.testing.assert_array_equal(out[mask], seed[mask])

    def test_rings_grow_along_the_axes(self):
        _img, mask = self._ramp()
        seed = np.where(mask[..., None], 1.0, 0.0).astype(np.float32)
        _out, one = ImgUtils.extrapolate_fill(seed, mask, rings=1)
        # One ring: the four sides, not the diagonal corners.
        self.assertTrue(one[2, 6] and one[9, 6] and one[5, 3] and one[5, 12])
        self.assertFalse(one[2, 3])
        _out, three = ImgUtils.extrapolate_fill(seed, mask, rings=3)
        self.assertTrue(three[0, 6] and three[5, 1])

    def test_a_steep_drop_is_clamped_positive(self):
        img = np.zeros((1, 6), np.float32)
        img[0, 2:] = [0.2, 2.0, 2.0, 2.0]
        mask = np.zeros((1, 6), bool)
        mask[0, 2:] = True
        out, grown = ImgUtils.extrapolate_fill(img, mask, rings=2)
        self.assertTrue(grown[0, 1] and grown[0, 0])
        self.assertGreater(float(out[0, 1]), 0.0)
        self.assertGreaterEqual(float(out[0, 1]), 0.5 * 0.2 - 1e-6)
        self.assertLessEqual(float(out[0, 1]), 2.0 * 0.2 + 1e-6)

    def test_a_texel_one_deep_is_left_for_the_dilation(self):
        # A one-texel-wide island has no slope to continue.
        img = np.zeros((5, 5), np.float32)
        mask = np.zeros((5, 5), bool)
        mask[:, 2] = True
        img[:, 2] = 1.0
        out, grown = ImgUtils.extrapolate_fill(img, mask, rings=1)
        np.testing.assert_array_equal(grown, mask)
        np.testing.assert_array_equal(out, img)

    def test_never_wraps_at_the_frame(self):
        img = np.zeros((1, 8), np.float32)
        img[0, :3] = [3.0, 2.0, 1.0]
        mask = np.zeros((1, 8), bool)
        mask[0, :3] = True
        out, grown = ImgUtils.extrapolate_fill(img, mask, rings=1)
        self.assertTrue(grown[0, 3])
        self.assertFalse(grown[0, 7], "the far frame edge is not a neighbour")
        self.assertAlmostEqual(float(out[0, 3]), 0.5, places=6)  # clamp: 0.5 * 1.0

    def test_an_8bit_ramp_saturates_instead_of_wrapping(self):
        """Extrapolated in float and cast straight back, an 8-bit ramp running
        past 255 WRAPPED: 60, 130, 200 continued to 270, stored as 14."""
        img = np.zeros((1, 6), np.uint8)
        img[0, :3] = [60, 130, 200]
        mask = np.zeros((1, 6), bool)
        mask[0, :3] = True
        out, grown = ImgUtils.extrapolate_fill(img, mask, rings=1)
        self.assertTrue(grown[0, 3])
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out[0].tolist(), [60, 130, 200, 255, 0, 0])


class FillEmptyTexelsTest(unittest.TestCase):
    """ImgUtils.fill_empty_texels — no background texel survives the fill."""

    def test_every_empty_texel_takes_its_nearest_valid_color(self):
        img = np.zeros((16, 16, 3), dtype=np.float32)
        img[2:6, 2:6] = (1.0, 2.0, 3.0)  # one island far from the far corner
        out = ImgUtils.fill_empty_texels(img)
        self.assertTrue((out > 0).all(), "background texels survived the fill")
        # The far corner's nearest valid texel is the island: same color.
        np.testing.assert_allclose(out[15, 15], (1.0, 2.0, 3.0))
        # Valid texels are untouched.
        np.testing.assert_allclose(out[2:6, 2:6], img[2:6, 2:6])

    def test_two_islands_fill_from_the_nearer_one(self):
        img = np.zeros((8, 32), dtype=np.float32)
        img[:, 0:2] = 1.0
        img[:, 30:32] = 5.0
        mask = img > 0
        out = ImgUtils.fill_empty_texels(img, mask=mask)
        self.assertEqual(float(out[4, 3]), 1.0)  # near the left island
        self.assertEqual(float(out[4, 28]), 5.0)  # near the right island

    def test_all_empty_returns_copy_unchanged(self):
        img = np.zeros((4, 4, 3), dtype=np.float32)
        out = ImgUtils.fill_empty_texels(img)
        self.assertEqual(float(out.sum()), 0.0)

    def test_mask_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ImgUtils.fill_empty_texels(
                np.zeros((4, 4), dtype=np.float32), mask=np.ones((2, 2), dtype=bool)
            )

    # ---- the cv2-less path (Blender's Python ships no cv2) ------------------
    # ``sys.modules[name] = None`` makes ``import cv2`` raise ImportError, which
    # is exactly the condition the fallback exists for.

    def test_without_cv2_every_texel_is_filled_from_nearby_content(self):
        """No background survives, the ring next to an island is exact, and no
        colour is invented -- every filled texel lies within the valid range."""
        import sys
        from unittest import mock

        img = np.zeros((70, 45, 3), dtype=np.float32)
        img[5:20, 5:15] = (1.0, 2.0, 3.0)
        img[50:66, 30:41] = (9.0, 8.0, 7.0)
        mask = img.any(axis=2)
        with mock.patch.dict(sys.modules, {"cv2": None}):
            out = ImgUtils.fill_empty_texels(img, mask=mask)
        self.assertEqual(out.shape, img.shape)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue((out > 0).all(), "background texels survived the fill")
        np.testing.assert_array_equal(out[mask], img[mask])  # valid untouched
        # The texels a bilinear tap can reach take their own island's colour.
        np.testing.assert_allclose(
            out[20:23, 5:15], np.broadcast_to((1.0, 2.0, 3.0), (3, 10, 3))
        )
        np.testing.assert_allclose(
            out[47:50, 30:41], np.broadcast_to((9.0, 8.0, 7.0), (3, 11, 3))
        )
        lo, hi = img[mask].min(axis=0), img[mask].max(axis=0)
        self.assertTrue((out >= lo - 1e-5).all() and (out <= hi + 1e-5).all())

    def test_without_cv2_odd_sizes_2d_and_integer_dtypes_round_trip(self):
        import sys
        from unittest import mock

        img = np.zeros((37, 53), dtype=np.uint8)
        img[10:14, 20:30] = 200
        with mock.patch.dict(sys.modules, {"cv2": None}):
            out = ImgUtils.fill_empty_texels(img)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, img.shape)
        self.assertTrue((out == 200).all())

    def test_without_cv2_matches_the_exact_fill_where_it_matters(self):
        """Against cv2's exact nearest fill, the two agree on every texel within
        the ring of an island; only the far field (mip fodder) may differ."""
        import sys
        from unittest import mock

        rng = np.random.default_rng(7)
        img = np.zeros((96, 96, 3), dtype=np.float32)
        for _ in range(6):
            y, x = rng.integers(0, 80, size=2)
            img[y : y + 12, x : x + 12] = rng.random(3) + 0.1
        mask = img.any(axis=2)
        exact = ImgUtils.fill_empty_texels(img, mask=mask)
        with mock.patch.dict(sys.modules, {"cv2": None}):
            approx = ImgUtils.fill_empty_texels(img, mask=mask)
        _, ring = ImgUtils.dilate_image(
            mask.astype(np.float32), mask, iterations=1, return_mask=True
        )
        near = ring & ~mask
        # One texel out, a neighbour average of a single island IS that island.
        single = np.isclose(approx[near], exact[near], atol=1e-5).all(axis=-1)
        self.assertGreater(float(single.mean()), 0.9)
        self.assertTrue((approx > 0).all())

    """ImgUtils.flip_rect_v — bottom-left UV rect <-> top-down glTF texture space."""

    def test_known_value(self):
        # Bottom-left quarter (UV) is the TOP-left quarter in glTF texture space.
        self.assertEqual(
            ImgUtils.flip_rect_v([0.5, 0.5, 0.0, 0.0]), [0.5, 0.5, 0.0, 0.5]
        )

    def test_identity_is_a_fixed_point(self):
        self.assertEqual(
            ImgUtils.flip_rect_v([1.0, 1.0, 0.0, 0.0]), [1.0, 1.0, 0.0, 0.0]
        )

    def test_involution(self):
        rect = [0.4375, 0.9375, 0.03125, 0.53125]
        double = ImgUtils.flip_rect_v(ImgUtils.flip_rect_v(rect))
        for got, want in zip(double, rect):
            self.assertAlmostEqual(got, want, places=9)


@unittest.skipUnless(HAS_CV2, "cv2 required for assemble_atlas resize")
class AtlasAssembleTest(unittest.TestCase):
    """ImgUtils.assemble_atlas — pack per-item images into one atlas at rects."""

    def test_two_stacked_rects_place_with_uv_flip(self):
        # layout([1,1], rows=2): rect[0] oy=0 (UV bottom), rect[1] oy=0.5 (UV top).
        # The UV-bottom rect must land in the BOTTOM image rows (the vertical flip).
        rects = ImgUtils.compute_atlas_layout([1.0, 1.0], rows=2)
        red = np.zeros((4, 4, 3), np.float32)
        red[..., 0] = 1.0
        green = np.zeros((4, 4, 3), np.float32)
        green[..., 1] = 1.0
        atlas = ImgUtils.assemble_atlas([red, green], rects, 8)
        self.assertEqual(atlas.shape, (8, 8, 3))
        self.assertAlmostEqual(float(atlas[7, 0, 0]), 1.0)  # bottom rows == red
        self.assertAlmostEqual(float(atlas[0, 0, 1]), 1.0)  # top rows == green
        self.assertAlmostEqual(float(atlas[7, 0, 1]), 0.0)  # no green at bottom

    def test_size_tuple_and_dtype_preserved(self):
        rects = ImgUtils.compute_atlas_layout([1.0])
        img = (np.ones((2, 2, 3)) * 0.5).astype(np.float32)
        atlas = ImgUtils.assemble_atlas([img], rects, (6, 4))
        self.assertEqual(atlas.shape, (4, 6, 3))  # (H, W, C) from (width, height)
        self.assertEqual(atlas.dtype, np.float32)

    def test_zero_weight_rect_is_skipped(self):
        # The heavy item fills the whole atlas; the degenerate zero-area rects
        # place nothing, so the background fill is fully overwritten.
        rects = ImgUtils.compute_atlas_layout([10.0, 0.0, 0.0], rows=2)
        imgs = [np.full((2, 2, 3), float(c), np.float32) for c in (1, 2, 3)]
        atlas = ImgUtils.assemble_atlas(imgs, rects, 8, background=-1.0)
        self.assertTrue((atlas == 1.0).all())  # all img0, no background, no img1/2

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ImgUtils.assemble_atlas([np.zeros((2, 2, 3), np.float32)], [], 4)

    def test_an_8bit_atlas_rounds_and_saturates(self):
        """Resized in float and cast straight back, an 8-bit atlas truncated
        each averaged texel toward dark (63.75 -> 63) and wrapped a background
        past the range (256 -> 0)."""
        img = np.zeros((2, 2, 3), np.uint8)
        img[0, 0] = 255  # a 2x2 averaged into one texel: 63.75
        atlas = ImgUtils.assemble_atlas(
            [img], [(0.5, 0.5, 0.0, 0.0)], 2, background=256.0
        )
        self.assertEqual(atlas.dtype, np.uint8)
        self.assertEqual(atlas[..., 0].tolist(), [[255, 255], [64, 255]])

    def test_grayscale_round_trips_2d(self):
        rects = ImgUtils.compute_atlas_layout([1.0])
        img = np.full((2, 2), 0.25, np.float32)
        atlas = ImgUtils.assemble_atlas([img], rects, 4)
        self.assertEqual(atlas.ndim, 2)  # 2D in -> 2D out
        self.assertTrue((atlas == 0.25).all())

    def test_a_rect_overhanging_the_canvas_places_what_fits(self):
        """A rect that rounds past the frame must not cost the whole atlas.

        The destination slice clips at the canvas edge while the resized source
        keeps its full size, so the assignment is a shape mismatch -- and the
        exception kills every OTHER item's placement too. mayatk's own mask
        builder clamps for exactly this reason, so the possibility is real
        upstream; the primitive has to survive it.
        """
        img = np.ones((8, 8, 3), np.float32)
        # Straddling each edge, plus a rect entirely outside (which must not
        # produce a negative slice stop -- that wraps and mismatches again).
        for rect in [
            (0.5, 0.5, -0.25, 0.25),  # off the left
            (0.5, 0.5, 0.75, 0.25),  # off the right
            (0.5, 0.5, 0.25, -0.25),  # off the bottom
            (0.5, 0.5, 0.25, 0.75),  # off the top
            (0.4, 0.4, 1.3, -0.6),  # wholly past two edges
            (1.5, 1.5, -0.25, -0.25),  # larger than the canvas
        ]:
            with self.subTest(rect=rect):
                atlas = ImgUtils.assemble_atlas([img], [rect], 16)
                self.assertEqual(atlas.shape, (16, 16, 3))

    def test_an_overhanging_rect_does_not_lose_its_neighbours(self):
        """One bad rect used to take the whole atlas down with it."""
        a = np.full((8, 8, 3), 1.0, np.float32)
        b = np.full((8, 8, 3), 2.0, np.float32)
        atlas = ImgUtils.assemble_atlas(
            [a, b], [(0.5, 1.0, 0.0, 0.0), (0.5, 1.0, 0.9, 0.0)], 16
        )
        self.assertTrue((atlas[:, :8] == 1.0).all())  # the in-bounds item landed
        self.assertTrue((atlas[:, 8:] == 2.0).any())  # so did the overhanging part


class RasterizeUvTrianglesTest(unittest.TestCase):
    """ImgUtils.rasterize_uv_triangles -- the coverage every lightmap refill reads.

    Vectorized 2026-10-02 (a per-triangle Python fill took 13 s of a 96-tile
    production bake's refill); the rule is unchanged and pinned here.
    """

    @staticmethod
    def _assert_matches_the_reference(rng, trials):
        from pythontk.img_utils._rasterize import _ImgRasterizeInternal as R

        for _ in range(trials):
            tris = rng.random((int(rng.integers(1, 12)), 3, 2)) * 1.2 - 0.1
            dim = int(rng.integers(6, 48))
            pts = np.stack([tris[..., 0] * dim, (1 - tris[..., 1]) * dim], -1)
            ref = np.zeros((dim, dim), np.uint8)
            for tri in pts:
                R._fill_triangle(ref, tri)
            np.testing.assert_array_equal(R._scanline_fill(pts, dim), ref)

    @staticmethod
    def _tall_strips(n):
        """*n* triangles, two to each of side-by-side strips 0.8 of the frame
        tall: their heights sum to 0.8 n frames (a fluted column's unwrap)."""
        xs = np.linspace(0.05, 0.95, n // 2 + 1)
        tris = []
        for a, b in zip(xs[:-1], xs[1:]):
            tris += [[(a, 0.1), (b, 0.1), (b, 0.9)], [(a, 0.1), (b, 0.9), (a, 0.9)]]
        return np.array(tris)

    def test_spans_match_the_per_triangle_centre_rule(self):
        self._assert_matches_the_reference(np.random.default_rng(11), 60)

    def test_bands_and_chunks_cover_what_one_pass_does(self):
        """Spans are built a band of rows and a bounded chunk of triangles at a
        time: bands a row or two tall and chunks of a triangle or two (a
        triangle taller than a chunk on its own) change no texel."""
        from unittest import mock
        from pythontk.img_utils._rasterize import _ImgRasterizeInternal as R

        with mock.patch.object(R, "_SCANLINE_CELLS", 64):
            with mock.patch.object(R, "_SCANLINE_SPANS", 5):
                self._assert_matches_the_reference(np.random.default_rng(23), 40)

    def test_scratch_does_not_grow_with_the_triangles_summed_height(self):
        """Every (triangle, row) span was built at once, so the scratch grew
        with the triangles' SUMMED height: 2048 tall strips at 2048 x 4
        supersample took 2.0 GB (now 0.2). Spans are built a bounded chunk at
        a time -- the budget lowered here, so a small frame needs many."""
        import tracemalloc
        from unittest import mock
        from pythontk.img_utils._rasterize import _ImgRasterizeInternal as R

        def peak(n):
            tris = self._tall_strips(n)
            tracing = tracemalloc.is_tracing()
            if not tracing:
                tracemalloc.start()
            tracemalloc.reset_peak()
            base = tracemalloc.get_traced_memory()[0]
            try:
                ImgUtils.rasterize_uv_triangles(tris, size=128, supersample=2)
                return tracemalloc.get_traced_memory()[1] - base
            finally:
                if not tracing:
                    tracemalloc.stop()

        with mock.patch.object(R, "_SCANLINE_SPANS", 1 << 12):
            few, many = peak(250), peak(2000)
        self.assertLess(many, 1.5 * few, f"{few} B for 250 strips, {many} for 2000")

    def test_a_quad_on_texel_lines_is_fully_covered_inside(self):
        quad = [
            [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75)],
            [(0.25, 0.25), (0.75, 0.75), (0.25, 0.75)],
        ]
        cov = ImgUtils.rasterize_uv_triangles(quad, size=8, supersample=4)
        self.assertTrue((cov[2:6, 2:6] == 255).all())
        self.assertEqual(int(cov[:2].max()), 0)
        self.assertEqual(int(cov[:, 6:].max()), 0)

    def test_degenerate_and_offframe_triangles_cover_nothing(self):
        cov = ImgUtils.rasterize_uv_triangles(
            [[(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)], [(2, 2), (3, 2), (2, 3)]], size=8
        )
        self.assertEqual(int(cov.max()), 0)


class RasterizeSilhouetteTest(unittest.TestCase):
    """ImgUtils.rasterize_silhouette — DCC-agnostic shadow-silhouette rasterizer (numpy only)."""

    # A unit quad on the XY plane (z const) -> two triangles.
    QUAD = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    TRIS = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)

    def test_returns_rgba_uint8(self):
        img = ImgUtils.rasterize_silhouette([(self.QUAD, self.TRIS)], size=64, axis="z")
        self.assertEqual(img.shape, (64, 64, 4))
        self.assertEqual(img.dtype, np.uint8)

    def test_uniform_sharp_coverage(self):
        # blur=0 -> sharp; quad [-1,1] with extent=2.2 maps to ~px 3..60 of 64 (~0.79 coverage).
        a = ImgUtils.rasterize_silhouette(
            [(self.QUAD, self.TRIS)],
            size=64,
            axis="z",
            uniform_alpha=True,
            blur_amount=0,
        )[:, :, 3]
        self.assertGreater((a > 0).mean(), 0.6)
        self.assertLess((a > 0).mean(), 0.95)
        self.assertGreater(a[32, 32], 0)  # centre filled
        self.assertEqual(a[0, 0], 0)  # corner empty (quad < frame)

    def test_contact_falloff_is_a_gradient(self):
        a = ImgUtils.rasterize_silhouette(
            [(self.QUAD, self.TRIS)], size=64, axis="z", blur_amount=0
        )[:, :, 3].astype(float)
        self.assertGreater(
            a[a > 0].std(), 5.0
        )  # non-uniform alpha under the silhouette

    def test_auto_axis_nonempty(self):
        img = ImgUtils.rasterize_silhouette(
            [(self.QUAD, self.TRIS)], size=32, axis="auto"
        )
        self.assertTrue((img[:, :, 3] > 0).any())

    def test_empty_geometry_raises(self):
        with self.assertRaises(ValueError):
            ImgUtils.rasterize_silhouette([], size=16)

    def test_degenerate_triangle_no_crash(self):
        deg = np.zeros((3, 3), dtype=float)
        img = ImgUtils.rasterize_silhouette(
            [(deg, np.array([[0, 1, 2]]))], size=16, uniform_alpha=True
        )
        self.assertEqual(img.shape, (16, 16, 4))

    def test_pil_free_blur_path(self):
        """The default (blurred) path must work without PIL — blendertk's ShadowRig runs under
        Blender's PIL-less Python. Force ``Image=None`` so the numpy Gaussian fallback is exercised
        end-to-end through rasterize_silhouette."""
        import pythontk.img_utils._img_utils as iu

        saved = iu.Image
        iu.Image = None
        try:
            img = ImgUtils.rasterize_silhouette(
                [(self.QUAD, self.TRIS)], size=64, axis="z", blur_amount=1.5
            )
        finally:
            iu.Image = saved
        self.assertEqual(img.shape, (64, 64, 4))
        self.assertTrue((img[:, :, 3] > 0).any())


class GaussianBlurNumpyFallbackTest(unittest.TestCase):
    """ImgUtils._gaussian_blur_array_numpy — pure-numpy PIL-free blur (the Blender/no-PIL path)."""

    def test_2d_blur_smooths_and_preserves_shape_dtype(self):
        a = np.zeros((32, 32), dtype=np.uint8)
        a[16, 16] = 255  # impulse
        out = ImgUtils._gaussian_blur_array_numpy(a, 2.0, None)
        self.assertEqual(out.shape, (32, 32))
        self.assertEqual(out.dtype, np.uint8)
        self.assertLess(int(out[16, 16]), 255)  # spread the impulse
        self.assertGreater(int(out[16, 17]), 0)  # energy bled to neighbours

    def test_rgba_channel_restricted(self):
        rgba = (np.random.RandomState(0).rand(16, 16, 4) * 255).astype(np.uint8)
        out = ImgUtils._gaussian_blur_array_numpy(rgba, 2.0, "A")
        self.assertTrue(np.array_equal(out[:, :, :3], rgba[:, :, :3]))  # RGB untouched
        self.assertFalse(np.array_equal(out[:, :, 3], rgba[:, :, 3]))  # alpha blurred

    def test_matches_pil_path_closely(self):
        a = (np.random.RandomState(1).rand(32, 32) * 255).astype(np.uint8)
        pil = ImgUtils._gaussian_blur_array(a, 1.5, None)  # PIL (present in this env)
        npy = ImgUtils._gaussian_blur_array_numpy(a, 1.5, None)
        self.assertLess(
            float(np.abs(pil.astype(float) - npy.astype(float)).mean()), 6.0
        )


class ValidateImageIntegrityTest(unittest.TestCase):
    """ImgUtils.validate_image_integrity — pure-Python completeness check.

    Motivation: a truncated/partially-synced HDR loads as a null texture in
    Maya's Viewport 2.0 and crashes the IBL path; this gate refuses such files
    before they reach a native loader.
    """

    HDR_HEADER = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 16 +X 16\n"

    def _write(self, suffix, blob):
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        f.write(blob)
        f.close()
        self.addCleanup(lambda: os.path.exists(f.name) and os.remove(f.name))
        return f.name

    def test_missing_file(self):
        ok, why = ImgUtils.validate_image_integrity(r"C:/no/such/file_xyz.hdr")
        self.assertFalse(ok)
        self.assertIn("not found", why)

    def test_empty_file(self):
        ok, why = ImgUtils.validate_image_integrity(self._write(".hdr", b""))
        self.assertFalse(ok)
        self.assertIn("empty", why)

    def test_truncated_hdr(self):
        # Declares 16x16 RLE but provides only the scanline marker, no data.
        path = self._write(".hdr", self.HDR_HEADER + b"\x02\x02\x00\x10")
        ok, why = ImgUtils.validate_image_integrity(path)
        self.assertFalse(ok)
        self.assertIn("truncated", why)

    def test_complete_flat_rgbe_hdr(self):
        # Old/flat RGBE: 16*16*4 bytes of pixel data, no run markers (width<8
        # marker path is skipped, so this exercises the flat fallback).
        blob = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 4\n" + b"\x10" * (
            4 * 4 * 4
        )
        ok, why = ImgUtils.validate_image_integrity(self._write(".hdr", blob))
        self.assertTrue(ok, why)

    def test_exr_bad_magic(self):
        ok, why = ImgUtils.validate_image_integrity(
            self._write(".exr", b"XXXX" + b"0" * 100)
        )
        self.assertFalse(ok)

    def test_exr_valid_magic(self):
        blob = b"\x76\x2f\x31\x01" + b"\x00" * 200
        ok, _ = ImgUtils.validate_image_integrity(self._write(".exr", blob))
        self.assertTrue(ok)

    def test_unknown_extension_is_not_rejected(self):
        ok, _ = ImgUtils.validate_image_integrity(
            self._write(".png", b"\x89PNG" + b"0" * 50)
        )
        self.assertTrue(ok)


class ListImageFilesTest(unittest.TestCase):
    """ImgUtils.list_image_files — the SfM-ingest directory-scan SSoT."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for name in ("b.png", "a.JPG", "notes.txt", "c.tiff"):
            with open(os.path.join(self.tmp, name), "wb") as f:
                f.write(b"x")

    def test_default_exts_sorted_names(self):
        # Case-insensitive ext match, non-images excluded, sorted by name.
        self.assertEqual(
            ImgUtils.list_image_files(self.tmp), ["a.JPG", "b.png", "c.tiff"]
        )

    def test_full_paths(self):
        paths = ImgUtils.list_image_files(self.tmp, full_paths=True)
        self.assertEqual(
            [os.path.basename(p) for p in paths], ["a.JPG", "b.png", "c.tiff"]
        )
        self.assertTrue(all(os.path.isfile(p) for p in paths))

    def test_custom_exts(self):
        self.assertEqual(ImgUtils.list_image_files(self.tmp, exts=(".png",)), ["b.png"])

    def test_exts_accepts_bare_string_and_any_case(self):
        # A bare string must not be tuple-ized into single characters,
        # and caller-supplied extensions match case-insensitively.
        self.assertEqual(ImgUtils.list_image_files(self.tmp, exts=".png"), ["b.png"])
        self.assertEqual(ImgUtils.list_image_files(self.tmp, exts=(".PNG",)), ["b.png"])
        self.assertEqual(ImgUtils.list_image_files(self.tmp, exts=(".jpg",)), ["a.JPG"])


class UniqueDirStemsTest(unittest.TestCase):
    """ImgUtils.unique_dir_stems — collision-proof per-source output names
    (the curator/equalizer key their per-source output dirs by these)."""

    def test_distinct_basenames_pass_through(self):
        from pythontk import ImgUtils

        self.assertEqual(
            ImgUtils.unique_dir_stems(["/x/capA", "/x/capB"]),
            ["capA", "capB"],
        )

    def test_same_basename_gets_parent_qualified(self):
        from pythontk import ImgUtils

        stems = ImgUtils.unique_dir_stems(["/x/capA/images", "/x/capB/images"])
        self.assertEqual(len(set(stems)), 2)
        self.assertEqual(stems, ["capA_images", "capB_images"])

    def test_identical_paths_fall_back_to_index(self):
        from pythontk import ImgUtils

        stems = ImgUtils.unique_dir_stems(["/x/a", "/x/a"])
        self.assertEqual(len(set(stems)), 2)

    def test_order_is_preserved(self):
        from pythontk import ImgUtils

        dirs = ["/p1/images", "/z/solo", "/p2/images"]
        stems = ImgUtils.unique_dir_stems(dirs)
        self.assertEqual(stems[1], "solo")
        self.assertEqual(len(set(stems)), 3)


class BitDepthAndBlurChannelRegressionTest(unittest.TestCase):
    """Regressions for the set_bit_depth canonical-mode remap and the
    gaussian_blur LA-alpha channel indexing (fix_groups_p2 entry 22)."""

    def test_set_bit_depth_normalizes_exotic_modes_without_crashing(self):
        """The old ``{v: k for k, v in bit_depth}`` inversion was order-dependent
        and kept only the LAST mode per bit count (16->'PA', 24->'HSV',
        32->'I;32LS'), so 'F'/'CMYK' raised ValueError on convert() and 'YCbCr'
        silently corrupted. A canonical map must normalize each exotic mode to a
        standard, loadable mode instead. (map_type is unknown so enforce_mode is
        skipped and the exotic input survives to the bit-depth branch.)"""
        for mode, expected in (
            ("F", "RGBA"),
            ("CMYK", "RGBA"),
            ("I", "RGBA"),
            ("YCbCr", "RGB"),
            ("LAB", "RGB"),
            ("HSV", "RGB"),
            ("LA", "I;16"),
        ):
            with self.subTest(mode=mode):
                out = ImgUtils.set_bit_depth(Image.new(mode, (4, 4)), "__unknown__")
                self.assertEqual(out.mode, expected)

    def test_set_bit_depth_leaves_standard_modes_untouched(self):
        for mode in ("RGB", "RGBA", "L", "1", "P"):
            with self.subTest(mode=mode):
                out = ImgUtils.set_bit_depth(Image.new(mode, (4, 4)), "__unknown__")
                self.assertEqual(out.mode, mode)

    def test_gaussian_blur_la_alpha_pil_path(self):
        """gaussian_blur(channel='A') on an LA image must resolve the alpha band
        by its real position (index 1), not the fixed RGBA slot 3 -- which made
        ``idx >= len(bands)`` fire and raise 'Channel A not present'."""
        im = Image.new("LA", (16, 16))
        px = im.load()
        for x in range(16):
            for y in range(16):
                px[x, y] = (100, 0 if (x + y) % 2 == 0 else 255)  # flat L, noisy A
        out = ImgUtils.gaussian_blur(im, radius=2.0, channel="A")
        self.assertEqual(out.mode, "LA")
        in_l, in_a = (np.asarray(b) for b in im.split())
        out_l, out_a = (np.asarray(b) for b in out.split())
        self.assertTrue(np.array_equal(out_l, in_l))  # luminance untouched
        self.assertFalse(np.array_equal(out_a, in_a))  # alpha blurred

    def test_gaussian_blur_la_alpha_numpy_fallback_only_blurs_alpha(self):
        """The pure-numpy LA path used a fixed RGBA index map, so channel='A'
        (idx 3, not < 2 channels) fell through to blurring EVERY channel. It must
        blur only the alpha (index 1) and leave luminance intact."""
        arr = np.zeros((16, 16, 2), dtype=np.float64)
        arr[..., 0] = 0.5  # flat L
        arr[::2, ::2, 1] = 1.0  # noisy A
        out = ImgUtils._gaussian_blur_array_numpy(arr, 2.0, "A")
        self.assertTrue(np.allclose(out[..., 0], arr[..., 0]))  # L untouched
        self.assertFalse(np.allclose(out[..., 1], arr[..., 1]))  # A blurred


class TestKelvinToLinearRgb(unittest.TestCase):
    """Blackbody colour temperature -> linear RGB, for authoring light colour."""

    def _fn(self):
        from pythontk import ImgUtils

        return ImgUtils.kelvin_to_linear_rgb

    def test_warm_is_red_dominant_and_cool_is_balanced(self):
        warm = self._fn()(2700)
        cool = self._fn()(6500)
        self.assertGreater(warm[0], warm[1])
        self.assertGreater(warm[1], warm[2])
        # 6500K is near the white point: all three channels close together.
        self.assertLess(max(cool) - min(cool), 0.1, f"6500K not neutral: {cool}")

    def test_blue_rises_monotonically_with_temperature(self):
        blues = [self._fn()(k)[2] for k in (2000, 3000, 4000, 5000, 6500)]
        self.assertEqual(blues, sorted(blues), f"not monotonic: {blues}")

    def test_normalised_to_peak_one(self):
        for kelvin in (1500, 4000, 10000):
            self.assertAlmostEqual(max(self._fn()(kelvin)), 1.0, places=6)

    def test_unnormalised_keeps_the_fit_magnitudes(self):
        raw = self._fn()(2700, normalize=False)
        self.assertLessEqual(max(raw), 1.0)

    def test_out_of_range_is_clamped_not_raised(self):
        self.assertEqual(self._fn()(0), self._fn()(1000))
        self.assertEqual(self._fn()(100000), self._fn()(40000))

    def test_returns_linear_not_srgb(self):
        """The whole point: a light wants linear, and the fit outputs sRGB.

        At 2700K the sRGB green is ~0.66; linearised it must be markedly
        lower. Handing a light the display-referred value makes it too pale.
        """
        self.assertLess(self._fn()(2700)[1], 0.5)


class WebPWriteTest(unittest.TestCase):
    """WebP's writer defaults are the hazard, not the format.

    Pillow writes WebP at *lossy q80* unless told otherwise, so picking
    ".webp" off a format menu would silently degrade every map. save_image
    inverts that: lossless unless a quality is asked for.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="img_webp_")
        # Noise, not a flat fill — a constant image survives any codec and
        # would make a lossy write look lossless.
        rng = np.random.default_rng(0)
        self.src = Image.fromarray(
            rng.integers(0, 255, (64, 64, 3), dtype=np.uint8), "RGB"
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _path(self, name):
        return os.path.join(self.test_dir, name)

    def _max_error(self, path):
        back = Image.open(path).convert("RGB")
        return int(
            np.abs(np.asarray(self.src, np.int16) - np.asarray(back, np.int16)).max()
        )

    def test_webp_is_writable(self):
        self.assertIn("webp", ImgUtils.writable)
        self.assertIn("webp", ImgUtils.readable)

    def test_defaults_to_lossless(self):
        path = self._path("a.webp")
        ImgUtils.save_image(self.src, path)
        self.assertEqual(self._max_error(path), 0)

    def test_quality_makes_it_lossy(self):
        lossless, lossy = self._path("b.webp"), self._path("c.webp")
        ImgUtils.save_image(self.src, lossless)
        ImgUtils.save_image(self.src, lossy, quality=80)
        self.assertGreater(self._max_error(lossy), 0)
        self.assertLess(os.path.getsize(lossy), os.path.getsize(lossless))

    def test_explicit_kwargs_outrank_the_default(self):
        # A caller who really wants lossless-with-effort keeps control.
        path = self._path("d.webp")
        ImgUtils.save_image(self.src, path, quality=80, lossless=True)
        self.assertEqual(self._max_error(path), 0)

    def test_quality_on_a_lossless_container_is_ignored(self):
        path = self._path("e.png")
        ImgUtils.save_image(self.src, path, quality=10)
        self.assertEqual(self._max_error(path), 0)

    def test_oversize_raises_before_the_encoder_does(self):
        big = Image.new("RGB", (ImgUtils.WEBP_MAX_DIMENSION + 1, 4))
        with self.assertRaises(ValueError) as ctx:
            ImgUtils.save_image(big, self._path("f.webp"))
        self.assertIn(str(ImgUtils.WEBP_MAX_DIMENSION), str(ctx.exception))

    def test_jpeg_defaults_to_444_chroma(self):
        """4:2:0 is what destroys a normal map's X/Y vectors."""
        path = self._path("g.jpg")
        ImgUtils.save_image(self.src, path)
        self.assertEqual(Image.open(path).layer[0][1:3], (1, 1))


class EffectiveModeTest(unittest.TestCase):
    """What a container actually stores, versus what it was handed.

    Invisible until after the write — half of these raise instead of
    degrading — so the table is measured, and the writer and the optimizer's
    prediction both read from it.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="img_mode_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_lossless_containers_hold_everything(self):
        for ext in ("png", "tiff"):
            for mode in ("L", "LA", "P", "RGB", "RGBA", "I;16"):
                self.assertEqual(ImgUtils.effective_mode(mode, ext), mode)

    def test_webp_has_no_grayscale_or_palette(self):
        self.assertEqual(ImgUtils.effective_mode("L", "webp"), "RGB")
        self.assertEqual(ImgUtils.effective_mode("P", "webp"), "RGB")
        self.assertEqual(ImgUtils.effective_mode("LA", "webp"), "RGBA")
        self.assertEqual(ImgUtils.effective_mode("RGB", "webp"), "RGB")

    def test_jpeg_has_no_alpha(self):
        for mode, expected in (("RGBA", "RGB"), ("LA", "L"), ("P", "RGB")):
            self.assertEqual(ImgUtils.effective_mode(mode, "jpg"), expected)
            self.assertEqual(ImgUtils.effective_mode(mode, "jpeg"), expected)

    def test_leading_dot_and_case_are_tolerated(self):
        self.assertEqual(ImgUtils.effective_mode("L", ".WEBP"), "RGB")

    def test_unknown_container_passes_the_mode_through(self):
        self.assertEqual(ImgUtils.effective_mode("L", "xyz"), "L")

    def test_table_matches_what_pillow_actually_writes(self):
        """The table is a measurement; this is the measurement."""
        for ext, fallbacks in ImgUtils._CONTAINER_MODE_FALLBACKS.items():
            if ext in ImgUtils.DELIVERY_FORMATS:
                # Not a Pillow writer: the ktx2 rows are measured against the
                # encoder's staged PNG instead — test_ktx2_encoder.py,
                # ``test_staged_modes_match_the_fallback_table``.
                continue
            for mode in fallbacks:
                path = os.path.join(
                    self.test_dir, f"m_{ext}_{mode.replace(';', '')}.{ext}"
                )
                ImgUtils.save_image(Image.new(mode, (8, 8)), path)
                self.assertEqual(
                    Image.open(path).mode,
                    ImgUtils.effective_mode(mode, ext),
                    f"{mode} -> .{ext}",
                )

    def test_modes_jpeg_used_to_reject_now_save(self):
        """Regression: save_image special-cased only RGBA, so P / LA / I;16
        raised OSError on a .jpg write."""
        for mode in ("P", "LA", "RGBA", "I;16"):
            path = os.path.join(self.test_dir, f"j_{mode.replace(';', '')}.jpg")
            ImgUtils.save_image(Image.new(mode, (16, 16)), path)
            self.assertTrue(os.path.isfile(path), mode)

    def test_high_entropy_jpeg_survives_optimize(self):
        """Regression: q95 + 4:4:4 + optimize=True overflowed Pillow's buffer.

        In optimize mode libjpeg needs one buffer sized for the whole encoded
        image, and Pillow guesses it from the pixel count (``2*w*h`` at quality
        >= 95) on the assumption of 4:2:0 chroma. The JPEG defaults changed to
        q95 at 4:4:4 -- right for a normal map, whose X/Y vectors chroma
        subsampling turns to mush -- and full-resolution chroma on a
        high-frequency map exceeds that guess, so Pillow raised ``OSError:
        broken data stream when writing image file`` instead of growing it.

        ``optimize=True`` is not hypothetical: ``MapOptimizer.optimize_map``
        passes it on every save, so this broke the .jpg path of the texture
        optimizer for exactly the detailed maps it exists to process. Measured
        on random-noise RGB before the fix: 256^2 encoded to 159 KB against a
        131 KB budget and 1024^2 to 2.48 MB against 2 MB -- every size failed.
        """
        import numpy as np

        rng = np.random.default_rng(0)
        for size in (256, 512):
            noise = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            path = os.path.join(self.test_dir, f"noise_{size}.jpg")
            ImgUtils.save_image(Image.fromarray(noise, "RGB"), path, optimize=True)
            self.assertTrue(os.path.isfile(path), f"{size} did not write")
            self.assertGreater(os.path.getsize(path), 0, f"{size} wrote empty")

    def test_rgb_to_gif_quantises_adaptively(self):
        """Regression: the container-mode fallback used the WEB palette.

        GIF is always palettised, so ``_CONTAINER_MODE_FALLBACKS`` coerces
        RGB -> P before the write. A bare ``convert("P")`` takes Pillow's
        default 216-colour WEB palette, where the ``GifImagePlugin`` write this
        fallback replaced quantised ADAPTIVELY. Measured on a gradient:
        49/255 max (19.4 mean) round-trip error against the plugin's 17/2.76 --
        a visible banding regression from a change meant only to pick the mode.
        """
        import numpy as np

        size = 128
        ramp = np.zeros((size, size, 3), np.uint8)
        ramp[..., 0] = np.linspace(0, 255, size, dtype=np.uint8)[None, :]
        ramp[..., 1] = np.linspace(0, 255, size, dtype=np.uint8)[:, None]
        ramp[..., 2] = 128

        path = os.path.join(self.test_dir, "ramp.gif")
        ImgUtils.save_image(Image.fromarray(ramp, "RGB"), path)
        back = np.asarray(Image.open(path).convert("RGB"), np.int16)
        error = np.abs(back - ramp.astype(np.int16))

        # The plugin's own adaptive quantisation scores 17 max / 2.76 mean; the
        # WEB palette scores 49 / 19.4. Bound between them, nearer the good one.
        self.assertLess(int(error.max()), 30, "palette looks non-adaptive")
        self.assertLess(float(error.mean()), 6.0, "palette looks non-adaptive")

    def test_jpeg_buffer_widening_is_not_leaked_globally(self):
        """The MAXBLOCK bump is per-save: it is process-wide state."""
        from PIL import ImageFile

        before = ImageFile.MAXBLOCK
        img = Image.new("RGB", (64, 64), (128, 64, 32))
        ImgUtils.save_image(
            img, os.path.join(self.test_dir, "restore.jpg"), optimize=True
        )
        self.assertEqual(ImageFile.MAXBLOCK, before)


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
                "mipmaps": mipmaps,
                "quality": quality,
                "size": getattr(source, "size", None),
                "mode": getattr(source, "mode", None),
            }
        )
        with open(output, "wb") as fh:
            fh.write(self.MAGIC + codec.encode("ascii"))
        return output


class ImgKtx2RoutingTest(BaseTestCase):
    """save_image's ktx2 delivery route and its capability surfaces."""

    def setUp(self):
        super().setUp()
        self.out_dir = os.path.join(
            os.path.dirname(__file__), "temp_tests", f"imgktx2_{self._testMethodName}"
        )
        os.makedirs(self.out_dir, exist_ok=True)
        self.addCleanup(shutil.rmtree, self.out_dir, ignore_errors=True)
        self.fake = _FakeKtx2Encoder()
        ImgUtils.register_ktx2_encoder(self.fake)
        self.addCleanup(ImgUtils.register_ktx2_encoder, None)

    def test_delivery_formats_stay_out_of_writable(self):
        """The index-persistence contract: ``writable`` must not grow."""
        self.assertIn("ktx2", ImgUtils.DELIVERY_FORMATS)
        self.assertNotIn("ktx2", ImgUtils.writable)
        self.assertNotIn("ktx2", ImgUtils.image_formats)

    def test_effective_mode_flattens_sixteen_bit(self):
        self.assertEqual(ImgUtils.effective_mode("I;16", "ktx2"), "L")
        self.assertEqual(ImgUtils.effective_mode("P", "ktx2"), "RGB")

    def test_save_image_routes_to_registered_encoder(self):
        path = os.path.join(self.out_dir, "wall.ktx2")
        ImgUtils.save_image(
            ImgUtils.create_image("RGB", (8, 8), (200, 30, 40)),
            path,
            compression="ETC1S",
            quality=70,
            colorspace="linear",
        )
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(len(self.fake.calls), 1)
        call = self.fake.calls[0]
        self.assertEqual(call["codec"], "ETC1S")
        self.assertFalse(call["srgb"])
        self.assertTrue(call["mipmaps"])
        self.assertEqual(call["quality"], 70)

    def test_the_rdo_pair_rides_a_uastc_save_only(self):
        """``save_image`` hands the RDO pair to a UASTC encode and to nothing
        else: an ETC1S encode has no RDO stage, and an encoder that models
        neither keyword -- this file's fake -- must still take every save that
        does not use them. Added: 2026-09-19"""
        image = ImgUtils.create_image("RGB", (8, 8))
        ImgUtils.save_image(
            image,
            os.path.join(self.out_dir, "color.ktx2"),
            compression="ETC1S",
            uastc_rdo=1.0,
            uastc_rdo_dictionary=1024,
        )
        self.assertEqual(self.fake.calls[-1]["codec"], "ETC1S")

        seen = {}

        class _RdoAware(_FakeKtx2Encoder):
            def encode(
                self, source, output, uastc_rdo=None, uastc_rdo_dictionary=None, **kw
            ):
                seen.update(rdo=uastc_rdo, rdo_dictionary=uastc_rdo_dictionary)
                return super().encode(source, output, **kw)

        ImgUtils.register_ktx2_encoder(_RdoAware())
        ImgUtils.save_image(
            image,
            os.path.join(self.out_dir, "normal.ktx2"),
            compression="UASTC",
            uastc_rdo=0.75,
            uastc_rdo_dictionary=1024,
        )
        self.assertEqual(seen, {"rdo": 0.75, "rdo_dictionary": 1024})

    def test_save_image_ktx2_defaults_to_uastc_srgb(self):
        """A bare save with no map-type context takes the quality-safe codec."""
        ImgUtils.save_image(
            ImgUtils.create_image("RGB", (8, 8)),
            os.path.join(self.out_dir, "bare.ktx2"),
        )
        call = self.fake.calls[0]
        self.assertEqual(call["codec"], "UASTC")
        self.assertTrue(call["srgb"])

    def test_ktx2_available_reflects_registration(self):
        self.assertTrue(ImgUtils.ktx2_available())

    def test_ktx2_available_is_a_predicate_when_the_resolver_refuses(self):
        """A resolver that raises (no binary, no catalog) means "not
        available" -- the caller asked a question, not for an install."""
        from unittest import mock

        with mock.patch.object(
            ImgUtils, "resolve_ktx2_encoder", side_effect=FileNotFoundError("none")
        ):
            self.assertFalse(ImgUtils.ktx2_available())

    def test_missing_encoder_raises_fix_shaped_error(self):
        from unittest import mock

        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        ImgUtils.register_ktx2_encoder(None)
        with (
            mock.patch.object(Ktx2Encoder, "available", return_value=False),
            mock.patch.object(
                Ktx2Encoder, "resolve_toktx", side_effect=FileNotFoundError("no toktx")
            ),
        ):
            self.assertFalse(ImgUtils.ktx2_available())
            with self.assertRaises(FileNotFoundError) as ctx:
                ImgUtils.save_image(
                    ImgUtils.create_image("RGB", (4, 4)),
                    os.path.join(self.out_dir, "no_encoder.ktx2"),
                )
        self.assertIn("register_ktx2_encoder", str(ctx.exception))

    def test_required_raises_even_if_resolve_toktx_disagrees_with_available(self):
        """`resolve_ktx2_encoder(required=True)` must not fall through to
        `return None` when `available()` is False but `resolve_toktx(required=True)`
        unexpectedly succeeds (e.g. a race between the two discovery calls, or
        a partially-mocked test elsewhere) -- required=True is a promise of a
        usable encoder or a raise, never a silent None."""
        from unittest import mock

        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        ImgUtils.register_ktx2_encoder(None)
        with (
            mock.patch.object(Ktx2Encoder, "available", return_value=False),
            mock.patch.object(
                Ktx2Encoder, "resolve_toktx", return_value=r"C:\fake\toktx.exe"
            ),
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                ImgUtils.resolve_ktx2_encoder(required=True)
        self.assertIn("register_ktx2_encoder", str(ctx.exception))

    def test_auto_install_returns_an_encoder_bound_to_the_installed_binary(self):
        """``auto_install=True`` forwards to ``Ktx2Encoder.resolve_toktx`` --
        the prompt policy included, so a GUI's consent dialog reaches the
        install -- and binds the path it returns, never a second discovery
        pass that could miss a just-written catalog entry."""
        from unittest import mock

        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        ImgUtils.register_ktx2_encoder(None)

        def consent(question):
            return True

        with (
            mock.patch.object(Ktx2Encoder, "available", return_value=False),
            mock.patch.object(
                Ktx2Encoder, "resolve_toktx", return_value=r"C:\managed\toktx.exe"
            ) as resolve,
        ):
            encoder = ImgUtils.resolve_ktx2_encoder(
                required=True, auto_install=True, prompt=consent
            )
        self.assertIsInstance(encoder, Ktx2Encoder)
        self.assertEqual(encoder._toktx, r"C:\managed\toktx.exe")
        resolve.assert_called_once_with(
            required=True, auto_install=True, prompt=consent
        )

    def test_ensure_returns_none_when_an_encoder_is_already_available(self):
        """``None`` is the contract's "nothing was installed" — the value every
        caller branches on to decide whether to report a path. It must not
        offer an install, or a user with toktx already on PATH is asked to
        download it."""
        from unittest import mock

        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        ImgUtils.register_ktx2_encoder(None)
        with mock.patch.object(Ktx2Encoder, "available", return_value=True):
            with mock.patch.object(Ktx2Encoder, "resolve_toktx") as resolve:
                self.assertIsNone(ImgUtils.ensure_ktx2_encoder())
        resolve.assert_not_called()

    def test_ensure_returns_the_path_it_installed(self):
        """The other half of the contract, and the reason it is a path rather
        than a bool: the caller reports WHAT answered."""
        from unittest import mock

        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        ImgUtils.register_ktx2_encoder(None)

        def consent(question):
            return True

        with mock.patch.object(Ktx2Encoder, "available", return_value=False):
            with mock.patch.object(
                Ktx2Encoder, "resolve_toktx", return_value=r"C:\managed\toktx.exe"
            ):
                installed = ImgUtils.ensure_ktx2_encoder(prompt=consent)
        self.assertEqual(installed, r"C:\managed\toktx.exe")

    def test_ensure_raises_the_fix_shaped_error_when_declined(self):
        """Callers surface this verbatim, so it has to keep naming the install.

        Also pins the ``auto_install`` backstop: ``resolve_toktx`` returning a
        falsy path without raising (mocked here; a partial install or a future
        signature change for real) must NOT become a silent "nothing to
        install" -- ``required=True`` promises an encoder or an exception."""
        from unittest import mock

        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        ImgUtils.register_ktx2_encoder(None)
        with mock.patch.object(Ktx2Encoder, "available", return_value=False):
            with mock.patch.object(Ktx2Encoder, "resolve_toktx", return_value=None):
                with self.assertRaises(FileNotFoundError) as ctx:
                    ImgUtils.ensure_ktx2_encoder(prompt=lambda _q: False)
        self.assertIn("register_ktx2_encoder", str(ctx.exception))


class OptionalPILGuardTest(unittest.TestCase):
    """Every module-level ``try: from PIL import …`` must bind ALL of its names.

    Pillow is optional here, and a guard that imports six names but only assigns
    ``Image = None`` in its ``except`` leaves the other five *undefined* — so the
    call site raises ``NameError: name 'ImageOps' is not defined`` instead of taking
    the intended "no Pillow" branch. It also defeats late provisioning: hosts that
    install Pillow *after* pythontk is imported (blendertk's ``ensure_image_deps``
    does exactly this inside Blender, whose bundled Python ships no PIL) re-bind
    these globals by looking for the ``None`` ones — a name that was never created
    is invisible to that repair, so the module stays broken for the whole session.
    That was a live failure: the Blender Material Updater could not pack
    Metallic/Smoothness.

    Static, so it holds for every module in the package — including ones added later
    — and it fails in an environment that *does* have Pillow, where the runtime
    NameError never shows up.
    """

    PKG_ROOT = REPO_ROOT / "pythontk"

    @staticmethod
    def _handler_bound_names(handler):
        """Names assigned anywhere in an ``except`` handler body (chained assignment
        included: ``a = b = c = None`` binds all three)."""
        bound = set()
        for node in ast.walk(handler):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bound.add(target.id)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(node.target, ast.Name):
                    bound.add(node.target.id)
        return bound

    def test_module_level_pil_guards_bind_every_imported_name(self):
        offenders = []
        for path in sorted(self.PKG_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for stmt in tree.body:  # module level only — a function-local import
                if not isinstance(stmt, ast.Try):  # raises at call time by design
                    continue
                imported = set()
                for node in stmt.body:
                    if (
                        isinstance(node, ast.ImportFrom)
                        and (node.module or "").split(".")[0] == "PIL"
                    ):
                        imported |= {a.asname or a.name for a in node.names}
                if not imported:
                    continue
                bound = set()
                for handler in stmt.handlers:
                    bound |= self._handler_bound_names(handler)
                unbound = imported - bound
                if unbound:
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT).as_posix()}:{stmt.lineno} "
                        f"leaves {sorted(unbound)} unbound"
                    )
        self.assertEqual(
            offenders,
            [],
            "guarded PIL imports must assign every name in their except branch:\n"
            + "\n".join(offenders),
        )


class DroppedChannelsByModeTest(unittest.TestCase):
    """`dropped_channels` answers for a target MODE as well as a container.

    A registry-declared mode discards bands exactly the way a container does,
    so the band rule is asked the same question from two directions rather
    than written twice.
    """

    def test_a_narrowing_mode_drops_alpha(self):
        self.assertEqual(ImgUtils.dropped_channels("RGBA", target_mode="RGB"), ("A",))

    def test_a_widening_mode_drops_nothing(self):
        """Widening replicates or unrolls; it never truncates."""
        self.assertEqual(ImgUtils.dropped_channels("L", target_mode="RGB"), ())
        self.assertEqual(ImgUtils.dropped_channels("RGB", target_mode="RGBA"), ())

    def test_an_unchanged_mode_drops_nothing(self):
        self.assertEqual(ImgUtils.dropped_channels("RGBA", target_mode="RGBA"), ())

    def test_the_container_form_is_unchanged(self):
        """The ext question still answers as it did, from the same rule."""
        self.assertEqual(ImgUtils.dropped_channels("RGBA", "jpg"), ("A",))
        self.assertEqual(ImgUtils.dropped_channels("RGBA", "png"), ())


class ChannelsCarryingDataTest(unittest.TestCase):
    """A discarded band is only a LOSS when something was in it."""

    def test_a_varying_band_is_data(self):
        img = ImgUtils.create_image("RGBA", (4, 4), (255, 0, 0, 255))
        img.putpixel((0, 0), (255, 0, 0, 0))
        self.assertEqual(ImgUtils.channels_carrying_data(img, ("A",)), ("A",))

    def test_a_uniform_band_is_not_data(self):
        """A fully opaque alpha costs nothing, and shouting trains the reader
        past the line that matters."""
        img = ImgUtils.create_image("RGBA", (4, 4), (255, 0, 0, 255))
        self.assertEqual(ImgUtils.channels_carrying_data(img, ("A",)), ())

    def test_a_SINGLE_band_image_reports_its_one_band(self):
        """PIL answers one ``(min, max)`` pair for a single-band image rather
        than a tuple of them, so zipping it raw pairs the band with the
        minimum and the range test reads it as absent."""
        img = ImgUtils.create_image("L", (4, 4), 10)
        img.putpixel((0, 0), 200)
        self.assertEqual(ImgUtils.channels_carrying_data(img, ("L",)), ("L",))

    def test_a_SINGLE_band_image_that_does_not_vary_is_quiet(self):
        img = ImgUtils.create_image("L", (4, 4), 10)
        self.assertEqual(ImgUtils.channels_carrying_data(img, ("L",)), ())

    def test_a_band_the_image_does_not_have_is_ignored(self):
        img = ImgUtils.create_image("RGB", (4, 4), (255, 0, 0))
        self.assertEqual(ImgUtils.channels_carrying_data(img, ("A",)), ())


class ComposeRectTest(unittest.TestCase):
    """``ImgUtils.compose_rect`` -- two UV rects folded into one."""

    def test_composed_samples_what_the_two_steps_sampled(self):
        inner = [0.5, 0.25, 0.1, 0.6]
        outer = [0.8, 0.9, 0.05, 0.02]
        both = ImgUtils.compose_rect(outer, inner)
        for u, v in ((0.0, 0.0), (1.0, 1.0), (0.3, 0.7)):
            step = (u * inner[0] + inner[2], v * inner[1] + inner[3])
            want = (step[0] * outer[0] + outer[2], step[1] * outer[1] + outer[3])
            got = (u * both[0] + both[2], v * both[1] + both[3])
            self.assertAlmostEqual(want[0], got[0])
            self.assertAlmostEqual(want[1], got[1])

    def test_no_outer_rect_is_the_identity(self):
        rect = [0.5, 0.25, 0.1, 0.6]
        self.assertEqual(ImgUtils.compose_rect(None, rect), rect)


class ImgUtilsLayoutTest(unittest.TestCase):
    """The facade split by job keeps its surface and its layering."""

    _PKG = Path(__file__).resolve().parents[1] / "pythontk" / "img_utils"

    def _modules(self):
        return [self._PKG / "_img_utils.py"] + sorted(
            p for p in self._PKG.glob("_*.py") if p.name not in ("_img_utils.py",)
        )

    def test_no_module_level_import_of_an_engine(self):
        # A *_utils root sits below core_utils/engines: the texture taxonomy is
        # reached only through deferred imports inside the methods that need it.
        offenders = []
        for path in self._modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and "engines" in (
                    node.module or ""
                ):
                    offenders.append(f"{path.name}: {ast.unparse(node)}")
        self.assertEqual(offenders, [])

    def test_public_methods_stay_on_the_facade(self):
        # The resolver registers flat names (ptk.<method>) from the class's own
        # body, so a public method living only on an internal base drops out.
        for base in ImgUtils.__mro__[1:]:
            if base.__name__.startswith("_Img"):
                on_base = {n for n in vars(base) if not n.startswith("_")}
                self.assertEqual(on_base - set(vars(ImgUtils)), set(), base.__name__)


class DetectNormalMapFormatTest(unittest.TestCase):
    """ImgUtils.detect_normal_map_format -- the handedness read off a map's content.

    It is image analysis, so it lives in img_utils: the UV transfer (geo_utils,
    which ranks below the texture engines) reads it as well as MapFactory.
    """

    @staticmethod
    def _relief(green_down: bool) -> "Image.Image":
        """A hemisphere's tangent-space normals: OpenGL (G ~ +dH/drow), or DirectX."""
        n = 256
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
        r2 = ((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (n / 2) ** 2
        h = np.clip(1.0 - r2, 0.0, 1.0) * 40.0
        x = -np.gradient(h, axis=1)
        y = np.gradient(h, axis=0) * (-1.0 if green_down else 1.0)
        v = np.stack([x, y, np.ones_like(h)], axis=-1)
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        return Image.fromarray(np.round((v * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")

    def test_relief_reads_its_convention_and_a_flat_fill_abstains(self):
        self.assertEqual(
            ImgUtils.detect_normal_map_format(self._relief(False)), "OpenGL"
        )
        self.assertEqual(
            ImgUtils.detect_normal_map_format(self._relief(True)), "DirectX"
        )
        flat = Image.new("RGB", (64, 64), (128, 128, 255))
        self.assertIsNone(ImgUtils.detect_normal_map_format(flat))

    def test_the_map_factory_reads_the_same(self):
        img = self._relief(False)
        self.assertEqual(
            TextureMapFactory.detect_normal_map_format(img),
            ImgUtils.detect_normal_map_format(img),
        )


if __name__ == "__main__":
    unittest.main(exit=False)
