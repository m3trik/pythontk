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
import os
import unittest
import shutil
import tempfile
import numpy as np
from PIL import Image

from pythontk import FileUtils, ImgUtils, MapFactory as TextureMapFactory

from conftest import BaseTestCase


class ImgTest(BaseTestCase):
    """Image utilities test class with comprehensive edge case coverage."""

    # Class-level test images
    im_h = ImgUtils.create_image("RGB", (1024, 1024), (0, 0, 0))
    im_n = ImgUtils.create_image("RGB", (1024, 1024), (127, 127, 255))

    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(
            os.path.dirname(__file__), "test_files", "imgtk_test"
        )
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        os.makedirs(cls.test_dir)

        # Create base images
        cls.im_h.save(os.path.join(cls.test_dir, "im_H.png"))
        cls.im_n.save(os.path.join(cls.test_dir, "im_N.png"))

        # Create other expected images
        ImgUtils.create_image("RGB", (1024, 1024), (255, 0, 0)).save(
            os.path.join(cls.test_dir, "im_Base_color.png")
        )
        # Create 16-bit height map
        ImgUtils.create_image("I", (1024, 1024), 32767).save(
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


if __name__ == "__main__":
    unittest.main(exit=False)


if __name__ == "__main__":
    unittest.main(exit=False)
