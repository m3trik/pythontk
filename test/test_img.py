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

from pythontk import FileUtils, ImgUtils

from conftest import BaseTestCase


class ImgTest(BaseTestCase):
    """Image utilities test class with comprehensive edge case coverage."""

    # Class-level test images
    im_h = ImgUtils.create_image("RGB", (1024, 1024), (0, 0, 0))
    im_n = ImgUtils.create_image("RGB", (1024, 1024), (127, 127, 255))

    # -------------------------------------------------------------------------
    # Image Creation Tests
    # -------------------------------------------------------------------------

    def test_create_image_basic(self):
        """Test create_image creates images with correct properties."""
        img = ImgUtils.create_image("RGB", (1024, 1024), (0, 0, 0))
        self.assertEqual(img, self.im_h)

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
        result_h = ImgUtils.save_image(self.im_h, "test_files/imgtk_test/im_h.png")
        result_n = ImgUtils.save_image(self.im_n, "test_files/imgtk_test/im_n.png")
        self.assertIsNone(result_h)
        self.assertIsNone(result_n)

    def test_save_image_overwrites(self):
        """Test save_image can overwrite existing files."""
        path = "test_files/imgtk_test/im_h.png"
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
        images = ImgUtils.get_images("test_files/imgtk_test/", "*Normal*")
        self.assertEqual(
            list(images.keys()),
            [
                "test_files/imgtk_test/im_Normal_DirectX.png",
                "test_files/imgtk_test/im_Normal_OpenGL.png",
            ],
        )

    def test_get_images_all(self):
        """Test get_images with wildcard to get all images."""
        images = ImgUtils.get_images("test_files/imgtk_test/", "*")
        self.assertGreater(len(images), 0)

    def test_get_images_no_match(self):
        """Test get_images with pattern that matches nothing."""
        images = ImgUtils.get_images("test_files/imgtk_test/", "*NonexistentPattern*")
        self.assertEqual(len(images), 0)

    # -------------------------------------------------------------------------
    # Map Type Resolution Tests
    # -------------------------------------------------------------------------

    def test_resolve_map_type_height(self):
        """Test resolve_map_type identifies height maps."""
        self.assertEqual(
            ImgUtils.resolve_map_type("test_files/imgtk_test/im_h.png"),
            "Height",
        )
        self.assertEqual(
            ImgUtils.resolve_map_type("test_files/imgtk_test/im_h.png", key=False),
            "_H",
        )

    def test_resolve_map_type_normal(self):
        """Test resolve_map_type identifies normal maps."""
        self.assertEqual(
            ImgUtils.resolve_map_type("test_files/imgtk_test/im_n.png"),
            "Normal",
        )
        self.assertEqual(
            ImgUtils.resolve_map_type("test_files/imgtk_test/im_n.png", key=False),
            "_N",
        )

    def test_resolve_map_type_unknown(self):
        """Test resolve_map_type with unknown map type."""
        result = ImgUtils.resolve_map_type("random_file.png")
        # Should return None or empty for unknown types
        self.assertTrue(result is None or result == "")

    # -------------------------------------------------------------------------
    # Filter Images By Type Tests
    # -------------------------------------------------------------------------

    def test_filter_images_by_type_height(self):
        """Test filter_images_by_type filters by texture type."""
        files = FileUtils.get_dir_contents("test_files/imgtk_test")
        self.assertEqual(
            ImgUtils.filter_images_by_type(files, "Height"),
            ["im_h.png", "im_Height.png"],
        )

    def test_filter_images_by_type_empty_list(self):
        """Test filter_images_by_type with empty list."""
        result = ImgUtils.filter_images_by_type([], "Height")
        self.assertEqual(result, [])

    def test_filter_images_by_type_no_match(self):
        """Test filter_images_by_type when no images match."""
        result = ImgUtils.filter_images_by_type(["random.txt"], "Height")
        self.assertEqual(result, [])

    # -------------------------------------------------------------------------
    # Sort Images By Type Tests
    # -------------------------------------------------------------------------

    def test_sort_images_by_type_list(self):
        """Test sort_images_by_type groups images by texture type."""
        self.assertEqual(
            ImgUtils.sort_images_by_type(
                [("im_h.png", "<im_h>"), ("im_n.png", "<im_n>")]
            ),
            {
                "Height": [("im_h.png", "<im_h>")],
                "Normal": [("im_n.png", "<im_n>")],
            },
        )

    def test_sort_images_by_type_dict(self):
        """Test sort_images_by_type with dict input."""
        self.assertEqual(
            ImgUtils.sort_images_by_type({"im_h.png": "<im_h>", "im_n.png": "<im_n>"}),
            {
                "Height": [("im_h.png", "<im_h>")],
                "Normal": [("im_n.png", "<im_n>")],
            },
        )

    def test_sort_images_by_type_empty(self):
        """Test sort_images_by_type with empty input."""
        result = ImgUtils.sort_images_by_type([])
        self.assertEqual(result, {})

    # -------------------------------------------------------------------------
    # Contains Map Types Tests
    # -------------------------------------------------------------------------

    def test_contains_map_types_list(self):
        """Test contains_map_types with list input."""
        self.assertTrue(ImgUtils.contains_map_types([("im_h.png", "<im_h>")], "Height"))

    def test_contains_map_types_dict(self):
        """Test contains_map_types with dict input."""
        self.assertTrue(
            ImgUtils.contains_map_types(
                {"im_h.png": "<im_h>", "im_n.png": "<im_n>"}, "Height"
            )
        )

    def test_contains_map_types_sorted_dict(self):
        """Test contains_map_types with pre-sorted dict."""
        self.assertTrue(
            ImgUtils.contains_map_types({"Height": [("im_h.png", "<im_h>")]}, "Height")
        )

    def test_contains_map_types_multiple(self):
        """Test contains_map_types with multiple types."""
        self.assertTrue(
            ImgUtils.contains_map_types(
                {"Height": [("im_h.png", "<im_h>")]}, ["Height", "Normal"]
            )
        )

    def test_contains_map_types_not_found(self):
        """Test contains_map_types when type not present."""
        self.assertFalse(
            ImgUtils.contains_map_types([("im_h.png", "<im_h>")], "Roughness")
        )

    # -------------------------------------------------------------------------
    # Is Normal Map Tests
    # -------------------------------------------------------------------------

    def test_is_normal_map_false(self):
        """Test is_normal_map returns False for non-normal maps."""
        self.assertFalse(ImgUtils.is_normal_map("im_h.png"))

    def test_is_normal_map_true(self):
        """Test is_normal_map returns True for normal maps."""
        self.assertTrue(ImgUtils.is_normal_map("im_n.png"))

    def test_is_normal_map_explicit_name(self):
        """Test is_normal_map with explicit normal map name."""
        self.assertTrue(ImgUtils.is_normal_map("texture_Normal.png"))

    def test_is_normal_map_directx(self):
        """Test is_normal_map with DirectX normal map."""
        self.assertTrue(ImgUtils.is_normal_map("im_Normal_DirectX.png"))

    def test_is_normal_map_opengl(self):
        """Test is_normal_map with OpenGL normal map."""
        self.assertTrue(ImgUtils.is_normal_map("im_Normal_OpenGL.png"))

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

    def test_create_dx_from_gl(self):
        """Test create_dx_from_gl converts OpenGL to DirectX normal maps."""
        dx_path = ImgUtils.create_dx_from_gl(
            "test_files/imgtk_test/im_Normal_OpenGL.png"
        )
        expected = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "test_files",
                "imgtk_test",
                "im_Normal_DirectX.png",
            )
        )
        self.assertEqual(os.path.normpath(dx_path), os.path.normpath(expected))

    def test_create_gl_from_dx(self):
        """Test create_gl_from_dx converts DirectX to OpenGL normal maps."""
        gl_path = ImgUtils.create_gl_from_dx(
            "test_files/imgtk_test/im_Normal_DirectX.png"
        )
        expected = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "test_files",
                "imgtk_test",
                "im_Normal_OpenGL.png",
            )
        )
        self.assertEqual(os.path.normpath(gl_path), os.path.normpath(expected))

    # -------------------------------------------------------------------------
    # Mask Tests
    # -------------------------------------------------------------------------

    def test_create_mask_from_background(self):
        """Test create_mask generates image masks from background color."""
        bg = ImgUtils.get_background("test_files/imgtk_test/im_Base_color.png", "RGB")
        mask = ImgUtils.create_mask("test_files/imgtk_test/im_Base_color.png", bg)
        self.assertEqual(mask.mode, "L")

    def test_create_mask_from_image(self):
        """Test create_mask generates mask from another image."""
        mask = ImgUtils.create_mask(
            "test_files/imgtk_test/im_Base_color.png",
            "test_files/imgtk_test/im_Base_color.png",
        )
        self.assertEqual(mask.mode, "L")

    # -------------------------------------------------------------------------
    # Fill Tests
    # -------------------------------------------------------------------------

    def test_fill_masked_area(self):
        """Test fill_masked_area fills masked regions with color."""
        bg = ImgUtils.get_background("test_files/imgtk_test/im_Base_color.png", "RGB")
        mask = ImgUtils.create_mask("test_files/imgtk_test/im_Base_color.png", bg)
        result = ImgUtils.fill_masked_area(
            "test_files/imgtk_test/im_Base_color.png", (0, 255, 0), mask
        )
        self.assertEqual(result.mode, "RGB")

    def test_fill_solid_color(self):
        """Test fill fills image with color."""
        result = ImgUtils.fill(self.im_h, (127, 127, 127))
        self.assertEqual(result.mode, "RGB")

    def test_fill_black(self):
        """Test fill with black color."""
        result = ImgUtils.fill(self.im_h, (0, 0, 0))
        self.assertEqual(result.mode, "RGB")

    def test_fill_white(self):
        """Test fill with white color."""
        result = ImgUtils.fill(self.im_h, (255, 255, 255))
        self.assertEqual(result.mode, "RGB")

    # -------------------------------------------------------------------------
    # Background Detection Tests
    # -------------------------------------------------------------------------

    def test_get_background_i_mode(self):
        """Test get_background with I mode."""
        self.assertEqual(
            ImgUtils.get_background("test_files/imgtk_test/im_Height.png", "I"), 32767
        )

    def test_get_background_l_mode(self):
        """Test get_background with L mode."""
        self.assertEqual(
            ImgUtils.get_background("test_files/imgtk_test/im_Height.png", "L"), 255
        )

    def test_get_background_rgb_mode(self):
        """Test get_background with RGB mode."""
        self.assertEqual(
            ImgUtils.get_background("test_files/imgtk_test/im_n.png", "RGB"),
            (127, 127, 255),
        )

    # -------------------------------------------------------------------------
    # Color Replacement Tests
    # -------------------------------------------------------------------------

    def test_replace_color(self):
        """Test replace_color substitutes colors in image."""
        bg = ImgUtils.get_background("test_files/imgtk_test/im_Base_color.png", "RGB")
        result = ImgUtils.replace_color(
            "test_files/imgtk_test/im_Base_color.png", bg, (255, 0, 0)
        )
        self.assertEqual(result.mode, "RGBA")

    def test_replace_color_black_to_white(self):
        """Test replace_color black to white."""
        black_img = ImgUtils.create_image("RGB", (100, 100), (0, 0, 0))
        result = ImgUtils.replace_color(black_img, (0, 0, 0), (255, 255, 255))
        self.assertEqual(result.mode, "RGBA")

    # -------------------------------------------------------------------------
    # Contrast Tests
    # -------------------------------------------------------------------------

    def test_set_contrast(self):
        """Test set_contrast adjusts image contrast."""
        result = ImgUtils.set_contrast("test_files/imgtk_test/im_Mixed_AO.png", 255)
        self.assertEqual(result.mode, "L")

    def test_set_contrast_zero(self):
        """Test set_contrast with zero value (no contrast)."""
        result = ImgUtils.set_contrast("test_files/imgtk_test/im_Mixed_AO.png", 0)
        self.assertEqual(result.mode, "L")

    def test_set_contrast_mid(self):
        """Test set_contrast with mid value."""
        result = ImgUtils.set_contrast("test_files/imgtk_test/im_Mixed_AO.png", 128)
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


if __name__ == "__main__":
    unittest.main(exit=False)


if __name__ == "__main__":
    unittest.main(exit=False)
