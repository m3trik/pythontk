#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk ImgUtils.

Run with:
    python -m pytest test_img.py -v
    python test_img.py
"""
import os
import unittest

from pythontk import FileUtils, ImgUtils

from conftest import BaseTestCase


class ImgTest(BaseTestCase):
    """Image utilities test class."""

    # Class-level test images
    im_h = ImgUtils.create_image("RGB", (1024, 1024), (0, 0, 0))
    im_n = ImgUtils.create_image("RGB", (1024, 1024), (127, 127, 255))

    def test_create_image(self):
        """Test create_image creates images with correct properties."""
        img = ImgUtils.create_image("RGB", (1024, 1024), (0, 0, 0))
        self.assertEqual(img, self.im_h)

    def test_resize_image(self):
        """Test resize_image changes image dimensions."""
        resized = ImgUtils.resize_image(self.im_h, 32, 32)
        self.assertEqual(resized.size, (32, 32))

    def test_save_image_file(self):
        """Test save_image writes image to disk."""
        result_h = ImgUtils.save_image(self.im_h, "test_files/imgtk_test/im_h.png")
        result_n = ImgUtils.save_image(self.im_n, "test_files/imgtk_test/im_n.png")
        self.assertIsNone(result_h)
        self.assertIsNone(result_n)

    def test_get_images(self):
        """Test get_images finds images by pattern."""
        images = ImgUtils.get_images("test_files/imgtk_test/", "*Normal*")
        self.assertEqual(
            list(images.keys()),
            [
                "test_files/imgtk_test/im_Normal_DirectX.png",
                "test_files/imgtk_test/im_Normal_OpenGL.png",
            ],
        )

    def test_resolve_map_type(self):
        """Test resolve_map_type identifies texture types from filename."""
        self.assertEqual(
            ImgUtils.resolve_map_type("test_files/imgtk_test/im_h.png"),
            "Height",
        )
        self.assertEqual(
            ImgUtils.resolve_map_type("test_files/imgtk_test/im_h.png", key=False),
            "_H",
        )
        self.assertEqual(
            ImgUtils.resolve_map_type("test_files/imgtk_test/im_n.png"),
            "Normal",
        )
        self.assertEqual(
            ImgUtils.resolve_map_type("test_files/imgtk_test/im_n.png", key=False),
            "_N",
        )

    def test_filter_images_by_type(self):
        """Test filter_images_by_type filters by texture type."""
        files = FileUtils.get_dir_contents("test_files/imgtk_test")
        self.assertEqual(
            ImgUtils.filter_images_by_type(files, "Height"),
            ["im_h.png", "im_Height.png"],
        )

    def test_sort_images_by_type(self):
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
        self.assertEqual(
            ImgUtils.sort_images_by_type({"im_h.png": "<im_h>", "im_n.png": "<im_n>"}),
            {
                "Height": [("im_h.png", "<im_h>")],
                "Normal": [("im_n.png", "<im_n>")],
            },
        )

    def test_contains_map_types(self):
        """Test contains_map_types checks for texture types."""
        self.assertTrue(ImgUtils.contains_map_types([("im_h.png", "<im_h>")], "Height"))
        self.assertTrue(
            ImgUtils.contains_map_types(
                {"im_h.png": "<im_h>", "im_n.png": "<im_n>"}, "Height"
            )
        )
        self.assertTrue(
            ImgUtils.contains_map_types({"Height": [("im_h.png", "<im_h>")]}, "Height")
        )
        self.assertTrue(
            ImgUtils.contains_map_types(
                {"Height": [("im_h.png", "<im_h>")]}, ["Height", "Normal"]
            )
        )

    def test_is_normal_map(self):
        """Test is_normal_map identifies normal maps."""
        self.assertFalse(ImgUtils.is_normal_map("im_h.png"))
        self.assertTrue(ImgUtils.is_normal_map("im_n.png"))

    def test_invert_channels(self):
        """Test invert_channels inverts specified color channels."""
        result = ImgUtils.invert_channels(self.im_n, "g")
        channel = result.getchannel("G")
        self.assertEqual(channel.mode, "L")

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

    def test_create_mask(self):
        """Test create_mask generates image masks."""
        bg = ImgUtils.get_background("test_files/imgtk_test/im_Base_color.png", "RGB")

        mask1 = ImgUtils.create_mask("test_files/imgtk_test/im_Base_color.png", bg)
        self.assertEqual(mask1.mode, "L")

        mask2 = ImgUtils.create_mask(
            "test_files/imgtk_test/im_Base_color.png",
            "test_files/imgtk_test/im_Base_color.png",
        )
        self.assertEqual(mask2.mode, "L")

    def test_fill_masked_area(self):
        """Test fill_masked_area fills masked regions with color."""
        bg = ImgUtils.get_background("test_files/imgtk_test/im_Base_color.png", "RGB")
        mask = ImgUtils.create_mask("test_files/imgtk_test/im_Base_color.png", bg)

        result = ImgUtils.fill_masked_area(
            "test_files/imgtk_test/im_Base_color.png", (0, 255, 0), mask
        )
        self.assertEqual(result.mode, "RGB")

    def test_fill(self):
        """Test fill fills image with color."""
        result = ImgUtils.fill(self.im_h, (127, 127, 127))
        self.assertEqual(result.mode, "RGB")

    def test_get_background(self):
        """Test get_background determines background color."""
        self.assertEqual(
            ImgUtils.get_background("test_files/imgtk_test/im_Height.png", "I"), 32767
        )
        self.assertEqual(
            ImgUtils.get_background("test_files/imgtk_test/im_Height.png", "L"), 255
        )
        self.assertEqual(
            ImgUtils.get_background("test_files/imgtk_test/im_n.png", "RGB"),
            (127, 127, 255),
        )

    def test_replace_color(self):
        """Test replace_color substitutes colors in image."""
        bg = ImgUtils.get_background("test_files/imgtk_test/im_Base_color.png", "RGB")
        result = ImgUtils.replace_color(
            "test_files/imgtk_test/im_Base_color.png", bg, (255, 0, 0)
        )
        self.assertEqual(result.mode, "RGBA")

    def test_set_contrast(self):
        """Test set_contrast adjusts image contrast."""
        result = ImgUtils.set_contrast("test_files/imgtk_test/im_Mixed_AO.png", 255)
        self.assertEqual(result.mode, "L")

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

    def test_are_identical(self):
        """Test are_identical compares images."""
        self.assertFalse(ImgUtils.are_identical(self.im_h, self.im_n))
        self.assertTrue(ImgUtils.are_identical(self.im_h, self.im_h))


if __name__ == "__main__":
    unittest.main(exit=False)
