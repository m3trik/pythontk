#!/usr/bin/python
# coding=utf-8
"""Tests for pythontk.MatReport — the DCC-agnostic material/texture report formatters
(shared SSoT for mayatk + blendertk get_mat_info / get_texture_info)."""
import unittest

from pythontk import MatReport


class TestFmtSizeAuto(unittest.TestCase):
    def test_units(self):
        self.assertEqual(MatReport._fmt_size_auto(None), "(unknown)")
        self.assertEqual(MatReport._fmt_size_auto(512), "512 bytes")
        self.assertEqual(MatReport._fmt_size_auto(2048), "2.0 KB")
        self.assertEqual(MatReport._fmt_size_auto(2 * 1024**2), "2.00 MB")
        self.assertEqual(MatReport._fmt_size_auto(3 * 1024**3), "3.00 GB")

    def test_non_numeric_passthrough(self):
        self.assertEqual(MatReport._fmt_size_auto("big"), "big")


class TestPathLink(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(MatReport._path_as_link(""), "")

    def test_special_chars_escaped_and_encoded(self):
        link = MatReport._path_as_link(r"C:\Dropbox (M+F)\a & b.png")
        self.assertIn("file:///", link)
        self.assertIn("a &amp; b.png", link)  # display escaped (keeps the original path text)
        self.assertIn("%20", link)            # space url-encoded in href
        href = link.split("href='", 1)[1].split("'", 1)[0]
        self.assertNotIn("\\", href)          # backslashes normalized in the href only


class TestMatInfoFormatters(unittest.TestCase):
    RECORDS = [
        {
            "material": "wood & oak",
            "type": "Principled BSDF",
            "textures": [
                {
                    "file_node": "Image Texture",
                    "path": r"C:\tex\a b.png",
                    "name": "a b.png",
                    "size": 2_500_000,
                    "width": 2048,
                    "height": 2048,
                    "mode": "RGBA",
                    "format": "PNG",
                    "bit_depth": "32bit (8x4)",
                    "optimization": {"recommended": True, "reasons": ["resize to 1024"]},
                }
            ],
        },
        {"material": "empty", "type": "Principled BSDF", "textures": []},
    ]

    def test_html(self):
        html = MatReport.format_mat_info_html(self.RECORDS)
        self.assertIn("Material Info", html)
        self.assertIn("<b>2</b> material(s)", html)
        self.assertIn("wood &amp; oak", html)        # escaped name
        self.assertIn("file:///", html)              # path link
        self.assertIn("Optimize:  YES", html)
        self.assertIn("(no textures)", html)         # second record

    def test_text(self):
        txt = MatReport.format_mat_info_text(self.RECORDS)
        self.assertIn("Material Info — 2 material(s)", txt)
        self.assertIn("[Principled BSDF] wood & oak", txt)  # text is NOT escaped
        self.assertIn("Optimize:  YES", txt)
        self.assertIn("(no textures)", txt)

    def test_optimization_error_branch(self):
        recs = [{
            "material": "m", "type": "t",
            "textures": [{"name": "x.png", "path": "/x.png", "size": 1,
                          "optimization": {"error": "boom"}}],
        }]
        self.assertIn("error: boom", MatReport.format_mat_info_text(recs))
        self.assertIn("error: boom", MatReport.format_mat_info_html(recs))


class TestTextureInfoFormatters(unittest.TestCase):
    INFO = [{"name": "a.png", "path": "/t/a.png", "size": 1024,
             "width": 16, "height": 16, "mode": "RGB", "format": "PNG"}]

    def test_html_and_text(self):
        self.assertIn("Texture Info", MatReport.format_texture_info_html(self.INFO))
        self.assertIn("Found 1 valid texture(s)", MatReport.format_texture_info_text(self.INFO))


if __name__ == "__main__":
    unittest.main()
