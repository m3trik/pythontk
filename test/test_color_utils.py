# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.color_utils (Color, ColorPair, Palette)."""

import unittest


class TestColor(unittest.TestCase):
    """Tests for the Color value type."""

    def _cls(self):
        from pythontk.color_utils._color_utils import Color

        return Color

    # -- Construction -------------------------------------------------------

    def test_from_hex_6(self):
        c = self._cls().from_hex("#5B8BD4")
        self.assertEqual(c.rgb, (91, 139, 212))
        self.assertEqual(c.rgba, (91, 139, 212, 255))

    def test_from_hex_3(self):
        c = self._cls().from_hex("#FFF")
        self.assertEqual(c.rgb, (255, 255, 255))

    def test_from_hex_8_alpha(self):
        c = self._cls().from_hex("#5B8BD480")
        self.assertEqual(c.rgba, (91, 139, 212, 128))

    def test_from_hex_invalid(self):
        with self.assertRaises(ValueError):
            self._cls().from_hex("#ZZZZZZ")

    def test_from_hex_wrong_length(self):
        with self.assertRaises(ValueError):
            self._cls().from_hex("#12345")

    def test_from_rgbf(self):
        c = self._cls().from_rgbf(1.0, 0.5, 0.0)
        self.assertEqual(c.rgb, (255, 128, 0))

    def test_int_constructor(self):
        c = self._cls()(91, 139, 212)
        self.assertEqual(c.hex, "#5B8BD4")

    def test_clamping(self):
        c = self._cls()(-10, 300, 128)
        self.assertEqual(c.rgb, (0, 255, 128))

    # -- Format properties --------------------------------------------------

    def test_hex_opaque(self):
        c = self._cls()(91, 139, 212)
        self.assertEqual(c.hex, "#5B8BD4")

    def test_hex_with_alpha(self):
        c = self._cls()(91, 139, 212, 128)
        self.assertEqual(c.hex, "#5B8BD480")

    def test_rgbf(self):
        c = self._cls()(255, 0, 0)
        self.assertAlmostEqual(c.rgbf[0], 1.0)
        self.assertAlmostEqual(c.rgbf[1], 0.0)
        self.assertAlmostEqual(c.rgbf[2], 0.0)

    def test_rgbaf(self):
        c = self._cls()(0, 0, 0, 128)
        self.assertAlmostEqual(c.rgbaf[3], 128 / 255)

    def test_luminance_white(self):
        c = self._cls()(255, 255, 255)
        self.assertAlmostEqual(c.luminance, 1.0, places=2)

    def test_luminance_black(self):
        c = self._cls()(0, 0, 0)
        self.assertAlmostEqual(c.luminance, 0.0)

    # -- Color math ---------------------------------------------------------

    def test_lighter(self):
        c = self._cls()(100, 100, 100)
        lighter = c.lighter(0.5)
        self.assertGreater(lighter.luminance, c.luminance)

    def test_darker(self):
        c = self._cls()(100, 100, 100)
        darker = c.darker(0.5)
        self.assertLess(darker.luminance, c.luminance)

    def test_blend_midpoint(self):
        black = self._cls()(0, 0, 0)
        white = self._cls()(255, 255, 255)
        mid = black.blend(white, 0.5)
        self.assertEqual(mid.rgb, (128, 128, 128))

    def test_with_alpha_int(self):
        c = self._cls()(100, 100, 100)
        c2 = c.with_alpha(128)
        self.assertEqual(c2.rgba[3], 128)
        self.assertEqual(c2.rgb, c.rgb)

    def test_with_alpha_float(self):
        c = self._cls()(100, 100, 100)
        c2 = c.with_alpha(0.5)
        self.assertEqual(c2.rgba[3], 128)

    def test_with_alpha_float_out_of_range(self):
        c = self._cls()(100, 100, 100)
        with self.assertRaises(ValueError):
            c.with_alpha(2.0)

    def test_subtle_bg_preserves_hue(self):
        """Background derived from blue fg should remain blue-ish."""
        Color = self._cls()
        import colorsys

        fg = Color.from_hex("#88B8D0")
        bg = fg.subtle_bg()
        h_fg, _, _ = colorsys.rgb_to_hsv(*fg.rgbf)
        h_bg, _, _ = colorsys.rgb_to_hsv(*bg.rgbf)
        self.assertAlmostEqual(h_fg, h_bg, places=2)
        self.assertLess(bg.luminance, fg.luminance)

    # -- Immutability -------------------------------------------------------

    def test_immutable(self):
        c = self._cls()(100, 100, 100)
        with self.assertRaises(AttributeError):
            c._r = 200

    # -- Dunder -------------------------------------------------------------

    def test_str(self):
        c = self._cls().from_hex("#AABBCC")
        self.assertEqual(str(c), "#AABBCC")

    def test_repr(self):
        c = self._cls().from_hex("#AABBCC")
        self.assertIn("#AABBCC", repr(c))

    def test_eq(self):
        Color = self._cls()
        self.assertEqual(Color(10, 20, 30), Color(10, 20, 30))
        self.assertNotEqual(Color(10, 20, 30), Color(10, 20, 31))

    def test_hash(self):
        Color = self._cls()
        s = {Color(1, 2, 3), Color(1, 2, 3), Color(4, 5, 6)}
        self.assertEqual(len(s), 2)

    def test_iter(self):
        c = self._cls()(10, 20, 30, 40)
        r, g, b, a = c
        self.assertEqual((r, g, b, a), (10, 20, 30, 40))


class TestColorPair(unittest.TestCase):
    """Tests for the ColorPair container."""

    def _make(self, fg=None, bg=None):
        from pythontk.color_utils._color_utils import ColorPair

        return ColorPair(fg, bg)

    def _auto(self, fg, **kw):
        from pythontk.color_utils._color_utils import ColorPair

        return ColorPair.auto(fg, **kw)

    # -- Construction -------------------------------------------------------

    def test_from_hex_strings(self):
        p = self._make("#88B8D0", "#28323D")
        self.assertEqual(p[0], "#88B8D0")
        self.assertEqual(p[1], "#28323D")

    def test_none_values(self):
        p = self._make(None, None)
        self.assertIsNone(p[0])
        self.assertIsNone(p[1])

    def test_fg_only(self):
        p = self._make("#888888", None)
        self.assertEqual(p[0], "#888888")
        self.assertIsNone(p[1])

    def test_auto_derives_bg(self):
        p = self._auto("#88B8D0")
        self.assertIsNotNone(p.fg)
        self.assertIsNotNone(p.bg)
        self.assertLess(p.bg.luminance, p.fg.luminance)

    # -- Sequence protocol --------------------------------------------------

    def test_iter_unpacking(self):
        fg, bg = self._make("#AABBCC", "#112233")
        self.assertEqual(fg, "#AABBCC")
        self.assertEqual(bg, "#112233")

    def test_getitem(self):
        p = self._make("#AABBCC", "#112233")
        self.assertEqual(p[0], "#AABBCC")
        self.assertEqual(p[1], "#112233")
        with self.assertRaises(IndexError):
            p[2]

    def test_len(self):
        self.assertEqual(len(self._make()), 2)

    # -- Equality -----------------------------------------------------------

    def test_eq_pair(self):
        a = self._make("#AABBCC", "#112233")
        b = self._make("#AABBCC", "#112233")
        self.assertEqual(a, b)

    def test_eq_tuple(self):
        p = self._make("#AABBCC", "#112233")
        self.assertEqual(p, ("#AABBCC", "#112233"))

    def test_eq_none_tuple(self):
        p = self._make(None, None)
        self.assertEqual(p, (None, None))

    # -- Immutability -------------------------------------------------------

    def test_immutable(self):
        p = self._make("#AABBCC", "#112233")
        with self.assertRaises(AttributeError):
            p.fg = None


class TestPalette(unittest.TestCase):
    """Tests for the Palette dict subclass."""

    def _cls(self):
        from pythontk.color_utils._color_utils import Palette

        return Palette

    def _color_cls(self):
        from pythontk.color_utils._color_utils import Color

        return Color

    def _pair_cls(self):
        from pythontk.color_utils._color_utils import ColorPair

        return ColorPair

    # -- Auto-wrapping ------------------------------------------------------

    def test_string_wraps_to_color(self):
        p = self._cls()({"blue": "#5B8BD4"})
        self.assertIsInstance(p["blue"], self._color_cls())

    def test_tuple_wraps_to_colorpair(self):
        p = self._cls()({"warn": ("#D4B878", "#3D3528")})
        self.assertIsInstance(p["warn"], self._pair_cls())

    def test_none_tuple_wraps_to_colorpair(self):
        p = self._cls()({"valid": (None, None)})
        self.assertIsInstance(p["valid"], self._pair_cls())

    def test_passthrough_color(self):
        c = self._color_cls().from_hex("#FF0000")
        p = self._cls()({"red": c})
        self.assertIs(p["red"], c)

    def test_setitem_wraps(self):
        p = self._cls()()
        p["new"] = "#00FF00"
        self.assertIsInstance(p["new"], self._color_cls())

    # -- Alias --------------------------------------------------------------

    def test_alias_creates_new_palette(self):
        Palette = self._cls()
        base = Palette({"info": ("#88B8D0", "#28323D"), "warn": ("#D4B878", "#3D3528")})
        extended = base.alias({"missing_shot": "info", "additional": "warn"})
        # Original unchanged
        self.assertNotIn("missing_shot", base)
        # Extended has both
        self.assertIn("info", extended)
        self.assertIn("missing_shot", extended)
        self.assertEqual(extended["missing_shot"], extended["info"])

    def test_alias_preserves_object_identity(self):
        Palette = self._cls()
        base = Palette({"error": ("#D4908F", "#3D2828")})
        extended = base.alias({"collision": "error"})
        # Values should be the same object
        self.assertIs(extended["collision"], base["error"])

    # -- Backwards compatibility with manifest patterns ---------------------

    def test_get_with_default(self):
        """Manifest pattern: PASTEL_STATUS.get(status, (None, None))"""
        p = self._cls()({"warn": ("#D4B878", "#3D3528")})
        fg, bg = p.get("warn", (None, None))
        self.assertEqual(fg, "#D4B878")
        self.assertEqual(bg, "#3D3528")
        # Missing key → default tuple
        fg2, bg2 = p.get("nonexistent", (None, None))
        self.assertIsNone(fg2)
        self.assertIsNone(bg2)

    def test_subscript_on_get(self):
        """Manifest pattern: BEHAVIOR_COLORS.get(b, (None, None))[0]"""
        p = self._cls()({"fade_in": ("#8ECFBF", None)})
        color = p.get("fade_in", (None, None))[0]
        self.assertEqual(color, "#8ECFBF")
        # Missing key
        color2 = p.get("nonexistent", (None, None))[0]
        self.assertIsNone(color2)

    # -- Override -----------------------------------------------------------

    def test_override_returns_new_palette(self):
        Palette = self._cls()
        base = Palette.status()
        custom = base.override(error=("#FF6666", "#3D2020"))
        # Original unchanged
        self.assertEqual(base["error"][0], "#D4908F")
        # Override applied
        self.assertEqual(custom["error"][0], "#FF6666")
        # Other keys preserved
        self.assertEqual(custom["info"][0], base["info"][0])

    # -- Built-in palettes --------------------------------------------------

    def test_status_palette(self):
        Palette = self._cls()
        p = Palette.status()
        # Has all five tiers
        for key in ("valid", "locked", "info", "warn", "error"):
            self.assertIn(key, p)
        # valid is (None, None)
        fg, bg = p["valid"]
        self.assertIsNone(fg)
        self.assertIsNone(bg)
        # error has fg and bg
        fg, bg = p["error"]
        self.assertIsNotNone(fg)
        self.assertIsNotNone(bg)

    def test_status_alias_workflow(self):
        """Full intended workflow: defaults + domain aliases."""
        Palette = self._cls()
        p = Palette.status().alias(
            {
                "missing_shot": "info",
                "collision": "error",
            }
        )
        self.assertEqual(p["missing_shot"], p["info"])
        self.assertEqual(p["collision"], p["error"])

    def test_axes_palette(self):
        Palette = self._cls()
        p = Palette.axes()
        for key in ("x", "y", "z"):
            self.assertIn(key, p)

    def test_channels_palette(self):
        Palette = self._cls()
        p = Palette.channels()
        for key in ("translateX", "rotateY", "scaleZ", "visibility", "consolidated"):
            self.assertIn(key, p)

    def test_ui_palette(self):
        Palette = self._cls()
        p = Palette.ui()
        for key in ("bg", "text", "accent", "text_dim", "border"):
            self.assertIn(key, p)


if __name__ == "__main__":
    unittest.main()
