# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk color primitives (Color, ColorPair, Palette)."""

import unittest


class TestColor(unittest.TestCase):
    """Tests for the Color value type."""

    def _cls(self):
        from pythontk import Color

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


class TestColorSpace(unittest.TestCase):
    """The scalar sRGB <-> linear twins of ImgUtils' array conversions."""

    def _cls(self):
        from pythontk import Color

        return Color

    def test_mid_grey_is_a_fifth_of_the_light(self):
        Color = self._cls()
        linear = Color.linear_from_srgb((0.5, 0.5, 0.5))
        self.assertEqual([round(c, 3) for c in linear], [0.214, 0.214, 0.214])
        back = Color.srgb_from_linear(linear)
        self.assertEqual([round(c, 6) for c in back], [0.5, 0.5, 0.5])

    def test_the_toe_is_linear_and_the_ends_are_fixed(self):
        Color = self._cls()
        self.assertEqual(Color.linear_from_srgb((0.0, 1.0, 0.5))[:2], (0.0, 1.0))
        self.assertAlmostEqual(
            Color.linear_from_srgb((0.02, 0.0, 0.0))[0], 0.02 / 12.92, places=9
        )
        self.assertAlmostEqual(
            Color.srgb_from_linear((0.002, 0.0, 0.0))[0], 0.002 * 12.92, places=9
        )

    def test_alpha_passes_and_hdr_is_not_clamped(self):
        Color = self._cls()
        self.assertEqual(Color.linear_from_srgb((1.0, 1.0, 1.0, 0.3))[3], 0.3)
        self.assertGreater(Color.srgb_from_linear((2.0, 0.0, 0.0))[0], 1.0)
        self.assertEqual(Color.linear_from_srgb((-1.0, 0.0, 0.0))[0], 0.0)

    def test_matches_the_array_conversion(self):
        from pythontk import ImgUtils

        Color = self._cls()
        rgb = (0.1, 0.5, 0.9)
        expected = [round(float(c), 5) for c in ImgUtils.srgb_to_linear(list(rgb))]
        self.assertEqual([round(c, 5) for c in Color.linear_from_srgb(rgb)], expected)


class TestColorPair(unittest.TestCase):
    """Tests for the ColorPair container."""

    def _make(self, fg=None, bg=None):
        from pythontk import ColorPair

        return ColorPair(fg, bg)

    def _auto(self, fg, **kw):
        from pythontk import ColorPair

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
        from pythontk import Palette

        return Palette

    def _color_cls(self):
        from pythontk import Color

        return Color

    def _pair_cls(self):
        from pythontk import ColorPair

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


class TestPaletteWritePaths(unittest.TestCase):
    """``Palette`` wrapped on ``__init__``/``__setitem__`` only.

    Every other dict write path let a raw value in, so a Palette could hold
    plain strings that later code indexes as ``Color``/``ColorPair``:

        update / setdefault / |=   stored the raw str
        copy / |                   returned a plain dict, losing the class

    ``copy()`` is the likeliest trap -- ``Palette.override`` and ``.alias``
    both build derived palettes, and a consumer reaching for the dict idiom
    instead got something that no longer wraps on write.
    """

    HEX = "#112233"

    @staticmethod
    def _types():
        from pythontk import Palette, Color, ColorPair

        return Palette, Color, ColorPair

    def _assert_wrapped(self, palette, key, label):
        Palette, Color, _ = self._types()
        self.assertIsInstance(
            palette,
            Palette,
            f"{label} must return a Palette, got {type(palette).__name__}",
        )
        self.assertIsInstance(
            palette[key],
            Color,
            f"{label} must wrap its value, got {type(palette[key]).__name__}",
        )

    def test_update_wraps(self):
        Palette, Color, ColorPair = self._types()
        p = Palette()
        p.update({"a": self.HEX})
        self._assert_wrapped(p, "a", "update")

    def test_update_wraps_kwargs_and_pairs(self):
        Palette, Color, ColorPair = self._types()
        p = Palette()
        p.update(a=self.HEX, b=("#445566", "#778899"))
        self.assertIsInstance(p["a"], Color)
        self.assertIsInstance(p["b"], ColorPair)

    def test_setdefault_wraps_and_returns_the_wrapped_value(self):
        Palette, Color, ColorPair = self._types()
        p = Palette()
        returned = p.setdefault("a", self.HEX)
        self._assert_wrapped(p, "a", "setdefault")
        self.assertIsInstance(
            returned, Color, "setdefault must RETURN the wrapped value"
        )

    def test_setdefault_leaves_an_existing_entry_alone(self):
        Palette, Color, ColorPair = self._types()
        p = Palette(a=self.HEX)
        first = p["a"]
        self.assertIs(p.setdefault("a", "#FFFFFF"), first)

    def test_copy_keeps_the_class(self):
        Palette, Color, ColorPair = self._types()
        p = Palette(a=self.HEX)
        self._assert_wrapped(p.copy(), "a", "copy")

    def test_or_keeps_the_class_and_wraps(self):
        Palette, Color, ColorPair = self._types()
        p = Palette(a=self.HEX)
        self._assert_wrapped(p | {"b": "#445566"}, "b", "|")

    def test_ror_keeps_the_class_and_wraps(self):
        Palette, Color, ColorPair = self._types()
        """A plain dict on the LEFT must still produce a Palette."""
        p = Palette(a=self.HEX)
        self._assert_wrapped({"b": "#445566"} | p, "b", "reflected |")

    def test_ior_wraps(self):
        Palette, Color, ColorPair = self._types()
        p = Palette(a=self.HEX)
        p |= {"b": "#445566"}
        self._assert_wrapped(p, "b", "|=")

    def test_fromkeys_was_already_correct(self):
        Palette, Color, ColorPair = self._types()
        """Pinned, not fixed: ``dict.fromkeys`` on a subclass routes through
        ``__setitem__``, so this one always wrapped. It is here so a later
        override of the write paths does not accidentally break it."""
        self._assert_wrapped(Palette.fromkeys(["a"], self.HEX), "a", "fromkeys")


class TestColorStops(unittest.TestCase):
    """Tests for the ColorStops ramp-endpoint value type."""

    def _cls(self):
        from pythontk import ColorStops

        return ColorStops

    # -- Shape --------------------------------------------------------------

    def test_one_stop_has_no_low(self):
        s = self._cls()("highlight_color")
        self.assertEqual(s.keys, ("highlight_color",))
        self.assertEqual(tuple(s), ("highlight_color", None))

    def test_two_stops_are_ordered_high_first(self):
        s = self._cls()("hi", "lo")
        self.assertEqual(s.keys, ("hi", "lo"))

    def test_is_immutable(self):
        s = self._cls()("hi", "lo")
        with self.assertRaises(AttributeError):
            s.hi = "other"

    # -- Defaults -----------------------------------------------------------

    def test_the_two_stops_default_differently(self):
        """The inversion trap: an unstated HIGH reads white so the ramp still
        shows, but an unstated LOW must read BLACK. One shared default would
        make every asset authored before the low stop existed invert."""
        s = self._cls()("hi", "lo")
        self.assertEqual(s.defaults, ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)))

    def test_resolve_none_gives_each_stop_its_own_default(self):
        s = self._cls()("hi", "lo")
        self.assertEqual(s.resolve(None), ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)))

    # -- resolve ------------------------------------------------------------

    def test_a_bare_triple_is_the_high_stop(self):
        """The legacy producer shape: one colour, meaning the bright end."""
        s = self._cls()("hi", "lo")
        self.assertEqual(s.resolve((0.2, 0.5, 1.0)), ((0.2, 0.5, 1.0), (0.0, 0.0, 0.0)))

    def test_a_pair_fills_both_stops(self):
        s = self._cls()("hi", "lo")
        self.assertEqual(
            s.resolve(((0.2, 0.5, 1.0), (0.4, 0.0, 0.0))),
            ((0.2, 0.5, 1.0), (0.4, 0.0, 0.0)),
        )

    def test_an_unpublished_high_stop_does_not_shift_the_low_one(self):
        """``[None, rgb]`` is a pair with a hole, not a flat triple. Sniffing
        only the first element would read the LOW colour as the HIGH one."""
        s = self._cls()("hi", "lo")
        self.assertEqual(
            s.resolve([None, (0.0, 0.0, 0.1)]), ((1.0, 1.0, 1.0), (0.0, 0.0, 0.1))
        )

    def test_malformed_entries_fall_back_rather_than_raise(self):
        """A colour is lookdev: one bad triple must not cost the deliverable."""
        s = self._cls()("hi", "lo")
        for bad in ("nonsense", 7, (1, 2), (("a", "b", "c"), None), [None, None]):
            with self.subTest(bad=bad):
                self.assertEqual(s.resolve(bad), s.defaults)

    def test_resolve_is_idempotent(self):
        """Its own output must read back unchanged -- the collector resolves,
        and the writer resolves again on the way out."""
        s = self._cls()("hi", "lo")
        once = s.resolve(((0.2, 0.5, 1.0), (0.4, 0.0, 0.0)))
        self.assertEqual(s.resolve(once), once)

    def test_one_stop_resolves_to_one_triple(self):
        s = self._cls()("only")
        self.assertEqual(s.resolve((0.2, 0.5, 1.0)), ((0.2, 0.5, 1.0),))

    # -- Value semantics ----------------------------------------------------

    def test_equality_and_hash_cover_the_defaults(self):
        cls = self._cls()
        self.assertEqual(cls("hi", "lo"), cls("hi", "lo"))
        self.assertNotEqual(
            cls("hi", "lo"), cls("hi", "lo", lo_default=(1.0, 0.0, 0.0))
        )
        self.assertEqual(len({cls("hi", "lo"), cls("hi", "lo")}), 1)


class TestColorHsv(unittest.TestCase):
    """HSV accessors, and the lossiness they are documented to have."""

    def _cls(self):
        from pythontk import Color

        return Color

    def test_hsv_round_trips_where_the_colour_is_representable(self):
        c = self._cls().from_hsvf(0.58, 0.8, 1.0)
        h, s, v = c.hsv
        self.assertAlmostEqual(h, 0.58, places=2)
        self.assertAlmostEqual(s, 0.8, places=2)
        self.assertAlmostEqual(v, 1.0, places=2)

    def test_hue_is_lost_at_black(self):
        """Pinned, not a defect: this is exactly why an EDITING widget must
        keep float HSV state instead of storing through Color."""
        black = self._cls().from_hsvf(0.58, 0.8, 0.0)
        self.assertEqual(black.rgb, (0, 0, 0))
        self.assertEqual(black.hsv, (0.0, 0.0, 0.0))

    def test_hue_wraps_and_sv_clamp(self):
        cls = self._cls()
        self.assertEqual(cls.from_hsvf(1.25, 0.5, 0.5), cls.from_hsvf(0.25, 0.5, 0.5))
        self.assertEqual(cls.from_hsvf(0.5, 9.0, 9.0), cls.from_hsvf(0.5, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
