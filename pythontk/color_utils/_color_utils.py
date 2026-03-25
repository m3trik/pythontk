# !/usr/bin/python
# coding=utf-8
"""Lightweight, DCC-agnostic color primitives.

Provides three building blocks for consistent color handling across tools:

    Color      – Immutable RGBA value with format conversions and basic math.
    ColorPair  – Foreground/background pair (iterable as ``(fg_hex, bg_hex)``).
    Palette    – Named color collection with alias support.

Only depends on the standard library (``colorsys``).
"""

import colorsys
from typing import Dict, Iterator, Optional, Tuple, Union


class Color:
    """Immutable RGBA color stored as 0–255 integers.

    Create from hex, 0–255 ints, or 0.0–1.0 floats::

        Color.from_hex("#5B8BD4")
        Color(91, 139, 212)
        Color.from_rgbf(0.36, 0.55, 0.83)
    """

    __slots__ = ("_r", "_g", "_b", "_a")

    def __init__(self, r: int, g: int, b: int, a: int = 255) -> None:
        object.__setattr__(self, "_r", max(0, min(255, int(r))))
        object.__setattr__(self, "_g", max(0, min(255, int(g))))
        object.__setattr__(self, "_b", max(0, min(255, int(b))))
        object.__setattr__(self, "_a", max(0, min(255, int(a))))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Color is immutable")

    # ---- Factories --------------------------------------------------------

    @classmethod
    def from_hex(cls, hex_str: str) -> "Color":
        """Parse ``#RGB``, ``#RRGGBB``, or ``#RRGGBBAA``."""
        h = hex_str.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) == 6:
            return cls(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        if len(h) == 8:
            return cls(
                int(h[0:2], 16),
                int(h[2:4], 16),
                int(h[4:6], 16),
                int(h[6:8], 16),
            )
        raise ValueError(f"Invalid hex color: {hex_str!r}")

    @classmethod
    def from_rgbf(cls, r: float, g: float, b: float, a: float = 1.0) -> "Color":
        """Create from 0.0–1.0 float components (Maya API convention)."""
        return cls(round(r * 255), round(g * 255), round(b * 255), round(a * 255))

    # ---- Format properties ------------------------------------------------

    @property
    def hex(self) -> str:
        """``'#RRGGBB'`` (or ``'#RRGGBBAA'`` when alpha < 255)."""
        if self._a == 255:
            return f"#{self._r:02X}{self._g:02X}{self._b:02X}"
        return f"#{self._r:02X}{self._g:02X}{self._b:02X}{self._a:02X}"

    @property
    def rgb(self) -> Tuple[int, int, int]:
        """``(r, g, b)`` in 0–255."""
        return (self._r, self._g, self._b)

    @property
    def rgba(self) -> Tuple[int, int, int, int]:
        """``(r, g, b, a)`` in 0–255."""
        return (self._r, self._g, self._b, self._a)

    @property
    def rgbf(self) -> Tuple[float, float, float]:
        """``(r, g, b)`` in 0.0–1.0 (Maya API format)."""
        return (self._r / 255.0, self._g / 255.0, self._b / 255.0)

    @property
    def rgbaf(self) -> Tuple[float, float, float, float]:
        """``(r, g, b, a)`` in 0.0–1.0."""
        return (self._r / 255.0, self._g / 255.0, self._b / 255.0, self._a / 255.0)

    @property
    def luminance(self) -> float:
        """Perceived luminance (ITU-R BT.709, linear approximation).

        Does not apply sRGB gamma linearisation — sufficient for
        text-contrast decisions but not colorimetrically accurate.
        """
        return (
            0.2126 * (self._r / 255)
            + 0.7152 * (self._g / 255)
            + 0.0722 * (self._b / 255)
        )

    # ---- Color math -------------------------------------------------------

    def lighter(self, factor: float = 0.2) -> "Color":
        """Return a lighter colour.  *factor* 0.0 = unchanged, 1.0 = white."""
        return self.blend(Color(255, 255, 255, self._a), factor)

    def darker(self, factor: float = 0.2) -> "Color":
        """Return a darker colour.  *factor* 0.0 = unchanged, 1.0 = black."""
        return self.blend(Color(0, 0, 0, self._a), factor)

    def with_alpha(self, a: Union[int, float]) -> "Color":
        """Return a copy with a new alpha (int 0–255 or float 0.0–1.0)."""
        if isinstance(a, float):
            if 0.0 <= a <= 1.0:
                a = round(a * 255)
            else:
                raise ValueError(
                    f"Float alpha must be 0.0–1.0, got {a}. "
                    f"Use int for 0–255 range."
                )
        return Color(self._r, self._g, self._b, int(a))

    def blend(self, other: "Color", t: float = 0.5) -> "Color":
        """Linear interpolation towards *other* by *t* (0.0 = self, 1.0 = other)."""
        inv = 1.0 - t
        return Color(
            round(self._r * inv + other._r * t),
            round(self._g * inv + other._g * t),
            round(self._b * inv + other._b * t),
            round(self._a * inv + other._a * t),
        )

    def subtle_bg(self, value: float = 0.24, sat_factor: float = 1.0) -> "Color":
        """Derive a tinted dark-theme background from this colour.

        Preserves hue, sets brightness to *value*, scales saturation
        by *sat_factor*.  Useful for generating ``(fg, bg)`` pairs::

            fg = Color.from_hex("#88B8D0")
            bg = fg.subtle_bg()  # dark blue-grey tint
        """
        h, s, _ = colorsys.rgb_to_hsv(self._r / 255, self._g / 255, self._b / 255)
        s2 = min(s * sat_factor, 1.0)
        r, g, b = colorsys.hsv_to_rgb(h, s2, value)
        return Color(round(r * 255), round(g * 255), round(b * 255), self._a)

    # ---- Dunder -----------------------------------------------------------

    def __str__(self) -> str:
        return self.hex

    def __repr__(self) -> str:
        return f"Color({self.hex!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Color):
            return self.rgba == other.rgba
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.rgba)

    def __iter__(self) -> Iterator[int]:
        """Yield ``(r, g, b, a)`` for direct unpacking."""
        return iter(self.rgba)


# -----------------------------------------------------------------------
# ColorPair
# -----------------------------------------------------------------------


class ColorPair:
    """Foreground / background pair for themed UIs.

    Iterates as ``(fg_hex_or_None, bg_hex_or_None)`` so existing code
    that unpacks ``(str, str)`` tuples keeps working::

        fg, bg = ColorPair.auto("#88B8D0")
        fg, bg = PASTEL_STATUS["collision"]
    """

    __slots__ = ("fg", "bg")

    def __init__(
        self,
        fg: Optional[Union[str, "Color"]] = None,
        bg: Optional[Union[str, "Color"]] = None,
    ) -> None:
        object.__setattr__(
            self, "fg", Color.from_hex(fg) if isinstance(fg, str) else fg
        )
        object.__setattr__(
            self, "bg", Color.from_hex(bg) if isinstance(bg, str) else bg
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ColorPair is immutable")

    @classmethod
    def auto(
        cls,
        fg: Union[str, "Color"],
        value: float = 0.24,
        sat_factor: float = 1.0,
    ) -> "ColorPair":
        """Derive background automatically from foreground for dark themes."""
        fg_color = Color.from_hex(fg) if isinstance(fg, str) else fg
        return cls(fg_color, fg_color.subtle_bg(value, sat_factor))

    # ---- Sequence protocol (backwards compat) -----------------------------

    def __iter__(self) -> Iterator[Optional[str]]:
        """Yield ``(fg_hex_or_None, bg_hex_or_None)``."""
        yield self.fg.hex if self.fg else None
        yield self.bg.hex if self.bg else None

    def __getitem__(self, index: int) -> Optional[str]:
        """Subscript access: ``pair[0]`` → fg hex, ``pair[1]`` → bg hex."""
        if index == 0:
            return self.fg.hex if self.fg else None
        if index == 1:
            return self.bg.hex if self.bg else None
        raise IndexError(index)

    def __len__(self) -> int:
        return 2

    # ---- Dunder -----------------------------------------------------------

    def __repr__(self) -> str:
        return f"ColorPair({self.fg!r}, {self.bg!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ColorPair):
            return self.fg == other.fg and self.bg == other.bg
        if isinstance(other, tuple) and len(other) == 2:
            return tuple(self) == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.fg, self.bg))


# -----------------------------------------------------------------------
# Palette
# -----------------------------------------------------------------------


class Palette(dict):
    """Named color collection with auto-wrapping and alias support.

    Accepts mixed input — strings become :class:`Color`, two-tuples
    become :class:`ColorPair`::

        p = Palette({
            "info":  "#88B8D0",
            "warn":  ("#D4B878", "#3D3528"),
            "valid": (None, None),
        })
        p["info"]           # Color('#88B8D0')
        fg, bg = p["warn"]  # ('#D4B878', '#3D3528')
    """

    def __init__(
        self,
        mapping: Optional[Union[Dict, "Palette"]] = None,
        **kwargs: object,
    ) -> None:
        entries = dict(mapping or {}, **kwargs)
        super().__init__({k: self._wrap(v) for k, v in entries.items()})

    def __setitem__(self, key: str, value: object) -> None:
        super().__setitem__(key, self._wrap(value))

    @staticmethod
    def _wrap(v: object) -> object:
        if v is None or isinstance(v, (Color, ColorPair)):
            return v
        if isinstance(v, str):
            return Color.from_hex(v)
        if isinstance(v, (tuple, list)) and len(v) == 2:
            return ColorPair(v[0], v[1])
        return v

    def alias(self, mapping: Dict[str, str]) -> "Palette":
        """Return a new Palette with additional keys pointing to existing values.

        ::

            base = Palette(info=("#88B8D0", "#28323D"))
            extended = base.alias({
                "missing_shot": "info",
                "user_animated": "info",
            })
        """
        out = Palette(self)
        for new_key, existing_key in mapping.items():
            out[new_key] = self[existing_key]
        return out

    def override(self, **kwargs: object) -> "Palette":
        """Return a new Palette with selected entries replaced.

        ::

            custom = Palette.status().override(error=("#FF6666", "#3D2020"))
        """
        out = Palette(self)
        for k, v in kwargs.items():
            out[k] = v
        return out

    # ---- Built-in palettes ------------------------------------------------

    @classmethod
    def status(cls) -> "Palette":
        """Standard severity palette for dark-theme UIs.

        Five tiers designed as soft pastels on dark grey backgrounds::

            p = Palette.status()
            fg, bg = p["info"]    # steel-blue
            fg, bg = p["warn"]    # warm gold
            fg, bg = p["error"]   # soft coral
            p["valid"]            # (None, None) — no color
            p["locked"][0]        # "#888888" — dimmed grey

        Extend with domain-specific aliases::

            manifest = Palette.status().alias({
                "missing_shot": "info",
                "collision":    "error",
            })
        """
        return cls(
            {
                "valid": (None, None),
                "locked": ("#888888", None),
                "info": ("#88B8D0", "#28323D"),  # soft steel-blue
                "warn": ("#D4B878", "#3D3528"),  # warm gold
                "error": ("#D4908F", "#3D2828"),  # soft coral
            }
        )

    @classmethod
    def axes(cls) -> "Palette":
        """Standard XYZ / RGB axis colours (Maya / 3D convention).

        ::

            p = Palette.axes()
            p["x"].hex  # red
            p["y"].hex  # green
            p["z"].hex  # blue
        """
        return cls({"x": "#E06666", "y": "#6AA84F", "z": "#6FA8DC"})

    @classmethod
    def channels(cls) -> "Palette":
        """Standard transform-attribute colours for animation editors.

        Maps ``translateX/Y/Z``, ``rotateX/Y/Z``, ``scaleX/Y/Z``,
        ``visibility``, and ``consolidated`` (for collapsed curves)::

            p = Palette.channels()
            p["translateX"].hex       # "#E06666"
            p["consolidated"].hex     # "#FFFFFF"

        Uses the Maya convention: X = red family, Y = green family,
        Z = blue family, with translate/rotate/scale as brightness variants.
        """
        return cls(
            {
                "translateX": "#E06666",
                "translateY": "#6AA84F",
                "translateZ": "#6FA8DC",
                "rotateX": "#CC4125",
                "rotateY": "#38761D",
                "rotateZ": "#3D85C6",
                "scaleX": "#F6B26B",
                "scaleY": "#93C47D",
                "scaleZ": "#76A5AF",
                "visibility": "#FFD966",
                "consolidated": "#FFFFFF",
            }
        )

    @classmethod
    def ui(cls) -> "Palette":
        """Common UI element colours for dark themes.

        Semantic colour names for backgrounds, text, and accents::

            p = Palette.ui()
            p["bg"].hex           # "#1E1E1E"
            p["text"].hex         # "#CCCCCC"
            p["accent"].hex       # "#5B8BD4"
            p["text_dim"].hex     # "#888888"
        """
        return cls(
            {
                "bg": "#1E1E1E",
                "bg_alt": "#252525",
                "bg_surface": "#2B2B2B",
                "text": "#CCCCCC",
                "text_dim": "#888888",
                "text_bright": "#FFFFFF",
                "accent": "#5B8BD4",
                "border": "#3A3A3A",
                "highlight": "#E8A84A",
            }
        )
