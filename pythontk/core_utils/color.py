# !/usr/bin/python
# coding=utf-8
"""Lightweight, DCC-agnostic color primitives.

Provides four building blocks for consistent color handling across tools:

    Color      – Immutable RGBA value with format conversions and basic math.
    ColorPair  – Foreground/background pair (iterable as ``(fg_hex, bg_hex)``).
    ColorStops – Named endpoints of a 0–1 ramp, each with its own default.
    Palette    – Named color collection with alias support.

Only depends on the standard library (``colorsys``).
"""

import colorsys
from typing import Dict, Iterator, Optional, Tuple, Union, Sequence


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

    @classmethod
    def from_hsvf(cls, h: float, s: float, v: float, a: float = 1.0) -> "Color":
        """Create from 0.0–1.0 HSV floats (hue wraps; s/v clamp).

        Not a faithful inverse of :attr:`hsv`: the 8-bit store quantises, and
        hue is undefined at ``s == 0`` or ``v == 0``. Anything that EDITS a
        colour (a picker, a slider) must keep its own float HSV state and use
        this only to hand the result out -- round-tripping a drag through
        ``hsv``/``from_hsvf`` walks the hue off and collapses it to grey at the
        black end.
        """
        r, g, b = colorsys.hsv_to_rgb(
            h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v))
        )
        return cls(round(r * 255), round(g * 255), round(b * 255), round(a * 255))

    # ---- Colour space -----------------------------------------------------

    @staticmethod
    def linear_from_srgb(rgb: Sequence[float]) -> Tuple[float, ...]:
        """Display-encoded components (0-1) to linear light (IEC 61966-2-1).

        The scalar twin of ``ImgUtils.srgb_to_linear``, which takes images and
        arrays. A picker's swatch is display-encoded; a DCC colour attribute
        and a glTF factor are linear. A widget that shows one as the other
        reads a mid value as a bright one -- and a page rendering the same
        number correctly then shows it brighter than the widget did. The first
        three components are converted; any alpha passes through.
        """
        return tuple(
            (
                (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
                if i < 3
                else c
            )
            for i, c in enumerate(max(0.0, float(v)) for v in rgb)
        )

    @staticmethod
    def srgb_from_linear(rgb: Sequence[float]) -> Tuple[float, ...]:
        """Linear light to display-encoded components; :meth:`linear_from_srgb`'s inverse.

        A value past 1.0 (HDR emission) encodes past 1.0 too: the caller
        clamps for an 8-bit display, so the clamp is not silently applied to
        a value it may want to read.
        """
        return tuple(
            (
                (c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055)
                if i < 3
                else c
            )
            for i, c in enumerate(max(0.0, float(v)) for v in rgb)
        )

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
    def hsv(self) -> Tuple[float, float, float]:
        """``(h, s, v)`` in 0.0–1.0.

        A read-out, not a working representation: hue and saturation are
        undefined as value approaches zero, so a slider that stores its result
        here and reads it back on the next drag loses what the artist set.
        See :meth:`from_hsvf`.
        """
        return colorsys.rgb_to_hsv(self._r / 255.0, self._g / 255.0, self._b / 255.0)

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
                    f"Float alpha must be 0.0–1.0, got {a}. Use int for 0–255 range."
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
# ColorStops
# -----------------------------------------------------------------------


class ColorStops:
    """The endpoints of the ramp a 0–1 channel drives, and their defaults.

    A render-effect channel is one keyable float; what it MEANS is a colour
    ramp that float indexes. ``hi`` names where a sample of 1.0 lands and the
    optional ``lo`` where 0.0 lands -- as attribute names, published track
    keys, or whatever string a consumer addresses a colour by, since the two
    sides of the glTF join spell the same concept differently.

    One object rather than two loose fields, because the stops are not
    independent: a ``lo`` without a ``hi`` is meaningless, and each stop needs
    its OWN fallback. A channel that publishes no colour reads WHITE so the
    ramp still shows; a missing LOW stop must read BLACK. Hand the low stop the
    high default and every asset authored before the low stop existed inverts
    on its next export -- which is why the defaults live here, beside the stops
    they belong to, instead of as one module constant a caller picks.
    """

    __slots__ = ("hi", "lo", "hi_default", "lo_default")

    #: A stop naming no colour still has to read: an unstated HIGH is white.
    WHITE: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    #: An unstated LOW is unlit. See the class docstring for why it differs.
    BLACK: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __init__(
        self,
        hi: str,
        lo: Optional[str] = None,
        hi_default: Tuple[float, float, float] = WHITE,
        lo_default: Tuple[float, float, float] = BLACK,
    ) -> None:
        object.__setattr__(self, "hi", hi)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi_default", tuple(hi_default))
        object.__setattr__(self, "lo_default", tuple(lo_default))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ColorStops is immutable")

    @property
    def keys(self) -> Tuple[str, ...]:
        """The stop names that exist, high first."""
        return (self.hi,) if self.lo is None else (self.hi, self.lo)

    @property
    def defaults(self) -> Tuple[Tuple[float, float, float], ...]:
        """Each stop's own fallback, in :attr:`keys` order."""
        return (
            (self.hi_default,)
            if self.lo is None
            else (self.hi_default, self.lo_default)
        )

    def resolve(self, value: object = None) -> Tuple[Tuple[float, float, float], ...]:
        """Read a published colour into one rgb triple per stop.

        Accepts the shapes producers actually hold, so one written before the
        low stop existed stays correct with no alias and no version check::

            None                    -> every stop takes its default
            (r, g, b)               -> the HIGH stop; low takes its default
            ((r, g, b), (r, g, b))  -> both stops, in :attr:`keys` order

        An entry that is absent or malformed falls back rather than raising: a
        colour is lookdev, and refusing to publish a whole scene over one bad
        triple trades a wrong tint for no deliverable.
        """
        defaults = self.defaults
        if value is None:
            return defaults
        # A flat triple is three NUMBERS; anything else sequence-shaped is one
        # entry per stop. Sniffing ``value[0]`` alone would misread a pair
        # whose high stop was never published (``[None, (r, g, b)]``) as a
        # flat triple and hand the low colour to the high stop.
        numeric = (int, float)
        flat = (
            isinstance(value, (list, tuple))
            and len(value) >= 3
            and all(
                isinstance(c, numeric) and not isinstance(c, bool) for c in value[:3]
            )
        )
        if flat or not isinstance(value, (list, tuple)):
            items: Tuple[object, ...] = (value,)
        else:
            items = tuple(value)
        out = []
        for index, fallback in enumerate(defaults):
            out.append(
                self._triple(items[index] if index < len(items) else None, fallback)
            )
        return tuple(out)

    @staticmethod
    def _triple(value: object, fallback: Tuple[float, float, float]):
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return fallback
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return fallback

    def __iter__(self) -> Iterator[Optional[str]]:
        """Yield ``(hi, lo)`` -- ``lo`` is ``None`` for a one-stop channel."""
        yield self.hi
        yield self.lo

    def __repr__(self) -> str:
        return f"ColorStops({self.hi!r}, {self.lo!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ColorStops):
            return (self.hi, self.lo, self.hi_default, self.lo_default) == (
                other.hi,
                other.lo,
                other.hi_default,
                other.lo_default,
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.hi, self.lo, self.hi_default, self.lo_default))


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

    # Every dict write path has to wrap, not just __setitem__: a Palette that
    # holds a raw str is indistinguishable from a correct one until something
    # indexes it as a Color, and the failure surfaces far from the write. The
    # class-returning paths (copy, |, reflected |) matter for the same reason
    # -- a plain dict silently stops wrapping everything written to it after.
    # fromkeys needs no override: dict.fromkeys on a subclass routes through
    # __setitem__ already.

    def update(self, mapping=None, **kwargs: object) -> None:  # type: ignore[override]
        """Wrap on update, as ``__setitem__`` does. Accepts a mapping, an
        iterable of pairs, or keywords -- the full ``dict.update`` contract."""
        if mapping is not None:
            items = mapping.items() if hasattr(mapping, "keys") else mapping
            for k, v in items:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def setdefault(self, key: str, default: object = None) -> object:  # type: ignore[override]
        """Wrap the inserted default, and return the WRAPPED value -- callers
        use the return value, so handing back the raw argument would leak it
        past the wrap just as surely as storing it would."""
        if key not in self:
            self[key] = default
        return self[key]

    def copy(self) -> "Palette":  # type: ignore[override]
        """A Palette, not a plain dict: ``dict.copy`` drops the subclass, and
        the copy then stops wrapping everything written to it."""
        return Palette(self)

    def __or__(self, other) -> "Palette":
        out = Palette(self)
        out.update(other)
        return out

    def __ror__(self, other) -> "Palette":
        out = Palette(other)
        out.update(self)
        return out

    def __ior__(self, other) -> "Palette":
        self.update(other)
        return self

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

    @classmethod
    def diff(cls) -> "Palette":
        """Comparison / diff palette for dark-theme tree views.

        Four universal categories with auto-derived backgrounds::

            p = Palette.diff()
            fg, bg = p["removed"]   # muted red
            fg, bg = p["added"]     # muted gold
            fg, bg = p["changed"]   # muted teal
            fg, bg = p["moved"]     # muted purple

        Alias domain-specific terms onto the base categories::

            hierarchy = Palette.diff().alias({
                "missing": "removed",
                "extra": "added",
                "fuzzy": "changed",
                "reparented": "moved",
            })
        """
        return cls(
            {
                "removed": ColorPair.auto("#9E6B6B"),
                "added": ColorPair.auto("#8E8555"),
                "changed": ColorPair.auto("#6B8C9E"),
                "moved": ColorPair.auto("#A87EC8"),
            }
        )
