# !/usr/bin/python
# coding=utf-8
"""Effect recipe -- how each render effect and audio clip is keyed, once per scene.

The one place the effects' settings live. The Render Effects and Audio Clips
panels edit it, the Shot Manifest's Build keys with it, and a writer called
from a script reads it by default -- so a highlight keyed by hand and one a
Build keys are the same pulse, and every default is declared here and nowhere
else.

The recipe rides the scene's shot store (:attr:`ShotStore.effect_recipe`): it
saves with the scene, crosses the Maya <-> Blender hand-off inside the store's
section, and is there for a headless build. It says HOW an effect is keyed,
never WHERE: a behavior template places an effect in a shot (``place``), a
panel keys at the playhead, and a length typed for one keying stays with the
panel that typed it.

Times are in the unit an artist reasons in. A fade is a frame count, as on
every sheet. The pulse is in SECONDS, because its cadence was measured in real
time on the WebXR reference and must read the same at any frame rate;
:meth:`EffectRecipe.pulse_cadence` converts once, for the writers, which take
frames.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, replace as _dataclass_replace
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

from pythontk.math_utils.ramp_keys import RampKeys

__all__ = ["EffectRecipe"]

RGB = Tuple[float, float, float]
Place = Union[str, float]


class _EffectRecipeInternal(object):
    """Internal helpers for EffectRecipe (field coercion)."""

    @staticmethod
    def _number(value: Any, default: float, lo: float, hi: float) -> float:
        """*value* as a float held to ``[lo, hi]``; *default* when it is not one."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(default)
        if math.isnan(number) or math.isinf(number):
            return float(default)
        return max(lo, min(hi, number))

    @staticmethod
    def _rgb(value: Any, default: RGB) -> RGB:
        """*value* as three non-negative floats; *default* when it is not.

        Not clamped to 1: an emissive colour may legitimately hold more (HDR).
        Rounded to 6 decimals, the precision the colour editors compare at, so
        a value that went through a picker reads back equal.
        """
        try:
            rgb = tuple(float(c) for c in tuple(value)[:3])
        except (TypeError, ValueError):
            return tuple(default)
        if len(rgb) != 3 or any(math.isnan(c) or math.isinf(c) for c in rgb):
            return tuple(default)
        return tuple(round(max(0.0, c), 6) for c in rgb)

    @staticmethod
    def _stamp_value(value: Any) -> Any:
        """*value* as it enters a fingerprint: floats at 6 decimals, so the
        float noise of a spin box never reads as a changed recipe."""
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, tuple):
            return [round(float(c), 6) for c in value]
        return value


@dataclass(frozen=True)
class EffectRecipe(_EffectRecipeInternal):
    """How each render effect and audio clip is keyed -- one per scene.

    A frozen value: change one through :meth:`replace` (or the store's
    ``update_effect_recipe``), so nothing edits the scene's recipe behind the
    store's back. Construction coerces every field and holds it to its range
    (:attr:`LIMITS`), so a recipe read from any scene is a usable one.
    """

    #: Frames a fade takes, in or out.
    fade_frames: float = 15.0
    #: One bright/dim cycle of the pulse, in seconds (2.86 s measured on the
    #: WebXR reference).
    pulse_period: float = 2.86
    #: Share of each cycle spent bright (59% measured on the reference).
    pulse_duty: float = 0.59
    #: Share of each cycle spent in EACH transition; the holds take the rest.
    pulse_ramp: float = 0.25
    #: Seconds the glow takes to come up at the start of a pulse. The default
    #: is one cycle's own transition (25% of 2.86 s), so the ends of the train
    #: read like every beat inside it.
    pulse_lead_in: float = 0.72
    #: Seconds it takes to fall away at the end.
    pulse_lead_out: float = 0.72
    #: The pulse's bright colour, read at intensity 1 -- LINEAR light, the
    #: space the attribute, the glTF factor and the Unity controller share.
    #: The linear value OF #0054DD (the production blue, adopted 2026-09-13),
    #: at the 6 decimals an 8-bit colour editor hands back, and what the
    #: channel's attribute preset seeds -- the DCC tests hold the three equal.
    pulse_bright: RGB = (0.0, 0.088656, 0.723055)
    #: The dim colour, read at intensity 0. Black: the glow fades to nothing,
    #: the look the channel had before it had two ends.
    pulse_dim: RGB = (0.0, 0.0, 0.0)
    #: Round audio key times to whole frames.
    audio_snap: bool = True

    #: The fields each effect is keyed from -- what its :meth:`fingerprint`
    #: covers. The pulse's colours are not among them: a Build writes them only
    #: when it creates the channel, so a revised object's colours survive a
    #: rebuild, and a changed colour is no reason to re-key.
    EFFECTS: ClassVar[Dict[str, Tuple[str, ...]]] = {
        "fade_in": ("fade_frames",),
        "fade_out": ("fade_frames",),
        "pulse": (
            "pulse_period",
            "pulse_duty",
            "pulse_ramp",
            "pulse_lead_in",
            "pulse_lead_out",
        ),
        "clip": ("audio_snap",),
    }
    #: Where a behavior template may place an effect in its shot: at the
    #: ``start``, at the ``end``, across it (``span``), or a fraction 0-1 of
    #: the way from the start placement to the end one.
    PLACES: ClassVar[Tuple[str, ...]] = ("start", "end", "span")
    #: Each numeric field's range.
    LIMITS: ClassVar[Dict[str, Tuple[float, float]]] = {
        "fade_frames": (1.0, 100000.0),
        "pulse_period": (0.05, 3600.0),
        "pulse_duty": (0.01, 0.99),
        "pulse_ramp": (0.0, 0.5),
        "pulse_lead_in": (0.0, 3600.0),
        "pulse_lead_out": (0.0, 3600.0),
    }
    #: The rate the pulse cadence was measured at. Only a reader that needs an
    #: effect's SHAPE and not its length (which channels, which phases) may
    #: stand on it; anything that sizes or keys passes the scene's rate.
    REFERENCE_FPS: ClassVar[float] = 30.0

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in self.LIMITS:
                lo, hi = self.LIMITS[f.name]
                value = self._number(value, f.default, lo, hi)
            elif isinstance(f.default, tuple):
                value = self._rgb(value, f.default)
            elif isinstance(f.default, bool):
                value = bool(value)
            object.__setattr__(self, f.name, value)

    # ---- serialisation ---------------------------------------------------

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EffectRecipe":
        """A recipe from :meth:`to_dict`'s shape.

        Tolerant, as a scene record must be: a missing or foreign payload is
        the defaults, an unknown key is ignored, and a bad value falls back to
        its field's default.
        """
        if not isinstance(data, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict form for the scene payload (colours as lists)."""
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out

    def replace(self, **changes: Any) -> "EffectRecipe":
        """A copy with *changes* applied (coerced like a new recipe).

        Raises:
            TypeError: A name that is not a field.
        """
        unknown = set(changes) - {f.name for f in fields(self)}
        if unknown:
            raise TypeError(
                f"EffectRecipe has no field(s) {sorted(unknown)}; "
                f"known: {[f.name for f in fields(self)]}"
            )
        return _dataclass_replace(self, **changes)

    # ---- identity --------------------------------------------------------

    @classmethod
    def effect_names(cls) -> List[str]:
        """Every effect a recipe keys, in declaration order."""
        return list(cls.EFFECTS)

    def fingerprint(self, effect: str) -> str:
        """A short stable stamp of what *effect* is keyed from.

        Recorded beside the keys a Build writes, so Assess can tell keys made
        under an older recipe from current ones. Two recipes that key
        *effect* the same way stamp the same, whatever their other fields.

        Raises:
            ValueError: An effect the recipe does not know.
        """
        names = self._fields_of(effect)
        payload = json.dumps(
            {name: self._stamp_value(getattr(self, name)) for name in names},
            sort_keys=True,
        )
        return hashlib.sha1(f"{effect}:{payload}".encode("utf-8")).hexdigest()[:10]

    @classmethod
    def _fields_of(cls, effect: str) -> Tuple[str, ...]:
        try:
            return cls.EFFECTS[effect]
        except KeyError:
            raise ValueError(
                f"Unknown effect {effect!r}; known: {', '.join(cls.EFFECTS)}."
            ) from None

    # ---- keying ----------------------------------------------------------

    @property
    def colors(self) -> Tuple[RGB, RGB]:
        """``(bright, dim)`` -- the pulse's colour ramp."""
        return self.pulse_bright, self.pulse_dim

    def pulse_cadence(self, fps: float) -> Dict[str, float]:
        """The pulse as ``RampKeys.pulse`` keyword arguments, in FRAMES at *fps*.

        The one seconds-to-frames conversion: the writers, the previews and
        the manifest all take their cadence from here, so none can mean a
        different pulse.
        """
        fps = float(fps)
        return {
            "period": self.pulse_period * fps,
            "bright_fraction": self.pulse_duty,
            "ramp_fraction": self.pulse_ramp,
            "lead_in": self.pulse_lead_in * fps,
            "lead_out": self.pulse_lead_out * fps,
        }

    def plan(
        self,
        effect: str,
        start: float,
        end: float,
        fps: float,
        whole_frames: bool = True,
    ) -> List[Tuple[float, float]]:
        """The ``(frame, value)`` keys *effect* writes over ``start..end``.

        Parameters:
            effect: ``"fade_in"``, ``"fade_out"`` or ``"pulse"``.
            start: First frame of the effect's window (see :meth:`window`).
            end: Last frame.
            fps: The scene's rate -- the pulse's seconds become frames at it.
            whole_frames: Snap the keys to whole frames (the default).

        Raises:
            ValueError: For an effect with no key plan (``"clip"`` keys an
                audio track, not a ramp).
        """
        if effect in ("fade_in", "fade_out"):
            return RampKeys.fade(
                start, end, "in" if effect == "fade_in" else "out", whole_frames
            )
        if effect == "pulse":
            return RampKeys.pulse(
                start, end, whole_frames=whole_frames, **self.pulse_cadence(fps)
            )
        self._fields_of(effect)  # an unknown name says so
        raise ValueError(f"The {effect!r} effect keys no ramp.")

    def window(
        self,
        effect: str,
        start: float,
        end: float,
        place: Place = "start",
        anchor: Optional[float] = None,
    ) -> Tuple[float, float]:
        """The frames *effect* occupies when placed in the range ``start..end``.

        A fade is :attr:`fade_frames` long, set at the range's start, its end,
        or a fraction of the way between those two placements -- *anchor*,
        when given, overrides *place* the way a behavior's anchor override
        does (the manifest spreads an object's point behaviors in doc order).
        The pulse and the clip span the range (the clip's own length sets its
        end; the audio writer measures it).
        """
        start, end = float(start), float(end)
        if effect not in ("fade_in", "fade_out"):
            self._fields_of(effect)
            return start, end
        length = self.fade_frames
        at: Place = anchor if anchor is not None else place
        if isinstance(at, (int, float)) and not isinstance(at, bool):
            base = start + float(at) * ((end - length) - start)
        elif at == "end":
            base = end - length
        else:
            base = start
        return base, base + length

    def envelope(
        self, effect: str, place: Place = "start", fps: Optional[float] = None
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """*effect* as a behavior template's ``attributes`` block.

        What every reader of a template that does not key -- shot sizing,
        Assess's verification, which channels the effect owns, whether it is a
        point or a span -- needs to know about it, stated in the shape those
        readers already speak; the keys themselves come from :meth:`plan`.

        A fade is one phase on ``visibility`` (the presence channel, which the
        appliers key as opacity mirrored to visibility). The pulse is a span on
        ``highlight``: up at the start and down at the end, its lead-in phase
        long enough for one whole cycle, so a shot sized to fit it holds a
        full beat. The clip has none (its length is its source's).

        Parameters:
            effect: An effect name (:attr:`EFFECTS`).
            place: The template's placement; an envelope anchors where the
                effect is placed.
            fps: The scene's rate, for the pulse's lengths. Only a reader of
                the SHAPE may leave it to :attr:`REFERENCE_FPS`.
        """
        self._fields_of(effect)
        fps = float(fps or self.REFERENCE_FPS)
        if effect in ("fade_in", "fade_out"):
            phase, values = (
                ("in", [0.0, 1.0]) if effect == "fade_in" else ("out", [1.0, 0.0])
            )
            numeric = isinstance(place, (int, float)) and not isinstance(place, bool)
            anchor = (
                place
                if numeric or place in ("start", "end")
                else ("start" if phase == "in" else "end")
            )
            block = {
                "anchor": anchor,
                "offset": 0,
                "duration": self.fade_frames,
                "values": values,
                "tangent": "linear",
            }
            return {"visibility": {phase: block}}
        if effect == "pulse":
            return {
                "highlight": {
                    "in": {
                        "anchor": "start",
                        "offset": 0,
                        "duration": (self.pulse_lead_in + self.pulse_period) * fps,
                        "values": [0.0, 1.0],
                        "tangent": "linear",
                    },
                    "out": {
                        "anchor": "end",
                        "offset": 0,
                        "duration": self.pulse_lead_out * fps,
                        "values": [1.0, 0.0],
                        "tangent": "linear",
                    },
                }
            }
        return {}

    def pulse_cycles(self, length: float) -> float:
        """How many cycles a pulse *length* seconds long holds.

        The train between the leads, fitted as ``RampKeys.pulse`` fits them
        (``pulse_gaps``): counting the whole length overstated it -- at the
        defaults a 4 s pulse holds one beat, not 1.4.
        """
        length = max(0.0, float(length))
        head, tail = RampKeys.pulse_gaps(
            0.0, length, 0.0, self.pulse_lead_in, self.pulse_lead_out, gap_min=0.0
        )
        train = max(0.0, length - head - tail)
        return train / self.pulse_period if self.pulse_period > 0 else 0.0
