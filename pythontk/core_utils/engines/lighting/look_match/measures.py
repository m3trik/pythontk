# !/usr/bin/python
# coding=utf-8
"""What a look match is scored on: measures taken on both frames, region by region.

Every measure reduces each side to a statistic and compares them as a signed
score (0 = match) plus the residuals a fit drives to zero. Magnitudes compare
as log ratios (0.1 is about a 10% miss, either way); colour compares as
differences in OKLab. Frames are display images (sRGB-encoded 0..1) of one size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None  # type: ignore

from pythontk.img_utils._img_utils import ImgUtils
from pythontk.core_utils.engines.lighting.look_match.target import (
    LookRegion,
    LookTarget,
)

_luma = ImgUtils.luminance


@dataclass
class MeasureResult:
    """One measure on one region: each side's statistic, the score, the residuals.

    Attributes:
        name: The measure.
        ours: Our side's statistic (a number, or an array for a profile).
        ref: The reference's.
        score: One signed number, 0 = a match (a log ratio, or a difference in ``unit``).
        residuals: What a fit drives to zero, scaled so 0.1 is a noticeable miss.
        unit: What ``score`` is in.
    """

    name: str
    ours: Any
    ref: Any
    score: float
    residuals: "np.ndarray" = field(repr=False)
    unit: str = ""

    @property
    def miss(self) -> float:
        """The residual furthest from zero (0 with none): the measure is within
        a tolerance when this is no more than it (:meth:`LookScore.summary`)."""
        return float(np.max(np.abs(self.residuals))) if self.residuals.size else 0.0


@dataclass
class _Measure:
    kind: str
    fn: Callable[..., Tuple[Any, Any]]
    compare: str
    eps: float
    scale: float
    unit: str


class LookMeasures:
    """A registry of measures: ``name -> (kind, statistic, comparison)``.

    A statistic is ``fn(ours, ref, **context) -> (ours_stat, ref_stat)``; what it
    receives depends on the kind:

    - ``area``: the region's pixels on each side, ``(N, 3)`` display values.
    - ``line``: the samples along the line, ``(n, 3)``.
    - ``quad``: ``(grid, keep)`` per side -- the rectified ``(rows, columns, 3)``
      values and which cells count.
    - ``frame``: the whole frames, with ``ours_focus`` / ``ref_focus`` masks in
      the context (``None`` when the target has no focus).

    The comparison turns the two statistics into the score and residuals:
    ``"log_ratio"`` (``log((o + eps) / (r + eps))``, element-wise for arrays),
    ``"difference"`` (``o - r``) or ``"angle"`` (a hue difference in degrees).
    ``scale`` divides the residuals.
    """

    #: The measures a region of each kind takes when it names none.
    DEFAULTS = {
        "area": ("mean", "peak", "chroma", "warmth"),
        "line": ("profile", "peak_floor"),
        "quad": ("columns", "rows"),
        "frame": ("median", "p95", "focus_share"),
    }

    def __init__(self):
        self._measures: Dict[str, _Measure] = {}

    def register(
        self,
        name: str,
        kind: str,
        fn: Callable[..., Tuple[Any, Any]],
        compare: str = "log_ratio",
        eps: float = 1 / 255,
        scale: float = 1.0,
        unit: str = "log",
    ) -> None:
        """Add (or replace) a measure."""
        if compare not in ("log_ratio", "difference", "angle"):
            raise ValueError(f"Unknown comparison {compare!r}.")
        self._measures[name] = _Measure(kind, fn, compare, eps, scale, unit)

    def names(self, kind: Optional[str] = None) -> List[str]:
        return sorted(
            n for n, m in self._measures.items() if kind is None or m.kind == kind
        )

    def kind(self, name: str) -> str:
        return self._get(name).kind

    def _get(self, name: str) -> _Measure:
        try:
            return self._measures[name]
        except KeyError:
            raise KeyError(f"No measure {name!r}; measures: {self.names()}") from None

    def measure(self, name: str, ours, ref, **context) -> MeasureResult:
        """Take measure *name* on both sides and compare them."""
        m = self._get(name)
        o, r = m.fn(ours, ref, **context)
        if m.compare == "log_ratio":
            o_arr, r_arr = np.asarray(o, np.float64), np.asarray(r, np.float64)
            res = np.log((o_arr + m.eps) / (r_arr + m.eps))
            score = float(np.mean(res)) if res.size else float("nan")
        elif m.compare == "angle":
            diff = (float(o) - float(r) + 180.0) % 360.0 - 180.0
            weight = float(context.get("hue_weight", 1.0))
            res = np.asarray([diff * weight])
            score = diff
        else:
            res = np.asarray(o, np.float64) - np.asarray(r, np.float64)
            score = float(np.mean(res)) if res.size else float("nan")
        res = np.atleast_1d(res).astype(np.float64) / m.scale
        return MeasureResult(name, o, r, score, res, m.unit)

    # ------------------------------------------------------------------ built-ins
    @classmethod
    def default(cls) -> "LookMeasures":
        """The built-in measures."""
        measures = cls()

        def mean(o, r, **_):
            return float(_luma(o).mean()), float(_luma(r).mean())

        def top_mean(values, share=0.01):
            luma = np.sort(_luma(values))
            return float(luma[-max(1, int(round(len(luma) * share))) :].mean())

        def peak(o, r, **_):
            return top_mean(o), top_mean(r)

        def footprint(o, r, **_):
            level = 0.75 * top_mean(r)
            return float((_luma(o) >= level).mean()), float((_luma(r) >= level).mean())

        def lab(values, given=None):
            if given is not None:  # the region's mean colour, converted once per score
                return given
            return ImgUtils.linear_to_oklab(
                ImgUtils.srgb_to_linear(np.asarray(values, np.float32)).mean(axis=0)
            )

        def chroma(o, r, ours_lab=None, ref_lab=None, **_):
            lo, lr = lab(o, ours_lab), lab(r, ref_lab)
            return float(math.hypot(lo[1], lo[2])), float(math.hypot(lr[1], lr[2]))

        def hue(o, r, ours_lab=None, ref_lab=None, **_):
            lo, lr = lab(o, ours_lab), lab(r, ref_lab)
            return math.degrees(math.atan2(lo[2], lo[1])), math.degrees(
                math.atan2(lr[2], lr[1])
            )

        def warmth(o, r, ours_lab=None, ref_lab=None, **_):
            return float(lab(o, ours_lab)[2]) * 100.0, float(lab(r, ref_lab)[2]) * 100.0

        def profile(o, r, **_):
            return _luma(o), _luma(r)

        def peak_floor(o, r, **_):
            def ratio(values):
                luma = _luma(values)
                return float(
                    (np.percentile(luma, 95) + 1 / 255)
                    / (np.percentile(luma, 5) + 1 / 255)
                )

            return ratio(o), ratio(r)

        def masked_medians(grid, keep, along_columns):
            """Per column (or row): the median of its kept cells, and the share kept."""
            luma = np.where(keep, _luma(grid), np.nan)
            axis = 0 if along_columns else 1
            share = keep.mean(axis=axis)
            with np.errstate(all="ignore"):
                medians = np.nanmedian(
                    np.where(keep.any(axis=axis, keepdims=True), luma, 0.0), axis=axis
                )
            return np.nan_to_num(medians), share

        def lines(o, r, along_columns):
            (go, ko), (gr, kr) = o, r
            mo, so = masked_medians(go, ko, along_columns)
            mr, sr = masked_medians(gr, kr, along_columns)
            usable = (so >= 0.5) & (
                sr >= 0.5
            )  # a line counts where both sides keep half its cells
            return mo[usable], mr[usable]

        def columns(o, r, **_):
            return lines(o, r, True)

        def rows(o, r, **_):
            return lines(o, r, False)

        def lumas(o, r, ours_luma=None, ref_luma=None):
            return (_luma(o) if ours_luma is None else ours_luma), (
                _luma(r) if ref_luma is None else ref_luma
            )

        def frame_median(o, r, ours_luma=None, ref_luma=None, **_):
            lo, lr = lumas(o, r, ours_luma, ref_luma)
            return float(np.median(lo)), float(np.median(lr))

        def frame_p95(o, r, ours_luma=None, ref_luma=None, **_):
            lo, lr = lumas(o, r, ours_luma, ref_luma)
            return float(np.percentile(lo, 95)), float(np.percentile(lr, 95))

        def focus_share(
            o, r, ours_focus=None, ref_focus=None, ours_luma=None, ref_luma=None, **_
        ):
            def share(luma, focus):
                bright = luma >= np.percentile(luma, 95)
                return float((bright & focus).sum() / max(bright.sum(), 1))

            lo, lr = lumas(o, r, ours_luma, ref_luma)
            return share(lo, ours_focus), share(lr, ref_focus)

        measures.register("mean", "area", mean)
        measures.register("peak", "area", peak)
        measures.register("footprint", "area", footprint, eps=0.005)
        measures.register("chroma", "area", chroma, eps=0.005)
        measures.register(
            "hue", "area", hue, compare="angle", scale=30.0, unit="degrees"
        )
        measures.register(
            "warmth",
            "area",
            warmth,
            compare="difference",
            scale=5.0,
            unit="OKLab b x100",
        )
        measures.register("profile", "line", profile)
        measures.register("peak_floor", "line", peak_floor, eps=0.0)
        measures.register("columns", "quad", columns)
        measures.register("rows", "quad", rows)
        measures.register("median", "frame", frame_median)
        measures.register("p95", "frame", frame_p95)
        measures.register("focus_share", "frame", focus_share, eps=0.02)
        return measures


class LookScore:
    """A target's measures taken on our frame and the reference's.

    Parameters:
        ours: Our display frame (sRGB-encoded 0..1, ``(height, width, 3)``).
        ref: The reference at the same size.
        target: The regions to measure.
        measures: The registry (default: :meth:`LookMeasures.default`).
        frame_weight: How much the frame measures count against the regions.

    Raises:
        ValueError: the frames differ in size.
        KeyError: a region names a measure the registry lacks.
    """

    def __init__(
        self,
        ours: "np.ndarray",
        ref: "np.ndarray",
        target: LookTarget,
        measures: Optional[LookMeasures] = None,
        frame_weight: float = 1.0,
    ):
        if ours.shape != ref.shape:
            raise ValueError(
                f"Frames differ in size: ours {ours.shape}, reference {ref.shape}."
            )
        self.target = target
        self.measures = measures or LookMeasures.default()
        self.frame_weight = float(frame_weight)
        self.size = (ours.shape[1], ours.shape[0])
        self._regions: Dict[str, Dict[str, MeasureResult]] = {}
        # The regions as scored, by the name they had: the target may change
        # after (a panel renames or adds one while an analysis runs), and a
        # reading of this score reads what it measured.
        self._scored: Dict[str, LookRegion] = {}
        for region in target.regions:
            self._scored[region.name] = region
            self._regions[region.name] = self._measure_region(region, ours, ref)
        self._frame = self._measure_frame(ours, ref)

    def _measure_region(self, region, ours, ref) -> Dict[str, MeasureResult]:
        names = (
            region.measures
            if region.measures is not None
            else LookMeasures.DEFAULTS[region.kind]
        )
        for name in names:
            if self.measures.kind(name) != region.kind:
                raise ValueError(
                    f"Region {region.name!r} is a {region.kind}; measure {name!r} is for a {self.measures.kind(name)}."
                )
        o, r = region.sample(ours, "ours"), region.sample(ref, "ref")
        context = {}
        if region.kind == "area":
            if not len(o) or not len(r):
                return {}
            lab_needed = any(n in ("chroma", "hue", "warmth") for n in names)
            ours_lab = (
                ImgUtils.linear_to_oklab(ImgUtils.srgb_to_linear(o).mean(axis=0))
                if lab_needed
                else None
            )
            ref_lab = (
                ImgUtils.linear_to_oklab(ImgUtils.srgb_to_linear(r).mean(axis=0))
                if lab_needed
                else None
            )
            context = {
                "hue_weight": min(1.0, math.hypot(ref_lab[1], ref_lab[2]) / 0.03)
                if lab_needed
                else 1.0,
                "ours_lab": ours_lab,
                "ref_lab": ref_lab,
            }
        return {name: self.measures.measure(name, o, r, **context) for name in names}

    def _measure_frame(self, ours, ref) -> Dict[str, MeasureResult]:
        out = {}
        flat_o, flat_r = ours.reshape(-1, 3), ref.reshape(-1, 3)
        shared = {
            "ours_luma": _luma(flat_o),
            "ref_luma": _luma(flat_r),
        }  # once for every frame measure
        for name in LookMeasures.DEFAULTS["frame"]:
            if name == "focus_share":
                if not self.target.focus:
                    continue
                focus = self.target.region(self.target.focus)
                context = {
                    "ours_focus": focus.mask("ours", self.size).ravel(),
                    "ref_focus": focus.mask("ref", self.size).ravel(),
                }
            else:
                context = {}
            out[name] = self.measures.measure(name, flat_o, flat_r, **shared, **context)
        return out

    # ------------------------------------------------------------------ reading
    def region(self, name: str) -> Dict[str, MeasureResult]:
        """``{measure: MeasureResult}`` for region *name*."""
        try:
            return self._regions[name]
        except KeyError:
            raise KeyError(
                f"No region {name!r} scored; regions: {sorted(self._regions)}"
            ) from None

    def frame(self) -> Dict[str, MeasureResult]:
        return dict(self._frame)

    def residuals(
        self, tolerance: float = 0.1, weights: Optional[Mapping[str, float]] = None
    ) -> "np.ndarray":
        """Every residual, weighted, inside a tolerance band (a miss under it counts 0).

        Parameters:
            tolerance: The band, in residual units (0.1 is about a 10% miss).
            weights: ``{region: weight}`` overriding the regions' own; the key
                ``"(frame)"`` weighs the frame measures.
        """
        weights = dict(weights or {})
        parts = []
        for name, region in self._scored.items():
            w = weights.get(name, region.weight)
            for result in self._regions[name].values():
                parts.append(w * self._soft(result.residuals, tolerance))
        w = weights.get("(frame)", self.frame_weight)
        for result in self._frame.values():
            parts.append(w * self._soft(result.residuals, tolerance))
        return np.concatenate(parts) if parts else np.zeros(0)

    @staticmethod
    def _soft(res: "np.ndarray", tolerance: float) -> "np.ndarray":
        return np.sign(res) * np.maximum(np.abs(res) - tolerance, 0.0)

    def cost(self, tolerance: float = 0.1) -> float:
        """Half the sum of squared residuals (what a fit minimises)."""
        r = self.residuals(tolerance)
        return 0.5 * float(r @ r)

    #: The measures a region's brightness reads from, the first it takes.
    BRIGHTNESS = ("mean", "profile", "columns", "rows")
    #: The measures a colour word reads from: ``{measure: (word over, word under)}``.
    COLOUR_WORDS = {"warmth": ("warm", "cool"), "chroma": ("vivid", "dull")}

    def summary(self, tolerance: float = 0.1) -> List[Dict[str, Any]]:
        """One row per region, as a person reads it: ``{"region", "kind",
        "within", "worst", "miss", "brightness", "colour"}``.

        ``within``: every measure's residuals inside *tolerance* (``None``
        for a region with nothing to measure); ``worst`` / ``miss``: the
        measure furthest out and by how much (residual units); ``brightness``:
        how much brighter on screen (display values, after the tone map, not
        light: +0.4 reads 40% brighter, -0.3 30% darker; ``None`` when no
        brightness measure ran); ``colour``: words for a colour that is off
        (``"warm"``, ``"cool"``, ``"vivid"``, ``"dull"``), comma-joined.
        """
        out = []
        for region_name, region in self._scored.items():
            results = self._regions[region_name]
            worst, miss = None, 0.0
            for name, result in results.items():
                if result.miss > miss:
                    worst, miss = name, result.miss
            level = next((results[n] for n in self.BRIGHTNESS if n in results), None)
            words = [
                over if results[name].score > 0 else under
                for name, (over, under) in self.COLOUR_WORDS.items()
                if name in results
                and results[name].residuals.size
                and abs(float(np.mean(results[name].residuals))) > tolerance
            ]
            out.append(
                {
                    "region": region_name,
                    "kind": region.kind,
                    "within": None if not results else miss <= tolerance,
                    "worst": worst,
                    "miss": miss,
                    "brightness": math.expm1(level.score)
                    if level is not None and math.isfinite(level.score)
                    else None,
                    "colour": ", ".join(words),
                }
            )
        return out

    def rows(self) -> List[Dict[str, Any]]:
        """One row per region and measure, then the frame's: for a table or a sheet."""
        out = []
        for region_name, results in self._regions.items():
            for result in results.values():
                out.append(self._row(region_name, result))
        for result in self._frame.values():
            out.append(self._row("(frame)", result))
        return out

    @staticmethod
    def _row(region: str, result: MeasureResult) -> Dict[str, Any]:
        def summary(value):  # a number, or an array's mean (a 1-element one too)
            arr = np.asarray(value, np.float64)
            return float(arr.mean()) if arr.size else float("nan")

        return {
            "region": region,
            "measure": result.name,
            "ours": summary(result.ours),
            "ref": summary(result.ref),
            "score": result.score,
            "unit": result.unit,
        }
