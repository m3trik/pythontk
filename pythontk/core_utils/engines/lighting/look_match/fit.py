# !/usr/bin/python
# coding=utf-8
"""Fit layer weights so a mix's frame scores best against a target."""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None  # type: ignore

from pythontk.core_utils.engines.lighting.light_layers import LightMix
from pythontk.core_utils.engines.lighting.light_predictor import LightPredictor
from pythontk.core_utils.engines.lighting.look_match.measures import (
    LookMeasures,
    LookScore,
)
from pythontk.core_utils.engines.lighting.look_match.target import LookTarget
from pythontk.img_utils._img_utils import ImgUtils


@dataclass
class FitResult:
    """What a fit did.

    Attributes:
        mix: The fitted mix.
        cost_before: The objective at the start.
        cost_after: At the end.
        iterations: Accepted steps.
        evaluations: Frames rendered.
        changes: ``{layer: (before, after)}`` weight, for every free layer.
        seconds: Wall-clock time.
        kelvins: ``{layer: (before, after)}`` colour temperature, for every
            layer whose colour was fitted.
    """

    mix: LightMix
    cost_before: float
    cost_after: float
    iterations: int
    evaluations: int
    changes: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    seconds: float = 0.0
    kelvins: Dict[str, Tuple[Optional[float], float]] = field(default_factory=dict)


class LookFit:
    """Fits free layer weights to a target's measures.

    The objective is the target's residuals (:meth:`LookScore.residuals`, a
    tolerance band so a region already close is not traded away) plus
    ``regularize * log(w / w_start)`` per free layer: changes stay small and
    deliberate, and a layer no measure sees keeps its weight. Weights are
    solved in log space -- positive, with multiplicative steps -- by
    Levenberg-Marquardt within *bounds*, on a basis rendered once (the fixed
    layers' sum and each free layer at unit weight), so an evaluation is a sum
    and a tone map.

    Parameters:
        predictor: Renders the frames.
        target: What is scored.
        size: ``(width, height)`` to fit at (default: the store's; smaller is faster).
        measures: The measure registry.
        tolerance: The residual band (0.1 is about a 10% miss).
        regularize: Pull toward the starting weights.
        bounds: ``(low, high)`` for every free weight.
        region_weights: ``{region: weight}`` overriding the target's.
    """

    def __init__(
        self,
        predictor: LightPredictor,
        target: LookTarget,
        size: Optional[Tuple[int, int]] = None,
        measures: Optional[LookMeasures] = None,
        tolerance: float = 0.1,
        regularize: float = 0.3,
        bounds: Tuple[float, float] = (1e-4, 1e3),
        region_weights: Optional[Mapping[str, float]] = None,
    ):
        self.predictor = predictor
        self.target = target
        self.size = tuple(size) if size else predictor.layers.size
        self.measures = measures or LookMeasures.default()
        self.tolerance = float(tolerance)
        self.regularize = float(regularize)
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.region_weights = dict(region_weights or {})

    #: Colour temperatures a fit may reach (kelvin), and its step in mireds.
    KELVIN_BOUNDS = (1800.0, 12000.0)
    MIRED_STEP = 4.0
    #: How far a mired is from a log-weight step, for the pull toward the start.
    MIRED_SCALE = 50.0

    def run(
        self,
        mix: LightMix,
        free: Sequence[str],
        max_iter: int = 30,
        step: float = 0.05,
        kelvin: Sequence[str] = (),
    ) -> FitResult:
        """Fit *free* layers' weights -- and *kelvin* layers' colour temperatures --
        starting from *mix*.

        A colour temperature is solved in mireds (1e6 / kelvin: even steps of
        perceived colour) within :attr:`KELVIN_BOUNDS`. Every varied layer is
        held white in the basis and tinted per evaluation (a tint is per
        channel), so a colour costs no more than a weight.

        Parameters:
            mix: The start; layers not in *free* keep their weights, and not in
                *kelvin* their colour.
            free: The layers whose weight to fit (each must be in the store).
            max_iter: The most steps to try.
            step: The finite-difference step, in log weight (0.05 ~ 5%).
            kelvin: Light layers whose colour temperature to fit.

        Raises:
            KeyError: a layer the store lacks.
        """
        t0 = time.time()
        free, kelvin = list(free), list(kelvin)
        varied = list(dict.fromkeys(free + kelvin))
        for name in varied:
            self.predictor.layers.info(name)
        start = mix.with_entries(
            {n: {"weight": max(mix.weight(n), self.bounds[0])} for n in free}
        )
        k_lo, k_hi = self.KELVIN_BOUNDS
        for name in kelvin:
            if not start.kelvin(name):
                start = start.with_entries({name: {"kelvin": 6500.0}})
        fixed = self.predictor.linear(
            start, self.size, names=[n for n in start.active() if n not in varied]
        )
        # each varied layer at unit weight, untinted: weight and tint per evaluation
        white = (
            np.stack([self.predictor.layers.load(n, self.size) for n in varied])
            if varied
            else np.zeros((0,) + fixed.shape, np.float32)
        )
        tinted = [self.predictor.layers.kind(n).tinted for n in varied]
        ref = self.target.reference_display(self.size)
        lo, hi = math.log(self.bounds[0]), math.log(self.bounds[1])
        m_lo, m_hi = 1e6 / k_hi, 1e6 / k_lo
        nw = len(free)
        x0 = np.concatenate(
            [
                np.clip(np.log([start.weight(n) for n in free]), lo, hi),
                np.clip([1e6 / start.kelvin(n) for n in kelvin], m_lo, m_hi),
            ]
        ).astype(np.float64)
        lower = np.concatenate([np.full(nw, lo), np.full(len(kelvin), m_lo)])
        upper = np.concatenate([np.full(nw, hi), np.full(len(kelvin), m_hi)])
        steps = np.concatenate(
            [np.full(nw, step), np.full(len(kelvin), self.MIRED_STEP)]
        )
        pull = np.concatenate(
            [
                np.full(nw, self.regularize),
                np.full(len(kelvin), self.regularize / self.MIRED_SCALE),
            ]
        )

        def coefficients(x: "np.ndarray") -> "np.ndarray":
            """``(len(varied), 3)``: each varied layer's weight times its tint."""
            out = np.empty((len(varied), 3), np.float32)
            for i, name in enumerate(varied):
                weight = (
                    math.exp(x[free.index(name)])
                    if name in free
                    else start.weight(name)
                )
                if not tinted[i]:
                    tint = np.ones(3, np.float32)
                elif name in kelvin:
                    tint = ImgUtils.kelvin_tint(1e6 / x[nw + kelvin.index(name)])
                else:
                    tint = ImgUtils.kelvin_tint(start.kelvin(name))
                out[i] = weight * np.asarray(tint, np.float32)
            return out

        def residuals(x: "np.ndarray") -> "np.ndarray":
            if varied:
                c = coefficients(x)
                # per channel, float32: a float64 coefficient promotes the stack
                linear = fixed + np.stack(
                    [
                        np.tensordot(c[:, k], white[..., k], axes=(0, 0))
                        for k in range(3)
                    ],
                    axis=-1,
                )
            else:
                linear = fixed
            frame = self.predictor.transform.display(linear, start.exposure)
            score = LookScore(frame, ref, self.target, self.measures)
            return np.concatenate(
                [score.residuals(self.tolerance, self.region_weights), pull * (x - x0)]
            )

        def column(args):
            x_, i = args
            xi = x_.copy()
            xi[i] += steps[i]
            return residuals(xi)

        x = x0.copy()
        r = residuals(x)
        evaluations = 1
        cost = cost_before = 0.5 * float(r @ r)
        lam, iterations = 1e-2, 0
        n = len(x)
        # The Jacobian's columns are independent frames: on threads (numpy
        # releases the GIL in its array work), one per parameter.
        workers = max(1, min(n, os.cpu_count() or 1))
        pool = ThreadPoolExecutor(max_workers=workers) if n > 1 else None
        try:
            for _ in range(max_iter if n else 0):
                shifted = (
                    list(pool.map(column, [(x, i) for i in range(n)]))
                    if pool
                    else [column((x, i)) for i in range(n)]
                )
                evaluations += n
                jac = np.stack(
                    [(r_i - r) / steps[i] for i, r_i in enumerate(shifted)], axis=1
                )
                jtj, jtr = jac.T @ jac, jac.T @ r
                improved = False
                for _ in range(8):
                    delta = np.linalg.solve(
                        jtj + lam * np.diag(np.diag(jtj) + 1e-9) + 1e-12 * np.eye(n),
                        -jtr,
                    )
                    trial = np.clip(x + delta, lower, upper)
                    r_trial = residuals(trial)
                    evaluations += 1
                    cost_trial = 0.5 * float(r_trial @ r_trial)
                    if cost_trial < cost:
                        improved = True
                        lam = max(lam * 0.3, 1e-7)
                        break
                    lam *= 4.0
                if not improved:
                    break
                converged = (
                    cost - cost_trial < 1e-6 * max(cost, 1e-12)
                    or float(np.abs((trial - x) / steps).max()) < 2e-3
                )
                x, r, cost = trial, r_trial, cost_trial
                iterations += 1
                if converged:
                    break
        finally:
            if pool:
                pool.shutdown()
        weights = {name: float(math.exp(v)) for name, v in zip(free, x[:nw])}
        kelvins = {name: float(1e6 / v) for name, v in zip(kelvin, x[nw:])}
        changes = {n: {"weight": w} for n, w in weights.items()}
        for name, k in kelvins.items():
            changes.setdefault(name, {})["kelvin"] = round(k, 1)
        return FitResult(
            mix=start.with_entries(changes),
            cost_before=cost_before,
            cost_after=cost,
            iterations=iterations,
            evaluations=evaluations,
            changes={n: (mix.weight(n), weights[n]) for n in free},
            seconds=time.time() - t0,
            kelvins={n: (mix.kelvin(n), round(kelvins[n], 1)) for n in kelvin},
        )
