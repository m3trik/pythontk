# !/usr/bin/python
# coding=utf-8
import math
import random
from typing import List, Tuple


class BandLimitedNoise:
    """Coherent, band-limited 2D noise built from summed sine octaves.

    A handful of smooth wave octaves — frequencies climbing per octave, with the
    amplitude rolling off ``~1/f`` — summed into a continuous, zero-mean field.
    Unlike per-vertex white noise (which reads as static), this is spatially
    smooth, so it suits procedural surface relief: cloth grain, gentle terrain,
    displacement, etc. Deterministic from ``seed``.

    The horizontal (``u``) frequency leads the vertical (``v``) one, so by
    default the grain is mildly anisotropic — useful where relief should follow
    a dominant direction (e.g. drape-following wrinkles). Set ``u_periodic`` to
    snap the ``u`` frequencies to whole cycles so the field wraps seamlessly
    across the ``u = 0 / u = 1`` seam (rings, tiling).

    Parameters:
        seed: RNG seed (deterministic output).
        octaves: Number of summed wave octaves (the band limit).
        falloff: Per-octave amplitude multiplier (``~1/f`` character).
        u_freq: ``(min, max)`` cycles-per-unit multiplier for the ``u`` axis.
        v_freq: ``(min, max)`` frequency multiplier for the ``v`` axis.
        u_periodic: Snap ``u`` frequencies to whole cycles (seamless ``u`` wrap).

    Example:
        n = BandLimitedNoise(seed=0)
        n.at(0.5, 0.5)  # -> a smooth value in roughly [-1, 1]
    """

    def __init__(
        self,
        seed: int = 0,
        octaves: int = 4,
        falloff: float = 0.55,
        u_freq: Tuple[float, float] = (0.8, 1.4),
        v_freq: Tuple[float, float] = (0.5, 1.1),
        u_periodic: bool = False,
    ):
        rng = random.Random(seed)
        comps: List[Tuple[float, float, float, float, float]] = []
        amp, total = 1.0, 0.0
        for octave in range(max(1, int(octaves))):
            cycles = (octave + 1) * rng.uniform(*u_freq)
            if u_periodic:  # whole cycles -> seamless across the u = 0/1 seam
                cycles = max(1.0, float(round(cycles)))
            fu = cycles * 2.0 * math.pi
            fv = (octave + 1) * math.pi * rng.uniform(*v_freq)
            comps.append(
                (amp, fu, rng.uniform(0.0, 2.0 * math.pi),
                 fv, rng.uniform(0.0, 2.0 * math.pi))
            )
            total += amp
            amp *= falloff
        # Normalize so the summed amplitude is 1 (output magnitude stays
        # ~independent of octave count / falloff).
        self._comps = [(a / total, fu, pu, fv, pv) for a, fu, pu, fv, pv in comps]

    def at(self, u: float, v: float) -> float:
        """Noise value at ``(u, v)`` over the unit square (≈ ``[-1, 1]``)."""
        n = 0.0
        for amp, fu, pu, fv, pv in self._comps:
            n += amp * math.sin(fu * u + pu) * math.sin(fv * v + pv)
        return n


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
