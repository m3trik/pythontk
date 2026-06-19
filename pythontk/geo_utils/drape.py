# !/usr/bin/python
# coding=utf-8
"""Procedural draped-cloth (curtain) generator — pure geometry, no DCC.

A curtain hangs from a *rail* — any polyline in world space (see
:class:`~pythontk.geo_utils.polyline.Polyline`). The cloth is pinned to the rail
at evenly-spaced **hanging points**; each hanging point is a pleat (the fabric
gathers there), and between consecutive points the fabric bellies into a fold and
its top edge sags under gravity along a real **catenary** (``y = a·cosh(x/a)``).

This module is the ecosystem SSoT for the drape math (extracted from
``mayatk.edit_utils.curtain``): it computes draped vertex positions from plain
values, so each DCC adapter only builds a grid mesh and assigns the points —
mayatk's ``CurtainMesh`` adds the Maya build (``polyPlane`` + ``MFnMesh``),
blendertk's ``create_curtain`` the bmesh one.

:class:`CurtainDrape` is the *deformation* engine: it consumes plain rail points
(generated/sampled via :class:`Polyline`), knows nothing about meshes, and drapes
a grid into the pleated, gravity-sagged cloth. :meth:`CurtainDrape.grid_points`
yields the full draped grid.
"""
from __future__ import annotations

import bisect
import math
import random
from typing import List, Sequence, Tuple

from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.math_utils._math_utils import MathUtils
from pythontk.math_utils.noise import BandLimitedNoise
from pythontk.geo_utils.polyline import Polyline

Vec = Tuple[float, float, float]

_lerp = MathUtils.lerp                # point/vector lerp
_unit = MathUtils.safe_normalize      # normalize with a degenerate fallback
_smoothstep = MathUtils.smoothstep    # clamped Hermite ease
_sag_profile = MathUtils.catenary_sag


def _v_arms(u: float, u0: float, spread: float, depth: float, half_width: float) -> float:
    """Sum of the two **mean-preserving** arms of a downward **V** apexed at ``u0``.

    Each arm is a Ricker wavelet (:meth:`MathUtils.ricker` — a ridge with
    flanking troughs), so a fold reads as in/out undulation rather than a
    one-sided bulge. The arms coincide at the apex (``depth == 0``) and fan
    symmetrically to ``u0 ± spread·depth`` as the V runs down (``depth`` =
    distance below the apex); ``half_width`` sets each arm's width. Shared by the
    surface creases and the mid-fold forks.
    """
    ricker = MathUtils.ricker
    return ricker((u - (u0 - spread * depth)) / half_width) + ricker(
        (u - (u0 + spread * depth)) / half_width
    )


# ----------------------------------------------------------------------------
# Deformation engine (the procedural cloth drape)
# ----------------------------------------------------------------------------


class CurtainDrape(LoggingMixin):
    """Drape a grid into a pleated, gravity-sagged curtain — pure math.

    Consumes plain rail points (see :class:`Polyline`) and emits draped vertex
    positions; building the mesh from them is the DCC adapter's job. See
    ``mayatk.edit_utils.curtain.CurtainMesh`` for the parameter reference (the
    adapters re-expose this signature unchanged).

    Parameters:
        rail: Ordered world-space points the cloth hangs from (the rail).
        height: Drop of the curtain below the rail.
        hanging_points: Number of evenly-spaced pins (pleats) along the rail.
        hang_jitter: ``0``–``1`` — randomize the hang-point *spacing*.
        hang_seed: RNG seed for the random hang-point spacing.
        gravity: Catenary sag depth between hanging points (span-width scaled).
        tension: Catenary shape parameter (see :meth:`MathUtils.catenary`).
        round_points: ``0``–``1`` — round each hang-point cusp into a dome.
        round_gather: ``≥0`` — push-pull gather at each hanging point.
        fullness: Drapery fullness ratio (≥1); drives fold/belly depth.
        taper: ``-1``–``1`` vertical bias of the fold depth.
        mid_folds: Intensity of V-folds forking down from the hang points.
        mid_fold_seed: RNG seed for the mid-fold selection/shape.
        creases: Intensity of extra V-creases radiating from the top.
        crease_seed: RNG seed for the crease placement/length/depth.
        sway: Lateral fold lean along the rail (random subset of folds).
        sway_seed: RNG seed for which folds sway and how far/which way.
        end_bend_left: Signed sideways bend of the left end.
        end_bend_right: Signed sideways bend of the right end.
        end_bend_falloff: ``0``–``1`` — bend ramp-in fraction of the width.
        irregularity: Coherent band-limited surface grain (subtle).
        density: Mesh resolution in segments per world unit.
        closed: Treat the rail as a closed loop.
        name: Base name for the created object (used by the adapters).

    The Maya-side post-ops (``reduce`` / ``thickness`` / ``invert`` /
    ``soften``) are mesh operations, not drape math — the adapters own them but
    the parameters are accepted (and stored) here so a single ``**options``
    dict drives any adapter.
    """

    def __init__(
        self,
        rail: Sequence[Vec],
        height: float = 3.0,
        hanging_points: int = 8,
        hang_jitter: float = 0.0,
        hang_seed: int = 0,
        gravity: float = 0.3,
        tension: float = 1.5,
        round_points: float = 0.0,
        round_gather: float = 0.0,
        fullness: float = 2.5,
        taper: float = 0.5,
        mid_folds: float = 0.0,
        mid_fold_seed: int = 0,
        creases: float = 0.0,
        crease_seed: int = 0,
        sway: float = 0.0,
        sway_seed: int = 0,
        end_bend_left: float = 0.0,
        end_bend_right: float = 0.0,
        end_bend_falloff: float = 0.25,
        irregularity: float = 0.15,
        density: float = 8.0,
        reduce: float = 0.0,
        thickness: float = 0.0,
        invert: bool = False,
        soften: bool = True,
        closed: bool = False,
        name: str = "curtain",
    ):
        rail = [tuple(float(c) for c in p) for p in rail]
        if len(rail) < 2:
            raise ValueError("rail must contain at least two points.")

        self.rail = rail
        self.height = float(height)
        self.hanging_points = max(2, int(hanging_points))
        self.hang_jitter = max(0.0, min(1.0, float(hang_jitter)))
        self.hang_seed = int(hang_seed)
        self.gravity = float(gravity)
        self.tension = float(tension)
        self.round_points = max(0.0, min(1.0, float(round_points)))
        self.round_gather = max(0.0, float(round_gather))
        self.fullness = max(1.0, float(fullness))
        self.taper = float(taper)
        self.mid_folds = max(0.0, float(mid_folds))
        self.mid_fold_seed = int(mid_fold_seed)
        self.creases = max(0.0, float(creases))
        self.crease_seed = int(crease_seed)
        self.sway = max(0.0, float(sway))
        self.sway_seed = int(sway_seed)
        self.end_bend_left = float(end_bend_left)
        self.end_bend_right = float(end_bend_right)
        self.end_bend_falloff = max(1e-4, min(1.0, float(end_bend_falloff)))
        self.irregularity = float(irregularity)
        self.density = max(0.1, float(density))
        self.reduce = float(reduce)
        self.thickness = float(thickness)
        self.invert = bool(invert)
        self.soften = bool(soften)
        self.closed = bool(closed)
        self.name = name

        # Spans between hanging points (a closed loop wraps, so one more span).
        self.spans = (
            self.hanging_points if self.closed else max(self.hanging_points - 1, 1)
        )
        # u-positions (0..1) of the hanging points along the rail — evenly spaced
        # unless hang_jitter randomizes the interior spacing. Param-derived, so
        # computed here (always available to the offset helpers, even standalone).
        self._hang_points = self._make_hang_points()

    # ------------------------------------------------------------- build prep

    def prepare(self) -> Tuple[int, int, List[Tuple[Vec, Vec, Vec]]]:
        """Precompute the per-build state and return ``(u_segs, v_segs, frames)``.

        Sets the total rail length, the seeded feature sets (span jitter,
        creases, mid-folds, sway, billow) and resolves the grid resolution —
        everything :meth:`drape` needs. Adapters call this once per build, then
        place each grid vertex via :meth:`drape` (or take the whole grid from
        :meth:`grid_points`).
        """
        self._total_length = Polyline.length(self.rail, self.closed)
        u_segs, v_segs = self._resolve_resolution()
        frames = Polyline.frames(self.rail, u_segs, self.closed)

        # Fixed-seed base RNG for the always-on subtle variation (the band-
        # limited grain and per-span depth jitter): deliberately subtle, so a
        # user-facing seed for it never earned its keep — the seeds that matter
        # are the per-feature ones (mid_fold_seed / crease_seed).
        rng = random.Random(self._BASE_SEED)
        # Per-span depth variation so the folds aren't mechanically identical.
        self._span_jitter = [rng.uniform(0.8, 1.2) for _ in range(self.spans)]
        self._creases = self._make_creases()
        self._midfolds = self._make_midfolds()
        self._sway = self._make_sway()
        self._billow = self._make_billow()
        return u_segs, v_segs, frames

    def grid_points(self) -> Tuple[int, int, List[Vec]]:
        """The full draped grid: ``(u_segs, v_segs, points)``.

        ``points`` is row-major over ``(v_segs + 1)`` rows of ``(u_segs + 1)``
        columns — row 0 is the hem (``v = 0``), the last row the rail
        (``v = 1``); ``points[row * (u_segs + 1) + col]``. Convenience for
        adapters that build the whole mesh in one pass.
        """
        u_segs, v_segs, frames = self.prepare()
        pts: List[Vec] = []
        for r in range(v_segs + 1):
            v = r / v_segs
            for c in range(u_segs + 1):
                pos, tan, normal = frames[c]
                pts.append(self.drape(c / u_segs, v, pos, tan, normal))
        return u_segs, v_segs, pts

    # ------------------------------------------------------------- internals

    def _make_hang_points(self) -> List[float]:
        """u-positions (0..1) of the hanging points, ``spans + 1`` of them.

        Evenly spaced unless ``hang_jitter`` perturbs the interior spacing. The
        ends (``0`` and ``1`` — the outer pins, or a closed rail's seam) stay
        pinned so the cloth still spans the full rail. Each interior point shifts
        by at most ``0.4`` of a span, so the points stay strictly ordered (no
        crossed/zero-width spans) at any jitter.
        """
        n = self.spans
        pts = [i / n for i in range(n + 1)]
        if self.hang_jitter > 0.0:
            rng = random.Random(self.hang_seed)
            span_u = 1.0 / n
            for i in range(1, n):  # interior points only
                pts[i] += rng.uniform(-1.0, 1.0) * self.hang_jitter * 0.4 * span_u
        return pts

    def _span_at(self, u: float) -> Tuple[int, float]:
        """Map a rail position ``u`` (0..1) to its ``(span index, local t)``.

        ``t`` is 0..1 within the (possibly uneven) span; ``span index + t`` is
        the old uniform ``u * spans`` phase, so the belly/sway half-sines stay
        zero at every (now uneven) hang point.
        """
        hp = self._hang_points
        k = max(0, min(bisect.bisect_right(hp, u) - 1, self.spans - 1))
        w = hp[k + 1] - hp[k]
        return k, (u - hp[k]) / w if w > 0.0 else 0.0

    def drape(self, u, v, pos, tan, normal) -> Vec:
        """Place one cloth vertex.

        ``u`` runs along the rail (0..1), ``v`` runs vertically (0 = hem,
        1 = rail). The fabric is pinned at each hanging point (``belly`` and
        ``sag`` both vanish there) and, between points, bellies into an
        alternating fold (``normal`` direction) while its whole strip sags down
        a catenary under gravity. ``sway`` additionally leans a fold sideways
        along ``tan`` (the in-plane rail tangent).
        """
        k, t = self._span_at(u)             # span index + local 0..1
        phase = k + t                       # integers = hang points

        # Fold belly: ``_BELLY_HUMPS_PER_SPAN`` half-sine humps per pleat-span (2
        # = one out-bulge + one in-recess = a full fold), zero at every hang
        # point *and* at the in/out crossover inside the span — so one dialed
        # hang point reads as one pleat with a full out-and-in fold between it
        # and the next (instead of a single half-hump per span). taper deepens it
        # toward the hem and gathers (shallows) it at the top.
        depth = (
            0.15
            * (self.fullness - 1.0)
            * (1.0 + self.taper * (1.0 - 2.0 * v))
            * self._span_jitter[k]
        )
        belly = depth * math.sin(math.pi * self._BELLY_HUMPS_PER_SPAN * phase)

        # Kept subtle (the deliberate folds come from fullness / mid_folds):
        # coherent, band-limited surface grain built from zero-mean waves, so it
        # ripples the cloth in and out rather than puffing it one way.
        irr = self.irregularity * 0.2 * self._billow_offset(u, v)
        # All sideways shaping rides the in-plane normal: belly fold, billow,
        # the mid-fold forks, the V-creases, and the per-end bend.
        offset = (
            belly
            + irr
            + self._midfold_offset(u, v)
            + self._crease_offset(u, v)
            + self._end_bend_offset(u)
        )

        # Lateral lean rides the in-plane tangent (along the rail) — the random
        # left/right drift of a fold, distinct from the in/out `offset`.
        lateral = self._sway_offset(u, v)
        x = pos[0] + normal[0] * offset + tan[0] * lateral
        z = pos[2] + normal[2] * offset + tan[2] * lateral

        # Gravity: the span between two hang points sags along a catenary
        # (optionally rounded and/or push-pull gathered at the pins); the whole
        # vertical strip drops with its (sagged) top edge. The span's *own* width
        # scales the sag, so a wider (jittered) gap falls further. Dividing by
        # ``_BELLY_HUMPS_PER_SPAN`` normalizes that to the per-hump width: the
        # catenary and its push-pull gather fire once per pleat (one clean cusp
        # at the rail — a smoother header) at the same depth as before, so
        # halving the dialed hang points (each span now that much wider)
        # reproduces the old sag rather than ballooning it. The 0.5 calibrates
        # the gather dial so a full slider lifts the pin ~half a sag-unit
        # (matches the other dials).
        sag_width = (
            (self._hang_points[k + 1] - self._hang_points[k])
            * self._total_length
            / self._BELLY_HUMPS_PER_SPAN
        )
        sag = self.gravity * sag_width * _sag_profile(
            2.0 * t - 1.0, self.tension, self.round_points, self.round_gather * 0.5
        )
        y = pos[1] - sag - (1.0 - v) * self.height
        return (x, y, z)

    # Belly half-sine humps per pleat-span (the run between two consecutive hang
    # points): ``2`` = one out-bulge + one in-recess = a single full fold per
    # span. Decouples the *body* fold density from the *top* gather frequency —
    # the catenary sag + push-pull gather fire once per hang point (one clean
    # pleat at the rail), while the belly runs at this many humps, so the dialed
    # hang-point count maps ~1:1 to the visible folds. ``2`` gives the previous
    # look at half the dialed points (and half the top cusps). The same factor
    # rescales the sag (per-hump width) and the mesh resolution (cols per hump).
    _BELLY_HUMPS_PER_SPAN = 2

    # Fixed seed for the always-on subtle grain + per-span depth jitter (the
    # tunable seeds are the per-feature mid_fold_seed / crease_seed).
    _BASE_SEED = 0
    # Crease half-width in u (narrow enough to read as a fold, wide enough that
    # a normal-density mesh resolves it without aliasing).
    _CREASE_WIDTH = 0.05
    # Peak of the spindle profile x^0.6·(1-x)^1.3 (at x = 0.6/1.9); the profile
    # is divided by this so its crest is 1, keeping the creases slider's
    # strength independent of the exact profile exponents.
    _CREASE_PEAK = 0.3058

    def _make_creases(self):
        """Seeded set of V-creases: ``(u0, length, amp, spread)`` per crease.

        ``u0`` is the apex (top) position, ``length`` how far down it runs,
        ``amp`` its *signed* depth — so creases push out **and** in, reading as
        a mix of ridges and valleys rather than bumps all on one side —, and
        ``spread`` how far the two arms diverge (each crease draws a downward
        **V** of a different length). Count scales with the number of spans;
        empty when ``creases`` is off.
        """
        if self.creases <= 0.0:
            return []
        rng = random.Random(self.crease_seed)
        n = max(3, round(self.spans * 2.5))
        return [
            (
                rng.uniform(0.04, 0.96),                          # u0 — apex position
                rng.uniform(0.3, 1.0),                            # length — fraction of the drop
                rng.choice((-1.0, 1.0)) * rng.uniform(0.4, 1.0),  # amp — signed ridge/valley
                rng.uniform(0.02, 0.10),                          # spread — V arm divergence
            )
            for _ in range(n)
        ]

    def _crease_offset(self, u: float, v: float) -> float:
        """Summed V-crease displacement at ``(u, v)`` (0 when creases off)."""
        if not self._creases:
            return 0.0
        depth_from_top = 1.0 - v  # 0 at the rail, 1 at the hem
        # A crease's gaussians vanish (``~e⁻²⁵``) past this much |u - u0|; skip
        # those so a dense live re-drape isn't O(verts × creases) ``exp`` calls.
        band_tail = 5.0 * self._CREASE_WIDTH
        total = 0.0
        for u0, length, amp, spread in self._creases:
            if depth_from_top > length:
                continue  # this crease has already petered out above the vertex
            if abs(u - u0) > spread + band_tail:
                continue  # the vertex is outside this crease's u-band
            # Spindle profile along the crease: fades in just below the rail
            # (so the pinned top edge stays put), crests in the upper third,
            # and tapers to nothing at the tip. ``> length`` guard above keeps
            # ``1 - x`` non-negative for the fractional power.
            x = depth_from_top / length
            fall = (x ** 0.6) * ((1.0 - x) ** 1.3) / self._CREASE_PEAK
            total += amp * fall * _v_arms(
                u, u0, spread, depth_from_top, self._CREASE_WIDTH
            )
        return self.creases * 0.15 * total

    # Mid-fold vertical fades. The fork ramps in over a *small* top fade so it
    # reads nearly to the rail (only the pinned rail row v=1 itself stays put)
    # and out over a wider tip fade at its lower end, holding full strength
    # between — so a long fork reads from just under the hooks all the way down.
    _MIDFOLD_TOP_FADE = 0.04
    _MIDFOLD_FADE = 0.12  # tip fade-out width

    def _make_midfolds(self):
        """Seeded mid-folds: downward **V** forks anchored at the hang points.

        Returns ``(u0, length, amp, spread, half_width)`` per fork. Unlike the
        belly (one in/out bulge per span) these apex *on* a hang point (``u0``)
        and fan **out and down** into the neighbouring spans, interrupting that
        plain in/out cycle the way a heavy gathered drape forks below each hook.
        Only a random ~1/4–1/2 of the (interior) hang points are chosen; lengths
        vary (some stop high, some run nearly to the hem) as do width/depth
        (narrow-deep vs wide-shallow). Empty when ``mid_folds`` is off.
        """
        if self.mid_folds <= 0.0:
            return []
        # A fork at the very end would be half-clipped, so anchor only on
        # interior hang points; a closed loop wraps, so every point is interior.
        points = (
            list(range(self.spans)) if self.closed else list(range(1, self.spans))
        )
        if not points:
            return []
        # Arm spread and fold width are stored in global u but authored as
        # fractions of a *span* (1/spans wide), so the fork keeps the same shape
        # relative to the gap whatever the hang-point count — amp stays absolute
        # (a fold's depth shouldn't change just because there are more pleats).
        span_u = 1.0 / self.spans
        rng = random.Random(self.mid_fold_seed)
        frac = rng.uniform(0.25, 0.5)
        chosen = rng.sample(points, max(1, round(len(points) * frac)))
        folds = []
        for i in chosen:
            if rng.random() < 0.5:                  # long, narrow-ish, deep
                length = rng.uniform(0.7, 1.0)
                width_frac = rng.uniform(0.18, 0.42)
                amp = rng.uniform(0.7, 1.1)
            else:                                   # short, wide, shallow
                length = rng.uniform(0.3, 0.6)
                width_frac = rng.uniform(0.45, 0.85)
                amp = rng.uniform(0.4, 0.7)
            spread_frac = rng.uniform(0.25, 0.6)    # arms fan this much of a span
            # Anchor on the hang point's *actual* (possibly jittered) u-position;
            # arm spread / width stay relative to the average span so the fork
            # shape is invariant to hang-point count.
            folds.append(
                (
                    self._hang_points[i],
                    length,
                    amp,
                    spread_frac * span_u,
                    width_frac * span_u,
                )
            )
        return folds

    def _midfold_offset(self, u: float, v: float) -> float:
        """Summed mid-fold V displacement at ``(u, v)`` (0 when off)."""
        if not self._midfolds:
            return 0.0
        depth_from_top = 1.0 - v  # 0 at the rail, 1 at the hem
        total = 0.0
        for u0, length, amp, spread, half_width in self._midfolds:
            if depth_from_top > length:
                continue  # this fork has petered out above the vertex
            if abs(u - u0) > spread * length + 4.5 * half_width:
                continue  # past the fork's u-band (incl. the ricker troughs)
            # Ramp in quickly below the rail (small top fade -> runs to the top)
            # and fade out at the tip; hold full strength between.
            v_prof = _smoothstep(depth_from_top / self._MIDFOLD_TOP_FADE) * _smoothstep(
                (length - depth_from_top) / self._MIDFOLD_FADE
            )
            total += amp * v_prof * _v_arms(u, u0, spread, depth_from_top, half_width)
        return self.mid_folds * 0.15 * total

    def _make_sway(self):
        """Seeded per-span lateral lean (one signed factor per span).

        Only a random ~half of the spans lean (the rest sit at ``0``, so the
        effect reads as *certain areas* drifting sideways, not a uniform shear);
        the chosen ones get a random sign (left/right) and magnitude. Empty when
        ``sway`` is off.
        """
        if self.sway <= 0.0:
            return []
        rng = random.Random(self.sway_seed)
        return [
            rng.choice((-1.0, 1.0)) * rng.uniform(0.4, 1.0)
            if rng.random() < 0.5
            else 0.0
            for _ in range(self.spans)
        ]

    def _sway_offset(self, u: float, v: float) -> float:
        """Lateral (along-rail) lean at ``(u, v)`` (0 when off).

        Rides the belly envelope — ``|sin(pi * _BELLY_HUMPS_PER_SPAN * phase)|``
        is zero at the pinned hang points and peaks where the fabric bellies most
        — so a fold drifts sideways most where it bulges most, and grows toward
        the free hem (calm at the gathered top).
        """
        if not self._sway:
            return 0.0
        k, t = self._span_at(u)
        lean = self._sway[k]
        if lean == 0.0:
            return 0.0
        phase = k + t
        # Track the belly: zero at the pinned hang points (and the in/out
        # crossover), peak where the fabric bellies most.
        env = abs(math.sin(math.pi * self._BELLY_HUMPS_PER_SPAN * phase))
        hem = 0.3 + 0.7 * (1.0 - v)           # more sway toward the free hem
        return self.sway * 0.2 * lean * env * hem

    def _end_bend_offset(self, u: float) -> float:
        """Signed bend ramped in from each end over ``end_bend_falloff``."""
        if self.end_bend_left == 0.0 and self.end_bend_right == 0.0:
            return 0.0
        fo = self.end_bend_falloff
        wl = _smoothstep((fo - u) / fo)            # 1 at u=0 -> 0 at u=fo
        wr = _smoothstep((u - (1.0 - fo)) / fo)    # 0 -> 1 at u=1
        return self.end_bend_left * wl + self.end_bend_right * wr

    # Band-limited surface grain: a handful of smooth wave octaves read as soft
    # fabric relief, where per-vertex white noise (the prior approach) read as
    # static. Amplitude rolls off ~1/f per octave.
    _BILLOW_OCTAVES = 4
    _BILLOW_FALLOFF = 0.55

    def _make_billow(self):
        """Build the coherent band-limited surface-grain field (``None`` when off).

        Delegates to :class:`BandLimitedNoise` (the reusable primitive); a
        ``closed`` rail makes the field wrap across the u-seam. The drape-
        specific weighting (stronger toward the free hem) lives in
        :meth:`_billow_offset`, not baked into the noise.
        """
        if self.irregularity <= 0.0:
            return None
        return BandLimitedNoise(
            seed=self._BASE_SEED,
            octaves=self._BILLOW_OCTAVES,
            falloff=self._BILLOW_FALLOFF,
            u_periodic=self.closed,
        )

    def _billow_offset(self, u: float, v: float) -> float:
        """Coherent surface grain at ``(u, v)`` (0 when off), weighted to the hem."""
        if self._billow is None:
            return 0.0
        # More relief toward the free hem; the gathered top stays calmer.
        return self._billow.at(u, v) * (0.4 + 0.6 * (1.0 - v))

    def _resolve_resolution(self) -> Tuple[int, int]:
        # Relies on self._total_length (set first in prepare()).
        # Resolve at least ~8 segments per belly hump: the belly runs
        # _BELLY_HUMPS_PER_SPAN humps per span, so guarantee that many more
        # columns.
        u_segs = max(
            int(math.ceil(self._total_length * self.density)),
            self.spans * 8 * self._BELLY_HUMPS_PER_SPAN,
            12,
        )
        v_segs = max(int(math.ceil(self.height * self.density)), 8)
        # Cap so an extreme density slider can't lock up the session.
        return min(u_segs, 4000), min(v_segs, 1000)
