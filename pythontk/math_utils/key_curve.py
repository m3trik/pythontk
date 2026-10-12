# !/usr/bin/python
# coding=utf-8
"""Keyed 1-D animation curves: cubic Bezier spans, step holds and infinity.

A host-free reading of one animation curve -- a Maya animCurve, a Blender
FCurve -- that a curve editor draws and edits without asking the host on
every mouse move.  Each key carries its tangent HANDLES as vectors in curve
units (frames along, value up), so the two hosts meet in one shape:

* Blender's ``handle_left`` / ``handle_right`` are the vectors minus ``co``.
* Maya's angle and weight are ``weight * (cos a, sin a)`` on a weighted curve;
  on an unweighted one only the SLOPE ``tan a`` (value per frame) is stored,
  and the control point sits a third of the span out (measured against
  ``keyframe -q -eval`` at film/ntsc, cm/m, translate/rotate/scale: exact).

``segments()`` emits the span dicts the uitk sequencer's
``CurveUtils.build_curve_path`` draws, so a track row's mini curve and a
graph editor's full curve are one reading.  ``MathUtils.evaluate_hermite`` is
the vectorised cousin for dense sample runs (key reduction); this class
evaluates one curve at a time, numpy-free, with the host's infinity modes.
"""

from __future__ import annotations

import bisect
import copy
import math
from dataclasses import dataclass, field, replace
from typing import Iterable, List, Optional, Sequence, Tuple

Vec = Tuple[float, float]

#: Shape of the span that STARTS at a key.
INTERPOLATIONS = ("bezier", "linear", "step", "stepnext")
#: What the curve does before its first key / after its last one.
INFINITY_MODES = ("constant", "linear", "cycle", "cycle_offset", "oscillate")

#: Two keys closer than this (frames) are one time.
TIME_EPS = 1e-3


@dataclass
class CurveKey:
    """One key of a :class:`KeyCurve`.

    Parameters:
        time: Frame.
        value: Value in the host's display units.
        in_handle: Vector from the key to its IN control point (``dx <= 0``).
        out_handle: Vector from the key to its OUT control point (``dx >= 0``).
        interpolation: Shape of the span after this key (:data:`INTERPOLATIONS`).
        in_type / out_type: The host's own tangent-type names (``"auto"``,
            ``"AUTO_CLAMPED"``...), carried for display and type edits.
        broken: The two handles move independently.
        breakdown: A breakdown key (Maya) -- drawn distinctly.
        hidden: Part of the curve, but not a key the user works with (the shot
            system's claimed samples): drawn without a dot.
        samples: ``[(t, v), ...]`` strictly inside the span after this key,
            for an eased span neither a line nor one cubic draws (Blender's
            SINE / BACK / BOUNCE...).

    Raises:
        ValueError: An interpolation outside :data:`INTERPOLATIONS`.
    """

    time: float
    value: float
    in_handle: Vec = (-1.0, 0.0)
    out_handle: Vec = (1.0, 0.0)
    interpolation: str = "bezier"
    in_type: str = ""
    out_type: str = ""
    broken: bool = False
    breakdown: bool = False
    hidden: bool = False
    samples: Optional[List[Vec]] = None

    def __post_init__(self):
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(
                f"interpolation must be one of {INTERPOLATIONS}, "
                f"got {self.interpolation!r}"
            )


@dataclass
class KeyCurve:
    """An animation curve: keys, handle meaning and infinity, host-free.

    Parameters:
        keys: The keys; sorted by time on construction.
        curve_id: The host's opaque name for the curve (an animCurve node, an
            fcurve's ``obj|data_path|index``) -- what an edit names it by.
        label: Display name (``"pCube1.translateX"``).
        color: Display colour, ``"#RRGGBB"``.
        weighted: True when a handle's LENGTH is meaningful (Blender, Maya
            weighted tangents); False when only its slope is (Maya's default).
        pre_infinity / post_infinity: :data:`INFINITY_MODES`.

    Raises:
        ValueError: An infinity mode outside :data:`INFINITY_MODES`.
    """

    keys: List[CurveKey] = field(default_factory=list)
    curve_id: str = ""
    label: str = ""
    color: Optional[str] = None
    weighted: bool = True
    pre_infinity: str = "constant"
    post_infinity: str = "constant"

    def __post_init__(self):
        for side in (self.pre_infinity, self.post_infinity):
            if side not in INFINITY_MODES:
                raise ValueError(
                    f"infinity must be one of {INFINITY_MODES}, got {side!r}"
                )
        self.keys = sorted(self.keys, key=lambda k: k.time)

    # -- reading ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.keys)

    def times(self) -> List[float]:
        """Key times, ascending."""
        return [k.time for k in self.keys]

    def index_at(self, t: float, eps: float = TIME_EPS) -> int:
        """Index of the key at *t* (within *eps*), or -1.

        Parameters:
            t: The frame.
            eps: Frames within which a key is ON *t*.

        Returns:
            The key's index, or -1 when no key is there.
        """
        times = self.times()
        i = bisect.bisect_left(times, t - eps)
        return i if i < len(times) and abs(times[i] - t) <= eps else -1

    def control_points(self, i: int) -> Tuple[Vec, Vec, Vec, Vec]:
        """The cubic of span *i* (key *i* to key *i + 1*): ``(p0, p1, p2, p3)``.

        A weighted curve places each control point at the key plus its
        handle; a handle whose own reach passes the span is pulled in along
        itself to the span (Blender's ``BKE_fcurve_correct_bezpart``; Maya
        agrees while no single handle overreaches -- past that it follows
        its own rule, which its adapter supplies as span ``samples``).  An
        unweighted curve pins them a third of the span out and keeps only
        each handle's slope.

        Parameters:
            i: The span's first key index (``0 <= i < len(self) - 1``).

        Returns:
            The four ``(time, value)`` control points.
        """
        k0, k1 = self.keys[i], self.keys[i + 1]
        span = k1.time - k0.time
        p0, p3 = (k0.time, k0.value), (k1.time, k1.value)
        if not self.weighted:
            third = span / 3.0
            s_out = self._slope(k0.out_handle)
            s_in = self._slope(k1.in_handle)
            return (
                p0,
                (k0.time + third, k0.value + third * s_out),
                (k1.time - third, k1.value - third * s_in),
                p3,
            )
        ox, oy = k0.out_handle
        ix, iy = k1.in_handle
        ox, ix = max(ox, 0.0), min(ix, 0.0)
        if span > 0 and ox > span:
            ox, oy = span, oy * span / ox
        if span > 0 and -ix > span:
            ix, iy = -span, iy * span / -ix
        return p0, (k0.time + ox, k0.value + oy), (k1.time + ix, k1.value + iy), p3

    def segments(
        self, lo: Optional[int] = None, hi: Optional[int] = None
    ) -> List[dict]:
        """Span dicts for keys *lo*..*hi* (indices, inclusive; default all).

        ``{t0, v0, t1, v1, out_type, cp1, cp2[, samples]}`` -- the shape
        ``CurveUtils.build_curve_path`` draws: ``out_type`` is the span's
        interpolation, and only a ``"bezier"`` span carries control points.

        Parameters:
            lo: First key index (clamped to the keys).
            hi: Last key index (clamped to the keys).

        Returns:
            One dict per span from key *lo* to key *hi*.
        """
        lo = 0 if lo is None else max(lo, 0)
        hi = len(self.keys) - 1 if hi is None else min(hi, len(self.keys) - 1)
        out = []
        for i in range(lo, hi):
            k0, k1 = self.keys[i], self.keys[i + 1]
            seg = {
                "t0": k0.time,
                "v0": k0.value,
                "t1": k1.time,
                "v1": k1.value,
                "out_type": k0.interpolation,
                "cp1": None,
                "cp2": None,
            }
            if k0.interpolation == "bezier":
                _p0, cp1, cp2, _p3 = self.control_points(i)
                seg["cp1"], seg["cp2"] = cp1, cp2
            if k0.samples:
                seg["samples"] = list(k0.samples)
            out.append(seg)
        return out

    def evaluate(self, t: float) -> float:
        """The curve's value at frame *t*, infinity included.

        Parameters:
            t: The frame.

        Returns:
            The value (0.0 for a curve with no keys).
        """
        if not self.keys:
            return 0.0
        first, last = self.keys[0], self.keys[-1]
        if t < first.time:
            return self._infinity(t, pre=True)
        if t > last.time:
            return self._infinity(t, pre=False)
        return self._inside(t)

    def value_bounds(
        self,
        t0: Optional[float] = None,
        t1: Optional[float] = None,
        handles: bool = False,
    ) -> Tuple[float, float]:
        """``(lo, hi)`` of the curve's values over ``[t0, t1]`` (default: its keys).

        The curve itself, cubic extrema included -- what framing a curve
        means -- and, past its keys, every repeat a repeating infinity plays
        there.  *handles* also takes in the control points, what the track
        rows' normalised preview fits.

        Parameters:
            t0: Window start; default the first key.
            t1: Window end; default the last key.
            handles: Take in the Bezier control points too.

        Returns:
            ``(lo, hi)``; ``(0.0, 0.0)`` for a curve with no keys.
        """
        if not self.keys:
            return 0.0, 0.0
        t0 = self.keys[0].time if t0 is None else t0
        t1 = self.keys[-1].time if t1 is None else t1
        vals = [self.evaluate(t0), self.evaluate(t1)]
        vals.extend(self._repeat_bounds(t0, t1, handles))
        for i, k in enumerate(self.keys):
            if t0 - TIME_EPS <= k.time <= t1 + TIME_EPS:
                vals.append(k.value)
            if i + 1 >= len(self.keys):
                continue
            k1 = self.keys[i + 1]
            if k1.time < t0 or k.time > t1:
                continue
            if k.interpolation == "bezier":
                p0, p1, p2, p3 = self.control_points(i)
                if handles:
                    vals.extend((p1[1], p2[1]))
                for s in self._extrema_params(p0[1], p1[1], p2[1], p3[1]):
                    x = self._bez(p0[0], p1[0], p2[0], p3[0], s)
                    if t0 <= x <= t1:
                        vals.append(self._bez(p0[1], p1[1], p2[1], p3[1], s))
            for st, sv in k.samples or ():
                if t0 <= st <= t1:
                    vals.append(sv)
        return min(vals), max(vals)

    def normalization(
        self, t0: Optional[float] = None, t1: Optional[float] = None
    ) -> Tuple[float, float]:
        """``(offset, scale)`` that maps the curve into ``[-1, 1]``:
        ``v_norm = (v - offset) * scale``.  A flat curve maps onto 0.

        Parameters:
            t0: Window start, as :meth:`value_bounds` reads it.
            t1: Window end.

        Returns:
            ``(offset, scale)``.
        """
        lo, hi = self.value_bounds(t0, t1)
        half = (hi - lo) / 2.0
        return (lo + hi) / 2.0, (1.0 / half if half > 1e-12 else 1.0)

    def to_preview(
        self, t_start: float, t_end: float, hidden_times: Iterable[float] = ()
    ) -> Optional[dict]:
        """The track rows' ``curve_preview`` dict over ``[t_start, t_end]``.

        The keys in range plus one bounding key on each side (the edge spans
        need them), or the two keys a span crossing the whole range runs
        between; ``None`` when the curve has nothing there.  A key on
        *hidden_times* (or flagged :attr:`CurveKey.hidden`) draws no dot.

        Parameters:
            t_start: The row's first frame.
            t_end: Its last frame.
            hidden_times: Frames whose keys draw no dot.

        Returns:
            ``{keys, segments, broken, hidden, weighted, val_min, val_max}``,
            or ``None``.
        """
        n = len(self.keys)
        if not n:
            return None
        times = self.times()
        inside = [
            i
            for i, t in enumerate(times)
            if t_start - TIME_EPS <= t <= t_end + TIME_EPS
        ]
        if inside:
            lo, hi = max(inside[0] - 1, 0), min(inside[-1] + 1, n - 1)
        else:
            before = [i for i, t in enumerate(times) if t < t_start]
            after = [i for i, t in enumerate(times) if t > t_end]
            if not before or not after:
                return None
            lo, hi = before[-1], after[0]
        hidden_times = list(hidden_times)
        keys = self.keys[lo : hi + 1]
        segments = self.segments(lo, hi)
        vals = [k.value for k in keys]
        for seg in segments:
            for cp in (seg["cp1"], seg["cp2"]):
                if cp is not None:
                    vals.append(cp[1])
            vals.extend(v for _t, v in seg.get("samples") or ())
        return {
            "keys": [(k.time, k.value) for k in keys],
            "segments": segments,
            "broken": [k.broken for k in keys],
            "hidden": [
                k.hidden or any(abs(k.time - h) <= TIME_EPS for h in hidden_times)
                for k in keys
            ],
            "weighted": self.weighted,
            "val_min": min(vals),
            "val_max": max(vals),
        }

    # -- editing (previews: every edit returns a new curve) --------------------

    def copy(self) -> "KeyCurve":
        """A deep copy -- what a live preview mutates."""
        return copy.deepcopy(self)

    def with_keys_moved(
        self, indices: Iterable[int], dt: float = 0.0, dv: float = 0.0
    ) -> "KeyCurve":
        """A copy with keys *indices* moved by ``(dt, dv)``.

        The time step is clamped so no moved key reaches an unmoved
        neighbour (both hosts refuse two keys on one frame); the handles
        ride along, being vectors from their key.

        Parameters:
            indices: The keys to move (out-of-range ones are ignored).
            dt: Frames to move them by, clamped as above.
            dv: Value to move them by.

        Returns:
            The moved copy; this curve is unchanged.
        """
        moved = sorted({i for i in indices if 0 <= i < len(self.keys)})
        out = self.copy()
        if not moved:
            return out
        dt = self._clamped_dt(moved, dt)
        for i in moved:
            k = out.keys[i]
            k.time += dt
            k.value += dv
        out.keys.sort(key=lambda k: k.time)
        return out

    def with_handle(
        self, index: int, side: str, vector: Vec, unify: Optional[bool] = None
    ) -> "KeyCurve":
        """A copy whose key *index* has its *side* (``"in"``/``"out"``) handle
        set to *vector*.

        An unbroken key (or ``unify=True``) swings its other handle opposite:
        a weighted curve keeps that handle's own length, an unweighted one
        its time offset (only the slope is stored).  ``unify=False`` leaves
        it -- a breaking drag.

        Parameters:
            index: The key.
            side: ``"in"`` or ``"out"``.
            vector: The handle, from the key, in frames and value; its time
                is clamped to its own side of the key.
            unify: Swing the other handle too; default: when unbroken.

        Returns:
            The edited copy; this curve is unchanged.

        Raises:
            ValueError: *side* is neither ``"in"`` nor ``"out"``.
        """
        if side not in ("in", "out"):
            raise ValueError(f"side must be 'in' or 'out', got {side!r}")
        out = self.copy()
        k = out.keys[index]
        vx, vy = vector
        if side == "out":
            vx = max(vx, 0.0)
        else:
            vx = min(vx, 0.0)
        setattr(k, f"{side}_handle", (vx, vy))
        if unify is None:
            unify = not k.broken
        if unify:
            other = "in" if side == "out" else "out"
            px, py = getattr(k, f"{other}_handle")
            if self.weighted:
                length = math.hypot(px, py)
                norm = math.hypot(vx, vy)
                if norm > 1e-12:
                    px, py = -vx / norm * length, -vy / norm * length
            elif abs(vx) > 1e-12:
                py = px * (vy / vx)
            setattr(k, f"{other}_handle", (px, py))
        return out

    def with_infinity(
        self, pre: Optional[str] = None, post: Optional[str] = None
    ) -> "KeyCurve":
        """A copy with new infinity modes.

        Parameters:
            pre: The mode before the first key; ``None`` keeps it.
            post: The mode after the last key; ``None`` keeps it.

        Returns:
            The edited copy; this curve is unchanged.

        Raises:
            ValueError: A mode outside :data:`INFINITY_MODES`.
        """
        return replace(
            self.copy(),
            pre_infinity=self.pre_infinity if pre is None else pre,
            post_infinity=self.post_infinity if post is None else post,
        )

    @staticmethod
    def bezier_parameter(x: float, x0: float, x1: float, x2: float, x3: float) -> float:
        """The parameter at which a cubic Bezier span's TIME is *x*.

        A weighted span bends time itself, so a value at a time is read at
        this parameter, not at the time's fraction of the span.  The span's
        own evaluation and an FBX weighted tangent's (``FbxFile``) share it.

        Parameters:
            x: The time, inside ``[x0, x3]``.
            x0, x1, x2, x3: The span's time control points, inner ones inside
                the span (as :meth:`control_points` places them), so time
                only grows along it.

        Returns:
            The parameter in ``[0, 1]``, by bisection to double precision.
        """
        lo, hi = 0.0, 1.0
        for _ in range(60):
            s = (lo + hi) / 2.0
            if KeyCurve._bez(x0, x1, x2, x3, s) < x:
                lo = s
            else:
                hi = s
        return (lo + hi) / 2.0

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _slope(handle: Vec) -> float:
        dx, dy = handle
        if abs(dx) < 1e-12:
            return math.copysign(1e6, dy) if dy else 0.0
        return dy / dx

    @staticmethod
    def _bez(a: float, b: float, c: float, d: float, s: float) -> float:
        u = 1.0 - s
        return u * u * u * a + 3 * u * u * s * b + 3 * u * s * s * c + s * s * s * d

    @staticmethod
    def _extrema_params(a: float, b: float, c: float, d: float) -> List[float]:
        """Parameters in (0, 1) where the cubic's derivative is zero."""
        qa = -a + 3 * b - 3 * c + d
        qb = 2 * (a - 2 * b + c)
        qc = b - a
        roots = []
        if abs(qa) < 1e-12:
            if abs(qb) > 1e-12:
                roots.append(-qc / qb)
        else:
            disc = qb * qb - 4 * qa * qc
            if disc >= 0:
                r = math.sqrt(disc)
                roots.extend(((-qb + r) / (2 * qa), (-qb - r) / (2 * qa)))
        return [s for s in roots if 0.0 < s < 1.0]

    def _span_value(self, i: int, t: float) -> float:
        k0, k1 = self.keys[i], self.keys[i + 1]
        if t >= k1.time:
            return k1.value
        if t <= k0.time:
            return k0.value
        if k0.interpolation == "step":
            return k0.value
        if k0.interpolation == "stepnext":
            return k1.value if t > k0.time else k0.value
        if k0.samples:
            pts = [(k0.time, k0.value), *k0.samples, (k1.time, k1.value)]
            j = max(bisect.bisect_right([p[0] for p in pts], t) - 1, 0)
            (ta, va), (tb, vb) = pts[j], pts[min(j + 1, len(pts) - 1)]
            return va if tb <= ta else va + (vb - va) * (t - ta) / (tb - ta)
        if k0.interpolation == "linear":
            span = k1.time - k0.time
            return k0.value + (k1.value - k0.value) * (t - k0.time) / span
        p0, p1, p2, p3 = self.control_points(i)
        s = self.bezier_parameter(t, p0[0], p1[0], p2[0], p3[0])
        return self._bez(p0[1], p1[1], p2[1], p3[1], s)

    def _inside(self, t: float) -> float:
        times = self.times()
        i = bisect.bisect_right(times, t) - 1
        if i >= len(self.keys) - 1:
            return self.keys[-1].value
        return self._span_value(max(i, 0), t)

    def _repeat_bounds(self, t0: float, t1: float, handles: bool) -> List[float]:
        """The values a repeating infinity (``cycle``, ``cycle_offset``,
        ``oscillate``) plays over the parts of ``[t0, t1]`` outside the keys:
        each repeat's piece folded back onto the keyed span.  Only the first
        two and last two repeats are read -- those between are whole, and
        ``cycle_offset``'s shift is linear in the repeat, so the outer ones
        bound it."""
        first, last = self.keys[0], self.keys[-1]
        period = last.time - first.time
        if period <= 0:
            return []
        out: List[float] = []
        for pre, a, b in (
            (True, t0, min(t1, first.time)),
            (False, max(t0, last.time), t1),
        ):
            mode = self.pre_infinity if pre else self.post_infinity
            if a >= b or mode not in ("cycle", "cycle_offset", "oscillate"):
                continue
            n0 = math.floor((a - first.time) / period)
            n1 = math.floor((b - first.time) / period)
            for n in sorted({n0, min(n0 + 1, n1), max(n1 - 1, n0), n1}):
                start = first.time + n * period
                lo_t = max(a, start) - start + first.time
                hi_t = min(b, start + period) - start + first.time
                if lo_t > hi_t:
                    continue
                if mode == "oscillate" and n % 2:
                    lo_t, hi_t = (
                        first.time + last.time - hi_t,
                        (first.time + last.time - lo_t),
                    )
                lo, hi = self.value_bounds(lo_t, hi_t, handles)
                shift = n * (last.value - first.value) if mode == "cycle_offset" else 0
                out.extend((lo + shift, hi + shift))
        return out

    def _end_slope(self, pre: bool) -> float:
        """The slope linear infinity continues with at one end.

        Read off the END key's own interpolation (Blender's rule; Maya's
        reads the end tangent, which the adapter maps the same way): a held
        end extends flat, a linear one along the line to its neighbour, a
        Bezier one along its outer handle.  A degenerate end -- its neighbour
        on the same frame, a vertical handle -- extends flat, as Blender's
        does.
        """
        if len(self.keys) < 2:
            return 0.0
        if pre:
            k, nxt = self.keys[0], self.keys[1]
            handle = k.in_handle
        else:
            k, nxt = self.keys[-1], self.keys[-2]
            handle = k.out_handle
        interp = k.interpolation
        if interp in ("step", "stepnext"):
            return 0.0
        dx, dy = (
            (nxt.time - k.time, nxt.value - k.value) if interp == "linear" else handle
        )
        return dy / dx if abs(dx) > 1e-12 else 0.0

    def _infinity(self, t: float, pre: bool) -> float:
        first, last = self.keys[0], self.keys[-1]
        mode = self.pre_infinity if pre else self.post_infinity
        end = first if pre else last
        period = last.time - first.time
        if mode == "constant" or len(self.keys) < 2 or period <= 0:
            return end.value
        if mode == "linear":
            return end.value + (t - end.time) * self._end_slope(pre)
        cycles, offset = divmod(t - first.time, period)
        if mode == "oscillate" and int(cycles) % 2:
            local = last.time - offset
        else:
            local = first.time + offset
        value = self._inside(local)
        if mode == "cycle_offset":
            value += cycles * (last.value - first.value)
        return value

    def _clamped_dt(self, moved: Sequence[int], dt: float) -> float:
        moved_set = set(moved)
        times = self.times()
        for i in moved:
            j = i - 1
            while j >= 0 and j in moved_set:
                j -= 1
            # A neighbour already nearer than TIME_EPS stops the move; the
            # limit never turns it around.
            if j >= 0 and dt < 0:
                dt = max(dt, min(0.0, times[j] - times[i] + TIME_EPS))
            j = i + 1
            while j < len(times) and j in moved_set:
                j += 1
            if j < len(times) and dt > 0:
                dt = min(dt, max(0.0, times[j] - times[i] - TIME_EPS))
        return dt
