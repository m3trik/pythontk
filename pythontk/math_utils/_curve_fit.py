# !/usr/bin/python
# coding=utf-8
"""Cubic Hermite fitting: slopes from samples, evaluation, and reducing a dense
sample run to the fewest keys within tolerance (the bodies behind the
:class:`MathUtils` facade).
"""

from __future__ import annotations

from typing import List, Tuple, Sequence, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class _MathCurveFitInternal:
    """Bodies of :class:`MathUtils`' Hermite fitting (a ``MathUtils`` base).

    The public signatures and docstrings stay on the facade; each delegates
    here.
    """

    @staticmethod
    def fit_hermite_slopes(
        times: Sequence[float],
        values: Sequence[float],
        keep_indices: Sequence[int],
        flat_tolerance: float = 0.0,
    ) -> Tuple[List[float], List[float]]:
        """Body of :meth:`MathUtils.fit_hermite_slopes`."""
        import numpy as np

        t = np.asarray(times, dtype=float)
        v = np.asarray(values, dtype=float)
        keep = np.asarray(keep_indices, dtype=int)
        k_count = len(keep)
        if k_count == 0:
            return [], []
        if k_count == 1 or len(t) < 2:
            return [0.0] * k_count, [0.0] * k_count

        # Weak prior: the sampled derivative at each kept key.  It only
        # decides keys the dropped samples leave unconstrained (adjacent
        # kept samples) and breaks least-squares ties; on a constrained key
        # its pull is ~1e-6 relative.
        prior = np.gradient(v, t)[keep]

        # Everything per SEGMENT is computed over the samples at once: a
        # numpy call per segment (the hold test alone) was 130 s of a
        # production extremes pass over ~2,500 baked curves.
        seg_count = k_count - 1
        lengths = np.diff(keep)
        if np.any(lengths < 0):
            raise ValueError("keep_indices must be ascending.")
        # Sample j in (keep[k], keep[k+1]] belongs to segment k -- the spans
        # tile keep[0]+1 .. keep[-1] exactly (a zero-length segment covers
        # nothing).
        seg_of = np.repeat(np.arange(seg_count), lengths)
        j = np.arange(keep[0] + 1, keep[-1] + 1)
        i0 = keep[:-1][seg_of]
        i1 = keep[1:][seg_of]

        # A hold: every sample of the segment within flat_tolerance of its
        # start value (the start itself deviates by 0). A zero-length
        # segment is trivially one.
        worst = np.zeros(seg_count)
        spans = lengths > 0
        if spans.any():
            # Non-empty spans are contiguous in ``j``: one reduceat, each
            # starting at its segment's first sample (keep[k] + 1).
            worst[spans] = np.maximum.reduceat(
                np.abs(v[j] - v[i0]), keep[:-1][spans] - keep[0]
            )
        flat = worst <= flat_tolerance

        # Each dropped sample couples only the two keys bounding its segment,
        # so the normal equations (A^T A + lam^2 I) m = A^T b + lam^2 prior
        # are tridiagonal: accumulate them per segment and solve in O(keys)
        # rather than materialise a samples x keys matrix (a production
        # bake is tens of thousands of samples by thousands of extrema).
        lam2 = 1e-6
        diag = np.full(k_count, lam2)
        off = np.zeros(seg_count)
        rhs = lam2 * prior
        dt_all = t[i1] - t[i0]
        use = (j < i1) & ~flat[seg_of] & (i1 - i0 >= 2) & (dt_all > 0)
        if use.any():
            seg = seg_of[use]
            left, right = i0[use], i1[use]
            dt = dt_all[use]
            s = (t[j[use]] - t[left]) / dt
            h00 = 2 * s**3 - 3 * s**2 + 1
            h01 = -2 * s**3 + 3 * s**2
            a = (s**3 - 2 * s**2 + s) * dt  # weight on the left key's out slope
            c = (s**3 - s**2) * dt  # weight on the right key's in slope
            r = v[j[use]] - h00 * v[left] - h01 * v[right]
            diag[:-1] += np.bincount(seg, a * a, seg_count)
            diag[1:] += np.bincount(seg, c * c, seg_count)
            off += np.bincount(seg, a * c, seg_count)
            rhs[:-1] += np.bincount(seg, a * r, seg_count)
            rhs[1:] += np.bincount(seg, c * r, seg_count)

        # Thomas sweep (the system is symmetric positive definite), on plain
        # floats: the same IEEE arithmetic without a numpy scalar per step.
        d, o, b = diag.tolist(), off.tolist(), rhs.tolist()
        c_prime = [0.0] * k_count
        d_prime = [0.0] * k_count
        c_prime[0] = o[0] / d[0]
        d_prime[0] = b[0] / d[0]
        for k in range(1, k_count):
            denom = d[k] - o[k - 1] * c_prime[k - 1]
            c_prime[k] = o[k] / denom if k < k_count - 1 else 0.0
            d_prime[k] = (b[k] - o[k - 1] * d_prime[k - 1]) / denom
        slopes = [0.0] * k_count
        slopes[-1] = d_prime[-1]
        for k in range(k_count - 2, -1, -1):
            slopes[k] = d_prime[k] - c_prime[k] * slopes[k + 1]

        # A hold pins the slopes facing into it; the key's other side keeps
        # its fitted slope (a broken tangent).
        in_slopes = np.asarray(slopes)
        out_slopes = in_slopes.copy()
        out_slopes[:-1][flat] = 0.0
        in_slopes[1:][flat] = 0.0
        # An endpoint has one facing side; its outward side mirrors it so the
        # key reads as unified rather than broken.
        in_slopes[0] = out_slopes[0]
        out_slopes[-1] = in_slopes[-1]
        return in_slopes.tolist(), out_slopes.tolist()

    @staticmethod
    def evaluate_hermite(
        times: Sequence[float],
        values: Sequence[float],
        keep_indices: Sequence[int],
        in_slopes: Sequence[float],
        out_slopes: Sequence[float],
        at: Optional[Sequence[float]] = None,
    ) -> "np.ndarray":
        """Body of :meth:`MathUtils.evaluate_hermite`."""
        import numpy as np

        t = np.asarray(times, dtype=float)
        v = np.asarray(values, dtype=float)
        keep = np.asarray(keep_indices, dtype=int)
        x = t if at is None else np.asarray(at, dtype=float)
        if len(keep) == 0:
            return np.zeros(len(x))
        kt, kv = t[keep], v[keep]
        if len(keep) == 1:
            return np.full(len(x), kv[0])
        m_out = np.asarray(out_slopes, dtype=float)
        m_in = np.asarray(in_slopes, dtype=float)
        seg = np.clip(np.searchsorted(kt, x, side="right") - 1, 0, len(kt) - 2)
        t0, t1 = kt[seg], kt[seg + 1]
        dt = t1 - t0
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(dt > 0, (x - t0) / dt, 0.0)
        s = np.clip(s, 0.0, 1.0)  # hold the end values past the kept range
        s2, s3 = s * s, s * s * s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        return (
            h00 * kv[seg]
            + h10 * dt * m_out[seg]
            + h01 * kv[seg + 1]
            + h11 * dt * m_in[seg + 1]
        )

    @staticmethod
    def reduce_samples(
        times: Sequence[float],
        values: Sequence[float],
        value_tolerance: float = 1e-5,
        max_error: Optional[float] = None,
    ) -> Tuple[List[int], List[float], List[float]]:
        """Body of :meth:`MathUtils.reduce_samples`."""
        import numpy as np

        from pythontk.iter_utils._iter_utils import IterUtils

        t = np.asarray(times, dtype=float)
        v = np.asarray(values, dtype=float)
        n = len(v)
        keep: List[int] = [
            int(i) for i in IterUtils.find_extrema_indices(v, value_tolerance)
        ]
        if max_error is None:
            max_error = 0.01 * float(v.max() - v.min()) if n else 0.0
        keep_set = set(keep)
        while True:
            in_slopes, out_slopes = _MathCurveFitInternal.fit_hermite_slopes(
                t, v, keep, flat_tolerance=value_tolerance
            )
            if max_error <= 0 or len(keep) >= n:
                break
            error = np.abs(
                _MathCurveFitInternal.evaluate_hermite(
                    t, v, keep, in_slopes, out_slopes
                )
                - v
            )
            # The worst sample strictly inside every segment with an interior,
            # found in one reduceat rather than a numpy call per segment; only
            # the (few) segments over the bound need their argmax.
            kept = np.asarray(keep)
            gaps = np.nonzero(np.diff(kept) >= 2)[0]
            if not len(gaps):
                break
            lo, hi = kept[:-1][gaps] + 1, kept[1:][gaps]
            bounds = np.empty(2 * len(gaps), dtype=int)
            bounds[0::2], bounds[1::2] = lo, hi
            over = np.maximum.reduceat(error, bounds)[0::2] > max_error
            if not over.any():
                break
            for a, b in zip(lo[over].tolist(), hi[over].tolist()):
                keep_set.add(a + int(np.argmax(error[a:b])))
            keep = sorted(keep_set)
        return keep, list(in_slopes), list(out_slopes)
