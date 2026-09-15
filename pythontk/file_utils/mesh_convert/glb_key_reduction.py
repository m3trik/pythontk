# !/usr/bin/python
# coding=utf-8
"""Reduce a GLB's animation keys to what its interpolation needs.

Why this exists: a converter bakes a key on EVERY frame of every channel
(FBX2glTF resamples per frame whatever the FBX carried, so a DCC-side key
optimisation never reaches the deliverable), which makes the animation 10-15x
the size of the motion in it. Measured on a production 4K assembly: 2,180,161
keys over 18 clips, 23.5 MiB of a 155 MiB GLB, of which 148,347 (6.8%)
reproduce every original sample within 1e-4 -- 0.1 mm of translation, 0.006
degrees of rotation.

What it does, per sampler:

* ``LINEAR`` -- Ramer-Douglas-Peucker over the sample list, the error measured
  the way the viewer will interpolate: lerp for translation / scale / weights /
  pointer floats, shortest-arc slerp for a rotation (the rules
  :class:`~pythontk.file_utils.mesh_convert.glb_clips.GlbClips` evaluates
  by). Every original sample is reproduced within the tolerance by the keys
  that survive -- and because both the original and the reduced curve are
  piecewise linear with vertices at the samples, the bound holds between
  samples too. The first and last keys always stay, so no clip changes
  duration or its origin (visibility gates, fades and ``KHR_animation_pointer``
  channels keep their timing).
* ``STEP`` -- drop every interior key whose value repeats its predecessor's:
  lossless, the held value is the same at every time.
* ``CUBICSPLINE`` -- untouched; tangents at a dropped key are not derivable
  from its neighbours.

Refused, and left as exported: a sampler whose output accessor another sampler
also reads (reducing it under one time input would freeze the other), a
sampler two channels drive to different paths, a non-float or sparse layout,
and a morph-weights output (its element count is keys x targets, which the
reader declines rather than half-understands).

Runs on the closed deliverable after the clips are rebuilt and the constant
channels collapsed -- it can only shrink what is already there -- and is safe
to run standalone on any GLB.
"""

from __future__ import annotations

import logging
import math
import struct
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pythontk.file_utils.mesh_convert.glb_clips import GlbClips

__all__ = ["GlbKeyReduction"]

logger = logging.getLogger(__name__)

#: ``(times, values, interpolation)`` as :meth:`GlbClips._read_sampler` returns.
Sampler = Tuple[List[float], List[Tuple[float, ...]], str]


class _GlbKeyReductionInternal:
    """Internal helpers for :class:`GlbKeyReduction`."""

    @staticmethod
    def _error(
        approx: Sequence[float], value: Sequence[float], quaternion: bool
    ) -> float:
        """Max component error of *approx* against *value*.

        A rotation is read sign-agnostically -- ``q`` and ``-q`` are the same
        rotation, and slerp's shortest-arc rule may return either sign.
        """
        direct = max(abs(a - b) for a, b in zip(approx, value))
        if not quaternion:
            return direct
        flipped = max(abs(a + b) for a, b in zip(approx, value))
        return min(direct, flipped)

    @classmethod
    def _segment_error(
        cls,
        times: Sequence[float],
        values: Sequence[Sequence[float]],
        lo: int,
        hi: int,
        quaternion: bool,
    ) -> Tuple[int, float]:
        """The worst interior sample of ``(lo, hi)`` against the chord joining
        its ends, as the viewer would interpolate it.

        Returns ``(index, error)``; ``(-1, 0.0)`` when the segment has no
        interior. The slerp constants are per SEGMENT, not per sample, which
        is what keeps a 2-million-key pass in pure Python affordable.
        """
        a, b = tuple(values[lo]), tuple(values[hi])
        span = times[hi] - times[lo]
        worst_i, worst_e = -1, 0.0
        theta = sin_theta = 0.0
        if quaternion:
            dot = sum(x * y for x, y in zip(a, b))
            if dot < 0.0:  # shortest arc, the glTF LINEAR rule for rotations
                b = tuple(-x for x in b)
                dot = -dot
            if dot <= 0.9995:  # else nearly parallel: lerp, then renormalize
                theta = math.acos(max(-1.0, min(1.0, dot)))
                sin_theta = math.sin(theta)
        for i in range(lo + 1, hi):
            u = (times[i] - times[lo]) / span if span > 0 else 0.0
            if quaternion:
                if sin_theta:
                    wa = math.sin((1.0 - u) * theta) / sin_theta
                    wb = math.sin(u * theta) / sin_theta
                    out = [x * wa + y * wb for x, y in zip(a, b)]
                else:
                    out = [x + (y - x) * u for x, y in zip(a, b)]
                norm = math.sqrt(sum(x * x for x in out)) or 1.0
                error = cls._error([x / norm for x in out], values[i], True)
            else:
                error = max(
                    abs(x + (y - x) * u - v) for x, y, v in zip(a, b, values[i])
                )
            if error > worst_e:
                worst_i, worst_e = i, error
        return worst_i, worst_e

    @classmethod
    def _keep_linear(
        cls,
        times: Sequence[float],
        values: Sequence[Sequence[float]],
        quaternion: bool,
        tolerance: float,
    ) -> List[int]:
        """Ramer-Douglas-Peucker: the indices whose keys reproduce every sample
        within *tolerance* under the viewer's interpolation. Both ends stay."""
        count = len(times)
        if count <= 2:
            return list(range(count))
        keep = [0, count - 1]
        pending = [(0, count - 1)]
        while pending:
            lo, hi = pending.pop()
            if hi - lo < 2:
                continue
            index, error = cls._segment_error(times, values, lo, hi, quaternion)
            if index >= 0 and error > tolerance:
                keep.append(index)
                pending.append((lo, index))
                pending.append((index, hi))
        keep.sort()
        return keep

    @staticmethod
    def _keep_step(values: Sequence[Sequence[float]]) -> List[int]:
        """Every change point plus both ends: what a STEP sampler holds at any
        time is unchanged, so this is lossless."""
        count = len(values)
        if count <= 2:
            return list(range(count))
        keep = [0]
        keep.extend(i for i in range(1, count - 1) if values[i] != values[i - 1])
        keep.append(count - 1)
        return keep


class GlbKeyReduction(_GlbKeyReductionInternal):
    """Tolerance-bound key reduction for a GLB's animation samplers.

    :meth:`reduce` is the pass; :meth:`evaluate` and :meth:`deviation` are the
    same interpolation the pass measures by, exposed so a caller (a test, an
    accuracy report) can verify a reduced file against the original with the
    rules the viewer applies rather than a private re-implementation.
    """

    @staticmethod
    def read_sampler(
        gltf: Dict[str, Any], blob: Optional[bytes], sampler: Dict[str, Any]
    ) -> Optional[Sampler]:
        """``(times, values, interpolation)`` for one sampler, or ``None``."""
        return GlbClips._read_sampler(gltf, blob, sampler)

    @staticmethod
    def evaluate(
        times: Sequence[float],
        values: Sequence[Sequence[float]],
        at: float,
        interpolation: str = "LINEAR",
        quaternion: bool = False,
    ) -> Tuple[float, ...]:
        """The sampler's value at *at* under glTF's rules, held outside its range."""
        return GlbClips._evaluate(times, values, at, interpolation, quaternion)

    @classmethod
    def deviation(
        cls, reference: Sampler, candidate: Sampler, quaternion: bool = False
    ) -> float:
        """Worst component error of *candidate* against *reference*, sampled at
        every reference time -- the bound :meth:`reduce` promises."""
        times, values, _interpolation = reference
        c_times, c_values, c_interpolation = candidate
        worst = 0.0
        for at, value in zip(times, values):
            approx = cls.evaluate(c_times, c_values, at, c_interpolation, quaternion)
            worst = max(worst, cls._error(approx, value, quaternion))
        return worst

    @classmethod
    def reduce(
        cls,
        glb: Any,
        tolerance: float,
        rotation_tolerance: Optional[float] = None,
    ) -> Dict[str, int]:
        """Rewrite every reducible sampler of *glb* with the keys it needs.

        Parameters:
            glb: Path to a ``.glb``, modified in place, or an open
                ``MeshConvert.GlbEdit`` session (the owner writes).
            tolerance: Max deviation any original sample may show under the
                viewer's interpolation, in the sampler's own units -- meters
                for translation and scale, quaternion components for rotation
                (1e-4 is 0.1 mm / 0.006 degrees), plain floats for a pointer.
            rotation_tolerance: A separate bound for rotations; ``None`` takes
                *tolerance*.

        Returns:
            ``{"samplers": n, "keys_before": n, "keys_after": n, "bytes": n}``
            -- samplers rewritten, keys over every sampler the pass READ (so
            a refused or already-minimal sampler counts on both sides), and
            BIN payload reclaimed. A file with nothing to reduce is not
            rewritten.

        Raises:
            ValueError: *tolerance* (or *rotation_tolerance*) is not positive.
        """
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        tolerance = float(tolerance)
        if not tolerance > 0:
            raise ValueError(f"tolerance must be positive, got {tolerance!r}")
        if rotation_tolerance is not None and not float(rotation_tolerance) > 0:
            raise ValueError(
                f"rotation_tolerance must be positive, got {rotation_tolerance!r}"
            )
        summary = {"samplers": 0, "keys_before": 0, "keys_after": 0, "bytes": 0}
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
            animations = gltf.get("animations") or []
            accessors = gltf.get("accessors") or []
            if not animations or not accessors:
                return summary
            # An output another sampler also reads is refused: reducing it
            # under one sampler's new time input would freeze the other.
            output_users: Counter = Counter(
                sampler.get("output")
                for animation in animations
                for sampler in animation.get("samplers") or []
            )
            jobs: List[tuple] = []
            payloads: List[bytes] = []
            for animation in animations:
                samplers = animation.get("samplers") or []
                paths: Dict[int, set] = {}
                for channel in animation.get("channels") or []:
                    index = channel.get("sampler")
                    if isinstance(index, int):
                        paths.setdefault(index, set()).add(
                            (channel.get("target") or {}).get("path")
                        )
                for index, sampler in enumerate(samplers):
                    interpolation = str(sampler.get("interpolation") or "LINEAR")
                    if interpolation == "CUBICSPLINE":
                        continue
                    output = sampler.get("output")
                    if output_users.get(output, 0) != 1:
                        continue
                    driven = paths.get(index) or set()
                    quaternion = driven == {"rotation"}
                    if "rotation" in driven and not quaternion:
                        continue  # one sampler, two kinds of target: ambiguous
                    read = GlbClips._read_sampler(gltf, edit.bin_data, sampler)
                    if read is None:
                        continue
                    times, values, _ = read
                    count = len(times)
                    summary["keys_before"] += count
                    if count <= 2:
                        summary["keys_after"] += count
                        continue
                    bound = (
                        float(rotation_tolerance)
                        if quaternion and rotation_tolerance is not None
                        else tolerance
                    )
                    keep = (
                        cls._keep_step(values)
                        if interpolation == "STEP"
                        else cls._keep_linear(times, values, quaternion, bound)
                    )
                    summary["keys_after"] += len(keep)
                    if len(keep) >= count:
                        continue
                    kept_times = [times[i] for i in keep]
                    flat = [c for i in keep for c in values[i]]
                    jobs.append((sampler, output, len(kept_times), len(payloads)))
                    payloads.append(struct.pack(f"<{len(kept_times)}f", *kept_times))
                    payloads.append(struct.pack(f"<{len(flat)}f", *flat))
            if not jobs:
                return summary
            added = MeshConvert._append_bin_views(edit, payloads)
            if not added:  # external buffer: nothing this pass can do
                logger.warning(
                    "Animation: this GLB's buffer is external, so the reduced "
                    "samplers have nowhere to live -- keys left as exported."
                )
                summary["keys_after"] = summary["keys_before"]
                return summary
            replaced_inputs: set = set()
            for sampler, output, count, slot in jobs:
                # Read back OUT of the packed bytes: `min`/`max` must be the
                # values the file stores, and float32 can round them.
                lo = struct.unpack("<f", payloads[slot][:4])[0]
                hi = struct.unpack("<f", payloads[slot][-4:])[0]
                accessors.append(
                    {
                        "bufferView": added[slot],
                        "componentType": 5126,
                        "count": count,
                        "type": "SCALAR",
                        "min": [lo],
                        "max": [hi],
                    }
                )
                # A converter shares one time input across a clip; each reduced
                # sampler keeps a DIFFERENT subset, so it gets its own. The
                # output is re-pointed in place -- the same channel, shorter --
                # which is also what lets the old view be reclaimed below.
                replaced_inputs.add(sampler.get("input"))
                sampler["input"] = len(accessors) - 1
                accessor = accessors[output]
                accessor["bufferView"] = added[slot + 1]
                accessor.pop("byteOffset", None)  # the new view starts at the data
                accessor["count"] = count
                accessor.pop("min", None)
                accessor.pop("max", None)
                summary["samplers"] += 1
            edit.dirty = True
            MeshConvert._drop_orphaned_accessors(edit, replaced_inputs)
            summary["bytes"] = MeshConvert._compact_bin(edit)
            logger.info(
                "Animation: reduced %d sampler(s) from %d to %d key(s) (%.1f%% "
                "kept) within %g, reclaiming %.2f MB.",
                summary["samplers"],
                summary["keys_before"],
                summary["keys_after"],
                100.0 * summary["keys_after"] / max(1, summary["keys_before"]),
                tolerance,
                summary["bytes"] / 1048576.0,
            )
        return summary
