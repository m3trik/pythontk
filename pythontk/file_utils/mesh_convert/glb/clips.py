# !/usr/bin/python
# coding=utf-8
"""Read and evaluate a GLB's animation samplers.

The shared sampler arithmetic of the animation passes: decoding an accessor
and a sampler (float components only, anything else declined rather than
half-understood) and evaluating a channel at any time with the sampler's own
interpolation (STEP, LINEAR with shortest-arc quaternions, CUBICSPLINE).

``glb.key_reduction`` reads and evaluates the samplers it reduces. The clips
themselves are the file's takes, one per shot as the FBX carries them, and a
render effect's curve is the file's own per take, so nothing here rebuilds a
clip or cuts one.
"""

from __future__ import annotations

import bisect
import logging
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pythontk.math_utils._math_utils import MathUtils
from pythontk.math_utils.quaternion import Quaternion

__all__ = ["GlbClips"]

logger = logging.getLogger(__name__)

#: Number of components per glTF accessor type.
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
#: The one component type an animation sampler may use for its input, and the
#: only one this pass will touch on an output. Quantized outputs (normalized
#: byte/short rotations) are legal glTF but nothing in this pipeline writes
#: them, and a half-understood requantization would be worse than declining.
_FLOAT = 5126


class _GlbClipsInternal:
    """Internal helpers for :class:`GlbClips`."""

    # ---- reading ------------------------------------------------------

    @staticmethod
    def _read_accessor(
        gltf: Dict[str, Any], blob: Optional[bytes], index: Any
    ) -> Optional[List[Tuple[float, ...]]]:
        """Decode accessor *index* as a list of float tuples, or ``None``.

        ``None`` for anything this pass will not rewrite faithfully -- a
        non-float component type, a sparse accessor, a missing bufferView -- so
        the caller can decline the whole rebuild rather than emit a clip it
        half-understood.
        """
        accessors = gltf.get("accessors") or []
        if not isinstance(index, int) or not 0 <= index < len(accessors):
            return None
        acc = accessors[index] or {}
        if acc.get("componentType") != _FLOAT or "sparse" in acc:
            return None
        width = _WIDTH.get(acc.get("type"))
        view_index = acc.get("bufferView")
        if width is None or not isinstance(view_index, int) or blob is None:
            return None
        views = gltf.get("bufferViews") or []
        if not 0 <= view_index < len(views):
            return None
        view = views[view_index] or {}
        base = int(view.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
        count = int(acc.get("count", 0))
        # An interleaved view strides past the elements between ours; a packed
        # one (the norm for animation) has no stride and reads contiguously.
        stride = int(view.get("byteStride") or 0) or width * 4
        end = base + (count - 1) * stride + width * 4 if count else base
        if end > len(blob):
            return None
        try:
            return [
                struct.unpack_from(f"<{width}f", blob, base + i * stride)
                for i in range(count)
            ]
        except struct.error:
            return None

    @classmethod
    def _read_sampler(
        cls, gltf: Dict[str, Any], blob: Optional[bytes], sampler: Dict[str, Any]
    ) -> Optional[Tuple[List[float], List[Tuple[float, ...]], str]]:
        """``(times, values, interpolation)`` for one sampler, or ``None``."""
        times = cls._read_accessor(gltf, blob, sampler.get("input"))
        values = cls._read_accessor(gltf, blob, sampler.get("output"))
        if times is None or values is None or not times:
            return None
        interpolation = str(sampler.get("interpolation") or "LINEAR")
        flat = [t[0] for t in times]
        if interpolation == "CUBICSPLINE":
            # Three entries per key (in-tangent, value, out-tangent). Slicing
            # one would need tangents at the cut, which are not derivable from
            # the neighbours -- so it is flattened to the key values and the
            # clip carries LINEAR. Nothing in this pipeline emits CUBICSPLINE;
            # this is the path that keeps a foreign file from being corrupted.
            if len(values) != 3 * len(flat):
                return None
            values = values[1::3]
            interpolation = "LINEAR"
        if len(values) != len(flat):
            return None
        return flat, list(values), interpolation

    # ---- sampling -----------------------------------------------------

    @staticmethod
    def _upper(times: Sequence[float], at: float) -> int:
        """Index of the key *at* interpolates toward -- always a valid pair.

        ``bisect_right``, so a time that IS a key pairs that key with its
        successor (``u == 0``): LINEAR reads the key either way, but STEP holds
        ``values[lo]`` and glTF says a STEP value holds FROM its key until the
        next -- ``bisect_left`` paired the key with its predecessor and read a
        gate one key late at its own change point. Clamped into
        ``[1, len - 1]`` so the caller can read ``lo = hi - 1``
        unconditionally; the out-of-range cases are handled before this runs.
        """
        return max(1, min(len(times) - 1, bisect.bisect_right(times, at)))

    @classmethod
    def _evaluate(
        cls,
        times: Sequence[float],
        values: Sequence[Sequence[float]],
        at: float,
        interpolation: str,
        quaternion: bool,
    ) -> Tuple[float, ...]:
        """The sampler's value at *at*, held outside its own range."""
        if at <= times[0]:
            return tuple(values[0])
        if at >= times[-1]:
            return tuple(values[-1])
        hi = cls._upper(times, at)
        lo = hi - 1
        if interpolation == "STEP":
            return tuple(values[lo])
        span = times[hi] - times[lo]
        if span <= 0:
            return tuple(values[lo])
        u = (at - times[lo]) / span
        if quaternion:  # shortest arc, the glTF LINEAR rule for rotations
            return Quaternion.slerp(values[lo], values[hi], u)
        return MathUtils.lerp(tuple(values[lo]), tuple(values[hi]), u)


class GlbClips(_GlbClipsInternal):
    """Read a GLB's animation samplers: the decoding and evaluation the other
    animation passes share (``glb.key_reduction`` reads and evaluates the
    samplers it reduces)."""
