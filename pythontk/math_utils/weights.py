# !/usr/bin/python
# coding=utf-8
"""Weight math for blendShape / shape-key morph animation — pure, DCC-agnostic.

Frame→weight mapping, evenly-spaced weight generation, and consistent rounding
(3 dp — Maya's blendShape precision, and fine for Blender shape keys). Shared by
mayatk's and blendertk's ``blendshape_animator`` (both call ``Weights`` directly).
"""
from typing import List, Tuple


class Weights:
    """Blend-weight calculations with consistent rounding precision."""

    PRECISION = 3  # 3 decimal places (Maya blendShape precision; ample for shape keys)

    @classmethod
    def round_weight(cls, weight: float) -> float:
        """Round a weight to the shared precision (:data:`PRECISION` decimal places)."""
        return round(float(weight), cls.PRECISION)

    @classmethod
    def frame_to_weight(cls, frame: int, start_frame: int, end_frame: int) -> float:
        """Convert frame number to blendShape weight."""
        if frame <= start_frame:
            return 0.0
        if frame >= end_frame:
            return 1.0

        frame_range = end_frame - start_frame
        frame_offset = frame - start_frame
        return cls.round_weight(frame_offset / float(frame_range))

    @classmethod
    def generate_weights(
        cls,
        count: int,
        weight_range: Tuple[float, float] = (0.0, 1.0),
        include_endpoints: bool = False,
    ) -> List[float]:
        """Generate ``count`` evenly spaced weights within ``weight_range``.

        ``count`` always equals ``len(returned)`` regardless of ``include_endpoints``.
        With ``include_endpoints=True`` the first and last entries are exactly
        ``weight_range[0]`` and ``weight_range[1]`` (requires ``count >= 2``).
        With ``include_endpoints=False`` the entries lie strictly inside the range.
        """
        if count < 1:
            return []
        start, end = weight_range

        if include_endpoints:
            if count == 1:
                weights = [start]
            else:
                step = (end - start) / float(count - 1)
                weights = [start + step * i for i in range(count)]
        else:
            step = (end - start) / float(count + 1)
            weights = [start + step * i for i in range(1, count + 1)]

        return [cls.round_weight(w) for w in weights]
