# !/usr/bin/python
# coding=utf-8
"""Render-effect key timelines -- the pulse and fade shapes, as ``(frame, value)`` pairs.

Pure and DCC-agnostic: what a keyer WRITES, planned before anything is written.
mayatk's and blendertk's render-effect writers key exactly these pairs, and the
WebXR preview publishes the same pairs straight into a GLB without keying
anything -- so the preview of a pulse and the pulse itself are one plan, not
two implementations that agree today.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pythontk.math_utils._math_utils import MathUtils

Key = Tuple[float, float]


class RampKeys:
    """Plan the linear keys of a render-effect ramp.

    Every plan is LINEAR keys, because the published ramp is read linearly
    (``MeshConvert.apply_glb_fades``): a spline through the extrema would ship
    as a triangle wave and lose the dwell.
    """

    #: The narrowest a pulse bracket may be. A gap of zero still needs the dim
    #: key to sit strictly BEFORE the bright one, or the two collide on one
    #: frame and the host keeps whichever landed last; a hundredth of a frame
    #: is invisible at any playback rate and survives float32 sampler buffers.
    #: Set by the tighter host: Blender MERGES an inserted key into an existing
    #: one within 0.01 frames, so the floor must clear that.
    PULSE_GAP_MIN = 0.05

    #: The same floor for a whole-frame pulse (the default): a snapped bracket
    #: has to be a whole frame wide, or it rounds onto the train key it exists
    #: to stay clear of.
    WHOLE_FRAME_GAP_MIN = 1.0

    FADE_DIRECTIONS = ("in", "out", "auto")

    @staticmethod
    def frames(whole: bool, *times: float) -> Tuple[float, ...]:
        """*times* as floats, snapped to whole frames when *whole*.

        Half-up rather than :func:`round`, whose banker's rounding would send
        two equal half-frames in one cycle to different frames.
        """
        return tuple(
            float(MathUtils.round_value(t, mode="half_up")) if whole else float(t)
            for t in times
        )

    @classmethod
    def pulse_gaps(
        cls,
        start: float,
        end: float,
        ramp: float,
        lead_in: Optional[float] = None,
        lead_out: Optional[float] = None,
        gap_min: Optional[float] = None,
    ) -> Tuple[float, float]:
        """``(head, tail)`` frames for a pulse's dim brackets, fitted to the window.

        ``None`` takes the cycle's own *ramp*. The pair is held to half the
        window so there is always as much pulse as bracket, and each is floored
        at *gap_min* (:attr:`PULSE_GAP_MIN`) so a bracket key never collides
        with a train key.
        """
        gap_min = cls.PULSE_GAP_MIN if gap_min is None else gap_min
        head = ramp if lead_in is None else max(0.0, float(lead_in))
        tail = ramp if lead_out is None else max(0.0, float(lead_out))
        budget = (float(end) - float(start)) / 2.0
        total = head + tail
        if total > budget and total > 0:
            scale = budget / total
            head, tail = head * scale, tail * scale
        return max(head, gap_min), max(tail, gap_min)

    @classmethod
    def pulse(
        cls,
        start: float,
        end: float,
        period: float,
        bright_fraction: float = 0.59,
        ramp_fraction: float = 0.25,
        lead_in: Optional[float] = None,
        lead_out: Optional[float] = None,
        whole_frames: bool = True,
    ) -> List[Key]:
        """A repeating bright/dim pulse over ``start..end``, bracketed dim at both ends.

        Four keys per cycle -- bright hold start, dim ramp end, dim hold end,
        bright ramp end. **The train is bracketed by dim keys**, and that is
        not cosmetic: a curve holds its first key value backwards and its last
        forwards, so a train that merely BEGAN bright glows for the whole
        timeline before it. Each bracket is the ramp between dim and the train,
        defaulting to the cycle's OWN ramp, so the ends read like every beat.

        Parameters:
            start: First frame; the value is dim here.
            end: Last frame; dim here too.
            period: One cycle, in FRAMES.
            bright_fraction: Share of the cycle spent bright (0-1).
            ramp_fraction: Share of the cycle spent in EACH transition (0-0.5);
                the holds take what remains.
            lead_in: Frames the pulse takes to come up from dim at *start*.
                ``None`` takes the cycle's own ramp; 0 cuts as hard as the
                floor allows (one frame under *whole_frames*).
            lead_out: The same at *end*, going back down to dim.
            whole_frames: Snap every key to a whole frame and widen the brackets
                to :attr:`WHOLE_FRAME_GAP_MIN`. The cycle still advances by the
                exact *period*, so only each key's placement is rounded and the
                cadence does not accumulate the rounding error.

        Returns:
            ``[(frame, value), ...]`` sorted by frame, one value per frame (a
            later key on the same frame replaces the earlier, as a host's does);
            empty when *period* is not positive or the window is empty.
        """
        start, end = cls.frames(whole_frames, start, end)
        if period <= 0 or end <= start:
            return []
        ramp = max(0.0, min(0.5, ramp_fraction)) * period
        bright = max(0.0, min(1.0, bright_fraction)) * period
        # Each hold gives up one ramp; the ramps then sit between the holds.
        bright_hold = max(0.0, bright - ramp)
        dim_hold = max(0.0, (period - bright) - ramp)
        cycle = (
            (0.0, 1.0),
            (bright_hold, 1.0),
            (bright_hold + ramp, 0.0),
            (bright_hold + ramp + dim_hold, 0.0),
        )
        gap_min = cls.WHOLE_FRAME_GAP_MIN if whole_frames else cls.PULSE_GAP_MIN
        head, tail = cls.pulse_gaps(start, end, ramp, lead_in, lead_out, gap_min)
        train_start, train_end = cls.frames(
            whole_frames, float(start) + head, float(end) - tail
        )

        keys: Dict[float, float] = {start: 0.0}  # the backward hold is dim
        last = None
        t0 = train_start
        while t0 < train_end:
            for offset, value in cycle:
                (t,) = cls.frames(whole_frames, t0 + offset)
                if t > train_end:
                    break
                keys[t] = value
                last = value
            t0 += period
        # The train's last value is stated at the cut, so the trail-out falls
        # over the gap it was given rather than over whatever is left of the
        # cycle it interrupted.
        if last is not None:
            keys[train_end] = last
        keys[end] = 0.0  # ...and the forward hold likewise
        return sorted(keys.items())

    @classmethod
    def fade(
        cls,
        start: float,
        end: float,
        direction: str = "in",
        whole_frames: bool = True,
    ) -> List[Key]:
        """A two-key ramp: ``"in"`` is 0 -> 1, ``"out"`` is 1 -> 0.

        ``"auto"`` is not planned here: it resolves per object from that
        object's last key, which only the host can read.

        Raises:
            ValueError: For a direction other than ``"in"`` / ``"out"``.
        """
        if direction not in ("in", "out"):
            raise ValueError(
                f"A fade plan needs 'in' or 'out', not {direction!r} -- resolve "
                "'auto' against the object's keys first."
            )
        start, end = cls.frames(whole_frames, start, end)
        low, high = (0.0, 1.0) if direction == "in" else (1.0, 0.0)
        return [(start, low), (end, high)]

    @classmethod
    def fade_loop(
        cls, duration: float, hold: float = 0.0, direction: str = "in"
    ) -> List[Key]:
        """One fade framed by the holds either side of it, from frame 0.

        What a fade LOOKS like played on repeat: the curve genuinely sits at its
        first value before the ramp and at its last after it, so a loop that
        showed the ramp alone would read as a pulse. ``"auto"`` ramps both ways
        -- in, hold, back out -- because the direction is decided per object at
        apply time, and a preview that picked one would claim to know it.

        Parameters:
            duration: Frames the ramp itself takes.
            hold: Frames held at each end.
            direction: ``"in"``, ``"out"`` or ``"auto"``; anything else reads
                as ``"auto"``.
        """
        duration, hold = max(0.0, float(duration)), max(0.0, float(hold))
        direction = direction if direction in cls.FADE_DIRECTIONS else "auto"
        low, high = (1.0, 0.0) if direction == "out" else (0.0, 1.0)
        plan = [(0.0, low), (hold, low), (hold + duration, high)]
        plan.append((plan[-1][0] + hold, high))
        if direction == "auto":
            plan.append((plan[-1][0] + duration, low))
        keys: Dict[float, float] = {}
        for frame, value in plan:
            keys[frame] = value
        return sorted(keys.items())
