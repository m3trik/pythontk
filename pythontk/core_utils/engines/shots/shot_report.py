# !/usr/bin/python
# coding=utf-8
"""What a shots panel says about the store: pure, one-line summaries.

The text both DCC Shots panels put in their footer -- the sequence at a glance,
and the outcome of a trim / pad -- worded once here, from plain shot records and
``(head, tail)`` deltas, so the two hosts cannot drift into different reports.
"""

from typing import Iterable, Sequence, Tuple

#: Below this a delta is float noise, not a move.
EPSILON = 1e-6


class ShotReport:
    """One-line summaries of a shot sequence and of boundary edits on it."""

    SEPARATOR = " · "

    @classmethod
    def summary(cls, shots: Sequence) -> str:
        """The sequence at a glance: ``"3 shots · 240f · 5 objects · [1–240]"``.

        Parameters:
            shots: Shot records in timeline order (``ShotStore.sorted_shots()``),
                each with ``start`` / ``end`` / ``duration`` / ``objects``.

        Returns:
            ``""`` for no shots.
        """
        if not shots:
            return ""
        n = len(shots)
        objs = {o for s in shots for o in s.objects}
        total_dur = sum(s.duration for s in shots)
        parts = [
            f"{n} shot{'s' if n != 1 else ''}",
            f"{total_dur:.0f}f",
            f"{len(objs)} object{'s' if len(objs) != 1 else ''}",
            f"[{shots[0].start:.0f}–{shots[-1].end:.0f}]",
        ]
        return cls.SEPARATOR.join(parts)

    @staticmethod
    def moved(deltas: Iterable[Tuple[float, float]]) -> bool:
        """Whether any ``(head, tail)`` delta actually moved a boundary."""
        return any(abs(d) > EPSILON for pair in deltas for d in pair)

    @classmethod
    def delta_summary(cls, label: str, deltas: Sequence[Tuple[float, float]]) -> str:
        """The outcome of a trim / pad: frames moved at the heads and the tails.

        ``"<label>: nothing to do"`` when no boundary moved (the caller then
        also drops the edit's restore point, or the next undo would visibly do
        nothing).
        """
        if not cls.moved(deltas):
            return f"{label}: nothing to do"
        head = sum(abs(pair[0]) for pair in deltas)
        tail = sum(abs(pair[1]) for pair in deltas)
        return f"{label}: {head:.0f}f head, {tail:.0f}f tail"
