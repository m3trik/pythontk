# !/usr/bin/python
# coding=utf-8
"""What a shots panel says about the store: pure summaries and questions.

The text the Shots panel (tentacle's, one for both hosts) and each host's Shot
Sequencer put in front of the user -- the sequence at a glance, the outcome of a
trim / pad, and the Delete Shots scopes with the question they ask -- worded
once here, from plain shot records and ``(head, tail)`` deltas, so no two panels
can drift into different words.
"""

from typing import Iterable, Sequence, Tuple

#: Below this a delta is float noise, not a move.
EPSILON = 1e-6


class ShotReport:
    """One-line summaries of a shot sequence and of boundary edits on it, and
    the words of a delete by scope (``ShotStore.shots_in_scope``)."""

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

    # ---- deleting shots by scope -------------------------------------------

    #: The delete scopes (``ShotStore.DELETE_SCOPES``) as ``(label, scope)``,
    #: narrowest first: the rows the Shots panel's Delete Shots picker and the
    #: Shot Sequencer's shot-list menu offer.
    SCOPE_LABELS = (("Stale", "stale"), ("Empty", "empty"), ("All", "all"))

    #: What each scope takes, as the clause after "shot(s)" -- "Delete 3
    #: shot(s) <clause>?" -- for the question and the menus' tooltips.
    SCOPE_CLAUSES = {
        "stale": "whose objects are all gone from the file and which key nothing",
        "empty": "with nothing keyed in their frames",
        "all": "",
    }

    #: How many shots :meth:`removal_question` names before "... and N more".
    LISTED_MAX = 12

    @classmethod
    def scope_label(cls, scope: str) -> str:
        """The menu word for *scope*: ``"stale"`` -> ``"Stale"``.

        Raises:
            ValueError: *scope* is none of :attr:`SCOPE_LABELS`.
        """
        for label, known in cls.SCOPE_LABELS:
            if known == scope:
                return label
        scopes = [known for _label, known in cls.SCOPE_LABELS]
        raise ValueError(f"shot delete scope {scope!r} is not one of {scopes}")

    @classmethod
    def scope_count(cls, shots: Sequence, scope: str) -> str:
        """*shots* counted the way the questions say it: ``"all 4 shot(s)"``,
        ``"2 shot(s) with nothing keyed in their frames"``."""
        if scope == "all":
            return f"all {len(shots)} shot(s)"
        return f"{len(shots)} shot(s) {cls.SCOPE_CLAUSES[scope]}"

    @classmethod
    def removal_question(cls, shots: Sequence, scope: str) -> Tuple[str, str]:
        """The question asked before the shots of *scope* go, naming them.

        One question for every entry point -- the Shots panel's Delete Shots
        and the Shot Sequencer's shot list -- so they cannot drift; each asks
        it in its own dialog.

        Parameters:
            shots: The shots the delete takes (``ShotStore.shots_in_scope``),
                each with ``name`` / ``start`` / ``end``.
            scope: ``"stale"``, ``"empty"`` or ``"all"``.

        Returns:
            ``(title, text)``: "Delete Stale Shots", and the shots counted and
            listed (the first :attr:`LISTED_MAX`) above what a delete by scope
            keeps -- every key, and every other shot where it is.
        """
        listed = "\n".join(
            f"  {s.name} [{s.start:.0f}–{s.end:.0f}]" for s in shots[: cls.LISTED_MAX]
        )
        if len(shots) > cls.LISTED_MAX:
            listed += f"\n  … and {len(shots) - cls.LISTED_MAX} more"
        return (
            f"Delete {cls.scope_label(scope)} Shots",
            f"Delete {cls.scope_count(shots, scope)}?\n\n{listed}\n\n"
            "Their records only: no keyframe is touched and no other shot moves.",
        )
