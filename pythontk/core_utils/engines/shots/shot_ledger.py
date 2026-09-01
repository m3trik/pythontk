# coding=utf-8
"""Ledger of the edits the shot system authors on a scene's animation.

The shot system writes on curves the animator owns.  Two of those writes
outlive the reason they were made:

* **Gap holds** - a stepped out-tangent on the last key before an inter-shot
  gap, so the gap plays as a hold instead of interpolating across it.
* **Boundary keys** - a sample on a shot bound, so a shot's content is its own
  and a neighbour's move cannot retime it.

Both are correct while the boundary that produced them is where it was.  Once
the boundary moves, the step reads as a hand-authored hold and the key as a
hand-placed pose - indistinguishable from the animator's own work, and left
behind on every adjust.

This ledger is what makes them distinguishable.  It records ONLY what the shot
system itself wrote (a step already on a key when the system arrived is the
animator's and is never claimed), together with what that key carried before,
so the write can be released exactly.  A boundary key additionally records the
bound it was made for, so it can FOLLOW that bound rather than be stranded by
it.

It is pure bookkeeping: matching, remapping and serialisation live here; the
DCC adapter supplies the scene reads and writes.

Times are matched by TOLERANCE, never by equality - a key sits where the last
move left it, which is the requested frame plus float noise.
"""

from typing import Any, Dict, List, Optional, Tuple

# Half-width of the window a recorded time is matched within.  Keys land on
# whole frames by default, so this only has to clear the float noise a relative
# move leaves behind (the same scale as the sequencer's ``_BATCH_MOVE_EPS``).
_LEDGER_EPS = 1.0e-3

#: Sentinel owner for a claim no shot bound is responsible for.
NO_OWNER = -1


class _ShotEditLedgerInternal(object):
    """Internal helpers for :class:`ShotEditLedger`.

    Both registers hold ``[time, *payload]`` records, so every time-indexed
    operation (match, shift, remap, sort) is written once here rather than
    twice with the payload shapes inlined.
    """

    @staticmethod
    def _index_of(recs: List[list], t: float, eps: float) -> Optional[int]:
        """Index of the record in *recs* whose time is nearest *t* within *eps*."""
        best = None
        best_d = eps
        for i, rec in enumerate(recs):
            d = abs(rec[0] - t)
            if d <= best_d:
                best, best_d = i, d
        return best

    @staticmethod
    def _sorted(recs: List[list]) -> List[list]:
        return sorted(recs, key=lambda r: r[0])

    def _registers(self):
        """Both registers, so maintenance walks them without naming each."""
        return (self._steps, self._keys)

    def _drop_if_empty(self, mapping: dict, curve: str) -> None:
        if curve in mapping and not mapping[curve]:
            del mapping[curve]


class ShotEditLedger(_ShotEditLedgerInternal):
    """What the shot system wrote on scene curves, and how to take it back.

    Two registers, both keyed by curve name and both holding time-first
    records:

    * ``steps`` - ``[time, in_type, out_type]``, the two types being what the
      key carried BEFORE the system stepped it.
    * ``keys`` - ``[time, owner_shot_id, edge]``, the owner naming the shot
      bound (``"start"`` / ``"end"``) the sample was created for.

    Every mutator is idempotent on an already-recorded entry, so the enforce
    pass can run as often as it likes without stacking duplicates.
    """

    def __init__(self, eps: float = _LEDGER_EPS):
        self.eps = float(eps)
        self._steps: Dict[str, List[list]] = {}
        self._keys: Dict[str, List[list]] = {}

    def __bool__(self) -> bool:
        return bool(self._steps or self._keys)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"<ShotEditLedger steps={self.step_count} "
            f"keys={self.key_count} curves={len(self.curves)}>"
        )

    # ---- counts / introspection ------------------------------------------

    @property
    def step_count(self) -> int:
        """Number of stepped tangents the system currently claims."""
        return sum(len(v) for v in self._steps.values())

    @property
    def key_count(self) -> int:
        """Number of samples the system currently claims."""
        return sum(len(v) for v in self._keys.values())

    @property
    def curves(self) -> set:
        """Every curve name either register mentions."""
        return set(self._steps) | set(self._keys)

    # ---- stepped tangents -------------------------------------------------

    def record_step(self, curve: str, time: float, in_type: str, out_type: str) -> bool:
        """Claim the step the system is about to write at ``(curve, time)``.

        Parameters:
            curve: Anim curve node name.
            time: Frame the step is written on.
            in_type: The key's in-tangent type BEFORE the step.
            out_type: The key's out-tangent type BEFORE the step.

        Returns:
            ``False`` when the pair is already claimed - the caller stepped it
            on an earlier pass and the ORIGINAL types recorded then must
            survive (re-recording would capture ``step`` as the original and
            make the release a no-op).
        """
        recs = self._steps.setdefault(curve, [])
        if self._index_of(recs, time, self.eps) is not None:
            return False
        recs.append([float(time), str(in_type), str(out_type)])
        self._steps[curve] = self._sorted(recs)
        return True

    def owns_step(self, curve: str, time: float) -> bool:
        """True when the system wrote the step at ``(curve, time)``."""
        recs = self._steps.get(curve)
        return bool(recs) and self._index_of(recs, time, self.eps) is not None

    def release_step(self, curve: str, time: float) -> Optional[Tuple[str, str]]:
        """Drop the claim at ``(curve, time)``, returning ``(in, out)`` types.

        Returns:
            The tangent types the key carried before the system stepped it, or
            ``None`` when nothing was claimed there - the caller must then
            leave the key alone, because an unclaimed step is the animator's.
        """
        recs = self._steps.get(curve)
        if not recs:
            return None
        i = self._index_of(recs, time, self.eps)
        if i is None:
            return None
        rec = recs.pop(i)
        self._drop_if_empty(self._steps, curve)
        return (rec[1], rec[2])

    def step_times(self, curve: str) -> List[float]:
        """Every time the system stepped on *curve*, ascending."""
        return [r[0] for r in self._steps.get(curve, ())]

    def stepped_curves(self) -> List[str]:
        """Curve names carrying at least one claimed step."""
        return sorted(self._steps)

    # ---- system-authored keys --------------------------------------------

    def record_key(
        self, curve: str, time: float, owner: int = NO_OWNER, edge: str = ""
    ) -> bool:
        """Claim a sample the system created for a shot bound.

        Parameters:
            curve: Anim curve node name.
            time: Frame the sample was created on.
            owner: ``shot_id`` of the shot whose bound the sample serves;
                :data:`NO_OWNER` when no bound is responsible for it.
            edge: ``"start"`` or ``"end"`` - which of the owner's bounds.

        Returns:
            ``False`` when the sample is already claimed.
        """
        recs = self._keys.setdefault(curve, [])
        if self._index_of(recs, time, self.eps) is not None:
            return False
        recs.append([float(time), int(owner), str(edge)])
        self._keys[curve] = self._sorted(recs)
        return True

    def release_key(self, curve: str, time: float) -> bool:
        """Drop the claim on a sample.  ``True`` when one was held."""
        recs = self._keys.get(curve)
        if not recs:
            return False
        i = self._index_of(recs, time, self.eps)
        if i is None:
            return False
        recs.pop(i)
        self._drop_if_empty(self._keys, curve)
        return True

    def key_times(self, curve: str) -> List[float]:
        """Every time the system created a sample on *curve*, ascending."""
        return [r[0] for r in self._keys.get(curve, ())]

    def key_records(self, curve: str) -> List[Tuple[float, int, str]]:
        """``(time, owner_shot_id, edge)`` for every claimed sample on *curve*."""
        return [(r[0], r[1], r[2]) for r in self._keys.get(curve, ())]

    def keyed_curves(self) -> List[str]:
        """Curve names carrying at least one claimed sample."""
        return sorted(self._keys)

    def disown_shot(self, shot_id: int) -> int:
        """Re-point every claim owned by *shot_id* at :data:`NO_OWNER`.

        Used when a shot is removed but its samples stay: they are still the
        system's to prune, but no bound is going to move them any more.

        Returns:
            The number of claims re-pointed.
        """
        n = 0
        for recs in self._keys.values():
            for rec in recs:
                if rec[1] == shot_id:
                    rec[1] = NO_OWNER
                    rec[2] = ""
                    n += 1
        return n

    # ---- remapping --------------------------------------------------------
    #
    # A claim is a (curve, time) pair, so anything that MOVES a key has to move
    # the claim with it.  Without this a rigid shot ripple would strand every
    # claim at the frame the key used to be on: the system would then neither
    # release its own step (it can no longer find it) nor recognise the moved
    # key as its own (nothing is recorded where it landed).

    def shift(self, curve: str, lo: float, hi: float, delta: float) -> int:
        """Add *delta* to every claim on *curve* inside ``[lo, hi]``.

        The window is inclusive on both ends and inflated by :attr:`eps`, to
        match the writers - a key exactly on a bound moved with the block.

        Returns:
            The number of claims remapped.
        """
        if abs(delta) < 1.0e-9:
            return 0
        lo_e, hi_e = lo - self.eps, hi + self.eps
        moved = 0
        for reg in self._registers():
            recs = reg.get(curve)
            if not recs:
                continue
            for rec in recs:
                if lo_e <= rec[0] <= hi_e:
                    rec[0] += delta
                    moved += 1
            reg[curve] = self._sorted(recs)
        return moved

    def remap(self, curve: str, pairs) -> int:
        """Move claims from each ``old_time`` to its ``new_time``.

        Parameters:
            curve: Anim curve node name.
            pairs: ``[(old_time, new_time), ...]``; times not claimed are
                ignored.

        Returns:
            The number of claims remapped.
        """
        moved = 0
        for old_t, new_t in pairs:
            if abs(new_t - old_t) < 1.0e-9:
                continue
            for reg in self._registers():
                recs = reg.get(curve)
                if not recs:
                    continue
                i = self._index_of(recs, old_t, self.eps)
                if i is not None:
                    recs[i][0] = float(new_t)
                    moved += 1
        for reg in self._registers():
            if curve in reg:
                reg[curve] = self._sorted(reg[curve])
        return moved

    # ---- disposal ---------------------------------------------------------

    def forget_curve(self, curve: str) -> None:
        """Drop every claim on *curve* (it was deleted, or is unreachable)."""
        self._steps.pop(curve, None)
        self._keys.pop(curve, None)

    # ---- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict form for the scene payload.  Empty registers are omitted."""
        out: Dict[str, Any] = {}
        if self._steps:
            out["steps"] = {
                crv: [list(r) for r in recs]
                for crv, recs in sorted(self._steps.items())
            }
        if self._keys:
            out["keys"] = {
                crv: [list(r) for r in recs] for crv, recs in sorted(self._keys.items())
            }
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ShotEditLedger":
        """Rebuild from :meth:`to_dict`.

        A missing or blank payload gives an empty ledger, so a scene written
        before the ledger existed loads without a migration step.  Records
        short of their payload are tolerated (a plain time list restores as
        unowned samples) rather than failing the whole scene load.
        """
        led = cls()
        for crv, recs in ((data or {}).get("steps") or {}).items():
            for rec in recs:
                try:
                    led._steps.setdefault(crv, []).append(
                        [float(rec[0]), str(rec[1]), str(rec[2])]
                    )
                except (IndexError, TypeError, ValueError):
                    continue
        for crv, recs in ((data or {}).get("keys") or {}).items():
            for rec in recs:
                try:
                    if isinstance(rec, (int, float)):
                        led._keys.setdefault(crv, []).append([float(rec), NO_OWNER, ""])
                        continue
                    owner = int(rec[1]) if len(rec) > 1 else NO_OWNER
                    edge = str(rec[2]) if len(rec) > 2 else ""
                    led._keys.setdefault(crv, []).append([float(rec[0]), owner, edge])
                except (IndexError, TypeError, ValueError):
                    continue
        for reg in led._registers():
            for crv in reg:
                reg[crv] = led._sorted(reg[crv])
        return led
