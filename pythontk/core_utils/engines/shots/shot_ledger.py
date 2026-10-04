# coding=utf-8
"""Ledger of the edits the shot system authors on a scene's animation.

The shot system writes on curves the animator owns.  Two of those writes
outlive the reason they were made:

* **Gap holds** - a stepped out-tangent on the last key before an inter-shot
  gap, so the gap plays as a hold instead of interpolating across it.
* **Boundary keys** - a sample on a shot bound, so a shot's content is its own
  and a neighbour's move cannot retime it.

* **Authored keys** - the samples a behavior (a manifest fade, a highlight)
  keys on its own channels, so re-applying it replaces exactly those and a
  key on the same channel the ledger does not hold reads as the animator's.

The first two are correct while the boundary that produced them is where it was.  Once
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

from typing import Any, Dict, Iterable, List, Optional, Tuple

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
        """Every register, so maintenance walks them without naming each."""
        return (self._steps, self._keys, self._authored)

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
    * ``authored`` - ``[time, owner_shot_id, behavior, obj, stamp]``, a key
      a behavior wrote on one of its channels for that shot's object; *stamp*
      is the effect recipe it was keyed under (``EffectRecipe.fingerprint``,
      ``""`` for a template that keys no recipe), so Assess can tell keys an
      older recipe made.

    Every mutator is idempotent on an already-recorded entry, so the enforce
    pass can run as often as it likes without stacking duplicates.
    """

    def __init__(self, eps: float = _LEDGER_EPS):
        self.eps = float(eps)
        self._steps: Dict[str, List[list]] = {}
        self._keys: Dict[str, List[list]] = {}
        self._authored: Dict[str, List[list]] = {}

    def __bool__(self) -> bool:
        return bool(self._steps or self._keys or self._authored)

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
        """Every curve name any register mentions."""
        return set(self._steps) | set(self._keys) | set(self._authored)

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

    def owns_key(self, curve: str, time: float) -> bool:
        """True when the system created the sample at ``(curve, time)``."""
        recs = self._keys.get(curve)
        return bool(recs) and self._index_of(recs, time, self.eps) is not None

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

    def release(self, curve: str, lo: float, hi: Optional[float] = None) -> int:
        """Drop every claim at *lo*, or within ``[lo, hi]`` -- step and sample alike.

        What a system CUT calls: the key is gone, and a move remaps only the
        claims of keys it finds, so a claim left on the frame would be
        inherited by whatever lands there next.

        Parameters:
            curve: Anim curve node name (or a DCC's stand-in key).
            lo: The cut key's time (matched within :attr:`eps`), or a
                window's start.
            hi: A window's end: every claim in ``[lo, hi]``, ends included
                and matched EXACTLY -- unlike :meth:`shift`, never inflated,
                because the window is the cut's own: a caller that cut an
                open end (a neighbour's shared sample) deflated it already.

        Returns:
            The number of claims dropped.
        """
        if hi is None:
            stepped = self.release_step(curve, lo) is not None
            return (
                int(stepped)
                + int(self.release_key(curve, lo))
                + int(self.release_authored(curve, lo))
            )
        steps = [t for t in self.step_times(curve) if lo <= t <= hi]
        keys = [t for t in self.key_times(curve) if lo <= t <= hi]
        authored = [r[0] for r in self._authored.get(curve, ()) if lo <= r[0] <= hi]
        return (
            sum(self.release_step(curve, t) is not None for t in steps)
            + sum(self.release_key(curve, t) for t in keys)
            + sum(self.release_authored(curve, t) for t in authored)
        )

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
        for recs in self._authored.values():
            for rec in recs:
                if rec[1] == shot_id:
                    rec[1] = NO_OWNER
                    n += 1
        return n

    def disown_absent(self, shot_ids: Iterable[int]) -> int:
        """Re-point every claim whose owner is not one of *shot_ids* at
        :data:`NO_OWNER`.

        For data saved before a removed shot's claims were disowned: such an
        owner is no loaded shot, and the next shot given its id would inherit
        its claims (its Build then deletes them as dropped behaviors).

        Returns:
            The number of claims re-pointed.
        """
        live = set(shot_ids)
        absent = {
            rec[1]
            for register in (self._keys, self._authored)
            for recs in register.values()
            for rec in recs
            if rec[1] != NO_OWNER and rec[1] not in live
        }
        return sum(self.disown_shot(owner) for owner in absent)

    # ---- authored keys (behaviors) ---------------------------------------

    def record_authored(
        self,
        curve: str,
        time: float,
        owner: int,
        behavior: str,
        obj: str,
        stamp: str = "",
    ) -> bool:
        """Claim a key a behavior wrote on *curve* for shot *owner*'s *obj*.

        A claim no shot owns (:meth:`disown_shot`: its shot was removed, its
        key stayed) is taken over: the key now on that frame is this
        behavior's, and refusing the claim left it owned by nobody -- the
        shot that keyed it could never take it out again.

        Parameters:
            stamp: The effect recipe the key was made under
                (``EffectRecipe.fingerprint``); ``""`` when none.

        Returns:
            ``False`` when a shot already claims the key (by any behavior).
        """
        rec = [float(time), int(owner), str(behavior), str(obj), str(stamp)]
        recs = self._authored.setdefault(curve, [])
        i = self._index_of(recs, time, self.eps)
        if i is not None:
            if recs[i][1] != NO_OWNER:
                return False
            recs[i] = rec
        else:
            recs.append(rec)
        self._authored[curve] = self._sorted(recs)
        return True

    def owns_authored(self, curve: str, time: float) -> bool:
        """True when a behavior wrote the key at ``(curve, time)``."""
        recs = self._authored.get(curve)
        return bool(recs) and self._index_of(recs, time, self.eps) is not None

    def owns_any(self, curve: str, time: float) -> bool:
        """True when the system wrote the KEY at ``(curve, time)`` -- a bound
        sample or a behavior key.  A stepped tangent claims only the key's
        tangent, so it does not count: the key is still the animator's."""
        return self.owns_key(curve, time) or self.owns_authored(curve, time)

    def release_authored(self, curve: str, time: float) -> bool:
        """Drop the behavior claim at ``(curve, time)``.  ``True`` when held."""
        recs = self._authored.get(curve)
        if not recs:
            return False
        i = self._index_of(recs, time, self.eps)
        if i is None:
            return False
        recs.pop(i)
        self._drop_if_empty(self._authored, curve)
        return True

    def authored(
        self,
        owner: Optional[int] = None,
        obj: Optional[str] = None,
        behavior: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """``(curve, time)`` of every behavior key matching the given filters,
        ordered by curve then time."""
        return [
            (crv, rec[0])
            for crv in sorted(self._authored)
            for rec in self._authored[crv]
            if (owner is None or rec[1] == owner)
            and (obj is None or rec[3] == obj)
            and (behavior is None or rec[2] == behavior)
        ]

    def authored_stamps(self, owner: int, obj: str, behavior: str) -> set:
        """The recipe stamps of the keys *behavior* wrote on shot *owner*'s
        *obj* -- one when they were all keyed in one pass, ``""`` for a key
        claimed before stamps existed; empty when it claims none."""
        return {
            rec[4] if len(rec) > 4 else ""
            for recs in self._authored.values()
            for rec in recs
            if rec[1] == owner and rec[3] == obj and rec[2] == behavior
        }

    def authored_pairs(self, owner: int) -> set:
        """``{(obj, behavior)}`` every behavior key shot *owner* holds names --
        what a build compares against the doc to find what it dropped."""
        return {
            (rec[3], rec[2])
            for recs in self._authored.values()
            for rec in recs
            if rec[1] == owner
        }

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

        Every pair is matched against the claims as they stood BEFORE the
        call, and all the moves land together.  Applied one after another, a
        claim moved onto the source frame of a LATER pair was moved a second
        time whenever that later key was not claimed itself -- and movers
        hand this the pairs of every key they moved, claimed or not, so any
        ripple whose delta equals a key spacing did it: a +10 respace over a
        start pin at 60 and an animator key at 70 left the pin at 70 and its
        claim at 80, on the animator's key.

        Parameters:
            curve: Anim curve node name.
            pairs: ``[(old_time, new_time), ...]``; times not claimed are
                ignored.

        Returns:
            The number of claims remapped.
        """
        todo = [(float(o), float(n)) for o, n in pairs if abs(n - o) >= 1.0e-9]
        moved = 0
        for reg in self._registers():
            recs = reg.get(curve)
            if not recs:
                continue
            landing: Dict[int, float] = {}
            for old_t, new_t in todo:
                i = self._index_of(recs, old_t, self.eps)
                if i is not None and i not in landing:
                    landing[i] = new_t
            for i, new_t in landing.items():
                recs[i][0] = new_t
            moved += len(landing)
            reg[curve] = self._sorted(recs)
        return moved

    def retime(self, ratio: float, offset: float = 0.0) -> int:
        """Put every claim on another clock: its time scaled by *ratio*, then
        shifted by *offset*.

        What a frame-rate change does to the keys the claims name -- a live
        store's (:meth:`~pythontk.ShotStore.rescale_to_fps`) and a hand-off's
        landing on the receiving scene's clock (``ShotTransfer``) alike.  A
        time is rounded to 1e-4, well inside :attr:`eps`: the claim still
        finds its key, and a saved record carries no float noise.

        Returns:
            The number of claims retimed.
        """
        if abs(ratio - 1.0) < 1.0e-12 and abs(offset) < 1.0e-12:
            return 0
        moved = 0
        for reg in self._registers():
            for curve in list(reg):
                recs = reg[curve]
                for rec in recs:
                    rec[0] = round(rec[0] * ratio + offset, 4)
                moved += len(recs)
                reg[curve] = self._sorted(recs)
        return moved

    # ---- disposal ---------------------------------------------------------

    def forget_curve(self, curve: str) -> None:
        """Drop every claim on *curve* (it was deleted, or is unreachable)."""
        for reg in self._registers():
            reg.pop(curve, None)

    def rename_curve(self, old: str, new: str) -> bool:
        """Move every claim on curve *old* to *new* (the curve was renamed).

        For a DCC whose curve key derives from its owner's name -- blendertk's
        ``"<object>|<data_path>|<index>"`` -- and so goes stale when the owner
        is renamed while the claims it carries still stand.  Claims already on
        *new* are kept; one at the same time (within :attr:`eps`) as a moved
        claim is not duplicated.

        Returns:
            ``True`` when *old* held any claim.
        """
        if old == new:
            return False
        moved = False
        for reg in self._registers():
            recs = reg.pop(old, None)
            if not recs:
                continue
            kept = list(reg.get(new, []))
            for rec in recs:
                if not any(abs(r[0] - rec[0]) <= self.eps for r in kept):
                    kept.append(rec)
            reg[new] = self._sorted(kept)
            moved = True
        return moved

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
        if self._authored:
            out["authored"] = {
                crv: [list(r) for r in recs]
                for crv, recs in sorted(self._authored.items())
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
        for crv, recs in ((data or {}).get("authored") or {}).items():
            for rec in recs:
                try:
                    led._authored.setdefault(crv, []).append(
                        [
                            float(rec[0]),
                            int(rec[1]),
                            str(rec[2]),
                            str(rec[3]),
                            str(rec[4]) if len(rec) > 4 else "",
                        ]
                    )
                except (IndexError, TypeError, ValueError):
                    continue
        for reg in led._registers():
            for crv in reg:
                reg[crv] = led._sorted(reg[crv])
        return led
