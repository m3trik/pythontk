# !/usr/bin/python
# coding=utf-8
"""What a lightmap re-bake must redo: the bake baseline and its stale rule.

A lightmap is a picture of the room at the moment it was rendered.  It goes
stale when its own object changes (moved, deformed, remodelled, its lightmap
UVs re-laid, its material regrouped, shown or hidden), when its map goes
missing, when the lighting it was rendered under changes, and when anything
static near enough to shade it arrives, leaves or changes.  The rule lives
here, over plain values, so both DCC bakers run the same one:

- :meth:`BakeBaseline.signature` fingerprints a mesh: an exact topology hash,
  its point count, and the points' first and second moments.  Moments, not a
  point hash: a ``.ma`` round trip keeps ~7 significant digits of float32 data
  and would flip any quantised hash, while moments compared with tolerances
  relative to the object's size (:meth:`BakeBaseline.diff`) ride it out.
- :meth:`BakeBaseline.neighbours` is the reach rule -- what a change shades.
- :meth:`BakeBaseline.assess` reads the scene against the record and returns
  a :class:`StaleReport` (the dry run); :meth:`BakeBaseline.advance` rolls the
  record forward when a bake commits, and :meth:`BakeBaseline.keep_baked` gives
  what a commit only re-packed back how it was baked.
- :meth:`BakeBaseline.lighting_state` / :meth:`~BakeBaseline.lighting_digest`
  / :meth:`~BakeBaseline.light_changes` fingerprint the lighting a bake saw,
  with the rule that chose its lights; :meth:`~BakeBaseline.lighting_rules`
  hands a re-bake that rule back and :meth:`~BakeBaseline.lights_used` names
  the lights that lit the maps -- so a re-bake relights, a dry run reads and
  an export ships the lights the way the bake used them.
- :meth:`BakeBaseline.cell_fits` says whether a re-baked object still fits the
  atlas cell it was written to.

Staleness is counted in commits, not time: the record's ``serial`` steps at
every bake, and each object records the serial its state was taken at
(``placed_at``) and its map last baked at (``baked_at``).  A neighbour that
changed at a later commit than B's bake is one B's map never saw.  What a
commit would otherwise forget -- an object's old state, a neighbour's old
place -- is kept as the object's ``stale`` reasons until it re-bakes.

The record is ``SceneRecords.LIGHTMAP_BAKE_STATE`` (private, the scene's own).
Each DCC supplies the scene's state -- node ids, matrices, points, UVs, the
Exclude / Isolate roles -- and the files on disk; nothing here touches either.
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

if TYPE_CHECKING:  # the return of StaleReport.doc, imported where it is built
    from pythontk.str_utils.report_doc import ReportDoc

# Eager on purpose: ``BakeBaseline.FORMAT`` is a class attribute, read at
# definition time.  ``scene_records`` is stdlib-only, so this costs nothing
# and cannot cycle.
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords

#: An axis-aligned world box: ``[[min x, y, z], [max x, y, z]]``.
_Box = Sequence[Sequence[float]]


class BakeBaseline:
    """The lightmap bake baseline: pure rules over plain values.

    Every method is a function of its arguments -- no scene, no file, no
    clock -- so a DCC can preview a re-bake, test one or replay one without
    rendering.

    A scene, as :meth:`assess` and :meth:`advance` take it, is
    ``{id: {"state": state, "role": role, "marker": marker}}`` over every
    static participant of a bake: the objects carrying a lightmap marker, the
    Exclude set's meshes and the Isolate set's, keyed by the DCC's id (Maya's
    long DAG path per instance, Blender's object name).

    - *state*: ``{"matrix": [16], "bbox": [[3], [3]], "geometry": ...,
      "layout": ... | None, "materials": str, "group": str, "rendered": bool}``
      -- the world matrix with its translation at ``[12:15]`` (Maya's
      ``xform -m`` order; Blender's ``matrix_world`` column by column), the
      world bounding box, :meth:`signature` of the object-space points and
      of the lightmap UV set, a hash of the material assignment, the primary
      material (the atlas group), and whether the renderer sees it.
    - *role*: a :class:`Role` value.
    - *marker*: the object's lightmap marker as read (``{"map": name,
      "scaleOffset": [4]}``), or ``None``.

    Tolerances are class attributes, each a fraction of the object's own size
    unless named otherwise; the ``tol`` argument multiplies all of them.
    """

    class Reason(str, Enum):
        """Why an object's map is stale.  Stored in the record: the values
        are saved data."""

        MOVED = "moved"
        DEFORMED = "deformed"
        REMODELLED = "remodelled"
        RELAID = "relaid"
        MATERIAL = "material"
        VISIBILITY = "visibility"
        MISSING_MAP = "missing_map"
        LIGHTING = "lighting"
        NEIGHBOUR = "neighbour"
        DUPLICATE = "duplicate"

    class Role(str, Enum):
        """What a participant is to the bake.  Stored in the record."""

        #: Carries a lightmap; renders in every pass's room.
        BAKED = "baked"
        #: The Exclude set: renders (casts shadow and bounce), never baked.
        EXCLUDED = "excluded"
        #: The Isolate set: baked in a pass of its own, out of every other
        #: pass's render -- so its changes reach no neighbour.
        ISOLATED = "isolated"

    #: How :class:`StaleReport` words each reason.
    LABELS: Dict[str, str] = {
        "moved": "moved",
        "deformed": "deformed",
        "remodelled": "remodelled",
        "relaid": "lightmap UVs re-laid",
        "material": "material changed",
        "visibility": "shown or hidden",
        "missing_map": "map missing",
        "lighting": "lighting changed",
        "neighbour": "a neighbour changed",
        "duplicate": "a duplicate reading its source's cell",
    }

    #: The changes that alter what an object casts on its neighbours -- a
    #: re-laid lightmap UV set does not.
    CAUSES = frozenset(
        {
            Reason.MOVED,
            Reason.DEFORMED,
            Reason.REMODELLED,
            Reason.MATERIAL,
            Reason.VISIBILITY,
        }
    )

    #: Record schema version, stamped into every :meth:`encode` -- the record
    #: declaration's, so the two cannot disagree.  A reader that does not
    #: recognise it treats the record as absent.
    FORMAT = SceneRecords.LIGHTMAP_BAKE_STATE.version

    #: Centroid drift tolerated, as a fraction of the object's size.
    CENTROID_TOL = 1e-5
    #: Covariance drift tolerated, as a fraction of the size squared.
    SPREAD_TOL = 1e-4
    #: Bounding-box corner drift tolerated, as a fraction of its diagonal.
    BOX_TOL = 1e-5
    #: Matrix entry drift tolerated, relative to the entry (at least 1).
    MATRIX_TOL = 1e-5
    #: Light value drift tolerated, relative to the value (at least 1).
    LIGHT_TOL = 1e-5
    #: The float noise a save round trip leaves, relative to a coordinate's
    #: magnitude: 7 significant digits out, float32 back in (~5.6e-7), with
    #: room to spare.  Added to every geometric tolerance, so an object whose
    #: frozen points sit far from the origin is not stale after a reopen.
    FLOAT_NOISE = 1e-6

    #: Reach of a change, as a multiple of the changed object's diagonal ...
    STALE_REACH = 1.0
    #: ... never under this fraction of the room's diagonal ...
    STALE_REACH_FLOOR = 0.01
    #: ... nor over this one.
    STALE_REACH_CAP = 0.25

    #: How far a re-baked object's atlas extent may drift, per axis, and
    #: still be written back into its recorded cell.
    CELL_TOLERANCE = 0.10

    #: The lighting state's keys for the environment, the bake settings and
    #: the rule that chose the lights, beside the light ids (``@`` sorts them
    #: first and no node name has one).
    ENVIRONMENT = "@environment"
    SETTINGS = "@settings"
    RULE = "@rule"

    _EPS = 1e-12

    # ------------------------------------------------------------------ state
    @classmethod
    def signature(
        cls, points: Any, counts: Sequence[int], indices: Sequence[int], dims: int = 3
    ) -> Dict[str, Any]:
        """A mesh's fingerprint: ``{"points", "topology", "moments"}``.

        Parameters:
            points: ``(n, d)`` coordinates -- object-space vertex positions
                (d = 3) or a UV set (d = 2). A flat sequence is read as
                *dims*-tuples.
            counts: Vertices (or UVs) per face.
            indices: The face-vertex (or face-UV) indices, face by face.
            dims: The tuple size of a flat *points*.

        Returns:
            ``points``: the count; ``topology``: blake2b-16 of *counts* and
            *indices*, exact; ``moments``: the centroid, then the variances,
            then the covariances (``[cx, cy, cz, sxx, syy, szz, sxy, sxz,
            syz]``; ``[cu, cv, suu, svv, suv]`` for UVs).
        """
        import numpy as np

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2:
            pts = pts.reshape(-1, dims)
        n, d = pts.shape
        if n:
            centre = pts.mean(axis=0)
            spread = pts - centre
            cov = spread.T @ spread / n
        else:
            centre, cov = np.zeros(d), np.zeros((d, d))
        upper = [cov[i, j] for i in range(d) for j in range(i + 1, d)]
        moments = list(centre) + list(np.diag(cov)) + upper
        return {
            "points": int(n),
            "topology": cls._topology(counts, indices),
            "moments": [float(v) for v in moments],
        }

    @staticmethod
    def _topology(counts: Sequence[int], indices: Sequence[int]) -> str:
        """blake2b-16 over *counts* and *indices* as little-endian int64."""
        import numpy as np

        counts = np.asarray(counts, dtype="<i8").ravel()
        indices = np.asarray(indices, dtype="<i8").ravel()
        digest = hashlib.blake2b(digest_size=16)
        digest.update(len(counts).to_bytes(8, "little"))
        digest.update(counts.tobytes())
        digest.update(indices.tobytes())
        return digest.hexdigest()

    @classmethod
    def diff(
        cls, a: Mapping, b: Mapping, tol: float = 1.0
    ) -> List["BakeBaseline.Reason"]:
        """Why state *b* differs from state *a*: :class:`Reason` values, in
        declaration order; empty when they are the same.

        Geometry is compared by moments, so a change that moves the mean and
        spread of the points by less than :attr:`CENTROID_TOL` /
        :attr:`SPREAD_TOL` of the object's size is not seen -- one vertex of a
        dense mesh nudged; the bounding box catches it when it moves the
        bounds.

        Parameters:
            a: The recorded state.
            b: The current state.
            tol: A multiplier on every tolerance.
        """
        R = cls.Reason
        reasons = []
        ga, gb = a.get("geometry"), b.get("geometry")
        if cls._remodelled(ga, gb):
            reasons.append(R.REMODELLED)
        elif ga is not None and not cls._moments_close(
            ga.get("moments"), gb.get("moments"), tol
        ):
            reasons.append(R.DEFORMED)
        if not cls._matrix_close(a.get("matrix"), b.get("matrix"), tol):
            reasons.append(R.MOVED)
        elif not reasons and not cls._box_close(
            a.get("bbox"), b.get("bbox"), tol, a.get("matrix")
        ):
            # The bounds moved under a still transform: a deformation the
            # moments were too coarse to see.
            reasons.append(R.DEFORMED)
        la, lb = a.get("layout"), b.get("layout")
        if cls._remodelled(la, lb) or (
            la is not None
            and not cls._moments_close(la.get("moments"), lb.get("moments"), tol)
        ):
            reasons.append(R.RELAID)
        if a.get("materials") != b.get("materials") or a.get("group") != b.get("group"):
            reasons.append(R.MATERIAL)
        if bool(a.get("rendered", True)) != bool(b.get("rendered", True)):
            reasons.append(R.VISIBILITY)
        return reasons

    @classmethod
    def same(cls, a: Mapping, b: Mapping, tol: float = 1.0) -> bool:
        """Whether states *a* and *b* are the same within tolerance
        (:meth:`diff` finds nothing)."""
        return not cls.diff(a, b, tol)

    @staticmethod
    def _remodelled(a: Optional[Mapping], b: Optional[Mapping]) -> bool:
        """One signature present and not the other, or a different point
        count or topology -- compared exactly."""
        if a is None or b is None:
            return (a is None) != (b is None)
        return a.get("points") != b.get("points") or a.get("topology") != b.get(
            "topology"
        )

    @classmethod
    def _moments_close(cls, ma, mb, tol: float) -> bool:
        """Moments within tolerance of the object's size.

        The size is twice the RMS radius (the variances' sum) -- a box's
        corners give its diagonal -- so the test needs nothing the signature
        does not carry. The noise term bounds what float rounding of
        coordinates of magnitude ``|centroid| + radius`` can do: the centroid
        moves by at most that times :attr:`FLOAT_NOISE`, a covariance by at
        most twice that times the radius.
        """
        if ma is None or mb is None:
            return ma is None and mb is None
        if len(ma) != len(mb):
            return False
        d = int(round((math.sqrt(9 + 8 * len(ma)) - 3) / 2))
        if d + d * (d + 1) // 2 != len(ma):
            return list(ma) == list(mb)
        radius = math.sqrt(max(sum(ma[d : 2 * d]), sum(mb[d : 2 * d]), 0.0))
        size = 2.0 * radius
        far = max(math.hypot(*ma[:d]), math.hypot(*mb[:d])) + radius
        centre_tol = cls.CENTROID_TOL * tol * size + cls.FLOAT_NOISE * far + cls._EPS
        spread_tol = (
            cls.SPREAD_TOL * tol * size * size
            + 2.0 * cls.FLOAT_NOISE * far * size
            + cls._EPS
        )
        return all(abs(x - y) <= centre_tol for x, y in zip(ma[:d], mb[:d])) and all(
            abs(x - y) <= spread_tol for x, y in zip(ma[d:], mb[d:])
        )

    @classmethod
    def _matrix_close(cls, ma, mb, tol: float) -> bool:
        """Every entry within :attr:`MATRIX_TOL` of its own magnitude."""
        if ma is None or mb is None:
            return ma is None and mb is None
        if len(ma) != len(mb):
            return False
        return all(
            abs(x - y) <= cls.MATRIX_TOL * tol * max(1.0, abs(x), abs(y))
            for x, y in zip(ma, mb)
        )

    @classmethod
    def _box_close(cls, ba, bb, tol: float, matrix=None) -> bool:
        """Corners within :attr:`BOX_TOL` of the diagonal, plus the noise of
        coordinates as far out as the box or the transform's translation."""
        if ba is None or bb is None:
            return ba is None and bb is None
        size = max(cls._diagonal(ba), cls._diagonal(bb))
        far = [abs(v) for corner in (*ba, *bb) for v in corner]
        if matrix is not None:
            far += [abs(v) for v in list(matrix)[12:15]]
        limit = cls.BOX_TOL * tol * size + cls.FLOAT_NOISE * max(far) + cls._EPS
        return all(
            abs(x - y) <= limit for ca, cb in zip(ba, bb) for x, y in zip(ca, cb)
        )

    @staticmethod
    def _diagonal(box: Optional[_Box]) -> float:
        if box is None or len(box) < 2:
            return 0.0
        return math.hypot(*(hi - lo for lo, hi in zip(box[0], box[1])))

    @staticmethod
    def _union(boxes: Iterable[Optional[_Box]]) -> Optional[List[List[float]]]:
        """The box around every box in *boxes* (``None`` ones skipped)."""
        boxes = [b for b in boxes if b is not None and len(b) == 2]
        if not boxes:
            return None
        return [
            [min(b[0][i] for b in boxes) for i in range(3)],
            [max(b[1][i] for b in boxes) for i in range(3)],
        ]

    # ------------------------------------------------------------------ reach
    @classmethod
    def neighbours(
        cls,
        causes: Mapping[str, Sequence[_Box]],
        boxes: Mapping[str, _Box],
        room: Optional[_Box] = None,
        reach: float = 1.0,
    ) -> Dict[str, List[str]]:
        """Which of *boxes* a change in each of *causes* reaches.

        The reach of a cause A is :attr:`STALE_REACH` times A's diagonal,
        clamped between :attr:`STALE_REACH_FLOOR` and :attr:`STALE_REACH_CAP`
        of the room's, times *reach*; a box within that distance of any of
        A's boxes (its old place or its new one) is its neighbour. One ring:
        a neighbour's re-bake reaches no further.

        Parameters:
            causes: Cause id -> its boxes (old and/or new).
            boxes: Candidate id -> its current box.
            room: The box around every static participant; ``None`` takes
                the box around *causes* and *boxes*.
            reach: A multiplier on the reach; 0 names no neighbour.

        Returns:
            Candidate id -> the sorted causes reaching it (a cause never
            names itself); only candidates something reaches.
        """
        causes = {k: [b for b in v if b is not None] for k, v in causes.items()}
        causes = {k: v for k, v in causes.items() if v}
        boxes = {k: b for k, b in boxes.items() if b is not None}
        if reach <= 0 or not causes or not boxes:
            return {}
        import numpy as np

        if room is None:
            room = cls._union(
                [*boxes.values(), *(b for v in causes.values() for b in v)]
            )
        room_size = cls._diagonal(room)
        ids = list(boxes)
        arr = np.asarray([boxes[k] for k in ids], dtype=np.float64)
        lo, hi = arr[:, 0], arr[:, 1]
        found: Dict[str, List[str]] = {}
        for cause in sorted(causes):
            own = causes[cause]
            distance = cls._reach(own, room_size, reach) + cls._EPS
            hit = np.zeros(len(ids), dtype=bool)
            for box in own:
                b = np.asarray(box, dtype=np.float64)
                gap = np.maximum(0.0, np.maximum(b[0] - hi, lo - b[1]))
                hit |= np.sqrt((gap * gap).sum(axis=1)) <= distance
            for i in np.flatnonzero(hit):
                if ids[i] != cause:
                    found.setdefault(ids[i], []).append(cause)
        return {k: found[k] for k in sorted(found)}

    @classmethod
    def _reach(cls, own: Sequence[_Box], room_size: float, reach: float) -> float:
        """The D4 rule for one cause's boxes."""
        size = max(cls._diagonal(b) for b in own)
        lo = cls.STALE_REACH_FLOOR * room_size
        hi = cls.STALE_REACH_CAP * room_size
        return reach * min(max(cls.STALE_REACH * size, lo), hi)

    # ----------------------------------------------------------------- assess
    @classmethod
    def assess(
        cls,
        record: Any,
        current: Mapping[str, Mapping],
        lighting: Union[
            Mapping[str, Any], Callable[[Optional[Dict[str, Any]]], Mapping], None
        ] = None,
        missing: Iterable[str] = (),
        reach: float = 1.0,
        objects: Optional[Iterable[str]] = None,
        tol: float = 1.0,
    ) -> "StaleReport":
        """What is stale in *current* against *record*, and why.

        An object with a lightmap is stale when its state differs from the
        one it was baked against (:meth:`diff`), when a reason was kept for
        it at an earlier commit, when its map is *missing*, when the
        *lighting* differs from the one it was baked under, or when it is a
        duplicate of a tracked object (it reads its source's cell). It is a
        neighbour when a participant within reach of it (:meth:`neighbours`)
        changed now, arrived, left, or changed at a commit later than its
        bake. An isolated participant's changes reach no neighbour.

        With no record there is nothing to diff: every marked object is
        untracked, and only a missing map is stale -- as an untracked object's
        is under any record.

        Parameters:
            record: The stored record (text, payload or :meth:`decode` form).
            current: The scene now (see the class docstring).
            lighting: The lighting now. A callable is handed each recorded
                lighting's rule (:attr:`RULE`; ``None`` for one recorded
                without) and returns the lighting now under it
                (:meth:`lighting_state`; its rule defaults to the one it was
                handed) -- read once per rule, so each map is judged by the
                lights its own bake chose, and a lighting recorded without a
                rule reads as changed. A mapping is that lighting for every
                map alike. ``None`` skips the lighting test.
            missing: Ids whose map file cannot be found.
            reach: A multiplier on the neighbour reach; 0 names none.
            objects: Limit the report to these ids; ``None`` reports all.
                Causes still come from the whole scene.
            tol: A multiplier on every tolerance.

        Returns:
            :class:`StaleReport` (``in_place`` and ``repack`` empty: the DCC's
            packing plan fills them).
        """
        survey = cls._survey(record, current, lighting, set(missing), reach, tol)
        scope = None if objects is None else set(objects)

        def kept(k):
            return scope is None or k in scope

        stale = {
            k: tuple(reasons)
            for k, reasons in sorted(survey["own"].items())
            if reasons and survey["marked"].get(k) and kept(k)
        }
        named = {k: tuple(v) for k, v in survey["neighbours"].items() if kept(k)}
        unbakeable = sorted(
            k for k in set(stale) | set(named) if not survey["bakes"].get(k)
        )
        targets = sorted((set(stale) | set(named)) - set(unbakeable))
        return StaleReport(
            targets=tuple(targets),
            stale=stale,
            neighbours=named,
            untracked=tuple(k for k in survey["untracked"] if kept(k)),
            unbakeable=tuple(unbakeable),
            lighting=tuple(sorted(survey["lighting"])),
            baked=len(survey["candidates"]),
        )

    @classmethod
    def _survey(cls, record, current, lighting, missing, reach, tol) -> Dict[str, Any]:
        """Everything :meth:`assess` reports and :meth:`advance` keeps, from
        one pass over the record and the scene."""
        R = cls.Reason
        rec = cls.decode(record)
        objects = rec["objects"]
        current = {str(k): cls._participant(p) for k, p in (current or {}).items()}
        renames, duplicates, untracked = cls._rematch(objects, current, tol)
        for old, new in renames.items():
            objects[new] = objects.pop(old)
        live = {k: e for k, e in objects.items() if e["removed_at"] is None}

        own: Dict[str, List] = {}
        changed: Dict[str, List] = {}
        now: Dict[str, List] = {}  # cause -> boxes, reaching every candidate
        then: Dict[str, Tuple[List, int]] = {}  # cause -> (boxes, serial)
        for k, p in current.items():
            e, st = live.get(k), p["state"]
            in_room = cls._in_room(p["role"], st)
            if e is None:  # arrived since the last commit
                own[k] = [R.DUPLICATE] if k in duplicates else []
                if in_room:
                    now[k] = [st.get("bbox")]
                continue
            reasons = cls.diff(e["state"], st, tol)
            changed[k] = reasons
            was = cls._in_room(e["role"], e["state"])
            if (was or in_room) and (
                was != in_room or cls.CAUSES.intersection(reasons)
            ):
                now[k] = [e["state"].get("bbox"), st.get("bbox")]
            elif in_room:
                then[k] = ([st.get("bbox")], e["placed_at"])
            own[k] = list(e["stale"] or []) + reasons
        # Tracked or not: an untracked map is one only the artist can vouch
        # for, and one on disk nowhere matches nothing.
        for k in missing.intersection(current):
            own[k].append(R.MISSING_MAP)
        for k, e in objects.items():
            if k in current or not cls._in_room(e["role"], e["state"]):
                continue
            if e["removed_at"] is None:  # gone since the last commit
                now[k] = [e["state"].get("bbox")]
            else:
                then[k] = ([e["state"].get("bbox")], e["removed_at"])

        marked = {k: cls._marker_map(p) is not None for k, p in current.items()}
        candidates = {
            k: current[k]["state"].get("bbox")
            for k in current
            if marked[k]
            and current[k]["role"] != cls.Role.EXCLUDED
            and (k in live or k in duplicates)
        }

        def since(k):
            e = live.get(k)
            if e is None:
                return rec["serial"]
            return e["baked_at"] if e["baked_at"] is not None else e["placed_at"]

        floor = min((since(k) for k in candidates), default=rec["serial"])
        then = {k: v for k, v in then.items() if v[1] > floor and k not in now}
        room = cls._union(
            [p["state"].get("bbox") for p in current.values()]
            + [b for v in now.values() for b in v]
            + [b for v, _ in then.values() for b in v]
        )
        pairs = cls.neighbours(
            {**{k: v for k, (v, _) in then.items()}, **now},
            candidates,
            room=room,
            reach=reach,
        )
        named = {}
        for b, names in pairs.items():
            names = [a for a in names if a in now or then[a][1] > since(b)]
            if names:
                named[b] = names

        lit = set()
        if lighting is not None:
            read = cls._lighting_reader(lighting)
            verdicts: Dict[str, List[str]] = {}
            for k in candidates:
                d = live[k]["lighting"] if k in live else None
                if d is None:
                    continue
                if d not in verdicts:
                    then_state = rec["lightings"].get(d)
                    verdicts[d] = []
                    if then_state is not None:
                        state_now, digest_now = read(then_state.get(cls.RULE))
                        if digest_now != d:
                            verdicts[d] = cls.light_changes(then_state, state_now, tol)
                if verdicts[d]:
                    own[k].append(R.LIGHTING)
                    lit.update(verdicts[d])

        bakes = {k: cls.can_bake(p) for k, p in current.items()}
        return {
            "record": rec,
            "objects": objects,
            "live": live,
            "current": current,
            "own": {k: cls._ordered(v) for k, v in own.items()},
            "changed": changed,
            "now": now,
            "neighbours": named,
            "duplicates": duplicates,
            "untracked": untracked,
            "marked": marked,
            "candidates": candidates,
            "bakes": bakes,
            "lighting": lit,
        }

    # ---------------------------------------------------------------- advance
    @classmethod
    def advance(
        cls,
        record: Any,
        current: Mapping[str, Mapping],
        baked: Optional[Mapping[str, Mapping]] = None,
        lighting: Optional[Mapping[str, Any]] = None,
        tol: float = 1.0,
    ) -> Dict[str, Any]:
        """The record after a bake of *baked* commits, at the next serial.

        - Each baked object: its current state, ``baked_at`` = the new serial,
          no stale reasons, the cell and map it was written to, *lighting*.
        - Every other participant: its current state.  When it differs from
          the recorded one the reasons are KEPT as its ``stale`` (the state
          they were found against is overwritten), and so is ``neighbour``
          when a change reaches it now (the old place it was near is about to
          be forgotten); its ``placed_at`` steps when the change reaches
          neighbours.  A newcomer is recorded at the new serial -- a
          duplicate of a tracked object kept stale, a marked one nothing
          baked kept ``untracked`` (its map is not vouched for by a bake of
          something else).
        - A participant that is gone is marked ``removed_at`` and pruned once
          no object within its reach still holds a map baked before it left.

        Neighbours are kept at the default reach whatever reach a re-bake
        chose: the record states what is true, a re-bake chooses what to redo.

        Parameters:
            record: The stored record (text, payload or :meth:`decode` form),
                or ``None`` for a scene's first.
            current: The scene now (see the class docstring) -- every static
                participant, not just the baked ones.
            baked: Id -> what its bake wrote: ``{"map", "cell" ([sx, sy, ox,
                oy]), "extent" ([w, h]), "size", "packing"}``, any omitted;
                ``map`` defaults to the object's marker. An id *current*
                lacks is ignored: only a participant can be recorded.
            lighting: The lighting the bake saw (:meth:`lighting_state`).
            tol: A multiplier on every tolerance.

        Returns:
            The new record, as :meth:`encode` returns it.
        """
        R = cls.Reason
        survey = cls._survey(record, current, None, set(), 1.0, tol)
        rec, current = survey["record"], survey["current"]
        serial = rec["serial"] + 1
        baked = {str(k): dict(v or {}) for k, v in (baked or {}).items()}
        light_state = digest = None
        if lighting is not None:
            light_state = cls.lighting_state(lighting)
            digest = cls._digest(light_state)

        out: Dict[str, Dict] = {}
        for k, p in current.items():
            e, marker_map = survey["live"].get(k), cls._marker_map(p)
            if e is None:
                entry = cls._entry({"state": p["state"]})
                entry.update(role=p["role"], placed_at=serial, map=marker_map)
                if k in survey["duplicates"]:
                    entry["stale"] = [R.DUPLICATE]
                elif marker_map is not None and k not in baked:
                    # Its arrival is recorded (it reaches its neighbours
                    # once), its map never vouched for: untracked until a
                    # bake of it, or a track, records it as baked.
                    entry["untracked"] = True
            else:
                entry = dict(e, state=p["state"], role=p["role"])
                if k in survey["now"]:
                    entry["placed_at"] = serial
                if survey["marked"].get(k):
                    kept = list(e["stale"] or []) + survey["changed"].get(k, [])
                    if k in survey["neighbours"]:
                        kept.append(R.NEIGHBOUR)
                    entry["stale"] = cls._ordered(kept) or None
                entry["map"] = entry["map"] or marker_map
            if k in baked:
                info = baked[k]
                entry.update(
                    baked_at=serial,
                    stale=None,
                    removed_at=None,
                    lighting=digest,
                    map=info.get("map") or marker_map or entry["map"],
                    cell=info.get("cell"),
                    extent=info.get("extent"),
                    size=info.get("size"),
                    packing=info.get("packing"),
                    untracked=False,
                )
            out[k] = entry
        for k, e in survey["objects"].items():
            if k not in out:
                out[k] = dict(e, removed_at=e["removed_at"] or serial)
        cls._prune(out)

        lightings = dict(rec["lightings"])
        if digest is not None:
            lightings[digest] = light_state
        used = {e["lighting"] for e in out.values() if e["removed_at"] is None}
        return cls.encode(
            {
                "version": cls.FORMAT,
                "serial": serial,
                "objects": out,
                "lightings": {d: v for d, v in lightings.items() if d in used},
            }
        )

    #: What :meth:`keep_baked` gives a re-packed object back from before.
    KEPT_ON_REPACK: Tuple[str, ...] = ("baked_at", "lighting", "stale", "untracked")

    @classmethod
    def keep_baked(cls, record: Any, before: Any, objects: Iterable[str]) -> Dict[str, Any]:
        """The record after a commit that only RE-PACKED *objects* (a Combine:
        their maps moved into one atlas, nothing rendered), each given back how
        it was baked: :attr:`KEPT_ON_REPACK` from *before*, and the lighting
        state its ``lighting`` names. One *before* never saw is untracked: a
        re-pack vouches for no map. An id *record* lacks is ignored; the cell
        each was re-packed to stays.

        Parameters:
            record: The stored record after that commit (any :meth:`decode`
                input).
            before: The record read before it (any :meth:`decode` input).
            objects: The re-packed ids.

        Returns:
            The record, as :meth:`encode` returns it.
        """
        record, before = cls.decode(record), cls.decode(before)
        for obj in objects:
            entry = record["objects"].get(obj)
            if entry is None:
                continue
            was = before["objects"].get(obj)
            if was is None:
                entry.update(baked_at=None, lighting=None, untracked=True)
                continue
            entry.update({key: was[key] for key in cls.KEPT_ON_REPACK})
            state = before["lightings"].get(was["lighting"] or "")
            if state is not None:
                record["lightings"][was["lighting"]] = state
        return cls.encode(record)

    @classmethod
    def _prune(cls, objects: Dict[str, Dict]) -> None:
        """Drop each removed entry no live map within its reach predates."""
        live = {k: e for k, e in objects.items() if e["removed_at"] is None}
        room = cls._union(e["state"].get("bbox") for e in live.values())
        maps = {
            k: (e["baked_at"] if e["baked_at"] is not None else e["placed_at"])
            for k, e in live.items()
            if e["role"] != cls.Role.EXCLUDED and e["map"]
        }
        for k, e in list(objects.items()):
            gone = e["removed_at"]
            if gone is None:
                continue
            older = {
                b: live[b]["state"].get("bbox") for b, s in maps.items() if s < gone
            }
            box = e["state"].get("bbox")
            if (
                not older
                or box is None
                or not cls._in_room(e["role"], e["state"])
                or not cls.neighbours({k: [box]}, older, room=room)
            ):
                del objects[k]

    # ---------------------------------------------------------------- rematch
    @classmethod
    def rematch(
        cls, record: Any, current: Mapping[str, Mapping], tol: float = 1.0
    ) -> Tuple[Dict[str, str], List[str], List[str]]:
        """Who the record's lost ids became, and what arrived unrecorded.

        - A rename: an entry gone from *current* and a newcomer with the same
          state and a compatible map (the entry recorded none, or the same
          file) -- re-keyed, never stale.
        - A duplicate: a marked newcomer whose marker reads exactly the map
          and ``scaleOffset`` of a tracked object still present -- it reads
          its source's cell, so its map is wrong.
        - Untracked: any other marked newcomer -- baked before change
          tracking, or brought in -- and any marked object a commit recorded
          without baking it (:meth:`advance`). Reported, never re-baked.

        Returns:
            ``(renames, duplicates, untracked)``: ``{old id: new id}`` and two
            sorted id lists.
        """
        current = {str(k): cls._participant(p) for k, p in (current or {}).items()}
        return cls._rematch(cls.decode(record)["objects"], current, tol)

    @classmethod
    def _rematch(cls, objects, current, tol):
        live = {k: e for k, e in objects.items() if e["removed_at"] is None}
        gone = sorted(k for k in live if k not in current)
        taken = {cls._marker_key(current[k]) for k in live if k in current}
        taken.discard(None)
        renames: Dict[str, str] = {}
        duplicates, untracked = [], []
        for k in sorted(current):
            if k in live:
                continue
            p, name = current[k], cls._marker_map(current[k])
            match = next(
                (
                    g
                    for g in gone
                    if g not in renames
                    and live[g]["map"] in (None, name)
                    and cls.same(live[g]["state"], p["state"], tol)
                ),
                None,
            )
            if match is not None:
                renames[match] = k
            elif name is None:
                continue  # an unmarked newcomer: placed, nothing to track
            elif cls._marker_key(p) in taken:
                duplicates.append(k)
            else:
                untracked.append(k)
        # ...and the marked ones a commit recorded unvouched (advance).
        untracked += [
            renames.get(k, k)
            for k, e in live.items()
            if e["untracked"]
            and renames.get(k, k) in current
            and cls._marker_map(current[renames.get(k, k)]) is not None
        ]
        return renames, duplicates, sorted(untracked)

    # --------------------------------------------------------------- lighting
    @classmethod
    def lighting_state(
        cls,
        lights: Optional[Mapping[str, Any]],
        environment: Optional[Any] = None,
        settings: Optional[Any] = None,
        rule: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """The lighting a bake sees, as one plain mapping.

        Parameters:
            lights: Light id -> its state (world matrix, keyable scalars,
                colour, an upstream file path, and ``"on"``: whether it lit
                the bake, :meth:`lights_used`) -- the lights the bake
                actually renders. A mapping that already holds
                :attr:`ENVIRONMENT` / :attr:`SETTINGS` / :attr:`RULE` passes
                through.
            environment: The environment's state when the bake includes it
                (an HDRI's path and rotation; a world node-tree hash).
            settings: Every bake setting that changes the light: GI depth.
            rule: What chose *lights*, in the form the DCC re-applies: its
                lighting switches and, when the Scope's lights alone light
                it, their ids. :meth:`lighting_rules` hands it back to a
                re-bake to light the same way, and :meth:`assess` to read the
                scene again by. Plain values, compared exactly; both DCCs
                record one shape.

        Returns:
            ``{light id: state, "@environment": ..., "@settings": ...,
            "@rule": ...}``, JSON-plain.
        """
        state = {str(k): v for k, v in (lights or {}).items()}
        if environment is not None:
            state[cls.ENVIRONMENT] = environment
        if settings is not None:
            state[cls.SETTINGS] = settings
        if rule is not None:
            state[cls.RULE] = rule
        return cls._copy(state)

    @classmethod
    def lighting_digest(
        cls,
        lights: Optional[Mapping[str, Any]],
        environment: Optional[Any] = None,
        settings: Optional[Any] = None,
        rule: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """blake2b-16 of :meth:`lighting_state` -- the key a lighting is
        recorded under.  Numbers are rounded to 6 significant digits first,
        so float noise rarely moves it; :meth:`light_changes` with tolerance
        decides whether a lighting under another digest really differs."""
        return cls._digest(cls.lighting_state(lights, environment, settings, rule))

    @classmethod
    def _lighting_reader(
        cls, lighting: Any
    ) -> Callable[[Optional[Dict[str, Any]]], Tuple[Dict[str, Any], str]]:
        """:meth:`assess`'s *lighting* as ``rule -> (state now, its digest)``,
        each rule read once."""
        if not callable(lighting):
            fixed = cls.lighting_state(lighting)
            answer = (fixed, cls._digest(fixed))
            return lambda _rule: answer
        seen: Dict[str, Tuple[Dict[str, Any], str]] = {}

        def read(rule):
            key = json.dumps(rule, sort_keys=True, default=cls._plain)
            if key not in seen:
                state = cls.lighting_state(lighting(cls._copy(rule)))
                state.setdefault(cls.RULE, cls._copy(rule))
                seen[key] = (state, cls._digest(state))
            return seen[key]

        return read

    @classmethod
    def lighting_rules(cls, record: Any) -> Dict[str, Optional[Dict[str, Any]]]:
        """Each live object's lighting rule: the :attr:`RULE` of the lighting
        its map was baked under (:meth:`lighting_state`) -- what a re-bake
        lights it by again, so it renders what the dry run read.

        Parameters:
            record: The stored record (text, payload or :meth:`decode` form).

        Returns:
            Object id -> its rule, or ``None`` where none is recorded (never
            baked, or baked before rules were).
        """
        rec = cls.decode(record)
        rules: Dict[str, Optional[Dict[str, Any]]] = {}
        for k, e in rec["objects"].items():
            if e["removed_at"] is not None:
                continue
            state = rec["lightings"].get(e["lighting"]) if e["lighting"] else None
            rule = (state or {}).get(cls.RULE)
            rules[k] = cls._copy(rule) if isinstance(rule, dict) else None
        return rules

    @classmethod
    def lights_used(
        cls, record: Any, ids: Optional[Iterable[str]] = None
    ) -> Optional[List[str]]:
        """The lights that lit the maps of *ids*: each light id recorded
        ``"on"`` (default true) in the lighting a map was baked under -- a
        hidden light the bake switched on included, a light it left out
        absent; never the environment, the settings or the rule. What an
        export ships for the maps' highlights, so they are the bake's own.

        Parameters:
            record: The stored record (text, payload or :meth:`decode` form).
            ids: The objects whose maps ship; ``None``: every live object
                baked under the record.

        Returns:
            The light ids, sorted -- or ``None`` when one of *ids* has no
            lighting recorded (untracked, or baked before the record kept
            it): the lights its map saw are unknown.
        """
        rec = cls.decode(record)
        objects, lightings = rec["objects"], rec["lightings"]
        if ids is None:
            ids = [
                k
                for k, e in objects.items()
                if e["removed_at"] is None and e["baked_at"] is not None
            ]
        used = set()
        for k in ids:
            e = objects.get(str(k))
            state = (
                lightings.get(e["lighting"])
                if e and e["removed_at"] is None and e["lighting"]
                else None
            )
            if state is None:
                return None
            used.update(
                light
                for light, value in state.items()
                if not light.startswith("@")  # the environment, settings, rule
                and not (isinstance(value, Mapping) and not value.get("on", True))
            )
        return sorted(used)

    @classmethod
    def _digest(cls, state: Mapping[str, Any]) -> str:
        text = json.dumps(cls._quantised(state), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

    @classmethod
    def _quantised(cls, value: Any) -> Any:
        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, (int, float)):
            return float(f"{float(value):.6g}") + 0.0  # + 0.0: no -0.0
        if isinstance(value, Mapping):
            return {str(k): cls._quantised(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._quantised(v) for v in value]
        return str(value)

    @classmethod
    def light_changes(
        cls,
        before: Optional[Mapping[str, Any]],
        after: Optional[Mapping[str, Any]],
        tol: float = 1.0,
    ) -> List[str]:
        """The light ids (and :attr:`ENVIRONMENT` / :attr:`SETTINGS`) that
        were added, removed or changed between two :meth:`lighting_state`
        values, sorted.  Numbers compare within :attr:`LIGHT_TOL` of their
        magnitude (at least 1); everything else exactly."""
        before = cls.lighting_state(before)
        after = cls.lighting_state(after)
        return sorted(
            k
            for k in set(before) | set(after)
            if k not in before
            or k not in after
            or not cls._values_close(before[k], after[k], tol)
        )

    @classmethod
    def _values_close(cls, a: Any, b: Any, tol: float) -> bool:
        if a is None or b is None or isinstance(a, bool) or isinstance(b, bool):
            return a == b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(a - b) <= cls.LIGHT_TOL * tol * max(1.0, abs(a), abs(b))
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            return a.keys() == b.keys() and all(
                cls._values_close(a[k], b[k], tol) for k in a
            )
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(
                cls._values_close(x, y, tol) for x, y in zip(a, b)
            )
        return a == b

    # ------------------------------------------------------------------- cell
    @classmethod
    def cell_fits(
        cls,
        recorded: Optional[Sequence[float]],
        current: Optional[Sequence[float]],
        tolerance: Optional[float] = None,
    ) -> bool:
        """Whether an object's atlas extent now (*current*, ``[w, h]``) is
        within *tolerance* (default :attr:`CELL_TOLERANCE`) of the *recorded*
        one on every axis -- so its re-bake still fits the cell it owns."""
        tolerance = cls.CELL_TOLERANCE if tolerance is None else tolerance
        if not recorded or not current or len(recorded) != len(current):
            return False
        return all(
            r > 0 and abs(c / r - 1.0) <= tolerance + 1e-9
            for r, c in zip(recorded, current)
        )

    # ---------------------------------------------------------------- atlases
    @dataclass
    class AtlasFiles:
        """The atlases a scene's objects read, each owned by an atlas group --
        what a baker reads once for its in-place plan and its split check
        (:meth:`BakeBaseline.atlas_files`).

        Attributes:
            markers: ``{reader: its marker record}``.
            entries: ``{object: baseline entry}`` of every object still recorded.
            by_file: ``{normcased path: [readers]}``, readers sorted.
            spelled: ``{normcased path: the path as read}``.
            lone: The one-reader maps an atlas bake wrote (a group of one), and
                those no record speaks for beside an atlas their group owns.
            group: ``{object: atlas group}`` for every reader and every object
                asked about.
            owned: ``{atlas group: {normcased paths}}`` -- each atlas belongs to
                the group most of its readers are in.
        """

        markers: Dict[str, Dict[str, Any]]
        entries: Dict[str, Dict[str, Any]]
        by_file: Dict[str, List[str]]
        spelled: Dict[str, str]
        lone: set
        group: Dict[str, str]
        owned: Dict[str, set]

        def splits(self) -> Dict[str, Tuple[List[str], List[str]]]:
            """``{atlas group: (map paths, objects)}`` -- each group that reads
            more than one map (one material in two shading groups, or a
            duplicate, baked apart before the groups were drawn as they are
            now): its maps, most-read first, and the group's objects reading
            them. What a Combine packs into one atlas."""
            out: Dict[str, Tuple[List[str], List[str]]] = {}
            for key, files in self.owned.items():
                if len(files) < 2:
                    continue
                files = sorted(files, key=lambda f: (-len(self.by_file[f]), f))
                readers = sorted(
                    r for f in files for r in self.by_file[f] if self.group[r] == key
                )
                out[key] = ([self.spelled[f] for f in files], readers)
            return out

    @classmethod
    def atlas_files(
        cls,
        markers: Mapping[str, Mapping[str, Any]],
        path_of: Mapping[str, str],
        record: Mapping[str, Any],
        group_of: Callable[[List[str]], Mapping[str, str]],
        objects: Sequence[str] = (),
    ) -> Optional["BakeBaseline.AtlasFiles"]:
        """The atlases a scene's objects read and the atlas group owning each
        (:class:`AtlasFiles`), with *objects*' groups read beside theirs;
        ``None`` when every map is its object's own (nothing to write into,
        nothing split).

        Each atlas is the group's most of its readers belong to: a reader whose
        material changed since (a merge, a reassigned object) is new to its NEW
        group's atlas, never the owner of its old. A group that reads two maps
        owns both -- a split.

        Parameters:
            markers: ``{reader: marker record}``, every object carrying one.
            path_of: ``{reader: the map's path on disk}``; a reader whose map
                is nowhere is absent.
            record: The bake baseline, decoded (:meth:`decode`).
            group_of: ``(objects) -> {object: atlas group}`` -- the host's rule.
            objects: More objects to group beside the readers.
        """
        by_file: Dict[str, List[str]] = {}
        spelled: Dict[str, str] = {}
        for reader in sorted(markers):
            path = path_of.get(reader)
            if path:
                key = os.path.normcase(os.path.abspath(path))
                by_file.setdefault(key, []).append(reader)
                spelled.setdefault(key, path)
        atlases = {f for f, readers in by_file.items() if len(readers) > 1}
        entries = {k: e for k, e in record["objects"].items() if e["removed_at"] is None}
        # An atlas bake gives a group of one its own map. Alone it is no atlas
        # to write into, but a group reading it beside another map is split
        # all the same: one material in two shading groups, or a duplicate.
        lone = {
            f
            for f, readers in by_file.items()
            if len(readers) == 1
            and (entries.get(readers[0]) or {}).get("packing") == "atlas"
        }
        if not atlases and len(lone) < 2:
            return None
        # A one-reader map whose packing no record speaks for (baked before
        # change tracking, then tracked as it stood) is a split-off only
        # beside an atlas its group owns: a group that owns none was baked per
        # object, and stays so.
        unknown = {
            f
            for f, readers in by_file.items()
            if len(readers) == 1 and not (entries.get(readers[0]) or {}).get("packing")
        }
        readers = [r for f in sorted(atlases | lone | unknown) for r in by_file[f]]
        group = group_of(list(dict.fromkeys(readers + list(objects))))
        owned: Dict[str, set] = {}
        for f in atlases:
            votes: Dict[str, int] = {}
            for reader in by_file[f]:
                votes[group[reader]] = votes.get(group[reader], 0) + 1
            owned.setdefault(min(votes, key=lambda g: (-votes[g], g)), set()).add(f)
        for f in sorted(unknown):
            if group[by_file[f][0]] in owned:
                lone.add(f)
        for f in lone:
            owned.setdefault(group[by_file[f][0]], set()).add(f)
        return cls.AtlasFiles(dict(markers), entries, by_file, spelled, lone, group, owned)

    # ----------------------------------------------------------------- record
    @classmethod
    def encode(cls, record: Mapping[str, Any]) -> Dict[str, Any]:
        """The payload to store for *record*: every key present, reasons and
        roles as their values, lists, plain numbers -- what
        ``SceneRecords.LIGHTMAP_BAKE_STATE.save`` takes.  ``version`` is
        :attr:`FORMAT` when *record* names none."""
        if isinstance(record, Mapping) and "version" not in record:
            record = dict(record, version=cls.FORMAT)
        return cls.decode(record)

    @classmethod
    def decode(cls, record: Any) -> Dict[str, Any]:
        """*record* as ``{"version", "serial", "objects", "lightings"}``.

        Tolerant and silent: text is parsed; anything unreadable, or of
        another :attr:`FORMAT`, reads as an empty record (serial 0) -- no
        baseline, so nothing reads as changed. An entry without a state is
        dropped; a missing field takes its default.
        """
        if isinstance(record, (bytes, bytearray)):
            record = record.decode("utf-8", "replace")
        if isinstance(record, str):
            try:
                record = json.loads(record)
            except ValueError:
                return cls._empty()
        if not isinstance(record, Mapping) or record.get("version") != cls.FORMAT:
            return cls._empty()
        try:
            record = cls._copy(record)
        except (TypeError, ValueError):
            return cls._empty()
        found = record.get("objects")
        objects = {}
        for key, entry in found.items() if isinstance(found, dict) else ():
            entry = cls._entry(entry)
            if entry is not None:
                objects[str(key)] = entry
        lightings = record.get("lightings")
        return {
            "version": cls.FORMAT,
            "serial": cls._int(record.get("serial")) or 0,
            "objects": objects,
            "lightings": {
                str(k): v
                for k, v in (lightings.items() if isinstance(lightings, dict) else ())
                if isinstance(v, dict)
            },
        }

    @classmethod
    def _empty(cls) -> Dict[str, Any]:
        return {"version": cls.FORMAT, "serial": 0, "objects": {}, "lightings": {}}

    @classmethod
    def _entry(cls, entry: Any) -> Optional[Dict[str, Any]]:
        """One object's entry with every field, or ``None`` without a state."""
        if not isinstance(entry, dict) or not isinstance(entry.get("state"), dict):
            return None
        stale = entry.get("stale")
        name = entry.get("map")
        packing = entry.get("packing")
        return {
            "state": entry["state"],
            "role": cls._text(entry.get("role") or cls.Role.BAKED),
            "placed_at": cls._int(entry.get("placed_at")) or 0,
            "baked_at": cls._int(entry.get("baked_at")),
            "stale": [cls._text(r) for r in cls._ordered(stale)] or None
            if isinstance(stale, list)
            else None,
            "removed_at": cls._int(entry.get("removed_at")),
            "lighting": entry.get("lighting")
            if isinstance(entry.get("lighting"), str)
            else None,
            "map": str(name).lower() if name else None,
            "cell": cls._floats(entry.get("cell"), 4),
            "extent": cls._floats(entry.get("extent"), 2),
            "size": cls._int(entry.get("size")),
            "packing": cls._text(packing) if packing else None,
            "untracked": bool(entry.get("untracked")),
        }

    @classmethod
    def _ordered(cls, reasons: Optional[Iterable[Any]]) -> List[Any]:
        """*reasons* deduplicated in declaration order, as :class:`Reason`
        members; a value this version does not know is kept, last."""
        order = {r.value: i for i, r in enumerate(cls.Reason)}
        values = {cls._text(r): None for r in reasons or ()}
        ranked = sorted(values, key=lambda v: (order.get(v, len(order)), v))
        return [cls.Reason(v) if v in order else v for v in ranked]

    @classmethod
    def label(cls, reason: Any) -> str:
        """How a report words *reason* (:attr:`LABELS`)."""
        value = cls._text(reason)
        return cls.LABELS.get(value, value)

    # ---------------------------------------------------------------- helpers
    @classmethod
    def _participant(cls, p: Mapping) -> Dict[str, Any]:
        """A scene row with a plain state, a role value and a marker dict."""
        marker = p.get("marker")
        return {
            "state": cls._copy(p.get("state") or {}),
            "role": cls._text(p.get("role") or cls.Role.BAKED),
            "marker": cls._copy(marker) if isinstance(marker, Mapping) else None,
        }

    @staticmethod
    def _marker_map(p: Mapping) -> Optional[str]:
        name = (p.get("marker") or {}).get("map")
        return str(name).lower() if name else None

    @classmethod
    def _marker_key(cls, p: Mapping) -> Optional[Tuple]:
        """``(map, scaleOffset)`` of *p*'s marker -- what a duplicate shares."""
        name, rect = cls._marker_map(p), (p.get("marker") or {}).get("scaleOffset")
        if not name or not isinstance(rect, (list, tuple)):
            return None
        return (name, tuple(float(v) for v in rect))

    @classmethod
    def can_bake(cls, participant: Mapping) -> bool:
        """Whether a scene-state *participant* (``{"role", "state"}``) can bake
        now: rendered, and not in the Exclude set -- what :meth:`assess`
        leaves out of the targets as ``unbakeable``."""
        return participant["role"] != cls.Role.EXCLUDED and bool(
            participant["state"].get("rendered", True)
        )

    @classmethod
    def _in_room(cls, role: Any, state: Mapping) -> bool:
        """Whether a participant renders in other objects' passes."""
        return cls._text(role) != cls.Role.ISOLATED.value and bool(
            state.get("rendered", True)
        )

    @staticmethod
    def _text(value: Any) -> str:
        return value.value if isinstance(value, Enum) else str(value)

    @staticmethod
    def _int(value: Any) -> Optional[int]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return int(value) if math.isfinite(value) else None

    @staticmethod
    def _floats(value: Any, n: int) -> Optional[List[float]]:
        if not isinstance(value, (list, tuple)) or len(value) != n:
            return None
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
        ):
            return None
        return [float(v) for v in value]

    @staticmethod
    def _plain(value: Any) -> Any:
        """``json.dumps`` default: numpy values and arrays as Python ones."""
        if isinstance(value, Enum):
            return value.value
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)

    @classmethod
    def _copy(cls, value: Any) -> Any:
        """A JSON-plain deep copy (tuples -> lists, enums -> values)."""
        return json.loads(json.dumps(value, default=cls._plain))


@dataclass(frozen=True)
class StaleReport:
    """What a re-bake of the changed objects redoes, and why -- the dry run.

    Built by :meth:`BakeBaseline.assess`; a DCC adds its packing plan
    (``in_place``, ``repack``) with ``dataclasses.replace``.

    Attributes:
        targets: The objects to re-bake, sorted: every stale object and every
            neighbour, less the unbakeable.
        stale: Object -> its own reasons (:class:`BakeBaseline.Reason`).
        neighbours: Object -> the changed participants that reach it.
        untracked: Marked objects the record has never seen (baked before
            change tracking, or brought in): reported, never re-baked.
        unbakeable: Stale objects and neighbours that cannot bake now (hidden,
            or in the Exclude set): they keep their maps.
        lighting: The lights (and ``@environment`` / ``@settings`` /
            ``@rule``) whose change stales every map lit by them -- the whole
            room when every light lit it.
        in_place: Targets written back into their own atlas cells.
        repack: Atlas file -> why its whole group re-packs.
        joining: Objects that re-bake only because their atlas re-packs
            whole (``repack``): neither stale nor a neighbour.
        forked: Atlas file -> the scene file that wrote it: copied under this
            scene's own name first, then written in place -- a Save As copy's
            first update, its source's map left as it was.
        split: Map file -> every map of its atlas group, when the group reads
            more than one (a material in two shading groups, a duplicate
            material, baked apart): one atlas would do. Never a re-bake
            target by itself -- a DCC combines the maps, no render.
        split_materials: Map file -> the materials its split group's objects
            use, sorted: more than one is a duplicate still in the scene.
        out_of_bake: Visible meshes no bake renders (never baked, not in the
            Exclude set): out of every map's lighting until they are baked or
            excluded, so they stale nothing.
        baked: How many baked objects the scene holds (marked and tracked,
            whatever the report is limited to) -- what :meth:`summary` counts
            a lighting change's reach against; 0 when unknown.
    """

    targets: Tuple[str, ...] = ()
    stale: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    neighbours: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    untracked: Tuple[str, ...] = ()
    unbakeable: Tuple[str, ...] = ()
    lighting: Tuple[str, ...] = ()
    in_place: Tuple[str, ...] = ()
    repack: Mapping[str, str] = field(default_factory=dict)
    joining: Tuple[str, ...] = ()
    forked: Mapping[str, str] = field(default_factory=dict)
    split: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    split_materials: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    out_of_bake: Tuple[str, ...] = ()
    baked: int = 0

    @staticmethod
    def _count(n: int, noun: str, plural: Optional[str] = None) -> str:
        """``"1 atlas"`` / ``"2 atlases"``."""
        return f"{n} {noun if n == 1 else plural or noun + 's'}"

    @staticmethod
    def leaf(item: str) -> str:
        """An id's last ``|`` component -- the default display name."""
        return str(item).rpartition("|")[2] or str(item)

    def summary(self, name: Optional[Callable[[str], str]] = None) -> str:
        """One sentence for a panel footer: how many re-bake and why.

        A lighting change names what it stales: "the whole room" only when
        that is every baked object (:attr:`baked`), else how many of them.

        Parameters:
            name: Id -> display name; default :meth:`leaf`.
        """
        name = name or self.leaf
        changed = sum(1 for k in self.targets if self.stale.get(k))
        near = sum(
            1 for k in self.targets if not self.stale.get(k) and k in self.neighbours
        )
        # A target with no reason was asked for (a re-bake of given objects).
        asked = len(self.targets) - changed - near
        clauses = []
        if self.lighting:
            relit = sum(
                1
                for reasons in self.stale.values()
                if BakeBaseline.Reason.LIGHTING in reasons
            )
            if self.baked and relit >= self.baked:
                reach = "the whole room"
            elif self.baked:
                plural = "s" if self.baked != 1 else ""
                reach = f"{relit} of {self.baked} baked object{plural}"
            else:
                reach = f"{relit} baked object{'s' if relit != 1 else ''}"
            clauses.append(
                f"lighting changed ({', '.join(name(k) for k in self.lighting)})"
                f" -- {reach}"
            )
        if self.in_place:
            clauses.append(f"{len(self.in_place)} written in place")
        if self.repack:
            clause = f"{self._count(len(self.repack), 'atlas', 'atlases')} re-packed whole"
            if self.joining:
                clause += f" (+{self._count(len(self.joining), 'more object')})"
            clauses.append(clause)
        if self.forked:
            clauses.append(
                f"{self._count(len(self.forked), 'atlas', 'atlases')} copied from "
                "another scene first"
            )
        if self.unbakeable:
            clauses.append(f"{len(self.unbakeable)} cannot bake")
        if self.untracked:
            clauses.append(
                f"{len(self.untracked)} untracked (no baseline; left as baked)"
            )
        if self.out_of_bake:
            clauses.append(
                f"{self._count(len(self.out_of_bake), 'visible mesh', 'visible meshes')}"
                " out of the bake"
            )
        if not self.targets:
            if not clauses:
                return "Nothing changed since the last bake."
            head = "Nothing to re-bake"
        else:
            counts = [f"{changed} changed"] if changed else []
            if near:
                counts.append(f"{near} neighbour{'s' if near != 1 else ''}")
            if asked:
                counts.append(f"{asked} requested")
            head = f"{len(self.targets)} to re-bake: {', '.join(counts)}"
        return "; ".join([head] + clauses) + "."

    def lines(self, name: Optional[Callable[[str], str]] = None) -> List[str]:
        """One line per object, then the lighting, re-packs, the unbakeable
        and the untracked -- for a log.

        Parameters:
            name: Id -> display name; default :meth:`leaf`.
        """
        name = name or self.leaf
        placed = set(self.in_place)
        out = [
            f"{name(k)}: {self.why(k, name)}" + (" [in place]" if k in placed else "")
            for k in self.targets
        ]
        if self.lighting:
            out.append(f"Lighting: {', '.join(name(k) for k in self.lighting)} changed")
        out += [
            f"{atlas}: re-packed whole -- {why}" for atlas, why in self.repack.items()
        ]
        out += [f"{name(k)}: {self.JOINS}" for k in self.joining]
        out += [
            f"{atlas}: copied first -- written by {writer}"
            for atlas, writer in self.forked.items()
        ]
        out += [
            f"{name(k)}: cannot bake ({self.why(k, name)}) -- keeps its map"
            for k in self.unbakeable
        ]
        out += [
            f"{name(k)}: untracked -- no baseline, not re-baked" for k in self.untracked
        ]
        out += [f"{name(k)}: {self.OUT_OF_BAKE}" for k in self.out_of_bake]
        return out

    #: How :meth:`lines` and :meth:`doc` word a ``joining`` object.
    JOINS = "re-bakes with its atlas, which re-packs whole"
    #: ... and an ``out_of_bake`` mesh.
    OUT_OF_BAKE = "visible, never baked and not excluded -- out of every map's lighting"

    def doc(
        self,
        maps: Sequence[Mapping[str, Any]] = (),
        name: Optional[Callable[[str], str]] = None,
    ) -> "ReportDoc":
        """The Bake State breakdown: every map, what re-bakes and why, and
        what is out of the bake -- a ``ReportDoc`` for a viewer
        (``to_html``, its ``action://`` links selecting objects and opening
        folders) or a log (``to_text``).

        Parameters:
            maps: One row per lightmap the scene reads: ``{"map": file name,
                "path": the file found, or None (missing), "folder": the folder
                as recorded, "objects": [reader ids], "writer": the OTHER scene
                file that wrote it, or None, "size": (w, h) or None}``; a probe
                row adds ``"probe": True`` (:meth:`map_rows`), and a map written
                since the scene was last saved ``"saves_to"``: the file it becomes
                at the save (``PendingFiles``), its row's ``"folder"`` that one's.
            name: Id -> display name; default :meth:`leaf`.
        """
        from pythontk.str_utils.report_doc import ReportDoc as D

        name = name or self.leaf
        doc = D().heading("Lightmap Bake State", 1).text(self.summary(name))
        rows = self.map_rows(maps)
        if not rows:
            return doc.text("Nothing in this scene is baked yet.")

        map_of: Dict[str, str] = {}
        table = []
        for row in rows:
            file, path = str(row["map"]), row.get("path")
            readers = list(row.get("objects") or ())
            for obj in readers:
                map_of[obj] = file
            folder = row.get("folder") or "-"
            size = row.get("size")
            state = D.span(row["state"], row["level"])
            saves_to = row.get("saves_to")
            if saves_to:  # waits in the session: its folder, once saved
                where = D.join(
                    [
                        D.action(folder, "open", path=os.path.dirname(saves_to)),
                        D.span(self.ON_SAVE, "info", italic=True),
                    ],
                    " ",
                )
            elif path:
                where = D.action(folder, "open", path=os.path.dirname(path))
            else:
                where = folder
            table.append(
                [
                    D.file(path, file) if path else D.span(file, "error"),
                    where,
                    "probe"
                    if row.get("probe")
                    else D.action(str(len(readers)), "select", node=readers),
                    f"{size[0]} x {size[1]}" if size else "-",
                    D.span(row["writer"], "warn") if row.get("writer") else "this scene",
                    state,
                ]
            )
        doc.heading("Maps").table(
            ["Map", "Folder", "Objects", "Size", "Written by", "State"],
            table,
            align="llrrll",
        )
        waiting = sum(1 for row in rows if row.get("saves_to"))
        if waiting:
            doc.text(
                D.span(
                    f"{waiting} map{'s' if waiting != 1 else ''} {self.ON_SAVE}: "
                    + self.UNSAVED,
                    "info",
                    italic=True,
                )
            )
        # What each state means and the button that fixes it -- the panel's
        # tooltip, for every map that is not up to date.
        attention = [row for row in rows if row["level"] != "ok"]
        if attention:
            doc.heading("Needs attention").items(
                [f"{row['map']} ({row['state']}): {row['hint']}" for row in attention]
            )

        placed = set(self.in_place)
        redo = [
            [
                D.action(name(k), "select", node=k),
                map_of.get(k, "-"),
                self.why(k, name),
                "in place" if k in placed else "re-pack",
            ]
            for k in self.targets
        ] + [
            [D.action(name(k), "select", node=k), map_of.get(k, "-"), self.JOINS, "re-pack"]
            for k in self.joining
        ]
        if redo:
            doc.heading("To re-bake").table(["Object", "Map", "Why", "How"], redo)
        for title, ids, hint in (
            (
                "Cannot bake now",
                self.unbakeable,
                "Hidden, or in the Exclude set: each keeps the map it has.",
            ),
            (
                "Untracked",
                self.untracked,
                "Baked before change tracking, or brought in: left as baked "
                "until a bake records them.",
            ),
            (
                "Out of the bake",
                self.out_of_bake,
                "Visible, never baked and not excluded: they cast no shadow or "
                "bounce into any map. Bake them, or add them to the Exclude "
                "set to have them cast without a map of their own.",
            ),
        ):
            if ids:
                doc.heading(title).text(hint).items(
                    [D.action(name(k), "select", node=k) for k in ids]
                )
        return doc

    def map_rows(self, maps: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """The maps a scene reads, each with its state -- what a panel's
        table shows and :meth:`doc` lays out, so both read one definition.

        Parameters:
            maps: As :meth:`doc`'s; a row with no readers and no ``"probe"``
                flag is left out.

        Returns:
            A copy of each row, adding ``"state"`` -- the map's condition now,
            never what a re-bake will do to it -- ``"level"`` (a ``ReportDoc``
            span class), ``"hint"`` (why, and which action remedies it) and
            ``"stale"`` (how many of its readers re-bake). The states, first
            that applies:

            - ``"missing"`` (error): the file is on disk nowhere.
            - ``"split"`` (warn): its atlas group reads more than one map
              (:attr:`split`).
            - ``"N changed"`` (warn): N of its readers changed since the bake.
            - ``"group changed"`` (warn): none of its readers did, but its
              atlas is rebuilt for its group (:attr:`repack` says why).
            - ``"untracked"`` (info): a reader was baked before change
              tracking, so its changes are not detected.
            - ``"up to date"`` (ok).
        """
        targets = set(self.targets)
        untracked = set(self.untracked)
        unbakeable = set(self.unbakeable)
        repack = {k.lower(): why for k, why in self.repack.items()}
        forked = {k.lower(): who for k, who in self.forked.items()}
        split = {k.lower(): files for k, files in self.split.items()}
        materials = {k.lower(): m for k, m in self.split_materials.items()}
        rows = []
        for row in maps:
            readers = list(row.get("objects") or ())
            if not (readers or row.get("probe")):
                continue
            file = str(row["map"]).lower()
            stale = sum(1 for obj in readers if obj in targets)
            state, level, hint = self._map_state(
                row, len(readers), stale, repack.get(file), forked.get(file),
                split.get(file), materials.get(file) or (),
                any(obj in untracked for obj in readers),
                bool(readers) and all(obj in unbakeable for obj in readers),
            )
            if row.get("saves_to"):
                hint = f"{hint} {self.UNSAVED}"
            rows.append(
                {**row, "state": state, "level": level, "hint": hint, "stale": stale}
            )
        return rows

    #: The mark :meth:`doc` puts on a map written since the scene was last
    #: saved (a row's ``"saves_to"``), and what it means -- every hint of one.
    ON_SAVE = "on save"
    UNSAVED = (
        "Written since the scene was last saved, so it waits in a temp folder "
        "the scene reads it from, and reaches its folder when the scene is "
        "saved. Not saved, the folder stays as the saved scene left it."
    )

    #: The states :meth:`tally` counts, in the order it names them.
    TALLY = ("missing", "split", "changed", "untracked", "up to date")

    @classmethod
    def tally(cls, rows: Sequence[Mapping[str, Any]]) -> str:
        """*rows*' (:meth:`map_rows`) states counted, worst first -- a footer's
        line: ``"1 split, 2 untracked, 1 up to date"``. "N changed" and "group
        changed" count as changed."""
        counts: Dict[str, int] = {}
        for row in rows:
            state = "changed" if row["state"].endswith("changed") else row["state"]
            counts[state] = counts.get(state, 0) + 1
        return ", ".join(f"{counts[s]} {s}" for s in cls.TALLY if s in counts) or "No maps."

    def fix_plan(self, rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[str]]:
        """What a Fix All does about *rows* (:meth:`map_rows`), so none is left
        needing a fix -- the one rule the DCC panels confirm, then run.

        Returns:
            ``{"combine": maps, "rebake": objects, "revert": objects,
            "track": objects}``:

            - ``combine``: the split maps no re-bake mends -- no render.
            - ``rebake``: :attr:`targets` -- what changed, a missing map's
              readers among them.
            - ``revert``: a missing map's readers no re-bake renders (hidden
              or excluded, :attr:`unbakeable`): nothing can make their map, so
              its marker comes off them.
            - ``track``: the untracked objects nothing above takes, recorded
              as they stand -- no render; confirming is the artist vouching
              for their maps.
        """
        rows = list(rows)
        redo = set(self.targets) | set(self.joining)
        unbakeable = set(self.unbakeable)
        revert = sorted(
            {
                obj
                for row in rows
                if row["state"] == "missing"
                for obj in row.get("objects") or ()
                if obj in unbakeable
            }
        )
        return {
            "combine": sorted(
                row["map"]
                for row in rows
                if row["state"] == "split"
                and not redo.intersection(row.get("objects") or ())
            ),
            "rebake": list(self.targets),
            "revert": revert,
            "track": sorted(set(self.untracked) - redo - set(revert)),
        }

    #: How a panel redoes what changed -- the remedy each changed state's hint
    #: names (the Lightmap Bakers' footer button, then their table menu).
    REDO = "Fix All, or right-click: Re-Bake,"

    @classmethod
    def _map_state(
        cls,
        row: Mapping[str, Any],
        readers: int,
        stale: int,
        repack: Optional[str],
        writer: Optional[str],
        split: Optional[Sequence[str]],
        materials: Sequence[str],
        untracked: bool,
        unbakeable: bool = False,
    ) -> Tuple[str, str, str]:
        """``(state, level, hint)`` of one :meth:`map_rows` row; *unbakeable*:
        every reader is one no re-bake renders (:attr:`unbakeable`)."""
        every = f"all {readers} objects reading it" if readers != 1 else "its object"
        if not row.get("path"):
            where = f"Not on disk (expected in {row.get('folder') or 'its folder'})."
            if unbakeable:
                return (
                    "missing",
                    "error",
                    f"{where} {'Its object is' if readers == 1 else every.capitalize() + ' are'} "
                    "hidden or excluded, so no re-bake renders it. Fix All, or "
                    "right-click: Revert, takes the lightmap off -- or show it "
                    "and re-bake, or put the file back.",
                )
            return (
                "missing",
                "error",
                f"{where} {cls.REDO} bakes it again -- or put the file back.",
            )
        if split:
            others = ", ".join(f for f in split if f.lower() != str(row["map"]).lower())
            if len(materials) > 1:
                why = (
                    f"Its objects use duplicate materials ({', '.join(materials)}), "
                    f"baked into {len(split)} maps (this and {others}) where one "
                    "atlas would do."
                )
                fix = "packs the maps into one atlas and merges the duplicates"
            else:
                why = (
                    "Its objects share one material"
                    + (f" ({materials[0]})" if materials else "")
                    + f", yet read {len(split)} maps (this and {others}): baked "
                    "while it was duplicated, or from two shading groups."
                )
                fix = "packs the maps into one atlas"
            return (
                "split",
                "warn",
                f"{why} Each extra map is one more texture to load, and its "
                f"objects can't batch with the rest. Fix All, or right-click: "
                f"Combine, {fix} -- no render.",
            )
        if stale:
            hint = f"{stale} of its {readers} objects changed since the bake. {cls.REDO} "
            if repack:
                hint += f"rebuilds the atlas with {every} ({repack})."
            else:
                hint += "puts them back into their own cells, every other cell kept."
            if writer:
                hint += (
                    f" {writer} wrote this map: this scene gets its own copy first, "
                    "and that file keeps the original."
                )
            return f"{stale} changed", "warn", hint
        if repack:
            return (
                "group changed",
                "warn",
                "Nothing in it changed, but its material group's atlas is rebuilt "
                f"({repack}). {cls.REDO} re-bakes {every}.",
            )
        if untracked:
            return (
                "untracked",
                "info",
                "Baked before change tracking, or brought in from another file: "
                "changes to its objects go unseen. Fix All, or right-click: Start "
                "Tracking, records them as they stand, no render -- or Re-Bake if "
                "they may have changed since this map was baked.",
            )
        return "up to date", "ok", "Matches the scene as baked: nothing to do."

    def why(self, k: str, name: Optional[Callable[[str], str]] = None) -> str:
        """Why object *k* re-bakes, in words: its own changes, else the
        neighbours whose change reaches it, else ``"requested"``.

        Parameters:
            k: An object id.
            name: Id -> display name; default :meth:`leaf`.
        """
        name = name or self.leaf
        causes = self.neighbours.get(k)
        parts = [
            BakeBaseline.label(r)
            for r in self.stale.get(k, ())
            if not (causes and r == BakeBaseline.Reason.NEIGHBOUR)
        ]
        if causes:
            parts.append("neighbour of " + ", ".join(name(a) for a in causes))
        return "; ".join(parts) or "requested"
