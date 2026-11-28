# !/usr/bin/python
# coding=utf-8
"""SpotPlacement -- where a thing placed for a purpose first stands: a viewpoint, a camera, an arrow.

One placement every tool uses for a first pose, whatever it places and whichever host it runs in: where a person stands
to see a part (``stand``), where a camera frames it (``view``), where an arrow's tip points at it (``point``). It reads
bounds and, where the host gives it rays (:class:`SpotRays`: :class:`~pythontk.ProbeRays` over a DCC's meshes,
:class:`BoxRays` over boxes, a port's own scene queries), what is seen and where there is room. Without rays the first
pose is the plain one, from bounds alone.

A first pose is a convenience: the thing placed is an object a person moves, and nothing places it again once it is
there. The rules, each forced by a real scene:

- **stand**: on the floor (where the scene has floor at that height, ground under the spot), around the part at the
  stand-off (further back, up to 3 m, where a field of view asks the part to fit), with room for a body (a capsule from
  the knees to over the eyes) and outside every keep-out box (where something comes to stand later: a door's swing),
  its eyes seeing the part. Of those: the one that sees most of the part and of what else the person acts on from there;
  a side the sheet names outranks sight, the delivery's outer side only breaks ties; then the nearest ring, then the
  nearest to where the person stood before.
- **view**: at the distance that frames the part (a tiny part framed with what it sits on), in free space, seeing the
  most of it read finely (27 points: a part behind a grille is seen through its slots), first; of spots that see it
  alike, a flat part face-on (a dial seen across its face reads, one seen along it does not: flat, its thinnest extent
  under three quarters of the next and under half its face's size), then the eyes' side, then the most level.
- **point**: the tip just off the part, the arrow's body and its bob in free space with room to turn, seen from the
  eyes: above where that works, else across the line of sight (never end-on: an arrow seen along its length is a dot),
  else beside, else straight out of what the part is mounted on (a button on a panel), where it also goes when nothing
  is free.

Where a purpose needs the person's eyes and none are given, they are those of the ``stand`` spot for the same part.

Lengths a caller passes as a setting (``stand_off``, ``gap``...) are metres; positions and boxes are in the scene's
units (``unit`` of them in a metre). The level axes turn from the first after up (``X``) toward the second (``Z``
where ``Y`` is up, ``-Y`` where ``Z`` is: the same turn in both frames).

A port of these rules to another runtime is held to them by the golden cases of
``ptk.Conformance.cases("spot_placement")`` (:class:`SpotPlacementConformance`).
"""

import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pythontk.math_utils._math_utils import MathUtils

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover - Python < 3.8
    Protocol = object

    def runtime_checkable(cls):
        return cls


Vector = Tuple[float, float, float]
Bounds = Tuple[Sequence[float], Sequence[float]]


@runtime_checkable
class SpotRays(Protocol):
    """What a placement casts and checks through, in the scene's units.

    ``cast`` is the nearest hit within *far* along a unit *direction*, passing over the owners in *skip*; a hit has
    ``distance``, ``normal`` (the face's, world space) and ``mesh`` (the index of its owner in ``owners``). A ray that
    starts inside a thing need not hit it; one that does meets its inside (a face turned away from the ray).
    ``occupied_sphere`` and ``occupied_capsule`` say whether a sphere or a capsule (the segment *a*-*b* swept by
    *radius*) overlaps anything, a capsule read along its axis at :meth:`SpotPlacement.stations`.
    """

    owners: List[Any]

    def cast(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        far: float = 1.0e7,
        skip: Any = None,
    ) -> Any: ...

    def occupied_sphere(self, at: Sequence[float], radius: float) -> bool: ...

    def occupied_capsule(
        self, a: Sequence[float], b: Sequence[float], radius: float
    ) -> bool: ...


class _SpotPlacementInternal(object):
    """Vector arithmetic on plain tuples (a few thousand rays need no numpy), and the frame. Products, lengths and
    units are :class:`~pythontk.MathUtils`' (measured on the golden cases: as fast as written out here, the rays'
    casts being the cost)."""

    #: A later candidate must beat the best by this much: equal scores keep the first, in every runtime alike.
    EPS = 1.0e-9

    @staticmethod
    def _add(a, b, k=1.0) -> Vector:
        return (a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k)

    @staticmethod
    def _sub(a, b) -> Vector:
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    @staticmethod
    def _scale(a, k) -> Vector:
        return (a[0] * k, a[1] * k, a[2] * k)

    @staticmethod
    def _unit(a) -> Vector:
        """*a* as a unit vector; zero where it has no length."""
        return MathUtils.safe_normalize(a, (0.0, 0.0, 0.0))

    @staticmethod
    def _centre(bounds: Bounds) -> Vector:
        low, high = bounds
        return tuple((float(low[i]) + float(high[i])) * 0.5 for i in range(3))

    @staticmethod
    def _extents(bounds: Bounds) -> Vector:
        low, high = bounds
        return tuple(abs(float(high[i]) - float(low[i])) * 0.5 for i in range(3))

    def _frame(self) -> Tuple[Vector, Vector, Vector]:
        """``(up, first level axis, second level axis)``: the turn from the first to the second is the same in both
        frames (``X`` to ``Z`` where ``Y`` is up, ``X`` to ``-Y`` where ``Z`` is)."""
        if self.up == 2:
            return (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)
        if self.up == 0:
            return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        return (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)

    def _turn(self, lead: Vector, angle: float, elevation: float) -> Vector:
        """A unit direction *angle* (radians) round from the level unit *lead*, the way the first level axis turns
        toward the second, and *elevation* above level: *lead* itself, exactly, at zero."""
        up = self._frame()[0]
        side = MathUtils.cross_product(lead, up)  # a quarter turn on from lead
        c = math.cos(elevation)
        return self._add(
            self._add(
                self._scale(lead, c * math.cos(angle)), side, c * math.sin(angle)
            ),
            up,
            math.sin(elevation),
        )

    def _flat(self, way: Sequence[float]) -> Vector:
        """*way* with its up part taken out, as a unit vector (zero where nothing is left)."""
        up = self._frame()[0]
        return self._unit(self._add(way, up, -MathUtils.dot_product(way, up)))

    def _samples(self, target: Bounds, fine: bool = False) -> List[Vector]:
        """Where a part is looked at to see how much of it is seen: its middle and its corners drawn in to 0.6 of its
        extents; *fine*: 27 points through it (middle, faces, edges, corners) at 0.8, which pass a grille's slots."""
        centre, extents = self._centre(target), self._extents(target)
        if fine:
            e = self._scale(extents, 0.8)
            return [
                (centre[0] + x * e[0], centre[1] + y * e[1], centre[2] + z * e[2])
                for x in (-1, 0, 1)
                for y in (-1, 0, 1)
                for z in (-1, 0, 1)
            ]
        e = self._scale(extents, 0.6)
        out = [centre]
        for i in range(1, 9):
            out.append(
                (
                    centre[0] + (e[0] if (i & 1) == 0 else -e[0]),
                    centre[1] + (e[1] if (i & 2) == 0 else -e[1]),
                    centre[2] + (e[2] if (i & 4) == 0 else -e[2]),
                )
            )
        return out


class SpotPlacement(_SpotPlacementInternal):
    """The first pose of a thing placed for a purpose (see the module docstring).

    Parameters:
        rays: What the host casts rays and checks room through (:class:`SpotRays`); ``None``: bounds alone, the
            plain first pose.
        up: The up axis's index (1: ``Y``; 2: ``Z``).
        unit: The scene's units in a metre (100 in a centimetre scene).
    """

    #: Metres between a viewpoint and the side of what its user faces.
    STAND_OFF = 1.5
    #: A person's eyes above the floor, in metres.
    EYE_HEIGHT = 1.6
    #: A person's room to stand, in metres: half their shoulders and an arm's bend.
    BODY_RADIUS = 0.35
    #: What a person steps over, in metres: the body's room starts this far above the floor.
    KNEES = 0.3
    #: The farthest a viewpoint stands back to see its part whole, in metres.
    MOST_BACK = 3.0
    #: The share of the view's height a part fills, at most, where a viewpoint stands back to see it whole.
    FILL = 0.8
    #: How far above the floor's lowest point its walkable top may be, in metres (higher is a table or a step).
    FLOOR_RISE = 0.3
    #: How far below a part's lowest point the floor may lie, in metres.
    FLOOR_DEPTH = 100.0
    #: A hit this close short of a sampled point still sees it, in metres.
    NEAR = 0.02
    #: A camera's room in free space, as a radius in metres.
    FREE = 0.05
    #: Metres between a pointer's tip and its part.
    GAP = 0.04
    #: A pointer's length, tip to tail, in metres.
    POINTER = 0.13
    #: A picture's vertical field of view, in degrees.
    VIEW_ANGLE = 30.0
    #: The room around a picture's part (1: it just fits).
    MARGIN = 1.4
    #: The least a picture frames, as a radius in metres: a smaller part is shown with what it sits on.
    LEAST_FRAMED = 0.06
    #: An arrow within this many degrees of the line of sight is end-on (seen as a dot).
    END_ON = 30.0
    #: How much a picture of a flat part (:meth:`_face`) prefers its face, among spots that see the part alike: over the
    #: eyes' side (0.4 either way) and the level (a sixtieth a degree).
    FACE_ON = 2.0
    #: A capsule's room is read along its axis at most this many of its radii apart (:meth:`stations`).
    STATION = 0.25

    def __init__(self, rays: Optional[SpotRays] = None, up: int = 1, unit: float = 1.0):
        self.rays = rays
        self.up = up
        self.unit = unit

    @classmethod
    def stations(
        cls, a: Sequence[float], b: Sequence[float], radius: float
    ) -> List[Vector]:
        """The points along a capsule's axis, *a* to *b* both included, at most :attr:`STATION` of its *radius*
        apart: where every :class:`SpotRays` reads a capsule's room (:class:`BoxRays` a sphere at each, a host's
        rays out of each), so a body's room is read alike over boxes and over a host's meshes."""
        length = math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))
        if length <= 0.0:
            return [tuple(float(c) for c in a)]
        spacing = radius * cls.STATION
        steps = int(math.ceil(length / spacing)) if spacing > 0.0 else 1
        return [
            tuple(a[i] + (b[i] - a[i]) * s / steps for i in range(3))
            for s in range(steps + 1)
        ]

    # ---- what is seen -------------------------------------------------------------------------------------------

    def clear(
        self,
        origin: Sequence[float],
        point: Sequence[float],
        owners: Iterable[Any] = (),
    ) -> bool:
        """Whether nothing but *owners* (the thing looked at) stands between *origin* and *point*. Without rays,
        always."""
        if self.rays is None:
            return True
        to = self._sub(point, origin)
        length = MathUtils.get_magnitude(to)
        if length < 1e-4 * self.unit:
            return True
        hit = self.rays.cast(origin, self._scale(to, 1.0 / length), length)
        if hit is None or self.rays.owners[hit.mesh] in set(owners):
            return True
        return hit.distance >= length - self.NEAR * self.unit

    def seen(
        self,
        origin: Sequence[float],
        target: Bounds,
        owners: Iterable[Any] = (),
        fine: bool = False,
    ) -> float:
        """The share (0..1) of *target* seen from *origin*: of its sample points (:meth:`_samples`), those a clear
        line reaches."""
        owners = set(owners)
        samples = self._samples(target, fine)
        return sum(1 for p in samples if self.clear(origin, p, owners)) / float(
            len(samples)
        )

    def floor_near(
        self, target: Bounds, depth: Optional[float] = None
    ) -> Tuple[float, bool]:
        """The floor around *target*: the lowest top that rays cast straight down, one to three metres out from its
        side, land on. A ray goes on through every top it lands on, so a ceiling or a shelf over the floor never
        stands for it; a face it meets turned away (a thing's inside: a wall's foot that reaches below the floor) is
        no top. ``(floor, found)``; not found, the target's lowest point."""
        depth = self.FLOOR_DEPTH if depth is None else depth
        low, high = target
        up, h1, h2 = self._frame()
        u, ui = self.unit, self.up
        if self.rays is None:
            return float(low[ui]), False
        centre, extents = self._centre(target), self._extents(target)
        top = float(high[ui]) + 1.0 * u
        edge = max(
            abs(MathUtils.dot_product(extents, h1)),
            abs(MathUtils.dot_product(extents, h2)),
        )
        down = self._scale(up, -1.0)
        far = top - float(low[ui]) + max(0.0, depth) * u
        floor = None
        for ring in range(1, 4):
            for k in range(16):
                a = math.radians(k * 22.5)
                reach = edge + ring * u
                start = list(
                    self._add(
                        self._add(centre, h1, math.cos(a) * reach),
                        h2,
                        math.sin(a) * reach,
                    )
                )
                start[ui] = top
                gone = 0.0
                while True:
                    hit = self.rays.cast(self._add(start, down, gone), down, far - gone)
                    if hit is None:
                        break
                    gone += hit.distance
                    # A top, met from above.
                    if MathUtils.dot_product(hit.normal, up) > 0.0:
                        floor = top - gone if floor is None else min(floor, top - gone)
                    gone += 1.0e-3 * u  # on from a millimetre past it
        return (float(low[ui]), False) if floor is None else (floor, True)

    # ---- the outer side ------------------------------------------------------------------------------------------

    def outer(self, target: Bounds, delivery: Optional[Bounds] = None) -> Vector:
        """The level way out of the delivery at *target*: from the delivery's middle toward the target's. Where the
        target stands over the delivery's middle (it is the body of it, so no side of it is outer), or there is no
        delivery, the second level axis (``+Z`` where ``Y`` is up, ``-Y`` where ``Z`` is: a model's front)."""
        centre = self._centre(target)
        middle = self._centre(delivery) if delivery is not None else centre
        low, high = target
        over = all(
            float(low[i]) <= middle[i] <= float(high[i])
            for i in range(3)
            if i != self.up
        )
        way = self._flat(self._sub(centre, middle))
        if over or MathUtils.get_magnitude(way) < 1e-9:
            return self._frame()[2]
        return way

    # ---- the purposes --------------------------------------------------------------------------------------------

    def stand(
        self,
        target: Bounds,
        delivery: Optional[Bounds] = None,
        floor: Optional[float] = None,
        owners: Iterable[Any] = (),
        side: Optional[Sequence[float]] = None,
        near: Optional[Sequence[float]] = None,
        also: Sequence[Tuple[Bounds, Iterable[Any]]] = (),
        keep_out: Sequence[Bounds] = (),
        field_of_view: float = 0.0,
        stand_off: Optional[float] = None,
        eye_height: Optional[float] = None,
        body_radius: Optional[float] = None,
        body_height: float = 0.0,
        grounded: Optional[bool] = None,
    ) -> Tuple[Vector, Vector]:
        """Where a person stands to see *target*, and the point they face (level): ``(feet, aim)``.

        Parameters:
            target: What they look at: its bounds.
            delivery: The whole delivery's bounds: its outer side at *target* breaks ties (:meth:`outer`).
            floor: The floor's height on the up axis. ``None``: what :meth:`floor_near` finds, else the delivery's
                lowest point.
            owners: The ray set's owners that are the target.
            side: The way, from the target, of the side to stand on (a sheet's "the front of the cabinet"): it
                outranks sight. ``None``: any side.
            near: Where they stood before: the nearer of spots alike wins.
            also: Other parts they act on from there, each ``(bounds, owners)``: a spot that sees them too wins.
            keep_out: Boxes where things come to stand while they are there (a door's swing): no spot within one, or
                a body's width of it.
            field_of_view: Their view's vertical angle in degrees: they stand back for the part to fill no more than
                :attr:`FILL` of it, up to :attr:`MOST_BACK` (0: at the stand-off).
            stand_off, eye_height, body_radius: Metres (defaults :attr:`STAND_OFF`, :attr:`EYE_HEIGHT`,
                :attr:`BODY_RADIUS`).
            body_height: The height they need clear, metres; 0: up to their eyes and a body's radius over.
            grounded: Whether a spot needs ground under it at the floor's height. ``None``: where rays find floor
                around the target within :attr:`FLOOR_RISE` of *floor*.
        """
        feet, aim, _found = self._stand(
            target,
            delivery,
            floor,
            owners,
            side,
            near,
            also,
            keep_out,
            field_of_view,
            stand_off,
            eye_height,
            body_radius,
            body_height,
            grounded,
        )
        return feet, aim

    def view(
        self,
        target: Bounds,
        delivery: Optional[Bounds] = None,
        eyes: Optional[Sequence[float]] = None,
        owners: Iterable[Any] = (),
        field_of_view: Optional[float] = None,
        margin: Optional[float] = None,
    ) -> Tuple[Vector, Vector]:
        """Where a camera frames *target*, and the point it looks at: ``(eye, aim)``.

        Parameters:
            eyes: Where the person's eyes are: of spots that see the part alike, the one on their side. ``None``:
                those of :meth:`stand`'s spot.
            field_of_view: The camera's vertical angle in degrees (default :attr:`VIEW_ANGLE`).
            margin: The room around the part (default :attr:`MARGIN`; 1: it just fits).
        """
        at, aim, _found = self._view(
            target, delivery, eyes, owners, field_of_view, margin
        )
        return at, aim

    def point(
        self,
        target: Bounds,
        delivery: Optional[Bounds] = None,
        eyes: Optional[Sequence[float]] = None,
        owners: Iterable[Any] = (),
        gap: Optional[float] = None,
        length: Optional[float] = None,
        travel: float = 0.0,
    ) -> Tuple[Vector, Vector]:
        """Where an arrow's tip rests by *target*, and the point it points at: ``(tip, aim)``. Its body lies behind
        the tip, away from the aim.

        Parameters:
            eyes: Where the person's eyes are. ``None``: those of :meth:`stand`'s spot.
            gap, length: Metres between the tip and the part, and tip to tail (defaults :attr:`GAP`, :attr:`POINTER`).
            travel: How far it bobs away from the part along its axis, in metres: the room it needs is its sweep.
        """
        tip, aim, _found = self._point(
            target, delivery, eyes, owners, gap, length, travel
        )
        return tip, aim

    # ---- the searches ----------------------------------------------------------------------------------------------

    def _stand(
        self,
        target,
        delivery=None,
        floor=None,
        owners=(),
        side=None,
        near=None,
        also=(),
        keep_out=(),
        field_of_view=0.0,
        stand_off=None,
        eye_height=None,
        body_radius=None,
        body_height=0.0,
        grounded=None,
    ):
        owners = set(owners)
        u, ui = self.unit, self.up
        up, _, _ = self._frame()
        centre, extents = self._centre(target), self._extents(target)
        stand_off = (self.STAND_OFF if stand_off is None else stand_off) * u
        eye = (self.EYE_HEIGHT if eye_height is None else eye_height) * u
        radius = max(0.01, self.BODY_RADIUS if body_radius is None else body_radius) * u
        top = (
            max(
                self.KNEES * u + 2.0 * radius,
                body_height * u if body_height > 0 else eye + radius,
            )
            - radius
        )
        asked = self.rays is not None and (floor is None or grounded is None)
        found_floor, found = self.floor_near(target) if asked else (None, False)
        if floor is None:
            floor = found_floor if found else float((delivery or target)[0][ui])
        if grounded is None:
            grounded = found and abs(found_floor - floor) <= self.FLOOR_RISE * u
        grounded = grounded and self.rays is not None
        if field_of_view > 0:
            whole = extents[ui] / math.tan(
                math.radians(min(max(field_of_view, 1.0), 179.0)) * 0.5 * self.FILL
            )
            stand_off = max(stand_off, min(whole, self.MOST_BACK * u))
        named = self._flat(side) if side is not None else None
        named = (
            named
            if named is not None and MathUtils.get_magnitude(named) > 0.5
            else None
        )
        out = self.outer(target, delivery) if delivery is not None else None
        lead = named or out
        first = lead or self._frame()[1]
        base = list(centre)
        base[ui] = floor
        best = pose = None
        for ring in range(3):
            for k in range(24):
                way = self._turn(first, math.radians(15.0 * k), 0.0)
                reach = (
                    sum(abs(extents[i] * way[i]) for i in range(3))
                    + stand_off
                    + ring * 0.5 * u
                )
                feet = list(self._add(base, way, reach))
                if grounded:
                    origin = self._add(feet, up, self.FLOOR_RISE * u + eye)
                    hit = self.rays.cast(
                        origin,
                        self._scale(up, -1.0),
                        eye + self.FLOOR_RISE * u + 0.3 * u,
                    )
                    if hit is None:
                        continue
                    ground = origin[ui] - hit.distance
                    if (
                        ground < floor - 0.05 * u
                        or ground > floor + self.FLOOR_RISE * u
                    ):
                        continue
                    feet[ui] = ground
                feet = tuple(feet)
                eyes = self._add(feet, up, eye)
                if self.rays is not None and self.rays.occupied_capsule(
                    self._add(feet, up, self.KNEES * u + radius),
                    self._add(feet, up, top),
                    radius,
                ):
                    continue
                if self._within(
                    keep_out,
                    feet,
                    feet[ui] + body_height * u if body_height > 0 else eyes[ui],
                    radius,
                ):
                    continue
                seen = self.seen(eyes, target, owners)
                if seen <= 0.0:
                    continue
                others = (
                    sum(self.seen(eyes, b, o) for b, o in also) / len(also)
                    if also
                    else 0.0
                )
                score = seen * 10.0 + others * 10.0 - ring
                if near is not None:
                    score -= 0.5 * MathUtils.get_magnitude(self._sub(feet, near)) / u
                if named:
                    score += 6.0 * MathUtils.dot_product(way, named)
                elif out:
                    score += MathUtils.dot_product(way, out)
                if best is None or score > best + self.EPS:
                    aim = list(centre)
                    aim[ui] = feet[ui]
                    best, pose = score, (feet, tuple(aim))
        if pose is not None:
            return pose[0], pose[1], True
        way = lead or self.outer(target, None)
        feet = list(
            self._add(
                base, way, sum(abs(extents[i] * way[i]) for i in range(3)) + stand_off
            )
        )
        aim = list(centre)
        aim[ui] = floor
        return tuple(feet), tuple(aim), False

    def _within(self, boxes, feet, top, radius) -> bool:
        """Whether a person standing at *feet* (the top of their head at *top*) is within one of *boxes*, or *radius*
        of it."""
        ui = self.up
        for low, high in boxes or ():
            if (
                float(high[ui]) < feet[ui] + self.KNEES * self.unit
                or float(low[ui]) > top
            ):
                continue  # under their knees, or over their head
            if all(
                float(low[i]) - radius < feet[i] < float(high[i]) + radius
                for i in range(3)
                if i != ui
            ):
                return True
        return False

    def _eyes(self, target, delivery, owners):
        """The eyes of :meth:`stand`'s spot for *target*."""
        feet, _aim = self.stand(target, delivery, owners=owners)
        return self._add(feet, self._frame()[0], self.EYE_HEIGHT * self.unit)

    def _view(
        self,
        target,
        delivery=None,
        eyes=None,
        owners=(),
        field_of_view=None,
        margin=None,
    ):
        owners = set(owners)
        u = self.unit
        _, _, h2 = self._frame()
        fov = self.VIEW_ANGLE if field_of_view is None else field_of_view
        margin = self.MARGIN if margin is None else margin
        centre, extents = self._centre(target), self._extents(target)
        radius = max(MathUtils.get_magnitude(extents), self.LEAST_FRAMED * u)
        distance = (
            radius * margin / math.sin(math.radians(min(max(fov, 1.0), 179.0)) * 0.5)
        )
        if eyes is None:
            eyes = self._eyes(target, delivery, owners)
        to = self._sub(eyes, centre)
        to_eyes = (
            self._unit(to)
            if MathUtils.get_magnitude(to) > 1e-4 * u
            else self._scale(h2, -1.0)
        )
        pose = (self._add(centre, to_eyes, distance), centre)
        lead = self._flat(to_eyes)
        first = lead if MathUtils.get_magnitude(lead) > 0.5 else self._frame()[1]
        axis = self._face(extents)
        best = None
        for elevation in (10.0, 35.0, -10.0, 60.0):
            for k in range(18):
                way = self._turn(first, math.radians(20.0 * k), math.radians(elevation))
                at = self._add(centre, way, distance)
                if self.rays is not None and self.rays.occupied_sphere(
                    at, self.FREE * u
                ):
                    continue
                seen = self.seen(at, target, owners, fine=True)
                if seen <= 0.0:
                    continue
                # How much of it is seen decides; among spots that see it alike, a flat part's face, the eyes' side, the level.
                rest = (
                    (self.FACE_ON * abs(way[axis]) if axis is not None else 0.0)
                    + 0.4 * MathUtils.dot_product(way, to_eyes)
                    - abs(elevation) / 60.0
                )
                if (
                    best is None
                    or seen > best[0] + self.EPS
                    or (seen > best[0] - self.EPS and rest > best[1] + self.EPS)
                ):
                    best, pose = (seen, rest), (at, centre)
        return pose[0], pose[1], best is not None

    @staticmethod
    def _face(extents) -> Optional[int]:
        """The axis a flat part's face looks along (its thinnest), or ``None`` where it is no flat part: flat, its
        thinnest extent under three quarters of the next (a rod's two thin sides are no face) and under half its face's
        size (the mean of the other two)."""
        order = sorted(range(3), key=lambda i: (extents[i], i))
        thin, mid, large = (extents[i] for i in order)
        return order[0] if thin < 0.75 * mid and thin < 0.25 * (mid + large) else None

    def _point(
        self,
        target,
        delivery=None,
        eyes=None,
        owners=(),
        gap=None,
        length=None,
        travel=0.0,
    ):
        owners = set(owners)
        u, ui = self.unit, self.up
        up, _, h2 = self._frame()
        gap = (self.GAP if gap is None else gap) * u
        length = (self.POINTER if length is None else length) * u
        centre, extents = self._centre(target), self._extents(target)
        if eyes is None:
            eyes = self._eyes(target, delivery, owners)
        level = self._flat(self._sub(eyes, centre))
        to_eyes = (
            level if MathUtils.get_magnitude(level) > 0.5 else self._scale(h2, -1.0)
        )
        side = MathUtils.cross_product(up, to_eyes)
        # Ways from the world's up and the eyes alone can all be blocked (a button on a door swung open sideways):
        # the way out of what the part is mounted on is one more, and the one taken where none is free.
        mount = self._mount(target, owners, eyes)
        face = mount or to_eyes
        neg = lambda v: self._scale(v, -1.0)  # noqa: E731
        ways = [
            up,
            self._unit(self._add(up, to_eyes)),
            to_eyes,
            self._unit(self._add(to_eyes, side)),
            self._unit(self._sub(to_eyes, side)),
            self._unit(self._add(up, side)),
            self._unit(self._sub(up, side)),
            side,
            neg(side),
            self._unit(self._add(face, up)),
            face,
        ]

        def tip_of(way):
            return self._add(
                centre, way, sum(abs(extents[i] * way[i]) for i in range(3)) + gap
            )

        pose = (
            (tip_of(face), centre)
            if mount
            else (self._add(centre, up, extents[ui] + gap), centre)
        )
        sight = self._unit(self._sub(centre, eyes))
        end_on = 1.0 - math.cos(math.radians(self.END_ON))
        best = None
        for i, way in enumerate(ways):
            if MathUtils.dot_product(way, way) < 0.5:
                continue  # a blend of opposites
            tip = tip_of(way)
            tail = self._add(tip, way, length)
            # Room for it to turn about its axis (a flat one faces the eyes), over its whole sweep.
            if self.rays is not None and self.rays.occupied_capsule(
                self._add(tip, way, 0.01 * u),
                self._add(tail, way, max(0.0, travel) * u),
                max(0.005 * u, length * 0.3),
            ):
                continue
            seen = (1 if self.clear(eyes, tip) else 0) + (
                1 if self.clear(eyes, tail) else 0
            )
            if seen == 0:
                continue
            # Read best across the line of sight; within END_ON of it, it is no arrow at all.
            across = 1.0 - abs(MathUtils.dot_product(way, sight))
            score = (
                seen * 5.0
                + 3.0 * MathUtils.dot_product(way, up)
                + 3.0 * across
                - (8.0 if across < end_on else 0.0)
                - (0.0 if self.clear(tip, centre, owners) else 5.0)
                - i * 0.01
            )
            if best is None or score > best + self.EPS:
                best, pose = score, (tip, centre)
        return pose[0], pose[1], best is not None

    def _mount(self, target, owners, eyes) -> Optional[Vector]:
        """The way out of what *target* is mounted on, toward the eyes: the face of the first thing a ray from the
        eyes' side meets at or behind it, its own owners passed over; ``None`` where nothing is behind it."""
        if self.rays is None:
            return None
        u = self.unit
        centre, extents = self._centre(target), self._extents(target)
        toward = self._sub(centre, eyes)
        if MathUtils.get_magnitude(toward) < 1e-4 * u:
            return None
        toward = self._unit(toward)
        depth = MathUtils.get_magnitude(extents)
        far = depth + 0.5 * u
        origin = self._add(centre, toward, -far)
        skip = set(owners)
        while True:
            hit = self.rays.cast(origin, toward, far + depth + 0.25 * u, skip=skip)
            if hit is None:
                return None
            if (
                hit.distance < far - depth
            ):  # in front of the part: in the way, not what it sits on
                skip.add(self.rays.owners[hit.mesh])
                continue
            normal = tuple(hit.normal)
            return self._unit(
                normal
                if MathUtils.dot_product(normal, toward) < 0
                else self._scale(normal, -1.0)
            )


class BoxRays(object):
    """A scene of axis-aligned boxes, as :class:`SpotRays`: the host-free scene :class:`SpotPlacement`'s tests and
    golden cases run in. A ray starting inside a box passes it, as a physics engine's does; room is exact for a
    sphere, and for a capsule read as a sphere at each of :meth:`SpotPlacement.stations`."""

    class Hit(object):
        def __init__(self, distance: float, mesh: int, normal: Vector):
            self.distance = distance
            self.mesh = mesh
            self.normal = normal

    def __init__(self):
        self.owners: List[Any] = []
        self._boxes: List[Tuple[List[float], List[float]]] = []

    def add(self, owner: Any, low: Sequence[float], high: Sequence[float]) -> None:
        """Add a box for *owner*: its lowest and highest corners."""
        self.owners.append(owner)
        self._boxes.append(([float(c) for c in low], [float(c) for c in high]))

    def cast(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        far: float = 1.0e7,
        skip: Any = None,
    ) -> Optional["BoxRays.Hit"]:
        best = None
        for n, (low, high) in enumerate(self._boxes):
            if skip and self.owners[n] in skip:
                continue
            if all(low[a] < origin[a] < high[a] for a in range(3)):
                continue  # starts inside it
            enter, leave, axis = 0.0, far, -1
            for a in range(3):
                d = direction[a] if abs(direction[a]) > 1e-12 else 1e-12
                t1, t2 = (low[a] - origin[a]) / d, (high[a] - origin[a]) / d
                if min(t1, t2) > enter:
                    enter, axis = min(t1, t2), a
                leave = min(leave, max(t1, t2))
            if (
                leave >= enter
                and enter <= far
                and (best is None or enter < best.distance)
            ):
                normal = [0.0, 0.0, 0.0]
                if axis >= 0:
                    normal[axis] = -1.0 if direction[axis] > 0 else 1.0
                best = BoxRays.Hit(enter, n, tuple(normal))
        return best

    def _gap(self, point: Sequence[float], box) -> float:
        low, high = box
        return math.sqrt(
            sum(max(low[a] - point[a], 0.0, point[a] - high[a]) ** 2 for a in range(3))
        )

    def occupied_sphere(self, at: Sequence[float], radius: float) -> bool:
        return any(self._gap(at, box) < radius for box in self._boxes)

    def occupied_capsule(
        self, a: Sequence[float], b: Sequence[float], radius: float
    ) -> bool:
        points = SpotPlacement.stations(a, b, radius)
        return any(self._gap(p, box) < radius for box in self._boxes for p in points)


class SpotPlacementConformance(object):
    """The golden cases every port of :class:`SpotPlacement` is held to (``ptk.Conformance.cases("spot_placement")``).

    Each case is a scene of boxes (:class:`BoxRays`) and one call: its purpose, the target, the settings, and what the
    reference gives (``position``, ``aim``, ``found``). The named scenes each pin one rule; ``per_kind`` random rooms per
    purpose (seeded) pin the search as a whole. Generated from the reference at test time, so a port is never held to a
    stale copy.
    """

    #: The largest difference a port may show, in the case's units: the search is closed arithmetic over the same
    #: candidates (measured exact to rounding).
    TOLERANCE: Dict[str, float] = {"position": 1.0e-6, "aim": 1.0e-6}

    # A cabinet's door (2 m tall, on the floor) with a latch on its front (+Z); a panel behind the door carries dials on
    # its back face (-Z). A floor slab under it all.
    FLOOR = ("FLOOR", (-6.0, -0.1, -6.0), (6.0, 0.0, 6.0))
    DOOR = ("DOOR", (-0.5, 0.0, -0.03), (0.5, 2.0, 0.03))
    LATCH = ("LATCH", (0.15, 1.05, 0.03), (0.25, 1.15, 0.07))
    PANEL = ("PANEL", (-0.4, 1.2, -0.43), (0.4, 1.8, -0.4))
    DIALS = ("DIALS", (-0.05, 1.45, -0.46), (0.05, 1.55, -0.43))

    @classmethod
    def scenes(cls) -> List[Dict[str, Any]]:
        """The named cases' scenes and calls (no answers)."""

        def box(row):
            return {"owner": row[0], "low": list(row[1]), "high": list(row[2])}

        def bounds(row):
            return {"low": list(row[1]), "high": list(row[2])}

        door, latch, panel, dials, floor = (
            cls.DOOR,
            cls.LATCH,
            cls.PANEL,
            cls.DIALS,
            cls.FLOOR,
        )
        grille = [
            ("GRILLE_%d" % i, (-0.3 + 0.1 * i, 0.9, 0.2), (-0.26 + 0.1 * i, 1.3, 0.22))
            for i in range(7)
        ]
        fan = ("FAN", (-0.2, 1.0, 0.0), (0.2, 1.2, 0.1))
        button = ("BUTTON", (-0.02, 1.38, 0.03), (0.02, 1.42, 0.05))
        shelf = ("SHELF", (-0.5, 1.17, -0.1), (0.5, 1.2, 0.4))
        gauge = (
            "GAUGE",
            (-0.54, 1.3, -0.02),
            (-0.5, 1.4, 0.02),
        )  # on the door's -X edge: hidden from +X
        table = ("TABLE", (-0.5, 0.9, -0.5), (0.5, 1.0, 0.5))
        knob = ("KNOB", (-0.02, 1.0, -0.02), (0.02, 1.04, 0.02))
        lid = (
            "LID",
            (-0.5, 1.25, -0.5),
            (0.5, 1.28, 0.1),
        )  # over the knob, short of the line from the eyes
        swing = {
            "low": [-0.6, 0.0, 0.03],
            "high": [0.6, 2.0, 2.0],
        }  # a cart parked there in a later step
        handle = (
            "HANDLE",
            (0.15, 1.45, 0.03),
            (0.25, 1.55, 0.07),
        )  # a metre over its top is over the ceiling
        ceiling = {
            "owner": "CEILING",
            "low": [-6.0, 2.3, -6.0],
            "high": [6.0, 2.4, 6.0],
        }
        return [
            {
                "name": "a picture stands where its part is seen, never behind the panel it is on",
                "purpose": "view",
                "boxes": [box(door), box(panel), box(dials)],
                "target": bounds(dials),
                "owners": ["DIALS"],
                "delivery": {"low": [-3.0, 0.0, -6.0], "high": [3.0, 2.5, -0.6]},
            },
            {
                "name": "a viewpoint stands on the floor where its eyes see the part",
                "purpose": "stand",
                "boxes": [
                    box(floor),
                    box(door),
                    box(latch),
                    {"owner": "WALL", "low": [-3.0, 0.0, 1.0], "high": [3.0, 3.0, 1.1]},
                ],
                "target": bounds(latch),
                "owners": ["LATCH"],
                "delivery": {"low": [-3.0, 0.0, -3.0], "high": [3.0, 2.5, 0.0]},
            },
            {
                "name": "a named side outranks sight, and a viewpoint keeps out of a door's swing",
                "purpose": "stand",
                "boxes": [box(floor), box(door), box(latch)],
                "target": bounds(latch),
                "owners": ["LATCH"],
                "side": [0.0, 0.0, 1.0],
                "keep_out": [swing],
            },
            {
                "name": "a viewpoint stands back for a tall part to fit its view, and near where it stood",
                "purpose": "stand",
                "boxes": [box(floor), box(door)],
                "target": bounds(door),
                "owners": ["DOOR"],
                "field_of_view": 60.0,
                "near": [2.0, 0.0, 2.0],
            },
            {
                "name": "a viewpoint that sees the other parts too wins",
                "purpose": "stand",
                "boxes": [box(floor), box(door), box(latch), box(gauge)],
                "target": bounds(latch),
                "owners": ["LATCH"],
                "also": [
                    {"low": list(gauge[1]), "high": list(gauge[2]), "owners": ["GAUGE"]}
                ],
            },
            {
                "name": "a viewpoint never stands under a shelf it has no room under",
                "purpose": "stand",
                "boxes": [
                    box(floor),
                    box(door),
                    box(latch),
                    {
                        "owner": "TABLE",
                        "low": [-1.5, 0.9, 0.5],
                        "high": [1.5, 1.0, 2.5],
                    },
                ],
                "target": bounds(latch),
                "owners": ["LATCH"],
                "side": [0.0, 0.0, 1.0],
            },
            {
                "name": "a viewpoint stands on the floor under a ceiling, never on it",
                "purpose": "stand",
                "boxes": [box(floor), box(door), box(handle), ceiling],
                "target": bounds(handle),
                "owners": ["HANDLE"],
            },
            {
                "name": "a picture of a part behind a grille stands where it is seen whole",
                "purpose": "view",
                "boxes": [box(floor), box(fan)] + [box(g) for g in grille],
                "target": bounds(fan),
                "owners": ["FAN"],
                "eyes": [0.0, 1.6, 2.0],
            },
            {
                "name": "a picture shows a flat part face-on, not from the eyes' side along it",
                "purpose": "view",
                "boxes": [
                    {
                        "owner": "WALL",
                        "low": [-1.0, 0.0, -0.1],
                        "high": [1.0, 2.5, 0.0],
                    },
                    {
                        "owner": "PLATE",
                        "low": [-0.15, 1.2, 0.0],
                        "high": [0.15, 1.5, 0.01],
                    },
                ],
                "target": {"low": [-0.15, 1.2, 0.0], "high": [0.15, 1.5, 0.01]},
                "owners": ["PLATE"],
                "eyes": [1.5, 1.6, 0.8],
            },
            {
                "name": "an arrow stands clear of a shelf and never end-on to the eyes",
                "purpose": "point",
                "boxes": [box(door), box(latch), box(shelf)],
                "target": bounds(latch),
                "owners": ["LATCH"],
                "eyes": [0.2, 1.6, 2.0],
            },
            {
                "name": "where nothing is free, an arrow comes straight out of what its part is mounted on",
                "purpose": "point",
                "boxes": [
                    box(door),
                    box(button),
                    {
                        "owner": "HOOD",
                        "low": [-0.5, 1.43, -0.3],
                        "high": [0.5, 1.5, 0.4],
                    },
                ],
                "target": bounds(button),
                "owners": ["BUTTON"],
                "eyes": [0.0, 1.0, 1.5],
            },
            {
                "name": "an arrow points down at its part from above where that is free",
                "purpose": "point",
                "boxes": [box(table), box(knob), box(lid)],
                "target": bounds(knob),
                "owners": ["KNOB"],
                "eyes": [0.0, 1.6, 1.5],
            },
            {
                "name": "an arrow's bob needs its whole sweep free",
                "purpose": "point",
                "boxes": [box(table), box(knob), box(lid)],
                "target": bounds(knob),
                "owners": ["KNOB"],
                "eyes": [0.0, 1.6, 1.5],
                "travel": 0.1,
            },
            {
                "name": "with no rays a first pose is the plain one",
                "purpose": "stand",
                "boxes": [],
                "target": {"low": [-0.1, 1.0, 0.4], "high": [0.1, 1.2, 0.6]},
                "delivery": {"low": [-1.0, 0.0, -1.0], "high": [1.0, 2.5, 1.0]},
            },
            {
                "name": "a Z-up centimetre scene places as a Y-up metre one",
                "purpose": "view",
                "up": 2,
                "unit": 100.0,
                "boxes": [
                    {
                        "owner": "DOOR",
                        "low": [-50.0, -3.0, 0.0],
                        "high": [50.0, 3.0, 200.0],
                    },
                    {
                        "owner": "PANEL",
                        "low": [-40.0, 40.0, 120.0],
                        "high": [40.0, 43.0, 180.0],
                    },
                    {
                        "owner": "DIALS",
                        "low": [-5.0, 43.0, 145.0],
                        "high": [5.0, 46.0, 155.0],
                    },
                ],
                "target": {"low": [-5.0, 43.0, 145.0], "high": [5.0, 46.0, 155.0]},
                "owners": ["DIALS"],
                "delivery": {"low": [-300.0, 60.0, 0.0], "high": [300.0, 600.0, 250.0]},
            },
        ]

    @classmethod
    def _room(cls, rng: random.Random, purpose: str) -> Dict[str, Any]:
        """A random room: a floor, a cabinet, a part on one of its faces and a few things around it."""
        boxes = [{"owner": "FLOOR", "low": [-6.0, -0.1, -6.0], "high": [6.0, 0.0, 6.0]}]
        w, h, d = rng.uniform(0.6, 1.6), rng.uniform(0.8, 2.2), rng.uniform(0.4, 1.0)
        boxes.append(
            {
                "owner": "CABINET",
                "low": [-w / 2, 0.0, -d / 2],
                "high": [w / 2, h, d / 2],
            }
        )
        s = rng.uniform(0.04, 0.2)
        x, y = rng.uniform(-w / 2 + s, w / 2 - s), rng.uniform(0.4, h - s)
        face = rng.choice((1.0, -1.0))
        z0 = face * d / 2
        part = {
            "low": [x - s / 2, y - s / 2, min(z0, z0 + face * s / 2)],
            "high": [x + s / 2, y + s / 2, max(z0, z0 + face * s / 2)],
        }
        boxes.append(dict(part, owner="PART"))
        for n in range(rng.randint(1, 4)):
            cx, cz = rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5)
            if abs(cx) < w / 2 + 0.3 and abs(cz) < d / 2 + 0.3:
                continue
            bw, bh, bd = (
                rng.uniform(0.1, 1.2),
                rng.uniform(0.3, 2.4),
                rng.uniform(0.1, 1.2),
            )
            boxes.append(
                {
                    "owner": "THING_%d" % n,
                    "low": [cx - bw / 2, 0.0, cz - bd / 2],
                    "high": [cx + bw / 2, bh, cz + bd / 2],
                }
            )
        case = {
            "name": "room %s" % purpose,
            "purpose": purpose,
            "boxes": boxes,
            "target": {"low": part["low"], "high": part["high"]},
            "owners": ["PART"],
        }
        if purpose != "stand":
            case["eyes"] = [rng.uniform(-1.0, 1.0), 1.6, face * rng.uniform(1.2, 2.5)]
        return case

    @classmethod
    def answer(cls, case: Dict[str, Any]) -> Dict[str, Any]:
        """What the reference gives for *case*: ``{"position", "aim", "found"}``."""
        rays = None
        if case["boxes"]:
            rays = BoxRays()
            for b in case["boxes"]:
                rays.add(b["owner"], b["low"], b["high"])
        place = SpotPlacement(
            rays=rays, up=case.get("up", 1), unit=case.get("unit", 1.0)
        )

        def bounds(b):
            return None if b is None else (tuple(b["low"]), tuple(b["high"]))

        target, delivery = bounds(case["target"]), bounds(case.get("delivery"))
        owners = case.get("owners", [])
        if case["purpose"] == "stand":
            position, aim, found = place._stand(
                target,
                delivery,
                case.get("floor"),
                owners,
                case.get("side"),
                case.get("near"),
                [(bounds(a), a["owners"]) for a in case.get("also", [])],
                [bounds(k) for k in case.get("keep_out", [])],
                case.get("field_of_view", 0.0),
                grounded=case.get("grounded"),
            )
        elif case["purpose"] == "view":
            position, aim, found = place._view(
                target, delivery, case.get("eyes"), owners
            )
        else:
            position, aim, found = place._point(
                target,
                delivery,
                case.get("eyes"),
                owners,
                travel=case.get("travel", 0.0),
            )
        return {"position": list(position), "aim": list(aim), "found": found}

    @classmethod
    def cases(cls, seed: int = 0, per_kind: int = 4) -> Dict[str, Any]:
        """The conformance document: ``{"tolerance", "cases"}``, each case its scene, call and ``expected`` answer."""
        rng = random.Random(seed)
        cases = cls.scenes()
        for purpose in ("stand", "view", "point"):
            for n in range(per_kind):
                room = cls._room(rng, purpose)
                room["name"] = "%s %d" % (room["name"], n)
                cases.append(room)
        for case in cases:
            case["expected"] = cls.answer(case)
        return {"tolerance": dict(cls.TOLERANCE), "cases": cases}
