# !/usr/bin/python
# coding=utf-8
"""Where a lightmap bake's reflection probe stands, and the room it projects onto.

The geometry half of a DCC ``LightmapBaker.bake_probe`` (mayatk's and
blendertk's), host-free: it has to hold in every scene a generic baker meets --
a furnished room, a hall, a courtyard, a car on a ground plane, a machine in a
warehouse. The first rule (the middle of what is baked, at half the room's
height, inside the room its rays found) held in the one room it was measured
in and put a probe where it could not see the room in each of the others:

* **Inside something.** The middle of what is baked is the room's middle in a
  room, and inside whatever stands there anywhere else -- the car on its
  ground plane, the statue, the column. A probe there renders the inside of a
  shell: black, lent to every reflection in the scene.
* **In an open cupboard.** Over what moves at eye level can be inside the
  cabinet those parts live in; with its front open the room's lights show
  through, so the point read as lit open air -- a production probe was
  two-thirds cabinet interior, lent dark to every reflection (2026-10-15).
* **Against something.** A surface a few centimetres from the camera fills
  half the panorama, and every reflection the probe lends shows it.
* **Too high.** Half a room's height is eye level in a room and five metres up
  in a hall; with no ceiling (outdoors) it was half the tallest building's.
* **All or nothing.** One open side (a courtyard, a ground with no walls)
  read the whole room as open, so its walls and floor reflected as if
  infinitely far.

So the probe is placed in steps that each guard one of those, all measured by
ray casts against what the probe's render sees (:class:`ProbeRays`) -- the
visible baked meshes, and the unbaked ones that surround the bake (a sky dome,
a ground nobody baked); what moves is see-through to the render and to the
rays:

1. **Open air** near the middle of the bake (:meth:`ProbePlacement._settle`):
   a point that sees the inside of a shell from every side, with no light in
   view, is in a solid, and is moved out by the shortest way; one inside a
   piece of furniture's bounds that fills half its view is in that piece,
   whatever shows through its open side, and leaves by the opening.
2. **The room** around it (:meth:`ProbePlacement.faces`), face by face: an open
   face is open on its own, and the rest still project.
3. **Over what matters**: what moves in the room, else the room's middle,
   else -- an open scene -- the baked objects that are not its ground.
4. **At eye level**: half the room's height, at most
   :attr:`ProbePlacement.MAX_HEIGHT_M` above its floor.
5. **Clear of every surface** by :attr:`ProbePlacement.CLEARANCE_M`.

**A host supplies the scene** through two short adapters, injected, never by
subclassing: a :class:`ProbeHost` (its node names, rays, bounds, lights,
units and up axis) and the :class:`ProbeMesh` its :class:`ProbeRays`
intersect through (the nearest hit on one mesh, a face's normal). mayatk's and
blendertk's ``ProbePlacement`` are this class bound to their host's adapter
class (:attr:`ProbePlacement.HOST_CLASS`, one adapter built per placement).
Everything inside is in the host's internal linear unit; positions cross the boundary in scene units. Nodes are whatever
the host names them by (a DAG path, an object name), compared by equality.
"""

import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

Vec = Tuple[float, float, float]
Box = List[List[float]]


class ProbeHit(NamedTuple):
    """A ray's nearest hit: how far, where, the face's world normal, and the
    index of the mesh it struck in its :class:`ProbeRays` (``owners``)."""

    distance: float
    point: Vec
    normal: Vec
    mesh: int


class ProbeMesh(Protocol):
    """How a host intersects its meshes: what :class:`ProbeRays` asks of each
    mesh's *payload* (the host's handle on it, given to :meth:`ProbeRays.add`)."""

    def intersect(
        self,
        payload: Any,
        origin: Sequence[float],
        direction: Sequence[float],
        far: float,
    ) -> Optional[Tuple[float, Vec, Any]]:
        """The nearest hit on one mesh: ``(distance, point, face)``, or
        ``None``. *face* is what :meth:`normal` reads the winner's normal
        from; only the nearest hit of a cast pays for one."""

    def normal(self, payload: Any, face: Any) -> Vec:
        """The world normal of *face* on the mesh *payload* names."""


class ProbeRays:
    """The nearest hit of a ray among a set of meshes, in world space.

    Each ray is tested against every mesh's world bounds first (numpy, all at
    once), and only the meshes it can reach are intersected, nearest bounds
    first, stopping once the next one starts past the best hit: a production
    room is hundreds of meshes, and a probe is placed with a few thousand rays.

    A host :meth:`add` s each mesh (its owner node, world bounds and whatever
    it intersects the mesh through) and supplies the intersection itself, the
    injected *mesh* adapter (:class:`ProbeMesh`).

    Parameters:
        mesh: The host's intersection (:class:`ProbeMesh`).
    """

    def __init__(self, mesh: Optional[ProbeMesh] = None):
        #: The host's intersection (:class:`ProbeMesh`).
        self.mesh = mesh
        #: Each intersectable mesh's node, by index.
        self.owners: List[Any] = []
        self._payloads: List[Any] = []
        self._bounds: List[Tuple[Sequence[float], Sequence[float]]] = []
        self._lo = None
        self._hi = None

    def add(
        self, owner: Any, lo: Sequence[float], hi: Sequence[float], payload: Any
    ) -> None:
        """Add one intersectable mesh: its *owner* node, its world bounds and
        the host's handle to intersect it by (:meth:`_intersect`)."""
        self.owners.append(owner)
        self._payloads.append(payload)
        self._bounds.append(([float(c) for c in lo], [float(c) for c in hi]))
        self._lo = self._hi = None

    def __bool__(self) -> bool:
        return bool(self._payloads)

    def _intersect(
        self,
        payload: Any,
        origin: Sequence[float],
        direction: Sequence[float],
        far: float,
    ) -> Optional[Tuple[float, Vec, Any]]:
        """The host's nearest hit on one mesh (:meth:`ProbeMesh.intersect`)."""
        return self.mesh.intersect(payload, origin, direction, far)

    def _normal(self, payload: Any, face: Any) -> Vec:
        """The world normal of *face* (:meth:`ProbeMesh.normal`)."""
        return self.mesh.normal(payload, face)

    def cast(
        self, origin: Sequence[float], direction: Sequence[float], far: float = 1.0e7
    ) -> Optional[ProbeHit]:
        """The nearest hit along *direction* (unit) from *origin* within *far*."""
        import numpy as np

        if self._lo is None:
            self._lo = np.array([b[0] for b in self._bounds], dtype=float).reshape(
                -1, 3
            )
            self._hi = np.array([b[1] for b in self._bounds], dtype=float).reshape(
                -1, 3
            )
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        # A zero component would divide to inf - inf = nan against a box the
        # ray grazes; a tiny one keeps the slab test's arithmetic finite.
        safe = np.where(np.abs(d) < 1.0e-12, 1.0e-12, d)
        t1 = (self._lo - o) / safe
        t2 = (self._hi - o) / safe
        enter = np.maximum(np.minimum(t1, t2).max(axis=1), 0.0)
        leave = np.maximum(t1, t2).min(axis=1)
        reach = np.nonzero((leave >= enter) & (enter <= far))[0]
        best = None
        for index in reach[np.argsort(enter[reach])]:
            if best is not None and enter[index] > best[0]:
                break
            hit = self._intersect(self._payloads[index], o, d, far)
            if hit is not None and (best is None or hit[0] < best[0]):
                best = (hit[0], hit[1], int(index), hit[2])
        if best is None:
            return None
        distance, point, index, face = best
        normal = self._normal(self._payloads[index], face)
        return ProbeHit(
            float(distance),
            tuple(float(c) for c in point),
            tuple(float(c) for c in normal),
            index,
        )


@dataclass
class ProbeSite:
    """Where a reflection probe is captured from, and what it projects onto.

    Attributes:
        position: The capture point, in scene units.
        box: ``[[min], [max]]`` its reflections project onto, in scene units,
            each open face pushed :attr:`ProbePlacement.OPEN_FAR_M` (or more)
            away so it reads as distant; ``None`` when every face is open.
        hide: What the capture sees through: the visible unbaked meshes that
            do not surround the bake -- what moves.
        reason: Where the point was taken over, for the log.
        notes: What was corrected on the way (a solid left, a surface cleared).
        open_faces: How many of the box's six faces are open.
    """

    position: List[float]
    box: Optional[Box]
    hide: List[Any] = field(default_factory=list)
    reason: str = ""
    notes: List[str] = field(default_factory=list)
    open_faces: int = 0


class _Look(NamedTuple):
    """What a point sees around it (:meth:`ProbePlacement._look`): whether it
    is shut in, the solid that shuts it in, and its nearest surface -- how
    far, which way is away from it, and whose it is."""

    trapped: bool
    solid: Any
    nearest: float
    away: Vec
    near: Any
    #: Shut in by being inside *solid* -- furniture filling half the view
    #: from within its bounds -- rather than by a closed shell.
    enclosed: bool = False


class ProbeHost(Protocol):
    """What :class:`ProbePlacement` asks of the host scene: a short, fixed
    interface each DCC implements (``MayaProbeHost``, ``BlenderProbeHost``:
    each library's ``ProbePlacement.HOST_CLASS``) and a fake stands in for in a
    plain-Python test. One adapter serves one placement (it may hold what it
    read)."""

    #: The host's internal linear unit in metres (Maya: 0.01 -- centimetres,
    #: whatever the scene's UI unit; Blender: the scene's ``scale_length``).
    internal_m: float
    #: The up axis, as an index (Maya: Y, 1; Blender: Z, 2).
    up: int

    def nodes(self, objects: Sequence[Any]) -> List[Any]:
        """*objects* as the nodes the placement names them by (a DAG path,
        an object name): what :attr:`ProbeSite.hide` lists."""

    def rays(self, nodes: Sequence[Any]) -> "ProbeRays":
        """A :class:`ProbeRays` over the meshes of *nodes*."""

    def bounds(self, nodes: Sequence[Any]) -> Box:
        """``[[min], [max]]`` of *nodes* in world space, internal units."""

    def lights(self) -> Tuple[List[Tuple[Vec, Optional[Any]]], bool]:
        """The lights that light the render: ``[(world point, own mesh node or
        None)]`` for each placed one -- a mesh light is in view when its own
        mesh is -- and whether a sky lights the scene."""

    def to_scene(self, value: float) -> float:
        """An internal length in scene units."""

    def to_internal(self, point: Sequence[float]) -> List[float]:
        """A scene-unit point in internal units."""


class ProbePlacement:
    """Place a lightmap bake's reflection probe (see the module docstring).

    Host-free: the scene comes through the injected *host* adapter
    (:class:`ProbeHost`), else one built from :attr:`HOST_CLASS` -- mayatk's and
    blendertk's ``ProbePlacement`` are this class with theirs set, and nothing
    else.

    Parameters:
        baked: The visible baked mesh nodes -- the room.
        unbaked: The visible mesh nodes no bake marks. Those whose footprint
            covers the bake's (a sky dome, a ground or terrain nobody baked)
            are the room's surroundings: seen, and walls. The rest move: the
            capture sees through them.
        host: The host adapter (:class:`ProbeHost`); ``None`` builds one from
            :attr:`HOST_CLASS`.
    """

    #: The host's adapter class, built once per placement when no *host* is
    #: passed (``HOST_CLASS() -> ProbeHost``): a FACTORY, unlike
    #: ``SceneDataSidecarBase.HOST``'s instance, because an adapter holds what
    #: it read of one placement's scene. ``None``: a host must be passed.
    HOST_CLASS: Optional[Callable[[], ProbeHost]] = None

    #: The cone each face is measured over: rays at these angles (degrees) off
    #: the axis, each ring of twelve, plus the axis itself. Wide, because what
    #: stands between the probe and a wall must not fill it: from 1.1 m above
    #: a production table a 30-degree cone met the table on 23 of its 25 rays.
    WALL_RINGS: Tuple[float, ...] = (10.0, 20.0, 30.0, 40.0, 50.0)
    #: The share of a cone's wall hits a plane needs to count as its wall. A
    #: wall is the FARTHEST surface many rays meet -- the floor past a table,
    #: not the table; a lone ray through a seam or a doorway is not one.
    WALL_SHARE: float = 0.1
    #: How squarely a surface must face a cone's axis (``|normal . axis|``) to
    #: be that face's wall: within 60 degrees. A floor met by a side cone's
    #: lower rays is the floor seen at a slant, not a wall -- counted as one,
    #: a ground plane walled an open scene in at wherever its first ray landed.
    WALL_FACING: float = 0.5
    #: The most the probe stands above its floor: half a room's height is eye
    #: level in a room and five metres up in a hall, and outdoors there is no
    #: half to take.
    MAX_HEIGHT_M: float = 2.0
    #: The least room it keeps from any surface it sees: nearer, a surface
    #: fills a share of the panorama every reflection shows.
    CLEARANCE_M: float = 0.3
    #: Below this median distance to what surrounds it, a point that sees no
    #: light is in a cavity (a vehicle's cabin, a cupboard), not a room.
    CAVITY_M: float = 1.0
    #: How far an open face is pushed: read as distant, while the closed
    #: faces still project.
    OPEN_FAR_M: float = 1000.0
    #: How far around a point its room is measured from as well (metres,
    #: along the floor's axes, from open air the point reaches without
    #: crossing a surface): what stands beside one point -- a column at 30 cm,
    #: the car roof under it, a shelf over it -- fills that point's cones and
    #: is passed over by the next. Each face keeps its farthest wall.
    ROOM_REACH_M: Tuple[float, ...] = (1.0, 2.5)
    #: The most of a probe's view the solid it stepped out of may fill (a
    #: share of :attr:`_AROUND`'s rays): just past a column's surface the
    #: column is a quarter of everything every reflection in the room shows.
    SOLID_SHARE: float = 0.125
    #: The share of a point's view (:attr:`_AROUND`'s rays) one piece of
    #: furniture -- not the room's own shell -- fills, from inside its bounds,
    #: for the point to be inside it: an open cupboard, whatever its opening
    #: shows (:meth:`_look`).
    ENCLOSED_SHARE: float = 0.5
    #: The 26 directions of a cube's faces, edges and corners: what a point
    #: is looked around with.
    _AROUND: Tuple[Vec, ...] = tuple(
        tuple(c / math.sqrt(i * i + j * j + k * k) for c in (i, j, k))
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        if (i, j, k) != (0, 0, 0)
    )

    def __init__(
        self,
        baked: Sequence[Any],
        unbaked: Sequence[Any] = (),
        host: Optional[ProbeHost] = None,
    ):
        if host is None:
            if self.HOST_CLASS is None:
                raise TypeError(
                    "ProbePlacement needs a host adapter (host=, or HOST_CLASS)"
                )
            host = self.HOST_CLASS()
        #: The host adapter every scene read goes through (:class:`ProbeHost`).
        self.host: ProbeHost = host
        self._extents: Dict[Any, Box] = {}
        self._lights: Optional[Tuple[List[Tuple[Vec, Optional[Any]]], bool]] = None
        #: The host's internal linear unit in metres, and its up axis.
        self.INTERNAL_M: float = float(host.internal_m)
        self.up: int = int(host.up)
        self.flat = [a for a in range(3) if a != self.up]
        #: A centimetre in the internal unit: how near two hits are one plane,
        #: and a light is reached.
        self._cm = 0.01 / self.INTERNAL_M
        self.baked = list(dict.fromkeys(host.nodes(baked)))
        self.span = host.bounds(self.baked) if self.baked else None
        self.surround: List[Any] = []
        self.moving: List[Any] = []
        for node in dict.fromkeys(host.nodes(unbaked)):
            if self.span and self._covers(self._extent(node), self.span):
                self.surround.append(node)
            else:
                self.moving.append(node)
        self.rays = host.rays(self.baked + self.surround)

    # ------------------------------------------------------------------
    # The placement
    # ------------------------------------------------------------------

    def place(self) -> Optional[ProbeSite]:
        """The probe's site, or ``None`` when nothing is baked to see."""
        if not self.rays or not self.span:
            return None
        # The room first, measured from open air near the bake's middle. What
        # was corrected on the way there describes no probe: not reported.
        seed = self._settle(self._centre(self.span), [])
        room = self.room(seed)
        reach = self._reach(room)
        inside = [
            node
            for node in self.moving
            if self._holds(reach, self._centre(self._extent(node)))
        ]
        if inside:
            target, reason = self._centre(self.host.bounds(inside)), "over what moves"
        elif all(room[side][a] is not None for side in (0, 1) for a in self.flat):
            target, reason = self._centre(reach), "the room's middle"
        else:
            props = [b for b in self.baked + self.surround if not self._shell(b)]
            target = self._centre(self.host.bounds(props or self.baked))
            reason = "over the baked objects" if props else "the bake's middle"
        # Eye level: half the room's height, at most MAX_HEIGHT_M over its
        # floor -- the floor the room's own cone found, past the furniture,
        # else (nothing below) the bake's lowest point.
        floor = room[0][self.up]
        if floor is None:
            floor = self.span[0][self.up]
        rise = self.MAX_HEIGHT_M / self.INTERNAL_M
        if room[1][self.up] is not None:
            rise = min(rise, (room[1][self.up] - floor) / 2.0)
        point = list(target)
        point[self.up] = floor + rise
        # What stands there (the car the probe hovers over, a column at the
        # room's middle) is stepped out of and kept clear of.
        notes: List[str] = []
        position = self._settle(point, notes)
        box, open_faces = self._box(self.room(position), position)
        return ProbeSite(
            position=self._to_scene(position),
            box=[self._to_scene(corner) for corner in box] if box else None,
            hide=list(self.moving),
            reason=reason,
            notes=list(dict.fromkeys(notes)),
            open_faces=open_faces,
        )

    def room(self, point: Sequence[float]) -> List[List[Optional[float]]]:
        """The room around *point* (internal units): :meth:`faces`, from
        *point* and from the open air :attr:`ROOM_REACH_M` around it.

        A face is open when it is open from *point*; a closed one takes the
        farthest wall any of the points found for it -- the floor past a car
        roof one point stood over, the wall behind the column another stood
        beside. A neighbour counts only when *point* reaches it in a straight
        line (a wall between is another room) and it is not inside anything.
        """
        base = self.faces(point, scene_units=False)
        found = [[[c] if c is not None else [] for c in side] for side in base]
        for reach in self.ROOM_REACH_M:
            step = reach / self.INTERNAL_M
            for axis in self.flat:
                for sign in (1.0, -1.0):
                    d = [0.0, 0.0, 0.0]
                    d[axis] = sign
                    if self.rays.cast(point, d, far=step) is not None:
                        continue
                    other = [a + b * step for a, b in zip(point, d)]
                    if self._look(other).trapped:
                        continue
                    for side, corner in enumerate(self.faces(other, scene_units=False)):
                        for a, c in enumerate(corner):
                            if c is not None:
                                found[side][a].append(c)
        return [
            [
                (min if side == 0 else max)(found[side][a])
                if base[side][a] is not None
                else None
                for a in range(3)
            ]
            for side in (0, 1)
        ]

    def faces(
        self, point: Sequence[float], scene_units: bool = True
    ) -> List[List[Optional[float]]]:
        """The room around *point*: ``[[min], [max]]``, ``None`` on an open face.

        Each face is measured by a cone of rays (:attr:`WALL_RINGS`) along its
        axis. A face more than half of whose rays meet nothing is open. Else
        its wall is the farthest plane :attr:`WALL_SHARE` of the hits that
        face it squarely (:attr:`WALL_FACING`) land on, within a centimetre --
        furniture between the point and a wall is passed over -- and where no
        plane gathers them (a curved or turned wall, a slope), the axis ray's
        own hit, else their median. A face no hit faces squarely is open.

        Parameters:
            point: Where to measure from.
            scene_units: *point* and the result in scene units (else the
                host's internal unit).
        """
        origin = self.host.to_internal(point) if scene_units else list(point)
        result: List[List[Optional[float]]] = [[None] * 3, [None] * 3]
        for axis in range(3):
            u, v = [a for a in range(3) if a != axis]
            for side, sign in ((0, -1.0), (1, 1.0)):
                rays = [(0.0, 0.0)] + [
                    (math.radians(tilt), math.radians(30.0 * k))
                    for tilt in self.WALL_RINGS
                    for k in range(12)
                ]
                hits = []
                for tilt, turn in rays:
                    d = [0.0, 0.0, 0.0]
                    d[axis] = sign * math.cos(tilt)
                    d[u] = math.sin(tilt) * math.cos(turn)
                    d[v] = math.sin(tilt) * math.sin(turn)
                    hits.append(self.rays.cast(origin, d))
                landed = [h for h in hits if h is not None]
                if len(landed) * 2 < len(rays):
                    continue
                facing = [
                    (i, h)
                    for i, h in enumerate(hits)
                    if h is not None and abs(h.normal[axis]) >= self.WALL_FACING
                ]
                if not facing:
                    continue
                coords = [h.point[axis] for _i, h in facing]
                need = max(3, int(math.ceil(self.WALL_SHARE * len(coords))))
                wall = next(
                    (
                        c
                        for c in sorted(coords, key=lambda c: -sign * c)
                        if sum(abs(o - c) <= self._cm for o in coords) >= need
                    ),
                    None,
                )
                if wall is None:
                    wall = (
                        facing[0][1].point[axis]
                        if facing[0][0] == 0
                        else sorted(coords)[len(coords) // 2]
                    )
                result[side][axis] = wall
        if scene_units:
            return [
                [self.host.to_scene(c) if c is not None else None for c in r]
                for r in result
            ]
        return result

    # ------------------------------------------------------------------
    # Open air
    # ------------------------------------------------------------------

    def _settle(self, point: Sequence[float], notes: List[str]) -> List[float]:
        """*point*, moved out of any solid it is in and clear of what it sees."""
        p = list(point)
        look = self._look(p)
        if look.trapped:
            out = (self._leave(p, look.solid) if look.enclosed else None) or (
                self._escape(p)
            )
            if out is not None:
                notes.append(
                    f"moved out of {self._leaf(look.solid)}, which it was inside"
                )
                p = out
            else:
                notes.append(
                    f"found no open air around {self._leaf(look.solid)}; "
                    "captured from inside it"
                )
        return self._clear(p, notes)

    def _look(self, p: Sequence[float]) -> _Look:
        """What *p* sees around it: whether it is shut in a solid or a cavity,
        and its nearest surface.

        Shut in, it meets something on all but a few of :attr:`_AROUND`'s
        rays, sees no light (no fixture in view, no sky through a gap), and
        either sees the BACK of what surrounds it on three rays in four -- the
        inside of a shell, which no runtime draws -- or is within
        :attr:`CAVITY_M` of it (a cabin, a cupboard). A room sees the front of
        its walls, or a light; a glazed room lit by the sun alone sees its
        walls' fronts, at a room's distance. A scene with no light to see
        (emissive surfaces only) leaves the shell's back as the one sign, and
        a shell with something in it is a room built inside out, not a solid.
        """
        hits = [self.rays.cast(p, d) for d in self._AROUND]
        landed = [(d, h) for d, h in zip(self._AROUND, hits) if h is not None]
        if not landed:
            return _Look(False, "", float("inf"), (0.0, 0.0, 0.0), "")
        towards, nearest = min(landed, key=lambda dh: dh[1].distance)
        trapped = False
        solid = ""
        if len(landed) >= len(self._AROUND) - 3:
            owners = [self.rays.owners[h.mesh] for _d, h in landed]
            # Over the list, not a set: a tie goes to the first ray's owner,
            # never to a set's hash order (a new one in every process).
            solid = max(owners, key=owners.count)
            back = (
                sum(sum(a * b for a, b in zip(d, h.normal)) > 0 for d, h in landed) * 4
                >= len(landed) * 3
            )
            points, sky = self._light_points()
            if points or sky:
                if not self._lit(p, len(landed)):
                    distances = sorted(h.distance for _d, h in landed)
                    median = distances[len(distances) // 2]
                    trapped = back or median < self.CAVITY_M / self.INTERNAL_M
            else:
                trapped = back and not self._holds_another(solid)
        enclosed = False
        if not trapped:
            props = [self.rays.owners[h.mesh] for _d, h in landed]
            props = [o for o in props if not self._shell(o)]
            if props:
                owner = max(props, key=props.count)
                enclosed = props.count(owner) >= self.ENCLOSED_SHARE * len(
                    self._AROUND
                ) and self._holds(self._extent(owner), p)
                if enclosed:
                    trapped, solid = True, owner
        return _Look(
            trapped,
            solid,
            nearest.distance,
            tuple(-c for c in towards),
            self.rays.owners[nearest.mesh],
            enclosed,
        )

    def _lit(self, p: Sequence[float], landed: int) -> bool:
        """Whether *p* sees a light: a placed one in plain view, or -- with a
        sky light in the scene -- the sky through a ray that met nothing."""
        points, sky = self._light_points()
        if sky and landed < len(self._AROUND):
            return True
        for q, own in points:
            d = [b - a for a, b in zip(p, q)]
            length = math.sqrt(sum(c * c for c in d))
            if length < self._cm:
                return True
            hit = self.rays.cast(p, [c / length for c in d], far=length)
            if hit is None or hit.distance >= length - self._cm:
                return True
            if own and self.rays.owners[hit.mesh] == own:
                return True
        return False

    def _leave(self, p: Sequence[float], solid: Any) -> Optional[List[float]]:
        """Out of the furniture *p* is inside (:attr:`_Look.enclosed`) by its
        opening: the nearest side of *solid*'s bounds, along the floor's axes,
        that a ray reaches without meeting a surface -- never out through a
        wall, into the gap behind a cabinet -- then stood off from it
        (:meth:`_stand_off`). ``None`` when no side is open."""
        clear = self.CLEARANCE_M / self.INTERNAL_M
        bounds = self._extent(solid)
        ways = []
        for axis in self.flat:
            for side, sign in ((0, -1.0), (1, 1.0)):
                d = [0.0, 0.0, 0.0]
                d[axis] = sign
                edge = bounds[side][axis] + sign * clear
                length = abs(edge - p[axis])
                if self.rays.cast(p, d, far=length) is not None:
                    continue
                out = list(p)
                out[axis] = edge
                if not self._look(out).trapped:
                    ways.append((length, out))
        if not ways:
            return None
        return self._stand_off(min(ways)[1], p)

    def _escape(self, p: Sequence[float]) -> Optional[List[float]]:
        """The nearest point out of the solid *p* is in, or ``None``.

        Sideways first, along the floor's two axes: that keeps the probe's
        height and its room -- a column reaches the ceiling, and up through
        it is the roof. Up only when no side lets out (a solid boxed in on
        every side); never down, which is through what the solid stands on.
        """
        clear = self.CLEARANCE_M / self.INTERNAL_M
        for ways in (
            [(axis, sign) for axis in self.flat for sign in (1.0, -1.0)],
            [(self.up, 1.0)],
        ):
            best = None
            for axis, sign in ways:
                d = [0.0, 0.0, 0.0]
                d[axis] = sign
                start, travelled = list(p), 0.0
                # Two hops: a skin with a lining (a car door) is two surfaces.
                for _hop in range(2):
                    hit = self.rays.cast(start, d)
                    step = (hit.distance if hit else 0.0) + clear
                    start = [a + b * step for a, b in zip(start, d)]
                    travelled += step
                    if best is not None and travelled >= best[0]:
                        break
                    if not self._look(start).trapped:
                        best = (travelled, start)
                        break
                    if hit is None:
                        break
            if best is not None:
                return self._stand_off(best[1], p)
        return None

    def _stand_off(self, point: List[float], inside: Sequence[float]) -> List[float]:
        """*point*, just out of a solid, stepped on away from it while the
        solid fills more than :attr:`SOLID_SHARE` of its view -- a step at a
        time, never into another surface, at most a few."""
        clear = self.CLEARANCE_M / self.INTERNAL_M
        solid = self._look(inside).solid
        away = [b - a for a, b in zip(inside, point)]
        length = math.sqrt(sum(c * c for c in away)) or 1.0
        away = [c / length for c in away]
        for _step in range(4):
            hits = [self.rays.cast(point, d) for d in self._AROUND]
            mine = sum(
                1 for h in hits if h is not None and self.rays.owners[h.mesh] == solid
            )
            if mine <= self.SOLID_SHARE * len(self._AROUND):
                break
            if self.rays.cast(point, away, far=2.0 * clear) is not None:
                break
            ahead = [a + b * clear for a, b in zip(point, away)]
            if self._look(ahead).trapped:
                break
            point = ahead
        return point

    def _clear(self, p: Sequence[float], notes: List[str]) -> List[float]:
        """*p* nudged away from any surface nearer than :attr:`CLEARANCE_M` --
        from the nearest, a few times -- or the clearest point on the way."""
        clear = self.CLEARANCE_M / self.INTERNAL_M
        start = list(p)
        look = self._look(start)
        if look.nearest >= clear:
            return start
        near = look.near
        best, best_reach = start, look.nearest
        point = start
        for _step in range(6):
            shift = (clear - look.nearest) * 1.05
            point = [a + b * shift for a, b in zip(point, look.away)]
            look = self._look(point)
            if look.trapped:
                break
            if look.nearest > best_reach:
                best, best_reach = point, look.nearest
            if look.nearest >= clear:
                break
        if best is not start:
            notes.append(
                f"moved clear of {self._leaf(near)} "
                f"({min(best_reach, clear) * self.INTERNAL_M:.2f} m off it)"
            )
        return best

    def _light_points(self) -> Tuple[List[Tuple[Vec, Optional[Any]]], bool]:
        """The host's lights (:meth:`ProbeHost.lights`), read once."""
        if self._lights is None:
            self._lights = self.host.lights()
        return self._lights

    # ------------------------------------------------------------------
    # Boxes
    # ------------------------------------------------------------------

    def _box(
        self, room: List[List[Optional[float]]], position: Sequence[float]
    ) -> Tuple[Optional[Box], int]:
        """*room* with each open face pushed far, and how many were open;
        ``None`` when all six are."""
        open_faces = sum(c is None for corner in room for c in corner)
        if open_faces == 6:
            return None, 6
        diagonal = math.sqrt(sum((b - a) ** 2 for a, b in zip(*self.span)))
        far = max(self.OPEN_FAR_M / self.INTERNAL_M, 10.0 * diagonal)
        box = [
            [
                c if c is not None else position[a] + (far if side else -far)
                for a, c in enumerate(room[side])
            ]
            for side in (0, 1)
        ]
        return box, open_faces

    def _reach(self, room: List[List[Optional[float]]]) -> Box:
        """*room*'s extent, its open faces stopped at the bake's own bounds:
        what moves past the baked area is no part of this room."""
        return [
            [
                c if c is not None else self.span[side][a]
                for a, c in enumerate(room[side])
            ]
            for side in (0, 1)
        ]

    def _covers(self, bounds: Box, span: Box) -> bool:
        """Whether *bounds*' footprint covers *span*'s (a ground, a dome)."""
        tolerance = 0.01 * math.sqrt(sum((b - a) ** 2 for a, b in zip(*span)))
        return all(
            bounds[0][a] <= span[0][a] + tolerance
            and bounds[1][a] >= span[1][a] - tolerance
            for a in self.flat
        )

    def _shell(self, node: Any) -> bool:
        """Whether *node* spans the bake on two axes or more: a wall, a floor,
        a ground -- the room, not something in it."""
        bounds = self._extent(node)
        return (
            sum(
                (bounds[1][a] - bounds[0][a])
                >= 0.8 * (self.span[1][a] - self.span[0][a])
                for a in range(3)
            )
            >= 2
        )

    # ------------------------------------------------------------------
    # Shapes and units
    # ------------------------------------------------------------------

    def _holds_another(self, solid: Any) -> bool:
        """Whether *solid*'s bounds hold anything else's middle: furniture in
        a room, which a solid has none of."""
        bounds = self._extent(solid)
        return any(
            self._holds(bounds, self._centre(self._extent(node)))
            for node in self.baked + self.surround + self.moving
            if node != solid
        )

    def _extent(self, node: Any) -> Box:
        """The host's bounds of one node (:meth:`ProbeHost.bounds`), read once."""
        if node not in self._extents:
            self._extents[node] = self.host.bounds([node])
        return self._extents[node]

    @staticmethod
    def _centre(bounds: Box) -> List[float]:
        return [(lo + hi) / 2.0 for lo, hi in zip(*bounds)]

    @staticmethod
    def _holds(bounds: Box, point: Sequence[float]) -> bool:
        return all(lo <= c <= hi for c, lo, hi in zip(point, *bounds))

    def _to_scene(self, point: Sequence[float]) -> List[float]:
        return [self.host.to_scene(c) for c in point]

    @staticmethod
    def _leaf(node: Any) -> str:
        return str(node).rsplit("|", 1)[-1] if node else "geometry"
