# !/usr/bin/python
# coding=utf-8
"""Emitter geometry for a flat light-fixture plate — pure math, no DCC.

A ceiling troffer, a wall panel, a lit sign: each is a thin slab whose broad face
is the thing that emits. Turning that slab into a rectangular area light is the
same arithmetic in every host — pick the thin axis as the emission normal, take
the other two as the rectangle, and stand the light clear of the housing — so it
lives here rather than once in ``mayatk.LightUtils`` and again in
``blendertk.LightUtils``, which is how the two would drift.

Two solvers, for the two things a caller can know about the plate:

- :meth:`PlateEmitter.from_bounds` reads a world AXIS-ALIGNED box. Cheap, needs
  nothing but a bounding box, and exact only while the plate lies on the world
  axes — right for the architectural case (ceiling and wall plates), wrong for a
  raked fitting, whose box is inflated by its own rotation.
- :meth:`PlateEmitter.from_points` reads the plate's actual vertices and solves
  an ORIENTED rectangle from their principal axes, so rotation costs it nothing.
  Prefer it whenever the caller has components in hand, and pass the geometry's
  own face normal with it: PCA recovers the normal's line but not its sign, and
  a face already knows which way it faces.
"""

from typing import NamedTuple, Optional, Sequence, Tuple


class PlateEmitter(NamedTuple):
    """The area light a flat plate implies."""

    #: Rectangle dimensions, larger edge first.
    size: Tuple[float, float]
    #: Unit emission direction.
    normal: Tuple[float, float, float]
    #: Where to stand the light — clear of the plate's own thickness.
    position: Tuple[float, float, float]
    #: Index of the plate's thin axis. ``None`` from :meth:`from_points`, where
    #: the plate is not assumed to lie on a world axis at all.
    axis: Optional[int] = None
    #: Unit in-plane direction of the rectangle's LONG edge — the roll a host
    #: needs to orient the emitter, not just where it points.
    tangent: Optional[Tuple[float, float, float]] = None

    @classmethod
    def from_bounds(
        cls,
        minimum: Sequence[float],
        maximum: Sequence[float],
        toward: Optional[Sequence[float]] = None,
        offset: float = 0.01,
        up_axis: int = 2,
    ) -> "PlateEmitter":
        """Solve the emitter for the plate bounded by *minimum* / *maximum*.

        Parameters:
            minimum / maximum: Opposite corners of the world axis-aligned box.
            toward: World point the plate should face — the room's centre. The
                displacement to it is only believed when it exceeds the plate's
                own thickness: a ceiling grid's members sit IN the plane of any
                centre derived from them, so the sign there is modelling noise,
                and trusting it aims half the fixtures at the ceiling (measured
                on a production room: 2 of 4). Below that threshold the plate is
                treated as coplanar and *up_axis* decides.
            offset: Clearance between the plate's surface and the light. Applied
                on top of half the plate's thickness — pushing from the CENTRE by
                offset alone leaves the light inside any housing thicker than
                twice it, where its own geometry blocks it.
            up_axis: Index of the host's up axis (Blender Z=2, Maya Y=1). Only
                consulted for the coplanar case, where "down" is the sole useful
                answer; an ambiguous plate on another axis keeps +axis, which is
                arbitrary but deterministic.

        Returns:
            (PlateEmitter) size / normal / position / axis.
        """
        extents = [float(maximum[i]) - float(minimum[i]) for i in range(3)]
        center = [(float(maximum[i]) + float(minimum[i])) / 2.0 for i in range(3)]

        axis = min(range(3), key=lambda i: extents[i])
        broad = sorted(extents[i] for i in range(3) if i != axis)
        size_y, size_x = broad[0], broad[1]

        sign = 1.0
        if toward is not None:
            delta = float(toward[axis]) - center[axis]
            # Floored at 1% of the long edge because a plate modelled as a flat
            # plane — the common way to model a lens — has zero thickness, which
            # would hand the decision straight back to the noise this guards.
            coplanar = max(extents[axis], size_x * 0.01)
            if abs(delta) > coplanar:
                sign = 1.0 if delta > 0 else -1.0
            elif axis == up_axis:
                sign = -1.0
        elif axis == up_axis:
            sign = -1.0

        normal = [0.0, 0.0, 0.0]
        normal[axis] = sign
        clearance = extents[axis] * 0.5 + float(offset)
        position = [center[i] + normal[i] * clearance for i in range(3)]

        # The long edge is a world axis here, so the tangent is exact rather
        # than a convention -- which spares every caller inventing one to
        # orient the rectangle's roll.
        long_axis = max(
            (i for i in range(3) if i != axis), key=lambda i: extents[i]
        )
        tangent = [0.0, 0.0, 0.0]
        tangent[long_axis] = 1.0

        return cls(
            size=(size_x, size_y),
            normal=tuple(normal),
            position=tuple(position),
            axis=axis,
            tangent=tuple(tangent),
        )

    @classmethod
    def from_points(
        cls,
        points: Sequence[Sequence[float]],
        normal: Optional[Sequence[float]] = None,
        toward: Optional[Sequence[float]] = None,
        offset: float = 0.0,
        up_axis: int = 2,
    ) -> "PlateEmitter":
        """Solve an ORIENTED emitter for an arbitrary flat patch of geometry.

        :meth:`from_bounds` reads a world axis-aligned box, which is exact only
        while the plate lies on the world axes: a raked or rotated fitting gets
        a rectangle inflated by its own rotation (a 45-degree plate's box is
        ~1.4x its real width) and an aim snapped to the nearest axis. Given the
        actual points -- the vertices of the faces an artist selected -- neither
        approximation is necessary.

        The frame comes from the point cloud's principal axes
        (:meth:`pythontk.PointCloud.pca_basis`): for a flat patch the
        smallest-variance axis IS the plane normal and the other two are the
        in-plane axes, with the largest being the rectangle's long edge. So one
        decomposition yields size and orientation together, and no bounding box
        is involved.

        Passing *normal* (the geometry's own averaged face normal) is strongly
        preferred and makes the result exact: PCA recovers the normal's LINE but
        has no sign convention, so without it the emitting side has to be
        guessed from *toward* -- and a face already knows which way it faces.

        Parameters:
            points: World-space vertex positions of the patch.
            normal: Unit emission direction, e.g. the averaged face normal.
                ``None`` derives the axis from PCA and resolves its sign from
                *toward* / *up_axis* as :meth:`from_bounds` does.
            toward: World point the plate should face; only consulted when
                *normal* is absent.
            offset: Clearance along the emission normal. No half-thickness term
                here: these points ARE the emitting surface, not a housing.
            up_axis: Host up axis, for the ambiguous-sign fallback.

        Returns:
            (PlateEmitter) with *tangent* set and *axis* ``None``.
        """
        pts = [tuple(float(c) for c in p[:3]) for p in points]
        if not pts:
            raise ValueError("from_points needs at least one point")
        count = len(pts)
        centroid = tuple(sum(p[i] for p in pts) / count for i in range(3))

        axes = cls._principal_axes(pts)
        if normal is not None:
            unit_normal = _normalize(normal)
            # Prefer the principal axis least parallel to the normal: for a
            # rectangle that is its long edge.
            candidates = sorted(axes, key=lambda a: abs(_dot(a, unit_normal)))
            tangent = _orthogonalize(candidates[0], unit_normal)
        else:
            # PCA orders by descending variance, so the LAST axis is the plane
            # normal of a flat patch and the first is the long in-plane edge.
            unit_normal = axes[2]
            tangent = _orthogonalize(axes[0], unit_normal)
            unit_normal = cls._resolve_sign(
                unit_normal, centroid, toward, up_axis, extent=0.0
            )
        bitangent = _cross(unit_normal, tangent)

        along = [_dot(_sub(p, centroid), tangent) for p in pts]
        across = [_dot(_sub(p, centroid), bitangent) for p in pts]
        size_x = max(along) - min(along)
        size_y = max(across) - min(across)
        if size_y > size_x:  # keep the long edge first, as from_bounds does
            size_x, size_y = size_y, size_x
            tangent, bitangent = bitangent, tangent

        # Re-centre on the rectangle rather than the centroid: an uneven vertex
        # distribution (a subdivided half) pulls the mean off the middle.
        middle = _add(
            centroid,
            _add(
                _scale(tangent, (max(along) + min(along)) / 2.0),
                _scale(bitangent, (max(across) + min(across)) / 2.0),
            ),
        )
        position = _add(middle, _scale(unit_normal, float(offset)))
        return cls(
            size=(size_x, size_y),
            normal=tuple(unit_normal),
            position=tuple(position),
            axis=None,
            tangent=tuple(tangent),
        )

    @staticmethod
    def _principal_axes(pts):
        """Three orthonormal axes, largest variance first.

        Falls back to the world axes when the cloud is degenerate or numpy is
        absent -- ``PointCloud`` treats it as an optional dependency and this
        primitive must not become the reason a light cannot be built.
        """
        try:
            from pythontk.geo_utils.pointcloud import PointCloud

            basis = PointCloud.pca_basis(pts)
        except Exception:  # noqa: BLE001 — numpy missing, or fewer than 3 points
            basis = None
        if not basis:
            return [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        # Flat row-major 4x4; rows 0-2 are the basis vectors.
        return [tuple(basis[row * 4 : row * 4 + 3]) for row in range(3)]

    @staticmethod
    def _resolve_sign(normal, center, toward, up_axis, extent):
        """Point *normal* at *toward*, falling back to 'down' about *up_axis*."""
        if toward is not None:
            delta = _dot(_sub(tuple(toward), center), normal)
            if abs(delta) > extent:
                return normal if delta > 0 else _scale(normal, -1.0)
        return normal if normal[up_axis] < 0 else _scale(normal, -1.0)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(v):
    length = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5 or 1.0
    return (v[0] / length, v[1] / length, v[2] / length)


def _orthogonalize(v, normal):
    """*v* with its *normal* component removed, unitized (any perpendicular if parallel)."""
    projected = _sub(v, _scale(normal, _dot(v, normal)))
    if (projected[0] ** 2 + projected[1] ** 2 + projected[2] ** 2) < 1e-12:
        # v is parallel to the normal: any perpendicular will do.
        seed = (1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0)
        projected = _cross(normal, seed)
    return _normalize(projected)
