# !/usr/bin/python
# coding=utf-8
"""DCC-agnostic procedural geometry.

Geometry *composed* from the :mod:`pythontk.math_utils` primitives — curves and
polylines, position grids, and parametric generators/deformers — that emit plain
values (points, frames) for any DCC adapter to build into real meshes. This is
the layer between the math primitives and the mesh data/IO in
:mod:`pythontk.file_utils`; nothing here imports a DCC or commits to mesh
topology (faces/edges/winding) — that stays the adapter's job.

All classes are lazy-loaded via the pythontk root package; import from pythontk
directly: ``from pythontk import Polyline, PointCloud``.

- :class:`~pythontk.geo_utils.polyline.Polyline` — pure polyline/curve geometry:
  generate, measure, resample, reshape, and frame an *ordered* sequence of points.
- :class:`~pythontk.geo_utils.pointcloud.PointCloud` — *unordered* point-set
  geometry: PCA alignment, proximity clustering, positional hashing.

- :class:`~pythontk.geo_utils.rail_surface.RailSurface` — a rail-driven
  parametric surface: frames a rail into a ``(u,v)`` grid and applies a
  caller-supplied displacement field. The reusable substrate under curve-driven
  surface generators.

The curtain *generator* that once lived here (``CurtainDrape``) was one tool's
displacement math, not a general geometry primitive: its rail→grid machinery
was extracted into the general :class:`RailSurface` primitive (which stays
here), and the curtain-specific remainder now lives with its consumers as the
vendored ``edit_utils._curtain_drape`` twins in mayatk and blendertk
(code-identical, drift-guarded by extapps' ``test_vendor_sync.py``).
"""

# Lazy-loaded via parent package - no explicit imports needed
