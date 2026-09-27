# !/usr/bin/python
# coding=utf-8
"""UV-layout primitives: seam, pack, plan and remap (arrays / numbers in -> out).

The root registers every name (``from pythontk import UvPack``); importing this
package loads none of its modules.

- :mod:`~pythontk.geo_utils.uv.pack` -- :class:`UvPack`: island packing via the
  optional ``xatlas`` engine.
- :mod:`~pythontk.geo_utils.uv.budget` -- :class:`UvBudget`: how many maps, at
  what texel density, before anything packs.
- :mod:`~pythontk.geo_utils.uv.transfer` -- :class:`UvTransfer`: texel remap
  between two UV layouts of the same triangles.
- :mod:`~pythontk.geo_utils.uv.cylinder_seams` -- :class:`CylinderSeams`: where
  to cut a cylinder / tube / turned mesh (and its developed seed layout).

Automatic unwrapping, which round-trips mesh FILES through external CLIs, is
mesh IO and lives in :mod:`pythontk.file_utils.uv_unwrap` (``ptk.UvUnwrap``).
"""
