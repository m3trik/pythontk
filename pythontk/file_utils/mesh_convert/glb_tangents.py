# !/usr/bin/python
# coding=utf-8
"""Repair a GLB's shipped tangents against the UVs they describe.

Why this exists: glTF defines a vertex's bitangent as ``cross(normal,
tangent.xyz) * tangent.w`` and a normal map's green as image-up, which under
glTF's top-left UV origin is the direction of DECREASING V. ``w`` is the one
bit that says which way that is, and it is -1 on every UV shell that is
mirrored. FBX2glTF (v0.13.1) transcribes an FBX's tangent layer but not its
binormal -- where that bit lives -- and ships ``w = +1`` on every vertex.
three.js takes a shipped TANGENT over the derivative frame it would otherwise
build, so each mirrored shell renders its normal map with green inverted: every
bump lit from the wrong side. Measured on a production assembly (2026-09-23,
reported as a button rendering bad normals in the WebXR preview, the day the
DCC exports began shipping tangents): 7 of 61 tangent-carrying primitives, and
on each exactly its mirrored triangles -- the button and four light fixtures
whole, 29% of a table, half of a pair of legs. Blender's MikkTSpace glTF export
writes ``w = -1`` on a mirrored shell, and this is that rule -- it parts from
MikkTSpace only on a triangle wound against its own vertex normals, where
MikkTSpace follows the winding and this the normal glTF's bitangent is built
from. The same push shipped 8 ZERO-length tangents, on UV slivers the DCC could
not orient: glTF requires a unit tangent (the Khronos validator's
ACCESSOR_NON_UNIT), and three.js normalizes the zero vector into NaN.

What it does, per TANGENT accessor: each triangle of every primitive reading it
adds, at its three corners, the directions its UVs run -- image-up (``-dP/dV``)
and ``dP/dU`` -- from its positions and the normal map's own texCoord
(``KHR_texture_transform``'s where it names one; the transform's offset,
rotation and scale are not applied, since a DCC's tangent space, like
MikkTSpace's, is the attribute's own). A corner's ``w`` becomes the sign that
points its bitangent image-up. Its direction is kept as shipped (FBX2glTF keeps
it along +U; only the bit is lost) unless it has none, when it is rebuilt from
``dP/dU`` made perpendicular to the normal -- or, where every UV triangle
around the vertex is degenerate too, from any direction perpendicular to it. A
vertex no triangle can vote for keeps its sign.

Refused, and left as shipped: a TANGENT whose readers are not all triangle lists
over one vertex set (the same POSITION and NORMAL), that needs an attribute
that is not float or that :meth:`GlbReader.accessor` cannot read (sparse, or
not in the file's own BIN), or whose readers' indices run past its vertices.
The rewrite is in place, at the TANGENT's own stride, so a file with nothing to
repair is not written at all.

Runs in :meth:`MeshConvert.fbx_to_glb`'s session and is safe to run standalone
on any GLB.
"""

from __future__ import annotations

import logging
import math
import struct
from typing import Dict, List, Optional, Sequence, Tuple

from pythontk.file_utils.mesh_convert._mesh_convert import GlbTarget, MeshConvert
from pythontk.file_utils.mesh_convert.glb_reader import GlbReader

__all__ = ["GlbTangents"]

logger = logging.getLogger(__name__)

Vec = Tuple[float, ...]


class _GlbTangentsInternal:
    """Decode, accumulate and rebuild helpers for :class:`GlbTangents`."""

    #: glTF ``mode`` of a triangle list: the only topology the vote walks.
    TRIANGLES = 4
    #: glTF ``componentType`` FLOAT: what every vertex attribute read here must
    #: be. A quantized one (KHR_mesh_quantization) is dequantized by the node
    #: transform, which may scale its axes unevenly -- directions this pass
    #: cannot trust without composing that transform in.
    FLOAT = 5126
    #: Unsigned index component types (UNSIGNED_BYTE / _SHORT / _INT).
    INDEX_TYPES = (5121, 5123, 5125)
    #: A shipped tangent shorter than this has no direction to keep; a hint
    #: with less than this fraction of its length across the normal has none
    #: to give (:meth:`_across`).
    DEGENERATE = 1e-6

    @staticmethod
    def _dot(a: Vec, b: Vec) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _cross(a: Vec, b: Vec) -> Vec:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    @classmethod
    def _across(cls, normal: Vec, hint: Vec) -> Optional[Vec]:
        """*hint* made perpendicular to *normal*, unit length; ``None`` when
        nothing of it is left (NaN included).

        Judged against *hint*'s own length, never an absolute floor: a sum of
        UV directions is weighted by its triangles' UV areas and extents, so on
        a sliver -- where a DCC leaves a zero tangent -- it is tiny without
        being absent (1e-8 on a 0.1 mm triangle spanning 1e-4 of the UVs).
        """
        d = cls._dot(hint, normal)
        v = (hint[0] - d * normal[0], hint[1] - d * normal[1], hint[2] - d * normal[2])
        length = math.sqrt(cls._dot(v, v))
        if not length > cls.DEGENERATE * math.sqrt(cls._dot(hint, hint)):
            return None
        return (v[0] / length, v[1] / length, v[2] / length)

    @classmethod
    def _decode(
        cls,
        reader: GlbReader,
        index: object,
        kind: str,
        types: Sequence[int] = (FLOAT,),
    ) -> Optional[List[Vec]]:
        """Accessor *index* decoded, or ``None`` unless it is a *kind* accessor
        of one of *types* that :meth:`GlbReader.accessor` can read."""
        accessors = reader.gltf.get("accessors") or []
        if not isinstance(index, int) or not 0 <= index < len(accessors):
            return None
        accessor = accessors[index] or {}
        if accessor.get("type") != kind or accessor.get("componentType") not in types:
            return None
        return reader.accessor(index) or None

    @staticmethod
    def _accumulate(
        positions: Sequence[Vec],
        uvs: Sequence[Vec],
        corners: Sequence[int],
        up: List[List[float]],
        along: List[List[float]],
    ) -> None:
        """Add each triangle's UV directions to its three corners' sums.

        Taken undivided by the UV area: ``(e1 * d2u - e2 * d1u) * sign(det)``
        is ``-dP/dV`` and ``(e1 * d2v - e2 * d1v) * sign(det)`` is ``dP/dU``,
        each scaled by ``|det|`` -- so a sliver of UV weighs little rather than
        blowing up, and a degenerate one (``det`` 0) adds nothing.
        """
        for k in range(0, len(corners) - 2, 3):
            a, b, c = corners[k], corners[k + 1], corners[k + 2]
            pa, pb, pc = positions[a], positions[b], positions[c]
            d1u, d1v = uvs[b][0] - uvs[a][0], uvs[b][1] - uvs[a][1]
            d2u, d2v = uvs[c][0] - uvs[a][0], uvs[c][1] - uvs[a][1]
            det = d1u * d2v - d1v * d2u
            if det == 0.0:
                continue
            sign = 1.0 if det > 0.0 else -1.0
            for i in range(3):
                e1, e2 = pb[i] - pa[i], pc[i] - pa[i]
                image_up = (e1 * d2u - e2 * d1u) * sign
                plus_u = (e1 * d2v - e2 * d1v) * sign
                for corner in (a, b, c):
                    up[corner][i] += image_up
                    along[corner][i] += plus_u

    @classmethod
    def _repairs(
        cls,
        reader: GlbReader,
        index: int,
        readers: Sequence[dict],
    ) -> Tuple[Dict[int, Vec], int, int]:
        """``({vertex: (x, y, z, w)}, signs, directions)`` for TANGENT accessor
        *index*: each vertex whose shipped value breaks the contract (module
        docstring), and how many of them changed sign / had their direction
        rebuilt. Empty when the accessor or any reader of it cannot be read as
        this pass needs -- a vertex one reader cannot vote on may mean something
        else to it, so the accessor is left whole."""
        refused: Tuple[Dict[int, Vec], int, int] = ({}, 0, 0)
        tangents = cls._decode(reader, index, "VEC4")
        if tangents is None:
            return refused
        count = len(tangents)
        attributes = [primitive.get("attributes") or {} for primitive in readers]
        vertex_set = {(a.get("POSITION"), a.get("NORMAL")) for a in attributes}
        if len(vertex_set) != 1:
            return refused
        position_index, normal_index = vertex_set.pop()
        positions = cls._decode(reader, position_index, "VEC3")
        normals = cls._decode(reader, normal_index, "VEC3")
        if positions is None or normals is None:
            return refused
        if not len(positions) == len(normals) == count:
            return refused

        materials = reader.gltf.get("materials") or []
        up = [[0.0, 0.0, 0.0] for _ in range(count)]
        along = [[0.0, 0.0, 0.0] for _ in range(count)]
        for primitive, attrs in zip(readers, attributes):
            if primitive.get("mode", cls.TRIANGLES) != cls.TRIANGLES:
                return refused
            material = primitive.get("material")
            normal_map = (
                (materials[material] or {}).get("normalTexture") or {}
                if isinstance(material, int) and 0 <= material < len(materials)
                else {}
            )
            # KHR_texture_transform's texCoord, where it names one, is the set
            # the map samples (it overrides the textureInfo's own).
            transform = (normal_map.get("extensions") or {}).get(
                "KHR_texture_transform"
            ) or {}
            tex_coord = transform.get("texCoord", normal_map.get("texCoord", 0))
            uvs = cls._decode(reader, attrs.get(f"TEXCOORD_{tex_coord}"), "VEC2")
            if "indices" in primitive:
                flat = cls._decode(
                    reader, primitive["indices"], "SCALAR", cls.INDEX_TYPES
                )
                corners = None if flat is None else [c[0] for c in flat]
            else:
                corners = list(range(count))
            if uvs is None or corners is None or len(uvs) != count:
                return refused
            if corners and max(corners) >= count:
                return refused
            cls._accumulate(positions, uvs, corners, up, along)

        repairs: Dict[int, Vec] = {}
        signs = directions = 0
        for vertex, shipped in enumerate(tangents):
            normal = normals[vertex]
            direction = shipped[:3]
            rebuilt = not math.sqrt(cls._dot(direction, direction)) > cls.DEGENERATE
            if rebuilt:
                direction = (
                    cls._across(normal, along[vertex])
                    or cls._across(normal, (1.0, 0.0, 0.0))
                    or cls._across(normal, (0.0, 1.0, 0.0))
                    or (1.0, 0.0, 0.0)
                )
            vote = cls._dot(cls._cross(normal, direction), up[vertex])
            w = shipped[3]
            if vote and math.isfinite(vote):
                want = 1.0 if vote > 0.0 else -1.0
            else:
                # Nothing to say which way the UVs run: the shipped sign
                # stands (a 0 or NaN one as +1, glTF's plain orientation).
                want = -1.0 if w < 0.0 else 1.0
            if rebuilt or w != want:
                repairs[vertex] = (*direction, want)
                signs += w != want
                directions += rebuilt
        return repairs, signs, directions


class GlbTangents(_GlbTangentsInternal):
    """Repair for the ``TANGENT`` attributes a GLB ships.

    Example:
        >>> GlbTangents.repair("asset.glb")
        {'primitives': 9, 'signs': 454, 'directions': 8}
    """

    @classmethod
    def repair(cls, glb: GlbTarget) -> Dict[str, int]:
        """Point every shipped tangent the way its UVs run, and give a
        zero-length one a direction.

        The rules and why they are needed are in the module docstring:
        FBX2glTF ships ``w = +1`` everywhere, which inverts green on a mirrored
        UV shell, and passes through the zero tangents a DCC writes where it
        cannot orient one.

        Parameters:
            glb: Path to a ``.glb``, modified in place, or an open session.

        Returns:
            ``{"primitives": n, "signs": n, "directions": n}`` -- primitives
            reading a rewritten TANGENT, vertices whose ``w`` was set, and
            vertices whose direction was rebuilt. A file with nothing to repair
            is not rewritten.
        """
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
            reader = GlbReader(edit)
            # TANGENT accessor -> every primitive reading it: a shared accessor
            # is ONE set of vertices, so all of their triangles vote on it.
            readers: Dict[int, List[dict]] = {}
            for mesh in gltf.get("meshes") or []:
                for primitive in mesh.get("primitives") or []:
                    tangent = (primitive.get("attributes") or {}).get("TANGENT")
                    if isinstance(tangent, int):
                        readers.setdefault(tangent, []).append(primitive)

            writes: List[Tuple[int, Vec]] = []
            summary = {"primitives": 0, "signs": 0, "directions": 0}
            for index, users in readers.items():
                repairs, signs, directions = cls._repairs(reader, index, users)
                if not repairs:
                    continue
                accessor = gltf["accessors"][index]
                view = MeshConvert._bin_view(gltf, accessor)
                base = int(view.get("byteOffset") or 0) + int(
                    accessor.get("byteOffset") or 0
                )
                stride = int(view.get("byteStride") or 0) or 16
                writes += [(base + stride * v, value) for v, value in repairs.items()]
                summary["primitives"] += len(users)
                summary["signs"] += signs
                summary["directions"] += directions
            if not writes:
                return summary

            blob = bytearray(edit.bin_data)
            for offset, value in writes:
                struct.pack_into("<4f", blob, offset, *value)
            edit.replace_rest(blob)
            logger.info(
                "Tangents: on %d primitive(s), %d handedness sign(s) set from the "
                "UVs (FBX2glTF ships w = +1, which inverts a normal map's green "
                "on every mirrored shell) and %d zero-length direction(s) rebuilt.",
                summary["primitives"],
                summary["signs"],
                summary["directions"],
            )
            return summary


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
