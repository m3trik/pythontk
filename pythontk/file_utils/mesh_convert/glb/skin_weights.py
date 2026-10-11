# !/usr/bin/python
# coding=utf-8
"""Restore the skin weights FBX2glTF hands one mesh from another.

Why this exists: FBX2glTF (v0.13.1, the latest release; its master is the
same) merges identical vertices across the WHOLE file before it splits them
back into meshes, and the equality it merges by (``RawVertex::operator==``)
compares ``jointIndices`` and ``jointWeights`` -- fields still empty while the
meshes are read -- but not ``skinningInfo``, where the weights are. So a vertex
of a skinned mesh whose position, normal, UVs and every other compared
attribute are bit-identical to a vertex of a mesh read earlier BECOMES that
vertex, weights and all, and the earlier mesh's cluster indices are then read
against the later mesh's own joint list. Duplicated rigged props are exactly
that: measured on a production assembly (2026-10-07), two electrical plugs, the
second a duplicate of the first rigged with its chain reversed -- 65 of the
second plug's 162 GLB vertices bound to the far end of its chain, its mesh
3.76 cm off by the end of its shot, while the FBX's own clusters reproduced
Maya to 0.08 cm and the plug converted on its own was exact.

What it does: screens the GLB for skinned meshes that share a vertex position
with any other mesh -- the merge needs identical positions, so a file with none
is left alone without reading the FBX -- and only then reads the source FBX's
clusters and control points for those meshes. Each GLB vertex maps back to its
control point by the position FBX2glTF computed for it (``x 0.01``, float32),
coincident points split by their UVs, and
``JOINTS_n``/``WEIGHTS_n`` are rewritten where they disagree with the clusters:
heaviest first, cut to the primitive's slots and normalized, as FBX2glTF
writes its defaults.

Refused, left as shipped and REPORTED (``refused`` / a warning -- a merged mesh
must never pass quietly): a mesh whose FBX model name is not unique or not
found, drawn under more than one skin, whose cluster joints are not all in its
skin, or whose vertices do not all map back to a control point (a model with a
geometric transform, which Maya never writes, is one); a vertex whose candidate
points carry different weights the UVs cannot tell apart; an attribute that is
not FLOAT weights over UNSIGNED_BYTE / _SHORT joints, or not in the file's own
BIN. An accessor read
by this primitive alone is rewritten in place; a shared one is replaced by a
new accessor for this primitive.

Runs first in :meth:`MeshConvert.fbx_to_glb`'s session; :meth:`audit` is the
read-only form the deliverable verifier gates on.
"""

from __future__ import annotations

import logging
import struct
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget
from pythontk.file_utils.mesh_convert.glb.reader import GlbReader

__all__ = ["GlbSkinWeights"]

logger = logging.getLogger(__name__)

#: One vertex's influences: ``[(skin joint index, weight)]``, heaviest first.
Influences = List[Tuple[int, float]]


class _GlbSkinWeightsInternal:
    """FBX reading, vertex mapping and accessor writing for :class:`GlbSkinWeights`."""

    #: FBX2glTF's ``scaleFactor`` (cm -> m), held as the float it is there.
    SCALE = struct.unpack("<f", struct.pack("<f", 0.01))[0]
    #: The FBX record names whose arrays the restore reads.
    FBX_ARRAYS = (
        "Vertices",
        "PolygonVertexIndex",
        "UV",
        "UVIndex",
        "Indexes",
        "Weights",
    )
    #: glTF component types this pass writes: joint indices, and FLOAT weights.
    JOINT_TYPES = {5121: "B", 5123: "H"}
    FLOAT = 5126
    #: A weight change below this is float noise, not a different binding.
    WEIGHT_TOLERANCE = 1e-5
    #: A UV this close to a control point's own is that UV.
    UV_TOLERANCE = 1e-5

    @staticmethod
    def _f32(value: float) -> float:
        return struct.unpack("<f", struct.pack("<f", value))[0]

    @classmethod
    def _position_key(cls, point: Sequence[float]) -> bytes:
        """A GLB ``POSITION`` element as FBX2glTF writes one for *point*."""
        return struct.pack("<3f", *(cls._f32(cls._f32(c) * cls.SCALE) for c in point))

    # ---- screen -------------------------------------------------------------

    @staticmethod
    def _skinned_meshes(gltf: Dict[str, Any]) -> Dict[int, List[dict]]:
        """``{mesh index: [skinned node]}`` for every mesh a skinned node draws."""
        out: Dict[int, List[dict]] = {}
        for node in gltf.get("nodes") or []:
            if isinstance(node.get("mesh"), int) and isinstance(node.get("skin"), int):
                out.setdefault(node["mesh"], []).append(node)
        return out

    @classmethod
    def _suspects(cls, edit: GlbEdit) -> Set[int]:
        """Skinned meshes holding a vertex position another mesh also holds.

        FBX2glTF merges only bit-identical vertices, so a skinned mesh whose
        positions no other mesh shares cannot have been handed foreign weights.
        Memory is the skinned meshes' positions alone: every other mesh is
        streamed against them.
        """
        gltf = edit.gltf
        skinned = cls._skinned_meshes(gltf)
        if not skinned:
            return set()
        owners: Dict[bytes, Set[int]] = {}
        meshes = gltf.get("meshes") or []
        for index in skinned:
            if not 0 <= index < len(meshes):
                continue
            for key in cls._mesh_positions(edit, meshes[index]):
                owners.setdefault(key, set()).add(index)
        for index, mesh in enumerate(meshes):
            if index in skinned:
                continue
            for key in cls._mesh_positions(edit, mesh):
                holders = owners.get(key)
                if holders is not None:
                    holders.add(index)
        suspects: Set[int] = set()
        for holders in owners.values():
            if len(holders) > 1:
                suspects.update(i for i in holders if i in skinned)
        return suspects

    @staticmethod
    def _mesh_positions(edit: GlbEdit, mesh: dict) -> Iterable[bytes]:
        for primitive in (mesh or {}).get("primitives") or []:
            index = (primitive.get("attributes") or {}).get("POSITION")
            if isinstance(index, int):
                yield from GlbEdit.accessor_elements(edit, index) or ()

    # ---- FBX ----------------------------------------------------------------

    @staticmethod
    def _field(record: Dict[str, Any], name: str) -> Optional[Any]:
        for child in record.get("children") or []:
            if child["name"] == name and child["props"]:
                return child["props"][0]
        return None

    @classmethod
    def _uv_sets(
        cls, geometry: Dict[str, Any], count: int
    ) -> List[Set[Tuple[float, float]]]:
        """Per control point, every UV its polygon corners carry in the FIRST
        UV layer (the one FBX2glTF reads as ``uv0``); empty sets when absent."""
        out: List[Set[Tuple[float, float]]] = [set() for _ in range(count)]
        layer = next(
            (
                c
                for c in geometry.get("children") or []
                if c["name"] == "LayerElementUV"
            ),
            None,
        )
        polygon = cls._field(geometry, "PolygonVertexIndex")
        if layer is None or not isinstance(polygon, list):
            return out
        uv = cls._field(layer, "UV")
        if not isinstance(uv, list):
            return out
        index = cls._field(layer, "UVIndex")
        mapping = bytes(cls._field(layer, "MappingInformationType") or b"")
        reference = bytes(cls._field(layer, "ReferenceInformationType") or b"")
        pairs = [(uv[i], uv[i + 1]) for i in range(0, len(uv) - 1, 2)]
        indexed = reference != b"Direct" and isinstance(index, list)
        for corner, raw in enumerate(polygon):
            point = raw if raw >= 0 else ~raw
            if not 0 <= point < count:
                continue
            slot = corner if mapping == b"ByPolygonVertex" else point
            if indexed:
                slot = index[slot] if 0 <= slot < len(index) else -1
            if 0 <= slot < len(pairs):
                out[point].add(pairs[slot])
        return out

    @classmethod
    def _fbx_meshes(cls, fbx_path: str, names: Set[str]) -> Dict[str, List[dict]]:
        """``{model name: [mesh]}`` for the skinned FBX meshes named in *names*.

        Each mesh: ``points`` (``{GLB POSITION bytes: [control point]}``),
        ``uvs`` (per point), ``weights`` (per point ``[(joint name, w)]``).
        """
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

        fbx = FbxFile.load(fbx_path, decode_arrays=cls.FBX_ARRAYS, raw_payloads=False)
        objects: Dict[Any, Dict[str, Any]] = {}
        for record in fbx.iter_objects():
            if record["props"]:
                objects[record["props"][0]] = record
        parents: Dict[Any, List[Any]] = {}
        children: Dict[Any, List[Any]] = {}
        for _kind, child, parent, _prop in fbx.connections():
            parents.setdefault(child, []).append(parent)
            children.setdefault(parent, []).append(child)

        def kind(oid: Any) -> Tuple[str, bytes]:
            record = objects.get(oid) or {"name": "", "props": []}
            sub = record["props"][2] if len(record["props"]) > 2 else b""
            return record["name"], bytes(sub) if isinstance(
                sub, (bytes, bytearray)
            ) else b""

        def name_of(oid: Any) -> Optional[str]:
            record = objects.get(oid)
            if record is None or record["name"] != "Model" or len(record["props"]) < 2:
                return None
            return FbxFile._display_name(record["props"][1])

        out: Dict[str, List[dict]] = {}
        for oid, record in objects.items():
            if record["name"] != "Model":
                continue
            name = name_of(oid)
            if name not in names:
                continue
            geometry = next(
                (c for c in children.get(oid, []) if kind(c) == ("Geometry", b"Mesh")),
                None,
            )
            if geometry is None:
                continue
            geo = objects[geometry]
            vertices = cls._field(geo, "Vertices")
            if not isinstance(vertices, list):
                continue
            count = len(vertices) // 3
            weights: List[List[Tuple[str, float]]] = [[] for _ in range(count)]
            for skin in children.get(geometry, []):
                if kind(skin) != ("Deformer", b"Skin"):
                    continue
                for cluster in children.get(skin, []):
                    if kind(cluster) != ("Deformer", b"Cluster"):
                        continue
                    link = next(
                        (name_of(c) for c in children.get(cluster, []) if name_of(c)),
                        None,
                    )
                    indexes = cls._field(objects[cluster], "Indexes") or []
                    values = cls._field(objects[cluster], "Weights") or []
                    if link is None or not isinstance(indexes, list):
                        continue
                    for point, w in zip(indexes, values):
                        if 0 <= point < count and w > 0.0:
                            weights[point].append((link, float(w)))
            if not any(weights):
                continue
            points: Dict[bytes, List[int]] = {}
            for i in range(count):
                key = cls._position_key(vertices[3 * i : 3 * i + 3])
                points.setdefault(key, []).append(i)
            out.setdefault(name, []).append(
                {
                    "points": points,
                    "uvs": cls._uv_sets(geo, count),
                    "weights": weights,
                }
            )
        return out

    # ---- plan ---------------------------------------------------------------

    @classmethod
    def _truth(
        cls, influences: List[Tuple[str, float]], joints: Dict[str, int], slots: int
    ) -> Optional[Influences]:
        """The influences FBX2glTF meant to write: heaviest first, cut to
        *slots*, normalized. ``None`` when a joint is not in the skin."""
        out: Influences = []
        for link, weight in influences:
            if link not in joints:
                return None
            out.append((joints[link], weight))
        out.sort(key=lambda pair: -pair[1])
        out = out[:slots]
        total = sum(w for _j, w in out)
        return [(j, w / total) for j, w in out] if total > 0 else out

    @classmethod
    def _uv_match(
        cls, uv: Optional[Sequence[float]], candidates: Set[Tuple[float, float]]
    ) -> bool:
        if uv is None:
            return False
        u, v = uv[0], uv[1]
        for cu, cv in candidates:
            if abs(cu - u) <= cls.UV_TOLERANCE and (
                abs(cv - v) <= cls.UV_TOLERANCE
                or abs((1.0 - cv) - v) <= cls.UV_TOLERANCE
            ):
                return True
        return False

    @classmethod
    def _same(cls, a: Influences, b: Influences) -> bool:
        da: Dict[int, float] = {}
        db: Dict[int, float] = {}
        for j, w in a:
            if w > 0:
                da[j] = da.get(j, 0.0) + w
        for j, w in b:
            if w > 0:
                db[j] = db.get(j, 0.0) + w
        return all(
            abs(da.get(j, 0.0) - db.get(j, 0.0)) <= cls.WEIGHT_TOLERANCE
            for j in set(da) | set(db)
        )

    @classmethod
    def _primitive_plan(
        cls,
        edit: GlbEdit,
        reader: GlbReader,
        primitive: dict,
        mesh: dict,
        joints: Dict[str, int],
    ) -> Optional[Tuple[List[Influences], int, int]]:
        """``(influences per vertex, rewritten, ambiguous)`` for one primitive,
        or ``None`` when it is refused (see the module docstring)."""
        attributes = primitive.get("attributes") or {}
        sets = sorted(
            int(name.split("_", 1)[1])
            for name in attributes
            if name.startswith("JOINTS_") and name.split("_", 1)[1].isdigit()
        )
        if not sets or sets != list(range(len(sets))):
            return None
        accessors = edit.gltf.get("accessors") or []
        for n in sets:
            joints_at = attributes.get(f"JOINTS_{n}")
            weights_at = attributes.get(f"WEIGHTS_{n}")
            if not (isinstance(joints_at, int) and isinstance(weights_at, int)):
                return None
            if not (
                0 <= joints_at < len(accessors) and 0 <= weights_at < len(accessors)
            ):
                return None
            joints_acc, weights_acc = accessors[joints_at], accessors[weights_at]
            if (
                joints_acc.get("componentType") not in cls.JOINT_TYPES
                or weights_acc.get("componentType") != cls.FLOAT
                or weights_acc.get("normalized")
            ):
                return None
        position_at = attributes.get("POSITION")
        if not isinstance(position_at, int):
            return None
        positions = GlbEdit.accessor_elements(edit, position_at)
        if not positions:
            return None
        uvs: Optional[list] = None  # decoded only when a vertex needs them
        current: List[Influences] = [[] for _ in positions]
        for n in sets:
            ji = reader.accessor(attributes[f"JOINTS_{n}"])
            wi = reader.accessor(attributes[f"WEIGHTS_{n}"])
            if (
                ji is None
                or wi is None
                or len(ji) != len(positions)
                or len(wi) != len(positions)
            ):
                return None
            for v, (jj, ww) in enumerate(zip(ji, wi)):
                current[v].extend((int(j), float(w)) for j, w in zip(jj, ww))
        slots = 4 * len(sets)
        planned: List[Influences] = []
        rewritten = ambiguous = 0
        for v, key in enumerate(positions):
            candidates = mesh["points"].get(key)
            if not candidates:
                return None  # the mesh is not the FBX's -- never guess
            truths = []
            for point in candidates:
                truth = cls._truth(mesh["weights"][point], joints, slots)
                if truth is None:
                    return None
                truths.append((point, truth))
            if len(truths) > 1 and not all(
                cls._same(t, truths[0][1]) for _p, t in truths
            ):
                if uvs is None and isinstance(attributes.get("TEXCOORD_0"), int):
                    uvs = reader.accessor(attributes["TEXCOORD_0"]) or []
                uv = uvs[v] if uvs and v < len(uvs) else None
                matched = [t for p, t in truths if cls._uv_match(uv, mesh["uvs"][p])]
                if not matched or not all(cls._same(t, matched[0]) for t in matched):
                    ambiguous += 1
                    planned.append(current[v])
                    continue
                truths = [(None, matched[0])]
            truth = truths[0][1]
            if not truth or cls._same(truth, current[v]):
                # An unweighted control point keeps what shipped: all-zero
                # weights are no binding at all, and invalid glTF.
                planned.append(current[v])
            else:
                planned.append(truth)
                rewritten += 1
        return planned, rewritten, ambiguous

    @classmethod
    def _mesh_plan(
        cls,
        edit: GlbEdit,
        reader: GlbReader,
        index: int,
        users: List[dict],
        fbx_meshes: Dict[str, List[dict]],
    ) -> Optional[List[Tuple[dict, Tuple[List[Influences], int, int]]]]:
        """``[(primitive, primitive plan)]`` for one suspect mesh, or ``None``
        when it is refused (see the module docstring)."""
        if len(users) != 1:  # one mesh drawn under two skins: no single truth
            return None
        node = users[0]
        candidates = fbx_meshes.get(node.get("name")) or []
        if len(candidates) != 1:  # not in the FBX, or not a unique name there
            return None
        gltf = edit.gltf
        nodes = gltf.get("nodes") or []
        skin = (gltf.get("skins") or [])[node["skin"]]
        joint_names: Dict[str, int] = {}
        for k, joint in enumerate(skin.get("joints") or []):
            joint_name = nodes[joint].get("name") if 0 <= joint < len(nodes) else None
            if joint_name is None or joint_name in joint_names:
                return None
            joint_names[joint_name] = k
        plans = []
        for primitive in gltf["meshes"][index].get("primitives") or []:
            plan = cls._primitive_plan(
                edit, reader, primitive, candidates[0], joint_names
            )
            if plan is None:
                return None
            plans.append((primitive, plan))
        return plans

    @classmethod
    def _plan(cls, edit: GlbEdit, fbx_path: str) -> Tuple[Dict[str, Any], List[tuple]]:
        """``(audit, writes)``: the :meth:`GlbSkinWeights.audit` report, and
        ``(primitive, influences per vertex)`` for each primitive to rewrite."""
        gltf = edit.gltf
        report: Dict[str, Any] = {
            "compared": [],
            "mismatched": {},
            "ambiguous": 0,
            "refused": [],
        }
        suspects = cls._suspects(edit)
        if not suspects:
            return report, []
        skinned = cls._skinned_meshes(gltf)
        names = {
            node.get("name")
            for index in suspects
            for node in skinned.get(index, [])
            if node.get("name")
        }
        fbx_meshes = cls._fbx_meshes(fbx_path, names)
        reader = GlbReader(edit)
        mismatched: Dict[str, int] = report["mismatched"]
        writes: List[tuple] = []
        for index in sorted(suspects):
            users = skinned[index]
            name = users[0].get("name") or f"mesh {index}"
            plans = cls._mesh_plan(edit, reader, index, users, fbx_meshes)
            if plans is None:
                report["refused"].append(name)
                continue
            report["compared"].append(name)
            for primitive, (influences, rewritten, ambiguous) in plans:
                report["ambiguous"] += ambiguous
                if rewritten:
                    mismatched[name] = mismatched.get(name, 0) + rewritten
                    writes.append((primitive, influences))
        return report, writes

    # ---- write --------------------------------------------------------------

    @staticmethod
    def _readers(gltf: Dict[str, Any]) -> Dict[int, int]:
        """How many core sites read each accessor (``GlbEdit.map_accessor_refs``)."""
        out: Dict[int, int] = {}

        def count(index: int) -> None:
            out[index] = out.get(index, 0) + 1

        GlbEdit.map_accessor_refs(gltf, count)
        return out

    @classmethod
    def _apply(cls, edit: GlbEdit, writes: List[tuple]) -> None:
        """Write every planned primitive's ``JOINTS_n``/``WEIGHTS_n``.

        Laid out and checked first, so a refusal (a joint past UNSIGNED_BYTE,
        an external buffer a new accessor cannot land in) raises before the
        file is touched. An accessor this primitive alone reads is rewritten
        in place; a shared one gets a new accessor for this primitive.
        """
        gltf = edit.gltf
        accessors = gltf["accessors"]
        readers = cls._readers(gltf)
        in_place: List[Tuple[int, int, str, List[tuple]]] = []
        appended: List[Tuple[dict, str, int, int, bytes]] = []
        for primitive, influences in writes:
            attributes = primitive["attributes"]
            n_sets = sum(1 for name in attributes if name.startswith("JOINTS_"))
            for n in range(n_sets):
                chunks = [
                    (list(vertex[4 * n : 4 * n + 4]) + [(0, 0.0)] * 4)[:4]
                    for vertex in influences
                ]
                rows_j = [tuple(j for j, _w in chunk) for chunk in chunks]
                rows_w = [tuple(w for _j, w in chunk) for chunk in chunks]
                for attr, rows in ((f"JOINTS_{n}", rows_j), (f"WEIGHTS_{n}", rows_w)):
                    index = attributes[attr]
                    accessor = accessors[index]
                    component = accessor["componentType"]
                    code = cls.JOINT_TYPES.get(component, "f")
                    if code == "B" and any(j > 255 for row in rows for j in row):
                        raise ValueError(f"{attr}: a joint index past UNSIGNED_BYTE")
                    view = GlbEdit.bin_view(gltf, accessor)
                    if readers.get(index, 0) == 1 and view is not None:
                        base = int(view.get("byteOffset") or 0) + int(
                            accessor.get("byteOffset") or 0
                        )
                        size = 4 * struct.calcsize("<" + code)
                        stride = int(view.get("byteStride") or 0) or size
                        in_place.append((base, stride, code, rows))
                    else:
                        payload = b"".join(
                            struct.pack(f"<4{code}", *row) for row in rows
                        )
                        appended.append(
                            (primitive, attr, component, len(rows), payload)
                        )
        buffers = gltf.get("buffers") or []
        if appended and buffers and (buffers[0] or {}).get("uri"):
            raise ValueError("the GLB's buffer is external; nothing can be appended")
        if in_place:
            blob = bytearray(edit.bin_data)
            for base, stride, code, rows in in_place:
                for v, row in enumerate(rows):
                    struct.pack_into(f"<4{code}", blob, base + stride * v, *row)
            edit.replace_rest(blob)
        if appended:
            released = {
                primitive["attributes"][attr] for primitive, attr, *_ in appended
            }
            views = GlbEdit.append_bin_views(edit, [row[-1] for row in appended])
            for (primitive, attr, component, count, _payload), view in zip(
                appended, views
            ):
                accessors.append(
                    {
                        "bufferView": view,
                        "componentType": component,
                        "count": count,
                        "type": "VEC4",
                    }
                )
                primitive["attributes"][attr] = len(accessors) - 1
            if GlbEdit.drop_orphaned_accessors(edit, released):
                GlbEdit.compact_bin(edit)
        edit.dirty = True


class GlbSkinWeights(_GlbSkinWeightsInternal):
    """Give every skinned mesh the weights its FBX clusters hold.

    Why and how: the module docstring -- FBX2glTF merges identical vertices
    across meshes and hands a later skinned mesh an earlier one's weights.

    Example:
        >>> GlbSkinWeights.restore("assembly.glb", "assembly.fbx")
        {'meshes': 1, 'vertices': 65, 'ambiguous': 0}
    """

    @classmethod
    def audit(cls, glb: GlbTarget, fbx: str) -> Dict[str, Any]:
        """What :meth:`restore` would rewrite, read-only.

        Parameters:
            glb: Path to a ``.glb`` or an open session.
            fbx: The FBX the GLB was converted from.

        Returns:
            ``{"compared": [mesh name], "mismatched": {mesh name: vertices},
            "ambiguous": n}`` -- the skinned meshes held against their FBX
            clusters (only those sharing a vertex position with another mesh:
            the rest cannot have been merged, and their FBX is never read), the
            vertices on each that disagree, those left uncompared because
            their control point could not be told apart, and ``refused``: the
            suspect meshes the FBX could not account for (see the module
            docstring) -- left as shipped, possibly wrong.
        """
        with GlbEdit.open(glb) as edit:
            report, _writes = cls._plan(edit, fbx)
            return report

    @classmethod
    def restore(cls, glb: GlbTarget, fbx: str) -> Dict[str, int]:
        """Rewrite the skin weights FBX2glTF took from another mesh.

        Parameters:
            glb: Path to a ``.glb``, modified in place, or an open session.
            fbx: The FBX the GLB was converted from.

        Returns:
            ``{"meshes": n, "vertices": n, "ambiguous": n}`` -- meshes and
            vertices rewritten, and vertices left as shipped because their
            control point could not be told apart. A file with nothing to
            restore is not rewritten, and its FBX is not read unless a skinned
            mesh shares a vertex position with another mesh.
        """
        with GlbEdit.open(glb) as edit:
            report, writes = cls._plan(edit, fbx)
            mismatched = report["mismatched"]
            summary = {
                "meshes": len(mismatched),
                "vertices": sum(mismatched.values()),
                "ambiguous": report["ambiguous"],
            }
            if report["refused"]:
                logger.warning(
                    "Skin weights: %d skinned mesh(es) share vertex positions "
                    "with another mesh but could not be mapped back to the FBX, "
                    "so FBX2glTF may have handed them another mesh's weights: %s.",
                    len(report["refused"]),
                    ", ".join(report["refused"]),
                )
            if summary["ambiguous"]:
                logger.warning(
                    "Skin weights: %d vertex(es) share a position with another "
                    "control point of different weights and could not be told "
                    "apart; left as FBX2glTF wrote them.",
                    summary["ambiguous"],
                )
            if not writes:
                return summary
            cls._apply(edit, writes)
            logger.info(
                "Skin weights: restored %d vertex(es) on %d mesh(es) from the "
                "FBX clusters (FBX2glTF merges identical vertices across meshes "
                "and hands a later skinned mesh an earlier one's weights): %s.",
                summary["vertices"],
                summary["meshes"],
                ", ".join(f"{name} ({n})" for name, n in sorted(mismatched.items())),
            )
            return summary


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
