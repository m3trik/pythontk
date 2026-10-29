# !/usr/bin/python
# coding=utf-8
"""glTF -> FBX with no DCC in the loop: :meth:`MeshConvert.gltf_to_fbx`.

A host with no glTF importer of its own (Maya) brings a ``.glb`` in through an
FBX, and the preferred converter for that is a headless Blender. This is the
route for when there is no Blender, or when the Blender at hand refuses the file
(Blender 5.1 refuses any glTF that *requires* ``KHR_texture_basisu`` -- every
KTX2 deliverable this ecosystem ships). It writes the same payload the Blender
pull writes, so one consumer serves both producers: the FBX, its images on disk
named for the map type they feed (``<material>_ORM.png`` -- the consumer
classifies by filename), and a :class:`~pythontk.HandoffManifest` sidecar with
what FBX cannot hold: the PBR maps a game shader is rebuilt from, what each
Empty was, which nodes shared a mesh, and an animated file's clock.

Carried: triangle / strip / fan primitives, welded back into shared control
points (glTF splits a vertex wherever an attribute differs; the source mesh did
not); the node hierarchy and mesh instancing; normals, two UV sets and vertex
colour; core metallic-roughness materials and their textures (PNG / JPEG copied,
KTX2 decoded through KTX-Software's ``ktx``); skins (Skin / Cluster deformers and
a bind pose); and one animation's translation / rotation / scale. Not carried,
and reported by name: morph targets, cameras, lights, further animations, and
point / line primitives. A file that requires a geometry-compression extension
(Draco, meshopt) is refused: decoding those needs a DCC.

glTF places a skinned mesh's vertices by its joints alone, ignoring the mesh
node's own transform, so that mesh lands at the FBX root at identity. Its
children still inherit the transform: a skinned node with children gets a
``<name>_pivot`` Null in its place in the hierarchy, carrying the node's
transform -- and its animation -- with the children linked under it.

glTF and FBX share a coordinate system (right-handed, Y up). Lengths are written
in centimetres with ``UnitScaleFactor`` 1, so a Maya scene in its default unit
takes the file 1:1 and the importer converts for any other -- no root node
carries a compensating scale.
"""

import base64
import bisect
import json
import logging
import math
import os
import struct
import subprocess
from array import array
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote

from pythontk.file_utils.mesh_convert._fbx_writer import (
    FBX_TICKS_PER_SECOND,
    FbxNode,
    FbxWriter,
)
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit

logger = logging.getLogger(__name__)

#: Containers :meth:`MeshConvert.gltf_to_fbx` reads.
GLTF_EXTENSIONS = (".glb", ".gltf")

#: Required extensions whose payload this converter cannot decode: the geometry
#: itself is compressed, so there is nothing to convert without a native decoder.
GLTF_UNDECODABLE_EXTENSIONS = frozenset(
    {"KHR_draco_mesh_compression", "EXT_meshopt_compression"}
)

#: Extensions read here, or safely ignored (a texture transform is dropped the
#: way the Blender route's texture manifest drops it).
_UNDERSTOOD_EXTENSIONS = frozenset(
    {
        "KHR_texture_basisu",
        "EXT_texture_webp",
        "KHR_texture_transform",
        "KHR_mesh_quantization",
        "KHR_materials_emissive_strength",
    }
)

#: glTF metres -> FBX centimetres.
_UNIT = 100.0

#: The frame rates an animation's key spacing is snapped to (see ``_infer_fps``).
_STANDARD_RATES = (12, 15, 23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 72, 96, 120)
#: Used when the keys reveal no rate: both Maya and Blender default to it.
_DEFAULT_FPS = 24.0

#: ``GlobalSettings.TimeMode`` for the rates FBX names; anything else is custom.
_TIME_MODES = {
    120: 1,
    100: 2,
    60: 3,
    50: 4,
    48: 5,
    30: 6,
    29.97: 9,
    25: 10,
    24: 11,
    23.976: 13,
    96: 15,
    72: 16,
    59.94: 17,
}
_TIME_MODE_CUSTOM = 14

#: Key interpolation flags (``FbxAnimCurveDef``).
_KEY_CONSTANT = 0x00000002
_KEY_LINEAR = 0x00000004
#: ``KeyAttrDataFloat``'s third slot is two packed int16 tangent weights
#: (1/3 each) stored in a float's bits -- the value every exporter writes.
_KEY_DATA = (
    0.0,
    0.0,
    struct.unpack("<f", struct.pack("<i", 218434821))[0],
    0.0,
)

# Map-type filename suffixes by the role sets an image feeds. Same rules as the
# Blender pull template's ``_map_suffix`` (mayatk ``templates/_import_scene.py``),
# which ``test_scene_import`` holds this to -- the consumer classifies both
# producers' files with one taxonomy.
_ROLE_BASE = "Base Color"
_ROLE_ALPHA = "Alpha"
_ROLE_METALLIC = "Metallic"
_ROLE_ROUGHNESS = "Roughness"
_ROLE_OCCLUSION = "Occlusion"
_ROLE_NORMAL = "Normal"
_ROLE_EMISSION = "Emission Color"

_IMAGE_KINDS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/ktx2": "ktx2",
    "image/webp": "webp",
}


def _map_suffix(roles) -> Optional[str]:
    """The canonical map-type name for an image feeding *roles*, or ``None``."""
    if _ROLE_BASE in roles:
        return "Base_Color"
    # Any two of occlusion / roughness / metallic in one image is an ORM:
    # R=AO, G=Roughness, B=Metallic is exactly glTF's own packing.
    if len(roles & {_ROLE_METALLIC, _ROLE_ROUGHNESS, _ROLE_OCCLUSION}) > 1:
        return "ORM"
    if _ROLE_METALLIC in roles:
        return "Metallic"
    if _ROLE_ROUGHNESS in roles:
        return "Roughness"
    if _ROLE_OCCLUSION in roles:
        return "Ambient_Occlusion"
    if _ROLE_NORMAL in roles:
        return "Normal_OpenGL"  # glTF normals are OpenGL-convention by spec
    if _ROLE_EMISSION in roles:
        return "Emissive"
    if _ROLE_ALPHA in roles:
        return "Opacity"
    return None


def _safe_name(name: str, fallback: str) -> str:
    """*name* as a node name every importer holds verbatim: each character
    outside ``[A-Za-z0-9_]`` folded to ``_``, a leading digit prefixed with
    ``_`` -- the fold the Blender pull applies at its source."""
    from pythontk.str_utils._str_utils import StrUtils

    safe = StrUtils.to_legal_name(str(name or "")) or fallback
    return ("_" + safe) if safe[:1].isdigit() else safe


def _unique(name: str, taken: set) -> str:
    """*name*, or ``name_001``... -- the first spelling *taken* lacks (added)."""
    unique, n = name, 1
    while unique in taken:
        unique, n = f"{name}_{n:03d}", n + 1
    taken.add(unique)
    return unique


# ---------------------------------------------------------------- matrix math
# Flat 16-float matrices in glTF's column-major layout: element (row r, col c)
# at ``m[c * 4 + r]``, translation at 12..14 -- the layout FBX stores too.


def _mat_mul(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [
        sum(a[k * 4 + r] * b[c * 4 + k] for k in range(4))
        for c in range(4)
        for r in range(4)
    ]


def _mat_inverse(m: Sequence[float]) -> List[float]:
    """General 4x4 inverse (identity for a singular matrix)."""
    a = [[m[c * 4 + r] for c in range(4)] for r in range(4)]
    inv = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return _identity()
        a[col], a[pivot] = a[pivot], a[col]
        inv[col], inv[pivot] = inv[pivot], inv[col]
        p = a[col][col]
        a[col] = [v / p for v in a[col]]
        inv[col] = [v / p for v in inv[col]]
        for r in range(4):
            if r != col and a[r][col]:
                f = a[r][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
                inv[r] = [x - f * y for x, y in zip(inv[r], inv[col])]
    return [inv[r][c] for c in range(4) for r in range(4)]


def _identity() -> List[float]:
    return [1.0 if i % 5 == 0 else 0.0 for i in range(16)]


def _quat_matrix3(q: Sequence[float]) -> List[List[float]]:
    """Rotation rows of quaternion ``(x, y, z, w)`` (column-vector convention)."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _trs_matrix(t, q, s) -> List[float]:
    rot = _quat_matrix3(q)
    m = _identity()
    for c in range(3):
        for r in range(3):
            m[c * 4 + r] = rot[r][c] * s[c]
    m[12], m[13], m[14] = t
    return m


def _euler_xyz(rot: List[List[float]]) -> Tuple[float, float, float]:
    """Degrees ``(x, y, z)`` for rotation rows *rot* in FBX's XYZ order
    (``R = Rz * Ry * Rx``: X applied first)."""
    sy = max(-1.0, min(1.0, -rot[2][0]))
    y = math.asin(sy)
    if abs(sy) < 0.9999999:
        x = math.atan2(rot[2][1], rot[2][2])
        z = math.atan2(rot[1][0], rot[0][0])
    else:  # gimbal: fold Z into X
        x = math.atan2(-rot[1][2], rot[1][1])
        z = 0.0
    return (math.degrees(x), math.degrees(y), math.degrees(z))


def _decompose(m: Sequence[float]):
    """``(translation, rotation rows, scale)`` of affine matrix *m*."""
    cols = [m[c * 4 : c * 4 + 3] for c in range(3)]
    scale = [math.sqrt(sum(v * v for v in col)) or 1.0 for col in cols]
    det = (
        cols[0][0] * (cols[1][1] * cols[2][2] - cols[2][1] * cols[1][2])
        - cols[1][0] * (cols[0][1] * cols[2][2] - cols[2][1] * cols[0][2])
        + cols[2][0] * (cols[0][1] * cols[1][2] - cols[1][1] * cols[0][2])
    )
    if det < 0:
        scale[0] = -scale[0]
    rot = [[cols[c][r] / scale[c] for c in range(3)] for r in range(3)]
    return list(m[12:15]), rot, scale


def _euler_near(euler: Sequence[float], prev: Sequence[float]) -> Tuple[float, ...]:
    """The spelling of *euler* closest to *prev* -- keyed rotation must not flip.

    Each XYZ rotation has a second Euler solution, and every angle is good to
    a multiple of 360; the key nearest the previous one keeps the curve
    continuous where independent extraction would jump 360 degrees.
    """
    x, y, z = euler
    best, best_cost = None, None
    for cand in ((x, y, z), (x + 180.0, 180.0 - y, z + 180.0)):
        wrapped = tuple(v + 360.0 * round((p - v) / 360.0) for v, p in zip(cand, prev))
        cost = sum(abs(v - p) for v, p in zip(wrapped, prev))
        if best_cost is None or cost < best_cost:
            best, best_cost = wrapped, cost
    return best


def _slerp(a, b, t):
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b, dot = [-v for v in b], -dot
    if dot > 0.9995:
        out = [x + (y - x) * t for x, y in zip(a, b)]
    else:
        theta = math.acos(dot)
        sa, sb = math.sin((1 - t) * theta), math.sin(t * theta)
        out = [(x * sa + y * sb) / math.sin(theta) for x, y in zip(a, b)]
    n = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / n for v in out]


class _GltfDocument(object):
    """A glTF 2.0 document -- ``.glb`` or ``.gltf`` -- with its buffers resolved."""

    def __init__(self, path: str):
        self.path = path
        self.dir = os.path.dirname(path)
        self._bin = None
        if path.lower().endswith(".glb"):
            edit = GlbEdit.read(path)
            self.gltf = edit.gltf
            self._bin = edit.bin_data
        else:
            with open(path, "r", encoding="utf-8") as fh:
                self.gltf = json.load(fh)
        self._buffers: Dict[int, Any] = {}

    def _external(self, uri: str) -> bytes:
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        with open(os.path.join(self.dir, unquote(uri)), "rb") as fh:
            return fh.read()

    def buffer(self, index: int):
        if index not in self._buffers:
            entry = (self.gltf.get("buffers") or [])[index]
            uri = entry.get("uri")
            if uri:
                data = self._external(uri)
            elif self._bin is not None:
                data = self._bin
            else:
                raise ValueError(f"buffer {index} has no data")
            self._buffers[index] = data
        return self._buffers[index]

    def _view(self, index: int) -> Tuple[Any, int, int, int]:
        view = (self.gltf.get("bufferViews") or [])[index]
        if view.get("extensions"):
            raise ValueError(
                f"bufferView {index} is compressed "
                f"({', '.join(view['extensions'])}); it needs a native decoder"
            )
        return (
            self.buffer(view.get("buffer", 0)),
            int(view.get("byteOffset", 0)),
            int(view.get("byteLength", 0)),
            int(view.get("byteStride") or 0),
        )

    def accessor(self, index: int) -> List[Tuple[float, ...]]:
        """Accessor *index* as element tuples, ``normalized`` and ``sparse`` applied."""
        acc = (self.gltf.get("accessors") or [])[index]
        ctype = acc["componentType"]
        fmt, size = GlbEdit.ACCESSOR_COMPONENT_TYPES[ctype]
        width = GlbEdit.ACCESSOR_TYPE_COUNT[acc["type"]]
        count = int(acc["count"])
        if acc.get("bufferView") is not None:
            blob, offset, _, stride = self._view(acc["bufferView"])
            base = offset + int(acc.get("byteOffset", 0))
            element = width * size
            if not stride or stride == element:
                flat = list(struct.unpack_from(f"<{count * width}{fmt}", blob, base))
            else:
                flat = []
                for i in range(count):
                    flat.extend(
                        struct.unpack_from(f"<{width}{fmt}", blob, base + i * stride)
                    )
        else:
            flat = [0] * (count * width)
        sparse = acc.get("sparse")
        if sparse:
            n = int(sparse["count"])
            ind, val = sparse["indices"], sparse["values"]
            ifmt = GlbEdit.ACCESSOR_COMPONENT_TYPES[ind["componentType"]][0]
            blob, offset, _, _ = self._view(ind["bufferView"])
            indices = struct.unpack_from(
                f"<{n}{ifmt}", blob, offset + int(ind.get("byteOffset", 0))
            )
            blob, offset, _, _ = self._view(val["bufferView"])
            values = struct.unpack_from(
                f"<{n * width}{fmt}", blob, offset + int(val.get("byteOffset", 0))
            )
            for k, i in enumerate(indices):
                flat[i * width : (i + 1) * width] = values[k * width : (k + 1) * width]
        if acc.get("normalized"):
            if ctype in (5121, 5123):
                peak = float((1 << (size * 8)) - 1)
                flat = [v / peak for v in flat]
            elif ctype in (5120, 5122):
                peak = float((1 << (size * 8 - 1)) - 1)
                flat = [max(v / peak, -1.0) for v in flat]
        it = iter(flat)
        return list(zip(*[it] * width))

    def image(self, index: int) -> Tuple[Optional[bytes], str]:
        """``(bytes, kind)`` of image *index*; kind is png / jpg / ktx2 / webp."""
        entry = (self.gltf.get("images") or [])[index]
        data = None
        if entry.get("uri"):
            try:
                data = self._external(entry["uri"])
            except OSError:
                data = None
        elif entry.get("bufferView") is not None:
            blob, offset, length, _ = self._view(entry["bufferView"])
            data = bytes(blob[offset : offset + length])
        kind = _IMAGE_KINDS.get(entry.get("mimeType", ""))
        if kind is None and data:
            if data[:8] == b"\x89PNG\r\n\x1a\n":
                kind = "png"
            elif data[:2] == b"\xff\xd8":
                kind = "jpg"
            elif data[:12] == b"\xabKTX 20\xbb\r\n\x1a\n":
                kind = "ktx2"
            elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
                kind = "webp"
        return data, kind or "png"


class _GltfToFbx(object):
    """One conversion: the FBX records and the manifest sections for a document."""

    def __init__(
        self,
        doc: _GltfDocument,
        *,
        texture_dir: str,
        include_animation: bool,
        fps: Optional[float],
    ):
        self.doc = doc
        self.gltf = doc.gltf
        self.texture_dir = texture_dir
        self.include_animation = include_animation
        self.fps = fps
        self._next_id = 1000000
        self.objects: List[FbxNode] = []
        self.links: List[Tuple[Any, ...]] = []
        self.counts: Dict[str, int] = {}
        self.skipped: Dict[str, List[str]] = {}
        self.report: Dict[str, Any] = {}
        self.take: Optional[Tuple[str, float, float]] = None
        self.scene_section: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------ bookkeeping
    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _object(self, record: FbxNode, kind: str) -> FbxNode:
        self.objects.append(record)
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return record

    def _link(self, child: int, parent: int, prop: Optional[str] = None) -> None:
        self.links.append((child, parent, prop))

    def _skip(self, what: str, name: str) -> None:
        names = self.skipped.setdefault(what, [])
        if name not in names:
            names.append(name)

    # ------------------------------------------------------------ the run
    def build(self) -> None:
        self._check_extensions()
        nodes = self.gltf.get("nodes") or []
        self.order, self.parent = self._scene_order()
        self.joints = {
            j for skin in self.gltf.get("skins") or [] for j in skin.get("joints") or []
        }
        taken: set = set()
        self.names: Dict[int, str] = {}
        for i in self.order:
            node = nodes[i]
            mesh = node.get("mesh")
            label = node.get("name") or (
                (self.gltf.get("meshes") or [])[mesh].get("name")
                if mesh is not None
                else None
            )
            self.names[i] = _unique(_safe_name(label, f"node{i}"), taken)

        self.geometries: Dict[Tuple[int, Optional[int]], Dict[str, Any]] = {}
        self.node_geometry: Dict[int, Tuple[int, Optional[int]]] = {}
        for i in self.order:
            node = nodes[i]
            if node.get("mesh") is None:
                continue
            key = (node["mesh"], node.get("skin"))
            if key not in self.geometries:
                self.geometries[key] = self._geometry(*key, label=self.names[i])
            if self.geometries[key] is not None:
                self.node_geometry[i] = key
        # Meshes that converted to nothing leave no geometry behind.
        self.geometries = {k: g for k, g in self.geometries.items() if g is not None}

        self._materials()
        self._models()
        self._skins()
        if self.include_animation:
            self._animation()
        elif self.gltf.get("animations"):
            self._skip("animations (include_animation=False)", "all")
        for what, names in self.skipped.items():
            logger.warning(
                "gltf_to_fbx: %s not carried: %s", what, ", ".join(names[:12])
            )
        self.report.update(
            skins=sum(1 for g in self.geometries.values() if g["skin"] is not None),
            nodes=len(self.order),
            meshes=len(self.geometries),
            materials=len(self.material_ids),
            skipped={k: list(v) for k, v in self.skipped.items()},
        )

    def _check_extensions(self) -> None:
        required = set(self.gltf.get("extensionsRequired") or [])
        refused = sorted(required & GLTF_UNDECODABLE_EXTENSIONS)
        if refused:
            raise ValueError(
                f"{os.path.basename(self.doc.path)} requires {', '.join(refused)} "
                "(compressed geometry); it needs a DCC importer such as Blender's."
            )
        for ext in sorted(required - _UNDERSTOOD_EXTENSIONS):
            self._skip("required extensions (ignored)", ext)

    def _scene_order(self) -> Tuple[List[int], Dict[int, int]]:
        """Nodes of the default scene, parents first, and each one's parent."""
        nodes = self.gltf.get("nodes") or []
        scenes = self.gltf.get("scenes") or []
        if scenes:
            index = self.gltf.get("scene", 0)
            roots = list(
                (scenes[index] if 0 <= index < len(scenes) else scenes[0]).get("nodes")
                or []
            )
        else:
            children = {c for n in nodes for c in n.get("children") or []}
            roots = [i for i in range(len(nodes)) if i not in children]
        order: List[int] = []
        parent: Dict[int, int] = {}
        seen = set()
        stack = [(r, None) for r in reversed(roots)]
        while stack:
            i, p = stack.pop()
            if i in seen or not 0 <= i < len(nodes):
                continue
            seen.add(i)
            order.append(i)
            if p is not None:
                parent[i] = p
            for c in reversed(nodes[i].get("children") or []):
                stack.append((c, i))
        return order, parent

    # ------------------------------------------------------------ geometry
    @staticmethod
    def _triangles(indices: Sequence[int], mode: int) -> List[Tuple[int, int, int]]:
        if mode == 4:
            return [tuple(indices[i : i + 3]) for i in range(0, len(indices) - 2, 3)]
        if mode == 5:  # strip: alternate winding
            return [
                (indices[i], indices[i + 1], indices[i + 2])
                if i % 2 == 0
                else (indices[i + 1], indices[i], indices[i + 2])
                for i in range(len(indices) - 2)
            ]
        return [  # fan
            (indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)
        ]

    def _geometry(self, mesh_index: int, skin_index: Optional[int], label: str):
        doc = self.doc
        mesh = (self.gltf.get("meshes") or [])[mesh_index]
        mesh_name = mesh.get("name") or label
        prims = []
        for prim in mesh.get("primitives") or []:
            if prim.get("mode", 4) not in (4, 5, 6):
                self._skip("point/line primitives", mesh_name)
            elif "POSITION" not in (prim.get("attributes") or {}):
                self._skip("primitives without positions", mesh_name)
            else:
                prims.append(prim)
                if prim.get("targets"):
                    self._skip("morph targets", mesh_name)
        if not prims:
            return None
        attrs = [p["attributes"] for p in prims]
        uv_sets = 0
        while uv_sets < 2 and any(f"TEXCOORD_{uv_sets}" in a for a in attrs):
            uv_sets += 1
        has_color = any("COLOR_0" in a for a in attrs)
        skinned = skin_index is not None and any("JOINTS_0" in a for a in attrs)

        welded: Dict[Any, int] = {}
        vertices = array("d")
        polys = array("i")
        normals = array("d")
        poly_slots = array("i")
        uvs = [(array("d"), {}, array("i")) for _ in range(uv_sets)]
        colors = (array("d"), {}, array("i")) if has_color else None
        slots: List[Optional[int]] = []
        influences: Dict[int, Dict[int, float]] = {}
        degenerate = 0

        def indexed(table, value):
            values, lookup, index = table
            at = lookup.get(value)
            if at is None:
                at = lookup[value] = len(lookup)
                values.extend(value)
            index.append(at)

        for prim in prims:
            a = prim["attributes"]
            positions = doc.accessor(a["POSITION"])
            prim_normals = doc.accessor(a["NORMAL"]) if "NORMAL" in a else None
            prim_uvs = [
                doc.accessor(a[f"TEXCOORD_{k}"]) if f"TEXCOORD_{k}" in a else None
                for k in range(uv_sets)
            ]
            prim_colors = doc.accessor(a["COLOR_0"]) if "COLOR_0" in a else None
            skin_sets = []
            if skinned:
                k = 0
                while f"JOINTS_{k}" in a and f"WEIGHTS_{k}" in a:
                    skin_sets.append(
                        (
                            doc.accessor(a[f"JOINTS_{k}"]),
                            doc.accessor(a[f"WEIGHTS_{k}"]),
                        )
                    )
                    k += 1

            local = []
            for v, pos in enumerate(positions):
                weights = None
                if skinned:
                    acc: Dict[int, float] = {}
                    for joints, ws in skin_sets:
                        for j, w in zip(joints[v], ws[v]):
                            if w > 0.0:
                                acc[int(j)] = acc.get(int(j), 0.0) + w
                    weights = tuple(sorted(acc.items()))
                key = (pos, weights)
                cp = welded.get(key)
                if cp is None:
                    cp = welded[key] = len(welded)
                    vertices.extend((pos[0] * _UNIT, pos[1] * _UNIT, pos[2] * _UNIT))
                    for j, w in weights or ():
                        influences.setdefault(j, {})[cp] = w
                local.append(cp)

            if prim.get("indices") is not None:
                indices = [e[0] for e in doc.accessor(prim["indices"])]
            else:
                indices = list(range(len(positions)))
            material = prim.get("material")
            if material not in slots:
                slots.append(material)
            slot = slots.index(material)
            for tri in self._triangles(indices, prim.get("mode", 4)):
                c = [local[i] for i in tri]
                if c[0] == c[1] or c[1] == c[2] or c[0] == c[2]:
                    degenerate += 1
                    continue
                polys.extend((c[0], c[1], -c[2] - 1))
                poly_slots.append(slot)
                if prim_normals is not None:
                    for i in tri:
                        normals.extend(prim_normals[i])
                else:  # glTF: no normals means flat shading
                    p0, p1, p2 = (positions[i] for i in tri)
                    e1 = [p1[k] - p0[k] for k in range(3)]
                    e2 = [p2[k] - p0[k] for k in range(3)]
                    n = [
                        e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0],
                    ]
                    length = math.sqrt(sum(x * x for x in n)) or 1.0
                    normals.extend([x / length for x in n] * 3)
                for k, table in enumerate(uvs):
                    source = prim_uvs[k]
                    for i in tri:
                        # glTF's V runs down the image, FBX's up.
                        indexed(
                            table,
                            (source[i][0], 1.0 - source[i][1])
                            if source
                            else (0.0, 0.0),
                        )
                if colors is not None:
                    for i in tri:
                        rgba = tuple(prim_colors[i]) if prim_colors else (1.0, 1.0, 1.0)
                        indexed(colors, rgba + (1.0,) * (4 - len(rgba)))
        if degenerate:
            logger.debug(
                "gltf_to_fbx: %s: %d degenerate triangle(s) dropped",
                mesh_name,
                degenerate,
            )
        if not polys:
            return None

        geo_id = self._id()
        geo = self._object(
            FbxNode(
                "Geometry",
                FbxWriter.i64(geo_id),
                f"{mesh_name}\x00\x01Geometry",
                "Mesh",
            ),
            "Geometry",
        )
        geo.add("GeometryVersion", 124)
        geo.add("Vertices", FbxWriter.array("d", vertices))
        geo.add("PolygonVertexIndex", FbxWriter.array("i", polys))
        layer = geo.add("LayerElementNormal", 0)
        layer.add("Version", 101)
        layer.add("Name", "")
        layer.add("MappingInformationType", "ByPolygonVertex")
        layer.add("ReferenceInformationType", "Direct")
        layer.add("Normals", FbxWriter.array("d", normals))
        for k, (values, _, index) in enumerate(uvs):
            layer = geo.add("LayerElementUV", k)
            layer.add("Version", 101)
            layer.add("Name", f"map{k + 1}")
            layer.add("MappingInformationType", "ByPolygonVertex")
            layer.add("ReferenceInformationType", "IndexToDirect")
            layer.add("UV", FbxWriter.array("d", values))
            layer.add("UVIndex", FbxWriter.array("i", index))
        if colors is not None:
            layer = geo.add("LayerElementColor", 0)
            layer.add("Version", 101)
            layer.add("Name", "colorSet1")
            layer.add("MappingInformationType", "ByPolygonVertex")
            layer.add("ReferenceInformationType", "IndexToDirect")
            layer.add("Colors", FbxWriter.array("d", colors[0]))
            layer.add("ColorIndex", FbxWriter.array("i", colors[2]))
        has_materials = any(s is not None for s in slots)
        if has_materials:
            layer = geo.add("LayerElementMaterial", 0)
            layer.add("Version", 101)
            layer.add("Name", "")
            uniform = len(slots) == 1
            layer.add("MappingInformationType", "AllSame" if uniform else "ByPolygon")
            layer.add("ReferenceInformationType", "IndexToDirect")
            layer.add("Materials", FbxWriter.array("i", [0] if uniform else poly_slots))
        layer = geo.add("Layer", 0)
        layer.add("Version", 100)
        kinds = ["LayerElementNormal"]
        kinds += ["LayerElementUV"] if uv_sets else []
        kinds += ["LayerElementColor"] if colors is not None else []
        kinds += ["LayerElementMaterial"] if has_materials else []
        for kind in kinds:
            element = layer.add("LayerElement")
            element.add("Type", kind)
            element.add("TypedIndex", 0)
        for k in range(1, uv_sets):
            layer = geo.add("Layer", k)
            layer.add("Version", 100)
            element = layer.add("LayerElement")
            element.add("Type", "LayerElementUV")
            element.add("TypedIndex", k)
        return {
            "id": geo_id,
            "slots": slots if has_materials else [],
            "skin": skin_index if skinned else None,
            "influences": influences,
            "label": mesh_name,
        }

    # ------------------------------------------------------------ materials
    def _image_of(self, texture_ref: Optional[dict]) -> Optional[int]:
        """The image a material's texture reference resolves to -- the core
        ``source`` when it is one any reader opens, else a container extension's."""
        if not isinstance(texture_ref, dict):
            return None
        textures = self.gltf.get("textures") or []
        index = texture_ref.get("index")
        if not isinstance(index, int) or not 0 <= index < len(textures):
            return None
        texture = textures[index]
        ext = texture.get("extensions") or {}
        for key in ("KHR_texture_basisu", "EXT_texture_webp"):
            if texture.get("source") is None and key in ext:
                return ext[key].get("source")
        return texture.get("source")

    def _materials(self) -> None:
        materials = self.gltf.get("materials") or []
        used = []
        for geo in self.geometries.values():
            for slot in geo["slots"]:
                if slot not in used:
                    used.append(slot)
        taken: set = set()
        self.material_names: Dict[Optional[int], str] = {}
        for index in used:
            label = materials[index].get("name") if index is not None else "default"
            self.material_names[index] = _unique(
                _safe_name(label, f"material{index}"), taken
            )

        # Which images each material uses, and in which roles.
        refs: Dict[Optional[int], List[Tuple[str, int, set]]] = {}
        roles: Dict[Tuple[str, int], set] = {}
        for index in used:
            if index is None:
                continue
            mat = materials[index]
            pbr = mat.get("pbrMetallicRoughness") or {}
            base_roles = {_ROLE_BASE}
            if mat.get("alphaMode", "OPAQUE") != "OPAQUE":
                base_roles.add(_ROLE_ALPHA)
            for slot, ref, role_set in (
                ("base", pbr.get("baseColorTexture"), base_roles),
                (
                    "mr",
                    pbr.get("metallicRoughnessTexture"),
                    {_ROLE_METALLIC, _ROLE_ROUGHNESS},
                ),
                ("normal", mat.get("normalTexture"), {_ROLE_NORMAL}),
                ("occlusion", mat.get("occlusionTexture"), {_ROLE_OCCLUSION}),
                ("emissive", mat.get("emissiveTexture"), {_ROLE_EMISSION}),
            ):
                image = self._image_of(ref)
                if image is None:
                    continue
                refs.setdefault(index, []).append((slot, image, role_set))
                roles.setdefault((self.material_names[index], image), set()).update(
                    role_set
                )
        # First claim names the file: one image is one file, whatever shares it.
        classified: Dict[int, str] = {}
        for (name, image), role_set in roles.items():
            suffix = _map_suffix(role_set)
            if suffix and image not in classified:
                classified[image] = f"{name}_{suffix}"
        files = self._write_images(
            sorted({image for entries in refs.values() for _, image, _ in entries}),
            classified,
        )

        self.material_ids: Dict[Optional[int], int] = {}
        self.manifest_materials: List[Dict[str, Any]] = []
        textures: Dict[str, int] = {}
        for index in used:
            mat = materials[index] if index is not None else {}
            name = self.material_names[index]
            pbr = mat.get("pbrMetallicRoughness") or {}
            base = list(pbr.get("baseColorFactor") or (0.8, 0.8, 0.8, 1.0))
            strength = (
                (mat.get("extensions") or {})
                .get("KHR_materials_emissive_strength", {})
                .get("emissiveStrength", 1.0)
            )
            emissive = [v * strength for v in mat.get("emissiveFactor") or (0, 0, 0)]
            mat_id = self._id()
            record = self._object(
                FbxNode(
                    "Material", FbxWriter.i64(mat_id), f"{name}\x00\x01Material", ""
                ),
                "Material",
            )
            record.add("Version", 102)
            record.add("ShadingModel", "lambert")
            record.add("MultiLayer", 0)
            props = record.add("Properties70")
            props.p("DiffuseColor", "Color", "", "A", *[float(v) for v in base[:3]])
            props.p("DiffuseFactor", "Number", "", "A", 1.0)
            if any(emissive):
                props.p(
                    "EmissiveColor", "Color", "", "A", *[float(v) for v in emissive]
                )
                props.p("EmissiveFactor", "Number", "", "A", 1.0)
            if mat.get("alphaMode") == "BLEND" and base[3] < 1.0:
                props.p("TransparentColor", "Color", "", "A", 1.0, 1.0, 1.0)
                props.p("TransparencyFactor", "Number", "", "A", 1.0 - float(base[3]))
            self.material_ids[index] = mat_id

            entries = refs.get(index)
            if not entries:
                continue
            listed: List[str] = []
            slot_paths: Dict[str, str] = {}
            for slot, image, _ in entries:
                path = files.get(image)
                if not path:
                    continue
                if path not in listed:
                    listed.append(path)
                channel = {
                    "base": "baseColor",
                    "normal": "normal",
                    "emissive": "emission",
                }.get(slot)
                if channel:
                    slot_paths[channel] = path
                if slot == "base" and mat.get("alphaMode", "OPAQUE") != "OPAQUE":
                    slot_paths["opacity"] = path
                prop = {
                    "base": "DiffuseColor",
                    "normal": "NormalMap",
                    "emissive": "EmissiveColor",
                }.get(slot)
                if prop:
                    if path not in textures:
                        textures[path] = self._texture(path)
                    self._link(textures[path], mat_id, prop)
            self.manifest_materials.append(
                {
                    "name": name,
                    "shader_type": "gltf_pbr",
                    "fbx_material": name,
                    "objects": [],  # filled per model in _models
                    "files": listed,
                    "slots": slot_paths,
                }
            )
        self.report["textures"] = sum(1 for p in files.values() if p)
        self.report["textures_missing"] = sum(1 for p in files.values() if not p)

    def _write_images(
        self, images: List[int], classified: Dict[int, str]
    ) -> Dict[int, Optional[str]]:
        """Write each image to the texture dir as a file the host reads.

        PNG and JPEG are copied as they are. KTX2 is decoded through
        KTX-Software's ``ktx``, every map at once (:meth:`Ktx2Encoder.decode_all`),
        and WebP through Pillow, else FFmpeg: Maya reads neither container (a file
        node pointed at a ``.webp`` reports 0 x 0). Returns ``{image: path or
        None}``; a ``None`` is reported under ``textures (not decoded)``.
        """
        from pythontk.core_utils.app_launcher import AppLauncher
        from pythontk.file_utils.temp_artifacts import TempArtifacts
        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        out: Dict[int, Optional[str]] = {}
        if not images:
            return out
        os.makedirs(self.texture_dir, exist_ok=True)
        entries = self.gltf.get("images") or []
        taken: set = set()
        failed: List[str] = []
        pending: List[Tuple[int, str, str, str]] = []  # image, stem, .ktx2, .png
        tmp = TempArtifacts("gltf_images", policy="scoped")
        try:
            for image in images:
                entry = entries[image] if 0 <= image < len(entries) else {}
                stem = classified.get(image) or _safe_name(
                    os.path.splitext(
                        os.path.basename(entry.get("name") or entry.get("uri") or "")
                    )[0],
                    f"image{image}",
                )
                data, kind = self.doc.image(image)
                if not data:
                    self._skip("unreadable images", stem)
                    out[image] = None
                    continue
                ext = kind if kind in ("png", "jpg") else "png"
                filename, n = f"{stem}.{ext}", 1
                while filename.lower() in taken:
                    filename, n = f"{stem}_{n}.{ext}", n + 1
                taken.add(filename.lower())
                path = os.path.join(self.texture_dir, filename)
                out[image] = path
                try:
                    if kind == "ktx2":
                        source = tmp.path(extension=".ktx2")
                        with open(source, "wb") as fh:
                            fh.write(data)
                        pending.append((image, stem, source, path))
                    elif kind == "webp":
                        self._decode_webp(data, path, tmp)
                    else:
                        with open(path, "wb") as fh:
                            fh.write(data)
                except (OSError, RuntimeError) as e:
                    failed.append(f"{stem} ({e})")
                    out[image] = None
            results = Ktx2Encoder.decode_all([(src, dst) for _, _, src, dst in pending])
            for (image, stem, _, _), result in zip(pending, results):
                if isinstance(result, Exception):
                    failed.append(f"{stem} ({result})")
                    out[image] = None
        finally:
            tmp.cleanup()
        if failed:
            self.skipped.setdefault("textures (not decoded)", []).extend(failed)
        return {
            image: AppLauncher.ansi_safe_path(path).replace("\\", "/") if path else None
            for image, path in out.items()
        }

    @staticmethod
    def _decode_webp(data: bytes, path: str, tmp) -> None:
        """Write WebP *data* to *path* as a PNG.

        Pillow where it imports, else FFmpeg (``VidUtils.resolve_ffmpeg``): mayapy
        ships no Pillow. The two decode identically (measured: a production
        lightmap atlas, every pixel equal).
        """
        try:
            from PIL import Image
        except ImportError:
            Image = None
        if Image is not None:
            import io

            with Image.open(io.BytesIO(data)) as im:
                im.save(path, format="PNG")
            return
        from pythontk.core_utils.app_launcher import AppLauncher
        from pythontk.vid_utils._vid_utils import VidUtils

        ffmpeg = VidUtils.resolve_ffmpeg(required=False)
        if not ffmpeg:
            raise RuntimeError("decoding WebP needs Pillow or FFmpeg")
        source = tmp.path(extension=".webp")
        with open(source, "wb") as fh:
            fh.write(data)
        try:
            result = AppLauncher.run(
                ffmpeg,
                args=["-y", "-loglevel", "error", "-i", source, path],
                timeout=120,
                hide_window=True,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg timed out decoding WebP") from None
        if result.returncode != 0 or not os.path.isfile(path):
            lines = ((result.stderr or "") + (result.stdout or "")).strip().splitlines()
            reason = lines[-1] if lines else f"exit {result.returncode}"
            raise RuntimeError(f"FFmpeg could not decode WebP: {reason}")

    def _texture(self, path: str) -> int:
        """A Texture + Video pair for the file at *path*; returns the texture id."""
        stem = os.path.splitext(os.path.basename(path))[0]
        tex_id, video_id = self._id(), self._id()
        texture = self._object(
            FbxNode("Texture", FbxWriter.i64(tex_id), f"{stem}\x00\x01Texture", ""),
            "Texture",
        )
        texture.add("Type", "TextureVideoClip")
        texture.add("Version", 202)
        texture.add("TextureName", f"{stem}\x00\x01Texture")
        texture.add("Properties70").p("UseMaterial", "bool", "", "", 1)
        texture.add("Media", f"{stem}\x00\x01Video")
        texture.add("FileName", path)
        texture.add("RelativeFilename", path)
        texture.add("ModelUVTranslation", 0.0, 0.0)
        texture.add("ModelUVScaling", 1.0, 1.0)
        texture.add("Texture_Alpha_Source", "None")
        texture.add("Cropping", 0, 0, 0, 0)
        video = self._object(
            FbxNode("Video", FbxWriter.i64(video_id), f"{stem}\x00\x01Video", "Clip"),
            "Video",
        )
        video.add("Type", "Clip")
        video.add("Properties70").p("Path", "KString", "XRefUrl", "", path)
        video.add("UseMipMap", 0)
        video.add("Filename", path)
        video.add("RelativeFilename", path)
        self._link(video_id, tex_id)
        return tex_id

    # ------------------------------------------------------------ models
    def _rest(self, i: int) -> Tuple[List[float], Tuple[float, ...], List[float]]:
        """Node *i*'s local ``(translation (cm), XYZ Euler degrees, scale)``."""
        node = (self.gltf.get("nodes") or [])[i]
        if "matrix" in node:
            t, rot, s = _decompose(node["matrix"])
        else:
            t = list(node.get("translation") or (0.0, 0.0, 0.0))
            rot = _quat_matrix3(node.get("rotation") or (0.0, 0.0, 0.0, 1.0))
            s = list(node.get("scale") or (1.0, 1.0, 1.0))
        # The spelling nearest zero: (360, 90, 360) is (0, 90, 0).
        euler = _euler_near(_euler_xyz(rot), (0.0, 0.0, 0.0))
        return [v * _UNIT for v in t], euler, s

    def _model(self, name: str, kind: str, parent_id: int, t, euler, s) -> int:
        """A *kind* (Mesh / LimbNode / Null) ``Model`` at local ``t, euler, s``,
        linked under *parent_id* (0: the root), with the node attribute a
        LimbNode or Null carries. Returns the model's id."""
        model_id = self._id()
        model = self._object(
            FbxNode("Model", FbxWriter.i64(model_id), f"{name}\x00\x01Model", kind),
            "Model",
        )
        model.add("Version", 232)
        props = model.add("Properties70")
        props.p("RotationActive", "bool", "", "", 1)
        props.p("InheritType", "enum", "", "", 1)
        props.p("DefaultAttributeIndex", "int", "Integer", "", 0)
        props.p("Lcl Translation", "Lcl Translation", "", "A", *[float(v) for v in t])
        props.p("Lcl Rotation", "Lcl Rotation", "", "A", *[float(v) for v in euler])
        props.p("Lcl Scaling", "Lcl Scaling", "", "A", *[float(v) for v in s])
        model.add("MultiLayer", 0)
        model.add("MultiTake", 0)
        model.add("Shading", True)
        model.add("Culling", "CullingOff")
        self._link(model_id, parent_id)
        if kind == "Mesh":
            return model_id
        attr_id = self._id()
        attr = self._object(
            FbxNode(
                "NodeAttribute", FbxWriter.i64(attr_id), "\x00\x01NodeAttribute", kind
            ),
            "NodeAttribute",
        )
        if kind == "LimbNode":
            attr.add("Properties70").p("Size", "double", "Number", "", 33.333333)
            attr.add("TypeFlags", "Skeleton")
        else:
            attr.add("TypeFlags", "Null")
            self.empties.append({"name": name, "display_type": "PLAIN_AXES"})
        self._link(attr_id, model_id)
        return model_id

    def _transform_model(self, i: Optional[int]) -> Optional[int]:
        """The model carrying node *i*'s own transform: a skinned mesh node's
        pivot (``None`` when it has none), else the node's model."""
        if i in self.skinned_nodes:
            return self.pivot_ids.get(i)
        return self.model_ids.get(i)

    def _models(self) -> None:
        nodes = self.gltf.get("nodes") or []
        self.model_ids: Dict[int, int] = {}
        self.pivot_ids: Dict[int, int] = {}
        self.empties: List[Dict[str, str]] = []
        self.skinned_nodes = {
            i
            for i, key in self.node_geometry.items()
            if self.geometries[key]["skin"] is not None
        }
        has_children = set(self.parent.values())
        taken = set(self.names.values())
        by_material = {e["fbx_material"]: e for e in self.manifest_materials}
        for i in self.order:
            name = self.names[i]
            if i in self.node_geometry:
                kind = "Mesh"
            elif i in self.joints:
                kind = "LimbNode"
            else:
                kind = "Null"
            parent = self.parent.get(i)
            parent_id = 0 if parent is None else self._transform_model(parent)
            if i in self.skinned_nodes:
                # glTF ignores a skinned mesh node's own transform: the joints
                # place every vertex. It lands at the root, at identity.
                identity = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
                model_id = self._model(name, kind, 0, *identity)
            else:
                model_id = self._model(name, kind, parent_id, *self._rest(i))
            self.model_ids[i] = model_id
            if kind == "Mesh":
                geo = self.geometries[self.node_geometry[i]]
                self._link(geo["id"], model_id)
                for slot in geo["slots"]:
                    self._link(self.material_ids[slot], model_id)
                    entry = by_material.get(self.material_names[slot])
                    if entry is not None and name not in entry["objects"]:
                        entry["objects"].append(name)
            if i in self.skinned_nodes and i in has_children:
                # Its children still inherit that transform: a Null pivot in
                # its place carries it (and its animation), and they link to it.
                pivot = _unique(f"{name}_pivot", taken)
                self.pivot_ids[i] = self._model(
                    pivot, "Null", parent_id, *self._rest(i)
                )
        for node_index in self.order:
            node = nodes[node_index]
            if node.get("camera") is not None:
                self._skip("cameras", self.names[node_index])
            if "KHR_lights_punctual" in (node.get("extensions") or {}):
                self._skip("lights", self.names[node_index])

    # ------------------------------------------------------------ skins
    def _skins(self) -> None:
        skins = self.gltf.get("skins") or []
        pose_nodes: Dict[int, List[float]] = {}
        unit = [_UNIT if i in (12, 13, 14) else 1.0 for i in range(16)]
        for key, geo in self.geometries.items():
            if geo["skin"] is None:
                continue
            skin = skins[geo["skin"]]
            joints = skin.get("joints") or []
            ibms = (
                self.doc.accessor(skin["inverseBindMatrices"])
                if skin.get("inverseBindMatrices") is not None
                else [tuple(_identity())] * len(joints)
            )
            skin_id = self._id()
            deformer = self._object(
                FbxNode(
                    "Deformer",
                    FbxWriter.i64(skin_id),
                    f"{geo['label']}\x00\x01Deformer",
                    "Skin",
                ),
                "Deformer",
            )
            deformer.add("Version", 101)
            deformer.add("Link_DeformAcuracy", 50.0)
            self._link(skin_id, geo["id"])
            for slot, joint in enumerate(joints):
                if joint not in self.model_ids:
                    continue
                ibm = [v * u for v, u in zip(ibms[slot], unit)]
                link = _mat_inverse(ibm)
                pose_nodes[self.model_ids[joint]] = link
                weights = geo["influences"].get(slot)
                if not weights:
                    continue
                cluster_id = self._id()
                cluster = self._object(
                    FbxNode(
                        "Deformer",
                        FbxWriter.i64(cluster_id),
                        f"{self.names[joint]}\x00\x01SubDeformer",
                        "Cluster",
                    ),
                    "Deformer",
                )
                cluster.add("Version", 100)
                cluster.add("UserData", "", "")
                cps = sorted(weights)
                cluster.add("Indexes", FbxWriter.array("i", cps))
                cluster.add("Weights", FbxWriter.array("d", [weights[c] for c in cps]))
                # Transform: the mesh in the joint's bind space (the mesh sits
                # at identity, so the inverse bind matrix itself). TransformLink:
                # the joint's bind-time world.
                cluster.add("Transform", FbxWriter.array("d", ibm))
                cluster.add("TransformLink", FbxWriter.array("d", link))
                self._link(cluster_id, skin_id)
                self._link(self.model_ids[joint], cluster_id)
            for i, node_key in self.node_geometry.items():
                if node_key == key:
                    pose_nodes[self.model_ids[i]] = _identity()
        if not pose_nodes:
            return
        pose_id = self._id()
        pose = self._object(
            FbxNode("Pose", FbxWriter.i64(pose_id), "BindPose\x00\x01Pose", "BindPose"),
            "Pose",
        )
        pose.add("Type", "BindPose")
        pose.add("Version", 100)
        pose.add("NbPoseNodes", len(pose_nodes))
        for model_id, matrix in pose_nodes.items():
            entry = pose.add("PoseNode")
            entry.add("Node", FbxWriter.i64(model_id))
            entry.add("Matrix", FbxWriter.array("d", matrix))

    # ------------------------------------------------------------ animation
    @staticmethod
    def _infer_fps(times: Sequence[float]) -> Optional[float]:
        """The frame rate the key spacing was baked at, when it is a standard one."""
        deltas = sorted(b - a for a, b in zip(times, times[1:]) if b - a > 1e-6)
        if not deltas:
            return None
        rate = 1.0 / deltas[len(deltas) // 2]
        best = min(_STANDARD_RATES, key=lambda r: abs(r - rate))
        return float(best) if abs(best - rate) <= best * 0.005 else None

    @staticmethod
    def _sample(times, values, interp: str, path: str, t: float) -> List[float]:
        """Evaluate one sampler at *t* (per the glTF spec, clamped at the ends)."""
        n = len(times)
        cubic = interp == "CUBICSPLINE"

        def value(k):
            return list(values[3 * k + 1] if cubic else values[k])

        if t <= times[0]:
            return value(0)
        if t >= times[-1]:
            return value(n - 1)
        k = bisect.bisect_right(times, t) - 1
        t0, t1 = times[k], times[k + 1]
        span = t1 - t0
        u = (t - t0) / span if span > 0 else 0.0
        if interp == "STEP":
            return value(k)
        if cubic:
            v0, b0 = values[3 * k + 1], values[3 * k + 2]
            a1, v1 = values[3 * (k + 1)], values[3 * (k + 1) + 1]
            u2, u3 = u * u, u * u * u
            h = (2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u, -2 * u3 + 3 * u2, u3 - u2)
            out = [
                h[0] * p0 + h[1] * span * m0 + h[2] * p1 + h[3] * span * m1
                for p0, m0, p1, m1 in zip(v0, b0, v1, a1)
            ]
            if path == "rotation":
                length = math.sqrt(sum(x * x for x in out)) or 1.0
                out = [x / length for x in out]
            return out
        if path == "rotation":
            return _slerp(value(k), value(k + 1), u)
        return [x + (y - x) * u for x, y in zip(value(k), value(k + 1))]

    def _keys(self, path, times, values, interp, fps):
        """``(seconds, xyz values, constant)`` for one channel, FBX-ready."""
        step = 1.0 / fps
        gaps = [b - a for a, b in zip(times, times[1:])]
        resample = interp == "CUBICSPLINE" or (
            path == "rotation"
            and interp == "LINEAR"
            and gaps
            and max(gaps) > step * 1.01
        )
        if resample:
            first, last = times[0], times[-1]
            count = int(math.floor((last - first) / step + 1e-6))
            key_times = [first + i * step for i in range(count + 1)]
            if last - key_times[-1] > 1e-6:
                key_times.append(last)
            samples = [self._sample(times, values, interp, path, t) for t in key_times]
        else:
            key_times = list(times)
            samples = [list(values[k]) for k in range(len(times))]
        if path == "translation":
            out = [tuple(v * _UNIT for v in s) for s in samples]
        elif path == "rotation":
            out, prev = [], None
            for q in samples:
                e = _euler_xyz(_quat_matrix3(q))
                e = _euler_near(e, prev if prev is not None else (0.0, 0.0, 0.0))
                out.append(e)
                prev = e
        else:
            out = [tuple(s) for s in samples]
        return key_times, out, interp == "STEP"

    def _animation(self) -> None:
        animations = self.gltf.get("animations") or []
        if not animations:
            return
        anim = animations[0]
        for extra in animations[1:]:
            self._skip(
                "further animations (the first is converted)", extra.get("name") or "?"
            )
        samplers = anim.get("samplers") or []
        # Keyed by the model each channel drives: a skinned mesh node's pivot,
        # and none for a skinned node without one (its transform moves nothing).
        tracks: Dict[int, Dict[str, Tuple[List[float], list, str]]] = {}
        for channel in anim.get("channels") or []:
            target = channel.get("target") or {}
            node, path = target.get("node"), target.get("path")
            if node not in self.model_ids:
                continue
            if path not in ("translation", "rotation", "scale"):
                self._skip("animated morph weights", self.names.get(node, str(node)))
                continue
            model_id = self._transform_model(node)
            if model_id is None:
                continue
            sampler = samplers[channel["sampler"]]
            times = [t[0] for t in self.doc.accessor(sampler["input"])]
            if not times:
                continue
            values = self.doc.accessor(sampler["output"])
            tracks.setdefault(model_id, {})[path] = (
                times,
                values,
                sampler.get("interpolation", "LINEAR"),
            )
        if not tracks:
            return
        all_times = sorted(
            {
                t
                for paths in tracks.values()
                for times, _, _ in paths.values()
                for t in times
            }
        )
        fps = self.fps or self._infer_fps(all_times)
        if fps is None:
            fps = _DEFAULT_FPS
            logger.info(
                "gltf_to_fbx: key spacing names no frame rate; using %g fps.", fps
            )
        start, stop = all_times[0], all_times[-1]
        take = _safe_name(anim.get("name"), "Take_001")

        stack_id, layer_id = self._id(), self._id()
        stack = self._object(
            FbxNode(
                "AnimationStack",
                FbxWriter.i64(stack_id),
                f"{take}\x00\x01AnimStack",
                "",
            ),
            "AnimationStack",
        )
        props = stack.add("Properties70")
        for key in ("LocalStart", "ReferenceStart"):
            props.p(
                key,
                "KTime",
                "Time",
                "",
                FbxWriter.i64(round(start * FBX_TICKS_PER_SECOND)),
            )
        for key in ("LocalStop", "ReferenceStop"):
            props.p(
                key,
                "KTime",
                "Time",
                "",
                FbxWriter.i64(round(stop * FBX_TICKS_PER_SECOND)),
            )
        self._object(
            FbxNode(
                "AnimationLayer",
                FbxWriter.i64(layer_id),
                "BaseLayer\x00\x01AnimLayer",
                "",
            ),
            "AnimationLayer",
        )
        self._link(layer_id, stack_id)

        keys = 0
        for model_id, paths in tracks.items():
            for path, prop, short in (
                ("translation", "Lcl Translation", "T"),
                ("rotation", "Lcl Rotation", "R"),
                ("scale", "Lcl Scaling", "S"),
            ):
                if path not in paths:
                    continue
                key_times, values, constant = self._keys(path, *paths[path], fps)
                ticks = [round(t * FBX_TICKS_PER_SECOND) for t in key_times]
                curve_node_id = self._id()
                curve_node = self._object(
                    FbxNode(
                        "AnimationCurveNode",
                        FbxWriter.i64(curve_node_id),
                        f"{short}\x00\x01AnimCurveNode",
                        "",
                    ),
                    "AnimationCurveNode",
                )
                cn_props = curve_node.add("Properties70")
                for axis, label in enumerate(("d|X", "d|Y", "d|Z")):
                    cn_props.p(label, "Number", "", "A", float(values[0][axis]))
                self._link(curve_node_id, layer_id)
                self._link(curve_node_id, model_id, prop)
                for axis, label in enumerate(("d|X", "d|Y", "d|Z")):
                    curve_id = self._id()
                    curve = self._object(
                        FbxNode(
                            "AnimationCurve",
                            FbxWriter.i64(curve_id),
                            "\x00\x01AnimCurve",
                            "",
                        ),
                        "AnimationCurve",
                    )
                    curve.add("Default", float(values[0][axis]))
                    curve.add("KeyVer", 4009)
                    curve.add("KeyTime", FbxWriter.array("l", ticks))
                    curve.add(
                        "KeyValueFloat", FbxWriter.array("f", [v[axis] for v in values])
                    )
                    curve.add(
                        "KeyAttrFlags",
                        FbxWriter.array(
                            "i", [_KEY_CONSTANT if constant else _KEY_LINEAR]
                        ),
                    )
                    curve.add("KeyAttrDataFloat", FbxWriter.array("f", _KEY_DATA))
                    curve.add("KeyAttrRefCount", FbxWriter.array("i", [len(ticks)]))
                    self._link(curve_id, curve_node_id, label)
                keys += len(ticks)
        self.take = (take, start, stop)
        self.fps = fps
        frames = (int(round(start * fps)), int(round(stop * fps)))
        self.scene_section = {
            "fps": fps,
            "frame_start": frames[0],
            "frame_end": frames[1],
            "anim_start": frames[0],
            "anim_end": frames[1],
            "frame_current": frames[0],
        }
        self.report.update(animation=take, fps=fps, keys=keys)

    # ------------------------------------------------------------ document
    def records(self, creator: str) -> List[FbxNode]:
        """The whole FBX document, header to takes."""
        fps = self.fps or _DEFAULT_FPS
        start, stop = (self.take[1], self.take[2]) if self.take else (0.0, 1.0)
        settings = FbxNode("GlobalSettings")
        settings.add("Version", 1000)
        p = settings.add("Properties70")
        for key, value in (
            ("UpAxis", 1),
            ("UpAxisSign", 1),
            ("FrontAxis", 2),
            ("FrontAxisSign", 1),
            ("CoordAxis", 0),
            ("CoordAxisSign", 1),
            ("OriginalUpAxis", 1),
            ("OriginalUpAxisSign", 1),
        ):
            p.p(key, "int", "Integer", "", value)
        p.p("UnitScaleFactor", "double", "Number", "", 1.0)
        p.p("OriginalUnitScaleFactor", "double", "Number", "", 1.0)
        p.p("AmbientColor", "ColorRGB", "Color", "", 0.0, 0.0, 0.0)
        p.p("DefaultCamera", "KString", "", "", "Producer Perspective")
        mode = next(
            (m for rate, m in _TIME_MODES.items() if abs(rate - fps) < 1e-3),
            _TIME_MODE_CUSTOM,
        )
        p.p("TimeMode", "enum", "", "", mode)
        p.p(
            "TimeSpanStart",
            "KTime",
            "Time",
            "",
            FbxWriter.i64(round(start * FBX_TICKS_PER_SECOND)),
        )
        p.p(
            "TimeSpanStop",
            "KTime",
            "Time",
            "",
            FbxWriter.i64(round(stop * FBX_TICKS_PER_SECOND)),
        )
        p.p("CustomFrameRate", "double", "Number", "", float(fps))

        documents = FbxNode("Documents")
        documents.add("Count", 1)
        document = documents.add(
            "Document", FbxWriter.i64(self._id()), "Scene", "Scene"
        )
        dp = document.add("Properties70")
        dp.p("SourceObject", "object", "", "")
        dp.p(
            "ActiveAnimStackName", "KString", "", "", self.take[0] if self.take else ""
        )
        document.add("RootNode", FbxWriter.i64(0))

        definitions = FbxNode("Definitions")
        definitions.add("Version", 100)
        definitions.add("Count", 1 + sum(self.counts.values()))
        definitions.add("ObjectType", "GlobalSettings").add("Count", 1)
        for kind, count in self.counts.items():
            definitions.add("ObjectType", kind).add("Count", count)

        objects = FbxNode("Objects")
        objects.children = self.objects
        connections = FbxNode("Connections")
        for child, parent, prop in self.links:
            if prop is None:
                connections.add("C", "OO", FbxWriter.i64(child), FbxWriter.i64(parent))
            else:
                connections.add(
                    "C", "OP", FbxWriter.i64(child), FbxWriter.i64(parent), prop
                )

        takes = FbxNode("Takes")
        takes.add("Current", self.take[0] if self.take else "")
        if self.take:
            take = takes.add("Take", self.take[0])
            take.add("FileName", f"{self.take[0]}.tak")
            span = (
                FbxWriter.i64(round(start * FBX_TICKS_PER_SECOND)),
                FbxWriter.i64(round(stop * FBX_TICKS_PER_SECOND)),
            )
            take.add("LocalTime", *span)
            take.add("ReferenceTime", *span)
        return FbxWriter.header_records(creator) + [
            settings,
            documents,
            FbxNode("References"),
            definitions,
            objects,
            connections,
            takes,
        ]

    def manifest(self):
        """The :class:`~pythontk.HandoffManifest` sidecar the consumer replays."""
        from pythontk.core_utils.handoff.manifest import HandoffManifest

        groups: Dict[Tuple[int, Optional[int]], List[str]] = {}
        for i, key in self.node_geometry.items():
            groups.setdefault(key, []).append(self.names[i])
        instances = sorted(sorted(g) for g in groups.values() if len(g) > 1)
        sections: Dict[str, Any] = {
            HandoffManifest.VERSION_KEY: HandoffManifest.VERSION,
            HandoffManifest.FORMAT_KEY: HandoffManifest.FORMAT_NAMES,
            HandoffManifest.MATERIALS: self.manifest_materials,
            HandoffManifest.SCENE_MATERIALS: [
                self.material_names[i] for i in self.material_ids
            ],
            HandoffManifest.EMPTIES: self.empties,
        }
        if self.scene_section:
            # Only an animated file has a clock; a static one leaves the host's.
            sections[HandoffManifest.SCENE] = self.scene_section
        if instances:
            sections[HandoffManifest.INSTANCES] = instances
        return HandoffManifest.build(**sections)

    @classmethod
    def gltf_to_fbx(
        cls,
        src: str,
        dst: Optional[str] = None,
        *,
        texture_dir: Optional[str] = None,
        include_animation: bool = True,
        fps: Optional[float] = None,
        manifest: bool = True,
        overwrite: bool = False,
        report: Optional[Dict[str, Any]] = None,
    ) -> str:
        src = os.path.abspath(os.path.expanduser(os.path.expandvars(str(src))))
        if not os.path.isfile(src):
            raise FileNotFoundError(f"glTF not found: {src}")
        if os.path.splitext(src)[1].lower() not in GLTF_EXTENSIONS:
            raise ValueError(
                f"Not a glTF container: {src} (expected {GLTF_EXTENSIONS})"
            )
        dst = os.path.abspath(str(dst)) if dst else os.path.splitext(src)[0] + ".fbx"
        if not dst.lower().endswith(".fbx"):
            dst += ".fbx"
        if os.path.exists(dst) and not overwrite:
            raise FileExistsError(f"Destination exists (pass overwrite=True): {dst}")
        texture_dir = os.path.abspath(
            texture_dir or os.path.splitext(dst)[0] + "_textures"
        )

        build = cls(
            _GltfDocument(src),
            texture_dir=texture_dir,
            include_animation=include_animation,
            fps=fps,
        )
        build.build()
        from pythontk import __version__

        FbxWriter.write(dst, build.records(f"pythontk {__version__} gltf_to_fbx"))
        if manifest:
            build.manifest().write(dst)
        logger.info(
            "gltf_to_fbx: %s -> %s (%d node(s), %d mesh(es), %d material(s), "
            "%d texture(s)%s).",
            os.path.basename(src),
            os.path.basename(dst),
            build.report.get("nodes", 0),
            build.report.get("meshes", 0),
            build.report.get("materials", 0),
            build.report.get("textures", 0),
            f", {build.report['keys']} key(s) at {build.report['fps']:g} fps"
            if build.report.get("keys")
            else "",
        )
        if report is not None:
            report.update(build.report)
        return dst
