# !/usr/bin/python
# coding=utf-8
"""Unit tests for :class:`pythontk.FbxMedia` -- the binary-FBX payload writer.

The fixture is a synthetic 7700 FBX built the way the SDK lays one out
(64-bit records, nested NULL sentinels, the 16-byte footer id, 16-byte
alignment padding, version echo, magic). The round-trip identity that pins
the writer here was also checked by hand against two Maya-written files
(a 21 MB rigged prop and a 366 MB production assembly): both re-serialised
byte for byte.
"""

import io
import json
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk.file_utils.mesh_convert.fbx_file import FbxFile  # noqa: E402
from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia  # noqa: E402
from pythontk.file_utils.temp_artifacts import TempArtifacts  # noqa: E402

MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
FOOTER_ID = bytes(range(16))
FOOT_MAGIC = b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"


class Raw(bytes):
    """A property written with the ``R`` (raw binary) tag rather than ``S``."""


class I32(int):
    """A property written with the ``I`` (int32) tag rather than ``L``."""


class F32(float):
    """A property written with the ``F`` (float32) tag rather than ``D``."""


class Doubles(list):
    """An array property written with the ``d`` (float64 array) tag."""


class Ints(list):
    """An array property written with the ``i`` (int32 array) tag."""


def _prop(value) -> bytes:
    if isinstance(value, Raw):
        return b"R" + struct.pack("<I", len(value)) + value
    if isinstance(value, bytes):
        return b"S" + struct.pack("<I", len(value)) + value
    if isinstance(value, bool):  # before int, which it is
        return b"C" + struct.pack("<?", value)
    if isinstance(value, I32):
        return b"I" + struct.pack("<i", value)
    if isinstance(value, int):
        return b"L" + struct.pack("<q", value)
    if isinstance(value, F32):
        return b"F" + struct.pack("<f", value)
    if isinstance(value, float):
        return b"D" + struct.pack("<d", value)
    if isinstance(value, (Doubles, Ints)):
        tag = "d" if isinstance(value, Doubles) else "i"
        data = struct.pack(f"<{len(value)}{tag}", *value)
        return tag.encode() + struct.pack("<III", len(value), 0, len(data)) + data
    raise TypeError(type(value))


def _record(name: str, props, children, offset: int) -> bytes:
    prop_bytes = b"".join(_prop(p) for p in props)
    name_bytes = name.encode("ascii")
    header_len = 24 + 1 + len(name_bytes)
    body = b""
    child_offset = offset + header_len + len(prop_bytes)
    for child in children:
        chunk = _record(child[0], child[1], child[2], child_offset)
        body += chunk
        child_offset += len(chunk)
    if children:
        body += b"\x00" * 25
    end = offset + header_len + len(prop_bytes) + len(body)
    return (
        struct.pack("<QQQB", end, len(props), len(prop_bytes), len(name_bytes))
        + name_bytes
        + prop_bytes
        + body
    )


def png_bytes(size, mode="RGB") -> bytes:
    from PIL import Image

    image = Image.new(mode, size)
    # A gradient, so a resize changes the pixels rather than the header only.
    image.putdata(
        [
            (x % 256, y % 256, (x * y) % 256)[: len(mode)] if len(mode) > 1 else x % 256
            for y in range(size[1])
            for x in range(size[0])
        ]
    )
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def jpeg_bytes(size) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", size, (200, 40, 40)).save(out, format="JPEG", quality=95)
    return out.getvalue()


def _media_records(media):
    """``(objects, connections)`` embedding *media* ``{name: bytes}``."""
    objects, connections = [], []
    for index, (name, payload) in enumerate(media.items()):
        video_id, texture_id = 5001 + index, 6001 + index
        objects.append(
            (
                "Video",
                [video_id, name.encode() + b"\x00\x01Video", b"Clip"],
                [
                    ("Type", [b"Clip"], []),
                    ("Filename", [("C:/tex/" + name).encode()], []),
                    ("RelativeFilename", [name.encode()], []),
                    ("Content", [Raw(payload)] if payload else [], []),
                ],
            )
        )
        objects.append(
            ("Texture", [texture_id, name.encode() + b"\x00\x01Texture", b""], [])
        )
        connections.append(("C", [b"OO", video_id, texture_id], []))
    return objects, connections


def _takes_section(takes):
    """The ``Takes`` section the SDK writes for ``{name: (start, end)}``."""
    return (
        "Takes",
        [],
        [("Current", [next(iter(takes)).encode()], [])]
        + [
            (
                "Take",
                [name.encode()],
                [
                    ("FileName", [name.replace(" ", "_").encode() + b".tak"], []),
                    ("LocalTime", [start, end], []),
                    ("ReferenceTime", [start, end], []),
                ],
            )
            for name, (start, end) in takes.items()
        ],
    )


def _write_fbx(path: str, roots, footer_id: bytes = FOOTER_ID) -> str:
    body = b""
    offset = len(MAGIC) + 4
    for name, props, children in roots:
        chunk = _record(name, props, children, offset)
        body += chunk
        offset += len(chunk)
    body += b"\x00" * 25
    body += footer_id
    at = offset + 25 + 16
    pad = ((at + 15) & ~15) - at or 16
    body += (
        b"\x00" * pad
        + b"\x00" * 4
        + struct.pack("<I", 7700)
        + b"\x00" * 120
        + FOOT_MAGIC
    )
    with open(path, "wb") as handle:
        handle.write(MAGIC + struct.pack("<I", 7700) + body)
    return path


#: The ``FileId`` / ``CreationTime`` / footer id Blender's exporter writes. The
#: FBX SDK reads NOTHING from a file whose header lacks the first two
#: (measured: FBX2glTF converts it to an empty scene and exits 0), so a fixture
#: the converter has to read carries this set.
SDK_IDS = (
    b"\x28\xb3\x2a\xeb\xb6\x24\xcc\xc2\xbf\xc8\xb0\x2a\xa9\x2b\xfc\xf1",
    b"1970-01-01 10:00:00:000",
    b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e",
)


def build_stingray_fbx(path: str, maps, records=None) -> str:
    """A quad with a Stingray PBS material the FBX SDK -- so FBX2glTF -- binds.

    *maps* ``{"color" | "roughness" | "metallic": (file name, image bytes)}``
    embeds each image and wires it to ``Maya|TEX_<slot>_map``, as Maya writes
    a ``GameShader`` network; a key naming a whole slot (``"TEX_brdf_lut"``)
    is wired to ``Maya|<slot>``. *records* ``{record key: payload}`` ride a
    ``data_export`` carrier. Beyond what :func:`build_fbx` writes, the SDK
    needs the :data:`SDK_IDS` header, a ``Definitions`` count per object type
    and the ``Texture`` / ``Video`` records' own properties: without those it
    finds the images and binds none ("could not find a image file for texture").
    """

    def p(*values):
        return ("P", list(values), [])

    def layer_element(kind, version, name, mapping, reference, data):
        head = [
            ("Version", [I32(version)], []),
            ("Name", [name], []),
            ("MappingInformationType", [mapping], []),
            ("ReferenceInformationType", [reference], []),
        ]
        return (kind, [I32(0)], head + data)

    geometry, model, material = 2001, 2002, 2003
    corners = [-0.5, 0, 0.5, 0.5, 0, 0.5, -0.5, 0, -0.5, 0.5, 0, -0.5]
    layer = [("Version", [I32(100)], [])] + [
        ("LayerElement", [], [("Type", [kind], []), ("TypedIndex", [I32(0)], [])])
        for kind in (b"LayerElementNormal", b"LayerElementMaterial", b"LayerElementUV")
    ]
    objects = [
        (
            "Geometry",
            [geometry, b"\x00\x01Geometry", b"Mesh"],
            [
                ("Vertices", [Doubles(corners)], []),
                ("PolygonVertexIndex", [Ints([0, 1, 3, -3])], []),
                ("GeometryVersion", [I32(124)], []),
                layer_element(
                    "LayerElementNormal",
                    102,
                    b"",
                    b"ByPolygonVertex",
                    b"Direct",
                    [("Normals", [Doubles([0, 1, 0] * 4)], [])],
                ),
                layer_element(
                    "LayerElementUV",
                    101,
                    b"map1",
                    b"ByPolygonVertex",
                    b"IndexToDirect",
                    [
                        ("UV", [Doubles([0, 0, 1, 0, 0, 1, 1, 1])], []),
                        ("UVIndex", [Ints([0, 1, 3, 2])], []),
                    ],
                ),
                layer_element(
                    "LayerElementMaterial",
                    101,
                    b"",
                    b"AllSame",
                    b"IndexToDirect",
                    [("Materials", [Ints([0])], [])],
                ),
                ("Layer", [I32(0)], layer),
            ],
        ),
        (
            "Model",
            [model, b"plane\x00\x01Model", b"Mesh"],
            [
                ("Version", [I32(232)], []),
                (
                    "Properties70",
                    [],
                    [p(b"DefaultAttributeIndex", b"int", b"Integer", b"", I32(0))],
                ),
                ("Shading", [True], []),
                ("Culling", [b"CullingOff"], []),
            ],
        ),
    ]
    shader = [
        p(b"Maya", b"Compound", b"", b""),
        p(b"Maya|TypeId", b"int", b"Integer", b"", I32(1166017)),  # Stingray PBS
    ]
    for slot in ("color", "metallic", "roughness"):
        use = F32(1.0 if slot in maps else 0.0)
        shader.append(p(f"Maya|use_{slot}_map".encode(), b"float", b"", b"", use))
        shader.append(
            p(
                f"Maya|TEX_{slot}_map".encode(),
                b"Vector3D",
                b"Vector",
                b"",
                0.0,
                0.0,
                0.0,
            )
        )
    shader.append(p(b"Maya|metallic", b"float", b"", b"", F32(0.0)))
    shader.append(p(b"Maya|roughness", b"float", b"", b"", F32(0.33)))
    objects.append(
        (
            "Material",
            [material, b"orm\x00\x01Material", b""],
            [
                ("Version", [I32(102)], []),
                ("ShadingModel", [b"unknown"], []),
                ("MultiLayer", [I32(0)], []),
                ("Properties70", [], shader),
            ],
        )
    )
    connections = [
        ("C", [b"OO", model, 0], []),
        ("C", [b"OO", geometry, model], []),
        ("C", [b"OO", material, model], []),
    ]
    for index, (slot, (name, payload)) in enumerate(maps.items()):
        video, texture = 3001 + index, 4001 + index
        stem = name.rsplit(".", 1)[0].encode()
        filename = b"C:/tex/" + name.encode()
        paths = [
            p(b"Path", b"KString", b"XRefUrl", b"", filename),
            p(b"RelPath", b"KString", b"XRefUrl", b"", name.encode()),
        ]
        objects.append(
            (
                "Video",
                [video, stem + b"\x00\x01Video", b"Clip"],
                [
                    ("Type", [b"Clip"], []),
                    ("Properties70", [], paths),
                    ("UseMipMap", [I32(0)], []),
                    ("Filename", [filename], []),
                    ("RelativeFilename", [name.encode()], []),
                    ("Content", [Raw(payload)], []),
                ],
            )
        )
        objects.append(
            (
                "Texture",
                [texture, stem + b"\x00\x01Texture", b""],
                [
                    ("Type", [b"TextureVideoClip"], []),
                    ("Version", [I32(202)], []),
                    ("TextureName", [stem + b"\x00\x01Texture"], []),
                    (
                        "Properties70",
                        [],
                        [p(b"UseMaterial", b"bool", b"", b"", I32(1))],
                    ),
                    ("Media", [stem + b"\x00\x01Video"], []),
                    ("FileName", [filename], []),
                    ("RelativeFilename", [name.encode()], []),
                    ("ModelUVTranslation", [0.0, 0.0], []),
                    ("ModelUVScaling", [1.0, 1.0], []),
                    ("Texture_Alpha_Source", [b"None"], []),
                    ("Cropping", [I32(0)] * 4, []),
                ],
            )
        )
        prop = f"Maya|{slot}" if slot.startswith("TEX_") else f"Maya|TEX_{slot}_map"
        connections += [
            ("C", [b"OP", texture, material, prop.encode()], []),
            ("C", [b"OO", video, texture], []),
        ]
    if records:
        held = [
            p(key.encode(), b"KString", b"", b"U", json.dumps(value).encode())
            for key, value in records.items()
        ]
        label = b"data_export\x00\x01Model"
        objects.append(("Model", [2004, label, b"Null"], [("Properties70", [], held)]))
        connections.append(("C", [b"OO", 2004, 0], []))
    counts = {}
    for kind, _props, _children in objects:
        counts[kind] = counts.get(kind, 0) + 1
    definitions = [
        ("Version", [I32(100)], []),
        ("Count", [I32(sum(counts.values()))], []),
    ] + [
        ("ObjectType", [kind.encode()], [("Count", [I32(n)], [])])
        for kind, n in counts.items()
    ]
    file_id, created, footer_id = SDK_IDS
    header = [("FBXHeaderVersion", [I32(1003)], []), ("FBXVersion", [I32(7700)], [])]
    roots = [
        ("FBXHeaderExtension", [], header),
        ("FileId", [Raw(file_id)], []),
        ("CreationTime", [created], []),
        ("Definitions", [], definitions),
        ("Objects", [], objects),
        ("Connections", [], connections),
    ]
    return _write_fbx(path, roots, footer_id=footer_id)


def build_instanced_fbx(path: str, instances, deformed: bool = False) -> str:
    """One ``Geometry`` under one Model per entry of *instances* -- each
    ``{property: (x, y, z)}``, its ``Properties70`` vectors -- as Maya writes
    an instanced mesh; the Models are ``WALL_A``, ``WALL_B`` ... *deformed*
    wires a ``Skin`` under the geometry."""
    geometry = 2001
    objects = [
        (
            "Geometry",
            [geometry, b"wallShape\x00\x01Geometry", b"Mesh"],
            [
                ("Vertices", [Doubles([0, 0, 0, 1, 0, 0, 0, 1, 0])], []),
                ("PolygonVertexIndex", [Ints([0, 1, -3])], []),
            ],
        )
    ]
    connections = []
    for index, props in enumerate(instances):
        model = 3001 + index
        vectors = [
            ("P", [name.encode(), b"Vector3D", b"Vector", b"", *map(float, xyz)], [])
            for name, xyz in props.items()
        ]
        label = f"WALL_{chr(65 + index)}\x00\x01Model".encode()
        objects.append(("Model", [model, label, b"Mesh"], [("Properties70", [], vectors)]))
        connections += [("C", [b"OO", model, 0], []), ("C", [b"OO", geometry, model], [])]
    if deformed:
        objects.append(("Deformer", [4001, b"skin\x00\x01Deformer", b"Skin"], []))
        connections.append(("C", [b"OO", 4001, geometry], []))
    counts = {}
    for kind, _props, _children in objects:
        counts[kind] = counts.get(kind, 0) + 1
    definitions = [
        ("Version", [I32(100)], []),
        ("Count", [I32(sum(counts.values()))], []),
    ] + [
        ("ObjectType", [kind.encode()], [("Count", [I32(n)], [])])
        for kind, n in counts.items()
    ]
    roots = [
        ("Definitions", [], definitions),
        ("Objects", [], objects),
        ("Connections", [], connections),
    ]
    return _write_fbx(path, roots)


def build_fbx(path: str, media, models: int = 1, takes=None) -> str:
    """Write a binary FBX whose ``Video`` objects embed *media* ``{name: bytes}``.

    *models* ``Model`` records are written, and *takes* -- ``{name: (start,
    end)}`` in FBX ticks -- becomes the ``Takes`` section the SDK writes.
    """
    objects = [
        ("Model", [1001 + i, f"cube{i}\x00\x01Model".encode(), b"Mesh"], [])
        for i in range(models)
    ]
    media_objects, connections = _media_records(media)
    roots = [("Objects", [], objects + media_objects), ("Connections", [], connections)]
    if takes:
        roots.append(_takes_section(takes))
    return _write_fbx(path, roots)


def build_animated_fbx(path: str, takes, declared=None, media=None, channels=None) -> str:
    """An FBX shaped like a Maya take split: one stack per take in *takes*.

    Each take owns a layer, a curve node (wired to the model's translation)
    and a curve. One extra curve node + curve is SHARED between the first two
    takes' layers, so a drop must keep it while either owner survives. The
    ``Definitions`` counts match the objects. *declared* take names are
    published the way the exporters do -- a ``data_export`` model's
    ``fbx_takes`` user property -- and *channels* ``{record key: payload}``
    ride the same carrier.
    """
    objects = [("Model", [1001, b"cube0\x00\x01Model", b"Mesh"], [])]
    connections = []
    for i, name in enumerate(takes):
        stack, layer, node, curve = 100 + i, 200 + i, 300 + i, 400 + i
        objects += [
            ("AnimationStack", [stack, name.encode() + b"\x00\x01AnimStack", b""], []),
            ("AnimationLayer", [layer, b"BaseLayer\x00\x01AnimLayer", b""], []),
            ("AnimationCurveNode", [node, b"T\x00\x01AnimCurveNode", b""], []),
            ("AnimationCurve", [curve, b"\x00\x01AnimCurve", b""], []),
        ]
        connections += [
            ("C", [b"OO", layer, stack], []),
            ("C", [b"OO", node, layer], []),
            ("C", [b"OP", node, 1001, b"Lcl Translation"], []),
            ("C", [b"OP", curve, node, b"d|X"], []),
        ]
    objects += [
        ("AnimationCurveNode", [399, b"S\x00\x01AnimCurveNode", b""], []),
        ("AnimationCurve", [499, b"\x00\x01AnimCurve", b""], []),
    ]
    connections += [
        ("C", [b"OO", 399, 200], []),
        ("C", [b"OO", 399, 201], []),
        ("C", [b"OP", 499, 399, b"d|X"], []),
    ]
    held = dict(channels or {})
    if declared is not None:
        held["fbx_takes"] = [{"name": n, "start": 0, "end": 10} for n in declared]
    if held:
        props = [
            ("P", [k.encode(), b"KString", b"", b"U", json.dumps(v).encode()], [])
            for k, v in held.items()
        ]
        objects.append(
            (
                "Model",
                [1002, b"data_export\x00\x01Model", b"Null"],
                [("Properties70", [], props)],
            )
        )
    media_objects, media_connections = _media_records(media or {})
    objects += media_objects
    connections += media_connections
    counts = {
        b"Model": 1 + bool(held),
        b"AnimationStack": len(takes),
        b"AnimationLayer": len(takes),
        b"AnimationCurveNode": len(takes) + 1,
        b"AnimationCurve": len(takes) + 1,
    }
    definitions = (
        "Definitions",
        [],
        [("Version", [I32(100)], []), ("Count", [I32(sum(counts.values()))], [])]
        + [
            ("ObjectType", [kind], [("Count", [I32(n)], [])])
            for kind, n in counts.items()
        ],
    )
    roots = [
        definitions,
        ("Objects", [], objects),
        ("Connections", [], connections),
        _takes_section({name: (0, 46186158000) for name in takes}),
    ]
    return _write_fbx(path, roots)


#: FBX ticks per frame at the curve fixtures' 30 fps.
FRAME = 46186158000 // 30
#: ``KeyAttrFlags`` the curve fixtures spell: interpolation, tangent mode,
#: weights (the FBX SDK's ``FbxAnimCurveDef`` bits).
CONSTANT, LINEAR, CUBIC = 0x2, 0x4, 0x8
CONSTANT_NEXT, AUTO, USER, BREAK = 0x100, 0x100, 0x400, 0x800
TIME_INDEPENDENT, CLAMP, CLAMP_PROGRESSIVE = 0x2000, 0x1000, 0x4000
WEIGHTED_RIGHT, WEIGHTED_NEXT_LEFT = 0x01000000, 0x02000000


def packed_weights(right: float = 1 / 3, next_left: float = 1 / 3) -> float:
    """A key's two tangent weights as ``KeyAttrDataFloat`` stores them: two
    16-bit fractions of 9999 in one float's bits (right weight low)."""
    word = struct.pack("<HH", int(right * 9999), int(next_left * 9999))
    return struct.unpack("<f", word)[0]


def build_curve_fbx(
    path: str,
    takes,
    current=None,
    layers=None,
    base_layer=lambda take: "BaseLayer",
    user=(),
) -> str:
    """An FBX whose takes animate real curves, written by :class:`FbxWriter`.

    *takes* is ``{take: {channel: keys}}`` in file order. A channel is
    ``(model, property, component)``, or ``(model, property, component,
    layer)`` to key it on that layer index of the take (0, ``BaseLayer``, by
    default). Its keys are ``[(frame, value), ...]`` -- linear, as Maya writes
    a linear tangent -- or ``{"keys": [...], "flags": [...], "data": [...]}``
    spelling each key's ``KeyAttrFlags`` and four ``KeyAttrDataFloat``
    values. Every take has a stack, its layers, a curve node per (model,
    property, layer) declaring ``d|X`` / ``d|Y`` / ``d|Z``, and a curve per
    channel; each declares its key extent as its span. ``Takes/Current``
    names *current* (default: the first take); *layers* ``{take: count}``
    gives a take extra, empty layers; *base_layer* names a take's first layer
    (Maya's ``BaseLayer``; Blender names it after the take); *user* names
    the user properties (``A+U``, a DCC's custom attributes) every model
    declares. 30 fps (``TimeMode`` 6).
    """
    from pythontk.file_utils.mesh_convert._fbx_writer import FbxNode, FbxWriter

    w = FbxWriter
    uid = iter(range(10_000, 1_000_000))
    models = {}
    objects = FbxNode("Objects")
    connections = FbxNode("Connections")
    counts = {}

    def obj(kind, oid, name, cls=""):
        counts[kind] = counts.get(kind, 0) + 1
        return objects.add(kind, w.i64(oid), f"{name}\x00\x01{cls}", "")

    def link(child, parent, prop=None):
        row = ["OO" if prop is None else "OP", w.i64(child), w.i64(parent)]
        connections.add("C", *row, *([] if prop is None else [prop]))

    for channels in takes.values():
        for model, *_rest in channels:
            if model not in models:
                models[model] = next(uid)
                record = obj("Model", models[model], model, "Model")
                record.props[2] = "Null"
                link(models[model], 0)
                if user:
                    declared = record.add("Properties70")
                    for prop in user:
                        declared.p(prop, "Number", "", "A+U", 0.0)

    spans = {}
    for take, channels in takes.items():
        stack = next(uid)
        stack_record = obj("AnimationStack", stack, take, "AnimStack")
        wanted = max(
            [(layers or {}).get(take, 1)]
            + [key[3] + 1 for key in channels if len(key) > 3]
        )
        layer_ids = []
        for index in range(wanted):
            layer_ids.append(next(uid))
            name = base_layer(take) if index == 0 else f"{take}_Layer{index}"
            obj("AnimationLayer", layer_ids[-1], name, "AnimLayer")
            link(layer_ids[-1], stack)
        nodes = {}
        first = last = None
        for key, spec in channels.items():
            model, prop, comp = key[:3]
            layer = key[3] if len(key) > 3 else 0
            if isinstance(spec, dict):
                keys, flags, data = spec["keys"], spec["flags"], spec["data"]
            else:
                keys = spec
                flags = [LINEAR | USER] * len(keys)
                data = [(0.0, 0.0, packed_weights(), 0.0)] * len(keys)
            node_key = (model, prop, layer)
            if node_key not in nodes:
                nodes[node_key] = next(uid)
                short = {"Lcl Translation": "T", "Lcl Rotation": "R"}.get(prop, "S")
                node = obj(
                    "AnimationCurveNode", nodes[node_key], short, "AnimCurveNode"
                )
                props = node.add("Properties70")
                for axis in ("d|X", "d|Y", "d|Z"):
                    props.p(axis, "Number", "", "A", 0.0)
                link(nodes[node_key], layer_ids[layer])
                link(nodes[node_key], models[model], prop)
            curve = next(uid)
            record = obj("AnimationCurve", curve, "", "AnimCurve")
            record.add("Default", 0.0)
            record.add("KeyVer", 4009)
            record.add("KeyTime", w.array("l", [round(f * FRAME) for f, _ in keys]))
            record.add("KeyValueFloat", w.array("f", [v for _, v in keys]))
            record.add("KeyAttrFlags", w.array("i", flags))
            record.add("KeyAttrDataFloat", w.array("f", [x for d in data for x in d]))
            record.add("KeyAttrRefCount", w.array("i", [1] * len(keys)))
            link(curve, nodes[node_key], comp)
            if keys:
                lo, hi = round(keys[0][0] * FRAME), round(keys[-1][0] * FRAME)
                first = lo if first is None else min(first, lo)
                last = hi if last is None else max(last, hi)
        spans[take] = (first or 0, last or 0)
        props = stack_record.add("Properties70")
        for name, value in (
            ("LocalStart", spans[take][0]),
            ("LocalStop", spans[take][1]),
            ("ReferenceStart", spans[take][0]),
            ("ReferenceStop", spans[take][1]),
        ):
            props.p(name, "KTime", "Time", "", w.i64(value))

    settings = FbxNode("GlobalSettings")
    settings.add("Version", 1000)
    grid = settings.add("Properties70")
    grid.p("TimeMode", "enum", "", "", 6)
    grid.p("CustomFrameRate", "double", "Number", "", -1.0)
    definitions = FbxNode("Definitions")
    definitions.add("Version", 100)
    definitions.add("Count", sum(counts.values()))
    for kind, count in counts.items():
        definitions.add("ObjectType", kind).add("Count", count)
    sections = FbxNode("Takes")
    if takes:
        sections.add("Current", current or next(iter(takes)))
    for take, (start, stop) in spans.items():
        entry = sections.add("Take", take)
        entry.add("FileName", take.replace(" ", "_") + ".tak")
        entry.add("LocalTime", w.i64(start), w.i64(stop))
        entry.add("ReferenceTime", w.i64(start), w.i64(stop))
    records = FbxWriter.header_records("pythontk test") + [
        settings,
        definitions,
        objects,
        connections,
        sections,
    ]
    return FbxWriter.write(path, records)


def frames(first: int, last: int, motion=lambda f: f * 0.5):
    """Per-frame keys over ``first..last`` -- a resampled take -- of *motion*."""
    return [(f, motion(f)) for f in range(first, last + 1)]


class FbxMediaTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = TempArtifacts("test_fbx_media", policy="scoped")
        self.dir = self.temp.dir_path()
        self.big = png_bytes((512, 256))
        self.small = png_bytes((64, 64))
        self.src = build_fbx(
            os.path.join(self.dir, "asset.fbx"),
            {"wall_Base_color.png": self.big, "trim_Normal.png": self.small},
        )

    def tearDown(self):
        self.temp.cleanup()

    def _read(self, path):
        with open(path, "rb") as fh:
            return fh.read()

    def test_rewrite_is_byte_identical(self):
        out = os.path.join(self.dir, "copy.fbx")
        FbxMedia.rewrite(self.src, out)
        self.assertEqual(self._read(self.src), self._read(out))

    def test_embedded_lists_every_video_payload(self):
        rows = FbxMedia.embedded(self.src)
        self.assertEqual(
            [(r["name"], r["format"], r["size"], r["bytes"]) for r in rows],
            [
                ("wall_Base_color.png", "PNG", (512, 256), len(self.big)),
                ("trim_Normal.png", "PNG", (64, 64), len(self.small)),
            ],
        )

    def test_a_failed_write_strands_no_partial_file(self):
        """``downsize`` streams into a sibling ``.part`` and only consumes it
        on the success path's ``os.replace``.

        A raise inside ``_write`` -- disk full is the realistic case for a
        multi-hundred-MB payload -- left a partial file beside the target with
        nothing to sweep it, and allocated it raw besides. It is now a tracked
        ``TempArtifacts`` allocation, so a process that dies mid-write is
        swept by the next store on the same prefix rather than never.
        """
        from unittest import mock

        out = os.path.join(self.dir, "failed.fbx")
        with mock.patch.object(
            FbxMedia, "_write", side_effect=OSError("No space left on device")
        ):
            with self.assertRaises(OSError):
                FbxMedia.downsize(self.src, out, max_size=128)
        strays = [n for n in os.listdir(self.dir) if n.endswith(".part")]
        self.assertEqual(strays, [], f"partial file left behind: {strays}")
        self.assertFalse(os.path.exists(out), "a failed run must not leave a target")

    def test_downsize_resizes_only_images_over_the_ceiling(self):
        out = os.path.join(self.dir, "small.fbx")
        report = FbxMedia.downsize(self.src, out, max_size=128)
        self.assertEqual(report["images"], 2)
        self.assertEqual(report["resized"], 1)
        self.assertLess(report["after"], report["before"])
        rows = {r["name"]: r for r in FbxMedia.embedded(out)}
        self.assertEqual(rows["wall_Base_color.png"]["size"], (128, 64))
        self.assertEqual(rows["wall_Base_color.png"]["format"], "PNG")
        self.assertEqual(rows["trim_Normal.png"]["bytes"], len(self.small))
        # Everything but the payload survived: same objects, same wiring.
        before, after = FbxFile.load(self.src), FbxFile.load(out)
        self.assertEqual(before.objects_census(), after.objects_census())
        self.assertEqual(before.connections(), after.connections())
        self.assertLess(os.path.getsize(out), os.path.getsize(self.src))
        # The untouched source is still the file it was.
        self.assertEqual(FbxMedia.embedded(self.src)[0]["size"], (512, 256))

    def test_downsize_resamples_a_normal_map_as_one(self):
        """A map the registry names a normal map is resized in float and
        renormalised (``ImgUtils.resize_image(normal_map=True)``): this pass
        is where the pipeline's normals actually shrink, so the GLB texture
        pass after it never resamples them again."""
        from PIL import Image
        from conftest import hard_edge_normal_map, unit_length_error

        def png(image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()

        normal = hard_edge_normal_map((64, 64))
        src = build_fbx(
            os.path.join(self.dir, "normals.fbx"),
            {
                "crease_Normal_OpenGL.png": png(normal),
                "crease_Base_Color.png": png(normal),
            },
        )
        out = os.path.join(self.dir, "normals_small.fbx")
        self.assertEqual(FbxMedia.downsize(src, out, max_size=27)["resized"], 2)
        payloads = ExpandGrayscaleTestCase._payloads(out)
        with Image.open(io.BytesIO(payloads["crease_Normal_OpenGL.png"])) as image:
            self.assertEqual(image.size, (27, 27))
            self.assertLess(unit_length_error(image), 0.01)
        # The same pixels named as colour take the plain resample.
        with Image.open(io.BytesIO(payloads["crease_Base_Color.png"])) as image:
            self.assertGreater(unit_length_error(image), 0.1)

    def test_zero_ceiling_and_exempt_names_write_nothing(self):
        out = os.path.join(self.dir, "untouched.fbx")
        self.assertEqual(FbxMedia.downsize(self.src, out, max_size=0)["resized"], 0)
        self.assertFalse(os.path.exists(out))
        report = FbxMedia.downsize(
            self.src, out, max_size=128, exempt=["C:/elsewhere/wall_Base_color.png"]
        )
        self.assertEqual(report["resized"], 0)
        self.assertFalse(os.path.exists(out))

    def test_in_place_downsize_replaces_the_source(self):
        report = FbxMedia.downsize(self.src, max_size=100)
        self.assertEqual(report["resized"], 1)
        self.assertFalse(os.path.exists(self.src + ".part"))
        self.assertEqual(FbxMedia.embedded(self.src)[0]["size"], (100, 50))

    def test_jpeg_keeps_its_container_and_other_formats_are_left_alone(self):
        src = build_fbx(
            os.path.join(self.dir, "mixed.fbx"),
            {
                "photo.jpg": jpeg_bytes((400, 300)),
                "cube.dds": b"DDS \x7c\x00\x00\x00" + b"\x00" * 200,
            },
        )
        out = os.path.join(self.dir, "mixed_small.fbx")
        report = FbxMedia.downsize(src, out, max_size=200)
        self.assertEqual(report, {**report, "images": 2, "resized": 1})
        rows = {r["name"]: r for r in FbxMedia.embedded(out)}
        self.assertEqual(
            (rows["photo.jpg"]["format"], rows["photo.jpg"]["size"]),
            ("JPEG", (200, 150)),
        )
        self.assertEqual(
            (rows["cube.dds"]["format"], rows["cube.dds"]["bytes"]), (None, 208)
        )

    def test_an_empty_content_record_is_not_an_image(self):
        src = build_fbx(os.path.join(self.dir, "ref.fbx"), {"linked.png": b""})
        self.assertEqual(FbxMedia.embedded(src), [])
        self.assertEqual(FbxMedia.downsize(src, max_size=16)["images"], 0)

    def test_not_an_fbx_raises(self):
        junk = os.path.join(self.dir, "junk.fbx")
        with open(junk, "wb") as fh:
            fh.write(b"not an fbx" * 40)
        with self.assertRaises(ValueError):
            FbxMedia.embedded(junk)


class DropTakesTestCase(unittest.TestCase):
    TAKES = ["Take 001", "Shot_A", "Shot_B"]

    def setUp(self):
        self.temp = TempArtifacts("test_fbx_media_takes", policy="scoped")
        self.dir = self.temp.dir_path()
        self.src = build_animated_fbx(
            os.path.join(self.dir, "split.fbx"),
            self.TAKES,
            declared=["Shot_A", "Shot_B"],
            media={"wall.png": png_bytes((32, 32))},
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_named_takes_go_with_everything_only_they_own(self):
        out = os.path.join(self.dir, "stripped.fbx")
        report = FbxMedia.drop_takes(self.src, out, names=["Shot_A", "Shot_B", "Nope"])
        self.assertEqual(report["takes"], ["Shot_A", "Shot_B"])
        self.assertEqual(
            report["objects"],
            {
                "AnimationStack": 2,
                "AnimationLayer": 2,
                "AnimationCurveNode": 2,
                "AnimationCurve": 2,
            },
        )
        fbx = FbxFile.load(out)
        self.assertEqual(fbx.take_names(), ["Take 001"])
        census = fbx.objects_census()
        # The curve node shared with the surviving take's layer stays, with its
        # curve; everything else survives untouched.
        self.assertEqual(census["AnimationCurveNode"], 2)
        self.assertEqual(census["AnimationCurve"], 2)
        self.assertEqual(
            (census["Model"], census["Video"], census["Texture"]), (2, 1, 1)
        )
        removed = {101, 102, 201, 202, 301, 302, 401, 402}
        rows = fbx.connections()
        self.assertFalse([r for r in rows if r[1] in removed or r[2] in removed])
        self.assertIn(("OO", 399, 200, None), rows)
        self.assertEqual(
            report["connections"], len(FbxFile.load(self.src).connections()) - len(rows)
        )
        # Takes section and Definitions agree with the objects that are left.
        takes = fbx.section("Takes")["children"]
        self.assertEqual(
            [c["props"][0] for c in takes if c["name"] == "Take"], [b"Take 001"]
        )
        self.assertEqual(
            [c["props"][0] for c in takes if c["name"] == "Current"], [b"Take 001"]
        )
        definitions = fbx.section("Definitions")["children"]
        counts = {
            c["props"][0]: c["children"][0]["props"][0]
            for c in definitions
            if c["name"] == "ObjectType"
        }
        self.assertEqual(counts[b"AnimationStack"], 1)
        self.assertEqual(counts[b"AnimationCurve"], 2)
        total = [c["props"][0] for c in definitions if c["name"] == "Count"][0]
        self.assertEqual(total, sum(counts.values()))
        # The media payload is copied, not re-encoded.
        self.assertEqual(FbxMedia.embedded(out), FbxMedia.embedded(self.src))

    def test_a_curve_node_is_dropped_once_every_owner_is(self):
        out = os.path.join(self.dir, "all_shots.fbx")
        FbxMedia.drop_takes(self.src, out, names=["Take 001", "Shot_A"])
        census = FbxFile.load(out).objects_census()
        self.assertEqual(census["AnimationCurveNode"], 1, "399 lost both owners")
        self.assertEqual(census["AnimationCurve"], 1)

    def test_no_named_take_present_writes_nothing(self):
        out = os.path.join(self.dir, "none.fbx")
        report = FbxMedia.drop_takes(self.src, out, names=["Missing"])
        self.assertEqual(report, {"takes": [], "objects": {}, "connections": 0})
        self.assertFalse(os.path.exists(out))

    def test_no_names_drops_every_take(self):
        """``names=None``: the file's static scene alone, what a build that
        reads only its geometry hands FBX2glTF -- which bakes every node at
        every frame of every take (the probe's highlight copies, 2026-10-19).
        Everything else is copied as it was; a file with no take is not
        written."""
        out = os.path.join(self.dir, "static.fbx")
        report = FbxMedia.drop_takes(self.src, out)
        self.assertEqual(report["takes"], self.TAKES)
        fbx = FbxFile.load(out)
        census = fbx.objects_census()
        for kind in ("AnimationStack", "AnimationLayer", "AnimationCurveNode", "AnimationCurve"):
            self.assertNotIn(kind, census)
        self.assertEqual((census["Model"], census["Video"], census["Texture"]), (2, 1, 1))
        self.assertEqual([c for c in fbx.section("Takes")["children"] if c["name"] == "Take"], [])
        self.assertEqual(fbx.user_properties("fbx_takes"), FbxFile.load(self.src).user_properties("fbx_takes"))
        self.assertEqual(FbxMedia.embedded(out), FbxMedia.embedded(self.src))
        again = os.path.join(self.dir, "static_again.fbx")
        self.assertEqual(FbxMedia.drop_takes(out, again)["takes"], [])
        self.assertFalse(os.path.exists(again))

    def test_user_properties_read_the_declared_channel(self):
        values = FbxFile.load(self.src).user_properties("fbx_takes")
        self.assertEqual(len(values), 1)
        self.assertEqual(
            [e["name"] for e in json.loads(values[0])], ["Shot_A", "Shot_B"]
        )
        self.assertEqual(FbxFile.load(self.src).user_properties("absent"), [])


def take_keys(path: str):
    """``{take: {channel label: [(ticks, value), ...]}}`` -- every curve's keys."""
    model = FbxFile.load(path)._take_model()
    labels = model["labels"]
    return {
        take["name"]: {
            labels[key]: list(zip(channel["curve"].times, channel["curve"].values))
            for key, channel in take["channels"].items()
        }
        for take in model["takes"]
    }


class TakeKeysTestCase(unittest.TestCase):
    """``FbxFile.take_keys``: the keys an importer reading only transforms
    leaves behind -- ``Visibility`` and the user properties -- take by take."""

    def setUp(self):
        self.temp = TempArtifacts("test_fbx_take_keys", policy="scoped")
        stepped = [(0.0, 0.0, packed_weights(), 0.0)] * 3
        self.src = build_curve_fbx(
            os.path.join(self.temp.dir_path(), "props.fbx"),
            {
                "Shot_A": {
                    ("Lamp", "Lcl Translation", "d|X"): frames(1, 10),
                    ("Lamp", "Visibility", "d|Visibility"): {
                        "keys": [(1, 1.0), (5, 0.0), (10, 0.0)],
                        "flags": [CONSTANT] * 3,
                        "data": stepped,
                    },
                    ("Lamp", "opacity", "d|opacity"): {
                        "keys": [(1, 0.0), (10, 1.0)],
                        "flags": [CUBIC | USER, LINEAR],
                        "data": stepped[:2],
                    },
                },
                "Shot_B": {("Lamp", "Lcl Translation", "d|X"): frames(20, 30)},
            },
            user=("opacity", "highlight"),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_reads_the_named_and_the_user_properties_per_take(self):
        keys = FbxFile.load(self.src).take_keys(("Visibility",), user=True)
        self.assertEqual(sorted(keys), ["Shot_A", "Shot_B"])
        self.assertEqual(
            keys["Shot_A"],
            {
                ("Lamp", "Visibility", "d|Visibility"): [
                    [FRAME, 1.0, "step"],
                    [5 * FRAME, 0.0, "step"],
                    [10 * FRAME, 0.0, "step"],
                ],
                ("Lamp", "opacity", "d|opacity"): [
                    [FRAME, 0.0, "smooth"],
                    [10 * FRAME, 1.0, "linear"],
                ],
            },
            "the transform curve is not chosen; each key keeps its own mode",
        )
        self.assertEqual(keys["Shot_B"], {}, "a take keying none of them")

    def test_a_user_property_is_read_only_when_asked_for(self):
        keys = FbxFile.load(self.src).take_keys(("Visibility",))
        self.assertEqual(list(keys["Shot_A"]), [("Lamp", "Visibility", "d|Visibility")])
        self.assertEqual(
            list(FbxFile.load(self.src).take_keys(user=True)["Shot_A"]),
            [("Lamp", "opacity", "d|opacity")],
        )


class MergeTakesTestCase(unittest.TestCase):
    """``merge_takes``: the takes ``FbxFile.take_layout`` calls live, as ONE take.

    No DCC merges takes on import (Maya plays the one ``FBXImportSetTake``
    names; Blender assigns the first stack per object), so a Shots Only file
    imported anywhere lost every shot but one.
    """

    T = ("cube", "Lcl Translation", "d|X")
    U = ("sphere", "Lcl Translation", "d|Z")
    LABEL_T = "cube.Lcl Translation.d|X"

    def setUp(self):
        self.temp = TempArtifacts("test_fbx_media_merge", policy="scoped")
        self.dir = self.temp.dir_path()

    def tearDown(self):
        self.temp.cleanup()

    def _src(self, takes, **kwargs):
        return build_curve_fbx(os.path.join(self.dir, "takes.fbx"), takes, **kwargs)

    def _windows(self):
        """Shots Only: per-frame keys at the shots' own frames, every animated
        node in every take (still where idle), a gap-0 seam between A and B,
        a gap between B and C."""
        return {
            "A": {self.T: frames(1, 10), self.U: frames(1, 10, lambda f: 3.0)},
            "B": {self.T: frames(10, 20), self.U: frames(10, 20, lambda f: 3.0)},
            "C": {self.T: frames(30, 40), self.U: frames(30, 40, lambda f: f - 27.0)},
        }

    def test_every_key_keeps_its_time_and_a_seam_is_kept_once(self):
        src = self._src(self._windows())
        out = os.path.join(self.dir, "merged.fbx")
        report = FbxMedia.merge_takes(src, out)

        self.assertEqual(report["path"], out)
        self.assertEqual(report["take"], "Take 001")
        self.assertEqual(report["takes"], ["A", "B", "C"])
        self.assertEqual(report["channels"], 2)
        self.assertEqual(report["seams"], 2)  # frame 10, on each channel
        self.assertEqual(report["keys"], 2 * (10 + 11 + 11 - 1))
        before, after = take_keys(src), take_keys(out)
        self.assertEqual(list(after), ["Take 001"])
        for label in (self.LABEL_T, "sphere.Lcl Translation.d|Z"):
            merged = dict(after["Take 001"][label])
            union = sorted({key for take in before.values() for key in take[label]})
            self.assertEqual(sorted(merged.items()), union)
            self.assertEqual(len(merged), len(union), "a seam key written twice")

    def test_a_channel_only_a_later_take_animates_is_rewired(self):
        """B animates what the kept take A does not: a component A's curve node
        lacks (cube ty), a property A has no curve node for (cube rotation), a
        model A never animates (sphere) and a layer A does not have. Each is
        wired onto the kept take, or the drop of B takes it away."""
        takes = {
            "A": {self.T: frames(1, 10)},
            "B": {
                self.T: frames(11, 20),
                ("cube", "Lcl Translation", "d|Y"): frames(11, 20),
                ("cube", "Lcl Rotation", "d|Y"): frames(11, 20),
                self.U: frames(11, 20),
                ("cube", "Lcl Scaling", "d|X", 1): frames(11, 20, lambda f: 1.0 + f),
            },
        }
        src = self._src(takes)
        out = os.path.join(self.dir, "merged.fbx")
        report = FbxMedia.merge_takes(src, out)

        self.assertEqual(report["takes"], ["A", "B"])
        self.assertGreaterEqual(report["rewired"], 4)
        after = take_keys(out)["Take 001"]
        self.assertEqual(len(after), 5)
        for key, keys in takes["B"].items():
            label = ".".join(key[:3]) + (f" [layer {key[3]}]" if len(key) > 3 else "")
            got = dict(after[label])
            for frame, value in keys:
                self.assertEqual(got[round(frame * FRAME)], value, (label, frame))
        model = FbxFile.load(out)._take_model()
        self.assertEqual(len(model["takes"][0]["layers"]), 2)
        self.assertEqual(FbxFile.load(out).take_layout()["kind"], "single")

    def test_spans_widen_and_the_kept_take_is_current(self):
        src = self._src(self._windows())
        out = os.path.join(self.dir, "merged.fbx")
        FbxMedia.merge_takes(src, out)

        fbx = FbxFile.load(out)
        self.assertEqual(fbx.take_names(), ["Take 001"])
        stack = next(r for r in fbx.iter_objects() if r["name"] == "AnimationStack")
        spans = {p["props"][0]: p["props"][4] for p in stack["children"][0]["children"]}
        whole = (1 * FRAME, 40 * FRAME)
        self.assertEqual((spans[b"LocalStart"], spans[b"LocalStop"]), whole)
        self.assertEqual((spans[b"ReferenceStart"], spans[b"ReferenceStop"]), whole)
        takes = fbx.section("Takes")["children"]
        self.assertEqual(
            [(c["name"], c["props"][0]) for c in takes],
            [("Current", b"Take 001"), ("Take", b"Take 001")],
        )
        entry = {c["name"]: c["props"] for c in takes[1]["children"]}
        self.assertEqual(entry["FileName"], [b"Take_001.tak"])
        self.assertEqual(tuple(entry["LocalTime"]), whole)
        self.assertEqual(tuple(entry["ReferenceTime"]), whole)
        definitions = fbx.section("Definitions")["children"]
        counts = {
            c["props"][0]: c["children"][0]["props"][0]
            for c in definitions
            if c["name"] == "ObjectType"
        }
        census = fbx.objects_census()
        self.assertEqual(
            {k.decode(): v for k, v in counts.items()},
            {k: census.get(k, 0) for k in (k.decode() for k in counts)},
        )
        self.assertEqual(census["AnimationStack"], 1)

    def test_overlapping_takes_that_agree_interleave_their_keys(self):
        """Merged on request, a sparse cover and a per-frame slice share the
        slice's frames: each time is kept once, and past the slice's last key
        the curve runs on to the cover's next key along the cover's line."""
        src = self._src(
            {
                "Scene": {self.T: [(1, 0.5), (100, 50.0)]},
                "early": {self.T: frames(1, 30)},
            }
        )
        out = os.path.join(self.dir, "merged.fbx")
        report = FbxMedia.merge_takes(src, out, takes=["Scene", "early"])

        self.assertEqual((report["keys"], report["seams"]), (31, 1))
        model = FbxFile.load(out)._take_model()
        (channel,) = model["takes"][0]["channels"].values()
        curve = channel["curve"]
        self.assertEqual(curve.times, [f * FRAME for f in list(range(1, 31)) + [100]])
        for frame in (10.5, 30, 45, 60, 99):
            self.assertAlmostEqual(
                FbxFile._evaluate_curve(curve, round(frame * FRAME)), frame * 0.5
            )

    #: Maya's computed tangents as its exporter writes them: cubic keys whose
    #: slopes the importer recomputes from the neighbouring keys.
    COMPUTED = {
        "spline": CUBIC | AUTO | TIME_INDEPENDENT,
        "clamped": CUBIC | AUTO | TIME_INDEPENDENT | CLAMP,
        "auto": CUBIC | AUTO | TIME_INDEPENDENT | CLAMP_PROGRESSIVE,
    }

    def test_each_take_plays_its_own_tangents_beside_a_boundary(self):
        """Bug (review 2026-10-12): a take's last key and the next take's first
        key are end keys alone, and merged each gained the other as its
        neighbour, so the importer recomputed their tangents and every edge
        segment changed shape -- auto tangents, take A at 7.5 played 8.19
        alone and 6.32 merged.
        Fixed: 2026-10-13 -- each key whose neighbours the merge changed is
        written with the slopes it plays alone (a user tangent, broken where
        its sides differ)."""
        a = [(1, 0.0), (4, 6.0), (8, 2.0), (10, 9.0)]
        b = [(14, 1.0), (17, 9.0), (20, 3.0)]
        layouts = {"gap": (a, b), "seam": (a, [(10, 9.0)] + b)}
        sampled = 0
        for kind, flag in self.COMPUTED.items():
            for layout, (keys_a, keys_b) in layouts.items():

                def spec(keys):
                    data = [(0.0, 0.0, packed_weights(), 0.0)] * len(keys)
                    return {"keys": keys, "flags": [flag] * len(keys), "data": data}

                src = self._src(
                    {"A": {self.T: spec(keys_a)}, "B": {self.T: spec(keys_b)}}
                )
                out = os.path.join(self.dir, f"merged_{kind}_{layout}.fbx")
                FbxMedia.merge_takes(src, out)
                alone = {
                    take["name"]: next(iter(take["channels"].values()))["curve"]
                    for take in FbxFile.load(src)._take_model()["takes"]
                }
                # The key-by-key union (overlapping takes) settles alike.
                self.assertEqual(
                    FbxFile._union_curves([alone["A"], alone["B"]]),
                    FbxFile._interleave_curves([alone["A"], alone["B"]]),
                )
                model = FbxFile.load(out)._take_model()
                merged = next(iter(model["takes"][0]["channels"].values()))["curve"]
                union = sorted({f for f, _v in keys_a + keys_b})
                self.assertEqual(merged.times, [f * FRAME for f in union])
                for take, keys in (("A", keys_a), ("B", keys_b)):
                    first, last = keys[0][0], keys[-1][0]
                    for step in range(int((last - first) * 4) + 1):
                        ticks = round((first + step / 4) * FRAME)
                        self.assertAlmostEqual(
                            FbxFile._evaluate_curve(merged, ticks),
                            FbxFile._evaluate_curve(alone[take], ticks),
                            delta=1e-4,
                            msg=(kind, layout, take, first + step / 4),
                        )
                        sampled += 1
        self.assertGreater(sampled, 300)

    def test_disagreeing_takes_raise_naming_take_channel_and_frame(self):
        src = self._src(
            {
                "Walk": {self.T: frames(1, 30)},
                "Run": {self.T: frames(1, 30, lambda f: f * 0.75)},
            }
        )
        out = os.path.join(self.dir, "merged.fbx")
        with self.assertRaises(ValueError) as caught:
            FbxMedia.merge_takes(src, out, takes=["Walk", "Run"])
        message = str(caught.exception)
        for part in ("Walk", "Run", self.LABEL_T, "frame 1"):
            self.assertIn(part, message)
        self.assertFalse(os.path.exists(out))

    def test_the_default_merges_the_live_takes_and_drops_the_rest(self):
        takes = self._windows()
        takes["Alt"] = {self.T: frames(1, 10, lambda f: -f)}
        src = self._src(takes)
        self.assertEqual(FbxFile.load(src).take_layout()["stash"], ["Alt"])
        out = os.path.join(self.dir, "merged.fbx")
        report = FbxMedia.merge_takes(src, out, name="Shots")

        self.assertEqual(report["takes"], ["A", "B", "C"])
        self.assertEqual(report["dropped"], ["Alt"])
        self.assertEqual(FbxFile.load(out).take_names(), ["Shots"])
        merged = dict(take_keys(out)["Shots"][self.LABEL_T])
        self.assertEqual(merged[1 * FRAME], 0.5, "Alt's motion leaked in")

    def test_one_take_merges_to_itself(self):
        src = self._src({"Clip": {self.T: frames(1, 30), self.U: frames(5, 9)}})
        out = os.path.join(self.dir, "merged.fbx")
        report = FbxMedia.merge_takes(src, out, name=None)

        self.assertEqual(
            (report["take"], report["seams"], report["rewired"]), ("Clip", 0, 0)
        )
        self.assertEqual(take_keys(out), take_keys(src))
        self.assertEqual(FbxFile.load(out).take_layout()["kind"], "single")

    def test_an_unknown_take_raises(self):
        src = self._src(self._windows())
        with self.assertRaises(ValueError) as caught:
            FbxMedia.merge_takes(src, os.path.join(self.dir, "x.fbx"), takes=["Nope"])
        self.assertIn("Nope", str(caught.exception))

    def test_in_place_merge_replaces_the_source(self):
        src = self._src(self._windows())
        FbxMedia.merge_takes(src)
        self.assertEqual(FbxFile.load(src).take_names(), ["Take 001"])
        self.assertFalse([n for n in os.listdir(self.dir) if n.endswith(".part")])

    def test_kept_takes_stay_whole_beside_the_merge(self):
        """A layout's stash rides beside the merged take, untouched, so ONE
        import carries what plays and what an importer parks (Blender makes
        an action per stack and object)."""
        takes = self._windows()
        takes["Alt"] = {self.T: frames(1, 10, lambda f: -f)}
        src = self._src(takes)
        out = os.path.join(self.dir, "merged.fbx")
        report = FbxMedia.merge_takes(src, out, keep=["Alt"])

        self.assertEqual(report["takes"], ["A", "B", "C"])
        self.assertEqual(report["dropped"], [])
        fbx = FbxFile.load(out)
        self.assertEqual(fbx.take_names(), ["Take 001", "Alt"])
        self.assertEqual(take_keys(out)["Alt"], take_keys(src)["Alt"])
        layout = fbx.take_layout()
        self.assertEqual(
            (layout["current"], layout["live"], layout["stash"]),
            ("Take 001", ["Take 001"], ["Alt"]),
        )
        self.assertEqual(fbx.objects_census()["AnimationStack"], 2)

    def test_keep_refuses_a_merged_take_its_name_or_an_unknown_take(self):
        takes = self._windows()
        takes["Alt"] = {self.T: frames(1, 10, lambda f: -f)}
        src = self._src(takes)
        out = os.path.join(self.dir, "x.fbx")
        for kwargs, part in (
            ({"keep": ["A"]}, "'A'"),
            ({"keep": ["Nope"]}, "Nope"),
            ({"keep": ["Alt"], "name": "Alt"}, "'Alt'"),
        ):
            with self.assertRaises(ValueError, msg=kwargs) as caught:
                FbxMedia.merge_takes(src, out, **kwargs)
            self.assertIn(part, str(caught.exception))
            self.assertFalse(os.path.exists(out))

    def test_a_corrupt_curve_array_is_a_value_error(self):
        """Bug (review 2026-10-12): a curve array whose zlib stream is broken
        raised ``zlib.error`` out of the merge, past the callers' ``(OSError,
        ValueError)`` fallbacks; ``FbxFile.load`` already said ValueError."""
        src = self._src(
            {"A": {self.T: frames(1, 300)}, "B": {self.T: frames(300, 600)}}
        )
        with open(src, "rb") as fh:
            data = bytearray(fh.read())
        at = data.find(b"KeyTime") + len(b"KeyTime")
        self.assertEqual(data[at : at + 1], b"l")  # an int64 array ...
        self.assertEqual(int.from_bytes(data[at + 5 : at + 9], "little"), 1)  # zlib
        for i in range(at + 15, at + 53):  # past the header, into the stream
            data[i] ^= 0xFF
        with open(src, "wb") as fh:
            fh.write(bytes(data))
        out = os.path.join(self.dir, "x.fbx")
        with self.assertRaises(ValueError):
            FbxMedia.merge_takes(src, out)
        self.assertFalse(os.path.exists(out))

    def test_a_layer_named_after_its_take_takes_the_merged_name(self):
        """Blender names each layer after its take, and its importer names an
        action ``<object>|<take>|<layer>`` once they differ -- a merged Shots
        Only file arrived as ``cube|Take 001|A``, named after one shot."""
        takes = self._windows()
        takes["Alt"] = {self.T: frames(1, 10, lambda f: -f)}
        src = self._src(takes, base_layer=lambda take: take)
        out = os.path.join(self.dir, "merged.fbx")
        FbxMedia.merge_takes(src, out, keep=["Alt"])

        fbx = FbxFile.load(out)
        layers = [
            FbxFile._display_name(r["props"][1])
            for r in fbx.iter_objects()
            if r["name"] == "AnimationLayer"
        ]
        self.assertEqual(sorted(layers), ["Alt", "Take 001"])


def encoded(image, fmt: str = "PNG") -> bytes:
    out = io.BytesIO()
    image.save(out, format=fmt)
    return out.getvalue()


class ExpandGrayscaleTestCase(unittest.TestCase):
    """``expand_grayscale`` gives every grayscale embed three colour channels.

    FBX2glTF packs a material's occlusion-roughness-metallic texture from each
    map's red, green and blue, and reads a channel the decoded image lacks as
    white. The converter-level proof is
    ``test_mesh_convert.TestGrayscaleMapsThroughTheConverter``.
    """

    def setUp(self):
        from PIL import Image

        self.temp = TempArtifacts("test_fbx_media_gray", policy="scoped")
        self.dir = self.temp.dir_path()
        self.ramp = Image.new("L", (32, 4))
        self.ramp.putdata([8 * x for _y in range(4) for x in range(32)])
        sixteen = self.ramp.convert("I").point(lambda v: v * 257).convert("I;16")
        self.gray = {
            "rough_Roughness.png": encoded(self.ramp),
            "cut_Opacity.png": encoded(self.ramp.convert("LA")),
            "bits_Mask.png": encoded(self.ramp.convert("1")),
            "deep_Height.png": encoded(sixteen),
            "ao_Ambient_Occlusion.jpg": encoded(self.ramp, "JPEG"),
        }
        self.colour = {
            "wall_Base_color.png": png_bytes((32, 4)),
            "decal_Mask.png": encoded(self.ramp.convert("P")),
            "cube.dds": b"DDS \x7c\x00\x00\x00" + b"\x00" * 200,
        }
        self.src = build_fbx(
            os.path.join(self.dir, "gray.fbx"), {**self.gray, **self.colour}
        )

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _payloads(path):
        """``{basename: embedded bytes}`` of every ``Video`` in *path*."""
        found = {}
        for record in FbxFile.load(path).iter_objects():
            fields = {child["name"]: child["props"] for child in record["children"]}
            if record["name"] == "Video" and fields.get("Content"):
                name = os.path.basename(fields["Filename"][0].decode())
                found[name] = fields["Content"][0]
        return found

    def test_each_grayscale_embed_carries_its_value_in_every_channel(self):
        from PIL import Image

        out = os.path.join(self.dir, "expanded.fbx")
        report = FbxMedia.expand_grayscale(self.src, out)
        self.assertEqual((report["images"], report["expanded"]), (8, len(self.gray)))
        payloads = self._payloads(out)
        modes = {}
        for name in self.gray:
            with Image.open(io.BytesIO(payloads[name])) as image:
                image.load()
                modes[name] = (image.format, image.mode)
                if image.format == "PNG":  # lossless: every channel IS the gray
                    red, green, blue = image.split()[:3]
                    self.assertEqual(red.tobytes(), green.tobytes(), name)
                    self.assertEqual(red.tobytes(), blue.tobytes(), name)
        self.assertEqual(
            modes,
            {
                "rough_Roughness.png": ("PNG", "RGB"),
                "cut_Opacity.png": ("PNG", "RGBA"),
                "bits_Mask.png": ("PNG", "RGB"),
                "deep_Height.png": ("PNG", "RGB"),
                "ao_Ambient_Occlusion.jpg": ("JPEG", "RGB"),
            },
        )
        for name in ("rough_Roughness.png", "cut_Opacity.png", "deep_Height.png"):
            with Image.open(io.BytesIO(payloads[name])) as image:
                # The 16-bit map full scale: 65535 is 255, not a clipped 255
                # for everything above it (Pillow's own "I" -> "L").
                self.assertEqual(
                    image.getchannel("G").tobytes(), self.ramp.tobytes(), name
                )
        # Already colour, or not PNG/JPEG: copied byte for byte.
        for name, data in self.colour.items():
            self.assertEqual(payloads[name], data, name)
        before, after = FbxFile.load(self.src), FbxFile.load(out)
        self.assertEqual(before.objects_census(), after.objects_census())
        self.assertEqual(before.connections(), after.connections())

    def test_a_file_with_no_grayscale_embed_is_not_written(self):
        src = build_fbx(os.path.join(self.dir, "colour.fbx"), self.colour)
        out = os.path.join(self.dir, "unchanged.fbx")
        report = FbxMedia.expand_grayscale(src, out)
        self.assertEqual((report["images"], report["expanded"]), (3, 0))
        self.assertFalse(os.path.exists(out))


#: ``(id, name, class, parent)`` of every Model :func:`build_rigged_fbx` writes:
#: a skinned tube, the chain it is bound to, and the rig that used to drive it.
RIG_MODELS = (
    (1001, "tube_GEO", "Mesh", 0),
    (1101, "tube_RIG", "Null", 0),
    (1102, "tube_jnt_1", "LimbNode", 1101),
    (1103, "tube_jnt_2", "LimbNode", 1102),
    (1110, "tube_proxy_GRP", "Null", 1101),
    (1111, "tube_proxy_jnt_1", "LimbNode", 1110),
    (1112, "tube_proxy_jnt_2", "LimbNode", 1111),
    (1120, "tube_ik_curve", "Line", 1110),
    (1130, "tube_driver_GRP", "Null", 1101),
    (1131, "tube_driver_jnt", "LimbNode", 1130),
    (1140, "tube_tweak_CTRL_GRP", "Null", 1101),
    (1141, "tube_tweak_CTRL", "Line", 1140),
    (1150, "tube_settings_CTRL", "Line", 1101),
    (1160, "tube_marker_GRP", "Null", 1101),
    (1161, "tube_marker_GEO", "Mesh", 1160),
    (1170, "tube_aux_curve", "Line", 1101),
    (1171, "tube_aux_jnt", "LimbNode", 1101),
    (1200, "data_export", "Null", 0),
)

#: What a census of that scene names -- everything but the tube, its bind chain
#: and the aux curve -- including four nodes the file itself must refuse.
RIG_SECTION = {
    "|tube_RIG|tube_proxy_GRP": "group",
    "|tube_RIG|tube_proxy_GRP|tube_proxy_jnt_1": "joint",
    "|tube_RIG|tube_proxy_GRP|tube_proxy_jnt_1|tube_proxy_jnt_2": "joint",
    "|tube_RIG|tube_proxy_GRP|tube_ik_curve": "control",
    "|tube_RIG|tube_driver_GRP": "group",
    "|tube_RIG|tube_driver_GRP|tube_driver_jnt": "joint",
    "|tube_RIG|tube_tweak_CTRL_GRP": "group",
    "|tube_RIG|tube_tweak_CTRL_GRP|tube_tweak_CTRL": "control",
    "|tube_RIG|tube_settings_CTRL": "control",
    "|tube_RIG|tube_marker_GRP": "group",
    "|tube_RIG|tube_aux_jnt": "joint",
    "|data_export": "locator",
}


def build_rigged_fbx(path: str, curve_pose: bool = False) -> str:
    """A skinned tube with the rig that drove it, laid out as Maya writes one.

    The mesh is skinned to ``tube_jnt_1/2``. The apparatus: a proxy chain
    under a group, the IK curve (skinned to ``tube_driver_jnt``), a tweak
    control in its offset group, animated proxy and tweak, a proxy joint in
    the bind pose. Four named nodes the file itself must keep: a control wired
    to a material (an object the writer does not own), a group holding a mesh,
    a joint deforming a curve that stays (``tube_aux_curve`` is not named) and
    the ``data_export`` carrier holding a scene record. *curve_pose* adds the
    IK curve's own bind pose, holding the curve alone -- Maya writes one per
    skinCluster.
    """

    def p(*values):
        return ("P", list(values), [])

    objects, connections = [], []
    for oid, name, kind, parent in RIG_MODELS:
        props = []
        if name == "data_export":
            props.append(p(b"shot_metadata", b"KString", b"", b"U", b"{}"))
        objects.append(
            (
                "Model",
                [oid, name.encode() + b"\x00\x01Model", kind.encode()],
                [("Version", [I32(232)], []), ("Properties70", [], props)],
            )
        )
        connections.append(("C", [b"OO", oid, parent], []))
        if kind in ("Null", "LimbNode"):
            attribute = oid + 3000
            objects.append(
                (
                    "NodeAttribute",
                    [attribute, b"\x00\x01NodeAttribute", kind.encode()],
                    [],
                )
            )
            connections.append(("C", [b"OO", attribute, oid], []))
        else:
            geometry = oid + 1000
            objects.append(
                (
                    "Geometry",
                    [geometry, b"\x00\x01Geometry", kind.encode()],
                    [],
                )
            )
            connections.append(("C", [b"OO", geometry, oid], []))

    def skin(skin_id, geometry, influences):
        objects.append(("Deformer", [skin_id, b"\x00\x01Deformer", b"Skin"], []))
        connections.append(("C", [b"OO", skin_id, geometry], []))
        for index, influence in enumerate(influences, 1):
            cluster = skin_id + index
            objects.append(
                ("Deformer", [cluster, b"\x00\x01SubDeformer", b"Cluster"], [])
            )
            connections.append(("C", [b"OO", cluster, skin_id], []))
            connections.append(("C", [b"OO", influence, cluster], []))

    skin(7001, 2001, [1102, 1103])  # the tube, on its bind chain
    skin(7120, 2120, [1131])  # the IK curve, on the driver joint
    skin(7170, 2170, [1171])  # a curve that stays, on a named joint

    objects.append(("Material", [5150, b"ctrl\x00\x01Material", b""], []))
    connections.append(("C", [b"OO", 5150, 1150], []))

    objects += [
        ("AnimationStack", [100, b"Take 001\x00\x01AnimStack", b""], []),
        ("AnimationLayer", [200, b"BaseLayer\x00\x01AnimLayer", b""], []),
    ]
    connections.append(("C", [b"OO", 200, 100], []))
    for node, target in ((300, 1111), (301, 1141), (302, 1102)):
        curve = node + 100
        objects += [
            ("AnimationCurveNode", [node, b"R\x00\x01AnimCurveNode", b""], []),
            ("AnimationCurve", [curve, b"\x00\x01AnimCurve", b""], []),
        ]
        connections += [
            ("C", [b"OO", node, 200], []),
            ("C", [b"OP", node, target, b"Lcl Rotation"], []),
            ("C", [b"OP", curve, node, b"d|X"], []),
        ]

    pose_nodes = [
        (
            "PoseNode",
            [],
            [("Node", [member], []), ("Matrix", [Doubles([1.0] * 16)], [])],
        )
        for member in (1001, 1102, 1103, 1111)
    ]
    objects.append(
        (
            "Pose",
            [6000, b"BindPose\x00\x01Pose", b"BindPose"],
            [
                ("Type", [b"BindPose"], []),
                ("Version", [I32(100)], []),
                ("NbPoseNodes", [I32(len(pose_nodes))], []),
            ]
            + pose_nodes,
        )
    )
    if curve_pose:
        objects.append(
            (
                "Pose",
                [6001, b"tube_ik_curveSkin\x00\x01Pose", b"BindPose"],
                [
                    ("Type", [b"BindPose"], []),
                    ("Version", [I32(100)], []),
                    ("NbPoseNodes", [I32(1)], []),
                    (
                        "PoseNode",
                        [],
                        [("Node", [1120], []), ("Matrix", [Doubles([1.0] * 16)], [])],
                    ),
                ],
            )
        )

    counts = {}
    for kind, _props, _children in objects:
        counts[kind] = counts.get(kind, 0) + 1
    definitions = (
        "Definitions",
        [],
        [("Version", [I32(100)], []), ("Count", [I32(sum(counts.values()))], [])]
        + [
            ("ObjectType", [kind.encode()], [("Count", [I32(n)], [])])
            for kind, n in counts.items()
        ],
    )
    roots = [
        definitions,
        ("Objects", [], objects),
        ("Connections", [], connections),
        _takes_section({"Take 001": (0, 46186158000)}),
    ]
    return _write_fbx(path, roots)


class DropApparatusTestCase(unittest.TestCase):
    """``drop_apparatus`` removes what a census named and the file agrees on."""

    DROPPED = {1110, 1111, 1112, 1120, 1130, 1131, 1140, 1141}

    def setUp(self):
        self.temp = TempArtifacts("test_fbx_media_rig", policy="scoped")
        self.dir = self.temp.dir_path()
        self.src = build_rigged_fbx(os.path.join(self.dir, "rigged.fbx"))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _models(fbx):
        return {
            record["props"][0]
            for record in fbx.iter_objects()
            if record["name"] == "Model"
        }

    def test_the_apparatus_goes_with_everything_only_it_owns(self):
        out = os.path.join(self.dir, "lean.fbx")
        report = FbxMedia.drop_apparatus(self.src, out, section=RIG_SECTION)
        self.assertEqual(report["models"], len(self.DROPPED))
        self.assertEqual(report["kinds"], {"control": 2, "group": 3, "joint": 3})
        fbx = FbxFile.load(out)
        self.assertEqual(self._models(fbx), {m[0] for m in RIG_MODELS} - self.DROPPED)
        # Six attributes, the IK curve's and the tweak's geometry, the curve's
        # skin and its cluster, and the proxy's and the tweak's animation. The
        # bind joint's curve node, the layer and the mesh's skin all stay.
        self.assertEqual(
            report["objects"],
            {
                "Model": 8,
                "NodeAttribute": 6,
                "Geometry": 2,
                "Deformer": 2,
                "AnimationCurveNode": 2,
                "AnimationCurve": 2,
            },
        )
        census = fbx.objects_census()
        self.assertEqual(census["AnimationCurveNode"], 1)
        self.assertEqual(census["AnimationLayer"], 1)
        self.assertEqual(census["Deformer"], 3 + 2)  # both surviving skins
        gone = set(self.DROPPED) | {3000 + m for m in self.DROPPED if m != 1120}
        gone |= {2120, 2141, 7120, 7121, 300, 301, 400, 401}
        self.assertTrue(gone.isdisjoint({c[1] for c in fbx.connections()}))
        self.assertTrue(gone.isdisjoint({c[2] for c in fbx.connections()}))
        self.assertEqual(
            report["connections"],
            len(FbxFile.load(self.src).connections()) - len(fbx.connections()),
        )
        # The bind pose loses the proxy joint's entry and is recounted.
        pose = next(r for r in fbx.iter_objects() if r["name"] == "Pose")
        members = [
            c["children"][0]["props"][0]
            for c in pose["children"]
            if c["name"] == "PoseNode"
        ]
        self.assertEqual(members, [1001, 1102, 1103])
        self.assertEqual(
            [c["props"][0] for c in pose["children"] if c["name"] == "NbPoseNodes"],
            [3],
        )
        # Definitions agree with the objects left.
        definitions = fbx.section("Definitions")["children"]
        counts = {
            c["props"][0].decode(): c["children"][0]["props"][0]
            for c in definitions
            if c["name"] == "ObjectType"
        }
        self.assertEqual({k: v for k, v in counts.items() if v}, census)
        total = [c["props"][0] for c in definitions if c["name"] == "Count"][0]
        self.assertEqual(total, sum(census.values()))

    def test_the_file_refuses_what_it_can_see_must_stay(self):
        report = FbxMedia.drop_apparatus(
            self.src, os.path.join(self.dir, "lean.fbx"), section=RIG_SECTION
        )
        # A control wired to a material, a group holding a mesh, a joint
        # deforming a curve nobody named, and the carrier of a scene record.
        self.assertEqual(
            report["refused"],
            ["data_export", "tube_aux_jnt", "tube_marker_GRP", "tube_settings_CTRL"],
        )

    def test_a_curve_influence_stays_while_its_curve_does(self):
        section = {
            k: v for k, v in RIG_SECTION.items() if not k.endswith("tube_ik_curve")
        }
        section = {k: v for k, v in section.items() if "proxy_GRP" not in k}
        report = FbxMedia.drop_apparatus(
            self.src, os.path.join(self.dir, "lean.fbx"), section=section
        )
        self.assertIn("tube_driver_jnt", report["refused"])
        self.assertIn("tube_driver_GRP", report["refused"])
        self.assertEqual(report["kinds"], {"control": 1, "group": 1})

    def test_a_bind_pose_it_empties_goes_with_its_members(self):
        """Maya writes a bind pose per skin, and the IK curve's holds the curve
        alone. The drop took the curve and kept its pose with no members --
        and Maya's importer, handed empty bind poses, bound the skins that
        stayed at the wrong pose: 6 of the 7 production wire looms 97-327 cm
        off, each exact once the seven emptied poses were gone (2026-10-17).
        An emptied pose goes, with its count; a pose that keeps a member
        stays.
        Added: 2026-10-17"""
        src = build_rigged_fbx(os.path.join(self.dir, "poses.fbx"), curve_pose=True)
        out = os.path.join(self.dir, "lean.fbx")
        report = FbxMedia.drop_apparatus(src, out, section=RIG_SECTION)
        fbx = FbxFile.load(out)
        poses = [r["props"][0] for r in fbx.iter_objects() if r["name"] == "Pose"]
        self.assertEqual(poses, [6000], "the emptied pose shipped")
        self.assertEqual(report["objects"].get("Pose"), 1)
        census = fbx.objects_census()
        definitions = fbx.section("Definitions")["children"]
        counts = {
            c["props"][0].decode(): c["children"][0]["props"][0]
            for c in definitions
            if c["name"] == "ObjectType"
        }
        self.assertEqual({k: v for k, v in counts.items() if v}, census)
        total = [c["props"][0] for c in definitions if c["name"] == "Count"][0]
        self.assertEqual(total, sum(census.values()))

    def test_nothing_to_drop_writes_nothing(self):
        out = os.path.join(self.dir, "same.fbx")
        for section in ({}, {"|elsewhere|ghost_CTRL": "control"}):
            report = FbxMedia.drop_apparatus(self.src, out, section=section)
            self.assertEqual(report["models"], 0)
            self.assertFalse(os.path.exists(out))

    def test_dropping_again_changes_nothing(self):
        FbxMedia.drop_apparatus(self.src, section=RIG_SECTION)  # in place
        self.assertFalse(os.path.exists(self.src + ".part"))
        out = os.path.join(self.dir, "again.fbx")
        report = FbxMedia.drop_apparatus(self.src, out, section=RIG_SECTION)
        self.assertEqual(report["models"], 0)
        self.assertFalse(os.path.exists(out))


def build_carrier_fbx(
    path: str, channels, media=None, definitions=True, others=None
) -> str:
    """An FBX shaped like a Scene Exporter's: a mesh, a ``data_export`` carrier
    holding *channels* ``{record key: payload}`` as user properties, and the
    material media *media* ``{name: bytes}`` (each Video wired to a Texture).
    *definitions* writes the ``Definitions`` counts the SDK writes. *others*
    ``{carrier name: channels}`` are further carriers written AHEAD of the
    scene's own (a module's ``NS:data_export``, a stray ``data_export1``)."""

    def carrier(uid, name, held):
        props = [
            ("P", [k.encode(), b"KString", b"", b"U", json.dumps(v).encode()], [])
            for k, v in held.items()
        ]
        label = name.encode() + b"\x00\x01Model"
        return ("Model", [uid, label, b"Null"], [("Properties70", [], props)])

    extra = [
        carrier(2001 + i, name, held)
        for i, (name, held) in enumerate((others or {}).items())
    ]
    objects = [
        ("Model", [1001, b"Crate\x00\x01Model", b"Mesh"], []),
        *extra,
        carrier(1002, "data_export", channels),
    ]
    media_objects, media_connections = _media_records(media or {})
    objects += media_objects
    connections = [
        ("C", [b"OO", 1001, 0], []),
        ("C", [b"OO", 1002, 0], []),
        *(("C", [b"OO", record[1][0], 0], []) for record in extra),
    ] + media_connections
    roots = []
    if definitions:
        counts = {}
        for kind, _props, _children in objects:
            counts[kind] = counts.get(kind, 0) + 1
        roots.append(
            (
                "Definitions",
                [],
                [
                    ("Version", [I32(100)], []),
                    ("Count", [I32(sum(counts.values()))], []),
                ]
                + [
                    ("ObjectType", [kind.encode()], [("Count", [I32(n)], [])])
                    for kind, n in counts.items()
                ],
            )
        )
    roots += [("Objects", [], objects), ("Connections", [], connections)]
    return _write_fbx(path, roots)


class EmbedTestCase(unittest.TestCase):
    """``embed`` / ``embed_dependencies``: an FBX deliverable carries every file
    its scene records name by file name, so nothing ships beside it.

    The form is the one Unity's FBX importer was MEASURED to hand back (2022.3
    and 6000.3, 2026-10-05): a bare ``Video`` with ``Content`` and no Texture
    or connection is extracted by ``ModelImporter.ExtractTextures`` byte for
    byte under its file name -- EXR, HDR, 8- and 16-bit PNG, WAV alike -- and
    leaves every material as authored.
    """

    LIGHTMAPS = {
        "version": 1,
        "objects": [
            {"name": "Crate", "map": "Room_Lightmap.exr", "uvIndex": 1},
            {"name": "Shelf", "map": "Room_Lightmap.exr", "uvIndex": 1},
            {"name": "Lamp", "map": "Lamp_Lightmap.exr", "uvIndex": 1},
        ],
        "probe": {"map": "Room_Probe.exr", "position": [0, 1, 2], "unit_scale": 0.01},
    }
    SHADOWS = {
        "version": 2,
        "unit_scale": 0.01,
        "planes": [
            {
                "name": "Box_shadow",
                "texture": "Box_shadow.png",
                "type": "horizon",
                "atlas": {"texture": "shadow_atlas_horizon.png", "rect": [1, 1, 0, 0]},
                "horizon": {"texture": "Box_horizon.png", "encoding": 2},
            }
        ],
    }
    #: What the records above name, in record order, the shared atlas once.
    NAMED = [
        "Room_Lightmap.exr",
        "Lamp_Lightmap.exr",
        "Room_Probe.exr",
        "Box_shadow.png",
        "shadow_atlas_horizon.png",
        "Box_horizon.png",
    ]

    def setUp(self):
        self.temp = TempArtifacts("test_fbx_media_embed", policy="scoped")
        self.dir = self.temp.dir_path()
        self.maps = os.path.join(self.dir, "sourceimages")
        os.makedirs(self.maps)
        # Distinct bytes per file, so a payload proves which file it came from.
        self.files = {
            name: name.encode() * 40 + bytes(range(256))
            for name in self.NAMED
            if name != "Box_shadow.png"
        }
        for name, data in self.files.items():
            with open(os.path.join(self.maps, name), "wb") as fh:
                fh.write(data)
        # The silhouette rides the plane's MATERIAL, as an Embed Media export
        # writes it.
        self.silhouette = png_bytes((16, 16), "RGBA")
        self.src = build_carrier_fbx(
            os.path.join(self.dir, "deliverable.fbx"),
            {"lightmap_metadata": self.LIGHTMAPS, "shadow_metadata": self.SHADOWS},
            media={"Box_shadow.png": self.silhouette},
        )

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _videos(path):
        """``{Video id: {field: first prop}}`` of every Video in *path*."""
        found = {}
        for record in FbxFile.load(path).iter_objects():
            if record["name"] != "Video":
                continue
            fields = {c["name"]: c["props"] for c in record["children"]}
            found[record["props"][0]] = {
                k: v[0] if v else None for k, v in fields.items()
            }
        return found

    @staticmethod
    def _counts(path):
        """``(Definitions Count, {ObjectType: Count})`` of *path*."""
        section = FbxFile.load(path).section("Definitions")
        total, kinds = None, {}
        for child in section["children"]:
            if child["name"] == "Count":
                total = child["props"][0]
            elif child["name"] == "ObjectType":
                kinds[child["props"][0].decode()] = next(
                    c["props"][0] for c in child["children"] if c["name"] == "Count"
                )
        return total, kinds

    def test_dependencies_are_the_file_names_the_records_name(self):
        self.assertEqual(FbxMedia.dependencies(self.src), self.NAMED)
        # No records: nothing named.
        bare = build_carrier_fbx(os.path.join(self.dir, "bare.fbx"), {})
        self.assertEqual(FbxMedia.dependencies(bare), [])

    def test_dependencies_read_every_carrier_and_skip_what_is_not_a_name(self):
        """A referenced module ships its own ``NS:data_export``; a malformed
        channel, a blank name or a folder-qualified one (an older manifest)
        must not stop the rest being read."""
        legacy = {"objects": [{"name": "A", "map": "C:/bake/Old.exr"}, {"name": "B"}]}
        path = build_carrier_fbx(
            os.path.join(self.dir, "two.fbx"), {"lightmap_metadata": self.LIGHTMAPS}
        )
        other = build_carrier_fbx(
            os.path.join(self.dir, "legacy.fbx"),
            {"lightmap_metadata": legacy, "shadow_metadata": "not json {"},
        )
        self.assertEqual(
            FbxMedia.dependencies(path),
            ["Room_Lightmap.exr", "Lamp_Lightmap.exr", "Room_Probe.exr"],
        )
        self.assertEqual(FbxMedia.dependencies(other), ["Old.exr"])

    def test_the_scenes_own_carrier_names_a_file_before_any_other(self):
        """Names compare without case, so the first carrier read spells a name
        for all: a stray ``data_export1`` (an older manifest an import left)
        written ahead of the scene's own named ``Room_LightMap.exr``, and the
        embed resolved THAT spelling -- on a case-folding volume, the stale
        same-named atlas the scene had stopped reading. The scene's own
        carrier is read first, as the GLB applier reads it."""
        stale = {
            "version": 1,
            "objects": [{"name": "Crate", "map": "ROOM_LIGHTMAP.exr"}],
        }
        path = build_carrier_fbx(
            os.path.join(self.dir, "stray.fbx"),
            {"lightmap_metadata": self.LIGHTMAPS},
            others={"data_export1": {"lightmap_metadata": stale}},
        )
        self.assertEqual(
            FbxMedia.dependencies(path),
            ["Room_Lightmap.exr", "Lamp_Lightmap.exr", "Room_Probe.exr"],
        )

    #: A referenced module's lightmaps as that module was last exported
    #: alone: its own spelling of each name (no namespace), a probe.
    MODULE_LIGHTMAPS = {
        "version": 1,
        "objects": [
            {"name": "ModA", "map": "ModA_Module.exr", "uvIndex": 1},
            {"name": "ModB", "map": "ModB_Module.exr", "uvIndex": 1},
        ],
        "probe": {"map": "Module_Probe.exr", "position": [0, 0, 0]},
    }

    def test_a_module_record_the_scenes_own_shadows_names_nothing(self):
        """A reader takes each record from the scene's own carrier and reads
        a module's only where the scene's carries none (unitytk
        ``CarrierImport.Collect``, the GLB applier) -- so a module's copy
        beside the scene's names no file the deliverable needs: its maps
        were embedded no reader reads, or failed the verifier when not found
        (BACKLOG 2026-10-09). Added: 2026-10-17"""
        path = build_carrier_fbx(
            os.path.join(self.dir, "assembly.fbx"),
            {"lightmap_metadata": self.LIGHTMAPS},
            others={
                "MOD:ModA": {},
                "MOD:data_export": {"lightmap_metadata": self.MODULE_LIGHTMAPS},
            },
        )
        self.assertEqual(
            FbxMedia.dependencies(path),
            ["Room_Lightmap.exr", "Lamp_Lightmap.exr", "Room_Probe.exr"],
        )

    def test_a_module_record_names_only_the_nodes_the_file_carries(self):
        """Read alone, a module's record is that module's whole bake, whatever
        the export set: an assembly export that left ``MOD:ModB`` out shipped
        its map. Only an entry whose node the file carries names a file
        (exact, else by namespace-stripped leaf -- the appliers' rule: the
        module spells ``ModA``, the file carries ``MOD:ModA``), and a module
        record none of whose nodes ships names nothing, its probe included:
        no reader binds it. Added: 2026-10-17"""
        partial = build_carrier_fbx(
            os.path.join(self.dir, "partial.fbx"),
            {},
            others={
                "MOD:ModA": {},
                "MOD:data_export": {"lightmap_metadata": self.MODULE_LIGHTMAPS},
            },
        )
        self.assertEqual(
            FbxMedia.dependencies(partial), ["ModA_Module.exr", "Module_Probe.exr"]
        )
        shadows = {
            "version": 2,
            "planes": [{"name": "ModB_shadow", "texture": "ModB_shadow.png"}],
        }
        left_out = build_carrier_fbx(
            os.path.join(self.dir, "left_out.fbx"),
            {},
            others={
                "MOD:data_export": {
                    "lightmap_metadata": self.MODULE_LIGHTMAPS,
                    "shadow_metadata": shadows,
                }
            },
        )
        self.assertEqual(FbxMedia.dependencies(left_out), [])
        # The scene's own record is the export's to scope (its producer does):
        # read as written, a node the fixture lacks included.
        own = build_carrier_fbx(
            os.path.join(self.dir, "own.fbx"), {"shadow_metadata": shadows}
        )
        self.assertEqual(FbxMedia.dependencies(own), ["ModB_shadow.png"])

    def test_embed_adds_each_file_as_media_named_by_its_file_name(self):
        out = os.path.join(self.dir, "out.fbx")
        paths = [
            os.path.join(self.maps, n) for n in ("Room_Probe.exr", "Box_horizon.png")
        ]
        before = self._videos(self.src)
        report = FbxMedia.embed(self.src, out, files=paths)
        self.assertEqual(report["embedded"], ["Room_Probe.exr", "Box_horizon.png"])
        self.assertEqual(report["present"], [])
        self.assertEqual(
            report["bytes"], sum(len(self.files[n]) for n in report["embedded"])
        )
        added = {k: v for k, v in self._videos(out).items() if k not in before}
        self.assertEqual(len(added), 2)
        for fields in added.values():
            name = fields["RelativeFilename"].decode()
            # The bare file name: the deliverable names no folder.
            self.assertEqual(fields["Filename"].decode(), name)
            self.assertEqual(fields["Content"], self.files[name])
        # Media no material uses: nothing is wired to them, and every other
        # object and connection is the file's own.
        self.assertEqual(
            FbxFile.load(out).connections(), FbxFile.load(self.src).connections()
        )
        census = FbxFile.load(out).objects_census()
        self.assertEqual(
            census, {**FbxFile.load(self.src).objects_census(), "Video": 3}
        )
        total, kinds = self._counts(out)
        self.assertEqual((total, kinds["Video"]), (self._counts(self.src)[0] + 2, 3))
        # The source is left as it was.
        self.assertEqual(self._videos(self.src), before)

    def test_a_refused_promotion_strands_no_part_and_keeps_the_file(self):
        """The rewrite is written beside its target and promoted onto it.

        Production (2026-10-07): a sync client still reading a 218 MB FBX it
        had just seen change refused that replace, the embed was lost -- the
        file shipped without its lightmaps -- and its 228 MB ``.part`` stayed
        in the synced export folder, one per pass per run. A refusal must
        leave the target exactly as it was and take its ``.part`` with it.
        Added: 2026-10-07
        """
        from unittest import mock

        from pythontk.file_utils._file_utils import FileUtils

        with open(self.src, "rb") as fh:
            before = fh.read()
        refusal = PermissionError(13, "held by another process", self.src, 32)
        with mock.patch.object(FileUtils, "replace_file", side_effect=refusal):
            with self.assertRaises(PermissionError):
                FbxMedia.embed(
                    self.src, files=[os.path.join(self.maps, "Room_Probe.exr")]
                )
        strays = [n for n in os.listdir(self.dir) if n.endswith(".part")]
        self.assertEqual(strays, [], f"partial file left behind: {strays}")
        with open(self.src, "rb") as fh:
            self.assertEqual(fh.read(), before)

    def test_a_name_the_file_already_embeds_is_not_embedded_twice(self):
        """The silhouette is the plane material's own map: already inside."""
        out = os.path.join(self.dir, "out.fbx")
        elsewhere = os.path.join(self.dir, "BOX_SHADOW.png")
        with open(elsewhere, "wb") as fh:
            fh.write(b"another copy")
        report = FbxMedia.embed(self.src, out, files=[elsewhere])
        self.assertEqual(
            (report["embedded"], report["present"]), ([], ["BOX_SHADOW.png"])
        )
        self.assertFalse(os.path.exists(out), "nothing to add, nothing written")

    def test_embedding_again_changes_nothing(self):
        paths = [os.path.join(self.maps, n) for n in self.files]
        FbxMedia.embed(self.src, files=paths)  # in place
        with open(self.src, "rb") as fh:
            first = fh.read()
        report = FbxMedia.embed(self.src, files=paths)
        self.assertEqual(report["embedded"], [])
        self.assertEqual(sorted(report["present"]), sorted(self.files))
        with open(self.src, "rb") as fh:
            self.assertEqual(fh.read(), first)
        self.assertFalse([n for n in os.listdir(self.dir) if n.endswith(".part")])

    def test_a_video_type_is_declared_when_the_file_had_none(self):
        src = build_carrier_fbx(os.path.join(self.dir, "plain.fbx"), {})
        out = os.path.join(self.dir, "plain_out.fbx")
        FbxMedia.embed(src, out, files=[os.path.join(self.maps, "Room_Probe.exr")])
        total, kinds = self._counts(out)
        self.assertEqual((total, kinds.get("Video")), (self._counts(src)[0] + 1, 1))
        # And a file with no Definitions at all still takes the media.
        bare = build_carrier_fbx(
            os.path.join(self.dir, "nodefs.fbx"), {}, definitions=False
        )
        report = FbxMedia.embed(bare, files=[os.path.join(self.maps, "Room_Probe.exr")])
        self.assertEqual(report["embedded"], ["Room_Probe.exr"])
        self.assertEqual(
            [r["name"] for r in FbxMedia.embedded(bare)], ["Room_Probe.exr"]
        )

    def test_media_ids_are_new_and_the_same_for_the_same_name(self):
        """Stable across re-exports (a diff of two deliverables shows content,
        not renumbering), and never an id the file already uses."""
        path = os.path.join(self.maps, "Room_Probe.exr")
        a, b = os.path.join(self.dir, "a.fbx"), os.path.join(self.dir, "b.fbx")
        FbxMedia.embed(self.src, a, files=[path])
        FbxMedia.embed(self.src, b, files=[path])
        taken = {r["props"][0] for r in FbxFile.load(self.src).iter_objects()}
        new_a = set(self._videos(a)) - set(self._videos(self.src))
        self.assertEqual(new_a, set(self._videos(b)) - set(self._videos(self.src)))
        self.assertFalse(new_a & taken)

    def test_embed_dependencies_resolves_each_named_file_where_the_host_keeps_it(self):
        """Search folders in order first (the host's own answer, as the GLB
        appliers take it), the FBX's folder last; a name the materials already
        embed is left; a name found nowhere is reported and warned."""
        stale = os.path.join(self.dir, "Room_Lightmap.exr")  # beside the FBX
        with open(stale, "wb") as fh:
            fh.write(b"an older atlas of the same name")
        beside = os.path.join(self.dir, "Lamp_Lightmap.exr")
        os.replace(os.path.join(self.maps, "Lamp_Lightmap.exr"), beside)
        os.remove(os.path.join(self.maps, "shadow_atlas_horizon.png"))
        empty = os.path.join(self.dir, "empty")
        os.makedirs(empty)
        out = os.path.join(self.dir, "out.fbx")
        with self.assertLogs(
            "pythontk.file_utils.mesh_convert.fbx_media", "WARNING"
        ) as logs:
            report = FbxMedia.embed_dependencies(
                self.src, out, search_dirs=[empty, "", self.maps]
            )
        self.assertEqual(report["named"], self.NAMED)
        self.assertEqual(
            report["embedded"],
            [
                "Room_Lightmap.exr",
                "Lamp_Lightmap.exr",
                "Room_Probe.exr",
                "Box_horizon.png",
            ],
        )
        self.assertEqual(report["present"], ["Box_shadow.png"])
        self.assertEqual(report["missing"], ["shadow_atlas_horizon.png"])
        self.assertIn("shadow_atlas_horizon.png", "\n".join(logs.output))
        payloads = {
            f["RelativeFilename"].decode(): f["Content"]
            for f in self._videos(out).values()
        }
        self.assertEqual(payloads["Room_Lightmap.exr"], self.files["Room_Lightmap.exr"])
        self.assertEqual(payloads["Lamp_Lightmap.exr"], self.files["Lamp_Lightmap.exr"])
        self.assertEqual(payloads["Box_shadow.png"], self.silhouette)

    def test_a_name_the_folders_miss_is_found_under_the_walk_root(self):
        """The folders a host records can lag the disk; the root it keeps its
        files under is walked for what they miss, and the warning names that
        root alone -- the folder list it once printed buried the one name
        that mattered. Two same-named copies are never guessed between.
        Added: 2026-10-14"""
        root = os.path.join(self.dir, "root")
        deep = os.path.join(root, "a", "b")
        os.makedirs(deep)
        os.replace(
            os.path.join(self.maps, "Room_Probe.exr"),
            os.path.join(deep, "Room_Probe.exr"),
        )
        for twin in ("x", "y"):
            os.makedirs(os.path.join(root, twin))
            with open(os.path.join(root, twin, "shadow_atlas_horizon.png"), "wb") as fh:
                fh.write(b"twin")
        os.remove(os.path.join(self.maps, "shadow_atlas_horizon.png"))
        out = os.path.join(self.dir, "out.fbx")
        log = "pythontk.file_utils.mesh_convert.fbx_media"
        with self.assertLogs(log, "DEBUG") as logs:
            report = FbxMedia.embed_dependencies(
                self.src, out, search_dirs=[self.maps], walk_root=root
            )
        self.assertIn("Room_Probe.exr", report["embedded"])
        self.assertEqual(report["missing"], ["shadow_atlas_horizon.png"])
        (warning,) = [r for r in logs.records if r.levelname == "WARNING"]
        text = warning.getMessage()
        self.assertIn(root, text)
        self.assertIn("subfolders", text)
        self.assertIn("2 same-named", text)
        self.assertNotIn(self.maps, text, "the folder list is the full log's")
        debug = "\n".join(
            r.getMessage() for r in logs.records if r.levelname == "DEBUG"
        )
        self.assertIn(self.maps, debug)

    def test_a_dependency_goes_in_under_the_records_spelling(self):
        """The name is the join key: Unity looks the extracted copy up by the
        record's name, and a case-insensitive disk can find the file under
        another spelling."""
        if not os.path.exists(os.path.join(self.maps, "ROOM_PROBE.EXR")):
            self.skipTest("case-sensitive file system")
        os.replace(
            os.path.join(self.maps, "Room_Probe.exr"),
            os.path.join(self.maps, "ROOM_PROBE.EXR"),
        )
        out = os.path.join(self.dir, "out.fbx")
        report = FbxMedia.embed_dependencies(self.src, out, search_dirs=[self.maps])
        self.assertIn("Room_Probe.exr", report["embedded"])
        self.assertIn("Room_Probe.exr", [r["name"] for r in FbxMedia.embedded(out)])

    def test_embed_dependencies_carries_the_extra_files_a_host_names(self):
        """A record that names its files by something other than a file name
        (the audio events) hands the paths over itself."""
        sound = os.path.join(self.dir, "Footstep.wav")
        with open(sound, "wb") as fh:
            fh.write(b"RIFF....WAVE")
        out = os.path.join(self.dir, "out.fbx")
        report = FbxMedia.embed_dependencies(
            self.src, out, search_dirs=[self.maps], files=[sound]
        )
        self.assertIn("Footstep.wav", report["embedded"])
        self.assertNotIn("Footstep.wav", report["named"])
        self.assertIn("Footstep.wav", [r["name"] for r in FbxMedia.embedded(out)])

    def test_a_host_file_can_go_in_under_its_join_name(self):
        """An audio event names its track, not the track's file: the host
        hands ``(name, path)`` and the file rides under the name the consumer
        matches it by."""
        take = os.path.join(self.dir, "footstep_take3.wav")
        with open(take, "wb") as fh:
            fh.write(b"RIFF....WAVE take three")
        out = os.path.join(self.dir, "out.fbx")
        report = FbxMedia.embed_dependencies(
            self.src, out, search_dirs=[self.maps], files=[("footstep.wav", take)]
        )
        self.assertIn("footstep.wav", report["embedded"])
        payloads = {
            f["RelativeFilename"].decode(): f["Content"]
            for f in self._videos(out).values()
        }
        self.assertEqual(payloads["footstep.wav"], b"RIFF....WAVE take three")

    def test_audio_events_names_the_tracks_the_manifest_plays(self):
        """An audio event names its TRACK, not a file: a host resolves each
        name to its file and hands it over (mayatk / blendertk
        ``FbxUtils.embed_dependencies``). Each name once; a blank or ``None``
        name names nothing."""
        audio = {
            "version": 2,
            "events": [
                {"clip": "Intro", "frame": 4, "name": "footstep"},
                {"clip": "Fight", "frame": 9, "name": "punch"},
                {"clip": "Fight", "frame": 20, "name": "footstep"},
                {"clip": "", "frame": 0, "name": "None"},
                {"clip": "", "frame": 0, "name": " "},
            ],
        }
        path = build_carrier_fbx(
            os.path.join(self.dir, "audio.fbx"), {"audio_manifest": audio}
        )
        self.assertEqual(FbxMedia.audio_events(path), ["footstep", "punch"])
        self.assertEqual(FbxMedia.audio_events(self.src), [])

    def test_audio_events_reads_the_carrier_the_readers_read(self):
        """An assembly ships a stray ``data_export1`` (and a module's
        ``NS:data_export``) beside the scene's own, each holding the events as
        last exported alone (measured on an assembly, 2026-10-13). The readers
        take ONE carrier per channel -- the scene's own, else the first met
        (the GLB applier; unitytk ``CarrierImport.Collect``) -- and the scene's
        own record is re-derived from the whole merged scene, so the stray
        copies' events are stale: their clips are not shipped."""
        own = {"version": 2, "events": [{"clip": "", "frame": 0, "name": "footstep"}]}
        stray = {"version": 2, "events": [{"clip": "", "frame": 3, "name": "door"}]}
        assembly = build_carrier_fbx(
            os.path.join(self.dir, "assembly.fbx"),
            {"audio_manifest": own},
            others={"data_export1": {"audio_manifest": stray}},
        )
        self.assertEqual(FbxMedia.audio_events(assembly), ["footstep"])
        modules = build_carrier_fbx(
            os.path.join(self.dir, "modules.fbx"),
            {},
            others={
                "Door:data_export": {"audio_manifest": stray},
                "Lift:data_export": {"audio_manifest": own},
            },
        )
        self.assertEqual(
            FbxMedia.audio_events(modules),
            ["door"],
            "no record on the scene's own carrier: the first met",
        )

    def test_embed_dependencies_writes_nothing_without_records(self):
        bare = build_carrier_fbx(os.path.join(self.dir, "bare.fbx"), {})
        out = os.path.join(self.dir, "out.fbx")
        report = FbxMedia.embed_dependencies(bare, out, search_dirs=[self.maps])
        self.assertEqual(
            report,
            {"named": [], "embedded": [], "present": [], "missing": [], "bytes": 0},
        )
        self.assertFalse(os.path.exists(out))

    def test_image_passes_leave_dependency_media_alone(self):
        """``downsize`` / ``expand_grayscale`` exist for the converter, which
        reads only what a material binds: a horizon map is DATA (resized, its
        bits are lost) and an EXR is no image Pillow reads (a warning per map)."""
        from PIL import Image

        src = build_fbx(
            os.path.join(self.dir, "mat.fbx"), {"wall.png": png_bytes((512, 256))}
        )
        horizon = os.path.join(self.dir, "Big_horizon.png")
        with open(horizon, "wb") as fh:
            fh.write(png_bytes((512, 512), "RGBA"))
        gray = os.path.join(self.dir, "Mask_horizon.png")
        with open(gray, "wb") as fh:
            fh.write(encoded(Image.new("L", (8, 8), 40)))
        exr = os.path.join(self.maps, "Room_Lightmap.exr")
        FbxMedia.embed(src, files=[horizon, gray, exr])
        with open(horizon, "rb") as fh:
            original = fh.read()

        out = os.path.join(self.dir, "small.fbx")
        with self.assertNoLogs("pythontk.file_utils.mesh_convert.fbx_media", "WARNING"):
            report = FbxMedia.downsize(src, out, max_size=128)
        self.assertEqual((report["images"], report["resized"]), (1, 1))
        rows = {r["name"]: r for r in FbxMedia.embedded(out)}
        self.assertEqual(rows["wall.png"]["size"], (128, 64))
        self.assertEqual(rows["Big_horizon.png"]["bytes"], len(original))
        report = FbxMedia.expand_grayscale(src, os.path.join(self.dir, "rgb.fbx"))
        self.assertEqual((report["images"], report["expanded"]), (1, 0))

    def test_dependency_media_survive_the_other_rewrites(self):
        src = build_animated_fbx(
            os.path.join(self.dir, "anim.fbx"),
            ["Take 001", "Shot_A"],
            media={"w.png": b"x"},
        )
        FbxMedia.embed(src, files=[os.path.join(self.maps, "Room_Probe.exr")])
        out = os.path.join(self.dir, "lean.fbx")
        self.assertEqual(
            FbxMedia.drop_takes(src, out, names=["Shot_A"])["takes"], ["Shot_A"]
        )
        self.assertEqual(
            [r["name"] for r in FbxMedia.embedded(out)], ["w.png", "Room_Probe.exr"]
        )

    def test_named_files_walks_decoded_payloads_by_record(self):
        """The one walk the FBX step, the GLB build and the verifier share:
        a reader that already holds the channels (the verifier's parsed FBX)
        asks it without a second parse."""
        self.assertEqual(
            FbxMedia.named_files(
                {
                    "shadow_metadata": [self.SHADOWS],
                    "lightmap_metadata": [self.LIGHTMAPS, {"objects": "not a list"}],
                    "unrelated": [{"map": "ignored.exr"}],
                }
            ),
            self.NAMED,
        )
        self.assertEqual(FbxMedia.named_files({}), [])

    def test_extract_dependencies_writes_the_embedded_named_files(self):
        """The reverse of the embed: what a reader of a DELIVERED FBX needs on
        disk (the GLB build's appliers join names against folders)."""
        FbxMedia.embed_dependencies(self.src, search_dirs=[self.maps])
        out = os.path.join(self.dir, "out")
        os.makedirs(out)
        written = FbxMedia.extract_dependencies(self.src, out)
        self.assertEqual(sorted(written), sorted(self.NAMED))
        for name, path in written.items():
            self.assertEqual(os.path.dirname(path), out)
            with open(path, "rb") as fh:
                data = fh.read()
            self.assertEqual(data, self.files.get(name, self.silhouette), name)
        # Only what the records name: an FBX naming nothing writes nothing.
        bare = build_fbx(os.path.join(self.dir, "bare.fbx"), {"wall.png": b"x"})
        self.assertEqual(FbxMedia.extract_dependencies(bare, out), {})

    def test_unreadable_input_raises(self):
        junk = os.path.join(self.dir, "junk.fbx")
        with open(junk, "wb") as fh:
            fh.write(b"; FBX 7.7.0 project file (ASCII)\n" * 20)
        exr = os.path.join(self.maps, "Room_Probe.exr")
        with self.assertRaises(ValueError):
            FbxMedia.embed(junk, files=[exr])
        with self.assertRaises(ValueError):
            FbxMedia.embed_dependencies(junk, search_dirs=[self.maps])
        with self.assertRaises(OSError):
            FbxMedia.embed(self.src, files=[os.path.join(self.dir, "nowhere.exr")])

    def test_a_truncated_fbx_is_a_value_error(self):
        """A binary FBX cut short (a copy that never finished) has the magic
        but not the body: reading it raised ``struct.error``, past every
        caller that leaves an FBX it cannot re-emit as it was."""
        from pythontk.file_utils.mesh_convert.fbx_file import FBX_MAGIC

        bodies = {
            "no_version.fbx": FBX_MAGIC + b"\x01",
            "cut_header.fbx": FBX_MAGIC + struct.pack("<I", 7700) + b"\x10" * 20,
            "cut_record.fbx": FBX_MAGIC
            + struct.pack("<I", 7400)
            + struct.pack("<III", 4000, 0, 0)
            + b"\x07Objects",
        }
        for name, body in bodies.items():
            path = os.path.join(self.dir, name)
            with open(path, "wb") as fh:
                fh.write(body)
            with self.subTest(name), self.assertRaises(ValueError):
                FbxMedia.embed_dependencies(path, files=[self.src])

    def test_a_child_ending_past_its_parent_is_a_value_error(self):
        """A child record whose end lies past its parent's, still inside the
        file, parsed: its tail swallowed its siblings and the next top-level
        record, and the read went on from the parent's end as if whole -- so
        an embed wrote those bytes twice into a file that read as fine.
        Added: 2026-10-10
        """
        with open(self.src, "rb") as fh:
            data = bytearray(fh.read())
        objects = data.index(b"\x07Objects") - 24  # 7700: 24-byte headers
        child = data.index(b"\x05Model", objects) - 24
        parent_end = struct.unpack_from("<Q", data, objects)[0]
        struct.pack_into("<Q", data, child, parent_end + 8)
        corrupt = os.path.join(self.dir, "overrun.fbx")
        with open(corrupt, "wb") as fh:
            fh.write(bytes(data))
        with self.assertRaises(ValueError):
            FbxMedia.dependencies(corrupt)
        with self.assertRaises(ValueError):
            FbxMedia.embed_dependencies(corrupt, search_dirs=[self.maps])


class RecordTestCase(unittest.TestCase):
    """``record`` / ``set_record``: a scene record read and replaced in the file.

    The post-write passes that learn something the scene could not say (the
    highlight copies a GLB build draws) write it into the deliverable's own
    record, in the copy a reader applies.
    """

    OWN = {"version": 1, "objects": [{"name": "Crate", "map": "Room_Lightmap.exr"}]}
    MODULE = {"version": 1, "objects": [{"name": "Shelf", "map": "Mod_Lightmap.exr"}]}

    def setUp(self):
        self.temp = TempArtifacts("test_fbx_media_record", policy="scoped")
        self.addCleanup(self.temp.cleanup)
        self.dir = self.temp.dir_path()

    def _fbx(self, own=True, module=True, name="deliverable.fbx"):
        return build_carrier_fbx(
            os.path.join(self.dir, name),
            {"lightmap_metadata": self.OWN, "shadow_metadata": {"planes": []}} if own else {},
            others={"MOD:data_export": {"lightmap_metadata": self.MODULE}} if module else None,
        )

    def test_the_scenes_own_record_is_read_over_a_modules(self):
        self.assertEqual(FbxMedia.record(self._fbx(), "lightmap_metadata"), self.OWN)
        self.assertEqual(
            FbxMedia.record(self._fbx(own=False), "lightmap_metadata"), self.MODULE
        )
        self.assertIsNone(FbxMedia.record(self._fbx(module=False), "audio_manifest"))

    def test_set_record_replaces_the_applied_copy_and_embeds_in_one_write(self):
        src = self._fbx()
        hdr = os.path.join(self.dir, "Room_Probe_Highlight1.hdr")
        with open(hdr, "wb") as fh:
            fh.write(b"#?RADIANCE\nhighlight bytes")
        payload = dict(self.OWN, probe={"map": "Room_Probe.exr", "highlights": [
            {"map": "Room_Probe_Highlight1.hdr", "objects": [{"name": "Crate"}]}]})
        report = FbxMedia.set_record(src, "lightmap_metadata", payload, files=[hdr])
        self.assertTrue(report["replaced"])
        self.assertEqual(report["embedded"], ["Room_Probe_Highlight1.hdr"])
        self.assertEqual(FbxMedia.record(src, "lightmap_metadata"), payload)
        loaded = FbxFile.load(src)
        # The other channels and the module's copy are as they were.
        carried = dict(
            (owner, json.loads(value))
            for owner, value in loaded.user_properties("lightmap_metadata", named=True)
        )
        self.assertEqual(carried["MOD:data_export"], self.MODULE)
        self.assertEqual(
            [json.loads(v) for v in loaded.user_properties("shadow_metadata")],
            [{"planes": []}],
        )
        # The copy the record now names rides in the file.
        self.assertIn("Room_Probe_Highlight1.hdr", FbxMedia.dependencies(src))
        self.assertIn(
            "Room_Probe_Highlight1.hdr", [e["name"] for e in FbxMedia.embedded(src)]
        )

    def test_a_file_carrying_no_such_record_is_left_alone(self):
        src = self._fbx(own=False, module=False)
        with open(src, "rb") as fh:
            before = fh.read()
        report = FbxMedia.set_record(src, "lightmap_metadata", self.OWN)
        self.assertFalse(report["replaced"])
        with open(src, "rb") as fh:
            self.assertEqual(fh.read(), before)


def _definition_counts(path):
    """``{object type: count}`` of the file's ``Definitions``, ``"*"`` the total."""
    counts = {}
    for record in FbxFile.load(path).section("Definitions")["children"]:
        if record["name"] == "Count":
            counts["*"] = record["props"][0]
        elif record["name"] == "ObjectType":
            for child in record["children"]:
                if child["name"] == "Count":
                    counts[record["props"][0].decode()] = child["props"][0]
    return counts


class SplitPivotInstancesTestCase(unittest.TestCase):
    """``split_pivot_instances``: an instanced mesh lands where each instance is.

    Unity's importer resets a node's pivots through its mesh, so a geometry
    shared by instances with different pivots is moved by ONE instance's:
    measured on a production office (2026-10-19), ``WALL_D`` -- its rotate
    pivot 131.44 cm below its row's -- imported exactly 1.3144 m low.
    """

    HIGH = {"RotationPivot": (-197.1654, 332.5522, -591.496)}
    LOW = {"RotationPivot": (-197.1654, 201.1086, -591.496)}

    def setUp(self):
        self.temp = TempArtifacts("test_fbx_media_split", policy="scoped")
        self.addCleanup(self.temp.cleanup)
        self.path = os.path.join(self.temp.dir_path(), "office.fbx")

    def _geometry_of(self, path):
        """``{model name: geometry id}`` and ``{geometry id: its Vertices}``."""
        loaded = FbxFile.load(path)
        names, vertices = {}, {}
        for record in loaded.iter_objects():
            if record["name"] == "Model":
                names[record["props"][0]] = FbxFile._display_name(record["props"][1])
            elif record["name"] == "Geometry":
                vertices[record["props"][0]] = next(
                    c["props"][0] for c in record["children"] if c["name"] == "Vertices"
                )
        owner = {
            names[parent]: child
            for _kind, child, parent, *_ in loaded.connections()
            if child in vertices and parent in names
        }
        return owner, vertices

    def test_instances_whose_pivots_differ_get_a_geometry_each(self):
        build_instanced_fbx(self.path, [self.HIGH, self.HIGH, self.LOW, {}])
        report = FbxMedia.split_pivot_instances(self.path)
        self.assertEqual(report["models"], ["WALL_C", "WALL_D"])
        self.assertEqual(report["geometries"], 2)
        owner, vertices = self._geometry_of(self.path)
        self.assertEqual(owner["WALL_A"], owner["WALL_B"], "agreeing instances stay one mesh")
        self.assertEqual(len({owner["WALL_A"], owner["WALL_C"], owner["WALL_D"]}), 3)
        self.assertEqual(len({str(v) for v in vertices.values()}), 1, "each a copy of the mesh")
        counts = _definition_counts(self.path)
        self.assertEqual(counts["Geometry"], 3)
        self.assertEqual(counts["*"], 7)  # 3 geometries + 4 models

    def test_instances_that_differ_only_in_their_offsets_stay_one_mesh(self):
        """Measured on the same office: instances that differ only in
        ``RotationOffset`` (``WALL_X`` against ``WALL_H``) landed right -- an
        offset is a translation the node keeps, not one the mesh is moved by.
        Float noise in a pivot is no pivot of its own."""
        build_instanced_fbx(self.path, [
            dict(self.HIGH, RotationOffset=(394.3308, 0.0, 1182.9921)),
            {"RotationPivot": (-197.1653, 332.5522, -591.496)},
        ])
        with open(self.path, "rb") as fh:
            before = fh.read()
        self.assertEqual(FbxMedia.split_pivot_instances(self.path)["geometries"], 0)
        with open(self.path, "rb") as fh:
            self.assertEqual(fh.read(), before, "not written")

    def test_a_deformed_geometry_stays_shared_and_is_said(self):
        """A copy would deform nothing: its skin binds the original."""
        build_instanced_fbx(self.path, [self.HIGH, self.LOW], deformed=True)
        report = FbxMedia.split_pivot_instances(self.path)
        self.assertEqual(report["geometries"], 0)
        self.assertEqual(report["deformed"], ["wallShape"])


class DropHostLightingTestCase(unittest.TestCase):
    """``drop_host_lighting_media``: no host's own lighting rides a deliverable.

    Maya's StingrayPBS wires Autodesk's environment cubes and BRDF table into
    every material, and an embedded-media export carries their files: a
    production office's FBX shipped ``specular_cube.png``, ``diffuse_cube.png``
    and ``ibl_brdf_lut.png`` three times over, and Unity extracted them beside
    the model, where nothing reads them (2026-10-19).
    """

    def setUp(self):
        self.temp = TempArtifacts("test_fbx_media_host_lighting", policy="scoped")
        self.addCleanup(self.temp.cleanup)
        self.path = build_stingray_fbx(
            os.path.join(self.temp.dir_path(), "office.fbx"),
            {
                "color": ("wall_color.png", png_bytes((4, 4))),
                "TEX_brdf_lut": ("ibl_brdf_lut.png", png_bytes((8, 8))),
                "TEX_global_specular_cube": ("specular_cube.png", png_bytes((6, 6))),
            },
        )

    def test_the_host_lighting_textures_go_with_their_media(self):
        report = FbxMedia.drop_host_lighting_media(self.path)
        self.assertEqual(report["textures"], 2)
        self.assertEqual(report["media"], ["ibl_brdf_lut.png", "specular_cube.png"])
        self.assertEqual([e["name"] for e in FbxMedia.embedded(self.path)], ["wall_color.png"])
        counts = _definition_counts(self.path)
        self.assertEqual((counts["Texture"], counts["Video"]), (1, 1))
        loaded = FbxFile.load(self.path)
        ids = {record["props"][0] for record in loaded.iter_objects()}
        dangling = [c for c in loaded.connections() if c[1] not in ids or (c[2] and c[2] not in ids)]
        self.assertEqual(dangling, [])

    def test_a_file_another_video_or_a_record_names_keeps_its_bytes(self):
        """The SDK embeds a file's bytes once, in the first ``Video`` naming
        it: a user's texture on the same file as a host-lighting slot carries
        none of its own, and a probe plugged into a slot may ride only there.
        Each such file stays, wired to nothing; only the textures go."""
        path = build_stingray_fbx(
            os.path.join(self.temp.dir_path(), "shared.fbx"),
            {
                "TEX_global_specular_cube": ("studio.png", png_bytes((6, 6))),
                "TEX_global_diffuse_cube": ("Room_Probe.png", png_bytes((5, 5))),
                "TEX_brdf_lut": ("ibl_brdf_lut.png", png_bytes((8, 8))),
                "color": ("studio.png", b""),  # the SDK's later Video of a file
            },
            records={"lightmap_metadata": {"version": 1, "probe": {"map": "Room_Probe.png"}}},
        )
        report = FbxMedia.drop_host_lighting_media(path)
        self.assertEqual(report["textures"], 3)
        self.assertEqual(report["media"], ["ibl_brdf_lut.png"])
        held = {}
        for entry in FbxMedia.embedded(path):
            held[entry["name"]] = max(held.get(entry["name"], 0), entry["bytes"])
        self.assertGreater(held.get("studio.png", 0), 0, "the user's texture keeps its bytes")
        self.assertGreater(held.get("Room_Probe.png", 0), 0, "the probe the record names")
        self.assertNotIn("ibl_brdf_lut.png", held)
        counts = _definition_counts(path)
        self.assertEqual((counts["Texture"], counts["Video"]), (1, 3))
        loaded = FbxFile.load(path)
        ids = {record["props"][0] for record in loaded.iter_objects()}
        dangling = [c for c in loaded.connections() if c[1] not in ids or (c[2] and c[2] not in ids)]
        self.assertEqual(dangling, [])

    def test_a_file_with_none_is_not_written(self):
        FbxMedia.drop_host_lighting_media(self.path)
        with open(self.path, "rb") as fh:
            before = fh.read()
        self.assertEqual(FbxMedia.drop_host_lighting_media(self.path)["textures"], 0)
        with open(self.path, "rb") as fh:
            self.assertEqual(fh.read(), before)


if __name__ == "__main__":
    unittest.main()
