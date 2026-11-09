#!/usr/bin/python
# coding=utf-8
"""The ``lightmaps`` record's declared shape, held to its producers.

:class:`LightmapRecord` (the record on a deliverable's ``data_export``
carrier, and, from version 3, a GLB's ``extras.lightmaps`` as it stands) is a
typed :class:`~pythontk.SchemaSpec` declaration: the types another language
reads a file through are generated from it. :class:`TestTheGlbCarriesTheRecordShape`
holds the record a GLB carries as it stands to the same shape. Each must admit exactly what every
producer writes today -- pythontk's own passes run here for real, the DCC
producers' entries are copied as they spell them -- and refuse a missing field
or a wrong type.
"""

import copy
import io
import json
import os
import struct
import unittest

from pythontk import LightmapRecord, MeshConvert, SceneRecords, TempArtifacts
from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia

# ------------------------------------------------------- what producers write
#: An entry as a host whose objects sit in a DAG of transforms spells one:
#: the transform's leaf name, namespace kept; the marker's map and rect; the
#: DAG path's names, root first (version 3: no UV set -- the contract reads
#: the second).
DAG_ENTRY = {
    "name": "PROPS:crate",
    "map": "room_Lightmap.exr",
    "scaleOffset": [0.5, 0.5, 0.5, 0.0],
    "hierarchy": ["ROOM", "PROPS:props", "PROPS:crate"],
}
#: The probe that host hands the builder: in its working unit (centimetres),
#: its box the ``[min, max]`` points the host keeps and its own ``unit_scale``
#: beside it.
DAG_PROBE = {
    "map": "room_Probe.exr",
    "position": [0.0, 150.0, 200.0],
    "box": [[-400.0, 0.0, -400.0], [400.0, 400.0, 400.0]],
    "unit_scale": 0.01,
}
#: That host's lights, one of each kind, as its producer hands them to the
#: builder, the name -- the scene's bookkeeping -- left out: every
#: light its type, position, direction, colour, intensity and unit; a spot's
#: angle, penumbra and radius, a point's radius, a rect's right, width and
#: height.
DAG_LIGHTS = [
    {
        "type": "point",
        "position": [0.0, 160.0, 30.0],
        "direction": [0.0, 0.0, -1.0],
        "color": [1.0, 0.9, 0.8],
        "intensity": 2.0,
        "unit_scale": 0.01,
        "radius": 2.5,
    },
    {
        "type": "spot",
        "position": [100.0, 300.0, 0.0],
        "direction": [0.0, -1.0, 0.0],
        "color": [1.0, 1.0, 1.0],
        "intensity": 40.0,
        "unit_scale": 0.01,
        "angle": 0.6,
        "penumbra": 0.25,
        "radius": 1.0,
    },
    {
        "type": "rect",
        "position": [0.0, 399.0, 0.0],
        "direction": [0.0, -1.0, 0.0],
        "color": [0.9, 0.95, 1.0],
        "intensity": 3.0,
        "unit_scale": 0.01,
        "right": [1.0, 0.0, 0.0],
        "width": 120.0,
        "height": 60.0,
    },
]
#: An entry as a host whose object names are unique in a file spells one:
#: the object's name, its marker's map and rect, its hierarchy (the names
#: root first, its own last).
OBJECT_ENTRY = {
    "name": "crate",
    "map": "room_Lightmap.exr",
    "scaleOffset": [1.0, 1.0, 0.0, 0.0],
    "hierarchy": ["crate"],
}
#: The probe that host hands the builder, as stored in its own units --
#: an open scene, its box ``None``.
OBJECT_PROBE = {
    "map": "room_Probe.exr",
    "position": [0.0, 1.5, 2.0],
    "box": None,
    "unit_scale": 1.0,
}
#: A highlight copy as ``MeshConvert.embed_fbx_highlight_probes`` lists one on
#: the FBX record's probe.
FBX_HIGHLIGHT = {
    "map": "room_Probe_Highlight1.hdr",
    "objects": [{"name": "TABLE", "hierarchy": ["ROOM", "TABLE"]}],
}
#: Names a nested C# record type must not take: sync_scene_records drops the
#: root's family prefix (``LightmapEntry`` is ``LightmapRecord.Shape.Entry``),
#: so these would shadow the engine's or System's types, or the generated
#: file's own.
_CS_TAKEN = {"Object", "Type", "String", "Light", "Payload", "Shape", "Wire"}

#: A mutation that removes the key (:func:`_mutated`).
_GONE = object()


def _verdict(shape, payload):
    """``(errors, warnings)`` of *payload* against *shape*."""
    result = shape.validate(payload)
    return result.errors, result.warnings


def _mutated(payload, path, key, value):
    """A copy of *payload* with *key*, under *path*, set to *value* (or gone)."""
    out = copy.deepcopy(payload)
    holder = out
    for step in path:
        holder = holder[step]
    if value is _GONE:
        del holder[key]
    else:
        holder[key] = value
    return out


def _scratch(case):
    """A folder of *case*'s own, removed after it."""
    artifacts = TempArtifacts("lightmap_record_shape", policy="scoped")
    case.addCleanup(artifacts.cleanup)
    return artifacts.dir_path()


def _require_cv2():
    """Skip without ``cv2``: the probe's highlight copies are Radiance bytes
    cv2 writes, an optional dependency CI does not install."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("cv2 unavailable: the highlight copies cannot be encoded")


# ------------------------------------------------- a room the real pass reads
class _Codec:
    """A host's HDR codec (``hdr_codec``), as one without cv2 hands it: a lightmap
    as a PNG and the scalar restoring its range, the probe as Radiance bytes
    whose header states its size -- the size its copies are drawn at."""

    SCALAR = 2.5
    WIDTH, HEIGHT = 64, 32

    @classmethod
    def encode_hdr_for_web(cls, path, percentile=None):
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (4, 4), (128, 128, 128)).save(buf, format="PNG")
        return buf.getvalue(), cls.SCALAR

    @classmethod
    def encode_hdr_radiance(cls, path):
        header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {cls.HEIGHT} +X {cls.WIDTH}\n"
        return header.encode("ascii") + bytes(cls.WIDTH * cls.HEIGHT * 4)


#: The room a bake lit, quad by quad -- ``(node, material, corners)``, each
#: facing +Y or +Z: a floor on the probe box's bottom face (its shell), a table
#: top inside the box, and a lamp the bake left out.
ROOM = (
    ("FLOOR", "floorMat", ((-4, 0, 4), (4, 0, 4), (4, 0, -4), (-4, 0, -4))),
    ("TABLE", "tableMat", ((-0.8, 0.9, 0.4), (0.8, 0.9, 0.4), (0.8, 0.9, -0.4), (-0.8, 0.9, -0.4))),
    ("LAMP", "lampMat", ((0.15, 1.05, 0.0), (0.45, 1.05, 0.0), (0.45, 1.35, 0.0), (0.15, 1.35, 0.0))),
)
#: A point light over the table, as a DAG host's producer hands it (its direction
#: the builder drops).
HANGING = {
    "type": "point",
    "position": [0.0, 1.6, 0.3],
    "direction": [0.0, -1.0, 0.0],
    "color": [1.0, 1.0, 1.0],
    "intensity": 2.0,
    "unit_scale": 1.0,
    "radius": 0.0,
}


def _write_glb(path, gltf, blob=b""):
    """*gltf*, and *blob* as its BIN chunk, written as a GLB at *path*."""
    text = json.dumps(gltf).encode("utf-8")
    text += b" " * (-len(text) % 4)
    tail = b""
    if blob:
        blob += b"\x00" * (-len(blob) % 4)
        tail = struct.pack("<I4s", len(blob), b"BIN\x00") + blob
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(text) + len(tail)))
        fh.write(struct.pack("<I4s", len(text), b"JSON") + text)
        fh.write(tail)
    return path


def _room_glb(path, record):
    """:data:`ROOM` as a GLB built from its FBX: each quad a mesh with both UV
    sets (the second the bake's), *record* on the ``data_export`` carrier as
    FBX2glTF leaves a user property."""
    materials = [material for _node, material, _corners in ROOM]
    blob, views, accessors, nodes, meshes = bytearray(), [], [], [], []

    def add(values, kind, fmt, component):
        data = struct.pack(f"<{len(values)}{fmt}", *values)
        data += b"\x00" * (-len(data) % 4)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob.extend(data)
        size = {"VEC3": 3, "VEC2": 2, "SCALAR": 1}[kind]
        accessors.append(
            {
                "bufferView": len(views) - 1,
                "componentType": component,
                "count": len(values) // size,
                "type": kind,
            }
        )
        return len(accessors) - 1

    for name, material, corners in ROOM:
        a, b, c = corners[:3]
        u = [b[i] - a[i] for i in range(3)]
        v = [c[i] - a[i] for i in range(3)]
        n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        length = sum(x * x for x in n) ** 0.5
        positions = [float(x) for corner in corners for x in corner]
        position = add(positions, "VEC3", "f", 5126)
        accessors[position]["min"] = [min(positions[i::3]) for i in range(3)]
        accessors[position]["max"] = [max(positions[i::3]) for i in range(3)]
        uv = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        attributes = {
            "POSITION": position,
            "NORMAL": add([x / length for x in n] * 4, "VEC3", "f", 5126),
            "TEXCOORD_0": add(uv, "VEC2", "f", 5126),
            "TEXCOORD_1": add(uv, "VEC2", "f", 5126),
        }
        primitive = {
            "attributes": attributes,
            "indices": add([0, 1, 2, 0, 2, 3], "SCALAR", "H", 5123),
            "material": materials.index(material),
        }
        meshes.append({"primitives": [primitive]})
        nodes.append({"name": name, "mesh": len(meshes) - 1})
    carried = {"type": "eFbxString", "value": json.dumps(record)}
    nodes.append(
        {
            "name": "data_export",
            "extras": {"fromFBX": {"userProperties": {"lightmaps": carried}}},
        }
    )
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": [{"name": m} for m in materials],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    return _write_glb(path, gltf, bytes(blob))


def _lit_room(folder, lights=(), box=True, probe_pixels=False):
    """The room through ``MeshConvert.apply_glb_lightmaps`` and the record it
    carried: the floor and table baked as a DAG host names them, the probe at
    head height in the room's box (or, without *box*, an open scene's), and
    *lights*. The files are stand-ins the host codec reads none of -- but
    with *probe_pixels* the probe is a real EXR, which drawing the lights
    into it reads (cv2). Returns ``(glb, record)``."""
    for name in ("room_Lightmap.exr", "room_Probe.exr"):
        with open(os.path.join(folder, name), "wb") as fh:
            fh.write(b"\x00")  # the codec reads none of it
    if probe_pixels:
        import cv2
        import numpy as np

        cv2.imwrite(
            os.path.join(folder, "room_Probe.exr"),
            np.full((32, 64, 3), 0.2, np.float32),
        )
    record = LightmapRecord.payload(
        [
            dict(DAG_ENTRY, name=n, hierarchy=[n], scaleOffset=[1.0, 1.0, 0.0, 0.0])
            for n in ("FLOOR", "TABLE")
        ],
        probe={
            "map": "room_Probe.exr",
            "position": [0.0, 1.8, 2.0],
            "box": [[-4.0, 0.0, -4.0], [4.0, 4.0, 4.0]] if box else None,
            "unit_scale": 1.0,
        },
        lights=[dict(light) for light in lights],
        unit_scale=1.0,
    )
    glb = _room_glb(os.path.join(folder, "room.glb"), record)
    bound = MeshConvert.apply_glb_lightmaps(glb, search_dirs=[folder], hdr_codec=_Codec)
    if not bound:
        raise AssertionError("the room's lightmaps did not bind")
    return glb, record


def _glb_block(glb):
    """The ``lightmaps`` record *glb* carries, as a reader reads it."""
    with MeshConvert.open_glb(glb) as edit:
        return copy.deepcopy(SceneRecords.glb_block(edit.gltf, SceneRecords.LIGHTMAPS))


# ------------------------------------------------------------------ the shapes
class TestLightmapRecordShape(unittest.TestCase):
    """:class:`LightmapRecord`: the record every host publishes through
    :meth:`LightmapRecord.payload`, and the copies the FBX highlight pass adds."""

    def setUp(self):
        self.tmp = _scratch(self)

    @staticmethod
    def _full():
        """A record with every part: objects, a boxed probe with a highlight
        copy, and a light of each kind."""
        return LightmapRecord.payload(
            [DAG_ENTRY],
            probe=dict(DAG_PROBE, highlights=[FBX_HIGHLIGHT]),
            lights=DAG_LIGHTS,
            unit_scale=0.01,
        )

    def test_every_branch_of_the_builder_is_the_shape(self):
        """With and without a probe and lights: objects alone (no unit to
        state), a probe in a room and in the open (its box ``null``), lights
        alone, and everything with a highlight copy."""
        cases = {
            "objects alone": LightmapRecord.payload([DAG_ENTRY]),
            "a probe in a room": LightmapRecord.payload(
                [DAG_ENTRY], probe=DAG_PROBE, unit_scale=0.01
            ),
            "a probe in the open": LightmapRecord.payload(
                [DAG_ENTRY], probe=dict(DAG_PROBE, box=None), unit_scale=0.01
            ),
            "lights alone": LightmapRecord.payload(
                [DAG_ENTRY], lights=DAG_LIGHTS, unit_scale=0.01
            ),
            "everything": self._full(),
        }
        self.assertNotIn("unit_scale", cases["objects alone"])
        self.assertIsNone(cases["a probe in the open"]["probe"]["box"])
        for label, payload in cases.items():
            with self.subTest(label):
                self.assertEqual(_verdict(LightmapRecord, payload), ([], []))

    def test_a_dag_hosts_record_is_the_shape(self):
        """A DAG host's producer as it assembles the record: its entries, its
        probe in the working unit and its lights, through the builder."""
        payload = LightmapRecord.payload(
            [DAG_ENTRY, dict(DAG_ENTRY, name="TABLE", hierarchy=["ROOM", "TABLE"])],
            probe=DAG_PROBE,
            lights=DAG_LIGHTS,
            unit_scale=0.01,
        )
        self.assertEqual(_verdict(LightmapRecord, payload), ([], []))
        bulb, spot, panel = payload["lights"]
        self.assertNotIn("direction", bulb, "a point light has none")
        self.assertIn("direction", spot)
        self.assertIn("right", panel)

    def test_an_object_hosts_record_is_the_shape(self):
        """A host whose object names are unique writes every entry's
        ``hierarchy`` too (one shape for every host), so its record is the
        shape whole."""
        payload = LightmapRecord.payload(
            [OBJECT_ENTRY], probe=OBJECT_PROBE, unit_scale=1.0
        )
        self.assertEqual(_verdict(LightmapRecord, payload), ([], []))

    def test_the_fbx_highlight_pass_writes_the_shape(self):
        """The real pass: drawn from the room's geometry and its probe, the
        delivered probe with the lights in it and its complete copies, which
        ``embed_fbx_highlight_probes`` names on the FBX record's probe -- the
        record as the deliverable then carries it."""
        _require_cv2()
        import cv2
        import numpy as np
        from test_fbx_media import build_carrier_fbx

        probe = np.full((32, 64, 3), 0.2, np.float32)
        cv2.imwrite(os.path.join(self.tmp, "room_Probe.exr"), probe)
        record = LightmapRecord.payload(
            [
                dict(DAG_ENTRY, name=n, hierarchy=[n], scaleOffset=[1.0, 1.0, 0.0, 0.0])
                for n in ("FLOOR", "TABLE")
            ],
            probe={
                "map": "room_Probe.exr",
                "position": [0.0, 1.8, 2.0],
                "box": [[-4.0, 0.0, -4.0], [4.0, 4.0, 4.0]],
            },
            lights=[HANGING],
            unit_scale=1.0,
        )
        glb = _room_glb(os.path.join(self.tmp, "room.glb"), record)
        fbx = build_carrier_fbx(os.path.join(self.tmp, "room.fbx"), {"lightmaps": record})
        names = MeshConvert.embed_fbx_highlight_probes(fbx, glb, search_dirs=[self.tmp])
        self.assertTrue(names, "the pass drew nothing")
        shipped = FbxMedia.record(fbx, SceneRecords.LIGHTMAPS.key)
        drawn = [shipped["probe"]["map"]] + [h["map"] for h in shipped["probe"]["highlights"]]
        self.assertEqual(drawn, names)
        self.assertEqual(_verdict(LightmapRecord, shipped), ([], []))

    def test_a_missing_required_field_is_refused(self):
        cases = (
            ((), "version", "missing required key 'version'"),
            ((), "objects", "missing required key 'objects'"),
            (("objects", 0), "map", "objects[0].missing required key 'map'"),
            (("objects", 0), "hierarchy", "objects[0].missing required key 'hierarchy'"),
            (("probe",), "box", "probe.missing required key 'box'"),
            (("probe", "box"), "max", "probe.box.missing required key 'max'"),
            (
                ("probe", "highlights", 0, "objects", 0),
                "hierarchy",
                "probe.highlights[0].objects[0].missing required key 'hierarchy'",
            ),
            (("lights", 0), "color", "lights[0].missing required key 'color'"),
        )
        full = self._full()
        for path, key, error in cases:
            with self.subTest(error):
                errors, _warnings = _verdict(
                    LightmapRecord, _mutated(full, path, key, _GONE)
                )
                self.assertEqual(errors, [error])

    def test_a_wrong_type_is_refused(self):
        """A value its field does not admit -- the host's ``[min, max]`` box,
        a ``null`` where a producer leaves the key out, a light of a kind no
        record carries."""
        cases = (
            (
                ("objects", 0),
                "scaleOffset",
                [1.0, 1.0, 0.0],
                "objects[0].scaleOffset: expected 4 items, got 3",
            ),
            (
                ("objects", 0),
                "hierarchy",
                "ROOM|PROPS:crate",
                "objects[0].hierarchy: expected array, got string",
            ),
            ((), "unit_scale", None, "unit_scale: expected number, got null"),
            (
                ("probe",),
                "box",
                [[-4.0, 0.0, -4.0], [4.0, 4.0, 4.0]],
                "probe.box.expected a mapping, got list",
            ),
            (("probe",), "position", None, "probe.position: expected array, got null"),
            (
                ("lights", 0),
                "type",
                "directional",
                "lights[0].type: 'directional' is not one of ['point', 'spot', 'rect']",
            ),
            (("lights", 0), "intensity", True, "lights[0].intensity: expected number, got boolean"),
        )
        full = self._full()
        for path, key, value, error in cases:
            with self.subTest(error):
                errors, _warnings = _verdict(
                    LightmapRecord, _mutated(full, path, key, value)
                )
                self.assertEqual(errors, [error])

    def test_its_json_schema_builds(self):
        """The document a C# reader's record types are generated from: each
        nested type named for what it is once the family prefix goes, and the
        probe's box the one field a producer writes as ``null``."""
        schema = LightmapRecord.json_schema()
        json.dumps(schema)
        self.assertEqual(schema["title"], "LightmapRecord")
        self.assertEqual(schema["required"], ["version", "objects"])
        self.assertEqual(
            sorted(schema["$defs"]),
            [
                "LightmapBakedLight",
                "LightmapBox",
                "LightmapEntry",
                "LightmapHighlight",
                "LightmapNode",
                "LightmapProbe",
            ],
        )
        self.assertEqual(
            schema["$defs"]["LightmapProbe"]["properties"]["box"]["anyOf"],
            [{"$ref": "#/$defs/LightmapBox"}, {"type": "null"}],
        )
        for name in schema["$defs"]:
            with self.subTest(name):
                rest = name[len("Lightmap") :]
                self.assertTrue(name.startswith("Lightmap") and rest[:1].isupper())
                self.assertNotIn(rest, _CS_TAKEN)


class TestTheGlbCarriesTheRecordShape(unittest.TestCase):
    """The record a GLB carries as it stands (version 3: one layout, two
    containers) is a :class:`LightmapRecord` whatever the lightmap pass did
    to it -- the FBX route's (``apply_glb_lightmaps``), its probe drawn or
    not, and a native export's (the record lifted into the root extras, its
    files riding after the write)."""

    def setUp(self):
        self.tmp = _scratch(self)

    def test_the_lightmap_pass_writes_the_shape(self):
        """The FBX route's pass over a baked room, its probe riding -- in a
        room's box, and in the open (``null``)."""
        for box in (True, False):
            with self.subTest(box=box):
                glb, record = _lit_room(_scratch(self), box=box)
                carried = _glb_block(glb)
                self.assertEqual(carried, record, "as the FBX's carrier holds it")
                self.assertEqual(carried["probe"]["box"] is None, not box)
                self.assertEqual(_verdict(LightmapRecord, carried), ([], []))

    def test_a_native_exports_record_is_the_shape(self):
        """A host's native glTF export: the record in the root extras, its
        files riding after the write."""
        record = LightmapRecord.payload(
            [dict(OBJECT_ENTRY, name="FLOOR", hierarchy=["FLOOR"])],
            probe=dict(OBJECT_PROBE, box=[[-4.0, 0.0, -4.0], [4.0, 4.0, 4.0]]),
            unit_scale=1.0,
        )
        glb = _room_glb(os.path.join(self.tmp, "native.glb"), {})
        with MeshConvert.open_glb(glb) as edit:
            edit.gltf["nodes"] = [n for n in edit.gltf["nodes"] if n["name"] != "data_export"]
            edit.gltf["scenes"][0]["nodes"] = list(range(len(edit.gltf["nodes"])))
            edit.gltf["extras"] = {"lightmaps": record}
            edit.dirty = True
        for name in ("room_Lightmap.exr", "room_Probe.exr"):
            with open(os.path.join(self.tmp, name), "wb") as fh:
                fh.write(b"\x00")
        bound = MeshConvert.apply_glb_lightmaps(glb, search_dirs=[self.tmp], hdr_codec=_Codec)
        self.assertTrue(bound, "the record lit nothing")
        with MeshConvert.open_glb(glb) as edit:
            self.assertIsNotNone(SceneRecords.GLB_MEDIA.view(edit.gltf, "room_Probe.exr"))
        self.assertEqual(_verdict(LightmapRecord, _glb_block(glb)), ([], []))

    def test_the_highlight_copies_are_the_shape(self):
        """The pass over a room with a light over its table: the probe drawn
        and its complete copies, one per group of receivers, named on the
        record the GLB carries."""
        _require_cv2()
        glb, _record = _lit_room(self.tmp, lights=[HANGING], probe_pixels=True)
        carried = _glb_block(glb)
        self.assertTrue(carried["probe"].get("highlights"), "the pass drew no copy")
        self.assertEqual(_verdict(LightmapRecord, carried), ([], []))

if __name__ == "__main__":
    unittest.main()
