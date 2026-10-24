#!/usr/bin/python
# coding=utf-8
"""The ``lightmaps`` record's and GLB block's declared shapes, held to their producers.

:class:`LightmapRecord` (the record on a deliverable's ``data_export``
carrier) and :class:`LightmapsBlock` (a GLB's ``extras.lightmaps``) are typed
:class:`~pythontk.SchemaSpec` declarations: the types another language reads a
file through are generated from them. Each must admit exactly what every
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
from pythontk.core_utils.engines.scene_export.lightmap_record import LightmapsBlock
from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia

# ------------------------------------------------------- what producers write
#: An entry as mayatk's ``LightmapRecords._record`` spells one
#: (mayatk/light_utils/lightmap_baker/lightmap_records.py): the transform's
#: leaf name, namespace kept; the marker's map and rect; the lightmap set's
#: index; the DAG path's names, root first.
MAYA_ENTRY = {
    "name": "PROPS:crate",
    "map": "room_Lightmap.exr",
    "uvIndex": 1,
    "scaleOffset": [0.5, 0.5, 0.5, 0.0],
    "hierarchy": ["ROOM", "PROPS:props", "PROPS:crate"],
}
#: The probe mayatk hands the builder: ``LightmapRecords.probe()`` in the
#: working unit (``_in_working_unit``), its box the ``[min, max]`` points the
#: host keeps and its own ``unit_scale`` beside it.
MAYA_PROBE = {
    "map": "room_Probe.exr",
    "position": [0.0, 150.0, 200.0],
    "box": [[-400.0, 0.0, -400.0], [400.0, 400.0, 400.0]],
    "unit_scale": 0.01,
}
#: mayatk's lights, one of each kind, as its producer hands them to the
#: builder: ``LightUtils.specular_lights``' form (mayatk/light_utils/
#: _light_utils.py), the name -- the scene's bookkeeping -- left out: every
#: light its type, position, direction, colour, intensity and unit; a spot's
#: angle, penumbra and radius, a point's radius, a rect's right, width and
#: height.
MAYA_LIGHTS = [
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
#: An entry as blendertk's ``LightmapRecords._record`` spells one
#: (blendertk/light_utils/lightmap_baker/lightmap_records.py): the object's
#: name, its marker's map and rect, the lightmap layer's index, its
#: hierarchy (``_hierarchy``: the names root first, its own last).
BLENDER_ENTRY = {
    "name": "crate",
    "map": "room_Lightmap.exr",
    "uvIndex": 1,
    "scaleOffset": [1.0, 1.0, 0.0, 0.0],
    "hierarchy": ["crate"],
}
#: The probe blendertk hands the builder: ``LightmapRecords.probe()`` as
#: stored, in Blender units -- an open scene, its box ``None``.
BLENDER_PROBE = {
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
#: The probe entry ``MeshConvert._embed_lightmap_probe`` writes, with one copy
#: as ``_embed_highlight_probes`` lists it.
GLB_PROBE = {
    "bufferView": 3,
    "position": [0.0, 1.5, 2.0],
    "box": {"min": [-4.0, 0.0, -4.0], "max": [4.0, 4.0, 4.0]},
    "highlights": [{"bufferView": 4, "materials": ["tableMat"]}],
}

#: Names a nested C# record type must not take: sync_scene_records drops the
#: root's family prefix (``LightmapEntry`` is ``LightmapRecord.Shape.Entry``),
#: so these would shadow UnityEngine's or System's types, or the generated
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
    """A host's HDR codec (``hdr_codec``), as Blender hands one: a lightmap
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
#: A point light over the table, as mayatk's producer hands it (its direction
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


def _lit_room(folder, lights=(), box=True):
    """The room through ``MeshConvert.apply_glb_lightmaps`` and the record it
    carried: the floor and table baked as mayatk names them, each a map of
    its own (an atlas rect clones its material per object), the probe at
    head height in the room's box (or, without *box*, an open scene's), and
    *lights*. Returns ``(glb, record)``."""
    for name in ("room_Lightmap.exr", "room_Probe.exr"):
        with open(os.path.join(folder, name), "wb") as fh:
            fh.write(b"\x00")  # the codec reads none of it
    record = LightmapRecord.payload(
        [
            dict(MAYA_ENTRY, name=n, hierarchy=[n], scaleOffset=[1.0, 1.0, 0.0, 0.0])
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
    """*glb*'s ``lightmaps`` block, as a reader reads it."""
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
            [MAYA_ENTRY],
            probe=dict(MAYA_PROBE, highlights=[FBX_HIGHLIGHT]),
            lights=MAYA_LIGHTS,
            unit_scale=0.01,
        )

    def test_every_branch_of_the_builder_is_the_shape(self):
        """With and without a probe and lights: objects alone (no unit to
        state), a probe in a room and in the open (its box ``null``), lights
        alone, and everything with a highlight copy."""
        cases = {
            "objects alone": LightmapRecord.payload([MAYA_ENTRY]),
            "a probe in a room": LightmapRecord.payload(
                [MAYA_ENTRY], probe=MAYA_PROBE, unit_scale=0.01
            ),
            "a probe in the open": LightmapRecord.payload(
                [MAYA_ENTRY], probe=dict(MAYA_PROBE, box=None), unit_scale=0.01
            ),
            "lights alone": LightmapRecord.payload(
                [MAYA_ENTRY], lights=MAYA_LIGHTS, unit_scale=0.01
            ),
            "everything": self._full(),
        }
        self.assertNotIn("unit_scale", cases["objects alone"])
        self.assertIsNone(cases["a probe in the open"]["probe"]["box"])
        for label, payload in cases.items():
            with self.subTest(label):
                self.assertEqual(_verdict(LightmapRecord, payload), ([], []))

    def test_mayatks_record_is_the_shape(self):
        """mayatk's producer as it assembles the record: its entries, its
        probe in the working unit and its lights, through the builder."""
        payload = LightmapRecord.payload(
            [MAYA_ENTRY, dict(MAYA_ENTRY, name="TABLE", hierarchy=["ROOM", "TABLE"])],
            probe=MAYA_PROBE,
            lights=MAYA_LIGHTS,
            unit_scale=0.01,
        )
        self.assertEqual(_verdict(LightmapRecord, payload), ([], []))
        bulb, spot, panel = payload["lights"]
        self.assertNotIn("direction", bulb, "a point light has none")
        self.assertIn("direction", spot)
        self.assertIn("right", panel)

    def test_blendertks_record_is_the_shape(self):
        """blendertk's ``LightmapRecords._record`` writes every entry's
        ``hierarchy`` now (it wrote none until 2026-10-20), so its record is
        the shape whole."""
        payload = LightmapRecord.payload(
            [BLENDER_ENTRY], probe=BLENDER_PROBE, unit_scale=1.0
        )
        self.assertEqual(_verdict(LightmapRecord, payload), ([], []))

    def test_the_fbx_highlight_pass_writes_the_shape(self):
        """The real pass: the GLB built from a lit room draws its probe's
        copies, and ``embed_fbx_highlight_probes`` lists them on the FBX
        record's probe -- the record as the deliverable then carries it."""
        _require_cv2()
        from test_fbx_media import build_carrier_fbx

        glb, record = _lit_room(self.tmp, lights=[HANGING])
        fbx = build_carrier_fbx(os.path.join(self.tmp, "room.fbx"), {"lightmaps": record})
        names = MeshConvert.embed_fbx_highlight_probes(fbx, glb)
        self.assertTrue(names, "the pass drew no copy")
        shipped = FbxMedia.record(fbx, SceneRecords.LIGHTMAPS.key)
        self.assertEqual([h["map"] for h in shipped["probe"]["highlights"]], names)
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
            (("objects", 0), "uvIndex", "1", "objects[0].uvIndex: expected integer, got string"),
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
        """The document unitytk's C# record types are generated from: each
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


class TestLightmapsBlockShape(unittest.TestCase):
    """:class:`LightmapsBlock`: the block ``MeshConvert.lightmaps_block``
    builds, with the probe and its copies the lightmap pass embeds -- the FBX
    route's (``apply_glb_lightmaps``) and a native export's
    (``embed_glb_lightmap_probe``, blendertk's)."""

    def setUp(self):
        self.tmp = _scratch(self)

    @staticmethod
    def _full():
        """A block with every part: two materials and a boxed probe with a
        highlight copy."""
        return MeshConvert.lightmaps_block(
            {"floorMat": 2.5, "tableMat": 13.656251234}, probe=GLB_PROBE
        )

    def test_what_the_block_builder_writes_is_the_shape(self):
        cases = {
            "materials alone": MeshConvert.lightmaps_block({"roomMat": 1}),
            "the emissive carrier": MeshConvert.lightmaps_block(
                {"roomMat": 1.0}, carrier="emissive"
            ),
            "no lightmapped material": MeshConvert.lightmaps_block({}),
            "everything": self._full(),
        }
        for label, block in cases.items():
            with self.subTest(label):
                self.assertEqual(_verdict(LightmapsBlock, block), ([], []))

    def test_the_lightmap_pass_writes_the_shape(self):
        """The FBX route's pass over a baked room, its probe embedded -- in a
        room's box, and in the open (``null``)."""
        for box in (True, False):
            with self.subTest(box=box):
                glb, _record = _lit_room(_scratch(self), box=box)
                block = _glb_block(glb)
                self.assertEqual(sorted(block["materials"]), ["floorMat", "tableMat"])
                self.assertEqual(block["probe"]["box"] is None, not box)
                self.assertEqual(_verdict(LightmapsBlock, block), ([], []))

    def test_a_native_exports_probe_is_the_shape(self):
        """blendertk's native glTF export: the builder's block in the root
        extras, its probe embedded after the write."""
        gltf = {
            "asset": {"version": "2.0"},
            "extras": {"lightmaps": MeshConvert.lightmaps_block({"roomMat": 3.19})},
        }
        glb = _write_glb(os.path.join(self.tmp, "native.glb"), gltf)
        with open(os.path.join(self.tmp, "room_Probe.exr"), "wb") as fh:
            fh.write(b"\x00")
        probe = dict(BLENDER_PROBE, box=[[-4.0, 0.0, -4.0], [4.0, 4.0, 4.0]])
        entry = MeshConvert.embed_glb_lightmap_probe(
            glb, probe, [self.tmp], hdr_codec=_Codec
        )
        self.assertIsNotNone(entry, "the probe was not embedded")
        self.assertEqual(_verdict(LightmapsBlock, _glb_block(glb)), ([], []))

    def test_the_highlight_copies_are_the_shape(self):
        """The pass over a room with a light over its table: the probe's
        copies, one per group of receivers, in the block."""
        _require_cv2()
        glb, _record = _lit_room(self.tmp, lights=[HANGING])
        block = _glb_block(glb)
        self.assertTrue(block["probe"].get("highlights"), "the pass drew no copy")
        self.assertEqual(_verdict(LightmapsBlock, block), ([], []))

    def test_a_missing_required_field_is_refused(self):
        cases = (
            ((), "version", "missing required key 'version'"),
            ((), "carrier", "missing required key 'carrier'"),
            ((), "materials", "missing required key 'materials'"),
            (
                ("materials", "tableMat"),
                "intensity",
                "materials.tableMat.missing required key 'intensity'",
            ),
            (("probe",), "bufferView", "probe.missing required key 'bufferView'"),
            (("probe",), "box", "probe.missing required key 'box'"),
            (
                ("probe", "highlights", 0),
                "materials",
                "probe.highlights[0].missing required key 'materials'",
            ),
        )
        full = self._full()
        for path, key, error in cases:
            with self.subTest(error):
                errors, _warnings = _verdict(
                    LightmapsBlock, _mutated(full, path, key, _GONE)
                )
                self.assertEqual(errors, [error])

    def test_a_wrong_type_is_refused(self):
        cases = (
            ((), "carrier", "lightmap", "carrier: 'lightmap' is not one of ['occlusion', 'emissive']"),
            ((), "materials", [{"intensity": 2.5}], "materials: expected object, got array"),
            (
                ("materials", "tableMat"),
                "intensity",
                "2.5",
                "materials.tableMat.intensity: expected number, got string",
            ),
            (("probe",), "bufferView", 3.0, "probe.bufferView: expected integer, got number"),
            (("probe",), "position", [0.0, 1.5], "probe.position: expected 3 items, got 2"),
            (
                ("probe", "highlights", 0),
                "materials",
                "tableMat",
                "probe.highlights[0].materials: expected array, got string",
            ),
        )
        full = self._full()
        for path, key, value, error in cases:
            with self.subTest(error):
                errors, _warnings = _verdict(
                    LightmapsBlock, _mutated(full, path, key, value)
                )
                self.assertEqual(errors, [error])

    def test_its_json_schema_builds(self):
        """The document the WebXR runtime's JSDoc typedefs are generated from,
        each def by its own name: the box is the record's, in metres here."""
        schema = LightmapsBlock.json_schema()
        json.dumps(schema)
        self.assertEqual(schema["title"], "LightmapsBlock")
        self.assertEqual(schema["required"], ["version", "carrier", "materials"])
        self.assertEqual(
            sorted(schema["$defs"]),
            [
                "LightmapBox",
                "LightmapsBlockHighlight",
                "LightmapsBlockMaterial",
                "LightmapsBlockProbe",
            ],
        )
        self.assertEqual(schema["properties"]["carrier"]["enum"], ["occlusion", "emissive"])
        self.assertEqual(
            schema["properties"]["materials"]["additionalProperties"],
            {"$ref": "#/$defs/LightmapsBlockMaterial"},
        )


if __name__ == "__main__":
    unittest.main()
