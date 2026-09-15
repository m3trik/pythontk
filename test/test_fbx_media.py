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


def build_stingray_fbx(path: str, maps) -> str:
    """A quad with a Stingray PBS material the FBX SDK -- so FBX2glTF -- binds.

    *maps* ``{"color" | "roughness" | "metallic": (file name, image bytes)}``
    embeds each image and wires it to ``Maya|TEX_<slot>_map``, as Maya writes
    a ``GameShader`` network. Beyond what :func:`build_fbx` writes, the SDK
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
        connections += [
            ("C", [b"OP", texture, material, f"Maya|TEX_{slot}_map".encode()], []),
            ("C", [b"OO", video, texture], []),
        ]
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


def build_animated_fbx(path: str, takes, declared=None, media=None) -> str:
    """An FBX shaped like a Maya take split: one stack per take in *takes*.

    Each take owns a layer, a curve node (wired to the model's translation)
    and a curve. One extra curve node + curve is SHARED between the first two
    takes' layers, so a drop must keep it while either owner survives. The
    ``Definitions`` counts match the objects. *declared* take names are
    published the way the exporters do -- a ``data_export`` model's
    ``fbx_takes`` user property.
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
    if declared is not None:
        channel = json.dumps([{"name": n, "start": 0, "end": 10} for n in declared])
        objects.append(
            (
                "Model",
                [1002, b"data_export\x00\x01Model", b"Null"],
                [
                    (
                        "Properties70",
                        [],
                        [
                            (
                                "P",
                                [b"fbx_takes", b"KString", b"", b"U", channel.encode()],
                                [],
                            )
                        ],
                    )
                ],
            )
        )
    media_objects, media_connections = _media_records(media or {})
    objects += media_objects
    connections += media_connections
    counts = {
        b"Model": 1 + (declared is not None),
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

    def test_user_properties_read_the_declared_channel(self):
        values = FbxFile.load(self.src).user_properties("fbx_takes")
        self.assertEqual(len(values), 1)
        self.assertEqual(
            [e["name"] for e in json.loads(values[0])], ["Shot_A", "Shot_B"]
        )
        self.assertEqual(FbxFile.load(self.src).user_properties("absent"), [])


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


if __name__ == "__main__":
    unittest.main()
