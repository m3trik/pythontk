# !/usr/bin/python
# coding=utf-8
"""MeshConvert.gltf_to_fbx -- glTF -> FBX + manifest with no DCC in the loop.

The route a .glb takes into Maya when there is no Blender, or when the Blender at
hand refuses the file (Blender 5.1 refuses any glTF that requires
``KHR_texture_basisu``). Every assertion reads the FBX back through
:class:`FbxFile` and the sidecar through :class:`HandoffManifest` -- the payload
mayatk's ``BlenderSceneImport.import_payload`` consumes. Maya's own reading of it
(transforms, joints following the animation, skins binding) was verified in a
fresh mayapy on 2026-10-09. Added: 2026-10-09
"""

import base64
import json
import math
import os
import struct
import sys
import unittest
import zlib
from unittest import mock

from pythontk.core_utils.handoff.manifest import HandoffManifest
from pythontk.file_utils.mesh_convert import _gltf2fbx
from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert
from pythontk.file_utils.mesh_convert.fbx_file import FbxFile
from pythontk.file_utils.temp_artifacts import TempArtifacts

TICKS = 46186158000


def png_bytes(rgba=(255, 0, 0, 255)):
    """A valid 1x1 RGBA PNG."""

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"\x00" + bytes(rgba)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class GltfBuilder:
    """A glTF document assembled accessor by accessor, written as .glb or .gltf."""

    def __init__(self):
        self.blob = bytearray()
        self.views, self.accessors = [], []
        self.gltf = {"asset": {"version": "2.0"}}

    def _view(self, data):
        while len(self.blob) % 4:
            self.blob.append(0)
        self.views.append(
            {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        )
        self.blob += data
        return len(self.views) - 1

    def accessor(self, values, kind, ctype=5126):
        fmt = {5126: "f", 5123: "H", 5121: "B", 5125: "I"}[ctype]
        flat = [
            v
            for item in values
            for v in (item if isinstance(item, (list, tuple)) else [item])
        ]
        entry = {
            "bufferView": self._view(struct.pack(f"<{len(flat)}{fmt}", *flat)),
            "componentType": ctype,
            "count": len(values),
            "type": kind,
        }
        if kind == "VEC3" and ctype == 5126:
            entry["min"] = [min(v[i] for v in values) for i in range(3)]
            entry["max"] = [max(v[i] for v in values) for i in range(3)]
        self.accessors.append(entry)
        return len(self.accessors) - 1

    def image(self, data, name):
        self.gltf.setdefault("images", []).append(
            {"bufferView": self._view(data), "mimeType": "image/png", "name": name}
        )
        return len(self.gltf["images"]) - 1

    def write_glb(self, path):
        gltf = dict(self.gltf, bufferViews=self.views, accessors=self.accessors)
        gltf["buffers"] = [{"byteLength": len(self.blob)}]
        payload = json.dumps(gltf).encode("utf-8")
        payload += b" " * ((4 - len(payload) % 4) % 4)
        blob = bytes(self.blob) + b"\x00" * ((4 - len(self.blob) % 4) % 4)
        rest = struct.pack("<I4s", len(blob), b"BIN\x00") + blob
        with open(path, "wb") as fh:
            fh.write(struct.pack("<4sII", b"glTF", 2, 20 + len(payload) + len(rest)))
            fh.write(struct.pack("<I4s", len(payload), b"JSON") + payload)
            fh.write(rest)
        return path

    def write_gltf(self, path):
        """The .gltf flavour: JSON beside an external .bin; images as data URIs."""
        gltf = json.loads(json.dumps(dict(self.gltf)))
        for image in gltf.get("images", []):
            view = self.views[image.pop("bufferView")]
            data = bytes(
                self.blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
            )
            image["uri"] = "data:image/png;base64," + base64.b64encode(data).decode()
        gltf.update(bufferViews=self.views, accessors=self.accessors)
        bin_name = os.path.splitext(os.path.basename(path))[0] + ".bin"
        gltf["buffers"] = [{"byteLength": len(self.blob), "uri": bin_name}]
        with open(os.path.join(os.path.dirname(path), bin_name), "wb") as fh:
            fh.write(bytes(self.blob))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(gltf, fh)
        return path


def scene_builder(animated=True):
    """Root -> two nodes sharing one two-material quad (split at its diagonal
    the way glTF splits a vertex), a matrix-placed group with a leaf marker,
    a skinned triangle on a two-joint chain, and one clip."""
    b = GltfBuilder()
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    uv = [(0, 1), (1, 1), (1, 0), (0, 0)]  # glTF V runs down the image
    normal = [(0, 0, 1)] * 3
    prims = []
    for material, tri in ((0, (0, 1, 2)), (1, (0, 2, 3))):
        prims.append(
            {
                "attributes": {
                    "POSITION": b.accessor([corners[i] for i in tri], "VEC3"),
                    "NORMAL": b.accessor(normal, "VEC3"),
                    "TEXCOORD_0": b.accessor([uv[i] for i in tri], "VEC2"),
                },
                "indices": b.accessor([0, 1, 2], "SCALAR", 5123),
                "material": material,
            }
        )
    skinned = {
        "attributes": {
            "POSITION": b.accessor([(0, 0, 0), (0, 1, 0), (1, 0.5, 0)], "VEC3"),
            "JOINTS_0": b.accessor(
                [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)], "VEC4", 5121
            ),
            "WEIGHTS_0": b.accessor(
                [(1, 0, 0, 0), (1, 0, 0, 0), (0.5, 0.5, 0, 0)], "VEC4"
            ),
        },
    }
    ibm = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    ibm_j1 = ibm[:12] + [0, -1, 0, 1]
    half = math.sqrt(0.5)
    b.gltf.update(
        scene=0,
        scenes=[{"nodes": [0, 6]}],
        nodes=[
            {"name": "Root", "translation": [0, 1, 0], "children": [1, 2, 3, 5]},
            {"name": "Box.001", "mesh": 0, "translation": [1, 0, 0]},
            {"name": "Box.001", "mesh": 0, "translation": [-1, 0, 0]},
            {
                "name": "Turned",
                # 90 degrees about Y, then 2 m along Z (column-major).
                "matrix": [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 1],
                "children": [4],
            },
            {"name": "Marker"},
            {"name": "Skinned", "mesh": 1, "skin": 0, "translation": [5, 5, 5]},
            {"name": "J0", "children": [7]},
            {"name": "J1", "translation": [0, 1, 0]},
        ],
        meshes=[
            {"name": "Quad", "primitives": prims},
            {"name": "Tri", "primitives": [skinned]},
        ],
        skins=[
            {
                "joints": [6, 7],
                "inverseBindMatrices": b.accessor([ibm, ibm_j1], "MAT4"),
            }
        ],
        materials=[
            {
                "name": "Mat A",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                },
                "occlusionTexture": {"index": 1},
            },
            {
                "name": "Mat B",
                "pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 1]},
            },
        ],
        textures=[{"source": 0}, {"source": 1}],
    )
    b.image(png_bytes((255, 0, 0, 255)), "red.png")
    b.image(png_bytes((255, 128, 0, 255)), "orm.png")
    if animated:
        times = [i / 30.0 for i in range(31)]
        b.gltf["animations"] = [
            {
                "name": "Move",
                "samplers": [
                    {
                        "input": b.accessor(times, "SCALAR"),
                        "output": b.accessor([(1 + t, 0, 0) for t in times], "VEC3"),
                    },
                    {
                        # Two keys a second apart: slerped, so resampled per frame.
                        "input": b.accessor([0.0, 1.0], "SCALAR"),
                        "output": b.accessor(
                            [(0, 0, 0, 1), (0, half, 0, half)], "VEC4"
                        ),
                    },
                ],
                "channels": [
                    {"sampler": 0, "target": {"node": 1, "path": "translation"}},
                    {"sampler": 1, "target": {"node": 3, "path": "rotation"}},
                ],
            }
        ]
    return b


class Gltf2FbxTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = TempArtifacts("test_gltf2fbx", policy="scoped")
        self.dir = self.temp.dir_path()

    def tearDown(self):
        self.temp.cleanup()

    def _convert(self, builder=None, gltf=False, **kw):
        builder = builder or scene_builder()
        src = os.path.join(self.dir, "scene.gltf" if gltf else "scene.glb")
        (builder.write_gltf if gltf else builder.write_glb)(src)
        report = {}
        dst = MeshConvert.gltf_to_fbx(
            src, os.path.join(self.dir, "out.fbx"), overwrite=True, report=report, **kw
        )
        fbx = FbxFile.load(dst, decode_arrays=True)
        return dst, fbx, HandoffManifest.read(dst), report

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _objects(fbx, kind):
        objects = next(r for r in fbx.roots if r["name"] == "Objects")
        return [c for c in objects["children"] if c["name"] == kind]

    @staticmethod
    def _name(record):
        return record["props"][1].split(b"\x00\x01")[0].decode()

    @staticmethod
    def _p70(record, name):
        p70 = next(c for c in record["children"] if c["name"] == "Properties70")
        row = next(c for c in p70["children"] if c["props"][0] == name.encode())
        return row["props"][4:]

    @staticmethod
    def _child(record, name):
        return next(c for c in record["children"] if c["name"] == name)

    @staticmethod
    def _links(fbx):
        conns = next(r for r in fbx.roots if r["name"] == "Connections")
        return [tuple(c["props"]) for c in conns["children"]]

    def _models(self, fbx):
        return {self._name(m): m for m in self._objects(fbx, "Model")}

    def _parent(self, fbx, name):
        """The name of model *name*'s parent model, or None at the scene root."""
        models = self._models(fbx)
        names = {m["props"][0]: n for n, m in models.items()}
        child = models[name]["props"][0]
        parents = [
            p
            for _, c, p, *_ in self._links(fbx)
            if c == child and (p == 0 or p in names)
        ]
        self.assertEqual(len(parents), 1, f"{name} links to {parents}")
        return names.get(parents[0])

    def _world_translation(self, fbx, name):
        """Model *name*'s world translation (cm), summed up its parent links --
        exact for the translate-only chains these fixtures build (asserted)."""
        models, world = self._models(fbx), [0.0, 0.0, 0.0]
        while name is not None:
            self.assertEqual(self._p70(models[name], "Lcl Rotation"), [0.0] * 3)
            self.assertEqual(self._p70(models[name], "Lcl Scaling"), [1.0] * 3)
            local = self._p70(models[name], "Lcl Translation")
            world = [w + t for w, t in zip(world, local)]
            name = self._parent(fbx, name)
        return world

    @staticmethod
    def _skinned_with_child(animated=False):
        """The scene with "Accessory" (1 m along Z) parented under the skinned
        node (5 m along each axis, under Root's 1 m along Y)."""
        builder = scene_builder(animated=animated)
        nodes = builder.gltf["nodes"]
        nodes.append({"name": "Accessory", "translation": [0, 0, 1]})
        nodes[5]["children"] = [len(nodes) - 1]
        return builder

    # ------------------------------------------------------------ hierarchy
    def test_every_node_lands_with_a_name_maya_holds(self):
        """Names fold to [A-Za-z0-9_] and collide apart with a numbered tail,
        the spelling the Blender pull's sanitize pass writes."""
        _, fbx, _, report = self._convert()
        models = self._models(fbx)
        self.assertEqual(
            set(models),
            {
                "Root",
                "Box_001",
                "Box_001_001",
                "Turned",
                "Marker",
                "Skinned",
                "J0",
                "J1",
            },
        )
        kinds = {name: m["props"][2].decode() for name, m in models.items()}
        self.assertEqual(kinds["Box_001"], "Mesh")
        self.assertEqual(kinds["Marker"], "Null")
        self.assertEqual(kinds["J1"], "LimbNode")
        self.assertEqual(report["nodes"], 8)

    def test_lengths_arrive_in_centimetres(self):
        _, fbx, _, _ = self._convert()
        models = self._models(fbx)
        self.assertEqual(
            self._p70(models["Box_001"], "Lcl Translation"), [100.0, 0.0, 0.0]
        )
        self.assertEqual(
            self._p70(models["Root"], "Lcl Translation"), [0.0, 100.0, 0.0]
        )
        settings = next(r for r in fbx.roots if r["name"] == "GlobalSettings")
        self.assertEqual(self._p70(settings, "UnitScaleFactor"), [1.0])
        self.assertEqual(self._p70(settings, "UpAxis"), [1])

    def test_a_matrix_node_decomposes_to_clean_euler(self):
        _, fbx, _, _ = self._convert()
        turned = self._models(fbx)["Turned"]
        rotation = self._p70(turned, "Lcl Rotation")
        for got, want in zip(rotation, (0.0, 90.0, 0.0)):
            self.assertAlmostEqual(got, want, places=4)
        self.assertEqual(self._p70(turned, "Lcl Translation"), [0.0, 0.0, 200.0])

    def test_a_shared_mesh_is_one_geometry_and_a_recorded_instance(self):
        _, fbx, manifest, _ = self._convert()
        models = self._models(fbx)
        quad = next(
            g for g in self._objects(fbx, "Geometry") if self._name(g) == "Quad"
        )
        parents = [
            p for kind, child, p, *_ in self._links(fbx) if child == quad["props"][0]
        ]
        self.assertEqual(
            sorted(parents),
            sorted([models["Box_001"]["props"][0], models["Box_001_001"]["props"][0]]),
        )
        self.assertEqual(
            manifest.get(HandoffManifest.INSTANCES), [["Box_001", "Box_001_001"]]
        )
        self.assertEqual(
            manifest.get(HandoffManifest.FORMAT_KEY), HandoffManifest.FORMAT_NAMES
        )

    # ------------------------------------------------------------ geometry
    def test_split_vertices_weld_back_into_shared_points(self):
        """glTF split the quad's diagonal into six vertices across two
        primitives; the FBX mesh has its four, and one polygon per triangle."""
        _, fbx, _, _ = self._convert()
        quad = next(
            g for g in self._objects(fbx, "Geometry") if self._name(g) == "Quad"
        )
        self.assertEqual(len(self._child(quad, "Vertices")["props"][0]), 4 * 3)
        index = list(self._child(quad, "PolygonVertexIndex")["props"][0])
        self.assertEqual(len(index), 6)
        self.assertEqual(sum(1 for i in index if i < 0), 2)
        mats = self._child(quad, "LayerElementMaterial")
        self.assertEqual(
            self._child(mats, "MappingInformationType")["props"][0], b"ByPolygon"
        )
        self.assertEqual(list(self._child(mats, "Materials")["props"][0]), [0, 1])

    def test_uv_v_is_flipped_to_fbx_convention(self):
        _, fbx, _, _ = self._convert()
        quad = next(
            g for g in self._objects(fbx, "Geometry") if self._name(g) == "Quad"
        )
        layer = self._child(quad, "LayerElementUV")
        uvs = list(self._child(layer, "UV")["props"][0])
        index = list(self._child(layer, "UVIndex")["props"][0])
        # Corner 0 of the first triangle sat at glTF (0, 1): image bottom-left.
        self.assertEqual(uvs[index[0] * 2 : index[0] * 2 + 2], [0.0, 0.0])
        self.assertEqual(self._child(layer, "Name")["props"][0], b"map1")

    # ------------------------------------------------------------ skins
    def test_a_skin_binds_through_clusters_and_a_bind_pose(self):
        _, fbx, _, report = self._convert()
        models = self._models(fbx)
        self.assertEqual(
            self._p70(models["Skinned"], "Lcl Translation"), [0.0, 0.0, 0.0]
        )
        clusters = {
            self._name(d): d
            for d in self._objects(fbx, "Deformer")
            if d["props"][2] == b"Cluster"
        }
        self.assertEqual(set(clusters), {"J0", "J1"})
        j1 = clusters["J1"]
        self.assertEqual(list(self._child(j1, "Indexes")["props"][0]), [1, 2])
        self.assertEqual(list(self._child(j1, "Weights")["props"][0]), [1.0, 0.5])
        link = list(self._child(j1, "TransformLink")["props"][0])
        self.assertEqual(link[12:15], [0.0, 100.0, 0.0])  # J1's bind world, in cm
        self.assertEqual(report["skins"], 1)
        self.assertEqual(len(self._objects(fbx, "Pose")), 1)

    def test_a_skinned_nodes_children_inherit_it_through_a_pivot(self):
        """glTF ignores a skinned mesh node's own transform for its vertices,
        not for its children. The mesh stays at the root at identity; a Null
        pivot in its place carries the transform the children inherit."""
        _, fbx, manifest, _ = self._convert(self._skinned_with_child())
        self.assertEqual(
            self._world_translation(fbx, "Accessory"), [500.0, 600.0, 600.0]
        )
        models = self._models(fbx)
        pivot = models["Skinned_pivot"]
        self.assertEqual(pivot["props"][2], b"Null")
        self.assertEqual(self._p70(pivot, "Lcl Translation"), [500.0, 500.0, 500.0])
        self.assertEqual(self._parent(fbx, "Skinned_pivot"), "Root")
        self.assertEqual(self._parent(fbx, "Accessory"), "Skinned_pivot")
        self.assertIsNone(self._parent(fbx, "Skinned"))
        self.assertEqual(
            self._p70(models["Skinned"], "Lcl Translation"), [0.0, 0.0, 0.0]
        )
        empties = {e["name"] for e in manifest.get(HandoffManifest.EMPTIES)}
        self.assertIn("Skinned_pivot", empties)
        # The bind pose holds the joints and the mesh, never the pivot.
        pose = self._objects(fbx, "Pose")[0]
        posed = {
            self._child(n, "Node")["props"][0]
            for n in pose["children"]
            if n["name"] == "PoseNode"
        }
        self.assertNotIn(pivot["props"][0], posed)
        self.assertIn(models["Skinned"]["props"][0], posed)

    def test_a_childless_skinned_node_gets_no_pivot(self):
        _, fbx, _, report = self._convert()
        models = self._models(fbx)
        self.assertEqual([n for n in models if n.endswith("_pivot")], [])
        self.assertEqual(len(models), report["nodes"])
        self.assertIsNone(self._parent(fbx, "Skinned"))

    def test_a_skinned_node_under_a_skinned_node_chains_both_pivots(self):
        """The inner node's pivot links under the outer one's. A pivot never
        takes a node's name: a node already called "Skinned_pivot" keeps it,
        and the pivot takes the numbered tail."""
        builder = self._skinned_with_child()
        nodes = builder.gltf["nodes"]
        inner, leaf, taken = len(nodes), len(nodes) + 1, len(nodes) + 2
        nodes[5]["children"].append(inner)
        nodes.append(
            {
                "name": "Inner",
                "mesh": 1,
                "skin": 0,
                "translation": [0, 2, 0],
                "children": [leaf],
            }
        )
        nodes.append({"name": "Leaf", "translation": [1, 0, 0]})
        nodes.append({"name": "Skinned_pivot"})
        nodes[0]["children"].append(taken)
        _, fbx, _, _ = self._convert(builder)
        self.assertEqual(self._world_translation(fbx, "Leaf"), [600.0, 800.0, 500.0])
        self.assertEqual(self._parent(fbx, "Leaf"), "Inner_pivot")
        self.assertEqual(self._parent(fbx, "Inner_pivot"), "Skinned_pivot_001")
        self.assertEqual(self._parent(fbx, "Skinned_pivot_001"), "Root")
        self.assertEqual(self._parent(fbx, "Skinned_pivot"), "Root")
        self.assertIsNone(self._parent(fbx, "Inner"))

    def test_a_skinned_nodes_animation_lands_on_its_pivot(self):
        """Its curves drive what its children inherit; on the mesh model they
        would move vertices glTF never moved."""
        builder = self._skinned_with_child(animated=True)
        builder.gltf["animations"][0]["channels"].append(
            {"sampler": 0, "target": {"node": 5, "path": "translation"}}
        )
        _, fbx, _, _ = self._convert(builder)
        names = {m["props"][0]: n for n, m in self._models(fbx).items()}
        driven = {
            (names[link[2]], link[3])
            for link in self._links(fbx)
            if link[0] == b"OP" and link[2] in names
        }
        self.assertIn(("Skinned_pivot", b"Lcl Translation"), driven)
        self.assertNotIn("Skinned", {name for name, _ in driven})

    def test_a_childless_skinned_nodes_animation_is_still_dropped(self):
        builder = scene_builder()
        builder.gltf["animations"][0]["channels"].append(
            {"sampler": 0, "target": {"node": 5, "path": "translation"}}
        )
        _, fbx, _, _ = self._convert(builder)
        skinned = self._models(fbx)["Skinned"]["props"][0]
        self.assertEqual(
            [
                link
                for link in self._links(fbx)
                if link[0] == b"OP" and link[2] == skinned
            ],
            [],
        )

    # ------------------------------------------------------------ animation
    def test_one_clip_keys_at_the_frame_rate_its_spacing_names(self):
        _, fbx, manifest, report = self._convert()
        self.assertEqual(report["fps"], 30.0)
        self.assertEqual(report["animation"], "Move")
        curves = self._objects(fbx, "AnimationCurve")
        lengths = sorted({len(self._child(c, "KeyTime")["props"][0]) for c in curves})
        self.assertEqual(lengths, [31])  # 31 baked keys; 2 slerped keys resampled
        values = [list(self._child(c, "KeyValueFloat")["props"][0]) for c in curves]
        self.assertIn(
            [round(100 + 100 * i / 30.0, 3) for i in range(31)],
            [[round(v, 3) for v in vs] for vs in values],
        )
        self.assertTrue(any(abs(vs[-1] - 90.0) < 1e-3 for vs in values))
        times = list(self._child(curves[0], "KeyTime")["props"][0])
        self.assertEqual((times[0], times[-1]), (0, TICKS))
        scene = manifest.get(HandoffManifest.SCENE)
        self.assertEqual(
            (scene["fps"], scene["frame_start"], scene["frame_end"]), (30.0, 0, 30)
        )

    def test_a_static_file_leaves_the_host_clock_alone(self):
        _, fbx, manifest, _ = self._convert(scene_builder(animated=False))
        self.assertIsNone(manifest.get(HandoffManifest.SCENE))
        self.assertEqual(self._objects(fbx, "AnimationStack"), [])

    # ------------------------------------------------------------ materials
    def test_images_are_named_for_the_map_they_feed(self):
        """The consumer classifies a manifest file by FILENAME, so an image the
        glTF names "orm.png" lands as <material>_ORM, and the base colour as
        <material>_Base_Color -- the Blender pull's naming."""
        _, _, manifest, report = self._convert()
        entries = manifest.get(HandoffManifest.MATERIALS)
        self.assertEqual([e["name"] for e in entries], ["Mat_A"])
        entry = entries[0]
        names = sorted(os.path.basename(f) for f in entry["files"])
        self.assertEqual(names, ["Mat_A_Base_Color.png", "Mat_A_ORM.png"])
        self.assertTrue(all(os.path.isfile(f) for f in entry["files"]))
        self.assertEqual(
            os.path.basename(entry["slots"]["baseColor"]), "Mat_A_Base_Color.png"
        )
        self.assertEqual(sorted(entry["objects"]), ["Box_001", "Box_001_001"])
        self.assertEqual(
            sorted(manifest.get(HandoffManifest.SCENE_MATERIALS)), ["Mat_A", "Mat_B"]
        )
        self.assertEqual(report["textures"], 2)

    def test_empties_are_recorded_for_the_locator_repair(self):
        _, _, manifest, _ = self._convert()
        empties = {e["name"] for e in manifest.get(HandoffManifest.EMPTIES)}
        self.assertEqual(empties, {"Root", "Turned", "Marker"})

    def test_an_undecodable_ktx2_image_costs_its_texture_not_the_file(self):
        builder = scene_builder(animated=False)
        builder.gltf["textures"][0] = {
            "extensions": {"KHR_texture_basisu": {"source": 0}}
        }
        builder.gltf["images"][0]["mimeType"] = "image/ktx2"
        with mock.patch(
            "pythontk.img_utils.ktx2_encoder.Ktx2Encoder.resolve_ktx", return_value=None
        ):
            _, fbx, manifest, report = self._convert(builder)
        self.assertEqual(report["textures_missing"], 1)
        self.assertIn("textures (not decoded)", report["skipped"])
        self.assertEqual(len(self._objects(fbx, "Geometry")), 2)

    def _webp_builder(self):
        """Texture 0's image as WebP -- what EXT_texture_webp ships, and what
        Maya cannot read (a file node on a .webp reports 0 x 0)."""
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (4, 4), (200, 30, 10)).save(buf, format="WEBP", lossless=True)
        builder = scene_builder(animated=False)
        index = builder.image(buf.getvalue(), "lightmap.png")
        builder.gltf["images"][index]["mimeType"] = "image/webp"
        builder.gltf["textures"][0] = {
            "extensions": {"EXT_texture_webp": {"source": index}}
        }
        return builder

    def _assert_png_red(self, manifest):
        from PIL import Image

        base = manifest.get(HandoffManifest.MATERIALS)[0]["slots"]["baseColor"]
        self.assertTrue(base.endswith("Mat_A_Base_Color.png"), base)
        with Image.open(base) as im:
            self.assertEqual(im.format, "PNG")
            self.assertEqual(im.convert("RGB").getpixel((1, 1)), (200, 30, 10))

    def test_a_webp_image_lands_as_png(self):
        _, _, manifest, report = self._convert(self._webp_builder())
        self._assert_png_red(manifest)
        self.assertEqual(report["textures_missing"], 0)

    def test_without_pillow_ffmpeg_decodes_webp(self):
        """mayapy ships no Pillow; FFmpeg decodes the same pixels."""
        import pythontk as ptk

        if not ptk.VidUtils.resolve_ffmpeg(required=False):
            self.skipTest("FFmpeg absent")
        builder = self._webp_builder()  # built WITH Pillow, decoded without
        with mock.patch.dict(sys.modules, {"PIL": None}):
            _, _, manifest, _ = self._convert(builder)
        self._assert_png_red(manifest)

    def test_a_mesh_that_converts_to_nothing_leaves_its_nodes_empty(self):
        builder = scene_builder(animated=False)
        for prim in builder.gltf["meshes"][0]["primitives"]:
            prim["mode"] = 1  # lines: not carried
        _, fbx, _, report = self._convert(builder)
        models = self._models(fbx)
        self.assertEqual(models["Box_001"]["props"][2], b"Null")
        self.assertEqual(report["skipped"]["point/line primitives"], ["Quad"])
        self.assertEqual(report["meshes"], 1)

    # ------------------------------------------------------------ containers
    def test_a_gltf_with_an_external_buffer_converts_the_same(self):
        _, fbx, manifest, _ = self._convert(gltf=True)
        self.assertEqual(len(self._models(fbx)), 8)
        self.assertEqual(len(manifest.get(HandoffManifest.MATERIALS)[0]["files"]), 2)

    def test_compressed_geometry_is_refused_before_anything_is_written(self):
        builder = scene_builder(animated=False)
        builder.gltf["extensionsRequired"] = ["KHR_draco_mesh_compression"]
        with self.assertRaises(ValueError):
            self._convert(builder)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "out.fbx")))

    def test_an_existing_destination_needs_overwrite(self):
        src = scene_builder().write_glb(os.path.join(self.dir, "a.glb"))
        dst = MeshConvert.gltf_to_fbx(src)
        with self.assertRaises(FileExistsError):
            MeshConvert.gltf_to_fbx(src, dst)

    # ------------------------------------------------------------ math
    def test_euler_takes_the_spelling_nearest_the_last_key(self):
        self.assertEqual(
            _gltf2fbx._euler_near((180.0, 90.0, 180.0), (0.0, 0.0, 0.0)),
            (0.0, 90.0, 0.0),
        )
        self.assertEqual(
            _gltf2fbx._euler_near((-170.0, 0.0, 0.0), (175.0, 0.0, 0.0)),
            (190.0, 0.0, 0.0),
        )

    def test_map_suffix_matches_the_blender_pull_rules(self):
        suffix = _gltf2fbx._map_suffix
        self.assertEqual(suffix({"Base Color", "Alpha"}), "Base_Color")
        self.assertEqual(suffix({"Metallic", "Roughness"}), "ORM")
        self.assertEqual(suffix({"Occlusion"}), "Ambient_Occlusion")
        self.assertEqual(suffix({"Normal"}), "Normal_OpenGL")
        self.assertIsNone(suffix(set()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
