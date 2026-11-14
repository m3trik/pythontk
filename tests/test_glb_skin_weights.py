# !/usr/bin/python
# coding=utf-8
"""Tests for :class:`GlbSkinWeights` -- the skin weights FBX2glTF hands one
mesh from another.

The fixture is two identical skinned strips whose chains run opposite ways, the
shape of a duplicated rigged prop rigged in reverse (a production assembly's
pair of electrical plugs, 2026-10-07): every vertex of the second strip is
bit-identical to one of the first, and FBX2glTF merges them, weights and all.
Added: 2026-10-07
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert  # noqa: E402
from pythontk.file_utils.mesh_convert.glb.reader import GlbReader  # noqa: E402
from pythontk.file_utils.mesh_convert.glb.skin_weights import (  # noqa: E402
    GlbSkinWeights,
)
from test_fbx_media import SDK_IDS, I32, Doubles, Ints, Raw, _write_fbx  # noqa: E402

_FBX2GLTF = MeshConvert.resolve_binary(required=False, auto_install=False)


def _p(*values):
    return ("P", list(values), [])


def _at(x, y, z):
    """A translation matrix, as the FBX stores a 4x4 (row-vector, flat)."""
    return Doubles([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1])


def build_skinned_fbx(path, meshes, split=False):
    """Write a binary FBX of 2-quad strips, each skinned to a 2-joint chain.

    *meshes* is ``[(name, (x, y, z), reverse, lift)]``. Each strip rises up +Y
    in rows ``y = 0, 1, 2`` (two control points a row) placed at ``(x, y, z)``;
    its chain roots at row 0 -- or, with *reverse*, at row 2 -- so the same
    point weighs onto the OTHER joint index. *lift* shifts the strip's local
    points, which makes its vertices its own. Every point is weighted wholly to
    one joint: the root owns its end row, the second joint the other two.

    *split* unwelds the middle row: the upper quad gets points of its own (6,
    7) coincident with 2 and 3 but UV'd apart (v 0.6), and a reversed strip
    weighs them onto its root -- two points at one position, told apart only by
    their UVs.
    """
    points = [-0.5, 0, 0, 0.5, 0, 0, -0.5, 1, 0, 0.5, 1, 0, -0.5, 2, 0, 0.5, 2, 0]
    polygons = [0, 1, 3, -3, 2, 3, 5, -5]
    uvs = [0, 0, 1, 0, 0, 0.5, 1, 0.5, 0, 1, 1, 1]
    if split:
        points += [-0.5, 1, 0, 0.5, 1, 0]
        polygons = [0, 1, 3, -3, 6, 7, 5, -5]
        uvs += [0, 0.6, 1, 0.6]
    count = len(points) // 3
    objects, connections, pose = [], [], []
    ids = iter(range(1001, 9999))

    def element(kind, version, name, data, mapping=b"ByControlPoint", ref=b"Direct"):
        head = [
            ("Version", [I32(version)], []),
            ("Name", [name], []),
            ("MappingInformationType", [mapping], []),
            ("ReferenceInformationType", [ref], []),
        ]
        return (kind, [I32(0)], head + data)

    # UVs per polygon corner, indexed -- the form Maya writes, and the one the
    # FBX SDK hands FBX2glTF (it drops a by-control-point UV set).
    corners = [p if p >= 0 else ~p for p in polygons]

    for name, (x, y, z), reverse, lift in meshes:
        local = [c + (lift if i % 3 == 1 else 0) for i, c in enumerate(points)]
        geometry, model = next(ids), next(ids)
        layers = [
            ("LayerElement", [], [("Type", [kind], []), ("TypedIndex", [I32(0)], [])])
            for kind in (b"LayerElementNormal", b"LayerElementUV")
        ]
        objects.append(
            (
                "Geometry",
                [geometry, b"\x00\x01Geometry", b"Mesh"],
                [
                    ("Vertices", [Doubles(local)], []),
                    ("PolygonVertexIndex", [Ints(polygons)], []),
                    ("GeometryVersion", [I32(124)], []),
                    element(
                        "LayerElementNormal",
                        102,
                        b"",
                        [("Normals", [Doubles([0, 0, 1] * count)], [])],
                    ),
                    element(
                        "LayerElementUV",
                        101,
                        b"map1",
                        [
                            ("UV", [Doubles(uvs)], []),
                            ("UVIndex", [Ints(corners)], []),
                        ],
                        mapping=b"ByPolygonVertex",
                        ref=b"IndexToDirect",
                    ),
                    ("Layer", [I32(0)], [("Version", [I32(100)], [])] + layers),
                ],
            )
        )
        translation = _p(
            b"Lcl Translation",
            b"Lcl Translation",
            b"",
            b"A",
            float(x),
            float(y),
            float(z),
        )
        objects.append(
            (
                "Model",
                [model, name.encode() + b"\x00\x01Model", b"Mesh"],
                [
                    ("Version", [I32(232)], []),
                    (
                        "Properties70",
                        [],
                        [_p(b"DefaultAttributeIndex", b"int", b"Integer", b"", I32(0))]
                        + [translation],
                    ),
                    ("Shading", [True], []),
                    ("Culling", [b"CullingOff"], []),
                ],
            )
        )
        connections += [
            ("C", [b"OO", model, 0], []),
            ("C", [b"OO", geometry, model], []),
        ]
        pose.append(
            ("PoseNode", [], [("Node", [model], []), ("Matrix", [_at(x, y, z)], [])])
        )
        rows = [2, 1] if reverse else [0, 1]
        top = [4, 5, 6, 7] if split else [4, 5]
        owned = {0: [0, 1], 2: top}
        skin = next(ids)
        objects.append(
            (
                "Deformer",
                [skin, b"\x00\x01Deformer", b"Skin"],
                [
                    ("Version", [I32(101)], []),
                    ("Link_DeformAcuracy", [50.0], []),
                    ("SkinningType", [b"Linear"], []),
                ],
            )
        )
        connections.append(("C", [b"OO", skin, geometry], []))
        parent = 0
        for k, row in enumerate(rows):
            joint, attribute, cluster = next(ids), next(ids), next(ids)
            height = row + lift if k == 0 else rows[k] - rows[k - 1]
            joint_t = (
                x if k == 0 else 0.0,
                (y if k == 0 else 0.0) + height,
                z if k == 0 else 0.0,
            )
            objects += [
                (
                    "NodeAttribute",
                    [attribute, b"\x00\x01NodeAttribute", b"LimbNode"],
                    [("TypeFlags", [b"Skeleton"], [])],
                ),
                (
                    "Model",
                    [
                        joint,
                        f"{name}_jnt_{k + 1}".encode() + b"\x00\x01Model",
                        b"LimbNode",
                    ],
                    [
                        ("Version", [I32(232)], []),
                        (
                            "Properties70",
                            [],
                            [
                                _p(
                                    b"Lcl Translation",
                                    b"Lcl Translation",
                                    b"",
                                    b"A",
                                    *map(float, joint_t),
                                )
                            ],
                        ),
                    ],
                ),
            ]
            rest = [p for p in range(count) if p not in (top if reverse else [0, 1])]
            points_of = owned[row] if row in owned else rest
            objects.append(
                (
                    "Deformer",
                    [cluster, b"\x00\x01SubDeformer", b"Cluster"],
                    [
                        ("Version", [I32(100)], []),
                        ("UserData", [b"", b""], []),
                        ("Indexes", [Ints(points_of)], []),
                        ("Weights", [Doubles([1.0] * len(points_of))], []),
                        ("Transform", [_at(0, -(row + lift), 0)], []),
                        ("TransformLink", [_at(x, y + row + lift, z)], []),
                    ],
                )
            )
            connections += [
                ("C", [b"OO", joint, parent], []),
                ("C", [b"OO", attribute, joint], []),
                ("C", [b"OO", cluster, skin], []),
                ("C", [b"OO", joint, cluster], []),
            ]
            pose.append(
                (
                    "PoseNode",
                    [],
                    [
                        ("Node", [joint], []),
                        ("Matrix", [_at(x, y + row + lift, z)], []),
                    ],
                )
            )
            parent = joint
    objects.append(
        (
            "Pose",
            [next(ids), b"BindPose\x00\x01Pose", b"BindPose"],
            [
                ("Type", [b"BindPose"], []),
                ("Version", [I32(100)], []),
                ("NbPoseNodes", [I32(len(pose))], []),
            ]
            + pose,
        )
    )
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


#: What each row of a strip is bound to: the root owns its end row.
FORWARD = {0: "jnt_1", 1: "jnt_2", 2: "jnt_2"}
REVERSED = {0: "jnt_2", 1: "jnt_2", 2: "jnt_1"}


def row_bindings(glb, name):
    """``{row: joint suffix}`` of *name*'s vertices, read off the GLB."""
    reader = GlbReader.load(glb)
    node = reader.gltf["nodes"][reader.node_index(name)]
    skin = reader.gltf["skins"][node["skin"]]
    joints = [reader.gltf["nodes"][j]["name"] for j in skin["joints"]]
    out = {}
    for primitive in reader.gltf["meshes"][node["mesh"]]["primitives"]:
        attributes = primitive["attributes"]
        positions = reader.accessor(attributes["POSITION"])
        indices = reader.accessor(attributes["JOINTS_0"])
        weights = reader.accessor(attributes["WEIGHTS_0"])
        lowest = min(p[1] for p in positions)
        for position, joint, weight in zip(positions, indices, weights):
            heaviest = max(range(4), key=lambda i: weight[i])
            row = round((position[1] - lowest) * 100)
            out.setdefault(row, set()).add(joints[joint[heaviest]].split("_", 1)[1])
    return {
        row: names.pop() if len(names) == 1 else names for row, names in out.items()
    }


def seam_bindings(glb, name):
    """``{(row, upper quad's own point): joint suffix}`` of *name*'s vertices.

    The split strip's middle row holds two points per position; the upper
    quad's are the ones UV'd at v 0.6 (0.4 once FBX2glTF flips V), and its
    top row belongs to the upper quad too."""
    reader = GlbReader.load(glb)
    node = reader.gltf["nodes"][reader.node_index(name)]
    joints = [
        reader.gltf["nodes"][j]["name"]
        for j in reader.gltf["skins"][node["skin"]]["joints"]
    ]
    out = {}
    for primitive in reader.gltf["meshes"][node["mesh"]]["primitives"]:
        attributes = primitive["attributes"]
        rows = zip(
            reader.accessor(attributes["POSITION"]),
            reader.accessor(attributes["TEXCOORD_0"]),
            reader.accessor(attributes["JOINTS_0"]),
            reader.accessor(attributes["WEIGHTS_0"]),
        )
        for position, uv, joint, weight in rows:
            row = round(position[1] * 100)
            upper = row == 2 or (row == 1 and abs(uv[1] - 0.5) > 0.05)
            heaviest = max(range(4), key=lambda i: weight[i])
            name = joints[joint[heaviest]].split("_", 1)[1]
            out.setdefault((row, upper), set()).add(name)
    return {
        key: names.pop() if len(names) == 1 else names for key, names in out.items()
    }


@unittest.skipUnless(
    _FBX2GLTF, "FBX2glTF is not installed; this test downloads nothing"
)
class TestSkinWeightsThroughTheConverter(unittest.TestCase):
    """A duplicated skinned mesh keeps its own weights through FBX2glTF."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="glb_skin_weights_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.fbx = build_skinned_fbx(
            os.path.join(self.tmp, "pair.fbx"),
            [("A", (0, 0, 0), False, 0.0), ("B", (3, 0, 0), True, 0.0)],
        )

    def _direct(self, fbx, *extra):
        """FBX2glTF alone, none of the passes."""
        base = os.path.join(self.tmp, "direct")
        media = os.path.join(self.tmp, "media")
        os.makedirs(media, exist_ok=True)
        subprocess.run(
            [_FBX2GLTF, "-i", fbx, "-o", base, "--binary", "--fbx-temp-dir", media]
            + list(extra),
            check=True,
            capture_output=True,
            timeout=120,
        )
        return base + ".glb"

    def test_the_converter_alone_hands_the_duplicate_the_first_meshs_weights(self):
        """The premise, pinned against the binary with none of the passes.

        Red after a converter upgrade means the restore may be retired.
        """
        glb = self._direct(self.fbx)
        self.assertEqual(row_bindings(glb, "A"), FORWARD)
        self.assertEqual(row_bindings(glb, "B"), FORWARD, "B carries A's indices")

    def test_fbx_to_glb_ships_each_mesh_its_own_weights(self):
        glb = MeshConvert.fbx_to_glb(
            self.fbx, auto_install=False, prompt=False, lightmaps=False
        )
        self.assertEqual(row_bindings(glb, "A"), FORWARD)
        self.assertEqual(row_bindings(glb, "B"), REVERSED)

    def test_audit_names_the_mesh_and_counts_its_vertices(self):
        glb = self._direct(self.fbx)
        self.assertEqual(
            GlbSkinWeights.audit(glb, self.fbx),
            {
                "compared": ["A", "B"],
                "mismatched": {"B": 4},
                "ambiguous": 0,
                "refused": [],
            },
        )

    def test_restore_rewrites_only_the_merged_mesh(self):
        glb = self._direct(self.fbx)
        summary = GlbSkinWeights.restore(glb, self.fbx)
        self.assertEqual(summary, {"meshes": 1, "vertices": 4, "ambiguous": 0})
        self.assertEqual(row_bindings(glb, "A"), FORWARD)
        self.assertEqual(row_bindings(glb, "B"), REVERSED)
        self.assertEqual(GlbSkinWeights.audit(glb, self.fbx)["mismatched"], {})

    def test_a_shared_accessor_is_replaced_for_its_one_reader(self):
        """FBX2glTF may hand two meshes one accessor; the fix must not leak."""
        glb = self._direct(self.fbx)
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
            nodes = gltf["nodes"]
            a = nodes[[n.get("name") for n in nodes].index("A")]["mesh"]
            b = nodes[[n.get("name") for n in nodes].index("B")]["mesh"]
            a_attrs = gltf["meshes"][a]["primitives"][0]["attributes"]
            b_attrs = gltf["meshes"][b]["primitives"][0]["attributes"]
            b_attrs["JOINTS_0"] = a_attrs["JOINTS_0"]
            b_attrs["WEIGHTS_0"] = a_attrs["WEIGHTS_0"]
            edit.dirty = True
        GlbSkinWeights.restore(glb, self.fbx)
        self.assertEqual(
            row_bindings(glb, "A"), FORWARD, "the other reader keeps its own"
        )
        self.assertEqual(row_bindings(glb, "B"), REVERSED)

    def test_coincident_points_are_told_apart_by_their_uvs(self):
        """An unwelded seam: two control points at one position, one on each
        joint. Position alone cannot say which GLB vertex is which; the UVs
        do, and each gets its own point's weights."""
        fbx = build_skinned_fbx(
            os.path.join(self.tmp, "split.fbx"),
            [("A", (0, 0, 0), False, 0.0), ("B", (3, 0, 0), True, 0.0)],
            split=True,
        )
        # UVs kept, as a textured material keeps them (FBX2glTF's default
        # drops a UV set no texture samples).
        glb = self._direct(fbx, "-k", "position", "-k", "normal", "-k", "uv0")
        summary = GlbSkinWeights.restore(glb, fbx)
        self.assertEqual(summary["ambiguous"], 0)
        self.assertEqual(
            seam_bindings(glb, "B"),
            {
                (0, False): "jnt_2",
                (1, False): "jnt_2",
                (1, True): "jnt_1",
                (2, True): "jnt_1",
            },
        )
        self.assertEqual(GlbSkinWeights.audit(glb, fbx)["mismatched"], {})

    def test_a_suspect_the_fbx_cannot_account_for_is_reported_refused(self):
        """Never a silent pass: a merged mesh the restore cannot map back to
        its FBX is named, not quietly left wrong."""
        glb = self._direct(self.fbx)
        with MeshConvert.open_glb(glb) as edit:
            for node in edit.gltf["nodes"]:
                if node.get("name") == "B":
                    node["name"] = "B_renamed"
            edit.dirty = True
        audit = GlbSkinWeights.audit(glb, self.fbx)
        self.assertEqual(audit["refused"], ["B_renamed"])
        with self.assertLogs(
            "pythontk.file_utils.mesh_convert.glb.skin_weights", "WARNING"
        ) as logs:
            GlbSkinWeights.restore(glb, self.fbx)
        self.assertIn("B_renamed", " ".join(logs.output))

    def test_meshes_sharing_no_position_are_left_alone_unread(self):
        """No shared position, no merge: the FBX is never parsed."""
        fbx = build_skinned_fbx(
            os.path.join(self.tmp, "apart.fbx"),
            [("A", (0, 0, 0), False, 0.0), ("B", (3, 0, 0), True, 5.0)],
        )
        glb = self._direct(fbx)
        before = open(glb, "rb").read()
        with patch(
            "pythontk.file_utils.mesh_convert.fbx_file.FbxFile.load",
            side_effect=AssertionError("the FBX was read"),
        ):
            summary = GlbSkinWeights.restore(glb, fbx)
        self.assertEqual(summary, {"meshes": 0, "vertices": 0, "ambiguous": 0})
        self.assertEqual(open(glb, "rb").read(), before)
        self.assertEqual(row_bindings(glb, "B"), REVERSED, "converted right already")


@unittest.skipUnless(
    _FBX2GLTF, "FBX2glTF is not installed; this test downloads nothing"
)
class TestSkinWeightsGate(unittest.TestCase):
    """The deliverable verifier reads the merge back off a shipped pair."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="glb_skin_gate_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _gate(self, glb, fbx):
        from pythontk.file_utils.mesh_convert.export_verify import ExportVerifier

        rows = ExportVerifier(glb=glb, fbx=fbx).check_glb_skin_weights()
        self.assertEqual(len(rows), 1)
        return rows[0]

    def _direct(self, fbx):
        base = os.path.join(self.tmp, os.path.splitext(os.path.basename(fbx))[0])
        media = os.path.join(self.tmp, "media")
        os.makedirs(media, exist_ok=True)
        subprocess.run(
            [_FBX2GLTF, "-i", fbx, "-o", base, "--binary", "--fbx-temp-dir", media],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return base + ".glb"

    def test_fails_on_the_converters_merge_and_passes_once_restored(self):
        fbx = build_skinned_fbx(
            os.path.join(self.tmp, "pair.fbx"),
            [("A", (0, 0, 0), False, 0.0), ("B", (3, 0, 0), True, 0.0)],
        )
        glb = self._direct(fbx)
        row = self._gate(glb, fbx)
        self.assertEqual(row.status, "FAIL")
        self.assertIn("B (4)", row.detail)
        GlbSkinWeights.restore(glb, fbx)
        row = self._gate(glb, fbx)
        self.assertEqual(row.status, "PASS")
        self.assertIn("2 skinned mesh(es)", row.detail, "says what it compared")

    def test_warns_on_a_suspect_it_could_not_compare(self):
        fbx = build_skinned_fbx(
            os.path.join(self.tmp, "pair.fbx"),
            [("A", (0, 0, 0), False, 0.0), ("B", (3, 0, 0), True, 0.0)],
        )
        glb = self._direct(fbx)
        with MeshConvert.open_glb(glb) as edit:
            for node in edit.gltf["nodes"]:
                if node.get("name") == "B":
                    node["name"] = "B_renamed"
            edit.dirty = True
        from pythontk.file_utils.mesh_convert.export_verify import ExportVerifier

        rows = ExportVerifier(glb=glb, fbx=fbx).check_glb_skin_weights()
        self.assertEqual([r.status for r in rows], ["WARN", "PASS"])
        self.assertIn("B_renamed", rows[0].detail)

    def test_says_when_nothing_could_be_merged(self):
        fbx = build_skinned_fbx(
            os.path.join(self.tmp, "apart.fbx"),
            [("A", (0, 0, 0), False, 0.0), ("B", (3, 0, 0), True, 5.0)],
        )
        row = self._gate(self._direct(fbx), fbx)
        self.assertEqual(row.status, "PASS")
        self.assertIn("no skinned mesh shares", row.detail)


if __name__ == "__main__":
    unittest.main()
