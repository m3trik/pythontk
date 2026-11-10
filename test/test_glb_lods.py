# !/usr/bin/python
# coding=utf-8
"""Tests for ``GlbLods``: a GLB states each LOD group in ``MSFT_lod``.

The FBX names a group's levels ``<base>_LOD0..n`` (engines read them); FBX2glTF
writes them as sibling nodes, which every glTF reader would draw at once. The
pass gives the LOD0 node ``extensions.MSFT_lod.ids`` (the lower levels, taken
out of the node tree) and ``extras.MSFT_screencoverage`` (one switch height per
level, the cull last), and lists the extension in ``extensionsUsed`` only, so
a reader without it draws LOD0 alone (the spec's fallback). Measured
2026-11-10: the Khronos validator passes such a file with 0 errors and 0
warnings.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from pythontk.file_utils.lod_chain import LodChain  # noqa: E402
from pythontk.file_utils.mesh_convert._fbx2gltf import _Fbx2Gltf  # noqa: E402
from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia  # noqa: E402
from pythontk.file_utils.mesh_convert.glb.lods import GlbLods  # noqa: E402
from pythontk.file_utils.temp_artifacts import TempArtifacts  # noqa: E402


def _gltf(triangles=(300, 120, 40)):
    """RootNode -> props -> chair (chair_LOD0..2) + lamp, as FBX2glTF writes a
    DCC's LOD group: each level a node with its own mesh."""
    accessors, meshes = [], []
    for count in (*triangles, 12):
        accessors.append({"count": count * 3, "type": "SCALAR", "componentType": 5125})
        meshes.append({"primitives": [{"attributes": {}, "indices": len(accessors) - 1}]})
    nodes = [
        {"name": "RootNode", "children": [1]},
        {"name": "props", "children": [2, 6]},
        {"name": "chair", "children": [3, 4, 5]},
        {"name": "chair_LOD0", "mesh": 0},
        {"name": "chair_LOD1", "mesh": 1},
        {"name": "chair_LOD2", "mesh": 2},
        {"name": "lamp", "mesh": 3},
    ]
    return {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
    }


class ApplyTest(unittest.TestCase):
    def test_a_group_is_stated_in_msft_lod(self):
        gltf = _gltf()
        report = GlbLods.apply(gltf)
        self.assertEqual((report["groups"], report["levels"]), (1, 3))
        nodes = gltf["nodes"]
        self.assertEqual(nodes[2]["children"], [3], "the lower levels leave the tree")
        self.assertEqual(nodes[3]["extensions"]["MSFT_lod"], {"ids": [4, 5]})
        self.assertEqual(nodes[3]["extras"]["MSFT_screencoverage"], [0.5, 0.25, 0.0])
        self.assertEqual(gltf["extensionsUsed"], ["MSFT_lod"])
        self.assertNotIn("extensionsRequired", gltf, "a reader without it draws LOD0")

    def test_the_heights_come_from_the_source_when_it_states_them(self):
        gltf = _gltf()
        GlbLods.apply(gltf, heights={"chair": [0.5, 0.2, 0.01]})
        self.assertEqual(gltf["nodes"][3]["extras"]["MSFT_screencoverage"], [0.5, 0.2, 0.01])

    def test_heights_for_another_level_count_fall_back_to_the_ladder(self):
        gltf = _gltf()
        GlbLods.apply(gltf, heights={"chair": [0.5, 0.0]})
        self.assertEqual(gltf["nodes"][3]["extras"]["MSFT_screencoverage"], [0.5, 0.25, 0.0])

    def test_a_malformed_group_is_left_as_the_fbx_wrote_it(self):
        """Triangles that do not fall: the group ships as plain nodes, as the
        Scene Exporter warned (``prepare_lods``)."""
        gltf = _gltf(triangles=(300, 300, 40))
        report = GlbLods.apply(gltf)
        self.assertEqual(report["groups"], 0)
        self.assertEqual(gltf["nodes"][2]["children"], [3, 4, 5])
        self.assertNotIn("extensionsUsed", gltf)

    def test_a_group_deep_in_a_tree_is_stated(self):
        """3000 nodes deep: the paths are walked up, not recursed (2026-11-10
        review: past the interpreter's recursion limit the pass raised)."""
        depth = 3000
        nodes = [{"name": f"n{i}", "children": [i + 1]} for i in range(depth)]
        nodes += [{"name": "chair", "children": [depth + 1, depth + 2]}, {"name": "chair_LOD0"}, {"name": "chair_LOD1"}]
        self.assertEqual(GlbLods.apply({"nodes": nodes, "scenes": [{"nodes": [0]}]})["names"], ["chair"])

    def test_a_second_run_changes_nothing(self):
        gltf = _gltf()
        GlbLods.apply(gltf)
        before = repr(gltf)
        self.assertEqual(GlbLods.apply(gltf)["groups"], 0)
        self.assertEqual(repr(gltf), before)

    def test_no_group_no_change(self):
        gltf = _gltf()
        gltf["nodes"][4]["name"] = "chair_back"
        gltf["nodes"][5]["name"] = "chair_seat"
        gltf["nodes"][3]["name"] = "chair_legs"
        before = repr(gltf)
        self.assertEqual(GlbLods.apply(gltf)["groups"], 0)
        self.assertEqual(repr(gltf), before)


class ReadersTest(unittest.TestCase):
    """Readers of a GLB's hierarchy see a lower level where it stands."""

    def test_ids_naming_an_ancestor_make_no_cycle(self):
        """A malformed file's ``ids`` naming the node's own ancestor: no level
        gains it as a child, and a walk up the tree ends (2026-11-10 review: it
        looped forever)."""
        nodes = [{"name": "root", "children": [1]}, {"name": "x", "extensions": {"MSFT_lod": {"ids": [0, 1]}}}]
        self.assertEqual(GlbLods.parents(nodes), {1: 0})

    def test_a_lower_level_answers_its_lod0s_parent(self):
        gltf = _gltf()
        GlbLods.apply(gltf)
        parents = GlbLods.parents(gltf["nodes"])
        self.assertEqual((parents[3], parents[4], parents[5]), (2, 2, 2))

    def test_the_fingerprint_keys_the_levels_under_their_group(self):
        import json
        import struct

        from pythontk.file_utils.mesh_convert.export_fingerprint import ExportFingerprint

        gltf = _gltf()
        GlbLods.apply(gltf)
        temp = TempArtifacts("test_glb_lods_fp", policy="scoped")
        self.addCleanup(temp.cleanup)
        path = os.path.join(temp.dir_path(), "lods.glb")
        payload = json.dumps(gltf).encode()
        payload += b" " * ((4 - len(payload) % 4) % 4)
        with open(path, "wb") as fh:
            fh.write(struct.pack("<4sII", b"glTF", 2, 20 + len(payload)))
            fh.write(struct.pack("<I4s", len(payload), b"JSON") + payload)
        nodes = ExportFingerprint.of_glb(path)["nodes"]
        self.assertIn("RootNode/props/chair/chair_LOD1", nodes)
        self.assertEqual(
            nodes["RootNode/props/chair/chair_LOD0"]["lod"],
            ["RootNode/props/chair/chair_LOD1", "RootNode/props/chair/chair_LOD2"],
        )


class VerifyTest(unittest.TestCase):
    """``ExportVerifier.check_glb_lods``: a written GLB's MSFT_lod reads as the
    spec reads it."""

    def _verify(self, gltf):
        import json
        import struct

        from pythontk.file_utils.mesh_convert.export_verify import ExportVerifier

        temp = TempArtifacts("test_glb_lods_verify", policy="scoped")
        self.addCleanup(temp.cleanup)
        path = os.path.join(temp.dir_path(), "lods.glb")
        payload = json.dumps(gltf).encode()
        payload += b" " * ((4 - len(payload) % 4) % 4)
        with open(path, "wb") as fh:
            fh.write(struct.pack("<4sII", b"glTF", 2, 20 + len(payload)))
            fh.write(struct.pack("<I4s", len(payload), b"JSON") + payload)
        (finding,) = ExportVerifier(glb=path, sidecar=None).check_glb_lods()
        return finding

    def test_the_pass_output_passes(self):
        gltf = _gltf()
        GlbLods.apply(gltf)
        finding = self._verify(gltf)
        self.assertEqual((finding.status, finding.detail), ("PASS", "1 LOD group(s)"))

    def test_a_level_left_in_the_tree_fails(self):
        gltf = _gltf()
        GlbLods.apply(gltf)
        gltf["nodes"][2]["children"].append(4)
        finding = self._verify(gltf)
        self.assertEqual(finding.status, "FAIL")
        self.assertIn("still in the node tree", finding.detail)

    def test_heights_that_do_not_fall_fail(self):
        gltf = _gltf()
        GlbLods.apply(gltf)
        gltf["nodes"][3]["extras"]["MSFT_screencoverage"] = [0.2, 0.5, 0.0]
        self.assertEqual(self._verify(gltf).status, "FAIL")

    def test_no_group_passes_naming_none(self):
        finding = self._verify(_gltf())
        self.assertEqual((finding.status, finding.detail), ("PASS", "no LOD group"))


class RemoveNodesTest(unittest.TestCase):
    """A pass that removes nodes after the LOD pass renumbers ``MSFT_lod`` too."""

    def test_the_ids_follow_a_renumber(self):
        gltf = _gltf()
        GlbLods.apply(gltf)
        _Fbx2Gltf._remove_glb_nodes(None, gltf, {6})  # the lamp
        lod0 = next(n for n in gltf["nodes"] if n["name"] == "chair_LOD0")
        names = [gltf["nodes"][i]["name"] for i in lod0["extensions"]["MSFT_lod"]["ids"]]
        self.assertEqual(names, ["chair_LOD1", "chair_LOD2"])

    def test_a_removed_level_takes_its_id_and_its_height(self):
        gltf = _gltf()
        GlbLods.apply(gltf, heights={"chair": [0.5, 0.2, 0.01]})
        _Fbx2Gltf._remove_glb_nodes(None, gltf, {4})  # chair_LOD1
        lod0 = next(n for n in gltf["nodes"] if n["name"] == "chair_LOD0")
        names = [gltf["nodes"][i]["name"] for i in lod0["extensions"]["MSFT_lod"]["ids"]]
        self.assertEqual(names, ["chair_LOD2"])
        self.assertEqual(lod0["extras"]["MSFT_screencoverage"], [0.5, 0.01])

    def test_every_lower_level_removed_takes_the_extension(self):
        gltf = _gltf()
        GlbLods.apply(gltf)
        _Fbx2Gltf._remove_glb_nodes(None, gltf, {4, 5})
        lod0 = next(n for n in gltf["nodes"] if n["name"] == "chair_LOD0")
        self.assertNotIn("MSFT_lod", lod0.get("extensions", {}))
        self.assertNotIn("MSFT_screencoverage", lod0.get("extras", {}))

    def test_a_removed_last_level_leaves_the_group_its_cull(self):
        """The level kept last draws down to the group's cull, not to the
        removed level's switch (the object would vanish early)."""
        gltf = _gltf()
        GlbLods.apply(gltf, heights={"chair": [0.5, 0.2, 0.01]})
        _Fbx2Gltf._remove_glb_nodes(None, gltf, {5})  # chair_LOD2
        lod0 = next(n for n in gltf["nodes"] if n["name"] == "chair_LOD0")
        self.assertEqual(lod0["extras"]["MSFT_screencoverage"], [0.5, 0.01])


class ExpandTest(unittest.TestCase):
    """``GlbLods.expand``, the way back: every ``MSFT_lod`` group becomes the
    convention's node tree, ``<base>`` over ``<base>_LOD0..n``, for a writer
    that holds levels as nodes (the GLB -> FBX conversion)."""

    @staticmethod
    def _names(gltf, indices):
        return [gltf["nodes"][i]["name"] for i in indices]

    def test_our_own_group_takes_its_levels_back_under_it(self):
        gltf = _gltf()
        GlbLods.apply(gltf, heights={"chair": [0.6, 0.3, 0.0]})
        groups = GlbLods.expand(gltf)
        chair = gltf["nodes"][2]
        self.assertEqual(self._names(gltf, chair["children"]), ["chair_LOD0", "chair_LOD1", "chair_LOD2"])
        self.assertEqual(groups, {2: [0.6, 0.3, 0.0]})
        self.assertNotIn("MSFT_lod", gltf["nodes"][3].get("extensions", {}))
        self.assertNotIn("MSFT_screencoverage", gltf["nodes"][3].get("extras", {}))
        self.assertEqual(len(gltf["nodes"]), 7, "no node added")

    def test_a_foreign_group_gets_a_group_node_and_the_level_names(self):
        """A node another writer gave MSFT_lod (any names, a parent holding
        other nodes): a ``<base>`` node at identity takes its place, the
        levels renamed ``<base>_LOD<n>``; no heights, the default ladder."""
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [
                {"name": "room", "children": [1, 4]},
                {"name": "chair", "translation": [1, 0, 0], "extensions": {"MSFT_lod": {"ids": [2, 3]}}},
                {"name": "chair_low", "translation": [1, 0, 0]},
                {"name": "chair_lowest", "translation": [1, 0, 0]},
                {"name": "table"},
            ],
            "extensionsUsed": ["MSFT_lod"],
        }
        groups = GlbLods.expand(gltf)
        (group,) = groups
        self.assertEqual(gltf["nodes"][group]["name"], "chair")
        self.assertNotIn("translation", gltf["nodes"][group])
        self.assertEqual(gltf["nodes"][0]["children"], [group, 4], "in the LOD0's place")
        self.assertEqual(
            self._names(gltf, gltf["nodes"][group]["children"]), ["chair_LOD0", "chair_LOD1", "chair_LOD2"]
        )
        self.assertEqual(gltf["nodes"][2]["translation"], [1, 0, 0], "each level keeps its own transform")
        self.assertEqual(groups[group], LodChain.default_heights(3))
        self.assertNotIn("MSFT_lod", gltf.get("extensionsUsed", []))

    def test_a_group_at_the_scene_root_replaces_its_lod0_there(self):
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [
                {"name": "rock_LOD0", "extensions": {"MSFT_lod": {"ids": [1]}}, "extras": {"MSFT_screencoverage": [0.4, 0.05]}},
                {"name": "rock_LOD1"},
            ],
        }
        groups = GlbLods.expand(gltf)
        (group,) = groups
        self.assertEqual(gltf["scenes"][0]["nodes"], [group])
        self.assertEqual(gltf["nodes"][group]["name"], "rock")
        self.assertEqual(groups[group], [0.4, 0.05])

    def test_ids_out_of_range_or_in_the_tree_are_left(self):
        gltf = _gltf()
        gltf["nodes"][3]["extensions"] = {"MSFT_lod": {"ids": [6, 99]}}  # the lamp is in the tree
        self.assertEqual(GlbLods.expand(gltf), {})
        self.assertEqual(gltf["nodes"][2]["children"], [3, 4, 5])

    def test_no_group_no_change(self):
        gltf = _gltf()
        self.assertEqual(GlbLods.expand(gltf), {})
        self.assertEqual(gltf, _gltf())

    def test_a_group_past_the_level_limit_is_left_as_written(self):
        """MSFT_lod sets no limit; the convention holds 8 levels. A group of 9
        stays as the file has it (LOD0 drawn, the rest out of the tree) and
        the conversion goes on."""
        nodes = [{"name": "rock", "children": [1]}, {"name": "rock_LOD0", "extensions": {"MSFT_lod": {"ids": list(range(2, 10))}}}]
        nodes += [{"name": f"rock_LOD{k}"} for k in range(1, 9)]
        gltf = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes}
        before = repr(gltf)
        self.assertEqual(GlbLods.expand(gltf), {})
        self.assertEqual(repr(gltf), before)


class MergeTest(unittest.TestCase):
    """``GlbLods.merge``: several GLBs of one node tree, finest first (a CAD
    import's profiles: each a tessellation of the exact surfaces), become one
    GLB whose every mesh node is a LOD group stated in ``MSFT_lod``."""

    def setUp(self):
        self.temp = TempArtifacts("test_glb_lods_merge", policy="scoped")
        self.addCleanup(self.temp.cleanup)
        self.dir = self.temp.dir_path()

    def _level(self, name, bolt, plate):
        """``asm`` -> two ``bolt`` instances (one mesh) and a ``plate``: each
        mesh a fan of that many triangles."""
        from test_gltf2fbx import GltfBuilder

        b = GltfBuilder()
        meshes = []
        for count in (bolt, plate):
            ring = [(0.0, 0.0, 0.0)] + [
                (float(i), float(count), 0.5 * i) for i in range(count + 1)
            ]
            fan = [k for i in range(1, count + 1) for k in (0, i, i + 1)]
            meshes.append(
                {
                    "primitives": [
                        {
                            "attributes": {"POSITION": b.accessor(ring, "VEC3")},
                            "indices": b.accessor(fan, "SCALAR", 5125),
                            "material": 0,
                        }
                    ]
                }
            )
        b.gltf.update(
            scene=0,
            scenes=[{"nodes": [0]}],
            nodes=[
                {"name": "asm", "children": [1, 2, 3]},
                {"name": "bolt", "mesh": 0, "translation": [0.0, 0.0, 1.0]},
                {"name": "bolt", "mesh": 0, "translation": [0.0, 0.0, 2.0]},
                {"name": "plate", "mesh": 1},
            ],
            meshes=meshes,
            materials=[{"name": "steel", "pbrMetallicRoughness": {"metallicFactor": 1.0}}],
        )
        return b.write_glb(os.path.join(self.dir, f"{name}.glb"))

    @staticmethod
    def _read(path):
        from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit

        return GlbEdit.read(path)

    def _merged(self):
        levels = [
            self._level("fine", 300, 50),
            self._level("mid", 120, 50),  # the plate's mesher floor: no fewer
            self._level("coarse", 40, 20),
        ]
        dst = os.path.join(self.dir, "lods.glb")
        report = GlbLods.merge(levels, dst)
        return dst, report

    def _levels_of(self, gltf, group):
        lod0 = gltf["nodes"][group]["children"][0]
        ids = gltf["nodes"][lod0]["extensions"]["MSFT_lod"]["ids"]
        return [lod0] + ids

    def test_every_mesh_and_accessor_the_merge_writes_is_drawn(self):
        """A coarser profile drawing a part with no fewer triangles adds no
        level, and none of its geometry (2026-11-10 review: dead meshes)."""
        dst, _ = self._merged()
        gltf = self._read(dst).gltf
        used = {n["mesh"] for n in gltf["nodes"] if "mesh" in n}
        self.assertEqual(used, set(range(len(gltf["meshes"]))))
        read = {a for m in gltf["meshes"] for p in m["primitives"] for a in [*p["attributes"].values(), p.get("indices")] if a is not None}
        self.assertEqual(read, set(range(len(gltf["accessors"]))))

    def test_a_copied_materials_extension_is_declared(self):
        """``graft_meshes`` lists a copied material's extension in
        ``extensionsUsed``, as its source did: the Khronos validator fails one
        used and not listed (2026-11-10 review)."""
        from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit

        fine, coarse = self._level("fine", 300, 50), self._level("coarse", 40, 20)
        with GlbEdit.open(coarse) as edit:
            edit.gltf["materials"][0]["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": 2.0}}
            edit.gltf["extensionsUsed"] = ["KHR_materials_emissive_strength"]
            edit.dirty = True
        dst = os.path.join(self.dir, "ext.glb")
        GlbLods.merge([fine, coarse], dst)
        self.assertIn("KHR_materials_emissive_strength", self._read(dst).gltf.get("extensionsUsed") or [])

    def test_every_mesh_node_becomes_a_group_of_its_levels(self):
        dst, report = self._merged()
        gltf = self._read(dst).gltf
        bolts = [i for i, n in enumerate(gltf["nodes"]) if n.get("name") == "bolt"]
        self.assertEqual(len(bolts), 2)
        for group in bolts:
            node = gltf["nodes"][group]
            self.assertNotIn("mesh", node, "the group holds its levels alone")
            levels = self._levels_of(gltf, group)
            self.assertEqual(
                [gltf["nodes"][i]["name"] for i in levels],
                ["bolt_LOD0", "bolt_LOD1", "bolt_LOD2"],
            )
            self.assertEqual(
                [GlbLods._triangles(gltf, gltf["nodes"][i]) for i in levels], [300, 120, 40]
            )
        self.assertEqual(
            [gltf["nodes"][i]["mesh"] for i in self._levels_of(gltf, bolts[0])],
            [gltf["nodes"][i]["mesh"] for i in self._levels_of(gltf, bolts[1])],
            "instances share each level's mesh",
        )
        self.assertEqual(report["groups"], 3)
        self.assertIn("MSFT_lod", gltf["extensionsUsed"])

    def test_a_level_with_no_fewer_triangles_is_left_out_for_that_node(self):
        dst, _ = self._merged()
        gltf = self._read(dst).gltf
        plate = next(i for i, n in enumerate(gltf["nodes"]) if n.get("name") == "plate")
        levels = self._levels_of(gltf, plate)
        self.assertEqual([gltf["nodes"][i]["name"] for i in levels], ["plate_LOD0", "plate_LOD1"])
        self.assertEqual([GlbLods._triangles(gltf, gltf["nodes"][i]) for i in levels], [50, 20])
        lod0 = gltf["nodes"][levels[0]]
        self.assertEqual(lod0["extras"]["MSFT_screencoverage"], LodChain.default_heights(2))

    def test_the_grafted_geometry_is_the_sources(self):
        from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit

        dst, _ = self._merged()
        merged = self._read(dst)
        coarse = self._read(os.path.join(self.dir, "coarse.glb"))
        bolt = next(i for i, n in enumerate(merged.gltf["nodes"]) if n.get("name") == "bolt")
        lod2 = self._levels_of(merged.gltf, bolt)[2]
        prim = merged.gltf["meshes"][merged.gltf["nodes"][lod2]["mesh"]]["primitives"][0]
        source = coarse.gltf["meshes"][0]["primitives"][0]
        for key in ("POSITION",):
            self.assertEqual(
                GlbEdit.accessor_elements(merged, prim["attributes"][key]),
                GlbEdit.accessor_elements(coarse, source["attributes"][key]),
            )
        self.assertEqual(
            GlbEdit.accessor_elements(merged, prim["indices"]),
            GlbEdit.accessor_elements(coarse, source["indices"]),
        )
        self.assertEqual(len(merged.gltf["materials"]), 1, "an equal material is shared")

    def test_trees_that_differ_are_refused(self):
        from test_gltf2fbx import GltfBuilder

        other = GltfBuilder()
        other.gltf.update(scene=0, scenes=[{"nodes": [0]}], nodes=[{"name": "else"}])
        stranger = other.write_glb(os.path.join(self.dir, "stranger.glb"))
        with self.assertRaises(ValueError):
            GlbLods.merge([self._level("fine", 300, 50), stranger], os.path.join(self.dir, "x.glb"))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "x.glb")), "a refusal writes nothing")

    def test_a_textured_level_is_refused_and_writes_nothing(self):
        """``GlbEdit.graft_meshes`` copies no image: a textured material is
        refused whole rather than copied half."""
        import json
        import struct

        fine = self._level("fine", 300, 50)
        coarse = self._level("coarse", 40, 20)
        with open(coarse, "rb") as fh:
            data = fh.read()
        size = struct.unpack("<I", data[12:16])[0]
        doc = json.loads(data[20 : 20 + size])
        doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
        payload = json.dumps(doc).encode("utf-8")
        payload += b" " * ((4 - len(payload) % 4) % 4)
        rest = data[20 + size :]
        with open(coarse, "wb") as fh:
            fh.write(struct.pack("<4sII", b"glTF", 2, 20 + len(payload) + len(rest)))
            fh.write(struct.pack("<I4s", len(payload), b"JSON") + payload + rest)
        dst = os.path.join(self.dir, "x.glb")
        with self.assertRaises(ValueError):
            GlbLods.merge([fine, coarse], dst)
        self.assertFalse(os.path.exists(dst))

    def test_a_skinned_mesh_node_is_left_at_one_level(self):
        """A skinned node's levels would need its skin on each: it keeps its
        one mesh, named in the report."""
        from test_gltf2fbx import GltfBuilder  # noqa: F401 -- the levels' writer

        paths = []
        for name, bolt, plate in (("fine", 300, 50), ("coarse", 40, 20)):
            path = self._level(name, bolt, plate)
            paths.append(path)
        from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit

        with GlbEdit.open(paths[0]) as edit:
            edit.gltf["nodes"][3]["skin"] = 0  # the plate
            edit.gltf["skins"] = [{"joints": [0]}]
            edit.dirty = True
        report = GlbLods.merge(paths, os.path.join(self.dir, "lods.glb"))
        gltf = self._read(os.path.join(self.dir, "lods.glb")).gltf
        plate = next(n for n in gltf["nodes"] if n.get("name") == "plate")
        self.assertEqual(plate.get("skin"), 0)
        self.assertIn("mesh", plate)
        self.assertTrue(any(s.startswith("plate:") for s in report["skipped"]), report["skipped"])

    def test_the_merged_glb_converts_to_fbx_lod_groups(self):
        """Through the GLB -> FBX conversion every group is a LodGroup, and a
        second instance's name takes its numbered tail on the group, its
        levels named after it (``bolt_001_LOD0``: still a level name)."""
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

        dst, _ = self._merged()
        fbx = MeshConvert.gltf_to_fbx(dst, os.path.join(self.dir, "lods.fbx"), overwrite=True)
        heights = FbxMedia.lod_heights(fbx)
        self.assertEqual(
            heights,
            {
                "bolt": LodChain.default_heights(3),
                "bolt_001": LodChain.default_heights(3),
                "plate": LodChain.default_heights(2),
            },
        )
        parsed = FbxFile.load(fbx)
        models = {
            parsed.object_name(r) for r in parsed.iter_objects() if r["name"] == "Model"
        }
        for group, count in (("bolt", 3), ("bolt_001", 3), ("plate", 2)):
            for k in range(count):
                self.assertIn(LodChain.level_name(group, k), models)


class FbxHeightsTest(unittest.TestCase):
    """The switch heights an FBX states on its LodGroup attributes (a DCC's
    writer: ``ThresholdsUsedAsPercentage`` and ``Thresholds|Level<i>``,
    measured), keyed by the group node's name."""

    def test_the_lod_groups_thresholds_read_as_heights(self):
        from test_fbx_media import I32, _write_fbx  # noqa: F401 -- the raw writer

        temp = TempArtifacts("test_glb_lods", policy="scoped")
        self.addCleanup(temp.cleanup)
        path = os.path.join(temp.dir_path(), "lod.fbx")

        def p(*values):
            return ("P", list(values), [])

        objects = [
            ("Model", [10, b"chair\x00\x01Model", b"LodGroup"], [("Version", [I32(232)], [])]),
            (
                "NodeAttribute",
                [20, b"\x00\x01NodeAttribute", b"LodGroup"],
                [
                    (
                        "Properties70",
                        [],
                        [
                            p(b"ThresholdsUsedAsPercentage", b"bool", b"", b"", I32(1)),
                            p(b"Thresholds|Level0", b"Distance", b"", b"", 50.0, b"cm"),
                            p(b"Thresholds|Level1", b"Distance", b"", b"", 20.0, b"cm"),
                        ],
                    )
                ],
            ),
            ("Model", [11, b"lamp\x00\x01Model", b"Mesh"], [("Version", [I32(232)], [])]),
        ]
        connections = [("C", [b"OO", 10, 0], []), ("C", [b"OO", 20, 10], []), ("C", [b"OO", 11, 0], [])]
        _write_fbx(path, [("Objects", [], objects), ("Connections", [], connections)])
        self.assertEqual(FbxMedia.lod_heights(path), {"chair": [0.5, 0.2, 0.0]})

    def test_a_group_switching_by_distance_states_no_height(self):
        """``ThresholdsUsedAsPercentage`` off: the thresholds are distances,
        which no screen height states, so the group is left out (its readers
        take the default ladder)."""
        from test_fbx_media import I32, _write_fbx  # noqa: F401 -- the raw writer

        temp = TempArtifacts("test_glb_lods", policy="scoped")
        self.addCleanup(temp.cleanup)
        path = os.path.join(temp.dir_path(), "distance.fbx")
        objects = [
            ("Model", [10, b"chair\x00\x01Model", b"LodGroup"], [("Version", [I32(232)], [])]),
            ("NodeAttribute", [20, b"\x00\x01NodeAttribute", b"LodGroup"], [("Properties70", [], [
                ("P", [b"ThresholdsUsedAsPercentage", b"bool", b"", b"", I32(0)], []),
                ("P", [b"Thresholds|Level0", b"Distance", b"", b"", 400.0, b"cm"], []),
            ])]),
        ]
        connections = [("C", [b"OO", 10, 0], []), ("C", [b"OO", 20, 10], [])]
        _write_fbx(path, [("Objects", [], objects), ("Connections", [], connections)])
        self.assertEqual(FbxMedia.lod_heights(path), {})

    def test_a_lod_group_written_without_thresholds_gains_them(self):
        """``FbxMedia.set_lod_thresholds``: a writer that states the LodGroup
        type alone (an empty ``Properties70``) gets the thresholds as a DCC's
        own LodGroup writer states them; one already stating its own keeps them."""
        from test_fbx_media import I32, _write_fbx  # noqa: F401 -- the raw writer



        temp = TempArtifacts("test_glb_lods", policy="scoped")
        self.addCleanup(temp.cleanup)
        path = os.path.join(temp.dir_path(), "bare.fbx")
        objects = [
            ("Model", [10, b"chair\x00\x01Model", b"LodGroup"], [("Version", [I32(232)], [])]),
            ("NodeAttribute", [20, b"chair\x00\x01NodeAttribute", b"LodGroup"],
             [("TypeFlags", [b"Null"], []), ("Properties70", [], [])]),
            ("Model", [11, b"sofa\x00\x01Model", b"LodGroup"], [("Version", [I32(232)], [])]),
            ("NodeAttribute", [21, b"sofa\x00\x01NodeAttribute", b"LodGroup"], [("Properties70", [], [
                ("P", [b"ThresholdsUsedAsPercentage", b"bool", b"", b"", I32(1)], []),
                ("P", [b"Thresholds|Level0", b"Distance", b"", b"", 40.0, b"cm"], []),
            ])]),
        ]
        connections = [("C", [b"OO", 10, 0], []), ("C", [b"OO", 20, 10], []),
                       ("C", [b"OO", 11, 0], []), ("C", [b"OO", 21, 11], [])]
        _write_fbx(path, [("Objects", [], objects), ("Connections", [], connections)])
        stated = FbxMedia.set_lod_thresholds(path, {"chair": [0.5, 0.2, 0.0], "sofa": [0.6, 0.0]})
        self.assertEqual(stated, ["chair"])
        self.assertEqual(FbxMedia.lod_heights(path), {"chair": [0.5, 0.2, 0.0], "sofa": [0.4, 0.0]})
        self.assertEqual(FbxMedia.set_lod_thresholds(path, {"chair": [0.5, 0.2, 0.0]}), [], "a second run changes nothing")
        # Read strictly: the block the writer left empty, with no NULL
        # sentinel, closes on one now -- a strict reader (a DCC's FBX importer)
        # refused the file otherwise, and a lenient one took its last row for
        # the sentinel (2026-11-10 end-to-end run).
        import mmap

        with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            _version, roots, _fid, _magic = FbxMedia._load(buf)
            objects = next(r for r in roots if r.name == b"Objects")
            chair = next(r for r in objects.children if r.name == b"NodeAttribute" and b"chair" in bytes(r.props))
            block = next(c for c in chair.children if c.name == b"Properties70")
            self.assertEqual(len(block.children), 6, [bytes(c.props)[:40] for c in block.children])
            self.assertIn(block.tail, (b"\0" * 13, b"\0" * 25), "the NULL sentinel of either header width")


@unittest.skipUnless(__import__("importlib").util.find_spec("pymeshlab"), "needs pythontk[mesh]")
class LodDeviationTest(unittest.TestCase):
    """``ExportVerifier.check_glb_lod_deviation``: each level of a written GLB
    measured against its LOD0, both in world space (``MeshOps.compare``), so a
    level that lost its shape is said before a viewer sees it pop."""

    def _glb(self, lift=0.0):
        from test_gltf2fbx import GltfBuilder

        temp = TempArtifacts("test_glb_lods_deviation", policy="scoped")
        self.addCleanup(temp.cleanup)
        b = GltfBuilder()
        meshes = []
        for cells, z in ((8, 0.0), (4, 0.0), (2, lift)):
            points = [(x / cells, y / cells, z) for y in range(cells + 1) for x in range(cells + 1)]
            quads = [(y * (cells + 1) + x) for y in range(cells) for x in range(cells)]
            faces = [k for q in quads for k in (q, q + 1, q + cells + 2, q, q + cells + 2, q + cells + 1)]
            meshes.append({"primitives": [{"attributes": {"POSITION": b.accessor(points, "VEC3")},
                                           "indices": b.accessor(faces, "SCALAR", 5125)}]})
        b.gltf.update(
            scene=0,
            scenes=[{"nodes": [0]}],
            nodes=[
                {"name": "chair", "children": [1, 2, 3], "translation": [10.0, 0.0, 0.0]},
                {"name": "chair_LOD0", "mesh": 0},
                {"name": "chair_LOD1", "mesh": 1},
                {"name": "chair_LOD2", "mesh": 2},
            ],
            meshes=meshes,
        )
        path = b.write_glb(os.path.join(temp.dir_path(), "lods.glb"))
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        MeshConvert.apply_glb_lods(path)
        return path

    @staticmethod
    def _check(path):
        from pythontk.file_utils.mesh_convert.export_verify import ExportVerifier

        return ExportVerifier(glb=path, sidecar=None).check_glb_lod_deviation()

    def test_levels_of_one_shape_pass_with_their_deviation(self):
        (finding,) = self._check(self._glb())
        self.assertEqual(finding.status, "PASS", finding.detail)
        self.assertIn("chair_LOD2", finding.detail)

    def test_the_measurement_is_bounded_and_says_what_it_left(self):
        """Past the group or triangle cap a group is named "not measured": a
        production scene's verification stays seconds (2026-11-10 review)."""
        from pythontk.file_utils.mesh_convert.export_verify import ExportVerifier

        verifier = ExportVerifier(glb=self._glb(), sidecar=None)
        verifier.LOD_DEVIATION_MAX_TRIANGLES = 10  # LOD0 draws 128
        (finding,) = verifier.check_glb_lod_deviation()
        self.assertEqual(finding.status, "SKIP", finding.detail)
        self.assertIn("not measured (past", finding.detail)
        self.assertIn("chair_LOD0", finding.detail)

    def test_a_level_that_lost_its_shape_warns_naming_it(self):
        (finding,) = self._check(self._glb(lift=0.5))
        self.assertEqual(finding.status, "WARN", finding.detail)
        self.assertIn("chair_LOD2", finding.detail)
        self.assertNotIn("chair_LOD1", finding.detail.split(";")[0])


class CheckFbxLodsTest(unittest.TestCase):
    """``ExportVerifier.check_fbx_lods``: each group an FBX names as levels is
    one every reader takes as the exporter meant it -- a LodGroup attribute
    stating one threshold per switch (DCCs and engines that read it), every
    level written visible (an engine's import visibility disables a hidden
    one inside its group), and the ``lods`` record (the engine consumer's
    groups and heights) naming the same groups, levels and heights."""

    def _fbx(self, thresholds=(50.0, 25.0), hidden=(), record=None, attribute=True):
        from test_fbx_media import I32, _write_fbx  # noqa: F401 -- the raw writer

        temp = TempArtifacts("test_glb_lods_fbx_verify", policy="scoped")
        self.addCleanup(temp.cleanup)
        path = os.path.join(temp.dir_path(), "lods.fbx")

        def p(*values):
            return ("P", list(values), [])

        def model(uid, name, kind=b"Mesh", props=()):
            children = [("Version", [I32(232)], [])]
            if props:
                children.append(("Properties70", [], list(props)))
            return ("Model", [uid, name.encode() + b"\x00\x01Model", kind], children)

        objects = [model(10, "chair", b"LodGroup" if attribute else b"Null")]
        connections = [("C", [b"OO", 10, 0], [])]
        if attribute:
            levels = [p(b"ThresholdsUsedAsPercentage", b"bool", b"", b"", I32(1))] + [
                p(f"Thresholds|Level{i}".encode(), b"Distance", b"", b"", float(t), b"cm")
                for i, t in enumerate(thresholds)
            ]
            objects.append(("NodeAttribute", [20, b"\x00\x01NodeAttribute", b"LodGroup"], [("Properties70", [], levels)]))
            connections.append(("C", [b"OO", 20, 10], []))
        for index in range(3):
            props = [p(b"Visibility", b"Visibility", b"", b"A", 0.0)] if index in hidden else []
            objects.append(model(11 + index, f"chair_LOD{index}", props=props))
            connections.append(("C", [b"OO", 11 + index, 10], []))
        if record is not None:
            payload = json.dumps(record).encode()
            objects.append(model(30, "data_export", b"Null", [p(b"lods", b"KString", b"", b"U", payload)]))
            connections.append(("C", [b"OO", 30, 0], []))
        _write_fbx(path, [("Objects", [], objects), ("Connections", [], connections)])
        return path

    @staticmethod
    def _check(path):
        from pythontk.file_utils.mesh_convert.export_verify import ExportVerifier

        return ExportVerifier(fbx=path, sidecar=None).check_fbx_lods()

    @staticmethod
    def _record(heights=(0.5, 0.25, 0.0), name="chair", levels=3):
        return {"version": 1, "groups": [{"name": name, "levels": [f"chair_LOD{i}" for i in range(levels)], "heights": list(heights)}]}

    def test_a_whole_group_passes(self):
        (finding,) = self._check(self._fbx(record=self._record()))
        self.assertEqual(finding.status, "PASS", finding.detail)
        self.assertIn("1 LOD group", finding.detail)

    def test_a_group_with_no_lod_group_attribute_warns(self):
        """An engine reads the names; a DCC reads plain meshes."""
        (finding,) = self._check(self._fbx(attribute=False))
        self.assertEqual(finding.status, "WARN", finding.detail)

    def test_thresholds_that_miss_a_switch_fail(self):
        (finding,) = self._check(self._fbx(thresholds=(50.0,)))
        self.assertEqual(finding.status, "FAIL", finding.detail)

    def test_a_hidden_level_fails(self):
        (finding,) = self._check(self._fbx(hidden=(1,)))
        self.assertEqual(finding.status, "FAIL", finding.detail)
        self.assertIn("chair_LOD1", finding.detail)

    def test_a_record_naming_a_group_the_file_lacks_fails(self):
        (finding,) = self._check(self._fbx(record=self._record(name="sofa")))
        self.assertEqual(finding.status, "FAIL", finding.detail)

    def test_record_heights_unlike_the_attributes_warn(self):
        """Each reader would switch the group at its own heights."""
        (finding,) = self._check(self._fbx(record=self._record(heights=(0.6, 0.3, 0.0))))
        self.assertEqual(finding.status, "WARN", finding.detail)

    def test_an_fbx_with_no_group_passes_naming_none(self):
        from test_fbx_media import I32, _write_fbx  # noqa: F401

        temp = TempArtifacts("test_glb_lods_fbx_verify", policy="scoped")
        self.addCleanup(temp.cleanup)
        path = os.path.join(temp.dir_path(), "plain.fbx")
        _write_fbx(path, [("Objects", [], [("Model", [11, b"box\x00\x01Model", b"Mesh"], [])]),
                          ("Connections", [], [("C", [b"OO", 11, 0], [])])])
        (finding,) = self._check(path)
        self.assertEqual((finding.status, finding.detail), ("PASS", "no LOD group"))


if __name__ == "__main__":
    unittest.main()
