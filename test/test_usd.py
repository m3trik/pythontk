# !/usr/bin/python
# coding=utf-8
"""Tests for ``pythontk.file_utils.usd`` — sniffing, USDZ packaging (spec
alignment rules), usda mesh authoring, and the OBJ→USD/USDZ converters.

All zero-dep and venv-runnable. When the optional ``usd-core`` wheel (``pxr``)
is importable, authored layers/packages are additionally validated by opening
them with the real USD runtime (skipped otherwise; the blendertk suite also
cross-validates against Blender's bundled ``pxr``).
"""
import os
import shutil
import tempfile
import unittest
import zipfile

from pythontk.file_utils.usd import (
    USD_EXTENSIONS,
    UsdFile,
    UsdMeshWriter,
    UsdzPackager,
    is_usd_file,
    obj_to_usd,
    obj_to_usdz,
)

try:
    from pxr import Usd, UsdGeom  # noqa: F401 — optional deep validation

    HAS_PXR = True
except ImportError:
    HAS_PXR = False


# A unit quad: 4 points, 1 face, per-face-vertex UVs and normals.
QUAD = dict(
    points=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
    face_vertex_counts=[4],
    face_vertex_indices=[0, 1, 2, 3],
    uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
    normals=[(0, 0, 1)] * 4,
)

OBJ_TEXT = """# quad with uvs + normals
mtllib quad.mtl
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
usemtl mat0
f 1/1/1 2/2/1 3/3/1 4/4/1
"""

MTL_TEXT = """newmtl mat0
Kd 0.8 0.8 0.8
map_Kd quad_diffuse.png
map_Pr quad_rough.png
"""

# Minimal valid 1x1 PNG (89 bytes).
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da63fccfc0f01f0005050202b8bcf3ed0000000049454e44ae426082"
)


class UsdTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ptk_usd_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.tmp, name)


class TestSniffing(UsdTestCase):
    def test_usda_by_magic(self):
        p = self.path("layer.usda")
        with open(p, "w") as fh:
            fh.write("#usda 1.0\n")
        self.assertEqual(UsdFile.sniff(p), "usda")
        self.assertTrue(is_usd_file(p))

    def test_usd_extension_with_text_content_sniffs_usda(self):
        p = self.path("layer.usd")  # .usd may be text OR crate — content wins
        with open(p, "w") as fh:
            fh.write("#usda 1.0\n")
        self.assertEqual(UsdFile.sniff(p), "usda")

    def test_usdc_by_magic(self):
        p = self.path("layer.usd")
        with open(p, "wb") as fh:
            fh.write(b"PXR-USDC" + b"\x00" * 8)
        self.assertEqual(UsdFile.sniff(p), "usdc")

    def test_missing_file_falls_back_to_extension(self):
        self.assertEqual(UsdFile.sniff(self.path("nope.usdz")), "usdz")
        self.assertEqual(UsdFile.sniff(self.path("nope.usd")), "usdc")
        self.assertIsNone(UsdFile.sniff(self.path("nope.fbx")))

    def test_wrong_content_under_usd_extension_rejected(self):
        p = self.path("fake.usda")
        with open(p, "wb") as fh:
            fh.write(b"not a usd file")
        self.assertIsNone(UsdFile.sniff(p))
        self.assertFalse(is_usd_file(p))

    def test_extensions_constant(self):
        self.assertIn(".usdz", USD_EXTENSIONS)
        self.assertIn(".usda", USD_EXTENSIONS)


class TestUsdzPackager(UsdTestCase):
    def _layer(self, name="model.usda"):
        p = self.path(name)
        with open(p, "w") as fh:
            fh.write("#usda 1.0\n(\n    defaultPrim = \"M\"\n)\n")
        return p

    def _png(self, name):
        p = self.path(name)
        with open(p, "wb") as fh:
            fh.write(PNG_BYTES)
        return p

    def test_package_is_aligned_stored_and_layer_first(self):
        layer = self._layer()
        tex = self._png("t.png")
        out = UsdzPackager.package(
            [tex, layer, (self._png("t2.png"), "textures/t2.png")],
            self.path("out.usdz"),
        )
        report = UsdzPackager.verify(out)
        self.assertTrue(report["valid"], report["issues"])
        names = [e[0] for e in report["entries"]]
        self.assertEqual(names[0], "model.usda")  # layer promoted to front
        self.assertIn("textures/t2.png", names)
        for name, offset, aligned, stored in report["entries"]:
            self.assertTrue(aligned, f"{name} data at {offset}")
            self.assertTrue(stored, name)
        # Round-trip: entries extract byte-identical.
        with zipfile.ZipFile(out) as zf:
            self.assertEqual(zf.read("t.png"), PNG_BYTES)

    def test_alignment_across_many_varied_names(self):
        # Varied filename lengths exercise every padding remainder incl. the
        # 1-3 byte case (which must bump a full 64).
        files = [self._layer()]
        for i in range(12):
            files.append((self._png(f"x{'y' * i}.png"), f"tex/{'n' * (i + 1)}.png"))
        out = UsdzPackager.package(files, self.path("many.usdz"))
        report = UsdzPackager.verify(out)
        self.assertTrue(report["valid"], report["issues"])

    def test_package_appends_usdz_extension(self):
        out = UsdzPackager.package([self._layer()], self.path("noext"))
        self.assertTrue(out.endswith(".usdz"))
        self.assertTrue(os.path.isfile(out))

    def test_no_layer_raises(self):
        with self.assertRaises(ValueError):
            UsdzPackager.package([self._png("only.png")], self.path("bad.usdz"))

    def test_missing_input_raises(self):
        with self.assertRaises(FileNotFoundError):
            UsdzPackager.package([self.path("ghost.usda")], self.path("bad.usdz"))

    def test_duplicate_arcnames_raise(self):
        layer = self._layer()
        with self.assertRaises(ValueError):
            UsdzPackager.package([layer, (layer, "model.usda")], self.path("dup.usdz"))

    def test_explicit_default_layer_wins(self):
        a = self._layer("a.usda")
        b = self._layer("b.usda")
        out = UsdzPackager.package([a, b], self.path("lead.usdz"), default_layer="b.usda")
        self.assertEqual(UsdFile.default_layer(out), "b.usda")
        self.assertEqual(UsdFile.list_package(out), ["b.usda", "a.usda"])

    def test_verify_flags_a_compressed_zip(self):
        # A plain deflated zip renamed .usdz must fail verification.
        bad = self.path("bad.usdz")
        with zipfile.ZipFile(bad, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.usda", "#usda 1.0\n" * 50)
        report = UsdzPackager.verify(bad)
        self.assertFalse(report["valid"])

    def test_sniff_recognizes_package(self):
        out = UsdzPackager.package([self._layer()], self.path("s.usdz"))
        self.assertEqual(UsdFile.sniff(out), "usdz")

    def test_from_layer_packages_and_rewrites_texture_refs(self):
        tex = self._png("albedo.png")
        layer = self.path("scene.usda")
        with open(layer, "w") as fh:
            fh.write(
                "#usda 1.0\n"
                f'def Shader "t" {{ asset inputs:file = @{tex.replace(chr(92), "/")}@ }}\n'
                "def Shader \"missing\" { asset inputs:file = @/no/such/file.png@ }\n"
            )
        out = UsdzPackager.from_layer(layer, self.path("pkg.usdz"))
        report = UsdzPackager.verify(out)
        self.assertTrue(report["valid"], report["issues"])
        names = UsdFile.list_package(out)
        self.assertEqual(names[0], "scene.usda")
        self.assertIn("textures/albedo.png", names)
        with zipfile.ZipFile(out) as zf:
            text = zf.read("scene.usda").decode("utf-8")
        self.assertIn("@textures/albedo.png@", text)
        self.assertIn("@/no/such/file.png@", text)  # unresolved ref untouched

    def test_from_layer_rejects_crate(self):
        crate = self.path("bin.usdc")
        with open(crate, "wb") as fh:
            fh.write(b"PXR-USDC" + b"\x00" * 8)
        with self.assertRaises(ValueError):
            UsdzPackager.from_layer(crate, self.path("bad.usdz"))


class TestUsdMeshWriter(UsdTestCase):
    def test_writes_quad_layer(self):
        out = UsdMeshWriter.write(self.path("quad"), **QUAD)
        self.assertTrue(out.endswith(".usda"))
        text = open(out, encoding="utf-8").read()
        self.assertTrue(text.startswith("#usda 1.0"))
        self.assertIn('defaultPrim = "Model"', text)
        self.assertIn("int[] faceVertexCounts = [4]", text)
        self.assertIn("int[] faceVertexIndices = [0, 1, 2, 3]", text)
        self.assertIn("point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]", text)
        # 4 uvs == 4 points: the length tie resolves to the compacter "vertex".
        self.assertIn('interpolation = "vertex"', text)
        self.assertIn('uniform token subdivisionScheme = "none"', text)
        self.assertNotIn("material:binding", text)  # no textures given

    def test_face_varying_when_uvs_exceed_points(self):
        # Two triangles sharing an edge: 4 points, 6 face-verts -> faceVarying.
        out = UsdMeshWriter.write(
            self.path("fv"),
            points=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
            face_vertex_counts=[3, 3],
            face_vertex_indices=[0, 1, 2, 0, 2, 3],
            uvs=[(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)],
        )
        text = open(out, encoding="utf-8").read()
        self.assertIn('interpolation = "faceVarying"', text)

    def test_material_network_authored_for_textures(self):
        out = UsdMeshWriter.write(
            self.path("tex"),
            textures={"diffuse": "textures/d.png", "roughness": "textures/r.png"},
            name="Scan01",
            **QUAD,
        )
        text = open(out, encoding="utf-8").read()
        self.assertIn('defaultPrim = "Scan01"', text)
        self.assertIn("rel material:binding = </Scan01/Materials/Scan01Mat>", text)
        self.assertIn('uniform token info:id = "UsdPreviewSurface"', text)
        self.assertIn("asset inputs:file = @textures/d.png@", text)
        self.assertIn(
            "color3f inputs:diffuseColor.connect = "
            "</Scan01/Materials/Scan01Mat/diffuseTex.outputs:rgb>", text)
        self.assertIn(
            "float inputs:roughness.connect = "
            "</Scan01/Materials/Scan01Mat/roughnessTex.outputs:r>", text)
        self.assertIn('token inputs:sourceColorSpace = "raw"', text)  # data map

    def test_emissive_is_srgb_not_raw(self):
        # Emissive is a color3f (sRGB) map like diffuse; it must NOT be tagged raw,
        # else viewers double-linearize it and render emission too dark.
        out = UsdMeshWriter.write(
            self.path("emis"),
            textures={"emissive": "textures/e.png"},
            name="Scan01",
            **QUAD,
        )
        text = open(out, encoding="utf-8").read()
        emis_block = text.split('def Shader "emissiveTex"', 1)[1].split("}", 1)[0]
        self.assertNotIn('sourceColorSpace = "raw"', emis_block)

    def test_large_coordinates_keep_precision(self):
        # .6g quantized ~1e6 coords to whole units; .9g preserves sub-unit detail.
        out = UsdMeshWriter.write(
            self.path("big"),
            points=[
                (1234567.89, 0, 0), (1234568.89, 0, 0),
                (1234568.89, 1, 0), (1234567.89, 1, 0),
            ],
            face_vertex_counts=[4],
            face_vertex_indices=[0, 1, 2, 3],
        )
        text = open(out, encoding="utf-8").read()
        self.assertIn("1234567.89", text)  # not collapsed to 1.23457e+06

    def test_bad_topology_raises(self):
        with self.assertRaises(ValueError):
            UsdMeshWriter.write(
                self.path("bad"), points=[(0, 0, 0)],
                face_vertex_counts=[3], face_vertex_indices=[0, 0],  # 3 != 2
            )

    def test_illegal_prim_name_sanitized(self):
        out = UsdMeshWriter.write(self.path("n"), name="1 bad.name", **QUAD)
        text = open(out, encoding="utf-8").read()
        self.assertIn('def Xform "_1_bad_name"', text)

    @unittest.skipUnless(HAS_PXR, "usd-core not installed")
    def test_pxr_opens_authored_layer(self):
        out = UsdMeshWriter.write(
            self.path("pxr_check"), textures={"diffuse": "d.png"}, **QUAD
        )
        stage = Usd.Stage.Open(out)
        self.assertIsNotNone(stage)
        prim = stage.GetDefaultPrim()
        self.assertTrue(prim.IsValid())
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{prim.GetPath()}/Geom"))
        self.assertEqual(len(mesh.GetPointsAttr().Get()), 4)
        self.assertEqual(list(mesh.GetFaceVertexCountsAttr().Get()), [4])


class TestObjConverters(UsdTestCase):
    def _write_obj(self):
        obj = self.path("quad.obj")
        with open(obj, "w") as fh:
            fh.write(OBJ_TEXT)
        with open(self.path("quad.mtl"), "w") as fh:
            fh.write(MTL_TEXT)
        for tex in ("quad_diffuse.png", "quad_rough.png"):
            with open(self.path(tex), "wb") as fh:
                fh.write(PNG_BYTES)
        return obj

    def test_from_obj_parses_geometry_and_mtl(self):
        data = UsdMeshWriter.from_obj(self._write_obj())
        self.assertEqual(len(data["points"]), 4)
        self.assertEqual(data["face_vertex_counts"], [4])
        self.assertEqual(data["face_vertex_indices"], [0, 1, 2, 3])
        self.assertEqual(len(data["uvs"]), 4)       # faceVarying expansion
        self.assertEqual(len(data["normals"]), 4)   # vn 1 referenced 4x
        self.assertEqual(data["name"], "quad")
        self.assertEqual(
            sorted(data["textures"]), ["diffuse", "roughness"]
        )
        self.assertTrue(os.path.isabs(data["textures"]["diffuse"]))

    def test_mtl_texture_path_with_spaces(self):
        # A texture filename containing spaces must not be truncated to its last
        # whitespace token (the old .split()[-1] dropped the map entirely).
        obj = self.path("spaced.obj")
        with open(obj, "w") as fh:
            fh.write("mtllib spaced.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        with open(self.path("spaced.mtl"), "w") as fh:
            fh.write("newmtl m\nmap_Kd My Scan Albedo.png\n")
        with open(self.path("My Scan Albedo.png"), "wb") as fh:
            fh.write(PNG_BYTES)
        data = UsdMeshWriter.from_obj(obj)
        self.assertIn("diffuse", data["textures"])
        self.assertTrue(data["textures"]["diffuse"].endswith("My Scan Albedo.png"))

    def test_negative_indices(self):
        obj = self.path("neg.obj")
        with open(obj, "w") as fh:
            fh.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n")
        data = UsdMeshWriter.from_obj(obj)
        self.assertEqual(data["face_vertex_indices"], [0, 1, 2])
        self.assertIsNone(data["uvs"])
        self.assertIsNone(data["normals"])

    def test_obj_to_usd_writes_relative_texture_refs(self):
        out = obj_to_usd(self._write_obj())
        self.assertTrue(os.path.isfile(out))
        text = open(out, encoding="utf-8").read()
        self.assertIn("@quad_diffuse.png@", text)  # relative to layer dir
        self.assertNotIn("@" + self.tmp.replace("\\", "/"), text)

    def test_obj_to_usdz_is_self_contained_and_valid(self):
        out = obj_to_usdz(self._write_obj())
        self.assertTrue(out.endswith(".usdz"))
        report = UsdzPackager.verify(out)
        self.assertTrue(report["valid"], report["issues"])
        names = UsdFile.list_package(out)
        self.assertEqual(names[0], "quad.usda")
        self.assertIn("textures/quad_diffuse.png", names)
        self.assertIn("textures/quad_rough.png", names)
        # The layer references the in-package texture paths.
        with zipfile.ZipFile(out) as zf:
            layer = zf.read("quad.usda").decode("utf-8")
        self.assertIn("@textures/quad_diffuse.png@", layer)

    def test_usdz_same_basename_textures_deduped(self):
        # Two different source files sharing a basename must land as distinct
        # package entries (textures/t.png + textures/t_1.png), not collide.
        obj = self.path("col.obj")
        for sub in ("sub1", "sub2"):
            os.makedirs(self.path(sub))
            with open(self.path(os.path.join(sub, "t.png")), "wb") as fh:
                fh.write(PNG_BYTES)
        with open(obj, "w") as fh:
            fh.write("mtllib col.mtl\n" + OBJ_TEXT.split("\n", 2)[2])
        with open(self.path("col.mtl"), "w") as fh:
            fh.write("newmtl m\nmap_Kd sub1/t.png\nmap_Pr sub2/t.png\n")
        out = obj_to_usdz(obj)
        names = UsdFile.list_package(out)
        self.assertIn("textures/t.png", names)
        self.assertIn("textures/t_1.png", names)
        self.assertTrue(UsdzPackager.verify(out)["valid"])

    @unittest.skipUnless(HAS_PXR, "usd-core not installed")
    def test_pxr_opens_authored_usdz(self):
        out = obj_to_usdz(self._write_obj())
        stage = Usd.Stage.Open(out)
        self.assertIsNotNone(stage)
        self.assertTrue(stage.GetDefaultPrim().IsValid())


class TestRootRegistration(unittest.TestCase):
    def test_symbols_resolve_from_package_root(self):
        import pythontk as ptk

        self.assertIs(ptk.UsdzPackager, UsdzPackager)
        self.assertIs(ptk.UsdMeshWriter, UsdMeshWriter)
        self.assertIs(ptk.UsdFile, UsdFile)
        self.assertIs(ptk.is_usd_file, is_usd_file)
        self.assertIs(ptk.obj_to_usdz, obj_to_usdz)
        self.assertEqual(ptk.USD_EXTENSIONS, USD_EXTENSIONS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
