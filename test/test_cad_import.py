# !/usr/bin/python
# coding=utf-8
"""Unit tests for CadImport (STEP / IGES to meshes, an assembled GLB and a record).

Strategy (the optional-dependency split test_mesh_ops.py pins): the contract
that runs before OpenCascade is touched is covered unconditionally; real files
are generated with OpenCascade itself, so every expected number is exact, and
those tests skip without the ``pythontk[cad]`` extra. The deviation check also
needs ``pythontk[mesh]``.

The fixture assembly: a 60x40x10 housing with one blue face, its instance left
unnamed; four placements of one radius-2, 12-long screw; and a sub-assembly
(a pin, offset in it) placed twice, so a placement composes through two levels.
The curved fixture: a radius-10 sphere and a 10/2 torus, the doubly curved
faces whose meshes stray past the mesher's chord inside their triangles.
"""

import json
import math
import os
import shutil
import unittest

from unittest import mock

import pythontk as ptk
from conftest import gltf_validator
from pythontk.file_utils.cad_import._cad_import import CadImport
from pythontk.file_utils.cad_import.cad_record import CadRecord

_HAVE = CadImport.available()
_HERE = os.path.dirname(os.path.abspath(__file__))

SCREWS = [(5, 5), (55, 5), (5, 35), (55, 35)]
HINGES = [(20.0, 45.0, 0.0), (40.0, 45.0, 0.0)]
PIN_OFFSET = (0.0, 0.0, 3.0)


def _build_fixture(path, unit=None, iges=False):
    """Write the fixture assembly (or, for IGES, the housing alone) to *path*."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.Interface import Interface_Static
    from OCP.Quantity import Quantity_ColorRGBA
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool
    from OCP.gp import gp_Trsf, gp_Vec

    def name(label, text):
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(text))

    def at(x, y, z):
        t = gp_Trsf()
        t.SetTranslation(gp_Vec(x, y, z))
        return TopLoc_Location(t)

    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    XCAFApp_Application.GetApplication_s().InitDocument(doc)
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    colors = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    gen = XCAFDoc_ColorType.XCAFDoc_ColorGen
    box = BRepPrimAPI_MakeBox(60.0, 40.0, 10.0).Shape()
    housing = shapes.AddShape(box, False)
    name(housing, "Housing")
    colors.SetColor(housing, Quantity_ColorRGBA(0.5, 0.5, 0.55, 1.0), gen)
    face = shapes.AddSubShape(housing, TopExp_Explorer(box, TopAbs_ShapeEnum.TopAbs_FACE).Current())
    colors.SetColor(face, Quantity_ColorRGBA(0.1, 0.3, 0.9, 1.0), XCAFDoc_ColorType.XCAFDoc_ColorSurf)
    if not iges:
        top = shapes.NewShape()
        name(top, "Product")
        shapes.AddComponent(top, housing, at(0, 0, 0))  # left unnamed on purpose
        screw = shapes.AddShape(BRepPrimAPI_MakeCylinder(2.0, 12.0).Shape(), False)
        name(screw, "Screw M4")
        colors.SetColor(screw, Quantity_ColorRGBA(0.8, 0.1, 0.1, 1.0), gen)
        for i, (x, y) in enumerate(SCREWS, 1):
            name(shapes.AddComponent(top, screw, at(x, y, 10)), f"Screw_M4_{i}")
        hinge = shapes.NewShape()
        name(hinge, "Hinge")
        pin = shapes.AddShape(BRepPrimAPI_MakeCylinder(1.0, 5.0).Shape(), False)
        name(pin, "Pin")
        name(shapes.AddComponent(hinge, pin, at(*PIN_OFFSET)), "Pin_1")
        for i, xyz in enumerate(HINGES, 1):
            name(shapes.AddComponent(top, hinge, at(*xyz)), f"Hinge_{i}")
    shapes.UpdateAssemblies()
    if iges:
        from OCP.IGESCAFControl import IGESCAFControl_Writer

        writer = IGESCAFControl_Writer()
        writer.SetColorMode(True)
        writer.SetNameMode(True)
        writer.Transfer(doc)
        assert writer.Write(path)
        return path
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_StepModelType

    previous = Interface_Static.CVal_s("write.step.unit")
    if unit:
        Interface_Static.SetCVal_s("write.step.unit", unit)
    try:
        writer = STEPCAFControl_Writer()
        writer.SetColorMode(True)
        writer.SetNameMode(True)
        writer.Transfer(doc, STEPControl_StepModelType.STEPControl_AsIs)
        writer.Write(path)
    finally:
        Interface_Static.SetCVal_s("write.step.unit", previous)
    return path


def _build_curved(path):
    """Write a radius-10 sphere and a 10/2 torus to *path*."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_StepModelType
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    XCAFApp_Application.GetApplication_s().InitDocument(doc)
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    sphere = shapes.AddShape(BRepPrimAPI_MakeSphere(10.0).Shape(), False)
    TDataStd_Name.Set_s(sphere, TCollection_ExtendedString("Sphere"))
    torus = shapes.AddShape(BRepPrimAPI_MakeTorus(10.0, 2.0).Shape(), False)
    TDataStd_Name.Set_s(torus, TCollection_ExtendedString("Torus"))
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.Transfer(doc, STEPControl_StepModelType.STEPControl_AsIs)
    writer.Write(path)
    return path


def _write_shapes(path, named_shapes):
    """Write each ``(name, shape)`` as its own part to the STEP file *path*."""
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_StepModelType
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    XCAFApp_Application.GetApplication_s().InitDocument(doc)
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    for name, shape in named_shapes:
        TDataStd_Name.Set_s(shapes.AddShape(shape, False), TCollection_ExtendedString(name))
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.Transfer(doc, STEPControl_StepModelType.STEPControl_AsIs)
    writer.Write(path)
    return path


def _freeform_face():
    """A degree-3 B-spline through a 25x25 grid of a wave, 60 mm across: the
    freeform kind of face OpenCascade's global projection misreads."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.collections import Array2_gp_Pnt
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.gp import gp_Pnt

    n = 25
    grid = Array2_gp_Pnt(1, n, 1, n)
    for i in range(n):
        for j in range(n):
            x, y = -30 + 60 * i / (n - 1), -30 + 60 * j / (n - 1)
            grid.SetValue(i + 1, j + 1, gp_Pnt(x, y, 5 * math.sin(x / 2) * math.cos(y / 2)))
    return BRepBuilderAPI_MakeFace(GeomAPI_PointsToBSplineSurface(grid, 3, 3).Surface(), 1e-6).Face()


def _glb_triangles(glb, mesh_name):
    """How many triangles the GLB's mesh named *mesh_name* holds."""
    doc = _glb_json(glb)
    mesh = next(m for m in doc["meshes"] if m.get("name") == mesh_name)
    return sum(doc["accessors"][p["indices"]]["count"] // 3 for p in mesh["primitives"])


def _out_dir(name):
    path = os.path.join(_HERE, "temp_tests", f"cad_import_{name}")
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


def _load(record_path):
    with open(record_path, encoding="utf-8") as fh:
        return json.load(fh)


def _world(record, index):
    """A node's world matrix: its parents' matrices composed over its own (4x4 lists)."""
    nodes = record["nodes"]

    def mat(n):
        m = nodes[n]["matrix"]
        return [m[r * 4 : r * 4 + 4] for r in range(4)]

    def mul(a, b):
        return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]

    world = mat(index)
    parent = nodes[index]["parent"]
    while parent is not None:
        world = mul(mat(parent), world)
        parent = nodes[parent]["parent"]
    return world


def _glb_json(path):
    import struct

    with open(path, "rb") as fh:
        fh.read(12)
        size, _ = struct.unpack("<II", fh.read(8))
        return json.loads(fh.read(size))


class TestContractBeforeTheBackend(unittest.TestCase):
    """What holds whether or not OpenCascade is installed."""

    def test_install_note_names_the_extra(self):
        self.assertIn("pythontk[cad]", CadImport._install_note())

    def test_available_matches_resolve(self):
        self.assertEqual(CadImport.available(), CadImport.resolve(required=False) is not None)

    def test_ci_runs_the_conversion(self):
        # Without the backend every conversion test skips: in CI that would
        # pass by not running, so there it fails instead.
        if os.environ.get("CI"):
            self.assertTrue(_HAVE, "CI installs pythontk[cad]; " + CadImport._install_note())

    def test_a_missing_file_is_refused_before_any_work(self):
        with self.assertRaises(FileNotFoundError):
            CadImport.convert(os.path.join(_HERE, "temp_tests", "no_such_file.step"))

    def test_an_unsupported_format_is_refused_before_any_work(self):
        out = _out_dir("unsupported")
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        path = os.path.join(out, "part.sldprt")
        with open(path, "w") as fh:
            fh.write("not a neutral format")
        with self.assertRaises(ValueError) as caught:
            CadImport.convert(path)
        self.assertIn(".step", str(caught.exception))

    def test_profiles_resolve_by_name_or_by_value(self):
        standard = CadImport.profile("standard")
        self.assertEqual(standard["name"], "standard")
        custom = CadImport.profile({"name": "mine", "chord": 0.5, "angle": 20.0})
        self.assertEqual(custom["budget"], 0.5)  # the tolerance the profile states
        with self.assertRaises(ValueError):
            CadImport.profile("no_such_profile")
        with self.assertRaises(ValueError):
            CadImport.profile({"chord": 0.0, "angle": 10.0})

    def test_a_misspelt_profile_key_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            CadImport.profile({"chord": 0.1, "angle": 10.0, "budjet": 0.05})
        self.assertIn("budjet", str(caught.exception))

    def test_a_profile_name_must_be_file_safe(self):
        for bad in ("my profile", "../escape", ""):
            with self.assertRaises(ValueError, msg=bad):
                CadImport.profile({"name": bad, "chord": 0.1, "angle": 10.0})

    def test_part_ids_never_collide_ignoring_case(self):
        # "Bolt" and "bolt" are one file on a case-insensitive filesystem.
        ids = CadImport._part_ids(["Bolt", "bolt", "Screw M4", "Screw_M4"])
        self.assertEqual(len({i.casefold() for i in ids}), 4, ids)
        self.assertEqual(ids[0], "Bolt")

    def test_profile_names_must_differ_ignoring_case(self):
        # The profiles are refused before the file is read, so its content never matters.
        out = _out_dir("profile_case")
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        path = os.path.join(out, "never_read.step")
        with open(path, "w") as fh:
            fh.write("not read")
        pair = [{"name": "Fine", "chord": 0.1, "angle": 9.0}, {"name": "fine", "chord": 0.2, "angle": 9.0}]
        with self.assertRaises(ValueError):
            CadImport.convert(path, profiles=pair)

    def test_an_unknown_up_axis_is_refused(self):
        with self.assertRaises(ValueError):
            CadImport.convert(__file__, up_axis="x")

    def test_root_export(self):
        self.assertIs(ptk.CadImport, CadImport)
        self.assertIs(ptk.CadRecord, CadRecord)


@unittest.skipUnless(_HAVE, "OpenCascade (pythontk[cad]) not installed")
class TestConvert(unittest.TestCase):
    """One conversion of the fixture, read from every side."""

    @classmethod
    def setUpClass(cls):
        cls.out = _out_dir("convert")
        cls.step = _build_fixture(os.path.join(cls.out, "product.step"))
        cls.record_path = CadImport.convert(
            cls.step, os.path.join(cls.out, "converted"), profiles=("realtime", "hero")
        )
        cls.record = _load(cls.record_path)
        cls.parts = {p["name"]: p for p in cls.record["parts"]}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_the_record_validates(self):
        result = CadRecord.validate(self.record)
        self.assertEqual(result.errors, [], result.errors)
        self.assertEqual(self.record["version"], CadRecord.VERSION)

    def test_each_unique_part_is_kept_once_with_every_placement(self):
        self.assertEqual(sorted(self.parts), ["Housing", "Pin", "Screw M4"])
        leaves = [n for n in self.record["nodes"] if n["part"]]
        by_part = {}
        for n in leaves:
            by_part.setdefault(n["part"], []).append(n["name"])
        self.assertEqual(len(leaves), 7)
        self.assertEqual(sorted(by_part[self.parts["Screw M4"]["id"]]), [f"Screw_M4_{i}" for i in range(1, 5)])
        self.assertEqual(len(by_part[self.parts["Pin"]["id"]]), 2)

    def test_an_unnamed_placement_takes_its_parts_name(self):
        housing = [n for n in self.record["nodes"] if n["part"] == self.parts["Housing"]["id"]]
        self.assertEqual([n["name"] for n in housing], ["Housing"])

    def test_part_ids_are_file_safe_and_unique(self):
        self.assertEqual(self.parts["Screw M4"]["id"], "Screw_M4")
        ids = [p["id"] for p in self.record["parts"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_a_placement_composes_through_its_assemblies(self):
        pins = [i for i, n in enumerate(self.record["nodes"]) if n["part"] == self.parts["Pin"]["id"]]
        got = sorted(tuple(round(_world(self.record, i)[r][3], 6) for r in range(3)) for i in pins)
        want = sorted(tuple(h[k] + PIN_OFFSET[k] for k in range(3)) for h in HINGES)
        self.assertEqual(got, want)

    def test_the_figures_are_exact(self):
        housing, screw = self.parts["Housing"], self.parts["Screw M4"]
        self.assertEqual([round(v, 6) for v in housing["size"]], [60.0, 40.0, 10.0])
        self.assertAlmostEqual(housing["area"], 6800.0, places=6)
        self.assertAlmostEqual(housing["volume"], 24000.0, places=6)
        self.assertAlmostEqual(screw["area"], 2 * math.pi * 2 * 12 + 2 * math.pi * 4, places=6)
        self.assertAlmostEqual(screw["volume"], math.pi * 4 * 12, places=6)

    def test_colours_are_linear_and_a_face_keeps_its_own(self):
        self.assertEqual([round(c, 4) for c in self.parts["Housing"]["color"]], [0.5, 0.5, 0.55, 1.0])
        mesh = self.parts["Housing"]["meshes"][0]
        with open(os.path.join(os.path.dirname(self.record_path), mesh["path"])[:-4] + ".mtl") as fh:
            kd = sorted(line.split()[1:4] for line in fh if line.startswith("Kd "))
        self.assertEqual(len(kd), 2)  # the housing's grey and its one blue face

    def test_every_profile_writes_every_part_and_coarser_is_lighter(self):
        for part in self.parts.values():
            self.assertEqual([m["profile"] for m in part["meshes"]], ["realtime", "hero"])
            for m in part["meshes"]:
                path = os.path.join(os.path.dirname(self.record_path), m["path"])
                self.assertTrue(os.path.isfile(path), path)
                self.assertFalse(os.path.isabs(m["path"]))
        realtime, hero = self.parts["Screw M4"]["meshes"]
        self.assertLess(realtime["triangles"], hero["triangles"])
        self.assertEqual(self.parts["Housing"]["meshes"][0]["triangles"], 12)

    def test_the_obj_holds_exact_normals_and_outward_triangles(self):
        mesh = self.parts["Screw M4"]["meshes"][1]
        v, vn, faces = [], [], []
        with open(os.path.join(os.path.dirname(self.record_path), mesh["path"])) as fh:
            for line in fh:
                head, *rest = line.split()
                if head == "v":
                    v.append([float(x) for x in rest])
                elif head == "vn":
                    vn.append([float(x) for x in rest])
                elif head == "f":
                    faces.append([int(t.split("//")[0]) - 1 for t in rest])
        self.assertEqual((len(v), len(faces)), (mesh["vertices"], mesh["triangles"]))
        self.assertEqual(len(vn), len(v))
        for (x, y, _), (nx, ny, nz) in zip(v, vn):
            r = math.hypot(x, y)
            if 0.1 < r and abs(nz) < 1e-6:  # a point on the side: the normal is radial
                self.assertAlmostEqual(nx, x / r, places=5)
                self.assertAlmostEqual(ny, y / r, places=5)
        inward = 0
        for a, b, c in faces:
            pa, pb, pc = v[a], v[b], v[c]
            n = [
                (pb[1] - pa[1]) * (pc[2] - pa[2]) - (pb[2] - pa[2]) * (pc[1] - pa[1]),
                (pb[2] - pa[2]) * (pc[0] - pa[0]) - (pb[0] - pa[0]) * (pc[2] - pa[2]),
                (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]),
            ]
            centre = [(pa[k] + pb[k] + pc[k]) / 3 - (0, 0, 6)[k] for k in range(3)]
            inward += sum(n[k] * centre[k] for k in range(3)) < 0
        self.assertEqual(inward, 0)

    def test_the_assembled_glb_is_valid_and_in_metres_y_up(self):
        glb = os.path.join(os.path.dirname(self.record_path), self.record["assemblies"]["hero"])
        doc = _glb_json(glb)
        named = {n.get("name"): n for n in doc["nodes"]}
        self.assertIn("Housing", named)
        self.assertNotIn("=>", "".join(named))
        self.assertEqual([round(t, 6) for t in named["Screw_M4_1"]["translation"]], [0.005, 0.01, -0.005])
        report = ptk.GltfValidator.validate(glb, executable=gltf_validator())  # skips locally, fails in CI
        self.assertEqual(report["issues"]["numErrors"], 0, ptk.GltfValidator.messages(report))
        self.assertEqual(report["issues"]["numWarnings"], 0, ptk.GltfValidator.messages(report))

    def test_the_record_names_no_folder_of_this_machine(self):
        self.assertEqual(self.record["source"]["file"], "product.step")
        text = json.dumps(self.record)
        self.assertNotIn(os.path.dirname(self.step).replace("\\", "/"), text.replace("\\\\", "/"))
        self.assertEqual(len(self.record["source"]["sha256"]), 64)


@unittest.skipUnless(_HAVE, "OpenCascade (pythontk[cad]) not installed")
class TestSources(unittest.TestCase):
    """Units, axes and formats."""

    def setUp(self):
        self.out = _out_dir(self._testMethodName)
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)

    def test_a_file_stated_in_metres_reads_as_millimetres(self):
        step = _build_fixture(os.path.join(self.out, "metres.step"), unit="M")
        record = _load(CadImport.convert(step, self.out))
        housing = next(p for p in record["parts"] if p["name"] == "Housing")
        self.assertEqual([round(v, 6) for v in housing["size"]], [60.0, 40.0, 10.0])

    def test_the_records_unit_holds_whatever_unit_the_session_reads_in(self):
        # Another library in the process may set OpenCascade's session unit;
        # without the document's own unit this read 0.06 for 60.
        from OCP.Interface import Interface_Static

        step = _build_fixture(os.path.join(self.out, "metres.step"), unit="M")
        previous = Interface_Static.CVal_s("xstep.cascade.unit")
        Interface_Static.SetCVal_s("xstep.cascade.unit", "M")
        try:
            record = _load(CadImport.convert(step, self.out))
        finally:
            Interface_Static.SetCVal_s("xstep.cascade.unit", previous)
        housing = next(p for p in record["parts"] if p["name"] == "Housing")
        self.assertEqual([round(v, 6) for v in housing["size"]], [60.0, 40.0, 10.0])

    def test_a_y_up_source_keeps_its_axes(self):
        step = _build_fixture(os.path.join(self.out, "product.step"))
        record = _load(CadImport.convert(step, self.out, up_axis="y"))
        self.assertEqual(record["source"]["up_axis"], "y")
        named = {n.get("name"): n for n in _glb_json(os.path.join(self.out, record["assemblies"]["standard"]))["nodes"]}
        self.assertEqual([round(t, 6) for t in named["Screw_M4_1"]["translation"]], [0.005, 0.005, 0.01])

    def test_an_iges_part_reads_with_its_figures(self):
        iges = _build_fixture(os.path.join(self.out, "housing.igs"), iges=True)
        record = _load(CadImport.convert(iges, self.out))
        self.assertEqual(record["source"]["format"], "iges")
        self.assertEqual(len(record["parts"]), 1)
        self.assertEqual([round(v, 4) for v in record["parts"][0]["size"]], [60.0, 40.0, 10.0])

    def test_a_file_that_fails_to_read_leaves_nothing_behind(self):
        bad = os.path.join(self.out, "bad.step")
        with open(bad, "w") as fh:
            fh.write("ISO-10303-21;\nnot a model\n")
        with self.assertRaises(ValueError) as caught:
            CadImport.convert(bad)
        self.assertIn("AP242", str(caught.exception))  # it says how to fix it
        self.assertFalse(os.path.exists(os.path.join(self.out, "bad_cad")))


@unittest.skipUnless(_HAVE, "OpenCascade (pythontk[cad]) not installed")
class TestVerify(unittest.TestCase):
    """Each mesh measured against its part's exact surfaces, and held to its budget."""

    def setUp(self):
        self.out = _out_dir(self._testMethodName)
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)

    def _mesh(self, record, name):
        return next(p for p in record["parts"] if p["name"] == name)["meshes"][0]

    def test_every_verified_mesh_meets_its_budget_on_doubly_curved_faces(self):
        # Red before: measured against a reference mesh, the sphere's realtime
        # mesh was 0.155 from its reference against a 0.11 budget.
        step = _build_curved(os.path.join(self.out, "curved.step"))
        record = _load(CadImport.convert(step, self.out, profiles=("realtime", "standard"), verify=True))
        for part in record["parts"]:
            for mesh in part["meshes"]:
                dev = mesh["deviation"]
                self.assertTrue(dev["within"], (part["name"], mesh["profile"], dev))
                self.assertGreater(dev["peak"], 0.0)
                self.assertLessEqual(mesh["chord"], CadImport.profile(mesh["profile"])["chord"])
        # The mesher's chord alone leaves the sphere up to twice the chord out:
        # its verified meshes were made again, tighter.
        self.assertLess(self._mesh(record, "Sphere")["chord"], CadImport.profile("realtime")["chord"])

    def test_a_mesh_left_over_budget_is_recorded_and_named(self):
        step = _build_curved(os.path.join(self.out, "curved.step"))
        with mock.patch.object(CadImport, "REFINE_ROUNDS", 0):
            with self.assertLogs(CadImport.logger, level="WARNING") as logs:
                record = _load(CadImport.convert(step, self.out, profiles=("realtime",), verify=True))
        sphere = self._mesh(record, "Sphere")
        self.assertFalse(sphere["deviation"]["within"])
        self.assertGreater(sphere["deviation"]["peak"], 1.2 * sphere["deviation"]["budget"])
        self.assertTrue(any("Sphere" in line for line in logs.output), logs.output)

    def test_the_peak_is_the_exact_distance_and_lies_on_the_mesh(self):
        step = _build_curved(os.path.join(self.out, "curved.step"))
        with mock.patch.object(CadImport, "REFINE_ROUNDS", 0):
            with self.assertLogs(CadImport.logger, level="WARNING"):
                record = _load(CadImport.convert(step, self.out, profiles=("realtime",), verify=True))
        dev = self._mesh(record, "Sphere")["deviation"]
        x, y, z = dev["peak_at"]
        self.assertAlmostEqual(10.0 - math.sqrt(x * x + y * y + z * z), dev["peak"], places=9)

    def test_a_freeform_b_spline_face_measures_its_true_distance(self):
        # Red before: OpenCascade's global projection put a seventh of this
        # face's own mesh nodes up to 1.6 mm off it, so the mesh read as far
        # over budget and the refinement chased the false peak.
        step = _write_shapes(os.path.join(self.out, "freeform.step"), [("Panel", _freeform_face())])
        coarse = {"name": "coarse", "chord": 1.0, "angle": 30.0}
        record = _load(CadImport.convert(step, self.out, profiles=(coarse,), verify=True))
        mesh = record["parts"][0]["meshes"][0]
        self.assertTrue(mesh["deviation"]["within"], mesh["deviation"])
        self.assertEqual(mesh["chord"], 1.0)  # nothing to chase: no tighter tessellation
        self.assertLess(mesh["deviation"]["peak"], 1.0)

    def test_the_mesh_shares_the_glbs_frame_for_a_located_part(self):
        # A STEP round trip bakes a placement into the geometry, so the
        # location is set in the document. OpenCascade's glTF writer places a
        # part by its placements alone; the meshes the record places must sit
        # in that same frame, and still measure their true distance.
        from OCP.TopLoc import TopLoc_Location
        from OCP.gp import gp_Trsf, gp_Vec

        from pythontk.file_utils.cad_import._occ import _OccDocument

        doc = _OccDocument(_build_curved(os.path.join(self.out, "curved.step")), "step")
        part = next(p for p in doc.parts if p["name"] == "Sphere")
        t = gp_Trsf()
        t.SetTranslation(gp_Vec(100.0, 0.0, 0.0))
        part["shape"] = part["shape"].Moved(TopLoc_Location(t))
        doc._shapes.SetShape(part["label"], part["shape"])
        doc.tessellate(0.1, 15.0)
        mesh = doc.mesh(part, measure=True)
        glb = os.path.join(self.out, "located.glb")
        doc.write_glb(glb, "y")
        gltf = _glb_json(glb)
        sphere = next(m for m in gltf["meshes"] if m.get("name") == "Sphere")
        low = gltf["accessors"][sphere["primitives"][0]["attributes"]["POSITION"]]["min"][0]
        self.assertAlmostEqual(mesh["vertices"][:, 0].min() / 1000.0, low, places=6)
        self.assertLess(mesh["deviation"]["peak"], 0.5, mesh["deviation"])

    def test_a_flat_part_still_says_where_it_was_measured(self):
        step = _write_shapes(os.path.join(self.out, "box.step"), [("Box", self._box())])
        record = _load(CadImport.convert(step, self.out, profiles=("realtime",), verify=True))
        dev = record["parts"][0]["meshes"][0]["deviation"]
        self.assertAlmostEqual(dev["peak"], 0.0, places=9)
        self.assertEqual(len(dev["peak_at"]), 3)

    def test_the_glb_holds_the_refined_tessellation(self):
        step = _build_curved(os.path.join(self.out, "curved.step"))
        record = _load(CadImport.convert(step, self.out, profiles=("realtime",), verify=True))
        recorded = self._mesh(record, "Sphere")["triangles"]
        in_glb = _glb_triangles(os.path.join(self.out, record["assemblies"]["realtime"]), "Sphere")
        # The GLB keeps the two degenerate pole triangles the OBJ drops.
        self.assertLessEqual(abs(in_glb - recorded), 4, (in_glb, recorded))

    def test_refinement_that_stops_helping_keeps_the_best_mesh_and_says_so(self):
        from pythontk.file_utils.cad_import._occ import _OccDocument

        real = _OccDocument.tessellate

        def no_finer(self, chord, angle, parts=None):  # a retry changes nothing
            if parts is None:
                real(self, chord, angle)

        step = _build_curved(os.path.join(self.out, "curved.step"))
        with mock.patch.object(_OccDocument, "tessellate", no_finer):
            with self.assertLogs(CadImport.logger, level="WARNING") as logs:
                record = _load(CadImport.convert(step, self.out, profiles=("realtime",), verify=True))
        sphere = self._mesh(record, "Sphere")
        self.assertEqual(sphere["chord"], CadImport.profile("realtime")["chord"])
        self.assertFalse(sphere["deviation"]["within"])
        self.assertTrue(any("Sphere" in line and "stopped" in line for line in logs.output), logs.output)

    def test_a_face_the_mesher_leaves_empty_is_recorded_and_named(self):
        from OCP.BRep import BRep_Builder
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCP.gp import gp_Pln
        from OCP.TopoDS import TopoDS_Compound

        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        builder.Add(compound, self._box())
        builder.Add(compound, BRepBuilderAPI_MakeFace(gp_Pln(), 0.0, 10.0, 0.0, 1e-9).Face())
        step = _write_shapes(os.path.join(self.out, "sliver.step"), [("Bracket", compound)])
        with self.assertLogs(CadImport.logger, level="WARNING") as logs:
            record = _load(CadImport.convert(step, self.out, profiles=("realtime",), verify=True))
        mesh = record["parts"][0]["meshes"][0]
        self.assertEqual(mesh["unmeshed_faces"], 1, mesh)
        self.assertLess(mesh["unmeshed_area"], 1e-6)
        self.assertTrue(mesh["deviation"]["within"])  # a sliver under the budget's square cannot matter
        self.assertTrue(any("Bracket" in line and "no mesh" in line for line in logs.output), logs.output)

    def test_the_heatmap_matches_the_obj_face_for_face(self):
        # Red before: with colours alternating face by face, the OBJ grouped
        # the faces by colour while the heatmap kept the mesher's order.
        from OCP.Quantity import Quantity_ColorRGBA
        from OCP.STEPCAFControl import STEPCAFControl_Writer
        from OCP.STEPControl import STEPControl_StepModelType
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDataStd import TDataStd_Name
        from OCP.TDocStd import TDocStd_Document
        from OCP.TopAbs import TopAbs_ShapeEnum
        from OCP.TopExp import TopExp_Explorer
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool

        doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
        XCAFApp_Application.GetApplication_s().InitDocument(doc)
        shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
        colors = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
        box = self._box()
        label = shapes.AddShape(box, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString("Striped"))
        colors.SetColor(label, Quantity_ColorRGBA(0.5, 0.5, 0.5, 1.0), XCAFDoc_ColorType.XCAFDoc_ColorGen)
        exp, i = TopExp_Explorer(box, TopAbs_ShapeEnum.TopAbs_FACE), 0
        while exp.More():
            if i % 2:  # every other face red
                face = shapes.AddSubShape(label, exp.Current())
                colors.SetColor(face, Quantity_ColorRGBA(0.9, 0.1, 0.1, 1.0), XCAFDoc_ColorType.XCAFDoc_ColorSurf)
            exp.Next()
            i += 1
        writer = STEPCAFControl_Writer()
        writer.SetColorMode(True)
        writer.SetNameMode(True)
        writer.Transfer(doc, STEPControl_StepModelType.STEPControl_AsIs)
        step = os.path.join(self.out, "striped.step")
        writer.Write(step)
        record_path = CadImport.convert(step, self.out, profiles=("realtime",), verify=True, heatmaps=True)
        mesh = self._mesh(_load(record_path), "Striped")
        with open(os.path.join(self.out, mesh["path"])) as fh:
            obj = [[int(t.split("//")[0]) - 1 for t in line.split()[1:]] for line in fh if line.startswith("f ")]
        with open(os.path.join(self.out, mesh["deviation"]["heatmap"])) as fh:
            lines = fh.read().splitlines()
        body = lines[lines.index("end_header") + 1:]
        vertices, faces = body[: mesh["vertices"]], body[mesh["vertices"]:]
        self.assertEqual([[int(x) for x in f.split()[1:]] for f in faces], obj)
        self.assertEqual(len(vertices), mesh["vertices"])
        self.assertTrue(all(len(v.split()) == 6 for v in vertices))

    @staticmethod
    def _box():
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

        return BRepPrimAPI_MakeBox(20.0, 10.0, 5.0).Shape()

if __name__ == "__main__":
    unittest.main(verbosity=2)
