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

import contextlib
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
LOOSE_GAP = 0.05


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
    face = shapes.AddSubShape(
        housing, TopExp_Explorer(box, TopAbs_ShapeEnum.TopAbs_FACE).Current()
    )
    colors.SetColor(
        face,
        Quantity_ColorRGBA(0.1, 0.3, 0.9, 1.0),
        XCAFDoc_ColorType.XCAFDoc_ColorSurf,
    )
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


def _write_shapes(
    path, named_shapes, properties=None, product_ids=None, ap242=False, unit=None
):
    """Write each ``(name, shape)`` as its own part to the STEP file *path*.

    *properties* maps a part's name to its user-defined attributes (strings,
    floats, ints), which OpenCascade writes as STEP properties; *product_ids*
    maps a name to the part number its STEP product states (OpenCascade writes
    the name there, so the file is edited after). *ap242* writes AP242, the
    schema that holds a face stated as triangles alone.
    """
    from OCP.Interface import Interface_Static
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_StepModelType
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name, TDataStd_NamedData
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    XCAFApp_Application.GetApplication_s().InitDocument(doc)
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    for name, shape in named_shapes:
        label = shapes.AddShape(shape, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))
        for key, value in (properties or {}).get(name, {}).items():
            data = TDataStd_NamedData.Set_s(label)
            k = TCollection_ExtendedString(key)
            if isinstance(value, str):
                data.SetString(k, TCollection_ExtendedString(value))
            elif isinstance(value, int):
                data.SetInteger(k, value)
            else:
                data.SetReal(k, value)
    STEPCAFControl_Writer()  # registers the STEP settings read below
    settings = {
        "write.step.schema": "AP242DIS" if ap242 else None,
        "write.step.unit": unit,
    }
    previous = {key: Interface_Static.CVal_s(key) for key in settings}
    for key, value in settings.items():
        if value:
            Interface_Static.SetCVal_s(key, value)
    try:
        writer = STEPCAFControl_Writer()
        writer.SetNameMode(True)
        writer.Transfer(doc, STEPControl_StepModelType.STEPControl_AsIs)
        writer.Write(path)
    finally:
        for key, value in previous.items():
            Interface_Static.SetCVal_s(key, value)
    if product_ids:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for name, number in product_ids.items():
            text = text.replace(
                f"PRODUCT('{name}','{name}'", f"PRODUCT('{number}','{name}'"
            )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return path


def _l_bracket(mirrored=False):
    """An L-shaped plate 20x15, 2 thick, or its mirror image across the YZ plane:
    the same figures but for which way the L turns."""
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_Transform,
    )
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

    outline = BRepBuilderAPI_MakePolygon()
    for x, y in [(1, 1), (21, 1), (21, 6), (6, 6), (6, 16), (1, 16)]:
        outline.Add(gp_Pnt(x, y, 0))
    outline.Close()
    solid = BRepPrimAPI_MakePrism(
        BRepBuilderAPI_MakeFace(outline.Wire()).Face(), gp_Vec(0, 0, 2)
    ).Shape()
    if not mirrored:
        return solid
    mirror = gp_Trsf()
    mirror.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)))
    return BRepBuilderAPI_Transform(solid, mirror, True).Shape()


def _loose_face():
    """A 20x10 plane on z=0 whose boundary edges lie 0.05 mm above it: the kind
    of face real exports hold, an edge stated within its tolerance of the
    surface, not on it (a STEP read states the tolerance)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.BRepLib import BRepLib
    from OCP.gp import gp_Pln, gp_Pnt

    h = LOOSE_GAP
    square = BRepBuilderAPI_MakePolygon(
        gp_Pnt(0, 0, h), gp_Pnt(20, 0, h), gp_Pnt(20, 10, h), gp_Pnt(0, 10, h), True
    )
    face = BRepBuilderAPI_MakeFace(gp_Pln(), square.Wire(), True).Face()
    BRepLib.SameParameter_s(face, 1e-7, True)
    return face


def _triangles_only():
    """A 10x10 square stated as two triangles and no surface: AP242's tessellated geometry."""
    from OCP.BRep import BRep_Builder
    from OCP.gp import gp_Pnt
    from OCP.Poly import Poly_Triangle, Poly_Triangulation
    from OCP.TopoDS import TopoDS_Face, TopoDS_Shell

    tri = Poly_Triangulation(4, 2, False)
    for i, (x, y) in enumerate([(0, 0), (10, 0), (10, 10), (0, 10)], 1):
        tri.SetNode(i, gp_Pnt(x, y, 0))
    tri.SetTriangle(1, Poly_Triangle(1, 2, 3))
    tri.SetTriangle(2, Poly_Triangle(1, 3, 4))
    face, shell, builder = TopoDS_Face(), TopoDS_Shell(), BRep_Builder()
    builder.MakeFace(face, tri)
    builder.MakeShell(shell)
    builder.Add(shell, face)
    return shell


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
            grid.SetValue(
                i + 1, j + 1, gp_Pnt(x, y, 5 * math.sin(x / 2) * math.cos(y / 2))
            )
    return BRepBuilderAPI_MakeFace(
        GeomAPI_PointsToBSplineSurface(grid, 3, 3).Surface(), 1e-6
    ).Face()


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
        return [
            [sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)]
            for r in range(4)
        ]

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
        self.assertEqual(
            CadImport.available(), CadImport.resolve(required=False) is not None
        )

    def test_ci_runs_the_conversion(self):
        # Without the backend every conversion test skips: in CI that would
        # pass by not running, so there it fails instead.
        if os.environ.get("CI"):
            self.assertTrue(
                _HAVE, "CI installs pythontk[cad]; " + CadImport._install_note()
            )

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

    def test_part_ids_keep_a_part_numbers_dashes_dots_and_letters(self):
        # Red before for the accent: "Café Bracket" became "Caf_Bracket".
        ids = CadImport._part_ids(
            ["HX-100.2", "Café Bracket", "a/b:c", "..hidden.", "///"]
        )
        self.assertEqual(ids, ["HX-100.2", "Café_Bracket", "a_b_c", "hidden", "Part"])

    def test_profile_names_must_differ_ignoring_case(self):
        # The profiles are refused before the file is read, so its content never matters.
        out = _out_dir("profile_case")
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        path = os.path.join(out, "never_read.step")
        with open(path, "w") as fh:
            fh.write("not read")
        pair = [
            {"name": "Fine", "chord": 0.1, "angle": 9.0},
            {"name": "fine", "chord": 0.2, "angle": 9.0},
        ]
        with self.assertRaises(ValueError):
            CadImport.convert(path, profiles=pair)

    def test_an_unknown_up_axis_is_refused(self):
        with self.assertRaises(ValueError):
            CadImport.convert(__file__, up_axis="x")

    def test_a_part_id_is_keys_or_callables_refused_before_any_work(self):
        out = _out_dir("part_id_contract")
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        path = os.path.join(out, "never_read.step")
        with open(path, "w") as fh:
            fh.write("not read")
        for bad, error in (
            (42, TypeError),
            (["name", 3], TypeError),
            ([], ValueError),
            ("", TypeError),
        ):
            with self.assertRaises(error, msg=repr(bad)):
                CadImport.convert(path, part_id=bad)

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
        self.assertEqual(
            sorted(by_part[self.parts["Screw M4"]["id"]]),
            [f"Screw_M4_{i}" for i in range(1, 5)],
        )
        self.assertEqual(len(by_part[self.parts["Pin"]["id"]]), 2)

    def test_an_unnamed_placement_takes_its_parts_name(self):
        housing = [
            n for n in self.record["nodes"] if n["part"] == self.parts["Housing"]["id"]
        ]
        self.assertEqual([n["name"] for n in housing], ["Housing"])

    def test_part_ids_are_file_safe_and_unique(self):
        self.assertEqual(self.parts["Screw M4"]["id"], "Screw_M4")
        ids = [p["id"] for p in self.record["parts"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_a_placement_composes_through_its_assemblies(self):
        pins = [
            i
            for i, n in enumerate(self.record["nodes"])
            if n["part"] == self.parts["Pin"]["id"]
        ]
        got = sorted(
            tuple(round(_world(self.record, i)[r][3], 6) for r in range(3))
            for i in pins
        )
        want = sorted(tuple(h[k] + PIN_OFFSET[k] for k in range(3)) for h in HINGES)
        self.assertEqual(got, want)

    def test_the_figures_are_exact(self):
        housing, screw = self.parts["Housing"], self.parts["Screw M4"]
        self.assertEqual([round(v, 6) for v in housing["size"]], [60.0, 40.0, 10.0])
        self.assertAlmostEqual(housing["area"], 6800.0, places=6)
        self.assertAlmostEqual(housing["volume"], 24000.0, places=6)
        self.assertAlmostEqual(
            screw["area"], 2 * math.pi * 2 * 12 + 2 * math.pi * 4, places=6
        )
        self.assertAlmostEqual(screw["volume"], math.pi * 4 * 12, places=6)

    def test_colours_are_linear_and_a_face_keeps_its_own(self):
        self.assertEqual(
            [round(c, 4) for c in self.parts["Housing"]["color"]], [0.5, 0.5, 0.55, 1.0]
        )
        mesh = self.parts["Housing"]["meshes"][0]
        with open(
            os.path.join(os.path.dirname(self.record_path), mesh["path"])[:-4] + ".mtl"
        ) as fh:
            kd = sorted(line.split()[1:4] for line in fh if line.startswith("Kd "))
        self.assertEqual(len(kd), 2)  # the housing's grey and its one blue face

    def test_every_profile_writes_every_part_and_coarser_is_lighter(self):
        for part in self.parts.values():
            self.assertEqual(
                [m["profile"] for m in part["meshes"]], ["realtime", "hero"]
            )
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
        glb = os.path.join(
            os.path.dirname(self.record_path), self.record["assemblies"]["hero"]
        )
        doc = _glb_json(glb)
        named = {n.get("name"): n for n in doc["nodes"]}
        self.assertIn("Housing", named)
        self.assertNotIn("=>", "".join(named))
        self.assertEqual(
            [round(t, 6) for t in named["Screw_M4_1"]["translation"]],
            [0.005, 0.01, -0.005],
        )
        report = ptk.GltfValidator.validate(
            glb, executable=gltf_validator()
        )  # skips locally, fails in CI
        self.assertEqual(
            report["issues"]["numErrors"], 0, ptk.GltfValidator.messages(report)
        )
        self.assertEqual(
            report["issues"]["numWarnings"], 0, ptk.GltfValidator.messages(report)
        )

    def test_the_profiles_merge_into_one_valid_lod_glb(self):
        """A CAD import's profiles are its LOD levels: ``MeshConvert.merge_glb_lods``
        takes the record's assemblies finest chord first (given here coarse
        first) into one GLB whose parts are ``MSFT_lod`` groups, levels named
        ``<part>_LOD<n>``, and the Khronos validator passes it."""
        glb = ptk.MeshConvert.merge_glb_lods(
            self.record_path, os.path.join(self.out, "lods", "product_lods.glb")
        )
        doc = _glb_json(glb)
        lod0s = [n for n in doc["nodes"] if "MSFT_lod" in (n.get("extensions") or {})]
        self.assertTrue(lod0s, "the curved parts draw fewer triangles at realtime")
        for lod0 in lod0s:
            self.assertTrue(lod0["name"].endswith("_LOD0"), lod0["name"])
            (lower,) = lod0["extensions"]["MSFT_lod"]["ids"]
            self.assertEqual(doc["nodes"][lower]["name"], lod0["name"][:-1] + "1")
        report = ptk.GltfValidator.validate(glb, executable=gltf_validator())
        self.assertEqual(report["issues"]["numErrors"], 0, ptk.GltfValidator.messages(report))
        self.assertEqual(report["issues"]["numWarnings"], 0, ptk.GltfValidator.messages(report))

    def test_the_record_names_no_folder_of_this_machine(self):
        self.assertEqual(self.record["source"]["file"], "product.step")
        text = json.dumps(self.record)
        self.assertNotIn(
            os.path.dirname(self.step).replace("\\", "/"), text.replace("\\\\", "/")
        )
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
        named = {
            n.get("name"): n
            for n in _glb_json(
                os.path.join(self.out, record["assemblies"]["standard"])
            )["nodes"]
        }
        self.assertEqual(
            [round(t, 6) for t in named["Screw_M4_1"]["translation"]],
            [0.005, 0.005, 0.01],
        )

    def test_an_iges_part_reads_with_its_figures(self):
        iges = _build_fixture(os.path.join(self.out, "housing.igs"), iges=True)
        record = _load(CadImport.convert(iges, self.out))
        self.assertEqual(record["source"]["format"], "iges")
        self.assertEqual(len(record["parts"]), 1)
        self.assertEqual(
            [round(v, 4) for v in record["parts"][0]["size"]], [60.0, 40.0, 10.0]
        )

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
        record = _load(
            CadImport.convert(
                step, self.out, profiles=("realtime", "standard"), verify=True
            )
        )
        for part in record["parts"]:
            for mesh in part["meshes"]:
                dev = mesh["deviation"]
                self.assertTrue(dev["within"], (part["name"], mesh["profile"], dev))
                self.assertGreater(dev["peak"], 0.0)
                self.assertLessEqual(
                    mesh["chord"], CadImport.profile(mesh["profile"])["chord"]
                )
        # The mesher's chord alone leaves the sphere up to twice the chord out:
        # its verified meshes were made again, tighter.
        self.assertLess(
            self._mesh(record, "Sphere")["chord"],
            CadImport.profile("realtime")["chord"],
        )

    def test_a_mesh_left_over_budget_is_recorded_and_named(self):
        step = _build_curved(os.path.join(self.out, "curved.step"))
        with mock.patch.object(CadImport, "REFINE_ROUNDS", 0):
            with self.assertLogs(CadImport.logger, level="WARNING") as logs:
                record = _load(
                    CadImport.convert(
                        step, self.out, profiles=("realtime",), verify=True
                    )
                )
        sphere = self._mesh(record, "Sphere")
        self.assertFalse(sphere["deviation"]["within"])
        self.assertGreater(
            sphere["deviation"]["peak"], 1.2 * sphere["deviation"]["budget"]
        )
        self.assertTrue(any("Sphere" in line for line in logs.output), logs.output)

    def test_the_peak_is_the_exact_distance_and_lies_on_the_mesh(self):
        step = _build_curved(os.path.join(self.out, "curved.step"))
        with mock.patch.object(CadImport, "REFINE_ROUNDS", 0):
            with self.assertLogs(CadImport.logger, level="WARNING"):
                record = _load(
                    CadImport.convert(
                        step, self.out, profiles=("realtime",), verify=True
                    )
                )
        dev = self._mesh(record, "Sphere")["deviation"]
        x, y, z = dev["peak_at"]
        self.assertAlmostEqual(
            10.0 - math.sqrt(x * x + y * y + z * z), dev["peak"], places=9
        )

    def test_a_freeform_b_spline_face_measures_its_true_distance(self):
        # Red before: OpenCascade's global projection put a seventh of this
        # face's own mesh nodes up to 1.6 mm off it, so the mesh read as far
        # over budget and the refinement chased the false peak.
        step = _write_shapes(
            os.path.join(self.out, "freeform.step"), [("Panel", _freeform_face())]
        )
        coarse = {"name": "coarse", "chord": 1.0, "angle": 30.0}
        record = _load(
            CadImport.convert(step, self.out, profiles=(coarse,), verify=True)
        )
        mesh = record["parts"][0]["meshes"][0]
        self.assertTrue(mesh["deviation"]["within"], mesh["deviation"])
        self.assertEqual(
            mesh["chord"], 1.0
        )  # nothing to chase: no tighter tessellation
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
        low = gltf["accessors"][sphere["primitives"][0]["attributes"]["POSITION"]][
            "min"
        ][0]
        self.assertAlmostEqual(mesh["vertices"][:, 0].min() / 1000.0, low, places=6)
        self.assertLess(mesh["deviation"]["peak"], 0.5, mesh["deviation"])

    def test_a_flat_part_still_says_where_it_was_measured(self):
        step = _write_shapes(os.path.join(self.out, "box.step"), [("Box", self._box())])
        record = _load(
            CadImport.convert(step, self.out, profiles=("realtime",), verify=True)
        )
        dev = record["parts"][0]["meshes"][0]["deviation"]
        self.assertAlmostEqual(dev["peak"], 0.0, places=9)
        self.assertEqual(len(dev["peak_at"]), 3)

    def test_the_glb_holds_the_refined_tessellation(self):
        step = _build_curved(os.path.join(self.out, "curved.step"))
        record = _load(
            CadImport.convert(step, self.out, profiles=("realtime",), verify=True)
        )
        recorded = self._mesh(record, "Sphere")["triangles"]
        in_glb = _glb_triangles(
            os.path.join(self.out, record["assemblies"]["realtime"]), "Sphere"
        )
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
                record = _load(
                    CadImport.convert(
                        step, self.out, profiles=("realtime",), verify=True
                    )
                )
        sphere = self._mesh(record, "Sphere")
        self.assertEqual(sphere["chord"], CadImport.profile("realtime")["chord"])
        self.assertFalse(sphere["deviation"]["within"])
        self.assertTrue(
            any("Sphere" in line and "stopped helping" in line for line in logs.output),
            logs.output,
        )

    def _scripted(self, peaks, **settings):
        """Convert the curved fixture at realtime (budget 0.1) with the sphere's
        measured peaks read from *peaks* in turn (the meshes stay real) and
        *settings* patched onto CadImport. Returns the record, the warnings
        naming the sphere, and each measured mesh's triangle count."""
        from pythontk.file_utils.cad_import._occ import _OccDocument

        real, calls = _OccDocument.mesh, []

        def scripted(doc, part, measure=False):
            mesh = real(doc, part, measure)
            if measure and part["name"] == "Sphere":
                mesh["deviation"]["peak"] = peaks[min(len(calls), len(peaks) - 1)]
                calls.append(len(mesh["triangles"]))
            return mesh

        step = _build_curved(os.path.join(self.out, "curved.step"))
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(_OccDocument, "mesh", scripted))
            for name, value in settings.items():
                stack.enter_context(mock.patch.object(CadImport, name, value))
            logs = stack.enter_context(
                self.assertLogs(CadImport.logger, level="WARNING")
            )
            record = _load(
                CadImport.convert(step, self.out, profiles=("realtime",), verify=True)
            )
        return record, [line for line in logs.output if "Sphere" in line], calls

    def test_a_worse_last_round_leaves_the_best_mesh_in_the_glb(self):
        # Review: deleting the restore failed no test. The second round is the
        # best; the two after it are worse, so the loop stops on them and the
        # GLB must hold the second round's tessellation, not the last one's.
        record, warned, calls = self._scripted([0.2, 0.12, 0.3, 0.3])
        sphere = self._mesh(record, "Sphere")
        self.assertEqual(len(calls), 4)
        self.assertEqual(sphere["deviation"]["peak"], 0.12)
        self.assertAlmostEqual(sphere["chord"], 0.04)  # 0.1 x 0.8 x 0.1 / 0.2
        self.assertEqual(sphere["triangles"], calls[1])
        in_glb = _glb_triangles(
            os.path.join(self.out, record["assemblies"]["realtime"]), "Sphere"
        )
        self.assertLessEqual(abs(in_glb - sphere["triangles"]), 4, (in_glb, calls))
        self.assertTrue(any("stopped helping" in line for line in warned), warned)

    def test_rounds_spent_while_gaining_are_told_apart_from_a_stall(self):
        record, warned, calls = self._scripted([0.2, 0.16, 0.13], REFINE_ROUNDS=2)
        self.assertEqual(len(calls), 3)
        self.assertEqual(self._mesh(record, "Sphere")["deviation"]["peak"], 0.13)
        self.assertTrue(
            any("after 2 tighter tessellations" in line for line in warned), warned
        )
        self.assertFalse(any("stopped helping" in line for line in warned), warned)

    def test_heatmaps_without_verify_say_why_none_is_written(self):
        # Review: red before, nothing written and nothing said.
        step = _write_shapes(os.path.join(self.out, "box.step"), [("Box", self._box())])
        with self.assertLogs(CadImport.logger, level="WARNING") as logs:
            record = _load(
                CadImport.convert(step, self.out, profiles=("realtime",), heatmaps=True)
            )
        self.assertNotIn("deviation", record["parts"][0]["meshes"][0])
        self.assertTrue(
            any("heatmaps" in line and "verify" in line for line in logs.output),
            logs.output,
        )

    def test_a_mesh_past_its_triangle_ceiling_stops_the_refinement(self):
        # Review: four rounds at a quarter of the chord each could multiply
        # the triangles 256-fold. Every round gains here, so only the ceiling
        # stops it.
        _, warned, calls = self._scripted([0.2, 0.15, 0.12, 0.11], REFINE_MAX_GROWTH=1)
        self.assertEqual(len(calls), 2)
        self.assertTrue(
            any("times the first one's triangles" in line for line in warned), warned
        )

    def test_a_round_that_gains_resets_the_patience(self):
        # A stall, a gain, then two stalls: the loop stops on the second stall
        # after the gain, not on the second stall overall.
        _, warned, calls = self._scripted([0.2, 0.199, 0.15, 0.149, 0.148])
        self.assertEqual(len(calls), 5)
        self.assertTrue(any("stopped helping" in line for line in warned), warned)

    def test_one_worse_round_never_stops_the_refinement(self):
        # OpenCascade's mesher is not monotonic in its chord: on a NIST AP242
        # B-spline part a 0.0088 chord measured 0.045 where 0.02 had measured
        # 0.036, and 0.0032 then measured 0.0098. Red before: one worse round
        # stopped the loop, the part over budget.
        from pythontk.file_utils.cad_import._occ import _OccDocument

        real, calls = _OccDocument.mesh, []

        def worse_once(doc, part, measure=False):
            mesh = real(doc, part, measure)
            if measure and part["name"] == "Sphere":
                calls.append(mesh["deviation"]["peak"])
                if len(calls) == 2:  # the first tighter round reads three times worse
                    mesh["deviation"]["peak"] *= 3.0
            return mesh

        step = _build_curved(os.path.join(self.out, "curved.step"))
        with mock.patch.object(_OccDocument, "mesh", worse_once):
            record = _load(
                CadImport.convert(step, self.out, profiles=("realtime",), verify=True)
            )
        sphere = self._mesh(record, "Sphere")
        self.assertTrue(sphere["deviation"]["within"], (sphere, calls))
        self.assertGreaterEqual(len(calls), 3)

    def test_a_face_the_mesher_leaves_empty_is_recorded_and_named(self):
        from OCP.BRep import BRep_Builder
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCP.gp import gp_Pln
        from OCP.TopoDS import TopoDS_Compound

        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        builder.Add(compound, self._box())
        builder.Add(
            compound, BRepBuilderAPI_MakeFace(gp_Pln(), 0.0, 10.0, 0.0, 1e-9).Face()
        )
        step = _write_shapes(
            os.path.join(self.out, "sliver.step"), [("Bracket", compound)]
        )
        with self.assertLogs(CadImport.logger, level="WARNING") as logs:
            record = _load(
                CadImport.convert(step, self.out, profiles=("realtime",), verify=True)
            )
        mesh = record["parts"][0]["meshes"][0]
        self.assertEqual(mesh["unmeshed_faces"], 1, mesh)
        self.assertLess(mesh["unmeshed_area"], 1e-6)
        self.assertTrue(
            mesh["deviation"]["within"]
        )  # a sliver under the budget's square cannot matter
        self.assertTrue(
            any("Bracket" in line and "no mesh" in line for line in logs.output),
            logs.output,
        )

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
        colors.SetColor(
            label,
            Quantity_ColorRGBA(0.5, 0.5, 0.5, 1.0),
            XCAFDoc_ColorType.XCAFDoc_ColorGen,
        )
        exp, i = TopExp_Explorer(box, TopAbs_ShapeEnum.TopAbs_FACE), 0
        while exp.More():
            if i % 2:  # every other face red
                face = shapes.AddSubShape(label, exp.Current())
                colors.SetColor(
                    face,
                    Quantity_ColorRGBA(0.9, 0.1, 0.1, 1.0),
                    XCAFDoc_ColorType.XCAFDoc_ColorSurf,
                )
            exp.Next()
            i += 1
        writer = STEPCAFControl_Writer()
        writer.SetColorMode(True)
        writer.SetNameMode(True)
        writer.Transfer(doc, STEPControl_StepModelType.STEPControl_AsIs)
        step = os.path.join(self.out, "striped.step")
        writer.Write(step)
        record_path = CadImport.convert(
            step, self.out, profiles=("realtime",), verify=True, heatmaps=True
        )
        mesh = self._mesh(_load(record_path), "Striped")
        with open(os.path.join(self.out, mesh["path"])) as fh:
            obj = [
                [int(t.split("//")[0]) - 1 for t in line.split()[1:]]
                for line in fh
                if line.startswith("f ")
            ]
        with open(os.path.join(self.out, mesh["deviation"]["heatmap"])) as fh:
            lines = fh.read().splitlines()
        body = lines[lines.index("end_header") + 1 :]
        vertices, faces = body[: mesh["vertices"]], body[mesh["vertices"] :]
        self.assertEqual([[int(x) for x in f.split()[1:]] for f in faces], obj)
        self.assertEqual(len(vertices), mesh["vertices"])
        self.assertTrue(all(len(v.split()) == 6 for v in vertices))

    @staticmethod
    def _box():
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

        return BRepPrimAPI_MakeBox(20.0, 10.0, 5.0).Shape()


@unittest.skipUnless(_HAVE, "OpenCascade (pythontk[cad]) not installed")
class TestRealExports(unittest.TestCase):
    """What NIST's AP242 test models, exported from several CAD systems, hold
    and OpenCascade's own writer never does."""

    def setUp(self):
        self.out = _out_dir(self._testMethodName)
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)

    def test_a_part_the_reader_leaves_unnamed_converts(self):
        # Red before: the process died (an access violation) reading the name
        # OpenCascade never set on a PMI file's geometry-only shape.
        from OCP.collections import Sequence_TDF_Label
        from OCP.TDataStd import TDataStd_Name

        from pythontk.file_utils.cad_import._occ import _OccDocument

        real = _OccDocument._read

        def read_then_unname(doc, path, fmt):
            real(doc, path, fmt)
            labels = Sequence_TDF_Label()
            doc._shapes.GetShapes(labels)
            for i in range(1, labels.Length() + 1):
                label = labels.Value(i)
                if _OccDocument._name(label) == "Screw M4":
                    label.ForgetAttribute(TDataStd_Name.GetID_s())

        step = _build_fixture(os.path.join(self.out, "product.step"))
        with mock.patch.object(_OccDocument, "_read", read_then_unname):
            record = _load(CadImport.convert(step, self.out, profiles=("realtime",)))
        names = sorted(p["name"] for p in record["parts"])
        self.assertEqual(names, ["Housing", "Part_2", "Pin"])
        self.assertEqual(CadRecord.validate(record).errors, [])

    def test_faces_the_source_states_as_triangles_are_kept_and_named(self):
        # Red before: verifying raised (no surface to measure) and the part's
        # size read 0 x 0 x 0.
        step = _write_shapes(
            os.path.join(self.out, "tessellated.step"),
            [("Mesh", _triangles_only())],
            ap242=True,
        )
        with self.assertLogs(CadImport.logger, level="WARNING") as logs:
            record = _load(
                CadImport.convert(step, self.out, profiles=("realtime",), verify=True)
            )
        part = record["parts"][0]
        self.assertEqual([round(v, 6) for v in part["size"]], [10.0, 10.0, 0.0])
        self.assertAlmostEqual(part["area"], 100.0, places=6)
        mesh = part["meshes"][0]
        self.assertEqual((mesh["triangles"], mesh["tessellated_faces"]), (2, 1))
        self.assertEqual(
            mesh["deviation"]["peak"], 0.0
        )  # it is the source's own triangles
        self.assertTrue(
            any("Mesh" in line and "triangles" in line for line in logs.output),
            logs.output,
        )

    def test_an_edge_the_source_states_off_its_surface_is_its_gap_not_the_meshes(self):
        # Red before: the mesh read 0.05 over a 0.02 budget, the refinement
        # stopped helping, and the part failed: no chord can move an edge the
        # file itself states 0.05 off its face.
        step = _write_shapes(
            os.path.join(self.out, "loose.step"), [("Loose", _loose_face())]
        )
        with self.assertLogs(CadImport.logger, level="WARNING") as logs:
            record = _load(
                CadImport.convert(step, self.out, profiles=("standard",), verify=True)
            )
        mesh = record["parts"][0]["meshes"][0]
        dev = mesh["deviation"]
        self.assertTrue(dev["within"], dev)
        self.assertLess(dev["peak"], 1e-6)  # a plane: the mesh's own sag is none
        self.assertAlmostEqual(dev["source_gap"], LOOSE_GAP, places=4)
        self.assertEqual(mesh["chord"], CadImport.profile("standard")["chord"])
        self.assertTrue(
            any("Loose" in line and "edges" in line for line in logs.output),
            logs.output,
        )

    def test_a_source_gap_never_hides_the_sag_of_a_curved_triangle(self):
        # Review: subtracting the nodes' offsets from the sample's distance
        # hid the sag wherever the offsets point against it. One triangle on a
        # radius-10 sphere, its nodes pushed outward off the surface: the sag
        # of the triangle through the surface points stays what it is (red
        # before: 0.0004 for h = 0.033, 0 past it).
        import numpy as np
        from OCP.Geom import Geom_SphericalSurface
        from OCP.gp import gp_Ax3

        from pythontk.file_utils.cad_import._occ import _OccDocument

        sphere = Geom_SphericalSurface(gp_Ax3(), 10.0)
        uv = np.array([[0.0, 0.0], [0.5, 0.0], [0.25, 0.4]])
        on_surface = np.array([sphere.Value(u, v).Coord() for u, v in uv])
        sag = 10.0 - np.linalg.norm(on_surface.mean(axis=0))
        for h in (0.0, 0.03, 0.1):
            p = on_surface * (1.0 + h / 10.0)  # each node h outward
            _, _, gaps, gap = _OccDocument._face_deviation(
                sphere, p, np.array([[0, 1, 2]]), uv
            )
            self.assertAlmostEqual(gaps[0], sag, places=6, msg=f"h={h}")
            self.assertAlmostEqual(gap, h, places=9)

    def test_a_loose_edge_never_stops_the_refinement_a_curved_face_needs(self):
        # Red before: the loose face's 0.05 held the part's peak, so tightening
        # the chord read as no gain and the sphere was left over budget.
        from OCP.BRep import BRep_Builder
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
        from OCP.gp import gp_Pnt
        from OCP.TopoDS import TopoDS_Compound

        compound, builder = TopoDS_Compound(), BRep_Builder()
        builder.MakeCompound(compound)
        builder.Add(compound, _loose_face())
        builder.Add(compound, BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 30), 10.0).Shape())
        step = _write_shapes(
            os.path.join(self.out, "mixed.step"), [("Mixed", compound)]
        )
        with self.assertLogs(CadImport.logger, level="WARNING"):
            record = _load(
                CadImport.convert(step, self.out, profiles=("standard",), verify=True)
            )
        mesh = record["parts"][0]["meshes"][0]
        self.assertTrue(mesh["deviation"]["within"], mesh["deviation"])
        self.assertLess(mesh["chord"], CadImport.profile("standard")["chord"])
        self.assertGreater(mesh["deviation"]["peak"], 0.0)

    def test_a_part_with_no_surface_is_recorded_without_a_warning(self):
        # Red before: every profile warned "no surface to mesh" for the point
        # or wireframe compound a PMI file carries beside its solid.
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        from OCP.gp import gp_Pnt

        step = _write_shapes(
            os.path.join(self.out, "annotated.step"),
            [
                ("Block", TestVerify._box()),
                ("Datum", BRepBuilderAPI_MakeVertex(gp_Pnt(30, 0, 0)).Shape()),
            ],
        )
        with self.assertLogs(CadImport.logger, level="INFO") as logs:
            record = _load(
                CadImport.convert(step, self.out, profiles=("realtime", "standard"))
            )
        datum = next(p for p in record["parts"] if p["name"] == "Datum")
        self.assertEqual((datum["meshes"], datum["area"]), ([], 0.0))
        warnings = [line for line in logs.output if line.startswith("WARNING")]
        self.assertEqual(warnings, [])
        self.assertEqual(sum("Datum" in line for line in logs.output), 1, logs.output)


@unittest.skipUnless(_HAVE, "OpenCascade (pythontk[cad]) not installed")
class TestIdentity(unittest.TestCase):
    """What identifies a part across the source's revisions: its part number,
    the properties its CAD system writes, or its shape."""

    PROPERTIES = {
        "Housing": {"Part Number": "HX-100", "Revision": "C", "Mass": 1.25, "Batch": 7}
    }

    def setUp(self):
        self.out = _out_dir(self._testMethodName)
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)

    def _convert(self, named_shapes, product_ids=None, **kwargs):
        step = _write_shapes(
            os.path.join(self.out, "product.step"),
            named_shapes,
            self.PROPERTIES,
            product_ids,
        )
        record = _load(
            CadImport.convert(
                step, os.path.join(self.out, "out"), profiles=("realtime",), **kwargs
            )
        )
        return {p["name"]: p for p in record["parts"]}

    def _pair(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder

        return [
            ("Housing", BRepPrimAPI_MakeBox(60.0, 40.0, 10.0).Shape()),
            ("Pin", BRepPrimAPI_MakeCylinder(1.0, 5.0).Shape()),
        ]

    def test_the_part_number_and_the_cad_properties_are_recorded(self):
        parts = self._convert(self._pair(), {"Housing": "HX-100-A"})
        housing = parts["Housing"]
        self.assertEqual(housing["part_number"], "HX-100-A")
        self.assertEqual(housing["properties"], self.PROPERTIES["Housing"])
        self.assertEqual(
            parts["Pin"]["part_number"], "Pin"
        )  # OpenCascade's writer states the name
        self.assertNotIn("properties", parts["Pin"])

    def test_ids_take_the_part_number_first(self):
        parts = self._convert(self._pair(), {"Housing": "HX-100-A"})
        self.assertEqual(parts["Housing"]["id"], "HX-100-A")
        self.assertEqual(parts["Pin"]["id"], "Pin")

    def test_part_id_takes_a_property_a_callable_or_a_chain(self):
        by_property = self._convert(self._pair(), part_id=("Part Number", "name"))
        self.assertEqual(by_property["Housing"]["id"], "HX-100")
        self.assertEqual(
            by_property["Pin"]["id"], "Pin"
        )  # no such property: the next key
        by_callable = self._convert(
            self._pair(), part_id=lambda part: part["properties"].get("Revision")
        )
        self.assertEqual(by_callable["Housing"]["id"], "C")
        self.assertEqual(
            by_callable["Pin"]["id"], "Pin"
        )  # nothing from any key: its name
        by_shape = self._convert(self._pair(), part_id="fingerprint")
        self.assertEqual(by_shape["Housing"]["id"], by_shape["Housing"]["fingerprint"])

    def test_a_part_id_key_no_part_holds_is_named_in_a_warning(self):
        with self.assertLogs(CadImport.logger, level="WARNING") as logs:
            parts = self._convert(self._pair(), part_id=("PartNumbr", "name"))
        self.assertEqual(parts["Housing"]["id"], "Housing")
        self.assertTrue(any("PartNumbr" in line for line in logs.output), logs.output)

    def test_parts_sharing_an_id_are_numbered_and_named(self):
        with self.assertLogs(CadImport.logger, level="WARNING") as logs:
            parts = self._convert(self._pair(), {"Housing": "X-1", "Pin": "X-1"})
        self.assertEqual(sorted(p["id"] for p in parts.values()), ["X-1", "X-1_2"])
        self.assertTrue(
            any("Housing" in line and "Pin" in line for line in logs.output),
            logs.output,
        )

    def test_the_fingerprint_names_the_shape_not_its_name_or_unit(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

        box = BRepPrimAPI_MakeBox(60.0, 40.0, 10.0).Shape()
        twin = BRepPrimAPI_MakeBox(
            60.0, 40.0, 10.0
        ).Shape()  # its own shape, so its own part
        parts = self._convert(
            [
                ("Housing", box),
                ("Housing renamed", twin),
                ("Bracket", _l_bracket()),
                ("Bracket mirrored", _l_bracket(True)),
            ]
        )
        self.assertEqual(
            parts["Housing"]["fingerprint"], parts["Housing renamed"]["fingerprint"]
        )
        self.assertNotEqual(
            parts["Bracket"]["fingerprint"], parts["Bracket mirrored"]["fingerprint"]
        )
        self.assertEqual(
            parts["Bracket"]["volume"], parts["Bracket mirrored"]["volume"]
        )
        metres = _write_shapes(
            os.path.join(self.out, "metres.step"), [("Housing", box)], unit="M"
        )
        in_metres = _load(
            CadImport.convert(
                metres, os.path.join(self.out, "m"), profiles=("realtime",)
            )
        )
        self.assertEqual(
            in_metres["parts"][0]["fingerprint"], parts["Housing"]["fingerprint"]
        )

    def test_a_piece_with_no_product_of_its_own_takes_its_products(self):
        # OpenCascade splits a product whose shape holds a solid beside PMI
        # points into an assembly of pieces with no product of their own (all
        # 17 NIST AP242 models); the pin's attributes are removed to stand for
        # one, so it states its hinge's part number.
        from OCP.collections import Sequence_TDF_Label
        from OCP.TDataStd import TDataStd_NamedData

        from pythontk.file_utils.cad_import._occ import _OccDocument

        real = _OccDocument._read

        def read_then_strip(doc, path, fmt):
            real(doc, path, fmt)
            labels = Sequence_TDF_Label()
            doc._shapes.GetShapes(labels)
            for i in range(1, labels.Length() + 1):
                if _OccDocument._name(labels.Value(i)) == "Pin":
                    labels.Value(i).ForgetAttribute(TDataStd_NamedData.GetID_s())

        step = _build_fixture(os.path.join(self.out, "product.step"))
        with mock.patch.object(_OccDocument, "_read", read_then_strip):
            record = _load(CadImport.convert(step, self.out, profiles=("realtime",)))
        parts = {p["name"]: p for p in record["parts"]}
        self.assertEqual(parts["Pin"]["part_number"], "Hinge")
        self.assertEqual(parts["Screw M4"]["part_number"], "Screw M4")

    def test_a_product_with_a_blank_id_is_its_own_with_no_number(self):
        # Review: red before, the screw (a product of its own, its id left
        # blank) took its assembly's number "Product" as its id, unwarned.
        step = _build_fixture(os.path.join(self.out, "product.step"))
        with open(step, encoding="utf-8") as fh:
            text = fh.read()
        assert text.count("PRODUCT('Screw M4','Screw M4'") == 1
        with open(step, "w", encoding="utf-8") as fh:
            fh.write(
                text.replace("PRODUCT('Screw M4','Screw M4'", "PRODUCT('','Screw M4'")
            )
        parts = {
            p["name"]: p
            for p in _load(CadImport.convert(step, self.out, profiles=("realtime",)))[
                "parts"
            ]
        }
        self.assertNotIn("part_number", parts["Screw M4"])
        self.assertEqual(parts["Screw M4"]["id"], "Screw_M4")

    def test_pieces_of_one_product_share_its_number_without_a_warning(self):
        # Red before: NIST's tessellated model (a solid and a shell, pieces of
        # one product) warned that the two shared an id, which no key could fix.
        from OCP.collections import Sequence_TDF_Label
        from OCP.TDataStd import TDataStd_NamedData

        from pythontk.file_utils.cad_import._occ import _OccDocument

        real = _OccDocument._read

        def read_then_strip(doc, path, fmt):
            real(doc, path, fmt)
            labels = Sequence_TDF_Label()
            doc._shapes.GetShapes(labels)
            for i in range(1, labels.Length() + 1):
                if _OccDocument._name(labels.Value(i)) in (
                    "Housing",
                    "Screw M4",
                ):  # both under "Product"
                    labels.Value(i).ForgetAttribute(TDataStd_NamedData.GetID_s())

        step = _build_fixture(os.path.join(self.out, "product.step"))
        with mock.patch.object(_OccDocument, "_read", read_then_strip):
            with self.assertNoLogs(CadImport.logger, level="WARNING"):
                record = _load(
                    CadImport.convert(step, self.out, profiles=("realtime",))
                )
        ids = sorted(
            p["id"] for p in record["parts"] if p["name"] in ("Housing", "Screw M4")
        )
        self.assertEqual(ids, ["Product", "Product_2"])

    def test_an_iges_part_states_no_part_number(self):
        iges = _build_fixture(os.path.join(self.out, "housing.igs"), iges=True)
        part = _load(CadImport.convert(iges, self.out))["parts"][0]
        self.assertNotIn("part_number", part)
        self.assertEqual(part["id"], "Housing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
