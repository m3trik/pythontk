# !/usr/bin/python
# coding=utf-8
"""Every OpenCascade call the CAD import makes, behind one document class.

Imported only by :meth:`CadImport.convert <pythontk.CadImport.convert>`, after
:meth:`CadImport.resolve` found the wheel (the ``pythontk[cad]`` extra), so a
module-level import is that one path's import. The names are OCCT 8's: several
differ from most published examples (``Sequence_TDF_Label`` lives in
``OCP.collections``, ``TopoDS.Face`` has no ``_s``, ``Bnd_Box.Get`` cannot be
called), which is why every call stays here.

What OpenCascade already does is adopted, not rewritten: its readers resolve
units, names, colours and instancing into an XCAF document; its face iterator
resolves each face's colour (falling back to its part's), orientation, location
and exact surface normals; its glTF writer writes the whole tree; its shape
analysis projects a point onto a face's exact surface. (The wheel ships no OBJ
writer, ``OCP.RWObj``, so the facade writes OBJ itself.)
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.collections import IndexedDataMap_TCollection_AsciiString_TCollection_AsciiString as _FileInfo
from OCP.collections import Sequence_TDF_Label as _Labels
from OCP.Extrema import Extrema_GenLocateExtPS
from OCP.GeomAdaptor import GeomAdaptor_Surface
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.gp import gp_Pnt, gp_Pnt2d
from OCP.IGESCAFControl import IGESCAFControl_Reader
from OCP.Message import Message, Message_Gravity, Message_ProgressRange
from OCP.Quantity import Quantity_ColorRGBA
from OCP.RWGltf import RWGltf_CafWriter
from OCP.RWMesh import RWMesh_CoordinateSystem, RWMesh_FaceIterator, RWMesh_NameFormat
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_Tool
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_ShapeEnum
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool
from OCP.XCAFPrs import XCAFPrs_Style

#: How OpenCascade names a placement its file left unnamed (``=>[0:1:1:2]``).
_AUTO_NAME = "=>["
#: A triangle whose doubled area is below this (mm²) is a point or a line: a
#: sphere's pole, where the surface's parameter lines meet.
_DEGENERATE = 1e-12
_IDENTITY = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


class _OccDocument:
    """One CAD file read into an XCAF document, in millimetres.

    ``nodes``: the assembly tree, parents first, each ``{"name", "parent",
    "matrix", "part", "color"}`` (``part`` an index into ``parts``, or None for
    an assembly). ``parts``: each unique part once, in the order first placed,
    ``{"name", "label", "shape", "color"}``. A name the file left out is filled
    in the document too, so the GLB writer names every node.
    """

    def __init__(self, path: str, fmt: str) -> None:
        self.doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
        XCAFApp_Application.GetApplication_s().InitDocument(self.doc)
        # Readers convert whatever unit a file states into the document's.
        XCAFDoc_DocumentTool.SetLengthUnit_s(self.doc, 0.001)
        self._shapes = XCAFDoc_DocumentTool.ShapeTool_s(self.doc.Main())
        self._read(path, fmt)
        self.nodes: List[Dict[str, Any]] = []
        self.parts: List[Dict[str, Any]] = []
        self._part_index: Dict[str, int] = {}
        free = _Labels()
        self._shapes.GetFreeShapes(free)
        for i in range(1, free.Length() + 1):
            label = free.Value(i)
            self._add(label, None, list(_IDENTITY), self._name(label), self._color(label))

    # ------------------------------------------------------------ reading
    def _read(self, path: str, fmt: str) -> None:
        reader = STEPCAFControl_Reader() if fmt == "step" else IGESCAFControl_Reader()
        reader.SetColorMode(True)
        reader.SetNameMode(True)
        # The readers print progress to the console at the default printer's
        # info level; for this read alone it hears warnings and worse.
        printers = [(p, p.GetTraceLevel()) for p in Message.DefaultMessenger_s().Printers()]
        for printer, _ in printers:
            printer.SetTraceLevel(Message_Gravity.Message_Warning)
        try:
            status = reader.ReadFile(path)
            if status != IFSelect_ReturnStatus.IFSelect_RetDone:
                raise ValueError(
                    f"OpenCascade could not read {path} as {fmt.upper()} ({status.name}). If "
                    "it opens in a CAD viewer, export it again as STEP AP214 or AP242."
                )
            if not reader.Transfer(self.doc):
                raise ValueError(
                    f"{path} holds no shape OpenCascade could transfer: it may hold only "
                    "drawings or annotations. Export the 3D model as STEP AP242."
                )
        finally:
            for printer, level in printers:
                printer.SetTraceLevel(level)

    def _add(self, label, parent: Optional[int], matrix: List[float], name: str, color) -> None:
        """Add *label* under *parent*: an assembly with its components, or a placement of a part."""
        index = len(self.nodes)
        node = {"name": name, "parent": parent, "matrix": matrix, "part": None, "color": color}
        self.nodes.append(node)
        if not self._shapes.IsAssembly_s(label):
            node["part"] = self._part(label)
            return
        components = _Labels()
        self._shapes.GetComponents_s(label, components, False)
        for i in range(1, components.Length() + 1):
            component = components.Value(i)
            referred = TDF_Label()
            self._shapes.GetReferredShape_s(component, referred)
            own = self._name(component)
            if not own or own.startswith(_AUTO_NAME):
                own = self._name(referred) or f"Node_{len(self.nodes)}"
                TDataStd_Name.Set_s(component, TCollection_ExtendedString(own))
            local = self._matrix(self._shapes.GetLocation_s(component))
            self._add(referred, index, local, own, self._color(component))

    def _part(self, label) -> int:
        """The index of *label*'s part, added on its first placement."""
        entry = TCollection_AsciiString()
        TDF_Tool.Entry_s(label, entry)
        key = entry.ToCString()
        if key not in self._part_index:
            name = self._name(label)
            if not name:
                name = f"Part_{len(self.parts) + 1}"
                TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))
            self._part_index[key] = len(self.parts)
            self.parts.append(
                {"name": name, "label": label, "shape": self._shapes.GetShape_s(label), "color": self._color(label)}
            )
        return self._part_index[key]

    @staticmethod
    def _name(label) -> str:
        attr = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
            return attr.Get().ToExtString()
        return ""

    def _color(self, label) -> Optional[List[float]]:
        """*label*'s own colour as linear RGBA (a surface colour first), or None."""
        for kind in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
            color = Quantity_ColorRGBA()
            if XCAFDoc_DocumentTool.ColorTool_s(self.doc.Main()).GetColor_s(label, kind, color):
                return self._rgba(color)
        return None

    @staticmethod
    def _rgba(color) -> List[float]:
        rgb = color.GetRGB()
        return [float(rgb.Red()), float(rgb.Green()), float(rgb.Blue()), float(color.Alpha())]

    @staticmethod
    def _matrix(location) -> List[float]:
        """A location as 16 numbers, row-major."""
        t = location.Transformation()
        return [float(t.Value(r, c)) for r in (1, 2, 3) for c in (1, 2, 3, 4)] + [0.0, 0.0, 0.0, 1.0]

    # ------------------------------------------------------------ figures
    @staticmethod
    def figures(part: Dict[str, Any]) -> Dict[str, Any]:
        """The part's exact size, area and volume, from its surfaces."""
        shape = part["shape"]
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(shape, props)
        area = props.Mass()
        volume = 0.0
        if TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_SOLID).More():
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(shape, props)
            volume = abs(props.Mass())
        box = Bnd_Box()
        BRepBndLib.AddOptimal_s(shape, box, False, False)
        if box.IsVoid():
            size = [0.0, 0.0, 0.0]
        else:
            low, high = box.CornerMin(), box.CornerMax()
            size = [high.X() - low.X(), high.Y() - low.Y(), high.Z() - low.Z()]
        return {"size": [float(v) for v in size], "area": float(area), "volume": float(volume)}

    # ------------------------------------------------------------ meshes
    def tessellate(self, chord: float, angle: float, parts: Optional[List[int]] = None) -> None:
        """Mesh *parts* (indices; default every part) afresh: *chord* in mm, *angle* in degrees.

        A placement shares its part's surfaces, so meshing each part once
        meshes every placement of it. The chord is the mesher's deflection
        target at its nodes and edges: inside a triangle a surface can stray
        further (an analytic sphere's mesh does, by up to twice the chord),
        which is what :meth:`mesh` with ``measure`` finds.
        """
        for index in range(len(self.parts)) if parts is None else parts:
            shape = self.parts[index]["shape"]
            BRepTools.Clean_s(shape)
            BRepMesh_IncrementalMesh(shape, chord, False, math.radians(angle), True)

    def mesh(self, part: Dict[str, Any], measure: bool = False) -> Dict[str, Any]:
        """The part's current triangulation as arrays, in its own frame.

        ``vertices`` and ``normals`` (N, 3), ``triangles`` (M, 3) wound
        outward, ``slots`` (M,) indexing ``colors`` (each linear RGBA, or None
        for a face with no colour). Each face keeps its own vertices, so the
        crease between two faces stays sharp.

        With *measure*, also how far the mesh strays from the part's exact
        surfaces: each triangle's centroid and each edge's midpoint projected
        onto its face's surface (where a flat triangle strays furthest; a
        45-point grid per triangle found the same peaks). ``deviation`` is
        ``{"peak", "mean", "peak_at", "samples"}`` and ``vertex_deviation``
        (N,) each vertex's largest neighbouring sample.

        The frame is the one OpenCascade's glTF writer places a part in: the
        iterator replaces a part shape's own root location with the one it is
        given (none), as the writer does, and the record's placements compose
        over the result.
        """
        points, normals, triangles, slots, colors, spread = [], [], [], [], [], []
        keys: Dict[Any, int] = {}
        offset, total, count, peak, peak_at = 0, 0.0, 0, -1.0, []
        it = RWMesh_FaceIterator(part["label"], TopLoc_Location(), True, XCAFPrs_Style())
        while it.More():
            if not it.IsEmptyMesh():
                low, high = it.NodeLower(), it.NodeUpper()
                p = np.array([it.NodeTransformed(i).Coord() for i in range(low, high + 1)], float)
                t = np.array(
                    [it.TriangleOriented(k).Get() for k in range(it.ElemLower(), it.ElemUpper() + 1)], int
                ) - low
                if it.HasNormals():
                    n = np.array([it.NormalTransformed(i).Coord() for i in range(low, high + 1)], float)
                else:
                    n = self._flat_normals(p, t)
                if measure:
                    tri = it.Triangulation()
                    uv = (
                        np.array([tri.UVNode(i).Coord() for i in range(low, high + 1)], float)
                        if tri.HasUVNodes()
                        else None
                    )
                    vdev, samples, gaps = self._face_deviation(it.Face(), p, t, uv)
                    spread.append(vdev)
                    total, count = total + float(gaps.sum()), count + len(gaps)
                    if len(gaps) and gaps.max() > peak:
                        peak, peak_at = float(gaps.max()), [float(x) for x in samples[int(gaps.argmax())]]
                color = self._rgba(it.FaceColor()) if it.HasFaceColor() else None
                key = None if color is None else tuple(round(c, 6) for c in color)
                if key not in keys:
                    keys[key] = len(colors)
                    colors.append(color)
                points.append(p)
                normals.append(n)
                triangles.append(t + offset)
                slots.append(np.full(len(t), keys[key], int))
                offset += len(p)
            it.Next()
        if not points:
            empty = np.zeros((0, 3))
            mesh = {"vertices": empty, "normals": empty, "triangles": np.zeros((0, 3), int),
                    "slots": np.zeros(0, int), "colors": [], "vertex_deviation": np.zeros(0)}
        else:
            mesh = self._compact(
                np.vstack(points), np.vstack(normals), np.vstack(triangles), np.concatenate(slots), colors,
                np.concatenate(spread) if measure else np.zeros(offset),
            )
        if measure:
            mesh["deviation"] = {"peak": max(peak, 0.0), "mean": total / count if count else 0.0,
                                 "peak_at": peak_at, "samples": count}
        return mesh

    @staticmethod
    def unmeshed(part: Dict[str, Any]) -> "tuple[int, float]":
        """How many of the part's faces the mesher left without a triangle, and their exact area (mm²)."""
        count, area = 0, 0.0
        exp = TopExp_Explorer(part["shape"], TopAbs_ShapeEnum.TopAbs_FACE)
        while exp.More():
            face = TopoDS.Face(exp.Current())
            tri = BRep_Tool.Triangulation_s(face, TopLoc_Location())
            if tri is None or tri.NbTriangles() == 0:
                props = GProp_GProps()
                BRepGProp.SurfaceProperties_s(face, props)
                count, area = count + 1, area + float(props.Mass())
            exp.Next()
        return count, area

    @staticmethod
    def _face_deviation(face, p: np.ndarray, t: np.ndarray, uv: Optional[np.ndarray]):
        """Each vertex's largest neighbouring sample distance, the samples, and their distances.

        Every distance taken lands on a point of the surface, so each can only
        overstate the true one, never hide it, and the least of them is kept.
        No single OpenCascade projection holds on a freeform B-spline: the
        global one read a seventh of a wavy panel's own mesh nodes as up to
        1.6 mm off it; the one seeded at a sample's own UV (its triangle's
        node UVs averaged) falls back to the global one where its local step
        fails; the local distance search from that seed diverges elsewhere.
        They fail on different samples, and their least matched the true peak
        on that panel (a dense-grid check) and the closed form on a sphere.
        The distance to the sample's nearest node, which lies on the surface,
        caps the rest at the triangle's own size. The surface comes with the
        face's location applied, the frame the nodes are in.
        """
        edges = np.unique(np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1), axis=0)
        centroids = p[t].mean(axis=1)
        samples = np.vstack([centroids, 0.5 * (p[edges[:, 0]] + p[edges[:, 1]])])
        bounds = np.concatenate([
            np.linalg.norm(p[t] - centroids[:, None, :], axis=2).min(axis=1),
            0.5 * np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1),
        ])
        seeds = None if uv is None else np.vstack([uv[t].mean(axis=1), 0.5 * (uv[edges[:, 0]] + uv[edges[:, 1]])])
        geometry = BRep_Tool.Surface_s(face)
        surface = ShapeAnalysis_Surface(geometry)
        adaptor = GeomAdaptor_Surface(geometry)  # the search holds a reference to it
        search = Extrema_GenLocateExtPS(adaptor)
        gaps = np.empty(len(samples))
        for i, (x, y, z) in enumerate(samples):
            point = gp_Pnt(x, y, z)
            if seeds is None:
                surface.ValueOfUV(point, 1e-7)
                gap = surface.Gap()
            else:
                u, v = seeds[i]
                surface.NextValueOfUV(gp_Pnt2d(u, v), point, 1e-7, -1.0)
                gap = surface.Gap()
                search.Perform(point, u, v, True)
                if search.IsDone():
                    gap = min(gap, math.sqrt(search.SquareDistance()))
            gaps[i] = min(gap, bounds[i])
        vdev = np.zeros(len(p))
        centroid_gaps, edge_gaps = gaps[: len(t)], gaps[len(t):]
        for k in range(3):
            np.maximum.at(vdev, t[:, k], centroid_gaps)
        np.maximum.at(vdev, edges[:, 0], edge_gaps)
        np.maximum.at(vdev, edges[:, 1], edge_gaps)
        return vdev, samples, gaps

    @staticmethod
    def _flat_normals(p: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Area-weighted vertex normals, for a face whose surface gave none."""
        n = np.zeros_like(p)
        cross = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
        for k in range(3):
            np.add.at(n, t[:, k], cross)
        length = np.linalg.norm(n, axis=1)
        length[length == 0] = 1.0
        return n / length[:, None]

    @staticmethod
    def _compact(v, n, t, s, colors, vdev) -> Dict[str, Any]:
        """Drop degenerate triangles, then the vertices nothing references;
        order the triangles by colour slot, the order the OBJ groups them in,
        so every file written from the mesh lists its faces alike."""
        doubled = np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
        keep = doubled > _DEGENERATE
        t, s = t[keep], s[keep]
        order = np.argsort(s, kind="stable")
        t, s = t[order], s[order]
        used, inverse = np.unique(t, return_inverse=True)
        return {"vertices": v[used], "normals": n[used], "triangles": inverse.reshape(t.shape),
                "slots": s, "colors": colors, "vertex_deviation": vdev[used]}

    # ------------------------------------------------------------ writing
    def write_glb(self, path: str, up_axis: str) -> None:
        """The whole tree as a GLB at the current tessellation: metres, +Y up."""
        writer = RWGltf_CafWriter(TCollection_AsciiString(path), True)
        converter = writer.ChangeCoordinateSystemConverter()
        converter.SetInputLengthUnit(0.001)
        converter.SetInputCoordinateSystem(
            RWMesh_CoordinateSystem.RWMesh_CoordinateSystem_Zup
            if up_axis == "z"
            else RWMesh_CoordinateSystem.RWMesh_CoordinateSystem_Yup
        )
        writer.SetNodeNameFormat(RWMesh_NameFormat.RWMesh_NameFormat_InstanceOrProduct)
        writer.SetMeshNameFormat(RWMesh_NameFormat.RWMesh_NameFormat_Product)
        if not writer.Perform(self.doc, _FileInfo(), Message_ProgressRange()):
            raise RuntimeError(f"OpenCascade's glTF writer could not write {path}.")
