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
units, names, colours, instancing and a STEP product's id and user-defined
attributes into an XCAF document; its face iterator resolves each face's colour
(falling back to its part's), orientation, location and exact surface normals;
its glTF writer writes the whole tree; its shape analysis projects a point onto
a face's exact surface. (The wheel ships no OBJ writer, ``OCP.RWObj``, so the
facade writes OBJ itself.)

Two OCP 8 bindings misbehave and are worked around here: ``FindAttribute``
crashes the process on a label without the attribute (``IsAttribute`` asks
first), and a ``TDataStd_NamedData`` map yields its values but never its keys
(the keys are the names the file gives its properties, read from the STEP model).
"""

import math
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.collections import (
    IndexedDataMap_TCollection_AsciiString_TCollection_AsciiString as _FileInfo,
)
from OCP.collections import Sequence_TDF_Label as _Labels
from OCP.Extrema import Extrema_GenLocateExtPS
from OCP.GeomAdaptor import GeomAdaptor_Surface
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.gp import gp_Pnt, gp_Pnt2d
from OCP.IGESCAFControl import IGESCAFControl_Reader
from OCP.IMeshTools import IMeshTools_Parameters
from OCP.Message import Message, Message_Gravity, Message_ProgressRange
from OCP.Quantity import Quantity_ColorRGBA
from OCP.RWGltf import RWGltf_CafWriter
from OCP.RWMesh import RWMesh_CoordinateSystem, RWMesh_FaceIterator, RWMesh_NameFormat
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name, TDataStd_NamedData
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
_IDENTITY = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]
#: The STEP entities whose names key a user-defined attribute: the property and
#: the representation item holding its value (``PartNumber`` = ``'HX-100'``).
_PROPERTY_ENTITIES = frozenset(
    {
        "StepRepr_PropertyDefinition",
        "StepRepr_DescriptiveRepresentationItem",
        "StepRepr_ValueRepresentationItem",
        "StepRepr_MeasureRepresentationItem",
        "StepRepr_IntegerRepresentationItem",
        "StepRepr_RealRepresentationItem",
    }
)
#: Where OpenCascade's product reading puts the product's id: its part number.
_PRODUCT_ID = "ProductID"


class _OccDocument:
    """One CAD file read into an XCAF document, in millimetres.

    ``nodes``: the assembly tree, parents first, each ``{"name", "parent",
    "matrix", "part", "color"}`` (``part`` an index into ``parts``, or None for
    an assembly). ``parts``: each unique part once, in the order first placed,
    ``{"name", "label", "shape", "color", "faces", "part_number",
    "properties", "product"}`` (``product`` the entry of the label whose
    number and attributes it states). A name the file left out is filled in
    the document too, so the GLB writer names every node.

    A part with no product of its own is a piece of its parent's: OpenCascade
    splits a product whose shape holds a solid beside points or curves (a PMI
    file's annotation geometry) into an assembly of the pieces, and each piece
    takes the product's part number and properties.
    """

    def __init__(self, path: str, fmt: str) -> None:
        self.doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
        XCAFApp_Application.GetApplication_s().InitDocument(self.doc)
        # Readers convert whatever unit a file states into the document's.
        XCAFDoc_DocumentTool.SetLengthUnit_s(self.doc, 0.001)
        self._shapes = XCAFDoc_DocumentTool.ShapeTool_s(self.doc.Main())
        self._property_keys: Set[str] = set()
        self._read(path, fmt)
        self.nodes: List[Dict[str, Any]] = []
        self.parts: List[Dict[str, Any]] = []
        self._part_index: Dict[str, int] = {}
        free = _Labels()
        self._shapes.GetFreeShapes(free)
        for i in range(1, free.Length() + 1):
            label = free.Value(i)
            self._add(
                label,
                None,
                list(_IDENTITY),
                self._name(label),
                self._color(label),
                (None, {}, None),
            )

    # ------------------------------------------------------------ reading
    def _read(self, path: str, fmt: str) -> None:
        reader = STEPCAFControl_Reader() if fmt == "step" else IGESCAFControl_Reader()
        reader.SetColorMode(True)
        reader.SetNameMode(True)
        if fmt == "step":  # a product's id and its user-defined attributes
            reader.SetMetaMode(True)
            reader.SetProductMetaMode(True)
        # The readers print progress to the console at the default printer's
        # info level; for this read alone it hears warnings and worse.
        printers = [
            (p, p.GetTraceLevel()) for p in Message.DefaultMessenger_s().Printers()
        ]
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
        if fmt == "step":
            self._property_keys = self._keys(reader.Reader().WS().Model())

    @staticmethod
    def _keys(model) -> Set[str]:
        """Every name the STEP model gives a property or a property's value: the
        keys a label's attributes can be stored under."""
        keys = set()
        for i in range(1, model.NbEntities() + 1):
            entity = model.Value(i)
            if entity.DynamicType().Name() in _PROPERTY_ENTITIES:
                name = entity.Name()
                if name is not None and name.ToCString().strip():
                    keys.add(name.ToCString())
        return keys

    def _add(
        self,
        label,
        parent: Optional[int],
        matrix: List[float],
        name: str,
        color,
        product,
    ) -> None:
        """Add *label* under *parent*: an assembly with its components, or a placement of a part.

        *product* is ``(part_number, properties, entry)`` of the nearest
        product above (``entry`` its label's), for a piece that states none of
        its own.
        """
        index = len(self.nodes)
        node = {
            "name": name,
            "parent": parent,
            "matrix": matrix,
            "part": None,
            "color": color,
        }
        self.nodes.append(node)
        own = self._metadata(label)
        if own is not None:  # a product of its own, even one whose id is blank
            product = (*own, self._entry(label))
        if not self._shapes.IsAssembly_s(label):
            node["part"] = self._part(label, product)
            return
        components = _Labels()
        self._shapes.GetComponents_s(label, components, False)
        for i in range(1, components.Length() + 1):
            component = components.Value(i)
            referred = TDF_Label()
            if not self._shapes.GetReferredShape_s(component, referred):
                continue  # an empty reference: nothing to place
            own_name = self._name(component)
            if not own_name or own_name.startswith(_AUTO_NAME):
                own_name = self._name(referred) or f"Node_{len(self.nodes)}"
                TDataStd_Name.Set_s(component, TCollection_ExtendedString(own_name))
            local = self._matrix(self._shapes.GetLocation_s(component))
            self._add(referred, index, local, own_name, self._color(component), product)

    @staticmethod
    def _entry(label) -> str:
        """*label*'s address in the document (``0:1:1:3``)."""
        entry = TCollection_AsciiString()
        TDF_Tool.Entry_s(label, entry)
        return entry.ToCString()

    def _part(self, label, product) -> int:
        """The index of *label*'s part, added on its first placement.

        ``product`` names the product whose number and attributes it states:
        its own label's, or the product it is a piece of.
        """
        key = self._entry(label)
        if key not in self._part_index:
            name = self._name(label)
            if not name:
                name = f"Part_{len(self.parts) + 1}"
                TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))
            shape = self._shapes.GetShape_s(label)
            faces, exp = 0, TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_FACE)
            while exp.More():
                faces += 1
                exp.Next()
            self._part_index[key] = len(self.parts)
            self.parts.append(
                {
                    "name": name,
                    "label": label,
                    "shape": shape,
                    "color": self._color(label),
                    "faces": faces,
                    "part_number": product[0],
                    "properties": dict(product[1]),
                    "product": product[2] or key,
                }
            )
        return self._part_index[key]

    @staticmethod
    def _name(label) -> str:
        if not label.IsAttribute(TDataStd_Name.GetID_s()):  # FindAttribute would crash
            return ""
        attr = TDataStd_Name()
        label.FindAttribute(TDataStd_Name.GetID_s(), attr)
        return attr.Get().ToExtString()

    def _metadata(
        self, label
    ) -> "Optional[tuple[Optional[str], Dict[str, Union[str, float, int]]]]":
        """*label*'s part number (its STEP product's id, None when blank) and its
        user-defined attributes, as the reader stored them; None when the label
        states no product (a piece the reader split from one, or IGES)."""
        if not label.IsAttribute(TDataStd_NamedData.GetID_s()):
            return None
        data = TDataStd_NamedData()
        label.FindAttribute(TDataStd_NamedData.GetID_s(), data)
        data.LoadDeferredData()

        def value(key: str):
            k = TCollection_ExtendedString(key)
            if data.HasString(k):
                return data.GetString(k).ToExtString()
            if data.HasReal(k):
                return float(data.GetReal(k))
            if data.HasInteger(k):
                return int(data.GetInteger(k))
            return None

        number = value(_PRODUCT_ID)
        number = number if isinstance(number, str) and number.strip() else None
        properties = {}
        for key in sorted(self._property_keys):
            found = value(key)
            if found is not None:
                properties[key] = found
        return number, properties

    def _color(self, label) -> Optional[List[float]]:
        """*label*'s own colour as linear RGBA (a surface colour first), or None."""
        for kind in (
            XCAFDoc_ColorType.XCAFDoc_ColorSurf,
            XCAFDoc_ColorType.XCAFDoc_ColorGen,
        ):
            color = Quantity_ColorRGBA()
            if XCAFDoc_DocumentTool.ColorTool_s(self.doc.Main()).GetColor_s(
                label, kind, color
            ):
                return self._rgba(color)
        return None

    @staticmethod
    def _rgba(color) -> List[float]:
        rgb = color.GetRGB()
        return [
            float(rgb.Red()),
            float(rgb.Green()),
            float(rgb.Blue()),
            float(color.Alpha()),
        ]

    @staticmethod
    def _matrix(location) -> List[float]:
        """A location as 16 numbers, row-major."""
        t = location.Transformation()
        return [float(t.Value(r, c)) for r in (1, 2, 3) for c in (1, 2, 3, 4)] + [
            0.0,
            0.0,
            0.0,
            1.0,
        ]

    # ------------------------------------------------------------ figures
    @staticmethod
    def figures(part: Dict[str, Any]) -> Dict[str, Any]:
        """The part's size (the tight box of its exact geometry), area and volume
        (OpenCascade's integration over its surfaces), and its
        ``moments``: ``{"mass", "centroid", "inertia"}`` of its volume (of its
        surface when it holds no solid), in its own frame; the inertia about
        the centroid as ``[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]``, its sign the mass's.
        A face the source states as triangles alone is measured from them."""
        shape = part["shape"]
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(shape, props)
        area = props.Mass()
        volume = 0.0
        if TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_SOLID).More():
            solid = GProp_GProps()
            BRepGProp.VolumeProperties_s(shape, solid)
            volume = abs(solid.Mass())
            if volume > 0.0:
                props = solid
        box = Bnd_Box()
        BRepBndLib.AddOptimal_s(shape, box, False, False)
        if box.IsVoid():  # no exact geometry to bound: the triangles the source states
            BRepBndLib.AddOptimal_s(shape, box, True, False)
        if box.IsVoid():
            size = [0.0, 0.0, 0.0]
        else:
            low, high = box.CornerMin(), box.CornerMax()
            size = [high.X() - low.X(), high.Y() - low.Y(), high.Z() - low.Z()]
        moments = None
        if props.Mass() != 0.0:  # a part with no surface has none
            m = props.MatrixOfInertia()
            moments = {
                "mass": float(props.Mass()),
                "centroid": [float(c) for c in props.CentreOfMass().Coord()],
                "inertia": [
                    float(m.Value(1, 1)),
                    float(m.Value(2, 2)),
                    float(m.Value(3, 3)),
                    float(m.Value(1, 2)),
                    float(m.Value(1, 3)),
                    float(m.Value(2, 3)),
                ],
            }
        return {
            "size": [float(v) for v in size],
            "area": float(area),
            "volume": float(volume),
            "moments": moments,
        }

    # ------------------------------------------------------------ meshes
    def tessellate(
        self, chord: float, angle: float, parts: Optional[List[int]] = None
    ) -> None:
        """Mesh *parts* (indices; default every part) afresh: *chord* in mm, *angle* in degrees.

        A placement shares its part's surfaces, so meshing each part once
        meshes every placement of it. The chord is the mesher's deflection
        target. By default OpenCascade checks it inside a face for freeform
        surfaces alone; it is checked on every surface here (an analytic
        sphere's realtime mesh strayed 1.6 times the chord, 1.3 with the check;
        a NIST B-spline part 0.059 where 0.035 with it). Inside a triangle a
        surface can still stray past the chord, which is what :meth:`mesh`
        with ``measure`` finds.
        """
        params = IMeshTools_Parameters()
        params.Deflection, params.Angle = chord, math.radians(angle)
        params.Relative, params.InParallel = False, True
        params.EnableControlSurfaceDeflectionAllSurfaces = True
        for index in range(len(self.parts)) if parts is None else parts:
            shape = self.parts[index]["shape"]
            BRepTools.Clean_s(shape)
            BRepMesh_IncrementalMesh(shape, params)

    def mesh(self, part: Dict[str, Any], measure: bool = False) -> Dict[str, Any]:
        """The part's current triangulation as arrays, in its own frame.

        ``vertices`` and ``normals`` (N, 3), ``triangles`` (M, 3) wound
        outward, ``slots`` (M,) indexing ``colors`` (each linear RGBA, or None
        for a face with no colour), and ``tessellated_faces``: how many faces
        the source states as triangles alone (no surface), written as given.
        Each face keeps its own vertices, so the crease between two faces stays
        sharp.

        With *measure*, also how far the mesh strays from the part's exact
        surfaces: each triangle's centroid and each edge's midpoint projected
        onto its face's surface (where a flat triangle strays furthest; a
        45-point grid per triangle found the same peaks), less the source's
        own gap there (see :meth:`_face_deviation`). ``deviation`` is
        ``{"peak", "mean", "peak_at", "samples", "source_gap"}`` and
        ``vertex_deviation`` (N,) each vertex's largest neighbouring sample.

        The frame is the one OpenCascade's glTF writer places a part in: the
        iterator replaces a part shape's own root location with the one it is
        given (none), as the writer does, and the record's placements compose
        over the result.
        """
        points, normals, triangles, slots, colors, spread = [], [], [], [], [], []
        keys: Dict[Any, int] = {}
        offset, total, count, peak, peak_at, source_gap, tessellated = (
            0,
            0.0,
            0,
            -1.0,
            [],
            0.0,
            0,
        )
        it = RWMesh_FaceIterator(
            part["label"], TopLoc_Location(), True, XCAFPrs_Style()
        )
        while it.More():
            if not it.IsEmptyMesh():
                low, high = it.NodeLower(), it.NodeUpper()
                p = np.array(
                    [it.NodeTransformed(i).Coord() for i in range(low, high + 1)], float
                )
                t = (
                    np.array(
                        [
                            it.TriangleOriented(k).Get()
                            for k in range(it.ElemLower(), it.ElemUpper() + 1)
                        ],
                        int,
                    )
                    - low
                )
                if it.HasNormals():
                    n = np.array(
                        [it.NormalTransformed(i).Coord() for i in range(low, high + 1)],
                        float,
                    )
                else:
                    n = self._flat_normals(p, t)
                geometry = BRep_Tool.Surface_s(it.Face())
                tessellated += geometry is None
                if measure and geometry is None:  # nothing exact to measure against
                    spread.append(np.zeros(len(p)))
                elif measure:
                    tri = it.Triangulation()
                    uv = (
                        np.array(
                            [tri.UVNode(i).Coord() for i in range(low, high + 1)], float
                        )
                        if tri.HasUVNodes()
                        else None
                    )
                    vdev, samples, gaps, gap = self._face_deviation(geometry, p, t, uv)
                    spread.append(vdev)
                    source_gap = max(source_gap, gap)
                    total, count = total + float(gaps.sum()), count + len(gaps)
                    if len(gaps) and gaps.max() > peak:
                        peak, peak_at = (
                            float(gaps.max()),
                            [float(x) for x in samples[int(gaps.argmax())]],
                        )
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
            mesh = {
                "vertices": empty,
                "normals": empty,
                "triangles": np.zeros((0, 3), int),
                "slots": np.zeros(0, int),
                "colors": [],
                "vertex_deviation": np.zeros(0),
            }
        else:
            mesh = self._compact(
                np.vstack(points),
                np.vstack(normals),
                np.vstack(triangles),
                np.concatenate(slots),
                colors,
                np.concatenate(spread) if measure else np.zeros(offset),
            )
        mesh["tessellated_faces"] = tessellated
        if measure:
            mesh["deviation"] = {
                "peak": max(peak, 0.0),
                "mean": total / count if count else 0.0,
                "peak_at": peak_at,
                "samples": count,
                "source_gap": source_gap,
            }
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
    def _face_deviation(
        geometry, p: np.ndarray, t: np.ndarray, uv: Optional[np.ndarray]
    ):
        """Each vertex's largest neighbouring sample distance, the samples,
        their distances, and the source's own gap on the face.

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
        caps the rest at the triangle's own size. *geometry* is the face's
        surface with its location applied, the frame the nodes are in.

        A node on a face's boundary lies on the edge's curve, which a real
        export states within its tolerance of the surface, not on it (NIST's
        test models: up to 0.1 mm). That gap is the source's: no chord moves
        it. So the triangle measured is the one through each node's surface
        point at its own UV: its sag is the tessellation's error, which the
        budget judges, and the largest node offset is returned as the source's
        gap; the written mesh lies within their sum of the surface. The
        samples returned are the matching points of the written mesh, where
        a peak is found.
        """
        edges = np.unique(
            np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1),
            axis=0,
        )
        q = (
            p
            if uv is None
            else np.array([geometry.Value(u, v).Coord() for u, v in uv], float)
        )
        node_gap = np.linalg.norm(p - q, axis=1)

        def at(points):
            return np.vstack(
                [
                    points[t].mean(axis=1),
                    0.5 * (points[edges[:, 0]] + points[edges[:, 1]]),
                ]
            )

        measured, samples = at(q), at(p)
        bounds = np.concatenate(
            [
                np.linalg.norm(q[t] - measured[: len(t), None, :], axis=2).min(axis=1),
                0.5 * np.linalg.norm(q[edges[:, 0]] - q[edges[:, 1]], axis=1),
            ]
        )
        seeds = (
            None
            if uv is None
            else np.vstack(
                [uv[t].mean(axis=1), 0.5 * (uv[edges[:, 0]] + uv[edges[:, 1]])]
            )
        )
        surface = ShapeAnalysis_Surface(geometry)
        adaptor = GeomAdaptor_Surface(geometry)  # the search holds a reference to it
        search = Extrema_GenLocateExtPS(adaptor)
        gaps = np.empty(len(measured))
        for i, (x, y, z) in enumerate(measured):
            point = gp_Pnt(x, y, z)
            if seeds is None:
                surface.ValueOfUV(point, 1e-7)
                distance = surface.Gap()
            else:
                u, v = seeds[i]
                surface.NextValueOfUV(gp_Pnt2d(u, v), point, 1e-7, -1.0)
                distance = surface.Gap()
                search.Perform(point, u, v, True)
                if search.IsDone():
                    distance = min(distance, math.sqrt(search.SquareDistance()))
            gaps[i] = min(distance, bounds[i])
        vdev = np.zeros(len(p))
        centroid_gaps, edge_gaps = gaps[: len(t)], gaps[len(t) :]
        for k in range(3):
            np.maximum.at(vdev, t[:, k], centroid_gaps)
        np.maximum.at(vdev, edges[:, 0], edge_gaps)
        np.maximum.at(vdev, edges[:, 1], edge_gaps)
        return vdev, samples, gaps, float(node_gap.max()) if len(node_gap) else 0.0

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
        doubled = np.linalg.norm(
            np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1
        )
        keep = doubled > _DEGENERATE
        t, s = t[keep], s[keep]
        order = np.argsort(s, kind="stable")
        t, s = t[order], s[order]
        used, inverse = np.unique(t, return_inverse=True)
        return {
            "vertices": v[used],
            "normals": n[used],
            "triangles": inverse.reshape(t.shape),
            "slots": s,
            "colors": colors,
            "vertex_deviation": vdev[used],
        }

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
