# !/usr/bin/python
# coding=utf-8
"""The CAD import's record, declared once.

:meth:`CadImport.convert <pythontk.CadImport.convert>` writes one beside the meshes
it makes: what the source file states about the product (its assembly tree, part
names, placements, colours) and what its geometry measures (size, area, volume,
integrated over the exact CAD surfaces, never read from a mesh), plus each mesh written from
it and, when measured, how far that mesh strays from the surfaces. Later stages read
the record instead of the CAD file, so none of them needs the CAD backend.

Conventions every field shares:

- Lengths are millimetres, the unit the import reads every file into whatever
  unit the file states. Areas are mm², volumes mm³.
- Matrices are 4x4, row-major, translation in the last column, in the source's
  own axes (``source.up_axis`` says which is up).
- Colours are linear RGBA, 0..1, as glTF's ``baseColorFactor`` states them. A STEP
  file states sRGB; the import converts it.
- Paths are relative to the record's own folder, and the source is named by file
  name alone, so a delivered record names no folder of the machine that wrote it.

A typed :class:`~pythontk.SchemaSpec`: ``CadRecord.validate(data)`` checks a record,
``CadRecord.json_schema()`` publishes its shape for readers in other languages.
Change a field together with ``CadImport.convert`` and bump :attr:`CadRecord.VERSION`
-- unless the field is optional and a reader that lacks it reads the record as before.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from pythontk.core_utils.schema_spec import SchemaSpec

_field = SchemaSpec.spec_field


@dataclass
class CadDeviation(SchemaSpec):
    """How far one mesh strays from its part's exact surfaces: each triangle's
    centroid and each edge's midpoint projected onto its face's surface, where
    a flat triangle strays furthest (``CadImport.convert`` with ``verify``)."""

    TYPED = True

    peak: float = _field(
        help=(
            "The largest distance from the exact surfaces among those points, in mm, "
            "taken on the triangles through each mesh node's surface point: the "
            "tessellation's error. The written mesh lies within `peak` plus "
            "`source_gap` of the surfaces."
        ),
        required=True,
        example=0.0146,
    )
    mean: float = _field(
        help="The mean distance over those points, in mm.",
        required=True,
        example=0.006,
    )
    peak_at: List[float] = _field(
        help=(
            "Where the peak lies: the point on the mesh, in the part's frame, in mm; "
            "empty when no face had a surface to measure (`tessellated_faces`)."
        ),
        required=True,
        example=[1.95, 0.31, 6.0],
    )
    budget: float = _field(
        help="The largest peak the profile accepts, in mm.",
        required=True,
        example=0.1,
    )
    within: bool = _field(
        help=(
            "Whether the peak is within the budget, and the faces the mesher left "
            "empty (`unmeshed_area`) hold no more than a budget-wide square of surface."
        ),
        required=True,
        example=True,
    )
    heatmap: str = _field(
        help=(
            "The mesh as a PLY coloured by each vertex's largest neighbouring "
            "deviation (viridis from 0 to the budget, magenta past it), when one "
            "was asked for."
        ),
        example="parts/Screw_M4_standard.deviation.ply",
    )
    source_gap: float = _field(
        help=(
            "How far the part's own edges lie off its surfaces, in mm: the largest "
            "distance of a mesh node (on an edge's curve) from its face's surface "
            "point, the file's tolerance, which no tessellation can beat. Past the "
            "budget, the source is coarser than the profile asks."
        ),
        example=0.0,
    )


@dataclass
class CadMesh(SchemaSpec):
    """One mesh of a part, at one tessellation profile."""

    TYPED = True

    profile: str = _field(
        help="The name of the profile it was tessellated at (a `profiles` entry).",
        required=True,
        example="standard",
    )
    path: str = _field(
        help=(
            "The mesh file: Wavefront OBJ with exact surface normals and one "
            "material per colour (its MTL beside it), in mm, in the source's axes."
        ),
        required=True,
        example="parts/Screw_M4_standard.obj",
    )
    chord: float = _field(
        help=(
            "The chord it was tessellated at, in mm: the profile's, or tighter "
            "where a verified mesh had to be made again to stay within budget."
        ),
        required=True,
        example=0.02,
    )
    triangles: int = _field(help="Its triangle count.", required=True, example=164)
    vertices: int = _field(
        help=(
            "Its vertex count. Each CAD face keeps its own vertices, so a crease "
            "between faces stays sharp."
        ),
        required=True,
        example=170,
    )
    unmeshed_faces: int = _field(
        help="How many of the part's faces the mesher left without a triangle, when any.",
        example=1,
    )
    unmeshed_area: float = _field(
        help="Their area, in mm², when any: surface this mesh does not show.",
        example=1e-08,
    )
    tessellated_faces: int = _field(
        help=(
            "How many of its faces the source states as triangles alone (AP242 "
            "tessellated geometry), when any: written as the file states them, at "
            "no profile's chord, with no exact surface to measure them against."
        ),
        example=272,
    )
    deviation: CadDeviation = _field(
        help="How far it strays from the part's surfaces, when the conversion measured it.",
        nested=CadDeviation,
    )


@dataclass
class CadPart(SchemaSpec):
    """One unique part: its figures, its identity and its meshes. Placed by `nodes`."""

    TYPED = True

    id: str = _field(
        help=(
            "The part's identifier, unique in the record and file-safe: by default its "
            "part number, else its name (the conversion's `part_id` picks the key). "
            "Parts sharing one are numbered in file order."
        ),
        required=True,
        example="Screw_M4",
    )
    name: str = _field(
        help="The part's name as the source states it.",
        required=True,
        example="Screw M4",
    )
    part_number: str = _field(
        help=(
            "The id its STEP product states, the usual home of the part number "
            "(some CAD systems write the file name and keep the part number in a "
            "property); absent when blank or for IGES."
        ),
        example="91274A118",
    )
    properties: Dict[str, Union[str, float, int]] = _field(
        help=(
            "The attributes its CAD system wrote for it (STEP user-defined "
            "attributes), keyed as written: a part number, revision, material, "
            "mass. A piece split from a product holds the product's."
        ),
        example={"Part Number": "HX-100", "Revision": "C"},
    )
    fingerprint: str = _field(
        help=(
            "A hash of its shape in its own frame (volume, centroid, inertia, scaled "
            "by its size), an exact-match key: the same for its geometry read again, "
            "renamed or in another unit, different for a mirrored copy. Geometry "
            "re-exported through another system can hash apart; match that within a "
            "tolerance on `size`, `area` and `volume`. Absent for a part with no surface."
        ),
        example="9f3c2a71d04be655",
    )
    size: List[float] = _field(
        help="The size of the tight bounding box of its exact geometry along each axis, in mm.",
        required=True,
        example=[4.0, 4.0, 12.0],
    )
    area: float = _field(
        help=(
            "Its surface area in mm², integrated over its exact surfaces (closed form "
            "on analytic faces; on NIST's AP242 models within 4e-5 of a 0.002 mm mesh, "
            "and 0.9% low on a deliberately wavy freeform panel)."
        ),
        required=True,
        example=175.93,
    )
    volume: float = _field(
        help="Its volume in mm³, integrated as its area is (0 for a part with no closed solid).",
        required=True,
        example=150.8,
    )
    color: List[float] = _field(
        help="The part's own colour, when the source states one (a face may state its own).",
        example=[0.8, 0.1, 0.1, 1.0],
    )
    meshes: List[CadMesh] = _field(
        help="Its meshes, one per profile.",
        default_factory=list,
        nested=[CadMesh],
    )


@dataclass
class CadNode(SchemaSpec):
    """One node of the assembly tree: an assembly, or a placement of a part."""

    TYPED = True

    name: str = _field(
        help="The node's name: the instance's own, else the part's or assembly's.",
        required=True,
        example="Screw_M4_1",
    )
    parent: Optional[int] = _field(
        help="The index of its parent in `nodes`, or null for a root.",
        required=True,
        example=0,
    )
    matrix: List[float] = _field(
        help="Its placement in its parent: 16 numbers, row-major, translation in mm.",
        required=True,
        example=[1, 0, 0, 5, 0, 1, 0, 5, 0, 0, 1, 10, 0, 0, 0, 1],
    )
    part: str = _field(
        help="The `id` of the part it places, or '' for an assembly.",
        required=True,
        example="Screw_M4",
    )
    color: List[float] = _field(
        help="A colour the source states on this placement alone, over the part's.",
        example=[0.1, 0.1, 0.1, 1.0],
    )


@dataclass
class CadProfile(SchemaSpec):
    """A tessellation profile, as the conversion used it."""

    TYPED = True

    name: str = _field(help="The profile's name.", required=True, example="standard")
    chord: float = _field(
        help=(
            "The mesher's deflection target, in mm: how far its nodes and edges "
            "may leave the surface. Inside a triangle a surface can stray further; "
            "the budget is what a verified mesh is held to."
        ),
        required=True,
        example=0.02,
    )
    angle: float = _field(
        help="The largest angle between neighbouring normals along a curve, in degrees.",
        required=True,
        example=8.0,
    )
    budget: float = _field(
        help="The largest deviation from the exact surfaces a verified mesh may have, in mm.",
        required=True,
        example=0.02,
    )


@dataclass
class CadSource(SchemaSpec):
    """The file the record was made from."""

    TYPED = True

    file: str = _field(
        help="Its file name, without a folder.", required=True, example="product.step"
    )
    sha256: str = _field(
        help="The SHA-256 of its bytes: a new revision of the file has another.",
        required=True,
        example="3f2a...",
    )
    format: str = _field(
        help="Its format.", required=True, choices=("step", "iges"), example="step"
    )
    unit: str = _field(
        help="The unit every length in the record is in.",
        required=True,
        choices=("mm",),
        example="mm",
    )
    up_axis: str = _field(
        help=(
            "The axis that points up in the source, as the conversion was told: "
            "the GLBs turn it to glTF's +Y."
        ),
        required=True,
        choices=("z", "y"),
        example="z",
    )


@dataclass
class CadRecord(SchemaSpec):
    """What a CAD import read from one file and wrote from it."""

    TYPED = True

    #: The record's schema version.
    VERSION = 1

    version: int = _field(help="The record's schema version.", required=True, example=1)
    source: CadSource = _field(
        help="The file it was made from.", required=True, nested=CadSource
    )
    profiles: List[CadProfile] = _field(
        help="The profiles it tessellated at.",
        required=True,
        default_factory=list,
        nested=[CadProfile],
    )
    nodes: List[CadNode] = _field(
        help="The assembly tree, parents before children.",
        required=True,
        default_factory=list,
        nested=[CadNode],
    )
    parts: List[CadPart] = _field(
        help="The unique parts, each tessellated once however often it is placed.",
        required=True,
        default_factory=list,
        nested=[CadPart],
    )
    assemblies: Dict[str, str] = _field(
        help=(
            "The assembled product per profile: a GLB holding the tree, names, "
            "colours and every placement, in metres and +Y up as glTF states them."
        ),
        default_factory=dict,
        example={"standard": "product_standard.glb"},
    )
