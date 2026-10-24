# !/usr/bin/python
# coding=utf-8
"""The ``shadows`` record's payload and its GLB block, declared once.

The two shapes of :attr:`pythontk.SceneRecords.SHADOWS`:

- :class:`ShadowRecord` -- what both DCCs' ``ShadowRig.export_record``
  publishes on the ``data_export`` carrier (one ``ShadowRig.plane_record``
  per plane), and what unitytk's ``ShadowPlaneController`` reads from an FBX:
  each plane joined to its nodes and images by NAME.
- :class:`ShadowBlock` -- the ``extras.shadows`` block
  ``MeshConvert.apply_glb_shadows`` derives from it for a GLB: the same
  planes, each bound to its glTF nodes and textures by INDEX instead, the
  horizon map's rect flipped to glTF's top-left origin.

A plane's projection model (:class:`ShadowPlaneModel`) and a horizon map's
layout (:class:`ShadowHorizonLayout`) are declared once and shared; each
carrier adds only how it addresses the nodes and images -- names in the
record, indices in the block, never both. A packed plane's atlas tile is the
quad's own UV bounds, so the block states none.

They are typed :class:`~pythontk.SchemaSpec` declarations, so this one
definition validates a payload (``ShadowRecord.validate(payload)``),
publishes its JSON Schema, and is what ``m3trik/scripts/sync_scene_records.py``
generates the WebXR runtime's JSDoc typedefs and unitytk's C# record types
from -- a field renamed here fails their checks until they are regenerated.
Every plane field is required: both rigs write each one. Change a field
together with its producers (mayatk's and blendertk's
``ShadowRig.plane_record``, ``ShadowHorizon.record`` for the horizon block,
``MeshConvert.apply_glb_shadows`` for the GLB) and bump the record's declared
version.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from pythontk.core_utils.schema_spec import SchemaSpec

from pythontk.geo_utils.shadow_projection import ShadowConformance, ShadowProjection

_field = SchemaSpec.spec_field

#: A plane's rig type: a silhouette drawn for one source position, or a
#: height-field map a shader marches for any (the DCC rigs' ``RIG_TYPES``).
RIG_TYPES: Tuple[str, ...] = ("projected", "horizon")
#: How a plane's source shines: from its position, or along one direction.
SOURCE_TYPES: Tuple[str, ...] = ("point", "directional")

_Vec3 = Tuple[float, float, float]
_Vec4 = Tuple[float, float, float, float]

# The help a rect's spelling in the record and in the block share; its V
# origin is the one thing that differs (the GLB pass flips it).
_MAP_RECT = (
    "The map's exact rect in its image (never inset), [scale_u, scale_v, "
    "offset_u, offset_v], origin {origin}."
)


@dataclass
class ShadowPlaneAtlas(SchemaSpec):
    """Where a plane's silhouette sits in the shared silhouette atlas."""

    TYPED = True

    texture: str = _field(help="The atlas image's file name.", required=True)
    rect: _Vec4 = _field(
        help=(
            "The plane's tile in the atlas, [scale_u, scale_v, offset_u, "
            "offset_v] (uv' = uv * scale + offset), origin bottom-left: the "
            "bounds the quad's UVs already address."
        ),
        required=True,
    )


@dataclass
class ShadowHorizonLayout(SchemaSpec):
    """A horizon map's layout: the footprint it covers and how its one image
    is tiled (pythontk ``HeightFieldMap.to_rgba``)."""

    TYPED = True

    size: int = _field(
        help=(
            "Footprint pixels per side, S (a power of two); the image is S tall "
            "and S x (spans + 2) wide."
        ),
        required=True,
    )
    spans: int = _field(help="Solid spans kept per pixel column, K.", required=True)
    bounds: _Vec4 = _field(
        help=(
            "The footprint the map covers, [a0, a1, b0, b1] along frame_a and "
            "frame_b from the contact node, in the record's unit."
        ),
        required=True,
    )
    height_scale: float = _field(
        help=(
            "The height above the ground a 16-bit channel value of 65535 "
            "stands for, in the record's unit."
        ),
        required=True,
    )
    frame_a: _Vec3 = _field(
        help=(
            "The footprint's first horizontal axis, a unit vector in the "
            "contact node's local frame as exported (FBX/glTF axes, +Y up)."
        ),
        required=True,
    )
    frame_b: _Vec3 = _field(
        help="The footprint's second horizontal axis, a unit vector in the same frame.",
        required=True,
    )


@dataclass
class ShadowPlaneHorizon(ShadowHorizonLayout):
    """A horizon plane's height-field shadow map, by file name, and where it
    sits in its image."""

    texture: str = _field(
        help="The map image's file name: the shared horizon atlas once packed.",
        required=True,
    )
    rect: _Vec4 = _field(help=_MAP_RECT.format(origin="bottom-left"), required=True)


@dataclass
class ShadowPlaneModel(SchemaSpec):
    """What a plane shows and how it is placed: its type, its source, and the
    projection model's inputs (pythontk ``ShadowProjection.model``), the same
    on every carrier."""

    TYPED = True

    type: str = _field(
        help=(
            "projected: a silhouette drawn for one source position, re-placed "
            "as the source moves; horizon: a height-field map a shader "
            "marches for any source position."
        ),
        required=True,
        choices=RIG_TYPES,
        default=RIG_TYPES[0],
    )
    intensity: float = _field(
        help=(
            "The shadow's overall darkness, 0 to 1, which the plane's opacity "
            "never exceeds: a reader draws the lesser of the two, so a file "
            "whose material states no opacity draws this."
        ),
        required=True,
        default=1.0,
    )
    source_type: str = _field(
        help=(
            "point: the source shines from its position; directional: along "
            "one direction, wherever it stands (a sun)."
        ),
        required=True,
        choices=SOURCE_TYPES,
        default=SOURCE_TYPES[0],
    )
    source_size: float = _field(
        help=(
            "A positional source's diameter, in the record's unit, which softens "
            "the shadow's edge (0 for a sharp edge, and for a directional source)."
        ),
        required=True,
        default=0.0,
    )
    source_angle: float = _field(
        help=(
            "A directional source's angular diameter, in radians (0 for a "
            "sharp edge, and for a positional source)."
        ),
        required=True,
        default=0.0,
    )
    follow_source: bool = _field(
        help=(
            "Whether a runtime re-places the plane from its source and contact "
            "every frame (both must resolve); false plays the imported animation."
        ),
        required=True,
        default=False,
    )
    ground: float = _field(
        help="The ground plane's height along the up axis, in the record's unit.",
        required=True,
        default=0.0,
    )
    # radius and height default as ShadowProjection.model's own parameters do.
    radius: float = _field(
        help=(
            "The target's footprint radius (half its footprint's diagonal), in "
            "the record's unit."
        ),
        required=True,
        default=0.5,
    )
    height: float = _field(
        help="The target's height above its contact, in the record's unit.",
        required=True,
        default=1.0,
    )
    max_stretch: float = _field(
        help=(
            "The cap on the shadow's length past the target, in target heights "
            "(6 is the length a sun 9.5 degrees up casts)."
        ),
        required=True,
        default=ShadowProjection.DEFAULT_MAX_STRETCH,
    )
    canvas: _Vec4 = _field(
        help=(
            "Where the texture's edges sit around the projected shadow, [u0, "
            "u1, w0, w1]: u1 the far edge in top-disk radii past where the "
            "target's top lands, u0 the near edge as a fraction of that far "
            "edge's distance from the anchor, w0 and w1 the sides as fractions "
            "of the shadow's width."
        ),
        required=True,
        default_factory=lambda: list(ShadowConformance.CANVAS),
    )


@dataclass
class ShadowPlane(ShadowPlaneModel):
    """One shadow plane of the record, joined to its nodes and images by name."""

    name: str = _field(
        help=(
            "The plane mesh node's name, namespace included: the join key to "
            "the imported plane (where an import drops namespaces, the part "
            "after the last ':' joins)."
        ),
        required=True,
    )
    texture: str = _field(
        help=(
            "The file name of the plane's own silhouette image (the shadow in "
            "its alpha), or an empty string when it has none."
        ),
        required=True,
    )
    source: str = _field(
        help=(
            "The name of the node the shadow is cast from, or an empty string "
            "when it has none."
        ),
        required=True,
        default="",
    )
    contact: str = _field(
        help=(
            "The name of the node the shadow is anchored to, at the footprint's "
            "centre on the target's underside, or an empty string when it has none."
        ),
        required=True,
        default="",
    )
    atlas: ShadowPlaneAtlas = _field(
        help=(
            "Where the silhouette sits in the shared silhouette atlas; absent "
            "when the plane is not packed."
        ),
        nested=ShadowPlaneAtlas,
    )
    horizon: ShadowPlaneHorizon = _field(
        help=(
            "The height-field map a horizon plane is shaded from; absent on a "
            "projected plane and on a horizon plane not yet baked."
        ),
        nested=ShadowPlaneHorizon,
    )


@dataclass
class ShadowRecord(SchemaSpec):
    """The ``shadows`` record's payload: the export's shadow planes, and the
    unit their lengths are in."""

    TYPED = True
    #: A plane's rig types (:data:`RIG_TYPES`), on the root-exposed class the
    #: DCC rigs read them from (``ptk.ShadowRecord.RIG_TYPES``). A class
    #: constant, never a field: it carries no annotation.
    RIG_TYPES = RIG_TYPES

    version: int = _field(
        help="The record's schema version (SceneRecords.SHADOWS.version).",
        required=True,
    )
    unit_scale: float = _field(
        help="Metres per unit of every length in the record (0.01 for centimetres).",
        required=True,
    )
    planes: List[ShadowPlane] = _field(
        help="The export's shadow planes, one entry each.",
        required=True,
        nested=[ShadowPlane],
        default_factory=list,
    )


# ------------------------------------------------------------- the GLB block
@dataclass
class ShadowBlockHorizon(ShadowHorizonLayout):
    """A horizon map bound to the file."""

    rect: _Vec4 = _field(help=_MAP_RECT.format(origin="top-left (glTF)"), required=True)
    texture_index: int = _field(
        help=(
            "The glTF texture index of the map, bound with no mipmaps and "
            "clamped: its texels are data, fetched whole, never colour-decoded."
        ),
        required=True,
    )


@dataclass
class ShadowBlockPlane(ShadowPlaneModel):
    """A plane bound to the file: its node, its source and contact nodes, and
    the textures it samples, by index."""

    node: int = _field(help="The plane mesh's glTF node index.", required=True)
    source_node: Optional[int] = _field(
        help=(
            "The source's glTF node index, or null when the file holds no "
            "single node of that name."
        ),
        required=True,
    )
    contact_node: Optional[int] = _field(
        help=(
            "The contact's glTF node index, or null when the file holds no "
            "single node of that name."
        ),
        required=True,
    )
    texture_index: Optional[int] = _field(
        help=(
            "The glTF texture index of a projected plane's colour map (its "
            "atlas once packed, which the quad's UVs then address), or null on "
            "a horizon plane, which the march shades."
        ),
        required=True,
    )
    horizon: ShadowBlockHorizon = _field(
        help=(
            "The height-field map the plane is shaded from; present on every "
            "horizon plane the block carries."
        ),
        nested=ShadowBlockHorizon,
    )


@dataclass
class ShadowBlock(ShadowRecord):
    """The ``extras.shadows`` block of a GLB: the record's planes, each bound
    to the file's glTF nodes and textures."""

    version: int = _field(
        help=(
            "The block's own schema version (SceneRecords.SHADOWS.glb.version), "
            "apart from the record's."
        ),
        required=True,
    )
    planes: List[ShadowBlockPlane] = _field(
        help=(
            "One entry per plane mesh node in the file (a name several nodes "
            "carry gives one each); a plane the file lacks, or cannot bind, is "
            "left out."
        ),
        required=True,
        nested=[ShadowBlockPlane],
        default_factory=list,
    )
