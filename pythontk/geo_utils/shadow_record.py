# !/usr/bin/python
# coding=utf-8
"""The ``shadows`` record's payload, declared once.

The one shape of :attr:`pythontk.SceneRecords.SHADOWS`, in both containers:
what both DCCs' ``ShadowRig.export_record`` publishes on the ``data_export``
carrier (one ``ShadowRig.plane_record`` per plane), and what a GLB carries as
it stands in its root ``extras`` (one layout, two containers). Each plane is
joined to its nodes and images by NAME in either; a GLB's images ride as
media (:attr:`pythontk.SceneRecords.GLB_MEDIA`).

Version 3: every plane carries its horizon map (:class:`ShadowPlaneHorizon`),
which a shader marches for any source position, the canvas a placement
spans, and the terms of its fade (``falloff_power``, ``fade_height``,
``source_ramp``), which every reader evaluates each frame
(``ShadowProjection.fade``). The projected silhouette is an optional
extension (:class:`ShadowPlaneSilhouette`, read by the GLB's readers alone);
a reader without it draws the horizon.

They are typed :class:`~pythontk.SchemaSpec` declarations, so this one
definition validates a payload (``ShadowRecord.validate(payload)``),
publishes its JSON Schema, and is what ``m3trik/scripts/sync_scene_records.py``
generates the WebXR runtime's JSDoc typedefs and unitytk's C# record types
from -- a field renamed here fails their checks until they are regenerated.
Every plane field is required unless it says otherwise: both rigs write each
one. Change a field together with its producers (mayatk's and blendertk's
``ShadowRig.plane_record``, ``ShadowHorizon.record`` for the horizon block)
and bump the record's declared version.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pythontk.core_utils.schema_spec import SchemaSpec

from pythontk.geo_utils.shadow_projection import ShadowConformance, ShadowProjection

_field = SchemaSpec.spec_field

#: A rig's type, an authoring choice of the producers (their ``RIG_TYPES``),
#: never a record field: ``projected`` draws its silhouette (the plane's
#: ``silhouette.draw``), ``horizon`` keeps it as its march's fallback. Every
#: plane of either carries its horizon map.
RIG_TYPES: Tuple[str, ...] = ("projected", "horizon")
#: How a plane's source shines: from its position, or along one direction.
SOURCE_TYPES: Tuple[str, ...] = ("point", "directional")

_Vec3 = Tuple[float, float, float]
_Vec4 = Tuple[float, float, float, float]


class _ShadowRecordInternal:
    """Field checks the declarations below share."""

    @staticmethod
    def positive(value: Any) -> List[str]:
        """A length a reader divides by: a number over zero."""
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        ):
            return []
        return [f"must be a number over 0, got {value!r}"]


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
    """A plane's height-field shadow map, by file name, and where it sits in
    its image."""

    texture: str = _field(
        help="The map image's file name: the shared horizon atlas once packed.",
        required=True,
    )
    rect: _Vec4 = _field(
        help=(
            "The map's exact rect in its image (never inset), [scale_u, "
            "scale_v, offset_u, offset_v], origin bottom-left."
        ),
        required=True,
    )


@dataclass
class ShadowPlaneSilhouette(SchemaSpec):
    """A plane's projected silhouette: the shadow drawn for one source
    position, re-placed as the source moves -- an extension, read by the
    GLB's readers alone."""

    TYPED = True

    texture: str = _field(
        help="The file name of the plane's own silhouette image (the shadow in its alpha).",
        required=True,
    )
    atlas: ShadowPlaneAtlas = _field(
        help=(
            "Where the silhouette sits in the shared silhouette atlas; absent "
            "when the plane is not packed."
        ),
        nested=ShadowPlaneAtlas,
    )
    draw: bool = _field(
        help=(
            "Whether a reader holding the silhouette draws it (true), or "
            "draws the horizon and keeps the silhouette as the fallback where "
            "it cannot march (false): the rig's authored choice."
        ),
        required=True,
        default=True,
    )


@dataclass
class ShadowPlane(SchemaSpec):
    """One shadow plane of the record: its nodes by name, its horizon map,
    the placement's inputs (pythontk ``ShadowProjection.model``) and its
    fade's terms (``ShadowProjection.fade``)."""

    TYPED = True

    name: str = _field(
        help=(
            "The plane mesh node's name, namespace included: the join key to "
            "the imported plane (where an import drops namespaces, the part "
            "after the last ':' joins)."
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
    intensity: float = _field(
        help="The shadow's overall darkness, 0 to 1: the fade's ceiling.",
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
            "Where the quad's edges sit around the projected shadow, [u0, u1, "
            "w0, w1]: u1 the far edge in top-disk radii past where the "
            "target's top lands, u0 the near edge as a fraction of that far "
            "edge's distance from the anchor, w0 and w1 the sides as fractions "
            "of the shadow's width -- what a follow_source placement spans, "
            "and the edges the silhouette was drawn to."
        ),
        required=True,
        default_factory=lambda: list(ShadowConformance.CANVAS),
    )
    falloff_power: float = _field(
        help=(
            "How fast the shadow lightens as it stretches past its footprint: "
            "the fade's exponent on the stretch."
        ),
        required=True,
        default=ShadowProjection.DEFAULT_FALLOFF_POWER,
    )
    # Twice the default height: the rigs' own default (2 x the target's height).
    fade_height: float = _field(
        help=(
            "The target's rise above the ground at which its shadow has faded "
            "out, in the record's unit."
        ),
        required=True,
        default=2.0,
    )
    source_ramp: float = _field(
        help=(
            "The source's height over the contact at which the shadow is all "
            "in, in the record's unit: the fade ramps in from none with the "
            "source at the contact's height -- the ramp the plane's producer "
            "previews (one unit of its rig: 0.01 in a metre record of a "
            "centimetre rig); over 0."
        ),
        required=True,
        default=1.0,
        validate=_ShadowRecordInternal.positive,
    )
    horizon: ShadowPlaneHorizon = _field(
        help="The height-field map the plane's shadow is marched from.",
        required=True,
        nested=ShadowPlaneHorizon,
    )
    silhouette: ShadowPlaneSilhouette = _field(
        help=(
            "The plane's projected silhouette; absent when it has none. A "
            "reader without it draws the horizon."
        ),
        nested=ShadowPlaneSilhouette,
        readers=("glb",),
    )


@dataclass
class ShadowRecord(SchemaSpec):
    """The ``shadows`` record's payload: the export's shadow planes, and the
    unit their lengths are in."""

    TYPED = True
    #: The rigs' types (:data:`RIG_TYPES`), on the root-exposed class the DCC
    #: rigs read them from (``ptk.ShadowRecord.RIG_TYPES``). A class constant,
    #: never a field: it carries no annotation.
    RIG_TYPES = RIG_TYPES

    version: int = _field(
        help=(
            "The record's schema version (SceneRecords.SHADOWS.version); a "
            "reader reads the version it knows and refuses any other -- a "
            "version 2 record (a plane's 'type', its silhouette's image on "
            "the plane, no fade terms) is re-exported."
        ),
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

    @staticmethod
    def fade(
        plane: Dict[str, Any],
        unit_scale: float,
        contact: Optional[Sequence[float]] = None,
        light: Optional[Sequence[float]] = None,
        *,
        direction: Optional[Sequence[float]] = None,
        up: int = 1,
    ) -> float:
        """The fade one plane of the record draws with its contact at
        *contact* and its source at *light* (or shining along *direction*):
        ``ShadowProjection.fade`` over the plane's terms, every length in the
        record's unit times *unit_scale* -- what each reader evaluates every
        frame. A plane whose contact or source does not resolve (*contact*
        None, or neither *light* nor *direction*) draws at its intensity.

        Parameters:
            plane: One entry of the record's ``planes``.
            unit_scale: The record's metres per unit.
            contact, light: Points in the record's unit, *up* vertical.
            direction: The way a directional source shines (any length).
            up: The vertical axis of the points (1 Y-up, 2 Z-up).

        Returns:
            The fade, ``0 <= fade <= intensity``.
        """
        intensity = float(plane.get("intensity", 1.0))
        if contact is None or (light is None and direction is None):
            return intensity
        m = float(unit_scale)
        height = float(plane.get("height", 1.0))
        return ShadowProjection.fade(
            [float(v) * m for v in contact],
            None if light is None else [float(v) * m for v in light],
            float(plane.get("ground", 0.0)) * m,
            float(plane.get("radius", 0.5)) * m,
            height * m,
            up=up,
            direction=direction,
            max_stretch=float(
                plane.get("max_stretch", ShadowProjection.DEFAULT_MAX_STRETCH)
            ),
            intensity=intensity,
            falloff_power=float(
                plane.get("falloff_power", ShadowProjection.DEFAULT_FALLOFF_POWER)
            ),
            fade_height=float(plane.get("fade_height", 2.0 * height)) * m,
            source_ramp=float(plane.get("source_ramp", 1.0)) * m,
        )

    @staticmethod
    def silhouette(
        texture: str,
        draw: bool,
        atlas: str = "",
        atlas_rect: Sequence[float] = (1.0, 1.0, 0.0, 0.0),
    ) -> Dict[str, Any]:
        """A plane's ``silhouette`` (:class:`ShadowPlaneSilhouette`), built
        once for both rigs' ``plane_record``, as ``ShadowHorizon.record``
        builds its ``horizon``.

        Parameters:
            texture: The silhouette image's path or file name; empty when the
                plane has none.
            draw: Whether a reader holding it draws it: the rig's type is
                ``projected``.
            atlas: The silhouette atlas's file name while the plane is packed.
            atlas_rect: Its tile in that atlas, ``(su, sv, ou, ov)``.

        Returns:
            The block, or ``{}`` for a plane with no silhouette image.
        """
        name = str(texture or "").replace("\\", "/").rsplit("/", 1)[-1]
        if not name:
            return {}
        out: Dict[str, Any] = {"texture": name, "draw": bool(draw)}
        if atlas:
            out["atlas"] = {
                "texture": str(atlas),
                "rect": [round(float(v), 6) for v in atlas_rect],
            }
        return out
