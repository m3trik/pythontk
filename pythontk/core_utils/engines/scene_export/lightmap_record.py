# !/usr/bin/python
# coding=utf-8
"""The ``lightmaps`` record and its GLB block: what a bake publishes, declared once.

Both hosts' bakers publish the record (mayatk's and blendertk's
``LightmapRecords``) and every reader takes it (the GLB build, unitytk's
importer). :meth:`LightmapRecord.payload` is what a producer publishes; a
reader reads the current schema alone (``SceneRecords.LIGHTMAPS.is_current``).

Both shapes are typed :class:`~pythontk.SchemaSpec` declarations, so this one
definition validates a payload (``LightmapRecord.validate(payload)``),
publishes its JSON Schema, and is what ``m3trik/scripts/sync_scene_records.py``
generates the other languages' types from:

- :class:`LightmapRecord` -- the record on a deliverable's ``data_export``
  carrier: each baked object's lightmap, the reflection probe the bake
  captured and the lights it was baked from, every length at one
  ``unit_scale``.
- :class:`LightmapsBlock` -- the ``extras.lightmaps`` block a GLB carries
  (``MeshConvert.lightmaps_block``): each lightmapped material's intensity,
  the texture slot its map rides in, and the probe embedded in the file's
  buffer, in metres.

Change a field together with its producers, and bump the version it is
declared under (``SceneRecords.LIGHTMAPS.version``; the block's
``SceneRecords.LIGHTMAPS.glb.version``).
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.core_utils.schema_spec import SchemaSpec

_field = SchemaSpec.spec_field
_Vec3 = Tuple[float, float, float]


# ------------------------------------------------------------------ the record
@dataclass
class LightmapBox(SchemaSpec):
    """An axis-aligned box, by its least and greatest corners."""

    TYPED = True

    min: _Vec3 = _field(
        help="The least corner (x, y, z), in the space and unit of the probe holding the box.",
        required=True,
    )
    max: _Vec3 = _field(
        help="The greatest corner (x, y, z), in the same space and unit.",
        required=True,
    )


@dataclass
class LightmapNode(SchemaSpec):
    """A mesh node, named as an ``objects`` entry names its object."""

    TYPED = True

    name: str = _field(
        help="The node's name as the file spells it, namespace included.",
        required=True,
    )
    hierarchy: List[str] = _field(
        help="Its path from the scene root: each name from the root down, its own last.",
        required=True,
        default_factory=list,
    )


@dataclass
class LightmapHighlight(SchemaSpec):
    """A copy of the probe with the bake's lights drawn in, for one group of
    receivers: each light where box projection shows it true for them."""

    TYPED = True

    map: str = _field(
        help=(
            "The copy's file name, no folder: a Radiance (.hdr) layer in the "
            "probe's layout and units, added to the probe (box-filtered down to "
            "the layer's size) and prefiltered as the probe is."
        ),
        required=True,
    )
    objects: List[LightmapNode] = _field(
        help=(
            "The mesh nodes that reflect the probe plus this layer in the "
            "probe's place, box-projected from its position and box; a node is "
            "under one copy at most."
        ),
        required=True,
        nested=[LightmapNode],
        default_factory=list,
    )


@dataclass
class LightmapProbe(SchemaSpec):
    """The reflection probe the bake captured: the room it lit, as an HDR seen
    from one point -- the reflections a lightmap cannot hold."""

    TYPED = True

    map: str = _field(
        help=(
            "The probe image's file name, no folder: an equirectangular linear "
            "HDR (top row +Y) the deliverable carries under that name."
        ),
        required=True,
    )
    position: _Vec3 = _field(
        help=(
            "Where it was captured, in the file's world axes (right-handed, +Y "
            "up); times unit_scale, metres."
        ),
        required=True,
    )
    box: Optional[LightmapBox] = _field(
        help=(
            "{min, max}: the box its reflections are projected onto, in the same "
            "space, a face open to the outside pushed far away; null for an open "
            "scene, whose probe is read as distant."
        ),
        required=True,
    )
    highlights: List[LightmapHighlight] = _field(
        help=(
            "Copies of the probe with the bake's lights drawn in, one per group "
            "of receivers -- a reader that uses them lights none of 'lights', or "
            "the highlights count twice; absent when the export drew none."
        ),
        nested=[LightmapHighlight],
    )


@dataclass
class LightmapEntry(SchemaSpec):
    """One baked object: the node it lights and how its lightmap is read."""

    TYPED = True

    name: str = _field(
        help=(
            "The object's node name as the file spells it, namespace included "
            "('NS:leaf'): the join key to the imported node, which a reader whose "
            "import drops the namespace matches by the name after the last ':'."
        ),
        required=True,
    )
    map: str = _field(
        help=(
            "The lightmap image's file name, no folder; the deliverable carries "
            "the image under that name (an FBX embeds it)."
        ),
        required=True,
    )
    uvIndex: int = _field(
        help="The UV set the map is read through, 0-based: 1 is the second set.",
        required=True,
        default=1,
    )
    scaleOffset: Tuple[float, float, float, float] = _field(
        help=(
            "Its rect in the map, [scaleX, scaleY, offsetX, offsetY]: read at "
            "uv * [scaleX, scaleY] + [offsetX, offsetY], [1, 1, 0, 0] for a map "
            "of its own."
        ),
        required=True,
        default=(1.0, 1.0, 0.0, 0.0),
    )
    hierarchy: List[str] = _field(
        help=(
            "Its node's path from the scene root, each name from the root down "
            "and its own last, namespaces kept: what tells apart objects that "
            "share a name."
        ),
        required=True,
        default_factory=list,
    )


@dataclass
class LightmapBakedLight(SchemaSpec):
    """A light the maps were baked from, for its highlight alone: its diffuse
    light is already in the maps, so a reader that lights it lights its
    specular only, or the scene is lit twice."""

    TYPED = True
    #: The kinds of light a record carries (mayatk ``LightUtils.SPECULAR_KINDS``).
    TYPES = ("point", "spot", "rect")

    type: str = _field(
        help="What kind it is: 'point', 'spot' or 'rect' (a rectangular area light).",
        required=True,
        choices=TYPES,
    )
    position: _Vec3 = _field(
        help=(
            "Where it is, in the file's world axes (right-handed, +Y up); times "
            "unit_scale, metres."
        ),
        required=True,
    )
    direction: _Vec3 = _field(
        help="The unit vector a spot or rect faces, world; absent on a point light."
    )
    right: _Vec3 = _field(
        help="A rect's width axis, a unit vector, world; a rect's alone."
    )
    color: _Vec3 = _field(help="Its colour, linear Rec.709 RGB.", required=True)
    intensity: float = _field(
        help=(
            "A point's or spot's radiant intensity, the irradiance it gives one "
            "unit away (times unit_scale squared: one metre away); a rect's "
            "radiance, which no unit changes."
        ),
        required=True,
    )
    angle: float = _field(
        help="A spot's half angle to the outer edge of its cone, in radians, at most pi/2."
    )
    penumbra: float = _field(
        help="The share of a spot's angle its edge fades over, 0 to 1."
    )
    radius: float = _field(
        help="A point's or spot's source radius, in the record's unit; 0 for a true point."
    )
    width: float = _field(help="A rect's width along 'right', in the record's unit.")
    height: float = _field(help="A rect's height, in the record's unit.")


@dataclass
class LightmapRecord(SchemaSpec):
    """The ``lightmaps`` record: each baked object's lightmap, the reflection
    probe the bake captured and the lights it was baked from.

    Its builder too (:meth:`payload`, what both hosts' producers publish
    through), each thing stated once: one ``unit_scale`` -- metres per unit
    of every length the record holds, the probe's and each light's alike;
    the probe's ``box`` as ``{min, max}``, the shape the GLB carries it in;
    and no ``direction`` on a point light, which has none.
    """

    TYPED = True
    #: The light types without a direction -- the one field dropped is the
    #: one known to mean nothing, so a type added later keeps its own.
    UNAIMED = frozenset({"point"})

    version: int = _field(
        help=(
            "The record's schema version (SceneRecords.LIGHTMAPS.version); a "
            "reader reads the version it knows and refuses any other."
        ),
        required=True,
    )
    unit_scale: float = _field(
        help=(
            "Metres per unit of every length in the record -- the probe's "
            "position and box, each light's position, size and radius; present "
            "exactly when 'probe' or 'lights' is."
        ),
    )
    objects: List[LightmapEntry] = _field(
        help=(
            "Each baked object, one entry per node; never empty, as a scene with "
            "nothing baked carries no record."
        ),
        required=True,
        nested=[LightmapEntry],
        default_factory=list,
    )
    probe: LightmapProbe = _field(
        help=(
            "The reflection probe the bake captured, the scene's environment; "
            "absent when it captured none."
        ),
        nested=LightmapProbe,
    )
    lights: List[LightmapBakedLight] = _field(
        help=(
            "The lights the maps were baked from, for their highlights alone; "
            "absent when none ships."
        ),
        nested=[LightmapBakedLight],
    )

    @classmethod
    def payload(
        cls,
        objects: Iterable[Mapping[str, Any]],
        probe: Optional[Mapping[str, Any]] = None,
        lights: Optional[Iterable[Mapping[str, Any]]] = None,
        unit_scale: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """The record's payload, version stamped; ``None`` when nothing is baked.

        Parameters:
            objects: Each baked object's entry (``name``, ``hierarchy``,
                ``map``, ``uvIndex``, ``scaleOffset``).
            probe: The bake's reflection probe (``map``, ``position``, ``box``
                as ``{min, max}`` or the host record's ``[min, max]``,
                ``highlights``), in the unit *unit_scale* measures.
            lights: The lights the maps were baked from (the host's specular
                lights), in that unit.
            unit_scale: Metres per unit of every length in *probe* and
                *lights*; required with either.

        Raises:
            ValueError: A probe or lights without a positive *unit_scale*.
        """
        entries = [dict(entry) for entry in objects or ()]
        if not entries:
            return None
        lights = [cls._light(light) for light in lights or ()]
        body: Dict[str, Any] = {}
        if probe or lights:
            try:
                unit = float(unit_scale)
            except (TypeError, ValueError):
                unit = 0.0
            if unit <= 0.0:
                raise ValueError(
                    "A lightmap record's probe and lights need the unit they are "
                    f"in: unit_scale {unit_scale!r} is not metres per unit."
                )
            body["unit_scale"] = unit
        body["objects"] = entries
        if probe:
            body["probe"] = cls._probe(probe)
        if lights:
            body["lights"] = lights
        return SceneRecords.LIGHTMAPS.make(body).payload

    @classmethod
    def box(cls, box: Any) -> Optional[Dict[str, List[float]]]:
        """*box* as the record states it, ``{"min", "max"}``: from that shape
        or from the ``[min, max]`` pair of points a host's probe record keeps;
        ``None`` for an open scene or no box."""
        if isinstance(box, Mapping):
            box = (box.get("min"), box.get("max"))
        if not (isinstance(box, (list, tuple)) and len(box) == 2):
            return None
        corners = [cls._point(corner) for corner in box]
        if any(corner is None for corner in corners):
            return None
        return {"min": corners[0], "max": corners[1]}

    @classmethod
    def _probe(cls, probe: Mapping[str, Any]) -> Dict[str, Any]:
        """*probe* in the current shape: its box ``{min, max}``, no unit of
        its own (the record states it)."""
        out = {k: v for k, v in probe.items() if k != "unit_scale"}
        if "box" in out:
            out["box"] = cls.box(out["box"])
        if "position" in out:
            out["position"] = cls._point(out["position"]) or out["position"]
        return out

    @classmethod
    def _light(cls, light: Mapping[str, Any]) -> Dict[str, Any]:
        """*light* in the current shape: no unit of its own, no direction
        where its type has none (:attr:`UNAIMED`), every vector free of
        negative zeros."""
        out = {k: v for k, v in light.items() if k != "unit_scale"}
        if out.get("type") in cls.UNAIMED:
            out.pop("direction", None)
            out.pop("right", None)
        for key in ("position", "direction", "right"):
            if key in out:
                out[key] = cls._point(out[key]) or out[key]
        return out

    @staticmethod
    def _point(value: Any) -> Optional[List[float]]:
        """*value* as three floats, ``-0.0`` read as ``0.0``; ``None`` when it
        is not three numbers."""
        if not (isinstance(value, (list, tuple)) and len(value) == 3):
            return None
        try:
            return [float(c) + 0.0 for c in value]
        except (TypeError, ValueError):
            return None


# --------------------------------------------------------------- the GLB block
@dataclass
class LightmapsBlockMaterial(SchemaSpec):
    """One lightmapped material: what its lightmap's texels are multiplied by."""

    TYPED = True

    intensity: float = _field(
        help=(
            "The scalar its lightmap's decoded linear texels are multiplied by, "
            "restoring the bake's range."
        ),
        required=True,
    )


@dataclass
class LightmapsBlockHighlight(SchemaSpec):
    """A highlight copy of the embedded probe: a layer with the bake's lights
    drawn in, for the materials it lists."""

    TYPED = True

    bufferView: int = _field(
        help=(
            "The bufferView holding the layer's Radiance (RGBE) bytes: radiance "
            "in the probe's layout and units, added to the probe (box-filtered "
            "down to the layer's size) and prefiltered as the probe is."
        ),
        required=True,
    )
    materials: List[str] = _field(
        help=(
            "The materials, by glTF name, whose surfaces reflect the probe plus "
            "this layer in the probe's place."
        ),
        required=True,
        default_factory=list,
    )


@dataclass
class LightmapsBlockProbe(SchemaSpec):
    """The reflection probe the bake captured, embedded in the file: an
    equirectangular Radiance HDR in one of its bufferViews, placed in metres."""

    TYPED = True

    bufferView: int = _field(
        help="The bufferView holding the probe image's Radiance (RGBE) bytes.",
        required=True,
    )
    position: _Vec3 = _field(
        help="Where it was captured, in metres, in the scene's world space (glTF axes, +Y up).",
        required=True,
    )
    box: Optional[LightmapBox] = _field(
        help=(
            "{min, max}: the box its reflections are projected onto, in metres in "
            "the same space; null for an open scene, whose probe is read as "
            "distant."
        ),
        required=True,
    )
    highlights: List[LightmapsBlockHighlight] = _field(
        help=(
            "Copies of the probe with the bake's lights drawn in, one per group "
            "of receivers; absent when the bake shipped no lights or none was "
            "drawn."
        ),
        nested=[LightmapsBlockHighlight],
    )


@dataclass
class LightmapsBlock(SchemaSpec):
    """The ``extras.lightmaps`` block: the bake's lightmaps bound to the file's
    materials, and the reflection probe embedded in its buffer."""

    TYPED = True
    #: The texture slots a lightmap rides (blendertk ``LightmapWebExport.CARRIERS``).
    CARRIERS = ("occlusion", "emissive")

    version: int = _field(
        help=(
            "The block's schema version (SceneRecords.LIGHTMAPS.glb.version); a "
            "reader reads the version it knows and refuses any other."
        ),
        required=True,
    )
    carrier: str = _field(
        help=(
            "The texture slot each lightmapped material carries its lightmap in, "
            "8-bit sRGB decoded to linear, read through that texture's own "
            "texCoord: 'occlusion' (occlusionTexture) or 'emissive' "
            "(emissiveTexture)."
        ),
        required=True,
        choices=CARRIERS,
    )
    materials: Dict[str, LightmapsBlockMaterial] = _field(
        help=(
            "{glTF material name: {intensity}}: each material carrying a "
            "lightmap -- the surface's whole diffuse light, direct and indirect, "
            "not ambient occlusion -- and the scalar its texels are multiplied by."
        ),
        required=True,
        default_factory=dict,
    )
    probe: LightmapsBlockProbe = _field(
        help=(
            "The reflection probe the bake captured, the scene's environment; "
            "absent when it captured none or its image was not found."
        ),
        nested=LightmapsBlockProbe,
    )
