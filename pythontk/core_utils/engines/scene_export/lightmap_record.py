# !/usr/bin/python
# coding=utf-8
"""The ``lightmaps`` record: what a bake publishes, declared once.

Both hosts' bakers publish the record (mayatk's and blendertk's
``LightmapRecords``) and every reader takes it (the GLB build, unitytk's
importer, the WebXR viewer). :meth:`LightmapRecord.payload` is what a
producer publishes; a reader reads the current schema alone
(``SceneRecords.LIGHTMAPS.is_current``).

:class:`LightmapRecord` is a typed :class:`~pythontk.SchemaSpec` declaration,
so this one definition validates a payload (``LightmapRecord.validate(
payload)``), publishes its JSON Schema, and is what
``m3trik/scripts/sync_scene_records.py`` generates the other languages' types
from: each baked object's lightmap (read through the second UV set, which no
entry names: :attr:`LightmapRecord.UV_SET`), the reflection probe the bake
captured -- the bake's lights drawn into it, and complete copies of it for
groups of receivers -- and the lights it was baked from, every length at one
``unit_scale``. One layout, two containers: an FBX carries it on its
``data_export`` carrier, a GLB as it stands in its root ``extras``
(``RecordSpec.glb_verbatim``), the files it names riding each as media
(``SceneRecords.GLB_MEDIA``).

Change a field together with its producers, and bump the version it is
declared under (``SceneRecords.LIGHTMAPS.version``).
"""

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

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
    """A copy of the probe for one group of receivers (the record's assembly
    says how it is read)."""

    TYPED = True

    map: str = _field(
        help=(
            "The copy's file name, no folder: an equirectangular linear HDR "
            "(top row +Y) in the probe's layout and units, at its own size, a "
            "whole fraction of the probe's, the deliverable carries under that "
            "name."
        ),
        required=True,
    )
    objects: List[LightmapNode] = _field(
        help="The mesh nodes that read this copy.",
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
            "Copies of the probe, one per group of receivers; absent when the "
            "export drew none."
        ),
        nested=[LightmapHighlight],
        # An extension: a reader that picks reflection probes by volume, not
        # per node, reads the probe alone, which holds the highlights.
        readers=("glb",),
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
    """A light the maps were baked from (the record's assembly says how a
    reader lights it)."""

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
            "position and box, each baked light's position, size and radius; "
            "present exactly when the record holds a length."
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
        help="The lights the maps were baked from; absent when none ships.",
        nested=[LightmapBakedLight],
        # An extension: a reader with no light that adds its specular alone
        # reads none of them.
        readers=("glb",),
    )

    #: The UV set every map is read through, 0-based: the second, where an
    #: engine reads a lightmap (glTF's TEXCOORD_1). The record states it
    #: once, in its contract; no entry names it.
    UV_SET = 1

    class UvSetRefused(ValueError):
        """A record that would read some object's map through the wrong UV
        set: its producer publishes none (:meth:`check_uv_sets`)."""

    @classmethod
    def check_uv_sets(cls, placed: Iterable[Sequence[Any]], fix: str) -> None:
        """Refuse a record some of whose maps sit off the second UV set.

        The one refusal both hosts' producers raise: every map is read
        through :attr:`UV_SET`, so an object whose lightmap UVs are another
        set -- or whose committed set is gone, renamed or removed since the
        bake, leaving the second slot to whatever set is there -- would be
        lit through the wrong UVs: the record is not published, and the
        message names each object and the fix.

        Parameters:
            placed: ``(object name, its committed lightmap UV set's name,
                that set's index)`` for each baked object whose marker names
                a set; the index ``None`` where the object no longer has it.
            fix: What the host's user does to put a set in the second slot.

        Raises:
            UvSetRefused: Some set's index is not :attr:`UV_SET`.
        """
        off = [
            (
                f"{name} ({uv_set!r} at index {index})"
                if index is not None
                else f"{name} ({uv_set!r}, which it no longer has)"
            )
            for name, uv_set, index in placed
            if index != cls.UV_SET
        ]
        if off:
            raise cls.UvSetRefused(
                "Lightmaps not published: every map is read through the "
                f"second UV set (index {cls.UV_SET}), where an engine reads a "
                f"lightmap (glTF's TEXCOORD_1), and {len(off)} object(s) "
                f"hold no lightmap set there: {', '.join(off)}. {fix}"
            )

    @classmethod
    def unless_refused(
        cls,
        build: Callable[[], Any],
        logger: Any,
        note: Optional[Callable[[str], Any]] = None,
    ) -> Any:
        """*build()*'s record, or ``None`` when it is refused
        (:class:`UvSetRefused`): no write carries a record that reads a map
        through the wrong UVs, and the work that published goes on.

        Parameters:
            build: Builds the record; anything else it raises propagates.
            logger: Where an authoring-time publish (a bake, a revert) says
                the refusal, at warning.
            note: Takes the refusal's sentence instead -- an export's
                producer passes ``ExportContext.note``: the export's report
                says it, and the snapshot clears the record (a raise was a
                failed producer, whose stale stored record shipped).
        """
        try:
            return build()
        except cls.UvSetRefused as refusal:
            if note is not None:
                note(str(refusal))
            else:
                logger.warning("%s", refusal)
            return None

    @classmethod
    def withheld(
        cls,
        ctx: Any,
        store: Any,
        placed: Callable[[], Iterable[Sequence[Any]]],
        fix: str,
    ) -> Dict[str, str]:
        """``{"lightmaps": why}`` when the write of *ctx* ships the record
        *store* holds as it stands and that copy is refused now, else ``{}``
        -- what both hosts' ``LightmapRecords.withheld`` answer for
        ``ExportContext.withheld``.

        A hand-off ships the authored record as stored
        (``ExportContext.refreshes``), and a lightmap set moved, renamed or
        removed since the bake published leaves that copy reading a map
        through the wrong UVs: refused as its producer refuses one
        (:meth:`check_uv_sets`), the write ships no lightmaps and the scene
        keeps its record. A write whose producer runs refuses for itself
        (:meth:`unless_refused`).

        Parameters:
            ctx: The write's ``ExportContext``.
            store: The scene's store.
            placed: The host's :meth:`check_uv_sets` placements for what the
                write ships, asked only when a current record is stored.
            fix: The host's fix, as its producer states it.
        """
        spec = SceneRecords.LIGHTMAPS
        if ctx.refreshes(spec) or not spec.is_current(spec.load(store)):
            return {}
        try:
            cls.check_uv_sets(placed(), fix)
        except cls.UvSetRefused as refusal:
            return {spec.key: str(refusal)}
        return {}

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
                ``map``, ``scaleOffset``), its map read through the second UV
                set (:meth:`check_uv_sets` refuses one that is not).
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
