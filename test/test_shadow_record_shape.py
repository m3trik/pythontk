# !/usr/bin/python
# coding=utf-8
"""``ShadowRecord`` (``geo_utils/shadow_record.py``): the ``shadows`` record's
payload, v3, held to what every producer writes rather than to itself --
both DCC rigs' ``plane_record`` (their dicts copied below as the source spells
them, each scene read replaced by the value it returns) and
``ShadowHorizon.record`` (called for real).

Pure Python, no DCC. A GLB carries the record as it stands (one layout, two
containers), so no second shape exists to hold it to.
"""

import json
import os
import sys
import unittest
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pythontk as ptk  # noqa: E402

from pythontk.geo_utils import shadow_record  # noqa: E402

ShadowRecord = shadow_record.ShadowRecord

# ============================================================================
# The DCC producers, copied
# ============================================================================

# The stamp names both rigs read (each twin's ShadowRig class attributes).
_CANVAS = ("canvasU0", "canvasU1", "canvasW0", "canvasW1")
_ATLAS_RECT = ("atlasScaleX", "atlasScaleY", "atlasOffsetX", "atlasOffsetY")
_HORIZON_INTS = ("horizonSize", "horizonSpans", "horizonLevels")
_HORIZON_FLOATS = (
    "horizonBoundsA0",
    "horizonBoundsA1",
    "horizonBoundsB0",
    "horizonBoundsB1",
    "horizonHeightScale",
)
_HORIZON_RECT = ("horizonScaleX", "horizonScaleY", "horizonOffsetX", "horizonOffsetY")
# Each twin's HORIZON_FRAME: Blender's exporter maps local +Y to FBX -Z.
_MAYA_FRAME = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
_BLENDER_FRAME = ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))


def _stamps(kind="projected", *, atlas=False, horizon=True, **overrides):
    """A plane's stamps as both rigs name them: a *kind* rig in centimetres,
    *atlas* packed into the silhouette atlas, *horizon* baked (every rig's
    map at export, v3) and packed into the horizon atlas."""
    stamps = {
        "shadowRigType": kind,
        "sourceSize": 3.0,
        "shadowIntensity": 0.54,
        "falloffPower": 1.4,
        "fadeHeight": 240.0,
        "followSource": True,
        "groundHeight": 0.0,
        "footprintRadius": 42.4264,
        "objectHeight": 120.0,
        "maxStretch": 6.0,
        "canvasU0": -0.8,
        "canvasU1": 1.2,
        "canvasW0": -0.5,
        "canvasW1": 0.5,
    }
    if atlas:
        stamps.update(
            {
                "atlasTexture": "shadow_atlas_projected.png",
                "atlasScaleX": 0.5,
                "atlasScaleY": 0.5,
                "atlasOffsetX": 0.0,
                "atlasOffsetY": 0.5,
            }
        )
    if horizon:
        stamps.update(
            {
                "horizonTexture": "Lamp_horizon.png",
                "horizonSize": 128,
                "horizonSpans": 2,
                "horizonLevels": 7,
                "horizonBoundsA0": -60.0,
                "horizonBoundsA1": 60.0,
                "horizonBoundsB0": -55.0,
                "horizonBoundsB1": 65.0,
                "horizonHeightScale": 150.0,
                "horizonAtlas": "shadow_atlas_horizon.png",
                "horizonScaleX": 0.5,
                "horizonScaleY": 1.0,
                "horizonOffsetX": 0.5,
                "horizonOffsetY": 0.0,
            }
        )
    stamps.update(overrides)
    return stamps


def _plane_type(stamps):
    """``ShadowRig.plane_type`` (both twins)."""
    value = stamps.get("shadowRigType", "") or "projected"
    return value if value in ("projected", "horizon") else "projected"


def _read_rect(stamps, names):
    """``ShadowRig._read_rect`` (both twins)."""
    return [
        round(float(stamps.get(a, d)), 6) for a, d in zip(names, (1.0, 1.0, 0.0, 0.0))
    ]


def _horizon_params(stamps, frame, unit=1.0):
    """``ShadowRig._horizon_params`` (both twins): *frame* the twin's
    ``HORIZON_FRAME``, *unit* mayatk's internal-to-working factor (its
    ``Units.convert``; blendertk converts nothing). Every rig type's."""
    if not stamps.get("horizonTexture", ""):
        return {}
    ints = [int(stamps.get(a, 0) or 0) for a in _HORIZON_INTS]
    floats = [float(stamps.get(a, 0.0) or 0.0) for a in _HORIZON_FLOATS]
    atlas = stamps.get("horizonAtlas", "")
    return ptk.ShadowHorizon.record(
        texture=atlas or stamps.get("horizonTexture", ""),
        size=ints[0],
        spans=ints[1],
        bounds=[v * unit for v in floats[:4]],
        height_scale=floats[4] * unit,
        frame_a=frame[0],
        frame_b=frame[1],
        rect=_read_rect(stamps, _HORIZON_RECT),
    )


def _silhouette(stamps, tex, get):
    """``ShadowRig._silhouette`` (both twins): the plane's projected
    silhouette through ``ShadowRecord.silhouette``, the one builder both rigs
    call; ``{}`` when it has no image; *get* the twin's stamp read."""
    return ptk.ShadowRecord.silhouette(
        tex,
        _plane_type(stamps) == "projected",
        atlas=get("atlasTexture", ""),
        atlas_rect=_read_rect(stamps, _ATLAS_RECT),
    )


def _maya_plane_record(
    stamps,
    plane="|Lamp_grp|Lamp_shadow",
    *,
    tex="C:/project/sourceimages/Lamp_shadow.png",
    source="|lights|shadow_source",
    contact="|Lamp_grp|Lamp|Lamp_contact_loc",
    directional=False,
    factor=1.0,
    unit=1.0,
):
    """``mayatk ShadowRig.plane_record`` (``rig_utils/shadow_rig/_shadow_rig.py``)
    as the source spells it: *plane*, *source* and *contact* DAG paths (``""``
    for none), *factor* the stamps' unit to the working one, *unit* Maya's
    internal unit to it."""

    def attr(_plane, name, default=None):  # ShadowRig._plane_attr
        return stamps.get(name, default)

    def leaf_name(node):  # CoreUtils.leaf_name
        return str(node).split("|")[-1]

    size = float(attr(plane, "sourceSize", 0.0) or 0.0)
    height = float(attr(plane, "objectHeight", 0.0))
    record = {
        "name": leaf_name(plane),
        "intensity": round(float(attr(plane, "shadowIntensity", 1.0)), 4),
        "source": leaf_name(source) if source else "",
        "source_type": "directional" if directional else "point",
        "source_size": 0.0 if directional else round(size * factor, 6),
        "source_angle": round(size, 6) if directional else 0.0,
        "follow_source": bool(attr(plane, "followSource", True)),
        "contact": leaf_name(contact) if contact else "",
        "ground": round(float(attr(plane, "groundHeight", 0.0)) * factor, 6),
        "radius": round(float(attr(plane, "footprintRadius", 0.0)) * factor, 6),
        "height": round(height * factor, 6),
        "max_stretch": round(
            float(attr(plane, "maxStretch", ptk.ShadowProjection.DEFAULT_MAX_STRETCH)),
            6,
        ),
        "falloff_power": round(
            float(
                attr(plane, "falloffPower", ptk.ShadowProjection.DEFAULT_FALLOFF_POWER)
            ),
            6,
        ),
        "fade_height": round(
            float(attr(plane, "fadeHeight", 2.0 * height)) * factor, 6
        ),
        # The preview ramps in over one unit of the rig.
        "source_ramp": round(factor, 6),
        "canvas": [
            round(float(attr(plane, a, d)), 6)
            for a, d in zip(_CANVAS, ptk.ShadowConformance.CANVAS)
        ],
    }
    horizon = _horizon_params(stamps, _MAYA_FRAME, unit)
    if horizon:
        record["horizon"] = horizon
    silhouette = _silhouette(stamps, tex, lambda n, d=None: attr(plane, n, d))
    if silhouette:
        record["silhouette"] = silhouette
    return record


def _blender_plane_record(
    stamps,
    plane="Lamp_shadow",
    *,
    tex="//textures/Lamp_shadow.png",
    source="shadow_source",
    contact="Lamp_contact_loc",
    directional=False,
):
    """``blendertk ShadowRig.plane_record`` (``rig_utils/shadow_rig/_shadow_rig.py``)
    as the source spells it: *plane*, *source* and *contact* object names
    (``None`` for no object)."""

    def prop(_plane, name, default=None):  # ShadowRig._plane_prop
        value = stamps.get(name)
        return default if value is None else value

    size = float(prop(plane, "sourceSize", 0.0) or 0.0)
    height = float(prop(plane, "objectHeight", 0.0))
    record = {
        "name": plane,
        "intensity": round(float(prop(plane, "shadowIntensity", 1.0)), 4),
        "source": source if source is not None else "",
        "source_type": "directional" if directional else "point",
        "source_size": 0.0 if directional else round(size, 6),
        "source_angle": round(size, 6) if directional else 0.0,
        "follow_source": bool(prop(plane, "followSource", True)),
        "contact": contact if contact is not None else "",
        "ground": round(float(prop(plane, "groundHeight", 0.0)), 6),
        "radius": round(float(prop(plane, "footprintRadius", 0.0)), 6),
        "height": round(height, 6),
        "max_stretch": round(
            float(prop(plane, "maxStretch", ptk.ShadowProjection.DEFAULT_MAX_STRETCH)),
            6,
        ),
        "falloff_power": round(
            float(
                prop(plane, "falloffPower", ptk.ShadowProjection.DEFAULT_FALLOFF_POWER)
            ),
            6,
        ),
        "fade_height": round(float(prop(plane, "fadeHeight", 2.0 * height)), 6),
        # The preview ramps in over one Blender unit, the record's.
        "source_ramp": 1.0,
        "canvas": [
            round(float(prop(plane, p, d)), 6)
            for p, d in zip(_CANVAS, ptk.ShadowConformance.CANVAS)
        ],
    }
    horizon = _horizon_params(stamps, _BLENDER_FRAME)
    if horizon:
        record["horizon"] = horizon
    silhouette = _silhouette(stamps, tex, lambda n, d=None: prop(plane, n, d))
    if silhouette:
        record["silhouette"] = silhouette
    return record


def _payload(planes, unit_scale=0.01):
    """The record ``ShadowRig.export_record`` makes of *planes* (the
    declaration stamps the version)."""
    return ptk.SceneRecords.SHADOWS.make(
        {"unit_scale": unit_scale, "planes": list(planes)}
    ).payload


#: Every plane a rig publishes, by what it carries: ``{label: (stamps,
#: producer overrides)}`` -- the overrides spelled for each twin's arguments.
_VARIANTS = {
    "projected": (_stamps(), {}),
    "projected, packed": (_stamps(atlas=True), {}),
    "horizon, both maps packed": (_stamps("horizon", atlas=True), {}),
    "horizon, its own map": (_stamps("horizon", horizonAtlas=""), {}),
    "no source, no contact, no silhouette": (
        _stamps(),
        {"source": None, "contact": None, "tex": ""},
    ),
    "directional source": (
        _stamps("horizon", sourceSize=0.0093),
        {"source": "sun", "directional": True},
    ),
}


def _maya(stamps, overrides):
    kwargs = dict(overrides)
    for key in ("source", "contact"):
        if key in kwargs:  # a DAG path, "" when the rig has none
            kwargs[key] = "" if kwargs[key] is None else f"|{kwargs[key]}"
    return _maya_plane_record(stamps, **kwargs)


def _blender(stamps, overrides):
    return _blender_plane_record(stamps, **overrides)


# ============================================================================
# Tests
# ============================================================================


class TestRecordShape(unittest.TestCase):
    """What both DCC rigs publish is a ``ShadowRecord``."""

    def assertDeclared(self, payload):
        res = ShadowRecord.validate(payload)
        self.assertEqual((res.errors, res.warnings), ([], []))

    def test_every_maya_plane_record_is_the_declared_shape(self):
        for label, (stamps, overrides) in _VARIANTS.items():
            with self.subTest(variant=label):
                self.assertDeclared(_payload([_maya(stamps, overrides)]))

    def test_every_blender_plane_record_is_the_declared_shape(self):
        for label, (stamps, overrides) in _VARIANTS.items():
            with self.subTest(variant=label):
                self.assertDeclared(_payload([_blender(stamps, overrides)], 1.0))

    def test_a_centimetre_rig_written_in_metres_is_the_declared_shape(self):
        """mayatk converts each length stamp from its rig's unit to the
        working one -- the fade height among them."""
        plane = _maya_plane_record(_stamps("horizon", atlas=True), factor=0.01, unit=0.01)
        self.assertAlmostEqual(plane["radius"], 0.424264)
        self.assertAlmostEqual(plane["fade_height"], 2.4)
        self.assertEqual(plane["falloff_power"], 1.4, "a ratio: never converted")
        self.assertEqual(plane["source_ramp"], 0.01, "one centimetre: the rig's ramp")
        self.assertDeclared(_payload([plane], unit_scale=1.0))

    def test_every_rig_type_carries_its_horizon_and_states_its_draw(self):
        """v3 (2026-11-07): a horizon map on every plane, whatever the rig's
        type; the type is the silhouette's draw choice, never a field."""
        for label, (stamps, overrides) in _VARIANTS.items():
            with self.subTest(variant=label):
                plane = _maya(stamps, overrides)
                self.assertIn("horizon", plane)
                self.assertNotIn("type", plane)
                if "silhouette" in plane:
                    self.assertEqual(
                        plane["silhouette"]["draw"], _plane_type(stamps) == "projected"
                    )
        planes = [_maya(s, o) for s, o in _VARIANTS.values()]
        self.assertTrue(any("silhouette" not in p for p in planes))
        self.assertTrue(any("atlas" in p.get("silhouette", {}) for p in planes))

    def test_the_horizon_builder_writes_the_declared_block(self):
        """``ShadowHorizon.record`` -- the one builder both rigs call."""
        block = _horizon_params(_stamps("horizon"), _MAYA_FRAME)
        res = shadow_record.ShadowPlaneHorizon.validate(block)
        self.assertEqual((res.errors, res.warnings), ([], []))

    def test_the_published_fade_terms_are_the_models(self):
        """What a plane states is what ``ShadowProjection.fade`` reads: its
        lengths, its ``source_ramp`` among them, times ``unit_scale`` are
        metres (``ShadowRecord.fade``); a plane whose contact or source does
        not resolve draws at its intensity. Changed: 2026-11-07 (the ramp
        was the record's unit, which a cm rig exported in metres is not)."""
        unit = 0.01
        C, L = (0.0, 0.0, 0.0), (120.0, 0.6, 40.0)  # the source 0.6 units up
        for plane in (_maya_plane_record(_stamps()), _blender_plane_record(_stamps())):
            with self.subTest(ramp=plane["source_ramp"]):
                fade = ptk.ShadowProjection.fade(
                    [v * unit for v in C],
                    [v * unit for v in L],
                    plane["ground"] * unit,
                    plane["radius"] * unit,
                    plane["height"] * unit,
                    max_stretch=plane["max_stretch"],
                    intensity=plane["intensity"],
                    falloff_power=plane["falloff_power"],
                    fade_height=plane["fade_height"] * unit,
                    source_ramp=plane["source_ramp"] * unit,
                )
                self.assertTrue(0.0 < fade <= plane["intensity"])
                self.assertEqual(ptk.ShadowRecord.fade(plane, unit, C, L), fade)
                self.assertEqual(ptk.ShadowRecord.fade(plane, unit), plane["intensity"])
        # The ramp scales the source's term alone: twice the ramp, half the fade.
        one, two = _blender_plane_record(_stamps()), _blender_plane_record(_stamps())
        two["source_ramp"] = 2.0
        self.assertAlmostEqual(
            ptk.ShadowRecord.fade(two, unit, C, L) / ptk.ShadowRecord.fade(one, unit, C, L),
            0.5,
            places=9,
        )

    def test_the_canvas_is_the_planes_for_every_reader(self):
        """The canvas decides where a following plane is placed, so every
        reader holds it -- never the GLB-only silhouette, whose absence left
        a reader placing by the default and clipping the penumbra. Added:
        2026-11-07."""
        plane = _maya(*_VARIANTS["projected"])
        self.assertEqual(plane["canvas"], [-0.8, 1.2, -0.5, 0.5])
        self.assertNotIn("canvas", plane["silhouette"])
        other = ShadowRecord.json_schema(reader="another")["$defs"]["ShadowPlane"]
        self.assertIn("canvas", other["required"])
        self.assertIn("source_ramp", other["required"])


class TestRejections(unittest.TestCase):
    """A payload that breaks the shape is named where it breaks."""

    def _plane(self):
        return _maya_plane_record(_stamps("horizon", atlas=True))

    def test_a_missing_required_key_is_an_error(self):
        plane = self._plane()
        del plane["contact"]
        del plane["horizon"]["frame_b"]
        del plane["silhouette"]["draw"]
        res = ShadowRecord.validate(_payload([plane]))
        self.assertEqual(
            res.errors,
            [
                "planes[0].missing required key 'contact'",
                "planes[0].horizon.missing required key 'frame_b'",
                "planes[0].silhouette.missing required key 'draw'",
            ],
        )

    def test_a_plane_with_no_horizon_map_is_refused(self):
        """Every plane's map is required: a producer bakes it at export, and
        leaves out a plane it cannot bake rather than publish it without."""
        plane = _maya_plane_record(_stamps(horizon=False))
        self.assertNotIn("horizon", plane)
        res = ShadowRecord.validate(_payload([plane]))
        self.assertEqual(res.errors, ["planes[0].missing required key 'horizon'"])

    def test_a_value_of_the_wrong_type_is_an_error(self):
        """No producer writes a null here: ``source`` is ``""`` when the rig
        has none, and a block the plane does not carry is absent."""
        plane = self._plane()
        plane.update(
            source=None, follow_source=1, radius="60", fade_height=None, canvas=[-1.0, 1.0]
        )
        plane["silhouette"].update(atlas=None, draw="yes")
        plane["horizon"]["size"] = 128.0
        res = ShadowRecord.validate(_payload([plane]))
        self.assertCountEqual(
            res.errors,
            [
                "planes[0].follow_source: expected boolean, got integer",
                "planes[0].radius: expected number, got string",
                "planes[0].source: expected string, got null",
                "planes[0].fade_height: expected number, got null",
                "planes[0].horizon.size: expected integer, got number",
                "planes[0].canvas: expected 4 items, got 2",
                "planes[0].silhouette.draw: expected boolean, got string",
                "planes[0].silhouette.atlas: expected a mapping for nested schema",
            ],
        )

    def test_a_ramp_a_reader_would_divide_by_zero_is_an_error(self):
        """``source_ramp`` is the fade's divisor: none, or under none, is
        refused rather than left to each reader's arithmetic. Added:
        2026-11-07."""
        for ramp in (0.0, -0.01):
            with self.subTest(ramp=ramp):
                plane = self._plane()
                plane["source_ramp"] = ramp
                res = ShadowRecord.validate(_payload([plane]))
                self.assertEqual(len(res.errors), 1, res.errors)
                self.assertIn("planes[0].source_ramp", res.errors[0])

    def test_a_value_outside_its_vocabulary_is_an_error(self):
        plane = self._plane()
        plane.update(source_type="spot")
        res = ShadowRecord.validate(_payload([plane]))
        self.assertEqual(
            res.errors,
            ["planes[0].source_type: 'spot' is not one of ['point', 'directional']"],
        )

    def test_a_v2_plane_leaves_its_retired_fields_behind(self):
        """A v2 plane's ``type`` and its silhouette's image on the plane
        itself (``texture``, ``atlas``) are no v3 field: ``make`` drops each,
        and the plane still lacks its fade terms -- and the record's version
        refuses it whole (``SceneRecords.problems``)."""
        v2 = {
            k: v
            for k, v in _maya_plane_record(_stamps()).items()
            if k not in ("silhouette", "falloff_power", "fade_height", "source_ramp")
        }
        v2.update(type="projected", texture="Lamp_shadow.png", atlas="atlas.png")
        made = _payload([v2])["planes"][0]
        for retired in ("type", "texture", "atlas"):
            self.assertNotIn(retired, made)
        res = ShadowRecord.validate(_payload([made]))
        self.assertEqual(
            res.errors,
            [
                "planes[0].missing required key 'falloff_power'",
                "planes[0].missing required key 'fade_height'",
                "planes[0].missing required key 'source_ramp'",
            ],
        )


class TestPublishedForms(unittest.TestCase):
    """The JSON Schema other languages generate their types from, and the
    defaults a reader fills an unsaid field with."""

    def test_the_shape_publishes_a_json_schema(self):
        schema = ShadowRecord.json_schema()
        json.dumps(schema)
        self.assertEqual(schema["title"], "ShadowRecord")
        self.assertEqual(
            set(schema["$defs"]),
            {"ShadowPlane", "ShadowPlaneHorizon", "ShadowPlaneSilhouette", "ShadowPlaneAtlas"},
        )
        self.assertEqual(schema["required"], ["version", "unit_scale", "planes"])
        plane = schema["$defs"]["ShadowPlane"]
        self.assertIn("horizon", plane["required"])
        self.assertNotIn("silhouette", plane["required"])
        self.assertNotIn("type", plane["properties"])

    def test_the_silhouette_is_the_glbs_alone(self):
        """An extension (``readers=("glb",)``): every other reader's view, the
        one its generated types come from, holds no silhouette, and a reader
        without it draws the horizon."""
        other = ShadowRecord.json_schema(reader="another")
        self.assertEqual(set(other["$defs"]), {"ShadowPlane", "ShadowPlaneHorizon"})
        self.assertNotIn("silhouette", other["$defs"]["ShadowPlane"]["properties"])
        glb = ShadowRecord.json_schema(reader="glb")
        self.assertIn("silhouette", glb["$defs"]["ShadowPlane"]["properties"])

    def test_every_field_says_what_it_is(self):
        schema = ShadowRecord.json_schema()
        objects = {"ShadowRecord": schema, **schema["$defs"]}
        for name, node in objects.items():
            for key, prop in node["properties"].items():
                with self.subTest(field=f"{name}.{key}"):
                    self.assertTrue(prop.get("description"))

    def test_the_vocabularies_are_pythontks_own(self):
        plane = ShadowRecord.json_schema()["$defs"]["ShadowPlane"]["properties"]
        self.assertEqual(plane["source_type"]["enum"], list(shadow_record.SOURCE_TYPES))

    def test_the_rig_types_are_reachable_from_the_root(self):
        """Both DCC rigs read the vocabulary off ``ptk.ShadowRecord`` -- never
        the submodule -- and it stays a class constant, not a record field."""
        self.assertEqual(ptk.ShadowRecord.RIG_TYPES, shadow_record.RIG_TYPES)
        self.assertNotIn("RIG_TYPES", {f.name for f in fields(ShadowRecord)})
        self.assertNotIn("RIG_TYPES", ShadowRecord.json_schema()["properties"])

    def test_a_plane_starts_as_the_rigs_write_it(self):
        """A plane's defaults are a rig's starting values; "" is no node, the
        spelling both rigs write and every reader takes as none; the fade's
        terms are the model's defaults."""
        plane = shadow_record.ShadowPlane()
        self.assertEqual((plane.source, plane.contact), ("", ""))
        self.assertEqual(plane.falloff_power, ptk.ShadowProjection.DEFAULT_FALLOFF_POWER)
        self.assertEqual(plane.fade_height, 2.0 * plane.height)
        self.assertEqual(plane.source_ramp, 1.0)
        # A mutable default is the plane's own, never shared.
        self.assertEqual(plane.canvas, list(ptk.ShadowConformance.CANVAS))
        self.assertIsNot(plane.canvas, shadow_record.ShadowPlane().canvas)

    def test_the_retired_block_names_say_where_they_went(self):
        """The v2 GLB block (``ShadowBlock`` and its parts) retired with v3:
        the moved-names table names the record that replaced it."""
        for name in ("ShadowBlock", "ShadowBlockPlane", "ShadowBlockHorizon"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(shadow_record, name))
                self.assertEqual(ptk.MOVED_NAMES[name][0], "pythontk.ShadowRecord")


if __name__ == "__main__":
    unittest.main()
