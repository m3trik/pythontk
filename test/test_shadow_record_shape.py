# !/usr/bin/python
# coding=utf-8
"""``ShadowRecord`` and ``ShadowBlock`` (``geo_utils/shadow_record.py``): the
``shadows`` record's payload and its GLB block, held to what every producer
writes rather than to themselves -- both DCC rigs' ``plane_record`` (their
dicts copied below as the source spells them, each scene read replaced by the
value it returns), ``ShadowHorizon.record`` (called for real) and a real
``MeshConvert.apply_glb_shadows`` run.

Pure Python, no DCC. The GLB fixtures are ``test_shadow_web``'s.
"""

import json
import os
import sys
import unittest
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pythontk as ptk  # noqa: E402
from pythontk import MeshConvert  # noqa: E402

from pythontk.geo_utils import shadow_record  # noqa: E402
from test_shadow_web import (  # noqa: E402
    MESH_CONVERT_LOGGER,
    _bare_material,
    _data_export_node,
    _data_uri,
    _plane_material,
    _silhouette_png,
    _write_glb,
)

ptk.TestSandbox.activate()

ShadowRecord = shadow_record.ShadowRecord
ShadowBlock = shadow_record.ShadowBlock

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


def _stamps(kind="projected", *, atlas=False, horizon=False, **overrides):
    """A plane's stamps as both rigs name them: a *kind* rig in centimetres,
    *atlas* packed into the silhouette atlas, *horizon* baked and packed into
    the horizon atlas."""
    stamps = {
        "shadowRigType": kind,
        "sourceSize": 3.0,
        "shadowIntensity": 0.54,
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
    ``Units.convert``; blendertk converts nothing)."""
    if _plane_type(stamps) != "horizon":
        return {}
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
    record = {
        "name": leaf_name(plane),
        "type": _plane_type(stamps),
        "texture": os.path.basename(tex) if tex else "",
        "intensity": round(float(attr(plane, "shadowIntensity", 1.0)), 4),
        "source": leaf_name(source) if source else "",
        "source_type": "directional" if directional else "point",
        "source_size": 0.0 if directional else round(size * factor, 6),
        "source_angle": round(size, 6) if directional else 0.0,
        "follow_source": bool(attr(plane, "followSource", True)),
        "contact": leaf_name(contact) if contact else "",
        "ground": round(float(attr(plane, "groundHeight", 0.0)) * factor, 6),
        "radius": round(float(attr(plane, "footprintRadius", 0.0)) * factor, 6),
        "height": round(float(attr(plane, "objectHeight", 0.0)) * factor, 6),
        "max_stretch": round(
            float(attr(plane, "maxStretch", ptk.ShadowProjection.DEFAULT_MAX_STRETCH)),
            6,
        ),
        "canvas": [
            round(float(attr(plane, a, d)), 6)
            for a, d in zip(_CANVAS, (-1.0, 1.0, -0.5, 0.5))
        ],
    }
    atlas = attr(plane, "atlasTexture", "")
    if atlas:
        record["atlas"] = {"texture": atlas, "rect": _read_rect(stamps, _ATLAS_RECT)}
    horizon = _horizon_params(stamps, _MAYA_FRAME, unit)
    if horizon:
        record["horizon"] = horizon
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
    record = {
        "name": plane,
        "type": _plane_type(stamps),
        "texture": os.path.basename(tex) if tex else "",
        "intensity": round(float(prop(plane, "shadowIntensity", 1.0)), 4),
        "source": source if source is not None else "",
        "source_type": "directional" if directional else "point",
        "source_size": 0.0 if directional else round(size, 6),
        "source_angle": round(size, 6) if directional else 0.0,
        "follow_source": bool(prop(plane, "followSource", True)),
        "contact": contact if contact is not None else "",
        "ground": round(float(prop(plane, "groundHeight", 0.0)), 6),
        "radius": round(float(prop(plane, "footprintRadius", 0.0)), 6),
        "height": round(float(prop(plane, "objectHeight", 0.0)), 6),
        "max_stretch": round(
            float(prop(plane, "maxStretch", ptk.ShadowProjection.DEFAULT_MAX_STRETCH)),
            6,
        ),
        "canvas": [
            round(float(prop(plane, p, d)), 6)
            for p, d in zip(_CANVAS, (-1.0, 1.0, -0.5, 0.5))
        ],
    }
    atlas = prop(plane, "atlasTexture", "")
    if atlas:
        record["atlas"] = {"texture": atlas, "rect": _read_rect(stamps, _ATLAS_RECT)}
    horizon = _horizon_params(stamps, _BLENDER_FRAME)
    if horizon:
        record["horizon"] = horizon
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
    "horizon, both maps packed": (_stamps("horizon", atlas=True, horizon=True), {}),
    "horizon, its own map": (_stamps("horizon", horizon=True, horizonAtlas=""), {}),
    "horizon, not baked yet": (_stamps("horizon"), {}),
    "no source, no contact, no silhouette": (
        _stamps(),
        {"source": None, "contact": None, "tex": ""},
    ),
    "directional source": (
        _stamps("horizon", horizon=True, sourceSize=0.0093),
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
        """mayatk converts each stamp from its rig's unit to the working one."""
        stamps = _stamps("horizon", atlas=True, horizon=True)
        plane = _maya_plane_record(stamps, factor=0.01, unit=0.01)
        self.assertAlmostEqual(plane["radius"], 0.424264)
        self.assertDeclared(_payload([plane], unit_scale=1.0))

    def test_the_variants_cover_both_optional_blocks_and_their_absence(self):
        planes = [_maya(s, o) for s, o in _VARIANTS.values()]
        for block in ("atlas", "horizon"):
            with self.subTest(block=block):
                self.assertTrue(any(block in p for p in planes))
                self.assertTrue(any(block not in p for p in planes))

    def test_the_horizon_builder_writes_the_declared_block(self):
        """``ShadowHorizon.record`` -- the one builder both rigs call."""
        block = _horizon_params(_stamps("horizon", horizon=True), _MAYA_FRAME)
        res = shadow_record.ShadowPlaneHorizon.validate(block)
        self.assertEqual((res.errors, res.warnings), ([], []))


class TestBlockShape(unittest.TestCase):
    """What ``MeshConvert.apply_glb_shadows`` writes is a ``ShadowBlock``."""

    @classmethod
    def setUpClass(cls):
        cls.temp = ptk.TempArtifacts("shadow_record_shape", policy="scoped")
        cls.maps = cls.temp.dir_path()
        # Opaque bytes: the pass embeds a horizon map as found, never decoding it.
        with open(os.path.join(cls.maps, "shadow_atlas_horizon.png"), "wb") as fh:
            fh.write(_silhouette_png())
        # A silhouette loose beside it, as a real export leaves one (a plane
        # material's file texture does not survive the FBX hop).
        with open(os.path.join(cls.maps, "Loose_shadow.png"), "wb") as fh:
            fh.write(_silhouette_png())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _apply(self, planes, nodes, materials, images, textures, prepare=None):
        """Write a GLB carrying *planes*' record, let *prepare* edit its glTF,
        run the pass, and return ``(the record, the block, the glTF)`` as the
        file holds them."""
        record = _payload(planes)
        path = self.temp.path(extension=".glb")
        _write_glb(
            path,
            nodes=nodes + [_data_export_node(record)],
            materials=materials,
            images=images,
            textures=textures,
        )
        if prepare is not None:
            with MeshConvert.open_glb(path) as edit:
                prepare(edit.gltf)
                edit.dirty = True
        self.assertIsNotNone(
            MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
        )
        with MeshConvert.open_glb(path) as edit:
            gltf = json.loads(json.dumps(edit.gltf))
        return record, gltf["extras"][ptk.SceneRecords.SHADOWS.key], gltf

    def assertDeclared(self, shape, payload):
        res = shape.validate(payload)
        self.assertEqual((res.errors, res.warnings), ([], []))

    def test_a_real_run_writes_the_declared_block(self):
        """A packed projected plane, a packed horizon plane lit by a sun, and a
        plane with no source or contact (null nodes) -- the record each was
        made from is a ``ShadowRecord``, the block a ``ShadowBlock``."""
        planes = [
            _maya_plane_record(
                _stamps(atlas=True),
                "|Box_shadow",
                tex="Box_shadow.png",
                source="|shadow_source",
                contact="|Box|Box_contact_loc",
            ),
            _maya_plane_record(
                _stamps("horizon", atlas=True, horizon=True, sourceSize=0.0093),
                source="|sun",
                directional=True,
            ),
            _maya_plane_record(
                _stamps(),
                "|Crate_shadow",
                tex="Crate_shadow.png",
                source="",
                contact="",
            ),
        ]
        record, block, gltf = self._apply(
            planes,
            nodes=[
                {"name": "Box_shadow", "mesh": 0},
                {"name": "shadow_source", "translation": [1.5, 2.0, -0.5]},
                {"name": "Box_contact_loc", "translation": [0.2, 0.0, 0.3]},
                {"name": "Lamp_shadow", "mesh": 1},
                {"name": "sun", "rotation": [-0.3826834, 0.0, 0.0, 0.9238795]},
                {"name": "Lamp_contact_loc", "translation": [2.0, 0.0, 0.0]},
                {"name": "Crate_shadow", "mesh": 2},
            ],
            materials=[
                _plane_material("Box_shadow_MAT", texture=0),
                _bare_material("Lamp_shadow_MAT"),
                _plane_material("Crate_shadow_MAT", texture=1),
            ],
            images=[
                {
                    "name": "shadow_atlas_projected.png",
                    "uri": _data_uri(_silhouette_png()),
                },
                {"name": "Crate_shadow.png", "uri": _data_uri(_silhouette_png(4))},
            ],
            textures=[{"source": 0, "sampler": 0}, {"source": 1, "sampler": 0}],
        )
        self.assertDeclared(ShadowRecord, record)
        self.assertDeclared(ShadowBlock, block)
        # Bound by index in place of every name: a plane is its node.
        by_name = {gltf["nodes"][p["node"]]["name"]: p for p in block["planes"]}
        self.assertEqual(set(by_name), {"Box_shadow", "Lamp_shadow", "Crate_shadow"})
        # The bindings each type admits were really exercised.
        self.assertEqual(by_name["Box_shadow"]["source_node"], 1)
        self.assertEqual(by_name["Box_shadow"]["texture_index"], 0, "its atlas")
        self.assertEqual(by_name["Lamp_shadow"]["source_type"], "directional")
        self.assertIsNone(by_name["Lamp_shadow"]["texture_index"], "the march's")
        self.assertIsInstance(by_name["Lamp_shadow"]["horizon"]["texture_index"], int)
        self.assertIsNone(by_name["Crate_shadow"]["source_node"])
        self.assertIsNone(by_name["Crate_shadow"]["contact_node"])
        self.assertEqual(by_name["Crate_shadow"]["texture_index"], 1, "its own map")

    def test_a_horizon_plane_looks_up_no_colour_map(self):
        """Bug: the pass resolved a horizon plane's atlas and silhouette as it
        does a projected plane's -- warning of each one it could not find
        ("keep their DCC material", of a plane it bound) and embedding each
        loose one it could, a texture nothing samples -- though the block
        binds a horizon plane to its height-field map alone: the march shades
        it. Fixed: 2026-10-20 -- a horizon plane binds its map and nothing
        else. Its atlas nowhere, its silhouette loose beside the maps and its
        mesh with no material, it binds without a warning, its colour map
        null, and the file gains its map's texture and no other."""
        stamps = _stamps(
            "horizon", atlas=True, horizon=True, atlasTexture="missing_atlas.png"
        )
        plane = _maya_plane_record(stamps, tex="Loose_shadow.png")

        def no_material(gltf):
            del gltf["meshes"][0]["primitives"][0]["material"]

        with self.assertNoLogs(MESH_CONVERT_LOGGER, level="WARNING"):
            _record, block, gltf = self._apply(
                [plane],
                nodes=[
                    {"name": "Lamp_shadow", "mesh": 0},
                    {"name": "shadow_source", "translation": [1.0, 2.0, 0.0]},
                    {"name": "Lamp_contact_loc"},
                ],
                materials=[_bare_material("Lamp_shadow_MAT")],
                images=[],
                textures=[],
                prepare=no_material,
            )
        self.assertDeclared(ShadowBlock, block)
        (bound,) = block["planes"]
        self.assertIsNone(bound["texture_index"])
        self.assertEqual(bound["horizon"]["texture_index"], 0)
        self.assertEqual(len(gltf["textures"]), 1, "the map's alone")


class TestRejections(unittest.TestCase):
    """A payload that breaks the shape is named where it breaks."""

    def _plane(self):
        return _maya_plane_record(_stamps("horizon", atlas=True, horizon=True))

    def test_a_missing_required_key_is_an_error(self):
        plane = self._plane()
        del plane["contact"]
        del plane["horizon"]["frame_b"]
        res = ShadowRecord.validate(_payload([plane]))
        self.assertEqual(
            res.errors,
            [
                "planes[0].missing required key 'contact'",
                "planes[0].horizon.missing required key 'frame_b'",
            ],
        )

    def test_a_value_of_the_wrong_type_is_an_error(self):
        """No producer writes a null here: ``source`` is ``""`` when the rig
        has none, and a block the plane does not carry is absent."""
        plane = self._plane()
        plane.update(
            source=None, follow_source=1, radius="60", canvas=[-1.0, 1.0], atlas=None
        )
        plane["horizon"]["size"] = 128.0
        res = ShadowRecord.validate(_payload([plane]))
        self.assertEqual(
            res.errors,
            [
                "planes[0].follow_source: expected boolean, got integer",
                "planes[0].radius: expected number, got string",
                "planes[0].canvas: expected 4 items, got 2",
                "planes[0].source: expected string, got null",
                "planes[0].atlas: expected a mapping for nested schema",
                "planes[0].horizon.size: expected integer, got number",
            ],
        )

    def test_a_value_outside_its_vocabulary_is_an_error(self):
        plane = self._plane()
        plane.update(type="morphing", source_type="spot")
        res = ShadowRecord.validate(_payload([plane]))
        self.assertEqual(
            res.errors,
            [
                "planes[0].type: 'morphing' is not one of ['projected', 'horizon']",
                "planes[0].source_type: 'spot' is not one of ['point', 'directional']",
            ],
        )

    def test_a_block_binding_of_the_wrong_kind_is_an_error(self):
        """A record plane bound as the pass binds one -- pruned to the block
        plane's shape, its names gone -- with bindings of the wrong kind."""
        plane = shadow_record.ShadowBlockPlane.prune(self._plane())
        plane.update(node=None, source_node="1", contact_node=2, texture_index=None)
        block = {"version": 2, "unit_scale": 0.01, "planes": [plane]}
        res = ShadowBlock.validate(block)
        self.assertEqual(
            (res.errors, res.warnings),
            (
                [
                    "planes[0].node: expected integer, got null",
                    "planes[0].source_node: expected integer, got string",
                    "planes[0].horizon.missing required key 'texture_index'",
                ],
                [],
            ),
        )

    def test_a_record_plane_is_not_yet_a_block_plane(self):
        """The bindings are required: a record read as a block names each."""
        res = ShadowBlock.validate(_payload([_maya_plane_record(_stamps())]))
        self.assertEqual(
            res.errors,
            [
                f"planes[0].missing required key {key!r}"
                for key in ("node", "source_node", "contact_node", "texture_index")
            ],
        )


class TestPublishedForms(unittest.TestCase):
    """The JSON Schema other languages generate their types from, and the
    defaults a reader fills an unsaid field with."""

    def test_both_shapes_publish_a_json_schema(self):
        for root, defs in (
            (ShadowRecord, {"ShadowPlane", "ShadowPlaneAtlas", "ShadowPlaneHorizon"}),
            (ShadowBlock, {"ShadowBlockPlane", "ShadowBlockHorizon"}),
        ):
            with self.subTest(shape=root.__name__):
                schema = root.json_schema()
                json.dumps(schema)
                self.assertEqual(schema["title"], root.__name__)
                self.assertEqual(set(schema["$defs"]), defs)
                self.assertEqual(
                    schema["required"], ["version", "unit_scale", "planes"]
                )

    def test_every_field_says_what_it_is(self):
        for root in (ShadowRecord, ShadowBlock):
            schema = root.json_schema()
            objects = {root.__name__: schema, **schema["$defs"]}
            for name, node in objects.items():
                for key, prop in node["properties"].items():
                    with self.subTest(field=f"{name}.{key}"):
                        self.assertTrue(prop.get("description"))

    def test_the_block_binds_by_index_and_says_null_where_it_may(self):
        props = ShadowBlock.json_schema()["$defs"]["ShadowBlockPlane"]["properties"]
        self.assertEqual(props["node"]["type"], "integer")
        nullable = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
        for key in ("source_node", "contact_node", "texture_index"):
            with self.subTest(field=key):
                self.assertEqual(
                    {k: v for k, v in props[key].items() if k != "description"},
                    nullable,
                )

    def test_the_vocabularies_are_pythontks_own(self):
        plane = ShadowRecord.json_schema()["$defs"]["ShadowPlane"]["properties"]
        self.assertEqual(plane["type"]["enum"], list(shadow_record.RIG_TYPES))
        self.assertEqual(plane["source_type"]["enum"], list(shadow_record.SOURCE_TYPES))

    def test_the_rig_types_are_reachable_from_the_root(self):
        """Both DCC rigs read the vocabulary off ``ptk.ShadowRecord`` -- never
        the submodule -- and it stays a class constant, not a record field."""
        self.assertEqual(ptk.ShadowRecord.RIG_TYPES, shadow_record.RIG_TYPES)
        self.assertNotIn("RIG_TYPES", {f.name for f in fields(ShadowRecord)})
        self.assertNotIn("RIG_TYPES", ShadowRecord.json_schema()["properties"])

    def test_a_plane_starts_with_no_source_or_contact_as_the_rigs_write_it(self):
        """A plane's defaults are a rig's starting values; "" is no node, the
        spelling both rigs write and every reader takes as none."""
        plane = shadow_record.ShadowPlane()
        self.assertEqual((plane.source, plane.contact), ("", ""))
        # A mutable default is the plane's own, never shared.
        self.assertIsNot(plane.canvas, shadow_record.ShadowPlane().canvas)


if __name__ == "__main__":
    unittest.main()
