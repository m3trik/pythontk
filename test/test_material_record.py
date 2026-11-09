# !/usr/bin/python
# coding=utf-8
"""The ``materials`` record's model: a material, in its family's own terms,
as the record states it.

Host-free, so every rule a producer or the FBX reader relies on is decided
here, with plain values: the hosts' producers (mayatk / blendertk
``MaterialRecords``) and the FBX reader only gather the values. Added:
2026-10-20. Version 2 (2026-11-05): the record is literal glTF 2.0 -- its
shape, the packing, and the Khronos glTF Validator over a record
(``TestTheRecordIsLiteralGltf``, ``TestPacking``, ``TestTheValidatorReadsIt``).
"""

import contextlib
import math
import os
import shutil
import tempfile
import unittest

from pythontk.core_utils.engines.scene_export.material_record import (
    MaterialFamilies,
    MaterialRecord,
)
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords


def rounded(values):
    return [round(v, 6) for v in values]


class TestPbr(unittest.TestCase):
    """A metallic-roughness surface: Standard Surface, aiStandardSurface,
    openPBR, Blender's Principled."""

    def test_the_weight_scales_the_colour_and_stands_alone_over_a_map(self):
        plain = MaterialFamilies.entry(
            "pbr", "a", {"base_color": (0.8, 0.4, 0.2), "base_weight": 0.5}
        )
        self.assertEqual(rounded(plain["baseColorFactor"]), [0.4, 0.2, 0.1, 1.0])
        mapped = MaterialFamilies.entry(
            "pbr", "a", {"base_weight": 0.5}, {"baseColor": "C:/tex/A_Base.png"}
        )
        self.assertEqual(mapped["baseColorFactor"], [0.5, 0.5, 0.5, 1.0])
        self.assertEqual(mapped["baseColorTexture"], "A_Base.png", "named by file name")
        self.assertEqual(
            mapped["sources"], {"A_Base.png": "C:/tex/A_Base.png"}, "kept for a packer"
        )
        (image,) = MaterialFamilies.payload([mapped])["images"]
        self.assertEqual(image, {"uri": "A_Base.png"}, "the record names a file")
        self.assertEqual(
            MaterialFamilies.source_files([mapped, plain]), ["A_Base.png"],
            "the scene's own maps, before any pack",
        )

    def test_a_map_stands_in_for_its_value(self):
        entry = MaterialFamilies.entry(
            "pbr",
            "a",
            {"metallic": 0.2, "roughness": 0.4},
            {"metallic": "A_Metal.png", "roughness": "A_Rough.png"},
        )
        self.assertEqual((entry["metallicFactor"], entry["roughnessFactor"]), (1.0, 1.0))
        self.assertEqual(
            (entry["metallicTexture"], entry["roughnessTexture"]),
            ("A_Metal.png", "A_Rough.png"),
        )

    def test_emission_is_its_colour_times_its_weight_and_none_at_zero(self):
        lit = MaterialFamilies.entry(
            "pbr", "a", {"emission_color": (1.0, 0.5, 0.0), "emission_weight": 2.0}
        )
        self.assertEqual(rounded(lit["emissiveFactor"]), [2.0, 1.0, 0.0])
        off = MaterialFamilies.entry(
            "pbr", "a", {"emission_weight": 0.0}, {"emission": "A_Glow.png"}
        )
        self.assertEqual(off["emissiveFactor"], [0.0, 0.0, 0.0])
        self.assertNotIn("emissiveTexture", off, "a map at weight 0 glows nothing")

    def test_a_driven_or_partial_opacity_blends_unless_the_host_states_a_mode(self):
        faded = MaterialFamilies.entry("pbr", "a", {"opacity": 0.836})
        self.assertEqual((faded["alphaMode"], round(faded["baseColorFactor"][3], 3)), ("BLEND", 0.836))
        driven = MaterialFamilies.entry("pbr", "a", {"opacity_driven": True})
        self.assertEqual(driven["alphaMode"], "BLEND")
        cut = MaterialFamilies.entry(
            "pbr", "a", {"alpha_mode": "mask", "alpha_cutoff": 0.3, "opacity_driven": True}
        )
        self.assertEqual((cut["alphaMode"], cut["alphaCutoff"]), ("MASK", 0.3))
        self.assertEqual(MaterialFamilies.entry("pbr", "a", {})["alphaMode"], "OPAQUE")

    def test_an_opacity_colour_reads_as_its_channels_mean(self):
        """The one opacity rule, in the model: a colour's channels' mean
        (the hosts' producers and the record-less FBX read each wrote it, and
        the Unity importer took the lowest channel). Added: 2026-10-31"""
        tinted = MaterialFamilies.entry("pbr", "a", {"opacity": (0.25, 0.5, 0.75)})
        self.assertEqual(round(tinted["baseColorFactor"][3], 6), 0.5)
        self.assertEqual(tinted["alphaMode"], "BLEND")
        self.assertEqual(MaterialFamilies.opacity_scalar([1.0, 1.0, 1.0]), 1.0)
        self.assertEqual(MaterialFamilies.opacity_scalar(0.3), 0.3)
        self.assertEqual(MaterialFamilies.opacity_scalar(None), 1.0)

    def test_a_transmission_is_said_not_stated(self):
        """The record has no transmission: a glass reads as its opacity alone,
        and the entry says so -- once, in the model, for every host.
        Added: 2026-10-31"""
        notes = []
        MaterialFamilies.entry("pbr", "glass", {"transmission": 0.9}, notes=notes)
        self.assertEqual(
            notes, [f"Material glass: {MaterialFamilies.TRANSMISSION_NOTE}"]
        )
        notes = []
        MaterialFamilies.entry("pbr", "solid", {"transmission": 0.0}, notes=notes)
        self.assertEqual(notes, [])

    def test_an_opacity_map_apart_from_the_colour_map_is_said(self):
        """REGRESSION (2026-10-24): a separate opacity map was dropped without
        a word, where the Stingray and legacy readers say it -- glTF takes
        opacity from the colour map's alpha alone. One the colour map is
        says nothing."""
        notes = []
        maps = {"baseColor": "a_BaseColor.png", "opacity": "a_Opacity.png"}
        entry = MaterialFamilies.entry("pbr", "a", {}, maps, notes=notes)
        self.assertEqual(entry["alphaMode"], "BLEND")
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("a_Opacity.png", notes[0])
        self.assertIn("colour map's alpha", notes[0])
        notes = []
        same = {"baseColor": "a_BaseColor.png", "opacity": "a_BaseColor.png"}
        MaterialFamilies.entry("pbr", "a", {}, same, notes=notes)
        self.assertEqual(notes, [])


class TestStingray(unittest.TestCase):
    MAPS = {
        "baseColor": "L_Base.png",
        "metallic": "L_Metal.png",
        "roughness": "L_Rough.png",
        "emission": "L_Glow.png",
        "ambientOcclusion": "L_AO.png",
    }

    def test_a_map_is_used_only_while_its_switch_is_on(self):
        on = {
            "use_color_map": 1.0,
            "use_metallic_map": 1.0,
            "use_roughness_map": 0.0,
            "use_emissive_map": 1.0,
            "use_ao_map": 1.0,
            "base_color": (0.5, 0.5, 0.5),
            "roughness": 0.3,
            "emissive_intensity": 0.75,
        }
        entry = MaterialFamilies.entry("stingray_pbs", "L", on, self.MAPS)
        self.assertEqual(entry["baseColorFactor"], [1.0, 1.0, 1.0, 1.0], "not the inert 0.5")
        self.assertEqual((entry["metallicFactor"], entry["metallicTexture"]), (1.0, "L_Metal.png"))
        self.assertEqual((entry["roughnessFactor"], "roughnessTexture" in entry), (0.3, False))
        self.assertEqual((rounded(entry["emissiveFactor"]), entry["emissiveTexture"]), ([0.75] * 3, "L_Glow.png"))
        self.assertEqual(entry["occlusionTexture"], "L_AO.png")

    def test_tiling_is_stated_as_a_texture_transform(self):
        """Version 2 states what version 1 could only say: the maps' UV
        transform rides each map as KHR_texture_transform, in glTF's UV
        space (V down). Added: 2026-11-05"""
        notes = []
        entry = MaterialFamilies.entry(
            "stingray_pbs",
            "L",
            {"use_color_map": 1.0, "uv_scale": (2.0, 3.0), "uv_offset": (0.25, 0.5)},
            {"baseColor": "L_Base.png"},
            notes=notes,
        )
        self.assertEqual(notes, [])
        record = MaterialFamilies.payload([entry])
        info = record["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]
        self.assertEqual(
            info["extensions"]["KHR_texture_transform"],
            {"offset": [0.25, 1.0 - 3.0 - 0.5], "scale": [2.0, 3.0]},
        )
        self.assertIn("KHR_texture_transform", record["extensionsUsed"])


class TestLegacy(unittest.TestCase):
    def test_the_named_policy_is_fbx2gltfs(self):
        """The look every approved WebXR deliverable was signed off through."""
        self.assertEqual(MaterialFamilies.LEGACY_POLICY, "fbx2gltf")
        phong = MaterialFamilies.entry(
            "legacy", "P", {"ShadingModel": "phong", "Shininess": 30.0}
        )
        self.assertEqual(phong["metallicFactor"], 0.4)
        self.assertAlmostEqual(phong["roughnessFactor"], math.sqrt(2.0 / 32.0))
        lambert = MaterialFamilies.entry("legacy", "L", {"ShadingModel": "lambert"})
        self.assertEqual((lambert["metallicFactor"], lambert["roughnessFactor"]), (0.2, 0.8))

    def test_any_transparency_blends(self):
        lens = MaterialFamilies.entry(
            "legacy",
            "GLASS",
            {
                "ShadingModel": "phong",
                "DiffuseColor": (0.5, 0.5, 0.5),
                "DiffuseFactor": 0.8,
                "TransparentColor": (0.9, 0.9, 0.9),
                "TransparencyFactor": 1.0,
            },
        )
        self.assertEqual(rounded(lens["baseColorFactor"]), [0.4, 0.4, 0.4, 0.1])
        self.assertEqual(lens["alphaMode"], "BLEND")


class TestUnlit(unittest.TestCase):
    def test_a_hosts_unlit_model_shows_its_colour_and_opacity(self):
        entry = MaterialFamilies.entry(
            "unlit", "U", {"color": (0.1, 0.2, 0.3), "opacity": 0.5}
        )
        self.assertTrue(entry["unlit"])
        self.assertEqual(rounded(entry["baseColorFactor"]), [0.1, 0.2, 0.3, 0.5])
        self.assertEqual(entry["alphaMode"], "BLEND")
        self.assertEqual((entry["metallicFactor"], entry["roughnessFactor"]), (0.0, 1.0))

    def test_whatever_it_wears_the_scene_can_shade_it_unlit(self):
        """A projected shadow's plane: darkness laid over what is under it,
        never a surface the environment lights."""
        entry = MaterialFamilies.entry(
            "pbr", "SHADOW", {"metallic": 1.0, "roughness": 0.1}, unlit=True
        )
        self.assertTrue(entry["unlit"])
        self.assertEqual((entry["metallicFactor"], entry["roughnessFactor"]), (0.0, 1.0))


class TestRecord(unittest.TestCase):
    def test_an_unknown_family_is_refused(self):
        with self.assertRaises(KeyError):
            MaterialFamilies.entry("nope", "a", {})

    def test_the_payload_keeps_the_first_of_a_repeated_name_and_validates(self):
        payload = MaterialFamilies.payload(
            [
                MaterialFamilies.entry("pbr", "a", {"metallic": 0.5}),
                MaterialFamilies.entry("pbr", "a", {"metallic": 0.9}),
                MaterialFamilies.entry("legacy", "b", {"ShadingModel": "lambert"}),
            ]
        )
        self.assertEqual([e["name"] for e in payload["materials"]], ["a", "b"])
        pbr = payload["materials"][0]["pbrMetallicRoughness"]
        self.assertEqual(pbr["metallicFactor"], 0.5)
        self.assertEqual(payload["version"], 2)
        result = MaterialRecord.validate(payload)
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.warnings, [])

    def test_finish_clamps_every_factor_and_drops_what_says_nothing(self):
        entry = MaterialFamilies.finish(
            {
                "name": "a",
                "baseColorFactor": [1.5, -1, 0.5, 2],
                "metallicFactor": 3,
                "roughnessFactor": -2,
                "emissiveFactor": [0, 0, 0],
                "emissiveTexture": "dark.png",
                "alphaMode": "nonsense",
                "normalTexture": None,
            }
        )
        self.assertEqual(entry["baseColorFactor"], [1.0, 0.0, 0.5, 1.0])
        self.assertEqual((entry["metallicFactor"], entry["roughnessFactor"]), (1.0, 0.0))
        self.assertEqual(entry["alphaMode"], "OPAQUE")
        self.assertNotIn("emissiveTexture", entry)
        self.assertNotIn("normalTexture", entry)

    def test_texture_files_names_each_image_once_in_record_order(self):
        payload = MaterialFamilies.payload(
            [
                MaterialFamilies.entry(
                    "pbr", "a", {}, {"baseColor": "A.png", "metallic": "A_Metal.png"}
                ),
                MaterialFamilies.entry("pbr", "b", {}, {"baseColor": "A.png"}),
            ]
        )
        self.assertEqual(
            MaterialFamilies.texture_files(payload),
            ["A.png", "A_ORM.png"],
            "a pack not written: the record names the image it would be",
        )

    def test_a_hosts_raw_value_reads_as_floats(self):
        """Public, for the hosts' producers that read raw shader values
        (they called the private ``_num`` / ``_vec`` across the package
        boundary). Added: 2026-10-24"""
        self.assertEqual(MaterialFamilies.as_float((0.25, 0.5, 0.75), 1.0), 0.25)
        self.assertEqual(MaterialFamilies.as_float("x", 0.3), 0.3)
        self.assertEqual(MaterialFamilies.as_float(None, 2), 2.0)
        self.assertEqual(MaterialFamilies.as_floats(0.5, 3, 0.0), [0.5, 0.5, 0.5])
        self.assertEqual(MaterialFamilies.as_floats((1, "x"), 3, 0.0), [1.0, 0.0, 0.0])
        self.assertEqual(MaterialFamilies.as_floats(None, 2, 1.0), [1.0, 1.0])


class TestBakedOcclusion(unittest.TestCase):
    """``without_baked_occlusion``: the bake holds an object's occlusion, so a
    material only lightmapped objects wear states none; one that baked and
    unbaked objects share keeps its map, said once with the fix. The rule
    both hosts' producers apply before the record is built (lightmaps v3,
    2026-11-07)."""

    @staticmethod
    def _entries():
        maps = {"baseColor": "W_BaseColor.png", "ambientOcclusion": "W_AO.png"}
        return [
            MaterialFamilies.entry("pbr", "Wall", {}, maps),
            MaterialFamilies.entry("pbr", "Shared", {}, dict(maps, ambientOcclusion="S_AO.png")),
            MaterialFamilies.entry("pbr", "Prop", {}, dict(maps, ambientOcclusion="P_AO.png")),
            MaterialFamilies.entry("pbr", "Plain", {}, {"baseColor": "Plain.png"}),
        ]

    def test_only_a_material_baked_objects_alone_wear_loses_its_occlusion(self):
        notes = []
        entries = self._entries()
        out = MaterialFamilies.without_baked_occlusion(
            entries,
            baked={"Wall", "Shared", "Plain"},
            unbaked={"Shared", "Prop"},
            notes=notes,
        )
        by_name = {e["name"]: e for e in out}
        self.assertNotIn("occlusionTexture", by_name["Wall"], "the bake holds it")
        self.assertEqual(by_name["Shared"]["occlusionTexture"], "S_AO.png")
        self.assertEqual(by_name["Prop"]["occlusionTexture"], "P_AO.png")
        self.assertEqual(by_name["Wall"]["baseColorTexture"], "W_BaseColor.png")
        self.assertEqual(len(notes), 1, notes)
        (note,) = notes
        for part in ("Shared", "S_AO.png", "lightmapped", "their own material"):
            with self.subTest(part=part):
                self.assertIn(part, note)
        self.assertEqual(entries[0]["occlusionTexture"], "W_AO.png", "the input is left as it was")
        payload = MaterialFamilies.payload(out)
        wall = next(m for m in payload["materials"] if m["name"] == "Wall")
        self.assertNotIn("occlusionTexture", wall)

    def test_nothing_baked_changes_nothing(self):
        entries = self._entries()
        notes = []
        self.assertEqual(
            MaterialFamilies.without_baked_occlusion(entries, (), {"Wall"}, notes=notes),
            entries,
        )
        self.assertEqual(notes, [])


class TestTheRecordAWriteShips(unittest.TestCase):
    """``MaterialFamilies.export_record``: the ``materials`` producer's body,
    written once for both hosts, the occlusion rule decided against the
    ``lightmaps`` record THIS write ships -- never the baker's markers -- with
    the materials each mesh SHAPE wears (a shape lightmapped when a transform
    it is instanced under is one of the record's objects). The hosts gather
    the shapes. Measured 2026-11-07: a group root as the scope landed every
    baked material among the unbaked ones too, so each kept its occlusion
    with a false note. Added: 2026-11-07"""

    def setUp(self):
        from pythontk.core_utils.engines.scene_export.export_snapshot import (
            ExportContext,
        )

        self.ExportContext = ExportContext
        self.entries = TestBakedOcclusion._entries()

    @staticmethod
    def _lightmaps(*paths):
        return SceneRecords.LIGHTMAPS.make(
            {
                "objects": [
                    {
                        "name": p[-1],
                        "hierarchy": list(p),
                        "map": "L.exr",
                        "scaleOffset": [1, 1, 0, 0],
                    }
                    for p in paths
                ]
            }
        )

    #: Per shape: the transforms it is instanced under, the materials it
    #: wears (a per-face assignment is one shape wearing several).
    WORN = [
        ([("ROOM", "wall")], ["Wall", "Plain"]),
        ([("ROOM", "crate"), ("ROOM", "crate1")], ["Shared"]),  # instanced
        ([("ROOM", "prop")], ["Prop", "Shared"]),
    ]

    def _record(self, ctx, worn=None, store=None):
        asked = []

        def gather():
            asked.append(True)
            return self.WORN if worn is None else worn

        record = MaterialFamilies.export_record(ctx, self.entries, gather, store=store)
        return {m["name"]: m for m in record.payload["materials"]}, asked

    def test_occlusion_leaves_what_only_the_shipped_bake_wears(self):
        ctx = self.ExportContext()
        # A shape instanced under a baked transform and an unbaked one is baked.
        ctx.records[SceneRecords.LIGHTMAPS.key] = self._lightmaps(
            ("ROOM", "wall"), ("ROOM", "crate")
        )
        materials, _ = self._record(ctx)
        self.assertNotIn("occlusionTexture", materials["Wall"], "the bake holds it")
        self.assertIn("occlusionTexture", materials["Shared"], "the prop needs it")
        self.assertIn("occlusionTexture", materials["Prop"])
        said = [n for n in ctx.notes if "occlusion" in n]
        self.assertEqual(len(said), 1, ctx.notes)
        self.assertIn("Shared", said[0])

    def test_a_baked_object_the_write_leaves_out_bakes_nothing_it_ships(self):
        ctx = self.ExportContext(nodes=frozenset({"ROOM", "crate", "crate1", "prop"}))
        ctx.records[SceneRecords.LIGHTMAPS.key] = self._lightmaps(("ROOM", "wall"))
        materials, _ = self._record(ctx, worn=self.WORN[1:])
        self.assertIn("occlusionTexture", materials["Shared"])
        self.assertEqual([n for n in ctx.notes if "occlusion" in n], [])
        # ...and one whose stored copy ships names only what ships.
        stored = self.ExportContext(
            mode=self.ExportContext.HANDOFF, nodes=frozenset({"ROOM", "prop"})
        )

        class Store:
            @staticmethod
            def read(scope, key):
                return (
                    self._lightmaps(("ROOM", "crate")).text
                    if key == "lightmaps"
                    else None
                )

        materials, _ = self._record(stored, worn=[self.WORN[2]], store=Store)
        self.assertIn("occlusionTexture", materials["Shared"], "crate does not ship")

    def test_no_lightmaps_shipped_keeps_every_occlusion_and_asks_nothing(self):
        for ctx in (
            self.ExportContext(),  # absent
            self.ExportContext(records={SceneRecords.LIGHTMAPS.key: None}),  # refused
            self.ExportContext(
                mode=self.ExportContext.HANDOFF,
                withheld={SceneRecords.LIGHTMAPS.key: "refused"},
            ),
        ):
            with self.subTest(mode=ctx.mode, withheld=ctx.withheld):
                materials, asked = self._record(ctx)
                for name in ("Wall", "Shared", "Prop"):
                    self.assertIn("occlusionTexture", materials[name], name)
                self.assertEqual(asked, [], "no shape is gathered for nothing")
                self.assertFalse([n for n in ctx.notes if "occlusion" in n], ctx.notes)

    def test_nothing_to_state_is_no_record(self):
        ctx = self.ExportContext()
        self.assertIsNone(MaterialFamilies.export_record(ctx, [], lambda: []))



def _surface(folder):
    """A metal-rough surface over real maps in *folder*: its statement."""
    from PIL import Image

    paths = {}
    for channel, name, value in (
        ("baseColor", "S_BaseColor.png", (200, 40, 40)),
        ("metallic", "S_Metallic.png", (255, 255, 255)),
        ("roughness", "S_Roughness.png", (64, 64, 64)),
        ("ambientOcclusion", "S_AO.png", (128, 128, 128)),
        ("normal", "S_Normal.png", (128, 128, 255)),
        ("emission", "S_Emissive.png", (255, 255, 255)),
    ):
        paths[channel] = os.path.join(folder, name)
        Image.new("RGB", (4, 4), value).save(paths[channel])
    return MaterialFamilies.entry(
        "pbr", "ns:S", {"emission_weight": 4.0, "uv_scale": (2.0, 2.0)}, paths
    )


class TestTheRecordIsLiteralGltf(unittest.TestCase):
    """Version 2: the record IS a glTF 2.0 document -- only materials,
    textures, images and samplers, with glTF's asset and extensionsUsed.
    Added: 2026-11-05"""

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="material_record_")
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)

    def test_a_surface_is_a_gltf_material_over_packed_metal_and_roughness(self):
        record = MaterialFamilies.payload(
            [_surface(self.folder)], pack=MaterialFamilies.pack_metallic_roughness
        )
        self.assertEqual(
            set(record),
            {"version", "asset", "extensionsUsed", "materials", "textures", "images"},
        )
        self.assertEqual(record["asset"], {"version": "2.0"})
        (material,) = record["materials"]
        self.assertEqual(
            MaterialFamilies.material_files(record, material),
            {
                "baseColorTexture": "S_BaseColor.png",
                "metallicRoughnessTexture": "S_ORM.png",
                "normalTexture": "S_Normal.png",
                "occlusionTexture": "S_ORM.png",
                "emissiveTexture": "S_Emissive.png",
            },
            "metal, roughness and occlusion in one image, glTF's layout",
        )
        pbr = material["pbrMetallicRoughness"]
        self.assertEqual((pbr["metallicFactor"], pbr["roughnessFactor"]), (1.0, 1.0))
        self.assertTrue(os.path.isfile(MaterialFamilies.packed_file("S_ORM.png")))
        self.assertFalse(
            os.path.exists(os.path.join(self.folder, "S_ORM.png")),
            "a delivery file, never written beside its sources",
        )
        self.assertEqual(SceneRecords.problems(SceneRecords.MATERIALS, record), [])

    def test_emission_above_one_is_the_strength_never_the_factor(self):
        """glTF caps emissiveFactor at 1: version 1 carried 4.0 there."""
        record = MaterialFamilies.payload(
            [
                MaterialFamilies.entry(
                    "pbr", "hot", {"emission_color": (1.0, 0.5, 0.0), "emission_weight": 4.0}
                )
            ]
        )
        (material,) = record["materials"]
        self.assertEqual(material["emissiveFactor"], [1.0, 0.5, 0.0])
        self.assertEqual(
            material["extensions"],
            {"KHR_materials_emissive_strength": {"emissiveStrength": 4.0}},
        )
        self.assertEqual(MaterialFamilies.emission(material), [4.0, 2.0, 0.0])
        self.assertEqual(record["extensionsUsed"], ["KHR_materials_emissive_strength"])
        dim = MaterialFamilies.payload(
            [MaterialFamilies.entry("pbr", "dim", {"emission_color": (0.5, 0.5, 0.5)})]
        )["materials"][0]
        self.assertNotIn("extensions", dim, "at or below 1 the factor holds it")

    def test_unlit_is_khr_materials_unlit(self):
        record = MaterialFamilies.payload(
            [MaterialFamilies.entry("unlit", "U", {"color": (0.1, 0.2, 0.3)})]
        )
        (material,) = record["materials"]
        self.assertEqual(material["extensions"], {"KHR_materials_unlit": {}})
        pbr = material["pbrMetallicRoughness"]
        self.assertEqual((pbr["metallicFactor"], pbr["roughnessFactor"]), (0.0, 1.0))
        self.assertEqual(record["extensionsUsed"], ["KHR_materials_unlit"])

    def test_one_map_already_in_gltfs_layout_is_bound_as_it_is(self):
        """An ORM used for both metal and roughness is not packed again."""
        packs = []
        record = MaterialFamilies.payload(
            [
                MaterialFamilies.entry(
                    "pbr",
                    "a",
                    {},
                    {"metallic": "C:/t/a_ORM.png", "roughness": "C:/t/a_ORM.png"},
                )
            ],
            pack=lambda *maps: packs.append(maps),
        )
        self.assertEqual(packs, [])
        self.assertEqual(MaterialFamilies.texture_files(record), ["a_ORM.png"])

    def test_maps_not_to_hand_read_as_absent_and_are_said(self):
        """A GLB build's packer finds no map the FBX carries: the one rule
        (``absent_map``) -- a metal map's factor 1 dielectric, a constant
        metal and the roughness factor kept."""
        notes = []
        record = MaterialFamilies.payload(
            [
                MaterialFamilies.entry("pbr", "a", {}, {"metallic": "C:/gone/a_Metal.png"}),
                MaterialFamilies.entry(
                    "pbr", "b", {"metallic": 0.4}, {"roughness": "C:/gone/b_Rough.png"}
                ),
            ],
            pack=lambda *maps: None,
            notes=notes,
        )
        a, b = (m["pbrMetallicRoughness"] for m in record["materials"])
        self.assertEqual(a["metallicFactor"], 0.0, "the map supplied it: dielectric")
        self.assertNotIn("metallicRoughnessTexture", a)
        self.assertEqual((b["metallicFactor"], b["roughnessFactor"]), (0.4, 1.0))
        self.assertEqual(len(notes), 2)
        self.assertIn("a_Metal.png", notes[0])

    def test_the_one_rule_for_a_missing_map(self):
        """``absent_map``: metal reset only where its factor is 1 or more,
        a roughness constant kept, emission zero (its strength with it)."""
        metal = {
            "pbrMetallicRoughness": {
                "metallicFactor": 1.0,
                "roughnessFactor": 0.3,
                "metallicRoughnessTexture": {"index": 0},
            }
        }
        MaterialFamilies.absent_map(metal, "metallicRoughnessTexture")
        self.assertEqual(
            metal, {"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.3}}
        )
        constant = {"pbrMetallicRoughness": {"metallicFactor": 0.6}}
        MaterialFamilies.absent_map(constant, "metallicRoughnessTexture")
        self.assertEqual(constant["pbrMetallicRoughness"]["metallicFactor"], 0.6)
        glow = {
            "emissiveFactor": [1.0, 0.5, 0.0],
            "emissiveTexture": {"index": 1},
            "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 4.0}},
        }
        MaterialFamilies.absent_map(glow, "emissiveTexture")
        self.assertEqual(glow, {"emissiveFactor": [0.0, 0.0, 0.0]})

    def test_version_1_is_refused(self):
        """The flat shape: a reader refuses it, its cure a re-export."""
        flat = {
            "version": 1,
            "materials": [
                {
                    "name": "a",
                    "baseColorFactor": [1, 1, 1, 1],
                    "metallicFactor": 0,
                    "roughnessFactor": 1,
                    "emissiveFactor": [2, 2, 2],
                    "alphaMode": "OPAQUE",
                }
            ],
        }
        self.assertEqual(
            SceneRecords.problems(SceneRecords.MATERIALS, flat),
            ["materials: v1 is not the v2 a reader reads"],
        )

    def test_overlay_takes_a_named_material_whole_and_states_the_rest(self):
        carried = MaterialFamilies.payload(
            [
                MaterialFamilies.entry(
                    "pbr", "a", {"base_color": (1, 0, 0)}, {"baseColor": "red.png"}
                )
            ]
        )
        merged = MaterialFamilies.overlay(
            carried,
            [
                MaterialFamilies.entry("pbr", "a", {"base_color": (0, 1, 0)}),
                MaterialFamilies.entry("pbr", "b", {}, {"normal": "b_Normal.png"}),
            ],
        )
        a, b = merged["materials"]
        self.assertEqual(
            MaterialFamilies.material_files(merged, a), {"baseColorTexture": "red.png"}
        )
        self.assertEqual(
            MaterialFamilies.material_files(merged, b), {"normalTexture": "b_Normal.png"}
        )


class TestPacking(unittest.TestCase):
    """The producer's cost: metal and roughness packed into one image at
    export, in glTF's layout. The packed image is a DELIVERY file: it is
    written into the pack store (``TempArtifacts``), stamped with a content
    hash of its sources, found by its file name by the delivery's
    dependency pass, and never written into a source folder. Added:
    2026-11-05"""

    def setUp(self):
        from PIL import Image

        self.folder = tempfile.mkdtemp(prefix="material_pack_")
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)
        self.metal = os.path.join(self.folder, "rock_Metallic.png")
        self.rough = os.path.join(self.folder, "rock_Roughness.png")
        Image.new("L", (4, 4), 200).save(self.metal)
        Image.new("L", (4, 4), 50).save(self.rough)

    def listing(self):
        return sorted(
            (name, os.path.getmtime(os.path.join(self.folder, name)))
            for name in os.listdir(self.folder)
        )

    def test_the_layout_is_gltfs_and_an_unmapped_channel_is_white(self):
        from PIL import Image

        path = MaterialFamilies.pack_metallic_roughness(None, self.rough, self.metal)
        self.assertEqual(os.path.basename(path), "rock_ORM.png")
        with Image.open(path) as image:
            self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 50, 200))

    def test_it_is_a_delivery_file_never_written_into_the_source_folder(self):
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        before = self.listing()
        path = MaterialFamilies.pack_metallic_roughness(None, self.rough, self.metal)
        self.assertEqual(self.listing(), before, "nothing written beside the sources")
        self.assertTrue(TempArtifacts.contains(path), "the pack store's, a temp artifact")
        self.assertEqual(MaterialFamilies.packed_file("ROCK_orm.png"), path)

    def test_a_read_only_source_folder_still_packs(self):
        with _read_only(self.folder):
            path = MaterialFamilies.pack_metallic_roughness(None, self.rough, self.metal)
        self.assertTrue(os.path.isfile(path))

    def test_an_unchanged_source_reuses_its_pack_and_a_changed_one_repacks(self):
        from PIL import Image

        path = MaterialFamilies.pack_metallic_roughness(None, self.rough, self.metal)
        stamp = os.stat(path).st_mtime_ns
        os.utime(self.metal)  # touched, not changed: the content is what counts
        self.assertEqual(
            MaterialFamilies.pack_metallic_roughness(None, self.rough, self.metal), path
        )
        with open(path, "rb") as fh:
            first = fh.read()
        Image.new("L", (4, 4), 10).save(self.metal)
        again = MaterialFamilies.pack_metallic_roughness(None, self.rough, self.metal)
        self.assertNotEqual(again, path, "another content, another pack")
        self.assertEqual(os.path.basename(again), "rock_ORM.png")
        with Image.open(again) as image:
            self.assertEqual(image.convert("RGB").getpixel((0, 0))[2], 10)
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(), first, "the reused pack was not rewritten")
        self.assertGreaterEqual(os.stat(path).st_mtime_ns, stamp)
        self.assertEqual(MaterialFamilies.packed_file("rock_ORM.png"), again)

    def test_a_16_bit_source_packs_at_its_value(self):
        import numpy as np
        from PIL import Image

        Image.fromarray(np.full((4, 4), 30000, dtype=np.uint16)).save(self.rough)
        path = MaterialFamilies.pack_metallic_roughness(None, self.rough, self.metal)
        with Image.open(path) as image:
            self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 117, 200))

    def test_nothing_to_pack_packs_nothing_and_a_missing_source_is_an_error(self):
        self.assertIsNone(MaterialFamilies.pack_metallic_roughness(self.metal, None, None))
        with self.assertRaises(FileNotFoundError) as caught:
            MaterialFamilies.pack_metallic_roughness(None, self.rough, "C:/gone/x.png")
        self.assertIn("x.png", str(caught.exception))

    def test_a_pack_that_cannot_be_written_is_an_export_error_with_its_fix(self):
        """Never a silent dielectric: the producer's record fails, saying why."""

        def refused(*maps, name=None):
            raise PermissionError("the pack store is not writable")

        with self.assertRaises(OSError) as caught:
            MaterialFamilies.payload(
                [MaterialFamilies.entry("pbr", "a", {}, {"metallic": self.metal})],
                pack=refused,
            )
        self.assertIn("Material a", str(caught.exception))
        self.assertIn("rock_Metallic.png", str(caught.exception))

    def test_two_packs_of_one_name_in_a_record_are_named_apart(self):
        from PIL import Image

        other = os.path.join(self.folder, "other")
        os.makedirs(other)
        rough = os.path.join(other, "rock_Roughness.png")
        Image.new("L", (4, 4), 90).save(rough)
        record = MaterialFamilies.payload(
            [
                MaterialFamilies.entry("pbr", "a", {}, {"roughness": self.rough}),
                MaterialFamilies.entry("pbr", "b", {}, {"roughness": rough}),
            ],
            pack=MaterialFamilies.pack_metallic_roughness,
        )
        a, b = (
            MaterialFamilies.material_files(record, m)["metallicRoughnessTexture"]
            for m in record["materials"]
        )
        self.assertEqual(a, "rock_ORM.png")
        self.assertRegex(b, r"^rock_[0-9a-f]{8}_ORM\.png$")
        self.assertIsNotNone(MaterialFamilies.packed_file(b))


@contextlib.contextmanager
def _read_only(folder):
    """*folder* made unwritable for the block: no file can be created in it."""
    import stat
    import subprocess
    import sys

    if sys.platform == "win32":
        subprocess.run(
            ["icacls", folder, "/deny", "*S-1-1-0:(W,AD,WD)"],
            check=True,
            capture_output=True,
        )
        try:
            yield
        finally:
            subprocess.run(
                ["icacls", folder, "/remove:d", "*S-1-1-0"], capture_output=True
            )
    else:
        mode = os.stat(folder).st_mode
        os.chmod(folder, stat.S_IREAD | stat.S_IEXEC)
        try:
            yield
        finally:
            os.chmod(folder, mode)


class TestTheValidatorReadsIt(unittest.TestCase):
    """The open-format rule: the Khronos glTF Validator checks the record as
    it stands -- the record (its 'version' left out, the one key that is not
    glTF's) wrapped as a minimal glTF document beside its images, every
    material on a primitive so none is unused. Added: 2026-11-05"""

    @classmethod
    def setUpClass(cls):
        from conftest import gltf_validator
        from pythontk.file_utils.mesh_convert.gltf_validator import GltfValidator

        gltf_validator()  # skips with its reason, or fails in CI
        cls.validator = GltfValidator

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="material_validator_")
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)

    @staticmethod
    def wrapped(record):
        """*record* as a minimal glTF document: one triangle per material."""
        import base64
        import struct

        document = {k: v for k, v in record.items() if k != "version"}
        parts = [
            struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0),  # POSITION
            struct.pack("<9f", 0, 0, 1, 0, 0, 1, 0, 0, 1),  # NORMAL
            struct.pack("<12f", *(1, 0, 0, 1) * 3),  # TANGENT
            struct.pack("<6f", 0, 0, 1, 0, 0, 1),  # TEXCOORD_0
        ]
        data = b"".join(parts)
        document["buffers"] = [
            {
                "byteLength": len(data),
                "uri": "data:application/octet-stream;base64,"
                + base64.b64encode(data).decode("ascii"),
            }
        ]
        offsets = [sum(len(p) for p in parts[:i]) for i in range(len(parts))]
        document["bufferViews"] = [
            {"buffer": 0, "byteOffset": o, "byteLength": len(p), "target": 34962}
            for o, p in zip(offsets, parts)
        ]
        document["accessors"] = [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [1, 1, 0],
            },
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC2"},
        ]
        attributes = {"POSITION": 0, "NORMAL": 1, "TANGENT": 2, "TEXCOORD_0": 3}
        document["meshes"] = [
            {
                "primitives": [
                    {"attributes": dict(attributes), "material": i}
                    for i in range(len(document["materials"]))
                ]
            }
        ]
        document["nodes"] = [{"mesh": 0}]
        document["scenes"] = [{"nodes": [0]}]
        document["scene"] = 0
        return document

    def test_a_record_of_every_feature_validates_clean(self):
        record = MaterialFamilies.payload(
            [
                _surface(self.folder),
                MaterialFamilies.entry("unlit", "U", {"color": (0, 0, 0), "opacity": 0.5}),
                MaterialFamilies.entry(
                    "pbr", "cut", {"alpha_mode": "MASK", "alpha_cutoff": 0.3}
                ),
            ],
            pack=MaterialFamilies.pack_metallic_roughness,
        )
        # The delivery's images beside the document: the pack is the store's.
        for name in MaterialFamilies.texture_files(record):
            if not os.path.isfile(os.path.join(self.folder, name)):
                shutil.copy(MaterialFamilies.packed_file(name), self.folder)
        report = self.validator.validate_document(self.wrapped(record), self.folder)
        issues = report["issues"]
        self.assertEqual(
            (issues["numErrors"], issues["numWarnings"]),
            (0, 0),
            self.validator.messages(report, worst=3),
        )
        self.assertEqual(
            sorted(report["info"].get("extensionsUsed") or []),
            [
                "KHR_materials_emissive_strength",
                "KHR_materials_unlit",
                "KHR_texture_transform",
            ],
        )

    def test_version_1s_emission_is_what_the_validator_refuses(self):
        """The reason for version 2, measured: emission above 1 in the
        factor is a glTF error."""
        record = MaterialFamilies.payload(
            [
                MaterialFamilies.entry(
                    "pbr", "hot", {"emission_color": (1, 1, 1), "emission_weight": 4.0}
                )
            ]
        )
        flat = self.wrapped(record)
        flat["materials"][0]["emissiveFactor"] = [4.0, 4.0, 4.0]
        flat["materials"][0].pop("extensions")
        flat.pop("extensionsUsed")
        report = self.validator.validate_document(flat, self.folder)
        self.assertGreater(report["issues"]["numErrors"], 0)


if __name__ == "__main__":
    unittest.main()
