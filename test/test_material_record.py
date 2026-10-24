# !/usr/bin/python
# coding=utf-8
"""The ``materials`` record's model: a material, in its family's own terms,
as the record states it.

Host-free, so every rule a producer or the FBX reader relies on is decided
here, with plain values: the hosts' producers (mayatk / blendertk
``MaterialRecords``) and the FBX reader only gather the values. Added:
2026-10-20
"""

import math
import unittest

from pythontk.core_utils.engines.scene_export.material_record import (
    MaterialFamilies,
    MaterialRecord,
)


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

    def test_tiling_is_said_not_stated(self):
        notes = []
        MaterialFamilies.entry(
            "stingray_pbs", "L", {"uv_scale": (2.0, 2.0)}, {}, notes=notes
        )
        self.assertEqual(len(notes), 1)
        self.assertIn("Material L", notes[0])
        self.assertIn("tile", notes[0])


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
        self.assertEqual(payload["materials"][0]["metallicFactor"], 0.5)
        self.assertEqual(payload["version"], 1)
        result = MaterialRecord.validate(payload)
        self.assertTrue(result.ok, result.errors)

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

    def test_texture_files_names_each_map_once_in_record_order(self):
        payload = MaterialFamilies.payload(
            [
                MaterialFamilies.entry(
                    "pbr", "a", {}, {"baseColor": "A.png", "metallic": "M.png"}
                ),
                MaterialFamilies.entry("pbr", "b", {}, {"baseColor": "A.png"}),
            ]
        )
        self.assertEqual(MaterialFamilies.texture_files(payload), ["A.png", "M.png"])

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


if __name__ == "__main__":
    unittest.main()
