#!/usr/bin/python
# coding=utf-8
"""Unit tests for the per-map output-format template layer.

Run with:
    python -m pytest test_output_template.py -v
"""
import unittest

from pythontk import DeliveryBudget, OutputSpec, OutputTemplate, OutputTemplates
from pythontk.core_utils.engines.textures.map_registry import WF, MapRegistry


class OutputSpecTest(unittest.TestCase):
    def test_defaults(self):
        s = OutputSpec()
        self.assertEqual((s.ext, s.bit_depth, s.compression), ("png", 8, None))

    def test_dict_roundtrip(self):
        s = OutputSpec("tga", 16, "DXT5")
        self.assertEqual(OutputSpec.from_dict(s.to_dict()), s)

    def test_from_dict_partial(self):
        # Missing keys fall back to defaults — tolerant of hand-written presets.
        self.assertEqual(OutputSpec.from_dict({"ext": "tiff"}), OutputSpec("tiff", 8, None))


class OutputTemplateTest(unittest.TestCase):
    def test_resolve_override_then_default(self):
        t = OutputTemplate(
            default=OutputSpec("png", 8),
            overrides={"Height": OutputSpec("png", 16)},
        )
        self.assertEqual(t.resolve("Height").bit_depth, 16)  # override hit
        self.assertEqual(t.resolve("Base_Color"), OutputSpec("png", 8))  # default
        self.assertEqual(t.resolve(None), OutputSpec("png", 8))  # no map type

    def test_dict_roundtrip(self):
        t = OutputTemplates.get(WF.UE)
        self.assertEqual(OutputTemplate.from_dict(t.to_dict()).resolve("Normal"), t.resolve("Normal"))

    def test_dict_roundtrip_carries_budget(self):
        # A template that loses its budget on the way through a preset file would
        # silently stop warning — the failure mode advisory limits can't afford.
        t = OutputTemplates.get(WF.GLTF)
        self.assertEqual(OutputTemplate.from_dict(t.to_dict()).budget, t.budget)

    def test_default_budget_is_unbudgeted(self):
        self.assertEqual(OutputTemplate().budget, DeliveryBudget())


class DeliveryBudgetTest(unittest.TestCase):
    """The advisory tier: reports, never mutates."""

    def test_defaults_are_inert(self):
        b = DeliveryBudget()
        self.assertEqual((b.max_size, b.force_pot), (None, False))
        self.assertEqual(b.check(8192, 8192), [])  # nothing configured, nothing to say

    def test_max_size_flags_only_when_exceeded(self):
        b = DeliveryBudget(max_size=2048)
        self.assertEqual(b.check(2048, 2048), [])  # at budget, not over
        self.assertEqual(b.check(1024, 512), [])
        (message,) = b.check(4096, 2048)  # longest edge decides
        self.assertIn("4096x2048", message)
        self.assertIn("2048", message)

    def test_pot_flag_is_independent_of_size(self):
        b = DeliveryBudget(force_pot=True)
        self.assertEqual(b.check(1024, 512), [])
        self.assertEqual(len(b.check(1000, 512)), 1)
        self.assertEqual(len(b.check(1000, 500)), 1)  # one message, not one per edge

    def test_both_rules_report_independently(self):
        self.assertEqual(len(DeliveryBudget(4096, True).check(8000, 8000)), 2)

    def test_zero_dimension_is_not_pot(self):
        self.assertEqual(len(DeliveryBudget(force_pot=True).check(0, 512)), 1)

    def test_dict_roundtrip(self):
        b = DeliveryBudget(max_size=2048, force_pot=True)
        self.assertEqual(DeliveryBudget.from_dict(b.to_dict()), b)

    def test_from_dict_partial(self):
        self.assertEqual(DeliveryBudget.from_dict({}), DeliveryBudget())
        self.assertEqual(
            DeliveryBudget.from_dict({"max_size": "2048"}), DeliveryBudget(2048)
        )


class ProfileBudgetTest(unittest.TestCase):
    def test_every_profile_has_a_budget(self):
        for name in OutputTemplates.BUILTIN:
            self.assertIsInstance(OutputTemplates.budget(name), DeliveryBudget, name)

    def test_unknown_and_absent_profiles_are_unbudgeted(self):
        # Rather than raising — callers pass a profile through without branching.
        self.assertEqual(OutputTemplates.budget(None), DeliveryBudget())
        self.assertEqual(OutputTemplates.budget("NotAProfile"), DeliveryBudget())

    def test_gltf_carries_the_web_budget(self):
        # glTF 2.0 is the WebXR delivery profile; the tighter ceiling + POT
        # expectation is what makes a separate "WebXR" workflow unnecessary.
        budget = OutputTemplates.budget(WF.GLTF)
        self.assertEqual(budget.max_size, 2048)
        self.assertTrue(budget.force_pot)

    def test_authoring_profiles_are_unbudgeted(self):
        # STD and Spec/Gloss produce intermediates, not shipped assets — a
        # ceiling there would be a guess presented as a recommendation.
        for name in (WF.STD, WF.SPEC):
            self.assertIsNone(OutputTemplates.budget(name).max_size, name)

    def test_realtime_profiles_are_budgeted(self):
        for name in (WF.URP, WF.HDRP, WF.UE, WF.GODOT):
            self.assertIsNotNone(OutputTemplates.budget(name).max_size, name)

    def test_budget_does_not_alter_the_hard_tier(self):
        # The web budget must not have quietly turned glTF's containers lossy.
        self.assertEqual(OutputTemplates.resolve("Base_Color", WF.GLTF).ext, "png")
        self.assertEqual(OutputTemplates.resolve("Normal", WF.GLTF).ext, "png")


class WorkflowDescriptionTest(unittest.TestCase):
    """Descriptions are the preset tooltip every tool renders — they carry the
    platform list a user picks a profile by, so they are part of the contract."""

    def test_every_profile_names_its_targets(self):
        for name, preset in MapRegistry().get_workflow_presets().items():
            self.assertIn("Targets:", preset["description"], name)

    def test_gltf_description_names_webxr(self):
        presets = MapRegistry().get_workflow_presets()
        self.assertIn("WebXR", presets[WF.GLTF]["description"])

    def test_descriptions_carry_no_markup(self):
        # Qt auto-detects rich text in tooltips: an angle bracket would be parsed
        # as a tag and silently swallow the word inside it.
        for name, preset in MapRegistry().get_workflow_presets().items():
            self.assertNotIn("<", preset["description"], name)


class ResolveOutputSpecTest(unittest.TestCase):
    def test_height_is_16bit_across_profiles(self):
        for wf in (WF.HDRP, WF.URP, WF.UE, WF.STD, None):
            self.assertEqual(
                OutputTemplates.resolve("Height", wf).bit_depth, 16, f"profile={wf}"
            )

    def test_color_is_8bit(self):
        self.assertEqual(OutputTemplates.resolve("Base_Color", WF.HDRP).bit_depth, 8)

    def test_ue_prefers_tga(self):
        self.assertEqual(OutputTemplates.resolve("Base_Color", WF.UE).ext, "tga")
        self.assertEqual(OutputTemplates.resolve("Normal", WF.UE).ext, "tga")

    def test_unknown_profile_uses_default_template(self):
        self.assertEqual(
            OutputTemplates.resolve("Height", "NotAProfile"),
            OutputTemplates.DEFAULT.resolve("Height"),
        )

    def test_unknown_map_uses_template_default(self):
        spec = OutputTemplates.resolve("CompletelyUnknownMap", WF.HDRP)
        self.assertEqual(spec, OutputTemplates.get(WF.HDRP).default)

    def test_all_wf_profiles_have_a_template(self):
        for name in (WF.STD, WF.URP, WF.HDRP, WF.UE, WF.GLTF, WF.GODOT, WF.SPEC):
            self.assertIn(name, OutputTemplates.BUILTIN, f"missing template for {name}")

    def test_registry_and_templates_stay_in_step(self):
        # A profile added to the registry without a template silently falls back
        # to DEFAULT — wrong containers, and no budget at all.
        for name in MapRegistry().get_workflow_presets():
            self.assertIn(name, OutputTemplates.BUILTIN, f"missing template for {name}")


if __name__ == "__main__":
    unittest.main(exit=False)
