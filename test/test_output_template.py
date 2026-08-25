#!/usr/bin/python
# coding=utf-8
"""Unit tests for the per-map output-format template layer.

Run with:
    python -m pytest test_output_template.py -v
"""

import html as _html
import unittest
import xml.etree.ElementTree as ET

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
        self.assertEqual(
            OutputSpec.from_dict({"ext": "tiff"}), OutputSpec("tiff", 8, None)
        )


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
        self.assertEqual(
            OutputTemplate.from_dict(t.to_dict()).resolve("Normal"), t.resolve("Normal")
        )

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


class LossySafetyGateTest(unittest.TestCase):
    """Which map types may be written to a lossy container.

    Measured on a 4K source at WebP q95: a base color deviates by at most
    9/255, a normal map by 122/255. The gate encodes that difference.
    """

    def setUp(self):
        self.reg = MapRegistry()

    def test_unpacked_srgb_maps_are_safe(self):
        for name in ("Base_Color", "Diffuse", "Emissive"):
            self.assertTrue(self.reg.is_lossy_safe(name), name)

    def test_linear_maps_are_refused(self):
        for name in ("Normal", "Normal_OpenGL", "Roughness", "Metallic", "Height"):
            self.assertFalse(self.reg.is_lossy_safe(name), name)

    def test_packed_maps_are_refused_even_when_srgb(self):
        # Albedo_Transparency is sRGB but packs opacity into alpha — the
        # channels are unrelated, which is the assumption a codec relies on.
        for name in (
            "ORM",
            "MSAO",
            "MRAO",
            "Metallic_Smoothness",
            "Albedo_Transparency",
        ):
            self.assertFalse(self.reg.is_lossy_safe(name), name)

    def test_unknown_map_defaults_to_refused(self):
        self.assertFalse(self.reg.is_lossy_safe("Not_A_Real_Map"))
        self.assertFalse(self.reg.is_lossy_safe(None))

    def test_every_registered_map_agrees_with_its_fields(self):
        # The predicate must stay derived, not drift into a hand-kept list.
        for name, m in MapRegistry()._maps.items():
            self.assertEqual(
                self.reg.is_lossy_safe(name),
                m.color_space == "sRGB" and not m.is_packed,
                name,
            )


class SelectionChoicesTest(unittest.TestCase):
    """The SSoT six panels populate their combos from."""

    def test_profile_choices_match_the_registry(self):
        names = [n for n, _ in OutputTemplates.profile_choices()]
        self.assertEqual(names, list(MapRegistry().get_workflow_presets()))

    def test_profile_choices_carry_descriptions(self):
        for name, description in OutputTemplates.profile_choices():
            self.assertTrue(description, f"{name} has no description tooltip")

    def test_format_choices_put_the_sentinel_last_by_default(self):
        choices = OutputTemplates.format_choices()
        self.assertEqual(choices[-1], (OutputTemplates.PROFILE_DEFAULT_LABEL, ""))
        self.assertEqual(choices[0][1], "png")

    def test_sentinel_first_is_available_for_existing_panels(self):
        # Position is a persistence contract — panels store the index.
        choices = OutputTemplates.format_choices(
            sentinel=OutputTemplates.ORIGINAL_LABEL, sentinel_first=True
        )
        self.assertEqual(choices[0], (OutputTemplates.ORIGINAL_LABEL, ""))

    def test_format_choices_can_omit_the_sentinel(self):
        choices = OutputTemplates.format_choices(sentinel=None)
        self.assertTrue(all(value for _, value in choices))

    def test_format_choices_accept_an_injected_container_list(self):
        self.assertEqual(
            OutputTemplates.format_choices(sentinel=None, writable=("png", "webp")),
            [("PNG", "png"), ("WEBP", "webp")],
        )

    def test_resolve_selection(self):
        self.assertEqual(
            OutputTemplates.resolve_selection(WF.GLTF, ""), (WF.GLTF, None)
        )
        self.assertEqual(
            OutputTemplates.resolve_selection(WF.GLTF, ".WEBP"), (WF.GLTF, "webp")
        )
        self.assertEqual(OutputTemplates.resolve_selection("", None), (None, None))


class ProfileOutlineTest(unittest.TestCase):
    """The rendered-tooltip outline every preset combo shows for a profile."""

    KEYS = {"title", "body", "sections", "notes"}

    def _writes(self, outline):
        return dict(outline["sections"])["Writes"]

    def test_outline_keys_are_the_documented_set(self):
        # The keys ARE the contract -- panels splat them into a formatter as
        # **kwargs, so an extra key is a TypeError at the call site.
        for name, outline in OutputTemplates.profile_outlines():
            self.assertEqual(set(outline), self.KEYS, name)

    def test_outlines_match_the_registry_order(self):
        names = [n for n, _ in OutputTemplates.profile_outlines()]
        self.assertEqual(names, [n for n, _ in OutputTemplates.profile_choices()])

    def test_body_is_the_profile_description(self):
        for name, description in OutputTemplates.profile_choices():
            self.assertEqual(OutputTemplates.profile_outline(name)["body"], description)

    def test_every_profile_writes_its_normal_convention(self):
        """The handedness is the one thing a user must match; a bare engine
        name never said which way the green channel points."""
        presets = MapRegistry().get_workflow_presets()
        for name, preset in presets.items():
            normal_type = preset["normal_type"]
            writes = self._writes(OutputTemplates.profile_outline(name))
            self.assertTrue(
                any(f"_Normal_{normal_type}." in item for item in writes),
                f"{name} does not name its {normal_type} normal output",
            )

    def test_packed_maps_are_listed_with_their_channel_layout(self):
        writes = self._writes(OutputTemplates.profile_outline(WF.UE))
        orm = next(item for item in writes if "_ORM." in item)
        for channel, carried in (
            ("R", "Ambient Occlusion"),
            ("G", "Roughness"),
            ("B", "Metallic"),
        ):
            self.assertIn(f"{channel}: {carried}", orm)

    def test_optional_channels_are_marked(self):
        """MSAO's Detail channel is filler -- coverage must not demand it, and
        neither may the tooltip."""
        writes = self._writes(OutputTemplates.profile_outline(WF.HDRP))
        msao = next(item for item in writes if "_MSAO." in item)
        self.assertIn("Detail Mask <i>(optional)</i>", msao)

    def test_declared_maps_come_from_the_same_source_as_the_preset_flags(self):
        """Outline and resolved config cannot disagree about what a profile
        asked for: both read MapType.workflows."""
        registry = MapRegistry()
        for name in registry.get_workflow_presets():
            writes = self._writes(OutputTemplates.profile_outline(name))
            for map_name in registry.get_map_types():
                entry = registry.get(map_name)
                if name in entry.workflows:
                    self.assertTrue(
                        any(f"_{map_name}." in item for item in writes),
                        f"{name} declares {map_name} but does not list it",
                    )

    def test_delivery_section_reports_containers_and_budget(self):
        sections = dict(OutputTemplates.profile_outline(WF.UE)["sections"])
        delivery = " | ".join(sections["Delivery"])
        self.assertIn("TGA", delivery)  # UE's default container
        self.assertIn("16-bit", delivery)  # the height-like override
        self.assertIn("4096", delivery)  # its realtime budget

    def test_delivery_off_omits_the_section_and_the_container(self):
        """A panel that does not pass the profile as ``output_profile`` leaves
        each map in its authored container -- naming one would be a lie."""
        outline = OutputTemplates.profile_outline(WF.UE, delivery=False)
        self.assertNotIn("Delivery", dict(outline["sections"]))
        for item in self._writes(outline):
            self.assertIn(_html.escape(OutputTemplates.EXT_TOKEN), item)
            self.assertNotIn(".tga", item)

    def test_base_name_is_substituted_and_escaped(self):
        writes = self._writes(
            OutputTemplates.profile_outline(WF.UE, base_name="rock<&>")
        )
        self.assertTrue(
            all(item.startswith("<b>rock&lt;&amp;&gt;_") for item in writes)
        )

    def test_default_base_name_token_is_escaped(self):
        outline = OutputTemplates.profile_outline(WF.UE)
        for item in self._writes(outline):
            self.assertNotIn("<name>", item)
            self.assertIn("&lt;name&gt;", item)

    def test_description_prose_is_escaped_not_injected(self):
        """Registry descriptions are plain text by contract; embedding them in
        markup must not let a stray bracket eat the word after it."""
        registry = MapRegistry()
        original = registry._workflow_settings[WF.STD]["description"]
        registry._workflow_settings[WF.STD]["description"] = (
            "a <b>bold</b> & wide claim"
        )
        try:
            self.assertEqual(
                OutputTemplates.profile_outline(WF.STD)["body"],
                "a &lt;b&gt;bold&lt;/b&gt; &amp; wide claim",
            )
        finally:
            registry._workflow_settings[WF.STD]["description"] = original

    def test_every_fragment_is_balanced_markup(self):
        """Qt eats the word after a bare ``<``. The outline embeds registry prose
        and a caller's base name in markup, so an unescaped one would silently
        swallow content -- and check_tooltips.py, being static, never sees a
        tooltip assembled at runtime."""
        hostile = """rock<&>'" 1001"""
        for delivery in (True, False):
            for name, outline in OutputTemplates.profile_outlines(
                delivery=delivery, base_name=hostile
            ):
                fragments = [outline["body"], *outline["notes"]]
                for heading, items in outline["sections"]:
                    fragments += [heading, *items]
                for fragment in fragments:
                    with self.subTest(profile=name, delivery=delivery):
                        ET.fromstring(f"<root>{fragment}</root>")

    def test_specgloss_conversion_is_called_out(self):
        notes = " ".join(OutputTemplates.profile_outline(WF.SPEC)["notes"])
        self.assertIn("Metallic / Roughness", notes)

    def test_unknown_profile_claims_nothing(self):
        """A profile that does not exist writes nothing knowable -- describing
        it from the default template would read as a real target's contract."""
        outline = OutputTemplates.profile_outline("not-a-profile")
        self.assertEqual(set(outline), self.KEYS)
        self.assertEqual(outline["body"], "")
        self.assertEqual(outline["sections"], [])
        self.assertEqual(outline["notes"], [])

    def test_every_declared_normal_type_has_its_handedness_spelled_out(self):
        """The fallback wording keeps a new handedness from crashing, which is
        also how it would go unnoticed -- so the coverage is pinned instead."""
        for name, preset in MapRegistry().get_workflow_presets().items():
            self.assertIn(
                preset["normal_type"], OutputTemplates.NORMAL_CONVENTIONS, name
            )

    def _declared(self, profile, predicate):
        registry = MapRegistry()
        return any(
            predicate(registry.get(m))
            for m in registry.get_map_types()
            if profile in registry.get(m).workflows
        )

    def test_transparency_condition_is_named_only_where_a_pack_carries_opacity(self):
        """The clause is derived from the pack's own channels, not matched on
        the name Albedo_Transparency -- Spec/Gloss packs Metallic_Smoothness and
        must not be told about transparency it never handles."""
        for name in MapRegistry().get_workflow_presets():
            carries_alpha = self._declared(
                name, lambda e: e.is_packed and "Opacity" in e.carried_types()
            )
            notes = " ".join(OutputTemplates.profile_outline(name)["notes"])
            self.assertEqual(carries_alpha, "real transparency" in notes, name)

    def test_the_missing_maps_rule_is_not_claimed_for_every_pack(self):
        """Only the ORM / MRAO / MSAO handlers consult `allow_incomplete_pack`,
        so a blanket citation would be wrong for Metallic_Smoothness and
        Albedo_Transparency -- and that control documents itself in the panel."""
        for name in MapRegistry().get_workflow_presets():
            notes = " ".join(OutputTemplates.profile_outline(name)["notes"])
            self.assertNotIn("Missing Maps", notes, name)

    def test_a_packed_map_is_advertised_as_conditional(self):
        """BaseColorHandler packs Albedo_Transparency only when the set carries
        real transparency, so an opaque set gets a plain Base Color. A tooltip
        that listed the packed name flat promised a file the run never writes."""
        registry = MapRegistry()
        for name, preset in registry.get_workflow_presets().items():
            packs = any(
                name in registry.get(m).workflows and registry.get(m).is_packed
                for m in registry.get_map_types()
            )
            notes = " ".join(OutputTemplates.profile_outline(name)["notes"])
            self.assertEqual(packs, "only when its sources" in notes, name)


class NoLossyDefaultsTest(unittest.TestCase):
    def test_builtin_catalogue_never_ships_lossy(self):
        """Lossy is opt-in. A template that defaulted to it would degrade
        every map of that type without the caller ever asking."""
        for name, template in OutputTemplates.BUILTIN.items():
            self.assertIsNone(template.default.quality, name)
            for map_type, spec in template.overrides.items():
                self.assertIsNone(spec.quality, f"{name}/{map_type}")

    def test_quality_round_trips(self):
        spec = OutputSpec("webp", 8, None, 90)
        self.assertEqual(OutputSpec.from_dict(spec.to_dict()), spec)
        self.assertEqual(OutputSpec.from_dict({"ext": "webp"}).quality, None)


if __name__ == "__main__":
    unittest.main(exit=False)
