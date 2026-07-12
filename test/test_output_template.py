#!/usr/bin/python
# coding=utf-8
"""Unit tests for the per-map output-format template layer.

Run with:
    python -m pytest test_output_template.py -v
"""
import unittest

from pythontk import OutputSpec, OutputTemplate, OutputTemplates
from pythontk.core_utils.engines.textures.map_registry import WF


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


if __name__ == "__main__":
    unittest.main(exit=False)
