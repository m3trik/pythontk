#!/usr/bin/python
# coding=utf-8
"""
Tests for MapRegistry ambiguity resolution and alias handling.
"""
import unittest
from pythontk.img_utils.map_registry import MapRegistry


class TestMapRegistryAmbiguity(unittest.TestCase):
    def setUp(self):
        self.registry = MapRegistry()

    def assert_map_type(self, filename, expected_type):
        """Helper to assert that a filename resolves to the expected map type."""
        resolved = self.registry.resolve_type_from_path(filename)
        self.assertEqual(
            resolved,
            expected_type,
            f"Failed for '{filename}': Expected '{expected_type}', got '{resolved}'",
        )

    def test_short_aliases_case_sensitivity(self):
        """Test that short aliases (<=3 chars) are case-sensitive for the first letter."""
        # Base Color (BC)
        self.assert_map_type("MyMat_BC.png", "Base_Color")
        self.assert_map_type("MyMat_bc.png", None)  # Should fail (lowercase)

        # Ambient Occlusion (AO)
        self.assert_map_type("MyMat_AO.png", "Ambient_Occlusion")
        self.assert_map_type("MyMat_ao.png", None)

        # Normal (N)
        self.assert_map_type("MyMat_N.png", "Normal")
        self.assert_map_type("MyMat_n.png", None)

        # Roughness (R)
        self.assert_map_type("MyMat_R.png", "Roughness")
        self.assert_map_type("MyMat_r.png", None)

        # Metallic (M)
        self.assert_map_type("MyMat_M.png", "Metallic")
        self.assert_map_type("MyMat_m.png", None)

        # Height (H)
        self.assert_map_type("MyMat_H.png", "Height")
        self.assert_map_type("MyMat_h.png", None)

        # Bump (B)
        self.assert_map_type("MyMat_B.png", "Bump")
        self.assert_map_type("MyMat_b.png", None)

        # Specular (S)
        self.assert_map_type("MyMat_S.png", "Specular")
        self.assert_map_type("MyMat_s.png", None)

    def test_long_aliases_case_insensitivity(self):
        """Test that long aliases (>3 chars) are case-insensitive."""
        self.assert_map_type("MyMat_BaseColor.png", "Base_Color")
        self.assert_map_type("MyMat_basecolor.png", "Base_Color")
        self.assert_map_type("MyMat_BASECOLOR.png", "Base_Color")

        self.assert_map_type("MyMat_Smoothness.png", "Smoothness")
        self.assert_map_type("MyMat_smoothness.png", "Smoothness")

    def test_ambiguity_resolution(self):
        """Test that longer matches take precedence over shorter ones."""
        # 'Smoothness' ends with 's', but 'S' is alias for Specular.
        # Should resolve to Smoothness, not Specular.
        self.assert_map_type("MyMat_Smoothness.png", "Smoothness")
        self.assert_map_type(
            "MyMat_smoothness.png", "Smoothness"
        )  # Case insensitive match for Smoothness

        # 'Normal' starts with 'N', 'N' is alias for Normal.
        self.assert_map_type("MyMat_Normal.png", "Normal")

        # 'Metallic' starts with 'M', 'M' is alias for Metallic.
        self.assert_map_type("MyMat_Metallic.png", "Metallic")

    def test_underscore_handling(self):
        """Test handling of underscores in filenames."""
        # With underscore
        self.assert_map_type("MyMat_BC.png", "Base_Color")

        # Without underscore (if supported by logic, currently logic checks endswith)
        # The current logic is: name_lower.endswith(alias.lower())
        # So "MyMatBC.png" -> name="MyMatBC" -> ends with "BC" -> True
        self.assert_map_type("MyMatBC.png", "Base_Color")
        self.assert_map_type("MyMatS.png", "Specular")

    def test_false_positives(self):
        """Test against potential false positives."""
        # "Shadow" ends with "ow", not a map type.
        # "Shadows" ends with "s". If "s" (lowercase) matched Specular, this would fail.
        self.assert_map_type("MyMat_Shadows.png", None)

        # "Bump" ends with "p". "BP" is alias for Bump.
        self.assert_map_type("MyMat_Bump.png", "Bump")

        # "Displacement"
        self.assert_map_type("MyMat_Displacement.png", "Displacement")


if __name__ == "__main__":
    unittest.main()
