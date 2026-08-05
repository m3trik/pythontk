#!/usr/bin/python
# coding=utf-8
"""
Tests for MapRegistry ambiguity resolution and alias handling.
"""
import unittest
from pythontk.core_utils.engines.textures.map_registry import MapRegistry


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

    def test_short_aliases_after_a_separator_are_case_insensitive(self):
        """A separator is boundary enough for a short alias, in either case.

        This used to demand a capital everywhere, but the base-name stripper's
        underscore branch never did — so ``MyMat_ao.png`` lost its suffix when
        grouping into a texture set and then classified as nothing, and the map
        was dropped as unknown. The case rule now applies only to the ATTACHED
        (CamelCase) boundary, where case is the *only* evidence a suffix exists
        (see ``test_short_aliases_attached_require_a_capital``).
        """
        for stem, expected in (
            ("BC", "Base_Color"),
            ("AO", "Ambient_Occlusion"),
            ("N", "Normal"),
            ("R", "Roughness"),
            ("M", "Metallic"),
            ("H", "Height"),
            ("B", "Bump"),
            ("S", "Specular"),
        ):
            self.assert_map_type(f"MyMat_{stem}.png", expected)
            self.assert_map_type(f"MyMat_{stem.lower()}.png", expected)

    def test_short_aliases_attached_require_a_capital(self):
        """Without a separator, only a CamelCase step marks a suffix."""
        self.assert_map_type("MyMatN.png", "Normal")
        self.assert_map_type("MyMatAO.png", "Ambient_Occlusion")
        # Ordinary words ending in an alias letter must stay unclassified.
        self.assert_map_type("wood_green.png", None)
        self.assert_map_type("chrome.png", None)

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


class NormalTypeSelectionTest(unittest.TestCase):
    """The shared precedence both DCC packages wire their normal map from."""

    def test_explicit_tags_outrank_the_ambiguous_generic_map(self):
        self.assertEqual(
            MapRegistry.select_normal_type({"Normal", "Normal_OpenGL"}), "Normal_OpenGL"
        )
        self.assertEqual(
            MapRegistry.select_normal_type({"Normal", "Normal_DirectX"}),
            "Normal_DirectX",
        )
        self.assertEqual(
            MapRegistry.select_normal_type(
                {"Normal", "Normal_OpenGL", "Normal_DirectX"}
            ),
            "Normal_OpenGL",
        )

    def test_generic_map_is_used_when_it_is_all_there_is(self):
        self.assertEqual(MapRegistry.select_normal_type({"Normal"}), "Normal")

    def test_none_when_no_normal_is_present(self):
        for available in ({}, set(), {"Base_Color": "x", "Roughness": "y"}):
            self.assertIsNone(MapRegistry.select_normal_type(available))

    def test_accepts_a_dict_keyed_by_map_type(self):
        """Both consumers pass their `by_type` mapping straight in."""
        self.assertEqual(
            MapRegistry.select_normal_type({"Normal_DirectX": "/x/a.png"}),
            "Normal_DirectX",
        )

    def test_every_declared_type_is_a_registered_map(self):
        registered = set(MapRegistry().get_map_types())
        for name in MapRegistry.NORMAL_TYPES:
            self.assertIn(name, registered, name)


class TileTokenTest(unittest.TestCase):
    """UDIM / UV-tile tokens must not hide the map-type suffix.

    ``os.path.splitext`` only removes the extension, so ``rock_Normal.1001.png``
    used to reach the alias matcher as ``rock_Normal.1001`` — which ends in no
    alias at all. Every tile of every UDIM set classified as None, grouped into
    its own single-file "set", and was reported as an unrecognized map.
    """

    def setUp(self):
        self.registry = MapRegistry()

    def assert_map_type(self, filename, expected_type):
        resolved = self.registry.resolve_type_from_path(filename)
        self.assertEqual(
            resolved,
            expected_type,
            f"Failed for '{filename}': Expected '{expected_type}', got '{resolved}'",
        )

    def test_udim_tiles_classify_like_their_untiled_twin(self):
        for name, expected in (
            ("rock_Normal.1001.png", "Normal"),
            ("rock_Normal.1042.png", "Normal"),
            ("rock_BaseColor.1001.png", "Base_Color"),
            ("rock_Normal_DirectX.1001.png", "Normal_DirectX"),
            ("rock_ao.1001.png", "Ambient_Occlusion"),
            ("rock_Normal.<UDIM>.png", "Normal"),
            ("rock_Normal_<UDIM>.png", "Normal"),
            ("rock_Normal.<UVTILE>.png", "Normal"),
            ("rock_Normal.u1_v1.png", "Normal"),
            ("rock_Normal_u2_v10.png", "Normal"),
        ):
            self.assert_map_type(name, expected)

    def test_split_tile_token_round_trips(self):
        for stem, base, token in (
            ("rock_Normal.1001", "rock_Normal", ".1001"),
            ("rock_Normal_<UDIM>", "rock_Normal", "_<UDIM>"),
            ("rock_Normal.u1_v1", "rock_Normal", ".u1_v1"),
            ("rock_Normal", "rock_Normal", ""),
        ):
            self.assertEqual(MapRegistry.split_tile_token(stem), (base, token))
            self.assertEqual(base + token, stem)

    def test_resolution_tags_are_not_mistaken_for_tiles(self):
        """``_1024`` is a resolution tag, not a tile — only dotted digits count.

        A bare underscore + 4 digits collides head-on with the ``_1024`` /
        ``_2048`` resolution convention, and 1024 sits inside the UDIM range, so
        the underscore form is deliberately not recognized for bare digits.
        """
        for stem in ("wall_Normal_1024", "wall_Normal_2048", "wall_Normal_1001"):
            self.assertEqual(MapRegistry.split_tile_token(stem), (stem, ""))
        self.assert_map_type("wall_Normal_1024.png", None)

    def test_out_of_range_dotted_numbers_are_not_tiles(self):
        for stem in ("shot_Normal.0999", "shot_Normal.2001", "shot_Normal.12"):
            self.assertEqual(MapRegistry.split_tile_token(stem), (stem, ""))


if __name__ == "__main__":
    unittest.main()
