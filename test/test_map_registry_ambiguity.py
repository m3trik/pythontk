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


class NormalConventionSymmetryTest(unittest.TestCase):
    """Both tangent-space conventions must classify from the same spellings.

    Regression: ``DX`` was a registered alias and ``OGL`` was not, so
    ``rock_NRML_DX.png`` resolved to Normal_DirectX while its twin
    ``rock_NRML_OGL.png`` resolved to None — one half of a bake pair silently
    dropped as an unknown map. ``GL`` could not cover it either: the short-alias
    boundary rule rejects an uppercase predecessor (the ``Agilent_E4419B``
    guard), and the ``O`` in ``OGL`` is exactly that.
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

    #: Every ``<normal token><sep><convention tag>`` spelling seen in the wild.
    NORMAL_STEMS = ("Normal", "NormalMap", "Norm", "NRML", "NRM", "NML", "N")
    OPENGL_TAGS = ("GL", "OGL", "OpenGL")
    DIRECTX_TAGS = ("DX", "DirectX")

    def test_every_convention_spelling_resolves_on_both_sides(self):
        for stem in self.NORMAL_STEMS:
            for sep in ("_", ""):
                for tag in self.OPENGL_TAGS:
                    self.assert_map_type(f"rock_{stem}{sep}{tag}.png", "Normal_OpenGL")
                for tag in self.DIRECTX_TAGS:
                    self.assert_map_type(f"rock_{stem}{sep}{tag}.png", "Normal_DirectX")

    def test_convention_spellings_are_case_insensitive(self):
        for name in ("rock_nrml_ogl.png", "rock_NRML_OGL.png", "rock_Nrml_Ogl.png"):
            self.assert_map_type(name, "Normal_OpenGL")

    def test_unqualified_normal_tokens_resolve_to_the_generic_type(self):
        for stem in self.NORMAL_STEMS:
            self.assert_map_type(f"rock_{stem}.png", "Normal")


class UtilityBakeMapTest(unittest.TestCase):
    """Bakes that are not tangent-space normals must not classify as ``Normal``.

    Measured: ``mat_WorldNormal.png`` and ``mat_BentNormal.png`` both resolved to
    ``Normal`` (longest-first matching found the trailing ``Normal`` token), so an
    object-space or bent-normal bake was wired into the tangent-normal slot and
    rendered wrong with no warning. ``mat_VectorDisplacement.png`` resolved to
    ``Displacement``, whose declared mode is ``L`` — flattening an RGB vector
    displacement to grayscale destroys two of its three offset axes.
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

    def test_object_and_world_space_normals_are_their_own_types(self):
        for name in ("mat_ObjectNormal.png", "mat_Normal_Object.png", "mat_OSN.png"):
            self.assert_map_type(name, "Normal_Object")
        for name in ("mat_WorldNormal.png", "mat_Normal_World.png", "mat_WSN.png"):
            self.assert_map_type(name, "Normal_World")

    def test_bent_normal_is_its_own_type(self):
        for name in ("mat_BentNormal.png", "mat_Bent_Normal.png"):
            self.assert_map_type(name, "Bent_Normal")

    def test_vector_displacement_is_its_own_type(self):
        for name in ("mat_VectorDisplacement.png", "mat_VDM.png"):
            self.assert_map_type(name, "Vector_Displacement")

    def test_vector_displacement_keeps_its_three_channels(self):
        """The point of the separate type: ``L`` would discard Y and Z."""
        self.assertEqual(
            MapRegistry().get_map_modes()["Vector_Displacement"], "RGB"
        )

    def test_utility_normals_are_not_offered_as_the_shader_normal(self):
        """``select_normal_type`` wires tangent-space maps only."""
        for name in ("Normal_Object", "Normal_World", "Bent_Normal"):
            self.assertNotIn(name, MapRegistry.NORMAL_TYPES)
            self.assertIsNone(MapRegistry.select_normal_type({name}))

    def test_tangent_normals_still_win_when_both_are_present(self):
        self.assertEqual(
            MapRegistry.select_normal_type(
                {"Normal_Object", "Bent_Normal", "Normal_OpenGL"}
            ),
            "Normal_OpenGL",
        )

    def test_new_aliases_do_not_claim_ordinary_words(self):
        """Aliases over 3 chars match with no boundary check, so they must be
        spellings no material name would end in.

        Caught during review: a bare ``Bent`` alias on Bent_Normal claimed
        ``mat_absorbent.png`` and ``mat_unbent.png``.
        """
        for name in ("mat_absorbent.png", "mat_unbent.png", "mat_normalcy.png"):
            self.assert_map_type(name, None)

    def test_unregistered_utility_bakes_stay_unclassified(self):
        """Unknown is the correct answer — it routes to passthrough, not a slot.

        ``Curvature`` in particular is the registry's documented custom-type
        example (`examples/texture_factory_extensibility_example.py`, and the
        registration tests assert it is unresolvable until registered), so the
        built-in table must leave that name free.
        """
        for name in ("mat_Curvature.png", "mat_Cavity.png", "mat_Position.png"):
            self.assert_map_type(name, None)


class CounterpartSpellingTest(unittest.TestCase):
    """Converting a normal map's handedness must keep the file's naming style.

    Two callers (``MapFactory.convert_normal_map_format`` and
    ``MapCompositor._try_invert_normal``) used to pair the DirectX and OpenGL
    alias tuples **by index** — a contract requiring both lists to stay the same
    length and in lockstep order. They never were: ``DXN`` sat past the end of
    the OpenGL tuple, raising IndexError in one caller and falling back in the
    other. Generating the convention spellings broke it outright — ``NDX``
    paired to ``NRMGL`` and ``Norm_DX`` to ``NormalMap_OGL``.
    """

    def test_spelling_style_survives_the_conversion(self):
        for spelling, expected in (
            ("Normal_DirectX", "Normal_OpenGL"),
            ("NormalDX", "NormalGL"),
            ("Normal_DX", "Normal_GL"),
            ("Norm_DX", "Norm_GL"),
            ("NDX", "NGL"),
            ("DX", "GL"),
            ("Normal_Tangent_DX", "Normal_Tangent_GL"),
        ):
            self.assertEqual(
                MapRegistry.counterpart_normal_spelling(spelling, "Normal_OpenGL"),
                expected,
                spelling,
            )

    def test_the_swap_is_symmetric(self):
        for spelling, expected in (
            ("Normal_OpenGL", "Normal_DirectX"),
            ("NormalGL", "NormalDX"),
            ("Normal_OGL", "Normal_DX"),
            ("NGL", "NDX"),
            ("GL", "DX"),
        ):
            self.assertEqual(
                MapRegistry.counterpart_normal_spelling(spelling, "Normal_DirectX"),
                expected,
                spelling,
            )

    def test_a_spelling_with_no_tag_falls_back_to_the_canonical_name(self):
        """``DXN`` carries no trailing tag — and used to run off the tuple's end."""
        self.assertEqual(
            MapRegistry.counterpart_normal_spelling("DXN", "Normal_OpenGL"),
            "Normal_OpenGL",
        )
        for bad in ("", "Roughness", None):
            self.assertEqual(
                MapRegistry.counterpart_normal_spelling(bad, "Normal_DirectX"),
                "Normal_DirectX",
            )

    def test_a_non_normal_destination_is_refused(self):
        self.assertEqual(
            MapRegistry.counterpart_normal_spelling("NormalDX", "Roughness"), "Roughness"
        )

    def test_every_counterpart_resolves_to_the_destination_type(self):
        """The swapped spelling must be a name the classifier actually accepts."""
        registry = MapRegistry()
        pairs = (("Normal_DirectX", "Normal_OpenGL"), ("Normal_OpenGL", "Normal_DirectX"))
        for src, dst in pairs:
            for spelling in registry.get_map_types()[src]:
                counterpart = MapRegistry.counterpart_normal_spelling(spelling, dst)
                self.assertEqual(
                    registry.resolve_type_from_path(f"mat_{counterpart}.png"),
                    dst,
                    f"{spelling!r} -> {counterpart!r} does not classify as {dst}",
                )


class AliasOwnershipTest(unittest.TestCase):
    """No spelling may be claimed by two map types.

    The table is hand-curated and grew a generated cross-product for the normal
    conventions; a duplicate alias would make classification depend on dict order
    and length-sort stability rather than on the taxonomy. Held at zero today —
    this pins it there.
    """

    def test_no_alias_is_claimed_by_two_map_types(self):
        from collections import defaultdict

        owners = defaultdict(set)
        for name, m in MapRegistry()._maps.items():
            owners[name.lower()].add(name)
            for alias in m.aliases:
                owners[alias.lower()].add(name)

        collisions = {k: sorted(v) for k, v in owners.items() if len(v) > 1}
        self.assertEqual(collisions, {}, f"aliases claimed twice: {collisions}")

    def test_every_alias_resolves_to_its_own_map_type(self):
        """A separator-delimited alias must classify as the type that declares it."""
        registry = MapRegistry()
        for name, m in registry._maps.items():
            for alias in [name] + list(m.aliases):
                self.assertEqual(
                    registry.resolve_type_from_path(f"mat_{alias}.png"),
                    name,
                    f"alias {alias!r} of {name!r} resolved elsewhere",
                )


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
