#!/usr/bin/python
# coding=utf-8
"""Short map-type aliases must not match glued onto a model/part number.

Regression: a real production scene (VDATS) carries reference textures named
after instrument model numbers -- ``Agilent_E4419B.png``, ``Agilent_PSG.png``,
``Agilent_8757D.png``. ``resolve_type_from_path`` matched the trailing ``B`` /
``G`` / ``D`` as the short aliases for Bump / Glossiness / Diffuse, so a plain
color map was silently wired into a normal or glossiness socket. The sibling
path (``MapFactory.resolve_map_type(key=False)``) already required a boundary
and documented why; the two disagreed.

Boundary rule: a short alias matches only after a separator, after a lowercase
letter (CamelCase, ``rockN``), or standing alone -- never glued to a digit or an
uppercase letter.
"""
import unittest

from pythontk import MapRegistry

from conftest import BaseTestCase


class ShortAliasBoundaryTest(BaseTestCase):
    def setUp(self):
        self.reg = MapRegistry()

    def test_model_numbers_do_not_resolve_as_map_types(self):
        """The exact filenames from the production scene must classify as None."""
        for name in (
            "Agilent_E4419B.png",   # trailing B -> was Bump
            "Agilent_PSG.png",      # trailing G -> was Glossiness
            "Agilent_8757D.png",    # trailing D -> was Diffuse
            "Agilent_PNA.png",
            "keysight_5315A.png",
            "Rohde&Schwarz_FSU.png",
        ):
            self.assertIsNone(
                self.reg.resolve_type_from_path(name),
                f"{name} is a model number, not a map-type suffix",
            )

    def test_separator_delimited_short_aliases_still_resolve(self):
        """The normal convention must be unaffected."""
        for name, expected in (
            ("im_H.png", "Height"),
            ("im_N.png", "Normal"),
            ("rock_AO.png", "Ambient_Occlusion"),
            ("test_ORM.png", "ORM"),
        ):
            self.assertEqual(
                self.reg.resolve_type_from_path(name), expected, name
            )

    def test_lowercase_separator_delimited_short_aliases_resolve(self):
        """A separator is boundary enough — case must not also be required.

        ``_nrm`` / ``_ao`` / ``_rgh`` (and the classic ``_d`` / ``_n`` / ``_s``
        game-texture convention) are everyday lowercase spellings. The strip
        pattern's underscore branch has always been case-insensitive, so these
        names already lost their suffix for base-name grouping while the
        resolver called them unclassifiable — the set grouped, then every map
        but the long-alias ones was dropped as unknown.
        """
        for name, expected in (
            ("rock_nrm.png", "Normal"),
            ("rock_n.png", "Normal"),
            ("rock_ao.png", "Ambient_Occlusion"),
            ("rock_rgh.png", "Roughness"),
            ("rock_bc.png", "Base_Color"),
            ("rock_gl.png", "Normal_OpenGL"),
            ("rock_dx.png", "Normal_DirectX"),
            ("rock_orm.png", "ORM"),
            ("rock-msao.png", "MSAO"),
        ):
            self.assertEqual(self.reg.resolve_type_from_path(name), expected, name)

    def test_attached_lowercase_short_alias_still_does_not_resolve(self):
        """Loosening the separator case must not loosen the ATTACHED case.

        Without a separator, a trailing lowercase letter is just the end of an
        ordinary word: ``green`` ends in ``n``, ``chrome`` in ``e``. Only a
        capital marks a deliberate CamelCase suffix.
        """
        for name in (
            "wood_green.png",  # trailing 'n' -> would be Normal
            "chrome.png",  # trailing 'e' -> would be Emissive
            "rockn.png",
            "backdrop.png",  # trailing 'p' is not an alias, but guards the sweep
        ):
            self.assertIsNone(self.reg.resolve_type_from_path(name), name)

    def test_camelcase_glued_alias_still_resolves(self):
        """A lowercase->uppercase step is a real boundary (``rockN`` -> Normal)."""
        self.assertEqual(self.reg.resolve_type_from_path("rockN.png"), "Normal")

    def test_alias_alone_still_resolves(self):
        self.assertEqual(self.reg.resolve_type_from_path("N.png"), "Normal")

    def test_base_name_and_classification_agree_about_suffixes(self):
        """The invariant that broke: strip and resolve must call the same thing a suffix.

        ``get_suffix_strip_pattern`` and ``resolve_type_from_path`` are separate
        implementations of "does this filename end in a map-type suffix". When only
        the resolver was tightened, ``Agilent_E4419B`` classified as None but still
        had its ``B`` stripped for base-name grouping -- so two distinct textures
        could collapse into one set.
        """
        import pythontk as ptk

        for name in (
            "Agilent_E4419B.png",
            "Agilent_PSG.png",
            "Agilent_8757D.png",
            "Agilent_PNA.png",
            "rockN.png",
            "rock_AO.png",
            "VDATS_cabinet_MSAO.png",
            # Lowercase separator-delimited: the strip pattern always took
            # these, so the resolver has to agree.
            "rock_nrm.png",
            "rock_ao.png",
            "rock_bc.png",
            "rock_basecolor.png",
            # Attached lowercase: neither side may treat these as suffixes.
            "wood_green.png",
            "chrome.png",
        ):
            stem = name.rsplit(".", 1)[0]
            resolved = self.reg.resolve_type_from_path(name)
            base = ptk.MapFactory.get_base_texture_name(name)
            stripped = base != stem
            self.assertEqual(
                stripped,
                resolved is not None,
                f"{name}: base_name {'stripped' if stripped else 'kept'} a suffix "
                f"but resolve_type_from_path returned {resolved!r}",
            )

    def test_distinct_model_numbers_stay_distinct_sets(self):
        """Two instruments must not collapse into one texture set."""
        import pythontk as ptk

        self.assertNotEqual(
            ptk.MapFactory.get_base_texture_name("Agilent_E4419A.png"),
            ptk.MapFactory.get_base_texture_name("Agilent_E4419B.png"),
        )



class LogicalChannelTypeTest(BaseTestCase):
    """Logical shader channel -> canonical map type (the manifest slot fallback)."""

    def test_known_channels_map_to_registered_types(self):
        reg = MapRegistry()
        registered = set(reg.get_map_types())
        for channel, expected in MapRegistry.LOGICAL_CHANNEL_TYPES.items():
            self.assertEqual(
                MapRegistry.resolve_type_from_channel(channel), expected, channel
            )
            self.assertIn(
                expected, registered, f"{channel} -> {expected} is not a real map type"
            )

    def test_case_insensitive(self):
        self.assertEqual(
            MapRegistry.resolve_type_from_channel("BASECOLOR"), "Base_Color"
        )

    def test_unknown_channel_is_none(self):
        for bad in ("", None, "notAChannel"):
            self.assertIsNone(MapRegistry.resolve_type_from_channel(bad))

if __name__ == "__main__":
    unittest.main()
