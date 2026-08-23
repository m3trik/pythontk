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

Boundary rule: an alias matches only after a separator, after a lowercase
letter (CamelCase, ``rockN``), or standing alone -- never glued to a digit or an
uppercase letter.
"""
import unittest

from pythontk import ImgUtils, MapRegistry

from conftest import BaseTestCase


class _RegistryTestCase(BaseTestCase):
    """A fresh MapRegistry per test - four suites here need one, and nothing else."""

    def setUp(self):
        self.reg = MapRegistry()


class ShortAliasBoundaryTest(_RegistryTestCase):
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



class SeparatorParityTest(_RegistryTestCase):
    """All four separators must mean the same thing to strip and to classify.

    Regression (backlog 2026-08-05): ``_alias_boundary`` (then named
    ``_short_alias_boundary``) accepted
    ``_ - . `` and space, but ``get_suffix_strip_pattern``'s delimited branch was
    ``_``-only, so ``rock-ao.png`` classified as Ambient_Occlusion yet based to
    ``rock-ao``; ``rock.ao.png`` to ``rock.ao``. The long-alias attached branch
    had no boundary requirement at all, so ``rock-basecolor.png`` based to
    ``rock-``. Three spellings of one texture set became three sets.
    """

    SEPARATORS = ("_", "-", ".", " ")

    def test_every_separator_bases_to_the_same_stem(self):
        import pythontk as ptk

        for sep in self.SEPARATORS:
            for alias, expected in (("ao", "Ambient_Occlusion"), ("basecolor", "Base_Color")):
                name = f"rock{sep}{alias}.png"
                self.assertEqual(
                    ptk.MapFactory.get_base_texture_name(name),
                    "rock",
                    f"{name}: separator {sep!r} left junk in the base name",
                )
                self.assertEqual(self.reg.resolve_type_from_path(name), expected, name)

    def test_a_set_spelled_with_mixed_separators_groups_as_one(self):
        import pythontk as ptk

        bases = {
            ptk.MapFactory.get_base_texture_name(n)
            for n in ("rock_BaseColor.png", "rock-ao.png", "rock.Normal.png", "rock Roughness.png")
        }
        self.assertEqual(bases, {"rock"})

    def test_separators_do_not_loosen_the_model_number_guard(self):
        """Widening the delimiter set must not re-open the ``Agilent`` hole."""
        import pythontk as ptk

        for name in ("Agilent_E4419B.png", "Agilent-E4419B.png", "keysight_5315A.png"):
            self.assertIsNone(self.reg.resolve_type_from_path(name), name)
            stem = name.rsplit(".", 1)[0]
            self.assertEqual(ptk.MapFactory.get_base_texture_name(name), stem, name)


class CompoundSuffixTest(_RegistryTestCase):
    """A ``<type>_<convention>`` suffix must come off whole.

    Measured: ``rock_NRML_DX.png`` classified as Normal_DirectX but based to
    ``rock_NRML`` — only the registered ``_DX`` token was stripped, and the
    unregistered ``NRML`` stayed glued on. Its sibling ``rock_DIFF.png`` based to
    ``rock``, so the two halves of one texture set landed in different sets.

    The fix is enumerated compound aliases, NOT a repeating stripper: stripping
    suffixes in a loop eats material names, turning ``Gold_Metal_Diffuse`` into
    ``Gold`` (``Metal`` is a registered alias of Metallic).
    """

    def test_compound_normal_suffixes_strip_whole(self):
        import pythontk as ptk

        for name in (
            "rock_NRML_OGL.png",
            "rock_NRML_DX.png",
            "rock_Normal_OGL.png",
            "rock_Normal_DX.png",
            "rock_NormalOGL.png",
            "rock_NRM_OGL.png",
            "rock_N_GL.png",
        ):
            self.assertEqual(
                ptk.MapFactory.get_base_texture_name(name),
                "rock",
                f"{name}: compound suffix left a type token in the base name",
            )

    def test_a_normal_pair_groups_with_the_rest_of_its_set(self):
        import pythontk as ptk

        bases = {
            ptk.MapFactory.get_base_texture_name(n)
            for n in ("rock_DIFF.png", "rock_MTL.png", "rock_RUFF.png", "rock_NRML_OGL.png")
        }
        self.assertEqual(bases, {"rock"})

    def test_compound_suffixes_strip_whole_under_every_delimiter(self):
        """The joiner INSIDE a compound is delimited like any other suffix.

        Caught during review of the fix above: the composed aliases covered ``_``
        and glued only, so ``rock-nrml-ogl.png`` classified as Normal_OpenGL
        (the trailing tag resolves after any delimiter) yet based to
        ``rock-nrml`` — the same split, surviving under a different delimiter.
        """
        import pythontk as ptk

        for sep in ("_", "-", ".", " ", ""):
            for stem, tag in (("NRML", "OGL"), ("Normal", "DX"), ("N", "GL")):
                name = f"rock_{stem}{sep}{tag}.png"
                self.assertEqual(
                    ptk.MapFactory.get_base_texture_name(name),
                    "rock",
                    f"{name}: joiner {sep!r} left a type token in the base name",
                )

    def test_a_trailing_delimiter_collapses_whatever_it_is(self):
        """``rock_`` collapsed to ``rock`` while ``rock-`` stayed ``rock-``."""
        import pythontk as ptk

        for name in ("rock_.png", "rock-.png", "rock .png", "rock..png"):
            self.assertEqual(ptk.MapFactory.get_base_texture_name(name), "rock", name)

    def test_material_names_ending_in_a_map_alias_survive(self):
        """The guard against fixing this by looping the stripper."""
        import pythontk as ptk

        self.assertEqual(
            ptk.MapFactory.get_base_texture_name("Gold_Metal_Diffuse.png"), "Gold_Metal"
        )
        self.assertEqual(
            ptk.MapFactory.get_base_texture_name("Brick_Normal_Color.png"), "Brick_Normal"
        )


class BaseNameTwinTest(BaseTestCase):
    """``ImgUtils`` and ``MapFactory`` base names are documented as one answer.

    Both docstrings claim the registry pattern keeps them from drifting, but only
    the MapFactory side stripped the UDIM/UV-tile token first — so a tiled
    filename produced two different base names depending on which entry point the
    caller reached (``_map_factory.py`` uses the ImgUtils one when naming packed
    outputs). They are now one implementation; this pins the contract.
    """

    def test_twins_agree_including_on_tiled_names(self):
        import pythontk as ptk

        for name in (
            "rock_Normal.1001.png",
            "rock_BaseColor.<UDIM>.png",
            "rock_Normal.u1_v1.png",
            "rock_ao.png",
            "Agilent_E4419B.png",
            "Mat_brick_.png",
        ):
            self.assertEqual(
                ptk.ImgUtils.get_base_texture_name(name),
                ptk.MapFactory.get_base_texture_name(name),
                f"{name}: base-name twins disagree",
            )

    def test_tiles_of_one_set_share_a_base_name(self):
        import pythontk as ptk

        bases = {
            ptk.ImgUtils.get_base_texture_name(n)
            for n in ("rock_Normal.1001.png", "rock_Normal.1002.png", "rock_BaseColor.1001.png")
        }
        self.assertEqual(bases, {"rock"})


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


class LongAliasBoundaryTest(_RegistryTestCase):
    """The boundary rule is about EVIDENCE, not about how long the alias is.

    Until 2026-08-20 ``_match_alias`` demanded a word boundary only for aliases
    of three characters or fewer; anything longer matched on a bare
    ``endswith``, and ``get_suffix_strip_pattern`` mirrored the same threshold.
    524 of the 570 registered aliases are longer than three characters, so
    ordinary asset names classified as map types -- and because strip and
    resolve agreed about it, one material SPLIT INTO TWO texture sets.
    """

    def test_ordinary_words_ending_in_a_long_alias_do_not_resolve(self):
        """A lowercase word that happens to end in an alias is not a suffix."""
        for name, was in (
            ("wall_watercolor.png", "Base_Color"),   # ...water+COLOR
            ("char_thigh.png", "Height"),            # ...t+HIGH
            ("panel_gunmetal.png", "Metallic"),      # ...gun+METAL
            ("prop_raincoat.png", "Clearcoat"),      # ...rain+COAT
            ("bone_abnormal.png", "Normal"),         # ...ab+NORMAL
            ("road_borough.png", "Roughness"),       # ...bo+ROUGH
            ("cloth_lipgloss.png", "Glossiness"),    # ...lip+GLOSS
            ("wall_damask.png", "Mask"),             # ...dam+ASK
            ("wall_afterglow.png", "Emissive"),      # ...after+GLOW
        ):
            with self.subTest(name=name):
                self.assertIsNone(
                    self.reg.resolve_type_from_path(name),
                    f"{name} is an ordinary word (was misread as {was})",
                )

    def test_camelcase_glued_long_aliases_still_resolve(self):
        """The counterweight: a capitalised glued alias IS a suffix.

        This is what rules out the 'require a non-alphabetic predecessor'
        alternative -- every one of these has a lowercase letter in front of
        the alias, and ``mat_SpecularGloss`` is the fixture behind extapps'
        shipped Unpack SpecularGloss workflow.
        """
        for name, expected in (
            ("mat_SpecularGloss.png", "Glossiness"),
            ("test_MetSmooth.png", "Smoothness"),
            ("test_SpecAlpha.png", "Opacity"),
            ("rockBaseColor.png", "Base_Color"),
        ):
            with self.subTest(name=name):
                self.assertEqual(self.reg.resolve_type_from_path(name), expected)

    def test_delimited_long_aliases_still_resolve_in_any_case(self):
        """A separator is an explicit authoring decision, so case is free."""
        for name, expected in (
            ("rock_basecolor.png", "Base_Color"),
            ("rock-basecolor.png", "Base_Color"),
            ("tile_roughness.png", "Roughness"),
            ("wall_Normal.png", "Normal"),
        ):
            with self.subTest(name=name):
                self.assertEqual(self.reg.resolve_type_from_path(name), expected)

    def test_an_ordinary_word_does_not_split_a_texture_set(self):
        """The damage is worse than misclassification: it splits sets.

        Because ``get_suffix_strip_pattern`` carried the same length threshold,
        the base name lost the false suffix too -- so the diffuse and the normal
        of one material based to different stems and grouped as two sets.
        """
        for name in ("char_thigh.png", "wall_watercolor.png"):
            with self.subTest(name=name):
                stem = name.rsplit(".", 1)[0]
                self.assertEqual(ImgUtils.get_base_texture_name(name), stem)

        self.assertEqual(
            ImgUtils.get_base_texture_name("char_thigh.png"),
            ImgUtils.get_base_texture_name("char_thigh_Normal.png"),
            "one material must not base to two different texture sets",
        )
    def test_a_bare_alias_filename_keeps_a_usable_base_name(self):
        """A file named only after its map type must not base to "".

        Fell out of the same threshold: the boundary-free long branch matched
        the WHOLE stem, so ``Normal.png`` based to the empty string and every
        such file collapsed into one anonymous texture set. The attached branch
        now needs a lowercase character in front of the suffix, which a
        stem-initial alias does not have, so the name survives intact.
        """
        for name in ("Normal.png", "Roughness.png", "basecolor.png"):
            with self.subTest(name=name):
                self.assertEqual(
                    ImgUtils.get_base_texture_name(name), name.rsplit(".", 1)[0]
                )
                # still classified - only the base name changed
                self.assertIsNotNone(self.reg.resolve_type_from_path(name))


if __name__ == "__main__":
    unittest.main()
