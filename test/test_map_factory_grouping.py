#!/usr/bin/python
# coding=utf-8
"""
Tests for MapFactory grouping/filtering helpers used by mayatk.MatUpdater:
- group_textures_by_set
- filter_redundant_maps

These drive multi-set detection and PBR map dedup. They have no on-disk
side effects — pure dict transforms.
"""
import os
import unittest

from pythontk import MapFactory

from conftest import BaseTestCase


class GroupTexturesBySetTest(BaseTestCase):
    def test_single_set(self):
        files = [
            "/x/asset_BaseColor.png",
            "/x/asset_Normal.png",
            "/x/asset_Roughness.png",
        ]
        sets = MapFactory.group_textures_by_set(files)
        self.assertEqual(len(sets), 1, f"Expected one set, got {sets}")
        only = next(iter(sets.values()))
        self.assertEqual(len(only), 3)

    def test_multiple_sets(self):
        files = [
            "/x/wood_BaseColor.png",
            "/x/wood_Normal.png",
            "/x/metal_BaseColor.png",
            "/x/metal_Roughness.png",
        ]
        sets = MapFactory.group_textures_by_set(files)
        self.assertEqual(len(sets), 2, f"Expected two sets, got {list(sets)}")
        # Each set has 2 files
        for files_in_set in sets.values():
            self.assertEqual(len(files_in_set), 2)

    def test_empty_input(self):
        self.assertEqual(MapFactory.group_textures_by_set([]), {})

    def test_returns_full_paths_not_basenames(self):
        files = ["/some/dir/asset_Normal.png"]
        sets = MapFactory.group_textures_by_set(files)
        only_files = next(iter(sets.values()))
        self.assertEqual(only_files, files)


class FilterRedundantMapsTest(BaseTestCase):
    def first_precedence_rule(self):
        """Return the first ``(dominant, redundants)`` pair that has redundants.

        The precedence table is derived from ``MapType.replaces`` over
        ``MapRegistry._maps`` -- a hard-coded class attribute, with no optional
        dependency or environment behind it. An empty table is therefore a
        regression, never a valid environment. Assert instead of skipping: a
        skip counts as a pass, so an emptied registry would read green.

        Returns:
            tuple: ``(dominant_map_name, [redundant_map_name, ...])``.
        """
        rules = MapFactory.get_precedence_rules()
        self.assertTrue(
            rules,
            "Precedence-rule registry is empty; MapRegistry._maps carries the "
            "'replaces' relationships as source, so this is a regression",
        )
        dominant, redundants = next(
            ((d, r) for d, r in rules.items() if r), (None, None)
        )
        self.assertIsNotNone(
            dominant,
            "No precedence rule declares any redundants; every 'replaces' list "
            "in MapRegistry._maps is empty, so this is a regression",
        )
        return dominant, redundants

    def test_empty_dict_no_op(self):
        d = {}
        MapFactory.filter_redundant_maps(d)
        self.assertEqual(d, {})

    def test_no_redundancy_preserved(self):
        d = {
            "Base_Color": ["/x/a_BaseColor.png"],
            "Normal": ["/x/a_Normal.png"],
        }
        original = {k: list(v) for k, v in d.items()}
        MapFactory.filter_redundant_maps(d)
        self.assertEqual(d, original)

    def test_in_place_mutation_with_report(self):
        """Mutates the dict in place; the return value is a decision report."""
        d = {"Base_Color": ["/x/a.png"]}
        ret = MapFactory.filter_redundant_maps(d)
        self.assertEqual(ret, {"dropped": {}, "extracted": {}})

    def test_dominant_removes_redundant(self):
        """Verify a precedence rule actually removes a redundant entry.

        Uses live precedence rules so this stays correct as the registry
        evolves.
        """
        # Find a rule where dominant has a non-empty redundant list
        dominant, redundants = self.first_precedence_rule()
        redundant = redundants[0]
        d = {
            dominant: ["/x/dominant.png"],
            redundant: ["/x/redundant.png"],
        }
        MapFactory.filter_redundant_maps(d)
        self.assertIn(dominant, d)
        self.assertNotIn(redundant, d, f"{redundant} should have been removed")

    def test_redundant_kept_when_dominant_empty(self):
        dominant, redundants = self.first_precedence_rule()
        redundant = redundants[0]
        d = {
            dominant: [],  # empty -> not actually present
            redundant: ["/x/redundant.png"],
        }
        MapFactory.filter_redundant_maps(d)
        self.assertIn(redundant, d, "Redundant kept when dominant has no files")

    # --- Workflow-aware redundancy (packed vs. separate maps) ---
    #
    # Regression: the "PBR Metallic/Roughness" preset (mask_map=False) left the
    # packed MSAO connected and dropped the separate Metallic/Roughness/AO maps,
    # because the packed map unconditionally "replaced" its loose components.

    def test_unpacked_preset_drops_packed_keeps_separate(self):
        """mask_map=False with separates present -> MSAO dropped, separates kept."""
        d = {
            "MSAO": ["/x/asset_MSAO.png"],
            "Metallic": ["/x/asset_Metallic.png"],
            "Roughness": ["/x/asset_Roughness.png"],
            "Ambient_Occlusion": ["/x/asset_AO.png"],
        }
        MapFactory.filter_redundant_maps(d, config={"mask_map": False})
        self.assertNotIn("MSAO", d, "Redundant MSAO should be dropped for an unpacked preset")
        self.assertIn("Metallic", d)
        self.assertIn("Roughness", d)
        self.assertIn("Ambient_Occlusion", d)

    def test_packed_preset_keeps_packed_drops_separate(self):
        """mask_map=True -> MSAO supersedes the separate components."""
        d = {
            "MSAO": ["/x/asset_MSAO.png"],
            "Metallic": ["/x/asset_Metallic.png"],
            "Roughness": ["/x/asset_Roughness.png"],
            "Ambient_Occlusion": ["/x/asset_AO.png"],
        }
        MapFactory.filter_redundant_maps(d, config={"mask_map": True})
        self.assertIn("MSAO", d)
        self.assertNotIn("Metallic", d)
        self.assertNotIn("Roughness", d)
        self.assertNotIn("Ambient_Occlusion", d)

    def test_unpacked_preset_keeps_packed_when_no_separates(self):
        """mask_map=False but only MSAO present -> keep it (sole source of channels)."""
        d = {"MSAO": ["/x/asset_MSAO.png"]}
        MapFactory.filter_redundant_maps(d, config={"mask_map": False})
        self.assertIn("MSAO", d, "Packed map kept when no separate components exist")

    def test_force_packed_overrides_unpacked_preset(self):
        """force_packed_maps=True keeps the packed map even when its flag is off."""
        d = {
            "MSAO": ["/x/asset_MSAO.png"],
            "Metallic": ["/x/asset_Metallic.png"],
        }
        MapFactory.filter_redundant_maps(
            d, config={"mask_map": False, "force_packed_maps": True}
        )
        self.assertIn("MSAO", d)
        self.assertNotIn("Metallic", d)

    def test_no_config_preserves_legacy_packed_wins(self):
        """Omitting config keeps the original packed-wins behavior."""
        d = {
            "MSAO": ["/x/asset_MSAO.png"],
            "Metallic": ["/x/asset_Metallic.png"],
        }
        MapFactory.filter_redundant_maps(d)
        self.assertIn("MSAO", d)
        self.assertNotIn("Metallic", d)

    # --- Albedo_Transparency / Metallic_Smoothness precedence ---
    #
    # Regression: neither declared what it replaces, so packing opacity into the
    # albedo (or smoothness into the metallic) left the separate Opacity /
    # Roughness maps in the set — both then wired into the same material slot,
    # and the loose map won.

    def test_albedo_transparency_supersedes_base_color_and_opacity(self):
        """albedo_transparency=True -> the packed albedo is the only source."""
        d = {
            "Albedo_Transparency": ["/x/asset_Albedo_Transparency.png"],
            "Base_Color": ["/x/asset_Base_color.png"],
            "Opacity": ["/x/asset_Opacity.png"],
        }
        MapFactory.filter_redundant_maps(d, config={"albedo_transparency": True})
        self.assertIn("Albedo_Transparency", d)
        self.assertNotIn("Base_Color", d)
        self.assertNotIn("Opacity", d)

    def test_unpacked_preset_drops_albedo_transparency(self):
        """albedo_transparency=False with separates present -> packed dropped."""
        d = {
            "Albedo_Transparency": ["/x/asset_Albedo_Transparency.png"],
            "Base_Color": ["/x/asset_Base_color.png"],
            "Opacity": ["/x/asset_Opacity.png"],
        }
        MapFactory.filter_redundant_maps(d, config={"albedo_transparency": False})
        self.assertNotIn("Albedo_Transparency", d)
        self.assertIn("Base_Color", d)
        self.assertIn("Opacity", d)

    def test_metallic_smoothness_supersedes_its_components(self):
        """metallic_smoothness=True -> the packed map supersedes the separates."""
        d = {
            "Metallic_Smoothness": ["/x/asset_MetallicSmoothness.png"],
            "Metallic": ["/x/asset_Metallic.png"],
            "Roughness": ["/x/asset_Roughness.png"],
        }
        MapFactory.filter_redundant_maps(d, config={"metallic_smoothness": True})
        self.assertIn("Metallic_Smoothness", d)
        self.assertNotIn("Metallic", d)
        self.assertNotIn("Roughness", d)


class CoverageAwareRedundancyTest(BaseTestCase):
    """Dropping a packed map must not lose channels no loose map covers.

    Regression class (flagged live): an unpacked preset meeting an MSAO with no
    separate AO dropped the MSAO — and its AO channel with it. The filter now
    extracts exactly the uncovered channels (via the conversion registry) before
    dropping the packed map, and reports every decision.
    """

    def setUp(self):
        super().setUp()
        import tempfile

        self.tmp = tempfile.mkdtemp(prefix="cov_")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def _msao(self, name="asset_MSAO.png", ao=200):
        """Write an RGBA MSAO: R=Metallic 30, G=AO `ao`, B=Detail 0, A=Smooth 90."""
        import os
        from PIL import Image

        path = os.path.join(self.tmp, name)
        Image.merge(
            "RGBA",
            [
                Image.new("L", (8, 8), v)
                for v in (30, ao, 0, 90)
            ],
        ).save(path)
        return path

    def _loose(self, name, value=128):
        import os
        from PIL import Image

        path = os.path.join(self.tmp, name)
        Image.new("L", (8, 8), value).save(path)
        return path

    def test_uncovered_channel_is_extracted_before_drop(self):
        """MSAO + Metallic + Roughness, no AO: AO is extracted, MSAO dropped."""
        import os
        from PIL import Image

        d = {
            "MSAO": [self._msao()],
            "Metallic": [self._loose("asset_Metallic.png")],
            "Roughness": [self._loose("asset_Roughness.png")],
        }
        report = MapFactory.filter_redundant_maps(d, config={"mask_map": False})

        self.assertNotIn("MSAO", d, "packed map should drop once channels are safe")
        self.assertIn("Ambient_Occlusion", d, "uncovered AO channel not extracted")
        ao_path = d["Ambient_Occlusion"][0]
        self.assertTrue(os.path.isfile(ao_path), "extracted AO not on disk")
        # The extracted file carries the packed G channel's data.
        self.assertEqual(Image.open(ao_path).convert("L").getpixel((4, 4)), 200)
        self.assertIn("Ambient_Occlusion", report["extracted"])
        self.assertIn("MSAO", report["dropped"])

    def test_covered_by_conversion_equivalent_needs_no_extraction(self):
        """A loose Roughness covers MSAO's Smoothness channel (derivable)."""
        d = {
            "MSAO": [self._msao()],
            "Metallic": [self._loose("asset_Metallic.png")],
            "Roughness": [self._loose("asset_Roughness.png")],
            "Ambient_Occlusion": [self._loose("asset_AO.png")],
        }
        report = MapFactory.filter_redundant_maps(d, config={"mask_map": False})

        self.assertNotIn("MSAO", d)
        self.assertEqual(
            report["extracted"], {}, "fully covered map must not extract anything"
        )

    def test_optional_filler_channel_never_demands_extraction(self):
        """MSAO's Detail channel is filler — no Detail_Mask is synthesized."""
        d = {
            "MSAO": [self._msao()],
            "Metallic": [self._loose("asset_Metallic.png")],
            "Roughness": [self._loose("asset_Roughness.png")],
            "Ambient_Occlusion": [self._loose("asset_AO.png")],
        }
        MapFactory.filter_redundant_maps(d, config={"mask_map": False})
        self.assertNotIn("Detail_Mask", d)

    def test_extraction_disabled_keeps_packed_over_losing_data(self):
        """extract_missing=False with an uncovered channel: packed map stays."""
        d = {
            "MSAO": [self._msao()],
            "Metallic": [self._loose("asset_Metallic.png")],
            "Roughness": [self._loose("asset_Roughness.png")],
        }
        report = MapFactory.filter_redundant_maps(
            d, config={"mask_map": False}, extract_missing=False
        )

        self.assertIn("MSAO", d, "dropping the packed map would lose its AO")
        self.assertNotIn("Metallic", d, "packed won — its components retire")
        self.assertIn("MSAO", str(report["dropped"].get("Metallic", "")))

    def test_dry_run_reports_extraction_without_writing(self):
        """dry_run plans the same outcome but writes nothing to disk."""
        import os

        d = {
            "MSAO": [self._msao()],
            "Metallic": [self._loose("asset_Metallic.png")],
            "Roughness": [self._loose("asset_Roughness.png")],
        }
        report = MapFactory.filter_redundant_maps(
            d, config={"mask_map": False, "dry_run": True}
        )

        self.assertNotIn("MSAO", d)
        self.assertIn("Ambient_Occlusion", report["extracted"])
        self.assertFalse(
            os.path.isfile(report["extracted"]["Ambient_Occlusion"]),
            "dry run must not write the extracted file",
        )

    def test_albedo_transparency_opacity_extracted_when_uncovered(self):
        """AT beside a loose Base_Color but no loose Opacity: opacity recovered.

        (A *lone* AT with no loose components stays — sole source of its
        channels; this pins the conflict shape.)
        """
        import os
        from PIL import Image

        at = os.path.join(self.tmp, "asset_Albedo_Transparency.png")
        Image.merge(
            "RGBA",
            [Image.new("L", (8, 8), v) for v in (180, 90, 40, 77)],
        ).save(at)

        d = {
            "Albedo_Transparency": [at],
            "Base_Color": [self._loose("asset_Base_color.png")],
        }
        report = MapFactory.filter_redundant_maps(
            d, config={"albedo_transparency": False}
        )

        self.assertNotIn("Albedo_Transparency", d)
        self.assertIn("Base_Color", d)
        self.assertIn("Opacity", d)
        opacity = Image.open(d["Opacity"][0]).convert("L")
        self.assertEqual(opacity.getpixel((4, 4)), 77, "opacity != packed alpha")
        self.assertEqual(list(report["extracted"]), ["Opacity"])

    def test_extraction_reuses_existing_file_instead_of_overwriting(self):
        """A real loose map on disk under the canonical name must survive.

        The set didn't list it, but extraction writes to exactly that name —
        overwriting user data with extracted channel data would make the
        'lossless' feature destructive.
        """
        from PIL import Image

        msao = self._msao()  # G (AO) channel = 200
        existing_ao = self._loose("asset_Ambient_Occlusion.png", value=55)

        d = {
            "MSAO": [msao],
            "Metallic": [self._loose("asset_Metallic.png")],
            "Roughness": [self._loose("asset_Roughness.png")],
        }
        report = MapFactory.filter_redundant_maps(d, config={"mask_map": False})

        self.assertNotIn("MSAO", d)
        self.assertEqual(d["Ambient_Occlusion"], [existing_ao])
        self.assertEqual(
            Image.open(existing_ao).convert("L").getpixel((4, 4)),
            55,
            "existing loose AO was overwritten with extracted channel data",
        )
        self.assertEqual(report["extracted"]["Ambient_Occlusion"], existing_ao)

    def test_single_path_values_supported(self):
        """mat_updater passes {type: path} (not lists) — both shapes work."""
        d = {
            "MSAO": self._msao(),
            "Metallic": self._loose("asset_Metallic.png"),
            "Roughness": self._loose("asset_Roughness.png"),
        }
        MapFactory.filter_redundant_maps(d, config={"mask_map": False})
        self.assertNotIn("MSAO", d)
        self.assertIn("Ambient_Occlusion", d)


class PackedVsPackedTest(BaseTestCase):
    """Two PACKED maps carrying the same channels must not both survive.

    Regression (found live): the glTF preset connected BOTH an ORM and an HDRP
    MSAO. Precedence is keyed off `replaces`, and no packing lists another
    packing — so the requested ORM retired the loose Metallic/Roughness/AO
    first, after which the MSAO saw no loose components left and took the
    "sole source of its channels" branch. Both then wired into the same
    metallic / roughness / AO slots, and the FBX-safe connection carried the
    foreign mask map into the GLB as dead payload.
    """

    def _preset(self, workflow, **overrides):
        from pythontk.core_utils.engines.textures.map_registry import MapRegistry

        cfg = dict(MapRegistry().get_workflow_presets()[workflow])
        cfg.update(overrides)
        return cfg

    def test_gltf_preset_drops_the_hdrp_mask_map(self):
        """The reported bug: glTF wants ORM, so the MSAO is foreign."""
        from pythontk.core_utils.engines.textures.map_registry import WF

        d = {
            "Base_Color": "/x/a_Base_color.png",
            "Normal_OpenGL": "/x/a_Normal_OpenGL.png",
            "ORM": "/x/a_ORM.png",
            "MSAO": "/x/a_MSAO.png",
        }
        report = MapFactory.filter_redundant_maps(d, config=self._preset(WF.GLTF))

        self.assertIn("ORM", d, "glTF's own packing must survive")
        self.assertNotIn("MSAO", d, "the HDRP mask map must not stay wired")
        self.assertIn("MSAO", report["dropped"])
        self.assertIn("ORM", report["dropped"]["MSAO"])
        self.assertEqual(sorted(d), ["Base_Color", "Normal_OpenGL", "ORM"])

    def test_hdrp_preset_drops_the_gltf_orm(self):
        """Symmetric: the gap kept whichever packing the preset did not want."""
        from pythontk.core_utils.engines.textures.map_registry import WF

        d = {"ORM": "/x/a_ORM.png", "MSAO": "/x/a_MSAO.png"}
        MapFactory.filter_redundant_maps(d, config=self._preset(WF.HDRP))

        self.assertEqual(list(d), ["MSAO"])

    def test_explicit_flag_outranks_force_packed_maps(self):
        """`force_packed_maps` must not promote a foreign packing to a peer.

        It means "emit a packed map even with a source channel missing", not
        "keep every packing" — otherwise ticking it re-opens the double-wire.
        """
        from pythontk.core_utils.engines.textures.map_registry import WF

        d = {"ORM": "/x/a_ORM.png", "MSAO": "/x/a_MSAO.png"}
        MapFactory.filter_redundant_maps(
            d, config=self._preset(WF.GLTF, force_packed_maps=True)
        )

        self.assertEqual(list(d), ["ORM"])

    def test_no_config_keeps_exactly_one_packing(self):
        """With no stated target, the broadly-targeted packing wins.

        The compositor calls this with no config; "legacy packed-wins" is about
        packed-vs-LOOSE and never licensed two packings fighting for one slot.
        ORM declares UE/glTF/Godot, MSAO only HDRP.
        """
        d = {"ORM": "/x/a_ORM.png", "MSAO": "/x/a_MSAO.png"}
        MapFactory.filter_redundant_maps(d)

        self.assertEqual(list(d), ["ORM"])

    def test_a_three_way_pile_up_collapses_in_one_pass(self):
        """All three drive metallic / roughness / AO, so only one may survive.

        Pins the descending walk: the MSAO is judged against both survivors
        that outrank it, and the Metallic_Smoothness against the ORM, in a
        single pass. This is the blendertk `resolve_pbr_plan` case at the SSoT
        layer, where the behaviour is actually decided.
        """
        d = {
            "ORM": "/x/a_ORM.png",
            "MSAO": "/x/a_MaskMap.png",
            "Metallic_Smoothness": "/x/a_MetallicSmoothness.png",
        }
        report = MapFactory.filter_redundant_maps(d)

        self.assertEqual(list(d), ["ORM"])
        self.assertEqual(sorted(report["dropped"]), ["MSAO", "Metallic_Smoothness"])
        self.assertEqual(
            report["extracted"], {}, "every channel is derivable from the ORM"
        )

    def test_non_overlapping_packings_both_survive(self):
        """Albedo_Transparency shares no channel with ORM — it is not a rival."""
        from pythontk.core_utils.engines.textures.map_registry import WF

        d = {
            "Albedo_Transparency": "/x/a_Albedo_Transparency.png",
            "ORM": "/x/a_ORM.png",
        }
        MapFactory.filter_redundant_maps(d, config=self._preset(WF.GLTF))

        self.assertEqual(sorted(d), ["Albedo_Transparency", "ORM"])

    def test_lone_packed_map_is_untouched(self):
        """No rival packing: the MSAO is still the sole source of its channels."""
        from pythontk.core_utils.engines.textures.map_registry import WF

        d = {"MSAO": "/x/a_MSAO.png"}
        MapFactory.filter_redundant_maps(d, config=self._preset(WF.GLTF))

        self.assertEqual(list(d), ["MSAO"])

    def test_loose_maps_still_retire_behind_the_surviving_packing(self):
        """The packed-vs-loose pass must still run on what the pre-pass leaves."""
        from pythontk.core_utils.engines.textures.map_registry import WF

        d = {
            "ORM": "/x/a_ORM.png",
            "MSAO": "/x/a_MSAO.png",
            "Metallic": "/x/a_Metallic.png",
            "Roughness": "/x/a_Roughness.png",
            "Ambient_Occlusion": "/x/a_AO.png",
        }
        MapFactory.filter_redundant_maps(d, config=self._preset(WF.GLTF))

        self.assertEqual(list(d), ["ORM"])


class PackedVsPackedExtractionTest(BaseTestCase):
    """A losing packing may carry a channel the winner does not — never lose it."""

    def setUp(self):
        super().setUp()
        import pythontk as ptk

        self._artifacts = ptk.TempArtifacts("packed_conflict")
        self.tmp = self._artifacts.dir_path()

    def tearDown(self):
        self._artifacts.cleanup()
        super().tearDown()

    def _msao(self, name="asset_MSAO.png", ao=200):
        """RGBA MSAO: R=Metallic 30, G=AO `ao`, B=Detail 0, A=Smoothness 90."""
        from PIL import Image

        path = os.path.join(self.tmp, name)
        Image.merge(
            "RGBA", [Image.new("L", (8, 8), v) for v in (30, ao, 0, 90)]
        ).save(path)
        return path

    def _metallic_smoothness(self, name="asset_MetallicSmoothness.png"):
        from PIL import Image

        path = os.path.join(self.tmp, name)
        Image.merge(
            "RGBA", [Image.new("L", (8, 8), v) for v in (60, 60, 60, 150)]
        ).save(path)
        return path

    def test_uncovered_channel_is_extracted_before_the_loser_drops(self):
        """URP wants Metallic_Smoothness, which carries no AO — extract MSAO's."""
        from PIL import Image
        from pythontk.core_utils.engines.textures.map_registry import MapRegistry, WF

        d = {
            "MSAO": self._msao(),
            "Metallic_Smoothness": self._metallic_smoothness(),
        }
        report = MapFactory.filter_redundant_maps(
            d, config=dict(MapRegistry().get_workflow_presets()[WF.URP])
        )

        self.assertNotIn("MSAO", d, "the foreign packing should retire")
        self.assertIn("Metallic_Smoothness", d)
        self.assertIn("Ambient_Occlusion", d, "AO channel lost with the MSAO")
        self.assertEqual(
            list(report["extracted"]),
            ["Ambient_Occlusion"],
            "only the channel the winner cannot supply should be written",
        )
        self.assertEqual(
            Image.open(d["Ambient_Occlusion"]).convert("L").getpixel((4, 4)), 200
        )

    def test_a_surviving_loose_map_is_not_re_extracted_from_the_loser(self):
        """A channel the inventory already carries loose must not be rewritten.

        The winner (Metallic_Smoothness) carries no AO, but the caller listed
        their own AO map. Extracting the loser's AO channel anyway swaps that
        inventory entry for a derived file — so the material gets channel data
        in place of the authored map, plus a redundant write. The on-disk
        `output_path_for` guard does not catch it: the caller's file need not
        sit under the canonical name (measured — it produced
        `asset_Ambient_Occlusion.png` beside an `asset_Mixed_AO.png`).
        """
        from pythontk.core_utils.engines.textures.map_registry import MapRegistry, WF

        loose_ao = os.path.join(self.tmp, "asset_Mixed_AO.png")
        from PIL import Image

        Image.new("L", (8, 8), 55).save(loose_ao)

        d = {
            "MSAO": self._msao(),
            "Metallic_Smoothness": self._metallic_smoothness(),
            "Ambient_Occlusion": loose_ao,
        }
        report = MapFactory.filter_redundant_maps(
            d, config=dict(MapRegistry().get_workflow_presets()[WF.URP])
        )

        self.assertNotIn("MSAO", d)
        self.assertEqual(report["extracted"], {}, "nothing needed extracting")
        self.assertEqual(
            d["Ambient_Occlusion"], loose_ao, "the caller's own AO was replaced"
        )
        self.assertNotIn(
            "asset_Ambient_Occlusion.png",
            os.listdir(self.tmp),
            "a redundant AO was extracted beside the caller's own",
        )

    def test_extraction_unavailable_keeps_both_rather_than_losing_a_channel(self):
        """Lossless direction: keep the rival rather than drop its only AO."""
        from pythontk.core_utils.engines.textures.map_registry import MapRegistry, WF

        d = {
            "MSAO": self._msao(),
            "Metallic_Smoothness": self._metallic_smoothness(),
        }
        MapFactory.filter_redundant_maps(
            d,
            config=dict(MapRegistry().get_workflow_presets()[WF.URP]),
            extract_missing=False,
        )

        self.assertIn("MSAO", d, "dropping it would lose the AO channel")
        self.assertIn("Metallic_Smoothness", d)


class PackedPrecedenceTest(BaseTestCase):
    """`MapRegistry.packed_precedence` — the deterministic packed-map ranking."""

    def setUp(self):
        super().setUp()
        from pythontk.core_utils.engines.textures.map_registry import MapRegistry

        self.registry = MapRegistry()

    def _rank(self, config, *names):
        order = self.registry.packed_precedence(config)
        return [n for n in order if n in names]

    def test_requested_packing_outranks_every_other(self):
        from pythontk.core_utils.engines.textures.map_registry import WF

        preset = self.registry.get_workflow_presets()[WF.GLTF]
        self.assertEqual(self._rank(preset, "ORM", "MSAO", "MRAO")[0], "ORM")

    def test_a_named_request_outranks_both_force_and_breadth(self):
        """MSAO is asked for by name; ORM only by force — and ORM is broader.

        If the force escape hatch conferred the same level as a named flag,
        the breadth tiebreak would put the ORM first and the mask map would
        lose under a preset that explicitly asked for it.
        """
        order = self._rank(
            {"mask_map": True, "orm_map": False, "force_packed_maps": True},
            "ORM",
            "MSAO",
        )
        self.assertEqual(order, ["MSAO", "ORM"])

    def test_breadth_breaks_ties_when_nothing_is_requested(self):
        """ORM targets UE/glTF/Godot; MSAO only HDRP; MRAO declares none."""
        self.assertEqual(
            self._rank(None, "ORM", "MSAO", "MRAO"), ["ORM", "MSAO", "MRAO"]
        )

    def test_every_packed_type_is_listed_exactly_once(self):
        order = self.registry.packed_precedence(None)
        packed = [
            n for n in self.registry.get_map_types() if self.registry.get(n).is_packed
        ]
        self.assertEqual(sorted(order), sorted(packed))
        self.assertEqual(len(order), len(set(order)), "duplicate entries")

    def test_loose_types_are_never_listed(self):
        order = self.registry.packed_precedence(None)
        self.assertNotIn("Metallic", order)
        self.assertNotIn("Base_Color", order)


class ResolveNormalMapsTest(BaseTestCase):
    """One normal map per inventory, optionally in a requested convention.

    `Normal`, `Normal_OpenGL` and `Normal_DirectX` have no `replaces` relation,
    so `filter_redundant_maps` never collapses them — yet all three drive the
    same shader input, and a set carrying two wired it twice.
    """

    def setUp(self):
        super().setUp()
        import pythontk as ptk

        self._artifacts = ptk.TempArtifacts("resolve_normals")
        self.tmp = self._artifacts.dir_path()

    def tearDown(self):
        self._artifacts.cleanup()
        super().tearDown()

    def _normal(self, name, color=(120, 60, 240)):
        from PIL import Image

        path = os.path.join(self.tmp, name)
        Image.new("RGB", (8, 8), color).save(path)
        return path

    def test_explicit_tag_supersedes_the_ambiguous_map(self):
        inv = {"Normal": "/x/a.png", "Normal_OpenGL": "/x/b.png"}
        report = MapFactory.resolve_normal_maps(inv)
        self.assertEqual(list(inv), ["Normal_OpenGL"])
        self.assertIn("Normal", report["dropped"])

    def test_all_losing_types_are_dropped(self):
        inv = {
            "Normal": "/x/a.png",
            "Normal_OpenGL": "/x/b.png",
            "Normal_DirectX": "/x/c.png",
            "Roughness": "/x/r.png",
        }
        MapFactory.resolve_normal_maps(inv)
        self.assertEqual(sorted(inv), ["Normal_OpenGL", "Roughness"])

    def test_no_target_format_leaves_the_survivor_untouched(self):
        """Blender's path: the graph flips green, nothing is written."""
        inv = {"Normal_DirectX": "/x/c.png"}
        report = MapFactory.resolve_normal_maps(inv)
        self.assertEqual(inv, {"Normal_DirectX": "/x/c.png"})
        self.assertEqual(report["converted"], {})

    def test_target_format_converts_the_opposite_tag(self):
        """Maya's path: the file has to be right, so it is rewritten."""
        from PIL import Image

        src = self._normal("conv_Normal_DirectX.png")
        inv = {"Normal_DirectX": src}
        report = MapFactory.resolve_normal_maps(inv, target_format="OpenGL")

        self.assertEqual(list(inv), ["Normal_OpenGL"])
        out = inv["Normal_OpenGL"]
        self.assertTrue(os.path.isfile(out), out)
        self.assertEqual(report["converted"]["Normal_OpenGL"], out)
        # The source type must ALSO be reported dropped, or a caller mapping the
        # report onto real paths keeps the original file and double-wires.
        self.assertIn("Normal_DirectX", report["dropped"])
        self.assertEqual(
            Image.open(out).convert("RGB").getpixel((4, 4))[1],
            255 - Image.open(src).convert("RGB").getpixel((4, 4))[1],
        )

    def test_matching_target_format_is_a_no_op(self):
        inv = {"Normal_OpenGL": "/x/b.png"}
        report = MapFactory.resolve_normal_maps(inv, target_format="OpenGL")
        self.assertEqual(inv, {"Normal_OpenGL": "/x/b.png"})
        self.assertEqual(report["converted"], {})

    def test_ambiguous_map_is_never_converted(self):
        """Its convention is unknown — flipping could invert a correct map."""
        inv = {"Normal": "/x/a.png"}
        report = MapFactory.resolve_normal_maps(inv, target_format="DirectX")
        self.assertEqual(inv, {"Normal": "/x/a.png"})
        self.assertEqual(report["converted"], {})

    def test_convert_false_keeps_the_mismatched_map(self):
        inv = {"Normal_DirectX": "/x/c.png"}
        MapFactory.resolve_normal_maps(inv, target_format="OpenGL", convert=False)
        self.assertEqual(inv, {"Normal_DirectX": "/x/c.png"})

    def test_target_format_is_matched_case_insensitively(self):
        """`convert_normal_map_format` takes "opengl"; so must this.

        Building the type by interpolation instead would yield the unregistered
        `Normal_opengl` and put a type nothing in the taxonomy recognises into
        the caller's inventory.
        """
        src = self._normal("case_Normal_DirectX.png")
        inv = {"Normal_DirectX": src}
        MapFactory.resolve_normal_maps(inv, target_format="opengl")
        self.assertEqual(list(inv), ["Normal_OpenGL"])

    def test_unknown_convention_leaves_the_map_untouched(self):
        inv = {"Normal_DirectX": "/x/c.png"}
        report = MapFactory.resolve_normal_maps(inv, target_format="Vulkan")
        self.assertEqual(inv, {"Normal_DirectX": "/x/c.png"})
        self.assertEqual(report["converted"], {})

    def test_empty_inventory_entry_does_not_raise(self):
        inv = {"Normal_DirectX": []}
        report = MapFactory.resolve_normal_maps(inv, target_format="OpenGL")
        self.assertEqual(report["converted"], {})

    def test_no_normals_is_a_no_op(self):
        inv = {"Base_Color": "/x/bc.png"}
        report = MapFactory.resolve_normal_maps(inv, target_format="OpenGL")
        self.assertEqual(inv, {"Base_Color": "/x/bc.png"})
        self.assertEqual(report, {"dropped": {}, "converted": {}})

    def test_list_valued_inventories_keep_their_shape(self):
        src = self._normal("shape_Normal_DirectX.png")
        inv = {"Normal_DirectX": [src]}
        MapFactory.resolve_normal_maps(inv, target_format="OpenGL")
        self.assertIsInstance(inv["Normal_OpenGL"], list)


class UdimGroupingTest(BaseTestCase):
    """A UDIM set is one complete texture set PER TILE.

    Each tile is an independent image, so it is its own processable set — but
    the tiles of one material must not fragment by map type the way they did
    when the tile token hid the suffix (``rock_Normal.1001`` grouped alone,
    keyed by its own full stem).
    """

    FILES = [
        "/x/rock_BaseColor.1001.png",
        "/x/rock_Normal.1001.png",
        "/x/rock_Roughness.1001.png",
        "/x/rock_BaseColor.1002.png",
        "/x/rock_Normal.1002.png",
        "/x/rock_Roughness.1002.png",
    ]

    def test_tiles_group_into_one_set_each(self):
        sets = MapFactory.group_textures_by_set(self.FILES)
        self.assertEqual(sorted(sets), ["rock.1001", "rock.1002"], f"got {list(sets)}")
        for key, files in sets.items():
            self.assertEqual(len(files), 3, f"{key}: {files}")

    def test_base_name_drops_the_tile_token(self):
        """The material name is the material's, not the tile's."""
        for name in ("/x/rock_Normal.1001.png", "/x/rock_Normal.1002.png"):
            self.assertEqual(MapFactory.get_base_texture_name(name), "rock")

    def test_get_tile_token_reads_the_token_back(self):
        self.assertEqual(MapFactory.get_tile_token("/x/rock_Normal.1001.png"), ".1001")
        self.assertEqual(MapFactory.get_tile_token("/x/rock_Normal.png"), "")

    def test_sort_by_type_sees_every_tile(self):
        by_type = MapFactory.sort_images_by_type(self.FILES)
        self.assertEqual(
            sorted(by_type), ["Base_Color", "Normal", "Roughness"], f"got {by_type}"
        )
        for map_type, paths in by_type.items():
            self.assertEqual(len(paths), 2, f"{map_type}: expected both tiles")

    def test_output_name_keeps_the_tile_token_last(self):
        """``rock_Normal_OpenGL.1001.png`` — a name Maya/Substance still reads
        as a tile sequence, and that two tiles cannot collide on."""
        from pythontk.core_utils.engines.textures.map_factory.processor import (
            TextureProcessor,
        )

        ctx = TextureProcessor(
            inventory={},
            config={},
            output_dir="/out",
            base_name="rock",
            tile_token=".1001",
            ext="png",
            conversion_registry=None,
        )
        self.assertEqual(
            os.path.normpath(ctx.output_path_for("Normal_OpenGL")),
            os.path.normpath("/out/rock_Normal_OpenGL.1001.png"),
        )

    def test_two_tiles_do_not_collide_on_output(self):
        a = MapFactory.resolve_texture_filename("/x/rock_Normal.1001.png", "Normal_OpenGL")
        b = MapFactory.resolve_texture_filename("/x/rock_Normal.1002.png", "Normal_OpenGL")
        self.assertNotEqual(a, b)
        self.assertTrue(a.endswith("rock_Normal_OpenGL.1001.png"), a)


if __name__ == "__main__":
    unittest.main()
