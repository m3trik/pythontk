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
        rules = MapFactory.get_precedence_rules()
        if not rules:
            self.skipTest("No precedence rules registered")
        # Find a rule where dominant has a non-empty redundant list
        dominant, redundants = next(
            ((d, r) for d, r in rules.items() if r), (None, None)
        )
        if not dominant:
            self.skipTest("No precedence rule with redundants")
        redundant = redundants[0]
        d = {
            dominant: ["/x/dominant.png"],
            redundant: ["/x/redundant.png"],
        }
        MapFactory.filter_redundant_maps(d)
        self.assertIn(dominant, d)
        self.assertNotIn(redundant, d, f"{redundant} should have been removed")

    def test_redundant_kept_when_dominant_empty(self):
        rules = MapFactory.get_precedence_rules()
        dominant, redundants = next(
            ((d, r) for d, r in rules.items() if r), (None, None)
        )
        if not dominant:
            self.skipTest("No precedence rule with redundants")
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
        import os
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
