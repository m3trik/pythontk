# !/usr/bin/python
# coding=utf-8
"""Tests for MapRegistry.register — runtime map-type extensibility.

The factory's plug-in story (custom handlers + conversions) only works if the
taxonomy itself is extensible: a conversion whose *source* type isn't in the
registry can never fire through ``prepare_maps``, because unresolvable files
are dropped at inventory build. These tests pin the full chain: register a
custom type → filename resolution / suffix stripping / MapFactory live views
pick it up → ``prepare_maps`` inventories the file → a custom conversion can
consume it.
"""
import os
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock

from pythontk import ImgUtils
from pythontk.core_utils.engines.textures.map_registry import MapRegistry, MapType
from pythontk.core_utils.engines.textures.map_factory import (
    MapFactory,
    TextureProcessor,
    ConversionRegistry,
)


class RegistryStateGuard(unittest.TestCase):
    """Snapshot/restore the process-wide registry around each test."""

    def setUp(self):
        self._saved_maps = dict(MapRegistry._maps)

    def tearDown(self):
        MapRegistry._maps.clear()
        MapRegistry._maps.update(self._saved_maps)
        MapRegistry._invalidate_caches()


class TestMapRegistryRegister(RegistryStateGuard):
    def _curvature(self, **overrides):
        defaults = dict(
            name="Curvature",
            aliases=["CurvatureMap", "Curv"],
            color_space="Linear",
            mode="L",
            default_background=(127, 127, 127, 255),
        )
        defaults.update(overrides)
        return MapType(**defaults)

    def test_registered_type_resolves_from_path(self):
        registry = MapRegistry()
        self.assertIsNone(registry.resolve_type_from_path("brick_Curvature.png"))

        registry.register(self._curvature())

        self.assertEqual(
            registry.resolve_type_from_path("brick_Curvature.png"), "Curvature"
        )
        self.assertEqual(
            registry.resolve_type_from_path("brick_CurvatureMap.png"), "Curvature"
        )

    def test_caches_invalidate_after_register(self):
        """A pre-registration miss must not stick: resolve caches, the sorted
        candidate list, and the suffix-strip pattern all rebuild."""
        registry = MapRegistry()
        # Prime every derived cache with pre-registration state (including a
        # cached None miss for the soon-to-be-registered suffix).
        self.assertIsNone(registry.resolve_type_from_path("wall_Curvature.png"))
        self.assertEqual(
            MapFactory.get_base_texture_name("wall_Curvature.png"), "wall_Curvature"
        )

        registry.register(self._curvature())

        self.assertEqual(
            registry.resolve_type_from_path("wall_Curvature.png"), "Curvature"
        )
        # Suffix-strip pattern rebuilt: the new suffix now strips.
        self.assertEqual(
            MapFactory.get_base_texture_name("wall_Curvature.png"), "wall"
        )

    def test_map_factory_views_are_live(self):
        registry = MapRegistry()
        self.assertNotIn("Curvature", MapFactory.map_types)
        self.assertNotIn("Curvature", MapFactory.passthrough_maps)

        registry.register(self._curvature(scale_as_mask=True))

        self.assertIn("Curvature", MapFactory.map_types)
        self.assertIn("CurvatureMap", MapFactory.map_types["Curvature"])
        self.assertIn("Curvature", MapFactory.passthrough_maps)
        self.assertIn("Curvature", MapFactory.packed_grayscale_maps)

    def test_duplicate_register_guarded(self):
        registry = MapRegistry()
        registry.register(self._curvature())

        # Identical re-registration is a no-op (module-reload safety): the
        # registered instance survives and nothing raises.
        first = registry.get("Curvature")
        registry.register(self._curvature())
        self.assertIs(registry.get("Curvature"), first)

        # A *different* definition under the same name conflicts.
        with self.assertRaises(ValueError):
            registry.register(self._curvature(aliases=["Curvy"]))

        replaced = self._curvature(aliases=["Curvy"])
        registry.register(replaced, overwrite=True)
        self.assertEqual(registry.get("Curvature").aliases, ["Curvy"])

    def test_register_rejects_non_maptype(self):
        with self.assertRaises(TypeError):
            MapRegistry().register({"name": "Curvature"})

    def test_maptype_exported_from_root(self):
        """The registration API's parameter type resolves from the lazy root."""
        import pythontk as ptk

        self.assertIs(ptk.MapType, MapType)
        self.assertIs(ptk.MapRegistry, MapRegistry)


class TestRegisteredTypeThroughFactory(RegistryStateGuard):
    """E2E: a registered custom type flows through prepare_maps and can feed
    a custom conversion — the extensibility example's promised workflow."""

    def setUp(self):
        super().setUp()
        self.test_dir = tempfile.mkdtemp(prefix="map_registry_register_")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.curvature_path = os.path.join(self.test_dir, "mat_Curvature.png")
        ImgUtils.save_image(
            ImgUtils.create_image("L", (16, 16), 200), self.curvature_path
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        super().tearDown()

    def test_prepare_maps_inventories_registered_type(self):
        # Unregistered: the file is unresolvable, so nothing is produced in
        # the output dir (prepare_maps falls back to returning the inputs).
        results = MapFactory.prepare_maps(
            [self.curvature_path], output_dir=self.output_dir, rename=True
        )
        self.assertEqual(results, [self.curvature_path])

        MapRegistry().register(
            MapType(name="Curvature", aliases=["Curv"], mode="L")
        )

        results = MapFactory.prepare_maps(
            [self.curvature_path], output_dir=self.output_dir, rename=True
        )
        self.assertTrue(
            any(
                os.path.basename(p) == "mat_Curvature.png"
                and os.path.dirname(os.path.abspath(p))
                == os.path.abspath(self.output_dir)
                for p in results
            ),
            f"registered type did not pass through prepare_maps: {results}",
        )

    def test_custom_conversion_fires_from_registered_source(self):
        MapRegistry().register(
            MapType(name="Curvature", aliases=["Curv"], mode="L")
        )

        conversions = ConversionRegistry()
        conversions.register(
            "Ambient_Occlusion",
            "Curvature",
            lambda inv, ctx: ImgUtils.invert_grayscale_image(
                ImgUtils.ensure_image(inv["Curvature"], "L")
            ),
        )
        ctx = TextureProcessor(
            inventory={"Curvature": self.curvature_path},
            config={},
            output_dir=self.output_dir,
            base_name="mat",
            ext="png",
            conversion_registry=conversions,
            logger=MagicMock(),
        )

        ao = ctx.resolve_map("Ambient_Occlusion", allow_conversion=True)
        self.assertIsNotNone(ao, "conversion from registered source did not fire")
        self.assertEqual(ImgUtils.ensure_image(ao, "L").getpixel((8, 8)), 55)


class TestPackedMapContract(RegistryStateGuard):
    """A packed map must declare `replaces` + `config_key` at definition time.

    Regression class: Albedo_Transparency and Metallic_Smoothness shipped
    without `replaces`, so packing a channel (opacity into the albedo) left the
    separate map it absorbed in the set — both then wired into the same
    material slot downstream. The contract is enforced in
    ``MapType.__post_init__`` so the next packed map type can't repeat it.
    """

    def test_packed_map_without_replaces_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "replaces"):
            MapType(
                name="RMA",
                aliases=["RoughMetalAO"],
                is_packed=True,
                config_key="rma_map",
            )

    def test_packed_map_without_config_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "config_key"):
            MapType(
                name="RMA",
                aliases=["RoughMetalAO"],
                is_packed=True,
                replaces=["Roughness", "Metallic", "Ambient_Occlusion"],
            )

    def test_replaces_without_is_packed_is_rejected(self):
        """The operative field can't bypass the contract.

        `filter_redundant_maps` keys precedence off `replaces`, not `is_packed`
        — and with no `config_key` the direction gate is skipped, so this shape
        would silently supersede its components in every preset.
        """
        with self.assertRaisesRegex(ValueError, "is_packed"):
            MapType(
                name="RMA",
                aliases=["RoughMetalAO"],
                replaces=["Roughness", "Metallic", "Ambient_Occlusion"],
                config_key="rma_map",
            )

    def test_packed_map_with_full_contract_is_accepted(self):
        m = MapType(
            name="RMA",
            aliases=["RoughMetalAO"],
            is_packed=True,
            replaces=["Roughness", "Metallic", "Ambient_Occlusion"],
            channels={"R": "Roughness", "G": "Metallic", "B": "Ambient_Occlusion"},
            config_key="rma_map",
        )
        self.assertTrue(m.is_packed)

    def test_packed_map_without_channels_is_rejected(self):
        """Coverage-aware filtering needs the per-channel layout."""
        with self.assertRaisesRegex(ValueError, "channels"):
            MapType(
                name="RMA",
                aliases=["RoughMetalAO"],
                is_packed=True,
                replaces=["Roughness", "Metallic", "Ambient_Occlusion"],
                config_key="rma_map",
            )

    def test_channel_type_missing_from_replaces_is_rejected(self):
        """A carried type the map doesn't replace would double-wire its slot.

        The live catch: MSAO carried Smoothness in its alpha but didn't list it
        in `replaces`, so a loose Smoothness stayed wired beside the mask map.
        """
        with self.assertRaisesRegex(ValueError, "carries"):
            MapType(
                name="RMA",
                aliases=["RoughMetalAO"],
                is_packed=True,
                replaces=["Roughness", "Metallic"],  # AO carried but not replaced
                channels={"R": "Roughness", "G": "Metallic", "B": "Ambient_Occlusion"},
                config_key="rma_map",
            )

    def test_optional_channels_are_exempt_from_replaces_consistency(self):
        """A '?' filler channel (MSAO's Detail) needn't be in `replaces`."""
        m = MapType(
            name="RMA",
            aliases=["RoughMetalAO"],
            is_packed=True,
            replaces=["Roughness", "Metallic", "Ambient_Occlusion"],
            channels={
                "R": "Roughness",
                "G": "Metallic",
                "B": "Ambient_Occlusion",
                "A": "Detail_Mask?",
            },
            config_key="rma_map",
        )
        self.assertEqual(
            m.carried_types(), ["Roughness", "Metallic", "Ambient_Occlusion"]
        )
        self.assertIn("Detail_Mask", m.carried_types(include_optional=True))

    def test_unpacked_maps_are_exempt(self):
        """Loose maps carry nothing — the contract must not burden them."""
        m = MapType(name="Curvature", aliases=["Curv"], mode="L")
        self.assertFalse(m.is_packed)

    def test_every_registered_packed_map_satisfies_the_contract(self):
        """Sweep the live registry: is_packed or replaces → the full trio.

        __post_init__ guards construction; this guards against the contract
        being weakened there without this suite noticing.
        """
        for name, m in MapRegistry._maps.items():
            if not (m.is_packed or m.replaces):
                continue
            self.assertTrue(m.is_packed, f"'{name}' has replaces but not is_packed")
            self.assertTrue(m.replaces, f"packed map '{name}' declares no replaces")
            self.assertTrue(
                m.config_key, f"packed map '{name}' declares no config_key"
            )
            self.assertTrue(
                m.channels, f"packed map '{name}' declares no channels layout"
            )
            for carried in m.carried_types():
                self.assertIn(
                    carried,
                    m.replaces,
                    f"packed map '{name}' carries {carried} but doesn't replace it",
                )


if __name__ == "__main__":
    unittest.main()


class TestSharesWorkflow(unittest.TestCase):
    """``shares_workflow`` — the registry's own map-vs-target compatibility answer.

    Exists so a converter never hardcodes an engine name to ask "is this source
    map the right packing for what I'm writing?". ``pack_orm_texture`` uses it to
    warn when it is handed a mask map from another engine family.
    """

    def setUp(self):
        self.registry = MapRegistry.instance()

    def test_orm_family_maps_share_a_target(self):
        """ORM is trivially compatible with itself, and Albedo+Transparency
        genuinely declares glTF — which is why it passes through the base-colour
        slot unchanged."""
        self.assertIs(self.registry.shares_workflow("ORM", "ORM"), True)
        self.assertIs(
            self.registry.shares_workflow("Albedo_Transparency", "ORM"), True
        )

    def test_foreign_engine_packings_report_a_mismatch(self):
        """MSAO targets HDRP and Metallic_Smoothness targets URP; neither shares
        a target with ORM, so both are reported as mismatches."""
        self.assertIs(self.registry.shares_workflow("MSAO", "ORM"), False)
        self.assertIs(
            self.registry.shares_workflow("Metallic_Smoothness", "ORM"), False
        )

    def test_an_absent_declaration_is_not_a_mismatch(self):
        """``None``, not ``False``, when either side declares no workflows.

        MRAO ships with an empty list and every loose map does, so a caller that
        tested falsiness rather than ``is False`` would warn about all of them.
        """
        self.assertIsNone(self.registry.shares_workflow("MRAO", "ORM"))
        self.assertIsNone(self.registry.shares_workflow("Metallic", "ORM"))

    def test_an_unknown_map_type_is_not_a_mismatch(self):
        self.assertIsNone(self.registry.shares_workflow("NotAMapType", "ORM"))
        self.assertIsNone(self.registry.shares_workflow("ORM", "NotAMapType"))


class TestPackOrmWarnsOnForeignPacking(unittest.TestCase):
    """The warning ``pack_orm_texture`` emits for a handled-but-mismatched map."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _map(self, name, mode, colour):
        from PIL import Image

        path = os.path.join(self.tmp, name)
        Image.new(mode, (8, 8), colour).save(path)
        return path

    def test_msao_is_handled_but_reported(self):
        """Correct output AND a warning naming both target sets — silence would
        leave the source set mismatched forever while every push pays for the
        channel split and the 8-bit smoothness inversion."""
        msao = self._map("c_MSAO.png", "RGBA", (200, 60, 0, 100))

        with self.assertLogs(MapFactory.logger, level="WARNING") as captured:
            orm = MapFactory.pack_orm_texture(None, None, msao, save=False)
        self.assertEqual(orm.getpixel((0, 0)), (60, 155, 200))
        message = "\n".join(captured.output)
        self.assertIn("MSAO", message)
        self.assertIn("Unity HDRP", message)

    def test_an_orm_source_is_silent(self):
        """ORM in, ORM out — nothing to report, and the values round-trip."""
        orm_path = self._map("y_ORM.png", "RGB", (40, 80, 120))

        with self.assertLogs(MapFactory.logger, level="WARNING") as captured:
            MapFactory.logger.warning("sentinel")  # assertLogs needs >=1 record
            orm = MapFactory.pack_orm_texture(None, None, orm_path, save=False)
        self.assertEqual(orm.getpixel((0, 0)), (40, 80, 120))
        self.assertEqual(
            [line for line in captured.output if "sentinel" not in line],
            [],
            "an ORM source must not warn",
        )

    def test_a_map_declaring_no_workflow_is_silent(self):
        """MRAO decomposes correctly and says nothing: its empty workflow list
        is an absent declaration, not an incompatible one."""
        mrao = self._map("z_MRAO.png", "RGB", (200, 90, 60))

        with self.assertLogs(MapFactory.logger, level="WARNING") as captured:
            MapFactory.logger.warning("sentinel")
            orm = MapFactory.pack_orm_texture(None, None, mrao, save=False)
        # MRAO rgb layout is R=Metallic, G=Roughness, B=AO.
        self.assertEqual(orm.getpixel((0, 0)), (60, 90, 200))
        self.assertEqual(
            [line for line in captured.output if "sentinel" not in line], []
        )


class TestForeignPackings(unittest.TestCase):
    """``foreign_packings`` — the single mismatch predicate.

    Read by ``pack_orm_texture``'s per-map warning, ``set_glb_metallic_roughness``'s
    highlighted summary, and both DCCs' ``check_material_compatibility`` gate, so
    the definition of "wrong packing for this target" exists exactly once.
    """

    def test_reports_foreign_packings_with_their_map_type(self):
        found = MapFactory.foreign_packings(
            ["/t/a_MSAO.png", "/t/b_MetallicSmoothness.png"], target="ORM"
        )
        self.assertEqual(
            found, {"/t/a_MSAO.png": "MSAO", "/t/b_MetallicSmoothness.png": "Metallic_Smoothness"}
        )

    def test_appropriate_and_undeclared_maps_are_not_reported(self):
        """ORM and Albedo+Transparency declare glTF; MRAO and loose maps declare
        nothing, and an absent declaration is not a mismatch."""
        self.assertEqual(
            MapFactory.foreign_packings(
                [
                    "/t/a_ORM.png",
                    "/t/b_Albedo_Transparency.png",
                    "/t/c_MRAO.png",
                    "/t/d_Roughness.png",
                    "/t/e_Normal_OpenGL.png",
                ],
                target="ORM",
            ),
            {},
        )

    def test_non_paths_and_duplicates_are_tolerated(self):
        """Mixed lists of paths and in-memory images are the normal input, and a
        path repeated across slots must be reported once."""
        from PIL import Image

        found = MapFactory.foreign_packings(
            [None, "", Image.new("L", (2, 2)), "/t/a_MSAO.png", "/t/a_MSAO.png"]
        )
        self.assertEqual(found, {"/t/a_MSAO.png": "MSAO"})

    def test_workflow_mode_judges_by_declared_membership(self):
        """``workflow=`` — the exporter's form of the question. The user chose a
        registry workflow (a texture template); a packed source is foreign when
        it does not declare that workflow. Note the reversal HDRP produces:
        there ORM is the foreign one and MSAO is native."""
        self.assertEqual(
            MapFactory.foreign_packings(
                ["/t/a_MSAO.png", "/t/b_ORM.png", "/t/c_Albedo_Transparency.png"],
                workflow="glTF 2.0",
            ),
            {"/t/a_MSAO.png": "MSAO"},
        )
        self.assertEqual(
            MapFactory.foreign_packings(
                ["/t/a_MSAO.png", "/t/b_ORM.png"], workflow="Unity HDRP"
            ),
            {"/t/b_ORM.png": "ORM"},
        )

    def test_an_unknown_workflow_reports_nothing(self):
        """A stale persisted UI value must never become "every mask map is
        foreign" and block an export — it warns and reports nothing."""
        with self.assertLogs(MapFactory.logger, level="WARNING") as captured:
            found = MapFactory.foreign_packings(
                ["/t/a_MSAO.png"], workflow="NotAWorkflow"
            )
        self.assertEqual(found, {})
        self.assertIn("NotAWorkflow", "\n".join(captured.output))

    def test_loose_maps_are_never_reported(self):
        """Regression: only a PACKED map can be foreign to a target.

        A loose map's ``workflows`` answers a different question — which presets
        *emit* it — so ``Ambient_Occlusion`` (Standard only) and ``Emissive``
        share no workflow with ORM and a general form reported both as engine
        mismatches. That fired the exporter gate on almost every scene: the
        production room's sidecar reported 6 offenders, of which 3 were an
        ordinary AO map and two ordinary emissive maps.
        """
        self.assertEqual(
            MapFactory.foreign_packings(
                [
                    "/t/a_AO.png",
                    "/t/b_Emissive.png",
                    "/t/c_Roughness.png",
                    "/t/d_Metallic.png",
                    "/t/e_Base_Color.png",
                    "/t/f_Normal_OpenGL.png",
                ],
                target="ORM",
            ),
            {},
        )

    def test_target_is_honoured(self):
        """MSAO is native to HDRP, so against an HDRP-family packing it is fine —
        the predicate is about the TARGET, not about MSAO."""
        self.assertEqual(
            MapFactory.foreign_packings(["/t/a_MSAO.png"], target="MSAO"), {}
        )
