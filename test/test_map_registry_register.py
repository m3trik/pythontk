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
