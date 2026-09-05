# !/usr/bin/python
# coding=utf-8
"""Regression tests for the MapFactory workflow handlers (Strategy pattern)."""

import ast
import importlib
import os
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock, patch

from pythontk import ImgUtils
from pythontk.core_utils.engines.textures.map_factory import (
    MapFactory,
    TextureProcessor,
    ConversionRegistry,
    BaseColorHandler,
    ORMMapHandler,
    MRAOMapHandler,
    MaskMapHandler,
)
from pythontk.core_utils.engines.textures.map_factory import handlers as _handlers_mod
from pythontk.core_utils.engines.textures.map_factory import (
    processor as _processor_mod,
)
from pythontk.core_utils.engines.textures.map_registry import MapRegistry


class TestBaseColorHandlerAlbedoTransparencyFailure(unittest.TestCase):
    """BaseColorHandler must honour the requested output slot even when the
    transparency pack fails, instead of silently degrading to a Base_Color map."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="handlers_albedo_trans_")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        # Real source images so save_map can persist a file.
        self.base_color = os.path.join(self.test_dir, "mat_BaseColor.png")
        self.opacity = os.path.join(self.test_dir, "mat_Opacity.png")
        ImgUtils.save_image(
            ImgUtils.create_image("RGB", (16, 16), (128, 128, 128)), self.base_color
        )
        ImgUtils.save_image(ImgUtils.create_image("L", (16, 16), 128), self.opacity)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_context(self):
        return TextureProcessor(
            inventory={"Base_Color": self.base_color, "Opacity": self.opacity},
            config={"albedo_transparency": True, "rename": True},
            output_dir=self.output_dir,
            base_name="mat",
            ext="png",
            conversion_registry=ConversionRegistry(),
            logger=MagicMock(),
        )

    def test_pack_failure_preserves_albedo_transparency_slot(self):
        """Regression: when pack_transparency_into_albedo raises (e.g. base-color
        and opacity textures have different resolutions), the handler must NOT
        fall through to the plain Base_Color path. It emits under the requested
        Albedo_Transparency slot so the failure is not silently mislabelled."""
        ctx = self._make_context()
        handler = BaseColorHandler()

        # Force the packing primitive to fail (mismatched resolutions etc.).
        mock_factory = MagicMock()
        mock_factory.pack_transparency_into_albedo.side_effect = ValueError(
            "images do not match in size"
        )
        with patch.object(_handlers_mod, "_factory", lambda: mock_factory):
            result = handler.process(ctx)

        # 0. The seam was actually reached. Without this the test passes just as
        #    happily when the patch misses and the real primitive succeeds.
        mock_factory.pack_transparency_into_albedo.assert_called_once()
        # 1. A map is still produced (not dropped entirely).
        self.assertIsNotNone(result, "handler dropped the map entirely on pack failure")
        base = os.path.basename(result)
        # 2. It is emitted under the requested Albedo_Transparency slot, NOT
        #    silently renamed to a Base_Color map.
        self.assertIn("Albedo_Transparency", base)
        self.assertNotIn("_Base_Color", base)
        # 3. The failed pack did not actually consume opacity, so a separate
        #    Opacity map still passes through to its own slot downstream.
        self.assertIn("Albedo_Transparency", ctx.used_maps)
        self.assertNotIn("Opacity", ctx.used_maps)


class TestORMHandlerExistingPassthrough(unittest.TestCase):
    """Regression: an ORM already in the inventory must pass through verbatim —
    mirroring MRAOMapHandler/MaskMapHandler. Without the passthrough the
    handler re-derives components via the conversion registry, and because the
    AO lookup runs *before* the ORM unpack caches its channels, the repacked
    output silently replaces the AO channel with white."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="handlers_orm_passthrough_")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        # Distinctive per-channel values: R=AO, G=Roughness, B=Metallic.
        self.orm_values = (100, 180, 30)
        self.orm_path = os.path.join(self.test_dir, "mat_ORM.png")
        ImgUtils.save_image(
            ImgUtils.create_image("RGB", (16, 16), self.orm_values), self.orm_path
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_existing_orm_passes_through_with_ao_intact(self):
        ctx = TextureProcessor(
            inventory={"ORM": self.orm_path},
            config={"orm_map": True, "rename": True},
            output_dir=self.output_dir,
            base_name="mat",
            ext="png",
            conversion_registry=MapFactory._conversion_registry,
            logger=MagicMock(),
        )
        handler = ORMMapHandler()
        self.assertTrue(handler.can_handle(ctx))

        result = handler.process(ctx)
        self.assertIsNotNone(result, "handler produced no ORM output")

        out = ImgUtils.ensure_image(result).convert("RGB")
        self.assertEqual(
            out.getpixel((8, 8)),
            self.orm_values,
            "existing ORM was not passed through verbatim (AO channel lost)",
        )


class TestORMHandlerFreshPack(unittest.TestCase):
    """The fresh-pack path delegates to ``MapFactory.pack_orm_texture`` — pin
    the channel order (AO->R, Roughness->G, Metallic->B) *through the handler*,
    so the delegation wiring and the packer cannot drift apart. The round-trip
    test in test_img.py covers the packer alone; this covers the strategy that
    ships it."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="handlers_orm_freshpack_")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        # Distinctive per-channel values so a swapped channel is unmistakable.
        self.channel_values = {
            "Ambient_Occlusion": 100,
            "Roughness": 180,
            "Metallic": 30,
        }
        self.inventory = {}
        for map_type, value in self.channel_values.items():
            path = os.path.join(self.test_dir, f"mat_{map_type}.png")
            ImgUtils.save_image(ImgUtils.create_image("L", (16, 16), value), path)
            self.inventory[map_type] = path

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_loose_maps_pack_in_orm_channel_order(self):
        ctx = TextureProcessor(
            inventory=dict(self.inventory),
            config={"orm_map": True, "rename": True},
            output_dir=self.output_dir,
            base_name="mat",
            ext="png",
            conversion_registry=MapFactory._conversion_registry,
            logger=MagicMock(),
        )
        handler = ORMMapHandler()
        self.assertTrue(handler.can_handle(ctx))

        result = handler.process(ctx)
        self.assertIsNotNone(result, "handler produced no ORM output")

        out = ImgUtils.ensure_image(result).convert("RGB")
        self.assertEqual(
            out.getpixel((8, 8)),
            (
                self.channel_values["Ambient_Occlusion"],
                self.channel_values["Roughness"],
                self.channel_values["Metallic"],
            ),
            "fresh pack lost the AO/Rough/Metal channel order",
        )


class TestMissingMapRule(unittest.TestCase):
    """The 'Missing Maps' rule decides what a packed map does when one of its
    source channels can't be resolved. Three settings, shared verbatim with the
    Map Packer panel: skip (never write an incomplete map), multi (write it once
    2+ channels resolved), force (always write). The legacy ``force_packed_maps``
    bool must keep resolving to ``force`` — configs and scripts still set it."""

    def setUp(self):
        import pythontk as ptk

        self._artifacts = ptk.TempArtifacts(
            "handlers_missing_map_rule", policy="scoped"
        )
        self.test_dir = self._artifacts.dir_path()
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        self._artifacts.cleanup()

    def _map(self, map_type, value):
        path = os.path.join(self.test_dir, f"mat_{map_type}.png")
        ImgUtils.save_image(ImgUtils.create_image("L", (16, 16), value), path)
        return path

    def _result(self, handler, inventory, **config):
        ctx = TextureProcessor(
            inventory=inventory,
            config={"orm_map": True, "mask_map": True, "rename": True, **config},
            output_dir=self.output_dir,
            base_name="mat",
            ext="png",
            conversion_registry=MapFactory._conversion_registry,
            logger=MagicMock(),
        )
        return handler().process(ctx)

    def _orm_result(self, inventory, **config):
        return self._result(ORMMapHandler, inventory, **config)

    def test_rule_resolution_and_legacy_alias(self):
        for config, expected in (
            ({}, MapRegistry.MISSING_SKIP),
            ({"missing_map_rule": "multi"}, MapRegistry.MISSING_MULTI),
            ({"missing_map_rule": "FORCE"}, MapRegistry.MISSING_FORCE),
            ({"missing_map_rule": "nonsense"}, MapRegistry.MISSING_SKIP),
            ({"force_packed_maps": True}, MapRegistry.MISSING_FORCE),
            # An explicit rule outranks the legacy bool.
            (
                {"force_packed_maps": True, "missing_map_rule": "skip"},
                MapRegistry.MISSING_SKIP,
            ),
        ):
            with self.subTest(config=config):
                self.assertEqual(MapRegistry.resolve_missing_map_rule(config), expected)

    def test_multi_sits_between_skip_and_force(self):
        """Two of three channels resolved: only 'skip' refuses to pack."""
        inventory = {
            "Ambient_Occlusion": self._map("Ambient_Occlusion", 100),
            "Roughness": self._map("Roughness", 180),
        }  # metallic absent -> would fill black
        self.assertIsNone(
            self._orm_result(dict(inventory)),
            "default (skip) wrote an ORM with an unresolved metallic channel",
        )
        for rule in (MapRegistry.MISSING_MULTI, MapRegistry.MISSING_FORCE):
            with self.subTest(rule=rule):
                self.assertIsNotNone(
                    self._orm_result(dict(inventory), missing_map_rule=rule),
                    f"'{rule}' refused a set with 2 of 3 channels resolved",
                )

    def test_single_channel_packs_only_under_force(self):
        """One of three resolved is a map wearing a packed name — 'multi' skips
        it; only 'force' writes it."""
        inventory = {"Ambient_Occlusion": self._map("Ambient_Occlusion", 100)}
        self.assertIsNone(
            self._orm_result(
                dict(inventory), missing_map_rule=MapRegistry.MISSING_MULTI
            ),
            "'multi' packed a set with a single resolved channel",
        )
        self.assertIsNotNone(
            self._orm_result(
                dict(inventory), missing_map_rule=MapRegistry.MISSING_FORCE
            ),
            "'force' refused to pack",
        )
        # Legacy spelling of the same intent.
        self.assertIsNotNone(
            self._orm_result(dict(inventory), force_packed_maps=True),
            "the legacy force_packed_maps bool no longer forces a pack",
        )

    def test_every_packing_answers_a_lone_channel_the_same_way(self):
        """ORM/MRAO refused any single-channel set while the Mask Map wrote one
        under EVERY rule — a metallic-only MSAO fills white smoothness, i.e.
        mirror-smooth, which is exactly what the 'skip' rule exists to prevent.
        All three packings must agree: a lone channel needs 'Pack Anyway'."""
        singles = {
            "Metallic": self._map("Metallic", 30),
            "Ambient_Occlusion": self._map("Ambient_Occlusion", 100),
            "Smoothness": self._map("Smoothness", 200),
        }
        for handler in (ORMMapHandler, MRAOMapHandler, MaskMapHandler):
            for map_type, path in singles.items():
                with self.subTest(handler=handler.__name__, only=map_type):
                    inventory = {map_type: path}
                    for rule in (
                        MapRegistry.MISSING_SKIP,
                        MapRegistry.MISSING_MULTI,
                    ):
                        self.assertIsNone(
                            self._result(
                                handler, dict(inventory), missing_map_rule=rule
                            ),
                            f"'{rule}' packed a set with only {map_type}",
                        )
                    self.assertIsNotNone(
                        self._result(
                            handler,
                            dict(inventory),
                            missing_map_rule=MapRegistry.MISSING_FORCE,
                        ),
                        f"'force' refused a set with only {map_type}",
                    )

    def test_mask_map_requires_smoothness_not_just_two_of_three(self):
        """Counting resolved channels hides the ONE unsafe 2-of-3 combination.

        Exactly one of the three pairs is dangerous, which is why a count
        cannot express the rule: absent smoothness fills WHITE (every surface
        mirror-smooth), while absent AO fills white (no occlusion) and absent
        metallic fills black (dielectric) are both neutral. So the gate has to
        name the channel, exactly as ORM/MRAO name theirs, rather than counting
        how many happened to resolve.

        A flat smoothness channel is what a legitimate mirror material looks
        like, so this ships undetected on review -- the reason it is asserted
        here rather than left to inspection.
        """
        metallic = self._map("Metallic", 30)
        ao = self._map("Ambient_Occlusion", 100)
        smoothness = self._map("Smoothness", 200)

        # The unsafe pair: two channels resolve, but the one that fills
        # non-neutrally is the missing one. Only the DEFAULT rule refuses it.
        unsafe = {"Metallic": metallic, "Ambient_Occlusion": ao}
        self.assertIsNone(
            self._result(
                MaskMapHandler,
                dict(unsafe),
                missing_map_rule=MapRegistry.MISSING_SKIP,
            ),
            "the default rule packed a Mask Map whose smoothness is a white "
            "fill -- a mirror-smooth surface the rule exists to prevent",
        )
        # multi/force are opt-ins, not oversights: allow_incomplete_pack is a
        # MINIMUM measured against what resolved, so 'multi' (minimum 2) is the
        # caller saying two channels is enough. Pinned so a later tightening of
        # the smoothness gate cannot quietly swallow the rule vocabulary.
        for rule in (MapRegistry.MISSING_MULTI, MapRegistry.MISSING_FORCE):
            with self.subTest(rule=rule):
                self.assertIsNotNone(
                    self._result(MaskMapHandler, dict(unsafe), missing_map_rule=rule),
                    f"'{rule}' is an explicit opt-in and must still ship it",
                )

        # ...and the fix must not over-tighten: the other two pairs are benign
        # (white AO = unoccluded, black metallic = dielectric) and still pack
        # under the DEFAULT rule.
        for label, inventory in (
            ("no AO", {"Metallic": metallic, "Smoothness": smoothness}),
            ("no metallic", {"Ambient_Occlusion": ao, "Smoothness": smoothness}),
        ):
            with self.subTest(missing=label):
                self.assertIsNotNone(
                    self._result(MaskMapHandler, dict(inventory)),
                    f"the default rule refused a benign pair ({label})",
                )


class TestNoImportTimeInjection(unittest.TestCase):
    """The package ``__init__`` used to assign ``MapFactory`` onto its own
    submodules at import time -- the package's only cross-module attribute
    assignment, and a direct violation of the no-side-effects-on-import rule.

    The submodules declared ``MapFactory = None`` and relied on that injection,
    so reloading one on its own put the ``None`` back and every primitive call
    raised ``AttributeError: 'NoneType' object has no attribute ...``. A full
    ``ModuleReloader`` run survived it only because the topological sort happens
    to re-run the parent after its children -- correctness resting on reload
    order. Each module now resolves the name in ``_factory()`` at call time.
    """

    SUBMODULES = (_handlers_mod, _processor_mod)

    def test_reloading_a_submodule_alone_keeps_map_factory_resolvable(self):
        for module in self.SUBMODULES:
            with self.subTest(module=module.__name__):
                reloaded = importlib.reload(module)
                self.assertIs(
                    reloaded._factory(),
                    MapFactory,
                    f"{module.__name__} lost MapFactory after a lone reload",
                )

    def test_the_package_init_assigns_nothing_onto_its_submodules(self):
        """Guards the rule itself, not just this symptom: no module-level
        assignment to an attribute of an imported module."""
        init = os.path.join(os.path.dirname(_handlers_mod.__file__), "__init__.py")
        with open(init, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        offenders = [
            f"{ast.unparse(t)} (line {node.lineno})"
            for node in tree.body
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Attribute)
        ]
        self.assertEqual(offenders, [], f"import-time side effect: {offenders}")

    def test_the_retired_seam_name_cannot_come_back_silently(self):
        """``MapFactory`` as a module global was the old DI seam, and a stale
        ``patch.object(module, "MapFactory", ...)`` would now guard nothing.
        It must stay absent so such a patch raises instead."""
        for module in self.SUBMODULES:
            with self.subTest(module=module.__name__):
                self.assertFalse(hasattr(module, "MapFactory"))
                with self.assertRaises(AttributeError):
                    with patch.object(module, "MapFactory", MagicMock()):
                        pass

    def test_factory_is_the_di_seam(self):
        """Patching ``_factory`` reaches the handlers, which is what the
        albedo-transparency regression above relies on."""
        stub = MagicMock()
        with patch.object(_handlers_mod, "_factory", lambda: stub):
            self.assertIs(_handlers_mod._factory(), stub)
        self.assertIs(_handlers_mod._factory(), MapFactory)


if __name__ == "__main__":
    unittest.main()
