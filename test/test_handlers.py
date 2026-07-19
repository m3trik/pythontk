# !/usr/bin/python
# coding=utf-8
"""Regression tests for the MapFactory workflow handlers (Strategy pattern)."""
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
)
from pythontk.core_utils.engines.textures.map_factory import handlers as _handlers_mod


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
        ImgUtils.save_image(
            ImgUtils.create_image("L", (16, 16), 128), self.opacity
        )

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
        with patch.object(_handlers_mod, "MapFactory", mock_factory):
            result = handler.process(ctx)

        # 1. A map is still produced (not dropped entirely).
        self.assertIsNotNone(
            result, "handler dropped the map entirely on pack failure"
        )
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


if __name__ == "__main__":
    unittest.main()
