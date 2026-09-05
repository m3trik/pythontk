#!/usr/bin/python
# coding=utf-8
"""
Refactored tests for MapFactory using the public API (Strategy Pattern).
"""

import os
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock, patch
import pythontk as ptk
from pythontk import ImgUtils
from pythontk.core_utils.engines.textures.map_factory import (
    MapFactory,
    TextureProcessor,
    ConversionRegistry,
)


class TestMapFactoryRefactored(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="texture_factory_test_")
        cls.test_files_dir = os.path.join(cls.test_dir, "textures")
        os.makedirs(cls.test_files_dir, exist_ok=True)
        cls._create_test_textures()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    @classmethod
    def _create_test_textures(cls):
        cls.test_textures = {
            "Base_Color": "test_material_BaseColor.png",
            "Metallic": "test_material_Metallic.png",
            "Roughness": "test_material_Roughness.png",
            "Normal_OpenGL": "test_material_Normal_OpenGL.png",
            "Normal_DirectX": "test_material_Normal_DirectX.png",
            "Ambient_Occlusion": "test_material_AO.png",
            "Opacity": "test_material_Opacity.png",
            "Height": "test_material_Height.png",
            "Emissive": "test_material_Emissive.png",
            "Smoothness": "test_material_Smoothness.png",
            "Specular": "test_material_Specular.png",
            "Glossiness": "test_material_Glossiness.png",
        }
        cls.texture_paths = []
        for map_type, filename in cls.test_textures.items():
            filepath = os.path.join(cls.test_files_dir, filename)
            if "Normal" in map_type:
                img = ImgUtils.create_image("RGB", (64, 64), (128, 128, 255))
            elif map_type in [
                "Metallic",
                "Roughness",
                "Smoothness",
                "AO",
                "Opacity",
                "Height",
            ]:
                img = ImgUtils.create_image("L", (64, 64), 128)
            else:
                img = ImgUtils.create_image("RGB", (64, 64), (128, 128, 128))
            ImgUtils.save_image(img, filepath)
            cls.texture_paths.append(filepath)

    def setUp(self):
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_prepare_maps_standard_pbr(self):
        """Test standard PBR workflow (separate maps)."""
        config = {"rename": True}
        # Only provide Standard PBR maps to avoid triggering implicit packed workflows
        subset = [
            p
            for p in self.texture_paths
            if any(
                x in p for x in ["BaseColor", "Metallic", "Roughness", "Normal_OpenGL"]
            )
        ]
        results = MapFactory.prepare_maps(
            subset, output_dir=self.output_dir, callback=lambda *args: None, **config
        )
        result_names = [os.path.basename(p) for p in results]

        self.assertTrue(any("Base_Color" in n for n in result_names))
        self.assertTrue(any("Metallic" in n for n in result_names))
        self.assertTrue(any("Roughness" in n for n in result_names))
        self.assertTrue(any("Normal_OpenGL" in n for n in result_names))

    def test_prepare_maps_unity_urp(self):
        """Test Unity URP workflow (Metallic+Smoothness)."""
        config = {"metallic_smoothness": True}
        results = MapFactory.prepare_maps(
            self.texture_paths,
            output_dir=self.output_dir,
            callback=lambda *args: None,
            **config,
        )
        result_names = [os.path.basename(p) for p in results]

        self.assertTrue(any("Metallic_Smoothness" in n for n in result_names))
        # Should NOT have separate Metallic or Roughness
        self.assertFalse(any("Roughness" in n for n in result_names))
        # Metallic might be present if packing failed, but here it should succeed.
        # Wait, Metallic_SmoothnessHandler consumes Metallic, Roughness, Smoothness.
        # So they should not appear in output if marked used.

    def test_prepare_maps_unity_hdrp(self):
        """Test Unity HDRP workflow (Mask Map/MSAO)."""
        config = {"mask_map": True}
        results = MapFactory.prepare_maps(
            self.texture_paths,
            output_dir=self.output_dir,
            callback=lambda *args: None,
            **config,
        )
        result_names = [os.path.basename(p) for p in results]

        self.assertTrue(any("MSAO" in n for n in result_names))
        self.assertFalse(any("Metallic" in n for n in result_names))
        self.assertFalse(any("Roughness" in n for n in result_names))
        # Check for AO specifically (avoid matching MSAO)
        self.assertFalse(
            any("_AO." in n or "_Ambient_Occlusion." in n for n in result_names)
        )

    def test_prepare_maps_unity_hdrp_rgb_layout(self):
        """MSAO workflow honours mask_map_layout='rgb' end-to-end."""
        config = {"mask_map": True, "mask_map_layout": "rgb"}
        results = MapFactory.prepare_maps(
            self.texture_paths,
            output_dir=self.output_dir,
            callback=lambda *args: None,
            **config,
        )
        result_names = [os.path.basename(p) for p in results]
        self.assertTrue(any("MSAO" in n for n in result_names))

        from PIL import Image

        msao_path = next(p for p in results if "MSAO" in os.path.basename(p))
        with Image.open(msao_path) as img:
            self.assertEqual(img.mode, "RGB")

    def test_prepare_maps_mrao(self):
        """Test MRAO packing workflow (default 3-channel layout)."""
        config = {"mrao_map": True}
        results = MapFactory.prepare_maps(
            self.texture_paths,
            output_dir=self.output_dir,
            callback=lambda *args: None,
            **config,
        )
        result_names = [os.path.basename(p) for p in results]

        self.assertTrue(any("MRAO" in n for n in result_names))
        # Check the produced file is RGB (3-channel default) and channels are M/R/AO.
        from PIL import Image

        mrao_path = next(p for p in results if "MRAO" in os.path.basename(p))
        with Image.open(mrao_path) as img:
            self.assertEqual(img.mode, "RGB")

    def test_prepare_maps_mrao_rgba_layout(self):
        """Test MRAO packing workflow with RGBA (MSAO-mirror) layout."""
        config = {"mrao_map": True, "mrao_layout": "rgba"}
        results = MapFactory.prepare_maps(
            self.texture_paths,
            output_dir=self.output_dir,
            callback=lambda *args: None,
            **config,
        )
        result_names = [os.path.basename(p) for p in results]
        self.assertTrue(any("MRAO" in n for n in result_names))

        from PIL import Image

        mrao_path = next(p for p in results if "MRAO" in os.path.basename(p))
        with Image.open(mrao_path) as img:
            self.assertEqual(img.mode, "RGBA")

    def test_prepare_maps_unreal_engine(self):
        """Test Unreal Engine workflow (ORM + DirectX Normal)."""
        config = {"orm_map": True, "normal_type": "DirectX"}
        results = MapFactory.prepare_maps(
            self.texture_paths,
            output_dir=self.output_dir,
            callback=lambda *args: None,
            **config,
        )
        result_names = [os.path.basename(p) for p in results]

        self.assertTrue(any("ORM" in n for n in result_names))
        self.assertTrue(any("Normal_DirectX" in n for n in result_names))
        self.assertFalse(any("Normal_OpenGL" in n for n in result_names))

    def test_prepare_base_color_with_packing(self):
        """Test packing transparency into Base Color."""
        config = {"albedo_transparency": True}
        # Only provide Base Color and Opacity
        subset = [p for p in self.texture_paths if "BaseColor" in p or "Opacity" in p]
        results = MapFactory.prepare_maps(
            subset, output_dir=self.output_dir, callback=lambda *args: None, **config
        )
        result_names = [os.path.basename(p) for p in results]

        self.assertTrue(any("Albedo_Transparency" in n for n in result_names))
        self.assertFalse(any("Base_Color" in n for n in result_names))

    def test_input_fallback_control(self):
        """Test disabling input fallbacks (e.g. Diffuse -> Base_Color)."""
        # Create a Diffuse map
        diffuse_path = os.path.join(self.test_files_dir, "test_material_Diffuse.png")
        img = ImgUtils.create_image("RGB", (64, 64), (128, 128, 128))
        ImgUtils.save_image(img, diffuse_path)

        # Case 1: Enabled (Default)
        results = MapFactory.prepare_maps(
            [diffuse_path],
            output_dir=self.output_dir,
            use_input_fallbacks=True,
            rename=True,
            dry_run=False,
            callback=lambda *args: None,
        )
        result_names = [os.path.basename(p) for p in results]
        self.assertTrue(
            any("Base_Color" in n for n in result_names),
            "Should resolve Base_Color from Diffuse when enabled",
        )

        # Case 2: Disabled
        # Clear output
        shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

        results = MapFactory.prepare_maps(
            [diffuse_path],
            output_dir=self.output_dir,
            use_input_fallbacks=False,
            rename=True,
            dry_run=False,
            callback=lambda *args: None,
        )
        result_names = [os.path.basename(p) for p in results]
        self.assertFalse(
            any("Base_Color" in n for n in result_names),
            "Should NOT resolve Base_Color from Diffuse when disabled",
        )
        self.assertTrue(
            any("Diffuse" in n for n in result_names),
            "Should pass through Diffuse when fallback disabled",
        )

    def test_output_fallback_control(self):
        """Test disabling output fallbacks (e.g. AO -> Mask)."""
        # Use existing AO map
        ao_path = next(p for p in self.texture_paths if "AO" in p)

        import logging

        logger = MapFactory.logger
        log_capture = []

        class ListHandler(logging.Handler):
            def emit(self, record):
                log_capture.append(record.getMessage())

        handler = ListHandler()
        logger.addHandler(handler)

        try:
            # Case 1: Enabled
            MapFactory.prepare_maps(
                [ao_path],
                output_dir=self.output_dir,
                mask_map=True,
                use_output_fallbacks=True,
                dry_run=False,
            )
            self.assertTrue(
                any("Outputting fallback map" in msg for msg in log_capture),
                "Should log fallback usage when enabled",
            )

            # Case 2: Disabled
            log_capture.clear()
            shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)

            MapFactory.prepare_maps(
                [ao_path],
                output_dir=self.output_dir,
                mask_map=True,
                use_output_fallbacks=False,
                dry_run=False,
            )
            self.assertFalse(
                any("Outputting fallback map" in msg for msg in log_capture),
                "Should NOT log fallback usage when disabled",
            )
        finally:
            logger.removeHandler(handler)

    def test_supplement_sets_from_dir_gap_fills(self):
        """_supplement_sets_from_dir pulls missing same-base-name siblings from disk."""
        base = MapFactory.get_base_texture_name(self.texture_paths[0])
        base_color = next(p for p in self.texture_paths if "BaseColor" in p)

        sets = {base: [base_color]}
        result = MapFactory._supplement_sets_from_dir(sets, self.test_files_dir)

        types = {MapFactory.resolve_map_type(p) for p in result[base]}
        self.assertIn("Base_Color", types)
        # Siblings sitting on disk but absent from the provided list are added.
        self.assertIn("Roughness", types)
        self.assertIn("Normal_OpenGL", types)
        # The already-present Base Color is not duplicated.
        self.assertEqual(
            sum(1 for p in result[base] if "BaseColor" in os.path.basename(p)), 1
        )

    def test_supplement_finds_siblings_in_a_subdirectory(self):
        """Gap-fill reaches maps in a SUBDIRECTORY of the scan root.

        Regression: a Maya project's ``sourceimages`` is routinely organized as
        one folder per asset, and the tool that opts into discovery hands over
        the ``sourceImages`` rule -- the ROOT. A root-only scan found nothing
        in that layout, so a material whose Metallic/Roughness sat one level
        down kept whatever was already wired (a stale packed map) instead of
        converting to the preset's loose maps.
        """
        root = os.path.join(self.test_dir, "discover_nested")
        nested = os.path.join(root, "gadget")
        os.makedirs(nested, exist_ok=True)
        try:
            for fn, mode, color in [
                ("gadget_BaseColor.png", "RGB", (128, 128, 128)),
                ("gadget_Roughness.png", "L", 128),
                ("gadget_Metallic.png", "L", 32),
            ]:
                ImgUtils.save_image(
                    ImgUtils.create_image(mode, (8, 8), color),
                    os.path.join(nested, fn),
                )

            base_color = os.path.join(nested, "gadget_BaseColor.png")
            sets = {MapFactory.get_base_texture_name(base_color): [base_color]}
            # The ROOT is what gets passed, not the folder the files live in.
            result = MapFactory._supplement_sets_from_dir(sets, root)

            types = {
                MapFactory.resolve_map_type(p) for p in next(iter(result.values()))
            }
            self.assertIn("Roughness", types)
            self.assertIn("Metallic", types)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_supplement_does_not_replace_present_type(self):
        """A connected map slot is never replaced by a same-type sibling on disk."""
        base = MapFactory.get_base_texture_name(self.texture_paths[0])
        # A "connected" Base Color living outside the scan directory.
        external = os.path.join(self.test_dir, "test_material_BaseColor.png")
        ImgUtils.save_image(ImgUtils.create_image("RGB", (8, 8), (1, 2, 3)), external)
        try:
            sets = {base: [external]}
            MapFactory._supplement_sets_from_dir(sets, self.test_files_dir)

            base_colors = [
                p for p in sets[base] if MapFactory.resolve_map_type(p) == "Base_Color"
            ]
            self.assertEqual(len(base_colors), 1)
            self.assertEqual(
                os.path.normcase(os.path.abspath(base_colors[0])),
                os.path.normcase(os.path.abspath(external)),
            )
        finally:
            if os.path.exists(external):
                os.remove(external)

    def test_prepare_maps_discover_dir_pulls_missing(self):
        """prepare_maps(discover_dir=...) processes siblings discovered on disk."""
        disc = os.path.join(self.test_dir, "discover")
        os.makedirs(disc, exist_ok=True)
        try:
            for fn, color in [
                ("widget_BaseColor.png", (128, 128, 128)),
                ("widget_Roughness.png", 128),
                ("widget_Normal_OpenGL.png", (128, 128, 255)),
            ]:
                mode = "L" if isinstance(color, int) else "RGB"
                ImgUtils.save_image(
                    ImgUtils.create_image(mode, (16, 16), color),
                    os.path.join(disc, fn),
                )

            base_color = os.path.join(disc, "widget_BaseColor.png")
            results = MapFactory.prepare_maps(
                [base_color],
                output_dir=self.output_dir,
                discover_dir=disc,
                group_by_set=False,
                rename=True,
                callback=lambda *args: None,
            )
            names = [os.path.basename(p) for p in results]
            self.assertTrue(any("Roughness" in n for n in names))
            self.assertTrue(any("Normal" in n for n in names))
        finally:
            shutil.rmtree(disc, ignore_errors=True)

    def test_discover_dir_does_not_mutate_input_list(self):
        """prepare_maps(discover_dir=...) must not mutate the caller's source list."""
        disc = os.path.join(self.test_dir, "discover_nomutate")
        os.makedirs(disc, exist_ok=True)
        try:
            for fn in ("gizmo_BaseColor.png", "gizmo_Roughness.png"):
                ImgUtils.save_image(
                    ImgUtils.create_image("RGB", (8, 8), (128, 128, 128)),
                    os.path.join(disc, fn),
                )
            source = [os.path.join(disc, "gizmo_BaseColor.png")]
            before = list(source)
            MapFactory.prepare_maps(
                source,
                output_dir=self.output_dir,
                discover_dir=disc,
                group_by_set=False,
                rename=True,
                # Empty ignored_patterns skips the filter rebind, so the input
                # list would be aliased into the working set without a guard.
                ignored_patterns=[],
                callback=lambda *a: None,
            )
            self.assertEqual(source, before, "source list was mutated by discovery")
        finally:
            shutil.rmtree(disc, ignore_errors=True)

    def test_prepare_maps_no_discover_dir_is_unchanged(self):
        """Without discover_dir, only the supplied files are processed."""
        base_color = next(p for p in self.texture_paths if "BaseColor" in p)
        results = MapFactory.prepare_maps(
            [base_color],
            output_dir=self.output_dir,
            group_by_set=False,
            rename=True,
            callback=lambda *args: None,
        )
        names = [os.path.basename(p) for p in results]
        self.assertFalse(any("Roughness" in n for n in names))
        self.assertFalse(any("Normal" in n for n in names))

    def test_height_passes_through_when_normal_present(self):
        """Regression: processing a Normal map must not consume a provided
        Height map — Height has its own engine slot (parallax/displacement)."""
        subset = [
            p for p in self.texture_paths if "Normal_OpenGL" in p or "Height" in p
        ]
        results = MapFactory.prepare_maps(
            subset, output_dir=self.output_dir, rename=True
        )
        names = [os.path.basename(p) for p in results]
        self.assertTrue(any("Normal" in n for n in names))
        self.assertTrue(
            any("Height" in n for n in names),
            "Height map was dropped by normal-map processing",
        )

    def test_opacity_passes_through_when_not_packed(self):
        """Regression: with albedo_transparency off, a separate Opacity map
        must pass through instead of being silently consumed."""
        subset = [p for p in self.texture_paths if "BaseColor" in p or "Opacity" in p]
        results = MapFactory.prepare_maps(
            subset, output_dir=self.output_dir, rename=True
        )
        names = [os.path.basename(p) for p in results]
        self.assertTrue(any("Base_Color" in n for n in names))
        self.assertTrue(
            any("Opacity" in n for n in names),
            "Opacity map was dropped by base-color processing",
        )

    def test_mask_map_alpha_defaults_white_without_smoothness(self):
        """Regression: with no smoothness/roughness source, the Mask Map's
        alpha channel must be the neutral white fill — not a copy of the
        metallic channel.

        Needs an explicit Pack Anyway now. This set (metallic + AO, no
        smoothness) is the one 2-of-3 combination whose missing channel fills
        NON-neutrally, so the default rule refuses it — a white alpha is a
        mirror-smooth surface. The leak this guards against is a different
        failure (metallic data reaching the alpha), and it stays pinned here
        by asking for the pack the caller would have to ask for.
        """
        from PIL import Image

        src_dir = os.path.join(self.test_dir, "mask_alpha_src")
        os.makedirs(src_dir, exist_ok=True)
        try:
            metallic = os.path.join(src_dir, "alphatest_Metallic.png")
            ao = os.path.join(src_dir, "alphatest_AO.png")
            ImgUtils.save_image(ImgUtils.create_image("L", (16, 16), 30), metallic)
            ImgUtils.save_image(ImgUtils.create_image("L", (16, 16), 200), ao)

            results = MapFactory.prepare_maps(
                [metallic, ao],
                output_dir=self.output_dir,
                mask_map=True,
                rename=True,
                missing_map_rule="force",
            )
            msao_path = next(
                (p for p in results if "MSAO" in os.path.basename(p)),
                None,
            )
            self.assertIsNotNone(msao_path, "Pack Anyway did not write a Mask Map")
            with Image.open(msao_path) as img:
                self.assertEqual(img.mode, "RGBA")
                alpha_min, alpha_max = img.getextrema()[3]
                self.assertEqual(
                    (alpha_min, alpha_max),
                    (255, 255),
                    "Mask Map alpha should be white fill when no smoothness exists",
                )
        finally:
            shutil.rmtree(src_dir, ignore_errors=True)


class TestMapFactoryExtended(unittest.TestCase):
    """Extended unit tests for MapFactory internal logic."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="texture_factory_extended_test_")
        cls.test_files_dir = os.path.join(cls.test_dir, "textures")
        os.makedirs(cls.test_files_dir, exist_ok=True)
        cls._create_test_textures()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    @classmethod
    def _create_test_textures(cls):
        cls.test_textures = {
            "Specular": "test_Specular.png",
            "Smoothness": "test_Smoothness.png",
            "Roughness": "test_Roughness.png",
            "Normal_DirectX": "test_Normal_DirectX.png",
            "Normal_OpenGL": "test_Normal_OpenGL.png",
            "Bump": "test_Bump.png",
            "Metallic": "test_Metallic.png",
            "AO": "test_AO.png",
            "Base_Color": "test_Base_Color.png",
            "Opacity": "test_Opacity.png",
        }
        cls.texture_paths = {}
        for map_type, filename in cls.test_textures.items():
            filepath = os.path.join(cls.test_files_dir, filename)
            if "Normal" in map_type:
                img = ImgUtils.create_image("RGB", (64, 64), (128, 128, 255))
            elif map_type in [
                "Metallic",
                "Roughness",
                "Smoothness",
                "AO",
                "Opacity",
                "Bump",
            ]:
                img = ImgUtils.create_image("L", (64, 64), 128)
            else:
                img = ImgUtils.create_image("RGB", (64, 64), (128, 128, 128))
            ImgUtils.save_image(img, filepath)
            cls.texture_paths[map_type] = filepath

    def setUp(self):
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Create a real context with mocked logger
        self.logger_mock = MagicMock()
        self.context = TextureProcessor(
            inventory={},
            config={},
            output_dir=self.output_dir,
            base_name="test",
            ext="png",
            conversion_registry=ConversionRegistry(),
            logger=self.logger_mock,
        )
        self.context.resolve_map = MagicMock(side_effect=self._mock_resolve_map)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def _mock_resolve_map(self, *args, **kwargs):
        # Simple mock that returns the first matching map from inventory
        for arg in args:
            if arg in self.context.inventory:
                return self.context.inventory[arg]
        return None

    def test_resolve_color_space(self):
        """resolve_color_space maps color textures to sRGB and data textures to Linear."""
        from pythontk import MapFactory

        self.assertEqual(MapFactory.resolve_color_space("rock_BaseColor.png"), "sRGB")
        self.assertEqual(MapFactory.resolve_color_space("rock_Emissive.png"), "sRGB")
        for data_map in (
            "rock_Normal.png",
            "rock_Roughness.png",
            "rock_Metallic.png",
            "rock_Height.png",
            "rock_AO.png",
        ):
            self.assertEqual(
                MapFactory.resolve_color_space(data_map), "Linear", msg=data_map
            )
        # Unresolved map type falls back to the supplied default.
        self.assertEqual(
            MapFactory.resolve_color_space("studio_environment.hdr", default=""), ""
        )

    def test_convert_specular_to_metallic(self):
        result = self.context.convert_specular_to_metallic(
            self.texture_paths["Specular"]
        )
        self.assertIsNotNone(result)
        self.context.logger.info.assert_called_with(
            "Created metallic from specular", extra={"preset": "highlight"}
        )

    def test_convert_smoothness_to_roughness(self):
        result = self.context.convert_smoothness_to_roughness(
            self.texture_paths["Smoothness"]
        )
        self.assertIsNotNone(result)
        self.context.logger.info.assert_called_with(
            "Converted smoothness to roughness", extra={"preset": "highlight"}
        )

    def test_convert_roughness_to_smoothness(self):
        result = self.context.convert_roughness_to_smoothness(
            self.texture_paths["Roughness"]
        )
        self.assertIsNotNone(result)
        self.context.logger.info.assert_called_with(
            "Converted roughness to smoothness", extra={"preset": "highlight"}
        )

    def test_convert_specular_to_roughness(self):
        result = self.context.convert_specular_to_roughness(
            self.texture_paths["Specular"]
        )
        self.assertIsNotNone(result)
        self.context.logger.info.assert_called_with(
            "Created roughness from specular", extra={"preset": "highlight"}
        )

    def test_convert_dx_to_gl(self):
        result = self.context.convert_dx_to_gl(self.texture_paths["Normal_DirectX"])
        self.assertIsNotNone(result)
        self.context.logger.info.assert_called_with(
            "Converted DirectX normal to OpenGL", extra={"preset": "highlight"}
        )

    def test_convert_gl_to_dx(self):
        result = self.context.convert_gl_to_dx(self.texture_paths["Normal_OpenGL"])
        self.assertIsNotNone(result)
        self.context.logger.info.assert_called_with(
            "Converted OpenGL normal to DirectX", extra={"preset": "highlight"}
        )

    def test_convert_bump_to_normal(self):
        result = self.context.convert_bump_to_normal(self.texture_paths["Bump"])
        self.assertIsNotNone(result)
        self.context.logger.info.assert_called_with(
            "Generated normal from bump/height", extra={"preset": "highlight"}
        )

    def test_extract_gloss_from_spec(self):
        # Create a specular map with alpha
        spec_alpha_path = os.path.join(self.test_files_dir, "test_SpecAlpha.png")
        img = ImgUtils.create_image("RGBA", (64, 64), (128, 128, 128, 200))
        ImgUtils.save_image(img, spec_alpha_path)

        result = self.context.extract_gloss_from_spec(spec_alpha_path)
        self.assertIsNotNone(result)
        self.context.logger.info.assert_called_with(
            "Extracted glossiness from specular", extra={"preset": "highlight"}
        )

    def test_create_orm_map(self):
        # Setup inventory
        self.context.inventory = {
            "Ambient_Occlusion": self.texture_paths["AO"],
            "Roughness": self.texture_paths["Roughness"],
            "Metallic": self.texture_paths["Metallic"],
        }

        result = self.context.create_orm_map(self.context.inventory)
        self.assertIsNotNone(result)
        self.context.logger.info.assert_called_with(
            "Created ORM map from components", extra={"preset": "highlight"}
        )

    def test_create_mask_map(self):
        # Setup inventory
        self.context.inventory = {
            "Metallic": self.texture_paths["Metallic"],
            "Ambient_Occlusion": self.texture_paths["AO"],
            "Smoothness": self.texture_paths["Smoothness"],
        }

        result = self.context.create_mask_map(self.context.inventory)
        self.assertIsNotNone(result)
        self.context.logger.info.assert_called_with(
            "Created Mask Map from components (layout=rgba)",
            extra={"preset": "highlight"},
        )

    def test_create_metallic_smoothness_map(self):
        # Setup inventory
        self.context.inventory = {
            "Metallic": self.texture_paths["Metallic"],
            "Smoothness": self.texture_paths["Smoothness"],
        }

        result = self.context.create_metallic_smoothness_map(self.context.inventory)
        self.assertIsNotNone(result)
        self.context.logger.info.assert_called_with(
            "Packed smoothness into metallic", extra={"preset": "highlight"}
        )

    def test_unpack_metallic_smoothness(self):
        # Create a packed map
        packed_path = os.path.join(self.test_files_dir, "test_MetSmooth.png")
        img = ImgUtils.create_image("RGBA", (64, 64), (128, 128, 128, 200))
        ImgUtils.save_image(img, packed_path)

        self.context.inventory = {}
        metallic = self.context.get_metallic_from_packed(packed_path)
        smoothness = self.context.get_smoothness_from_packed(packed_path)

        self.assertIsNotNone(metallic)
        self.assertIsNotNone(smoothness)
        # Unpacking caches both components in the inventory.
        self.assertIn("Metallic", self.context.inventory)
        self.assertIn("Smoothness", self.context.inventory)

    def test_unpack_msao(self):
        # Create a packed map
        packed_path = os.path.join(self.test_files_dir, "test_MSAO.png")
        img = ImgUtils.create_image("RGBA", (64, 64), (128, 128, 128, 200))
        ImgUtils.save_image(img, packed_path)

        self.context.inventory = {}

        metallic = self.context.get_metallic_from_msao(packed_path)
        ao = self.context.get_ao_from_msao(packed_path)
        smoothness = self.context.get_smoothness_from_msao(packed_path)

        self.assertIsNotNone(metallic)
        self.assertIsNotNone(ao)
        self.assertIsNotNone(smoothness)

    def test_unpack_mrao_rgb(self):
        """Unpack a 3-channel MRAO texture (auto-detect)."""
        packed_path = os.path.join(self.test_files_dir, "test_MRAO.png")
        img = ImgUtils.create_image("RGB", (64, 64), (200, 100, 150))
        ImgUtils.save_image(img, packed_path)

        self.context.inventory = {}

        metallic = self.context.get_metallic_from_mrao(packed_path)
        roughness = self.context.get_roughness_from_mrao(packed_path)
        ao = self.context.get_ao_from_mrao(packed_path)

        self.assertIsNotNone(metallic)
        self.assertIsNotNone(roughness)
        self.assertIsNotNone(ao)

    def test_unpack_mrao_rgba(self):
        """Unpack a 4-channel MRAO texture (MSAO-mirror layout, auto-detect)."""
        packed_path = os.path.join(self.test_files_dir, "test_MRAO_rgba.png")
        img = ImgUtils.create_image("RGBA", (64, 64), (200, 150, 0, 100))
        ImgUtils.save_image(img, packed_path)

        self.context.inventory = {}

        metallic = self.context.get_metallic_from_mrao(packed_path)
        roughness = self.context.get_roughness_from_mrao(packed_path)
        ao = self.context.get_ao_from_mrao(packed_path)

        self.assertIsNotNone(metallic)
        self.assertIsNotNone(roughness)
        self.assertIsNotNone(ao)

    def test_unpack_orm(self):
        # Create a packed map
        packed_path = os.path.join(self.test_files_dir, "test_ORM.png")
        img = ImgUtils.create_image("RGB", (64, 64), (128, 128, 128))
        ImgUtils.save_image(img, packed_path)

        self.context.inventory = {}

        ao = self.context.get_ao_from_orm(packed_path)
        roughness = self.context.get_roughness_from_orm(packed_path)
        metallic = self.context.get_metallic_from_orm(packed_path)

        self.assertIsNotNone(ao)
        self.assertIsNotNone(roughness)
        self.assertIsNotNone(metallic)

    def test_unpack_albedo_transparency(self):
        # Create a packed map
        packed_path = os.path.join(self.test_files_dir, "test_AlbedoTrans.png")
        img = ImgUtils.create_image("RGBA", (64, 64), (128, 128, 128, 200))
        ImgUtils.save_image(img, packed_path)

        self.context.inventory = {}

        base_color = self.context.get_base_color_from_albedo_transparency(packed_path)
        opacity = self.context.get_opacity_from_albedo_transparency(packed_path)

        self.assertIsNotNone(base_color)
        self.assertIsNotNone(opacity)


class TestMapFactoryEdgeCases(unittest.TestCase):
    """Edge case tests for MapFactory."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="texture_factory_edge_test_")
        cls.test_files_dir = os.path.join(cls.test_dir, "textures")
        os.makedirs(cls.test_files_dir, exist_ok=True)
        # Create dummy images
        cls.tex_path = os.path.join(cls.test_files_dir, "test.png")
        ImgUtils.save_image(ImgUtils.create_image("L", (32, 32), 128), cls.tex_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def setUp(self):
        self.context = TextureProcessor(
            inventory={},
            config={},
            output_dir=self.test_dir,
            base_name="test",
            ext="png",
            conversion_registry=ConversionRegistry(),
            logger=MagicMock(),
        )
        # Mock resolve_map to behave like the real one for simple inventory lookups
        self.context.resolve_map = MagicMock(side_effect=self._mock_resolve_map)

    def _mock_resolve_map(self, *args, **kwargs):
        for arg in args:
            if arg in self.context.inventory:
                return self.context.inventory[arg]
        return None

    def test_convert_methods_raise_on_none(self):
        """Test that conversion methods raise ValueError when input is None."""
        methods = [
            self.context.convert_specular_to_metallic,
            self.context.convert_smoothness_to_roughness,
            self.context.convert_roughness_to_smoothness,
            self.context.convert_specular_to_roughness,
            self.context.convert_dx_to_gl,
            self.context.convert_gl_to_dx,
            self.context.convert_bump_to_normal,
            self.context.extract_gloss_from_spec,
        ]
        for method in methods:
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(ValueError, "missing"):
                    method(None)

    def test_create_orm_missing_all(self):
        """Test ORM creation fails if all components are missing."""
        self.context.inventory = {}
        with self.assertRaisesRegex(ValueError, "Missing components"):
            self.context.create_orm_map({})

    def test_create_orm_partial(self):
        """Test ORM creation works with partial components (e.g. only AO)."""
        self.context.inventory = {"Ambient_Occlusion": self.tex_path}
        # Should not raise
        result = self.context.create_orm_map(self.context.inventory)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result)

    def test_create_mask_map_missing_all(self):
        """Test Mask Map creation fails if all components are missing."""
        self.context.inventory = {}
        with self.assertRaisesRegex(ValueError, "Missing components"):
            self.context.create_mask_map({})

    def test_create_metallic_smoothness_missing_one(self):
        """Test Metallic-Smoothness fails if one component is missing."""
        # Missing Smoothness
        self.context.inventory = {"Metallic": self.tex_path}
        with self.assertRaisesRegex(ValueError, "Missing components"):
            self.context.create_metallic_smoothness_map(self.context.inventory)

        # Missing Metallic
        self.context.inventory = {"Smoothness": self.tex_path}
        with self.assertRaisesRegex(ValueError, "Missing components"):
            self.context.create_metallic_smoothness_map(self.context.inventory)

    def test_unpack_methods_handle_missing_file(self):
        """Test unpack methods handle missing files gracefully (by raising or logging)."""
        with self.assertRaises(Exception):
            self.context.get_metallic_from_packed("nonexistent.png")


class TestTextureProcessorLogic(unittest.TestCase):
    """Tests for TextureProcessor logic."""

    def setUp(self):
        self.registry = ConversionRegistry()
        self.context = TextureProcessor(
            inventory={"A": "a.png", "B": "b.png"},
            config={},
            output_dir=".",
            base_name="test",
            ext="png",
            conversion_registry=self.registry,
        )

    def test_resolve_map_priority(self):
        # If I ask for A then B, and A exists, I get A
        self.assertEqual(self.context.resolve_map("A", "B"), "a.png")
        # If I ask for C then B, and C missing, I get B
        self.assertEqual(self.context.resolve_map("C", "B"), "b.png")
        # If I ask for C then D, both missing, I get None
        self.assertIsNone(self.context.resolve_map("C", "D"))

    def test_resolve_map_conversion(self):
        # Register conversion C -> D
        mock_converter = MagicMock(return_value="d_converted.png")
        self.registry.register("D", "C", mock_converter)

        self.context.inventory = {"C": "c.png"}
        # Ask for D. Should convert from C.
        result = self.context.resolve_map("D")
        self.assertEqual(result, "d_converted.png")
        mock_converter.assert_called()

    def test_resolve_map_skips_planted_none_inventory_entry(self):
        """Regression: unpack helpers can cache None for a missing channel
        (e.g. Smoothness from an alpha-less packed map). A None entry must
        read as absent — not short-circuit resolution as a direct match."""
        self.registry.register("D", "C", lambda inv, ctx: "d_from_c.png")

        self.context.inventory = {"D": None, "C": "c.png"}
        self.assertEqual(self.context.resolve_map("D"), "d_from_c.png")

    def test_resolve_map_skips_failed_conversion_result(self):
        """Regression: a converter returning None must not be cached into the
        inventory (poisoning later lookups) and must not shadow a
        lower-priority conversion that can succeed."""
        self.registry.register("D", "C", lambda inv, ctx: None, priority=10)
        self.registry.register("D", "B", lambda inv, ctx: "d_from_b.png", priority=5)

        self.context.inventory = {"C": "c.png", "B": "b.png"}
        result = self.context.resolve_map("D")

        self.assertEqual(result, "d_from_b.png")
        self.assertTrue(
            all(v is not None for v in self.context.inventory.values()),
            f"None cached into inventory: {self.context.inventory}",
        )

    def test_unpack_fills_gaps_without_clobbering_loose_maps(self):
        """Regression: asking a packed map for one channel replaced a REAL
        loose sibling with the extraction — e.g. get_ao_from_msao overwrote a
        provided loose Metallic file with the MSAO R channel."""
        import tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            msao_path = os.path.join(tmp, "asset_MSAO.png")
            Image.merge(
                "RGBA", [Image.new("L", (8, 8), v) for v in (30, 200, 0, 90)]
            ).save(msao_path)

            context = TextureProcessor(
                inventory={
                    "MSAO": msao_path,
                    "Metallic": "loose_metallic.png",  # provided file must win
                },
                config={},
                output_dir=tmp,
                base_name="asset",
                ext="png",
                conversion_registry=ConversionRegistry(),
            )
            context.unpack_msao(msao_path)

            self.assertEqual(
                context.inventory["Metallic"],
                "loose_metallic.png",
                "unpack replaced a provided loose map with its extraction",
            )
            # The genuinely missing channels were filled from the packed map.
            self.assertIsNotNone(context.inventory.get("Ambient_Occlusion"))
            self.assertIsNotNone(context.inventory.get("Smoothness"))

    def test_normal_from_bump_does_not_swallow_height(self):
        """Regression: generating a normal from Bump marked Height used too,
        so a real Height map (parallax/displacement slot) never passed
        through."""
        import tempfile
        from PIL import Image
        from pythontk.core_utils.engines.textures.map_factory.handlers import (
            NormalMapHandler,
        )

        with tempfile.TemporaryDirectory() as tmp:
            bump = os.path.join(tmp, "asset_Bump.png")
            height = os.path.join(tmp, "asset_Height.png")
            Image.new("L", (8, 8), 128).save(bump)
            Image.new("L", (8, 8), 64).save(height)

            context = TextureProcessor(
                inventory={"Bump": bump, "Height": height},
                config={"normal_type": "OpenGL", "dry_run": True},
                output_dir=tmp,
                base_name="asset",
                ext="png",
                conversion_registry=MapFactory._conversion_registry,
            )
            result = NormalMapHandler().process(context)

            self.assertIsNotNone(result, "normal not generated from bump")
            self.assertIn("Bump", context.used_maps)
            self.assertNotIn(
                "Height",
                context.used_maps,
                "Height swallowed by a Bump-sourced normal",
            )

    def test_build_map_inventory_prefers_specific_basename(self):
        """Specificity is judged on the filename, not the full path — a longer
        directory must not decide which same-type file wins."""
        generic_in_long_dir = "/a/very/long/directory/name/here/x_AO.png"
        specific_in_short_dir = "/d/x_Mixed_AO.png"
        inventory = MapFactory._build_map_inventory(
            [generic_in_long_dir, specific_in_short_dir]
        )
        self.assertEqual(
            inventory.get("Ambient_Occlusion"),
            specific_in_short_dir,
            "full-path length outweighed filename specificity",
        )

    def test_save_map_dry_run_without_logger(self):
        """Regression: save_map's dry-run path crashed when the processor was
        constructed without a logger (a public, supported configuration)."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            context = TextureProcessor(
                inventory={},
                config={"dry_run": True},
                output_dir=tmp,
                base_name="test",
                ext="png",
                conversion_registry=ConversionRegistry(),
                logger=None,
            )
            img = ImgUtils.create_image("L", (8, 8), 128)
            path = context.save_map(img, "Roughness")
            self.assertTrue(path.endswith("_Roughness.png"))

    def test_packed_alpha_map_never_lands_in_a_jpg_container(self):
        """A JPEG cannot carry alpha, so a packed map whose alpha IS a
        material input must escalate to PNG.

        ``Metallic_Smoothness`` declares ``mode="RGBA"``, ``is_packed=True``
        and ``channels={"RGB": "Metallic", "A": "Smoothness"}`` -- URP reads
        smoothness out of that alpha. The escalation was driven by a
        hand-written five-name list that omitted it, so choosing JPG wrote a
        .jpg that reopens RGB and the smoothness was simply gone.
        ``ImgUtils.dropped_channels`` names this exact failure in its own
        docstring. ``MaskMap`` is here as an alias that must keep resolving,
        and ``MSAO`` / ``Albedo_Transparency`` guard the types the old list
        already covered.

        ``Emissive_Mask`` is pinned deliberately: it declares ``mode="RGBA"``
        without any packed-channel semantics, so the registry-derived rule
        escalates it too. That is a SECOND behaviour change beyond the
        reported defect and it is correct -- JPEG drops its alpha the same
        way -- but it is listed here so it stays intentional rather than
        emergent. Measured: these two are the only types the derived rule
        adds over the old hand-written list.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            context = TextureProcessor(
                inventory={},
                config={"dry_run": True},
                output_dir=tmp,
                base_name="test",
                ext="jpg",
                conversion_registry=ConversionRegistry(),
                logger=None,
            )
            img = ImgUtils.create_image("RGBA", (8, 8), (128, 128, 128, 255))
            for key in (
                "Metallic_Smoothness",
                "Emissive_Mask",
                "Albedo_Transparency",
                "MSAO",
                "MaskMap",
            ):
                with self.subTest(map_type=key):
                    path = context.save_map(img, key)
                    self.assertTrue(
                        path.lower().endswith(".png"),
                        f"{key} was written to a container that drops its "
                        f"alpha: {path}",
                    )

    def test_normal_map_orientation_convention(self):
        """Regression: convert_bump_to_normal produced DirectX orientation
        under the 'opengl' label (green = -row-derivative instead of +),
        and detect_normal_map_format had its correlation sign flipped to
        match. Pin the convention: OpenGL = green bright on TOP edges of
        raised detail; detector agrees with the generator AND with a
        labeled real-world map."""
        import numpy as np
        from PIL import Image as PILImage

        # Hemisphere bump: top flank faces up -> OpenGL green > 128 on top.
        n = 256
        yy, xx = np.mgrid[0:n, 0:n]
        r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (n / 2)
        h = np.clip(1 - r**2, 0, 1)
        bump = PILImage.fromarray((h * 255).astype(np.uint8), "L")

        nm = MapFactory.convert_bump_to_normal(
            bump,
            output_format="opengl",
            save=False,
            smooth_filter=False,
            intensity=30.0,
        )
        arr = np.array(nm).astype(float)
        g_top = arr[: n // 2 - 10, :, 1].mean()
        g_bottom = arr[n // 2 + 10 :, :, 1].mean()
        self.assertGreater(g_top, g_bottom, "OpenGL green must face up")

        # Generator and detector must agree.
        self.assertEqual(MapFactory.detect_normal_map_format(nm), "OpenGL")
        nm_dx = MapFactory.convert_bump_to_normal(
            bump,
            output_format="directx",
            save=False,
            smooth_filter=False,
            intensity=30.0,
        )
        self.assertEqual(MapFactory.detect_normal_map_format(nm_dx), "DirectX")

        # Real-world labeled asset. The assertion is about the SIGN (the bug was
        # a sign flip that read every real OpenGL map as DirectX-leaning), so it
        # probes below the conservative default threshold. It used to need 0.05
        # -- noise level, per the parameter's own docstring -- because the
        # internal thumbnail low-passed away the detail the statistic reads;
        # native-resolution escalation doubled the measured correlation
        # (-0.094 -> -0.189), so 0.15 is now clear of the noise floor. This map
        # still abstains at the 0.25 default, correctly: shallow relief over a
        # large neutral field is genuinely weak evidence.
        asset = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_assets",
            "imgtk_test",
            "im_Normal_OpenGL.png",
        )
        # NOT guarded by os.path.exists: this is the only reference to
        # test/test_assets/ anywhere in the suite, and behind that guard the
        # test reported green with the fixture deleted -- a zero-sample pass
        # on the one assertion that reads a real bake rather than a generated
        # hemisphere. A missing fixture must fail loudly.
        self.assertTrue(os.path.exists(asset), f"missing test fixture: {asset}")
        self.assertEqual(
            MapFactory.detect_normal_map_format(asset, threshold=0.15),
            "OpenGL",
        )

    def test_fine_detail_survives_the_downsample_shortcut(self):
        """A low-amplitude, high-frequency map must not read as indeterminate.

        The detector thumbnails to 512 before correlating, which is a low-pass
        filter over exactly the gradients the integrability statistic measures.
        On maps whose relief is fine and shallow that erases the signal:
        measured on two real OpenGL bakes, r fell from -0.368 to -0.105
        (a 2048 production map) and from -0.19 to -0.09 (the 4096 asset above),
        both dropping under the 0.25 threshold and returning None -- a
        confident, correct answer downgraded to "don't know" by an
        optimization. The shortcut stays for the common case; an inconclusive
        result now re-reads at native resolution before giving up.
        """
        import numpy as np
        from PIL import Image as PILImage

        # Fine, shallow relief on a 2048 map: detail at a spatial frequency the
        # 4x reduction cannot represent. Unambiguous at native resolution
        # (r = -0.999); at 512 it averages to r = +0.012 -- indeterminate, and
        # on the wrong side of zero.
        n = 2048
        yy, xx = np.mgrid[0:n, 0:n]
        h = (np.sin(xx * 1.4) * np.cos(yy * 1.1)) * 0.5 + 0.5
        bump = PILImage.fromarray((h * 255).astype(np.uint8), "L")
        nm = MapFactory.convert_bump_to_normal(
            bump,
            output_format="opengl",
            save=False,
            smooth_filter=False,
            intensity=2.0,
        )
        self.assertEqual(
            MapFactory.detect_normal_map_format(nm),
            "OpenGL",
            "fine detail was averaged away by the 512px shortcut",
        )

    def test_bump_to_normal_conversion_reads_registered_source(self):
        """Regression: the Bump/Height->Normal registration loop late-bound
        its loop variable, so the converter registered for 'Bump' read
        inv['Height'] — KeyError when only a Bump map exists."""
        registry = ConversionRegistry()
        MapFactory.register_conversions(registry)

        bump_only = [
            c
            for c in registry._conversions["Normal_OpenGL"]
            if c.source_types == ["Bump"]
        ]
        self.assertEqual(len(bump_only), 1)

        ctx = MagicMock()
        ctx.convert_bump_to_normal.return_value = "normal.png"
        # Inventory contains ONLY Bump — must not require 'Height'.
        result = bump_only[0].converter({"Bump": "bump.png"}, ctx)
        self.assertEqual(result, "normal.png")
        ctx.convert_bump_to_normal.assert_called_once_with("bump.png")


class TestMapFactoryImageInputRegressions(unittest.TestCase):
    """Regression tests for caller-supplied Image / path handling in MapFactory."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="map_factory_img_regress_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_detect_normal_map_format_does_not_mutate_input_image(self):
        """Regression: detect_normal_map_format() thumbnailed a caller-supplied
        RGB Image in place (ensure_image returns the same object, the mode
        already matched so no copy was made), silently downsizing it to 512px.
        It must operate on its own copy and leave the caller's Image untouched."""
        from PIL import Image

        img = Image.new("RGB", (1024, 768))
        original_size = img.size
        MapFactory.detect_normal_map_format(img)
        self.assertEqual(img.size, original_size)

    def test_convert_spec_gloss_write_files_rejects_image_specular(self):
        """Regression: convert_spec_gloss_to_pbr(write_files=True) with a PIL
        Image specular_map raised an obscure TypeError from resolve_texture_filename's
        assert_pathlike. It must raise a clear ValueError at the write boundary."""
        from PIL import Image

        spec = Image.new("RGB", (8, 8), (128, 128, 128))
        gloss = Image.new("L", (8, 8), 128)
        with self.assertRaises(ValueError):
            MapFactory.convert_spec_gloss_to_pbr(
                specular_map=spec,
                glossiness_map=gloss,
                write_files=True,
            )

    def test_get_converted_map_normal_branch_accepts_path(self):
        """Regression: get_converted_map's `available` values are source file
        paths (its docstring falsely claimed images). The Normal_OpenGL<->DirectX
        branches call convert_normal_map_format, which requires a path; passing a
        path per the corrected contract must return a converted Image."""
        from PIL import Image

        normal_gl = os.path.join(self.test_dir, "mat_Normal_OpenGL.png")
        ImgUtils.save_image(
            ImgUtils.create_image("RGB", (8, 8), (128, 128, 255)), normal_gl
        )
        result = MapFactory.get_converted_map(
            "Normal_DirectX", {"Normal_OpenGL": normal_gl}
        )
        self.assertIsInstance(result, Image.Image)


class TestRegisterHandlerIdempotent(unittest.TestCase):
    """register_handler must replace (not duplicate) a handler re-registered
    after a module reload, where the class object identity changes but the
    module+qualname does not."""

    def setUp(self):
        self._saved_handlers = list(MapFactory._workflow_handlers)

    def tearDown(self):
        MapFactory._workflow_handlers[:] = self._saved_handlers

    def _make_handler_class(self):
        from pythontk.core_utils.engines.textures.map_factory import WorkflowHandler

        class ProbeHandler(WorkflowHandler):
            def can_handle(self, context):
                return False

            def process(self, context):
                return None

            def get_consumed_types(self):
                return []

        # Same module+qualname each call, new class object — a reload stand-in.
        ProbeHandler.__module__ = "test_map_factory_probe"
        ProbeHandler.__qualname__ = "ProbeHandler"
        return ProbeHandler

    def test_reload_reregistration_replaces_in_place(self):
        first = self._make_handler_class()
        MapFactory.register_handler(first)
        count_after_first = len(MapFactory._workflow_handlers)
        position = MapFactory._workflow_handlers.index(first)

        reloaded = self._make_handler_class()
        self.assertIsNot(reloaded, first)
        MapFactory.register_handler(reloaded)

        self.assertEqual(len(MapFactory._workflow_handlers), count_after_first)
        self.assertIs(MapFactory._workflow_handlers[position], reloaded)
        self.assertNotIn(first, MapFactory._workflow_handlers)


class TestArchiveSupersededOriginals(unittest.TestCase):
    """``old_files_folder`` retires the inputs the output set replaced.

    Canonicalizing an aliased filename (``_BaseMap`` -> ``_Base_Color``) writes
    a COPY and leaves the original behind, so a folder grows a duplicate on
    every run. The archive folder is the opt-in cleanup for that; before this
    it was a config key nothing in the ``prepare_maps`` path ever read.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="map_factory_archive_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _texture(self, name, size=(16, 16), color=(128, 128, 128)):
        from PIL import Image

        path = os.path.join(self.test_dir, name)
        Image.new("RGB", size, color).save(path)
        return path

    def _names(self):
        return sorted(
            f
            for f in os.listdir(self.test_dir)
            if os.path.isfile(os.path.join(self.test_dir, f))
        )

    def test_aliased_original_is_archived_not_duplicated(self):
        """``_BaseMap`` canonicalizes to ``_Base_Color``; the alias must not linger."""
        source = self._texture("mat_BaseMap.png")
        self._texture("mat_Roughness.png")

        MapFactory.prepare_maps(
            [source, os.path.join(self.test_dir, "mat_Roughness.png")],
            group_by_set=False,
            rename=True,
            old_files_folder="_originals",
        )

        remaining = self._names()
        self.assertIn("mat_Base_Color.png", remaining)
        self.assertNotIn(
            "mat_BaseMap.png",
            remaining,
            "the superseded alias was left beside its canonical copy",
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.test_dir, "_originals", "mat_BaseMap.png")
            ),
            "the superseded original never reached the archive folder",
        )

    def test_no_archive_folder_leaves_everything_in_place(self):
        """Absent the opt-in, nothing moves — the pre-existing default."""
        source = self._texture("mat_BaseMap.png")

        MapFactory.prepare_maps([source], group_by_set=False, rename=True)

        self.assertIn("mat_BaseMap.png", self._names())
        self.assertFalse(os.path.isdir(os.path.join(self.test_dir, "_originals")))

    def test_dry_run_never_moves_files(self):
        source = self._texture("mat_BaseMap.png")

        MapFactory.prepare_maps(
            [source],
            group_by_set=False,
            rename=True,
            old_files_folder="_originals",
            dry_run=True,
        )

        self.assertIn("mat_BaseMap.png", self._names())

    def test_outputs_are_never_archived(self):
        """A file that survives into the result set must stay put."""
        source = self._texture("mat_Base_Color.png")

        result = MapFactory.prepare_maps(
            [source], group_by_set=False, old_files_folder="_originals"
        )

        self.assertIn("mat_Base_Color.png", self._names())
        self.assertTrue(all(os.path.isfile(p) for p in result))


class TestMapFactoryCancellation(unittest.TestCase):
    """``prepare_maps`` had no cancel checkpoint in either batch branch.

    mayatk's material updater runs it under ``@Cancelable(300)``, and every
    live caller leaves ``max_workers`` at 1, so a cancelled scope was simply
    never observed: the user held Esc, ``execution_monitor`` set the flag, and
    the batch ground through every remaining texture set. ``OperationCancelled``
    already derives from ``BaseException`` precisely so it slips past the loop's
    ``except Exception`` handlers -- nothing was raising it.
    """

    SETS = 6

    def setUp(self):
        self.processed = []
        self.files = [f"asset_{i}_Base_color.png" for i in range(self.SETS)]

    def _stub(self, on_set=None):
        """Patch the per-set worker out, recording each call.

        *on_set* runs after each recorded set, so a test can cancel mid-batch;
        the record lives on ``self`` so an ``OperationCancelled`` unwinding the
        batch does not take the count with it.
        """

        def fake(textures, config, output_dir=None, logger=None):
            self.processed.append(textures[0])
            if on_set is not None:
                on_set(len(self.processed))
            return list(textures)

        return patch.object(MapFactory, "_process_map_set", side_effect=fake)

    def test_serial_batch_stops_at_the_next_set_after_cancel(self):
        """The branch every live caller takes: ``max_workers`` defaults to 1."""
        with ptk.CancelScope(name="batch") as scope:
            stop = lambda n: scope.cancel("Esc") if n >= 2 else None  # noqa: E731
            with self._stub(on_set=stop):
                with self.assertRaises(ptk.OperationCancelled):
                    MapFactory.prepare_maps(self.files)
        self.assertEqual(len(self.processed), 2, "batch ran past the cancel")

    def test_a_scope_cancelled_up_front_does_no_work_at_all(self):
        with ptk.CancelScope(name="batch") as scope:
            scope.cancel("Esc")
            with self._stub():
                with self.assertRaises(ptk.OperationCancelled):
                    MapFactory.prepare_maps(self.files)
        self.assertEqual(self.processed, [])

    def test_parallel_batch_honors_cancel_on_the_collecting_thread(self):
        """The ambient scope is a ``ContextVar``, so a pool worker starts with
        an empty context and never sees it -- the checkpoint has to live on the
        thread that submits and collects, not inside the submitted task."""
        with ptk.CancelScope(name="batch") as scope:
            scope.cancel("Esc")
            with self._stub():
                with self.assertRaises(ptk.OperationCancelled):
                    MapFactory.prepare_maps(self.files, max_workers=4)
        self.assertLess(len(self.processed), self.SETS)

    def test_no_active_scope_leaves_the_batch_untouched(self):
        """The checkpoint is a no-op when nothing is monitoring."""
        self.assertIsNone(ptk.CancelScope.current())
        with self._stub():
            result = MapFactory.prepare_maps(self.files)
        self.assertEqual(len(self.processed), self.SETS)
        self.assertEqual(len(result), self.SETS)

    def test_progress_callback_returning_false_cancels(self):
        """``CancelScope.tick`` is written to match the progress-bar
        ``update()`` contract, and ``Cancelable`` documents progress reporting
        as the free checkpoint; the return value was discarded."""
        calls = []

        def progress(current, total, message):
            calls.append(current)
            return len(calls) < 3  # False from the third call on

        with self._stub():
            with self.assertRaises(ptk.OperationCancelled):
                MapFactory.prepare_maps(self.files, progress_callback=progress)
        self.assertLess(len(self.processed), self.SETS)

    def test_a_progress_callback_returning_none_is_not_a_cancel(self):
        """A plain callback that only prints returns ``None``; treating that as
        a cancel would break every existing caller."""
        seen = []

        def progress(current, total, message):
            seen.append(current)  # returns None

        with self._stub():
            MapFactory.prepare_maps(self.files, progress_callback=progress)
        self.assertEqual(len(self.processed), self.SETS)
        self.assertEqual(len(seen), self.SETS)


class TestConversionPluginSeam(unittest.TestCase):
    """``ConversionRegistry`` carried two registration protocols, and only one
    of them could ever run.

    ``_scan_pending`` preferred ``cls.register_conversions(registry)`` and fell
    back to ``register_from_class``, which scans members for a
    ``_conversion_info`` attribute. Nothing in any of the seven ecosystem
    packages sets ``_conversion_info`` -- there is no decorator that produces
    it -- and the one registered plugin (``MapFactory``) defines
    ``register_conversions``, so the fallback was measured at zero invocations
    while five real conversions resolved. Its practical effect was to turn a
    plugin that forgot ``register_conversions`` into a silent no-op.
    """

    def test_a_plugin_without_register_conversions_is_refused(self):
        """Previously accepted, then silently contributed nothing."""
        registry = ConversionRegistry()

        class NotAPlugin:
            pass

        with self.assertRaises(TypeError) as caught:
            registry.add_plugin(NotAPlugin)
        self.assertIn("register_conversions", str(caught.exception))

    def test_a_valid_plugin_is_still_scanned_lazily(self):
        """The deferral is the point of ``add_plugin``: no registration work
        happens until the first lookup."""
        registry = ConversionRegistry()
        scanned = []

        class Plugin:
            @classmethod
            def register_conversions(cls, reg):
                scanned.append(reg)
                reg.register(
                    target_type="Widget",
                    source_types=["Gadget"],
                    converter=lambda inv, ctx: "converted",
                )

        registry.add_plugin(Plugin)
        self.assertEqual(scanned, [], "add_plugin scanned eagerly")
        found = registry.get_conversions_for("Widget")
        self.assertEqual(len(scanned), 1)
        self.assertEqual(len(found), 1)
        # Scanned once, not on every lookup.
        registry.get_conversions_for("Widget")
        self.assertEqual(len(scanned), 1)

    def test_register_from_class_is_deprecated(self):
        registry = ConversionRegistry()

        class Legacy:
            pass

        with self.assertWarns(DeprecationWarning) as caught:
            registry.register_from_class(Legacy)
        self.assertIn("register_conversions", str(caught.warning))

    def test_the_live_registration_path_is_unaffected(self):
        """Guards the deletion: MapFactory's own conversions still resolve."""
        registry = ConversionRegistry()
        registry.add_plugin(MapFactory)
        self.assertTrue(registry.get_conversions_for("Metallic"))


if __name__ == "__main__":
    unittest.main()
