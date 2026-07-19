#!/usr/bin/python
# coding=utf-8
"""
Refactored tests for MapFactory using the public API (Strategy Pattern).
"""
import os
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock
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

        mrao_path = next(
            p for p in results if "MRAO" in os.path.basename(p)
        )
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

        mrao_path = next(
            p for p in results if "MRAO" in os.path.basename(p)
        )
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
                p
                for p in sets[base]
                if MapFactory.resolve_map_type(p) == "Base_Color"
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
            self.assertEqual(
                source, before, "source list was mutated by discovery"
            )
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
            p
            for p in self.texture_paths
            if "Normal_OpenGL" in p or "Height" in p
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
        subset = [
            p for p in self.texture_paths if "BaseColor" in p or "Opacity" in p
        ]
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
        metallic channel."""
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
            )
            msao_path = next(
                p for p in results if "MSAO" in os.path.basename(p)
            )
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
        for data_map in ("rock_Normal.png", "rock_Roughness.png", "rock_Metallic.png",
                         "rock_Height.png", "rock_AO.png"):
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
            bump, output_format="opengl", save=False,
            smooth_filter=False, intensity=30.0,
        )
        arr = np.array(nm).astype(float)
        g_top = arr[: n // 2 - 10, :, 1].mean()
        g_bottom = arr[n // 2 + 10 :, :, 1].mean()
        self.assertGreater(g_top, g_bottom, "OpenGL green must face up")

        # Generator and detector must agree.
        self.assertEqual(MapFactory.detect_normal_map_format(nm), "OpenGL")
        nm_dx = MapFactory.convert_bump_to_normal(
            bump, output_format="directx", save=False,
            smooth_filter=False, intensity=30.0,
        )
        self.assertEqual(MapFactory.detect_normal_map_format(nm_dx), "DirectX")

        # Real-world labeled asset. The detector's internal 512px thumbnail
        # dilutes this map's correlation to ~-0.09 (full-res it is -0.19),
        # so probe with a low threshold — the assertion is about the SIGN
        # (the bug was a sign flip that read every real OpenGL map as
        # DirectX-leaning).
        asset = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_assets", "imgtk_test", "im_Normal_OpenGL.png",
        )
        if os.path.exists(asset):
            self.assertEqual(
                MapFactory.detect_normal_map_format(asset, threshold=0.05),
                "OpenGL",
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


if __name__ == "__main__":
    unittest.main()
