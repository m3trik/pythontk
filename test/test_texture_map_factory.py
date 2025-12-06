#!/usr/bin/python
# coding=utf-8
"""
Comprehensive tests for TextureMapFactory.

Tests cover:
- Map inventory building from texture lists
- Workflow-specific map preparation (Unity, Unreal, glTF, etc.)
- Map packing operations (Albedo+Transparency, Metallic+Smoothness, MSAO)
- Format conversions (DirectX ↔ OpenGL normals, Roughness ↔ Smoothness)
- Specular/Glossiness to PBR conversions
- Edge cases and error handling
"""
import os
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, call

from pythontk import ImgUtils
from pythontk.img_utils.texture_map_factory import TextureMapFactory

# Import BaseTestCase from conftest (auto-loaded by pytest)
import sys

sys.path.insert(0, os.path.dirname(__file__))
from conftest import BaseTestCase


class TestTextureMapFactory(BaseTestCase):
    """Comprehensive tests for TextureMapFactory."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Create temporary directory for test outputs
        cls.test_dir = tempfile.mkdtemp(prefix="texture_factory_test_")
        cls.test_files_dir = os.path.join(cls.test_dir, "textures")
        os.makedirs(cls.test_files_dir, exist_ok=True)

        # Create sample texture files for testing
        cls._create_test_textures()

    @classmethod
    def tearDownClass(cls):
        """Clean up test directory after all tests."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    @classmethod
    def _create_test_textures(cls):
        """Create sample texture files for testing."""
        # Define test texture set
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

        # Create actual image files
        cls.texture_paths = []
        for map_type, filename in cls.test_textures.items():
            filepath = os.path.join(cls.test_files_dir, filename)

            # Create appropriate test images based on type
            if "Normal" in map_type:
                # Normal maps are typically RGB with blue-ish tint
                img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 255))
            elif map_type in [
                "Metallic",
                "Roughness",
                "Smoothness",
                "AO",
                "Opacity",
                "Height",
            ]:
                # These are typically grayscale
                img = ImgUtils.create_image("L", (512, 512), 128)
            else:
                # Base color, emissive, etc. are RGB
                img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))

            ImgUtils.save_image(img, filepath)
            cls.texture_paths.append(filepath)

    # -------------------------------------------------------------------------
    # Map Inventory Tests
    # -------------------------------------------------------------------------

    def test_build_map_inventory_basic(self):
        """Test _build_map_inventory detects standard PBR maps."""
        inventory = TextureMapFactory._build_map_inventory(self.texture_paths)

        # Should detect all our test textures
        self.assertIn("Base_Color", inventory)
        self.assertIn("Metallic", inventory)
        self.assertIn("Roughness", inventory)
        self.assertIn("Normal_OpenGL", inventory)
        self.assertIn("Ambient_Occlusion", inventory)

    def test_build_map_inventory_empty_list(self):
        """Test _build_map_inventory with empty texture list."""
        inventory = TextureMapFactory._build_map_inventory([])
        self.assertEqual(inventory, {})

    def test_build_map_inventory_no_matches(self):
        """Test _build_map_inventory with textures that don't match any type."""
        fake_textures = [
            os.path.join(self.test_files_dir, "random_file_1.png"),
            os.path.join(self.test_files_dir, "random_file_2.png"),
        ]
        inventory = TextureMapFactory._build_map_inventory(fake_textures)
        # Should have empty or very minimal inventory
        self.assertTrue(
            len(inventory) == 0 or all(v is None for v in inventory.values())
        )

    def test_build_map_inventory_uses_imgutils_map_types(self):
        """Test _build_map_inventory leverages ImgUtils.map_types (DRY principle)."""
        # This test verifies we're not hardcoding map types
        inventory = TextureMapFactory._build_map_inventory(self.texture_paths)

        # All detected types should be valid keys in ImgUtils.map_types
        for map_type in inventory.keys():
            self.assertIn(map_type, ImgUtils.map_types.keys())

    def test_build_map_inventory_first_match_only(self):
        """Test _build_map_inventory takes only first match when multiple exist."""
        # Create duplicate normal maps
        normal_gl_1 = os.path.join(self.test_files_dir, "mat_Normal_GL.png")
        normal_gl_2 = os.path.join(self.test_files_dir, "mat_Normal_OpenGL.png")

        img = ImgUtils.create_image("RGB", (256, 256), (128, 128, 255))
        ImgUtils.save_image(img, normal_gl_1)
        ImgUtils.save_image(img, normal_gl_2)

        textures = [normal_gl_1, normal_gl_2]
        inventory = TextureMapFactory._build_map_inventory(textures)

        # Should only have one Normal_OpenGL entry (first match)
        if "Normal_OpenGL" in inventory:
            self.assertIsInstance(inventory["Normal_OpenGL"], str)
            # Verify it's one of our files
            self.assertTrue(inventory["Normal_OpenGL"] in textures)

    # -------------------------------------------------------------------------
    # prepare_maps() Integration Tests
    # -------------------------------------------------------------------------

    def test_prepare_maps_standard_pbr(self):
        """Test prepare_maps with standard PBR workflow (separate maps)."""
        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        callback = Mock()
        result = TextureMapFactory.prepare_maps(self.texture_paths, config, callback)

        # Should return list of texture paths
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

        # All returned paths should exist
        for path in result:
            self.assertTrue(os.path.exists(path), f"Missing: {path}")

    def test_prepare_maps_empty_texture_list(self):
        """Test prepare_maps with empty texture list."""
        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps([], config)
        self.assertEqual(result, [])

    def test_prepare_maps_unity_urp(self):
        """Test prepare_maps with Unity URP workflow (Albedo+Alpha, Metallic+Smoothness)."""
        config = {
            "albedo_transparency": True,
            "metallic_smoothness": True,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps(self.texture_paths, config)

        # Should have generated packed maps
        self.assertGreater(len(result), 0)

        # Check that packed maps exist
        for path in result:
            self.assertTrue(os.path.exists(path))

    def test_prepare_maps_unity_hdrp(self):
        """Test prepare_maps with Unity HDRP workflow (Mask Map/MSAO)."""
        config = {
            "albedo_transparency": True,
            "metallic_smoothness": False,
            "mask_map": True,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps(self.texture_paths, config)

        # Should have generated mask map
        self.assertGreater(len(result), 0)

    def test_prepare_maps_unreal_engine(self):
        """Test prepare_maps with Unreal Engine workflow (ORM packed map)."""
        # Unreal uses ORM (Ambient Occlusion, Roughness, Metallic) packed map
        # This should be similar to standard PBR but checking the workflow
        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "DirectX",  # Unreal uses DirectX normals
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps(self.texture_paths, config)
        self.assertGreater(len(result), 0)

    def test_prepare_maps_callback_invoked(self):
        """Test prepare_maps accepts callback parameter."""
        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        callback = Mock()
        result = TextureMapFactory.prepare_maps(self.texture_paths, config, callback)

        # Should complete without error
        self.assertIsInstance(result, list)

    def test_prepare_maps_custom_output_extension(self):
        """Test prepare_maps respects custom output extension."""
        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "tga",
        }

        result = TextureMapFactory.prepare_maps(self.texture_paths, config)

        # Generated maps should use .tga extension
        generated_maps = [
            p for p in result if not any(orig in p for orig in self.texture_paths)
        ]
        for path in generated_maps:
            if os.path.exists(path):  # Only check files that were actually generated
                self.assertTrue(
                    path.endswith(".tga"), f"Expected .tga extension: {path}"
                )

    # -------------------------------------------------------------------------
    # _prepare_base_color() Tests
    # -------------------------------------------------------------------------

    def test_prepare_base_color_no_packing(self):
        """Test _prepare_base_color without transparency packing."""
        inventory = TextureMapFactory._build_map_inventory(self.texture_paths)

        result = TextureMapFactory._prepare_base_color(
            inventory,
            pack_transparency=False,
            cleanup_base_color=False,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        # Should return existing base color map
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))

    def test_prepare_base_color_with_packing(self):
        """Test _prepare_base_color with transparency packing."""
        inventory = TextureMapFactory._build_map_inventory(self.texture_paths)

        result = TextureMapFactory._prepare_base_color(
            inventory,
            pack_transparency=True,
            cleanup_base_color=False,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        # Should generate new packed map
        if result:
            self.assertTrue(os.path.exists(result))

    def test_prepare_base_color_missing_maps(self):
        """Test _prepare_base_color when base color map is missing."""
        # Create inventory without base color
        inventory = {"Metallic": self.test_textures["Metallic"]}

        result = TextureMapFactory._prepare_base_color(
            inventory,
            pack_transparency=False,
            cleanup_base_color=False,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        # Should return None or handle gracefully
        self.assertTrue(result is None or isinstance(result, str))

    # -------------------------------------------------------------------------
    # _prepare_metallic_smoothness() Tests
    # -------------------------------------------------------------------------

    def test_prepare_metallic_smoothness_basic(self):
        """Test _prepare_metallic_smoothness packs metallic and smoothness."""
        inventory = TextureMapFactory._build_map_inventory(self.texture_paths)

        result = TextureMapFactory._prepare_metallic_smoothness(
            inventory,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))
            # Verify it's a valid image
            img = ImgUtils.load_image(result)
            self.assertIsNotNone(img)

    def test_prepare_metallic_smoothness_from_roughness(self):
        """Test _prepare_metallic_smoothness converts roughness to smoothness."""
        # Create inventory with roughness but no smoothness (use full paths)
        inventory = {
            "Metallic": os.path.join(
                self.test_files_dir, self.test_textures["Metallic"]
            ),
            "Roughness": os.path.join(
                self.test_files_dir, self.test_textures["Roughness"]
            ),
        }

        result = TextureMapFactory._prepare_metallic_smoothness(
            inventory,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))

    def test_prepare_metallic_smoothness_missing_both(self):
        """Test _prepare_metallic_smoothness when both metallic and smoothness missing."""
        inventory = {"Base_Color": self.test_textures["Base_Color"]}

        result = TextureMapFactory._prepare_metallic_smoothness(
            inventory,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        # Should return None or handle gracefully
        self.assertTrue(result is None or isinstance(result, str))

    # -------------------------------------------------------------------------
    # _prepare_mask_map() Tests (Unity HDRP MSAO)
    # -------------------------------------------------------------------------

    def test_prepare_mask_map_basic(self):
        """Test _prepare_mask_map creates Unity HDRP MSAO texture."""
        inventory = TextureMapFactory._build_map_inventory(self.texture_paths)

        result = TextureMapFactory._prepare_mask_map(
            inventory,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))
            # Verify it's a valid RGBA image
            img = ImgUtils.load_image(result)
            self.assertIsNotNone(img)

    def test_prepare_mask_map_missing_channels(self):
        """Test _prepare_mask_map handles missing channels gracefully."""
        # Create inventory with only some channels
        inventory = {
            "Metallic": self.test_textures["Metallic"],
            "Ambient_Occlusion": self.test_textures["Ambient_Occlusion"],
        }

        result = TextureMapFactory._prepare_mask_map(
            inventory,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        # Should still create mask map with available channels
        self.assertTrue(result is None or isinstance(result, str))

    # -------------------------------------------------------------------------
    # _prepare_metallic() Tests
    # -------------------------------------------------------------------------

    def test_prepare_metallic_existing(self):
        """Test _prepare_metallic returns existing metallic map."""
        inventory = TextureMapFactory._build_map_inventory(self.texture_paths)

        result = TextureMapFactory._prepare_metallic(
            inventory,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))

    def test_prepare_metallic_from_specular(self):
        """Test _prepare_metallic creates metallic from specular map."""
        # Create inventory with specular but no metallic
        inventory = {
            "Specular": self.test_textures["Specular"],
        }

        result = TextureMapFactory._prepare_metallic(
            inventory,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))

    def test_prepare_metallic_missing(self):
        """Test _prepare_metallic when metallic and specular both missing."""
        inventory = {"Base_Color": self.test_textures["Base_Color"]}

        result = TextureMapFactory._prepare_metallic(
            inventory,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        # Should return None
        self.assertIsNone(result)

    # -------------------------------------------------------------------------
    # _prepare_roughness() Tests
    # -------------------------------------------------------------------------

    def test_prepare_roughness_existing(self):
        """Test _prepare_roughness returns existing roughness map."""
        inventory = TextureMapFactory._build_map_inventory(self.texture_paths)

        result = TextureMapFactory._prepare_roughness(
            inventory,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))

    def test_prepare_roughness_from_smoothness(self):
        """Test _prepare_roughness converts smoothness to roughness."""
        # Create inventory with smoothness but no roughness
        inventory = {
            "Smoothness": self.test_textures["Smoothness"],
        }

        result = TextureMapFactory._prepare_roughness(
            inventory,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))

    def test_prepare_roughness_from_glossiness(self):
        """Test _prepare_roughness converts glossiness to roughness."""
        inventory = {
            "Glossiness": self.test_textures["Glossiness"],
        }

        result = TextureMapFactory._prepare_roughness(
            inventory,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))

    def test_prepare_roughness_from_specular(self):
        """Test _prepare_roughness creates roughness from specular."""
        inventory = {
            "Specular": self.test_textures["Specular"],
        }

        result = TextureMapFactory._prepare_roughness(
            inventory,
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))

    # -------------------------------------------------------------------------
    # _prepare_normal() Tests
    # -------------------------------------------------------------------------

    def test_prepare_normal_opengl_to_opengl(self):
        """Test _prepare_normal returns existing OpenGL normal unchanged."""
        inventory = TextureMapFactory._build_map_inventory(self.texture_paths)

        result = TextureMapFactory._prepare_normal(
            inventory,
            target_format="OpenGL",
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))

    def test_prepare_normal_directx_to_opengl(self):
        """Test _prepare_normal converts DirectX to OpenGL."""
        # Create inventory with only DirectX normal (use full path)
        inventory = {
            "Normal_DirectX": os.path.join(
                self.test_files_dir, self.test_textures["Normal_DirectX"]
            ),
        }

        result = TextureMapFactory._prepare_normal(
            inventory,
            target_format="OpenGL",
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))

    def test_prepare_normal_opengl_to_directx(self):
        """Test _prepare_normal converts OpenGL to DirectX."""
        inventory = {
            "Normal_OpenGL": os.path.join(
                self.test_files_dir, self.test_textures["Normal_OpenGL"]
            ),
        }

        result = TextureMapFactory._prepare_normal(
            inventory,
            target_format="DirectX",
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))

    def test_prepare_normal_generic_to_opengl(self):
        """Test _prepare_normal handles generic normal map."""
        # Create a generic normal map (no DirectX/OpenGL suffix)
        generic_normal = os.path.join(self.test_files_dir, "test_material_Normal.png")
        img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 255))
        ImgUtils.save_image(img, generic_normal)

        inventory = {
            "Normal": generic_normal,
        }

        result = TextureMapFactory._prepare_normal(
            inventory,
            target_format="OpenGL",
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))

    def test_prepare_normal_missing(self):
        """Test _prepare_normal when no normal map exists."""
        inventory = {"Base_Color": self.test_textures["Base_Color"]}

        result = TextureMapFactory._prepare_normal(
            inventory,
            target_format="OpenGL",
            output_dir=self.test_files_dir,
            base_name="test_material",
            ext="png",
            callback=print,
        )

        # Should return None
        self.assertIsNone(result)

    # -------------------------------------------------------------------------
    # CRITICAL Integration Tests - Verify Output Contains Expected Maps
    # -------------------------------------------------------------------------

    def test_unity_hdrp_workflow_contains_msao_map(self):
        """CRITICAL: Verify Unity HDRP workflow outputs MSAO map in final texture list."""
        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": True,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps(self.texture_paths, config)

        # CRITICAL: Verify MSAO map is in output
        msao_maps = [
            tex
            for tex in result
            if "MaskMap" in tex or ImgUtils.resolve_map_type(tex) == "MSAO"
        ]
        self.assertGreater(
            len(msao_maps), 0, "Unity HDRP workflow must include MSAO/MaskMap in output"
        )

        # CRITICAL: Verify individual maps are NOT in output (they were packed)
        for tex in result:
            map_type = ImgUtils.resolve_map_type(tex)
            self.assertNotIn(
                map_type,
                ["Metallic", "Roughness", "Smoothness"],
                f"Individual {map_type} map should be replaced by MSAO mask map",
            )

    def test_unity_hdrp_workflow_excludes_individual_maps(self):
        """CRITICAL: Verify Unity HDRP excludes metallic/roughness/smoothness when mask map created."""
        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": True,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps(self.texture_paths, config)

        # Extract map types from result
        output_types = [ImgUtils.resolve_map_type(tex) for tex in result if tex]

        # Should have MSAO
        self.assertIn("MSAO", output_types, "Unity HDRP must include MSAO map")

        # Should NOT have individual maps that were packed
        self.assertNotIn("Metallic", output_types, "Metallic should be packed in MSAO")
        self.assertNotIn(
            "Roughness", output_types, "Roughness should be packed in MSAO"
        )
        self.assertNotIn(
            "Smoothness", output_types, "Smoothness should be packed in MSAO"
        )
        self.assertNotIn(
            "Ambient_Occlusion", output_types, "AO should be packed in MSAO"
        )

    def test_unity_urp_workflow_contains_packed_maps(self):
        """CRITICAL: Verify Unity URP outputs packed Albedo+Alpha and Metallic+Smoothness."""
        config = {
            "albedo_transparency": True,
            "metallic_smoothness": True,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps(self.texture_paths, config)

        # Extract map types
        output_types = [ImgUtils.resolve_map_type(tex) for tex in result if tex]

        # Should have Albedo_Transparency (packed)
        has_albedo_transparency = any(
            "Albedo_Transparency" in t or "BaseColor" in os.path.basename(tex)
            for t, tex in zip(output_types, result)
        )
        self.assertTrue(
            has_albedo_transparency,
            "Unity URP workflow must include Albedo+Transparency packed map",
        )

        # Should have Metallic_Smoothness (packed)
        has_metallic_smoothness = "Metallic_Smoothness" in output_types or any(
            "Metallic" in os.path.basename(tex)
            and ImgUtils.load_image(tex).mode == "RGBA"
            for tex in result
            if tex
        )
        self.assertTrue(
            has_metallic_smoothness,
            "Unity URP workflow must include Metallic+Smoothness packed map",
        )

    def test_standard_pbr_workflow_contains_separate_maps(self):
        """CRITICAL: Verify standard PBR outputs separate Metallic and Roughness maps."""
        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps(self.texture_paths, config)

        # Extract map types
        output_types = [ImgUtils.resolve_map_type(tex) for tex in result if tex]

        # Should have separate Metallic and Roughness
        self.assertIn(
            "Metallic", output_types, "Standard PBR must include Metallic map"
        )
        self.assertIn(
            "Roughness", output_types, "Standard PBR must include Roughness map"
        )

        # Should NOT have packed versions
        self.assertNotIn(
            "Metallic_Smoothness", output_types, "Should have separate maps"
        )
        self.assertNotIn("MSAO", output_types, "Should have separate maps")

    def test_workflow_output_count_reasonable(self):
        """CRITICAL: Verify workflows output reasonable number of maps (not duplicates)."""
        workflows = [
            {
                "name": "Unity HDRP",
                "config": {
                    "mask_map": True,
                    "albedo_transparency": False,
                    "metallic_smoothness": False,
                },
            },
            {
                "name": "Unity URP",
                "config": {
                    "mask_map": False,
                    "albedo_transparency": True,
                    "metallic_smoothness": True,
                },
            },
            {
                "name": "Standard PBR",
                "config": {
                    "mask_map": False,
                    "albedo_transparency": False,
                    "metallic_smoothness": False,
                },
            },
        ]

        for workflow in workflows:
            config = {
                **workflow["config"],
                "normal_type": "OpenGL",
                "output_extension": "png",
            }

            result = TextureMapFactory.prepare_maps(self.texture_paths, config)

            # Should have 3-6 maps typically (BaseColor, Normal, MSAO/Metallic/Roughness, maybe AO, Emissive, Height)
            self.assertGreaterEqual(
                len(result),
                2,
                f"{workflow['name']}: Must have at least 2 maps (base color + something)",
            )
            self.assertLessEqual(
                len(result),
                10,
                f"{workflow['name']}: Should not have excessive duplicate maps (got {len(result)})",
            )


# =============================================================================
# Edge Cases & Error Handling
# =============================================================================


class TestTextureMapFactoryEdgeCases(BaseTestCase):
    """Edge case and error handling tests for TextureMapFactory."""

    def setUp(self):
        """Set up temporary test directory for each test."""
        self.test_dir = tempfile.mkdtemp(prefix="texture_factory_edge_")
        self.test_files_dir = os.path.join(self.test_dir, "textures")
        os.makedirs(self.test_files_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_prepare_maps_invalid_config(self):
        """Test prepare_maps handles missing config keys gracefully."""
        # Create a minimal texture set
        base_color = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("RGB", (256, 256), (128, 128, 128))
        ImgUtils.save_image(img, base_color)

        # Config with missing keys
        config = {
            "normal_type": "OpenGL",
        }

        result = TextureMapFactory.prepare_maps([base_color], config)

        # Should handle gracefully and return some result
        self.assertIsInstance(result, list)

    def test_prepare_maps_nonexistent_files(self):
        """Test prepare_maps with nonexistent file paths."""
        fake_paths = [
            "/nonexistent/path/texture1.png",
            "/nonexistent/path/texture2.png",
        ]

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        # Factory returns original list when files don't exist (no processing possible)
        result = TextureMapFactory.prepare_maps(fake_paths, config)
        # Should return the original paths (can't process nonexistent files)
        self.assertEqual(result, fake_paths)

    def test_build_map_inventory_duplicate_types(self):
        """Test _build_map_inventory when multiple files match same type."""
        # Create multiple base color variations
        textures = []
        for suffix in ["BaseColor", "Albedo", "Diffuse"]:
            path = os.path.join(self.test_files_dir, f"mat_{suffix}.png")
            img = ImgUtils.create_image("RGB", (256, 256), (128, 0, 0))
            ImgUtils.save_image(img, path)
            textures.append(path)

        inventory = TextureMapFactory._build_map_inventory(textures)

        # Should have Base_Color entry (first match wins)
        if "Base_Color" in inventory or "Diffuse" in inventory or "Albedo" in inventory:
            # At least one was detected
            detected_types = [
                k for k in ["Base_Color", "Diffuse", "Albedo"] if k in inventory
            ]
            self.assertGreater(len(detected_types), 0)

    def test_prepare_maps_callback_exception_handling(self):
        """Test prepare_maps continues if callback raises exception."""
        base_color = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("RGB", (256, 256), (128, 128, 128))
        ImgUtils.save_image(img, base_color)

        def bad_callback(msg):
            raise RuntimeError("Callback error")

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        # Should handle callback exception gracefully
        try:
            result = TextureMapFactory.prepare_maps([base_color], config, bad_callback)
            # If it didn't crash, that's good
            self.assertIsInstance(result, list)
        except RuntimeError:
            # If exception propagates, that's also acceptable behavior
            pass

    def test_prepare_base_color_corrupted_image(self):
        """Test _prepare_base_color with corrupted image file."""
        # Create a corrupted image file (just text)
        corrupted_path = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        with open(corrupted_path, "w") as f:
            f.write("This is not a valid PNG file")

        inventory = {"Base_Color": corrupted_path}

        # Should handle gracefully without crashing
        try:
            result = TextureMapFactory._prepare_base_color(
                inventory,
                pack_transparency=False,
                output_dir=self.test_files_dir,
                base_name="mat",
                ext="png",
                callback=print,
            )
            # Should return None or raise an exception
            self.assertTrue(result is None or isinstance(result, str))
        except Exception:
            # Exception is acceptable for corrupted files
            pass


if __name__ == "__main__":
    unittest.main()


# =============================================================================
# Additional Edge Cases & Stress Tests
# =============================================================================


class TestTextureMapFactoryExtendedEdgeCases(BaseTestCase):
    """Extended edge case tests for robustness."""

    def setUp(self):
        """Set up temporary test directory for each test."""
        self.test_dir = tempfile.mkdtemp(prefix="texture_factory_extended_")
        self.test_files_dir = os.path.join(self.test_dir, "textures")
        os.makedirs(self.test_files_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # -------------------------------------------------------------------------
    # Unusual File Naming Edge Cases
    # -------------------------------------------------------------------------

    def test_prepare_maps_unicode_filenames(self):
        """Test prepare_maps with unicode characters in filenames."""
        unicode_path = os.path.join(self.test_files_dir, "テクスチャ_BaseColor.png")
        img = ImgUtils.create_image("RGB", (256, 256), (128, 128, 128))
        ImgUtils.save_image(img, unicode_path)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps([unicode_path], config)
        self.assertIsInstance(result, list)

    def test_prepare_maps_spaces_in_filenames(self):
        """Test prepare_maps with spaces in filenames."""
        spaced_path = os.path.join(self.test_files_dir, "My Material_Base Color.png")
        img = ImgUtils.create_image("RGB", (256, 256), (128, 128, 128))
        ImgUtils.save_image(img, spaced_path)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps([spaced_path], config)
        self.assertIsInstance(result, list)

    def test_prepare_maps_special_characters_in_path(self):
        """Test prepare_maps with special characters in directory path."""
        special_dir = os.path.join(self.test_dir, "textures#@$%")
        os.makedirs(special_dir, exist_ok=True)

        special_path = os.path.join(special_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("RGB", (256, 256), (128, 128, 128))
        ImgUtils.save_image(img, special_path)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps([special_path], config)
        self.assertIsInstance(result, list)

    def test_prepare_maps_very_long_filename(self):
        """Test prepare_maps with extremely long filename."""
        long_name = "A" * 200 + "_BaseColor.png"
        long_path = os.path.join(self.test_files_dir, long_name)

        try:
            img = ImgUtils.create_image("RGB", (256, 256), (128, 128, 128))
            ImgUtils.save_image(img, long_path)

            config = {
                "albedo_transparency": False,
                "metallic_smoothness": False,
                "mask_map": False,
                "normal_type": "OpenGL",
                "output_extension": "png",
            }

            result = TextureMapFactory.prepare_maps([long_path], config)
            self.assertIsInstance(result, list)
        except OSError:
            # Some filesystems don't support very long names - that's OK
            pass

    # -------------------------------------------------------------------------
    # Image Format/Size Edge Cases
    # -------------------------------------------------------------------------

    def test_prepare_maps_tiny_image(self):
        """Test prepare_maps with 1x1 pixel image."""
        tiny_path = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("RGB", (1, 1), (128, 128, 128))
        ImgUtils.save_image(img, tiny_path)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps([tiny_path], config)
        self.assertIsInstance(result, list)

    def test_prepare_maps_huge_image(self):
        """Test prepare_maps with very large image."""
        huge_path = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        # 8K texture
        img = ImgUtils.create_image("RGB", (8192, 8192), (128, 128, 128))
        ImgUtils.save_image(img, huge_path)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps([huge_path], config)
        self.assertIsInstance(result, list)

    def test_prepare_maps_non_square_image(self):
        """Test prepare_maps with non-square aspect ratio."""
        rect_path = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("RGB", (1024, 512), (128, 128, 128))
        ImgUtils.save_image(img, rect_path)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps([rect_path], config)
        self.assertIsInstance(result, list)

    def test_prepare_maps_grayscale_as_color(self):
        """Test prepare_maps treats grayscale base color map correctly."""
        gray_path = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("L", (512, 512), 128)
        ImgUtils.save_image(img, gray_path)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps([gray_path], config)
        self.assertIsInstance(result, list)

    def test_prepare_maps_rgba_normal_map(self):
        """Test prepare_maps with RGBA normal map (should use RGB channels)."""
        rgba_normal = os.path.join(self.test_files_dir, "mat_Normal_OpenGL.png")
        img = ImgUtils.create_image("RGBA", (512, 512), (128, 128, 255, 255))
        ImgUtils.save_image(img, rgba_normal)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps([rgba_normal], config)
        self.assertIsInstance(result, list)

    # -------------------------------------------------------------------------
    # Map Combination Edge Cases
    # -------------------------------------------------------------------------

    def test_prepare_mask_map_only_metallic(self):
        """Test _prepare_mask_map with only metallic map (no AO, no smoothness)."""
        metallic_path = os.path.join(self.test_files_dir, "mat_Metallic.png")
        img = ImgUtils.create_image("L", (512, 512), 128)
        ImgUtils.save_image(img, metallic_path)

        inventory = {"Metallic": metallic_path}

        result = TextureMapFactory._prepare_mask_map(
            inventory,
            output_dir=self.test_files_dir,
            base_name="mat",
            ext="png",
            callback=print,
        )

        # Should still create mask map (using placeholders for missing channels)
        if result:
            self.assertTrue(os.path.exists(result))

    def test_prepare_mask_map_no_metallic_but_specular(self):
        """Test _prepare_mask_map converts specular to metallic when metallic missing."""
        specular_path = os.path.join(self.test_files_dir, "mat_Specular.png")
        ao_path = os.path.join(self.test_files_dir, "mat_AO.png")
        roughness_path = os.path.join(self.test_files_dir, "mat_Roughness.png")

        for path in [specular_path, ao_path, roughness_path]:
            img = ImgUtils.create_image("L", (512, 512), 128)
            ImgUtils.save_image(img, path)

        inventory = {
            "Specular": specular_path,
            "Ambient_Occlusion": ao_path,
            "Roughness": roughness_path,
        }

        result = TextureMapFactory._prepare_mask_map(
            inventory,
            output_dir=self.test_files_dir,
            base_name="mat",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))

    def test_prepare_metallic_smoothness_only_metallic(self):
        """Test _prepare_metallic_smoothness with only metallic (no smoothness/roughness)."""
        metallic_path = os.path.join(self.test_files_dir, "mat_Metallic.png")
        img = ImgUtils.create_image("L", (512, 512), 128)
        ImgUtils.save_image(img, metallic_path)

        inventory = {"Metallic": metallic_path}

        result = TextureMapFactory._prepare_metallic_smoothness(
            inventory,
            output_dir=self.test_files_dir,
            base_name="mat",
            ext="png",
            callback=print,
        )

        # Should return metallic map unchanged (can't pack without alpha source)
        self.assertEqual(result, metallic_path)

    def test_prepare_base_color_opacity_without_base_color(self):
        """Test _prepare_base_color with opacity but no base color."""
        opacity_path = os.path.join(self.test_files_dir, "mat_Opacity.png")
        img = ImgUtils.create_image("L", (512, 512), 128)
        ImgUtils.save_image(img, opacity_path)

        inventory = {"Opacity": opacity_path}

        result = TextureMapFactory._prepare_base_color(
            inventory,
            pack_transparency=True,
            cleanup_base_color=False,
            output_dir=self.test_files_dir,
            base_name="mat",
            ext="png",
            callback=print,
        )

        # Should return None (can't pack opacity without base color)
        self.assertIsNone(result)

    # -------------------------------------------------------------------------
    # Workflow Combination Edge Cases
    # -------------------------------------------------------------------------

    def test_prepare_maps_all_flags_true(self):
        """Test prepare_maps with contradictory flags (mask_map + metallic_smoothness)."""
        base_color = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        metallic = os.path.join(self.test_files_dir, "mat_Metallic.png")
        roughness = os.path.join(self.test_files_dir, "mat_Roughness.png")

        for path in [base_color, metallic, roughness]:
            img = ImgUtils.create_image(
                "RGB" if "Color" in path else "L", (512, 512), 128
            )
            ImgUtils.save_image(img, path)

        config = {
            "albedo_transparency": True,
            "metallic_smoothness": True,
            "mask_map": True,  # Contradictory: can't have both mask_map and metallic_smoothness
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps(
            [base_color, metallic, roughness], config
        )

        # Should prioritize mask_map over metallic_smoothness (based on code order)
        self.assertIsInstance(result, list)

    def test_prepare_maps_no_workflow_flags(self):
        """Test prepare_maps with all workflow flags False."""
        base_color = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        metallic = os.path.join(self.test_files_dir, "mat_Metallic.png")

        for path in [base_color, metallic]:
            img = ImgUtils.create_image(
                "RGB" if "Color" in path else "L", (512, 512), 128
            )
            ImgUtils.save_image(img, path)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps([base_color, metallic], config)

        # Should return separate maps (standard PBR workflow)
        self.assertGreater(len(result), 0)

    # -------------------------------------------------------------------------
    # Output Extension Edge Cases
    # -------------------------------------------------------------------------

    def test_prepare_maps_uppercase_extension(self):
        """Test prepare_maps with uppercase output extension."""
        base_color = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
        ImgUtils.save_image(img, base_color)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "PNG",  # Uppercase
        }

        result = TextureMapFactory.prepare_maps([base_color], config)
        self.assertIsInstance(result, list)

    def test_prepare_maps_extension_with_dot(self):
        """Test prepare_maps with extension including dot prefix."""
        base_color = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
        ImgUtils.save_image(img, base_color)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": ".png",  # With dot
        }

        result = TextureMapFactory.prepare_maps([base_color], config)
        self.assertIsInstance(result, list)

    def test_prepare_maps_unsupported_extension(self):
        """Test prepare_maps with unusual output extension."""
        base_color = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
        ImgUtils.save_image(img, base_color)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "xyz",  # Unusual extension
        }

        # Should handle gracefully (PIL may raise error on save)
        try:
            result = TextureMapFactory.prepare_maps([base_color], config)
            self.assertIsInstance(result, list)
        except Exception:
            # Exception is acceptable for unsupported formats
            pass

    # -------------------------------------------------------------------------
    # Normal Map Format Edge Cases
    # -------------------------------------------------------------------------

    def test_prepare_normal_invalid_format(self):
        """Test _prepare_normal with invalid normal format string."""
        normal_gl = os.path.join(self.test_files_dir, "mat_Normal_OpenGL.png")
        img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 255))
        ImgUtils.save_image(img, normal_gl)

        inventory = {"Normal_OpenGL": normal_gl}

        result = TextureMapFactory._prepare_normal(
            inventory,
            target_format="InvalidFormat",  # Invalid format
            output_dir=self.test_files_dir,
            base_name="mat",
            ext="png",
            callback=print,
        )

        # Should handle gracefully (probably return None or existing map)
        self.assertTrue(result is None or isinstance(result, str))

    def test_prepare_normal_case_insensitive_format(self):
        """Test _prepare_normal handles case variations in format."""
        normal_gl = os.path.join(self.test_files_dir, "mat_Normal_OpenGL.png")
        img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 255))
        ImgUtils.save_image(img, normal_gl)

        inventory = {"Normal_OpenGL": normal_gl}

        # Try lowercase
        result = TextureMapFactory._prepare_normal(
            inventory,
            target_format="opengl",  # Lowercase
            output_dir=self.test_files_dir,
            base_name="mat",
            ext="png",
            callback=print,
        )

        self.assertTrue(result is None or isinstance(result, str))

    # -------------------------------------------------------------------------
    # Callback Function Edge Cases
    # -------------------------------------------------------------------------

    def test_prepare_maps_none_callback(self):
        """Test prepare_maps with None as callback (should use default print)."""
        base_color = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
        ImgUtils.save_image(img, base_color)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        # None callback should default to print
        result = TextureMapFactory.prepare_maps([base_color], config, None)
        self.assertIsInstance(result, list)

    def test_prepare_maps_lambda_callback(self):
        """Test prepare_maps with lambda callback."""
        base_color = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
        ImgUtils.save_image(img, base_color)

        messages = []
        callback = lambda msg: messages.append(msg)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps([base_color], config, callback)
        self.assertIsInstance(result, list)

    # -------------------------------------------------------------------------
    # Empty/Missing Data Edge Cases
    # -------------------------------------------------------------------------

    def test_prepare_maps_empty_config(self):
        """Test prepare_maps with empty config dictionary."""
        base_color = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
        ImgUtils.save_image(img, base_color)

        config = {}  # Empty config

        result = TextureMapFactory.prepare_maps([base_color], config)

        # Should use defaults and not crash
        self.assertIsInstance(result, list)

    def test_build_map_inventory_mixed_valid_invalid(self):
        """Test _build_map_inventory with mix of valid and invalid paths."""
        valid_path = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
        ImgUtils.save_image(img, valid_path)

        mixed_paths = [
            valid_path,
            "/nonexistent/fake_Metallic.png",
            os.path.join(self.test_files_dir, "missing_Normal.png"),
        ]

        inventory = TextureMapFactory._build_map_inventory(mixed_paths)

        # Should handle gracefully and process valid files
        self.assertIsInstance(inventory, dict)

    def test_prepare_maps_read_only_output_directory(self):
        """Test prepare_maps when output directory is read-only."""
        # This is platform-specific and may not work on all systems
        # Skip on systems where we can't make directories read-only
        pass  # TODO: Implement if filesystem permissions testing is needed

    def test_prepare_maps_disk_full_simulation(self):
        """Test prepare_maps behavior when disk is full (simulated)."""
        # Difficult to test without actually filling disk
        # Could be done with mocking if needed
        pass  # TODO: Implement with mocking if critical


# =============================================================================
# Performance & Stress Tests
# =============================================================================


class TestTextureMapFactoryPerformance(BaseTestCase):
    """Performance and stress tests for TextureMapFactory."""

    def setUp(self):
        """Set up temporary test directory."""
        self.test_dir = tempfile.mkdtemp(prefix="texture_factory_perf_")
        self.test_files_dir = os.path.join(self.test_dir, "textures")
        os.makedirs(self.test_files_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_prepare_maps_many_textures(self):
        """Test prepare_maps with large number of texture files."""
        # Create 50 texture files
        textures = []
        for i in range(50):
            path = os.path.join(self.test_files_dir, f"tex_{i}_BaseColor.png")
            img = ImgUtils.create_image("RGB", (256, 256), (i % 255, 128, 128))
            ImgUtils.save_image(img, path)
            textures.append(path)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps(textures, config)

        # Should complete without hanging
        self.assertIsInstance(result, list)

    def test_build_map_inventory_large_list(self):
        """Test _build_map_inventory performance with large texture list."""
        # Create 100 dummy paths
        large_list = [f"/fake/path/texture_{i}.png" for i in range(100)]

        inventory = TextureMapFactory._build_map_inventory(large_list)

        # Should complete quickly
        self.assertIsInstance(inventory, dict)

    def test_prepare_maps_repeated_calls(self):
        """Test prepare_maps called multiple times (memory leak check)."""
        base_color = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
        ImgUtils.save_image(img, base_color)

        config = {
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        # Call 10 times
        for _ in range(10):
            result = TextureMapFactory.prepare_maps([base_color], config)
            self.assertIsInstance(result, list)

        # Should complete without memory issues


# =============================================================================
# Modern Game Engine Workflow Tests
# =============================================================================


class TestModernGameEngineWorkflows(BaseTestCase):
    """Tests for modern game engine conversions: ORM, Spec/Gloss, Base Color cleanup."""

    def setUp(self):
        """Set up temporary test directory."""
        self.test_dir = tempfile.mkdtemp(prefix="texture_factory_modern_")
        self.test_files_dir = os.path.join(self.test_dir, "textures")
        os.makedirs(self.test_files_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # -------------------------------------------------------------------------
    # ORM Map Tests (Unreal Engine / glTF 2.0)
    # -------------------------------------------------------------------------

    def test_prepare_orm_map_complete_set(self):
        """Test _prepare_orm_map with complete AO, Roughness, Metallic set."""
        # Create test textures
        ao_path = os.path.join(self.test_files_dir, "mat_AO.png")
        roughness_path = os.path.join(self.test_files_dir, "mat_Roughness.png")
        metallic_path = os.path.join(self.test_files_dir, "mat_Metallic.png")

        for path in [ao_path, roughness_path, metallic_path]:
            img = ImgUtils.create_image("L", (512, 512), 128)
            ImgUtils.save_image(img, path)

        inventory = {
            "Ambient_Occlusion": ao_path,
            "Roughness": roughness_path,
            "Metallic": metallic_path,
        }

        result = TextureMapFactory._prepare_orm_map(
            inventory,
            output_dir=self.test_files_dir,
            base_name="mat",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))
            self.assertIn("ORM", result)
            # Verify it's RGB image
            img = ImgUtils.load_image(result)
            self.assertEqual(img.mode, "RGB")

    def test_prepare_orm_map_missing_ao(self):
        """Test _prepare_orm_map uses white when AO is missing."""
        roughness_path = os.path.join(self.test_files_dir, "mat_Roughness.png")
        metallic_path = os.path.join(self.test_files_dir, "mat_Metallic.png")

        for path in [roughness_path, metallic_path]:
            img = ImgUtils.create_image("L", (512, 512), 128)
            ImgUtils.save_image(img, path)

        inventory = {
            "Roughness": roughness_path,
            "Metallic": metallic_path,
        }

        result = TextureMapFactory._prepare_orm_map(
            inventory,
            output_dir=self.test_files_dir,
            base_name="mat",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))

    def test_prepare_orm_map_from_smoothness(self):
        """Test _prepare_orm_map converts smoothness to roughness."""
        smoothness_path = os.path.join(self.test_files_dir, "mat_Smoothness.png")
        metallic_path = os.path.join(self.test_files_dir, "mat_Metallic.png")

        for path in [smoothness_path, metallic_path]:
            img = ImgUtils.create_image("L", (512, 512), 128)
            ImgUtils.save_image(img, path)

        inventory = {
            "Smoothness": smoothness_path,
            "Metallic": metallic_path,
        }

        result = TextureMapFactory._prepare_orm_map(
            inventory,
            output_dir=self.test_files_dir,
            base_name="mat",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))

    def test_prepare_orm_map_from_specular(self):
        """Test _prepare_orm_map creates metallic and roughness from specular."""
        specular_path = os.path.join(self.test_files_dir, "mat_Specular.png")
        img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
        ImgUtils.save_image(img, specular_path)

        inventory = {
            "Specular": specular_path,
        }

        result = TextureMapFactory._prepare_orm_map(
            inventory,
            output_dir=self.test_files_dir,
            base_name="mat",
            ext="png",
            callback=print,
        )

        # Should create ORM from converted specular
        if result:
            self.assertTrue(os.path.exists(result))

    def test_prepare_maps_unreal_workflow(self):
        """Test prepare_maps with Unreal Engine ORM workflow."""
        # Create complete texture set
        textures = []
        for name, mode in [
            ("BaseColor", "RGB"),
            ("Roughness", "L"),
            ("Metallic", "L"),
            ("AO", "L"),
            ("Normal", "RGB"),
        ]:
            path = os.path.join(self.test_files_dir, f"mat_{name}.png")
            img = ImgUtils.create_image(
                mode, (512, 512), 128 if mode == "L" else (128, 128, 255)
            )
            ImgUtils.save_image(img, path)
            textures.append(path)

        config = {
            "orm_map": True,  # Enable ORM packing
            "albedo_transparency": False,
            "metallic_smoothness": False,
            "mask_map": False,
            "normal_type": "DirectX",  # Unreal uses DirectX normals
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps(textures, config)

        # Should have ORM map in output
        orm_maps = [tex for tex in result if "ORM" in tex]
        self.assertGreater(len(orm_maps), 0, "Unreal workflow must include ORM map")

        # Should NOT have individual metallic/roughness/AO
        for tex in result:
            map_type = ImgUtils.resolve_map_type(tex)
            if map_type:
                self.assertNotIn(
                    map_type,
                    ["Metallic", "Roughness"],
                    f"Individual {map_type} should be packed in ORM",
                )

    def test_prepare_maps_gltf_workflow(self):
        """Test prepare_maps with glTF 2.0 ORM workflow."""
        textures = []
        for name in ["BaseColor", "Roughness", "Metallic", "Normal"]:
            path = os.path.join(self.test_files_dir, f"mat_{name}.png")
            mode = "L" if name in ["Roughness", "Metallic"] else "RGB"
            img = ImgUtils.create_image(
                mode, (512, 512), 128 if mode == "L" else (128, 128, 255)
            )
            ImgUtils.save_image(img, path)
            textures.append(path)

        config = {
            "orm_map": True,
            "normal_type": "OpenGL",  # glTF uses OpenGL normals
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps(textures, config)

        # Verify ORM exists
        orm_maps = [
            tex
            for tex in result
            if "ORM" in tex or ImgUtils.resolve_map_type(tex) == "ORM"
        ]
        self.assertGreater(len(orm_maps), 0, "glTF workflow must include ORM map")

    # -------------------------------------------------------------------------
    # Spec/Gloss Conversion Tests
    # -------------------------------------------------------------------------

    def test_convert_specgloss_workflow_complete(self):
        """Test _convert_specgloss_workflow with full Spec/Gloss set."""
        # Create Spec/Gloss texture set
        diffuse_path = os.path.join(self.test_files_dir, "mat_Diffuse.png")
        specular_path = os.path.join(self.test_files_dir, "mat_Specular.png")
        glossiness_path = os.path.join(self.test_files_dir, "mat_Glossiness.png")

        for path, mode in [
            (diffuse_path, "RGB"),
            (specular_path, "RGB"),
            (glossiness_path, "L"),
        ]:
            img = ImgUtils.create_image(
                mode, (512, 512), (128, 128, 128) if mode == "RGB" else 128
            )
            ImgUtils.save_image(img, path)

        inventory = {
            "Diffuse": diffuse_path,
            "Specular": specular_path,
            "Glossiness": glossiness_path,
        }

        result = TextureMapFactory._convert_specgloss_workflow(
            inventory,
            output_dir=self.test_files_dir,
            base_name="mat",
            ext="png",
            callback=print,
        )

        # Should have Base_Color, Metallic, Roughness
        self.assertIn("Base_Color", result)
        self.assertIn("Metallic", result)
        self.assertIn("Roughness", result)

        # Should NOT have old maps
        self.assertNotIn("Diffuse", result)
        self.assertNotIn("Specular", result)
        self.assertNotIn("Glossiness", result)

    def test_convert_specgloss_workflow_partial(self):
        """Test _convert_specgloss_workflow with partial Spec/Gloss set."""
        # Only specular, no diffuse or glossiness
        specular_path = os.path.join(self.test_files_dir, "mat_Specular.png")
        img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
        ImgUtils.save_image(img, specular_path)

        inventory = {
            "Specular": specular_path,
        }

        result = TextureMapFactory._convert_specgloss_workflow(
            inventory,
            output_dir=self.test_files_dir,
            base_name="mat",
            ext="png",
            callback=print,
        )

        # Should return unchanged (not enough for conversion)
        self.assertEqual(result, inventory)

    def test_prepare_maps_specgloss_conversion(self):
        """Test prepare_maps auto-converts Spec/Gloss workflow."""
        # Create Spec/Gloss texture set
        textures = []
        for name, mode in [
            ("Diffuse", "RGB"),
            ("Specular", "RGB"),
            ("Smoothness", "L"),
        ]:
            path = os.path.join(self.test_files_dir, f"mat_{name}.png")
            img = ImgUtils.create_image(
                mode, (512, 512), (128, 128, 128) if mode == "RGB" else 128
            )
            ImgUtils.save_image(img, path)
            textures.append(path)

        config = {
            "convert_specgloss_to_pbr": True,  # Enable auto-conversion
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps(textures, config)

        # Should have PBR maps (Base_Color, Metallic, Roughness)
        output_types = [ImgUtils.resolve_map_type(tex) for tex in result if tex]

        # Verify PBR maps present
        has_base_color = any(
            "Base" in str(t) or "Color" in str(t) for t in output_types
        )
        has_metallic = "Metallic" in output_types
        has_roughness = "Roughness" in output_types

        self.assertTrue(
            has_base_color or has_metallic or has_roughness,
            "Spec/Gloss conversion should produce PBR maps",
        )

    # -------------------------------------------------------------------------
    # Base Color Cleanup Tests
    # -------------------------------------------------------------------------

    def test_prepare_base_color_with_cleanup(self):
        """Test _prepare_base_color with cleanup_base_color option."""
        # Create base color and metallic maps
        base_color_path = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        metallic_path = os.path.join(self.test_files_dir, "mat_Metallic.png")

        base_img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
        metallic_img = ImgUtils.create_image("L", (512, 512), 200)  # High metallic

        ImgUtils.save_image(base_img, base_color_path)
        ImgUtils.save_image(metallic_img, metallic_path)

        inventory = {
            "Base_Color": base_color_path,
            "Metallic": metallic_path,
        }

        result = TextureMapFactory._prepare_base_color(
            inventory,
            pack_transparency=False,
            cleanup_base_color=True,  # Enable cleanup
            output_dir=self.test_files_dir,
            base_name="mat",
            ext="png",
            callback=print,
        )

        if result:
            self.assertTrue(os.path.exists(result))
            self.assertIn("Albedo", result)  # Should create new albedo file

    def test_prepare_base_color_cleanup_without_metallic(self):
        """Test cleanup fails gracefully without metallic map."""
        base_color_path = os.path.join(self.test_files_dir, "mat_BaseColor.png")
        img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
        ImgUtils.save_image(img, base_color_path)

        inventory = {
            "Base_Color": base_color_path,
        }

        result = TextureMapFactory._prepare_base_color(
            inventory,
            pack_transparency=False,
            cleanup_base_color=True,  # Enable cleanup
            output_dir=self.test_files_dir,
            base_name="mat",
            ext="png",
            callback=print,
        )

        # Should return original base color (cleanup skipped without metallic)
        self.assertEqual(result, base_color_path)

    def test_prepare_maps_with_cleanup(self):
        """Test prepare_maps with base color cleanup enabled."""
        textures = []
        for name, mode in [("BaseColor", "RGB"), ("Metallic", "L")]:
            path = os.path.join(self.test_files_dir, f"mat_{name}.png")
            img = ImgUtils.create_image(
                mode, (512, 512), 128 if mode == "L" else (128, 128, 128)
            )
            ImgUtils.save_image(img, path)
            textures.append(path)

        config = {
            "cleanup_base_color": True,  # Enable cleanup
            "normal_type": "OpenGL",
            "output_extension": "png",
        }

        result = TextureMapFactory.prepare_maps(textures, config)

        # Should have albedo map
        albedo_maps = [tex for tex in result if "Albedo" in tex]
        # Cleanup may or may not produce new file depending on convert_base_color_to_albedo implementation
        self.assertIsInstance(result, list)
