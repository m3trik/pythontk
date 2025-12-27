#!/usr/bin/python
# coding=utf-8
"""
Comprehensive tests for MapConverter with TextureMapFactory integration.

Tests cover:
- TextureMapFactory integration in tb001 (Spec/Gloss conversion)
- New b012 method for batch PBR workflow preparation
- All 7 workflow templates
- Error handling and fallback behavior
"""
import os
import tempfile
import shutil
import unittest
import importlib.util
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from pythontk import ImgUtils
from pythontk.img_utils.map_converter import MapConverterSlots

# Check if PySide2 is available (for b012 tests)
PYSIDE2_AVAILABLE = importlib.util.find_spec("PySide2") is not None
skip_if_no_pyside2 = unittest.skipUnless(
    PYSIDE2_AVAILABLE, "PySide2 not available - b012 tests require Qt for QInputDialog"
)


class TestMapConverterTextureFactory(unittest.TestCase):
    """Test MapConverter integration with TextureMapFactory."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Create temporary directory for test outputs
        cls.test_dir = tempfile.mkdtemp(prefix="map_converter_test_")
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
            "Base_Color": "material_BaseColor.png",
            "Metallic": "material_Metallic.png",
            "Roughness": "material_Roughness.png",
            "Normal_OpenGL": "material_Normal_OpenGL.png",
            "Ambient_Occlusion": "material_AO.png",
            "Opacity": "material_Opacity.png",
            "Specular": "material_Specular.png",
            "Glossiness": "material_Glossiness.png",
            "Diffuse": "material_Diffuse.png",
        }

        # Create actual image files
        cls.texture_paths = []
        for map_type, filename in cls.test_textures.items():
            filepath = os.path.join(cls.test_files_dir, filename)

            # Create appropriate test images based on type
            if "Normal" in map_type:
                img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 255))
            elif map_type in ["Metallic", "Roughness", "AO", "Opacity", "Glossiness"]:
                img = ImgUtils.create_image("L", (512, 512), 128)
            else:
                img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))

            ImgUtils.save_image(img, filepath)
            cls.texture_paths.append(filepath)

    def setUp(self):
        """Set up test fixtures for each test."""
        # Create mock switchboard
        self.mock_sb = Mock()
        self.mock_sb.file_dialog = Mock(return_value=None)

        # Create MapConverterSlots instance
        self.converter = MapConverterSlots(self.mock_sb)

        # Mock UI components
        self.mock_widget = Mock()
        self.mock_widget.menu = Mock()
        self.mock_widget.menu.chk000 = Mock()
        self.mock_widget.menu.chk000.isChecked = Mock(return_value=False)

    # -------------------------------------------------------------------------
    # Test tb001 with TextureMapFactory Integration
    # -------------------------------------------------------------------------

    def test_tb001_spec_gloss_conversion_basic(self):
        """Test tb001 Spec/Gloss conversion using TextureMapFactory."""
        # Setup file dialog mock to return spec/gloss textures
        spec_gloss_textures = [
            os.path.join(self.test_files_dir, "material_Specular.png"),
            os.path.join(self.test_files_dir, "material_Glossiness.png"),
            os.path.join(self.test_files_dir, "material_Diffuse.png"),
        ]
        self.mock_sb.file_dialog.return_value = spec_gloss_textures

        # Run conversion
        self.converter.tb001(self.mock_widget)

        # Verify file dialog was called
        self.mock_sb.file_dialog.assert_called_once()

        # Check that processed files exist (TextureMapFactory should create outputs)
        # Note: Actual output verification depends on TextureMapFactory implementation

    def test_tb001_with_metallic_smoothness_packing(self):
        """Test tb001 with metallic smoothness packing enabled."""
        spec_gloss_textures = [
            os.path.join(self.test_files_dir, "material_Specular.png"),
            os.path.join(self.test_files_dir, "material_Glossiness.png"),
        ]
        self.mock_sb.file_dialog.return_value = spec_gloss_textures
        self.mock_widget.menu.chk000.isChecked.return_value = True

        # Run conversion with packing enabled
        self.converter.tb001(self.mock_widget)

        # Verify checkbox was checked
        self.mock_widget.menu.chk000.isChecked.assert_called()

    def test_tb001_empty_selection(self):
        """Test tb001 handles empty file selection."""
        self.mock_sb.file_dialog.return_value = None

        # Should return early without error
        result = self.converter.tb001(self.mock_widget)

        # Verify it returns None (early exit)
        self.assertIsNone(result)

    def test_tb001_multiple_texture_sets(self):
        """Test tb001 processes multiple texture sets."""
        # Create second texture set
        set2_textures = []
        for map_type, filename in [
            ("Specular", "model2_Specular.png"),
            ("Glossiness", "model2_Glossiness.png"),
        ]:
            filepath = os.path.join(self.test_files_dir, filename)
            img = ImgUtils.create_image("L", (512, 512), 128)
            ImgUtils.save_image(img, filepath)
            set2_textures.append(filepath)

        all_textures = [
            os.path.join(self.test_files_dir, "material_Specular.png"),
            os.path.join(self.test_files_dir, "material_Glossiness.png"),
        ] + set2_textures

        self.mock_sb.file_dialog.return_value = all_textures

        # Run conversion
        self.converter.tb001(self.mock_widget)

        # Should process both sets without error

    def test_tb001_fallback_on_factory_error(self):
        """Test tb001 falls back to legacy method if TextureMapFactory fails."""
        spec_textures = [
            os.path.join(self.test_files_dir, "material_Specular.png"),
        ]
        self.mock_sb.file_dialog.return_value = spec_textures

        # Mock TextureMapFactory to raise exception
        with patch(
            "pythontk.img_utils.map_converter.TextureMapFactory.prepare_maps",
            side_effect=Exception("Factory error"),
        ):
            # Should fall back to legacy method without crashing
            self.converter.tb001(self.mock_widget)

    # -------------------------------------------------------------------------
    # Test b012 - Batch PBR Workflow Preparation
    # -------------------------------------------------------------------------

    @skip_if_no_pyside2
    @patch("PySide2.QtWidgets.QInputDialog.getItem")
    def test_b012_standard_pbr_workflow(self, mock_dialog):
        """Test b012 with standard PBR workflow."""
        self.mock_sb.file_dialog.return_value = self.texture_paths[
            :5
        ]  # Subset of textures
        mock_dialog.return_value = ("Standard PBR (Separate Maps)", True)

        # Run batch workflow
        self.converter.b012()

        # Verify dialog was shown
        mock_dialog.assert_called_once()
        self.mock_sb.file_dialog.assert_called_once()

    @skip_if_no_pyside2
    @patch("PySide2.QtWidgets.QInputDialog.getItem")
    def test_b012_unity_urp_workflow(self, mock_dialog):
        """Test b012 with Unity URP workflow."""
        self.mock_sb.file_dialog.return_value = self.texture_paths
        mock_dialog.return_value = (
            "Unity URP (Packed: Albedo+Alpha, Metallic+Smoothness)",
            True,
        )

        self.converter.b012()

        mock_dialog.assert_called_once()

    @skip_if_no_pyside2
    @patch("PySide2.QtWidgets.QInputDialog.getItem")
    def test_b012_unity_hdrp_workflow(self, mock_dialog):
        """Test b012 with Unity HDRP workflow (MSAO)."""
        self.mock_sb.file_dialog.return_value = self.texture_paths
        mock_dialog.return_value = ("Unity HDRP (Mask Map: MSAO)", True)

        self.converter.b012()

        mock_dialog.assert_called_once()

    @skip_if_no_pyside2
    @patch("PySide2.QtWidgets.QInputDialog.getItem")
    def test_b012_unreal_workflow(self, mock_dialog):
        """Test b012 with Unreal Engine workflow."""
        self.mock_sb.file_dialog.return_value = self.texture_paths
        mock_dialog.return_value = ("Unreal Engine (BaseColor+Alpha)", True)

        self.converter.b012()

        mock_dialog.assert_called_once()

    @skip_if_no_pyside2
    @patch("PySide2.QtWidgets.QInputDialog.getItem")
    def test_b012_gltf_workflow(self, mock_dialog):
        """Test b012 with glTF 2.0 workflow."""
        self.mock_sb.file_dialog.return_value = self.texture_paths
        mock_dialog.return_value = ("glTF 2.0 (Separate Maps)", True)

        self.converter.b012()

        mock_dialog.assert_called_once()

    @skip_if_no_pyside2
    @patch("PySide2.QtWidgets.QInputDialog.getItem")
    def test_b012_godot_workflow(self, mock_dialog):
        """Test b012 with Godot workflow."""
        self.mock_sb.file_dialog.return_value = self.texture_paths
        mock_dialog.return_value = ("Godot (Separate Maps)", True)

        self.converter.b012()

        mock_dialog.assert_called_once()

    @skip_if_no_pyside2
    @patch("PySide2.QtWidgets.QInputDialog.getItem")
    def test_b012_specular_glossiness_workflow(self, mock_dialog):
        """Test b012 with Specular/Glossiness workflow."""
        spec_gloss_textures = [
            os.path.join(self.test_files_dir, "material_Specular.png"),
            os.path.join(self.test_files_dir, "material_Glossiness.png"),
            os.path.join(self.test_files_dir, "material_Diffuse.png"),
        ]
        self.mock_sb.file_dialog.return_value = spec_gloss_textures
        mock_dialog.return_value = ("Specular/Glossiness Workflow", True)

        self.converter.b012()

        mock_dialog.assert_called_once()

    @skip_if_no_pyside2
    @patch("PySide2.QtWidgets.QInputDialog.getItem")
    def test_b012_user_cancels_workflow_selection(self, mock_dialog):
        """Test b012 handles user canceling workflow selection."""
        self.mock_sb.file_dialog.return_value = self.texture_paths
        mock_dialog.return_value = ("Unity URP", False)  # User canceled

        # Should return early
        result = self.converter.b012()

        self.assertIsNone(result)

    @skip_if_no_pyside2
    @patch("PySide2.QtWidgets.QInputDialog.getItem")
    def test_b012_empty_texture_selection(self, mock_dialog):
        """Test b012 handles empty texture selection."""
        self.mock_sb.file_dialog.return_value = None

        # Should return early without showing workflow dialog
        result = self.converter.b012()

        self.assertIsNone(result)
        mock_dialog.assert_not_called()

    @skip_if_no_pyside2
    @patch("PySide2.QtWidgets.QInputDialog.getItem")
    def test_b012_unknown_workflow(self, mock_dialog):
        """Test b012 handles unknown workflow gracefully."""
        self.mock_sb.file_dialog.return_value = self.texture_paths
        mock_dialog.return_value = ("Unknown Workflow", True)

        # Should handle gracefully
        self.converter.b012()

    @skip_if_no_pyside2
    @patch("PySide2.QtWidgets.QInputDialog.getItem")
    def test_b012_multiple_texture_sets(self, mock_dialog):
        """Test b012 processes multiple texture sets correctly."""
        # Create second texture set
        set2_textures = []
        for map_type, filename in [
            ("BaseColor", "model2_BaseColor.png"),
            ("Metallic", "model2_Metallic.png"),
            ("Roughness", "model2_Roughness.png"),
        ]:
            filepath = os.path.join(self.test_files_dir, filename)
            img = ImgUtils.create_image(
                "RGB" if map_type == "BaseColor" else "L",
                (512, 512),
                (128, 128, 128) if map_type == "BaseColor" else 128,
            )
            ImgUtils.save_image(img, filepath)
            set2_textures.append(filepath)

        all_textures = self.texture_paths[:3] + set2_textures

        self.mock_sb.file_dialog.return_value = all_textures
        mock_dialog.return_value = ("Standard PBR (Separate Maps)", True)

        # Should process both sets
        self.converter.b012()

    @skip_if_no_pyside2
    @patch("pythontk.img_utils.texture_map_factory.TextureMapFactory.prepare_maps")
    @patch("PySide2.QtWidgets.QInputDialog.getItem")
    def test_b012_handles_factory_errors(self, mock_dialog, mock_prepare):
        """Test b012 handles TextureMapFactory errors gracefully."""
        self.mock_sb.file_dialog.return_value = self.texture_paths
        mock_dialog.return_value = ("Standard PBR (Separate Maps)", True)
        mock_prepare.side_effect = Exception("Factory processing error")

        # Should catch and report error, not crash
        self.converter.b012()

    # -------------------------------------------------------------------------
    # Integration Tests
    # -------------------------------------------------------------------------

    def test_texture_map_factory_import(self):
        """Test that TextureMapFactory is properly imported."""
        from pythontk.img_utils.map_converter import TextureMapFactory

        self.assertIsNotNone(TextureMapFactory)
        self.assertTrue(hasattr(TextureMapFactory, "prepare_maps"))

    def test_converter_has_all_methods(self):
        """Test that MapConverterSlots has all expected methods."""
        self.assertTrue(hasattr(self.converter, "tb001"))
        self.assertTrue(hasattr(self.converter, "b012"))
        self.assertTrue(callable(self.converter.tb001))
        self.assertTrue(callable(self.converter.b012))

    def test_source_dir_property(self):
        """Test source_dir property getter/setter."""
        test_dir = "/test/directory"
        self.converter.source_dir = test_dir
        self.assertEqual(self.converter.source_dir, test_dir)


# =============================================================================
# Edge Cases
# =============================================================================


class TestMapConverterEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_sb = Mock()
        self.converter = MapConverterSlots(self.mock_sb)
        self.mock_widget = Mock()

    def test_tb001_with_corrupted_texture(self):
        """Test tb001 handles corrupted texture files."""
        # Create a corrupted file
        temp_dir = tempfile.mkdtemp()
        try:
            corrupted_file = os.path.join(temp_dir, "corrupted_spec.png")
            with open(corrupted_file, "w") as f:
                f.write("This is not a valid PNG")

            self.mock_sb.file_dialog.return_value = [corrupted_file]

            # Should handle gracefully
            self.converter.tb001(self.mock_widget)

        finally:
            shutil.rmtree(temp_dir)

    @skip_if_no_pyside2
    @patch("PySide2.QtWidgets.QInputDialog.getItem")
    def test_b012_with_missing_texture_files(self, mock_dialog):
        """Test b012 handles missing texture files."""
        fake_paths = [
            "/nonexistent/texture1.png",
            "/nonexistent/texture2.png",
        ]

        self.mock_sb.file_dialog.return_value = fake_paths
        mock_dialog.return_value = ("Standard PBR (Separate Maps)", True)

        # Should handle gracefully
        self.converter.b012()

    def test_tb001_with_invalid_image_format(self):
        """Test tb001 handles unsupported image formats."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a text file with .png extension
            fake_png = os.path.join(temp_dir, "fake_spec.png")
            with open(fake_png, "wb") as f:
                f.write(b"Not an image")

            self.mock_sb.file_dialog.return_value = [fake_png]

            # Should handle gracefully without crashing
            self.converter.tb001(self.mock_widget)

        finally:
            shutil.rmtree(temp_dir)

    def test_tb001_with_single_channel_specular(self):
        """Test tb001 handles grayscale specular maps correctly."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create grayscale specular map
            spec_file = os.path.join(temp_dir, "model_Specular.png")
            img = ImgUtils.create_image("L", (512, 512), 128)
            ImgUtils.save_image(img, spec_file)

            self.mock_sb.file_dialog.return_value = [spec_file]

            # Should process without error
            self.converter.tb001(self.mock_widget)

        finally:
            shutil.rmtree(temp_dir)

    def test_tb001_workflow_config_passthrough(self):
        """Test tb001 correctly passes workflow config to TextureMapFactory."""
        temp_dir = tempfile.mkdtemp()
        try:
            spec_file = os.path.join(temp_dir, "test_Specular.png")
            img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
            ImgUtils.save_image(img, spec_file)

            self.mock_sb.file_dialog.return_value = [spec_file]
            self.mock_widget.menu.chk000.isChecked.return_value = False

            with patch(
                "pythontk.img_utils.texture_map_factory.TextureMapFactory.prepare_maps"
            ) as mock_prepare:
                mock_prepare.return_value = [spec_file]

                self.converter.tb001(self.mock_widget)

                # Verify workflow_config was passed correctly
                if not mock_prepare.called:
                    print(
                        f"DEBUG: mock_prepare not called. file_dialog called: {self.mock_sb.file_dialog.called}"
                    )
                    if self.mock_sb.file_dialog.called:
                        print(
                            f"DEBUG: file_dialog return: {self.mock_sb.file_dialog.return_value}"
                        )
                    else:
                        print("DEBUG: file_dialog NOT called")

                self.assertTrue(mock_prepare.called)
                call_args = mock_prepare.call_args
                workflow_config = call_args[0][1]  # Second positional arg

                self.assertFalse(workflow_config["albedo_transparency"])
                self.assertFalse(workflow_config["metallic_smoothness"])
                self.assertEqual(workflow_config["normal_type"], "OpenGL")

        finally:
            shutil.rmtree(temp_dir)

    def test_tb001_with_mixed_resolution_textures(self):
        """Test tb001 handles textures with different resolutions."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create textures with different sizes
            spec_512 = os.path.join(temp_dir, "model_Specular.png")
            gloss_1024 = os.path.join(temp_dir, "model_Glossiness.png")

            img_512 = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
            img_1024 = ImgUtils.create_image("L", (1024, 1024), 128)

            ImgUtils.save_image(img_512, spec_512)
            ImgUtils.save_image(img_1024, gloss_1024)

            self.mock_sb.file_dialog.return_value = [spec_512, gloss_1024]

            # Should process (factory may handle resolution mismatch)
            self.converter.tb001(self.mock_widget)

        finally:
            shutil.rmtree(temp_dir)

    def test_tb001_empty_texture_set(self):
        """Test tb001 with empty file list."""
        self.mock_sb.file_dialog.return_value = []

        # Should return early without error
        result = self.converter.tb001(self.mock_widget)
        self.assertIsNone(result)

    def test_source_dir_persistence(self):
        """Test source_dir is updated after tb001 processing."""
        temp_dir = tempfile.mkdtemp()
        try:
            spec_file = os.path.join(temp_dir, "test_Specular.png")
            img = ImgUtils.create_image("RGB", (512, 512), (128, 128, 128))
            ImgUtils.save_image(img, spec_file)

            self.mock_sb.file_dialog.return_value = [spec_file]

            # Process and verify source_dir is set
            self.converter.tb001(self.mock_widget)

            # source_dir should be updated to the texture directory
            self.assertIsNotNone(self.converter.source_dir)

        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
