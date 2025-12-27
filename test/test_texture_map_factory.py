#!/usr/bin/python
# coding=utf-8
"""
Refactored tests for TextureMapFactory using the public API (Strategy Pattern).
"""
import os
import tempfile
import shutil
import unittest
from pythontk import ImgUtils
from pythontk.img_utils.texture_map_factory import TextureMapFactory


class TestTextureMapFactoryRefactored(unittest.TestCase):
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
        results = TextureMapFactory.prepare_maps(
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
        results = TextureMapFactory.prepare_maps(
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
        results = TextureMapFactory.prepare_maps(
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

    def test_prepare_maps_unreal_engine(self):
        """Test Unreal Engine workflow (ORM + DirectX Normal)."""
        config = {"orm_map": True, "normal_type": "DirectX"}
        results = TextureMapFactory.prepare_maps(
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
        results = TextureMapFactory.prepare_maps(
            subset, output_dir=self.output_dir, callback=lambda *args: None, **config
        )
        result_names = [os.path.basename(p) for p in results]

        self.assertTrue(any("Albedo_Transparency" in n for n in result_names))
        self.assertFalse(any("Base_Color" in n for n in result_names))


if __name__ == "__main__":
    unittest.main()
