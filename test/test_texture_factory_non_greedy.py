#!/usr/bin/python
# coding=utf-8
"""
Test non-greedy map generation in TextureMapFactory.
"""
import os
import tempfile
import shutil
import unittest
from pythontk import ImgUtils
from pythontk.img_utils.texture_map_factory import TextureMapFactory


class TestTextureFactoryNonGreedy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="texture_factory_nongreedy_test_")
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
            "Metallic": "test_material_Metallic.png",
            "Smoothness": "test_material_Smoothness.png",
            "Ambient_Occlusion": "test_material_AO.png",
            "Roughness": "test_material_Roughness.png",
        }
        cls.texture_paths = []
        for map_type, filename in cls.test_textures.items():
            filepath = os.path.join(cls.test_files_dir, filename)
            img = ImgUtils.create_image("L", (64, 64), 128)
            ImgUtils.save_image(img, filepath)
            cls.texture_paths.append(filepath)

    def setUp(self):
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_default_behavior_is_pass_through(self):
        """Test that default config (no flags) does NOT generate packed maps."""
        config = {"rename": True}

        results = TextureMapFactory.prepare_maps(
            self.texture_paths,
            output_dir=self.output_dir,
            callback=lambda *args: None,
            **config,
        )
        result_names = [os.path.basename(p) for p in results]

        # Should NOT have packed maps
        self.assertFalse(
            any("Metallic_Smoothness" in n for n in result_names),
            "Greedy Metallic_Smoothness generated",
        )
        self.assertFalse(
            any("MSAO" in n for n in result_names), "Greedy MSAO generated"
        )
        self.assertFalse(any("ORM" in n for n in result_names), "Greedy ORM generated")

        # Should have individual maps (Standard PBR normalization converts Smoothness to Roughness)
        self.assertTrue(any("Metallic" in n for n in result_names))
        self.assertTrue(any("Roughness" in n for n in result_names))
        self.assertTrue(any("Ambient_Occlusion" in n for n in result_names))

    def test_explicit_enable_metallic_smoothness(self):
        """Test that explicit flag enables generation."""
        config = {"metallic_smoothness": True, "rename": True}

        results = TextureMapFactory.prepare_maps(
            self.texture_paths,
            output_dir=self.output_dir,
            callback=lambda *args: None,
            **config,
        )
        result_names = [os.path.basename(p) for p in results]

        self.assertTrue(any("Metallic_Smoothness" in n for n in result_names))

    def test_explicit_enable_mask_map(self):
        """Test that explicit flag enables generation."""
        config = {"mask_map": True, "rename": True}

        results = TextureMapFactory.prepare_maps(
            self.texture_paths,
            output_dir=self.output_dir,
            callback=lambda *args: None,
            **config,
        )
        result_names = [os.path.basename(p) for p in results]

        self.assertTrue(any("MSAO" in n for n in result_names))


if __name__ == "__main__":
    unittest.main()
