#!/usr/bin/python
# coding=utf-8
"""Tests for MapRegistry.is_resolution_critical / get_resolution_critical_types,
and estimate_gpu_bytes (what a map costs once loaded, by its type)."""

import unittest

from pythontk import MapRegistry

from conftest import BaseTestCase


class ResolutionCriticalTest(BaseTestCase):
    def setUp(self):
        self.reg = MapRegistry()

    def test_color_and_normal_maps_are_critical(self):
        for name in (
            "Base_Color",
            "Diffuse",
            "Albedo_Transparency",
            "Normal",
            "Normal_OpenGL",
            "Normal_DirectX",
            "Emissive",
        ):
            self.assertTrue(
                self.reg.is_resolution_critical(name),
                f"{name} should be flagged resolution_critical",
            )

    def test_secondary_maps_are_not_critical(self):
        for name in (
            "Roughness",
            "Metallic",
            "Smoothness",
            "Ambient_Occlusion",
            "Height",
            "Bump",
            "ORM",
            "MSAO",
            "MRAO",
            "Metallic_Smoothness",
            "Opacity",
        ):
            self.assertFalse(
                self.reg.is_resolution_critical(name),
                f"{name} should not be flagged resolution_critical",
            )

    def test_unknown_name_defaults_to_critical(self):
        # Fail-safe: unrecognised types are treated as critical so callers
        # don't silently downscale maps the registry doesn't know about.
        self.assertTrue(self.reg.is_resolution_critical("NotARealMap"))
        self.assertTrue(self.reg.is_resolution_critical(None))

    def test_get_resolution_critical_types_matches_per_map_flag(self):
        listed = set(self.reg.get_resolution_critical_types())
        self.assertGreater(len(listed), 0)
        for name in listed:
            self.assertTrue(self.reg.is_resolution_critical(name))


class EstimateGpuBytesTest(BaseTestCase):
    """The block format follows the map type's own mode, with a full mip chain."""

    MIPS = 4.0 / 3.0

    def setUp(self):
        self.reg = MapRegistry()

    def test_single_channel_and_rgb_maps_take_half_a_byte(self):
        for name in ("Roughness", "Ambient_Occlusion", "Base_Color", "Emissive"):
            compressed, raw = self.reg.estimate_gpu_bytes(1024, 1024, name)
            self.assertAlmostEqual(compressed, 1024 * 1024 * 0.5 * self.MIPS, msg=name)
            self.assertAlmostEqual(raw, 1024 * 1024 * 4 * self.MIPS, msg=name)

    def test_normals_alpha_and_unknown_maps_take_a_byte(self):
        for name in (
            "Normal",
            "Normal_OpenGL",
            "Bent_Normal",
            "Albedo_Transparency",
            None,
            "NotARealMap",
        ):
            compressed, _ = self.reg.estimate_gpu_bytes(1024, 1024, name)
            self.assertAlmostEqual(compressed, 1024 * 1024 * self.MIPS, msg=name)

    def test_a_packed_map_without_a_mode_follows_its_channel_layout(self):
        """``mode=None`` ("keep the packer's natural mode") is judged by the
        canonical channels: MRAO's R/G/B costs what ORM's does; MSAO carries
        smoothness in A. Regression: every mode-less type took the alpha tier."""
        orm, _ = self.reg.estimate_gpu_bytes(2048, 2048, "ORM")
        mrao, _ = self.reg.estimate_gpu_bytes(2048, 2048, "MRAO")
        msao, _ = self.reg.estimate_gpu_bytes(2048, 2048, "MSAO")
        self.assertAlmostEqual(mrao, orm)
        self.assertAlmostEqual(msao, orm * 2)

    def test_tiles_multiply(self):
        one, _ = self.reg.estimate_gpu_bytes(512, 512, "Roughness")
        four, _ = self.reg.estimate_gpu_bytes(512, 512, "Roughness", tiles=4)
        self.assertAlmostEqual(four, one * 4)


if __name__ == "__main__":
    unittest.main()
