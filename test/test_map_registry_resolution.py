#!/usr/bin/python
# coding=utf-8
"""Tests for MapRegistry.is_resolution_critical / get_resolution_critical_types."""
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


if __name__ == "__main__":
    unittest.main()
