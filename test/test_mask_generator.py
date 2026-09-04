# !/usr/bin/python
# coding=utf-8
"""Tests for ``img_utils/mask_generator.py``.

``rembg`` is an optional native dependency and is not installed here, so
these cover the capability gate rather than a real matting run.
"""
import unittest
from unittest import mock

import pythontk.img_utils.mask_generator as mod

from conftest import BaseTestCase


class MaskGeneratorAvailabilityTest(BaseTestCase):
    """The optional-dependency gate behind :meth:`MaskGenerator.is_available`."""

    @staticmethod
    def _with_rembg():
        """Patch in a rembg that loaded, leaving only the PIL half under test."""
        return mock.patch.multiple(
            mod, REMBG_AVAILABLE=True, new_session=lambda name: object()
        )

    def test_unavailable_without_rembg(self):
        """The documented degraded posture: no rembg, no masks, no raise."""
        with mock.patch.object(mod, "REMBG_AVAILABLE", False):
            self.assertFalse(mod.MaskGenerator().is_available())

    @unittest.skipUnless(mod.Image is not None, "PIL not available")
    def test_late_provisioned_pillow_reports_available(self):
        """A stale ``PIL_AVAILABLE`` must not report a usable PIL as missing.

        In Blender, ``import pythontk`` runs before ``ensure_image_deps()``
        can provision Pillow, so this module caches ``Image = None``.
        blendertk's ``_rebind_pil_globals`` repairs that binding afterwards
        -- but its own docstring scopes the repair to "a name the module
        itself set to ``None``", so the ``PIL_AVAILABLE`` bool stays False
        forever. This pins the post-repair state: bool stale, ``Image`` live.
        Gating on the bool made ``generate_masks`` log "PIL=False" and return
        [] with a perfectly usable Pillow on ``sys.path``.
        """
        with self._with_rembg():
            gen = mod.MaskGenerator()
            with mock.patch.object(mod, "PIL_AVAILABLE", False):
                self.assertTrue(gen.is_available())

    @unittest.skipUnless(mod.Image is not None, "PIL not available")
    def test_generate_masks_reports_the_live_pil_state(self):
        """The unavailable log must not quote the stale bool either -- it is
        the line an operator reads to decide whether to install anything."""
        with mock.patch.object(mod, "REMBG_AVAILABLE", False), mock.patch.object(
            mod, "PIL_AVAILABLE", False
        ):
            gen = mod.MaskGenerator()
            with self.assertLogs(mod.logger, level="ERROR") as caught:
                self.assertEqual(gen.generate_masks("nowhere", "nowhere_out"), [])
        self.assertIn("PIL=True", "".join(caught.output))


if __name__ == "__main__":
    unittest.main(verbosity=2)
