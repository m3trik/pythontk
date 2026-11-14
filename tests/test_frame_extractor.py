#!/usr/bin/python
# coding=utf-8
"""Unit tests for pythontk FrameExtractor.

Focused regression coverage for argument validation on
``FrameExtractor.extract_frames`` -- those tests need neither cv2 nor a
real video file, because the ``step`` guard runs before any cv2 use -- and
for frames written to a folder ``cv2.imwrite`` cannot open (these write a
short clip, so they need cv2).

Run with:
    python -m pytest test_frame_extractor.py -v
"""

import os
import unittest

from pythontk import FrameExtractor, TempArtifacts

from conftest import BaseTestCase

try:  # after conftest, which turns OpenEXR on before cv2 loads
    import cv2
    import numpy as np
except ImportError:
    cv2 = None


class FrameExtractorTest(BaseTestCase):
    """FrameExtractor argument-validation tests."""

    def test_extract_frames_step_zero_raises_value_error(self):
        """step=0 must raise ValueError (previously a swallowed
        ZeroDivisionError from ``count % step`` that returned [])."""
        with self.assertRaises(ValueError):
            FrameExtractor().extract_frames("nonexistent_clip.mp4", "out", step=0)

    def test_extract_frames_negative_step_raises_value_error(self):
        """Any step < 1 is invalid and must raise ValueError."""
        with self.assertRaises(ValueError):
            FrameExtractor().extract_frames("nonexistent_clip.mp4", "out", step=-3)

    def test_extract_frames_valid_step_does_not_raise_value_error(self):
        """A valid step must not trip the guard; a nonexistent input
        degrades to an empty list rather than raising ValueError."""
        result = FrameExtractor().extract_frames(
            "definitely_missing_clip.mp4", "out", step=5
        )
        self.assertEqual(result, [])


@unittest.skipUnless(cv2 is not None, "cv2 not available")
class FrameExtractorOutputPathTest(BaseTestCase):
    """Frames written to a folder ``cv2.imwrite`` cannot open by itself.

    On Windows ``cv2.imwrite`` opens its file by a narrow (ANSI) path, so an
    output folder named ``résumé ★`` got no frames: every write returned False
    and the extraction reported none. Fixed: 2026-10-05.
    """

    def setUp(self):
        artifacts = TempArtifacts("frame_extractor_path", policy="scoped")
        self.addCleanup(artifacts.cleanup)
        root = artifacts.dir_path()
        self.video = os.path.join(root, "clip.avi")
        writer = cv2.VideoWriter(
            self.video, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32)
        )
        for i in range(6):
            frame = np.full((32, 32, 3), 40 * i, np.uint8)
            frame[::4] = 255  # edges: every frame has a sharpness to rank
            writer.write(frame)
        writer.release()
        self.out = os.path.join(root, "résumé ★")

    def test_every_step_frame_is_written(self):
        saved = FrameExtractor().extract_frames(self.video, self.out, step=2)
        self.assertEqual(len(saved), 3)
        self.assertEqual(
            sorted(os.listdir(self.out)), [os.path.basename(p) for p in saved]
        )

    def test_the_sharpest_frame_of_each_window_is_written(self):
        saved = FrameExtractor().extract_frames_sharpest(
            self.video, self.out, window_sec=0.3
        )
        self.assertTrue(saved, "no frame written")
        self.assertEqual(
            sorted(os.listdir(self.out)), [os.path.basename(p) for p in saved]
        )


if __name__ == "__main__":
    unittest.main()
