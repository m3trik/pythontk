#!/usr/bin/python
# coding=utf-8
"""Unit tests for pythontk FrameExtractor.

Focused regression coverage for argument validation on
``FrameExtractor.extract_frames``. These tests do not require cv2 or a
real video file because the ``step`` guard runs before any cv2 use.

Run with:
    python -m pytest test_frame_extractor.py -v
"""
import unittest

from pythontk import FrameExtractor

from conftest import BaseTestCase


class FrameExtractorTest(BaseTestCase):
    """FrameExtractor argument-validation tests."""

    def test_extract_frames_step_zero_raises_value_error(self):
        """step=0 must raise ValueError (previously a swallowed
        ZeroDivisionError from ``count % step`` that returned [])."""
        with self.assertRaises(ValueError):
            FrameExtractor().extract_frames(
                "nonexistent_clip.mp4", "out", step=0
            )

    def test_extract_frames_negative_step_raises_value_error(self):
        """Any step < 1 is invalid and must raise ValueError."""
        with self.assertRaises(ValueError):
            FrameExtractor().extract_frames(
                "nonexistent_clip.mp4", "out", step=-3
            )

    def test_extract_frames_valid_step_does_not_raise_value_error(self):
        """A valid step must not trip the guard; a nonexistent input
        degrades to an empty list rather than raising ValueError."""
        result = FrameExtractor().extract_frames(
            "definitely_missing_clip.mp4", "out", step=5
        )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
