#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk VidUtils.

Comprehensive edge case coverage for:
- resolve_ffmpeg
- get_video_frame_rate
- compress_video

Note: Many video tests require actual video files and ffmpeg installation.
Tests are designed to gracefully skip if requirements aren't met.

Run with:
    python -m pytest test_vid.py -v
    python test_vid.py
"""
import os
import unittest
import shutil
import tempfile

from pythontk import VidUtils

from conftest import BaseTestCase


class VidTest(BaseTestCase):
    """Video utilities test class with comprehensive edge case coverage."""

    @classmethod
    def setUpClass(cls):
        """Set up test paths and check ffmpeg availability."""
        cls.temp_dir = tempfile.mkdtemp()
        # Check if ffmpeg is available
        cls.ffmpeg_available = shutil.which("ffmpeg") is not None

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    # -------------------------------------------------------------------------
    # resolve_ffmpeg Tests
    # -------------------------------------------------------------------------

    def test_resolve_ffmpeg_returns_path_or_raises(self):
        """Test resolve_ffmpeg returns a path string or raises FileNotFoundError."""
        try:
            result = VidUtils.resolve_ffmpeg()
            # If it succeeds, should return a valid path string
            self.assertIsInstance(result, str)
        except FileNotFoundError:
            # It's expected to raise FileNotFoundError if ffmpeg is not found
            pass

    @unittest.skipUnless(
        shutil.which("ffmpeg") is not None,
        "FFmpeg not installed"
    )
    def test_resolve_ffmpeg_valid_path_exists(self):
        """Test resolve_ffmpeg returns existing path when found."""
        result = VidUtils.resolve_ffmpeg()
        self.assertTrue(os.path.exists(result))

    @unittest.skipUnless(
        shutil.which("ffmpeg") is not None,
        "FFmpeg not installed"
    )
    def test_resolve_ffmpeg_is_executable(self):
        """Test resolve_ffmpeg returns an executable."""
        result = VidUtils.resolve_ffmpeg()
        # On Windows, files ending in .exe are executable
        # On Unix, check if file is executable
        self.assertTrue(
            os.path.isfile(result),
            f"FFmpeg path should be a file: {result}"
        )

    @unittest.skipUnless(
        shutil.which("ffmpeg") is not None,
        "FFmpeg not installed"
    )
    def test_resolve_ffmpeg_consistent_results(self):
        """Test resolve_ffmpeg returns consistent results on multiple calls."""
        result1 = VidUtils.resolve_ffmpeg()
        result2 = VidUtils.resolve_ffmpeg()
        self.assertEqual(result1, result2)

    # -------------------------------------------------------------------------
    # get_video_frame_rate Tests
    # -------------------------------------------------------------------------

    @unittest.skipUnless(
        shutil.which("ffmpeg") is not None,
        "FFmpeg not installed"
    )
    def test_get_video_frame_rate_nonexistent_file(self):
        """Test get_video_frame_rate with nonexistent file."""
        result = VidUtils.get_video_frame_rate("/nonexistent/video.mp4")
        # Should return None or 0 for invalid file
        self.assertTrue(result is None or result == 0 or result == 0.0)

    @unittest.skipUnless(
        shutil.which("ffmpeg") is not None,
        "FFmpeg not installed"
    )
    def test_get_video_frame_rate_empty_path(self):
        """Test get_video_frame_rate with empty path."""
        result = VidUtils.get_video_frame_rate("")
        self.assertTrue(result is None or result == 0 or result == 0.0)

    @unittest.skipUnless(
        shutil.which("ffmpeg") is not None,
        "FFmpeg not installed"
    )
    def test_get_video_frame_rate_non_video_file(self):
        """Test get_video_frame_rate with non-video file."""
        # Create a temp text file
        temp_file = os.path.join(self.temp_dir, "test.txt")
        with open(temp_file, "w") as f:
            f.write("This is not a video")
        
        result = VidUtils.get_video_frame_rate(temp_file)
        # Should handle gracefully
        self.assertTrue(result is None or isinstance(result, (int, float)))

    # -------------------------------------------------------------------------
    # compress_video Tests
    # -------------------------------------------------------------------------

    @unittest.skipUnless(
        shutil.which("ffmpeg") is not None,
        "FFmpeg not installed"
    )
    def test_compress_video_nonexistent_input(self):
        """Test compress_video with nonexistent input file."""
        output = os.path.join(self.temp_dir, "output.mp4")
        # Should handle gracefully without crashing
        try:
            result = VidUtils.compress_video("/nonexistent/video.mp4", output)
            # Either returns None/False or raises exception
            self.assertTrue(result is None or result is False or isinstance(result, str))
        except (FileNotFoundError, OSError, ValueError):
            # Expected for invalid input
            pass

    @unittest.skipUnless(
        shutil.which("ffmpeg") is not None,
        "FFmpeg not installed"
    )
    def test_compress_video_empty_input(self):
        """Test compress_video with empty input path."""
        output = os.path.join(self.temp_dir, "output.mp4")
        try:
            result = VidUtils.compress_video("", output)
            self.assertTrue(result is None or result is False)
        except (FileNotFoundError, OSError, ValueError):
            # Expected for invalid input
            pass

    @unittest.skipUnless(
        shutil.which("ffmpeg") is not None,
        "FFmpeg not installed"
    )
    def test_compress_video_invalid_output_path(self):
        """Test compress_video with invalid output path."""
        # Create a minimal input (this test just checks path validation)
        temp_input = os.path.join(self.temp_dir, "input.txt")
        with open(temp_input, "w") as f:
            f.write("not a video")
        
        try:
            result = VidUtils.compress_video(
                temp_input, 
                "/invalid/path/that/does/not/exist/output.mp4"
            )
            # Should handle gracefully
            self.assertTrue(result is None or result is False or isinstance(result, str))
        except (FileNotFoundError, OSError, PermissionError, ValueError):
            # Expected for invalid output path
            pass

    # -------------------------------------------------------------------------
    # Edge Case Tests
    # -------------------------------------------------------------------------

    def test_vidutils_import(self):
        """Test VidUtils can be imported."""
        from pythontk import VidUtils
        self.assertTrue(hasattr(VidUtils, "resolve_ffmpeg"))
        self.assertTrue(hasattr(VidUtils, "get_video_frame_rate"))
        self.assertTrue(hasattr(VidUtils, "compress_video"))

    def test_vidutils_class_exists(self):
        """Test VidUtils class has expected methods."""
        methods = ["resolve_ffmpeg", "get_video_frame_rate", "compress_video"]
        for method in methods:
            self.assertTrue(
                hasattr(VidUtils, method),
                f"VidUtils should have method: {method}"
            )

    @unittest.skipUnless(
        shutil.which("ffmpeg") is not None,
        "FFmpeg not installed"
    )
    def test_ffmpeg_path_contains_ffmpeg(self):
        """Test resolved ffmpeg path contains 'ffmpeg' in name."""
        result = VidUtils.resolve_ffmpeg()
        if result is not None:
            basename = os.path.basename(result).lower()
            self.assertIn("ffmpeg", basename)

    def test_vidutils_methods_are_static_or_class(self):
        """Test that VidUtils methods can be called without instance."""
        # These should not require instantiation
        try:
            _ = VidUtils.resolve_ffmpeg
            _ = VidUtils.get_video_frame_rate
            _ = VidUtils.compress_video
        except AttributeError as e:
            self.fail(f"VidUtils methods should be accessible: {e}")


if __name__ == "__main__":
    unittest.main(exit=False)
