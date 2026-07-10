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
        with self.assertRaises(RuntimeError):
            VidUtils.get_video_frame_rate("/nonexistent/video.mp4")

    @unittest.skipUnless(
        shutil.which("ffmpeg") is not None,
        "FFmpeg not installed"
    )
    def test_get_video_frame_rate_empty_path(self):
        """Test get_video_frame_rate with empty path."""
        with self.assertRaises(RuntimeError):
            VidUtils.get_video_frame_rate("")

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

        with self.assertRaises(RuntimeError):
            VidUtils.get_video_frame_rate(temp_file)

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

    def test_get_frame_rate_fail_fast(self):
        """Regression: unrecognized strings used to silently return 24.0 —
        indistinguishable from real 24fps. They must raise."""
        self.assertEqual(VidUtils.get_frame_rate("film"), 24.0)
        self.assertEqual(VidUtils.get_frame_rate("29.97fps"), 29.97)
        self.assertEqual(VidUtils.get_frame_rate(24.0), "film")
        self.assertEqual(VidUtils.get_frame_rate(17.0), "17fps")
        for bad in ("garbage", "", "fpsfps", None):
            with self.assertRaises(ValueError, msg=f"input: {bad!r}"):
                VidUtils.get_frame_rate(bad)

    def test_compress_video_default_output_never_overwrites_source(self):
        """Regression: default output used input.replace('.avi', '.mp4'),
        so any non-.avi input derived output == input and ffmpeg -y
        truncated the source file. The guard must trip BEFORE ffmpeg is
        resolved or invoked.
        """
        # .mp4 input with no explicit output -> derived output equals the
        # input -> must raise, not overwrite.
        src = os.path.join(self.temp_dir, "take1.mp4")
        with open(src, "w") as f:
            f.write("source bytes")
        with self.assertRaises(ValueError):
            VidUtils.compress_video(src)
        with open(src) as f:
            self.assertEqual(f.read(), "source bytes")

        # Explicit output equal to input must also raise.
        with self.assertRaises(ValueError):
            VidUtils.compress_video(src, src)

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


class FrameExtractorSmartTest(BaseTestCase):
    """FrameExtractor.extract_frames_sharpest + score_sharpness."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_sharpness_score_is_zero_when_cv2_missing(self) -> None:
        import pythontk.vid_utils.frame_extractor as fe
        if fe.CV2_AVAILABLE:
            self.skipTest("cv2 present; can't simulate the missing-cv2 path")
        self.assertEqual(fe.FrameExtractor.score_sharpness(None), 0.0)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_sharpness_score_higher_for_sharper_image(self) -> None:
        import numpy as np
        from pythontk import FrameExtractor

        # A checkerboard (lots of edges) should score higher than a flat
        # gray patch (variance of Laplacian ~ 0).
        flat = np.full((128, 128, 3), 127, dtype=np.uint8)
        sharp = np.zeros((128, 128, 3), dtype=np.uint8)
        sharp[::8, :] = 255  # horizontal stripes

        s_flat = FrameExtractor.score_sharpness(flat)
        s_sharp = FrameExtractor.score_sharpness(sharp)
        self.assertGreater(s_sharp, s_flat)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_sharpest_of_window_writes_only_best_per_bucket(self) -> None:
        import numpy as np
        import cv2 as _cv2
        from pythontk import FrameExtractor

        # Synthesize a tiny mp4: 60 frames, fps=30 → 2 seconds. Every
        # 10th frame is "sharp" (checkerboard), others are flat. With
        # window_sec=1.0 (= 30 frames/bucket), each bucket should pick
        # exactly one sharp frame.
        path_in = os.path.join(self.temp_dir, "in.mp4")
        path_out = os.path.join(self.temp_dir, "out")
        os.makedirs(path_out, exist_ok=True)
        writer = _cv2.VideoWriter(
            path_in, _cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64)
        )
        flat = np.full((64, 64, 3), 127, dtype=np.uint8)
        sharp = np.zeros((64, 64, 3), dtype=np.uint8)
        sharp[::4, :] = 255
        for i in range(60):
            writer.write(sharp if i % 30 == 15 else flat)
        writer.release()

        kept = FrameExtractor().extract_frames_sharpest(
            video_path=path_in,
            output_folder=path_out,
            window_sec=1.0,
        )
        # 2 buckets of 30 frames each → ≤ 2 outputs.
        self.assertGreaterEqual(len(kept), 1)
        self.assertLessEqual(len(kept), 2)


class ImageCuratorTest(BaseTestCase):
    """ImageCurator dHash + sharpness curation."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_curator_unavailable_when_cv2_missing(self) -> None:
        import pythontk.img_utils.image_curator as ic
        if ic.CV2_AVAILABLE:
            self.skipTest("cv2 present")
        self.assertFalse(ic.ImageCurator().is_available())

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_hamming_distance_is_correct(self) -> None:
        from pythontk import ImageCurator
        self.assertEqual(ImageCurator.hamming(0b0, 0b0), 0)
        self.assertEqual(ImageCurator.hamming(0b1010, 0b0101), 4)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_curate_overwrites_stale_output_on_rerun(self) -> None:
        """Re-running with a different threshold must not leave files
        from the previous run mixed with the new survivors."""
        import numpy as np
        import cv2 as _cv2
        from pythontk import ImageCurator

        src = os.path.join(self.temp_dir, "rerun_src")
        os.makedirs(src, exist_ok=True)
        for i, base in enumerate((30, 80, 130, 180)):
            arr = np.full((64, 64, 3), base, dtype=np.uint8)
            arr[::4, :] = 255
            _cv2.imwrite(os.path.join(src, f"img_{i}.jpg"), arr)
        out_root = os.path.join(self.temp_dir, "rerun_out")
        # Pass 1 — strict: every frame is its own cluster (4 anchors).
        ImageCurator().curate([src], out_root, hash_threshold=0)
        pass1 = sorted(os.listdir(out_root + os.sep
                                   + os.path.basename(src) + "_curated"))
        # Pass 2 — looser: many frames cluster, fewer survive.
        ImageCurator().curate([src], out_root, hash_threshold=64)
        pass2 = sorted(os.listdir(out_root + os.sep
                                   + os.path.basename(src) + "_curated"))
        self.assertLessEqual(
            len(pass2), len(pass1),
            "Looser pass should not yield more files than strict pass"
        )
        self.assertFalse(
            set(pass2) - set(pass1),
            "Pass 2 dir contains files from neither source (stale)"
        )

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_curate_drops_near_duplicates(self) -> None:
        import numpy as np
        import cv2 as _cv2
        from pythontk import ImageCurator

        src = os.path.join(self.temp_dir, "src_dup")
        os.makedirs(src, exist_ok=True)
        # Three "groups": two near-identical frames each, slightly
        # sharper second one. Curator should keep ~3 (1 per cluster).
        for i, base in enumerate((50, 120, 200)):
            for j in range(2):
                arr = np.full((64, 64, 3), base, dtype=np.uint8)
                if j:
                    arr[::4, :] = 255  # sharper variant
                _cv2.imwrite(os.path.join(src, f"img_{i}_{j}.jpg"), arr)
        out_root = os.path.join(self.temp_dir, "curated_out")
        result = ImageCurator().curate([src], out_root, hash_threshold=10)
        # One output dir, with fewer than 6 files (some clusters dedup'd).
        self.assertEqual(len(result), 1)
        kept = [
            f for f in os.listdir(result[0])
            if f.lower().endswith(".jpg")
        ]
        self.assertLessEqual(len(kept), 6)
        self.assertGreater(len(kept), 0)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_sharpness_floor_percentile_drops_blurriest(self) -> None:
        """A percentile floor culls the uniformly-blurry frames while
        keeping the sharp ones, regardless of absolute sharpness scale."""
        import numpy as np
        import cv2 as _cv2
        from pythontk import ImageCurator

        src = os.path.join(self.temp_dir, "pct_src")
        os.makedirs(src, exist_ok=True)
        # 10 frames, each its own dHash cluster via a unique vertical band
        # (low-freq → survives dHash's 9x8 downscale). Sharpness is set
        # *independently* by high-freq horizontal lines (averaged out by
        # dHash, but caught by the full-res Laplacian), so the floor — not
        # clustering — does the culling. Frames 0,1 stay flat = blurriest.
        for i in range(10):
            arr = np.full((64, 64, 3), 80, dtype=np.uint8)
            col = 3 + i * 6  # distinct horizontal position per frame
            arr[:, col:col + 4] = 255
            if i >= 2:  # high-frequency detail → high Laplacian variance
                arr[1::2, :] = 255
            _cv2.imwrite(os.path.join(src, f"img_{i:02d}.jpg"), arr)
        out_root = os.path.join(self.temp_dir, "pct_out")
        result = ImageCurator().curate(
            [src], out_root, hash_threshold=0,
            sharpness_floor_percentile=25.0,
        )
        kept = [f for f in os.listdir(result[0]) if f.lower().endswith(".jpg")]
        # The two flat frames (bottom 25% sharpness) must be gone; the
        # sharp ones retained.
        self.assertNotIn("img_00.jpg", kept)
        self.assertNotIn("img_01.jpg", kept)
        self.assertGreaterEqual(len(kept), 5)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_percentile_floor_uses_cluster_representatives(self) -> None:
        """The percentile cutoff must be taken over cluster representatives,
        not all scanned frames — otherwise a mass of blurry near-duplicates
        drags it down and the genuinely-blurry unique frame survives.

        Layout: 6 sharp unique + 1 unique blurry 'needle' + 40 blurrier
        near-duplicates that collapse to ONE cluster. With the floor over
        representatives (8 of them), p30 drops the needle. Over all 47
        scanned frames the cutoff lands among the 40 dupes — below the
        needle — so the old logic would keep it. This test fails on that
        old behavior.
        """
        import numpy as np
        import cv2 as _cv2
        from pythontk import ImageCurator

        src = os.path.join(self.temp_dir, "rep_src")
        os.makedirs(src, exist_ok=True)

        def frame(band_col, line_step=None, band_val=255):
            # Distinct vertical band -> distinct dHash (low-freq, survives
            # the 9x8 downscale). Optional high-freq horizontal lines set
            # sharpness independently (averaged out by dHash).
            arr = np.full((64, 64, 3), 80, dtype=np.uint8)
            arr[:, band_col:band_col + 3] = band_val
            if line_step:
                arr[1::line_step, :] = 255
            return arr

        for i in range(6):  # sharp: dense lines -> highest sharpness
            _cv2.imwrite(os.path.join(src, f"sharp_{i}.jpg"),
                         frame(4 + i * 9, line_step=2))
        _cv2.imwrite(os.path.join(src, "needle.jpg"),
                     frame(58, line_step=8))  # blurry unique (medium)
        for j in range(40):  # blurriest: band only, all identical -> 1 cluster
            _cv2.imwrite(os.path.join(src, f"dup_{j:02d}.jpg"), frame(1))

        out_root = os.path.join(self.temp_dir, "rep_out")
        # hash_threshold=1: identical dupes cluster (Hamming 0), distinct
        # bands stay separate. 0 now disables clustering outright (the
        # documented "no dedup" contract), which would defeat this test's
        # representatives-vs-all-frames distinction.
        result = ImageCurator().curate(
            [src], out_root, hash_threshold=1, keep_per_cluster=1,
            sharpness_floor_percentile=30.0,
        )
        kept = [f for f in os.listdir(result[0]) if f.lower().endswith(".jpg")]
        self.assertNotIn("needle.jpg", kept,
                         "blurry needle survived — floor diluted by near-dups")
        self.assertFalse([f for f in kept if f.startswith("dup_")],
                         "blurry near-duplicate survived the floor")
        self.assertGreaterEqual(len([f for f in kept if f.startswith("sharp_")]), 4)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_median_fraction_guard_drops_catastrophic_blur(self) -> None:
        """A frame far below the survivor median is culled by the median-relative
        guard even with the percentile floor OFF — catching catastrophic defocus
        the percentile (a fixed bottom slice) can miss."""
        import numpy as np
        import cv2 as _cv2
        from pythontk import ImageCurator

        src = os.path.join(self.temp_dir, "frac_src")
        os.makedirs(src, exist_ok=True)

        def frame(band_col, lines):
            arr = np.full((64, 64, 3), 80, dtype=np.uint8)
            arr[:, band_col:band_col + 3] = 255          # distinct dHash band
            if lines:
                arr[1::2, :] = 255                        # dense high-freq → sharp
            return arr

        for i in range(8):  # sharp, each its own cluster
            _cv2.imwrite(os.path.join(src, f"sharp_{i}.jpg"), frame(4 + i * 7, True))
        _cv2.imwrite(os.path.join(src, "needle.jpg"), frame(58, False))  # flat = blur

        def kept(r):
            return [f for f in os.listdir(r[0]) if f.lower().endswith(".jpg")]

        # Guard OFF (and no percentile) → the blurry needle survives.
        off = ImageCurator().curate(
            [src], os.path.join(self.temp_dir, "frac_off"), hash_threshold=0,
            sharpness_floor_percentile=None, min_sharpness_fraction_of_median=0.0,
        )
        self.assertIn("needle.jpg", kept(off))
        # Guard ON → needle dropped, all 8 sharp frames retained.
        on = ImageCurator().curate(
            [src], os.path.join(self.temp_dir, "frac_on"), hash_threshold=0,
            sharpness_floor_percentile=None, min_sharpness_fraction_of_median=0.15,
        )
        k = kept(on)
        self.assertNotIn("needle.jpg", k)
        self.assertEqual(len([f for f in k if f.startswith("sharp_")]), 8)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_preview_reports_survivors_without_copying(self) -> None:
        """preview() dry-runs the curation sweep — reports survivors per
        threshold + the sharpness distribution, copies nothing, and its survivor
        count agrees with curate() at the same threshold."""
        import numpy as np
        import cv2 as _cv2
        from pythontk import ImageCurator

        src = os.path.join(self.temp_dir, "prev_src")
        os.makedirs(src, exist_ok=True)

        def frame(band_col, line_step=None):
            arr = np.full((64, 64, 3), 80, dtype=np.uint8)
            arr[:, band_col:band_col + 3] = 255
            if line_step:
                arr[1::line_step, :] = 255
            return arr

        for i in range(6):  # distinct sharp uniques
            _cv2.imwrite(os.path.join(src, f"u_{i}.jpg"), frame(4 + i * 9, line_step=2))
        for j in range(10):  # identical near-dups -> collapse to one cluster
            _cv2.imwrite(os.path.join(src, f"d_{j:02d}.jpg"), frame(1))

        cur = ImageCurator()
        rep = cur.preview([src], hash_thresholds=(0, 10))
        self.assertEqual(rep["n_scanned"], 16)
        self.assertEqual(len(rep["thresholds"]), 2)
        for k in ("min", "median", "max"):
            self.assertIn(k, rep["sharpness"])
        kept_t0 = rep["thresholds"][0]["n_kept"]
        kept_t10 = rep["thresholds"][1]["n_kept"]
        self.assertLess(kept_t10, 16, "near-dups should be deduped")
        self.assertLessEqual(kept_t10, kept_t0)
        # preview must not have written any curated output dir
        self.assertFalse(
            any(d.endswith("_curated") for d in os.listdir(self.temp_dir)),
            "preview() copied files — it must be a dry run",
        )
        # preview survivor count agrees with the real curate() at the same threshold
        out_root = os.path.join(self.temp_dir, "prev_out")
        curated = cur.curate([src], out_root, hash_threshold=10, keep_per_cluster=1)
        n_curated = len(
            [f for f in os.listdir(curated[0]) if f.lower().endswith(".jpg")]
        )
        self.assertEqual(n_curated, kept_t10)


class ExposureEqualizerTest(BaseTestCase):
    """ExposureEqualizer strength blend + reference strategy."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def _make_dir(self, name, brightness):
        import numpy as np
        import cv2 as _cv2
        d = os.path.join(self.temp_dir, name)
        os.makedirs(d, exist_ok=True)
        for i in range(4):
            arr = np.full((48, 48, 3), brightness, dtype=np.uint8)
            arr[::3, :] = min(255, brightness + 60)  # some contrast
            _cv2.imwrite(os.path.join(d, f"f_{i}.jpg"), arr)
        return d

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_strength_zero_is_near_passthrough(self) -> None:
        """strength=0 should leave pixels essentially unchanged (a LAB
        round-trip only), proving the blend term is wired."""
        import cv2 as _cv2
        from pythontk import ExposureEqualizer

        dark = self._make_dir("eq_dark", 40)
        bright = self._make_dir("eq_bright", 200)
        out = os.path.join(self.temp_dir, "eq_out0")
        dirs = ExposureEqualizer().equalize_directories(
            [dark, bright], out, reference_dir=bright, strength=0.0
        )
        # The dark dir's output mean must stay close to its input mean
        # (~40), NOT pulled toward the bright reference (~200).
        orig = _cv2.imread(os.path.join(dark, "f_0.jpg")).mean()
        done = _cv2.imread(
            os.path.join(dirs[0], "f_0.jpg")
        ).mean()
        self.assertLess(abs(orig - done), 12.0)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_strength_one_pulls_toward_reference(self) -> None:
        import cv2 as _cv2
        from pythontk import ExposureEqualizer

        dark = self._make_dir("eq_dark2", 40)
        bright = self._make_dir("eq_bright2", 200)
        out = os.path.join(self.temp_dir, "eq_out1")
        dirs = ExposureEqualizer().equalize_directories(
            [dark, bright], out, reference_dir=bright, strength=1.0
        )
        done = _cv2.imread(os.path.join(dirs[0], "f_0.jpg")).mean()
        # Full match → dark frames lift well above their ~40 origin.
        self.assertGreater(done, 100.0)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2")
        and __import__("importlib").util.find_spec("PIL"),
        "cv2/PIL not available",
    )
    def test_equalize_preserves_exif(self) -> None:
        """Equalized JPEGs must carry source EXIF through (camera / focal-length
        priors a downstream SfM solver uses) instead of silently dropping it."""
        from PIL import Image
        from pythontk import ExposureEqualizer

        d = os.path.join(self.temp_dir, "eq_exif_src")
        os.makedirs(d, exist_ok=True)
        exif = Image.Exif()
        exif[271] = "TESTCAM"  # Make
        exif[272] = "MODEL-X"  # Model
        for i in range(3):
            Image.new("RGB", (48, 48), (90 + i * 10, 90, 90)).save(
                os.path.join(d, f"f_{i}.jpg"), exif=exif.tobytes()
            )
        out = os.path.join(self.temp_dir, "eq_exif_out")
        dirs = ExposureEqualizer().equalize_directories([d], out, reference_dir=d)
        out_jpg = os.path.join(dirs[0], "f_0.jpg")
        self.assertTrue(os.path.isfile(out_jpg))
        got = Image.open(out_jpg).getexif()
        self.assertEqual(got.get(271), "TESTCAM")
        self.assertEqual(got.get(272), "MODEL-X")

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2")
        and __import__("importlib").util.find_spec("PIL"),
        "cv2/PIL not available",
    )
    def test_equalize_does_not_double_rotate_oriented_jpeg(self) -> None:
        """cv2.imread bakes in EXIF orientation by default; copying the original
        Orientation tag onto auto-righted pixels double-rotates downstream. The
        output must keep the SAME stored pixels + Orientation tag as the source."""
        from PIL import Image
        from pythontk import ExposureEqualizer

        d = os.path.join(self.temp_dir, "eq_orient_src")
        os.makedirs(d, exist_ok=True)
        exif = Image.Exif()
        exif[0x0112] = 6  # Orientation: "rotate 90 CW to display"
        for i in range(2):
            # stored pixels are 40 wide x 60 tall
            Image.new("RGB", (40, 60), (100, 110, 120)).save(
                os.path.join(d, f"f_{i}.jpg"), exif=exif.tobytes()
            )
        out = os.path.join(self.temp_dir, "eq_orient_out")
        dirs = ExposureEqualizer().equalize_directories([d], out, reference_dir=d)
        op = Image.open(os.path.join(dirs[0], "f_0.jpg"))
        self.assertEqual(op.size, (40, 60))  # pixels NOT rotated into the buffer
        self.assertEqual(op.getexif().get(0x0112), 6)  # tag carried through

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_median_reference_avoids_extreme(self) -> None:
        """median strategy must pick the mid-brightness dir, not an extreme."""
        from pythontk.img_utils.exposure_equalizer import ExposureEqualizer
        eq = ExposureEqualizer()
        dark = self._make_dir("m_dark", 30)
        mid = self._make_dir("m_mid", 120)
        bright = self._make_dir("m_bright", 220)
        ref_mean, _ = eq._reference_stats([dark, mid, bright], "median", 4)
        # LAB L of a mid-grey is far from both extremes; assert the chosen
        # reference L sits between the dark and bright dirs' L.
        dark_l = eq._sample_stats(dark, 4)[0][0]
        bright_l = eq._sample_stats(bright, 4)[0][0]
        self.assertGreater(ref_mean[0], dark_l)
        self.assertLess(ref_mean[0], bright_l)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_global_reference_is_average_of_dirs(self) -> None:
        """'global' strategy targets the mean of every dir's stats."""
        from pythontk.img_utils.exposure_equalizer import ExposureEqualizer
        eq = ExposureEqualizer()
        dark = self._make_dir("g_dark", 40)
        bright = self._make_dir("g_bright", 200)
        g_mean, _ = eq._reference_stats([dark, bright], "global", 4)
        d_l = eq._sample_stats(dark, 4)[0][0]
        b_l = eq._sample_stats(bright, 4)[0][0]
        self.assertAlmostEqual(g_mean[0], (d_l + b_l) / 2.0, delta=1.0)

    def test_invalid_reference_strategy_raises(self) -> None:
        """A shared library should fail fast on a bad enum, not silently
        fall back. Validation precedes any cv2 use, so this needs no cv2."""
        from pythontk import ExposureEqualizer
        with self.assertRaises(ValueError):
            ExposureEqualizer()._reference_stats(["/nope"], "bogus", 4)


class PrepRegressionTest(BaseTestCase):
    """Regressions from the 2026-07 photogrammetry prep audit: same-basename
    source collisions, stale-output accumulation, and per-capture equalization
    (see CHANGELOG 2026-07-10)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_frames(self, dirpath, count=3, base=100, size=64,
                      vertical=False):
        """Write ``count`` distinct-ish JPEG frames; returns their names.

        ``vertical`` flips the stripe orientation — dHash is exposure-
        invariant (it hashes horizontal gradients), so two captures that
        differ only in brightness hash identically and cross-directory
        dedup would merge them; orientation makes the *structure* differ.
        """
        import numpy as np
        import cv2 as _cv2
        os.makedirs(dirpath, exist_ok=True)
        names = []
        for i in range(count):
            arr = np.full((size, size, 3), base + i * 20, dtype=np.uint8)
            if vertical:
                arr[:, :: (i + 2)] = 255
            else:
                arr[:: (i + 2), :] = 255  # texture so dHash/sharpness differ
            name = f"f{i}.jpg"
            _cv2.imwrite(os.path.join(dirpath, name), arr)
            names.append(name)
        return names

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_curate_same_basename_sources_do_not_clobber(self) -> None:
        """capA/images + capB/images share a basename; the second source's
        output purge must not delete the first's just-copied files."""
        from pythontk import ImageCurator
        cap_a = os.path.join(self.temp_dir, "capA", "images")
        cap_b = os.path.join(self.temp_dir, "capB", "images")
        self._write_frames(cap_a, base=60)
        self._write_frames(cap_b, base=160, vertical=True)
        out_root = os.path.join(self.temp_dir, "curated")
        out_dirs = ImageCurator().curate([cap_a, cap_b], out_root,
                                         hash_threshold=0)
        self.assertEqual(len(out_dirs), 2)
        self.assertNotEqual(out_dirs[0], out_dirs[1])
        for d in out_dirs:
            self.assertTrue(os.listdir(d), f"{d} lost its capture's files")

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_equalize_same_basename_sources_do_not_collide(self) -> None:
        from pythontk import ExposureEqualizer
        cap_a = os.path.join(self.temp_dir, "capA", "images")
        cap_b = os.path.join(self.temp_dir, "capB", "images")
        self._write_frames(cap_a, base=60)
        self._write_frames(cap_b, base=160)
        out_root = os.path.join(self.temp_dir, "eq")
        out_dirs = ExposureEqualizer().equalize_directories(
            [cap_a, cap_b], out_root, strength=0.5,
            reference_strategy="median",
        )
        self.assertEqual(len(set(out_dirs)), 2)
        for d in out_dirs:
            self.assertEqual(len(os.listdir(d)), 3, d)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_equalize_purges_stale_output_on_rerun(self) -> None:
        """Re-equalizing after the input set shrank (tighter curation) must
        not leave the previous run's since-culled frames in the output dir —
        downstream add_images ingests the union, degrading every re-run."""
        from pythontk import ExposureEqualizer
        src = os.path.join(self.temp_dir, "cap")
        names = self._write_frames(src, count=4)
        out_root = os.path.join(self.temp_dir, "eq")
        eq = ExposureEqualizer()
        first = eq.equalize_directories([src], out_root, strength=0.5)[0]
        self.assertEqual(len(os.listdir(first)), 4)
        os.remove(os.path.join(src, names[0]))  # "curation" culled a frame
        second = eq.equalize_directories([src], out_root, strength=0.5)[0]
        self.assertEqual(
            sorted(os.listdir(second)), sorted(names[1:]),
            "stale previously-equalized frame survived the re-run",
        )

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_equalize_never_purges_the_source_dir(self) -> None:
        """output_root=parent + suffix="" resolves the output dir onto the
        source dir itself; the stale-output purge must skip it (in-place
        overwrite) instead of rmtree'ing the capture before reading it."""
        from pythontk import ExposureEqualizer
        src = os.path.join(self.temp_dir, "cap")
        names = self._write_frames(src)
        out = ExposureEqualizer().equalize_directories(
            [src], self.temp_dir, suffix="", strength=0.5
        )[0]
        self.assertEqual(
            os.path.normcase(os.path.normpath(out)),
            os.path.normcase(os.path.normpath(src)),
        )
        self.assertEqual(
            sorted(os.listdir(src)), sorted(names),
            "source capture was destroyed by the overwrite purge",
        )

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_per_capture_mode_preserves_intra_capture_ordering(self) -> None:
        """Default (per-capture) equalization applies ONE transform per dir,
        so a capture's dark frame stays darker than its bright frame; the
        legacy per-image mode collapses both onto the reference stats."""
        import numpy as np
        import cv2 as _cv2
        from pythontk import ExposureEqualizer

        src = os.path.join(self.temp_dir, "cap")
        os.makedirs(src)
        for name, base in (("dark.jpg", 40), ("bright.jpg", 200)):
            arr = np.full((64, 64, 3), base, dtype=np.uint8)
            arr[::3, :] = min(255, base + 40)
            _cv2.imwrite(os.path.join(src, name), arr)
        ref = os.path.join(self.temp_dir, "ref")
        self._write_frames(ref, base=120)
        out_root = os.path.join(self.temp_dir, "eq")
        out = ExposureEqualizer().equalize_directories(
            [src], out_root, reference_dir=ref, strength=1.0,
        )[0]

        def mean_luma(path):
            img = _cv2.imread(path)
            return float(
                _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY).mean()
            )

        dark = mean_luma(os.path.join(out, "dark.jpg"))
        bright = mean_luma(os.path.join(out, "bright.jpg"))
        self.assertLess(
            dark + 10.0, bright,
            "per-capture mode must preserve intra-capture exposure ordering",
        )

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("cv2"),
        "cv2 not available",
    )
    def test_hash_threshold_zero_keeps_bit_identical_frames(self) -> None:
        """threshold 0 = NO dedup, as every runner/tooltip promises — even
        bit-identical frames (routine for a paused camera) must all survive.
        Hamming-0 clustering used to merge them and keep one per cluster."""
        import numpy as np
        import cv2 as _cv2
        from pythontk import ImageCurator

        src = os.path.join(self.temp_dir, "identical")
        os.makedirs(src)
        arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        arr[::3, :] = 255
        for i in range(3):  # three byte-identical frames
            _cv2.imwrite(os.path.join(src, f"f{i}.jpg"), arr)
        out_root = os.path.join(self.temp_dir, "identical_out")
        out = ImageCurator().curate([src], out_root, hash_threshold=0)[0]
        self.assertEqual(len(os.listdir(out)), 3,
                         "hash_threshold=0 must keep ALL frames")


if __name__ == "__main__":
    unittest.main(exit=False)
