#!/usr/bin/python
# coding=utf-8
"""Tests for pythontk.MapOptimizer — the plan / apply / assess trio.

Focus: the dry-run contract. ``assess`` is the read-only twin of
``optimize_map``, so anything it predicts (ops, resulting size/mode, output
path, byte count) has to match what a real run then produces — a projection
that drifts is worse than no projection at all.
"""
import os
import shutil
import tempfile
import unittest

from pythontk import FileUtils, ImgUtils
from pythontk.core_utils.engines.textures.map_optimizer import MapOptimizer, Op


class _TextureFixture(unittest.TestCase):
    """Writes real texture files — compression ratios can't be faked."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="map_optimizer_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def texture(self, name="rock_BaseColor.png", size=(256, 256), mode="RGB"):
        """A noisy (i.e. genuinely compressible) texture on disk."""
        path = os.path.join(self.test_dir, name)
        image = ImgUtils.create_image(mode, size)
        pixels = image.load()
        width, height = size
        for y in range(height):
            for x in range(width):
                value = (x * 7 + y * 13) % 256
                pixels[x, y] = value if mode == "L" else (value, (value * 3) % 256, 64)
        ImgUtils.save_image(image, path)
        return path


class TestFormatBytes(unittest.TestCase):
    """FileUtils.format_bytes / format_bytes_delta — the size vocabulary the
    optimizer's reporting is written in."""

    def test_units(self):
        self.assertEqual(FileUtils.format_bytes(None), "(unknown)")
        self.assertEqual(FileUtils.format_bytes(512), "512 bytes")
        self.assertEqual(FileUtils.format_bytes(2048), "2.0 KB")
        self.assertEqual(FileUtils.format_bytes(2 * 1024**2), "2.00 MB")
        self.assertEqual(FileUtils.format_bytes(3 * 1024**3), "3.00 GB")

    def test_non_numeric_passthrough(self):
        self.assertEqual(FileUtils.format_bytes("big"), "big")

    def test_custom_unknown_label(self):
        self.assertEqual(FileUtils.format_bytes(None, unknown="n/a"), "n/a")

    def test_delta_reports_percent(self):
        self.assertEqual(
            FileUtils.format_bytes_delta(12 * 1024**2, 3 * 1024**2),
            "12.00 MB -> 3.00 MB (-75%)",
        )

    def test_delta_percent_is_signed(self):
        self.assertIn("(+100%)", FileUtils.format_bytes_delta(1024, 2048))

    def test_delta_omits_percent_when_undefined(self):
        """A missing side or a zero-byte source has no meaningful percentage."""
        self.assertEqual(FileUtils.format_bytes_delta(None, 1024), "(unknown) -> 1.0 KB")
        self.assertEqual(FileUtils.format_bytes_delta(0, 1024), "0 bytes -> 1.0 KB")

    def test_mat_report_delegates_here(self):
        """MatReport._fmt_size_auto is an alias, not a second implementation."""
        from pythontk import MatReport

        for value in (None, 512, 2048, 2 * 1024**2, "big"):
            self.assertEqual(
                MatReport._fmt_size_auto(value), FileUtils.format_bytes(value)
            )


class TestProject(unittest.TestCase):
    """``project`` replays a plan's params — no pixel work, no second copy of
    plan()'s decision logic."""

    def test_replays_resize_and_mode(self):
        plan = [
            Op("resize", "", {"size": 512}),
            Op("mode_coerce", "", {"target_mode": "RGB"}),
        ]
        self.assertEqual(MapOptimizer.project(plan, 2048, 2048, "P"), (512, 512, "RGB"))

    def test_replays_force_pot(self):
        plan = [Op("force_pot", "", {"size": (256, 128)})]
        self.assertEqual(
            MapOptimizer.project(plan, 300, 100, "RGB"), (256, 128, "RGB")
        )

    def test_empty_plan_is_identity(self):
        self.assertEqual(MapOptimizer.project([], 64, 32, "L"), (64, 32, "L"))


class TestAssessPrediction(_TextureFixture):
    def test_predicted_matches_the_real_run(self):
        """Every predicted field must equal what optimize_map then writes."""
        path = self.texture(size=(256, 256))

        predicted = MapOptimizer.assess(path, max_size=128, predict_size=True)[
            "predicted"
        ]
        written = MapOptimizer.optimize_map(path, max_size=128)

        with ImgUtils.ensure_image(written) as result:
            self.assertEqual((result.width, result.height), (128, 128))
            self.assertEqual(result.mode, predicted["mode"])
            self.assertEqual((predicted["width"], predicted["height"]), (128, 128))
        self.assertEqual(predicted["size_bytes"], os.path.getsize(written))
        self.assertEqual(
            os.path.normcase(os.path.normpath(predicted["path"])),
            os.path.normcase(os.path.normpath(written)),
        )

    def test_predicted_path_tracks_an_output_type_change(self):
        path = self.texture(size=(64, 64))

        predicted = MapOptimizer.assess(path, output_type="tga")["predicted"]
        written = MapOptimizer.optimize_map(path, output_type="tga")

        self.assertEqual(predicted["ext"], "tga")
        self.assertEqual(
            os.path.normcase(os.path.normpath(predicted["path"])),
            os.path.normcase(os.path.normpath(written)),
        )

    def test_assess_writes_nothing(self):
        """The whole point of a dry run: the folder is untouched afterwards."""
        path = self.texture(size=(256, 256))
        before = sorted(os.listdir(self.test_dir))
        stat_before = os.stat(path)

        MapOptimizer.assess(path, max_size=64, predict_size=True)

        self.assertEqual(sorted(os.listdir(self.test_dir)), before)
        self.assertEqual(os.stat(path).st_size, stat_before.st_size)
        self.assertEqual(os.stat(path).st_mtime, stat_before.st_mtime)

    def test_size_prediction_is_opt_in(self):
        """Bulk report callers assess every texture in a scene; they must not
        pay for an encode they didn't ask for."""
        path = self.texture(size=(64, 64))

        predicted = MapOptimizer.assess(path, max_size=32)["predicted"]

        self.assertIsNone(predicted["size_bytes"])
        self.assertEqual((predicted["width"], predicted["height"]), (32, 32))

    def test_size_error_is_reported_not_raised(self):
        """A format that can't be encoded degrades to a reason, not a crash."""
        path = self.texture(size=(32, 32))

        predicted = MapOptimizer.assess(
            path, output_type="nonsense", predict_size=True
        )["predicted"]

        self.assertIsNone(predicted["size_bytes"])
        self.assertIn("size_error", predicted)

    def test_missing_file_returns_an_empty_predicted_block(self):
        report = MapOptimizer.assess(os.path.join(self.test_dir, "nope.png"))

        self.assertIn("error", report)
        self.assertEqual(report["predicted"], {})

    def test_does_not_mutate_the_callers_image(self):
        """``assess`` takes a pre-loaded image from bulk report callers; the
        throwaway encode must not resize the object they still hold."""
        path = self.texture(size=(256, 256))
        image = ImgUtils.ensure_image(path)

        MapOptimizer.assess(path, max_size=64, image=image, predict_size=True)

        self.assertEqual(image.size, (256, 256))

    def test_output_profile_is_honored_like_optimize_map(self):
        """A profile can dictate the extension (UE writes normals as TGA), so
        ignoring it would predict a file the real run never writes."""
        from pythontk.core_utils.engines.textures.map_registry import WF

        path = self.texture(name="rock_Normal_OpenGL.png", size=(64, 64))

        predicted = MapOptimizer.assess(path, output_profile=WF.UE)["predicted"]
        written = MapOptimizer.optimize_map(path, output_profile=WF.UE)

        self.assertEqual(predicted["ext"], "tga")
        self.assertEqual(os.path.splitext(written)[1].lstrip("."), "tga")
        self.assertEqual(
            os.path.normcase(os.path.normpath(predicted["path"])),
            os.path.normcase(os.path.normpath(written)),
        )

    def test_explicit_output_type_still_beats_the_profile(self):
        """Mirrors optimize_map's precedence (``if spec and not output_type``)."""
        path = self.texture(name="rock_Normal_OpenGL.png", size=(64, 64))

        predicted = MapOptimizer.assess(
            path, output_type="tga", output_profile="nonexistent_profile"
        )["predicted"]

        self.assertEqual(predicted["ext"], "tga")


class TestOptimizeReporting(_TextureFixture):
    def test_format_result_shows_both_transitions(self):
        path = self.texture(size=(64, 64))
        image = ImgUtils.ensure_image(path)

        summary = MapOptimizer.format_result(path, 4096, (256, 256), image)

        self.assertIn("256x256 -> 64x64", summary)
        self.assertIn("4.0 KB -> ", summary)
        self.assertIn(ImgUtils.format_bit_depth(image), summary)

    def test_format_result_collapses_unchanged_dimensions(self):
        path = self.texture(size=(64, 64))
        image = ImgUtils.ensure_image(path)

        summary = MapOptimizer.format_result(path, 4096, (64, 64), image)

        self.assertIn("64x64,", summary)
        self.assertNotIn("64x64 -> 64x64", summary)

    def test_format_result_survives_an_unknown_source_size(self):
        path = self.texture(size=(64, 64))
        image = ImgUtils.ensure_image(path)

        summary = MapOptimizer.format_result(path, None, None, image)

        self.assertIn("(unknown) -> ", summary)

    def test_optimize_map_prints_the_size_transition(self):
        """Shrinking the file is the point of optimizing, so the size has to be
        in the output — it previously reported only resolution and bit depth."""
        import io
        from contextlib import redirect_stdout

        path = self.texture(size=(256, 256))
        size_before = os.path.getsize(path)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            written = MapOptimizer.optimize_map(path, max_size=64)
        printed = buffer.getvalue()

        self.assertIn(FileUtils.format_bytes(size_before), printed)
        self.assertIn(FileUtils.format_bytes(os.path.getsize(written)), printed)
        self.assertIn("256x256 -> 64x64", printed)

    def test_size_before_survives_the_archive_move(self):
        """The original is moved to the old-files folder *before* the save, so
        its size has to be read up front or the report loses its 'from' half."""
        import io
        from contextlib import redirect_stdout

        path = self.texture(size=(128, 128))
        size_before = os.path.getsize(path)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            MapOptimizer.optimize_map(path, max_size=64, old_files_folder="old")

        self.assertIn(FileUtils.format_bytes(size_before), buffer.getvalue())
        self.assertTrue(
            os.path.isfile(os.path.join(self.test_dir, "old", os.path.basename(path)))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
