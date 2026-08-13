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

import numpy as np
from PIL import Image

from pythontk import DeliveryBudget, FileUtils, ImgUtils, OutputSpec, OutputTemplates
from pythontk.core_utils.engines.textures.map_factory import MapFactory
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
        # A resize op carries the full (w, h) it resolved to — one scalar could
        # only ever describe a square, which is what used to squash non-square
        # maps.
        plan = [
            Op("resize", "", {"size": (512, 512)}),
            Op("mode_coerce", "", {"target_mode": "RGB"}),
        ]
        self.assertEqual(MapOptimizer.project(plan, 2048, 2048, "P"), (512, 512, "RGB"))

    def test_replays_a_non_square_resize(self):
        plan = [Op("resize", "", {"size": (512, 256)})]
        self.assertEqual(MapOptimizer.project(plan, 2048, 1024, "RGB"), (512, 256, "RGB"))

    def test_replays_force_pot(self):
        plan = [Op("force_pot", "", {"size": (256, 128)})]
        self.assertEqual(
            MapOptimizer.project(plan, 300, 100, "RGB"), (256, 128, "RGB")
        )

    def test_empty_plan_is_identity(self):
        self.assertEqual(MapOptimizer.project([], 64, 32, "L"), (64, 32, "L"))


class TestPotCeiling(unittest.TestCase):
    """POT must never snap back over a max_size the caller just imposed.

    A ceiling is a hard constraint; rounding to the NEAREST power of two after
    honoring it silently undoes it (measured: 256x256 with max_size=200 resized
    to 200, then snapped back up to 256).
    """

    def test_pot_never_rounds_back_over_an_explicit_max_size(self):
        image = ImgUtils.create_image("RGB", (256, 256))
        plan = MapOptimizer.plan(image, max_size=200, force_pot=True)
        width, height, _mode = MapOptimizer.project(plan, 256, 256, "RGB")
        self.assertLessEqual(width, 200, "the max_size ceiling was not honored")
        self.assertEqual((width, height), (128, 128))

    def test_pot_cannot_cross_the_ceiling_when_no_resize_fired(self):
        """The source already fits under max_size, so no resize op runs — but
        snapping to the NEAREST power of two can still cross the ceiling on its
        own (200 under a 250 limit snaps to 256). Keying the downward snap off
        "did a resize happen" misses this entirely."""
        image = ImgUtils.create_image("RGB", (200, 200))
        plan = MapOptimizer.plan(image, max_size=250, force_pot=True)
        width, height, _mode = MapOptimizer.project(plan, 200, 200, "RGB")
        self.assertLessEqual(width, 250, "POT snapped past the ceiling")
        self.assertEqual((width, height), (128, 128))

    def test_pot_keeps_nearest_when_it_stays_under_the_ceiling(self):
        """Clamping must only engage where the snap would actually violate the
        limit — a legal upward snap is still the closest match."""
        image = ImgUtils.create_image("RGB", (200, 200))
        plan = MapOptimizer.plan(image, max_size=2048, force_pot=True)
        self.assertEqual(MapOptimizer.project(plan, 200, 200, "RGB")[:2], (256, 256))

    def test_pot_without_a_ceiling_still_snaps_to_nearest(self):
        """No ceiling means no constraint to violate — a caller asking for POT
        wants the closest match, which is the long-standing behavior."""
        image = ImgUtils.create_image("RGB", (200, 200))
        plan = MapOptimizer.plan(image, force_pot=True)
        self.assertEqual(MapOptimizer.project(plan, 200, 200, "RGB")[:2], (256, 256))


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


class TestDeliveryBudget(_TextureFixture):
    """The advisory tier end-to-end: reported by default, applied only on opt-in.

    Registers a tiny throwaway profile rather than exercising a shipped one — a
    128px ceiling keeps the fixtures small, and the built-in budgets are free to
    be re-tuned without dragging these assertions along.
    """

    PROFILE = "_test_delivery_budget"

    def setUp(self):
        super().setUp()
        OutputTemplates.BUILTIN[self.PROFILE] = OutputTemplates._build(
            "png", budget=DeliveryBudget(max_size=128, force_pot=True)
        )

    def tearDown(self):
        OutputTemplates.BUILTIN.pop(self.PROFILE, None)
        super().tearDown()

    def test_assess_reports_without_planning_a_resize(self):
        path = self.texture(size=(256, 256))

        result = MapOptimizer.assess(path, output_profile=self.PROFILE)

        self.assertTrue(result["warnings"])
        self.assertIn("256x256", result["warnings"][0])
        self.assertEqual(result["predicted"]["width"], 256)  # nothing resampled
        self.assertFalse([r for r in result["reasons"] if "Resize" in r])

    def test_assess_enforces_on_opt_in(self):
        path = self.texture(size=(256, 256))

        result = MapOptimizer.assess(
            path, output_profile=self.PROFILE, enforce_budget=True
        )

        self.assertEqual(result["predicted"]["width"], 128)
        self.assertEqual(result["warnings"], [])  # inside budget by construction

    def test_warnings_key_is_always_present(self):
        """Consumers can render it unconditionally — including on the error paths."""
        path = self.texture(size=(64, 64))

        self.assertEqual(MapOptimizer.assess(path)["warnings"], [])
        missing = MapOptimizer.assess(os.path.join(self.test_dir, "nope.png"))
        self.assertEqual(missing["warnings"], [])

    def test_non_pot_is_flagged_independently_of_size(self):
        path = self.texture(size=(100, 100))  # under the ceiling, but not POT

        warnings = MapOptimizer.assess(path, output_profile=self.PROFILE)["warnings"]

        self.assertEqual(len(warnings), 1)
        self.assertIn("power-of-two", warnings[0].lower())

    def test_optimize_map_leaves_pixels_alone_by_default(self):
        path = self.texture(size=(256, 256))

        written = MapOptimizer.optimize_map(path, output_profile=self.PROFILE)

        self.assertEqual(ImgUtils.ensure_image(written).size, (256, 256))

    def test_optimize_map_warns_by_default(self):
        import io
        from contextlib import redirect_stdout

        path = self.texture(size=(256, 256))

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            MapOptimizer.optimize_map(path, output_profile=self.PROFILE)

        self.assertIn("Over delivery budget", buffer.getvalue())

    def test_optimize_map_applies_the_budget_on_opt_in(self):
        path = self.texture(size=(256, 256))

        written = MapOptimizer.optimize_map(
            path, output_profile=self.PROFILE, enforce_budget=True
        )

        self.assertEqual(ImgUtils.ensure_image(written).size, (128, 128))

    def test_explicit_max_size_outranks_the_budget(self):
        """Same precedence as output_type over the template's ext."""
        path = self.texture(size=(256, 256))

        written = MapOptimizer.optimize_map(
            path, max_size=64, output_profile=self.PROFILE, enforce_budget=True
        )

        self.assertEqual(ImgUtils.ensure_image(written).size, (64, 64))

    def test_an_oversized_explicit_max_size_still_warns(self):
        """The report is against what gets written, not against the source — an
        explicit ceiling that overshoots the profile is exactly the case a
        silent budget would hide. (256 is POT, so the POT step can't move it.)"""
        path = self.texture(size=(512, 512))

        result = MapOptimizer.assess(
            path, max_size=256, output_profile=self.PROFILE, enforce_budget=True
        )

        self.assertEqual(result["predicted"]["width"], 256)  # explicit wins
        self.assertTrue(any("Over delivery budget" in w for w in result["warnings"]))

    def test_budget_never_upscales(self):
        """A *delivery* budget exists to make an asset cheaper. POT snapping
        rounds to the NEAREST power of two, so a budget-driven snap used to
        round 1536 UP to 2048 (+78% pixels) and then report itself as within
        budget — the exact opposite of the flag's purpose."""
        path = self.texture(size=(96, 96))  # nearest POT is 128 (up), floor is 64

        written = MapOptimizer.optimize_map(
            path, output_profile=self.PROFILE, enforce_budget=True
        )
        with ImgUtils.ensure_image(written) as out:
            self.assertLessEqual(out.size[0], 96, "budget must never upscale")
            self.assertEqual(out.size, (64, 64))

    def test_explicit_force_pot_false_opts_out_of_the_budget(self):
        """``optimize_map``'s docstring promises an explicit argument outranks
        the budget. ``force_pot or budget.force_pot`` cannot honor that — False
        is indistinguishable from unset."""
        path = self.texture(size=(96, 96))

        written = MapOptimizer.optimize_map(
            path, output_profile=self.PROFILE, enforce_budget=True, force_pot=False
        )
        with ImgUtils.ensure_image(written) as out:
            self.assertEqual(out.size, (96, 96), "explicit False must disable POT")

    def test_resize_preserves_aspect_ratio(self):
        """The resize step drove both edges from one scalar, squashing every
        non-square map to a square. Harmless while only an explicit max_size
        reached it; automatic for every budgeted profile once enforce_budget
        existed."""
        path = self.texture(size=(256, 128))

        result = MapOptimizer.assess(
            path, max_size=128, force_pot=False, output_profile=self.PROFILE
        )
        self.assertEqual(
            (result["predicted"]["width"], result["predicted"]["height"]), (128, 64)
        )

        written = MapOptimizer.optimize_map(
            path, max_size=128, force_pot=False, output_profile=self.PROFILE
        )
        with ImgUtils.ensure_image(written) as out:
            self.assertEqual(out.size, (128, 64))

    def test_apply_honors_the_planned_pot_size(self):
        """``apply`` is a pure dispatcher by contract — re-deciding a size it
        was handed is the drift the plan/apply split exists to prevent."""
        image = ImgUtils.create_image("RGB", (96, 96))
        plan = MapOptimizer.plan(image, force_pot=True, pot_mode="down")
        self.assertEqual(MapOptimizer.apply(image, plan).size, (64, 64))

    def test_the_two_twins_resolve_a_profile_identically(self):
        """Both read the profile through one helper — pinned here because a
        second copy of the precedence rules is what makes a dry run start
        describing a different run than the one that follows it.

        Asserted across the twins (what ``assess`` predicts == what
        ``optimize_map`` writes); comparing ``_resolve_profile`` to itself
        passed unconditionally and would survive re-inlining the rules.
        """
        for enforce in (False, True):
            for explicit in (None, 64):
                with self.subTest(enforce=enforce, explicit=explicit):
                    path = self.texture(
                        name=f"twin_{enforce}_{explicit}.png", size=(256, 256)
                    )
                    predicted = MapOptimizer.assess(
                        path,
                        max_size=explicit,
                        output_profile=self.PROFILE,
                        enforce_budget=enforce,
                    )["predicted"]
                    written = MapOptimizer.optimize_map(
                        path,
                        max_size=explicit,
                        output_profile=self.PROFILE,
                        enforce_budget=enforce,
                    )
                    with ImgUtils.ensure_image(written) as out:
                        self.assertEqual(
                            (predicted["width"], predicted["height"]), out.size
                        )

        # The precedence itself, stated once: an enforced budget fills a gap
        # (the caller said nothing == None), never overrides. A budget-sourced
        # POT snaps DOWN so it cannot inflate the asset.
        _, _, _, filled, pot, mode = MapOptimizer._resolve_profile(
            self.PROFILE, "Base_Color", None, None, None, True
        )
        self.assertEqual((filled, pot, mode), (128, True, "down"))
        _, _, _, kept, _, _ = MapOptimizer._resolve_profile(
            self.PROFILE, "Base_Color", None, 64, None, True
        )
        self.assertEqual(kept, 64)
        _, _, _, _, pot_opt_out, _ = MapOptimizer._resolve_profile(
            self.PROFILE, "Base_Color", None, None, False, True
        )
        self.assertFalse(pot_opt_out, "an explicit False must outrank the budget")
        _, _, _, untouched, pot_off, _ = MapOptimizer._resolve_profile(
            self.PROFILE, "Base_Color", None, None, None, False
        )
        self.assertEqual((untouched, pot_off), (None, False))

    def test_no_profile_means_no_budget(self):
        path = self.texture(size=(256, 256))

        self.assertEqual(MapOptimizer.assess(path, enforce_budget=True)["warnings"], [])
        written = MapOptimizer.optimize_map(path, enforce_budget=True)
        self.assertEqual(ImgUtils.ensure_image(written).size, (256, 256))


class LossyQualityTest(_TextureFixture):
    """The three vetoes in resolve_quality, and that assess honours all of them.

    The dry run and the real run must reach the same verdict — a preview that
    promised lossy while the run refused it would be worse than no preview.
    """

    def test_safe_map_and_lossy_container_pass(self):
        quality, skipped = MapOptimizer.resolve_quality(90, "Base_Color", "webp")
        self.assertEqual(quality, 90)
        self.assertIsNone(skipped)

    def test_unsafe_map_is_refused_with_a_reason(self):
        for map_type in ("Normal", "ORM", "Roughness"):
            quality, skipped = MapOptimizer.resolve_quality(90, map_type, "webp")
            self.assertIsNone(quality, map_type)
            self.assertIn(map_type, skipped)

    def test_lossless_container_is_refused_with_a_reason(self):
        quality, skipped = MapOptimizer.resolve_quality(90, "Base_Color", "png")
        self.assertIsNone(quality)
        self.assertIn("lossless container", skipped)

    def test_no_request_is_not_a_refusal(self):
        self.assertEqual(
            MapOptimizer.resolve_quality(None, "Normal", "webp"), (None, None)
        )

    def test_a_jpeg_container_is_itself_a_lossy_request(self):
        """JPEG has no lossless mode, so naming it IS asking for lossy.

        This gate's own docstring says it is what stops "a batch run from
        destroying every normal map in a folder because the operator set one
        dropdown" -- and the container dropdown is exactly such a dropdown. An
        explicit ``lossy_quality`` on a normal map was loudly refused, while
        selecting ``.jpg`` and saying nothing about quality sailed past the
        gate entirely and wrote it at the q95 default with ``warnings: []``.
        The guarded path was the one nobody takes.

        WebP must stay unaffected: it HAS a lossless mode and ``_apply_lossy_
        kwargs`` defaults to it when no quality is requested, so nothing is
        degraded there and a warning would be noise.
        """
        for map_type in ("Normal", "ORM", "Roughness"):
            quality, skipped = MapOptimizer.resolve_quality(None, map_type, "jpg")
            self.assertIsNone(quality, map_type)
            self.assertIsNotNone(skipped, f"{map_type} into .jpg was not reported")
            self.assertIn(map_type, skipped)

        self.assertEqual(
            MapOptimizer.resolve_quality(None, "Normal", "webp"), (None, None)
        )
        # A lossy-safe map into JPEG is the intended use -- no warning.
        self.assertIsNone(MapOptimizer.resolve_quality(None, "Base_Color", "jpg")[1])

    def test_call_level_quality_outranks_the_profile_spec(self):
        spec = OutputSpec("webp", 8, None, 80)
        quality, _ = MapOptimizer.resolve_quality(95, "Base_Color", "webp", spec)
        self.assertEqual(quality, 95)
        # ...and the spec still applies when the caller says nothing.
        quality, _ = MapOptimizer.resolve_quality(None, "Base_Color", "webp", spec)
        self.assertEqual(quality, 80)

    def test_a_refused_request_surfaces_in_assess_warnings(self):
        path = self.texture(name="rock_Normal.png")
        report = MapOptimizer.assess(path, output_type="webp", lossy_quality=90)
        self.assertIsNone(report["predicted"]["quality"])
        self.assertTrue(any("refused" in w for w in report["warnings"]))

    def test_an_accepted_request_shows_in_assess(self):
        path = self.texture(name="rock_BaseColor.png")
        report = MapOptimizer.assess(path, output_type="webp", lossy_quality=90)
        self.assertEqual(report["predicted"]["quality"], 90)
        self.assertEqual(report["warnings"], [])

    def test_refused_map_is_written_losslessly(self):
        """The guarantee that matters: asking for lossy on a normal map must
        not degrade it, no matter how the request arrived."""
        path = self.texture(name="rock_Normal.png")
        written = MapOptimizer.optimize_map(
            path, output_dir=self.test_dir, output_type="webp", lossy_quality=80
        )
        source = ImgUtils.ensure_image(path).convert("RGB")
        result = ImgUtils.ensure_image(written).convert("RGB")
        self.assertEqual(source.tobytes(), result.tobytes())

    def test_assess_predicts_the_size_the_run_writes(self):
        path = self.texture(name="rock_BaseColor.png")
        predicted = MapOptimizer.assess(
            path, output_type="webp", lossy_quality=80, predict_size=True
        )["predicted"]["size_bytes"]
        written = MapOptimizer.optimize_map(
            path, output_dir=self.test_dir, output_type="webp", lossy_quality=80
        )
        self.assertEqual(predicted, os.path.getsize(written))


class ContainerModeAgreementTest(_TextureFixture):
    """assess must predict the mode the container will really store.

    WebP has no grayscale mode, so an "L" map lands as RGB. Reporting the
    in-memory mode claimed 8-bit for a 24-bit file and made the dry run
    predict something the real run could not produce.
    """

    def test_assess_predicts_the_container_widened_mode(self):
        path = self.texture(name="rock_Roughness.png", mode="L")
        report = MapOptimizer.assess(path, output_type="webp")
        self.assertEqual(report["predicted"]["mode"], "RGB")
        self.assertEqual(report["predicted"]["bit_depth"], "24bit (8x3)")

    def test_real_run_matches_the_prediction(self):
        path = self.texture(name="rock_Roughness.png", mode="L")
        predicted = MapOptimizer.assess(path, output_type="webp")["predicted"]
        written = MapOptimizer.optimize_map(
            path, output_dir=self.test_dir, output_type="webp"
        )
        self.assertEqual(ImgUtils.ensure_image(written).mode, predicted["mode"])

    def test_png_keeps_grayscale(self):
        path = self.texture(name="rock_Roughness.png", mode="L")
        report = MapOptimizer.assess(path, output_type="png")
        self.assertEqual(report["predicted"]["mode"], "L")
        written = MapOptimizer.optimize_map(
            path, output_dir=self.test_dir, output_type="png"
        )
        self.assertEqual(ImgUtils.ensure_image(written).mode, "L")


class PackedMapIntegrityTest(_TextureFixture):
    """A packed map's channels are independent data, not colour.

    Optimize must carry all of them through — including for a map whose
    filename the registry does not resolve, which is the common case for
    project-specific packings (a "_Full" lightmap, an "LAOM" atlas).
    """

    def _packed(self, name="rock_MSAO.png", opaque_alpha=False):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, (64, 64, 4), dtype=np.uint8)
        if opaque_alpha:
            arr[..., 3] = 255
        path = os.path.join(self.test_dir, name)
        Image.fromarray(arr, "RGBA").save(path)
        return path

    def test_rgba_packing_survives_every_alpha_capable_container(self):
        path = self._packed()
        for ext in ("png", "webp", "tga"):
            written = MapOptimizer.optimize_map(
                path, output_dir=self.test_dir, output_type=ext
            )
            with ImgUtils.ensure_image(written) as result:
                self.assertEqual(result.mode, "RGBA", ext)
                self.assertEqual(len(result.getbands()), 4, ext)

    def test_an_unresolved_map_type_is_passed_through_generically(self):
        """No resolved type means no mode coercion — the plan's map-type
        branches are all gated on it, so a custom packing keeps its channels."""
        path = self._packed(name="WALL_mat_Full.png")
        self.assertIsNone(MapFactory.resolve_map_type(path, key=True))

        report = MapOptimizer.assess(path)
        self.assertEqual(report["predicted"]["mode"], "RGBA")

        written = MapOptimizer.optimize_map(path, output_dir=self.test_dir)
        with ImgUtils.ensure_image(written) as result:
            self.assertEqual(result.mode, "RGBA")

    def test_resize_preserves_every_channel(self):
        path = self._packed()
        written = MapOptimizer.optimize_map(
            path, output_dir=self.test_dir, max_size=32
        )
        with ImgUtils.ensure_image(written) as result:
            self.assertEqual(result.size, (32, 32))
            self.assertEqual(result.mode, "RGBA")

    def test_dropping_a_channel_that_carries_data_is_called_out(self):
        warning = MapOptimizer.channel_loss_warning(
            ImgUtils.ensure_image(self._packed()), "jpg"
        )
        self.assertIsNotNone(warning)
        self.assertIn("channel A", warning)

    def test_dropping_a_uniform_channel_is_not_called_out(self):
        """A lightmap's fully-opaque alpha costs nothing — shouting about it
        would train the reader past the line that matters."""
        self.assertIsNone(
            MapOptimizer.channel_loss_warning(
                ImgUtils.ensure_image(self._packed(opaque_alpha=True)), "jpg"
            )
        )

    def test_no_warning_when_nothing_is_dropped(self):
        image = ImgUtils.ensure_image(self._packed())
        for ext in ("png", "webp", "tga"):
            self.assertIsNone(MapOptimizer.channel_loss_warning(image, ext))

    def test_the_dry_run_warns_before_the_channel_is_destroyed(self):
        """The whole point of a dry run: learn that jpg kills Smoothness
        BEFORE writing the batch, not from the log afterwards."""
        path = self._packed()
        report = MapOptimizer.assess(path, output_type="jpg")
        self.assertTrue(
            any("discarded" in w for w in report["warnings"]), report["warnings"]
        )

    def test_the_dry_run_stays_quiet_when_the_container_keeps_everything(self):
        path = self._packed()
        for ext in ("png", "webp", "tga"):
            self.assertEqual(MapOptimizer.assess(path, output_type=ext)["warnings"], [])

    def test_no_warning_when_the_plan_itself_dropped_the_channel(self):
        """An ORM declares mode RGB, so the plan coerces an RGBA source and
        reports it as an op — the container is not what loses the alpha, and
        warning twice for one drop would misattribute it."""
        path = self._packed(name="rock_ORM.png")
        report = MapOptimizer.assess(path, output_type="jpg")
        self.assertTrue(
            any("map_type=ORM" in r and "RGBA -> RGB" in r for r in report["reasons"]),
            report["reasons"],
        )
        # Scoped to the CHANNEL-drop warning, which is what this test is about:
        # the plan already accounted for the alpha, so the container must not
        # report it a second time. The separate lossy-codec warning (an ORM has
        # no business in a container with no lossless mode) is a different axis
        # and is pinned in LossyQualityTest -- asserting "no warnings at all"
        # here would have silently forbidden it.
        self.assertFalse(
            [w for w in report["warnings"] if "discarded" in w], report["warnings"]
        )

    def test_grayscale_to_webp_widens_without_dropping(self):
        # L -> RGB duplicates the channel; it is not a loss.
        self.assertEqual(ImgUtils.dropped_channels("L", "webp"), ())
        self.assertEqual(ImgUtils.dropped_channels("RGBA", "jpg"), ("A",))
        self.assertEqual(ImgUtils.dropped_channels("RGBA", "png"), ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
