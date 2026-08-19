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
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image

from pythontk import (
    DeliveryBudget,
    FileUtils,
    ImgUtils,
    OutputSpec,
    OutputTemplate,
    OutputTemplates,
)
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

    def test_unknown_pot_mode_is_rejected(self):
        """``pot_mode`` is public on all three entry points but was matched as
        ``"down"`` or else round() — so "downward", "Down" and every typo took
        the nearest snap silently, growing the very asset a budget caller was
        rounding down to shrink."""
        image = ImgUtils.create_image("RGB", (96, 96))
        for bogus in ("downward", "Down", "floor"):
            with self.subTest(pot_mode=bogus):
                with self.assertRaises(ValueError) as ctx:
                    MapOptimizer.plan(image, force_pot=True, pot_mode=bogus)
                self.assertIn("pot_mode", str(ctx.exception))
        # The documented default survives: omitted/None still means nearest.
        self.assertEqual(
            MapOptimizer._snap_pot(96, 96, None), MapOptimizer._snap_pot(96, 96)
        )

    def test_pot_without_a_ceiling_still_snaps_to_nearest(self):
        """No ceiling means no constraint to violate — a caller asking for POT
        wants the closest match, which is the long-standing behavior."""
        image = ImgUtils.create_image("RGB", (200, 200))
        plan = MapOptimizer.plan(image, force_pot=True)
        self.assertEqual(MapOptimizer.project(plan, 200, 200, "RGB")[:2], (256, 256))


class TestPotAspectRatio(_TextureFixture):
    """A POT snap must not reshape the map.

    ``_snap_pot`` rounded each axis on its own, so a 4:3 source came out 2:1:
    1024x768 under a profile whose budget carries the POT rule (glTF) was
    rewritten 1024x512 — well under that profile's 2048 ceiling, and
    ``budget.check`` runs post-snap, so nothing warned about the reshape
    either. POT is a statement about how BIG a texture is, not what shape it
    is.
    """

    PROFILE = "_test_pot_aspect"

    def setUp(self):
        super().setUp()
        # Mirrors the shipped glTF budget (2048 + POT) without pinning these
        # assertions to a built-in that is free to be re-tuned.
        OutputTemplates.BUILTIN[self.PROFILE] = OutputTemplates._build(
            "png", budget=DeliveryBudget(max_size=2048, force_pot=True)
        )

    def tearDown(self):
        OutputTemplates.BUILTIN.pop(self.PROFILE, None)
        super().tearDown()

    def test_pot_leaves_a_4x3_map_4x3(self):
        """The reported repro: the long edge is already POT and already inside
        the ceiling, so the only correct snap is no snap at all."""
        image = ImgUtils.create_image("RGB", (1024, 768))
        plan = MapOptimizer.plan(
            image, max_size=2048, force_pot=True, pot_mode="down"
        )
        self.assertEqual(
            MapOptimizer.project(plan, 1024, 768, "RGB")[:2], (1024, 768)
        )

    def test_short_edge_is_derived_from_the_source_aspect(self):
        """When the long edge really does move, the other one follows it."""
        self.assertEqual(MapOptimizer._snap_pot(1500, 1000, "down"), (1024, 683))
        self.assertEqual(MapOptimizer._snap_pot(1000, 1500, "down"), (683, 1024))

    def test_a_ceiling_clamp_keeps_the_aspect_too(self):
        """``max_size`` halves the snapped long edge back under the limit, so
        the derived edge has to ride the same factor rather than be snapped."""
        self.assertEqual(MapOptimizer._snap_pot(3000, 1000, "down", 1024), (1024, 341))

    def test_square_sources_are_unchanged(self):
        """The long-edge form must reduce to the old behavior for a square."""
        self.assertEqual(MapOptimizer._snap_pot(96, 96, "down"), (64, 64))
        self.assertEqual(MapOptimizer._snap_pot(96, 96), (128, 128))

    def test_enforced_budget_run_keeps_the_aspect(self):
        """End to end through the profile shape the Map Converter's
        'Clamp: Target' uses: assess predicts what optimize_map writes, and
        neither reshapes the map. The edge POT could NOT reach is still
        reported — the budget's check is the honest place for that, and a
        silent reshape is what this replaced."""
        path = self.texture(size=(96, 72))  # 4:3, same shape as 1024x768

        result = MapOptimizer.assess(
            path, output_profile=self.PROFILE, enforce_budget=True
        )
        self.assertEqual(
            (result["predicted"]["width"], result["predicted"]["height"]), (64, 48)
        )
        self.assertTrue(
            any("power-of-two" in w.lower() for w in result["warnings"]),
            f"the residual non-POT edge must be reported: {result['warnings']}",
        )

        written = MapOptimizer.optimize_map(
            path, output_profile=self.PROFILE, enforce_budget=True
        )
        with ImgUtils.ensure_image(written) as out:
            self.assertEqual(out.size, (64, 48))

    def test_a_reshaping_snap_warns(self):
        """The min-1 floor under the derived edge can still change an extreme
        ratio's shape — the surprising half of a POT rule, so it says so."""
        path = self.texture(size=(96, 1))

        result = MapOptimizer.assess(path, force_pot=True)

        self.assertTrue(
            any("aspect ratio" in w.lower() for w in result["warnings"]),
            f"a reshaping snap must warn: {result['warnings']}",
        )

    def test_a_faithful_snap_does_not_warn(self):
        """Integer rounding on the derived edge is dust, not a reshape —
        warning about it would train the reader past the line that matters."""
        path = self.texture(size=(96, 72))

        result = MapOptimizer.assess(path, force_pot=True, pot_mode="down")

        self.assertEqual(
            [w for w in result["warnings"] if "aspect ratio" in w.lower()], []
        )

    def test_optimize_map_prints_the_aspect_warning(self):
        """The twins must say the same thing — a dry run that warns and a real
        run that stays quiet is the drift the plan/apply split exists to stop."""
        import io
        from contextlib import redirect_stdout

        path = self.texture(size=(96, 1))

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            MapOptimizer.optimize_map(path, force_pot=True)

        self.assertIn("aspect ratio", buffer.getvalue().lower())

    # -- The common case, and the one a future "fix" is most likely to break --
    #
    # A POT rule bounds SIZE, not shape, so a map whose LONG edge is already a
    # power of two passes through untouched and keeps a non-POT short edge.
    # That is deliberate: snapping the short edge too is what destroyed the
    # aspect ratio in the first place (1024x768 -> 1024x512). These pin the
    # trade so that "make force_pot force POT on both axes" cannot be
    # reintroduced without a failing test.

    def test_long_edge_already_pot_is_left_alone(self):
        """1024x768 is already legal on its long edge -- do not touch it."""
        for mode in ("nearest", "down"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    MapOptimizer._snap_pot(1024, 768, mode), (1024, 768)
                )

    def test_short_edge_is_deliberately_left_non_pot(self):
        """The residual non-POT short edge is the aspect ratio not being paid for."""
        _, short = MapOptimizer._snap_pot(2048, 1080, "down")
        self.assertEqual(short, 1080)
        self.assertNotEqual(
            short & (short - 1), 0, "short edge must NOT have been snapped to POT"
        )

    def test_portrait_is_handled_on_its_own_long_edge(self):
        """Orientation must not change the rule -- the LONG edge is the height here."""
        self.assertEqual(MapOptimizer._snap_pot(768, 1024, "nearest"), (768, 1024))
        self.assertEqual(MapOptimizer._snap_pot(600, 1000, "down"), (307, 512))

    def test_assess_predicts_the_untouched_size(self):
        """The dry run must agree that nothing happens, and must not warn."""
        path = self.texture(size=(1024, 768))
        result = MapOptimizer.assess(path, force_pot=True)

        self.assertEqual(
            (result["predicted"]["width"], result["predicted"]["height"]),
            (1024, 768),
        )
        self.assertFalse(result["recommended"], "nothing to do on an already-legal map")
        self.assertEqual(
            [w for w in result.get("warnings", []) if "aspect" in w.lower()], []
        )


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

    def test_explicit_pot_mode_reaches_both_twins_without_a_profile(self):
        """A caller that resolves a DeliveryBudget itself (a DCC export task
        enforcing a template's budget without adopting its container) passes
        the budget as plain max_size/force_pot — so it must also be able to
        pass the budget's never-grow snap, or 96 rounds UP to 128 and the
        enforcement inflates the asset it exists to shrink."""
        path = self.texture(size=(96, 96))

        result = MapOptimizer.assess(path, force_pot=True, pot_mode="down")
        self.assertEqual(
            (result["predicted"]["width"], result["predicted"]["height"]), (64, 64)
        )

        written = MapOptimizer.optimize_map(path, force_pot=True, pot_mode="down")
        with ImgUtils.ensure_image(written) as out:
            self.assertEqual(out.size, (64, 64))

    def test_explicit_pot_mode_outranks_the_profile_derived_snap(self):
        """Same precedence as the other advisory-tier arguments: an explicit
        value wins over what the profile derived (here, "nearest" overriding
        the enforced budget's "down")."""
        path = self.texture(size=(96, 96))

        result = MapOptimizer.assess(
            path,
            output_profile=self.PROFILE,
            enforce_budget=True,
            pot_mode="nearest",
        )
        self.assertEqual(result["predicted"]["width"], 128)  # nearest, not floor

    def test_sixteen_bit_grayscale_is_not_planned_down_to_L(self):
        """A 16-bit grayscale mode is the L target's channel layout at higher
        precision — planning it down to 8-bit L would discard the precision a
        16-bit OutputSpec deliberately pays for. Both the map-type coercion
        step and the strict-mode step must leave it alone.

        "I" is narrowed rather than left alone: Height resolves to a PNG spec,
        and Pillow's PNG writer stores mode "I" as 16-bit regardless (a path it
        deprecates for removal in Pillow 13). Naming the narrowing keeps every
        bit the container can hold AND makes assess predict the mode that
        really lands on disk — leaving it implicit had the dry run promise "I"
        for a file that reads back "I;16"."""
        from PIL import Image as PILImage

        expected = {"I": ["Mode (map_type=Height): I -> I;16"], "I;16": []}
        for mode, descriptions in expected.items():
            with self.subTest(mode=mode):
                image = PILImage.new(mode, (64, 64))
                plan = MapOptimizer.plan(image, map_type_key="Height")
                self.assertEqual([op.description for op in plan], descriptions)

    def test_eight_bit_spec_still_coerces_high_precision_source_to_L(self):
        """The high-precision tolerance is spec-blind if it isn't gated on the
        RESOLVED spec's bit depth: Roughness/AO/Metallic all target 'L' with an
        8-bit OutputSpec (only Height/Displacement/Bump are 16-bit), so an
        I;16 source under one of those map types must still plan a coercion
        down to L, or an 8-bit spec silently ships a 16-bit file."""
        from PIL import Image as PILImage

        for mode in ("I", "I;16"):
            with self.subTest(mode=mode):
                image = PILImage.new(mode, (64, 64))
                plan = MapOptimizer.plan(image, map_type_key="Roughness")
                self.assertEqual(
                    [op.description for op in plan],
                    [f"Mode (map_type=Roughness): {mode} -> L"],
                )

    def test_the_writer_and_the_dry_run_agree_on_every_builtin_profile(self):
        """optimize_map under a profile writes a file that assess (same
        profile) must call DONE — for every shipped profile. Probe-proven
        2026-08-14 before the high-precision rule: a 16-bit Height spec wrote
        an I;16 file that assessed 'recommended: I;16 -> L' on every
        subsequent run, so a task/check pair built on the twins could never
        converge."""
        out_dir = os.path.join(self.test_dir, "roundtrip")
        for index, profile in enumerate(OutputTemplates.BUILTIN):
            path = self.texture(
                name=f"probe{index}_Height.png", mode="L", size=(64, 64)
            )
            written = MapOptimizer.optimize_map(
                path, output_dir=out_dir, output_profile=profile
            )
            result = MapOptimizer.assess(written, output_profile=profile)
            with self.subTest(profile=profile):
                self.assertFalse(
                    result["recommended"],
                    f"{profile}: {result['reasons']}",
                )

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


class HighPrecisionContainerTest(_TextureFixture):
    """The 16-bit tolerance has to name the CONTAINER, not just the spec.

    Height/Displacement/Bump resolve to ``OutputSpec("png", 16)`` under every
    built-in profile, so gating the I/I;16 tolerance on the resolved bit depth
    alone let a 16-bit mode reach every container: dds raised
    ``OSError: cannot write mode I;16 as DDS`` while assess predicted a clean
    no-op, and tga widened a linear height map to 24-bit RGB (+4434%).
    """

    #: The containers the audit walked, and the on-disk mode each must end up
    #: with for a single-channel 16-bit map: only png can hold the precision;
    #: everything else reduces to the map type's own target mode, never RGB.
    #: WebP is the one widening left, and it is the container's own (it has no
    #: grayscale mode at all), applied to "L" rather than to "I;16".
    CONTAINERS = {
        "png": "I;16",
        "dds": "L",
        "tga": "L",
        "jpg": "L",
        "webp": "RGB",
    }

    def high_precision_texture(self, mode="I;16", name="rock_Height.png"):
        """A full-scale 0..65535 height ramp — the range a clip destroys."""
        path = os.path.join(self.test_dir, name)
        image = Image.new(mode, (64, 64))
        pixels = image.load()
        for y in range(64):
            for x in range(64):
                pixels[x, y] = (x * 1024 + y * 16) % 65536
        image.save(path)
        return path

    def test_the_twins_agree_on_every_container(self):
        """assess(src, output_type=X) must predict what optimize_map(src,
        output_type=X) then writes — for every container, on both
        high-precision modes. This is the contract the whole plan/apply split
        exists to hold, and the audited break was assess reporting
        ``recommended=False, reasons=[], warnings=[]`` for a run that raised.
        """
        for source_mode, suffix in (("I;16", "png"), ("I", "tif")):
            src = self.high_precision_texture(
                source_mode, name=f"twin_{suffix}_Height.{suffix}"
            )
            for ext, expected in self.CONTAINERS.items():
                with self.subTest(source_mode=source_mode, ext=ext):
                    out_dir = os.path.join(self.test_dir, f"hp_{suffix}_{ext}")
                    report = MapOptimizer.assess(src, output_type=ext)
                    written = MapOptimizer.optimize_map(
                        src, output_dir=out_dir, output_type=ext
                    )
                    with ImgUtils.ensure_image(written) as out:
                        self.assertEqual(out.mode, expected)
                        self.assertEqual(
                            out.mode,
                            report["predicted"]["mode"],
                            "the dry run named a mode the writer did not produce",
                        )
                        self.assertEqual(
                            out.size,
                            (
                                report["predicted"]["width"],
                                report["predicted"]["height"],
                            ),
                        )
                    # A run that coerces the mode is a change, and a dry run
                    # calling it a no-op is exactly what let the dds crash
                    # through unannounced.
                    self.assertEqual(
                        report["recommended"],
                        expected != source_mode,
                        report["reasons"],
                    )

    def test_the_fallback_is_the_map_targets_mode_not_rgb(self):
        """``_CONTAINER_MODE_FALLBACKS`` widens I;16 to RGB for tga — right for
        an unknown image, wrong for a single-channel map, whose own target mode
        is "L". The plan must get there first so the container never sees the
        16-bit mode."""
        src = self.high_precision_texture(name="tga_Height.png")
        written = MapOptimizer.optimize_map(
            src, output_dir=os.path.join(self.test_dir, "tga"), output_type="tga"
        )
        with ImgUtils.ensure_image(written) as out:
            self.assertEqual(out.mode, "L", "widened to RGB instead of reducing")

    def test_the_reduction_rescales_instead_of_clipping(self):
        """Pillow's I;16 -> L is a CLIP at 255, not a rescale, so the fallback
        to the 8-bit target would hand back a 99.6%-white map. The reduction
        has to keep the ramp."""
        image = Image.new("I;16", (256, 1))
        pixels = image.load()
        for x in range(256):
            pixels[x, 0] = x * 257
        plan = MapOptimizer.plan(image, map_type_key="Roughness")
        reduced = MapOptimizer.apply(image, plan)
        self.assertEqual(reduced.mode, "L")
        values = list(reduced.get_flattened_data())
        self.assertEqual(len(set(values)), 256, "the ramp was clipped, not rescaled")
        self.assertEqual((values[0], values[-1]), (0, 255))


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

    def test_float_source_is_reduced_not_flattened_to_black(self):
        """The same clip-vs-rescale trap one dtype over, and the worst of the
        family: a float map's data lives in 0..1, so Pillow's ``F`` -> ``L``
        truncation sends everything below 1.0 to 0. A 0..1 height ramp came
        back as a 2-value, essentially black 84-byte PNG reported as a
        successful -99% optimization, with ``assess`` predicting a clean L."""
        ramp = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
        path = os.path.join(self.test_dir, "rock_Height.tif")
        Image.fromarray(ramp, mode="F").save(path)

        report = MapOptimizer.assess(path, output_type="png", map_type="Height")
        written = MapOptimizer.optimize_map(
            path, output_dir=self.test_dir, output_type="png", map_type="Height"
        )
        with Image.open(written) as out:
            data = list(out.get_flattened_data())
            self.assertEqual(out.mode, report["predicted"]["mode"])
            self.assertGreater(len(set(data)), 200, "the float range was flattened")
            self.assertEqual(max(data), 255)

    def test_big_endian_source_reduces_instead_of_raising(self):
        """``I;16B`` is Pillow's mode for a big-endian 16-bit TIFF, and
        ``Image.point`` refuses byte-order-qualified modes -- so a rescale
        gated on ``startswith("I;")`` and executed with ``point`` raised
        ``ValueError: point operation not supported for this mode`` where the
        old (clipping) convert had at least produced a file. The reduction
        belongs to ``ImgUtils.convert_i_to_l``, which goes through numpy and
        reads every byte order."""
        ramp = np.array(
            [(x * 65535) // 63 for x in range(64)] * 64, dtype=np.uint16
        ).reshape(64, 64)
        source = Image.frombytes("I;16B", (64, 64), ramp.astype(">u2").tobytes())

        reduced = MapOptimizer._coerce_mode(source, "L")

        self.assertEqual(reduced.mode, "L")
        # Rescaled, not clipped: the full-scale ramp keeps its 64 steps.
        self.assertEqual(len(set(reduced.get_flattened_data())), 64)


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


class _FakeKtx2Encoder:
    """Records encode calls and writes a KTX2-magic marker file."""

    MAGIC = b"\xabKTX 20\xbb\r\n\x1a\n"

    def __init__(self):
        self.calls = []

    def encode(self, source, output, codec="UASTC", srgb=True, mipmaps=True, quality=None):
        self.calls.append({"codec": codec, "srgb": srgb, "quality": quality})
        width, height = source.size
        with open(output, "wb") as fh:
            # A real KTX 2.0 header, so assess() can introspect what a run
            # wrote: vkFormat (UNDEFINED for Basis), typeSize, then geometry.
            fh.write(self.MAGIC + struct.pack("<5I", 0, 1, width, height, 0))
            fh.write(codec.encode("ascii"))
        return output


class TestKtx2Compression(_TextureFixture):
    """The per-map Basis codec decision: derivation, refusal, and run parity."""

    def setUp(self):
        super().setUp()
        self.fake = _FakeKtx2Encoder()
        ImgUtils.register_ktx2_encoder(self.fake)
        self.addCleanup(ImgUtils.register_ktx2_encoder, None)

    # ------------------------------------------------------- resolution rules
    def test_derivation_follows_the_lossy_gate(self):
        # Perceptual sRGB color -> the low-bitrate codec; its safety condition
        # is literally MapRegistry.is_lossy_safe.
        codec, colorspace, note = MapOptimizer.resolve_compression(
            "Base_Color", "ktx2"
        )
        self.assertEqual((codec, colorspace, note), ("ETC1S", "sRGB", None))
        # Normals, packed masks, and unknown types are all UASTC.
        for map_type in ("Normal_OpenGL", "ORM", None):
            codec, _, note = MapOptimizer.resolve_compression(map_type, "ktx2")
            self.assertEqual(codec, "UASTC", map_type)
            self.assertIsNone(note)

    def test_normal_colorspace_is_linear(self):
        _, colorspace, _ = MapOptimizer.resolve_compression("Normal_OpenGL", "ktx2")
        self.assertEqual(colorspace, "Linear")

    def test_etc1s_on_a_data_map_is_upgraded_and_reported(self):
        codec, _, note = MapOptimizer.resolve_compression(
            "Normal_OpenGL", "ktx2", OutputSpec("ktx2", 8, compression="ETC1S")
        )
        self.assertEqual(codec, "UASTC")
        self.assertIn("ETC1S refused", note)

    def test_explicit_uastc_on_color_is_honored(self):
        codec, _, note = MapOptimizer.resolve_compression(
            "Base_Color", "ktx2", OutputSpec("ktx2", 8, compression="UASTC")
        )
        self.assertEqual((codec, note), ("UASTC", None))

    def test_dds_vocabulary_is_rejected_for_ktx2(self):
        with self.assertRaises(ValueError):
            MapOptimizer.resolve_compression(
                "Base_Color", "ktx2", OutputSpec("ktx2", 8, compression="DXT5")
            )

    def test_non_ktx2_targets_pass_through(self):
        spec = OutputSpec("dds", 8, compression="DXT5")
        self.assertEqual(
            MapOptimizer.resolve_compression("Base_Color", "dds", spec),
            ("DXT5", None, None),
        )

    def test_quality_gate_mirrors_the_codec_gate(self):
        # ETC1S territory keeps its dial; UASTC territory refuses it aloud.
        self.assertEqual(
            MapOptimizer.resolve_quality(70, "Base_Color", "ktx2"), (70, None)
        )
        quality, skipped = MapOptimizer.resolve_quality(70, "Normal_OpenGL", "ktx2")
        self.assertIsNone(quality)
        self.assertIn("refused", skipped)
        self.assertEqual(
            MapOptimizer.resolve_quality(None, "Base_Color", "ktx2"), (None, None)
        )

    # ------------------------------------------------------------ run parity
    def test_optimize_map_encodes_color_as_etc1s(self):
        path = self.texture("wall_Base_Color.png")
        out = MapOptimizer.optimize_map(path, output_type="ktx2")
        self.assertTrue(out.endswith(".ktx2"), out)
        self.assertTrue(os.path.isfile(out))
        self.assertEqual(self.fake.calls[-1]["codec"], "ETC1S")
        self.assertTrue(self.fake.calls[-1]["srgb"])

    def test_optimize_map_encodes_normal_as_uastc_linear(self):
        path = self.texture("wall_Normal_OpenGL.png")
        MapOptimizer.optimize_map(path, output_type="ktx2")
        self.assertEqual(self.fake.calls[-1]["codec"], "UASTC")
        self.assertFalse(self.fake.calls[-1]["srgb"])

    def test_assess_predicts_the_codec_the_run_uses(self):
        path = self.texture("wall_Normal_OpenGL.png")
        report = MapOptimizer.assess(path, output_type="ktx2")
        self.assertEqual(report["predicted"]["ext"], "ktx2")
        self.assertEqual(report["predicted"]["compression"], "UASTC")
        MapOptimizer.optimize_map(path, output_type="ktx2")
        self.assertEqual(
            report["predicted"]["compression"], self.fake.calls[-1]["codec"]
        )

    def test_assess_surfaces_the_ktx2_quality_refusal(self):
        """The end-to-end cover for ``resolve_quality``'s ktx2 branch: an
        explicit dial on a UASTC map is refused, and the refusal reaches the
        report. With no profile there is no OutputSpec, so nothing here can
        exercise ``resolve_compression``'s note — that wiring is pinned
        separately below.
        """
        path = self.texture("wall_Normal_OpenGL.png")
        report = MapOptimizer.assess(
            path,
            output_type="ktx2",
            lossy_quality=80,
        )
        self.assertTrue(
            [w for w in report["warnings"] if "refused" in w], report["warnings"]
        )

    def test_assess_surfaces_the_profiles_etc1s_refusal(self):
        """A template that names ETC1S for a map the codec bands has its
        upgrade REPORTED, not applied in silence — the dry-run half of
        ``resolve_compression``'s refusal. This needs a profile: the note only
        exists when an OutputSpec carries a compression request, so a
        profile-less assess never reaches the branch at all.
        """
        profile = "test-etc1s-profile"
        OutputTemplates.BUILTIN[profile] = OutputTemplate(
            default=OutputSpec("ktx2", 8, compression="ETC1S")
        )
        self.addCleanup(OutputTemplates.BUILTIN.pop, profile, None)

        path = self.texture("wall_Normal_OpenGL.png")
        report = MapOptimizer.assess(path, output_profile=profile)
        self.assertEqual(report["predicted"]["compression"], "UASTC")
        self.assertTrue(
            [w for w in report["warnings"] if "ETC1S refused" in w],
            report["warnings"],
        )
        # No quality was requested, so resolve_quality contributes nothing --
        # the warning above can only have come from resolve_compression.
        self.assertIsNone(report["predicted"]["quality"])

    def test_assess_reads_back_the_ktx2_the_run_just_wrote(self):
        """The consumer is the exporters' optimize-textures gate, which
        assesses the staged output the task just wrote — and PIL cannot open a
        Basis payload, so assess returned ``error="Failed to read image"`` with
        ``predicted={}``: a read failure reported for every ktx2 map the gate
        delivers, and a KeyError for anything reading predicted["width"].
        """
        path = self.texture("wall_Base_Color.png", size=(128, 64))
        written = MapOptimizer.optimize_map(path, output_type="ktx2")

        report = MapOptimizer.assess(written)
        self.assertIsNone(report.get("error"))
        self.assertEqual(report["current"]["format"], "KTX2")
        self.assertEqual(
            (report["predicted"]["width"], report["predicted"]["height"]), (128, 64)
        )
        # Nothing here can transcode a Basis payload back to pixels, so a
        # delivered file is by definition done.
        self.assertFalse(report["recommended"], report["reasons"])
        self.assertEqual(report["predicted"]["ext"], "ktx2")

    def test_assess_still_reports_a_corrupt_ktx2(self):
        """The header reader must not turn a truncated/garbage file into a
        confident report of its own."""
        broken = os.path.join(self.test_dir, "broken_Base_Color.ktx2")
        with open(broken, "wb") as fh:
            fh.write(b"not a ktx2 file at all")
        report = MapOptimizer.assess(broken)
        self.assertIn("Failed to read image", report.get("error") or "")
        self.assertEqual(report["predicted"], {})

    def test_predict_size_encodes_through_the_fake(self):
        path = self.texture("wall_Base_Color.png")
        report = MapOptimizer.assess(path, output_type="ktx2", predict_size=True)
        size = report["predicted"]["size_bytes"]
        self.assertIsNotNone(size, report["predicted"].get("size_error"))
        self.assertGreater(size, 0)


class TestResolveSizeClamp(_TextureFixture):
    """``resolve_size_clamp`` — the one interpretation of a "max size" dial.

    Extracted 2026-08-17 from byte-identical private copies in the mayatk and
    blendertk scene exporters (BACKLOG "settings-definition row builder
    duplicated verbatim in blendertk"), so both DCCs and any future caller
    answer the dial identically instead of each restating the rule.
    """

    def test_off_modes_never_clamp(self):
        """Falsy and the literal OFF token leave a template's budget ADVISORY.

        ``True`` is rejected explicitly: ``int(True)`` is 1, which would
        silently clamp every map to a single pixel.
        """
        for off in (0, None, "", "OFF", "off", False, True):
            self.assertEqual(
                MapOptimizer.resolve_size_clamp(off, "glTF 2.0"), {}, repr(off)
            )

    def test_positive_value_is_a_hard_ceiling(self):
        """Ints and their string form (a hand-edited preset sends strings)."""
        self.assertEqual(MapOptimizer.resolve_size_clamp(1024), {"max_size": 1024})
        self.assertEqual(
            MapOptimizer.resolve_size_clamp("2048", "glTF 2.0"), {"max_size": 2048}
        )

    def test_sentinel_enforces_the_template_budget_without_pot(self):
        """The budget's POT rule is deliberately not adopted — snapping each
        axis independently would reshape a non-square map, and a ceiling only
        ever shrinks and keeps aspect. With no template there is no budget."""
        self.assertEqual(
            MapOptimizer.resolve_size_clamp(
                MapOptimizer.SIZE_CLAMP_TEMPLATE, "glTF 2.0"
            ),
            {"enforce_budget": True, "force_pot": False},
        )
        self.assertEqual(
            MapOptimizer.resolve_size_clamp(MapOptimizer.SIZE_CLAMP_TEMPLATE, None), {}
        )

    def test_sentinel_stays_truthy_and_out_of_pixel_range(self):
        """0/None already mean "no clamp", so the sentinel must be truthy, and
        it must never collide with a real pixel dimension."""
        self.assertTrue(MapOptimizer.SIZE_CLAMP_TEMPLATE)
        self.assertLess(MapOptimizer.SIZE_CLAMP_TEMPLATE, 0)

    def test_unparseable_value_warns_and_declines_rather_than_guessing(self):
        warnings = []

        class _Logger:
            def warning(self, message):
                warnings.append(message)

        self.assertEqual(
            MapOptimizer.resolve_size_clamp("wide", "glTF 2.0", logger=_Logger()), {}
        )
        self.assertEqual(len(warnings), 1, warnings)
        # A logger is optional — an omitted one must not raise.
        self.assertEqual(MapOptimizer.resolve_size_clamp("wide"), {})

    def test_describe_matches_what_resolve_returns(self):
        """Empty when nothing is clamped, so a caller can splice it into a
        sentence without branching."""
        self.assertEqual(MapOptimizer.describe_size_clamp(0), "")
        self.assertEqual(MapOptimizer.describe_size_clamp(1024), "clamped to 1024 px")
        budget_size = getattr(OutputTemplates.budget("glTF 2.0"), "max_size", None)
        self.assertEqual(
            MapOptimizer.describe_size_clamp(
                MapOptimizer.SIZE_CLAMP_TEMPLATE, "glTF 2.0"
            ),
            f"clamped to the template's budget ({budget_size} px)",
        )

    def test_result_is_accepted_by_assess(self):
        """The returned dict is kwargs for the pass it describes — pinned so a
        renamed optimizer parameter cannot leave this resolver behind."""
        path = os.path.join(self.test_dir, "rock_BaseColor.png")
        Image.new("RGB", (4096, 2048), (120, 120, 120)).save(path)

        report = MapOptimizer.assess(
            path, **MapOptimizer.resolve_size_clamp(2048, "glTF 2.0")
        )
        self.assertEqual(
            (report["predicted"]["width"], report["predicted"]["height"]),
            (2048, 1024),
        )
        report = MapOptimizer.assess(
            path,
            output_profile="glTF 2.0",
            **MapOptimizer.resolve_size_clamp(
                MapOptimizer.SIZE_CLAMP_TEMPLATE, "glTF 2.0"
            ),
        )
        self.assertEqual(
            max(report["predicted"]["width"], report["predicted"]["height"]),
            getattr(OutputTemplates.budget("glTF 2.0"), "max_size", None),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
