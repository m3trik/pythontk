# !/usr/bin/python
# coding=utf-8
"""Tests for :class:`pythontk.SequenceExporter` / :class:`pythontk.SequenceEncoder`.

The host-independent half of a playblast: the target registry, the plan that
captures once and derives every encoded output from it, the frame bookkeeping,
and the seam a host produces its own output kinds through. Maya's exporter is
tested against a real viewport in ``mayatk/test/test_playblast_exporter.py``;
what is checked here is the contract that makes the SAME code serve a producer
that has no viewport at all.
"""

import os
import sys
import unittest
import unittest.mock
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk.file_utils.temp_artifacts import TempArtifacts
from pythontk.vid_utils.sequence_exporter import (
    CaptureResult,
    ExportTarget,
    SequenceEncoder,
    SequenceExporter,
)


class _FakeExporter(SequenceExporter):
    """A producer that writes placeholder frames instead of reading a viewport.

    Stands in for a host without needing one: the plan under test cares that a
    capture happened, once per format, over the resolved range -- not what the
    pixels were.
    """

    RANGE_MODES = ("custom", "playback")
    DEFAULT_RANGE_MODE = "playback"
    PLAYBACK_RANGE = (1, 5)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.captures: List[Dict[str, Any]] = []
        self.stills: List[str] = []
        self.fail_formats: set = set()

    @classmethod
    def _frame_range_for_mode(cls, mode):
        return cls.PLAYBACK_RANGE

    def sequence_name(self) -> str:
        return "fake"

    def sequence_fps(self) -> float:
        return 25.0

    def capture_sequence(
        self,
        directory,
        prefix=None,
        start=None,
        end=None,
        camera=None,
        image_format="png",
        **overrides,
    ) -> CaptureResult:
        if image_format in self.fail_formats:
            raise RuntimeError(f"capture of {image_format} refused")
        os.makedirs(directory, exist_ok=True)
        prefix = prefix or self.sequence_name()
        self.captures.append(
            {"format": image_format, "start": start, "end": end, "dir": directory}
        )
        frames = []
        for number in range(int(start), int(end) + 1):
            path = os.path.join(
                directory, f"{prefix}.{number:0{self.frame_padding}d}.{image_format}"
            )
            with open(path, "wb") as handle:
                handle.write(b"frame")
            frames.append(path)
        return CaptureResult(
            directory=directory,
            prefix=prefix,
            image_format=image_format,
            start=int(start),
            end=int(end),
            padding=self.frame_padding,
            frames=frames,
            fps=self.sequence_fps(),
        )

    def capture_still(
        self, filepath, frame=None, camera=None, image_format="png", **overrides
    ) -> str:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as handle:
            handle.write(b"still")
        self.stills.append(filepath)
        return filepath

    def encode_sequence(self, capture, output_filepath, **kwargs) -> str:
        """Stub the ffmpeg call: the encode itself is covered by test_vid."""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "wb") as handle:
            handle.write(b"movie")
        return output_filepath


class _HostExporter(_FakeExporter):
    """A host with an output kind of its own — Maya's ``native``, in miniature."""

    TARGETS = {
        **SequenceExporter.TARGETS,
        "avi": ExportTarget(
            "avi", "AVI", "native", extension="avi", native_format="avi"
        ),
    }

    def _export_extra_target(self, spec, output_dir, name, **kwargs):
        path = os.path.join(output_dir, f"{name}.{spec.extension}")
        with open(path, "wb") as handle:
            handle.write(b"native")
        return path


class SequenceExporterTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = TempArtifacts("test_sequence_exporter", policy="scoped")
        self.out = self.temp.dir_path()
        self.exporter = _FakeExporter()
        # ffmpeg is never invoked (encode_sequence is stubbed), but the plan
        # PRE-FLIGHTS it before capturing — so it must look present.
        self._ffmpeg = unittest.mock.patch(
            "pythontk.vid_utils.sequence_exporter.VidUtils.resolve_ffmpeg",
            return_value="ffmpeg",
        )
        self._ffmpeg.start()

    def tearDown(self):
        self._ffmpeg.stop()
        self.temp.cleanup()

    # -- registry -------------------------------------------------------

    def test_target_registry_integrity(self):
        targets = SequenceExporter.available_targets()
        self.assertEqual(len(targets), len(SequenceExporter.TARGETS))
        for name, spec in SequenceExporter.TARGETS.items():
            self.assertEqual(name, spec.name)
            self.assertTrue(spec.label and spec.extension)
            self.assertIn(spec.kind, ("encode", "sequence", "still"))

    def test_a_host_extends_the_registry_without_reordering_it(self):
        """A host appends; the shared entries keep their index (pickers persist
        their choice by index, so a reorder would silently repoint it)."""
        shared = list(SequenceExporter.TARGETS)
        self.assertEqual(list(_HostExporter.TARGETS)[: len(shared)], shared)

    def test_quality_to_crf_mapping(self):
        self.assertEqual(SequenceEncoder._quality_to_crf(100), 16)
        self.assertEqual(SequenceEncoder._quality_to_crf(0), 40)
        self.assertEqual(SequenceEncoder._quality_to_crf(999), 16)  # clamped
        self.assertGreater(
            SequenceEncoder._quality_to_crf(50), SequenceEncoder._quality_to_crf(90)
        )

    # -- ranges ---------------------------------------------------------

    def test_custom_range_needs_both_bounds_and_must_run_forwards(self):
        self.assertEqual(SequenceExporter.resolve_frame_range("custom", 3, 9), (3, 9))
        with self.assertRaises(ValueError):
            SequenceExporter.resolve_frame_range("custom")
        with self.assertRaises(ValueError):
            SequenceExporter.resolve_frame_range("custom", 10, 2)

    def test_an_unregistered_range_mode_is_refused(self):
        with self.assertRaises(ValueError):
            SequenceExporter.resolve_frame_range("playback")  # not in the base

    def test_a_host_mode_resolves_and_explicit_bounds_still_override_it(self):
        self.assertEqual(_FakeExporter.resolve_frame_range("playback"), (1, 5))
        self.assertEqual(_FakeExporter.resolve_frame_range("playback", start=3), (3, 5))

    def test_a_host_export_that_names_no_range_uses_the_hosts_default(self):
        """Not the base's. ``custom`` is the only mode an exporter with no
        timeline HAS, and it requires both bounds -- so a host inheriting it as
        its export default turns ``export(dir, targets="mp4")`` into a
        ValueError for every caller that ever relied on the timeline."""
        results = self.exporter.export(self.out, targets=["mp4"])

        self.assertTrue(results[0].ok, results[0].error)
        self.assertEqual(
            (self.exporter.captures[0]["start"], self.exporter.captures[0]["end"]),
            _FakeExporter.PLAYBACK_RANGE,
        )

    def test_an_exporter_with_no_timeline_still_demands_its_bounds(self):
        """The other half of the rule above: nothing silently invents a range."""
        with self.assertRaises(ValueError):
            SequenceExporter().export(self.out, targets=["mp4"])

    def test_a_declared_mode_with_no_resolver_says_so(self):
        class _Sloppy(SequenceExporter):
            RANGE_MODES = ("custom", "playback")

        with self.assertRaises(NotImplementedError):
            _Sloppy.resolve_frame_range("playback")

    # -- the plan -------------------------------------------------------

    def test_one_capture_feeds_every_encode(self):
        results = self.exporter.export(
            self.out, targets=["mp4", "mov"], range_mode="playback"
        )
        self.assertEqual(len(self.exporter.captures), 1)
        self.assertEqual(self.exporter.captures[0]["format"], "png")
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual({r.target for r in results}, {"mp4", "mov"})

    def test_an_encode_rides_a_requested_png_sequence_rather_than_a_second_capture(
        self,
    ):
        self.exporter.export(
            self.out, targets=["mp4", "png_sequence"], range_mode="playback"
        )
        self.assertEqual(len(self.exporter.captures), 1)
        # ...and the frames the user ASKED for are kept.
        self.assertTrue(os.path.isdir(os.path.join(self.out, "fake_png")))

    def test_encode_only_cleans_its_intermediate_frames(self):
        self.exporter.export(self.out, targets=["mp4"], range_mode="playback")
        self.assertFalse(os.path.exists(os.path.join(self.out, "fake_png_tmp")))

    def test_keep_frames_leaves_the_intermediate_capture(self):
        self.exporter.export(
            self.out, targets=["mp4"], range_mode="playback", keep_frames=True
        )
        self.assertTrue(os.path.isdir(os.path.join(self.out, "fake_png_tmp")))

    def test_a_capture_failure_isolates_only_the_targets_that_needed_it(self):
        self.exporter.fail_formats = {"jpg"}
        results = {
            r.target: r
            for r in self.exporter.export(
                self.out, targets=["mp4", "jpg_sequence"], range_mode="playback"
            )
        }
        self.assertTrue(results["mp4"].ok)
        self.assertFalse(results["jpg_sequence"].ok)
        self.assertIn("refused", results["jpg_sequence"].error)

    def test_frames_of_a_capture_that_raised_are_still_swept(self):
        """The sweep is by disk scan, not by CaptureResult: an interrupted
        capture has no result yet leaves frames, and an mp4 request must not
        end as a folder of PNGs."""

        def half_then_fail(directory, prefix=None, **kwargs):
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, f"{prefix}.0001.png"), "wb") as handle:
                handle.write(b"frame")
            raise RuntimeError("interrupted")

        self.exporter.capture_sequence = half_then_fail
        results = self.exporter.export(self.out, targets=["mp4"], range_mode="playback")
        self.assertFalse(results[0].ok)
        self.assertFalse(os.path.exists(os.path.join(self.out, "fake_png_tmp")))

    def test_without_ffmpeg_an_encode_fails_before_anything_is_captured(self):
        self._ffmpeg.stop()
        with unittest.mock.patch(
            "pythontk.vid_utils.sequence_exporter.VidUtils.resolve_ffmpeg",
            side_effect=FileNotFoundError("no ffmpeg"),
        ):
            results = self.exporter.export(
                self.out, targets=["mp4"], range_mode="playback"
            )
        self._ffmpeg.start()
        self.assertFalse(results[0].ok)
        self.assertEqual(self.exporter.captures, [])

    def test_an_unknown_target_is_refused_before_the_plan_starts(self):
        with self.assertRaises(ValueError):
            self.exporter.export(self.out, targets=["mp4", "bogus"])
        self.assertEqual(self.exporter.captures, [])

    def test_progress_runs_to_done(self):
        seen = []
        self.exporter.export(
            self.out,
            targets=["mp4", "png_sequence"],
            range_mode="playback",
            progress_callback=lambda step, total, label: seen.append(
                (step, total, label)
            ),
        )
        self.assertEqual(seen[-1][2], "Done")
        self.assertEqual(seen[-1][0], seen[-1][1])

    # -- the host seam --------------------------------------------------

    def test_a_host_kind_is_produced_through_the_seam(self):
        results = {
            r.target: r
            for r in _HostExporter().export(
                self.out, targets=["avi", "mp4"], range_mode="playback"
            )
        }
        self.assertTrue(results["avi"].ok)
        self.assertTrue(os.path.isfile(results["avi"].output))
        self.assertTrue(results["mp4"].ok)

    def test_a_kind_with_no_producer_fails_that_target_alone(self):
        class _Unanswered(_FakeExporter):
            TARGETS = {
                **SequenceExporter.TARGETS,
                "weird": ExportTarget("weird", "Weird", "weird", extension="wrd"),
            }

        results = {
            r.target: r
            for r in _Unanswered().export(
                self.out, targets=["weird", "mp4"], range_mode="playback"
            )
        }
        self.assertFalse(results["weird"].ok)
        self.assertIn("weird", results["weird"].error)
        self.assertTrue(results["mp4"].ok)

    def test_a_bare_exporter_states_what_it_cannot_do(self):
        with self.assertRaises(NotImplementedError):
            SequenceExporter().capture_sequence(self.out, start=1, end=2)
        with self.assertRaises(NotImplementedError):
            SequenceExporter().capture_still(os.path.join(self.out, "x.png"))

    # -- bookkeeping ----------------------------------------------------

    def test_collect_frames_sorts_numerically_and_honours_bounds(self):
        for number in (9999, 10000, 10001):
            with open(os.path.join(self.out, f"seq.{number}.png"), "wb") as handle:
                handle.write(b"f")
        frames = SequenceEncoder._collect_frames(self.out, "seq", "png", 9999, 10001)
        self.assertEqual(
            [os.path.basename(f) for f in frames],
            ["seq.9999.png", "seq.10000.png", "seq.10001.png"],
        )
        self.assertEqual(
            len(SequenceEncoder._collect_frames(self.out, "seq", "png", 10000, 10000)),
            1,
        )

    def test_capture_result_pattern_is_a_printf_sequence(self):
        capture = CaptureResult(
            directory=self.out,
            prefix="shot",
            image_format="png",
            start=101,
            end=110,
            padding=4,
            frames=[],
            fps=24.0,
        )
        self.assertTrue(capture.pattern.endswith("shot.%04d.png"))

    def test_audio_offset_is_rebased_onto_the_captures_first_frame(self):
        """Audio armed at frame 20 in a capture that starts at frame 10 sits
        10 frames in, not 20 -- the movie starts where the capture did."""

        class _WithAudio(SequenceEncoder):
            def _resolve_audio_source(self):
                return __file__, 20.0  # any real file

        path, offset = _WithAudio()._resolve_audio(True, capture_start=10, fps=25.0)
        self.assertEqual(path, __file__)
        self.assertAlmostEqual(offset, 0.4)

    def test_a_missing_audio_source_is_silently_no_audio(self):
        self.assertEqual(SequenceEncoder()._resolve_audio(True, 0, 24.0), (None, 0.0))


if __name__ == "__main__":
    unittest.main()
