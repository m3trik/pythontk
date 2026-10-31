# !/usr/bin/python
# coding=utf-8
"""Tests for :class:`pythontk.SequenceExporter` / :class:`pythontk.SequenceEncoder`.

The host-independent half of a playblast: the target registry, the plan that
captures once and derives every encoded output from it, the frame bookkeeping
around each capture, and the adapter seam (:class:`pythontk.SequenceHost`) a
host's pixels and its own output kinds come through. Everything here runs over
a fake host adapter -- a plain-Python stand-in for a DCC viewport -- so the
plan, the file layout and the ffmpeg arguments are pinned in CI; Maya's and
Blender's adapters are tested against their hosts in
``mayatk/test/test_playblast_exporter.py`` and
``blendertk/test/test_playblast_exporter.py``.
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
    PureSequenceHost,
    SequenceEncoder,
    SequenceExporter,
)

#: The shared registry plus a host's own movie writer and a kind only it has
#: (Maya's ``native`` AVI and ``arnold``, in miniature).
_HOST_TARGETS = {
    **SequenceExporter.TARGETS,
    "avi": ExportTarget(
        "avi", "AVI", "native", extension="avi", native_format="avi",
        native_compression="none",
    ),
    "render": ExportTarget("render", "Render", "render", extension="exr"),
}


class _FakeHost(PureSequenceHost):
    """A host adapter that writes placeholder frames instead of reading a
    viewport: the plan under test cares that a capture happened, once per
    format, over the resolved range, through the right camera -- not what the
    pixels were. Its still is the pure host's (a one-frame capture moved into
    place) unless :attr:`native_still` is set."""

    PLAYBACK_RANGE = (1, 5)
    CURRENT_FRAME = 3

    def __init__(self, audio=None, sound="track1", native_still=True):
        self.captures: List[Dict[str, Any]] = []
        self.stills: List[Dict[str, Any]] = []
        self.movies: List[Dict[str, Any]] = []
        self.extras: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self.warnings: List[str] = []
        self.fail_formats: set = set()
        self.short_by = 0
        self.audio = audio
        self.sound = sound
        self.native_still = native_still

    def frame_range(self, mode):
        if mode == "current":
            return self.CURRENT_FRAME, self.CURRENT_FRAME
        return self.PLAYBACK_RANGE

    @staticmethod
    def sequence_name():
        return "fake"

    @staticmethod
    def sequence_fps():
        return 25.0

    def sound_source(self):
        return self.sound

    def audio_source(self):
        return (self.audio, 20.0) if self.audio else (None, 0.0)

    def capture_frames(self, exporter, directory, prefix, start, end, camera,
                       image_format, padding, overrides):
        if image_format in self.fail_formats:
            raise RuntimeError(f"capture of {image_format} refused")
        self.captures.append({
            "format": image_format, "start": start, "end": end, "dir": directory,
            "camera": camera, "padding": padding, "overrides": dict(overrides),
            "size": (exporter.width, exporter.height),
        })
        for number in range(int(start), int(end) + 1 - self.short_by):
            path = os.path.join(directory, f"{prefix}.{number:0{padding}d}.{image_format}")
            with open(path, "wb") as handle:
                handle.write(b"frame")

    def capture_still(self, exporter, filepath, frame, camera, image_format,
                      overrides):
        if not self.native_still:
            return super().capture_still(
                exporter, filepath, frame, camera, image_format, overrides
            )
        self.stills.append({"path": filepath, "frame": frame, "camera": camera})
        with open(filepath, "wb") as handle:
            handle.write(b"still")

    def capture_movie(self, exporter, filepath, fmt, compression, start, end,
                      camera, sound, overrides):
        self.movies.append({
            "path": filepath, "fmt": fmt, "compression": compression,
            "range": (start, end), "camera": camera, "sound": sound,
            "overrides": dict(overrides),
        })
        with open(filepath, "wb") as handle:
            handle.write(b"movie")
        return filepath

    def export_target(self, exporter, spec, output_dir, name, start, end, camera,
                      sound, overrides):
        if spec.kind != "render":
            return super().export_target(
                exporter, spec, output_dir, name, start, end, camera, sound, overrides
            )
        self.extras.append({
            "name": name, "range": (start, end), "camera": camera,
            "overrides": dict(overrides),
        })
        path = os.path.join(output_dir, f"{name}.{spec.extension}")
        with open(path, "wb") as handle:
            handle.write(b"render")
        return path

    def notify(self, message):
        self.notes.append(message)
        return True

    def warn(self, message):
        self.warnings.append(message)
        return True


def _bind(host, **config):
    """A binding of the shared class to *host*: configuration only, the shape
    mayatk's and blendertk's ``PlayblastExporter`` have."""
    config.setdefault("RANGE_MODES", ("custom", "playback", "current"))
    config.setdefault("DEFAULT_RANGE_MODE", "playback")
    return type("_BoundExporter", (SequenceExporter,), {"HOST": host, **config})


class SequenceExporterTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = TempArtifacts("test_sequence_exporter", policy="scoped")
        self.out = self.temp.dir_path()
        self.host = _FakeHost()
        self.Exporter = _bind(self.host, TARGETS=_HOST_TARGETS)
        self.exporter = self.Exporter()
        # ffmpeg is never run: the plan PRE-FLIGHTS it before capturing (so it
        # must look present), and the encode's call into it is recorded -- the
        # arguments ARE the encode's contract -- and writes the movie.
        self.compress_calls: List[Dict[str, Any]] = []
        patches = (
            unittest.mock.patch(
                "pythontk.vid_utils.sequence_exporter.VidUtils.resolve_ffmpeg",
                return_value="ffmpeg",
            ),
            unittest.mock.patch(
                "pythontk.vid_utils.sequence_exporter.VidUtils.compress_video",
                side_effect=self._compress,
            ),
            unittest.mock.patch.object(
                SequenceEncoder, "_audio_peak_db", return_value=None
            ),
        )
        self._ffmpeg = patches[0]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self.temp.cleanup()

    def _compress(self, **kwargs):
        self.compress_calls.append(dict(kwargs))
        with open(kwargs["output_filepath"], "wb") as handle:
            handle.write(b"movie")
        return kwargs["output_filepath"]

    def _files(self):
        return sorted(
            os.path.relpath(os.path.join(root, name), self.out).replace("\\", "/")
            for root, _dirs, names in os.walk(self.out)
            for name in names
        )

    def _wav(self):
        path = os.path.join(self.out, "track.wav")
        with open(path, "wb") as handle:
            handle.write(b"RIFF")
        return path

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
        self.assertEqual(list(self.Exporter.TARGETS)[: len(shared)], shared)

    def test_quality_to_crf_mapping(self):
        self.assertEqual(SequenceEncoder._quality_to_crf(100), 16)
        self.assertEqual(SequenceEncoder._quality_to_crf(0), 40)
        self.assertEqual(SequenceEncoder._quality_to_crf(999), 16)  # clamped
        self.assertGreater(
            SequenceEncoder._quality_to_crf(50), SequenceEncoder._quality_to_crf(90)
        )

    # -- what the host knows --------------------------------------------

    def test_name_and_rate_are_the_hosts_class_level(self):
        """``scene_name`` / ``scene_fps`` answer before any exporter exists (a
        panel's placeholder), and agree with the instance's."""
        self.assertEqual(self.Exporter.scene_name(), "fake")
        self.assertEqual(self.Exporter.scene_fps(), 25.0)
        self.assertEqual(self.exporter.sequence_name(), "fake")
        self.assertEqual(self.exporter.sequence_fps(), 25.0)

    def test_a_host_that_does_not_say_falls_back_to_the_defaults(self):
        self.assertEqual(SequenceExporter.scene_name(), SequenceEncoder.DEFAULT_NAME)
        self.assertEqual(SequenceExporter.scene_fps(), SequenceEncoder.DEFAULT_FPS)
        self.assertEqual(SequenceEncoder().sequence_fps(), 24.0)

    def test_the_hosts_lines_take_its_notices_else_the_logger(self):
        logger = self.exporter.logger
        with unittest.mock.patch.object(
            logger, "info"
        ) as info, unittest.mock.patch.object(logger, "warning") as warning:
            self.exporter._notify("made")
            self.exporter._warn("broke")
        info.assert_not_called()
        warning.assert_not_called()
        self.assertEqual((self.host.notes, self.host.warnings), (["made"], ["broke"]))

        bare = SequenceEncoder()
        with self.assertLogs(bare.logger, level="INFO") as logs:
            bare._notify("made")
            bare._warn("broke")
        self.assertEqual(
            [r.levelname for r in logs.records], ["INFO", "WARNING"]
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
        self.assertEqual(self.Exporter.resolve_frame_range("playback"), (1, 5))
        self.assertEqual(self.Exporter.resolve_frame_range("playback", start=3), (3, 5))
        self.assertEqual(self.Exporter.resolve_frame_range("current"), (3, 3))

    def test_a_host_export_that_names_no_range_uses_the_hosts_default(self):
        """Not the base's. ``custom`` is the only mode an exporter with no
        timeline HAS, and it requires both bounds -- so a host inheriting it as
        its export default turns ``export(dir, targets="mp4")`` into a
        ValueError for every caller that ever relied on the timeline."""
        results = self.exporter.export(self.out, targets=["mp4"])

        self.assertTrue(results[0].ok, results[0].error)
        capture = self.host.captures[0]
        self.assertEqual((capture["start"], capture["end"]), _FakeHost.PLAYBACK_RANGE)

    def test_an_exporter_with_no_timeline_still_demands_its_bounds(self):
        """The other half of the rule above: nothing silently invents a range."""
        with self.assertRaises(ValueError):
            SequenceExporter().export(self.out, targets=["mp4"])

    def test_a_declared_mode_with_no_resolver_says_so(self):
        sloppy = _bind(PureSequenceHost())

        with self.assertRaises(NotImplementedError):
            sloppy.resolve_frame_range("playback")

    # -- what the exporters produce (pinned) ----------------------------

    def test_the_layout_and_the_ffmpeg_arguments_are_pinned(self):
        """``<dir>/<name>.<ext>`` per movie off ONE png capture in
        ``<dir>/<name>_png/`` (requested, so kept), frames numbered by the real
        frame; each encode is ffmpeg from that sequence's first frame at the
        host's rate, the CRF from the quality, even dimensions forced."""
        exporter = self.Exporter(quality=80)
        results = exporter.export(
            self.out, name="shot", targets=["mp4", "mov", "png_sequence"],
            range_mode="custom", start=3, end=5,
        )

        self.assertTrue(all(r.ok for r in results), [r.error for r in results])
        self.assertEqual(
            self._files(),
            ["shot.mov", "shot.mp4", "shot_png/shot.0003.png",
             "shot_png/shot.0004.png", "shot_png/shot.0005.png"],
        )
        png_dir = os.path.join(self.out, "shot_png").replace("\\", "/")
        self.assertEqual(
            [{k: v for k, v in c.items() if k != "output_filepath"}
             for c in self.compress_calls],
            [
                {
                    "input_filepath": f"{png_dir}/shot.%04d.png",
                    "frame_rate": 25.0,
                    "start_number": 3,
                    "audio_filepath": None,
                    "audio_offset": 0.0,
                    "crf": 21,
                    "vf": "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "movflags": "+faststart",
                }
            ] * 2,
        )
        self.assertEqual(
            [os.path.basename(c["output_filepath"]) for c in self.compress_calls],
            ["shot.mp4", "shot.mov"],
        )
        self.assertEqual(
            self.host.notes,
            [f"Encoded movie created: {r.output}" for r in results[:2]],
        )

    def test_an_export_names_its_outputs_after_the_host_scene(self):
        results = self.exporter.export(self.out, targets=["still", "tif_sequence"])
        self.assertEqual(
            [os.path.relpath(r.output if isinstance(r.output, str) else r.output[0],
                             self.out).replace("\\", "/") for r in results],
            ["fake.png", "fake_tif/fake.0001.tif"],
        )

    def test_audio_is_muxed_only_when_asked_and_the_host_has_a_source(self):
        """The armed source decides whether an encode asks for audio at all;
        the file the host names is muxed at its offset rebased onto the
        capture's first frame (armed at 20, capture from 3, 25 fps: 0.68 s)."""
        self.host.audio = self._wav()
        self.Exporter(include_audio=True).export(
            self.out, name="a", targets="mp4", range_mode="custom", start=3, end=6
        )
        self.Exporter().export(self.out, name="b", targets="mp4")
        self.host.sound = None
        self.Exporter(include_audio=True).export(self.out, name="c", targets="mp4")

        muxed = [(c["audio_filepath"], c["audio_offset"]) for c in self.compress_calls]
        self.assertEqual(muxed[0][0], self.host.audio)
        self.assertAlmostEqual(muxed[0][1], 0.68)
        self.assertEqual(muxed[1:], [(None, 0.0), (None, 0.0)])

    def test_a_native_movie_is_the_hosts_writer_with_its_sound(self):
        exporter = self.Exporter(include_audio=True, camera="cam1")
        results = exporter.export(
            self.out, name="nat", targets=["avi"], range_mode="custom", start=2,
            end=4, flag=1,
        )

        self.assertTrue(results[0].ok, results[0].error)
        self.assertEqual(os.path.basename(results[0].output), "nat.avi")
        self.assertEqual(
            {k: v for k, v in self.host.movies[0].items() if k != "path"},
            {"fmt": "avi", "compression": "none", "range": (2, 4), "camera": "cam1",
             "sound": "track1", "overrides": {"flag": 1}},
        )
        self.assertEqual(self.host.captures, [], "a native movie reads no capture")
        self.assertIn(f"Playblast movie created: {results[0].output}", self.host.notes)

    def test_a_movie_the_host_did_not_write_fails_that_target(self):
        self.host.capture_movie = lambda *args: os.path.join(self.out, "missing.avi")
        results = self.exporter.export(self.out, name="m", targets=["avi", "still"])
        self.assertIn("no file", results[0].error)
        self.assertTrue(results[1].ok)

    def test_a_captures_settings_and_overrides_reach_the_host(self):
        """Reserved keys are the plan's; ``frame_padding`` is the shared
        capture's; the rest are the host's capture options, passed through."""
        exporter = self.Exporter(width=320, height=200, frame_padding=6, camera="c")
        exporter.export(
            self.out, name="pad", targets=["png_sequence", "mp4"],
            frame_padding=3, filepath="ignored", custom_flag=1,
        )
        capture = self.host.captures[0]
        self.assertEqual(
            (capture["padding"], capture["overrides"], capture["camera"],
             capture["size"]),
            (3, {"custom_flag": 1}, "c", (320, 200)),
        )
        self.assertTrue(self.compress_calls[0]["input_filepath"].endswith("pad.%03d.png"))

    def test_a_sequences_digits_never_reach_a_movie_a_still_or_a_host_kind(self):
        """``frame_padding`` is the image capture's own: the host's movie
        writer, its still and its own kinds are handed the rest (Maya's
        playblast refuses a flag it does not know)."""
        self.host.native_still = True
        results = self.exporter.export(
            self.out, name="pad", targets=["png_sequence", "avi", "render", "still"],
            frame_padding=3, flag=1,
        )

        self.assertTrue(all(r.ok for r in results), [r.error for r in results])
        self.assertEqual(self.host.captures[0]["padding"], 3)
        self.assertEqual(self.host.movies[0]["overrides"], {"flag": 1})
        self.assertEqual(self.host.extras[0]["overrides"], {"flag": 1})

    # -- the plan -------------------------------------------------------

    def test_one_capture_feeds_every_encode(self):
        results = self.exporter.export(
            self.out, targets=["mp4", "mov"], range_mode="playback"
        )
        self.assertEqual(len(self.host.captures), 1)
        self.assertEqual(self.host.captures[0]["format"], "png")
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual({r.target for r in results}, {"mp4", "mov"})

    def test_an_encode_rides_a_requested_png_sequence_rather_than_a_second_capture(
        self,
    ):
        self.exporter.export(
            self.out, targets=["mp4", "png_sequence"], range_mode="playback"
        )
        self.assertEqual(len(self.host.captures), 1)
        # ...and the frames the user ASKED for are kept.
        self.assertTrue(os.path.isdir(os.path.join(self.out, "fake_png")))

    def test_encode_only_cleans_its_intermediate_frames(self):
        self.exporter.export(self.out, targets=["mp4"], range_mode="playback")
        self.assertFalse(os.path.exists(os.path.join(self.out, "fake_png_tmp")))

    def test_intermediate_frames_never_land_in_the_deliverable_folder(self):
        """An mp4 request staged its PNG capture as ``<output>/<name>_png_tmp``:
        a scratch folder inside the user's deliverable folder, synced and seen
        there for the length of the capture, and left behind by a killed run."""
        results = self.exporter.export(self.out, targets=["mp4"], range_mode="playback")
        self.assertTrue(results[0].ok)
        capture_dir = os.path.normcase(os.path.abspath(self.host.captures[0]["dir"]))
        self.assertFalse(
            capture_dir.startswith(os.path.normcase(os.path.abspath(self.out))),
            capture_dir,
        )
        self.assertFalse(os.path.exists(capture_dir), "the scratch capture was kept")
        self.assertEqual(os.listdir(self.out), ["fake.mp4"])

    def test_no_scratch_folder_without_an_intermediate_capture(self):
        """A plan whose encodes ride a requested png sequence, or that has no
        encode at all, captures nothing to scratch: no store folder is made."""
        real = TempArtifacts.scoped
        made = []

        def scoped(prefix, **kw):
            made.append(prefix)
            return real(prefix, **kw)

        with unittest.mock.patch.object(
            TempArtifacts, "scoped", side_effect=scoped
        ):
            self.exporter.export(
                self.out, targets=["mp4", "png_sequence"], range_mode="playback"
            )
            self.exporter.export(self.out, targets=["jpg_sequence"], range_mode="playback")
        self.assertNotIn("sequence_frames", made)

    def test_keep_frames_leaves_the_intermediate_capture(self):
        self.exporter.export(
            self.out, targets=["mp4"], range_mode="playback", keep_frames=True
        )
        self.assertTrue(os.path.isdir(os.path.join(self.out, "fake_png_tmp")))

    def test_a_capture_failure_isolates_only_the_targets_that_needed_it(self):
        self.host.fail_formats = {"jpg"}
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

        def half_then_fail(exporter, directory, prefix, *args):
            with open(os.path.join(directory, f"{prefix}.0001.png"), "wb") as handle:
                handle.write(b"frame")
            raise RuntimeError("interrupted")

        self.host.capture_frames = half_then_fail
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
        self.assertEqual(self.host.captures, [])
        self.assertEqual(self.host.warnings, ["Export target 'mp4' failed: no ffmpeg"])

    def test_an_unknown_target_is_refused_before_the_plan_starts(self):
        with self.assertRaises(ValueError):
            self.exporter.export(self.out, targets=["mp4", "bogus"])
        self.assertEqual(self.host.captures, [])

    def test_progress_runs_to_done(self):
        seen = []
        self.exporter.export(
            self.out,
            targets=["mp4", "png_sequence", "still", "avi"],
            range_mode="playback",
            progress_callback=lambda step, total, label: seen.append(
                (step, total, label)
            ),
        )
        self.assertEqual(
            [label for _, _, label in seen],
            ["Capturing png frames", "Encoding MP4 (H.264)",
             "Capturing PNG Still (Current Frame)", "Producing AVI", "Done"],
        )
        self.assertEqual(seen[-1][0], seen[-1][1])

    # -- the host seam --------------------------------------------------

    def test_a_host_kind_is_produced_through_the_seam(self):
        results = {
            r.target: r
            for r in self.Exporter(camera="cam2").export(
                self.out, name="r", targets=["render", "mp4"], range_mode="custom",
                start=2, end=3,
            )
        }
        self.assertTrue(results["render"].ok)
        self.assertTrue(os.path.isfile(results["render"].output))
        self.assertEqual(
            self.host.extras,
            [{"name": "r", "range": (2, 3), "camera": None, "overrides": {}}],
        )
        self.assertTrue(results["mp4"].ok)

    def test_a_kind_with_no_producer_fails_that_target_alone(self):
        unanswered = _bind(
            self.host,
            TARGETS={
                **SequenceExporter.TARGETS,
                "weird": ExportTarget("weird", "Weird", "weird", extension="wrd"),
            },
        )

        results = {
            r.target: r
            for r in unanswered().export(
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
        with self.assertRaises(NotImplementedError):
            SequenceExporter()._capture_movie(
                os.path.join(self.out, "x.avi"), "avi", "", 1, 2, None, None, {}
            )

    def test_a_host_path_is_read_through_the_host(self):
        """Blender's ``//`` is the open file's folder: the host maps it, the
        plan lands there."""
        self.host.resolve_path = lambda path: path.replace("//", self.out + "/", 1)
        results = self.exporter.export("//renders", name="rel", targets=["still"])
        self.assertEqual(
            os.path.normcase(os.path.abspath(results[0].output)),
            os.path.normcase(os.path.join(self.out, "renders", "rel.png")),
        )

    # -- the capture bookkeeping ----------------------------------------

    def test_a_capture_keeps_real_frame_numbers_and_clears_stale_ones(self):
        """Frames an earlier, wider run left under the prefix would pass the
        count check and -- read by ffmpeg's contiguous pattern -- be encoded
        past the requested end."""
        folder = os.path.join(self.out, "seq")
        os.makedirs(folder)
        for stale in (8, 12):
            with open(os.path.join(folder, f"rc.{stale:04d}.jpg"), "wb") as handle:
                handle.write(b"stale")
        capture = self.exporter.capture_sequence(
            folder, prefix="rc", start=7, end=9, image_format="jpg"
        )

        self.assertEqual(
            sorted(os.listdir(folder)), ["rc.0007.jpg", "rc.0008.jpg", "rc.0009.jpg"]
        )
        self.assertEqual((capture.start, capture.end, capture.fps), (7, 9, 25.0))
        self.assertTrue(capture.pattern.endswith("rc.%04d.jpg"))

    def test_a_capture_the_host_cut_short_raises(self):
        self.host.short_by = 1
        with self.assertRaisesRegex(RuntimeError, "2/3 frames"):
            self.exporter.capture_sequence(self.out, prefix="cut", start=1, end=3)

    def test_a_still_defaults_to_the_current_frame_and_the_default_camera(self):
        exporter = self.Exporter(camera="c2")
        path = exporter.capture_still(os.path.join(self.out, "sub", "s.png"))

        self.assertEqual(self._files(), ["sub/s.png"])
        self.assertEqual(
            self.host.stills, [{"path": path, "frame": 3, "camera": "c2"}]
        )
        self.assertEqual(self.host.notes, [f"Still frame captured: {path}"])

    def test_a_host_without_a_still_capture_moves_a_one_frame_capture_into_place(
        self,
    ):
        self.host.native_still = False
        path = self.exporter.capture_still(
            os.path.join(self.out, "one.jpg"), frame=6, image_format="jpg"
        )

        self.assertEqual(self._files(), ["one.jpg"])
        self.assertEqual(
            (self.host.captures[0]["start"], self.host.captures[0]["end"]), (6, 6)
        )
        self.assertTrue(os.path.isfile(path))

    def test_a_still_the_host_did_not_write_raises(self):
        self.host.capture_still = lambda *args: None
        with self.assertRaisesRegex(RuntimeError, "no file"):
            self.exporter.capture_still(os.path.join(self.out, "none.png"))

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
        encoder = type("_WithAudio", (SequenceEncoder,), {"HOST": _FakeHost(__file__)})

        path, offset = encoder()._resolve_audio(True, capture_start=10, fps=25.0)
        self.assertEqual(path, __file__)
        self.assertAlmostEqual(offset, 0.4)

    def test_a_missing_audio_source_is_silently_no_audio(self):
        self.assertEqual(SequenceEncoder()._resolve_audio(True, 0, 24.0), (None, 0.0))

    def test_an_encode_of_a_bare_pattern_uses_the_hosts_rate(self):
        wav = self._wav()
        out = self.exporter.encode_sequence(
            os.path.join(self.out, "x.%04d.png"),
            os.path.join(self.out, "enc", "x.mp4"),
            audio=wav,
            quality=0,
        )
        call = self.compress_calls[0]
        self.assertEqual(
            (call["frame_rate"], call["start_number"], call["audio_filepath"],
             call["audio_offset"], call["crf"]),
            (25.0, None, wav, 0.0, 40),
        )
        self.assertTrue(os.path.isfile(out))


if __name__ == "__main__":
    unittest.main()
