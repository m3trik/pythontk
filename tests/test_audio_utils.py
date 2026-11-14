#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk AudioUtils.

Tests cover:
- build_composite_wav (including the sampwidth bug fix)
- trim_silence
- resolve_playable_path
- build_audio_map / build_audio_map_from_files / build_audio_map_from_file_map

Run with:
    python -m pytest test_audio_utils.py -v
    python test_audio_utils.py
"""

import array
import hashlib
import logging
import os
import shutil
import tempfile
import unittest
import wave

from pythontk import AudioUtils

from conftest import BaseTestCase


def _write_wav(
    path,
    samples,
    sample_rate=44100,
    channels=1,
    sampwidth=2,
):
    """Write a 16-bit (or other width) PCM WAV file from an array of ints.

    Parameters:
        path: Output file path.
        samples: List/array of integer sample values.
        sample_rate: Samples per second.
        channels: Number of audio channels.
        sampwidth: Sample width in bytes (2 = 16-bit).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fmt = "h" if sampwidth == 2 else "i"
    arr = array.array(fmt, samples)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(arr.tobytes())


def _read_wav_samples(path):
    """Read a 16-bit WAV and return (params, array('h'))."""
    with wave.open(path, "rb") as wf:
        params = wf.getparams()
        raw = wf.readframes(params.nframes)
    samples = array.array("h")
    samples.frombytes(raw)
    return params, samples


class TestBuildCompositeWav(BaseTestCase):
    """Tests for AudioUtils.build_composite_wav."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="audio_test_")
        self.logger = logging.getLogger("test_audio")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _wav(self, name, samples, **kw):
        """Create a WAV in the temp dir and return its path."""
        path = os.path.join(self.tmpdir, name)
        _write_wav(path, samples, **kw)
        return path

    # -----------------------------------------------------------------

    def test_basic_composite(self):
        """Two clips placed at different frames mix correctly."""
        sr = 44100
        tone_a = [1000] * sr  # 1 second of 1000
        tone_b = [2000] * sr  # 1 second of 2000

        path_a = self._wav("a.wav", tone_a, sample_rate=sr)
        path_b = self._wav("b.wav", tone_b, sample_rate=sr)

        events = [(0, "a"), (24, "b")]  # 24 fps → b starts at 1s
        audio_map = {"a": path_a, "b": path_b}
        out = os.path.join(self.tmpdir, "comp.wav")

        result = AudioUtils.build_composite_wav(
            events=events,
            audio_map=audio_map,
            fps=24.0,
            output_path=out,
            logger=self.logger,
        )
        self.assertIsNotNone(result)
        self.assertTrue(os.path.isfile(result))

        params, samples = _read_wav_samples(result)
        self.assertEqual(params.sampwidth, 2)
        self.assertEqual(params.nchannels, 1)

        # First second should be tone_a (1000), second second should be tone_b (2000).
        # At frame 24 (1.0s), b starts. At samples 0..sr-1 only a plays.
        self.assertEqual(samples[0], 1000)
        # At the start of b (sample_pos = 1*sr), a and b overlap for remainder of a.
        # Since a is exactly 1s, at sample sr only b plays.
        self.assertEqual(samples[sr], 2000)

    def test_empty_events_returns_none(self):
        """build_composite_wav returns None for empty events."""
        out = os.path.join(self.tmpdir, "comp.wav")
        result = AudioUtils.build_composite_wav([], {}, 24.0, out)
        self.assertIsNone(result)

    def test_zero_fps_returns_none(self):
        """build_composite_wav returns None when fps <= 0."""
        path = self._wav("a.wav", [100] * 100)
        out = os.path.join(self.tmpdir, "comp.wav")
        result = AudioUtils.build_composite_wav([(0, "a")], {"a": path}, 0, out)
        self.assertIsNone(result)

    def test_missing_label_skipped(self):
        """Events with no matching audio_map entry are skipped."""
        path = self._wav("a.wav", [100] * 100)
        out = os.path.join(self.tmpdir, "comp.wav")
        result = AudioUtils.build_composite_wav(
            [(0, "a"), (10, "missing")],
            {"a": path},
            24.0,
            out,
            logger=self.logger,
        )
        self.assertIsNotNone(result)

    def test_first_file_non_16bit_skipped_correctly(self):
        """When the first WAV is 24-bit, it's skipped and the next 16-bit
        file becomes the reference.

        Bug: Previously the first file set sample_rate/channels/sampwidth
        before the 16-bit check, poisoning the reference values.
        Fixed: 2026-02-25
        """
        # 24-bit WAV (sampwidth=3) — use array('i') for storage
        bad_path = os.path.join(self.tmpdir, "bad.wav")
        os.makedirs(os.path.dirname(bad_path), exist_ok=True)
        buf = array.array("i", [0] * 100)
        raw_bytes = buf.tobytes()
        # Manually write 24-bit: take lower 3 bytes of each 4-byte int
        raw_24 = b"".join(raw_bytes[i : i + 3] for i in range(0, len(raw_bytes), 4))
        with wave.open(bad_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(3)
            wf.setframerate(44100)
            wf.writeframes(raw_24)

        # Good 16-bit WAV
        good_path = self._wav("good.wav", [500] * 100)

        out = os.path.join(self.tmpdir, "comp.wav")
        result = AudioUtils.build_composite_wav(
            events=[(0, "bad"), (0, "good")],
            audio_map={"bad": bad_path, "good": good_path},
            fps=24.0,
            output_path=out,
            logger=self.logger,
        )

        self.assertIsNotNone(result)
        params, samples = _read_wav_samples(result)
        # The output must be 16-bit, not poisoned by the 24-bit file
        self.assertEqual(params.sampwidth, 2)
        self.assertEqual(samples[0], 500)

    def test_non_first_24bit_file_skipped(self):
        """A 24-bit WAV after a valid 16-bit reference is skipped.

        Bug: Previously non-first files never checked sampwidth, silently
        producing corrupt output via array('h').frombytes on 3-byte data.
        Fixed: 2026-02-25
        """
        good_path = self._wav("good.wav", [500] * 100)

        bad_path = os.path.join(self.tmpdir, "bad.wav")
        buf = array.array("i", [0] * 100)
        raw_bytes = buf.tobytes()
        raw_24 = b"".join(raw_bytes[i : i + 3] for i in range(0, len(raw_bytes), 4))
        with wave.open(bad_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(3)
            wf.setframerate(44100)
            wf.writeframes(raw_24)

        out = os.path.join(self.tmpdir, "comp.wav")
        result = AudioUtils.build_composite_wav(
            events=[(0, "good"), (0, "bad")],
            audio_map={"good": good_path, "bad": bad_path},
            fps=24.0,
            output_path=out,
            logger=self.logger,
        )
        self.assertIsNotNone(result)
        params, _ = _read_wav_samples(result)
        self.assertEqual(params.sampwidth, 2)

    def test_mismatched_sample_rate_skipped(self):
        """Files with different sample rates are skipped."""
        path_a = self._wav("a.wav", [100] * 100, sample_rate=44100)
        path_b = self._wav("b.wav", [200] * 100, sample_rate=22050)

        out = os.path.join(self.tmpdir, "comp.wav")
        result = AudioUtils.build_composite_wav(
            events=[(0, "a"), (0, "b")],
            audio_map={"a": path_a, "b": path_b},
            fps=24.0,
            output_path=out,
            logger=self.logger,
        )
        self.assertIsNotNone(result)
        params, _ = _read_wav_samples(result)
        self.assertEqual(params.framerate, 44100)

    def test_mismatched_channels_skipped(self):
        """Files with different channel counts are skipped."""
        path_mono = self._wav("mono.wav", [100] * 100, channels=1)
        path_stereo = self._wav("stereo.wav", [200, 200] * 100, channels=2)

        out = os.path.join(self.tmpdir, "comp.wav")
        result = AudioUtils.build_composite_wav(
            events=[(0, "mono"), (0, "stereo")],
            audio_map={"mono": path_mono, "stereo": path_stereo},
            fps=24.0,
            output_path=out,
            logger=self.logger,
        )
        self.assertIsNotNone(result)
        params, _ = _read_wav_samples(result)
        self.assertEqual(params.nchannels, 1)

    def test_clipping_clamped(self):
        """Mixed values exceeding 16-bit range are clamped."""
        path_a = self._wav("a.wav", [30000] * 10)
        path_b = self._wav("b.wav", [30000] * 10)

        out = os.path.join(self.tmpdir, "comp.wav")
        AudioUtils.build_composite_wav(
            events=[(0, "a"), (0, "b")],
            audio_map={"a": path_a, "b": path_b},
            fps=24.0,
            output_path=out,
        )
        _, samples = _read_wav_samples(out)
        self.assertEqual(samples[0], 32767)

    def test_duration_log_fractional(self):
        """Composite log message reports fractional seconds.

        Bug: Integer division (//) truncated sub-second durations.
        Fixed: 2026-02-25
        """
        sr = 44100
        # 0.5 seconds of audio
        half_sec = [100] * (sr // 2)
        path = self._wav("half.wav", half_sec, sample_rate=sr)

        out = os.path.join(self.tmpdir, "comp.wav")

        class _Capture:
            msgs = []

            def info(self, msg):
                self.msgs.append(msg)

            def warning(self, msg):
                pass

        cap = _Capture()
        AudioUtils.build_composite_wav(
            events=[(0, "half")],
            audio_map={"half": path},
            fps=24.0,
            output_path=out,
            logger=cap,
        )
        self.assertTrue(any("0.5s" in m for m in cap.msgs), cap.msgs)

    def test_negative_frame_no_broadcast_error(self):
        """Negative frame values must not cause a numpy broadcast error.

        Bug: A negative ``sample_pos`` caused numpy negative-indexing,
        resulting in ``composite[pos:end]`` being an empty slice while
        ``arr[:end-pos]`` was non-empty → ValueError on ``+=``.
        Fixed: 2026-03-11
        """
        sr = 44100
        tone = [500] * sr  # 1 second
        path = self._wav("neg.wav", tone, sample_rate=sr)

        events = [(-24, "neg"), (24, "neg")]  # -1s and +1s at 24 fps
        audio_map = {"neg": path}
        out = os.path.join(self.tmpdir, "comp.wav")

        # Before the fix this raised ValueError
        result = AudioUtils.build_composite_wav(
            events=events,
            audio_map=audio_map,
            fps=24.0,
            output_path=out,
            logger=self.logger,
        )
        self.assertIsNotNone(result)
        self.assertTrue(os.path.isfile(result))

    def test_bare_filename_output_path(self):
        """A bare output filename (no directory) writes to the cwd.

        Bug: ``os.makedirs(os.path.dirname(output))`` received ``''`` for a
        bare filename and raised FileNotFoundError instead of writing the
        file into the current working directory.
        Fixed: 2026-07-18
        """
        path = self._wav("a.wav", [100] * 100)
        cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            result = AudioUtils.build_composite_wav(
                events=[(0, "a")],
                audio_map={"a": path},
                fps=24.0,
                output_path="comp_bare.wav",  # bare filename, no directory
                logger=self.logger,
            )
            self.assertIsNotNone(result)
            self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "comp_bare.wav")))
        finally:
            os.chdir(cwd)


class TestTrimSilence(BaseTestCase):
    """Tests for AudioUtils.trim_silence."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="audio_trim_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _wav(self, name, samples, **kw):
        path = os.path.join(self.tmpdir, name)
        _write_wav(path, samples, **kw)
        return path

    def test_trim_leading_trailing_silence(self):
        """Leading/trailing silent samples are removed."""
        silence = [0] * 100
        tone = [1000] * 50
        samples = silence + tone + silence
        path = self._wav("test.wav", samples)

        result = AudioUtils.trim_silence(path)
        # trim_silence normalises to forward slashes
        self.assertEqual(result, path.replace("\\", "/"))

        _, trimmed = _read_wav_samples(result)
        self.assertEqual(len(trimmed), 50)
        self.assertTrue(all(s == 1000 for s in trimmed))

    def test_trim_with_threshold(self):
        """Samples at or below threshold are treated as silence."""
        low = [5] * 20
        tone = [100] * 30
        samples = low + tone + low
        path = self._wav("test.wav", samples)

        AudioUtils.trim_silence(path, threshold=10)
        _, trimmed = _read_wav_samples(path)
        self.assertEqual(len(trimmed), 30)
        self.assertTrue(all(s == 100 for s in trimmed))

    def test_trim_to_separate_output(self):
        """Trim writes to a different output file."""
        samples = [0] * 10 + [500] * 20 + [0] * 10
        src = self._wav("src.wav", samples)
        dst = os.path.join(self.tmpdir, "dst.wav")

        result = AudioUtils.trim_silence(src, output_path=dst)
        self.assertEqual(result, dst.replace("\\", "/"))
        self.assertTrue(os.path.isfile(dst))

        _, trimmed = _read_wav_samples(dst)
        self.assertEqual(len(trimmed), 20)

    def test_trim_all_silence(self):
        """Entirely silent file produces empty WAV."""
        path = self._wav("silent.wav", [0] * 100)

        AudioUtils.trim_silence(path)
        _, trimmed = _read_wav_samples(path)
        self.assertEqual(len(trimmed), 0)

    def test_trim_no_silence(self):
        """File with no silence is unchanged."""
        tone = [1000] * 50
        path = self._wav("tone.wav", tone)

        AudioUtils.trim_silence(path)
        _, trimmed = _read_wav_samples(path)
        self.assertEqual(len(trimmed), 50)
        self.assertTrue(all(s == 1000 for s in trimmed))

    def test_trim_stereo(self):
        """Trim works correctly on stereo (2-channel) files."""
        silence = [0, 0] * 10  # 10 silent frames
        tone = [500, -500] * 20  # 20 audible frames
        samples = silence + tone + silence
        path = self._wav("stereo.wav", samples, channels=2)

        AudioUtils.trim_silence(path)
        params, trimmed = _read_wav_samples(path)
        self.assertEqual(params.nchannels, 2)
        # 20 frames × 2 channels = 40 samples
        self.assertEqual(len(trimmed), 40)

    def test_trim_rejects_non_16bit(self):
        """trim_silence raises ValueError for non-16-bit WAV."""
        path = os.path.join(self.tmpdir, "bad.wav")
        buf = array.array("i", [0] * 100)
        raw_bytes = buf.tobytes()
        raw_24 = b"".join(raw_bytes[i : i + 3] for i in range(0, len(raw_bytes), 4))
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(3)
            wf.setframerate(44100)
            wf.writeframes(raw_24)

        with self.assertRaises(ValueError):
            AudioUtils.trim_silence(path)

    def test_trim_preserves_sample_rate(self):
        """Output preserves original sample rate and channels."""
        samples = [0] * 10 + [1000] * 30 + [0] * 10
        path = self._wav("test.wav", samples, sample_rate=22050)

        AudioUtils.trim_silence(path)
        params, _ = _read_wav_samples(path)
        self.assertEqual(params.framerate, 22050)


class TestResolvePlayablePath(BaseTestCase):
    """Tests for AudioUtils.resolve_playable_path."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="audio_resolve_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_wav_returned_as_is(self):
        """A .wav file is returned without conversion."""
        path = os.path.join(self.tmpdir, "test.wav")
        _write_wav(path, [100] * 10)

        result = AudioUtils.resolve_playable_path(path)
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith(".wav"))

    def test_nonexistent_returns_none(self):
        """Non-existent file returns None (no exception)."""
        result = AudioUtils.resolve_playable_path(
            os.path.join(self.tmpdir, "ghost.wav")
        )
        self.assertIsNone(result)

    def test_backslash_normalised(self):
        """Backslashes in path are normalised to forward slashes."""
        path = os.path.join(self.tmpdir, "test.wav")
        _write_wav(path, [100] * 10)

        result = AudioUtils.resolve_playable_path(path.replace("/", "\\"))
        self.assertIsNotNone(result)
        self.assertNotIn("\\", result)


class TestPlayableCache(BaseTestCase):
    """Where a transcode is cached, and that a failed one never poisons it.

    The cache was ``<source dir>/_audio_cache/``: a folder written into the
    user's own audio folder. A transcode a saved scene references now goes to
    its project's ``cache/audio`` (``Workspace.cache_dir``), anything else to
    the per-user cache (``UserConfig.cache_dir``). And ffmpeg wrote straight to
    the cache name, so a failed run left a partial ``<stem>_<md5>.wav`` that the
    next call returned as the conversion.
    """

    def setUp(self):
        from unittest import mock

        from pythontk.core_utils.user_config import UserConfig

        self.tmpdir = tempfile.mkdtemp(prefix="audio_cache_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.user_cache = os.path.join(self.tmpdir, "user_cache")
        for patcher in (
            mock.patch.dict(
                os.environ, {UserConfig.CACHE_ROOT_ENV_VAR: self.user_cache}
            ),
            mock.patch.object(AudioUtils, "resolve_ffmpeg", return_value="ffmpeg"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.calls = []

    def _source(self, *parts):
        path = os.path.join(self.tmpdir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"ID3 not really an mp3")
        return path

    def _ffmpeg(self, ok=True):
        """A stand-in ffmpeg writing its last argument: a whole WAV, or (failed)
        the header-only partial a killed run leaves."""
        import subprocess
        from unittest import mock

        def run(cmd, **kwargs):
            self.calls.append(list(cmd))
            out = cmd[-1]
            if ok:
                _write_wav(out, [100] * 10)
            else:
                with open(out, "wb") as fh:
                    fh.write(b"RIFF\xff\xff\xff\xffWAVEfmt ")
            return subprocess.CompletedProcess(cmd, 0 if ok else 1, "", "boom")

        return mock.patch(
            "pythontk.audio_utils._audio_utils.subprocess.run", side_effect=run
        )

    @staticmethod
    def _norm(path):
        return os.path.normcase(os.path.normpath(path))

    def test_a_source_in_a_project_caches_in_its_cache_audio_folder(self):
        from pythontk.file_utils.workspace import Workspace

        project = Workspace.create(os.path.join(self.tmpdir, "proj"), create_dirs=False)
        src = self._source("proj", "sound", "vo.mp3")
        folder = os.path.join(project.root, "cache", "audio")
        self.assertEqual(self._norm(AudioUtils.cache_dir_for(src)), self._norm(folder))
        with self._ffmpeg():
            out = AudioUtils.ensure_playable_path(src)
        self.assertEqual(self._norm(os.path.dirname(out)), self._norm(folder))
        self.assertEqual(os.listdir(os.path.dirname(src)), ["vo.mp3"])

    def test_a_source_in_no_project_caches_in_the_user_cache(self):
        src = self._source("loose", "vo.mp3")
        folder = os.path.join(self.user_cache, "audio")
        self.assertEqual(self._norm(AudioUtils.cache_dir_for(src)), self._norm(folder))
        with self._ffmpeg():
            out = AudioUtils.resolve_playable_path(src)
        self.assertEqual(self._norm(os.path.dirname(out)), self._norm(folder))
        self.assertEqual(os.listdir(os.path.dirname(src)), ["vo.mp3"])

    def test_a_failed_conversion_leaves_nothing_the_next_call_returns(self):
        src = self._source("loose", "vo.mp3")
        with self._ffmpeg(ok=False), self.assertRaises(RuntimeError):
            AudioUtils.ensure_playable_path(src)
        folder = AudioUtils.cache_dir_for(src)
        self.assertEqual(os.listdir(folder), [], "the partial was left behind")
        with self._ffmpeg():
            out = AudioUtils.ensure_playable_path(src)
        self.assertEqual(len(self.calls), 2, "the partial was served as the cache")
        _read_wav_samples(out)

    def test_a_partial_left_in_the_cache_is_converted_again(self):
        src = self._source("loose", "vo.mp3")
        with self._ffmpeg():
            out = AudioUtils.ensure_playable_path(src)
        with open(out, "wb") as fh:  # an interrupted run of an older version
            fh.write(b"RIFF\xff\xff\xff\xffWAVEfmt ")
        with self._ffmpeg():
            again = AudioUtils.ensure_playable_path(src)
        self.assertEqual(again, out)
        self.assertEqual(len(self.calls), 2)
        _read_wav_samples(again)

    def test_a_complete_cached_wav_is_reused(self):
        src = self._source("loose", "vo.mp3")
        with self._ffmpeg():
            first = AudioUtils.ensure_playable_path(src)
            second = AudioUtils.ensure_playable_path(src)
        self.assertEqual(first, second)
        self.assertEqual(len(self.calls), 1)

    def test_a_cached_transcode_needs_no_ffmpeg(self):
        """The cache hit is checked before ffmpeg is looked up: a machine
        without ffmpeg still plays a transcode a saved scene references."""
        from unittest import mock

        src = self._source("loose", "vo.mp3")
        with self._ffmpeg():
            first = AudioUtils.ensure_playable_path(src)
        with mock.patch.object(
            AudioUtils, "resolve_ffmpeg", side_effect=FileNotFoundError("no ffmpeg")
        ):
            self.assertEqual(AudioUtils.ensure_playable_path(src), first)


class TestBuildAudioMapFromFiles(BaseTestCase):
    """Tests for AudioUtils.build_audio_map_from_files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="audio_map_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _wav(self, name, samples=None):
        samples = samples or [100] * 10
        path = os.path.join(self.tmpdir, name)
        _write_wav(path, samples)
        return path

    def test_basic_map(self):
        """File stems are lowered and mapped."""
        a = self._wav("Footstep.wav")
        b = self._wav("Jump.wav")

        result = AudioUtils.build_audio_map_from_files([a, b])
        self.assertIn("footstep", result)
        self.assertIn("jump", result)

    def test_duplicate_stem_first_wins(self):
        """First occurrence of a stem wins; duplicate emits warning."""
        a = self._wav("clip.wav", [100] * 10)
        subdir = os.path.join(self.tmpdir, "sub")
        os.makedirs(subdir)
        b = os.path.join(subdir, "clip.wav")
        _write_wav(b, [200] * 10)

        result = AudioUtils.build_audio_map_from_files([a, b])
        self.assertEqual(len(result), 1)
        # First file wins
        self.assertEqual(result["clip"].replace("\\", "/"), a.replace("\\", "/"))

    def test_nonexistent_skipped(self):
        """Non-existent files are silently skipped."""
        a = self._wav("real.wav")
        fake = os.path.join(self.tmpdir, "fake.wav")

        result = AudioUtils.build_audio_map_from_files([a, fake])
        self.assertIn("real", result)
        self.assertNotIn("fake", result)


class TestBuildAudioMapFromFileMap(BaseTestCase):
    """Tests for AudioUtils.build_audio_map_from_file_map."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="audio_fmap_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _wav(self, name, samples=None):
        samples = samples or [100] * 10
        path = os.path.join(self.tmpdir, name)
        _write_wav(path, samples)
        return path

    def test_stems_preserved(self):
        """Stems from the file_map keys are used, not re-derived from filename."""
        path = self._wav("actual_file.wav")
        fmap = {"MyLabel": path}

        result = AudioUtils.build_audio_map_from_file_map(fmap)
        self.assertIn("mylabel", result)
        self.assertNotIn("actual_file", result)

    def test_missing_file_skipped(self):
        """Missing file path in map is skipped."""
        fmap = {"gone": os.path.join(self.tmpdir, "nope.wav")}
        result = AudioUtils.build_audio_map_from_file_map(fmap)
        self.assertEqual(len(result), 0)


class TestBuildAudioMap(BaseTestCase):
    """Tests for AudioUtils.build_audio_map (directory scan)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="audio_scan_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_recursive_scan(self):
        """Finds WAV files in sub-directories."""
        sub = os.path.join(self.tmpdir, "sub")
        os.makedirs(sub)
        _write_wav(os.path.join(self.tmpdir, "root.wav"), [100] * 10)
        _write_wav(os.path.join(sub, "nested.wav"), [200] * 10)

        result = AudioUtils.build_audio_map(self.tmpdir)
        self.assertIn("root", result)
        self.assertIn("nested", result)

    def test_non_audio_ignored(self):
        """Non-audio files are not included."""
        _write_wav(os.path.join(self.tmpdir, "audio.wav"), [100] * 10)
        txt = os.path.join(self.tmpdir, "notes.txt")
        with open(txt, "w") as f:
            f.write("hello")

        result = AudioUtils.build_audio_map(self.tmpdir)
        self.assertIn("audio", result)
        self.assertNotIn("notes", result)

    def test_empty_dir_returns_empty(self):
        """Empty directory returns empty map."""
        result = AudioUtils.build_audio_map(self.tmpdir)
        self.assertEqual(result, {})

    def test_custom_extensions_filter(self):
        """Only specified extensions are included."""
        _write_wav(os.path.join(self.tmpdir, "a.wav"), [100] * 10)

        # With a filter that excludes .wav
        result = AudioUtils.build_audio_map(self.tmpdir, extensions={".mp3"})
        self.assertEqual(result, {})


class TestMemoDuration(BaseTestCase):
    """``AudioUtils.memo_duration``: a host's clip-length reader runs once per
    file as it is on disk -- a table resolves its ranges several times a
    redraw, and each read opened a WAV on a synced drive -- and again once the
    file is replaced (2026-10-13)."""

    def setUp(self):
        import pythontk as ptk

        self.tmp = ptk.TempArtifacts("ptk_memo_duration", policy="scoped")
        self.addCleanup(self.tmp.cleanup)
        self.path = self.tmp.path(".wav")
        self.reads = []

    def probe(self, path, fps):
        self.reads.append(path)
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / wf.getframerate() * fps, path

    def test_a_file_is_read_once_until_it_changes(self):
        _write_wav(self.path, [0] * 4410, sample_rate=4410)  # 1 s
        self.assertEqual(AudioUtils.memo_duration(self.path, 24, self.probe)[0], 24)
        self.assertEqual(AudioUtils.memo_duration(self.path, 24, self.probe)[0], 24)
        self.assertEqual(len(self.reads), 1, "read once")
        self.assertEqual(AudioUtils.memo_duration(self.path, 30, self.probe)[0], 30)
        self.assertEqual(len(self.reads), 2, "another frame rate reads again")
        _write_wav(self.path, [0] * 8820, sample_rate=4410)  # replaced: 2 s
        self.assertEqual(AudioUtils.memo_duration(self.path, 24, self.probe)[0], 48)
        self.assertEqual(len(self.reads), 3, "a replaced file reads again")

    def test_no_reading_and_no_file_are_never_kept(self):
        empty = lambda path, fps: (self.reads.append(path) or 0.0, path)  # noqa: E731
        _write_wav(self.path, [0] * 10, sample_rate=4410)
        for _ in range(2):
            AudioUtils.memo_duration(self.path, 24, empty)
            AudioUtils.memo_duration(self.path + ".gone", 24, empty)
        self.assertEqual(len(self.reads), 4)


#: The rate the cut's cases are stated at.
CUT_FPS = 30.0


def _cues(sounds, takes, window=None):
    """``[(clip, sound, time, offset, stop)]`` in frames at :data:`CUT_FPS` --
    what :meth:`AudioUtils.clip_cues` cuts, read back on the frame grid so a
    case states it as the timeline does -- and the dropped list."""
    cues, dropped = AudioUtils.clip_cues(sounds, takes, CUT_FPS, window=window)
    return [
        (
            c["clip"],
            c["sound"],
            round(c["time"] * CUT_FPS, 4),
            round(c["offset"] * CUT_FPS, 4),
            round(c["stop"] * CUT_FPS, 4),
        )
        for c in cues
    ], dropped


def _take(name, start, end):
    return {"name": name, "start": start, "end": end}



class TestClipCues(BaseTestCase):
    """``clip_cues``: the ``audio`` record's cues, each sound cut per clip in
    seconds from the clip's start -- the one rule both DCCs' audio producers
    ship by. Every case is stated in frames at 30 fps (time, offset into the
    sound, stop), the cues themselves being seconds. Added: 2026-11-05"""

    CONTIGUOUS = [_take("Intro", 1, 40), _take("Fight", 40, 80)]

    def test_a_sound_inside_one_shot_is_one_cue_counted_from_its_clip(self):
        cues, dropped = _cues(
            [{"sound": "a.wav", "start": 10, "stop": 20}], self.CONTIGUOUS
        )
        self.assertEqual(cues, [("Intro", "a.wav", 9, 0, 19)])
        self.assertEqual(dropped, [])

    def test_the_cues_are_seconds(self):
        cues, _ = AudioUtils.clip_cues(
            [{"sound": "a.wav", "start": 16, "stop": 46}], [_take("Intro", 1, 80)], 30.0
        )
        self.assertEqual(cues, [
            {"clip": "Intro", "sound": "a.wav", "time": 0.5, "offset": 0.0, "stop": 1.5}
        ])

    def test_a_sound_across_a_cut_is_two_cues_each_stopping_at_the_cut(self):
        """Contiguous shots share frame 40: the sound stops at Intro's end
        and Fight plays it on from 10 frames into the sound."""
        cues, _ = _cues([{"sound": "a.wav", "start": 30, "stop": 50}], self.CONTIGUOUS)
        self.assertEqual(
            cues,
            [("Intro", "a.wav", 29, 0, 39), ("Fight", "a.wav", 0, 10, 10)],
        )

    def test_a_sound_across_a_gap_plays_on_in_the_next_shot_at_its_offset(self):
        cues, dropped = _cues(
            [{"sound": "a.wav", "start": 30, "stop": 70}],
            [_take("Intro", 1, 40), _take("Outro", 60, 80)],
        )
        self.assertEqual(
            cues,
            [("Intro", "a.wav", 29, 0, 39), ("Outro", "a.wav", 0, 30, 10)],
        )
        self.assertEqual(dropped, [])

    def test_a_sound_starting_before_its_shot_starts_at_its_offset(self):
        cues, dropped = _cues(
            [{"sound": "a.wav", "start": 35, "stop": 55}], [_take("Fight", 40, 80)]
        )
        self.assertEqual(cues, [("Fight", "a.wav", 0, 5, 15)])
        self.assertEqual(dropped, [])

    def test_a_sound_longer_than_its_clip_stops_at_the_clips_end(self):
        cues, _ = _cues([{"sound": "a.wav", "start": 70, "stop": 200}], self.CONTIGUOUS)
        self.assertEqual(cues, [("Fight", "a.wav", 30, 0, 40)])

    def test_a_sound_with_no_known_end_stops_at_its_clips_end(self):
        """A stop on every cue: the producer keys an end on every track, and
        a sound whose end it cannot measure ends with its clip."""
        cues, _ = _cues([{"sound": "a.wav", "start": 70, "stop": None}], self.CONTIGUOUS)
        self.assertEqual(cues, [("Fight", "a.wav", 30, 0, 40)])

    def test_the_authors_head_trim_adds_to_the_offset(self):
        """A sound its author trimmed at the head starts into its file."""
        cues, _ = _cues(
            [{"sound": "a.wav", "start": 30, "stop": 50, "offset": 0.5}],
            self.CONTIGUOUS,
        )
        self.assertEqual(
            cues,
            [("Intro", "a.wav", 29, 15, 39), ("Fight", "a.wav", 0, 25, 10)],
        )

    def test_a_sound_in_no_clip_is_dropped_and_named(self):
        cues, dropped = _cues(
            [
                {"sound": "late.wav", "start": 130, "stop": 140},
                {"sound": "edge.wav", "start": 80, "stop": 90},
            ],
            self.CONTIGUOUS,
        )
        self.assertEqual(cues, [])
        self.assertEqual(dropped, ["late.wav@130", "edge.wav@80"])

    def test_the_cues_run_clip_by_clip_in_take_order_then_by_time(self):
        cues, _ = _cues(
            [
                {"sound": "b.wav", "start": 60, "stop": 62},
                {"sound": "a.wav", "start": 50, "stop": 52},
                {"sound": "c.wav", "start": 5, "stop": 6},
            ],
            self.CONTIGUOUS,
        )
        self.assertEqual(
            [(c[0], c[1]) for c in cues],
            [("Intro", "c.wav"), ("Fight", "a.wav"), ("Fight", "b.wav")],
        )

    def test_no_sound_cuts_no_cue(self):
        self.assertEqual(AudioUtils.clip_cues([], self.CONTIGUOUS, CUT_FPS), ([], []))

    def test_a_shotless_take_counts_from_the_window_it_declares(self):
        """The S2 case (2026-11-05): the file's one take holds keys 0-100
        under a 20-80 range; counted from 20, a cue at 50 played 20 frames
        early. The window is the file's: measured off it once written (its
        declared window fitted to the keys, ``FbxMedia.fit_take_windows``)."""
        sound = [{"sound": "a.wav", "start": 50, "stop": 60}]
        measured, _ = _cues(sound, [], window=(0, 100))
        self.assertEqual(measured, [("", "a.wav", 50, 0, 60)])
        declared, _ = _cues(sound, [], window=(20, 80))
        self.assertEqual(declared, [("", "a.wav", 30, 0, 40)])

    def test_a_shotless_file_holding_nothing_has_no_clip_to_cue(self):
        cues, dropped = _cues([{"sound": "a.wav", "start": 50, "stop": 60}], [], window=None)
        self.assertEqual(cues, [])
        self.assertEqual(dropped, ["a.wav@50"])

    def test_a_take_missing_its_name_or_range_cuts_nothing(self):
        """Defaulted, it would be a phantom take swallowing frame-0 sounds."""
        cues, dropped = _cues(
            [{"sound": "a.wav", "start": 0, "stop": 5}],
            [{"name": "X", "start": None, "end": 10}, {"start": 0, "end": 10}],
            window=(0, 10),
        )
        self.assertEqual(cues, [("", "a.wav", 0, 0, 5)])
        self.assertEqual(dropped, [])


class TestMediaDigests(BaseTestCase):
    """``media_digests``: the ``audio`` record's ``media`` -- the content
    digest of each sound file a delivery embeds, by the cue's ``sound``, the
    file name it is delivered under -- what a reader compares a same-named
    sound against before it stands that sound in for the delivery's own. Two
    models each embedding a different ``door.wav``: Unity bound the first
    model's clip for the second (2026-10-19; matched by content since
    2026-10-31). Given the bytes inside, never a path: the FBX's dependency
    pass digests what it embedded."""

    @staticmethod
    def cues(*sounds):
        return [
            {"clip": "", "sound": s, "time": 0.0, "offset": 0.0, "stop": 1.0}
            for s in sounds
        ]

    def test_each_audio_file_is_the_sha256_of_its_bytes_by_the_cues_sound(self):
        files = {"door.wav": b"door one", "Lamp Hum.mp3": b"hum"}
        self.assertEqual(
            AudioUtils.media_digests(self.cues("door.wav", "lamp hum.MP3"), files),
            {
                "door.wav": hashlib.sha256(b"door one").hexdigest(),
                "lamp hum.MP3": hashlib.sha256(b"hum").hexdigest(),
            },
        )

    def test_a_sound_with_no_file_inside_is_left_out_and_a_name_is_digested_once(self):
        files = {"door.wav": b"door one"}
        media = AudioUtils.media_digests(
            self.cues("door.wav", "ghost.wav", "door.wav"), files
        )
        self.assertEqual(media, {"door.wav": hashlib.sha256(b"door one").hexdigest()})

    def test_only_audio_files_answer_and_a_name_two_share_names_none(self):
        """A texture is no sound, and a file name only one file holds is the
        sound's: another extension is another sound; two files whose names
        differ in case alone leave the reader nothing to tell apart."""
        self.assertEqual(
            AudioUtils.media_digests(
                self.cues("door.wav", "bell.wav", "door.png"),
                {
                    "door.png": b"wood",
                    "door.wav": b"door",
                    "bell.wav": b"a",
                    "Bell.WAV": b"b",
                    "door.ogg": b"c",
                },
            ),
            {"door.wav": hashlib.sha256(b"door").hexdigest()},
        )

    def test_no_cues_digest_nothing(self):
        self.assertEqual(AudioUtils.media_digests([], {"door.wav": b"door"}), {})


if __name__ == "__main__":
    unittest.main()
