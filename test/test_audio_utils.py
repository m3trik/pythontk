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
        handler = logging.handlers = []

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


if __name__ == "__main__":
    unittest.main()
