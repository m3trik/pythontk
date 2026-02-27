# !/usr/bin/python
# coding=utf-8
import hashlib
import array as _array
import os
import shutil
import subprocess
import wave as _wave
from typing import Dict, List, Optional, Set

from pythontk.core_utils.help_mixin import HelpMixin


class AudioUtils(HelpMixin):
    """Utility helpers for portable audio-file preparation.

    Provides ffmpeg-backed conversion and WAV-compositing helpers.
    """

    PLAYABLE_EXTENSIONS = {".wav", ".aif", ".aiff"}
    SOURCE_EXTENSIONS = PLAYABLE_EXTENSIONS | {".mp3", ".ogg", ".m4a", ".flac"}

    @staticmethod
    def resolve_ffmpeg(required: bool = True) -> Optional[str]:
        """Resolve ffmpeg executable from PATH.

        Parameters:
            required: If True, raises when ffmpeg is unavailable.

        Returns:
            Path to ffmpeg executable, or None when not found and
            ``required`` is False.
        """
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path

        if required:
            raise FileNotFoundError(
                "FFmpeg is required but was not found in the system PATH."
            )
        return None

    @classmethod
    def is_playable_extension(cls, file_path: str) -> bool:
        """Return True if extension is already timeline-playable."""
        return os.path.splitext(file_path)[1].lower() in cls.PLAYABLE_EXTENSIONS

    @classmethod
    def is_supported_source_extension(cls, file_path: str) -> bool:
        """Return True if extension is accepted as conversion source."""
        return os.path.splitext(file_path)[1].lower() in cls.SOURCE_EXTENSIONS

    @classmethod
    def ensure_playable_path(
        cls,
        audio_path: str,
        cache_dir: Optional[str] = None,
    ) -> str:
        """Return a playable audio path, converting with ffmpeg if required.

        Parameters:
            audio_path: Source audio file path.
            cache_dir: Optional cache directory for converted WAV output.

        Returns:
            Playable file path (original or converted WAV).

        Raises:
            ValueError: Unsupported file extension.
            FileNotFoundError: Missing source file or ffmpeg.
            RuntimeError: ffmpeg conversion failed.
        """
        source = os.path.normpath(audio_path).replace("\\", "/")
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Audio file does not exist: {source}")

        ext = os.path.splitext(source)[1].lower()
        if ext in cls.PLAYABLE_EXTENSIONS:
            return source

        if ext not in cls.SOURCE_EXTENSIONS:
            raise ValueError(f"Unsupported audio extension '{ext}': {source}")

        ffmpeg_cmd = cls.resolve_ffmpeg(required=True)

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(source), "_audio_cache")
        cache_dir = os.path.normpath(cache_dir).replace("\\", "/")
        os.makedirs(cache_dir, exist_ok=True)

        stat = os.stat(source)
        token = f"{source}|{stat.st_mtime_ns}|{stat.st_size}"
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()[:10]
        stem = os.path.splitext(os.path.basename(source))[0]
        output = os.path.join(cache_dir, f"{stem}_{digest}.wav").replace("\\", "/")

        if os.path.isfile(output):
            return output

        creation_flags = 0
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NO_WINDOW

        cmd = [
            ffmpeg_cmd,
            "-y",
            "-i",
            source,
            "-vn",
            "-acodec",
            "pcm_s16le",
            output,
        ]

        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            creationflags=creation_flags,
        )

        if completed.returncode != 0 or not os.path.isfile(output):
            stderr = (completed.stderr or "").strip().splitlines()
            tail = stderr[-1] if stderr else "unknown ffmpeg error"
            raise RuntimeError(f"ffmpeg conversion failed for '{source}': {tail}")

        return output

    @classmethod
    def build_composite_wav(
        cls,
        events: list,
        audio_map: dict,
        fps: float,
        output_path: str,
        logger=None,
    ) -> Optional[str]:
        """Mix source WAV clips into one composite WAV.

        Parameters:
            events: ``[(frame, label), ...]`` list.
            audio_map: Mapping of lowercase label -> playable file path.
            fps: Scene frames per second.
            output_path: Output WAV path.
            logger: Optional logger with ``warning/info`` methods.

        Returns:
            Output path on success, else None.
        """
        if not events or fps <= 0:
            return None

        clips = []
        sample_rate = None
        channels = None
        sampwidth = None

        for frame, label in events:
            path = audio_map.get(str(label).lower())
            if not path:
                continue

            try:
                with _wave.open(path, "rb") as wf:
                    sr = wf.getframerate()
                    ch = wf.getnchannels()
                    sw = wf.getsampwidth()

                    if sw != 2:
                        if logger:
                            logger.warning(
                                f"Skipping '{path}': only 16-bit PCM WAV "
                                f"is supported (got {sw * 8}-bit)"
                            )
                        continue

                    if sample_rate is None:
                        sample_rate = sr
                        channels = ch
                        sampwidth = sw
                    elif sr != sample_rate:
                        if logger:
                            logger.warning(
                                f"Skipping '{path}': sample rate "
                                f"{sr} != {sample_rate}"
                            )
                        continue
                    elif ch != channels:
                        if logger:
                            logger.warning(
                                f"Skipping '{path}': channels {ch} != {channels}"
                            )
                        continue

                    raw = wf.readframes(wf.getnframes())
            except Exception as exc:
                if logger:
                    logger.warning(f"Cannot read '{path}': {exc}")
                continue

            samples = _array.array("h")
            samples.frombytes(raw)

            time_sec = float(frame) / fps
            sample_pos = int(time_sec * sample_rate) * channels
            clips.append((sample_pos, samples))

        if not clips or sample_rate is None:
            return None

        total_samples = max(pos + len(s) for pos, s in clips)
        composite = _array.array("h", bytes(total_samples * 2))

        for pos, samples in clips:
            for i, val in enumerate(samples):
                idx = pos + i
                if idx < len(composite):
                    mixed = composite[idx] + val
                    composite[idx] = max(-32768, min(32767, mixed))

        output = os.path.normpath(output_path).replace("\\", "/")
        os.makedirs(os.path.dirname(output), exist_ok=True)

        with _wave.open(output, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(composite.tobytes())

        if logger:
            logger.info(
                f"Built composite: {total_samples // channels} samples, "
                f"{total_samples / (channels * sample_rate):.1f}s"
            )

        return output

    # ------------------------------------------------------------------
    # Audio-map helpers
    # ------------------------------------------------------------------

    @classmethod
    def resolve_playable_path(
        cls,
        audio_path: str,
        cache_dir: Optional[str] = None,
        logger=None,
    ) -> Optional[str]:
        """Return a playable path, converting to WAV when required.

        Thin convenience wrapper around ``ensure_playable_path`` that
        returns ``None`` instead of raising on failure.

        Parameters:
            audio_path: Source audio file path.
            cache_dir: Cache directory for converted files.  Defaults to
                ``_audio_cache`` next to the source file.
            logger: Optional logger with ``warning``/``debug`` methods.

        Returns:
            Resolved playable path, or ``None`` on error.
        """
        source = audio_path.replace("\\", "/")
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(source), "_audio_cache").replace(
                "\\", "/"
            )

        try:
            resolved = cls.ensure_playable_path(source, cache_dir=cache_dir)
            if resolved != source and logger:
                logger.debug(f"Converted '{source}' -> '{resolved}'")
            return resolved
        except Exception as exc:
            if logger:
                logger.warning(f"Cannot import '{source}': {exc}")
            return None

    @classmethod
    def build_audio_map(
        cls,
        search_dir: str,
        extensions: Optional[Set[str]] = None,
        cache_dir: Optional[str] = None,
        logger=None,
    ) -> Dict[str, str]:
        """Recursively scan a directory for audio files.

        Parameters:
            search_dir: Root directory to scan.
            extensions: Accepted source extensions.  Defaults to
                ``SOURCE_EXTENSIONS``.
            cache_dir: Cache directory for converted files.  When
                ``None`` a ``_audio_cache`` folder is created next to
                each source file.
            logger: Optional logger with ``warning``/``debug`` methods.

        Returns:
            Dict mapping lowercase filename stem to playable path.
            First file found wins; collisions emit a warning.
        """
        exts = extensions or cls.SOURCE_EXTENSIONS
        audio_map: Dict[str, str] = {}

        for root, _, files in os.walk(search_dir):
            for file in files:
                name, ext = os.path.splitext(file)
                if ext.lower() not in exts:
                    continue
                key = name.lower()
                full_path = os.path.join(root, file).replace("\\", "/")
                playable = cls.resolve_playable_path(
                    full_path, cache_dir=cache_dir, logger=logger
                )
                if not playable:
                    continue
                if key in audio_map:
                    if logger:
                        logger.warning(
                            f"Duplicate audio stem '{name}': keeping "
                            f"'{audio_map[key]}', ignoring '{playable}'"
                        )
                else:
                    audio_map[key] = playable

        return audio_map

    @classmethod
    def build_audio_map_from_file_map(
        cls,
        file_map: Dict[str, str],
        cache_dir: Optional[str] = None,
        logger=None,
    ) -> Dict[str, str]:
        """Build an audio map from a ``{stem: path}`` dict.

        The provided *stems* are used as dictionary keys (lowered)
        instead of re-extracting them from file paths, guaranteeing
        that keys match event labels derived from the same stems.

        Parameters:
            file_map: ``{stem: original_path}`` mapping.
            cache_dir: Cache directory for converted files.
            logger: Optional logger.

        Returns:
            Dict mapping lowercase stem -> resolved playable path.
        """
        audio_map: Dict[str, str] = {}
        for stem, path in file_map.items():
            full = path.replace("\\", "/")
            playable = cls.resolve_playable_path(
                full, cache_dir=cache_dir, logger=logger
            )
            if not playable:
                continue
            key = stem.lower()
            if key not in audio_map:
                audio_map[key] = playable
        return audio_map

    @classmethod
    def build_audio_map_from_files(
        cls,
        audio_files: List[str],
        cache_dir: Optional[str] = None,
        logger=None,
    ) -> Dict[str, str]:
        """Build an audio map from an explicit list of file paths.

        Parameters:
            audio_files: Absolute paths to audio files.
            cache_dir: Cache directory for converted files.
            logger: Optional logger.

        Returns:
            Dict mapping lowercase filename stem to playable path.
        """
        audio_map: Dict[str, str] = {}
        for path in audio_files:
            full = path.replace("\\", "/")
            playable = cls.resolve_playable_path(
                full, cache_dir=cache_dir, logger=logger
            )
            if not playable:
                continue
            stem = os.path.splitext(os.path.basename(full))[0].lower()
            if stem in audio_map:
                if logger:
                    logger.warning(
                        f"Duplicate audio stem '{stem}': keeping "
                        f"'{audio_map[stem]}', ignoring '{playable}'"
                    )
            else:
                audio_map[stem] = playable
        return audio_map

    # ------------------------------------------------------------------
    # WAV trimming
    # ------------------------------------------------------------------

    @classmethod
    def trim_silence(
        cls,
        wav_path: str,
        output_path: Optional[str] = None,
        threshold: int = 8,
    ) -> str:
        """Trim leading and trailing silence from a 16-bit PCM WAV file.

        Silence is defined as any sample whose absolute value is at or
        below *threshold* (default ``8``, on a 16-bit scale of 0–32767).

        Parameters:
            wav_path: Input WAV file path.
            output_path: Destination path.  Defaults to overwriting
                the input file in-place.
            threshold: Absolute sample value at or below which audio is
                considered silent.

        Returns:
            The output path.

        Raises:
            ValueError: If the file is not 16-bit PCM WAV.
        """
        wav_path = os.path.normpath(wav_path).replace("\\", "/")
        if output_path is None:
            output_path = wav_path
        else:
            output_path = os.path.normpath(output_path).replace("\\", "/")

        with _wave.open(wav_path, "rb") as wf:
            params = wf.getparams()
            if params.sampwidth != 2:
                raise ValueError(
                    f"Only 16-bit PCM WAV is supported (got {params.sampwidth * 8}-bit)"
                )
            raw = wf.readframes(params.nframes)

        samples = _array.array("h")
        samples.frombytes(raw)

        channels = params.nchannels

        # Find first and last non-silent *frame* (group of `channels` samples)
        num_frames = len(samples) // channels
        first_frame = 0
        for f in range(num_frames):
            offset = f * channels
            if any(abs(samples[offset + c]) > threshold for c in range(channels)):
                first_frame = f
                break
        else:
            # Entire file is silent — write an empty WAV
            first_frame = 0
            num_frames = 0

        last_frame = num_frames
        if num_frames > 0:
            for f in range(num_frames - 1, first_frame - 1, -1):
                offset = f * channels
                if any(abs(samples[offset + c]) > threshold for c in range(channels)):
                    last_frame = f + 1
                    break

        trimmed = samples[first_frame * channels : last_frame * channels]

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with _wave.open(output_path, "wb") as wf:
            wf.setparams(params)
            wf.writeframes(trimmed.tobytes())

        return output_path
