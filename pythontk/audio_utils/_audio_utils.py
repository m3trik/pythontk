# !/usr/bin/python
# coding=utf-8
import hashlib
import array as _array
import os
import shutil
import subprocess
import wave as _wave
from typing import Optional

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
                    if sample_rate is None:
                        sample_rate = wf.getframerate()
                        channels = wf.getnchannels()
                        sampwidth = wf.getsampwidth()
                        if sampwidth != 2:
                            if logger:
                                logger.warning(
                                    f"Skipping '{path}': only 16-bit PCM WAV is supported"
                                )
                            continue
                    elif wf.getframerate() != sample_rate:
                        if logger:
                            logger.warning(
                                f"Skipping '{path}': sample rate "
                                f"{wf.getframerate()} != {sample_rate}"
                            )
                        continue
                    elif wf.getnchannels() != channels:
                        if logger:
                            logger.warning(
                                f"Skipping '{path}': channels {wf.getnchannels()} != {channels}"
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
                f"{total_samples // (channels * sample_rate):.1f}s"
            )

        return output
