# !/usr/bin/python
# coding=utf-8
import os
import logging
import subprocess
import shutil
import re
from typing import Optional, Union

# from this package:
from pythontk.core_utils.help_mixin import HelpMixin

logger = logging.getLogger(__name__)


class VidUtils(HelpMixin):
    """ """

    # Standard frame rate mappings
    FRAME_RATES = {
        "game": 15.0,
        "film": 24.0,
        "pal": 25.0,
        "ntsc": 30.0,
        "show": 48.0,
        "palf": 50.0,
        "ntscf": 60.0,
        "23.976fps": 23.976,
        "29.97fps": 29.97,
        "47.952fps": 47.952,
        "59.94fps": 59.94,
        "44100fps": 44100.0,
        "48000fps": 48000.0,
    }

    @classmethod
    def get_frame_rate(cls, value: Union[str, float, int]) -> Union[float, str]:
        """Converts between frame rate names and values.

        If a string is provided (e.g. 'film', '24fps'), returns the float value (24.0).
        If a number is provided (e.g. 24.0), returns the standard name ('film') or formatted string ('24fps').

        Parameters:
            value: The frame rate name (str) or value (float/int).

        Returns:
            float | str: The converted value.

        Raises:
            ValueError: If a string can't be resolved to a frame rate —
                a silent default would be indistinguishable from real 24fps.
        """
        # Handle number -> name
        if isinstance(value, (int, float)):
            for name, rate in cls.FRAME_RATES.items():
                if abs(rate - value) < 0.001:
                    return name
            return f"{value:g}fps"

        # Handle string -> float
        if isinstance(value, str):
            if value in cls.FRAME_RATES:
                return cls.FRAME_RATES[value]
            if "fps" in value:
                try:
                    return float(value.replace("fps", ""))
                except ValueError:
                    pass

        raise ValueError(f"Unrecognized frame rate: {value!r}")

    @classmethod
    def resolve_ffmpeg(
        cls, required: bool = True, auto_install: bool = False
    ) -> Optional[str]:
        """Finds FFmpeg executable path in system path or managed installs.

        Parameters:
            required:      If True (default), raises when ffmpeg is
                           unavailable.
            auto_install:  If True, downloads and installs ffmpeg when
                           it cannot be found on the system PATH.

        Returns:
            str: Path to the FFmpeg executable, or None if not found
            and *required* is False.

        Raises:
            FileNotFoundError: If FFmpeg is not located and *required*
                is True.
        """
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path

        # Check managed installs from a previous session.
        from pythontk.core_utils.app_installer import (
            AppInstaller,
            FFMPEG_PLATFORMS,
        )

        managed = AppInstaller.get_path("ffmpeg", executable="ffmpeg", add_to_path=True)
        if managed:
            return managed

        if auto_install:
            try:
                return AppInstaller.ensure(
                    "ffmpeg",
                    platforms=FFMPEG_PLATFORMS,
                    executable="ffmpeg",
                )
            except Exception:
                pass  # Fall through to required/None logic below

        if required:
            raise FileNotFoundError(
                "FFmpeg is required but not found in the system path."
            )
        return None

    @classmethod
    def get_video_frame_rate(cls, filepath: str) -> float:
        """Extracts frame rate from a video file using FFmpeg.

        Parameters:
            filepath (str): Path to the video file.

        Returns:
            float: The frame rate of the video.

        Raises:
            RuntimeError: If frame rate cannot be determined.
        """
        ffmpeg_path = cls.resolve_ffmpeg()
        result = subprocess.run(
            [ffmpeg_path, "-i", filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        fps_pattern = re.compile(r"(?P<fps>\d+(?:\.\d+)?)\s*fps", re.IGNORECASE)
        tbr_pattern = re.compile(r"(?P<tbr>\d+(?:\.\d+)?)\s*tbr", re.IGNORECASE)

        for line in result.stderr.splitlines():
            fps_match = fps_pattern.search(line)
            if fps_match:
                return float(fps_match.group("fps"))

            tbr_match = tbr_pattern.search(line)
            if tbr_match:
                return float(tbr_match.group("tbr"))

        _lines = result.stderr.strip().splitlines()
        error_excerpt = _lines[-1] if _lines else ""
        raise RuntimeError(
            "Could not determine frame rate of the video. "
            f"FFmpeg last message: {error_excerpt}"
        )

    @classmethod
    def get_sequence_start_number(cls, input_filepath: str) -> Optional[int]:
        """Find the first frame number of a printf-style image sequence on disk.

        ffmpeg's image2 demuxer only auto-detects start numbers in the 0-4
        range; a sequence beginning at any other frame (e.g. a playblast of
        frames 101-150) needs an explicit ``-start_number``. This scans the
        pattern's directory for matching files and returns the lowest number.

        Parameters:
            input_filepath: A printf-style pattern such as ``shot.%04d.png``.

        Returns:
            int | None: The lowest frame number found, or None when the
            pattern contains no ``%`` token or no files match.
        """
        directory, basename = os.path.split(input_filepath)
        token = re.search(r"%0?\d*d", basename)
        if not token:
            return None
        regex = re.compile(
            re.escape(basename[: token.start()])
            + r"(\d+)"
            + re.escape(basename[token.end():])
            + r"$"
        )
        try:
            entries = os.listdir(directory or ".")
        except OSError:
            return None
        numbers = [int(m.group(1)) for f in entries for m in [regex.match(f)] if m]
        return min(numbers) if numbers else None

    @classmethod
    def compress_video(
        cls,
        input_filepath: str,
        output_filepath: str = None,
        frame_rate: Union[float, int] = None,
        delete_original: bool = False,
        start_number: Optional[int] = None,
        audio_filepath: Optional[str] = None,
        audio_offset: float = 0.0,
        **ffmpeg_options,
    ) -> Union[str, None]:
        """Compresses a video file or image sequence using FFmpeg.

        Parameters:
            input_filepath (str): Path of the video to compress, or a
                printf-style image-sequence pattern (e.g. ``shot.%04d.png``).
            output_filepath (str): Path for the compressed video, defaults to .mp4 version of input.
            frame_rate (float | int): Frame rate for output video. Defaults to original video frame rate.
            delete_original (bool): Deletes original file after compression if True.
            start_number (int): First frame number of an image-sequence input.
                Defaults to auto-detection from the files on disk (ffmpeg
                itself only detects sequences starting near 0).
            audio_filepath (str): Optional audio file to mux into the output
                (AAC-encoded, clamped to the shorter stream).
            audio_offset (float): Audio placement in seconds relative to the
                video start: positive delays the audio, negative skips into it.
            **ffmpeg_options: Additional FFmpeg output options like codec, crf, and preset.

        Returns:
            str | None: The path to the compressed video if successful, None otherwise.

        Raises:
            FileNotFoundError: If FFmpeg is not found.
        """
        if output_filepath is None:
            output_filepath = os.path.splitext(input_filepath)[0] + ".mp4"

        # ffmpeg -y would truncate the source before reading it. normcase:
        # Windows paths are case-insensitive, abspath alone won't catch
        # 'take1.MP4' vs 'take1.mp4'.
        if os.path.normcase(os.path.abspath(output_filepath)) == os.path.normcase(
            os.path.abspath(input_filepath)
        ):
            raise ValueError(
                f"Output path equals input path ('{input_filepath}'); "
                "pass a different output_filepath."
            )

        ffmpeg_path = cls.resolve_ffmpeg()

        if frame_rate is None:
            try:
                frame_rate = cls.get_video_frame_rate(input_filepath)
            except RuntimeError as err:
                logger.warning(f"{err} - falling back to 24 fps for compression.")
                frame_rate = 24

        ffmpeg_cmd = [ffmpeg_path, "-y"]

        is_sequence = "%" in os.path.basename(input_filepath)
        if is_sequence:
            # Input options must precede -i to apply to the sequence demuxer.
            if frame_rate is not None:
                ffmpeg_cmd.extend(["-framerate", str(frame_rate)])
            if start_number is None:
                start_number = cls.get_sequence_start_number(input_filepath)
            if start_number is not None:
                ffmpeg_cmd.extend(["-start_number", str(start_number)])

        ffmpeg_cmd.extend(["-i", input_filepath])

        if audio_filepath:
            if audio_offset > 0:
                ffmpeg_cmd.extend(["-itsoffset", str(audio_offset)])
            elif audio_offset < 0:
                ffmpeg_cmd.extend(["-ss", str(-audio_offset)])
            ffmpeg_cmd.extend(["-i", audio_filepath])

        ffmpeg_cmd.extend(
            [
                "-c:v",
                ffmpeg_options.get("codec", "libx264"),
                "-crf",
                str(ffmpeg_options.get("crf", 18)),
                "-preset",
                ffmpeg_options.get("preset", "slow"),
                "-pix_fmt",
                ffmpeg_options.get("pixel_format", "yuv420p"),
            ]
        )

        if audio_filepath:
            # Explicit maps: without them ffmpeg picks a single "best" input.
            # apad pads short audio with silence so -shortest is bounded by
            # the VIDEO — bare -shortest would truncate the video to a
            # shorter audio track.
            ffmpeg_cmd.extend(
                [
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:a", "aac",
                    "-af", "apad",
                    "-shortest",
                ]
            )

        ffmpeg_cmd.extend(["-r", str(frame_rate)])

        # Remaining options are output options — they must precede the
        # output path (trailing args would be parsed as another output).
        for option, value in ffmpeg_options.items():
            if option not in {"codec", "crf", "preset", "pixel_format"}:
                ffmpeg_cmd.extend([f"-{option}", str(value)])

        ffmpeg_cmd.append(output_filepath)

        logger.info(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")

        try:
            # Use Popen to stream output to avoid hanging UI without feedback
            # CREATE_NO_WINDOW hides the console window on Windows
            creation_flags = 0
            if os.name == "nt":
                creation_flags = subprocess.CREATE_NO_WINDOW

            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                creationflags=creation_flags,
            )

            # Stream output
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    logger.info(line.strip())

            if process.returncode != 0:
                logger.error(
                    f"FFmpeg compression failed for '{input_filepath}'. "
                    f"Exit code: {process.returncode}"
                )
                return None

            logger.info(f"Compressed video saved to: {output_filepath}")

            if (
                delete_original
                and os.path.abspath(input_filepath) != os.path.abspath(output_filepath)
                and os.path.exists(input_filepath)
            ):
                os.remove(input_filepath)
                logger.info(f"Original file {input_filepath} deleted.")

            return output_filepath

        except Exception as e:
            logger.error(f"Error executing FFmpeg: {e}")
            return None


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
