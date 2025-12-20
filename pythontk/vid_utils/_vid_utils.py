# !/usr/bin/python
# coding=utf-8
import os
import subprocess
import shutil
import re
from typing import Union

# from this package:
from pythontk.core_utils.help_mixin import HelpMixin


class VidUtils(HelpMixin):
    """ """

    # Standard frame rate mappings (Maya compatible)
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
        """
        # Handle number -> name
        if isinstance(value, (int, float)):
            for name, rate in cls.FRAME_RATES.items():
                if abs(rate - value) < 0.001:
                    return name
            return f"{value:g}fps"

        # Handle string -> float
        if isinstance(value, str):
            if not value:
                return 24.0
            if value in cls.FRAME_RATES:
                return cls.FRAME_RATES[value]
            if "fps" in value:
                try:
                    return float(value.replace("fps", ""))
                except ValueError:
                    pass

        return 24.0

    @staticmethod
    def resolve_ffmpeg() -> str:
        """Finds FFmpeg executable path in system path or Maya scripts.

        Returns:
            str: Path to the FFmpeg executable.

        Raises:
            FileNotFoundError: If FFmpeg is not located in either system path or Maya scripts path.
        """
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path

        maya_script_paths = os.getenv("MAYA_SCRIPT_PATH", "").split(";")
        for path in maya_script_paths:
            ffmpeg_in_bin = os.path.join(
                path, "ffmpeg", "bin", "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            )
            if os.path.isfile(ffmpeg_in_bin):
                return ffmpeg_in_bin
            ffmpeg_direct_bin = os.path.join(
                path, "bin", "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            )
            if os.path.isfile(ffmpeg_direct_bin):
                return ffmpeg_direct_bin

        raise FileNotFoundError(
            "FFmpeg is required but not found in the system path or Maya scripts path."
        )

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

        error_excerpt = result.stderr.strip().splitlines()[-1] if result.stderr else ""
        raise RuntimeError(
            "Could not determine frame rate of the video. "
            f"FFmpeg last message: {error_excerpt}"
        )

    @classmethod
    def compress_video(
        cls,
        input_filepath: str,
        output_filepath: str = None,
        frame_rate: Union[float, int] = None,
        delete_original: bool = False,
        **ffmpeg_options,
    ) -> Union[str, None]:
        """Compresses a video file using FFmpeg.

        Parameters:
            input_filepath (str): Path of the video to compress.
            output_filepath (str): Path for the compressed video, defaults to .mp4 version of input.
            frame_rate (float | int): Frame rate for output video. Defaults to original video frame rate.
            delete_original (bool): Deletes original file after compression if True.
            **ffmpeg_options: Additional FFmpeg command options like codec, crf, and preset.

        Returns:
            str | None: The path to the compressed video if successful, None otherwise.

        Raises:
            FileNotFoundError: If FFmpeg is not found.
        """
        ffmpeg_path = cls.resolve_ffmpeg()
        if output_filepath is None:
            output_filepath = input_filepath.replace(".avi", ".mp4")

        if frame_rate is None:
            try:
                frame_rate = cls.get_video_frame_rate(input_filepath)
            except RuntimeError as err:
                print(
                    f"[VidUtils] Warning: {err} - falling back to 24 fps for compression."
                )
                frame_rate = 24

        ffmpeg_cmd = [ffmpeg_path, "-y"]

        # If input is a sequence, specify the framerate to ensure correct duration
        if frame_rate is not None and "%" in input_filepath:
            ffmpeg_cmd.extend(["-framerate", str(frame_rate)])

        ffmpeg_cmd.extend(
            [
                "-i",
                input_filepath,
                "-c:v",
                ffmpeg_options.get("codec", "libx264"),
                "-crf",
                str(ffmpeg_options.get("crf", 18)),
                "-preset",
                ffmpeg_options.get("preset", "slow"),
                "-pix_fmt",
                ffmpeg_options.get("pixel_format", "yuv420p"),
                "-r",
                str(frame_rate),
                output_filepath,
            ]
        )

        for option, value in ffmpeg_options.items():
            if option not in {"codec", "crf", "preset", "pixel_format"}:
                ffmpeg_cmd.extend([f"-{option}", str(value)])

        print(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")

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
                    print(line.strip())

            if process.returncode != 0:
                print(
                    f"Error during FFmpeg compression. Exit code: {process.returncode}"
                )
                return None

            print(f"Compressed video saved to: {output_filepath}")

            if (
                delete_original
                and os.path.abspath(input_filepath) != os.path.abspath(output_filepath)
                and os.path.exists(input_filepath)
            ):
                os.remove(input_filepath)
                print(f"Original file {input_filepath} deleted.")

            return output_filepath

        except Exception as e:
            print(f"Error executing FFmpeg: {e}")
            return None


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
