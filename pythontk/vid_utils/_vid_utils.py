# !/usr/bin/python
# coding=utf-8
import os
import subprocess
import shutil
import re

# from this package:
from pythontk.core_utils.help_mixin import HelpMixin


class VidUtils(HelpMixin):
    """ """

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
        frame_rate: int = None,
        delete_original: bool = False,
        **ffmpeg_options,
    ) -> None:
        """Compresses a video file using FFmpeg.

        Parameters:
            input_filepath (str): Path of the video to compress.
            output_filepath (str): Path for the compressed video, defaults to .mp4 version of input.
            frame_rate (int): Frame rate for output video. Defaults to original video frame rate.
            delete_original (bool): Deletes original file after compression if True.
            **ffmpeg_options: Additional FFmpeg command options like codec, crf, and preset.

        Raises:
            FileNotFoundError: If FFmpeg is not found.
            subprocess.CalledProcessError: If FFmpeg encounters an error.
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

        ffmpeg_cmd = [
            ffmpeg_path,
            "-y",
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

        for option, value in ffmpeg_options.items():
            if option not in {"codec", "crf", "preset", "pixel_format"}:
                ffmpeg_cmd.extend([f"-{option}", str(value)])

        try:
            print("Running FFmpeg command:", " ".join(ffmpeg_cmd))
            subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print(f"Compressed video saved to: {output_filepath}")

            if (
                delete_original
                and os.path.abspath(input_filepath) != os.path.abspath(output_filepath)
                and os.path.exists(input_filepath)
            ):
                os.remove(input_filepath)
                print(f"Original file {input_filepath} deleted.")

            return output_filepath

        except subprocess.CalledProcessError as e:
            print(f"Error during FFmpeg compression: {e}")
            print(f"FFmpeg output:\n{e.stderr}")
            return None


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
