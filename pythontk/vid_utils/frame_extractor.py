# !/usr/bin/python
# coding=utf-8
"""Extract still frames from a video file via OpenCV.

OpenCV (``cv2``) is an optional dependency; this module imports it inside
a guard so pythontk's zero-required-deps surface is preserved. Calls to
:meth:`FrameExtractor.extract_frames` short-circuit with a logged error
when ``cv2`` is unavailable.
"""
import logging
import os
from typing import List, Optional

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from a video file at a configurable step interval."""

    SUPPORTED_FORMATS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v")

    def extract_frames(
        self,
        video_path: str,
        output_folder: str,
        step: int = 5,
        quality: int = 95,
        prefix: str = "frame",
        max_frames: Optional[int] = None,
    ) -> List[str]:
        """Save every ``step``-th frame from ``video_path`` to ``output_folder``.

        Returns the list of saved frame paths (empty when ``cv2`` is
        unavailable or the input can't be opened).
        """
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available; cannot extract frames.")
            return []

        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return []

        if os.path.splitext(video_path)[1].lower() not in self.SUPPORTED_FORMATS:
            logger.warning(f"Video format may not be supported: {video_path}")

        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        logger.info(
            f"Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration"
        )

        saved_frames: List[str] = []
        count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if count % step == 0:
                    frame_filename = f"{prefix}_{count:06d}.jpg"
                    frame_path = os.path.join(output_folder, frame_filename)
                    if cv2.imwrite(
                        frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
                    ):
                        saved_frames.append(frame_path)
                        if max_frames and len(saved_frames) >= max_frames:
                            logger.info(f"Reached maximum frame limit: {max_frames}")
                            break
                    else:
                        logger.warning(f"Failed to save frame: {frame_filename}")

                count += 1

                if count % (step * 100) == 0 and total_frames > 0:
                    progress = (count / total_frames) * 100
                    logger.info(
                        f"Progress: {progress:.1f}% ({count}/{total_frames} frames processed)"
                    )
        except Exception as e:
            logger.error(f"Error during frame extraction: {e}")
        finally:
            cap.release()

        logger.info(f"Extracted {len(saved_frames)} frames from {video_path}")
        return saved_frames

    def get_video_info(self, video_path: str) -> dict:
        """Return metadata for ``video_path`` (filename, frame count, fps,
        duration, dimensions, size).

        Returns an empty dict when ``cv2`` is unavailable or the file
        can't be opened.
        """
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available; cannot read video info.")
            return {}

        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return {}

        try:
            info = {
                "filename": os.path.basename(video_path),
                "path": video_path,
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            info["duration"] = (
                info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
            )
            info["size_mb"] = os.path.getsize(video_path) / (1024 * 1024)
            return info
        finally:
            cap.release()


def extract_frames(video_path: str, output_folder: str, step: int = 5) -> List[str]:
    """Convenience wrapper around :meth:`FrameExtractor.extract_frames`."""
    return FrameExtractor().extract_frames(video_path, output_folder, step)
