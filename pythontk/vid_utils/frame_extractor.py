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
    """Extract frames from a video file at a configurable step interval.

    Also exposes quality-aware extraction:

    * :meth:`extract_frames_sharpest` — bucket frames by time window and
      save only the sharpest per bucket (variance-of-Laplacian).
    * :meth:`score_sharpness` — module helper for callers that want to
      score arbitrary images.
    """

    SUPPORTED_FORMATS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v")

    @staticmethod
    def score_sharpness(frame) -> float:
        """Variance-of-Laplacian sharpness score. Higher == sharper.

        Returns 0.0 when cv2 is unavailable. Accepts a BGR ndarray.
        """
        if not CV2_AVAILABLE:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

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

        Raises:
            ValueError: if ``step`` is less than 1.
        """
        if step < 1:
            raise ValueError("step must be >= 1")

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

    def extract_frames_sharpest(
        self,
        video_path: str,
        output_folder: str,
        window_sec: float = 1.0,
        quality: int = 95,
        prefix: str = "frame",
        max_frames: Optional[int] = None,
        min_sharpness: float = 0.0,
    ) -> List[str]:
        """Bucket frames by time window; save the sharpest per bucket.

        Tuned for handheld video — fixed-step extraction is wasteful when
        the camera is still and starves overlap when it moves quickly.
        Sharpest-of-window picks a useful frame from every part of the
        timeline regardless of pacing.

        Parameters:
            window_sec: Bucket size in seconds. 1.0 means "one frame per
                second of source video, but pick the sharpest of each
                second's worth of frames."
            min_sharpness: Reject windows whose best score is below this
                floor (0 = accept all). Use to skip blank-wall / sky
                segments.
        """
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available; cannot extract frames.")
            return []
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return []

        os.makedirs(output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        window_frames = max(1, int(round(window_sec * fps)))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            f"Sharpest-of-window: window={window_sec}s ({window_frames} frames), "
            f"source={total} frames @ {fps:.2f} fps"
        )

        saved: List[str] = []
        best_score = -1.0
        best_frame = None
        best_index = 0
        bucket_start = 0
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                score = FrameExtractor.score_sharpness(frame)
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()
                    best_index = idx
                idx += 1
                if idx - bucket_start >= window_frames:
                    if best_frame is not None and best_score >= min_sharpness:
                        out = os.path.join(
                            output_folder, f"{prefix}_{best_index:06d}.jpg"
                        )
                        if cv2.imwrite(out, best_frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, quality]):
                            saved.append(out)
                            if max_frames and len(saved) >= max_frames:
                                break
                    bucket_start = idx
                    best_score = -1.0
                    best_frame = None
            # flush the trailing partial bucket
            if (best_frame is not None and best_score >= min_sharpness
                    and (not max_frames or len(saved) < max_frames)):
                out = os.path.join(output_folder, f"{prefix}_{best_index:06d}.jpg")
                if cv2.imwrite(out, best_frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                    saved.append(out)
        finally:
            cap.release()

        logger.info(f"Sharpest-of-window kept {len(saved)} frames from {video_path}")
        return saved

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
