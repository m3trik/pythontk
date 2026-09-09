# !/usr/bin/python
# coding=utf-8
"""Record a clip playing in the preview page to a movie file.

The viewer's transport can already play one shot, or every shot laid back onto
the timeline they were cut from. This is what turns that playback into a file:
the page steps the clip frame by frame, posts each rendered frame here, and the
frames are encoded by :class:`pythontk.SequenceEncoder` -- the same encode
Maya's playblast exporter runs, with the same target vocabulary and the same
even-dimension and CRF rules.

**A playblast, not a screen recording.** Frames are *stepped*, never sampled off
a wall clock: the page poses the clip at frame N, renders, and hands the result
over before asking for N+1. A slow device, a paused tab and a fast one all
produce the same file, at exactly the clip's authoring frame rate -- which is
the property that makes the result comparable with a viewport playblast of the
same shot rather than merely similar to it.

Frames arrive one at a time over HTTP because that is what a browser can send:
a canvas readback is a Blob, and holding a few hundred of them in page memory
to send at the end is how a headset tab dies. Each is written straight to a
scratch sequence, so the page's memory ceiling is one frame regardless of clip
length.

The recording is addressed by a *token* rather than by name: two pages can be
looking at the same server (a desktop tab and a headset), and a frame from one
must never land in the other's sequence.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.temp_artifacts import TempArtifacts
from pythontk.str_utils._str_utils import StrUtils
from pythontk.vid_utils.sequence_exporter import CaptureResult, SequenceEncoder


@dataclass
class _Recording:
    """One in-flight recording: where its frames go and what they describe."""

    token: str
    name: str
    directory: str
    image_format: str
    fps: float
    start_frame: int
    expected: int
    started: float = field(default_factory=time.time)
    received: Dict[int, str] = field(default_factory=dict)

    @property
    def frame_numbers(self) -> List[int]:
        return sorted(self.received)

    @property
    def duration(self) -> float:
        return self.expected / self.fps if self.fps else 0.0


class PreviewPlayblast(SequenceEncoder):
    """Frames pushed in by the viewer page, encoded by the shared core.

    A :class:`~pythontk.SequenceEncoder` rather than a
    :class:`~pythontk.SequenceExporter`: the capture loop lives in the browser,
    so there is no viewport for this object to read and nothing here could
    honour an exporter's ``capture_sequence`` contract. What it wants from the
    shared core is the half that is genuinely shared -- the ffmpeg encode, the
    quality mapping, the frame bookkeeping and the cleanup.

    Thread-safe: the server answers each frame POST on its own handler thread.

    Parameters:
        quality: 0-100, mapped onto the H.264 CRF scale.
        max_frames: Refuse a recording longer than this. A guard against a page
            that asks for a 40-minute sequence by mistake, not a policy -- raise
            it for a genuinely long shot.
        max_frame_bytes: Refuse a single frame larger than this.
    """

    #: Recording -> ``TARGETS`` key. Only encoded movie outputs make sense from
    #: a page: the page already has the frames, so a "PNG sequence" round trip
    #: would post them here only to hand them back.
    DEFAULT_TARGET: str = "mp4"

    #: Extensions a page may claim its frames are in. PNG is what a canvas
    #: readback encodes losslessly; JPEG is the escape hatch for a long clip
    #: over a slow (headset Wi-Fi) link.
    IMAGE_FORMATS: Dict[str, str] = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
    }

    def __init__(
        self,
        quality: int = 100,
        max_frames: int = 7200,
        max_frame_bytes: int = 64 * 1024 * 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(quality=quality, **kwargs)
        self.max_frames = int(max_frames)
        self.max_frame_bytes = int(max_frame_bytes)
        self._lock = threading.Lock()
        self._recordings: Dict[str, _Recording] = {}
        # "session": the frames outlive the request that wrote them and there
        # is no completion signal to delete on except finish/cancel, either of
        # which may never arrive (a tab closed mid-record). Interpreter exit is
        # the backstop, and the store's own age sweep collects a DCC that was
        # killed outright.
        self._temp = TempArtifacts("webxr_playblast", policy="session")

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------
    def begin(
        self,
        name: str,
        fps: float,
        start_frame: int = 1,
        frames: int = 0,
        content_type: str = "image/png",
    ) -> Dict[str, Any]:
        """Open a recording and return ``{"token", "name", "frames"}``.

        Parameters:
            name: Clip name, used for the output basename. Sanitized -- it
                comes off a page and reaches a filesystem.
            fps: The clip's authoring frame rate; the movie's rate.
            start_frame: Authoring frame the first captured frame IS, in the
                numbers the DCC's timeline and the picker's shot ranges quote.
                Carried for :meth:`finish` to report; the frames on disk are
                numbered from zero regardless (see :meth:`add_frame`). May be
                negative -- a scene with pre-roll declares shots there.
            frames: How many frames the page intends to send. Checked against
                :attr:`max_frames` up front, so a mistake costs one request
                instead of a full capture.
            content_type: MIME type of the frames to follow.

        Raises:
            ValueError: An unusable frame count, rate or content type.
        """
        image_format = self.IMAGE_FORMATS.get(str(content_type).split(";", 1)[0])
        if image_format is None:
            raise ValueError(
                f"Unsupported frame type {content_type!r}; expected one of "
                f"{sorted(self.IMAGE_FORMATS)}."
            )
        frames = int(frames)
        if frames < 1:
            raise ValueError(f"A recording needs at least one frame; got {frames}.")
        if frames > self.max_frames:
            raise ValueError(
                f"Refusing a {frames}-frame recording; the ceiling is "
                f"{self.max_frames} frames (raise max_frames to allow it)."
            )
        fps = float(fps)
        if not fps > 0:
            raise ValueError(f"A recording needs a positive frame rate; got {fps}.")

        # ``to_legal_name`` and not a bespoke regex: this is the same rule every
        # other filename this toolkit derives from user text goes through, and a
        # shot named "shot 01 / take 2" must not be able to write outside the
        # scratch directory.
        safe = StrUtils.to_legal_name(str(name).strip()) or "clip"
        token = uuid.uuid4().hex
        directory = self._temp.dir_path(name=f"rec_{token}")
        recording = _Recording(
            token=token,
            name=safe,
            directory=FileUtils.format_path(directory),
            image_format=image_format,
            fps=fps,
            start_frame=int(start_frame),
            expected=frames,
        )
        with self._lock:
            self._recordings[token] = recording
        self.logger.info(
            "Recording %r opened: %s frames @ %g fps from frame %s (%s).",
            safe,
            frames,
            fps,
            recording.start_frame,
            image_format,
        )
        return {"token": token, "name": safe, "frames": frames}

    def add_frame(self, token: str, index: int, data: bytes) -> Dict[str, Any]:
        """Store one rendered frame; returns ``{"received", "expected"}``.

        Parameters:
            token: The recording, from :meth:`begin`.
            index: 0-based position in the recording, which is also what the
                frame is called on disk -- see the numbering note below.
            data: The encoded image bytes.

        Raises:
            KeyError: No such recording (finished, cancelled, or never opened).
            ValueError: An out-of-range index or an oversized/empty frame.

        The scratch sequence is numbered from ZERO, not from the clip's
        authoring frame. The frames are deleted the moment the movie exists, so
        nothing carries their numbering anywhere -- while numbering them the way
        the timeline does breaks on a scene with pre-roll, where the authoring
        frame is negative: ``shot.-010.png`` matches no printf pattern, and
        ffmpeg's ``-start_number`` will not take a negative either, so the run
        would fail at the encode having already paid for the whole capture. The
        authoring range is carried on the recording and reported by
        :meth:`finish`, which is the only place it was ever read.
        """
        recording = self._get(token)
        index = int(index)
        if not 0 <= index < recording.expected:
            raise ValueError(
                f"Frame index {index} is outside the {recording.expected}-frame "
                f"recording {recording.name!r}."
            )
        if not data:
            raise ValueError(f"Frame {index} of {recording.name!r} arrived empty.")
        if len(data) > self.max_frame_bytes:
            raise ValueError(
                f"Frame {index} is {len(data)} bytes; the per-frame ceiling is "
                f"{self.max_frame_bytes}."
            )
        path = os.path.join(
            recording.directory,
            f"{recording.name}.{index:0{self.frame_padding}d}.{recording.image_format}",
        )
        # Written whole then moved into place: a GET of the scratch directory is
        # not possible, but an interrupted POST would otherwise leave a
        # half-written frame that _collect_frames counts and ffmpeg chokes on.
        staging = f"{path}.part"
        with open(staging, "wb") as handle:
            handle.write(data)
        os.replace(staging, path)
        with self._lock:
            recording.received[index] = FileUtils.format_path(path)
            received = len(recording.received)
        return {"received": received, "expected": recording.expected}

    def finish(
        self,
        token: str,
        output_dir: str,
        target: Optional[str] = None,
        stem: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Encode the recording and drop its scratch frames.

        Parameters:
            token: The recording, from :meth:`begin`.
            output_dir: Directory the movie is written to; created if missing.
            target: A :attr:`SequenceExporter.TARGETS` encode key
                (default :attr:`DEFAULT_TARGET`).
            stem: Output basename without extension. Defaults to the clip
                name. Sanitized like one: this reaches ``os.path.join``, and a
                caller that built it from anything a page said must not be able
                to place the movie outside *output_dir*.

        Returns:
            ``{"output", "frames", "fps", "start_frame", "duration"}``.

        Raises:
            KeyError: No such recording.
            RuntimeError: Frames are missing, or ffmpeg produced nothing. The
                scratch frames are removed either way -- a failed encode is not
                worth a gigabyte of PNGs nobody will look at again.
        """
        recording = self._get(token)
        try:
            capture = self._capture_of(recording)
            spec = self._target(target or self.DEFAULT_TARGET)
            safe_stem = StrUtils.to_legal_name(str(stem).strip()) if stem else ""
            output = os.path.join(
                output_dir,
                f"{safe_stem or recording.name}.{spec.extension}",
            )
            encoded = self.encode_sequence(
                capture, output, **dict(spec.encoder_options)
            )
        finally:
            self._discard(recording)
        self.logger.info(
            "Recording %r encoded: %s frames -> %s (%.2fs).",
            recording.name,
            len(capture.frames),
            encoded,
            recording.duration,
        )
        # ``start_frame`` is the CLIP's, off the recording rather than off the
        # capture: the scratch is numbered from zero (see :meth:`add_frame`),
        # and what a reviewer wants back is the frame the shot starts on.
        return {
            "output": encoded,
            "frames": len(capture.frames),
            "fps": capture.fps,
            "start_frame": recording.start_frame,
            "end_frame": recording.start_frame + recording.expected - 1,
            "duration": recording.duration,
        }

    def cancel(self, token: str) -> bool:
        """Drop a recording and its frames; whether there was one to drop.

        Never raises: this is what an unload beacon and an aborted capture
        call, and neither has anywhere to report a failure to.
        """
        try:
            recording = self._get(token)
        except KeyError:
            return False
        self._discard(recording)
        self.logger.debug("Recording %r cancelled.", recording.name)
        return True

    def clip_name(self, token: str) -> str:
        """The sanitized clip name a recording was opened under.

        Public because the caller that names the OUTPUT is the server (which
        alone knows the deliverable the clip belongs to), and it should not have
        to reach into a recording to ask.

        Raises:
            KeyError: No such recording.
        """
        return self._get(token).name

    def active(self) -> List[Dict[str, Any]]:
        """One entry per in-flight recording — for diagnostics and tests."""
        with self._lock:
            recordings = list(self._recordings.values())
        return [
            {
                "token": r.token,
                "name": r.name,
                "received": len(r.received),
                "expected": r.expected,
                "age": time.time() - r.started,
            }
            for r in recordings
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get(self, token: str) -> _Recording:
        with self._lock:
            recording = self._recordings.get(str(token))
        if recording is None:
            raise KeyError(f"No such recording: {token!r}.")
        return recording

    def _target(self, name: str):
        """The encode target *name*, refusing anything that is not an encode."""
        # Imported through the exporter rather than duplicated: the page picks
        # from the same registry the Playblast Export panel builds its picker
        # from, so "MP4 (H.264)" means one thing in this toolkit.
        from pythontk.vid_utils.sequence_exporter import SequenceExporter

        spec = SequenceExporter.TARGETS.get(name)
        if spec is None:
            raise ValueError(
                f"Unknown target {name!r}; available: "
                f"{sorted(SequenceExporter.TARGETS)}."
            )
        if spec.kind != "encode":
            raise ValueError(
                f"Target {name!r} is a {spec.kind} output; a page recording can "
                "only be encoded to a movie."
            )
        return spec

    def _capture_of(self, recording: _Recording) -> CaptureResult:
        """The recording as a :class:`CaptureResult`, or raise saying what is missing.

        Contiguity is the requirement, not merely the count: ffmpeg reads a
        printf pattern straight through and simply STOPS at the first gap, so a
        recording missing frame 12 of 300 would otherwise encode silently as an
        11-frame movie.
        """
        # Snapshot under the lock: a frame POST landing while this iterates
        # would otherwise resize the dict mid-read. The page does not send one
        # after asking to finish -- but nothing here can make that a guarantee.
        with self._lock:
            received = dict(recording.received)
        numbers = sorted(received)
        expected = list(range(recording.expected))
        if numbers != expected:
            missing = sorted(set(expected) - set(numbers))
            raise RuntimeError(
                f"Recording {recording.name!r} is incomplete: "
                f"{len(numbers)}/{recording.expected} frames, missing "
                f"{missing[:8]}{'...' if len(missing) > 8 else ''}."
            )
        return CaptureResult(
            directory=recording.directory,
            prefix=recording.name,
            image_format=recording.image_format,
            start=0,
            end=recording.expected - 1,
            padding=self.frame_padding,
            frames=[received[n] for n in numbers],
            fps=recording.fps,
        )

    def _discard(self, recording: _Recording) -> None:
        """Forget a recording and remove its whole scratch directory."""
        with self._lock:
            self._recordings.pop(recording.token, None)
        # The directory is this recording's alone (named for its token), so the
        # tree goes rather than the frames one by one -- which also takes any
        # ``.part`` file an interrupted POST left behind.
        try:
            shutil.rmtree(recording.directory, ignore_errors=True)
        except OSError as error:  # pragma: no cover - rmtree swallows its own
            self.logger.warning(
                "Could not remove recording scratch %s: %s",
                recording.directory,
                error,
            )

    @staticmethod
    def resolve_output_dir(source: Optional[Path], fallback: Path) -> Path:
        """Where a recording of *source* should land.

        Beside the published file when that file is still on disk -- the case
        that matters, because it means the user has a real deliverable (an
        exporter's GLB, or one they picked with External GLB) and the movie
        belongs next to it. A scene push has no such file: its GLB is the
        bridge's own scratch and is released the moment it is published, so
        there is nothing to sit beside and the movie goes to *fallback* (the
        serve root), from which the page downloads it.
        """
        if source is not None and source.is_file():
            return source.parent
        return fallback
