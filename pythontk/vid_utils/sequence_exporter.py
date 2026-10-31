# !/usr/bin/python
# coding=utf-8
"""Image-sequence capture planning and encoding, with no idea what drew the frames.

The half of a "playblast" that is not a viewport: a registry of output targets,
one shared numbered image sequence, and an ffmpeg encode derived from it. What
produces the pixels -- a DCC viewport, a browser canvas, a renderer -- is the
host adapter, :attr:`SequenceEncoder.HOST` (a :class:`SequenceHost`; mayatk's
and blendertk's ``PlayblastExporter`` each bind theirs), so the same plan, the
same target vocabulary and the same encode serve every producer:

======================  ========================================================
``frame_range``         the timeline's ``(start, end)`` for a range mode
``sequence_name``       an output's default basename (the open scene's)
``sequence_fps``        the frame rate of what the host captures
``sound_source``        the armed audio source (a node, a track), or None
``audio_source``        that audio as ``(file, offset in frames)`` for the mux
``resolve_path``        a host path (Blender's ``//``) as the filesystem reads it
``capture_frames``      write ``<prefix>.<frame>.<ext>`` over a frame range
``capture_still``       write one frame to an exact file
``capture_movie``       write a movie with the host's own writer (``native``)
``export_target``       produce a target kind only the host has (Maya's Arnold)
``notify`` / ``warn``   show a notice on the host's message / warning line
======================  ========================================================

With the pure host (:class:`PureSequenceHost`) there is no timeline, nothing to
capture and no audio: the encode and the plan's bookkeeping are what remain.

Two classes, because two callers need different amounts of it:

- :class:`SequenceEncoder` -- frames already exist on disk; turn them into a
  movie. That is all a producer with its OWN capture loop needs (the WebXR
  preview's page recorder pushes frames in one at a time and then asks for an
  mp4), and asking it to satisfy a capture interface it cannot honour would be
  an LSP violation dressed up as reuse.
- :class:`SequenceExporter` -- adds the multi-target plan: capture the source
  ONCE per image format and derive every encoded output from it, isolating a
  per-target failure rather than losing the run. Everything both hosts do
  alike around a capture -- the folder, the range, clearing a prefix's earlier
  frames, counting what was written, the notices -- is written here once; the
  host writes the frames.

The one rule worth restating: **the viewport is read once.** Every encoded
output comes off the same lossless PNG capture, so asking for mp4 + mov costs
one capture and two ffmpeg passes rather than two captures.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from pythontk.core_utils.cancel_scope import CancelScope
from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.temp_artifacts import TempArtifacts
from pythontk.vid_utils._vid_utils import VidUtils


@dataclass(frozen=True)
class ExportTarget:
    """One entry in an exporter's target registry.

    Attributes:
        name: Registry key (e.g. ``"mp4"``).
        label: Human-readable label for UI pickers.
        kind: ``"encode"`` (ffmpeg from the shared capture), ``"sequence"``
            (numbered image frames), ``"still"`` (single frame), ``"native"``
            (a movie the host writes itself, :meth:`SequenceHost.capture_movie`),
            or any name the host answers in :meth:`SequenceHost.export_target`
            (Maya adds ``"arnold"``).
        image_format: Capture compression for image-based kinds.
        extension: Output extension without the dot.
        encoder_options: Extra ffmpeg options for ``encode`` targets.
        native_format: Host movie format for ``native`` targets.
        native_compression: Host movie compression for ``native`` targets.
    """

    name: str
    label: str
    kind: str
    image_format: str = "png"
    extension: str = ""
    encoder_options: Dict[str, Any] = field(default_factory=dict)
    native_format: str = ""
    native_compression: str = ""


@dataclass
class CaptureResult:
    """A captured image sequence on disk."""

    directory: str
    prefix: str
    image_format: str
    start: int
    end: int
    padding: int
    frames: List[str]
    fps: float

    @property
    def pattern(self) -> str:
        """printf-style pattern for the sequence (ffmpeg input)."""
        return FileUtils.format_path(
            os.path.join(
                self.directory,
                f"{self.prefix}.%0{self.padding}d.{self.image_format}",
            )
        )


@dataclass
class ExportResult:
    """Outcome of one export target."""

    target: str
    kind: str
    output: Optional[Union[str, List[str]]] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class SequenceHost(Protocol):
    """What a sequence exporter asks of the host whose frames it captures.

    A short, fixed interface each DCC implements (mayatk's
    ``MayaPlayblastHost``, blendertk's ``BlenderPlayblastHost``: each
    ``PlayblastExporter.HOST``) and :class:`PureSequenceHost` holds with no
    host behind it; a fake stands in for either in a plain-Python test. One
    adapter serves every exporter of its class and holds no state: a capture
    is handed the exporter whose settings it reads (``width``, ``height``,
    ``percent``, ``quality``, ``off_screen``, ``show_ornaments``) and whose
    ``logger`` it reports on. The exporter has already resolved the path, the
    frame range and the camera default, made the folder and cleared the
    prefix's earlier frames; it checks what the host wrote.
    """

    def frame_range(self, mode: str) -> Tuple[float, float]:
        """The timeline's ``(start, end)`` for *mode*, one of the modes the
        exporter's ``RANGE_MODES`` declare beside ``custom`` (``current``: the
        current frame, twice)."""

    def sequence_name(self) -> Optional[str]:
        """An output's default basename (the open scene's name); None when
        the host does not say (the exporter's ``DEFAULT_NAME``)."""

    def sequence_fps(self) -> Optional[float]:
        """The frame rate of the frames the host captures; None when it does
        not say (the exporter's ``DEFAULT_FPS``)."""

    def sound_source(self) -> Optional[str]:
        """The armed audio SOURCE (a node, a track) the host's own movie writer
        mixes in, or None. Its presence is also what makes an encode ask for
        audio at all."""

    def audio_source(self) -> Tuple[Optional[str], float]:
        """``(audio filepath, offset in FRAMES)`` of that source for the ffmpeg
        mux, ``(None, 0.0)`` with none. The exporter rebases the offset onto
        the capture's first frame."""

    def resolve_path(self, path: str) -> str:
        """*path* as the filesystem reads it (Blender's ``//`` is relative to
        the open file); the exporter makes the result absolute."""

    def capture_frames(
        self,
        exporter: "SequenceExporter",
        directory: str,
        prefix: str,
        start: int,
        end: int,
        camera: Any,
        image_format: str,
        padding: int,
        overrides: Dict[str, Any],
    ) -> None:
        """Write ``<directory>/<prefix>.<frame>.<image_format>`` for every frame
        of ``start..end``, numbered by the real frame padded to *padding*
        digits, through *camera* (None: the view as it is). *overrides* are
        the caller's host-specific capture options."""

    def capture_still(
        self,
        exporter: "SequenceExporter",
        filepath: str,
        frame: int,
        camera: Any,
        image_format: str,
        overrides: Dict[str, Any],
    ) -> None:
        """Write *frame* to exactly *filepath* (its folder exists)."""

    def capture_movie(
        self,
        exporter: "SequenceExporter",
        filepath: str,
        fmt: str,
        compression: str,
        start: int,
        end: int,
        camera: Any,
        sound: Optional[str],
        overrides: Dict[str, Any],
    ) -> str:
        """Write ``start..end`` as a movie with the host's own writer (a
        ``native`` target: *fmt* / *compression* are its ``native_format`` /
        ``native_compression``), mixing *sound* in when given. Returns the file
        written: the host may correct the extension."""

    def export_target(
        self,
        exporter: "SequenceExporter",
        spec: ExportTarget,
        output_dir: str,
        name: str,
        start: int,
        end: int,
        camera: Any,
        sound: Optional[str],
        overrides: Dict[str, Any],
    ) -> Union[str, List[str]]:
        """Produce a target of a kind only this host has (Maya's ``arnold``),
        registered in its exporter's ``TARGETS``; the shared plan never grows a
        branch per host."""

    def notify(self, message: str) -> bool:
        """Show a produced-file notice on the host's message line; False when
        it has none (the exporter's logger takes it then)."""

    def warn(self, message: str) -> bool:
        """Show a recoverable failure on the host's warning line; False when
        it has none (the exporter's logger takes it then)."""


class PureSequenceHost:
    """The :class:`SequenceHost` with no host: no timeline, nothing to capture,
    no audio, and paths read as given.

    A host adapter builds on it for what it lacks: :meth:`capture_still` is a
    one-frame :meth:`SequenceExporter.capture_sequence` moved into place, which
    serves any host without a single-frame capture of its own (Blender's).
    """

    def frame_range(self, mode: str) -> Tuple[float, float]:
        raise NotImplementedError(
            f"{type(self).__name__} has no timeline to resolve range mode {mode!r}."
        )

    @staticmethod
    def sequence_name() -> Optional[str]:
        return None

    @staticmethod
    def sequence_fps() -> Optional[float]:
        return None

    @staticmethod
    def sound_source() -> Optional[str]:
        return None

    @staticmethod
    def audio_source() -> Tuple[Optional[str], float]:
        return None, 0.0

    @staticmethod
    def resolve_path(path: str) -> str:
        return path

    def capture_frames(
        self,
        exporter: "SequenceExporter",
        directory: str,
        prefix: str,
        start: int,
        end: int,
        camera: Any,
        image_format: str,
        padding: int,
        overrides: Dict[str, Any],
    ) -> None:
        raise NotImplementedError(
            f"{type(exporter).__name__} has no host to capture frames from."
        )

    def capture_still(
        self,
        exporter: "SequenceExporter",
        filepath: str,
        frame: int,
        camera: Any,
        image_format: str,
        overrides: Dict[str, Any],
    ) -> None:
        directory, basename = os.path.split(filepath)
        capture = exporter.capture_sequence(
            directory,
            prefix=f"{os.path.splitext(basename)[0]}_still",
            start=frame,
            end=frame,
            camera=camera,
            image_format=image_format,
            **overrides,
        )
        os.replace(capture.frames[0], filepath)

    def capture_movie(
        self,
        exporter: "SequenceExporter",
        filepath: str,
        fmt: str,
        compression: str,
        start: int,
        end: int,
        camera: Any,
        sound: Optional[str],
        overrides: Dict[str, Any],
    ) -> str:
        raise NotImplementedError(
            f"{type(exporter).__name__} has no movie writer of its own ({fmt!r})."
        )

    def export_target(
        self,
        exporter: "SequenceExporter",
        spec: ExportTarget,
        output_dir: str,
        name: str,
        start: int,
        end: int,
        camera: Any,
        sound: Optional[str],
        overrides: Dict[str, Any],
    ) -> Union[str, List[str]]:
        raise NotImplementedError(
            f"{type(exporter).__name__} has no producer for target kind "
            f"{spec.kind!r} ({spec.name})."
        )

    @staticmethod
    def notify(message: str) -> bool:
        return False

    @staticmethod
    def warn(message: str) -> bool:
        return False


class SequenceEncoder(LoggingMixin):
    """Encode a numbered image sequence to a movie, and clean up after it.

    No capture: everything here operates on files that already exist. Use it
    as is when you own the capture loop yourself and only want the encode
    (the preview page's recorder); use :class:`SequenceExporter` when you want
    the plan as well. What the host knows -- its frame rate, its armed audio,
    its message line -- comes through :attr:`HOST`.

    Parameters:
        quality: 0-100, mapped onto the H.264 CRF scale for encoded outputs.
        frame_padding: Digits in an image sequence's frame numbers.
        include_audio: Mux the host's audio into movie outputs. Inert until
            the host answers :meth:`SequenceHost.audio_source`.
    """

    #: The host adapter (:class:`SequenceHost`): a DCC's exporter binds its own.
    HOST: SequenceHost = PureSequenceHost()

    #: Peak level (dBFS) below which a muxed audio track is reported as
    #: silent. Digital silence measures ~-91 dB; audible content sits far
    #: above -60 dB.
    _SILENT_PEAK_DB: float = -60.0

    #: ffmpeg video filter rounding both dimensions DOWN to even. H.264 in
    #: ``yuv420p`` rejects an odd width or height, and a scaled capture lands
    #: on one easily (1920 x 51% = 979): ffmpeg then writes a 0-byte file.
    #:
    #: It deliberately does NOT force ``out_range=tv``. That was tried, to make
    #: the limited range explicit rather than inferred: measured, it makes the
    #: scaler run a real pass over every frame (+8s on a 151-frame 720p encode)
    #: and changes the output not at all, because every capture that reaches
    #: here is RGB and ffmpeg already converts RGB to limited-range YUV. It is
    #: only worth its cost for a FULL-range source -- a JPEG capture, which
    #: would otherwise ship as ``yuvj420p (pc)`` -- and nothing here produces
    #: one. A caller that starts to must pass its own ``vf``.
    _EVEN_DIMENSIONS_FILTER: str = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

    #: Frame rate assumed when the host does not state one.
    DEFAULT_FPS: float = 24.0

    #: Name used when the host does not state one.
    DEFAULT_NAME: str = "sequence"

    def __init__(
        self,
        quality: int = 100,
        frame_padding: int = 4,
        include_audio: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.quality = int(quality)
        self.frame_padding = int(frame_padding)
        self.include_audio = bool(include_audio)

    # ------------------------------------------------------------------
    # What the host knows and this module does not
    # ------------------------------------------------------------------
    def sequence_fps(self) -> float:
        """Frame rate of the frames the host captures."""
        return self.HOST.sequence_fps() or self.DEFAULT_FPS

    def sequence_name(self) -> str:
        """Default basename for an output the host writes."""
        return self.HOST.sequence_name() or self.DEFAULT_NAME

    def _notify(self, message: str) -> None:
        """Report a produced file: the host's message line, else the logger."""
        if not self.HOST.notify(message):
            self.logger.info(message)

    def _warn(self, message: str) -> None:
        """Report a recoverable failure: the host's warning line, else the logger."""
        if not self.HOST.warn(message):
            self.logger.warning(message)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    def encode_sequence(
        self,
        capture: Union[CaptureResult, str],
        output_filepath: str,
        fps: Optional[float] = None,
        audio: Optional[Union[bool, str]] = None,
        quality: Optional[int] = None,
        **ffmpeg_options: Any,
    ) -> str:
        """Encode a captured image sequence to a movie via ffmpeg.

        Parameters:
            capture: A :class:`CaptureResult` or a printf-style pattern.
            output_filepath: Destination file; its directory is created.
            fps: Frame rate of the OUTPUT. Defaults to the capture's, then to
                :meth:`sequence_fps`.
            audio: True resolves the host's audio source; a string is an audio
                filepath used as-is.
            quality: 0-100, mapped onto the H.264 CRF scale (100 -> 16).
            **ffmpeg_options: Output options for ffmpeg. ``vf`` defaults to an
                even-dimensions scale (yuv420p rejects odd sizes); a caller
                supplying its own filter chain takes that over.

        Returns:
            The encoded filepath.

        Raises:
            RuntimeError: ffmpeg produced no usable file.

        A warning is logged when audio was muxed but the resulting track is
        effectively silent -- e.g. the armed sound has no audible content over
        the encoded frame range.
        """
        if isinstance(capture, CaptureResult):
            pattern = capture.pattern
            start_number: Optional[int] = capture.start
            fps = fps if fps is not None else capture.fps
            capture_start = capture.start
        else:
            pattern = capture
            start_number = None
            capture_start = None
        fps = fps if fps is not None else self.sequence_fps()

        audio_filepath, audio_offset = None, 0.0
        if audio:
            audio_filepath, audio_offset = self._resolve_audio(
                audio, capture_start, fps
            )

        quality = self.quality if quality is None else int(quality)
        ffmpeg_options.setdefault("crf", self._quality_to_crf(quality))
        ffmpeg_options.setdefault("vf", self._EVEN_DIMENSIONS_FILTER)

        output_filepath = FileUtils.format_path(os.path.abspath(output_filepath))
        output_dir = os.path.dirname(output_filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        encoded = VidUtils.compress_video(
            input_filepath=pattern,
            output_filepath=output_filepath,
            frame_rate=fps,
            start_number=start_number,
            audio_filepath=audio_filepath,
            audio_offset=audio_offset,
            **ffmpeg_options,
        )
        if not encoded or not self._is_valid_file(encoded):
            raise RuntimeError(
                f"ffmpeg encode failed for {pattern!r} -> {output_filepath!r}."
            )
        if audio_filepath:
            peak = self._audio_peak_db(encoded)
            if peak is not None and peak < self._SILENT_PEAK_DB:
                self.logger.warning(
                    f"Encoded audio track is effectively silent (peak "
                    f"{peak:.1f} dB): {audio_filepath!r} has no audible "
                    "content over the encoded frame range."
                )
        self._notify(f"Encoded movie created: {encoded}")
        return encoded

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _quality_to_crf(quality: int) -> int:
        """Map 0-100 quality onto the H.264 CRF scale (100 -> 16, 0 -> 40)."""
        quality = max(0, min(100, int(quality)))
        return round(40 - quality * 0.24)

    @staticmethod
    def _audio_peak_db(filepath: str) -> Optional[float]:
        """Peak level (dBFS) of a file's first audio stream, or None when it
        can't be measured (no ffmpeg, no/undecodable audio stream)."""
        ffmpeg = VidUtils.resolve_ffmpeg(required=False)
        if not ffmpeg:
            return None
        try:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-i",
                    filepath,
                    "-map",
                    "0:a:0",
                    "-af",
                    "volumedetect",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except OSError:  # a diagnostics probe must never fail the export
            return None
        match = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", result.stderr)
        return float(match.group(1)) if match else None

    @staticmethod
    def _is_valid_file(path: Optional[str]) -> bool:
        return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0

    @staticmethod
    def _collect_frames(
        directory: str,
        prefix: str,
        image_format: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> List[str]:
        """Frames on disk matching ``<prefix>.<number>.<ext>``, numerically
        sorted; ``start``/``end`` bound the frame numbers when given."""
        regex = re.compile(
            rf"{re.escape(prefix)}\.(\d+)\.{re.escape(image_format)}$", re.IGNORECASE
        )
        numbered = []
        try:
            entries = os.listdir(directory)
        except OSError:
            return []
        for entry in entries:
            match = regex.match(entry)
            if not match:
                continue
            number = int(match.group(1))
            if (start is None or number >= start) and (end is None or number <= end):
                numbered.append(
                    (number, FileUtils.format_path(os.path.join(directory, entry)))
                )
        return [path for _, path in sorted(numbered)]

    def _remove_frames(self, directory: str, prefix: str, image_format: str) -> None:
        """Delete every frame on disk matching ``<prefix>.<number>.<ext>``."""
        for frame in self._collect_frames(directory, prefix, image_format):
            try:
                os.remove(frame)
            except OSError as exc:
                self.logger.warning(f"Could not remove frame {frame!r}: {exc}")

    def _resolve_audio(
        self,
        audio: Union[bool, str],
        capture_start: Optional[int],
        fps: float,
    ) -> Tuple[Optional[str], float]:
        """``(audio filepath, offset seconds)`` for an encode; ``(None, 0)`` if
        unresolvable.

        A string is a filepath used as-is; True asks the host
        (:meth:`SequenceHost.audio_source`) and rebases its frame offset
        against the capture's first frame, so audio armed against the timeline
        lands where it belongs in a movie that starts mid-range.
        """
        if isinstance(audio, str):
            return (audio if os.path.isfile(audio) else None), 0.0
        filepath, offset_frames = self.HOST.audio_source()
        if not filepath:
            return None, 0.0
        offset_seconds = (
            (float(offset_frames) - capture_start) / fps
            if capture_start is not None and fps
            else 0.0
        )
        return filepath, offset_seconds


class SequenceExporter(SequenceEncoder):
    """Plan and run several outputs off ONE capture of a host's frames.

    The host adapter (:attr:`HOST`) writes the pixels and answers the target
    kinds only it has. Everything else -- what the targets are, which captures
    the plan needs, in what order, the folders and frame bookkeeping around
    each capture, how failures are isolated and what gets cleaned up -- lives
    here and is shared. A DCC's ``PlayblastExporter`` is this class bound to
    its host: configuration (``HOST``, ``TARGETS``, ``RANGE_MODES``) plus the
    host's own additions, never an override.

    Instance attributes hold capture *defaults*; every public method accepts
    per-call overrides. Frame ranges are resolved at call time (never cached),
    so the exporter tracks timeline changes made after construction.

    Parameters:
        camera: Default camera for captures (the host's spelling: a node name,
            an object). ``None`` keeps whatever the view is looking through.
        width/height: Capture resolution.
        percent: The host's capture scale percentage.
        quality: 0-100; drives the ffmpeg CRF for encoded targets (and, for a
            host that has one, its own movie quality).
        off_screen: Capture offscreen where the host can choose (avoids
            viewport redraw artifacts).
        show_ornaments: Draw the viewport's HUD / overlays into the capture.
        frame_padding: Digits for image-sequence frame numbers.
        include_audio: Attach the host's audio to movie outputs.
    """

    #: Registry of exportable outputs. Extend with new :class:`ExportTarget`
    #: entries -- UIs build their pickers from :meth:`available_targets`.
    TARGETS: Dict[str, ExportTarget] = {
        t.name: t
        for t in (
            ExportTarget(
                "mp4",
                "MP4 (H.264)",
                "encode",
                extension="mp4",
                encoder_options={"movflags": "+faststart"},
            ),
            ExportTarget(
                "mov",
                "MOV (H.264)",
                "encode",
                extension="mov",
                encoder_options={"movflags": "+faststart"},
            ),
            ExportTarget("png_sequence", "PNG Sequence", "sequence", extension="png"),
            ExportTarget(
                "jpg_sequence",
                "JPEG Sequence",
                "sequence",
                image_format="jpg",
                extension="jpg",
            ),
            ExportTarget(
                "tif_sequence",
                "TIFF Sequence",
                "sequence",
                image_format="tif",
                extension="tif",
            ),
            ExportTarget(
                "tga_sequence",
                "TGA Sequence",
                "sequence",
                image_format="tga",
                extension="tga",
            ),
            ExportTarget(
                "still", "PNG Still (Current Frame)", "still", extension="png"
            ),
        )
    }

    #: Frame-range modes accepted by :meth:`resolve_frame_range`. A host with a
    #: timeline declares its own (``playback``, ``animation``, ``current``) and
    #: its adapter answers them (:meth:`SequenceHost.frame_range`).
    RANGE_MODES: Tuple[str, ...] = ("custom",)

    #: Range :meth:`export` uses when the caller names none. ``custom`` here
    #: because it is the only mode a host-less exporter HAS -- and a host with a
    #: timeline sets its own rather than leaving the default to whichever mode
    #: happens to be listed first, which is how "export the current shot"
    #: quietly becomes "export nothing at all".
    DEFAULT_RANGE_MODE: str = "custom"

    #: ``export`` keys owned by its own named parameters / per-target planning.
    #: A stray duplicate in ``**overrides`` would TypeError one target mid-plan.
    #: A host whose own target kinds take further arguments EXTENDS this rather
    #: than restating it -- the shared plan cannot know what a host's capture
    #: call is named, and listing a host's flags here made the base pretend to.
    _RESERVED_OVERRIDES: Tuple[str, ...] = (
        "image_format",
        "camera",
        "start",
        "end",
        "prefix",
        "filepath",
        "directory",
    )

    #: Override keys the shared image capture reads itself (an image
    #: sequence's digits): a movie, a still and a host's own kind have none,
    #: so these never reach the host's call (Maya's playblast refuses a flag
    #: it does not know).
    _CAPTURE_ONLY_OVERRIDES: Tuple[str, ...] = ("frame_padding",)

    def __init__(
        self,
        camera: Any = None,
        width: int = 1920,
        height: int = 1080,
        percent: int = 100,
        quality: int = 100,
        off_screen: bool = True,
        show_ornaments: bool = True,
        frame_padding: int = 4,
        include_audio: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            quality=quality,
            frame_padding=frame_padding,
            include_audio=include_audio,
            **kwargs,
        )
        self.camera = camera
        self.width = int(width)
        self.height = int(height)
        self.percent = int(percent)
        self.off_screen = bool(off_screen)
        self.show_ornaments = bool(show_ornaments)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @classmethod
    def available_targets(cls) -> List[Tuple[str, str]]:
        """(name, label) pairs in registry order — for building UI pickers."""
        return [(t.name, t.label) for t in cls.TARGETS.values()]

    @classmethod
    def scene_name(cls) -> str:
        """An output's default basename, class-level (a panel shows it before
        it builds an exporter): the host scene's name, else
        :attr:`DEFAULT_NAME`."""
        return cls.HOST.sequence_name() or cls.DEFAULT_NAME

    @classmethod
    def scene_fps(cls) -> float:
        """The host scene's frame rate, class-level, else :attr:`DEFAULT_FPS`."""
        return cls.HOST.sequence_fps() or cls.DEFAULT_FPS

    @classmethod
    def resolve_frame_range(
        cls,
        mode: str = "custom",
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Resolve a frame range from a mode, with explicit overrides.

        ``custom`` is the only mode every host has (both bounds required);
        a host with a timeline registers its modes in :attr:`RANGE_MODES` and
        its adapter answers them. Explicit ``start``/``end`` override the
        mode's values individually.

        Raises:
            ValueError: An unknown mode, a ``custom`` call missing a bound, or
                a resolved range that runs backwards.
            NotImplementedError: A declared mode the host does not answer.
        """
        if mode not in cls.RANGE_MODES:
            raise ValueError(
                f"Unknown range mode {mode!r}; expected one of {cls.RANGE_MODES}."
            )
        if mode == "custom":
            if start is None or end is None:
                raise ValueError("Custom range mode requires both start and end.")
            mode_start, mode_end = start, end
        else:
            mode_start, mode_end = cls.HOST.frame_range(mode)

        resolved_start = int(start if start is not None else mode_start)
        resolved_end = int(end if end is not None else mode_end)
        if resolved_start > resolved_end:
            raise ValueError(
                f"Start frame {resolved_start} is after end frame {resolved_end}."
            )
        return resolved_start, resolved_end

    # ------------------------------------------------------------------
    # Capture primitives -- the host writes, this checks
    # ------------------------------------------------------------------
    def capture_sequence(
        self,
        directory: str,
        prefix: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        camera: Any = None,
        image_format: str = "png",
        **overrides: Any,
    ) -> CaptureResult:
        """Capture the frame range as a numbered image sequence.

        Frames keep their real frame numbers (``<prefix>.<frame>.<ext>``).
        ``frame_padding`` in *overrides* sets this capture's digits; the rest
        are the host's capture options.

        Raises:
            RuntimeError: The host wrote fewer frames than the range holds (an
                interrupted capture stops early).
        """
        directory = self._resolve_path(directory)
        os.makedirs(directory, exist_ok=True)
        prefix = prefix or self.sequence_name()
        start, end = self.resolve_frame_range(self.DEFAULT_RANGE_MODE, start, end)
        padding = int(overrides.pop("frame_padding", self.frame_padding))
        camera = self.camera if camera is None else camera

        # directory+prefix identifies THIS capture: frames left by an
        # earlier/wider run would pass the count check below and -- worse --
        # ffmpeg reads a printf pattern contiguously past ``end``, encoding
        # stale frames into the movie.
        self._remove_frames(directory, prefix, image_format)
        self.HOST.capture_frames(
            self, directory, prefix, start, end, camera, image_format, padding,
            overrides,
        )

        frames = self._collect_frames(directory, prefix, image_format, start, end)
        expected = end - start + 1
        if len(frames) < expected:
            raise RuntimeError(
                f"The capture wrote {len(frames)}/{expected} frames under "
                f"{directory!r} (prefix {prefix!r}) -- an interrupted capture "
                "(Esc) stops early."
            )
        return CaptureResult(
            directory=directory,
            prefix=prefix,
            image_format=image_format,
            start=start,
            end=end,
            padding=padding,
            frames=frames,
            fps=self.sequence_fps(),
        )

    def capture_still(
        self,
        filepath: str,
        frame: Optional[int] = None,
        camera: Any = None,
        image_format: str = "png",
        **overrides: Any,
    ) -> str:
        """Capture a single frame (default: the current one) to an exact filepath.

        Raises:
            RuntimeError: No file was written.
        """
        filepath = self._resolve_path(filepath)
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        frame = int(frame if frame is not None else self.HOST.frame_range("current")[0])
        camera = self.camera if camera is None else camera
        overrides = self._host_overrides(overrides)

        self.HOST.capture_still(self, filepath, frame, camera, image_format, overrides)

        if not self._is_valid_file(filepath):
            raise RuntimeError(f"Still capture failed; no file at {filepath!r}.")
        self._notify(f"Still frame captured: {filepath}")
        return filepath

    def _capture_movie(
        self,
        filepath: str,
        fmt: str,
        compression: str,
        start: Optional[int],
        end: Optional[int],
        camera: Any,
        sound: Optional[str],
        overrides: Dict[str, Any],
    ) -> str:
        """A movie written by the host's own writer (a ``native`` target): the
        shared half of each host's ``capture_movie``, which names *fmt* and
        *compression* in its own vocabulary."""
        filepath = self._resolve_path(filepath)
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        start, end = self.resolve_frame_range(self.DEFAULT_RANGE_MODE, start, end)
        camera = self.camera if camera is None else camera

        produced = self.HOST.capture_movie(
            self, filepath, fmt, compression, start, end, camera, sound,
            self._host_overrides(overrides),
        )

        if not self._is_valid_file(produced):
            raise RuntimeError(f"Movie capture failed; no file at {produced!r}.")
        self._notify(f"Playblast movie created: {produced}")
        return produced

    @classmethod
    def _host_overrides(cls, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """*overrides* less :attr:`_CAPTURE_ONLY_OVERRIDES`: what a host call
        other than the image capture is handed."""
        return {
            key: value
            for key, value in overrides.items()
            if key not in cls._CAPTURE_ONLY_OVERRIDES
        }

    def _resolve_path(self, path: str) -> str:
        """*path* absolute and normalized, read through the host's own path
        syntax first (:meth:`SequenceHost.resolve_path`)."""
        return FileUtils.format_path(os.path.abspath(self.HOST.resolve_path(path)))

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def export(
        self,
        output_dir: str,
        name: Optional[str] = None,
        targets: Union[str, Sequence[str]] = ("mp4",),
        range_mode: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        camera: Any = None,
        keep_frames: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **overrides: Any,
    ) -> List[ExportResult]:
        """Produce one or more registered targets from a single plan.

        The source is captured once per required image format; every
        ``encode`` target reuses the lossless PNG capture. When no
        ``png_sequence`` target is requested, the intermediate frames live in a
        scoped temp store (``TempArtifacts``), never the output folder, and are
        deleted afterward; ``keep_frames=True`` makes them an output instead,
        kept in ``<output_dir>/<name>_png_tmp``.

        Outputs: ``<dir>/<name>.<ext>`` for movies and the still;
        ``<dir>/<name>_<fmt>/`` for sequences. *output_dir* is read through
        the host's path syntax (Blender's ``//``). Per-target failures are
        captured on the returned :class:`ExportResult`\\ s rather than raised.

        Encoded targets are pre-flighted: with no ffmpeg they fail before any
        frame is captured. Between plan steps the export honours the ambient
        :class:`~pythontk.CancelScope` (a cancelled scope raises
        :class:`~pythontk.OperationCancelled` out of it, after cleanup).

        Raises:
            ValueError: An unregistered target name, or an unresolvable range.
        """
        if isinstance(targets, str):
            targets = [targets]
        unknown = [t for t in targets if t not in self.TARGETS]
        if unknown:
            raise ValueError(
                f"Unknown export target(s) {unknown}; available: {sorted(self.TARGETS)}."
            )
        ordered = list(dict.fromkeys(targets))  # dedupe, keep order
        specs = [self.TARGETS[t] for t in ordered]

        for owned in self._RESERVED_OVERRIDES:
            overrides.pop(owned, None)

        output_dir = self._resolve_path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        name = name or self.sequence_name()
        start, end = self.resolve_frame_range(
            range_mode or self.DEFAULT_RANGE_MODE, start, end
        )

        # The host's armed source: a native movie mixes it in itself, and its
        # presence decides whether an encode asks for audio at all.
        sound_node = self.HOST.sound_source() if self.include_audio else None

        results: Dict[str, ExportResult] = {
            s.name: ExportResult(target=s.name, kind=s.kind) for s in specs
        }
        encode_specs = [s for s in specs if s.kind == "encode"]
        sequence_specs = [s for s in specs if s.kind == "sequence"]

        # Fail fast: an encode needs ffmpeg for a second at the END of minutes
        # of capture. Resolve it before paying for frames nothing can consume;
        # a miss errors every encode target and drops them from the plan, so
        # the shared PNG capture is skipped when nothing else needs it.
        if encode_specs:
            try:
                VidUtils.resolve_ffmpeg(required=True)
            except FileNotFoundError as exc:
                for spec in encode_specs:
                    results[spec.name].error = str(exc)
                    self._warn(f"Export target '{spec.name}' failed: {exc}")
                encode_specs = []

        # One capture per required image format. Encodes ride on the png
        # capture — shared with a requested png_sequence when present.
        capture_formats = {s.image_format for s in sequence_specs}
        needs_tmp_png = bool(encode_specs) and "png" not in capture_formats
        plan_formats = sorted(capture_formats | ({"png"} if needs_tmp_png else set()))

        total_steps = (
            len(plan_formats)
            + len(encode_specs)
            + sum(1 for s in specs if s.kind not in ("sequence", "encode"))
        )
        step = 0

        def progress(label: str) -> None:
            nonlocal step
            # Cooperative cancel between plan steps (a no-op with no ambient
            # scope). OperationCancelled is a BaseException: the per-target
            # isolation below cannot swallow it, and ``finally`` still cleans up.
            CancelScope.check()
            if progress_callback:
                progress_callback(step, total_steps, label)
            step += 1

        captures: Dict[str, CaptureResult] = {}
        # The encodes' own PNG capture is scratch: in a scoped temp store, never
        # the deliverable folder -- unless the caller asked to keep the frames,
        # which makes them an output like any other. No such capture, no folder.
        scratch = None
        tmp_png_dir = None
        if needs_tmp_png and keep_frames:
            tmp_png_dir = FileUtils.format_path(
                os.path.join(output_dir, f"{name}_png_tmp")
            )
        elif needs_tmp_png:
            scratch = TempArtifacts.scoped("sequence_frames")
            tmp_png_dir = FileUtils.format_path(scratch.dir_path())

        try:
            for fmt in plan_formats:
                progress(f"Capturing {fmt} frames")
                is_tmp = fmt == "png" and needs_tmp_png
                seq_dir = (
                    tmp_png_dir
                    if is_tmp
                    else FileUtils.format_path(
                        os.path.join(output_dir, f"{name}_{fmt}")
                    )
                )
                try:
                    captures[fmt] = self.capture_sequence(
                        directory=seq_dir,
                        prefix=name,
                        start=start,
                        end=end,
                        camera=camera,
                        image_format=fmt,
                        **overrides,
                    )
                except Exception as exc:  # noqa: BLE001 - isolate per plan step
                    self.logger.warning(f"Capture ({fmt}) failed: {exc}")
                    dependents = [s for s in sequence_specs if s.image_format == fmt]
                    if fmt == "png":
                        dependents += encode_specs
                    for spec in dependents:
                        results[spec.name].error = str(exc)

            for spec in specs:
                result = results[spec.name]
                if result.error is not None:
                    continue
                try:
                    if spec.kind == "sequence":
                        capture = captures.get(spec.image_format)
                        if capture is None:  # invariant: errored above otherwise
                            raise RuntimeError(
                                f"{spec.image_format} capture unavailable."
                            )
                        result.output = capture.frames
                    elif spec.kind == "encode":
                        capture = captures.get("png")
                        if capture is None:
                            raise RuntimeError("Shared PNG capture unavailable.")
                        progress(f"Encoding {spec.label}")
                        result.output = self.encode_sequence(
                            capture,
                            os.path.join(output_dir, f"{name}.{spec.extension}"),
                            audio=bool(sound_node),
                            **dict(spec.encoder_options),
                        )
                    elif spec.kind == "still":
                        progress(f"Capturing {spec.label}")
                        result.output = self.capture_still(
                            os.path.join(output_dir, f"{name}.{spec.extension}"),
                            camera=camera,
                            image_format=spec.image_format,
                            **overrides,
                        )
                    elif spec.kind == "native":
                        # Not off the shared capture: the host's own writer
                        # makes the container in its own pass.
                        progress(f"Producing {spec.label}")
                        result.output = self._capture_movie(
                            os.path.join(output_dir, f"{name}.{spec.extension}"),
                            spec.native_format,
                            spec.native_compression,
                            start,
                            end,
                            camera,
                            sound_node,
                            overrides,
                        )
                    else:
                        progress(f"Producing {spec.label}")
                        result.output = self.HOST.export_target(
                            self,
                            spec,
                            output_dir,
                            name,
                            start,
                            end,
                            camera,
                            sound_node,
                            self._host_overrides(overrides),
                        )
                except Exception as exc:  # noqa: BLE001 - isolate per target
                    result.error = str(exc)
                    self._warn(f"Export target '{spec.name}' failed: {exc}")
        finally:
            # By disk scan, never the CaptureResult: a capture that raised
            # part-way (an interrupted playblast) has no result yet leaves its
            # frames behind -- an mp4 request must not end as a folder of PNGs.
            if scratch is not None:
                scratch.cleanup()

        if progress_callback:
            progress_callback(total_steps, total_steps, "Done")
        return [results[s.name] for s in specs]
