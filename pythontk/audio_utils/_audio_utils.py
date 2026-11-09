# !/usr/bin/python
# coding=utf-8
import hashlib
import array as _array
import os
import struct
import subprocess
import wave as _wave
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from pythontk.core_utils.help_mixin import HelpMixin


class AudioUtils(HelpMixin):
    """Utility helpers for portable audio-file preparation.

    Provides ffmpeg-backed conversion and WAV-compositing helpers.
    """

    PLAYABLE_EXTENSIONS = {".wav", ".aif", ".aiff"}
    SOURCE_EXTENSIONS = PLAYABLE_EXTENSIONS | {".mp3", ".ogg", ".m4a", ".flac"}

    @classmethod
    def resolve_ffmpeg(
        cls, required: bool = True, auto_install: bool = False
    ) -> Optional[str]:
        """Resolve the ffmpeg executable from PATH or a managed install.

        Delegates to :meth:`VidUtils.resolve_ffmpeg`. Both names stay
        published because they have different callers -- the audio side passes
        ``auto_install=True`` -- but there is one implementation: these were
        statement-for-statement identical apart from a single error string,
        and only the video copy is reachable as ``ptk.resolve_ffmpeg`` (
        ``vid_utils`` is a wildcard root), so a fix applied to the bare name
        silently missed this one.

        Parameters:
            required: Raise when ffmpeg cannot be resolved.
            auto_install: Download and install it when it is not found.

        Returns:
            Path to the executable, or None when not found and *required* is
            False.

        Raises:
            FileNotFoundError: Not resolved and *required* is True. The
                message names the auto-install failure when there was one.
        """
        from pythontk.vid_utils._vid_utils import VidUtils

        return VidUtils.resolve_ffmpeg(required=required, auto_install=auto_install)

    #: :meth:`memo_duration`'s readings: ``(path, mtime_ns, size, fps)`` ->
    #: the probe's ``(frames, resolved_path)``.
    _durations: Dict[tuple, Tuple[float, str]] = {}
    #: How many readings it keeps before starting over.
    _DURATIONS_CAP = 4096

    @classmethod
    def memo_duration(
        cls,
        file_path: str,
        fps: float,
        probe: Callable[[str, float], Tuple[float, str]],
    ) -> Tuple[float, str]:
        """*probe* -- a host's reader of a clip's length -- read once per file
        as it is on disk.

        Kept by the file's resolved path, modification time and size, and by
        *fps*: a file replaced or rewritten reads again, an unchanged one is
        not re-read (a caller that sizes shots from their clips asks per table
        resolve, and each read opened a WAV on a synced drive).  A file that
        cannot be stat'ed, or a reading of no length (a source not converted
        yet), is never kept.

        Parameters:
            file_path: The clip's source.
            fps: The frame rate *probe* counts in.
            probe: ``(file_path, fps) -> (frames, resolved_path)``, the host's
                reader (mayatk's ``AudioUtils.audio_duration_frames``).

        Returns:
            *probe*'s ``(frames, resolved_path)`` for the file as it is now.
        """
        try:
            stat = os.stat(file_path)
        except (OSError, TypeError, ValueError):
            return probe(file_path, fps)
        key = (
            os.path.normcase(os.path.abspath(file_path)),
            stat.st_mtime_ns,
            stat.st_size,
            float(fps),
        )
        hit = cls._durations.get(key)
        if hit is not None:
            return hit
        result = probe(file_path, fps)
        if result and result[0] and result[0] > 0:
            if len(cls._durations) >= cls._DURATIONS_CAP:
                cls._durations.clear()
            cls._durations[key] = result
        return result

    @classmethod
    def is_playable_extension(cls, file_path: str) -> bool:
        """Return True if extension is already timeline-playable."""
        return os.path.splitext(file_path)[1].lower() in cls.PLAYABLE_EXTENSIONS

    @classmethod
    def is_supported_source_extension(cls, file_path: str) -> bool:
        """Return True if extension is accepted as conversion source."""
        return os.path.splitext(file_path)[1].lower() in cls.SOURCE_EXTENSIONS

    @classmethod
    def cache_dir_for(cls, audio_path: str, create: bool = False) -> str:
        """Where a playable transcode of *audio_path* is cached -- the one place
        that decides it.

        A source inside a marked project caches in that project's
        ``cache/audio`` (:meth:`Workspace.cache_dir <pythontk.Workspace.cache_dir>`):
        the scene that plays the clip references the transcode, so it travels
        with the project. Any other source caches in the per-user cache
        (:meth:`UserConfig.cache_dir <pythontk.UserConfig.cache_dir>`). Never a
        folder written beside the source.

        Parameters:
            audio_path: The source audio file.
            create: Create the folder.

        Returns:
            The cache folder, forward slashes.
        """
        from pythontk.core_utils.user_config import UserConfig
        from pythontk.file_utils.workspace import Workspace

        project = Workspace.find_containing(audio_path)
        if project is not None:
            folder = project.cache_dir("audio", create=create)
        else:
            folder = str(UserConfig.cache_dir("audio", create=create))
        return folder.replace("\\", "/")

    @staticmethod
    def _is_complete_wav(path: str) -> bool:
        """Whether *path* is a whole WAV file: its RIFF header's size field
        matches the file, which a writer fills in only once it has finished
        (ffmpeg writes a placeholder first and patches it at the end)."""
        try:
            size = os.path.getsize(path)
            with open(path, "rb") as fh:
                head = fh.read(12)
        except OSError:
            return False
        if size < 44 or head[8:12] != b"WAVE":
            return False
        if head[:4] in (b"RF64", b"BW64"):  # 64-bit sizes live in a ds64 chunk
            return True
        return head[:4] == b"RIFF" and int.from_bytes(head[4:8], "little") + 8 == size

    @classmethod
    def ensure_playable_path(
        cls,
        audio_path: str,
        cache_dir: Optional[str] = None,
    ) -> str:
        """Return a playable audio path, converting with ffmpeg if required.

        The transcode is written whole, then moved into its cache name
        (:meth:`FileUtils.atomic_write <pythontk.FileUtils.atomic_write>`), and
        a cached file is reused only when it is a complete WAV: a failed or
        interrupted run never leaves a file the next call would return.

        Parameters:
            audio_path: Source audio file path.
            cache_dir: Cache directory for the converted WAV; default
                :meth:`cache_dir_for` (the source's project, else the per-user
                cache).

        Returns:
            Playable file path (original or converted WAV).

        Raises:
            ValueError: Unsupported file extension.
            FileNotFoundError: Missing source file, or ffmpeg when no complete
                transcode is cached.
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

        if cache_dir is None:
            cache_dir = cls.cache_dir_for(source)
        cache_dir = os.path.normpath(cache_dir).replace("\\", "/")

        stat = os.stat(source)
        token = f"{source}|{stat.st_mtime_ns}|{stat.st_size}"
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()[:10]
        stem = os.path.splitext(os.path.basename(source))[0]
        output = os.path.join(cache_dir, f"{stem}_{digest}.wav").replace("\\", "/")

        if cls._is_complete_wav(output):
            return output  # a cached transcode plays without ffmpeg

        ffmpeg_cmd = cls.resolve_ffmpeg(required=True)
        os.makedirs(cache_dir, exist_ok=True)
        creation_flags = 0
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NO_WINDOW

        def transcode(part: str) -> None:
            # -f: the part's extension is not ``.wav``, so name the muxer.
            cmd = [
                ffmpeg_cmd,
                "-y",
                "-i",
                source,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-f",
                "wav",
                part,
            ]
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                creationflags=creation_flags,
            )
            if completed.returncode != 0 or not cls._is_complete_wav(part):
                stderr = (completed.stderr or "").strip().splitlines()
                tail = stderr[-1] if stderr else "unknown ffmpeg error"
                raise RuntimeError(f"ffmpeg conversion failed for '{source}': {tail}")

        from pythontk.file_utils._file_utils import FileUtils

        FileUtils.atomic_write(output, transcode)
        return output

    @staticmethod
    def clip_cues(
        sounds: Iterable[Mapping[str, Any]],
        takes: Iterable[Mapping[str, Any]],
        fps: float,
        window: Optional[Tuple[float, float]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Cut each sound on the timeline into the cues of the clips that
        play it: the ``audio`` record's ``cues`` (``SceneRecords.AUDIO``), one
        rule for every DCC's producer.

        A clip is a take the file declares: one per shot (*takes*), or, in a
        file with no shots, its one take (``clip: ""``) over *window* -- the
        window the file declares for it, which is what an engine's importer
        cuts the clip by: measured off the written file
        (``FbxMedia.fit_take_windows``, which makes it the keys it holds), or,
        before a write, what the host's writer declares by itself. A take
        missing its name or range cuts nothing: defaulted, it would be a
        phantom take swallowing frame-0 sounds.

        A sound plays from ``start`` to ``stop`` (timeline frames, the stop
        exclusive; ``None`` when no end is known: it plays to its clip's
        end), heard from ``offset`` seconds into its file. Each clip whose
        take (``start``..``end``, inclusive) overlaps it gets one cue: the
        sound starts at the later of the two starts, ``offset`` further into
        the sound for each frame it began before the clip, and stops at the
        earlier of the two ends -- so a sound across a cut stops at the cut
        and the next clip plays it on from there. A cue of no length is none.
        Times are seconds from the clip's start (frames over *fps*), rounded
        to the microsecond.

        Weighed: FBX's own audio (``FbxAudio`` / ``FbxAudioLayer``, SDK 2017)
        is written by neither DCC's exporter and read by no engine importer;
        OpenTimelineIO models clips and markers on an editorial timeline, not
        sounds timed inside an animation clip, and no reader of the delivery
        reads it; glTF's ``KHR_audio_emitter`` places emitters on nodes and
        times nothing per animation; Unity's ``AnimationEvent`` (a time, a
        function, an object) is the shape adopted: each cue is what one
        ``PlayAudio`` / ``StopAudio`` event pair holds, and a Unity user with
        no DCC authors the same events.

        Parameters:
            sounds: ``{"sound", "start", "stop", "offset"?}`` -- the delivered
                file's name, its span in timeline frames, and seconds into the
                file it is heard from at ``start`` (0 when absent).
            takes: ``{"name", "start", "end"}`` entries, timeline frames
                (``ShotStore.declared_takes``: one per shot).
            fps: The scene's frame rate.
            window: For a file with no usable take, the ``(start, end)``
                frames its one take declares; None, and it has no clip to cue.

        Returns:
            ``(cues, dropped)``: the record's ``{"clip", "sound", "time",
            "offset", "stop"}`` cues, clip by clip in *takes*' order and by
            time within each; and ``"sound@frame"`` for each sound no clip
            plays.
        """
        sounds = list(sounds)
        clips = [
            (str(t["name"]), float(t["start"]), float(t["end"]))
            for t in takes
            if t.get("name") is not None
            and t.get("start") is not None
            and t.get("end") is not None
        ]
        if not clips and window:
            clips = [("", float(window[0]), float(window[1]))]
        rate = float(fps)
        cues: List[Dict[str, Any]] = []
        heard = set()
        for name, clip_start, clip_end in clips:
            within = []
            for index, sound in enumerate(sounds):
                start = float(sound["start"])
                stop = sound.get("stop")
                end = clip_end if stop is None else min(float(stop), clip_end)
                begin = max(start, clip_start)
                if end <= begin:
                    continue
                heard.add(index)
                offset = float(sound.get("offset") or 0.0) + (begin - start) / rate
                within.append(
                    {
                        "clip": name,
                        "sound": str(sound["sound"]),
                        "time": round((begin - clip_start) / rate, 6),
                        "offset": round(offset, 6),
                        "stop": round((end - clip_start) / rate, 6),
                    }
                )
            within.sort(key=lambda cue: cue["time"])
            cues.extend(within)
        dropped = [
            f"{sound['sound']}@{sound['start']:g}"
            for index, sound in enumerate(sounds)
            if index not in heard
        ]
        return cues, dropped

    @classmethod
    def media_digests(
        cls,
        cues: Iterable[Mapping[str, Any]],
        files: Mapping[str, bytes],
    ) -> Dict[str, str]:
        """The content digest of each audio file of *files* a cue plays, by
        the cue's ``sound``.

        The ``audio`` record's ``media`` (``SceneRecords.AUDIO``). A cue names
        its sound by the file the delivery carries, and a reader finds the
        sound by that name, so two deliveries can name different files alike
        -- two models each embedding their own ``door.wav`` -- and the second
        bound the first's (2026-10-19). A reader stands a same-named sound in
        for the delivery's only when its bytes hash to this; else it takes
        the delivery's own. So it is digested from the bytes the delivery
        carries, never a source that may not be in it: the FBX dependency
        pass writes it from what it embedded (``FbxMedia.embed_dependencies``),
        and Unity extracts those bytes byte for byte, so both hash one string
        of bytes.

        A cue's file is the audio file (:attr:`SOURCE_EXTENSIONS`) of its
        ``sound``'s name, ignoring case -- the reader's join; a name two files
        share ignoring case tells the reader nothing, so it gets no digest.

        Parameters:
            cues: The record's cues (``{"sound", ...}``).
            files: ``{file name: bytes}`` -- what the delivery embeds.

        Returns:
            ``{sound: SHA-256 of its file's bytes, lowercase hex}``, in cue
            order; a sound with no file among *files* is left out.
        """
        by_name: Dict[str, List[bytes]] = {}
        for file_name, data in files.items():
            if os.path.splitext(file_name)[1].lower() in cls.SOURCE_EXTENSIONS:
                by_name.setdefault(file_name.casefold(), []).append(data)
        media: Dict[str, str] = {}
        for cue in cues:
            sound = cue.get("sound") if isinstance(cue, Mapping) else None
            if not isinstance(sound, str) or not sound or sound in media:
                continue
            found = by_name.get(sound.casefold(), [])
            if len(found) == 1:
                media[sound] = hashlib.sha256(found[0]).hexdigest()
        return media

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
                                f"Skipping '{path}': sample rate {sr} != {sample_rate}"
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

        # Optional dep — only the mixing path needs numpy.
        import numpy as np

        total_samples = max(pos + len(s) for pos, s in clips)
        if total_samples <= 0:
            return None

        composite = np.zeros(total_samples, dtype=np.int32)
        for pos, samples in clips:
            arr = np.frombuffer(samples, dtype=np.int16).astype(np.int32)
            if pos < 0:
                arr = arr[-pos:]
                pos = 0
            if not len(arr):
                continue
            end = min(pos + len(arr), total_samples)
            if pos >= end:
                continue
            composite[pos:end] += arr[: end - pos]
        np.clip(composite, -32768, 32767, out=composite)
        raw_out = composite.astype(np.int16).tobytes()

        output = os.path.normpath(output_path)
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

        with _wave.open(output, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_out)

        if logger:
            logger.info(
                f"Built composite: {total_samples // channels} samples, "
                f"{total_samples / (channels * sample_rate):.1f}s"
            )

        return output.replace("\\", "/")

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
            cache_dir: Cache directory for converted files; default
                :meth:`cache_dir_for` (the source's project, else the per-user
                cache).
            logger: Optional logger with ``warning``/``debug`` methods.

        Returns:
            Resolved playable path, or ``None`` on error.
        """
        source = audio_path.replace("\\", "/")

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
            cache_dir: Cache directory for converted files; default
                :meth:`cache_dir_for` per source (its project, else the
                per-user cache).
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

    # ------------------------------------------------------------------
    # Waveform envelope
    # ------------------------------------------------------------------

    @staticmethod
    def compute_waveform_envelope(
        wav_path: str,
        num_bins: int = 512,
    ) -> List[tuple]:
        """Read a WAV file and return a downsampled min/max envelope.

        Parameters:
            wav_path: Path to a PCM WAV file (8, 16, or 24-bit).
            num_bins: Number of (min, max) pairs to return.

        Returns:
            List of ``(min_sample, max_sample)`` tuples normalised to
            [-1.0, 1.0].  Empty list if the file cannot be read.
        """
        try:
            with _wave.open(wav_path, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()

                if sampwidth not in (1, 2, 3) or n_frames == 0:
                    return []

                raw = wf.readframes(n_frames)
        except Exception:
            return []

        # Decode samples (mono-mix if stereo)
        if sampwidth == 2:
            fmt = f"<{n_frames * n_channels}h"
            samples = struct.unpack(fmt, raw)
            scale = 1.0 / 32768.0
        elif sampwidth == 3:
            total = n_frames * n_channels
            samples = []
            for i in range(total):
                off = i * 3
                lo = raw[off]
                mid = raw[off + 1]
                hi = raw[off + 2]
                val = lo | (mid << 8) | (hi << 16)
                if val >= 0x800000:
                    val -= 0x1000000
                samples.append(val)
            scale = 1.0 / 8388608.0
        else:
            samples = [b - 128 for b in raw]
            scale = 1.0 / 128.0

        # Mono-mix
        if n_channels > 1:
            mono = []
            for i in range(0, len(samples), n_channels):
                mono.append(sum(samples[i : i + n_channels]) / n_channels)
            samples = mono
            n_frames = len(samples)

        if n_frames == 0:
            return []

        # Down-sample into bins
        bin_size = max(1, n_frames // num_bins)
        envelope = []
        for i in range(0, n_frames, bin_size):
            chunk = samples[i : i + bin_size]
            lo = min(chunk) * scale
            hi = max(chunk) * scale
            envelope.append((lo, hi))

        return envelope[:num_bins]
