# !/usr/bin/python
# coding=utf-8
"""Animation: say which clip is which in a shot-split deliverable, and ship
only the keys it needs.

Rebuilds the declared shot clips from the whole timeline, publishes
``extras.animation``, and compacts, reduces and prunes the samplers.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through the facade it holds (``self._mc``).
"""

import bisect
import logging
import struct
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Eager: ANIMATION_CLIP_MODES reads ExportProfile's options at class
# definition and the signatures name GlbTarget; SceneRecords is the light
# declaration module every pass reads.
from pythontk.core_utils.engines.scene_export.export_profile import ExportProfile
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget

logger = logging.getLogger(__name__)


class _Animation:
    """Animation: shot clips, the ``animation`` manifest, key compaction.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: The run's Animation Clips mode (``ExportProfile.ANIMATION_CLIPS_OPTIONS``:
    #: ``full`` / ``shots`` / ``both``), DECLARED by the exporter on the shots
    #: record beside ``fps``. The record lists the scene's shots in every
    #: mode, so only this says whether the file was meant to carry them as
    #: clips; one stack alone is also what a split that silently failed leaves.
    SHOT_CLIP_MODE_KEY = "clip_mode"

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    @staticmethod
    def _animation_span(gltf: dict, animation: dict) -> Optional[Tuple[float, float]]:
        """``(first, last)`` keyframe time of *animation*, in seconds, or ``None``.

        Read off the sampler INPUT accessors' ``min``/``max``, which glTF
        requires an animation sampler input to carry -- so this costs no buffer
        decode, and a file that omits them (not a legal one, but this is public
        API pointed at whatever it is handed) reports no span rather than a
        wrong one.
        """
        accessors = gltf.get("accessors") or []
        lo: Optional[float] = None
        hi: Optional[float] = None
        for sampler in animation.get("samplers") or []:
            index = sampler.get("input")
            if not isinstance(index, int) or not 0 <= index < len(accessors):
                continue
            accessor = accessors[index] or {}
            low, high = accessor.get("min"), accessor.get("max")
            # Shape-checked, not just truthiness: this is public API pointed at
            # whatever GLB it is handed, and an accessor whose min is a bare
            # number rather than the spec's array would raise out of a pass
            # whose whole contract is that it degrades to "no span".
            if not (isinstance(low, list) and isinstance(high, list)):
                continue
            if not low or not high:
                continue
            if not isinstance(low[0], (int, float)) or not isinstance(
                high[0], (int, float)
            ):
                continue
            lo = low[0] if lo is None else min(lo, low[0])
            hi = high[0] if hi is None else max(hi, high[0])
        return None if lo is None or hi is None else (float(lo), float(hi))

    @staticmethod
    def _numeric_pairs(keys: Any) -> List[Sequence[Any]]:
        """The ``[frame, value]`` entries of *keys* that are actually numbers.

        The tracks arrive as JSON decoded out of a string channel on a node in
        the deliverable, so their shape is whatever a producer wrote -- and a
        non-numeric entry would otherwise raise out of a pass whose whole
        contract is that a malformed channel degrades to "no tracks". Same rule
        :meth:`_animation_span` applies to accessor bounds, for the same reason.
        """
        out: List[Sequence[Any]] = []
        for key in keys if isinstance(keys, (list, tuple)) else ():
            if not isinstance(key, (list, tuple)) or len(key) < 2:
                continue
            # bool passes the int check, which is correct: a JSON ``true`` is a
            # legitimate way to write a visibility value.
            if isinstance(key[0], (int, float)) and isinstance(key[1], (int, float)):
                out.append(key)
        return out

    #: What a deliverable ships as its animation clips. ``both`` is the
    #: historical shape (the shots AND the whole-timeline stack they were cut
    #: from); the other two drop the half a given consumer never plays, which
    #: is the same performance either way -- measured on a production assembly,
    #: the whole-timeline stack alone was 66.52 MB of a 158.69 MB file.
    ANIMATION_CLIP_MODES = tuple(ExportProfile.ANIMATION_CLIPS_OPTIONS.values())

    def apply_glb_clips(
        self, glb: GlbTarget, *, mode: str = "both"
    ) -> Optional[Dict[str, Any]]:
        from pythontk.file_utils.mesh_convert.glb.clips import GlbClips

        if mode not in self._mc.ANIMATION_CLIP_MODES:
            raise ValueError(
                f"Unknown animation clip mode {mode!r}; expected one of "
                f"{', '.join(self._mc.ANIMATION_CLIP_MODES)}."
            )

        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            takes = self._declared_takes(gltf)
            if not takes:
                return None
            metadata = self._mc.data_export_channel(gltf, SceneRecords.SHOTS.key)
            channel = self._mc.data_export_channel(gltf, SceneRecords.VISIBILITY.key)
            fps = self._resolve_clip_fps(
                metadata if isinstance(metadata, dict) else {}, [], channel
            )
            if not fps:
                logger.warning(
                    "Clips: the declared takes are quoted in frames and the "
                    "file carries no frame rate, so they cannot be placed in "
                    "time -- clips left as exported."
                )
                return None

            windows = self._mc._take_windows(gltf)
            animations = gltf.get("animations") or []
            if not animations:
                # Loud, because the combination is wrong in BOTH deliverables at
                # once: the handoff names shots a consumer will plan for, and
                # the file cannot play any of them. Measured on a production
                # assembly: the FBX was written with bake/takes disarmed, so
                # the conversion arrived animationless and this pass -- the one
                # place that knows both halves -- said nothing.
                logger.warning(
                    "Clips: the scene declares %d take(s) but the file carries "
                    "no animations at all -- the FBX was written with animation "
                    "off (bake/takes disarmed at export), so there is nothing "
                    "to cut the shots from.",
                    len(takes),
                )
                return None
            index = GlbClips._source_stack(animations, list(windows))
            if index is None:
                return None
            source = animations[index]

            zero = GlbEdit.extras_of(source).get(GlbClips.ZERO_FRAME_KEY)
            if not isinstance(zero, (int, float)):
                span = self._mc._published_origin(
                    lambda key: self._mc.data_export_channel(gltf, key)
                )
                if span is None:
                    logger.warning(
                        "Clips: nothing says which authored frame %r puts at "
                        "t=0 (no %r span published), and the converter rebases "
                        "every stack onto its first key -- clips left as "
                        "exported.",
                        source.get("name"),
                        self._mc.DEFAULT_CLIP_SPAN,
                    )
                    return None
                zero = span[0]

            return GlbClips.rebuild(
                edit,
                takes,
                float(fps),
                float(zero),
                cut_shots=mode != "full",
                keep_sequence=mode != "shots",
            )

    #: ``clip_span`` entry describing the clip a take split does NOT name -- the
    #: converter's retained whole-timeline stack. Not a legal take name, so it
    #: cannot collide with one.
    DEFAULT_CLIP_SPAN = "*"

    def _published_origin(
        self, read: Callable[[str], Any]
    ) -> Optional[Tuple[float, float]]:
        """The first and last frame the whole-timeline stack carries, as the
        producer published them (the ``render_effects`` record's
        :attr:`DEFAULT_CLIP_SPAN` entry); ``None`` when nothing publishes one.

        The one reader of that number, for every party that needs it: the
        clip rebuild cuts the shots against it, the converter-input strip
        keeps the converter's split takes without it, and the verifier checks
        the shipped stack against it.

        Parameters:
            read: Channel key -> that channel's DECODED payload, ``None`` when
                absent -- the reader ``SceneRecords.declared_takes`` takes.
        """
        channel = read(SceneRecords.VISIBILITY.key)
        spans = channel.get("clip_span") if isinstance(channel, dict) else None
        span = (
            spans.get(self._mc.DEFAULT_CLIP_SPAN) if isinstance(spans, dict) else None
        )
        pair = self._numeric_pairs([span])
        return (float(pair[0][0]), float(pair[0][1])) if pair else None

    #: Frames a published clip span may sit from the file's own before
    #: :meth:`_stamp_clip_spans` says so: ticks convert to frames through the
    #: published rate, and an NTSC scene (29.97 against the file's 30) moves a
    #: frame by a hundredth.
    CLIP_SPAN_DRIFT_FRAMES = 0.5

    def _stamp_clip_spans(
        self, gltf: Dict[str, Any], measured: Dict[str, Tuple[float, float]]
    ) -> Dict[str, List[float]]:
        """Write each clip's MEASURED span onto the file's ``clip_span``.

        *measured* is ``FbxFile.take_spans`` of the FBX the converter read --
        each take's first and last key in seconds, which is exactly what
        FBX2glTF sizes the clip by and rebases it onto. A producer's
        published span predicts that number before the file exists, and the
        predictions drift: a hand-off published the range the last export
        left in the FBX plugin (shots 23 frames early), Blender the first
        authored key of a take it bakes over its whole window (a gate lost a
        shot's visible run) -- both measured 2026-10-04. So the file's own
        wins, ahead of every pass that reads the channel, and a disagreement
        is logged naming both.

        A declared take is stamped under its name; ``*`` from the one stack
        the file does not declare (the whole-timeline stack the shots are cut
        from) -- with two, which one the rebuild cuts from is not this
        stamp's to guess. The rest of the channel rides through, back into
        the carrier it was read from: another carrier's own channel (a
        referenced module brings its own) is left as it is. A file that
        publishes none gets the measurement alone.

        Returns:
            The entries stamped; empty when nothing was measured or no rate
            places the frames in time (the published spans then stand).
        """
        metadata = self._mc.data_export_channel(gltf, SceneRecords.SHOTS.key)
        channel = self._mc.data_export_channel(gltf, SceneRecords.VISIBILITY.key)
        fps = self._resolve_clip_fps(
            metadata if isinstance(metadata, dict) else {}, [], channel
        )
        if not fps or not measured:
            return {}
        windows = self._mc._take_windows(gltf)
        keys = {take: take for take in measured if take in windows}
        loose = [take for take in measured if take not in windows]
        if len(loose) == 1:
            keys[loose[0]] = self._mc.DEFAULT_CLIP_SPAN
        stamped = self._mc._visibility._rounded(
            {
                key: [measured[take][0] * fps, measured[take][1] * fps]
                for take, key in keys.items()
            },
            self._mc.VISIBILITY_TRACK_DIGITS,
        )
        if not stamped:
            return {}
        payload = (
            dict(channel)
            if isinstance(channel, dict)
            else {"version": SceneRecords.VISIBILITY.version, "tracks": [], "fps": fps}
        )
        published = payload.get("clip_span")
        published = published if isinstance(published, dict) else {}
        for key, (first, _last) in stamped.items():
            pair = self._numeric_pairs([published.get(key)])
            if (
                pair
                and abs(float(pair[0][0]) - first) > self._mc.CLIP_SPAN_DRIFT_FRAMES
            ):
                logger.warning(
                    "Clips: %r was published as opening at frame %g, but its "
                    "take in the FBX opens at %g -- placed by the file.",
                    key,
                    pair[0][0],
                    first,
                )
        payload["clip_span"] = {**published, **stamped}
        # Not through overlay_data_export, which clears the channel from EVERY
        # carrier -- right for a caller's statement about this build, but this
        # read one carrier, and on every conversion a module's tracks left the
        # GLB with the others.
        found = self._mc._sidecar._data_export_carrier(
            gltf, SceneRecords.VISIBILITY.key
        )
        if found is None:
            self._mc.overlay_data_export(gltf, {SceneRecords.VISIBILITY.key: payload})
        else:
            self._mc._sidecar._write_data_export(
                found[0], SceneRecords.VISIBILITY.key, payload
            )
        return stamped

    #: What Maya (and so FBX2glTF) names the whole-timeline AnimStack. The
    #: clip :meth:`_synthesize_clips` makes for a file that carries none is
    #: named the same, so a consumer sees the clip it would have seen had the
    #: exporter armed animation.
    DEFAULT_CLIP_NAME = "Take 001"

    def _synthesize_clips(
        self,
        gltf: Dict[str, Any],
        windows: Dict[str, Tuple[float, float]],
        union: Optional[Tuple[float, float]],
    ) -> List[Dict[str, Any]]:
        """Give a clip-less file the clips its authored tracks need to ride on.

        A GLB converted from an FBX exported with animation OFF carries no
        animation at all (Blender always; Maya when the hollow ``Take 001``
        stack is absent), yet its ``data_export`` carrier still declares
        visibility, opacity and highlight tracks -- the WebXR preview's default
        export is exactly this file, and a highlight-only prop previewed with
        no highlight (2026-09-05). One empty clip per declared take, else one
        default clip over the tracks' own extent. No origin is declared: the
        passes infer it through :meth:`_clip_zero` from the published
        ``clip_span``, exactly as they do for the converter's own hollow
        ``Take 001`` -- so the same scene places its keys identically whether
        or not the exporter armed animation. Empty when there is nothing to
        span. :meth:`prune_glb_animations` drops any clip that stays hollow.
        """
        if windows:
            spans = list(windows.items())
        elif union is not None:
            spans = [(self._mc.DEFAULT_CLIP_NAME, union)]
        else:
            return []
        clips = [{"name": name, "samplers": [], "channels": []} for name, _ in spans]
        gltf["animations"] = clips
        logger.info(
            "No clip in the file: %d synthesized for its authored tracks (%s).",
            len(clips),
            ", ".join(c["name"] for c in clips),
        )
        return clips

    def clip_spans(
        self,
        frames: Iterable[float],
        takes: Iterable[Any],
        stack_range: Optional[Sequence[float]] = None,
        key_spans: Optional[
            Callable[
                [List[Tuple[Optional[float], Optional[float]]]],
                Sequence[Optional[Sequence[float]]],
            ]
        ] = None,
    ) -> Dict[str, List[float]]:
        named: List[Tuple[str, float, float]] = []
        for take in takes or ():
            if not isinstance(take, dict):
                continue
            try:
                named.append(
                    (str(take["name"]), float(take["start"]), float(take["end"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
        windows: List[Tuple[Optional[float], Optional[float]]] = [(None, None)]
        windows += [(start, end) for _name, start, end in named]
        if key_spans is not None:
            found = list(key_spans(windows))
        else:
            # Bisect the sorted key times rather than filter them: a
            # production bake is millions of keys, and a scan per take was
            # 23 s of every export (measured on an 18-take assembly).
            every = sorted(float(f) for f in frames if isinstance(f, (int, float)))
            found = []
            for start, end in windows:
                lo = 0 if start is None else bisect.bisect_left(every, start)
                hi = len(every) if end is None else bisect.bisect_right(every, end)
                found.append((every[lo], every[hi - 1]) if lo < hi else None)
        bounds = None
        if stack_range is not None:
            try:
                bounds = [float(stack_range[0]), float(stack_range[1])]
            except (TypeError, ValueError, IndexError):
                bounds = None
        whole = found[0] if found else None
        if whole is None:
            return {self._mc.DEFAULT_CLIP_SPAN: bounds} if bounds else {}
        spans: Dict[str, List[float]] = {
            self._mc.DEFAULT_CLIP_SPAN: bounds or [float(whole[0]), float(whole[1])]
        }
        for (name, _start, _end), hit in zip(named, found[1:]):
            if hit is not None:
                spans[name] = [float(hit[0]), float(hit[1])]
        return spans

    def _declared_takes(self, gltf: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The take list *gltf*'s carrier declares
        (:meth:`SceneRecords.declared_takes` over its records): every GLB
        reader of the takes goes through here."""
        return SceneRecords.declared_takes(
            lambda key: self._mc.data_export_channel(gltf, key)
        )

    def _take_windows(self, gltf: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """``{take name: (start frame, end frame)}`` from the declared takes.

        Shared by the visibility gate and the clip manifest so a take whose
        bounds one of them cannot read is skipped by BOTH -- the alternative
        being a clip that publishes an origin the gates were never placed
        against.
        """
        windows: Dict[str, Tuple[float, float]] = {}
        for take in self._declared_takes(gltf):
            if take.get("name") is None:
                continue
            try:
                windows[str(take["name"])] = (
                    float(take.get("start")),
                    float(take.get("end")),
                )
            except (TypeError, ValueError):
                continue
        return windows

    def _clip_span_for(
        self,
        name: str,
        spans: Dict[str, Any],
        windows: Dict[str, Any],
    ) -> Optional[Sequence[float]]:
        """The authored span that applies to the clip called *name*.

        One rule, in one place, because both readers of it place a clip against
        the frame it returns: the visibility gate and the manifest's
        ``zero_frame``. Two copies that disagree would put a node's switch and
        the clip's own published origin at different instants in the same file.

        :attr:`DEFAULT_CLIP_SPAN` describes the converter's retained
        whole-timeline stack ONLY. Letting a DECLARED take fall back to it
        would place that take against the whole timeline's zero -- measured at
        42.5s of drift on a shot 42.5s into the sequence.
        """
        span = spans.get(name)
        if span is None and name not in windows:
            span = spans.get(self._mc.DEFAULT_CLIP_SPAN)
        return span

    def _clip_zero(
        self,
        gltf: Dict[str, Any],
        animation: Dict[str, Any],
        window: Tuple[float, float],
        span: Optional[Sequence[float]],
        fps: float,
        verify: bool = True,
    ) -> float:
        """The authored frame the converter placed at this clip's ``t=0``.

        Not the take's start frame, which is the intuitive answer and the wrong
        one. Measured against FBX2glTF 0.13.1: a take is emitted spanning its
        authored keys, not its declared window, and rebased so the FIRST of
        them lands at zero -- counting the VISIBILITY keys, which size the take
        even though no channel is emitted for them. A gate placed against the
        window start therefore drifts by the take's lead-in, which on a
        production assembly ran to 43 frames.

        The producer publishes that span (it is the only party that can see the
        curves), and this VERIFIES it against the clip actually in the file:
        the two agree when the producer's view of the export set matched the
        converter's. A clip with no channels has nothing to check against and
        nothing to be misaligned with -- its zero is whatever this returns.

        Parameters:
            verify: Compare the published span against the clip in the file.
                Only meaningful BEFORE the gate pass writes, because a gate
                holds its final state to the end of the shot's window and so
                legitimately makes the clip longer than the keys the producer
                measured. A caller reading the zero back out of a finished file
                (the clip manifest) would otherwise report that growth as a
                mismatch on every clip it succeeded on.
        """
        # A clip built by :meth:`apply_glb_clips` was cut to its window, so it
        # KNOWS its origin and says so. Everything below is the inference for a
        # clip that came straight off the converter, where nobody does.
        from pythontk.file_utils.mesh_convert.glb.clips import GlbClips

        declared = GlbEdit.extras_of(animation).get(GlbClips.ZERO_FRAME_KEY)
        if isinstance(declared, (int, float)):
            return float(declared)

        pair = self._numeric_pairs([span])
        if not pair:
            return window[0]
        first, last = float(pair[0][0]), float(pair[0][1])
        if not verify:
            return first
        measured = self._animation_span(gltf, animation)
        if measured is None:
            return first  # empty clip: this pass alone defines its zero
        # One frame of tolerance: the span arrives as authored frames and comes
        # back as seconds through the converter's own rounding.
        if abs((last - first) - measured[1] * fps) > 1.0:
            logger.warning(
                "Visibility: clip %r spans %.3fs in the file but the scene "
                "reports frames %g-%g (%.3fs) -- the export set the gate was "
                "computed against is not the one that shipped, so its switches "
                "may sit up to %.2fs off.",
                animation.get("name"),
                measured[1],
                first,
                last,
                (last - first) / fps,
                abs((last - first) / fps - measured[1]),
            )
        return first

    def prune_glb_animations(self, glb: GlbTarget) -> List[str]:
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            animations = gltf.get("animations")
            if not animations:
                return []
            kept: List[Dict[str, Any]] = []
            dropped: List[str] = []
            for index, animation in enumerate(animations):
                if animation.get("channels") and animation.get("samplers"):
                    kept.append(animation)
                else:
                    dropped.append(str(animation.get("name") or f"animations[{index}]"))
            if not dropped:
                return []
            if kept:
                gltf["animations"] = kept
            else:
                # ``animations`` has ``minItems: 1``: an empty array is as
                # invalid as the hollow entry was.
                del gltf["animations"]
            edit.dirty = True

            declared = self._mc._take_windows(gltf)
            shots = [name for name in dropped if name in declared]
            if shots:
                # Warned: the scene declared these and the file will not list
                # them, which the manifest pass reports next as takes it lacks.
                logger.warning(
                    "Animation: dropped %d declared take(s) with no keyframes "
                    "(%s) -- glTF cannot carry an empty clip.",
                    len(shots),
                    ", ".join(shots),
                )
            rest = [name for name in dropped if name not in declared]
            if rest:
                logger.info(
                    "Animation: dropped %d empty animation(s) (%s) -- nothing "
                    "in the exported set is keyed there (Maya writes its "
                    "whole-timeline stack whether or not anything is).",
                    len(rest),
                    ", ".join(rest),
                )
            return dropped

    def compact_glb_animations(self, glb: GlbTarget) -> Dict[str, int]:
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            animations = gltf.get("animations") or []
            accessors = gltf.get("accessors") or []
            if not animations or not accessors:
                return {"channels": 0, "accessors": 0, "bytes": 0}

            # Pass 1 -- classify. An output accessor is collapsible only when
            # EVERY channel reading it is constant: samplers are shared, and
            # shrinking one that something still animates through would freeze
            # that motion.
            constant: Dict[int, bytes] = {}
            refused: set = set()
            per_animation: List[List[tuple]] = []
            for animation in animations:
                samplers = animation.get("samplers") or []
                rows = []
                for channel in animation.get("channels") or []:
                    index = channel.get("sampler")
                    if not isinstance(index, int) or not 0 <= index < len(samplers):
                        continue
                    sampler = samplers[index] or {}
                    output = sampler.get("output")
                    if not isinstance(output, int):
                        continue
                    if (sampler.get("interpolation") or "LINEAR") == "CUBICSPLINE":
                        # Three elements per key -- in-tangent, value,
                        # out-tangent -- so two keys is SIX elements, not two.
                        # Collapsing one would write a sampler whose output
                        # count no longer matches its input, which is a
                        # malformed file rather than a smaller one.
                        refused.add(output)
                        continue
                    if output in refused:
                        continue
                    if output not in constant:
                        elements = self._mc._accessor_elements(edit, output)
                        if not elements or len(elements) < 3:
                            # Nothing to win on a channel already at two keys,
                            # and an unreadable layout is left alone.
                            refused.add(output)
                            continue
                        first = elements[0]
                        if any(e != first for e in elements):
                            refused.add(output)
                            continue
                        constant[output] = first
                    rows.append((index, output))
                per_animation.append(rows)

            # Constancy is a property of the ACCESSOR, so every channel reading
            # one agrees about it -- but a refusal is a property of the SAMPLER
            # (interpolation), so one clip can refuse an output another clip
            # already accepted. The survivors are what both agree on.
            collapsible = {k: v for k, v in constant.items() if k not in refused}
            if not collapsible:
                return {"channels": 0, "accessors": 0, "bytes": 0}

            # Pass 2 -- one shared two-key time input per animation, spanning
            # the clip's own range so its duration is unchanged, plus one
            # two-key value block per collapsible output.
            payloads: List[bytes] = []
            input_slot: Dict[int, int] = {}
            for position, animation in enumerate(animations):
                # Against `collapsible`, not against the rows: an output one
                # clip accepted can still be refused by a LATER clip's
                # interpolation, and a clip left with nothing to collapse must
                # not mint a time input no sampler will read -- an accessor
                # keeps its bufferView alive, so that orphan survives the
                # collector and the next run mints another.
                if not any(out in collapsible for _s, out in per_animation[position]):
                    continue
                lo, hi = self._mc._animation_time_span(edit, animation)
                if lo is None:
                    continue
                input_slot[position] = len(payloads)
                payloads.append(struct.pack("<2f", lo, hi))
            output_slot: Dict[int, int] = {}
            for output, element in collapsible.items():
                output_slot[output] = len(payloads)
                payloads.append(element + element)

            added_views = self._mc._append_bin_views(edit, payloads)
            if not added_views:  # external buffer: nothing this pass can do
                return {"channels": 0, "accessors": 0, "bytes": 0}

            # Pass 3 -- mint the accessors and rewire the samplers.
            new_input: Dict[int, int] = {}
            for position, slot in input_slot.items():
                # Read back OUT of the packed bytes, not from the float64 the
                # span was measured as: `min` must not exceed the value the
                # file actually stores, and packing to float32 can round it up.
                lo, hi = struct.unpack("<2f", payloads[slot])
                accessors.append(
                    {
                        "bufferView": added_views[slot],
                        "componentType": 5126,
                        "count": 2,
                        "type": "SCALAR",
                        # Required on an animation input by the spec, and the
                        # only place a reader learns the clip's length.
                        "min": [lo],
                        "max": [hi],
                    }
                )
                new_input[position] = len(accessors) - 1
            # The outputs are re-pointed IN PLACE rather than appended beside
            # the originals. An accessor keeps its bufferView alive whether or
            # not anything still reads the accessor, so minting replacements
            # and abandoning the old ones collects nothing -- measured: 13,187
            # channels rewired, 0.00 MB reclaimed. Editing them is also the
            # truthful shape: this is the same channel, shorter. Safe because
            # `collapsible` already excludes every output any channel refused,
            # and a mesh never reads an animation accessor.
            for output, slot in output_slot.items():
                accessor = accessors[output]
                accessor["bufferView"] = added_views[slot]
                # The new view starts at the data: an offset INTO the old view
                # (a packer sharing one view between accessors) would read
                # past the two keys it was just given.
                accessor.pop("byteOffset", None)
                accessor["count"] = 2
                accessor.pop("min", None)
                accessor.pop("max", None)

            channels = 0
            replaced_inputs: Set[int] = set()
            for position, animation in enumerate(animations):
                if position not in new_input:
                    continue
                samplers = animation.get("samplers") or []
                for sampler_index, output in per_animation[position]:
                    if output not in output_slot:
                        continue
                    sampler = samplers[sampler_index]
                    replaced_inputs.add(sampler.get("input"))
                    sampler["input"] = new_input[position]
                    channels += 1

            edit.dirty = True
            # A converter shares one time input across a clip, which its moving
            # channels keep alive; a clip rebuild writes one PER CHANNEL, and
            # each collapsed channel's would stay behind as a dead accessor.
            self._mc._drop_orphaned_accessors(edit, replaced_inputs)
            reclaimed = self._mc._compact_bin(edit)
            logger.info(
                "Animation: collapsed %d constant channel(s) across %d clip(s) "
                "to two keys, reclaiming %.2f MB -- a clip PINS the pose of "
                "what it does not animate, which two keys say as well as "
                "thousands.",
                channels,
                len(animations),
                reclaimed / 1048576.0,
            )
            return {
                "channels": channels,
                "accessors": len(output_slot),
                "bytes": reclaimed,
            }

    def reduce_glb_animations(
        self,
        glb: GlbTarget,
        tolerance: float,
        rotation_tolerance: Optional[float] = None,
    ) -> Dict[str, int]:
        from pythontk.file_utils.mesh_convert.glb.key_reduction import (
            GlbKeyReduction,
        )

        return GlbKeyReduction.reduce(glb, tolerance, rotation_tolerance)

    def _animation_time_span(self, edit: "GlbEdit", animation: Dict[str, Any]) -> tuple:
        """``(first, last)`` input time over every sampler, or ``(None, None)``.

        Read from each input accessor's ``min``/``max``, which the spec
        requires an animation input to carry, and decoded only when a writer
        left them off.
        """
        lo = hi = None
        accessors = edit.gltf.get("accessors") or []
        for sampler in animation.get("samplers") or []:
            index = (sampler or {}).get("input")
            if not isinstance(index, int) or not 0 <= index < len(accessors):
                continue
            accessor = accessors[index] or {}
            low, high = accessor.get("min"), accessor.get("max")
            if isinstance(low, list) and low and isinstance(high, list) and high:
                start, end = float(low[0]), float(high[0])
            else:
                elements = self._mc._accessor_elements(edit, index)
                if not elements:
                    continue
                start = struct.unpack("<f", elements[0])[0]
                end = struct.unpack("<f", elements[-1])[0]
            lo = start if lo is None else min(lo, start)
            hi = end if hi is None else max(hi, end)
        return lo, hi

    def apply_glb_animations(self, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            animations = gltf.get("animations") or []
            metadata = self._mc.data_export_channel(gltf, SceneRecords.SHOTS.key) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            # The record's own entries, as it states them: the block's shape
            # prunes anything it does not declare.
            shots = [
                dict(s)
                for s in (metadata.get("shots") or [])
                if isinstance(s, dict) and s.get("clip") is not None
            ]
            if not animations and not shots:
                return None
            by_clip = {str(s["clip"]): s for s in shots}

            clips: List[Dict[str, Any]] = []
            #: Clip names carrying no channels: what the default clip skips.
            hollow_names = set()
            #: Each clip's measured span in seconds, for deriving the rate when
            #: no record states it (:meth:`_resolve_clip_fps`); never in the
            #: block, whose reader measures the clip itself.
            durations: Dict[str, float] = {}
            for index, animation in enumerate(animations):
                name = animation.get("name") or f"animation_{index}"
                clips.append({"name": name, "animation": index})
                span = self._animation_span(gltf, animation)
                if span:
                    durations[name] = round(span[1] - span[0], 6)
                if not (animation.get("channels") or []):
                    # Named, listed, and carrying nothing. Maya's split emits an
                    # AnimStack per declared range but bakes no curve for a range
                    # in which nothing moves (a hold, or a shot whose motion
                    # belongs to objects outside the export), so this is a normal
                    # authoring outcome rather than a conversion failure. In the
                    # conversion chain :meth:`prune_glb_animations` has already
                    # removed these (glTF forbids them); this is the path for a
                    # file handed in directly, or a prune that was skipped.
                    # Not stated in the block: a reader sees it in the file.
                    hollow_names.add(name)

            # The fourth resolution in the same session: this one derived from
            # clip spans but never read the visibility channel, so a file whose
            # only stated rate lives there fell through to derivation while the
            # other passes used the stated number.
            channel = self._mc.data_export_channel(gltf, SceneRecords.VISIBILITY.key)
            measured = [
                {
                    "start_frame": by_clip[c["name"]].get("start"),
                    "end_frame": by_clip[c["name"]].get("end"),
                    "duration": durations[c["name"]],
                }
                for c in clips
                if c["name"] in by_clip and c["name"] in durations
            ]
            fps = self._resolve_clip_fps(metadata, measured, channel)
            if fps and clips:
                windows = self._mc._take_windows(gltf)
                union = (
                    (
                        min(s for s, _ in windows.values()),
                        max(e for _, e in windows.values()),
                    )
                    if windows
                    else None
                )
                spans = (channel or {}).get("clip_span")
                if not isinstance(spans, dict):
                    spans = {}
                for index, clip in enumerate(clips):
                    window = windows.get(clip["name"]) or union
                    span = self._clip_span_for(clip["name"], spans, windows)
                    if window is None and span is None:
                        # Neither a declared window nor a published span: there
                        # is nothing to place this clip against, and a guessed
                        # origin is worse than an absent one.
                        continue
                    # NOT the shot's start, and that is the whole point: the
                    # converter rebases a take to its first authored KEY, not
                    # to its declared window, so t=0 sits at the take's lead-in
                    # -- measured at 43 frames (1.43s) into Shot_5 of a
                    # production assembly. Every frame number published
                    # elsewhere in this block is an AUTHORING frame, so this is
                    # the one value that lets a consumer convert between the
                    # two:
                    #
                    #     authoring_frame = zero_frame + clip_time * fps
                    #
                    # Falls back to the window start when the scene published
                    # no spans -- the best available answer, and the one a
                    # consumer would have assumed anyway.
                    clip["zero_frame"] = round(
                        self._clip_zero(
                            gltf,
                            animations[index],
                            # ``_clip_zero`` reads the window ONLY when there is
                            # no span, and the guard above has ruled out both
                            # being absent -- so this placeholder can never be
                            # the answer it returns.
                            window or (0.0, 0.0),
                            span,
                            fps,
                            # The gate pass already checked this span against
                            # the file, and has since extended these clips to
                            # their window ends; re-checking here would report
                            # its own tail-hold as a misalignment on every clip.
                            verify=False,
                        ),
                        6,
                    )

            manifest: Dict[str, Any] = {
                "version": SceneRecords.SHOTS.glb.version,
                "shots": shots,
                "clips": clips,
            }
            # The clip to open on: the first SHOT's clip that can actually
            # PLAY, because a deliverable that went to the trouble of
            # splitting shots means the shots -- but opening on an empty one
            # shows 0.00s of nothing and reads as broken animation rather than
            # as a shot that holds. Measured on a production assembly whose
            # Shot_1 was a hold: the preview opened dead with 11 populated
            # clips behind it. Each fallback is narrower than the last, and
            # the final one is unconditional: a file where every clip is empty
            # still gets a default, since naming no clip at all is worse than
            # naming a quiet one.
            shot_clips = [c for c in clips if c["name"] in by_clip]
            playable = [c for c in shot_clips if c["name"] not in hollow_names] or [
                c for c in clips if c["name"] not in hollow_names
            ]
            opening = playable or shot_clips or clips
            if opening:
                manifest["default_clip"] = opening[0]["name"]
            if fps:
                manifest["fps"] = fps
            if metadata.get("clip_mode"):
                manifest["clip_mode"] = metadata["clip_mode"]

            hollow = [c["name"] for c in shot_clips if c["name"] in hollow_names]
            if hollow:
                # Not an error -- a declared range can legitimately hold still --
                # but the one thing a reviewer needs told, because the clip is
                # in the list, in the player's dropdown, and plays as nothing.
                logger.info(
                    "Animation: %d declared shot(s) carry no keyframes (%s); "
                    "listed, and not opened on.",
                    len(hollow),
                    ", ".join(hollow),
                )

            # A 'full' file cuts no shot by design: each plays its window of
            # the whole-timeline clip, which the block states.
            uncut = sorted(set(by_clip) - {c["name"] for c in clips})
            if uncut and metadata.get("clip_mode") != "full":
                # Loud, like every lightmap miss: the takes were declared and
                # the file does not carry them, which means the split did not
                # run (the exporter's task is off) or the names disagree. The
                # block still states each shot, so a player can play it as its
                # window of the whole-timeline clip -- but a consumer that
                # plays a named shot's own clip will not find one.
                #
                # The CONSEQUENCE is spelled out because the bare fact reads as
                # bookkeeping: measured on a production assembly, the WebXR
                # preview of the same scene carried 12 named shots and this
                # deliverable carried one continuous clip -- the reviewer's
                # only clue that they were not looking at the same thing.
                logger.warning(
                    "Animation: %d declared take(s) have no clip in the GLB "
                    "(%s) -- the take split did not reach this file, or the "
                    "take held no keyframes and was pruned, so it "
                    "carries %d clip(s) where the scene declares %d shot(s). "
                    "Each still plays as its window of the whole-timeline "
                    "clip; a consumer that plays a named shot's own clip will "
                    "not find one.",
                    len(uncut),
                    ", ".join(uncut),
                    len(clips),
                    len(by_clip),
                )

            edit.dirty = True
            return SceneRecords.put_glb_block(gltf, SceneRecords.SHOTS, manifest)

    @staticmethod
    def _resolve_clip_fps(
        metadata: Dict[str, Any],
        clips: List[Dict[str, Any]],
        channel: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """The rate the frame numbers were authored at, or ``None``.

        One rule for the whole conversion, in this order:

        1. ``animation.fps`` -- published by the shot system, which owns
           the frame numbers every other record quotes.
        2. ``render_effects.fps`` -- the producer's own rate, stated only
           by files whose shot system did not publish one.
        3. Derived from the first declared clip carrying both a frame range
           and a measured span: the two describe the same interval, one in
           frames and one in seconds, so their ratio IS the rate. Derivation
           needs at least two frames; a single-frame take spans zero seconds
           and would divide by nothing.

        The order is the point. One ``fbx_to_glb`` session ran four passes
        that resolved this number three different ways -- two carried
        byte-identical inline copies of (1) then (2), a third did (1) then (3)
        and never looked at the channel, and the visibility pass did (2) then
        (1), inverted. The channels are written by different producers and can
        disagree, so the passes placed their keys at times that did not match
        each other, in a converter whose output ships.

        Parameters:
            metadata: The shots record (``animation``), or ``{}``.
            clips: Declared clips, for derivation; ``[]`` to skip it.
            channel: The ``render_effects`` record, when the caller has it.
        """
        published = metadata.get("fps")
        if isinstance(published, (int, float)) and published > 0:
            return round(float(published), 6)
        if isinstance(channel, dict):
            try:
                stated = float(channel.get("fps") or 0.0)
            except (TypeError, ValueError):
                stated = 0.0
            if stated > 0:
                return round(stated, 6)
        for clip in clips:
            start, end = clip.get("start_frame"), clip.get("end_frame")
            duration = clip.get("duration")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            if not duration or end <= start:
                continue
            return round((end - start) / duration, 6)
        return None
