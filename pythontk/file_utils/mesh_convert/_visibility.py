# !/usr/bin/python
# coding=utf-8
"""Keyed visibility and authored material ramps, realized as channels the GLB
can play.

Reads the ``render_effects`` record the DCC publishes, writes on/off
presence as STEP ``scale`` channels and fades/highlights as
``KHR_animation_pointer`` material channels (:class:`GlbFades`), and builds
the preview channels one render effect needs.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through the facade it holds (``self._mc``).
"""

import logging
import struct
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Eager: the channel keys below are SceneRecords' own, read at class
# definition.
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget

logger = logging.getLogger(__name__)


class _Visibility:
    """Visibility tracks, authored fades/highlights, render-effect previews.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    def effect_preview_channels(
        self,
        nodes: Sequence[str],
        channel: str,
        keys: Sequence[Sequence[float]],
        colors: Optional[Sequence[Optional[Sequence[float]]]] = None,
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        from pythontk.file_utils.mesh_convert.glb.fades import CHANNELS

        if channel not in CHANNELS:
            raise KeyError(
                f"Unknown render-effect channel {channel!r}; expected one of "
                f"{', '.join(CHANNELS)}."
            )
        names = list(dict.fromkeys(str(n) for n in nodes or () if n))
        if not names:
            raise ValueError("An effect preview needs at least one node.")
        ramp = [[float(frame), float(value)] for frame, value in keys]
        if len(ramp) < 2:
            raise ValueError("An effect preview needs a ramp of at least two keys.")
        spec = CHANNELS[channel]

        tracks = []
        for name in names:
            # No visibility beside an opacity ramp: the ramp IS the presence
            # channel, and the gate reads it directly (_presence_keys).
            track: Dict[str, Any] = {"node": name, channel: ramp}
            if spec.color_stops is not None:
                for key, rgb in zip(spec.color_stops.keys, colors or ()):
                    if rgb is not None:
                        track[key] = [float(c) for c in list(rgb)[:3]]
            tracks.append(track)
        return {
            SceneRecords.VISIBILITY.key: self._mc.build_visibility_tracks(
                tracks,
                fps=fps,
                clip_spans={self._mc.DEFAULT_CLIP_SPAN: [ramp[0][0], ramp[-1][0]]},
            ),
            SceneRecords.SHOTS.key: None,
        }

    def _presence_keys(self, track: Dict[str, Any]) -> List[Sequence[float]]:
        """The on/off timeline to GATE this track on: the one definition of presence.

        A DCC that keys opacity as a custom attribute (mayatk's ``RenderOpacity``
        in its recommended "attribute" mode is one) drives nothing with it: the
        attribute is metadata for the engine downstream, and the viewport shows
        only the boolean ``visibility`` mirrored from it. That mirror is
        ``opacity > 0``, evaluated at the KEYS -- so a fade-in keyed 0 at frame
        8 and 1 at frame 23 mirrors to visibility 0 at 8 and 1 at 23, and a
        stepped boolean holds 0 across the whole ramp. Gate on that and the
        object is ABSENT for the entire fade-in and fully opaque for the entire
        fade-out: the fade cannot be seen, in Maya or in anything reading the
        mirror, which is why an authored fade has always arrived as a pop.

        So a track carrying a real ramp is gated on the ramp instead: present
        wherever the alpha is, or is about to become, non-zero. The object is
        then there to BE faded, and the ``KHR_animation_pointer`` channels
        :meth:`apply_glb_fades` writes supply the alpha.

        A track whose "ramp" only ever holds 0 or 1 is not a fade and keeps its
        authored ``visibility`` exactly as before -- same predicate as
        :meth:`_authored_fades`, so the two cannot disagree about what a fade
        is. With no visibility beside it (a hand-keyed opacity nothing
        mirrored), the opacity keys ARE the presence channel, read by the rule
        every mirror writer applies: ``> 0`` is present. A track carrying
        neither -- a highlight alone -- is not gated.

        Presence is therefore a property of the authored channels alone: no
        producer has to write a mirror, or run before anything, for a node to
        be gated where it is present.
        """
        keys = self._mc._animation._numeric_pairs(track.get("opacity") or [])
        pairs = [(float(k[0]), float(k[1])) for k in keys]
        if len(pairs) < 2 or not self._mc._is_fade(pairs):
            authored = track.get("visibility")
            if authored:
                return authored
            return [[t, 1.0 if v > 0.0 else 0.0] for t, v in pairs]
        runs: List[Sequence[float]] = []
        for (t0, v0), (_t1, v1) in zip(pairs, pairs[1:]):
            # A segment is PRESENT when either end is non-zero: that covers the
            # ramp out of zero (which is the frame the object has to appear on)
            # as well as every fully-visible stretch.
            runs.append([t0, 1.0 if (v0 > 0.0 or v1 > 0.0) else 0.0])
        runs.append([pairs[-1][0], 1.0 if pairs[-1][1] > 0.0 else 0.0])
        if runs[0][1] and pairs[0][1] <= 0.0:
            # BEFORE the ramp begins the object is not there, and a stepped
            # track is read by holding its first key BACKWARDS -- so a rising
            # ramp whose first key says "present" would make the object present
            # for the whole file up to it. Measured: seven nodes turned up in
            # every shot before the one they fade into. The extra key sits on
            # the same frame and loses to it (ties resolve to the LAST value),
            # so it changes nothing from the ramp onward and everything before.
            runs.insert(0, [pairs[0][0], 0.0])
        return runs

    @staticmethod
    def _is_fade(pairs: Sequence[Sequence[float]]) -> bool:
        """Does this ramp actually fade, or is it a boolean in float clothing?

        Either an intermediate alpha exists, or two keys differ in BOTH time
        and value -- an endpoint-only ramp still fades when its endpoints
        straddle time. What makes one a STEP is both keys landing on the same
        frame.
        """
        if any(0.0 < value < 1.0 for _t, value in pairs):
            return True
        return any(a[0] != b[0] and a[1] != b[1] for a, b in zip(pairs, pairs[1:]))

    def _visibility_runs(
        self, keys: Sequence[Sequence[float]], window: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """Sample a stepped on/off track across *window*, as ``(frame, value)``.

        Three rules, and the last is the one that matters most:

        * The value at the window's first frame is the one HELD there -- the
          last key at or before it, or the first key's value when the window
          opens before the track does (a DCC holds the first key backwards).
        * Keys strictly inside the window follow, in order.
        * Consecutive equal values collapse, so a track that never switches
          inside a clip costs one key rather than one per key authored.

        The held-value rule is what stops an object switched off in shot 5 from
        reappearing in shot 7: shot 7's window contains no key at all, and the
        naive reading of "no keys here" is "nothing to write", which leaves the
        node at its full authored scale for the whole clip.

        Keys sharing a frame keep their authored order, so the later one is the
        state from that frame on -- a one-frame cut, whichever way it switches
        (the tie rule :meth:`_strictly_increasing` applies too). Sorting on the
        value as well resolved every such tie to "visible".
        """
        ordered = sorted(
            (
                (float(k[0]), float(k[1]))
                for k in self._mc._animation._numeric_pairs(keys)
            ),
            key=lambda key: key[0],
        )
        if not ordered:
            return []
        start, end = window
        held = ordered[0][1]
        for time, value in ordered:
            if time <= start:
                held = value
            else:
                break
        samples = [(start, held)]
        samples.extend((t, v) for t, v in ordered if start < t <= end)
        runs: List[Tuple[float, float]] = []
        for time, value in samples:
            if not runs or runs[-1][1] != value:
                runs.append((time, value))
        return runs

    @staticmethod
    def _strictly_increasing(
        samples: Sequence[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Collapse *samples* onto strictly increasing times, then onto runs.

        glTF requires an animation sampler's input times to strictly increase,
        and clamping negative times to zero (a key authored before the clip's
        own zero) is exactly what produces a tie. The LAST value at a repeated
        time wins, because it is the later state; the run-collapse then drops
        any key that changes nothing.
        """
        merged: List[Tuple[float, float]] = []
        for time, value in samples:
            if merged and merged[-1][0] >= time:
                merged[-1] = (merged[-1][0], value)
            else:
                merged.append((time, value))
        runs: List[Tuple[float, float]] = []
        for time, value in merged:
            if not runs or runs[-1][1] != value:
                runs.append((time, value))
        return runs

    def apply_glb_visibility(self, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            channel = self._mc._current_record(gltf, SceneRecords.VISIBILITY)
            if channel is None:
                return None
            # Each track paired with what it is GATED on, resolved once: the one
            # definition of presence (:meth:`_presence_keys`). A track qualifies
            # by what its authored channels say, never by whether some producer
            # wrote a visibility mirror beside them first.
            to_gate = []
            for t in channel.get("tracks") or []:
                if isinstance(t, dict) and t.get("node"):
                    presence = self._mc._presence_keys(t)
                    if presence:
                        to_gate.append((t, presence))
            if not to_gate:
                return None

            metadata = self._mc.data_export_channel(gltf, SceneRecords.SHOTS.key)
            # Through the shared resolver, in the session's one order. This
            # read the channel FIRST and fell back to the shots record, so when
            # the two producers disagreed the visibility keys landed at times
            # no other pass in the same conversion agreed with.
            fps = (
                self._mc._animation._resolve_clip_fps(
                    metadata if isinstance(metadata, dict) else {}, [], channel
                )
                or 0.0
            )
            if fps <= 0:
                # Every key here is a FRAME number, and without the rate it was
                # authored at there is no time to place it at. Refusing beats
                # guessing 24 and stepping every gate at the wrong moment.
                logger.warning(
                    "Visibility: %d track(s) carry no frame rate, so their frame "
                    "numbers cannot be placed in time -- visibility not applied.",
                    len(to_gate),
                )
                return None

            windows = self._mc._take_windows(gltf)
            # A clip that is not a declared take -- the exporter's retained
            # whole-timeline stack -- spans every take, which is the range the
            # bake covered. With no takes at all (an animated prop that was
            # never cut into shots) the tracks' own extent is the only window
            # there is, and it is the right one: gating still has to happen.
            if windows:
                union = (
                    min(s for s, _ in windows.values()),
                    max(e for _, e in windows.values()),
                )
            else:
                frames = [
                    float(key[0])
                    for _track, presence in to_gate
                    for key in self._mc._animation._numeric_pairs(presence)
                ]
                union = (min(frames), max(frames)) if frames else None

            animations = gltf.get(
                "animations"
            ) or self._mc._animation._synthesize_clips(gltf, windows, union)
            if not animations:
                return None

            by_name: Dict[str, List[int]] = {}
            for index, node in enumerate(gltf.get("nodes") or []):
                name = node.get("name")
                if name is not None:
                    by_name.setdefault(str(name), []).append(index)

            spans = channel.get("clip_span")
            return self._write_visibility_channels(
                edit,
                to_gate,
                animations,
                windows,
                union,
                by_name,
                fps,
                spans if isinstance(spans, dict) else {},
            )

    def build_visibility_tracks(
        self,
        tracks: Sequence[Dict[str, Any]],
        fps: Optional[float] = None,
        clip_spans: Optional[Dict[str, List[float]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not tracks and not clip_spans:
            return None
        payload: Dict[str, Any] = {
            "version": SceneRecords.VISIBILITY.version,
            "tracks": self._rounded(list(tracks), self._mc.VISIBILITY_TRACK_DIGITS),
        }
        if fps:
            payload["fps"] = float(fps)
        if clip_spans:
            payload["clip_span"] = self._rounded(
                clip_spans, self._mc.VISIBILITY_TRACK_DIGITS
            )
        return payload

    #: Decimal places a visibility track's floats are published at.
    VISIBILITY_TRACK_DIGITS = 6

    def _rounded(self, value: Any, digits: int) -> Any:
        """*value* with every float rounded to *digits* places, containers
        rebuilt, everything else (ints, bools, strings) as given.  ``-0.0``
        comes out as ``0.0``: a sign on nothing is noise too."""
        if isinstance(value, float):
            return round(value, digits) + 0.0
        if isinstance(value, dict):
            return {k: self._rounded(v, digits) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._rounded(v, digits) for v in value]
        return value

    def _write_visibility_channels(
        self,
        edit: "GlbEdit",
        to_gate: List[Tuple[Dict[str, Any], List[Sequence[float]]]],
        animations: List[Dict[str, Any]],
        windows: Dict[str, Tuple[float, float]],
        union: Optional[Tuple[float, float]],
        by_name: Dict[str, List[int]],
        fps: float,
        spans: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Sample every track into every clip and write the channels. One BIN append.

        Split from :meth:`apply_glb_visibility` so the reading of the file
        (which channel, which schema, which rate) stays separate from the
        writing of it, and because the write is the half with the ordering
        constraint: the samples for EVERY clip are computed first, so the
        buffer grows once rather than once per clip.

        *to_gate* pairs each track with what it is GATED on -- its presence
        keys (:meth:`_presence_keys`), resolved once by the caller rather than
        per clip: the answer is a property of the track, and deriving it inside
        the clip loop would redo the same work once per clip per node.
        """
        nodes = edit.gltf.get("nodes") or []
        # (times, values) -> payload slot, so the three nodes that switch on the
        # same frame with the same authored scale share one accessor pair.
        payloads: List[bytes] = []
        slots: Dict[bytes, int] = {}

        def payload_slot(values: Sequence[float]) -> int:
            raw = struct.pack(f"<{len(values)}f", *values)
            if raw not in slots:
                slots[raw] = len(payloads)
                payloads.append(raw)
            return slots[raw]

        planned: List[Tuple[Dict[str, Any], int, int, int, float, float, int]] = []
        missing: Set[str] = set()
        conflicted: Set[str] = set()
        baked: Set[str] = set()
        gated: Set[str] = set()
        clips: Set[str] = set()

        def report_skips() -> None:
            """Name every node that asked for a gate and did not get one.

            Runs on BOTH exits -- a pass that wrote nothing has exactly the
            same thing to report as one that wrote something, and the first
            draft of this only said it on the way out of the second.
            """
            if missing:
                logger.warning(
                    "Visibility: %d keyed node(s) are not in this GLB (%s) -- "
                    "they will be visible for the whole deliverable.",
                    len(missing),
                    ", ".join(sorted(missing)),
                )
            if conflicted:
                logger.warning(
                    "Visibility: %d node(s) already carry a scale animation "
                    "(%s), so their visibility gate was NOT written -- they "
                    "stay visible.",
                    len(conflicted),
                    ", ".join(sorted(conflicted)),
                )
            if baked:
                logger.warning(
                    "Visibility: %d node(s) carry a baked matrix (%s), which "
                    "glTF forbids animating -- their gate was NOT written. "
                    "Re-export with separate translation/rotation/scale.",
                    len(baked),
                    ", ".join(sorted(baked)),
                )

        for animation in animations:
            name = str(animation.get("name") or "")
            window = windows.get(name) or union
            if window is None:
                continue
            zero = self._mc._animation._clip_zero(
                edit.gltf,
                animation,
                window,
                self._mc._animation._clip_span_for(name, spans, windows),
                fps,
            )
            # Nodes this clip already scales: a second channel on the same
            # node/path is undefined behaviour, so the gate stands down rather
            # than corrupt an authored scale animation.
            taken = {
                (c.get("target") or {}).get("node")
                for c in (animation.get("channels") or [])
                if (c.get("target") or {}).get("path") == "scale"
            }
            for track, presence in to_gate:
                target = str(track["node"])
                indices = by_name.get(target)
                if not indices:
                    missing.add(target)
                    continue
                runs = self._mc._visibility_runs(presence, window)
                if not runs or (len(runs) == 1 and runs[0][1]):
                    # Visible for the whole clip: the default, so no channel.
                    continue
                for index in indices:
                    if index in taken:
                        conflicted.add(target)
                        continue
                    node = nodes[index]
                    if node.get("matrix"):
                        # glTF forbids animating a node that carries a matrix;
                        # TRS is required. Never seen from FBX2glTF, but this is
                        # public API pointed at whatever it is handed. Reported
                        # apart from a scale collision: the two need different
                        # things done to them, and one warning naming both would
                        # send the reader looking for the wrong thing.
                        baked.add(target)
                        continue
                    # glTF requires three components; a file that carries fewer
                    # would raise on the axis walk below rather than degrade.
                    scale = node.get("scale")
                    base = (
                        scale
                        if isinstance(scale, (list, tuple))
                        and len(scale) >= 3
                        and all(isinstance(v, (int, float)) for v in scale[:3])
                        else [1.0, 1.0, 1.0]
                    )
                    # Clamped, because a run can open before the clip's zero --
                    # and then deduplicated, because clamping is what ties two
                    # keys to the same instant.
                    placed = self._strictly_increasing(
                        [(max(0.0, (f - zero) / fps), v) for f, v in runs]
                    )
                    if not placed or (len(placed) == 1 and placed[0][1]):
                        continue
                    # Hold the final state to the END of the shot's window. A
                    # clip is only as long as its longest sampler, and the
                    # converter sizes a take from its authored KEYS -- so a
                    # shot whose content is one object appearing would arrive
                    # as a clip that ends the instant it appears (measured:
                    # Shot_1, 93 authored frames, a 0.5s clip). The extra key
                    # changes nothing on screen and makes the clip last as long
                    # as the shot it is named after.
                    tail = max(0.0, (window[1] - zero) / fps)
                    if tail > placed[-1][0]:
                        placed.append((tail, placed[-1][1]))
                    times = [t for t, _ in placed]
                    output: List[float] = []
                    for _, value in placed:
                        on = 1.0 if value else 0.0
                        output.extend(float(base[axis]) * on for axis in range(3))
                    planned.append(
                        (
                            animation,
                            index,
                            payload_slot(times),
                            payload_slot(output),
                            times[0],
                            times[-1],
                            len(times),
                        )
                    )
                    gated.add(target)
                    clips.add(name)

        if not planned:
            report_skips()
            return None

        views = self._mc._append_bin_views(edit, payloads)
        if not views:
            logger.warning(
                "Visibility: this GLB's buffer is external, so the tracks have "
                "nowhere to live -- visibility not applied."
            )
            return None

        accessors = edit.gltf.setdefault("accessors", [])
        # accessor index per (payload slot, kind), so a shared payload is also a
        # shared accessor -- the same three nodes again.
        built: Dict[Tuple[int, str], int] = {}

        def accessor(slot: int, kind: str, count: int, span=None) -> int:
            key = (slot, kind)
            if key not in built:
                entry: Dict[str, Any] = {
                    "bufferView": views[slot],
                    "componentType": 5126,  # FLOAT
                    "count": count,
                    "type": kind,
                }
                if span is not None:  # required on an animation sampler input
                    entry["min"], entry["max"] = [span[0]], [span[1]]
                accessors.append(entry)
                built[key] = len(accessors) - 1
            return built[key]

        for animation, index, in_slot, out_slot, first, last, count in planned:
            samplers = animation.setdefault("samplers", [])
            samplers.append(
                {
                    "input": accessor(in_slot, "SCALAR", count, (first, last)),
                    "output": accessor(out_slot, "VEC3", count),
                    "interpolation": "STEP",
                }
            )
            animation.setdefault("channels", []).append(
                {
                    "sampler": len(samplers) - 1,
                    "target": {"node": index, "path": "scale"},
                }
            )
        edit.dirty = True

        report_skips()
        logger.info(
            "Visibility: %d node(s) gated across %d clip(s) via %d STEP scale "
            "channel(s) -- keyed visibility now plays in any glTF viewer.",
            len(gated),
            len(clips),
            len(planned),
        )
        return {"nodes": len(gated), "channels": len(planned), "clips": len(clips)}

    def _authored_fades(self, gltf: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """Per-node alpha ramps from the visibility channel, or ``{}``.

        The ``opacity`` slice of :meth:`_authored_ramps`, kept because the
        presence gate reasons about the fade alone.
        """
        return self._authored_ramps(gltf)[0].get("opacity", {})

    def _authored_ramps(
        self, gltf: Dict[str, Any]
    ) -> Tuple[Dict[str, Dict[str, List[List[float]]]], Dict[str, Dict[str, Any]]]:
        """Every published per-node ramp, by channel, plus each channel's colours.

        Reads the ``render_effects`` record once for every row of the
        pointer-channel table: a track is a per-node dict whose sibling keys
        are named ramps (``opacity``, ``highlight``, ...), so a new channel is
        a new key on the same record rather than a new carrier.

        Only ``opacity`` is filtered to ramps that actually ramp: a track whose
        ``opacity`` merely mirrors its on/off keys says nothing the stepped
        scale channel does not already say, and writing it as an alpha channel
        would animate a "fade" that was never authored as one. One predicate,
        in one place: the gate that makes a node PRESENT for its fade
        (:meth:`_presence_keys`) reads the same answer. A stepped highlight,
        by contrast, is a real on/off highlight and ships.

        Returns:
            ``({channel: {node: [[frame, value], ...]}},
            {channel: {node: ((r, g, b), ...)}})`` -- one resolved triple per
            stop of that channel, high first.
        """
        from pythontk.file_utils.mesh_convert.glb.fades import CHANNELS

        channel = self._mc.data_export_channel(gltf, SceneRecords.VISIBILITY.key)
        ramps: Dict[str, Dict[str, List[List[float]]]] = {}
        colors: Dict[str, Dict[str, Any]] = {}
        if not isinstance(channel, dict):
            return ramps, colors
        for track in channel.get("tracks") or []:
            if not isinstance(track, dict) or not track.get("node"):
                continue
            node = str(track["node"])
            for name, spec in CHANNELS.items():
                keys = [
                    [float(k[0]), float(k[1])]
                    for k in self._mc._animation._numeric_pairs(track.get(name) or [])
                ]
                if len(keys) < 2:
                    continue
                if name == "opacity" and not self._mc._is_fade(keys):
                    continue
                ramps.setdefault(name, {})[node] = keys
                if spec.color_stops is None:
                    continue
                # One sibling key per stop, read positionally. A track that
                # states only the high stop -- every track authored before the
                # channel grew a low one -- leaves the rest to their own
                # defaults rather than to the high colour.
                published = [track.get(k) for k in spec.color_stops.keys]
                if any(v is not None for v in published):
                    colors.setdefault(name, {})[node] = spec.color_stops.resolve(
                        published
                    )
        return ramps, colors

    def apply_glb_fades(self, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        from pythontk.file_utils.mesh_convert.glb.fades import GlbFades

        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            ramps, colors = self._authored_ramps(gltf)
            if not ramps:
                return None
            metadata = self._mc.data_export_channel(gltf, SceneRecords.SHOTS.key)
            channel = self._mc.data_export_channel(gltf, SceneRecords.VISIBILITY.key)
            fps = self._mc._animation._resolve_clip_fps(
                metadata if isinstance(metadata, dict) else {}, [], channel
            )
            if not fps:
                logger.warning(
                    "Fades: %d authored ramp(s) carry no frame rate, so their "
                    "frame numbers cannot be placed in time -- not applied.",
                    sum(len(per_node) for per_node in ramps.values()),
                )
                return None

            windows = self._mc._take_windows(gltf)
            spans = (channel or {}).get("clip_span")
            if not isinstance(spans, dict):
                spans = {}
            if windows:
                union = (
                    min(s for s, _ in windows.values()),
                    max(e for _, e in windows.values()),
                )
            else:
                # No declared takes (an animated prop never cut into shots):
                # the ramps' own extent is the only window there is, and it is
                # the right one -- the same fallback the visibility gate makes.
                frames = [
                    float(key[0])
                    for per_node in ramps.values()
                    for keys in per_node.values()
                    for key in keys
                ]
                union = (min(frames), max(frames)) if frames else None
            # Each clip's own window and origin, resolved exactly the way the
            # gate resolves them -- a ramp placed against a different zero than
            # the gate it accompanies would fade at a different instant than
            # the object appears.
            clip_windows: Dict[str, Tuple[float, float]] = {}
            zeros: Dict[str, float] = {}
            animations = gltf.get(
                "animations"
            ) or self._mc._animation._synthesize_clips(gltf, windows, union)
            for animation in animations:
                name = str(animation.get("name") or "")
                window = windows.get(name) or union
                if window is None:
                    continue
                clip_windows[name] = window
                zeros[name] = self._mc._animation._clip_zero(
                    gltf,
                    animation,
                    window,
                    self._mc._animation._clip_span_for(name, spans, windows),
                    float(fps),
                    verify=False,
                )
            if not clip_windows:
                return None
            return GlbFades.apply_channels(
                edit, ramps, colors, clip_windows, zeros, float(fps)
            )
