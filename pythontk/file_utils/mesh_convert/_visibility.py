# !/usr/bin/python
# coding=utf-8
"""Animated render effects, realized as channels the GLB can play.

The ``render_effects`` record (v2) names which nodes carry which effect
channels and which materials an animated emission; the curves are the file's
own -- each take's curve proxies (``SceneRecords.TRANSPORT``), read off the FBX
(:meth:`_Visibility.render_effect_curves`), in seconds from the clip's start --
never the record's. Presence (``visibility``, and ``opacity`` at 0) becomes a
STEP ``scale`` channel per clip; every material value becomes a
``KHR_animation_pointer`` channel per clip (:class:`GlbFades`).

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through the facade it holds (``self._mc``).
"""

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Eager: the channel keys below are SceneRecords' own, read at class
# definition.
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbTarget, _SamplerBatch
from pythontk.file_utils.mesh_convert.glb.fades import STEP_SECONDS

logger = logging.getLogger(__name__)

#: ``{clip: {(target, channel): [(seconds, value, interpolation), ...]}}``.
EffectCurves = Dict[str, Dict[Tuple[str, str], List[List[Any]]]]


class _Visibility:
    """Presence gates, pointer channels, the curve reader and the preview.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: The ``data_export`` key a preview states its curves under, for ONE
    #: build: a build input, never a record (no FBX carries it, and the pass
    #: that reads it takes it off the file). ``{"curves": [{"target",
    #: "channel", "keys": [[seconds, value, interpolation?], ...],
    #: "interpolation"?}]}`` -- what a panel previews, or a push's scene keys,
    #: in a file that carries no curve of it.
    EFFECT_PREVIEW_KEY = "render_effects_preview"
    #: How long a cut is given in a pointer channel's LINEAR sampler, seconds.
    EFFECT_STEP_SECONDS = STEP_SECONDS
    #: Decimal places a record's floats (its colours) are published at.
    VISIBILITY_TRACK_DIGITS = 6

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    # ------------------------------------------------------------- the record
    def build_visibility_tracks(
        self,
        tracks: Sequence[Dict[str, Any]],
        materials: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not tracks and not materials:
            return None
        digits = self.VISIBILITY_TRACK_DIGITS
        payload: Dict[str, Any] = {
            "version": SceneRecords.VISIBILITY.version,
            "tracks": self._rounded(list(tracks or ()), digits),
        }
        if materials:
            payload["materials"] = self._rounded(list(materials), digits)
        return payload

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

    # ------------------------------------------------------------- the curves
    def render_effect_curves(self, fbx: Any, bake_fps: Optional[float] = None) -> EffectCurves:
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

        if not isinstance(fbx, FbxFile):
            fbx = FbxFile.load(
                str(fbx), decode_arrays=FbxFile.CURVE_ARRAYS, raw_payloads=False
            )
        transport = SceneRecords.TRANSPORT
        proxies: Dict[str, Tuple[str, str]] = {}
        for owner, name, value in fbx.user_property_items("Model"):
            if name != transport.marker or not owner:
                continue
            try:
                marked = bool(float(value[0] if isinstance(value, tuple) else value))
            except (TypeError, ValueError, IndexError):
                marked = bool(value)
            parsed = transport.parse(owner)
            if marked and parsed:
                proxies[owner] = parsed
        if not proxies:
            return {}

        rate = float(bake_fps or self._mc.CONVERSION_BAKE_FPS)
        ticks_per_second = FbxFile.TICKS_PER_SECOND
        zeros = {
            take: self._clip_zero(first, rate, ticks_per_second)
            for take, (first, _last) in fbx.take_ticks().items()
        }
        frame = 1.0 / float(fbx._fps() or rate)
        keyed = fbx.take_keys(properties=("Lcl Scaling",))
        samplers = None
        curves: EffectCurves = {}
        for take, channels in keyed.items():
            zero = zeros.get(take)
            if zero is None:
                continue
            for (label, _prop, component), keys in channels.items():
                parsed = proxies.get(str(label).split("#", 1)[0])
                if parsed is None or component != "d|X" or not keys:
                    continue
                absolute = [
                    (int(t) / ticks_per_second, float(v), str(i)) for t, v, i in keys
                ]
                if any(i == "smooth" for _t, _v, i in absolute[:-1]):
                    if samplers is None:
                        samplers = fbx.take_samplers(properties=("Lcl Scaling",))
                    sample = samplers[take][(label, _prop, component)]
                    absolute = self._sampled(absolute, sample, frame, ticks_per_second)
                by_target = curves.setdefault(take, {})
                if parsed in by_target:
                    logger.warning(
                        "Render effects: two curve proxies carry %s's %s in %s "
                        "-- the first is read.",
                        *parsed,
                        take,
                    )
                    continue
                by_target[parsed] = [
                    [round(t - zero, 9) + 0.0, round(v, 6) + 0.0, i]
                    for t, v, i in absolute
                ]
        return curves

    @staticmethod
    def _clip_zero(first: int, rate: float, ticks_per_second: int) -> float:
        """The instant, in seconds, a take's clip puts at t=0: its first key
        (over every curve the take holds) floored to the converter's bake
        grid -- measured on FBX2glTF 0.13.1 (2026-11-05): a take keyed from
        frame 1 at 24 fps opens its 30 fps clip on 1/30 s. In the file's own
        ticks, so the floor is exact."""
        grid = ticks_per_second / rate
        if float(grid).is_integer():
            return (int(first) // int(grid)) * int(grid) / ticks_per_second
        return math.floor(int(first) / grid) * grid / ticks_per_second

    @staticmethod
    def _sampled(keys, sample, frame: float, ticks_per_second: int):
        """*keys* with each cubic segment replaced by the curve's own values
        at each of the file's frames inside it, read by *sample* (``ticks ->
        value``), every key then linear: the pass's samplers are LINEAR, and a
        chord between a cubic segment's ends is not the curve."""
        out = []
        for index, (time, value, interpolation) in enumerate(keys):
            if interpolation != "smooth" or index + 1 >= len(keys):
                last = index + 1 >= len(keys) and interpolation == "smooth"
                out.append((time, value, "linear" if last else interpolation))
                continue
            out.append((time, value, "linear"))
            following = keys[index + 1][0]
            step = math.floor(time / frame + 1e-9) + 1
            while step * frame < following - 1e-9:
                at = step * frame
                out.append(
                    (at, float(sample(round(at * ticks_per_second))), "linear")
                )
                step += 1
        return out

    # -------------------------------------------------------------- presence
    def apply_glb_visibility(
        self,
        glb: GlbTarget,
        curves: Optional[EffectCurves] = None,
        carriers: Optional[Dict[Tuple[str, str], List[str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        from pythontk.file_utils.mesh_convert.glb.fades import GlbFades

        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            record = self._mc._current_record(gltf, SceneRecords.VISIBILITY)
            if record is None or not curves:
                return None
            animations = gltf.get("animations") or []
            nodes = gltf.get("nodes") or []
            paths = GlbFades._paths(gltf)
            if carriers is None:
                carriers = GlbFades.carriers(gltf)
            batch = _SamplerBatch()
            planned: List[Tuple[Dict[str, Any], int, int, int]] = []
            missing: Set[str] = set()
            tied: Set[str] = set()
            conflicted: Set[str] = set()
            baked: Set[str] = set()
            gated: Set[str] = set()
            clips: Set[str] = set()
            presence = set(SceneRecords.TRANSPORT.stepped) | {"opacity"}
            for animation in animations:
                clip = str(animation.get("name") or "")
                per_clip = curves.get(clip) or {}
                end = self._mc._animation._animation_time_span(edit, animation)[1]
                taken = {
                    (c.get("target") or {}).get("node")
                    for c in (animation.get("channels") or [])
                    if (c.get("target") or {}).get("path") == "scale"
                }
                for track in record.get("tracks") or ():
                    if not isinstance(track, dict) or not track.get("node"):
                        continue
                    node = str(track["node"])
                    # The record names the node as its host does; its curves
                    # go by the transport's name, and the node they play on is
                    # their proxy's parent, else the one the name states
                    # (``SceneRecords.TRANSPORT.join``).
                    joined = SceneRecords.TRANSPORT.target_name(node)
                    named = presence & set(track.get("channels") or ())
                    visibility = per_clip.get((joined, "visibility")) if "visibility" in named else None
                    opacity = per_clip.get((joined, "opacity")) if "opacity" in named else None
                    if not visibility and not opacity:
                        continue
                    runs = GlbFades._both(
                        GlbFades._stepped_runs(visibility) if visibility else [],
                        GlbFades._positive_runs(opacity) if opacity else [],
                    )
                    if not runs or all(on for _t, on in runs):
                        continue  # present for the whole clip: the default
                    indices, status = GlbFades._resolve(
                        node,
                        [c for c, curve in (("visibility", visibility), ("opacity", opacity)) if curve],
                        carriers,
                        paths,
                    )
                    if not indices:
                        (tied if status == "ambiguous" else missing).add(node)
                        continue
                    last = max(
                        float(k[0]) for k in (visibility or []) + (opacity or [])
                    )
                    tail = max(float(end or 0.0), last, runs[-1][0])
                    if tail > runs[-1][0]:
                        runs = runs + [(tail, runs[-1][1])]
                    times = [max(0.0, t) for t, _v in runs]
                    placed = GlbFades._strictly_increasing(
                        list(zip(times, [v for _t, v in runs]))
                    )
                    for index in indices:
                        if index in taken:
                            conflicted.add(node)
                            continue
                        if nodes[index].get("matrix"):
                            baked.add(node)
                            continue
                        scale = nodes[index].get("scale")
                        base = (
                            scale
                            if isinstance(scale, (list, tuple))
                            and len(scale) >= 3
                            and all(isinstance(v, (int, float)) for v in scale[:3])
                            else [1.0, 1.0, 1.0]
                        )
                        output: List[float] = []
                        for _t, on in placed:
                            output.extend(float(base[axis]) * on for axis in range(3))
                        stamps = [t for t, _v in placed]
                        planned.append((animation, index, batch.slot(stamps), batch.slot(output)))
                        gated.add(node)
                        clips.add(clip)
            self._report_skips(missing, conflicted, baked, tied)
            if not planned:
                return None
            if not batch.commit(edit):
                logger.warning(
                    "Visibility: this GLB's buffer is external, so the gates have "
                    "nowhere to live -- presence not applied."
                )
                return None
            for animation, index, in_slot, out_slot in planned:
                sampler = batch.sampler(animation, in_slot, out_slot, "VEC3", "STEP")
                animation.setdefault("channels", []).append(
                    {"sampler": sampler, "target": {"node": index, "path": "scale"}}
                )
            edit.dirty = True
            logger.info(
                "Visibility: %d node(s) gated across %d clip(s) via %d STEP scale "
                "channel(s) -- presence now plays in any glTF viewer.",
                len(gated),
                len(clips),
                len(planned),
            )
            return {"nodes": len(gated), "channels": len(planned), "clips": len(clips)}

    @staticmethod
    def _report_skips(
        missing: Set[str], conflicted: Set[str], baked: Set[str], tied: Set[str] = frozenset()
    ) -> None:
        """Name every node that asked for a gate and did not get one."""
        if tied:
            logger.warning(
                "Visibility: %d keyed node(s) name a node several nodes share, and "
                "no curve proxy says which (%s) -- never guessed, no gate written. "
                "Give each node its own name, or re-export so each curve rides "
                "its node's proxy.",
                len(tied),
                ", ".join(sorted(tied)),
            )
        if missing:
            logger.warning(
                "Visibility: %d keyed node(s) are not in this GLB (%s) -- they "
                "will be visible for the whole deliverable.",
                len(missing),
                ", ".join(sorted(missing)),
            )
        if conflicted:
            logger.warning(
                "Visibility: %d node(s) already carry a scale animation (%s), so "
                "their presence gate was NOT written -- they stay visible.",
                len(conflicted),
                ", ".join(sorted(conflicted)),
            )
        if baked:
            logger.warning(
                "Visibility: %d node(s) carry a baked matrix (%s), which glTF "
                "forbids animating -- their gate was NOT written. Re-export with "
                "separate translation/rotation/scale.",
                len(baked),
                ", ".join(sorted(baked)),
            )

    # ------------------------------------------------------- pointer channels
    def apply_glb_fades(
        self,
        glb: GlbTarget,
        curves: Optional[EffectCurves] = None,
        carriers: Optional[Dict[Tuple[str, str], List[str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        from pythontk.file_utils.mesh_convert.glb.fades import GlbFades

        with self._mc.open_glb(glb) as edit:
            record = self._mc._current_record(edit.gltf, SceneRecords.VISIBILITY)
            if record is None or not curves:
                return None
            return GlbFades.apply_curves(edit, record, curves, carriers)

    # ---------------------------------------------------------------- preview
    def effect_preview_channels(
        self,
        nodes: Sequence[str],
        channel: str,
        keys: Sequence[Sequence[float]],
        colors: Optional[Sequence[Optional[Sequence[float]]]] = None,
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        from pythontk.file_utils.mesh_convert.glb.fades import CHANNELS

        spec = CHANNELS.get(channel)
        if spec is None or spec.target != "node":
            raise KeyError(
                f"Unknown render-effect channel {channel!r}; expected one of "
                f"{', '.join(n for n, s in CHANNELS.items() if s.target == 'node')}."
            )
        names = list(dict.fromkeys(str(n) for n in nodes or () if n))
        if not names:
            raise ValueError("An effect preview needs at least one node.")
        if not fps or float(fps) <= 0:
            raise ValueError("An effect preview needs the rate its keys are quoted in.")
        ramp = [
            [round(float(frame) / float(fps), 9) + 0.0, float(value)]
            for frame, value in keys
        ]
        if len(ramp) < 2:
            raise ValueError("An effect preview needs a ramp of at least two keys.")
        tracks = []
        for name in names:
            track: Dict[str, Any] = {"node": name, "channels": [channel]}
            if spec.color_stops is not None:
                for key, rgb in zip(spec.color_stops.keys, colors or ()):
                    if rgb is not None:
                        track[key] = [float(c) for c in list(rgb)[:3]]
            tracks.append(track)
        return {
            SceneRecords.VISIBILITY.key: self._mc.build_visibility_tracks(tracks),
            SceneRecords.SHOTS.key: None,
            self.EFFECT_PREVIEW_KEY: {
                "curves": [
                    {"target": name, "channel": channel, "keys": ramp} for name in names
                ]
            },
        }

    def effect_preview_curves(
        self,
        curves: Sequence[Dict[str, Any]],
        start: float,
        fps: float,
        clips: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not fps or float(fps) <= 0:
            raise ValueError("Effect curves need the rate their frames are quoted in.")
        rate = float(fps)
        framed = [
            (
                str(entry["target"]),
                str(entry.get("channel")),
                [
                    (float(k[0]), float(k[1]), str(k[2]) if len(k) > 2 else "linear")
                    for k in entry.get("keys") or ()
                    if isinstance(k, (list, tuple)) and len(k) >= 2
                ],
            )
            for entry in curves or ()
            if isinstance(entry, dict) and entry.get("target")
        ]

        def stated(origin: float, end: Optional[float]) -> List[Dict[str, Any]]:
            out = []
            for target, channel, keys in framed:
                cut = self._windowed(
                    [((t - origin) / rate, v, i) for t, v, i in keys], end
                )
                if cut:
                    out.append(
                        {
                            "target": target,
                            "channel": channel,
                            "keys": [[round(t, 9) + 0.0, v, i] for t, v, i in cut],
                        }
                    )
            return out

        windows = [
            (str(c["name"]), float(c["start"]), float(c["end"]))
            for c in clips or ()
            if isinstance(c, dict) and c.get("name") is not None
            and c.get("start") is not None and c.get("end") is not None
        ]
        if windows:
            if not framed:
                return {}
            # Every clip, whatever its curves do in it: a shot in which no
            # effect moves still spans its shot (its duration, held).
            per_clip = [
                {
                    "name": name,
                    "duration": round((last - first) / rate, 9) + 0.0,
                    "curves": stated(first, (last - first) / rate),
                }
                for name, first, last in windows
            ]
            return {self.EFFECT_PREVIEW_KEY: {"clips": per_clip}}
        flat = stated(float(start), None)
        return {self.EFFECT_PREVIEW_KEY: {"curves": flat}} if flat else {}

    @staticmethod
    def _value_at(keys: List[Tuple[float, float, str]], time: float) -> Tuple[float, str]:
        """``(value, interpolation)`` the time-ordered *keys* hold at *time*:
        the last key at or before it (ties: the later), its segment's
        interpolation carried; a step's value held, else interpolated toward
        the next key; the first key's value before it, the last's after."""
        prev = None
        for key in keys:
            if key[0] <= time:
                prev = key
                continue
            if prev is None:
                return key[1], "linear"
            if prev[2] == "step":
                return prev[1], prev[2]
            span = key[0] - prev[0]
            return prev[1] + (key[1] - prev[1]) * (time - prev[0]) / span, prev[2]
        return prev[1], prev[2]

    @classmethod
    def _windowed(
        cls, keys: List[Tuple[float, float, str]], end: Optional[float] = None
    ) -> List[Tuple[float, float, str]]:
        """*keys* (seconds, from the window's start) cut to ``[0, end]``.

        A key before 0 gives way to one at 0 holding the curve's value there
        (glTF has no time before 0). With an *end* -- a clip of a shot, which
        spans its shot -- both ends are pinned at the value the curve holds
        there, so the curve covers the clip whatever it keys inside it; with
        none the curve keeps its own extent after 0, as an animated write's
        one take carries it."""
        keys = sorted(keys, key=lambda k: k[0])
        if not keys:
            return []
        out = [k for k in keys if k[0] >= 0.0 and (end is None or k[0] <= end)]
        pin = end is not None or keys[0][0] < 0.0
        if pin and not (out and out[0][0] == 0.0):
            out.insert(0, (0.0, *cls._value_at(keys, 0.0)))
        if end is not None and out[-1][0] < end:
            out.append((float(end), cls._value_at(keys, end)[0], "linear"))
        return out

    @staticmethod
    def _curve_map(entries: Any) -> Dict[Tuple[str, str], List[List[Any]]]:
        """An overlay's ``curves`` list as ``{(target, channel): keys}``, each
        target by the transport's name (``SceneRecords.TRANSPORT.target_name``),
        as the file's own curves are keyed: a host states it as it names it."""
        curve_map: Dict[Tuple[str, str], List[List[Any]]] = {}
        for entry in entries if isinstance(entries, list) else ():
            if not isinstance(entry, dict) or not entry.get("target"):
                continue
            interpolation = str(entry.get("interpolation") or "linear")
            keys = [
                [float(k[0]), float(k[1]), str(k[2]) if len(k) > 2 else interpolation]
                for k in entry.get("keys") or ()
                if isinstance(k, (list, tuple)) and len(k) >= 2
            ]
            if keys:
                target = SceneRecords.TRANSPORT.target_name(entry["target"])
                curve_map[(target, str(entry.get("channel")))] = keys
        return curve_map

    def take_preview_curves(self, gltf: Dict[str, Any]) -> EffectCurves:
        return self.take_preview(gltf)[0]

    def take_preview(self, gltf: Dict[str, Any]) -> Tuple[EffectCurves, Dict[str, float]]:
        """:meth:`take_preview_curves`, and ``{clip: seconds}`` each clip an
        overlay states must span (its ``duration``) -- what the build holds a
        clip to when nothing in it moves (``_Animation._hold_take_spans``);
        ``{}`` for an overlay of curves alone."""
        stated = self._mc.data_export_channel(gltf, self.EFFECT_PREVIEW_KEY)
        if stated is None:
            return {}, {}
        self._mc._sidecar._drop_data_export_channel(gltf, self.EFFECT_PREVIEW_KEY)
        if isinstance(stated, dict) and isinstance(stated.get("clips"), list):
            named = [
                clip
                for clip in stated["clips"]
                if isinstance(clip, dict) and clip.get("name") is not None
            ]
            per_clip = {
                str(clip["name"]): curve_map
                for clip in named
                if (curve_map := self._curve_map(clip.get("curves")))
            }
            spans = {
                str(clip["name"]): float(clip["duration"])
                for clip in named
                if isinstance(clip.get("duration"), (int, float)) and clip["duration"] > 0
            }
            names = list(dict.fromkeys(str(clip["name"]) for clip in named))
            if names:
                self._mc._animation._synthesize_clips(gltf, names)
            return per_clip, spans
        entries = stated.get("curves") if isinstance(stated, dict) else None
        if not isinstance(entries, list):
            logger.warning("Render effects: an unreadable preview was ignored.")
            return {}, {}
        curve_map = self._curve_map(entries)
        if not curve_map:
            return {}, {}
        animations = gltf.get("animations") or self._mc._animation._synthesize_clips(gltf)
        return {str(a.get("name") or ""): dict(curve_map) for a in animations}, {}
