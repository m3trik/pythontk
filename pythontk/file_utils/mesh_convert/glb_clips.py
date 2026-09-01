# !/usr/bin/python
# coding=utf-8
"""Rebuild a GLB's shot clips from its one whole-timeline animation.

Why this exists, measured on a 12-shot production assembly (Maya 2025 ->
FBX2glTF 0.13.1):

Maya's ``FBXExportSplitAnimationIntoTakes`` does not slice the baked animation.
It restricts each curve to the take's window FIRST and bakes what survives, so
a curve whose keys all sit outside a shot contributes **no channel at all** to
that shot -- and the node then plays its rest pose for the shot's whole
duration, wherever that happens to be. On a scene with 358 keys spread over
2635 frames that is not an edge case: every one of Shot_1..Shot_11 was wrong on
**every frame**, by up to 3.73 m and 90 degrees, while the whole-timeline stack
the same export retains was correct on all 2629 of them.

Two more consequences of the same mechanism, both of which this pass removes:

* A take is emitted spanning its authored KEYS rather than its declared window,
  so a shot whose motion starts late is SHORTER than the shot -- measured at 43
  missing frames on Shot_5 (declared 915-1015, shipped 958-1015).
* The retained whole-timeline stack ships as ``Take 001`` beside the shots,
  which reads as "the full sequence is a take" to anyone opening the file.

So the clips are built here instead, from the stack that is already correct: for
each declared take, every source channel is windowed to the take, pinned at both
ends, and rebased so the take's first frame is ``t=0``. Nothing is resampled --
interior keys are copied verbatim and only the two boundary samples are
evaluated -- so a clip reproduces the source exactly on its window, whatever the
source's own key density.

This runs on the deliverable, so it needs no scene, mutates nothing, and both
the exporter and the preview push get the same clips from the same code. It is
also indifferent to whether the FBX split ran: the declared clips are REPLACED
either way.
"""

from __future__ import annotations

import bisect
import logging
import math
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["GlbClips"]

logger = logging.getLogger(__name__)

#: Number of components per glTF accessor type.
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
#: The one component type an animation sampler may use for its input, and the
#: only one this pass will touch on an output. Quantized outputs (normalized
#: byte/short rotations) are legal glTF but nothing in this pipeline writes
#: them, and a half-understood requantization would be worse than declining.
_FLOAT = 5126


class _GlbClipsInternal:
    """Internal helpers for :class:`GlbClips`."""

    # ---- reading ------------------------------------------------------

    @staticmethod
    def _read_accessor(
        gltf: Dict[str, Any], blob: Optional[bytes], index: Any
    ) -> Optional[List[Tuple[float, ...]]]:
        """Decode accessor *index* as a list of float tuples, or ``None``.

        ``None`` for anything this pass will not rewrite faithfully -- a
        non-float component type, a sparse accessor, a missing bufferView -- so
        the caller can decline the whole rebuild rather than emit a clip it
        half-understood.
        """
        accessors = gltf.get("accessors") or []
        if not isinstance(index, int) or not 0 <= index < len(accessors):
            return None
        acc = accessors[index] or {}
        if acc.get("componentType") != _FLOAT or "sparse" in acc:
            return None
        width = _WIDTH.get(acc.get("type"))
        view_index = acc.get("bufferView")
        if width is None or not isinstance(view_index, int) or blob is None:
            return None
        views = gltf.get("bufferViews") or []
        if not 0 <= view_index < len(views):
            return None
        view = views[view_index] or {}
        base = int(view.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
        count = int(acc.get("count", 0))
        # An interleaved view strides past the elements between ours; a packed
        # one (the norm for animation) has no stride and reads contiguously.
        stride = int(view.get("byteStride") or 0) or width * 4
        end = base + (count - 1) * stride + width * 4 if count else base
        if end > len(blob):
            return None
        try:
            return [
                struct.unpack_from(f"<{width}f", blob, base + i * stride)
                for i in range(count)
            ]
        except struct.error:
            return None

    @classmethod
    def _read_sampler(
        cls, gltf: Dict[str, Any], blob: Optional[bytes], sampler: Dict[str, Any]
    ) -> Optional[Tuple[List[float], List[Tuple[float, ...]], str]]:
        """``(times, values, interpolation)`` for one sampler, or ``None``."""
        times = cls._read_accessor(gltf, blob, sampler.get("input"))
        values = cls._read_accessor(gltf, blob, sampler.get("output"))
        if times is None or values is None or not times:
            return None
        interpolation = str(sampler.get("interpolation") or "LINEAR")
        flat = [t[0] for t in times]
        if interpolation == "CUBICSPLINE":
            # Three entries per key (in-tangent, value, out-tangent). Slicing
            # one would need tangents at the cut, which are not derivable from
            # the neighbours -- so it is flattened to the key values and the
            # clip carries LINEAR. Nothing in this pipeline emits CUBICSPLINE;
            # this is the path that keeps a foreign file from being corrupted.
            if len(values) != 3 * len(flat):
                return None
            values = values[1::3]
            interpolation = "LINEAR"
        if len(values) != len(flat):
            return None
        return flat, list(values), interpolation

    # ---- sampling -----------------------------------------------------

    @staticmethod
    def _lerp(a: Sequence[float], b: Sequence[float], u: float) -> Tuple[float, ...]:
        return tuple(x + (y - x) * u for x, y in zip(a, b))

    @classmethod
    def _slerp(
        cls, a: Sequence[float], b: Sequence[float], u: float
    ) -> Tuple[float, ...]:
        """Shortest-arc quaternion interpolation, per the glTF LINEAR rule."""
        dot = sum(x * y for x, y in zip(a, b))
        if dot < 0.0:
            b = tuple(-x for x in b)
            dot = -dot
        if dot > 0.9995:  # nearly parallel: lerp, then renormalize
            out = cls._lerp(a, b, u)
        else:
            theta = math.acos(max(-1.0, min(1.0, dot)))
            sin_theta = math.sin(theta)
            wa = math.sin((1.0 - u) * theta) / sin_theta
            wb = math.sin(u * theta) / sin_theta
            out = tuple(x * wa + y * wb for x, y in zip(a, b))
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        return tuple(x / norm for x in out)

    @staticmethod
    def _upper(times: Sequence[float], at: float) -> int:
        """Index of the key *at* interpolates toward -- always a valid pair.

        Clamped into ``[1, len - 1]`` so the caller can read ``lo = hi - 1``
        unconditionally; the out-of-range cases are handled before this runs.
        """
        return max(1, min(len(times) - 1, bisect.bisect_left(times, at)))

    @classmethod
    def _evaluate(
        cls,
        times: Sequence[float],
        values: Sequence[Sequence[float]],
        at: float,
        interpolation: str,
        quaternion: bool,
    ) -> Tuple[float, ...]:
        """The sampler's value at *at*, held outside its own range."""
        if at <= times[0]:
            return tuple(values[0])
        if at >= times[-1]:
            return tuple(values[-1])
        hi = cls._upper(times, at)
        lo = hi - 1
        if interpolation == "STEP":
            return tuple(values[lo])
        span = times[hi] - times[lo]
        if span <= 0:
            return tuple(values[lo])
        u = (at - times[lo]) / span
        if quaternion:
            return cls._slerp(values[lo], values[hi], u)
        return cls._lerp(values[lo], values[hi], u)

    # ---- slicing ------------------------------------------------------

    @classmethod
    def _window(
        cls,
        times: Sequence[float],
        values: Sequence[Sequence[float]],
        interpolation: str,
        quaternion: bool,
        lo: float,
        hi: float,
        eps: float,
    ) -> Tuple[List[float], List[Sequence[float]]]:
        """Every key in ``[lo, hi]``, with both ends pinned, rebased to ``lo``.

        Interior keys are COPIED, never resampled: the clip then reproduces the
        source exactly on its window regardless of the source's key density, and
        a sparse STEP channel stays sparse instead of exploding to one key per
        frame.  The two boundary samples are evaluated with the sampler's own
        interpolation, which is what makes a shot whose window starts mid-curve
        open on the pose the DCC shows there rather than on the previous key.

        *eps* is how close a key has to be to a bound to BE that bound. It has
        to clear float32: the source's times come back through a float32 buffer,
        where a key authored on frame 2500 reads 2500.0001 -- so an exact
        comparison would pin a second key a ten-thousandth of a frame from the
        first one, and the clip would open on whichever of the two won. A key
        inside the tolerance is snapped ONTO the bound rather than merely
        accepted, so the clip's own first time is exactly zero.
        """
        if not times:
            return [], []
        kept_t = [t for t in times if lo - eps <= t <= hi + eps]
        kept_v = [v for t, v in zip(times, values) if lo - eps <= t <= hi + eps]
        if not kept_t or kept_t[0] > lo + eps:
            kept_t.insert(0, lo)
            kept_v.insert(
                0, cls._evaluate(times, values, lo, interpolation, quaternion)
            )
        else:
            kept_t[0] = lo
        if kept_t[-1] < hi - eps:
            kept_t.append(hi)
            kept_v.append(cls._evaluate(times, values, hi, interpolation, quaternion))
        else:
            kept_t[-1] = hi
        # glTF requires an animation sampler's input to STRICTLY increase, and
        # snapping an end onto its bound can tie it with a neighbour that was
        # also inside the tolerance. Only reachable on a source sampled finer
        # than four times a frame -- but a tie is a malformed file, not a
        # rounding artefact, so it is closed here rather than assumed away.
        keep = [i for i in range(len(kept_t)) if i == 0 or kept_t[i] > kept_t[i - 1]]
        if len(keep) != len(kept_t):
            kept_t = [kept_t[i] for i in keep]
            kept_v = [kept_v[i] for i in keep]
        if quaternion:
            # A synthesized end may come back on the far side of the double
            # cover from its neighbour; the sign is free, but a flip between
            # adjacent keys is a full extra revolution at playback.
            for i in range(1, len(kept_v)):
                if sum(x * y for x, y in zip(kept_v[i - 1], kept_v[i])) < 0.0:
                    kept_v[i] = tuple(-x for x in kept_v[i])
        return [t - lo for t in kept_t], kept_v

    # ---- writing ------------------------------------------------------

    @staticmethod
    def _source_stack(
        animations: Sequence[Dict[str, Any]], declared: Sequence[str]
    ) -> Optional[int]:
        """Index of the whole-timeline stack to slice, or ``None``.

        The one animation no declared take names -- what Maya's exporter retains
        beside the takes it was asked to split.  With several (nothing in this
        pipeline writes more, but a hand-assembled file might) the widest wins,
        since only one of them can be the whole timeline.
        """
        named = set(declared)
        best: Optional[Tuple[int, int]] = None
        for index, animation in enumerate(animations):
            if str(animation.get("name") or "") in named:
                continue
            weight = len(animation.get("channels") or [])
            if best is None or weight > best[1]:
                best = (index, weight)
        return None if best is None else best[0]


class GlbClips(_GlbClipsInternal):
    """Build a GLB's declared shot clips from its whole-timeline animation."""

    #: What the retained whole-timeline stack is renamed to.  Maya calls it
    #: ``Take 001``, which reads as one more shot in a clip list that is
    #: otherwise all shots -- the confusion this name exists to remove.  It is
    #: kept rather than dropped because playing the sequence end to end
    #: (through the GAPS between shots, which no concatenation of clips
    #: reproduces) is a thing consumers actually do.
    SEQUENCE_CLIP = "FULL_SEQUENCE"

    #: Per-animation ``extras`` key naming the authoring frame this clip places
    #: at ``t=0``.  A clip built here knows its own origin exactly, so it says
    #: so rather than leaving every reader to re-derive it from the take list
    #: and the converter's rebasing rule.
    ZERO_FRAME_KEY = "zero_frame"

    @classmethod
    def rebuild(
        cls,
        edit: Any,
        takes: Sequence[Dict[str, Any]],
        fps: float,
        source_zero: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Replace the declared clips with exact slices of the source stack.

        Parameters:
            edit: An open ``MeshConvert.GlbEdit``.
            takes: ``[{"name", "start", "end"}, ...]`` -- the declared shots,
                in authoring frames.
            fps: The rate those frame numbers are quoted in.
            source_zero: The authoring frame the source stack places at its own
                ``t=0``.  Zero for a stack the converter did not rebase.

        Returns:
            ``{"clips": n, "channels": n, "source": name, "bytes": n}``, or
            ``None`` when there is nothing to rebuild (no takes, no source
            stack, or a source this pass will not rewrite faithfully).
        """
        gltf = edit.gltf
        animations = gltf.get("animations") or []
        if not animations or not takes or fps <= 0:
            return None

        windows: List[Tuple[str, float, float]] = []
        for take in takes:
            try:
                windows.append(
                    (str(take["name"]), float(take["start"]), float(take["end"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
        if not windows:
            return None

        index = cls._source_stack(animations, [name for name, _s, _e in windows])
        if index is None:
            logger.debug(
                "Clips: every animation is a declared take, so there is no "
                "whole-timeline stack to slice -- clips left as exported."
            )
            return None
        source = animations[index]
        source_name = str(source.get("name") or "")

        # Decode the whole source once. A channel this pass cannot read is a
        # reason to decline the REBUILD, not to emit a clip missing it: a clip
        # silently short one channel is the exact failure being repaired.
        samplers = source.get("samplers") or []
        decoded: List[Optional[Tuple[List[float], List[Any], str]]] = []
        for sampler in samplers:
            decoded.append(cls._read_sampler(gltf, edit.bin_data, sampler))
        every = source.get("channels") or []
        channels = [
            channel
            for channel in every
            if isinstance(channel.get("sampler"), int)
            and 0 <= channel["sampler"] < len(decoded)
        ]
        if not channels:
            return None
        # Same rule as an unreadable sampler: a channel pointing at a sampler
        # that is not there is a malformed source, and dropping it quietly
        # would rebuild clips missing exactly what this pass exists to keep.
        if len(channels) != len(every) or any(
            decoded[channel["sampler"]] is None for channel in channels
        ):
            logger.warning(
                "Clips: %r carries a sampler this pass cannot rewrite "
                "(quantized, sparse or malformed) -- clips left as exported.",
                source_name,
            )
            return None

        # Plan every clip before touching the buffer, so it grows once.
        payloads: List[bytes] = []
        slots: Dict[bytes, int] = {}

        def slot(values: Sequence[float]) -> int:
            raw = struct.pack(f"<{len(values)}f", *values)
            if raw not in slots:
                slots[raw] = len(payloads)
                payloads.append(raw)
            return slots[raw]

        # A quarter frame: far above the float32 drift on a long timeline
        # (~1e-4 frames at frame 2500) and far below any authored spacing, so
        # "this key IS the boundary" cannot be decided by rounding.
        eps = 0.25 / fps
        planned: List[Dict[str, Any]] = []
        for name, start, end in windows:
            lo = (start - source_zero) / fps
            hi = (end - source_zero) / fps
            built: List[Dict[str, Any]] = []
            for channel in channels:
                target = channel.get("target") or {}
                if target.get("node") is None or not target.get("path"):
                    continue
                times, values, interpolation = decoded[channel["sampler"]]
                quaternion = target["path"] == "rotation"
                keys, vals = cls._window(
                    times, values, interpolation, quaternion, lo, hi, eps
                )
                if not keys:
                    continue
                flat = [component for value in vals for component in value]
                built.append(
                    {
                        "target": dict(target),
                        "interpolation": interpolation,
                        "input": slot(keys),
                        "output": slot(flat),
                        "count": len(keys),
                        "width": len(vals[0]),
                        "min": keys[0],
                        "max": keys[-1],
                    }
                )
            planned.append({"name": name, "start": start, "channels": built})

        if not payloads:
            logger.debug(
                "Clips: %r drives nothing inside any declared window -- clips "
                "left as exported.",
                source_name,
            )
            return None
        added = cls._append(edit, payloads)
        if added is None:
            return None

        accessors = gltf.setdefault("accessors", [])
        written = 0
        rebuilt: List[Dict[str, Any]] = []
        for plan in planned:
            samplers_out: List[Dict[str, Any]] = []
            channels_out: List[Dict[str, Any]] = []
            for spec in plan["channels"]:
                accessors.append(
                    {
                        "bufferView": added[spec["input"]],
                        "componentType": _FLOAT,
                        "count": spec["count"],
                        "type": "SCALAR",
                        # Required on an animation sampler input, and what
                        # ``_animation_span`` reads to place the clip.
                        "min": [spec["min"]],
                        "max": [spec["max"]],
                    }
                )
                input_index = len(accessors) - 1
                accessors.append(
                    {
                        "bufferView": added[spec["output"]],
                        "componentType": _FLOAT,
                        "count": spec["count"],
                        "type": {2: "VEC2", 3: "VEC3", 4: "VEC4"}.get(
                            spec["width"], "SCALAR"
                        ),
                    }
                )
                samplers_out.append(
                    {
                        "input": input_index,
                        "output": len(accessors) - 1,
                        "interpolation": spec["interpolation"],
                    }
                )
                channels_out.append(
                    {"sampler": len(samplers_out) - 1, "target": spec["target"]}
                )
                written += 1
            rebuilt.append(
                {
                    "name": plan["name"],
                    "samplers": samplers_out,
                    "channels": channels_out,
                    # Built to the window, so the origin is the window -- no
                    # lead-in to discover and no rebasing rule to reproduce.
                    "extras": {cls.ZERO_FRAME_KEY: plan["start"]},
                }
            )

        # The sequence last: a consumer that ignores the manifest and opens
        # ``animations[0]`` then lands on the first SHOT, which is what the
        # clip list looks like it promises.
        declared_names = {name for name, _s, _e in windows}
        source["name"] = (
            cls.SEQUENCE_CLIP
            if cls.SEQUENCE_CLIP not in declared_names
            else source_name
        )
        source.setdefault("extras", {})[cls.ZERO_FRAME_KEY] = source_zero
        gltf["animations"] = rebuilt + [source]
        edit.dirty = True

        total = sum(len(raw) for raw in payloads)
        logger.info(
            "Clips: rebuilt %d shot clip(s) with %d channel(s) from %r "
            "(+%.2f MB); the whole-timeline stack ships as %r.",
            len(rebuilt),
            written,
            source_name,
            total / 1e6,
            source["name"],
        )
        return {
            "clips": len(rebuilt),
            "channels": written,
            "source": source_name,
            "bytes": total,
        }

    @staticmethod
    def _append(edit: Any, payloads: Sequence[bytes]) -> Optional[List[int]]:
        """Append the planned buffers, or ``None`` when the GLB has no BIN.

        Split out so the buffer growth stays the one call it is, and so a GLB
        with an EXTERNAL buffer (which ``_append_bin_views`` declines) aborts
        the rebuild instead of writing accessors onto views that do not exist.
        """
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        added = MeshConvert._append_bin_views(edit, payloads)
        if not added:
            logger.warning(
                "Clips: this GLB's buffer is external, so the rebuilt clips "
                "have nowhere to live -- clips left as exported."
            )
            return None
        return added
