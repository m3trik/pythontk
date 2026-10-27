# !/usr/bin/python
# coding=utf-8
"""Zero-dependency binary-FBX reader: header, node records, objects, takes.

The FBX half of deliverable verification. :meth:`MeshConvert.fbx_to_glb`
*converts* FBX through an external binary but nothing in the ecosystem could
*look inside* one — so a truncated write, a dropped take or a missing
animation stack surfaced only after a GLB conversion or a DCC import. This
reads Kaydara's binary container directly (both the 32-bit pre-7500 and the
64-bit 7500+ record layouts), decodes property records, and answers the
census questions a verifier asks. Array payloads are **skipped by default** —
a 200 MB production FBX parses in well under a second because the geometry is
never inflated.

It also reads the animation as an importer does: each curve evaluated the way
Maya reads it back (constant, linear, cubic with stored or Maya-computed
slopes, weighted), which is how :meth:`FbxFile.take_layout` tells which takes
together hold all of a file's motion -- the input to
:meth:`FbxMedia.merge_takes`.

Read-only by design: there is no writer here, so this can never be what
damages the file it inspects.
"""

import math
import os
import struct
import zlib
from bisect import bisect_left, bisect_right
from collections import Counter
from typing import (
    Any,
    Collection,
    Dict,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pythontk.math_utils.key_curve import KeyCurve

#: The 23-byte magic every binary FBX starts with.
FBX_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"

#: FBX splits an object's display name from its class with this token:
#: ``b"Shot_1\\x00\\x01AnimStack"`` -> name ``Shot_1``, class ``AnimStack``.
_NAME_CLASS_SEPARATOR = b"\x00\x01"


class _FbxCurve(NamedTuple):
    """One ``AnimationCurve`` as stored -- what :class:`FbxFile` evaluates.

    *times* (FBX ticks) and *values* per key; the attribute runs
    ``KeyAttrRefCount`` encodes: each run's ``KeyAttrFlags`` (*flags*), its
    four ``KeyAttrDataFloat`` values (*data*) and the key it starts at
    (*starts*); *clamp*, the value step under which a Maya clamped key on this
    curve's channel is flat.
    """

    times: List[int]
    values: List[float]
    flags: List[int]
    data: List[Tuple[float, ...]]
    starts: List[int]
    clamp: float


class _FbxFileInternal:
    """Record parsing for :class:`FbxFile`, and the curve model its take
    layout -- and :meth:`FbxMedia.merge_takes` -- judge takes by."""

    #: FBX ticks per second (``FbxTime``'s unit) in every file this reader
    #: has met -- Maya and Blender both write the 7.x default. A file whose
    #: ``OtherFlags/TCDefinition`` is not 127 pins another rate; the times
    #: below are returned as stored, in that file's ticks.
    TICKS_PER_SECOND = 46186158000

    #: ``KeyAttrFlags`` bits, the FBX SDK's ``FbxAnimCurveDef``: interpolation
    #: (the segment AFTER the key), tangent mode, weights.
    _INTERPOLATION = 0x0E
    _CONSTANT, _LINEAR, _CUBIC = 0x02, 0x04, 0x08
    #: On a constant key: the segment takes the NEXT key's value (step-next).
    _CONSTANT_NEXT = 0x100
    _TANGENT_AUTO, _TANGENT_TCB, _TANGENT_USER = 0x100, 0x200, 0x400
    #: With :attr:`_TANGENT_USER`: the key's two sides carry their own slopes.
    _TANGENT_BREAK = 0x800
    _CLAMP, _CLAMP_PROGRESSIVE = 0x1000, 0x4000
    _TIME_INDEPENDENT = 0x2000
    #: The tangent-mode bits a computed key carries (:meth:`_computed`),
    #: cleared where its slopes are written out (:meth:`_keep_own_tangents`).
    _COMPUTED_BITS = (
        _TANGENT_AUTO | _TANGENT_TCB | _CLAMP | _TIME_INDEPENDENT | _CLAMP_PROGRESSIVE
    )
    _WEIGHTED_RIGHT, _WEIGHTED_NEXT_LEFT = 0x01000000, 0x02000000
    #: A tangent weight is a 16-bit fraction of this (``0.3333`` = 3333).
    _WEIGHT_DIVIDER = 9999.0
    #: Maya's clamped tangent goes flat beside a key nearer than this in
    #: Maya's internal unit (measured, Maya 2025: 0.049 flat, 0.051 not, at
    #: any key spacing or magnitude) -- radians on a rotation.
    _CLAMP_THRESHOLD = 0.05
    _ANGULAR_PROPERTIES = ("Lcl Rotation",)
    #: ``GlobalSettings/TimeMode`` -> frames per second (``FbxTime::EMode``);
    #: 14 is custom (``CustomFrameRate``).
    _TIME_MODE_FPS = {
        1: 120.0,
        2: 100.0,
        3: 60.0,
        4: 50.0,
        5: 48.0,
        6: 30.0,
        7: 30.0,
        8: 30000 / 1001,
        9: 30000 / 1001,
        10: 25.0,
        11: 24.0,
        12: 1000.0,
        13: 24000 / 1001,
        15: 96.0,
        16: 72.0,
        17: 60000 / 1001,
        18: 120000 / 1001,
    }

    _SCALAR_PROPS = {
        b"Y": ("<h", 2),
        b"C": ("<b", 1),
        b"I": ("<i", 4),
        b"F": ("<f", 4),
        b"D": ("<d", 8),
        b"L": ("<q", 8),
    }
    _ARRAY_PROPS = {
        b"f": ("<f", 4),
        b"d": ("<d", 8),
        b"l": ("<q", 8),
        b"i": ("<i", 4),
        b"b": ("<b", 1),
        b"c": ("<B", 1),
    }

    @classmethod
    def _read_property(
        cls, f, decode_arrays: Union[bool, str], raw_payloads: bool = True
    ) -> Any:
        """One property. *decode_arrays* is True (an array as a list), False
        (a ``("ARRAY", type, count)`` placeholder) or ``"span"`` -- a
        ``("SPAN", type, count, first, last)`` read without building the list,
        ``first``/``last`` ``None`` for an empty array."""
        kind = f.read(1)
        scalar = cls._SCALAR_PROPS.get(kind)
        if scalar:
            fmt, size = scalar
            return struct.unpack(fmt, f.read(size))[0]
        array = cls._ARRAY_PROPS.get(kind)
        if array:
            fmt, size = array
            count, encoding, byte_len = struct.unpack("<III", f.read(12))
            payload = f.read(byte_len)
            if not decode_arrays:
                return ("ARRAY", kind.decode(), count)
            if encoding == 1:
                payload = zlib.decompress(payload)
            if decode_arrays == "span":
                if not count:
                    return ("SPAN", kind.decode(), 0, None, None)
                first = struct.unpack_from(fmt, payload, 0)[0]
                last = struct.unpack_from(fmt, payload, (count - 1) * size)[0]
                return ("SPAN", kind.decode(), count, first, last)
            return list(struct.unpack(f"<{count}{fmt[-1]}", payload))
        if kind in (b"S", b"R"):
            length = struct.unpack("<I", f.read(4))[0]
            if kind == b"R" and not raw_payloads:
                f.seek(length, 1)
                return ("RAW", length)
            return f.read(length)
        raise ValueError(f"Unknown FBX property type {kind!r}")

    @classmethod
    def _read_record(
        cls,
        f,
        wide: bool,
        decode_arrays: Union[bool, Collection[str]],
        raw_payloads: bool = True,
        span_arrays: Collection[str] = (),
    ) -> Optional[Dict[str, Any]]:
        """One node record, or ``None`` at a NULL sentinel.

        *decode_arrays* is a bool for every record, or the names of the records
        whose array properties are inflated (``("KeyTime",)``); *span_arrays*
        names records whose arrays are read as their span alone
        (:meth:`_read_property`)."""
        if wide:
            header = f.read(24)
            if len(header) < 24:
                return None
            end, prop_count, _prop_len = struct.unpack("<QQQ", header)
        else:
            header = f.read(12)
            if len(header) < 12:
                return None
            end, prop_count, _prop_len = struct.unpack("<III", header)
        name_len = struct.unpack("<B", f.read(1))[0]
        name = f.read(name_len).decode("utf-8", "replace")
        if end == 0:
            return None
        decode = (
            decode_arrays if isinstance(decode_arrays, bool) else name in decode_arrays
        )
        if not decode and name in span_arrays:
            decode = "span"
        props = [cls._read_property(f, decode, raw_payloads) for _ in range(prop_count)]
        sentinel = 25 if wide else 13
        children: List[Dict[str, Any]] = []
        while f.tell() < end - sentinel:
            child = cls._read_record(f, wide, decode_arrays, raw_payloads, span_arrays)
            if child is None:
                break
            children.append(child)
        f.seek(end)
        return {"name": name, "props": props, "children": children}

    @staticmethod
    def _display_name(raw: Any) -> Optional[str]:
        """The human half of an FBX object-name property, or ``None``."""
        if not isinstance(raw, (bytes, bytearray)):
            return None
        return bytes(raw).split(_NAME_CLASS_SEPARATOR, 1)[0].decode("utf-8", "replace")

    @staticmethod
    def _merged_span(a: Tuple[Any, ...], b: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Two curves' ``(first, last, count)`` as one channel's: the earlier
        first key, the later last, every key counted. An end a curve was read
        without (``None``) defers to the other's."""
        firsts = [time for time in (a[0], b[0]) if time is not None]
        lasts = [time for time in (a[1], b[1]) if time is not None]
        return (
            min(firsts) if firsts else None,
            max(lasts) if lasts else None,
            a[2] + b[2],
        )

    # ---- curve model ------------------------------------------------------

    @classmethod
    def _clamp_threshold(cls, prop: str) -> float:
        """The value step under which a Maya clamped key on *prop* is flat, in
        the file's unit: radians in Maya, so degrees for a rotation."""
        if prop in cls._ANGULAR_PROPERTIES:
            return math.degrees(cls._CLAMP_THRESHOLD)
        return cls._CLAMP_THRESHOLD

    @classmethod
    def _frame_rate(cls, time_mode: Any, custom: Any = None) -> Optional[float]:
        """Frames per second for a ``TimeMode`` (and ``CustomFrameRate``)."""
        if time_mode == 14 and isinstance(custom, (int, float)) and custom > 0:
            return float(custom)
        return cls._TIME_MODE_FPS.get(time_mode)

    @classmethod
    def _key_curve(
        cls, arrays: Mapping[str, Any], clamp: float = _CLAMP_THRESHOLD
    ) -> _FbxCurve:
        """A :class:`_FbxCurve` from one curve's decoded arrays.

        Attribute runs that do not account for every key (a writer that left
        them out) read as one linear run -- what Blender's importer assumes of
        every key.

        Raises:
            ValueError: The curve has a different number of times and values.
        """
        times = list(arrays.get("KeyTime") or ())
        values = list(arrays.get("KeyValueFloat") or ())
        if len(times) != len(values):
            raise ValueError(
                f"FBX curve has {len(times)} key times and {len(values)} values"
            )
        flags = list(arrays.get("KeyAttrFlags") or ())
        floats = list(arrays.get("KeyAttrDataFloat") or ())
        counts = list(arrays.get("KeyAttrRefCount") or ())
        if (
            len(flags) != len(counts)
            or len(floats) != 4 * len(counts)
            or min(counts, default=0) < 0
            or sum(counts) != len(times)
        ):
            flags, floats, counts = [cls._LINEAR], [0.0] * 4, [len(times)]
        starts: List[int] = []
        at = 0
        for count in counts:
            starts.append(at)
            at += count
        data = [tuple(floats[4 * j : 4 * j + 4]) for j in range(len(counts))]
        return _FbxCurve(times, values, flags, data, starts, clamp)

    @staticmethod
    def _key_attr(curve: _FbxCurve, index: int) -> Tuple[int, Tuple[float, ...]]:
        """``(flags, data)`` of key *index*: the attribute run it falls in."""
        run = bisect_right(curve.starts, index) - 1
        return curve.flags[run], curve.data[run]

    @classmethod
    def _computed(cls, flags: int) -> bool:
        """True for a key whose slopes the importer computes rather than reads.

        Maya writes its spline, clamped and auto keys as cubic keys in AUTO
        tangent mode with ZERO slopes, and its importer recomputes them from
        the neighbouring keys (measured, Maya 2025 / FBX 2020.3.6: a plain
        export re-imports within 1e-7 of the original). A TCB key, which
        neither Maya nor Blender writes, is evaluated as a spline key.
        """
        return (
            (flags & cls._INTERPOLATION) == cls._CUBIC
            and bool(flags & (cls._TANGENT_AUTO | cls._TANGENT_TCB))
            and not flags & cls._TANGENT_USER
        )

    @classmethod
    def _auto_slope(cls, curve: _FbxCurve, index: int, flags: int) -> float:
        """The slope (value per second) Maya computes for a computed key.

        Spline (no clamp bit): the central difference; at an end, the one
        secant. Clamped (``GenericClamp``): spline, but flat beside a
        neighbour nearer than the curve's clamp step. Auto
        (``ClampProgressive``): flat at an end or an extremum, else spline
        capped at three times either side's secant, so no segment overshoots.
        Measured against Maya's own tangents on irregular key sets.
        """
        times, values = curve.times, curve.values
        last = len(times) - 1
        if last < 1:
            return 0.0
        tps = cls.TICKS_PER_SECOND
        if index in (0, last):
            other = 1 if index == 0 else last - 1
            rise, run = values[other] - values[index], times[other] - times[index]
            if flags & cls._CLAMP_PROGRESSIVE or not run:
                return 0.0
            if flags & cls._CLAMP and abs(rise) < curve.clamp:
                return 0.0
            return rise / (run / tps)
        v0, v1, v2 = values[index - 1 : index + 2]
        before, after = times[index] - times[index - 1], times[index + 1] - times[index]
        if not before or not after:
            return 0.0
        left, right = (v1 - v0) / (before / tps), (v2 - v1) / (after / tps)
        spline = (v2 - v0) / ((before + after) / tps)
        if flags & cls._CLAMP_PROGRESSIVE:
            if left * right <= 0:
                return 0.0
            return math.copysign(
                min(abs(spline), 3 * abs(left), 3 * abs(right)), spline
            )
        if flags & cls._CLAMP and (
            abs(v1 - v0) < curve.clamp or abs(v2 - v1) < curve.clamp
        ):
            return 0.0
        return spline

    @classmethod
    def _out_slope(cls, curve: _FbxCurve, index: int) -> float:
        """Key *index*'s right slope, value per second: its ``RightSlope``,
        or computed (:meth:`_computed`)."""
        flags, data = cls._key_attr(curve, index)
        if cls._computed(flags):
            return cls._auto_slope(curve, index, flags)
        return data[0]

    @classmethod
    def _in_slope(cls, curve: _FbxCurve, index: int) -> float:
        """Key *index*'s left slope: computed, or the ``NextLeftSlope`` the key
        BEFORE it stores. A first key has none stored; its right slope stands
        in (an unbroken tangent)."""
        flags, _data = cls._key_attr(curve, index)
        if cls._computed(flags):
            return cls._auto_slope(curve, index, flags)
        if index == 0:
            return cls._out_slope(curve, 0)
        return cls._key_attr(curve, index - 1)[1][1]

    @classmethod
    def _weights(cls, flags: int, data: Sequence[float]) -> Tuple[float, float]:
        """``(right, next left)`` tangent weights, as fractions of the segment
        duration: two 16-bit fractions of 9999 in the bits of the third data
        float (right low), each read only when its weighted bit is set --
        otherwise a third, which makes the cubic a plain Hermite."""
        right = left = 1.0 / 3.0
        if flags & (cls._WEIGHTED_RIGHT | cls._WEIGHTED_NEXT_LEFT):
            low, high = struct.unpack("<HH", struct.pack("<f", data[2]))
            if flags & cls._WEIGHTED_RIGHT:
                right = low / cls._WEIGHT_DIVIDER
            if flags & cls._WEIGHTED_NEXT_LEFT:
                left = high / cls._WEIGHT_DIVIDER
        return right, left

    @classmethod
    def _evaluate_curve(cls, curve: _FbxCurve, ticks: int) -> float:
        """The curve's value at *ticks*, as Maya's importer evaluates it.

        Each key's flags shape the segment AFTER it: constant (or step-next),
        linear, or cubic -- a Bezier whose control points sit a weight's
        fraction of the segment in from each key along its slope (a third,
        unweighted: a Hermite). Before the first key and after the last the
        curve holds its end value. Pinned against Maya's own evaluation of a
        Maya-written file (``test_assets/fbx/maya_tangents.fbx``: every tangent
        type within 1e-7). Velocity bits, which no DCC here writes, are ignored.
        """
        times, values = curve.times, curve.values
        if not times:
            return 0.0
        if ticks <= times[0]:
            return values[0]
        if ticks >= times[-1]:
            return values[-1]
        i = bisect_right(times, ticks) - 1
        t0, t1 = times[i], times[i + 1]
        v0, v1 = values[i], values[i + 1]
        if ticks == t0:
            return v0
        flags, data = cls._key_attr(curve, i)
        mode = flags & cls._INTERPOLATION
        if mode == cls._CONSTANT:
            return v1 if flags & cls._CONSTANT_NEXT else v0
        u = (ticks - t0) / (t1 - t0)
        if mode != cls._CUBIC:
            return v0 + (v1 - v0) * u
        span = (t1 - t0) / cls.TICKS_PER_SECOND
        right, left = cls._weights(flags, data)
        y1 = v0 + cls._out_slope(curve, i) * right * span
        y2 = v1 - cls._in_slope(curve, i + 1) * left * span
        if flags & (cls._WEIGHTED_RIGHT | cls._WEIGHTED_NEXT_LEFT):
            # Time control points as fractions of the segment, in [0, 1].
            u = KeyCurve.bezier_parameter(u, 0.0, right, 1.0 - left, 1.0)
        a = 1.0 - u
        return a * a * a * v0 + 3.0 * a * u * (a * y1 + u * y2) + u * u * u * v1

    @staticmethod
    def _values_match(a: float, b: float, tolerance: float) -> bool:
        """Equal within *tolerance*: absolute below 1, relative above -- a
        float32 key carries about seven significant digits."""
        return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))

    @classmethod
    def _disagreement(
        cls, a: _FbxCurve, b: _FbxCurve, tolerance: float
    ) -> Optional[Tuple[int, float, float]]:
        """The earliest key, of either curve, inside their overlap where the
        other evaluates differently: ``(ticks, a's value, b's value)``, or
        ``None``. Curves that only touch compare that one instant -- a seam
        agrees when both hold one value there."""
        if not a.times or not b.times:
            return None
        lo, hi = max(a.times[0], b.times[0]), min(a.times[-1], b.times[-1])
        if lo > hi:
            return None
        found = None
        for mine, other, flip in ((a, b, False), (b, a, True)):
            for i in range(bisect_left(mine.times, lo), bisect_right(mine.times, hi)):
                at, own = mine.times[i], mine.values[i]
                theirs = cls._evaluate_curve(other, at)
                if not cls._values_match(own, theirs, tolerance):
                    hit = (at, theirs, own) if flip else (at, own, theirs)
                    if found is None or at < found[0]:
                        found = hit
                    break
        return found

    @classmethod
    def _build_take_model(
        cls,
        rows: Sequence[Tuple[str, int, str, Optional[Mapping[str, Any]]]],
        links: Sequence[Tuple[str, Any, Any, Optional[str]]],
    ) -> Dict[str, Any]:
        """Every take's channels and curves, from a file's objects and wiring.

        Parameters:
            rows: ``(record name, uid, display name, arrays)`` per object, in
                file order; *arrays* holds an ``AnimationCurve``'s decoded
                :attr:`FbxFile.CURVE_ARRAYS`, ``None`` for anything else.
            links: ``(kind, child, parent, property)`` per connection.

        Returns:
            ``{"takes": [...], "labels": {channel: label}}``. Each take (file
            order): ``{"name", "stack", "layers": [layer uid per index],
            "channels": {channel: {"curve": _FbxCurve, "id", "node",
            "layer"}}, "nodes": {(target, property, layer index): node}}``. A
            channel is ``(target uid, property, component, layer index)`` --
            layers matched by their order in the stack, the one identity both
            Maya (scene layers by name) and Blender (a layer named after its
            take) share. Its label is ``target.property.component``, the
            target ``name#uid`` where names repeat, `` [layer i]`` past the
            base layer. A curve node without curves (a still channel) has no
            channel: nothing keys it.
        """
        kind: Dict[Any, str] = {}
        name: Dict[Any, str] = {}
        arrays: Dict[Any, Mapping[str, Any]] = {}
        stacks: List[Any] = []
        for record, uid, display, curve_arrays in rows:
            kind[uid], name[uid] = record, display
            if record == "AnimationStack":
                stacks.append(uid)
            if curve_arrays is not None:
                arrays[uid] = curve_arrays
        members: Dict[Any, List[Tuple[Any, Optional[str]]]] = {}
        targets: Dict[Any, List[Tuple[Any, str]]] = {}
        for _kind, child, parent, prop in links:
            members.setdefault(parent, []).append((child, prop))
            if prop is not None and kind.get(child) == "AnimationCurveNode":
                targets.setdefault(child, []).append((parent, prop))

        takes: List[Dict[str, Any]] = []
        for stack in stacks:
            layers = [
                c for c, _p in members.get(stack, ()) if kind.get(c) == "AnimationLayer"
            ]
            channels: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
            nodes: Dict[Tuple[Any, ...], Any] = {}
            for index, layer in enumerate(layers):
                for node, _p in members.get(layer, ()):
                    if kind.get(node) != "AnimationCurveNode":
                        continue
                    for target, prop in targets.get(node, ()):
                        nodes.setdefault((target, prop, index), node)
                        clamp = cls._clamp_threshold(prop)
                        for curve, component in members.get(node, ()):
                            key = (target, prop, component, index)
                            if (
                                component is None
                                or curve not in arrays
                                or key in channels
                            ):
                                continue
                            channels[key] = {
                                "curve": cls._key_curve(arrays[curve], clamp),
                                "id": curve,
                                "node": node,
                                "layer": index,
                            }
            takes.append(
                {
                    "name": name[stack],
                    "stack": stack,
                    "layers": layers,
                    "channels": channels,
                    "nodes": nodes,
                }
            )

        used = {key[0] for take in takes for key in take["channels"]}
        repeats = Counter(name.get(target, "") for target in used)
        labels: Dict[Tuple[Any, ...], str] = {}
        for take in takes:
            for key in take["channels"]:
                if key in labels:
                    continue
                target, prop, component, index = key
                label = name.get(target, "")
                if repeats[label] > 1:
                    label = f"{label}#{target}"
                label = f"{label}.{prop}.{component}"
                labels[key] = label + (f" [layer {index}]" if index else "")
        return {"takes": takes, "labels": labels}

    @staticmethod
    def _take_extent(take: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
        """``(first, last)`` key over every curve of *take*, in ticks."""
        curves = [c["curve"].times for c in take["channels"].values()]
        firsts = [times[0] for times in curves if times]
        if not firsts:
            return None
        return min(firsts), max(times[-1] for times in curves if times)

    @classmethod
    def _when(cls, ticks: int, fps: Optional[float]) -> str:
        """*ticks* for a message: a frame at *fps*, else seconds."""
        seconds = ticks / cls.TICKS_PER_SECOND
        if fps:
            return f"frame {round(seconds * fps, 3):g}"
        return f"{seconds:g} s"

    @classmethod
    def _first_conflict(
        cls,
        take: Mapping[str, Any],
        others: Sequence[Mapping[str, Any]],
        labels: Mapping[Tuple[Any, ...], str],
        tolerance: float,
        fps: Optional[float],
    ) -> Optional[str]:
        """Why *take* cannot join *others*: the earliest overlap, over the
        channels they share, where a key of one evaluates differently on the
        other -- ``"'Run' disagrees with 'Walk' on cube.Lcl
        Translation.d|X at frame 1: 0.75 vs 0.5"`` -- or ``None``."""
        worst = None
        for other in others:
            for key, channel in take["channels"].items():
                theirs = other["channels"].get(key)
                if theirs is None:
                    continue
                hit = cls._disagreement(channel["curve"], theirs["curve"], tolerance)
                if hit is not None and (worst is None or hit[0] < worst[0]):
                    worst = (hit[0], hit[1], hit[2], other["name"], labels[key])
        if worst is None:
            return None
        at, mine, theirs, other, label = worst
        return (
            f"{take['name']!r} disagrees with {other!r} on {label} at "
            f"{cls._when(at, fps)}: {mine:.7g} vs {theirs:.7g}"
        )

    @staticmethod
    def _adds_keys(take: Mapping[str, Any], kept: Sequence[Mapping[str, Any]]) -> bool:
        """True when *take* keys a channel at a time no curve of *kept* spans
        (a key on a channel none of them animates included)."""
        for key, channel in take["channels"].items():
            times = channel["curve"].times
            if not times:
                continue
            spans = sorted(
                (
                    k["channels"][key]["curve"].times[0],
                    k["channels"][key]["curve"].times[-1],
                )
                for k in kept
                if key in k["channels"] and k["channels"][key]["curve"].times
            )
            inside = 0
            lo = hi = None
            for first, last in spans + [(None, None)]:
                if lo is not None and (first is None or first > hi):
                    inside += bisect_right(times, hi) - bisect_left(times, lo)
                    lo = None
                if first is None:
                    break
                if lo is None:
                    lo, hi = first, last
                else:
                    hi = max(hi, last)
            if inside < len(times):
                return True
        return False

    @classmethod
    def _layout(
        cls,
        model: Mapping[str, Any],
        current: Optional[str],
        tolerance: float,
        fps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """:meth:`FbxFile.take_layout` of a :meth:`_build_take_model`."""
        takes = list(model["takes"])
        names = [take["name"] for take in takes]
        extents = [cls._take_extent(take) for take in takes]
        tps = cls.TICKS_PER_SECOND
        rows = [
            {
                "name": take["name"],
                "index": index,
                "start": None if extent is None else extent[0] / tps,
                "end": None if extent is None else extent[1] / tps,
            }
            for index, (take, extent) in enumerate(zip(takes, extents), 1)
        ]
        keyed = [i for i, extent in enumerate(extents) if extent is not None]
        seed = next(
            (i for i in keyed if names[i] == current), keyed[0] if keyed else None
        )

        accepted: List[int] = [] if seed is None else [seed]
        stash: List[int] = []
        conflicts: Dict[str, str] = {}
        for i in keyed:
            if i == seed:
                continue
            reason = cls._first_conflict(
                takes[i], [takes[j] for j in accepted], model["labels"], tolerance, fps
            )
            if reason is None:
                accepted.append(i)
            else:
                stash.append(i)
                conflicts[names[i]] = reason

        # The widest takes first, so a cover keeps its place and its slices
        # add nothing; then any kept take the rest now cover goes too.
        def coverage(i: int) -> Tuple[Any, ...]:
            curves = [c["curve"].times for c in takes[i]["channels"].values()]
            spread = sum(t[-1] - t[0] for t in curves if t)
            return (-spread, -sum(len(t) for t in curves), i != seed, i)

        live: List[int] = []
        for i in sorted(accepted, key=coverage):
            if cls._adds_keys(takes[i], [takes[j] for j in live]):
                live.append(i)
        for i in list(reversed(live)):
            rest = [takes[j] for j in live if j != i]
            if rest and not cls._adds_keys(takes[i], rest):
                live.remove(i)
        live.sort()
        redundant = [i for i in range(len(takes)) if i not in live and i not in stash]

        if not live:
            kind = "none"
        elif stash:
            kind = "conflicted"
        elif len(live) > 1:
            kind = "merged"
        elif redundant:
            kind = "covered"
        else:
            kind = "single"
        return {
            "takes": rows,
            "current": current if current in names else None,
            "kind": kind,
            "live": [names[i] for i in live],
            "redundant": [names[i] for i in redundant],
            "stash": [names[i] for i in sorted(stash)],
            "conflicts": conflicts,
        }

    @classmethod
    def _union_curves(
        cls, curves: Sequence[_FbxCurve]
    ) -> Tuple[
        List[int], List[float], List[int], List[Tuple[float, ...]], List[int], int
    ]:
        """ONE curve holding every key of *curves* (in take order) at its time.

        A time several curves key is kept once -- the key of a curve that
        continues past it, since a key's flags and slopes describe the segment
        AFTER it -- and counted as a seam. Each key keeps its own flags and
        data; where the next key comes from another curve (a gap between two
        takes' keys), the left side of that next key is that curve's own: its
        stored left slope and weight, or its right slope at a first key (an
        unbroken tangent). Across a gap the earlier take's last key still
        decides the interpolation -- a constant last key holds its pose until
        the next take starts.

        A key whose slopes the importer computes from its neighbours (Maya's
        spline, clamped and auto) and whose neighbours the union changed -- a
        take's last key and the next take's first, ends alone and inner keys
        merged -- is written with the slopes it plays in its own curve
        (:meth:`_keep_own_tangents`), so each take plays its own segments.

        Curves that follow one another (a Shots Only file: each take its own
        window, at most a shared seam frame between two) are joined run by
        run; overlapping ones key by key.

        Returns:
            ``(times, values, flags, data, counts, seams)`` -- the keys, their
            attribute runs (``KeyAttrRefCount``'s encoding, each curve's own
            runs kept as stored) and how many keys a shared time collapsed.
        """
        keyed = sorted(
            (o for o, curve in enumerate(curves) if curve.times),
            key=lambda o: (curves[o].times[0], o),
        )
        # A seam drops the earlier curve's key there, so a one-key curve on
        # either side of one is joined key by key: run by run, a gap before a
        # dropped lone key met that key's slope, not the kept one's.
        follows = all(
            curves[a].times[-1] < curves[b].times[0]
            or (
                curves[a].times[-1] == curves[b].times[0]
                and len(curves[a].times) > 1
                and len(curves[b].times) > 1
            )
            for a, b in zip(keyed, keyed[1:])
        )
        if not follows:
            return cls._interleave_curves(curves)
        times: List[int] = []
        values: List[float] = []
        flags: List[int] = []
        data: List[Tuple[float, ...]] = []
        counts: List[int] = []
        seams = 0
        # Per key: the (curve, index) its left side and its right side come
        # from -- one key, but a seam's left is the dropped key's.
        sources: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        seamed: Optional[Tuple[int, int]] = None

        def run(f: int, d: Tuple[float, ...], count: int, join: bool) -> None:
            if not count:
                return
            if join and counts and flags[-1] == f and data[-1] == d:
                counts[-1] += count
            else:
                flags.append(f)
                data.append(d)
                counts.append(count)

        for position, o in enumerate(keyed):
            curve = curves[o]
            nxt = keyed[position + 1] if position + 1 < len(keyed) else None
            seam = nxt is not None and curve.times[-1] == curves[nxt].times[0]
            keep = len(curve.times) - (1 if seam else 0)
            seams += 1 if seam else 0
            times.extend(curve.times[:keep])
            values.extend(curve.values[:keep])
            sources.extend(((o, i), (o, i)) for i in range(keep))
            if seamed is not None and keep:
                sources[len(sources) - keep] = (seamed, (o, 0))
            seamed = (o, len(curve.times) - 1) if seam else None
            gap = nxt is not None and not seam
            ends = curve.starts[1:] + [len(curve.times)]
            first = True
            for f, d, start, end in zip(curve.flags, curve.data, curve.starts, ends):
                end = min(end, keep)
                if gap and end == keep:
                    end -= 1  # the last key is re-pointed below
                run(f, d, max(0, end - start), first)
                first = first and end <= start
            if gap:
                f, d = cls._junction(*cls._key_attr(curve, keep - 1), nxt, 0, curves)
                run(f, d, 1, True)
        keyed_flags = [f for f, n in zip(flags, counts) for _ in range(n)]
        keyed_data = [d for d, n in zip(data, counts) for _ in range(n)]
        if cls._keep_own_tangents(
            curves, sources, times, values, keyed_flags, keyed_data
        ):
            return (times, values, *cls._attr_runs(keyed_flags, keyed_data), seams)
        return times, values, flags, data, counts, seams

    @classmethod
    def _interleave_curves(
        cls, curves: Sequence[_FbxCurve]
    ) -> Tuple[
        List[int], List[float], List[int], List[Tuple[float, ...]], List[int], int
    ]:
        """:meth:`_union_curves` of curves that overlap, key by key."""
        entries = sorted(
            (t, order, i)
            for order, curve in enumerate(curves)
            for i, t in enumerate(curve.times)
        )
        picked: List[Tuple[int, int]] = []
        groups: List[List[Tuple[int, int]]] = []
        seams = 0
        at = 0
        while at < len(entries):
            stop = at
            while stop < len(entries) and entries[stop][0] == entries[at][0]:
                stop += 1
            group = entries[at:stop]
            seams += len(group) - 1
            picked.append(
                next(
                    ((o, i) for _t, o, i in group if i + 1 < len(curves[o].times)),
                    (group[0][1], group[0][2]),
                )
            )
            groups.append([(o, i) for _t, o, i in group])
            at = stop
        times = [curves[o].times[i] for o, i in picked]
        values = [curves[o].values[i] for o, i in picked]
        # A key's left side: the picked key's own, else that of a key at its
        # time whose curve runs on from the key before it (a seam's left).
        sources = []
        for p, (pick, group) in enumerate(zip(picked, groups)):
            left = pick
            if p and not cls._follows_on(curves, pick, times[p - 1]):
                left = next(
                    (k for k in group if cls._follows_on(curves, k, times[p - 1])),
                    pick,
                )
            sources.append((left, pick))
        flags: List[int] = []
        data: List[Tuple[float, ...]] = []
        for p, (o, i) in enumerate(picked):
            curve = curves[o]
            key_flags, key_data = cls._key_attr(curve, i)
            following = times[p + 1] if p + 1 < len(times) else None
            joined = i + 1 < len(curve.times) and curve.times[i + 1] == following
            if following is not None and not joined:
                key_flags, key_data = cls._junction(
                    key_flags, key_data, *picked[p + 1], curves
                )
            flags.append(key_flags)
            data.append(key_data)
        cls._keep_own_tangents(curves, sources, times, values, flags, data)
        return (times, values, *cls._attr_runs(flags, data), seams)

    @staticmethod
    def _follows_on(
        curves: Sequence[_FbxCurve], key: Tuple[int, int], prev: int
    ) -> bool:
        """True when *key* ``(curve, index)``'s own previous key sits at *prev*."""
        order, index = key
        return index > 0 and curves[order].times[index - 1] == prev

    @classmethod
    def _keep_own_tangents(
        cls,
        curves: Sequence[_FbxCurve],
        sources: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
        times: Sequence[int],
        values: Sequence[float],
        flags: List[int],
        data: List[Tuple[float, ...]],
    ) -> bool:
        """Write out the slopes of every computed key a union moved; True when
        any was.

        *flags* / *data* are the union's per-key attributes, edited in place;
        *sources* names each key's left and right ``(curve, index)``. A key
        the importer computes (:meth:`_computed`) is solved from its
        neighbours, so where the union gave it others -- a take's end key
        gains the next take's first as one -- it would play another tangent
        than in its own curve. Such a key becomes a user tangent carrying the
        slopes it plays there: the right one as its ``RightSlope``, the left
        as the previous key's ``NextLeftSlope``, broken where the two differ
        (a seam whose sides come from two takes). A key the importer reads as
        stored is left alone, and so is a computed key whose neighbours are
        its own.
        """
        moved = False
        last = len(times) - 1

        def same(curve: _FbxCurve, own: int, merged: int) -> bool:
            inside = 0 <= own < len(curve.times)
            if inside != (0 <= merged <= last):
                return False
            return not inside or (
                curve.times[own] == times[merged]
                and curve.values[own] == values[merged]
            )

        for p, (left, right) in enumerate(sources):
            curve_in, curve_out = curves[left[0]], curves[right[0]]
            if not (
                cls._computed(cls._key_attr(curve_in, left[1])[0])
                or cls._computed(cls._key_attr(curve_out, right[1])[0])
            ):
                continue
            if (
                left == right
                and same(curve_in, left[1] - 1, p - 1)
                and same(curve_out, right[1] + 1, p + 1)
            ):
                continue
            moved = True
            slope_in = cls._in_slope(curve_in, left[1])
            slope_out = cls._out_slope(curve_out, right[1])
            f, d = flags[p], data[p]
            if (f & cls._INTERPOLATION) == cls._CUBIC:
                broken = struct.pack("<f", slope_in) != struct.pack("<f", slope_out)
                f &= ~(cls._COMPUTED_BITS | cls._TANGENT_BREAK)
                f |= cls._TANGENT_USER | (cls._TANGENT_BREAK if broken else 0)
                d = (slope_out, d[1], d[2], d[3])
            flags[p], data[p] = f, d
            if p:
                before = data[p - 1]
                data[p - 1] = (before[0], slope_in, before[2], before[3])
        return moved

    @classmethod
    def _junction(
        cls,
        flags: int,
        data: Tuple[float, ...],
        order: int,
        index: int,
        curves: Sequence[_FbxCurve],
    ) -> Tuple[int, Tuple[float, ...]]:
        """A key's attributes re-pointed at a next key from another curve:
        the left slope, weight and weighted bit become that key's own."""
        nxt = curves[order]
        if index > 0:
            left_flags, left_data = cls._key_attr(nxt, index - 1)
            slope = left_data[1]
            weighted = left_flags & cls._WEIGHTED_NEXT_LEFT
            left_word = struct.unpack("<HH", struct.pack("<f", left_data[2]))[1]
        else:
            slope = cls._key_attr(nxt, index)[1][0]
            weighted = 0
            left_word = int(cls._WEIGHT_DIVIDER / 3)
        right_word = struct.unpack("<HH", struct.pack("<f", data[2]))[0]
        word = struct.unpack("<f", struct.pack("<HH", right_word, left_word))[0]
        flags = (flags & ~cls._WEIGHTED_NEXT_LEFT) | weighted
        return flags, (data[0], slope, word, data[3])

    @staticmethod
    def _attr_runs(
        flags: Sequence[int], data: Sequence[Tuple[float, ...]]
    ) -> Tuple[List[int], List[Tuple[float, ...]], List[int]]:
        """Per-key attributes as ``KeyAttrRefCount`` runs: ``(flags, data,
        counts)``, consecutive identical keys sharing one run."""
        run_flags: List[int] = []
        run_data: List[Tuple[float, ...]] = []
        counts: List[int] = []
        for f, d in zip(flags, data):
            if counts and run_flags[-1] == f and run_data[-1] == d:
                counts[-1] += 1
            else:
                run_flags.append(f)
                run_data.append(d)
                counts.append(1)
        return run_flags, run_data, counts


class FbxFile(_FbxFileInternal):
    """A parsed binary FBX, held read-only.

    Example:
        >>> fbx = FbxFile.load("asset.fbx")
        >>> fbx.version
        7700
        >>> fbx.take_names()
        ['Shot_1', 'Shot_2']
        >>> fbx.objects_census()["AnimationCurveNode"]
        27989
    """

    def __init__(self, path: str, version: int, roots: List[Dict[str, Any]]):
        self.path = path
        self.version = version
        self.roots = roots
        self._by_name = {r["name"]: r for r in roots}

    @classmethod
    def load(
        cls,
        path: str,
        decode_arrays: Union[bool, Collection[str]] = False,
        raw_payloads: bool = True,
        span_arrays: Collection[str] = (),
    ) -> "FbxFile":
        """Parse *path*.

        Parameters:
            path: Binary FBX file.
            decode_arrays: When False (default), array properties are read as
                ``("ARRAY", type_char, count)`` placeholders and their
                payloads skipped — the fast census mode. True inflates them
                (zlib where encoded) into python lists. A collection of record
                names inflates only those records' arrays.
            raw_payloads: When False, ``R`` (raw binary) properties — the
                embedded media, hundreds of MB on a textured export — are
                skipped as ``("RAW", length)`` placeholders instead of being
                copied into memory. A census that never looks at the bytes
                (node and take counts) has no reason to hold them.
            span_arrays: Names of records whose arrays are read as their span
                alone -- ``("SPAN", type, count, first, last)``, no list built.
                ``("KeyTime",)`` is what :meth:`take_curves` reads: every
                curve's extent, at a fraction of the memory a full decode of a
                production file's per-frame keys costs.

        Raises:
            ValueError: Not a binary FBX, or truncated or corrupt: a record
                that runs past the end of the file, or data that cannot be
                decoded.
        """
        with open(path, "rb") as f:
            magic = f.read(len(FBX_MAGIC))
            if magic != FBX_MAGIC:
                raise ValueError(f"Not a binary FBX: {path}")
            size = os.fstat(f.fileno()).st_size
            # A cut or corrupt file is this method's documented refusal, never
            # an exception no caller catches: a short read the struct or zlib
            # decode chokes on (``zlib.error`` reached past every caller's
            # ``(OSError, ValueError)``, a conversion's timeout budget among
            # them), or -- where nothing is decoded -- a record that ends past
            # the file, whose remains read as a whole file otherwise.
            try:
                version = struct.unpack("<I", f.read(4))[0]
                wide = version >= 7500
                roots: List[Dict[str, Any]] = []
                while True:
                    record = cls._read_record(
                        f, wide, decode_arrays, raw_payloads, span_arrays
                    )
                    if record is None:
                        break
                    if f.tell() > size:
                        raise ValueError(
                            f"record {record['name']!r} runs past the end of the file"
                        )
                    roots.append(record)
            except (struct.error, zlib.error, ValueError) as error:
                raise ValueError(
                    f"Truncated or corrupt FBX: {path}: {error}"
                ) from error
        return cls(path, version, roots)

    @staticmethod
    def is_fbx(path: str) -> bool:
        """True when *path* starts with the binary-FBX magic."""
        try:
            with open(path, "rb") as f:
                return f.read(len(FBX_MAGIC)) == FBX_MAGIC
        except OSError:
            return False

    # ---- navigation -------------------------------------------------------

    def section(self, name: str) -> Optional[Dict[str, Any]]:
        """Top-level record *name* (``"Objects"``, ``"Connections"`` …)."""
        return self._by_name.get(name)

    def iter_objects(self) -> Iterator[Dict[str, Any]]:
        """Yield every child record of the ``Objects`` section."""
        objects = self.section("Objects")
        if objects:
            yield from objects["children"]

    # ---- census -----------------------------------------------------------

    def objects_census(self) -> Dict[str, int]:
        """``{record name: count}`` over the Objects section.

        Keys are FBX record names — ``Model``, ``Geometry``,
        ``AnimationStack``, ``AnimationLayer``, ``AnimationCurveNode``,
        ``AnimationCurve``, ``Deformer`` … — which is the census a
        deliverable check compares run over run.
        """
        return dict(Counter(record["name"] for record in self.iter_objects()))

    def object_names(self, kind: str) -> List[str]:
        """Display names of every Objects child whose record name is *kind*."""
        names: List[str] = []
        for record in self.iter_objects():
            if record["name"] != kind:
                continue
            name = self.object_name(record)
            if name is not None:
                names.append(name)
        return names

    def object_name(self, record: Dict[str, Any]) -> Optional[str]:
        """An Objects child's display name (a record :meth:`iter_objects` yields):
        the first of its props that is one."""
        return next(
            (n for n in map(self._display_name, record["props"]) if n is not None),
            None,
        )

    def take_names(self) -> List[str]:
        """Animation take names — the ``AnimationStack`` display names."""
        return self.object_names("AnimationStack")

    def materials(self) -> List[Dict[str, Any]]:
        """Every ``Material`` as written: name, shading model, properties and
        the file bound to each slot.

        A material's slots are its ``Properties70`` names (``DiffuseColor``,
        StingrayPBS's ``Maya|TEX_color_map``); a texture binds to one through
        an object-property connection -- a ``LayeredTexture`` there stands for
        its first layer, the one FBX2glTF reads -- and its file is named by
        the texture's own ``RelativeFilename`` / ``FileName``, else by the
        ``Video`` it is connected under -- folder-less, the spelling a
        delivered file embeds its media under (:meth:`FbxMedia.embedded`).

        Returns:
            ``[{"name", "shading_model", "props", "textures"}]`` in file
            order: ``shading_model`` lowercased (``"phong"``, ``"lambert"``,
            ``"unknown"`` for a host's own model), ``props`` ``{name: value}``
            -- one value as itself, several as a tuple, text decoded --
            and ``textures`` ``{slot: file name}``.
        """
        objects: Dict[Any, Dict[str, Any]] = {}
        for record in self.iter_objects():
            if record["props"]:
                objects[record["props"][0]] = record

        def text(raw: Any) -> Any:
            if isinstance(raw, (bytes, bytearray)):
                return bytes(raw).decode("utf-8", "replace")
            return raw

        def file_of(record: Dict[str, Any]) -> str:
            for wanted in ("RelativeFilename", "FileName", "Filename"):
                for child in record["children"]:
                    if child["name"] == wanted and child["props"]:
                        name = os.path.basename(
                            str(text(child["props"][0])).replace("\\", "/")
                        )
                        if name:
                            return name
            return ""

        videos: Dict[Any, str] = {}
        layers: Dict[Any, Any] = {}
        slots: Dict[Any, Dict[str, Any]] = {}
        for kind, child, parent, prop in self.connections():
            source, target = objects.get(child), objects.get(parent)
            if source is None or target is None:
                continue
            if source["name"] == "Video" and target["name"] == "Texture":
                videos[parent] = file_of(source)
            elif source["name"] == "Texture" and target["name"] == "LayeredTexture":
                layers.setdefault(parent, child)
            elif (
                source["name"] in ("Texture", "LayeredTexture")
                and target["name"] == "Material"
            ):
                if kind == "OP" and prop:
                    slots.setdefault(parent, {})[prop] = child

        found: List[Dict[str, Any]] = []
        for uid, record in objects.items():
            if record["name"] != "Material":
                continue
            shading = ""
            props: Dict[str, Any] = {}
            for child in record["children"]:
                if child["name"] == "ShadingModel" and child["props"]:
                    shading = str(text(child["props"][0])).lower()
                elif child["name"] == "Properties70":
                    for row in child["children"]:
                        values = row["props"]
                        if not values:
                            continue
                        stored = [text(v) for v in values[4:]]
                        props[str(text(values[0]))] = (
                            stored[0] if len(stored) == 1 else tuple(stored)
                        )
            textures = {}
            for slot, texture in slots.get(uid, {}).items():
                texture = layers.get(texture, texture)
                bound = objects.get(texture)
                if bound is None or bound["name"] != "Texture":
                    continue
                name = file_of(bound) or videos.get(texture, "")
                if name:
                    textures[slot] = name
            found.append(
                {
                    "name": self.object_name(record),
                    "shading_model": shading,
                    "props": props,
                    "textures": textures,
                }
            )
        return found

    def user_properties(
        self, name: str, kind: str = "Model", named: bool = False
    ) -> List[Any]:
        """Every value a *kind* object carries for the user property *name*.

        User properties live in each object's ``Properties70`` as ``P``
        records -- ``(name, type, label, flags, value...)`` -- which is where a
        DCC's custom attributes (the ``data_export`` channels among them) land.
        Values come back as stored: ``bytes`` for a string property, numbers
        for a numeric one, in file order. *named* pairs each with the display
        name of the object carrying it -- ``(object name, value)`` -- which
        tells the scene's own ``data_export`` from a referenced module's
        ``NS:data_export``.
        """
        found: List[Any] = []
        wanted = name.encode("utf-8")
        for record in self.iter_objects():
            if record["name"] != kind:
                continue
            owner = self.object_name(record) if named else None
            for child in record["children"]:
                if child["name"] != "Properties70":
                    continue
                for prop in child["children"]:
                    props = prop["props"]
                    if len(props) >= 5 and props[0] == wanted:
                        value = props[4] if len(props) == 5 else props[4:]
                        found.append((owner, value) if named else value)
        return found

    def user_property_items(self, kind: str = "Model") -> List[Tuple[str, str, Any]]:
        """``(object name, property name, value)`` for every user property (a
        ``P`` record flagged ``U``) a *kind* object carries, in file order and
        in one pass -- what :meth:`user_properties` reads by name, listed;
        values as it returns them."""
        found: List[Tuple[str, str, Any]] = []
        for record in self.iter_objects():
            if record["name"] != kind:
                continue
            owner = self.object_name(record)
            for child in record["children"]:
                if child["name"] != "Properties70":
                    continue
                for prop in child["children"]:
                    props = prop["props"]
                    flags = props[3] if len(props) > 3 else b""
                    if (
                        len(props) >= 5
                        and isinstance(flags, (bytes, bytearray))
                        and b"U" in flags
                    ):
                        name = bytes(props[0]).decode("utf-8", "replace")
                        value = props[4] if len(props) == 5 else props[4:]
                        found.append((owner, name, value))
        return found

    #: The ``AnimationCurve`` arrays :meth:`take_layout` reads. Load with
    #: ``decode_arrays=FbxFile.CURVE_ARRAYS`` to have them at hand; a file
    #: loaded without them is read again for them.
    CURVE_ARRAYS = (
        "KeyTime",
        "KeyValueFloat",
        "KeyAttrFlags",
        "KeyAttrDataFloat",
        "KeyAttrRefCount",
    )

    def take_curves(self) -> Dict[str, Dict[Tuple[str, str, str], Tuple[Any, ...]]]:
        """What each take animates: ``{take: {(target, property, channel):
        (first_key, last_key, key_count)}}``, times in FBX ticks
        (:attr:`TICKS_PER_SECOND`).

        The census a take split is judged by. A channel the whole-timeline take
        animates but a shot's take does not is a node that plays its rest pose
        for that whole shot -- what Maya's ``FBXExportSplitAnimationIntoTakes``
        makes of every curve with no key inside a take, unless every curve is
        resampled. *target* is the animated object's name (a ``Model``, or the
        object a custom property lives on) -- ``name#uid`` where more than one
        animated object carries that name, *property* the property animated
        (``"Lcl Translation"``), *channel* the curve's component (``"d|X"``).
        A channel keyed on more than one layer is their union.

        Reads the ``KeyTime`` arrays: load with ``span_arrays=("KeyTime",)``
        (their extent alone -- what this needs) or ``decode_arrays``. A curve
        read without either reports ``(None, None, count)``.
        """
        kind: Dict[Any, str] = {}
        name: Dict[Any, str] = {}
        spans: Dict[Any, Tuple[Any, ...]] = {}
        for record in self.iter_objects():
            props = record["props"]
            if not props or not isinstance(props[0], int):
                continue
            kind[props[0]] = record["name"]
            name[props[0]] = (
                self._display_name(props[1]) if len(props) > 1 else None
            ) or ""
            if record["name"] != "AnimationCurve":
                continue
            times = next(
                (
                    child["props"][0]
                    for child in record["children"]
                    if child["name"] == "KeyTime" and child["props"]
                ),
                None,
            )
            if isinstance(times, list):
                spans[props[0]] = (
                    (times[0], times[-1], len(times)) if times else (None, None, 0)
                )
            elif isinstance(times, tuple) and len(times) == 5:  # SPAN
                spans[props[0]] = (times[3], times[4], times[2])
            elif isinstance(times, tuple) and len(times) == 3:  # ("ARRAY", t, n)
                spans[props[0]] = (None, None, times[2])
            else:
                spans[props[0]] = (None, None, 0)

        parents: Dict[Any, list] = {}
        for _kind, child, parent, prop in self.connections():
            parents.setdefault(child, []).append((parent, prop))

        found: List[Tuple[str, Any, str, str, Tuple[Any, ...]]] = []
        for curve, span in spans.items():
            for node, channel in parents.get(curve, ()):
                if kind.get(node) != "AnimationCurveNode":
                    continue
                links = parents.get(node, ())
                stacks = dict.fromkeys(
                    stack
                    for layer, _ in links
                    if kind.get(layer) == "AnimationLayer"
                    for stack, _ in parents.get(layer, ())
                    if kind.get(stack) == "AnimationStack"
                )
                targets = [(t, prop) for t, prop in links if prop is not None]
                for stack in stacks:
                    for target, prop in targets:
                        found.append((name[stack], target, prop, channel or "", span))

        # Keyed by name alone, one target's curve replaced another's: Maya
        # writes every NodeAttribute nameless, and a referenced module's nodes
        # beside the scene's own short names (478 repeats in one 12-shot
        # production FBX) -- and ``take_spans`` measured whichever was written
        # last. A uid is stable across the takes of one file, the comparison
        # the take gate makes.
        named = Counter(name.get(target, "") for target in {f[1] for f in found})
        takes: Dict[str, Dict[Tuple[str, str, str], Tuple[Any, ...]]] = {}
        for take, target, prop, channel, span in found:
            label = name.get(target, "")
            if named[label] > 1:
                label = f"{label}#{target}"
            curves = takes.setdefault(take, {})
            key = (label, prop, channel)
            curves[key] = (
                self._merged_span(curves[key], span) if key in curves else span
            )
        return takes

    def take_spans(self) -> Dict[str, Tuple[float, float]]:
        """Each take's ``(first, last)`` key in SECONDS, over every curve in it.

        What FBX2glTF sizes a take by: the key extent of all its curves -- a
        camera attribute's or a custom property's as much as a transform's --
        whatever span the take itself declares, and the first of them is what
        it rebases the clip onto (measured on 0.13.1, 2026-10-04: a focal
        length keyed at 3 opened a take whose transforms start at 10, and the
        take's declared 1-120 was ignored). Needs the key times
        (:meth:`take_curves`); a take none of whose curves carries them is
        left out rather than guessed.
        """
        spans: Dict[str, Tuple[float, float]] = {}
        for take, curves in self.take_curves().items():
            firsts = [c[0] for c in curves.values() if c[0] is not None]
            lasts = [c[1] for c in curves.values() if c[1] is not None]
            if firsts and lasts:
                spans[take] = (
                    min(firsts) / self.TICKS_PER_SECOND,
                    max(lasts) / self.TICKS_PER_SECOND,
                )
        return spans

    #: A key's interpolation (the ``KeyAttrFlags`` mode of the segment after
    #: it) in the one vocabulary a key travels with between the DCCs
    #: (``ShotTransfer.KEY_INTERPOLATIONS``): a hold, a straight line, and a
    #: cubic whatever its tangents. A mode outside these reads as linear --
    #: what an importer assumes of a key it cannot place.
    KEY_INTERPOLATIONS = {
        _FbxFileInternal._CONSTANT: "step",
        _FbxFileInternal._LINEAR: "linear",
        _FbxFileInternal._CUBIC: "smooth",
    }

    def take_keys(
        self,
        properties: Collection[str] = (),
        user: bool = False,
        kind: str = "Model",
    ) -> Dict[str, Dict[Tuple[str, str, str], List[List[Any]]]]:
        """Every key of the chosen channels, take by take.

        What an importer reading only transforms leaves behind -- a node's
        ``Visibility``, a DCC's custom attributes -- read off the file itself.
        The base layer only (each take's first ``AnimationLayer``, the one
        both DCCs key). Needs the curves' arrays: a file loaded without them
        is read again, payloads skipped (as :meth:`take_layout` is).

        Parameters:
            properties: Property names whose curves are read, on any *kind*
                target (``("Visibility",)``).
            user: Also read every animated user property -- ``U`` among the
                target's ``Properties70`` flags: where Maya's extra attributes
                and Blender's custom properties land.
            kind: The record the targets are (``"Model"``).

        Returns:
            ``{take: {(target, property, component): [[ticks, value,
            interpolation], ...]}}`` for every take in the file (``{}`` for
            one keying none of the chosen channels). *target* is the object's
            name -- ``name#uid`` where more than one *kind* object carries that
            name; *component* the curve's channel as stored
            (``"d|Visibility"``, ``"d|X"``); *ticks* the key time as stored
            (:attr:`TICKS_PER_SECOND`); *interpolation* the segment after the
            key in :attr:`KEY_INTERPOLATIONS`' words.

        Raises:
            ValueError: A curve whose key times and values differ in count.
        """
        wanted = set(properties)
        source = self
        if not self._curves_decoded():
            source = type(self).load(
                self.path, decode_arrays=self.CURVE_ARRAYS, raw_payloads=False
            )
        kinds: Dict[Any, str] = {}
        names: Dict[Any, str] = {}
        users: Dict[Any, set] = {}
        arrays: Dict[Any, Dict[str, Any]] = {}
        stacks: List[Any] = []
        for record in source.iter_objects():
            props = record["props"]
            if not props or not isinstance(props[0], int):
                continue
            uid, what = props[0], record["name"]
            kinds[uid] = what
            names[uid] = (self._display_name(props[1]) if len(props) > 1 else "") or ""
            if what == "AnimationStack":
                stacks.append(uid)
            elif what == "AnimationCurve":
                arrays[uid] = {
                    child["name"]: child["props"][0]
                    for child in record["children"]
                    if child["name"] in self.CURVE_ARRAYS and child["props"]
                }
            elif what == kind and user:
                users[uid] = {
                    p["props"][0].decode("utf-8", "replace")
                    for child in record["children"]
                    if child["name"] == "Properties70"
                    for p in child["children"]
                    if len(p["props"]) > 3
                    and isinstance(p["props"][0], bytes)
                    and isinstance(p["props"][3], bytes)
                    and b"U" in p["props"][3]
                }
        members: Dict[Any, List[Tuple[Any, Optional[str]]]] = {}
        for _link, child, parent, prop in source.connections():
            members.setdefault(parent, []).append((child, prop))
        repeats = Counter(names[u] for u, k in kinds.items() if k == kind)

        def label(uid: Any) -> str:
            return f"{names[uid]}#{uid}" if repeats[names[uid]] > 1 else names[uid]

        # A curve node's targets: the (object, property) it drives.
        driven: Dict[Any, List[Tuple[Any, str]]] = {}
        for parent, links in members.items():
            if kinds.get(parent) != kind:
                continue
            for child, prop in links:
                if prop is None or kinds.get(child) != "AnimationCurveNode":
                    continue
                if prop in wanted or prop in users.get(parent, ()):
                    driven.setdefault(child, []).append((parent, prop))

        takes: Dict[str, Dict[Tuple[str, str, str], List[List[Any]]]] = {}
        for stack in stacks:
            channels = takes.setdefault(names[stack], {})
            layer = next(
                (
                    c
                    for c, _p in members.get(stack, ())
                    if kinds.get(c) == "AnimationLayer"
                ),
                None,
            )
            for node, _p in members.get(layer, ()):
                for target, prop in driven.get(node, ()):
                    for curve, component in members.get(node, ()):
                        if component is None or curve not in arrays:
                            continue
                        key = (label(target), prop, component)
                        if key in channels:
                            continue  # FBX plays the first curve on a channel
                        parsed = self._key_curve(arrays[curve])
                        channels[key] = [
                            [
                                time,
                                value,
                                self.KEY_INTERPOLATIONS.get(
                                    self._key_attr(parsed, i)[0] & self._INTERPOLATION,
                                    "linear",
                                ),
                            ]
                            for i, (time, value) in enumerate(
                                zip(parsed.times, parsed.values)
                            )
                        ]
        return takes

    def take_layout(self, tolerance: float = 1e-4) -> Dict[str, Any]:
        """Which takes, imported together, carry ALL of the file's animation.

        No DCC merges takes on import -- Maya plays the one take
        ``FBXImportSetTake`` names, Blender assigns each object the first
        stack's action -- so a file whose motion is split across takes (a
        Shots Only export, per-object stacks) imports one take's worth. This
        sorts the takes so a caller imports everything: the ``live`` takes,
        merged into one by :meth:`FbxMedia.merge_takes` when there is more
        than one, and the ``stash`` takes -- alternatives -- parked whole.

        One rule decides every layout. Each take keys channels (an animated
        object's property component, per layer) over intervals, judged by the
        file's own interpolation (constant, linear, cubic with stored or
        Maya-computed slopes, weighted). The seed is the file's current take
        if it keys anything, else the first take that does. Every other take,
        in file order, is compared with each take accepted so far on every
        channel both key: a key of either inside their overlap that the other
        evaluates differently (beyond *tolerance*) sends the take to
        ``stash`` WHOLE -- a take is one coherent motion, never split.
        Touching at one frame with one value is a seam, not a conflict. The
        accepted takes are then kept widest first while each still keys some
        channel at a time no kept take spans: those are ``live``; the rest
        (a slice of a covering take, a take that keys nothing) are
        ``redundant``.

        Edge cases: a cover listed after its slices still covers them (the
        widest take is weighed first; equal widths prefer the seed, then file
        order); two identical takes keep the first; a take that keys only
        channels the live takes span adds nothing even where the live takes
        leave a gap between them; a curve node with no curves keys nothing;
        a channel two layers key is two channels (layers are matched by
        their order in the stack).

        Parameters:
            tolerance: Two values agree when they differ by at most this --
                absolute below 1, relative above (a float32 key holds about
                seven significant digits).

        Returns:
            ``{"takes", "current", "kind", "live", "redundant", "stash",
            "conflicts"}``. ``takes`` lists every take in file order as
            ``{"name", "index", "start", "end"}``: ``index`` is 1-based --
            Maya's ``FBXImportSetTake -takeIndex`` -- and ``start`` / ``end``
            the take's key extent in SECONDS (:meth:`take_spans`' unit),
            ``None`` for a take that keys nothing. ``current`` is the take the
            file declares current, or ``None``. ``live`` / ``redundant`` /
            ``stash`` are take names in file order; ``conflicts`` maps each
            stashed take to why (take, channel, frame, both values). ``kind``
            is for logging: ``"none"`` (nothing is keyed), ``"conflicted"``
            (something is stashed), ``"merged"`` (more than one live take),
            ``"covered"`` (one live take, others redundant) or ``"single"``.
            Only ``"merged"`` -- or ``"conflicted"`` with several live takes
            -- needs a merge; any other file imports as it is.

        Raises:
            ValueError: A curve whose key times and values differ in count.
        """
        return self._layout(
            self._take_model(), self._current_take(), tolerance, self._fps()
        )

    def _curves_decoded(self) -> bool:
        """True when every curve's :attr:`CURVE_ARRAYS` were read as lists."""
        for record in self.iter_objects():
            if record["name"] != "AnimationCurve":
                continue
            for child in record["children"]:
                if child["name"] in self.CURVE_ARRAYS and child["props"]:
                    if not isinstance(child["props"][0], list):
                        return False
        return True

    def _take_model(self) -> Dict[str, Any]:
        """:meth:`_build_take_model` of this file -- read again, payloads
        skipped, when it was loaded without its curve arrays."""
        source = self
        if not self._curves_decoded():
            source = type(self).load(
                self.path, decode_arrays=self.CURVE_ARRAYS, raw_payloads=False
            )
        rows = []
        for record in source.iter_objects():
            props = record["props"]
            if not props or not isinstance(props[0], int):
                continue
            arrays = None
            if record["name"] == "AnimationCurve":
                arrays = {
                    child["name"]: child["props"][0]
                    for child in record["children"]
                    if child["name"] in self.CURVE_ARRAYS and child["props"]
                }
            display = self._display_name(props[1]) if len(props) > 1 else None
            rows.append((record["name"], props[0], display or "", arrays))
        return self._build_take_model(rows, source.connections())

    def _current_take(self) -> Optional[str]:
        """The take ``Takes/Current`` names, or ``None``."""
        for child in (self.section("Takes") or {}).get("children", []):
            if child["name"] == "Current" and child["props"]:
                value = child["props"][0]
                if isinstance(value, (bytes, bytearray)) and value:
                    return bytes(value).decode("utf-8", "replace")
        return None

    def _fps(self) -> Optional[float]:
        """The file's frame rate (``GlobalSettings`` ``TimeMode``), or ``None``."""
        found: Dict[bytes, Any] = {}
        for child in (self.section("GlobalSettings") or {}).get("children", []):
            if child["name"] != "Properties70":
                continue
            for prop in child["children"]:
                props = prop["props"]
                if len(props) >= 5 and props[0] in (b"TimeMode", b"CustomFrameRate"):
                    found[props[0]] = props[4]
        return self._frame_rate(found.get(b"TimeMode"), found.get(b"CustomFrameRate"))

    def connections(self) -> List[Tuple[str, Any, Any, Optional[str]]]:
        """Every ``C`` record as ``(kind, child_id, parent_id, property)``.

        Property-name is ``None`` for object-object (``"OO"``) links and the
        target attribute for object-property (``"OP"``) ones.
        """
        section = self.section("Connections")
        rows: List[Tuple[str, Any, Any, Optional[str]]] = []
        for record in (section or {}).get("children", []):
            props = record["props"]
            if len(props) < 3:
                continue
            kind = (
                props[0].decode("ascii", "replace")
                if isinstance(props[0], (bytes, bytearray))
                else str(props[0])
            )
            prop_name = None
            if len(props) > 3 and isinstance(props[3], (bytes, bytearray)):
                prop_name = props[3].decode("utf-8", "replace")
            rows.append((kind, props[1], props[2], prop_name))
        return rows


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
