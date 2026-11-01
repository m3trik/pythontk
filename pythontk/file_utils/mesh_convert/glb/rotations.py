# !/usr/bin/python
# coding=utf-8
"""Restore the rotation FBX2glTF drops from a node scaled flat on one axis.

Why this exists: FBX2glTF (v0.13.1) bakes each node's local matrix per frame
and decomposes it, and a zero scale leaves no rotation to recover -- the
decomposition comes out NaN. The converter then drops what it cannot say,
measured 2026-10-31: the node's rest ``rotation`` when its rest pose is flat,
and with it EVERY rotation channel the node has, since each frame is compared
against that NaN rest (a node at rest scale (1, 0, 1) keyed back to scale 1
still ships none); and a take's channel when every frame of it is flat. The
node plays unrotated while its FBX twin keeps the tilt -- before any repair
can run, since the channel is not NaN but gone.

What it does: screens the GLB for that evidence -- a node flat at rest that
states no rotation, or one flat at every key of a take's scale channel (not
zero on every axis: that node is not seen) that the take does not rotate -- so
a file with none is left alone without reading the FBX; only then reads the
source FBX. The rotation is the FBX's own, each ``Lcl Rotation`` curve
evaluated as :class:`FbxFile` reads it (pinned within float precision against
the converter's sampling of Maya's tangents) and composed as the converter
composes it (measured against its own channel for a node that is not flat)::

    rotation = PreRotation (x) R(RotationOrder; Lcl Rotation) (x) PostRotation^-1

PreRotation and PostRotation in XYZ order whatever the ``RotationOrder``; all
three ignored -- plain XYZ -- where ``RotationActive`` is off or absent (the
SDK's default). A child of the file's root also takes the turn from the file's
axis system to the converter's Y-up: the SDK's ``ConvertScene`` re-orients the
root's children only. A property a model omits reads as the SDK default (its
``PropertyTemplate`` is not read).

A channel shares the time input the converter wrote for its take's animation
-- one animation per take, named after it -- and reads it as the converter
does: t = 0 is the take's first key (over every curve in it) truncated to the
bake grid. A take whose every channel was dropped has no input; one is minted
at the bake rate over the take's span. Written: a channel only where the
animation does not rotate the node, never one that only restates the rest
pose; a rest rotation only where the node states none.

Refused and REPORTED (a warning): a flat node whose name the FBX gives more
than one model, when one of them carries a rotation -- their curves cannot be
told apart. Read from the base layer only (each take's first
``AnimationLayer``, the one both DCCs key). A root child of a mirrored axis
system (no rotation reaches the converter's) is left as shipped.

Runs in :meth:`MeshConvert.fbx_to_glb`'s session on the converter's own output,
ahead of every pass that reads a key.
"""

from __future__ import annotations

import logging
import math
import struct
from array import array
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from pythontk.file_utils.mesh_convert.fbx_file import FbxFile
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget
from pythontk.geo_utils.articulation.model import _ArticulationModelInternal as _Q

__all__ = ["GlbRotations"]

logger = logging.getLogger(__name__)

Quat = Tuple[float, float, float, float]
_IDENTITY: Quat = (0.0, 0.0, 0.0, 1.0)


class _GlbRotationsInternal:
    """GLB screening, FBX pose reading and channel writing for
    :class:`GlbRotations`."""

    #: FBX ``RotationOrder`` (``EFbxRotationOrder``) -> axis order, first
    #: applied first; 6 (spheric XYZ) and anything past it read as XYZ.
    ORDERS = ("xyz", "xzy", "yzx", "yxz", "zxy", "zyx")
    #: A rotation's curve components, in axis order.
    COMPONENTS = ("d|X", "d|Y", "d|Z")
    #: Two unit quaternions closer than this (``1 - |dot|``, about 0.016
    #: degrees) state one pose: float32 noise, not motion.
    SAME = 1e-9

    @staticmethod
    def _flat(values: Sequence[Any]) -> bool:
        """True when a scale has a zero component: no rotation to decompose."""
        return any(v == 0.0 for v in values)

    @classmethod
    def _same(cls, a: Sequence[float], b: Sequence[float]) -> bool:
        return 1.0 - abs(sum(x * y for x, y in zip(a, b))) < cls.SAME

    @staticmethod
    def _rows(edit: GlbEdit, index: Any) -> Optional[List[Tuple[float, ...]]]:
        """A tightly packed float accessor's elements, or ``None``."""
        accessors = edit.gltf.get("accessors") or []
        if not isinstance(index, int) or not 0 <= index < len(accessors):
            return None
        if (accessors[index] or {}).get("componentType") != 5126:
            return None
        elements = GlbEdit.accessor_elements(edit, index)
        if not elements:
            return None
        width = len(elements[0]) // 4
        return [struct.unpack(f"<{width}f", element) for element in elements]

    @staticmethod
    def _squashed(edit: GlbEdit, index: Any) -> bool:
        """True when scale output *index* is flat at every key and not zero on
        every axis at all of them (a node no one sees needs no rotation). A
        channel with no zero at all -- nearly every one -- is passed over at C
        speed: this screen runs on every conversion."""
        elements = (
            GlbEdit.accessor_elements(edit, index) if isinstance(index, int) else None
        )
        accessors = edit.gltf.get("accessors") or []
        if not elements or (accessors[index] or {}).get("componentType") != 5126:
            return False
        values = array("f", b"".join(elements))
        if 0.0 not in values:
            return False
        width = len(elements[0]) // 4
        rows = [values[k : k + width] for k in range(0, len(values), width)]
        return all(0.0 in row for row in rows) and any(any(row) for row in rows)

    @classmethod
    def _suspects(
        cls, edit: GlbEdit, skip: Collection[int] = ()
    ) -> Tuple[Set[int], Dict[int, Set[int]]]:
        """``(rest, channels)``: the nodes flat at rest that state no rotation,
        and per animation the nodes it does not rotate that are one of those
        or flat at every key of their scale channel there (:meth:`_squashed`);
        none of *skip*."""
        nodes = edit.gltf.get("nodes") or []
        rest = {
            i
            for i, node in enumerate(nodes)
            if cls._flat(node.get("scale") or ())
            and "rotation" not in node
            and i not in skip
        }
        channels: Dict[int, Set[int]] = {}
        for position, animation in enumerate(edit.gltf.get("animations") or []):
            samplers = animation.get("samplers") or []
            turned: Set[int] = set()
            squashed = set(rest)
            for channel in animation.get("channels") or []:
                target = channel.get("target") or {}
                node, path = target.get("node"), target.get("path")
                index = channel.get("sampler")
                if not isinstance(node, int):
                    continue
                if path == "rotation":
                    turned.add(node)
                elif path == "scale" and node not in squashed and node not in skip:
                    sampler = (
                        samplers[index]
                        if isinstance(index, int) and 0 <= index < len(samplers)
                        else {}
                    )
                    if cls._squashed(edit, (sampler or {}).get("output")):
                        squashed.add(node)
            if squashed - turned:
                channels[position] = squashed - turned
        return rest, channels

    @staticmethod
    def _poses(fbx: FbxFile) -> Dict[str, List[Dict[str, Any]]]:
        """``{model name: [{"root", "props"}]}``: each ``Model``'s
        ``Properties70`` (``{name: values}``) and whether it is a child of the
        file's root, by name -- one entry per model carrying it."""
        records = [r for r in fbx.iter_objects() if r["name"] == "Model" and r["props"]]
        models = {r["props"][0] for r in records}
        parents: Dict[Any, Any] = {}
        for kind, child, parent, _prop in fbx.connections():
            # A display layer takes its members as OO children too.
            if kind == "OO" and (parent == 0 or parent in models):
                parents.setdefault(child, parent)
        poses: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            props: Dict[str, Sequence[Any]] = {}
            for child in record["children"]:
                if child["name"] != "Properties70":
                    continue
                for row in child["children"]:
                    values = row["props"]
                    if values and isinstance(values[0], (bytes, bytearray)):
                        props[bytes(values[0]).decode("utf-8", "replace")] = values[4:]
            poses.setdefault(fbx.object_name(record) or "", []).append(
                {"root": parents.get(record["props"][0], 0) == 0, "props": props}
            )
        return poses

    @staticmethod
    def _matrix_quat(m: Sequence[Sequence[float]]) -> Quat:
        """The unit quaternion of rotation matrix *m* (rows, column vectors)."""
        trace = m[0][0] + m[1][1] + m[2][2]
        if trace > 0.0:
            s = 2.0 * math.sqrt(trace + 1.0)
            return (
                (m[2][1] - m[1][2]) / s,
                (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s,
                0.25 * s,
            )
        i = max(range(3), key=lambda k: m[k][k])
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + m[i][i] - m[j][j] - m[k][k])
        q = [0.0, 0.0, 0.0, (m[k][j] - m[j][k]) / s]
        q[i], q[j], q[k] = 0.25 * s, (m[j][i] + m[i][j]) / s, (m[k][i] + m[i][k]) / s
        return (q[0], q[1], q[2], q[3])

    @classmethod
    def _axis_turn(cls, fbx: FbxFile) -> Optional[Quat]:
        """The turn FBX2glTF gives the root's children, bringing *fbx*'s axis
        system (``GlobalSettings``; absent, Y-up) to its own -- Maya's Y-up:
        +X right, +Y up, +Z front. ``None`` for a system no rotation reaches
        (mirrored, or malformed)."""
        settings: Dict[str, Any] = {}
        for child in (fbx.section("GlobalSettings") or {}).get("children", []):
            if child["name"] != "Properties70":
                continue
            for row in child["children"]:
                values = row["props"]
                if len(values) >= 5 and isinstance(values[0], (bytes, bytearray)):
                    settings[bytes(values[0]).decode("utf-8", "replace")] = values[4]
        rows = []  # each source axis, as the row taking it to its Y-up axis
        for name, default in (("CoordAxis", 0), ("UpAxis", 1), ("FrontAxis", 2)):
            axis, sign = settings.get(name, default), settings.get(name + "Sign", 1)
            if axis not in (0, 1, 2) or not isinstance(sign, (int, float)):
                return None
            unit = 1.0 if sign >= 0 else -1.0
            rows.append([unit if c == axis else 0.0 for c in range(3)])
        a, b, c = rows
        det = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        return cls._matrix_quat(rows) if det > 0.5 else None

    @staticmethod
    def _euler(order: str, degrees: Sequence[float]) -> Quat:
        return _Q._euler_quat(order, dict(zip(("rx", "ry", "rz"), degrees)))

    @staticmethod
    def _vector(props: Dict[str, Sequence[Any]], name: str) -> List[float]:
        """A model's 3-vector property; zeros where it is absent."""
        values = [float(v) for v in list(props.get(name) or ())[:3]]
        return values + [0.0] * (3 - len(values))

    @classmethod
    def _composer(
        cls, props: Dict[str, Sequence[Any]], lead: Quat
    ) -> Callable[[Sequence[float]], Quat]:
        """Euler degrees -> the node's local rotation, as the converter
        composes a model with *props*; *lead* (the axis turn) applied last."""
        active = bool((props.get("RotationActive") or (0,))[0])
        order, pre, post = "xyz", _IDENTITY, _IDENTITY
        if active:
            index = (props.get("RotationOrder") or (0,))[0]
            order = cls.ORDERS[index] if index in range(len(cls.ORDERS)) else "xyz"
            pre = cls._euler("xyz", cls._vector(props, "PreRotation"))
            post = _Q._qconj(cls._euler("xyz", cls._vector(props, "PostRotation")))
        head = _Q._qmul(lead, pre)
        return lambda degrees: _Q._qnormalize(
            _Q._qmul(_Q._qmul(head, cls._euler(order, degrees)), post)
        )

    @staticmethod
    def _extents(fbx: FbxFile) -> Dict[str, Tuple[int, int]]:
        """Each take's ``(first, last)`` key in ticks, over every curve in it --
        what the converter sizes a take by (:meth:`FbxFile.take_spans`)."""
        found: Dict[str, Tuple[int, int]] = {}
        for take, curves in fbx.take_curves().items():
            firsts = [c[0] for c in curves.values() if c[0] is not None]
            lasts = [c[1] for c in curves.values() if c[1] is not None]
            if firsts and lasts:
                found[take] = (min(firsts), max(lasts))
        return found

    @classmethod
    def _plan(
        cls, edit: GlbEdit, fbx: str, fps: float, skip: Collection[int] = ()
    ) -> Tuple[Dict[int, Quat], List[Tuple[int, int, List[Quat], Any]], Set[str]]:
        """``(rest, channels, refused)``: the rest rotation per node; each
        channel as ``(animation, node, keys, input)``, *input* the accessor the
        animation's channels share or the frames to mint one over; and the
        flat nodes whose model cannot be told apart from another."""
        rest, wanted = cls._suspects(edit, skip)
        if not rest and not wanted:
            return {}, [], set()
        source = FbxFile.load(
            fbx, decode_arrays=FbxFile.CURVE_ARRAYS, raw_payloads=False
        )
        poses = cls._poses(source)
        turn = cls._axis_turn(source)
        nodes = edit.gltf.get("nodes") or []
        refused: Set[str] = set()

        def posed(name: Any) -> Optional[Tuple[Callable, List[float]]]:
            """The composer and rest angles of the one model named *name*."""
            found = poses.get(name) or []
            lead = turn if len(found) == 1 and found[0]["root"] else _IDENTITY
            if len(found) != 1 or lead is None:
                return None
            props = found[0]["props"]
            return cls._composer(props, lead), cls._vector(props, "Lcl Rotation")

        restored: Dict[int, Quat] = {}
        for index in sorted(rest):
            name = nodes[index].get("name")
            pose = posed(name)
            if pose is not None:
                rotation = pose[0](pose[1])
                if not cls._same(rotation, _IDENTITY):
                    restored[index] = rotation
            elif len(poses.get(name) or ()) > 1 and any(
                # A namesake carries a rotation: which model is this?
                not cls._same(
                    cls._composer(p["props"], _IDENTITY)(
                        cls._vector(p["props"], "Lcl Rotation")
                    ),
                    _IDENTITY,
                )
                for p in poses[name]
            ):
                refused.add(str(name))

        curves = source._take_key_curves(("Lcl Rotation",))
        extents = cls._extents(source)
        per_frame = FbxFile.TICKS_PER_SECOND / fps
        animations = edit.gltf.get("animations") or []
        planned: List[Tuple[int, int, List[Quat], Any]] = []
        for position, indices in sorted(wanted.items()):
            take = str(animations[position].get("name") or "")
            keyed, extent = curves.get(take) or {}, extents.get(take)
            if not keyed or extent is None:
                continue
            first = int(extent[0] / per_frame)  # toward zero, as FbxTime counts
            samplers = animations[position].get("samplers") or []
            shared = (samplers[0] or {}).get("input") if samplers else None
            times = cls._rows(edit, shared)
            if times is None:
                frames = list(range(int(extent[1] / per_frame) - first + 1))
                feed: Any = frames
            elif all(math.isfinite(t[0]) for t in times):
                frames, feed = [round(t[0] * fps) for t in times], shared
            else:
                continue
            for index in sorted(indices):
                name = nodes[index].get("name")
                own = [keyed.get((name, "Lcl Rotation", c)) for c in cls.COMPONENTS]
                pose = posed(name) if any(own) else None
                if pose is None:
                    # Keyed under a namesake's label (``name#uid``) instead.
                    if any(str(label).startswith(f"{name}#") for label, *_ in keyed):
                        refused.add(str(name))
                    continue
                compose, still = pose
                keys: List[Quat] = []
                for frame in frames:
                    ticks = round((first + frame) * per_frame)
                    q = compose(
                        [
                            FbxFile._evaluate_curve(curve, ticks) if curve else still[i]
                            for i, curve in enumerate(own)
                        ]
                    )
                    if keys and sum(a * b for a, b in zip(keys[-1], q)) < 0.0:
                        q = (-q[0], -q[1], -q[2], -q[3])  # one hemisphere: no flip
                    keys.append(q)
                held = restored.get(index) or nodes[index].get("rotation") or _IDENTITY
                if not all(cls._same(q, held) for q in keys):
                    planned.append((position, index, keys, feed))
        return restored, planned, refused

    @staticmethod
    def _write(
        edit: GlbEdit, planned: List[Tuple[int, int, List[Quat], Any]], fps: float
    ) -> bool:
        """Append *planned*'s channels -- each animation's minted input once --
        or ``False`` when the GLB has no BIN of its own to append to."""
        payloads: List[bytes] = []
        minted: Dict[int, int] = {}  # animation -> its minted input's payload
        outputs: List[int] = []  # each channel's payload
        for position, _index, keys, feed in planned:
            if not isinstance(feed, int) and position not in minted:
                minted[position] = len(payloads)
                seconds = [frame / fps for frame in feed]
                payloads.append(struct.pack(f"<{len(seconds)}f", *seconds))
            outputs.append(len(payloads))
            values = [v for q in keys for v in q]
            payloads.append(struct.pack(f"<{len(values)}f", *values))
        views = GlbEdit.append_bin_views(edit, payloads)
        if not views:
            return False
        accessors = edit.gltf.setdefault("accessors", [])
        inputs: Dict[int, int] = {}
        for position, slot in minted.items():
            # Bounds read back out of the packed bytes: float32 can round up.
            seconds = struct.unpack(f"<{len(payloads[slot]) // 4}f", payloads[slot])
            accessors.append(
                {
                    "bufferView": views[slot],
                    "componentType": 5126,
                    "count": len(seconds),
                    "type": "SCALAR",
                    "min": [min(seconds)],
                    "max": [max(seconds)],
                }
            )
            inputs[position] = len(accessors) - 1
        for (position, index, keys, feed), slot in zip(planned, outputs):
            accessors.append(
                {
                    "bufferView": views[slot],
                    "componentType": 5126,
                    "count": len(keys),
                    "type": "VEC4",
                }
            )
            animation = edit.gltf["animations"][position]
            samplers = animation.setdefault("samplers", [])
            samplers.append(
                {
                    "input": feed if isinstance(feed, int) else inputs[position],
                    "interpolation": "LINEAR",
                    "output": len(accessors) - 1,
                }
            )
            animation.setdefault("channels", []).append(
                {
                    "sampler": len(samplers) - 1,
                    "target": {"node": index, "path": "rotation"},
                }
            )
        edit.dirty = True
        return True


class GlbRotations(_GlbRotationsInternal):
    """Restore the rotation FBX2glTF drops from a node scaled flat on one axis.

    Why and how: the module docstring.

    Example:
        >>> GlbRotations.restore("assembly.glb", "assembly.fbx", fps=30.0)
        {'nodes': 1, 'channels': 2}
    """

    @classmethod
    def restore(
        cls, glb: GlbTarget, fbx: str, fps: float, skip: Collection[int] = ()
    ) -> Dict[str, int]:
        """Give each flat node the rotation FBX2glTF dropped from it.

        Parameters:
            glb: Path to a ``.glb``, modified in place, or an open session --
                FBX2glTF's own output, before any pass rebuilds its clips.
            fbx: The FBX the GLB was converted from.
            fps: The rate the converter baked at (its ``--anim-framerate``).
            skip: Node indices no one sees, never screened: the transport
                nodes a later pass strips (a curve proxy carries its ramp on
                ``scale.x``, so it is flat wherever the ramp reads 0).

        Returns:
            ``{"nodes": n, "channels": n}`` -- rest rotations and rotation
            channels written. A file with nothing to restore is not rewritten,
            and its FBX is not read without a flat node lacking a rotation.
        """
        with GlbEdit.open(glb) as edit:
            restored, planned, refused = cls._plan(edit, fbx, fps, skip)
            if refused:
                logger.warning(
                    "Rotations: %d node(s) scaled flat on an axis share their "
                    "name with another FBX model that carries a rotation, so the "
                    "one FBX2glTF dropped cannot be read back: %s.",
                    len(refused),
                    ", ".join(sorted(refused)),
                )
            nodes = edit.gltf.get("nodes") or []
            for index, rotation in restored.items():
                nodes[index]["rotation"] = list(rotation)
                edit.dirty = True
            if planned and not cls._write(edit, planned, fps):
                logger.warning(
                    "Rotations: this GLB's buffer is external, so the %d rotation "
                    "channel(s) FBX2glTF dropped have nowhere to live.",
                    len(planned),
                )
                planned = []
            if restored or planned:
                animations = edit.gltf.get("animations") or []
                logger.info(
                    "Rotations: FBX2glTF drops the rotation of a node scaled flat "
                    "on an axis (a zero scale leaves its decomposition none to "
                    "recover); restored from the FBX's own curves: %s.",
                    ", ".join(
                        [f"{nodes[i].get('name')} (rest)" for i in sorted(restored)]
                        + [
                            f"{nodes[i].get('name')} ({animations[a].get('name')})"
                            for a, i, _keys, _feed in planned
                        ]
                    ),
                )
            return {"nodes": len(restored), "channels": len(planned)}


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
