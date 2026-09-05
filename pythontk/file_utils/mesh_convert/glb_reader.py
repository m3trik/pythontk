# !/usr/bin/python
# coding=utf-8
"""Read-only structured access to a GLB: accessors, animation sampling, worlds.

The inspection half of the ``mesh_convert`` family. :class:`MeshConvert` owns
container parsing and *editing*; :class:`GlbClips`/:class:`GlbFades` rewrite
animation; this module only ever **reads** — it decodes accessors of every
component type, evaluates animation samplers (LINEAR / STEP / CUBICSPLINE) at
arbitrary times, and composes node world matrices, which is what deliverable
verification and world-space spot probes need and none of the writers expose.

Proven against production: the world composition here reproduced live-Maya
world positions to under 2 mm across a 2,480-node assembly, and caught a
7.5 cm export defect the DCC's own log never mentioned.
"""

import math
import struct
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert


class _GlbReaderInternal:
    """Decode tables and matrix math for :class:`GlbReader`."""

    #: glTF componentType -> (struct format char, byte size).
    _COMPONENT_TYPES: Dict[int, Tuple[str, int]] = {
        5120: ("b", 1),
        5121: ("B", 1),
        5122: ("h", 2),
        5123: ("H", 2),
        5125: ("I", 4),
        5126: ("f", 4),
    }

    #: glTF accessor type -> component count.
    _TYPE_WIDTHS: Dict[str, int] = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }

    # ---- animation sampler decode -----------------------------------------

    @staticmethod
    def _sampler_frames(
        times: Sequence[Sequence[float]],
        values: Sequence[Sequence[float]],
        interpolation: str,
    ) -> Optional[List[Tuple[float, ...]]]:
        """One flat pose tuple per key time, or ``None`` if the pair is unusable.

        An output accessor holds more than one element per key whenever the
        target does: ``weights`` carries one scalar per morph target, and
        CUBICSPLINE triples everything into (in-tangents, values,
        out-tangents) per the spec's ``a1..an v1..vn b1..bn`` layout. Both
        are folded here so callers compare like with like -- a plain
        ``len(values) == len(times)`` test silently skips every morph-target
        channel in the file.
        """
        count = len(times)
        if not count or len(values) % count:
            return None
        per_key = len(values) // count
        if interpolation == "CUBICSPLINE":
            if per_key % 3:
                return None
            width = per_key // 3
            # The tangents are not the pose; take the middle third.
            lo, hi = width, 2 * width
        else:
            lo, hi = 0, per_key
        return [
            tuple(
                c
                for element in values[i * per_key + lo : i * per_key + hi]
                for c in element
            )
            for i in range(count)
        ]

    @staticmethod
    def _moving_span(
        poses: Sequence[Sequence[float]], tolerance: float
    ) -> Optional[Tuple[int, int]]:
        """Key indices bracketing the motion in *poses*, or ``None`` if static.

        Trims the leading and trailing HOLDS -- the run of keys still at the
        opening pose, and the run already at the closing pose -- rather than
        looking for change between neighbours.

        The neighbour test this replaces could not see motion that accumulates
        below *tolerance*: measured, 400 keys stepping 0.0005 each (half the
        1e-3 default) travel 0.2 m over 13 seconds and every individual step
        is under the threshold, so the whole pan read as static. The same
        0.2 m as one jump was detected -- the answer depended on how the
        motion was distributed rather than on whether it happened. Comparing
        against the resting pose at each end has no such blind spot, and
        still brackets a single discrete move exactly as before.
        """
        count = len(poses)
        if count < 2:
            return None

        def apart(a, b) -> bool:
            return any(abs(x - y) > tolerance for x, y in zip(a, b))

        first, opening = 0, poses[0]
        while first + 1 < count and not apart(poses[first + 1], opening):
            first += 1
        last, closing = count - 1, poses[-1]
        while last - 1 >= 0 and not apart(poses[last - 1], closing):
            last -= 1
        return (first, last) if first < last else None

    # ---- matrix math ------------------------------------------------------
    #
    # Convention: a matrix is four row-lists, world = local x parentWorld,
    # translation in row 3. This is the transpose-view of glTF's column-major
    # flat arrays — reshaping a glTF ``node.matrix`` four floats at a time
    # yields exactly these rows, and composing child-then-parent in this view
    # equals glTF's parent-then-child in column-major. Verified against a DCC
    # ground truth (sub-2 mm over thousands of samples), so keep the
    # convention; "fixing" it to column-major flips every product below.

    @staticmethod
    def _identity() -> List[List[float]]:
        return [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]

    @staticmethod
    def _mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        return [
            [sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)]
            for r in range(4)
        ]

    @staticmethod
    def _trs_matrix(
        t: Sequence[float], r: Sequence[float], s: Sequence[float]
    ) -> List[List[float]]:
        """Compose translation / rotation-quaternion / scale into row form."""
        x, y, z, w = r
        length = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
        x, y, z, w = x / length, y / length, z / length, w / length
        rot = [
            [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
            [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
            [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)],
        ]
        m = _GlbReaderInternal._identity()
        for r_ in range(3):
            for c in range(3):
                m[r_][c] = rot[r_][c] * s[r_]
        m[3][0], m[3][1], m[3][2] = t[0], t[1], t[2]
        return m

    @staticmethod
    def _rows_from_flat(flat: Sequence[float]) -> List[List[float]]:
        return [list(flat[i : i + 4]) for i in (0, 4, 8, 12)]


class GlbReader(_GlbReaderInternal):
    """Read-only GLB inspector: accessors, animation evaluation, node worlds.

    Wraps :class:`MeshConvert.GlbEdit` — one container parser in the
    ecosystem — and never marks the session dirty, so nothing here can write.

    Example:
        >>> reader = GlbReader.load("asset.glb")
        >>> reader.counts()["nodes"]
        2480
        >>> reader.world_position("hand_L", time=1.5)
        (-0.554, 1.054, 2.174)
    """

    def __init__(self, edit: "MeshConvert.GlbEdit"):
        """Wrap an already-open :class:`MeshConvert.GlbEdit` session.

        Parameters:
            edit: An open GLB session (see :meth:`load` for the path form).
                The reader holds it for lazy BIN access and never sets
                ``dirty``.
        """
        self._edit = edit
        self.gltf: Dict[str, Any] = edit.gltf
        self.path: Optional[str] = getattr(edit, "path", None)
        self._parents: Optional[Dict[int, int]] = None
        self._names: Optional[Dict[str, int]] = None

    @classmethod
    def load(cls, path: str) -> "GlbReader":
        """Open *path* read-only and return a reader.

        Uses :meth:`MeshConvert._read_glb` — the single owner of GLB
        container parsing — rather than a second parser. The session is
        read-only by construction (nothing here sets ``dirty``), so no
        write-on-close context is needed.
        """
        return cls(MeshConvert._read_glb(path))

    # ---- structure --------------------------------------------------------

    def counts(self) -> Dict[str, int]:
        """Section lengths for the usual census keys (missing -> 0)."""
        return {
            key: len(self.gltf.get(key) or [])
            for key in (
                "scenes",
                "nodes",
                "meshes",
                "materials",
                "images",
                "textures",
                "skins",
                "animations",
                "cameras",
                "accessors",
            )
        }

    def image_mimes(self) -> Dict[str, int]:
        """``{mimeType: count}`` over ``images`` (missing type -> "?")."""
        out: Dict[str, int] = {}
        for image in self.gltf.get("images") or []:
            mime = image.get("mimeType") or "?"
            out[mime] = out.get(mime, 0) + 1
        return out

    def extensions(self) -> Tuple[List[str], List[str]]:
        """``(extensionsUsed, extensionsRequired)`` (each possibly empty)."""
        return (
            list(self.gltf.get("extensionsUsed") or []),
            list(self.gltf.get("extensionsRequired") or []),
        )

    def skins_summary(self) -> Dict[str, int]:
        """Counts the skin checks need: skins, with-IBM, skinned mesh nodes."""
        skins = self.gltf.get("skins") or []
        return {
            "skins": len(skins),
            "with_inverse_bind_matrices": sum(
                1 for s in skins if "inverseBindMatrices" in s
            ),
            "skinned_nodes": sum(
                1 for n in self.gltf.get("nodes") or [] if "skin" in n
            ),
        }

    # ---- accessors --------------------------------------------------------

    def accessor(self, index: int) -> Optional[List[Tuple[float, ...]]]:
        """Decode accessor *index* into element tuples, or ``None``.

        Every component type is supported (unlike :class:`GlbClips`'
        deliberately FLOAT-only fast path); interleaved ``byteStride`` is
        honored. ``None`` for a sparse accessor, an out-of-range index, a
        missing bufferView, or a BIN too short for the promised count —
        callers treat ``None`` as "unreadable", never as empty. Integer
        values are returned raw (``normalized`` is not applied).
        """
        accessors = self.gltf.get("accessors") or []
        if not isinstance(index, int) or not 0 <= index < len(accessors):
            return None
        acc = accessors[index] or {}
        if "sparse" in acc:
            return None
        spec = self._COMPONENT_TYPES.get(acc.get("componentType"))
        width = self._TYPE_WIDTHS.get(acc.get("type"))
        view_index = acc.get("bufferView")
        if spec is None or width is None or not isinstance(view_index, int):
            return None
        views = self.gltf.get("bufferViews") or []
        if not 0 <= view_index < len(views):
            return None
        blob = self._edit.bin_data
        if blob is None:
            return None
        fmt, size = spec
        view = views[view_index] or {}
        base = int(view.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
        count = int(acc.get("count", 0))
        stride = int(view.get("byteStride") or 0) or width * size
        end = base + (count - 1) * stride + width * size if count else base
        if end > len(blob):
            return None
        try:
            return [
                struct.unpack_from(f"<{width}{fmt}", blob, base + i * stride)
                for i in range(count)
            ]
        except struct.error:
            return None

    # ---- animation --------------------------------------------------------

    def animations(self) -> List[str]:
        """Clip names, in file order (unnamed -> ``"<i>"``)."""
        return [
            a.get("name") or str(i)
            for i, a in enumerate(self.gltf.get("animations") or [])
        ]

    def animation(self, key: Union[int, str]) -> Optional[Dict[str, Any]]:
        """The animation dict for an index or name, or ``None``."""
        anims = self.gltf.get("animations") or []
        if isinstance(key, int):
            return anims[key] if 0 <= key < len(anims) else None
        for a in anims:
            if a.get("name") == key:
                return a
        return None

    def clip_spans(self, fps: float = 30.0) -> Dict[str, Tuple[float, float, int]]:
        """Per clip: ``(min_seconds, max_seconds, end_frame)``.

        Read from input-accessor ``min``/``max`` fields — no BIN decode —
        so it is safe on a file whose geometry never loads.
        """
        out: Dict[str, Tuple[float, float, int]] = {}
        accessors = self.gltf.get("accessors") or []
        for i, anim in enumerate(self.gltf.get("animations") or []):
            lows: List[float] = []
            highs: List[float] = []
            for sampler in anim.get("samplers") or []:
                acc = accessors[sampler["input"]] if "input" in sampler else {}
                if acc.get("min"):
                    lows.append(acc["min"][0])
                if acc.get("max"):
                    highs.append(acc["max"][0])
            low = min(lows) if lows else 0.0
            high = max(highs) if highs else 0.0
            out[anim.get("name") or str(i)] = (low, high, round(high * fps))
        return out

    def motion_span(
        self, key: Union[int, str], tolerance: float = 1e-3
    ) -> Optional[Tuple[float, float]]:
        """The clip's MOTION extent in seconds: ``(low, high)``, or ``None``.

        Key extent is not play extent. A bake writes keys across the whole
        range it is handed, so a clip routinely carries a held pose at one
        or both ends -- frames that occupy the timeline without animating
        anything. This walks each sampler's decoded outputs and reports the
        span between the first and last key where some channel actually
        changes, dropping those holds. ``None`` when nothing moves at all.

        *tolerance* is an absolute per-component threshold. The default
        1e-3 is below visibility on every animated path in a glTF: 1 mm of
        translation (the unit is metres), 0.1% of scale, and ~0.11 degrees
        of quaternion rotation. Bake residue lands well under it; real
        motion does not.

        Decodes BIN data for the named clip only -- pay for it once a
        cheaper extent test has already flagged something.

        Parameters:
            key (int/str): Animation index or clip name.
            tolerance (float): Per-component change treated as movement.

        Returns:
            tuple/None: ``(min_seconds, max_seconds)``, or ``None`` when the
            clip is static (or unreadable).
        """
        # Morph-target `weights` and CUBICSPLINE both pack several elements
        # per key; `_sampler_frames` folds them so every path is comparable.
        anim = self.animation(key)
        if anim is None:
            return None
        lows: List[float] = []
        highs: List[float] = []
        for sampler in anim.get("samplers") or []:
            times = self.accessor(sampler.get("input"))
            values = self.accessor(sampler.get("output"))
            if not times or not values:
                continue
            poses = self._sampler_frames(
                times, values, sampler.get("interpolation") or "LINEAR"
            )
            if poses is None:
                continue
            span = self._moving_span(poses, tolerance)
            if span is not None:
                first, last = span
                lows.append(times[first][0])
                highs.append(times[last][0])
        if not lows:
            return None
        return min(lows), max(highs)

    def channel_table(self, key: Union[int, str]) -> List[Dict[str, Any]]:
        """One row per channel: node index/name, path, interpolation, keys."""
        anim = self.animation(key)
        if anim is None:
            return []
        nodes = self.gltf.get("nodes") or []
        accessors = self.gltf.get("accessors") or []
        rows = []
        for channel in anim.get("channels") or []:
            target = channel.get("target") or {}
            sampler = (anim.get("samplers") or [])[channel.get("sampler", -1)]
            node_index = target.get("node")
            acc = accessors[sampler["input"]] if "input" in sampler else {}
            rows.append(
                {
                    "node": node_index,
                    "name": (
                        nodes[node_index].get("name")
                        if isinstance(node_index, int) and node_index < len(nodes)
                        else None
                    ),
                    "path": target.get("path"),
                    "interpolation": sampler.get("interpolation") or "LINEAR",
                    "keys": int(acc.get("count", 0)),
                }
            )
        return rows

    def sample(
        self,
        key: Union[int, str],
        node: Union[int, str],
        path: str,
        time: float,
    ) -> Optional[Tuple[float, ...]]:
        """Evaluate one channel at *time* seconds, or ``None`` when absent.

        LINEAR interpolates; STEP holds the previous key; CUBICSPLINE reads
        the value row (``values[1::3]``, matching :class:`GlbClips`) and
        interpolates it linearly — tangential easing is not reconstructed,
        which is exact at the keys and conservative between them. Times
        outside the sampled range clamp to the end keys.
        """
        anim = self.animation(key)
        index = self.node_index(node) if isinstance(node, str) else node
        if anim is None or index is None:
            return None
        for channel in anim.get("channels") or []:
            target = channel.get("target") or {}
            if target.get("node") != index or target.get("path") != path:
                continue
            sampler = (anim.get("samplers") or [])[channel.get("sampler", 0)]
            times = self.accessor(sampler.get("input"))
            values = self.accessor(sampler.get("output"))
            if not times or values is None:
                return None
            flat = [t[0] for t in times]
            interpolation = str(sampler.get("interpolation") or "LINEAR")
            if interpolation == "CUBICSPLINE":
                if len(values) != 3 * len(flat):
                    return None
                values = values[1::3]
            if len(values) != len(flat):
                return None
            if time <= flat[0]:
                return tuple(values[0])
            if time >= flat[-1]:
                return tuple(values[-1])
            hi = 0
            while flat[hi] < time:
                hi += 1
            # Key times are stored as float32 while callers compute *time* in
            # double (``(frame - zero) / fps``), so a sample landing exactly ON
            # a key can fall a few ULPs short of it -- measured at 6.4e-08 s for
            # frame 356 of a 30 fps clip. An exact compare then holds the
            # PREVIOUS key, which on a STEP channel (how visibility ships, as
            # zero-scale keys) reads as a one-frame pop that is not in the file.
            # The tolerance is relative because float32 keeps ~7 significant
            # digits, so the absolute error grows with the clip length.
            on_key = abs(flat[hi] - time) <= 1e-6 + abs(time) * 1e-6
            if on_key or interpolation == "STEP":
                at = hi if on_key else hi - 1
                return tuple(values[at])
            lo = hi - 1
            u = (time - flat[lo]) / (flat[hi] - flat[lo])
            return tuple(a + (b - a) * u for a, b in zip(values[lo], values[hi]))
        return None

    def nan_findings(self, huge: float = 1e7, deep: bool = False) -> List[str]:
        """Animation outputs that are NaN or beyond *huge* world units.

        Two tiers, because writers differ in whether a NaN datum poisons the
        stamped accessor bounds (many min/max implementations simply skip
        NaN): the default reads ``min``/``max`` fields only — free on any
        file size — and ``deep=True`` additionally DECODES every animation
        output accessor and scans the values themselves. Deep never touches
        geometry, so even on a production file it costs megabytes, not the
        mesh. Returns human-readable findings; empty means clean.
        """
        findings: List[str] = []
        accessors = self.gltf.get("accessors") or []
        for i, anim in enumerate(self.gltf.get("animations") or []):
            name = anim.get("name") or str(i)
            for j, sampler in enumerate(anim.get("samplers") or []):
                index = sampler.get("output")
                acc = (
                    accessors[index]
                    if isinstance(index, int) and 0 <= index < len(accessors)
                    else {}
                )
                for bound in ("min", "max"):
                    for value in acc.get(bound) or []:
                        if value != value or abs(value) > huge:
                            findings.append(
                                f"clip {name!r} sampler {j} {bound} = {value!r}"
                            )
                            break
                if not deep:
                    continue
                values = self.accessor(index)
                for element in values or []:
                    if any(v != v or abs(v) > huge for v in element):
                        findings.append(
                            f"clip {name!r} sampler {j} value = {element!r}"
                        )
                        break
        return findings

    # ---- hierarchy / worlds ------------------------------------------------

    def node_index(self, name: str) -> Optional[int]:
        """First node index with *name*, or ``None``."""
        if self._names is None:
            self._names = {}
            for i, node in enumerate(self.gltf.get("nodes") or []):
                self._names.setdefault(node.get("name"), i)
        return self._names.get(name)

    def parent_of(self, index: int) -> Optional[int]:
        """Parent node index, or ``None`` for a root."""
        if self._parents is None:
            self._parents = {}
            for i, node in enumerate(self.gltf.get("nodes") or []):
                for child in node.get("children") or []:
                    self._parents[child] = i
        return self._parents.get(index)

    def local_matrix(
        self,
        index: int,
        time: Optional[float] = None,
        animation: Union[int, str, None] = None,
    ) -> List[List[float]]:
        """Node *index*'s local matrix, animated when *time* is given."""
        node = (self.gltf.get("nodes") or [])[index]
        if "matrix" in node:
            # glTF forbids animating a ``matrix`` node, so the static matrix
            # wins even when time/animation are supplied — falling through to
            # the TRS branch would sample nothing and silently compose
            # identity in place of the authored transform.
            return self._rows_from_flat(node["matrix"])
        t = r = s = None
        if time is not None and animation is not None:
            t = self.sample(animation, index, "translation", time)
            r = self.sample(animation, index, "rotation", time)
            s = self.sample(animation, index, "scale", time)
        t = t or tuple(node.get("translation") or (0.0, 0.0, 0.0))
        r = r or tuple(node.get("rotation") or (0.0, 0.0, 0.0, 1.0))
        s = s or tuple(node.get("scale") or (1.0, 1.0, 1.0))
        return self._trs_matrix(t, r, s)

    def world_matrix(
        self,
        node: Union[int, str],
        time: Optional[float] = None,
        animation: Union[int, str, None] = None,
    ) -> Optional[List[List[float]]]:
        """World matrix of *node* (name or index), or ``None`` when absent."""
        index = self.node_index(node) if isinstance(node, str) else node
        if index is None:
            return None
        m = self.local_matrix(index, time, animation)
        parent = self.parent_of(index)
        while parent is not None:
            m = self._mat_mul(m, self.local_matrix(parent, time, animation))
            parent = self.parent_of(parent)
        return m

    def world_position(
        self,
        node: Union[int, str],
        time: Optional[float] = None,
        animation: Union[int, str, None] = None,
    ) -> Optional[Tuple[float, float, float]]:
        """World-space translation of *node*, or ``None`` when absent."""
        m = self.world_matrix(node, time, animation)
        return (m[3][0], m[3][1], m[3][2]) if m else None

    def walk(self) -> Iterator[Tuple[int, Optional[str]]]:
        """Yield ``(index, name)`` for every node, in file order."""
        for i, node in enumerate(self.gltf.get("nodes") or []):
            yield i, node.get("name")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
