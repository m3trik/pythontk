# !/usr/bin/python
# coding=utf-8
"""Unit tests for GlbReader, FbxFile and ExportVerifier.

Fixtures are built byte-by-byte in the test — a minimal but well-formed GLB
(JSON + BIN chunks, animation, skin, images) and a minimal binary FBX
(64-bit record layout) — so the suite needs no checked-in binaries and no
network.

Run with:
    python -m pytest test_export_verify.py -v
    python test_export_verify.py
"""

import json
import os
import shutil
import struct
import zlib
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk.file_utils.mesh_convert.export_verify import (
    ExportVerifier,
    FAIL,
    PASS,
    SKIP,
    WARN,
    _main,
)
from pythontk.file_utils.mesh_convert.fbx_file import FbxFile
from pythontk.file_utils.mesh_convert.glb.reader import GlbReader


# ---------------------------------------------------------------------------
# GLB fixture builder
# ---------------------------------------------------------------------------


def _pad4(data: bytes, pad: bytes) -> bytes:
    return data + pad * (-len(data) % 4)


def build_glb(
    path: str,
    nan_output: bool = False,
    drop_ibm: bool = False,
    stub_skin: bool = False,
    bad_skeleton: bool = False,
    undeclared_basisu: bool = False,
    clip_end: float = 0.5,
    clip_name: str = "Shot_1",
    zero_frame: float = None,
    hold_frames: float = 0.0,
    hold_moves: bool = False,
    data_export: dict = None,
) -> str:
    """Write a tiny, valid GLB: two nodes, one clip, one skin, two images.

    *data_export* (``{channel key: value}``) adds a ``data_export`` carrier
    node publishing those channels, as a conversion leaves them."""
    bin_parts = []
    views = []
    accessors = []

    def accessor(values, type_, component=5126, fmt="f"):
        flat = [c for v in values for c in (v if isinstance(v, tuple) else (v,))]
        payload = _pad4(struct.pack(f"<{len(flat)}{fmt}", *flat), b"\x00")
        offset = sum(len(p) for p in bin_parts)
        bin_parts.append(payload)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(payload)})
        width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[type_]
        columns = [flat[i::width] for i in range(width)]
        accessors.append(
            {
                "bufferView": len(views) - 1,
                "componentType": component,
                "count": len(values),
                "type": type_,
                "min": [min(c) for c in columns],
                "max": [max(c) for c in columns],
            }
        )
        return len(accessors) - 1

    key_times = [0.0, clip_end]
    translate_values = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    scale_values = [(1.0, 1.0, 1.0), (3.0, 3.0, 3.0)]
    if nan_output:
        scale_values[1] = (float("nan"), 3.0, 3.0)
    if hold_frames:
        # A bake's trailing pad: one more key further down the timeline that
        # REPEATS the last pose, so the clip occupies frames it does not
        # animate. With hold_moves it genuinely animates instead.
        key_times.append(clip_end + hold_frames / 30.0)
        translate_values.append((7.0, 0.0, 0.0) if hold_moves else translate_values[-1])
        scale_values.append(scale_values[-1])
    times = accessor(key_times, "SCALAR")
    translations = accessor(translate_values, "VEC3")
    scales = accessor(scale_values, "VEC3")
    ibm = accessor([tuple(float(r == c) for r in range(4) for c in range(4))], "MAT4")
    positions = accessor([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], "VEC3")

    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    ktx2 = b"\xabKTX 20\xbb\r\n\x1a\n" + b"\x00" * 8
    images = []
    for payload, mime in ((png, "image/png"), (ktx2, "image/ktx2")):
        offset = sum(len(p) for p in bin_parts)
        bin_parts.append(_pad4(payload, b"\x00"))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(payload)})
        images.append({"bufferView": len(views) - 1, "mimeType": mime})

    gltf = {
        "asset": {"version": "2.0", "generator": "test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "root", "children": [1], "translation": [0.0, 1.0, 0.0]},
            {"name": "arm", "mesh": 0, "skin": 0},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": positions}}]}],
        "skins": [{"joints": [1], "inverseBindMatrices": ibm}],
        "images": images,
        "textures": [
            {"source": 0, "extensions": {"KHR_texture_basisu": {"source": 1}}}
        ],
        "materials": [{"name": "mat"}],
        "extensionsUsed": [] if undeclared_basisu else ["KHR_texture_basisu"],
        "animations": [
            {
                "name": clip_name,
                **(
                    {} if zero_frame is None else {"extras": {"zero_frame": zero_frame}}
                ),
                "channels": [
                    {
                        "sampler": 0,
                        "target": {"node": 1, "path": "translation"},
                    },
                    {"sampler": 1, "target": {"node": 1, "path": "scale"}},
                ],
                "samplers": [
                    {"input": times, "output": translations},
                    {
                        "input": times,
                        "output": scales,
                        "interpolation": "STEP",
                    },
                ],
            }
        ],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": sum(len(p) for p in bin_parts)}],
    }
    if drop_ibm:
        gltf["skins"][0].pop("inverseBindMatrices")
    if stub_skin:
        # Converter-style bookkeeping: IBM-less and referenced by nothing.
        gltf["skins"].append({"joints": [0]})
    if bad_skeleton:
        # A skeleton the skin's joints do not hang under: a stray root node.
        gltf["nodes"].append({"name": "stray"})
        gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)
        gltf["skins"][0]["skeleton"] = len(gltf["nodes"]) - 1
    if data_export:
        extras = {key: json.dumps(value) for key, value in data_export.items()}
        gltf["nodes"].append({"name": "data_export", "extras": extras})
        gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)

    json_chunk = _pad4(json.dumps(gltf).encode("utf-8"), b" ")
    bin_chunk = _pad4(b"".join(bin_parts), b"\x00")
    body = (
        struct.pack("<I4s", len(json_chunk), b"JSON")
        + json_chunk
        + struct.pack("<I4s", len(bin_chunk), b"BIN\x00")
        + bin_chunk
    )
    with open(path, "wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
    return path


# ---------------------------------------------------------------------------
# FBX fixture builder (64-bit record layout, version 7700)
# ---------------------------------------------------------------------------


class Longs(list):
    """An array property written with the ``l`` (int64 array) tag -- a
    curve's ``KeyTime``. *compress* writes it zlib-encoded, as Maya does."""

    def __init__(self, values, compress=False):
        super().__init__(values)
        self.compress = compress


def _fbx_prop(value) -> bytes:
    if isinstance(value, bytes):
        return b"S" + struct.pack("<I", len(value)) + value
    if isinstance(value, Longs):
        data = struct.pack(f"<{len(value)}q", *value)
        if value.compress:
            data = zlib.compress(data)
        return (
            b"l"
            + struct.pack("<III", len(value), int(value.compress), len(data))
            + data
        )
    if isinstance(value, int):
        return b"L" + struct.pack("<q", value)
    raise TypeError(type(value))


def _fbx_record(name: str, props, children, offset: int) -> bytes:
    prop_bytes = b"".join(_fbx_prop(p) for p in props)
    name_bytes = name.encode("ascii")
    header_len = 24 + 1 + len(name_bytes)
    body = b""
    child_offset = offset + header_len + len(prop_bytes)
    for child in children:
        chunk = _fbx_record(child[0], child[1], child[2], child_offset)
        body += chunk
        child_offset += len(chunk)
    if children:
        body += b"\x00" * 25  # nested NULL sentinel
    end = offset + header_len + len(prop_bytes) + len(body)
    return (
        struct.pack("<QQQB", end, len(props), len(prop_bytes), len(name_bytes))
        + name_bytes
        + prop_bytes
        + body
    )


def build_fbx(path: str, takes=("Shot_1",)) -> str:
    """Write a tiny, valid binary FBX with Objects + Connections sections."""
    objects_children = [
        ("Model", [1001, b"cube\x00\x01Model", b"Mesh"], []),
        ("AnimationCurveNode", [2001, b"T\x00\x01AnimCurveNode", b""], []),
    ]
    for i, take in enumerate(takes):
        objects_children.append(
            (
                "AnimationStack",
                [3001 + i, take.encode() + b"\x00\x01AnimStack", b""],
                [],
            )
        )
    roots = [
        ("Objects", [], objects_children),
        ("Connections", [], [("C", [b"OO", 2001, 1001], [])]),
    ]
    payload = b""
    offset = len(b"Kaydara FBX Binary  \x00\x1a\x00") + 4
    for name, props, children in roots:
        chunk = _fbx_record(name, props, children, offset)
        payload += chunk
        offset += len(chunk)
    payload += b"\x00" * 25  # top-level NULL sentinel
    with open(path, "wb") as handle:
        handle.write(b"Kaydara FBX Binary  \x00\x1a\x00")
        handle.write(struct.pack("<I", 7700))
        handle.write(payload)
    return path


#: FBX ticks per frame at the fixtures' 30 fps.
TICK = 46186158000 // 30


def build_take_fbx(path: str, takes, names=None) -> str:
    """A binary FBX whose takes animate real curves.

    *takes* is ``{take: {(model, channel): (first_frame, last_frame)}}``; each
    curve gets a key on every frame of its span (zlib-encoded, as Maya writes
    them), wired stack <- layer <- curve node (on the model's
    ``Lcl Translation``) <- curve, the graph a DCC writes. A key
    ``(model, channel, layer)`` keys its curve on that layer of the take
    instead of ``BaseLayer``. *names* writes a model under another display
    name -- ``{"b": "a"}`` is two models both called ``a``, uids 1000 and
    1001 in sorted key order.
    """
    names = names or {}
    models = sorted({key[0] for curves in takes.values() for key in curves})
    model_ids = {model: 1000 + i for i, model in enumerate(models)}
    objects = [
        (
            "Model",
            [uid, names.get(model, model).encode() + b"\x00\x01Model", b"Mesh"],
            [],
        )
        for model, uid in model_ids.items()
    ]
    connections = []
    uid = 5000
    for take, curves in takes.items():
        stack, layer = uid, uid + 1
        uid += 2
        objects += [
            ("AnimationStack", [stack, take.encode() + b"\x00\x01AnimStack", b""], []),
            ("AnimationLayer", [layer, b"BaseLayer\x00\x01AnimLayer", b""], []),
        ]
        connections.append(("C", [b"OO", layer, stack], []))
        layers = {"BaseLayer": layer}
        for (model, channel, *on), (first, last) in curves.items():
            layer_name = on[0] if on else "BaseLayer"
            if layer_name not in layers:
                layers[layer_name] = uid
                uid += 1
                objects.append(
                    (
                        "AnimationLayer",
                        [uid - 1, layer_name.encode() + b"\x00\x01AnimLayer", b""],
                        [],
                    )
                )
                connections.append(("C", [b"OO", uid - 1, stack], []))
            layer = layers[layer_name]
            node, curve = uid, uid + 1
            uid += 2
            times = Longs(range(first * TICK, (last + 1) * TICK, TICK), compress=True)
            objects += [
                ("AnimationCurveNode", [node, b"T\x00\x01AnimCurveNode", b""], []),
                (
                    "AnimationCurve",
                    [curve, b"\x00\x01AnimCurve", b""],
                    [("KeyTime", [times], [])],
                ),
            ]
            connections += [
                ("C", [b"OO", node, layer], []),
                ("C", [b"OP", node, model_ids[model], b"Lcl Translation"], []),
                ("C", [b"OP", curve, node, channel.encode()], []),
            ]
    roots = [("Objects", [], objects), ("Connections", [], connections)]
    payload = b""
    offset = len(b"Kaydara FBX Binary  \x00\x1a\x00") + 4
    for name, props, children in roots:
        chunk = _fbx_record(name, props, children, offset)
        payload += chunk
        offset += len(chunk)
    payload += b"\x00" * 25
    with open(path, "wb") as handle:
        handle.write(b"Kaydara FBX Binary  \x00\x1a\x00")
        handle.write(struct.pack("<I", 7700))
        handle.write(payload)
    return path


def build_truncated_fbx(path: str, inside: bytes = b"KeyTime") -> str:
    """A take FBX cut short part way through the first record named *inside*:
    by default inside its zlib-encoded key array, the cut a span read
    decompresses; ``b"Connections"`` cuts where nothing is decoded at all."""
    build_take_fbx(
        path, {"Take 001": {("lift", "d|Y"): (1, 2000), ("door", "d|Y"): (1, 2000)}}
    )
    with open(path, "rb") as handle:
        data = handle.read()
    at = data.index(inside) + len(inside)
    cut = data.index(b"\x78\x9c", at) + 40 if inside == b"KeyTime" else at + 30
    with open(path, "wb") as handle:
        handle.write(data[:cut])
    return path


def build_sidecar(path: str, takes, span=None, fps=30.0, clip_mode=None) -> str:
    """*span* publishes ``clip_span["*"]`` -- the origin clips are cut against.
    *clip_mode* declares the run's Animation Clips mode on ``shot_metadata``."""
    data_export = {"fbx_takes": list(takes)}
    if clip_mode is not None:
        data_export["shot_metadata"] = {
            "version": 1,
            "fps": fps,
            "shots": [],
            "clip_mode": clip_mode,
        }
    if span is not None:
        data_export["visibility_tracks"] = {
            "fps": fps,
            "version": 1,
            "tracks": [],
            "clip_span": {"*": list(span)},
        }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "format": 3,
                "hierarchy": {"paths": ["root", "root|arm"]},
                "data_export": data_export,
            },
            handle,
        )
    return path


# ---------------------------------------------------------------------------


class _FixtureCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="export_verify_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def path(self, name: str) -> str:
        return os.path.join(self.tmp, name)


class TestGlbReader(_FixtureCase):
    def setUp(self):
        super().setUp()
        self.reader = GlbReader.load(build_glb(self.path("asset.glb")))

    def test_counts_and_mimes(self):
        counts = self.reader.counts()
        self.assertEqual(counts["nodes"], 2)
        self.assertEqual(counts["animations"], 1)
        self.assertEqual(self.reader.image_mimes(), {"image/png": 1, "image/ktx2": 1})

    def test_accessor_decode(self):
        spans = self.reader.clip_spans()
        self.assertIn("Shot_1", spans)
        translations = self.reader.accessor(1)
        self.assertEqual(translations, [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)])

    def test_an_accessor_whose_bytes_are_not_in_the_bin_is_unreadable(self):
        """Added 2026-09-23: a view on another buffer, an embedded buffer that
        names an external file, or a compressed view decoded the BIN's bytes
        at its offsets -- plausible numbers, none of them the accessor's. Now
        the "unreadable" None every caller already handles."""
        view = self.reader.gltf["accessors"][1]["bufferView"]
        cases = {
            "another buffer": lambda g: g["bufferViews"][view].update(buffer=1),
            "external": lambda g: g["buffers"][0].update(uri="asset.bin"),
            "compressed": lambda g: g["bufferViews"][view].update(
                extensions={"EXT_meshopt_compression": {}}
            ),
        }
        for name, edit in cases.items():
            with self.subTest(name):
                reader = GlbReader.load(self.path("asset.glb"))
                edit(reader.gltf)
                self.assertIsNone(reader.accessor(1))

    def test_motion_span_drops_a_trailing_hold(self):
        """A held pose occupies frames without animating them."""
        reader = GlbReader.load(
            build_glb(self.path("held.glb"), clip_end=0.5, hold_frames=48)
        )
        span = reader.motion_span("Shot_1")
        self.assertIsNotNone(span)
        self.assertAlmostEqual(span[0], 0.0, places=6)
        self.assertAlmostEqual(span[1], 0.5, places=6)
        # The KEY extent still runs out to the pad -- the two differ, which
        # is the whole point of measuring motion separately.
        self.assertAlmostEqual(reader.clip_spans()["Shot_1"][1], 2.1, places=5)

    def test_motion_span_keeps_a_tail_that_actually_moves(self):
        reader = GlbReader.load(
            build_glb(
                self.path("moving.glb"),
                clip_end=0.5,
                hold_frames=48,
                hold_moves=True,
            )
        )
        self.assertAlmostEqual(reader.motion_span("Shot_1")[1], 2.1, places=5)

    def test_motion_span_sees_motion_that_accumulates_below_the_tolerance(self):
        """A slow pan is motion, even when no single step clears the threshold.

        ``motion_span`` compared each key only against the one before it, so a
        channel drifting by less than *tolerance* per key read as perfectly
        static however far it travelled. Measured: 400 keys stepping 0.0005
        each -- half the 1e-3 default -- move 0.2 m in total (a 13-second
        camera pan) and the method returned ``None``. The same 0.2 m delivered
        as one jump between two keys was detected, so the gate's answer
        depended on how the motion was distributed, not whether it happened.

        This matters because ``check_clips_vs_takes`` FAILs on motion past the
        last take and only WARNs on an inert margin -- so a slow drift out
        there was reported as a held pose and shipped.
        """
        reader = self._stub_reader(
            times=[(i / 30.0,) for i in range(400)],
            values=[(i * 0.0005, 0.0, 0.0) for i in range(400)],
        )
        span = reader.motion_span(0)
        self.assertIsNotNone(
            span, "0.2 m of travel reported as static because each step was small"
        )
        # Motion runs essentially the whole clip: it starts near the top and
        # continues to the end.
        self.assertLess(span[0], 0.5)
        self.assertGreater(span[1], 12.0)

    def test_motion_span_still_locates_a_single_late_jump(self):
        """The drift fix must not smear a discrete move across the clip: a
        channel that holds and then jumps once still reports only the jump."""
        reader = self._stub_reader(
            times=[(i / 30.0,) for i in range(400)],
            values=[(0.0, 0.0, 0.0)] * 399 + [(0.2, 0.0, 0.0)],
        )
        span = reader.motion_span(0)
        self.assertIsNotNone(span)
        self.assertAlmostEqual(span[0], 398 / 30.0, places=6)
        self.assertAlmostEqual(span[1], 399 / 30.0, places=6)

    def test_motion_span_drops_a_leading_hold_too(self):
        """Symmetry: a clip that waits, moves, then holds reports only the
        middle. The trailing half was covered; the leading half was not."""
        values = [(0.0, 0.0, 0.0)] * 10 + [(1.0, 0.0, 0.0)] * 10
        reader = self._stub_reader(
            times=[(i / 30.0,) for i in range(20)], values=values
        )
        span = reader.motion_span(0)
        self.assertAlmostEqual(span[0], 9 / 30.0, places=6)
        self.assertAlmostEqual(span[1], 10 / 30.0, places=6)

    @staticmethod
    def _stub_reader(times, values, interpolation="LINEAR"):
        """A GlbReader whose two decode seams are stubbed -- no file needed,
        so a sampler shape can be stated directly."""
        reader = GlbReader.__new__(GlbReader)
        reader.animation = lambda key: {
            "samplers": [{"input": 0, "output": 1, "interpolation": interpolation}]
        }
        reader.accessor = lambda i: times if i == 0 else values
        return reader

    def test_motion_span_is_none_when_nothing_moves(self):
        """No motion at all is not a zero-length span -- it is no span."""
        reader = GlbReader.load(self.path("asset.glb"))
        # Every channel changes by less than the threshold it is judged on.
        self.assertIsNone(reader.motion_span("Shot_1", tolerance=10.0))

    def test_sampler_frames_folds_multi_element_keys(self):
        """A key is not always one element -- and the ones that are not are
        exactly the channels a naive length check drops.

        ``weights`` stores one scalar per morph target per key, so a two-target
        channel has twice as many outputs as inputs; CUBICSPLINE triples any
        of them into (in-tangents, values, out-tangents). Comparing raw output
        counts against key counts skipped both, which would have read a morph
        animation as perfectly still.
        """
        fold = GlbReader._sampler_frames
        times = [(0.0,), (1.0,)]

        # Two morph targets, LINEAR: four scalars fold into two poses.
        self.assertEqual(
            fold(times, [(0.0,), (1.0,), (0.5,), (0.25,)], "LINEAR"),
            [(0.0, 1.0), (0.5, 0.25)],
        )
        # CUBICSPLINE VEC3: the middle of each triple is the pose.
        cubic = [
            (9.0, 9.0, 9.0),
            (1.0, 2.0, 3.0),
            (8.0, 8.0, 8.0),
            (7.0, 7.0, 7.0),
            (4.0, 5.0, 6.0),
            (6.0, 6.0, 6.0),
        ]
        self.assertEqual(
            fold(times, cubic, "CUBICSPLINE"),
            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        )
        # Ragged or impossible pairings are refused, never guessed at.
        self.assertIsNone(fold(times, [(0.0,), (1.0,), (2.0,)], "LINEAR"))
        self.assertIsNone(fold(times, [(0.0,), (1.0,)], "CUBICSPLINE"))
        self.assertIsNone(fold([], [(0.0,)], "LINEAR"))

    def test_motion_span_on_an_unknown_clip_is_none(self):
        self.assertIsNone(self.reader.motion_span("nope"))

    def test_sample_linear_midpoint(self):
        value = self.reader.sample("Shot_1", "arm", "translation", 0.25)
        self.assertAlmostEqual(value[0], 1.0, places=6)

    def test_sample_step_holds_previous_key(self):
        value = self.reader.sample("Shot_1", "arm", "scale", 0.49)
        self.assertEqual(value, (1.0, 1.0, 1.0))
        value = self.reader.sample("Shot_1", "arm", "scale", 0.5)
        self.assertEqual(value, (3.0, 3.0, 3.0))

    def test_sample_lands_on_a_step_key_whose_time_is_not_representable(self):
        """A frame that falls exactly ON a STEP key must read that key.

        glTF stores key times as float32 while callers compute the sample time
        in double (``(frame - zero) / fps``). For any frame whose time is not
        exactly representable the two differ in the last bits -- measured at
        6.4e-08 s for frame 356 of a 30 fps clip -- so an exact ``==`` compare
        falls through to "hold the previous key" and reports the transition one
        frame late. On a visibility channel, which ships as STEP zero-scale
        keys, that reads as a one-frame pop that is not in the file.
        """
        end = 88 / 30.0  # frame 88 at 30 fps: not exactly representable
        reader = GlbReader.load(build_glb(self.path("stepkey.glb"), clip_end=end))
        self.assertEqual(reader.sample("Shot_1", "arm", "scale", end), (3.0, 3.0, 3.0))

    def test_world_position_composes_parent(self):
        # arm rides root's static +1 Y; at t=0.25 its own X is 1.0.
        x, y, z = self.reader.world_position("arm", time=0.25, animation="Shot_1")
        self.assertAlmostEqual(x, 1.0, places=6)
        self.assertAlmostEqual(y, 1.0, places=6)
        self.assertAlmostEqual(z, 0.0, places=6)

    def test_matrix_node_wins_over_animation_args(self):
        """A ``matrix`` node keeps its static matrix even when sampling args
        are supplied — glTF forbids animating such a node, and the TRS
        fallback would silently compose identity instead."""
        gltf = self.reader.gltf
        gltf["nodes"].append(
            {
                "name": "matrix_node",
                # column-major: translation (5, 6, 7)
                "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1],
            }
        )
        gltf["scenes"][0]["nodes"].append(2)
        self.reader._names = None  # rebuild the name cache
        position = self.reader.world_position(
            "matrix_node", time=0.25, animation="Shot_1"
        )
        self.assertEqual(position, (5.0, 6.0, 7.0))

    def test_clip_end_frames(self):
        low, high, end_frame = self.reader.clip_spans(fps=30.0)["Shot_1"]
        self.assertEqual((low, high, end_frame), (0.0, 0.5, 15))

    def test_nan_findings_shallow_vs_deep(self):
        """Bounds-only misses a NaN the writer's min/max skipped; deep finds it.

        Python's ``min``/``max`` (and many writers') simply skip NaN, so the
        fixture's stamped bounds are clean while the data is poisoned — the
        exact writer-dependent gap the two tiers exist for.
        """
        self.assertEqual(self.reader.nan_findings(deep=True), [])
        bad = GlbReader.load(build_glb(self.path("nan.glb"), nan_output=True))
        self.assertEqual(bad.nan_findings(), [])  # bounds look clean
        self.assertTrue(bad.nan_findings(deep=True))


class TestFbxFile(_FixtureCase):
    def setUp(self):
        super().setUp()
        self.fbx = FbxFile.load(build_fbx(self.path("asset.fbx"), ("Shot_1", "Shot_2")))

    def test_version_and_sections(self):
        self.assertEqual(self.fbx.version, 7700)
        self.assertIsNotNone(self.fbx.section("Objects"))
        self.assertIsNotNone(self.fbx.section("Connections"))

    def test_census_and_takes(self):
        census = self.fbx.objects_census()
        self.assertEqual(census["Model"], 1)
        self.assertEqual(census["AnimationStack"], 2)
        self.assertEqual(self.fbx.take_names(), ["Shot_1", "Shot_2"])

    def test_connections(self):
        rows = self.fbx.connections()
        self.assertEqual(rows, [("OO", 2001, 1001, None)])

    def test_raw_payloads_can_be_skipped_without_changing_the_census(self):
        """``raw_payloads=False`` leaves embedded media on disk -- a census
        that counts nodes and takes has no reason to hold hundreds of MB."""
        from test_fbx_media import build_fbx as build_media_fbx, png_bytes

        path = build_media_fbx(self.path("media.fbx"), {"wall.png": png_bytes((8, 8))})
        full, lean = FbxFile.load(path), FbxFile.load(path, raw_payloads=False)
        self.assertEqual(full.objects_census(), lean.objects_census())
        content = [
            child["props"][0]
            for fbx in (full, lean)
            for record in fbx.iter_objects()
            if record["name"] == "Video"
            for child in record["children"]
            if child["name"] == "Content"
        ]
        self.assertIsInstance(content[0], bytes)
        self.assertEqual(content[1], ("RAW", len(content[0])))

    def test_take_curves_reads_each_take_s_channels_and_their_spans(self):
        """The census a take split is judged by -- read from the key times
        alone (``span_arrays``), with the same answer a full decode gives."""
        path = build_take_fbx(
            self.path("takes.fbx"),
            {
                "Take 001": {("lift", "d|Y"): (1, 100), ("door", "d|Y"): (1, 100)},
                "A02": {("door", "d|Y"): (51, 100)},
            },
        )
        lean = FbxFile.load(path, span_arrays=("KeyTime",), raw_payloads=False)
        full = FbxFile.load(path, decode_arrays=("KeyTime",))
        curves = lean.take_curves()

        self.assertEqual(curves, full.take_curves())
        self.assertEqual(
            curves["A02"],
            {("door", "Lcl Translation", "d|Y"): (51 * TICK, 100 * TICK, 50)},
        )
        self.assertEqual(len(curves["Take 001"]), 2)

    def test_take_spans_is_each_take_s_key_extent_over_every_curve(self):
        """What FBX2glTF sizes a take by: the first and last key of ANY curve
        in it, whatever the take declares (measured 2026-10-04 on FBX2glTF
        0.13.1: a camera's focal length keyed at 3 opened a take whose
        transforms start at 10, and the take's own 1-120 span was ignored).
        So the converter rebases each clip onto exactly this first key."""
        path = build_take_fbx(
            self.path("spans.fbx"),
            {
                "Take 001": {("lift", "d|Y"): (10, 100), ("door", "d|Y"): (3, 120)},
                "A02": {("door", "d|Y"): (51, 100)},
            },
        )
        spans = FbxFile.load(
            path, span_arrays=("KeyTime",), raw_payloads=False
        ).take_spans()

        second = TICK / FbxFile.TICKS_PER_SECOND
        self.assertEqual(sorted(spans), ["A02", "Take 001"])
        self.assertAlmostEqual(spans["Take 001"][0], 3 * second)
        self.assertAlmostEqual(spans["Take 001"][1], 120 * second)
        self.assertAlmostEqual(spans["A02"][0], 51 * second)

    def test_take_spans_leaves_out_a_take_it_cannot_measure(self):
        """Read without key times, nothing can be measured -- and nothing is
        guessed."""
        path = build_take_fbx(self.path("spans.fbx"), {"T": {("a", "d|X"): (1, 10)}})
        self.assertEqual(FbxFile.load(path).take_spans(), {})

    def test_take_curves_without_key_times_still_counts_them(self):
        path = build_take_fbx(self.path("takes.fbx"), {"T": {("a", "d|X"): (1, 10)}})
        self.assertEqual(
            FbxFile.load(path).take_curves()["T"],
            {("a", "Lcl Translation", "d|X"): (None, None, 10)},
        )

    def test_same_named_targets_and_layered_curves_all_count(self):
        """A channel was keyed by its target's NAME, so a later curve replaced
        an earlier one: two models sharing a name (a production 12-shot Maya
        FBX holds 478 such names -- each referenced module brings its own), or
        one channel keyed on two layers. ``take_spans`` then measured whichever
        curve was written last: a stack keyed 1-100 measured 5-50, the origin
        every clip is cut against (2026-10-04 review). A name that is not
        unique among the targets is labelled ``name#uid``; a channel keyed on
        two layers is their union."""
        second = TICK / FbxFile.TICKS_PER_SECOND
        wide, narrow = (("a", "d|Y"), (1, 100)), (("b", "d|Y"), (5, 50))
        for order in ((wide, narrow), (narrow, wide)):
            with self.subTest("twins", order=[key for key, _ in order]):
                path = build_take_fbx(
                    self.path("twins.fbx"), {"Take 001": dict(order)}, names={"b": "a"}
                )
                fbx = FbxFile.load(path, span_arrays=("KeyTime",))
                self.assertEqual(
                    sorted(fbx.take_curves()["Take 001"]),
                    [
                        ("a#1000", "Lcl Translation", "d|Y"),
                        ("a#1001", "Lcl Translation", "d|Y"),
                    ],
                )
                first, last = fbx.take_spans()["Take 001"]
                self.assertAlmostEqual(first, 1 * second)
                self.assertAlmostEqual(last, 100 * second)
        base, layer = (("a", "d|Y"), (10, 50)), (("a", "d|Y", "Layer1"), (1, 100))
        for order in ((base, layer), (layer, base)):
            with self.subTest("layers", order=[key for key, _ in order]):
                path = build_take_fbx(self.path("layers.fbx"), {"T": dict(order)})
                self.assertEqual(
                    FbxFile.load(path, span_arrays=("KeyTime",)).take_curves()["T"],
                    {("a", "Lcl Translation", "d|Y"): (1 * TICK, 100 * TICK, 141)},
                )

    def test_a_truncated_file_is_refused_wherever_it_is_cut(self):
        """``ValueError``, as documented, whatever the cut lands in. Inside a
        zlib-encoded key array the span read raised ``zlib.error`` -- past
        every caller's ``(OSError, ValueError)``, out of a conversion's
        timeout budget (2026-10-04 review); anywhere else the reader parsed
        what was left and called the file sound."""
        for inside in (b"KeyTime", b"Connections"):
            path = build_truncated_fbx(self.path("cut.fbx"), inside)
            for kwargs in ({}, {"span_arrays": ("KeyTime",)}, {"decode_arrays": True}):
                with self.subTest(inside=inside, read=kwargs):
                    with self.assertRaises(ValueError) as caught:
                        FbxFile.load(path, raw_payloads=False, **kwargs)
                    self.assertIn("Truncated or corrupt FBX", str(caught.exception))

    def test_not_an_fbx(self):
        junk = self.path("junk.fbx")
        with open(junk, "wb") as handle:
            handle.write(b"not an fbx")
        self.assertFalse(FbxFile.is_fbx(junk))
        with self.assertRaises(ValueError):
            FbxFile.load(junk)


class TestFbxTakeChannels(_FixtureCase):
    """``check_fbx_take_channels``: a declared take that lost a channel to the
    split, or stops short of its window, is a shot Unity plays wrong."""

    WINDOWS = [
        {"name": "A01", "start": 1, "end": 50},
        {"name": "A02", "start": 51, "end": 100},
    ]

    def _run(self, takes, clip_mode=None, names=None):
        fbx = build_take_fbx(self.path("asset.fbx"), takes, names=names)
        sidecar = build_sidecar(self.path("s.json"), self.WINDOWS, clip_mode=clip_mode)
        rows = (
            ExportVerifier(fbx=fbx, sidecar=sidecar)
            .run(["check_fbx_take_channels"])
            .rows
        )
        return [(row.status, row.detail) for row in rows]

    def _exact(self):
        whole = {("lift", "d|Y"): (1, 100), ("door", "d|Y"): (1, 100)}
        return {
            "Take 001": whole,
            "A01": {key: (1, 50) for key in whole},
            "A02": {key: (51, 100) for key in whole},
        }

    def test_exact_slices_pass(self):
        rows = self._run(self._exact())
        self.assertEqual([status for status, _ in rows], ["PASS"], rows)
        self.assertIn("2 take(s)", rows[0][1])

    def test_a_channel_lost_to_the_split_fails_naming_it(self):
        """The measured loss: a node keyed only inside A01 has no channel in A02."""
        takes = self._exact()
        del takes["A02"][("lift", "d|Y")]
        rows = self._run(takes)
        self.assertEqual([status for status, _ in rows], ["FAIL"], rows)
        self.assertIn("A02", rows[0][1])
        self.assertIn("lift.Lcl Translation.d|Y", rows[0][1])

    def test_a_take_keyed_short_of_its_window_fails(self):
        """The other half: a take whose keys stop at its last in-window key."""
        takes = self._exact()
        takes["A01"][("door", "d|Y")] = (1, 40)
        rows = self._run(takes)
        self.assertEqual([status for status, _ in rows], ["FAIL"], rows)
        self.assertIn("short of the take's 1-50 window", rows[0][1])

    def test_a_file_without_a_whole_timeline_take_is_held_to_the_windows(self):
        takes = self._exact()
        del takes["Take 001"]
        rows = self._run(takes)
        self.assertEqual([status for status, _ in rows], ["PASS"], rows)
        self.assertIn("no whole-timeline take", rows[0][1])

    def test_full_sequence_only_has_no_takes_to_slice(self):
        rows = self._run({"Take 001": {("lift", "d|Y"): (1, 100)}}, clip_mode="full")
        self.assertEqual([status for status, _ in rows], ["SKIP"], rows)

    def test_a_take_with_no_curves_fails(self):
        """A declared take whose stack carries no curve at all -- the split's
        worst case, every channel lost -- read as ABSENT: skipped as
        ``fbx_takes``' to name, and that gate checks names alone, so it passed
        both; with every take hollow, "no declared take is in the file"
        (2026-10-04 review). Present, it is judged like any take."""
        takes = self._exact()
        takes["A02"] = {}
        rows = self._run(takes)
        self.assertEqual([status for status, _ in rows], ["FAIL"], rows)
        self.assertIn("A02: 2 channel(s)", rows[0][1])

        takes["A01"] = {}
        rows = self._run(takes)
        self.assertEqual([status for status, _ in rows], ["FAIL", "FAIL"], rows)

    def test_a_take_with_no_curves_is_legal_without_a_whole_timeline_take(self):
        """Blender windows each take itself, so there is nothing to compare a
        shot against: one that animates nothing is a still shot."""
        takes = self._exact()
        del takes["Take 001"]
        takes["A02"] = {}
        rows = self._run(takes)
        self.assertEqual([status for status, _ in rows], ["PASS"], rows)
        self.assertIn("2 take(s)", rows[0][1])

    def test_a_twin_s_channel_lost_to_the_split_fails(self):
        """Two nodes sharing a name were one channel to the gate, so a shot
        that lost one twin's curve still held the other's and passed."""
        whole = {("a", "d|Y"): (1, 100), ("b", "d|Y"): (1, 100)}
        takes = {
            "Take 001": whole,
            "A01": {key: (1, 50) for key in whole},
            "A02": {("a", "d|Y"): (51, 100)},
        }
        rows = self._run(takes, names={"b": "a"})
        self.assertEqual([status for status, _ in rows], ["FAIL"], rows)
        self.assertIn("A02: 1 channel(s)", rows[0][1])
        self.assertIn("a#1001.Lcl Translation.d|Y", rows[0][1])


class TestExportVerifier(_FixtureCase):
    def _good_pair(self):
        glb = build_glb(self.path("asset.glb"))
        fbx = build_fbx(self.path("asset.fbx"), ("Shot_1",))
        build_sidecar(
            self.path(".asset.scene_data.json"),
            [{"name": "Shot_1", "start": 0, "end": 15}],
        )
        return glb, fbx

    def test_good_pair_passes(self):
        glb, fbx = self._good_pair()
        report = ExportVerifier(glb=glb, fbx=fbx).run()
        self.assertTrue(report.ok, report.summary())
        statuses = {row.check: row.status for row in report.rows}
        self.assertEqual(statuses["clips_vs_takes"], PASS)
        self.assertEqual(statuses["fbx_takes"], PASS)

    # ---- a declared Full Sequence Only export ------------------------------

    def _full_sequence(self, clip_mode):
        """What a Full Sequence Only export ships: ONE whole-timeline stack and
        clip, while the takes still list every shot -- they describe the scene."""
        glb = build_glb(self.path("asset.glb"), clip_name="Take 001")
        fbx = build_fbx(self.path("asset.fbx"), ("Take 001",))
        build_sidecar(
            self.path(".asset.scene_data.json"),
            [{"name": "Shot_1", "start": 0, "end": 15}],
            clip_mode=clip_mode,
        )
        return glb, fbx

    def test_a_declared_full_sequence_export_does_not_fail_its_takes(self):
        """Measured on a real Full Sequence Only export: ``fbx_takes`` failed
        "declared but absent" on every shot of a deliverable that was exactly
        what was asked for. The takes describe the scene in every mode, so the
        gates read the DECLARED mode instead of calling the shots missing."""
        glb, fbx = self._full_sequence("full")
        statuses = {}
        for row in ExportVerifier(glb=glb, fbx=fbx).run().rows:
            statuses.setdefault(row.check, []).append(row.status)
        self.assertEqual(statuses["fbx_takes"], [SKIP])
        self.assertNotIn(FAIL, statuses["clips_vs_takes"])

    def test_an_undeclared_single_stack_still_fails_its_takes(self):
        """Inferred from "only one stack", the gate would pass a split that
        silently failed -- the defect it was written for -- so a file that does
        not declare the mode, or declares a shot-bearing one, still fails."""
        for mode in (None, "both"):
            glb, fbx = self._full_sequence(mode)
            rows = ExportVerifier(glb=glb, fbx=fbx).run().rows
            self.assertEqual(
                [r.status for r in rows if r.check == "fbx_takes"], [FAIL], mode
            )
            self.assertIn(
                FAIL, [r.status for r in rows if r.check == "clips_vs_takes"], mode
            )

    # ---- clip origin ------------------------------------------------------
    #
    # The defect these cover shipped a production assembly: the exporter
    # published the BAKE RANGE as the exported stack's span, but the FBX
    # plug-in does not trim authored curves to that range (measured on Maya
    # 2025 / FBX 2020.3.6: a curve keyed 0-100 exports as 0-100 under a 20-80
    # range). The stack carried frames 80-4281 while 161-4275 was published, so
    # all 18 shots were cut 81 frames early and played the end of the previous
    # shot -- with every other gate green, because the clip LENGTHS still
    # matched their takes and no shot was missing.

    def test_clip_origin_fails_when_the_stack_outruns_its_published_span(self):
        glb = build_glb(self.path("asset.glb"), clip_end=0.5)  # 15 frames carried
        build_sidecar(self.path(".asset.scene_data.json"), [], span=[100, 108])
        report = ExportVerifier(glb=glb).run(["check_clip_origin"])
        self.assertFalse(report.ok, report.summary())
        row = report.rows[0]
        self.assertEqual(row.status, FAIL)
        # Names both numbers and the drift between them, so the reader can see
        # which end is wrong without opening the file.
        self.assertIn("carries 15f", row.detail)
        self.assertIn("8f span (100-108)", row.detail)
        self.assertIn("+7f", row.detail)

    def test_clip_origin_passes_when_the_span_describes_the_stack(self):
        glb = build_glb(self.path("asset.glb"), clip_end=0.5)  # 15 frames carried
        build_sidecar(self.path(".asset.scene_data.json"), [], span=[100, 115])
        report = ExportVerifier(glb=glb).run(["check_clip_origin"])
        self.assertTrue(report.ok, report.summary())
        self.assertEqual(report.rows[0].status, PASS)

    def test_fps_comes_from_the_sidecar_not_a_hardcoded_30(self):
        """Every frame a gate quotes is scaled by the rate it assumes.

        ``TaskManager.verify_deliverables`` passes only paths, so a 24 or 60
        fps scene was measured at 30 and every frame count came out wrong by
        the ratio.
        """
        glb = build_glb(self.path("asset.glb"), clip_end=1.0)  # 1 second
        build_sidecar(self.path(".asset.scene_data.json"), [], span=[0, 24], fps=24.0)
        verifier = ExportVerifier(glb=glb)
        self.assertEqual(verifier.fps, 24.0)
        # 1s at 24 fps is 24 frames, which is the span the sidecar publishes.
        report = verifier.run(["check_clip_origin"])
        self.assertTrue(report.ok, report.summary())

    def test_explicit_fps_overrides_the_sidecar(self):
        glb = build_glb(self.path("asset.glb"))
        build_sidecar(self.path(".asset.scene_data.json"), [], span=[0, 24], fps=24.0)
        self.assertEqual(ExportVerifier(glb=glb, fps=60.0).fps, 60.0)

    def test_fps_falls_back_to_30_without_a_sidecar(self):
        glb = build_glb(self.path("asset.glb"))
        self.assertEqual(ExportVerifier(glb=glb, sidecar=None).fps, 30.0)

    def test_a_versioned_deliverable_finds_the_manifest_its_series_shares(self):
        """The exporters key a versioned export's sidecar to the BASE stem, so a
        series diffs against one baseline. Discovery looked for
        '.asset_v003.scene_data.json' alone, so the gates that catch a dropped
        take SKIPped for every versioned deliverable checked from disk."""
        glb = build_glb(self.path("asset_v003.glb"))
        shared = build_sidecar(self.path(".asset.scene_data.json"), [])
        self.assertEqual(ExportVerifier(glb=glb).sidecar_path, shared)

    def test_a_deliverable_s_own_sidecar_wins_over_its_series(self):
        glb = build_glb(self.path("asset_v003.glb"))
        build_sidecar(self.path(".asset.scene_data.json"), [])
        own = build_sidecar(self.path(".asset_v003.scene_data.json"), [])
        self.assertEqual(ExportVerifier(glb=glb).sidecar_path, own)

    def test_the_series_fallback_strips_only_a_trailing_version(self):
        glb = build_glb(self.path("arch_v2_proxy.glb"))
        build_sidecar(self.path(".arch.scene_data.json"), [])
        self.assertIsNone(ExportVerifier(glb=glb).sidecar_path)

    def test_clip_origin_skips_without_a_published_span(self):
        """An older sidecar publishes no span; there is nothing to check against."""
        glb = build_glb(self.path("asset.glb"))
        build_sidecar(self.path(".asset.scene_data.json"), [])
        report = ExportVerifier(glb=glb).run(["check_clip_origin"])
        self.assertEqual(report.rows[0].status, SKIP)

    def test_clip_origin_skips_when_no_clip_is_the_whole_timeline(self):
        """Every clip declared: none of them is the stack, so none can show drift."""
        glb = build_glb(self.path("asset.glb"), clip_name="Shot_1", clip_end=0.5)
        build_sidecar(
            self.path(".asset.scene_data.json"),
            [{"name": "Shot_1", "start": 0, "end": 15}],
            span=[100, 108],
        )
        report = ExportVerifier(glb=glb).run(["check_clip_origin"])
        self.assertEqual(report.rows[0].status, SKIP)

    # The conversion measures the stack's span from the FBX it reads and
    # stamps it into the GLB (``MeshConvert._stamp_clip_spans``), so the clips
    # are cut against the FILE's span, whatever the producer predicted.

    @staticmethod
    def _cut_against(span):
        """The GLB carrier's ``visibility_tracks``, publishing *span* as ``*``."""
        return {
            "visibility_tracks": {
                "version": 1,
                "fps": 30.0,
                "tracks": [],
                "clip_span": {"*": list(span)},
            }
        }

    def test_clip_origin_judges_the_span_the_file_was_cut_against(self):
        """A producer whose prediction was off FAILed a deliverable cut right
        -- "every clip is cut from the wrong frame" -- when the conversion had
        placed every clip by the FBX's own keys (2026-10-04 review). The stack
        is judged against the span the GLB carries; the producer's is a WARN
        naming both."""
        glb = build_glb(
            self.path("asset.glb"),
            clip_end=0.5,  # 15 frames carried
            data_export=self._cut_against([100, 115]),
        )
        build_sidecar(self.path(".asset.scene_data.json"), [], span=[100, 108])
        report = ExportVerifier(glb=glb).run(["check_clip_origin"])
        self.assertTrue(report.ok, report.summary())
        self.assertEqual([row.status for row in report.rows], [WARN])
        self.assertIn("published 100-108", report.rows[0].detail)
        self.assertIn("cut against 100-115", report.rows[0].detail)

        # Without a sidecar the file still says what it was cut against.
        report = ExportVerifier(glb=glb, sidecar=None).run(["check_clip_origin"])
        self.assertEqual([row.status for row in report.rows], [PASS])

    def test_clip_origin_fails_a_file_cut_against_a_span_its_stack_lacks(self):
        glb = build_glb(
            self.path("asset.glb"),
            clip_end=0.5,
            data_export=self._cut_against([100, 108]),
        )
        build_sidecar(self.path(".asset.scene_data.json"), [], span=[100, 115])
        report = ExportVerifier(glb=glb).run(["check_clip_origin"])
        self.assertEqual([row.status for row in report.rows], [FAIL, WARN])
        self.assertIn("8f span (100-108)", report.rows[0].detail)

    def test_a_truncated_fbx_fails_its_container_gate_with_the_reason(self):
        """Cut inside a key array, the reader's ``zlib.error`` escaped as
        "check raised"; cut anywhere else, the gate PASSed the remains."""
        for inside in (b"KeyTime", b"Connections"):
            with self.subTest(inside=inside):
                fbx = build_truncated_fbx(self.path("cut.fbx"), inside)
                report = ExportVerifier(fbx=fbx, sidecar=None).run(
                    ["check_fbx_container"]
                )
                self.assertEqual([row.status for row in report.rows], [FAIL])
                self.assertIn("Truncated or corrupt FBX", report.rows[0].detail)

    def test_nan_fails_animation_gate(self):
        glb = build_glb(self.path("asset.glb"), nan_output=True)
        report = ExportVerifier(glb=glb, sidecar=None).run()
        self.assertFalse(report.ok)
        self.assertIn(
            FAIL,
            [r.status for r in report.rows if r.check == "glb_animation"],
        )

    def test_missing_ibm_fails_skins_gate(self):
        glb = build_glb(self.path("asset.glb"), drop_ibm=True)
        report = ExportVerifier(glb=glb, sidecar=None).run(["check_glb_skins"])
        self.assertFalse(report.ok)

    def test_stub_skins_warn_not_fail(self):
        """An IBM-less skin nothing references is converter noise: WARN only.

        Pinned from the tool's first production run — FBX2glTF wrote 93 such
        stubs and the raw-count gate failed a deliverable whose 7 real skins
        were all perfectly bound.
        """
        glb = build_glb(self.path("asset.glb"), stub_skin=True)
        report = ExportVerifier(glb=glb, sidecar=None).run(["check_glb_skins"])
        self.assertTrue(report.ok, report.summary())
        self.assertIn("WARN", [row.status for row in report.rows])

    def test_a_skeleton_that_is_not_its_joints_root_fails_skins_gate(self):
        """glTF requires ``skin.skeleton`` to be a common root of the joints.

        three.js never reads the field, so no viewer shows a bad one: a
        production assembly shipped 7 skins naming a joint their other joints
        do not hang under, caught only by the Khronos validator
        (SKIN_SKELETON_INVALID x7, 2026-09-14).
        """
        glb = build_glb(self.path("asset.glb"), bad_skeleton=True)
        report = ExportVerifier(glb=glb, sidecar=None).run(["check_glb_skins"])
        self.assertFalse(report.ok, report.summary())
        failed = [row.detail for row in report.rows if row.status == FAIL]
        self.assertTrue(any("skeleton" in detail for detail in failed), failed)

    def test_undeclared_basisu_fails_images_gate(self):
        glb = build_glb(self.path("asset.glb"), undeclared_basisu=True)
        report = ExportVerifier(glb=glb, sidecar=None).run(["check_glb_images"])
        self.assertFalse(report.ok)

    def test_image_bytes_are_reported_and_warned_past_a_limit(self):
        """The GLB's own image bytes are what a web deliverable ships.

        A scene exporter's per-map size check failed a production export on a
        57 MB source PNG that the GLB carried as a 3.12 MB KTX2 -- only the
        written file can say what an image costs. The gate reports the bytes
        and never fails: the file already shipped.
        Added: 2026-09-13
        """
        glb = build_glb(self.path("asset.glb"))  # a 16-byte PNG, a 20-byte KTX2
        report = ExportVerifier(glb=glb, sidecar=None).run(["check_glb_image_bytes"])
        self.assertEqual([row.status for row in report.rows], [PASS], report.summary())
        self.assertIn("2 image(s)", report.rows[0].detail)
        self.assertIn("image 1", report.rows[0].detail, "the largest is named")

        report = ExportVerifier(glb=glb, sidecar=None, max_image_bytes=16).run(
            ["check_glb_image_bytes"]
        )
        self.assertTrue(report.ok, report.summary())
        warned = [row for row in report.rows if row.status == WARN]
        self.assertEqual(len(warned), 1, report.summary())
        self.assertIn("image 1", warned[0].detail)
        self.assertNotIn("image 0", warned[0].detail, "16 bytes is not past 16")

    def _drop_png_fallback(self, glb: str, *, require: bool) -> str:
        """Strip the texture's PNG twin, as drop_glb_texture_fallbacks does.

        *require* decides whether the file then DECLARES that a reader without
        the extension cannot open it -- the one fact that tells a deliberate
        delivery mode apart from a broken one.
        """
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        with MeshConvert.open_glb(glb) as edit:
            del edit.gltf["textures"][0]["source"]
            if require:
                edit.gltf["extensionsRequired"] = ["KHR_texture_basisu"]
            edit.dirty = True
        return glb

    def test_a_missing_fallback_warns_while_basisu_is_optional(self):
        """Optional means a stock reader is promised a picture, and there is
        none: that is a real defect, and the gate must keep saying so."""
        glb = self._drop_png_fallback(build_glb(self.path("asset.glb")), require=False)
        report = ExportVerifier(glb=glb, sidecar=None).run(["check_glb_images"])
        self.assertIn(WARN, [row.status for row in report.rows], report.summary())

    def test_a_missing_fallback_passes_once_basisu_is_REQUIRED(self):
        """The shape ``drop_glb_texture_fallbacks`` produces on purpose.

        Shipping one encoding instead of two is the whole saving; once the
        file declares the extension REQUIRED it has stopped promising anything
        it cannot deliver. Warning on it every run is how a reader learns to
        scroll past the gate that would name a real one.
        """
        glb = self._drop_png_fallback(build_glb(self.path("asset.glb")), require=True)
        report = ExportVerifier(glb=glb, sidecar=None).run(["check_glb_images"])
        self.assertTrue(report.ok, report.summary())
        self.assertNotIn(WARN, [row.status for row in report.rows], report.summary())
        self.assertIn(PASS, [row.status for row in report.rows])

    def test_a_declared_requirement_is_stated_not_warned(self):
        """The envelope gate restated ``verify_glb``'s extension prerequisite
        as a WARN -- on every WebP deliverable, and on every KTX2 one once the
        web policy stopped writing fallback twins -- while ``glb_images``
        PASSes the same fact as the declared contract. The prerequisite is the
        extension gates' to state; the envelope's own notes still warn.
        """
        from unittest.mock import patch

        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        glb = self._drop_png_fallback(build_glb(self.path("asset.glb")), require=True)
        verified = MeshConvert.verify_glb(glb)
        self.assertTrue(
            any("KHR_texture_basisu" in note for note in verified["notes"]),
            f"precondition: verify_glb states the prerequisite: {verified['notes']}",
        )
        verified["notes"].append("a note the envelope owns")
        with MeshConvert.open_glb(glb) as edit:  # the gate skips a GLB with none
            edit.gltf.setdefault("extras", {})["scene_sidecar"] = {"version": 1}
            edit.dirty = True

        with patch.object(MeshConvert, "verify_glb", return_value=verified):
            report = ExportVerifier(glb=glb, sidecar=None).run(
                ["check_glb_envelope", "check_glb_extensions"]
            )

        warned = [row.detail for row in report.rows if row.status == WARN]
        self.assertIn("a note the envelope owns", warned)
        self.assertFalse(
            any("KHR_texture_basisu" in detail for detail in warned), warned
        )
        extensions = [row for row in report.rows if row.check == "glb_extensions"]
        self.assertEqual([row.status for row in extensions], [PASS])
        self.assertIn("required=['KHR_texture_basisu']", extensions[0].detail)

    def test_take_length_mismatch_fails(self):
        glb = build_glb(self.path("asset.glb"), clip_end=1.0)  # 30 frames
        build_sidecar(
            self.path(".asset.scene_data.json"),
            [{"name": "Shot_1", "start": 0, "end": 15}],
        )
        report = ExportVerifier(glb=glb).run(["check_clips_vs_takes"])
        self.assertFalse(report.ok)

    def test_a_rebased_full_timeline_clip_is_measured_from_its_own_zero(self):
        """The whole-timeline stack is REBASED, and says so.

        The converter puts the stack's first key at ``t=0``, so a clip whose
        content starts at authoring frame 33 ends 33 frames short of the
        takes' declared end while being perfectly correct. Reading its raw
        end frame called that a failure -- the same blind spot that, on the
        producer side, slid every shot by 33 frames on the PROPS assembly.
        The clip publishes its origin in ``extras.zero_frame``; the gate has
        to use it.
        """
        glb = build_glb(
            self.path("asset.glb"),
            clip_name="FULL_SEQUENCE",
            clip_end=(1989 - 33) / 30.0,
            zero_frame=33,
        )
        build_sidecar(
            self.path(".asset.scene_data.json"),
            [{"name": "Shot_1", "start": 33, "end": 1989}],
        )
        report = ExportVerifier(glb=glb).run(["check_clips_vs_takes"])
        # Isolate the branch under test: this minimal fixture carries only the
        # whole-timeline clip, so the declared take is legitimately "absent".
        # Assert on the STATUS, not on the wording -- a message-shaped assert
        # passes for free the moment the message is reworded.
        self.assertEqual(
            [r.detail for r in report.rows if "FULL_SEQUENCE" in r.detail],
            [],
            report.summary(),
        )

    def test_a_rebased_clip_that_is_actually_short_still_fails(self):
        """Honouring the origin must not blunt the gate."""
        glb = build_glb(
            self.path("asset.glb"),
            clip_name="FULL_SEQUENCE",
            clip_end=(1500 - 33) / 30.0,
            zero_frame=33,
        )
        build_sidecar(
            self.path(".asset.scene_data.json"),
            [{"name": "Shot_1", "start": 33, "end": 1989}],
        )
        report = ExportVerifier(glb=glb).run(["check_clips_vs_takes"])
        self.assertFalse(report.ok, report.summary())
        self.assertTrue(
            [
                r
                for r in report.rows
                if r.status == FAIL and "FULL_SEQUENCE" in r.detail
            ],
            report.summary(),
        )

    def test_a_trailing_held_pose_is_a_note_not_a_failure(self):
        """The bake pads the whole-timeline clip; padding is not an overrun.

        The bake range is handed to the exporter, and it writes keys across
        all of it -- so the full-timeline clip routinely ends on a held pose
        some frames past the last take. Judging it on key occupancy called
        that a failure on a correct file (PROPS: 48 inert frames, 1109 of
        1185 channels flat, the other 76 moving by 3e-4), and a gate that is
        permanently red is a gate nobody reads. Only motion out there counts.
        """
        glb = build_glb(
            self.path("asset.glb"),
            clip_name="FULL_SEQUENCE",
            clip_end=1989 / 30.0,
            hold_frames=48,
        )
        build_sidecar(
            self.path(".asset.scene_data.json"),
            [{"name": "Shot_1", "start": 0, "end": 1989}],
        )
        report = ExportVerifier(glb=glb).run(["check_clips_vs_takes"])
        # As in the sibling rebase test, the lone clip means the declared take
        # is legitimately absent -- judge the padding row, not the whole run.
        held = [r for r in report.rows if "FULL_SEQUENCE" in r.detail]
        self.assertEqual(len(held), 1, report.summary())
        self.assertEqual(held[0].status, WARN, report.summary())
        self.assertIn("+48f", held[0].detail)
        self.assertIn("2037f", held[0].detail)

    def test_motion_past_the_last_take_still_fails(self):
        """Forgiving the hold must not forgive a real overrun."""
        glb = build_glb(
            self.path("asset.glb"),
            clip_name="FULL_SEQUENCE",
            clip_end=1989 / 30.0,
            hold_frames=48,
            hold_moves=True,
        )
        build_sidecar(
            self.path(".asset.scene_data.json"),
            [{"name": "Shot_1", "start": 0, "end": 1989}],
        )
        report = ExportVerifier(glb=glb).run(["check_clips_vs_takes"])
        self.assertFalse(report.ok, report.summary())
        self.assertTrue(
            [r for r in report.rows if "animates to" in r.detail], report.summary()
        )

    def test_a_clip_closing_on_a_held_pose_is_not_truncation(self):
        """Motion ending before the take does is ordinary animation.

        Shots routinely finish their action and hold for a beat. Judging the
        clip on where motion STOPS would call every one of those truncated --
        the real PROPS assembly stops moving 79 frames before its last take
        ends. Only the KEYS falling short means content is missing.
        """
        glb = build_glb(
            self.path("asset.glb"),
            clip_name="FULL_SEQUENCE",
            clip_end=1989 / 30.0,
            hold_frames=48,
        )
        build_sidecar(
            self.path(".asset.scene_data.json"),
            [{"name": "Shot_1", "start": 0, "end": 2037}],
        )
        report = ExportVerifier(glb=glb).run(["check_clips_vs_takes"])
        self.assertEqual(
            [r.detail for r in report.rows if "FULL_SEQUENCE" in r.detail],
            [],
            report.summary(),
        )

    def test_a_clip_whose_keys_end_short_of_its_take_fails(self):
        """Keys that stop early cannot play the declared range at all."""
        glb = build_glb(
            self.path("asset.glb"),
            clip_name="FULL_SEQUENCE",
            clip_end=1500 / 30.0,
        )
        build_sidecar(
            self.path(".asset.scene_data.json"),
            [{"name": "Shot_1", "start": 0, "end": 1989}],
        )
        report = ExportVerifier(glb=glb).run(["check_clips_vs_takes"])
        self.assertFalse(report.ok, report.summary())

    def test_corrupt_sidecar_degrades_to_skip(self):
        """A broken sidecar must never crash the run — its gates SKIP with
        the reason instead."""
        glb = build_glb(self.path("asset.glb"))
        bad = self.path(".asset.scene_data.json")
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        report = ExportVerifier(glb=glb).run(["check_clips_vs_takes"])
        self.assertTrue(report.ok, report.summary())
        self.assertEqual(report.rows[0].status, SKIP)
        self.assertIn("unreadable", report.rows[0].detail)

    def test_no_sidecar_skips_not_fails(self):
        glb = build_glb(self.path("asset.glb"))
        report = ExportVerifier(glb=glb, sidecar=None).run(["check_clips_vs_takes"])
        self.assertTrue(report.ok)
        self.assertEqual(report.rows[0].status, SKIP)

    def test_baseline_clip_rename_fails(self):
        old = build_glb(self.path("old.glb"))
        new = build_glb(self.path("new.glb"), clip_name="Renamed")
        report = ExportVerifier(glb=new, sidecar=None, baseline_glb=old).run(
            ["check_baseline_diff"]
        )
        self.assertFalse(report.ok)

    def test_declared_take_missing_from_fbx_fails(self):
        glb, _ = self._good_pair()
        fbx = build_fbx(self.path("other.fbx"), ("Different",))
        report = ExportVerifier(glb=glb, fbx=fbx).run(["check_fbx_takes"])
        self.assertFalse(report.ok)

    def test_cli_exit_codes_and_json(self):
        glb, fbx = self._good_pair()
        self.assertEqual(_main([glb, fbx]), 0)
        bad = build_glb(self.path("bad.glb"), nan_output=True)
        self.assertEqual(_main([bad, "--sidecar", "none"]), 1)

    def test_cli_list_checks(self):
        glb, _ = self._good_pair()
        self.assertEqual(_main([glb, "--list-checks"]), 0)


if __name__ == "__main__":
    unittest.main()
