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
from pythontk.file_utils.mesh_convert.glb_reader import GlbReader


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
    undeclared_basisu: bool = False,
    clip_end: float = 0.5,
    clip_name: str = "Shot_1",
    zero_frame: float = None,
    hold_frames: float = 0.0,
    hold_moves: bool = False,
) -> str:
    """Write a tiny, valid GLB: two nodes, one clip, one skin, two images."""
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


def _fbx_prop(value) -> bytes:
    if isinstance(value, bytes):
        return b"S" + struct.pack("<I", len(value)) + value
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


def build_sidecar(path: str, takes) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "format": 3,
                "hierarchy": {"paths": ["root", "root|arm"]},
                "data_export": {"fbx_takes": list(takes)},
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

    def test_not_an_fbx(self):
        junk = self.path("junk.fbx")
        with open(junk, "wb") as handle:
            handle.write(b"not an fbx")
        self.assertFalse(FbxFile.is_fbx(junk))
        with self.assertRaises(ValueError):
            FbxFile.load(junk)


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

    def test_undeclared_basisu_fails_images_gate(self):
        glb = build_glb(self.path("asset.glb"), undeclared_basisu=True)
        report = ExportVerifier(glb=glb, sidecar=None).run(["check_glb_images"])
        self.assertFalse(report.ok)

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
        producer side, slid every shot by 33 frames on the VDATS assembly.
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
        that a failure on a correct file (VDATS: 48 inert frames, 1109 of
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
        the real VDATS assembly stops moving 79 frames before its last take
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
