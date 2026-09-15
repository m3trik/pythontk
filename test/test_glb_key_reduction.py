# !/usr/bin/python
# coding=utf-8
"""GlbKeyReduction -- tolerance-bound key reduction on a GLB's samplers.

Every reduced sampler must reproduce every ORIGINAL sample within the bound
under the interpolation the viewer applies (lerp, shortest-arc slerp for a
rotation, held values for STEP), keep both ends, and leave alone what it
cannot rewrite faithfully. Added: 2026-09-13
"""

import json
import math
import os
import struct
import unittest

from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert
from pythontk.file_utils.mesh_convert.glb_key_reduction import GlbKeyReduction
from pythontk.file_utils.temp_artifacts import TempArtifacts

_WIDTH = {1: "SCALAR", 3: "VEC3", 4: "VEC4"}


def _animated_glb(path, samplers, shared_input=False):
    """A one-node GLB whose ``animations[0]`` carries *samplers*: each a
    ``(times, values, interpolation, target_path)`` tuple. With
    *shared_input* every sampler reads ONE time accessor, the converter's shape.
    """
    views, accessors, bin_chunk = [], [], b""
    gltf_samplers, channels = [], []

    def add(payload, count, kind, bounds=None):
        nonlocal bin_chunk
        views.append(
            {"buffer": 0, "byteOffset": len(bin_chunk), "byteLength": len(payload)}
        )
        accessor = {
            "bufferView": len(views) - 1,
            "componentType": 5126,
            "count": count,
            "type": kind,
        }
        if bounds:
            accessor["min"], accessor["max"] = [bounds[0]], [bounds[1]]
        accessors.append(accessor)
        bin_chunk += payload + b"\x00" * ((4 - len(payload) % 4) % 4)
        return len(accessors) - 1

    shared = None
    for times, values, interpolation, target_path in samplers:
        if shared_input and shared is not None:
            time_accessor = shared
        else:
            time_accessor = add(
                struct.pack(f"<{len(times)}f", *times),
                len(times),
                "SCALAR",
                (times[0], times[-1]),
            )
            shared = time_accessor
        flat = [c for v in values for c in v]
        width = len(values[0])
        value_accessor = add(
            struct.pack(f"<{len(flat)}f", *flat), len(values), _WIDTH[width]
        )
        gltf_samplers.append(
            {
                "input": time_accessor,
                "output": value_accessor,
                "interpolation": interpolation,
            }
        )
        channels.append(
            {
                "sampler": len(gltf_samplers) - 1,
                "target": {"node": 0, "path": target_path},
            }
        )
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_chunk)}],
        "bufferViews": views,
        "accessors": accessors,
        "nodes": [{"name": "node"}],
        "animations": [
            {"name": "clip", "samplers": gltf_samplers, "channels": channels}
        ],
    }
    payload = json.dumps(gltf).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    rest = struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(payload) + len(rest)))
        fh.write(struct.pack("<I4s", len(payload), b"JSON") + payload)
        fh.write(rest)
    return path


def _read_samplers(path):
    edit = MeshConvert._read_glb(path)
    animation = edit.gltf["animations"][0]
    return [
        GlbKeyReduction.read_sampler(edit.gltf, edit.bin_data, sampler)
        for sampler in animation["samplers"]
    ], edit


def _sine_translation(count=200, seconds=4.0):
    times = [seconds * i / (count - 1) for i in range(count)]
    values = [(math.sin(t * 1.7), 0.5 * math.cos(t * 0.9), t * 0.25) for t in times]
    return times, values


def _rotation_about_y(count=120, degrees=170.0, flip_from=None):
    """An ACCELERATING sweep: a constant-rate turn about one axis is exactly
    the slerp between its ends, which any tolerance would reduce to two keys."""
    times = [i / 24.0 for i in range(count)]
    values = []
    for i, _t in enumerate(times):
        half = math.radians(degrees * (i / (count - 1)) ** 2) / 2.0
        q = (0.0, math.sin(half), 0.0, math.cos(half))
        if flip_from is not None and i >= flip_from:
            q = tuple(-c for c in q)  # the same rotation, the other sign
        values.append(q)
    return times, values


class GlbKeyReductionTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = TempArtifacts("test_glb_key_reduction", policy="scoped")

    def tearDown(self):
        self.temp.cleanup()

    def _glb(self, samplers, **kw):
        return _animated_glb(self.temp.path(extension=".glb"), samplers, **kw)

    # ------------------------------------------------------------ the bound
    def test_a_linear_sampler_reproduces_every_sample_within_the_tolerance(self):
        times, values = _sine_translation()
        path = self._glb([(times, values, "LINEAR", "translation")])
        summary = GlbKeyReduction.reduce(path, 1e-3)
        (reduced,), edit = _read_samplers(path)
        self.assertLess(len(reduced[0]), len(times) // 2, "the sine needs fewer keys")
        self.assertEqual(summary["keys_after"], len(reduced[0]))
        self.assertEqual(summary["keys_before"], len(times))
        self.assertEqual(summary["samplers"], 1)
        self.assertLessEqual(
            GlbKeyReduction.deviation((times, values, "LINEAR"), reduced), 1e-3
        )
        self.assertEqual((reduced[0][0], reduced[0][-1]), (times[0], times[-1]))
        accessor = edit.gltf["accessors"][
            edit.gltf["animations"][0]["samplers"][0]["input"]
        ]
        self.assertEqual(accessor["count"], len(reduced[0]))
        self.assertEqual((accessor["min"], accessor["max"]), ([times[0]], [times[-1]]))
        self.assertGreater(summary["bytes"], 0, "the dense payload was reclaimed")

    def test_a_tighter_tolerance_keeps_more_keys(self):
        times, values = _sine_translation()
        counts = []
        for tolerance in (1e-2, 1e-4, 1e-6):
            path = self._glb([(times, values, "LINEAR", "translation")])
            counts.append(GlbKeyReduction.reduce(path, tolerance)["keys_after"])
        self.assertEqual(counts, sorted(counts))
        self.assertLess(counts[0], counts[-1])

    def test_a_straight_line_collapses_to_its_ends(self):
        times = [i / 10.0 for i in range(50)]
        values = [(t, 2 * t, -t) for t in times]
        path = self._glb([(times, values, "LINEAR", "translation")])
        self.assertEqual(GlbKeyReduction.reduce(path, 1e-6)["keys_after"], 2)

    def test_a_rotation_is_measured_by_slerp_and_reads_both_signs_alike(self):
        times, values = _rotation_about_y(flip_from=60)
        path = self._glb([(times, values, "LINEAR", "rotation")])
        summary = GlbKeyReduction.reduce(path, 1e-3)
        (reduced,), _edit = _read_samplers(path)
        self.assertLess(summary["keys_after"], summary["keys_before"])
        self.assertLessEqual(
            GlbKeyReduction.deviation((times, values, "LINEAR"), reduced, True), 1e-3
        )
        # Measured as plain component lerp instead, the same keys deviate
        # MORE: the chord dips inside the unit sphere between two rotations.
        self.assertGreater(
            GlbKeyReduction.deviation((times, values, "LINEAR"), reduced, False),
            GlbKeyReduction.deviation((times, values, "LINEAR"), reduced, True),
        )

    def test_a_separate_rotation_tolerance_applies_to_rotations_only(self):
        t_times, t_values = _sine_translation()
        r_times, r_values = _rotation_about_y()
        loose = self._glb(
            [
                (t_times, t_values, "LINEAR", "translation"),
                (r_times, r_values, "LINEAR", "rotation"),
            ]
        )
        tight = self._glb(
            [
                (t_times, t_values, "LINEAR", "translation"),
                (r_times, r_values, "LINEAR", "rotation"),
            ]
        )
        GlbKeyReduction.reduce(loose, 1e-3)
        GlbKeyReduction.reduce(tight, 1e-3, rotation_tolerance=1e-6)
        (loose_t, loose_r), _ = _read_samplers(loose)
        (tight_t, tight_r), _ = _read_samplers(tight)
        self.assertEqual(len(loose_t[0]), len(tight_t[0]))
        self.assertGreater(len(tight_r[0]), len(loose_r[0]))

    # ------------------------------------------------------------ STEP / refusals
    def test_step_dedupe_is_lossless_and_keeps_the_change_points(self):
        times = [float(i) for i in range(8)]
        values = [(0.0,), (0.0,), (0.0,), (1.0,), (1.0,), (0.0,), (0.0,), (0.0,)]
        path = self._glb([(times, values, "STEP", "pointer")])
        summary = GlbKeyReduction.reduce(path, 1e-4)
        (reduced,), _ = _read_samplers(path)
        self.assertEqual(reduced[0], [0.0, 3.0, 5.0, 7.0])
        self.assertEqual(reduced[2], "STEP")
        self.assertEqual(summary["keys_after"], 4)
        for at in (0.5, 2.99, 3.0, 4.5, 5.0, 6.5):
            self.assertEqual(
                GlbKeyReduction.evaluate(reduced[0], reduced[1], at, "STEP"),
                GlbKeyReduction.evaluate(times, values, at, "STEP"),
            )
        # at the change point itself the NEW value holds (glTF STEP)
        self.assertEqual(
            GlbKeyReduction.evaluate(reduced[0], reduced[1], 3.0, "STEP"), (1.0,)
        )
        self.assertEqual(
            GlbKeyReduction.evaluate(reduced[0], reduced[1], 5.0, "STEP"), (0.0,)
        )

    def test_a_step_sampler_does_not_deviate_from_itself(self):
        """``deviation`` samples the candidate at every reference time -- for a
        STEP gate that includes its own change points, where the value is the
        new key's. It read the hold one key late there and reported a 1.0
        deviation between identical samplers (production, 2026-09-13)."""
        times = [0.0, 1.0, 2.0]
        values = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]
        sampler = (times, values, "STEP")
        self.assertEqual(GlbKeyReduction.deviation(sampler, sampler), 0.0)
        self.assertEqual(
            GlbKeyReduction.evaluate(times, values, 1.0, "STEP"), (1.0, 1.0, 1.0)
        )

    def test_cubicspline_and_a_shared_output_are_left_alone(self):
        times, values = _sine_translation(count=30)
        cubic = [(0.0,) * 3 + v + (0.0,) * 3 for v in values]  # in, value, out
        cubic_values = [c[i : i + 3] for c in cubic for i in (0, 3, 6)]
        path = self._glb([(times, cubic_values, "CUBICSPLINE", "translation")])
        before = os.path.getsize(path)
        summary = GlbKeyReduction.reduce(path, 1e-2)
        self.assertEqual(summary["samplers"], 0)
        self.assertEqual(os.path.getsize(path), before, "nothing rewritten")
        shared = self._glb([(times, values, "LINEAR", "translation")])
        edit = MeshConvert._read_glb(shared)
        animation = edit.gltf["animations"][0]
        animation["samplers"].append(dict(animation["samplers"][0]))
        animation["channels"].append(
            {"sampler": 1, "target": {"node": 0, "path": "scale"}}
        )
        edit.dirty = True
        with MeshConvert.open_glb(shared) as session:
            session.gltf.update(edit.gltf)
            session.dirty = True
        summary = GlbKeyReduction.reduce(shared, 1e-2)
        self.assertEqual(
            summary["samplers"], 0, "an output two samplers read is refused"
        )

    def test_the_converters_shared_time_input_is_split_per_sampler(self):
        times, values = _sine_translation()
        ramp = [(t, t, t) for t in times]
        path = self._glb(
            [
                (times, values, "LINEAR", "translation"),
                (times, ramp, "LINEAR", "scale"),
            ],
            shared_input=True,
        )
        summary = GlbKeyReduction.reduce(path, 1e-3)
        (sine, line), edit = _read_samplers(path)
        self.assertEqual(summary["samplers"], 2)
        self.assertEqual(len(line[0]), 2, "the ramp needs its ends only")
        self.assertGreater(len(sine[0]), 2)
        samplers = edit.gltf["animations"][0]["samplers"]
        self.assertNotEqual(samplers[0]["input"], samplers[1]["input"])
        self.assertLessEqual(
            GlbKeyReduction.deviation((times, values, "LINEAR"), sine), 1e-3
        )

    def test_an_external_buffer_leaves_the_file_and_says_why(self):
        """A GLB whose buffer 0 declares a ``uri`` has no BIN to append to:
        nothing is rewritten, the summary says nothing was reduced, and the
        warning names THIS pass. Added: 2026-09-13"""
        times, values = _sine_translation(count=40)
        path = self._glb([(times, values, "LINEAR", "translation")])
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["buffers"][0]["uri"] = "scene.bin"
            edit.dirty = True
        before = os.path.getsize(path)
        with self.assertLogs(
            "pythontk.file_utils.mesh_convert.glb_key_reduction", level="WARNING"
        ) as captured:
            summary = GlbKeyReduction.reduce(path, 1e-3)
        self.assertIn("reduced samplers", captured.output[0])
        self.assertEqual(summary["samplers"], 0)
        self.assertEqual(summary["keys_after"], summary["keys_before"])
        self.assertEqual(os.path.getsize(path), before)

    def test_the_tolerance_must_be_positive(self):
        path = self._glb([(*_sine_translation(count=10), "LINEAR", "translation")])
        with self.assertRaises(ValueError):
            GlbKeyReduction.reduce(path, 0)
        with self.assertRaises(ValueError):
            GlbKeyReduction.reduce(path, 1e-3, rotation_tolerance=-1)

    def test_meshconvert_entry_point_delegates(self):
        times, values = _sine_translation(count=60)
        path = self._glb([(times, values, "LINEAR", "translation")])
        summary = MeshConvert.reduce_glb_animations(path, 1e-3)
        self.assertEqual(summary["samplers"], 1)
        self.assertLess(summary["keys_after"], summary["keys_before"])


if __name__ == "__main__":
    unittest.main()
