# !/usr/bin/python
# coding=utf-8
"""Tests for the pure Shot Manifest core (``pythontk.core_utils.engines.shots.manifest``).

DCC-free.  Covers the parser + column map + behavior detection (``manifest_model``),
the column-mapping templates (``mapping``), the behavior template loading + keying
math (``behaviors``), and the range resolver (``range_resolver``).  The DCC-hooked
engine (``manifest_engine``) is exercised separately.
"""
import os
import sys
import tempfile
import unittest

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pythontk.core_utils.engines.shots.manifest.manifest_model import (
    parse_csv,
    ColumnMap,
    BuilderStep,
    BuilderObject,
    detect_behaviors,
    StepStatus,
    ObjectStatus,
)
from pythontk.core_utils.engines.shots.manifest import mapping as mapping_mod
from pythontk.core_utils.engines.shots.manifest import behaviors as beh
from pythontk.core_utils.engines.shots.manifest import range_resolver as rr
from pythontk.core_utils.engines.shots.manifest.manifest_engine import (
    ShotManifest,
    resolve_duration,
)
from pythontk.core_utils.engines.shots.shot_model import ShotStore


def _write_csv(text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.write(fd, text.encode("utf-8"))
    os.close(fd)
    return path


_SAMPLE_CSV = """SECTION A: AILERON RIGGING
Step,Step Contents,Asset Names,Voice Support,Priority
A01.),Aileron fades in then fades out,aileron_geo,Narrator intro,high
,wing detail continuation,wing_geo,,
A02.),Static fuselage step,fuselage,,low
SETUP,should be excluded,setup_obj,,
"""


class TestManifestParse(unittest.TestCase):
    def setUp(self):
        self.path = _write_csv(_SAMPLE_CSV)

    def tearDown(self):
        os.remove(self.path)

    def test_step_count_excludes_setup(self):
        steps = parse_csv(self.path)
        self.assertEqual([s.step_id for s in steps], ["A01", "A02"])

    def test_section_and_title(self):
        steps = parse_csv(self.path)
        self.assertEqual(steps[0].section, "A")
        self.assertEqual(steps[0].section_title, "AILERON RIGGING")

    def test_continuation_row_merges_description_and_object(self):
        steps = parse_csv(self.path)
        a01 = steps[0]
        self.assertIn("wing detail continuation", a01.description)
        self.assertEqual([o.name for o in a01.objects], ["aileron_geo", "wing_geo"])

    def test_behaviors_detected_from_description(self):
        steps = parse_csv(self.path)
        a01 = steps[0]
        self.assertEqual(a01.objects[0].behaviors, ["fade_in", "fade_out"])

    def test_audio_column_captured(self):
        steps = parse_csv(self.path)
        self.assertEqual(steps[0].audio, "Narrator intro")

    def test_metadata_pass_first_row_wins(self):
        cm = ColumnMap(metadata_pass={"priority": ("Priority",)})
        steps = parse_csv(self.path, columns=cm)
        self.assertEqual(steps[0]._pass_through.get("priority"), "high")
        self.assertEqual(steps[1]._pass_through.get("priority"), "low")

    def test_missing_header_yields_zero_steps(self):
        path = _write_csv("just,some,data\nwith,no,header\n")
        try:
            self.assertEqual(parse_csv(path), [])
        finally:
            os.remove(path)


class TestColumnMap(unittest.TestCase):
    def test_defaults(self):
        cm = ColumnMap()
        self.assertEqual(cm.step_id, ("Step",))
        self.assertIn("Asset Names", cm.assets)

    def test_dict_round_trip_preserves_tuples(self):
        cm = ColumnMap(metadata_pass={"priority": ("Priority", "Prio")})
        back = ColumnMap.from_dict(cm.to_dict())
        self.assertEqual(back.step_id, cm.step_id)
        self.assertEqual(back.metadata_pass["priority"], ("Priority", "Prio"))

    def test_from_dict_ignores_unknown_keys(self):
        back = ColumnMap.from_dict({"step_id": ["S"], "bogus": 1})
        self.assertEqual(back.step_id, ("S",))


class TestDetectBehaviors(unittest.TestCase):
    def test_fade_in(self):
        self.assertEqual(detect_behaviors("the panel fades in slowly"), ["fade_in"])

    def test_fade_out(self):
        self.assertEqual(detect_behaviors("then it fades out"), ["fade_out"])

    def test_both_independent(self):
        self.assertEqual(
            detect_behaviors("fades in then fades out"), ["fade_in", "fade_out"]
        )

    def test_none(self):
        self.assertEqual(detect_behaviors("a static object"), [])


class TestMapping(unittest.TestCase):
    def test_discover_builtins(self):
        names = mapping_mod.discover()
        self.assertIn("default", names)
        self.assertIn("speedrun", names)

    def test_load_default(self):
        spec = mapping_mod.load_mapping("default")
        self.assertIsNotNone(spec)

    def test_resolve_csv_to_steps(self):
        path = _write_csv(_SAMPLE_CSV)
        try:
            steps = mapping_mod.resolve(path, name="default")
            self.assertTrue(all(isinstance(s, BuilderStep) for s in steps))
            self.assertEqual([s.step_id for s in steps], ["A01", "A02"])
        finally:
            os.remove(path)


class TestBehaviors(unittest.TestCase):
    def test_list_builtins(self):
        self.assertEqual(
            sorted(beh.list_behaviors()), ["fade_in", "fade_out", "set_clip"]
        )

    def test_load_behavior(self):
        fi = beh.load_behavior("fade_in")
        self.assertIn("attributes", fi)
        self.assertIn("visibility", fi["attributes"])

    def test_resolve_keys_start_anchored_block(self):
        block = beh.load_behavior("fade_in")["attributes"]["visibility"]["in"]
        keys = beh.resolve_keys(block, 10, 60)
        # anchor=start, offset=0, duration=15, values=[0,1] -> keys at 10 and 25
        self.assertEqual([k["time"] for k in keys], [10.0, 25.0])
        self.assertEqual([k["value"] for k in keys], [0.0, 1.0])

    def test_resolve_keys_end_anchored_block(self):
        out = beh.load_behavior("fade_out")["attributes"]["visibility"]["out"]
        keys = beh.resolve_keys(out, 10, 60)
        # end-anchored: keys land near the range end (60), ascending times
        self.assertTrue(keys)
        self.assertLessEqual(keys[-1]["time"], 60.0 + 1e-6)
        self.assertEqual([k["time"] for k in keys], sorted(k["time"] for k in keys))

    def test_compute_duration_dict_and_object_forms_agree(self):
        dict_form = beh.compute_duration([{"name": "g", "behavior": "fade_in"}])
        obj_form = beh.compute_duration([BuilderObject(name="g", behaviors=["fade_in"])])
        self.assertEqual(dict_form, obj_form)

    def test_compute_duration_fallback_when_empty(self):
        self.assertEqual(beh.compute_duration([], fallback=200), 200)

    def test_compute_duration_audio_callback(self):
        obj = BuilderObject(name="vo", behaviors=["set_clip"], kind="audio", source_path="x.wav")
        got = beh.compute_duration(
            [obj], fallback=10, audio_duration_fn=lambda src: 123.0
        )
        self.assertGreaterEqual(got, 123.0)

    def test_compute_duration_resolve_source_fallback(self):
        # An entry with NO source_path resolves via resolve_source_fn and is
        # then measured by audio_duration_fn — the seam a DCC layer binds to
        # its own track registry (populated independently of the CSV).
        obj = BuilderObject(name="vo", behaviors=["set_clip"], kind="audio", source_path="")
        seen = {}

        def resolver(name, kind):
            seen["args"] = (name, kind)
            return "resolved.wav"

        got = beh.compute_duration(
            [obj],
            fallback=10,
            audio_duration_fn=lambda src: 77.0 if src == "resolved.wav" else None,
            resolve_source_fn=resolver,
        )
        self.assertEqual(seen["args"], ("vo", "audio"))
        self.assertGreaterEqual(got, 77.0)

    def test_compute_duration_raising_resolver_falls_back(self):
        # A raising resolver is swallowed; the unresolvable entry contributes
        # nothing so the manifest falls back instead of crashing the build.
        obj = BuilderObject(name="vo", behaviors=["set_clip"], kind="audio", source_path="")

        def boom(name, kind):
            raise RuntimeError("no registry")

        got = beh.compute_duration(
            [obj],
            fallback=10,
            audio_duration_fn=lambda src: 99.0,
            resolve_source_fn=boom,
        )
        self.assertEqual(got, 10)


class TestRangeResolver(unittest.TestCase):
    def _steps(self, n):
        return [
            BuilderStep(f"A0{i}", "A", "t", "", [BuilderObject(f"g{i}")])
            for i in range(1, n + 1)
        ]

    def test_prune_to_top_boundaries(self):
        # starts 0,1,2,100,101 -> largest gap before 100; keep 2 -> [0,100]
        self.assertEqual(
            rr.prune_to_top_boundaries([0, 1, 2, 100, 101], 2), [0, 100]
        )

    def test_prune_noop_when_within_budget(self):
        self.assertEqual(rr.prune_to_top_boundaries([0, 10], 3), [0, 10])

    def test_sequential_default_duration(self):
        steps = self._steps(3)
        out = rr.resolve_ranges(
            steps, {}, [], {}, gap=10, use_selected_keys=False,
            last_resolved=[], default_duration=200,
        )
        # uniform 200f each, gap 10, anchored at 0
        self.assertEqual(
            [(sid, s, e) for sid, s, e, _ in out],
            [("A01", 0.0, 200.0), ("A02", 210.0, 410.0), ("A03", 420.0, 620.0)],
        )

    def test_user_range_pins_and_advances_cursor(self):
        steps = self._steps(2)
        out = rr.resolve_ranges(
            steps, {"A01": (50.0, 100.0)}, [], {}, gap=10,
            use_selected_keys=False, last_resolved=[], default_duration=200,
        )
        first = out[0]
        self.assertEqual((first[0], first[1], first[2], first[3]), ("A01", 50.0, 100.0, True))
        # A02 starts after A01's end + gap
        self.assertEqual(out[1][1], 110.0)

    def test_gap_boundaries_place_steps(self):
        steps = self._steps(2)
        out = rr.resolve_ranges(
            steps, {}, [30.0, 80.0], {}, gap=5,
            use_selected_keys=False, last_resolved=[], default_duration=0,
        )
        self.assertEqual(out[0][1], 30.0)
        self.assertEqual(out[1][1], 80.0)

    def test_selected_keys_no_regions_returns_empty(self):
        out = rr.resolve_ranges(
            self._steps(2), {}, [], {}, gap=0,
            use_selected_keys=True, last_resolved=[],
        )
        self.assertEqual(out, [])


class TestStatusRollup(unittest.TestCase):
    def test_worst_of_children(self):
        st = StepStatus(
            step_id="A01", built=True,
            objects=[
                ObjectStatus("a", True, "valid"),
                ObjectStatus("b", False, "missing_object"),
            ],
        )
        self.assertEqual(st.status, "missing_object")
        self.assertEqual(st.missing_count, 1)

    def test_locked_wins(self):
        st = StepStatus(step_id="A01", built=True, locked=True,
                        objects=[ObjectStatus("a", False, "missing_object")])
        self.assertEqual(st.status, "locked")

    def test_unbuilt(self):
        st = StepStatus(step_id="A01", built=False)
        self.assertEqual(st.status, "missing_shot")


class TestManifestEngine(unittest.TestCase):
    """The pure ShotManifest planner + commit against a real ShotStore."""

    def setUp(self):
        self._prefs = tempfile.mkdtemp(prefix="mani_engine_")
        ShotStore._prefs_dir_override = self._prefs
        self.store = ShotStore()
        self.mani = ShotManifest(self.store)

    def tearDown(self):
        ShotStore._prefs_dir_override = None

    def _steps(self, ids):
        return [
            BuilderStep(sid, "A", "t", "", [BuilderObject(f"{sid}_geo")])
            for sid in ids
        ]

    def test_sync_creates_shots_sequentially(self):
        actions, _, assessment = self.mani.sync(
            self._steps(["A01", "A02"]), initial_shot_length=100
        )
        self.assertEqual(actions, {"A01": "created", "A02": "created"})
        shots = self.store.sorted_shots()
        self.assertEqual([s.name for s in shots], ["A01", "A02"])
        self.assertEqual((shots[0].start, shots[0].end), (1.0, 101.0))
        self.assertEqual({a.status for a in assessment}, {"valid"})

    def test_resync_unchanged_is_skipped(self):
        steps = self._steps(["A01"])
        self.mani.sync(steps, initial_shot_length=100)
        actions, _, _ = self.mani.sync(steps, initial_shot_length=100)
        self.assertEqual(actions, {"A01": "skipped"})

    def test_remove_missing(self):
        self.mani.sync(self._steps(["A01", "A02"]), initial_shot_length=100)
        actions, _, _ = self.mani.sync(self._steps(["A01"]), initial_shot_length=100)
        self.assertEqual(actions["A02"], "removed")
        self.assertEqual([s.name for s in self.store.sorted_shots()], ["A01"])

    def test_explicit_ranges_place_shots(self):
        steps = self._steps(["A01", "A02"])
        ranges = {"A01": (10.0, 40.0), "A02": (50.0, 90.0)}
        self.mani.sync(steps, ranges=ranges, initial_shot_length=100)
        shots = {s.name: (s.start, s.end) for s in self.store.sorted_shots()}
        self.assertEqual(shots["A01"], (10.0, 40.0))
        self.assertEqual(shots["A02"], (50.0, 90.0))

    def test_resync_changed_objects_is_patched(self):
        self.mani.sync(self._steps(["A01"]), initial_shot_length=100)
        steps2 = [BuilderStep("A01", "A", "t", "", [BuilderObject("new_geo")])]
        actions, _, _ = self.mani.sync(steps2, initial_shot_length=100)
        self.assertEqual(actions["A01"], "patched")
        self.assertIn("new_geo", self.store.shot_by_name("A01").objects)

    def test_existing_audio_shot_regrows_when_clip_lengthens(self):
        measured = {"v": 100.0}

        class AudioMani(ShotManifest):
            def _measure_audio(self, obj):
                return measured["v"]

        store = ShotStore()
        mani = AudioMani(store)
        step = [
            BuilderStep("V01", "V", "t", "",
                        [BuilderObject("vo", kind="audio", source_path="x.wav")])
        ]
        mani.sync(step, initial_shot_length=50)
        self.assertAlmostEqual(store.sorted_shots()[0].duration, 100.0)
        # clip lengthens -> re-sync must grow the EXISTING shot (the fixed path)
        measured["v"] = 300.0
        mani.sync(step, initial_shot_length=50)
        self.assertAlmostEqual(store.sorted_shots()[0].duration, 300.0)

    def test_audio_grow_via_measure_hook(self):
        class AudioMani(ShotManifest):
            def _measure_audio(self, obj):
                return 500.0

        store = ShotStore()
        mani = AudioMani(store)
        step = [
            BuilderStep(
                "V01", "V", "t", "",
                [BuilderObject("vo", kind="audio", source_path="x.wav")],
            )
        ]
        mani.sync(step, initial_shot_length=50)
        sh = store.sorted_shots()[0]
        self.assertAlmostEqual(sh.end - sh.start, 500.0)

    def test_audio_grow_duration_override_hook_is_respected(self):
        # A DCC layer may override _audio_grow_duration (e.g. to route through
        # its own compute_duration binding); the re-sync grow pass must honor
        # the override, not just the _measure_audio default underneath it.
        class GrowMani(ShotManifest):
            def _measure_audio(self, obj):
                return 100.0

            def _audio_grow_duration(self, audio_objs):
                return 250.0

        store = ShotStore()
        mani = GrowMani(store)
        step = [
            BuilderStep(
                "V01", "V", "t", "",
                [BuilderObject("vo", kind="audio", source_path="x.wav")],
            )
        ]
        mani.sync(step, initial_shot_length=50)
        mani.sync(step, initial_shot_length=50)  # grow pass on the existing shot
        sh = store.sorted_shots()[0]
        self.assertAlmostEqual(sh.end - sh.start, 250.0)

    def test_audio_grow_duration_default_is_max_of_template_and_measured(self):
        # Pure default contract: the larger of the behavior-template duration
        # and the measured clip length.
        objs = [
            BuilderObject(
                "vo", kind="audio", source_path="x.wav", behaviors=["fade_in"]
            )
        ]

        class ShortClip(ShotManifest):
            def _measure_audio(self, obj):
                return 5.0

        class LongClip(ShotManifest):
            def _measure_audio(self, obj):
                return 99.0

        self.assertEqual(ShortClip(ShotStore())._audio_grow_duration(objs), 15.0)
        self.assertEqual(LongClip(ShotStore())._audio_grow_duration(objs), 99.0)

    def test_from_csv_classmethod_returns_subclass(self):
        class Sub(ShotManifest):
            pass

        st = ShotStore()
        m = Sub(st)
        self.assertIsInstance(m, Sub)

    def test_resolve_duration_pure_behavior_summation(self):
        step = BuilderStep(
            "A01", "A", "t", "", [BuilderObject("g", behaviors=["fade_in"])]
        )
        dur, beh_span, aud = resolve_duration(
            step, initial_shot_length=200, fit_mode="extend_only", fps=24.0
        )
        self.assertGreaterEqual(dur, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
