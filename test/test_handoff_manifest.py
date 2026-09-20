# !/usr/bin/python
# coding=utf-8
"""Tests for the hand-off sidecar (``core_utils/handoff_manifest.py``) and its
replay plan (``core_utils/manifest_plan.py``).

Pure: the only I/O is a real sidecar written into a temp dir, so the read rules
(absent / malformed / not-an-object) are exercised against real files rather
than a stubbed ``open``.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pythontk as ptk
from pythontk.core_utils.handoff_manifest import HandoffManifest
from pythontk.core_utils.manifest_plan import ManifestPlan


class _SidecarCase(unittest.TestCase):
    """A temp dir holding a payload's sidecar."""

    def setUp(self):
        self._tmp = ptk.TempArtifacts("test_handoff_manifest", policy="scoped")
        self.dir = self._tmp.dir_path()
        self.payload = os.path.join(self.dir, "payload.fbx")

    def tearDown(self):
        self._tmp.cleanup()

    def write_sidecar(self, data, *, raw=None):
        """Write *data* (or *raw* text) as the payload's sidecar; return its path."""
        path = self.payload + HandoffManifest.SUFFIX
        with open(path, "w", encoding="utf-8") as fh:
            if raw is not None:
                fh.write(raw)
            else:
                json.dump(data, fh)
        return path


class TestPathRules(unittest.TestCase):
    """The suffix is the whole naming rule, and applying it twice is a no-op."""

    def test_path_for_appends_the_suffix(self):
        self.assertEqual(
            HandoffManifest.path_for("C:/tmp/scene.fbx"),
            "C:/tmp/scene.fbx" + HandoffManifest.SUFFIX,
        )

    def test_path_for_is_idempotent(self):
        once = HandoffManifest.path_for("C:/tmp/scene.usd")
        self.assertEqual(HandoffManifest.path_for(once), once)

    def test_payload_path_is_the_inverse(self):
        m = HandoffManifest.read("C:/tmp/missing.fbx")
        self.assertEqual(m.payload_path, "C:/tmp/missing.fbx")

    def test_payload_path_is_none_when_built_in_memory(self):
        self.assertIsNone(HandoffManifest.build(scene={}).payload_path)


class TestReading(_SidecarCase):
    """Reading is tolerant: a damaged sidecar never costs a landed payload."""

    def test_reads_the_sections(self):
        self.write_sidecar({"version": 1, "materials": [{"name": "M"}]})
        m = HandoffManifest.read(self.payload)
        self.assertEqual(m.version, 1)
        self.assertEqual(m.get(HandoffManifest.MATERIALS), [{"name": "M"}])

    def test_accepts_the_sidecar_path_as_well_as_the_payload(self):
        path = self.write_sidecar({"scene": {"fps": 24}})
        self.assertEqual(
            HandoffManifest.read(path).get("scene"),
            HandoffManifest.read(self.payload).get("scene"),
        )

    def test_absent_sidecar_is_empty_and_not_a_failure(self):
        m = HandoffManifest.read(self.payload)
        self.assertEqual(dict(m), {})
        self.assertFalse(
            m.unreadable, "an absent sidecar is a producer with nothing to say"
        )

    def test_malformed_json_is_empty_and_says_so(self):
        self.write_sidecar(None, raw="{not json")
        m = HandoffManifest.read(self.payload)
        self.assertEqual(dict(m), {})
        self.assertTrue(m.unreadable)

    def test_a_json_list_carries_no_section(self):
        self.write_sidecar([1, 2, 3])
        m = HandoffManifest.read(self.payload)
        self.assertEqual(dict(m), {})
        self.assertFalse(m.unreadable, "readable, just not a document with sections")

    def test_a_null_sidecar_is_readable_and_empty(self):
        self.write_sidecar(None)
        m = HandoffManifest.read(self.payload)
        self.assertEqual(dict(m), {})
        self.assertFalse(
            m.unreadable, "the sentinel must not confuse null with a failure"
        )

    def test_write_is_atomic(self):
        # An unencodable value must leave the previous sidecar intact rather
        # than truncating it: a half-written manifest reads as "no sections",
        # which would silently drop every bit of fidelity the producer sent.
        HandoffManifest.build(scene={"fps": 24}).write(self.payload)
        with self.assertRaises(TypeError):
            HandoffManifest.build(scene={"fps": object()}).write(self.payload)
        self.assertEqual(HandoffManifest.read(self.payload).get("scene"), {"fps": 24})

    def test_path_survives_a_failed_read(self):
        self.write_sidecar(None, raw="{")
        self.assertTrue(
            HandoffManifest.read(self.payload).path.endswith(".manifest.json")
        )


class TestMappingSurface(unittest.TestCase):
    """It stands in for the plain dict the appliers used to pass around."""

    def setUp(self):
        self.m = HandoffManifest({"rig": {"graph": {}}, "materials": []})

    def test_getitem_get_and_contains(self):
        self.assertEqual(self.m["rig"], {"graph": {}})
        self.assertEqual(self.m.get("nope", "dflt"), "dflt")
        self.assertIn("materials", self.m)
        self.assertNotIn("shots", self.m)

    def test_dict_round_trip(self):
        self.assertEqual(dict(self.m), {"rig": {"graph": {}}, "materials": []})

    def test_truthiness_follows_the_sections(self):
        self.assertTrue(self.m)
        self.assertFalse(HandoffManifest())

    def test_data_is_a_copy(self):
        self.m.data["materials"].append("x")
        self.assertEqual(
            self.m["materials"], [], "the document must not be mutable through data"
        )

    def test_carries_is_truthiness_not_presence(self):
        self.assertTrue(self.m.carries("rig"))
        self.assertFalse(
            self.m.carries("materials"),
            "an empty section asks for the same no-op as an absent one",
        )


class TestBuildAndWrite(_SidecarCase):
    """``None`` is the 'nothing to say' value; an empty list is a real answer."""

    def test_build_drops_none_and_keeps_empty(self):
        m = HandoffManifest.build(materials=[], shots=None, rig={"graph": {}})
        self.assertEqual(sorted(m), ["materials", "rig"])

    def test_write_lands_beside_the_payload(self):
        path = HandoffManifest.build(scene={"fps": 24}).write(self.payload)
        self.assertEqual(path, self.payload + HandoffManifest.SUFFIX)
        with open(path, "r", encoding="utf-8") as fh:
            self.assertEqual(json.load(fh), {"scene": {"fps": 24}})

    def test_write_round_trips_through_read(self):
        HandoffManifest.build(
            version=2, format=HandoffManifest.FORMAT_PATHS, instances=[["a", "b"]]
        ).write(self.payload)
        m = HandoffManifest.read(self.payload)
        self.assertEqual(m.version, 2)
        self.assertEqual(m.format, "paths")
        self.assertEqual(m[HandoffManifest.INSTANCES], [["a", "b"]])

    def test_write_without_a_path_raises(self):
        with self.assertRaises(ValueError):
            HandoffManifest.build(scene={}).write()

    def test_write_remembers_where_it_landed(self):
        m = HandoffManifest.build(scene={})
        m.write(self.payload)
        self.assertEqual(m.path, self.payload + HandoffManifest.SUFFIX)


class TestSectionVocabulary(unittest.TestCase):
    """The names are the contract both packages import."""

    def test_sections_are_unique_and_listed(self):
        self.assertEqual(
            len(HandoffManifest.SECTIONS), len(set(HandoffManifest.SECTIONS))
        )

    def test_every_listed_section_has_a_constant(self):
        for name in HandoffManifest.SECTIONS:
            attr = name.upper()
            self.assertTrue(hasattr(HandoffManifest, attr), f"no constant for {name!r}")
            self.assertEqual(getattr(HandoffManifest, attr), name)

    def test_the_dialect_values_are_pinned(self):
        # The instance gates in both consumers compare against these, and cached
        # sidecars on disk already carry them.
        self.assertEqual(HandoffManifest.FORMAT_NAMES, "names")
        self.assertEqual(HandoffManifest.FORMAT_PATHS, "paths")

    def test_the_document_keys_are_not_sections(self):
        for key in (HandoffManifest.VERSION_KEY, HandoffManifest.FORMAT_KEY):
            self.assertNotIn(key, HandoffManifest.SECTIONS)

    def test_the_on_disk_spellings_are_pinned(self):
        # These strings are the wire format: four producer templates, two
        # bridges and two consumers agree on them, and cached sidecars on disk
        # already carry them. Renaming one is a contract change, not a rename.
        self.assertEqual(
            HandoffManifest.SECTIONS,
            (
                "materials",
                "scene_materials",
                "shading_groups",
                "empties",
                "transforms",
                "instances",
                "visibility",
                "lights",
                "world",
                "skins",
                "bones",
                "scene",
                "shots",
                "rig",
                "machinery",
                # 2026-09-19: every portable scene record, keyed by record --
                # RecordTransfer writes and reads it (its RECORDS_SECTION).
                "records",
            ),
        )
        self.assertEqual(HandoffManifest.SUFFIX, ".manifest.json")


class TestPlanGating(unittest.TestCase):
    """A plan's length is the number of steps that will really run."""

    def setUp(self):
        self.m = HandoffManifest({"materials": [{"name": "M"}], "lights": []})

    def test_admits_a_carried_section(self):
        plan = self.m.plan().add("materials", "Materials", lambda: None)
        self.assertEqual(plan.labels, ["Materials"])

    def test_drops_an_absent_section(self):
        plan = self.m.plan().add("shots", "Shots", lambda: None)
        self.assertEqual(plan.labels, [])

    def test_drops_an_empty_section(self):
        plan = self.m.plan().add("lights", "Lights", lambda: None)
        self.assertEqual(plan.labels, [])

    def test_none_section_always_runs(self):
        plan = ManifestPlan(None).add(None, "Import", lambda: None)
        self.assertEqual(plan.labels, ["Import"])

    def test_when_false_drops_a_carried_section(self):
        plan = self.m.plan().add("materials", "Materials", lambda: None, when=False)
        self.assertEqual(plan.labels, [])

    def test_gates_are_and_not_or(self):
        plan = self.m.plan().add("shots", "Shots", lambda: None, when=True)
        self.assertEqual(
            plan.labels, [], "a when=True must not resurrect an absent section"
        )

    def test_a_malformed_manifest_admits_nothing_section_driven(self):
        plan = ManifestPlan("not a mapping").add("materials", "Materials", lambda: None)
        self.assertEqual(plan.labels, [])

    def test_add_chains(self):
        plan = self.m.plan().add(None, "A", lambda: None).add(None, "B", lambda: None)
        self.assertEqual(plan.labels, ["A", "B"])


class TestPlanRun(unittest.TestCase):
    """Order, results, progress and cancellation."""

    def test_runs_in_order_and_returns_results(self):
        seen = []
        plan = ManifestPlan()
        plan.add(None, "A", lambda: seen.append("a") or "ra")
        plan.add(None, "B", lambda: seen.append("b") or "rb")
        self.assertEqual(plan.run(), ["ra", "rb"])
        self.assertEqual(seen, ["a", "b"])

    def test_progress_counts_only_admitted_steps(self):
        m = HandoffManifest({"materials": []})
        ticks = []
        plan = m.plan()
        plan.add(None, "A", lambda: None)
        plan.add("materials", "Dropped", lambda: None)
        plan.add(None, "B", lambda: None)
        plan.run(progress=lambda d, t, x: ticks.append((d, t, x)), done_label="Done")
        self.assertEqual(ticks, [(0, 2, "A"), (1, 2, "B"), (2, 2, "Done")])

    def test_no_final_report_without_a_done_label(self):
        ticks = []
        ManifestPlan().add(None, "A", lambda: None).run(
            progress=lambda d, t, x: ticks.append(x)
        )
        self.assertEqual(ticks, ["A"])

    def test_false_from_progress_stops_before_the_step(self):
        ran = []
        plan = ManifestPlan()
        plan.add(None, "A", lambda: ran.append("a"))
        plan.add(None, "B", lambda: ran.append("b"))
        with self.assertRaises(ptk.OperationCancelled):
            plan.run(progress=lambda d, t, x: x != "B")
        self.assertEqual(
            ran, ["a"], "what already applied stays; the stopped step never ran"
        )

    def test_the_final_report_cannot_cancel_a_finished_run(self):
        """Every step has already run, so a stop request there has nothing left
        to stop -- raising would fail a run that actually succeeded."""
        ran = []
        plan = ManifestPlan()
        plan.add(None, "A", lambda: ran.append("a"))
        # False on every call, including the last one.
        plan.run(progress=lambda d, t, x: x == "A", done_label="Done")
        self.assertEqual(ran, ["a"])

    def test_cancel_message_uses_the_prefix(self):
        plan = ManifestPlan(cancel_prefix="Import stopped before")
        plan.add(None, "Rebuilding shots", lambda: None)
        with self.assertRaises(ptk.OperationCancelled) as cm:
            plan.run(progress=lambda d, t, x: False)
        self.assertEqual(str(cm.exception), "Import stopped before: Rebuilding shots")


class TestPlanErrorPolicy(unittest.TestCase):
    """Loud by default; quiet only where the author said so."""

    @staticmethod
    def _boom():
        raise RuntimeError("nope")

    def test_a_step_raises_by_default(self):
        plan = ManifestPlan(on_error=lambda label, e: None)
        plan.add(None, "Structural", self._boom)
        with self.assertRaises(RuntimeError):
            plan.run()

    def test_best_effort_is_reported_and_the_run_continues(self):
        failures, ran = [], []
        plan = ManifestPlan(on_error=lambda label, e: failures.append((label, str(e))))
        plan.add(None, "Fidelity", self._boom, best_effort=True)

        def after():
            ran.append("after")
            return "after"

        plan.add(None, "After", after)
        self.assertEqual(plan.run(), [None, "after"])
        self.assertEqual(failures, [("Fidelity", "nope")])
        self.assertEqual(ran, ["after"])

    def test_best_effort_without_a_handler_still_raises(self):
        plan = ManifestPlan()
        plan.add(None, "Fidelity", self._boom, best_effort=True)
        with self.assertRaises(RuntimeError):
            plan.run()

    def test_best_effort_never_swallows_a_stop_request(self):
        def stop():
            raise ptk.OperationCancelled("stop")

        seen = []
        plan = ManifestPlan(on_error=lambda label, e: seen.append(label))
        plan.add(None, "Fidelity", stop, best_effort=True)
        with self.assertRaises(ptk.OperationCancelled):
            plan.run()
        self.assertEqual(seen, [], "a cancel outranks any step's error policy")


class TestPublicSurface(unittest.TestCase):
    """Both names are reachable the way consumers import them."""

    def test_exported_from_the_package_root(self):
        self.assertIs(ptk.HandoffManifest, HandoffManifest)
        self.assertIs(ptk.ManifestPlan, ManifestPlan)


if __name__ == "__main__":
    unittest.main(verbosity=2)
