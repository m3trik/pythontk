# !/usr/bin/python
# coding=utf-8
"""Tests for :class:`pythontk.SceneDataSidecarBase` -- the host-agnostic sidecar.

The DCC suites (mayatk / blendertk ``test_scene_data_sidecar``) drive the same
code through their subclasses and scene hooks; these pin the base contract on
plain path strings, and the hook seams a subclass plugs into.
"""

import json
import os
import shutil
import unittest

from pythontk.core_utils.engines.scene_export.scene_data_sidecar import (
    SceneDataSidecarBase as SD,
)

HERE = os.path.dirname(os.path.abspath(__file__))


class _Closing(SD):
    """A host whose export ships every node's parent chain (Maya's rule)."""

    @classmethod
    def _close(cls, paths):
        return cls.with_ancestors(paths)


class SidecarCase(unittest.TestCase):
    def setUp(self):
        self.dir = os.path.join(HERE, "temp_tests", "scene_data_sidecar")
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir)
        self.export = os.path.join(self.dir, "asset_v003.fbx")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestPaths(SidecarCase):
    def test_one_sidecar_per_stem_and_per_series(self):
        self.assertEqual(
            os.path.basename(SD.manifest_path_for(self.export)),
            ".asset_v003.scene_data.json",
        )
        self.assertEqual(
            os.path.basename(SD.manifest_path_for(self.export, base_stem=True)),
            ".asset.scene_data.json",
        )
        self.assertEqual(SD.base_stem("arch_v2_proxy.fbx"), "arch_v2_proxy")


class TestManifestIO(SidecarCase):
    def test_write_then_read_round_trips_paths_and_data(self):
        written = SD.write_manifest(
            self.export, {"A|B", "A"}, data={"shot_metadata": {"x": 1}}
        )
        self.assertEqual(written, SD.manifest_path_for(self.export))
        self.assertEqual(SD.read_manifest(self.export), {"A", "A|B"})
        self.assertEqual(SD.read_data(self.export), {"shot_metadata": {"x": 1}})
        with open(written, encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertEqual(payload["format"], SD.FORMAT_VERSION)
        self.assertEqual(payload["hierarchy"]["object_count"], 2)
        self.assertFalse(os.path.exists(written + ".tmp"))

    def test_a_rewrite_lands_whole_and_keeps_names_readable(self):
        SD.write_manifest(self.export, {"A"})
        written = SD.write_manifest(self.export, {"Café|B"})  # over a hidden file
        with open(written, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("Café|B", text)  # not escaped: read by eye
        self.assertEqual(SD.read_manifest(self.export), {"Café|B"})
        self.assertEqual(
            [n for n in os.listdir(self.dir) if n.endswith(".tmp")], [], "stray tmp"
        )

    def test_a_failed_write_returns_none_and_leaves_nothing(self):
        export = os.path.join(self.dir, "missing_subdir", "shot.fbx")
        self.assertIsNone(SD.write_manifest(export, {"A"}))
        self.assertEqual(os.listdir(self.dir), [])

    def test_a_write_sweeps_the_v2_companions(self):
        manifest = SD.manifest_path_for(self.export)
        for stale in (manifest + ".prev", SD.diff_report_path_for(self.export)):
            with open(stale, "w") as fh:
                fh.write("{}")
        SD.write_manifest(self.export, {"A"})
        self.assertFalse(os.path.exists(manifest + ".prev"))
        self.assertFalse(os.path.exists(SD.diff_report_path_for(self.export)))

    def test_a_v1_name_migrates_to_the_current_one(self):
        legacy = SD._legacy_manifest_path_for(self.export)
        with open(legacy, "w", encoding="utf-8") as fh:
            json.dump({"paths": ["A"], "object_count": 1, "hash": "h"}, fh)
        self.assertEqual(
            SD.migrate_legacy(self.export), SD.manifest_path_for(self.export)
        )
        self.assertFalse(os.path.exists(legacy))
        self.assertEqual(SD.read_manifest(self.export), {"A"})

    def test_the_newest_version_becomes_the_series_sidecar(self):
        for n in (2, 10):
            path = os.path.join(self.dir, ".asset_v%d.scene_data.json" % n)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"format": 3, "hierarchy": {"paths": ["v%d" % n]}}, fh)
        SD.migrate_legacy(self.export, base_stem=True)
        self.assertEqual(SD.read_manifest(self.export, base_stem=True), {"v10"})


class TestCompare(SidecarCase):
    def test_no_baseline_passes(self):
        self.assertEqual(SD.compare(self.export, {"A"}), (True, [], []))

    def test_the_set_diff_by_default(self):
        SD.write_manifest(self.export, {"A|B"})
        self.assertEqual(
            SD.compare(self.export, {"A", "A|B", "C"}), (False, [], ["A", "C"])
        )

    def test_a_closing_host_reads_a_leaves_only_record_as_its_group(self):
        SD.write_manifest(self.export, {"A|B"})
        self.assertEqual(_Closing.compare(self.export, {"A", "A|B"}), (True, [], []))


class TestHooks(unittest.TestCase):
    def test_the_defaults_record_exactly_the_given_paths(self):
        self.assertEqual(SD.build_full_path_set(["A|B", "A|B", "C"]), {"A|B", "C"})

    def test_a_host_hook_is_what_the_full_build_goes_through(self):
        class _Host(SD):
            @staticmethod
            def expand_to_descendants(objects):
                return [o + "|child" for o in objects] + list(objects)

            @staticmethod
            def drop_intermediate(nodes):
                return [n for n in nodes if not n.endswith("Orig")]

        self.assertEqual(
            _Host.build_full_path_set(["A", "AOrig"]), {"A", "A|child", "AOrig|child"}
        )

    def test_with_ancestors_closes_a_set(self):
        self.assertEqual(SD.with_ancestors({"A|B|C"}), {"A", "A|B", "A|B|C"})


class TestReport(unittest.TestCase):
    def test_reparenting_is_detected_and_reported(self):
        missing, extra = ["B", "B|c"], ["G|B", "G|B|c"]
        self.assertEqual(SD.detect_reparenting(missing, extra), [("B", "G", 2)])
        report = SD.format_diff_report(missing, extra)
        self.assertIn("Reparented: 'B' moved under 'G' (2 nodes)", report)
        self.assertIn("  - B|c", report)
        self.assertIn("  + G|B|c", report)

    def test_top_level_rollup(self):
        self.assertEqual(SD.get_top_level(["A", "A|B", "C|D"]), ["A", "C|D"])
        self.assertEqual(SD.count_descendants("A", {"A", "A|B", "AB"}), 2)


class TestRootExport(unittest.TestCase):
    def test_the_root_serves_the_class(self):
        import pythontk as ptk

        self.assertIs(ptk.SceneDataSidecarBase, SD)


if __name__ == "__main__":
    unittest.main(verbosity=2)
