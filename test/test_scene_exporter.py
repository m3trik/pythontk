# !/usr/bin/python
# coding=utf-8
"""Tests for :class:`pythontk.SceneExporterBase` -- the exporter's DCC-free shell.

The DCC suites (mayatk / blendertk ``test_scene_exporter``) drive it through
their ``SceneExporter`` subclasses and real task managers; these pin the shell
on a stub task manager: the progress stream, the check-override consent and
resume, and the hooks a host plugs into.
"""

import os
import unittest

from pythontk.core_utils.engines.scene_export.scene_exporter import (
    SceneExporterBase,
)


class _StubTasks:
    """The slice of a host TaskManager the shell talks to."""

    def __init__(self, logger):
        self.logger = logger
        self.progress_callback = None
        self.kept_edits = []
        self._last_failed_checks = []
        self._last_skipped_tasks = []
        self._last_skipped_checks = []
        self._last_task_count = 0
        self._last_check_count = 0
        self.dispatched = []
        self.task_definitions = {}
        self.check_definitions = {}

    def _dispatchable_count(self, tasks):
        return len(tasks)

    def _execute_tasks_and_checks(self, tasks, checks):
        self.dispatched.append(dict(tasks))
        self._last_task_count = self._last_check_count = -1  # a pass re-stamps


class _Exporter(SceneExporterBase):
    TASK_MANAGER_CLASS = _StubTasks


class TestHooks(unittest.TestCase):
    def test_the_host_task_manager_is_built_with_the_exporter_logger(self):
        exporter = _Exporter()
        self.assertIsInstance(exporter.task_manager, _StubTasks)
        self.assertIs(exporter.task_manager.logger, exporter.logger)
        self.assertEqual(exporter.logger.name.rsplit(".", 1)[-1], "_Exporter")

    def test_the_export_folder_falls_back_to_the_saved_scene(self):
        exporter = _Exporter()
        self.assertEqual(exporter._resolve_export_dir(None), "")
        exporter._saved_scene_path = lambda: os.path.join("proj", "scenes", "a.ma")
        self.assertEqual(
            exporter._resolve_export_dir(None), os.path.join("proj", "scenes")
        )
        self.assertEqual(exporter._resolve_export_dir("out"), os.path.abspath("out"))

    def test_the_kept_edits_advice_is_the_hosts_word(self):
        exporter = _Exporter()
        exporter.task_manager.kept_edits = ["tied keys"]
        exporter.KEPT_EDITS_ADVICE = "Undo is on you."
        with self.assertLogs(exporter.logger, "WARNING") as log:
            exporter._warn_stopped_before_write("Stopped.")
        self.assertIn("tied keys. Undo is on you.", log.output[0])


class TestProgress(unittest.TestCase):
    def test_one_stream_counts_pipeline_entries_then_phases(self):
        exporter = _Exporter()
        seen = []
        exporter._progress_begin(
            lambda c, t, m: seen.append((c, t, m)), {"a": 1, "b": 2}, phases=2
        )
        exporter.task_manager.progress_callback(1, 2, "task a")
        exporter._progress_step("write")
        exporter._progress_step("glb")
        exporter._progress_finish("done")
        exporter._progress_end()
        self.assertEqual(
            seen,
            [(1, 4, "task a"), (1, 4, "write"), (2, 4, "glb"), (4, 4, "done")],
        )
        self.assertIsNone(exporter.task_manager.progress_callback)

    def test_a_cancel_is_honoured_only_before_the_write(self):
        import pythontk as ptk

        exporter = _Exporter()
        exporter._progress_begin(lambda c, t, m: False, {}, phases=1)
        with self.assertRaises(ptk.OperationCancelled):
            exporter._progress_step("write")
        exporter._progress_cancellable = False
        exporter._progress_note("finishing")  # ignored, not raised


class TestOverride(unittest.TestCase):
    def test_the_consent_names_the_failed_checks_and_defaults_to_no(self):
        exporter = _Exporter()
        exporter.task_manager._last_failed_checks = ["check_a", "check_b"]
        asked = []
        exporter.confirm = lambda question: asked.append(question) or True
        self.assertTrue(exporter.confirm_check_override())
        self.assertIn("2 validation check(s) failed: check_a, check_b.", asked[0])

    def test_resume_redispatches_only_the_skipped_tasks(self):
        exporter = _Exporter()
        tm = exporter.task_manager
        tm._last_skipped_tasks = ["b", "not_requested"]
        tm._last_skipped_checks = ["check_c"]
        tm._last_task_count, tm._last_check_count = 3, 2
        exporter._resume_skipped_tasks({"a": 1, "b": 2})
        self.assertEqual(tm.dispatched, [{"b": 2}])
        self.assertEqual((tm._last_task_count, tm._last_check_count), (3, 2))
        self.assertEqual(tm._last_skipped_checks, ["check_c"])


class TestRootExport(unittest.TestCase):
    def test_the_root_serves_the_class(self):
        import pythontk as ptk

        self.assertIs(ptk.SceneExporterBase, SceneExporterBase)


if __name__ == "__main__":
    unittest.main(verbosity=2)
