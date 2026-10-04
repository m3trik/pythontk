# !/usr/bin/python
# coding=utf-8
"""Tests for :class:`pythontk.SceneExporterBase` -- the exporter's DCC-free shell.

The DCC suites (mayatk / blendertk ``test_scene_exporter``) drive it through
their ``SceneExporter`` subclasses and real task managers; these pin the shell
on a stub task manager: the progress stream, the per-check override decision
and its dialog body, and the hooks a host plugs into.
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
        self.task_definitions = {}
        self.check_definitions = {}

    def _dispatchable_count(self, tasks):
        return len(tasks)

    def run_tasks(self, tasks):
        """Fail every ``check_*`` in *tasks*, asking the handler as the real
        runner does; the run proceeds when each failure was overridden."""
        self.handler_seen = self.failed_check_handler
        checks = [n for n in tasks if n.startswith("check_")]
        overridden, every = [], False
        for i, name in enumerate(checks):
            if every:
                overridden.append(name)
                continue
            answer = self.failed_check_handler(name, ["bad"], checks[i + 1 :])
            if answer == "abort":
                break
            every = answer == "override_all"
            overridden.append(name)
        self._last_failed_checks = list(checks)
        self._last_overridden_checks = overridden
        return len(overridden) == len(checks)


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


class TestCheckFailure(unittest.TestCase):
    """A failed check is decided where it fails: override it, override it and
    every later one, or stop (``decide_check_failure``; added 2026-10-04)."""

    def test_headless_asks_on_the_console_for_this_one_check(self):
        import pythontk as ptk

        exporter = _Exporter()
        asked = []
        exporter.confirm = lambda question: asked.append(question) or True
        answer = exporter.decide_check_failure(
            "check_path_length", ["too long: C:/a"], ["check_b"]
        )
        self.assertEqual(answer, ptk.TaskFactory.CHECK_OVERRIDE)
        self.assertIn("Check failed: Path Length.", asked[0])
        self.assertIn("too long: C:/a", asked[0])
        self.assertIn("1 check(s) still to run.", asked[0])
        exporter.confirm = lambda question: False
        self.assertEqual(
            exporter.decide_check_failure("check_a", [], []),
            ptk.TaskFactory.CHECK_ABORT,
        )

    def test_a_check_reads_as_its_panel_row(self):
        exporter = _Exporter()
        exporter.task_manager.check_definitions = {
            "check_a": {"setText": "Check For A"},
            "check_b": {"set_row_label": "Max B"},
        }
        self.assertEqual(exporter.check_label("check_a"), "Check For A")
        self.assertEqual(exporter.check_label("check_b"), "Max B")
        self.assertEqual(exporter.check_label("check_no_row"), "No Row")

    def test_the_dialog_body_names_the_failure_and_what_is_left(self):
        exporter = _Exporter()
        exporter.task_manager.check_definitions = {
            "check_a": {"setText": "Check <A>"},
            "check_b": {"setText": "Check B\n"},  # a stray break in a label
        }
        exporter._overridden_checks = ["check_b"]
        body = exporter.check_failure_html(
            "check_a",
            ["C:/x & <y>.png"] + [f"m{i}" for i in range(9)],
            ["check_b"] + [f"check_r{i}" for i in range(14)],
        )
        self.assertNotIn(chr(10), body, "a newline becomes a break in the box")
        self.assertIn("Check &lt;A&gt;", body, "labels are escaped")
        self.assertIn("C:/x &amp; &lt;y&gt;.png", body, "messages are escaped")
        self.assertIn("4 more in the log", body)
        self.assertIn("Still to run (15):", body)
        self.assertIn("and 3 more", body)
        self.assertIn("Already overridden this run: Check B", body)
        for button in exporter.CHECK_FAILURE_CHOICES:
            self.assertIn(f"<b>{button}</b>", body, "each button is explained")

    def test_the_pipeline_records_each_override_for_the_banner(self):
        import pythontk as ptk

        exporter = _Exporter()
        answers = [ptk.TaskFactory.CHECK_OVERRIDE, ptk.TaskFactory.CHECK_OVERRIDE]
        exporter.decide_check_failure = lambda c, m, r: answers.pop(0)
        exporter._overridden_checks = []
        with self.assertLogs(exporter.logger, "WARNING") as log:
            self.assertTrue(exporter._run_task_pipeline({"check_a": 1, "check_b": 1}))
        self.assertEqual(exporter._overridden_checks, ["check_a", "check_b"])
        self.assertIn("2 failed check(s): check_a, check_b", log.output[-1])
        self.assertIsNone(
            exporter.task_manager.failed_check_handler,
            "the handler is armed for the run only",
        )

    def test_an_override_all_is_recorded_whole_from_the_runner(self):
        """After Override All the runner lets failures through without asking,
        so only its list names every one."""
        import pythontk as ptk

        exporter = _Exporter()
        exporter.decide_check_failure = lambda c, m, r: (
            ptk.TaskFactory.CHECK_OVERRIDE_ALL
        )
        exporter._overridden_checks = []
        self.assertTrue(
            exporter._run_task_pipeline({"check_a": 1, "check_b": 1, "check_c": 1})
        )
        self.assertEqual(exporter._overridden_checks, ["check_a", "check_b", "check_c"])

    def test_a_cancel_blocks_the_write_and_names_what_was_kept(self):
        import pythontk as ptk

        exporter = _Exporter()
        exporter.decide_check_failure = lambda c, m, r: ptk.TaskFactory.CHECK_ABORT
        exporter.task_manager.kept_edits = ["repaired names"]
        exporter._overridden_checks = []
        with self.assertLogs(exporter.logger, "WARNING") as log:
            self.assertFalse(exporter._run_task_pipeline({"check_a": 1}))
        self.assertIn("Export blocked by failed checks.", log.output[-1])
        self.assertIn("repaired names", log.output[-1])
        self.assertEqual(exporter._overridden_checks, [])

    def test_the_retired_after_the_run_consent_still_answers(self):
        exporter = _Exporter()
        exporter.task_manager._last_failed_checks = ["check_a", "check_b"]
        asked = []
        exporter.confirm = lambda question: asked.append(question) or True
        with self.assertWarns(DeprecationWarning):
            self.assertTrue(exporter.confirm_check_override())
        self.assertIn("2 validation check(s) failed: check_a, check_b.", asked[0])


class TestRootExport(unittest.TestCase):
    def test_the_root_serves_the_class(self):
        import pythontk as ptk

        self.assertIs(ptk.SceneExporterBase, SceneExporterBase)


if __name__ == "__main__":
    unittest.main(verbosity=2)
