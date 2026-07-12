# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.task_factory (TaskFactory) — the generic,
host-free task/check pipeline primitive shared by mayatk's and blendertk's
scene exporters (their ``TaskManager`` subclasses it). The dispatch / ordering /
revert / check contract is pinned here once, DCC-free; the DCC suites cover the
concrete ``task_*`` / ``check_*`` methods each supplies.
"""
import unittest
from unittest.mock import MagicMock

from pythontk import TaskFactory


class _Recorder(TaskFactory):
    """A concrete TaskFactory with synthetic task/check/set methods."""

    def __init__(self):
        super().__init__(MagicMock())
        self.calls = []

    def task_plain(self, value):
        self.calls.append(("task_plain", value))

    def task_boom(self, value):
        self.calls.append(("task_boom", value))
        raise RuntimeError("boom")

    def task_noargs(self):
        self.calls.append(("task_noargs",))

    def task_two(self, a, b):
        self.calls.append(("task_two", a, b))

    def set_flag(self, value):
        self.calls.append(("set_flag", value))
        return "ORIGINAL"  # non-None result -> revertible

    def revert_flag(self, original):
        self.calls.append(("revert_flag", original))

    def set_a(self, value):
        self.calls.append(("set_a", value))
        return "A"

    def revert_a(self, original):
        self.calls.append(("revert_a", original))

    def set_b(self, value):
        self.calls.append(("set_b", value))
        return "B"

    def revert_b(self, original):
        self.calls.append(("revert_b", original))

    def check_ok(self, value):
        self.calls.append(("check_ok", value))
        return True

    def check_bad(self, value):
        self.calls.append(("check_bad", value))
        return (False, ["reason"])

    def check_empty(self, value):
        self.calls.append(("check_empty", value))
        return []  # a natural "no messages" shape -- must not crash


class TaskFactoryTest(unittest.TestCase):
    def test_no_tasks_returns_true(self):
        self.assertTrue(_Recorder().run_tasks({}))

    def test_tasks_run_and_passing_checks_return_true(self):
        r = _Recorder()
        self.assertTrue(r.run_tasks({"task_plain": "x", "check_ok": True}))
        names = [c[0] for c in r.calls]
        self.assertIn("task_plain", names)
        self.assertIn("check_ok", names)

    def test_failing_check_returns_false(self):
        self.assertFalse(_Recorder().run_tasks({"check_bad": True}))

    def test_set_task_is_reverted_with_its_return_value(self):
        r = _Recorder()
        r.run_tasks({"set_flag": True})
        self.assertIn(("set_flag", True), r.calls)
        # revert runs (LIFO) after the task pass, fed the set task's return value
        self.assertIn(("revert_flag", "ORIGINAL"), r.calls)

    def test_missing_method_is_skipped_not_crashed(self):
        # An unknown task name is warned + skipped; with no failing checks -> True.
        self.assertTrue(_Recorder().run_tasks({"task_absent": True}))

    def test_revert_runs_when_a_later_task_raises(self):
        # set_flag runs first (alphabetical), then task_boom raises: the set
        # state must still be reverted, or a failed export leaves the host
        # scene permanently mutated.
        r = _Recorder()
        with self.assertRaises(RuntimeError):
            r.run_tasks({"set_flag": True, "task_boom": True})
        self.assertIn(("revert_flag", "ORIGINAL"), r.calls)

    def test_revert_runs_when_the_with_body_raises(self):
        # An exception thrown into the generator at the yield point (e.g. from
        # check-result processing) must not skip reversion.
        r = _Recorder()
        with self.assertRaises(RuntimeError):
            with r._manage_context({"set_flag": True}):
                raise RuntimeError("body boom")
        self.assertIn(("revert_flag", "ORIGINAL"), r.calls)

    def test_empty_sequence_check_result_is_failure_not_crash(self):
        # An empty tuple/list result is falsy -> failed check, never IndexError.
        self.assertFalse(_Recorder().run_tasks({"check_empty": True}))

    def test_reverts_run_lifo(self):
        r = _Recorder()
        r.run_tasks({"set_a": True, "set_b": True})
        reverts = [c for c in r.calls if c[0].startswith("revert_")]
        self.assertEqual(reverts, [("revert_b", "B"), ("revert_a", "A")])

    def test_task_order_controls_execution_sequence(self):
        class _Ordered(_Recorder):
            TASK_ORDER = ["task_second", "task_first"]

            def task_first(self, value):
                self.calls.append(("task_first", value))

            def task_second(self, value):
                self.calls.append(("task_second", value))

        r = _Ordered()
        r.run_tasks({"task_first": 1, "task_second": 2, "task_plain": 3})
        names = [c[0] for c in r.calls]
        # TASK_ORDER first, then the unlisted task appended alphabetically.
        self.assertEqual(names, ["task_second", "task_first", "task_plain"])

    def test_zero_param_task_treats_value_as_enable_flag(self):
        r = _Recorder()
        r.run_tasks({"task_noargs": False})
        self.assertNotIn(("task_noargs",), r.calls)
        r.run_tasks({"task_noargs": True})
        self.assertIn(("task_noargs",), r.calls)

    def test_multi_param_task_splats_list_and_dict_values(self):
        r = _Recorder()
        r.run_tasks({"task_two": [1, 2]})
        self.assertIn(("task_two", 1, 2), r.calls)
        r = _Recorder()
        r.run_tasks({"task_two": {"a": 3, "b": 4}})
        self.assertIn(("task_two", 3, 4), r.calls)


if __name__ == "__main__":
    unittest.main()
