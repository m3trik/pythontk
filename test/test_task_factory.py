# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.task_factory (TaskFactory) — the generic,
host-free task/check pipeline primitive shared by mayatk's and blendertk's
scene exporters (their ``TaskManager`` subclasses it). The dispatch / ordering /
deferred-restore / check contract is pinned here once, DCC-free; the DCC suites
cover the concrete ``task_*`` / ``check_*`` methods each supplies.
"""

import unittest
from unittest.mock import MagicMock

from pythontk import TaskFactory, OperationCancelled


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
        """A ``set_`` task is an ordinary task: its result is just its result."""
        self.calls.append(("set_flag", value))
        return "ORIGINAL"

    def revert_flag(self, original):
        """The retired pairing's other half -- never called by the runner."""
        self.calls.append(("revert_flag", original))

    def check_ok(self, value):
        self.calls.append(("check_ok", value))
        return True

    def check_bad(self, value):
        self.calls.append(("check_bad", value))
        return (False, ["reason"])

    def check_empty(self, value):
        self.calls.append(("check_empty", value))
        return []  # a natural "no messages" shape -- must not crash

    def check_none(self, value):
        self.calls.append(("check_none", value))
        return None  # no verdict at all -- must fail closed

    def check_bare_messages(self, value):
        self.calls.append(("check_bare_messages", value))
        return ["problem: verdict bool missing"]  # malformed (bool, messages)

    def check_boom(self, value):
        self.calls.append(("check_boom", value))
        raise RuntimeError("check boom")

    def set_switch(self):
        """A zero-arg ``set_`` task: its value is a CHECKBOX, not an argument."""
        self.calls.append(("set_switch",))
        return "ORIGINAL_SWITCH"

    def set_deferred(self, value):
        """A task whose mutation the real work READS: it must survive run_tasks."""
        self.calls.append(("set_deferred", value))
        self.stage_deferred_restore(
            "deferred", lambda: self.calls.append(("restore_deferred",))
        )


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

    def test_tasks_the_abort_dropped_are_recorded_for_the_caller(self):
        """A failed check stops the runner dispatching the tasks below it. A
        caller that then decides to proceed anyway needs to run exactly those
        -- the ones above already ran, and repeating them repeats their
        mutation -- so the names are recorded alongside the verdict.
        Added: 2026-09-03
        """

        class _Gated(_Recorder):
            TASK_ORDER = ["task_plain", "task_noargs"]
            CHECK_DEPENDENCIES = {"check_bad": ("task_plain",)}

        r = _Gated()
        self.assertFalse(
            r.run_tasks({"task_plain": "x", "task_noargs": True, "check_bad": True})
        )
        # check_bad is hoisted to run right after its only dependency, so the
        # task below it never dispatches.
        self.assertNotIn("task_noargs", [c[0] for c in r.calls])
        self.assertEqual(r._last_skipped_tasks, ["task_noargs"])

    def test_checks_the_abort_dropped_are_recorded_for_the_caller(self):
        """A check that reads a dropped task is dropped with it. A caller that
        proceeds anyway must not report it as passed -- it never ran -- so its
        name is recorded beside the skipped tasks, and a clean run clears it.
        Added: 2026-09-12
        """

        class _Gated(_Recorder):
            TASK_ORDER = ["task_plain", "task_noargs"]
            CHECK_DEPENDENCIES = {
                "check_bad": ("task_plain",),
                "check_ok": ("task_noargs",),
            }

        r = _Gated()
        self.assertFalse(
            r.run_tasks(
                {
                    "task_plain": "x",
                    "task_noargs": True,
                    "check_bad": True,
                    "check_ok": True,
                }
            )
        )
        self.assertNotIn("check_ok", [c[0] for c in r.calls])
        self.assertEqual(r._last_skipped_checks, ["check_ok"])
        self.assertTrue(r.run_tasks({"task_plain": "x", "check_ok": True}))
        self.assertEqual(r._last_skipped_checks, [])

    def test_failed_check_names_are_recorded_for_the_caller(self):
        """The bool verdict names nothing a caller can act on, so a consumer
        that wants to report (or ask about) the failure had to re-derive it
        from the log. ``_last_failed_checks`` carries the names, and a passing
        run clears the previous run's list.
        Added: 2026-09-03
        """
        r = _Recorder()
        self.assertFalse(r.run_tasks({"check_bad": True, "check_ok": True}))
        self.assertEqual(r._last_failed_checks, ["check_bad"])
        self.assertTrue(r.run_tasks({"check_ok": True}))
        self.assertEqual(r._last_failed_checks, [])

    def test_what_a_run_keeps_is_named_once_and_closed_with_the_run(self):
        """A task whose edit no restore unwinds -- a repair, a write-back edit --
        records it where it makes it, so a run that stops before its write can
        name exactly what it left in the host. A hardcoded list said key
        snapping and tying remained after a blocked export whose key edits had
        all been restored. Each edit is named once, in the order it was made,
        and ``run_deferred_restores`` -- which closes the run -- clears the
        record along with the restores.
        Added: 2026-09-15
        """

        class _Keeper(_Recorder):
            def task_merge(self, value):
                self.record_kept_edit("merged duplicate materials")

            def task_repair(self, value):
                self.record_kept_edit("repaired node and shape names")
                self.record_kept_edit("repaired node and shape names")

        r = _Keeper()
        self.assertFalse(
            r.run_tasks({"task_repair": 1, "task_merge": 1, "check_bad": True})
        )
        self.assertEqual(
            r.kept_edits,
            ["merged duplicate materials", "repaired node and shape names"],
        )
        r.kept_edits.append("not a record")  # a copy: a reader cannot edit it
        self.assertEqual(len(r.kept_edits), 2)
        r.run_deferred_restores()
        self.assertEqual(r.kept_edits, [])

    def test_a_set_task_has_no_revert_pairing(self):
        """``set_<x>`` / ``revert_<x>`` was retired (2026-09-13): the revert ran
        when ``run_tasks`` returned, BEFORE the work the tasks prepared for, so
        every shipped ``set_`` task had to disarm it. A ``revert_`` method is
        just a method now; the deferred registry is the one undo mechanism."""
        r = _Recorder()
        r.run_tasks({"set_flag": True})
        self.assertIn(("set_flag", True), r.calls)
        self.assertNotIn(("revert_flag", "ORIGINAL"), r.calls)
        self.assertFalse([c for c in r.calls if c[0].startswith("revert_")])

    def test_missing_method_is_skipped_not_crashed(self):
        # An unknown task name is warned + skipped; with no failing checks -> True.
        self.assertTrue(_Recorder().run_tasks({"task_absent": True}))

    def test_a_staged_restore_survives_a_later_task_raising(self):
        # set_deferred runs first (alphabetical), then task_boom raises: the
        # staged restore must still be there for the caller's ``finally`` --
        # or a failed export leaves the host scene permanently mutated.
        r = _Recorder()
        with self.assertRaises(RuntimeError):
            r.run_tasks({"set_deferred": True, "task_boom": True})
        self.assertNotIn(("restore_deferred",), r.calls)
        r.run_deferred_restores()
        self.assertIn(("restore_deferred",), r.calls)

    def test_empty_sequence_check_result_is_failure_not_crash(self):
        # An empty tuple/list result is falsy -> failed check, never IndexError.
        self.assertFalse(_Recorder().run_tasks({"check_empty": True}))

    def test_none_check_result_fails_closed(self):
        # A check that returns no verdict at all must count as a failure.
        self.assertFalse(_Recorder().run_tasks({"check_none": True}))

    def test_raising_check_aborts_the_run_but_cleanup_still_happens(self):
        # Fail-closed by design: a check that RAISES propagates (aborting the
        # whole run) rather than being recorded as a failed check -- pinned so
        # a refactor can't silently soften it. Cleanup must still hold: the
        # staged deferred restores survive for the caller's finally.
        r = _Recorder()
        with self.assertRaises(RuntimeError):
            r.run_tasks({"set_flag": True, "set_deferred": True, "check_boom": True})
        self.assertIn("deferred", r._deferred_restores)
        r.run_deferred_restores()  # the caller's finally
        self.assertIn(("restore_deferred",), r.calls)

    def test_bare_message_list_check_passes_but_warns_malformed(self):
        # Hazard pin: a check that forgets the bool verdict and returns a bare
        # non-empty message list PASSES (element 0 is a truthy string). That
        # verdict is kept for backward compatibility, but the malformed shape
        # must be loud -- a WARNING naming the check.
        r = _Recorder()
        self.assertTrue(r.run_tasks({"check_bare_messages": True}))
        warnings = [str(c.args[0]) for c in r.logger.warning.call_args_list]
        self.assertTrue(
            any("check_bare_messages" in w and "malformed" in w for w in warnings),
            f"expected a malformed-shape warning, got: {warnings}",
        )

    def test_well_formed_check_results_do_not_warn_malformed(self):
        # (bool, messages) tuples -- pass or fail -- must stay warning-free.
        r = _Recorder()
        r.run_tasks({"check_ok": True, "check_bad": True})
        warnings = [str(c.args[0]) for c in r.logger.warning.call_args_list]
        self.assertFalse([w for w in warnings if "malformed" in w], warnings)

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

    def test_a_disabled_task_is_reported_as_skipped_not_executed(self):
        """The log is what a user reads to know what an export DID.

        A switched-off task announced as "Executing … / Completed in 0.000s"
        reads as a task that ran and found nothing to do -- which is a
        different diagnosis from one that never ran, and it cost real time
        during a 2026-08-30 production investigation.
        """
        r = _Recorder()
        r.run_tasks({"task_noargs": False})
        lines = [str(c.args[0]) for c in r.logger.info.call_args_list if c.args]
        self.assertTrue(
            any("Skipping" in line and "task_noargs" in line for line in lines), lines
        )
        self.assertFalse(
            any("Executing" in line and "task_noargs" in line for line in lines), lines
        )

    def test_a_zero_arg_set_task_is_a_checkbox_like_any_other(self):
        """Its value switches it off; enabled, it runs once and that is all."""
        r = _Recorder()
        r.run_tasks({"set_switch": False})
        self.assertNotIn(("set_switch",), r.calls, "a disabled task must not run")

        r = _Recorder()
        r.run_tasks({"set_switch": True})
        self.assertEqual(
            [c for c in r.calls if c[0] == "set_switch"], [("set_switch",)]
        )

    def test_multi_param_task_splats_list_and_dict_values(self):
        r = _Recorder()
        r.run_tasks({"task_two": [1, 2]})
        self.assertIn(("task_two", 1, 2), r.calls)
        r = _Recorder()
        r.run_tasks({"task_two": {"a": 3, "b": 4}})
        self.assertIn(("task_two", 3, 4), r.calls)

    # -- deferred restores: the counterpart to the too-early set_/revert_ pair --

    def test_deferred_restore_survives_run_tasks(self):
        """The whole point: a mutation the caller's real work READS must still
        be in place when run_tasks returns, unlike a revert_-paired one."""
        r = _Recorder()
        r.run_tasks({"set_deferred": True})
        self.assertNotIn(("restore_deferred",), r.calls)
        self.assertIn("deferred", r._deferred_restores)

        r.run_deferred_restores()
        self.assertIn(("restore_deferred",), r.calls)
        self.assertFalse(r._deferred_restores)

    def test_deferred_restore_is_first_wins_per_key(self):
        """A later stager of the same key must not displace the first
        capture — that is what lets one task build on another's mutation."""
        r = _Recorder()
        self.assertTrue(
            r.stage_deferred_restore("k", lambda: r.calls.append(("first",)))
        )
        self.assertFalse(
            r.stage_deferred_restore("k", lambda: r.calls.append(("second",)))
        )
        r.run_deferred_restores()
        self.assertEqual(
            [c for c in r.calls if c[0] in ("first", "second")], [("first",)]
        )

    def test_deferred_restores_run_lifo_and_isolate_failures(self):
        r = _Recorder()
        r.stage_deferred_restore("a", lambda: r.calls.append(("restore_a",)))
        r.stage_deferred_restore("boom", lambda: 1 / 0)
        r.stage_deferred_restore("b", lambda: r.calls.append(("restore_b",)))

        r.run_deferred_restores()  # must not raise — it runs from a finally
        self.assertEqual(
            [c[0] for c in r.calls if c[0].startswith("restore_")],
            ["restore_b", "restore_a"],
        )
        # Cleared even though one restore raised, so the next run re-stages.
        self.assertFalse(r._deferred_restores)

    def test_deferred_context_enters_now_and_exits_at_restore_time(self):
        """A context manager staged here is the SAME primitive a script would
        `with`: entered immediately, exited LIFO from run_deferred_restores."""
        import contextlib

        r = _Recorder()

        @contextlib.contextmanager
        def scope(tag):
            r.calls.append((f"enter_{tag}",))
            try:
                yield
            finally:
                r.calls.append((f"exit_{tag}",))

        self.assertTrue(r.stage_deferred_context("a", scope("a")))
        self.assertTrue(r.stage_deferred_context("b", scope("b")))
        self.assertEqual([c[0] for c in r.calls], ["enter_a", "enter_b"])
        r.run_deferred_restores()
        self.assertEqual(
            [c[0] for c in r.calls], ["enter_a", "enter_b", "exit_b", "exit_a"]
        )

    def test_deferred_context_first_wins_and_does_not_enter_a_loser(self):
        import contextlib

        r = _Recorder()

        @contextlib.contextmanager
        def scope(tag):
            r.calls.append((f"enter_{tag}",))
            yield

        self.assertTrue(r.stage_deferred_context("k", scope("first")))
        self.assertFalse(r.stage_deferred_context("k", scope("second")))
        self.assertEqual([c[0] for c in r.calls], ["enter_first"])

    def test_deferred_context_composes_via_exit_stack(self):
        """Several scopes under ONE key (the exporter's snapshot + temp-dir pair)."""
        import contextlib

        r = _Recorder()
        stack = contextlib.ExitStack()
        stack.callback(lambda: r.calls.append(("cleanup_2",)))
        stack.callback(lambda: r.calls.append(("cleanup_1",)))
        r.stage_deferred_context("pair", stack)
        r.run_deferred_restores()
        self.assertEqual([c[0] for c in r.calls], ["cleanup_1", "cleanup_2"])


class CheckCountTest(unittest.TestCase):
    """The counts the DCC exporters print as "Checks Passed: N/N"."""

    def test_counts_exclude_a_check_whose_method_does_not_exist(self):
        """A check with no method is warn-skipped and never runs, so counting
        it makes the banner claim a check passed that was never made."""
        r = _Recorder()
        self.assertTrue(
            r.run_tasks({"check_ok": True, "check_absent": True, "task_plain": 1})
        )
        self.assertEqual(r._last_check_count, 1)
        self.assertEqual(r._last_task_count, 1)

    def test_counts_exclude_a_task_whose_method_does_not_exist(self):
        r = _Recorder()
        r.run_tasks({"check_ok": True, "task_plain": 1, "task_absent": 1})
        self.assertEqual(r._last_task_count, 1)


class _Scheduled(_Recorder):
    """A factory whose checks declare which tasks can change their verdict."""

    TASK_ORDER = ["task_cheap", "task_costly"]

    CHECK_DEPENDENCIES = {
        "check_ok": (),  # nothing the pipeline does can change it
        "check_bad": ("task_cheap",),  # decided as soon as the cheap task ran
        "check_none": ("task_costly",),  # needs the whole pipeline
    }

    def task_cheap(self, value):
        self.calls.append(("task_cheap", value))

    def task_costly(self, value):
        self.calls.append(("task_costly", value))


class CheckSchedulingTest(unittest.TestCase):
    """``CHECK_DEPENDENCIES`` hoists each check above the tasks it cannot read.

    The point is the work that does NOT happen: a gate that was always going
    to fail must fail before the expensive tasks below it have run, instead of
    after a full pipeline whose output the aborted write throws away.
    """

    def test_an_undeclared_check_still_runs_after_every_task(self):
        # The default (no CHECK_DEPENDENCIES) is the historical order --
        # subclasses that declare nothing must behave exactly as before.
        r = _Recorder()
        r.run_tasks({"task_plain": 1, "check_ok": True})
        self.assertEqual([c[0] for c in r.calls], ["task_plain", "check_ok"])

    def test_a_check_with_no_dependencies_runs_before_the_first_task(self):
        r = _Scheduled()
        r.run_tasks({"task_cheap": 1, "task_costly": 2, "check_ok": True})
        self.assertEqual(
            [c[0] for c in r.calls], ["check_ok", "task_cheap", "task_costly"]
        )

    def test_a_check_runs_directly_after_its_last_dependency(self):
        r = _Scheduled()
        r.run_tasks({"task_cheap": 1, "task_costly": 2, "check_none": True})
        self.assertEqual(
            [c[0] for c in r.calls], ["task_cheap", "task_costly", "check_none"]
        )

    def test_a_failed_check_skips_the_tasks_below_it(self):
        # check_bad reads task_cheap only, so it is decided before task_costly
        # runs -- and task_costly must then never run at all.
        r = _Scheduled()
        self.assertFalse(
            r.run_tasks({"task_cheap": 1, "task_costly": 2, "check_bad": True})
        )
        self.assertEqual([c[0] for c in r.calls], ["task_cheap", "check_bad"])

    def test_a_dependency_that_is_not_running_does_not_hold_a_check_back(self):
        # task_costly was not requested this run, so a check that reads it is
        # decidable immediately -- an absent task can change nothing, and
        # waiting on one would forfeit the early abort for no verdict change.
        class _Absent(_Scheduled):
            CHECK_DEPENDENCIES = {"check_ok": ("task_costly",)}

        r = _Absent()
        r.run_tasks({"task_cheap": 1, "check_ok": True})
        self.assertEqual([c[0] for c in r.calls], ["check_ok", "task_cheap"])

    def test_a_switched_off_dependency_is_not_a_barrier(self):
        # A task present in the payload but toggled OFF never runs, so a check
        # that reads it is decidable immediately -- the scheduler and
        # _manage_context must agree on which tasks exist this pass.
        class _Off(_Scheduled):
            CHECK_DEPENDENCIES = {"check_ok": ("task_noargs",)}
            TASK_ORDER = ["task_noargs", "task_costly"]

        r = _Off()
        r.run_tasks({"task_noargs": False, "task_costly": 1, "check_ok": True})
        names = [c[0] for c in r.calls]
        self.assertEqual(names, ["check_ok", "task_costly"])

    def test_checks_already_decidable_still_report_after_one_fails(self):
        # One pass must report every failure the user can act on: a check that
        # needs no further task is free to run even after check_bad failed.
        class _Pair(_Scheduled):
            CHECK_DEPENDENCIES = {
                "check_bad": (),
                "check_empty": (),
                "check_none": ("task_costly",),
            }

        r = _Pair()
        self.assertFalse(
            r.run_tasks(
                {
                    "task_costly": 1,
                    "check_bad": True,
                    "check_empty": True,
                    "check_none": True,
                }
            )
        )
        names = [c[0] for c in r.calls]
        self.assertIn("check_bad", names)
        self.assertIn("check_empty", names)  # decidable -> still reported
        self.assertNotIn("task_costly", names)  # pointless -> skipped
        self.assertNotIn("check_none", names)  # needs the skipped task

    def test_a_hoisted_check_runs_before_the_task_it_does_not_read(self):
        class _WithSet(_Recorder):
            TASK_ORDER = ["set_flag"]
            CHECK_DEPENDENCIES = {"check_ok": ()}

        r = _WithSet()
        r.run_tasks({"set_flag": True, "check_ok": True})
        names = [c[0] for c in r.calls]
        self.assertEqual(names, ["check_ok", "set_flag"])

    def test_scheduling_survives_a_run_with_no_tasks_at_all(self):
        r = _Scheduled()
        self.assertFalse(r.run_tasks({"check_ok": True, "check_bad": True}))
        self.assertEqual([c[0] for c in r.calls], ["check_bad", "check_ok"])


class ProgressReportingTest(unittest.TestCase):
    """``progress_callback(current, total, message)`` -- the per-entry stream a
    consumer's progress bar is driven by, and its cancel contract. Added:
    2026-09-04.
    """

    def _streamed(self):
        r = _Recorder()
        events = []
        r.progress_callback = lambda current, total, message: events.append(
            (current, total, message)
        )
        return r, events

    def test_every_entry_is_reported_before_it_runs_and_the_list_is_closed(self):
        r, events = self._streamed()
        r.run_tasks({"task_plain": 1, "check_ok": 2})
        self.assertEqual(
            events,
            [
                (0, 2, "Task 1/1: task_plain"),
                (1, 2, "Check 1/1: check_ok"),
                (2, 2, None),
            ],
        )

    def test_a_disabled_entry_counts_as_done_but_is_not_reported(self):
        r, events = self._streamed()
        r.run_tasks({"task_noargs": False, "task_plain": 1})
        self.assertEqual(events, [(1, 2, "Task 2/2: task_plain"), (2, 2, None)])

    def test_the_entries_a_failed_check_drops_are_closed_as_done(self):
        r, events = self._streamed()
        # check_bad is undeclared, so it runs after every task -- nothing is
        # dropped here; the close still reports the whole list as done.
        self.assertFalse(r.run_tasks({"task_plain": 1, "check_bad": 2}))
        self.assertEqual(events[-1], (2, 2, None))

    def test_false_from_the_callback_cancels_before_the_next_entry(self):
        r = _Recorder()
        r.progress_callback = lambda current, total, message: (
            not (message or "").endswith("task_boom")
        )
        with self.assertRaises(OperationCancelled):
            r.run_tasks({"set_flag": 1, "task_boom": 2})
        names = [c[0] for c in r.calls]
        self.assertIn("set_flag", names)
        self.assertNotIn("task_boom", names, "the refused entry must not run")

    def test_none_from_the_callback_is_not_a_cancel(self):
        r = _Recorder()
        r.progress_callback = lambda *args: None
        self.assertTrue(r.run_tasks({"task_plain": 1}))
        self.assertIn(("task_plain", 1), r.calls)

    def test_a_raising_callback_never_fails_the_run(self):
        r = _Recorder()

        def broken(*args):
            raise ValueError("feedback bug")

        r.progress_callback = broken
        self.assertTrue(r.run_tasks({"task_plain": 1}))
        self.assertIn(("task_plain", 1), r.calls)

    def test_a_text_only_tick_forwards_none_counts(self):
        r, events = self._streamed()
        r._report_progress(None, None, "converting...")
        self.assertEqual(events, [(None, None, "converting...")])

    def test_no_callback_is_a_no_op(self):
        r = _Recorder()
        r._report_progress(0, 1, "anything")  # must not raise
        self.assertTrue(r.run_tasks({"task_plain": 1}))


if __name__ == "__main__":
    unittest.main()
