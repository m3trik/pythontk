# !/usr/bin/python
# coding=utf-8
"""Generic task/check pipeline primitive -- host- and Qt-free.

A reflection-based task runner: :class:`TaskFactory` discovers the ``task_*`` /
``check_*`` methods a subclass supplies (``getattr`` + :func:`inspect.signature`),
orders them by a declared ``TASK_ORDER``, runs each with LIFO set/revert state
management, and processes check results. It knows nothing about any DCC -- the
scene-exporter ``TaskManager`` in mayatk and blendertk each subclass it and
supply the host-specific task/check methods it discovers by name.

Like :mod:`pythontk.core_utils.app_handoff`, this is a *general* orchestration
base (no domain model or planner), so it lives in ``core_utils`` beside the other
shared infrastructure rather than in ``core_utils/engines/``. Formerly vendored
byte-identical in mayatk and blendertk; now the single source of truth.
"""

import contextlib
import time
from inspect import signature
from typing import Callable, Dict, Any, List, Optional, Tuple


class TaskFactory:
    """A factory class for managing and executing tasks in a scene export pipeline."""

    #: ``{check_name: (task_name, ...)}`` -- for each check, the tasks whose
    #: execution can change its verdict.  It is what lets :meth:`_schedule`
    #: hoist a check ABOVE the tasks it does not depend on, so a gate that was
    #: always going to fail fails BEFORE the expensive tasks below it have run
    #: (and, for a check that depends on nothing enabled, before the host is
    #: mutated at all).  A check with no entry here is treated as depending on
    #: every task -- the historical run-all-tasks-then-all-checks order -- so a
    #: subclass that declares nothing behaves exactly as it did.
    CHECK_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {}

    def __init__(self, logger):
        self.logger = logger
        self._method_cache = {}
        #: Restores registered by :meth:`stage_deferred_restore`, keyed so the
        #: first stager of a given state wins. Insertion-ordered; run LIFO.
        self._deferred_restores: Dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Deferred restores — the counterpart to the set_/revert_ pair
    # ------------------------------------------------------------------

    def stage_deferred_restore(self, key: str, restore: Callable) -> bool:
        """Register *restore* to run **after** the caller's real work — once per *key*.

        The ``set_<x>``/``revert_<x>`` pair runs its revert when
        :meth:`run_tasks` returns (see :meth:`_get_revert_method`), i.e. *before*
        the caller does whatever the tasks were preparing for. That is correct
        only for mutations the real work does not read. Anything the work reads
        as it runs — an exporter's working unit, its frame range, transient
        scene objects that must ship and then vanish — cannot use that pair
        without becoming inert, and stages here instead: the caller runs
        :meth:`run_deferred_restores` once the work is done (typically from a
        ``finally``, so every exit path is covered).

        Keying makes staging idempotent and gives the *first* stager priority,
        so a later task that builds on an already-staged mutation (widening a
        frame range another task set) still restores the true original.

        Parameters:
            key: Identity of the state being staged.
            restore: Zero-arg callable that puts the state back. Capture the
                original value in its closure — nothing is stored for it.

        Returns:
            True when this call registered the restore (i.e. it was first).
        """
        if key in self._deferred_restores:
            return False
        self._deferred_restores[key] = restore
        return True

    def stage_deferred_context(self, key: str, cm) -> bool:
        """Enter context manager *cm* now and stage its exit as the deferred restore.

        The bridge between scope-shaped cleanup and the deferred lifetime: a
        ``snapshot -> mutate -> restore`` primitive is written ONCE as a
        context manager (usable as a plain ``with`` by a script), and a task
        whose mutation the real work must still see hands the same object
        here — its ``__exit__`` then runs LIFO, isolated, from
        :meth:`run_deferred_restores` exactly like a hand-rolled restore.
        A ``contextlib.ExitStack`` composes several such scopes under one key.

        Same first-wins keying as :meth:`stage_deferred_restore`; when the key
        is already staged *cm* is NOT entered (nothing to undo).

        Returns:
            True when *cm* was entered and its exit staged.
        """
        if key in self._deferred_restores:
            return False
        cm.__enter__()
        return self.stage_deferred_restore(key, lambda: cm.__exit__(None, None, None))

    def run_deferred_restores(self) -> None:
        """Run + clear every restore staged by :meth:`stage_deferred_restore`.

        LIFO, matching :meth:`_revert_states`. Each restore is isolated — a
        failure is logged, never re-raised, since this normally runs from a
        ``finally`` where raising would mask the original error. The registry is
        cleared regardless, so a failed restore cannot make the next run treat
        its stale key as already staged.
        """
        restores, self._deferred_restores = self._deferred_restores, {}
        for key, restore in reversed(restores.items()):
            try:
                restore()
            except Exception as e:
                self.logger.warning(f"Deferred restore {key!r} failed: {e}")

    def _get_cached_method(self, method_name: str):
        """Get method with caching to avoid repeated getattr calls."""
        if method_name not in self._method_cache:
            self._method_cache[method_name] = getattr(self, method_name, None)
        return self._method_cache[method_name]

    @contextlib.contextmanager
    def _manage_context(
        self,
        tasks: Dict[str, Any],
        gate: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        """Manage task states by setting them once and reverting after, returning task results.

        Parameters:
            tasks: The run list, in execution order.  Tasks and checks may be
                interleaved (see :meth:`_schedule`); each entry is labelled by
                its ``check_`` prefix, so the log still numbers the two kinds
                separately.
            gate: ``gate(name, results_so_far) -> bool``, consulted before each
                entry.  Returning False skips it -- it never runs and never
                reaches *results*, so a skipped check is not a failed one.
                Used by :meth:`_execute_tasks_and_checks` to drop the work that
                a failed check has already made pointless.
        """
        original_states = {}
        task_results = {}

        # Pre-validate and cache all methods
        valid_tasks = {}
        for name, value in tasks.items():
            method = self._get_cached_method(name)
            if method:
                valid_tasks[name] = value
            else:
                self.logger.warning(f"Missing method: {name}. Skipping.")

        if not valid_tasks:
            yield {}
            return

        # self.logger.info(f"Running {len(valid_tasks)} tasks")

        # Revert in a finally: a task raising mid-loop, or an exception thrown
        # into the generator at the yield (from the with-body), must not leave
        # set_* state applied to the host scene.
        # Number tasks and checks in their own sequences: an interleaved run
        # list would otherwise report "Task #7/12" over a check.
        totals = {True: 0, False: 0}
        for name in valid_tasks:
            totals[name.startswith("check_")] += 1
        counters = {True: 0, False: 0}

        try:
            for task_name, value in valid_tasks.items():
                method = self._method_cache[task_name]  # Already cached
                is_check = task_name.startswith("check_")
                counters[is_check] += 1
                label = "Check" if is_check else "Task"
                index, total = counters[is_check], totals[is_check]

                # An unchecked toggle is not a task that ran. Reported before
                # the call, because the log is what a user (and anyone
                # diagnosing a deliverable from it) reads to know what the
                # export DID: "Executing apply_declared_takes" followed by
                # "Completed in 0.000s" over a task that was switched off cost
                # real time during a 2026-08-30 production investigation, since
                # it reads as a task that ran and did nothing rather than one
                # that never ran.
                #
                # Ahead of the gate deliberately: a task that was switched off
                # can change no verdict, so counting it among the work an abort
                # skipped would hold back checks that are still decidable.
                if self._task_is_disabled(method, value):
                    self.logger.info(
                        f"Skipping {label} #{index}/{total}: {task_name} (disabled)"
                    )
                    # The value the executor would have returned, so callers
                    # reading the results see no change in shape.
                    task_results[task_name] = True
                    # Deliberately NO revert staged: the task never ran, so
                    # there is no mutation to undo, and the `True` above is not
                    # a captured original state -- pairing them would hand
                    # ``revert_<x>(True)`` to a task that was switched off.
                    continue

                if gate is not None and not gate(task_name, task_results):
                    self.logger.info(
                        f"Skipping {label} #{index}/{total}: {task_name} "
                        "(a check has already failed)"
                    )
                    continue

                self.logger.info(f"Executing {label} #{index}/{total}: {task_name}")

                # Get revert method BEFORE executing the task
                revert_method = self._get_revert_method(task_name)

                try:
                    t0 = time.perf_counter()
                    result = self._execute_task_method(method, value)
                    elapsed = time.perf_counter() - t0
                    self.logger.success(f"  Completed {task_name} in {elapsed:.3f}s")
                    task_results[task_name] = result

                    # Store original state for reversion if this is a "set_" task
                    if revert_method and result is not None:
                        original_states[task_name] = {
                            "revert_method": revert_method,
                            "original_value": result,
                        }
                        self.logger.debug(
                            f"Stored original state for {task_name}: {result}"
                        )

                    # Handle check failures efficiently
                    if is_check and not self._is_success(result):
                        self._log_check_failed(
                            task_name, self._get_log_messages(result)
                        )

                except Exception as e:
                    self.logger.error(f"Error during task {task_name}: {e}")
                    raise

            yield task_results
        finally:
            self._revert_states(original_states)

    def _positional_count(self, method) -> int:
        """How many positional parameters *method* takes.

        The one piece of introspection both the dispatcher and the
        disabled-task predicate need; split out so they cannot disagree about
        which shape of task a falsy value switches OFF versus passes THROUGH.
        """
        return len(
            [
                p
                for p in signature(method).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
        )

    def _task_is_disabled(self, method, value: Any) -> bool:
        """Whether *value* switches this task OFF rather than parameterising it.

        Only a zero-argument task can be disabled: its value is a checkbox and
        nothing else. A task that TAKES a value is being handed one --
        ``0``/``""`` may be a legitimate argument, so a falsy value there is
        never read as "off" (:meth:`_execute_task_method` passes it through).
        """
        try:
            return self._positional_count(method) == 0 and not value
        except (TypeError, ValueError):
            # Unintrospectable callable (a builtin, a C extension): treat it as
            # enabled and let the executor deal with it -- the same thing that
            # happened before this predicate existed.
            return False

    def _execute_task_method(self, method, value: Any):
        """Execute a task method with proper parameter handling."""
        try:
            param_count = self._positional_count(method)

            if param_count == 0:
                # If the method takes no arguments, treat 'value' as a boolean flag
                if not value:
                    return True
                return method()
            elif param_count == 1:
                return method(value)
            else:
                # Handle methods that accept multiple parameters
                if isinstance(value, (list, tuple)):
                    return method(*value)
                elif isinstance(value, dict):
                    return method(**value)
                else:
                    return method(value)

        except TypeError as e:
            self.logger.error(f"Parameter mismatch for {method.__name__}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error executing task method {method.__name__}: {e}")
            raise

    def run_tasks(self, tasks: Dict[str, Any]) -> bool:
        """Run tasks and checks, returning True if all checks pass, False if any fail."""
        if not tasks:
            self.logger.notice("No tasks provided to run.")
            return True

        # Split tasks and checks
        tasks_only = {k: v for k, v in tasks.items() if not k.startswith("check_")}
        checks_only = {k: v for k, v in tasks.items() if k.startswith("check_")}

        return self._execute_tasks_and_checks(tasks_only, checks_only)

    def run_tasks_by_category(
        self, task_definitions: Dict[str, Any], check_definitions: Dict[str, Any]
    ) -> bool:
        """Alternative method to run tasks and checks separately with better organization."""
        return self._execute_tasks_and_checks(task_definitions, check_definitions)

    def _order_tasks(self, tasks: Dict[str, Any]) -> Dict[str, Any]:
        """Order tasks according to TASK_ORDER if defined, else alphabetically.

        Subclasses (e.g. TaskManager) can define a ``TASK_ORDER`` list to
        control execution sequence.  Tasks not in the list are appended
        alphabetically after the ordered ones.
        """
        explicit_order = getattr(self, "TASK_ORDER", None)
        if not explicit_order:
            return tasks if self._is_sorted(tasks) else dict(sorted(tasks.items()))

        ordered: Dict[str, Any] = {}
        for key in explicit_order:
            if key in tasks:
                ordered[key] = tasks[key]
        # Append any remaining tasks not in TASK_ORDER (alphabetical)
        for key in sorted(tasks.keys()):
            if key not in ordered:
                ordered[key] = tasks[key]
        return ordered

    def _schedule(
        self, ordered_tasks: Dict[str, Any], ordered_checks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge tasks and checks into ONE ordered run list.

        Tasks keep the sequence :meth:`_order_tasks` gave them, unchanged: that
        order is load-bearing (a subclass' ``TASK_ORDER`` encodes which task
        must see another's output) and no dependency graph describes it, so a
        task is never moved.  Each CHECK is hoisted instead, to the earliest
        point at which every task it declares in :attr:`CHECK_DEPENDENCIES` has
        already run -- and a check whose declared dependencies are all absent
        or switched off for this run is hoisted in front of the first task, so
        it is decided before the host is touched at all.

        The pay-off is what does NOT happen: a check that fails takes the rest
        of the run down with it (:meth:`_execute_tasks_and_checks`), and every
        task below it in the list is work the aborted write would have thrown
        away.  A check with no entry in the map is scheduled after every task,
        which is the order the runner had before this existed.
        """
        deps = getattr(self, "CHECK_DEPENDENCIES", None) or {}
        task_names = list(ordered_tasks)
        legacy_slot = len(task_names) - 1  # after every task

        # A task that will not actually run is not a barrier. Same two
        # predicates _manage_context filters on, so the scheduler and the
        # runner cannot disagree about which tasks exist this pass.
        slot_of = {}
        for index, name in enumerate(task_names):
            method = self._get_cached_method(name)
            if method is None or self._task_is_disabled(method, ordered_tasks[name]):
                continue
            slot_of[name] = index

        # slot -> the checks decided once the task at that index has run;
        # slot -1 holds those decided before the first task.
        buckets: Dict[int, List[Tuple[str, Any]]] = {}
        for name, value in ordered_checks.items():
            if name in deps:
                # Only dependencies that will actually RUN this pass count: a
                # task the caller left out, switched off, or has no method for
                # can change nothing, so it must not hold its dependents back.
                slots = [slot_of[d] for d in deps[name] if d in slot_of]
                slot = max(slots) if slots else -1
            else:
                slot = legacy_slot
            buckets.setdefault(slot, []).append((name, value))

        schedule: Dict[str, Any] = dict(buckets.get(-1, ()))
        for index, name in enumerate(task_names):
            schedule[name] = ordered_tasks[name]
            schedule.update(buckets.get(index, ()))
        return schedule

    def _execute_tasks_and_checks(
        self,
        tasks_only: Dict[str, Any],
        checks_only: Dict[str, Any],
    ) -> bool:
        """Execute tasks and checks with unified logic."""
        failed_checks = []
        all_checks_passed = True

        ordered_tasks = self._order_tasks(tasks_only) if tasks_only else {}
        ordered_checks = {}
        if checks_only:
            ordered_checks = (
                checks_only
                if self._is_sorted(checks_only)
                else dict(sorted(checks_only.items()))
            )

        skipped_tasks: List[str] = []
        skipped_checks: List[str] = []

        def gate(name: str, results: Dict[str, Any]) -> bool:
            """Whether *name* is still worth running.

            Everything runs until a check fails.  After that the run is over --
            its only consumer was a write that will not happen -- so every
            remaining TASK is dropped, and with it every check that needed one.
            Checks that were already decidable keep running: they cost nothing
            more, and reporting every failure the user can act on in one pass
            is what the old run-all-checks-last order gave for free.
            """
            if not any(
                k.startswith("check_") and not self._is_success(v)
                for k, v in results.items()
            ):
                return True
            if not name.startswith("check_"):
                skipped_tasks.append(name)
                return False
            if skipped_tasks:  # a task this check reads never ran
                skipped_checks.append(name)
                return False
            return True

        schedule = self._schedule(ordered_tasks, ordered_checks)
        if schedule:
            self.logger.info(
                f"Running {len(ordered_tasks)} export task(s) and "
                f"{len(ordered_checks)} validation check(s)..."
            )
            # ONE context over the merged list: the set_/revert_ pairing still
            # unwinds when run_tasks returns (its documented contract), and a
            # check hoisted above a set_ task must not see it already reverted.
            with self._manage_context(schedule, gate=gate) as results:
                all_checks_passed = self._process_check_results(
                    {k: v for k, v in results.items() if k.startswith("check_")},
                    failed_checks,
                )

        # Store counts so the caller (SceneExporter) can include them
        # in a single consolidated summary after the file is written.
        #
        # Counted by what DISPATCHED, not by what was requested: a name with no
        # method behind it is warn-skipped in _manage_context and never runs, so
        # including it made the exporters' "Checks Passed: N/N" banner report a
        # check that was never made.
        self._last_task_count = self._dispatchable_count(tasks_only)
        self._last_check_count = self._dispatchable_count(checks_only)
        # The NAMES behind the verdict, for a caller that wants to report (or
        # ask about) what failed rather than just that something did -- the
        # bool return says nothing a user can act on. Always rewritten, so a
        # passing run clears the previous run's list.
        self._last_failed_checks = list(failed_checks)
        # ...and the tasks the abort dropped, in their scheduled order. A
        # caller that decides to proceed ANYWAY (an exporter overriding the
        # verdict) must be able to run exactly these rather than the whole
        # list again -- the ones above them already ran, and re-running them
        # would repeat their mutation.
        self._last_skipped_tasks = list(skipped_tasks)

        if skipped_tasks or skipped_checks:
            # Name what the abort bought, and what it cost: an unexplained
            # gap between the requested task list and the log is the thing
            # that makes a user suspect the export silently misbehaved.
            lines = []
            if skipped_tasks:
                lines.append(f"Tasks not run: {', '.join(skipped_tasks)}")
            if skipped_checks:
                lines.append(
                    "Checks not evaluated (they read a task that was skipped): "
                    + ", ".join(skipped_checks)
                )
            self.logger.info(
                f"Stopped after the first failed check — skipped "
                f"{len(skipped_tasks)} task(s) and {len(skipped_checks)} "
                f"check(s). " + " ".join(lines)
            )

        self._log_execution_summary(
            failed_checks,
            all_checks_passed,
            self._last_task_count,
            self._last_check_count,
        )
        return all_checks_passed

    def _dispatchable_count(self, tasks: Dict[str, Any]) -> int:
        """How many of *tasks* have a method behind them.

        The same predicate ``_manage_context`` filters on, so a reported count
        can never exceed what actually ran.
        """
        return sum(1 for name in tasks if self._get_cached_method(name))

    def _is_sorted(self, d: Dict[str, Any]) -> bool:
        """Check if dictionary keys are already sorted."""
        keys = list(d.keys())
        return keys == sorted(keys)

    def _process_check_results(
        self,
        check_results: Dict[str, Any],
        failed_checks: list,
    ) -> bool:
        """Process check results and return overall success status.

        Per-check completion is already reported (at SUCCESS level) by the
        task runner's "Completed {name}" line, so passing checks need no
        extra log entry here — only failures are tracked.
        """
        all_checks_passed = True

        for check_name, result in check_results.items():
            if (
                isinstance(result, (tuple, list))
                and result
                and not isinstance(result[0], bool)
            ):
                # Malformed (bool, messages) shape — e.g. a bare non-empty
                # message list, whose truthy first string would silently PASS.
                # The verdict stays element 0's truthiness for backward
                # compatibility; just make the bad shape loud.
                self.logger.warning(
                    f"check '{check_name}' returned a malformed result "
                    f"(expected (bool, messages), got {type(result).__name__} "
                    f"with {type(result[0]).__name__} first element); treating "
                    f"first element's truthiness as the verdict"
                )
            if not self._is_success(result):
                failed_checks.append(check_name)
                all_checks_passed = False

        return all_checks_passed

    def _log_execution_summary(
        self,
        failed_checks: list,
        all_checks_passed: bool,
        tasks_count: int,
        checks_count: int,
    ) -> None:
        """Log the execution summary."""
        if not all_checks_passed:
            self.logger.log_box(
                "SUMMARY OF FAILED CHECKS",
                [f"- {check}" for check in failed_checks],
                level="ERROR",
            )
            self.logger.error("Export aborted due to failed checks.")
        else:
            self._log_checks_passed(tasks_count, checks_count, len(failed_checks))

    def _log_checks_passed(
        self, tasks_count: int, checks_count: int, failed_checks_count: int
    ) -> None:
        """Log a lightweight confirmation that tasks/checks finished.

        The real EXPORT SUCCESSFUL banner is emitted by SceneExporter
        after the file is written to disk.
        """
        parts = [f"{tasks_count} task(s)"]
        if checks_count > 0:
            parts.append(
                f"{checks_count - failed_checks_count}/{checks_count} check(s)"
            )
        self.logger.info(f"Pre-export pipeline complete: {', '.join(parts)} passed.")

    def _log_check_failed(self, task_name: str, log_messages: list):
        """Log the 'CHECK FAILED' box after task fails."""
        self.logger.log_box(f"CHECK FAILED: {task_name}", level="ERROR")
        for message in log_messages:
            self.logger.error(message)

    def _get_revert_method(self, task_name: str):
        """Get revert method for a task if it exists.

        Only ``set_<x>`` tasks pair with a ``revert_<x>``.  Note the timing:
        reverts run when ``run_tasks`` returns — BEFORE the actual export
        write — so only mutations the export itself doesn't depend on may be
        reverted this way. A task whose mutation the export *does* read must
        return ``None`` (which disarms this pairing) and register its restore
        with :meth:`stage_deferred_restore` instead.
        """
        if task_name.startswith("set_"):
            return getattr(self, f"revert_{task_name[4:]}", None)
        return None

    def _is_success(self, result) -> bool:
        """Check if a task result indicates success."""
        if isinstance(result, (tuple, list)):
            return bool(result[0]) if result else False
        return bool(result)

    def _get_log_messages(self, result) -> list:
        """Extract log messages from a task result."""
        return (
            result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else []
        )

    def _revert_states(self, original_states: Dict[str, Any]) -> None:
        """Revert all stored states."""
        if not original_states:
            self.logger.debug("No states to revert.")
            return

        self.logger.info("Reverting temporary states...")

        # Revert in reverse order (LIFO)
        for task_name, state_info in reversed(original_states.items()):
            revert_method = state_info["revert_method"]
            original_value = state_info["original_value"]

            try:
                revert_method(original_value)
                self.logger.debug(f"Reverted {task_name} to: {original_value}")
            except Exception as e:
                self.logger.error(f"Error reverting {task_name}: {e}")

        self.logger.info("State reversion completed.")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
