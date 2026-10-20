# !/usr/bin/python
# coding=utf-8
"""The Scene Exporter's orchestration shell, written once for both DCCs.

mayatk's and blendertk's ``SceneExporter`` subclass :class:`SceneExporterBase`
and keep only what reaches their scene: the export write itself
(``perform_export``), the object scope, the FBX/USD options and presets, and
the naming tokens a scene spells.  What they share is here: the logging
setup, the run's ``(current, total, message)`` progress stream, the task
pipeline run with its per-check override decision, the export button's
contract (:meth:`SceneExporterBase.run_config_from_values` over
:class:`~pythontk.ExportProfile`), the export folder, and the run log -- one
file per scene in the per-user cache (:class:`~pythontk.RunLog`).

Hooks (override in the DCC subclass; defaults describe no scene):

=====================================  ======================================
``TASK_MANAGER_CLASS``                 the host's ``TaskManager`` (a
                                       :class:`~pythontk.TaskFactory`), built
                                       with the exporter's logger
``_saved_scene_path()``                the open scene's saved path, ``""``
                                       unsaved
``KEPT_EDITS_ADVICE``                  what a run that stopped before its
                                       write tells the user about the edits
                                       it kept
``confirm(question)``                  consent for an export-time side effect
                                       (the panel overrides it with a dialog)
``decide_check_failure(check, ...)``   override this failed check, every one
                                       from here, or stop (the panel overrides
                                       it with a three-button dialog)
=====================================  ======================================
"""

import html
import logging
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pythontk as ptk
from pythontk.core_utils.logging_mixin import LoggingMixin, RunLog


class SceneExporterBase(LoggingMixin):
    """The DCC-free half of a Scene Exporter: progress, consent, run config.

    Subclass per host and set :attr:`TASK_MANAGER_CLASS`; see the module
    docstring for the hooks.  The ``export_dir`` / ``export_path`` attributes
    the run-log helpers read are stamped by the host's ``perform_export``.
    """

    #: The host's task/check pipeline class (a ``ptk.TaskFactory`` subclass),
    #: constructed as ``TASK_MANAGER_CLASS(logger)``; set by the DCC subclass.
    TASK_MANAGER_CLASS: Any = None

    #: Said after the list of edits a run kept when it stopped before its
    #: write -- the host words its own undo story.
    KEPT_EDITS_ADVICE = "Revert to the saved file if that is not what you want."

    def __init__(
        self, log_level: str = "WARNING", log_handler: Optional[object] = None
    ):
        self._setup_logging(log_level, log_handler)

        self.task_manager = self.TASK_MANAGER_CLASS(self.logger)
        #: Checks a run failed but the user chose to override at the failure
        #: point (see decide_check_failure). Re-stamped by every
        #: ``perform_export``, so the success banner reports the deliverable
        #: as shipped-with-failures rather than claiming a clean pass.
        self._overridden_checks: List[str] = []
        #: The ``(current, total, message)`` stream of the run in flight and
        #: its bookkeeping -- see :meth:`_progress_begin`; cleared when
        #: ``perform_export`` returns.
        self._progress_callback: Optional[Callable] = None
        self._progress_current = 0
        self._progress_total = 0
        self._progress_base = 0
        self._progress_open = False
        self._progress_cancellable = False
        self._progress_cancel_ignored = False
        #: Whether the last run stopped on a cancel (vs. any other abort) --
        #: what the panel's footer reports after the run.
        self._export_cancelled = False
        self.logger.debug("Task manager initialized in SceneExporter.")

    def _setup_logging(
        self, log_level: Optional[str], log_handler: Optional[object]
    ) -> None:
        """Apply a log level and/or handler; ``None`` leaves the level as it is.

        ``perform_export`` calls this with its own ``log_level`` argument, so
        a level there would silently override the one the constructor set --
        ``SceneExporter(log_level="DEBUG").perform_export(...)`` used to run
        at WARNING and drop every per-task line the caller had asked for.
        """
        if log_level is not None:
            self.logger.setLevel(log_level)
        if log_handler:
            self.logger.addHandler(log_handler)

    def _setup_file_logging(self) -> None:
        """Open this scene's run log for the run in flight.

        Appends to :meth:`log_file_path` -- the SCENE's log, whatever the
        Output Filename resolves to -- after trimming it to its newest runs and
        sweeping the folder's stale logs (:class:`~pythontk.RunLog`), and first
        clears the per-export logs earlier versions left beside the
        deliverables in this export folder (:meth:`_remove_legacy_logs`).
        The host's ``perform_export`` calls it once the export path is known,
        and :meth:`_close_file_logging` on every exit. A log that cannot open
        (an unwritable cache, a full disk, a header that cannot encode) is
        warned about, never the reason an export stops: the panel shows the
        run either way.
        """
        self._remove_legacy_logs()
        path = self.log_file_path()
        RunLog.sweep(os.path.dirname(path), keep=[path])
        RunLog.trim(path)
        scene = self._saved_scene_path() or "(unsaved)"
        # The files the run's format ships (the run is adopted before the log
        # opens): a GLB-only export's path names its temp FBX, not the GLB.
        output_format = getattr(
            getattr(self.task_manager, "run", None), "output_format", None
        )
        stem = os.path.splitext(self.export_path)[0]
        shipped = self.OUTPUT_EXTENSIONS.get(output_format) or ()
        output = ", ".join(stem + ext for ext in shipped) or self.export_path
        try:
            self.setup_file_logging(
                path, header=f"Scene Exporter | scene: {scene} | output: {output}"
            )
        except (OSError, ValueError) as error:  # UnicodeError is a ValueError
            self.logger.warning(f"The run log could not be opened ({error}): {path}")

    def _saved_scene_path(self) -> str:
        """Scene hook: the open scene's saved path, ``""`` while it is unsaved
        (the default: no scene)."""
        return ""

    def confirm(self, question: str) -> bool:
        """Yes/no consent for an export-time side effect (a tool download).

        The seam the panel overrides with a dialog. Headless it asks on the
        console when there is one -- an interactive headless session gets a
        ``[y/N]`` -- and answers no otherwise: nobody is there to consent, and
        the caller's own message names the manual install.

        Parameters:
            question: Plain-text question; newlines allowed.
        """
        return bool(ptk.AppInstaller.consent(True, question))

    #: The check-failure dialog's buttons, in order: label -> the answer
    #: :attr:`pythontk.TaskFactory.failed_check_handler` takes.
    CHECK_FAILURE_CHOICES: Dict[str, str] = {
        "Override All": ptk.TaskFactory.CHECK_OVERRIDE_ALL,
        "Override": ptk.TaskFactory.CHECK_OVERRIDE,
        "Cancel": ptk.TaskFactory.CHECK_ABORT,
    }
    #: How much of a failure the dialog shows before deferring to the log.
    CHECK_FAILURE_MAX_MESSAGES = 6
    CHECK_FAILURE_MAX_REMAINING = 12

    def decide_check_failure(
        self, check: str, messages: List[str], remaining: List[str]
    ) -> str:
        """Seam: what to do about *check*, which just failed mid-run.

        Asked at the failure point, while the tasks that ran are still staged:
        overriding costs nothing here, where re-running with the panel's
        Override Checks armed repeats every task over a scene the first run
        already mutated. Answers one of ``ptk.TaskFactory.CHECK_*``: carry on
        past this check (a later failure asks again), past every check from
        here on, or stop before anything is written.

        The panel overrides this with a three-button dialog over
        :meth:`check_failure_html`. Headless it routes through
        :meth:`confirm` -- a console ``[y/N]`` for this one check, and no when
        nobody is there to ask, so a batch run still stops on a failed check.

        Parameters:
            check: The failed check's name.
            messages: What the check reported.
            remaining: The checks still to run, in order.
        """
        shown = messages[: self.CHECK_FAILURE_MAX_MESSAGES]
        lines = [f"Check failed: {self.check_label(check)}."]
        lines += [f"  {m}" for m in shown]
        if len(messages) > len(shown):
            lines.append(f"  ... {len(messages) - len(shown)} more in the log.")
        if remaining:
            lines.append(f"{len(remaining)} check(s) still to run.")
        lines.append("\nOverride this check and continue the export?")
        if self.confirm("\n".join(lines)):
            return ptk.TaskFactory.CHECK_OVERRIDE
        return ptk.TaskFactory.CHECK_ABORT

    def check_label(self, check: str) -> str:
        """*check*'s name as the panel spells it -- its row's label.

        A check with no row (a headless caller's own) reads as its name made
        legible: ``check_path_length`` -> "Path Length".
        """
        definition = self._definition_tables()[1].get(check) or {}
        label = definition.get("setText") or definition.get("set_row_label")
        if label:
            return str(label)
        name = check[len("check_") :] if check.startswith("check_") else check
        return name.replace("_", " ").strip().title()

    def check_failure_html(
        self, check: str, messages: List[str], remaining: List[str]
    ) -> str:
        """The check-failure dialog's body: the failure, then what is left.

        Rich text for the panel's message box: the failed check and what it
        reported (capped at :attr:`CHECK_FAILURE_MAX_MESSAGES`, the rest
        deferred to the log), the checks still to run (capped at
        :attr:`CHECK_FAILURE_MAX_REMAINING`), any this run already overrode,
        and what each button does. Every piece of text is escaped -- check
        messages carry paths and node names, never markup. Built without
        newlines: the box's rich-text pipeline turns each one into a break.
        """
        error = self.LOG_COLORS.get("ERROR", "#FFCCCC")
        muted = self.LOG_COLORS.get("DEBUG", "#AAAAAA")

        def esc(text) -> str:
            # Whitespace collapsed: a newline inside a label or message is a
            # stray break in the box (the pipeline renders each one).
            return html.escape(" ".join(str(text).split()))

        def block(lines, *, small=True, color="", gap=0) -> str:
            # One block per section, each line inside its own font: a break
            # left OUTSIDE a smaller font is laid out at the box's larger base
            # size, which opened a gap above every section's last line.
            tint = f" color='{color}'" if color else ""
            size = " size='3'" if small else ""
            margin = f" style='margin-top:{gap}px'" if gap else ""
            body = "<br>".join(lines)
            return f"<div{margin}><font{size}{tint}>{body}</font></div>"

        parts = [
            block(
                [
                    f"<b><font color='{error}'>Check failed:</font></b> "
                    f"<hl>{esc(self.check_label(check))}</hl>"
                ],
                small=False,
            )
        ]
        reported = [m for m in messages if str(m).strip()]
        shown = reported[: self.CHECK_FAILURE_MAX_MESSAGES]
        if shown:
            lines = [f"&nbsp;&nbsp;{esc(m)}" for m in shown]
            if len(reported) > len(shown):
                more = len(reported) - len(shown)
                lines.append(
                    f"<font color='{muted}'>&nbsp;&nbsp;&hellip; {more} more in "
                    "the log</font>"
                )
            parts.append(block(lines, color=error))

        if remaining:
            listed = remaining[: self.CHECK_FAILURE_MAX_REMAINING]
            lines = [f"&nbsp;&nbsp;&bull; {esc(self.check_label(c))}" for c in listed]
            if len(remaining) > len(listed):
                more = len(remaining) - len(listed)
                lines.append(f"&nbsp;&nbsp;&hellip; and {more} more")
            parts.append(
                block([f"<b>Still to run ({len(remaining)}):</b>"], small=False, gap=10)
            )
            parts.append(block(lines))
        else:
            parts.append(
                block(["No checks left to run after this one."], color=muted, gap=10)
            )
        overridden = list(self._overridden_checks)
        if overridden:
            parts.append(
                block(
                    [
                        "Already overridden this run: "
                        + ", ".join(esc(self.check_label(c)) for c in overridden)
                    ],
                    color=muted,
                    gap=6,
                )
            )
        parts.append(
            block(
                [
                    "<b>Override All</b> &mdash; export, and let every check from "
                    "here on through (failures are still logged).",
                    "<b>Override</b> &mdash; continue past this check; the next one "
                    "that fails asks again.",
                    "<b>Cancel</b> &mdash; stop before anything is written.",
                ],
                color=muted,
                gap=12,
            )
        )
        return "".join(parts)

    @ptk.Deprecation.symbol(
        "SceneExporterBase.decide_check_failure",
        remove_in="0.14.0",
        since="2026-10-04",
        reason="A failed check is decided where it fails, one check at a time.",
    )
    def confirm_check_override(self) -> bool:
        """Ask whether to export despite the failed checks of the last run.

        Retired: a run asks :meth:`decide_check_failure` the moment each check
        fails. Kept for callers that still ask once, after the run.
        """
        failed = list(getattr(self.task_manager, "_last_failed_checks", ()) or ())
        listed = ", ".join(failed[:10]) + (" \u2026" if len(failed) > 10 else "")
        headline = (
            f"{len(failed)} validation check(s) failed: {listed}."
            if failed
            else "A validation check failed."
        )
        return self.confirm(
            f"{headline}\n\n"
            "The export tasks have already run, so overriding now finishes THIS "
            "run instead of re-running the whole pipeline over an already-"
            "mutated scene.\n\n"
            "Override the checks and export anyway?"
        )

    def _warn_stopped_before_write(self, verdict: str) -> None:
        """Warn that the run stopped before its write, naming what it left behind.

        Every staged edit unwinds in :meth:`perform_export`'s ``finally``, on this
        exit as on any other. What stays is what a finished export keeps too --
        the repairs, and a write-back mode's in-place edits -- which the tasks
        record as they make them (``TaskFactory.kept_edits``). Named only when a
        task left one: a fixed list here said key snapping and tying remained
        after runs that had restored every key.
        """
        kept = self.task_manager.kept_edits
        if not kept:
            self.logger.warning(verdict)
            return
        self.logger.warning(
            f"{verdict} Kept in the scene, as a finished export keeps them: "
            f"{', '.join(kept)}. {self.KEPT_EDITS_ADVICE}"
        )

    def _run_task_pipeline(self, tasks: Dict[str, Any]) -> bool:
        """Run *tasks* through the task manager; True when the write may follow.

        Each failed check is put to :meth:`decide_check_failure` the moment it
        fails (``TaskFactory.failed_check_handler``), so an override carries
        the SAME run on -- every task and check below the failure still runs,
        and a later failure asks again unless the answer was Override All.
        What was overridden is recorded for the success banner
        (``_overridden_checks``). A failure that stands stops the run: the
        verdict is logged with the edits the tasks kept, and the staged ones
        unwind in ``perform_export``'s ``finally``. A raising task does the
        same and re-raises.
        """
        tm = self.task_manager

        def decide(check: str, messages: List[str], remaining: List[str]) -> str:
            answer = self.decide_check_failure(check, messages, remaining)
            if answer in (
                ptk.TaskFactory.CHECK_OVERRIDE,
                ptk.TaskFactory.CHECK_OVERRIDE_ALL,
            ):
                self._overridden_checks.append(check)
            return answer

        tm.failed_check_handler = decide
        try:
            proceed = tm.run_tasks(tasks)
        except Exception as e:
            # A raising task stops the run before its write, as a failed check
            # does: the staged edits unwind in perform_export's finally, and
            # what the tasks kept is named before the error goes on.
            self._warn_stopped_before_write(f"Export stopped by an error: {e}.")
            raise
        finally:
            tm.failed_check_handler = None
        # The runner's list is the record: an Override All lets later failures
        # through without asking, so only it names every one.
        self._overridden_checks = list(
            getattr(tm, "_last_overridden_checks", None) or self._overridden_checks
        )
        if not proceed:
            self._warn_stopped_before_write("Export blocked by failed checks.")
            return False
        if self._overridden_checks:
            self.logger.warning(
                "Checks overridden — writing the file despite "
                f"{len(self._overridden_checks)} failed check(s): "
                f"{', '.join(self._overridden_checks)}."
            )
        return True

    # ------------------------------------------------------------------
    # Progress -- one (current, total, message) stream for the whole run
    # ------------------------------------------------------------------

    def _progress_begin(
        self, callback: Optional[Callable], tasks: Dict[str, Any], phases: int
    ) -> None:
        """Arm the run's progress stream (see ``perform_export``).

        ``current`` counts finished steps: every pipeline entry that will
        dispatch is one (the task manager reports them through its
        ``progress_callback``), and each of the *phases* after the pipeline
        -- the write, a GLB conversion, the sidecar, ... -- is one more.
        """
        self._progress_callback = callback
        self._progress_total = self.task_manager._dispatchable_count(tasks) + phases
        self._progress_current = 0
        self._progress_base = 0
        self._progress_open = False
        self._progress_cancellable = True
        self._progress_cancel_ignored = False
        self._export_cancelled = False
        self.task_manager.progress_callback = self._on_pipeline_progress

    def _progress_end(self) -> None:
        """Disarm the stream; a later run of the task manager reports nothing."""
        self.task_manager.progress_callback = None
        self._progress_callback = None

    def _emit_progress(self, message: Optional[str]) -> bool:
        """Report the current position; False when the caller asked to stop.

        A ``False`` from the callback is honoured only while nothing has been
        written. Once the write starts the deliverable is finished regardless
        -- a GLB abandoned between its conversion and its texture pass is a
        file that looks complete and is not -- and the request is reported
        once instead. A callback that raises is a feedback bug: logged, never
        allowed to fail the export.
        """
        callback = self._progress_callback
        if callback is None:
            return True
        try:
            keep_going = callback(self._progress_current, self._progress_total, message)
        except ptk.OperationCancelled:
            raise
        except Exception as e:  # noqa: BLE001 -- feedback never fails an export
            self.logger.debug(f"Progress callback failed: {e}")
            return True
        if keep_going is not False:
            return True
        if self._progress_cancellable:
            return False
        if not self._progress_cancel_ignored:
            self._progress_cancel_ignored = True
            self.logger.warning(
                "Cancel requested after the write began — finishing the "
                "deliverable rather than leaving it half-written."
            )
        return True

    def _on_pipeline_progress(self, current, total, message) -> bool:
        """The task manager's hook: its entry index rides on the run's base.

        ``(None, None, text)`` is a text-only tick and leaves the count alone.
        """
        if current is not None:
            self._progress_current = self._progress_base + int(current)
            self._progress_open = False
        return self._emit_progress(message)

    def _progress_step(self, message: str) -> None:
        """Start a post-pipeline phase; the one before it is thereby done."""
        if self._progress_open:
            self._progress_current += 1
        self._progress_open = True
        if not self._emit_progress(message):
            raise ptk.OperationCancelled(f"cancelled before {message}")

    def _progress_note(self, message: str) -> None:
        """Narrate inside a phase without moving the count."""
        if not self._emit_progress(message):
            raise ptk.OperationCancelled(f"cancelled before {message}")

    def _progress_finish(self, message: str) -> None:
        """The last tick, snapped to the total (skipped checks leave a gap)."""
        self._progress_current = self._progress_total
        self._progress_open = False
        self._emit_progress(message)

    # ------------------------------------------------------------------
    # The export button's contract (the panel's b000, written once)
    # ------------------------------------------------------------------

    #: The Output Format row (``cmb004``): label -> ``output_format`` token.
    #: APPEND-ONLY -- the combo (and every saved preset) persists by index.
    OUTPUT_FORMATS = ptk.ExportProfile.OUTPUT_FORMATS

    def _definition_tables(self):
        """``(tasks, checks)`` -- the panel's two definition tables, built once.

        The two properties assemble every row's tooltip on each access, and a
        button press consults them more than once.
        """
        tables = getattr(self, "_definition_tables_cache", None)
        if tables is None:
            tm = self.task_manager
            tables = (tm.task_definitions, tm.check_definitions)
            self._definition_tables_cache = tables
        return tables

    def run_config_from_values(
        self,
        values: Dict[str, Any],
        override_checks: bool = False,
        ignore_groups_case_sensitive: bool = False,
    ) -> Dict[str, Any]:
        """Widget values -> the inputs :meth:`perform_export` takes.

        The export button's contract, through
        :meth:`pythontk.ExportProfile.run_config` (the one copy both DCC panels
        read their widgets with); this adds the settings row that is not a task
        definition: ``output_format`` (``cmb004``) into the tasks.

        Returns:
            ``{"tasks", "export_mode", "export_visible"}``.
        """
        tasks_def, checks_def = self._definition_tables()
        config = ptk.ExportProfile.run_config(
            values,
            tasks_def,
            checks_def,
            override_checks=override_checks,
            ignore_groups_case_sensitive=ignore_groups_case_sensitive,
        )
        output_format = values.get("cmb004")
        if output_format:
            config["tasks"]["output_format"] = output_format
        return config

    def _note_retired_preset_rows(
        self, name: str, stored: Optional[Dict[str, Any]]
    ) -> None:
        """Log what preset *name* asked of a row that has since retired.

        Both panels call this when a preset is picked: the load skips a key no
        widget takes, so a preset saved before a row retired would otherwise
        stop doing what it asked without a word
        (:meth:`pythontk.ExportProfile._retired_row_notes`).

        Parameters:
            name: The preset's name, as its combo shows it.
            stored: Its ``{objectName: value}`` (``PresetManager.read``);
                ``None`` when it cannot be read.
        """
        if not name or not stored:
            return
        tasks_def, checks_def = self._definition_tables()
        for level, message in ptk.ExportProfile._retired_row_notes(
            stored, tasks_def, checks_def
        ):
            getattr(self.logger, level)(f"Preset '{name}': {message}")

    @staticmethod
    def _saved_widget_values(store: Any, names: Iterable[str]) -> Dict[str, Any]:
        """``{objectName: value}`` a panel's settings store holds for *names*.

        The switchboard saves each widget under ``<objectName>/<signal>``
        (uitk's ``StateManager``), so a row is found by its key's first
        segment, whatever signal saved it.

        Parameters:
            store: The panel's settings store (``keys()`` / ``value(key)``).
            names: The objectNames to read.
        """
        wanted = set(names)
        found: Dict[str, Any] = {}
        for key in store.keys():
            name = str(key).split("/", 1)[0]
            if name in wanted and name not in found:
                found[name] = store.value(key)
        return found

    def _note_retired_panel_state(self, store: Any) -> None:
        """Say once what the panel's saved state asked of a row since retired.

        The state a panel saves outlives a row it no longer builds: a restore
        skips a key no widget takes, so a user whose saved state ran a check
        through its own row -- beside its fix row off -- would lose the check
        with nothing said (:meth:`_note_retired_preset_rows` says the same for
        a preset). Both panels call this when they open, with their window's
        settings store; the notes are
        :meth:`pythontk.ExportProfile._retired_row_notes`, read off the
        retired rows' saved values and the live fix rows' (saved, or their
        definition's default). The retired rows' keys are then removed
        through the same store, so the notes are said once.

        Parameters:
            store: The panel's settings store -- ``keys()``, ``value(key)``,
                ``remove(key)`` and optionally ``sync()``, as uitk's
                ``SettingsManager`` (the window's ``ui.settings``) has.
        """
        try:
            profile = ptk.ExportProfile
            tasks_def, checks_def = self._definition_tables()
            retired = profile._retired_row_keys(tasks_def, checks_def)
            fixes = {
                profile.widget_key(task, tasks_def[task]): tasks_def[task]
                for task in profile.VERIFYING_CHECKS
                if task in tasks_def
            }
            saved = self._saved_widget_values(store, (*retired, *fixes))
            orphaned = [key for key in retired if key in saved]
            if not orphaned:
                return
            # A fix row with no saved value opens at its definition's default.
            stored = {
                key: spec["setChecked"]
                for key, spec in fixes.items()
                if "setChecked" in spec
            }
            stored.update(saved)
            for level, message in profile._retired_row_notes(
                stored, tasks_def, checks_def
            ):
                getattr(self.logger, level)(f"The panel's saved state: {message}")
            for key in orphaned:
                store.remove(key)
            sync = getattr(store, "sync", None)
            if callable(sync):
                sync()
        except Exception as e:  # noqa: BLE001 -- a notice never blocks the panel
            self.logger.debug(f"Retired-row state not read: {e}")

    #: The objectName the retired RegEx field saved its value under.
    _RETIRED_REGEX_ROW = "txt002"

    def _migrate_legacy_regex(
        self, store: Any, name_row: str, save: Callable[[str], Any]
    ) -> Optional[str]:
        """Fold the retired RegEx field's saved value into the Output Filename, once.

        The field is retired: the regex is an inline modifier on the name token
        now (``{scene:PATTERN->REPLACEMENT}``). Its saved value outlives its
        widget, so the user's naming rule would quietly stop applying. Both
        panels call this when they open -- BEFORE the switchboard restores their
        widgets: the Output Filename's saved pattern and the RegEx field's saved
        value (:meth:`_saved_widget_values`) fold through the SAME call the
        export makes (:meth:`pythontk.ExportProfile.fold_legacy_naming`), and
        *save* stores the result as the Output Filename's value, so the restore
        that follows shows it. (Set on the widget instead, the fold was
        overwritten by that restore.) The retired key goes only once the fold
        is stored -- so this runs once and never loses the rule; a blank one
        just goes.

        Parameters:
            store: The panel's settings store -- ``keys()``, ``value(key)``,
                ``remove(key)`` and optionally ``sync()``.
            name_row: The Output Filename's objectName.
            save: Stores a pattern as the Output Filename's value -- the
                panel's ``ui.state.save(widget, value)``, which keys it the way
                the restore reads it.

        Returns:
            The folded pattern; ``None`` when nothing was folded.
        """
        retired = self._RETIRED_REGEX_ROW
        try:
            saved = self._saved_widget_values(store, (name_row, retired))
            if retired not in saved:
                return None
            regex = str(saved[retired] or "")
            folded = None
            if regex.strip():
                folded = ptk.ExportProfile.fold_legacy_naming(
                    str(saved.get(name_row) or ""), name_regex=regex
                )
                save(folded)
                stored = self._saved_widget_values(store, (name_row,)).get(name_row)
                if stored != folded:
                    self.logger.debug(
                        "The retired RegEx field's fold was not stored; it is "
                        "kept for the next open."
                    )
                    return None
            store.remove(retired)
            sync = getattr(store, "sync", None)
            if callable(sync):
                sync()
            if folded:
                self.logger.info(
                    "The Output Filename's RegEx field is retired: its pattern "
                    f"is now part of the name itself &mdash; {folded!r}."
                )
            return folded
        except Exception as e:  # noqa: BLE001 -- a migration never blocks the panel
            self.logger.debug(f"Retired RegEx field not migrated: {e}")
            return None

    # ------------------------------------------------------------------
    # Output naming and the export folder
    # ------------------------------------------------------------------

    #: Stamped by ``perform_export``: the Output Filename pattern, any retired
    #: naming input already folded in. A class-level default so a name can be
    #: resolved before the first run -- the panel's live tooltip preview
    #: resolves one on every hover.
    output_name: Optional[str] = None

    #: The Output Filename's wildcard (``*``), version counter (``{n}``) and the
    #: files each output format ships: the contract ``ptk.ExportProfile`` owns
    #: for both DCC panels, re-exposed so the panel reads them off ``self``.
    NAME_WILDCARD = ptk.ExportProfile.NAME_WILDCARD
    VERSION_TOKEN = ptk.ExportProfile.VERSION_TOKEN
    OUTPUT_EXTENSIONS = ptk.ExportProfile.OUTPUT_EXTENSIONS

    def _resolve_export_dir(self, export_dir: Optional[str]) -> str:
        """The folder an export writes to: *export_dir* expanded, else the saved
        scene's own folder; ``""`` when there is neither (an unsaved scene with
        no directory set, which :meth:`perform_export` refuses)."""
        if export_dir:
            return os.path.abspath(os.path.expandvars(export_dir))
        scene_path = self._saved_scene_path()
        return os.path.dirname(scene_path) if scene_path else ""

    # ------------------------------------------------------------------
    # The run's log
    # ------------------------------------------------------------------

    #: Where every scene's run log lives, under the per-user cache
    #: (``UserConfig.cache_dir``): one ``RunLog`` file per scene, shared by
    #: both DCCs -- never beside the deliverable, where one per export piled
    #: up in every delivery folder (2026-10-13).
    LOG_FOLDER: Tuple[str, ...] = ("logs", "scene_exporter")

    #: The first line of a log an earlier version wrote beside a deliverable:
    #: a record in ``RunLog.FORMAT``, or a box (a quiet level logs only the
    #: closing banner). :meth:`_remove_legacy_logs` deletes nothing else.
    _LEGACY_LOG_START = re.compile(
        r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} - [A-Z]+ - |[\u2500-\u257f])"
    )

    def log_file_path(self) -> str:
        """This scene's run log: ``<scene>_<hash>.log`` in :attr:`LOG_FOLDER`.

        Keyed by the saved scene's path (:meth:`RunLog.file_name
        <pythontk.RunLog.file_name>`), so every export of a scene appends to
        one file whatever its Output Filename resolves to, two same-named
        scenes keep apart, and an unsaved scene logs to ``untitled.log``.
        """
        folder = ptk.UserConfig.cache_dir(*self.LOG_FOLDER)
        return os.path.join(str(folder), RunLog.file_name(self._saved_scene_path()))

    @ptk.Deprecation.symbol(
        "SceneExporterBase.log_file_path",
        remove_in="0.17.0",
        since="2026-10-14",
        reason="The run log is keyed by the scene, not the export file.",
    )
    def generate_log_file_path(self, export_path: Optional[str] = None) -> str:
        """Retired: the run log is the scene's (:meth:`log_file_path`), so
        *export_path* no longer names it."""
        return self.log_file_path()

    def setup_file_logging(self, log_file_path: str, header: Optional[str] = None):
        """Tee :attr:`logger` into *log_file_path* for the run in flight.

        On :attr:`logger` -- the one the exporter and its task manager both
        log through. It sat on a bare logger named after the class, which
        nothing logs to, so every run's ``.log`` was empty (2026-10-09).
        UTF-8 and plain text (:meth:`RunLog.attach <pythontk.RunLog.attach>`):
        in the locale's code page the run's closing banner -- box glyphs, a
        check mark -- raised on write, so a clean run at the ERROR level left
        the file empty (2026-10-10); without the strip its markup and links
        landed as tags. Appends, the run opening on *header*, and records
        every level down to DEBUG whatever the panel shows (``capture``): the
        panel tells the run's story, the file keeps the detail to diagnose it.
        """
        self.close_file_handlers()  # a run that never closed its own
        self.file_handler = RunLog.attach(
            self.logger, log_file_path, header=header, capture=logging.DEBUG
        )
        self.logger.debug(f"File logging setup complete. Log file: {log_file_path}")

    def _glb_created(self, path: str) -> None:
        """Say the run's GLB landed at *path*, and keep it for the run's last
        line: :meth:`_close_file_logging` links it to the WebXR preview."""
        self._shipped_glb = path
        self.logger.success(f"GLB created: {path}")

    def _close_file_logging(self, succeeded: bool) -> None:
        """Close the run's log on how the run ended, and link it in the panel.

        The footer reaches the file at every level, so a run that failed at a
        quiet one still says so; the links go to the handlers that remain
        (the panel, the console) once the file's is gone: the log, then one
        closing line of what to do with the result -- the folder a successful
        run delivered into (``action://open``) and the GLB the run shipped
        (:meth:`_glb_created`) as an ``action://preview`` link the host's
        dispatcher opens in the WebXR preview. The host's ``perform_export``
        calls it on every exit.
        """
        glb, self._shipped_glb = getattr(self, "_shipped_glb", None), None
        handler = getattr(self, "file_handler", None)
        if handler is not None:
            if getattr(self, "_export_cancelled", False):
                outcome = "Export cancelled"
            else:
                outcome = "Export succeeded" if succeeded else "Export failed"
            self.close_file_handlers(footer=outcome)
            path = handler.baseFilename
            link = self.logger.log_link(path, "open", path=path)
            self.logger.log_raw(f"Log: {link}")
        links = []
        folder = os.path.dirname(getattr(self, "export_path", None) or "")
        if succeeded and folder:
            links.append(
                self.logger.log_link("Open containing folder", "open", path=folder)
            )
        if glb:
            name = os.path.basename(glb)
            links.append(
                self.logger.log_link(f"Open {name} in WebXR", "preview", path=glb)
            )
        if links:
            self.logger.log_raw(" · ".join(links))

    def close_file_handlers(self, footer: Optional[str] = None):
        """Close and remove the handler :meth:`setup_file_logging` added; a
        caller's own handlers on the same logger are left alone. *footer*,
        when given, closes the run in the file (:meth:`RunLog.detach
        <pythontk.RunLog.detach>`)."""
        handler = getattr(self, "file_handler", None)
        if handler is None:
            return
        self.file_handler = None
        RunLog.detach(self.logger, handler, footer=footer)
        self.logger.debug("File handler closed and removed.")

    def _remove_legacy_logs(self) -> List[str]:
        """Delete the per-export logs earlier versions left in the export folder.

        Each export used to write ``<export name>.log`` (``.<name>.log`` when
        hidden off Windows) beside its deliverable, and nothing removed one.
        A log goes only when it is the exporter's own: a deliverable of its
        name sits beside it, and it is empty (the 2026-10-09 wrong-logger
        runs) or opens on the exporter's record or banner
        (:attr:`_LEGACY_LOG_START`). Best-effort, per file.

        Returns:
            list: The paths removed.
        """
        folder = getattr(self, "export_dir", None)
        if not folder or not os.path.isdir(folder):
            return []
        extensions = {
            ext for exts in ptk.ExportProfile.OUTPUT_EXTENSIONS.values() for ext in exts
        }
        names = os.listdir(folder)
        shipped = {
            os.path.splitext(n)[0].casefold()
            for n in names
            if os.path.splitext(n)[1].lower() in extensions
        }
        removed = []
        for name in names:
            stem, ext = os.path.splitext(name)
            if ext.lower() != RunLog.EXTENSION:
                continue
            if stem.lstrip(".").casefold() not in shipped:
                continue
            path = os.path.join(folder, name)
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    first = next((ln for ln in fh if ln.strip()), "")
                if first and not self._LEGACY_LOG_START.match(first):
                    continue
                os.remove(path)
                removed.append(path)
            except OSError:
                continue
        if removed:
            self.logger.info(
                f"Removed {len(removed)} old export log(s) from {folder}: the "
                "run log is one file per scene now."
            )
        return removed
