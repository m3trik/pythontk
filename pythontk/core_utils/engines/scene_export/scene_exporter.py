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
:class:`~pythontk.ExportProfile`), and the export folder / log file plumbing.

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
from typing import Any, Callable, Dict, List, Optional

import pythontk as ptk
from pythontk.core_utils.logging_mixin import LoggingMixin


class SceneExporterBase(LoggingMixin):
    """The DCC-free half of a Scene Exporter: progress, consent, run config.

    Subclass per host and set :attr:`TASK_MANAGER_CLASS`; see the module
    docstring for the hooks.  The ``export_dir`` / ``hide_log_file`` /
    ``export_path`` attributes the log-file helpers read are stamped by the
    host's ``perform_export``.
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
        """Setup file logging."""
        log_file_path = self.generate_log_file_path(self.export_path)
        self.logger.info(f"Generating log file path: {log_file_path}")
        self.setup_file_logging(log_file_path)

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
    # The run's log file
    # ------------------------------------------------------------------

    def generate_log_file_path(self, export_path: str) -> str:
        """Generate the log file path based on the export path.

        ``hide_log_file`` off Windows means a dot-prefixed name, which is what
        hidden is there; on Windows :meth:`setup_file_logging` sets the
        hidden attribute instead.
        """
        base_name = os.path.splitext(os.path.basename(export_path))[0]
        prefix = "." if self.hide_log_file and os.name != "nt" else ""
        return os.path.join(self.export_dir, f"{prefix}{base_name}.log")

    def setup_file_logging(self, log_file_path: str):
        """Setup file logging to log actions during export."""
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.file_handler = file_handler
        root_logger = logging.getLogger(self.__class__.__name__)
        root_logger.addHandler(self.file_handler)
        self.logger.debug(f"File logging setup complete. Log file: {log_file_path}")

        if self.hide_log_file and os.name == "nt":
            import ctypes

            ctypes.windll.kernel32.SetFileAttributesW(log_file_path, 2)

    def close_file_handlers(self):
        """Close and remove file handlers after logging is complete."""
        root_logger = logging.getLogger(self.__class__.__name__)
        handlers = root_logger.handlers[:]
        for handler in handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                root_logger.removeHandler(handler)
                self.logger.debug("File handler closed and removed.")
