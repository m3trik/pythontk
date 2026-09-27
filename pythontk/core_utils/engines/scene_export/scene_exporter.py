# !/usr/bin/python
# coding=utf-8
"""The Scene Exporter's orchestration shell, written once for both DCCs.

mayatk's and blendertk's ``SceneExporter`` subclass :class:`SceneExporterBase`
and keep only what reaches their scene: the export write itself
(``perform_export``), the object scope, the FBX/USD options and presets, and
the naming tokens a scene spells.  What they share is here: the logging
setup, the run's ``(current, total, message)`` progress stream, the
check-override consent and the resume of the tasks a failed check stopped,
the export button's contract (:meth:`SceneExporterBase.run_config_from_values`
over :class:`~pythontk.ExportProfile`), and the export folder / log file
plumbing.

Hooks (override in the DCC subclass; defaults describe no scene):

==============================  =============================================
``TASK_MANAGER_CLASS``          the host's ``TaskManager`` (a
                                :class:`~pythontk.TaskFactory`), built with
                                the exporter's logger
``_saved_scene_path()``         the open scene's saved path, ``""`` unsaved
``KEPT_EDITS_ADVICE``           what a run that stopped before its write
                                tells the user about the edits it kept
``confirm(question)``           consent for an export-time side effect (the
                                panel overrides it with a dialog)
==============================  =============================================
"""

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
        #: point (see confirm_check_override). Re-stamped by every
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

    def confirm_check_override(self) -> bool:
        """Ask, at the failure point, whether to export despite failed checks.

        The tasks have already run and the scene is still staged, so this is
        the ONE moment at which overriding costs nothing. Arming the panel's
        Override Checks toggle *after* a failed run instead means a second
        export from scratch: every task re-runs (re-bake, re-optimize the
        textures, re-rewrite the paths) on a scene the first run already
        mutated. Answering yes here continues the SAME run straight to the
        write.

        Consent only, never an automatic pass: it routes through
        :meth:`confirm`, whose default answers no when nobody is there to ask
        (a batch run still aborts on a failed check).
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

    def _resume_skipped_tasks(self, tasks: Dict[str, Any]) -> None:
        """Run the tasks the failed check aborted, so an override still ships a
        fully processed file.

        The runner stops dispatching tasks at the first failed check -- every
        one below it in the schedule is work an aborted write would throw
        away. Overriding turns that write back on, so those tasks are no
        longer wasted and must run before it: without this an overridden
        export silently shipped a file that skipped, say, the texture
        conversion the user asked for.

        Only the skipped names are re-dispatched; the tasks above the failed
        check already ran, and re-running them would repeat their mutation.
        The first pass's staged state is still in effect (deferred restores
        unwind once, from perform_export's ``finally``), and the run's modes
        are NOT re-derived from this subset -- ``run_tasks`` reads them off
        the full dict, so the resume goes through the dispatcher directly.
        """
        tm = self.task_manager
        skipped = [
            n for n in (getattr(tm, "_last_skipped_tasks", ()) or ()) if n in tasks
        ]
        if not skipped:
            return
        self.logger.info(
            f"Resuming {len(skipped)} task(s) the failed check had stopped: "
            f"{', '.join(skipped)}."
        )
        # The second pass re-stamps the run counters the success banner reads.
        # The first pass already counted every REQUESTED task, so its numbers
        # are the ones that describe the run; keep them.
        counts = (
            getattr(tm, "_last_task_count", 0),
            getattr(tm, "_last_check_count", 0),
        )
        # ...and the checks the abort skipped, which the second pass (tasks
        # only) would clear: the banner must not count them as passed.
        skipped_checks = list(getattr(tm, "_last_skipped_checks", ()) or ())
        # The first pass closed its progress stream with every entry done,
        # these included; rewind so the resumed entries advance to, never
        # past, that mark.
        self._progress_base = max(0, self._progress_current - len(skipped))
        try:
            tm._execute_tasks_and_checks({name: tasks[name] for name in skipped}, {})
        finally:
            tm._last_task_count, tm._last_check_count = counts
            tm._last_skipped_checks = skipped_checks

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
